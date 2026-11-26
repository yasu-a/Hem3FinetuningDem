# Adapted from THeatA for the bundled workbench.
"""planner v2: per-memory-zone state machine + JIT staging (2026-09-08 設計).

v1 (MovementPlanner) との差分は 3 点:

1. 納品キュー: 残 DAG を min-id 決定順 (Kahn) で射影した ion 需要列を
   測定窓ごとに再計算する。各 memory zone はこの需要列の射影を納品順として持つ。
2. 先行整列: MEASURE emit 直後 (LIZ が 1000 μs 占有される窓) に、各 memory
   zone で「次に納品する horizon 個」を LIZ 側端へ、物理展開済み swap
   (split/swap/merge、フォノン会計込み) で整列する。整列 op は memory zone と
   その temp しか使わないため、ASAP / イベント両スケジューラが測定と重ねる。
   v1 ではこの整列が納品時 (_position_for_shuttle) に LIZ 待ちの直列部で
   発生していた。
3. JIT staging (prefetch_depth=1): 整列後、次需要イオンを corridor temp へ
   split だけしておく。shuttle+merge (納品) は需要時に行う (JIT)。temp は
   即時手渡し / 短期バッファのみ — 他の op がその temp を必要とした場合は
   merge で memory の LIZ 側端へ flush して返す (正しさ優先、コストは
   merge 1 回 + 後の再 split)。

正しさの論法は v1 と同一: すべての状態変化は QCCDState を通じて emission
順に適用され、flat_ops は per-zone 順序保存でスケジュールされる。検証は
既存の validate_order_against / build_ion_events (busy 非重複) / Validator を
そのまま使う。
"""

import heapq
from typing import Dict, List, Optional, Tuple

from .ast_def import Operation, SplitOp, MergeOp, ShuttleOp, Location
from .qccd_compiler import MovementPlanner, QCCDState
from .dag_builder import DAG


class StateMachinePlanner(MovementPlanner):
    """per-zone 納品キュー + JIT staging の MovementPlanner 置き換え。

    CompilerConfig.planner = "v2" で QCCDCompiler から選択される。
    パラメータ: prefetch_depth (0 = 厳密 JIT / 1 = 次需要 1 個を temp へ
    先行切り出し)、v2_sort_horizon (窓ごとに整列する需要イオン数の上限)。
    """

    def __init__(self, architecture, config=None, time_params=None):
        super().__init__(architecture, config, time_params=time_params)
        # staging 台帳: ion_id -> (temp_zone_id, home_memory_zone_id)
        self._staged: Dict[int, Tuple[int, int]] = {}
        # delivery トリガー用: _generate_movements 中の DAG 参照と
        # 需要列キャッシュ (key = 実行済みノード数)
        self._current_dag: Optional[DAG] = None
        self._demand_cache: Optional[Tuple[int, Dict[int, int]]] = None

    def plan(self, dag, initial_placement=None, state=None):
        ops = super().plan(dag, initial_placement, state)
        # 終端で staging を清算する。temp に残したままラウンドを跨ぐと、
        # 次ラウンドの planner (fresh instance) は temp 発の搬送経路を
        # 持たないため、必ず memory へ返してから終わる。
        for temp in list({info[0] for info in self._staged.values()}):
            ops.extend(self._flush_staged_in_temp(temp, self._last_state))
        return ops

    # ------------------------------------------
    # 納品キュー (需要列)
    # ------------------------------------------

    def _demand_order(self, dag: DAG) -> List[int]:
        """残ノードを min-id 決定順 (Kahn, id 優先) で走査し、
        ターゲットイオンの初出順 = LIZ 需要列を返す。"""
        executed = dag.get_executed()
        indeg = {}
        heap = []
        for nid, node in dag.nodes.items():
            if nid in executed:
                continue
            d = sum(1 for p in node.predecessors if p not in executed)
            indeg[nid] = d
            if d == 0:
                heapq.heappush(heap, nid)
        order: List[int] = []
        seen = set()
        while heap:
            nid = heapq.heappop(heap)
            node = dag.nodes[nid]
            for t in getattr(node.operation, 'targets', None) or []:
                if t not in seen:
                    seen.add(t)
                    order.append(t)
            for s in node.successors:
                if s in indeg:
                    indeg[s] -= 1
                    if indeg[s] == 0:
                        heapq.heappush(heap, s)
        return order

    def _nearest_liz(self, zone_id: int) -> int:
        return min((z.id for z in self.architecture.get_liz_zones()),
                   key=lambda lz: abs(lz - zone_id))

    def _demand_rank(self, dag: DAG) -> Dict[int, int]:
        """需要列の rank マップ (実行済みノード数でキャッシュ)。"""
        key = len(dag.get_executed())
        if self._demand_cache is not None and self._demand_cache[0] == key:
            return self._demand_cache[1]
        rank = {ion: i for i, ion in enumerate(self._demand_order(dag))}
        self._demand_cache = (key, rank)
        return rank

    # ------------------------------------------
    # 測定窓フック (v1 の no-op を置き換え)
    # ------------------------------------------

    def _measure_window_ops(self, dag: DAG, state: QCCDState) -> List[Operation]:
        ops: List[Operation] = []
        # 前窓の未納品 staging を清算 (temp は短期バッファのみ)
        for temp in list({info[0] for info in self._staged.values()}):
            ops.extend(self._flush_staged_in_temp(temp, state))
        rank = self._demand_rank(dag)
        if self.config.v2_sort_trigger in ("measure", "both"):
            for mz in self.architecture.get_memory_zones():
                ops.extend(self._sort_zone(mz.id, rank, state))
        if self.config.prefetch_depth >= 1:
            for mz in self.architecture.get_memory_zones():
                ops.extend(self._stage_next(mz.id, rank, state))
        return ops

    # ------------------------------------------
    # delivery トリガー (v3 案, 2026-09-10):
    # 納品で端が空いた直後に、その zone の次需要イオンを真端まで整列する。
    # edge-1 止めが不要になるため v1 と同一 swap 列の前倒し = 加熱中立が狙い。
    # ------------------------------------------

    def _generate_movements(self, dag: DAG, front_layer, state):
        self._current_dag = dag
        try:
            return super()._generate_movements(dag, front_layer, state)
        finally:
            self._current_dag = None

    def _post_delivery_sort(self, mem_zone: int,
                            state: QCCDState) -> List[Operation]:
        if (self.config.v2_sort_trigger not in ("delivery", "both")
                or self._current_dag is None
                or not self.architecture.is_memory(mem_zone)):
            return []
        rank = self._demand_rank(self._current_dag)
        return self._sort_zone(mem_zone, rank, state)

    def _sort_zone(self, zone_id: int, rank: Dict[int, int],
                   state: QCCDState) -> List[Operation]:
        """需要 rank の小さい順に horizon 個を LIZ 側端へ整列する
        (selection sort、隣接 swap は物理展開済み)。
        emit した op には origin="sort" を付ける (flush 由来は除く)。"""
        ops: List[Operation] = []
        contents = state.zone_contents.get(zone_id, [])
        if len(contents) <= 1:
            return ops
        liz = self._nearest_liz(zone_id)
        liz_right = liz > zone_id
        needed = sorted((i for i in contents if i in rank),
                        key=lambda i: rank[i])
        h = min(self.config.v2_sort_horizon, len(needed))
        for k in range(h):
            ion = needed[k]
            while True:
                arr = state.zone_contents[zone_id]
                target_pos = len(arr) - 1 - k if liz_right else k
                p = arr.index(ion)
                if p == target_pos:
                    break
                step = 1 if p < target_pos else -1
                nb = arr[p + step]
                left, right = (ion, nb) if step == 1 else (nb, ion)
                ops.extend(self._expand_swap_inline(left, right, state))
        for op in ops:
            if not hasattr(op, "origin"):
                op.origin = "sort"
        return ops

    def _stage_next(self, zone_id: int, rank: Dict[int, int],
                    state: QCCDState) -> List[Operation]:
        """次需要イオンが LIZ 側端に居れば corridor temp へ split しておく。"""
        contents = state.zone_contents.get(zone_id, [])
        if not contents:
            return []
        liz = self._nearest_liz(zone_id)
        liz_right = liz > zone_id
        edge_ion = contents[-1] if liz_right else contents[0]
        ranks_here = [rank[i] for i in contents if i in rank]
        if not ranks_here or edge_ion not in rank:
            return []
        if rank[edge_ion] != min(ranks_here):
            # 整列 horizon 外で端に別イオンが居る → 見送り (JIT に任せる)
            return []
        temp = self.architecture.get_adjacent_temp(
            zone_id, "right" if liz_right else "left")
        if temp is None or state.zone_contents.get(temp):
            return []
        ops: List[Operation] = []
        if len(contents) == 1 or not liz_right:
            # 単独 or 左端切り出し: ion 自身を境界に split
            ops.append(SplitOp(ion_id=edge_ion, dst_zone_id=temp,
                               src_zone_id=zone_id))
            state.split_chain(edge_ion, temp)
        else:
            # 右端切り出し: 左隣を境界に split (v1 _move_ion_one_hop と同型)
            split_ref = contents[-2]
            ops.append(SplitOp(ion_id=split_ref, dst_zone_id=temp,
                               src_zone_id=zone_id))
            state.split_chain(split_ref, temp)
        self._staged[edge_ion] = (temp, zone_id)
        for op in ops:
            op.origin = "stage"
        return ops

    # ------------------------------------------
    # staging の清算と納品
    # ------------------------------------------

    def _flush_staged_in_temp(self, temp_zone: int,
                              state: QCCDState) -> List[Operation]:
        """temp に居る staged イオンを home memory の LIZ 側端へ merge で返す。"""
        occupants = state.zone_contents.get(temp_zone, [])
        staged_here = [i for i in occupants
                       if i in self._staged and self._staged[i][0] == temp_zone]
        if not staged_here:
            return []
        mem = self._staged[staged_here[0]][1]
        left, right = min(temp_zone, mem), max(temp_zone, mem)
        op = MergeOp(zone_left=left, zone_right=right)
        op.origin = "flush"
        state.merge_chains(left, right)
        for i in staged_here:
            del self._staged[i]
        return [op]

    def _move_ion_to_zone(self, ion_id: int, dst_zone: int,
                          state: QCCDState) -> List[Operation]:
        src_zone = state.locate(ion_id)
        info = self._staged.get(ion_id)
        if info is not None:
            temp, home_mem = info
            if state.locate(ion_id) != temp:
                # 台帳と実態の不一致 (防御的): 台帳だけ破棄して通常経路
                del self._staged[ion_id]
            elif dst_zone in self.architecture.get_adjacent_zones(temp):
                # 納品: shuttle + merge のみ (split は staging 済み)
                ops: List[Operation] = []
                is_right = temp < dst_zone
                dst_contents = state.zone_contents.get(dst_zone, [])
                site = 0 if is_right else len(dst_contents)
                ops.append(ShuttleOp(
                    target=ion_id,
                    dst=Location(trap=dst_zone, site=site),
                    dst_chain_length=len(dst_contents),
                    src_zone=temp))
                state.apply_shuttle_phonon(ion_id, len(dst_contents))
                left, right = min(temp, dst_zone), max(temp, dst_zone)
                ops.append(MergeOp(zone_left=left, zone_right=right))
                state.merge_chains(left, right)
                del self._staged[ion_id]
                for op in ops:
                    op.origin = "deliver"
                if self.architecture.is_liz(dst_zone):
                    ops.extend(self._post_delivery_sort(home_mem, state))
                return ops
            else:
                # 納品口と逆方向への搬送要求 → memory へ返してから通常経路
                ops = self._flush_staged_in_temp(temp, state)
                return ops + super()._move_ion_to_zone(ion_id, dst_zone, state)
        ops = super()._move_ion_to_zone(ion_id, dst_zone, state)
        if (ops and self.architecture.is_liz(dst_zone)
                and self.architecture.is_memory(src_zone)):
            # 納品完了 → 端が空いた source memory を需要順に整列
            ops.extend(self._post_delivery_sort(src_zone, state))
        return ops

    # ------------------------------------------
    # temp 衝突ガード (staged は短期バッファのみ、という不変条件の実装)
    # ------------------------------------------

    def _move_ion_one_hop(self, ion_id: int, dst_zone: int,
                          state: QCCDState) -> List[Operation]:
        src_zone = state.locate(ion_id)
        ops: List[Operation] = []
        if src_zone != dst_zone:
            direction = "right" if src_zone < dst_zone else "left"
            temp = self.architecture.get_adjacent_temp(src_zone, direction)
            if temp is not None:
                ops.extend(self._flush_staged_in_temp(temp, state))
        return ops + super()._move_ion_one_hop(ion_id, dst_zone, state)

    def _expand_swap_inline(self, ion_a: int, ion_b: int,
                            state: QCCDState) -> List[Operation]:
        zone_id = state.locate(ion_a)
        ops: List[Operation] = []
        for d in ("left", "right"):
            temp = self.architecture.get_adjacent_temp(zone_id, d)
            if temp is not None:
                ops.extend(self._flush_staged_in_temp(temp, state))
        return ops + super()._expand_swap_inline(ion_a, ion_b, state)
