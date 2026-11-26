# Adapted from THeatA for the bundled workbench.
"""Per-LIZ stream compilation (Phase 1).

単一の global TimeStep 列を捨て、core (= LIZ + 隣接 Memory) ごとの
独立 stream を生成する。狙いは「各 LIZ が互いを待たずに動き、
1-LIZ より速く終わる」こと。

方式 (docs/per_liz_stream_design.md の Phase 1 を実装向けに簡略化):
  1. partition_spatial_row で stabilizer を LIZ に割当て (row banding)
  2. 各 LIZ の stabilizer 列を local block (共有 data を使わない) と
     boundary block (共有 data を使う) に分ける
  3. phase_alternation と同じ交互順 (偶数 idx LIZ: local→boundary、
     奇数 idx: boundary→local) で block を大域の計画順に並べ、
     **単一のグローバル状態上で逐次 plan** しながら ops を core ごとの
     stream に振り分ける
  4. 各 block の ops を「全 Temp zone が空になる静止点」で segment に分割し、
     計画順で資源 (zone / ion) が交差する cross-core の segment ペアに
     sync 辺 (happens-before) を張る。交差しない segment は完全並列。
     segment 粒度にするのは、shared M の swap 退避が隣 core の T に一時的に
     溢れる (T4/T6 問題) だけで block 全体が直列化されるのを防ぐため
  5. event simulation で per-core タイムライン・wall clock・sync 待ちを評価

正しさの根拠: plan は単一状態上で逐次に行われるため、「衝突しうる
アクセスの実行順 = 計画順」を sync 辺が保証する限り、どの実行
interleaving でもイオン位置は計画時と厳密に一致する (race-freedom)。
segment 境界は Temp 静止点なので、segment を跨いだ資源の暗黙保持はない。
検証は「per-segment スケジュールを計画順に連結した TimeStep 列」を
既存 ParallelValidator に通す (時間的に重なる segment は資源素であり、
状態更新が可換なので、この逐次化された検証が並列実行を代表する)。
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from .ast_def import (
    Program, Operation, GateOp, MeasureOp, ResetOp, SplitOp, MergeOp,
    TimeStep,
)
from .dag_builder import DAGBuilder
from .qccd_compiler import (
    QCCDArchitecture, QCCDState, MovementPlanner, Scheduler, CompilerConfig,
)
from .stabilizer_partitioner import StabilizerPartition
from .spatial_partitioner import (
    partition_spatial_row, compute_initial_placement_3tier,
    _identify_stabilizers,
)
from .validator import ParallelValidator
from .parallel_estimator import ParallelEstimator
from .sim_params import phonon_params_from


# 後方互換のための名前。正本は estimator.TimeParameters (sim_params 参照)。
DEFAULT_PHONON_PARAMS = phonon_params_from()


# ==========================================
# Core view: architecture facade restricted to one core
# ==========================================

class CoreView:
    """QCCDArchitecture の facade。1 つの core の zone 集合に視野を制限する。

    - get_liz_zones: 自 core の LIZ だけを返す (LIZ 選択・batch flush の誘導)
    - get_adjacent_zones / get_memory_zones: core 外 zone と blocked zone を
      候補から外す (eviction BFS が core 外へ流出しないように)
    - それ以外は base architecture へ委譲
    blocked は epoch ごとに差し替える (このブロックが必要としない shared M)。
    """

    def __init__(self, base: QCCDArchitecture, liz_id: int, allowed_zones: Set[int]):
        self._base = base
        self._liz_id = liz_id
        self._allowed = set(allowed_zones)
        self.blocked: Set[int] = set()

    def __getattr__(self, name):
        return getattr(self._base, name)

    def _ok(self, zone_id: int) -> bool:
        return zone_id in self._allowed and zone_id not in self.blocked

    def get_liz_zones(self):
        return [z for z in self._base.get_liz_zones() if z.id == self._liz_id]

    def get_memory_zones(self):
        return [z for z in self._base.get_memory_zones() if self._ok(z.id)]

    def get_adjacent_zones(self, zone_id: int) -> List[int]:
        return [z for z in self._base.get_adjacent_zones(zone_id) if self._ok(z)]


def _core_zones(arch: QCCDArchitecture, liz_id: int) -> Set[int]:
    """core = 左 M .. 右 M の連続 zone 区間 (端 core は余り T も含む)。"""
    memory_zones = [z.id for z in arch.zones if z.zone_type == "M"]
    liz_zones = [z.id for z in arch.zones if z.zone_type == "L"]
    i = liz_zones.index(liz_id)
    lo, hi = memory_zones[i], memory_zones[i + 1]
    zones = set(range(lo, hi + 1))
    if i == 0:
        zones.add(lo - 1)
    if i == len(liz_zones) - 1:
        zones.add(hi + 1)
    return zones


# ==========================================
# Result types
# ==========================================

@dataclass
class Segment:
    """stream の最小同期単位。Temp 静止点で区切った連続 ops 列。"""
    core_id: int                # LIZ zone id
    label: str                  # 'local' | 'boundary' (所属 block)
    ops: List[Operation] = field(default_factory=list)
    zones: Set[int] = field(default_factory=set)
    ions: Set[int] = field(default_factory=set)
    # event simulation で埋める
    timesteps: List[TimeStep] = field(default_factory=list)
    duration_us: float = 0.0
    start_us: float = 0.0
    end_us: float = 0.0


@dataclass
class SyncEdge:
    """cross-core の同期制約 (src segment ↔ dst segment)。

    kind='hb':    happens-before。実 ion の受け渡し / 非 Temp zone の共有。
                  計画順 (src → dst) を実行順として強制する。
    kind='mutex': 相互排他のみ。共有資源が Temp zone だけで ion 素な場合。
                  segment は Temp を空の状態で使い始めて空で返すため
                  状態効果が可換であり、時間が重ならなければ順序は自由。
    """
    src: int                    # segment index (計画順)
    dst: int
    shared_zones: Set[int] = field(default_factory=set)
    shared_ions: Set[int] = field(default_factory=set)
    kind: str = 'hb'
    wait_us: float = 0.0        # dst がこの辺のために実際に待った時間


@dataclass
class PerLIZResult:
    segments: List[Segment]                  # 計画順
    streams: Dict[int, List[int]]            # core_id -> segment indices
    syncs: List[SyncEdge]
    merged_flat: List[Operation]             # 計画順の連結 (canonical 併合ビュー)
    initial_placement: Dict[int, int]
    state: QCCDState                         # 最終状態 (フォノン取得用)
    arch: QCCDArchitecture
    # event simulation の結果
    wall_clock_us: float = 0.0
    per_core_busy_us: Dict[int, float] = field(default_factory=dict)
    per_core_end_us: Dict[int, float] = field(default_factory=dict)
    # 並べ替え後の入力回路 (DSL 形式、検証用ダンプ)
    input_circuit: str = ''

    def phonon_summary(self) -> Dict[str, float]:
        st = self.state
        return {
            'max_axial': max(st.phonon_axial.values()) if st.phonon_axial else 0.0,
            'max_transversal': (max(st.phonon_transversal.values())
                                if st.phonon_transversal else 0.0),
            'max_transversal2': (max(st.phonon_transversal2.values())
                                 if st.phonon_transversal2 else 0.0),
        }

    def epoch_summary(self) -> List[dict]:
        """(core, label) ごとの集計 (segment を束ねた block 単位のビュー)。"""
        out = []
        seen = {}
        for idx, s in enumerate(self.segments):
            key = (s.core_id, s.label)
            if key not in seen:
                seen[key] = {
                    'core_id': s.core_id, 'label': s.label, 'segments': 0,
                    'ops': 0, 'busy_us': 0.0,
                    'start_us': s.start_us, 'end_us': s.end_us,
                }
                out.append(seen[key])
            rec = seen[key]
            rec['segments'] += 1
            rec['ops'] += len(s.ops)
            rec['busy_us'] += s.duration_us
            rec['start_us'] = min(rec['start_us'], s.start_us)
            rec['end_us'] = max(rec['end_us'], s.end_us)
        return out


# ==========================================
# Helpers
# ==========================================

def _op_zones(op: Operation, pv: ParallelValidator) -> Set[int]:
    """op が占有する zone 集合 (gate は zone_id アノテーションを含める)。"""
    zones = set(pv._zones_used(op))
    if isinstance(op, (GateOp, MeasureOp, ResetOp)) and getattr(op, 'zone_id', -1) != -1:
        zones.add(op.zone_id)
    return zones


def _split_at_temp_quiescence(ops: List[Operation],
                              arch: QCCDArchitecture) -> List[List[Operation]]:
    """全 Temp zone が空になる点で ops を segment に分割する。

    Temp の占有は Split(dst=T) で始まり、その T を含む Merge で終わる
    (Shuttle は状態上 T 内の移動扱い)。静止点で切ることで、segment 間に
    「T にイオンが残ったまま」という暗黙の資源保持がなくなり、segment を
    独立な同期単位として扱える。
    """
    segments: List[List[Operation]] = []
    current: List[Operation] = []
    occupied: Set[int] = set()
    for op in ops:
        current.append(op)
        if isinstance(op, SplitOp) and arch.is_temp(op.dst_zone_id):
            occupied.add(op.dst_zone_id)
        elif isinstance(op, MergeOp):
            if arch.is_temp(op.zone_left):
                occupied.discard(op.zone_left)
            if arch.is_temp(op.zone_right):
                occupied.discard(op.zone_right)
        if not occupied:
            segments.append(current)
            current = []
    if current:
        segments.append(current)
    return segments


def _order_cross_ions(placement: Dict[int, int], spatial, liz_ids: List[int],
                      memory_zones: List[int],
                      blocks: Dict[Tuple[int, str], List[Operation]]) -> List[Tuple[int, int]]:
    """place_ion_any の呼び出し順を返す。

    shared M 内の cross-LIZ ion の初期並びを「最初に使う core が端から
    swap なしで取れる」順に整える。それ以外は placement の辞書順。
    """
    cross_by_zone: Dict[int, List[int]] = {}
    for q in spatial.shared_data:
        cross_by_zone.setdefault(placement[q], []).append(q)

    ordered_cross: Dict[int, List[int]] = {}
    for i in range(1, len(liz_ids)):
        m = memory_zones[i]
        if m not in cross_by_zone:
            continue
        # phase 0 で boundary を実行するのは奇数 idx の core
        left_i, right_i = i - 1, i
        first_i = left_i if left_i % 2 == 1 else right_i
        ops = blocks.get((liz_ids[first_i], 'boundary'), [])
        first_use: Dict[int, int] = {}
        for k, op in enumerate(ops):
            for t in getattr(op, 'targets', []):
                first_use.setdefault(t, k)
        ions = cross_by_zone[m]
        # 右側 core は list 末尾 (右端) から取る → 先に使うものを末尾へ
        take_from_right = (first_i == right_i)
        ions.sort(key=lambda q: first_use.get(q, float('inf')),
                  reverse=take_from_right)
        ordered_cross[m] = ions

    items: List[Tuple[int, int]] = []
    for q, z in placement.items():
        if q in spatial.shared_data and z in ordered_cross:
            continue  # 後で順序付きで入れる
        items.append((q, z))
    for m, ions in ordered_cross.items():
        for q in ions:
            items.append((q, m))
    return items


# ==========================================
# Op → DSL (入力回路ダンプ用)
# ==========================================

def _op_dsl(op: Operation) -> str:
    if isinstance(op, ResetOp):
        return "RESET " + " ".join(str(t) for t in op.targets)
    if isinstance(op, MeasureOp):
        return "MEASURE " + " ".join(str(t) for t in op.targets)
    if isinstance(op, GateOp):
        parts = [op.type] + [str(t) for t in op.targets]
        parts += [f"{p:g}" for p in (op.params or [])]
        return " ".join(parts)
    return f"# {op}"


# ==========================================
# Greedy-alternate placement (per-LIZ)
# ==========================================

def _placement_greedy_alt(liz_ids, memory_zones, liz_ops, spatial,
                          ancilla_assignments) -> List[Tuple[int, int]]:
    """per-LIZ greedy_alt 配置 (2-LIZ 専用)。

    各 core の pure イオンを使用順に、排他 M と shared M へ交互に割り当てる。
    チェーン内の並びは各 core のアクセス端に「先に使うものが近い」よう挿入:
      M1 (core-L は右端から取る) : 先に使うものを末尾へ
      M9 (core-R は左端から取る) : ancilla を先頭、以降先に使うもの順
      M5 : [L-core pure (使用順, 左端側) | cross | R-core pure (使用順, 右端側)]
    """
    if len(liz_ids) != 2:
        raise NotImplementedError("greedy_alt placement は 2-LIZ のみ対応")
    left_liz, right_liz = liz_ids[0], liz_ids[1]
    m_left, m_shared, m_right = memory_zones[0], memory_zones[1], memory_zones[2]

    def first_use(liz):
        fu: Dict[int, int] = {}
        for k, op in enumerate(liz_ops[liz]):
            for t in getattr(op, 'targets', []):
                fu.setdefault(t, k)
        return fu

    fus = {liz: first_use(liz) for liz in liz_ids}
    pure_by_liz: Dict[int, List[int]] = {}
    for q, liz in spatial.pure_data.items():
        pure_by_liz.setdefault(liz, []).append(q)

    excl_list = {left_liz: [], right_liz: []}
    m5_side = {left_liz: [], right_liz: []}
    for liz in (left_liz, right_liz):
        ions = sorted(pure_by_liz.get(liz, []),
                      key=lambda q: fus[liz].get(q, float('inf')))
        for i, q in enumerate(ions):
            (excl_list if i % 2 == 0 else m5_side)[liz].append(q)

    cross = sorted(spatial.shared_data,
                   key=lambda q: fus[right_liz].get(q, float('inf')))

    items: List[Tuple[int, int]] = []
    # M1: 先に使うものを右端 (末尾) へ → 逆順で挿入、ancilla は最後 (右端)
    for q in reversed(excl_list[left_liz]):
        items.append((q, m_left))
    items.append((ancilla_assignments[left_liz], m_left))
    # M9: ancilla 先頭 (左端)、以降使用順
    items.append((ancilla_assignments[right_liz], m_right))
    for q in excl_list[right_liz]:
        items.append((q, m_right))
    # M5: [L pure (使用順=左端側) | cross (R の使用が早いものを右寄り) | R pure]
    for q in m5_side[left_liz]:
        items.append((q, m_shared))
    for q in reversed(cross):
        items.append((q, m_shared))
    for q in reversed(m5_side[right_liz]):
        items.append((q, m_shared))
    return items


# ==========================================
# Compilation
# ==========================================

def compile_per_liz(program: Program, ancilla_qubits: List[int],
                    data_qubits: List[int], arch: QCCDArchitecture,
                    code_distance: int,
                    config: Optional[CompilerConfig] = None,
                    stab_order: str = 'phase_alt',
                    placement_mode: str = '3tier',
                    time_params=None,
                    scan_reverse: bool = False,
                    initial_zone_contents: Optional[Dict[int, List[int]]] = None,
                    ) -> PerLIZResult:
    """回路を per-core stream にコンパイルする。

    stab_order:
      'phase_alt'     : pure/boundary block + 交互 phase (従来)
      'boustrophedon' : 各 LIZ が自分の帯を左上から牛耕式に走査。block = 行。
                        plan 順は行ごとに LIZ を交互に (両 LIZ が同時に進む想定)
    placement_mode:
      '3tier'      : pure→排他 M / cross→shared M (従来)
      'greedy_alt' : per-LIZ で排他 M / shared M へ使用順に交互配置
    time_params:
      estimator.TimeParameters (フォノンパラメータの外部注入用、None でデフォルト)
    scan_reverse:
      True で牛耕式走査を完全逆順にする (複数ラウンドの実行順反転用)。
      ラウンド k の最終配置から逆走を始めると再配置コストが小さい。
      boustrophedon のみ対応。
    initial_zone_contents:
      zone_id -> [ion_ids] (順序付き)。前ラウンドの最終状態
      (PerLIZResult.state.zone_contents) から次ラウンドを開始する用。
      None なら placement_mode の計算配置から開始。指定時も home_map
      (退避定位置) は placement_mode の計算結果を使う。
    """
    config = config or CompilerConfig(eviction_strategy="balanced")
    if scan_reverse and stab_order != 'boustrophedon':
        raise NotImplementedError("scan_reverse は boustrophedon のみ対応")
    liz_ids = [z.id for z in arch.get_liz_zones()]
    memory_zones = [z.id for z in arch.zones if z.zone_type == "M"]

    spatial = partition_spatial_row(
        program=program, ancilla_qubits=ancilla_qubits,
        liz_zone_ids=liz_ids, code_distance=code_distance,
        phase_alternation=(stab_order == 'phase_alt'))
    new_prog = spatial.new_program
    ancilla_to_liz = {a: liz for liz, a in spatial.ancilla_assignments.items()}
    partition = StabilizerPartition(ancilla_to_liz=ancilla_to_liz)

    stabs = _identify_stabilizers(new_prog, set(ancilla_to_liz.keys()))
    blocks: Dict[Tuple[int, str], List[Operation]] = {}
    block_needs: Dict[Tuple[int, str], Set[int]] = {}
    scan_lines: List[str] = []

    if stab_order == 'phase_alt':
        # --- block 分割: (liz, 'local'|'boundary') ---
        for stab in stabs:
            liz = ancilla_to_liz[stab.ancilla_original]
            label = 'boundary' if (stab.data_qubits & spatial.shared_data) else 'local'
            key = (liz, label)
            blocks.setdefault(key, []).extend(
                new_prog.instructions[i] for i in stab.op_indices)
            block_needs.setdefault(key, set()).update(
                stab.data_qubits | {stab.ancilla_original})
        # 計画順: 偶数 idx local→boundary / 奇数 逆
        epoch_keys: List[Tuple[int, str]] = []
        for phase in (0, 1):
            for i, liz in enumerate(liz_ids):
                if i % 2 == 1:
                    label = 'boundary' if phase == 0 else 'local'
                else:
                    label = 'local' if phase == 0 else 'boundary'
                if blocks.get((liz, label)):
                    epoch_keys.append((liz, label))
    elif stab_order == 'boustrophedon':
        # --- block 分割: (liz, 'rowK') 牛耕式 ---
        coord = {d.id: (float(d.location.trap), float(d.location.site))
                 for d in new_prog.declarations}
        geo = []
        for stab in stabs:
            rows = [coord[q][0] for q in stab.data_qubits]
            cols = [coord[q][1] for q in stab.data_qubits]
            geo.append((sum(rows) / len(rows), sum(cols) / len(cols)))
        per_liz: Dict[int, List[int]] = {}
        for s_idx, stab in enumerate(stabs):
            per_liz.setdefault(ancilla_to_liz[stab.ancilla_original],
                               []).append(s_idx)
        n_rows: Dict[int, int] = {}
        for liz in liz_ids:
            buckets: Dict[float, List[int]] = {}
            for s in per_liz.get(liz, []):
                buckets.setdefault(round(geo[s][0] * 2) / 2, []).append(s)
            for k, rk in enumerate(sorted(buckets)):
                # 順走: 偶数行は列昇順・奇数行は降順 (牛耕)。
                # 逆走 (scan_reverse) は順走パスの完全な逆順なので、
                # 行内の向きも反転する
                desc = (k % 2 == 1) if not scan_reverse else (k % 2 == 0)
                row_stabs = sorted(buckets[rk], key=lambda s: geo[s][1],
                                   reverse=desc)
                key = (liz, f"row{k}")
                for s in row_stabs:
                    blocks.setdefault(key, []).extend(
                        new_prog.instructions[i] for i in stabs[s].op_indices)
                    block_needs.setdefault(key, set()).update(
                        stabs[s].data_qubits | {stabs[s].ancilla_original})
                    scan_lines.append(
                        f"#   LIZ{liz} row{k}: stab@({geo[s][0]:.1f},{geo[s][1]:.1f})"
                        f" anc={stabs[s].ancilla_original}"
                        f" data={sorted(stabs[s].data_qubits)}")
            n_rows[liz] = len(buckets)
        # 計画順: 行ごとに LIZ を交互 (両 LIZ が同時に自帯を上から走査する想定)。
        # 逆走時は最終行から上へ戻る
        epoch_keys = []
        row_order = range(max(n_rows.values()))
        if scan_reverse:
            row_order = reversed(list(row_order))
        for k in row_order:
            for liz in liz_ids:
                if blocks.get((liz, f"row{k}")):
                    epoch_keys.append((liz, f"row{k}"))
    else:
        raise ValueError(f"unknown stab_order: {stab_order}")

    # 各 LIZ の plan 順 op 列 (placement の使用順ソートに使う)
    liz_ops: Dict[int, List[Operation]] = {liz: [] for liz in liz_ids}
    for (liz, label) in epoch_keys:
        liz_ops[liz].extend(blocks[(liz, label)])

    # --- placement ---
    if placement_mode == '3tier':
        placement = compute_initial_placement_3tier(spatial, arch, data_qubits)
        # 最終 LIZ の ancilla はデフォルトで左 (shared) M に置かれるため右端の
        # 排他 M へ移す。中間 LIZ (N>=3) は排他 M が無いのでそのまま。
        placement[spatial.ancilla_assignments[liz_ids[-1]]] = memory_zones[-1]
        # cross イオンの初期並び: 最初に使う core が端から取れる順
        pseudo_blocks = {(liz, 'boundary'): liz_ops[liz] for liz in liz_ids}
        ordered_items = _order_cross_ions(placement, spatial, liz_ids,
                                          memory_zones, pseudo_blocks)
    elif placement_mode == 'greedy_alt':
        ordered_items = _placement_greedy_alt(
            liz_ids, memory_zones, liz_ops, spatial,
            spatial.ancilla_assignments)
        placement = dict(ordered_items)
    elif placement_mode == '3tier_usage':
        # 3tier の割当 (pure→排他 M, cross→shared M) のまま、チェーン内の
        # 並びだけ使用順にする: 各 core のアクセス端に先に使うイオンを置く。
        # greedy_alt の「排他 M の掘り出し激減」だけを取り、M5 混雑は避ける。
        if len(liz_ids) != 2:
            raise NotImplementedError("3tier_usage は 2-LIZ のみ対応")
        left_liz, right_liz = liz_ids[0], liz_ids[1]
        m_left, m_shared, m_right = memory_zones[0], memory_zones[1], memory_zones[2]

        def _first_use(liz):
            fu: Dict[int, int] = {}
            for k, op in enumerate(liz_ops[liz]):
                for t in getattr(op, 'targets', []):
                    fu.setdefault(t, k)
            return fu

        fus = {liz: _first_use(liz) for liz in liz_ids}
        pure_by_liz: Dict[int, List[int]] = {}
        for q, liz in spatial.pure_data.items():
            pure_by_liz.setdefault(liz, []).append(q)
        left_pure = sorted(pure_by_liz.get(left_liz, []),
                           key=lambda q: fus[left_liz].get(q, float('inf')))
        right_pure = sorted(pure_by_liz.get(right_liz, []),
                            key=lambda q: fus[right_liz].get(q, float('inf')))
        cross = sorted(spatial.shared_data,
                       key=lambda q: fus[right_liz].get(q, float('inf')))
        ordered_items = []
        # M1: 先に使うものを右端へ → 逆順挿入、ancilla 最後 (右端)
        for q in reversed(left_pure):
            ordered_items.append((q, m_left))
        ordered_items.append((spatial.ancilla_assignments[left_liz], m_left))
        # M5: cross のみ。右 core が先に使うものを右端へ
        for q in reversed(cross):
            ordered_items.append((q, m_shared))
        # M9: ancilla 先頭 (左端)、以降使用順
        ordered_items.append((spatial.ancilla_assignments[right_liz], m_right))
        for q in right_pure:
            ordered_items.append((q, m_right))
        placement = dict(ordered_items)
    else:
        raise ValueError(f"unknown placement_mode: {placement_mode}")

    # 前ラウンドの最終状態から開始する場合: 物理配置だけ差し替え、
    # home_map (退避定位置) は placement_mode の計算結果を維持する
    if initial_zone_contents is not None:
        ordered_items = [(ion, z)
                         for z in sorted(initial_zone_contents)
                         for ion in initial_zone_contents[z]]

    ordered_placement = dict(ordered_items)
    state = QCCDState(architecture=arch)
    state.phonon_params = phonon_params_from(time_params)
    state.home_map = dict(placement)
    for ion, z in ordered_items:
        state.place_ion_any(ion, z)

    views = {liz: CoreView(arch, liz, _core_zones(arch, liz)) for liz in liz_ids}
    planners: Dict[int, MovementPlanner] = {}
    for liz in liz_ids:
        p = MovementPlanner(views[liz], config, time_params=time_params)
        p.partition = partition
        planners[liz] = p

    # 各 core が隣接 core と共有する Memory
    shared_ms: Dict[int, Set[int]] = {liz: set() for liz in liz_ids}
    for i, liz in enumerate(liz_ids):
        if i > 0:
            shared_ms[liz].add(memory_zones[i])
        if i < len(liz_ids) - 1:
            shared_ms[liz].add(memory_zones[i + 1])

    # --- block を計画順に逐次 plan し、Temp 静止点で segment に分割 ---
    pv = ParallelValidator(arch)
    segments: List[Segment] = []
    for key in epoch_keys:
        liz, label = key
        view = views[liz]
        # このブロックが必要としないイオンしか住んでいない shared M を
        # 退避・探索候補から外す (相手 core との無用な競合を作らないため)
        homes_needed = {placement[q] for q in block_needs[key] if q in placement}
        view.blocked = {m for m in shared_ms[liz] if m not in homes_needed}
        block_prog = Program(declarations=new_prog.declarations,
                             instructions=blocks[key])
        dag = DAGBuilder(block_prog, stabilizer_parallel=False).build()
        state.architecture = view
        try:
            ops = planners[liz].plan(dag, state=state)
        finally:
            state.architecture = arch
        for seg_ops in _split_at_temp_quiescence(ops, arch):
            s = Segment(core_id=liz, label=label, ops=seg_ops)
            for op in seg_ops:
                s.zones |= _op_zones(op, pv)
                s.ions |= pv._ions_used(op)
            segments.append(s)

    # --- sync 辺: 計画順で資源が交差する cross-core segment ペア ---
    # 同 core の segment は stream 順で暗黙に直列なので辺は張らない。
    # 実 ion / 非 Temp zone の共有 → 'hb' (計画順を実行順として強制)。
    # Temp zone だけの共有 (swap 一時退避など) → 'mutex' (重なり禁止のみ)。
    temp_zones = {z.id for z in arch.zones if z.zone_type == "T"}
    syncs: List[SyncEdge] = []
    for j in range(len(segments)):
        for i in range(j):
            if segments[i].core_id == segments[j].core_id:
                continue
            sz = segments[i].zones & segments[j].zones
            si = segments[i].ions & segments[j].ions
            if sz or si:
                kind = 'mutex' if (not si and sz <= temp_zones) else 'hb'
                syncs.append(SyncEdge(src=i, dst=j, shared_zones=sz,
                                      shared_ions=si, kind=kind))

    streams: Dict[int, List[int]] = {liz: [] for liz in liz_ids}
    for idx, s in enumerate(segments):
        streams[s.core_id].append(idx)

    merged_flat = [op for s in segments for op in s.ops]

    # --- 並べ替え後の入力回路 (DSL) ---
    circ_lines = [f"# rewritten input circuit  stab_order={stab_order}  "
                  f"placement={placement_mode}"]
    if scan_lines:
        circ_lines.append("# scan order:")
        circ_lines.extend(scan_lines)
    circ_lines.append("")
    for d in new_prog.declarations:
        circ_lines.append(f"QUBIT {d.id} AT ({d.location.trap}, {d.location.site})")
    circ_lines.append("")
    circ_lines.append("# initial placement (挿入順 = チェーン内並び):")
    for q, z in ordered_items:
        circ_lines.append(f"#   ion {q} -> zone {z}")
    for (liz, label) in epoch_keys:
        circ_lines.append(f"\n# === LIZ {liz} {label} ===")
        for op in blocks[(liz, label)]:
            circ_lines.append(_op_dsl(op))
    input_circuit = "\n".join(circ_lines) + "\n"

    return PerLIZResult(segments=segments, streams=streams, syncs=syncs,
                        merged_flat=merged_flat,
                        initial_placement=ordered_placement,
                        state=state, arch=arch,
                        input_circuit=input_circuit)


# ==========================================
# Event simulation (estimator v2)
# ==========================================

def estimate_streams(result: PerLIZResult, params=None) -> PerLIZResult:
    """per-core タイムラインを event simulation で評価する。

    各 segment は core 内 Scheduler (zone-conflict ASAP) で並列化し、
    duration = その timesteps の並列時間。segment の開始は
    max(同 core の直前 segment の終了, sync 辺の src segment の終了)。
    wall clock = 全 core の終了時刻の max。
    """
    pe = ParallelEstimator(params) if params else ParallelEstimator()
    sch = Scheduler()
    for s in result.segments:
        s.timesteps = sch.schedule(s.ops)
        s.duration_us = pe.estimate(s.timesteps).total_time if s.timesteps else 0.0

    # dst ごとの入力辺を索引 (mutex は両方向から見えるように)
    in_hb: Dict[int, List[SyncEdge]] = {}
    mutex_partners: Dict[int, List[Tuple[int, SyncEdge]]] = {}
    for e in result.syncs:
        if e.kind == 'hb':
            in_hb.setdefault(e.dst, []).append(e)
        else:
            mutex_partners.setdefault(e.dst, []).append((e.src, e))
            mutex_partners.setdefault(e.src, []).append((e.dst, e))

    core_time: Dict[int, float] = {c: 0.0 for c in result.streams}
    for idx, s in enumerate(result.segments):
        start = core_time[s.core_id]
        for edge in in_hb.get(idx, []):
            dep_end = result.segments[edge.src].end_us
            if dep_end > start:
                edge.wait_us = dep_end - start
                start = dep_end
        # mutex: 既にスケジュール済みの partner (計画順で前) と時間帯が
        # 重ならない最早開始を探す。順序は自由なので隙間に入ってよい。
        busy = sorted(
            (result.segments[p].start_us, result.segments[p].end_us, e)
            for p, e in mutex_partners.get(idx, []) if p < idx)
        ready = start
        for b_start, b_end, edge in busy:
            if start + s.duration_us <= b_start:
                break  # 手前の隙間に収まる
            if start < b_end:
                edge.wait_us += b_end - start
                start = b_end
        s.start_us = start
        s.end_us = start + s.duration_us
        core_time[s.core_id] = s.end_us

    result.per_core_end_us = dict(core_time)
    result.per_core_busy_us = {
        c: sum(result.segments[i].duration_us for i in idxs)
        for c, idxs in result.streams.items()}
    result.wall_clock_us = max(core_time.values()) if core_time else 0.0
    return result


def save_streams(result: PerLIZResult, out_dir, prefix: str) -> List[str]:
    """コンパイル結果をファイルに保存する。

    per-LIZ ファイル分割 (format 案 B): 1 core = 1 stream = 1 ファイル。
    併合ビュー (検証用の canonical linearization) と summary も出す。

    Returns: 書き出したファイルパスのリスト
    """
    from pathlib import Path
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    pe = ParallelEstimator()
    written: List[str] = []

    def ms(x):
        return x / 1000.0

    # sync 索引: segment -> 入出力辺
    in_edges: Dict[int, List[SyncEdge]] = {}
    out_edges: Dict[int, List[SyncEdge]] = {}
    for e in result.syncs:
        in_edges.setdefault(e.dst, []).append(e)
        out_edges.setdefault(e.src, []).append(e)

    # --- per-core stream ファイル ---
    for core, idxs in sorted(result.streams.items()):
        path = out / f"{prefix}_core{core}.txt"
        with open(path, 'w', encoding='utf-8') as f:
            f.write(f"# per-LIZ stream: core {core} (LIZ zone {core})\n")
            f.write(f"# segments={len(idxs)}  "
                    f"busy={ms(result.per_core_busy_us.get(core, 0)):.2f} ms  "
                    f"end={ms(result.per_core_end_us.get(core, 0)):.2f} ms\n\n")
            for idx in idxs:
                s = result.segments[idx]
                f.write(f"== seg {idx} ({s.label})  "
                        f"t=[{ms(s.start_us):.3f} .. {ms(s.end_us):.3f}] ms  "
                        f"ops={len(s.ops)}\n")
                for e in in_edges.get(idx, []):
                    src = result.segments[e.src]
                    f.write(f"   <- sync[{e.kind}] after seg {e.src} "
                            f"(core {src.core_id}.{src.label}) "
                            f"zones={sorted(e.shared_zones)} "
                            f"ions={sorted(e.shared_ions)}"
                            f"{f'  wait={ms(e.wait_us):.2f} ms' if e.wait_us else ''}\n")
                for t, step in enumerate(s.timesteps):
                    ops_txt = "; ".join(pe._op_summary(op) for op in step.ops)
                    f.write(f"   TS {t}: {ops_txt}\n")
                f.write("\n")
        written.append(str(path))

    # --- 併合ビュー (canonical linearization, 計画順) ---
    path = out / f"{prefix}_merged.txt"
    with open(path, 'w', encoding='utf-8') as f:
        f.write("# merged canonical view (計画順 = 検証に使う linearization)\n")
        f.write(f"# total ops={len(result.merged_flat)}\n\n")
        gt = 0
        for idx, s in enumerate(result.segments):
            f.write(f"== seg {idx} (core {s.core_id}.{s.label})\n")
            for step in s.timesteps:
                ops_txt = "; ".join(pe._op_summary(op) for op in step.ops)
                f.write(f"   TS {gt}: {ops_txt}\n")
                gt += 1
    written.append(str(path))

    # --- 並べ替え後の入力回路 ---
    if result.input_circuit:
        path = out / f"{prefix}_input_circuit.txt"
        path.write_text(result.input_circuit, encoding='utf-8')
        written.append(str(path))

    # --- summary ---
    path = out / f"{prefix}_summary.txt"
    ph = result.phonon_summary()
    with open(path, 'w', encoding='utf-8') as f:
        f.write(f"wall_clock_ms: {ms(result.wall_clock_us):.2f}\n")
        for core, idxs in sorted(result.streams.items()):
            f.write(f"core {core}: busy={ms(result.per_core_busy_us[core]):.2f} ms  "
                    f"end={ms(result.per_core_end_us[core]):.2f} ms  "
                    f"segments={len(idxs)}\n")
        binding = [e for e in result.syncs if e.wait_us > 0]
        f.write(f"sync_edges: {len(result.syncs)} "
                f"(waits: {len(binding)}, "
                f"total {ms(sum(e.wait_us for e in binding)):.2f} ms)\n")
        f.write(f"phonon_max_axial: {ph['max_axial']:.2f}\n")
        f.write(f"phonon_max_transversal: {ph['max_transversal']:.3f}\n")
        f.write(f"phonon_max_transversal2: {ph['max_transversal2']:.3f}\n")
    written.append(str(path))

    return written


def validate_merged(result: PerLIZResult):
    """per-segment スケジュールを計画順に連結した TimeStep 列を検証する。

    計画順は正当な linearization であり、時間的に重なりうる segment は
    sync 辺の構成上資源素 (状態更新が可換) なので、この逐次化された検証で
    並列実行の正しさを代表できる。global re-scheduling はしない
    (ion 依存を追わない Scheduler で異 stream の ops を再パックすると
    不正な並びが生じうるため)。
    """
    timesteps: List[TimeStep] = []
    for s in result.segments:
        timesteps.extend(s.timesteps if s.timesteps
                         else Scheduler().schedule(s.ops))
    return ParallelValidator(result.arch).validate(
        timesteps, result.initial_placement)
