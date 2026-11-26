# Adapted from THeatA for the bundled workbench.
"""
Memory Reorder Optimizer

長時間操作（Measure, Reset）の実行中に、Memory内のイオンを
将来のゲート実行に最適な並びに整列させるSwap操作を生成する。

設計思想:
- ゲートの実行順（DAGの依存関係）は不変
- 搬送操作（Split/Shuttle/Merge/Swap）の順番は自由に変更可能
- Measure(1000μs) の間に Memory 内 Swap を詰め込んで配置を最適化

使用方法:
    optimizer = MemoryReorderOptimizer(architecture)
    reorder_ops = optimizer.generate_reorder_ops(
        dag, state, current_node_id
    )
    # reorder_ops を flat op 列に追加 → Scheduler が Measure と並列配置

複数Memory対応:
    各Memoryゾーンについて独立に最適並びを計算する。
    LIZに近い端に最初に使うイオンを配置する。
"""

from typing import List, Dict, Set, Optional, Tuple
from .ast_def import Operation, GateOp, MeasureOp, ResetOp, SwapOp, SplitOp, MergeOp
from .dag_builder import DAG, DAGNode


class MemoryReorderOptimizer:
    """
    Measure/Reset の後に Memory 内イオンを将来のゲート順に並べ替える

    generate_reorder_ops() は、現在の Measure/Reset 以降〜次の Measure まで
    に登場するイオンの出現順を計算し、各 Memory ゾーンで最適な並びにする
    Swap 列を返す。
    """

    def __init__(self, architecture):
        """
        Args:
            architecture: QCCDArchitecture インスタンス
        """
        self.architecture = architecture

    def generate_reorder_ops(
        self,
        dag: DAG,
        state,  # QCCDState
        just_executed_node_id: int,
    ) -> List[Operation]:
        """
        Measure/Reset 直後に呼ばれ、Memory 内の並べ替え Swap 列を生成する

        Args:
            dag: 現在の DAG（execute 済み状態）
            state: 現在の QCCDState（Measure/Reset 実行後の状態）
            just_executed_node_id: 今実行した Measure/Reset ノードの ID

        Returns:
            List[Operation]: Swap 操作の列（zone_id アノテーション付き）
        """
        # 次の Measure/Reset までに登場するイオンの出現順を取得
        upcoming_ions = self._collect_upcoming_ions(dag, just_executed_node_id)
        if not upcoming_ions:
            return []

        # 各 Memory ゾーンについて Swap 列を生成
        all_ops = []
        for mem_zone in self.architecture.get_memory_zones():
            zone_id = mem_zone.id
            current_contents = list(state.zone_contents.get(zone_id, []))
            if len(current_contents) <= 1:
                continue

            # この Memory にいるイオンのうち、upcoming に登場するものの目標順
            target_order = self._compute_target_order(
                current_contents, upcoming_ions, zone_id
            )
            if target_order is None:
                continue

            # 現在の並びから目標への Swap 列を生成
            swaps = self._generate_swaps(current_contents, target_order, zone_id, state)
            all_ops.extend(swaps)

        return all_ops

    def _collect_upcoming_ions(
        self, dag: DAG, after_node_id: int
    ) -> List[int]:
        """
        DAG を先読みして、次の Measure まで (or DAG 終端まで) に
        登場するイオンを出現順で返す。

        BFS で DAG のフロントレイヤーから辿り、ゲートのターゲットイオンを収集。
        Measure/Reset に到達したらその分岐は停止。
        """
        # 現在の front layer から BFS
        front_layer = dag.get_front_layer()
        if not front_layer:
            return []

        visited = set(dag.get_executed())
        queue = sorted(front_layer)  # ID順で処理（決定的順序）
        ion_order = []
        seen_ions = set()

        while queue:
            node_id = queue.pop(0)
            if node_id in visited:
                continue
            visited.add(node_id)

            node = dag.nodes[node_id]
            op = node.operation

            # Measure/Reset に到達したら、この分岐は停止
            if isinstance(op, (MeasureOp, ResetOp)):
                continue

            # ゲートのターゲットイオンを収集
            if isinstance(op, GateOp):
                for ion_id in op.targets:
                    if ion_id not in seen_ions:
                        seen_ions.add(ion_id)
                        ion_order.append(ion_id)

            # 後続ノードをキューに追加（依存が全て解決済みのもの）
            for succ_id in sorted(node.successors):
                succ_node = dag.nodes[succ_id]
                if succ_node.predecessors.issubset(visited):
                    queue.append(succ_id)

        return ion_order

    def _compute_target_order(
        self,
        current_contents: List[int],
        upcoming_ions: List[int],
        zone_id: int,
    ) -> Optional[List[int]]:
        """
        Memory ゾーン内のイオンの目標並びを計算する

        LIZ に近い端に、最初に使うイオンを配置する。
        upcoming に含まれないイオンは奥側に移動。

        Args:
            current_contents: Memory 内の現在のイオン並び
            upcoming_ions: 出現順のイオンリスト
            zone_id: Memory ゾーン ID

        Returns:
            目標並び or None（変更不要の場合）
        """
        contents_set = set(current_contents)

        # この Memory にいて upcoming に登場するイオン（出現順）
        relevant = [ion for ion in upcoming_ions if ion in contents_set]

        # この Memory にいて upcoming に登場しないイオン
        irrelevant = [ion for ion in current_contents if ion not in set(relevant)]

        if not relevant:
            return None

        # LIZ に近い方を判定
        # zone_id より小さい LIZ があれば左側が LIZ に近い
        # zone_id より大きい LIZ があれば右側が LIZ に近い
        liz_zones = self.architecture.get_liz_zones()
        liz_on_right = any(lz.id > zone_id for lz in liz_zones)
        liz_on_left = any(lz.id < zone_id for lz in liz_zones)

        if liz_on_right and not liz_on_left:
            # LIZ は右側 → 右端に最初に使うイオンを配置
            # [irrelevant..., relevant(逆順)]
            target = irrelevant + list(reversed(relevant))
        elif liz_on_left and not liz_on_right:
            # LIZ は左側 → 左端に最初に使うイオンを配置
            # [relevant..., irrelevant]
            target = list(relevant) + irrelevant
        else:
            # 両側に LIZ がある（または見つからない）→ 最も近い LIZ 方向を使う
            min_dist_right = min(
                (lz.id - zone_id for lz in liz_zones if lz.id > zone_id),
                default=float('inf')
            )
            min_dist_left = min(
                (zone_id - lz.id for lz in liz_zones if lz.id < zone_id),
                default=float('inf')
            )
            if min_dist_right <= min_dist_left:
                target = irrelevant + list(reversed(relevant))
            else:
                target = list(relevant) + irrelevant

        if target == current_contents:
            return None

        return target

    def _generate_swaps(
        self,
        current: List[int],
        target: List[int],
        zone_id: int,
        state,
    ) -> List[Operation]:
        """
        current → target への最小隣接 Swap 列を生成する

        バブルソート方式: 隣接 Swap のみ許可（物理制約）
        各 Swap は _expand_swap_inline と同じ形式で生成。

        state も同時に更新する。
        """
        arr = list(current)
        ops = []

        # target 内の各要素のインデックスマップ
        target_pos = {ion: i for i, ion in enumerate(target)}

        # バブルソート: target の並びに向かって隣接 Swap
        n = len(arr)
        changed = True
        while changed:
            changed = False
            for i in range(n - 1):
                # arr[i] と arr[i+1] を比較
                if target_pos.get(arr[i], i) > target_pos.get(arr[i+1], i+1):
                    # Swap が必要
                    left_ion = arr[i]
                    right_ion = arr[i + 1]

                    swap_op = SwapOp(targets=[left_ion, right_ion], zone_id=zone_id)
                    ops.append(swap_op)
                    state.swap_ions(left_ion, right_ion)

                    arr[i], arr[i + 1] = arr[i + 1], arr[i]
                    changed = True

        return ops
