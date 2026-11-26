# Adapted from THeatA for the bundled workbench.
"""
DAG Builder: AST から Directed Acyclic Graph を構築

依存関係の種類:
1. データ依存 - 同じion_idに対する連続した操作
2. (将来) 位置依存 - 同じ物理位置を使う操作
3. (将来) リソース依存 - 同じハードウェアリソースを使う操作

現在の実装: データ依存のみを追跡

注意:
- 移動操作 (SHUTTLE, SWAP, SPLIT, MERGE) はDAGに含めない
  → 後段のスケジューリングで挿入
- MEASUREは別管理（アーキテクチャ依存のため）
"""

from dataclasses import dataclass, field
from typing import List, Dict, Set, Optional
from .ast_def import (
    Program, Operation, QubitDecl, Location,
    GateOp, ShuttleOp, SwapOp, SplitOp, MergeOp,
    MeasureOp, ResetOp, BarrierOp, CoolOp
)


# ==========================================
# DAG ノード
# ==========================================

@dataclass
class DAGNode:
    """DAGの各ノード（1つの操作に対応）"""
    id: int                          # ノードの一意識別子
    operation: Operation             # 対応するAST操作
    predecessors: Set[int] = field(default_factory=set)  # 依存先ノードID
    successors: Set[int] = field(default_factory=set)    # 後続ノードID
    stabilizer_id: Optional[int] = None  # 所属スタビライザのアンシラID（parallel時）

    def __repr__(self):
        return f"Node({self.id}, {self.operation.type}, pred={self.predecessors}, succ={self.successors})"


# ==========================================
# DAG
# ==========================================

@dataclass
class DAG:
    """Directed Acyclic Graph with Front Layer management"""
    nodes: Dict[int, DAGNode] = field(default_factory=dict)
    entry_nodes: Set[int] = field(default_factory=set)   # 依存なしノード（開始点）
    exit_nodes: Set[int] = field(default_factory=set)    # 後続なしノード（終了点）

    # Front Layer管理用
    _executed: Set[int] = field(default_factory=set)     # 実行済みノード
    _front_layer: Set[int] = field(default_factory=set)  # 現在のFront Layer

    def add_node(self, node: DAGNode):
        """ノードを追加"""
        self.nodes[node.id] = node

    def add_edge(self, from_id: int, to_id: int):
        """エッジを追加（from_id -> to_id の依存関係）"""
        if from_id in self.nodes and to_id in self.nodes:
            self.nodes[from_id].successors.add(to_id)
            self.nodes[to_id].predecessors.add(from_id)

    def compute_entry_exit(self):
        """entry_nodes と exit_nodes を計算"""
        self.entry_nodes = {
            nid for nid, node in self.nodes.items()
            if len(node.predecessors) == 0
        }
        self.exit_nodes = {
            nid for nid, node in self.nodes.items()
            if len(node.successors) == 0
        }
        # Front Layerを初期化
        self._front_layer = self.entry_nodes.copy()
        self._executed = set()

    # ==========================================
    # Front Layer Management
    # ==========================================

    def get_front_layer(self) -> Set[int]:
        """
        現在のFront Layer（実行可能なノードの集合）を取得

        Front Layer = 全ての先行ノードが実行済みのノード
        """
        return self._front_layer.copy()

    def execute(self, node_id: int) -> bool:
        """
        ノードを実行済みとしてマークし、Front Layerを更新

        Returns:
            True if successful, False if node not in front layer
        """
        if node_id not in self._front_layer:
            return False

        # Front Layerから削除
        self._front_layer.remove(node_id)
        self._executed.add(node_id)

        # 後続ノードをチェック：全先行が実行済みならFront Layerに追加
        node = self.nodes[node_id]
        for succ_id in node.successors:
            succ_node = self.nodes[succ_id]
            # 全ての先行ノードが実行済みか確認
            if succ_node.predecessors.issubset(self._executed):
                self._front_layer.add(succ_id)

        return True

    def is_complete(self) -> bool:
        """全ノードが実行済みか"""
        return len(self._executed) == len(self.nodes)

    def get_executed(self) -> Set[int]:
        """実行済みノードの集合"""
        return self._executed.copy()

    def get_remaining(self) -> Set[int]:
        """未実行ノードの集合"""
        return set(self.nodes.keys()) - self._executed

    def reset_execution(self):
        """実行状態をリセット"""
        self._executed = set()
        self._front_layer = self.entry_nodes.copy()

    # ==========================================
    # Analysis Methods
    # ==========================================

    def topological_sort(self) -> List[int]:
        """トポロジカルソート（Kahnのアルゴリズム）"""
        in_degree = {nid: len(node.predecessors) for nid, node in self.nodes.items()}
        queue = [nid for nid, deg in in_degree.items() if deg == 0]
        result = []

        while queue:
            # 同じin_degreeのノードはID順で処理（決定的な順序のため）
            queue.sort()
            nid = queue.pop(0)
            result.append(nid)

            for succ_id in self.nodes[nid].successors:
                in_degree[succ_id] -= 1
                if in_degree[succ_id] == 0:
                    queue.append(succ_id)

        if len(result) != len(self.nodes):
            raise RuntimeError("DAG contains a cycle (should not happen)")

        return result

    def get_parallel_groups(self) -> List[List[int]]:
        """
        並列実行可能なグループを取得（静的解析用）
        各グループ内のノードは互いに依存がない

        注意: これは静的解析用。動的なスケジューリングには
        get_front_layer() + execute() を使用すること
        """
        in_degree = {nid: len(node.predecessors) for nid, node in self.nodes.items()}
        remaining = set(self.nodes.keys())
        groups = []

        while remaining:
            # 現在のin_degreeが0のノードを1グループに
            group = [nid for nid in remaining if in_degree[nid] == 0]
            if not group:
                raise RuntimeError("DAG contains a cycle")

            groups.append(sorted(group))

            # 処理済みとしてin_degreeを更新
            for nid in group:
                remaining.remove(nid)
                for succ_id in self.nodes[nid].successors:
                    in_degree[succ_id] -= 1

        return groups


# ==========================================
# DAG Builder
# ==========================================

class DAGBuilder:
    """ProgramからDAGを構築

    注意:
    - 移動操作 (SHUTTLE, SWAP, SPLIT, MERGE) はDAGに含めない

    stabilizer_parallel モード:
    - 入力形式: スタビライザ測定が上から順に並ぶ形式を前提
    - RESET ancilla → 新しいスタビライザ開始
    - MEASURE ancilla → そのスタビライザ終了
    - スタビライザ間は依存関係なし（独立したサブグラフ）
    """

    def __init__(self, program: Program, stabilizer_parallel: bool = False):
        """
        DAGBuilderを初期化

        Args:
            program: 入力プログラム
            stabilizer_parallel: Trueの場合、スタビライザ単位で依存関係を分離
                                入力形式: [RESET a → gates → MEASURE a] が順に並ぶ

        Note:
            stabilizer_parallel=Trueの場合:
            - RESETでスタビライザ開始、MEASUREで終了
            - 各スタビライザ内は直列、スタビライザ間は並列
            - 前提: 入力回路がスタビライザ単位で順に記述されていること
        """
        self.program = program
        self.stabilizer_parallel = stabilizer_parallel
        self.dag = DAG()
        self.node_counter = 0

        # 各ion_idの最後の操作を追跡（通常モード用）
        self.last_op_for_ion: Dict[int, int] = {}

        # MEASURE対象からアンシラIDを事前取得（RESETなし回路対応）
        self._known_ancillas: Set[int] = self._prescan_ancillas(program)

        # スタビライザ並列モード用
        # ancilla_id -> {ion_id -> last_node_id}
        self.last_op_per_stabilizer: Dict[int, Dict[int, int]] = {}
        self.active_stabilizer: Optional[int] = None  # 現在処理中のアンシラ
        self.completed_stabilizers: Set[int] = set()  # 完了したアンシラ
        # アンシラ再利用対応: 各アンシラの最後のMEASUREノードID
        self.last_measure_per_ancilla: Dict[int, int] = {}
        # スタビライザの出現順序（interleave用）
        self.stabilizer_order: List[int] = []
        # stabilizer_id未割当のノード（後から割り当てる）
        self._pending_stabilizer_nodes: List[int] = []

        # 除外された操作を保存（後段で使用可能）
        self.excluded_movements: List[Operation] = []
        self.measurements: List[MeasureOp] = []

        # 直近の COOL バリアノード（COOL 以降に始まる新規イオン/スタビライザの
        # 操作が COOL より前に浮き上がらないようにするフォールバック依存先）
        self._cool_barrier_node: Optional[int] = None

    def _get_ancilla_from_reset(self, op: ResetOp) -> Optional[int]:
        """RESETからアンシラIDを取得（単一ターゲット前提）"""
        if len(op.targets) == 1:
            return op.targets[0]
        return None

    def _get_ancilla_from_measure(self, op: MeasureOp) -> Optional[int]:
        """MEASUREからアンシラIDを取得（単一ターゲット前提）"""
        if len(op.targets) == 1:
            return op.targets[0]
        return None

    @staticmethod
    def _prescan_ancillas(program: "Program") -> Set[int]:
        """MEASURE対象からアンシラIDを事前取得"""
        ancillas = set()
        for op in program.instructions:
            if isinstance(op, MeasureOp):
                for t in op.targets:
                    ancillas.add(t)
        return ancillas

    def _is_dag_operation(self, op: Operation) -> bool:
        """DAGに含める操作かどうかを判定"""
        # 移動操作は除外
        if isinstance(op, (ShuttleOp, SwapOp, SplitOp, MergeOp)):
            return False
        return True

    def _get_affected_ions(self, op: Operation) -> List[int]:
        """操作が影響するion_idのリストを取得"""
        if isinstance(op, GateOp):
            return op.targets
        elif isinstance(op, ResetOp):
            return op.targets
        elif isinstance(op, MeasureOp):
            return op.targets
        elif isinstance(op, BarrierOp):
            return op.targets
        elif isinstance(op, CoolOp):
            return [op.coolant_id] if op.coolant_id >= 0 else []
        else:
            return []

    def _create_node(self, op: Operation) -> DAGNode:
        """操作からDAGノードを作成"""
        node = DAGNode(id=self.node_counter, operation=op)
        self.node_counter += 1
        return node

    def build(self) -> DAG:
        """
        DAGを構築

        stabilizer_parallel=True の場合:
        - RESET: 新しいスタビライザ開始
        - MEASURE: そのスタビライザ終了
        - スタビライザ間は依存関係なし
        """
        for op in self.program.instructions:
            # 移動操作は除外して別保存
            if isinstance(op, (ShuttleOp, SwapOp, SplitOp, MergeOp)):
                self.excluded_movements.append(op)
                continue

            node = self._create_node(op)
            self.dag.add_node(node)

            # COOL: ゾーン単位の操作で、対象イオン集合が実行時まで確定しない。
            # 保守的に「これまでの全イオンの最後の操作」に依存するバリアとして
            # 扱い、以降の操作は全てこのノードの後に来るようにする。
            if isinstance(op, CoolOp):
                preds: Set[int] = set()
                if self.stabilizer_parallel:
                    for last_ops in self.last_op_per_stabilizer.values():
                        preds.update(last_ops.values())
                    preds.update(self.last_measure_per_ancilla.values())
                    # 以降の操作の依存先を COOL に付け替える
                    for last_ops in self.last_op_per_stabilizer.values():
                        for ion_id in last_ops:
                            last_ops[ion_id] = node.id
                    for anc in self.last_measure_per_ancilla:
                        self.last_measure_per_ancilla[anc] = node.id
                else:
                    preds = set(self.last_op_for_ion.values())
                    for ion_id in self.last_op_for_ion:
                        self.last_op_for_ion[ion_id] = node.id
                if not preds and self._cool_barrier_node is not None:
                    preds.add(self._cool_barrier_node)
                for pred_id in preds:
                    self.dag.add_edge(pred_id, node.id)
                self._cool_barrier_node = node.id
                continue

            # データ依存関係を設定
            affected_ions = self._get_affected_ions(op)
            predecessor_ids = set()

            if self.stabilizer_parallel:
                # === スタビライザ並列モード ===

                # RESET検出: 新しいスタビライザ開始
                if isinstance(op, ResetOp):
                    ancilla = self._get_ancilla_from_reset(op)
                    if ancilla is not None:
                        # 同じアンシラの前回のMEASUREがあれば依存を追加
                        if ancilla in self.last_measure_per_ancilla:
                            predecessor_ids.add(self.last_measure_per_ancilla[ancilla])

                        self.active_stabilizer = ancilla
                        self.last_op_per_stabilizer[ancilla] = {}
                        if ancilla not in self.completed_stabilizers:
                            self.stabilizer_order.append(ancilla)

                # RESETなし回路対応: R/MS gateからアンシラを検出してスタビライザ開始
                if self.active_stabilizer is None:
                    ancilla_candidates = [t for t in affected_ions if t in self._known_ancillas]
                    if ancilla_candidates:
                        ancilla = ancilla_candidates[0]
                        self.active_stabilizer = ancilla
                        if ancilla not in self.last_op_per_stabilizer:
                            self.last_op_per_stabilizer[ancilla] = {}
                        if ancilla not in set(self.stabilizer_order):
                            self.stabilizer_order.append(ancilla)
                        # 未割当ノードを遡及的に割り当て
                        for pending_id in self._pending_stabilizer_nodes:
                            self.dag.nodes[pending_id].stabilizer_id = ancilla
                            # 依存関係もスタビライザ内で構築
                            pending_node = self.dag.nodes[pending_id]
                            for ion_id in self._get_affected_ions(pending_node.operation):
                                self.last_op_per_stabilizer[ancilla][ion_id] = pending_id
                        self._pending_stabilizer_nodes = []

                # アクティブなスタビライザがあれば、その中で依存関係を構築
                if self.active_stabilizer is not None:
                    node.stabilizer_id = self.active_stabilizer
                    last_op_dict = self.last_op_per_stabilizer[self.active_stabilizer]

                    for ion_id in affected_ions:
                        if ion_id in last_op_dict:
                            predecessor_ids.add(last_op_dict[ion_id])

                    # 最後の操作を更新
                    for ion_id in affected_ions:
                        last_op_dict[ion_id] = node.id
                else:
                    # まだスタビライザ未検出: pending に追加
                    self._pending_stabilizer_nodes.append(node.id)

                # MEASURE検出: このスタビライザ終了
                if isinstance(op, MeasureOp):
                    ancilla = self._get_ancilla_from_measure(op)
                    if ancilla is not None and ancilla == self.active_stabilizer:
                        self.completed_stabilizers.add(ancilla)
                        # アンシラ再利用対応: このMEASUREのノードIDを保存
                        self.last_measure_per_ancilla[ancilla] = node.id
                        self.active_stabilizer = None
            else:
                # === 通常モード: 全体で依存関係を管理 ===
                for ion_id in affected_ions:
                    if ion_id in self.last_op_for_ion:
                        predecessor_ids.add(self.last_op_for_ion[ion_id])

                # 最後の操作を更新
                for ion_id in affected_ions:
                    self.last_op_for_ion[ion_id] = node.id

            # COOL バリア以降に初登場するイオンの操作は COOL の後に置く
            if not predecessor_ids and self._cool_barrier_node is not None:
                predecessor_ids.add(self._cool_barrier_node)

            # エッジを追加
            for pred_id in predecessor_ids:
                self.dag.add_edge(pred_id, node.id)

        self.dag.compute_entry_exit()
        return self.dag

    def get_stabilizer_pairs(self) -> List[tuple]:
        """連続するスタビライザのペアを返す（interleave用）"""
        pairs = []
        order = self.stabilizer_order
        i = 0
        while i + 1 < len(order):
            pairs.append((order[i], order[i + 1]))
            i += 2
        return pairs

    def add_hook_error_edges(self, pairs: List[tuple] = None):
        """
        Hook error avoidance エッジを追加

        各スタビライザペアの共有データ量子ビットについて、
        先頭スタビライザのCNOTブロック終了 → 後続スタビライザのCNOTブロック開始
        のエッジを追加する。

        条件: T(A_1) < T(B_1) かつ T(A_2) < T(B_2) を保証
        （Aが先頭、Bが後続）
        """
        if pairs is None:
            pairs = self.get_stabilizer_pairs()

        for stab_a, stab_b in pairs:
            # 各スタビライザのMS gateノードを収集
            a_ms_nodes = []
            b_ms_nodes = []
            for node in self.dag.nodes.values():
                if node.stabilizer_id == stab_a and isinstance(node.operation, GateOp) and node.operation.type == 'MS':
                    a_ms_nodes.append(node)
                if node.stabilizer_id == stab_b and isinstance(node.operation, GateOp) and node.operation.type == 'MS':
                    b_ms_nodes.append(node)

            # 各MS gateのデータ量子ビットを特定
            # アンシラ = stab_a or stab_b
            a_data_map = {}  # data_qubit -> node
            for n in a_ms_nodes:
                data_qubits = [t for t in n.operation.targets if t != stab_a]
                for dq in data_qubits:
                    a_data_map[dq] = n

            b_data_map = {}
            for n in b_ms_nodes:
                data_qubits = [t for t in n.operation.targets if t != stab_b]
                for dq in data_qubits:
                    b_data_map[dq] = n

            # 共有データ量子ビット
            shared_qubits = set(a_data_map.keys()) & set(b_data_map.keys())

            for sq in shared_qubits:
                a_ms_node = a_data_map[sq]
                b_ms_node = b_data_map[sq]

                # A のCNOTブロック最終ノード（MS nodeのsuccessor chain内で
                # 同じスタビライザのpost-rotation R gatesの最後）を見つける
                a_last = self._find_cnot_block_end(a_ms_node, stab_a, sq)
                # B のCNOTブロック最初のノード（MS nodeのpredecessor chain内で
                # 同じスタビライザのpre-rotation R gatesの最初）を見つける
                b_first = self._find_cnot_block_start(b_ms_node, stab_b, sq)

                # Hook error avoidance エッジ: A の CNOT 完了 → B の CNOT 開始
                self.dag.add_edge(a_last.id, b_first.id)

        # エッジ追加後にentry/exitを再計算
        self.dag.compute_entry_exit()

    def _find_cnot_block_end(self, ms_node: DAGNode, stab_id: int, data_qubit: int) -> DAGNode:
        """MS gateの後続post-rotation Rの最後のノードを返す"""
        current = ms_node
        # MS → post-R chain: 同じスタビライザ内で同じdata_qubitまたはancilla(=stab_id)を操作するR
        for succ_id in sorted(ms_node.successors):
            succ = self.dag.nodes[succ_id]
            if succ.stabilizer_id == stab_id and isinstance(succ.operation, GateOp) and succ.operation.type == 'R':
                if data_qubit in succ.operation.targets or stab_id in succ.operation.targets:
                    # post-rotation R: さらに先を探す
                    deeper = self._find_cnot_block_end(succ, stab_id, data_qubit)
                    if deeper.id > current.id:
                        current = deeper
        return current

    def _find_cnot_block_start(self, ms_node: DAGNode, stab_id: int, data_qubit: int) -> DAGNode:
        """MS gateの先行pre-rotation Rの最初のノードを返す"""
        current = ms_node
        for pred_id in sorted(ms_node.predecessors):
            pred = self.dag.nodes[pred_id]
            if pred.stabilizer_id == stab_id and isinstance(pred.operation, GateOp) and pred.operation.type == 'R':
                if data_qubit in pred.operation.targets or stab_id in pred.operation.targets:
                    deeper = self._find_cnot_block_start(pred, stab_id, data_qubit)
                    if deeper.id < current.id:
                        current = deeper
        return current


def build_dag(program: Program) -> DAG:
    """便利関数: ProgramからDAGを構築"""
    builder = DAGBuilder(program)
    return builder.build()


# ==========================================
# 可視化 (デバッグ用)
# ==========================================

def dag_to_dot(dag: DAG) -> str:
    """DAGをGraphviz DOT形式に変換（シンプル版）"""
    lines = ["digraph DAG {", "  rankdir=TB;"]

    for nid, node in dag.nodes.items():
        label = f"{nid}: {node.operation.type}"
        if hasattr(node.operation, 'targets'):
            targets = getattr(node.operation, 'targets', None)
            if targets:
                label += f"\\n{targets}"
        elif hasattr(node.operation, 'target'):
            label += f"\\n[{node.operation.target}]"

        lines.append(f'  {nid} [label="{label}"];')

    for nid, node in dag.nodes.items():
        for succ_id in node.successors:
            lines.append(f"  {nid} -> {succ_id};")

    lines.append("}")
    return "\n".join(lines)


def dag_to_dot_by_qubit(dag: DAG, num_qubits: int) -> str:
    """
    論文スタイルのDAG可視化（qubitごとに縦列配置）

    IonShuttler論文のFigure 4bのようなレイアウト:
    - 各qubitが縦の列を形成
    - 同じレベルの操作は横に並ぶ
    - 2-qubit gateは両方のqubitにまたがる
    """
    lines = [
        "digraph DAG {",
        "  rankdir=TB;",
        "  splines=ortho;",  # 直角の線
        "  nodesep=0.5;",
        "  ranksep=0.6;",
        "",
        "  // Qubit labels",
    ]

    # Qubitラベルを追加
    for q in range(num_qubits):
        lines.append(f'  q{q}_label [label="|q{q}⟩" shape=plaintext fontsize=14];')

    # Qubitラベルを同じrankに
    lines.append(f'  {{ rank=same; {" ".join(f"q{q}_label" for q in range(num_qubits))} }}')
    lines.append("")

    # ノードをレベル（並列グループ）ごとに配置
    groups = dag.get_parallel_groups()

    # 各ノードがどのqubitに属するかを決定
    node_qubits: Dict[int, List[int]] = {}
    for nid, node in dag.nodes.items():
        if hasattr(node.operation, 'targets'):
            node_qubits[nid] = node.operation.targets
        else:
            node_qubits[nid] = []

    lines.append("  // Operation nodes")

    # ゲートタイプに応じた色
    color_map = {
        "R": "#AED6F1",      # 薄い青
        "MS": "#ABEBC6",     # 薄い緑
        "RESET": "#F9E79F",  # 薄い黄
    }

    for nid, node in dag.nodes.items():
        op_type = node.operation.type
        color = color_map.get(op_type, "#D5D8DC")
        qubits = node_qubits.get(nid, [])

        if len(qubits) == 1:
            label = f"{nid}: {op_type}"
        elif len(qubits) == 2:
            label = f"{nid}: {op_type}"
        else:
            label = f"{nid}: {op_type}\\n{qubits}"

        lines.append(f'  {nid} [label="{label}" shape=box style=filled fillcolor="{color}"];')

    lines.append("")
    lines.append("  // Rank constraints (parallel groups)")

    # 各レベルを同じrankに
    for level, group in enumerate(groups):
        if group:
            nodes_str = " ".join(str(nid) for nid in group)
            lines.append(f'  {{ rank=same; {nodes_str} }}')

    lines.append("")
    lines.append("  // Edges (dependencies)")

    for nid, node in dag.nodes.items():
        for succ_id in sorted(node.successors):
            lines.append(f"  {nid} -> {succ_id};")

    # Qubitラベルから最初のノードへの不可視エッジ（整列用）
    lines.append("")
    lines.append("  // Invisible edges for alignment")

    if groups and groups[0]:
        first_level_nodes = groups[0]
        for nid in first_level_nodes:
            qubits = node_qubits.get(nid, [])
            if qubits:
                q = qubits[0]
                lines.append(f'  q{q}_label -> {nid} [style=invis];')

    lines.append("}")
    return "\n".join(lines)


# ==========================================
# 動作確認
# ==========================================

if __name__ == "__main__":
    from .qccd_parser import parse

    # シンプルな回路で Front Layer の動作を確認
    code = """
    QUBIT 0 AT (1, 0)
    QUBIT 1 AT (1, 1)
    QUBIT 2 AT (1, 2)

    # 独立したRESET (並列可能)
    RESET 0
    RESET 1
    RESET 2

    # 各qubitへのR gate
    R 0 1.57 0.0
    R 1 1.57 0.0
    R 2 1.57 0.0

    # 2-qubit gates
    MS 0 1 0.25
    MS 1 2 0.25
    """

    print("=" * 60)
    print("DAG Front Layer Test")
    print("=" * 60)

    program = parse(code)
    builder = DAGBuilder(program)
    dag = builder.build()

    print("\n--- DAG Structure ---")
    for nid in sorted(dag.nodes.keys()):
        node = dag.nodes[nid]
        op = node.operation
        targets = op.targets if hasattr(op, 'targets') else []
        print(f"  {nid}: {op.type} {targets}  pred={node.predecessors} succ={node.successors}")

    print("\n--- Front Layer Simulation ---")
    print("(Simulating scheduler selecting gates one by one)\n")

    step = 0
    while not dag.is_complete():
        front = dag.get_front_layer()
        print(f"Step {step}: Front Layer = {front}")

        # Front Layerの各ノードの詳細
        for nid in sorted(front):
            node = dag.nodes[nid]
            op = node.operation
            targets = op.targets if hasattr(op, 'targets') else []
            print(f"         -> {nid}: {op.type} on {targets}")

        # ここでSchedulerが最適なゲートを選ぶ
        # (今は単純に最小IDを選択)
        selected = min(front)
        print(f"         Selected: {selected}")

        dag.execute(selected)
        step += 1
        print()

    print(f"Execution complete! Total steps: {step}")
    print(f"Executed nodes: {sorted(dag.get_executed())}")

    # リセットして静的解析と比較
    print("\n--- Comparison with Static Parallel Groups ---")
    dag.reset_execution()
    groups = dag.get_parallel_groups()
    for i, group in enumerate(groups):
        ops = [f"{nid}:{dag.nodes[nid].operation.type}" for nid in group]
        print(f"  Level {i}: {ops}")