# Adapted from THeatA for the bundled workbench.
from lark import Lark, Transformer
from .ast_def import (
    Location, QubitDecl, Operation,
    GateOp, ShuttleOp, SwapOp, SplitOp, MergeOp,
    MeasureOp, ResetOp, BarrierOp, CoolOp, Program
)

# ==========================================
# EBNF (文法定義)
# ==========================================

qccd_grammar = """
    start: (declaration | instruction)+

    // --- 宣言部 (初期配置) ---
    declaration: "QUBIT" ion_id "AT" location

    // --- 命令部 ---
    instruction: gate_op | movement_op | measure_op | reset_op | barrier_op | cool_op | comment

    // 基本型
    ion_id:    INT
    trap_id:   INT
    site_id:   INT
    angle:     SIGNED_NUMBER
    location:  "(" trap_id "," site_id ")"

    // ゲート操作
    gate_op: ms_op | r_op
    ms_op: "MS" ion_id ion_id angle
    r_op:  "R"  ion_id angle angle

    // 移動操作
    movement_op: shuttle_op | split_op | merge_op | swap_op

    // SHUTTLE: 移動先のみ指定 (現在位置は状態から導出)
    shuttle_op: "SHUTTLE" ion_id "->" location

    // SWAP: 2つのion_idのみ (同一trap内にいることはsemantic analysisで検証)
    swap_op: "SWAP" ion_id ion_id

    // SPLIT: 指定イオンが右端になる位置で分割、右側に新しいzone_idを付与
    split_op: "SPLIT" ion_id "->" zone_id

    // MERGE: 左と右のゾーンを結合、左のIDを継承
    merge_op: "MERGE" zone_id zone_id

    zone_id: INT

    // その他の操作
    measure_op:  "MEASURE" ion_id+
    reset_op:    "RESET"   ion_id+
    barrier_op:  "BARRIER" ion_id+

    // COOL: zone_id 内のチェーン全体を冷却。coolant の ion_id は省略可
    cool_op:     "COOL" zone_id ion_id?

    comment: /#.*/

    %import common.INT
    %import common.SIGNED_NUMBER
    %import common.WS
    %import common.NEWLINE
    %ignore WS
    %ignore NEWLINE
"""


# ==========================================
# Transformer (Parse Tree -> AST)
# ==========================================

class QCCDASTTransformer(Transformer):
    """LarkのParse TreeをASTデータクラスに変換"""

    # --- トークン処理 ---
    def INT(self, token):
        return int(token)

    def SIGNED_NUMBER(self, token):
        return float(token)

    # --- 基本型 ---
    def ion_id(self, items):
        return items[0]

    def trap_id(self, items):
        return items[0]

    def site_id(self, items):
        return items[0]

    def angle(self, items):
        return items[0]

    def location(self, items):
        return Location(trap=items[0], site=items[1])

    # --- 宣言 ---
    def declaration(self, items):
        return QubitDecl(id=items[0], location=items[1])

    # --- ゲート操作 ---
    def ms_op(self, items):
        return GateOp(type="MS", targets=[items[0], items[1]], params=[items[2]])

    def r_op(self, items):
        return GateOp(type="R", targets=[items[0]], params=[items[1], items[2]])

    # --- 移動操作 ---
    def shuttle_op(self, items):
        # SHUTTLE ion_id -> location
        return ShuttleOp(target=items[0], dst=items[1])

    def swap_op(self, items):
        # SWAP ion_id ion_id
        return SwapOp(targets=[items[0], items[1]])

    def split_op(self, items):
        # SPLIT ion_id -> zone_id
        return SplitOp(ion_id=items[0], right_zone_id=items[1])

    def merge_op(self, items):
        # MERGE zone_left zone_right
        return MergeOp(zone_left=items[0], zone_right=items[1])

    def zone_id(self, items):
        return items[0]

    # --- その他の操作 ---
    def measure_op(self, items):
        return MeasureOp(targets=list(items))

    def reset_op(self, items):
        return ResetOp(targets=list(items))

    def barrier_op(self, items):
        return BarrierOp(targets=list(items))

    def cool_op(self, items):
        # COOL zone_id [coolant_ion_id]
        coolant = items[1] if len(items) > 1 else -1
        return CoolOp(zone_id=items[0], coolant_id=coolant)

    # --- パススルー ---
    def gate_op(self, items):
        return items[0]

    def movement_op(self, items):
        return items[0]

    def instruction(self, items):
        return items[0]

    def comment(self, items):
        return None

    # --- ルート ---
    def start(self, items):
        valid_items = [x for x in items if x is not None]
        decls = [x for x in valid_items if isinstance(x, QubitDecl)]
        insts = [x for x in valid_items if isinstance(x, Operation)]
        return Program(declarations=decls, instructions=insts)


# ==========================================
# パーサーインスタンス生成
# ==========================================

def create_parser():
    """パーサーとトランスフォーマーを生成"""
    parser = Lark(qccd_grammar, start='start')
    transformer = QCCDASTTransformer()
    return parser, transformer


def parse(code: str) -> Program:
    """コード文字列をパースしてProgramを返す"""
    parser, transformer = create_parser()
    tree = parser.parse(code)
    return transformer.transform(tree)


# ==========================================
# 動作確認
# ==========================================

if __name__ == "__main__":
    # サンプルコード（修正後の文法）
    code = """
    # Initial Placement
    QUBIT 0 AT (1, 0)
    QUBIT 1 AT (1, 1)
    QUBIT 2 AT (2, 0)

    # Reset and Gate Operations
    RESET 0 1
    R 0 1.57 0.0
    MS 0 1 0.25

    # Movement (新文法: srcを省略)
    SHUTTLE 2 -> (1, 2)

    # Swap (新文法: 位置指定なし)
    SWAP 0 1

    # Split/Merge (新仕様)
    SPLIT 1 -> 3
    MERGE 0 1

    # Measurement
    MEASURE 0 1
    """

    print("=" * 50)
    print("QCCD Parser Test (Modified Grammar)")
    print("=" * 50)
    print("\n--- Source Code ---")
    print(code)

    try:
        program = parse(code)

        print("\n--- Parsed Program ---")
        print(f"\nDeclarations ({len(program.declarations)}):")
        for decl in program.declarations:
            print(f"  {decl}")

        print(f"\nInitial positions: {program.get_initial_positions()}")

        print(f"\nInstructions ({len(program.instructions)}):")
        for i, inst in enumerate(program.instructions):
            print(f"  [{i}] {inst}")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()