from dataclasses import dataclass
from typing import List, Optional, Set

# ==========================================
# 基本型
# ==========================================

@dataclass(frozen=True)
class Location:
    """イオンの物理的位置を表す"""
    trap: int
    site: int

    def __repr__(self):
        return f"({self.trap}, {self.site})"


# ==========================================
# 宣言
# ==========================================

@dataclass
class QubitDecl:
    """量子ビットの初期配置宣言"""
    id: int
    location: Location


# ==========================================
# 操作の基底クラス
# ==========================================

@dataclass
class Operation:
    """全操作の基底クラス"""
    pass


# ==========================================
# ゲート操作
# ==========================================

@dataclass
class GateOp(Operation):
    """量子ゲート操作 (MS, R)"""
    type: str           # "MS" or "R"
    targets: List[int]  # ion_id のリスト
    params: List[float] # 角度パラメータ
    zone_id: int = -1   # 実行するLIZゾーン（MovementPlannerが設定）


# ==========================================
# 移動操作
# ==========================================

@dataclass
class ShuttleOp(Operation):
    """シャトル操作: イオンを別の位置へ移動

    dst_chain_length: Shuttle直前のdstトラップ内のイオン数
                     （Shuttle時間の計算に使用）
    """
    target: int         # 移動するion_id
    dst: Location       # 移動先 (srcは状態から導出)
    dst_chain_length: int = 0  # dstトラップのイオン数（コンパイラが設定）
    src_zone: int = -1  # シャトル中のTempゾーン（MovementPlannerが設定）
    type: str = "SHUTTLE"


@dataclass
class SwapOp(Operation):
    """スワップ操作: 同一trap内の2イオンの順序を入れ替え"""
    targets: List[int]  # 入れ替える2つのion_id
    zone_id: int = -1   # スワップが行われるゾーン（MovementPlannerが設定）
    type: str = "SWAP"


@dataclass
class SplitOp(Operation):
    """
    スプリット操作: イオンチェーンを分割し、片方をdst_zone_idへ移す

    指定したイオンが右端になる位置でチェーンを分割する。
    分割されたチェーンのうち、dst_zone_id側がdst_zone_idへ移動する。

    方向の決定（ゾーンIDの大小関係から自動推論）:
    - dst_zone_id > 元のゾーン → 右側チェーンがdstへ
    - dst_zone_id < 元のゾーン → 左側チェーンがdstへ

    ゾーンに1イオンのみの場合、そのイオンをdstへ移動する。

    例1: Zone1: [A, B, C, D] で Split(ion_id=B, dst_zone_id=2)
        → Zone1: [A, B], Zone2: [C, D]  (dst=2 > src=1 → 右チェーンがdstへ)

    例2: Zone1: [A, B, C, D] で Split(ion_id=B, dst_zone_id=0)
        → Zone0: [A, B], Zone1: [C, D]  (dst=0 < src=1 → 左チェーンがdstへ)
    """
    ion_id: int             # このイオンが右端になる位置で分割
    dst_zone_id: int        # 分割されたチェーンの移動先
    src_zone_id: int = -1   # 分割元ゾーン（MovementPlannerが設定）
    type: str = "SPLIT"


@dataclass
class MergeOp(Operation):
    """
    マージ操作: 2つの隣接イオンチェーンを結合

    結合順序は物理配置に従い、左側のイオンが常に前になる。
    結果はMemory/LIZゾーン側に保存され、Temp Zoneは空になる。
    （どちらが保存先かはゾーンタイプから自動決定される）

    例: Zone2(T): [C], Zone3(L): [D] で Merge(zone_left=2, zone_right=3)
        → Zone3: [C, D], Zone2: []  (LIZ側に保存)

    例: Zone0(T): [A], Zone1(M): [B,C] で Merge(zone_left=0, zone_right=1)
        → Zone1: [A, B, C], Zone0: []  (Memory側に保存)
    """
    zone_left: int       # 左側のゾーン
    zone_right: int      # 右側のゾーン
    type: str = "MERGE"


# ==========================================
# その他の操作
# ==========================================

@dataclass
class MeasureOp(Operation):
    """測定操作"""
    targets: List[int]  # 測定するion_idのリスト
    zone_id: int = -1   # 実行するLIZゾーン（MovementPlannerが設定）
    type: str = "MEASURE"


@dataclass
class ResetOp(Operation):
    """リセット操作"""
    targets: List[int]  # リセットするion_idのリスト
    zone_id: int = -1   # 実行するLIZゾーン（MovementPlannerが設定）
    type: str = "RESET"


@dataclass
class CoolOp(Operation):
    """冷却操作: zone_id 内のイオンチェーン全体を残留フォノン数まで冷やす

    Cool(trap_zone_id, coolant_id) 形式。coolant_id は sympathetic cooling
    用の補助イオン。-1 は「coolant をモデル化しない」暫定モード
    （詳細仕様の確定までのデフォルト）。coolant_id >= 0 の場合、
    実行時に coolant が zone_id に居ることを要求する。

    duration_us: この命令固有の冷却時間。None なら TimeParameters.cool
    （共通デフォルト）を使う。用途例: アンシラ測定後の基底状態への
    深い再冷却 (2000μs) と、ゲート前の軽い冷却 (400μs) の使い分け。
    """
    zone_id: int
    coolant_id: int = -1
    type: str = "COOL"
    duration_us: Optional[float] = None


@dataclass
class TimeStep:
    """並列実行可能な操作の集合（Schedulerが生成）"""
    ops: List[Operation]
    type: str = "TIMESTEP"


@dataclass
class BarrierOp(Operation):
    """バリア操作: 同期点"""
    targets: List[int]  # 対象のion_idのリスト
    type: str = "BARRIER"


# ==========================================
# プログラム全体
# ==========================================

@dataclass
class Program:
    """パース結果のプログラム全体"""
    declarations: List[QubitDecl]
    instructions: List[Operation]

    def get_all_ion_ids(self) -> Set[int]:
        """宣言された全ion_idを取得"""
        return {decl.id for decl in self.declarations}

    def get_initial_positions(self) -> dict:
        """ion_id -> 初期位置 のマッピングを取得"""
        return {decl.id: decl.location for decl in self.declarations}