# Adapted from THeatA for the bundled workbench.
"""
QCCD Compiler: DAG → Parallel Executable QCCD Native Circuit

アーキテクチャ:
  DAG (ゲートのみ)
       ↓
  [MovementPlanner]  初期配置 + DAG → 平坦なAtomicOp列
       ↓              (Split/Swap/Merge/Gate、ゾーンアノテーション付き)
  [Scheduler]        AtomicOp列 → TimeStep列（ASAP並列化）
       ↓
  List[TimeStep]

各クラスの責務:
  MovementPlanner: 量子回路の実行順を守りながら移動操作をインライン展開して出力
  Scheduler:       ゾーン競合ルールのみを見てTimeStepに割り当て（配置を知らない）
"""

from dataclasses import dataclass, field, asdict, replace
from typing import List, Dict, Set, Optional, Tuple
from .ast_def import (
    Location, QubitDecl, Operation, TimeStep,
    GateOp, ShuttleOp, SwapOp, SplitOp, MergeOp, MeasureOp, ResetOp, CoolOp,
    Program
)
from .dag_builder import DAG, DAGNode
from .stabilizer_partitioner import StabilizerPartition
from .sim_params import phonon_params_from
from .estimator import Estimator


# ==========================================
# Compiler Configuration
# ==========================================

@dataclass
class CompilerConfig:
    """
    コンパイラ最適化設定

    Attributes:
        gate_selection: ゲート選択戦略
            - "min_id": Front Layerから最小IDを選択（baseline）
        memory_reorder: Measure/Reset中にMemory内イオンを先読み整列するか
        measure_batch_size: Measureのバッチサイズ
            - 1: 即時測定（デフォルト、従来動作）
            - 2+: 指定数のアンシラが測定待ちになるまで遅延し一括測定
            - LIZ容量を超える値は自動的にLIZ容量にクランプされる
        eviction_strategy: LIZからの退避戦略
            - "balanced": アンシラの左右で振り分け（複数ラウンド向き、配置バランス保持）
            - "nearest": 最寄りMemoryへBFS（1ラウンド最小ops）
        cool_threshold_transversal: 閾値ベース自動冷却 (None で無効)
            - MS 実行直前にゲート対象チェーンの transversal フォノンが
              この値を超えていたら、LIZ に CoolOp を挿入してから MS を実行する
            - 「ゲート時フォノンの最大値を閾値以下に抑える」ポリシー。
              エラーモデル (PauliErrorLookup) が使うのは transversal のため
              transversal のみ監視する
        cool_ancilla_after_measure_us: アンシラ測定後冷却 (None で無効)
            - 各 MEASURE の直後、アンシラの居る LIZ に指定時間 (μs) の
              CoolOp を挿入し、フォノンを残留値 (デフォルト 0) に戻す
            - 測定時はアンシラ以外が退避済みのため実質アンシラのみ冷える。
              基底状態への深い再冷却の想定 (例: 2000μs)
        cool_ancilla_measure_interval: アンシラ測定後冷却の周期 (デフォルト 1)
            - N を指定すると MEASURE N 回につき 1 回だけ冷却する
              (1 = 毎回 = 従来動作)。カウンタは compile 単位でリセット
            - cool_ancilla_after_measure_us が None のときは無視される
        merge_cool_threshold_transversal: merge 平均化冷却 (None で無効)
            - 各 MEASURE 直後 (アンシラが LIZ に居る時点) に、LIZ に隣接する
              memory zone (間に他の LIZ を挟まないもの) のチェーン平均
              transversal フォノンがこの値を超えていたら:
              直接冷却したアンシラを split→shuttle→merge で当該 memory へ
              合流させ、merge のフォノン平均化でチェーンを冷やし、LIZ へ戻す
            - 左右両方が閾値超なら両側に順に行う。アンシラの冷却命令は
              側ごとの合流前 + 帰還後の再冷却 (吸った熱を捨て、次の
              スタビライザ測定を温まったアンシラで始めない) に挿入される
            - 1 回の合流で除去できるのは超過分の約 1/(k+1) (k = チェーン長)
              のため、閾値超が続く限り以降の MEASURE イベントでも繰り返し
              発動する
        predictive_cool_threshold_transversal: 予測型 merge 冷却 (None で無効)
            - 2026-10-02 会議プロトコル。2-pass コンパイル:
              pass 1 (本トリガのみ無効) で測定窓ごとの per-ion 増分 Δn̄ を
              記録し、pass 2 の各 MEASURE 直後 (= 次の生成元の測定開始前) に
                予測値 = 現在値 + 次窓の Δ (pass 1)
              が θ を超えるデータ量子ビットがあれば、その居る memory へ
              merge 平均化冷却エピソード (冷 anc 合流 → 帰還 → 再冷却) を挿入
            - 予測は「現在値 + 窓増分」の合成なので、窓内 op 列が pass 間で
              同一である限り時間シフトに対して厳密。窓境界の整合は
              測定イベント列 (ancilla 順) の一致で検証し、不一致時は
              そのイベントの予測をスキップして警告する
        predictive_cool_max_iters: 1 チェックポイントあたりのエピソード上限
            (デフォルト 1)。予測が θ 超のままなら上限まで繰り返す
            («複数回挟む» の粒度ノブ)
        resident_coolant_ids: LIZ 常駐 coolant のイオン ID リスト (None で無効)
            - Resident coolant configuration: LIZ を coolant + 測定アンシラで
              埋め、データ量子ビットがゲートのために LIZ へ来るたびに
              冷たい常駐チェーンとの merge 平均化で冷える (受動冷却)。
              1 訪問の除去率 = m/(m+1) (m = 常駐冷イオン数) で
              チェーン長に依存しない
            - 指定イオンは回路外 (DAG に現れない)。初期配置で LIZ に置き、
              全ての退避 (測定前退避・容量退避) から保護される
            - 仮定: 測定・リセット中も coolant は LIZ に留まれる
              (coolant は量子情報を持たないため検出光散乱は無害とみなす)
        coolant_recool_us: 補助量子ビット (coolant + アンシラ) の一斉再冷却
            (None で無効)。データが LIZ から退避され LIZ が補助のみに
            なった時点で、残留値超なら指定時間の CoolOp を挿入する。
            「補助量子ビットの冷却は一斉にできる」仮定の実装
        predictive_cool_actuator: 予測型トリガのアクチュエータ (2026-10-07 設計合意)
            - "ancilla" (既定): 測定アンシラを熱シャトルとして memory チェーンへ
              往復させるエピソード (チェーン一括冷却)
            - "data": 予測超過データイオン自身を 1 個ずつ LIZ へ搬送し、
              冷たい常駐チェーン (coolant + アンシラ) との merge 平均化で
              冷やして元の memory ゾーンへ戻す (per-ion 冷却)。
              合意事項: 帰還はチェーン端 (順序は保存しない)、冷却訪問と
              ゲート訪問は別往復、反復なし (1 訪問で除去率 m/(m+1))。
              訪問間の補助再冷却は coolant_recool_us を使う
    """
    gate_selection: str = "min_id"
    memory_reorder: bool = False
    measure_batch_size: int = 1
    interleave: bool = False
    eviction_strategy: str = "balanced"
    cool_threshold_transversal: Optional[float] = None
    cool_ancilla_after_measure_us: Optional[float] = None
    cool_ancilla_measure_interval: int = 1
    merge_cool_threshold_transversal: Optional[float] = None
    predictive_cool_threshold_transversal: Optional[float] = None
    predictive_cool_max_iters: int = 1
    resident_coolant_ids: Optional[List[int]] = None
    coolant_recool_us: Optional[float] = None
    predictive_cool_actuator: str = "ancilla"
    # 窓内の合法再冷却 (2026-10-25 著者案): 訪問間再冷却がもつれ保持中の
    # アンシラでブロックされるとき、アンシラを memory へ一時退避して
    # coolant のみ冷却し呼び戻す (輸送コストを払って冷却規則に適合させる)
    isolate_ancilla_recool: bool = False
    # 隔離サイクル (退避→Cool→帰還) の反復数 (2026-10-26)。各サイクルは
    # アンシラを熱シャトルとした memory チェーンの exchange 冷却 1 回に相当。
    # 利得はチェーン平均の低下 (除去率 ~1/(チェーン長+1)/回) で逓減し、
    # コストは CoolOp + 往復輸送 + per-ion merge コストが毎回乗る
    isolate_recool_iters: int = 1
    # LIZ predictive trigger (2026-10-26 会議)。CX (MS) 直後にゲート対象
    # データの transversal フォノンを検査し、閾値超なら:
    #   左側 Memory のチェーンを外側 Temp へ一括退避 (SplitOp 1 発)
    #   → (A, d_i) を Memory へ隔離 → coolant のみの LIZ を冷却
    #   → (A, d_i) 帰還 (冷えた coolant との merge で ≈1/4 に希釈)
    #   → 閾値超なら反復 (最大 liz_pt_max_iters) → チェーン復帰。
    # 隔離中 (A, d_i) は専用 2 イオンチェーンなので反復で幾何的に冷える。
    # coolant 必須 (居なければ発動しない)。仮定: Temp への長期滞在可、
    # 空ゾーンへの merge もコスト計上 (2026-10-27、実機確認待ちの安全側)
    liz_pt_threshold_transversal: Optional[float] = None
    liz_pt_max_iters: int = 1
    # ヒステリシス化 (2026-10-27): 発火閾値 (threshold) と到達目標 (target) の
    # 分離。None = 従来どおり threshold まで冷やす。target < threshold に
    # 設定すると「発火頻度は threshold 並み、ゲート品質は target 並み」を狙える
    liz_pt_target_transversal: Optional[float] = None
    # 隔離先 (2026-10-27): "memory" = 10/26 会議設計 (チェーンを外側 Temp へ
    # 退避して M を専有。エピソードごとにチェーン全員へ split+merge コストが
    # 乗り、チェーンが長いと自滅することが初計測で判明)。
    # "temp" = LIZ 隣接 Temp にペアを退避 (チェーン無接触・最小コスト。
    # ただし冷却中の隣接 Temp 滞在の可否は実機確認待ち)
    liz_pt_park: str = "memory"
    # planner v2 (per-zone state machine + JIT staging, 2026-09-08 設計):
    #   planner: "v1" = 従来 MovementPlanner / "v2" = StateMachinePlanner
    #   prefetch_depth: 測定窓中に次需要イオンを memory→corridor temp へ
    #     先行切り出しする深さ (0 = 厳密 JIT, 1 = 1 イオン先読み)
    #   v2_sort_horizon: 測定窓中に各 memory zone で納品順へ先行整列する
    #     需要イオン数の上限 (LIZ 端から horizon 枠)
    #   v2_sort_trigger: 整列を emit するタイミング
    #     "measure"  = MEASURE 直後 (測定窓に重ねる。needed[1] 以降は
    #                  edge-1 止め = 中間パターンの加熱オーバーヘッドあり)
    #     "delivery" = 納品完了直後 (端が空いた瞬間に次需要イオンを真端まで
    #                  整列。v1 と同一 swap 列の前倒し = 加熱中立が狙い)
    #     "both"     = 両方
    planner: str = "v1"
    prefetch_depth: int = 1
    v2_sort_horizon: int = 4
    v2_sort_trigger: str = "measure"

    def to_dict(self) -> dict:
        return asdict(self)

    def __repr__(self) -> str:
        parts = [f"gate={self.gate_selection}"]
        if self.measure_batch_size > 1:
            parts.append(f"batch={self.measure_batch_size}")
        if self.interleave:
            parts.append("interleave")
        return f"CompilerConfig({', '.join(parts)})"

    @classmethod
    def baseline(cls) -> "CompilerConfig":
        return cls()

    def validate(self):
        valid_gate = {"min_id"}
        if self.gate_selection not in valid_gate:
            raise ValueError(f"Invalid gate_selection: {self.gate_selection}. Valid: {valid_gate}")
        if not isinstance(self.memory_reorder, bool):
            raise ValueError(f"memory_reorder must be bool, got {type(self.memory_reorder)}")
        valid_eviction = {"balanced", "nearest"}
        if self.eviction_strategy not in valid_eviction:
            raise ValueError(f"Invalid eviction_strategy: {self.eviction_strategy}. Valid: {valid_eviction}")
        if not isinstance(self.measure_batch_size, int) or self.measure_batch_size < 1:
            raise ValueError(f"measure_batch_size must be int >= 1, got {self.measure_batch_size}")
        if self.cool_threshold_transversal is not None and self.cool_threshold_transversal < 0:
            raise ValueError(
                f"cool_threshold_transversal must be >= 0 or None, "
                f"got {self.cool_threshold_transversal}")
        if (self.cool_ancilla_after_measure_us is not None
                and self.cool_ancilla_after_measure_us <= 0):
            raise ValueError(
                f"cool_ancilla_after_measure_us must be > 0 or None, "
                f"got {self.cool_ancilla_after_measure_us}")
        if (not isinstance(self.cool_ancilla_measure_interval, int)
                or self.cool_ancilla_measure_interval < 1):
            raise ValueError(
                f"cool_ancilla_measure_interval must be int >= 1, "
                f"got {self.cool_ancilla_measure_interval}")
        if (self.merge_cool_threshold_transversal is not None
                and self.merge_cool_threshold_transversal < 0):
            raise ValueError(
                f"merge_cool_threshold_transversal must be >= 0 or None, "
                f"got {self.merge_cool_threshold_transversal}")
        if (self.predictive_cool_threshold_transversal is not None
                and self.predictive_cool_threshold_transversal < 0):
            raise ValueError(
                f"predictive_cool_threshold_transversal must be >= 0 or None, "
                f"got {self.predictive_cool_threshold_transversal}")
        if (not isinstance(self.predictive_cool_max_iters, int)
                or self.predictive_cool_max_iters < 1):
            raise ValueError(
                f"predictive_cool_max_iters must be int >= 1, "
                f"got {self.predictive_cool_max_iters}")
        if self.resident_coolant_ids is not None:
            if (not isinstance(self.resident_coolant_ids, (list, tuple))
                    or not all(isinstance(i, int)
                               for i in self.resident_coolant_ids)):
                raise ValueError(
                    f"resident_coolant_ids must be a list of int or None, "
                    f"got {self.resident_coolant_ids}")
        if self.coolant_recool_us is not None and self.coolant_recool_us <= 0:
            raise ValueError(
                f"coolant_recool_us must be > 0 or None, "
                f"got {self.coolant_recool_us}")
        if (self.liz_pt_threshold_transversal is not None
                and self.liz_pt_threshold_transversal < 0):
            raise ValueError(
                f"liz_pt_threshold_transversal must be >= 0 or None, "
                f"got {self.liz_pt_threshold_transversal}")
        if (not isinstance(self.liz_pt_max_iters, int)
                or self.liz_pt_max_iters < 1):
            raise ValueError(
                f"liz_pt_max_iters must be int >= 1, "
                f"got {self.liz_pt_max_iters}")
        if self.liz_pt_target_transversal is not None:
            if self.liz_pt_target_transversal < 0:
                raise ValueError(
                    f"liz_pt_target_transversal must be >= 0 or None, "
                    f"got {self.liz_pt_target_transversal}")
            if (self.liz_pt_threshold_transversal is not None
                    and self.liz_pt_target_transversal
                    > self.liz_pt_threshold_transversal):
                raise ValueError(
                    "liz_pt_target_transversal must not exceed "
                    "liz_pt_threshold_transversal")
        if self.liz_pt_park not in {"memory", "temp", "home"}:
            raise ValueError(
                f"liz_pt_park must be 'memory', 'temp' or 'home', "
                f"got {self.liz_pt_park}")
        if self.predictive_cool_actuator not in {"ancilla", "data"}:
            raise ValueError(
                f"predictive_cool_actuator must be 'ancilla' or 'data', "
                f"got {self.predictive_cool_actuator}")
        if self.planner not in {"v1", "v2"}:
            raise ValueError(f"planner must be 'v1' or 'v2', got {self.planner}")
        if self.prefetch_depth not in (0, 1):
            raise ValueError(
                f"prefetch_depth must be 0 or 1, got {self.prefetch_depth}")
        if not isinstance(self.v2_sort_horizon, int) or self.v2_sort_horizon < 0:
            raise ValueError(
                f"v2_sort_horizon must be int >= 0, got {self.v2_sort_horizon}")
        if self.v2_sort_trigger not in {"measure", "delivery", "both"}:
            raise ValueError(
                f"v2_sort_trigger must be 'measure', 'delivery' or 'both', "
                f"got {self.v2_sort_trigger}")


# ==========================================
# QCCD Architecture
# ==========================================

@dataclass
class Zone:
    """ゾーンの定義"""
    id: int
    zone_type: str   # "L" (LIZ), "M" (Memory), "T" (Temporary)
    capacity: int
    position: int


class QCCDArchitecture:
    """
    1D QCCDアーキテクチャの構成定義

    物理的な電極配置に基づく7ゾーン構成:
      T(20) - M(20) - T(20) - L(4) - T(20) - M(20) - T(20)

    入力形式: [(type, capacity), ...]
    """

    def __init__(self, zone_specs: List[Tuple[str, int]]):
        self.zones: List[Zone] = []
        for i, (zone_type, capacity) in enumerate(zone_specs):
            if zone_type not in ("M", "L", "T"):
                raise ValueError(f"Invalid zone type '{zone_type}'. Must be 'M', 'L', or 'T'.")
            self.zones.append(Zone(id=i, zone_type=zone_type, capacity=capacity, position=i))

        self._zone_by_id: Dict[int, Zone] = {z.id: z for z in self.zones}
        self._liz_zones: List[Zone] = [z for z in self.zones if z.zone_type == "L"]
        self._memory_zones: List[Zone] = [z for z in self.zones if z.zone_type == "M"]
        self._temp_zones: List[Zone] = [z for z in self.zones if z.zone_type == "T"]

        if not self._liz_zones:
            raise ValueError("Architecture must have at least one LIZ")

    def get_zone(self, zone_id: int) -> Optional[Zone]:
        return self._zone_by_id.get(zone_id)

    def get_liz_zones(self) -> List[Zone]:
        return self._liz_zones

    def get_memory_zones(self) -> List[Zone]:
        return self._memory_zones

    def get_temp_zones(self) -> List[Zone]:
        return self._temp_zones

    def get_adjacent_zones(self, zone_id: int) -> List[int]:
        adjacent = []
        if zone_id > 0:
            adjacent.append(zone_id - 1)
        if zone_id < len(self.zones) - 1:
            adjacent.append(zone_id + 1)
        return adjacent

    def is_liz(self, zone_id: int) -> bool:
        zone = self.get_zone(zone_id)
        return zone is not None and zone.zone_type == "L"

    def is_temp(self, zone_id: int) -> bool:
        zone = self.get_zone(zone_id)
        return zone is not None and zone.zone_type == "T"

    def is_memory(self, zone_id: int) -> bool:
        zone = self.get_zone(zone_id)
        return zone is not None and zone.zone_type == "M"

    def get_adjacent_temp(self, zone_id: int, direction: str) -> Optional[int]:
        """
        指定ゾーンの左("left")または右("right")の隣接Temporaryゾーンを取得

        例: T(0)-M(1)-T(2)-L(3)-T(4)-M(5)-T(6)
            get_adjacent_temp(1, "left")  → 0
            get_adjacent_temp(3, "left")  → 2
        """
        adj_id = zone_id - 1 if direction == "left" else zone_id + 1
        if adj_id < 0 or adj_id >= len(self.zones):
            return None
        if self.zones[adj_id].zone_type == "T":
            return adj_id
        return None

    def get_path(self, src_zone: int, dst_zone: int) -> List[int]:
        """srcからdstへの経路（両端含む）"""
        if src_zone <= dst_zone:
            return list(range(src_zone, dst_zone + 1))
        else:
            return list(range(src_zone, dst_zone - 1, -1))

    def find_nearest_memory(self, zone_id: int) -> Optional[int]:
        visited = {zone_id}
        queue = [zone_id]
        while queue:
            current = queue.pop(0)
            zone = self.get_zone(current)
            if zone and zone.zone_type == "M":
                return current
            for adj in self.get_adjacent_zones(current):
                if adj not in visited:
                    visited.add(adj)
                    queue.append(adj)
        return None

    def __repr__(self):
        layout = " - ".join(f"{z.zone_type}({z.capacity})" for z in self.zones)
        return f"QCCDArchitecture([{layout}])"

    @classmethod
    def simple_mlm(cls, liz_capacity: int = 4, memory_capacity: int = 20,
                   temp_capacity: int = 20):
        """T-M-T-L-T-M-T 構成"""
        return cls([
            ("T", temp_capacity),    # Zone 0
            ("M", memory_capacity),  # Zone 1
            ("T", temp_capacity),    # Zone 2
            ("L", liz_capacity),     # Zone 3
            ("T", temp_capacity),    # Zone 4
            ("M", memory_capacity),  # Zone 5
            ("T", temp_capacity),    # Zone 6
        ])

    @classmethod
    def chained_mlm(cls, num_liz: int, liz_capacity: int = 4,
                    memory_capacity: int = 20, temp_capacity: int = 20):
        """T-M-T-(L-T-M-T)*num_liz 構成

        num_liz=1 で simple_mlm と同等。
        num_liz=2 で T-M-T-L-T-M-T-L-T-M-T （複数 LIZ ベースケース）。
        """
        if num_liz < 1:
            raise ValueError(f"num_liz must be >= 1, got {num_liz}")
        zones = [
            ("T", temp_capacity),
            ("M", memory_capacity),
            ("T", temp_capacity),
        ]
        for _ in range(num_liz):
            zones.extend([
                ("L", liz_capacity),
                ("T", temp_capacity),
                ("M", memory_capacity),
                ("T", temp_capacity),
            ])
        return cls(zones)


# ==========================================
# QCCD State
# ==========================================

@dataclass
class QCCDState:
    """
    QCCDの実行時状態

    不変条件:
    - Temporary Zoneは操作前後で空であること
    - move_ionは隣接ゾーン間のみ
    - SwapはTemporary Zoneでは実行不可

    フォノン追跡:
    - 各イオンの 3 モード (axial 1 本 + transverse 2 本) のフォノン数を管理
      (2026-10-02 会議: 1 イオンあたり 3 パラメータ)。
      phonon_transversal = 第 1 transverse モード (誤り参照モード: MS LUT・
      閾値ポリシーが参照)、phonon_transversal2 = 第 2 transverse モード。
    - 同一チェイン内のイオンは同じフォノン数を持つ（不変条件）
    - phonon_params が None の場合はフォノン追跡を無効化
    """
    architecture: QCCDArchitecture
    ion_locations: Dict[int, int] = field(default_factory=dict)   # ion_id -> zone_id
    zone_contents: Dict[int, List[int]] = field(default_factory=dict)  # zone_id -> [ion_ids]
    phonon_axial: Dict[int, float] = field(default_factory=dict)       # ion_id -> axial phonon
    phonon_transversal: Dict[int, float] = field(default_factory=dict) # ion_id -> transverse mode 1
    phonon_transversal2: Dict[int, float] = field(default_factory=dict)  # ion_id -> transverse mode 2
    phonon_params: Optional[dict] = None  # フォノンパラメータ（Noneで無効化）
    # 計画時近似時計 (μs): emit した op の duration の直列和。
    # 直列和 ≥ 実時間 (並列スケジュール後) なので background heating は
    # 過大側の近似になる (1-LIZ では差は小、multi-LIZ ではより保守的)。
    clock_us: float = 0.0
    # MSゲート実行時のフォノン記録
    phonon_at_gates: List[dict] = field(default_factory=list)
    # 退避先の home 指定 (per-LIZ stream compile 用): ion_id -> home zone_id。
    # None なら従来動作（最寄り Memory への BFS）。
    home_map: Optional[Dict[int, int]] = None

    def __post_init__(self):
        for zone in self.architecture.zones:
            if zone.id not in self.zone_contents:
                self.zone_contents[zone.id] = []

    def _init_phonon(self, ion_id: int):
        """イオンのフォノン数を初期化"""
        if ion_id not in self.phonon_axial:
            self.phonon_axial[ion_id] = 0.0
            self.phonon_transversal[ion_id] = 0.0
            self.phonon_transversal2[ion_id] = 0.0

    def place_ion(self, ion_id: int, zone_id: int):
        """イオンをMemory Zoneに配置（初期配置用、Memory限定）"""
        zone = self.architecture.get_zone(zone_id)
        if zone is None:
            raise ValueError(f"Zone {zone_id} does not exist")
        if zone.zone_type != "M":
            raise ValueError(
                f"Initial placement only allowed in Memory zones. "
                f"Zone {zone_id} is type '{zone.zone_type}'."
            )
        if len(self.zone_contents[zone_id]) >= zone.capacity:
            raise ValueError(f"Zone {zone_id} is at capacity")
        self.ion_locations[ion_id] = zone_id
        self.zone_contents[zone_id].append(ion_id)
        self._init_phonon(ion_id)

    def place_ion_any(self, ion_id: int, zone_id: int):
        """イオンを任意ゾーンに配置（Memory/LIZ/Temp対応、初期配置用）"""
        zone = self.architecture.get_zone(zone_id)
        if zone is None:
            raise ValueError(f"Zone {zone_id} does not exist")
        if len(self.zone_contents[zone_id]) >= zone.capacity:
            raise ValueError(f"Zone {zone_id} is at capacity")
        self.ion_locations[ion_id] = zone_id
        self.zone_contents[zone_id].append(ion_id)
        self._init_phonon(ion_id)

    def locate(self, ion_id: int) -> int:
        if ion_id not in self.ion_locations:
            raise ValueError(f"Ion {ion_id} not found")
        return self.ion_locations[ion_id]

    def is_in_liz(self, ion_id: int) -> bool:
        return self.architecture.is_liz(self.locate(ion_id))

    def can_execute(self, gate: GateOp) -> bool:
        """ゲート実行可能か: 全イオンが同じLIZにいる"""
        if not gate.targets:
            return True
        first_zone = self.locate(gate.targets[0])
        if not self.architecture.is_liz(first_zone):
            return False
        return all(self.locate(ion_id) == first_zone for ion_id in gate.targets)

    def can_execute_reset_measure(self, op, extra_allowed: Set[int] = None) -> bool:
        """RESET/MEASURE実行可能か: 全ターゲットが同じLIZ、かつそのLIZに
        ターゲット以外がいない。

        extra_allowed (常駐 coolant 等) に含まれるイオンは同居を許す。
        coolant は量子情報を持たないため検出光の散乱が無害、という
        抽象化 (resident_coolant_ids の仮定)。
        """
        targets = op.targets
        if not targets:
            return True
        first_zone = self.locate(targets[0])
        if not self.architecture.is_liz(first_zone):
            return False
        if not all(self.locate(t) == first_zone for t in targets):
            return False
        liz_contents = set(self.zone_contents[first_zone])
        allowed = set(targets) | (extra_allowed or set())
        return liz_contents <= allowed

    def get_liz_with_space(self) -> Optional[int]:
        for liz in self.architecture.get_liz_zones():
            if len(self.zone_contents[liz.id]) < liz.capacity:
                return liz.id
        return None

    def get_liz_ions(self, liz_id: Optional[int] = None) -> List[int]:
        if liz_id is not None:
            return self.zone_contents[liz_id].copy()
        ions = []
        for liz in self.architecture.get_liz_zones():
            ions.extend(self.zone_contents[liz.id])
        return ions

    def get_zone_free_space(self, zone_id: int) -> int:
        zone = self.architecture.get_zone(zone_id)
        if zone is None:
            return 0
        return zone.capacity - len(self.zone_contents[zone_id])

    def find_eviction_destination(self, from_zone_id: int, ion_id: int = None) -> Optional[int]:
        """空きのある最寄りのMemoryゾーンを探す（home_map があれば home 優先）"""
        if ion_id is not None and self.home_map:
            home = self.home_map.get(ion_id)
            if (home is not None and home != from_zone_id
                    and self.get_zone_free_space(home) > 0):
                return home
        return self._find_memory_in_direction(from_zone_id, prefer_left=True)

    def _find_memory_in_direction(self, from_zone_id: int, prefer_left: bool = True) -> Optional[int]:
        """指定方向を優先してMemoryゾーンを探す"""
        if prefer_left:
            search_order_fn = lambda zid: self.architecture.get_adjacent_zones(zid)
        else:
            search_order_fn = lambda zid: list(reversed(self.architecture.get_adjacent_zones(zid)))

        visited = {from_zone_id}
        queue = [from_zone_id]
        while queue:
            current = queue.pop(0)
            zone = self.architecture.get_zone(current)
            if zone and zone.zone_type == "M" and self.get_zone_free_space(current) > 0:
                return current
            for adj in search_order_fn(current):
                if adj not in visited:
                    visited.add(adj)
                    queue.append(adj)
        return None

    def plan_evictions(self, zone_id: int, ions_to_evict: list, protected: set = None) -> list:
        """
        複数イオンの退避先をprotected（アンシラ）の左右で振り分ける

        LIZ内の配置は自然に [z1由来, アンシラ, z5由来] となっているため、
        アンシラより左のイオンは左方向(zone_idが小さいMemory)へ、
        アンシラより右のイオンは右方向(zone_idが大きいMemory)へ退避する。
        これによりLIZ内でのSwapが不要になる。

        protectedがない場合は左右交互にフォールバック。

        Returns:
            [(ion_id, evict_destination_zone_id), ...]
            退避順序: 左端→右端→左端→... の順（両端から内側へ）
        """
        if protected is None:
            protected = set()

        contents = list(self.zone_contents.get(zone_id, []))
        evict_set = set(ions_to_evict)

        if self.home_map:
            # home 優先モード (per-LIZ stream compile 用):
            # 両端から内側の順で退避し、各イオンは home に空きがあれば home へ、
            # なければ位置に近い側の Memory へ。core 外の Memory にイオンが
            # 流出して pass-through や cross-core 競合を生むのを防ぐ。
            in_pos = [ion for ion in contents if ion in evict_set]
            order = []
            lo, hi = 0, len(in_pos) - 1
            take_left = True
            while lo <= hi:
                if take_left:
                    order.append(in_pos[lo]); lo += 1
                else:
                    order.append(in_pos[hi]); hi -= 1
                take_left = not take_left
            reserved: Dict[int, int] = {}
            plan = []
            for ion in order:
                home = self.home_map.get(ion)
                dst = None
                if home is not None and home != zone_id:
                    if self.get_zone_free_space(home) - reserved.get(home, 0) > 0:
                        dst = home
                if dst is None:
                    prefer_left = contents.index(ion) < len(contents) / 2
                    dst = self._find_memory_in_direction(zone_id, prefer_left=prefer_left)
                if dst is not None:
                    plan.append((ion, dst))
                    reserved[dst] = reserved.get(dst, 0) + 1
            return plan

        # protectedイオンの位置を特定
        protected_positions = [i for i, ion in enumerate(contents) if ion in protected]

        if protected_positions:
            # protectedの最小位置より左 → 左方向へ退避
            # protectedの最大位置より右 → 右方向へ退避
            protected_min = min(protected_positions)
            protected_max = max(protected_positions)
        else:
            # protectedがなければ退避先の空き容量で振り分ける
            # 左Memory と右Memory の空きを比較して均等に分配
            left_mem = self._find_memory_in_direction(zone_id, prefer_left=True)
            right_mem = self._find_memory_in_direction(zone_id, prefer_left=False)
            left_free = self.get_zone_free_space(left_mem) if left_mem is not None else 0
            right_free = self.get_zone_free_space(right_mem) if right_mem is not None else 0

            plan = []
            evict_list = [ion for ion in contents if ion in evict_set]
            for ion in evict_list:
                if left_free > right_free:
                    plan.append((ion, left_mem))
                    left_free -= 1
                elif right_free > 0:
                    plan.append((ion, right_mem))
                    right_free -= 1
                else:
                    plan.append((ion, left_mem))
                    left_free -= 1
            return plan

        # 左側イオン（左端から順に）と右側イオン（右端から順に）に分類
        left_ions = []   # 左方向へ退避（左端から内側へ）
        right_ions = []  # 右方向へ退避（右端から内側へ）

        for i, ion in enumerate(contents):
            if ion not in evict_set:
                continue
            if i < protected_min:
                left_ions.append(ion)
            elif i > protected_max:
                right_ions.append(ion)
            else:
                # protectedの間にいる → 近い方へ
                if i - protected_min <= protected_max - i:
                    left_ions.append(ion)
                else:
                    right_ions.append(ion)

        # 退避計画を構築（両端から交互に実行）
        plan = []
        # 左側は左端から（リスト順 = 左端から内側へ）
        for ion in left_ions:
            dst = self._find_memory_in_direction(zone_id, prefer_left=True)
            if dst is not None:
                plan.append((ion, dst))

        # 右側は右端から（リスト逆順 = 右端から内側へ）
        for ion in reversed(right_ions):
            dst = self._find_memory_in_direction(zone_id, prefer_left=False)
            if dst is not None:
                plan.append((ion, dst))

        return plan

    def move_ion(self, ion_id: int, dst_zone_id: int):
        """イオンを隣接ゾーンへ移動"""
        src_zone_id = self.locate(ion_id)
        dst_zone = self.architecture.get_zone(dst_zone_id)
        if dst_zone is None:
            raise ValueError(f"Destination zone {dst_zone_id} does not exist")
        if abs(src_zone_id - dst_zone_id) != 1:
            raise ValueError(f"move_ion only allows adjacent zones. src={src_zone_id}, dst={dst_zone_id}")
        if len(self.zone_contents[dst_zone_id]) >= dst_zone.capacity:
            raise ValueError(f"Destination zone {dst_zone_id} is at capacity")
        self.zone_contents[src_zone_id].remove(ion_id)
        if src_zone_id < dst_zone_id:
            self.zone_contents[dst_zone_id].insert(0, ion_id)
        else:
            self.zone_contents[dst_zone_id].append(ion_id)
        self.ion_locations[ion_id] = dst_zone_id

    def swap_ions(self, ion_a: int, ion_b: int):
        """同一ゾーン内の隣接イオンをスワップ"""
        zone_a = self.locate(ion_a)
        zone_b = self.locate(ion_b)
        if zone_a != zone_b:
            raise ValueError(f"Cannot swap ions in different zones")
        if self.architecture.is_temp(zone_a):
            raise ValueError(f"Cannot swap ions in Temporary zone {zone_a}")
        contents = self.zone_contents[zone_a]
        pos_a = contents.index(ion_a)
        pos_b = contents.index(ion_b)
        if abs(pos_a - pos_b) != 1:
            raise ValueError(f"Cannot swap non-adjacent ions")
        contents[pos_a], contents[pos_b] = contents[pos_b], contents[pos_a]
        # フォノン: +0.1 (axial, transversal)
        if self.phonon_params:
            ax = self.phonon_params.get('swap_axial', 0.1)
            tr = self.phonon_params.get('swap_transversal', 0.1)
            tr2 = self.phonon_params.get('swap_transversal2', tr)
            self.phonon_axial[ion_a] += ax
            self.phonon_transversal[ion_a] += tr
            self.phonon_transversal2[ion_a] = (
                self.phonon_transversal2.get(ion_a, 0.0) + tr2)
            self.phonon_axial[ion_b] += ax
            self.phonon_transversal[ion_b] += tr
            self.phonon_transversal2[ion_b] = (
                self.phonon_transversal2.get(ion_b, 0.0) + tr2)

    def split_chain(self, ion_id: int, dst_zone_id: int):
        """
        イオンチェーンを分割し、片方をdst_zone_idへ移す

        方向はdst_zone_idとソースゾーンの位置関係から自動決定:
        - dst_zone_id > source → 右側チェーンがdstへ
        - dst_zone_id < source → 左側チェーンがdstへ

        フォノン: 移動するチェインのイオンにのみ加算
        """
        zone_id = self.locate(ion_id)
        contents = self.zone_contents[zone_id]
        pos = contents.index(ion_id)

        if len(contents) == 1:
            self.zone_contents[dst_zone_id] = [ion_id]
            self.zone_contents[zone_id] = []
            self.ion_locations[ion_id] = dst_zone_id
            # フォノン: 移動したイオンに加算
            if self.phonon_params:
                tr = self.phonon_params.get('split_transversal', 0.05)
                self.phonon_axial[ion_id] += self.phonon_params.get('split_axial', 10.0)
                self.phonon_transversal[ion_id] += tr
                self.phonon_transversal2[ion_id] = (
                    self.phonon_transversal2.get(ion_id, 0.0)
                    + self.phonon_params.get('split_transversal2', tr))
            return

        move_right = dst_zone_id > zone_id
        left_contents = contents[:pos + 1]
        right_contents = contents[pos + 1:]

        if move_right:
            if not right_contents:
                raise ValueError(
                    f"Cannot split: no right chain. ion {ion_id} is at right edge of zone {zone_id}"
                )
            moved = right_contents
            self.zone_contents[zone_id] = left_contents
            self.zone_contents[dst_zone_id] = right_contents
            for ion in right_contents:
                self.ion_locations[ion] = dst_zone_id
        else:
            moved = left_contents
            self.zone_contents[zone_id] = right_contents
            self.zone_contents[dst_zone_id] = left_contents
            for ion in left_contents:
                self.ion_locations[ion] = dst_zone_id

        # フォノン: 移動チェインにのみ加算
        if self.phonon_params:
            ax = self.phonon_params.get('split_axial', 10.0)
            tr = self.phonon_params.get('split_transversal', 0.05)
            tr2 = self.phonon_params.get('split_transversal2', tr)
            for ion in moved:
                self.phonon_axial[ion] += ax
                self.phonon_transversal[ion] += tr
                self.phonon_transversal2[ion] = (
                    self.phonon_transversal2.get(ion, 0.0) + tr2)

    def merge_chains(self, zone_left: int, zone_right: int):
        """
        2つの隣接イオンチェーンを結合

        結果はMemory/LIZゾーン側に保存、Temp Zoneは空になる。

        フォノン: 全イオンの平均を取ってからmergeコストを加算
        """
        left_contents = self.zone_contents.get(zone_left, [])
        right_contents = self.zone_contents.get(zone_right, [])
        merged = left_contents + right_contents

        # フォノン: 全イオンの平均 + merge cost
        if self.phonon_params and merged:
            n = len(merged)
            avg_ax = sum(self.phonon_axial.get(i, 0.0) for i in merged) / n
            avg_tr = sum(self.phonon_transversal.get(i, 0.0) for i in merged) / n
            avg_tr2 = sum(self.phonon_transversal2.get(i, 0.0) for i in merged) / n
            merge_ax = self.phonon_params.get('merge_axial', 10.0)
            merge_tr = self.phonon_params.get('merge_transversal', 0.05)
            merge_tr2 = self.phonon_params.get('merge_transversal2', merge_tr)
            for ion in merged:
                self.phonon_axial[ion] = avg_ax + merge_ax
                self.phonon_transversal[ion] = avg_tr + merge_tr
                self.phonon_transversal2[ion] = avg_tr2 + merge_tr2

        left_is_temp = self.architecture.is_temp(zone_left)
        right_is_temp = self.architecture.is_temp(zone_right)

        if left_is_temp and not right_is_temp:
            result_zone, empty_zone = zone_right, zone_left
        elif right_is_temp and not left_is_temp:
            result_zone, empty_zone = zone_left, zone_right
        else:
            result_zone, empty_zone = zone_left, zone_right

        self.zone_contents[result_zone] = merged
        self.zone_contents[empty_zone] = []
        for ion in merged:
            self.ion_locations[ion] = result_zone

    def apply_shuttle_phonon(self, ion_id: int, dst_chain_length: int = 0):
        """Shuttle操作のフォノン加算（ShuttleOp生成時に呼ぶ）

        Temp zone にいる全イオンに距離依存の axial + 固定 transversal を加算
        """
        if not self.phonon_params:
            return
        src_zone = self.locate(ion_id)
        ions_in_chain = self.zone_contents.get(src_zone, [])
        # 距離依存: shuttle_base_distance から計算
        base_dist = self.phonon_params.get('shuttle_base_distance', 1000.0)
        ion_factor = self.phonon_params.get('shuttle_ion_factor', 5.0)
        distance_um = base_dist - ion_factor * dst_chain_length
        axial_per_180 = self.phonon_params.get('shuttle_axial_per_180um', 0.5)
        axial_add = axial_per_180 * (distance_um / 180.0)
        tr_add = self.phonon_params.get('shuttle_transversal', 0.05)
        tr2_add = self.phonon_params.get('shuttle_transversal2', tr_add)
        for ion in ions_in_chain:
            self.phonon_axial[ion] += axial_add
            self.phonon_transversal[ion] += tr_add
            self.phonon_transversal2[ion] = (
                self.phonon_transversal2.get(ion, 0.0) + tr2_add)

    def cool_chain(self, zone_id: int):
        """ゾーン内チェーン全体を残留フォノン数まで冷却する（CoolOp の効果）

        残留値 (cool_residual_axial / cool_residual_transversal) より
        低いイオンはそのまま（冷却でフォノンが増えることはない）。
        """
        if not self.phonon_params:
            return
        res_ax = self.phonon_params.get('cool_residual_axial', 0.0)
        res_tr = self.phonon_params.get('cool_residual_transversal', 0.0)
        res_tr2 = self.phonon_params.get('cool_residual_transversal2', res_tr)
        for ion in self.zone_contents.get(zone_id, []):
            self.phonon_axial[ion] = min(self.phonon_axial.get(ion, 0.0), res_ax)
            self.phonon_transversal[ion] = min(
                self.phonon_transversal.get(ion, 0.0), res_tr)
            self.phonon_transversal2[ion] = min(
                self.phonon_transversal2.get(ion, 0.0), res_tr2)

    def advance_time(self, duration_us: float):
        """計画時近似時計を進め、background heating を全イオンに連続加算する。

        heating rate (quanta/s) は phonon_params の heating_rate_*_per_s。
        全レート 0 (既定) なら時計だけ進める = 既存挙動と完全互換。
        一様加算なので merge の平均化と可換 (適用順序による誤差なし)。
        """
        if duration_us <= 0:
            return
        self.clock_us += duration_us
        if not self.phonon_params:
            return
        r_ax = self.phonon_params.get('heating_rate_axial_per_s', 0.0)
        r_tr = self.phonon_params.get('heating_rate_transversal_per_s', 0.0)
        r_tr2 = self.phonon_params.get('heating_rate_transversal2_per_s', r_tr)
        if not (r_ax or r_tr or r_tr2):
            return
        dt_s = duration_us * 1e-6
        for ion in self.phonon_axial:
            self.phonon_axial[ion] += r_ax * dt_s
            self.phonon_transversal[ion] += r_tr * dt_s
            self.phonon_transversal2[ion] = (
                self.phonon_transversal2.get(ion, 0.0) + r_tr2 * dt_s)

    def can_execute_cool(self, op: CoolOp) -> bool:
        """CoolOp 実行可能か: coolant 指定時はそれが対象ゾーンに居ること"""
        if op.coolant_id < 0:
            return True
        return self.ion_locations.get(op.coolant_id) == op.zone_id

    def record_gate_phonon(self, gate_type: str, targets: List[int]):
        """ゲート実行時のフォノン数を記録"""
        if not self.phonon_params:
            return
        # 記録形式は (axial, transversal mode1, transversal mode2)。
        # 既存の解析コードは p[0]/p[1] を参照するため後方互換。
        phonons = {ion: (self.phonon_axial.get(ion, 0.0),
                         self.phonon_transversal.get(ion, 0.0),
                         self.phonon_transversal2.get(ion, 0.0))
                   for ion in targets}
        self.phonon_at_gates.append({
            "gate": f"{gate_type}({','.join(str(t) for t in targets)})",
            "phonons": phonons,
            # 計画時近似時計でのゲート開始時刻 (ゲート自身の duration は含まない)
            "time_us": self.clock_us,
        })

    def get_ion_position_in_zone(self, ion_id: int) -> int:
        zone_id = self.locate(ion_id)
        return self.zone_contents[zone_id].index(ion_id)

    def is_at_right_edge(self, ion_id: int) -> bool:
        zone_id = self.locate(ion_id)
        return self.get_ion_position_in_zone(ion_id) == len(self.zone_contents[zone_id]) - 1


# ==========================================
# Movement Planner
# ==========================================

class _ClockedOps(list):
    """MovementPlanner.plan の出力リスト。

    op が追加されるたびに計画時近似時計 (QCCDState.clock_us) を
    その op の duration だけ進め、background heating を適用する。
    plan 本体・planner v2 の後処理を含む全 emit 経路を 1 箇所で捕捉する。
    冷却閾値ポリシー・record_gate_phonon は emit 済み op の加熱を
    反映した値を見る (= 計画時判定が rate 寄与を見られる)。
    """

    def __init__(self, state: "QCCDState", op_timer: Estimator):
        super().__init__()
        self._state = state
        self._op_timer = op_timer

    def append(self, op):
        super().append(op)
        self._state.advance_time(self._op_timer.operation_time(op))

    def extend(self, ops_iter):
        for op in ops_iter:
            self.append(op)


class MovementPlanner:
    """
    DAG + 初期配置 → 平坦なAtomicOp列

    責務:
    - 量子回路の実行順を守りながら移動操作を生成
    - ShuttleOp/SwapOp を Split/Merge にインライン展開
    - 各操作にゾーンアノテーション(zone_id, src_zone_id等)を付与
    - Schedulerに渡す平坦なList[Operation]を出力

    Schedulerへの約束:
    - 各操作に zones_used() に必要なフィールドが設定されている
    - 出力の順序は論理的な依存関係を反映している（Schedulerはこれを並列化する）
    """

    def __init__(self, architecture: QCCDArchitecture, config: "CompilerConfig" = None,
                 time_params=None):
        self.architecture = architecture
        self.config = config or CompilerConfig.baseline()
        # フォノン追跡パラメータ (外部注入可能。正本は estimator.TimeParameters)
        self.phonon_params = phonon_params_from(time_params)
        # 計画時近似時計用の per-op duration (heating rate の連続加算に使う)
        self._op_timer = Estimator(time_params)
        self._reorder_optimizer = None
        if self.config.memory_reorder:
            from .memory_reorder_optimizer import MemoryReorderOptimizer
            self._reorder_optimizer = MemoryReorderOptimizer(architecture)
        # Interleave用: MS gate後にスタビライザを切り替える
        self._last_ms_stabilizer: Optional[int] = None
        # アンシラ測定後冷却の周期カウンタ (MEASURE イベント数)
        self._measures_since_cool: int = 0
        # --- 予測型 merge 冷却 (predictive_cool_threshold_transversal) ---
        # 測定窓 = MEASURE イベント境界で区切った op 区間。
        # pass 1 (記録モード) で窓ごとの per-ion Δtr を記録し、pass 2 は
        # QCCDCompiler から注入された window_table を参照して
        # 「現在値 + 次窓Δ > θ」の予測判定を行う。
        self.window_table: Optional[List[dict]] = None   # pass 2 で参照
        self._record_windows: bool = False               # pass 1 記録モード
        self._recorded_windows: List[dict] = []
        # 診断用: merge 冷却エピソードごとの構成 op (id のリスト)。
        # スケジュール後に「エピソード実行中に測定/ゲートが走っていないか」を
        # 検証するために使う (tests/test_cooling_schedule_rules.py)
        self.episodes: List[List[int]] = []
        self._window_start_tr: Dict[int, float] = {}
        self._window_data_ions: Set[int] = set()
        self._measure_event_idx: int = 0
        self._measure_ancillas: Set[int] = set()
        # ゲート済み・未測定のアンシラ (= もつれ保持中)。直接冷却 (CoolOp) は
        # 内部状態を破壊するため、この集合が居るチェーンへは挿入できない
        self._entangled_ancillas: Set[int] = set()
        self._preferred_stabilizer: Optional[int] = None
        # 複数 LIZ 用: ancilla_id -> liz_zone_id の partition（None なら単一 LIZ 互換動作）
        self.partition: Optional[StabilizerPartition] = None
        # LIZ 常駐 coolant (resident_coolant_ids): 全退避から保護される回路外イオン
        self._resident: Set[int] = set(self.config.resident_coolant_ids or [])

    def _get_target_liz_for_op(self, op: Operation) -> Optional[int]:
        """Partition で割り当てられた LIZ を返す。partition が None または
        op.targets に partition 登録の ancilla が含まれない場合は None。
        """
        if self.partition is None:
            return None
        for target in op.targets:
            if self.partition.has(target):
                return self.partition.get_liz(target)
        return None

    def _get_target_liz_for_ancilla_set(self, ancilla_ids) -> Optional[int]:
        """ancilla_id 集合から共通の LIZ を返す。
        partition が None または ancilla_ids が複数 LIZ にまたがる場合は None。
        """
        if self.partition is None or not ancilla_ids:
            return None
        liz_ids = set()
        for a in ancilla_ids:
            if self.partition.has(a):
                liz_ids.add(self.partition.get_liz(a))
        if len(liz_ids) == 1:
            return next(iter(liz_ids))
        return None

    def _evict_ions(self, zone_id: int, ions: list, state: QCCDState,
                    protected: set = None) -> List[Operation]:
        """退避戦略に基づいて複数イオンをLIZから退避する

        Returns:
            退避に必要な操作列
        """
        ops = []
        if self.config.eviction_strategy == "balanced":
            eviction_plan = state.plan_evictions(zone_id, ions, protected=protected)
            for ion, dst in eviction_plan:
                ops.extend(self._move_ion_to_zone(ion, dst, state))
        else:
            # nearest: 従来のBFS
            for ion in ions:
                evict_dst = state.find_eviction_destination(zone_id)
                if evict_dst is not None:
                    ops.extend(self._move_ion_to_zone(ion, evict_dst, state))
        return ops

    def plan(self, dag: DAG, initial_placement: Dict[int, int] = None,
             state: QCCDState = None) -> List[Operation]:
        """
        メインエントリーポイント: DAG → 平坦なAtomicOp列

        Args:
            dag: 依存関係グラフ（事前にreset_execution済み想定）
            initial_placement: ion_id -> zone_id（state 指定時は不要）
            state: 既存の QCCDState を引き継いで plan する（per-LIZ stream 用。
                複数の sub-DAG を同一状態上で順に plan するときに使う）

        Returns:
            List[Operation]: Split/Swap/Merge/Gate の平坦な列（ゾーンアノテーション付き）
        """
        if state is None:
            state = QCCDState(architecture=self.architecture)
            # フォノン追跡を有効化（sim_params 経由の一元化パラメータ）
            state.phonon_params = dict(self.phonon_params)
            for ion_id, zone_id in initial_placement.items():
                state.place_ion_any(ion_id, zone_id)

        dag.reset_execution()
        # 出力リスト = 計画時近似時計のフック (op 追加ごとに background heating)
        ops: List[Operation] = _ClockedOps(state, self._op_timer)
        max_iterations = len(dag.nodes) * 200
        self._last_state = state  # フォノン結果取得用

        # Measure バッチ化: 遅延中の Measure ノード情報
        # batch_size > 1 の場合、Measure を即時実行せず溜めてからバッチ実行
        batch_size = self.config.measure_batch_size
        # LIZ容量でクランプ
        liz_cap = self.architecture.get_liz_zones()[0].capacity
        batch_size = min(batch_size, liz_cap)
        # バッチRESET済みアンシラ: flush時に自動RESETしたアンシラを記録し、
        # DAG上のRESETノードが後から出てきた場合にスキップする
        batch_reset_done: Set[int] = set()
        # 補助量子ビットのユニーク数でクランプ
        # reuse回路では同じアンシラが複数回MEASUREされるため、
        # 同時測定可能な数はユニークなアンシラ数が上限
        measure_ancillas = set()
        for n in dag.nodes.values():
            if isinstance(n.operation, MeasureOp):
                for t in n.operation.targets:
                    measure_ancillas.add(t)
        if len(measure_ancillas) < batch_size:
            batch_size = 1
        # 測定待ちリスト: [(ancilla_ion_id, MeasureOp, node_id), ...]
        pending_measures: List[tuple] = []

        # 予測型冷却: 測定窓トラッキングの初期化
        self._measure_ancillas = set(measure_ancillas)
        self._measure_event_idx = 0
        self._entangled_ancillas = set()
        self.episodes = []
        if self._record_windows:
            self._recorded_windows = []
            self._window_data_ions = set()
            self._window_start_tr = dict(state.phonon_transversal)

        for _ in range(max_iterations):
            if dag.is_complete() and not pending_measures:
                break

            # DAG完了だが測定待ちが残っている → フラッシュ
            if dag.is_complete() and pending_measures:
                ops.extend(self._flush_pending_measures(pending_measures, state, dag, batch_reset_done))
                pending_measures.clear()
                break

            front_layer = dag.get_front_layer()

            # バッチ化: Measureが実行可能でも遅延させるか判定
            node = self._find_executable_node(dag, front_layer, state)
            if node is not None:
                if isinstance(node.operation, MeasureOp) and batch_size > 1:
                    # Measureを遅延: DAG上では実行済みにするが、ops には追加しない
                    ancilla_id = node.operation.targets[0]
                    # 同じアンシラが既にpendingにいる場合はバッチ化せず先にフラッシュ
                    pending_ancillas = set(item[0] for item in pending_measures)
                    if ancilla_id in pending_ancillas:
                        ops.extend(self._flush_pending_measures(pending_measures, state, dag, batch_reset_done))
                        pending_measures.clear()
                    pending_measures.append((ancilla_id, node.operation, node.id))
                    dag.execute(node.id)

                    # アンシラをLIZからMemoryに退避（次のスタビライザに進むため）
                    if state.is_in_liz(ancilla_id):
                        liz_zone = state.locate(ancilla_id)
                        evict_dst = state.find_eviction_destination(liz_zone, ancilla_id)
                        if evict_dst is not None:
                            ops.extend(self._move_ion_to_zone(ancilla_id, evict_dst, state))

                    # バッチが溜まったら一括測定
                    if len(pending_measures) >= batch_size:
                        ops.extend(self._flush_pending_measures(pending_measures, state, dag, batch_reset_done))
                        pending_measures.clear()
                elif isinstance(node.operation, ResetOp) and batch_size > 1:
                    # バッチRESET済みのアンシラなら DAG上でスキップ
                    reset_ancilla = node.operation.targets[0]
                    if reset_ancilla in batch_reset_done:
                        dag.execute(node.id)
                        batch_reset_done.discard(reset_ancilla)
                        continue
                    else:
                        ops.append(self._annotate_gate(node.operation, state))
                        dag.execute(node.id)
                else:
                    # 閾値ベース自動冷却: MS 直前にゲート対象チェーンの
                    # transversal フォノンが閾値超なら CoolOp を先に挿入
                    cool_op = self._maybe_auto_cool(node.operation, state)
                    if cool_op is not None:
                        ops.append(cool_op)

                    # LIZ predictive trigger (10/26 会議): CX の直前にゲート対象
                    # データの追跡フォノンを検査し、閾値超なら隔離冷却エピソード
                    # を挿入してからゲートを撃つ (ゲート自身が冷えた状態で走る)
                    ops.extend(self._maybe_liz_pt(node.operation, state))

                    ops.append(self._annotate_gate(node.operation, state))
                    dag.execute(node.id)

                    # アンシラ測定後冷却: MEASURE 直後に深い再冷却を挿入
                    post_cool = self._maybe_cool_after_measure(node.operation, state)
                    if post_cool is not None:
                        ops.append(post_cool)

                    # merge 平均化冷却: memory チェーンが閾値超なら
                    # 冷アンシラを合流させて平均化で冷やし、LIZ へ戻す
                    ops.extend(self._maybe_merge_cool_memory(node.operation, state))

                    # 測定イベント境界: 予測型冷却の窓記録 / 予測判定
                    ops.extend(self._on_measure_event(node.operation, state))

                    # Interleave: MS gate実行後にスタビライザを切り替え
                    if self.config.interleave and isinstance(node.operation, GateOp) and node.operation.type == 'MS':
                        current_stab = node.stabilizer_id
                        if current_stab is not None:
                            self._last_ms_stabilizer = current_stab

                    # Measure/Reset 直後にMemory内先読み整列を挿入
                    if self._reorder_optimizer and isinstance(node.operation, (MeasureOp, ResetOp)):
                        reorder_ops = self._reorder_optimizer.generate_reorder_ops(
                            dag, state, node.id
                        )
                        ops.extend(reorder_ops)

                    # planner v2 フック: 測定窓 (LIZ 長時間占有) に重ねる
                    # memory zone 側の先行整列 / JIT staging を emit する
                    if isinstance(node.operation, MeasureOp):
                        ops.extend(self._measure_window_ops(dag, state))
            else:
                movement_ops = self._generate_movements(dag, front_layer, state)
                if not movement_ops:
                    raise RuntimeError(
                        "MovementPlanner: deadlock — no executable gate and no movement possible.\n"
                        f"Front layer: {front_layer}\n"
                        f"State: {state.ion_locations}"
                    )
                ops.extend(movement_ops)
        else:
            if not dag.is_complete():
                raise RuntimeError("MovementPlanner: max iterations exceeded")

        return ops

    def _measure_window_ops(self, dag: DAG, state: QCCDState) -> List[Operation]:
        """MEASURE emit 直後に呼ばれるフック。planner v2 が memory zone の
        先行整列 / JIT staging をここで emit する (v1 では no-op)。
        戻り値の op 列はそのまま flat_ops に追加される。"""
        return []

    def _flush_pending_measures(
        self,
        pending_measures: List[tuple],
        state: QCCDState,
        dag: DAG = None,
        batch_reset_done: Set[int] = None
    ) -> List[Operation]:
        """
        測定待ちアンシラを全てLIZに集めて一括Measure + 一括Resetする

        Interleaveモードでは、回路側のRESETに依存せず、MEASURE直後に
        コンパイラが自動的にRESETを挿入する。アンシラがまだLIZ上にいる
        ため搬送コストはゼロ。DAG上の対応するRESETノードはスキップする。

        Args:
            pending_measures: [(ancilla_id, MeasureOp, node_id), ...]
            state: 現在の QCCDState
            dag: DAG（RESETノードのスキップ用、省略可）

        Returns:
            移動 + Measure + Reset の操作列
        """
        ops: List[Operation] = []
        # 複数 LIZ: partition で全 ancilla が同じ LIZ に割り当てられていればそれを使う
        ancilla_ids = [item[0] for item in pending_measures]
        partition_liz = self._get_target_liz_for_ancilla_set(ancilla_ids)
        if partition_liz is not None:
            target_liz = partition_liz
        else:
            target_liz = self.architecture.get_liz_zones()[0].id

        # LIZ 上の非ターゲットイオンを退避 (常駐 coolant は残留)
        ancilla_set = set(item[0] for item in pending_measures)
        extra_ions = [ion for ion in state.zone_contents.get(target_liz, [])
                      if ion not in ancilla_set and ion not in self._resident]
        if extra_ions:
            ops.extend(self._evict_ions(target_liz, extra_ions, state,
                                        protected=ancilla_set | self._resident))

        # アンシラを全てLIZに集める
        for ancilla_id, _, _ in pending_measures:
            if state.locate(ancilla_id) != target_liz:
                ops.extend(self._move_ion_to_zone(ancilla_id, target_liz, state))

        # バッチMeasure: 全アンシラを一括測定
        all_targets = [item[0] for item in pending_measures]
        batch_measure = MeasureOp(targets=all_targets, zone_id=target_liz)
        ops.append(batch_measure)

        # アンシラ測定後冷却: バッチ測定直後に LIZ 上の全アンシラを再冷却
        # (バッチ測定 1 回 = MEASURE イベント 1 回として周期カウント)
        if (self.config.cool_ancilla_after_measure_us is not None
                and self._post_measure_cool_due()):
            state.cool_chain(target_liz)
            ops.append(CoolOp(
                zone_id=target_liz,
                duration_us=self.config.cool_ancilla_after_measure_us))

        # merge 平均化冷却 (バッチ測定直後・RESET 前)
        ops.extend(self._maybe_merge_cool_memory(batch_measure, state))

        # 測定イベント境界: 予測型冷却の窓記録 / 予測判定
        ops.extend(self._on_measure_event(batch_measure, state))

        # バッチReset: MEASURE直後に無条件で全アンシラをRESET
        # アンシラはまだLIZ上にいるので搬送不要
        # 最後のバッチ（DAG完了直前）ではRESET不要
        if dag is not None:
            # DAG上に残りのノードがある = まだスタビライザが残っている → RESET必要
            remaining_after = set(dag.nodes.keys()) - dag.get_executed()
            has_remaining_work = any(
                not isinstance(dag.nodes[nid].operation, ResetOp)
                for nid in remaining_after
            )
            if has_remaining_work:
                batch_reset = ResetOp(targets=all_targets, zone_id=target_liz)
                ops.append(batch_reset)
                # バッチRESET済みとして記録（後からDAGのRESETノードが出たらスキップ）
                if batch_reset_done is not None:
                    batch_reset_done.update(all_targets)

            # DAG上の対応するRESETノードをスキップ（実行済みとしてマーク）
            front_layer = dag.get_front_layer()
            for node_id in list(front_layer):
                node = dag.nodes[node_id]
                if isinstance(node.operation, ResetOp):
                    reset_ancilla = node.operation.targets[0]
                    if reset_ancilla in ancilla_set:
                        dag.execute(node_id)

        # planner v2 フック: バッチ測定窓に重ねる先行整列 / JIT staging
        if dag is not None:
            ops.extend(self._measure_window_ops(dag, state))

        return ops

    # ------------------------------------------
    # Gate Execution
    # ------------------------------------------

    def _find_executable_node(
        self, dag: DAG, front_layer: Set[int], state: QCCDState
    ) -> Optional[DAGNode]:
        """Front Layerから実行可能なノードを探す

        interleaveモード時: MS gate実行後に他のスタビライザのノードを優先
        """
        def is_executable(node):
            op = node.operation
            if isinstance(op, GateOp) and state.can_execute(op):
                return True
            if isinstance(op, (ResetOp, MeasureOp)) and state.can_execute_reset_measure(
                    op, extra_allowed=self._resident):
                return True
            if isinstance(op, CoolOp) and state.can_execute_cool(op):
                return True
            return False

        if not self.config.interleave:
            # 従来通り: min_id
            for node_id in sorted(front_layer):
                node = dag.nodes[node_id]
                if is_executable(node):
                    return node
            return None

        # === Interleave モード ===
        # MS gate 実行直後 → 他のスタビライザを優先
        preferred = []   # 優先スタビライザのノード
        fallback = []    # それ以外

        for node_id in sorted(front_layer):
            node = dag.nodes[node_id]
            if not is_executable(node):
                continue
            if self._last_ms_stabilizer is not None and node.stabilizer_id != self._last_ms_stabilizer:
                preferred.append(node)
            else:
                fallback.append(node)

        # 優先グループから選択、なければフォールバック
        if preferred:
            return preferred[0]
        if fallback:
            return fallback[0]
        return None

    def _maybe_auto_cool(self, op: Operation, state: QCCDState) -> Optional[CoolOp]:
        """閾値ベース自動冷却の判定 (cool_threshold_transversal ポリシー)。

        MS 直前にゲート対象イオンの transversal フォノンを見て、閾値を
        超えていたらその LIZ の CoolOp を生成し、状態にも冷却効果を適用する。
        冷却はチェーン全体 (LIZ 内の同乗イオン含む) に効く。
        record_gate_phonon は _annotate_gate 側 (冷却後) で走るため、
        記録されるゲート時フォノン = 冷却後の値になる。
        """
        thr = self.config.cool_threshold_transversal
        if thr is None:
            return None
        if not (isinstance(op, GateOp) and op.type == 'MS' and op.targets):
            return None
        if not state.phonon_params:
            return None
        if max(state.phonon_transversal.get(t, 0.0) for t in op.targets) <= thr:
            return None
        zone = state.locate(op.targets[0])
        state.cool_chain(zone)
        return CoolOp(zone_id=zone)

    def _maybe_cool_after_measure(self, op: Operation,
                                  state: QCCDState) -> Optional[CoolOp]:
        """アンシラ測定後冷却 (cool_ancilla_after_measure_us ポリシー)。

        MEASURE の直後にアンシラの LIZ へ指定時間の CoolOp を挿入する。
        測定時は非対象イオンが退避済みなので、実質アンシラだけが冷える。
        """
        dur = self.config.cool_ancilla_after_measure_us
        if dur is None or not isinstance(op, MeasureOp) or not op.targets:
            return None
        if not self._post_measure_cool_due():
            return None
        zone = state.locate(op.targets[0])
        return self._emit_cool_op(zone, state, dur)

    def _emit_cool_op(self, zone: int, state: QCCDState,
                      dur: float) -> 'CoolOp':
        """CoolOp の共通発行点: 冷却規則 (量子情報を保持するイオンが居る
        チェーンへの直接冷却禁止) を発行時に検査し、違反なら即例外を
        投げる (2026-10-25、窓内違反冷却バグの再発防止)。

        違反 = ゾーン内に (a) ゲート済み・未測定のアンシラ、または
        (b) データイオン (coolant でも測定アンシラでもない回路イオン) が居る。
        測定済み/未ゲートのアンシラと coolant への直接冷却は合法。
        (b) の判定は _measure_ancillas が plan() で設定済みの場合のみ
        (planner を直接呼ぶ単体テストでは (a) のみ検査)。"""
        bad = [i for i in state.zone_contents.get(zone, [])
               if i not in self._resident
               and (i in self._entangled_ancillas
                    or (bool(self._measure_ancillas)
                        and i not in self._measure_ancillas))]
        if bad:
            raise RuntimeError(
                f"cooling-rule violation: CoolOp on zone {zone} while "
                f"ions holding quantum information {bad} share the chain")
        state.cool_chain(zone)
        return CoolOp(zone_id=zone, duration_us=dur)

    def _post_measure_cool_due(self) -> bool:
        """アンシラ測定後冷却の周期判定 (cool_ancilla_measure_interval)。

        MEASURE イベントごとに呼ばれる前提でカウンタを進め、
        interval 回に 1 回だけ True を返す。
        """
        self._measures_since_cool += 1
        if self._measures_since_cool < self.config.cool_ancilla_measure_interval:
            return False
        self._measures_since_cool = 0
        return True

    def _maybe_merge_cool_memory(self, op: Operation,
                                 state: QCCDState) -> List[Operation]:
        """merge 平均化冷却 (merge_cool_threshold_transversal ポリシー)。

        MEASURE 直後 (非対象イオン退避済みでアンシラが LIZ に居る時点) に、
        LIZ に隣接する memory zone のチェーン平均 transversal フォノンを見て、
        閾値を超えていたら次の操作列を挿入する:

          1. アンシラを直接冷却 (残留値まで。既に残留値以下ならスキップ)
          2. アンシラを memory へ搬送・合流
             (merge_chains のフォノン平均化でチェーン全体が冷える)
          3. アンシラを LIZ へ戻す
             (合流直後は LIZ 側の端に居るため swap は発生しない)

        左右両方の memory が閾値超なら両側に順に行う。2 側目の前には
        1 側目の合流で温まったアンシラの再冷却が入るため、両側実行時は
        アンシラの冷却命令が 2 回挿入される。

        1 回の合流で除去できる熱は超過分の約 1/(k+1) (k = チェーン長) で、
        transversal の到達下限は残留値 + 輸送コスト
        (split + shuttle + merge の増分) で決まる。閾値超が続く限り、
        以降の MEASURE イベントでも繰り返し発動する。

        multi-LIZ 安全性: 経路上に他の LIZ を挟む memory zone は対象外
        (pass-through を構成的に回避)。
        """
        thr = self.config.merge_cool_threshold_transversal
        if thr is None or not isinstance(op, MeasureOp) or not op.targets:
            return []
        if not state.phonon_params:
            return []
        ancilla = op.targets[0]
        liz = state.locate(ancilla)
        if not self.architecture.is_liz(liz):
            return []

        target_zones = []
        for mem in self._reachable_memory_zones(liz):
            chain = state.zone_contents.get(mem.id, [])
            if not chain:
                continue
            mean_tr = sum(state.phonon_transversal.get(i, 0.0)
                          for i in chain) / len(chain)
            if mean_tr > thr:
                target_zones.append(mem.id)
        return self._merge_cool_episode(ancilla, liz, target_zones, state)

    def _reachable_memory_zones(self, liz: int):
        """LIZ から他 LIZ を跨がず到達できる memory zone 列
        (multi-LIZ 安全性: pass-through を構成的に回避)。"""
        zones = []
        for mem in self.architecture.get_memory_zones():
            path = self.architecture.get_path(liz, mem.id)
            if any(self.architecture.is_liz(w) and w != liz for w in path):
                continue
            zones.append(mem)
        return zones

    def _merge_cool_episode(self, ancilla: int, liz: int,
                            target_zones: List[int],
                            state: QCCDState) -> List[Operation]:
        """merge 平均化冷却エピソード: 冷アンシラを target_zones の各 memory へ
        順に合流させて平均化で冷やし、LIZ へ戻して再冷却する。

        往復でイオン配置が元に戻るため、op 列の構造 (測定窓の中身) を
        変えない — 予測型トリガの窓増分が pass 間で不変である根拠。
        """
        if not target_zones:
            return []
        res_ax = state.phonon_params.get('cool_residual_axial', 0.0)
        res_tr = state.phonon_params.get('cool_residual_transversal', 0.0)
        res_tr2 = state.phonon_params.get('cool_residual_transversal2', res_tr)
        # エピソード境界の冷却は「LIZ の補助をゾーン一括で冷やす」操作で、
        # 補助一斉再冷却 (coolant_recool_us) と物理的に同一。
        # coolant_recool_us が設定されていればそれを使い、アクチュエータ間で
        # cool op あたりの時間を揃える (2026-10-07、公平比較のため)。
        # 未設定 (常駐なしの merge/pred arm) は従来通り
        # cool_ancilla_after_measure_us。
        cool_dur = (self.config.coolant_recool_us
                    if self.config.coolant_recool_us is not None
                    else self.config.cool_ancilla_after_measure_us)

        def _ancilla_warm(anc: int) -> bool:
            return (state.phonon_transversal.get(anc, 0.0) > res_tr + 1e-12
                    or state.phonon_transversal2.get(anc, 0.0) > res_tr2 + 1e-12
                    or state.phonon_axial.get(anc, 0.0) > res_ax + 1e-12)

        ops: List[Operation] = []
        for mem_id in target_zones:
            # 1) アンシラを冷却 (前の側への合流で温まっていれば再冷却)
            if _ancilla_warm(ancilla):
                ops.append(self._emit_cool_op(liz, state, cool_dur))
            # 2) memory へ合流 (merge の平均化で memory チェーンが冷える)
            ops.extend(self._move_ion_to_zone(ancilla, mem_id, state))
            # 3) LIZ へ帰還
            ops.extend(self._move_ion_to_zone(ancilla, liz, state))
        # 4) 熱シャトルとして吸った熱を捨てる: 最後の合流で温まったまま
        #    次のスタビライザ測定に使われないよう、帰還後に再冷却する
        if ops and _ancilla_warm(ancilla):
            ops.append(self._emit_cool_op(liz, state, cool_dur))
        if ops:
            self.episodes.append([id(o) for o in ops])
        return ops

    def _maybe_recool_auxiliaries(self, liz: int,
                                  state: QCCDState) -> List[Operation]:
        """補助量子ビット (coolant + 測定アンシラ) の一斉再冷却
        (coolant_recool_us ポリシー)。

        LIZ がデータを含まず補助のみになった時点で、残留値を超えて
        温まっていれば CoolOp を 1 つ挿入する。次に訪れるデータ量子
        ビットが冷たい常駐チェーンと merge できるようにするための
        「訪問間」冷却。CoolOp はゾーン一括なので一斉冷却の仮定と一致する。
        """
        dur = self.config.coolant_recool_us
        if dur is None or not state.phonon_params:
            return []
        contents = state.zone_contents.get(liz, [])
        if not contents:
            return []
        aux = self._measure_ancillas | self._resident
        if not all(ion in aux for ion in contents):
            return []
        # 冷却規則ガード (2026-10-25): ゲート済み・未測定のアンシラが居る
        # チェーンへの直接冷却は、もつれた内部状態を破壊するため挿入しない。
        # (測定窓の内部でのデータグループ入れ替え時に該当。10/25 検証では
        #  本ポリシーの発動の 100% がこのケースだった)
        dirty = [i for i in contents if i in self._entangled_ancillas]
        if dirty:
            if not self.config.isolate_ancilla_recool:
                return []
            return self._isolated_recool(liz, contents, dirty, state, dur)
        if not self._aux_warm(contents, state):
            return []
        return [self._emit_cool_op(liz, state, dur)]

    def _aux_warm(self, ions, state: QCCDState) -> bool:
        """指定イオンのいずれかが残留値を超えて温まっているか"""
        res_ax = state.phonon_params.get('cool_residual_axial', 0.0)
        res_tr = state.phonon_params.get('cool_residual_transversal', 0.0)
        res_tr2 = state.phonon_params.get('cool_residual_transversal2', res_tr)
        return any(
            state.phonon_transversal.get(i, 0.0) > res_tr + 1e-12
            or state.phonon_transversal2.get(i, 0.0) > res_tr2 + 1e-12
            or state.phonon_axial.get(i, 0.0) > res_ax + 1e-12
            for i in ions)

    def _isolated_recool(self, liz: int, contents, dirty, state: QCCDState,
                         dur: float) -> List[Operation]:
        """窓内の合法再冷却 (isolate_ancilla_recool, 2026-10-25 著者案)。

        もつれ保持中のアンシラを memory へ一時退避させ、LIZ に残った
        coolant のみを CoolOp で冷やし、アンシラを呼び戻す。
        - 退避は輸送のみなので内部状態を保存する (冷却規則に適合)
        - アンシラは memory チェーンとの merge 平均化を経て
          ≈ チェーン平均のフォノンを背負って帰還し、冷えた coolant
          (m 個) との merge で ≈ 1/(m+1) に希釈される
        - 往復で配置が元に戻るため窓の op 構造は不変 (pass 1/2 で
          条件・挙動が一致し、予測型トリガの窓整合を壊さない)
        """
        coolants = [i for i in contents if i not in self._entangled_ancillas]
        if not coolants or not self._aux_warm(coolants, state):
            return []
        ops: List[Operation] = []
        for _ in range(max(1, self.config.isolate_recool_iters)):
            cycle: List[Operation] = []
            ok = True
            for anc in dirty:
                dest = state.find_eviction_destination(liz, anc)
                if dest is None:
                    ok = False    # 退避先なし — このサイクルを諦める
                    break
                cycle.extend(self._move_ion_to_zone(anc, dest, state))
            if not ok:
                break
            cycle.append(self._emit_cool_op(liz, state, dur))
            for anc in reversed(dirty):
                cycle.extend(self._move_ion_to_zone(anc, liz, state))
            ops.extend(cycle)
            # 2 サイクル目以降は coolant が温まっていなければ打ち切り
            # (帰還 merge で吸う熱が尽きた = 反復の利得なし)
            if not self._aux_warm(coolants, state):
                break
        return ops

    def _maybe_liz_pt(self, op: Operation,
                      state: QCCDState) -> List[Operation]:
        """LIZ predictive trigger (2026-10-26 会議、liz_pt_threshold_transversal)。

        CX (MS) 直前の検査: ゲート対象データ d_i の transversal フォノンが
        閾値超なら、隔離冷却エピソードを挿入してからゲートを撃つ。
        注意 (11/3 診断): 停止条件は①閾値到達 ②floor_break (1 サイクルの
        低下が 1e-3 未満 = サイクル床 n̄_res + e_split + 2e_merge に到達)
        ③N_max。θ < 床 のとき②が支配的 (= cool-to-floor として動作)。
        背景加熱は op 追加時に時計で加算されるため、エピソード自身の
        所要時間分 (n_cycles × t_cool × rate) はループ内の cur には
        見えず、ゲート時値に乗る。
          1) 左側 Memory のチェーンを外側 Temp へ一括退避 (SplitOp 1 発。
             split_chain の方向非対称により一括退避は左方向のみ可能なため
             v1 は左側 M に固定)
          2) A → M、d_i → M と隔離 (M は専有、merge は (A, d_i) 間のみ)
          3) coolant のみになった LIZ を冷却 (冷却規則ガードに適合)
          4) d_i → LIZ、A → LIZ と帰還 (冷えた coolant との merge で希釈)
          5) d_i がまだ閾値超なら 2–4 を反復 (最大 liz_pt_max_iters)。
             隔離中はチェーン平均に戻されないため反復は幾何的に効く
          6) チェーンを M へ復帰 (MergeOp)
        往復で配置が元に戻るため、MPT の窓構造 (測定イベント列) は不変。
        """
        thr = self.config.liz_pt_threshold_transversal
        if thr is None or not state.phonon_params:
            return []
        if not hasattr(self, 'lpt_skips'):
            from collections import Counter
            self.lpt_skips = Counter()
        if not (isinstance(op, GateOp) and op.type == 'MS' and op.targets):
            return []
        data = [t for t in op.targets if t not in self._measure_ancillas]
        ancs = [t for t in op.targets if t in self._measure_ancillas]
        if len(data) != 1 or len(ancs) != 1:
            self.lpt_skips['bad_targets'] += 1
            return []
        d_i, anc = data[0], ancs[0]
        if state.phonon_transversal.get(d_i, 0.0) <= thr:
            self.lpt_skips['below_thr'] += 1
            return []
        liz = state.locate(anc)
        if not self.architecture.is_liz(liz) or state.locate(d_i) != liz:
            self.lpt_skips['not_in_liz'] += 1
            return []
        coolants = [i for i in state.zone_contents.get(liz, [])
                    if i in self._resident]
        if not coolants:
            self.lpt_skips['no_coolant'] += 1
            return []    # coolant 無しでは不可 (10/26 会議)
        cool_dur = (self.config.coolant_recool_us
                    if self.config.coolant_recool_us is not None
                    else self.config.cool_ancilla_after_measure_us)
        if cool_dur is None:
            return []
        if self.config.liz_pt_park == "temp":
            return self._liz_pt_temp_park(liz, anc, d_i, thr, cool_dur, state)
        if self.config.liz_pt_park == "home":
            # home 変種 (配達前判定) は _liz_pt_home_isolation が搬入直前に
            # 処理する。LIZ 到着後のここでは何もしない。
            # (実測 10/27: 配達前チェックは移動由来の加熱を見逃す — 比較用に残置)
            return []
        # park == "memory": LIZ 内チェック + 左チェーン一括退避 (10/26 会議設計)
        mems = sorted(m.id for m in self._reachable_memory_zones(liz))
        if not mems:
            return []
        m_zone = mems[0]
        f_zone = self.architecture.get_adjacent_temp(m_zone, "left")
        if f_zone is None or state.zone_contents.get(f_zone):
            return []    # 外側 Temp がない / 空いていない — 発動を諦める

        tgt = (self.config.liz_pt_target_transversal
               if self.config.liz_pt_target_transversal is not None else thr)
        ops_out: List[Operation] = []
        # 1) チェーン一括退避 (M が空なら不要)
        chain = list(state.zone_contents.get(m_zone, []))
        displaced = bool(chain)
        if displaced:
            boundary = chain[-1]
            ops_out.append(SplitOp(ion_id=boundary, dst_zone_id=f_zone,
                                   src_zone_id=m_zone))
            state.split_chain(boundary, f_zone)
        # 2)–5) 隔離冷却サイクル (到達目標 = tgt、発火閾値 thr と分離可)
        n_cyc = 0
        for _ in range(max(1, self.config.liz_pt_max_iters)):
            cur = state.phonon_transversal.get(d_i, 0.0)
            if cur <= tgt:
                break
            # 収束判定 (temp 変種と同じ): 下がらなくなったら空転前に打ち切る
            if n_cyc > 0 and cur >= prev_v - 1e-3:
                self.lpt_skips['floor_break'] += 1
                break
            prev_v = cur
            n_cyc += 1
            ops_out.extend(self._move_ion_to_zone(anc, m_zone, state))
            ops_out.extend(self._move_ion_to_zone(d_i, m_zone, state))
            ops_out.append(self._emit_cool_op(liz, state, cool_dur))
            ops_out.extend(self._move_ion_to_zone(d_i, liz, state))
            ops_out.extend(self._move_ion_to_zone(anc, liz, state))
        # 6) チェーン復帰
        if displaced:
            left_z, right_z = min(f_zone, m_zone), max(f_zone, m_zone)
            ops_out.append(MergeOp(zone_left=left_z, zone_right=right_z))
            state.merge_chains(left_z, right_z)
        if ops_out:
            self.episodes.append([id(o) for o in ops_out])
        return ops_out

    def _liz_pt_temp_park(self, liz: int, anc: int, d_i: int, thr: float,
                          cool_dur: float, state: QCCDState) -> List[Operation]:
        """LIZ predictive trigger の Temp 退避変種 (liz_pt_park="temp")。

        (A, d_i) を LIZ 隣接 Temp へ 1 SplitOp で退避し、coolant のみの
        LIZ を冷却して 1 MergeOp で戻す。memory チェーンに一切触れない
        ため、チェーン全員への split+merge コストが発生しない (memory
        変種の自滅要因を回避)。整列 swap は初回サイクルのみ必要
        (merge 後の並びが [d_i, A, coolants...] に保たれるため)。
        仮定: 冷却中の隣接 Temp 滞在可 (実機確認待ち)。
        """
        park = self.architecture.get_adjacent_temp(liz, "left")
        if park is None or state.zone_contents.get(park):
            self.lpt_skips['park_busy'] += 1
            return []
        tgt = (self.config.liz_pt_target_transversal
               if self.config.liz_pt_target_transversal is not None else thr)
        self.lpt_skips['fired'] += 1
        entry_v = state.phonon_transversal.get(d_i, 0.0)
        self.lpt_skips[f'entry_{round(entry_v, 0):g}'] += 1
        ops_out: List[Operation] = []
        # ペアを LIZ チェーンの左端に整列: anc を左端へ → d_i を左端へ
        # (anc は 2 番目へ押し込まれ、[d_i, anc, coolants...] になる)
        ops_out.extend(self._position_for_shuttle(anc, liz, False, state))
        ops_out.extend(self._position_for_shuttle(d_i, liz, False, state))
        n_cycles = 0
        for _ in range(max(1, self.config.liz_pt_max_iters)):
            cur = state.phonon_transversal.get(d_i, 0.0)
            if cur <= tgt:
                break
            # 収束判定: 前サイクルから実質下がっていなければ、サイクルの
            # 到達可能床 (per-cycle コストと coolant 残留値で決まる) が
            # tgt より上 — これ以上回しても空転するため打ち切る
            if n_cycles > 0 and cur >= prev_v - 1e-3:
                self.lpt_skips['floor_break'] += 1
                break
            prev_v = cur
            contents = state.zone_contents[liz]
            if contents[:2] != [d_i, anc]:
                self.lpt_skips['order_break'] += 1
                break    # 想定外の並び — 発動を諦める
            n_cycles += 1
            boundary = contents[1]
            ops_out.append(SplitOp(ion_id=boundary, dst_zone_id=park,
                                   src_zone_id=liz))
            state.split_chain(boundary, park)
            ops_out.append(self._emit_cool_op(liz, state, cool_dur))
            left_z, right_z = min(park, liz), max(park, liz)
            ops_out.append(MergeOp(zone_left=left_z, zone_right=right_z))
            state.merge_chains(left_z, right_z)
        # 診断: 使用サイクル数とエピソード終了時の d_i の値の分布
        self.lpt_skips[f'cycles_{n_cycles}'] += 1
        exit_v = state.phonon_transversal.get(d_i, 0.0)
        self.lpt_skips[f'exit_{round(exit_v, 1):g}'] += 1
        if ops_out:
            self.episodes.append([id(o) for o in ops_out])
        return ops_out

    def _on_measure_event(self, op: Operation,
                          state: QCCDState) -> List[Operation]:
        """測定イベント境界の処理 (predictive_cool_threshold_transversal)。

        呼び出し位置 = MEASURE emit + 測定後冷却ポリシー群の後
        (= 次の生成元の測定窓が始まる直前のチェックポイント)。

        - pass 1 (_record_windows): この境界で窓を閉じ、per-ion Δtr と
          窓内 MS のデータ量子ビット集合を記録する
        - pass 2 (window_table あり): 次窓の予測値 = 現在値 + Δ を評価し、
          θ 超のデータ量子ビットが居る memory へ merge 冷却エピソードを挿入
        """
        if not isinstance(op, MeasureOp) or not op.targets:
            return []
        idx = self._measure_event_idx
        self._measure_event_idx += 1

        if self._record_windows:
            end_tr = state.phonon_transversal
            self._recorded_windows.append({
                'ancillas': list(op.targets),
                'delta_tr': {ion: end_tr.get(ion, 0.0)
                             - self._window_start_tr.get(ion, 0.0)
                             for ion in end_tr},
                'data_ions': set(self._window_data_ions),
            })
            self._window_start_tr = dict(end_tr)
            self._window_data_ions = set()
            return []

        if (self.config.predictive_cool_threshold_transversal is None
                or self.window_table is None or not state.phonon_params):
            return []
        return self._maybe_predictive_cool(idx, op, state)

    def _maybe_predictive_cool(self, idx: int, op: MeasureOp,
                               state: QCCDState) -> List[Operation]:
        """予測型 merge 冷却 (2026-10-02 プロトコル)。

        予測値 = 現在値 (pass 2 が逐次シミュレートした正確な値) +
                 次窓の増分 Δtr (pass 1 記録; 窓内 op 列が pass 間で同一なら
                 時間シフトに対して厳密)。
        予測が θ 超のデータ量子ビットの居る memory チェーンへ
        merge 冷却エピソードを最大 predictive_cool_max_iters 回挿入する。
        """
        table = self.window_table
        # 窓整合の検証: この測定イベントが pass 1 と同一列上にあるか
        if idx >= len(table) or list(op.targets) != table[idx]['ancillas']:
            import warnings
            warnings.warn(
                f"predictive cool: measure event #{idx} diverged from pass 1 "
                f"(got {list(op.targets)}, expected "
                f"{table[idx]['ancillas'] if idx < len(table) else 'EOF'}); "
                f"disabling prediction for the rest of this compile")
            self.window_table = None
            return []
        if idx + 1 >= len(table):
            return []    # 最終窓の後にはゲートが無い

        thr = self.config.predictive_cool_threshold_transversal
        nxt = table[idx + 1]
        delta, data_ions = nxt['delta_tr'], nxt['data_ions']
        if not data_ions:
            return []
        ancilla = op.targets[0]
        liz = state.locate(ancilla)
        if not self.architecture.is_liz(liz):
            return []

        if self.config.predictive_cool_actuator == "data":
            return self._data_visit_episodes(data_ions, delta, thr, liz, state)

        ops: List[Operation] = []
        for _ in range(self.config.predictive_cool_max_iters):
            hot = {i for i in data_ions
                   if state.phonon_transversal.get(i, 0.0)
                   + delta.get(i, 0.0) > thr}
            if not hot:
                break
            target_zones = [mem.id for mem in self._reachable_memory_zones(liz)
                            if hot & set(state.zone_contents.get(mem.id, []))]
            episode = self._merge_cool_episode(ancilla, liz, target_zones, state)
            if not episode:
                break    # 対象が memory に居ない等 — 空回りを防ぐ
            ops.extend(episode)
        return ops

    def _data_visit_episodes(self, data_ions, delta, thr, liz: int,
                             state: QCCDState) -> List[Operation]:
        """データ移動型アクチュエータ (2026-10-07 設計合意)。

        予測超過のデータイオンを 1 個ずつ LIZ へ搬送し、冷たい常駐
        チェーン (coolant + アンシラ) との merge 平均化で冷やして
        元の memory ゾーンへ戻す (帰還はチェーン端、順序は保存しない)。
        各訪問の前に補助が温まっていれば一斉再冷却を挟む。
        イオンごとに予測を再評価する (先行イオンの帰還 merge が
        チェーンを平均化で冷やし、後続が不要になることがあるため)。
        """
        reachable = {mem.id for mem in self._reachable_memory_zones(liz)}
        ops: List[Operation] = []
        # 走査順を決定的にする (予測超過量の大きい順 → 同値は id 順)
        order = sorted(
            data_ions,
            key=lambda i: (-(state.phonon_transversal.get(i, 0.0)
                             + delta.get(i, 0.0)), i))
        for ion in order:
            predicted = (state.phonon_transversal.get(ion, 0.0)
                         + delta.get(ion, 0.0))
            if predicted <= thr:
                continue
            src = state.locate(ion)
            if src == liz or src not in reachable:
                continue    # LIZ 在中 / 他 LIZ の先に居るイオンは対象外
            episode: List[Operation] = []
            # 1) 補助一斉再冷却 (前の訪問者の熱を捨て、冷たいプールで迎える)
            episode.extend(self._maybe_recool_auxiliaries(liz, state))
            # 2) 搬入 (LIZ への merge = 平均化冷却)
            episode.extend(self._move_ion_to_zone(ion, liz, state))
            # 3) 帰還 (元の memory ゾーンのチェーン端へ)
            episode.extend(self._move_ion_to_zone(ion, src, state))
            if episode:
                ops.extend(episode)
                self.episodes.append([id(o) for o in episode])
        # 4) 最後の訪問者が残した熱を捨てる
        tail = self._maybe_recool_auxiliaries(liz, state)
        if tail:
            ops.extend(tail)
            if self.episodes:
                self.episodes[-1].extend(id(o) for o in tail)
        return ops

    def _annotate_gate(self, op: Operation, state: QCCDState) -> Operation:
        """GateOp/ResetOp/MeasureOpにzone_idを付与した新しいオペレーションを返す"""
        if isinstance(op, GateOp) and op.targets:
            liz_id = state.locate(op.targets[0])
            # MSゲート実行時のフォノン数を記録
            if op.type == 'MS':
                # もつれ追跡: MS に参加したアンシラは測定/リセットまで
                # 直接冷却の対象から外す (冷却規則ガード)
                self._entangled_ancillas.update(
                    t for t in op.targets if t in self._measure_ancillas)
                state.record_gate_phonon(op.type, op.targets)
                if self._record_windows:
                    # 予測型冷却 pass 1: 窓内のデータ量子ビットを収集
                    self._window_data_ions.update(
                        t for t in op.targets
                        if t not in self._measure_ancillas)
            return GateOp(type=op.type, targets=op.targets, params=op.params, zone_id=liz_id)
        if isinstance(op, ResetOp) and op.targets:
            liz_id = state.locate(op.targets[0])
            self._entangled_ancillas.difference_update(op.targets)
            return ResetOp(targets=op.targets, zone_id=liz_id)
        if isinstance(op, MeasureOp) and op.targets:
            liz_id = state.locate(op.targets[0])
            self._entangled_ancillas.difference_update(op.targets)
            return MeasureOp(targets=op.targets, zone_id=liz_id)
        if isinstance(op, CoolOp):
            # zone_id は命令に明示されているため付与不要。
            # 実行時点のチェーンにフォノン効果を適用する
            # (record_gate_phonon と同様、状態への副作用はここで起こす)
            state.cool_chain(op.zone_id)
            return op
        return op

    # ------------------------------------------
    # Movement Generation
    # ------------------------------------------

    def _generate_movements(
        self, dag: DAG, front_layer: Set[int], state: QCCDState
    ) -> List[Operation]:
        """
        Front Layerの最優先ノードに必要な移動操作を生成

        優先順: interleaveモード時は他のスタビライザを優先、通常はnode_id昇順
        """
        # Interleaveモード: 他のスタビライザのノードを先に処理
        sorted_ids = sorted(front_layer)
        if self.config.interleave and self._last_ms_stabilizer is not None:
            preferred = [nid for nid in sorted_ids
                         if dag.nodes[nid].stabilizer_id != self._last_ms_stabilizer]
            others = [nid for nid in sorted_ids
                      if dag.nodes[nid].stabilizer_id == self._last_ms_stabilizer]
            sorted_ids = preferred + others

        for node_id in sorted_ids:
            node = dag.nodes[node_id]
            op = node.operation

            if isinstance(op, CoolOp):
                # coolant が対象ゾーンに居なければ搬送する
                if op.coolant_id >= 0 and state.locate(op.coolant_id) != op.zone_id:
                    return self._move_ion_to_zone(op.coolant_id, op.zone_id, state)
                continue

            if not isinstance(op, (GateOp, ResetOp, MeasureOp)):
                continue

            targets = op.targets
            target_set = set(targets)

            # 各イオンの現在地を確認
            ions_in_liz = [ion for ion in targets if state.is_in_liz(ion)]
            ions_not_in_liz = [ion for ion in targets if not state.is_in_liz(ion)]

            # 全イオンがLIZ上にいる場合
            if not ions_not_in_liz:
                liz_zones = set(state.locate(ion) for ion in targets)
                if len(liz_zones) == 1:
                    # 同じLIZ内にいる
                    target_liz = state.locate(targets[0])
                    if isinstance(op, (ResetOp, MeasureOp)):
                        # RESET/MEASURE: ターゲット以外のイオンを退避（左右交互）
                        # 常駐 coolant は残留
                        extra_ions = [
                            ion for ion in state.zone_contents.get(target_liz, [])
                            if ion not in target_set and ion not in self._resident
                        ]
                        if extra_ions:
                            return self._evict_ions(
                                target_liz, extra_ions, state,
                                protected=target_set | self._resident)
                    # Gate or RESET/MEASURE with no extra ions → 実行可能なはず
                    continue
                # 異なるLIZ → 一方のLIZに集める（バグ修正: 旧実装の無限ループを解消）
                target_liz = state.locate(targets[0])
                result = []
                for ion in targets:
                    if state.locate(ion) != target_liz:
                        src_liz = state.locate(ion)
                        evict_dst = state.find_eviction_destination(src_liz, ion)
                        if evict_dst is None:
                            raise RuntimeError(
                                f"Cannot evict ion {ion} from LIZ {src_liz}: no memory space available"
                            )
                        result.extend(self._move_ion_to_zone(ion, evict_dst, state))
                        result.extend(self._move_ion_to_zone(ion, target_liz, state))
                if result:
                    return result
                continue

            # 目標LIZを決定
            if ions_in_liz:
                # 既にLIZにいるイオンのLIZを目標に
                target_liz = state.locate(ions_in_liz[0])
            else:
                # 複数 LIZ: partition で割り当てられた LIZ を優先
                partition_liz = self._get_target_liz_for_op(op)
                if partition_liz is not None:
                    target_liz = partition_liz
                elif targets:
                    # partition 不明 (data qubit への単一 qubit ゲート等):
                    # イオン現在地の最寄り LIZ を選ぶ。「空きのある最初の LIZ」
                    # だと満杯時に遠い LIZ が選ばれ、占有 LIZ の通過
                    # (pass-through guard 違反) を誘発する。空きは後段の
                    # 退避処理が確保する
                    ion_zone = state.locate(targets[0])
                    target_liz = min(
                        (z.id for z in self.architecture.get_liz_zones()),
                        key=lambda lz: abs(lz - ion_zone))
                else:
                    target_liz = state.get_liz_with_space()
                    if target_liz is None:
                        target_liz = self.architecture.get_liz_zones()[0].id

            result = []

            # RESET/MEASUREの場合: LIZ上のターゲット以外を全て退避 (常駐 coolant は残留)
            if isinstance(op, (ResetOp, MeasureOp)):
                extra_ions = [ion for ion in state.zone_contents.get(target_liz, [])
                              if ion not in target_set and ion not in self._resident]
                if extra_ions:
                    result.extend(self._evict_ions(
                        target_liz, extra_ions, state,
                        protected=target_set | self._resident))

            # LIZに必要な空きを確保
            ions_to_bring = [ion for ion in targets if state.locate(ion) != target_liz]
            free = state.get_zone_free_space(target_liz)
            needed = len(ions_to_bring)

            if free < needed:
                liz_contents = list(state.zone_contents.get(target_liz, []))
                candidates = [ion for ion in liz_contents
                              if ion not in target_set and ion not in self._resident]
                n_to_evict = needed - free
                if len(candidates) < n_to_evict:
                    raise RuntimeError(
                        f"Cannot free {n_to_evict} spots in LIZ {target_liz}. "
                        f"Only {len(candidates)} evictable ions "
                        f"(protected: {target_set | self._resident})"
                    )
                result.extend(self._evict_ions(
                    target_liz, candidates[:n_to_evict], state,
                    protected=target_set | self._resident))
                # 補助一斉再冷却: データ退避で LIZ が補助のみになった瞬間、
                # 次のデータ到着前に冷たいチェーンへ戻す
                result.extend(self._maybe_recool_auxiliaries(target_liz, state))

            # ターゲットイオンをLIZへ移動。
            # LIZ predictive trigger (memory 変種, 10/26 会議設計) が有効な
            # MS ゲートでは、アンシラを先に搬入し、データは搬入の直前に
            # home 側で隔離冷却してから運ぶ (配達前判定 — 熱いまま LIZ へ
            # 運んで送り返す無駄を避ける)。
            lpt_memory = (self.config.liz_pt_threshold_transversal is not None
                          and self.config.liz_pt_park == "home"
                          and isinstance(op, GateOp) and op.type == 'MS')
            if lpt_memory:
                order = ([i for i in ions_to_bring
                          if i in self._measure_ancillas]
                         + [i for i in ions_to_bring
                            if i not in self._measure_ancillas])
                for ion in order:
                    if state.locate(ion) == target_liz:
                        continue
                    if ion not in self._measure_ancillas:
                        pre, restore_zones = self._liz_pt_home_isolation(
                            ion, op, target_liz, state)
                        result.extend(pre)
                        result.extend(
                            self._move_ion_to_zone(ion, target_liz, state))
                        if restore_zones is not None:
                            # チェーン復帰は dᵢ の搬出後 (状態変更もここで行う)
                            f_z, s_z = restore_zones
                            left_z = min(f_z, s_z)
                            right_z = max(f_z, s_z)
                            merge_op = MergeOp(zone_left=left_z,
                                               zone_right=right_z)
                            state.merge_chains(left_z, right_z)
                            result.append(merge_op)
                            if self.episodes:
                                self.episodes[-1].append(id(merge_op))
                    else:
                        result.extend(
                            self._move_ion_to_zone(ion, target_liz, state))
            else:
                for ion in ions_to_bring:
                    if state.locate(ion) != target_liz:
                        result.extend(self._move_ion_to_zone(ion, target_liz, state))

            if result:
                return result

        return []

    def _liz_pt_home_isolation(self, d_i: int, op: Operation, liz: int,
                               state: QCCDState):
        """LIZ predictive trigger の memory 変種 (配達前判定、2026-10-27 v2)。

        データ dᵢ を home Memory から動かさずに隔離する (10/26 会議設計の
        正しい読み):
          1) dᵢ を home チェーンの LIZ 側の端へ整列 (配達に必要な整列と共用)
          2) チェーンの残りを「LIZ と反対側」の外側 Temp へ 1 SplitOp で退避
             (dᵢ が端に居るため split_chain の方向非対称が両側で都合よく働く)
          3) サイクル: アンシラが LIZ → dᵢ の元へ出向いて merge (平均化)
             → coolant のみの LIZ を冷却 → アンシラ帰還 (冷えた coolant へ
             熱を捨てる)。dᵢ が閾値以下になるまで反復 (最大 liz_pt_max_iters)
          4) 戻り値 = (エピソード ops, チェーン復帰 ops)。復帰は dᵢ の搬出後に
             行う必要がある (dᵢ 残留中に復帰するとチェーン平均を浴び直す)
        """
        if not hasattr(self, 'lpt_skips'):
            from collections import Counter
            self.lpt_skips = Counter()
        thr = self.config.liz_pt_threshold_transversal
        if (thr is None or not state.phonon_params
                or state.phonon_transversal.get(d_i, 0.0) <= thr):
            self.lpt_skips['below_thr'] += 1
            return [], None
        ancs = [t for t in op.targets if t in self._measure_ancillas]
        if len(ancs) != 1 or state.locate(ancs[0]) != liz:
            self.lpt_skips['anc_not_in_liz'] += 1
            return [], None
        anc = ancs[0]
        coolants = [i for i in state.zone_contents.get(liz, [])
                    if i in self._resident]
        if not coolants:
            self.lpt_skips['no_coolant'] += 1
            return [], None
        s_zone = state.locate(d_i)
        if (self.architecture.is_liz(s_zone)
                or self.architecture.is_temp(s_zone)):
            self.lpt_skips['bad_src_zone'] += 1
            return [], None
        cool_dur = (self.config.coolant_recool_us
                    if self.config.coolant_recool_us is not None
                    else self.config.cool_ancilla_after_measure_us)
        if cool_dur is None:
            self.lpt_skips['no_duration'] += 1
            return [], None
        liz_side_right = s_zone < liz
        f_zone = self.architecture.get_adjacent_temp(
            s_zone, "left" if liz_side_right else "right")
        if f_zone is None or state.zone_contents.get(f_zone):
            self.lpt_skips['outer_temp_busy'] += 1
            return [], None
        self.lpt_skips['fired'] += 1

        pre: List[Operation] = []
        # 1) dᵢ を LIZ 側の端へ整列
        pre.extend(self._position_for_shuttle(d_i, s_zone, liz_side_right,
                                              state))
        # 2) チェーンの残りを外側 Temp へ退避
        contents = state.zone_contents[s_zone]
        restore_zones = None
        if len(contents) > 1:
            boundary = contents[-2] if liz_side_right else d_i
            pre.append(SplitOp(ion_id=boundary, dst_zone_id=f_zone,
                               src_zone_id=s_zone))
            state.split_chain(boundary, f_zone)
            restore_zones = (f_zone, s_zone)
        # 3) 隔離冷却サイクル (アンシラが出向く)
        for _ in range(max(1, self.config.liz_pt_max_iters)):
            if state.phonon_transversal.get(d_i, 0.0) <= thr:
                break
            # LIZ に coolant 以外 (他のデータ等) が残っていると、アンシラ退出後も
            # 冷却が許されないためサイクルを諦める
            others = [i for i in state.zone_contents.get(liz, [])
                      if i not in self._resident and i != anc]
            if others:
                break
            pre.extend(self._move_ion_to_zone(anc, s_zone, state))
            pre.append(self._emit_cool_op(liz, state, cool_dur))
            pre.extend(self._move_ion_to_zone(anc, liz, state))
        if pre:
            self.episodes.append([id(o) for o in pre])
        return pre, restore_zones

    # ------------------------------------------
    # Ion Movement (inline expansion)
    # ------------------------------------------

    def _move_ion_to_zone(
        self, ion_id: int, dst_zone: int, state: QCCDState
    ) -> List[Operation]:
        """
        イオンをdst_zoneへ移動する操作列を生成（Split+Shuttle+Mergeをインライン展開）

        1ホップ（隣接M/L間）の場合: src → temp → dst
        複数ホップの場合: 経路上の中間ゾーンを1ホップずつ進む
        """
        src_zone = state.locate(ion_id)
        if src_zone == dst_zone:
            return []

        # 経路を計算: src → ... → dst
        path = self.architecture.get_path(src_zone, dst_zone)
        # 経路上の非Tempゾーン（M/L）を中間目標として抽出
        # 各中間目標に対して1ホップずつ移動する
        waypoints = [z for z in path if not self.architecture.is_temp(z)]

        # multi-LIZ 安全条件: 占有中の LIZ を「通過」してはならない。
        # 中間 waypoint (src でも最終 dst でもない) が LIZ かつ非空なら、
        # その LIZ のゲート作業集合を踏み潰す不正搬送 → エラーで弾く。
        self._check_no_occupied_liz_passthrough(
            ion_id, waypoints, src_zone, dst_zone, state)

        ops = []
        for i in range(1, len(waypoints)):
            ops.extend(self._move_ion_one_hop(ion_id, waypoints[i], state))

        # 出所タグ (可視化/分析用): LIZ への搬送 = deliver、それ以外 = evict。
        # 既にタグ済みの op (planner v2 の sort/flush 等) は上書きしない
        origin = "deliver" if self.architecture.is_liz(dst_zone) else "evict"
        for op in ops:
            if not hasattr(op, "origin"):
                op.origin = origin

        return ops

    def _check_no_occupied_liz_passthrough(
        self, ion_id: int, waypoints: List[int], src_zone: int,
        dst_zone: int, state: QCCDState
    ) -> None:
        """搬送経路が「占有中の LIZ」を通過しないことを保証する。

        中間 waypoint (waypoints[1:-1]) のうち、LIZ かつ非空のものがあれば
        pass-through 違反としてエラーを送出する。空 LIZ の通過は許容
        (物理的に容量内・ゲート作業を妨げない)。

        診断: 違反は 2 系統の原因に分かれる ([[project_per_liz_stream_design]]):
        - cross-LIZ data が非隣接 LIZ を訪問する構造 (partition の不変条件破れ)
          → 本質的に transit protocol が要る (現状は非対応 = エラーが正しい)
        - eviction で遠い Memory へ飛ばされた動的配置ミス
          → placement / 退避方向の修正で解消できる
        """
        for w in waypoints[1:-1]:
            if self.architecture.is_liz(w) and state.zone_contents.get(w):
                occupants = list(state.zone_contents[w])
                raise RuntimeError(
                    f"Pass-through of OCCUPIED LIZ {w} while transporting ion "
                    f"{ion_id} from zone {src_zone} to zone {dst_zone} "
                    f"(waypoints={waypoints}). LIZ {w} currently holds "
                    f"{occupants}. Multi-LIZ routing must not transit an occupied "
                    f"LIZ. Cause is either (a) a non-adjacent LIZ visit required by "
                    f"the partition (fundamental: needs a transit protocol, not yet "
                    f"supported), or (b) the ion was evicted to a far Memory "
                    f"(fix initial placement / eviction direction)."
                )

    def _move_ion_one_hop(
        self, ion_id: int, dst_zone: int, state: QCCDState
    ) -> List[Operation]:
        """
        イオンを隣接する非Tempゾーンへ1ホップで移動する

        物理的な1ホップ: src(M/L) → temp(T) → dst(M/L)

        dst_zone が LIZ で容量が足りない場合、既存イオンを一時退避して
        通過させた後に戻す。
        """
        src_zone = state.locate(ion_id)
        if src_zone == dst_zone:
            return []

        is_right = src_zone < dst_zone
        direction = "right" if is_right else "left"
        temp_zone = self.architecture.get_adjacent_temp(src_zone, direction)

        if temp_zone is None:
            raise RuntimeError(
                f"No temp zone adjacent to zone {src_zone} (direction={direction}). "
                f"Cannot move ion {ion_id}."
            )

        ops = []

        # Step 0: dst_zone に空きがない場合の処理
        # TODO: LIZ通過時の一時退避を実装（複数LIZ対応）
        # 現在は容量チェックのみ行い、超過時はそのまま進む（Validation で検出）

        # Step 1: イオンを端に移動（展開済みSwap列を生成）
        ops.extend(self._position_for_shuttle(ion_id, src_zone, is_right, state))

        # Step 2: Split（ion_idをtemp_zoneへ分離）
        src_contents = state.zone_contents[src_zone]
        if len(src_contents) == 1:
            split_op = SplitOp(ion_id=ion_id, dst_zone_id=temp_zone, src_zone_id=src_zone)
            ops.append(split_op)
            state.split_chain(ion_id, temp_zone)
        else:
            ion_pos = src_contents.index(ion_id)
            if is_right:
                assert ion_pos == len(src_contents) - 1, \
                    f"ion {ion_id} should be at right end (pos={ion_pos}, len={len(src_contents)})"
                split_ref = src_contents[ion_pos - 1]
                split_op = SplitOp(ion_id=split_ref, dst_zone_id=temp_zone, src_zone_id=src_zone)
                ops.append(split_op)
                state.split_chain(split_ref, temp_zone)
            else:
                assert ion_pos == 0, \
                    f"ion {ion_id} should be at left end (pos={ion_pos})"
                split_op = SplitOp(ion_id=ion_id, dst_zone_id=temp_zone, src_zone_id=src_zone)
                ops.append(split_op)
                state.split_chain(ion_id, temp_zone)

        # Step 3: Shuttle（物理的な移動）
        dst_contents = state.zone_contents.get(dst_zone, [])
        site = 0 if is_right else len(dst_contents)
        shuttle_op = ShuttleOp(
            target=ion_id,
            dst=Location(trap=dst_zone, site=site),
            dst_chain_length=len(dst_contents),
            src_zone=temp_zone
        )
        ops.append(shuttle_op)
        state.apply_shuttle_phonon(ion_id, len(dst_contents))

        # Step 4: Merge
        left_z = min(temp_zone, dst_zone)
        right_z = max(temp_zone, dst_zone)
        merge_op = MergeOp(zone_left=left_z, zone_right=right_z)
        ops.append(merge_op)
        state.merge_chains(left_z, right_z)

        return ops

    def _position_for_shuttle(
        self, ion_id: int, src_zone: int, is_right: bool, state: QCCDState
    ) -> List[Operation]:
        """
        イオンをシャトル方向の端へ移動する展開済みSwap操作列を生成

        is_right=True  → 右端へ移動
        is_right=False → 左端へ移動
        """
        ops = []
        direction = 1 if is_right else -1

        while True:
            contents = state.zone_contents[src_zone]
            ion_pos = contents.index(ion_id)
            target_pos = len(contents) - 1 if is_right else 0

            if ion_pos == target_pos:
                break

            neighbor = contents[ion_pos + direction]
            # 左イオンと右イオンの順でexpand
            if direction == 1:
                ops.extend(self._expand_swap_inline(ion_id, neighbor, state))
            else:
                ops.extend(self._expand_swap_inline(neighbor, ion_id, state))

        return ops

    def _expand_swap_inline(
        self, ion_a: int, ion_b: int, state: QCCDState
    ) -> List[Operation]:
        """
        ion_a と ion_b を同一ゾーン内でスワップ（Split+Swap+Mergeに展開）

        ion_a と ion_b は隣接している必要がある。
        ion_a が ion_b の左にいることを前提とする（呼び出し側の責務）。

        パターン:
        1. 2イオンのみ: Swap のみ
        2. 左端ペア [A,B,...]: 右チェーンを右Tempへ退避
        3. 右端ペア [...,C,D]: 左チェーンを左Tempへ退避
        4. 中間ペア: 左を左Temp、右を右Tempへ退避
        """
        zone_id = state.locate(ion_a)
        assert state.locate(ion_b) == zone_id, "Ions must be in the same zone"

        contents = state.zone_contents[zone_id]
        pos_a = contents.index(ion_a)
        pos_b = contents.index(ion_b)
        left_pos = min(pos_a, pos_b)
        right_pos = max(pos_a, pos_b)
        left_ion = contents[left_pos]
        right_ion = contents[right_pos]

        arch = self.architecture
        left_temp = arch.get_adjacent_temp(zone_id, "left")
        right_temp = arch.get_adjacent_temp(zone_id, "right")

        ops = []

        if len(contents) == 2:
            # パターン1: 2イオンのみ → そのままSwap
            ops.append(SwapOp(targets=[left_ion, right_ion], zone_id=zone_id))
            state.swap_ions(left_ion, right_ion)

        elif left_pos == 0:
            # パターン2: 左端ペア [A, B, C, D] の Swap(A, B)
            # 右チェーン[C,D]を右Tempへ退避
            assert right_temp is not None, f"No right temp zone for zone {zone_id}"
            ops.append(SplitOp(ion_id=right_ion, dst_zone_id=right_temp, src_zone_id=zone_id))
            state.split_chain(right_ion, right_temp)
            ops.append(SwapOp(targets=[left_ion, right_ion], zone_id=zone_id))
            state.swap_ions(left_ion, right_ion)
            ops.append(MergeOp(zone_left=zone_id, zone_right=right_temp))
            state.merge_chains(zone_id, right_temp)

        elif right_pos == len(contents) - 1:
            # パターン3: 右端ペア [..., C, D] の Swap(C, D)
            # 左チェーンを左Tempへ退避
            assert left_temp is not None, f"No left temp zone for zone {zone_id}"
            split_ref = contents[left_pos - 1]
            ops.append(SplitOp(ion_id=split_ref, dst_zone_id=left_temp, src_zone_id=zone_id))
            state.split_chain(split_ref, left_temp)
            ops.append(SwapOp(targets=[left_ion, right_ion], zone_id=zone_id))
            state.swap_ions(left_ion, right_ion)
            ops.append(MergeOp(zone_left=left_temp, zone_right=zone_id))
            state.merge_chains(left_temp, zone_id)

        else:
            # パターン4: 中間ペア [A, B, C, D] の Swap(B, C)
            # 左チェーンを左Temp、右チェーンを右Tempへ退避
            assert left_temp is not None, f"No left temp zone for zone {zone_id}"
            assert right_temp is not None, f"No right temp zone for zone {zone_id}"
            split_ion1 = contents[left_pos - 1]
            ops.append(SplitOp(ion_id=split_ion1, dst_zone_id=left_temp, src_zone_id=zone_id))
            state.split_chain(split_ion1, left_temp)
            ops.append(SplitOp(ion_id=right_ion, dst_zone_id=right_temp, src_zone_id=zone_id))
            state.split_chain(right_ion, right_temp)
            ops.append(SwapOp(targets=[left_ion, right_ion], zone_id=zone_id))
            state.swap_ions(left_ion, right_ion)
            ops.append(MergeOp(zone_left=zone_id, zone_right=right_temp))
            state.merge_chains(zone_id, right_temp)
            ops.append(MergeOp(zone_left=left_temp, zone_right=zone_id))
            state.merge_chains(left_temp, zone_id)

        return ops


# ==========================================
# Scheduler
# ==========================================

class Scheduler:
    """
    AtomicOp列 → TimeStep列（ASAP並列スケジューリング）

    責務:
    - 各操作が占有するゾーン（zones_used）を見て競合を検出
    - 競合しない操作をできるだけ同じTimeStepに詰める
    - 入力の順序制約（同じゾーンを使う操作は入力順を維持）を守る

    配置情報を持たない: zones_usedフィールドのみを参照
    """

    def schedule(self, ops: List[Operation]) -> List[TimeStep]:
        """
        Args:
            ops: MovementPlannerが生成したAtomicOp列

        Returns:
            List[TimeStep]: 並列実行可能な操作グループの列
        """
        if not ops:
            return []

        n = len(ops)
        op_times = [0] * n              # 各opのTimeStepインデックス
        ts_zones: Dict[int, Set[int]] = {}    # ts -> 使用中ゾーン集合
        zone_last_ts: Dict[int, int] = {}     # zone_id -> そのゾーンを最後に使ったTS

        for i, op in enumerate(ops):
            zones = self._zones_used(op)

            # このopより前にzoneを使ったopのTS+1が最早開始TS
            earliest = 0
            for z in zones:
                if z in zone_last_ts:
                    earliest = max(earliest, zone_last_ts[z] + 1)

            # 競合しない最早TSを探す
            t = earliest
            while True:
                if t not in ts_zones or not (zones & ts_zones[t]):
                    break
                t += 1

            op_times[i] = t
            if t not in ts_zones:
                ts_zones[t] = set()
            ts_zones[t] |= zones
            for z in zones:
                zone_last_ts[z] = max(zone_last_ts.get(z, -1), t)

        # TimeStepに集約
        max_t = max(op_times)
        result = []
        for t in range(max_t + 1):
            step_ops = [ops[i] for i in range(n) if op_times[i] == t]
            if step_ops:
                result.append(TimeStep(ops=step_ops))

        return result

    def _zones_used(self, op: Operation) -> Set[int]:
        """操作が占有するゾーンの集合を返す"""
        if isinstance(op, SplitOp):
            zones = {op.dst_zone_id}
            if op.src_zone_id != -1:
                zones.add(op.src_zone_id)
            return zones
        if isinstance(op, ShuttleOp):
            # ShuttleはTempゾーン（src_zone）と移動先ゾーンの両方をロック
            # 移動先をロックしないと、退避が完了する前に他のイオンがそのゾーンへ
            # シャトルされるタイムステップに早期スケジューリングされる可能性がある
            zones = set()
            if op.src_zone != -1:
                zones.add(op.src_zone)
            dst_zone = op.dst.trap if hasattr(op.dst, 'trap') else op.dst[0]
            if dst_zone != -1:
                zones.add(dst_zone)
            return zones
        if isinstance(op, MergeOp):
            zones = set()
            if op.zone_left != -1:
                zones.add(op.zone_left)
            if op.zone_right != -1:
                zones.add(op.zone_right)
            return zones
        if isinstance(op, SwapOp):
            if op.zone_id != -1:
                return {op.zone_id}
            return set()
        if isinstance(op, (GateOp, ResetOp, MeasureOp, CoolOp)):
            if op.zone_id != -1:
                return {op.zone_id}
            return set()
        return set()


# ==========================================
# Compile Result
# ==========================================

@dataclass
class CompileResult:
    """コンパイル結果"""
    timesteps: List[TimeStep]   # Scheduler出力（並列化済み）
    stats: Dict[str, int]       # 統計情報
    config: "CompilerConfig" = None
    flat_ops: List[Operation] = field(default_factory=list)  # Scheduler前の元の生成順
    phonon_data: Optional[Dict] = None  # フォノン追跡結果
    final_placement: Optional[Dict[int, int]] = None  # ラウンド終了時のion配置 (ion_id -> zone_id)
    final_zone_contents: Optional[Dict[int, List[int]]] = None  # ゾーン内の並び順 (zone_id -> [ion_ids])
    dag: Optional[DAG] = None    # 依存関係グラフ（stabilizer_id 等のメタ情報用、Stim 再変換で利用）
    initial_placement: Optional[Dict[int, int]] = None  # コンパイル開始時の配置（ion_id -> zone_id）

    @property
    def operations(self) -> List[Operation]:
        """後方互換: timestepsを平坦化した操作列"""
        return [op for ts in self.timesteps for op in ts.ops]


# ==========================================
# QCCD Compiler (MovementPlanner + Scheduler のラッパー)
# ==========================================

class QCCDCompiler:
    """
    MovementPlanner と Scheduler を組み合わせたコンパイラ

    使用方法:
        compiler = QCCDCompiler(architecture)
        result = compiler.compile(dag, initial_placement)
        # result.timesteps: 並列TimeStep列
        # result.operations: 平坦な操作列（後方互換）
    """

    def __init__(self, architecture: QCCDArchitecture, config: CompilerConfig = None,
                 time_params=None):
        """
        Args:
            time_params: estimator.TimeParameters。フォノン追跡パラメータの
                外部注入用 (None ならデフォルト値)。sim_params.load_time_parameters
                でファイルから読み込んだものを渡せる。
        """
        self.architecture = architecture
        self.config = config or CompilerConfig.baseline()
        self.config.validate()
        self._time_params = time_params
        if self.config.planner == "v2":
            from .planner_v2 import StateMachinePlanner
            self._planner = StateMachinePlanner(architecture, self.config,
                                                time_params=time_params)
        else:
            self._planner = MovementPlanner(architecture, self.config,
                                            time_params=time_params)
        self._scheduler = Scheduler()

    def compile(
        self,
        dag: DAG,
        initial_placement: Dict[int, int],
        partition: Optional[StabilizerPartition] = None,
    ) -> CompileResult:
        """
        Args:
            dag: 依存関係グラフ
            initial_placement: ion_id -> zone_id の初期配置
            partition: ancilla -> LIZ の割り当て。None の場合は単一 LIZ 互換動作

        Returns:
            CompileResult: timesteps（並列化済み）と統計情報
        """
        # 複数 LIZ 用の partition を MovementPlanner に渡す
        self._planner.partition = partition

        # Phase 0: 予測型冷却の pass 1 — 本トリガのみ無効にした同一設定で
        # コンパイルし、測定窓ごとの per-ion 増分テーブルを作って注入する。
        # 予測 = 「pass 2 の現在値 + pass 1 の窓増分」(絶対値は使わない) なので
        # pass 2 側の冷却挿入による時間シフトの影響を受けない。
        if self.config.predictive_cool_threshold_transversal is not None:
            cfg1 = replace(self.config,
                           predictive_cool_threshold_transversal=None)
            planner1 = type(self._planner)(self.architecture, cfg1,
                                           time_params=self._time_params)
            planner1.partition = partition
            planner1._record_windows = True
            planner1.plan(dag, dict(initial_placement))
            self._planner.window_table = planner1._recorded_windows

        # Phase 1: 平坦なAtomicOp列を生成
        flat_ops = self._planner.plan(dag, initial_placement)

        # Phase 2: TimeStep列に並列化
        timesteps = self._scheduler.schedule(flat_ops)

        stats = self._calculate_stats(flat_ops)
        stats["timesteps"] = len(timesteps)

        # フォノン結果を抽出
        phonon_data = None
        if hasattr(self._planner, '_last_state') and self._planner._last_state.phonon_params:
            st = self._planner._last_state
            max_ax = max(st.phonon_axial.values()) if st.phonon_axial else 0.0
            max_tr = max(st.phonon_transversal.values()) if st.phonon_transversal else 0.0
            max_tr2 = max(st.phonon_transversal2.values()) if st.phonon_transversal2 else 0.0
            phonon_data = {
                'max_axial': max_ax,
                'max_transversal': max_tr,
                'max_transversal2': max_tr2,
                'per_ion_axial': dict(st.phonon_axial),
                'per_ion_transversal': dict(st.phonon_transversal),
                'per_ion_transversal2': dict(st.phonon_transversal2),
                'at_gates': st.phonon_at_gates,
                # 計画時近似時計の終端値 (直列和 μs)。並列スケジュール後の
                # 実時間とは別物 (こちらが常に長い)
                'planning_clock_us': st.clock_us,
            }

        # 最終配置を抽出（zone_contents の順序を保持）
        final_placement = None
        final_zone_contents = None
        if hasattr(self._planner, '_last_state'):
            final_placement = dict(self._planner._last_state.ion_locations)
            final_zone_contents = {
                z: list(ions) for z, ions in self._planner._last_state.zone_contents.items()
                if ions
            }

        return CompileResult(timesteps=timesteps, stats=stats, config=self.config,
                             flat_ops=flat_ops, phonon_data=phonon_data,
                             final_placement=final_placement,
                             final_zone_contents=final_zone_contents,
                             dag=dag,
                             initial_placement=dict(initial_placement))

    def _calculate_stats(self, ops: List[Operation]) -> Dict[str, int]:
        stats = {"gates": 0, "shuttles": 0, "swaps": 0, "splits": 0, "merges": 0,
                 "cools": 0}
        for op in ops:
            if isinstance(op, GateOp):
                stats["gates"] += 1
            elif isinstance(op, (ResetOp, MeasureOp)):
                stats["gates"] += 1
            elif isinstance(op, CoolOp):
                stats["cools"] += 1
            elif isinstance(op, ShuttleOp):
                stats["shuttles"] += 1
            elif isinstance(op, SwapOp):
                stats["swaps"] += 1
            elif isinstance(op, SplitOp):
                stats["splits"] += 1
            elif isinstance(op, MergeOp):
                stats["merges"] += 1
        return stats
