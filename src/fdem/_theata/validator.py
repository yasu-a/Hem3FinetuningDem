# Adapted from THeatA for the bundled workbench.
"""
Validator: コンパイラ出力の命令列を検証する

命令列を先頭から再生し、各ステップで以下の制約を検証する:
1. ゾーン容量: 各操作後、全ゾーンがcapacity以下
2. 隣接移動: Split/Merge/Shuttleが隣接ゾーン間でのみ発生
3. ゲート実行場所: Gate/Reset/MeasureがLIZ内のイオンに対してのみ実行
4. Swap禁止場所: SwapがTemp Zoneで実行されない
5. イオン保存: イオンが消えたり複製されたりしない
6. チェイン順序整合: Split/Mergeでゾーン内の順序が矛盾しない

使い方:
    validator = Validator(architecture)
    result = validator.validate(operations, initial_placement)
    if result.is_valid:
        print("OK")
    else:
        for error in result.errors:
            print(error)
"""

from dataclasses import dataclass, field
from typing import List, Dict, Set, Optional

from .ast_def import (
    Operation, GateOp, ShuttleOp, SwapOp, SplitOp, MergeOp, MeasureOp, ResetOp,
    CoolOp, TimeStep
)
from .qccd_compiler import QCCDArchitecture


# ==========================================
# 検証結果
# ==========================================

@dataclass
class ValidationError:
    """検証エラー1件"""
    step: int               # 何番目の操作で発生したか
    operation: Operation     # 問題の操作
    constraint: str          # 違反した制約の名前
    message: str             # 詳細メッセージ

    def __str__(self):
        return f"[Step {self.step}] {self.constraint}: {self.message}\n  op: {self.operation}"


@dataclass
class ValidationResult:
    """検証結果"""
    errors: List[ValidationError] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        return len(self.errors) == 0

    def summary(self) -> str:
        if self.is_valid:
            return "Validation PASSED: no errors found."
        lines = [f"Validation FAILED: {len(self.errors)} error(s) found."]
        for err in self.errors:
            lines.append(f"  {err}")
        return "\n".join(lines)


# ==========================================
# 内部State（検証専用、軽量版）
# ==========================================

class _ValidatorState:
    """
    検証用の軽量ゾーン状態トラッカー

    QCCDState と同じ状態遷移を行うが、エラーを raise せず
    Validator 側で検証する設計。
    """

    def __init__(self, architecture: QCCDArchitecture):
        self.architecture = architecture
        self.zone_contents: Dict[int, List[int]] = {z.id: [] for z in architecture.zones}
        self.ion_locations: Dict[int, int] = {}

    def place_ion(self, ion_id: int, zone_id: int):
        self.zone_contents[zone_id].append(ion_id)
        self.ion_locations[ion_id] = zone_id

    def all_ions(self) -> Set[int]:
        return set(self.ion_locations.keys())

    def locate(self, ion_id: int) -> Optional[int]:
        return self.ion_locations.get(ion_id)

    def zone_count(self, zone_id: int) -> int:
        return len(self.zone_contents.get(zone_id, []))

    def zone_capacity(self, zone_id: int) -> int:
        z = self.architecture.get_zone(zone_id)
        return z.capacity if z else 0

    # --- 状態更新（検証後に呼ぶ） ---

    def apply_split(self, ion_id: int, dst_zone_id: int):
        """SplitOp を適用して状態を更新"""
        zone_id = self.locate(ion_id)
        if zone_id is None:
            return  # イオンが見つからない（エラーは別途検出済み）

        contents = self.zone_contents[zone_id]
        if ion_id not in contents:
            return

        pos = contents.index(ion_id)

        # 1イオンのみ
        if len(contents) == 1:
            self.zone_contents[dst_zone_id] = [ion_id]
            self.zone_contents[zone_id] = []
            self.ion_locations[ion_id] = dst_zone_id
            return

        # 方向を決定
        if dst_zone_id > zone_id:
            # 右チェーンがdstへ
            left = contents[:pos + 1]
            right = contents[pos + 1:]
            if right:
                self.zone_contents[zone_id] = left
                self.zone_contents[dst_zone_id] = right
                for ion in right:
                    self.ion_locations[ion] = dst_zone_id
            else:
                # 右チェーンが空（エラーケース）— 状態は変更しない
                pass
        else:
            # 左チェーンがdstへ
            left = contents[:pos + 1]
            right = contents[pos + 1:]
            self.zone_contents[zone_id] = right
            self.zone_contents[dst_zone_id] = left
            for ion in left:
                self.ion_locations[ion] = dst_zone_id

    def apply_merge(self, zone_left: int, zone_right: int):
        """MergeOp を適用して状態を更新"""
        left_contents = self.zone_contents.get(zone_left, [])
        right_contents = self.zone_contents.get(zone_right, [])
        merged = left_contents + right_contents

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

    def apply_shuttle(self, ion_id: int, dst_zone_id: int):
        """ShuttleOp を適用（隣接移動）"""
        src_zone_id = self.locate(ion_id)
        if src_zone_id is None:
            return

        self.zone_contents[src_zone_id].remove(ion_id)

        # 方向で挿入位置を決定
        if src_zone_id < dst_zone_id:
            self.zone_contents[dst_zone_id].insert(0, ion_id)
        else:
            self.zone_contents[dst_zone_id].append(ion_id)

        self.ion_locations[ion_id] = dst_zone_id

    def apply_swap(self, ion_a: int, ion_b: int):
        """SwapOp を適用"""
        zone_id = self.locate(ion_a)
        if zone_id is None:
            return

        contents = self.zone_contents[zone_id]
        if ion_a in contents and ion_b in contents:
            pos_a = contents.index(ion_a)
            pos_b = contents.index(ion_b)
            contents[pos_a], contents[pos_b] = contents[pos_b], contents[pos_a]

    def snapshot(self) -> Dict[int, List[int]]:
        """現在のゾーン状態のコピーを返す"""
        return {zid: list(ions) for zid, ions in self.zone_contents.items()}


# ==========================================
# Validator 本体
# ==========================================

class Validator:
    """命令列の制約検証器"""

    def __init__(self, architecture: QCCDArchitecture):
        self.architecture = architecture

    def validate(
        self,
        operations: List[Operation],
        initial_placement: Dict[int, int],
    ) -> ValidationResult:
        """
        命令列を再生しながら制約を検証する

        Args:
            operations: 検証対象の命令列
            initial_placement: ion_id -> zone_id の初期配置

        Returns:
            ValidationResult: 検証結果（エラーリスト）
        """
        state = _ValidatorState(self.architecture)
        for ion_id, zone_id in initial_placement.items():
            state.place_ion(ion_id, zone_id)

        result = ValidationResult()
        expected_ions = state.all_ions().copy()

        # 初期配置の容量チェック
        for zone in self.architecture.zones:
            count = state.zone_count(zone.id)
            if count > zone.capacity:
                result.errors.append(ValidationError(
                    step=-1,
                    operation=None,
                    constraint="zone_capacity",
                    message=f"Initial placement exceeds zone {zone.id} capacity: "
                            f"{count} > {zone.capacity}",
                ))

        # 命令列を再生
        for step, op in enumerate(operations):
            self._validate_step(step, op, state, expected_ions, result)

        return result

    def _validate_step(
        self,
        step: int,
        op: Operation,
        state: _ValidatorState,
        expected_ions: Set[int],
        result: ValidationResult,
    ):
        """1ステップの検証と状態更新"""

        def err(constraint: str, message: str):
            result.errors.append(ValidationError(step, op, constraint, message))

        type_name = type(op).__name__

        # --- 操作ごとの事前検証 + 状態更新 ---

        if type_name == "SplitOp":
            self._validate_split(step, op, state, err)
            state.apply_split(op.ion_id, op.dst_zone_id)

        elif type_name == "MergeOp":
            self._validate_merge(step, op, state, err)
            state.apply_merge(op.zone_left, op.zone_right)

        elif type_name == "ShuttleOp":
            self._validate_shuttle(step, op, state, err)
            dst_trap = op.dst.trap if hasattr(op.dst, 'trap') else op.dst[0]
            state.apply_shuttle(op.target, dst_trap)

        elif type_name == "SwapOp":
            self._validate_swap(step, op, state, err)
            state.apply_swap(op.targets[0], op.targets[1])

        elif type_name in ("GateOp", "MeasureOp", "ResetOp"):
            self._validate_gate(step, op, state, err)
            # Gate/Measure/Reset は状態を変えない（イオン位置は不変）

        elif type_name == "CoolOp":
            self._validate_cool(step, op, state, err)
            # Cool は状態を変えない（イオン位置は不変、フォノンのみ）

        else:
            err("unknown_operation", f"Unknown operation type: {type_name}")
            return

        # --- 操作後の共通検証 ---

        # 1. ゾーン容量
        for zone in self.architecture.zones:
            count = state.zone_count(zone.id)
            if count > zone.capacity:
                err("zone_capacity",
                    f"Zone {zone.id}({zone.zone_type}) has {count} ions, "
                    f"capacity is {zone.capacity}")

        # 2. イオン保存
        current_ions = state.all_ions()
        if current_ions != expected_ions:
            missing = expected_ions - current_ions
            extra = current_ions - expected_ions
            parts = []
            if missing:
                parts.append(f"missing ions: {missing}")
            if extra:
                parts.append(f"extra ions: {extra}")
            err("ion_conservation", "; ".join(parts))

        # 3. イオン重複（同じイオンが複数ゾーンに存在）
        seen = set()
        for zone_id, contents in state.zone_contents.items():
            for ion in contents:
                if ion in seen:
                    err("ion_duplication",
                        f"Ion {ion} appears in multiple zones (found in zone {zone_id})")
                seen.add(ion)

    # --- 操作種別ごとの検証 ---

    def _validate_split(self, step, op: SplitOp, state: _ValidatorState, err):
        """SplitOp の事前検証"""
        ion_id = op.ion_id
        dst = op.dst_zone_id

        # イオンの存在確認
        src = state.locate(ion_id)
        if src is None:
            err("split_ion_not_found", f"Ion {ion_id} not found in any zone")
            return

        # 隣接チェック
        if abs(src - dst) != 1:
            err("split_adjacency",
                f"Split dst zone {dst} is not adjacent to source zone {src} "
                f"(distance={abs(src - dst)})")

        # dst ゾーンの存在確認
        dst_zone = self.architecture.get_zone(dst)
        if dst_zone is None:
            err("split_invalid_zone", f"Destination zone {dst} does not exist")
            return

        # dst ゾーンが空であることの確認
        if state.zone_count(dst) > 0:
            err("split_dst_not_empty",
                f"Split destination zone {dst} is not empty: "
                f"{state.zone_contents[dst]}")

        # 方向整合性: 右方向Splitなのにion_idが右端
        contents = state.zone_contents[src]
        if len(contents) > 1:
            pos = contents.index(ion_id)
            if dst > src and pos == len(contents) - 1:
                err("split_direction",
                    f"Right-direction split but ion {ion_id} is at right edge "
                    f"of zone {src} — no right chain to split off")

    def _validate_merge(self, step, op: MergeOp, state: _ValidatorState, err):
        """MergeOp の事前検証"""
        zl, zr = op.zone_left, op.zone_right

        # ゾーン存在確認
        for zid in (zl, zr):
            if self.architecture.get_zone(zid) is None:
                err("merge_invalid_zone", f"Zone {zid} does not exist")
                return

        # 隣接チェック
        if abs(zl - zr) != 1:
            err("merge_adjacency",
                f"Merge zones {zl} and {zr} are not adjacent "
                f"(distance={abs(zl - zr)})")

        # 左右の順序
        if zl >= zr:
            err("merge_order",
                f"zone_left ({zl}) should be less than zone_right ({zr})")

        # 少なくとも片方にイオンがいること
        if state.zone_count(zl) == 0 and state.zone_count(zr) == 0:
            err("merge_both_empty",
                f"Both merge zones are empty: zone {zl} and zone {zr}")

    def _validate_shuttle(self, step, op: ShuttleOp, state: _ValidatorState, err):
        """ShuttleOp の事前検証"""
        ion_id = op.target

        # dst はLocation or tuple
        if hasattr(op.dst, 'trap'):
            dst_zone = op.dst.trap
        else:
            dst_zone = op.dst[0]

        # イオンの存在確認
        src = state.locate(ion_id)
        if src is None:
            err("shuttle_ion_not_found", f"Ion {ion_id} not found in any zone")
            return

        # 隣接チェック（展開後のShuttleは1ホップ）
        if abs(src - dst_zone) != 1:
            err("shuttle_adjacency",
                f"Shuttle from zone {src} to zone {dst_zone} is not adjacent "
                f"(distance={abs(src - dst_zone)})")

        # dstゾーンの存在
        if self.architecture.get_zone(dst_zone) is None:
            err("shuttle_invalid_zone",
                f"Destination zone {dst_zone} does not exist")

    def _validate_swap(self, step, op: SwapOp, state: _ValidatorState, err):
        """SwapOp の事前検証"""
        ion_a, ion_b = op.targets

        # イオンの存在確認
        zone_a = state.locate(ion_a)
        zone_b = state.locate(ion_b)
        if zone_a is None:
            err("swap_ion_not_found", f"Ion {ion_a} not found in any zone")
            return
        if zone_b is None:
            err("swap_ion_not_found", f"Ion {ion_b} not found in any zone")
            return

        # 同一ゾーンチェック
        if zone_a != zone_b:
            err("swap_different_zones",
                f"Swap targets in different zones: "
                f"ion {ion_a} in zone {zone_a}, ion {ion_b} in zone {zone_b}")
            return

        # Temp Zone 禁止
        if self.architecture.is_temp(zone_a):
            err("swap_in_temp",
                f"Swap in Temporary zone {zone_a} is not allowed")

        # 2イオン制約: Swap実行時はゾーンに2イオンのみ
        contents = state.zone_contents[zone_a]
        if len(contents) != 2:
            err("swap_not_two_ions",
                f"Swap requires exactly 2 ions in zone {zone_a}, "
                f"but found {len(contents)}: {contents}")

        # 隣接チェック
        pos_a = contents.index(ion_a)
        pos_b = contents.index(ion_b)
        if abs(pos_a - pos_b) != 1:
            err("swap_not_adjacent",
                f"Swap targets not adjacent in zone {zone_a}: "
                f"ion {ion_a} at pos {pos_a}, ion {ion_b} at pos {pos_b}")

    def _validate_gate(self, step, op, state: _ValidatorState, err):
        """GateOp/MeasureOp/ResetOp の事前検証"""
        type_name = type(op).__name__

        if type_name == "GateOp":
            targets = op.targets
        elif type_name == "MeasureOp":
            targets = op.targets
        elif type_name == "ResetOp":
            targets = op.targets
        else:
            return

        # 各ターゲットがLIZにいるか
        for ion_id in targets:
            zone_id = state.locate(ion_id)
            if zone_id is None:
                err("gate_ion_not_found",
                    f"Ion {ion_id} not found in any zone")
                continue
            if not self.architecture.is_liz(zone_id):
                err("gate_not_in_liz",
                    f"Ion {ion_id} is in zone {zone_id} "
                    f"({self.architecture.get_zone(zone_id).zone_type}), not in LIZ")

        # 全ターゲットが同一LIZにいるか（2量子ビットゲートの場合）
        if len(targets) >= 2:
            zones = [state.locate(t) for t in targets]
            zones = [z for z in zones if z is not None]
            if len(set(zones)) > 1:
                err("gate_different_liz",
                    f"Gate targets in different zones: {dict(zip(targets, zones))}")

    def _validate_cool(self, step, op, state: _ValidatorState, err):
        """CoolOp の事前検証: ゾーンの存在 + coolant の所在"""
        zone = self.architecture.get_zone(op.zone_id)
        if zone is None:
            err("cool_zone_not_found",
                f"Cool target zone {op.zone_id} does not exist")
            return
        # Temp zone は操作前後で空という不変条件があるため冷却対象にできない
        if zone.zone_type == "T":
            err("cool_in_temp",
                f"Cool cannot target Temporary zone {op.zone_id}")
        if op.coolant_id >= 0:
            coolant_zone = state.locate(op.coolant_id)
            if coolant_zone is None:
                err("cool_coolant_not_found",
                    f"Coolant ion {op.coolant_id} not found in any zone")
            elif coolant_zone != op.zone_id:
                err("cool_coolant_not_in_zone",
                    f"Coolant ion {op.coolant_id} is in zone {coolant_zone}, "
                    f"expected zone {op.zone_id}")


# ==========================================
# 並列 Validator
# ==========================================

class ParallelValidator:
    """
    TimeStep 単位の並列命令列を検証する

    Sequential Validator との違い:
    1. 同一TimeStep内のops間にゾーン競合がないか検証
    2. 同一TimeStep内のopsが同一ionを操作していないか検証
    3. 全opsをTimeStep開始時のスナップショットに対して検証
       （Sequential Validatorは前のopの結果を次のopが見る）
    """

    def __init__(self, architecture: QCCDArchitecture):
        self.architecture = architecture
        self._seq_validator = Validator(architecture)

    def _zones_used(self, op: Operation) -> Set[int]:
        """1つのopが使用するゾーンを返す（Schedulerの_zones_usedと同等）"""
        if isinstance(op, SplitOp):
            zones = {op.src_zone_id, op.dst_zone_id}
            zones.discard(-1)
            return zones
        if isinstance(op, MergeOp):
            return {op.zone_left, op.zone_right}
        if isinstance(op, ShuttleOp):
            zones = set()
            if op.src_zone != -1:
                zones.add(op.src_zone)
            dst_zone = op.dst.trap if hasattr(op.dst, 'trap') else op.dst[0]
            if dst_zone != -1:
                zones.add(dst_zone)
            return zones
        if isinstance(op, SwapOp):
            return {op.zone_id} if hasattr(op, 'zone_id') and op.zone_id != -1 else set()
        if isinstance(op, (GateOp, MeasureOp, ResetOp)):
            return set()  # ゲートはLIZ上で実行、ゾーンはlocateで取得
        if isinstance(op, CoolOp):
            return {op.zone_id} if op.zone_id != -1 else set()
        return set()

    def _ions_used(self, op: Operation) -> Set[int]:
        """1つのopが操作するionのIDを返す"""
        if isinstance(op, SplitOp):
            return {op.ion_id}
        if isinstance(op, ShuttleOp):
            return {op.target}
        if isinstance(op, (SwapOp, GateOp, MeasureOp, ResetOp)):
            return set(op.targets)
        if isinstance(op, MergeOp):
            return set()  # Mergeはzone操作、特定ionを指定しない
        if isinstance(op, CoolOp):
            return {op.coolant_id} if op.coolant_id >= 0 else set()
        return set()

    def validate(
        self,
        timesteps: List[TimeStep],
        initial_placement: Dict[int, int],
    ) -> ValidationResult:
        """
        TimeStep列を検証する

        Args:
            timesteps: 並列TimeStep列
            initial_placement: ion_id -> zone_id の初期配置

        Returns:
            ValidationResult
        """
        state = _ValidatorState(self.architecture)
        for ion_id, zone_id in initial_placement.items():
            state.place_ion(ion_id, zone_id)

        result = ValidationResult()
        expected_ions = state.all_ions().copy()
        global_step = 0

        # 初期配置の容量チェック
        for zone in self.architecture.zones:
            count = state.zone_count(zone.id)
            if count > zone.capacity:
                result.errors.append(ValidationError(
                    step=-1,
                    operation=None,
                    constraint="zone_capacity",
                    message=f"Initial placement exceeds zone {zone.id} capacity: "
                            f"{count} > {zone.capacity}",
                ))

        for ts_idx, ts in enumerate(timesteps):
            ops = ts.ops

            # --- 並列固有の検証（TimeStep開始時の状態で） ---

            # 1. ゾーン競合チェック
            for i in range(len(ops)):
                zones_i = self._zones_used(ops[i])
                for j in range(i + 1, len(ops)):
                    zones_j = self._zones_used(ops[j])
                    overlap = zones_i & zones_j
                    if overlap:
                        result.errors.append(ValidationError(
                            step=global_step + i,
                            operation=ops[i],
                            constraint="parallel_zone_conflict",
                            message=f"TS{ts_idx}: ops[{i}] and ops[{j}] share "
                                    f"zone(s) {overlap}. "
                                    f"op_i={ops[i]}, op_j={ops[j]}",
                        ))

            # 2. 同一ion操作チェック
            for i in range(len(ops)):
                ions_i = self._ions_used(ops[i])
                for j in range(i + 1, len(ops)):
                    ions_j = self._ions_used(ops[j])
                    overlap = ions_i & ions_j
                    if overlap:
                        result.errors.append(ValidationError(
                            step=global_step + i,
                            operation=ops[i],
                            constraint="parallel_ion_conflict",
                            message=f"TS{ts_idx}: ops[{i}] and ops[{j}] operate "
                                    f"on same ion(s) {overlap}. "
                                    f"op_i={ops[i]}, op_j={ops[j]}",
                        ))

            # 3. 各opをTimeStep開始時のスナップショットに対して検証
            snapshot = state.snapshot()
            for i, op in enumerate(ops):
                # スナップショットからの一時stateで検証
                temp_state = _ValidatorState(self.architecture)
                for zid, ions in snapshot.items():
                    for ion in ions:
                        temp_state.place_ion(ion, zid)
                self._seq_validator._validate_step(
                    global_step + i, op, temp_state, expected_ions, result
                )

            # 4. 全opsを実stateに適用（順序は任意、並列なので）
            for i, op in enumerate(ops):
                self._seq_validator._validate_step(
                    global_step + i, op, state, expected_ions,
                    ValidationResult()  # エラーは既に検出済み、状態更新のみ
                )

            global_step += len(ops)

        return result


# ==========================================
# 便利関数
# ==========================================

def validate_operations(
    operations: List[Operation],
    initial_placement: Dict[int, int],
    architecture: QCCDArchitecture,
    verbose: bool = True,
) -> ValidationResult:
    """
    命令列を検証するヘルパー関数（Sequential）

    Args:
        operations: 命令列
        initial_placement: 初期配置
        architecture: アーキテクチャ
        verbose: 結果を表示するか

    Returns:
        ValidationResult
    """
    validator = Validator(architecture)
    result = validator.validate(operations, initial_placement)

    if verbose:
        print(result.summary())

    return result


def validate_timesteps(
    timesteps: List[TimeStep],
    initial_placement: Dict[int, int],
    architecture: QCCDArchitecture,
    verbose: bool = True,
) -> ValidationResult:
    """
    並列TimeStep列を検証するヘルパー関数

    Args:
        timesteps: TimeStep列
        initial_placement: 初期配置
        architecture: アーキテクチャ
        verbose: 結果を表示するか

    Returns:
        ValidationResult
    """
    validator = ParallelValidator(architecture)
    result = validator.validate(timesteps, initial_placement)

    if verbose:
        print(result.summary())

    return result
