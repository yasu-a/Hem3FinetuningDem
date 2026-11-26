# Adapted from THeatA for the bundled workbench.
"""
QCCD Parallel Time Estimator

TimeStep構造を解釈して、並列実行を考慮した実行時間を推定する。

Sequential Estimator (estimator.py):
    全操作の時間を単純に足し算 → 上限値（並列性ゼロの場合）

Parallel Estimator (このファイル):
    各TimeStepの実行時間 = max(各opの所要時間)
    合計時間 = Σ max(TimeStep内のop時間)
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional
from .ast_def import (
    Operation, TimeStep, GateOp, ShuttleOp, SwapOp,
    SplitOp, MergeOp, MeasureOp, ResetOp, BarrierOp, CoolOp
)
from .estimator import TimeParameters


@dataclass
class PerLIZEstimationResult:
    """LIZごとに ops を振り分けて、各 LIZ の serial 時間と
    全 LIZ 並列を仮定した total 時間を推定した結果。

    Multi-LIZ 構成で Scheduler の TimeStep が捉えきれない LIZ 間の並列性を
    補完するための補助 estimator。
    """
    per_liz_time: Dict[int, float] = field(default_factory=dict)   # liz_zone_id -> serial time
    per_liz_op_count: Dict[int, int] = field(default_factory=dict)  # liz_zone_id -> ops count
    total_time_max: float = 0.0                                     # max over per-liz times（並列上限）
    total_time_sum: float = 0.0                                     # sum over per-liz times（参考、ほぼ sequential）
    unassigned_time: float = 0.0                                    # どの LIZ にも attrit できなかった ops の時間

    def __repr__(self):
        lines = [
            f"Per-LIZ Parallel Time Estimate",
            f"  Total (max over LIZ): {self.total_time_max:.2f} μs ({self.total_time_max/1000:.2f} ms)",
            f"  Sum   (reference)   : {self.total_time_sum:.2f} μs",
            f"  Unassigned          : {self.unassigned_time:.2f} μs",
            "  Per-LIZ breakdown:",
        ]
        for liz_id in sorted(self.per_liz_time.keys()):
            t = self.per_liz_time[liz_id]
            n = self.per_liz_op_count[liz_id]
            lines.append(f"    LIZ {liz_id}: {t:.2f} μs ({t/1000:.2f} ms),  {n} ops")
        return "\n".join(lines)


@dataclass
class ParallelEstimationResult:
    """並列時間推定の結果"""
    total_time: float                         # 並列考慮後の合計時間（μs）
    sequential_time: float                    # 逐次実行の合計時間（μs）
    total_timesteps: int                      # TimeStep数
    total_operations: int                     # 総操作数
    breakdown: Dict[str, float]               # 操作種別ごとの時間（逐次合計）
    counts: Dict[str, int]                    # 操作種別ごとのカウント
    timestep_details: List[Dict]              # 各TimeStepの詳細

    @property
    def speedup(self) -> float:
        """逐次実行に対する高速化率"""
        if self.total_time == 0:
            return 1.0
        return self.sequential_time / self.total_time

    @property
    def parallelism(self) -> float:
        """平均並列度（ops / timestep）"""
        if self.total_timesteps == 0:
            return 0.0
        return self.total_operations / self.total_timesteps

    def __repr__(self):
        lines = [
            f"Parallel Time: {self.total_time:.2f} μs ({self.total_time/1000:.2f} ms)",
            f"Sequential Time: {self.sequential_time:.2f} μs ({self.sequential_time/1000:.2f} ms)",
            f"Speedup: {self.speedup:.2f}x",
            f"TimeSteps: {self.total_timesteps}  |  Operations: {self.total_operations}  |  Avg parallelism: {self.parallelism:.2f}",
            "",
            "Breakdown by operation type:",
        ]
        for op_type, time in sorted(self.breakdown.items(), key=lambda x: -x[1]):
            count = self.counts.get(op_type, 0)
            pct = (time / self.sequential_time * 100) if self.sequential_time > 0 else 0
            lines.append(f"  {op_type:12s}: {time:10.2f} μs ({pct:5.1f}%) x {count}")
        return "\n".join(lines)


class ParallelEstimator:
    """
    TimeStep構造を解釈して並列実行時間を推定

    使用例:
        estimator = ParallelEstimator()
        result = estimator.estimate(compile_result.timesteps)
        print(result)
    """

    def __init__(self, params: Optional[TimeParameters] = None):
        self.params = params or TimeParameters()

    def estimate(self, timesteps: List[TimeStep]) -> ParallelEstimationResult:
        """
        TimeStep列から並列実行時間を推定

        各TimeStepの実行時間 = max(各opの所要時間)
        合計 = Σ max(TimeStep内のop時間)

        フォノン追跡は QCCDState 側で行われ、CompileResult.phonon_data に格納される。
        """
        total_time = 0.0
        sequential_time = 0.0
        breakdown: Dict[str, float] = {}
        counts: Dict[str, int] = {}
        timestep_details: List[Dict] = []
        total_ops = 0

        for ts_idx, ts in enumerate(timesteps):
            step_max_time = 0.0
            step_ops_info = []

            for op in ts.ops:
                op_type, op_time = self._get_operation_time(op)
                total_ops += 1

                breakdown[op_type] = breakdown.get(op_type, 0.0) + op_time
                counts[op_type] = counts.get(op_type, 0) + 1
                sequential_time += op_time

                if op_time > step_max_time:
                    step_max_time = op_time

                step_ops_info.append({
                    "type": op_type,
                    "time": op_time,
                    "detail": self._op_summary(op),
                })

            total_time += step_max_time

            timestep_details.append({
                "index": ts_idx,
                "parallel_time": step_max_time,
                "ops_count": len(ts.ops),
                "ops": step_ops_info,
            })

        return ParallelEstimationResult(
            total_time=total_time,
            sequential_time=sequential_time,
            total_timesteps=len(timesteps),
            total_operations=total_ops,
            breakdown=breakdown,
            counts=counts,
            timestep_details=timestep_details,
        )

    def estimate_per_liz(
        self,
        timesteps: List[TimeStep],
        architecture,
    ) -> PerLIZEstimationResult:
        """Per-LIZ 時間推定（multi-LIZ 用の補助 estimator）

        Scheduler の TimeStep が捉えきれない LIZ 間の並列性を補完する。
        各 LIZ について、その LIZ のドメインに属する op を input 順に逐次合計。

        Algorithm:
            1. 各ゾーンを最近隣 LIZ に attrit する zone→LIZ マッピング構築
               （同距離なら左の LIZ）
            2. 各 op について、それが触るゾーン群から所属 LIZ を投票で決定
            3. 各 LIZ の op 群について duration を逐次合計
            4. Total = max(per-LIZ times)
        """
        liz_zones = [z.id for z in architecture.zones if z.zone_type == "L"]
        if not liz_zones:
            return PerLIZEstimationResult()

        # Zone -> nearest LIZ
        zone_to_liz: Dict[int, int] = {}
        for z in architecture.zones:
            best_liz = liz_zones[0]
            best_dist = abs(z.id - best_liz)
            for liz in liz_zones[1:]:
                d = abs(z.id - liz)
                if d < best_dist:
                    best_dist = d
                    best_liz = liz
            zone_to_liz[z.id] = best_liz

        per_liz_time: Dict[int, float] = {liz: 0.0 for liz in liz_zones}
        per_liz_count: Dict[int, int] = {liz: 0 for liz in liz_zones}
        unassigned_time = 0.0

        # Use zone_used logic from Scheduler to determine zones an op touches
        from .qccd_compiler import Scheduler
        scheduler = Scheduler()

        for ts in timesteps:
            for op in ts.ops:
                _, op_time = self._get_operation_time(op)
                zones = scheduler._zones_used(op)
                if not zones:
                    unassigned_time += op_time
                    continue
                # Vote: which LIZ does this op touch most?
                liz_votes: Dict[int, int] = {}
                for z in zones:
                    if z in zone_to_liz:
                        liz = zone_to_liz[z]
                        liz_votes[liz] = liz_votes.get(liz, 0) + 1
                if not liz_votes:
                    unassigned_time += op_time
                    continue
                # Tie-break: lower-index LIZ
                best_liz = max(liz_votes.keys(), key=lambda l: (liz_votes[l], -l))
                per_liz_time[best_liz] += op_time
                per_liz_count[best_liz] += 1

        total_max = max(per_liz_time.values()) if per_liz_time else 0.0
        total_sum = sum(per_liz_time.values())

        return PerLIZEstimationResult(
            per_liz_time=per_liz_time,
            per_liz_op_count=per_liz_count,
            total_time_max=total_max,
            total_time_sum=total_sum,
            unassigned_time=unassigned_time,
        )

    def _get_operation_time(self, op: Operation) -> tuple:
        """操作の種類と時間を返す"""
        if isinstance(op, GateOp):
            if op.type == "MS":
                return ("MS", self.params.ms_gate)
            elif op.type == "R":
                return ("R", self.params.r_gate)
            else:
                return (op.type, self.params.ms_gate)
        elif isinstance(op, ResetOp):
            return ("Reset", self.params.reset)
        elif isinstance(op, MeasureOp):
            return ("Measure", self.params.measure)
        elif isinstance(op, SplitOp):
            return ("Split", self.params.split)
        elif isinstance(op, MergeOp):
            return ("Merge", self.params.merge)
        elif isinstance(op, SwapOp):
            return ("Swap", self.params.swap)
        elif isinstance(op, ShuttleOp):
            return ("Shuttle", self.params.shuttle_time(dst_chain_length=op.dst_chain_length))
        elif isinstance(op, CoolOp):
            dur = op.duration_us if op.duration_us is not None else self.params.cool
            return ("Cool", dur)
        elif isinstance(op, BarrierOp):
            return ("Barrier", 0.0)
        else:
            return ("Unknown", 0.0)

    def _op_summary(self, op: Operation) -> str:
        """操作の短い要約文字列"""
        if isinstance(op, GateOp):
            ions = ",".join(str(t) for t in op.targets)
            return f"{op.type}({ions}) z{op.zone_id}"
        elif isinstance(op, ResetOp):
            ions = ",".join(str(t) for t in op.targets)
            return f"Reset({ions}) z{op.zone_id}"
        elif isinstance(op, MeasureOp):
            ions = ",".join(str(t) for t in op.targets)
            return f"Measure({ions}) z{op.zone_id}"
        elif isinstance(op, SplitOp):
            return f"Split(ion{op.ion_id}) z{op.src_zone_id}->z{op.dst_zone_id}"
        elif isinstance(op, MergeOp):
            return f"Merge z{op.zone_left}+z{op.zone_right}"
        elif isinstance(op, SwapOp):
            ions = ",".join(str(t) for t in op.targets)
            return f"Swap({ions}) z{op.zone_id}"
        elif isinstance(op, ShuttleOp):
            return f"Shuttle(ion{op.target}) z{op.src_zone}->{op.dst}"
        elif isinstance(op, CoolOp):
            coolant = f" coolant={op.coolant_id}" if op.coolant_id >= 0 else ""
            return f"Cool z{op.zone_id}{coolant}"
        else:
            return str(op)


# ==========================================
# 出力関数
# ==========================================

def format_timesteps_text(
    timesteps: List[TimeStep],
    estimator: Optional[ParallelEstimator] = None,
) -> str:
    """
    TimeStep列を人間が読めるテキスト形式に変換

    並列実行されているopsが視覚的に分かる形式:
        TS  0 [100.00μs] Split(ion8) z1→z2  |  Split(ion2) z5→z4
        TS  1 [111.11μs] Shuttle(ion8) z2→(3, 0)
        TS  2 [100.00μs] Merge z2+z3
        TS  3 [ 59.00μs] MS(8,2) z3
    """
    if estimator is None:
        estimator = ParallelEstimator()

    result = estimator.estimate(timesteps)
    lines = []

    # ヘッダー
    lines.append(f"# TimeSteps: {result.total_timesteps}  |  "
                 f"Operations: {result.total_operations}  |  "
                 f"Avg parallelism: {result.parallelism:.2f}")
    lines.append(f"# Parallel time: {result.total_time:.2f} μs ({result.total_time/1000:.2f} ms)  |  "
                 f"Sequential time: {result.sequential_time:.2f} μs ({result.sequential_time/1000:.2f} ms)  |  "
                 f"Speedup: {result.speedup:.2f}x")
    lines.append("")

    # 各TimeStep
    ts_width = len(str(result.total_timesteps - 1)) if result.total_timesteps > 0 else 1
    cumulative_time = 0.0

    for detail in result.timestep_details:
        ts_idx = detail["index"]
        par_time = detail["parallel_time"]
        ops = detail["ops"]
        cumulative_time += par_time

        n_ops = len(ops)
        ts_prefix = (
            f"TS {ts_idx:>{ts_width}} "
            f"[{par_time:>8.2f}μs  @{cumulative_time:>10.2f}μs]"
        )

        if n_ops == 1:
            # 単独ops: 1行表示
            lines.append(f"{ts_prefix}  {ops[0]['detail']}")
        else:
            # 並列ops: ブロック表示
            lines.append(f"{ts_prefix}  [{n_ops}P]")
            # 時間降順でソート（最長 = ボトルネックを先頭に）
            sorted_ops = sorted(ops, key=lambda o: -o["time"])
            for op in sorted_ops:
                lines.append(f"  | {op['detail']:<40s} {op['time']:>8.2f}μs")

    return "\n".join(lines)


def save_parallel_txt(
    timesteps: List[TimeStep],
    filepath: str,
    initial_placement: dict = None,
    arch=None,
    estimator: Optional[ParallelEstimator] = None,
):
    """TimeStep列をテキストファイルに保存"""
    from pathlib import Path

    if estimator is None:
        estimator = ParallelEstimator()

    lines = []

    # ヘッダー
    if arch:
        layout = " - ".join(f"z{z.id}:{z.zone_type}({z.capacity})" for z in arch.zones)
        lines.append(f"# Architecture: {layout}")
    if initial_placement:
        lines.append(f"# Initial placement: {initial_placement}")
    lines.append("")

    # 本体
    lines.append(format_timesteps_text(timesteps, estimator))

    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write("\n".join(lines) + "\n")

    print(f"[Output] Saved to: {filepath}")


# ==========================================
# 動作確認
# ==========================================

if __name__ == "__main__":
    from .qccd_parser import parse
    from .dag_builder import DAGBuilder
    from .qccd_compiler import QCCDArchitecture, QCCDCompiler
    from .estimator import Estimator

    print("=" * 70)
    print("Parallel Estimator Test")
    print("=" * 70)

    code = """
    QUBIT 0 AT (1, 0)
    QUBIT 1 AT (1, 1)
    QUBIT 2 AT (0, 0)

    RESET 0
    RESET 1
    RESET 2

    R 0 1.57 0.0
    R 1 1.57 0.0

    MS 0 1 0.25
    MS 1 2 0.25
    """

    program = parse(code)
    builder = DAGBuilder(program)
    dag = builder.build()

    arch = QCCDArchitecture.simple_mlm(liz_capacity=4)
    initial = {0: 1, 1: 1, 2: 1}

    compiler = QCCDCompiler(arch)
    result = compiler.compile(dag, initial)

    # Sequential estimation
    seq_est = Estimator().estimate(result.operations)
    print("\n--- Sequential Estimation ---")
    print(seq_est)

    # Parallel estimation
    par_est = ParallelEstimator()
    par_result = par_est.estimate(result.timesteps)
    print("\n--- Parallel Estimation ---")
    print(par_result)

    # TimeStep text output
    print("\n--- TimeStep View ---")
    print(format_timesteps_text(result.timesteps))
