# Adapted from THeatA for the bundled workbench.
"""Event-driven (continuous-time) scheduler.

flat_ops をゾーン空き時刻ベースで絶対時刻に割り当てる。既存 Scheduler
(ロックステップ timestep) と同じ競合モデル (ゾーンロック) の連続時間版:

  - 各 op は「自分が使うゾーン集合の空き時刻の max」で開始し、所要時間だけ占有
  - 同一ゾーンを使う op 同士は emission 順を保つ (Scheduler と同じ順序保証)
  - ゾーンを共有しない op は時間的に自由に重なる
    → Measure (1000 μs) 中に memory/temp 側の搬送チェーンが流れ込む

意味論 (搬送が Measure 終了までに完了しない場合):
  - op は非中断 (non-preemptible)。Measure 終了は同期点ではない
  - 後続 op は必要ゾーンが空いた時刻に各自開始する
    (RESET は LIZ が空いた瞬間、次のゲートはデータイオン到着まで待つ)
  - 「間に合わなかった」分は例外ではなく待ち時間として時間軸に現れる

安全性は既存 Scheduler と同等の論法:
  連続する同一イオンの op は、必ずそのイオンがその境界で居るゾーンを両方が
  ロックする (gate はイオンの居るゾーン、搬送系は src/dst の両方)。
  ゾーンごとの emission 順保存 ⇒ per-ion 順序保存。
  実測検証は ion_timeline.build_ion_events (busy 区間の非重複 assert) と
  validate_order_against (linearization 順との一致 assert) で行う。

フォノン会計は planner の仮想逐次状態で op 回数ベースに行われるため、本
スケジューラの影響を受けない。変わるのは wall clock と per-ion idle
(dephasing gap) のみ。
"""

from typing import Dict, List, Optional, Tuple

from .ast_def import Operation
from .qccd_compiler import Scheduler
from .parallel_estimator import ParallelEstimator
from .estimator import TimeParameters
from .ion_timeline import OpInterval


def schedule_events(
    flat_ops: List[Operation],
    time_params: Optional[TimeParameters] = None,
    t_offset: float = 0.0,
) -> Tuple[List[OpInterval], float]:
    """flat_ops (planner の emission 順) を絶対時刻区間に割り当てる。

    Args:
        flat_ops: MovementPlanner が生成した AtomicOp 列 (Scheduler 入力と同じ)
        time_params: op 所要時間パラメータ (None でデフォルト)
        t_offset: タイムライン開始時刻

    Returns:
        (intervals, t_end):
            intervals: flat_ops と同順の OpInterval 列 (core=None)
            t_end: 全 op の終了時刻の max (wall clock)
    """
    est = ParallelEstimator(time_params) if time_params else ParallelEstimator()
    zones_of = Scheduler()._zones_used

    zone_free: Dict[int, float] = {}
    intervals: List[OpInterval] = []
    t_end = t_offset

    for op in flat_ops:
        zones = zones_of(op)
        dur = est._get_operation_time(op)[1]
        if zones:
            start = max(zone_free.get(z, t_offset) for z in zones)
        else:
            # ゾーン注釈なし (Barrier 等、duration 0 のみ想定)。
            # 既存 Scheduler は ts0 に置くので、ここでも t_offset に置く。
            start = t_offset
        end = start + dur
        for z in zones:
            zone_free[z] = end
        intervals.append(OpInterval(op=op, t0=start, t1=end))
        if end > t_end:
            t_end = end

    return intervals, t_end
