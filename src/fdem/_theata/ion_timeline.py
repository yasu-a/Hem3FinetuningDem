# Adapted from THeatA for the bundled workbench.
"""Per-ion event timeline — コンパイル結果から各イオンの物理タイムラインを構築する。

目的 (docs/ 及び memory の multi-LIZ Stim 設計方針):
  ノイズ挿入を「ゲート実行軸」(各ゲートステップで非対象 qubit に Z_ERROR) から
  「イオン軸」(各イオンの連続 op 間の物理ギャップ Δt に 1 個の Z_ERROR) へ移行する。
  ギャップ方式は 1-LIZ でも本来正しく、per-LIZ stream (複数 core の実時間が重なる)
  では必須になる。

データソースは 2 系統:
  - CompileResult.timesteps (1-LIZ / global scheduler): timestep 累積で絶対時刻
  - PerLIZResult.segments (multi-LIZ streams): estimate_streams 済みの segment が
    sync 解決済みの共通時計絶対時刻 (start_us) を持つので、core 間の突き合わせは不要

イベントの定義 (旧 noise injection の会計と整合させる):
  - busy (イオンがゲート対象) = GateOp(MS/R), ResetOp, MeasureOp の targets
  - Swap/Shuttle/Split/Merge の時間はイオンにとって idle 扱い
    (移動中も dephasing するという物理。旧方式でも transport は wait に計上)
  - anchor (Stim 論理 op と 1:1 で対応し、エラー挿入点になる op)
    = ResetOp / GateOp('MS') / MeasureOp   (stim_recompiler の fold-up と対応)
  - GateOp('R') は busy イベントだが anchor ではない (recompile で H に fold される
    基底変換。その手前の idle は次の anchor まで持ち越す)

ギャップ規則:
  - gap = anchor 直前までに累積した idle 時間 (自分の busy 時間は除く)
  - ResetOp anchor ではギャップを **破棄** する (reset 前の Z エラーは reset に
    吸収されるため挿入不要)。以降のギャップは reset 後から数え直す
  - 複数ラウンドは max 一律オフセット (ラウンド境界で全 core 同期) で連結する

検証 (タイムラインの整合性 assert):
  - 同一イオンの busy 区間は時間的に重ならない (race-freedom の実測確認)
  - 各イオンの時刻順イベント列は、canonical linearization (flat_ops /
    merged_flat) 上の出現順と一致する (cross-LIZ 順序の検証)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .ast_def import (
    Operation, GateOp, ResetOp, MeasureOp, TimeStep,
)
from .parallel_estimator import ParallelEstimator
from .estimator import TimeParameters

_EPS = 1e-9


# ==========================================
# データ型
# ==========================================

@dataclass
class OpInterval:
    """1 つの op の絶対時刻区間。"""
    op: Operation
    t0: float
    t1: float
    core: Optional[int] = None   # multi-LIZ のとき LIZ zone id


@dataclass
class IonEvent:
    """あるイオンが busy な区間 (= そのイオンを target とするゲート系 op)。"""
    ion: int
    t0: float
    t1: float
    op: Operation
    core: Optional[int] = None

    @property
    def is_anchor(self) -> bool:
        return is_anchor_op(self.op)


@dataclass
class IonGaps:
    """per-ion ギャップ計算の結果。

    gaps_by_op: id(anchor_op) -> {ion: gap_us}
        anchor の直前に挿入すべき per-ion idle 時間。ResetOp は含まれない
        (吸収規則)。gap 0 のイオンは省略される。
    trailing_idle: ion -> 最後のイベント終了から t_end までの idle 時間
        (最終 transversal 測定を回路に足す場合、その直前ギャップとして使える)
    t_end: タイムライン終端 (wall clock)
    """
    gaps_by_op: Dict[int, Dict[int, float]] = field(default_factory=dict)
    trailing_idle: Dict[int, float] = field(default_factory=dict)
    t_end: float = 0.0


# ==========================================
# 基本述語
# ==========================================

def is_anchor_op(op: Operation) -> bool:
    """Stim 論理 op (R/CX/M) と 1:1 対応するネイティブ op か。"""
    if isinstance(op, (ResetOp, MeasureOp)):
        return True
    if isinstance(op, GateOp) and op.type == 'MS':
        return True
    return False


def _busy_ions(op: Operation) -> List[int]:
    """イオンが busy になる op の target リスト (それ以外は空)。"""
    if isinstance(op, (ResetOp, MeasureOp)):
        return list(op.targets)
    if isinstance(op, GateOp):
        return list(op.targets)
    return []


# ==========================================
# op 絶対時刻の付与
# ==========================================

def op_intervals_from_timesteps(
    timesteps: List[TimeStep],
    time_params: Optional[TimeParameters] = None,
    t_offset: float = 0.0,
) -> Tuple[List[OpInterval], float]:
    """global TimeStep 列 (1-LIZ / Scheduler 出力) から op 絶対時刻を計算する。

    ParallelEstimator と同じ会計: timestep 内の全 op は timestep 開始と同時に
    始まり、timestep の長さは max(op 時間)。

    Returns:
        (intervals, t_end): t_end = t_offset + 全 timestep の並列時間合計
    """
    est = ParallelEstimator(time_params) if time_params else ParallelEstimator()
    intervals: List[OpInterval] = []
    t = t_offset
    for ts in timesteps:
        if not ts.ops:
            continue
        durs = [est._get_operation_time(op)[1] for op in ts.ops]
        for op, d in zip(ts.ops, durs):
            intervals.append(OpInterval(op=op, t0=t, t1=t + d))
        t += max(durs)
    return intervals, t


def op_intervals_from_streams(
    per_liz_result,
    time_params: Optional[TimeParameters] = None,
    t_offset: float = 0.0,
) -> Tuple[List[OpInterval], float]:
    """PerLIZResult (estimate_streams 済み) から op 絶対時刻を計算する。

    segment.start_us は sync 辺を解決した共通時計の絶対時刻なので、
    core 間の突き合わせは不要。segment 内は timesteps の累積。

    Returns:
        (intervals, t_end): t_end = t_offset + wall_clock (= max per-core end)
    """
    est = ParallelEstimator(time_params) if time_params else ParallelEstimator()
    intervals: List[OpInterval] = []
    for seg in per_liz_result.segments:
        if not seg.timesteps and seg.ops:
            raise ValueError(
                "segment.timesteps が空です。estimate_streams(result) を先に"
                "実行してから op_intervals_from_streams を呼んでください。")
        t = t_offset + seg.start_us
        for ts in seg.timesteps:
            if not ts.ops:
                continue
            durs = [est._get_operation_time(op)[1] for op in ts.ops]
            for op, d in zip(ts.ops, durs):
                intervals.append(OpInterval(op=op, t0=t, t1=t + d,
                                            core=seg.core_id))
            t += max(durs)
    return intervals, t_offset + per_liz_result.wall_clock_us


# ==========================================
# per-ion タイムライン
# ==========================================

def build_ion_events(intervals: List[OpInterval]) -> Dict[int, List[IonEvent]]:
    """op 区間列を per-ion の busy イベント列 (時刻順) に射影する。

    同一イオンの busy 区間が重なっていたら AssertionError
    (per-LIZ stream の race-freedom が破れている兆候)。
    """
    events: Dict[int, List[IonEvent]] = {}
    for iv in intervals:
        for ion in _busy_ions(iv.op):
            events.setdefault(ion, []).append(
                IonEvent(ion=ion, t0=iv.t0, t1=iv.t1, op=iv.op, core=iv.core))
    for ion, evs in events.items():
        evs.sort(key=lambda e: (e.t0, e.t1))
        for prev, cur in zip(evs, evs[1:]):
            assert cur.t0 >= prev.t1 - _EPS, (
                f"ion {ion} の busy 区間が重複: "
                f"{type(prev.op).__name__}[{prev.t0:.2f},{prev.t1:.2f}] と "
                f"{type(cur.op).__name__}[{cur.t0:.2f},{cur.t1:.2f}]")
    return events


def validate_order_against(
    ion_events: Dict[int, List[IonEvent]],
    linear_ops: List[Operation],
) -> None:
    """各イオンの時刻順イベント列が linearization 上の出現順と一致するか検証。

    per-LIZ stream では merged_flat を渡す。cross-LIZ 共有イオンの
    「sync 辺の向きと併合順の整合」がここで実測検証される。
    """
    pos = {id(op): i for i, op in enumerate(linear_ops)}
    for ion, evs in ion_events.items():
        last_pos = -1
        for ev in evs:
            p = pos.get(id(ev.op))
            assert p is not None, (
                f"ion {ion} のイベント op が linearization に存在しません: "
                f"{type(ev.op).__name__} t=[{ev.t0:.2f},{ev.t1:.2f}]")
            assert p > last_pos, (
                f"ion {ion} の時刻順イベントが linearization 順と食い違います: "
                f"op@{p} (t0={ev.t0:.2f}) が前のイベント (pos {last_pos}) より前")
            last_pos = p


def compute_ion_gaps(
    ion_events: Dict[int, List[IonEvent]],
    t_start: float = 0.0,
    t_end: float = 0.0,
) -> IonGaps:
    """anchor ごとの per-ion idle ギャップを計算する。

    各イオンについて時刻順にイベントを走査し、busy でない時間を累積する。
    anchor に到達したら:
      - ResetOp     → 累積を破棄 (reset 吸収規則)、挿入なし
      - MS/MEASURE  → gaps_by_op[id(op)][ion] = 累積 idle
    非 anchor イベント (GateOp 'R') はその手前の idle を持ち越し、
    自身の busy 時間だけ累積から外す。

    不変条件: 各イオンについて
      Σ(gaps) + Σ(busy) + Σ(reset で吸収された idle) + trailing_idle
        == t_end − t_start
    """
    result = IonGaps(t_end=t_end)
    for ion, evs in ion_events.items():
        acc = 0.0
        last_end = t_start
        for ev in evs:
            acc += max(0.0, ev.t0 - last_end)
            if is_anchor_op(ev.op):
                if isinstance(ev.op, ResetOp):
                    acc = 0.0     # reset に吸収
                else:
                    if acc > _EPS:
                        result.gaps_by_op.setdefault(id(ev.op), {})[ion] = acc
                    acc = 0.0
            last_end = max(last_end, ev.t1)
        result.trailing_idle[ion] = acc + max(0.0, t_end - last_end)
    return result


# ==========================================
# 複数ラウンド連結 (max 一律オフセット)
# ==========================================

def gaps_from_compile_results(
    compile_results: List,
    time_params: Optional[TimeParameters] = None,
    mode: str = "lockstep",
) -> Tuple[List[Operation], IonGaps]:
    """複数ラウンドの CompileResult (1-LIZ / global scheduler) からギャップを構築。

    ラウンド境界は max 一律オフセット (= 前ラウンドの並列時間合計) で同期連結。
    バリア待ちや ラウンド跨ぎの idle は次ラウンド最初の anchor のギャップに
    自動的に計上される。

    Args:
        mode: "lockstep" = 既存 timestep モデル /
              "event" = イベント駆動 (event_scheduler.schedule_events) の
              連続時間区間からギャップを構築する。wall が短くなる分
              per-ion idle (dephasing) が減る。

    Returns:
        (anchor_ops, ion_gaps):
          anchor_ops = 全ラウンドの flat_ops から anchor を回路順に並べたリスト
                       (stim_recompiler の Stim 論理 op 順と 1:1)
    """
    if mode not in ("lockstep", "event"):
        raise ValueError(f"mode must be 'lockstep' or 'event', got {mode}")
    intervals: List[OpInterval] = []
    offset = 0.0
    linear: List[Operation] = []
    for cr in compile_results:
        if mode == "event":
            # 遅延 import (event_scheduler は本モジュールを import するため)
            from .event_scheduler import schedule_events
            ivs, offset = schedule_events(
                cr.flat_ops, time_params, t_offset=offset)
        else:
            ivs, offset = op_intervals_from_timesteps(
                cr.timesteps, time_params, t_offset=offset)
        intervals.extend(ivs)
        linear.extend(cr.flat_ops)

    events = build_ion_events(intervals)
    validate_order_against(events, linear)
    gaps = compute_ion_gaps(events, t_start=0.0, t_end=offset)

    anchor_ops = [op for op in linear if is_anchor_op(op)]
    return anchor_ops, gaps


def gaps_from_per_liz_results(
    per_liz_results: List,
    time_params: Optional[TimeParameters] = None,
) -> Tuple[List[Operation], IonGaps]:
    """複数ラウンドの PerLIZResult (multi-LIZ streams) からギャップを構築。

    各ラウンドは estimate_streams 済みであること。ラウンド境界は
    max 一律オフセット (= wall_clock_us) で同期連結する。

    Returns:
        (anchor_ops, ion_gaps): anchor_ops は merged_flat (canonical
        linearization) 順。Stim 生成も同じ merged_flat から行うこと。
    """
    intervals: List[OpInterval] = []
    offset = 0.0
    linear: List[Operation] = []
    for r in per_liz_results:
        ivs, offset = op_intervals_from_streams(r, time_params, t_offset=offset)
        intervals.extend(ivs)
        linear.extend(r.merged_flat)

    events = build_ion_events(intervals)
    validate_order_against(events, linear)
    gaps = compute_ion_gaps(events, t_start=0.0, t_end=offset)

    anchor_ops = [op for op in linear if is_anchor_op(op)]
    return anchor_ops, gaps


# ==========================================
# 可視化 / ダンプ用 (optional)
# ==========================================

def timeline_to_dict(ion_events: Dict[int, List[IonEvent]]) -> Dict:
    """timeline.json 用の serializable dict (ion -> [(t0, t1, kind, core)])。"""
    out = {}
    for ion, evs in sorted(ion_events.items()):
        out[str(ion)] = [
            {
                't0': ev.t0, 't1': ev.t1,
                'kind': (ev.op.type if isinstance(ev.op, GateOp)
                         else type(ev.op).__name__.replace('Op', '').upper()),
                'core': ev.core,
            }
            for ev in evs
        ]
    return out
