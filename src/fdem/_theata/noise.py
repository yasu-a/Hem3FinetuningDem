"""Noise-injection routines adapted from the THeatA correlation workbench."""

import math

import stim

from .ast_def import GateOp, MeasureOp, ResetOp

def phonon_to_pauli_lookup_rates(
    phonon_at_gates: list,
    lookup,  # PauliErrorLookup instance
    mode: str = "strict",
    override_n_bar: float = None,
    phonon_scale: float = 1.0,
    warn_on_clip: bool = True,
) -> list:
    """各 MS ゲートのフォノン数 (transversal) を n_bar として lookup table を引き、
    15 個の Pauli 確率 (PAULI_CHANNEL_2 入力) を返す。

    Args:
        phonon_at_gates: compile_result.phonon_data['at_gates'] と同形式の list
        lookup: PauliErrorLookup インスタンス
        mode: 'strict' (ceil) / 'optimistic' (floor) / 'round_off' (nearest)
        override_n_bar: None なら通常通り phonon を見る。 数値 (例: 0.01) を渡すと
            **全 MS でこの n_bar を使う** (= heating 寄与 0 のシナリオ)。
        phonon_scale: 全 op 一律で transversal phonon に掛ける倍率。 1.0 = scale なし
            (現状)、 0.5 = ハードウェアで heating を半分に抑えた仮想シナリオ、 等。
            **すべての phonon 蓄積が加算的なため、 「compiler レベルで全 op の
            phonon 増分 rate を α 倍する」 と「最終 phonon を α 倍する」 は
            数学的に等価**。 後者の方が安価に試せるのでこちらで実装。
            (per-op-type の不等倍率は post-hoc では出来ないが、 当面は全op一律で十分)。
            override_n_bar が指定されたらこちらの方が優先される。
        warn_on_clip: False で table 上限超過の per-gate RuntimeWarning を抑制し、
            代わりに超過ゲート数の 1 行サマリを print する (クリップ動作は不変)。

    n_bar の取り方 (override_n_bar=None のとき):
        その MS 関与する 2 qubit の transversal phonon の max × phonon_scale。

    Returns:
        [(gate_name, pauli_probs_15, n_bar_used, max_axial, max_transversal), ...]
        ※ 最後の max_axial/transversal は **scale 前の生値** (記録用)
    """
    n_max = lookup.n_bar_range[1]
    n_clipped = 0
    worst_n = 0.0
    rates = []
    for gate_info in phonon_at_gates:
        phonons = gate_info['phonons']
        max_ax = max(p[0] for p in phonons.values())
        max_tr = max(p[1] for p in phonons.values())
        # phonon_scale を transversal にかける (heating 倍率制御)
        max_tr_scaled = max_tr * phonon_scale
        # override が指定されていればそちら優先、 そうでなければ scaled を使う
        n_input = override_n_bar if override_n_bar is not None else max_tr_scaled
        if n_input > n_max:
            n_clipped += 1
            worst_n = max(worst_n, n_input)
        n_bar_used = lookup.select_n_bar(n_input, mode=mode,
                                         warn_on_clip=warn_on_clip)
        # select_n_bar で警告済みのため lookup 側は常に抑制 (従来は同一ゲートで 2 重警告)
        pauli_probs = lookup.lookup(n_input, mode=mode, warn_on_clip=False)
        rates.append((gate_info['gate'], pauli_probs, n_bar_used, max_ax, max_tr))
    if n_clipped and not warn_on_clip:
        print(f"  [clip] {n_clipped}/{len(rates)} MS gates exceed lookup table "
              f"max n_bar={n_max:g} (worst {worst_n:.1f}); clipped to max")
    return rates


def default_noise_params() -> dict:
    """Default noise parameters for Approach B.

    The workbench can override the noise probabilities and idle rate.
    """
    return {
        # Op durations (μs)
        't_R_us': 4.0,
        't_MS_us': 59.0,
        't_meas_us': 1000.0,
        't_reset_us': 41.0,
        # Gate error probabilities
        'p_R': 1e-4,
        'p_MS_default': 1e-3,   # fallback when phonon-based not available
        'p_meas': 3.2e-3,
        'p_reset': 8e-4,
        # Idle spin dephasing rate Γ_φ in inverse seconds.
        'gamma_phi_per_s': 3.0,
    }


def compute_p_idle(duration_us: float, gamma_phi_per_s: float) -> float:
    """Z dephasing probability for `duration_us` of idle time.

    p_Z = (1 - exp(-2 Γ_φ t)) / 2, with t in seconds.
    Pure dephasing channel (Kraus: K_0 = √((1+e^{-2Γt})/2) I,
    K_1 = √((1-e^{-2Γt})/2) Z) → Z_ERROR(p_Z) in Stim.
    """
    t_sec = duration_us * 1e-6
    return (1.0 - math.exp(-2 * gamma_phi_per_s * t_sec)) / 2.0


def _append_idle_zerr(noisy: stim.Circuit, all_qubits: list,
                      busy_qubits: set, p_idle: float) -> None:
    """Append Z_ERROR on all qubits not in `busy_qubits`.

    `all_qubits` must be the set of actually-used qubits (from QUBIT_COORDS),
    not `range(num_qubits)`: Stim's num_qubits returns max_id + 1, which
    creates phantom qubits when there are gaps in qubit ID numbering.
    """
    if p_idle <= 0.0:
        return
    idle = [q for q in all_qubits if q not in busy_qubits]
    if idle:
        noisy.append("Z_ERROR", idle, p_idle)


def _get_declared_qubits(circuit: stim.Circuit) -> list:
    """Return sorted list of qubit IDs actually used in the circuit.

    Prefers `QUBIT_COORDS`-declared qubits if present; otherwise scans
    operation targets. Baseline-generated circuits don't include
    QUBIT_COORDS so the fallback is required.
    """
    coords = circuit.get_final_qubit_coordinates()
    if coords:
        return sorted(coords.keys())

    used = set()
    annotation_names = {'TICK', 'DETECTOR', 'OBSERVABLE_INCLUDE',
                        'QUBIT_COORDS', 'SHIFT_COORDS'}
    for instruction in circuit:
        if instruction.name in annotation_names:
            continue
        for t in instruction.targets_copy():
            if t.is_qubit_target:
                used.add(t.value)
    return sorted(used)


def _append_5step_cx(noisy: stim.Circuit, q1: int, q2: int, all_qubits: list,
                     params: dict, ms_error,
                     p_idle_R: float, p_idle_MS: float) -> None:
    """Insert 5-step error pattern after a logical CX(q1, q2).

    Physical decomposition of trapped-ion CNOT: R(q1) - MS(q1,q2) - R(q1) - R(q2) - R(q1).
    The five steps are 3 R on q1, 1 MS on (q1,q2), 1 R on q2.

    ms_error:
      - float (scalar)             → DEPOLARIZE2(ms_error) で 1 つの depolarizing 確率
      - list/tuple of 15 floats    → PAULI_CHANNEL_2(*15) で 15 個の Pauli 確率
        (Stim 引数順: IX, IY, IZ, XI, XX, XY, XZ, YI, YX, YY, YZ, ZI, ZX, ZY, ZZ)
    """
    p_R = params['p_R']

    # Step 1: R(q1), t_R
    if p_R > 0:
        noisy.append("DEPOLARIZE1", [q1], p_R)
    _append_idle_zerr(noisy, all_qubits, {q1}, p_idle_R)
    noisy.append("TICK")

    # Step 2: MS(q1, q2), t_MS
    if isinstance(ms_error, (list, tuple)):
        # PAULI_CHANNEL_2 — 15 個の Pauli 確率
        if len(ms_error) != 15:
            raise ValueError(
                f"PAULI_CHANNEL_2 needs exactly 15 probs, got {len(ms_error)}"
            )
        if any(p > 0 for p in ms_error):
            noisy.append("PAULI_CHANNEL_2", [q1, q2], list(ms_error))
    else:
        # 旧: DEPOLARIZE2(p_MS)
        if ms_error > 0:
            noisy.append("DEPOLARIZE2", [q1, q2], float(ms_error))
    _append_idle_zerr(noisy, all_qubits, {q1, q2}, p_idle_MS)
    noisy.append("TICK")

    # Step 3: R(q1), t_R
    if p_R > 0:
        noisy.append("DEPOLARIZE1", [q1], p_R)
    _append_idle_zerr(noisy, all_qubits, {q1}, p_idle_R)
    noisy.append("TICK")

    # Step 4: R(q2), t_R
    if p_R > 0:
        noisy.append("DEPOLARIZE1", [q2], p_R)
    _append_idle_zerr(noisy, all_qubits, {q2}, p_idle_R)
    noisy.append("TICK")

    # Step 5: R(q1), t_R
    if p_R > 0:
        noisy.append("DEPOLARIZE1", [q1], p_R)
    _append_idle_zerr(noisy, all_qubits, {q1}, p_idle_R)
    noisy.append("TICK")


def _append_h_2steps(noisy: stim.Circuit, h_targets: list, all_qubits: list,
                     params: dict, p_idle_R: float) -> None:
    """Insert 2-step error pattern (= R-R) after a logical H gate.

    H ≈ 2 single-qubit rotations physically, so emit the R gate error
    pattern twice for each H target.
    """
    p_R = params['p_R']
    busy = set(h_targets)
    for _ in range(2):
        if p_R > 0:
            noisy.append("DEPOLARIZE1", list(h_targets), p_R)
        _append_idle_zerr(noisy, all_qubits, busy, p_idle_R)
        noisy.append("TICK")


def inject_noise_approach_b(
    circuit: stim.Circuit,
    error_rates_ms: list,
    params: dict = None,
    include_idle: bool = True,
) -> stim.Circuit:
    """Inject noise per Approach B (5-step CX, 2-step H, target-only MEASURE/RESET).

    Noise insertion by instruction:
      - CX → CX + 5-step error block (R-MS-R-R-R)
      - H  → H + 2-step error block (R-R)
      - M/MR/MX → X_ERROR(p_meas) on target before measurement
      - R/RX/RZ → X_ERROR(p_reset) on target after reset
      - Idle qubits (= non-target) get Z_ERROR(p_idle) during R / MS /
        MEASURE / RESET steps (nominal durations)
      - DETECTOR / OBSERVABLE_INCLUDE / TICK は pass-through

    include_idle=False にするとゲート系の idle Z_ERROR を全て抑止し、
    ゲートエラー (DEPOLARIZE / X_ERROR) だけを挿入する。per-ion タイムライン
    方式 (inject_per_ion_idle) と組み合わせるときはこちらを使う
    (idle の会計をタイムライン側へ一本化するため)。
    """
    if params is None:
        params = default_noise_params()

    noisy = stim.Circuit()
    all_qubits = _get_declared_qubits(circuit)
    ms_idx = 0  # index into error_rates_ms (one per MS gate / CX pair)

    if include_idle:
        p_idle_R = compute_p_idle(params['t_R_us'], params['gamma_phi_per_s'])
        p_idle_MS = compute_p_idle(params['t_MS_us'], params['gamma_phi_per_s'])
        p_idle_meas = compute_p_idle(params['t_meas_us'], params['gamma_phi_per_s'])
        p_idle_reset = compute_p_idle(params['t_reset_us'], params['gamma_phi_per_s'])
        p_idle_mr = compute_p_idle(
            params['t_meas_us'] + params['t_reset_us'], params['gamma_phi_per_s'])
    else:
        p_idle_R = p_idle_MS = p_idle_meas = p_idle_reset = p_idle_mr = 0.0

    for instruction in circuit:
        name = instruction.name
        targets = instruction.targets_copy()

        if name == 'CX':
            # Split into individual pairs (each gets its own phonon-based error)
            for i in range(0, len(targets) - 1, 2):
                q1 = targets[i].value
                q2 = targets[i + 1].value

                # Per-pair MS error: スカラー (DEPOLARIZE2) or 15-float (PAULI_CHANNEL_2)
                if ms_idx < len(error_rates_ms):
                    ms_error = error_rates_ms[ms_idx][1]
                    ms_idx += 1
                else:
                    ms_error = params['p_MS_default']

                noisy.append("CX", [q1, q2])
                _append_5step_cx(noisy, q1, q2, all_qubits, params,
                                 ms_error, p_idle_R, p_idle_MS)

        elif name == 'H':
            tq = [t.value for t in targets]
            noisy.append("H", tq)
            _append_h_2steps(noisy, tq, all_qubits, params, p_idle_R)

        elif name in ('M', 'MX', 'MY', 'MZ'):
            tq = [t.value for t in targets]
            if params['p_meas'] > 0:
                noisy.append("X_ERROR", tq, params['p_meas'])
            noisy.append(name, tq)
            # Idle dephasing during measurement duration (t_meas)
            _append_idle_zerr(noisy, all_qubits, set(tq), p_idle_meas)
            noisy.append("TICK")

        elif name in ('R', 'RX', 'RY', 'RZ'):
            tq = [t.value for t in targets]
            noisy.append(name, tq)
            if params['p_reset'] > 0:
                noisy.append("X_ERROR", tq, params['p_reset'])
            # Idle dephasing during reset duration (t_reset)
            _append_idle_zerr(noisy, all_qubits, set(tq), p_idle_reset)
            noisy.append("TICK")

        elif name == 'MR':
            # Combined measure+reset: error before measure, error after reset
            tq = [t.value for t in targets]
            if params['p_meas'] > 0:
                noisy.append("X_ERROR", tq, params['p_meas'])
            noisy.append("MR", tq)
            if params['p_reset'] > 0:
                noisy.append("X_ERROR", tq, params['p_reset'])
            # Idle dephasing during combined measure+reset duration
            _append_idle_zerr(noisy, all_qubits, set(tq), p_idle_mr)
            noisy.append("TICK")

        else:
            # TICK, DETECTOR, OBSERVABLE_INCLUDE, etc. — pass through
            noisy.append(instruction)

    return noisy


def inject_per_ion_idle(
    noisy_circuit: stim.Circuit,
    anchor_ops: list,
    ion_gaps,
    params: dict = None,
) -> stim.Circuit:
    """(C) Per-ion timeline variant — idle dephasing の正確な per-qubit 挿入。

    ion_timeline.gaps_from_compile_results / gaps_from_per_liz_results で
    計算した per-ion ギャップを、各 Stim anchor (R / CX / M) の直前に
    `Z_ERROR(p_idle(Δt_ion))` として **そのイオンにだけ** 挿入する。

    旧方式 (inject_noise_approach_b の idle Z_ERROR + transport dephasing) との
    関係: 旧方式は大域直列タイムラインを前提に idle 時間を「ゲート回数×公称時間 +
    全 qubit 一律残差」の 2 経路で近似配分していた。本方式は各イオンの実経過時間
    そのものを使うため 1-LIZ でも本来正しく、multi-LIZ (per-LIZ stream) では
    旧方式が二重計上で破綻するため必須。組み合わせは:

        noisy = inject_noise_approach_b(circuit, rates, params, include_idle=False)
        noisy = inject_per_ion_idle(noisy, anchor_ops, ion_gaps, params)

    Per-ion timing:
      per-qubit の op 順序が物理順と一致する線形化上では、ギャップ内に
      そのイオンの op は存在せず、他イオンの op とは可換なので、
      「次 op の直前」への挿入は物理タイムライン上の任意位置と厳密に等価。

    Args:
        noisy_circuit: ゲートエラー挿入済み Stim 回路 (anchor 列は入力回路と同一)
        anchor_ops: コンパイル済み anchor op (ResetOp / GateOp('MS') / MeasureOp)
                    を Stim 論理 op 順 (= flat_ops / merged_flat 順) に並べたリスト
        ion_gaps: ion_timeline.IonGaps (gaps_by_op: id(op) -> {ion: gap_us})
        params: noise params dict (gamma_phi_per_s を使用)

    Returns:
        per-ion idle Z_ERROR を挿入した Stim 回路
    """
    if params is None:
        params = default_noise_params()
    gamma = params['gamma_phi_per_s']

    def _kind_matches(inst_name: str, op, targets: list) -> bool:
        if inst_name == 'R':
            return isinstance(op, ResetOp) and set(targets) == set(op.targets)
        if inst_name == 'CX':
            return (isinstance(op, GateOp) and op.type == 'MS'
                    and set(targets) == set(op.targets))
        if inst_name == 'M':
            return isinstance(op, MeasureOp) and set(targets) == set(op.targets)
        return False

    out = stim.Circuit()
    a_idx = 0
    anchor_names = {'R', 'CX', 'M'}

    for inst in noisy_circuit:
        if inst.name in anchor_names:
            qubit_targets = [t.value for t in inst.targets_copy()
                             if t.is_qubit_target]
            # CX は 1 命令に複数ペアがあり得るのでペア単位で anchor を消費
            if inst.name == 'CX':
                groups = [qubit_targets[i:i + 2]
                          for i in range(0, len(qubit_targets), 2)]
            else:
                groups = [[q] for q in qubit_targets]

            for grp in groups:
                if a_idx >= len(anchor_ops):
                    # 全 compiled anchor 消費後の M = 最終 transversal data 測定
                    # (recompiler が合成、コンパイル済み op に対応物なし)。
                    # そのイオンの trailing idle (最後の op → 回路終端) を挿入する
                    if inst.name == 'M' and len(grp) == 1:
                        gap_us = ion_gaps.trailing_idle.get(grp[0], 0.0)
                        p = compute_p_idle(gap_us, gamma)
                        if p > 0:
                            out.append("Z_ERROR", [grp[0]], p)
                        continue
                    raise ValueError(
                        f"Stim anchor が compiled anchor より多い: "
                        f"{inst.name} {grp} に対応する op がありません")
                op = anchor_ops[a_idx]
                a_idx += 1
                if not _kind_matches(inst.name, op, grp):
                    raise ValueError(
                        f"anchor 不一致: Stim {inst.name} {grp} vs compiled "
                        f"{type(op).__name__}({getattr(op, 'targets', None)}) "
                        f"@ index {a_idx - 1}")
                gaps = ion_gaps.gaps_by_op.get(id(op), {})
                for ion, gap_us in sorted(gaps.items()):
                    p = compute_p_idle(gap_us, gamma)
                    if p > 0:
                        out.append("Z_ERROR", [ion], p)
        out.append(inst)

    if a_idx != len(anchor_ops):
        raise ValueError(
            f"compiled anchor が Stim 回路より多い: {len(anchor_ops) - a_idx} 個未消費 "
            f"(Stim 回路と compile_results の対応を確認してください)")

    return out
