"""Physical-device circuit models and syndrome sampling."""

import numpy as np
import stim


_PAULI_CHANNEL_2_WEIGHTS = np.array(
    [4, 4, 10, 4, 30, 4, 4, 4, 4, 4, 4, 10, 4, 4, 6], dtype=float
)
_ONE_QUBIT_CLIFFORD_GATES = {
    "C_XYZ", "C_ZYX", "H", "H_XY", "H_XZ", "H_YZ", "S", "S_DAG",
    "SQRT_X", "SQRT_X_DAG", "SQRT_Y", "SQRT_Y_DAG", "X", "Y", "Z",
}
_RESET_BASES = {"R": "Z", "RX": "X", "RY": "Y"}
_MEASUREMENT_BASES = {"M": "Z", "MX": "X", "MY": "Y"}
_MEASUREMENT_RESET_BASES = {"MR": "Z", "MRX": "X", "MRY": "Y"}


def _validate_probability(name: str, value: float) -> None:
    """Validate a probability parameter.

    Args:
        name: Parameter name used in the exception message.
        value: Probability to validate.

    Returns:
        None.

    Raises:
        ValueError: If the value is not finite or is outside ``[0, 1]``.
    """
    if not np.isfinite(value) or not 0 <= value <= 1:
        raise ValueError(f"{name} must be a finite probability in [0, 1]")


def _basis_flip_error(basis: str) -> str:
    """Return a Pauli error that flips a result in the given basis.

    Args:
        basis: Measurement or reset basis (``X``, ``Y``, or ``Z``).

    Returns:
        The corresponding Stim error instruction name.

    Raises:
        ValueError: If the basis is unsupported.
    """
    if basis == "X":
        return "Z_ERROR"
    if basis in {"Y", "Z"}:
        return "X_ERROR"
    raise ValueError(f"Unsupported basis: {basis}")


def _qubit_targets(instruction: stim.CircuitInstruction) -> list[int]:
    """Extract physical-qubit targets from a Stim instruction.

    Args:
        instruction: Instruction whose targets will be inspected.

    Returns:
        Physical-qubit indices, excluding record and sweep-bit targets.
    """
    return [target.value for target in instruction.targets_copy() if target.is_qubit_target]


def _append_basis_error(
    circuit: stim.Circuit,
    basis: str,
    targets: list[int],
    probability: float,
    tag: str,
) -> None:
    """Append an error that flips a reset state or measurement result.

    Args:
        circuit: Circuit receiving the error instruction.
        basis: Reset or measurement basis.
        targets: Affected qubits.
        probability: Error probability.
        tag: Stim tag identifying the physical noise source.

    Returns:
        None.
    """
    if probability > 0:
        circuit.append(_basis_flip_error(basis), targets, probability, tag=tag)


def _pauli_channel_2_probabilities(error_probability: float) -> list[float]:
    """Distribute a two-qubit error probability over asymmetric Pauli errors.

    Args:
        error_probability: Total probability of a non-identity Pauli error.

    Returns:
        Probabilities in Stim's ``IX, IY, ..., ZZ`` order.
    """
    weights = _PAULI_CHANNEL_2_WEIGHTS / _PAULI_CHANNEL_2_WEIGHTS.sum()
    return (error_probability * weights).tolist()


def _build_configurable_circuit(
    distance: int,
    rounds: int,
    model: int,
    one_qubit_error_probability: float,
    two_qubit_error_probability: float,
    reset_error_probability: float,
    measurement_error_probability: float,
    idle_z_error_probability: float,
    spatial_variation: float,
    temporal_growth_coefficient: float,
    seed: int,
) -> stim.Circuit:
    """Add a selected physical noise model to an ideal surface-code circuit.

    Args:
        distance: Code distance of the rotated surface code.
        rounds: Number of syndrome-measurement rounds.
        model: Physical model number from 0 through 4.
        one_qubit_error_probability: Error probability after one-qubit gates.
        two_qubit_error_probability: Baseline error probability after two-qubit gates.
        reset_error_probability: State-preparation error probability.
        measurement_error_probability: Measurement-result error probability.
        idle_z_error_probability: Idle Z-error probability per tick interval.
        spatial_variation: Half-width of the uniform pair variation.
        temporal_growth_coefficient: Fractional two-qubit error growth per round.
        seed: Random seed used to assign pair-specific spatial variation.

    Returns:
        A noisy, flattened rotated surface-code memory circuit.

    Raises:
        ValueError: If a generated two-qubit channel has invalid probability.
    """
    ideal_circuit = stim.Circuit.generated(
        "surface_code:rotated_memory_x", distance=distance, rounds=rounds
    ).flattened()
    used_qubits = {
        target
        for instruction in ideal_circuit
        if instruction.name == "QUBIT_COORDS"
        for target in _qubit_targets(instruction)
    }
    rng = np.random.default_rng(seed)
    pair_variations: dict[tuple[int, int], float] = {}
    result = stim.Circuit()
    active_qubits: set[int] = set()
    qec_round = 0

    for instruction in ideal_circuit:
        name = instruction.name
        targets = _qubit_targets(instruction)

        if name == "TICK":
            if model >= 4 and idle_z_error_probability > 0:
                idle_qubits = sorted(used_qubits - active_qubits)
                if idle_qubits:
                    result.append(
                        "Z_ERROR", idle_qubits, idle_z_error_probability,
                        tag="fdem_idle_noise",
                    )
            result.append(instruction)
            active_qubits.clear()
            continue

        active_qubits.update(targets)
        if name in _RESET_BASES:
            result.append(instruction)
            _append_basis_error(
                result, _RESET_BASES[name], targets, reset_error_probability,
                "fdem_reset_noise",
            )
        elif name in _MEASUREMENT_BASES:
            _append_basis_error(
                result, _MEASUREMENT_BASES[name], targets,
                measurement_error_probability, "fdem_measurement_noise",
            )
            result.append(instruction)
        elif name in _MEASUREMENT_RESET_BASES:
            basis = _MEASUREMENT_RESET_BASES[name]
            _append_basis_error(
                result, basis, targets, measurement_error_probability,
                "fdem_measurement_noise",
            )
            result.append(instruction)
            _append_basis_error(
                result, basis, targets, reset_error_probability, "fdem_reset_noise"
            )
            qec_round += 1
        elif name in _ONE_QUBIT_CLIFFORD_GATES:
            result.append(instruction)
            if one_qubit_error_probability > 0:
                result.append(
                    "DEPOLARIZE1", targets, one_qubit_error_probability,
                    tag="fdem_one_qubit_noise",
                )
        elif name == "CX":
            for first, second in zip(targets[::2], targets[1::2], strict=True):
                pair = (first, second)
                result.append("CX", pair, tag=instruction.tag)
                pair_probability = two_qubit_error_probability
                if model >= 2:
                    pair_key = tuple(sorted(pair))
                    if pair_key not in pair_variations:
                        pair_variations[pair_key] = rng.uniform(
                            -spatial_variation, spatial_variation
                        )
                    pair_probability *= 1 + pair_variations[pair_key]
                if model >= 3:
                    pair_probability *= 1 + temporal_growth_coefficient * qec_round
                _validate_probability(
                    "Generated two-qubit error probability", pair_probability
                )
                if pair_probability > 0:
                    if model >= 1:
                        result.append(
                            "PAULI_CHANNEL_2", pair,
                            _pauli_channel_2_probabilities(pair_probability),
                            tag="fdem_two_qubit_noise",
                        )
                    else:
                        result.append(
                            "DEPOLARIZE2", pair, pair_probability,
                            tag="fdem_two_qubit_noise",
                        )
        else:
            result.append(instruction)

    # The trailing operations form the final interval even without a closing TICK.
    if model >= 4 and idle_z_error_probability > 0:
        idle_qubits = sorted(used_qubits - active_qubits)
        if idle_qubits:
            result.append(
                "Z_ERROR", idle_qubits, idle_z_error_probability,
                tag="fdem_idle_noise",
            )
    return result


def build_physical_device_circuit(
    distance: int,
    rounds: int,
    physical_error_rate: float | None = None,
    *,
    model: int = 0,
    one_qubit_error_probability: float = 0.0001,
    two_qubit_error_probability: float = 0.005,
    reset_error_probability: float = 0.001,
    measurement_error_probability: float = 0.002,
    idle_z_error_probability: float = 0.0005,
    spatial_variation: float = 0.3,
    temporal_growth_coefficient: float = 0.2,
    seed: int = 0,
) -> stim.Circuit:
    """Build the circuit model treated as the physical device.

    Passing ``physical_error_rate`` preserves the original uniform-noise API and,
    for Model 0, produces the exact circuit returned by the original implementation.
    Otherwise, the individual error probabilities are used. Models 1 through 4
    cumulatively add asymmetric Pauli noise, spatial variation, temporal growth,
    and idle dephasing. For Model 4, an idle interval starts at circuit initialization
    or immediately after a ``TICK`` and ends immediately before the next ``TICK`` or
    at circuit termination. A qubit is idle only if no gate, reset, or measurement
    acts on it during that interval. No idle noise is added before initialization or
    after circuit termination.

    Args:
        distance: Code distance of the rotated surface code.
        rounds: Number of syndrome-measurement rounds.
        physical_error_rate: Optional uniform error probability for one- and two-qubit
            gates, reset, and measurement. This is the legacy configuration shortcut.
        model: Physical model number from 0 through 4.
        one_qubit_error_probability: Error probability after one-qubit gates.
        two_qubit_error_probability: Baseline error probability after two-qubit gates.
        reset_error_probability: State-preparation error probability.
        measurement_error_probability: Measurement-result error probability.
        idle_z_error_probability: Idle Z-error probability per tick interval.
        spatial_variation: Half-width of the uniform pair variation.
        temporal_growth_coefficient: Fractional two-qubit error growth per QEC round.
        seed: Random seed used to assign pair-specific spatial variation.

    Returns:
        A noisy rotated surface-code memory circuit.

    Raises:
        ValueError: If the model number, circuit size, or a physical parameter is invalid.
    """
    if model not in range(5):
        raise ValueError("model must be an integer from 0 through 4")
    if distance < 2:
        raise ValueError("distance must be at least 2")
    if rounds < 1:
        raise ValueError("rounds must be at least 1")
    if not np.isfinite(spatial_variation) or not 0 <= spatial_variation <= 1:
        raise ValueError("spatial_variation must be finite and in [0, 1]")
    if temporal_growth_coefficient < 0 or not np.isfinite(
        temporal_growth_coefficient
    ):
        raise ValueError(
            "temporal_growth_coefficient must be finite and nonnegative"
        )

    if physical_error_rate is not None:
        _validate_probability("physical_error_rate", physical_error_rate)
        one_qubit_error_probability = physical_error_rate
        two_qubit_error_probability = physical_error_rate
        reset_error_probability = physical_error_rate
        measurement_error_probability = physical_error_rate

    probabilities = {
        "one_qubit_error_probability": one_qubit_error_probability,
        "two_qubit_error_probability": two_qubit_error_probability,
        "reset_error_probability": reset_error_probability,
        "measurement_error_probability": measurement_error_probability,
        "idle_z_error_probability": idle_z_error_probability,
    }
    for name, probability in probabilities.items():
        _validate_probability(name, probability)

    if physical_error_rate is not None and model == 0:
        return stim.Circuit.generated(
            "surface_code:rotated_memory_x",
            distance=distance,
            rounds=rounds,
            after_clifford_depolarization=physical_error_rate,
            after_reset_flip_probability=physical_error_rate,
            before_measure_flip_probability=physical_error_rate,
        )

    return _build_configurable_circuit(
        distance, rounds, model, one_qubit_error_probability,
        two_qubit_error_probability, reset_error_probability,
        measurement_error_probability, idle_z_error_probability,
        spatial_variation, temporal_growth_coefficient, seed,
    )


def sample_physical_device_syndromes(
    physical_device_circuit: stim.Circuit,
    shots: int,
    seed: int,
) -> np.ndarray:
    """Sample syndrome data from the circuit model of the physical device.

    Args:
        physical_device_circuit: Circuit model treated as the physical device.
        shots: Number of circuit executions to sample.
        seed: Random seed used by the Stim sampler.

    Returns:
        Detection-event samples with shape ``(shots, detectors)``.
    """
    sampler = physical_device_circuit.compile_detector_sampler(seed=seed)
    return sampler.sample(shots=shots)
