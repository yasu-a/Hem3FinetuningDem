"""Syndrome sampling from physical-device experiment circuits."""

from pathlib import Path

import numpy as np
import pymatching
import stim


_CIRCUIT_FILENAME = "circuit.stim"
_Z_RESETS = {"R", "RZ"}
_X_RESETS = {"RX"}
_Z_MEASUREMENTS = {"M", "MZ"}
_X_MEASUREMENTS = {"MX"}
_Z_MEASURE_RESETS = {"MR", "MRZ"}
_X_MEASURE_RESETS = {"MRX"}


def _load_physical_device_circuit(
    experiment_name: str,
    experiments_directory: Path = Path("experiments"),
) -> stim.Circuit:
    """Load a THeatA-generated circuit from an experiment directory.

    The circuit is loaded from
    ``experiments_directory / experiment_name / circuit.stim``. Other files,
    such as a future ``config.json``, may coexist in the same directory.

    Args:
        experiment_name: Name of one direct child of the experiments directory.
        experiments_directory: Root directory containing experiment directories.

    Returns:
        The noisy Stim circuit treated as the physical device.

    Raises:
        ValueError: If the experiment name is unsafe or the circuit does not define
            detectors and at least one logical observable.
        FileNotFoundError: If the experiment circuit does not exist.
    """
    experiment_path = Path(experiment_name)
    if (
        not experiment_name
        or experiment_name in {".", ".."}
        or experiment_path.name != experiment_name
    ):
        raise ValueError("Experiment name must be a single directory name")

    circuit_path = experiments_directory / experiment_name / _CIRCUIT_FILENAME
    if not circuit_path.is_file():
        raise FileNotFoundError(f"Experiment circuit not found: {circuit_path}")

    circuit = stim.Circuit.from_file(circuit_path)
    if circuit.num_detectors == 0:
        raise ValueError("Experiment circuit must define at least one detector")
    if circuit.num_observables == 0:
        raise ValueError("Experiment circuit must define at least one observable")
    return circuit


def _qubit_targets(instruction: stim.CircuitInstruction) -> list[int]:
    """Return the direct qubit targets of a circuit instruction.

    Args:
        instruction: Stim circuit instruction to inspect.

    Returns:
        Qubit indices in instruction target order. Non-qubit targets are omitted.
    """
    return [
        target.qubit_value
        for target in instruction.targets_copy()
        if target.is_qubit_target
    ]


def _append_error(
    circuit: stim.Circuit,
    name: str,
    probability: float,
    targets: list[int],
) -> None:
    """Append an error instruction when it has at least one target.

    Args:
        circuit: Circuit receiving the error instruction.
        name: Stim error instruction name.
        probability: Error probability supplied to Stim.
        targets: Qubits affected by the error.

    Returns:
        None.
    """
    if targets:
        circuit.append(name, targets, probability)


def _add_uniform_noise(
    ideal_circuit: stim.Circuit,
    error_probability: float,
) -> stim.Circuit:
    """Add uniform gate, reset, and measurement noise to an ideal circuit.

    Repeat blocks and noiseless structural instructions are preserved. Z- and X-basis
    measure-reset instructions receive measurement noise before the instruction and
    reset noise after it.

    Args:
        ideal_circuit: Circuit whose existing noise has already been removed.
        error_probability: Probability used by every inserted noise instruction.

    Returns:
        A copy of the ideal circuit with the uniform noise model inserted.
    """
    noisy_circuit = stim.Circuit()
    for operation in ideal_circuit:
        if isinstance(operation, stim.CircuitRepeatBlock):
            noisy_body = _add_uniform_noise(
                operation.body_copy(),
                error_probability,
            )
            noisy_circuit.append(
                stim.CircuitRepeatBlock(operation.repeat_count, noisy_body)
            )
            continue

        name = operation.name
        targets = _qubit_targets(operation)
        if name in _Z_MEASURE_RESETS:
            _append_error(noisy_circuit, "X_ERROR", error_probability, targets)
            noisy_circuit.append(operation)
            _append_error(noisy_circuit, "X_ERROR", error_probability, targets)
        elif name in _X_MEASURE_RESETS:
            _append_error(noisy_circuit, "Z_ERROR", error_probability, targets)
            noisy_circuit.append(operation)
            _append_error(noisy_circuit, "Z_ERROR", error_probability, targets)
        elif name in _Z_MEASUREMENTS:
            _append_error(noisy_circuit, "X_ERROR", error_probability, targets)
            noisy_circuit.append(operation)
        elif name in _X_MEASUREMENTS:
            _append_error(noisy_circuit, "Z_ERROR", error_probability, targets)
            noisy_circuit.append(operation)
        elif name in _Z_RESETS:
            noisy_circuit.append(operation)
            _append_error(noisy_circuit, "X_ERROR", error_probability, targets)
        elif name in _X_RESETS:
            noisy_circuit.append(operation)
            _append_error(noisy_circuit, "Z_ERROR", error_probability, targets)
        else:
            noisy_circuit.append(operation)
            gate = stim.gate_data(name)
            if gate.is_unitary and gate.is_single_qubit_gate:
                _append_error(
                    noisy_circuit,
                    "DEPOLARIZE1",
                    error_probability,
                    targets,
                )
            elif gate.is_unitary and gate.is_two_qubit_gate:
                quantum_pairs = [
                    [target.qubit_value for target in group]
                    for group in operation.target_groups()
                    if all(target.is_qubit_target for target in group)
                ]
                _append_error(
                    noisy_circuit,
                    "DEPOLARIZE2",
                    error_probability,
                    [qubit for pair in quantum_pairs for qubit in pair],
                )
    return noisy_circuit


def get_reference_dem(
    experiment_name: str,
    error_probability: float = 0.001,
    *,
    experiments_directory: Path = Path("experiments"),
) -> stim.DetectorErrorModel:
    """Build a reference DEM from circuit structure and uniform artificial noise.

    All physical-device noise is removed before the uniform noise model is inserted.
    Consequently, the original noise types, locations, and probabilities do not affect
    the returned detector error model.

    Args:
        experiment_name: Name of one direct child of the experiments directory.
        error_probability: Probability used for all artificial physical errors.
        experiments_directory: Root directory containing experiment directories.

    Returns:
        A flattened detector error model generated from the uniformly noisy reference
        circuit.

    Raises:
        ValueError: If the probability is invalid, the experiment name is unsafe, or
            the circuit does not define detectors and at least one observable.
        FileNotFoundError: If the experiment circuit does not exist.
    """
    if not np.isfinite(error_probability) or not 0 <= error_probability <= 1:
        raise ValueError("Error probability must be finite and between 0 and 1")
    physical_circuit = _load_physical_device_circuit(
        experiment_name,
        experiments_directory,
    )
    ideal_circuit = physical_circuit.without_noise()
    reference_circuit = _add_uniform_noise(ideal_circuit, error_probability)
    return reference_circuit.detector_error_model(
        decompose_errors=True
    ).flattened()


def logical_error_rate_from_physical_device_dem(
    experiment_name: str,
    detectors: np.ndarray,
    observables: np.ndarray,
    *,
    experiments_directory: Path = Path("experiments"),
) -> float:
    """Calculate LER using a DEM derived directly from the physical-device circuit.

    This comparison DEM retains the physical circuit's noise model and is not used as
    the training reference DEM. Mutually exclusive ``PAULI_CHANNEL_2`` branches require
    Stim's disjoint-error approximation; its threshold is limited to the largest total
    channel probability present in the circuit.

    Args:
        experiment_name: Name of one direct child of the experiments directory.
        detectors: Evaluation detection events with shape ``(shots, detectors)``.
        observables: Actual logical flips with shape ``(shots, observables)``.
        experiments_directory: Root directory containing experiment directories.

    Returns:
        The fraction of evaluation shots decoded with an incorrect observable.

    Raises:
        ValueError: If sample dimensions do not match the physical circuit or if DEM
            conversion or decoding fails.
        FileNotFoundError: If the experiment circuit does not exist.
    """
    physical_circuit = _load_physical_device_circuit(
        experiment_name,
        experiments_directory,
    )
    if detectors.ndim != 2 or detectors.shape[1] != physical_circuit.num_detectors:
        raise ValueError("Detector samples must match the physical circuit")
    if observables.ndim != 2 or observables.shape != (
        detectors.shape[0],
        physical_circuit.num_observables,
    ):
        raise ValueError("Observable samples must match the physical circuit and shots")

    channel_probabilities = [
        sum(instruction.gate_args_copy())
        for instruction in physical_circuit.flattened()
        if instruction.name == "PAULI_CHANNEL_2"
    ]
    if channel_probabilities:
        physical_dem = physical_circuit.detector_error_model(
            decompose_errors=True,
            approximate_disjoint_errors=np.nextafter(
                max(channel_probabilities),
                np.inf,
            ),
        ).flattened()
    else:
        physical_dem = physical_circuit.detector_error_model(
            decompose_errors=True,
        ).flattened()

    decoder = pymatching.Matching.from_detector_error_model(physical_dem)
    predictions = decoder.decode_batch(detectors)
    return float((predictions != observables).any(axis=1).mean())


def sample_physical_device_syndromes(
    experiment_name: str,
    shots: int,
    seed: int,
    *,
    separate_observables: bool = False,
    experiments_directory: Path = Path("experiments"),
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Sample syndrome data from the selected physical-device experiment.

    The experiment circuit remains an implementation detail of this module. Logical
    observable samples can be returned together with the syndromes only for evaluating
    decoder performance; they are not used to construct or train the DEM.

    Args:
        experiment_name: Name of one direct child of the experiments directory.
        shots: Number of circuit executions to sample.
        seed: Random seed used by the Stim sampler.
        separate_observables: Whether to return actual logical-observable flips in a
            separate array.
        experiments_directory: Root directory containing experiment directories.

    Returns:
        Detection-event samples with shape ``(shots, detectors)``. When
        ``separate_observables`` is true, returns a tuple containing the detection-event
        samples and logical-observable flips.

    Raises:
        ValueError: If the experiment name is unsafe or the circuit does not define
            detectors and at least one logical observable.
        FileNotFoundError: If the experiment circuit does not exist.
    """
    physical_device_circuit = _load_physical_device_circuit(
        experiment_name,
        experiments_directory,
    )
    sampler = physical_device_circuit.compile_detector_sampler(seed=seed)
    return sampler.sample(
        shots=shots,
        separate_observables=separate_observables,
    )
