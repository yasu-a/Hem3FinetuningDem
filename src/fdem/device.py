"""Syndrome sampling from physical-device experiment circuits."""

from pathlib import Path

import numpy as np
import stim


_CIRCUIT_FILENAME = "circuit.stim"


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
