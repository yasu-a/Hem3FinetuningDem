"""Physical-device circuit loading and syndrome sampling."""

from pathlib import Path

import numpy as np
import stim


CIRCUIT_FILENAME = "circuit.stim"


def load_physical_device_circuit(
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

    circuit_path = experiments_directory / experiment_name / CIRCUIT_FILENAME
    if not circuit_path.is_file():
        raise FileNotFoundError(f"Experiment circuit not found: {circuit_path}")

    circuit = stim.Circuit.from_file(circuit_path)
    if circuit.num_detectors == 0:
        raise ValueError("Experiment circuit must define at least one detector")
    if circuit.num_observables == 0:
        raise ValueError("Experiment circuit must define at least one observable")
    return circuit


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
