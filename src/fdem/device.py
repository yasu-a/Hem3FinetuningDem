"""Physical-device circuit model and syndrome sampling."""

import numpy as np
import stim


def build_physical_device_circuit(
    distance: int,
    rounds: int,
    physical_error_rate: float,
) -> stim.Circuit:
    """Build the circuit model treated as the physical device.

    Args:
        distance: Code distance of the rotated surface code.
        rounds: Number of syndrome-measurement rounds.
        physical_error_rate: Error probability used by each circuit noise channel.

    Returns:
        A noisy rotated surface code memory circuit.
    """
    return stim.Circuit.generated(
        "surface_code:rotated_memory_x",
        distance=distance,
        rounds=rounds,
        after_clifford_depolarization=physical_error_rate,
        after_reset_flip_probability=physical_error_rate,
        before_measure_flip_probability=physical_error_rate,
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
