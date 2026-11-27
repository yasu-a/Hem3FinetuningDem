"""Read shot-level data from a recorded surface-code experiment."""

import json
from pathlib import Path

import numpy as np
import stim


def load_recorded_shots(directory: Path) -> tuple[stim.Circuit, np.ndarray, np.ndarray]:
    """Read a Willow-format run in its published Stim detector order.

    Args:
        directory: Run directory containing the ideal circuit and shot files.

    Returns:
        The ideal circuit, boolean detection events, and measured logical flips.

    Raises:
        OSError: If a required input file cannot be read.
        ValueError: If the circuit, counts, or optional raw measurements disagree.
    """
    circuit = stim.Circuit.from_file(directory / "circuit_ideal.stim")
    if circuit.num_detectors < 2 or circuit.num_observables < 1:
        raise ValueError("The recorded circuit must define at least two detectors and one observable")

    detectors = stim.read_shot_data_file(
        path=directory / "detection_events.b8", format="b8", num_detectors=circuit.num_detectors
    )
    observables = stim.read_shot_data_file(
        path=directory / "obs_flips_actual.b8", format="b8", num_observables=circuit.num_observables
    )
    if len(detectors) != len(observables):
        raise ValueError("Recorded detection events and logical flips have different shot counts")
    if len(detectors) < 2:
        raise ValueError("At least two recorded shots are required for training and evaluation")

    metadata_path = directory / "metadata.json"
    if metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        expected_shots = metadata.get("shots") if isinstance(metadata, dict) else None
        if type(expected_shots) is not int or expected_shots != len(detectors):
            raise ValueError("Recorded shot count does not match the shots field in metadata.json")

    if (directory / "measurements.b8").is_file():
        _verify_raw_measurements(directory, circuit, detectors, observables)
    return circuit, detectors, observables


def _verify_raw_measurements(
    directory: Path,
    circuit: stim.Circuit,
    detectors: np.ndarray,
    observables: np.ndarray,
) -> None:
    """Check the published shot arrays against the circuit's measurement parities.

    Args:
        directory: Run directory containing the raw measurement files.
        circuit: Ideal annotated circuit for the run.
        detectors: Published detection events in shot and detector order.
        observables: Published logical flips in shot and observable order.

    Raises:
        OSError: If the measurement or required sweep file cannot be read.
        ValueError: If the converted and published shot arrays differ.
    """
    measurements = stim.read_shot_data_file(
        path=directory / "measurements.b8", format="b8", num_measurements=circuit.num_measurements
    )
    sweep_bits = None
    if circuit.num_sweep_bits:
        sweep_bits = stim.read_shot_data_file(
            path=directory / "sweep_bits.b8", format="b8", num_measurements=circuit.num_sweep_bits
        )
    converted_detectors, converted_observables = circuit.compile_m2d_converter().convert(
        measurements=measurements, sweep_bits=sweep_bits, separate_observables=True
    )
    if not np.array_equal(detectors, converted_detectors):
        raise ValueError("Raw measurements do not reproduce detection_events.b8")
    if not np.array_equal(observables, converted_observables):
        raise ValueError("Raw measurements do not reproduce obs_flips_actual.b8")


def split_recorded_shots(
    detectors: np.ndarray, observables: np.ndarray, *, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Use one permutation to hold out one fifth of a run's recorded shots.

    Args:
        detectors: Recorded detection events.
        observables: Recorded logical flips in the same shot order.
        seed: Seed for the NumPy shot permutation.

    Returns:
        Training detection events, held-out detection events, and held-out flips.

    Raises:
        ValueError: If the shot counts differ or there are fewer than two shots.
    """
    shots = len(detectors)
    if shots != len(observables) or shots < 2:
        raise ValueError("Expected the same number of detector and observable rows with at least two shots")
    indices = np.random.default_rng(seed).permutation(shots)
    test_shots = max(1, shots // 5)
    training_indices = indices[:-test_shots]
    test_indices = indices[-test_shots:]
    return detectors[training_indices], detectors[test_indices], observables[test_indices]
