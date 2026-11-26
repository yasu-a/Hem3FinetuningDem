"""Tests for physical-device syndrome sampling."""

from pathlib import Path

import numpy as np
import pymatching
import pytest
import stim

from fdem.device import (
    _add_uniform_noise,
    get_reference_dem,
    logical_error_rate_from_physical_device_dem,
    sample_physical_device_syndromes,
)


def _write_experiment(experiments_directory: Path, name: str) -> stim.Circuit:
    """Write a small noisy circuit in the experiment directory layout.

    Args:
        experiments_directory: Root directory for the test experiment.
        name: Experiment directory name.

    Returns:
        The circuit written to disk.
    """
    circuit = stim.Circuit.generated(
        "surface_code:rotated_memory_x",
        distance=3,
        rounds=1,
        after_clifford_depolarization=0.001,
    )
    directory = experiments_directory / name
    directory.mkdir(parents=True)
    circuit.to_file(directory / "circuit.stim")
    return circuit


def _write_circuit(
    experiments_directory: Path,
    name: str,
    circuit: stim.Circuit,
) -> None:
    """Write a supplied circuit in the experiment directory layout.

    Args:
        experiments_directory: Root directory for test experiments.
        name: Experiment directory name.
        circuit: Stim circuit to write.

    Returns:
        None.
    """
    directory = experiments_directory / name
    directory.mkdir(parents=True)
    circuit.to_file(directory / "circuit.stim")


def test_named_experiment_supports_syndrome_sampling(tmp_path: Path) -> None:
    """The public device interface returns detector samples by experiment name."""
    circuit = _write_experiment(tmp_path, "sample")

    syndromes = sample_physical_device_syndromes(
        "sample",
        shots=5,
        seed=0,
        experiments_directory=tmp_path,
    )

    assert syndromes.shape == (5, circuit.num_detectors)
    assert syndromes.dtype == np.bool_


def test_sampling_can_return_observables_for_evaluation(tmp_path: Path) -> None:
    """Evaluation sampling returns observables without exposing the device circuit."""
    circuit = _write_experiment(tmp_path, "evaluation")

    detectors, observables = sample_physical_device_syndromes(
        "evaluation",
        shots=5,
        seed=0,
        separate_observables=True,
        experiments_directory=tmp_path,
    )

    assert detectors.shape == (5, circuit.num_detectors)
    assert observables.shape == (5, circuit.num_observables)
    assert detectors.dtype == np.bool_
    assert observables.dtype == np.bool_


def test_uniform_noise_is_inserted_at_operation_specific_positions() -> None:
    """Gate, reset, measurement, and measure-reset noise uses the required positions."""
    ideal = stim.Circuit(
        """
        R 0
        RX 1
        H 2
        CX 2 3
        M 0
        MX 1
        MR 4
        MRX 5
        TICK
        """
    )
    expected = stim.Circuit(
        """
        R 0
        X_ERROR(0.01) 0
        RX 1
        Z_ERROR(0.01) 1
        H 2
        DEPOLARIZE1(0.01) 2
        CX 2 3
        DEPOLARIZE2(0.01) 2 3
        X_ERROR(0.01) 0
        M 0
        Z_ERROR(0.01) 1
        MX 1
        X_ERROR(0.01) 4
        MR 4
        X_ERROR(0.01) 4
        Z_ERROR(0.01) 5
        MRX 5
        Z_ERROR(0.01) 5
        TICK
        """
    )

    assert _add_uniform_noise(ideal, 0.01) == expected


def test_reference_dem_ignores_original_physical_noise(tmp_path: Path) -> None:
    """Different noise on the same ideal circuit produces the same reference DEM."""
    ideal = stim.Circuit.generated(
        "surface_code:rotated_memory_x",
        distance=3,
        rounds=2,
    )
    first = stim.Circuit("X_ERROR(0.1) 0") + ideal
    second = stim.Circuit("DEPOLARIZE1(0.2) 0") + ideal
    _write_circuit(tmp_path, "first", first)
    _write_circuit(tmp_path, "second", second)

    first_dem = get_reference_dem("first", experiments_directory=tmp_path)
    second_dem = get_reference_dem("second", experiments_directory=tmp_path)

    assert first_dem == second_dem
    assert first_dem.num_detectors == first.num_detectors
    assert first_dem.num_observables == first.num_observables


def test_reference_probability_changes_generated_priors(tmp_path: Path) -> None:
    """Changing the artificial physical error probability changes DEM priors."""
    _write_experiment(tmp_path, "probability")

    lower = get_reference_dem(
        "probability",
        error_probability=0.001,
        experiments_directory=tmp_path,
    )
    higher = get_reference_dem(
        "probability",
        error_probability=0.01,
        experiments_directory=tmp_path,
    )
    lower_priors = [
        instruction.args_copy()[0]
        for instruction in lower
        if instruction.type == "error"
    ]
    higher_priors = [
        instruction.args_copy()[0]
        for instruction in higher
        if instruction.type == "error"
    ]

    assert len(lower_priors) == len(higher_priors)
    assert not np.allclose(lower_priors, higher_priors)


def test_uniform_noise_preserves_ideal_circuit_structure() -> None:
    """Noise insertion preserves repeats, records, detectors, and observables."""
    ideal = stim.Circuit.generated(
        "surface_code:rotated_memory_x",
        distance=3,
        rounds=3,
    )

    reference = _add_uniform_noise(ideal, 0.001)

    assert reference.without_noise() == ideal
    assert reference.num_measurements == ideal.num_measurements
    assert reference.num_detectors == ideal.num_detectors
    assert reference.num_observables == ideal.num_observables
    assert any(isinstance(operation, stim.CircuitRepeatBlock) for operation in reference)


def test_reference_dem_builds_a_matching_decoder(tmp_path: Path) -> None:
    """The generated reference DEM is graphlike and accepted by PyMatching."""
    circuit = _write_experiment(tmp_path, "decoder")

    dem = get_reference_dem("decoder", experiments_directory=tmp_path)
    decoder = pymatching.Matching.from_detector_error_model(dem)

    assert dem.num_detectors == circuit.num_detectors
    assert dem.num_observables == circuit.num_observables
    assert decoder.num_detectors == circuit.num_detectors


def test_physical_device_dem_ler_supports_disjoint_pauli_channels(
    tmp_path: Path,
) -> None:
    """The physical-noise baseline decodes a circuit containing PAULI_CHANNEL_2."""
    circuit = stim.Circuit(
        """
        R 0 1
        PAULI_CHANNEL_2(0.01, 0, 0, 0.02, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0) 0 1
        M 0 1
        DETECTOR rec[-2]
        OBSERVABLE_INCLUDE(0) rec[-1]
        """
    )
    _write_circuit(tmp_path, "physical-dem", circuit)
    detectors, observables = sample_physical_device_syndromes(
        "physical-dem",
        shots=1_000,
        seed=3,
        separate_observables=True,
        experiments_directory=tmp_path,
    )

    ler = logical_error_rate_from_physical_device_dem(
        "physical-dem",
        detectors,
        observables,
        experiments_directory=tmp_path,
    )

    assert 0 <= ler <= 1


@pytest.mark.parametrize("probability", [-0.1, 1.1, np.nan, np.inf])
def test_reference_dem_rejects_invalid_probabilities(
    tmp_path: Path,
    probability: float,
) -> None:
    """Reference DEM generation rejects invalid artificial error probabilities."""
    _write_experiment(tmp_path, "invalid-probability")

    with pytest.raises(ValueError, match="probability"):
        get_reference_dem(
            "invalid-probability",
            error_probability=probability,
            experiments_directory=tmp_path,
        )


@pytest.mark.parametrize("name", ["", ".", "..", "../outside", "nested/name"])
def test_rejects_unsafe_experiment_names(tmp_path: Path, name: str) -> None:
    """Experiment selection cannot escape or nest below the experiment root."""
    with pytest.raises(ValueError):
        sample_physical_device_syndromes(
            name,
            shots=1,
            seed=0,
            experiments_directory=tmp_path,
        )


def test_reports_missing_experiment_circuit(tmp_path: Path) -> None:
    """A missing circuit.stim produces a clear file-not-found error."""
    with pytest.raises(FileNotFoundError, match="circuit.stim"):
        sample_physical_device_syndromes(
            "missing",
            shots=1,
            seed=0,
            experiments_directory=tmp_path,
        )


@pytest.mark.parametrize(
    "circuit_text",
    [
        "R 0\nM 0\nOBSERVABLE_INCLUDE(0) rec[-1]\n",
        "R 0\nM 0\nDETECTOR rec[-1]\n",
    ],
)
def test_rejects_circuits_missing_decode_metadata(
    tmp_path: Path,
    circuit_text: str,
) -> None:
    """Experiments must define both detectors and logical observables."""
    directory = tmp_path / "invalid"
    directory.mkdir()
    (directory / "circuit.stim").write_text(circuit_text, encoding="utf-8")

    with pytest.raises(ValueError):
        sample_physical_device_syndromes(
            "invalid",
            shots=1,
            seed=0,
            experiments_directory=tmp_path,
        )
