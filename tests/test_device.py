"""Tests for physical-device syndrome sampling."""

from pathlib import Path

import numpy as np
import pytest
import stim

from fdem.device import sample_physical_device_syndromes


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
