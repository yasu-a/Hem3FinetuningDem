"""Tests for loading physical-device experiment circuits."""

from pathlib import Path

import numpy as np
import pytest
import stim

from fdem.device import (
    load_physical_device_circuit,
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


def test_loads_circuit_from_named_experiment_directory(tmp_path: Path) -> None:
    """The loader reads circuit.stim from the selected experiment directory."""
    expected = _write_experiment(tmp_path, "d3-r1-default")

    actual = load_physical_device_circuit("d3-r1-default", tmp_path)

    assert actual == expected


def test_loaded_circuit_supports_syndrome_sampling(tmp_path: Path) -> None:
    """A loaded experiment circuit can produce detector-event samples."""
    circuit = _write_experiment(tmp_path, "sample")
    loaded = load_physical_device_circuit("sample", tmp_path)

    syndromes = sample_physical_device_syndromes(loaded, shots=5, seed=0)

    assert syndromes.shape == (5, circuit.num_detectors)
    assert syndromes.dtype == np.bool_


@pytest.mark.parametrize("name", ["", ".", "..", "../outside", "nested/name"])
def test_rejects_unsafe_experiment_names(tmp_path: Path, name: str) -> None:
    """Experiment selection cannot escape or nest below the experiment root."""
    with pytest.raises(ValueError):
        load_physical_device_circuit(name, tmp_path)


def test_reports_missing_experiment_circuit(tmp_path: Path) -> None:
    """A missing circuit.stim produces a clear file-not-found error."""
    with pytest.raises(FileNotFoundError, match="circuit.stim"):
        load_physical_device_circuit("missing", tmp_path)


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
        load_physical_device_circuit("invalid", tmp_path)
