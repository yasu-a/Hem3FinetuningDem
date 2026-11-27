"""Tests for recorded shot input and held-out DEM evaluation."""

import json
from pathlib import Path

import numpy as np
import pytest
import stim

from fdem.__main__ import main
from fdem.shot_data import load_recorded_shots, split_recorded_shots


@pytest.fixture
def recorded_run(tmp_path: Path) -> Path:
    """Write an annotated circuit and consistently aligned Willow-format files."""
    circuit = stim.Circuit("""
        R 0 1
        CX 0 1
        CX sweep[0] 0
        M 0 1
        DETECTOR(0, 0) rec[-2]
        DETECTOR(1, 0) rec[-1]
        OBSERVABLE_INCLUDE(0) rec[-2] rec[-1]
    """)
    sweeps = np.tile(np.array([[False], [True]]), (600, 1))
    patterns = [[0, 0], [1, 0], [0, 1], [1, 1]]
    faults = np.tile(np.repeat(patterns, [102, 8, 7, 3], axis=0), (10, 1))
    measurements = faults.astype(bool)
    measurements[:, :1] ^= sweeps
    detectors, observables = circuit.compile_m2d_converter().convert(
        measurements=measurements, sweep_bits=sweeps, separate_observables=True
    )
    circuit.to_file(tmp_path / "circuit_ideal.stim")
    for filename, data in (
        ("measurements.b8", measurements),
        ("sweep_bits.b8", sweeps),
        ("detection_events.b8", detectors),
        ("obs_flips_actual.b8", observables),
    ):
        stim.write_shot_data_file(
            data=data, path=tmp_path / filename, format="b8", num_measurements=data.shape[1]
        )
    (tmp_path / "metadata.json").write_text(json.dumps({"shots": len(detectors)}), encoding="utf-8")
    return tmp_path


def test_recorded_run_reads_aligned_detector_and_observable_bits(recorded_run: Path) -> None:
    """Decode the stored binary shots and verify them from raw data and sweep bits."""
    circuit, detectors, observables = load_recorded_shots(recorded_run)

    assert circuit.num_detectors == 2
    assert detectors.shape == (1200, 2)
    assert observables.shape == (1200, 1)
    assert detectors.dtype == observables.dtype == np.bool_
    np.testing.assert_array_equal(observables[:, 0], detectors[:, 0] ^ detectors[:, 1])


def test_recorded_run_rejects_truncated_or_reordered_logical_bits(recorded_run: Path) -> None:
    """Reject shot count and row order mismatches before the data is used for LER."""
    path = recorded_run / "obs_flips_actual.b8"
    correct = path.read_bytes()
    path.write_bytes(correct[:-1])
    with pytest.raises(ValueError, match="different shot counts"):
        load_recorded_shots(recorded_run)

    path.write_bytes(correct[1:] + correct[:1])
    with pytest.raises(ValueError, match="do not reproduce obs_flips_actual.b8"):
        load_recorded_shots(recorded_run)


def test_recorded_run_rejects_mismatched_metadata(recorded_run: Path) -> None:
    """Report a shot count that disagrees with the published metadata."""
    (recorded_run / "metadata.json").write_text('{"shots": 1199}', encoding="utf-8")
    with pytest.raises(ValueError, match="metadata.json"):
        load_recorded_shots(recorded_run)


def test_recorded_run_accepts_published_bits_without_raw_measurements(recorded_run: Path) -> None:
    """Allow the minimum shot bundle when the optional provenance files are absent."""
    for filename in ("measurements.b8", "sweep_bits.b8", "metadata.json"):
        (recorded_run / filename).unlink()

    _, detectors, observables = load_recorded_shots(recorded_run)

    assert len(detectors) == len(observables) == 1200


def test_split_keeps_shots_disjoint_and_labels_aligned() -> None:
    """Use the same deterministic shot indices for detector bits and logical flips."""
    shot_id = np.arange(100, dtype=np.uint8).reshape(-1, 1)
    detectors = np.unpackbits(shot_id, axis=1).astype(bool)
    observables = (shot_id % 2).astype(bool)

    training, test, expected = split_recorded_shots(detectors, observables, seed=0)
    training_ids = np.packbits(training, axis=1).ravel()
    test_ids = np.packbits(test, axis=1).ravel()

    assert training.shape == (80, 8)
    assert test.shape == (20, 8)
    np.testing.assert_array_equal(np.sort(np.concatenate((training_ids, test_ids))), shot_id[:, 0])
    np.testing.assert_array_equal(expected[:, 0], test_ids % 2)
    for original, repeated in zip(
        (training, test, expected), split_recorded_shots(detectors, observables, seed=0), strict=True
    ):
        np.testing.assert_array_equal(original, repeated)


def test_cli_fits_and_scores_recorded_shots(
    recorded_run: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Run the LSQ optimizer and persist both held-out scores using recorded shots."""
    monkeypatch.chdir(recorded_run)
    monkeypatch.setattr(
        "sys.argv",
        ["fdem", "--experiment", "recorded", "--shot-data", str(recorded_run), "--training-method", "lsq"],
    )
    main()

    output_directory = Path("var/recorded/lsq/weight-auto")
    run = json.loads((output_directory / "run.json").read_text(encoding="utf-8"))
    assert run["train_shots"] == 960
    assert run["test_shots"] == 240
    for model in ("initial", "fitted"):
        assert run[model]["ler"] == run[model]["logical_failures"] / run["test_shots"]
        assert np.isfinite(run[model]["single_rmse"])
        assert np.isfinite(run[model]["pair_rmse"])
    assert run["fitted"]["single_rmse"] < run["initial"]["single_rmse"]
    assert stim.DetectorErrorModel.from_file(output_directory / "fitted.dem").num_detectors == 2
    assert (output_directory / "ler.png").is_file()
    stdout = capsys.readouterr().out
    assert "held-out logical failures" in stdout
    assert "physical-device DEM LER" not in stdout
