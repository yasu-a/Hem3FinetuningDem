"""Verify recorded-shot provenance, pairing, and portable NumPy exports."""

from hashlib import sha256
import json
from pathlib import Path
from shutil import copytree

import numpy as np
import pytest
import stim

from fdem.hardware_data import export_run, load_run


DATA_ROOT = Path(__file__).resolve().parents[1] / "data" / "ler"


@pytest.mark.parametrize(
    ("run", "shots", "detectors"),
    [
        ("d3_at_q6_7/Z/r10", 50_000, 80),
        ("d3_at_q6_7/Z/r13", 50_000, 104),
        ("d3_at_q6_7/Z/r30", 50_000, 240),
        ("shenzhen_qec_19908642/Z/r8", 10_000, 64),
        ("shenzhen_qec_19908642/Z/r10", 10_000, 80),
        ("shenzhen_qec_19908642/Z/r13", 10_000, 104),
        ("shenzhen_qec_19908642/Z/r20", 10_000, 160),
    ],
)
def test_recorded_run_loads_in_stim_order(run, shots, detectors):
    """Check the original checksums and every run's boolean array dimensions."""
    circuit, events, flips, metadata = load_run(DATA_ROOT / run)
    assert events.dtype == flips.dtype == np.bool_
    assert events.shape == (shots, detectors)
    assert flips.shape == (shots, 1)
    assert metadata["shots"] == shots
    assert circuit.num_detectors == detectors


def test_npz_preserves_pairs_and_circuit(tmp_path):
    """Confirm the export preserves the original row pairing and circuit bytes."""
    folder = DATA_ROOT / "shenzhen_qec_19908642/Z/r8"
    circuit, events, flips, metadata = load_run(folder)
    output = export_run(folder, tmp_path / "nested" / "shots.npz")
    with np.load(output, allow_pickle=False) as data:
        np.testing.assert_array_equal(data["detection_events"], events)
        np.testing.assert_array_equal(data["observable_flips"], flips)
        assert stim.Circuit(data["circuit_stim"].item()) == circuit
        assert json.loads(data["metadata_json"].item()) == metadata
        mapping = data["detector_mapping_stim"].item().encode("utf-8")
        assert sha256(mapping).hexdigest() == metadata["sha256"]["detector_mapping.stim"]


@pytest.mark.parametrize("update_checksum", [False, True])
def test_load_rejects_truncated_logical_flips(tmp_path, update_checksum):
    """Detect a damaged file even when its manifest checksum was updated."""
    copied = tmp_path / "r8"
    copytree(DATA_ROOT / "shenzhen_qec_19908642/Z/r8", copied)
    flips = copied / "obs_flips_actual.b8"
    flips.write_bytes(flips.read_bytes()[:-1])
    error = "Checksum mismatch"
    if update_checksum:
        path = copied / "metadata.json"
        metadata = json.loads(path.read_text())
        metadata["sha256"][flips.name] = sha256(flips.read_bytes()).hexdigest()
        path.write_text(json.dumps(metadata))
        error = "Shot count"
    with pytest.raises(ValueError, match=error):
        load_run(copied)


def test_catalog_resolves_recordings_and_correlation_exports():
    """Keep all advertised source files and generated paths aligned with each run."""
    repo = DATA_ROOT.parents[1]
    catalog = json.loads((DATA_ROOT / "catalog.json").read_text())
    cataloged_manifests = set()
    assert catalog["schema_version"] == 1
    assert catalog["path_base"] == "repository_root"
    for run in catalog["runs"]:
        files = {name: repo / path for name, path in run["recorded_stim"].items()}
        metadata = json.loads(files["metadata"].read_text())
        cataloged_manifests.add(files["metadata"])
        assert run["run_id"] == metadata["name"]
        for field in ("distance", "basis", "rounds", "shots", "detectors", "observables", "source_doi"):
            assert run[field] == metadata[field]
        assert all(path.is_file() for path in files.values())
        assert {path.name for name, path in files.items() if name != "metadata"} == metadata["sha256"].keys()
        assert Path(run["generated_numpy"]) == Path("var/hardware-data") / run["run_id"] / "shots.npz"
    assert len(cataloged_manifests) == len(catalog["runs"])
    assert cataloged_manifests == set(DATA_ROOT.rglob("metadata.json"))

    viewer = catalog["computed_correlations"]["eight_round_viewer"]
    assert (repo / viewer["html"]).is_file()
    assert (repo / viewer["json_format"]).is_file()
    for export in viewer["exports"]:
        assert export["source_run_id"] in {run["run_id"] for run in catalog["runs"]}
        assert all((repo / path).is_file() for path in export["files"].values())
        document = json.loads((repo / export["files"]["qubit_json"]).read_text())
        assert export["final_readout_included"] == document["source"]["final_readout_included"]
        assert export["selection"] == document["source"]["selection"]
        assert export["rounds_shown"] == document["rounds"]
        assert export["shots"] == document["shots"]
