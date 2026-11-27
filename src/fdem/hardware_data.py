"""Load or export paired detection events and logical flips from recorded runs."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import stim


DEFAULT_DATA_ROOT = Path("data/ler")
"""Default repository-relative directory containing recorded hardware runs."""

DEFAULT_OUTPUT_ROOT = Path("var/hardware-data")
"""Default repository-relative directory for generated NumPy bundles."""


def load_run(
    run_directory: str | Path,
) -> tuple[stim.Circuit, np.ndarray, np.ndarray, dict]:
    """Load a recorded run in original Stim column and acquisition order.

    Args:
        run_directory: Directory with the circuit, two b8 files, and manifest.

    Returns:
        The Stim circuit, a boolean detector array of shape (shots, detectors),
        a boolean logical-flip array of shape (shots, observables), and the
        source metadata. A row index identifies the same shot in both arrays.

    Raises:
        FileNotFoundError: If a source file named by the manifest is missing.
        ValueError: If source checksums, Stim dimensions, or shot counts disagree.
    """
    folder = Path(run_directory)
    metadata = json.loads((folder / "metadata.json").read_text(encoding="utf-8"))
    integer_fields = ("shots", "detectors", "observables")
    if any(type(metadata[key]) is not int or metadata[key] < 1 for key in integer_fields):
        raise ValueError("The manifest must give positive shot, detector, and observable counts.")
    shot_count = metadata["shots"]
    detector_count = metadata["detectors"]
    observable_count = metadata["observables"]

    files = ["circuit.stim", "detection_events.b8", "obs_flips_actual.b8"]
    checksums = metadata["sha256"]
    if "detector_mapping.stim" in checksums:
        files.append("detector_mapping.stim")
    for filename in files:
        data = (folder / filename).read_bytes()
        if sha256(data).hexdigest() != checksums.get(filename):
            raise ValueError(f"Checksum mismatch for {folder / filename}.")

    circuit = stim.Circuit.from_file(folder / "circuit.stim")
    if (circuit.num_detectors, circuit.num_observables) != (detector_count, observable_count):
        raise ValueError("The circuit's detector or observable count differs from the manifest.")

    for filename, columns in (
        ("detection_events.b8", detector_count),
        ("obs_flips_actual.b8", observable_count),
    ):
        expected_bytes = shot_count * ((columns + 7) // 8)
        if (folder / filename).stat().st_size != expected_bytes:
            raise ValueError(f"Shot count in {filename} differs from the manifest.")
    detectors = stim.read_shot_data_file(
        path=folder / "detection_events.b8", format="b8", num_detectors=detector_count,
    )
    observables = stim.read_shot_data_file(
        path=folder / "obs_flips_actual.b8", format="b8", num_observables=observable_count,
    )
    if detectors.shape != (shot_count, detector_count) or observables.shape != (shot_count, observable_count):
        raise ValueError("Decoded detector or logical-flip dimensions disagree with the manifest.")
    return circuit, detectors, observables, metadata


def export_run(run_directory: str | Path, destination: str | Path) -> Path:
    """Write a compressed NumPy bundle with paired boolean arrays and provenance.

    Args:
        run_directory: Directory accepted by :func:`load_run`.
        destination: Path for the resulting .npz file.

    Returns:
        The path of the completed bundle. It can be opened with
        `np.load(path, allow_pickle=False)`.
    """
    folder = Path(run_directory)
    _, detectors, observables, metadata = load_run(folder)
    payload = {
        "detection_events": detectors,
        "observable_flips": observables,
        "metadata_json": json.dumps(metadata, ensure_ascii=False),
        "circuit_stim": (folder / "circuit.stim").read_bytes().decode("utf-8"),
    }
    if "detector_mapping.stim" in metadata["sha256"]:
        payload["detector_mapping_stim"] = (folder / "detector_mapping.stim").read_bytes().decode("utf-8")
    output = Path(destination)
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent) as tmp:
        completed = Path(tmp) / "shots.npz"
        np.savez_compressed(completed, **payload)
        completed.replace(output)
    return output


def main() -> None:
    """Export one recorded run or every run beneath the selected data root."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT,
                        help="Root of the hardware data bundles (default: data/ler).")
    parser.add_argument("--run", type=Path,
                        help="One path relative to the data root, for example d3_at_q6_7/Z/r10.")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT_ROOT,
                        help="Output root for NumPy bundles (default: var/hardware-data).")
    args = parser.parse_args()
    root = args.data_root.resolve()
    if args.run is None:
        runs = sorted(path.parent for path in root.rglob("metadata.json"))
    else:
        run = (root / args.run).resolve()
        if not run.is_relative_to(root):
            parser.error("The selected run must be beneath the data root.")
        runs = [run]
    if not runs:
        parser.error(f"No recorded run manifests found beneath {root}.")
    for run in runs:
        print(export_run(run, args.out / run.relative_to(root) / "shots.npz"))


if __name__ == "__main__":
    main()
