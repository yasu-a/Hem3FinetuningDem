# Recorded surface-code hardware shots

All runs are distance-3 Z-memory experiments. The files retain the complete
recorded shot sequence for each run.

| Dataset | QEC rounds | Shots per run | Source |
| --- | --- | ---: | --- |
| `d3_at_q6_7/Z` | 10, 13, 30 | 50,000 | Google Quantum AI, [Zenodo 13273331](https://doi.org/10.5281/zenodo.13273331) |
| `shenzhen_qec_19908642/Z` | 8, 10, 13, 20 | 10,000 | Xuandong Sun, Shenzhen IQA / SUSTech, [Zenodo 19908642](https://doi.org/10.5281/zenodo.19908642) |

Both source archives are licensed under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). The per-run
`metadata.json` identifies the archive or archive member and records checksums.

| File | Content |
| --- | --- |
| `detection_events.b8` | Stim detection-event bits in original detector-ID order |
| `obs_flips_actual.b8` | Recorded logical-observable flips in matching shot order |
| `circuit.stim` | Circuit defining detector IDs, measurement parities, and logical observables |
| `metadata.json` | Shot count, dimensions, experiment details, source, and SHA-256 hashes |
| `detector_mapping.stim` | Shenzhen measurement-to-detector map, including final readout |

## Use the data in Python

From the repository root, install the package and export the data.

~~~console
python -m pip install -e .
python -m fdem.hardware_data
~~~

For one run, use `python -m fdem.hardware_data --run d3_at_q6_7/Z/r10`.
The default output root is `var/hardware-data`. Use `--out` or `--data-root`
to select another location.

~~~python
import json

import numpy as np
import stim

with np.load("var/hardware-data/d3_at_q6_7/Z/r10/shots.npz", allow_pickle=False) as run:
    detectors = run["detection_events"]
    observable_flips = run["observable_flips"]
    circuit = stim.Circuit(run["circuit_stim"].item())
    metadata = json.loads(run["metadata_json"].item())

assert detectors.shape == (50_000, 80)
assert observable_flips.shape == (50_000, 1)
~~~

Both arrays are boolean. Row `i` in each array is the same original shot.
Logical bits indicate an observable flip relative to the circuit's reference.
Apply the same row indices to both arrays when splitting or shuffling shots.
The Python API `fdem.hardware_data.load_run` reads a run directly from its b8
files and verifies checksums and dimensions.

## Provenance

**Google Willow.** The detector and logical-flip files are the original
archive members. The ideal circuit is the archive's `circuit_ideal.stim`
stored here as `circuit.stim`. These are complete experiments with their
final data readout. [Google Quantum AI data release](https://doi.org/10.5281/zenodo.13273331)

**Shenzhen.** The b8 files were reconstructed from the published classified
readouts for logical-zero memory runs with active feedback off. The detector
records were cross-checked against the FPGA events. The archive supplies a
detector map and readouts. The associated gate-level circuit uses a
reconstructed CNOT schedule and preserves the reported no-reset timing,
detector definitions, and logical parity. The bundled CNOT order was selected
from symmetric hook-avoiding schedules by comparing graph coverage with
measured pair correlations. Rebuild it with `python -m fdem.shenzhen_circuit
detector_mapping.stim --output circuit.stim`. [Shenzhen data release](https://doi.org/10.5281/zenodo.19908642)
