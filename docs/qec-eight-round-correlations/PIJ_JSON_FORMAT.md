# Reading the distance-3, eight-round correlation JSON

Each file contains signed two-point detector correlations for a logical $Z$ memory experiment. The files use the same schema and estimator. Open [index.html](index.html) locally for the interactive viewer.

| Dataset | File | Shots | Detectors | Source |
| --- | --- | ---: | ---: | --- |
| Google Willow | [pij.json](d3_at_q6_7/Z/r8/pij.json) | 50,000 | 60 | Rounds 0–7 of the published 10-round `q6_7` shots |
| Shenzhen IQA / SUSTech | [pij.json](shenzhen_qec_19908642/Z/r8/pij.json) | 10,000 | 64 | Dedicated eight-round run 42676, initialized in logical zero, with no active feedback |

The Google prefix ends at its eighth ancilla readout. There is no final data measurement at that point in those shots. The Shenzhen run has four final boundary detectors formed from final data measurements and the last measured stabilizers. Source file details and selection are recorded under `source` in each JSON and in the run's `metadata.json`. [Google source archive](https://zenodo.org/records/13273331) · [Shenzhen source archive](https://zenodo.org/records/19908642)

## Lookup

`qubits` is an object keyed by physical qubit label. All 17 patch qubits have a `coord` array in `[x, y]` order and a `role` of `ancilla` or `data`. Coordinates are lattice positions in the dataset's own coordinate system. Treat qubit labels and coordinates as local to their dataset. A label shared by both files does not establish a physical correspondence.

Each ancilla has a `correlations` object keyed by the eight ancilla labels, including itself. Data qubits enter the Shenzhen final readout through check parities.

`qubits[a].correlations[b][i][j]` is the signed correlation between the detector at qubit `a`, round `round_order[i]`, and the detector at qubit `b`, round `round_order[j]`. The first matrix axis belongs to the outer qubit. The second belongs to the inner qubit.

```python
import json
from pathlib import Path

doc = json.loads(Path("pij.json").read_text())
a, b = [label for label, q in doc["qubits"].items() if q["role"] == "ancilla"][:2]
i = doc["round_order"].index(2)
j = doc["round_order"].index(3)
value = doc["qubits"][a]["correlations"][b][i][j]
if value is not None:
    print(f"{a} at round 2, {b} at round 3: {value:+.6g}")
```

## Round and value conventions

- Round IDs 0–7 refer to the eight QEC syndrome rounds. Round 0 is the first ancilla measurement and has only four initialization detectors for a $Z$-basis memory. Round ID 8 is reserved for the final data-readout boundary. The PNGs show rounds 0–7. The JSON has slots for all nine round IDs.
- Each ancilla's `detector_rounds` lists the rounds in which that ancilla has a detector. If either round is absent from its qubit's list, the matrix entry is `null`. A detector paired with itself also has `null`. Any other `null` represents an undefined estimate. Keep `null` as missing data and keep numeric zero as a valid estimate.
- The estimate is a signed decimal, with `0.01` corresponding to 1%. Preserve its sign and precision. Saturation in a heatmap changes only the displayed color.
- The reverse lookup swaps both qubit and round. In other words, `C[a][b][i][j]` equals `C[b][a][j][i]`. Each unordered detector pair is represented twice. When counting pairs, keep one orientation, for example by comparing the tuples `(qubit_label, round_id)`.
- The neighboring `detectors.csv` maps detector IDs to ancilla coordinates and rounds. The neighboring `pij.csv` contains the same estimates in detector-ID order. Matrix entries in the JSON can be matched to the CSV using that map.

The estimator is $p_{ij}=(1-\sqrt{b_i b_j/b_{ij}})/2$, with $b_i=\mathbb{E}[(-1)^{d_i}]$ and $b_{ij}=\mathbb{E}[(-1)^{d_i\oplus d_j}]$. Here $d_i$ is the binary detection event for detector $i$. Undefined estimates include zero pair-parity bias or a negative square-root argument.
