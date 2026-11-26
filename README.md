# fdem

A minimal example that estimates the logical error rate (LER) of a rotated surface
code from syndrome samples produced by a noisy circuit exported by the THeatA
workbench.

The device module exposes sampled detector events and a reference DEM. To build the
reference, it removes all noise from the experiment circuit, adds uniform gate, reset,
and measurement noise, and asks Stim to derive the DEM. This uses the QEC circuit
structure without using the physical device's noise model. The training program then
optimizes the reference DEM probabilities against detector firing rates and sampled
pair correlations.

Place each exported circuit in a directory named after the experiment. The circuit
must use the fixed filename `circuit.stim`.

```text
experiments/
└── d3-r3-default/
    ├── circuit.stim
    └── config.json       # Reserved for future simulator settings
```

```console
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python -m fdem --experiment d3-r3-default
```

The loss independently normalizes detector firing-rate MSE and off-diagonal
pair-correlation MSE, then adds them with a configurable pair weight. The default
`cobyla` method repeatedly samples the candidate DEM and minimizes the resulting
combined RMSE. To fit the same diagonal and off-diagonal statistics directly with
bounded linear least squares in parity log-rate space, select `pair-lsq`:

```console
python -m fdem --experiment d3-r3-default --training-method pair-lsq
```

To iteratively minimize the exact pair-probability MSE with its analytic log-rate
gradient, select `gradient-descent`:

```console
python -m fdem --experiment d3-r3-default --training-method gradient-descent
```

All methods start from the same uniformly noisy reference DEM. Detector firing rates
occupy the diagonal of the correlation matrix, allowing the direct fit and gradient
descent to update single-detector mechanisms as well as pair-visible mechanisms.

The generated figures are grouped by training method. For example, the default method
saves them as `var/d3-r3-default/cobyla/loss.png` and
`var/d3-r3-default/cobyla/ler.png`.

For PowerShell, activate the environment with `.\.venv\Scripts\Activate.ps1`.

## Local detector-correlation workbench

After installing the package with Python 3.11 or newer, start the workbench.

```console
python -m fdem.web
```

Open <http://127.0.0.1:8765/>. Use **Run simulation** to apply the selected
trap, timing, heating, cooling, and noise settings. **Run uniform DEM**
generates a phenomenological baseline. Both runs show a signed $p_{ij}$
heatmap and a circuit inspector. Download the PNG, the CSV in Stim detector
order, or the noisy Stim circuit. The uniform run also exports its DEM.

When using the exports for DEM fitting, verify that each detector ID and
logical observable refers to the same parity in the two circuits. For the
distance-3, three-round example, the one-LIZ and uniform runs share detector
order. The two-LIZ and uniform runs require a detector mapping first.

The launcher listens on this computer's loopback interface. Use
`python -m fdem.web --port 8766` to choose a different port. The browser
assets, compiler, circuit schedules, and default calibration table are
bundled with the package. [THeatA license and attribution](src/fdem/_theata/NOTICE).

## Calibrate a DEM from saved Stim circuits

```console
python -m fdem.calibrate --output-dem fitted.dem
```

The built-in benchmark uses a distance-3 rotated surface code with three
rounds. It samples a target Stim circuit with gate, reset, and measurement
error rates of `0.002`, `0.001`, and `0.02` and starts from a baseline
circuit with all three rates set to `0.001`.

Calibration uses the training shots' detector bits and the baseline DEM's
existing error mechanisms. It fits their probabilities jointly from detector
parities, including higher-order parities for baseline hyperedges. The
regularization penalty pulls the fitted log rates toward the baseline. For
mechanisms that flip the same detectors, the fit estimates their combined
rate and uses the baseline to allocate it. Rates for errors that flip only
logical observables come from the baseline.

The benchmark reports single-detector and pair-parity RMSE on held-out shots.
It also reports the logical failure count for each decoder on those same
shots. The target DEM in the built-in benchmark comes from the known target
circuit and is used as a reference in evaluation.

### Supply a hardware model

```console
python -m fdem.calibrate --target-circuit hardware.stim --baseline-circuit baseline.stim --output-dem fitted.dem
```

Use the same detector and logical observable definitions and ordering in
both circuits. The program checks detector coordinates and the observable
count. Use `--train-shots`, `--test-shots`, and `--seed` to control sampling
and `--regularization` to adjust the strength of the baseline prior.
For an external simulator, pass its boolean detector-event array to
`fdem.calibrate.fit_dem` with the baseline DEM. Measurements can be converted
to detector events with Stim's measurement-to-detection-event converter.

## Fit a DEM from detector-pair probabilities

Pass the device experiment and the separate `.dem` noise model you want to fit.
Use matching detector definitions and ordering in both inputs.

```console
python -m fdem.pij --experiment theata-d3-r5-pij --baseline-dem baseline.dem --output-dem fitted.dem --output-matrices pij.npz
```

The command estimates signed $p_{ij}$ from device syndrome shots and computes
the baseline DEM's matrix analytically from its error mechanisms. The optional
NPZ contains `device`, `baseline`, and `fitted` matrices in detector ID order.
Use `--shots` and `--seed` to control device sampling.

The fit minimizes the squared residuals in $-\log(1-2p_{ij})$ with nonnegative
mechanism log rates. It uses the upper triangle and skips pairs whose estimates
are undefined or reach $0.5$. Fitted mechanisms need baseline probabilities
below $0.5$. Mechanisms with fewer than two detector targets after cancellation,
including logical-only mechanisms, keep their baseline rates. Several mechanisms
can contribute to the same pair, so the pair data can leave individual rates
underdetermined.

For detector-event arrays collected outside the bundled device simulator, call
`fdem.correlations.detector_pair_correlations(syndromes)` and pass the resulting
matrix with the baseline DEM to `fdem.pij.fit_dem_from_pij`. Use
`fdem.pij.dem_pair_probabilities` to calculate the matrix for a DEM directly.

## Experimental eight-round surface-code correlations

The distance-3 logical Z memory exports include qubit-indexed JSON,
detector-indexed CSV, heatmaps, and a standalone comparison viewer.
Google Willow uses the first eight rounds of the released 10-round run.
The Shenzhen IQA / SUSTech run includes a final readout after eight rounds.
[Dataset links and JSON format](docs/qec-eight-round-correlations/PIJ_JSON_FORMAT.md)
