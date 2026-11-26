# fdem

A minimal example that estimates the logical error rate (LER) of a rotated surface
code using a Stim detector error model (DEM). The physical-device circuit supports
five cumulative noise models:

- Model 0: symmetric depolarizing noise
- Model 1: asymmetric two-qubit Pauli noise
- Model 2: pair-dependent spatial variation
- Model 3: QEC-round-dependent temporal growth
- Model 4: idle-qubit dephasing per tick interval

Select the model and edit its physical parameters in `src/fdem/__main__.py`. Model 0
uses the original uniform error rate by default. Set
`MODEL_0_USE_LEGACY_UNIFORM_NOISE = False` to compare it with the same separate gate,
reset, and measurement probabilities used by the other models.

## Modeling assumptions

The asymmetric two-qubit Pauli weights are validation assumptions, not parameters
derived from hardware or THeatA. Spatial coefficients are sampled once per unordered
qubit pair and reused in every QEC round. Temporal growth uses the QEC round index,
starting from zero, rather than the `TICK` index.

For Model 4, an idle interval runs from circuit initialization or the previous `TICK`
to the next `TICK` or circuit termination. A used physical qubit receives an idle
`Z_ERROR` only when no gate, reset, or measurement acts on it in that interval. There
is no modeled time before initialization or after circuit termination.

Models 1 through 4 require Stim's disjoint-error approximation when converting
`PAULI_CHANNEL_2` into a DEM because the channel's mutually exclusive branches cannot
be represented exactly as independent DEM error mechanisms. The code enables the
approximation only when such channels are present and limits it to the largest total
Pauli-channel probability in the circuit. Model 0 does not enable this approximation.

```console
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python -m fdem
```

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
