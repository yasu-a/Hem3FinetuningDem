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
.\.venv\Scripts\Activate.ps1
python -m pip install -e .
python -m fdem
```
