# Experiments

Store each THeatA export in a directory named after the experiment. The Stim circuit
always uses the fixed filename `circuit.stim`.

```text
experiments/
└── d3-r3-default/
    ├── circuit.stim
    └── config.json       # Reserved for future simulator settings
```

Select the directory by name when running the optimizer.

```console
python -m fdem --experiment d3-r3-default
```

Generated figures are grouped under the same experiment name.

```text
var/
└── d3-r3-default/
    ├── loss.png
    └── ler.png
```
