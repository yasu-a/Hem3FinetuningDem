"""Tests for the selectable physical-device noise models."""

from collections import defaultdict

import numpy as np
import pymatching
import pytest
import stim

from fdem.__main__ import build_dem
from fdem.device import build_physical_device_circuit


def _targets(instruction: stim.CircuitInstruction) -> tuple[int, ...]:
    """Extract qubit targets from an instruction.

    Args:
        instruction: Stim instruction to inspect.

    Returns:
        Physical-qubit target indices.
    """
    return tuple(
        target.value
        for target in instruction.targets_copy()
        if target.is_qubit_target
    )


def _two_qubit_channels(
    circuit: stim.Circuit,
) -> list[tuple[tuple[int, ...], np.ndarray]]:
    """Extract tagged two-qubit Pauli channels from a circuit.

    Args:
        circuit: Physical-device circuit to inspect.

    Returns:
        Pairs of channel targets and Pauli probabilities.
    """
    return [
        (_targets(instruction), np.array(instruction.gate_args_copy()))
        for instruction in circuit.flattened()
        if instruction.name == "PAULI_CHANNEL_2"
        and instruction.tag == "fdem_two_qubit_noise"
    ]


def test_model_zero_legacy_configuration_is_unchanged() -> None:
    """Model 0 with the legacy shortcut exactly matches the original circuit."""
    actual = build_physical_device_circuit(3, 3, 0.001, model=0)
    expected = stim.Circuit.generated(
        "surface_code:rotated_memory_x",
        distance=3,
        rounds=3,
        after_clifford_depolarization=0.001,
        after_reset_flip_probability=0.001,
        before_measure_flip_probability=0.001,
    )

    assert actual == expected


@pytest.mark.parametrize("model", range(5))
def test_all_models_support_sampling_dem_conversion_and_decoding(model: int) -> None:
    """Every model preserves circuit dimensions and supports the full decode path."""
    reference = stim.Circuit.generated(
        "surface_code:rotated_memory_x", distance=3, rounds=2
    )
    circuit = build_physical_device_circuit(3, 2, model=model)

    assert circuit.num_detectors == reference.num_detectors
    assert circuit.num_observables == reference.num_observables

    detectors, observables = circuit.compile_detector_sampler(seed=10).sample(
        shots=20,
        separate_observables=True,
    )
    dem = build_dem(circuit)
    decoder = pymatching.Matching.from_detector_error_model(dem)
    predictions = decoder.decode_batch(detectors)

    assert detectors.shape == (20, reference.num_detectors)
    assert observables.shape == predictions.shape == (20, reference.num_observables)
    assert dem.num_detectors == reference.num_detectors
    assert dem.num_observables == reference.num_observables


def test_asymmetric_pauli_probabilities_have_requested_weights_and_total() -> None:
    """Model 1 normalizes the requested Pauli weights to the baseline error rate."""
    circuit = build_physical_device_circuit(
        3,
        1,
        model=1,
        two_qubit_error_probability=0.005,
    )
    _, probabilities = _two_qubit_channels(circuit)[0]
    weights = np.array([4, 4, 10, 4, 30, 4, 4, 4, 4, 4, 4, 10, 4, 4, 6])

    assert probabilities.sum() == pytest.approx(0.005)
    assert probabilities / probabilities.sum() == pytest.approx(weights / weights.sum())


def test_spatial_variation_is_fixed_for_each_pair() -> None:
    """Model 2 reuses one random spatial coefficient for a pair in every round."""
    circuit = build_physical_device_circuit(3, 3, model=2, seed=7)
    totals_by_pair: dict[tuple[int, ...], list[float]] = defaultdict(list)
    for pair, probabilities in _two_qubit_channels(circuit):
        totals_by_pair[pair].append(float(probabilities.sum()))

    assert all(len(totals) == 3 for totals in totals_by_pair.values())
    assert all(np.allclose(totals, totals[0]) for totals in totals_by_pair.values())
    assert len({round(totals[0], 12) for totals in totals_by_pair.values()}) > 1


def test_temporal_growth_uses_qec_round_number() -> None:
    """Model 3 applies temporal growth once per QEC round instead of per tick."""
    circuit = build_physical_device_circuit(
        3,
        3,
        model=3,
        temporal_growth_coefficient=0.2,
        seed=7,
    )
    totals_by_pair: dict[tuple[int, ...], list[float]] = defaultdict(list)
    for pair, probabilities in _two_qubit_channels(circuit):
        totals_by_pair[pair].append(float(probabilities.sum()))

    for totals in totals_by_pair.values():
        assert np.array(totals) / totals[0] == pytest.approx([1.0, 1.2, 1.4])


def test_idle_noise_targets_only_inactive_used_qubits() -> None:
    """Model 4 applies idle Z errors to exactly the inactive qubits per interval."""
    circuit = build_physical_device_circuit(3, 2, model=4)
    used_qubits = {
        target
        for instruction in circuit
        if instruction.name == "QUBIT_COORDS"
        for target in _targets(instruction)
    }
    active_qubits: set[int] = set()
    idle_qubits: set[int] = set()

    for instruction in circuit:
        if instruction.name == "TICK":
            assert idle_qubits == used_qubits - active_qubits
            active_qubits.clear()
            idle_qubits.clear()
        elif instruction.tag == "fdem_idle_noise":
            idle_qubits.update(_targets(instruction))
        elif not instruction.tag.startswith("fdem_") and instruction.name != "QUBIT_COORDS":
            active_qubits.update(_targets(instruction))

    assert idle_qubits == used_qubits - active_qubits


def test_spatial_noise_is_reproducible_from_seed() -> None:
    """The same seed reproduces a circuit while another seed changes pair noise."""
    first = build_physical_device_circuit(3, 2, model=2, seed=11)
    repeated = build_physical_device_circuit(3, 2, model=2, seed=11)
    changed = build_physical_device_circuit(3, 2, model=2, seed=12)

    assert first == repeated
    assert first != changed


@pytest.mark.parametrize(
    ("keyword", "value"),
    [
        ("one_qubit_error_probability", -0.1),
        ("two_qubit_error_probability", 1.1),
        ("reset_error_probability", np.nan),
        ("measurement_error_probability", np.inf),
        ("idle_z_error_probability", -0.1),
        ("spatial_variation", 1.1),
        ("temporal_growth_coefficient", -0.1),
    ],
)
def test_invalid_physical_parameters_are_rejected(keyword: str, value: float) -> None:
    """Invalid physical parameters raise an explicit exception."""
    with pytest.raises(ValueError):
        build_physical_device_circuit(3, 2, **{keyword: value})
