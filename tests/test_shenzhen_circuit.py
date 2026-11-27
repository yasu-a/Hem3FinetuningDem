from pathlib import Path

import pytest
import stim

from fdem.shenzhen_circuit import build_circuit


SHENZHEN_RUNS = Path(__file__).resolve().parents[1] / "data/ler/shenzhen_qec_19908642/Z"


@pytest.mark.parametrize("rounds", [8, 10, 13, 20])
def test_committed_circuit_is_rebuilt_from_its_mapping(rounds):
    """The committed circuit equals the one rebuilt from the committed detector map."""
    run = SHENZHEN_RUNS / f"r{rounds}"
    mapping = stim.Circuit.from_file(run / "detector_mapping.stim")

    circuit = build_circuit(mapping)

    assert circuit == stim.Circuit.from_file(run / "circuit.stim")
    assert circuit.num_detectors == mapping.num_detectors
    assert circuit.num_observables == 1


def test_noiseless_circuit_has_no_detection_events():
    """Without noise, no detector fires and the observable never flips."""
    mapping = stim.Circuit.from_file(SHENZHEN_RUNS / "r8/detector_mapping.stim")
    sampler = build_circuit(mapping).compile_detector_sampler(seed=1)

    detectors, observables = sampler.sample(100, separate_observables=True)

    assert not detectors.any()
    assert not observables.any()


def test_gates_are_inserted_only_before_ancilla_measurements():
    """Each ancilla measurement layer gets four CNOT layers; the data readout gets none."""
    mapping = stim.Circuit.from_file(SHENZHEN_RUNS / "r8/detector_mapping.stim")

    circuit = build_circuit(mapping)

    assert sum(instruction.name == "CX" for instruction in circuit) == 4 * 8
    assert sum(instruction.name == "R" for instruction in circuit) == 1


def test_mapping_without_ancilla_layer_is_rejected():
    """A map that measures only data qubits cannot define syndrome extraction."""
    mapping = stim.Circuit("""
        QUBIT_COORDS(0, 0) 0
        QUBIT_COORDS(1, 1) 1
        M 0
    """)

    with pytest.raises(ValueError, match="no ancilla measurement layer"):
        build_circuit(mapping)
