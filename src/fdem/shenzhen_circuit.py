"""Gate-level Stim circuits for the Shenzhen distance-3 memory runs.

The Shenzhen archive (Zenodo 19908642) provides only a measurement-to-detector map
(``detector_mapping.stim``). This module inserts a rotated surface-code syndrome
extraction circuit before every ancilla measurement layer of that map, so Stim can
derive a detector error model whose detectors and observable match the hardware
records. Ancillas are never reset, as on the hardware; the map's detectors already
compare readouts two rounds apart for that reason.
"""

import argparse
from pathlib import Path

import stim


X_ANCILLA_CNOT_ORDER = ((1, 1), (-1, 1), (1, -1), (-1, -1))
"""Data-qubit offsets visited by X ancillas in the four CNOT layers."""

Z_ANCILLA_CNOT_ORDER = ((1, 1), (1, -1), (-1, 1), (-1, -1))
"""Data-qubit offsets visited by Z ancillas in the four CNOT layers."""


def _syndrome_extraction(
    coordinates: dict[int, tuple[int, int]],
    x_ancillas: list[int],
    z_ancillas: list[int],
) -> stim.Circuit:
    """Build one round of rotated surface-code syndrome extraction.

    Args:
        coordinates: Integer ``(x, y)`` coordinates keyed by qubit index.
        x_ancillas: Qubits measuring X stabilizers.
        z_ancillas: Qubits measuring Z stabilizers.

    Returns:
        A circuit with Hadamards on X ancillas around four CNOT layers.
    """
    qubit_at = {xy: qubit for qubit, xy in coordinates.items()}
    ancillas = set(x_ancillas) | set(z_ancillas)
    block = stim.Circuit()
    block.append("H", x_ancillas)
    block.append("TICK")
    for layer in range(4):
        targets = []
        for ancillas_of_type, order, ancilla_controls in (
            (x_ancillas, X_ANCILLA_CNOT_ORDER, True),
            (z_ancillas, Z_ANCILLA_CNOT_ORDER, False),
        ):
            dx, dy = order[layer]
            for ancilla in ancillas_of_type:
                x, y = coordinates[ancilla]
                data = qubit_at.get((x + dx, y + dy))
                if data is None or data in ancillas:
                    continue
                targets += [ancilla, data] if ancilla_controls else [data, ancilla]
        block.append("CX", targets)
        block.append("TICK")
    block.append("H", x_ancillas)
    block.append("TICK")
    return block


def build_circuit(detector_mapping: stim.Circuit) -> stim.Circuit:
    """Insert syndrome-extraction gates into a Shenzhen measurement-to-detector map.

    Data qubits sit at even coordinates and ancillas at odd coordinates. Z ancillas
    are identified by their detectors in round 0, which a Z-basis memory defines
    only for Z checks.

    Args:
        detector_mapping: The run's ``detector_mapping.stim`` circuit.

    Returns:
        A noiseless circuit with an initial reset of all qubits, gates before every
        ancilla measurement layer, and the map's measurements, detectors, and
        observable unchanged.

    Raises:
        ValueError: If the map contains repeat blocks, has no ancilla measurement
            layer, or yields detectors that are not deterministic.
    """
    if any(isinstance(item, stim.CircuitRepeatBlock) for item in detector_mapping):
        raise ValueError("Detector mapping must not contain repeat blocks")

    coordinates = {
        qubit: (int(values[0]), int(values[1]))
        for qubit, values in detector_mapping.get_final_qubit_coordinates().items()
    }
    data = {qubit for qubit, (x, y) in coordinates.items() if x % 2 == 0 and y % 2 == 0}
    round_zero_checks = {
        (int(values[0]), int(values[1]))
        for values in detector_mapping.get_detector_coordinates().values()
        if values[2] == 0
    }
    ancillas = sorted(set(coordinates) - data)
    z_ancillas = [qubit for qubit in ancillas if coordinates[qubit] in round_zero_checks]
    x_ancillas = [qubit for qubit in ancillas if qubit not in z_ancillas]
    extraction = _syndrome_extraction(coordinates, x_ancillas, z_ancillas)

    circuit = stim.Circuit()
    reset_done = False
    ancilla_layers = 0
    for instruction in detector_mapping:
        if instruction.name == "M":
            measured = {target.qubit_value for target in instruction.targets_copy()}
            if not reset_done:
                circuit.append("R", sorted(coordinates))
                circuit.append("TICK")
                reset_done = True
            if measured <= set(ancillas):
                circuit += extraction
                ancilla_layers += 1
        circuit.append(instruction)
    if ancilla_layers == 0:
        raise ValueError("Detector mapping has no ancilla measurement layer")

    # Raises if any detector or the observable is not deterministic without noise.
    circuit.detector_error_model()
    return circuit


def main() -> None:
    """Write the gate-level circuit for one Shenzhen detector map.

    Returns:
        None.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("detector_mapping", type=Path, help="detector_mapping.stim")
    parser.add_argument("--output", type=Path, required=True, help="Output circuit path")
    arguments = parser.parse_args()
    mapping = stim.Circuit.from_file(arguments.detector_mapping)
    build_circuit(mapping).to_file(arguments.output)
    print(f"circuit = {arguments.output}")


if __name__ == "__main__":
    main()
