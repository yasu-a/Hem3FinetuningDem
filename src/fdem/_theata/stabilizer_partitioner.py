"""Stabilizer-to-LIZ partition for multi-LIZ compilation.

Maps each ancilla qubit to a specific LIZ zone, so that all stabilizers
measured by that ancilla are routed to the assigned LIZ.

For Reuse-1-per-LIZ in a multi-LIZ architecture, the input circuit must
provide as many ancillae as there are LIZs (e.g., d5_reuse2 + 2-LIZ).
"""

from dataclasses import dataclass
from typing import Dict, List


@dataclass
class StabilizerPartition:
    """Maps ancilla qubit ID -> LIZ zone ID.

    A None partition signals single-LIZ legacy behavior.
    """
    ancilla_to_liz: Dict[int, int]

    def get_liz(self, ancilla_id: int) -> int:
        return self.ancilla_to_liz[ancilla_id]

    def has(self, ancilla_id: int) -> bool:
        return ancilla_id in self.ancilla_to_liz

    def ancilla_ids(self) -> List[int]:
        return list(self.ancilla_to_liz.keys())

    def liz_ids(self) -> List[int]:
        """Distinct LIZ zone IDs in use."""
        return sorted(set(self.ancilla_to_liz.values()))


class StabilizerPartitioner:
    """Build a StabilizerPartition from circuit ancillae and architecture LIZs."""

    SUPPORTED_STRATEGIES = ("ancilla_locked",)

    def __init__(self, strategy: str = "ancilla_locked"):
        if strategy not in self.SUPPORTED_STRATEGIES:
            raise ValueError(
                f"Unknown partition strategy: {strategy}. "
                f"Supported: {self.SUPPORTED_STRATEGIES}"
            )
        self.strategy = strategy

    def partition(
        self,
        ancilla_qubits: List[int],
        liz_zone_ids: List[int],
    ) -> StabilizerPartition:
        if not ancilla_qubits:
            raise ValueError("ancilla_qubits is empty")
        if not liz_zone_ids:
            raise ValueError("liz_zone_ids is empty")

        if self.strategy == "ancilla_locked":
            return self._ancilla_locked(ancilla_qubits, liz_zone_ids)
        raise ValueError(f"Unknown partition strategy: {self.strategy}")

    def _ancilla_locked(
        self,
        ancilla_qubits: List[int],
        liz_zone_ids: List[int],
    ) -> StabilizerPartition:
        """Round-robin: ancilla i -> LIZ (i mod T).

        Uses the input circuit's ancilla ordering as-is to determine LIZ
        placement. If there are more ancillae than LIZs, multiple ancillae
        share a LIZ in a round-robin fashion.
        """
        mapping = {
            ancilla: liz_zone_ids[i % len(liz_zone_ids)]
            for i, ancilla in enumerate(ancilla_qubits)
        }
        return StabilizerPartition(ancilla_to_liz=mapping)
