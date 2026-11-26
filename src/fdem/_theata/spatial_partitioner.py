# Adapted from THeatA for the bundled workbench.
"""Spatial partitioning for multi-LIZ compilation.

Rewrites the input circuit so that each stabilizer is measured by the ancilla
whose assigned LIZ is "spatially nearest" to the stabilizer's data qubits.
This reduces inter-LIZ ion transport (and therefore LIZ-passthrough congestion).

Approach:
  1. Identify stabilizer boundaries (RESET → ... → MEASURE blocks per ancilla)
  2. For each stabilizer, compute a spatial center from its data qubits
  3. Assign the stabilizer to a LIZ partition based on the center
  4. Rewrite the operations: replace the original ancilla with the LIZ's ancilla

Caveat (5/4 noted by author): this approach exploits the human-friendly
topology of rotated surface code (data qubits laid out on a 2D grid).
Generalization to other code families is future work.
"""

import dataclasses
from typing import Dict, List, Set, Tuple
from dataclasses import dataclass

from .ast_def import (
    Program, Operation, GateOp, ResetOp, MeasureOp,
    ShuttleOp, SwapOp, SplitOp, MergeOp, BarrierOp,
    QubitDecl, Location,
)


@dataclass
class StabilizerInfo:
    """A single stabilizer measurement block in the program."""
    op_indices: List[int]          # indices into program.instructions
    ancilla_original: int          # ancilla used in the input circuit
    data_qubits: Set[int]          # data qubit IDs touched by MS gates


@dataclass
class SpatialPartitionResult:
    """Output of spatial partitioning."""
    new_program: Program                       # program with rewritten ancillae
    ancilla_assignments: Dict[int, int]        # liz_zone_id -> ancilla qubit ID (post-rewrite)
    stabilizer_to_liz: Dict[int, int]          # stabilizer index -> liz_zone_id
    data_to_liz_set: Dict[int, Set[int]]       # data qubit ID -> set of liz_zone_ids that use it
    pure_data: Dict[int, int]                  # data qubit ID -> the single liz_zone_id
    shared_data: Set[int]                      # data qubit IDs used by >1 LIZ


def _identify_stabilizers(
    program: Program,
    ancilla_qubits: Set[int],
) -> List[StabilizerInfo]:
    """Walk the program and identify RESET → ... → MEASURE blocks per ancilla.

    第 1 パス: ancilla を target に含む op を、その ancilla の開いている
    stabilizer に帰属させる。
    第 2 パス: どの stabilizer にも属さなかった op (data qubit への R 回転
    など、ancilla を含まないもの) を「同じ qubit を使う直近の MS」の
    stabilizer に帰属させる。従来はこれらが黙って脱落しており、partition
    経由のコンパイルから data R 回転が消えていた。
    """
    stabilizers: List[StabilizerInfo] = []
    active: Dict[int, StabilizerInfo] = {}  # ancilla -> current open stabilizer
    uncovered: List[int] = []

    for idx, op in enumerate(program.instructions):
        if isinstance(op, ResetOp):
            for t in op.targets:
                if t in ancilla_qubits:
                    # New stabilizer for this ancilla
                    active[t] = StabilizerInfo(
                        op_indices=[idx],
                        ancilla_original=t,
                        data_qubits=set(),
                    )
            continue

        if isinstance(op, MeasureOp):
            for t in op.targets:
                if t in ancilla_qubits and t in active:
                    active[t].op_indices.append(idx)
                    stabilizers.append(active.pop(t))
            continue

        # MS/R/Other: attribute to the active stabilizer whose ancilla is involved
        owners = []
        if isinstance(op, GateOp):
            owners = [t for t in op.targets if t in active]
        else:
            owners = [t for t in getattr(op, "targets", []) if t in active]

        if not owners:
            uncovered.append(idx)
            continue

        owner = owners[0]
        active[owner].op_indices.append(idx)
        if isinstance(op, GateOp) and op.type == "MS":
            for t in op.targets:
                if t != owner:
                    active[owner].data_qubits.add(t)

    # --- 第 2 パス: 無帰属 op を「同じ qubit の直近 MS」の stabilizer へ ---
    if uncovered:
        # qubit -> sorted list of (ms_op_index, stabilizer)
        ms_by_qubit: Dict[int, List] = {}
        for stab in stabilizers:
            for op_idx in stab.op_indices:
                op = program.instructions[op_idx]
                if isinstance(op, GateOp) and op.type == "MS":
                    for t in op.targets:
                        ms_by_qubit.setdefault(t, []).append((op_idx, stab))
        for lst in ms_by_qubit.values():
            lst.sort(key=lambda x: x[0])

        for idx in uncovered:
            op = program.instructions[idx]
            targets = getattr(op, "targets", [])
            stab = None
            for t in targets:
                entries = ms_by_qubit.get(t, [])
                after = [s for i, s in entries if i > idx]
                before = [s for i, s in entries if i < idx]
                stab = after[0] if after else (before[-1] if before else None)
                if stab is not None:
                    break
            if stab is None:
                raise ValueError(
                    f"stabilizer 識別: op #{idx} ({op}) をどの stabilizer にも"
                    f"帰属できません。partition 対象の回路は stabilizer 測定"
                    f"のみで構成されている必要があります。")
            stab.op_indices.append(idx)

        for stab in stabilizers:
            stab.op_indices.sort()

    return stabilizers


def _classify_stabilizer_row(
    stab: StabilizerInfo,
    qubit_row: Dict[int, float],
    code_distance: int,
    num_partitions: int,
) -> int:
    """Return partition index (0..num_partitions-1) based on average row of data qubits.

    Row range [0, d-1] is split evenly into num_partitions bands.
    """
    if not stab.data_qubits:
        return 0
    avg_row = sum(qubit_row[q] for q in stab.data_qubits) / len(stab.data_qubits)
    band = int(avg_row * num_partitions / code_distance)
    return min(band, num_partitions - 1)


def _replace_targets(op: Operation, mapping: Dict[int, int]) -> Operation:
    """Return a new op with targets remapped via `mapping`."""
    if not hasattr(op, "targets"):
        return op
    new_targets = [mapping.get(t, t) for t in op.targets]
    if new_targets == op.targets:
        return op
    return dataclasses.replace(op, targets=new_targets)


def partition_spatial_row(
    program: Program,
    ancilla_qubits: List[int],
    liz_zone_ids: List[int],
    code_distance: int,
    phase_alternation: bool = False,
) -> SpatialPartitionResult:
    """Build a row-band spatial partition.

    Args:
        program: parsed input program
        ancilla_qubits: list of ancilla qubit IDs in the program (e.g., [25, 26] for d5_reuse2)
        liz_zone_ids: list of LIZ zone IDs, ordered by physical position (left to right)
        code_distance: side length of the rotated surface code grid
        phase_alternation: if True, reorder each LIZ's stabilizer chain so that
            LIZs with even index do pure stabilizers first then shared,
            while LIZs with odd index do shared first then pure.
            This anti-correlates central-memory accesses across the two LIZs:
            while LIZ-A processes its 7 pure stabilizers (no central-M traffic),
            LIZ-B processes its 5 shared stabilizers (using central-M), and
            vice versa.

    Stabilizers are split into len(liz_zone_ids) horizontal bands by data-qubit
    row average. Within each band, the stabilizer is reassigned to the
    band-corresponding ancilla (and therefore the corresponding LIZ).
    """
    # Build row lookup from declarations: location.trap == row in our convention
    qubit_row: Dict[int, float] = {}
    for decl in program.declarations:
        qubit_row[decl.id] = float(decl.location.trap)

    ancilla_set = set(ancilla_qubits)
    stabilizers = _identify_stabilizers(program, ancilla_set)

    # 完全性チェック: 入力の全 op がいずれかの stabilizer に帰属していること。
    # (黙って op を落とすと「動くが物理的に別の回路」になり検出困難なため)
    covered = set()
    for stab in stabilizers:
        covered.update(stab.op_indices)
    if len(covered) != len(program.instructions):
        missing = [i for i in range(len(program.instructions)) if i not in covered]
        raise ValueError(
            f"partition_spatial_row: {len(missing)} 個の op が stabilizer に"
            f"帰属していません (先頭: #{missing[0]} = "
            f"{program.instructions[missing[0]]})")

    # Assign one reusable ancilla per LIZ. The number of ancillae is a compile
    # choice (= number of LIZs), not a circuit property, so if the circuit
    # declares fewer ancillae than LIZs we allocate fresh ancilla qubit IDs.
    # Ancillae are reset/reused and fungible; the per-stabilizer rewrite below
    # remaps each band's RESET/MS/MEASURE to its LIZ ancilla, so allocating a
    # fresh ID needs only a QUBIT declaration (no extra ops).
    existing_ids = {decl.id for decl in program.declarations}
    next_id = (max(existing_ids) + 1) if existing_ids else 0
    park_trap = max((decl.location.trap for decl in program.declarations),
                    default=0) + 1
    extra_decls: List[QubitDecl] = []
    liz_ancilla: Dict[int, int] = {}
    for i, liz in enumerate(liz_zone_ids):
        if i < len(ancilla_qubits):
            liz_ancilla[liz] = ancilla_qubits[i]
        else:
            aid = next_id
            next_id += 1
            liz_ancilla[liz] = aid
            extra_decls.append(
                QubitDecl(id=aid, location=Location(trap=park_trap,
                                                    site=len(extra_decls)))
            )

    # Classify each stabilizer
    stab_to_liz: Dict[int, int] = {}
    rewrite_per_stab: List[Dict[int, int]] = []  # per-stabilizer ancilla rewrite map
    for s_idx, stab in enumerate(stabilizers):
        band = _classify_stabilizer_row(stab, qubit_row, code_distance, len(liz_zone_ids))
        target_liz = liz_zone_ids[band]
        target_ancilla = liz_ancilla[target_liz]
        stab_to_liz[s_idx] = target_liz
        rewrite_per_stab.append({stab.ancilla_original: target_ancilla})

    # Build data-qubit usage info (needed for phase classification)
    data_to_liz: Dict[int, Set[int]] = {}
    for s_idx, stab in enumerate(stabilizers):
        liz = stab_to_liz[s_idx]
        for d in stab.data_qubits:
            data_to_liz.setdefault(d, set()).add(liz)
    pure_data = {d: next(iter(s)) for d, s in data_to_liz.items() if len(s) == 1}
    shared_data = {d for d, s in data_to_liz.items() if len(s) > 1}

    # Determine emission order per LIZ chain
    # liz_to_stab_order[liz_id] = ordered list of stabilizer indices to emit
    liz_to_stab_order: Dict[int, List[int]] = {liz: [] for liz in liz_zone_ids}
    for s_idx in range(len(stabilizers)):
        liz_to_stab_order[stab_to_liz[s_idx]].append(s_idx)

    if phase_alternation:
        for liz_idx, liz in enumerate(liz_zone_ids):
            indices = liz_to_stab_order[liz]
            pure_idx = [
                i for i in indices
                if not any(d in shared_data for d in stabilizers[i].data_qubits)
            ]
            shared_idx = [
                i for i in indices
                if any(d in shared_data for d in stabilizers[i].data_qubits)
            ]
            if liz_idx % 2 == 0:
                liz_to_stab_order[liz] = pure_idx + shared_idx
            else:
                liz_to_stab_order[liz] = shared_idx + pure_idx

    # Emit new instruction list: chain by chain (stabilizer_parallel makes
    # cross-chain order irrelevant for DAG dependencies)
    if phase_alternation:
        new_instructions: List[Operation] = []
        for liz in liz_zone_ids:
            for s_idx in liz_to_stab_order[liz]:
                stab = stabilizers[s_idx]
                mapping = rewrite_per_stab[s_idx]
                for op_idx in stab.op_indices:
                    new_instructions.append(
                        _replace_targets(program.instructions[op_idx], mapping)
                    )
    else:
        # 入力順そのままの保存は不可: 入力で時間的にインターリーブしている
        # 2 つの stabilizer (reuse 回路の ancilla 交互使用) が同じ LIZ の
        # 同じ ancilla に書き換えられると、RESET が二重に開いて後段の
        # stabilizer 識別が壊れる (op が黙って脱落する)。
        # stabilizer 単位で入力初出順に直列化して emit する。boustrophedon の
        # compile_per_liz は後段で幾何順に並べ替えるため命令順への依存はない。
        order = sorted(range(len(stabilizers)),
                       key=lambda i: stabilizers[i].op_indices[0])
        new_instructions = []
        for s_idx in order:
            stab = stabilizers[s_idx]
            mapping = rewrite_per_stab[s_idx]
            for op_idx in stab.op_indices:
                new_instructions.append(
                    _replace_targets(program.instructions[op_idx], mapping)
                )

    new_program = Program(
        declarations=list(program.declarations) + extra_decls,
        instructions=new_instructions,
    )

    return SpatialPartitionResult(
        new_program=new_program,
        ancilla_assignments=liz_ancilla,
        stabilizer_to_liz=stab_to_liz,
        data_to_liz_set=data_to_liz,
        pure_data=pure_data,
        shared_data=shared_data,
    )


def compute_initial_placement_3tier(
    result: SpatialPartitionResult,
    architecture,
    all_data_qubits: List[int],
) -> Dict[int, int]:
    """Compute an N-LIZ initial placement (T-M-T-(L-T-M-T)* chain).

    Pure-i data   → memory zone immediately left of LIZ-i
                    (rightmost LIZ's pure data → memory right of it)
    Shared data   → memory zone BETWEEN the specific adjacent LIZ pair that
                    shares it (from data_to_liz_set), so it is one hop from
                    both and never needs to transit an occupied LIZ
    Each ancilla  → memory zone immediately left of its LIZ

    For the chain zones [T,M,T,L,T,M,T,L,...,T,M,T], memory_zones[i] is left
    of liz_zones[i], and memory_zones[i+1] sits between liz_zones[i] and
    liz_zones[i+1] (their shared memory).
    """
    memory_zones = [z.id for z in architecture.zones if z.zone_type == "M"]
    liz_zones = [z.id for z in architecture.zones if z.zone_type == "L"]
    num_liz = len(liz_zones)

    # Map each LIZ to the memory zone immediately to its left
    liz_left_memory = {liz_zones[i]: memory_zones[i] for i in range(num_liz)}
    liz_index = {liz_zones[i]: i for i in range(num_liz)}
    rightmost_memory = memory_zones[-1]

    placement: Dict[int, int] = {}

    # Ancillae: place each in the memory immediately left of its LIZ
    for liz_id, ancilla_id in result.ancilla_assignments.items():
        placement[ancilla_id] = liz_left_memory[liz_id]

    # Data qubits: distribute by classification
    for d in all_data_qubits:
        if d in result.pure_data:
            liz_id = result.pure_data[d]
            if liz_id == liz_zones[-1]:
                placement[d] = rightmost_memory
            else:
                placement[d] = liz_left_memory[liz_id]
        elif d in result.shared_data:
            # Place in the memory between the specific LIZ pair sharing it.
            visited = sorted(liz_index[l] for l in result.data_to_liz_set[d])
            lo, hi = visited[0], visited[-1]
            if hi == lo + 1:
                # adjacent pair -> shared memory sits at memory_zones[hi]
                placement[d] = memory_zones[hi]
            else:
                # non-adjacent / >2-LIZ span: no single memory is adjacent to
                # all visited LIZs -> pass-through is unavoidable with static
                # placement. Park mid-span (best effort); the compiler's
                # pass-through guard will flag it if transport crosses an
                # occupied LIZ. (Does not occur for rotated surface-code
                # row-band partitions, where sharing is between adjacent bands.)
                placement[d] = memory_zones[(lo + hi + 1) // 2]
        else:
            # Data qubit not touched by any stabilizer (shouldn't happen)
            placement[d] = memory_zones[0]

    return placement
