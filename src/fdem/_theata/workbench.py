"""THeatA workbench model adapted for use from the bundled package."""

import csv
import io
import math
from dataclasses import asdict
from importlib import import_module
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import stim

from ..correlations import estimate_pij
from .ast_def import (
    BarrierOp, CoolOp, GateOp, MeasureOp, MergeOp, Operation, ResetOp,
    ShuttleOp, SplitOp, SwapOp,
)
from .dag_builder import DAGBuilder
from .ion_timeline import gaps_from_compile_results, gaps_from_per_liz_results
from .noise import (
    default_noise_params,
    inject_noise_approach_b,
    inject_per_ion_idle,
    phonon_to_pauli_lookup_rates,
)
from .parallel_estimator import ParallelEstimator, format_timesteps_text
from .pauli_lookup import PauliErrorLookup, STIM_PAULI_ORDER
from .per_liz_compiler import compile_per_liz, estimate_streams, validate_merged
from .qccd_compiler import CompilerConfig, QCCDArchitecture, QCCDCompiler
from .qccd_parser import parse
from .sim_params import load_time_parameters
from .stim_recompiler import recompile_multi_round


# The reference logical Z observable is the top row of data qubits.
DISTANCE_CIRCUITS = {
    3: ('snake_reuse1', 'snake_reuse1_reverse', [0, 1, 2]),
    5: ('d5_reuse1', 'd5_reuse1_reverse_cnot', [0, 1, 2, 3, 4]),
    7: ('d7_reuse1', 'd7_reuse1_reverse_cnot', [0, 1, 2, 3, 4, 5, 6]),
    9: ('d9_reuse1', 'd9_reuse1_reverse_cnot', [0, 1, 2, 3, 4, 5, 6, 7, 8]),
    11: ('d11_reuse1', 'd11_reuse1_reverse_cnot', list(range(11))),
    13: ('d13_reuse1', 'd13_reuse1_reverse_cnot', list(range(13))),
}
PAULI_LOOKUP = PauliErrorLookup()
NOISE_FIELDS = ("p_R", "p_meas", "p_reset", "gamma_phi_per_s")
TIMING_DEFAULTS = asdict(load_time_parameters())
MAX_DETECTORS = 768



def placement_greedy_alt(data_qubits, ancilla_qubits, circuit_module):
    """Place ions alternately from the interaction-zone sides in gate order."""
    program = parse(circuit_module.CIRCUIT)
    assigned = set()
    left_order = []
    right_order = []
    zones = [1, 5]
    zone_idx = 0

    for op in program.instructions:
        targets = []
        if hasattr(op, 'targets'):
            targets = op.targets
        for qid in targets:
            if qid not in assigned:
                assigned.add(qid)
                zone = zones[zone_idx]
                if zone == 1:
                    left_order.insert(0, qid)
                else:
                    right_order.append(qid)
                zone_idx = 1 - zone_idx

    for qid in list(data_qubits) + list(ancilla_qubits):
        if qid not in assigned:
            assigned.add(qid)
            zone = zones[zone_idx]
            if zone == 1:
                left_order.insert(0, qid)
            else:
                right_order.append(qid)
            zone_idx = 1 - zone_idx

    initial = {}
    for qid in left_order:
        initial[qid] = 1
    for qid in right_order:
        initial[qid] = 5
    return initial


def format_op_summary(op: Operation) -> str:
    """Format one operation in a multi-LIZ schedule."""
    if isinstance(op, GateOp):
        ions = ",".join(str(t) for t in op.targets)
        return f"{op.type}({ions}) z{op.zone_id}"
    if isinstance(op, ResetOp):
        ions = ",".join(str(t) for t in op.targets)
        return f"Reset({ions}) z{op.zone_id}"
    if isinstance(op, MeasureOp):
        ions = ",".join(str(t) for t in op.targets)
        return f"Measure({ions}) z{op.zone_id}"
    if isinstance(op, SplitOp):
        return f"Split(ion{op.ion_id}) z{op.src_zone_id}->z{op.dst_zone_id}"
    if isinstance(op, MergeOp):
        return f"Merge z{op.zone_left}+z{op.zone_right}"
    if isinstance(op, SwapOp):
        ions = ",".join(str(t) for t in op.targets)
        return f"Swap({ions}) z{op.zone_id}"
    if isinstance(op, ShuttleOp):
        return f"Shuttle(ion{op.target}) z{op.src_zone}->{op.dst}"
    if isinstance(op, BarrierOp):
        ions = ",".join(str(t) for t in op.targets)
        return f"Barrier({ions})"
    if isinstance(op, CoolOp):
        coolant = f" coolant={op.coolant_id}" if op.coolant_id >= 0 else ""
        return f"Cool z{op.zone_id}{coolant}"
    return str(op)


def defaults():
    """Return the distances and initial hardware settings for the form."""
    return {
        "distances": list(DISTANCE_CIRCUITS),
        "max_detectors": MAX_DETECTORS,
        "zones": [[z.zone_type, z.capacity] for z in QCCDArchitecture.simple_mlm().zones],
        "timing": TIMING_DEFAULTS,
        "noise": {key: value for key, value in default_noise_params().items() if key in NOISE_FIELDS},
    }


def integer(value, label, minimum, maximum):
    """Read an integer within the supported bounds."""
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError(f"{label} must be an integer from {minimum:,} to {maximum:,}.")
    return value


def number(value, label, maximum=math.inf):
    """Read a finite nonnegative value within the supported bounds."""
    if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= maximum:
        expected = "a finite nonnegative number" if maximum == math.inf else f"a number from 0 to {maximum:g}"
        raise ValueError(f"Enter {expected} for {label}.")
    return float(value)


def architecture(zones):
    """Validate the user-selected zone layout and construct a QCCD architecture."""
    if not isinstance(zones, list) or not 3 <= len(zones) <= 25:
        raise ValueError("Enter a linear layout of 3 to 25 zones.")
    specs = []
    for zone in zones:
        if not isinstance(zone, list) or len(zone) != 2 or zone[0] not in ("T", "M", "L"):
            raise ValueError("Each zone needs a type (T, M, or L) and an ion capacity.")
        kind, size = zone
        specs.append((kind, integer(size, "Zone capacity", 2 if kind == "L" else 1, 200)))
    types = "".join(kind for kind, _ in specs)
    if types.count("L") not in (1, 2) or "M" not in types:
        raise ValueError("The workbench supports one or two LIZs and requires a memory zone.")
    if types.count("L") == 2 and types != "TMTLTMTLTMT":
        raise ValueError("Two LIZs use the T–M–T–L–T–M–T–L–T–M–T layout.")
    return QCCDArchitecture(specs)


def initial_chains(chains, arch, code_ions, coolant_ids):
    """Validate custom ordered ion chains, or request automatic placement."""
    if chains is None:
        return None
    if not isinstance(chains, dict):
        raise ValueError("Placement needs one ordered chain per occupied zone.")
    placement = {}
    seen = set()
    for zone_id, ions in chains.items():
        zone = arch.get_zone(int(zone_id))
        if zone is None or zone.zone_type == "T":
            raise ValueError(f"Use a memory or interaction zone for starting chain {zone_id}.")
        if not isinstance(ions, list) or len(ions) > zone.capacity:
            raise ValueError(f"Starting chain z{zone_id} exceeds its capacity.")
        for ion in ions:
            if type(ion) is not int or ion in seen:
                raise ValueError("Each ion must appear exactly once in the starting chains.")
            seen.add(ion)
        placement[zone.id] = ions
    if seen != set(code_ions) | set(coolant_ids):
        raise ValueError("Place each data, ancilla, and coolant ion exactly once.")
    return placement


def load_lookup(contents):
    """Load a validated custom calibration CSV or use the bundled table."""
    if contents is None:
        return PAULI_LOOKUP
    if not isinstance(contents, str):
        raise ValueError("The MS calibration must be a CSV file.")
    rows = csv.DictReader(io.StringIO(contents.lstrip("\ufeff")))
    if not {"n_bar", "pauli", "probability"}.issubset(rows.fieldnames or []):
        raise ValueError("MS calibration columns must include n_bar, pauli, and probability.")
    sampled = {}
    for row in rows:
        n_bar = number(float(row["n_bar"]), "n_bar")
        pauli = row["pauli"]
        probability = number(float(row["probability"]), "Pauli probability", 1)
        probabilities = sampled.setdefault(n_bar, {})
        if pauli not in (*STIM_PAULI_ORDER, "II") or pauli in probabilities:
            raise ValueError(f"Invalid or duplicate Pauli label at n_bar={n_bar:g}.")
        probabilities[pauli] = probability
    if not sampled or any(
        not set(STIM_PAULI_ORDER).issubset(p) or sum(p.values()) > 1 + 1e-10
        for p in sampled.values()
    ):
        raise ValueError("Each n_bar needs all 15 nonidentity Pauli probabilities with sum at most 1.")
    with TemporaryDirectory() as directory:
        path = Path(directory) / "calibration.csv"
        path.write_text(contents.lstrip("\ufeff"), encoding="utf-8")
        return PauliErrorLookup(path)


def compile_single(arch, time, config, modules, rounds, chains):
    """Compile the selected rounds and collect timing data for a single LIZ."""
    forward = modules[0]
    liz = arch.get_liz_zones()[0]
    coolant_ids = config.resident_coolant_ids or []
    if len(coolant_ids) + 2 > liz.capacity:
        raise ValueError("The interaction zone needs room for the coolants and both MS gate ions.")
    if chains is None:
        left = [z.id for z in arch.get_memory_zones() if z.id < liz.id]
        right = [z.id for z in arch.get_memory_zones() if z.id > liz.id]
        if not left or not right:
            raise ValueError("Automatic placement needs memory on both sides of the LIZ. Enter starting chains.")
        auto = placement_greedy_alt(forward.DATA_QUBITS, forward.ANCILLA_QUBITS, forward)
        chains = {
            left[-1]: [ion for ion, zone in auto.items() if zone == 1],
            right[0]: [ion for ion, zone in auto.items() if zone == 5],
            liz.id: coolant_ids,
        }
        for zone, ions in chains.items():
            if len(ions) > arch.get_zone(zone).capacity:
                raise ValueError(f"Automatic placement exceeds capacity in z{zone}. Increase it or enter starting chains.")
    if not set(coolant_ids).issubset(chains.get(liz.id, [])):
        raise ValueError("Start resident coolant ions in the interaction zone.")
    compiler = QCCDCompiler(arch, config=config, time_params=time)
    results = []
    for index in range(rounds):
        mode = modules[index % 2]
        dag = DAGBuilder(parse(mode.CIRCUIT), stabilizer_parallel=False).build()
        placed = {ion: zone for zone, ions in sorted(chains.items()) for ion in ions}
        result = compiler.compile(dag, placed)
        results.append(result)
        chains = result.final_zone_contents
    estimator = ParallelEstimator(time)
    duration = sum(estimator.estimate(result.timesteps).total_time for result in results)
    gates = [gate for result in results for gate in result.phonon_data["at_gates"]]
    anchors, gaps = gaps_from_compile_results(results, time_params=time)
    schedules = [format_timesteps_text(result.timesteps, estimator) for result in results]
    return results, gates, anchors, gaps, duration, schedules


def compile_double(arch, time, config, module, distance, rounds, chains):
    """Compile and validate both LIZ streams for each round."""
    program = parse(module.CIRCUIT)
    results = []
    for index in range(rounds):
        result = compile_per_liz(
            program, module.ANCILLA_QUBITS, module.DATA_QUBITS, arch, distance,
            config=config, time_params=time, stab_order="boustrophedon",
            placement_mode="3tier_usage", scan_reverse=bool(index % 2),
            initial_zone_contents=chains,
        )
        estimate_streams(result, params=time)
        validation = validate_merged(result)
        if not validation.is_valid:
            raise ValueError(f"Compiled round {index + 1} failed transport validation. {validation.summary()}")
        results.append(result)
        chains = {zone: ions for zone, ions in result.state.zone_contents.items() if ions}
    gates = [gate for result in results for gate in result.state.phonon_at_gates]
    anchors, gaps = gaps_from_per_liz_results(results, time_params=time)
    streams = [SimpleNamespace(flat_ops=result.merged_flat) for result in results]
    schedules = []
    for result in results:
        lines = []
        for segment in sorted(result.segments, key=lambda item: (item.start_us, item.core_id)):
            lines.append(f"LIZ z{segment.core_id}   {segment.start_us:.2f}–{segment.end_us:.2f} µs")
            lines.extend(f"  {format_op_summary(op)}" for op in segment.ops)
            lines.append("")
        schedules.append("\n".join(lines))
    return streams, gates, anchors, gaps, sum(result.wall_clock_us for result in results), schedules


def check_blocks(circuit, measurements, rounds):
    """Group the sampled detector IDs and actual Stim gates by stabilizer."""
    by_check = {}
    for measured in measurements:
        key = (measured.basis, measured.data_qubits)
        events = by_check.setdefault(key, [None] * rounds)
        gates = [
            [instruction.name, [target.value for target in instruction.targets_copy()]]
            for instruction in circuit[measured.instruction_start:measured.instruction_stop]
            if instruction.name != "TICK"
        ]
        events[measured.round_number - 1] = {
            "detector": measured.detector,
            "ancilla": measured.ancilla,
            "gates": gates,
        }

    blocks = []
    for basis in ("Z", "X"):
        supports = sorted(support for check_basis, support in by_check if check_basis == basis)
        for index, support in enumerate(supports, start=1):
            blocks.append({
                "label": f"{basis}{index}",
                "data": support,
                "rounds": by_check[basis, support],
            })
    return blocks


def simulation_parameters(settings):
    """Read the sampling settings shared by hardware and uniform runs."""
    if not isinstance(settings, dict):
        raise ValueError("Send simulation settings as a JSON object.")
    distance = integer(settings["distance"], "Distance", 3, 13)
    if distance not in DISTANCE_CIRCUITS:
        raise ValueError("Choose a supported surface-code distance.")
    rounds = integer(settings["rounds"], "Rounds", 1, MAX_DETECTORS // (distance**2 - 1))
    shots = integer(settings["shots"], "Shots", 2, 2_000_000)
    seed = integer(settings["seed"], "Seed", 0, 2**53 - 1)
    return distance, rounds, shots, seed


def correlation_result(distance, rounds, shots, seed, clean, noisy, measurements, dem=None):
    """Sample the selected model and package detector and circuit results."""
    sampler_source = noisy if dem is None else dem
    matrix = estimate_pij(sampler_source, shots, seed=seed)
    return {
        "distance": distance,
        "rounds": rounds,
        "shots": shots,
        "detectors": sampler_source.num_detectors,
        "matrix": [[value if math.isfinite(value) else None for value in row] for row in matrix],
        "checks": check_blocks(clean, measurements, rounds),
        "logical_stim": str(clean),
        "stim": str(noisy),
    }


def simulate_uniform(settings):
    """Sample the phenomenological surface-code DEM at one fault probability."""
    distance, rounds, shots, seed = simulation_parameters(settings)
    p = number(settings["p"], "p", 0.5)
    forward, reverse, observable = DISTANCE_CIRCUITS[distance]
    modules = [import_module(f".circuits.{name}", __package__) for name in (forward, reverse)]
    programs = [parse(module.CIRCUIT) for module in modules]
    round_ops = [SimpleNamespace(flat_ops=programs[index % 2].instructions)
                 for index in range(rounds)]
    measurements = []
    clean = recompile_multi_round(round_ops, observable_column=observable,
                                  check_measurements=measurements)
    round_starts = {}
    for measurement in measurements:
        round_starts.setdefault(measurement.round_number, measurement.instruction_start)
    starts = set(round_starts.values())
    noisy = stim.Circuit()
    data = modules[0].DATA_QUBITS
    for index, instruction in enumerate(clean):
        if index in starts and p:
            noisy.append("X_ERROR", data, p)
            noisy.append("Z_ERROR", data, p)
        if instruction.name == "M" and p:
            noisy.append("M", instruction.targets_copy(), p)
        else:
            noisy.append(instruction)
    dem = noisy.detector_error_model()
    result = correlation_result(distance, rounds, shots, seed, clean, noisy, measurements, dem)
    result.update(uniform_p=p, dem=str(dem))
    return result


def simulate(settings):
    """Compile the configured hardware, inject noise, and estimate correlations."""
    distance, rounds, shots, seed = simulation_parameters(settings)
    arch = architecture(settings["zones"])
    two_liz = len(arch.get_liz_zones()) == 2
    timings = settings["timing"]
    if not isinstance(timings, dict):
        raise ValueError("Send the operation and heating parameters as a JSON object.")
    time = load_time_parameters(overrides={
        key: number(value, key) for key, value in timings.items()
        if value is not None or key not in TIMING_DEFAULTS or TIMING_DEFAULTS[key] is not None
    })
    if time.shuttle_speed == 0 or time.shuttle_time(max(z.capacity for z in arch.zones)) <= 0:
        raise ValueError("Shuttle speed and distance must be positive at the specified zone capacities.")
    noise = default_noise_params()
    for key in NOISE_FIELDS:
        noise[key] = number(settings["noise"][key], key, 1 if key.startswith("p_") else math.inf)
    lookup = load_lookup(settings.get("lookup_csv"))
    mode = settings["lookup_mode"]
    if mode not in lookup.SUPPORTED_MODES:
        raise ValueError("Choose a valid MS lookup rule.")
    options = settings["cooling"]
    coolant_ids = options["resident_ids"]
    if (not isinstance(coolant_ids, list)
            or any(type(ion) is not int or ion < 0 for ion in coolant_ids)
            or len(set(coolant_ids)) != len(coolant_ids)):
        raise ValueError("Coolant IDs must be distinct nonnegative integers.")
    if two_liz and coolant_ids:
        raise ValueError("Resident coolant ions are available with the single-LIZ planner.")
    if options["recool_us"] is not None and not coolant_ids:
        raise ValueError("Enter resident coolant IDs to use resident re-cooling.")
    cooling = {key: None if options[name] is None else number(options[name], name)
               for key, name in [
                   ("cool_threshold_transversal", "threshold"),
                   ("cool_ancilla_after_measure_us", "after_measure_us"),
                   ("coolant_recool_us", "recool_us"),
               ]}
    config = CompilerConfig(resident_coolant_ids=coolant_ids, **cooling)
    config.validate()

    forward, reverse, observable = DISTANCE_CIRCUITS[distance]
    if two_liz:
        forward = "snake_reuse2" if distance == 3 else f"d{distance}_reuse2"
    first = import_module(f".circuits.{forward}", __package__)
    code_ions = first.DATA_QUBITS + first.ANCILLA_QUBITS
    if set(coolant_ids) & set(code_ions):
        raise ValueError("Choose coolant IDs outside the data and ancilla IDs shown for this run.")
    chains = initial_chains(settings.get("chains"), arch, code_ions, coolant_ids)
    if two_liz:
        results, gates, anchors, gaps, duration, schedules = compile_double(
            arch, time, config, first, distance, rounds, chains)
    else:
        results, gates, anchors, gaps, duration, schedules = compile_single(
            arch, time, config, (first, import_module(f".circuits.{reverse}", __package__)), rounds, chains)
    measurements = []
    clean = recompile_multi_round(results, observable_column=observable,
                                  check_measurements=measurements)
    clean.detector_error_model()
    rates = phonon_to_pauli_lookup_rates(gates, lookup, mode=mode, warn_on_clip=False)
    noisy = inject_noise_approach_b(clean, rates, noise, include_idle=False)
    noisy = inject_per_ion_idle(noisy, anchors, gaps, noise)
    max_phonon = max(rate[4] for rate in rates)
    max_lookup = lookup.n_bar_range[1]
    result = correlation_result(distance, rounds, shots, seed, clean, noisy, measurements)
    result.update({
        "duration_ms": duration / 1000,
        "max_phonon": max_phonon,
        "lookup_ceiling": max_lookup,
        "clipped_gates": sum(rate[4] > max_lookup for rate in rates),
        "schedules": schedules,
    })
    return result
