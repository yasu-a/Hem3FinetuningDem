"""Fit a reference DEM to detector events sampled from a separate noisy circuit."""

import argparse
from itertools import combinations
from pathlib import Path

import numpy as np
import pymatching
import scipy.optimize
import scipy.sparse
import stim


def _detector_support(instruction: stim.DemInstruction) -> tuple[int, ...]:
    """Return the detectors flipped by the whole mechanism, including all components."""
    detectors = set()
    for target in instruction.targets_copy():
        if target.is_relative_detector_id():
            detectors.symmetric_difference_update((target.val,))
    return tuple(sorted(detectors))


def _log_rates(dem: stim.DetectorErrorModel) -> dict[tuple[int, ...], float]:
    """Combine independent mechanisms with identical detector support in log space."""
    rates: dict[tuple[int, ...], float] = {}
    for instruction in dem.flattened():
        if instruction.type != "error":
            continue
        support = _detector_support(instruction)
        if not support:
            continue
        probability = instruction.args_copy()[0]
        if not 0 <= probability < 0.5:
            raise ValueError("Detector-visible baseline probabilities must be below 0.5.")
        rates[support] = rates.get(support, 0.0) - np.log1p(-2 * probability)
    return rates


def _parity_matrix(
    detector_count: int,
    parity_sets: list[tuple[int, ...]],
    supports: list[tuple[int, ...]],
) -> scipy.sparse.csr_matrix:
    """Map each mechanism to the parities it flips by odd detector overlap."""
    def incidence(sets: list[tuple[int, ...]]) -> scipy.sparse.csr_matrix:
        """Build an incidence matrix with one row per detector set."""
        rows = [row for row, subset in enumerate(sets) for _ in subset]
        cols = [detector for subset in sets for detector in subset]
        return scipy.sparse.csr_matrix(
            (np.ones(len(rows), dtype=np.int64), (rows, cols)),
            shape=(len(sets), detector_count),
        )

    matrix = incidence(parity_sets) @ incidence(supports).T
    matrix.data %= 2
    matrix.eliminate_zeros()
    return matrix.astype(float)


def _parity_rates(
    samples: np.ndarray, parity_sets: list[tuple[int, ...]]
) -> np.ndarray:
    """Estimate the probability of odd parity for each detector set."""
    return np.array([
        np.logical_xor.reduce(samples[:, subset], axis=1).mean()
        for subset in parity_sets
    ])


def fit_dem(
    baseline: stim.DetectorErrorModel,
    detector_events: np.ndarray,
    *,
    regularization: float = 0.1,
) -> stim.DetectorErrorModel:
    """Fit nonnegative mechanism rates using detector events and a baseline prior."""
    if (
        detector_events.ndim != 2
        or detector_events.shape[1] != baseline.num_detectors
        or detector_events.shape[0] == 0
        or detector_events.dtype != np.bool_
    ):
        raise ValueError("Expected a nonempty boolean array of detector shots in DEM order.")
    if not np.isfinite(regularization) or regularization <= 0:
        raise ValueError("Regularization must be finite and positive.")

    prior_by_support = _log_rates(baseline)
    if not prior_by_support:
        return baseline.copy()

    supports = list(prior_by_support)
    parity_sets = sorted({
        subset
        for support in supports
        for order in range(1, len(support) + 1)
        for subset in combinations(support, order)
    })
    odd_rates = _parity_rates(detector_events, parity_sets)
    if np.any(odd_rates >= 0.5):
        raise ValueError("Fit requires positive empirical parity expectations.")

    # Independent error contributions multiply in parity space and add in log space.
    design = _parity_matrix(baseline.num_detectors, parity_sets, supports)
    penalty = np.sqrt(regularization)
    prior = np.array(list(prior_by_support.values()))
    design = scipy.sparse.vstack((design, penalty * scipy.sparse.eye(len(supports))))
    observations = np.concatenate((-np.log1p(-2 * odd_rates), penalty * prior))
    result = scipy.optimize.lsq_linear(
        design, observations, bounds=(0, np.inf), lsmr_tol="auto", max_iter=300
    )
    if not result.success:
        raise RuntimeError(f"DEM fit failed. {result.message}")

    fitted_by_support = dict(zip(supports, result.x, strict=True))
    duplicates: dict[tuple[int, ...], int] = {}
    flattened = baseline.flattened()
    for instruction in flattened:
        if instruction.type == "error":
            support = _detector_support(instruction)
            duplicates[support] = duplicates.get(support, 0) + 1

    fitted = stim.DetectorErrorModel()
    for instruction in flattened:
        if instruction.type != "error":
            fitted.append(instruction)
            continue
        support = _detector_support(instruction)
        if not support:
            fitted.append(instruction)
            continue
        # Preserve the prior split when detector data cannot distinguish mechanisms.
        prior_rate = prior_by_support[support]
        if prior_rate:
            share = -np.log1p(-2 * instruction.args_copy()[0]) / prior_rate
        else:
            share = 1 / duplicates[support]
        rate = fitted_by_support[support] * share
        probability = min(-np.expm1(-rate) / 2, np.nextafter(0.5, 0))
        fitted.append("error", probability, instruction.targets_copy(), tag=instruction.tag)
    return fitted


def _parity_predictions(
    dem: stim.DetectorErrorModel, parity_sets: list[tuple[int, ...]]
) -> np.ndarray:
    """Calculate the exact odd-parity probabilities of a DEM."""
    rates = _log_rates(dem)
    design = _parity_matrix(dem.num_detectors, parity_sets, list(rates))
    return -np.expm1(-(design @ np.array(list(rates.values())))) / 2


def _demo_circuit(
    gate_error: float, reset_error: float, measurement_error: float
) -> stim.Circuit:
    """Generate the small rotated memory circuit used in the benchmark."""
    return stim.Circuit.generated(
        "surface_code:rotated_memory_x",
        distance=3,
        rounds=3,
        after_clifford_depolarization=gate_error,
        after_reset_flip_probability=reset_error,
        before_measure_flip_probability=measurement_error,
    )


def main() -> None:
    """Fit and evaluate a DEM using disjoint shot sets from a target circuit."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-circuit", type=Path, help="Noisy target .stim circuit.")
    parser.add_argument("--baseline-circuit", type=Path, help="Reference .stim circuit.")
    parser.add_argument("--train-shots", type=int, default=50_000)
    parser.add_argument("--test-shots", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--regularization", type=float, default=0.1, help="Strength of the baseline prior."
    )
    parser.add_argument("--output-dem", type=Path, help="Optional fitted .dem output.")
    args = parser.parse_args()
    if (args.target_circuit is None) != (args.baseline_circuit is None):
        parser.error("Provide both target and baseline circuit files.")
    if min(args.train_shots, args.test_shots) <= 0:
        parser.error("Shot counts must be positive.")
    if args.target_circuit is None:
        baseline_circuit = _demo_circuit(0.001, 0.001, 0.001)
        target = _demo_circuit(0.002, 0.001, 0.02)
    else:
        baseline_circuit = stim.Circuit.from_file(args.baseline_circuit)
        target = stim.Circuit.from_file(args.target_circuit)
    if (
        target.get_detector_coordinates() != baseline_circuit.get_detector_coordinates()
        or target.num_observables != baseline_circuit.num_observables
    ):
        parser.error("Circuits must use the same detector coordinates and observable count.")
    if target.num_detectors == 0:
        parser.error("Circuits must define at least one detector.")

    baseline = baseline_circuit.detector_error_model(decompose_errors=True)
    sampler = target.compile_detector_sampler(seed=args.seed)
    training_detectors = sampler.sample(args.train_shots)
    fitted = fit_dem(baseline, training_detectors, regularization=args.regularization)
    test_detectors, test_observables = sampler.sample(
        args.test_shots, separate_observables=True
    )

    detector_ids = range(target.num_detectors)
    parity_sets = [(detector,) for detector in detector_ids]
    parity_sets.extend(combinations(detector_ids, 2))
    observed = _parity_rates(test_detectors, parity_sets)
    models = {"Baseline": baseline, "Fitted": fitted}
    if args.target_circuit is None:
        models["Target DEM"] = target.detector_error_model(decompose_errors=True)

    for name, dem in models.items():
        residual = _parity_predictions(dem, parity_sets) - observed
        single_rmse = np.sqrt(np.mean(residual[:target.num_detectors] ** 2))
        print(f"{name} single-detector RMSE = {single_rmse:.6g}")
        if len(residual) > target.num_detectors:
            pair_rmse = np.sqrt(np.mean(residual[target.num_detectors:] ** 2))
            print(f"{name} pair-parity RMSE = {pair_rmse:.6g}")
        if target.num_observables:
            decoder = pymatching.Matching.from_detector_error_model(dem)
            predictions = decoder.decode_batch(test_detectors)
            errors = np.count_nonzero(np.any(predictions != test_observables, axis=1))
            print(f"{name} logical failures = {errors} / {args.test_shots} "
                  f"(LER {errors / args.test_shots:.6g})")

    if args.output_dem is not None:
        fitted.to_file(args.output_dem)


if __name__ == "__main__":
    main()
