"""Fit the probabilities of a reference DEM to device detector-pair statistics."""

import argparse
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.optimize import lsq_linear
from scipy.sparse import csr_matrix
import stim

from .correlations import detector_pair_correlations
from .dem import detector_support, with_priors
from .device import sample_physical_device_syndromes


def _pair_design(
    dem: stim.DetectorErrorModel,
) -> tuple[np.ndarray, csr_matrix, np.ndarray]:
    """Build the pair-to-mechanism incidence matrix for a flattened DEM.

    Args:
        dem: Flattened Stim detector error model.

    Returns:
        Detector pairs in row order, the sparse incidence matrix, and the baseline
        probabilities in column order. Each column represents one error instruction.
        Only detector pairs present in the DEM occupy rows in the matrix.
    """
    pair_rows: dict[tuple[int, int], int] = {}
    probabilities = []
    rows = []
    columns = []
    for instruction in dem:
        if instruction.type != "error":
            continue
        column = len(probabilities)
        probabilities.append(instruction.args_copy()[0])
        for pair in combinations(detector_support(instruction), 2):
            rows.append(pair_rows.setdefault(pair, len(pair_rows)))
            columns.append(column)

    design = csr_matrix(
        (np.ones(len(rows)), (rows, columns)),
        shape=(len(pair_rows), len(probabilities)),
    )
    pairs = np.array(list(pair_rows), dtype=np.intp).reshape(-1, 2)
    return pairs, design, np.array(probabilities)


def _log_rates(probabilities: np.ndarray) -> np.ndarray:
    """Convert DEM probabilities to nonnegative log rates.

    Args:
        probabilities: Valid Stim error probabilities for pair-visible mechanisms.

    Returns:
        Log rates in the same order as the probabilities.

    Raises:
        ValueError: If a probability is at least 0.5.
    """
    if np.any(probabilities >= 0.5):
        raise ValueError("Pair-visible DEM probabilities must be below 0.5")
    return -np.log1p(-2 * probabilities)


def dem_pair_probabilities(dem: stim.DetectorErrorModel) -> np.ndarray:
    """Calculate the exact pair probabilities of a DEM, including hyperedges.

    Args:
        dem: Stim model whose independent mechanisms contribute to every pair of
            detectors they flip together.

    Returns:
        A symmetric matrix in Stim detector order with NaN on the diagonal. An
        off-diagonal entry is the probability of an odd number of shared mechanisms.

    Raises:
        ValueError: If a mechanism with two or more detectors has probability at least 0.5.
    """
    pairs, design, probabilities = _pair_design(dem.flattened())
    visible = design.getnnz(axis=0) > 0
    pair_rates = design[:, visible] @ _log_rates(probabilities[visible])
    pair_probabilities = np.zeros((dem.num_detectors, dem.num_detectors))
    left, right = pairs.T
    pair_probabilities[left, right] = pair_probabilities[right, left] = (
        -np.expm1(-pair_rates) / 2
    )
    np.fill_diagonal(pair_probabilities, np.nan)
    return pair_probabilities


def pair_probability_mse_and_gradient(
    dem: stim.DetectorErrorModel,
    observed_pij: np.ndarray,
) -> tuple[float, np.ndarray]:
    """Calculate pair-probability MSE and its analytic log-rate gradient.

    The gradient is taken with respect to one parity log rate
    ``-log(1 - 2p)`` per flattened DEM error instruction. Detector pairs absent from
    the DEM contribute a constant residual to the MSE but no gradient.

    Args:
        dem: Detector error model supplying the predicted pair probabilities.
        observed_pij: Square observed pair-probability matrix. The finite entries in
            its upper triangle define the MSE.

    Returns:
        The pair-probability MSE and its gradient in flattened error-instruction order.

    Raises:
        ValueError: If the matrix shape is invalid, no finite pair is available, no
            DEM-supported pair is finite, or a pair-visible probability is at least
            0.5.
    """
    if observed_pij.shape != (dem.num_detectors, dem.num_detectors):
        raise ValueError("Pair matrix shape must match the number of DEM detectors")

    flattened = dem.flattened()
    pairs, design, probabilities = _pair_design(flattened)
    visible = design.getnnz(axis=0) > 0
    pair_rates = design[:, visible] @ _log_rates(probabilities[visible])
    supported_predictions = -np.expm1(-pair_rates) / 2

    rows, columns = np.triu_indices(dem.num_detectors, k=1)
    observed = observed_pij[rows, columns]
    finite = np.isfinite(observed)
    if not finite.any():
        raise ValueError("No detector pair has a finite observed probability")

    predictions = np.zeros_like(observed, dtype=float)
    pair_indices = {
        (int(left), int(right)): index
        for index, (left, right) in enumerate(zip(rows, columns))
    }
    supported_indices = np.array(
        [pair_indices[tuple(pair)] for pair in pairs],
        dtype=np.intp,
    )
    predictions[supported_indices] = supported_predictions
    residual = predictions[finite] - observed[finite]
    mse = float(np.mean(residual**2))

    supported_observed = observed_pij[tuple(pairs.T)]
    usable = np.isfinite(supported_observed)
    if not usable.any():
        raise ValueError("No DEM-supported detector pair has a finite estimate")
    supported_residual = supported_predictions[usable] - supported_observed[usable]
    supported_slope = 0.5 - supported_predictions[usable]
    gradient = np.asarray(
        2
        / finite.sum()
        * design[usable].T
        @ (supported_residual * supported_slope)
    ).reshape(-1)
    return mse, gradient


def fit_dem_from_pij(
    baseline: stim.DetectorErrorModel,
    observed_pij: np.ndarray,
) -> stim.DetectorErrorModel:
    """Fit the baseline DEM by nonnegative least squares in pair log-rate space.

    Args:
        baseline: Noise model to fit, with detector IDs matching the device data.
        observed_pij: Square pair matrix from ``detector_pair_correlations``.
            Only the upper triangle is used. Signed estimates are retained, and
            nonfinite entries and entries at or above 0.5 are omitted.

    Returns:
        A flattened fitted DEM. Mechanisms absent from the usable pair equations
        retain their baseline probabilities.

    Raises:
        ValueError: If detector counts differ, there are no usable DEM-supported
            pairs, or a fitted mechanism has a baseline probability at least 0.5.
        RuntimeError: If the least-squares solver does not converge.
    """
    if observed_pij.shape != (baseline.num_detectors, baseline.num_detectors):
        raise ValueError("Pair matrix shape must match the number of DEM detectors")

    flattened = baseline.flattened()
    pairs, design, probabilities = _pair_design(flattened)
    observed = observed_pij[tuple(pairs.T)]
    usable = np.isfinite(observed) & (observed < 0.5)
    if not np.any(usable):
        raise ValueError("No DEM-supported detector pairs have finite estimates below 0.5")

    design = design[usable]
    fitted = design.getnnz(axis=0) > 0
    design = design[:, fitted]
    prior = _log_rates(probabilities[fitted])
    target = -np.log1p(-2 * observed[usable])

    # Solve for the change from the baseline with the equivalent constraint x >= 0.
    result = lsq_linear(
        design,
        target - design @ prior,
        bounds=(-prior, np.inf),
        tol=1e-12,
    )
    if not result.success:
        raise RuntimeError(f"Pair-probability fit failed: {result.message}")

    fitted_rates = prior + result.x
    probabilities[fitted] = np.clip(-np.expm1(-fitted_rates) / 2, 0, np.nextafter(0.5, 0))
    return with_priors(flattened, probabilities)


def main() -> None:
    """Sample a named device experiment and fit the independently supplied DEM."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", required=True, help="Device experiment under experiments/.")
    parser.add_argument("--baseline-dem", required=True, type=Path, help="Noise model to fit.")
    parser.add_argument("--output-dem", required=True, type=Path, help="Fitted DEM file.")
    parser.add_argument("--output-matrices", type=Path, help="Optional NPZ of the three pair matrices.")
    parser.add_argument("--shots", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.shots <= 0:
        parser.error("Shot count must be positive")

    try:
        baseline = stim.DetectorErrorModel.from_file(args.baseline_dem)
        syndromes = sample_physical_device_syndromes(args.experiment, args.shots, args.seed)
        device_pij = detector_pair_correlations(syndromes)
        fitted = fit_dem_from_pij(baseline, device_pij)
        if args.output_matrices is not None:
            baseline_pij = dem_pair_probabilities(baseline)
            fitted_pij = dem_pair_probabilities(fitted)
            with args.output_matrices.open("wb") as output:
                np.savez_compressed(output, device=device_pij, baseline=baseline_pij, fitted=fitted_pij)
        fitted.to_file(args.output_dem)
    except (OSError, ValueError, RuntimeError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
