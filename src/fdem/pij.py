"""Fit reference DEM probabilities to detector firing and pair statistics."""

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
from .statistics import correlation_matrix_metrics


def _pair_design(
    dem: stim.DetectorErrorModel,
) -> tuple[np.ndarray, csr_matrix, np.ndarray]:
    """Build the correlation-entry incidence matrix for a flattened DEM.

    Args:
        dem: Flattened Stim detector error model.

    Returns:
        Matrix entries ``(i, j)`` in row order, the sparse incidence matrix, and the
        baseline probabilities in column order. Diagonal entries represent individual
        detector firing, and off-diagonal entries represent detector pairs. Each
        column represents one error instruction.
    """
    entry_rows: dict[tuple[int, int], int] = {}
    probabilities = []
    rows = []
    columns = []
    for instruction in dem:
        if instruction.type != "error":
            continue
        column = len(probabilities)
        probabilities.append(instruction.args_copy()[0])
        detectors = detector_support(instruction)
        entries = [(detector, detector) for detector in detectors]
        entries.extend(combinations(detectors, 2))
        for entry in entries:
            rows.append(entry_rows.setdefault(entry, len(entry_rows)))
            columns.append(column)

    design = csr_matrix(
        (np.ones(len(rows)), (rows, columns)),
        shape=(len(entry_rows), len(probabilities)),
    )
    matrix_entries = np.array(list(entry_rows), dtype=np.intp).reshape(-1, 2)
    return matrix_entries, design, np.array(probabilities)


def _log_rates(probabilities: np.ndarray) -> np.ndarray:
    """Convert DEM probabilities to nonnegative log rates.

    Args:
        probabilities: Valid Stim error probabilities for detector-visible mechanisms.

    Returns:
        Log rates in the same order as the probabilities.

    Raises:
        ValueError: If a probability is at least 0.5.
    """
    if np.any(probabilities >= 0.5):
        raise ValueError("Detector-visible DEM probabilities must be below 0.5")
    return -np.log1p(-2 * probabilities)


def dem_pair_probabilities(dem: stim.DetectorErrorModel) -> np.ndarray:
    """Calculate exact detector firing and pair probabilities of a DEM.

    Args:
        dem: Stim model whose independent mechanisms contribute to every pair of
            detectors they flip together.

    Returns:
        A symmetric matrix in Stim detector order. A diagonal entry is the probability
        of an odd number of mechanisms flipping that detector. An off-diagonal entry
        is the probability of an odd number of shared mechanisms.

    Raises:
        ValueError: If a detector-visible mechanism has probability at least 0.5.
    """
    entries, design, probabilities = _pair_design(dem.flattened())
    visible = design.getnnz(axis=0) > 0
    entry_rates = design[:, visible] @ _log_rates(probabilities[visible])
    probabilities_matrix = np.zeros((dem.num_detectors, dem.num_detectors))
    left, right = entries.T
    entry_probabilities = -np.expm1(-entry_rates) / 2
    probabilities_matrix[left, right] = entry_probabilities
    probabilities_matrix[right, left] = entry_probabilities
    return probabilities_matrix


def _validate_pair_loss_weight(pair_loss_weight: float) -> float:
    """Validate and normalize the pair-loss weight.

    Args:
        pair_loss_weight: Candidate multiplier for off-diagonal pair MSE.

    Returns:
        The validated weight as a float.

    Raises:
        ValueError: If the weight is negative or not finite.
    """
    if not np.isfinite(pair_loss_weight) or pair_loss_weight < 0:
        raise ValueError("Pair loss weight must be finite and nonnegative")
    return float(pair_loss_weight)


def _entry_predictions(
    design: csr_matrix,
    probabilities: np.ndarray,
) -> np.ndarray:
    """Calculate probabilities for the entries represented by a design matrix.

    Args:
        design: Correlation-entry to error-mechanism incidence matrix.
        probabilities: DEM error probabilities in design-column order.

    Returns:
        Predicted odd-parity probabilities in design-row order.
    """
    visible = design.getnnz(axis=0) > 0
    entry_rates = design[:, visible] @ _log_rates(probabilities[visible])
    return -np.expm1(-entry_rates) / 2


def _matrix_from_entries(
    detector_count: int,
    entries: np.ndarray,
    entry_probabilities: np.ndarray,
) -> np.ndarray:
    """Expand correlation-entry probabilities into a symmetric matrix.

    Args:
        detector_count: Size of the square output matrix.
        entries: Matrix indices in probability order.
        entry_probabilities: Probabilities corresponding to ``entries``.

    Returns:
        A symmetric detector-correlation matrix with zero for absent entries.
    """
    matrix = np.zeros((detector_count, detector_count))
    left, right = entries.T
    matrix[left, right] = entry_probabilities
    matrix[right, left] = entry_probabilities
    return matrix


def pair_probability_mse_and_gradient(
    dem: stim.DetectorErrorModel,
    observed_pij: np.ndarray,
    pair_loss_weight: float = 1.0,
) -> tuple[float, np.ndarray]:
    """Calculate unified correlation loss and its analytic log-rate gradient.

    The gradient is taken with respect to one parity log rate
    ``-log(1 - 2p)`` per flattened DEM error instruction. Detector pairs absent from
    the DEM contribute a constant residual to the loss but no gradient. Diagonal and
    off-diagonal MSE values are independently normalized before weighting.

    Args:
        dem: Detector error model supplying the predicted pair probabilities.
        observed_pij: Square observed pair-probability matrix. The finite entries in
            its diagonal and upper triangle define the loss.
        pair_loss_weight: Nonnegative multiplier for off-diagonal pair MSE.

    Returns:
        ``single_mse + pair_loss_weight * pair_mse`` and its gradient in flattened
        error-instruction order.

    Raises:
        ValueError: If the matrix shape or weight is invalid, either component has no
            finite entry, or a detector-visible probability is at least 0.5.
    """
    if observed_pij.shape != (dem.num_detectors, dem.num_detectors):
        raise ValueError("Pair matrix shape must match the number of DEM detectors")

    weight = _validate_pair_loss_weight(pair_loss_weight)
    flattened = dem.flattened()
    entries, design, probabilities = _pair_design(flattened)
    entry_predictions = _entry_predictions(design, probabilities)
    model_pij = _matrix_from_entries(
        dem.num_detectors,
        entries,
        entry_predictions,
    )
    metrics = correlation_matrix_metrics(observed_pij, model_pij, weight)
    loss = metrics["rmse"] ** 2

    diagonal_observed = np.diag(observed_pij)
    diagonal_finite_count = np.isfinite(diagonal_observed).sum()
    pair_rows, pair_columns = np.triu_indices(dem.num_detectors, k=1)
    pair_finite_count = np.isfinite(observed_pij[pair_rows, pair_columns]).sum()

    entry_observed = observed_pij[tuple(entries.T)]
    finite = np.isfinite(entry_observed)
    diagonal = entries[:, 0] == entries[:, 1]
    residual_slope = (
        (entry_predictions - entry_observed) * (0.5 - entry_predictions)
    )
    single_rows = finite & diagonal
    pair_rows_mask = finite & ~diagonal
    gradient = np.zeros(probabilities.shape)
    if single_rows.any():
        gradient += np.asarray(
            2
            / diagonal_finite_count
            * design[single_rows].T
            @ residual_slope[single_rows]
        ).reshape(-1)
    if pair_rows_mask.any() and weight > 0:
        gradient += np.asarray(
            2
            * weight
            / pair_finite_count
            * design[pair_rows_mask].T
            @ residual_slope[pair_rows_mask]
        ).reshape(-1)
    return loss, gradient


def fit_dem_from_pij(
    baseline: stim.DetectorErrorModel,
    observed_pij: np.ndarray,
    pair_loss_weight: float = 1.0,
) -> stim.DetectorErrorModel:
    """Fit a DEM by weighted nonnegative least squares in correlation log-rate space.

    Args:
        baseline: Noise model to fit, with detector IDs matching the device data.
        observed_pij: Square pair matrix from ``detector_pair_correlations``.
            The diagonal and upper triangle are used. Signed estimates are retained,
            and nonfinite entries and entries at or above 0.5 are omitted.
        pair_loss_weight: Nonnegative multiplier for off-diagonal pair equations.

    Returns:
        A flattened fitted DEM. Mechanisms absent from the usable pair equations
        retain their baseline probabilities.

    Raises:
        ValueError: If detector counts or the weight are invalid, required equations
            are unavailable, or a fitted mechanism has probability at least 0.5.
        RuntimeError: If the least-squares solver does not converge.
    """
    if observed_pij.shape != (baseline.num_detectors, baseline.num_detectors):
        raise ValueError("Pair matrix shape must match the number of DEM detectors")

    weight = _validate_pair_loss_weight(pair_loss_weight)
    flattened = baseline.flattened()
    entries, design, probabilities = _pair_design(flattened)
    observed = observed_pij[tuple(entries.T)]
    usable = np.isfinite(observed) & (observed < 0.5)
    diagonal = entries[:, 0] == entries[:, 1]
    single_count = np.count_nonzero(usable & diagonal)
    pair_count = np.count_nonzero(usable & ~diagonal)
    if single_count == 0:
        raise ValueError("No DEM-supported detector has a finite firing estimate")
    if weight > 0 and pair_count == 0:
        raise ValueError("No DEM-supported detector pair has a finite estimate")

    selected = usable & (diagonal | (weight > 0))
    pair_row_weight = 0.0 if pair_count == 0 else np.sqrt(weight / pair_count)
    row_weights = np.where(
        diagonal[selected],
        np.sqrt(1 / single_count),
        pair_row_weight,
    )
    selected_design = design[selected]
    weighted_design = selected_design.multiply(row_weights[:, None]).tocsr()
    fitted = weighted_design.getnnz(axis=0) > 0
    weighted_design = weighted_design[:, fitted]
    prior = _log_rates(probabilities[fitted])
    target = -np.log1p(-2 * observed[selected])
    weighted_target = row_weights * (target - selected_design[:, fitted] @ prior)

    # Solve for the change from the baseline with the equivalent constraint x >= 0.
    result = lsq_linear(
        weighted_design,
        weighted_target,
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
