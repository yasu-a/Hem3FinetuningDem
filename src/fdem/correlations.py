"""The signed pair estimator from the THeatA workbench, using McEwen et al. Eq. (1)."""

# https://doi.org/10.1038/s41467-021-21982-y

import numpy as np
import stim


def detector_pair_correlations(syndromes: np.ndarray) -> np.ndarray:
    """Estimate signed pair-error probabilities from detection-event samples.

    This implements Eq. (1) of McEwen et al. for binary detection events. The
    diagonal contains each detector's empirical firing probability. Finite-sample
    off-diagonal combinations with a negative square-root argument are undefined.

    Args:
        syndromes: Binary detection-event samples with shape ``(shots, detectors)``.

    Returns:
        A symmetric ``(detectors, detectors)`` matrix of signed pair-error
        probabilities with detector firing probabilities on the diagonal. Undefined
        off-diagonal entries are ``NaN``.

    Raises:
        ValueError: If the samples are empty, are not two-dimensional, contain
            fewer than two detectors, or contain values other than zero and one.
    """
    if syndromes.ndim != 2 or syndromes.shape[0] == 0:
        raise ValueError("Syndromes must be a non-empty two-dimensional array")
    if syndromes.shape[1] < 2:
        raise ValueError("Syndromes must contain at least two detectors")
    if np.any((syndromes != 0) & (syndromes != 1)):
        raise ValueError("Syndromes must contain only binary values")

    samples = syndromes.astype(np.float64, copy=False)
    joint_hits = samples.T @ samples
    return _pair_correlations_from_counts(joint_hits, syndromes.shape[0])


def _pair_correlations_from_counts(
    joint_hits: np.ndarray,
    shots: int,
) -> np.ndarray:
    """Calculate signed pair-error probabilities from joint hit counts.

    Args:
        joint_hits: Matrix whose entry ``(i, j)`` counts coincident detector hits.
        shots: Number of samples represented by the counts.

    Returns:
        A symmetric pair-correlation matrix with detector firing probabilities on its
        diagonal.
    """
    hits = joint_hits.diagonal()
    xor_hits = hits[:, None] + hits - 2 * joint_hits
    pair_bias = 1 - 2 * xor_hits / shots
    detector_bias = 1 - 2 * hits / shots
    ratio = np.full_like(joint_hits, np.nan, dtype=float)
    np.divide(
        np.outer(detector_bias, detector_bias),
        pair_bias,
        out=ratio,
        where=pair_bias != 0,
    )
    ratio[ratio < 0] = np.nan
    correlations = (1 - np.sqrt(ratio)) / 2
    np.fill_diagonal(correlations, hits / shots)
    return correlations


def estimate_pij(circuit: stim.Circuit | stim.DetectorErrorModel, shots: int,
                 seed: int | None = None) -> np.ndarray:
    """Return a symmetric matrix indexed by Stim detector ID.

    Sample detection events from a circuit or detector error model and keep
    signed estimates. The diagonal contains detector firing probabilities, and
    undefined off-diagonal pairs are represented by NaN.

    Args:
        circuit: Stim circuit or detector error model to sample.
        shots: Number of detection-event samples.
        seed: Optional Stim sampling seed.

    Returns:
        A symmetric signed pair-correlation matrix in Stim detector order.

    Raises:
        ValueError: If fewer than two shots or detectors are requested.
    """
    if shots < 2:
        raise ValueError("at least two shots are required")
    if circuit.num_detectors < 2:
        raise ValueError("the circuit or error model must contain at least two detectors")

    from_dem = isinstance(circuit, stim.DetectorErrorModel)
    sampler = (circuit.compile_sampler(seed=seed) if from_dem
               else circuit.compile_detector_sampler(seed=seed))
    joint_hits = np.zeros((circuit.num_detectors, circuit.num_detectors))
    # Cast before multiplying so the product counts coincident hits.
    batch_size = 8192
    for start in range(0, shots, batch_size):
        batch = sampler.sample(min(batch_size, shots - start))
        samples = (batch[0] if from_dem else batch).astype(np.float64)
        joint_hits += samples.T @ samples

    return _pair_correlations_from_counts(joint_hits, shots)
