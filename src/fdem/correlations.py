"""The signed pair estimator from the THeatA workbench, using McEwen et al. Eq. (1)."""

# https://doi.org/10.1038/s41467-021-21982-y

import numpy as np
import stim


def estimate_pij(circuit: stim.Circuit | stim.DetectorErrorModel, shots: int,
                 seed: int | None = None) -> np.ndarray:
    """Return a symmetric matrix indexed by Stim detector ID.

    Sample detection events from a circuit or detector error model and keep
    signed estimates. The diagonal and undefined pairs are represented by NaN.
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

    # Write the square-root argument in Eq. (1) as the product of the
    # individual parity biases divided by the two-detector parity bias.
    hits = joint_hits.diagonal()
    xor_hits = hits[:, None] + hits - 2 * joint_hits
    pair_bias = 1 - 2 * xor_hits / shots
    detector_bias = 1 - 2 * hits / shots
    ratio = np.full_like(joint_hits, np.nan)
    np.divide(np.outer(detector_bias, detector_bias), pair_bias,
              out=ratio, where=pair_bias != 0)
    ratio[ratio < 0] = np.nan
    pij = (1 - np.sqrt(ratio)) / 2
    np.fill_diagonal(pij, np.nan)
    return pij
