"""Tests for analytical detector-pair probabilities and their fit to device samples."""

import numpy as np
import stim

from fdem.correlations import detector_pair_correlations
from fdem.dem import with_priors
from fdem.pij import (
    dem_pair_probabilities,
    fit_dem_from_pij,
    pair_probability_mse_and_gradient,
)


def test_dem_pair_probabilities_combine_shared_mechanisms() -> None:
    """Overlapping edges and hyperedges contribute by odd parity to each pair."""
    dem = stim.DetectorErrorModel("""
        error(0.125) D0 D1
        error(0.125) D0 D2 ^ D1 D2 L0
        error(0.0625) D0 D1 D2
        error(0.2) D2
        detector D3
    """)
    pair = (1 - 0.75**2 * 0.875) / 2
    np.testing.assert_allclose(dem_pair_probabilities(dem), [
        [np.nan, pair, 0.0625, 0],
        [pair, np.nan, 0.0625, 0],
        [0.0625, 0.0625, np.nan, 0],
        [0, 0, 0, np.nan],
    ])


def test_fit_recovers_pair_and_hyperedge_rates_from_syndromes() -> None:
    """Exact shot frequencies recover identifiable rates and retain DEM targets."""
    baseline = stim.DetectorErrorModel("""
        error[pair](0.03) D0 D2 ^ D1 D2 L0
        error(0.02) D0 D1 D2
        error(0.04) D1
        error(0.02) L0
        detector(0, 1) D2
    """)
    # Exact frequencies for independent mechanisms with rates 1/8 and 1/16.
    patterns = [[0, 0, 0], [1, 1, 0], [1, 1, 1], [0, 0, 1]]
    syndromes = np.repeat(patterns, [105, 15, 7, 1], axis=0)
    observed = detector_pair_correlations(syndromes)
    fitted = fit_dem_from_pij(baseline, observed)

    np.testing.assert_allclose(dem_pair_probabilities(fitted), observed)
    np.testing.assert_allclose([fitted[0].args_copy()[0], fitted[1].args_copy()[0]], [1/8, 1/16])
    assert fitted[0].tag == baseline[0].tag
    assert fitted[0].targets_copy() == baseline[0].targets_copy()
    assert fitted[2:] == baseline[2:]


def test_fit_uses_signed_estimates_and_keeps_unobserved_rates() -> None:
    """Negative estimates meet a nonnegative bound and undefined pairs are skipped."""
    baseline = stim.DetectorErrorModel("error(0.1) D0 D1\nerror(0.05) D1 D2")
    observed = np.array([
        [np.nan, -0.01, 0.2],
        [-0.01, np.nan, np.nan],
        [0.2, np.nan, np.nan],
    ])
    fitted = fit_dem_from_pij(baseline, observed)

    np.testing.assert_allclose(fitted[0].args_copy(), [0], atol=1e-10)
    assert fitted[1] == baseline[1]


def test_pair_probability_gradient_matches_central_difference() -> None:
    """The analytic log-rate gradient agrees with numerical differentiation."""
    dem = stim.DetectorErrorModel(
        """
        error(0.08) D0 D1
        error(0.04) D0 D1 D2
        error(0.2) D2
        """
    )
    observed = np.array(
        [
            [np.nan, 0.15, 0.07],
            [0.15, np.nan, 0.09],
            [0.07, 0.09, np.nan],
        ]
    )
    _, analytic = pair_probability_mse_and_gradient(dem, observed)
    probabilities = np.array(
        [
            instruction.args_copy()[0]
            for instruction in dem
            if instruction.type == "error"
        ]
    )
    log_rates = -np.log1p(-2 * probabilities)
    numerical = np.empty_like(log_rates)
    epsilon = 1e-6

    for index in range(log_rates.size):
        upper = log_rates.copy()
        lower = log_rates.copy()
        upper[index] += epsilon
        lower[index] -= epsilon
        upper_dem = with_priors(dem, -np.expm1(-upper) / 2)
        lower_dem = with_priors(dem, -np.expm1(-lower) / 2)
        upper_loss, _ = pair_probability_mse_and_gradient(upper_dem, observed)
        lower_loss, _ = pair_probability_mse_and_gradient(lower_dem, observed)
        numerical[index] = (upper_loss - lower_loss) / (2 * epsilon)

    np.testing.assert_allclose(analytic, numerical, rtol=1e-6, atol=1e-9)
    assert analytic[-1] == 0
