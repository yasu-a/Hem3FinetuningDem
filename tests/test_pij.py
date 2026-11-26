"""Tests for analytical detector-pair probabilities and their fit to device samples."""

import numpy as np
import stim

from fdem.correlations import detector_pair_correlations
from fdem.pij import dem_pair_probabilities, fit_dem_from_pij


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
