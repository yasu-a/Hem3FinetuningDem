"""Regression tests for detector-level DEM calibration."""

import numpy as np
import stim

from fdem.calibrate import fit_dem


def test_fit_preserves_the_logical_split_of_decomposed_faults() -> None:
    """Fit a hyperedge and a shared support while retaining their logical targets."""
    baseline = stim.DetectorErrorModel("""
        error[pair](0.01) D0 D1 ^ D1 D2 L0
        error(0.02) D0 D2
        error(0.03) D0 D1 D2
        error(0.01) L0
        detector(0, 0) D0
    """)
    # Exact syndrome frequencies for independent rates of 1/8 and 1/16.
    patterns = np.array(
        [[0, 0, 0], [1, 0, 1], [1, 1, 1], [0, 1, 0]], dtype=bool
    )
    detector_events = np.repeat(patterns, [105, 15, 7, 1], axis=0)
    fitted = fit_dem(baseline, detector_events, regularization=1e-6)

    first, second, hyperedge, logical_only, detector = fitted
    rates = [instruction.args_copy()[0] for instruction in (first, second, hyperedge)]
    np.testing.assert_allclose(
        [rates[0] + rates[1] - 2 * rates[0] * rates[1], rates[2]],
        [1 / 8, 1 / 16],
        atol=1e-6,
    )
    fitted_split = np.log1p(-2 * rates[0]) / np.log1p(-2 * rates[1])
    np.testing.assert_allclose(fitted_split, np.log1p(-0.02) / np.log1p(-0.04))
    assert first.tag == "pair"
    assert first.targets_copy() == baseline[0].targets_copy()
    assert logical_only == baseline[3]
    assert detector == baseline[4]
