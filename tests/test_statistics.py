"""Tests for syndrome summary statistics."""

import numpy as np
import pytest

from fdem.correlations import detector_pair_correlations
from fdem.statistics import (
    DetectorCombinedStatistics,
    DetectorFiringRateStatistics,
    DetectorPairCorrelationStatistics,
)


def test_pair_correlations_estimate_shared_detection_process() -> None:
    """Perfectly shared detector hits recover their pair-process probability."""
    syndromes = np.array(
        [
            [0, 0],
            [0, 0],
            [0, 0],
            [1, 1],
        ],
        dtype=bool,
    )

    correlations = detector_pair_correlations(syndromes)

    assert np.isnan(correlations[0, 0])
    assert np.isnan(correlations[1, 1])
    assert correlations[0, 1] == pytest.approx(0.25)
    assert correlations[1, 0] == pytest.approx(0.25)


def test_pair_correlation_difference_is_upper_triangle_rmse() -> None:
    """Correlation difference uses each finite detector pair exactly once."""
    target = np.array(
        [
            [np.nan, 0.1, 0.2],
            [0.1, np.nan, np.nan],
            [0.2, np.nan, np.nan],
        ]
    )
    model = np.array(
        [
            [np.nan, 0.2, 0.4],
            [0.2, np.nan, 0.5],
            [0.4, 0.5, np.nan],
        ]
    )

    difference = DetectorPairCorrelationStatistics().difference(target, model)

    assert difference == pytest.approx(np.sqrt((0.1**2 + 0.2**2) / 2))


@pytest.mark.parametrize(
    "syndromes",
    [
        np.array([]),
        np.empty((0, 2), dtype=bool),
        np.zeros((2, 1), dtype=bool),
        np.array([[0, 2], [1, 0]]),
    ],
)
def test_pair_correlations_reject_invalid_syndromes(syndromes: np.ndarray) -> None:
    """Correlation calculation rejects invalid or insufficient samples."""
    with pytest.raises(ValueError):
        detector_pair_correlations(syndromes)


def test_pair_correlation_difference_requires_a_valid_pair() -> None:
    """Correlation RMSE rejects matrices without a jointly finite detector pair."""
    statistics = DetectorPairCorrelationStatistics()
    undefined = np.full((2, 2), np.nan)

    with pytest.raises(ValueError):
        statistics.difference(undefined, undefined)


def test_combined_statistics_reuses_firing_rates_and_pair_correlations() -> None:
    """Combined calculation matches both existing component implementations."""
    syndromes = np.array(
        [
            [0, 0, 0],
            [1, 1, 0],
            [0, 0, 1],
            [1, 1, 0],
        ],
        dtype=bool,
    )

    firing_rates, correlations = DetectorCombinedStatistics().calculate(syndromes)

    np.testing.assert_allclose(
        firing_rates,
        DetectorFiringRateStatistics().calculate(syndromes),
    )
    np.testing.assert_allclose(
        correlations,
        DetectorPairCorrelationStatistics().calculate(syndromes),
        equal_nan=True,
    )


def test_combined_loss_matches_independently_normalized_definition() -> None:
    """Combined loss equals the weighted sum of separately normalized MSE values."""
    target_firing = np.array([0.1, 0.2, 0.3])
    model_firing = np.array([0.2, 0.4, 0.3])
    target_correlations = np.array(
        [
            [np.nan, 0.1, 0.2],
            [0.1, np.nan, np.nan],
            [0.2, np.nan, np.nan],
        ]
    )
    model_correlations = np.array(
        [
            [np.nan, 0.2, 0.4],
            [0.2, np.nan, 0.9],
            [0.4, 0.9, np.nan],
        ]
    )
    statistics = DetectorCombinedStatistics(correlation_weight=2.0)

    metrics = statistics.metrics(
        (target_firing, target_correlations),
        (model_firing, model_correlations),
    )

    firing_rate_mse = (0.1**2 + 0.2**2) / 3
    pair_correlation_mse = (0.1**2 + 0.2**2) / 2
    assert metrics["firing_rate_rmse"] == pytest.approx(np.sqrt(firing_rate_mse))
    assert metrics["pair_correlation_rmse"] == pytest.approx(
        np.sqrt(pair_correlation_mse)
    )
    assert metrics["loss"] == pytest.approx(
        np.sqrt(firing_rate_mse + 2.0 * pair_correlation_mse)
    )


def test_combined_loss_is_zero_for_identical_statistics() -> None:
    """Comparing identical firing rates and correlations gives zero loss."""
    syndromes = np.array([[0, 0], [0, 0], [1, 1], [0, 0]], dtype=bool)
    statistics = DetectorCombinedStatistics()
    values = statistics.calculate(syndromes)

    assert statistics.difference(values, values) == pytest.approx(0.0)


def test_zero_correlation_weight_matches_firing_rate_rmse() -> None:
    """A zero pair weight reduces combined loss to firing-rate RMSE."""
    target_firing = np.array([0.1, 0.2])
    model_firing = np.array([0.3, 0.1])
    target_correlations = np.array([[np.nan, 0.1], [0.1, np.nan]])
    model_correlations = np.array([[np.nan, 0.4], [0.4, np.nan]])
    statistics = DetectorCombinedStatistics(correlation_weight=0.0)

    loss = statistics.difference(
        (target_firing, target_correlations),
        (model_firing, model_correlations),
    )

    firing_rate_rmse = np.sqrt(((0.3 - 0.1) ** 2 + (0.1 - 0.2) ** 2) / 2)
    assert loss == pytest.approx(firing_rate_rmse)


@pytest.mark.parametrize("weight", [-1.0, np.nan, np.inf])
def test_combined_statistics_rejects_invalid_weights(weight: float) -> None:
    """Correlation weight must be finite and nonnegative."""
    with pytest.raises(ValueError):
        DetectorCombinedStatistics(weight)
