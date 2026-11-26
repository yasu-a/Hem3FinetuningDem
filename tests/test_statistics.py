"""Tests for syndrome summary statistics."""

import numpy as np
import pytest

from fdem.correlations import detector_pair_correlations
from fdem.statistics import DetectorPairCorrelationStatistics


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
