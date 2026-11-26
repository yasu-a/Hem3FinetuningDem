"""Tests for DEM optimization metric reporting."""

import numpy as np
import pytest
import stim

from fdem.__main__ import build_candidate_dem, train_dem
from fdem.statistics import DetectorCombinedStatistics


def test_candidate_dem_contains_all_single_and_pair_mechanisms() -> None:
    """The generic candidate covers every detector and unordered detector pair."""
    dem = build_candidate_dem(
        detector_count=4,
        observable_count=1,
        initial_prior=0.01,
    )
    errors = [instruction for instruction in dem if instruction.type == "error"]
    detector_target_counts = [
        sum(target.is_relative_detector_id() for target in instruction.targets_copy())
        for instruction in errors
    ]

    assert len(errors) == 10
    assert detector_target_counts.count(1) == 4
    assert detector_target_counts.count(2) == 6
    assert all(instruction.args_copy() == [0.01] for instruction in errors)
    assert dem.num_detectors == 4
    assert dem.num_observables == 1


def test_candidate_dem_logical_targets_follow_fixed_detector_cut() -> None:
    """Logical targets mark singleton endpoints and pairs crossing the fixed cut."""
    dem = build_candidate_dem(
        detector_count=4,
        observable_count=1,
        initial_prior=0.01,
    )
    errors = [instruction for instruction in dem if instruction.type == "error"]
    errors_with_logical_target = [
        instruction
        for instruction in errors
        if any(
            target.is_logical_observable_id()
            for target in instruction.targets_copy()
        )
    ]

    assert len(errors_with_logical_target) == 6


@pytest.mark.parametrize(
    ("detector_count", "observable_count", "initial_prior"),
    [
        (1, 1, 0.01),
        (2, 0, 0.01),
        (2, 1, 0.0),
        (2, 1, 0.5),
        (2, 1, np.nan),
    ],
)
def test_candidate_dem_rejects_invalid_settings(
    detector_count: int,
    observable_count: int,
    initial_prior: float,
) -> None:
    """Invalid dimensions and probabilities cannot define a candidate DEM."""
    with pytest.raises(ValueError):
        build_candidate_dem(detector_count, observable_count, initial_prior)


def test_training_callback_reports_combined_loss_components() -> None:
    """Each objective evaluation reports both RMSE components and combined loss."""
    initial_dem = stim.DetectorErrorModel(
        """
        error(0.1) D0 D1
        detector D0
        detector D1
        """
    )
    physical_syndromes, _, _ = initial_dem.compile_sampler(seed=1).sample(1_000)
    reported_metrics: list[dict[str, float]] = []

    def record_metrics(
        evaluation: int,
        metrics: dict[str, float],
        candidate_dem: stim.DetectorErrorModel,
    ) -> None:
        """Record metrics supplied by one objective evaluation.

        Args:
            evaluation: Number of completed objective evaluations.
            metrics: Named loss metrics for the candidate.
            candidate_dem: DEM used for the objective evaluation.

        Returns:
            None.
        """
        assert evaluation == len(reported_metrics) + 1
        assert candidate_dem.num_detectors == initial_dem.num_detectors
        reported_metrics.append(metrics)

    _, loss_history = train_dem(
        initial_dem,
        physical_syndromes,
        DetectorCombinedStatistics(),
        steps=3,
        model_shots=1_000,
        learning_rate=0.0001,
        seed=2,
        callback=record_metrics,
    )

    assert len(reported_metrics) == len(loss_history)
    assert len(loss_history) > 0
    np.testing.assert_allclose(
        [metrics["loss"] for metrics in reported_metrics],
        loss_history,
    )
    assert all(
        set(metrics) == {
            "firing_rate_rmse",
            "pair_correlation_rmse",
            "loss",
        }
        for metrics in reported_metrics
    )
