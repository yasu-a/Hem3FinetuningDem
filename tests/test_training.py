"""Tests for DEM optimization metric reporting."""

import numpy as np
import pytest
import stim

from fdem.__main__ import (
    parse_arguments,
    prior_change_summary,
    train_dem_with_cobyla,
    train_dem_with_gradient_descent,
    train_dem_with_pair_least_squares,
)
from fdem.pij import dem_pair_probabilities
from fdem.statistics import (
    DetectorCombinedStatistics,
    DetectorPairCorrelationStatistics,
)


@pytest.mark.parametrize(
    ("extra_arguments", "expected_method"),
    [
        ([], "cobyla"),
        (["--training-method", "pair-lsq"], "pair-lsq"),
        (["--training-method", "gradient-descent"], "gradient-descent"),
    ],
)
def test_training_method_can_be_selected_from_command_line(
    monkeypatch: pytest.MonkeyPatch,
    extra_arguments: list[str],
    expected_method: str,
) -> None:
    """The command line selects the iterative or direct pair fitting method."""
    monkeypatch.setattr(
        "sys.argv",
        ["fdem", "--experiment", "sample", *extra_arguments],
    )

    arguments = parse_arguments()

    assert arguments.training_method == expected_method


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

    _, loss_history = train_dem_with_cobyla(
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
            "single_rmse",
            "pair_rmse",
            "rmse",
            "loss",
        }
        for metrics in reported_metrics
    )


def test_pair_least_squares_training_reduces_unified_rmse() -> None:
    """The direct trainer reduces unified RMSE and updates single mechanisms."""
    initial_dem = stim.DetectorErrorModel(
        """
        error(0.01) D0
        error(0.01) D1
        error(0.01) D2 L0
        error(0.01) D0 D1
        error(0.01) D0 D2 L0
        error(0.01) D1 D2 L0
        """
    )
    target_dem = stim.DetectorErrorModel(
        """
        error(0.08) D0
        error(0.06) D1
        error(0.04) D2 L0
        error(0.12) D0 D1
        error(0.07) D0 D2 L0
        error(0.05) D1 D2 L0
        """
    )
    physical_syndromes, _, _ = target_dem.compile_sampler(seed=5).sample(100_000)
    reported: list[tuple[int, dict[str, float]]] = []

    def record_result(
        evaluation: int,
        metrics: dict[str, float],
        candidate_dem: stim.DetectorErrorModel,
    ) -> None:
        """Record the one-shot pair-fit result.

        Args:
            evaluation: Number of completed fitting evaluations.
            metrics: Named loss metrics for the fitted DEM.
            candidate_dem: DEM returned by the direct fit.

        Returns:
            None.
        """
        assert candidate_dem.num_detectors == 3
        reported.append((evaluation, metrics))

    fitted_dem, loss_history = train_dem_with_pair_least_squares(
        initial_dem,
        physical_syndromes,
        callback=record_result,
    )

    observed = DetectorPairCorrelationStatistics().calculate(physical_syndromes)
    fitted = dem_pair_probabilities(fitted_dem)
    initial = dem_pair_probabilities(initial_dem)
    statistics = DetectorPairCorrelationStatistics()
    initial_loss = statistics.difference(observed, initial)
    fitted_loss = statistics.difference(observed, fitted)
    assert fitted_loss < initial_loss
    assert fitted_dem[0].args_copy()[0] != pytest.approx(
        initial_dem[0].args_copy()[0]
    )
    assert loss_history == pytest.approx([fitted_loss])
    assert reported[0][0] == 1
    assert reported[0][1]["loss"] == pytest.approx(loss_history[0])
    assert set(reported[0][1]) == {
        "single_rmse",
        "pair_rmse",
        "rmse",
        "loss",
    }


def test_gradient_descent_reduces_rmse_and_updates_single_prior() -> None:
    """Analytic gradient descent lowers unified RMSE and updates a single prior."""
    initial_dem = stim.DetectorErrorModel(
        """
        error(0.03) D0 D1
        error(0.02) D0 D2
        error(0.01) D1 D2
        error(0.04) D2
        """
    )
    target_dem = stim.DetectorErrorModel(
        """
        error(0.12) D0 D1
        error(0.08) D0 D2
        error(0.06) D1 D2
        error(0.2) D2
        """
    )
    physical_syndromes, _, _ = target_dem.compile_sampler(seed=4).sample(50_000)

    fitted_dem, loss_history = train_dem_with_gradient_descent(
        initial_dem,
        physical_syndromes,
        steps=50,
        learning_rate=1.0,
    )

    assert len(loss_history) == 50
    assert loss_history[-1] < loss_history[0]
    assert fitted_dem[3].args_copy()[0] > initial_dem[3].args_copy()[0]


def test_prior_change_summary_groups_mechanisms_by_detector_support() -> None:
    """Prior diagnostics count and compare zero-, single-, and multi-detector errors."""
    initial = stim.DetectorErrorModel(
        """
        error(0.01) L0
        error(0.02) D0
        error(0.03) D0 D1
        """
    )
    optimized = stim.DetectorErrorModel(
        """
        error(0.01) L0
        error(0.05) D0
        error(0.07) D0 D1
        """
    )

    summary = prior_change_summary(initial, optimized)

    assert summary["zero_detector_count"] == 1
    assert summary["single_detector_count"] == 1
    assert summary["multiple_detector_count"] == 1
    assert summary["single_prior_mean_absolute_change"] == pytest.approx(0.03)
    assert summary["multiple_prior_mean_absolute_change"] == pytest.approx(0.04)
