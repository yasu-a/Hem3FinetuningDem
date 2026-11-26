"""Tests for DEM optimization metric reporting."""

import numpy as np
import pytest
import stim

from fdem.__main__ import (
    parse_arguments,
    train_dem_with_cobyla,
    train_dem_with_pair_least_squares,
)
from fdem.pij import dem_pair_probabilities
from fdem.statistics import (
    DetectorCombinedStatistics,
    DetectorPairCorrelationStatistics,
)


@pytest.mark.parametrize(
    ("extra_arguments", "expected_method"),
    [([], "cobyla"), (["--training-method", "pair-lsq"], "pair-lsq")],
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
            "firing_rate_rmse",
            "pair_correlation_rmse",
            "loss",
        }
        for metrics in reported_metrics
    )


def test_pair_least_squares_training_fits_observed_correlations() -> None:
    """The direct trainer fits pair correlations and reports its single RMSE."""
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
    patterns = [[0, 0, 0], [1, 1, 0], [1, 1, 1], [0, 0, 1]]
    physical_syndromes = np.repeat(patterns, [105, 15, 7, 1], axis=0)
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
    np.testing.assert_allclose(fitted, observed, equal_nan=True, atol=1e-10)
    assert loss_history == pytest.approx([0.0], abs=1e-10)
    assert reported == [(1, {"loss": loss_history[0]})]
