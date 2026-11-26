"""Tests for DEM optimization metric reporting."""

import numpy as np
import pytest
import stim

from fdem.__main__ import (
    parse_arguments,
    pair_loss_weight_directory,
    prior_change_summary,
    record_rmse_metrics,
    resolve_pair_loss_weight,
    train_dem_with_cobyla,
    train_dem_with_gradient_descent,
    train_dem_with_least_squares,
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
        (["--training-method", "lsq"], "lsq"),
        (["--training-method", "gradient-descent"], "gradient-descent"),
    ],
)
def test_training_method_can_be_selected_from_command_line(
    monkeypatch: pytest.MonkeyPatch,
    extra_arguments: list[str],
    expected_method: str,
) -> None:
    """The command line selects an iterative or direct correlation fitting method."""
    monkeypatch.setattr(
        "sys.argv",
        ["fdem", "--experiment", "sample", *extra_arguments],
    )

    arguments = parse_arguments()

    assert arguments.training_method == expected_method
    assert arguments.pair_loss_weight == "auto"


@pytest.mark.parametrize(
    ("value", "expected"),
    [("auto", "auto"), ("1", 1.0), ("275.5", 275.5)],
)
def test_pair_loss_weight_accepts_auto_or_positive_number(
    monkeypatch: pytest.MonkeyPatch,
    value: str,
    expected: str | float,
) -> None:
    """The command line accepts automatic and positive manual pair weights."""
    monkeypatch.setattr(
        "sys.argv",
        ["fdem", "--experiment", "sample", "--pair-loss-weight", value],
    )

    arguments = parse_arguments()

    assert arguments.pair_loss_weight == expected


@pytest.mark.parametrize("value", ["0", "-1", "nan", "inf", "invalid"])
def test_pair_loss_weight_rejects_invalid_command_line_value(
    monkeypatch: pytest.MonkeyPatch,
    value: str,
) -> None:
    """The command line rejects nonpositive, nonfinite, and nonnumeric weights."""
    monkeypatch.setattr(
        "sys.argv",
        ["fdem", "--experiment", "sample", "--pair-loss-weight", value],
    )

    with pytest.raises(SystemExit):
        parse_arguments()


def test_auto_pair_loss_weight_balances_initial_mse() -> None:
    """Automatic weighting makes initial single and weighted pair MSE equal."""
    single_rmse = 0.136
    pair_rmse = 0.0082

    mode, weight = resolve_pair_loss_weight("auto", single_rmse, pair_rmse)

    assert mode == "auto"
    assert weight == pytest.approx(single_rmse**2 / pair_rmse**2)
    assert weight * pair_rmse**2 == pytest.approx(single_rmse**2)


def test_manual_pair_loss_weight_is_preserved() -> None:
    """A positive manual weight is returned without data-dependent changes."""
    mode, weight = resolve_pair_loss_weight(10.0, 0.2, 0.01)

    assert mode == "manual"
    assert weight == 10.0


@pytest.mark.parametrize(
    ("single_rmse", "pair_rmse"),
    [(0.0, 0.1), (0.1, 0.0), (np.nan, 0.1), (0.1, np.inf)],
)
def test_auto_pair_loss_weight_rejects_invalid_initial_mse(
    single_rmse: float,
    pair_rmse: float,
) -> None:
    """Automatic weighting rejects zero or nonfinite initial MSE components."""
    with pytest.raises(ValueError):
        resolve_pair_loss_weight("auto", single_rmse, pair_rmse)


@pytest.mark.parametrize(
    ("requested_weight", "expected_directory"),
    [("auto", "weight-auto"), (1.0, "weight-1"), (100.0, "weight-100")],
)
def test_pair_loss_weight_uses_distinct_result_directory(
    requested_weight: str | float,
    expected_directory: str,
) -> None:
    """Automatic and manual selections produce stable weight directories."""
    assert pair_loss_weight_directory(requested_weight) == expected_directory


def test_rmse_histories_record_initial_and_objective_metrics() -> None:
    """RMSE histories preserve the initial value and every evaluation in order."""
    single_history: list[float] = []
    pair_history: list[float] = []
    combined_history: list[float] = []
    metrics_history = [
        {"single_rmse": 0.3, "pair_rmse": 0.2, "rmse": 0.4},
        {"single_rmse": 0.2, "pair_rmse": 0.1, "rmse": 0.25},
        {"single_rmse": 0.1, "pair_rmse": 0.05, "rmse": 0.12},
    ]

    for metrics in metrics_history:
        record_rmse_metrics(
            metrics,
            single_history,
            pair_history,
            combined_history,
        )

    assert single_history == [0.3, 0.2, 0.1]
    assert pair_history == [0.2, 0.1, 0.05]
    assert combined_history == [0.4, 0.25, 0.12]


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
    pair_loss_weight = 7.0

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
        DetectorCombinedStatistics(pair_loss_weight),
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
    assert all(
        metrics["loss"] ** 2
        == pytest.approx(
            metrics["single_rmse"] ** 2
            + pair_loss_weight * metrics["pair_rmse"] ** 2
        )
        for metrics in reported_metrics
    )


def test_least_squares_training_uses_single_and_pair_correlations() -> None:
    """Least squares reduces unified RMSE and updates single and pair mechanisms."""
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

    fitted_dem, loss_history = train_dem_with_least_squares(
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
    assert fitted_dem[3].args_copy()[0] != pytest.approx(
        initial_dem[3].args_copy()[0]
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
