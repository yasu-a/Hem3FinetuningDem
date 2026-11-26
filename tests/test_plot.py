"""Tests for optimization metric plots."""

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest

from fdem.plot import (
    save_correlation_heatmaps,
    save_ler_plot,
    save_loss_plot,
    save_pair_loss_plot,
    save_single_loss_plot,
)


def test_correlation_heatmaps_save_three_comparable_matrices(tmp_path: Path) -> None:
    """The correlation heatmap accepts device, initial, and optimized matrices."""
    output_path = tmp_path / "correlation_heatmaps.png"
    device = np.array([[0.2, 0.03], [0.03, 0.1]])
    initial = np.array([[0.1, 0.01], [0.01, 0.08]])
    optimized = np.array([[0.18, 0.025], [0.025, 0.09]])

    save_correlation_heatmaps(device, initial, optimized, output_path)

    assert output_path.is_file()
    assert output_path.stat().st_size > 0


def test_correlation_heatmaps_reject_mismatched_shapes(tmp_path: Path) -> None:
    """The correlation heatmap rejects matrices that cannot be compared by detector."""
    square = np.zeros((2, 2))

    with pytest.raises(ValueError, match="same shape"):
        save_correlation_heatmaps(
            square,
            np.zeros((3, 3)),
            square,
            tmp_path / "correlation_heatmaps.png",
        )


def test_correlation_heatmaps_reject_all_nonfinite_values(tmp_path: Path) -> None:
    """The correlation heatmap rejects input with no displayable values."""
    undefined = np.full((2, 2), np.nan)

    with pytest.raises(ValueError, match="finite value"):
        save_correlation_heatmaps(
            undefined,
            undefined,
            undefined,
            tmp_path / "correlation_heatmaps.png",
        )


def test_loss_plot_saves_rmse_history(tmp_path: Path) -> None:
    """The loss plot accepts a unified detector-correlation RMSE history."""
    output_path = tmp_path / "loss.png"

    save_loss_plot([0.3, 0.2], output_path)

    assert output_path.is_file()
    assert output_path.stat().st_size > 0


def test_loss_plot_rejects_empty_history(tmp_path: Path) -> None:
    """The loss plot rejects an empty RMSE history."""
    with pytest.raises(ValueError, match="must not be empty"):
        save_loss_plot([], tmp_path / "loss.png")


@pytest.mark.parametrize(
    ("save_plot", "filename"),
    [
        (save_single_loss_plot, "single_loss.png"),
        (save_pair_loss_plot, "pair_loss.png"),
    ],
)
def test_component_loss_plot_saves_initial_and_evaluation_history(
    tmp_path: Path,
    save_plot: Callable[[list[float], Path], None],
    filename: str,
) -> None:
    """Each component plot accepts an initial value followed by evaluation values."""
    output_path = tmp_path / filename

    save_plot([0.3, 0.2, 0.1], output_path)

    assert output_path.is_file()
    assert output_path.stat().st_size > 0


def test_ler_plot_connects_initial_intermediate_and_optimized_values(
    tmp_path: Path,
) -> None:
    """The LER plot accepts one history from evaluation zero to the final result."""
    output_path = tmp_path / "ler.png"

    save_ler_plot(
        [(10, 0.2), (20, 0.15), (30, 0.14)],
        initial_ler=0.25,
        optimized_ler=0.12,
        baseline_ler=0.18,
        final_evaluation=30,
        output_path=output_path,
    )

    assert output_path.is_file()
    assert output_path.stat().st_size > 0


def test_ler_plot_supports_optimization_without_intermediate_samples(
    tmp_path: Path,
) -> None:
    """Initial and optimized LER values can form the line without periodic samples."""
    output_path = tmp_path / "ler.png"

    save_ler_plot([], 0.25, 0.12, 0.18, 5, output_path)

    assert output_path.is_file()
