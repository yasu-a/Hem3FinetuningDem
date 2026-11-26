"""Tests for optimization metric plots."""

from pathlib import Path

import pytest

from fdem.plot import save_ler_plot, save_loss_plot


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
