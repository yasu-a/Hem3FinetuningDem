"""Tests for optimization metric plots."""

from pathlib import Path

import pytest

from fdem.plot import save_loss_plot


def test_loss_plot_saves_rmse_history(tmp_path: Path) -> None:
    """The loss plot accepts a pair-correlation RMSE history."""
    output_path = tmp_path / "loss.png"

    save_loss_plot([0.3, 0.2], output_path)

    assert output_path.is_file()
    assert output_path.stat().st_size > 0


def test_loss_plot_rejects_empty_history(tmp_path: Path) -> None:
    """The loss plot rejects an empty RMSE history."""
    with pytest.raises(ValueError, match="must not be empty"):
        save_loss_plot([], tmp_path / "loss.png")
