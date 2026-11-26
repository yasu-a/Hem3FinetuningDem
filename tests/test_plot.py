"""Tests for optimization metric plots."""

from pathlib import Path

import pytest

from fdem.plot import save_loss_plot


def test_loss_plot_saves_all_metric_histories(tmp_path: Path) -> None:
    """The loss plot accepts aligned component and combined histories."""
    output_path = tmp_path / "loss.png"

    save_loss_plot(
        [0.3, 0.2],
        [0.4, 0.25],
        [0.5, 0.32],
        output_path,
    )

    assert output_path.is_file()
    assert output_path.stat().st_size > 0


def test_loss_plot_rejects_different_history_lengths(tmp_path: Path) -> None:
    """The loss plot rejects metric histories from different evaluations."""
    with pytest.raises(ValueError, match="same length"):
        save_loss_plot([0.3], [0.4, 0.2], [0.5], tmp_path / "loss.png")
