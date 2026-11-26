"""Plotting helpers for DEM optimization metrics."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def save_loss_plot(
    firing_rate_rmse_history: list[float],
    pair_correlation_rmse_history: list[float],
    combined_loss_history: list[float],
    output_path: Path,
) -> None:
    """Save the component RMSE values and combined optimization loss.

    Args:
        firing_rate_rmse_history: Firing-rate RMSE from each objective evaluation.
        pair_correlation_rmse_history: Pair-correlation RMSE from each evaluation.
        combined_loss_history: Combined loss from each objective evaluation.
        output_path: Output path for the generated figure.

    Returns:
        None.

    Raises:
        ValueError: If a history is empty or the history lengths differ.
    """
    histories = (
        firing_rate_rmse_history,
        pair_correlation_rmse_history,
        combined_loss_history,
    )
    if any(not history for history in histories):
        raise ValueError("Loss histories must not be empty")
    if len({len(history) for history in histories}) != 1:
        raise ValueError("Loss histories must have the same length")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(figsize=(3.2, 2.8))
    evaluations = range(1, len(combined_loss_history) + 1)
    axes.plot(evaluations, firing_rate_rmse_history, label="Firing-rate RMSE")
    axes.plot(
        evaluations,
        pair_correlation_rmse_history,
        label="Pair-correlation RMSE",
    )
    axes.plot(evaluations, combined_loss_history, label="Combined loss")
    axes.set_title("Optimization Loss", fontsize=12)
    axes.set_xlabel("Objective Evaluation", fontsize=11)
    axes.set_ylabel("RMSE", fontsize=11)
    axes.tick_params(labelsize=10)
    axes.grid(alpha=0.3)
    axes.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def save_ler_plot(
    ler_history: list[tuple[int, float]],
    initial_ler: float,
    optimized_ler: float,
    final_evaluation: int,
    output_path: Path,
) -> None:
    """Save the logical-error-rate history.

    Args:
        ler_history: Objective-evaluation numbers and their candidate LER values.
        initial_ler: Logical error rate before optimization.
        optimized_ler: Logical error rate after optimization.
        final_evaluation: Evaluation number at which to show the optimized result.
        output_path: Output path for the generated figure.

    Returns:
        None.

    Raises:
        ValueError: If the LER history is empty.
    """
    if not ler_history:
        raise ValueError("LER history must not be empty")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(figsize=(3.2, 2.8))
    evaluations, values = zip(*ler_history)
    axes.plot(evaluations, values, marker="o", markersize=3)
    axes.axhline(initial_ler, linestyle="--", color="gray", label="Initial")
    axes.scatter(
        [final_evaluation],
        [optimized_ler],
        marker="*",
        s=60,
        color="tab:red",
        label="Optimized",
        zorder=3,
    )
    axes.set_title("LER History", fontsize=12)
    axes.set_xlabel("Objective Evaluation", fontsize=11)
    axes.set_ylabel("LER", fontsize=11)
    axes.tick_params(labelsize=10)
    axes.grid(axis="y", alpha=0.3)
    axes.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)
