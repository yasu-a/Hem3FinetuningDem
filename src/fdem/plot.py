"""Plotting helpers for DEM optimization metrics."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def save_loss_plot(
    loss_history: list[float],
    output_path: Path,
) -> None:
    """Save the optimization RMSE history.

    Args:
        loss_history: Pair-correlation RMSE from each objective evaluation.
        output_path: Output path for the generated figure.

    Returns:
        None.

    Raises:
        ValueError: If the loss history is empty.
    """
    if not loss_history:
        raise ValueError("Loss history must not be empty")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(figsize=(3.2, 2.8))
    evaluations = range(1, len(loss_history) + 1)
    axes.plot(evaluations, loss_history)
    axes.set_title("Pair-correlation RMSE", fontsize=12)
    axes.set_xlabel("Objective Evaluation", fontsize=11)
    axes.set_ylabel("RMSE", fontsize=11)
    axes.tick_params(labelsize=10)
    axes.grid(alpha=0.3)
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
    """Save the complete logical-error-rate history as one line.

    Args:
        ler_history: Intermediate objective-evaluation numbers and candidate LER
            values.
        initial_ler: Logical error rate at evaluation zero, before optimization.
        optimized_ler: Logical error rate of the DEM returned by the optimizer.
        final_evaluation: Evaluation number used for the final optimized point.
        output_path: Output path for the generated figure.

    Returns:
        None.

    Raises:
        ValueError: If the final evaluation is not positive.
    """
    if final_evaluation <= 0:
        raise ValueError("Final evaluation must be positive")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(figsize=(3.2, 2.8))
    intermediate_history = [
        (evaluation, ler)
        for evaluation, ler in ler_history
        if 0 < evaluation < final_evaluation
    ]
    complete_history = [
        (0, initial_ler),
        *intermediate_history,
        (final_evaluation, optimized_ler),
    ]
    evaluations, values = zip(*complete_history)
    axes.plot(evaluations, values, marker="o", markersize=3)
    axes.set_title("LER History", fontsize=12)
    axes.set_xlabel("Objective Evaluation", fontsize=11)
    axes.set_ylabel("LER", fontsize=11)
    axes.tick_params(labelsize=10)
    axes.grid(axis="y", alpha=0.3)
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)
