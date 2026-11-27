"""Plotting helpers for DEM optimization metrics."""

from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import AsinhNorm


def save_correlation_heatmaps(
    device_correlations: np.ndarray,
    initial_correlations: np.ndarray,
    optimized_correlations: np.ndarray,
    output_path: Path,
) -> None:
    """Save device, initial, and optimized correlation matrices side by side.

    A shared zero-centered asinh color scale makes values directly comparable across
    all three panels while preserving small pair correlations beside larger diagonal
    firing probabilities. Matrix diagonals contain single-detector firing probabilities
    and off-diagonal entries contain detector-pair correlations.

    Args:
        device_correlations: Correlations estimated from device syndrome samples.
        initial_correlations: Correlations calculated analytically from the initial DEM.
        optimized_correlations: Correlations calculated analytically from the optimized
            DEM.
        output_path: Output path for the generated figure.

    Returns:
        None.

    Raises:
        ValueError: If the matrices are not same-shaped square matrices or contain no
            finite values.
    """
    matrices = (
        device_correlations,
        initial_correlations,
        optimized_correlations,
    )
    shape = device_correlations.shape
    if device_correlations.ndim != 2 or shape[0] != shape[1]:
        raise ValueError("Correlation heatmaps require square matrices")
    if any(matrix.shape != shape for matrix in matrices[1:]):
        raise ValueError("Correlation heatmap matrices must have the same shape")
    finite_values = np.concatenate(
        [matrix[np.isfinite(matrix)] for matrix in matrices]
    )
    if finite_values.size == 0:
        raise ValueError("Correlation heatmaps require at least one finite value")
    color_limit = float(np.max(np.abs(finite_values)))
    if color_limit == 0:
        color_limit = 1.0
    off_diagonal = ~np.eye(shape[0], dtype=bool)
    finite_pair_values = np.concatenate(
        [
            matrix[off_diagonal & np.isfinite(matrix)]
            for matrix in matrices
        ]
    )
    pair_limit = (
        float(np.max(np.abs(finite_pair_values)))
        if finite_pair_values.size > 0
        else 0.0
    )
    linear_width = (
        pair_limit * 0.05
        if pair_limit > 0
        else color_limit * 0.05
    )
    color_norm = AsinhNorm(
        linear_width=linear_width,
        vmin=-color_limit,
        vmax=color_limit,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(
        1,
        3,
        figsize=(8.0, 2.8),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    titles = ("Device", "Initial DEM", "Optimized DEM")
    image = None
    for axes_item, matrix, title in zip(axes, matrices, titles):
        image = axes_item.imshow(
            matrix,
            cmap="RdBu_r",
            norm=color_norm,
            interpolation="nearest",
        )
        axes_item.set_title(title, fontsize=12)
        axes_item.set_xlabel("Detector", fontsize=10)
        axes_item.tick_params(labelsize=9)
    axes[0].set_ylabel("Detector", fontsize=10)
    colorbar = figure.colorbar(image, ax=axes, shrink=0.82, label="Correlation")
    colorbar_ticks = [-color_limit, -linear_width, 0.0, linear_width, color_limit]
    colorbar.set_ticks(colorbar_ticks)
    colorbar.set_ticklabels([f"{value:.3g}" for value in colorbar_ticks])
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def save_loss_plot(
    loss_history: list[float],
    output_path: Path,
) -> None:
    """Save the optimization RMSE history.

    Args:
        loss_history: Combined detector-correlation RMSE at evaluation zero and after
            each objective evaluation.
        output_path: Output path for the generated figure.

    Returns:
        None.

    Raises:
        ValueError: If the loss history is empty.
    """
    if not loss_history:
        raise ValueError("Loss history must not be empty")

    _save_rmse_plot(loss_history, "Detector-correlation RMSE", output_path)


def save_single_loss_plot(
    single_rmse_history: list[float],
    output_path: Path,
) -> None:
    """Save the single-detector RMSE history.

    Args:
        single_rmse_history: Single-detector RMSE at evaluation zero and after each
            objective evaluation.
        output_path: Output path for the generated figure.

    Returns:
        None.

    Raises:
        ValueError: If the RMSE history is empty.
    """
    if not single_rmse_history:
        raise ValueError("Single-detector RMSE history must not be empty")

    _save_rmse_plot(single_rmse_history, "Single-detector RMSE", output_path)


def save_pair_loss_plot(
    pair_rmse_history: list[float],
    output_path: Path,
) -> None:
    """Save the detector-pair RMSE history.

    Args:
        pair_rmse_history: Detector-pair RMSE at evaluation zero and after each
            objective evaluation.
        output_path: Output path for the generated figure.

    Returns:
        None.

    Raises:
        ValueError: If the RMSE history is empty.
    """
    if not pair_rmse_history:
        raise ValueError("Detector-pair RMSE history must not be empty")

    _save_rmse_plot(pair_rmse_history, "Detector-pair RMSE", output_path)


def _save_rmse_plot(
    rmse_history: list[float],
    title: str,
    output_path: Path,
) -> None:
    """Save an RMSE history using the common loss-plot style.

    Args:
        rmse_history: RMSE at evaluation zero and after each objective evaluation.
        title: Figure title.
        output_path: Output path for the generated figure.

    Returns:
        None.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(figsize=(3.2, 2.8))
    evaluations = range(len(rmse_history))
    axes.plot(evaluations, rmse_history)
    axes.set_title(title, fontsize=12)
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
    baseline_ler: float | None,
    final_evaluation: int,
    output_path: Path,
) -> None:
    """Save the LER history or the two held-out LER measurements.

    Args:
        ler_history: Intermediate objective-evaluation numbers and candidate LER
            values.
        initial_ler: Logical error rate at evaluation zero, before optimization.
        optimized_ler: Logical error rate of the DEM returned by the optimizer.
        baseline_ler: LER from the simulator's DEM, or None for a held-out comparison.
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
    if baseline_ler is None:
        axes.plot(["Initial DEM", "Fitted DEM"], [initial_ler, optimized_ler], marker="o")
        axes.set_title("Held-out LER", fontsize=12)
    else:
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
        axes.plot(evaluations, values, marker="o", markersize=3, label="Training")
        axes.axhline(baseline_ler, color="tab:orange", linestyle="--", label="Physical-device DEM")
        axes.set_title("LER History", fontsize=12)
        axes.set_xlabel("Objective Evaluation", fontsize=11)
        axes.legend(fontsize=8)
    axes.set_ylabel("LER", fontsize=11)
    axes.tick_params(labelsize=10)
    axes.grid(axis="y", alpha=0.3)
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)
