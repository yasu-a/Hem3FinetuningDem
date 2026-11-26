"""Minimal example for estimating a rotated surface code LER using a Stim DEM."""

import argparse
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pymatching
import stim
from scipy.optimize import minimize
from tqdm import tqdm

from .dem import detector_support, with_priors
from .device import (
    get_reference_dem,
    logical_error_rate_from_physical_device_dem,
    sample_physical_device_syndromes,
)
from .pij import (
    dem_pair_probabilities,
    fit_dem_from_pij,
    pair_probability_mse_and_gradient,
)
from .plot import save_ler_plot, save_loss_plot
from .statistics import DetectorPairCorrelationStatistics, ISyndromeStatistics


# Optimization settings. Change these values to try different configurations.
SHOTS = 100_000
MODEL_SHOTS = 10_000
OPTIMIZATION_STEPS = 10000
LEARNING_RATE = 0.0001
GRADIENT_LEARNING_RATE = 0.1
REFERENCE_ERROR_PROBABILITY = 0.001
PAIR_LOSS_WEIGHT = 1.0
LER_EVALUATION_INTERVAL = 10
SEED = 0
RESULTS_DIRECTORY = Path("var")
TRAINING_METHODS = ("cobyla", "pair-lsq", "gradient-descent")


def parse_arguments() -> argparse.Namespace:
    """Parse the experiment selection from the command line.

    Returns:
        Parsed command-line arguments containing the experiment directory name.
    """
    parser = argparse.ArgumentParser(prog="fdem", description=__doc__)
    parser.add_argument(
        "--experiment",
        required=True,
        help="Experiment directory name under experiments/",
    )
    parser.add_argument(
        "--training-method",
        choices=TRAINING_METHODS,
        default="cobyla",
        help="DEM probability fitting method (default: cobyla)",
    )
    return parser.parse_args()


def logical_error_rate(
    dem: stim.DetectorErrorModel,
    detectors: np.ndarray,
    observables: np.ndarray,
) -> float:
    """Decode evaluation data and calculate its logical error rate.

    Args:
        dem: Detector error model used to build the matching decoder.
        detectors: Detection-event samples with shape ``(shots, detectors)``.
        observables: Actual logical-observable flips with shape
            ``(shots, observables)``.

    Returns:
        The fraction of shots decoded with at least one incorrect observable.
    """
    decoder = pymatching.Matching.from_detector_error_model(dem)
    predictions = decoder.decode_batch(detectors)
    return float((predictions != observables).any(axis=1).mean())


def prior_change_summary(
    initial_dem: stim.DetectorErrorModel,
    optimized_dem: stim.DetectorErrorModel,
) -> dict[str, int | float]:
    """Summarize mechanism counts and absolute prior changes by detector support.

    Args:
        initial_dem: DEM before optimization.
        optimized_dem: DEM with the same error-mechanism structure after optimization.

    Returns:
        Counts for zero-, single-, and multiple-detector mechanisms and mean absolute
        prior changes for the latter two visible groups.

    Raises:
        ValueError: If the two DEMs do not have matching error-mechanism structures.
    """
    initial_errors = [
        instruction for instruction in initial_dem.flattened()
        if instruction.type == "error"
    ]
    optimized_errors = [
        instruction for instruction in optimized_dem.flattened()
        if instruction.type == "error"
    ]
    if len(initial_errors) != len(optimized_errors):
        raise ValueError("Initial and optimized DEM error counts must match")

    changes: dict[int, list[float]] = {0: [], 1: [], 2: []}
    for initial, optimized in zip(initial_errors, optimized_errors):
        if initial.targets_copy() != optimized.targets_copy():
            raise ValueError("Initial and optimized DEM error targets must match")
        support_size = len(detector_support(initial))
        group = min(support_size, 2)
        changes[group].append(
            abs(optimized.args_copy()[0] - initial.args_copy()[0])
        )

    return {
        "zero_detector_count": len(changes[0]),
        "single_detector_count": len(changes[1]),
        "multiple_detector_count": len(changes[2]),
        "single_prior_mean_absolute_change": (
            float(np.mean(changes[1])) if changes[1] else 0.0
        ),
        "multiple_prior_mean_absolute_change": (
            float(np.mean(changes[2])) if changes[2] else 0.0
        ),
    }


def train_dem_with_cobyla(
    initial_dem: stim.DetectorErrorModel,
    physical_device_syndromes: np.ndarray,
    syndrome_statistics: ISyndromeStatistics,
    *,
    steps: int,
    model_shots: int,
    learning_rate: float,
    seed: int,
    callback: Callable[
        [int, dict[str, float], stim.DetectorErrorModel],
        None,
    ] | None = None,
) -> tuple[stim.DetectorErrorModel, list[float]]:
    """Optimize DEM probabilities against physical-device syndrome statistics.

    Args:
        initial_dem: Detector error model used as the optimization starting point.
        physical_device_syndromes: Syndrome samples collected from the physical device,
            with shape ``(shots, detectors)``.
        syndrome_statistics: Strategy used to calculate and compare syndrome statistics.
        steps: Maximum number of COBYLA objective-function evaluations.
        model_shots: Number of DEM samples generated for each evaluation.
        learning_rate: Initial COBYLA trust-region radius.
        seed: Random seed used for every DEM sampling evaluation.
        callback: Optional function called after each objective-function evaluation with
            the evaluation number, current loss metrics, and candidate DEM.

    Returns:
        A tuple containing the optimized DEM and the loss from each objective-function
        evaluation.

    Raises:
        ValueError: If the initial DEM contains no error instructions.
    """
    initial_priors = np.array(
        [
            instruction.args_copy()[0]
            for instruction in initial_dem
            if instruction.type == "error"
        ],
        dtype=float,
    )
    if initial_priors.size == 0:
        raise ValueError("Initial DEM must contain at least one error instruction")

    target_statistics = syndrome_statistics.calculate(physical_device_syndromes)
    loss_history: list[float] = []

    def objective(priors: np.ndarray) -> float:
        """Calculate the statistics loss for one candidate prior vector.

        Args:
            priors: Candidate error probabilities for the DEM.

        Returns:
            The difference between physical-device and model statistics.
        """
        model_dem = with_priors(initial_dem, priors)
        model_syndromes, _, _ = model_dem.compile_sampler(seed=seed).sample(model_shots)
        model_statistics = syndrome_statistics.calculate(model_syndromes)
        metrics = syndrome_statistics.metrics(target_statistics, model_statistics)
        loss = metrics["loss"]
        loss_history.append(loss)
        if callback is not None:
            callback(len(loss_history), metrics, model_dem)
        return loss

    result = minimize(
        objective,
        initial_priors,
        method="COBYLA",
        bounds=[(1e-9, 0.5 - 1e-9)] * initial_priors.size,
        options={
            "maxiter": steps,
            "rhobeg": learning_rate,
        },
    )
    return with_priors(initial_dem, result.x), loss_history


def train_dem_with_pair_least_squares(
    initial_dem: stim.DetectorErrorModel,
    physical_device_syndromes: np.ndarray,
    pair_loss_weight: float = 1.0,
    callback: Callable[
        [int, dict[str, float], stim.DetectorErrorModel],
        None,
    ] | None = None,
) -> tuple[stim.DetectorErrorModel, list[float]]:
    """Fit DEM probabilities directly from unified detector correlations.

    The fit converts independent error probabilities into additive parity log rates
    and solves firing-rate and detector-pair incidence equations with nonnegative
    bounded least squares. Error mechanisms that flip no detectors retain their initial
    probabilities.

    Args:
        initial_dem: Detector error model whose fixed structure will be fitted.
        physical_device_syndromes: Syndrome samples collected from the physical device,
            with shape ``(shots, detectors)``.
        pair_loss_weight: Nonnegative weight applied to off-diagonal pair MSE.
        callback: Optional function called once with the fitted loss and DEM.

    Returns:
        A tuple containing the fitted DEM and a one-element combined RMSE history.

    Raises:
        ValueError: If the syndrome statistics or DEM-supported pair equations are
            invalid.
        RuntimeError: If the bounded least-squares solver does not converge.
    """
    statistics = DetectorPairCorrelationStatistics(pair_loss_weight)
    observed_correlations = statistics.calculate(physical_device_syndromes)
    fitted_dem = fit_dem_from_pij(
        initial_dem,
        observed_correlations,
        pair_loss_weight,
    )
    fitted_correlations = dem_pair_probabilities(fitted_dem)
    metrics = statistics.metrics(observed_correlations, fitted_correlations)
    loss = metrics["loss"]
    if callback is not None:
        callback(1, metrics, fitted_dem)
    return fitted_dem, [loss]


def train_dem_with_gradient_descent(
    initial_dem: stim.DetectorErrorModel,
    physical_device_syndromes: np.ndarray,
    *,
    steps: int,
    learning_rate: float,
    pair_loss_weight: float = 1.0,
    callback: Callable[
        [int, dict[str, float], stim.DetectorErrorModel],
        None,
    ] | None = None,
) -> tuple[stim.DetectorErrorModel, list[float]]:
    """Fit DEM probabilities using projected analytic gradient descent.

    The optimization variables are parity log rates. Each iteration evaluates the
    exact DEM firing and pair probabilities, differentiates their weighted MSE
    analytically, and projects updated log rates onto the nonnegative domain.
    Detector-invisible mechanisms have zero gradient and retain their initial
    probabilities.

    Args:
        initial_dem: Detector error model whose fixed structure will be fitted.
        physical_device_syndromes: Syndrome samples collected from the physical device,
            with shape ``(shots, detectors)``.
        steps: Number of loss and gradient evaluations.
        learning_rate: Positive gradient-descent step size in log-rate space.
        pair_loss_weight: Nonnegative weight applied to off-diagonal pair MSE.
        callback: Optional function called after each evaluation with its RMSE and DEM.

    Returns:
        A tuple containing the fitted DEM and the combined RMSE history.

    Raises:
        ValueError: If steps or the learning rate is invalid, an initial prior is at
            least 0.5, or the pair statistics cannot define the loss and gradient.
    """
    if steps <= 0:
        raise ValueError("Gradient-descent steps must be positive")
    if not np.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("Gradient learning rate must be finite and positive")

    flattened = initial_dem.flattened()
    initial_priors = np.array(
        [
            instruction.args_copy()[0]
            for instruction in flattened
            if instruction.type == "error"
        ],
        dtype=float,
    )
    if initial_priors.size == 0:
        raise ValueError("Initial DEM must contain at least one error instruction")
    if np.any(initial_priors >= 0.5):
        raise ValueError("Gradient-descent DEM probabilities must be below 0.5")

    statistics = DetectorPairCorrelationStatistics(pair_loss_weight)
    observed_correlations = statistics.calculate(
        physical_device_syndromes
    )
    log_rates = -np.log1p(-2 * initial_priors)
    loss_history: list[float] = []
    candidate_dem = flattened

    for evaluation in range(1, steps + 1):
        priors = -np.expm1(-log_rates) / 2
        candidate_dem = with_priors(flattened, priors)
        mse, gradient = pair_probability_mse_and_gradient(
            candidate_dem,
            observed_correlations,
            pair_loss_weight,
        )
        rmse = float(np.sqrt(mse))
        loss_history.append(rmse)
        if callback is not None:
            model_correlations = dem_pair_probabilities(candidate_dem)
            metrics = statistics.metrics(
                observed_correlations,
                model_correlations,
            )
            callback(evaluation, metrics, candidate_dem)
        if evaluation < steps:
            log_rates = np.maximum(0.0, log_rates - learning_rate * gradient)

    return candidate_dem, loss_history


def main() -> None:
    """Collect syndrome statistics and estimate the initial decoder LER.

    Returns:
        None.
    """
    arguments = parse_arguments()
    experiment_results_directory = (
        RESULTS_DIRECTORY / arguments.experiment / arguments.training_method
    )
    loss_plot_path = experiment_results_directory / "loss.png"
    ler_plot_path = experiment_results_directory / "ler.png"
    print(f"experiment = {arguments.experiment}")
    print(f"training method = {arguments.training_method}")

    # 1. Collect training and evaluation data through the device sampling interface.
    syndrome_samples = sample_physical_device_syndromes(
        arguments.experiment,
        SHOTS,
        SEED,
    )
    detector_samples, actual_observables = sample_physical_device_syndromes(
        arguments.experiment,
        SHOTS,
        SEED + 2,
        separate_observables=True,
    )
    if syndrome_samples.shape[1] != detector_samples.shape[1]:
        raise ValueError("Training and evaluation detector counts must match")

    print(f"detectors = {syndrome_samples.shape[1]}")
    print(f"observables = {actual_observables.shape[1]}")

    # 2. Build the reference DEM from ideal circuit structure and uniform noise.
    initial_dem = get_reference_dem(
        arguments.experiment,
        error_probability=REFERENCE_ERROR_PROBABILITY,
    )
    if initial_dem.num_detectors != syndrome_samples.shape[1]:
        raise ValueError("Reference DEM and device detector counts must match")
    if initial_dem.num_observables != actual_observables.shape[1]:
        raise ValueError("Reference DEM and device observable counts must match")
    error_instructions = [
        instruction for instruction in initial_dem if instruction.type == "error"
    ]
    statistics_calculator = DetectorPairCorrelationStatistics(PAIR_LOSS_WEIGHT)
    syndrome_correlations = statistics_calculator.calculate(syndrome_samples)
    initial_metrics = statistics_calculator.metrics(
        syndrome_correlations,
        dem_pair_probabilities(initial_dem),
    )

    # 3. Track LER using the fixed evaluation samples collected above.
    initial_ler = logical_error_rate(initial_dem, detector_samples, actual_observables)
    physical_dem_ler = logical_error_rate_from_physical_device_dem(
        arguments.experiment,
        detector_samples,
        actual_observables,
    )
    ler_history: list[tuple[int, float]] = []
    latest_ler: float | None = None

    progress_total = (
        1 if arguments.training_method == "pair-lsq" else OPTIMIZATION_STEPS
    )
    with tqdm(total=progress_total, desc="Optimizing DEM", unit="eval") as progress:

        def update_progress(
            evaluation: int,
            metrics: dict[str, float],
            candidate_dem: stim.DetectorErrorModel,
        ) -> None:
            """Update progress and periodically record the candidate LER.

            Args:
                evaluation: Number of completed objective-function evaluations.
                metrics: Loss metrics from the latest evaluation.
                candidate_dem: DEM evaluated by the current objective-function call.

            Returns:
                None.
            """
            nonlocal latest_ler

            loss = metrics["loss"]
            if evaluation % LER_EVALUATION_INTERVAL == 0:
                latest_ler = logical_error_rate(
                    candidate_dem,
                    detector_samples,
                    actual_observables,
                )
                ler_history.append((evaluation, latest_ler))
            if latest_ler is None:
                progress.set_postfix(loss=f"{loss:.3g}", refresh=False)
            else:
                progress.set_postfix(
                    loss=f"{loss:.3g}",
                    ler=f"{latest_ler:.3g}",
                    refresh=False,
                )
            progress.update(evaluation - progress.n)

        if arguments.training_method == "cobyla":
            optimized_dem, loss_history = train_dem_with_cobyla(
                initial_dem,
                syndrome_samples,
                statistics_calculator,
                steps=OPTIMIZATION_STEPS,
                model_shots=MODEL_SHOTS,
                learning_rate=LEARNING_RATE,
                seed=SEED + 1,
                callback=update_progress,
            )
        elif arguments.training_method == "pair-lsq":
            optimized_dem, loss_history = train_dem_with_pair_least_squares(
                initial_dem,
                syndrome_samples,
                PAIR_LOSS_WEIGHT,
                callback=update_progress,
            )
        else:
            optimized_dem, loss_history = train_dem_with_gradient_descent(
                initial_dem,
                syndrome_samples,
                steps=OPTIMIZATION_STEPS,
                learning_rate=GRADIENT_LEARNING_RATE,
                pair_loss_weight=PAIR_LOSS_WEIGHT,
                callback=update_progress,
            )

    # 4. Evaluate the optimized decoder on the same fixed data.
    optimized_ler = logical_error_rate(optimized_dem, detector_samples, actual_observables)
    optimized_metrics = statistics_calculator.metrics(
        syndrome_correlations,
        dem_pair_probabilities(optimized_dem),
    )
    prior_summary = prior_change_summary(initial_dem, optimized_dem)
    save_loss_plot(loss_history, loss_plot_path)
    save_ler_plot(
        ler_history,
        initial_ler,
        optimized_ler,
        physical_dem_ler,
        len(loss_history),
        ler_plot_path,
    )

    print(f"DEM error instructions = {len(error_instructions)}")
    print(f"first initial error = {error_instructions[0]}")
    print(f"syndrome samples = {syndrome_samples.shape}")
    pair_rows, pair_columns = np.triu_indices(syndrome_correlations.shape[0], k=1)
    print(
        "mean pair correlation = "
        f"{np.nanmean(syndrome_correlations[pair_rows, pair_columns]):.6g}"
    )
    print(f"initial single RMSE = {initial_metrics['single_rmse']:.8g}")
    print(f"optimized single RMSE = {optimized_metrics['single_rmse']:.8g}")
    print(f"initial pair RMSE = {initial_metrics['pair_rmse']:.8g}")
    print(f"optimized pair RMSE = {optimized_metrics['pair_rmse']:.8g}")
    print(f"initial RMSE = {initial_metrics['rmse']:.8g}")
    print(f"optimized RMSE = {optimized_metrics['rmse']:.8g}")
    print(f"best RMSE = {min(loss_history):.8g}")
    print(f"initial LER = {initial_ler:.6g}")
    print(f"optimized LER = {optimized_ler:.6g}")
    print(f"physical-device DEM LER = {physical_dem_ler:.6g}")
    print(f"single-detector error mechanisms = {prior_summary['single_detector_count']}")
    print(
        "multiple-detector error mechanisms = "
        f"{prior_summary['multiple_detector_count']}"
    )
    print(f"zero-detector error mechanisms = {prior_summary['zero_detector_count']}")
    print(
        "single-detector prior mean absolute change = "
        f"{prior_summary['single_prior_mean_absolute_change']:.8g}"
    )
    print(
        "multiple-detector prior mean absolute change = "
        f"{prior_summary['multiple_prior_mean_absolute_change']:.8g}"
    )
    print(f"loss plot = {loss_plot_path}")
    print(f"LER plot = {ler_plot_path}")


if __name__ == "__main__":
    main()
