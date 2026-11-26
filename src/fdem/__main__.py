"""Minimal example for estimating a rotated surface code LER using a Stim DEM."""

import argparse
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pymatching
import stim
from scipy.optimize import minimize
from tqdm import tqdm

from .dem import with_priors
from .device import get_reference_dem, sample_physical_device_syndromes
from .pij import dem_pair_probabilities, fit_dem_from_pij
from .plot import save_ler_plot, save_loss_plot
from .statistics import DetectorPairCorrelationStatistics, ISyndromeStatistics


# Optimization settings. Change these values to try different configurations.
SHOTS = 100_000
MODEL_SHOTS = 10_000
OPTIMIZATION_STEPS = 300
LEARNING_RATE = 0.0001
REFERENCE_ERROR_PROBABILITY = 0.001
LER_EVALUATION_INTERVAL = 10
SEED = 0
RESULTS_DIRECTORY = Path("var")
TRAINING_METHODS = ("cobyla", "pair-lsq")


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
    callback: Callable[
        [int, dict[str, float], stim.DetectorErrorModel],
        None,
    ] | None = None,
) -> tuple[stim.DetectorErrorModel, list[float]]:
    """Fit DEM probabilities directly from detector-pair correlations.

    The fit converts independent error probabilities into additive parity log rates
    and solves their detector-pair incidence equations with nonnegative bounded least
    squares. Error mechanisms that do not contribute to any detector pair retain their
    initial probabilities.

    Args:
        initial_dem: Detector error model whose fixed structure will be fitted.
        physical_device_syndromes: Syndrome samples collected from the physical device,
            with shape ``(shots, detectors)``.
        callback: Optional function called once with the fitted loss and DEM.

    Returns:
        A tuple containing the fitted DEM and a one-element pair-correlation RMSE
        history.

    Raises:
        ValueError: If the syndrome statistics or DEM-supported pair equations are
            invalid.
        RuntimeError: If the bounded least-squares solver does not converge.
    """
    statistics = DetectorPairCorrelationStatistics()
    observed_correlations = statistics.calculate(physical_device_syndromes)
    fitted_dem = fit_dem_from_pij(initial_dem, observed_correlations)
    fitted_correlations = dem_pair_probabilities(fitted_dem)
    loss = statistics.difference(observed_correlations, fitted_correlations)
    if callback is not None:
        callback(1, {"loss": loss}, fitted_dem)
    return fitted_dem, [loss]


def main() -> None:
    """Collect syndrome statistics and estimate the initial decoder LER.

    Returns:
        None.
    """
    arguments = parse_arguments()
    experiment_results_directory = RESULTS_DIRECTORY / arguments.experiment
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
    statistics_calculator = DetectorPairCorrelationStatistics()
    syndrome_correlations = statistics_calculator.calculate(syndrome_samples)

    # 3. Track LER using the fixed evaluation samples collected above.
    initial_ler = logical_error_rate(initial_dem, detector_samples, actual_observables)
    ler_history: list[tuple[int, float]] = []

    progress_total = (
        OPTIMIZATION_STEPS if arguments.training_method == "cobyla" else 1
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
            loss = metrics["loss"]
            if evaluation % LER_EVALUATION_INTERVAL == 0:
                candidate_ler = logical_error_rate(
                    candidate_dem,
                    detector_samples,
                    actual_observables,
                )
                ler_history.append((evaluation, candidate_ler))
                progress.set_postfix(
                    loss=f"{loss:.3g}",
                    ler=f"{candidate_ler:.3g}",
                    refresh=False,
                )
            else:
                progress.set_postfix(loss=f"{loss:.3g}", refresh=False)
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
        else:
            optimized_dem, loss_history = train_dem_with_pair_least_squares(
                initial_dem,
                syndrome_samples,
                callback=update_progress,
            )

    # 4. Evaluate the optimized decoder on the same fixed data.
    optimized_ler = logical_error_rate(optimized_dem, detector_samples, actual_observables)
    save_loss_plot(loss_history, loss_plot_path)
    save_ler_plot(
        ler_history,
        initial_ler,
        optimized_ler,
        len(loss_history),
        ler_plot_path,
    )

    print(f"DEM error instructions = {len(error_instructions)}")
    print(f"first initial error = {error_instructions[0]}")
    print(f"syndrome samples = {syndrome_samples.shape}")
    print(f"mean pair correlation = {np.nanmean(syndrome_correlations):.6g}")
    print(f"initial RMSE = {loss_history[0]:.8g}")
    print(f"best RMSE = {min(loss_history):.8g}")
    print(f"initial LER = {initial_ler:.6g}")
    print(f"optimized LER = {optimized_ler:.6g}")
    print(f"loss plot = {loss_plot_path}")
    print(f"LER plot = {ler_plot_path}")


if __name__ == "__main__":
    main()
