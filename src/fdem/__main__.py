"""Minimal example for estimating a rotated surface code LER using a Stim DEM."""

import argparse
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pymatching
import stim
from scipy.optimize import minimize
from tqdm import tqdm

from .device import sample_physical_device_syndromes
from .plot import save_ler_plot, save_loss_plot
from .statistics import DetectorCombinedStatistics, ISyndromeStatistics


# Optimization settings. Change these values to try different configurations.
SHOTS = 100_000
MODEL_SHOTS = 10_000
OPTIMIZATION_STEPS = 300
LEARNING_RATE = 0.0001
CORRELATION_WEIGHT = 1.0
INITIAL_PRIOR = 0.001
LER_EVALUATION_INTERVAL = 10
SEED = 0
RESULTS_DIRECTORY = Path("var")


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
    return parser.parse_args()

def build_candidate_dem(
    detector_count: int,
    observable_count: int,
    initial_prior: float,
) -> stim.DetectorErrorModel:
    """Build a generic graphlike DEM without inspecting the physical-device circuit.

    The candidate structure contains one error mechanism for each detector and one for
    every unordered detector pair. Logical targets are assigned using fixed cuts in
    detector-index order. These cuts make the model decodable but are only an ansatz;
    syndrome statistics cannot determine logical-observable targets.

    Args:
        detector_count: Number of detectors in the sampled syndrome data.
        observable_count: Number of logical observables in the evaluation data.
        initial_prior: Initial probability assigned to every candidate mechanism.

    Returns:
        A graphlike detector error model with fixed single-detector and detector-pair
        mechanisms.

    Raises:
        ValueError: If fewer than two detectors or one observable are requested, or if
            the initial prior is not a finite probability below 0.5.
    """
    if detector_count < 2:
        raise ValueError("Candidate DEM requires at least two detectors")
    if observable_count < 1:
        raise ValueError("Candidate DEM requires at least one observable")
    if not np.isfinite(initial_prior) or not 0 < initial_prior < 0.5:
        raise ValueError("Initial prior must be finite and between 0 and 0.5")

    logical_cuts = [
        (observable + 1) * detector_count // (observable_count + 1)
        for observable in range(observable_count)
    ]
    dem = stim.DetectorErrorModel()

    def logical_targets(left: int, right: int | None = None) -> list[stim.DemTarget]:
        """Return logical targets crossed by a candidate error mechanism.

        Args:
            left: Index of the first detector endpoint.
            right: Index of the second endpoint, or the implicit boundary for a
                single-detector mechanism.

        Returns:
            Logical-observable targets crossed by the mechanism.
        """
        return [
            stim.target_logical_observable_id(observable)
            for observable, cut in enumerate(logical_cuts)
            if (
                (left >= cut)
                if right is None
                else ((left >= cut) != (right >= cut))
            )
        ]

    for detector in range(detector_count):
        targets = [stim.target_relative_detector_id(detector)]
        targets.extend(logical_targets(detector))
        dem.append("error", initial_prior, targets, tag="candidate_single")

    for left in range(detector_count):
        for right in range(left + 1, detector_count):
            targets = [
                stim.target_relative_detector_id(left),
                stim.target_relative_detector_id(right),
            ]
            targets.extend(logical_targets(left, right))
            dem.append("error", initial_prior, targets, tag="candidate_pair")
    return dem


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


def with_priors(
    dem: stim.DetectorErrorModel,
    priors: np.ndarray,
) -> stim.DetectorErrorModel:
    """Replace the error probabilities in a detector error model.

    Args:
        dem: Flattened detector error model whose structure will be preserved.
        priors: Replacement probabilities, one for each ``error`` instruction.

    Returns:
        A new detector error model with replaced probabilities and unchanged targets.

    Raises:
        ValueError: If the number of priors does not match the number of error
            instructions.
    """
    error_count = sum(instruction.type == "error" for instruction in dem)
    if priors.shape != (error_count,):
        raise ValueError(f"Expected {error_count} priors, got shape {priors.shape}")

    result = stim.DetectorErrorModel()
    prior_index = 0
    for instruction in dem:
        if instruction.type == "error":
            result.append(
                "error",
                float(priors[prior_index]),
                instruction.targets_copy(),
                tag=instruction.tag,
            )
            prior_index += 1
        else:
            result.append(instruction)
    return result


def train_dem(
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

    # 2. Build a generic candidate DEM from sample dimensions, not from the device.
    initial_dem = build_candidate_dem(
        syndrome_samples.shape[1],
        actual_observables.shape[1],
        INITIAL_PRIOR,
    )
    error_instructions = [
        instruction for instruction in initial_dem if instruction.type == "error"
    ]
    statistics_calculator = DetectorCombinedStatistics(CORRELATION_WEIGHT)
    syndrome_firing_rates, syndrome_correlations = statistics_calculator.calculate(
        syndrome_samples
    )

    # 3. Track LER using the fixed evaluation samples collected above.
    initial_ler = logical_error_rate(initial_dem, detector_samples, actual_observables)
    ler_history: list[tuple[int, float]] = []
    firing_rate_rmse_history: list[float] = []
    pair_correlation_rmse_history: list[float] = []

    with tqdm(total=OPTIMIZATION_STEPS, desc="Optimizing DEM", unit="eval") as progress:

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
            firing_rate_rmse_history.append(metrics["firing_rate_rmse"])
            pair_correlation_rmse_history.append(
                metrics["pair_correlation_rmse"]
            )
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

        optimized_dem, loss_history = train_dem(
            initial_dem,
            syndrome_samples,
            statistics_calculator,
            steps=OPTIMIZATION_STEPS,
            model_shots=MODEL_SHOTS,
            learning_rate=LEARNING_RATE,
            seed=SEED + 1,
            callback=update_progress,
        )

    # 4. Evaluate the optimized decoder on the same fixed data.
    optimized_ler = logical_error_rate(optimized_dem, detector_samples, actual_observables)
    save_loss_plot(
        firing_rate_rmse_history,
        pair_correlation_rmse_history,
        loss_history,
        loss_plot_path,
    )
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
    print(f"mean detector firing rate = {syndrome_firing_rates.mean():.6g}")
    print(f"mean pair correlation = {np.nanmean(syndrome_correlations):.6g}")
    best_evaluation = int(np.argmin(loss_history))
    print(f"correlation weight = {CORRELATION_WEIGHT:.6g}")
    print(f"initial firing-rate RMSE = {firing_rate_rmse_history[0]:.8g}")
    print(f"initial pair-correlation RMSE = {pair_correlation_rmse_history[0]:.8g}")
    print(f"initial combined loss = {loss_history[0]:.8g}")
    print(
        "firing-rate RMSE at best loss = "
        f"{firing_rate_rmse_history[best_evaluation]:.8g}"
    )
    print(
        "pair-correlation RMSE at best loss = "
        f"{pair_correlation_rmse_history[best_evaluation]:.8g}"
    )
    print(f"best combined loss = {loss_history[best_evaluation]:.8g}")
    print(f"initial LER = {initial_ler:.6g}")
    print(f"optimized LER = {optimized_ler:.6g}")
    print(f"loss plot = {loss_plot_path}")
    print(f"LER plot = {ler_plot_path}")


if __name__ == "__main__":
    main()
