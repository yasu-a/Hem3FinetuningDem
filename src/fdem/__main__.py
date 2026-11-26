"""Minimal example for estimating a rotated surface code LER using a Stim DEM."""

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pymatching
import stim
from scipy.optimize import minimize
from tqdm import tqdm

from .device import (
    build_physical_device_circuit,
    sample_physical_device_syndromes,
)
from .plot import save_ler_plot, save_loss_plot
from .statistics import DetectorFiringRateStatistics, ISyndromeStatistics


# Experiment settings. Change these values to try different configurations.
DISTANCE = 3
ROUNDS = 3
PHYSICAL_ERROR_RATE = 0.001
SHOTS = 100_000
MODEL_SHOTS = 10_000
OPTIMIZATION_STEPS = 300
LEARNING_RATE = 0.0001
LER_EVALUATION_INTERVAL = 10
SEED = 0
LOSS_PLOT_PATH = Path("var/loss.png")
LER_PLOT_PATH = Path("var/ler.png")

physical_device_circuit = build_physical_device_circuit(
    DISTANCE,
    ROUNDS,
    PHYSICAL_ERROR_RATE,
)

def build_naive_dem() -> stim.DetectorErrorModel:
    """Build a flattened naive DEM from the physical device circuit model.

    Returns:
        The circuit-derived detector error model with repeat blocks expanded.
    """
    return physical_device_circuit.detector_error_model(
        decompose_errors=True
    ).flattened()


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
    callback: Callable[[int, float, stim.DetectorErrorModel], None] | None = None,
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
            the evaluation number, current loss, and candidate DEM.

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
        loss = syndrome_statistics.difference(target_statistics, model_statistics)
        loss_history.append(loss)
        if callback is not None:
            callback(len(loss_history), loss, model_dem)
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
    # 1. Use the circuit-derived naive DEM directly as the initial DEM.
    naive_dem = build_naive_dem()
    initial_dem = naive_dem
    error_instructions = [
        instruction for instruction in initial_dem if instruction.type == "error"
    ]

    # 2. Collect syndrome data for the future optimization step.
    syndrome_samples = sample_physical_device_syndromes(
        physical_device_circuit,
        SHOTS,
        SEED,
    )
    statistics_calculator = DetectorFiringRateStatistics()
    syndrome_rates = statistics_calculator.calculate(syndrome_samples)

    # 3. Generate fixed evaluation data for tracking LER during optimization.
    evaluation_sampler = physical_device_circuit.compile_detector_sampler(seed=SEED + 2)
    detector_samples, actual_observables = evaluation_sampler.sample(
        shots=SHOTS,
        separate_observables=True,
    )
    initial_ler = logical_error_rate(initial_dem, detector_samples, actual_observables)
    ler_history: list[tuple[int, float]] = []

    with tqdm(total=OPTIMIZATION_STEPS, desc="Optimizing DEM", unit="eval") as progress:

        def update_progress(
            evaluation: int,
            loss: float,
            candidate_dem: stim.DetectorErrorModel,
        ) -> None:
            """Update progress and periodically record the candidate LER.

            Args:
                evaluation: Number of completed objective-function evaluations.
                loss: Loss from the latest evaluation.
                candidate_dem: DEM evaluated by the current objective-function call.

            Returns:
                None.
            """
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
    save_loss_plot(loss_history, LOSS_PLOT_PATH)
    save_ler_plot(
        ler_history,
        initial_ler,
        optimized_ler,
        len(loss_history),
        LER_PLOT_PATH,
    )

    print(f"distance = {DISTANCE}")
    print(f"physical error rate = {PHYSICAL_ERROR_RATE}")
    print(f"DEM error instructions = {len(error_instructions)}")
    print(f"first initial error = {error_instructions[0]}")
    print(f"syndrome samples = {syndrome_samples.shape}")
    print(f"mean detector rate = {syndrome_rates.mean():.6g}")
    print(f"initial loss = {loss_history[0]:.8g}")
    print(f"best loss = {min(loss_history):.8g}")
    print(f"initial LER = {initial_ler:.6g}")
    print(f"optimized LER = {optimized_ler:.6g}")
    print(f"loss plot = {LOSS_PLOT_PATH}")
    print(f"LER plot = {LER_PLOT_PATH}")


if __name__ == "__main__":
    main()
