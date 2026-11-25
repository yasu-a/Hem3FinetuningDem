"""Minimal example for estimating a rotated surface code LER using a Stim DEM."""

import pymatching
import stim


# Experiment settings. Change these values to try different configurations.
DISTANCE = 3
ROUNDS = 3
PHYSICAL_ERROR_RATE = 0.001
SHOTS = 100_000


def main() -> None:
    # 1. Generate a noisy rotated surface code circuit.
    circuit = stim.Circuit.generated(
        "surface_code:rotated_memory_x",
        distance=DISTANCE,
        rounds=ROUNDS,
        after_clifford_depolarization=PHYSICAL_ERROR_RATE,
        after_reset_flip_probability=PHYSICAL_ERROR_RATE,
        before_measure_flip_probability=PHYSICAL_ERROR_RATE,
    )

    # 2. Convert the circuit into a detector error model (DEM).
    dem = circuit.detector_error_model(decompose_errors=True)
    print(dem)

    # 3. Use the same DEM for sampling and building the MWPM decoder.
    detector_samples, actual_observables, _ = dem.compile_sampler().sample(SHOTS)
    decoder = pymatching.Matching.from_detector_error_model(dem)
    predicted_observables = decoder.decode_batch(detector_samples)

    # 4. Count shots where the prediction differs from the actual logical observable.
    logical_errors = (predicted_observables != actual_observables).any(axis=1).sum()
    ler = logical_errors / SHOTS

    print(f"distance = {DISTANCE}")
    print(f"physical error rate = {PHYSICAL_ERROR_RATE}")
    print(f"logical errors = {logical_errors} / {SHOTS}")
    print(f"LER = {ler:.6g}")


if __name__ == "__main__":
    main()
