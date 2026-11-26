"""Helpers for reading and updating Stim detector error mechanisms."""

import numpy as np
import stim


def detector_support(instruction: stim.DemInstruction) -> tuple[int, ...]:
    """Return the detectors flipped by the whole mechanism.

    Args:
        instruction: An error instruction from a flattened DEM.

    Returns:
        The sorted detector IDs appearing an odd number of times across all components.
    """
    detectors = set()
    for target in instruction.targets_copy():
        if target.is_relative_detector_id():
            detectors.symmetric_difference_update((target.val,))
    return tuple(sorted(detectors))


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
