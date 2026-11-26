"""Syndrome statistics interfaces and implementations."""

from abc import ABC, abstractmethod

import numpy as np

from .correlations import detector_pair_correlations


StatisticsValue = np.ndarray | tuple[np.ndarray, np.ndarray]


class ISyndromeStatistics(ABC):
    """Interface for converting syndrome samples into summary statistics."""

    @abstractmethod
    def calculate(self, syndromes: np.ndarray) -> StatisticsValue:
        """Calculate statistics from syndrome samples.

        Args:
            syndromes: Detection-event samples with shape ``(shots, detectors)``.

        Returns:
            The calculated statistics.

        Raises:
            NotImplementedError: Always raised by the interface implementation.
        """
        raise NotImplementedError()

    @abstractmethod
    def difference(
        self,
        target_statistics: StatisticsValue,
        model_statistics: StatisticsValue,
    ) -> float:
        """Calculate the difference between target and model statistics.

        Args:
            target_statistics: Statistics calculated from physical-device syndromes.
            model_statistics: Statistics calculated from model syndromes.

        Returns:
            A scalar difference between the two statistics.

        Raises:
            NotImplementedError: Always raised by the interface implementation.
        """
        raise NotImplementedError()

    def metrics(
        self,
        target_statistics: StatisticsValue,
        model_statistics: StatisticsValue,
    ) -> dict[str, float]:
        """Calculate named loss metrics between two statistics values.

        Args:
            target_statistics: Statistics calculated from physical-device syndromes.
            model_statistics: Statistics calculated from model syndromes.

        Returns:
            A mapping containing the scalar optimization loss.
        """
        return {"loss": self.difference(target_statistics, model_statistics)}


class DetectorFiringRateStatistics(ISyndromeStatistics):
    """Calculate the empirical firing rate of each detector."""

    def calculate(self, syndromes: np.ndarray) -> np.ndarray:
        """Calculate the mean firing rate of each detector.

        Args:
            syndromes: Detection-event samples with shape ``(shots, detectors)``.

        Returns:
            The mean firing rate of each detector with shape ``(detectors,)``.

        Raises:
            ValueError: If syndromes is not a non-empty two-dimensional array.
        """
        if syndromes.ndim != 2 or syndromes.shape[0] == 0:
            raise ValueError("Syndromes must be a non-empty two-dimensional array")
        return syndromes.mean(axis=0)

    def difference(
        self,
        target_statistics: np.ndarray,
        model_statistics: np.ndarray,
    ) -> float:
        """Calculate the mean squared error between detector firing rates.

        Args:
            target_statistics: Detector firing rates from physical-device syndromes.
            model_statistics: Detector firing rates from model syndromes.

        Returns:
            The mean squared error between the detector firing rates.

        Raises:
            ValueError: If either input is not one-dimensional or their shapes differ.
        """
        if target_statistics.ndim != 1 or model_statistics.ndim != 1:
            raise ValueError("Statistics must be one-dimensional arrays")
        if target_statistics.shape != model_statistics.shape:
            raise ValueError("Target and model statistics must have the same shape")
        return float(np.mean((model_statistics - target_statistics) ** 2))


class DetectorPairCorrelationStatistics(ISyndromeStatistics):
    """Calculate detector firing rates and signed pair-error probabilities."""

    def __init__(self, pair_loss_weight: float = 1.0) -> None:
        """Initialize the unified detector-correlation statistics.

        Args:
            pair_loss_weight: Nonnegative weight applied to off-diagonal pair MSE.

        Raises:
            ValueError: If the pair loss weight is negative or not finite.
        """
        if not np.isfinite(pair_loss_weight) or pair_loss_weight < 0:
            raise ValueError("Pair loss weight must be finite and nonnegative")
        self._pair_loss_weight = float(pair_loss_weight)

    @property
    def pair_loss_weight(self) -> float:
        """Return the weight applied to off-diagonal pair MSE."""
        return self._pair_loss_weight

    def calculate(self, syndromes: np.ndarray) -> np.ndarray:
        """Calculate the detector pair-correlation matrix.

        Args:
            syndromes: Detection-event samples with shape ``(shots, detectors)``.

        Returns:
            A symmetric matrix with shape ``(detectors, detectors)``. Its diagonal
            contains detector firing probabilities, and statistically undefined
            off-diagonal entries are ``NaN``.

        Raises:
            ValueError: If syndromes is invalid or contains fewer than two detectors.
        """
        return detector_pair_correlations(syndromes)

    def difference(
        self,
        target_statistics: np.ndarray,
        model_statistics: np.ndarray,
    ) -> float:
        """Calculate combined RMSE over detector firing and pair statistics.

        Each detector pair is included once using the upper triangle. A pair is
        compared only when its correlation estimate is finite in both matrices.

        Args:
            target_statistics: Correlation matrix from physical-device syndromes.
            model_statistics: Correlation matrix from model syndromes.

        Returns:
            ``sqrt(single_mse + pair_loss_weight * pair_mse)``.

        Raises:
            ValueError: If the inputs are not same-shaped square matrices or no
                detector pair has a finite estimate in both matrices.
        """
        return self.metrics(target_statistics, model_statistics)["loss"]

    def metrics(
        self,
        target_statistics: np.ndarray,
        model_statistics: np.ndarray,
    ) -> dict[str, float]:
        """Calculate separately normalized single and pair RMSE values.

        Args:
            target_statistics: Unified correlation matrix from physical syndromes.
            model_statistics: Unified correlation matrix from model syndromes.

        Returns:
            Single-detector RMSE, pair-correlation RMSE, combined RMSE, and loss.

        Raises:
            ValueError: If the matrices are invalid or either component has no jointly
                finite entry.
        """
        return correlation_matrix_metrics(
            target_statistics,
            model_statistics,
            self._pair_loss_weight,
        )


class DetectorCombinedStatistics(DetectorPairCorrelationStatistics):
    """Backward-compatible name for unified detector-correlation statistics."""

    def __init__(self, correlation_weight: float = 1.0) -> None:
        """Initialize the combined detector statistics.

        Args:
            correlation_weight: Nonnegative weight applied to pair-correlation MSE.

        Raises:
            ValueError: If the correlation weight is negative or not finite.
        """
        if not np.isfinite(correlation_weight) or correlation_weight < 0:
            raise ValueError("Correlation weight must be finite and nonnegative")
        super().__init__(pair_loss_weight=correlation_weight)

    @property
    def correlation_weight(self) -> float:
        """Return the weight applied to pair-correlation MSE."""
        return self.pair_loss_weight


def correlation_matrix_metrics(
    target_statistics: np.ndarray,
    model_statistics: np.ndarray,
    pair_loss_weight: float,
) -> dict[str, float]:
    """Calculate unified loss metrics from two detector-correlation matrices.

    Args:
        target_statistics: Matrix with firing rates on its diagonal.
        model_statistics: Matrix with firing rates on its diagonal.
        pair_loss_weight: Nonnegative multiplier for off-diagonal pair MSE.

    Returns:
        Single-detector RMSE, pair-correlation RMSE, combined RMSE, and loss.

    Raises:
        ValueError: If the matrices, weight, or finite component sets are invalid.
    """
    if not np.isfinite(pair_loss_weight) or pair_loss_weight < 0:
        raise ValueError("Pair loss weight must be finite and nonnegative")
    if target_statistics.ndim != 2 or model_statistics.ndim != 2:
        raise ValueError("Correlation statistics must be two-dimensional arrays")
    if target_statistics.shape != model_statistics.shape:
        raise ValueError("Target and model statistics must have the same shape")
    if target_statistics.shape[0] != target_statistics.shape[1]:
        raise ValueError("Correlation statistics must be square matrices")

    target_single = np.diag(target_statistics)
    model_single = np.diag(model_statistics)
    valid_single = np.isfinite(target_single) & np.isfinite(model_single)
    if not valid_single.any():
        raise ValueError("No detector has finite target and model firing rates")
    single_mse = np.mean(
        (model_single[valid_single] - target_single[valid_single]) ** 2
    )

    rows, columns = np.triu_indices(target_statistics.shape[0], k=1)
    target_pairs = target_statistics[rows, columns]
    model_pairs = model_statistics[rows, columns]
    valid_pairs = np.isfinite(target_pairs) & np.isfinite(model_pairs)
    if not valid_pairs.any():
        raise ValueError("No detector pair has finite target and model statistics")
    pair_mse = np.mean(
        (model_pairs[valid_pairs] - target_pairs[valid_pairs]) ** 2
    )
    rmse = float(np.sqrt(single_mse + pair_loss_weight * pair_mse))
    return {
        "single_rmse": float(np.sqrt(single_mse)),
        "pair_rmse": float(np.sqrt(pair_mse)),
        "rmse": rmse,
        "loss": rmse,
    }
