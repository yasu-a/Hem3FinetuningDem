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
    """Calculate signed pair-error probabilities between detectors."""

    def calculate(self, syndromes: np.ndarray) -> np.ndarray:
        """Calculate the detector pair-correlation matrix.

        Args:
            syndromes: Detection-event samples with shape ``(shots, detectors)``.

        Returns:
            A symmetric matrix with shape ``(detectors, detectors)``. Its diagonal
            and statistically undefined entries are ``NaN``.

        Raises:
            ValueError: If syndromes is invalid or contains fewer than two detectors.
        """
        return detector_pair_correlations(syndromes)

    def difference(
        self,
        target_statistics: np.ndarray,
        model_statistics: np.ndarray,
    ) -> float:
        """Calculate RMSE between finite detector pair correlations.

        Each detector pair is included once using the upper triangle. A pair is
        compared only when its correlation estimate is finite in both matrices.

        Args:
            target_statistics: Correlation matrix from physical-device syndromes.
            model_statistics: Correlation matrix from model syndromes.

        Returns:
            Root mean squared error over valid distinct detector pairs.

        Raises:
            ValueError: If the inputs are not same-shaped square matrices or no
                detector pair has a finite estimate in both matrices.
        """
        if target_statistics.ndim != 2 or model_statistics.ndim != 2:
            raise ValueError("Correlation statistics must be two-dimensional arrays")
        if target_statistics.shape != model_statistics.shape:
            raise ValueError("Target and model statistics must have the same shape")
        if target_statistics.shape[0] != target_statistics.shape[1]:
            raise ValueError("Correlation statistics must be square matrices")

        rows, columns = np.triu_indices(target_statistics.shape[0], k=1)
        target_pairs = target_statistics[rows, columns]
        model_pairs = model_statistics[rows, columns]
        valid = np.isfinite(target_pairs) & np.isfinite(model_pairs)
        if not valid.any():
            raise ValueError("No detector pair has finite target and model statistics")
        return float(np.sqrt(np.mean((model_pairs[valid] - target_pairs[valid]) ** 2)))


class DetectorCombinedStatistics(ISyndromeStatistics):
    """Combine detector firing rates and pair correlations in one loss."""

    def __init__(self, correlation_weight: float = 1.0) -> None:
        """Initialize the combined detector statistics.

        Args:
            correlation_weight: Nonnegative weight applied to pair-correlation MSE.

        Raises:
            ValueError: If the correlation weight is negative or not finite.
        """
        if not np.isfinite(correlation_weight) or correlation_weight < 0:
            raise ValueError("Correlation weight must be finite and nonnegative")
        self._correlation_weight = float(correlation_weight)
        self._firing_rate_statistics = DetectorFiringRateStatistics()
        self._pair_correlation_statistics = DetectorPairCorrelationStatistics()

    @property
    def correlation_weight(self) -> float:
        """Return the weight applied to pair-correlation MSE."""
        return self._correlation_weight

    def calculate(
        self,
        syndromes: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Calculate detector firing rates and pair correlations.

        Args:
            syndromes: Detection-event samples with shape ``(shots, detectors)``.

        Returns:
            A tuple containing detector firing rates and the pair-correlation matrix.

        Raises:
            ValueError: If syndromes is invalid for either component statistic.
        """
        return (
            self._firing_rate_statistics.calculate(syndromes),
            self._pair_correlation_statistics.calculate(syndromes),
        )

    def difference(
        self,
        target_statistics: StatisticsValue,
        model_statistics: StatisticsValue,
    ) -> float:
        """Calculate the combined firing-rate and pair-correlation loss.

        Args:
            target_statistics: Firing rates and correlations from physical syndromes.
            model_statistics: Firing rates and correlations from model syndromes.

        Returns:
            ``sqrt(firing_rate_mse + correlation_weight * correlation_mse)``.

        Raises:
            ValueError: If either combined statistics value is invalid.
        """
        return self.metrics(target_statistics, model_statistics)["loss"]

    def metrics(
        self,
        target_statistics: StatisticsValue,
        model_statistics: StatisticsValue,
    ) -> dict[str, float]:
        """Calculate component RMSE values and the combined loss.

        Args:
            target_statistics: Firing rates and correlations from physical syndromes.
            model_statistics: Firing rates and correlations from model syndromes.

        Returns:
            Firing-rate RMSE, pair-correlation RMSE, and combined loss.

        Raises:
            ValueError: If either value is not a two-array tuple or its component
                statistics are invalid.
        """
        target_firing, target_correlations = self._split(target_statistics)
        model_firing, model_correlations = self._split(model_statistics)
        firing_rate_mse = self._firing_rate_statistics.difference(
            target_firing,
            model_firing,
        )
        pair_correlation_rmse = self._pair_correlation_statistics.difference(
            target_correlations,
            model_correlations,
        )
        loss = np.sqrt(
            firing_rate_mse
            + self._correlation_weight * pair_correlation_rmse**2
        )
        return {
            "firing_rate_rmse": float(np.sqrt(firing_rate_mse)),
            "pair_correlation_rmse": pair_correlation_rmse,
            "loss": float(loss),
        }

    @staticmethod
    def _split(statistics: StatisticsValue) -> tuple[np.ndarray, np.ndarray]:
        """Validate and unpack one combined statistics value.

        Args:
            statistics: Value returned by ``calculate``.

        Returns:
            Detector firing rates and pair correlations.

        Raises:
            ValueError: If the value is not a two-array tuple.
        """
        if (
            not isinstance(statistics, tuple)
            or len(statistics) != 2
            or not all(isinstance(value, np.ndarray) for value in statistics)
        ):
            raise ValueError("Combined statistics must be a tuple of two arrays")
        return statistics
