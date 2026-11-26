
"""Syndrome statistics interfaces and implementations."""

from abc import ABC, abstractmethod

import numpy as np


class ISyndromeStatistics(ABC):
    """Interface for converting syndrome samples into summary statistics."""

    @abstractmethod
    def calculate(self, syndromes: np.ndarray) -> np.ndarray:
        """Calculate statistics from syndrome samples.

        Args:
            syndromes: Detection-event samples with shape ``(shots, detectors)``.

        Returns:
            A one-dimensional array containing the calculated statistics.

        Raises:
            NotImplementedError: Always raised by the interface implementation.
        """
        raise NotImplementedError()

    @abstractmethod
    def difference(
        self,
        target_statistics: np.ndarray,
        model_statistics: np.ndarray,
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
