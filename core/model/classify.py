from dataclasses import dataclass, field
import numpy as np

@dataclass
class LogisticRegression:
    _mean: np.ndarray = field(default_factory=lambda: np.array([]), init=False)
    _std: np.ndarray = field(default_factory=lambda: np.array([]), init=False)
    _is_fitted: bool = field(default=False, init=False)

    def __post_init__(self):
        pass


    def fit_normalize(self, X):
        """Learn normalization stats from training data ONCE"""
        self._mean = np.mean(X, axis=0)
        self._std = np.std(X, axis=0)
        self._std = np.where(self._std == 0, 1, self._std)
        self._is_fitted = True
        return (X - self._mean) / self._std

    def transform(self, X):
        """Apply SAME normalization to any data"""
        if not self._is_fitted:
            raise ValueError("Must call fit_normalize first!")
        return (X - self._mean) / self._std




    def fit():
        pass

    def predict():
        pass

    def save_model():
        pass

