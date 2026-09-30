from math import factorial
from typing import Callable, Tuple

import numpy as np


class ShapleyExplainer:
    def __init__(self, predict_proba: Callable[[np.ndarray], np.ndarray], background: np.ndarray):
        self.predict_proba = predict_proba
        self.background = np.asarray(background, dtype=float)
        n = self.background.shape[1]
        self.n_features = n
        coalitions = np.arange(2**n)
        self.masks = ((coalitions[:, None] >> np.arange(n)) & 1).astype(bool)
        sizes = self.masks.sum(axis=1)
        self.weights = np.array(
            [factorial(s) * factorial(n - s - 1) / factorial(n) if s < n else 0.0 for s in sizes]
        )

    def coalition_values(self, x: np.ndarray) -> np.ndarray:
        rows = np.where(self.masks[:, None, :], x[None, None, :], self.background[None, :, :])
        proba = self.predict_proba(rows.reshape(-1, self.n_features))
        return proba.reshape(len(self.masks), len(self.background), -1).mean(axis=1)

    def explain(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        x = np.asarray(x, dtype=float)
        values = self.coalition_values(x)
        phi = np.zeros((self.n_features, values.shape[1]))
        for i in range(self.n_features):
            without = np.flatnonzero(~self.masks[:, i])
            with_i = without | (1 << i)
            phi[i] = (self.weights[without, None] * (values[with_i] - values[without])).sum(axis=0)
        return values[0], phi
