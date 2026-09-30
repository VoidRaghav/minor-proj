from dataclasses import dataclass
from typing import Callable, Tuple

import numpy as np


@dataclass(frozen=True)
class Feature:
    key: str
    label: str
    unit: str
    step: float


@dataclass(frozen=True)
class DomainSpec:
    id: str
    name: str
    sector: str
    task: str
    description: str
    data_source: str
    features: Tuple[Feature, ...]
    classes: Tuple[str, ...]
    loader: Callable[[int], Tuple[np.ndarray, np.ndarray]]

    @property
    def feature_keys(self) -> Tuple[str, ...]:
        return tuple(f.key for f in self.features)
