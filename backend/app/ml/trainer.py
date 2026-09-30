import math
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split

from app.core.config import BACKGROUND_SIZE, RANDOM_STATE
from app.ml.spec import DomainSpec


@dataclass
class TrainedModel:
    spec_id: str
    model: RandomForestClassifier
    background: np.ndarray
    bounds: List[Dict[str, float]]
    metrics: Dict[str, object]
    importance: List[Dict[str, float]]
    test_X: np.ndarray
    test_y: np.ndarray


def decimals(step: float) -> int:
    return max(0, -int(math.floor(math.log10(step)))) if step < 1 else 0


def feature_bounds(spec: DomainSpec, X: np.ndarray) -> List[Dict[str, float]]:
    bounds = []
    for i, feature in enumerate(spec.features):
        step = feature.step
        column = X[:, i]
        low = math.floor(column.min() / step) * step
        high = math.ceil(column.max() / step) * step
        default = round(np.median(column) / step) * step
        places = decimals(step)
        bounds.append({"min": round(low, places), "max": round(high, places), "default": round(default, places)})
    return bounds


def train(spec: DomainSpec) -> TrainedModel:
    X, y = spec.loader(RANDOM_STATE)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=RANDOM_STATE
    )
    model = RandomForestClassifier(
        n_estimators=150, min_samples_leaf=2, class_weight="balanced", random_state=RANDOM_STATE
    )
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    metrics = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "macro_f1": float(f1_score(y_test, predictions, average="macro")),
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "confusion_matrix": confusion_matrix(y_test, predictions, labels=range(len(spec.classes))).tolist(),
    }

    result = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=RANDOM_STATE)
    importance = sorted(
        (
            {"key": f.key, "mean": float(result.importances_mean[i]), "std": float(result.importances_std[i])}
            for i, f in enumerate(spec.features)
        ),
        key=lambda item: item["mean"],
        reverse=True,
    )

    rng = np.random.default_rng(RANDOM_STATE)
    background = X_train[rng.choice(len(X_train), size=min(BACKGROUND_SIZE, len(X_train)), replace=False)]

    return TrainedModel(
        spec_id=spec.id,
        model=model,
        background=background,
        bounds=feature_bounds(spec, X),
        metrics=metrics,
        importance=importance,
        test_X=X_test,
        test_y=y_test,
    )
