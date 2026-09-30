import time
import warnings

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from app.core.config import RANDOM_STATE
from app.ml.domains import DOMAINS
from app.ml.store import store


def candidates():
    return {
        "Majority class": DummyClassifier(strategy="most_frequent"),
        "Logistic regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
        "Random forest": RandomForestClassifier(
            n_estimators=150, min_samples_leaf=2, class_weight="balanced", random_state=RANDOM_STATE
        ),
    }


def model_table():
    print("| Domain | Model | Test accuracy | Test macro F1 | 5-fold CV accuracy |")
    print("| --- | --- | --- | --- | --- |")
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    for spec in DOMAINS.values():
        X, y = spec.loader(RANDOM_STATE)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, stratify=y, random_state=RANDOM_STATE
        )
        for name, model in candidates().items():
            predictions = model.fit(X_train, y_train).predict(X_test)
            cv = cross_val_score(model, X, y, cv=folds)
            print(
                f"| {spec.name} | {name} | {accuracy_score(y_test, predictions):.1%} "
                f"| {f1_score(y_test, predictions, average='macro'):.3f} | {cv.mean():.1%} ± {cv.std():.1%} |"
            )


def explainer_table(runs: int = 50):
    store.load_all()
    print("| Domain | Features | Coalitions | Median latency | p95 latency | Max additivity error |")
    print("| --- | --- | --- | --- | --- | --- |")
    for spec in DOMAINS.values():
        trained = store.get(spec.id)
        explainer = store.explainer(spec.id)
        rows = trained.test_X[:runs]
        timings, errors = [], []
        for x in rows:
            start = time.perf_counter()
            base, phi = explainer.explain(x)
            timings.append(time.perf_counter() - start)
            proba = trained.model.predict_proba(x[None, :])[0]
            errors.append(np.abs(base + phi.sum(axis=0) - proba).max())
        print(
            f"| {spec.name} | {len(spec.features)} | {2 ** len(spec.features)} "
            f"| {np.median(timings) * 1000:.0f} ms | {np.percentile(timings, 95) * 1000:.0f} ms "
            f"| {max(errors):.1e} |"
        )


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    model_table()
    print()
    explainer_table()
