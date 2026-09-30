from typing import Dict, List

import numpy as np

from app.ml.spec import DomainSpec
from app.ml.store import ModelStore
from app.ml.trainer import decimals


class InvalidInput(ValueError):
    pass


def domain_detail(spec: DomainSpec, store: ModelStore) -> Dict:
    trained = store.get(spec.id)
    labels = {f.key: f.label for f in spec.features}
    return {
        "id": spec.id,
        "name": spec.name,
        "sector": spec.sector,
        "task": spec.task,
        "description": spec.description,
        "data_source": spec.data_source,
        "model": f"Random forest ({trained.model.n_estimators} trees)",
        "classes": list(spec.classes),
        "features": [
            {"key": f.key, "label": f.label, "unit": f.unit, "step": f.step, **trained.bounds[i]}
            for i, f in enumerate(spec.features)
        ],
        "metrics": trained.metrics,
        "importance": [{**item, "label": labels[item["key"]]} for item in trained.importance],
    }


def parse_values(spec: DomainSpec, store: ModelStore, values: Dict[str, float]) -> np.ndarray:
    missing = [k for k in spec.feature_keys if k not in values]
    if missing:
        raise InvalidInput(f"Missing values: {', '.join(missing)}")
    unknown = [k for k in values if k not in spec.feature_keys]
    if unknown:
        raise InvalidInput(f"Unknown features: {', '.join(unknown)}")
    bounds = store.get(spec.id).bounds
    out_of_range = [
        f"{f.label} must be between {bounds[i]['min']} and {bounds[i]['max']}"
        for i, f in enumerate(spec.features)
        if not bounds[i]["min"] <= values[f.key] <= bounds[i]["max"]
    ]
    if out_of_range:
        raise InvalidInput("; ".join(out_of_range))
    return np.array([values[k] for k in spec.feature_keys], dtype=float)


def format_value(value: float, step: float, unit: str) -> str:
    text = f"{value:,.{decimals(step)}f}"
    return f"{text} {unit}" if unit and not unit[0].isdigit() else text


def build_summary(label: str, confidence: float, base: float, contributions: List[Dict], steps: Dict[str, float]) -> List[str]:
    lines = [
        f"Predicted {label} with {confidence:.0%} probability, against a reference rate of {base:.0%} for {label} across the background data."
    ]
    supporting = [c for c in contributions if c["contribution"] > 0.005][:3]
    opposing = [c for c in contributions if c["contribution"] < -0.005][:2]

    def describe(items: List[Dict]) -> str:
        return ", ".join(
            f"{c['label']} = {format_value(c['value'], steps[c['key']], c['unit'])} ({c['contribution'] * 100:+.1f} pp)"
            for c in items
        )

    if supporting:
        lines.append(f"Strongest evidence for {label}: {describe(supporting)}.")
    if opposing:
        lines.append(f"Evidence against {label}: {describe(opposing)}.")
    if not supporting and not opposing:
        lines.append("No single measurement moves the prediction by more than half a percentage point.")
    return lines


def analyze(spec: DomainSpec, store: ModelStore, values: Dict[str, float]) -> Dict:
    x = parse_values(spec, store, values)
    trained = store.get(spec.id)
    probabilities = trained.model.predict_proba(x[None, :])[0]
    predicted = int(np.argmax(probabilities))

    base, phi = store.explainer(spec.id).explain(x)
    contributions = sorted(
        (
            {"key": f.key, "label": f.label, "unit": f.unit, "value": float(x[i]), "contribution": float(phi[i, predicted])}
            for i, f in enumerate(spec.features)
        ),
        key=lambda c: abs(c["contribution"]),
        reverse=True,
    )
    label = spec.classes[predicted]
    confidence = float(probabilities[predicted])
    steps = {f.key: f.step for f in spec.features}

    return {
        "domain": spec.id,
        "prediction": label,
        "confidence": confidence,
        "base_value": float(base[predicted]),
        "probabilities": [
            {"label": name, "probability": float(p)} for name, p in zip(spec.classes, probabilities)
        ],
        "contributions": contributions,
        "summary": build_summary(label, confidence, float(base[predicted]), contributions, steps),
    }


def random_sample(spec: DomainSpec, store: ModelStore, rng: np.random.Generator) -> Dict:
    trained = store.get(spec.id)
    index = int(rng.integers(len(trained.test_y)))
    row = trained.test_X[index]
    return {
        "values": {
            f.key: round(float(row[i]), decimals(f.step)) for i, f in enumerate(spec.features)
        },
        "label": spec.classes[int(trained.test_y[index])],
    }

