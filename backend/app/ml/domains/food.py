import numpy as np

from app.ml.spec import DomainSpec, Feature

FEATURES = (
    Feature("ph", "pH", "", 0.01),
    Feature("storage_temp", "Storage temperature", "°C", 0.5),
    Feature("storage_hours", "Time since milking", "h", 1),
    Feature("fat", "Fat content", "%", 0.1),
    Feature("bacterial_load", "Bacterial load", "log CFU/ml", 0.1),
    Feature("turbidity", "Turbidity", "NTU", 1),
    Feature("odour", "Odour score", "1-5", 0.5),
)

CLASSES = ("High", "Medium", "Low")


def load(seed: int, n: int = 3000):
    rng = np.random.default_rng(seed)
    storage_temp = np.clip(rng.gamma(2.2, 4.0, n), 2, 40)
    storage_hours = np.clip(rng.gamma(2.0, 10.0, n), 0, 72)
    growth = storage_hours * np.maximum(storage_temp - 4, 0.5) / 60
    bacterial_load = np.clip(3.2 + growth + rng.normal(0, 0.5, n), 2, 9)
    ph = np.clip(6.72 - 0.11 * (bacterial_load - 3.2) + rng.normal(0, 0.06, n), 6.0, 7.2)
    fat = np.clip(rng.normal(4.0, 0.9, n), 1, 7)
    turbidity = np.clip(8 + 4 * (bacterial_load - 3.2) + rng.normal(0, 5, n), 0, 60)
    odour = np.clip(np.round((5 - 0.6 * (bacterial_load - 3.2) + rng.normal(0, 0.5, n)) * 2) / 2, 1, 5)

    score = (
        -1.1 * (bacterial_load - 4.5)
        - 6.0 * np.abs(ph - 6.7)
        + 0.7 * (odour - 3.5)
        + 0.35 * (fat - 4.0)
        - 0.03 * (turbidity - 15)
        + rng.normal(0, 0.35, n)
    )
    high_cut, low_cut = np.quantile(score, [0.62, 0.30])
    y = np.where(score >= high_cut, 0, np.where(score >= low_cut, 1, 2))
    X = np.column_stack([ph, storage_temp, storage_hours, fat, bacterial_load, turbidity, odour])
    return X, y


SPEC = DomainSpec(
    id="food",
    name="Food Quality",
    sector="Food",
    task="Raw milk quality grading from lab and sensory measurements",
    description="Grades a raw milk sample as high, medium or low quality using chemical, microbial, storage and sensory measurements.",
    data_source="Synthetic dairy lab data generated from spoilage and grading rules (3,000 samples)",
    features=FEATURES,
    classes=CLASSES,
    loader=load,
)
