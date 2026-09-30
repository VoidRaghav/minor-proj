import numpy as np
from sklearn.datasets import load_breast_cancer

from app.ml.spec import DomainSpec, Feature

FEATURES = (
    Feature("mean radius", "Mean radius", "", 0.1),
    Feature("mean texture", "Mean texture", "", 0.1),
    Feature("mean smoothness", "Mean smoothness", "", 0.001),
    Feature("mean compactness", "Mean compactness", "", 0.001),
    Feature("mean concavity", "Mean concavity", "", 0.001),
    Feature("mean concave points", "Mean concave points", "", 0.001),
    Feature("mean symmetry", "Mean symmetry", "", 0.001),
    Feature("worst area", "Worst area", "", 1),
)


def load(seed: int):
    data = load_breast_cancer()
    names = list(data.feature_names)
    columns = [names.index(f.key) for f in FEATURES]
    return data.data[:, columns].astype(float), data.target.astype(int)


SPEC = DomainSpec(
    id="disease",
    name="Disease Diagnosis",
    sector="Healthcare",
    task="Breast tumour classification from cell nucleus measurements",
    description="Classifies a fine needle aspirate sample as malignant or benign using eight nucleus shape and size measurements.",
    data_source="Wisconsin Diagnostic Breast Cancer dataset (569 samples, UCI)",
    features=FEATURES,
    classes=("Malignant", "Benign"),
    loader=load,
)
