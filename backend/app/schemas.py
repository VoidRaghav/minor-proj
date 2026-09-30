from typing import Dict, List

from pydantic import BaseModel


class FeatureOut(BaseModel):
    key: str
    label: str
    unit: str
    step: float
    min: float
    max: float
    default: float


class ImportanceOut(BaseModel):
    key: str
    label: str
    mean: float
    std: float


class MetricsOut(BaseModel):
    accuracy: float
    macro_f1: float
    n_train: int
    n_test: int
    confusion_matrix: List[List[int]]


class DomainSummary(BaseModel):
    id: str
    name: str
    sector: str
    task: str
    description: str
    accuracy: float


class DomainDetail(BaseModel):
    id: str
    name: str
    sector: str
    task: str
    description: str
    data_source: str
    model: str
    classes: List[str]
    features: List[FeatureOut]
    metrics: MetricsOut
    importance: List[ImportanceOut]


class AnalyzeRequest(BaseModel):
    values: Dict[str, float]


class ClassProbability(BaseModel):
    label: str
    probability: float


class Contribution(BaseModel):
    key: str
    label: str
    unit: str
    value: float
    contribution: float


class AnalyzeResponse(BaseModel):
    domain: str
    prediction: str
    confidence: float
    base_value: float
    probabilities: List[ClassProbability]
    contributions: List[Contribution]
    summary: List[str]


class SampleResponse(BaseModel):
    values: Dict[str, float]
    label: str
