from pathlib import Path
from typing import Dict

import joblib

from app.core.config import ARTIFACTS_DIR
from app.ml.domains import DOMAINS
from app.ml.explainer import ShapleyExplainer
from app.ml.trainer import TrainedModel, train


class ModelStore:
    def __init__(self, directory: Path = ARTIFACTS_DIR):
        self.directory = Path(directory)
        self.models: Dict[str, TrainedModel] = {}
        self.explainers: Dict[str, ShapleyExplainer] = {}

    def path(self, domain_id: str) -> Path:
        return self.directory / f"{domain_id}.joblib"

    def load_all(self, retrain: bool = False) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        for domain_id, spec in DOMAINS.items():
            path = self.path(domain_id)
            if retrain or not path.exists():
                trained = train(spec)
                joblib.dump(trained, path)
            else:
                trained = joblib.load(path)
            self.models[domain_id] = trained
            self.explainers[domain_id] = ShapleyExplainer(trained.model.predict_proba, trained.background)

    def get(self, domain_id: str) -> TrainedModel:
        return self.models[domain_id]

    def explainer(self, domain_id: str) -> ShapleyExplainer:
        return self.explainers[domain_id]


store = ModelStore()
