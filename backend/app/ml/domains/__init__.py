from typing import Dict

from app.ml.domains import crop, disease, food
from app.ml.spec import DomainSpec

DOMAINS: Dict[str, DomainSpec] = {spec.id: spec for spec in (disease.SPEC, crop.SPEC, food.SPEC)}
