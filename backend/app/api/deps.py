from fastapi import HTTPException

from app.ml.domains import DOMAINS
from app.ml.spec import DomainSpec
from app.ml.store import ModelStore, store


def get_store() -> ModelStore:
    return store


def get_domain(domain_id: str) -> DomainSpec:
    spec = DOMAINS.get(domain_id)
    if spec is None:
        raise HTTPException(status_code=404, detail=f"Unknown domain '{domain_id}'")
    return spec
