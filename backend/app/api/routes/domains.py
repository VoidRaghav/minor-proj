from typing import List

import numpy as np
from fastapi import APIRouter, Depends, HTTPException

from app.api.deps import get_domain, get_store
from app.ml.domains import DOMAINS
from app.ml.spec import DomainSpec
from app.ml.store import ModelStore
from app.schemas import AnalyzeRequest, AnalyzeResponse, DomainDetail, DomainSummary, SampleResponse
from app.services.analysis import InvalidInput, analyze, domain_detail, random_sample

router = APIRouter(prefix="/domains", tags=["domains"])
rng = np.random.default_rng()


@router.get("", response_model=List[DomainSummary])
def list_domains(store: ModelStore = Depends(get_store)):
    return [
        {
            "id": spec.id,
            "name": spec.name,
            "sector": spec.sector,
            "task": spec.task,
            "description": spec.description,
            "accuracy": store.get(spec.id).metrics["accuracy"],
        }
        for spec in DOMAINS.values()
    ]


@router.get("/{domain_id}", response_model=DomainDetail)
def read_domain(spec: DomainSpec = Depends(get_domain), store: ModelStore = Depends(get_store)):
    return domain_detail(spec, store)


@router.get("/{domain_id}/sample", response_model=SampleResponse)
def read_sample(spec: DomainSpec = Depends(get_domain), store: ModelStore = Depends(get_store)):
    return random_sample(spec, store, rng)


@router.post("/{domain_id}/analyze", response_model=AnalyzeResponse)
def run_analysis(
    payload: AnalyzeRequest,
    spec: DomainSpec = Depends(get_domain),
    store: ModelStore = Depends(get_store),
):
    try:
        return analyze(spec, store, payload.values)
    except InvalidInput as error:
        raise HTTPException(status_code=422, detail=str(error))
