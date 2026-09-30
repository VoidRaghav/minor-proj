from fastapi import APIRouter, Depends

from app.api.deps import get_store
from app.ml.store import ModelStore

router = APIRouter(tags=["health"])


@router.get("/health")
def health(store: ModelStore = Depends(get_store)):
    return {"status": "ok", "models": sorted(store.models)}
