from fastapi import APIRouter

from .agents import router as agents_router
from .analysis import router as analysis_router
from .datasets import router as datasets_router

router = APIRouter()

router.include_router(datasets_router, prefix="/datasets", tags=["Datasets"])
router.include_router(analysis_router, prefix="/analysis", tags=["Analysis"])
router.include_router(agents_router, prefix="/agents", tags=["Agents"])
