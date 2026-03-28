from __future__ import annotations

from contextlib import asynccontextmanager
import logging
import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.routes import router as inference_router
from src.core.inference import get_inference_service
from src.utils.prediction_logging import initialize_prediction_log_db

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_: FastAPI):
    """Preload models on startup to avoid first-request download latency."""
    logger.info("Preloading inference service...")
    get_inference_service()
    db_path = initialize_prediction_log_db()
    logger.info("Prediction logs will be stored at %s", db_path)
    logger.info("Inference service ready.")
    yield


app = FastAPI(
    title="Face Age Estimator API",
    version="0.1.0",
    lifespan=lifespan,
    description=(
        "Inference API for face preprocessing, gender prediction, "
        "gender-agnostic age prediction, and gender-specific age prediction."
    ),
)

cors_origins = os.getenv("CORS_ALLOW_ORIGINS", "*")
if cors_origins.strip() == "*":
    allow_origins = ["*"]
else:
    allow_origins = [
        origin.strip() for origin in cors_origins.split(",") if origin.strip()
    ]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(inference_router)


@app.get("/health")
def health() -> dict[str, str]:
    """Simple health check endpoint."""
    return {"status": "ok"}
