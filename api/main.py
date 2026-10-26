from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, field_validator

from api.model_service import TranslationError, TranslationService

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FRONTEND_DIR = Path(__file__).parent / "frontend"
CHECKPOINT_PATH = PROJECT_ROOT / "checkpoints" / "best_transformer.pt"
SP_MODEL_PATH = PROJECT_ROOT / "tokenizer" / "translator_sp.model"

SERVICE: TranslationService | None = None


class TranslateRequest(BaseModel):
    text: str

    @field_validator("text")
    @classmethod
    def text_must_not_be_blank(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("Text field must not be empty.")
        return value


class TranslateResponse(BaseModel):
    translation: str
    source_text: str


@asynccontextmanager
async def lifespan(app: FastAPI) -> Any:
    global SERVICE

    try:
        SERVICE = TranslationService(
            checkpoint_path=CHECKPOINT_PATH,
            sp_model_path=SP_MODEL_PATH,
            device="auto",
        )
        logger.info("Translation model loaded successfully.")
    except Exception:
        # Keep the API running so /health can report whether the model loaded.
        logger.exception("Failed to load translation model at startup.")
        SERVICE = None

    yield
    SERVICE = None


app = FastAPI(
    title="Amharic-to-Afaan-Oromo Translation API",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", include_in_schema=False)
async def home() -> FileResponse:
    return FileResponse(FRONTEND_DIR / "index.html")


@app.get("/prof.png", include_in_schema=False)
async def profile_image() -> FileResponse:
    return FileResponse(FRONTEND_DIR / "prof.png")


@app.get("/health")
def health_check() -> dict[str, bool]:
    return {"status": True, "model_loaded": SERVICE is not None}


@app.post("/translate", response_model=TranslateResponse)
def translate(request: TranslateRequest) -> TranslateResponse:
    if SERVICE is None:
        raise HTTPException(
            status_code=503,
            detail="Translation service not ready. Check server logs.",
        )

    try:
        translation = SERVICE.translate(request.text)
    except TranslationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Unexpected error during translation.")
        raise HTTPException(
            status_code=500,
            detail="Internal error occurred during translation.",
        ) from exc

    return TranslateResponse(
        translation=translation,
        source_text=request.text,
    )
