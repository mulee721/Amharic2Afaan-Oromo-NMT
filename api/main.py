from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, field_validator

<<<<<<< HEAD
from model_service import TranslationError, TranslationService
=======
from api.model_service import TranslationError, TranslationService
>>>>>>> 51da30f (Prepare NMT project for deployment)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

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
        # Don't crash the whole process if the model fails to load --
        # start the app anyway so /health reports the real status instead
        # of the server refusing to boot at all.
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
        # Expected, user-facing failure (empty text, bad tokenization, etc.)
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception:
        # Unexpected failure -- log full details server-side, but never
        # leak internals to the client.
        logger.exception("Unexpected error during translation.")
        raise HTTPException(
            status_code=500,
            detail="Internal error occurred during translation.",
        )

    return TranslateResponse(
        translation=translation,
        source_text=request.text,
    )