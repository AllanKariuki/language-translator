"""FastAPI service for translation and feedback.

Usage:
    uvicorn api:app --host 0.0.0.0 --port 8000

Environment:
    MODEL_PATH   model dir (default: models/nllb-en-sw-ct2 if it exists,
                 else models/nllb-en-sw)
    FEEDBACK_DB  SQLite feedback db path (default: feedback.db)
"""
import os
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from engine import LANGUAGES, Translator
from feedback import FeedbackStore


class TranslateRequest(BaseModel):
    text: str
    source_lang: str
    target_lang: str


class TranslateResponse(BaseModel):
    translation: str
    source_lang: str
    target_lang: str


class FeedbackRequest(BaseModel):
    source_text: str
    source_lang: str
    target_lang: str
    translation: str
    rating: Literal["up", "down"] | None = None
    correction: str | None = None


class FeedbackResponse(BaseModel):
    id: int


def default_model_path():
    ct2 = "models/nllb-en-sw-ct2"
    return ct2 if os.path.exists(ct2) else "models/nllb-en-sw"


def create_app(translator=None, store=None):
    if translator is not None and store is not None:
        app = FastAPI()
        app.state.translator = translator
        app.state.store = store
    else:

        @asynccontextmanager
        async def lifespan(app):
            app.state.translator = Translator(os.environ.get("MODEL_PATH", default_model_path()))
            app.state.store = FeedbackStore(os.environ.get("FEEDBACK_DB", "feedback.db"))
            yield

        app = FastAPI(lifespan=lifespan)

    @app.post("/translate", response_model=TranslateResponse)
    def translate(req: TranslateRequest):
        try:
            out = app.state.translator.translate(req.text, req.source_lang, req.target_lang)
        except ValueError as e:
            raise HTTPException(status_code=422, detail=str(e))
        return {"translation": out, "source_lang": req.source_lang, "target_lang": req.target_lang}

    @app.post("/feedback", response_model=FeedbackResponse)
    def feedback(req: FeedbackRequest):
        if req.source_lang not in LANGUAGES or req.target_lang not in LANGUAGES:
            raise HTTPException(status_code=422, detail="Unsupported language")
        try:
            row_id = app.state.store.add(
                req.source_lang,
                req.target_lang,
                req.source_text,
                req.translation,
                rating=req.rating,
                correction=req.correction,
            )
        except ValueError as e:
            raise HTTPException(status_code=422, detail=str(e))
        return {"id": row_id}

    @app.get("/languages")
    def languages():
        return LANGUAGES

    @app.get("/health")
    def health():
        return {"status": "ok", "backend": app.state.translator.backend, "model": app.state.translator.model_name}

    return app


app = create_app()
