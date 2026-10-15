"""Inboxa AI service. Phase 1 only has a health check; Phase 2 adds /classify and /draft-reply."""
import psycopg
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import ValidationError

from app.classify import Classification, EmailIn, classify_email
from app.config import settings
from app.llm import LLMError

app = FastAPI(title="Inboxa AI service")

# Lets the Next.js dashboard (port 3000) call this API (port 8000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    """Checks that the API is up, the database answers, and pgvector is installed."""
    db_ok, pgvector_ok = False, False
    try:
        with psycopg.connect(settings.database_url, connect_timeout=3) as conn:
            db_ok = True
            row = conn.execute(
                "SELECT 1 FROM pg_extension WHERE extname = 'vector'"
            ).fetchone()
            pgvector_ok = row is not None
    except Exception:
        pass

    return {
        "api": "ok",
        "database": "ok" if db_ok else "down",
        "pgvector": "ok" if pgvector_ok else "missing",
        "llm_provider": settings.llm_provider,
        "llm_model": settings.llm_model,
    }


@app.post("/classify", response_model=Classification)
def classify(email: EmailIn):
    """Reads one email and returns its category, urgency, language and confidence."""
    try:
        return classify_email(email)
    except LLMError as e:
        raise HTTPException(status_code=502, detail=str(e))
    except ValidationError as e:
        raise HTTPException(status_code=502, detail=f"AI answer had the wrong shape: {e}")