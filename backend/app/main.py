"""Inboxa AI service: classify emails, manage the knowledge base, draft replies."""
from contextlib import asynccontextmanager

import psycopg
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ValidationError

from app import kb
from app.classify import Classification, EmailIn, classify_email
from app.config import settings
from app.db import init_db
from app.llm import LLMError
from app.reply import ReplyOut, draft_reply


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()  # create tables on startup
    yield


app = FastAPI(title="Inboxa AI service", lifespan=lifespan)

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


# ---------- Knowledge base ----------

class SearchIn(BaseModel):
    question: str
    k: int = 4


@app.post("/kb/upload")
def kb_upload(file: UploadFile = File(...)):
    """Upload a .txt, .md or .pdf FAQ file. It is split into chunks and embedded."""
    try:
        return kb.ingest(file.filename or "file.txt", file.file.read())
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except LLMError as e:
        raise HTTPException(status_code=502, detail=str(e))


@app.get("/kb/documents")
def kb_documents():
    """List uploaded documents and how many chunks each has."""
    return kb.list_documents()


@app.delete("/kb/documents/{doc_id}")
def kb_delete(doc_id: int):
    if not kb.delete_document(doc_id):
        raise HTTPException(status_code=404, detail="Document not found")
    return {"deleted": doc_id}


@app.post("/kb/search")
def kb_search(body: SearchIn):
    """Test search: shows which chunks match a question, with similarity scores."""
    try:
        return kb.search(body.question, body.k)
    except LLMError as e:
        raise HTTPException(status_code=502, detail=str(e))


# ---------- Replies ----------

@app.post("/draft-reply", response_model=ReplyOut)
def reply(email: EmailIn):
    """Writes a reply from the knowledge base and says whether a person should check it."""
    try:
        return draft_reply(email)
    except LLMError as e:
        raise HTTPException(status_code=502, detail=str(e))
    except (ValidationError, TypeError) as e:
        raise HTTPException(status_code=502, detail=f"AI answer had the wrong shape: {e}")
