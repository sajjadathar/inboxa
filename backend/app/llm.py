"""One place that talks to the AI model.

Every other file calls generate_json(...) and never cares which provider is used.
To switch provider, change LLM_PROVIDER in .env — no code changes needed.
"""
import json

import httpx

from app.config import settings

TIMEOUT = 60  # seconds


class LLMError(Exception):
    """Raised when the model call fails or returns something we can't use."""


def generate_json(system: str, user: str, schema: dict) -> dict:
    """Ask the model a question and get back a Python dict that matches `schema`."""
    provider = settings.llm_provider.lower()
    if provider == "gemini":
        return _gemini(system, user, schema)
    if provider == "ollama":
        return _ollama(system, user, schema)
    raise LLMError(f"Provider '{provider}' is not set up yet. Use gemini or ollama.")


# ---------- Gemini ----------

def _to_gemini_schema(schema: dict) -> dict:
    """Gemini wants type names in capitals (STRING, OBJECT...). Convert our schema."""
    out = {}
    for key, value in schema.items():
        if key == "type":
            out["type"] = value.upper()
        elif key == "properties":
            out["properties"] = {k: _to_gemini_schema(v) for k, v in value.items()}
        elif key == "items":
            out["items"] = _to_gemini_schema(value)
        elif key in ("enum", "required", "description"):
            out[key] = value
        # anything else (like "minimum") is skipped — Gemini's schema support is limited
    return out


def _gemini(system: str, user: str, schema: dict) -> dict:
    if not settings.gemini_api_key:
        raise LLMError("No Gemini key. Add GOOGLE_API_KEY=... to your .env file.")

    url = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        f"{settings.llm_model}:generateContent"
    )
    body = {
        "systemInstruction": {"parts": [{"text": system}]},
        "contents": [{"role": "user", "parts": [{"text": user}]}],
        "generationConfig": {
            "temperature": 0,  # same email -> same answer
            "responseMimeType": "application/json",  # forces JSON output
            "responseSchema": _to_gemini_schema(schema),  # forces OUR shape of JSON
        },
    }
    try:
        r = httpx.post(
            url,
            json=body,
            headers={"x-goog-api-key": settings.gemini_api_key},
            timeout=TIMEOUT,
        )
    except httpx.HTTPError as e:
        raise LLMError(f"Could not reach Gemini: {e}") from e

    if r.status_code == 429:
        raise LLMError("Gemini free-tier limit reached. Wait a minute and try again.")
    if r.status_code != 200:
        raise LLMError(f"Gemini error {r.status_code}: {r.text[:300]}")

    try:
        text = r.json()["candidates"][0]["content"]["parts"][0]["text"]
        return json.loads(text)
    except (KeyError, IndexError, json.JSONDecodeError) as e:
        raise LLMError(f"Gemini returned an unexpected answer: {r.text[:300]}") from e


# ---------- Ollama (free, offline) ----------

def _ollama(system: str, user: str, schema: dict) -> dict:
    url = f"{settings.ollama_base_url}/api/chat"
    body = {
        "model": settings.llm_model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "format": schema,  # Ollama also supports forcing a JSON shape
        "stream": False,
        "options": {"temperature": 0},
    }
    try:
        r = httpx.post(url, json=body, timeout=TIMEOUT * 3)
        r.raise_for_status()
        return json.loads(r.json()["message"]["content"])
    except Exception as e:
        raise LLMError(f"Ollama error: {e}") from e


# ---------- Embeddings (turn text into numbers for search) ----------

def embed(texts: list[str], task: str = "document") -> list[list[float]]:
    """Turn each text into a list of numbers (a vector).

    task="document" for knowledge-base chunks, task="query" for a customer question.
    Similar meanings give similar vectors, which is how search finds the right FAQ.
    """
    provider = settings.llm_provider.lower()
    if provider == "gemini":
        return _gemini_embed(texts, task)
    if provider == "ollama":
        return _ollama_embed(texts)
    raise LLMError(f"Embeddings for '{provider}' are not set up yet. Use gemini or ollama.")


def _gemini_embed(texts: list[str], task: str) -> list[list[float]]:
    if not settings.gemini_api_key:
        raise LLMError("No Gemini key. Add GOOGLE_API_KEY=... to your .env file.")

    model = settings.embed_model
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:batchEmbedContents"
    task_type = "RETRIEVAL_DOCUMENT" if task == "document" else "RETRIEVAL_QUERY"
    vectors: list[list[float]] = []

    for start in range(0, len(texts), 100):  # Gemini accepts up to 100 texts per call
        batch = texts[start:start + 100]
        body = {
            "requests": [
                {
                    "model": f"models/{model}",
                    "content": {"parts": [{"text": t}]},
                    "taskType": task_type,
                    "outputDimensionality": settings.embed_dim,
                }
                for t in batch
            ]
        }
        try:
            r = httpx.post(
                url, json=body,
                headers={"x-goog-api-key": settings.gemini_api_key},
                timeout=TIMEOUT,
            )
        except httpx.HTTPError as e:
            raise LLMError(f"Could not reach Gemini: {e}") from e
        if r.status_code == 429:
            raise LLMError("Gemini free-tier limit reached. Wait a minute and try again.")
        if r.status_code != 200:
            raise LLMError(f"Gemini embedding error {r.status_code}: {r.text[:300]}")
        vectors += [e["values"] for e in r.json()["embeddings"]]

    return vectors


def _ollama_embed(texts: list[str]) -> list[list[float]]:
    try:
        r = httpx.post(
            f"{settings.ollama_base_url}/api/embed",
            json={"model": settings.embed_model, "input": texts},
            timeout=TIMEOUT * 3,
        )
        r.raise_for_status()
        return r.json()["embeddings"]
    except Exception as e:
        raise LLMError(f"Ollama embedding error: {e}") from e
