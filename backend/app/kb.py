"""Knowledge base: upload FAQ files, split them into chunks, and search them.

How search works:
1. Each chunk is turned into a vector (a list of numbers) and saved in Postgres.
2. A customer question is turned into a vector the same way.
3. pgvector finds the chunks whose vectors point in the closest direction
   (cosine distance). Close direction = similar meaning.
"""
import io
import re

from pypdf import PdfReader

from app.db import connect, to_vector
from app.llm import embed

ALLOWED = (".txt", ".md", ".pdf")
MAX_CHARS = 800  # one chunk = roughly one or two FAQ answers


def extract_text(filename: str, data: bytes) -> str:
    name = filename.lower()
    if name.endswith(".pdf"):
        reader = PdfReader(io.BytesIO(data))
        return "\n\n".join(page.extract_text() or "" for page in reader.pages)
    if name.endswith((".txt", ".md")):
        return data.decode("utf-8", errors="ignore")
    raise ValueError(f"Unsupported file type. Use one of: {', '.join(ALLOWED)}")


def chunk_text(text: str, max_chars: int = MAX_CHARS) -> list[str]:
    """Split text into chunks of up to max_chars, keeping paragraphs together.

    A markdown heading (# Shipping) is added to the top of each chunk under it,
    so a chunk like "7-12 business days" still knows it is about shipping.
    """
    chunks: list[str] = []
    section = ""
    buf: list[str] = []

    def flush():
        if buf:
            body = "\n\n".join(buf)
            chunks.append(f"{section}\n{body}" if section else body)
            buf.clear()

    for block in re.split(r"\n\s*\n", text):
        block = block.strip()
        if not block:
            continue
        if block.startswith("#"):
            flush()
            heading, _, block = block.partition("\n")
            section = heading.lstrip("#").strip()
            block = block.strip()
            if not block:
                continue
        for i in range(0, len(block), max_chars):  # very long paragraphs are cut
            piece = block[i:i + max_chars]
            if buf and sum(len(b) for b in buf) + len(piece) > max_chars:
                flush()
            buf.append(piece)

    flush()
    return chunks


def ingest(filename: str, data: bytes) -> dict:
    """Read a file, chunk it, embed the chunks and save everything."""
    text = extract_text(filename, data)
    chunks = chunk_text(text)
    if not chunks:
        raise ValueError("No text found in this file.")

    vectors = embed(chunks, task="document")

    with connect() as conn:
        doc_id = conn.execute(
            "INSERT INTO kb_documents (name) VALUES (%s) RETURNING id", (filename,)
        ).fetchone()[0]
        with conn.cursor() as cur:
            cur.executemany(
                "INSERT INTO kb_chunks (document_id, content, embedding) "
                "VALUES (%s, %s, %s::vector)",
                [(doc_id, c, to_vector(v)) for c, v in zip(chunks, vectors)],
            )

    return {"id": doc_id, "name": filename, "chunks": len(chunks)}


def search(question: str, k: int = 4) -> list[dict]:
    """Return the k chunks closest in meaning to the question, best first."""
    vec = to_vector(embed([question], task="query")[0])
    with connect() as conn:
        rows = conn.execute(
            """
            SELECT c.content, d.name, 1 - (c.embedding <=> %s::vector) AS score
            FROM kb_chunks c
            JOIN kb_documents d ON d.id = c.document_id
            ORDER BY c.embedding <=> %s::vector
            LIMIT %s
            """,
            (vec, vec, k),
        ).fetchall()
    return [
        {"content": content, "document": name, "score": round(float(score), 3)}
        for content, name, score in rows
    ]


def list_documents() -> list[dict]:
    with connect() as conn:
        rows = conn.execute(
            """
            SELECT d.id, d.name, d.created_at, COUNT(c.id)
            FROM kb_documents d
            LEFT JOIN kb_chunks c ON c.document_id = d.id
            GROUP BY d.id
            ORDER BY d.created_at DESC
            """
        ).fetchall()
    return [
        {"id": i, "name": n, "created_at": t.isoformat(), "chunks": c}
        for i, n, t, c in rows
    ]


def delete_document(doc_id: int) -> bool:
    """Deletes the document and (through ON DELETE CASCADE) all its chunks."""
    with connect() as conn:
        cur = conn.execute("DELETE FROM kb_documents WHERE id = %s", (doc_id,))
        return cur.rowcount > 0
