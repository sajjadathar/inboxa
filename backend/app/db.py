"""Database connection and table setup."""
import psycopg

from app.config import settings


def connect() -> psycopg.Connection:
    return psycopg.connect(settings.database_url)


def to_vector(values: list[float]) -> str:
    """pgvector reads vectors written like '[0.12,-0.5,...]'."""
    return "[" + ",".join(f"{v:.7f}" for v in values) + "]"


def init_db() -> None:
    """Creates the tables if they don't exist yet. Runs every time the API starts."""
    with connect() as conn:
        conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS kb_documents (
                id         SERIAL PRIMARY KEY,
                name       TEXT NOT NULL,
                created_at TIMESTAMPTZ NOT NULL DEFAULT now()
            )
            """
        )
        conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS kb_chunks (
                id          SERIAL PRIMARY KEY,
                document_id INT NOT NULL REFERENCES kb_documents(id) ON DELETE CASCADE,
                content     TEXT NOT NULL,
                embedding   vector({settings.embed_dim}) NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS emails (
                id          SERIAL PRIMARY KEY,
                gmail_id    TEXT NOT NULL UNIQUE,
                thread_id   TEXT NOT NULL,
                sender      TEXT NOT NULL,
                subject     TEXT NOT NULL,
                body        TEXT NOT NULL,
                category    TEXT NOT NULL,
                confidence  REAL NOT NULL,
                urgency     TEXT NOT NULL,
                language    TEXT NOT NULL,
                summary     TEXT NOT NULL,
                draft_reply TEXT,
                needs_human BOOLEAN NOT NULL DEFAULT TRUE,
                status      TEXT NOT NULL DEFAULT 'processing',
                received_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                updated_at  TIMESTAMPTZ NOT NULL DEFAULT now()
            )
            """
        )
