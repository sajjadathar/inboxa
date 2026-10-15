"""Emails: save, list, read and update the emails that n8n processes."""
from datetime import datetime
from typing import Literal, Optional

from psycopg.rows import dict_row
from pydantic import BaseModel

from app.db import connect

# Every place an email can be in its journey
Status = Literal[
    "processing", "auto_sent", "needs_approval", "sent", "rejected", "archived", "logged"
]


class EmailCreate(BaseModel):
    """What n8n sends when it saves an email."""
    gmail_id: str
    thread_id: str
    sender: str
    subject: str
    body: str
    category: str
    confidence: float
    urgency: str
    language: str
    summary: str
    draft_reply: Optional[str] = None
    needs_human: bool = True
    status: Status = "processing"


class EmailUpdate(BaseModel):
    """What can be changed later. Both fields are optional."""
    status: Optional[Status] = None
    draft_reply: Optional[str] = None


class EmailOut(EmailCreate):
    """What the API sends back: everything above, plus fields the database adds."""
    id: int
    received_at: datetime
    updated_at: datetime


def save_email(data: EmailCreate) -> EmailOut:
    """Insert a new email. If the same Gmail message comes again, update it instead."""
    with connect() as conn, conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            """
            INSERT INTO emails (gmail_id, thread_id, sender, subject, body,
                                category, confidence, urgency, language, summary,
                                draft_reply, needs_human, status)
            VALUES (%(gmail_id)s, %(thread_id)s, %(sender)s, %(subject)s, %(body)s,
                    %(category)s, %(confidence)s, %(urgency)s, %(language)s, %(summary)s,
                    %(draft_reply)s, %(needs_human)s, %(status)s)
            ON CONFLICT (gmail_id) DO UPDATE SET
                category    = EXCLUDED.category,
                confidence  = EXCLUDED.confidence,
                urgency     = EXCLUDED.urgency,
                language    = EXCLUDED.language,
                summary     = EXCLUDED.summary,
                draft_reply = EXCLUDED.draft_reply,
                needs_human = EXCLUDED.needs_human,
                status      = EXCLUDED.status,
                updated_at  = now()
            RETURNING *
            """,
            data.model_dump(),
        )
        return EmailOut(**cur.fetchone())


def list_emails(
    status: Optional[str] = None, category: Optional[str] = None, limit: int = 50
) -> list[EmailOut]:
    """Newest first. Optionally only one status and/or one category."""
    where, params = [], {"limit": limit}
    if status:
        where.append("status = %(status)s")
        params["status"] = status
    if category:
        where.append("category = %(category)s")
        params["category"] = category
    where_sql = ("WHERE " + " AND ".join(where)) if where else ""

    with connect() as conn, conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            f"SELECT * FROM emails {where_sql} ORDER BY received_at DESC LIMIT %(limit)s",
            params,
        )
        return [EmailOut(**row) for row in cur.fetchall()]


def get_email(email_id: int) -> Optional[EmailOut]:
    with connect() as conn, conn.cursor(row_factory=dict_row) as cur:
        cur.execute("SELECT * FROM emails WHERE id = %s", (email_id,))
        row = cur.fetchone()
        return EmailOut(**row) if row else None


def update_email(email_id: int, changes: EmailUpdate) -> Optional[EmailOut]:
    """Change only the fields that were sent. Returns None if the email doesn't exist."""
    fields = changes.model_dump(exclude_none=True)  # drop fields that weren't sent
    if not fields:
        return get_email(email_id)

    set_sql = ", ".join(f"{name} = %({name})s" for name in fields)
    with connect() as conn, conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            f"UPDATE emails SET {set_sql}, updated_at = now() "
            f"WHERE id = %(id)s RETURNING *",
            {**fields, "id": email_id},
        )
        row = cur.fetchone()
        return EmailOut(**row) if row else None