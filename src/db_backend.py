"""Postgres-backed (Supabase) results storage — the durable alternative to
validation.py's local CSV, for when this app runs somewhere that doesn't
guarantee local disk survives a restart/redeploy (e.g. Streamlit Community
Cloud's free tier).

Uses Streamlit's built-in `st.connection(..., type="sql")` against a plain
Postgres connection string (Supabase exposes one directly — no Supabase-
specific client library needed). Table is created on first use if it
doesn't exist yet, so the only setup on the Supabase side is "create a
project and copy the connection string."

NOT independently unit-tested against a live database in this session —
there's no Supabase project to test against yet. Written against
Streamlit's documented st.connection/SQLConnection API and plain
SQLAlchemy; verify it for real once secrets are configured (see README).
"""
from __future__ import annotations

import pandas as pd
import streamlit as st
from sqlalchemy import text

from .validation import COLUMNS, RESULTS  # reuse the same schema/result vocabulary

CONNECTION_NAME = "validation_db"  # matches the [connections.validation_db] secrets section

_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS validation_results (
    id BIGSERIAL PRIMARY KEY,
    case_id TEXT NOT NULL,
    label_type TEXT NOT NULL,
    reviewer TEXT NOT NULL,
    result TEXT NOT NULL,
    notes TEXT NOT NULL DEFAULT '',
    timestamp TIMESTAMPTZ NOT NULL DEFAULT now()
);
"""


def is_configured() -> bool:
    """True if a [connections.validation_db] secrets section exists — the
    signal to use this backend instead of the local CSV. Never raises:
    a missing/absent secrets.toml (the normal local-dev case) just means
    "not configured", not an error."""
    try:
        return "connections" in st.secrets and CONNECTION_NAME in st.secrets["connections"]
    except Exception:
        return False


@st.cache_resource(show_spinner=False)
def _get_connection():
    conn = st.connection(CONNECTION_NAME, type="sql")
    with conn.session as s:
        s.execute(text(_CREATE_TABLE_SQL))
        s.commit()
    return conn


def load_results() -> pd.DataFrame:
    conn = _get_connection()
    df = conn.query(
        "SELECT case_id, label_type, reviewer, result, notes, timestamp "
        "FROM validation_results ORDER BY id",
        ttl=0,  # always fresh — this table is small and writes must show up immediately
    )
    return df[COLUMNS] if not df.empty else pd.DataFrame(columns=COLUMNS)


def append_result(case_id: str, label_type: str, reviewer: str, result: str, notes: str) -> None:
    if result not in RESULTS:
        raise ValueError(f"result must be one of {RESULTS}, got {result!r}")
    conn = _get_connection()
    with conn.session as s:
        s.execute(
            text(
                "INSERT INTO validation_results (case_id, label_type, reviewer, result, notes) "
                "VALUES (:case_id, :label_type, :reviewer, :result, :notes)"
            ),
            {
                "case_id": case_id,
                "label_type": label_type,
                "reviewer": reviewer or "unknown",
                "result": result,
                "notes": notes or "",
            },
        )
        s.commit()
