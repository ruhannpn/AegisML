"""
audit_log.py
============
Step 6 of the AI-Governed Multi-Agent Platform.

Provides a dedicated SQLite audit log storage for pipeline runs.

DESIGN PRINCIPLES:
  - Zero LLM calls. Fully deterministic, rule-based database logger.
  - Storage is completely independent of LangGraph's checkpointer (uses audit_log.db,
    never pipeline_state.db).
  - Schema captures structured, timestamped events per pipeline stage:
      run_id, timestamp, event_type, event_source, summary, details_json
  - Distinguishes event_source: 'automated' vs 'human_reviewer'.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from typing import Any

DEFAULT_AUDIT_DB = "audit_log.db"


def init_audit_db(db_path: str = DEFAULT_AUDIT_DB) -> None:
    """Initialize the audit_entries table in audit_log.db if it does not exist."""
    with sqlite3.connect(db_path, check_same_thread=False) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS audit_entries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                event_type TEXT NOT NULL,
                event_source TEXT NOT NULL,
                summary TEXT NOT NULL,
                details_json TEXT NOT NULL
            )
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_audit_run_id ON audit_entries(run_id)
        """)
        conn.commit()


def log_audit_event(
    run_id: str,
    event_type: str,
    event_source: str,
    summary: str,
    details: dict,
    db_path: str = DEFAULT_AUDIT_DB,
) -> None:
    """
    Log a structured audit event row into audit_log.db.

    Parameters
    ----------
    run_id : str
        Unique session / thread_id identifier for the pipeline run.
    event_type : str
        Stage identifier (e.g. 'planner_run', 'data_agent_run', 'training_run',
        'fairness_run', 'human_decision', 'final_outcome').
    event_source : {"automated", "human_reviewer"}
    summary : str
        Short human-readable event description.
    details : dict
        Structured details to be serialized as JSON text.
    db_path : str
    """
    init_audit_db(db_path)
    now_iso = datetime.now(timezone.utc).isoformat()
    details_str = json.dumps(details, default=str)

    with sqlite3.connect(db_path, check_same_thread=False) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO audit_entries (run_id, timestamp, event_type, event_source, summary, details_json)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (run_id, now_iso, event_type, event_source, summary, details_str),
        )
        conn.commit()


def get_audit_trail(run_id: str, db_path: str = DEFAULT_AUDIT_DB) -> list[dict[str, Any]]:
    """
    Retrieve all audit log entries for a given run_id, ordered chronologically.

    Returns
    -------
    list[dict]:
        List of event entries with deserialized 'details' dict.
    """
    init_audit_db(db_path)
    with sqlite3.connect(db_path, check_same_thread=False) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT id, run_id, timestamp, event_type, event_source, summary, details_json
            FROM audit_entries
            WHERE run_id = ?
            ORDER BY id ASC
            """,
            (run_id,),
        )
        rows = cursor.fetchall()

    entries = []
    for row in rows:
        entry_id, r_id, ts, ev_type, ev_source, summary, det_json = row
        try:
            parsed_details = json.loads(det_json)
        except Exception:
            parsed_details = {"raw": det_json}

        entries.append({
            "id": entry_id,
            "run_id": r_id,
            "timestamp": ts,
            "event_type": ev_type,
            "event_source": ev_source,
            "summary": summary,
            "details": parsed_details,
        })

    return entries
