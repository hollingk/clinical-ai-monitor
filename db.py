"""
db.py: SQLite audit log helpers
DataConnect Conference Workshop, pre-built (do not modify)

Provides:
  init_db()          creates tables (starts with an empty audit log)
  seed_sample_data() adds 30 days of sample history (Lab 4)
  log_action()       writes a human review action to the audit log
  get_audit_log()    returns the full audit log as a DataFrame
  get_db_metrics()   returns aggregate stats for the governance panel
  save_correction()  appends a corrected label to corrections_store.csv
"""

import sqlite3
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
import random

DB_PATH = Path(__file__).parent / "data" / "audit_log.db"
CORRECTIONS_PATH = Path(__file__).parent / "data" / "corrections_store.csv"

# ─── Schema ──────────────────────────────────────────────────────────────────

CREATE_AUDIT_TABLE = """
CREATE TABLE IF NOT EXISTS audit_log (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    decision_id       TEXT NOT NULL,
    reviewer          TEXT NOT NULL,
    action            TEXT NOT NULL,          -- approved | corrected | rejected
    ai_suggested_code TEXT NOT NULL,          -- code the AI suggested, before review
    corrected_code    TEXT,                   -- NULL unless action = corrected
    note              TEXT,
    escalation_reason TEXT,
    risk_level        TEXT,
    risk_flags        TEXT,                   -- JSON array stored as text
    reviewed_at       TEXT NOT NULL,
    queued_at         TEXT
);
"""

SEED_CHECK = "SELECT COUNT(*) FROM audit_log WHERE decision_id LIKE 'HIST-%'"


def _get_connection() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(DB_PATH)


# ─── Public API ──────────────────────────────────────────────────────────────

def init_db() -> None:
    """Create tables. The audit log starts empty; see seed_sample_data()."""
    with _get_connection() as conn:
        conn.execute(CREATE_AUDIT_TABLE)
        # Databases created before the rename still call this column original_code
        columns = [row[1] for row in conn.execute("PRAGMA table_info(audit_log)")]
        if "original_code" in columns:
            conn.execute(
                "ALTER TABLE audit_log RENAME COLUMN original_code TO ai_suggested_code"
            )
        conn.commit()


def seed_sample_data() -> None:
    """
    Add 30 days of sample audit history (decision IDs prefixed HIST-) so the
    Lab 4 governance charts have data. Safe to call repeatedly: it only seeds
    once, and leaves real review actions untouched.
    """
    with _get_connection() as conn:
        if conn.execute(SEED_CHECK).fetchone()[0] == 0:
            _seed_sample_data(conn)


def log_action(
    decision_id: str,
    reviewer: str,
    action: str,
    ai_suggested_code: str,
    corrected_code: str | None = None,
    note: str | None = None,
    escalation_reason: str | None = None,
    risk_level: str | None = None,
    risk_flags: list | None = None,
    queued_at: str | None = None,
) -> None:
    """
    Write a single human review action to the audit log.

    action must be one of: "approved" | "corrected" | "rejected"
    """
    import json
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    flags_json = json.dumps(risk_flags or [])

    with _get_connection() as conn:
        conn.execute(
            """
            INSERT INTO audit_log
              (decision_id, reviewer, action, ai_suggested_code, corrected_code,
               note, escalation_reason, risk_level, risk_flags, reviewed_at, queued_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                decision_id, reviewer, action, ai_suggested_code, corrected_code,
                note, escalation_reason, risk_level, flags_json, now, queued_at,
            ),
        )
        conn.commit()


def get_audit_log() -> pd.DataFrame:
    """Return the full audit log as a DataFrame, newest first."""
    with _get_connection() as conn:
        df = pd.read_sql_query(
            "SELECT * FROM audit_log ORDER BY reviewed_at DESC",
            conn,
        )
    return df


def get_db_metrics() -> dict:
    """
    Return aggregate stats for the governance panel.

    Keys: total_reviewed, approved, corrected, rejected, override_rate,
          avg_review_seconds (approximated from seeded data)
    """
    with _get_connection() as conn:
        row = conn.execute("""
            SELECT
                COUNT(*)                                              AS total,
                SUM(CASE WHEN action = 'approved'  THEN 1 ELSE 0 END) AS approved,
                SUM(CASE WHEN action = 'corrected' THEN 1 ELSE 0 END) AS corrected,
                SUM(CASE WHEN action = 'rejected'  THEN 1 ELSE 0 END) AS rejected
            FROM audit_log
        """).fetchone()

    total, approved, corrected, rejected = row
    total = total or 0
    approved = approved or 0
    corrected = corrected or 0
    rejected = rejected or 0
    overrides = corrected + rejected
    override_rate = overrides / total if total > 0 else 0.0

    return {
        "total_reviewed": total,
        "approved": approved,
        "corrected": corrected,
        "rejected": rejected,
        "override_rate": override_rate,
    }


def get_daily_flag_rates() -> pd.DataFrame:
    """
    Return a DataFrame with columns [date, total, flagged, flag_rate]
    for the last 30 days. Used for the drift signal chart in Lab 4.
    """
    with _get_connection() as conn:
        df = pd.read_sql_query(
            """
            SELECT
                DATE(reviewed_at)  AS date,
                COUNT(*)           AS total_reviewed,
                SUM(CASE WHEN action IN ('corrected', 'rejected') THEN 1 ELSE 0 END) AS overrides
            FROM audit_log
            GROUP BY DATE(reviewed_at)
            ORDER BY date
            """,
            conn,
        )
    if not df.empty:
        df["override_rate"] = df["overrides"] / df["total_reviewed"]
    return df


def save_correction(
    decision_id: str,
    ai_suggested_code: str,
    corrected_code: str,
    note: str = "",
) -> None:
    """
    Append a corrected label to corrections_store.csv.
    This CSV accumulates over time and can be used to fine-tune or retrain
    the underlying model.
    """
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    row = pd.DataFrame([{
        "decision_id": decision_id,
        "ai_suggested_code": ai_suggested_code,
        "corrected_code": corrected_code,
        "note": note,
        "corrected_at": now,
    }])

    write_header = not CORRECTIONS_PATH.exists()
    row.to_csv(CORRECTIONS_PATH, mode="a", header=write_header, index=False)


def get_correction_count() -> int:
    """Return the number of corrections accumulated so far."""
    if not CORRECTIONS_PATH.exists():
        return 0
    df = pd.read_csv(CORRECTIONS_PATH)
    return len(df)


# ─── Seed Data ───────────────────────────────────────────────────────────────

def _seed_sample_data(conn: sqlite3.Connection) -> None:
    """
    Insert 30 days of realistic-looking historical audit data.
    Used to make the governance charts meaningful from the start.
    """
    import json

    reviewers = ["Dr. Patel", "M. Okonkwo", "T. Bergstrom", "S. Vasquez"]
    code_categories = [
        ("I10", "cardiovascular"), ("N39.0", "genitourinary"),
        ("J06.9", "respiratory"), ("E11.9", "endocrine"),
        ("M54.50", "musculoskeletal"), ("F32.1", "mental_health"),
        ("C34.10", "oncology"), ("F41.1", "mental_health"),
    ]

    rows = []
    base_date = datetime.now() - timedelta(days=30)

    for day_offset in range(30):
        date = base_date + timedelta(days=day_offset)
        # Volume ramps up slightly over time (simulates growing deployment)
        n_decisions = random.randint(8, 12) + day_offset // 5

        for i in range(n_decisions):
            code, category = random.choice(code_categories)
            reviewer = random.choice(reviewers)
            confidence = random.uniform(0.45, 0.99)
            risk_level = "GREEN"
            flags = []

            if category in ("mental_health", "oncology"):
                risk_level = "YELLOW"
                flags = ["MANDATORY_REVIEW_CATEGORY"]
            if confidence < 0.70:
                risk_level = "YELLOW"
                flags.append("LOW_CONFIDENCE")

            # Determine action based on risk level
            if risk_level == "GREEN":
                action = random.choices(
                    ["approved", "corrected"],
                    weights=[0.92, 0.08]
                )[0]
            else:
                action = random.choices(
                    ["approved", "corrected", "rejected"],
                    weights=[0.55, 0.30, 0.15]
                )[0]

            review_time = date + timedelta(
                hours=random.randint(0, 7),
                minutes=random.randint(0, 59),
                seconds=random.randint(0, 59),
            )

            rows.append((
                f"HIST-{day_offset:02d}-{i:03d}",
                reviewer,
                action,
                code,
                "I10" if action == "corrected" else None,  # simplified correction
                None,  # note
                "Confidence too low, escalated" if action == "rejected" else None,
                risk_level,
                json.dumps(flags),
                review_time.strftime("%Y-%m-%d %H:%M:%S"),
                (review_time - timedelta(minutes=random.randint(2, 45))).strftime("%Y-%m-%d %H:%M:%S"),
            ))

    conn.executemany(
        """
        INSERT INTO audit_log
          (decision_id, reviewer, action, ai_suggested_code, corrected_code,
           note, escalation_reason, risk_level, risk_flags, reviewed_at, queued_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
    )
    conn.commit()
