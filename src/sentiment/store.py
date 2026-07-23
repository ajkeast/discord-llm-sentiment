"""Persist sentiment results to MySQL (message_sentiment)."""

from __future__ import annotations

from typing import Iterable

import pandas as pd
from sqlalchemy import text

from sentiment.config import MySQLConfig, DEFAULT_MODEL
from sentiment.db import get_engine
from sentiment.schema import SentimentResult, results_to_dataframe

CREATE_MESSAGE_SENTIMENT_SQL = """
CREATE TABLE IF NOT EXISTS message_sentiment (
  message_id BIGINT(20) NOT NULL,
  polarity ENUM('positive','negative','neutral','mixed') NOT NULL,
  polarity_score FLOAT NOT NULL,
  emotions VARCHAR(128) NOT NULL,
  sarcasm TINYINT(1) NOT NULL,
  toxicity ENUM('none','mild','moderate','severe') NOT NULL,
  directed_at ENUM('general','person','group','self','topic') NOT NULL,
  confidence FLOAT NOT NULL,
  rationale VARCHAR(255) NOT NULL,
  model VARCHAR(64) NOT NULL,
  scored_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  PRIMARY KEY (message_id),
  KEY idx_sentiment_scored_at (scored_at),
  KEY idx_sentiment_polarity (polarity),
  CONSTRAINT fk_sentiment_message FOREIGN KEY (message_id) REFERENCES messages (id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
"""

UPSERT_SQL = """
INSERT INTO message_sentiment (
  message_id, polarity, polarity_score, emotions, sarcasm, toxicity,
  directed_at, confidence, rationale, model
) VALUES (
  :message_id, :polarity, :polarity_score, :emotions, :sarcasm, :toxicity,
  :directed_at, :confidence, :rationale, :model
)
ON DUPLICATE KEY UPDATE
  polarity = VALUES(polarity),
  polarity_score = VALUES(polarity_score),
  emotions = VALUES(emotions),
  sarcasm = VALUES(sarcasm),
  toxicity = VALUES(toxicity),
  directed_at = VALUES(directed_at),
  confidence = VALUES(confidence),
  rationale = VALUES(rationale),
  model = VALUES(model),
  updated_at = CURRENT_TIMESTAMP
"""

UNSCORED_MESSAGES_QUERY = """
SELECT
    m.id AS message_id,
    m.member_id,
    m.channel_id,
    m.content,
    m.created_at,
    COALESCE(mem.display_name, mem.user_name, m.member_id) AS author_name,
    mem.user_name,
    c.channel_name
FROM messages m
LEFT JOIN members mem ON m.member_id = mem.id
LEFT JOIN channels c ON m.channel_id = c.id
LEFT JOIN message_sentiment s ON s.message_id = m.id
WHERE s.message_id IS NULL
  AND m.content IS NOT NULL
  AND TRIM(m.content) <> ''
ORDER BY m.channel_id, m.created_at, m.id
"""


def ensure_sentiment_table(mysql: MySQLConfig) -> None:
    engine = get_engine(mysql)
    with engine.begin() as conn:
        conn.execute(text(CREATE_MESSAGE_SENTIMENT_SQL))


def fetch_scored_message_ids(mysql: MySQLConfig) -> set[str]:
    engine = get_engine(mysql)
    with engine.connect() as conn:
        rows = conn.execute(text("SELECT message_id FROM message_sentiment")).fetchall()
    return {str(r[0]) for r in rows}


def count_sentiment_rows(mysql: MySQLConfig) -> int:
    engine = get_engine(mysql)
    with engine.connect() as conn:
        return int(conn.execute(text("SELECT COUNT(*) FROM message_sentiment")).scalar_one())


def fetch_unscored_messages(mysql: MySQLConfig) -> pd.DataFrame:
    """Messages with content that do not yet have a sentiment row."""
    engine = get_engine(mysql)
    with engine.connect() as conn:
        df = pd.read_sql(text(UNSCORED_MESSAGES_QUERY), conn)
    if df.empty:
        return df
    df["message_id"] = df["message_id"].astype("int64")
    df["channel_id"] = df["channel_id"].astype("Int64")
    df["created_at"] = pd.to_datetime(df["created_at"])
    df["content"] = df["content"].fillna("").astype(str)
    df["author_name"] = df["author_name"].fillna("unknown").astype(str)
    df["channel_name"] = df["channel_name"].fillna("unknown").astype(str)
    df["content_len"] = df["content"].str.len()
    return df


def _normalize_results_df(df: pd.DataFrame, *, model: str) -> pd.DataFrame:
    required = {
        "message_id",
        "polarity",
        "polarity_score",
        "emotions",
        "sarcasm",
        "toxicity",
        "directed_at",
        "confidence",
        "rationale",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"results missing columns: {sorted(missing)}")

    out = df[list(required)].copy()
    # Filter non-numeric / oversized ids before int cast (model occasionally hallucinates)
    out["message_id"] = out["message_id"].astype(str).str.strip()
    out = out[out["message_id"].str.fullmatch(r"\d+")].copy()
    out = out[out["message_id"].str.len() <= 20].copy()
    out["message_id"] = out["message_id"].astype("int64")
    out["polarity"] = out["polarity"].astype(str).str.lower().str.strip()
    out["emotions"] = out["emotions"].astype(str)
    out["sarcasm"] = out["sarcasm"].astype(bool).astype(int)
    out["toxicity"] = out["toxicity"].astype(str).str.lower().str.strip()
    out["directed_at"] = out["directed_at"].astype(str).str.lower().str.strip()
    out["rationale"] = out["rationale"].astype(str).str.slice(0, 255)
    out["model"] = model
    out["polarity_score"] = out["polarity_score"].astype(float)
    out["confidence"] = out["confidence"].astype(float)
    return out


def fetch_existing_message_ids(
    mysql: MySQLConfig,
    message_ids: list[int] | list[str],
) -> set[int]:
    """Return the subset of message_ids that exist in messages."""
    if not message_ids:
        return set()
    engine = get_engine(mysql)
    existing: set[int] = set()
    ids = [int(x) for x in message_ids]
    with engine.connect() as conn:
        for start in range(0, len(ids), 1000):
            chunk = ids[start : start + 1000]
            placeholders = ", ".join(f":id{i}" for i in range(len(chunk)))
            params = {f"id{i}": mid for i, mid in enumerate(chunk)}
            rows = conn.execute(
                text(f"SELECT id FROM messages WHERE id IN ({placeholders})"),
                params,
            ).fetchall()
            existing.update(int(r[0]) for r in rows)
    return existing


def upsert_sentiment_dataframe(
    mysql: MySQLConfig,
    results: pd.DataFrame,
    *,
    model: str = DEFAULT_MODEL,
    chunk_size: int = 200,
    skip_missing_fk: bool = True,
) -> int:
    """Upsert a results DataFrame into message_sentiment. Returns rows written."""
    if results is None or results.empty:
        return 0

    ensure_sentiment_table(mysql)
    normalized = _normalize_results_df(results, model=model)

    if skip_missing_fk:
        existing = fetch_existing_message_ids(
            mysql, normalized["message_id"].tolist()
        )
        before = len(normalized)
        normalized = normalized[normalized["message_id"].isin(existing)].copy()
        skipped = before - len(normalized)
        if skipped:
            print(
                f"  skipped {skipped:,} rows with message_id missing from messages",
                flush=True,
            )
        if normalized.empty:
            return 0

    engine = get_engine(mysql)
    written = 0
    records = normalized.to_dict(orient="records")
    with engine.begin() as conn:
        for start in range(0, len(records), chunk_size):
            chunk = records[start : start + chunk_size]
            try:
                conn.execute(text(UPSERT_SQL), chunk)
                written += len(chunk)
            except Exception:
                # Fall back to per-row so one bad id doesn't kill the batch
                for row in chunk:
                    try:
                        conn.execute(text(UPSERT_SQL), row)
                        written += 1
                    except Exception as row_exc:
                        print(
                            f"  skip message_id={row.get('message_id')}: {row_exc}",
                            flush=True,
                        )
    return written


def upsert_sentiment_results(
    mysql: MySQLConfig,
    results: Iterable[SentimentResult],
    *,
    model: str = DEFAULT_MODEL,
) -> int:
    df = results_to_dataframe(list(results))
    return upsert_sentiment_dataframe(mysql, df, model=model)


def filter_windows_not_in_db(
    mysql: MySQLConfig,
    windows: pd.DataFrame,
) -> pd.DataFrame:
    """Drop window rows whose message_id already has a sentiment row."""
    scored = fetch_scored_message_ids(mysql)
    if not scored:
        return windows
    mask = ~windows["message_id"].astype(str).isin(scored)
    return windows.loc[mask].reset_index(drop=True)
