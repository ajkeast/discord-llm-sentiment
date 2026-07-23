"""MySQL export helpers."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine, text

from sentiment.config import MySQLConfig, ensure_data_dirs

MESSAGES_QUERY = """
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
ORDER BY m.channel_id, m.created_at, m.id
"""


def get_engine(mysql: MySQLConfig):
    return create_engine(mysql.url, pool_pre_ping=True)


def fetch_messages(mysql: MySQLConfig) -> pd.DataFrame:
    """Pull messages joined with member/channel metadata."""
    engine = get_engine(mysql)
    with engine.connect() as conn:
        df = pd.read_sql(text(MESSAGES_QUERY), conn)

    df["message_id"] = df["message_id"].astype("int64")
    df["channel_id"] = df["channel_id"].astype("Int64")
    df["created_at"] = pd.to_datetime(df["created_at"])
    df["content"] = df["content"].fillna("").astype(str)
    df["author_name"] = df["author_name"].fillna("unknown").astype(str)
    df["channel_name"] = df["channel_name"].fillna("unknown").astype(str)
    df["content_len"] = df["content"].str.len()
    df["is_empty"] = df["content"].str.strip().eq("")
    return df


def export_messages_parquet(
    mysql: MySQLConfig,
    path: Path | None = None,
) -> Path:
    """Export full message table to parquet and return the path."""
    paths = ensure_data_dirs()
    out = path or (paths["data"] / "messages.parquet")
    df = fetch_messages(mysql)
    df.to_parquet(out, index=False)
    return out


def load_messages_parquet(path: Path | None = None) -> pd.DataFrame:
    paths = ensure_data_dirs()
    src = path or (paths["data"] / "messages.parquet")
    if not src.exists():
        raise FileNotFoundError(
            f"Missing {src}. Run export_messages_parquet() or notebook 01 first."
        )
    df = pd.read_parquet(src)
    df["created_at"] = pd.to_datetime(df["created_at"])
    return df
