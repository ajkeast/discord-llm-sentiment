"""Incremental nightly scorer for messages missing from message_sentiment.

Usage:
    .venv/bin/python -m sentiment.nightly
    .venv/bin/python -m sentiment.nightly --limit 50
"""

from __future__ import annotations

import sys

import pandas as pd
from sqlalchemy import text

from sentiment.client import make_client
from sentiment.config import get_pipeline_config
from sentiment.db import get_engine
from sentiment.schema import results_to_dataframe
from sentiment.store import (
    count_sentiment_rows,
    ensure_sentiment_table,
    fetch_unscored_messages,
    upsert_sentiment_dataframe,
)
from sentiment.windows import build_windows


def _load_channel_histories(mysql, channel_ids: list) -> pd.DataFrame:
    """Load channel timelines so preceding-message context is available."""
    if not channel_ids:
        return pd.DataFrame()

    engine = get_engine(mysql)
    frames: list[pd.DataFrame] = []
    with engine.connect() as conn:
        for cid in channel_ids:
            if pd.isna(cid):
                continue
            part = pd.read_sql(
                text(
                    """
                    SELECT
                        m.id AS message_id,
                        m.member_id,
                        m.channel_id,
                        m.content,
                        m.created_at,
                        COALESCE(mem.display_name, mem.user_name, m.member_id) AS author_name,
                        c.channel_name
                    FROM messages m
                    LEFT JOIN members mem ON m.member_id = mem.id
                    LEFT JOIN channels c ON m.channel_id = c.id
                    WHERE m.channel_id = :cid
                    ORDER BY m.created_at, m.id
                    """
                ),
                conn,
                params={"cid": int(cid)},
            )
            frames.append(part)

    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if df.empty:
        return df

    df["created_at"] = pd.to_datetime(df["created_at"])
    df["content"] = df["content"].fillna("").astype(str)
    df["author_name"] = df["author_name"].fillna("unknown").astype(str)
    df["channel_name"] = df["channel_name"].fillna("unknown").astype(str)
    df["content_len"] = df["content"].str.len()
    return df


def run_nightly(*, limit: int | None = None) -> int:
    cfg = get_pipeline_config(require_mysql=True)
    if not cfg.gemini_api_key:
        raise SystemExit("GEMINI_API_KEY is required")

    ensure_sentiment_table(cfg.mysql)
    before = count_sentiment_rows(cfg.mysql)
    unscored = fetch_unscored_messages(cfg.mysql)
    if unscored.empty:
        print(f"No unscored messages. message_sentiment rows={before}")
        return 0

    if limit is not None:
        unscored = unscored.head(limit)

    channel_ids = unscored["channel_id"].dropna().unique().tolist()
    history = _load_channel_histories(cfg.mysql, channel_ids)
    if history.empty:
        history = unscored.copy()
        if "content_len" not in history.columns:
            history["content_len"] = history["content"].str.len()

    windows_all = build_windows(
        history,
        context_size=cfg.context_size,
        max_content_chars=cfg.max_content_chars,
    )
    target_ids = set(unscored["message_id"].astype(str))
    windows = windows_all[
        windows_all["message_id"].astype(str).isin(target_ids)
    ].reset_index(drop=True)

    print(f"Scoring {len(windows):,} unscored messages with {cfg.model}...")
    client = make_client()
    run = client.score_windows(
        windows,
        batch_size=cfg.batch_size,
        run_name=None,
        show_progress=True,
    )
    df = run.dataframe if run.dataframe is not None else results_to_dataframe(run.results)
    written = upsert_sentiment_dataframe(cfg.mysql, df, model=cfg.model)
    after = count_sentiment_rows(cfg.mysql)
    print(f"Upserted {written}. DB {before} → {after}. Usage: {run.usage.as_dict()}")
    return written


def main(argv: list[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    limit = None
    if "--limit" in argv:
        i = argv.index("--limit")
        limit = int(argv[i + 1])
    run_nightly(limit=limit)


if __name__ == "__main__":
    main()
