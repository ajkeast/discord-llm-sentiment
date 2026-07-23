#!/usr/bin/env python3
"""Download succeeded Gemini Batch result files and upsert into MySQL.

Does NOT re-run scoring — recovers results already paid for.
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from google import genai
from google.genai import types

from sentiment.batch_results import parse_batch_job
from sentiment.config import get_pipeline_config
from sentiment.io import append_checkpoint, finalize_results
from sentiment.schema import results_to_dataframe
from sentiment.store import count_sentiment_rows, ensure_sentiment_table, upsert_sentiment_dataframe
from sentiment.windows import load_windows_parquet


# First full-corpus chunk from notebook run (exclude earlier probes)
FULL_RUN_START = datetime(2026, 7, 18, 22, 39, 0, tzinfo=timezone.utc)


def main() -> None:
    cfg = get_pipeline_config(require_mysql=True)
    ensure_sentiment_table(cfg.mysql)
    client = genai.Client(api_key=cfg.gemini_api_key)

    jobs = []
    for job in client.batches.list(config=types.ListBatchJobsConfig(page_size=50)):
        state = str(job.state)
        if "SUCCEEDED" not in state:
            continue
        created = job.create_time
        if created is None or created < FULL_RUN_START:
            continue
        jobs.append(job)

    # oldest → newest (chunk order)
    jobs.sort(key=lambda j: j.create_time or datetime.min.replace(tzinfo=timezone.utc))
    print(f"Found {len(jobs)} succeeded full-run batch job(s) to recover.", flush=True)

    before = count_sentiment_rows(cfg.mysql)
    total_parsed = 0
    for i, job in enumerate(jobs, start=1):
        print(f"\n[{i}/{len(jobs)}] {job.name}", flush=True)
        results = parse_batch_job(client, job)
        print(f"  parsed {len(results):,} sentiment rows", flush=True)
        if not results:
            continue
        df = results_to_dataframe(results)
        n = upsert_sentiment_dataframe(cfg.mysql, df, model=cfg.model)
        try:
            append_checkpoint("full_corpus", results)
        except Exception as exc:
            print(f"  checkpoint warn (DB upsert still ok): {exc}", flush=True)
        total_parsed += len(results)
        print(
            f"  upserted {n:,} | DB now {count_sentiment_rows(cfg.mysql):,}",
            flush=True,
        )

    after = count_sentiment_rows(cfg.mysql)
    print(f"\nDone. Parsed {total_parsed:,}. DB {before:,} → {after:,}", flush=True)

    # Coverage vs windows
    windows = load_windows_parquet()
    from sentiment.store import fetch_scored_message_ids

    scored = fetch_scored_message_ids(cfg.mysql)
    missing = set(windows["message_id"].astype(str)) - scored
    print(
        f"Windows coverage: {len(windows) - len(missing):,}/{len(windows):,} "
        f"(missing {len(missing):,})",
        flush=True,
    )

    try:
        out = finalize_results("full_corpus", output_name="sentiment_results")
        print(f"Parquet backup → {out}", flush=True)
    except FileNotFoundError:
        print("No checkpoint to finalize.", flush=True)


if __name__ == "__main__":
    main()
