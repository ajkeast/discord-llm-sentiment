"""Parquet I/O and resume checkpointing for scoring runs."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from sentiment.config import ensure_data_dirs
from sentiment.schema import SentimentResult, results_to_dataframe


def results_dir() -> Path:
    return ensure_data_dirs()["results"]


def checkpoint_path(run_name: str) -> Path:
    return results_dir() / f"{run_name}_checkpoint.parquet"


def final_results_path(run_name: str = "sentiment_results") -> Path:
    return ensure_data_dirs()["data"] / f"{run_name}.parquet"


def load_scored_ids(run_name: str) -> set[str]:
    path = checkpoint_path(run_name)
    if not path.exists():
        return set()
    df = pd.read_parquet(path, columns=["message_id"])
    return set(df["message_id"].astype(str))


def append_checkpoint(
    run_name: str,
    results: list[SentimentResult],
) -> Path:
    """Append validated results to a checkpoint parquet (dedupe by message_id)."""
    path = checkpoint_path(run_name)
    new_df = results_to_dataframe(results)
    if new_df.empty:
        return path

    new_df = new_df.copy()
    new_df["message_id"] = new_df["message_id"].astype(str)

    if path.exists():
        old = pd.read_parquet(path)
        old["message_id"] = old["message_id"].astype(str)
        combined = pd.concat([old, new_df], ignore_index=True)
        combined = combined.drop_duplicates(subset=["message_id"], keep="last")
    else:
        combined = new_df

    # String message_id avoids int64 overflow from rare malformed model outputs
    combined["message_id"] = combined["message_id"].astype(str)
    combined.to_parquet(path, index=False)
    return path


def finalize_results(
    run_name: str,
    *,
    output_name: str = "sentiment_results",
) -> Path:
    """Copy checkpoint to the final results parquet path."""
    src = checkpoint_path(run_name)
    if not src.exists():
        raise FileNotFoundError(f"No checkpoint at {src}")
    dest = final_results_path(output_name)
    df = pd.read_parquet(src)
    df.to_parquet(dest, index=False)
    return dest


def filter_unscored(windows: pd.DataFrame, run_name: str) -> pd.DataFrame:
    done = load_scored_ids(run_name)
    if not done:
        return windows
    mask = ~windows["message_id"].astype(str).isin(done)
    return windows.loc[mask].reset_index(drop=True)


def save_sample(df: pd.DataFrame, name: str) -> Path:
    paths = ensure_data_dirs()
    out = paths["samples"] / f"{name}.parquet"
    df.to_parquet(out, index=False)
    return out


def load_sample(name: str) -> pd.DataFrame:
    path = ensure_data_dirs()["samples"] / f"{name}.parquet"
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_parquet(path)


def gold_template_path() -> Path:
    return ensure_data_dirs()["gold"] / "gold_labels.csv"


def write_gold_template(sample_df: pd.DataFrame, n: int = 200) -> Path:
    """Write a CSV template for manual gold labels (with model preds as hints)."""
    base_cols = [
        "message_id",
        "channel_name",
        "author_name",
        "content",
        "context_text",
    ]
    cols = [c for c in base_cols if c in sample_df.columns]
    subset = sample_df[cols].head(n).copy()

    for src, dest in (
        ("polarity", "pred_polarity"),
        ("sarcasm", "pred_sarcasm"),
        ("toxicity", "pred_toxicity"),
    ):
        if src in sample_df.columns:
            subset[dest] = sample_df[src].head(n).values

    subset["gold_polarity"] = ""
    subset["gold_sarcasm"] = ""
    subset["gold_toxicity"] = ""
    out = gold_template_path()
    subset.to_csv(out, index=False)
    return out
