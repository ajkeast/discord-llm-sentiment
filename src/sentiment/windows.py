"""Build preceding-message context windows per channel."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from sentiment.config import ensure_data_dirs


def _truncate(text: str, max_chars: int) -> str:
    text = text.replace("\n", " ").strip()
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 1] + "…"


def format_context_block(
    priors: list[tuple[str, str]],
    target_author: str,
    target_content: str,
    max_content_chars: int = 500,
) -> str:
    """Format prior messages + marked target for the LLM prompt."""
    lines: list[str] = []
    for author, content in priors:
        lines.append(f"[{_truncate(author, 40)}] {_truncate(content, max_content_chars)}")
    lines.append(
        f">>> TARGET [{_truncate(target_author, 40)}] "
        f"{_truncate(target_content, max_content_chars)}"
    )
    return "\n".join(lines)


def build_windows(
    messages: pd.DataFrame,
    *,
    context_size: int = 3,
    max_content_chars: int = 500,
    skip_empty_targets: bool = True,
) -> pd.DataFrame:
    """
    For each scorable message, attach up to `context_size` preceding
    messages from the same channel (by created_at, then message_id).
    """
    required = {
        "message_id",
        "channel_id",
        "content",
        "created_at",
        "author_name",
        "channel_name",
        "member_id",
    }
    missing = required - set(messages.columns)
    if missing:
        raise ValueError(f"messages missing columns: {sorted(missing)}")

    df = messages.sort_values(
        ["channel_id", "created_at", "message_id"], kind="mergesort"
    ).reset_index(drop=True)

    if "content_len" not in df.columns:
        df["content_len"] = df["content"].astype(str).str.len()

    # Avoid leading underscores — pandas itertuples renames those columns.
    for i in range(1, context_size + 1):
        df[f"prior_author_{i}"] = df.groupby("channel_id")["author_name"].shift(i)
        df[f"prior_content_{i}"] = df.groupby("channel_id")["content"].shift(i)

    if skip_empty_targets:
        df = df[~df["content"].astype(str).str.strip().eq("")].copy()

    context_texts: list[str] = []
    prior_counts: list[int] = []

    # Column arrays (faster than iterrows for ~270k messages)
    target_authors = df["author_name"].astype(str).to_numpy()
    target_contents = df["content"].astype(str).to_numpy()
    prior_authors = [
        df[f"prior_author_{i}"].to_numpy() for i in range(context_size, 0, -1)
    ]
    prior_contents = [
        df[f"prior_content_{i}"].to_numpy() for i in range(context_size, 0, -1)
    ]

    n = len(df)
    for idx in range(n):
        priors: list[tuple[str, str]] = []
        for authors_arr, contents_arr in zip(prior_authors, prior_contents):
            author = authors_arr[idx]
            content = contents_arr[idx]
            if pd.isna(author) or pd.isna(content):
                continue
            content_str = str(content).strip()
            if not content_str:
                continue
            priors.append((str(author), content_str))
        context_texts.append(
            format_context_block(
                priors,
                target_authors[idx],
                target_contents[idx],
                max_content_chars=max_content_chars,
            )
        )
        prior_counts.append(len(priors))

    out = pd.DataFrame(
        {
            "message_id": df["message_id"].to_numpy(),
            "member_id": df["member_id"].to_numpy(),
            "channel_id": df["channel_id"].to_numpy(),
            "channel_name": df["channel_name"].to_numpy(),
            "author_name": df["author_name"].to_numpy(),
            "content": df["content"].to_numpy(),
            "created_at": df["created_at"].to_numpy(),
            "content_len": df["content_len"].to_numpy(),
            "context_text": context_texts,
            "n_priors": prior_counts,
        }
    )
    out["year"] = pd.to_datetime(out["created_at"]).dt.year
    out["length_bucket"] = pd.cut(
        out["content_len"],
        bins=[-1, 20, 100, 10_000],
        labels=["short", "medium", "long"],
    ).astype(str)
    return out.reset_index(drop=True)


def export_windows_parquet(
    messages: pd.DataFrame,
    *,
    context_size: int = 3,
    max_content_chars: int = 500,
    path: Path | None = None,
) -> Path:
    paths = ensure_data_dirs()
    out = path or (paths["data"] / "windows.parquet")
    windows = build_windows(
        messages,
        context_size=context_size,
        max_content_chars=max_content_chars,
    )
    windows.to_parquet(out, index=False)
    return out


def load_windows_parquet(path: Path | None = None) -> pd.DataFrame:
    paths = ensure_data_dirs()
    src = path or (paths["data"] / "windows.parquet")
    if not src.exists():
        raise FileNotFoundError(
            f"Missing {src}. Run build_windows / notebook 01 first."
        )
    df = pd.read_parquet(src)
    df["created_at"] = pd.to_datetime(df["created_at"])
    return df
