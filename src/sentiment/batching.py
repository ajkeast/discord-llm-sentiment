"""Pack window rows into API-sized batches."""

from __future__ import annotations

from collections.abc import Iterator

import pandas as pd


def iter_batches(
    windows: pd.DataFrame,
    *,
    batch_size: int = 15,
) -> Iterator[list[dict]]:
    """Yield lists of prompt item dicts from a windows DataFrame."""
    required = {"message_id", "channel_name", "context_text"}
    missing = required - set(windows.columns)
    if missing:
        raise ValueError(f"windows missing columns: {sorted(missing)}")

    n = len(windows)
    for start in range(0, n, batch_size):
        chunk = windows.iloc[start : start + batch_size]
        items: list[dict] = []
        for row in chunk.itertuples(index=False):
            items.append(
                {
                    "message_id": str(row.message_id),
                    "channel_name": str(row.channel_name),
                    "context_text": str(row.context_text),
                }
            )
        yield items


def batch_count(n_messages: int, batch_size: int = 15) -> int:
    if n_messages <= 0:
        return 0
    return (n_messages + batch_size - 1) // batch_size
