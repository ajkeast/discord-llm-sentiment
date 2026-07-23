"""Stratified sampling for eval sets."""

from __future__ import annotations

import pandas as pd


def stratified_sample(
    windows: pd.DataFrame,
    n: int = 1000,
    *,
    random_state: int = 42,
    channel_floor: int = 15,
) -> pd.DataFrame:
    """
    Stratify by channel, year, and length_bucket.

    Small channels get up to `channel_floor` rows (or all if fewer) so they
    aren't drowned out by #general.
    """
    df = windows.copy()
    if "year" not in df.columns:
        df["year"] = pd.to_datetime(df["created_at"]).dt.year
    if "length_bucket" not in df.columns:
        df["length_bucket"] = pd.cut(
            df["content_len"],
            bins=[-1, 20, 100, 10_000],
            labels=["short", "medium", "long"],
        ).astype(str)

    # Guarantee representation from smaller channels
    reserved_parts: list[pd.DataFrame] = []
    reserved_ids: set = set()
    for channel, group in df.groupby("channel_name"):
        take = min(len(group), channel_floor)
        part = group.sample(n=take, random_state=random_state)
        reserved_parts.append(part)
        reserved_ids.update(part["message_id"].tolist())

    reserved = pd.concat(reserved_parts, ignore_index=True) if reserved_parts else df.iloc[0:0]
    remaining_slots = max(0, n - len(reserved))

    pool = df[~df["message_id"].isin(reserved_ids)]
    if remaining_slots == 0 or pool.empty:
        out = reserved.sample(n=min(n, len(reserved)), random_state=random_state)
        return out.reset_index(drop=True)

    # Proportional stratified fill from remaining pool
    strata = pool.groupby(["channel_name", "year", "length_bucket"], observed=True)
    weights = strata.size()
    if weights.sum() == 0:
        fill = pool.sample(n=min(remaining_slots, len(pool)), random_state=random_state)
    else:
        alloc = (weights / weights.sum() * remaining_slots).round().astype(int)
        # Fix rounding drift
        drift = remaining_slots - int(alloc.sum())
        if drift != 0:
            order = weights.sort_values(ascending=False).index
            for key in order:
                if drift == 0:
                    break
                alloc[key] += 1 if drift > 0 else -1
                drift += -1 if drift > 0 else 1

        parts: list[pd.DataFrame] = []
        for key, count in alloc.items():
            if count <= 0:
                continue
            group = strata.get_group(key)
            parts.append(group.sample(n=min(count, len(group)), random_state=random_state))
        fill = pd.concat(parts, ignore_index=True) if parts else pool.iloc[0:0]

    combined = pd.concat([reserved, fill], ignore_index=True)
    if len(combined) > n:
        combined = combined.sample(n=n, random_state=random_state)
    elif len(combined) < n:
        extra_pool = df[~df["message_id"].isin(combined["message_id"])]
        need = min(n - len(combined), len(extra_pool))
        if need:
            combined = pd.concat(
                [combined, extra_pool.sample(n=need, random_state=random_state)],
                ignore_index=True,
            )
    return combined.reset_index(drop=True)
