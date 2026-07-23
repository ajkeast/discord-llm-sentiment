"""Eval metrics against gold labels."""

from __future__ import annotations

import pandas as pd


def _norm_bool(series: pd.Series) -> pd.Series:
    mapping = {
        "true": True,
        "false": False,
        "1": True,
        "0": False,
        "yes": True,
        "no": False,
        True: True,
        False: False,
    }
    return series.map(lambda v: mapping.get(v if not isinstance(v, str) else v.strip().lower(), v))


def evaluate_against_gold(
    predictions: pd.DataFrame,
    gold: pd.DataFrame,
) -> dict:
    """
    Compare model predictions to gold labels.

    Gold CSV expects: message_id, gold_polarity, gold_sarcasm, gold_toxicity
    Predictions expect: message_id, polarity, sarcasm, toxicity
    """
    g = gold.copy()
    g["message_id"] = g["message_id"].astype(str)
    p = predictions.copy()
    p["message_id"] = p["message_id"].astype(str)

    merged = g.merge(p, on="message_id", how="inner", suffixes=("_gold", "_pred"))
    # handle column naming from template
    if "gold_polarity" in merged.columns:
        gold_pol = merged["gold_polarity"].astype(str).str.lower().str.strip()
        pred_pol = merged["polarity"].astype(str).str.lower().str.strip()
    else:
        raise ValueError("gold file needs gold_polarity column")

    # drop unlabeled rows
    labeled = gold_pol.ne("") & gold_pol.ne("nan")
    merged = merged.loc[labeled].copy()
    gold_pol = gold_pol.loc[labeled]
    pred_pol = pred_pol.loc[labeled]

    n = len(merged)
    if n == 0:
        return {"n_labeled": 0}

    polarity_acc = float((gold_pol == pred_pol).mean())

    gold_sarc = _norm_bool(merged["gold_sarcasm"])
    pred_sarc = _norm_bool(merged["sarcasm"])
    sarc_mask = gold_sarc.notna()
    sarcasm_acc = (
        float((gold_sarc[sarc_mask] == pred_sarc[sarc_mask]).mean())
        if sarc_mask.any()
        else None
    )

    # simple F1 for sarcasm=true
    sarcasm_f1 = None
    if sarc_mask.any():
        yt = gold_sarc[sarc_mask].astype(bool)
        yp = pred_sarc[sarc_mask].astype(bool)
        tp = int((yt & yp).sum())
        fp = int((~yt & yp).sum())
        fn = int((yt & ~yp).sum())
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        sarcasm_f1 = (
            2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        )

    gold_tox = merged["gold_toxicity"].astype(str).str.lower().str.strip()
    pred_tox = merged["toxicity"].astype(str).str.lower().str.strip()
    tox_mask = gold_tox.ne("") & gold_tox.ne("nan")
    toxicity_acc = (
        float((gold_tox[tox_mask] == pred_tox[tox_mask]).mean())
        if tox_mask.any()
        else None
    )

    return {
        "n_labeled": int(n),
        "polarity_accuracy": round(polarity_acc, 4),
        "sarcasm_accuracy": None if sarcasm_acc is None else round(sarcasm_acc, 4),
        "sarcasm_f1": None if sarcasm_f1 is None else round(sarcasm_f1, 4),
        "toxicity_accuracy": None if toxicity_acc is None else round(toxicity_acc, 4),
    }
