#!/usr/bin/env python3
"""Build a shareable executive-summary HTML (Discord-themed) from analysis tables."""

from __future__ import annotations

import html as html_lib
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
ANALYSIS = DATA / "analysis"
OUT = ROOT / "reports" / "executive_summary.html"

# Fixed methodology examples (from the live corpus)
CONTEXT_SARCASTIC = {
    "channel": "general",
    "lines": [
        ("goatgoatgoatgoat", "And the dude said it represents all the undiagnosed autistic kids"),
        ("goatgoatgoatgoat", "The guy selling it"),
        ("goatgoatgoatgoat", "That's meeee!!"),
        ("conkeastador", "👏 Lets 👏 watch 👏 an 👏 anime! 👏 👏 Lets 👏 watch 👏 an 👏 anime! 👏", True),
    ],
    "result": {
        "polarity": "negative",
        "sarcasm": True,
        "emotions": "annoyance",
        "rationale": "Spamming to drown out the previous conversation.",
    },
}

CONTEXT_ORDINARY = {
    "channel": "general",
    "lines": [
        ("Moey", "1706 artists"),
        ("goatgoatgoatgoat", "I got 1987 my most proud stat"),
        ("goatgoatgoatgoat", "Oh this album sounds great already"),
        ("Moey", "Yea I goofed this year. Like half as many mins as usual", True),
    ],
    "result": {
        "polarity": "mixed",
        "sarcasm": False,
        "emotions": "sadness,amusement",
        "rationale": "Self-deprecating note about listening stats.",
    },
}


def pct(n: float, total: float) -> str:
    if not total:
        return "0%"
    return f"{100 * n / total:.1f}%"


def fmt_n(n: float | int) -> str:
    return f"{int(n):,}"


def load(name: str) -> pd.DataFrame:
    return pd.read_parquet(ANALYSIS / f"{name}.parquet")


EXCLUDED_AUTHORS = frozenset({"mee6", "peter dinklage", "unknown"})


def clean_name(s: str) -> str:
    return str(s).split(":")[0].strip() or str(s)


def is_excluded_author(name: object) -> bool:
    if name is None or (isinstance(name, float) and pd.isna(name)):
        return True
    key = clean_name(name).strip().lower()
    return (not key) or key in EXCLUDED_AUTHORS or key == "nan"


def filter_real_members(df: pd.DataFrame, author_col: str = "author") -> pd.DataFrame:
    return df.loc[~df[author_col].map(is_excluded_author)].copy()


def member_polarity_compare(min_messages: int = 500) -> pd.DataFrame:
    """Lifetime vs last-12-month mean polarity for high-volume members."""
    sr = pd.read_parquet(
        DATA / "sentiment_results.parquet",
        columns=["message_id", "polarity_score"],
    )
    msg = pd.read_parquet(
        DATA / "messages.parquet",
        columns=["message_id", "created_at", "author_name", "user_name"],
    )
    sr["message_id"] = sr["message_id"].astype(str)
    msg["message_id"] = msg["message_id"].astype(str)
    df = msg.merge(sr, on="message_id", how="inner")
    df["created_at"] = pd.to_datetime(df["created_at"], utc=True)
    df["author"] = df["author_name"].fillna(df["user_name"]).astype(str)
    cutoff = df["created_at"].max() - pd.Timedelta(days=365)

    life = df.groupby("author", as_index=False).agg(
        n=("polarity_score", "size"),
        life=("polarity_score", "mean"),
    )
    recent = (
        df[df["created_at"] >= cutoff]
        .groupby("author", as_index=False)
        .agg(n_year=("polarity_score", "size"), year=("polarity_score", "mean"))
    )
    out = life.merge(recent, on="author", how="left")
    out = filter_real_members(out)
    out = out[out["n"] >= min_messages].sort_values("life", ascending=False)
    return out.reset_index(drop=True)


def score_to_pct(score: float, lo: float, hi: float) -> float:
    if hi <= lo:
        return 50.0
    return 100.0 * (float(score) - lo) / (hi - lo)


def render_polarity_strips(members: pd.DataFrame) -> str:
    """One horizontal polarity axis per member, with lifetime + last-year dots."""
    if members.empty:
        return "<p class=\"section-lead\">No members met the volume threshold.</p>"

    scores = list(members["life"].astype(float))
    scores += [float(x) for x in members["year"].dropna()]
    lo = min(scores + [-0.05])
    hi = max(scores + [0.05])
    pad = max(0.02, 0.08 * (hi - lo))
    lo, hi = lo - pad, hi + pad
    zero_pct = score_to_pct(0.0, lo, hi)

    rows = []
    for r in members.itertuples():
        name = html_lib.escape(clean_name(r.author))
        life = float(r.life)
        life_pct = score_to_pct(life, lo, hi)
        year_html = ""
        year_label = "n/a"
        if pd.notna(r.year):
            year = float(r.year)
            year_pct = score_to_pct(year, lo, hi)
            year_html = (
                f'<span class="polar-dot year" style="left:{year_pct:.2f}%" '
                f'title="Last year: {year:+.3f}"></span>'
            )
            year_label = f"{year:+.3f}"
        rows.append(
            f"""
            <div class="polar-row">
              <div class="polar-name">
                <strong>{name}</strong>
                <small>{fmt_n(r.n)} msgs</small>
              </div>
              <div class="polar-track" aria-label="Polarity for {name}">
                <div class="polar-zero" style="left:{zero_pct:.2f}%"></div>
                <span class="polar-dot life" style="left:{life_pct:.2f}%"
                      title="Lifetime: {life:+.3f}"></span>
                {year_html}
              </div>
              <div class="polar-vals">
                <span class="life-val">{life:+.3f}</span>
                <span class="year-val">{year_label}</span>
              </div>
            </div>"""
        )

    return f"""
    <div class="polar-legend">
      <span><i class="swatch life"></i> Lifetime</span>
      <span><i class="swatch year"></i> Last 12 months</span>
      <span class="polar-scale">Scale {lo:+.2f} to {hi:+.2f}</span>
    </div>
    <div class="polar-rows">{"".join(rows)}</div>"""


def render_context_card(example: dict, title: str) -> str:
    bubbles = []
    for item in example["lines"]:
        author, content = item[0], item[1]
        is_target = len(item) > 2 and item[2]
        cls = "msg target" if is_target else "msg"
        badge = '<span class="target-badge">TARGET</span>' if is_target else ""
        bubbles.append(
            f"""<div class="{cls}">
              <div class="msg-author">{html_lib.escape(author)} {badge}</div>
              <div class="msg-body">{html_lib.escape(content)}</div>
            </div>"""
        )
    r = example["result"]
    return f"""
    <div class="context-card">
      <div class="context-head">
        <span class="channel-pill">{html_lib.escape(example["channel"])}</span>
        <span class="context-title">{html_lib.escape(title)}</span>
      </div>
      <div class="msg-stack">{"".join(bubbles)}</div>
      <div class="model-out">
        <div class="model-out-label">Model output</div>
        <div class="chips">
          <span class="chip">{html_lib.escape(r["polarity"])}</span>
          <span class="chip {"chip-warn" if r["sarcasm"] else ""}">sarcasm: {str(r["sarcasm"]).lower()}</span>
          <span class="chip">{html_lib.escape(r["emotions"])}</span>
        </div>
        <p class="rationale">{html_lib.escape(r["rationale"])}</p>
      </div>
    </div>"""


def main() -> None:
    overview = load("overview").iloc[0]
    date_range = load("date_range").iloc[0]
    polarity = load("polarity")
    toxicity = load("toxicity")
    by_channel = load("by_channel").dropna(subset=["channel_name"])
    by_member = filter_real_members(load("by_member").dropna(subset=["author"]))
    by_year = load("by_year")
    by_month = load("by_month_recent")
    sarcastic = load("sarcastic_sample").dropna(subset=["content"])
    member_compare = member_polarity_compare(500)

    total = int(overview["n"])
    pol_map = {r.polarity: int(r.n) for r in polarity.itertuples()}
    tox_map = {r.toxicity: int(r.n) for r in toxicity.itertuples()}

    chillest_channel = by_channel.sort_values("avg_score", ascending=False).iloc[0]
    saltiest_channel = by_channel.sort_values("avg_score", ascending=True).iloc[0]
    sarcasm_channel = by_channel.sort_values("sarcasm_rate", ascending=False).iloc[0]
    volume_kings = by_member.sort_values("n", ascending=False).head(5)
    # Rate among higher-volume members so small samples don't dominate.
    toxic_candidates = by_member[by_member["n"] >= 500]
    most_toxic = toxic_candidates.sort_values("tox_rate", ascending=False).iloc[0]
    polarity_strips_html = render_polarity_strips(member_compare)

    chart_payload = {
        "polarity": {
            "labels": [r.polarity for r in polarity.itertuples()],
            "values": [int(r.n) for r in polarity.itertuples()],
        },
        "years": {
            "labels": [str(int(r.yr)) for r in by_year.itertuples()],
            "scores": [round(float(r.avg_score), 4) for r in by_year.itertuples()],
            "counts": [int(r.n) for r in by_year.itertuples()],
        },
        "months": {
            "labels": [str(r.ym) for r in by_month.itertuples()],
            "scores": [round(float(r.avg_score), 4) for r in by_month.itertuples()],
        },
        "channels": {
            "labels": [f"#{r.channel_name}" for r in by_channel.itertuples()][:12],
            "scores": [round(float(r.avg_score), 4) for r in by_channel.itertuples()][:12],
        },
    }

    member_rows = "".join(
        f"""<tr>
          <td>{html_lib.escape(clean_name(r.author))}</td>
          <td class="num">{fmt_n(r.n)}</td>
          <td class="num {'up' if r.avg_score >= 0 else 'down'}">{r.avg_score:+.3f}</td>
          <td class="num">{100 * r.sarcasm_rate:.1f}%</td>
          <td class="num">{100 * r.tox_rate:.1f}%</td>
          <td class="num">{100 * r.pos_rate:.0f}%</td>
          <td class="num">{100 * r.neg_rate:.0f}%</td>
        </tr>"""
        for r in by_member.sort_values("tox_rate", ascending=False).itertuples()
    )

    channel_rows = "".join(
        f"""<tr>
          <td>#{html_lib.escape(str(r.channel_name))}</td>
          <td class="num">{fmt_n(r.n)}</td>
          <td class="num {'up' if r.avg_score >= 0 else 'down'}">{r.avg_score:+.3f}</td>
          <td class="num">{100 * r.sarcasm_rate:.1f}%</td>
          <td class="num">{fmt_n(r.pos)}</td>
          <td class="num">{fmt_n(r.neg)}</td>
        </tr>"""
        for r in by_channel.itertuples()
    )

    quote_cards = ""
    for r in sarcastic.head(8).itertuples():
        author = clean_name(r.author) if pd.notna(r.author) else "unknown"
        ch = r.channel_name if pd.notna(r.channel_name) else "?"
        quote_cards += f"""
        <figure class="quote">
          <blockquote>“{html_lib.escape(str(r.content))}”</blockquote>
          <figcaption>
            <span>{html_lib.escape(author)}</span>
            · #{html_lib.escape(str(ch))}
            · <em>{html_lib.escape(str(r.rationale))}</em>
          </figcaption>
        </figure>"""

    context_sarcastic_html = render_context_card(
        CONTEXT_SARCASTIC, "Sarcasm is resolved from prior messages"
    )
    context_ordinary_html = render_context_card(
        CONTEXT_ORDINARY, "Ordinary chat where prior turns still inform the score"
    )

    page = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>The Vibes Report · Server Sentiment Summary</title>
<link rel="preconnect" href="https://fonts.googleapis.com" />
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin />
<link href="https://fonts.googleapis.com/css2?family=Noto+Sans:wght@400;500;600;700&display=swap" rel="stylesheet" />
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
<style>
  :root {{
    /* Discord brand + app chrome */
    --blurple: #5865F2;
    --blurple-dark: #4752C4;
    --green: #57F287;
    --yellow: #FEE75C;
    --fuchsia: #EB459E;
    --red: #ED4245;
    --bg-primary: #1E1F22;
    --bg-secondary: #2B2D31;
    --bg-tertiary: #313338;
    --bg-modifier: #35373C;
    --text-normal: #DBDEE1;
    --text-muted: #949BA4;
    --text-strong: #F2F3F5;
    --channel: #949BA4;
    --interactive: #B5BAC1;
    --header: #F2F3F5;
  }}
  * {{ box-sizing: border-box; }}
  body {{
    margin: 0;
    font-family: "Noto Sans", "gg sans", "Helvetica Neue", Helvetica, Arial, sans-serif;
    background: var(--bg-primary);
    color: var(--text-normal);
    line-height: 1.55;
  }}
  .app {{
    display: grid;
    grid-template-columns: 72px 1fr;
    min-height: 100vh;
  }}
  .rail {{
    background: #111214;
    padding: 12px 0;
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 8px;
    border-right: 1px solid #1a1b1e;
  }}
  .rail-orb {{
    width: 48px;
    height: 48px;
    border-radius: 50%;
    background: var(--blurple);
    color: white;
    display: grid;
    place-items: center;
    font-weight: 700;
    font-size: 0.85rem;
    transition: border-radius .15s;
  }}
  .rail-orb:hover {{ border-radius: 16px; }}
  .rail-orb.home {{ background: var(--green); color: #111; }}
  .main {{
    background: var(--bg-secondary);
  }}
  .topbar {{
    height: 48px;
    border-bottom: 1px solid #1f2023;
    display: flex;
    align-items: center;
    padding: 0 20px;
    gap: 8px;
    background: var(--bg-secondary);
    position: sticky;
    top: 0;
    z-index: 5;
  }}
  .topbar .hash {{ color: var(--channel); font-size: 1.35rem; font-weight: 700; }}
  .topbar h1 {{
    margin: 0;
    font-size: 1rem;
    font-weight: 600;
    color: var(--header);
  }}
  .wrap {{ max-width: 980px; margin: 0 auto; padding: 28px 22px 72px; }}
  .hero {{
    background: linear-gradient(135deg, #5865F2 0%, #4752C4 45%, #3c45a5 100%);
    border-radius: 12px;
    padding: 28px 26px;
    margin-bottom: 28px;
    color: #fff;
    box-shadow: 0 8px 24px rgba(0,0,0,.35);
  }}
  .hero .kicker {{
    font-size: 0.75rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    opacity: 0.85;
    font-weight: 700;
  }}
  .hero h2 {{
    margin: 8px 0 10px;
    font-size: clamp(1.8rem, 4vw, 2.4rem);
    line-height: 1.15;
    font-weight: 700;
  }}
  .hero p {{ margin: 0; max-width: 56ch; opacity: 0.95; }}
  .hero .meta {{
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    margin-top: 18px;
  }}
  .pill {{
    background: rgba(0,0,0,.22);
    border: 1px solid rgba(255,255,255,.12);
    border-radius: 999px;
    padding: 5px 11px;
    font-size: 0.82rem;
    font-weight: 600;
  }}
  section {{ margin: 36px 0 44px; }}
  h3.section-title {{
    margin: 0 0 6px;
    font-size: 1.25rem;
    color: var(--header);
    font-weight: 700;
  }}
  .section-lead {{
    color: var(--text-muted);
    margin: 0 0 18px;
    max-width: 65ch;
  }}
  .panel {{
    background: var(--bg-tertiary);
    border-radius: 8px;
    padding: 16px 18px;
    border: 1px solid rgba(0,0,0,.25);
  }}
  .panel + .panel {{ margin-top: 12px; }}
  .panel h4 {{
    margin: 0 0 10px;
    font-size: 0.95rem;
    color: var(--header);
  }}
  .grid-2 {{
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 12px;
  }}
  @media (max-width: 820px) {{
    .app {{ grid-template-columns: 1fr; }}
    .rail {{ display: none; }}
    .grid-2 {{ grid-template-columns: 1fr; }}
  }}
  .method-bits {{
    margin: 0;
    padding-left: 1.15rem;
    color: var(--text-normal);
    font-size: 0.95rem;
  }}
  .method-bits li {{ margin: 6px 0; }}
  .method-bits strong {{ color: var(--text-strong); }}
  .schema {{
    font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
    font-size: 0.78rem;
    background: #1E1F22;
    border-radius: 8px;
    padding: 14px 16px;
    overflow-x: auto;
    color: #B5BAC1;
    border: 1px solid #111214;
    line-height: 1.45;
    margin: 0;
  }}
  .schema .k {{ color: #00A8FC; }}
  .schema .s {{ color: var(--green); }}
  .schema .c {{ color: #6D6F78; }}
  .flow {{
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 8px;
    margin: 14px 0 4px;
  }}
  @media (max-width: 820px) {{
    .flow {{ grid-template-columns: 1fr 1fr; }}
  }}
  .flow-step {{
    background: var(--bg-modifier);
    border-radius: 8px;
    padding: 12px;
    text-align: center;
  }}
  .flow-step .n {{
    width: 22px;
    height: 22px;
    border-radius: 50%;
    background: var(--blurple);
    color: #fff;
    display: inline-grid;
    place-items: center;
    font-size: 0.75rem;
    font-weight: 700;
    margin-bottom: 6px;
  }}
  .flow-step .t {{ font-size: 0.82rem; font-weight: 600; color: var(--text-strong); }}
  .context-card {{
    background: var(--bg-primary);
    border-radius: 8px;
    overflow: hidden;
    border: 1px solid #111214;
  }}
  .context-head {{
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 10px 12px;
    background: var(--bg-tertiary);
    border-bottom: 1px solid #1f2023;
  }}
  .channel-pill {{
    color: var(--text-strong);
    font-weight: 700;
    font-size: 0.9rem;
  }}
  .channel-pill::before {{
    content: "#";
    color: var(--channel);
    margin-right: 1px;
  }}
  .context-title {{ color: var(--text-muted); font-size: 0.82rem; }}
  .msg-stack {{ padding: 8px 0; }}
  .msg {{
    padding: 8px 14px;
  }}
  .msg:hover {{ background: rgba(79,84,92,0.16); }}
  .msg.target {{
    background: rgba(88, 101, 242, 0.14);
    border-left: 3px solid var(--blurple);
  }}
  .msg-author {{
    font-weight: 600;
    color: var(--header);
    font-size: 0.92rem;
    margin-bottom: 2px;
  }}
  .target-badge {{
    display: inline-block;
    margin-left: 6px;
    background: var(--blurple);
    color: #fff;
    font-size: 0.65rem;
    font-weight: 700;
    letter-spacing: 0.04em;
    padding: 2px 6px;
    border-radius: 4px;
    vertical-align: middle;
  }}
  .msg-body {{ color: var(--text-normal); white-space: pre-wrap; word-break: break-word; }}
  .model-out {{
    border-top: 1px solid #1f2023;
    padding: 12px 14px 14px;
    background: #232428;
  }}
  .model-out-label {{
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: var(--text-muted);
    font-weight: 700;
    margin-bottom: 8px;
  }}
  .chips {{ display: flex; flex-wrap: wrap; gap: 6px; }}
  .chip {{
    background: var(--bg-modifier);
    color: var(--text-strong);
    border-radius: 4px;
    padding: 3px 8px;
    font-size: 0.78rem;
    font-weight: 600;
  }}
  .chip-warn {{ background: rgba(254, 231, 92, 0.15); color: var(--yellow); }}
  .rationale {{
    margin: 8px 0 0;
    color: var(--text-muted);
    font-size: 0.88rem;
  }}
  .grid-4 {{
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 10px;
  }}
  @media (max-width: 800px) {{
    .grid-4 {{ grid-template-columns: 1fr 1fr; }}
  }}
  .stat {{
    background: var(--bg-tertiary);
    border-radius: 8px;
    padding: 14px;
    border: 1px solid rgba(0,0,0,.2);
  }}
  .stat .label {{
    color: var(--text-muted);
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    font-weight: 700;
  }}
  .stat .value {{
    margin-top: 6px;
    font-size: 1.55rem;
    font-weight: 700;
    color: var(--header);
  }}
  .stat .sub {{ color: var(--text-muted); font-size: 0.8rem; margin-top: 2px; }}
  .callouts {{
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 10px;
  }}
  @media (max-width: 1100px) {{
    .callouts {{ grid-template-columns: 1fr 1fr; }}
  }}
  @media (max-width: 860px) {{
    .callouts {{ grid-template-columns: 1fr; }}
  }}
  .callout {{
    background: var(--bg-tertiary);
    border-radius: 8px;
    padding: 16px;
    border-left: 4px solid var(--blurple);
  }}
  .callout.hot {{ border-left-color: var(--red); }}
  .callout.green {{ border-left-color: var(--green); }}
  .callout .tag {{
    display: inline-block;
    font-size: 0.68rem;
    font-weight: 700;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    color: var(--blurple);
    margin-bottom: 6px;
  }}
  .callout.hot .tag {{ color: var(--red); }}
  .callout.green .tag {{ color: var(--green); }}
  .callout h4 {{
    margin: 0 0 6px;
    font-size: 1.15rem;
    color: var(--header);
  }}
  .callout p {{ margin: 0; color: var(--text-muted); font-size: 0.9rem; }}
  canvas {{ max-height: 280px; }}
  table {{
    width: 100%;
    border-collapse: collapse;
    font-size: 0.9rem;
  }}
  th, td {{
    text-align: left;
    padding: 9px 8px;
    border-bottom: 1px solid #3f4147;
  }}
  th {{
    color: var(--text-muted);
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    font-weight: 700;
  }}
  td.num {{ font-variant-numeric: tabular-nums; text-align: right; }}
  td.up {{ color: var(--green); }}
  td.down {{ color: var(--red); }}
  ol.rank {{ margin: 0; padding-left: 1.2rem; }}
  ol.rank li {{ margin: 8px 0; }}
  ol.rank span {{ color: var(--blurple); margin-left: 6px; font-weight: 700; }}
  ol.rank small {{ color: var(--text-muted); margin-left: 6px; }}
  .polar-legend {{
    display: flex;
    flex-wrap: wrap;
    gap: 14px;
    align-items: center;
    margin: 0 0 14px;
    color: var(--text-muted);
    font-size: 0.85rem;
  }}
  .polar-legend .swatch {{
    display: inline-block;
    width: 10px;
    height: 10px;
    border-radius: 50%;
    margin-right: 6px;
    vertical-align: middle;
  }}
  .polar-legend .swatch.life {{ background: var(--blurple); }}
  .polar-legend .swatch.year {{ background: var(--yellow); box-shadow: 0 0 0 1px rgba(0,0,0,.35); }}
  .polar-legend .polar-scale {{ margin-left: auto; font-variant-numeric: tabular-nums; }}
  .polar-rows {{
    display: flex;
    flex-direction: column;
    gap: 10px;
  }}
  .polar-row {{
    display: grid;
    grid-template-columns: 150px 1fr 110px;
    gap: 12px;
    align-items: center;
  }}
  @media (max-width: 700px) {{
    .polar-row {{ grid-template-columns: 1fr; gap: 6px; }}
    .polar-legend .polar-scale {{ margin-left: 0; }}
  }}
  .polar-name strong {{
    display: block;
    color: var(--header);
    font-size: 0.95rem;
  }}
  .polar-name small {{ color: var(--text-muted); font-size: 0.78rem; }}
  .polar-track {{
    position: relative;
    height: 14px;
    border-radius: 999px;
    background: linear-gradient(90deg, rgba(237,66,69,.35), rgba(148,155,164,.18) 50%, rgba(87,242,135,.35));
    border: 1px solid #3f4147;
  }}
  .polar-zero {{
    position: absolute;
    top: -3px;
    bottom: -3px;
    width: 1px;
    background: rgba(242,243,245,.45);
    transform: translateX(-50%);
  }}
  .polar-dot {{
    position: absolute;
    top: 50%;
    width: 12px;
    height: 12px;
    border-radius: 50%;
    transform: translate(-50%, -50%);
    box-shadow: 0 0 0 2px var(--bg-tertiary);
  }}
  .polar-dot.life {{ background: var(--blurple); z-index: 2; }}
  .polar-dot.year {{ background: var(--yellow); z-index: 3; }}
  .polar-vals {{
    display: flex;
    justify-content: flex-end;
    gap: 10px;
    font-variant-numeric: tabular-nums;
    font-size: 0.85rem;
    font-weight: 600;
  }}
  .polar-vals .life-val {{ color: #a5b4fc; }}
  .polar-vals .year-val {{ color: var(--yellow); }}
  .quotes {{
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 10px;
  }}
  @media (max-width: 800px) {{
    .quotes {{ grid-template-columns: 1fr; }}
  }}
  .quote {{
    margin: 0;
    background: var(--bg-tertiary);
    border-radius: 8px;
    padding: 14px;
  }}
  .quote blockquote {{
    margin: 0 0 8px;
    color: var(--text-strong);
    font-size: 0.98rem;
  }}
  .quote figcaption {{ color: var(--text-muted); font-size: 0.8rem; }}
  .divider {{
    height: 1px;
    background: #3f4147;
    margin: 40px 0 8px;
  }}
  .findings-banner {{
    display: flex;
    align-items: center;
    gap: 10px;
    margin: 8px 0 20px;
    color: var(--text-muted);
    font-size: 0.9rem;
  }}
  .findings-banner::before {{
    content: "";
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: var(--green);
    box-shadow: 0 0 0 4px rgba(87,242,135,.15);
  }}
  footer {{
    margin-top: 40px;
    padding-top: 16px;
    border-top: 1px solid #3f4147;
    color: var(--text-muted);
    font-size: 0.82rem;
  }}
  .scroll {{ overflow-x: auto; }}
  .toc {{
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    margin: 0 0 8px;
  }}
  .toc a {{
    color: var(--interactive);
    text-decoration: none;
    background: var(--bg-tertiary);
    border-radius: 16px;
    padding: 6px 12px;
    font-size: 0.82rem;
    font-weight: 600;
  }}
  .toc a:hover {{ color: #fff; background: var(--blurple); }}
</style>
</head>
<body>
<div class="app">
  <aside class="rail" aria-hidden="true">
    <div class="rail-orb home">D</div>
    <div class="rail-orb">SA</div>
  </aside>
  <div class="main">
    <div class="topbar">
      <span class="hash">#</span>
      <h1>sentiment-report</h1>
    </div>
    <div class="wrap">
      <header class="hero">
        <div class="kicker">Discord archive · LLM sentiment</div>
        <h2>The vibes report</h2>
        <p class="lede">
          An executive summary of ~{fmt_n(total)} scored messages across
          {int(date_range.members)} members and {int(date_range.channels)} channels,
          from {str(date_range.mn)[:10]} to {str(date_range.mx)[:10]}.
          Context-aware Gemini analysis was applied, with preceding channel messages included.
        </p>
        <div class="meta">
          <span class="pill">{pct(pol_map.get("positive", 0), total)} positive</span>
          <span class="pill">{pct(pol_map.get("negative", 0), total)} negative</span>
          <span class="pill">{pct(pol_map.get("neutral", 0), total)} neutral</span>
          <span class="pill">{100 * float(overview.sarcasm_rate):.1f}% sarcastic</span>
        </div>
      </header>

      <nav class="toc">
        <a href="#methodology">Methodology</a>
        <a href="#schema">Output format</a>
        <a href="#findings">Findings</a>
        <a href="#people">People</a>
        <a href="#channels">Channels</a>
        <a href="#quotes">Sarcasm</a>
      </nav>

      <!-- ========== METHODOLOGY ========== -->
      <section id="methodology">
        <h3 class="section-title">1. Methodology</h3>
        <p class="section-lead">
          Non-empty Discord messages were scored with Gemini 3.1 Flash-Lite.
          Up to three prior same-channel messages were included as context; only the
          <span class="chip" style="vertical-align:middle">TARGET</span> line was labeled.
        </p>
        <div class="flow">
          <div class="flow-step"><div class="n">1</div><div class="t">Export archive</div></div>
          <div class="flow-step"><div class="n">2</div><div class="t">Build context windows</div></div>
          <div class="flow-step"><div class="n">3</div><div class="t">Score with Gemini</div></div>
          <div class="flow-step"><div class="n">4</div><div class="t">Store + analyze</div></div>
        </div>
        <div class="panel" style="margin-top:14px">
          <ul class="method-bits">
            <li><strong>{fmt_n(total)}</strong> of ~265k scorable messages were scored (~95%).</li>
            <li>Scores were stored by <code>message_id</code> so historical results are not re-billed.</li>
          </ul>
        </div>
        <div class="grid-2" style="margin-top:12px">
          {context_sarcastic_html}
          {context_ordinary_html}
        </div>
      </section>

      <section id="schema">
        <h3 class="section-title">Output format</h3>
        <p class="section-lead">
          Each scored message was stored as one JSON object. A typical readout looks like this:
        </p>
        <div class="panel">
          <pre class="schema"><span class="c">// one row per Discord message</span>
{{
  <span class="k">"message_id"</span>: <span class="s">"1407586567366905948"</span>,
  <span class="k">"polarity"</span>: <span class="s">"positive | negative | neutral | mixed"</span>,
  <span class="k">"polarity_score"</span>: <span class="s">-1.0 to 1.0</span>,
  <span class="k">"emotions"</span>: <span class="s">"joy,amusement"</span>,       <span class="c">// 1 to 3 from fixed set</span>
  <span class="k">"sarcasm"</span>: <span class="s">true</span>,
  <span class="k">"toxicity"</span>: <span class="s">"none | mild | moderate | severe"</span>,
  <span class="k">"directed_at"</span>: <span class="s">"general | person | group | self | topic"</span>,
  <span class="k">"confidence"</span>: <span class="s">0.0 to 1.0</span>,
  <span class="k">"rationale"</span>: <span class="s">"up to 15 words"</span>,
  <span class="k">"model"</span>: <span class="s">"gemini-3.1-flash-lite"</span>,
  <span class="k">"scored_at"</span>: <span class="s">"2026-07-18T..."</span>
}}</pre>
        </div>
      </section>

      <div class="divider"></div>
      <div class="findings-banner">Aggregate findings for the server are presented below.</div>

      <!-- ========== FINDINGS ========== -->
      <section id="findings">
        <h3 class="section-title">2. Headline findings</h3>
        <p class="section-lead">Overall polarity is observed to be slightly positive on average. The distribution is neither uniformly warm nor uniformly negative.</p>
        <div class="grid-4">
          <div class="stat">
            <div class="label">Messages scored</div>
            <div class="value">{fmt_n(total)}</div>
            <div class="sub">~95% of non-empty archive</div>
          </div>
          <div class="stat">
            <div class="label">Mean polarity</div>
            <div class="value">{float(overview.avg_score):+.3f}</div>
            <div class="sub">scale -1 to +1</div>
          </div>
          <div class="stat">
            <div class="label">Sarcasm rate</div>
            <div class="value">{100 * float(overview.sarcasm_rate):.1f}%</div>
            <div class="sub">with local context</div>
          </div>
          <div class="stat">
            <div class="label">Harsh toxicity</div>
            <div class="value">{pct(tox_map.get("moderate", 0) + tox_map.get("severe", 0), total)}</div>
            <div class="sub">moderate + severe</div>
          </div>
        </div>
      </section>

      <section>
        <h3 class="section-title">Where the energy lives</h3>
        <p class="section-lead">Distinct channel-level tone profiles are observed. Warmer tone is concentrated in a few channels; colder tone is concentrated in one.</p>
        <div class="callouts">
          <div class="callout green">
            <div class="tag">Chillest</div>
            <h4>#{html_lib.escape(str(chillest_channel.channel_name))}</h4>
            <p>The highest average polarity among active channels is observed here ({chillest_channel.avg_score:+.3f}).</p>
          </div>
          <div class="callout hot">
            <div class="tag">Salt mine</div>
            <h4>#{html_lib.escape(str(saltiest_channel.channel_name))}</h4>
            <p>The lowest average polarity among active channels is observed here ({saltiest_channel.avg_score:+.3f}).</p>
          </div>
          <div class="callout">
            <div class="tag">Most sarcastic</div>
            <h4>#{html_lib.escape(str(sarcasm_channel.channel_name))}</h4>
            <p>{100 * sarcasm_channel.sarcasm_rate:.1f}% of messages in this channel were flagged as sarcastic.</p>
          </div>
          <div class="callout hot">
            <div class="tag">Most toxic</div>
            <h4>{html_lib.escape(clean_name(most_toxic.author))}</h4>
            <p>{100 * most_toxic.tox_rate:.1f}% of their {fmt_n(most_toxic.n)} messages were flagged toxic (mild+).</p>
          </div>
        </div>
      </section>

      <section>
        <div class="grid-2">
          <div class="panel">
            <h4>Polarity mix</h4>
            <canvas id="polarityChart"></canvas>
          </div>
          <div class="panel">
            <h4>Mood by year</h4>
            <canvas id="yearChart"></canvas>
          </div>
        </div>
      </section>

      <section>
        <div class="panel">
          <h4>Recent mood (monthly mean polarity, 2022+)</h4>
          <canvas id="monthChart"></canvas>
        </div>
      </section>

      <section id="people">
        <h3 class="section-title">3. People of the server</h3>
        <p class="section-lead">
          Mean polarity is shown for every member with 500 or more scored messages.
          Lifetime average is compared with the last 12 months on one shared axis.
        </p>
        <div class="panel">
          {polarity_strips_html}
        </div>
      </section>

      <section>
        <div class="panel">
          <h4>Channels by polarity</h4>
          <canvas id="channelChart"></canvas>
        </div>
      </section>

      <section>
        <h3 class="section-title">Member leaderboard</h3>
        <p class="section-lead">Members with 200 or more scored messages, ranked by toxicity rate.</p>
        <div class="panel scroll">
          <table>
            <thead>
              <tr>
                <th>Member</th><th>Msgs</th><th>Avg score</th><th>Sarcasm</th><th>Tox%</th><th>Pos%</th><th>Neg%</th>
              </tr>
            </thead>
            <tbody>{member_rows}</tbody>
          </table>
        </div>
      </section>

      <section id="channels">
        <h3 class="section-title">4. Channel leaderboard</h3>
        <div class="panel scroll">
          <table>
            <thead>
              <tr>
                <th>Channel</th><th>Msgs</th><th>Avg score</th><th>Sarcasm</th><th>Pos</th><th>Neg</th>
              </tr>
            </thead>
            <tbody>{channel_rows}</tbody>
          </table>
        </div>
      </section>

      <section id="quotes">
        <h3 class="section-title">5. Sarcasm examples</h3>
        <p class="section-lead">High-confidence sarcastic lines are shown below. Preceding channel messages were provided as context during scoring.</p>
        <div class="quotes">{quote_cards}</div>
      </section>

      <section>
        <h3 class="section-title">Takeaways</h3>
        <div class="panel">
          <ol class="rank">
            <li><strong>#general</strong> accounts for ~{pct(int(by_channel.iloc[0].n), total)} of scored traffic and is near the server-wide average polarity.</li>
            <li>The warmest active channel is <strong>#{html_lib.escape(str(chillest_channel.channel_name))}</strong>. The coldest is <strong>#{html_lib.escape(str(saltiest_channel.channel_name))}</strong>.</li>
            <li>Mean polarity is <strong>somewhat higher in recent years</strong> than in the 2016 to 2018 period.</li>
            <li>Sarcasm is uncommon (~{100 * float(overview.sarcasm_rate):.1f}%). Context windows were used so ironic readings could still be recovered when they occurred.</li>
            <li>The highest-volume members observed are {", ".join(html_lib.escape(clean_name(a)) for a in volume_kings.author.head(3))}.</li>
            <li>The most toxic high-volume member is <strong>{html_lib.escape(clean_name(most_toxic.author))}</strong> ({100 * most_toxic.tox_rate:.1f}% of messages mild+ toxic).</li>
          </ol>
        </div>
      </section>

      <footer>
        Generated from MySQL <code>message_sentiment</code> · Gemini 3.1 Flash-Lite · 3-message channel context ·
        {fmt_n(total)} messages · regenerate with <code>scripts/build_executive_summary.py</code>
      </footer>
    </div>
  </div>
</div>

<script>
const DATA = {json.dumps(chart_payload)};
const blurple = '#5865F2';
const green = '#57F287';
const red = '#ED4245';
const yellow = '#FEE75C';
const muted = '#949BA4';
const grid = '#3f4147';

Chart.defaults.color = muted;
Chart.defaults.borderColor = grid;
Chart.defaults.font.family = '"Noto Sans", "gg sans", sans-serif';

new Chart(document.getElementById('polarityChart'), {{
  type: 'doughnut',
  data: {{
    labels: DATA.polarity.labels,
    datasets: [{{
      data: DATA.polarity.values,
      backgroundColor: ['#4E5058', green, red, yellow],
      borderWidth: 0
    }}]
  }},
  options: {{
    plugins: {{ legend: {{ position: 'bottom', labels: {{ color: muted }} }} }},
    cutout: '58%'
  }}
}});

new Chart(document.getElementById('yearChart'), {{
  type: 'line',
  data: {{
    labels: DATA.years.labels,
    datasets: [{{
      label: 'Mean polarity',
      data: DATA.years.scores,
      borderColor: blurple,
      backgroundColor: 'rgba(88,101,242,0.15)',
      fill: true,
      tension: 0.25,
      pointRadius: 3,
      pointBackgroundColor: blurple
    }}]
  }},
  options: {{
    scales: {{
      y: {{ suggestedMin: -0.1, suggestedMax: 0.15, grid: {{ color: grid }} }},
      x: {{ grid: {{ display: false }} }}
    }},
    plugins: {{ legend: {{ display: false }} }}
  }}
}});

new Chart(document.getElementById('monthChart'), {{
  type: 'line',
  data: {{
    labels: DATA.months.labels,
    datasets: [{{
      label: 'Mean polarity',
      data: DATA.months.scores,
      borderColor: '#EB459E',
      tension: 0.3,
      pointRadius: 0,
      borderWidth: 2
    }}]
  }},
  options: {{
    scales: {{
      y: {{ grid: {{ color: grid }} }},
      x: {{ ticks: {{ maxTicksLimit: 10 }}, grid: {{ display: false }} }}
    }},
    plugins: {{ legend: {{ display: false }} }}
  }}
}});


new Chart(document.getElementById('channelChart'), {{
  type: 'bar',
  data: {{
    labels: DATA.channels.labels,
    datasets: [{{
      data: DATA.channels.scores,
      backgroundColor: DATA.channels.scores.map(v => v >= 0 ? blurple : red)
    }}]
  }},
  options: {{
    indexAxis: 'y',
    scales: {{
      x: {{ grid: {{ color: grid }} }},
      y: {{ grid: {{ display: false }} }}
    }},
    plugins: {{ legend: {{ display: false }} }}
  }}
}});
</script>
</body>
</html>
"""

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(page, encoding="utf-8")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
