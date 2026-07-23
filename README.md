# Discord LLM Sentiment Analysis

Cost-gated pipeline that scores Discord messages with **Gemini 3.1 Flash-Lite**, using preceding channel context and a rich sentiment schema (polarity, emotions, sarcasm, toxicity).

Built for a private Discord archive (~265k messages): export from MySQL → stratified eval/cost gate → Batch API backfill → nightly incremental scoring.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate   # or: .venv/bin/pip / .venv/bin/python
.venv/bin/pip install -r requirements.txt
.venv/bin/pip install -e .
cp .env.example .env
# Fill in GEMINI_API_KEY and MYSQL_* credentials
```

If your shell aliases `python` to system Python, prefer `.venv/bin/python` / `.venv/bin/pip` explicitly.

**Secrets:** copy `.env.example` → `.env` and never commit `.env`. API keys and MySQL credentials stay local (see `.gitignore`).

## Workflow

1. **`notebooks/01_eda_and_export.ipynb`** — Export MySQL → Parquet, build context windows, EDA.
2. **`notebooks/02_eval_sample.ipynb`** — Stratified 1k sample, cost check, gold-label QA gate.
3. **`notebooks/03_full_run.ipynb`** — Batch API backfill → local parquet **and** upsert into MySQL `message_sentiment`.

Do **not** start the full run until the eval notebook’s cost estimate looks right and gold-label quality is acceptable.

### Database + nightly bot

- Table: `message_sentiment` (PK = `messages.id`). Created in MySQL; DDL in `src/sentiment/store.py`.
- Nightly incremental: `.venv/bin/python -m sentiment.nightly`
- Integration notes for the Discord bot: [`docs/DISCORD_BOT_NIGHTLY.md`](docs/DISCORD_BOT_NIGHTLY.md)

### Executive summary (shareable HTML)

```bash
.venv/bin/python scripts/build_executive_summary.py
open reports/executive_summary.html
```

Regenerates charts/tables from MySQL aggregates in `data/analysis/`. Generated reports are gitignored (they can include private server aggregates).

## Cost ballpark (Gemini 3.1 Flash-Lite)

| Stage | Messages | Approx cost |
|-------|----------|-------------|
| Eval sample | 1,000 | ~$0.30–0.50 |
| Full corpus (live) | ~265k | ~$35–45 |
| Full corpus (Batch API) | ~265k | ~$18–25 |

## Output schema (per message)

```json
{
  "message_id": "123",
  "polarity": "positive|negative|neutral|mixed",
  "polarity_score": -1.0,
  "emotions": ["amusement"],
  "sarcasm": true,
  "toxicity": "none|mild|moderate|severe",
  "directed_at": "general|person|group|self|topic",
  "confidence": 0.0,
  "rationale": "≤15 words"
}
```

## Project layout

```
src/sentiment/     # reusable library
notebooks/         # EDA → eval → full run
scripts/           # report builder + batch recovery
docs/              # bot integration notes
data/              # gitignored local parquet + results
reports/           # gitignored generated HTML
```
