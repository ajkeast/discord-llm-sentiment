# Discord bot — nightly sentiment routine

This note describes how to hook the existing Discord bot (the one that already inserts into `messages`) into an **incremental** sentiment job so you never re-score the historical corpus.

## Goal

| Job | When | Cost |
|-----|------|------|
| One-time backfill | Notebook 03 + Batch API | ~$18–25 once |
| Nightly incremental | Bot cron / scheduled task | Pennies (~900 msgs/month) |

**Idempotency rule:** only score `messages` rows that do **not** already exist in `message_sentiment`.

## Table (already created)

`message_sentiment` is keyed by `message_id` (= `messages.id`) with FK to `messages`.  
Full DDL lives in [`src/sentiment/store.py`](../src/sentiment/store.py) (`CREATE_MESSAGE_SENTIMENT_SQL`).

## Recommended architecture

Do **not** call the LLM inside the Discord message event handler (latency + rate limits + failure modes). Keep the bot’s job as today: write the message row.

Add a separate nightly path:

```
Discord message event
        │
        ▼
   INSERT messages          ← existing bot code
        │
        ▼
   (optional) enqueue message_id

Nightly cron / bot task (e.g. 04:00 local)
        │
        ▼
   SELECT unscored messages (LEFT JOIN message_sentiment IS NULL)
        │
        ▼
   Build 3 prior-message context per channel
        │
        ▼
   Gemini score (live API is fine at ~30 msgs/day)
        │
        ▼
   UPSERT message_sentiment
```

## Option A — Call this repo from the bot (simplest)

If the bot can run Python (or shell out to the venv):

```bash
# cron example — 4:15 AM daily
15 4 * * * cd "/path/to/Sentiment-Analysis" && \
  .venv/bin/python -m sentiment.nightly >> /var/log/sentiment-nightly.log 2>&1
```

The module [`src/sentiment/nightly.py`](../src/sentiment/nightly.py):

1. Loads `.env` (`MYSQL_*`, `GEMINI_API_KEY`)
2. Ensures `message_sentiment` exists
3. Fetches unscored messages
4. Builds context windows (needs recent channel neighbors — see note below)
5. Scores with Gemini 3.1 Flash-Lite
6. Upserts results

**Context neighbors:** for each new message, preceding messages should come from the same channel even if those priors were already scored. `nightly.py` loads a short recent channel history from MySQL for that purpose (not only unscored rows).

## Option B — Bot-native (Node/TS Discord bot)

If the bot is Node-based (common for discord.js):

1. Keep inserting into `messages` as today.
2. Add a scheduled job (node-cron, BullMQ, systemd timer, or host cron hitting an HTTP endpoint).
3. Either:
   - **Shell out** to `python -m sentiment.nightly` (least code duplication), or
   - Reimplement the SQL + Gemini JSON call in TS using the same prompt/schema as `src/sentiment/prompt.py` / `schema.py`.

Prefer shelling out to Python until the prompt/schema stabilize.

### Minimal bot-side pseudocode

```ts
// after successful INSERT into messages — optional fast-path for low volume
// Better: only schedule nightly; don't block message handling.

cron.schedule("15 4 * * *", async () => {
  await execFileAsync(
    "/path/to/Sentiment-Analysis/.venv/bin/python",
    ["-m", "sentiment.nightly"],
    { cwd: "/path/to/Sentiment-Analysis", env: process.env }
  );
});
```

Ensure the bot host has:

- Network access to MySQL + Google AI
- Env vars: `GEMINI_API_KEY`, `MYSQL_HOST`, `MYSQL_USER`, `MYSQL_PASSWORD`, `MYSQL_DATABASE`

## SQL the nightly job relies on

Unscored targets:

```sql
SELECT m.id
FROM messages m
LEFT JOIN message_sentiment s ON s.message_id = m.id
WHERE s.message_id IS NULL
  AND m.content IS NOT NULL
  AND TRIM(m.content) <> '';
```

For context (3 priors in-channel), for a target `(channel_id, created_at, id)`:

```sql
SELECT id, member_id, content, created_at
FROM messages
WHERE channel_id = ?
  AND (created_at < ? OR (created_at = ? AND id < ?))
ORDER BY created_at DESC, id DESC
LIMIT 3;
```

## Failure / resume

- Upserts are `ON DUPLICATE KEY UPDATE` — safe to re-run.
- If Gemini fails mid-night, the next run only picks up rows still missing from `message_sentiment`.
- Never delete `message_sentiment` rows to “refresh” unless you intentionally accept re-billing.

## After backfill checklist

1. Notebook 03 Batch job completes.
2. `SELECT COUNT(*) FROM message_sentiment;` ≈ scorable message count (~265k).
3. Spot-check: `SELECT * FROM message_sentiment ORDER BY scored_at DESC LIMIT 20;`
4. Enable nightly cron / bot schedule.
5. Keep `data/sentiment_results.parquet` as an offline backup, but treat MySQL as source of truth.

## Cost sanity (incremental)

~900 messages/month × ~15/batch ≈ 60 live requests/month → well under **$1/month** at Flash-Lite rates.
