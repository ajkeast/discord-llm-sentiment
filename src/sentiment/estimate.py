"""Token and USD cost estimators for Gemini Flash-Lite runs."""

from __future__ import annotations

from dataclasses import dataclass

from sentiment.prompt import SYSTEM_PROMPT, build_batch_user_prompt

# Gemini 3.1 Flash-Lite published rates (USD / 1M tokens)
FLASH_LITE_INPUT = 0.25
FLASH_LITE_OUTPUT = 1.50
FLASH_LITE_BATCH_INPUT = 0.125
FLASH_LITE_BATCH_OUTPUT = 0.75

# Rough chars→tokens for English Discord text
CHARS_PER_TOKEN = 4.0
# Compact rich JSON per result
DEFAULT_OUTPUT_TOKENS_PER_MSG = 90


@dataclass
class CostEstimate:
    n_messages: int
    batch_size: int
    n_batches: int
    est_input_tokens: int
    est_output_tokens: int
    cost_live_usd: float
    cost_batch_usd: float
    system_tokens: int
    avg_user_tokens_per_batch: float

    def as_dict(self) -> dict:
        return {
            "n_messages": self.n_messages,
            "batch_size": self.batch_size,
            "n_batches": self.n_batches,
            "est_input_tokens": self.est_input_tokens,
            "est_output_tokens": self.est_output_tokens,
            "cost_live_usd": round(self.cost_live_usd, 4),
            "cost_batch_usd": round(self.cost_batch_usd, 4),
            "system_tokens": self.system_tokens,
            "avg_user_tokens_per_batch": round(self.avg_user_tokens_per_batch, 1),
        }


def estimate_tokens(text: str) -> int:
    """Fast heuristic token estimate (no API call)."""
    if not text:
        return 0
    return max(1, int(len(text) / CHARS_PER_TOKEN))


def estimate_cost(
    context_texts: list[str],
    *,
    channel_names: list[str] | None = None,
    message_ids: list[str | int] | None = None,
    batch_size: int = 15,
    output_tokens_per_msg: int = DEFAULT_OUTPUT_TOKENS_PER_MSG,
) -> CostEstimate:
    """Estimate input/output tokens and USD for a list of context windows."""
    n = len(context_texts)
    if n == 0:
        return CostEstimate(0, batch_size, 0, 0, 0, 0.0, 0.0, 0, 0.0)

    channel_names = channel_names or ["unknown"] * n
    message_ids = message_ids or list(range(n))
    system_tokens = estimate_tokens(SYSTEM_PROMPT)

    n_batches = (n + batch_size - 1) // batch_size
    total_user_tokens = 0

    for start in range(0, n, batch_size):
        chunk_ctx = context_texts[start : start + batch_size]
        chunk_ch = channel_names[start : start + batch_size]
        chunk_ids = message_ids[start : start + batch_size]
        items = [
            {
                "message_id": str(mid),
                "channel_name": ch,
                "context_text": ctx,
            }
            for mid, ch, ctx in zip(chunk_ids, chunk_ch, chunk_ctx)
        ]
        total_user_tokens += estimate_tokens(build_batch_user_prompt(items))

    # System prompt sent once per batch (conservative; caching may reduce this)
    est_input = system_tokens * n_batches + total_user_tokens
    est_output = output_tokens_per_msg * n

    cost_live = (
        est_input / 1_000_000 * FLASH_LITE_INPUT
        + est_output / 1_000_000 * FLASH_LITE_OUTPUT
    )
    cost_batch = (
        est_input / 1_000_000 * FLASH_LITE_BATCH_INPUT
        + est_output / 1_000_000 * FLASH_LITE_BATCH_OUTPUT
    )

    return CostEstimate(
        n_messages=n,
        batch_size=batch_size,
        n_batches=n_batches,
        est_input_tokens=est_input,
        est_output_tokens=est_output,
        cost_live_usd=cost_live,
        cost_batch_usd=cost_batch,
        system_tokens=system_tokens,
        avg_user_tokens_per_batch=total_user_tokens / n_batches,
    )


def estimate_from_windows_df(windows_df, *, batch_size: int = 15) -> CostEstimate:
    return estimate_cost(
        windows_df["context_text"].astype(str).tolist(),
        channel_names=windows_df["channel_name"].astype(str).tolist(),
        message_ids=windows_df["message_id"].tolist(),
        batch_size=batch_size,
    )


def cost_from_usage(
    input_tokens: int,
    output_tokens: int,
    *,
    use_batch_pricing: bool = False,
) -> float:
    if use_batch_pricing:
        return (
            input_tokens / 1_000_000 * FLASH_LITE_BATCH_INPUT
            + output_tokens / 1_000_000 * FLASH_LITE_BATCH_OUTPUT
        )
    return (
        input_tokens / 1_000_000 * FLASH_LITE_INPUT
        + output_tokens / 1_000_000 * FLASH_LITE_OUTPUT
    )
