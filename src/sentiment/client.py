"""Gemini client for contextual sentiment scoring."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any

import pandas as pd
from tqdm.auto import tqdm

from sentiment.batching import iter_batches
from sentiment.estimate import cost_from_usage
from sentiment.io import append_checkpoint, filter_unscored
from sentiment.prompt import SYSTEM_PROMPT, build_batch_user_prompt
from sentiment.schema import SentimentBatchResponse, SentimentResult, results_to_dataframe


@dataclass
class UsageTracker:
    input_tokens: int = 0
    output_tokens: int = 0
    requests: int = 0
    failures: int = 0

    def add_response(self, response: Any) -> None:
        self.requests += 1
        usage = getattr(response, "usage_metadata", None)
        if usage is None:
            return
        self.input_tokens += int(getattr(usage, "prompt_token_count", 0) or 0)
        self.output_tokens += int(
            getattr(usage, "candidates_token_count", 0)
            or getattr(usage, "response_token_count", 0)
            or 0
        )

    @property
    def cost_live_usd(self) -> float:
        return cost_from_usage(self.input_tokens, self.output_tokens, use_batch_pricing=False)

    @property
    def cost_batch_usd(self) -> float:
        return cost_from_usage(self.input_tokens, self.output_tokens, use_batch_pricing=True)

    def as_dict(self) -> dict:
        return {
            "requests": self.requests,
            "failures": self.failures,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cost_live_usd": round(self.cost_live_usd, 6),
            "cost_batch_equivalent_usd": round(self.cost_batch_usd, 6),
        }


@dataclass
class ScoreRunResult:
    results: list[SentimentResult] = field(default_factory=list)
    usage: UsageTracker = field(default_factory=UsageTracker)
    dataframe: pd.DataFrame | None = None


class SentimentClient:
    """Thin wrapper around google-genai for batched sentiment scoring."""

    def __init__(
        self,
        api_key: str,
        *,
        model: str = "gemini-3.1-flash-lite",
        temperature: float = 0.2,
        max_retries: int = 3,
    ):
        if not api_key:
            raise ValueError("GEMINI_API_KEY is required")
        from google import genai

        self._genai = genai
        self.client = genai.Client(api_key=api_key)
        self.model = model
        self.temperature = temperature
        self.max_retries = max_retries

    def score_batch(self, items: list[dict]) -> tuple[list[SentimentResult], Any]:
        """Score one batch of prompt items; retry once on validation failure."""
        from google.genai import types

        user_prompt = build_batch_user_prompt(items)
        expected_ids = {item["message_id"] for item in items}
        last_error: Exception | None = None

        for attempt in range(self.max_retries):
            try:
                response = self.client.models.generate_content(
                    model=self.model,
                    contents=user_prompt,
                    config=types.GenerateContentConfig(
                        system_instruction=SYSTEM_PROMPT,
                        temperature=self.temperature,
                        response_mime_type="application/json",
                        response_schema=SentimentBatchResponse,
                    ),
                )
                results = self._parse_response(response, expected_ids)
                return results, response
            except Exception as exc:  # noqa: BLE001 — surface and retry
                last_error = exc
                time.sleep(1.5 * (attempt + 1))

        raise RuntimeError(f"Batch failed after {self.max_retries} attempts: {last_error}")

    def _parse_response(
        self,
        response: Any,
        expected_ids: set[str],
    ) -> list[SentimentResult]:
        parsed = getattr(response, "parsed", None)
        if isinstance(parsed, SentimentBatchResponse):
            batch = parsed
        elif isinstance(parsed, dict):
            batch = SentimentBatchResponse.model_validate(parsed)
        else:
            text = getattr(response, "text", None) or ""
            batch = SentimentBatchResponse.model_validate(json.loads(text))

        results = list(batch.results)
        got_ids = {r.message_id for r in results}
        missing = expected_ids - got_ids
        if missing:
            raise ValueError(f"Model omitted message_ids: {sorted(missing)[:5]}")
        # Keep only expected ids (drop extras)
        return [r for r in results if r.message_id in expected_ids]

    def score_windows(
        self,
        windows: pd.DataFrame,
        *,
        batch_size: int = 15,
        run_name: str | None = None,
        checkpoint_every: int = 1,
        resume: bool = True,
        show_progress: bool = True,
    ) -> ScoreRunResult:
        """
        Score all rows in a windows DataFrame.

        If `run_name` is set, results are checkpointed for resume.
        """
        work = windows
        if run_name and resume:
            work = filter_unscored(windows, run_name)

        usage = UsageTracker()
        all_results: list[SentimentResult] = []
        batches = list(iter_batches(work, batch_size=batch_size))
        iterator = tqdm(batches, disable=not show_progress, desc="Scoring")

        for batch_idx, items in enumerate(iterator, start=1):
            try:
                results, response = self.score_batch(items)
                usage.add_response(response)
                all_results.extend(results)
                if run_name and batch_idx % checkpoint_every == 0:
                    append_checkpoint(run_name, results)
            except Exception:
                usage.failures += 1
                if run_name:
                    # skip failed batch but continue
                    continue
                raise

        # If checkpointing, reload full checkpoint as source of truth
        if run_name:
            from sentiment.io import checkpoint_path

            path = checkpoint_path(run_name)
            if path.exists():
                df = pd.read_parquet(path)
            else:
                df = results_to_dataframe(all_results)
        else:
            df = results_to_dataframe(all_results)

        return ScoreRunResult(results=all_results, usage=usage, dataframe=df)

    def _batch_request_line(self, items: list[dict]) -> str:
        user_prompt = build_batch_user_prompt(items)
        key = items[0]["message_id"]
        request = {
            "key": f"batch-{key}",
            "request": {
                "contents": [
                    {
                        "role": "user",
                        "parts": [{"text": user_prompt}],
                    }
                ],
                "system_instruction": {
                    "parts": [{"text": SYSTEM_PROMPT}],
                },
                "generation_config": {
                    "temperature": self.temperature,
                    "response_mime_type": "application/json",
                },
            },
        }
        return json.dumps(request)

    def write_batch_jsonl(
        self,
        windows: pd.DataFrame,
        path: str | Any,
        *,
        batch_size: int = 15,
    ) -> int:
        """
        Write Gemini Batch API JSONL requests (one generateContent line per batch).

        Upload with the Gemini Batch API for ~50% discounted pricing.
        Returns number of request lines written.
        """
        from pathlib import Path

        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        n = 0
        with out.open("w", encoding="utf-8") as fh:
            for items in iter_batches(windows, batch_size=batch_size):
                fh.write(self._batch_request_line(items) + "\n")
                n += 1
        return n

    @staticmethod
    def _log(msg: str) -> None:
        """Print and flush so Jupyter shows progress immediately."""
        print(msg, flush=True)

    def write_batch_jsonl_chunks(
        self,
        windows: pd.DataFrame,
        output_dir: str | Any,
        *,
        batch_size: int = 15,
        lines_per_chunk: int = 2500,
        prefix: str = "chunk",
    ) -> list[Any]:
        """
        Split a large corpus into multiple JSONL files.

        Gemini Batch enqueued-token caps (often ~10M on lower tiers) are far
        below a full ~265k-message run (~23M tokens). Chunking avoids 429s.
        Default 2500 request-lines ≈ ~3M input tokens per job.
        """
        from pathlib import Path

        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        for old in out_dir.glob(f"{prefix}_*.jsonl"):
            old.unlink()

        total_msgs = len(windows)
        n_request_lines = (total_msgs + batch_size - 1) // batch_size if total_msgs else 0
        self._log(
            f"Writing chunked JSONL for {total_msgs:,} messages "
            f"({n_request_lines:,} request-lines, {lines_per_chunk}/chunk)..."
        )

        paths: list[Path] = []
        chunk_idx = 0
        lines_in_chunk = 0
        lines_total = 0
        fh = None
        try:
            batch_iter = list(iter_batches(windows, batch_size=batch_size))
            for items in tqdm(batch_iter, desc="Write JSONL chunks"):
                if fh is None or lines_in_chunk >= lines_per_chunk:
                    if fh is not None:
                        fh.close()
                        self._log(
                            f"  closed {paths[-1].name} "
                            f"({lines_in_chunk:,} lines, "
                            f"{paths[-1].stat().st_size / 1e6:.1f} MB)"
                        )
                    chunk_idx += 1
                    path = out_dir / f"{prefix}_{chunk_idx:03d}.jsonl"
                    fh = path.open("w", encoding="utf-8")
                    paths.append(path)
                    lines_in_chunk = 0
                    self._log(f"  opening {path.name}")
                fh.write(self._batch_request_line(items) + "\n")
                lines_in_chunk += 1
                lines_total += 1
        finally:
            if fh is not None:
                fh.close()
                self._log(
                    f"  closed {paths[-1].name} "
                    f"({lines_in_chunk:,} lines, "
                    f"{paths[-1].stat().st_size / 1e6:.1f} MB)"
                )

        self._log(f"Done writing {len(paths)} chunk file(s), {lines_total:,} request-lines total.")
        return paths

    def submit_batch_job(
        self,
        jsonl_path: str | Any,
        *,
        display_name: str = "sentiment-batch",
        max_retries: int = 6,
    ):
        """Upload a JSONL file and create a Gemini Batch job (~50% cheaper)."""
        from pathlib import Path

        from google.genai import types

        path = Path(jsonl_path)
        size_mb = path.stat().st_size / 1e6
        self._log(f"  Uploading {path.name} ({size_mb:.1f} MB)...")
        uploaded = self.client.files.upload(
            file=str(path),
            config=types.UploadFileConfig(
                display_name=display_name,
                mime_type="jsonl",
            ),
        )
        src = uploaded.name if uploaded.name.startswith("files/") else f"files/{uploaded.name}"
        self._log(f"  Upload complete → {src}. Creating batch job...")

        last_error: Exception | None = None
        for attempt in range(max_retries):
            try:
                job = self.client.batches.create(model=self.model, src=src)
                self._log(f"  Batch created: {job.name} ({job.state})")
                return job
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                msg = str(exc)
                if "429" not in msg and "RESOURCE_EXHAUSTED" not in msg:
                    raise
                sleep_s = min(300, 30 * (2**attempt))
                self._log(
                    f"  429 RESOURCE_EXHAUSTED on create "
                    f"(attempt {attempt + 1}/{max_retries}); sleeping {sleep_s}s..."
                )
                time.sleep(sleep_s)
        raise RuntimeError(f"Batch create failed after {max_retries} retries: {last_error}")

    def wait_for_batch_job(self, job_name: str, *, poll_seconds: int = 30):
        """Poll until a batch job reaches a terminal state."""
        completed = {
            "JOB_STATE_SUCCEEDED",
            "JOB_STATE_FAILED",
            "JOB_STATE_CANCELLED",
            "JOB_STATE_PAUSED",
        }
        started = time.time()
        job = self.client.batches.get(name=job_name)
        while True:
            state = getattr(job.state, "name", None) or str(job.state)
            state_str = state.split(".")[-1] if isinstance(state, str) else str(state)
            elapsed_m = (time.time() - started) / 60
            self._log(f"  … waiting on batch ({elapsed_m:.1f} min elapsed) state={state_str}")
            if state_str in completed:
                return job
            time.sleep(poll_seconds)
            job = self.client.batches.get(name=job_name)

    def run_chunked_batch(
        self,
        chunk_paths: list[Any],
        *,
        on_job_succeeded: Any | None = None,
        poll_seconds: int = 60,
        display_prefix: str = "discord-sentiment",
    ) -> list[Any]:
        """
        Submit chunk JSONL files one at a time (wait for each to finish).

        Keeps enqueued tokens under tier caps. Optional callback receives each
        succeeded job for parse/upsert.
        """
        jobs = []
        n = len(chunk_paths)
        run_started = time.time()
        for i, path in enumerate(chunk_paths, start=1):
            self._log(f"\n=== Chunk {i}/{n}: {path.name} ===")
            job = self.submit_batch_job(
                path,
                display_name=f"{display_prefix}-{i:03d}",
            )
            job = self.wait_for_batch_job(job.name, poll_seconds=poll_seconds)
            state = getattr(job.state, "name", None) or str(job.state)
            self._log(f"  finished: {state}")
            if "SUCCEEDED" not in str(state):
                raise RuntimeError(f"Chunk {path.name} ended in {state}")
            if on_job_succeeded is not None:
                on_job_succeeded(job, i, n)
            jobs.append(job)
            elapsed_m = (time.time() - run_started) / 60
            self._log(f"  Progress: {i}/{n} chunks done ({elapsed_m:.1f} min total)")
        self._log(f"\nAll {n} chunk(s) complete in {(time.time() - run_started) / 60:.1f} min.")
        return jobs


def make_client(api_key: str | None = None, model: str | None = None) -> SentimentClient:
    from sentiment.config import get_pipeline_config

    cfg = get_pipeline_config()
    key = api_key or cfg.gemini_api_key
    return SentimentClient(key, model=model or cfg.model)
