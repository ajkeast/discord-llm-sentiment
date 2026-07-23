"""Parse Gemini Batch API results (inline or downloaded result files)."""

from __future__ import annotations

import json
from typing import Any, Iterable

from sentiment.schema import SentimentBatchResponse, SentimentResult


def _extract_text_from_response_obj(resp: Any) -> str | None:
    if resp is None:
        return None
    if hasattr(resp, "text") and resp.text:
        return resp.text
    if isinstance(resp, dict):
        # Common Batch file shape:
        # {"candidates":[{"content":{"parts":[{"text": "..."}]}}]}
        try:
            return resp["candidates"][0]["content"]["parts"][0]["text"]
        except Exception:
            pass
        if isinstance(resp.get("text"), str):
            return resp["text"]
    return None


def results_from_response_text(text: str) -> list[SentimentResult]:
    batch = SentimentBatchResponse.model_validate(json.loads(text))
    return list(batch.results)


def parse_batch_result_jsonl(raw: str | bytes) -> list[SentimentResult]:
    """Parse a Batch API output JSONL (one generateContent response per line)."""
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8", errors="replace")

    all_results: list[SentimentResult] = []
    errors = 0
    for line_no, line in enumerate(raw.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            errors += 1
            continue

        # File format variants: {"response": {...}} or {"key":..., "response":...}
        # or error: {"error": {...}}
        if "error" in obj and "response" not in obj:
            errors += 1
            continue

        resp = obj.get("response", obj)
        text = _extract_text_from_response_obj(resp)
        if not text:
            errors += 1
            continue
        try:
            all_results.extend(results_from_response_text(text))
        except Exception:
            errors += 1
            continue

    if errors:
        print(f"  skipped {errors} malformed/error line(s) in batch result file", flush=True)
    return all_results


def parse_job_inline_responses(job: Any) -> list[SentimentResult]:
    dest = getattr(job, "dest", None)
    responses = getattr(dest, "inlined_responses", None) if dest else None
    if not responses:
        return []

    all_results: list[SentimentResult] = []
    for item in responses:
        resp = getattr(item, "response", None) or item
        text = _extract_text_from_response_obj(resp)
        if not text:
            continue
        all_results.extend(results_from_response_text(text))
    return all_results


def download_batch_result_file(client: Any, file_name: str) -> bytes:
    name = file_name if file_name.startswith("files/") else f"files/{file_name}"
    content = client.files.download(file=name)
    if isinstance(content, (bytes, bytearray)):
        return bytes(content)
    raw = getattr(content, "content", None)
    if isinstance(raw, (bytes, bytearray)):
        return bytes(raw)
    raise TypeError(f"Unexpected download type: {type(content)}")


def parse_batch_job(client: Any, job: Any) -> list[SentimentResult]:
    """
    Extract SentimentResult rows from a succeeded Batch job.

    Prefers inlined_responses; otherwise downloads dest.file_name JSONL.
    """
    inline = parse_job_inline_responses(job)
    if inline:
        return inline

    dest = getattr(job, "dest", None)
    file_name = getattr(dest, "file_name", None) if dest else None
    if not file_name:
        print("  WARN: job has neither inlined_responses nor file_name", flush=True)
        return []

    print(f"  Downloading results {file_name}...", flush=True)
    raw = download_batch_result_file(client, file_name)
    print(f"  Downloaded {len(raw):,} bytes; parsing...", flush=True)
    return parse_batch_result_jsonl(raw)


def iter_succeeded_batch_jobs(client: Any, *, page_size: int = 50) -> Iterable[Any]:
    from google.genai import types

    for job in client.batches.list(config=types.ListBatchJobsConfig(page_size=page_size)):
        state = str(getattr(job, "state", ""))
        if "SUCCEEDED" in state:
            yield job
