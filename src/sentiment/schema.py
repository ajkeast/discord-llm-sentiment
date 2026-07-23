"""Pydantic models for rich sentiment outputs."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator


class Polarity(str, Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"
    MIXED = "mixed"


class Toxicity(str, Enum):
    NONE = "none"
    MILD = "mild"
    MODERATE = "moderate"
    SEVERE = "severe"


class DirectedAt(str, Enum):
    GENERAL = "general"
    PERSON = "person"
    GROUP = "group"
    SELF = "self"
    TOPIC = "topic"


ALLOWED_EMOTIONS = frozenset(
    {
        "joy",
        "anger",
        "annoyance",
        "amusement",
        "sadness",
        "fear",
        "surprise",
        "disgust",
        "neutral",
    }
)


class SentimentResult(BaseModel):
    message_id: str
    polarity: Polarity
    polarity_score: float = Field(..., ge=-1.0, le=1.0)
    emotions: list[str] = Field(..., min_length=1, max_length=3)
    sarcasm: bool
    toxicity: Toxicity
    directed_at: DirectedAt
    confidence: float = Field(..., ge=0.0, le=1.0)
    rationale: str = Field(..., max_length=200)

    @field_validator("emotions")
    @classmethod
    def validate_emotions(cls, values: list[str]) -> list[str]:
        cleaned: list[str] = []
        for emotion in values:
            key = emotion.strip().lower()
            if key not in ALLOWED_EMOTIONS:
                raise ValueError(
                    f"emotion '{emotion}' not in {sorted(ALLOWED_EMOTIONS)}"
                )
            if key not in cleaned:
                cleaned.append(key)
        if not cleaned:
            raise ValueError("emotions must contain at least one label")
        return cleaned[:3]

    @field_validator("rationale")
    @classmethod
    def shorten_rationale(cls, value: str) -> str:
        words = value.strip().split()
        if len(words) > 15:
            return " ".join(words[:15])
        return value.strip()

    @field_validator("message_id", mode="before")
    @classmethod
    def coerce_message_id(cls, value: Any) -> str:
        return str(value)


class SentimentBatchResponse(BaseModel):
    results: list[SentimentResult]


def results_to_dataframe(results: list[SentimentResult]):
    import pandas as pd

    rows = []
    for r in results:
        mid = str(r.message_id).strip()
        # Keep as string for parquet safety; cast to int only when writing MySQL.
        rows.append(
            {
                "message_id": mid,
                "polarity": r.polarity.value,
                "polarity_score": r.polarity_score,
                "emotions": ",".join(r.emotions),
                "sarcasm": bool(r.sarcasm),
                "toxicity": r.toxicity.value,
                "directed_at": r.directed_at.value,
                "confidence": r.confidence,
                "rationale": r.rationale,
            }
        )
    return pd.DataFrame(rows)


def sentiment_json_schema() -> dict[str, Any]:
    """JSON schema fragment embedded in prompts / response_schema."""
    return {
        "type": "object",
        "properties": {
            "results": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "message_id": {"type": "string"},
                        "polarity": {
                            "type": "string",
                            "enum": [p.value for p in Polarity],
                        },
                        "polarity_score": {
                            "type": "number",
                            "minimum": -1,
                            "maximum": 1,
                        },
                        "emotions": {
                            "type": "array",
                            "items": {
                                "type": "string",
                                "enum": sorted(ALLOWED_EMOTIONS),
                            },
                            "minItems": 1,
                            "maxItems": 3,
                        },
                        "sarcasm": {"type": "boolean"},
                        "toxicity": {
                            "type": "string",
                            "enum": [t.value for t in Toxicity],
                        },
                        "directed_at": {
                            "type": "string",
                            "enum": [d.value for d in DirectedAt],
                        },
                        "confidence": {
                            "type": "number",
                            "minimum": 0,
                            "maximum": 1,
                        },
                        "rationale": {"type": "string"},
                    },
                    "required": [
                        "message_id",
                        "polarity",
                        "polarity_score",
                        "emotions",
                        "sarcasm",
                        "toxicity",
                        "directed_at",
                        "confidence",
                        "rationale",
                    ],
                },
            }
        },
        "required": ["results"],
    }
