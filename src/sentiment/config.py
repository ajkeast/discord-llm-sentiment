"""Environment-backed configuration."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_MODEL = "gemini-3.1-flash-lite"

load_dotenv(PROJECT_ROOT / ".env")
load_dotenv()  # also pick up CWD .env if present


@dataclass(frozen=True)
class MySQLConfig:
    host: str
    port: int
    user: str
    password: str
    database: str

    @property
    def url(self) -> str:
        return (
            f"mysql+pymysql://{self.user}:{self.password}"
            f"@{self.host}:{self.port}/{self.database}"
        )


@dataclass(frozen=True)
class PipelineConfig:
    gemini_api_key: str
    model: str
    context_size: int
    batch_size: int
    max_content_chars: int
    mysql: MySQLConfig | None


def get_mysql_config() -> MySQLConfig:
    host = os.getenv("MYSQL_HOST", "").strip()
    user = os.getenv("MYSQL_USER", "").strip()
    password = os.getenv("MYSQL_PASSWORD", "")
    database = os.getenv("MYSQL_DATABASE", "discord_messages").strip()
    port = int(os.getenv("MYSQL_PORT", "3306"))
    if not host or not user:
        raise ValueError(
            "MYSQL_HOST and MYSQL_USER must be set in .env "
            "(see .env.example)."
        )
    return MySQLConfig(
        host=host,
        port=port,
        user=user,
        password=password,
        database=database,
    )


def get_pipeline_config(*, require_mysql: bool = False) -> PipelineConfig:
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    mysql: MySQLConfig | None = None
    if require_mysql:
        mysql = get_mysql_config()
    else:
        try:
            mysql = get_mysql_config()
        except ValueError:
            mysql = None

    return PipelineConfig(
        gemini_api_key=api_key,
        model=os.getenv("GEMINI_MODEL", DEFAULT_MODEL).strip(),
        context_size=int(os.getenv("CONTEXT_SIZE", "3")),
        batch_size=int(os.getenv("BATCH_SIZE", "15")),
        max_content_chars=int(os.getenv("MAX_CONTENT_CHARS", "500")),
        mysql=mysql,
    )


def ensure_data_dirs() -> dict[str, Path]:
    """Create standard data directories and return their paths."""
    paths = {
        "data": DATA_DIR,
        "results": DATA_DIR / "results",
        "samples": DATA_DIR / "samples",
        "gold": DATA_DIR / "gold",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths
