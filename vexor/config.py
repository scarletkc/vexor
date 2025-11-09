"""Global configuration management for Vexor."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

CONFIG_DIR = Path(os.path.expanduser("~")) / ".vexor"
CONFIG_FILE = CONFIG_DIR / "config.json"
DEFAULT_MODEL = "gemini-embedding-001"
DEFAULT_BATCH_SIZE = 0
ENV_API_KEY = "GOOGLE_GENAI_API_KEY"


@dataclass
class Config:
    api_key: str | None = None
    model: str = DEFAULT_MODEL
    batch_size: int = DEFAULT_BATCH_SIZE


def load_config() -> Config:
    if not CONFIG_FILE.exists():
        return Config()
    raw = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
    return Config(
        api_key=raw.get("api_key") or None,
        model=raw.get("model") or DEFAULT_MODEL,
        batch_size=int(raw.get("batch_size", DEFAULT_BATCH_SIZE)),
    )


def save_config(config: Config) -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    data: Dict[str, Any] = {}
    if config.api_key:
        data["api_key"] = config.api_key
    if config.model:
        data["model"] = config.model
    data["batch_size"] = config.batch_size
    CONFIG_FILE.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def set_api_key(value: str | None) -> None:
    config = load_config()
    config.api_key = value
    save_config(config)


def set_model(value: str) -> None:
    config = load_config()
    config.model = value
    save_config(config)


def set_batch_size(value: int) -> None:
    config = load_config()
    config.batch_size = value
    save_config(config)
