# This file has been created (totally or partially) with the assistance of artificial intelligence tools.
# All content has been generated under the direct supervision of a named individual,
# and under the AI.Backbone Orchestrator Compliance framework

"""Conversation logging utilities."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def ensure_data_dir(data_dir: Path = DATA_DIR) -> Path:
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir


def daily_conversations_path(date: datetime | None = None, data_dir: Path = DATA_DIR) -> Path:
    current = date or datetime.now(timezone.utc)
    return ensure_data_dir(data_dir) / f"conversations-{current:%Y-%m-%d}.jsonl"


def append_conversation(
    model: str,
    user: str,
    assistant: str,
    metadata: dict[str, Any] | None = None,
    data_dir: Path = DATA_DIR,
) -> Path:
    """Append one conversation turn to a daily JSONL file."""
    record: dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model": model,
        "user": user,
        "assistant": assistant,
        "text": f"user: {user}\nassistant: {assistant}",
    }
    if metadata:
        record["metadata"] = metadata

    path = daily_conversations_path(data_dir=data_dir)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return path


def list_conversation_files(data_dir: Path = DATA_DIR) -> list[Path]:
    return sorted(ensure_data_dir(data_dir).glob("conversations-*.jsonl"))


def combine_conversations(output_path: Path | None = None, data_dir: Path = DATA_DIR) -> Path:
    """Combine daily conversation files into a single JSONL file."""
    target = output_path or ensure_data_dir(data_dir) / "combined_training.jsonl"
    with target.open("w", encoding="utf-8") as output:
        for source in list_conversation_files(data_dir):
            with source.open("r", encoding="utf-8") as input_file:
                for line in input_file:
                    output.write(line)
    return target