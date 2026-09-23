# This file has been created (totally or partially) with the assistance of artificial intelligence tools.
# All content has been generated under the direct supervision of a named individual,
# and under the AI.Backbone Orchestrator Compliance framework

"""Normalize conversation and text files into supervised training records."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable


SUPPORTED_TEXT = {".txt", ".md", ".rst", ".py", ".js", ".ts", ".tsx", ".json", ".yaml", ".yml"}
SKIP_NAMES = {".env", ".env.local", "id_rsa", "id_ed25519"}
SKIP_SUFFIXES = {".pem", ".key", ".p12", ".pfx"}


def record_to_text(record: dict[str, Any]) -> str:
    if isinstance(record.get("text"), str) and record["text"].strip():
        return record["text"].strip()
    if isinstance(record.get("messages"), list):
        rows = []
        for message in record["messages"]:
            if not isinstance(message, dict):
                continue
            role = str(message.get("role", "user"))
            content = message.get("content", message.get("text", ""))
            if isinstance(content, list):
                content = " ".join(str(item) for item in content)
            if content:
                rows.append(f"{role}: {content}")
        return "\n".join(rows).strip()
    user = record.get("user", record.get("prompt", record.get("instruction", "")))
    assistant = record.get("assistant", record.get("response", record.get("output", "")))
    return f"user: {user}\nassistant: {assistant}".strip() if user or assistant else ""


def iter_input_files(paths: Iterable[str | Path]) -> Iterable[Path]:
    for raw_path in paths:
        path = Path(raw_path).expanduser()
        if path.is_file():
            yield path
        elif path.is_dir():
            yield from (child for child in sorted(path.rglob("*")) if child.is_file())
        else:
            raise FileNotFoundError(f"Training input not found: {path}")


def load_records(paths: Iterable[str | Path]) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    seen: set[Path] = set()
    for path in iter_input_files(paths):
        resolved = path.resolve()
        if resolved in seen or path.name in SKIP_NAMES or path.suffix.lower() in SKIP_SUFFIXES:
            continue
        seen.add(resolved)
        if path.suffix.lower() in {".jsonl", ".json"}:
            with path.open("r", encoding="utf-8") as handle:
                raw_records = [json.loads(line) for line in handle if line.strip()] if path.suffix.lower() == ".jsonl" else json.loads(handle.read())
            if isinstance(raw_records, dict):
                raw_records = [raw_records]
            for raw in raw_records:
                if isinstance(raw, dict):
                    text = record_to_text(raw)
                    if text:
                        records.append({"text": text, "source": str(path)})
        elif path.suffix.lower() in SUPPORTED_TEXT:
            text = path.read_text(encoding="utf-8", errors="replace").strip()
            if text:
                records.append({"text": text, "source": str(path)})
    if not records:
        raise ValueError("No usable training records found")
    return records


def write_prepared_dataset(records: list[dict[str, str]], output_path: str | Path) -> Path:
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return destination


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare conversation and text data for training")
    parser.add_argument("paths", nargs="+", help="JSONL, JSON, text files, or directories")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    records = load_records(args.paths)
    output = write_prepared_dataset(records, args.output)
    print(f"Prepared {len(records)} records in {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())