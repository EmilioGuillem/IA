# This file has been created (totally or partially) with the assistance of artificial intelligence tools.
# All content has been generated under the direct supervision of a named individual,
# and under the AI.Backbone Orchestrator Compliance framework

import json
from pathlib import Path

from data_pipeline import load_records, write_prepared_dataset


def test_load_records_supports_conversations_and_text(tmp_path: Path) -> None:
    conversation = tmp_path / "conversation.jsonl"
    conversation.write_text(
        json.dumps({"messages": [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello"}]}) + "\n",
        encoding="utf-8",
    )
    notes = tmp_path / "notes.md"
    notes.write_text("Use small functions.\n", encoding="utf-8")

    records = load_records([conversation, notes])
    output = write_prepared_dataset(records, tmp_path / "prepared.jsonl")

    assert len(records) == 2
    assert output.exists()
    assert "assistant: Hello" in output.read_text(encoding="utf-8")