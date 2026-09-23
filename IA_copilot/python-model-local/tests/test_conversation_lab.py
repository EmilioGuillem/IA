# This file has been created (totally or partially) with the assistance of artificial intelligence tools.
# All content has been generated under the direct supervision of a named individual,
# and under the AI.Backbone Orchestrator Compliance framework

from pathlib import Path

from conversation_lab import ConversationLab


def test_conversation_lab_records_two_models(tmp_path: Path) -> None:
    lab = ConversationLab("unused-a", "unused-b", "echo", "echo", tmp_path)
    lab.start("Say something useful.", rounds=1)
    assert lab.state.worker is not None
    lab.state.worker.join(timeout=5)

    snapshot = lab.snapshot()
    assert snapshot["running"] is False
    assert len(snapshot["messages"]) == 2
    assert list(tmp_path.rglob("conversation.jsonl"))
    assert list(tmp_path.rglob("conversation.md"))