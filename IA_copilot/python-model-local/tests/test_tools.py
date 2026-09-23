# This file has been created (totally or partially) with the assistance of artificial intelligence tools.
# All content has been generated under the direct supervision of a named individual,
# and under the AI.Backbone Orchestrator Compliance framework

from pathlib import Path

import pytest

from code_agent.tools import WorkspaceTools


def test_tools_read_and_find_inside_workspace(tmp_path: Path) -> None:
    source = tmp_path / "example.py"
    source.write_text("def greet():\n    return 'hello'\n", encoding="utf-8")
    tools = WorkspaceTools(tmp_path)

    assert "return 'hello'" in tools.read("example.py").content
    assert "example.py:2" in tools.find("hello").content


def test_tools_block_sensitive_and_outside_paths(tmp_path: Path) -> None:
    (tmp_path / ".env").write_text("SECRET=value", encoding="utf-8")
    tools = WorkspaceTools(tmp_path)

    with pytest.raises(PermissionError):
        tools.read(".env")
    with pytest.raises(PermissionError):
        tools.read("../outside.txt")