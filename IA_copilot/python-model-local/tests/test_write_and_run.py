# This file has been created (totally or partially) with the assistance of artificial intelligence tools.
# All content has been generated under the direct supervision of a named individual,
# and under the AI.Backbone Orchestrator Compliance framework

from pathlib import Path

import pytest

from code_agent.agent import CodingAgent
from code_agent.query_engine import QueryEngine
from code_agent.tools import WorkspaceTools


def test_replace_and_run_allowlisted_command(tmp_path: Path) -> None:
    target = tmp_path / "example.py"
    target.write_text("value = 1\n", encoding="utf-8")
    tools = WorkspaceTools(tmp_path)

    result = tools.replace("example.py", "value = 1", "value = 2")
    assert "Updated" in result.content
    command_result = tools.run("python -m pytest --version")
    assert "exit_code=" in command_result.content
    assert "value = 2" in target.read_text(encoding="utf-8")


def test_command_policy_blocks_shell_code_and_unknown_executable(tmp_path: Path) -> None:
    tools = WorkspaceTools(tmp_path)
    with pytest.raises(PermissionError):
        tools.run("python -c pass")
    with pytest.raises(PermissionError):
        tools.run("powershell Write-Output blocked")


def test_agent_requires_confirmation_for_write(tmp_path: Path) -> None:
    agent = CodingAgent(QueryEngine(backend="echo"), tmp_path, confirm=lambda _message: False)
    result = agent.execute_tool_command('/write {"path":"new.py","content":"print(1)"}')
    assert result is not None
    assert "cancelled" in result.content.lower()
    assert not (tmp_path / "new.py").exists()