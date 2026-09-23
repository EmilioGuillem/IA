# This file has been created (totally or partially) with the assistance of artificial intelligence tools.
# All content has been generated under the direct supervision of a named individual,
# and under the AI.Backbone Orchestrator Compliance framework

from code_agent.cli import main


def test_cli_run_echo_backend(capsys) -> None:
    exit_code = main(["run", "--backend", "echo", "--message", "ping", "--no-log"])
    captured = capsys.readouterr()
    assert exit_code == 0
    assert "Echo backend ready" in captured.out