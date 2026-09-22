# This file has been created (totally or partially) with the assistance of artificial intelligence tools.
# All content has been generated under the direct supervision of a named individual,
# and under the AI.Backbone Orchestrator Compliance framework

from code_agent.query_engine import QueryEngine


def test_echo_backend_generates_without_ml_dependencies() -> None:
    engine = QueryEngine(backend="echo")
    output = engine.generate("Hola, puedes responder?")
    assert "Echo backend ready" in output
    assert "Hola" in output