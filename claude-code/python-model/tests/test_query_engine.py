import pytest
from claude_port.query_engine import QueryEngine


def test_build_prompt_and_generate_smoke():
    # This is a smoke test that only constructs the engine with a tiny model
    # The user must provide a local tiny model path for this to run; otherwise skip.
    import os
    model_path = os.getenv('TEST_LOCAL_MODEL')
    if not model_path:
        pytest.skip('No local model specified in TEST_LOCAL_MODEL')
    engine = QueryEngine(model_path)
    out = engine.generate('Hola, ¿puedes saludarme?', max_new_tokens=8)
    assert isinstance(out, str)
