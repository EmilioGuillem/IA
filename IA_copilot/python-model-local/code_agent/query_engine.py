# This file has been created (totally or partially) with the assistance of artificial intelligence tools.
# All content has been generated under the direct supervision of a named individual,
# and under the AI.Backbone Orchestrator Compliance framework

"""Query engine for a local clean-room coding agent."""

from __future__ import annotations

from pathlib import Path

from .model_backend import GenerationConfig, ModelBackend, create_backend
from .prompts import SYSTEM_PROMPT, build_chat_prompt


class QueryEngine:
    """Small model-agnostic query engine.

    The engine intentionally avoids importing model libraries at module import
    time. This keeps tests and CLI help functional on Python versions where ML
    wheels are not yet available.
    """

    def __init__(
        self,
        model_path: str | Path | None = None,
        backend: str = "echo",
        system_prompt: str = SYSTEM_PROMPT,
        device: str | None = None,
        model_backend: ModelBackend | None = None,
    ) -> None:
        self.system_prompt = system_prompt
        self.backend = model_backend or create_backend(backend=backend, model_path=model_path, device=device)

    def generate(
        self,
        user_message: str,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.95,
        device: str | None = None,
    ) -> str:
        """Generate an assistant response for a user message."""
        prompt = build_chat_prompt(user_message, system_prompt=self.system_prompt)
        config = GenerationConfig(
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            device=device,
        )
        return self.backend.generate(prompt, config)