# This file has been created (totally or partially) with the assistance of artificial intelligence tools.
# All content has been generated under the direct supervision of a named individual,
# and under the AI.Backbone Orchestrator Compliance framework

"""Model backend adapters.

The module is importable without heavy ML dependencies. Optional dependencies are
loaded only when their backend is selected.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


@dataclass(frozen=True)
class GenerationConfig:
    """Generation options shared by model backends."""

    max_new_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.95
    device: str | None = None


class ModelBackend(Protocol):
    """Protocol implemented by all generation backends."""

    def generate(self, prompt: str, config: GenerationConfig) -> str:
        """Generate assistant text for a fully constructed prompt."""


class EchoBackend:
    """Dependency-free backend used for smoke tests and Python 3.14 validation."""

    def generate(self, prompt: str, config: GenerationConfig) -> str:
        last_user_marker = "<|user|>"
        assistant_marker = "<|assistant|>"
        user_section = prompt
        if last_user_marker in prompt:
            user_section = prompt.rsplit(last_user_marker, 1)[-1]
        if assistant_marker in user_section:
            user_section = user_section.split(assistant_marker, 1)[0]
        message = user_section.strip()
        return f"Echo backend ready. Received: {message}"


class TransformersBackend:
    """Hugging Face Transformers backend loaded lazily."""

    def __init__(self, model_path: str | Path, device: str | None = None) -> None:
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "The Hugging Face backend requires optional dependencies. "
                "Install them with: python -m pip install -e .[hf]"
            ) from exc

        self.model_path = str(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, use_fast=True)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        if device:
            self.model.to(device)

    def generate(self, prompt: str, config: GenerationConfig) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt")
        if config.device:
            inputs = inputs.to(config.device)
        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=config.max_new_tokens,
            temperature=config.temperature,
            top_p=config.top_p,
            do_sample=config.temperature > 0,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        prompt_tokens = inputs["input_ids"].shape[-1]
        return self.tokenizer.decode(output_ids[0][prompt_tokens:], skip_special_tokens=True).strip()


def create_backend(backend: str = "echo", model_path: str | Path | None = None, device: str | None = None) -> ModelBackend:
    """Create a model backend by name."""
    normalized = backend.lower().strip()
    if normalized == "echo":
        return EchoBackend()
    if normalized in {"hf", "transformers"}:
        if model_path is None:
            raise ValueError("--model-path is required for the Hugging Face backend")
        return TransformersBackend(model_path, device=device)
    raise ValueError(f"Unknown backend: {backend}")