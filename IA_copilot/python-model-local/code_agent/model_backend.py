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


DEFAULT_MODEL_ID = "Qwen/Qwen2.5-Coder-7B-Instruct-GGUF"
DEFAULT_MODEL_FILENAME = "qwen2.5-coder-7b-instruct-q4_k_m.gguf"
DEFAULT_MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / DEFAULT_MODEL_FILENAME


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

    def __init__(self, model_path: str | Path, device: str | None = None, adapter_path: str | Path | None = None) -> None:
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "The Hugging Face backend requires optional dependencies. "
                "Install them with: python -m pip install -e .[hf]"
            ) from exc

        self.model_path = str(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, use_fast=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype="auto",
            low_cpu_mem_usage=True,
        )
        if adapter_path is not None:
            try:
                from peft import PeftModel
            except ModuleNotFoundError as exc:
                raise RuntimeError(
                    "Loading a LoRA adapter requires peft. Install: python -m pip install -e .[training]"
                ) from exc
            self.model = PeftModel.from_pretrained(self.model, str(adapter_path))
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


class LlamaCppBackend:
    """GGUF backend for CPU-friendly local inference."""

    def __init__(self, model_path: str | Path, device: str | None = "cpu") -> None:
        try:
            from llama_cpp import Llama
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "The GGUF backend requires llama-cpp-python. "
                "Install it with: python -m pip install -e .[llama]"
            ) from exc

        selected_path = Path(model_path)
        if not selected_path.is_file():
            raise FileNotFoundError(f"GGUF model not found: {selected_path}")
        self.model = Llama(
            model_path=str(selected_path),
            n_ctx=8192,
            n_threads=6,
            n_gpu_layers=0 if device in {None, "cpu"} else -1,
            verbose=False,
        )

    def generate(self, prompt: str, config: GenerationConfig) -> str:
        result = self.model(
            prompt,
            max_tokens=config.max_new_tokens,
            temperature=config.temperature,
            top_p=config.top_p,
            stop=["<|user|>", "<|endoftext|>"],
        )
        return str(result["choices"][0]["text"]).strip()


def create_backend(
    backend: str = "llama-cpp",
    model_path: str | Path | None = DEFAULT_MODEL_PATH,
    device: str | None = "cpu",
    adapter_path: str | Path | None = None,
) -> ModelBackend:
    """Create a model backend by name."""
    normalized = backend.lower().strip()
    if normalized == "echo":
        return EchoBackend()
    if normalized in {"llama-cpp", "llama", "gguf"}:
        selected_path = Path(model_path) if model_path is not None else DEFAULT_MODEL_PATH
        if not selected_path.exists():
            raise FileNotFoundError(
                f"Default GGUF model not found at {selected_path}. "
                "Run: python -m code_agent.cli download-model"
            )
        return LlamaCppBackend(selected_path, device=device)
    if normalized in {"hf", "transformers"}:
        selected_model = str(model_path) if model_path is not None else str(DEFAULT_MODEL_PATH)
        if model_path is None or (selected_model == str(DEFAULT_MODEL_PATH) and not Path(selected_model).exists()):
            raise FileNotFoundError(
                f"Hugging Face model path is required and the default GGUF file was not found at {selected_model}."
            )
        return TransformersBackend(selected_model, device=device, adapter_path=adapter_path)
    raise ValueError(f"Unknown backend: {backend}")