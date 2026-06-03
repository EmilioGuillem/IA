from typing import Tuple
from pathlib import Path

from transformers import AutoTokenizer, AutoModelForCausalLM

from .prompts import build_prompt


class QueryEngine:
    def __init__(self, model_path: str | Path):
        self.model_path = str(model_path)
        self.tokenizer, self.model = self.load_model(self.model_path)

    def load_model(self, model_path: str) -> Tuple[AutoTokenizer, AutoModelForCausalLM]:
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
        model = AutoModelForCausalLM.from_pretrained(model_path)
        return tokenizer, model

    def generate(self, user_message: str, max_new_tokens: int = 256, temperature: float = 0.7, top_p: float = 0.95) -> str:
        prompt = build_prompt(user_message)
        inputs = self.tokenizer(prompt, return_tensors='pt')
        input_ids = inputs.input_ids
        output_ids = self.model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        # decode only the newly generated tokens
        generated = self.tokenizer.decode(output_ids[0][input_ids.shape[-1]:], skip_special_tokens=True)
        return generated.strip()
