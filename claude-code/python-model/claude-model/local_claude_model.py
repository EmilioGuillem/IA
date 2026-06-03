import argparse
from pathlib import Path

from transformers import AutoModelForCausalLM, AutoTokenizer

CLAUDE_SYSTEM_PROMPT = """
You are Claude Code, Anthropic's official CLI for Claude.

You are an agent for Claude Code, Anthropic's official CLI for Claude. Given the user's message, you should use the tools available to complete the task. Complete the task fully—don't gold-plate, but don't leave it half-done. When you complete the task, respond with a concise report covering what was done and any key findings — the caller will relay this to the user, so it only needs the essentials.

Notes:
- Use absolute file paths when referring to files.
- Avoid emojis.
- Include code snippets only when the exact text is load-bearing.
- Do not use a colon before tool calls. Use plain sentences before invoking a tool.
"""

PROMPT_TEMPLATE = """
<|system|>
{system_prompt}
<|endoftext|>
<|user|>
{user}
<|assistant|>
"""


def build_prompt(user_message: str, system_prompt: str = CLAUDE_SYSTEM_PROMPT) -> str:
    return PROMPT_TEMPLATE.format(system_prompt=system_prompt.strip(), user=user_message.strip())


def parse_args():
    parser = argparse.ArgumentParser(description='Local Claude Code-style model wrapper.')
    parser.add_argument('--model_path', required=True, help='Ruta local al modelo compatible con Hugging Face')
    parser.add_argument('--temperature', type=float, default=0.7)
    parser.add_argument('--max_new_tokens', type=int, default=256)
    parser.add_argument('--top_p', type=float, default=0.95)
    return parser.parse_args()


def load_model(model_path: Path):
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(model_path)
    return tokenizer, model


def main():
    args = parse_args()
    model_path = Path(args.model_path)

    if not model_path.exists():
        raise FileNotFoundError(f'Model path not found: {model_path}')

    tokenizer, model = load_model(model_path)

    print('Claude Code local wrapper iniciado. Escribe "salir" para terminar.')

    while True:
        user_message = input('\nUsuario: ').strip()
        if not user_message:
            continue
        if user_message.lower() in {'salir', 'exit', 'quit'}:
            print('Cerrando chat...')
            break

        prompt = build_prompt(user_message)
        input_ids = tokenizer(prompt, return_tensors='pt').input_ids

        output_ids = model.generate(
            input_ids,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

        response = tokenizer.decode(output_ids[0][input_ids.shape[-1]:], skip_special_tokens=True)
        print('\nClaude Code local:')
        print(response.strip())


if __name__ == '__main__':
    main()
