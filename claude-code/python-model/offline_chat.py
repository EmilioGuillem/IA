#!/usr/bin/env python3
"""Load a locally saved base or fine-tuned causal model and run an interactive chat loop.

Usage:
  python offline_chat.py --model_dir ./models/my-finetuned --device cpu
"""
from argparse import ArgumentParser
from pathlib import Path
import torch

from transformers import AutoTokenizer, AutoModelForCausalLM

from claude_port.prompts import build_prompt
from conversation_logger import append_conversation


def main():
    p = ArgumentParser()
    p.add_argument('--model_dir', required=True, help='Local model directory (base or fine-tuned)')
    p.add_argument('--device', default='cpu', help='torch device (cpu|cuda)')
    p.add_argument('--max_new_tokens', type=int, default=256)
    p.add_argument('--temperature', type=float, default=0.7)
    p.add_argument('--top_p', type=float, default=0.95)
    args = p.parse_args()

    model_dir = Path(args.model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(model_dir, trust_remote_code=True)
    model.to(args.device)

    print('Modelo cargado desde', model_dir)
    print('Escribe "salir" para terminar.')

    while True:
        user = input('\nUsuario: ').strip()
        if not user:
            continue
        if user.lower() in {'salir', 'exit', 'quit'}:
            break
        prompt = build_prompt(user)
        inputs = tokenizer(prompt, return_tensors='pt').to(args.device)
        with torch.no_grad():
            outs = model.generate(**inputs, max_new_tokens=args.max_new_tokens, temperature=args.temperature, top_p=args.top_p, do_sample=True, pad_token_id=tokenizer.eos_token_id)
        generated = tokenizer.decode(outs[0][inputs['input_ids'].shape[-1]:], skip_special_tokens=True)
        assistant_text = generated.strip()
        print('\nRespuesta:')
        print(assistant_text)

        # log the exchange to today's rotating log file
        try:
            model_name = Path(model_dir).name
            path = append_conversation(model_name, user, assistant_text, metadata={'source': 'offline_chat'})
            print(f'Conversación guardada en {path}')
        except Exception as e:
            print('Advertencia: no se pudo guardar la conversación:', e)


if __name__ == '__main__':
    main()
