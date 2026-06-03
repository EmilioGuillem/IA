import argparse
from prompt_template import build_chat_prompt, SYSTEM_PROMPT
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser(description='Chat with a fine-tuned model.')
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--max_length', type=int, default=1024)
    parser.add_argument('--temperature', type=float, default=0.7)
    return parser.parse_args()


def main():
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(args.model_path)

    print('Chat initialized. Escribe "salir" para terminar.')
    while True:
        user_message = input('\nUsuario: ').strip()
        if not user_message:
            continue
        if user_message.lower() in {'salir', 'exit', 'quit'}:
            print('Finalizando chat...')
            break

        prompt = build_chat_prompt(user_message, system_prompt=SYSTEM_PROMPT)
        input_ids = tokenizer(prompt, return_tensors='pt').input_ids

        output_ids = model.generate(
            input_ids,
            max_length=args.max_length,
            temperature=args.temperature,
            do_sample=True,
            top_p=0.95,
            pad_token_id=tokenizer.eos_token_id,
        )
        decoded = tokenizer.decode(output_ids[0][input_ids.shape[-1]:], skip_special_tokens=True)
        print('\nClaude Code Python Model:')
        print(decoded.strip())


if __name__ == '__main__':
    main()
