import argparse
from pathlib import Path
from claude_port.query_engine import QueryEngine


def main():
    parser = argparse.ArgumentParser(prog='claude-port')
    sub = parser.add_subparsers(dest='cmd')

    chat = sub.add_parser('chat', help='Start interactive chat with a local model')
    chat.add_argument('--model_path', required=True, help='Local Hugging Face model path')
    chat.add_argument('--max_new_tokens', type=int, default=256)
    chat.add_argument('--temperature', type=float, default=0.7)
    chat.add_argument('--top_p', type=float, default=0.95)

    run = sub.add_parser('run', help='Run a single message and print response')
    run.add_argument('--model_path', required=True)
    run.add_argument('--message', required=True)

    args = parser.parse_args()

    if args.cmd == 'chat':
        engine = QueryEngine(args.model_path)
        print('Chat started. Type "salir" to exit.')
        while True:
            user = input('\nUsuario: ').strip()
            if not user:
                continue
            if user.lower() in {'salir', 'exit', 'quit'}:
                print('Saliendo...')
                break
            resp = engine.generate(user, max_new_tokens=args.max_new_tokens, temperature=args.temperature, top_p=args.top_p)
            print('\nRespuesta:')
            print(resp)
            # append to daily conversation logs
            try:
                from conversation_logger import append_conversation
                model_name = Path(args.model_path).name
                append_conversation(model_name, user, resp, metadata={'source': 'claude_port_cli'})
            except Exception:
                pass

    elif args.cmd == 'run':
        engine = QueryEngine(args.model_path)
        resp = engine.generate(args.message)
        print(resp)

    else:
        parser.print_help()


if __name__ == '__main__':
    main()
