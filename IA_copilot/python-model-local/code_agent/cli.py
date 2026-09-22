# This file has been created (totally or partially) with the assistance of artificial intelligence tools.
# All content has been generated under the direct supervision of a named individual,
# and under the AI.Backbone Orchestrator Compliance framework

"""Command line interface for the local coding agent."""

from __future__ import annotations

import argparse
from pathlib import Path

from .conversation_logger import append_conversation
from .query_engine import QueryEngine


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="code-agent")
    subcommands = parser.add_subparsers(dest="command")

    run = subcommands.add_parser("run", help="Run one message and print the response")
    add_generation_args(run)
    run.add_argument("--message", required=True)

    chat = subcommands.add_parser("chat", help="Start an interactive chat")
    add_generation_args(chat)

    return parser


def add_generation_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--backend", choices=["echo", "hf", "transformers"], default="echo")
    parser.add_argument("--model-path", default=None, help="Local model path for the Hugging Face backend")
    parser.add_argument("--device", default=None, help="Optional torch device such as cpu or cuda")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--no-log", action="store_true", help="Disable daily conversation logging")


def create_engine(args: argparse.Namespace) -> QueryEngine:
    return QueryEngine(model_path=args.model_path, backend=args.backend, device=args.device)


def run_once(args: argparse.Namespace) -> int:
    engine = create_engine(args)
    response = engine.generate(
        args.message,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        device=args.device,
    )
    print(response)
    if not args.no_log:
        model_name = Path(args.model_path).name if args.model_path else args.backend
        append_conversation(model_name, args.message, response, metadata={"source": "cli_run"})
    return 0


def run_chat(args: argparse.Namespace) -> int:
    engine = create_engine(args)
    print('Chat started. Type "salir", "exit", or "quit" to finish.')
    while True:
        user_message = input("\nUsuario: ").strip()
        if not user_message:
            continue
        if user_message.lower() in {"salir", "exit", "quit"}:
            print("Closing chat.")
            return 0
        response = engine.generate(
            user_message,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            device=args.device,
        )
        print("\nAssistant:")
        print(response)
        if not args.no_log:
            model_name = Path(args.model_path).name if args.model_path else args.backend
            append_conversation(model_name, user_message, response, metadata={"source": "cli_chat"})


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "run":
        return run_once(args)
    if args.command == "chat":
        return run_chat(args)
    parser.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())