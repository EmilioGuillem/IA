# This file has been created (totally or partially) with the assistance of artificial intelligence tools.
# All content has been generated under the direct supervision of a named individual,
# and under the AI.Backbone Orchestrator Compliance framework

"""Command line interface for the local coding agent."""

from __future__ import annotations

import argparse
from pathlib import Path

from .agent import CodingAgent
from .conversation_logger import append_conversation
from .model_backend import DEFAULT_MODEL_FILENAME, DEFAULT_MODEL_ID, DEFAULT_MODEL_PATH
from .query_engine import QueryEngine


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="code-agent")
    subcommands = parser.add_subparsers(dest="command")

    run = subcommands.add_parser("run", help="Run one message and print the response")
    add_generation_args(run)
    run.add_argument("--message", required=True)
    run.add_argument("--workspace", default=".", help="Workspace root used for context retrieval")

    chat = subcommands.add_parser("chat", help="Start an interactive chat")
    add_generation_args(chat)
    chat.add_argument("--workspace", default=".", help="Workspace root used by the agent")

    download = subcommands.add_parser("download-model", help="Download the default local model")
    download.add_argument("--repo-id", default=DEFAULT_MODEL_ID)
    download.add_argument("--local-dir", default=str(DEFAULT_MODEL_PATH.parent))
    download.add_argument("--filename", default=DEFAULT_MODEL_FILENAME)
    download.add_argument("--revision", default=None)

    lab = subcommands.add_parser("lab", help="Open the visual two-model conversation laboratory")
    lab.add_argument("--model-a", required=True, help="First model path, normally a GGUF file")
    lab.add_argument("--model-b", required=True, help="Second local model path or Hugging Face model id")
    lab.add_argument("--backend-a", default="llama-cpp", choices=["llama-cpp", "hf", "echo"])
    lab.add_argument("--backend-b", default="hf", choices=["llama-cpp", "hf", "echo"])
    lab.add_argument("--output-dir", default="conversations")
    lab.add_argument("--port", type=int, default=8765)
    lab.add_argument("--no-browser", action="store_true")

    return parser


def add_generation_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--backend", choices=["echo", "llama-cpp", "hf", "transformers"], default="llama-cpp")
    parser.add_argument("--model-path", default=str(DEFAULT_MODEL_PATH), help="Local GGUF or Hugging Face model path")
    parser.add_argument("--adapter-path", default=None, help="Optional saved LoRA adapter directory for the HF backend")
    parser.add_argument("--device", default="cpu", help="Optional torch device such as cpu or cuda")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--no-log", action="store_true", help="Disable daily conversation logging")


def create_engine(args: argparse.Namespace) -> QueryEngine:
    return QueryEngine(
        model_path=args.model_path,
        backend=args.backend,
        device=args.device,
        adapter_path=args.adapter_path,
    )


def create_agent(args: argparse.Namespace) -> CodingAgent:
    def confirm(message: str) -> bool:
        answer = input(f"{message} [y/N] ").strip().casefold()
        return answer in {"y", "yes", "s", "si", "sí"}

    return CodingAgent(create_engine(args), args.workspace, confirm=confirm)


def run_once(args: argparse.Namespace) -> int:
    agent = create_agent(args)
    response = agent.answer(
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
    agent = create_agent(args)
    print('Agent started. Use /help for tools. Type "salir", "exit", or "quit" to finish.')
    while True:
        user_message = input("\nUsuario: ").strip()
        if not user_message:
            continue
        if user_message.lower() in {"salir", "exit", "quit"}:
            print("Closing chat.")
            return 0
        try:
            tool_result = agent.execute_tool_command(user_message)
            response = tool_result.content if tool_result else agent.answer(
                user_message,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                device=args.device,
            )
            if response == "quit":
                return 0
        except (FileNotFoundError, NotADirectoryError, PermissionError, ValueError) as exc:
            response = f"Tool error: {exc}"
        print("\nAssistant:")
        print(response)
        if not args.no_log:
            model_name = Path(args.model_path).name if args.model_path else args.backend
            append_conversation(model_name, user_message, response, metadata={"source": "cli_chat"})


def download_model(args: argparse.Namespace) -> int:
    from pathlib import Path

    try:
        from huggingface_hub import hf_hub_download
    except ModuleNotFoundError as exc:
        raise SystemExit("Install model dependencies first: python -m pip install -e .[llama]") from exc

    destination = Path(args.local_dir)
    destination.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {args.repo_id} to {destination}")
    downloaded = hf_hub_download(
        repo_id=args.repo_id,
        filename=args.filename,
        revision=args.revision,
        local_dir=str(destination),
    )
    print(f"Model ready at {downloaded}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "run":
        return run_once(args)
    if args.command == "chat":
        return run_chat(args)
    if args.command == "download-model":
        return download_model(args)
    if args.command == "lab":
        from conversation_lab import main as lab_main

        lab_args = [
            "--model-a", args.model_a,
            "--model-b", args.model_b,
            "--backend-a", args.backend_a,
            "--backend-b", args.backend_b,
            "--output-dir", args.output_dir,
            "--port", str(args.port),
        ]
        if args.no_browser:
            lab_args.append("--no-browser")
        return lab_main(lab_args)
    parser.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())