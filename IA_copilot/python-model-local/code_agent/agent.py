# This file has been created (totally or partially) with the assistance of artificial intelligence tools.
# All content has been generated under the direct supervision of a named individual,
# and under the AI.Backbone Orchestrator Compliance framework

"""Bounded local coding-agent orchestration."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Callable

from .query_engine import QueryEngine
from .retrieval import WorkspaceRetriever
from .tools import ToolResult, WorkspaceTools


@dataclass
class AgentSession:
    messages: list[tuple[str, str]] = field(default_factory=list)


class CodingAgent:
    """Model plus workspace context and explicit, confirmed tool commands."""

    def __init__(self, engine: QueryEngine, workspace_root: str | Path, confirm: Callable[[str], bool] | None = None) -> None:
        self.engine = engine
        self.tools = WorkspaceTools(workspace_root)
        self.retriever = WorkspaceRetriever(workspace_root)
        self.session = AgentSession()
        self.confirm = confirm or (lambda _message: False)

    def execute_tool_command(self, command: str) -> ToolResult | None:
        parts = command.strip().split(maxsplit=2)
        if not parts or not parts[0].startswith("/"):
            return None
        name = parts[0].lower()
        if name == "/tree":
            return self.tools.tree(parts[1] if len(parts) > 1 else ".")
        if name == "/read":
            if len(parts) < 2:
                raise ValueError("Usage: /read path [start:end]")
            path = parts[1]
            start, end = 1, 200
            if len(parts) == 3 and ":" in parts[2]:
                start_text, end_text = parts[2].split(":", 1)
                start, end = int(start_text), int(end_text)
            return self.tools.read(path, start, end)
        if name == "/find":
            if len(parts) < 2:
                raise ValueError("Usage: /find text [path]")
            query = parts[1]
            path = parts[2] if len(parts) == 3 else "."
            return self.tools.find(query, path)
        if name == "/context":
            query = command.partition(" ")[2]
            return ToolResult("context", self.retriever.search(query) or "No relevant context found.")
        if name in {"/replace", "/write", "/run"}:
            if len(parts) < 2:
                raise ValueError(f"Usage: {name} JSON_PAYLOAD_OR_COMMAND")
            if name == "/run":
                command = command.partition(" ")[2].strip()
                if not self.confirm(f"Execute allowlisted command? {command}"):
                    return ToolResult("run", "Execution cancelled.")
                return self.tools.run(command)
            try:
                payload = json.loads(command.partition(" ")[2])
            except json.JSONDecodeError as exc:
                raise ValueError("Edit commands require a JSON payload") from exc
            if name == "/replace":
                required = {"path", "old", "new"}
                if set(payload) < required:
                    raise ValueError("/replace requires path, old, and new")
                description = f"Replace text in {payload['path']}?"
                if not self.confirm(description):
                    return ToolResult("replace", "Edit cancelled.")
                return self.tools.replace(payload["path"], payload["old"], payload["new"])
            required = {"path", "content"}
            if set(payload) < required:
                raise ValueError("/write requires path and content")
            overwrite = bool(payload.get("overwrite", False))
            if not self.confirm(f"Write file {payload['path']}?"):
                return ToolResult("write", "Edit cancelled.")
            return self.tools.write(payload["path"], payload["content"], overwrite=overwrite)
        if name == "/help":
            return ToolResult("help", "/tree [path]\n/read path [start:end]\n/find text [path]\n/context question\n/replace JSON\n/write JSON\n/run command\n/quit")
        if name == "/quit":
            return ToolResult("quit", "quit")
        raise ValueError(f"Unknown command: {name}")

    def answer(
        self,
        user_message: str,
        include_context: bool = True,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.95,
        device: str | None = None,
    ) -> str:
        context = self.retriever.search(user_message) if include_context else ""
        prompt = user_message
        if context:
            prompt = f"Workspace context:\n{context}\n\nUser request:\n{user_message}"
        response = self.engine.generate(
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            device=device,
        )
        self.session.messages.extend([("user", user_message), ("assistant", response)])
        return response