# This file has been created (totally or partially) with the assistance of artificial intelligence tools.
# All content has been generated under the direct supervision of a named individual,
# and under the AI.Backbone Orchestrator Compliance framework

"""Small dependency-free lexical retrieval layer."""

from __future__ import annotations

import re
from pathlib import Path

from .policy import WorkspacePolicy
from .tools import TEXT_EXTENSIONS


class WorkspaceRetriever:
    """Find relevant text snippets without an embeddings dependency."""

    def __init__(self, workspace_root: str | Path, max_file_bytes: int = 100_000) -> None:
        self.policy = WorkspacePolicy(workspace_root)
        self.max_file_bytes = max_file_bytes

    def search(self, query: str, max_results: int = 5, max_lines: int = 8) -> str:
        terms = {term.casefold() for term in re.findall(r"[\w.-]+", query) if len(term) > 2}
        if not terms:
            return ""
        scored: list[tuple[int, Path, list[str]]] = []
        for path in self.policy.root.rglob("*"):
            if not path.is_file() or self.policy.is_sensitive(path) or path.suffix.lower() not in TEXT_EXTENSIONS:
                continue
            try:
                lines = path.read_bytes()[: self.max_file_bytes].decode("utf-8", errors="replace").splitlines()
            except OSError:
                continue
            hits = [line for line in lines if terms.intersection(re.findall(r"[\w.-]+", line.casefold()))]
            if hits:
                score = sum(sum(term in line.casefold() for term in terms) for line in hits)
                scored.append((score, path, hits[:max_lines]))
        scored.sort(key=lambda item: (-item[0], str(item[1])))
        blocks = []
        for score, path, lines in scored[:max_results]:
            relative = path.relative_to(self.policy.root)
            blocks.append(f"[{relative}] score={score}\n" + "\n".join(lines))
        return "\n\n".join(blocks)