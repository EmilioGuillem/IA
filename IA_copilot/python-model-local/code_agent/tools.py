# This file has been created (totally or partially) with the assistance of artificial intelligence tools.
# All content has been generated under the direct supervision of a named individual,
# and under the AI.Backbone Orchestrator Compliance framework

"""Policy-gated workspace tools."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import subprocess
import time

from .policy import WorkspacePolicy


TEXT_EXTENSIONS = {
    ".py", ".ts", ".tsx", ".js", ".jsx", ".json", ".md", ".txt", ".yml", ".yaml",
    ".toml", ".ini", ".cfg", ".css", ".html", ".sql", ".java", ".cs", ".go",
}


@dataclass(frozen=True)
class ToolResult:
    tool: str
    content: str
    truncated: bool = False


class WorkspaceTools:
    """Bounded file inspection tools backed by a workspace policy."""

    def __init__(self, workspace_root: str | Path, max_bytes: int = 32_000) -> None:
        self.policy = WorkspacePolicy(workspace_root)
        self.max_bytes = max_bytes
        self.audit_path = self.policy.root / ".code-agent-audit.jsonl"

    def tree(self, relative_path: str = ".", max_entries: int = 200) -> ToolResult:
        root = self.policy.resolve(relative_path)
        if not root.is_dir():
            raise NotADirectoryError(str(root))
        entries: list[str] = []
        for path in sorted(root.rglob("*")):
            if len(entries) >= max_entries:
                break
            if self.policy.is_ignored_directory(path) or self.policy.is_sensitive(path):
                continue
            entries.append(str(path.relative_to(self.policy.root)))
        truncated = len(entries) >= max_entries
        suffix = "\n[truncated]" if truncated else ""
        return ToolResult("tree", "\n".join(entries) + suffix, truncated)

    def read(self, relative_path: str, start_line: int = 1, end_line: int = 200) -> ToolResult:
        path = self.policy.resolve(relative_path)
        if not path.is_file():
            raise FileNotFoundError(str(path))
        if path.suffix.lower() not in TEXT_EXTENSIONS:
            raise ValueError("Only known text file extensions can be read")
        raw = path.read_bytes()
        truncated = len(raw) > self.max_bytes
        text = raw[: self.max_bytes].decode("utf-8", errors="replace")
        lines = text.splitlines()
        selected = lines[max(0, start_line - 1):end_line]
        numbered = [f"{index}: {line}" for index, line in enumerate(selected, start=max(1, start_line))]
        return ToolResult("read", "\n".join(numbered), truncated)

    def find(self, query: str, relative_path: str = ".", max_matches: int = 100) -> ToolResult:
        root = self.policy.resolve(relative_path)
        matches: list[str] = []
        for path in root.rglob("*") if root.is_dir() else [root]:
            if len(matches) >= max_matches or not path.is_file():
                break
            if self.policy.is_sensitive(path) or path.suffix.lower() not in TEXT_EXTENSIONS:
                continue
            try:
                for line_number, line in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
                    if query.casefold() in line.casefold():
                        matches.append(f"{path.relative_to(self.policy.root)}:{line_number}: {line.strip()}")
                        if len(matches) >= max_matches:
                            break
            except OSError:
                continue
        truncated = len(matches) >= max_matches
        return ToolResult("find", "\n".join(matches) or "No matches found.", truncated)

    def replace(self, relative_path: str, old: str, new: str) -> ToolResult:
        """Replace text in a workspace file after policy validation."""
        path = self.policy.resolve(relative_path)
        if not path.is_file():
            raise FileNotFoundError(str(path))
        if path.suffix.lower() not in TEXT_EXTENSIONS:
            raise ValueError("Only known text file extensions can be edited")
        text = path.read_text(encoding="utf-8")
        occurrences = text.count(old)
        if occurrences == 0:
            raise ValueError("The requested text was not found")
        if occurrences > 1:
            raise ValueError("The requested text occurs more than once; make the edit more specific")
        updated = text.replace(old, new, 1)
        self._write_text(path, updated)
        self._audit("replace", str(path.relative_to(self.policy.root)))
        return ToolResult("replace", f"Updated {path.relative_to(self.policy.root)}")

    def write(self, relative_path: str, content: str, overwrite: bool = False) -> ToolResult:
        """Create or overwrite a text file inside the workspace."""
        path = self.policy.resolve(relative_path)
        if path.exists() and not overwrite:
            raise FileExistsError(f"File exists; pass overwrite=true: {path}")
        if path.suffix.lower() not in TEXT_EXTENSIONS:
            raise ValueError("Only known text file extensions can be written")
        if len(content.encode("utf-8")) > self.max_bytes:
            raise ValueError("Content exceeds the write size limit")
        path.parent.mkdir(parents=True, exist_ok=True)
        self._write_text(path, content)
        self._audit("write", str(path.relative_to(self.policy.root)))
        return ToolResult("write", f"Wrote {path.relative_to(self.policy.root)}")

    def run(self, command: str, timeout: int = 30, max_output: int = 20_000) -> ToolResult:
        """Run one allowlisted command without shell expansion."""
        if timeout < 1 or timeout > 300:
            raise ValueError("Timeout must be between 1 and 300 seconds")
        parts = self.policy.validate_command(command)
        started = time.monotonic()
        try:
            completed = subprocess.run(
                parts,
                cwd=self.policy.root,
                shell=False,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout,
                check=False,
            )
            output = (completed.stdout + completed.stderr).strip()
            if len(output) > max_output:
                output = output[:max_output] + "\n[output truncated]"
            content = f"exit_code={completed.returncode}\nduration={time.monotonic() - started:.2f}s\n{output}"
        except subprocess.TimeoutExpired as exc:
            content = f"timeout after {timeout}s\n{exc}"
        self._audit("run", command)
        return ToolResult("run", content, len(content) >= max_output)

    def _write_text(self, path: Path, content: str) -> None:
        path.write_text(content, encoding="utf-8", newline="")

    def _audit(self, operation: str, target: str) -> None:
        record = {"timestamp": time.time(), "operation": operation, "target": target}
        with self.audit_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")