# This file has been created (totally or partially) with the assistance of artificial intelligence tools.
# All content has been generated under the direct supervision of a named individual,
# and under the AI.Backbone Orchestrator Compliance framework

"""Safety policy for local workspace access."""

from __future__ import annotations

from pathlib import Path
import shlex


SENSITIVE_NAMES = {
    ".env",
    ".env.local",
    ".env.production",
    "id_rsa",
    "id_ed25519",
}
SENSITIVE_SUFFIXES = {".pem", ".key", ".p12", ".pfx"}
IGNORED_DIRECTORIES = {".git", ".venv", "venv", "node_modules", "__pycache__", ".pytest_cache"}
ALLOWED_COMMANDS = {"python", "python.exe", "py", "pytest", "pytest.exe", "ruff", "ruff.exe", "git", "git.exe"}
BLOCKED_COMMAND_ARGUMENTS = {
    "-c", "--command", "reset", "clean", "push", "checkout", "restore", "commit",
    "add", "rm", "mv", "rebase", "merge", "cherry-pick", "revert", "stash",
}


class WorkspacePolicy:
    """Allow access only to non-sensitive files below one workspace root."""

    def __init__(self, workspace_root: str | Path) -> None:
        self.root = Path(workspace_root).expanduser().resolve()
        if not self.root.is_dir():
            raise NotADirectoryError(f"Workspace does not exist: {self.root}")

    def resolve(self, requested_path: str | Path = ".") -> Path:
        candidate = (self.root / requested_path).resolve()
        try:
            candidate.relative_to(self.root)
        except ValueError as exc:
            raise PermissionError("Path is outside the workspace") from exc
        if self.is_sensitive(candidate):
            raise PermissionError("Access to sensitive files is blocked")
        return candidate

    def is_sensitive(self, path: Path) -> bool:
        if any(part in IGNORED_DIRECTORIES for part in path.relative_to(self.root).parts):
            return True
        return path.name in SENSITIVE_NAMES or path.suffix.lower() in SENSITIVE_SUFFIXES

    def is_ignored_directory(self, path: Path) -> bool:
        return path.name in IGNORED_DIRECTORIES

    def validate_command(self, command: str) -> list[str]:
        """Validate a command without invoking a shell."""
        parts = shlex.split(command, posix=True)
        if not parts:
            raise ValueError("Command cannot be empty")
        executable = Path(parts[0]).name.casefold()
        if executable not in ALLOWED_COMMANDS:
            raise PermissionError(f"Command is not allowlisted: {executable}")
        normalized = {part.casefold() for part in parts[1:]}
        if normalized.intersection(BLOCKED_COMMAND_ARGUMENTS):
            raise PermissionError("Command contains a blocked argument")
        if executable in {"python", "python.exe", "py"} and "-m" in normalized:
            module_index = next(index for index, part in enumerate(parts[1:], 1) if part.casefold() == "-m")
            module = parts[module_index + 1].casefold() if module_index + 1 < len(parts) else ""
            if module in {"pip", "ensurepip", "venv"}:
                raise PermissionError("Package installation and environment creation are blocked")
        return parts