# This file has been created (totally or partially) with the assistance of artificial intelligence tools.
# All content has been generated under the direct supervision of a named individual,
# and under the AI.Backbone Orchestrator Compliance framework

"""Prompt construction for the local coding agent."""

SYSTEM_PROMPT = """
You are a local coding assistant running on the user's machine.
Help with programming tasks using concise, practical answers.
When you mention files, prefer explicit paths supplied by the user or discovered by tools.
Do not claim to have executed code unless a tool or command actually did so.
"""

CHAT_PROMPT_TEMPLATE = """<|system|>
{system_prompt}
<|user|>
{user_message}
<|assistant|>
"""


def build_chat_prompt(user_message: str, system_prompt: str = SYSTEM_PROMPT) -> str:
    """Build a neutral chat prompt for causal language models."""
    return CHAT_PROMPT_TEMPLATE.format(
        system_prompt=system_prompt.strip(),
        user_message=user_message.strip(),
    )