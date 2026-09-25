"""FPCL system-prompt framework: Role, Context, Limits, Format.

Every agent in this project is configured with the same four-part structure
instead of a free-form system prompt. This keeps agent behavior predictable
and auditable: each part answers one question.

- role:    who is the agent, in one or two sentences
- context: what it knows / has access to right now
- limits:  what it must never do, and when it must hand off
- format:  the shape its response must take
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FPCLPrompt:
    role: str
    context: str
    limits: str
    format: str

    def render(self) -> str:
        return (
            f"# Role\n{self.role.strip()}\n\n"
            f"# Context\n{self.context.strip()}\n\n"
            f"# Limits\n{self.limits.strip()}\n\n"
            f"# Format\n{self.format.strip()}"
        )


def build_system_prompt(role: str, context: str, limits: str, format: str) -> str:
    """Convenience function for callers that don't need the dataclass itself."""
    return FPCLPrompt(role=role, context=context, limits=limits, format=format).render()
