"""Specialist agents: sales, support, billing.

Each agent is just an FPCL system prompt plus the shared tool-calling loop.
Adding a fourth specialist means writing one new FPCLPrompt, not new
orchestration code.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .claude_client import ClaudeClient
from .prompts import FPCLPrompt
from .tools import TOOL_SCHEMAS, ToolExecutor

MAX_TOOL_ITERATIONS = 4


@dataclass
class AgentResult:
    agent_name: str
    text: str
    escalated: bool = False
    escalation_reason: str | None = None
    tool_calls_made: list[str] = None  # tool names, in order
    usage: dict[str, int] | None = None  # Etapa 14: summed across every .send() in this handle()

    def __post_init__(self):
        if self.tool_calls_made is None:
            self.tool_calls_made = []


def _sum_usage(a: dict[str, int] | None, b: dict[str, int] | None) -> dict[str, int] | None:
    """Adds two usage dicts (either may be None if the client didn't
    report usage, e.g. hand-built ClaudeResponse in a test). Returns None
    only if BOTH are None -- a single real API call is enough to start
    tracking real cost for the rest of the loop."""
    if a is None and b is None:
        return None
    a = a or {"input_tokens": 0, "output_tokens": 0}
    b = b or {"input_tokens": 0, "output_tokens": 0}
    return {
        "input_tokens": a["input_tokens"] + b["input_tokens"],
        "output_tokens": a["output_tokens"] + b["output_tokens"],
    }


_SHARED_LIMITS = (
    "Never invent a policy detail (price, timeframe, refund rule) — look it up with "
    "search_knowledge_base first. If the knowledge base doesn't cover it, or the "
    "customer needs an account-specific action you cannot perform, call "
    "escalate_to_human instead of guessing."
)
_SHARED_FORMAT = (
    "Reply in plain, friendly prose, 2-4 sentences unless the customer asked for a list. "
    "No markdown headers. Answer in the same language the customer wrote in."
)


class Specialist:
    """One specialist agent: FPCL prompt + the standard tool-use loop."""

    def __init__(
        self,
        name: str,
        prompt: FPCLPrompt,
        client: ClaudeClient,
        tool_executor: ToolExecutor,
    ):
        self.name = name
        self._system = prompt.render()
        self._client = client
        self._tools = tool_executor

    def handle(self, conversation: list[dict[str, str]]) -> AgentResult:
        messages: list[dict[str, Any]] = list(conversation)
        tool_calls_made: list[str] = []
        usage: dict[str, int] | None = None

        for _ in range(MAX_TOOL_ITERATIONS):
            response = self._client.send(
                system=self._system,
                messages=messages,
                tools=TOOL_SCHEMAS,
            )
            usage = _sum_usage(usage, response.usage)

            if response.stop_reason != "tool_use" or not response.tool_calls:
                return AgentResult(
                    agent_name=self.name,
                    text=response.text,
                    tool_calls_made=tool_calls_made,
                    usage=usage,
                )

            # Claude asked to call one or more tools: execute them locally and
            # feed the results back so it can produce a final answer.
            assistant_content = [
                {"type": "tool_use", "id": tc["id"], "name": tc["name"], "input": tc["input"]}
                for tc in response.tool_calls
            ]
            if response.text:
                assistant_content.insert(0, {"type": "text", "text": response.text})
            messages.append({"role": "assistant", "content": assistant_content})

            tool_result_content = []
            for tool_call in response.tool_calls:
                tool_calls_made.append(tool_call["name"])
                result = self._tools.execute(tool_call)
                tool_result_content.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": result.tool_use_id,
                        "content": result.content,
                    }
                )
                if result.escalated:
                    return AgentResult(
                        agent_name=self.name,
                        text=result.content,
                        escalated=True,
                        escalation_reason=result.escalation_reason,
                        tool_calls_made=tool_calls_made,
                        usage=usage,
                    )
            messages.append({"role": "user", "content": tool_result_content})

        # Safety valve: tool-calling loop did not converge to a final answer.
        return AgentResult(
            agent_name=self.name,
            text=(
                "I wasn't able to finish looking this up. Let me connect you with a "
                "human agent."
            ),
            escalated=True,
            escalation_reason="tool-use loop exceeded max iterations",
            tool_calls_made=tool_calls_made,
            usage=usage,
        )


def build_sales_agent(client: ClaudeClient, tool_executor: ToolExecutor) -> Specialist:
    prompt = FPCLPrompt(
        role="You are a sales agent for a B2B SaaS product.",
        context=(
            "You handle pricing questions, plan comparisons, upgrades, and trial "
            "requests. Plan details live in the knowledge base, not in your own memory."
        ),
        limits=_SHARED_LIMITS
        + " Never quote a custom Enterprise price yourself — always route that to a human.",
        format=_SHARED_FORMAT,
    )
    return Specialist("sales", prompt, client, tool_executor)


def build_support_agent(client: ClaudeClient, tool_executor: ToolExecutor) -> Specialist:
    prompt = FPCLPrompt(
        role="You are a technical support agent for a B2B SaaS product.",
        context="You handle how-to questions, troubleshooting, and shipping/delivery questions.",
        limits=_SHARED_LIMITS,
        format=_SHARED_FORMAT,
    )
    return Specialist("support", prompt, client, tool_executor)


def build_billing_agent(client: ClaudeClient, tool_executor: ToolExecutor) -> Specialist:
    prompt = FPCLPrompt(
        role="You are a billing agent for a B2B SaaS product.",
        context="You handle invoices, refunds, payment methods, and failed-payment questions.",
        limits=_SHARED_LIMITS
        + " Never process a refund yourself — confirm the policy, then escalate "
        "any action that actually moves money.",
        format=_SHARED_FORMAT,
    )
    return Specialist("billing", prompt, client, tool_executor)
