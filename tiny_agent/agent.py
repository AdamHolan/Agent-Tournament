"""The orchestration loop: observe, choose an action, execute, repeat."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Protocol

from .tools import Tool


SYSTEM_PROMPT = """You are a small local assistant. Use tools when they help.
Never invent tool results. Paths must be relative to the workspace. After a tool
result, either use another tool or give the user a concise final answer."""


class Model(Protocol):
    def complete(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> dict[str, Any]: ...


@dataclass
class AgentResult:
    answer: str
    steps: int
    messages: list[dict[str, Any]]


class Agent:
    def __init__(self, model: Model, tools: dict[str, Tool], max_steps: int = 8, verbose: bool = True) -> None:
        self.model = model
        self.tools = tools
        self.max_steps = max_steps
        self.verbose = verbose

    def run(self, goal: str) -> AgentResult:
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": goal},
        ]
        schemas = [tool.schema() for tool in self.tools.values()]

        for step in range(1, self.max_steps + 1):
            reply = self.model.complete(messages, schemas)
            messages.append(reply)
            calls = reply.get("tool_calls") or []
            if not calls:
                answer = reply.get("content") or "Model returned no answer."
                return AgentResult(answer, step, messages)

            for call in calls:
                function = call.get("function", {})
                name = function.get("name", "")
                if self.verbose:
                    print(f"[step {step}] model requested tool: {name}")
                try:
                    arguments = json.loads(function.get("arguments") or "{}")
                    tool = self.tools.get(name)
                    if tool is None:
                        result = f"Error: unknown tool {name!r}"
                    elif not isinstance(arguments, dict):
                        result = "Error: tool arguments must be a JSON object"
                    else:
                        result = tool.function(**arguments)
                except (ValueError, TypeError, OSError, json.JSONDecodeError) as exc:
                    result = f"Error: {exc}"
                if self.verbose:
                    print(f"[step {step}] tool result: {result[:300]}")
                messages.append({"role": "tool", "tool_call_id": call.get("id", "missing-id"), "content": result})

        return AgentResult(f"Stopped after the safety limit of {self.max_steps} steps.", self.max_steps, messages)
