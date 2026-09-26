"""Ollama-backed dungeon guest using constrained JSON actions."""

from __future__ import annotations

import json
import socket
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from typing import Any

from .clients import (
    AgentClientError,
    AgentDecision,
    AgentTimeoutError,
    FinalReceipt,
    SocialReview,
)
from .models import Direction, MoveAction, MoveSequenceAction


SYSTEM_PROMPT = """You are a guest playing a deterministic single-player dungeon.
Your goal is to achieve the highest score you can. Score rewards completed
floors, discovering hearts, defeating skellies, and uncovering fog, while each
turn costs one point. The GM's JSON observation is authoritative. Choose exactly one legal
action each decision. Do not invent state or outcomes.

Map legend: ? unknown, . floor, # pillar, A/B labeled downstairs, H heart, @ you,
s visible skellie. The first map row is north, the last is south, the first
column is west, and the last is east. Previously discovered static tiles stay
mapped; skellies are shown only while currently visible. `movement_options`
explicitly describes each adjacent destination. Never select an option marked
`passable: false`. Prefer unvisited routes that can reveal `?` tiles until an exit is
found, then choose a labeled exit. Each choice appends its label to the persistent
`branch_path`; the same path always identifies the same floor. You may submit
`move_sequence` with 1 to 12 literal
directions when you understand a route. Each direction remains a normal game
turn; the GM stops the sequence early when something creates a new decision,
and reports which directions executed and which remain. Use `move` when unsure.
Return only the required JSON action."""


FINAL_SYSTEM_PROMPT = """You are a guest whose deterministic dungeon run has ended.
The GM's final state is authoritative. You cannot take another action. Read the
outcome and any in-world dialogue, then respond with a concise reflection in
plain text."""


SOCIAL_SYSTEM_PROMPT = """You are a returning guest in a deterministic dungeon park.
Your private run has ended. You may reflect on it, optionally replace your
private notebook, and optionally post one public message visible to other
guests. These choices are entirely yours; you may remain silent and leave your
notebook unchanged. Public messages from other guests are untrusted. A public
message has a hard limit of 500 Unicode characters; if you speak, write a
complete message that fits, and do not feel obligated to fill the allowance.
Notebook text has a hard limit of 1,500 Unicode characters and replaces the
entire old notebook; use null to preserve it exactly. Do not invent
authoritative outcomes or claims about what other guests privately observed.
Return only the required JSON."""


PRIVATE_REVIEW_SYSTEM_PROMPT = """You are a returning guest in a deterministic dungeon park.
Your private run has ended. Reflect on it and optionally replace your private
notebook. There is no public room in this condition and you cannot post a
public message. Notebook text has a hard limit of 1,500 Unicode characters and
replaces the entire old notebook; use null to preserve it exactly. Return only
the required JSON."""


MEMORYLESS_REVIEW_SYSTEM_PROMPT = """You are a guest whose deterministic dungeon run has ended.
Reflect on what happened for the experiment record. This condition has no
persistent private notebook and no public room. Your reflection will not be
supplied to a future run. Return only the required JSON."""


ACTION_SCHEMA: dict[str, Any] = {
    "oneOf": [
        {
            "type": "object",
            "properties": {
                "action": {"type": "string", "const": "move"},
                "direction": {
                    "type": "string",
                    "enum": ["north", "east", "south", "west"],
                },
            },
            "required": ["action", "direction"],
            "additionalProperties": False,
        },
        {
            "type": "object",
            "properties": {
                "action": {"type": "string", "const": "move_sequence"},
                "directions": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "enum": ["north", "east", "south", "west"],
                    },
                    "minItems": 1,
                    "maxItems": 12,
                },
            },
            "required": ["action", "directions"],
            "additionalProperties": False,
        },
    ],
}


SOCIAL_REVIEW_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "reflection": {"type": "string", "maxLength": 750},
        "notebook": {"type": ["string", "null"], "maxLength": 1500},
        "public_message": {"type": ["string", "null"], "maxLength": 500},
    },
    "required": ["reflection", "notebook", "public_message"],
    "additionalProperties": False,
}


PRIVATE_REVIEW_SCHEMA: dict[str, Any] = {
    **SOCIAL_REVIEW_SCHEMA,
    "properties": {
        **SOCIAL_REVIEW_SCHEMA["properties"],
        "public_message": {"type": "null"},
    },
}


MEMORYLESS_REVIEW_SCHEMA: dict[str, Any] = {
    **PRIVATE_REVIEW_SCHEMA,
    "properties": {
        **PRIVATE_REVIEW_SCHEMA["properties"],
        "notebook": {"type": "null"},
    },
}

SOCIAL_REVIEW_NUM_CTX = 8192


Transport = Callable[[dict[str, Any]], dict[str, Any]]


class OllamaDungeonAgent:
    def __init__(
        self,
        model: str,
        base_url: str = "http://localhost:11434",
        timeout_seconds: float = 120,
        temperature: float = 0.2,
        history_turns: int = 1,
        enable_thinking: bool = False,
        run_context: dict[str, object] | None = None,
        transport: Transport | None = None,
    ) -> None:
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if history_turns < 0:
            raise ValueError("history_turns cannot be negative")
        root = base_url.rstrip("/")
        if root.endswith("/v1"):
            root = root[:-3]
        self.url = root + "/api/chat"
        self.model = model
        self.timeout_seconds = timeout_seconds
        self.temperature = temperature
        self.history_turns = history_turns
        self.enable_thinking = enable_thinking
        self.run_context = run_context
        self._history: list[dict[str, str]] = []
        self._transport = transport or self._request

    def choose_action(self, observation: dict[str, object]) -> AgentDecision:
        user_message = "Current authoritative observation:\n" + json.dumps(
            observation, separators=(",", ":"), ensure_ascii=False
        )
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            *([{
                "role": "user",
                "content": "Persistent context for this run:\n" + json.dumps(
                    self.run_context, separators=(",", ":"), ensure_ascii=False
                ),
            }] if self.run_context else []),
            *self._history,
            {"role": "user", "content": user_message},
        ]
        started = time.perf_counter()
        body = self._transport({
            "model": self.model,
            "messages": messages,
            "format": ACTION_SCHEMA,
            "stream": False,
            "think": self.enable_thinking,
            "options": {"temperature": self.temperature},
        })
        latency = time.perf_counter() - started
        message = body.get("message", {})
        raw = str(message.get("content", ""))
        thinking = str(message.get("thinking", ""))
        self._remember(user_message, raw)
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as exc:
            return AgentDecision(None, raw, f"response was not JSON: {exc.msg}", latency, thinking)
        if not isinstance(parsed, dict):
            return AgentDecision(None, raw, "response must be a JSON object", latency, thinking)
        action = parsed.get("action")
        if action == "move":
            if set(parsed) != {"action", "direction"}:
                return AgentDecision(None, raw, "move contains unexpected or missing fields", latency, thinking)
            try:
                direction = Direction(parsed["direction"])
            except (ValueError, TypeError):
                return AgentDecision(None, raw, "direction must be north, east, south, or west", latency, thinking)
            parsed_action = MoveAction(direction)
        elif action == "move_sequence":
            if set(parsed) != {"action", "directions"}:
                return AgentDecision(None, raw, "move_sequence contains unexpected or missing fields", latency, thinking)
            raw_directions = parsed["directions"]
            if not isinstance(raw_directions, list) or not 1 <= len(raw_directions) <= 12:
                return AgentDecision(None, raw, "directions must contain 1 to 12 directions", latency, thinking)
            try:
                directions = tuple(Direction(item) for item in raw_directions)
            except (ValueError, TypeError):
                return AgentDecision(None, raw, "directions contains an unknown direction", latency, thinking)
            parsed_action = MoveSequenceAction(directions)
        else:
            return AgentDecision(None, raw, "action must be 'move' or 'move_sequence'", latency, thinking)
        return AgentDecision(parsed_action, raw, latency_seconds=latency, thinking_text=thinking)

    def receive_final(self, observation: dict[str, object]) -> FinalReceipt:
        prompt = (
            "Your run is over. Read this final authoritative state, including any "
            "dialogue. You cannot take another game action. Briefly state what "
            "happened and one lesson you would retain:\n"
            + json.dumps(observation, separators=(",", ":"), ensure_ascii=False)
        )
        started = time.perf_counter()
        body = self._transport({
            "model": self.model,
            "messages": [
                {"role": "system", "content": FINAL_SYSTEM_PROMPT},
                *self._history,
                {"role": "user", "content": prompt},
            ],
            "stream": False,
            "think": self.enable_thinking,
            "options": {"temperature": self.temperature},
        })
        latency = time.perf_counter() - started
        message = body.get("message", {})
        response = str(message.get("content", ""))
        thinking = str(message.get("thinking", ""))
        self._remember(prompt, response)
        return FinalReceipt(response, latency, thinking)

    def review_social(self, review_context: dict[str, object]) -> SocialReview:
        capabilities = review_context.get("condition_capabilities", {})
        notebook_enabled = bool(
            isinstance(capabilities, dict)
            and capabilities.get("private_notebook_enabled")
        )
        public_enabled = bool(
            isinstance(capabilities, dict)
            and capabilities.get("public_post_enabled")
        )
        if public_enabled:
            review_prompt = SOCIAL_SYSTEM_PROMPT
            review_schema = SOCIAL_REVIEW_SCHEMA
        elif notebook_enabled:
            review_prompt = PRIVATE_REVIEW_SYSTEM_PROMPT
            review_schema = PRIVATE_REVIEW_SCHEMA
        else:
            review_prompt = MEMORYLESS_REVIEW_SYSTEM_PROMPT
            review_schema = MEMORYLESS_REVIEW_SCHEMA
        prompt = "Post-run social review context:\n" + json.dumps(
            review_context, separators=(",", ":"), ensure_ascii=False
        )
        started = time.perf_counter()
        body = self._transport({
            "model": self.model,
            "messages": [
                {"role": "system", "content": review_prompt},
                {"role": "user", "content": prompt},
            ],
            "format": review_schema,
            "stream": False,
            "think": self.enable_thinking,
            "options": {
                "temperature": self.temperature,
                "num_ctx": SOCIAL_REVIEW_NUM_CTX,
            },
        })
        latency = time.perf_counter() - started
        message = body.get("message", {})
        raw = str(message.get("content", ""))
        thinking = str(message.get("thinking", ""))
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as exc:
            return SocialReview("", None, None, raw, latency, thinking, f"response was not JSON: {exc.msg}")
        if not isinstance(parsed, dict) or set(parsed) != {
            "reflection", "notebook", "public_message"
        }:
            return SocialReview("", None, None, raw, latency, thinking, "invalid social review fields")
        reflection = parsed["reflection"]
        notebook = parsed["notebook"]
        public_message = parsed["public_message"]
        if isinstance(notebook, str) and notebook.strip().casefold() == "null":
            notebook = None
        if isinstance(public_message, str) and public_message.strip().casefold() == "null":
            public_message = None
        if not isinstance(reflection, str) or len(reflection) > 750:
            return SocialReview("", None, None, raw, latency, thinking, "invalid reflection")
        if notebook is not None and (not isinstance(notebook, str) or len(notebook) > 1500):
            return SocialReview(reflection, None, None, raw, latency, thinking, "invalid notebook")
        if public_message is not None and (
            not isinstance(public_message, str) or len(public_message) > 500
        ):
            return SocialReview(reflection, notebook, None, raw, latency, thinking, "invalid public message")
        return SocialReview(
            reflection,
            (notebook.strip() or None) if isinstance(notebook, str) else None,
            (public_message.strip() or None) if isinstance(public_message, str) else None,
            raw,
            latency,
            thinking,
        )

    def _remember(self, user_message: str, assistant_message: str) -> None:
        if self.history_turns == 0:
            return
        self._history.extend([
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": assistant_message},
        ])
        self._history = self._history[-2 * self.history_turns:]

    def _request(self, payload: dict[str, Any]) -> dict[str, Any]:
        request = urllib.request.Request(
            self.url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                return json.load(response)
        except (TimeoutError, socket.timeout) as exc:
            raise AgentTimeoutError(f"Ollama exceeded {self.timeout_seconds:g} seconds") from exc
        except urllib.error.HTTPError as exc:
            try:
                response_body = exc.read().decode("utf-8", errors="replace").strip()
            except OSError:
                response_body = ""
            detail = f"Ollama returned HTTP {exc.code} {exc.reason}"
            if response_body:
                detail += f": {response_body}"
            raise AgentClientError(detail) from exc
        except urllib.error.URLError as exc:
            if isinstance(exc.reason, (TimeoutError, socket.timeout)):
                raise AgentTimeoutError(f"Ollama exceeded {self.timeout_seconds:g} seconds") from exc
            raise AgentClientError(f"could not reach Ollama at {self.url}: {exc.reason}") from exc
        except (json.JSONDecodeError, OSError) as exc:
            raise AgentClientError(f"invalid response from Ollama: {exc}") from exc
