"""Model adapters. The agent depends only on the small `complete` interface."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
import uuid
from typing import Any


class OpenAICompatibleModel:
    def __init__(self, model: str, base_url: str, api_key: str = "ollama") -> None:
        self.model = model
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.api_key = api_key

    def complete(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> dict[str, Any]:
        payload = json.dumps({"model": self.model, "messages": messages, "tools": tools, "temperature": 0}).encode()
        request = urllib.request.Request(
            self.url,
            data=payload,
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"},
        )
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                body = json.load(response)
        except (urllib.error.URLError, TimeoutError) as exc:
            raise RuntimeError(f"Could not reach model server at {self.url}: {exc}") from exc
        return body["choices"][0]["message"]


class OllamaModel:
    """Ollama's native API, including streamed, separately exposed thinking."""

    def __init__(self, model: str, base_url: str, show_thinking: bool = False) -> None:
        self.model = model
        root = base_url.rstrip("/")
        if root.endswith("/v1"):
            root = root[:-3]
        self.url = root + "/api/chat"
        self.show_thinking = show_thinking

    @staticmethod
    def _messages_for_ollama(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        call_names: dict[str, str] = {}
        converted: list[dict[str, Any]] = []
        for message in messages:
            item = {key: value for key, value in message.items() if key in {"role", "content", "thinking"}}
            calls = message.get("tool_calls") or []
            if calls:
                native_calls = []
                for call in calls:
                    function = dict(call.get("function", {}))
                    arguments = function.get("arguments", {})
                    if isinstance(arguments, str):
                        try:
                            arguments = json.loads(arguments)
                        except json.JSONDecodeError:
                            arguments = {}
                    function["arguments"] = arguments
                    native_calls.append({"function": function})
                    call_names[call.get("id", "")] = function.get("name", "")
                item["tool_calls"] = native_calls
            if message.get("role") == "tool":
                item["tool_name"] = call_names.get(message.get("tool_call_id", ""), "")
            converted.append(item)
        return converted

    def complete(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> dict[str, Any]:
        payload = json.dumps({
            "model": self.model,
            "messages": self._messages_for_ollama(messages),
            "tools": tools,
            "think": self.show_thinking,
            "stream": True,
            "options": {"temperature": 0},
        }).encode()
        request = urllib.request.Request(self.url, data=payload, headers={"Content-Type": "application/json"})
        thinking_parts: list[str] = []
        content_parts: list[str] = []
        native_calls: list[dict[str, Any]] = []
        printed_header = False
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                for raw_line in response:
                    if not raw_line.strip():
                        continue
                    chunk = json.loads(raw_line)
                    message = chunk.get("message", {})
                    thinking = message.get("thinking", "")
                    if thinking:
                        thinking_parts.append(thinking)
                        if self.show_thinking:
                            if not printed_header:
                                print("\n[model thinking]", flush=True)
                                printed_header = True
                            print(thinking, end="", flush=True)
                    if message.get("content"):
                        content_parts.append(message["content"])
                    native_calls.extend(message.get("tool_calls") or [])
        except (urllib.error.URLError, TimeoutError) as exc:
            raise RuntimeError(f"Could not reach Ollama at {self.url}: {exc}") from exc

        if printed_header:
            print("\n[end thinking]", flush=True)
        elif self.show_thinking:
            print("\n[model returned no separate thinking trace]", flush=True)

        reply: dict[str, Any] = {"role": "assistant", "content": "".join(content_parts) or None}
        thinking_text = "".join(thinking_parts)
        if thinking_text:
            reply["thinking"] = thinking_text
        if native_calls:
            reply["tool_calls"] = []
            for call in native_calls:
                function = call.get("function", {})
                arguments = function.get("arguments", {})
                reply["tool_calls"].append({
                    "id": f"call-{uuid.uuid4().hex}",
                    "type": "function",
                    "function": {
                        "name": function.get("name", ""),
                        "arguments": arguments if isinstance(arguments, str) else json.dumps(arguments),
                    },
                })
        return reply


class DemoModel:
    """A scripted model proving the tool loop works without a model server."""

    def __init__(self) -> None:
        self.turn = 0

    def complete(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> dict[str, Any]:
        self.turn += 1
        if self.turn == 1:
            return {
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": "demo-call-1",
                    "type": "function",
                    "function": {"name": "calculate", "arguments": '{"expression":"(17 * 23) + 11"}'},
                }],
            }
        result = messages[-1]["content"]
        return {"role": "assistant", "content": f"The calculator returned {result}. The agent loop is working!"}
