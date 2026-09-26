"""OpenAI-compatible transport for the dungeon guest.

The parsing and prompt construction remain shared with the Ollama client.  This
adapter only translates Ollama's structured-output request shape to the
OpenAI-compatible chat-completions shape used by servers such as vLLM.
"""

from __future__ import annotations

import json
import socket
import urllib.error
import urllib.request
from typing import Any

from .clients import AgentClientError, AgentTimeoutError
from .ollama_agent import OllamaDungeonAgent


class OpenAICompatibleDungeonAgent(OllamaDungeonAgent):
    def __init__(
        self,
        model: str,
        base_url: str = "http://localhost:8000/v1",
        api_key: str | None = None,
        **kwargs: Any,
    ) -> None:
        root = base_url.rstrip("/")
        if not root.endswith("/v1"):
            root += "/v1"
        self.openai_url = root + "/chat/completions"
        self.api_key = api_key
        super().__init__(model=model, base_url="http://unused", **kwargs)
        self._transport = self._request_openai

    def _request_openai(self, payload: dict[str, Any]) -> dict[str, Any]:
        request_payload: dict[str, Any] = {
            "model": payload["model"],
            "messages": payload["messages"],
            "stream": False,
            "temperature": payload.get("options", {}).get("temperature", 0.2),
        }
        schema = payload.get("format")
        if schema is not None:
            request_payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "dungeon_response",
                    "schema": schema,
                    "strict": True,
                },
            }
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        request = urllib.request.Request(
            self.openai_url,
            data=json.dumps(request_payload).encode("utf-8"),
            headers=headers,
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                body = json.load(response)
        except (TimeoutError, socket.timeout) as exc:
            raise AgentTimeoutError(
                f"OpenAI-compatible server exceeded {self.timeout_seconds:g} seconds"
            ) from exc
        except urllib.error.HTTPError as exc:
            try:
                response_body = exc.read().decode("utf-8", errors="replace").strip()
            except OSError:
                response_body = ""
            detail = f"OpenAI-compatible server returned HTTP {exc.code} {exc.reason}"
            if response_body:
                detail += f": {response_body}"
            raise AgentClientError(detail) from exc
        except urllib.error.URLError as exc:
            if isinstance(exc.reason, (TimeoutError, socket.timeout)):
                raise AgentTimeoutError(
                    f"OpenAI-compatible server exceeded {self.timeout_seconds:g} seconds"
                ) from exc
            raise AgentClientError(
                f"could not reach OpenAI-compatible server at {self.openai_url}: {exc.reason}"
            ) from exc
        except (json.JSONDecodeError, OSError) as exc:
            raise AgentClientError(f"invalid response from OpenAI-compatible server: {exc}") from exc

        try:
            message = body["choices"][0]["message"]
        except (KeyError, IndexError, TypeError) as exc:
            raise AgentClientError("OpenAI-compatible response has no assistant message") from exc
        return {
            "message": {
                "content": message.get("content") or "",
                "thinking": (
                    message.get("reasoning_content")
                    or message.get("reasoning")
                    or ""
                ),
            }
        }
