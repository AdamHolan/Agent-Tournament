"""Construct model clients without coupling runners to one inference server."""

from __future__ import annotations

from typing import Any

from .ollama_agent import OllamaDungeonAgent
from .openai_agent import OpenAICompatibleDungeonAgent


BACKENDS = ("ollama", "vllm")


def create_dungeon_client(
    *,
    backend: str,
    model: str,
    base_url: str,
    api_key: str | None = None,
    **kwargs: Any,
) -> OllamaDungeonAgent:
    if backend == "ollama":
        return OllamaDungeonAgent(model=model, base_url=base_url, **kwargs)
    if backend == "vllm":
        return OpenAICompatibleDungeonAgent(
            model=model,
            base_url=base_url,
            api_key=api_key,
            **kwargs,
        )
    raise ValueError(f"unknown inference backend: {backend}")
