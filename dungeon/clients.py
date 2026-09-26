"""Model-independent client boundary for dungeon participants."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from .models import MoveAction, MoveSequenceAction


class AgentClientError(RuntimeError):
    """A model client failed for a reason other than a decision timeout."""


class AgentTimeoutError(AgentClientError):
    """A model client missed its configured decision deadline."""


@dataclass(frozen=True)
class AgentDecision:
    action: MoveAction | MoveSequenceAction | None
    raw_text: str
    error: str | None = None
    latency_seconds: float | None = None
    thinking_text: str = ""


@dataclass(frozen=True)
class FinalReceipt:
    response_text: str
    latency_seconds: float | None = None
    thinking_text: str = ""


@dataclass(frozen=True)
class SocialReview:
    reflection: str
    notebook: str | None
    public_message: str | None
    raw_text: str
    latency_seconds: float | None = None
    thinking_text: str = ""
    error: str | None = None


class AgentClient(Protocol):
    def choose_action(self, observation: dict[str, object]) -> AgentDecision: ...

    def receive_final(self, observation: dict[str, object]) -> FinalReceipt: ...
