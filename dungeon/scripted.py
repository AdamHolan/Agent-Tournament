"""Simple non-model players for deterministic engine tests and smoke runs."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Protocol

from .engine import DungeonEngine
from .generation import neighbors
from .models import Direction, MoveAction, Position, RunState, Transition


class ScriptedPlayer(Protocol):
    def choose_action(self, state: RunState) -> MoveAction: ...


def _direction(origin: Position, target: Position) -> Direction:
    delta = (target.x - origin.x, target.y - origin.y)
    return {
        (0, -1): Direction.NORTH,
        (1, 0): Direction.EAST,
        (0, 1): Direction.SOUTH,
        (-1, 0): Direction.WEST,
    }[delta]


@dataclass
class StairPathPlayer:
    """A test oracle that follows a shortest terrain path to downstairs."""

    def choose_action(self, state: RunState) -> MoveAction:
        start = state.player_position
        target = state.floor.stairs_down
        queue = deque([start])
        previous: dict[Position, Position | None] = {start: None}
        while queue:
            current = queue.popleft()
            if current == target:
                break
            for candidate in neighbors(current):
                if candidate not in previous and state.floor.traversable(candidate):
                    previous[candidate] = current
                    queue.append(candidate)
        if target not in previous:
            raise RuntimeError("scripted player found unreachable downstairs")
        cursor = target
        while previous[cursor] != start:
            parent = previous[cursor]
            if parent is None:
                raise RuntimeError("player is already on downstairs")
            cursor = parent
        return MoveAction(_direction(start, cursor))


@dataclass
class SequencePlayer:
    directions: list[Direction]
    index: int = 0

    def choose_action(self, state: RunState) -> MoveAction:
        if not self.directions:
            raise RuntimeError("sequence has no actions")
        direction = self.directions[min(self.index, len(self.directions) - 1)]
        self.index += 1
        return MoveAction(direction)


def play_scripted(
    engine: DungeonEngine,
    state: RunState,
    player: ScriptedPlayer,
    max_decisions: int = 10_000,
) -> tuple[RunState, tuple[Transition, ...]]:
    transitions: list[Transition] = []
    for _ in range(max_decisions):
        if not state.active:
            return state, tuple(transitions)
        transition = engine.step(state, player.choose_action(state))
        transitions.append(transition)
        state = transition.state
    raise RuntimeError(f"scripted run exceeded {max_decisions} decisions")
