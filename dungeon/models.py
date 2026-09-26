"""Small serializable values used by the dungeon engine."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from types import MappingProxyType
from typing import Any, Mapping


@dataclass(frozen=True, order=True)
class Position:
    x: int
    y: int

    def manhattan(self, other: "Position") -> int:
        return abs(self.x - other.x) + abs(self.y - other.y)


class Direction(StrEnum):
    NORTH = "north"
    EAST = "east"
    SOUTH = "south"
    WEST = "west"

    @property
    def delta(self) -> tuple[int, int]:
        return {
            Direction.NORTH: (0, -1),
            Direction.EAST: (1, 0),
            Direction.SOUTH: (0, 1),
            Direction.WEST: (-1, 0),
        }[self]


@dataclass(frozen=True)
class MoveAction:
    direction: Direction | str


@dataclass(frozen=True)
class MoveSequenceAction:
    directions: tuple[Direction | str, ...]


class RunStatus(StrEnum):
    ACTIVE = "active"
    DEAD = "dead"
    DEPTH_CAP_REACHED = "depth_cap_reached"
    ADMINISTRATIVELY_TERMINATED = "administratively_terminated"
    ENGINE_ERROR = "engine_error"


@dataclass(frozen=True)
class SkellieSpawn:
    actor_id: str
    position: Position


@dataclass(frozen=True)
class SkellieState:
    actor_id: str
    position: Position
    health: int = 2
    aggro: bool = False


@dataclass(frozen=True)
class BranchExit:
    label: str
    position: Position


@dataclass(frozen=True)
class FloorDefinition:
    depth: int
    width: int
    height: int
    spawn: Position
    stairs_down: Position
    pillars: frozenset[Position] = frozenset()
    heart: Position | None = None
    skellies: tuple[SkellieSpawn, ...] = ()
    branch_exits: tuple[BranchExit, ...] = ()
    generation_attempt: int = 0

    @property
    def exits(self) -> tuple[BranchExit, ...]:
        """Return explicit branch exits, or the legacy single A exit."""
        return self.branch_exits or (BranchExit("A", self.stairs_down),)

    def in_bounds(self, position: Position) -> bool:
        return 0 <= position.x < self.width and 0 <= position.y < self.height

    def traversable(self, position: Position) -> bool:
        return self.in_bounds(position) and position not in self.pillars


@dataclass(frozen=True)
class Event:
    kind: str
    data: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "data", MappingProxyType(dict(self.data)))

    def as_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "data": dict(self.data)}


@dataclass(frozen=True)
class RunState:
    run_id: str
    agent_id: str
    run_seed: int | str
    floor: FloorDefinition
    player_position: Position
    branch_path: tuple[str, ...] = ()
    player_health: int = 3
    max_health: int = 3
    skellies: tuple[SkellieState, ...] = ()
    heart_available: bool = False
    revealed: frozenset[Position] = frozenset()
    visited: frozenset[Position] = frozenset()
    turns_used: int = 0
    floor_turn: int = 0
    floors_completed: int = 0
    fog_tiles_uncovered: int = 0
    hearts_discovered: int = 0
    skellies_defeated: int = 0
    stairs_discovered_turn: int | None = None
    random_index: int = 0
    malformed_count: int = 0
    status: RunStatus = RunStatus.ACTIVE
    cause: str | None = None

    @property
    def score(self) -> int:
        return (
            100 * self.floors_completed
            + 25 * self.hearts_discovered
            + 50 * self.skellies_defeated
            + self.fog_tiles_uncovered
            - self.turns_used
        )

    @property
    def active(self) -> bool:
        return self.status is RunStatus.ACTIVE


@dataclass(frozen=True)
class Transition:
    state: RunState
    events: tuple[Event, ...]


@dataclass(frozen=True)
class SequenceTransition:
    state: RunState
    transitions: tuple[Transition, ...]
    executed: tuple[Direction, ...]
    remaining: tuple[Direction, ...]
    interrupted_by: str | None
