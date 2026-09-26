"""Seeded, validated floor generation."""

from __future__ import annotations

import hashlib
import random
from collections import deque
from typing import Iterable

from .models import BranchExit, FloorDefinition, Position, SkellieSpawn


TIE_DELTAS = ((0, -1), (1, 0), (0, 1), (-1, 0))


def neighbors(position: Position) -> Iterable[Position]:
    for dx, dy in TIE_DELTAS:
        yield Position(position.x + dx, position.y + dy)


def reachable_positions(floor: FloorDefinition, start: Position | None = None) -> frozenset[Position]:
    start = start or floor.spawn
    if not floor.traversable(start):
        return frozenset()
    seen = {start}
    queue = deque([start])
    while queue:
        current = queue.popleft()
        for candidate in neighbors(current):
            if candidate not in seen and floor.traversable(candidate):
                seen.add(candidate)
                queue.append(candidate)
    return frozenset(seen)


def _stable_seed(*parts: object) -> int:
    encoded = "\x1f".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(encoded).digest()[:16], "big")


class FloorGenerator:
    def __init__(self, map_seed: int | str, max_attempts: int = 10_000) -> None:
        self.map_seed = map_seed
        self.max_attempts = max_attempts

    def generate(self, depth: int) -> FloorDefinition:
        """Generate the canonical all-A address at a depth (compatibility API)."""
        return self.generate_branch(("A",) * depth)

    def generate_branch(self, branch_path: tuple[str, ...]) -> FloorDefinition:
        if any(label not in {"A", "B"} for label in branch_path):
            raise ValueError("branch labels must be A or B")
        depth = len(branch_path)
        if depth < 0:
            raise ValueError("depth must be non-negative")
        size = 7 + depth // 100
        address = "/".join(branch_path) or "root"
        for attempt in range(self.max_attempts):
            rng = random.Random(_stable_seed("floor", self.map_seed, address, attempt))
            positions = [Position(x, y) for y in range(size) for x in range(size)]
            rng.shuffle(positions)
            spawn = positions.pop()
            exits = tuple(
                BranchExit(label, positions.pop()) for label in ("A", "B")
            )

            pillar_count = rng.randint(0, size - 1)
            pillars = frozenset(positions.pop() for _ in range(pillar_count))

            heart = positions.pop() if rng.random() < 0.33 else None
            spawn_probability = min(0.50 + 0.01 * depth, 1.0)
            skellie_count = 1 + depth // 100 if rng.random() < spawn_probability else 0
            eligible = [position for position in positions if position.manhattan(spawn) != 1]
            if len(eligible) < skellie_count:
                continue
            selected = rng.sample(eligible, skellie_count)
            skellies = tuple(
                SkellieSpawn(f"skellie-{index}", position)
                for index, position in enumerate(selected)
            )
            floor = FloorDefinition(
                depth=depth,
                width=size,
                height=size,
                spawn=spawn,
                stairs_down=exits[0].position,
                pillars=pillars,
                heart=heart,
                skellies=skellies,
                branch_exits=exits,
                generation_attempt=attempt,
            )
            reachable = reachable_positions(floor)
            required = {
                *(exit_.position for exit_ in exits),
                *(spawn.position for spawn in skellies),
            }
            if heart is not None:
                required.add(heart)
            if required <= reachable:
                return floor
        raise RuntimeError(f"could not generate valid floor at depth {depth}")
