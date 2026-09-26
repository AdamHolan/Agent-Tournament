"""Authoritative state transitions for dungeon runs."""

from __future__ import annotations

import hashlib
import random
from collections import deque
from dataclasses import replace

from .generation import FloorGenerator, neighbors
from .models import (
    Direction,
    Event,
    MoveAction,
    MoveSequenceAction,
    Position,
    RunState,
    RunStatus,
    SequenceTransition,
    SkellieState,
    Transition,
)


GRIM_MESSAGES = {
    "floor_turn_limit": "Mwah hah hah! You cannot escape my blistering speed of one floor per 99 turns!",
    "malformed_output": "Your continued attempts to undermine reality have summoned my evil power!",
    "model_timeout": "Hello? Is anyone home?",
}


def visible_positions(state: RunState) -> frozenset[Position]:
    floor = state.floor
    player = state.player_position
    return frozenset(
        Position(x, y)
        for y in range(floor.height)
        for x in range(floor.width)
        if player.manhattan(Position(x, y)) <= 2
    )


def _roll(seed: int | str, index: int) -> float:
    digest = hashlib.sha256(f"run\x1f{seed}\x1f{index}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


class DungeonEngine:
    PLAYER_HIT_CHANCE = 0.75
    SKELLIE_HIT_CHANCE = 0.50
    FLOOR_TURN_LIMIT = 99
    MALFORMED_LIMIT = 3
    MAX_MOVE_SEQUENCE = 12

    def __init__(
        self,
        generator: FloorGenerator,
        depth_cap: int = 300,
        run_seeded_skellies: bool = True,
    ) -> None:
        if depth_cap <= 0:
            raise ValueError("depth_cap must be a positive integer")
        self.generator = generator
        self.depth_cap = depth_cap
        self.run_seeded_skellies = run_seeded_skellies

    def _generate_floor(self, branch_path: tuple[str, ...]):
        generate_branch = getattr(self.generator, "generate_branch", None)
        if generate_branch is not None:
            return generate_branch(branch_path)
        return self.generator.generate(len(branch_path))

    def _skellies_for(
        self, floor, run_seed: int | str, branch_path: tuple[str, ...]
    ) -> tuple[SkellieState, ...]:
        # Hand-authored/test generators retain their declared actors. Production
        # floors can instead derive placement from circumstances of this run.
        if not self.run_seeded_skellies or not hasattr(self.generator, "generate_branch"):
            return tuple(
                SkellieState(spawn.actor_id, spawn.position) for spawn in floor.skellies
            )
        address = "/".join(branch_path) or "root"
        digest = hashlib.sha256(
            f"skellies\x1f{run_seed}\x1f{address}".encode("utf-8")
        ).digest()
        rng = random.Random(int.from_bytes(digest[:16], "big"))
        probability = min(0.50 + 0.01 * floor.depth, 1.0)
        count = 1 + floor.depth // 100 if rng.random() < probability else 0
        reserved = {
            floor.spawn,
            *(exit_.position for exit_ in floor.exits),
            *(() if floor.heart is None else (floor.heart,)),
        }
        eligible = [
            Position(x, y)
            for y in range(floor.height)
            for x in range(floor.width)
            if floor.traversable(Position(x, y))
            and Position(x, y) not in reserved
            and Position(x, y).manhattan(floor.spawn) != 1
        ]
        return tuple(
            SkellieState(f"skellie-{index}", position)
            for index, position in enumerate(rng.sample(eligible, count))
        )

    def start_run(self, run_id: str, agent_id: str, run_seed: int | str) -> RunState:
        floor = self._generate_floor(())
        state = RunState(
            run_id=run_id,
            agent_id=agent_id,
            run_seed=run_seed,
            floor=floor,
            player_position=floor.spawn,
            skellies=self._skellies_for(floor, run_seed, ()),
            heart_available=floor.heart is not None,
        )
        state = replace(
            state,
            revealed=visible_positions(state),
            visited=frozenset({state.player_position}),
        )
        return replace(
            state,
            fog_tiles_uncovered=len(state.revealed),
            hearts_discovered=int(
                state.floor.heart is not None and state.floor.heart in state.revealed
            ),
            stairs_discovered_turn=(
                0 if any(exit_.position in state.revealed for exit_ in state.floor.exits) else None
            ),
        )

    def step(self, state: RunState, action: MoveAction) -> Transition:
        if not state.active:
            return Transition(state, (Event("action_rejected", {"reason": "run_not_active"}),))

        events: list[Event] = []
        state = replace(
            state,
            turns_used=state.turns_used + 1,
            floor_turn=state.floor_turn + 1,
            malformed_count=0,
        )
        try:
            direction = Direction(action.direction)
        except (ValueError, TypeError):
            direction = None

        if direction is None:
            events.append(Event("invalid_action", {"reason": "unknown_direction"}))
        else:
            dx, dy = direction.delta
            target = Position(state.player_position.x + dx, state.player_position.y + dy)
            if not state.floor.in_bounds(target):
                events.append(Event("invalid_action", {"reason": "out_of_bounds"}))
            elif target in state.floor.pillars:
                events.append(Event("invalid_action", {"reason": "pillar"}))
            else:
                origin = state.player_position
                destination_was_visited = target in state.visited
                state = replace(
                    state,
                    player_position=target,
                    visited=state.visited | {target},
                )
                events.append(Event("player_moved", {
                    "from": [origin.x, origin.y],
                    "to": [target.x, target.y],
                    "direction": direction.value,
                    "destination_was_visited": destination_was_visited,
                }))

                enemy = next((enemy for enemy in state.skellies if enemy.position == target), None)
                if enemy is not None:
                    state, combat_events = self._ordinary_combat(state, enemy.actor_id)
                    events.extend(combat_events)
                if not state.active:
                    return self._finish_transition(state, events)

                selected_exit = next(
                    (exit_ for exit_ in state.floor.exits if exit_.position == target), None
                )
                if selected_exit is not None:
                    events.append(Event("floor_completed", {
                        "depth": state.floor.depth,
                        "fog_tiles_uncovered": len(state.revealed),
                        "floor_tiles": state.floor.width * state.floor.height,
                        "fog_coverage": len(state.revealed) / (
                            state.floor.width * state.floor.height
                        ),
                        "turns_since_stairs_discovered": (
                            None if state.stairs_discovered_turn is None
                            else state.floor_turn - state.stairs_discovered_turn
                        ),
                        "branch_choice": selected_exit.label,
                        "branch_path_before": list(state.branch_path),
                        "branch_path_after": [*state.branch_path, selected_exit.label],
                    }))
                    state, discovery_events = self._descend(state, selected_exit.label)
                    events.extend(discovery_events)
                    if state.status is RunStatus.DEPTH_CAP_REACHED:
                        events.append(Event("run_ended", {"cause": "depth_cap_reached"}))
                    else:
                        events.append(Event("floor_entered", {"depth": state.floor.depth}))
                    return Transition(state, tuple(events))

                if state.heart_available and target == state.floor.heart:
                    old_health = state.player_health
                    state = replace(state, player_health=state.max_health, heart_available=False)
                    events.append(Event("heart_consumed", {
                        "health_before": old_health,
                        "health_after": state.max_health,
                    }))

        state, enemy_events = self._move_skellies(state)
        events.extend(enemy_events)
        if not state.active:
            return self._finish_transition(state, events)

        state, discovery_events = self._reveal_visible(state)
        events.extend(discovery_events)
        if state.floor_turn >= self.FLOOR_TURN_LIMIT:
            state, grim_events = self._grim_combat(state, "floor_turn_limit")
            events.extend(grim_events)
        return self._finish_transition(state, events)

    def step_sequence(self, state: RunState, action: MoveSequenceAction) -> SequenceTransition:
        """Execute literal primitive moves until completion or a decision event."""
        if not 1 <= len(action.directions) <= self.MAX_MOVE_SEQUENCE:
            raise ValueError(f"move sequence must contain 1..{self.MAX_MOVE_SEQUENCE} directions")
        try:
            directions = tuple(Direction(direction) for direction in action.directions)
        except (ValueError, TypeError) as exc:
            raise ValueError("move sequence contains an unknown direction") from exc

        transitions: list[Transition] = []
        executed: list[Direction] = []
        interrupted_by: str | None = None
        current = state

        for index, direction in enumerate(directions):
            before = current
            before_visible_enemies = {
                enemy.actor_id for enemy in before.skellies
                if enemy.position in visible_positions(before)
            }
            known_exits_before = {
                exit_.label for exit_ in before.floor.exits
                if exit_.position in before.revealed
            }
            heart_was_known = before.floor.heart is None or before.floor.heart in before.revealed

            transition = self.step(current, MoveAction(direction))
            transitions.append(transition)
            executed.append(direction)
            current = transition.state
            event_kinds = {event.kind for event in transition.events}

            if not current.active:
                interrupted_by = "run_ended"
            elif current.floor.depth != before.floor.depth:
                interrupted_by = "floor_transition"
            elif "invalid_action" in event_kinds:
                interrupted_by = "invalid_move"
            elif "combat_started" in event_kinds:
                interrupted_by = "combat"
            elif "heart_consumed" in event_kinds:
                interrupted_by = "heart_consumed"
            else:
                after_visible_enemies = {
                    enemy.actor_id for enemy in current.skellies
                    if enemy.position in visible_positions(current)
                }
                if after_visible_enemies - before_visible_enemies:
                    interrupted_by = "skellie_sighted"
                elif {
                    exit_.label for exit_ in current.floor.exits
                    if exit_.position in current.revealed
                } - known_exits_before:
                    interrupted_by = "stairs_discovered"
                elif (
                    not heart_was_known
                    and current.floor.heart is not None
                    and current.heart_available
                    and current.floor.heart in current.revealed
                ):
                    interrupted_by = "heart_discovered"

            if interrupted_by is not None:
                remaining = directions[index + 1:]
                break
        else:
            remaining = ()

        return SequenceTransition(
            state=current,
            transitions=tuple(transitions),
            executed=tuple(executed),
            remaining=tuple(remaining),
            interrupted_by=interrupted_by,
        )

    def record_malformed(self, state: RunState, detail: str = "malformed_action") -> Transition:
        if not state.active:
            return Transition(state, (Event("malformed_ignored", {"reason": "run_not_active"}),))
        count = state.malformed_count + 1
        state = replace(state, malformed_count=count)
        events: list[Event] = [Event("malformed_output", {"detail": detail, "consecutive": count})]
        if count >= self.MALFORMED_LIMIT:
            state, grim_events = self._grim_combat(state, "malformed_output")
            events.extend(grim_events)
        return self._finish_transition(state, events)

    def record_timeout(self, state: RunState) -> Transition:
        if not state.active:
            return Transition(state, (Event("timeout_ignored", {"reason": "run_not_active"}),))
        state, events = self._grim_combat(state, "model_timeout")
        return self._finish_transition(state, list(events))

    def terminate(self, state: RunState, cause: str = "operator_limit") -> Transition:
        if not state.active:
            return Transition(state, (Event("termination_ignored", {"reason": "run_not_active"}),))
        state = replace(
            state,
            status=RunStatus.ADMINISTRATIVELY_TERMINATED,
            cause=cause,
        )
        return Transition(state, (
            Event("run_ended", {"cause": cause, "administrative": True}),
        ))

    def _descend(
        self, state: RunState, branch_choice: str
    ) -> tuple[RunState, tuple[Event, ...]]:
        completed = state.floors_completed + 1
        if completed >= self.depth_cap:
            return replace(
                state,
                floors_completed=completed,
                status=RunStatus.DEPTH_CAP_REACHED,
                cause="depth_cap_reached",
            ), ()
        branch_path = (*state.branch_path, branch_choice)
        floor = self._generate_floor(branch_path)
        next_state = replace(
            state,
            floor=floor,
            player_position=floor.spawn,
            branch_path=branch_path,
            skellies=self._skellies_for(floor, state.run_seed, branch_path),
            heart_available=floor.heart is not None,
            revealed=frozenset(),
            visited=frozenset({floor.spawn}),
            floor_turn=0,
            floors_completed=completed,
            stairs_discovered_turn=None,
        )
        return self._reveal_visible(next_state)

    def _reveal_visible(self, state: RunState) -> tuple[RunState, tuple[Event, ...]]:
        newly_revealed = visible_positions(state) - state.revealed
        if not newly_revealed:
            return state, ()
        heart_discovered = (
            state.floor.heart is not None and state.floor.heart in newly_revealed
        )
        stairs_discovered = any(
            exit_.position in newly_revealed for exit_ in state.floor.exits
        )
        state = replace(
            state,
            revealed=state.revealed | newly_revealed,
            fog_tiles_uncovered=state.fog_tiles_uncovered + len(newly_revealed),
            hearts_discovered=state.hearts_discovered + int(heart_discovered),
            stairs_discovered_turn=(
                state.floor_turn if stairs_discovered else state.stairs_discovered_turn
            ),
        )
        events: list[Event] = [Event("fog_uncovered", {
            "tiles": len(newly_revealed),
            "total": state.fog_tiles_uncovered,
        })]
        if heart_discovered:
            events.append(Event("heart_discovered", {
                "position": [state.floor.heart.x, state.floor.heart.y],
                "total": state.hearts_discovered,
            }))
        return state, tuple(events)

    def _ordinary_combat(self, state: RunState, actor_id: str) -> tuple[RunState, tuple[Event, ...]]:
        enemy = next(enemy for enemy in state.skellies if enemy.actor_id == actor_id)
        health = enemy.health
        player_health = state.player_health
        random_index = state.random_index
        events: list[Event] = [Event("combat_started", {"enemy": actor_id})]

        while health > 0 and player_health > 0:
            value = _roll(state.run_seed, random_index)
            random_index += 1
            hit = value < self.PLAYER_HIT_CHANCE
            if hit:
                health -= 1
            events.append(Event("attack", {
                "attacker": "player", "target": actor_id, "roll": value,
                "hit": hit, "damage": 1 if hit else 0, "target_health": max(health, 0),
            }))
            if health <= 0:
                break

            value = _roll(state.run_seed, random_index)
            random_index += 1
            hit = value < self.SKELLIE_HIT_CHANCE
            if hit:
                player_health -= 1
            events.append(Event("attack", {
                "attacker": actor_id, "target": "player", "roll": value,
                "hit": hit, "damage": 1 if hit else 0, "target_health": max(player_health, 0),
            }))

        if health <= 0:
            skellies = tuple(item for item in state.skellies if item.actor_id != actor_id)
            state = replace(
                state,
                skellies=skellies,
                player_health=player_health,
                random_index=random_index,
                skellies_defeated=state.skellies_defeated + 1,
            )
            events.append(Event("combat_won", {"enemy": actor_id, "player_health": player_health}))
        else:
            updated = tuple(
                replace(item, health=health) if item.actor_id == actor_id else item
                for item in state.skellies
            )
            state = replace(
                state,
                skellies=updated,
                player_health=max(player_health, 0),
                random_index=random_index,
                status=RunStatus.DEAD,
                cause="killed_by_skellie",
            )
            events.append(Event("player_died", {"cause": "killed_by_skellie", "enemy": actor_id}))
        return state, tuple(events)

    def _move_skellies(self, state: RunState) -> tuple[RunState, tuple[Event, ...]]:
        events: list[Event] = []
        skellies = list(sorted(state.skellies, key=lambda item: item.actor_id))
        index = 0
        while index < len(skellies) and state.active:
            enemy = skellies[index]
            if not enemy.aggro and enemy.position.manhattan(state.player_position) <= 2:
                enemy = replace(enemy, aggro=True)
                skellies[index] = enemy
                events.append(Event("skellie_aggroed", {"enemy": enemy.actor_id}))
            if enemy.aggro:
                occupied = {item.position for offset, item in enumerate(skellies) if offset != index}
                step = self._shortest_step(state, enemy.position, state.player_position, occupied)
                if step == state.player_position:
                    state = replace(state, skellies=tuple(skellies))
                    state, combat_events = self._ordinary_combat(state, enemy.actor_id)
                    events.extend(combat_events)
                    skellies = list(sorted(state.skellies, key=lambda item: item.actor_id))
                    if state.active:
                        index = next(
                            (offset for offset, item in enumerate(skellies) if item.actor_id > enemy.actor_id),
                            len(skellies),
                        )
                    continue
                if step is not None and step != enemy.position:
                    origin = enemy.position
                    enemy = replace(enemy, position=step)
                    skellies[index] = enemy
                    events.append(Event("skellie_moved", {
                        "enemy": enemy.actor_id,
                        "from": [origin.x, origin.y],
                        "to": [step.x, step.y],
                    }))
            index += 1
        if state.active:
            state = replace(state, skellies=tuple(skellies))
        return state, tuple(events)

    @staticmethod
    def _shortest_step(
        state: RunState,
        start: Position,
        target: Position,
        occupied: set[Position],
    ) -> Position | None:
        queue = deque([start])
        previous: dict[Position, Position | None] = {start: None}
        while queue:
            current = queue.popleft()
            if current == target:
                break
            for candidate in neighbors(current):
                if candidate in previous:
                    continue
                if candidate != target and (
                    not state.floor.traversable(candidate)
                    or candidate == state.floor.stairs_down
                    or candidate in occupied
                ):
                    continue
                previous[candidate] = current
                queue.append(candidate)
        if target not in previous:
            return None
        cursor = target
        while previous[cursor] != start:
            parent = previous[cursor]
            if parent is None:
                return start
            cursor = parent
        return cursor

    def _grim_combat(self, state: RunState, reason: str) -> tuple[RunState, tuple[Event, ...]]:
        message = GRIM_MESSAGES[reason]
        events = [
            Event("grim_skellie_summoned", {"reason": reason, "message": message}),
            Event("combat_started", {"enemy": "grim-skellie", "initiative": "enemy"}),
        ]
        # Grim acts first with a guaranteed hit in ruleset 0.1.
        health = max(state.player_health - 99, 0)
        events.append(Event("attack", {
            "attacker": "grim-skellie", "target": "player", "roll": 0.0,
            "hit": True, "damage": 99, "target_health": health,
        }))
        state = replace(state, player_health=health, status=RunStatus.DEAD, cause=f"grim_skellie:{reason}")
        events.append(Event("player_died", {"cause": state.cause, "enemy": "grim-skellie"}))
        return state, tuple(events)

    @staticmethod
    def _finish_transition(state: RunState, events: list[Event]) -> Transition:
        if state.status is RunStatus.DEAD and not any(event.kind == "run_ended" for event in events):
            events.append(Event("run_ended", {"cause": state.cause}))
        return Transition(state, tuple(events))
