"""Fog-safe structured observations and compact ASCII maps."""

from __future__ import annotations

from typing import Iterable

from .engine import visible_positions
from .models import Direction, Event, Position, RunState


def _movement_options(state: RunState) -> dict[str, dict[str, object]]:
    options: dict[str, dict[str, object]] = {}
    enemy_positions = {enemy.position for enemy in state.skellies}
    for direction in Direction:
        dx, dy = direction.delta
        target = Position(state.player_position.x + dx, state.player_position.y + dy)
        if not state.floor.in_bounds(target):
            terrain = "out_of_bounds"
            passable = False
        elif target in state.floor.pillars:
            terrain = "pillar"
            passable = False
        elif target in enemy_positions:
            terrain = "skellie"
            passable = True
        elif exit_ := next(
            (item for item in state.floor.exits if item.position == target), None
        ):
            terrain = f"stairs_{exit_.label}"
            passable = True
        elif state.heart_available and target == state.floor.heart:
            terrain = "heart"
            passable = True
        else:
            terrain = "floor"
            passable = True
        options[direction.value] = {
            "position": [target.x, target.y],
            "terrain": terrain,
            "passable": passable,
            "visited": target in state.visited,
        }
    return options


def render_map(state: RunState) -> tuple[str, ...]:
    visible = visible_positions(state)
    rows: list[str] = []
    for y in range(state.floor.height):
        row: list[str] = []
        for x in range(state.floor.width):
            position = Position(x, y)
            if position not in state.revealed:
                glyph = "?"
            elif position in state.floor.pillars:
                glyph = "#"
            elif exit_ := next(
                (item for item in state.floor.exits if item.position == position), None
            ):
                glyph = exit_.label
            elif state.heart_available and position == state.floor.heart:
                glyph = "H"
            else:
                glyph = "."
            if position in visible:
                enemy = next((item for item in state.skellies if item.position == position), None)
                if enemy is not None:
                    glyph = "s"
            if position == state.player_position:
                glyph = "@"
            row.append(glyph)
        rows.append("".join(row))
    return tuple(rows)


def build_observation(
    state: RunState,
    recent_events: Iterable[Event] = (),
    recent_move_trace: Iterable[str] = (),
) -> dict[str, object]:
    visible = sorted(visible_positions(state))
    move_trace = tuple(recent_move_trace)[-16:]
    return {
        "protocol_version": "0.1",
        "run_id": state.run_id,
        "depth": state.floor.depth,
        "branch_path": list(state.branch_path),
        "branch_address": "/".join(state.branch_path) or "root",
        "turn_on_floor": state.floor_turn,
        "turns_used": state.turns_used,
        "score": state.score,
        "score_breakdown": {
            "floors_completed": state.floors_completed,
            "hearts_discovered": state.hearts_discovered,
            "skellies_defeated": state.skellies_defeated,
            "fog_tiles_uncovered": state.fog_tiles_uncovered,
            "turns_used": state.turns_used,
            "points_per_event": {
                "completed_floor": 100,
                "heart_discovered": 25,
                "skellie_defeated": 50,
                "fog_tile_uncovered_per_tile": 1,
                "turn": -1,
            },
        },
        "status": state.status.value,
        "cause": state.cause,
        "player": {
            "position": [state.player_position.x, state.player_position.y],
            "health": state.player_health,
            "max_health": state.max_health,
        },
        "floor": {
            "width": state.floor.width,
            "height": state.floor.height,
            "map": list(render_map(state)),
            "exit_labels": [exit_.label for exit_ in state.floor.exits],
        },
        "currently_visible": [[position.x, position.y] for position in visible],
        "visited": [[position.x, position.y] for position in sorted(state.visited)],
        "recent_move_trace": {
            "directions": "".join(move_trace),
            "order": "oldest_to_newest",
            "scope": "current_floor",
            "maximum_moves": 16,
            "contents": "successful executed primitive moves only",
        },
        "movement_options": _movement_options(state),
        "recent_events": [event.as_dict() for event in recent_events],
        "legal_actions": [] if not state.active else [
            {
                "action": "move",
                "direction": ["north", "east", "south", "west"],
            },
            {
                "action": "move_sequence",
                "directions": {
                    "allowed": ["north", "east", "south", "west"],
                    "minimum": 1,
                    "maximum": 12,
                },
                "execution": "literal primitive turns; stops before executing the remainder after a decision event",
                "interrupts_on": [
                    "skellie_sighted", "stairs_discovered", "heart_discovered", "heart_consumed",
                    "combat", "invalid_move", "floor_transition", "run_ended",
                ],
            },
        ],
    }


def build_final_observation(
    state: RunState,
    run_events: Iterable[Event],
    recent_event_limit: int = 25,
    recent_move_trace: Iterable[str] = (),
) -> dict[str, object]:
    """Build the bounded terminal state delivered to an agent after its run."""
    if state.active:
        raise ValueError("final observations require an inactive run")
    if recent_event_limit <= 0:
        raise ValueError("recent_event_limit must be positive")

    events = tuple(run_events)
    counts = {
        "moves": sum(event.kind == "player_moved" for event in events),
        "invalid_actions": sum(event.kind == "invalid_action" for event in events),
        "combats_started": sum(event.kind == "combat_started" for event in events),
        "ordinary_skellies_defeated": sum(
            event.kind == "combat_won" and event.data.get("enemy") != "grim-skellie"
            for event in events
        ),
        "hearts_consumed": sum(event.kind == "heart_consumed" for event in events),
        "floors_completed": state.floors_completed,
    }
    dialogue = [
        {
            "speaker": "grim_skellie",
            "reason": event.data["reason"],
            "text": event.data["message"],
        }
        for event in events
        if event.kind == "grim_skellie_summoned"
    ]
    defeated = [
        {"kind": "skellie", "actor_id": event.data["enemy"]}
        for event in events
        if event.kind == "combat_won" and event.data.get("enemy") != "grim-skellie"
    ]

    observation = build_observation(
        state, events[-recent_event_limit:], recent_move_trace
    )
    observation["observation_type"] = "run_final"
    observation["run_result"] = {
        "status": state.status.value,
        "cause": state.cause,
        "score": state.score,
        "floors_completed": state.floors_completed,
        "turns_used": state.turns_used,
        "final_depth": state.floor.depth,
        "health_remaining": state.player_health,
        "action_summary": counts,
        "score_breakdown": observation["score_breakdown"],
        "defeated_enemies": defeated,
        "dialogue": dialogue,
        "events_in_audit_log": len(events),
        "recent_events_included": min(len(events), recent_event_limit),
    }
    return observation
