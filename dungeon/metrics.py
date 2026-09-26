"""Behavioral measurements and collision-resistant per-run JSON artifacts."""

from __future__ import annotations

import json
import re
import statistics
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .models import MoveAction, MoveSequenceAction
from .runner import RunOutcome


def _movement_event(resolved: Any) -> Any | None:
    return next(
        (event for event in resolved.transition.events if event.kind == "player_moved"),
        None,
    )


def _fog_before_floor_exit(resolved: Any) -> int:
    tiles = 0
    for event in resolved.transition.events:
        if event.kind == "floor_completed":
            break
        if event.kind == "fog_uncovered":
            tiles += int(event.data["tiles"])
    return tiles


def _trajectory_metrics(outcome: RunOutcome) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    floors: dict[int, dict[str, Any]] = {}
    trace: list[dict[str, Any]] = []

    def floor_row(state: Any) -> dict[str, Any]:
        depth = state.floor.depth
        if depth not in floors:
            floors[depth] = {
                "depth": depth,
                "branch_path": list(state.branch_path),
                "branch_address": "/".join(state.branch_path) or "root",
                "floor_tiles": state.floor.width * state.floor.height,
                "turns": 0,
                "moves": 0,
                "invalid_actions": 0,
                "turns_with_zero_new_fog": 0,
                "positions": [[state.player_position.x, state.player_position.y]],
                "completed": False,
                "fog_tiles_uncovered": len(state.revealed),
                "stairs_discovered": state.stairs_discovered_turn is not None,
                "stairs_discovered_turn": state.stairs_discovered_turn,
                "turns_since_stairs_discovered": None,
            }
        return floors[depth]

    for resolved in outcome.resolved_turns:
        before = resolved.before
        after = resolved.transition.state
        row = floor_row(before)
        consumed_turn = after.turns_used > before.turns_used
        movement = _movement_event(resolved)
        new_fog = _fog_before_floor_exit(resolved)
        kinds = [event.kind for event in resolved.transition.events]
        if consumed_turn:
            row["turns"] += 1
            row["turns_with_zero_new_fog"] += int(new_fog == 0)
        row["invalid_actions"] += int("invalid_action" in kinds)
        if movement is not None:
            row["moves"] += 1
            row["positions"].append(list(movement.data["to"]))
        if before.floor.depth == after.floor.depth:
            row["fog_tiles_uncovered"] = len(after.revealed)
            row["stairs_discovered"] = after.stairs_discovered_turn is not None
            row["stairs_discovered_turn"] = after.stairs_discovered_turn

        completed = next(
            (event for event in resolved.transition.events if event.kind == "floor_completed"),
            None,
        )
        if completed is not None:
            row["completed"] = True
            row["fog_tiles_uncovered"] = completed.data["fog_tiles_uncovered"]
            row["turns_since_stairs_discovered"] = completed.data[
                "turns_since_stairs_discovered"
            ]

        if consumed_turn:
            trace.append({
                "decision_index": resolved.decision_index,
                "depth": before.floor.depth,
                "branch_path": list(before.branch_path),
                "turn_before": before.floor_turn,
                "position_before": [before.player_position.x, before.player_position.y],
                "direction": movement.data["direction"] if movement else None,
                "position_after": list(movement.data["to"]) if movement else [
                    before.player_position.x, before.player_position.y
                ],
                "destination_was_visited": (
                    bool(movement.data["destination_was_visited"]) if movement else None
                ),
                "new_fog_tiles": new_fog,
                "score_before": before.score,
                "score_after": after.score,
                "score_delta": after.score - before.score,
                "stairs_known_before": before.stairs_discovered_turn is not None,
                "events": kinds,
            })

    floor_row(outcome.state)
    per_floor: list[dict[str, Any]] = []
    for depth in sorted(floors):
        row = floors[depth]
        positions = [tuple(position) for position in row.pop("positions")]
        immediate_backtracks = sum(
            positions[index] == positions[index - 2]
            for index in range(2, len(positions))
        )
        loop_closures = {
            str(period): sum(
                positions[index] == positions[index - period]
                for index in range(period, len(positions))
            )
            for period in (2, 3, 4)
        }
        row.update({
            "fog_coverage": row["fog_tiles_uncovered"] / row["floor_tiles"],
            "unique_positions_visited": len(set(positions)),
            "revisited_position_arrivals": len(positions) - len(set(positions)),
            "immediate_backtracks": immediate_backtracks,
            "immediate_backtrack_rate": (
                immediate_backtracks / row["moves"] if row["moves"] else 0.0
            ),
            "loop_closures_by_period": loop_closures,
        })
        if not row["completed"] and row["stairs_discovered_turn"] is not None:
            row["turns_since_stairs_discovered"] = max(
                0, outcome.state.floor_turn - row["stairs_discovered_turn"]
            ) if outcome.state.floor.depth == depth else None
        per_floor.append(row)
    return per_floor, trace


def build_behavioral_metrics(
    outcome: RunOutcome,
    *,
    model: str,
    map_seed: int | str,
    run_seed: int | str,
) -> dict[str, Any]:
    events = outcome.events
    moves = [event for event in events if event.kind == "player_moved"]
    retraced = sum(bool(event.data.get("destination_was_visited")) for event in moves)
    invalid = sum(event.kind == "invalid_action" for event in events)
    floor_events = [event for event in events if event.kind == "floor_completed"]
    branch_choices = [str(event.data["branch_choice"]) for event in floor_events]
    branch_addresses = ["root"] + [
        "/".join(event.data["branch_path_after"]) for event in floor_events
    ]
    single_decisions = sum(isinstance(item.action, MoveAction) for item in outcome.decisions)
    sequence_decisions = sum(
        isinstance(item.action, MoveSequenceAction) for item in outcome.decisions
    )
    rejected_decisions = sum(item.action is None for item in outcome.decisions)
    latencies = [
        item.latency_seconds for item in outcome.decisions
        if item.latency_seconds is not None
    ]
    sequence_events = [
        event for event in events if event.kind == "move_sequence_resolved"
    ]
    requests = len(outcome.observations)
    turns = outcome.state.turns_used
    per_floor, action_trace = _trajectory_metrics(outcome)

    return {
        "schema_version": "0.1",
        "recorded_at_utc": datetime.now(UTC).isoformat(),
        "identity": {
            "run_id": outcome.state.run_id,
            "agent_id": outcome.state.agent_id,
            "model": model,
            "map_seed": str(map_seed),
            "run_seed": str(run_seed),
        },
        "result": {
            "status": outcome.state.status.value,
            "cause": outcome.state.cause,
            "score": outcome.state.score,
            "floors_completed": outcome.state.floors_completed,
            "turns_used": turns,
            "health_remaining": outcome.state.player_health,
            "final_branch_path": list(outcome.state.branch_path),
            "final_branch_address": "/".join(outcome.state.branch_path) or "root",
            "score_breakdown": outcome.final_observation["score_breakdown"],
        },
        "final_reflection": {
            "received": outcome.final_receipt is not None,
            "response_text": (
                outcome.final_receipt.response_text if outcome.final_receipt else None
            ),
            "thinking_text": (
                outcome.final_receipt.thinking_text if outcome.final_receipt else None
            ),
            "latency_seconds": (
                outcome.final_receipt.latency_seconds if outcome.final_receipt else None
            ),
            "delivery_error": outcome.final_delivery_error,
        },
        "behavior": {
            "model_requests": requests,
            "model_responses": len(outcome.decisions),
            "rejected_responses": rejected_decisions,
            "single_move_decisions": single_decisions,
            "sequence_decisions": sequence_decisions,
            "sequence_adoption_rate": sequence_decisions / len(outcome.decisions)
            if outcome.decisions else 0.0,
            "sequence_moves_submitted": sum(
                len(event.data["submitted"]) for event in sequence_events
            ),
            "sequence_moves_executed": sum(
                len(event.data["executed"]) for event in sequence_events
            ),
            "moves_executed": len(moves),
            "moves_to_visited_tiles": retraced,
            "retrace_rate": retraced / len(moves) if moves else 0.0,
            "invalid_actions": invalid,
            "invalid_action_rate": invalid / turns if turns else 0.0,
            "score_per_model_request": outcome.state.score / requests if requests else 0.0,
            "score_per_turn": outcome.state.score / turns if turns else None,
            "action_latency_seconds": {
                "total": sum(latencies),
                "mean": statistics.fmean(latencies) if latencies else None,
                "median": statistics.median(latencies) if latencies else None,
                "maximum": max(latencies) if latencies else None,
            },
            "completed_floor_exploration": [
                {
                    "depth": event.data["depth"],
                    "branch_choice": event.data["branch_choice"],
                    "branch_path_after": event.data["branch_path_after"],
                    "fog_tiles_uncovered": event.data["fog_tiles_uncovered"],
                    "floor_tiles": event.data["floor_tiles"],
                    "fog_coverage": event.data["fog_coverage"],
                    "turns_since_stairs_discovered": event.data[
                        "turns_since_stairs_discovered"
                    ],
                }
                for event in floor_events
            ],
            "branch_choices": branch_choices,
            "branch_choice_counts": {
                label: branch_choices.count(label) for label in ("A", "B")
            },
            "branch_addresses_visited": branch_addresses,
            "unique_branch_addresses_visited": len(set(branch_addresses)),
            "per_floor": per_floor,
            "turns_with_zero_new_fog": sum(
                row["turns_with_zero_new_fog"] for row in per_floor
            ),
            "immediate_backtracks": sum(row["immediate_backtracks"] for row in per_floor),
            "loop_closures_by_period": {
                period: sum(
                    row["loop_closures_by_period"][period] for row in per_floor
                )
                for period in ("2", "3", "4")
            },
        },
        "action_trace": action_trace,
    }


def write_metrics_file(metrics: dict[str, Any], directory: str | Path) -> Path:
    destination = Path(directory)
    destination.mkdir(parents=True, exist_ok=True)
    identity = metrics.get("identity", {})
    model = _safe_name(str(identity.get("model", "model")))
    run_id = _safe_name(str(identity.get("run_id", "run")))
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S.%fZ")
    path = destination / f"{timestamp}_{model}_{run_id}_{uuid.uuid4().hex}.json"
    with path.open("x", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    return path


def _safe_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-._")
    return (cleaned or "unknown")[:80]
