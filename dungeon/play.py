"""Run one Ollama-controlled dungeon attempt from the terminal."""

from __future__ import annotations

import argparse
import os

from .clients import AgentDecision, FinalReceipt
from .engine import DungeonEngine
from .generation import FloorGenerator
from .metrics import build_behavioral_metrics, write_metrics_file
from .models import MoveAction, MoveSequenceAction, SequenceTransition, Transition
from .ollama_agent import OllamaDungeonAgent
from .runner import run_agent


def _print_observer(kind: str, value: object, show_model_output: bool = False) -> None:
    if kind == "observation":
        observation = value
        print(
            f"\nDepth {observation['depth']} | turn {observation['turn_on_floor']} "
            f"| HP {observation['player']['health']} | score {observation['score']}"
        )
        for row in observation["floor"]["map"]:
            print(row)
    elif kind == "decision" and isinstance(value, AgentDecision):
        if show_model_output and value.thinking_text:
            print(f"\n[model thinking]\n{value.thinking_text}\n[end thinking]")
        if show_model_output:
            print(f"[raw model content]\n{value.raw_text}\n[end model content]")
        if value.action is None:
            print(f"Agent response rejected: {value.error}\nRaw: {value.raw_text}")
        elif isinstance(value.action, MoveAction):
            print(f"Agent moves {value.action.direction.value} ({value.latency_seconds:.2f}s)")
        elif isinstance(value.action, MoveSequenceAction):
            directions = ", ".join(str(item) for item in value.action.directions)
            print(f"Agent plans sequence [{directions}] ({value.latency_seconds:.2f}s)")
    elif kind == "sequence_transition" and isinstance(value, SequenceTransition):
        executed = ", ".join(item.value for item in value.executed) or "none"
        remaining = ", ".join(item.value for item in value.remaining) or "none"
        reason = value.interrupted_by or "completed"
        print(f"  sequence resolved: {reason}; executed [{executed}]; remaining [{remaining}]")
    elif kind == "transition" and isinstance(value, Transition):
        notable = {
            "invalid_action", "combat_started", "combat_won", "heart_consumed",
            "floor_completed", "grim_skellie_summoned", "player_died", "run_ended",
        }
        for event in value.events:
            if event.kind in notable:
                print(f"  {event.kind}: {dict(event.data)}")
    elif kind == "client_error":
        print(f"Model client error: {value}")
    elif kind == "final_receipt" and isinstance(value, FinalReceipt):
        if show_model_output and value.thinking_text:
            print(f"\n[final model thinking]\n{value.thinking_text}\n[end thinking]")
        print(f"\nAgent's final reflection ({value.latency_seconds:.2f}s):\n{value.response_text}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Let an Ollama model play one private dungeon run")
    parser.add_argument("--model", default=os.getenv("AGENT_MODEL"))
    parser.add_argument("--base-url", default=os.getenv("AGENT_BASE_URL", "http://localhost:11434"))
    parser.add_argument("--map-seed", default="land-of-wonders")
    parser.add_argument("--run-seed", default="first-model-run")
    parser.add_argument("--depth-cap", type=int, default=3)
    parser.add_argument("--max-decisions", type=int, default=300)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--history-turns", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument(
        "--run-seeded-skellies",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="vary skellie presence and placement by run seed (default: enabled)",
    )
    parser.add_argument(
        "--metrics-dir",
        default="run_metrics",
        help="directory for unique per-run behavioral metrics JSON files",
    )
    parser.add_argument(
        "--show-model-output",
        action="store_true",
        help="print raw action JSON and any separate thinking returned by Ollama",
    )
    parser.add_argument(
        "--enable-thinking",
        action="store_true",
        help="request thinking from a thinking-capable model",
    )
    args = parser.parse_args()
    if not args.model:
        parser.error("provide --model or set AGENT_MODEL")

    engine = DungeonEngine(
        FloorGenerator(args.map_seed),
        depth_cap=args.depth_cap,
        run_seeded_skellies=args.run_seeded_skellies,
    )
    client = OllamaDungeonAgent(
        model=args.model,
        base_url=args.base_url,
        timeout_seconds=args.timeout,
        temperature=args.temperature,
        history_turns=args.history_turns,
        enable_thinking=args.enable_thinking,
    )
    outcome = run_agent(
        engine,
        client,
        run_id="local-preview",
        agent_id=args.model,
        run_seed=args.run_seed,
        max_decisions=args.max_decisions,
        observer=lambda kind, value: _print_observer(kind, value, args.show_model_output),
    )
    result = outcome.final_observation["run_result"]
    print(
        f"\nFinal result: {result['status']} ({result['cause']}); "
        f"score {result['score']}, floors {result['floors_completed']}, "
        f"turns {result['turns_used']}"
    )
    if outcome.final_delivery_error:
        print(f"Final state could not be delivered to the model: {outcome.final_delivery_error}")
    metrics = build_behavioral_metrics(
        outcome,
        model=args.model,
        map_seed=args.map_seed,
        run_seed=args.run_seed,
    )
    metrics_path = write_metrics_file(metrics, args.metrics_dir)
    print(f"Behavioral metrics saved to {metrics_path}")


if __name__ == "__main__":
    main()
