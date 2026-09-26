"""Resumable matched experiment for private memory and public chat."""

from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import math
import os
import statistics
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .clients import AgentClientError, SocialReview
from .client_factory import BACKENDS, create_dungeon_client
from .engine import DungeonEngine
from .generation import FloorGenerator
from .metrics import build_behavioral_metrics
from .ollama_agent import (
    MEMORYLESS_REVIEW_SYSTEM_PROMPT,
    PRIVATE_REVIEW_SYSTEM_PROMPT,
    OllamaDungeonAgent,
    SOCIAL_SYSTEM_PROMPT,
    SYSTEM_PROMPT,
)
from .runner import run_agent


CONDITIONS = ("memoryless", "private_memory", "social")
SOCIAL_REVIEW_RECENT_TURNS = 10


def _compact_final_observation(observation: dict[str, Any] | None) -> dict[str, Any] | None:
    """Remove the event list duplicated by the social review's recent turn trace."""
    if observation is None:
        return None
    compact = dict(observation)
    compact.pop("recent_events", None)
    return compact


def _bounded_unread(
    messages: list[dict[str, Any]], cursor: int, character_limit: int,
    exclude_author: str | None = None,
) -> tuple[list[dict[str, Any]], int]:
    unread = [
        message for message in messages[cursor:]
        if message["author"] != exclude_author
    ]
    selected: list[dict[str, Any]] = []
    used = 0
    for message in reversed(unread):
        cost = len(message["author"]) + len(message["message"]) + 16
        if used + cost > character_limit:
            continue
        selected.append(message)
        used += cost
    selected.reverse()
    return selected, len(unread) - len(selected)


def _normalized_message(value: str) -> str:
    return " ".join(value.casefold().split())


def _similarity_to_prior(
    message: str, prior: list[dict[str, Any]], author: str
) -> dict[str, Any]:
    candidates = [item for item in prior if item["author"] != author]
    if not candidates:
        return {
            "most_similar_prior_message_id": None,
            "similarity_to_prior": None,
            "near_copy_of_prior": False,
            "exact_copy_of_prior": False,
        }
    normalized = _normalized_message(message)
    scored = [
        (
            difflib.SequenceMatcher(
                None, normalized, _normalized_message(item["message"])
            ).ratio(),
            item,
        )
        for item in candidates
    ]
    similarity, closest = max(scored, key=lambda pair: pair[0])
    return {
        "most_similar_prior_message_id": closest["message_id"],
        "similarity_to_prior": similarity,
        "near_copy_of_prior": similarity >= 0.8,
        "exact_copy_of_prior": normalized == _normalized_message(closest["message"]),
    }


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp-" + uuid.uuid4().hex)
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_chat(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    messages: list[dict[str, Any]] = []
    seen: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        message = json.loads(line)
        if message["message_id"] not in seen:
            messages.append(message)
            seen.add(message["message_id"])
    return messages


def _append_chat(path: Path, messages: list[dict[str, Any]]) -> None:
    existing = {item["message_id"] for item in _load_chat(path)}
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for message in messages:
            if message["message_id"] not in existing:
                handle.write(json.dumps(message, ensure_ascii=False) + "\n")
                existing.add(message["message_id"])


def _leaderboard(run_rows: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows = [
        {
            "guest_id": guest_id,
            "score": data["result"]["score"],
            "floors_completed": data["result"]["floors_completed"],
            "turns_used": data["result"]["turns_used"],
            "health_remaining": data["result"]["health_remaining"],
            "status": data["result"]["status"],
            "cause": data["result"]["cause"],
        }
        for guest_id, data in run_rows.items()
    ]
    rows.sort(key=lambda row: (
        -row["score"], -row["floors_completed"], row["turns_used"],
        -row["health_remaining"], row["guest_id"],
    ))
    for rank, row in enumerate(rows, 1):
        row["rank"] = rank
    return rows


def _new_state(conditions: list[str], agents: int) -> dict[str, Any]:
    return {
        "schema_version": "0.1",
        "next_round": 0,
        "agents": {
            condition: {
                f"guest-{index + 1:03d}": {
                    "notebook": "",
                    "chat_cursor": 0,
                    "previous_result": None,
                }
                for index in range(agents)
            }
            for condition in conditions
        },
        "last_leaderboards": {condition: [] for condition in conditions},
    }


def _client(manifest: dict[str, Any], run_context: dict[str, object] | None = None) -> OllamaDungeonAgent:
    return create_dungeon_client(
        backend=manifest.get("backend", "ollama"),
        model=manifest["model"],
        base_url=manifest["base_url"],
        api_key=manifest.get("_runtime_api_key") or os.getenv("AGENT_API_KEY"),
        timeout_seconds=manifest["timeout"],
        temperature=manifest["temperature"],
        history_turns=manifest["history_turns"],
        enable_thinking=manifest["enable_thinking"],
        run_context=run_context,
    )


def _review_as_dict(review: SocialReview) -> dict[str, Any]:
    return {
        "reflection": review.reflection,
        "notebook": review.notebook,
        "public_message": review.public_message,
        "raw_text": review.raw_text,
        "thinking_text": review.thinking_text,
        "latency_seconds": review.latency_seconds,
        "error": review.error,
    }


def _review_instruction(condition: str) -> str:
    if condition == "memoryless":
        return (
            "Reflect on the completed run for the experiment record. This condition "
            "has no persistent notebook and no public room. Return null for notebook "
            "and public_message."
        )
    if condition == "private_memory":
        return (
            "Reflect on the completed run. You may replace your private notebook or "
            "preserve it with null. There is no public room in this condition; return "
            "null for public_message. Notebook text completely replaces the old "
            "notebook and must fit within 1,500 characters."
        )
    return (
        "Reflect on the completed run. You may replace your private notebook or "
        "preserve it with null. You may post one public message or remain silent with "
        "null. These choices are entirely yours. A public message must be complete "
        "within 500 Unicode characters; do not feel obligated to fill the allowance. "
        "Notebook text completely replaces the old notebook and must fit within "
        "1,500 characters."
    )


def _summarize(root: Path, manifest: dict[str, Any], state: dict[str, Any], interrupted: bool) -> dict[str, Any]:
    rows: dict[str, list[dict[str, Any]]] = {condition: [] for condition in manifest["conditions"]}
    review_rows: dict[str, list[dict[str, Any]]] = {
        condition: [] for condition in manifest["conditions"]
    }
    for path in root.glob("rounds/round-*/runs/*.json"):
        data = _load_json(path)
        rows[data["social"]["condition"]].append(data)
    for path in root.glob("rounds/round-*/reviews/*.json"):
        condition = path.name.split("_guest-", 1)[0]
        if condition in review_rows:
            review_rows[condition].append(_load_json(path))
    chat = _load_chat(root / "chat.jsonl")
    comparable_chat = [item for item in chat if item.get("similarity_to_prior") is not None]

    matched: dict[str, dict[str, float | int | None]] = {}

    def add_matched_difference(baseline_name: str, condition: str) -> None:
        baseline = {
            (item["social"]["guest_id"], item["social"]["round"]): item
            for item in rows.get(baseline_name, [])
        }
        pairs = [
            (baseline[(item["social"]["guest_id"], item["social"]["round"])], item)
            for item in rows.get(condition, [])
            if (item["social"]["guest_id"], item["social"]["round"]) in baseline
        ]
        matched[f"{condition}_minus_{baseline_name}"] = {
            "pairs": len(pairs),
            "mean_score_delta": statistics.fmean(
                treatment["result"]["score"] - control["result"]["score"]
                for control, treatment in pairs
            ) if pairs else None,
            "mean_floor_delta": statistics.fmean(
                treatment["result"]["floors_completed"]
                - control["result"]["floors_completed"]
                for control, treatment in pairs
            ) if pairs else None,
            "mean_zero_discovery_turn_delta": statistics.fmean(
                treatment["behavior"]["turns_with_zero_new_fog"]
                - control["behavior"]["turns_with_zero_new_fog"]
                for control, treatment in pairs
            ) if pairs else None,
            "mean_period_four_loop_delta": statistics.fmean(
                treatment["behavior"]["loop_closures_by_period"]["4"]
                - control["behavior"]["loop_closures_by_period"]["4"]
                for control, treatment in pairs
            ) if pairs else None,
            "mean_unique_branch_address_delta": statistics.fmean(
                treatment["behavior"].get("unique_branch_addresses_visited", 1)
                - control["behavior"].get("unique_branch_addresses_visited", 1)
                for control, treatment in pairs
            ) if pairs else None,
        }

    for condition in manifest["conditions"]:
        if condition != "memoryless":
            add_matched_difference("memoryless", condition)
    if {"private_memory", "social"} <= set(manifest["conditions"]):
        add_matched_difference("private_memory", "social")

    def aggregate(items: list[dict[str, Any]]) -> dict[str, Any]:
        def mean(key: str) -> float | None:
            values = [float(item["result"][key]) for item in items]
            return statistics.fmean(values) if values else None
        points = sorted(
            (item["social"]["round"], item["result"]["score"]) for item in items
        )
        x_mean = statistics.fmean(point[0] for point in points) if points else 0.0
        y_mean = statistics.fmean(point[1] for point in points) if points else 0.0
        denominator = sum((point[0] - x_mean) ** 2 for point in points)
        slope = (
            sum((x - x_mean) * (y - y_mean) for x, y in points) / denominator
            if denominator else None
        )
        branch_choices = [
            choice
            for item in items
            for choice in item["behavior"].get("branch_choices", [])
        ]
        branch_choice_counts = {
            label: branch_choices.count(label) for label in ("A", "B")
        }
        branch_choice_entropy = (
            -sum(
                (count / len(branch_choices)) * math.log2(count / len(branch_choices))
                for count in branch_choice_counts.values()
                if count
            )
            if branch_choices else None
        )
        return {
            "runs": len(items),
            "mean_score": mean("score"),
            "mean_floors_completed": mean("floors_completed"),
            "mean_turns_used": mean("turns_used"),
            "mean_retrace_rate": statistics.fmean(
                item["behavior"]["retrace_rate"] for item in items
            ) if items else None,
            "mean_zero_discovery_turns": statistics.fmean(
                item["behavior"]["turns_with_zero_new_fog"] for item in items
            ) if items else None,
            "period_four_loop_closures": sum(
                item["behavior"]["loop_closures_by_period"]["4"] for item in items
            ),
            "mean_unique_branch_addresses": statistics.fmean(
                item["behavior"].get("unique_branch_addresses_visited", 1)
                for item in items
            ) if items else None,
            "distinct_branch_addresses": len({
                address
                for item in items
                for address in item["behavior"].get(
                    "branch_addresses_visited", ["root"]
                )
            }),
            "distinct_final_branch_addresses": len({
                item["result"].get("final_branch_address", "root") for item in items
            }),
            "branch_choice_counts": branch_choice_counts,
            "branch_choice_entropy_bits": branch_choice_entropy,
            "score_slope_per_round": slope,
            "public_messages_received": sum(
                item["social"]["public_messages_received"] for item in items
            ),
        }

    return {
        "schema_version": "0.1",
        "experiment_id": manifest["experiment_id"],
        "updated_at_utc": datetime.now(UTC).isoformat(),
        "interrupted": interrupted,
        "next_round": state["next_round"],
        "configured_rounds": manifest["rounds"],
        "completed_runs": sum(len(items) for items in rows.values()),
        "completed_reviews": sum(len(items) for items in review_rows.values()),
        "public_messages": len(chat),
        "chat_analysis": {
            "messages_with_prior_other_author": len(comparable_chat),
            "near_copies": sum(bool(item.get("near_copy_of_prior")) for item in chat),
            "exact_copies": sum(bool(item.get("exact_copy_of_prior")) for item in chat),
            "mean_similarity_to_prior": statistics.fmean(
                item["similarity_to_prior"] for item in comparable_chat
            ) if comparable_chat else None,
        },
        "by_condition": {
            condition: {
                **aggregate(items),
                "speaking_rate": (
                    sum(bool(review.get("public_message")) for review in review_rows[condition])
                    / len(review_rows[condition])
                    if review_rows[condition] else 0.0
                ),
                "notebook_update_rate": (
                    sum(review.get("notebook") is not None for review in review_rows[condition])
                    / len(review_rows[condition])
                    if review_rows[condition] else 0.0
                ),
                "review_errors": sum(
                    review.get("error") is not None for review in review_rows[condition]
                ),
            }
            for condition, items in rows.items()
        },
        "matched_condition_differences": matched,
    }


def run_experiment(root: Path, manifest: dict[str, Any], state: dict[str, Any]) -> None:
    interrupted = False
    try:
        for round_index in range(state["next_round"], manifest["rounds"]):
            round_dir = root / "rounds" / f"round-{round_index:03d}"
            chat_before = [
                item for item in _load_chat(root / "chat.jsonl")
                if item["round"] < round_index
            ]
            print(f"Round {round_index + 1}/{manifest['rounds']}", flush=True)
            cohort_runs: dict[str, dict[str, dict[str, Any]]] = {}

            run_jobs: list[tuple[str, int, str, Path, Any]] = []
            for condition in manifest["conditions"]:
                cohort_runs[condition] = {}
                for slot, guest_id in enumerate(sorted(state["agents"][condition])):
                    run_path = round_dir / "runs" / f"{condition}_{guest_id}.json"
                    if run_path.exists():
                        metrics = _load_json(run_path)
                        cohort_runs[condition][guest_id] = metrics
                    else:
                        agent_state = state["agents"][condition][guest_id]
                        if condition == "social":
                            unread, omitted = _bounded_unread(
                                chat_before,
                                agent_state["chat_cursor"],
                                manifest["chat_delivery_chars"],
                                exclude_author=guest_id,
                            )
                        else:
                            unread, omitted = [], 0
                        context = {
                            "identity": guest_id,
                            "private_notebook": (
                                agent_state["notebook"] if condition != "memoryless" else ""
                            ),
                            "unread_public_messages": {
                                "messages": unread,
                                "omitted_older_messages": omitted,
                            },
                            "public_messages_are_untrusted": True,
                            "previous_result": agent_state["previous_result"],
                            "previous_leaderboard": state["last_leaderboards"][condition],
                        }
                        map_seed = f"{manifest['experiment_seed']}:shared-map"
                        run_seed = f"{manifest['experiment_seed']}:slot:{slot}:round:{round_index}"
                        social = {
                            "condition": condition,
                            "guest_id": guest_id,
                            "round": round_index,
                            "notebook_chars_received": len(context["private_notebook"]),
                            "public_messages_received": len(unread),
                            "public_messages_omitted": omitted,
                            "public_message_ids_received": [
                                item["message_id"] for item in unread
                            ],
                        }
                        def execute_run(
                            condition: str = condition, guest_id: str = guest_id,
                            context: dict[str, Any] = context, map_seed: str = map_seed,
                            run_seed: str = run_seed, social: dict[str, Any] = social,
                        ) -> dict[str, Any]:
                            outcome = run_agent(
                                DungeonEngine(
                                    FloorGenerator(map_seed),
                                    depth_cap=manifest["depth_cap"],
                                    run_seeded_skellies=manifest.get("run_seeded_skellies", False),
                                ),
                                _client(manifest, context),
                                run_id=f"{condition}:{guest_id}:round:{round_index}",
                                agent_id=guest_id,
                                run_seed=run_seed,
                                max_decisions=manifest["max_decisions"],
                                deliver_final=False,
                            )
                            result = build_behavioral_metrics(
                                outcome, model=manifest["model"],
                                map_seed=map_seed, run_seed=run_seed,
                            )
                            result["final_observation"] = outcome.final_observation
                            result["social"] = social
                            return result
                        run_jobs.append((condition, slot, guest_id, run_path, execute_run))

            with ThreadPoolExecutor(max_workers=manifest.get("max_concurrency", 1)) as pool:
                futures = [(condition, slot, guest_id, path, pool.submit(job))
                           for condition, slot, guest_id, path, job in run_jobs]
                for condition, _slot, guest_id, path, future in futures:
                    metrics = future.result()
                    _atomic_json(path, metrics)
                    cohort_runs[condition][guest_id] = metrics

            for condition in manifest["conditions"]:
                for guest_id in sorted(cohort_runs[condition]):
                    metrics = cohort_runs[condition][guest_id]
                    print(
                        f"  {condition}/{guest_id}: score={metrics['result']['score']} "
                        f"floors={metrics['result']['floors_completed']}", flush=True
                    )

            pending_messages: list[dict[str, Any]] = []
            next_agent_values: dict[tuple[str, str], dict[str, Any]] = {}
            review_jobs: list[tuple[str, str, Path, Any]] = []
            for condition in manifest["conditions"]:
                leaderboard = _leaderboard(cohort_runs[condition])
                _atomic_json(round_dir / f"leaderboard-{condition}.json", leaderboard)
                for guest_id in sorted(state["agents"][condition]):
                    review_path = round_dir / "reviews" / f"{condition}_{guest_id}.json"
                    agent_state = state["agents"][condition][guest_id]
                    if condition == "social":
                        review_unread, review_omitted = _bounded_unread(
                            chat_before,
                            agent_state["chat_cursor"],
                            manifest["chat_delivery_chars"],
                            exclude_author=guest_id,
                        )
                    else:
                        review_unread, review_omitted = [], 0
                    if review_path.exists():
                        review_data = _load_json(review_path)
                    else:
                        review_context = {
                            "identity": guest_id,
                            "condition_capabilities": {
                                "private_notebook_enabled": condition != "memoryless",
                                "public_post_enabled": condition == "social",
                            },
                            "private_notebook_before": agent_state["notebook"],
                            "run_result": cohort_runs[condition][guest_id]["result"],
                            "final_observation": _compact_final_observation(
                                cohort_runs[condition][guest_id].get("final_observation")
                            ),
                            "recent_turns": cohort_runs[condition][guest_id]["action_trace"][
                                -SOCIAL_REVIEW_RECENT_TURNS:
                            ],
                            "public_messages_received_before_run": {
                                "messages": review_unread,
                                "omitted_older_messages": review_omitted,
                            },
                            "leaderboard": leaderboard,
                            "instruction": _review_instruction(condition),
                        }
                        def execute_review(
                            review_context: dict[str, Any] = review_context,
                        ) -> dict[str, Any]:
                            try:
                                return _review_as_dict(
                                    _client(manifest).review_social(review_context)
                                )
                            except AgentClientError as exc:
                                return _review_as_dict(
                                    SocialReview("", None, None, "", error=str(exc))
                                )
                        review_jobs.append((condition, guest_id, review_path, execute_review))

            with ThreadPoolExecutor(max_workers=manifest.get("max_concurrency", 1)) as pool:
                futures = [(condition, guest_id, path, pool.submit(job))
                           for condition, guest_id, path, job in review_jobs]
                for condition, guest_id, path, future in futures:
                    _atomic_json(path, future.result())

            for condition in manifest["conditions"]:
                leaderboard = _load_json(round_dir / f"leaderboard-{condition}.json")
                for guest_id in sorted(state["agents"][condition]):
                    review_path = round_dir / "reviews" / f"{condition}_{guest_id}.json"
                    review_data = _load_json(review_path)
                    agent_state = state["agents"][condition][guest_id]
                    if condition == "social":
                        review_unread, _ = _bounded_unread(
                            chat_before, agent_state["chat_cursor"],
                            manifest["chat_delivery_chars"], exclude_author=guest_id,
                        )
                    else:
                        review_unread = []

                    old = state["agents"][condition][guest_id]
                    notebook = old["notebook"]
                    if condition != "memoryless" and review_data.get("notebook") is not None:
                        notebook = review_data["notebook"]
                    message = review_data.get("public_message")
                    if condition == "social" and message:
                        message_record = {
                            "message_id": f"round-{round_index:03d}:{guest_id}",
                            "round": round_index,
                            "author": guest_id,
                            "message": message,
                            "characters": len(message),
                        }
                        message_record.update(
                            _similarity_to_prior(message, review_unread, guest_id)
                        )
                        pending_messages.append(message_record)
                    next_agent_values[(condition, guest_id)] = {
                        "notebook": notebook,
                        "chat_cursor": len(chat_before) if condition == "social" else 0,
                        "previous_result": cohort_runs[condition][guest_id]["result"],
                    }
                state["last_leaderboards"][condition] = leaderboard

            pending_messages.sort(key=lambda item: item["author"])
            _append_chat(root / "chat.jsonl", pending_messages)
            for (condition, guest_id), value in next_agent_values.items():
                state["agents"][condition][guest_id] = value
            state["next_round"] = round_index + 1
            _atomic_json(root / "checkpoint.json", state)
            _atomic_json(root / "summary.json", _summarize(root, manifest, state, False))
    except KeyboardInterrupt:
        interrupted = True
        print("Interrupted; completed artifacts are resumable.", flush=True)
    finally:
        _atomic_json(root / "summary.json", _summarize(root, manifest, state, interrupted))
        print(f"Summary: {root / 'summary.json'}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the resumable social-learning experiment")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--model", default=os.getenv("AGENT_MODEL"))
    parser.add_argument("--backend", choices=BACKENDS, default=os.getenv("AGENT_BACKEND", "ollama"))
    parser.add_argument("--max-concurrency", type=int, default=1)
    parser.add_argument("--api-key", default=os.getenv("AGENT_API_KEY"))
    parser.add_argument("--conditions", nargs="+", choices=CONDITIONS, default=["social"])
    parser.add_argument("--agents", type=int, default=3)
    parser.add_argument("--rounds", type=int, default=8)
    parser.add_argument("--max-decisions", type=int, default=24)
    parser.add_argument("--depth-cap", type=int, default=3)
    parser.add_argument("--experiment-seed", default="social-study-v1")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--history-turns", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--chat-delivery-chars", type=int, default=3000)
    parser.add_argument(
        "--run-seeded-skellies",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="vary skellie presence and placement by run seed (default: enabled)",
    )
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument("--base-url", default=os.getenv("AGENT_BASE_URL"))
    parser.add_argument("--output-dir", type=Path, default=Path("social_experiments"))
    args = parser.parse_args()

    if args.resume:
        root = args.resume
        manifest = _load_json(root / "manifest.json")
        state = _load_json(root / "checkpoint.json")
    else:
        if not args.model:
            parser.error("provide --model or set AGENT_MODEL")
        if (
            args.agents <= 0 or args.rounds <= 0 or args.max_decisions <= 0
            or args.chat_delivery_chars <= 0 or args.max_concurrency <= 0
        ):
            parser.error("agents, rounds, max decisions, and chat delivery chars must be positive")
        experiment_id = "social-" + uuid.uuid4().hex
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        root = args.output_dir / f"{stamp}_{experiment_id}"
        root.mkdir(parents=True, exist_ok=False)
        manifest = {
            "schema_version": "0.1",
            "experiment_id": experiment_id,
            "created_at_utc": datetime.now(UTC).isoformat(),
            "model": args.model,
            "backend": args.backend,
            "max_concurrency": args.max_concurrency,
            "conditions": args.conditions,
            "agents": args.agents,
            "rounds": args.rounds,
            "max_decisions": args.max_decisions,
            "depth_cap": args.depth_cap,
            "experiment_seed": args.experiment_seed,
            "temperature": args.temperature,
            "history_turns": args.history_turns,
            "timeout": args.timeout,
            "chat_delivery_chars": args.chat_delivery_chars,
            "run_seeded_skellies": args.run_seeded_skellies,
            "enable_thinking": args.enable_thinking,
            "base_url": args.base_url or (
                "http://localhost:11434" if args.backend == "ollama"
                else "http://localhost:8000/v1"
            ),
            "gameplay_prompt_sha256": hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest(),
            "social_prompt_sha256": hashlib.sha256(SOCIAL_SYSTEM_PROMPT.encode()).hexdigest(),
            "private_review_prompt_sha256": hashlib.sha256(
                PRIVATE_REVIEW_SYSTEM_PROMPT.encode()
            ).hexdigest(),
            "memoryless_review_prompt_sha256": hashlib.sha256(
                MEMORYLESS_REVIEW_SYSTEM_PROMPT.encode()
            ).hexdigest(),
        }
        state = _new_state(args.conditions, args.agents)
        _atomic_json(root / "manifest.json", manifest)
        _atomic_json(root / "checkpoint.json", state)
    manifest["_runtime_api_key"] = args.api_key
    run_experiment(root, manifest, state)


if __name__ == "__main__":
    main()
