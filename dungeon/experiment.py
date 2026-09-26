"""Run a reproducible unattended matrix of Ollama dungeon guests."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .engine import DungeonEngine
from .client_factory import BACKENDS, create_dungeon_client
from .generation import FloorGenerator
from .metrics import build_behavioral_metrics, write_metrics_file
from .runner import run_agent


def _mean(rows: list[dict[str, Any]], path: tuple[str, ...]) -> float | None:
    values: list[float] = []
    for row in rows:
        value: Any = row
        for key in path:
            value = value[key]
        if value is not None:
            values.append(float(value))
    return statistics.fmean(values) if values else None


def _summary(
    experiment_id: str,
    started_at: str,
    rows: list[dict[str, Any]],
    failures: list[dict[str, str]],
    interrupted: bool,
) -> dict[str, Any]:
    by_model: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_model.setdefault(row["identity"]["model"], []).append(row)

    def aggregate(items: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "runs": len(items),
            "mean_score": _mean(items, ("result", "score")),
            "mean_floors_completed": _mean(items, ("result", "floors_completed")),
            "mean_turns_used": _mean(items, ("result", "turns_used")),
            "mean_retrace_rate": _mean(items, ("behavior", "retrace_rate")),
            "mean_invalid_action_rate": _mean(items, ("behavior", "invalid_action_rate")),
            "mean_sequence_adoption_rate": _mean(items, ("behavior", "sequence_adoption_rate")),
            "mean_immediate_backtracks": _mean(
                items, ("behavior", "immediate_backtracks")
            ),
            "mean_turns_with_zero_new_fog": _mean(
                items, ("behavior", "turns_with_zero_new_fog")
            ),
            "mean_score_per_model_request": _mean(items, ("behavior", "score_per_model_request")),
            "mean_action_latency_seconds": _mean(
                items, ("behavior", "action_latency_seconds", "mean")
            ),
            "total_model_requests": sum(
                item["behavior"]["model_requests"] for item in items
            ),
        }

    return {
        "schema_version": "0.1",
        "experiment_id": experiment_id,
        "started_at_utc": started_at,
        "finished_at_utc": datetime.now(UTC).isoformat(),
        "interrupted": interrupted,
        "completed_runs": len(rows),
        "failed_runs": len(failures),
        "aggregate": aggregate(rows),
        "by_model": {model: aggregate(items) for model, items in by_model.items()},
        "failures": failures,
        "run_metric_files": [row["artifact"] for row in rows],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run an unattended dungeon experiment")
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument("--backend", choices=BACKENDS, default=os.getenv("AGENT_BACKEND", "ollama"))
    parser.add_argument("--max-concurrency", type=int, default=1)
    parser.add_argument("--api-key", default=os.getenv("AGENT_API_KEY"))
    parser.add_argument("--maps", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--experiment-seed", default="park-study-v1")
    parser.add_argument("--depth-cap", type=int, default=3)
    parser.add_argument("--max-decisions", type=int, default=75)
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
        "--enable-thinking",
        action="store_true",
        help="request and log separate thinking traces from capable models",
    )
    parser.add_argument("--base-url", default=os.getenv("AGENT_BASE_URL"))
    parser.add_argument("--output-dir", default="experiments")
    final_group = parser.add_mutually_exclusive_group()
    final_group.add_argument(
        "--deliver-final",
        dest="deliver_final",
        action="store_true",
        help="request and log the guest's final reflection (the default)",
    )
    final_group.add_argument(
        "--skip-final",
        dest="deliver_final",
        action="store_false",
        help="skip the extra final-reflection inference for faster experiments",
    )
    parser.set_defaults(deliver_final=True)
    args = parser.parse_args()
    models = args.models or ([os.environ["AGENT_MODEL"]] if os.getenv("AGENT_MODEL") else [])
    if not models:
        parser.error("provide --models MODEL [MODEL ...] or set AGENT_MODEL")
    if args.maps <= 0 or args.repeats <= 0 or args.max_concurrency <= 0:
        parser.error("--maps, --repeats, and --max-concurrency must be positive")
    base_url = args.base_url or (
        "http://localhost:11434" if args.backend == "ollama"
        else "http://localhost:8000/v1"
    )

    started_at = datetime.now(UTC).isoformat()
    experiment_id = "experiment-" + uuid.uuid4().hex
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    root = Path(args.output_dir) / f"{stamp}_{experiment_id}"
    metrics_dir = root / "runs"
    root.mkdir(parents=True, exist_ok=False)
    manifest = {
        "experiment_id": experiment_id,
        "models": models,
        "backend": args.backend,
        "base_url": base_url,
        "max_concurrency": args.max_concurrency,
        "maps": args.maps,
        "repeats": args.repeats,
        "experiment_seed": args.experiment_seed,
        "depth_cap": args.depth_cap,
        "max_decisions": args.max_decisions,
        "temperature": args.temperature,
        "history_turns": args.history_turns,
        "enable_thinking": args.enable_thinking,
        "deliver_final": args.deliver_final,
        "run_seeded_skellies": args.run_seeded_skellies,
    }
    (root / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )

    jobs = [
        (model, map_index, repeat)
        for model in models
        for map_index in range(args.maps)
        for repeat in range(args.repeats)
    ]
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    interrupted = False
    print(
        f"Experiment {experiment_id}: {len(jobs)} runs, "
        f"concurrency={args.max_concurrency}; output {root}"
    )

    def execute_job(job: tuple[str, int, int]) -> tuple[Any, ...]:
        model, map_index, repeat = job
        map_seed = f"{args.experiment_seed}:map:{map_index}"
        run_seed = f"{args.experiment_seed}:map:{map_index}:repeat:{repeat}"
        run_id = f"map-{map_index:03d}-repeat-{repeat:03d}-{uuid.uuid4().hex[:8]}"
        outcome = run_agent(
            DungeonEngine(
                FloorGenerator(map_seed),
                depth_cap=args.depth_cap,
                run_seeded_skellies=args.run_seeded_skellies,
            ),
            create_dungeon_client(
                backend=args.backend,
                model=model,
                base_url=base_url,
                api_key=args.api_key,
                timeout_seconds=args.timeout,
                temperature=args.temperature,
                history_turns=args.history_turns,
                enable_thinking=args.enable_thinking,
            ),
            run_id=run_id,
            agent_id=model,
            run_seed=run_seed,
            max_decisions=args.max_decisions,
            deliver_final=args.deliver_final,
        )
        metrics = build_behavioral_metrics(
            outcome, model=model, map_seed=map_seed, run_seed=run_seed
        )
        return model, map_index, repeat, outcome, metrics

    try:
        with ThreadPoolExecutor(max_workers=args.max_concurrency) as pool:
            futures = [pool.submit(execute_job, job) for job in jobs]
            for number, (job, future) in enumerate(zip(jobs, futures), 1):
                model, map_index, repeat = job
                print(f"[{number}/{len(jobs)}] {model} map={map_index} repeat={repeat}", flush=True)
                try:
                    _model, _map, _repeat, outcome, metrics = future.result()
                    artifact = write_metrics_file(metrics, metrics_dir)
                    metrics["artifact"] = str(artifact)
                    rows.append(metrics)
                    print(
                        f"  score={outcome.state.score} floors={outcome.state.floors_completed} "
                        f"turns={outcome.state.turns_used}", flush=True
                    )
                except Exception as exc:  # preserve the rest of an unattended matrix
                    failure = {
                        "model": model, "map": str(map_index),
                        "repeat": str(repeat), "error": repr(exc),
                    }
                    failures.append(failure)
                    print(f"  FAILED: {exc}", flush=True)
    except KeyboardInterrupt:
        interrupted = True
        print("Interrupted; writing a partial summary.", flush=True)
    finally:
        summary = _summary(experiment_id, started_at, rows, failures, interrupted)
        (root / "summary.json").write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8"
        )
        print(f"Summary saved to {root / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
