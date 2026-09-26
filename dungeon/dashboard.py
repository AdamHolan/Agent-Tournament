"""Read-only local web dashboard for dungeon experiment artifacts."""

from __future__ import annotations

import argparse
import json
import mimetypes
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any


STATIC_DIR = Path(__file__).with_name("dashboard_static")
EXPERIMENT_DIRS = ("social_experiments", "experiments")


def _json(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return default


def _jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    rows = []
    for line in lines:
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            rows.append(value)
    return rows


def discover_experiments(workspace: Path) -> list[dict[str, Any]]:
    experiments: list[dict[str, Any]] = []
    for directory in EXPERIMENT_DIRS:
        base = workspace / directory
        if not base.is_dir():
            continue
        for root in base.iterdir():
            if not root.is_dir() or not (root / "manifest.json").is_file():
                continue
            manifest = _json(root / "manifest.json", {})
            summary = _json(root / "summary.json", {})
            checkpoint = _json(root / "checkpoint.json", {})
            configured = manifest.get("rounds")
            next_round = checkpoint.get("next_round", summary.get("next_round"))
            experiments.append({
                "id": f"{directory}/{root.name}",
                "name": root.name,
                "kind": "social" if directory == "social_experiments" else "sequential",
                "modified": root.stat().st_mtime,
                "model": manifest.get("model", ", ".join(manifest.get("models", []))),
                "conditions": manifest.get("conditions", []),
                "configured_rounds": configured,
                "next_round": next_round,
                "completed_runs": summary.get("completed_runs", 0),
                "interrupted": summary.get("interrupted", False),
            })
    return sorted(experiments, key=lambda item: item["modified"], reverse=True)


def _safe_experiment(workspace: Path, identifier: str) -> tuple[Path, str]:
    parts = Path(identifier).parts
    if len(parts) != 2 or parts[0] not in EXPERIMENT_DIRS:
        raise ValueError("unknown experiment")
    root = (workspace / parts[0] / parts[1]).resolve()
    allowed = (workspace / parts[0]).resolve()
    if root.parent != allowed or not root.is_dir():
        raise ValueError("unknown experiment")
    return root, "social" if parts[0] == "social_experiments" else "sequential"


def _social_runs(root: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(root.glob("rounds/round-*/runs/*.json")):
        data = _json(path, {})
        result = data.get("result", {})
        behavior = data.get("behavior", {})
        social = data.get("social", {})
        identity = data.get("identity", {})
        rows.append({
            "file": str(path.relative_to(root)),
            "round": social.get("round"),
            "condition": social.get("condition"),
            "guest": social.get("guest_id", identity.get("agent_id")),
            "score": result.get("score"),
            "floors": result.get("floors_completed"),
            "turns": result.get("turns_used"),
            "health": result.get("health_remaining"),
            "status": result.get("status"),
            "cause": result.get("cause"),
            "final_branch": result.get("final_branch_address"),
            "branch_choices": behavior.get("branch_choices", []),
            "unique_branches": behavior.get("unique_branch_addresses_visited"),
            "retrace_rate": behavior.get("retrace_rate"),
            "zero_fog_turns": behavior.get("turns_with_zero_new_fog"),
            "invalid_rate": behavior.get("invalid_action_rate"),
            "model_requests": behavior.get("model_requests"),
            "mean_latency": behavior.get("action_latency_seconds", {}).get("mean"),
            "messages_received": social.get("public_messages_received", 0),
        })
    return rows


def _social_reviews(root: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(root.glob("rounds/round-*/reviews/*.json")):
        data = _json(path, {})
        round_name = path.parents[1].name.removeprefix("round-")
        condition, _, guest_suffix = path.stem.partition("_guest-")
        rows.append({
            "round": int(round_name) if round_name.isdigit() else None,
            "condition": condition,
            "guest": "guest-" + guest_suffix if guest_suffix else path.stem,
            "reflection": data.get("reflection"),
            "notebook": data.get("notebook"),
            "public_message": data.get("public_message"),
            "thinking": data.get("thinking_text"),
            "latency": data.get("latency_seconds"),
            "error": data.get("error"),
        })
    return rows


def _sequential_runs(root: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted((root / "runs").glob("*.json")):
        data = _json(path, {})
        result = data.get("result", {})
        behavior = data.get("behavior", {})
        identity = data.get("identity", {})
        rows.append({
            "file": str(path.relative_to(root)),
            "round": None,
            "condition": "sequential",
            "guest": identity.get("agent_id", identity.get("model")),
            "score": result.get("score"),
            "floors": result.get("floors_completed"),
            "turns": result.get("turns_used"),
            "health": result.get("health_remaining"),
            "status": result.get("status"),
            "cause": result.get("cause"),
            "final_branch": result.get("final_branch_address"),
            "branch_choices": behavior.get("branch_choices", []),
            "unique_branches": behavior.get("unique_branch_addresses_visited"),
            "retrace_rate": behavior.get("retrace_rate"),
            "zero_fog_turns": behavior.get("turns_with_zero_new_fog"),
            "invalid_rate": behavior.get("invalid_action_rate"),
            "model_requests": behavior.get("model_requests"),
            "mean_latency": behavior.get("action_latency_seconds", {}).get("mean"),
            "messages_received": 0,
        })
    return rows


def experiment_detail(workspace: Path, identifier: str) -> dict[str, Any]:
    root, kind = _safe_experiment(workspace, identifier)
    detail = {
        "id": identifier,
        "name": root.name,
        "kind": kind,
        "manifest": _json(root / "manifest.json", {}),
        "summary": _json(root / "summary.json", {}),
        "checkpoint": _json(root / "checkpoint.json", {}),
        "chat": _jsonl(root / "chat.jsonl"),
        "runs": [],
        "reviews": [],
    }
    if kind == "social":
        detail["runs"] = _social_runs(root)
        detail["reviews"] = _social_reviews(root)
    else:
        detail["runs"] = _sequential_runs(root)
    return detail


def make_handler(workspace: Path):
    class DashboardHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802 - stdlib handler API
            parsed = urllib.parse.urlparse(self.path)
            try:
                if parsed.path == "/api/experiments":
                    self._send_json({"experiments": discover_experiments(workspace)})
                    return
                if parsed.path == "/api/experiment":
                    query = urllib.parse.parse_qs(parsed.query)
                    identifier = query.get("id", [""])[0]
                    self._send_json(experiment_detail(workspace, identifier))
                    return
                relative = "index.html" if parsed.path == "/" else parsed.path.lstrip("/")
                target = (STATIC_DIR / relative).resolve()
                if target.parent != STATIC_DIR.resolve() or not target.is_file():
                    self.send_error(404)
                    return
                content = target.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", mimetypes.guess_type(target.name)[0] or "application/octet-stream")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(content)))
                self.end_headers()
                self.wfile.write(content)
            except ValueError as exc:
                self._send_json({"error": str(exc)}, 404)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def _send_json(self, value: Any, status: int = 200) -> None:
            content = json.dumps(value, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)

        def log_message(self, format: str, *args: Any) -> None:
            return

    return DashboardHandler


def main() -> None:
    parser = argparse.ArgumentParser(description="Monitor dungeon experiments in a local web dashboard")
    parser.add_argument("--workspace", type=Path, default=Path.cwd())
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    workspace = args.workspace.resolve()
    server = ThreadingHTTPServer((args.host, args.port), make_handler(workspace))
    print(f"Dungeon dashboard: http://{args.host}:{args.port}")
    print(f"Reading experiments beneath: {workspace}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nDashboard stopped.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
