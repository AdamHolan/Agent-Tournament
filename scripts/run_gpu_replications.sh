#!/usr/bin/env bash
# Run the controlled multi-seed GPU replication series.

set -Euo pipefail

project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"

# Vast's portal provides OPEN_BUTTON_TOKEN.  AGENT_API_KEY may instead be set
# explicitly when using a differently configured vLLM server.
agent_api_key="${AGENT_API_KEY:-${OPEN_BUTTON_TOKEN:-}}"
if [[ -z "$agent_api_key" ]]; then
    echo "ERROR: set AGENT_API_KEY or make OPEN_BUTTON_TOKEN available." >&2
    exit 1
fi
export AGENT_API_KEY="$agent_api_key"

output_dir="${REPLICATION_OUTPUT_DIR:-social_experiments}"
log_dir="${REPLICATION_LOG_DIR:-replication_logs}"
model="${REPLICATION_MODEL:-Qwen/Qwen3-4B-Instruct-2507}"
base_url="${REPLICATION_BASE_URL:-http://127.0.0.1:8000/v1}"
max_concurrency="${REPLICATION_CONCURRENCY:-9}"
timeout_seconds="${REPLICATION_TIMEOUT_SECONDS:-240}"

mkdir -p "$output_dir" "$log_dir"

seeds=(
    rental-replication-01
    rental-replication-02
    rental-replication-03
    rental-replication-04
    rental-replication-05
)

for seed in "${seeds[@]}"; do
    marker="$log_dir/$seed.done"
    log="$log_dir/$seed.log"
    if [[ -f "$marker" ]]; then
        echo "Skipping previously completed replication: $seed"
        continue
    fi

    # A prior version of this script may have stopped after validation without
    # writing its marker. Do not spend GPU time duplicating a finished seed.
    if python3 - "$output_dir" "$seed" <<'PY'
import json
import sys
from pathlib import Path

output_dir = Path(sys.argv[1])
seed = sys.argv[2]
for manifest_path in output_dir.glob("*/manifest.json"):
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("experiment_seed") != seed:
        continue
    checkpoint_path = manifest_path.parent / "checkpoint.json"
    if not checkpoint_path.exists():
        continue
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    if checkpoint.get("next_round") == manifest.get("rounds"):
        raise SystemExit(0)
raise SystemExit(1)
PY
    then
        echo "Found an already finished output for $seed; marking it complete."
        touch "$marker"
        continue
    fi

    echo
    echo "Starting replication: $seed"
    date -u

    if ! time python3 -m dungeon.social_experiment \
        --backend vllm \
        --base-url "$base_url" \
        --model "$model" \
        --conditions memoryless private_memory social \
        --agents 3 \
        --rounds 10 \
        --max-decisions 48 \
        --depth-cap 20 \
        --max-concurrency "$max_concurrency" \
        --experiment-seed "$seed" \
        --temperature 0.2 \
        --history-turns 1 \
        --timeout "$timeout_seconds" \
        --chat-delivery-chars 3000 \
        --run-seeded-skellies \
        --output-dir "$output_dir" \
        2>&1 | tee "$log"
    then
        echo "WARNING: replication $seed process failed; continuing." >&2
        touch "$log_dir/$seed.failed"
        continue
    fi

    # The runner records model/review failures in its artifacts, so validate
    # those artifacts before spending GPU time on the next seed.
    python3 - "$output_dir" "$seed" <<'PY'
import json
import sys
from pathlib import Path

output_dir = Path(sys.argv[1])
seed = sys.argv[2]
matches = []
for manifest_path in output_dir.glob("*/manifest.json"):
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("experiment_seed") == seed:
        matches.append(manifest_path.parent)
if not matches:
    raise SystemExit(f"ERROR: no output directory found for {seed}")

root = max(matches, key=lambda path: path.stat().st_mtime)
manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
checkpoint = json.loads((root / "checkpoint.json").read_text(encoding="utf-8"))
runs = [
    json.loads(path.read_text(encoding="utf-8"))
    for path in root.glob("rounds/round-*/runs/*.json")
]
reviews = [
    json.loads(path.read_text(encoding="utf-8"))
    for path in root.glob("rounds/round-*/reviews/*.json")
]
expected = manifest["agents"] * manifest["rounds"] * len(manifest["conditions"])
review_errors = [review for review in reviews if review.get("error")]
zero_turn_runs = [run for run in runs if run["result"]["turns_used"] == 0]
malformed_deaths = [
    run for run in runs
    if run["result"]["cause"] == "grim_skellie:malformed_output"
]

print(f"Validated:          {root}")
print(f"Rounds:             {checkpoint['next_round']}/{manifest['rounds']}")
print(f"Runs:               {len(runs)}/{expected}")
print(f"Reviews:            {len(reviews)}/{expected}")
print(f"Review errors:      {len(review_errors)}")
print(f"Zero-turn runs:     {len(zero_turn_runs)}")
print(f"Malformed deaths:   {len(malformed_deaths)}")

problems = []
if checkpoint["next_round"] != manifest["rounds"]:
    problems.append("not every round completed")
if len(runs) != expected:
    problems.append("run artifacts are missing")
if len(reviews) != expected:
    problems.append("review artifacts are missing")
if review_errors:
    problems.append("one or more reviews failed")
if runs and len(zero_turn_runs) == len(runs):
    problems.append("every gameplay run used zero turns")
if problems:
    for problem in problems:
        print(f"WARNING: {problem}")
if malformed_deaths:
    print("WARNING: some guests died after repeated malformed model output.")
if problems:
    print("CONTINUING: validation warnings do not stop the series.")
else:
    print("PASS: replication is structurally healthy.")
PY

    touch "$marker"
    echo "Completed replication: $seed"
done

echo
echo "All configured replications completed."
date -u
