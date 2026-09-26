# Local AI Agents and Dungeon Experiments

This Python project began as a small, readable tool-calling agent and grew into a controlled environment for studying language-model agents. Guests play private runs of a seeded dungeon, then optionally retain a private notebook or post a short message for other guests to read in the next round. The engine owns the rules and validates every action; model output is never authoritative game state.

The platform includes a deterministic dungeon engine, Ollama and OpenAI-compatible model clients, three information conditions, resumable experiment runners, behavioral metrics, tests, and a read-only local dashboard. Five completed replications produced 450 gameplay runs. Their mixed outcomes are in [RESULTS.md](RESULTS.md); they do not establish that public chat improves performance.

## Start without a model

The code uses Python 3.11 or newer and its standard library. From Windows
PowerShell in this project folder:

```powershell
py -m tiny_agent --demo
py -m dungeon --map-seed land-of-wonders
py -m unittest discover -s tests -v
```

The first command demonstrates the tool-calling loop with a deterministic fake model. The second prints a seeded dungeon observation. The tests exercise the agent, game rules, model runners, and dashboard without requiring an inference server.

## Run with a local model

Install and start Ollama, then pull a model you want to use. Set its installed name in PowerShell; the example below uses the model name used during local development:

```powershell
ollama pull qwen3:4b-instruct
$env:AGENT_MODEL = 'qwen3:4b-instruct'
py -m tiny_agent
py -m dungeon.play --depth-cap 3
```

If Ollama is not already running, start it with `ollama serve` in another terminal. The tiny agent can calculate, list and read workspace files, and request writes. Its file tools stay inside the current directory, writes require confirmation by default, and its loop has a step limit. `py -m tiny_agent --show-thinking` displays a separate diagnostic trace when the selected Ollama model provides one.

The dungeon command runs one guest, prints observations and notable events, requests a final reflection, and writes a uniquely named JSON report under `run_metrics/`. It accepts `--map-seed`, `--run-seed`, `--max-decisions`, `--history-turns`, `--timeout`, and `--metrics-dir`. `--show-model-output` prints raw action output; `--enable-thinking` requests a separate thinking trace from a capable model. Each move in a `move_sequence` still consumes a game turn, and sequences stop early when a new decision is needed.

## Run an experiment

The default social experiment runs three neutral guests for eight rounds, with at most 24 model decisions per run. It runs only the `social` condition unless you request the controls:

```powershell
py -m dungeon.social_experiment
py -m dungeon.social_experiment --conditions memoryless private_memory social
```

The three conditions are:

| Condition | Private notebook across rounds | Public messages from other guests |
|---|---|---|
| `memoryless` | No | No |
| `private_memory` | Yes | No |
| `social` | Yes | Yes |

Matched guest slots face the same map and run-randomness seeds across conditions. Runs and post-run reviews may execute concurrently with `--max-concurrency N`, but messages become visible only in the next round. Set concurrency to suit your inference server; the default is one. A smaller pilot is:

```powershell
py -m dungeon.social_experiment --conditions memoryless private_memory social --agents 2 --rounds 3 --max-decisions 18
```

Each experiment writes a manifest, checkpoint, per-run and review JSON files, chat log, and summary in `social_experiments/`. To resume one, pass its actual directory:

```powershell
py -m dungeon.social_experiment --resume '.\social_experiments\YOUR_EXPERIMENT_DIRECTORY'
```

Resume uses the stored manifest and checkpoint. The generated experiment folders are excluded from Git; [RESULTS.md](RESULTS.md) preserves a compact public summary of the completed replications.

For an OpenAI-compatible server such as vLLM, use its model identifier and endpoint. Set `AGENT_API_KEY` if the server requires a bearer token:

```powershell
py -m dungeon.social_experiment --backend vllm --base-url http://127.0.0.1:8000/v1 --model MODEL_NAME --conditions memoryless private_memory social --max-concurrency 4
```

The separate `py -m dungeon.experiment` command runs a model-comparison matrix. Its default is three seeded maps with two run-randomness repeats per model. It writes under `experiments/` and supports multiple models, final reflections, and the same Ollama or OpenAI-compatible backends. See `--help` for each runner's full options. The Bash script in `scripts/run_gpu_replications.sh` records the five-replication rented-GPU configuration; it is intended for a Bash environment with an inference server, rather than local Windows PowerShell.

## Watch runs in a browser

```powershell
py -m dungeon.dashboard
```

Open `http://127.0.0.1:8765`. The read-only dashboard discovers local `social_experiments/` and `experiments/` folders and refreshes every five seconds. It shows run progress, scores, depth, branch choices, chat, failures, and post-run reviews. It binds to localhost by default.

## Project map

| Path | Purpose |
|---|---|
| `tiny_agent/` | Small tool-calling agent and model adapters |
| `dungeon/engine.py`, `generation.py`, `observation.py` | Game state transitions, seeded floors, and fog-safe observations |
| `dungeon/social_experiment.py`, `experiment.py` | Resumable social study and model-comparison runner |
| `dungeon/metrics.py`, `dashboard.py`, `dashboard_static/` | Behavioral reports and local dashboard |
| `tests/` | Automated behavior and protocol tests |
| [GAME_SPEC.md](GAME_SPEC.md) | Game rules and design context |
| [SOCIAL_EXPERIMENT_DESIGN.md](SOCIAL_EXPERIMENT_DESIGN.md) | Study design and implementation notes |
| [PROJECT_BRIEF.md](PROJECT_BRIEF.md) | Original product vision and historical rationale |
| [RESULTS.md](RESULTS.md) | Summary of completed replications and their limits |

This is a local research and engineering playground. The public multi-user agent park described in the project brief is a future direction, not a deployed feature.
