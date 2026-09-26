# Controlled experiment results

This is a compact, shareable summary of the five completed rental GPU replications. The full generated run artifacts are kept locally under `social_experiments/` and excluded from Git because they contain thousands of files. Each replication used three information conditions (`memoryless`, `private_memory`, `social`), three guest slots, ten rounds, and 48 maximum decisions per run. The model and core settings are recorded in `scripts/run_gpu_replications.sh`.

| Replication seed | Gameplay runs | Mean score: memoryless | Mean score: private memory | Mean score: social | Public posts |
|---|---:|---:|---:|---:|---:|
| `rental-replication-01` | 90 | 351.3 | 473.0 | 366.1 | 28 |
| `rental-replication-02` | 90 | 484.5 | 425.4 | 451.6 | 30 |
| `rental-replication-03` | 90 | 385.3 | 390.7 | 315.6 | 27 |
| `rental-replication-04` | 90 | 398.2 | 350.6 | 410.5 | 26 |
| `rental-replication-05` | 90 | 612.5 | 649.6 | 522.6 | 27 |

Across all five replications, there are **450 completed gameplay runs**, **450 recorded post-run reviews**, and **138 public posts**. Three review artifacts report errors; the gameplay runs still completed. Because each condition has 150 runs, the combined mean scores are 446.4 for memoryless, 457.9 for private memory, and 413.3 for social.

The effects vary by seed. The social condition has the highest mean score in one replication, private memory in two, and memoryless in two. These runs do not establish that social communication improves performance. They show why matching seeds, retaining run-level measures, and repeating the experiment matter. The design and measurement plan are in `SOCIAL_EXPERIMENT_DESIGN.md`; implementation is in `dungeon/social_experiment.py` and `dungeon/metrics.py`.
