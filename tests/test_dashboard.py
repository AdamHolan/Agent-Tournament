import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from dungeon.dashboard import discover_experiments, experiment_detail


class DashboardTests(unittest.TestCase):
    def test_discovers_and_reads_partial_social_experiment(self):
        with TemporaryDirectory() as directory:
            workspace = Path(directory)
            root = workspace / "social_experiments" / "sample"
            run_dir = root / "rounds" / "round-000" / "runs"
            review_dir = root / "rounds" / "round-000" / "reviews"
            run_dir.mkdir(parents=True)
            review_dir.mkdir(parents=True)
            (root / "manifest.json").write_text(json.dumps({
                "model": "test-model", "conditions": ["social"], "rounds": 2,
            }), encoding="utf-8")
            (root / "checkpoint.json").write_text(
                json.dumps({"next_round": 0}), encoding="utf-8"
            )
            (run_dir / "social_guest-001.json").write_text(json.dumps({
                "identity": {"agent_id": "guest-001"},
                "result": {
                    "score": 12, "floors_completed": 1, "turns_used": 3,
                    "health_remaining": 3, "status": "active", "cause": None,
                    "final_branch_address": "A",
                },
                "behavior": {
                    "branch_choices": ["A"], "retrace_rate": 0.0,
                    "action_latency_seconds": {"mean": 1.5},
                },
                "social": {"round": 0, "condition": "social", "guest_id": "guest-001"},
            }), encoding="utf-8")
            (root / "chat.jsonl").write_text(
                json.dumps({"round": 0, "author": "guest-001", "message": "hello"}) + "\n"
                + "partially-written-line",
                encoding="utf-8",
            )

            found = discover_experiments(workspace)
            self.assertEqual(found[0]["id"], "social_experiments/sample")
            detail = experiment_detail(workspace, found[0]["id"])
            self.assertEqual(detail["runs"][0]["final_branch"], "A")
            self.assertEqual(detail["chat"][0]["message"], "hello")
            self.assertEqual(len(detail["chat"]), 1)

    def test_rejects_path_traversal(self):
        with TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                experiment_detail(Path(directory), "social_experiments/../secret")


if __name__ == "__main__":
    unittest.main()
