import unittest
import urllib.error
import json
import threading
import time
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from dungeon.clients import AgentClientError, AgentDecision, AgentTimeoutError, FinalReceipt, SocialReview
from dungeon.client_factory import create_dungeon_client
from dungeon.engine import DungeonEngine
from dungeon.models import (
    Direction,
    FloorDefinition,
    MoveAction,
    MoveSequenceAction,
    Position,
    RunStatus,
)
from dungeon.metrics import build_behavioral_metrics, write_metrics_file
from dungeon.ollama_agent import (
    ACTION_SCHEMA,
    MEMORYLESS_REVIEW_SCHEMA,
    MEMORYLESS_REVIEW_SYSTEM_PROMPT,
    PRIVATE_REVIEW_SCHEMA,
    PRIVATE_REVIEW_SYSTEM_PROMPT,
    OllamaDungeonAgent,
    SOCIAL_REVIEW_NUM_CTX,
    SOCIAL_REVIEW_SCHEMA,
    SOCIAL_SYSTEM_PROMPT,
)
from dungeon.openai_agent import OpenAICompatibleDungeonAgent
from dungeon.runner import run_agent
from dungeon.social_experiment import (
    _bounded_unread,
    _compact_final_observation,
    _load_chat,
    _load_json,
    _new_state,
    _review_instruction,
    _similarity_to_prior,
    run_experiment,
)


class StaticGenerator:
    def __init__(self, floors):
        self.floors = floors

    def generate(self, depth):
        return self.floors[depth]


def floor(depth=0):
    return FloorDefinition(
        depth=depth,
        width=7,
        height=7,
        spawn=Position(1, 1),
        stairs_down=Position(2, 1),
    )


class RecordingClient:
    def __init__(self, decisions):
        self.decisions = iter(decisions)
        self.observations = []
        self.final = None

    def choose_action(self, observation):
        self.observations.append(observation)
        return next(self.decisions)

    def receive_final(self, observation):
        self.final = observation
        return FinalReceipt("I saw the final state.", 0.01)


class TimeoutClient:
    def choose_action(self, observation):
        raise AgentTimeoutError("timed out")

    def receive_final(self, observation):
        self.final = observation
        return FinalReceipt("Grim Skellie got me.")


class OllamaDungeonAgentTests(unittest.TestCase):
    def test_social_review_is_strictly_parsed(self):
        payloads = []

        def transport(payload):
            payloads.append(payload)
            return {"message": {"content": (
                '{"reflection":"I explored.","notebook":"Seek stairs.",'
                '"public_message":"Stairs may be east."}'
            )}}

        client = OllamaDungeonAgent(
            "model",
            transport=transport,
        )
        review = client.review_social({
            "result": "ended",
            "condition_capabilities": {
                "private_notebook_enabled": True,
                "public_post_enabled": True,
            },
        })
        self.assertEqual(review.notebook, "Seek stairs.")
        self.assertEqual(review.public_message, "Stairs may be east.")
        self.assertIsNone(review.error)
        self.assertEqual(payloads[0]["options"]["num_ctx"], SOCIAL_REVIEW_NUM_CTX)
        self.assertEqual(payloads[0]["messages"][0]["content"], SOCIAL_SYSTEM_PROMPT)
        self.assertEqual(payloads[0]["format"], SOCIAL_REVIEW_SCHEMA)

    def test_review_prompt_and_schema_match_non_social_capabilities(self):
        payloads = []

        def transport(payload):
            payloads.append(payload)
            return {"message": {"content": (
                '{"reflection":"Done.","notebook":null,"public_message":null}'
            )}}

        client = OllamaDungeonAgent("model", transport=transport)
        client.review_social({
            "condition_capabilities": {
                "private_notebook_enabled": True,
                "public_post_enabled": False,
            }
        })
        self.assertEqual(payloads[-1]["messages"][0]["content"], PRIVATE_REVIEW_SYSTEM_PROMPT)
        self.assertEqual(payloads[-1]["format"], PRIVATE_REVIEW_SCHEMA)

        client.review_social({
            "condition_capabilities": {
                "private_notebook_enabled": False,
                "public_post_enabled": False,
            }
        })
        self.assertEqual(payloads[-1]["messages"][0]["content"], MEMORYLESS_REVIEW_SYSTEM_PROMPT)
        self.assertEqual(payloads[-1]["format"], MEMORYLESS_REVIEW_SCHEMA)

    def test_http_error_includes_ollama_response_body(self):
        client = OllamaDungeonAgent("model")
        error = urllib.error.HTTPError(
            client.url,
            400,
            "Bad Request",
            {},
            BytesIO(b'{"error":"request exceeds context size"}'),
        )
        with patch("urllib.request.urlopen", side_effect=error):
            with self.assertRaisesRegex(
                AgentClientError,
                'HTTP 400 Bad Request.*request exceeds context size',
            ):
                client.choose_action({"status": "active"})

    def test_social_review_normalizes_quoted_null_fields(self):
        client = OllamaDungeonAgent(
            "model",
            transport=lambda payload: {"message": {"content": (
                '{"reflection":"Done.","notebook":" null ",'
                '"public_message":"NULL"}'
            )}},
        )
        review = client.review_social({"result": "ended"})
        self.assertIsNone(review.notebook)
        self.assertIsNone(review.public_message)
        self.assertIsNone(review.error)

    def test_valid_move_sequence_is_parsed(self):
        client = OllamaDungeonAgent(
            "model",
            transport=lambda payload: {"message": {"content": (
                '{"action":"move_sequence","directions":["east","south"]}'
            )}},
        )
        decision = client.choose_action({"status": "active"})
        self.assertEqual(
            decision.action,
            MoveSequenceAction((Direction.EAST, Direction.SOUTH)),
        )

    def test_valid_structured_action_is_parsed(self):
        payloads = []

        def transport(payload):
            payloads.append(payload)
            return {"message": {"content": '{"action":"move","direction":"east"}'}}

        client = OllamaDungeonAgent("qwen3:4b-instruct", transport=transport)
        decision = client.choose_action({"status": "active"})
        self.assertEqual(decision.action, MoveAction(Direction.EAST))
        self.assertIsNone(decision.error)
        self.assertEqual(payloads[0]["format"], ACTION_SCHEMA)
        self.assertFalse(payloads[0]["think"])

    def test_thinking_trace_is_preserved_for_diagnostics(self):
        payloads = []

        def transport(payload):
            payloads.append(payload)
            return {"message": {
                "thinking": "East is blocked, so I should move south.",
                "content": '{"action":"move","direction":"south"}',
            }}

        client = OllamaDungeonAgent("thinking-model", enable_thinking=True, transport=transport)
        decision = client.choose_action({"status": "active"})
        self.assertTrue(payloads[0]["think"])
        self.assertEqual(decision.thinking_text, "East is blocked, so I should move south.")
        self.assertEqual(decision.action, MoveAction(Direction.SOUTH))

    def test_invalid_model_json_becomes_rejected_decision(self):
        client = OllamaDungeonAgent(
            "qwen3:4b-instruct",
            transport=lambda payload: {"message": {"content": "north"}},
        )
        decision = client.choose_action({"status": "active"})
        self.assertIsNone(decision.action)
        self.assertIn("not JSON", decision.error)

    def test_context_history_is_bounded(self):
        observed_message_counts = []

        def transport(payload):
            observed_message_counts.append(len(payload["messages"]))
            return {"message": {"content": '{"action":"move","direction":"east"}'}}

        client = OllamaDungeonAgent("model", history_turns=2, transport=transport)
        for turn in range(6):
            client.choose_action({"turn": turn})
        self.assertEqual(observed_message_counts, [2, 4, 6, 6, 6, 6])

    def test_default_context_keeps_one_prior_observation_action_pair(self):
        observed_message_counts = []

        def transport(payload):
            observed_message_counts.append(len(payload["messages"]))
            return {"message": {"content": '{"action":"move","direction":"east"}'}}

        client = OllamaDungeonAgent("model", transport=transport)
        for turn in range(4):
            client.choose_action({"turn": turn})
        self.assertEqual(observed_message_counts, [2, 4, 4, 4])

    def test_final_receipt_does_not_request_an_action_schema(self):
        payloads = []

        def transport(payload):
            payloads.append(payload)
            return {"message": {"content": "I was defeated."}}

        client = OllamaDungeonAgent("model", transport=transport)
        receipt = client.receive_final({"observation_type": "run_final"})
        self.assertEqual(receipt.response_text, "I was defeated.")
        self.assertNotIn("format", payloads[0])


class OpenAICompatibleDungeonAgentTests(unittest.TestCase):
    def test_vllm_backend_selects_compatible_transport(self):
        client = create_dungeon_client(
            backend="vllm", model="model", base_url="http://gpu:8000/v1"
        )
        self.assertIsInstance(client, OpenAICompatibleDungeonAgent)

    def test_translates_schema_and_response_without_leaking_api_key(self):
        captured = {}

        class Response:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return None

            def read(self):
                return json.dumps({
                    "choices": [{"message": {
                        "content": '{"action":"move","direction":"east"}',
                        "reasoning_content": "east is open",
                    }}]
                }).encode()

        def open_request(request, timeout):
            captured["url"] = request.full_url
            captured["headers"] = dict(request.header_items())
            captured["payload"] = json.loads(request.data)
            captured["timeout"] = timeout
            return Response()

        client = OpenAICompatibleDungeonAgent(
            "model", base_url="http://gpu:8000", api_key="secret",
            enable_thinking=True,
        )
        with patch("urllib.request.urlopen", side_effect=open_request):
            decision = client.choose_action({"status": "active"})
        self.assertEqual(captured["url"], "http://gpu:8000/v1/chat/completions")
        self.assertEqual(captured["headers"]["Authorization"], "Bearer secret")
        self.assertEqual(
            captured["payload"]["response_format"]["json_schema"]["schema"],
            ACTION_SCHEMA,
        )
        self.assertNotIn("think", captured["payload"])
        self.assertEqual(decision.action, MoveAction(Direction.EAST))
        self.assertEqual(decision.thinking_text, "east is open")


class ModelRunnerTests(unittest.TestCase):
    def test_final_model_call_can_be_skipped_for_bulk_experiments(self):
        engine = DungeonEngine(StaticGenerator({0: floor()}), depth_cap=1)
        client = RecordingClient([
            AgentDecision(MoveAction(Direction.EAST), "east"),
        ])
        outcome = run_agent(
            engine, client, "run", "agent", 1, deliver_final=False
        )
        self.assertIsNone(client.final)
        self.assertIsNone(outcome.final_receipt)
        self.assertEqual(outcome.state.status, RunStatus.DEPTH_CAP_REACHED)


class SocialExperimentTests(unittest.TestCase):
    def test_review_instructions_match_condition_capabilities(self):
        self.assertIn("no persistent notebook", _review_instruction("memoryless"))
        self.assertIn("no public room", _review_instruction("private_memory"))
        self.assertIn("post one public message", _review_instruction("social"))

    def test_compact_final_observation_removes_duplicated_events(self):
        original = {
            "score": 10,
            "recent_events": [{"kind": "player_moved"}],
            "run_result": {"dialogue": [{"text": "hello?"}]},
        }
        compact = _compact_final_observation(original)
        self.assertNotIn("recent_events", compact)
        self.assertEqual(compact["run_result"], original["run_result"])
        self.assertIn("recent_events", original)

    def test_unread_delivery_excludes_the_guests_own_posts(self):
        messages = [
            {"author": "guest-001", "message": "mine", "message_id": "one"},
            {"author": "guest-002", "message": "theirs", "message_id": "two"},
        ]
        selected, omitted = _bounded_unread(
            messages, 0, 1000, exclude_author="guest-001"
        )
        self.assertEqual([item["message_id"] for item in selected], ["two"])
        self.assertEqual(omitted, 0)

    def test_message_similarity_uses_only_other_authors(self):
        prior = [
            {"author": "guest-001", "message": "stairs east", "message_id": "own"},
            {"author": "guest-002", "message": "stairs are east", "message_id": "other"},
        ]
        result = _similarity_to_prior("stairs are east", prior, "guest-001")
        self.assertEqual(result["most_similar_prior_message_id"], "other")
        self.assertTrue(result["exact_copy_of_prior"])

    def test_unread_delivery_keeps_newest_complete_messages(self):
        messages = [
            {"author": "guest-001", "message": "a" * 20, "message_id": str(index)}
            for index in range(4)
        ]
        selected, omitted = _bounded_unread(messages, 0, 60)
        self.assertEqual([item["message_id"] for item in selected], ["3"])
        self.assertEqual(omitted, 3)

    def test_one_round_persists_notebook_and_publishes_once(self):
        class FakeClient:
            def choose_action(self, observation):
                return AgentDecision(MoveAction(Direction.NORTH), "move")

            def receive_final(self, observation):
                raise AssertionError("social runs use the combined review instead")

            def review_social(self, context):
                return SocialReview("learned", "go east", "stairs east", "review")

        with TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = {
                "experiment_id": "test",
                "model": "fake",
                "base_url": "unused",
                "timeout": 1,
                "temperature": 0,
                "history_turns": 1,
                "enable_thinking": False,
                "conditions": ["social"],
                "agents": 1,
                "rounds": 1,
                "max_decisions": 1,
                "depth_cap": 1,
                "experiment_seed": "test",
                "chat_delivery_chars": 3000,
            }
            state = _new_state(["social"], 1)
            with patch("dungeon.social_experiment._client", return_value=FakeClient()):
                run_experiment(root, manifest, state)
            checkpoint = _load_json(root / "checkpoint.json")
            self.assertEqual(checkpoint["next_round"], 1)
            self.assertEqual(
                checkpoint["agents"]["social"]["guest-001"]["notebook"],
                "go east",
            )
            lines = (root / "chat.jsonl").read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(lines), 1)
            with patch("dungeon.social_experiment._client", return_value=FakeClient()):
                run_experiment(root, manifest, checkpoint)
            self.assertEqual(
                len((root / "chat.jsonl").read_text(encoding="utf-8").splitlines()),
                1,
            )

    def test_runs_and_reviews_are_concurrent_but_publish_after_barrier(self):
        lock = threading.Lock()
        active = {"run": 0, "review": 0}
        maximum = {"run": 0, "review": 0}
        review_contexts = []

        class ConcurrentFakeClient:
            def choose_action(self, observation):
                with lock:
                    active["run"] += 1
                    maximum["run"] = max(maximum["run"], active["run"])
                time.sleep(0.03)
                with lock:
                    active["run"] -= 1
                return AgentDecision(MoveAction(Direction.NORTH), "move")

            def receive_final(self, observation):
                raise AssertionError("not used")

            def review_social(self, context):
                with lock:
                    review_contexts.append(context)
                    active["review"] += 1
                    maximum["review"] = max(maximum["review"], active["review"])
                time.sleep(0.03)
                with lock:
                    active["review"] -= 1
                return SocialReview("learned", None, "hello", "review")

        with TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = {
                "experiment_id": "concurrent-test", "model": "fake",
                "base_url": "unused", "timeout": 1, "temperature": 0,
                "history_turns": 1, "enable_thinking": False,
                "conditions": ["social"], "agents": 2, "rounds": 1,
                "max_decisions": 1, "depth_cap": 1,
                "experiment_seed": "test", "chat_delivery_chars": 3000,
                "max_concurrency": 2,
            }
            with patch(
                "dungeon.social_experiment._client",
                side_effect=lambda *args, **kwargs: ConcurrentFakeClient(),
            ):
                run_experiment(root, manifest, _new_state(["social"], 2))
            self.assertEqual(maximum, {"run": 2, "review": 2})
            self.assertTrue(all(
                not context["public_messages_received_before_run"]["messages"]
                for context in review_contexts
            ))
            messages = _load_chat(root / "chat.jsonl")
            self.assertEqual([item["author"] for item in messages], [
                "guest-001", "guest-002",
            ])

    def test_behavioral_metrics_are_unique_and_measure_retracing(self):
        test_floor = FloorDefinition(
            depth=0, width=7, height=7,
            spawn=Position(1, 1), stairs_down=Position(6, 6),
        )
        engine = DungeonEngine(StaticGenerator({0: test_floor}))
        client = RecordingClient([
            AgentDecision(MoveAction(Direction.EAST), "east", latency_seconds=0.2),
            AgentDecision(MoveAction(Direction.WEST), "west", latency_seconds=0.4),
        ])
        outcome = run_agent(engine, client, "metrics-run", "agent", 1, max_decisions=2)
        metrics = build_behavioral_metrics(
            outcome, model="model:tag", map_seed="map", run_seed=1
        )
        behavior = metrics["behavior"]
        self.assertTrue(metrics["final_reflection"]["received"])
        self.assertEqual(
            metrics["final_reflection"]["response_text"],
            "I saw the final state.",
        )
        self.assertEqual(behavior["moves_executed"], 2)
        self.assertEqual(behavior["moves_to_visited_tiles"], 1)
        self.assertEqual(behavior["retrace_rate"], 0.5)
        self.assertEqual(behavior["immediate_backtracks"], 1)
        self.assertEqual(behavior["loop_closures_by_period"]["2"], 1)
        self.assertEqual(behavior["turns_with_zero_new_fog"], 1)
        self.assertFalse(behavior["per_floor"][0]["completed"])
        self.assertEqual(behavior["per_floor"][0]["unique_positions_visited"], 2)
        self.assertEqual(len(metrics["action_trace"]), 2)
        self.assertEqual(metrics["action_trace"][1]["new_fog_tiles"], 0)
        self.assertAlmostEqual(behavior["action_latency_seconds"]["mean"], 0.3)

        with TemporaryDirectory() as directory:
            first = write_metrics_file(metrics, directory)
            second = write_metrics_file(metrics, directory)
            self.assertNotEqual(first, second)
            self.assertTrue(first.is_file())
            self.assertEqual(len(list(Path(directory).glob("*.json"))), 2)

    def test_sequence_is_executed_and_resolution_is_in_next_observation(self):
        test_floor = FloorDefinition(
            depth=0,
            width=7,
            height=7,
            spawn=Position(0, 0),
            stairs_down=Position(4, 0),
        )
        engine = DungeonEngine(StaticGenerator({0: test_floor}))
        client = RecordingClient([
            AgentDecision(
                MoveSequenceAction((Direction.EAST, Direction.EAST, Direction.EAST)),
                '{"action":"move_sequence","directions":["east","east","east"]}',
            ),
            AgentDecision(MoveAction(Direction.EAST), '{"action":"move","direction":"east"}'),
        ])
        outcome = run_agent(engine, client, "run", "agent", 1, max_decisions=2)
        sequence_event = next(
            event for event in outcome.events if event.kind == "move_sequence_resolved"
        )
        self.assertEqual(sequence_event.data["interrupted_by"], "stairs_discovered")
        self.assertEqual(sequence_event.data["remaining"], ["east"])
        self.assertEqual(
            client.observations[1]["recent_events"][-1]["kind"],
            "move_sequence_resolved",
        )
        self.assertEqual(
            client.observations[1]["recent_move_trace"]["directions"], "EE"
        )

    def test_recent_move_trace_exposes_a_bounded_ordered_path(self):
        test_floor = FloorDefinition(
            depth=0, width=7, height=7,
            spawn=Position(3, 3), stairs_down=Position(6, 6),
        )
        engine = DungeonEngine(StaticGenerator({0: test_floor}))
        client = RecordingClient([
            AgentDecision(MoveAction(direction), direction.value)
            for direction in (
                Direction.NORTH, Direction.EAST, Direction.SOUTH,
                Direction.WEST, Direction.NORTH,
            )
        ])
        outcome = run_agent(engine, client, "trace", "agent", 1, max_decisions=5)
        trace = client.observations[4]["recent_move_trace"]
        self.assertEqual(trace["directions"], "NESW")
        self.assertEqual(trace["scope"], "current_floor")
        self.assertEqual(trace["maximum_moves"], 16)
        self.assertEqual(outcome.final_observation["recent_move_trace"]["directions"], "NESWN")

    def test_recent_move_trace_excludes_invalid_attempts(self):
        test_floor = FloorDefinition(
            depth=0, width=7, height=7,
            spawn=Position(1, 1), stairs_down=Position(6, 6),
            pillars=frozenset({Position(1, 0)}),
        )
        engine = DungeonEngine(StaticGenerator({0: test_floor}))
        client = RecordingClient([
            AgentDecision(MoveAction(Direction.NORTH), "north"),
            AgentDecision(MoveAction(Direction.EAST), "east"),
            AgentDecision(MoveAction(Direction.SOUTH), "south"),
        ])
        run_agent(engine, client, "invalid-trace", "agent", 1, max_decisions=3)
        self.assertEqual(
            client.observations[2]["recent_move_trace"]["directions"], "E"
        )

    def test_recent_move_trace_resets_after_floor_transition(self):
        floors = {
            0: floor(0),
            1: FloorDefinition(
                depth=1, width=7, height=7,
                spawn=Position(3, 3), stairs_down=Position(6, 6),
            ),
        }
        engine = DungeonEngine(StaticGenerator(floors), depth_cap=2)
        client = RecordingClient([
            AgentDecision(MoveAction(Direction.EAST), "east"),
            AgentDecision(MoveAction(Direction.NORTH), "north"),
        ])
        run_agent(engine, client, "floor-reset", "agent", 1, max_decisions=2)
        self.assertEqual(client.observations[1]["depth"], 1)
        self.assertEqual(
            client.observations[1]["recent_move_trace"]["directions"], ""
        )

    def test_client_reaches_depth_cap_and_receives_final_state(self):
        engine = DungeonEngine(StaticGenerator({0: floor()}), depth_cap=1)
        client = RecordingClient([
            AgentDecision(MoveAction(Direction.EAST), '{"action":"move","direction":"east"}'),
        ])
        outcome = run_agent(engine, client, "run", "agent", 1)
        self.assertEqual(outcome.state.status, RunStatus.DEPTH_CAP_REACHED)
        self.assertIs(client.final, outcome.final_observation)
        self.assertEqual(client.final["observation_type"], "run_final")
        self.assertEqual(outcome.final_receipt.response_text, "I saw the final state.")

    def test_three_bad_decisions_flow_through_grim_final_state(self):
        engine = DungeonEngine(StaticGenerator({0: floor()}))
        bad = AgentDecision(None, "bad", "bad response")
        client = RecordingClient([bad, bad, bad])
        outcome = run_agent(engine, client, "run", "agent", 1)
        self.assertEqual(outcome.state.cause, "grim_skellie:malformed_output")
        dialogue = outcome.final_observation["run_result"]["dialogue"]
        self.assertIn("undermine reality", dialogue[0]["text"])
        self.assertEqual(len(client.observations), 3)

    def test_timeout_flows_through_grim_final_state(self):
        engine = DungeonEngine(StaticGenerator({0: floor()}))
        client = TimeoutClient()
        outcome = run_agent(engine, client, "run", "agent", 1)
        self.assertEqual(outcome.state.cause, "grim_skellie:model_timeout")
        self.assertEqual(client.final["run_result"]["dialogue"][0]["text"], "Hello? Is anyone home?")


if __name__ == "__main__":
    unittest.main()
