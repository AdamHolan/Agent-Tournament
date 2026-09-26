import unittest
from dataclasses import replace

from dungeon.engine import DungeonEngine, visible_positions
from dungeon.generation import FloorGenerator, reachable_positions
from dungeon.models import (
    BranchExit,
    Direction,
    FloorDefinition,
    MoveAction,
    MoveSequenceAction,
    Position,
    RunState,
    RunStatus,
    SkellieSpawn,
    SkellieState,
)
from dungeon.observation import build_final_observation, build_observation, render_map
from dungeon.scripted import StairPathPlayer, play_scripted


class StaticGenerator:
    def __init__(self, floors):
        self.floors = floors

    def generate(self, depth):
        return self.floors[depth]


def empty_floor(depth=0, spawn=Position(1, 1), stairs=Position(5, 5), **changes):
    size = 7 + depth // 100
    values = {
        "depth": depth,
        "width": size,
        "height": size,
        "spawn": spawn,
        "stairs_down": stairs,
    }
    values.update(changes)
    return FloorDefinition(**values)


class FloorGenerationTests(unittest.TestCase):
    def test_branch_addresses_generate_stable_distinct_floors_with_two_exits(self):
        generator = FloorGenerator("branching-map")
        ab = generator.generate_branch(("A", "B"))
        ba = generator.generate_branch(("B", "A"))
        self.assertEqual(ab, generator.generate_branch(("A", "B")))
        self.assertNotEqual(ab, ba)
        self.assertEqual([exit_.label for exit_ in ab.exits], ["A", "B"])
        self.assertEqual(len({exit_.position for exit_ in ab.exits}), 2)

    def test_generation_is_repeatable_and_scales(self):
        generator = FloorGenerator("shared-map")
        self.assertEqual(generator.generate(0), generator.generate(0))
        self.assertEqual(generator.generate(99).width, 7)
        self.assertEqual(generator.generate(100).width, 8)
        self.assertEqual(generator.generate(299).width, 9)
        self.assertEqual(generator.generate(300).width, 10)

    def test_hundreds_of_floors_satisfy_invariants(self):
        generator = FloorGenerator(8675309)
        for depth in range(400):
            floor = generator.generate(depth)
            reachable = reachable_positions(floor)
            required = {floor.spawn, floor.stairs_down}
            if floor.heart is not None:
                required.add(floor.heart)
            required.update(enemy.position for enemy in floor.skellies)
            self.assertTrue(required <= reachable, depth)
            self.assertNotIn(floor.spawn, floor.pillars)
            self.assertLessEqual(len(floor.pillars), min(floor.width, floor.height) - 1)
            for enemy in floor.skellies:
                self.assertNotEqual(enemy.position.manhattan(floor.spawn), 1)
            if depth >= 50:
                self.assertEqual(len(floor.skellies), 1 + depth // 100)


class EngineTests(unittest.TestCase):
    def test_labeled_exit_appends_branch_and_generates_that_address(self):
        class BranchGenerator:
            def generate_branch(self, path):
                if not path:
                    return empty_floor(
                        spawn=Position(0, 0),
                        stairs=Position(1, 0),
                        branch_exits=(
                            BranchExit("A", Position(1, 0)),
                            BranchExit("B", Position(0, 1)),
                        ),
                    )
                return empty_floor(depth=len(path), spawn=Position(3, 3))

        engine = DungeonEngine(BranchGenerator(), depth_cap=3)
        state = engine.start_run("run", "agent", "seed")
        result = engine.step(state, MoveAction(Direction.SOUTH))
        self.assertEqual(result.state.branch_path, ("B",))
        completed = next(e for e in result.events if e.kind == "floor_completed")
        self.assertEqual(completed.data["branch_choice"], "B")
        self.assertEqual(completed.data["branch_path_after"], ["B"])

    def test_run_seeded_skellie_placement_is_repeatable_and_varies(self):
        engine = DungeonEngine(FloorGenerator("stable-world"))
        first = engine.start_run("one", "agent", "circumstance-0")
        repeated = engine.start_run("two", "agent", "circumstance-0")
        self.assertEqual(first.skellies, repeated.skellies)
        placements = {
            tuple((enemy.position.x, enemy.position.y) for enemy in
                  engine.start_run(str(index), "agent", f"circumstance-{index}").skellies)
            for index in range(20)
        }
        self.assertGreater(len(placements), 1)

    def test_move_sequence_executes_each_direction_as_a_turn(self):
        floor = empty_floor(spawn=Position(0, 0), stairs=Position(6, 6))
        engine = DungeonEngine(StaticGenerator({0: floor}))
        state = engine.start_run("run", "agent", 1)
        result = engine.step_sequence(
            state, MoveSequenceAction((Direction.EAST, Direction.EAST))
        )
        self.assertEqual(result.state.player_position, Position(2, 0))
        self.assertEqual(result.state.turns_used, 2)
        self.assertEqual(result.executed, (Direction.EAST, Direction.EAST))
        self.assertEqual(result.remaining, ())
        self.assertIsNone(result.interrupted_by)

    def test_move_sequence_stops_when_stairs_are_discovered(self):
        floor = empty_floor(spawn=Position(0, 0), stairs=Position(4, 0))
        engine = DungeonEngine(StaticGenerator({0: floor}))
        state = engine.start_run("run", "agent", 1)
        result = engine.step_sequence(
            state,
            MoveSequenceAction((Direction.EAST, Direction.EAST, Direction.EAST)),
        )
        self.assertEqual(result.state.player_position, Position(2, 0))
        self.assertEqual(result.interrupted_by, "stairs_discovered")
        self.assertEqual(result.executed, (Direction.EAST, Direction.EAST))
        self.assertEqual(result.remaining, (Direction.EAST,))

    def test_move_sequence_stops_after_invalid_move(self):
        floor = empty_floor(spawn=Position(0, 0), stairs=Position(6, 6))
        engine = DungeonEngine(StaticGenerator({0: floor}))
        state = engine.start_run("run", "agent", 1)
        result = engine.step_sequence(
            state, MoveSequenceAction((Direction.NORTH, Direction.EAST))
        )
        self.assertEqual(result.state.player_position, Position(0, 0))
        self.assertEqual(result.state.turns_used, 1)
        self.assertEqual(result.interrupted_by, "invalid_move")
        self.assertEqual(result.remaining, (Direction.EAST,))

    def test_move_sequence_stops_after_floor_transition(self):
        first = empty_floor(spawn=Position(0, 0), stairs=Position(1, 0))
        second = empty_floor(depth=1)
        engine = DungeonEngine(StaticGenerator({0: first, 1: second}), depth_cap=2)
        state = engine.start_run("run", "agent", 1)
        result = engine.step_sequence(
            state, MoveSequenceAction((Direction.EAST, Direction.SOUTH))
        )
        self.assertEqual(result.state.floor.depth, 1)
        self.assertEqual(result.interrupted_by, "floor_transition")
        self.assertEqual(result.remaining, (Direction.SOUTH,))

    def test_private_states_share_definition_but_do_not_mutate_each_other(self):
        floor = empty_floor()
        engine = DungeonEngine(StaticGenerator({0: floor}))
        first = engine.start_run("one", "a", 1)
        second = engine.start_run("two", "b", 2)
        moved = engine.step(first, MoveAction(Direction.EAST)).state
        self.assertIs(first.floor, second.floor)
        self.assertEqual(second.player_position, floor.spawn)
        self.assertNotEqual(moved.player_position, second.player_position)

    def test_invalid_move_consumes_turn_and_enemies_still_act(self):
        floor = empty_floor(
            spawn=Position(0, 0),
            pillars=frozenset({Position(1, 0)}),
            skellies=(SkellieSpawn("skellie-0", Position(0, 2)),),
        )
        engine = DungeonEngine(StaticGenerator({0: floor}))
        state = engine.start_run("run", "agent", "seed")
        result = engine.step(state, MoveAction(Direction.EAST))
        self.assertEqual(result.state.turns_used, 1)
        self.assertIn("invalid_action", [event.kind for event in result.events])
        self.assertIn("skellie_aggroed", [event.kind for event in result.events])

    def test_entering_stairs_has_priority_over_enemy_turn(self):
        first = empty_floor(
            spawn=Position(1, 1),
            stairs=Position(2, 1),
            skellies=(SkellieSpawn("skellie-0", Position(1, 2)),),
        )
        second = empty_floor(depth=1)
        engine = DungeonEngine(StaticGenerator({0: first, 1: second}), depth_cap=2)
        state = engine.start_run("run", "agent", 42)
        result = engine.step(state, MoveAction(Direction.EAST))
        self.assertEqual(result.state.floor.depth, 1)
        self.assertEqual(result.state.floors_completed, 1)
        self.assertEqual(result.state.player_health, 3)
        self.assertNotIn("attack", [event.kind for event in result.events])

    def test_heart_restores_health_and_is_consumed(self):
        floor = empty_floor(heart=Position(2, 1))
        engine = DungeonEngine(StaticGenerator({0: floor}))
        state = replace(engine.start_run("run", "agent", 1), player_health=1)
        result = engine.step(state, MoveAction(Direction.EAST))
        self.assertEqual(result.state.player_health, 3)
        self.assertFalse(result.state.heart_available)
        self.assertIn("heart_consumed", [event.kind for event in result.events])

    def test_dead_skellie_does_not_attack_after_killing_blow(self):
        floor = empty_floor(
            skellies=(SkellieSpawn("skellie-0", Position(2, 1)),),
        )
        engine = DungeonEngine(StaticGenerator({0: floor}))
        state = engine.start_run("run", "agent", "combat-seed")
        result = engine.step(state, MoveAction(Direction.EAST))
        events = list(result.events)
        winning_index = next(index for index, event in enumerate(events) if event.kind == "combat_won")
        last_player_attack = max(
            index for index, event in enumerate(events[:winning_index])
            if event.kind == "attack" and event.data["attacker"] == "player"
        )
        self.assertEqual(events[last_player_attack].data["target_health"], 0)
        self.assertFalse(any(
            event.kind == "attack" and event.data["attacker"] == "skellie-0"
            for event in events[last_player_attack + 1:winning_index]
        ))

    def test_turn_99_summons_grim_skellie(self):
        floor = empty_floor(spawn=Position(0, 0))
        engine = DungeonEngine(StaticGenerator({0: floor}))
        state = replace(engine.start_run("run", "agent", 1), floor_turn=98)
        result = engine.step(state, MoveAction(Direction.NORTH))
        self.assertEqual(result.state.status, RunStatus.DEAD)
        self.assertEqual(result.state.cause, "grim_skellie:floor_turn_limit")
        self.assertIn("grim_skellie_summoned", [event.kind for event in result.events])

    def test_three_malformed_outputs_summon_grim_without_using_turns(self):
        floor = empty_floor()
        engine = DungeonEngine(StaticGenerator({0: floor}))
        state = engine.start_run("run", "agent", 1)
        for _ in range(3):
            result = engine.record_malformed(state)
            state = result.state
        self.assertEqual(state.turns_used, 0)
        self.assertEqual(state.status, RunStatus.DEAD)
        self.assertEqual(state.cause, "grim_skellie:malformed_output")

    def test_timeout_summons_grim_skellie(self):
        floor = empty_floor()
        engine = DungeonEngine(StaticGenerator({0: floor}))
        state = engine.start_run("run", "agent", 1)
        result = engine.record_timeout(state)
        self.assertEqual(result.state.status, RunStatus.DEAD)
        self.assertEqual(result.state.cause, "grim_skellie:model_timeout")
        summoned = next(event for event in result.events if event.kind == "grim_skellie_summoned")
        self.assertEqual(summoned.data["message"], "Hello? Is anyone home?")

    def test_aggroed_skellie_uses_documented_tie_order(self):
        floor = empty_floor(
            spawn=Position(2, 3),
            skellies=(SkellieSpawn("skellie-0", Position(4, 4)),),
        )
        engine = DungeonEngine(StaticGenerator({0: floor}))
        state = engine.start_run("run", "agent", 1)
        result = engine.step(state, MoveAction(Direction.EAST))
        enemy = result.state.skellies[0]
        self.assertTrue(enemy.aggro)
        self.assertEqual(enemy.position, Position(4, 3))

    def test_depth_cap_accepts_any_positive_value(self):
        floor = empty_floor(spawn=Position(1, 1), stairs=Position(2, 1))
        engine = DungeonEngine(StaticGenerator({0: floor}), depth_cap=1)
        state = engine.start_run("run", "agent", 1)
        result = engine.step(state, MoveAction(Direction.EAST))
        self.assertEqual(result.state.status, RunStatus.DEPTH_CAP_REACHED)
        self.assertEqual(result.state.fog_tiles_uncovered, 11)
        self.assertEqual(result.state.score, 110)
        with self.assertRaises(ValueError):
            DungeonEngine(StaticGenerator({}), depth_cap=0)

    def test_scoring_rewards_initial_and_newly_uncovered_tiles_once(self):
        floor = empty_floor(spawn=Position(0, 0), stairs=Position(6, 6))
        engine = DungeonEngine(StaticGenerator({0: floor}))
        state = engine.start_run("run", "agent", 1)
        self.assertEqual(state.fog_tiles_uncovered, 6)
        self.assertEqual(state.score, 6)

        east = engine.step(state, MoveAction(Direction.EAST)).state
        self.assertEqual(east.fog_tiles_uncovered, 9)
        self.assertEqual(east.score, 8)

        west = engine.step(east, MoveAction(Direction.WEST)).state
        self.assertEqual(west.fog_tiles_uncovered, 9)
        self.assertEqual(west.score, 7)

    def test_scoring_rewards_heart_discovery_only_once(self):
        floor = empty_floor(
            spawn=Position(0, 0), stairs=Position(6, 6), heart=Position(2, 0)
        )
        engine = DungeonEngine(StaticGenerator({0: floor}))
        state = engine.start_run("run", "agent", 1)
        self.assertEqual(state.hearts_discovered, 1)
        self.assertEqual(state.score, 31)

        state = engine.step(state, MoveAction(Direction.EAST)).state
        state = engine.step(state, MoveAction(Direction.EAST)).state
        self.assertEqual(state.hearts_discovered, 1)

    def test_scoring_rewards_defeating_a_skellie(self):
        floor = empty_floor(
            stairs=Position(6, 6),
            skellies=(SkellieSpawn("skellie-0", Position(2, 1)),),
        )
        engine = DungeonEngine(StaticGenerator({0: floor}))
        state = engine.start_run("run", "agent", "combat-seed")
        result = engine.step(state, MoveAction(Direction.EAST))
        self.assertEqual(result.state.skellies_defeated, 1)
        self.assertEqual(
            result.state.score,
            50 + result.state.fog_tiles_uncovered - result.state.turns_used,
        )


class ObservationTests(unittest.TestCase):
    def test_recent_move_trace_is_limited_to_latest_sixteen_moves(self):
        engine = DungeonEngine(StaticGenerator({0: empty_floor()}))
        state = engine.start_run("trace-limit", "agent", 1)
        observation = build_observation(
            state, recent_move_trace=("N", "E", "S", "W") * 5
        )
        trace = observation["recent_move_trace"]
        self.assertEqual(trace["directions"], "NESW" * 4)
        self.assertEqual(len(trace["directions"]), 16)

    def test_branch_address_and_labeled_exits_are_observed(self):
        floor = empty_floor(
            spawn=Position(0, 0),
            stairs=Position(1, 0),
            branch_exits=(
                BranchExit("A", Position(1, 0)),
                BranchExit("B", Position(0, 1)),
            ),
        )
        state = DungeonEngine(StaticGenerator({0: floor})).start_run(
            "run", "agent", 1
        )
        observation = build_observation(state)
        self.assertEqual(observation["branch_address"], "root")
        self.assertEqual(observation["floor"]["exit_labels"], ["A", "B"])
        self.assertEqual(observation["movement_options"]["east"]["terrain"], "stairs_A")
        self.assertEqual(observation["movement_options"]["south"]["terrain"], "stairs_B")
        rendered = render_map(state)
        self.assertEqual(rendered[0][1], "A")
        self.assertEqual(rendered[1][0], "B")

    def test_initial_visibility_is_manhattan_two_and_map_is_ascii(self):
        floor = empty_floor(spawn=Position(3, 3))
        engine = DungeonEngine(StaticGenerator({0: floor}))
        state = engine.start_run("run", "agent", 1)
        self.assertEqual(len(visible_positions(state)), 13)
        rendered = render_map(state)
        self.assertEqual(rendered[3][3], "@")
        self.assertEqual(rendered[0][0], "?")
        observation = build_observation(state)
        self.assertEqual(observation["floor"]["map"], list(rendered))
        self.assertEqual(observation["protocol_version"], "0.1")
        self.assertEqual(observation["visited"], [[3, 3]])

    def test_movement_options_explain_pillars_and_bounds_without_hidden_data(self):
        floor = empty_floor(
            spawn=Position(0, 0),
            pillars=frozenset({Position(1, 0)}),
        )
        engine = DungeonEngine(StaticGenerator({0: floor}))
        state = engine.start_run("run", "agent", 1)
        options = build_observation(state)["movement_options"]
        self.assertEqual(options["north"]["terrain"], "out_of_bounds")
        self.assertFalse(options["north"]["passable"])
        self.assertEqual(options["east"]["terrain"], "pillar")
        self.assertFalse(options["east"]["passable"])
        self.assertEqual(options["south"]["terrain"], "floor")
        self.assertTrue(options["south"]["passable"])

    def test_successful_movement_updates_visited_trail(self):
        floor = empty_floor()
        engine = DungeonEngine(StaticGenerator({0: floor}))
        state = engine.start_run("run", "agent", 1)
        state = engine.step(state, MoveAction(Direction.EAST)).state
        self.assertEqual(state.visited, frozenset({Position(1, 1), Position(2, 1)}))

    def test_hidden_dynamic_actor_does_not_leak_into_remembered_map(self):
        floor = empty_floor(
            spawn=Position(0, 0),
            skellies=(SkellieSpawn("skellie-0", Position(6, 6)),),
        )
        engine = DungeonEngine(StaticGenerator({0: floor}))
        state = engine.start_run("run", "agent", 1)
        state = replace(state, revealed=frozenset(
            Position(x, y) for y in range(7) for x in range(7)
        ))
        self.assertNotIn("s", "".join(render_map(state)))

    def test_final_observation_delivers_grim_dialogue_and_cause(self):
        floor = empty_floor()
        engine = DungeonEngine(StaticGenerator({0: floor}))
        state = engine.start_run("run", "agent", 1)
        transition = engine.record_timeout(state)
        final = build_final_observation(transition.state, transition.events)
        result = final["run_result"]
        self.assertEqual(final["observation_type"], "run_final")
        self.assertEqual(result["cause"], "grim_skellie:model_timeout")
        self.assertEqual(result["dialogue"], [{
            "speaker": "grim_skellie",
            "reason": "model_timeout",
            "text": "Hello? Is anyone home?",
        }])
        self.assertEqual(final["legal_actions"], [])
        self.assertEqual(result["events_in_audit_log"], len(transition.events))

    def test_final_observation_summarizes_defeated_enemies(self):
        floor = empty_floor(
            skellies=(SkellieSpawn("skellie-0", Position(2, 1)),),
        )
        engine = DungeonEngine(StaticGenerator({0: floor}))
        state = engine.start_run("run", "agent", "combat-seed")
        combat = engine.step(state, MoveAction(Direction.EAST))
        death = engine.record_timeout(combat.state)
        events = combat.events + death.events
        final = build_final_observation(death.state, events)
        result = final["run_result"]
        self.assertEqual(result["action_summary"]["ordinary_skellies_defeated"], 1)
        self.assertEqual(result["defeated_enemies"], [
            {"kind": "skellie", "actor_id": "skellie-0"},
        ])

    def test_final_observation_rejects_active_run(self):
        floor = empty_floor()
        engine = DungeonEngine(StaticGenerator({0: floor}))
        state = engine.start_run("run", "agent", 1)
        with self.assertRaises(ValueError):
            build_final_observation(state, ())


class ScriptedPlayerTests(unittest.TestCase):
    def test_scripted_path_player_reaches_configured_cap(self):
        floors = {
            depth: empty_floor(depth=depth, spawn=Position(1, 1), stairs=Position(3, 1))
            for depth in range(3)
        }
        engine = DungeonEngine(StaticGenerator(floors), depth_cap=3)
        state = engine.start_run("run", "oracle", 123)
        final, transitions = play_scripted(engine, state, StairPathPlayer())
        self.assertEqual(final.status, RunStatus.DEPTH_CAP_REACHED)
        self.assertEqual(final.floors_completed, 3)
        self.assertEqual(final.turns_used, 6)
        self.assertEqual(len(transitions), 6)


if __name__ == "__main__":
    unittest.main()
