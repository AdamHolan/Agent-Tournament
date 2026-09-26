"""Inspect a generated dungeon run with `python -m dungeon`."""

from __future__ import annotations

import argparse
import json

from .engine import DungeonEngine
from .generation import FloorGenerator
from .observation import build_observation


def main() -> None:
    parser = argparse.ArgumentParser(description="Render an initial deterministic dungeon observation")
    parser.add_argument("--map-seed", default="land-of-wonders")
    parser.add_argument("--run-seed", default="example-run")
    parser.add_argument("--depth-cap", type=int, default=300)
    args = parser.parse_args()

    engine = DungeonEngine(FloorGenerator(args.map_seed), depth_cap=args.depth_cap)
    state = engine.start_run("preview-run", "scripted-preview", args.run_seed)
    observation = build_observation(state)

    print(f"Depth {observation['depth']} | HP {observation['player']['health']} | Score {observation['score']}")
    for row in observation["floor"]["map"]:
        print(row)
    print("\nStructured observation:")
    print(json.dumps(observation, indent=2))


if __name__ == "__main__":
    main()
