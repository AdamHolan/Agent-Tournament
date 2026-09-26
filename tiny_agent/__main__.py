"""Command-line entry point for `python -m tiny_agent`."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from .agent import Agent
from .models import DemoModel, OllamaModel
from .tools import default_tools


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a tiny educational AI agent")
    parser.add_argument("goal", nargs="*", help="goal for the agent; prompted if omitted")
    parser.add_argument("--demo", action="store_true", help="run without a language model")
    parser.add_argument("--max-steps", type=int, default=8)
    parser.add_argument("--yes", action="store_true", help="allow file writes without confirmation")
    parser.add_argument(
        "--show-thinking",
        action="store_true",
        help="stream a thinking-capable Ollama model's reasoning trace",
    )
    args = parser.parse_args()

    goal = " ".join(args.goal).strip()
    if args.demo:
        goal = goal or "Demonstrate the agent tool loop"
        model = DemoModel()
    else:
        goal = goal or input("What should the agent do? ").strip()
        model_name = os.getenv("AGENT_MODEL")
        if not model_name:
            parser.error("Set AGENT_MODEL to your local model name, or use --demo")
        model = OllamaModel(
            model=model_name,
            base_url=os.getenv("AGENT_BASE_URL", "http://localhost:11434/v1"),
            show_thinking=args.show_thinking,
        )

    if not goal:
        parser.error("The goal cannot be empty")
    agent = Agent(model, default_tools(Path.cwd(), approve_writes=not args.yes), max_steps=args.max_steps)
    result = agent.run(goal)
    print(f"\nAgent: {result.answer}")


if __name__ == "__main__":
    main()
