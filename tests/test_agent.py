import tempfile
import unittest
from pathlib import Path

from tiny_agent.agent import Agent
from tiny_agent.models import DemoModel, OllamaModel
from tiny_agent.tools import calculate, default_tools


class TinyAgentTests(unittest.TestCase):
    def test_calculator(self):
        self.assertEqual(calculate("(17 * 23) + 11"), "402")

    def test_calculator_rejects_code(self):
        with self.assertRaises(ValueError):
            calculate("__import__('os').getcwd()")

    def test_paths_cannot_escape_workspace(self):
        with tempfile.TemporaryDirectory() as directory:
            tools = default_tools(Path(directory), approve_writes=False)
            with self.assertRaises(ValueError):
                tools["read_file"].function("../secret.txt")

    def test_agent_calls_tool_and_finishes(self):
        with tempfile.TemporaryDirectory() as directory:
            result = Agent(DemoModel(), default_tools(Path(directory)), verbose=False).run("demo")
            self.assertEqual(result.steps, 2)
            self.assertIn("402", result.answer)
            self.assertEqual(result.messages[-2]["role"], "tool")

    def test_ollama_message_conversion_preserves_tool_name(self):
        messages = [
            {"role": "assistant", "content": None, "tool_calls": [{
                "id": "call-1",
                "type": "function",
                "function": {"name": "calculate", "arguments": '{"expression":"2+2"}'},
            }]},
            {"role": "tool", "tool_call_id": "call-1", "content": "4"},
        ]
        converted = OllamaModel._messages_for_ollama(messages)
        self.assertEqual(converted[0]["tool_calls"][0]["function"]["arguments"], {"expression": "2+2"})
        self.assertEqual(converted[1]["tool_name"], "calculate")

if __name__ == "__main__":
    unittest.main()
