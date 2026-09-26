"""Small, bounded tools exposed to the model."""

from __future__ import annotations

import ast
import operator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    parameters: dict[str, Any]
    function: Callable[..., str]

    def schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


_BINARY_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}


def _evaluate(node: ast.AST) -> int | float:
    if isinstance(node, ast.Expression):
        return _evaluate(node.body)
    if isinstance(node, ast.Constant) and type(node.value) in (int, float):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _BINARY_OPS:
        return _BINARY_OPS[type(node.op)](_evaluate(node.left), _evaluate(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        return _UNARY_OPS[type(node.op)](_evaluate(node.operand))
    raise ValueError("Only numeric arithmetic is allowed")


def calculate(expression: str) -> str:
    if len(expression) > 200:
        raise ValueError("Expression is too long")
    value = _evaluate(ast.parse(expression, mode="eval"))
    if isinstance(value, (int, float)) and abs(value) > 10**100:
        raise ValueError("Result is too large")
    return str(value)


def _safe_path(workspace: Path, relative_path: str) -> Path:
    workspace = workspace.resolve()
    target = (workspace / relative_path).resolve()
    if target != workspace and workspace not in target.parents:
        raise ValueError("Path must stay inside the workspace")
    return target


def default_tools(workspace: Path, approve_writes: bool = True) -> dict[str, Tool]:
    workspace = workspace.resolve()

    def list_files(path: str = ".") -> str:
        target = _safe_path(workspace, path)
        if not target.is_dir():
            raise ValueError("Path is not a directory")
        entries = sorted(str(p.relative_to(workspace)) for p in target.iterdir())
        return "\n".join(entries[:200]) or "(empty directory)"

    def read_file(path: str) -> str:
        target = _safe_path(workspace, path)
        data = target.read_text(encoding="utf-8")
        return data[:20_000] + ("\n[truncated]" if len(data) > 20_000 else "")

    def write_file(path: str, content: str) -> str:
        target = _safe_path(workspace, path)
        if approve_writes:
            answer = input(f"Allow agent to write {target}? [y/N] ").strip().lower()
            if answer not in {"y", "yes"}:
                return "Write denied by user"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        return f"Wrote {len(content)} characters to {target.relative_to(workspace)}"

    string_arg = lambda description: {  # noqa: E731 - compact schema factory
        "type": "string",
        "description": description,
    }
    return {
        "calculate": Tool(
            "calculate",
            "Evaluate a numeric arithmetic expression.",
            {"type": "object", "properties": {"expression": string_arg("Arithmetic expression")}, "required": ["expression"]},
            calculate,
        ),
        "list_files": Tool(
            "list_files",
            "List files directly inside a workspace directory.",
            {"type": "object", "properties": {"path": string_arg("Relative directory path")}},
            list_files,
        ),
        "read_file": Tool(
            "read_file",
            "Read a UTF-8 text file inside the workspace.",
            {"type": "object", "properties": {"path": string_arg("Relative file path")}, "required": ["path"]},
            read_file,
        ),
        "write_file": Tool(
            "write_file",
            "Write text to a file inside the workspace. The user must approve.",
            {
                "type": "object",
                "properties": {"path": string_arg("Relative file path"), "content": string_arg("Complete file content")},
                "required": ["path", "content"],
            },
            write_file,
        ),
    }
