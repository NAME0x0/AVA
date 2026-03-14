from __future__ import annotations

import ast
import math
from dataclasses import dataclass
from typing import Any


SAFE_MATH_NAMES = {
    "abs": abs,
    "ceil": math.ceil,
    "cos": math.cos,
    "e": math.e,
    "exp": math.exp,
    "factorial": math.factorial,
    "floor": math.floor,
    "log": math.log,
    "pi": math.pi,
    "pow": pow,
    "sin": math.sin,
    "sqrt": math.sqrt,
    "tan": math.tan,
}


@dataclass(frozen=True, slots=True)
class ToolProtocol:
    name: str
    template: str


PROTOCOLS = (
    ToolProtocol("compact_tags", "[calc]{expression}=>{result}[/calc]"),
    ToolProtocol("compact_line", "calc:{expression}={result}"),
    ToolProtocol("compact_json", '{{"tool":"calculator","input":"{expression}","result":"{result}"}}'),
    ToolProtocol(
        "compact_xml",
        '<tool name="calculator"><input>{expression}</input><result>{result}</result></tool>',
    ),
)


class SafeEvaluator(ast.NodeVisitor):
    def visit_Expression(self, node: ast.Expression) -> Any:
        return self.visit(node.body)

    def visit_BinOp(self, node: ast.BinOp) -> Any:
        left = self.visit(node.left)
        right = self.visit(node.right)
        operators = {
            ast.Add: lambda a, b: a + b,
            ast.Sub: lambda a, b: a - b,
            ast.Mult: lambda a, b: a * b,
            ast.Div: lambda a, b: a / b,
            ast.Pow: lambda a, b: a**b,
            ast.Mod: lambda a, b: a % b,
        }
        for operator_type, func in operators.items():
            if isinstance(node.op, operator_type):
                return func(left, right)
        raise ValueError("unsupported binary operation")

    def visit_UnaryOp(self, node: ast.UnaryOp) -> Any:
        value = self.visit(node.operand)
        if isinstance(node.op, ast.USub):
            return -value
        if isinstance(node.op, ast.UAdd):
            return value
        raise ValueError("unsupported unary operation")

    def visit_Call(self, node: ast.Call) -> Any:
        if not isinstance(node.func, ast.Name):
            raise ValueError("unsafe function call")
        if node.func.id not in SAFE_MATH_NAMES:
            raise ValueError(f"unknown function: {node.func.id}")
        args = [self.visit(arg) for arg in node.args]
        return SAFE_MATH_NAMES[node.func.id](*args)

    def visit_Name(self, node: ast.Name) -> Any:
        if node.id not in SAFE_MATH_NAMES:
            raise ValueError(f"unknown symbol: {node.id}")
        return SAFE_MATH_NAMES[node.id]

    def visit_Constant(self, node: ast.Constant) -> Any:
        if not isinstance(node.value, (int, float)):
            raise ValueError("non-numeric constant")
        return node.value

    def generic_visit(self, node: ast.AST) -> Any:
        raise ValueError(f"unsafe syntax: {type(node).__name__}")


def calculate(expression: str) -> str:
    tree = ast.parse(expression, mode="eval")
    result = SafeEvaluator().visit(tree)
    if isinstance(result, float) and result.is_integer():
        return str(int(result))
    return str(result)


def render_tool_trace(protocol_name: str, expression: str, result: str) -> str:
    for protocol in PROTOCOLS:
        if protocol.name == protocol_name:
            return protocol.template.format(expression=expression, result=result)
    raise KeyError(f"unknown protocol: {protocol_name}")


def list_protocol_names() -> list[str]:
    return [protocol.name for protocol in PROTOCOLS]
