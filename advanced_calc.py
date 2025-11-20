#!/usr/bin/env python3
"""
Advanced Calculator - single-file CLI tool

Features:
- Safe expression evaluation using AST with a whitelist of operations
- Variables and assignment (e.g. x = 2)
- Many math functions: sin, cos, tan, asin, acos, atan, sinh, cosh, exp, log, ln, sqrt, factorial, gamma
- Complex numbers supported (e.g. 1+2j)
- Numeric derivative: derivative("sin(x)", "x", 1.0)
- Unit conversions: convert(value, from_unit, to_unit) and shorthand "10 cm to m"
- Basic statistics: mean, median, stdev, var, sum, prod
- Matrix class with +, -, *, transpose, det, inv (Gauss-Jordan)
- History, variables inspection, saving/loading sessions
- Small CLI REPL and one-shot evaluation via command-line argument

Usage:
$ python advanced_calculator.py
calc> x = 2
calc> sin(pi/2) + x
...
Commands (prefix with :) - :help, :vars, :history, :save filename, :load filename, :clear, :exit
"""

from __future__ import annotations
import ast
import math
import json
import re
from statistics import mean, median, stdev, variance, pstdev
from functools import reduce
from operator import mul
from typing import Any, Dict, List, Union

try:
    import readline
except Exception:
    readline = None
# ---------------------------
# Utility & math environment
# ---------------------------
Number = Union[int, float, complex]


def prod(iterable):
    return reduce(mul, iterable, 1)


_UNIT_MAP = {
    "m": 1.0,
    "meter": 1.0,
    "meters": 1.0,
    "cm": 0.01,
    "mm": 0.001,
    "km": 1000.0,
    "in": 0.0254,
    "ft": 0.3048,
    "yd": 0.9144,
    "mi": 1609.344,
    # mass (kilograms)
    "kg": 1.0,
    "g": 0.001,
    "lb": 0.45359237,
    # time (seconds)
    "s": 1.0,
    "sec": 1.0,
    "second": 1.0,
    "min": 60.0,
    "h": 3600.0,
    "hr": 3600.0,
    # angle
    "deg": math.pi / 180.0,
    "rad": 1.0,
}


def convert(value: Number, from_unit: str, to_unit: str) -> float:
    fu = from_unit.strip().lower()
    tu = to_unit.strip().lower()
    if fu not in _UNIT_MAP or tu not in _UNIT_MAP:
        raise ValueError(f"Unknown unit: {from_unit} or {to_unit}")
    base = value * _UNIT_MAP[fu]
    return base / _UNIT_MAP[tu]


# ---------------------------
# Matrix class (simple)
# ---------------------------
class Matrix:
    def __init__(self, data: List[List[Number]]):
        if not data or not all(isinstance(row, list) for row in data):
            raise ValueError("Matrix requires a non-empty list of lists")
        rows = len(data)
        cols = len(data[0])
        for r in data:
            if len(r) != cols:
                raise ValueError("All rows must have the same length")
        self.data = [[float(x) for x in row] for row in data]
        self.rows = rows
        self.cols = cols

    def __repr__(self):
        rows = ["[" + ", ".join(f"{v:g}" for v in r) + "]" for r in self.data]
        return "Matrix([" + ",\n        ".join(rows) + "])"

    def __add__(self, other):
        if not isinstance(other, Matrix):
            raise TypeError("Can only add Matrix + Matrix")
        if (self.rows, self.cols) != (other.rows, other.cols):
            raise ValueError("Matrix dimension mismatch")
        return Matrix(
            [
                [self.data[i][j] + other.data[i][j] for j in range(self.cols)]
                for i in range(self.rows)
            ]
        )

    def __sub__(self, other):
        if not isinstance(other, Matrix):
            raise TypeError("Can only subtract Matrix - Matrix")
        if (self.rows, self.cols) != (other.rows, other.cols):
            raise ValueError("Matrix dimension mismatch")
        return Matrix(
            [
                [self.data[i][j] - other.data[i][j] for j in range(self.cols)]
                for i in range(self.rows)
            ]
        )

    def __mul__(self, other):
        if isinstance(other, (int, float, complex)):
            return Matrix(
                [
                    [self.data[i][j] * other for j in range(self.cols)]
                    for i in range(self.rows)
                ]
            )
        if isinstance(other, Matrix):
            if self.cols != other.rows:
                raise ValueError("Matrix multiply dimension mismatch")
            result = [
                [
                    sum(self.data[i][k] * other.data[k][j] for k in range(self.cols))
                    for j in range(other.cols)
                ]
                for i in range(self.rows)
            ]
            return Matrix(result)
        raise TypeError("Matrix can be multiplied by number or Matrix")

    def transpose(self):
        return Matrix(
            [[self.data[i][j] for i in range(self.rows)] for j in range(self.cols)]
        )

    def copy(self):
        return Matrix([row[:] for row in self.data])

    def det(self) -> float:
        if self.rows != self.cols:
            raise ValueError("Determinant requires a square matrix")
        # Use LU-like recursion for small sizes
        n = self.rows
        A = [row[:] for row in self.data]
        det = 1.0
        for i in range(n):
            # find pivot
            pivot = i
            while pivot < n and abs(A[pivot][i]) < 1e-12:
                pivot += 1
            if pivot == n:
                return 0.0
            if pivot != i:
                A[i], A[pivot] = A[pivot], A[i]
                det *= -1
            det *= A[i][i]
            # scale row
            for j in range(i + 1, n):
                if abs(A[j][i]) > 1e-12:
                    factor = A[j][i] / A[i][i]
                    for k in range(i, n):
                        A[j][k] -= factor * A[i][k]
        return det

    def inv(self):
        if self.rows != self.cols:
            raise ValueError("Inverse requires a square matrix")
        n = self.rows
        A = [row[:] for row in self.data]
        I = [[float(i == j) for j in range(n)] for i in range(n)]
        for i in range(n):
            # pivot
            pivot = i
            while pivot < n and abs(A[pivot][i]) < 1e-12:
                pivot += 1
            if pivot == n:
                raise ValueError("Matrix is singular")
            A[i], A[pivot] = A[pivot], A[i]
            I[i], I[pivot] = I[pivot], I[i]
            # normalize pivot row
            piv = A[i][i]
            if abs(piv) < 1e-12:
                raise ValueError("Matrix is singular")
            invp = 1.0 / piv
            A[i] = [x * invp for x in A[i]]
            I[i] = [x * invp for x in I[i]]
            # eliminate other rows
            for r in range(n):
                if r == i:
                    continue
                factor = A[r][i]
                if abs(factor) > 1e-12:
                    A[r] = [A[r][c] - factor * A[i][c] for c in range(n)]
                    I[r] = [I[r][c] - factor * I[i][c] for c in range(n)]
        return Matrix(I)


# ---------------------------
# Safe AST evaluator
# ---------------------------
ALLOWED_BINOPS = {
    ast.Add: lambda a, b: a + b,
    ast.Sub: lambda a, b: a - b,
    ast.Mult: lambda a, b: a * b,
    ast.Div: lambda a, b: a / b,
    ast.FloorDiv: lambda a, b: a // b,
    ast.Mod: lambda a, b: a % b,
    ast.Pow: lambda a, b: a**b,
    ast.MatMult: None,
    ast.LShift: None,
    ast.RShift: None,
    ast.BitOr: None,
    ast.BitXor: None,
    ast.BitAnd: None,
}

ALLOWED_UNARYOPS = {
    ast.UAdd: lambda a: +a,
    ast.USub: lambda a: -a,
    ast.Invert: None,
    ast.Not: None,
}

ALLOWED_CMPOP = {ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE}


class EvalError(Exception):
    pass


class SafeEvaluator(ast.NodeVisitor):
    def __init__(self, env: Dict[str, Any], funcs: Dict[str, Any]):
        self.env = env
        self.funcs = funcs

    def visit(self, node):
        meth = getattr(self, "visit_" + node.__class__.__name__, None)
        if meth is None:
            raise EvalError(f"Unsupported syntax: {node.__class__.__name__}")
        return meth(node)

    def visit_Module(self, node: ast.Module):
        if len(node.body) != 1:
            raise EvalError("Only single expressions or assignments allowed")
        return self.visit(node.body[0])

    def visit_Expr(self, node: ast.Expr):
        return self.visit(node.value)

    def visit_Constant(self, node: ast.Constant):
        return node.value

    def visit_Num(self, node: ast.Num):
        return node.n

    def visit_Str(self, node: ast.Str):
        return node.s

    def visit_NameConstant(self, node: ast.NameConstant):
        return node.value

    def visit_BinOp(self, node: ast.BinOp):
        op_type = type(node.op)
        fn = ALLOWED_BINOPS.get(op_type)
        if fn is None:
            raise EvalError(f"Operator {op_type.__name__} not allowed")
        left = self.visit(node.left)
        right = self.visit(node.right)
        return fn(left, right)

    def visit_UnaryOp(self, node: ast.UnaryOp):
        op_type = type(node.op)
        fn = ALLOWED_UNARYOPS.get(op_type)
        if fn is None:
            raise EvalError(f"Unary operator {op_type.__name__} not allowed")
        return fn(self.visit(node.operand))

    def visit_Name(self, node: ast.Name):
        idn = node.id
        if idn in self.env:
            return self.env[idn]
        if idn in self.funcs:
            return self.funcs[idn]
        # common constants
        if idn == "pi":
            return math.pi
        if idn == "e":
            return math.e
        raise EvalError(f"Unknown name: {idn}")

    def visit_Call(self, node: ast.Call):
        if isinstance(node.func, ast.Name):
            fname = node.func.id
            if fname in self.funcs:
                func = self.funcs[fname]
            else:
                raise EvalError(f"Function {fname} not allowed")
        else:
            raise EvalError("Only simple function calls allowed")
        args = [self.visit(a) for a in node.args]
        kwargs = {}
        for kw in node.keywords:
            kwargs[kw.arg] = self.visit(kw.value)
        try:
            return func(*args, **kwargs)
        except Exception as e:
            raise EvalError(f"Error calling {fname}: {e}") from e

    def visit_List(self, node: ast.List):
        return [self.visit(elt) for elt in node.elts]

    def visit_Tuple(self, node: ast.Tuple):
        return tuple(self.visit(elt) for elt in node.elts)

    def visit_Dict(self, node: ast.Dict):
        return {self.visit(k): self.visit(v) for k, v in zip(node.keys, node.values)}

    def visit_Subscript(self, node: ast.Subscript):
        value = self.visit(node.value)
        idx = self.visit(node.slice)
        return value[idx]

    # visit_Index is deprecated in Python 3.9+
    def visit_Slice(self, node: ast.Slice):
        lower = self.visit(node.lower) if node.lower else None
        upper = self.visit(node.upper) if node.upper else None
        step = self.visit(node.step) if node.step else None
        return slice(lower, upper, step)

    def visit_Assign(self, node: ast.Assign):
        if len(node.targets) != 1:
            raise EvalError("Only single assignment supported")
        target = node.targets[0]
        if not isinstance(target, ast.Name):
            raise EvalError("Can only assign to variable names")
        val = self.visit(node.value)
        self.env[target.id] = val
        return val

    def visit_IfExp(self, node: ast.IfExp):
        cond = self.visit(node.test)
        return self.visit(node.body) if cond else self.visit(node.orelse)

    def visit_Compare(self, node: ast.Compare):
        left = self.visit(node.left)
        for op, comp in zip(node.ops, node.comparators):
            if type(op) not in ALLOWED_CMPOP:
                raise EvalError("Comparison operator not allowed")
            right = self.visit(comp)
            if isinstance(op, ast.Eq) and not (left == right):
                return False
            if isinstance(op, ast.NotEq) and not (left != right):
                return False
            if isinstance(op, ast.Lt) and not (left < right):
                return False
            if isinstance(op, ast.LtE) and not (left <= right):
                return False
            if isinstance(op, ast.Gt) and not (left > right):
                return False
            if isinstance(op, ast.GtE) and not (left >= right):
                return False
            left = right
        return True


# ---------------------------
# Builtins exposed to eval
# ---------------------------
def numeric_derivative(expr: str, var: str, x0: float, h: float = 1e-6) -> float:
    env = {}

    def eval_at(x):
        env[var] = x
        return safe_eval(expr, env)

    return (eval_at(x0 + h) - eval_at(x0 - h)) / (2 * h)


def gamma(x):
    return math.gamma(x)


def combinations(n, k):
    n = int(n)
    k = int(k)
    if k < 0 or k > n:
        return 0
    return math.comb(n, k)


def permutations(n, k=None):
    n = int(n)
    if k is None:
        return math.perm(n)
    k = int(k)
    return math.perm(n, k)


_builtin_funcs = {
    # math aliases
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "asin": math.asin,
    "acos": math.acos,
    "atan": math.atan,
    "sinh": math.sinh,
    "cosh": math.cosh,
    "tanh": math.tanh,
    "sqrt": math.sqrt,
    "log": lambda x, base=10: math.log(x, base),
    "ln": math.log,
    "exp": math.exp,
    "abs": abs,
    "round": round,
    "floor": math.floor,
    "ceil": math.ceil,
    "gcd": math.gcd,
    "lcm": lambda a, b: abs(a * b) // math.gcd(int(a), int(b)) if int(b) != 0 else 0,
    "hypot": math.hypot,
    "degrees": math.degrees,
    "radians": math.radians,
    "mean": mean,
    "median": median,
    "stdev": stdev,
    "var": variance,
    "pstdev": pstdev,
    "sum": sum,
    "prod": prod,
    "complex": complex,
    "conj": lambda z: z.conjugate() if hasattr(z, "conjugate") else z,
    "derivative": numeric_derivative,
    "convert": convert,
    "gamma": gamma,
    "comb": combinations,
    "perm": permutations,
    "Matrix": Matrix,
}


# ---------------------------
# High-level safe_eval wrapper
# ---------------------------
def safe_eval(src: str, user_env: Dict[str, Any] = None):
    if user_env is None:
        user_env = {}
    env = {}
    env.update(_builtin_funcs)

    env.update(user_env)

    evaluator = SafeEvaluator(env=user_env, funcs=_builtin_funcs)
    try:
        tree = ast.parse(src, mode="exec")
    except Exception as e:
        raise EvalError(f"Parse error: {e}") from e
    result = evaluator.visit(tree)
    return result


# ---------------------------
# REPL, history & session
# ---------------------------
PROMPT = "calc> "


class CalculatorREPL:
    def __init__(self):
        self.env: Dict[str, Any] = {}
        self.history: List[str] = []
        self.last_result = None

    def handle_command(self, line: str) -> bool:
        parts = line.strip().split(None, 1)
        cmd = parts[0][1:] if parts else ""
        arg = parts[1] if len(parts) > 1 else ""
        if cmd in ("exit", "quit"):
            return False
        if cmd == "help":
            print(__doc__)
            return True
        if cmd == "vars":
            if not self.env:
                print("(no variables)")
            else:
                for k, v in sorted(self.env.items()):
                    print(f"{k} = {v!r}")
            return True
        if cmd == "history":
            for i, h in enumerate(self.history, 1):
                print(f"{i}: {h}")
            return True
        if cmd == "clear":
            self.env.clear()
            self.history.clear()
            self.last_result = None
            print("Cleared variables and history")
            return True
        if cmd == "save":
            if not arg:
                print("Usage: :save filename")
                return True
            try:
                with open(arg, "w") as f:
                    json.dump(
                        {k: self._serializable(v) for k, v in self.env.items()},
                        f,
                        indent=2,
                    )
                print(f"Saved session to {arg}")
            except Exception as e:
                print("Save error:", e)
            return True
        if cmd == "load":
            if not arg:
                print("Usage: :load filename")
                return True
            try:
                with open(arg) as f:
                    data = json.load(f)
                for k, v in data.items():
                    self.env[k] = v
                print(f"Loaded session from {arg}")
            except Exception as e:
                print("Load error:", e)
            return True
        print(f"Unknown command: :{cmd}")
        return True

    def _serializable(self, v):
        if isinstance(v, (int, float, str, bool)) or v is None:
            return v
        if isinstance(v, complex):
            return {"__complex__": True, "re": v.real, "im": v.imag}
        if isinstance(v, list):
            return [self._serializable(x) for x in v]
        if isinstance(v, Matrix):
            return {"__matrix__": True, "data": v.data}
        # fallback to repr
        return repr(v)

    def _deserialize_value(self, v):
        if isinstance(v, dict):
            if v.get("__complex__"):
                return complex(v["re"], v["im"])
            if v.get("__matrix__"):
                return Matrix(v["data"])
        return v

    def preprocess(self, line: str) -> str:
        # convert "10 cm to m" or "10cm to m" to convert(10, 'cm', 'm')
        m = re.match(r"^\s*([0-9.+\-eE]+)\s*([A-Za-z]+)\s+to\s+([A-Za-z]+)\s*$", line)
        if m:
            val, fu, tu = m.groups()
            return f"convert({val}, '{fu}', '{tu}')"
        # support "expr to unit" where expr can be complex: use regex for trailing " to unit"
        m2 = re.match(r"^\s*(.+?)\s+to\s+([A-Za-z]+)\s*$", line)
        if m2:
            expr, tu = m2.groups()
            return (
                f"convert(({expr}), '{tu}', '{tu}')" if False else f"({expr})"
            )  # fallback to normal eval
        return line

    def eval_line(self, line: str):
        if not line.strip():
            return
        if line.strip().startswith(":"):
            return self.handle_command(line)
        self.history.append(line)
        line = self.preprocess(line)
        try:
            result = safe_eval(line, self.env)
            # update last result
            if result is not None:
                self.last_result = result
                self.env["_"] = result
            if isinstance(result, dict):
                # if load or something returns dict, print nicely
                print(result)
            elif result is not None:
                print(result)
        except EvalError as e:
            print("Error:", e)
        except Exception as e:
            print("Error:", e)
        return True

    def repl(self):
        print("Advanced Calculator. Type :help for help.")
        try:
            while True:
                try:
                    line = input(PROMPT)
                except EOFError:
                    print()
                    break
                cont = self.eval_line(line)
                if cont is False:
                    break
        except KeyboardInterrupt:
            print("\nInterrupted")
        print("Goodbye.")


# ---------------------------
# CLI entry
# ---------------------------
def main():
    import argparse

    p = argparse.ArgumentParser(description="Advanced Calculator")
    p.add_argument("-e", "--expr", help="evaluate expression and exit")
    args = p.parse_args()
    repl = CalculatorREPL()
    if args.expr:
        repl.eval_line(args.expr)
        return
    repl.repl()


if __name__ == "__main__":
    main()
