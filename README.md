# calc

This repository contains two small Python calculator programs:

- advanced_calculator.py — a single-file, feature-rich calculator and safe
  expression evaluator (REPL + one-shot).
- calc.py — a very small interactive menu-driven simple calculator
  (add/sub/mul/div).

## advanced_calculator.py — features

- Safe evaluation of expressions using Python's AST with a strict whitelist of
  operations and functions.
- Variables and assignment: x = 2 (persistent during a REPL session).
- Many math functions and aliases: sin, cos, tan, asin, acos, atan, sinh, cosh,
  exp, log, ln, sqrt, factorial (via math), gamma, comb, perm, etc.
- Complex numbers: e.g. 1+2j, complex(z), conj(z).
- Numeric derivative: derivative("sin(x)", "x", 1.0).
- Unit conversion: convert(value, from_unit, to_unit) and a simple shorthand "10
  cm to m".
- Basic statistics: mean, median, stdev, var, pstdev, sum, prod.
- Matrix class supporting +, -, \*, transpose, det, inv (Gauss–Jordan inverse).
- Session features: history, inspect variables, save/load variables to JSON.
- Small CLI REPL and one-shot evaluation with -e / --expr.

## Usage

From the repo root:

Run the advanced calculator REPL:

```bash
python advanced_calculator.py
```

Evaluate a single expression and exit:

```bash
python advanced_calculator.py -e "sin(pi/2) + 2"
```

Run the simple calculator:

```bash
python calc.py
```

REPL quick examples:

```bash
x = 2
```

```bash
sin(pi/2) + x
```

```bash
derivative("x**2", "x", 3)
```

```bash
convert(10, 'cm', 'm')
```

```bash
Matrix([[1,2],[3,4]]).inv()
```

Shortcut unit conversion (basic parsing — limited):

```bash
10 cm to m
```

Commands in the advanced REPL (prefix with colon `:`):

- ```bash
  :help
  ```

  — print file docstring and help

- ```bash
  :vars
  ```

  — list variables in the session

- ```bash
  :history
  ```

  — show command history

- ```bash
  :save filename
  ```

  — save session variables to filename (JSON)

- ```bash
  :load filename
  ```

  — load variables from a saved session

- ```bash
  :clear
  ```

  — clear variables and history

- ```bash
  :exit
  ```

  or

  ```bash
  :quit
  ```

  — exit REPL

## Environment and compatibility

- Python 3.8+ is recommended. The code contains compatibility stubs for older
  AST node classes.
- Uses only Python standard library modules (ast, math, json, re, statistics,
  functools, operator, typing). Readline is optional.

## Library usage

You can import the safe evaluator functions from advanced_calculator.py (or copy
the file into your project). Example usage from Python:

```python
from advanced_calculator import safe_eval, numeric_derivative, Matrix
env = {}
result = safe_eval("sin(pi/2) + 3", env)
d = numeric_derivative("x**2", "x", 2.0)
m = Matrix([[1,2],[3,4]]).inv()
```

## Simple calculator (calc.py)

A tiny interactive menu that demonstrates basic arithmetic operations (add,
subtract, multiply, divide).

## Files of interest

- advanced_calculator.py — main feature-rich evaluator and REPL.
- calc.py — simple CLI calculator.
