"""Static discovery of argparse options (AST only: target scripts are never imported or executed).

For every script, the first ``ArgumentParser`` found is described by a list of ``CliOption``:
flags, type, default, choices, nargs, mutually-exclusive group, help text. Defaults that depend on
module constants (``DATA_YAML``, ``Path(__file__).parent`` ...) are resolved with a small safe
evaluator; when that is impossible the default is reported as unresolved instead of being guessed.
"""

from __future__ import annotations

import ast
import operator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

UNRESOLVED = "<unresolved>"
BOOL_ACTIONS = {"store_true", "store_false", "store_const", "BooleanOptionalAction"}


class _Unresolved(Exception):
    pass


@dataclass
class CliOption:
    script: str
    dest: str
    flags: list[str]
    kind: str = "value"  # "value" | "flag" | "positional"
    action: str = "store"
    type_name: str = "str"  # str | int | float | path
    default: Any = None
    default_resolved: bool = True
    choices: list | None = None
    nargs: str | int | None = None
    required: bool = False
    help: str = ""
    metavar: str | None = None
    const: Any = None
    group: str | None = None  # id of the mutually exclusive / argument group
    lineno: int = 0

    @property
    def flag(self) -> str:
        longs = [f for f in self.flags if f.startswith("--")]
        return (longs or self.flags)[0]

    @property
    def is_positional(self) -> bool:
        return not self.flags[0].startswith("-")


@dataclass
class ParserSpec:
    script: str
    description: str = ""
    options: list[CliOption] = field(default_factory=list)
    error: str | None = None

    def by_dest(self) -> dict[str, list[CliOption]]:
        out: dict[str, list[CliOption]] = {}
        for opt in self.options:
            out.setdefault(opt.dest, []).append(opt)
        return out


_BINOPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
           ast.FloorDiv: operator.floordiv}


class _Evaluator:
    def __init__(self, tree: ast.Module, script: Path):
        self.script = script
        self.consts: dict[str, ast.AST] = {}
        self.cache: dict[str, Any] = {}
        self.busy: set[str] = set()
        for node in tree.body:
            if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                self.consts[node.targets[0].id] = node.value
            elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.value is not None:
                self.consts[node.target.id] = node.value

    def eval(self, node: ast.AST) -> Any:
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.Name):
            return self._name(node.id)
        if isinstance(node, (ast.Tuple, ast.List)):
            return [self.eval(e) for e in node.elts]
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            return -self.eval(node.operand)
        if isinstance(node, ast.BinOp) and type(node.op) in _BINOPS:
            return _BINOPS[type(node.op)](self.eval(node.left), self.eval(node.right))
        if isinstance(node, ast.JoinedStr):
            parts = []
            for value in node.values:
                parts.append(str(value.value) if isinstance(value, ast.Constant) else str(self.eval(value.value)))
            return "".join(parts)
        if isinstance(node, ast.Attribute):
            base = self.eval(node.value)
            if isinstance(base, Path) and node.attr in {"parent", "name", "stem", "suffix"}:
                return getattr(base, node.attr)
            raise _Unresolved(ast.dump(node))
        if isinstance(node, ast.Subscript):
            if isinstance(node.value, ast.Attribute) and node.value.attr == "parents":
                return self.eval(node.value.value).parents[self.eval(node.slice)]
            raise _Unresolved(ast.dump(node))
        if isinstance(node, ast.Call):
            return self._call(node)
        raise _Unresolved(ast.dump(node))

    def _name(self, name: str) -> Any:
        if name == "__file__":
            return str(self.script)
        if name in self.cache:
            return self.cache[name]
        if name not in self.consts or name in self.busy:
            raise _Unresolved(name)
        self.busy.add(name)
        try:
            value = self.eval(self.consts[name])
        finally:
            self.busy.discard(name)
        self.cache[name] = value
        return value

    def _call(self, node: ast.Call) -> Any:
        func = node.func
        if isinstance(func, ast.Name) and func.id in {"Path", "str", "int", "float", "bool"}:
            args = [self.eval(a) for a in node.args]
            return {"Path": lambda *a: Path(*a), "str": str, "int": int, "float": float, "bool": bool}[func.id](*args)
        if isinstance(func, ast.Attribute) and func.attr in {"resolve", "joinpath", "as_posix"}:
            base = self.eval(func.value)
            if isinstance(base, str):
                base = Path(base)
            if func.attr == "resolve":
                return base.resolve()
            if func.attr == "as_posix":
                return base.as_posix()
            return base.joinpath(*[self.eval(a) for a in node.args])
        raise _Unresolved(ast.dump(node))


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _type_name(node: ast.AST | None) -> str:
    if node is None:
        return "str"
    name = node.id if isinstance(node, ast.Name) else getattr(node, "attr", "str")
    return {"int": "int", "float": "float", "str": "str", "Path": "path"}.get(name, "str")


def extract_parser(script: str | Path) -> ParserSpec:
    """Return the options declared by the script's argparse parser."""
    script = Path(script)
    spec = ParserSpec(script=str(script))
    try:
        tree = ast.parse(script.read_text(encoding="utf-8", errors="replace"))
    except (OSError, SyntaxError) as exc:
        spec.error = f"{type(exc).__name__}: {exc}"
        return spec

    evaluator = _Evaluator(tree, script.resolve())
    parser_vars: set[str] = set()
    groups: dict[str, str] = {}
    calls: list[ast.Call] = []
    overrides: dict[str, Any] = {}
    doc = ast.get_docstring(tree) or ""

    assigns = [n for n in ast.walk(tree) if isinstance(n, ast.Assign)]
    for node in sorted(assigns, key=lambda n: n.lineno):
        if not (isinstance(node.value, ast.Call) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)):
            continue
        func, target = node.value.func, node.targets[0].id
        if (isinstance(func, ast.Attribute) and func.attr == "ArgumentParser") or (
                isinstance(func, ast.Name) and func.id == "ArgumentParser"):
            parser_vars.add(target)
            for kw in node.value.keywords:
                if kw.arg == "description":
                    try:
                        spec.description = str(evaluator.eval(kw.value))
                    except _Unresolved:
                        spec.description = doc
        elif isinstance(func, ast.Attribute) and func.attr == "add_mutually_exclusive_group":
            groups[target] = f"mutex:{target}"
        elif isinstance(func, ast.Attribute) and func.attr == "add_argument_group":
            groups[target] = f"group:{target}"

    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr == "add_argument":
                calls.append(node)
            elif node.func.attr == "set_defaults":
                for kw in node.keywords:
                    try:
                        overrides[kw.arg] = evaluator.eval(kw.value)
                    except _Unresolved:
                        pass

    for call in sorted(calls, key=lambda c: (c.lineno, c.col_offset)):
        receiver = call.func.value
        group = groups.get(receiver.id) if isinstance(receiver, ast.Name) else None
        if isinstance(receiver, ast.Name) and receiver.id not in parser_vars and group is None:
            continue
        flags = [a.value for a in call.args if isinstance(a, ast.Constant) and isinstance(a.value, str)]
        if not flags:
            continue
        kwargs = {kw.arg: kw.value for kw in call.keywords if kw.arg}

        def lit(key, default=None):
            if key not in kwargs:
                return default
            try:
                return evaluator.eval(kwargs[key])
            except _Unresolved:
                return UNRESOLVED

        action_node = kwargs.get("action")
        action = "store"
        if isinstance(action_node, ast.Constant):
            action = str(action_node.value)
        elif isinstance(action_node, ast.Attribute):
            action = action_node.attr
        elif isinstance(action_node, ast.Name):
            action = action_node.id

        longs = [f for f in flags if f.startswith("--")]
        dest = lit("dest")
        if not dest or dest == UNRESOLVED:
            dest = (longs or flags)[0].lstrip("-").replace("-", "_")

        is_flag = action in BOOL_ACTIONS
        default = lit("default", None)
        default_resolved = default != UNRESOLVED
        if is_flag and "default" not in kwargs:
            default = action == "store_false"
        choices = lit("choices")
        positional = not flags[0].startswith("-")
        spec.options.append(CliOption(
            script=str(script), dest=dest, flags=flags,
            kind="flag" if is_flag else ("positional" if positional else "value"),
            action=action, type_name=_type_name(kwargs.get("type")),
            default=_jsonable(default) if default_resolved else None, default_resolved=default_resolved,
            choices=list(choices) if isinstance(choices, (list, tuple)) else None,
            nargs=lit("nargs"), required=bool(lit("required", False)) or positional,
            help=str(lit("help", "") or ""), metavar=lit("metavar"), const=_jsonable(lit("const")),
            group=group, lineno=call.lineno,
        ))

    for opt in spec.options:
        if opt.dest in overrides:
            opt.default = _jsonable(overrides[opt.dest])
            opt.default_resolved = True
    return spec


def argument_usage_guards(script: str | Path, args_name: str = "args") -> dict[str, str]:
    """Detect options only ever read inside ``if args.<flag>:`` blocks.

    Returns ``{dest: guard_dest}``: ``dest`` is meaningless unless ``guard_dest`` is truthy, because
    every read of ``args.<dest>`` in the script sits inside that single guarded block.
    """
    tree = ast.parse(Path(script).read_text(encoding="utf-8", errors="replace"))

    def arg_attr(node):
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == args_name:
            return node.attr
        return None

    total: dict[str, int] = {}
    for node in ast.walk(tree):
        name = arg_attr(node)
        if name:
            total[name] = total.get(name, 0) + 1

    inside: dict[str, dict[str, int]] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.If) and arg_attr(node.test):
            guard = arg_attr(node.test)
            for stmt in node.body:
                for sub in ast.walk(stmt):
                    name = arg_attr(sub)
                    if name and name != guard:
                        counts = inside.setdefault(name, {})
                        counts[guard] = counts.get(guard, 0) + 1

    guards: dict[str, str] = {}
    for name, per_guard in inside.items():
        if len(per_guard) == 1:
            guard, count = next(iter(per_guard.items()))
            if count == total.get(name, 0):
                guards[name] = guard
    return guards
