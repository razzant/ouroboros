"""Resolve metadata dict composition without accepting unknown emitted keys.

Only identity filters of existing progress metadata and closed local helpers'
literal keyword arguments may widen the strict dict-literal scan. Other splats,
renamed comprehension keys, escaping helpers and dynamic call keywords fail.
"""
from __future__ import annotations

import ast


def progress_meta_keys(tree: ast.AST, literal: ast.Dict) -> tuple[set[str], list[str]]:
    parents = {child: node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)}
    functions = (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)

    def scope(node):
        node = parents.get(node)
        while node is not None and not isinstance(node, functions):
            node = parents.get(node)
        return node or tree

    def own_nodes(owner):
        return [node for node in ast.walk(owner) if scope(node) is owner]

    def incoming(value, owner, seen=frozenset()):
        """Prove a filtered dict forwards the existing progress_meta plane."""
        if isinstance(value, ast.Name):
            if value.id == "progress_meta":
                return True  # The caller scans its literal producers and mutations.
            binding = (owner, value.id)
            if binding in seen:
                return False
            nodes = own_nodes(owner)
            values = [node.value for node in nodes if isinstance(node, (ast.Assign, ast.AnnAssign))
                      and any(isinstance(t, ast.Name) and t.id == value.id for t in
                              (node.targets if isinstance(node, ast.Assign) else [node.target]))]
            if values:
                # An alias must remain a read-only view, not a place to author
                # keys the caller's progress_meta-name scan would miss.
                for use in ast.walk(owner):
                    if isinstance(use, ast.Name) and use.id == value.id:
                        parent = parents[use]
                        if isinstance(use.ctx, ast.Store) and isinstance(parent, (ast.Assign, ast.AnnAssign)):
                            continue
                        if not (isinstance(use.ctx, ast.Load) and isinstance(parent, ast.Attribute)
                                and parent.attr in {"get", "items"}):
                            return False
                return all(incoming(v, owner, seen | {binding}) for v in values)
            if isinstance(owner, functions) and any(arg.arg == value.id for arg in ast.walk(owner.args)
                                                    if isinstance(arg, ast.arg)):
                return False
            return owner is not tree and incoming(value, scope(owner), seen | {binding})
        if isinstance(value, ast.IfExp):
            return incoming(value.body, owner, seen) and empty(value.orelse)
        if isinstance(value, ast.BoolOp) and isinstance(value.op, ast.Or):
            return incoming(value.values[0], owner, seen) and all(empty(v) for v in value.values[1:])
        return (isinstance(value, ast.Call) and isinstance(value.func, ast.Attribute)
                and value.func.attr == "get" and len(value.args) == 1
                and isinstance(value.args[0], ast.Constant) and value.args[0].value == "progress_meta")

    def empty(node):
        return isinstance(node, ast.Dict) and not node.keys or isinstance(node, ast.Constant) and node.value is None

    def forwarded(node):
        if not isinstance(node, ast.DictComp) or len(node.generators) != 1:
            return False
        gen = node.generators[0]
        target, iterator = gen.target, gen.iter
        for condition in gen.ifs:
            if not (isinstance(condition, ast.Compare) and isinstance(condition.left, ast.Name)
                    and isinstance(node.key, ast.Name) and condition.left.id == node.key.id
                    and len(condition.ops) == 1 and isinstance(condition.ops[0], (ast.In, ast.NotIn))
                    and isinstance(condition.comparators[0], (ast.Tuple, ast.List, ast.Set))
                    and all(isinstance(n, ast.Constant) and isinstance(n.value, str)
                            for n in condition.comparators[0].elts)):
                return False
        return (not gen.is_async and isinstance(target, ast.Tuple) and len(target.elts) == 2
                and all(isinstance(n, ast.Name) for n in [node.key, node.value, *target.elts])
                and node.key.id == target.elts[0].id and node.value.id == target.elts[1].id
                and isinstance(iterator, ast.Call) and isinstance(iterator.func, ast.Attribute)
                and iterator.func.attr == "items" and not iterator.args and not iterator.keywords
                and incoming(iterator.func.value, scope(node)))

    def call_keywords(node):
        owner = scope(node)
        if (not isinstance(node, ast.Name) or not isinstance(owner, (ast.FunctionDef, ast.AsyncFunctionDef))
                or owner.args.kwarg is None or owner.args.kwarg.arg != node.id):
            return None
        outer = scope(owner)
        if outer is tree:
            return None  # A module-level helper may have callers outside this file.
        siblings = own_nodes(outer)
        bindings = [n for n in siblings if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
                    and n.name == owner.name or isinstance(n, ast.Name)
                    and isinstance(n.ctx, ast.Store) and n.id == owner.name]
        if bindings != [owner]:
            return None
        # No mutation of the kwargs mapping or escape of the callable: every
        # argument key must come from a directly visible call to this helper.
        for use in ast.walk(owner):
            if isinstance(use, ast.Name) and use.id == node.id:
                parent = parents[use]
                if not (isinstance(parent, ast.Dict) and any(
                        key is None and value is use for key, value in zip(parent.keys, parent.values))):
                    return None
        uses = [n for n in ast.walk(outer) if isinstance(n, ast.Name) and n.id == owner.name]
        if not uses:
            return None
        keys = set()
        parameters = {arg.arg for arg in owner.args.args + owner.args.kwonlyargs}
        for use in uses:
            call = parents[use]
            if (scope(use) is not outer or not isinstance(call, ast.Call) or call.func is not use
                    or any(kw.arg is None for kw in call.keywords)):
                return None
            keys.update(kw.arg for kw in call.keywords if kw.arg not in parameters)
        return keys

    keys, unknown = set(), []
    for key, value in zip(literal.keys, literal.values):
        if isinstance(key, ast.Constant) and isinstance(key.value, str):
            keys.add(key.value)
        elif key is None and isinstance(value, ast.Dict):
            nested_keys, nested_unknown = progress_meta_keys(tree, value)
            keys.update(nested_keys)
            unknown.extend(nested_unknown)
        elif key is None and forwarded(value):
            continue
        elif key is None and (expanded := call_keywords(value)) is not None:
            keys.update(expanded)
        else:
            unknown.append(ast.dump(value if key is None else key))
    return keys, unknown
