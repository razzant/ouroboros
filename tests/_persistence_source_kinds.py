"""Source-only evidence for the actor-source store; not a Python interpreter.

Follow the named writer/relay calls and just their naming arguments. The supported
syntax is literal strings, f-strings/concatenation, single local bindings, direct
string constants imported from a scanned module, literal ``PurePosixPath(...).name``,
literal alternatives and the enclosing literal loop (including a local module tuple).
A relay's branch on its own parameter (``if part``, ``isinstance(payload, str)``)
is decided where a caller names it; an undecidable payload type leaves the
extension opaque. Any other rebinding makes a name unknown, and so does importing
or defining it again as another object; a call still follows its last import or
definition. A new, opaque naming shape needs review; computed direct paths, dynamic
dispatch and assignments through aliases are not promised coverage.
"""
from __future__ import annotations

import ast
import itertools
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path, PurePosixPath

from tests._persistence_rows import inventory_rows, lifetime

_WRITE = "ouroboros.artifacts.store_actor_source_bytes"
_STREAM = "ouroboros.artifacts.stream_actor_source"
_ADDRESS = "ouroboros.artifacts.actor_source_reference"
_ADDRESS_COPY = "ouroboros.result_material._retained_copy"
_SESSION_WRITER = "ouroboros.review_session_reads._stored_read_history"
_SESSION_CALL = "ouroboros.review_session_reads.session_read_facts"
_PRICE_ONLY = {
    "ouroboros.context_renewal.physical_window_cut",
    "ouroboros.context_renewal.renewal_view",
    "ouroboros.context_source_view.fresh_room_rows",
    "ouroboros.section_floor.predicted_section",
    "ouroboros.tools.process_facts.check_output_views",
}
# These helpers forward naming arguments; the call supplying the name is the
# producer. The recipes below are read from their source, not copied signatures.
_RELAYS = {
    "ouroboros.artifacts.persist_exact_text_source",
    "ouroboros.budget_pause._exact_continuation_row",
    "ouroboros.chat_chain.retain_memory_source",
    "ouroboros.delegate_collection._retain",
    "ouroboros.delegate_plan._retain",
    "ouroboros.owner_wait.store_continuation_source",
    "ouroboros.plan_intention_ledger.settle_ledger_seat",
    "ouroboros.plan_intention_ledger.write_intention_record",
    "ouroboros.review_ledger._retain_seat_text",
    "ouroboros.review_ledger._retain_wave_sources",
    "ouroboros.review_ledger.retain_text_source",
    "ouroboros.review_native_episode.NativeToolRoundReviewExecutor._store_source",
    "ouroboros.review_thread_retention._Pass.thread",
    "ouroboros.review_thread_retention._transfer",
    "ouroboros.review_thread_retention.retain_export",
    "ouroboros.tool_result_delivery.persist_result_source",
    "ouroboros.tools.github_workflows._retain",
    "ouroboros.tools.review_brief_coupling._store_review_source",
}
_SINKS = {_WRITE, _STREAM, _ADDRESS}
_CALLS = _RELAYS | _SINKS
_WORDS = re.compile(r"\b(?:" + "|".join(sorted({key.rsplit(".", 1)[1] for key in _CALLS | {_SESSION_CALL}})) + r")\b")
_VALUE_HELPERS = {"ouroboros.delegate_activity._source_id", "ouroboros.review_ledger.new_record_id"}
_COPY_ONLY = {"ouroboros.source_retention.RetentionWalk._source", "ouroboros.review_source_closure.promote_source_payload"}
_FN = (ast.FunctionDef, ast.AsyncFunctionDef)
_HOLE = "\x01"
_PARAM = re.compile(r"\x00(\w+)\x00")
_GUARD = re.compile(r"\x02(\w+):(\w+)\x02")  # a relay branch on that parameter, still undecided


def _literal(value):
    """The text a name certainly has, without holes, parameters or branches."""
    return _GUARD.sub("", _PARAM.sub("", value)).replace(_HOLE, "")


def _names(part):
    return {*_PARAM.findall(part), *(name for name, _ in _GUARD.findall(part))}


def _concat(items):
    items = tuple(items)
    if len(items) > 80:
        raise ValueError("source name has too many parts")
    n = 1
    for item in items:
        n *= len(item)
    if n > 64:
        raise ValueError("source name has too many alternatives")
    return {"".join(parts) for parts in itertools.product(*items)}


def _target_pairs(target, value):
    if isinstance(target, ast.Name):
        return {target.id: value}
    if isinstance(target, (ast.Tuple, ast.List)) and isinstance(value, (ast.Tuple, ast.List)):
        if len(target.elts) == len(value.elts) and all(isinstance(t, ast.Name) for t in target.elts):
            return {t.id: v for t, v in zip(target.elts, value.elts)}
    return {}


@dataclass
class Scope:
    key: str
    module: str
    parent: Scope | None = None
    fn: ast.FunctionDef | ast.AsyncFunctionDef | None = None
    cls: str = ""
    names: dict[str, str] = field(default_factory=dict)
    unsettled: set[str] = field(default_factory=set)
    modules: set[str] = field(default_factory=set)  # bound by a plain ``import``: a module, never a value
    assigned: dict[str, list[ast.expr | None]] = field(default_factory=dict)
    dict_edits: dict[str, set] = field(default_factory=dict)
    returns: list[ast.expr] = field(default_factory=list)

    @property
    def parameters(self):
        if self.fn is None:
            return ()
        args = self.fn.args
        return tuple(a.arg for a in (*args.posonlyargs, *args.args, *args.kwonlyargs))

    def bind(self, name, value):
        self.assigned.setdefault(name, []).append(value)

    def bind_name(self, name, target):
        """An import, def or class: the last one is the callee; another target leaves the name unsettled."""
        if self.names.get(name, target) != target:
            self.unsettled.add(name)
        self.names[name] = target

    def settled(self, node):
        """``node``'s first name is one imported or defined object in its scope: no other binding or ``*`` import."""
        while isinstance(node, ast.Attribute):
            node = node.value
        owner = self.defining(node.id) if isinstance(node, ast.Name) else None
        return (owner is not None and node.id in owner.names and "*" not in owner.names
                and node.id not in {*owner.assigned, *owner.parameters, *owner.unsettled})

    def defining(self, name):
        scope = self
        while scope is not None:
            if name in scope.names or name in scope.assigned or name in scope.parameters:
                return scope
            scope = scope.parent
        return None

    def callee(self, node):
        if isinstance(node, ast.Name):
            owner = self.defining(node.id)
            return owner.names.get(node.id, "") if owner and node.id not in owner.assigned else ""
        if isinstance(node, ast.Attribute):
            if isinstance(node.value, ast.Name) and node.value.id in ("self", "cls") and self.cls:
                return f"{self.cls}.{node.attr}"
            base = self.callee(node.value)
            return f"{base}.{node.attr}" if base else ""
        return ""


@dataclass
class Site:
    scope: Scope
    node: ast.AST
    loops: tuple[ast.For, ...]

    def environments(self, names):
        """Only a loop in this call's ancestry can supply one of its naming values."""
        for loop in reversed(self.loops):
            targets = {n.id for n in ast.walk(loop.target) if isinstance(n, ast.Name)}
            if not targets & names:
                continue
            values = self.expression(loop.iter)
            # A named list may have been populated (the browser intake starts
            # with []). Only a literal list at this site or a bound tuple is fixed.
            if isinstance(values, ast.Tuple) or isinstance(loop.iter, ast.List):
                if len(values.elts) > 32:
                    raise ValueError("too many values in a source naming loop")
                return [_target_pairs(loop.target, item) for item in values.elts]
            break
        return [{}]

    def expression(self, node):
        if isinstance(node, ast.Name):
            owner = self.scope.defining(node.id)
            values = owner.assigned.get(node.id, []) if owner else []
            if len(values) == 1:
                return values[0]
        return node

    @property
    def address(self):
        return f"{self.scope.key}:{self.node.lineno}"


class SourceScan:
    def __init__(self, root, categories):
        self.root, self.categories = root, categories
        self.scopes, self.references, self.calls = {}, [], []
        self.writes, self.findings = set(), set()
        self.recipes = {}
        paths = [*root.glob("ouroboros/**/*.py"), *root.glob("supervisor/**/*.py"), root / "server.py", root / "launcher.py"]
        for path in sorted(p for p in paths if p.is_file()):
            text = path.read_text(encoding="utf-8")
            if _WORDS.search(text):
                module = ".".join(path.relative_to(root).with_suffix("").parts)
                if module.endswith(".__init__"):
                    module = module.removesuffix(".__init__")
                scope = self.scopes[module] = Scope(module, module)
                for node in ast.parse(text, filename=str(path)).body:
                    self.visit(node, scope)
        self.finish()

    def visit(self, node, scope, loops=(), cls="", *, is_callee=False):
        if isinstance(node, _FN):
            for value in (*node.decorator_list, *node.args.defaults, *filter(None, node.args.kw_defaults)):
                self.visit(value, scope, loops)
            key = f"{cls or scope.key}.{node.name}"
            if not cls:
                scope.bind_name(node.name, key)
            inner = self.scopes[key] = Scope(key, scope.module, scope, node, cls or scope.cls)
            for item in node.body:
                self.visit(item, inner, loops)
            return
        if isinstance(node, ast.ClassDef):
            key = f"{scope.key}.{node.name}"
            scope.bind_name(node.name, key)
            for item in node.body:
                if isinstance(item, _FN):
                    self.visit(item, scope, loops, cls=key)
            return
        if isinstance(node, ast.For):
            for name in ast.walk(node.target):
                if isinstance(name, ast.Name):
                    scope.bind(name.id, None)
            self.visit(node.iter, scope, loops)
            for item in node.body:
                self.visit(item, scope, (*loops, node))
            for item in node.orelse:
                self.visit(item, scope, loops)
            return
        if isinstance(node, ast.Lambda):
            return  # no production writer uses a lambda as its naming scope
        if isinstance(node, ast.ImportFrom):
            prefix = scope.module.split(".")[:-node.level] if node.level else []
            module = ".".join([*prefix, *([node.module] if node.module else [])])
            for a in node.names:
                scope.bind_name(a.asname or a.name, f"{module}.{a.name}")
        if isinstance(node, ast.Import):
            for a in node.names:
                name = a.asname or a.name.split(".")[0]
                scope.bind_name(name, a.name if a.asname else name)
                scope.modules.add(name)
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                pairs = _target_pairs(target, node.value)
                for name in ast.walk(target):  # an unpaired name (``PREFIX, _ = pair``) is unknown
                    if isinstance(name, ast.Name) and isinstance(name.ctx, ast.Store):
                        scope.bind(name.id, pairs.get(name.id))
                if isinstance(target, ast.Subscript) and isinstance(target.value, ast.Name):
                    key = target.slice.value if isinstance(target.slice, ast.Constant) else None
                    scope.dict_edits.setdefault(target.value.id, set()).add(key)
        # Other rebinding (``+=``, ``:=``, comprehension, ``with``/``except as``) makes the name unknown.
        rebound = (node.target if isinstance(node, (ast.AugAssign, ast.NamedExpr, ast.comprehension))
                   else node.optional_vars if isinstance(node, ast.withitem) else None)
        for name in ast.walk(rebound) if rebound is not None else ():
            if isinstance(name, ast.Name) and isinstance(name.ctx, ast.Store):
                scope.bind(name.id, None)
        if isinstance(node, ast.ExceptHandler) and node.name:
            scope.bind(node.name, None)
        if isinstance(node, ast.Return) and node.value is not None:
            scope.returns.append(node.value)
        if isinstance(node, ast.Call):
            self.calls.append(Site(scope, node, loops))
            func = node.func
            if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
                keys = None
                if func.attr == "update":
                    keys = {k.arg for k in node.keywords} if not node.args else {None}
                elif func.attr in {"pop", "setdefault", "clear", "popitem"}:
                    first = node.args[0] if node.args else None
                    keys = {first.value if isinstance(first, ast.Constant) else None}
                if keys is not None:
                    scope.dict_edits.setdefault(func.value.id, set()).update(keys)
            self.visit(func, scope, loops, is_callee=True)
            for item in (*node.args, *(k.value for k in node.keywords)):
                self.visit(item, scope, loops)
            return
        if not is_callee and isinstance(node, (ast.Name, ast.Attribute)) and isinstance(node.ctx, ast.Load):
            self.references.append(Site(scope, node, loops))
        for item in ast.iter_child_nodes(node):
            self.visit(item, scope, loops)

    def value(self, site, node, local, depth=0):
        if depth > 12:
            raise ValueError("source name is not a bounded expression")

        def get(item):
            return self.value(site, item, local, depth + 1)

        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return {node.value}
        if isinstance(node, ast.Name):
            if node.id in local:
                return get(local[node.id])
            owner = site.scope.defining(node.id)
            if owner is not None:
                bindings = owner.assigned.get(node.id)
                if bindings and len(bindings) == 1 and node.id not in owner.names:
                    # ``record_id = record_id or new_record_id()`` reads the
                    # earlier parameter, not the local being assigned.
                    return self.value(Site(owner, site.node, ()), bindings[0], {**local, node.id: None}, depth + 1)
                if not bindings and node.id in owner.parameters and owner.key in _RELAYS | _VALUE_HELPERS:
                    return {f"\0{node.id}\0"}
                if site.scope.settled(node) and node.id not in owner.modules:  # a ``from``-imported string constant
                    module, _, attr = owner.names[node.id].rpartition(".")
                    mod = self.scopes.get(module)
                    if mod is not None and mod.parent is None and attr not in mod.names and "*" not in mod.names:
                        assigned = mod.assigned.get(attr)
                        if (assigned and len(assigned) == 1
                                and isinstance(assigned[0], ast.Constant) and isinstance(assigned[0].value, str)):
                            return {assigned[0].value}
        if (isinstance(node, ast.Attribute) and node.attr == "name"
                and isinstance(node.value, ast.Call) and len(node.value.args) == 1 and not node.value.keywords
                and site.scope.callee(node.value.func) == "pathlib.PurePosixPath"
                and site.scope.settled(node.value.func)):
            parts = get(node.value.args[0])
            if parts and all(v == _literal(v) for v in parts):
                names = {PurePosixPath(v).name for v in parts}
                if all(n not in {"", ".", ".."} for n in names):
                    return names
            return {_HOLE}
        if isinstance(node, ast.JoinedStr):
            return _concat(get(v) for v in node.values)
        if isinstance(node, ast.FormattedValue) and node.format_spec is None and node.conversion in (-1, ord("s")):
            return get(node.value)
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
            return _concat((get(node.left), get(node.right)))
        if isinstance(node, ast.IfExp):
            if (guard := self.guard(site, node.test)) is None:
                return get(node.body) | get(node.orelse)
            name, yes, no = guard
            return ({f"\x02{name}:{yes}\x02{v}" for v in get(node.body)}
                    | {f"\x02{name}:{no}\x02{v}" for v in get(node.orelse)})
        if isinstance(node, ast.BoolOp) and isinstance(node.op, ast.Or):
            return (set().union(*(get(v) for v in node.values)) - {""}) or {""}
        if isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Slice):
            values = get(node.value)
            if node.slice.lower is None and node.slice.step is None and isinstance(node.slice.upper, ast.Constant):
                end = node.slice.upper.value
                if isinstance(end, int) and end >= 0:
                    return {v[:end] if v == _literal(v) else v for v in values}
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id == "str" and len(node.args) == 1:
                return get(node.args[0])
            key = site.scope.callee(node.func)
            if key in _VALUE_HELPERS and key in self.scopes:
                owner = self.scopes[key]
                if len(owner.returns) == 1:
                    parts = self.value(Site(owner, owner.returns[0], ()), owner.returns[0], {}, depth + 1)
                    bound = self.arguments(Site(site.scope, node, site.loops), owner,
                                           {name for part in parts for name in _PARAM.findall(part)})
                    return set().union(*(self.substitute(p, self.evaluate(site, bound, local, depth + 1)) for p in parts))
        return {_HOLE}

    @staticmethod
    def guard(site, test):
        """``if part`` or ``isinstance(payload, str)`` on a relay's own parameter."""
        if isinstance(test, ast.Name):
            name, outcomes = test.id, ("set", "unset")
        elif (isinstance(test, ast.Call) and isinstance(test.func, ast.Name) and test.func.id == "isinstance"
              and len(test.args) == 2 and isinstance(test.args[0], ast.Name)
              and isinstance(test.args[1], ast.Name) and test.args[1].id == "str"):
            name, outcomes = test.args[0].id, ("str", "other")
        else:
            return None
        owner = site.scope.defining(name)
        if owner is None or owner.key not in _RELAYS or name in owner.assigned or name not in owner.parameters:
            return None
        return (name, *outcomes)

    @staticmethod
    def payload_type(site, node):
        """What the caller visibly passes: text, a container, or unknown (None)."""
        node = site.expression(node) if node is not None else None
        if isinstance(node, ast.JoinedStr) or isinstance(node, ast.Constant) and isinstance(node.value, str):
            return "str"
        if isinstance(node, (ast.Dict, ast.List, ast.Tuple, ast.Set, ast.ListComp, ast.DictComp, ast.SetComp)):
            return "other"
        return None

    def arguments(self, site, target, wanted):
        call = site.node
        actual = {k.arg: k.value for k in call.keywords if k.arg in wanted}
        if target and target.fn:
            args = target.fn.args
            positional = [a.arg for a in (*args.posonlyargs, *args.args)]
            defaults = dict(zip(positional[len(positional) - len(args.defaults):], args.defaults))
            defaults.update(zip((a.arg for a in args.kwonlyargs), args.kw_defaults))
            if target.cls:
                positional = positional[1:]
            actual.update({name: arg for name, arg in zip(positional, call.args) if name in wanted})
        else:
            defaults = {}
        for kw in call.keywords:
            if kw.arg is None:
                expr = site.expression(kw.value)
                if isinstance(expr, ast.Call) and isinstance(expr.func, ast.Name) and expr.func.id == "dict":
                    fields = {k.arg: k.value for k in expr.keywords}
                else:
                    fields = {}
                owner = site.scope.defining(kw.value.id) if isinstance(kw.value, ast.Name) else None
                edited = owner.dict_edits.get(kw.value.id, set()) if owner else set()
                actual.update({k: fields.get(k) if k not in edited and None not in edited else None
                               for k in wanted - actual.keys()})
        return {name: (actual[name], site) if name in actual else (defaults.get(name),
                Site(target.parent, call, ()) if target and target.parent else site) for name in wanted}

    def evaluate(self, site, bound, local, depth=0):
        return {k: self.value(s, v, local if s is site else {}, depth + 1) for k, (v, s) in bound.items()}

    @staticmethod
    def substitute(text, values, types=None):
        for name, outcome in _GUARD.findall(text):
            marker, given = f"\x02{name}:{outcome}\x02", values.get(name, {_HOLE})
            relayed = _PARAM.fullmatch(next(iter(given))) if len(given) == 1 else None
            if relayed:  # passed on by the caller's own parameter: its caller decides
                text = text.replace(marker, f"\x02{relayed[1]}:{outcome}\x02")
                continue
            if outcome in ("str", "other"):
                kind = (types or {}).get(name)
                if kind is None:
                    return {_HOLE}  # an unseen payload type leaves the extension opaque
                taken = kind == outcome
            else:
                truth = False if given == {""} else True if all(map(_literal, given)) else None
                taken = truth is None or truth == (outcome == "set")
            if not taken:
                return set()  # this caller can never produce the branch
            text = text.replace(marker, "")
        return _concat({part} if i % 2 == 0 else values.get(part, {_HOLE})
                       for i, part in enumerate(_PARAM.split(text)))

    def names_at(self, site, target, templates):
        wanted = {name for template in templates for part in template for name in _names(part)}
        bound = self.arguments(site, self.scopes.get(target), wanted)
        types = {name: self.payload_type(s, v) for name, (v, s) in bound.items()}
        locals_used = {n.id for value, scope in bound.values() if value is not None and scope is site
                       for n in ast.walk(value) if isinstance(n, ast.Name)}
        for local in site.environments(locals_used):
            values = self.evaluate(site, bound, local)
            for template in sorted(templates):
                yield from itertools.product(*(self.substitute(part, values, types) for part in template))

    def finish(self):
        seed = {("\0category\0", "\0source_id\0", "\0extension\0")}
        for name in _SINKS:
            self.recipes[name] = seed
        candidates = [(site, site.scope.callee(site.node.func)) for site in self.calls]
        relevant = [(site, key) for site, key in candidates if key in _CALLS]
        for _ in range(len(_RELAYS) + 1):
            size = sum(map(len, self.recipes.values()))
            for site, key in relevant:
                if site.scope.key in _RELAYS and key in self.recipes:
                    self.recipes.setdefault(site.scope.key, set()).update(self.names_at(site, key, self.recipes[key]))
            if size == sum(map(len, self.recipes.values())):
                break
        else:
            raise AssertionError("known source naming relay recipes did not settle")
        for site, key in relevant:
            owner = site.scope.key
            if key == _ADDRESS and owner != _ADDRESS_COPY:
                if owner not in _PRICE_ONLY | {_WRITE, _STREAM}:
                    self.findings.add(f"{site.address}: classify the address-only call as a copy or preview")
                continue
            if owner in _RELAYS:
                continue
            try:
                if key not in self.recipes:
                    raise ValueError(f"missing naming recipe for {key}")
                for folder, name, extension in self.names_at(site, key, self.recipes[key]):
                    self.record(site, folder, name, extension)
            except ValueError as exc:
                self.findings.add(f"{site.address}: {exc}")
        for site in self.references:
            if site.scope.callee(site.node) in _CALLS:
                self.findings.add(f"{site.address}: writer passed as a value; review its naming path")
        for site, key in candidates:
            call = site.node
            if isinstance(call.func, ast.Name) and call.func.id == "getattr" and len(call.args) >= 2:
                obj, name = call.args[:2]
                if isinstance(name, ast.Constant) and f"{site.scope.callee(obj)}.{name.value}" in _CALLS:
                    self.findings.add(f"{site.address}: writer looked up dynamically; review its naming path")
        self.bind_session_names(candidates)

    def bind_session_names(self, candidates):
        """One dictionary handoff: each store a caller passes names the file; none, no write."""
        own = {write for write in self.writes if write[2] == _SESSION_WRITER}
        opaque = ("context_checkpoints", "*-*.json", _SESSION_WRITER)
        if opaque not in own:
            return
        self.writes -= own
        for site, key in candidates:
            if key != _SESSION_CALL:
                continue
            store = next((kw.value for kw in site.node.keywords if kw.arg == "store"), None)
            if store is None or isinstance(store, ast.Constant) and store.value is None:
                continue  # _stored_read_history returns before writing
            if not isinstance(store, ast.Dict) or None in store.keys:
                self.findings.add(f"{site.address}: cannot see the review session store's source_id")
                continue
            fields = {k.value: v for k, v in zip(store.keys, store.values) if isinstance(k, ast.Constant)}
            root, label = fields.get("root"), fields.get("source_id")
            if root is None or isinstance(root, ast.Constant) and root.value is None:
                continue  # no root: the writer returns before writing
            falsy = label is None or isinstance(label, ast.Constant) and not label.value
            given = set() if falsy else self.value(site, label, {})
            names = given - {""}
            self.writes.update((opaque[0], leaf, opaque[2]) for name in names for leaf in file_shapes(name, "json"))
            if not names or "" in given or not all(map(_literal, names)):
                self.writes |= own - {opaque}  # the writer's own fallback, used for an empty id

    def record(self, site, folder, name, extension):
        producer = site.scope.key
        if producer in _COPY_ONLY and (folder, name, extension) == (_HOLE, _HOLE, _HOLE):
            return  # reuse the already-verified address; a new named write would still be tested
        if folder.strip() not in self.categories:
            raise ValueError(f"unknown source folder {folder!r}")
        if _GUARD.search(name + extension):
            raise ValueError("a relay branch reached no deciding caller")
        name, extension = (_PARAM.sub(_HOLE, part) for part in (name, extension))
        for leaf in file_shapes(name, extension):
            self.writes.add((folder.strip(), leaf, producer))


def file_shapes(name, extension):
    """Literal filename pieces plus holes, not a representative filename."""
    pieces = [re.sub(r"[^A-Za-z0-9_.-]+", "_", part) for part in name.split(_HOLE)]
    pieces[0] = pieces[0].lstrip("._")
    pieces[-1] = pieces[-1].rstrip("._")
    template = "*".join(pieces)
    suffix = "*" if _HOLE in extension else (re.sub(r"[^A-Za-z0-9]+", "", extension)[:12] or "bin")
    if "*" not in template:
        return {f"{template[:160] or 'source'}-*.{suffix}"}
    return {f"{template}-*.{suffix}"}


@dataclass(frozen=True)
class ScanResult:
    categories: frozenset[str]
    writes: frozenset[tuple[str, str, str]]
    findings: frozenset[str]


@lru_cache(maxsize=2)
def source_scan(root: Path, categories: frozenset[str]) -> ScanResult:
    """Only the results are kept; the parsed trees and call sites go with the scan."""
    scan = SourceScan(root, categories)
    return ScanResult(categories, frozenset(scan.writes), frozenset(scan.findings))


STORE = "task_results/artifacts/*/source_handles"
_STORE_TOKEN = re.compile(r"task_results/artifacts/(?:<[^/>]+>|\*)/source_handles(?:/(.*))?$")
_PLACEHOLDER = re.compile(r"<[^/>]+>")
_BRACES = re.compile(r"\{([^{}]+)\}")
_CODE = re.compile(r"`([^`]+)`")
_BARE_LEAF = re.compile(r"[^/\s]+\.(?:\w+|\*|<ext>|\{[\w,]+\})$")
_DIGEST_LEAF = re.compile(r"[^/\s]+-<sha256>\.(?:\w+|<ext>)")
# A Writer cell's declared list, ``writes <what>: `a`, `b` and `c```, up to ``;`` or the sentence's end.
_WRITES = re.compile(r"\bwrites [^:;`]*:((?:`[^`]*`|[^;`.]|\.(?=\S))*)")

# A fully dynamic first component needs its identity as well as its shape:
# the cut's digest row is not the compactor's fingerprint or the author's receipt.
_TYPED_IDS = {
    "ouroboros.context_renewal.physical_window_cut": ("context_checkpoints", "*-*.json", "digest"),
    "ouroboros.context_renewal.renewal_view": ("context_checkpoints", "*-*.json", "digest"),
    "ouroboros.context_compaction._persist_reclaim_checkpoint": ("context_checkpoints", "*-*.json", "fingerprint"),
    "ouroboros.loop_round_limits._run_authored_context_view": ("context_checkpoints", "*-*.json", "receipt"),
    "ouroboros.artifacts.materialize_tool_result_source": ("tool_results", "*-*.txt", "call"),
    "ouroboros.tool_result_delivery._accepted_prefix": ("tool_results", "*-*.txt", "call"),
    "ouroboros.tool_result_delivery._measured_frame": ("tool_results", "*-*.txt", "call"),
    "ouroboros.tool_result_delivery._measured_frame.render": ("tool_results", "*-*.txt", "call"),
}

# No blanket prefix equivalence: only these existing alternate names can use
# the primary kind's row. Either one can instead have a separately named row.
_ROW_VARIANTS = {
    ("context_checkpoints", "budget-pause-sleep-*-*.json", "supervisor.restart_retention.retain_sleep_checkpoint"):
        "budget-pause-*-*.json",
    ("context_checkpoints", "budget-pause-owner-pause-*-*.json", "supervisor.restart_retention.retain_owner_pause_park"):
        "budget-pause-*-*.json",
    ("tool_results", "task-wait-owed-*.txt", "ouroboros.native_task_wait.deliver_owed.step"):
        "task-wait-*-*.txt",
    ("tool_results", "run-wait-owed-*.txt", "ouroboros.native_task_wait.deliver_owed.step"):
        "run-wait-*-*.txt",
}

# Two older rows name only the pre-digest id. Other shorthand has no standing.
_ROW_SHORTHAND = {
    ("tool_results", "result-material-<sha16>"): "result-material-*-*.*",
    ("tool_results", "delegated-capture-<job>-<run>"): "delegated-capture-*-*-*.json",
}
# The unqualified capture label is real only in its own prose-led Path cell.
_CAPTURE_LABEL = "delegated-capture-<job>-<run>"


def _expand(shape):
    braces = _BRACES.search(shape)
    if not braces:
        return (shape,)
    return tuple(result for choice in braces[1].split(",")
                 for result in _expand(shape[:braces.start()] + choice.strip() + shape[braces.end():]))


def _row_key(folder, leaf):
    shape = _ROW_SHORTHAND.get((folder, leaf), _PLACEHOLDER.sub("*", leaf))
    if shape in {"*-*.json", "*-*.txt"}:
        # Literal/star aliases for tagged identities would merge unrelated writers.
        match = _PLACEHOLDER.match(leaf)
        shape = (match[0] if match else "<?>") + shape[1:]
    return f"{folder}/{shape}" if folder else shape


def store_rows(text, categories):
    """Explicit store paths from valid Retention rows, split into containers and named files."""
    containers, named = set(), set()
    for cell, retention, writer in inventory_rows(text, whole=True):
        if lifetime(retention) is None:
            continue
        spans = _CODE.findall(cell)
        located = [(token, match[1] or "") for token in spans if (match := _STORE_TOKEN.fullmatch(token.rstrip("/")))]
        if spans and spans[0] == _CAPTURE_LABEL and not located:
            named.add(_row_key("tool_results", _CAPTURE_LABEL))
        for _, relative in located:
            for item in _expand(relative):
                folder, _, leaf = item.partition("/")
                if folder in categories and leaf:
                    named.add(_row_key(folder, leaf))
                elif item and folder not in {"*", "**"}:
                    containers.add(_PLACEHOLDER.sub("*", item))
        # A plain leaf is in this namespace only beside an explicit container.
        folders = {relative for _, relative in located if relative in categories}
        if len(folders) == 1:
            folder = next(iter(folders))
            leaves = [leaf for leaf in spans[1:] if _BARE_LEAF.fullmatch(leaf) or (folder, leaf) in _ROW_SHORTHAND]
            if len(spans) == 1:  # a bare container's Writer may declare its files, in digest form; examples don't
                leaves = [leaf for declared in _WRITES.findall(writer)
                          for leaf in _CODE.findall(declared) if _DIGEST_LEAF.fullmatch(leaf)]
            named.update(_row_key(folder, expanded) for leaf in leaves for expanded in _expand(leaf))
    return containers, named


def file_kinds(writes):
    """Observed kind -> producers; do not infer a known extension from an opaque one."""
    kinds = {}
    for folder, leaf, producer in sorted(writes):
        stem = leaf.rsplit(".", 1)[0]
        if not leaf.endswith(".*") and (folder, stem + ".*", producer) in writes:
            continue  # same writer has a genuinely variable extension, also covering its fallback
        tagged = _TYPED_IDS.get(producer)
        if tagged and tagged[:2] == (folder, leaf):
            leaf = f"<{tagged[2]}>" + leaf[1:]
        kinds.setdefault(f"{folder}/{leaf}", set()).add(producer)
    return kinds


def match_rows(named, kinds):
    """One matching rule for the forward debt check and backward stale rows."""
    backed, missing = set(), {}
    for key, producers in kinds.items():
        folder, leaf = key.split("/", 1)
        for producer in sorted(producers):
            alias = _ROW_VARIANTS.get((folder, leaf, producer))
            matches = named & ({key, f"{folder}/{alias}"} if alias else {key})
            backed.update(matches)
            if not matches:
                missing.setdefault(key, set()).add(producer)
    return backed, missing


def debt_findings(missing, debt):
    """No new member, no stale listed member. The reviewer owns historical shrink-only."""
    actual = {(kind, producer) for kind, writers in missing.items() for producer in writers}
    return [*(f"give {kind} a matching Retention row (writer {producer})" for kind, producer in sorted(actual - debt)),
            *(f"drop or update paid/retired debt {kind} (writer {producer})" for kind, producer in sorted(debt - actual))]


# These are existing documentation debts, not permission to add new ones. Each
# named shape and producer must leave once it has a row or stops writing. Keep
# the suffix and extension: another shape from the same function isn't owed.
_NATIVE = "review_native_episode.NativeToolRoundReviewExecutor."


def _debt(folder, rows):
    return {(f"{folder}/{name}", producer if producer.startswith("supervisor.") else f"ouroboros.{producer}")
            for shape, writers in rows.items() for name in _expand(shape) for producer in writers.split()}


UNROWED_SOURCE_KINDS = frozenset().union(
    _debt("context_checkpoints", {
        "*-history-*.json": _NATIVE + "_episode_source_facts",
        "*-{initial-brief,required-sources}-*.*": _NATIVE + "_open_episode",
        "*-round-*-*.*": _NATIVE + "_record_round_source",
        "*-session-reads-*.json": "review_session_reads._stored_read_history",
        "*-view-*-*.*": _NATIVE + "_apply_working_view",
        "<fingerprint>-*.json": "context_compaction._persist_reclaim_checkpoint",
        "<receipt>-*.json": "loop_round_limits._run_authored_context_view",
        "attachments-*.json": "artifacts.attachment_manifest_projection",
        "completion-*.json": "task_finalization.build_completion_observations",
        "host-*-*.md": "section_floor.retain_section",
        "host-state-*-*.json": "context_source_view.retire_superseded_at_departure",
        "late-result{,-a*}-*.json": "history_retention.retain_late_result",
        "left-review-*-*-*.json": "review_pause._left_review_mark",
        "legacy_{blocks,meta}-*.json": "chronicle_import._import",
        "legacy_flat-*.md": "chronicle_import._import",
        "memory_fallback_response-*.md": "memory_fallback._refuse",
        "plan-dialogue-*-*.jsonl": "tools.plan_dialogue.attach_own_dialogue",
        "plan-evidence-*.txt": "tools.plan_evidence.own_source_retention.retain",
        "plan-revision-*-*.json": "plan_record.retain_revision",
        "review-complete-brief-*.txt": "tools.review_brief_coupling.build_retrieving_brief",
        "review-ledger-*-*-{failure,pending,response}{,-*}-*.txt":
            "review_ledger.build_wave_record review_ledger.settle_from_collected._mutate",
        "review-ledger-*-*-redacted_run_answer-*.txt": "review_history.retain_original_run_answers",
        "review-ledger-*-*-response-intention-*.txt": "tools.plan_ask.record_reply",
        "review-ledger-*-*-response-report-*.txt": "tools.review_change._as_report",
        "review-ledger-*-author-author_decision-*.txt": "review_history.retain_author_decision",
        "review-ledger-{*,preparation}-author-rebuttal-*.txt": "review_history.prepare_history",
        "review-ledger-*-panel-failure-*.txt": "tools.review_change.wave_failure_note",
        "review-ledger-{*,rl-*-*}-panel-prompt{,-*}-*.txt": "review_ledger.build_wave_record",
        "review-ledger-rl-*-*-*-{failure,pending,response}{,-*}-*.txt": "review_ledger.build_wave_record",
        "review-ledger-rl-*-*-*-{prompt,response}-intention-*.txt": "tools.plan_ask.dispatch_ask",
        "review-preimage-*-*.txt": "tools.review_brief_coupling.build_retrieving_brief",
        "review-staged-diff-*.txt": "tools.review_brief_coupling.build_retrieving_brief",
        "review-turn-*-*.txt": "review_session_preparation.prepare_review_session_request",
        "task_input-*.json": "context_input_selection.task_input_pointer",
        "terminal-delivery-*.json": "supervisor.terminal_delivery._capture_emitted_answer",
        "view-*.json": "tools.compact_context._inspect",
        "wake_observation-*.jsonl":
            "consciousness_transitions.export_for_older_body consciousness_wake.bind_wake_observation",
        "working-a*-s*-*.json": "working_checkpoint.prepare_recovery",
        "working-notes-*-*.json": "main_context_authority.publish_working_notes",
    }),
    _debt("delegated_activity", {
        "*-*-*-*.jsonl": "delegate_activity._persist",
    }),
    _debt("tool_results", {
        "*-tool-*-*-*.*": _NATIVE + "_execute_inspection_call",
        "*.producer-*.txt": "tool_result_delivery.prepare_producer_sources",
        "<call>-*.txt": "artifacts.materialize_tool_result_source tool_result_delivery._accepted_prefix "
                        "tool_result_delivery._measured_frame tool_result_delivery._measured_frame.render",
        "browser_upload-*.*": "mcp_browser_guard.BrowserGuard._file_intake",
        "commit_review-*.txt": "review_evidence.capture_commit_review_evidence",
        "delegate-wake-*-*.json": "delegate_supervision._render_wake_payload",
        "delegated-{interactions,terminal}-*-*.json": "delegate_collection.observe_run",
        "gh-artifact-*-*-*-*.zip": "tools.github_workflows.download",
        "gh-run-*-artifacts-page-*-*.json": "tools.github_workflows.list_artifacts",
        "gh-run-*-attempt-*{-job-*,}{-failed,}-log-*.log": "tools.github_workflows.get_run",
        "knowledge-previous-*.txt": "tools.knowledge._capture_previous",
        "owner-dialogue-*.txt": "tools.owner_delivery._retain_dialogue",
        "{run,task}-wait-{*,owed}-*.txt": "native_task_wait.deliver_owed.step",
        "service-log-*-*.txt": "tools.services.retained_log_display",
        "verify_and_record-output-*.txt": "tools.process_facts.check_output_views",
    }),
)
