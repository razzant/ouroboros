#!/usr/bin/env python3
"""Census of the default test lane: which tests each retirement rule names.

DEVELOPMENT "Retiring and merging tests": a class of tests retires by a re-runnable rule plus a
sampled independent read. This script is that rule. It claims every test function for the first
rule that applies, so nothing counts twice, and writes one row per function. It lists candidates
and decides nothing: a reader overturns any row, and `devtools/coverage_census.py` returns the
rows whose removal would lose a product line.

    python -m devtools.test_census --out DIR [--root .] [--protected tests/protected_tests.json]
        [--journal events.jsonl ...] [--collected ids.txt] [--base REF]

--journal    `tests/ci_evidence.py` events of a complete run: seconds and, without --collected,
             items. An earlier journal wins a function a later one also records.
--collected  `pytest --collect-only -q` output for the default lane: the exact lane and items.
--base       the commit the change under review started from. D6 calls a machinery test obsolete only
             when every product function it names was rewritten since then; without --base it claims none.
Without either, items come from literal parametrize lists and lane markers read from the source.
A function no journal records weighs zero seconds, so D3 cannot tell it from a cheap list.

Writes DIR/decisions.csv (one row per default-lane function: cluster, rule, verdict) and
DIR/totals.json (one block per rule, plus the deleted and folded numbers reported separately).
The same tree and inputs give the same bytes, so a rerun after a target merge regenerates the
candidates; output stays outside the checkout (devtools/README.md).
"""
from __future__ import annotations

import argparse
import ast
import csv
import json
import math
import re
import subprocess
from collections import Counter, defaultdict
from pathlib import Path

from ouroboros.preflight_runner import LANE_EXCLUSION_EXPR

LANE_MARKERS = {term.removeprefix("not ") for term in LANE_EXCLUSION_EXPR.split(" and ")}
PRODUCT_ROOTS = {"ouroboros", "supervisor", "server", "launcher", "devtools", "skills", "scripts", "web",
                 "android", "packaging"}
BUILTIN_FIXTURES = {"tmp_path", "tmp_path_factory", "tmpdir", "tmpdir_factory", "monkeypatch", "capsys", "capfd",
                    "capsysbinary", "capfdbinary", "caplog", "request", "recwarn", "pytestconfig",
                    "record_property", "record_testsuite_property", "record_xml_attribute", "cache",
                    "doctest_namespace", "testdir", "pytester", "subtests", "self", "cls", "event_loop",
                    "anyio_backend", "mocker", "worker_id"}
WORLD_BLIND_FIXTURES = {"monkeypatch", "capsys", "caplog", "request", "capfd", "recwarn"}
SOURCE_HANDLES = {"getsource", "getsourcefile", "getfile", "getsourcelines", "getmodule"}
ITEM_FLOOR_S = 0.041  # median setup+teardown of a test whose call took under 10 ms
CHEAP_CASE_S = 0.2  # D3: a folded list must not become one item past the 300 s hang bound

# What the body of a test does (shape only: a reader confirms what it means).
SPAWN = re.compile(r"\bsubprocess\.\w+\(|\bPopen\(|\bsys\.executable\b|create_subprocess_|\bos\.fork\(|"
                   r"multiprocessing|ProcessPoolExecutor|\bos\.system\(")
GIT = re.compile(r"""["']git["']|\brun_git\(|\b_git\(|\bgit_init\b|init_repo\(|init_installed_body|_init_git""")
NESTED_PYTEST = re.compile(r"""["']-m["']\s*,\s*["']pytest["']|\bpytest\.main\(|run_hermetic_pytest""")
ISSUE = re.compile(r"(?:razzant/ouroboros#|ouroboros#|(?<![\w/#&])#)(\d{3,4})\b")
READS = re.compile(r"\.read_text\(|\bopen\(|getsource\(|\.read_bytes\(|read_lines\(|\.splitlines\(")
PROSE = re.compile(r"BIBLE|README|ARCHITECTURE|DEVELOPMENT|CHECKLISTS|CONTRIBUTING|SYSTEM\.md|SAFETY\.md|"
                   r"CONSCIOUSNESS|prompts|[\"']docs[\"/]|docs/|\.md\b|CHANGELOG|PERSISTENCE|DESIGN")
WEB_TEXT = re.compile(r"[\"']web[\"/]|web/|\.js\b|\.css\b|\.html\b|\.mjs\b")
DYNAMIC_PRODUCT = re.compile(r"^(ouroboros|supervisor|server|launcher|devtools|skills|scripts)(\.[a-z_]\w*)+$|"
                             r"^(server|launcher)\.py$|^(scripts|devtools|ouroboros|supervisor)/[\w/]+\.py$")

# What a test file is named after or imports.
MACHINERY_IMPORT = re.compile(
    r"^(ouroboros\.(preflight_\w+|safe_test_supervisor|test_environment|test_evidence|test_parallelism|"
    r"test_selection|size_ratchet_manifest|tools\.skill_preflight|tools\.preflight_review)|"
    r"scripts\.(run_tests|safe_test|ci_retry|browser_ci_scope|cleanup_test_pollution|regenerate_size_ratchet|"
    r"release_proof)|tests\.(ci_evidence|ci_verdicts|ci_accepted|default_lane|browser_lane|group_run|conftest))$")
MACHINERY_FILE = re.compile(
    r"test_(preflight|ci_|safe_test|run_tests|test_selection|test_parallelism|test_environment|test_evidence|"
    r"default_lane|default_suite|browser_lane|hermetic_pytest|serial_|shard|battery|accepted_fail|ci_verdict|"
    r"ci_evidence|retention|pytest|conftest|tests_affected|post_merge_workflow|commit_gate|test_proof|"
    r"focused_test|selected_test|full_test|evidence_root|scratch_teardown)")
EVENT_FILE = re.compile(
    r"batch\d|_r\d+(?:_|\.py)|_v\d{2,}|issue_?\d+|_\d{3,5}(?:_|\.py)|follow_?up|_fix(?:es)?(?:_|\.py)|"
    r"round\d|_wave|_w\d+(?:_|\.py|[ab]\.py)|sprint|hotfix|_regression|tier\d|_phase\d|_p\d+(?:_|\.py)|"
    r"_pr\d+|_t\d+(?:_|\.py)|_q\d+(?:_|\.py)|_d\d+(?:_|\.py)|_f\d+(?:_|\.py)|_s\d+(?:_|\.py)|"
    r"_h\d+(?:_|\.py)|_audit|_rerun|_hardening|_closure|_residual|_lessons|_review_r\d|_c\d+(?:_|\.py)|"
    r"_g\d+(?:_|\.py)|_b\d+(?:_|\.py)|_m\d+(?:_|\.py)|_a\d+(?:_|\.py)|_e\d+(?:_|\.py)|_k\d+(?:_|\.py)")
COST_STAND_FILE = re.compile(r"test_cost_(contract|live_set)")  # Q7=A: built once, its axes stay whole
SECRET_WORD = re.compile(r"secret|credential|token|redact|private|leak|password|key", re.I)

RULES = {  # cluster -> (the rule an agent can apply, the verdict of DEVELOPMENT "Retiring and merging tests")
    "K_protected": ("a file of the protected set (tests/protected_tests.json)", "keep"),
    "P_postmerge_protected": ("a function named in a post-merge-red, FLAKY or ACCEPTED record keeps all its cases",
                              "keep"),
    "D4_wording": ("reaches no product behaviour and reads prose or UI text, or every assert is a substring of "
                   "that text; never a spawning, machinery, secret or incident test",
                   "delete: review owns (CHECKLISTS 5, 11(d), 26)"),
    "D6_machinery": ("spawning, nested-pytest or git test of the runner, gate, evidence or scratch machinery "
                     "whose every named product function was rewritten since the base, beyond the cheapest "
                     "witness per file; never a secret test", "delete: obsolete (machinery rewritten)"),
    "D5_event_file": ("file named after a delivery event whose product modules are each reached by at least "
                      "three unit-file tests; never a secret test", "merge into the unit file, or delete: obsolete"),
    "D1_cartesian": ("at least two literal parametrize axes: every value of every axis once, unless the "
                     "docstring names an interaction", "merge into the surviving cases"),
    "D2_cases": ("a hand-written case list, or a heavy computed one: its distinct classes, at most three "
                 "(boundary, negative, incident)", "merge into the surviving cases"),
    "D3_enumeration": ("a cheap computed case list: one invariant test listing every violator",
                       "merge into one invariant test"),
    "D7_facets": ("tests of one file building the same world: one scenario test with subtests",
                  "merge into the scenario test"),
    "D7_survivor": ("the scenario test its same-world group folds into", "keep"),
    "": ("no rule names this test", "keep"),
}
DELETED = ("D4_wording", "D6_machinery", "D5_event_file", "D1_cartesian", "D2_cases")  # assertions removed
FOLDED = ("D3_enumeration", "D7_facets")  # every assertion kept
FIELDS = ("file", "qual", "line", "items", "seconds", "setup_s", "cluster", "rule", "verdict", "items_removed",
          "seconds_saved_lo", "seconds_saved_hi", "survivor", "cited", "exercise", "reads")


def _dotted(node) -> str:
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
    elif isinstance(node, ast.Call):
        inner = _dotted(node.func)
        if inner:
            parts.append(inner + "()")
    return ".".join(reversed(parts))


def _decorated_name(decorator) -> str:
    return _dotted(decorator.func if isinstance(decorator, ast.Call) else decorator)


def _is_fixture(node) -> bool:
    return any(_decorated_name(decorator).endswith("fixture") for decorator in getattr(node, "decorator_list", []))


def _axes(decorators) -> list:
    """The case count of each parametrize decorator, None when its list is computed."""
    sizes = []
    for decorator in decorators:
        if isinstance(decorator, ast.Call) and _dotted(decorator.func).endswith("parametrize"):
            values = decorator.args[1] if len(decorator.args) > 1 else next(
                (keyword.value for keyword in decorator.keywords if keyword.arg == "argvalues"), None)
            if isinstance(values, (ast.List, ast.Tuple, ast.Set)):
                sizes.append(len(values.elts))
            else:
                sizes.append(len(values.keys) if isinstance(values, ast.Dict) else None)
    return sizes


def _parametrized_names(decorators) -> set:
    names = set()
    for decorator in decorators:
        if isinstance(decorator, ast.Call) and _dotted(decorator.func).endswith("parametrize") and decorator.args:
            first = decorator.args[0]
            if isinstance(first, ast.Constant) and isinstance(first.value, str):
                names |= {name.strip() for name in first.value.split(",") if name.strip()}
            elif isinstance(first, (ast.List, ast.Tuple)):
                names |= {item.value for item in first.elts if isinstance(item, ast.Constant)}
    return names


def _marks(decorators) -> set:
    return {name.rsplit(".", 1)[-1] for name in map(_decorated_name, decorators) if ".mark." in f".{name}"}


def _pytestmark(body) -> set:
    marks = set()
    for node in body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "pytestmark" for t in node.targets):
            values = node.value.elts if isinstance(node.value, (ast.List, ast.Tuple)) else [node.value]
            marks |= _marks(values)
    return marks


def _test_functions(tree):
    """(qualname, node, inherited and own parametrize axes, marks) of every statically visible test."""
    def walk(body, cls, axes, marks):
        for node in body:
            if isinstance(node, ast.ClassDef) and node.name.startswith("Test"):
                inner = f"{cls}::{node.name}" if cls else node.name
                yield from walk(node.body, inner, axes + _axes(node.decorator_list),
                                marks | _marks(node.decorator_list) | _pytestmark(node.body))
            elif (isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith("test_")
                  and not _is_fixture(node)):
                yield (f"{cls}::{node.name}" if cls else node.name, node, axes + _axes(node.decorator_list),
                       marks | _marks(node.decorator_list))
    yield from walk(tree.body, "", [], _pytestmark(tree.body))


def _classify(source: str) -> str:
    root = source.split(".", 1)[0]
    return "product" if root in PRODUCT_ROOTS else "helper" if root == "tests" else "other"


def _imports_of(tree) -> dict:
    """Local name -> imported source, for module-level imports and those under `if`/`try`."""
    found = {}
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                found[alias.asname or alias.name.split(".")[0]] = alias.name
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if node.level:
                module = f"tests.{module}" if module else "tests"
            for alias in node.names:
                found[alias.asname or alias.name] = f"{module}.{alias.name}"
        elif isinstance(node, (ast.If, ast.Try)):
            for sub in ast.walk(node):
                if isinstance(sub, ast.ImportFrom) and sub.module:
                    module = f"tests.{sub.module}" if sub.level else sub.module
                    for alias in sub.names:
                        found.setdefault(alias.asname or alias.name, f"{module}.{alias.name}")
                elif isinstance(sub, ast.Import):
                    for alias in sub.names:
                        found.setdefault(alias.asname or alias.name.split(".")[0], alias.name)
    return found


def _imports_machinery(tree) -> bool:
    names = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names += [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            module = f"tests.{node.module or ''}".rstrip(".") if node.level else node.module or ""
            names += [f"{module}.{alias.name}" for alias in node.names]
    return any(MACHINERY_IMPORT.match(name) or MACHINERY_IMPORT.match(name.rsplit(".", 1)[0]) for name in names)


def _handle_only(node, names: set) -> bool:
    """True when every use of `names` inside `node` only points at source text instead of running it."""
    parents = {child: parent for parent in ast.walk(node) for child in ast.iter_child_nodes(parent)}

    def in_handle_call(child) -> bool:
        call = parents.get(child)
        if isinstance(call, ast.Call) and child in call.args:
            func = call.func
            return (func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")) in SOURCE_HANDLES
        return False

    seen = False
    for found in ast.walk(node):
        if isinstance(found, ast.Name) and found.id in names:
            seen = True
            parent = parents.get(found)
            if isinstance(parent, ast.Attribute) and parent.attr in {"__file__", "__name__", "__doc__"}:
                continue
            if in_handle_call(found) or (isinstance(parent, ast.Attribute) and in_handle_call(parent)):
                continue
            return False
    return seen


class _Scope:
    """What the tests of one file reach through its imports, definitions and fixtures (static, conservative)."""

    def __init__(self, tree):
        self.imports = _imports_of(tree)
        self.local: dict = {}
        self.fixtures: set = set()
        self._closures: dict = {}
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                self.local[node.name] = node
                if _is_fixture(node):
                    self.fixtures.add(node.name)
            elif isinstance(node, (ast.Assign, ast.AnnAssign)):
                for target in node.targets if isinstance(node, ast.Assign) else [node.target]:
                    for name in ast.walk(target):
                        if isinstance(name, ast.Name):
                            self.local[name.id] = node
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                for sub in node.body:
                    if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)) and _is_fixture(sub):
                        self.local.setdefault(sub.name, sub)
                        self.fixtures.add(sub.name)

    def _bind_local_import(self, node, product: set, helper: set) -> None:
        modules = ([alias.name for alias in node.names] if isinstance(node, ast.Import)
                   else [f"tests.{node.module or ''}" if node.level else node.module or ""])
        bound = [alias.asname or alias.name.split(".")[0] for alias in node.names]
        for module in modules:
            for name in bound:
                self.imports.setdefault(name, module)
            kind = _classify(module)
            if kind == "product":
                product.update(bound)
            elif kind == "helper":
                helper.update(bound)

    def _reach(self, node, stack=()) -> tuple:
        product, helper, deps = set(), set(), set()
        for found in ast.walk(node):
            if isinstance(found, (ast.Import, ast.ImportFrom)) and found is not node:
                self._bind_local_import(found, product, helper)
            elif (isinstance(found, ast.Constant) and isinstance(found.value, str)
                  and DYNAMIC_PRODUCT.match(found.value)):
                product.add("<dynamic:" + found.value.split(".")[0] + ">")
            elif isinstance(found, ast.Name):
                if found.id in self.imports:
                    kind = _classify(self.imports[found.id])
                    (product if kind == "product" else helper if kind == "helper" else set()).add(found.id)
                elif found.id in self.local and self.local[found.id] is not node and found.id not in stack:
                    deps.add(found.id)
            elif isinstance(found, ast.arg):
                if found.arg in self.fixtures and found.arg not in stack:
                    deps.add(found.arg)
                elif found.arg in self.imports and _classify(self.imports[found.arg]) == "helper":
                    helper.add(found.arg)
        return product, helper, deps

    def _closure(self, name, stack=()) -> tuple:
        if name not in self._closures:
            product, helper, deps = self._reach(self.local[name], stack + (name,))
            source = [ast.unparse(self.local[name])]
            for dep in sorted(deps):
                if dep in self.local:
                    more_product, more_helper, text = self._closure(dep, stack + (name,))
                    product |= more_product
                    helper |= more_helper
                    source.append(text)
            self._closures[name] = (product, helper, "\n".join(source)[:200_000])
        return self._closures[name]

    def facts(self, fn, parametrized: set) -> dict:
        product, helper, deps = self._reach(fn)
        source = [ast.unparse(fn)]
        for dep in sorted(deps):
            if dep in self.local:
                more_product, more_helper, text = self._closure(dep)
                product |= more_product
                helper |= more_helper
                source.append(text)
        external = [arg.arg for arg in fn.args.args + fn.args.kwonlyargs
                    if arg.arg not in BUILTIN_FIXTURES | parametrized | self.fixtures]
        if product:
            handles = (not any(name.startswith("<dynamic:") for name in product) and _handle_only(fn, product)
                       and all(_handle_only(self.local[dep], product) for dep in sorted(deps) if dep in self.local))
            exercise = "source_handle" if handles else "product"
        else:
            exercise = "helper" if helper or external else "none"
        whole = "\n".join(source)
        reads = [] if not READS.search(whole) else [
            kind for kind, pattern in (("prose", PROSE), ("web", WEB_TEXT)) if pattern.search(whole)]
        return {"exercise": exercise, "reads": "+".join(reads),
                "product_names": sorted(self.imports.get(name, name) for name in product)}


def _assert_shape(fn) -> tuple:
    """(asserts, asserts of the form `"literal" in x`)."""
    asserts = substring = 0
    for node in ast.walk(fn):
        if isinstance(node, ast.Assert):
            asserts += 1
            test = node.test
            if isinstance(test, ast.Compare) and any(isinstance(op, (ast.In, ast.NotIn)) for op in test.ops):
                literal = test.left
                if (isinstance(literal, ast.JoinedStr)
                        or (isinstance(literal, ast.Constant) and not isinstance(literal.value, (int, float)))):
                    substring += 1
    return asserts, substring


def _file_rows(rel: str, text: str, tree) -> list:
    lines = text.splitlines()
    scope, machinery, rows = _Scope(tree), _imports_machinery(tree), []
    for qual, fn, axes, marks in _test_functions(tree):
        body = "\n".join(lines[fn.lineno - 1:fn.end_lineno])
        parametrized = _parametrized_names(fn.decorator_list)
        asserts, substring = _assert_shape(fn)
        rows.append({
            "file": rel, "qual": qual, "line": fn.lineno, "axes": axes, "lane_marked": bool(marks & LANE_MARKERS),
            "fixtures": [arg.arg for arg in fn.args.args + fn.args.kwonlyargs
                         if arg.arg not in parametrized and arg.arg not in ("self", "cls")],
            "asserts": asserts, "substring_asserts": substring, "machinery_file": machinery,
            "spawn": bool(SPAWN.search(body)), "git": bool(GIT.search(body)),
            "nested_pytest": bool(NESTED_PYTEST.search(body)),
            "cited": " ".join(sorted(set(ISSUE.findall(f"{ast.get_docstring(fn) or ''}\n{body}")))),
            **scope.facts(fn, parametrized),
        })
    return rows


def scan_tests(root: Path) -> list:
    """One row per statically visible test function of `root/tests`, in file and definition order."""
    rows, tests = [], root / "tests"
    for path in sorted(tests.rglob("test_*.py")):
        if "fixtures" not in path.relative_to(tests).parts:
            text = path.read_text(encoding="utf-8", errors="replace")
            try:
                rows += _file_rows(path.relative_to(root).as_posix(), text, ast.parse(text))
            except SyntaxError:
                continue
    return rows


def function_key(nodeid: str) -> tuple:
    """(file, qualname) of a node id, without its parametrize id."""
    path, _, rest = nodeid.partition("::")
    return path, rest.split("[", 1)[0]


def _without_xdist_group(nodeid: str) -> str:
    """`--dist loadgroup` appends `@serialN` or `@<file-or-class scope>` to a report's node id."""
    head, at, group = nodeid.rpartition("@")
    if at and (re.fullmatch(r"serial\d+", group) or (head.startswith(group) and "::" in head)):
        return head
    return nodeid


def load_journals(paths) -> dict:
    """(file, qual) -> {items, seconds, setup_s} over report rows; an earlier journal wins a function."""
    merged: dict = {}
    for path in paths:
        nodes: dict = defaultdict(lambda: [0.0, 0.0])
        with open(path, "rb") as stream:
            for raw in stream:
                if b'"report"' in raw:
                    row = json.loads(raw)
                    if row.get("event") == "report":
                        node = nodes[_without_xdist_group(row["nodeid"])]
                        node[0] += float(row.get("duration_seconds") or 0.0)
                        node[1] += float(row.get("duration_seconds") or 0.0) if row.get("phase") == "setup" else 0.0
        functions: dict = {}
        for nodeid, (seconds, setup) in nodes.items():
            entry = functions.setdefault(function_key(nodeid), {"items": 0, "seconds": 0.0, "setup_s": 0.0})
            entry["items"] += 1
            entry["seconds"] += seconds
            entry["setup_s"] += setup
        for key, entry in functions.items():
            merged.setdefault(key, entry)
    return merged


def load_collected(path) -> Counter:
    """(file, qual) -> collected items, from `pytest --collect-only -q` output."""
    with open(path, encoding="utf-8") as stream:
        return Counter(function_key(line.strip()) for line in stream if line.startswith("tests/") and "::" in line)


def join_lane(rows: list, journal: dict, collected: Counter | None) -> list:
    """The rows of the default lane, each with its items and recorded seconds."""
    lane = []
    for row in rows:
        key = (row["file"], row["qual"])
        in_lane = key in collected if collected is not None else not row["lane_marked"]
        if in_lane:
            record = journal.get(key, {})
            static = math.prod(row["axes"]) if None not in row["axes"] else 1
            row["items"] = collected[key] if collected is not None else record.get("items", static)
            row["seconds"] = round(record.get("seconds", 0.0), 3)
            row["setup_s"] = round(record.get("setup_s", 0.0), 3)
            lane.append(row)
    return lane


def _modules(row) -> list:
    return sorted({name.rsplit(".", 1)[0] for name in row["product_names"]})


def _rule_for(row, unit_reach, witness, protected_files, rewritten=frozenset()) -> tuple:
    """(cluster, items removed, seconds saved lo, hi) of the first rule that applies, D7 aside."""
    name = row["file"].rsplit("/", 1)[-1]
    items, seconds = row["items"], row["seconds"]
    spawns = row["spawn"] or row["nested_pytest"]
    cited = bool(row["cited"])
    text_read = "prose" in row["reads"] or "web" in row["reads"]
    if row["file"] in protected_files:
        return "K_protected", 0, 0.0, 0.0
    if (not cited and not spawns and not row["machinery_file"] and not SECRET_WORD.search(row["qual"]) and text_read
            and (row["exercise"] in ("none", "source_handle")
                 or 0 < row["asserts"] == row["substring_asserts"])):
        return "D4_wording", items, seconds, seconds
    secret = bool(SECRET_WORD.search(row["qual"]))
    if (not cited and not secret and MACHINERY_FILE.match(name) and (spawns or row["git"])
            and witness.get(row["file"]) is not row and row["product_names"]
            and all(product in rewritten for product in row["product_names"])):
        return "D6_machinery", items, seconds, seconds
    if (not cited and not secret and EVENT_FILE.search(name) and row["product_names"]
            and all(unit_reach[module] >= 3 for module in _modules(row))):
        return "D5_event_file", items, seconds, seconds
    axes = row["axes"]
    computed = None in axes
    if len(axes) >= 2 and not computed and items > 1 and math.prod(axes):
        kept = max(1, round(items * max(axes) / math.prod(axes)))
        if kept < items:
            return ("D1_cartesian", items - kept, *[seconds * (items - kept) / items] * 2)
    if computed and items > 1 and (seconds / items < CHEAP_CASE_S or COST_STAND_FILE.match(name)):
        return ("D3_enumeration", items - 1, *[(items - 1) * min(seconds / items, ITEM_FLOOR_S)] * 2)
    if items > 3 and not cited and (computed or len(axes) == 1):
        return ("D2_cases", items - 3, *[seconds * (items - 3) / items] * 2)
    return "", 0, 0.0, 0.0


def _fold_facets(rows) -> None:
    """D7 on what is still unclaimed: the costliest setup of a same-world group is its survivor."""
    groups = defaultdict(list)
    for row in rows:
        if row["items"] == 1:
            world = tuple(sorted(name for name in row["fixtures"] if name not in WORLD_BLIND_FIXTURES))
            groups[row["file"], world, tuple(_modules(row))].append(row)
    for members in groups.values():
        live = [row for row in members if not row["cluster"]]
        if len(live) >= 2:
            ordered = [row for _, row in sorted(enumerate(live), key=lambda pair: (pair[1]["setup_s"], pair[0]))]
            ordered[-1]["cluster"] = "D7_survivor"
            for row in ordered[:-1]:
                row.update(cluster="D7_facets", items_removed=row["items"], seconds_saved_lo=row["setup_s"],
                           seconds_saved_hi=row["seconds"] * (len(live) - 1) / len(live),
                           survivor=ordered[-1]["qual"])


def _definitions(text: str) -> dict:
    """qualname -> ast dump of every function and class of a module's source."""
    found = {}

    def walk(body, prefix):
        for node in body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                found[prefix + node.name] = ast.dump(node)
                if isinstance(node, ast.ClassDef):
                    walk(node.body, prefix + node.name + ".")

    walk(ast.parse(text).body, "")
    return found


def rewritten_since(root: Path, base: str, rows: list) -> set:
    """The product names the rows use whose function or class definition differs from the one at `base`."""
    names = {product for row in rows for product in row["product_names"]}
    modules: dict = {}
    changed = set()
    for name in sorted(names):
        parts = name.split(".")
        for cut in range(len(parts) - 1, 0, -1):
            rel = "/".join(parts[:cut]) + ".py"
            if (root / rel).is_file():
                break
        else:
            continue
        qual = ".".join(parts[cut:])
        if rel not in modules:
            shown = subprocess.run(["git", "-C", str(root), "show", f"{base}:{rel}"], capture_output=True, text=True)
            before = _definitions(shown.stdout) if shown.returncode == 0 else {}
            modules[rel] = (before, _definitions((root / rel).read_text(encoding="utf-8")))
        before, after = modules[rel]
        if qual in after and before.get(qual) != after[qual]:
            changed.add(name)
    return changed


def decide(rows: list, protected_files, protected_functions, rewritten=frozenset()) -> list:
    """Claim every row for its first rule, then give named functions back: nothing counts twice.

    `rewritten` holds the dotted product names (`ouroboros.mod.function`) whose definition changed since the
    base (`rewritten_since`). A machinery test is obsolete only when all it names is in it: a module, a constant
    or one unchanged function it names may be what it witnesses.
    """
    unit_reach: Counter = Counter()
    witness: dict = {}
    for row in rows:
        name = row["file"].rsplit("/", 1)[-1]
        if not EVENT_FILE.search(name):
            unit_reach.update(_modules(row))
        if MACHINERY_FILE.match(name) and (row["spawn"] or row["nested_pytest"] or row["git"]):
            if row["file"] not in witness or row["seconds"] < witness[row["file"]]["seconds"]:
                witness[row["file"]] = row
    for row in rows:
        cluster, removed, lo, hi = _rule_for(row, unit_reach, witness, protected_files, rewritten)
        row.update(cluster=cluster, items_removed=removed, seconds_saved_lo=lo, seconds_saved_hi=hi, survivor="")
    _fold_facets(rows)
    for row in rows:
        if (row["file"], row["qual"]) in protected_functions and row["cluster"] in DELETED:
            row.update(cluster="P_postmerge_protected", items_removed=0, seconds_saved_lo=0.0, seconds_saved_hi=0.0)
        row.update(rule=RULES[row["cluster"]][0], verdict=RULES[row["cluster"]][1],
                   seconds_saved_lo=round(row["seconds_saved_lo"], 3),
                   seconds_saved_hi=round(row["seconds_saved_hi"], 3))
    return rows


def _block(rows) -> dict:
    return {"functions": len(rows), "items": sum(row["items"] for row in rows),
            "seconds": round(sum(row["seconds"] for row in rows)),
            "items_removed": sum(row["items_removed"] for row in rows),
            "seconds_saved_lo": round(sum(row["seconds_saved_lo"] for row in rows)),
            "seconds_saved_hi": round(sum(row["seconds_saved_hi"] for row in rows))}


def totals(rows: list) -> dict:
    """One block per rule plus the deleted, folded and composite numbers."""
    by_cluster = defaultdict(list)
    for row in rows:
        by_cluster[row["cluster"]].append(row)
    report = {"lane": _block(rows), "rules": {}}
    for cluster, (rule, verdict) in RULES.items():
        report["rules"][cluster or "untouched"] = {"rule": rule, "verdict": verdict, **_block(by_cluster[cluster])}
    report["deleted"] = _block([row for cluster in DELETED for row in by_cluster[cluster]])
    report["folded"] = _block([row for cluster in FOLDED for row in by_cluster[cluster]])
    report["composite"] = _block([row for cluster in DELETED + FOLDED for row in by_cluster[cluster]])
    return report


def _print_table(report: dict) -> None:
    lane = report["lane"]
    print(f"{'rule':24s} {'funcs':>6s} {'items':>6s} {'secs':>7s} | {'removed':>7s} {'%':>5s} | "
          f"{'saved lo':>8s} {'%':>5s} {'saved hi':>8s} {'%':>5s}")
    rows = [*report["rules"].items(), ("deleted", report["deleted"]), ("folded", report["folded"]),
            ("composite", report["composite"])]
    for name, block in rows:
        removed, lo, hi = block["items_removed"], block["seconds_saved_lo"], block["seconds_saved_hi"]
        print(f"{name:24s} {block['functions']:6d} {block['items']:6d} {block['seconds']:7d} | "
              f"{removed:7d} {removed / max(lane['items'], 1) * 100:5.1f} | "
              f"{lo:8d} {lo / max(lane['seconds'], 1) * 100:5.1f} {hi:8d} {hi / max(lane['seconds'], 1) * 100:5.1f}")
    print(f"lane: {lane['functions']} functions, {lane['items']} items, {lane['seconds']} recorded test-seconds")


def _head(root: Path) -> str:
    done = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True)
    return done.stdout.strip() if done.returncode == 0 else ""


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=Path("."), help="checkout whose tests/ is read")
    parser.add_argument("--out", type=Path, required=True, help="directory for decisions.csv and totals.json")
    parser.add_argument("--journal", type=Path, action="append", default=[], help="events.jsonl of a complete run")
    parser.add_argument("--collected", type=Path, help="`pytest --collect-only -q` output of the default lane")
    parser.add_argument("--protected", type=Path, help="default: <root>/tests/protected_tests.json")
    parser.add_argument("--base", help="the commit the change started from; D6 needs it")
    args = parser.parse_args(argv)
    protected_path = args.protected or args.root / "tests" / "protected_tests.json"
    if not protected_path.is_file():
        parser.error(f"{protected_path} is missing: run devtools/protected_tests.py first")
    protected = json.loads(protected_path.read_text(encoding="utf-8"))
    journal = load_journals(args.journal)
    collected = load_collected(args.collected) if args.collected else None
    rows = join_lane(scan_tests(args.root), journal, collected)
    rewritten = rewritten_since(args.root, args.base, rows) if args.base else set()
    decide(rows, set(protected["files"]), {function_key(key) for key in protected["functions"]}, rewritten)
    report = {"head": _head(args.root), **totals(rows)}
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / "decisions.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    (args.out / "totals.json").write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    _print_table(report)
    known = {(row["file"], row["qual"]) for row in rows}
    print(f"journal functions outside this lane: {len(set(journal) - known)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
