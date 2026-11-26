#!/usr/bin/env python3
"""Which browser tests a post-merge window runs: the selection rule of the browser check (pilot).

    python3 -I -S scripts/post_merge_browser_select.py scope --base SHA --head SHA
    python3 -I -S scripts/post_merge_browser_select.py plan --base SHA --head SHA --out plan.json
        [--durations ci-durations.json] [--budget-seconds N] [--args-out args.txt]

.github/workflows/post-merge-browser.yml (owner choice 1, 2026-11-25: a narrow, non-blocking browser check of the
windows that change the interface; a pilot until 2026-12-10 or the first release after v7.12.1) runs a small
set chosen here, browserless and deterministic, from the window's first-parent commits BASE..HEAD. post-merge.yml's
window job runs `scope` to decide whether to start that workflow; its `check` job asks again before it installs
anything, then runs `plan`.

- A module is a `ui_browser` module when its code declares the marker: a `pytestmark`, or a decorator of a test, a
  test class or a method of one, read with `ast` (a module that does not parse counts by its text and runs whole,
  so its collection error surfaces). A string or a comment that mentions the marker declares nothing.
- The check APPLIES when a commit of the window touches `web/**`, a `ui_browser` module (at the head, or at the base
  where the window deleted it or took its marker away), a file under `tests/` that such a module names (a helper or a
  fixture), or the browser lane every browser test loads (SHARED: tests/browser_lane.py and the conftests), which
  runs the sentinels. That is the owner's scope; dependencies (uv.lock, pyproject.toml), the CI runners and the
  setup action stay with the release and manual lanes. Otherwise the window is NOT APPLICABLE: nothing runs, and
  nothing is green.
- Candidates are the `ui_browser` modules at the head: (1) a changed module; (2) a module whose name holds the stem of a
  changed `web/` file (`web/room_work.css` → `test_room_work_browser.py`); (3) a module that names a changed `web/`
  file (`chat_cards.js`), test helper or fixture, or imports from a changed module whose change reaches beyond its
  tests. A module runs whole, unless every reason that chose it is a change confined to some of its tests: then it
  runs those tests (a hunk that only deletes lines, or removes a base line outside the base's tests, is confined to
  none: the module runs whole). A changed test that
  the lane's `-m ui_browser` deselects (a Docker-only test) is EXCLUDED by the lane, named, never passed. A file or stem more than WIDE modules name (`chat.js`, `style.css`, the stem `chat`) is a hub: it
  selects nothing apart from the module named exactly after it, and stays with the release lane. Within a tier the
  window's commits take turns (each commit's cheapest candidate first), so one large merge cannot crowd out the
  others of a burst; then the cheaper candidate first.
- The SENTINELS, the WebKit scenarios whose product regressions reached the 7.12.0 release lane, always run as their
  tests (every engine and parameter the test declares), and run first: pytest runs each test once, in its module's
  order wherever an argument overlaps a wider one, so scripts/post_merge_browser.py (a plugin of the head session)
  moves the two exact cases to the front of the collected items. A sentinel whose test the head no longer defines is
  named, never passed to pytest; scripts/post_merge_browser.py names a scenario the session did not collect or
  complete. An applicable window whose plan has nothing to run (no candidate, no sentinel the head defines) is not
  runnable: scripts/post_merge_browser.py reports it CHECKED PARTIALLY, naming the missing sentinels, and no browser
  is installed.
- The plan takes candidates in that order while the estimate fits BUDGET_SECONDS and at most MAX_MODULES of them; the
  rest are DROPPED and named with their reasons. The estimate counts each node once: the run's frozen weight table
  (scripts/ci_shard_plan.py `ci-durations.json`, `ui_nodes`, milliseconds) times a node; a candidate the table does
  not time weighs the table's median module and is counted as unknown; a sentinel's nodes count under the sentinel.
- `browser`-marked modules without `ui_browser` run in no lane and are not candidates; neither are the tests that
  need the Caddy only the full lane installs (EXCLUDED, named as excluded by the lane): a module of such tests runs
  its other tests by their ids.

Stdlib only: the window job runs `scope` on the runner's bare interpreter.
"""
from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path, PurePosixPath
import re
import statistics
import subprocess
import sys

REPO = Path(__file__).resolve().parents[1]
WIDE = 10  # a file named by more modules than this is a hub: naming it selects nothing
MAX_MODULES = 24
# The plan's estimated test phases. A dry run over the post-merge windows of 18–25.11.2026 measured a median plan of
# 31.1 minutes against a median Linux leg of 60.5 minutes in the windows it applies to.
BUDGET_SECONDS = 2100
SESSION_FACTOR = 1.3
MIN_STEM = 4  # shorter stems (`app`, `ws`) match by accident
CADDY = "needs the Caddy only the full lane installs"
# A module or a test (its node id without parameters) the plan never passes: those that start a Caddy proxy. The rest of
# tests/test_client_views_browser.py stubs its network in the page and runs normally.
EXCLUDED = {"tests/test_separate_ui_browser.py": CADDY,
            **{f"tests/test_client_views_browser.py::{name}": CADDY for name in (
                "test_independent_pages_sign_in_configure_and_use_the_real_core",
                "test_copied_builtin_parent_recovers_a_broken_selected_setup",
                "test_builtin_recovery_is_a_whole_working_page_before_and_after_auth")}}
DESELECTED = "`-m ui_browser` deselects it (a Docker-only or unmarked test)"
# (scenario, the web/ files it guards): the 7.12.0 release lane found these WebKit-only product regressions
# (web/modules/chat_cards.js: job 137377977467; web/room_work.css: job 138790711236).
SENTINELS = (
    ("tests/test_living_cards_browser.py::test_chat_repaint_keeps_place_across_face_switch_and_temp_rows[webkit]",
     ("web/modules/chat_cards.js",)),
    ("tests/test_room_work_browser.py::"
     "test_a_queued_row_says_behind_which_work_it_stands_and_never_overflows[webkit-282]", ("web/room_work.css",)),
)
# The browser lane every browser test loads, whatever it selects (owner choice 1: web/, browser tests, their helpers and
# the lane's own infrastructure): a change here can break any browser test, so the window runs the sentinels.
# Browser modules live in tests/ itself, so the conftests they load are these two.
SHARED = ("tests/browser_lane.py", "tests/conftest.py", "conftest.py")


def shared(path: str) -> bool:
    return path in SHARED
_MODULE = re.compile(r"tests/test_[^/]+\.py")
_UI_TEXT = re.compile(r"\bmark\.ui_browser\b(?!_)")  # a module that does not parse: its text decides


def git(*args: str, cwd: Path = REPO) -> str:
    return subprocess.run(["git", "-c", "core.quotePath=false", *args], cwd=cwd, check=True, capture_output=True,
                          text=True, encoding="utf-8", errors="replace").stdout


def one_line(text: str) -> str:
    """`text` for a single-line Actions output (`name=value`): a backslash or control character in a path is escaped."""
    return re.sub(r"[\\\x00-\x1f\x7f]", lambda match: "\\" + {"\\": "\\", "\n": "n", "\t": "t"}.get(
        match.group(), f"x{ord(match.group()):02x}"), text)


def window_commits(base: str, head: str, cwd: Path = REPO) -> list[dict]:
    """[{sha, files}] of each first-parent commit of BASE..HEAD, oldest first, each against its first parent. Paths
    are read NUL-delimited (`-z`): git never quotes them there, whatever bytes they hold."""
    rows = []
    for sha in git("rev-list", "--first-parent", "--reverse", f"{base}..{head}", cwd=cwd).split():
        files = git("diff", "--name-only", "-z", "--no-renames", f"{sha}^1", sha, cwd=cwd).split("\0")
        rows.append({"sha": sha, "files": sorted(path for path in files if path)})
    return rows


def first_parent(sha: str, cwd: Path = REPO) -> str:
    """`sha^1`, or "" for a root commit: a merge is named only when it is the window's one commit after its base."""
    done = subprocess.run(["git", "rev-parse", "--verify", "--quiet", f"{sha}^1^{{commit}}"], cwd=cwd,
                          capture_output=True, text=True)
    return done.stdout.strip() if done.returncode == 0 else ""


def _marks(expression) -> bool:
    """Whether an expression names `mark.ui_browser` (`pytest.mark.ui_browser`, called, listed or in `marks=`)."""
    return any(isinstance(node, ast.Attribute) and node.attr == "ui_browser" and (
        isinstance(node.value, ast.Name) and node.value.id == "mark"
        or isinstance(node.value, ast.Attribute) and node.value.attr == "mark") for node in ast.walk(expression))


def _pytestmark(statement) -> bool:
    if isinstance(statement, ast.Assign):
        targets = statement.targets
    elif isinstance(statement, (ast.AnnAssign, ast.AugAssign)):
        targets = [statement.target]
    else:
        return False
    return (statement.value is not None and _marks(statement.value)
            and any(isinstance(target, ast.Name) and target.id == "pytestmark" for target in targets))


def declares_ui_browser(text: str) -> bool:
    """Whether a module's code marks a test `ui_browser`: its `pytestmark`, or a decorator of a `test*` function, a
    `Test*` class, or a method or `pytestmark` of one. A string or comment that mentions the marker marks nothing; a
    module that does not parse counts by its text."""
    if "ui_browser" not in text:
        return False
    try:
        tree = ast.parse(text)
    except (SyntaxError, ValueError):
        return _UI_TEXT.search(text) is not None
    functions = (ast.FunctionDef, ast.AsyncFunctionDef)
    for node in tree.body:
        if _pytestmark(node):
            return True
        if isinstance(node, functions) and node.name.startswith("test"):
            if any(_marks(decorator) for decorator in node.decorator_list):
                return True
        elif isinstance(node, ast.ClassDef) and node.name.startswith("Test"):
            if any(_marks(decorator) for decorator in node.decorator_list) or any(
                    _pytestmark(item) or isinstance(item, functions) and item.name.startswith("test")
                    and any(_marks(decorator) for decorator in item.decorator_list) for item in node.body):
                return True
    return False


def ui_modules(texts: dict[str, str]) -> dict[str, str]:
    """{path: source} of the `ui_browser` test modules among `texts` ({path: source} of tests/*.py)."""
    return {path: text for path, text in texts.items() if _MODULE.fullmatch(path) and declares_ui_browser(text)}


def _stem(path: str) -> str:
    return PurePosixPath(path).stem.replace("-", "_").lower()


def token_index(modules: dict[str, str]) -> dict[str, set[str]]:
    """{word: modules naming it}: each module's file-name-like tokens (`chat_cards.js`) and identifiers
    (`separate_ui_support` in `from tests.separate_ui_support import`)."""
    index: dict[str, set[str]] = {}
    for module, text in modules.items():
        for word in set(re.findall(r"[\w.-]+", text)) | set(re.findall(r"\w+", text)):
            index.setdefault(word, set()).add(module)
    return index


def _named_by(path: str, index: dict[str, set[str]]) -> set[str]:
    """The modules that name `path`: a web file or fixture by its file name, a Python module by its module name."""
    word = PurePosixPath(path).stem if path.endswith(".py") else PurePosixPath(path).name
    return index.get(word, set()) - {path}


def applicability(changed: list[str], head_modules: dict[str, str], base_modules=(),
                  index: dict[str, set[str]] | None = None) -> tuple[bool, str]:
    """(applies, why) for the files the window's commits touched."""
    index = token_index(head_modules) if index is None else index
    web = [path for path in changed if path.startswith("web/")]
    if web:
        return True, f"{len(web)} changed file(s) under web/, e.g. {web[0]}"
    tests = [path for path in changed if path in head_modules or path in base_modules]
    if tests:
        return True, f"{len(tests)} changed browser test module(s), e.g. {tests[0]}"
    helpers = [path for path in changed if path.startswith("tests/") and _named_by(path, index)]
    if helpers:
        return True, f"{len(helpers)} changed browser test helper(s) or fixture(s), e.g. {helpers[0]}"
    shared_ = [path for path in changed if shared(path)]
    if shared_:
        return True, f"the shared browser infrastructure changed ({', '.join(shared_)}): the sentinels run"
    return False, ("no commit of the window touches web/, a browser test module, a file such a module names or the "
                   "shared browser infrastructure")


def _span(text: str) -> range:
    """The lines of one side of a hunk header (`12,3`, `12`; `12,0` is none)."""
    start, _, count = text.partition(",")
    return range(int(start), int(start) + (int(count) if count else 1))


def changed_lines(base: str, head: str, paths, cwd: Path = REPO) -> dict[str, set[int]]:
    """{module: head line numbers the window changed} from one `git diff -U0 BASE HEAD`; a hunk that only deletes
    lines, or removes or replaces a base line outside the base's tests (a helper, a fixture, an import), adds line 0,
    which no test holds, so its module runs whole (`changed_tests`)."""
    paths = sorted(paths)
    if not paths:
        return {}
    lines: dict[str, set[int]] = {}
    removed: dict[str, set[int]] = {}
    current = old = None
    for line in git("diff", "-U0", "--no-renames", base, head, "--", *paths, cwd=cwd).splitlines():
        if line.startswith("--- "):
            old = line[6:] if line.startswith("--- a/") else None
        elif line.startswith("+++ "):
            current = line[6:] if line.startswith("+++ b/") else None
        elif line.startswith("@@") and current:
            before, after = line.split(" ", 3)[1:3]
            new = _span(after[1:])
            lines.setdefault(current, set()).update(new or (0,))
            if old == current and _span(before[1:]):
                removed.setdefault(current, set()).update(_span(before[1:]))
    for module, numbers in removed.items():
        shown = subprocess.run(["git", "show", f"{base}:{module}"], cwd=cwd, capture_output=True, text=True,
                               encoding="utf-8", errors="replace")
        spans = collectable_spans(module, shown.stdout) if shown.returncode == 0 else None
        if spans is None or any(not any(first <= number <= last for first, last, _ in spans) for number in numbers):
            lines[module].add(0)
    return lines


def collectable_spans(module: str, text: str) -> list[tuple[int, int, str]] | None:
    """(first line, last line, node id without parameters) of each test pytest collects from `text` (`test_*`
    functions, `test_*` methods of `Test*` classes; decorators included), or None when it does not parse."""
    marked = lane_tests(module, text)
    return None if marked is None else [(first, last, name) for first, last, name, _ in marked]


def lane_tests(module: str, text: str) -> list[tuple[int, int, str, bool]] | None:
    """`collectable_spans` with whether `-m ui_browser` keeps each test: its module's `pytestmark`, its class's
    decorator or `pytestmark`, or its own decorators (a parameter's `marks=` included) name the marker."""
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return None
    spans, everywhere = [], any(_pytestmark(node) for node in tree.body)
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name.startswith("Test"):
            members = [(f"{node.name}::", item) for item in node.body]
            outer = everywhere or any(_marks(d) for d in node.decorator_list) or any(map(_pytestmark, node.body))
        else:
            members, outer = [("", node)], everywhere
        for prefix, item in members:
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) and item.name.startswith("test_"):
                first = min([item.lineno] + [decorator.lineno for decorator in item.decorator_list])
                spans.append((first, item.end_lineno or item.lineno, f"{module}::{prefix}{item.name}",
                              outer or any(_marks(decorator) for decorator in item.decorator_list)))
    return spans


def changed_tests(module: str, text: str, lines) -> list[str] | None:
    """The tests (node ids without parameters) that hold every changed line, or None for the whole module: a change
    outside a test (a fixture, a helper, an import) can reach any of its tests, and unknown lines are no narrowing."""
    spans = collectable_spans(module, text)
    if spans is None or not lines:
        return None
    hit = set()
    for number in lines:
        owner = next((name for first, last, name in spans if first <= number <= last), None)
        if owner is None:
            return None
        hit.add(owner)
    return sorted(hit)


def _under(node: str, prefix: str) -> bool:
    return node == prefix or node.startswith(prefix + ("::" if prefix.endswith(".py") else "["))


def _timed(table: dict | None, prefixes, minus=()) -> tuple[int, dict, bool]:
    """(ms, {chromium, webkit, other} node counts, timed) of the table's nodes under `prefixes`; nodes under `minus`
    are counted elsewhere (a sentinel's) and add nothing, though they still show the candidate is timed."""
    weights = (table or {}).get("ui_nodes") or {}
    total, engines, found = 0, {"chromium": 0, "webkit": 0, "other": 0}, False
    for node, milliseconds in weights.items():
        if any(_under(node, prefix) for prefix in prefixes):
            found = True
            params = node.rsplit("[", 1)[1] if node.endswith("]") and "[" in node else ""
            engines["chromium" if "chromium" in params else "webkit" if "webkit" in params else "other"] += 1
            if not any(_under(node, prefix) for prefix in minus):
                total += int(milliseconds)
    return total, engines, found


def _fallback(table: dict | None) -> int:
    """A module the table does not time weighs its median module."""
    modules: dict[str, int] = {}
    for node, milliseconds in ((table or {}).get("ui_nodes") or {}).items():
        modules[node.split("::", 1)[0]] = modules.get(node.split("::", 1)[0], 0) + int(milliseconds)
    return int(statistics.median(modules.values())) if modules else 60_000


def _name_matches(stem: str, modules) -> list[str]:
    return [module for module in modules if f"_{stem}_" in f"_{PurePosixPath(module).stem[len('test_'):].lower()}_"]


def _covers(arg: str, test: str) -> bool:
    return test == arg or test.startswith(arg + "::")


def select(commits: list[dict], head_modules: dict[str, str], *, base_modules=(), table: dict | None = None,
           budget_seconds: int = BUDGET_SECONDS, max_modules: int = MAX_MODULES, wide: int = WIDE,
           index: dict[str, set[str]] | None = None, lines: dict[str, set[int]] | None = None) -> dict:
    """The plan of one window: its applicability, candidates with reasons, what runs and what was dropped.

    `lines` ({module: changed head lines}) narrows a module chosen only for its own change to the tests that hold
    it; without them a changed module runs whole."""
    changed = sorted({path for commit in commits for path in commit["files"]})
    index = token_index(head_modules) if index is None else index
    lines = lines or {}
    applies, why = applicability(changed, head_modules, base_modules, index)
    plan = {"format": 2, "applicable": applies, "runnable": False, "reason": why, "changed": changed,
            "commits": [{"sha": c["sha"], "files": c["files"]} for c in commits],
            "shared": [path for path in changed if shared(path)],
            "budget_ms": budget_seconds * 1000, "max_modules": max_modules, "selected": [], "dropped": [],
            "excluded": [], "wide": [], "sentinels": [], "args": [], "estimate_ms": 0, "unknown_modules": 0}
    if not applies:
        return plan
    candidates: dict[str, dict] = {}
    offered: dict[int, set[str]] = {}

    def offer(module, tier, reason, path, commit):
        if module in EXCLUDED:
            if module not in {row["module"] for row in plan["excluded"]}:
                plan["excluded"].append({"module": module, "why": EXCLUDED[module]})
            return
        row = candidates.setdefault(module, {"module": module, "tier": tier, "reasons": [], "files": [],
                                             "whole": False})
        row["tier"] = min(row["tier"], tier)
        row["whole"] = row["whole"] or tier > 1  # any reason but its own change runs it whole
        if reason not in row["reasons"]:
            row["reasons"].append(reason)
        if path not in row["files"]:
            row["files"].append(path)
        offered.setdefault(commit, set()).add(module)

    def offer_naming(path, number, reason):
        naming = _named_by(path, index)
        if len(naming) > wide:
            hubs.setdefault(path, {"file": path, "modules": len(naming), "by": "mention"})
        else:
            for module in sorted(naming):
                offer(module, 3, reason, path, number)

    hubs = {}
    for number, commit in enumerate(commits):
        for path in commit["files"]:
            if path in head_modules:
                offer(path, 1, "changed browser test module", path, number)
                if changed_tests(path, head_modules[path], lines.get(path)) is None:  # beyond its tests: its helpers
                    offer_naming(path, number, f"imports from {PurePosixPath(path).name}")
            if path.startswith("web/") and len(_stem(path)) >= MIN_STEM:
                named = _name_matches(_stem(path), head_modules)
                if len(named) > wide:  # a hub stem (`chat`): only its own module, if there is one
                    hubs[path] = {"file": path, "modules": len(named), "by": "name"}
                    named = [m for m in named if PurePosixPath(m).stem in (f"test_{_stem(path)}",
                                                                           f"test_{_stem(path)}_browser")]
                for module in named:
                    offer(module, 2, f"its name holds the stem of {path}", path, number)
            if path.startswith("web/") or (path.startswith("tests/") and path not in head_modules):
                offer_naming(path, number, f"names {PurePosixPath(path).name}")
    plan["wide"] = [hubs[path] for path in sorted(hubs)]
    for scenario, guards in SENTINELS:
        module, test = scenario.split("::", 1)[0], scenario.split("[", 1)[0]
        spans = collectable_spans(module, head_modules[module]) if module in head_modules else None
        present = test in {name for _, _, name in spans or ()}
        plan["sentinels"].append({"scenario": scenario, "test": test, "guards": list(guards), "present": present,
                                  "estimate_ms": _timed(table, [test])[0] if present else 0})
    pinned = [row["test"] for row in plan["sentinels"] if row["present"]]
    fallback = _fallback(table)
    for module, row in list(candidates.items()):
        tests = None if row["whole"] else changed_tests(module, head_modules[module], lines.get(module))
        marked = lane_tests(module, head_modules[module])
        if marked is None:  # it does not parse: passed whole, so its collection error is the session's to report
            tests = None
        elif tests is not None:  # a changed test the lane deselects is excluded by the lane, never passed
            kept = {name for _, _, name, lane in marked if lane}
            plan["excluded"] += [{"module": test, "why": DESELECTED} for test in tests if test not in kept]
            tests = [test for test in tests if test in kept]
        if marked is not None and any(_under(key, module) for key in EXCLUDED):  # its Caddy tests stay out: the
            # others run by their ids
            wanted = tests if tests is not None else [name for _, _, name, lane in marked if lane]
            plan["excluded"] += [{"module": test, "why": EXCLUDED[test]} for test in wanted if test in EXCLUDED]
            tests = [test for test in wanted if test not in EXCLUDED]
        if tests is not None and not tests:
            del candidates[module]
            for own in offered.values():
                own.discard(module)
            continue
        estimate, engines, timed = _timed(table, tests or [module], minus=pinned)
        row.update(tests=tests, estimate_ms=estimate if timed else fallback, timed=timed, nodes=engines)
    # Each commit's own candidates take turns, so one large merge cannot crowd out the others of its window.
    rank = {}
    for number in sorted(offered):
        own = sorted(offered[number], key=lambda m: (candidates[m]["tier"], candidates[m]["estimate_ms"], m))
        for position, module in enumerate(own):
            rank[module] = min(rank.get(module, position), position)
    order = sorted(candidates.values(), key=lambda row: (row["tier"], rank[row["module"]], row["estimate_ms"],
                                                         row["module"]))
    used = sum(row["estimate_ms"] for row in plan["sentinels"])  # each node once: a sentinel's under the sentinel
    for row in order:
        if len(plan["selected"]) >= max_modules:
            plan["dropped"].append({**row, "why": f"over the cap of {max_modules} modules"})
        elif used + row["estimate_ms"] > plan["budget_ms"]:
            plan["dropped"].append({**row, "why": "over the time budget"})
        else:
            plan["selected"].append(row)
            used += row["estimate_ms"]
    # A session stopped early keeps the head of the plan: the sentinels first (the head session's plugin moves their
    # exact cases to the front, whichever argument covers them), then the plan's priority order. pytest runs each
    # test once however many arguments overlap it.
    chosen = [arg for row in plan["selected"] for arg in row["tests"] or [row["module"]]]
    for arg in [test for test in pinned if not any(_covers(other, test) for other in chosen)] + chosen:
        if not any(_covers(other, arg) for other in plan["args"]):
            plan["args"].append(arg)
    plan["estimate_ms"] = used
    plan["unknown_modules"] = sum(not row["timed"] for row in plan["selected"])
    plan["runnable"] = bool(plan["args"])
    if not plan["args"]:  # never pytest without arguments (the whole lane): checked partially, nothing installed
        missing = [row["scenario"] for row in plan["sentinels"] if not row["present"]]
        plan["reason"] = (f"{why}, but no browser test matches it and the head defines no sentinel"
                          + (f" (missing: {', '.join(missing)})" if missing else ""))
    return plan


def session_seconds(budget_seconds: int) -> int:
    """The run's `--session-timeout`: the plan's budget plus the lane's measured overhead over its summed test phases
    (collection, server start; release run 45583955362: 58 min median shard wall against 57.6 min of phases), with
    room for estimates that run slow. It stops the session between tests: what it never started is named."""
    return round(budget_seconds * SESSION_FACTOR)


def tree_modules(root: Path) -> dict[str, str]:
    return ui_modules({f"tests/{path.name}": path.read_text(encoding="utf-8", errors="replace")
                       for path in sorted((root / "tests").glob("test_*.py"))})


def base_modules(base: str, changed, head_modules, cwd: Path = REPO) -> set[str]:
    """The changed test modules that declare `ui_browser` at BASE but not at the head (deleted, or the marker gone): a
    window that drops a browser test is a browser change too."""
    found = set()
    for path in changed:
        if not _MODULE.fullmatch(path) or path in head_modules:
            continue
        shown = subprocess.run(["git", "show", f"{base}:{path}"], cwd=cwd, capture_output=True, text=True,
                               encoding="utf-8", errors="replace")
        if shown.returncode == 0 and declares_ui_browser(shown.stdout):
            found.add(path)
    return found


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="post_merge_browser_select.py", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("scope", "plan"):
        command = commands.add_parser(name)
        command.add_argument("--base", required=True)
        command.add_argument("--head", required=True)
        if name == "plan":
            command.add_argument("--out", type=Path, required=True)
            command.add_argument("--durations", type=Path)
            command.add_argument("--budget-seconds", type=int, default=BUDGET_SECONDS)
            command.add_argument("--args-out", type=Path, help="the pytest arguments of the plan, one per line")
    args = parser.parse_args(argv)
    commits = window_commits(args.base, args.head)
    modules = tree_modules(REPO)
    deleted = base_modules(args.base, sorted({path for commit in commits for path in commit["files"]}), modules)
    if args.command == "scope":  # whether the window is the owner's scope; an empty plan is the check's to report
        applies, why = applicability(sorted({path for commit in commits for path in commit["files"]}), modules, deleted)
        print(f"applicable={'true' if applies else 'false'}")
        print(f"why={one_line(why)}")
        return 0
    table = None
    if args.durations and args.durations.is_file():
        table = json.loads(args.durations.read_text(encoding="utf-8"))
    touched = {path for commit in commits for path in commit["files"] if path in modules}
    plan = select(commits, modules, base_modules=deleted, table=table, budget_seconds=args.budget_seconds,
                  lines=changed_lines(args.base, args.head, touched))
    plan.update(base=args.base, head=args.head, head_parent=first_parent(args.head), timed=table is not None,
                session_seconds=session_seconds(args.budget_seconds))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(plan, indent=1) + "\n", encoding="utf-8")
    if args.args_out:
        args.args_out.write_text("".join(arg + "\n" for arg in plan["args"]), encoding="utf-8")
    print(f"applicable={'true' if plan['applicable'] else 'false'}")
    print(f"runnable={'true' if plan['runnable'] else 'false'}")
    print(f"modules={len(plan['selected'])}")
    print(f"session_seconds={plan['session_seconds']}")
    print(f"dropped={len(plan['dropped'])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
