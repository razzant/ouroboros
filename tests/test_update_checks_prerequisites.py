"""The browser-module suite of a prepared Update's checks: its Node prerequisite (``update_checks._node``), decided
before any test runs and before the checks' budget begins, and its run (``run_tests``'s ``node`` lane), in either tier.
The Node the gate selects (``preflight_node.resolve_node``) must answer ``--version`` at the suite's floor through the
tracked owner: absent, unstartable, a timeout and unusable are each named, unverified; a begun stop raises its typed
refusal; a tree without the suite needs none. The suite then runs whole on W', first, inside the checks' one budget and
contained: its failure blocks, measured on W' alone (never compared on W or attributed to T), and a run the budget cut
reaches no verdict."""
from __future__ import annotations

import os
import sys

import pytest

from supervisor import update_checks
from tests.test_preflight_node import requires_node
from tests.test_update_checks import TOY_FLOOR, _outcome, _toy_workspace

# POSIX stand-ins for the Node the gate selected, by what its --version probe meets.
_FAKE_NODES = {"usable": "echo v24.0.0", "too_old": "echo v18.19.0", "failing": "echo 'node: cannot run' >&2; exit 3",
               "killed": "kill -9 $$", "silent": "sleep 30", "unstartable": "echo v24.0.0"}


def _fake_node(tmp_path, kind):
    node = tmp_path / f"node-{kind}" / "bin" / "node"
    node.parent.mkdir(parents=True)
    node.write_text(f"#!/bin/sh\n{_FAKE_NODES[kind]}\n", encoding="utf-8")
    node.chmod(0o644 if kind == "unstartable" else 0o755)  # not executable: the system refuses the start
    return str(node)


def _node_scene(tmp_path, monkeypatch, *, node, tier="full", browser=True):
    """A tree with one browser-module test (none without ``browser``), the given tier and ``resolve_node`` answering
    ``node``: ``(verdict, (the targets, the Node) of every run started)``."""
    ws, commit = _toy_workspace(tmp_path, {**({"web/tests/a.test.js": ""} if browser else {}),
                                           "tests/test_ok.py": "def test_ok():\n    pass\n"})

    class Selection:
        files, reasons = ("tests/test_ok.py",), ()

        def record(self):
            return {"tier": self.tier, "files": list(self.files), "reasons": []}

    Selection.tier = tier
    ran = []
    monkeypatch.setattr(update_checks, "_contribution", lambda *_a: ({"server.py"}, set()))
    monkeypatch.setattr(update_checks, "_target_floor", lambda *_a: dict(TOY_FLOOR))
    monkeypatch.setattr(update_checks, "_materialize", lambda *_a: None)
    monkeypatch.setattr("ouroboros.test_selection.select_from_paths", lambda *_a, **_kw: Selection())
    monkeypatch.setattr(update_checks, "_environments", lambda *_a: ((sys.executable, dict(os.environ)),
                                                                     (sys.executable, dict(os.environ))))
    monkeypatch.setattr("ouroboros.preflight_node.resolve_node", lambda: node)
    monkeypatch.setattr(update_checks, "run_tests", lambda _ws, _commit, targets, *_a, **kw: ran.append(
        (list(targets), kw.get("node"))) or {**_outcome(rc=0), "evidence": "junit"})
    row = {"path": str(ws), "composition_commit": commit, "layers_base": {"k2": commit},
           "target": {"target_sha": commit}, "environment": {"same": True}}
    return update_checks.selected_tests(row, tmp_path / "scratch"), ran


@pytest.mark.skipif(os.name == "nt", reason="the stand-in Nodes are POSIX shell scripts")
@pytest.mark.parametrize("kind, status", [("absent", "absent"), ("unstartable", "unstartable"), ("silent", "timeout"),
                                          ("failing", "unusable"), ("killed", "unusable"), ("too_old", "unusable")])
def test_a_browser_suite_without_a_usable_node_is_unverified_before_any_test_runs(tmp_path, monkeypatch, kind, status):
    """The browser-module tests need the Node the gate selects at their floor. Absent, a start the system
    refuses, no answer within the probe's own bound, a failed or killed probe (the code-signing SIGKILL class) and an
    older version are each named, before any test runs: unverified, never a test failure, no other Node tried."""
    from ouroboros import config

    monkeypatch.setattr(config, "UPDATE_CHECKS_PROBE_TIMEOUT_SEC", 2)
    verdict, ran = _node_scene(tmp_path, monkeypatch, node=None if kind == "absent" else _fake_node(tmp_path, kind))

    assert ran == [] and verdict["ok"] is False and verdict["blocking"] == [] and verdict["failed"] == {}
    assert (verdict["prerequisite"]["prerequisite"], verdict["prerequisite"]["status"]) == ("node", status)
    assert verdict["unverified"].startswith(
        "a prerequisite of the checks is missing: Node.js for the 1 browser-module test file(s)")


def test_a_begun_stop_refuses_the_node_probe_typed_and_starts_no_test(tmp_path, monkeypatch):
    from ouroboros.tools import shell_process
    from supervisor.update_prepare import PrepareRefused

    monkeypatch.setattr(shell_process, "_panic_requested", True)
    ran = []
    monkeypatch.setattr(update_checks, "run_tests", lambda *_a, **_kw: ran.append(_a))
    with pytest.raises(PrepareRefused) as refused:
        _node_scene(tmp_path, monkeypatch, node=sys.executable)
    assert refused.value.code == "UPDATE_PREPARE_STOPPED" and ran == []


def test_only_a_tree_with_browser_tests_needs_node_in_either_tier_and_the_one_that_answered_runs_them(tmp_path,
                                                                                                     monkeypatch):
    verdict, ran = _node_scene(tmp_path / "empty", monkeypatch, node=None, tier="selected", browser=False)
    assert verdict["ok"] is True and verdict["node"] == {} and ran == [(["tests/test_ok.py"], "")]
    verdict, ran = _node_scene(tmp_path / "selected", monkeypatch, node=None, tier="selected")
    assert ran == [] and (verdict["prerequisite"]["prerequisite"], verdict["prerequisite"]["status"]) == ("node", "absent")
    if os.name != "nt":  # a POSIX stand-in answering at the floor
        node = _fake_node(tmp_path, "usable")
        for tier, targets in (("selected", ["tests/test_ok.py"]), ("full", ["tests/"])):
            verdict, ran = _node_scene(tmp_path / f"{tier}-usable", monkeypatch, node=node, tier=tier)
            assert verdict["ok"] is True and verdict["node"] == {"node": node, "version": "24.0.0"}
            assert ran == [(targets, node)]  # the W' run, and it alone, runs the suite with that Node


def test_a_declared_installation_user_base_must_exist_and_one_without_site_packages_adds_nothing(tmp_path):
    """Before any test runs, the interpreter that will read a declared installation user base is asked where its user
    site-packages are (the harness's own computation). A base that is missing or not absolute is named ``absent`` with
    that path, never a silent fallback to the isolated defaults. An existing base with them passes; one without them
    (a packaged boot whose ``pip --user`` the bundle already satisfied) passes too, its site named as empty."""
    import pathlib

    from ouroboros.test_environment import INSTALLATION_USER_BASE_ENV, user_site_packages

    root, present, empty, missing = (tmp_path / name for name in ("root", "present", "empty", "missing"))
    root.mkdir()
    empty.mkdir()
    pathlib.Path(user_site_packages(present)).mkdir(parents=True)
    runs = [(sys.executable, {**os.environ, INSTALLATION_USER_BASE_ENV: str(base)}) for base in (present, empty)]
    assert update_checks._declared(runs, root) == [user_site_packages(empty)]
    for base in (missing, pathlib.Path("relative-userbase")):
        with pytest.raises(update_checks._Unverified) as absent:
            update_checks._declared([(sys.executable, {**os.environ, INSTALLATION_USER_BASE_ENV: str(base)})], root)
        record = absent.value.record
        assert (record["prerequisite"], record["status"], record["user_base"]) == ("user_site", "absent", str(base))
        assert user_site_packages(base) in str(absent.value)


def test_the_declared_site_probe_answers_through_a_pipe_that_cannot_encode_its_path(tmp_path):
    """The probe's own answer travels as ASCII (``ascii`` of the path, read back as a literal), whatever the pipe's
    encoding and independently of the ``.pth``'s: a Windows ANSI pipe, stood in for by ``PYTHONIOENCODING=cp1252``,
    cannot encode a Cyrillic profile path, and the probe still answers with that exact site."""
    import pathlib

    from ouroboros.test_environment import INSTALLATION_USER_BASE_ENV, user_site_packages

    root, present, empty = tmp_path / "root", tmp_path / "Антон Ä" / "present", tmp_path / "Антон Ä" / "empty"
    root.mkdir()
    empty.mkdir(parents=True)
    pathlib.Path(user_site_packages(present)).mkdir(parents=True)
    runs = [(sys.executable, {**os.environ, INSTALLATION_USER_BASE_ENV: str(base), "PYTHONIOENCODING": "cp1252"})
            for base in (present, empty)]
    assert update_checks._declared(runs, root) == [user_site_packages(empty)]


@pytest.mark.parametrize("empty_roles", [("W'", "W"), ("W",), ("W'",)], ids=["unchanged_lock", "serving", "generation"])
@pytest.mark.parametrize("shared_failure", [False, True], ids=["passing", "shared_failure"])
def test_an_existing_user_base_without_site_packages_still_runs_the_checks_and_stays_declared(
        tmp_path, monkeypatch, empty_roles, shared_failure):
    """A packaged install whose boot sync installed nothing keeps an empty ``state/python-userbase``. Its Update's
    checks run their tests (with an unchanged lock W' and W share that base; with a changed one the generation and the
    serving base are each asked on their own), the record still names each declared base and which site holds no
    packages. The empty base never excuses a shared failure W's harness could not have read: this toy tree predates
    the opt-in, so the failure blocks."""
    import pathlib

    from ouroboros.test_environment import user_site_packages

    ws, commit = _toy_workspace(tmp_path, {"tests/test_ok.py": "def test_ok():\n    pass\n"})
    serving, generation = tmp_path / "state" / "python-userbase", tmp_path / "state" / "python-userbase.update-x"
    unchanged = empty_roles == ("W'", "W")
    for base, role in ((serving, "W"), (generation, "W'")):
        if unchanged and base is generation:
            continue
        base.mkdir(parents=True)
        if role not in empty_roles:
            pathlib.Path(user_site_packages(base)).mkdir(parents=True)

    class Selection:
        tier, files, reasons = "selected", ("tests/test_ok.py",), ()

        def record(self):
            return {"tier": self.tier, "files": list(self.files), "reasons": []}

    ran, case = [], "tests/test_ok.py::test_ok"
    monkeypatch.setattr(update_checks, "_contribution", lambda *_a: ({"server.py"}, set()))
    monkeypatch.setattr(update_checks, "_target_floor", lambda *_a: dict(TOY_FLOOR))
    monkeypatch.setattr("ouroboros.test_selection.select_from_paths", lambda *_a, **_kw: Selection())
    monkeypatch.setattr("supervisor.update_prepare.serving_environment", lambda: ("user_site", serving))
    monkeypatch.setattr(update_checks, "run_tests", lambda _ws, _c, targets, *_a, **_kw: ran.append(list(targets)) or {
        **_outcome(rc=1 if shared_failure else 0, failed={case: "failure:AssertionError:x"} if shared_failure else {}),
        "evidence": "junit"})
    environment = {"same": True} if unchanged else {"generation": {
        "kind": "user_site", "user_base": str(generation), "current": str(serving)}}
    row = {"path": str(ws), "composition_commit": commit, "layers_base": {"k2": commit},
           "target": {"target_sha": commit}, "environment": environment}

    verdict = update_checks.selected_tests(row, tmp_path / "scratch")

    assert "prerequisite" not in verdict and ran and ran[0] == ["tests/test_ok.py"], verdict.get("unverified")
    bases = {"W'": serving if unchanged else generation, "W": serving}
    assert verdict["user_site"] == {
        "declared": str(bases["W'"]), "serving": str(serving),
        "harness": {"W'": "predates_opt_in", "W": "predates_opt_in", "T": "predates_opt_in"},
        "no_site_packages": list(dict.fromkeys(user_site_packages(bases[role]) for role in empty_roles))}
    if not shared_failure:
        assert verdict["ok"] is True and "baseline" not in verdict
    else:
        assert verdict["ok"] is False and verdict["blocking"] == [case] and verdict["already_failing"] == []
        assert (verdict["baseline"]["excuses"], verdict["baseline"]["not_excused"]) == (False, [case])


def test_whether_a_tested_tree_reads_a_declared_installation_site_is_read_from_its_harness_never_run(tmp_path):
    """Per role, from Git alone: a tree whose ``tests/conftest.py`` and ``ouroboros/test_environment.py`` both name
    ``installation_user_site`` reads the declaration; one that names it in neither, or in one only, predates it."""
    from tests.test_update_checks import _toy_commit, _toy_git

    ws, old = _toy_workspace(tmp_path, {"tests/conftest.py": "import os\n",
                                        "ouroboros/test_environment.py": "import os\n"})
    new, _ = _toy_commit(ws, old, {"tests/conftest.py": "_environment['installation_user_site'](os.environ, b)\n",
                                   "ouroboros/test_environment.py": "def installation_user_site(source, base):\n"
                                                                    "    return ''\n"}, branch="opt-in")
    half, _ = _toy_commit(ws, old, {"ouroboros/test_environment.py": "def installation_user_site(source, base):\n"
                                                                     "    return ''\n"}, branch="half")
    _toy_git(ws, "checkout", "-q", "--detach", old)

    assert update_checks._harnesses(ws, {"W'": new, "W": old, "T": half}) == {
        "W'": "opt_in", "W": "predates_opt_in", "T": "predates_opt_in"}


_GREEN_JS = "const test = require('node:test');\n\ntest('green', () => {});\n"
_RED_JS = "const test = require('node:test');\n\ntest('red', () => {\n    throw new Error('BROWSER-RED');\n});\n"
_SLOW_JS = "const test = require('node:test');\n\ntest('slow', () => new Promise((done) => setTimeout(done, 60000)));\n"


def _browser_row(tmp_path, monkeypatch, *, local, js, floor="def test_floor():\n    assert True\n"):
    """Real Git and the real selector: an official base P with the toy floor and browser suite ``js``, the target T
    (P and an official note), this installation's W (P and ``local``) and W' (T and ``local``). The serving packages
    are the interpreter's own, so nothing is declared and every run is the gate's environment."""
    import pathlib

    from tests.test_update_checks import _toy_commit, _toy_git

    monkeypatch.setattr("supervisor.update_prepare.serving_environment", lambda: ("", pathlib.Path(sys.prefix)))
    ws = tmp_path / "ws"
    ws.mkdir()
    _toy_git(ws, "init", "-q")
    _toy_git(ws, "checkout", "-q", "--orphan", "official-base")
    for rel, text in {"ouroboros/test_selection.py": 'FLOOR_TESTS = {"tests/test_floor.py": "toy floor"}\n',
                      "ouroboros/__init__.py": "", "ouroboros/core.py": "GEN = 1\n", "server.py": "print('p')\n",
                      "tests/test_floor.py": floor, "web/tests/a.test.js": js}.items():
        (ws / rel).parent.mkdir(parents=True, exist_ok=True)
        (ws / rel).write_text(text)
    _toy_git(ws, "add", "-A")
    _toy_git(ws, "commit", "-q", "-m", "P")
    p = _toy_git(ws, "rev-parse", "HEAD")
    target, _ = _toy_commit(ws, p, {"notes.md": "official\n"}, branch="official")
    head, head_tree = _toy_commit(ws, p, local, branch="local")
    composition, composition_tree = _toy_commit(ws, target, local, branch="composition")
    return {"path": str(ws), "composition_commit": composition, "target": {"target_sha": target},
            "layers": {"old_commit": head, "old_files": head_tree, "new_files": composition_tree},
            "layers_base": {"k2": head}, "environment": {"same": True}}


def _kept(verdict):
    import json
    import pathlib

    folder = pathlib.Path(verdict["new"]["evidence"])
    return json.loads((folder / "run.json").read_text(encoding="utf-8")), folder


@requires_node
@pytest.mark.parametrize("tier, local", [("selected", {"ouroboros/core.py": "GEN = 2\n"}),
                                         ("full", {"server.py": "print('mine')\n"})])
@pytest.mark.parametrize("js", ["green", "red"])
def test_the_browser_suite_runs_on_w_prime_in_either_tier_and_its_failure_blocks_uncompared(tmp_path, monkeypatch,
                                                                                            tier, local, js):
    """Real Git, selector, environment, pytest and the Node the gate selects. A local core edit keeps the selected
    tier, a local cutover edit the full one; either way the whole browser suite runs first on W', from ``web/``, in
    the checks' budget. Green: ready. Red: the Python tests pass, the suite blocks as ``NODE_SUITE``, and nothing is
    run on W or T for it (no case to compare): never excused, never attributed."""
    from ouroboros.preflight_node import resolve_node

    row = _browser_row(tmp_path, monkeypatch, local=local, js=_GREEN_JS if js == "green" else _RED_JS)

    verdict = update_checks.selected_tests(row, tmp_path / "scratch")

    manifest, folder = _kept(verdict)
    node = resolve_node()
    assert verdict["tier"] == tier and verdict["node"]["node"] == node, verdict.get("unverified")
    assert verdict["new"]["complete"] is True and verdict["failed"] == {}  # every Python case passed
    assert verdict["new"]["lanes"][0] == {**verdict["new"]["lanes"][0], "lane": "node", "files": 1,
                                          "returncode": 0 if js == "green" else 1}
    assert manifest["lanes"][0]["argv"][:2] == [node, "--test"] and manifest["lanes"][0]["argv"][-1] == "tests/a.test.js"
    assert manifest["lanes"][0]["cwd"] == str(_web(row)) and manifest["deadline_seconds"] <= verdict["budget_seconds"]
    if js == "green":
        assert verdict["ok"] is True and verdict["blocking"] == []
    else:
        assert verdict["ok"] is False and verdict["blocking"] == [update_checks.NODE_SUITE] and "unverified" not in verdict
        assert "BROWSER-RED" in (folder / "node.stdout.txt").read_text(encoding="utf-8")
        assert "baseline" not in verdict and "target_failing" not in verdict  # no W or T run for it
    assert verdict["process_root_release"]["state"] == "removed"


def _web(row):
    import pathlib

    return pathlib.Path(row["path"]) / "web"


@requires_node
def test_a_python_case_w_shares_is_excused_while_the_red_browser_suite_still_blocks(tmp_path, monkeypatch):
    """The exact W case/signature exception is unchanged for Python: the floor fails the same way on W, complete, so it
    is already failing. The browser suite failing beside it is not part of that comparison and still blocks."""
    row = _browser_row(tmp_path, monkeypatch, local={"ouroboros/core.py": "GEN = 2\n"}, js=_RED_JS,
                       floor="def test_floor():\n    assert False, 'known'\n")

    verdict = update_checks.selected_tests(row, tmp_path / "scratch")

    assert verdict["ok"] is False and verdict["blocking"] == [update_checks.NODE_SUITE]
    assert verdict["already_failing"] == ["tests/test_floor.py::test_floor"]
    assert verdict["baseline"]["complete"] is True and update_checks.NODE_SUITE not in verdict["baseline"]["failed"]


@requires_node
def test_a_browser_suite_the_budget_cuts_reaches_no_verdict(tmp_path, monkeypatch):
    """The suite shares the checks' one budget (no budget of its own): cut there, its lane is 124, the Python lanes
    never start, and the checks are unverified, never ready and never a measured failure."""
    monkeypatch.setattr(update_checks, "_budget", lambda: 6)
    row = _browser_row(tmp_path, monkeypatch, local={"ouroboros/core.py": "GEN = 2\n"}, js=_SLOW_JS)

    verdict = update_checks.selected_tests(row, tmp_path / "scratch")

    assert verdict["ok"] is False and verdict["blocking"] == [] and verdict["new"]["complete"] is False
    assert [(lane["lane"], lane["returncode"]) for lane in verdict["new"]["lanes"]] == [
        ("node", 124), ("parallel", 124), ("serial", 124)]
    assert verdict["unverified"].startswith("the selected run did not complete (node: exit 124, 1 file(s)")
    assert verdict["process_root_release"]["state"] == "removed"
