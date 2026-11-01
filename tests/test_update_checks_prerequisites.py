"""A full tier's Node prerequisite for a prepared Update's checks (``update_checks._node``), decided before any test
runs and before the checks' budget begins: the Node the gate selects (``preflight_node.resolve_node``) must answer
``--version`` at the browser-module suite's floor through the tracked owner. Absent, unstartable, a timeout and unusable
are each named, unverified; a begun stop raises its typed refusal; a selected tier needs none."""
from __future__ import annotations

import os
import sys

import pytest

from supervisor import update_checks
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


def _node_scene(tmp_path, monkeypatch, *, node, tier="full"):
    """A tree with one browser-module test, the given tier and ``resolve_node`` answering ``node``:
    ``(verdict, the targets of every run started)``."""
    ws, commit = _toy_workspace(tmp_path, {"web/tests/a.test.js": "", "tests/test_ok.py": "def test_ok():\n    pass\n"})

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
    monkeypatch.setattr(update_checks, "run_tests", lambda _ws, _commit, targets, *_a, **_kw: ran.append(
        list(targets)) or {**_outcome(rc=0), "evidence": "junit"})
    row = {"path": str(ws), "composition_commit": commit, "layers_base": {"k2": commit},
           "target": {"target_sha": commit}, "environment": {"same": True}}
    return update_checks.selected_tests(row, tmp_path / "scratch"), ran


@pytest.mark.skipif(os.name == "nt", reason="the stand-in Nodes are POSIX shell scripts")
@pytest.mark.parametrize("kind, status", [("absent", "absent"), ("unstartable", "unstartable"), ("silent", "timeout"),
                                          ("failing", "unusable"), ("killed", "unusable"), ("too_old", "unusable")])
def test_a_full_tier_without_a_usable_node_is_unverified_before_any_test_runs(tmp_path, monkeypatch, kind, status):
    """The full tier's browser-module tests need the Node the gate selects at their floor. Absent, a start the system
    refuses, no answer within the probe's own bound, a failed or killed probe (the code-signing SIGKILL class) and an
    older version are each named, before any test runs: unverified, never a test failure, no other Node tried."""
    from ouroboros import config

    monkeypatch.setattr(config, "UPDATE_CHECKS_PROBE_TIMEOUT_SEC", 2)
    verdict, ran = _node_scene(tmp_path, monkeypatch, node=None if kind == "absent" else _fake_node(tmp_path, kind))

    assert ran == [] and verdict["ok"] is False and verdict["blocking"] == [] and verdict["failed"] == {}
    assert (verdict["prerequisite"]["prerequisite"], verdict["prerequisite"]["status"]) == ("node", status)
    assert verdict["unverified"].startswith(
        "a prerequisite of the checks is missing: Node.js for the full tier's 1 browser-module test file(s)")


def test_a_begun_stop_refuses_the_node_probe_typed_and_starts_no_test(tmp_path, monkeypatch):
    from ouroboros.tools import shell_process
    from supervisor.update_prepare import PrepareRefused

    monkeypatch.setattr(shell_process, "_panic_requested", True)
    ran = []
    monkeypatch.setattr(update_checks, "run_tests", lambda *_a, **_kw: ran.append(_a))
    with pytest.raises(PrepareRefused) as refused:
        _node_scene(tmp_path, monkeypatch, node=sys.executable)
    assert refused.value.code == "UPDATE_PREPARE_STOPPED" and ran == []


def test_a_selected_tier_needs_no_node_and_a_full_tier_records_the_one_that_answered(tmp_path, monkeypatch):
    verdict, ran = _node_scene(tmp_path / "selected", monkeypatch, node=None, tier="selected")
    assert verdict["ok"] is True and verdict["node"] == {} and ran == [["tests/test_ok.py"]]
    if os.name != "nt":  # a POSIX stand-in answering at the floor
        node = _fake_node(tmp_path, "usable")
        verdict, ran = _node_scene(tmp_path / "full", monkeypatch, node=node)
        assert verdict["ok"] is True and verdict["node"] == {"node": node, "version": "24.0.0"} and ran == [["tests/"]]


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
