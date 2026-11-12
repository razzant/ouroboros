"""The post-merge check: base choice, book-growth mode, one retry, the size check, bisection to the
culprit merge and the summary (the hand-over itself: tests/test_post_merge_routing.py).

Deterministic: GitHub is a fake, test sessions are fakes that write tests/ci_evidence.py's files,
and git is a throwaway local repository without remotes, beside one read-only `git grep` of this
checkout for the modules of its size check.
"""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import urllib.error

import pytest

from scripts import post_merge_check as pmc

REPO = Path(__file__).resolve().parents[1]
HEAD = "f" * 40


def _row(nodeid, outcome="passed", phase="call"):
    return {"nodeid": nodeid, "phase": phase, "outcome": outcome, "is_subtest": False}


def _results(directory, reports, *, exit_code=None, collection=(), selected=None, skips=None):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    failed = any(row["outcome"] == "failed" for row in reports) or collection
    result = {
        "format": 1, "session_exit_code": (1 if failed else 0) if exit_code is None else exit_code,
        "reports": reports,
        "collection_failures": [{"nodeid": path, "outcome": "failed"} for path in collection],
    }
    if selected is not None:  # what tests/ci_evidence.py's session left to run after every filter
        result["selected_nodeids"] = list(selected)
    if skips is not None:  # the collectors that skipped at collection; a producer older than the field names none
        result["collection_skips"] = list(skips)
    (directory / "results.json").write_text(json.dumps(result), encoding="utf-8")


def _selection(evidence, *, parallel="success", serial="success", node="success", inventory=None):
    evidence.mkdir(parents=True, exist_ok=True)
    (evidence / "selection.json").write_text(json.dumps({
        "tier": "selected", "file_count": 3, "available": 40, "base": "b" * 40, "node": node,
        "outcomes": {"parallel": parallel, "serial": serial}, "inventory": inventory or {},
    }), encoding="utf-8")


# --- base ------------------------------------------------------------------------------------

def _run(sha, conclusion="success", status="completed", number=1):
    return {"id": number, "head_sha": sha, "status": status, "conclusion": conclusion}


def test_the_base_is_the_head_of_the_newest_completed_check():
    ancestors = {"a" * 40, "c" * 40}
    runs = [_run(HEAD, number=9), _run("d" * 40, status="in_progress", conclusion=None, number=8),
            _run("e" * 40, conclusion="cancelled", number=7), _run("x" * 40, number=6),
            _run("c" * 40, conclusion="failure", number=5), _run("a" * 40, number=4)]
    base, why = pmc.choose_base(runs, "a" * 40, HEAD, ancestor=lambda sha, head: sha in ancestors,
                                parent=lambda head: pytest.fail("no fallback"))
    # The run on HEAD itself, the unfinished, the cancelled and the non-ancestor runs are skipped.
    assert base == "c" * 40 and "run 5, failure" in why


def test_the_base_falls_back_to_the_previous_tip_and_then_to_the_first_parent():
    parent = lambda head: "p" * 40  # noqa: E731
    assert pmc.choose_base([], "a" * 40, HEAD, ancestor=lambda sha, head: True, parent=parent) == (
        "a" * 40, "the branch tip before this push")
    for before in ("", pmc.ZERO_SHA, "n" * 40):  # missing, a new branch, not an ancestor
        base, why = pmc.choose_base([_run("n" * 40)], before, HEAD,
                                    ancestor=lambda sha, head: sha != "n" * 40, parent=parent)
        assert (base, why) == ("p" * 40, "the first parent of this push")


def test_the_base_command_survives_an_api_error_and_publishes_its_choice(tmp_path, monkeypatch, capsys):
    def broken(self, path):
        raise urllib.error.HTTPError(path, 503, "Service Unavailable", None, None)

    monkeypatch.setattr(pmc.GitHub, "get", broken)
    monkeypatch.setattr(pmc, "is_ancestor", lambda sha, head: sha == "a" * 40)
    output = tmp_path / "github-output"
    environ ={"GITHUB_SHA": HEAD, "BEFORE": "a" * 40, "GITHUB_OUTPUT": str(output), "GITHUB_REPOSITORY": "o/r"}
    parsed = type("Args", (), {"head": None, "evidence_dir": tmp_path / "evidence"})
    assert pmc.command_base(parsed, environ) == 0
    assert output.read_text(encoding="utf-8") == f"base={'a' * 40}\n"
    assert json.loads((tmp_path / "evidence" / "base.json").read_text(encoding="utf-8"))["base"] == "a" * 40
    assert "HTTP 503" in capsys.readouterr().out


# --- growth ----------------------------------------------------------------------------------

def test_each_merge_of_the_window_is_judged_as_its_own_push_and_one_approval_decides(tmp_path, monkeypatch):
    asked = []

    def decide(environ):
        asked.append(environ)
        return "approved" if environ["SHA"] == "1" * 40 else "block"

    environ = {"REPO": "razzant/ouroboros", "OWNER": "razzant", "GH_TOKEN": "t", "GITHUB_OUTPUT": "/dev/null",
               "EVENT": "pull_request", "PR_NUMBER": "9"}
    assert pmc.approved_merges(["2" * 40, "1" * 40], environ, decide) == ["1" * 40]
    assert asked[0] == {"REPO": "razzant/ouroboros", "OWNER": "razzant", "GH_TOKEN": "t", "EVENT": "push",
                        "REF": "refs/heads/ouroboros", "SHA": "2" * 40}  # nothing else of this job leaks in
    monkeypatch.setattr(pmc, "merges", lambda base, head: [("2" * 40, "Merge #2"), ("1" * 40, "Merge #1")])
    args = type("Args", (), {"evidence_dir": tmp_path, "base": "b" * 40, "head": HEAD})
    for verdict, mode, approved in ((decide, "approved", ["1" * 40]), (lambda environ: "block", "block", [])):
        assert pmc.command_growth(args, environ, verdict) == 0
        assert json.loads((tmp_path / "growth.json").read_text(encoding="utf-8")) == {
            "format": 1, "base": "b" * 40, "head": HEAD, "mode": mode, "merges": ["2" * 40, "1" * 40], "approved": approved}


def test_the_growth_decision_is_book_growth_mode_for_one_push(monkeypatch):
    """scripts/book_growth_mode.py itself decides each merge: the owner labelled its merged PR last."""
    import io

    from scripts import book_growth_mode

    sha, events = "1" * 40, []
    pages = {f"commits/{sha}/pulls": [{"number": 7, "state": "closed", "merged_at": "2026-10-28T00:00:00Z",
                                       "merge_commit_sha": sha,
                                       "base": {"ref": "ouroboros", "repo": {"full_name": "razzant/ouroboros"}}}],
             "issues/7/events": events}

    def urlopen(request, timeout):
        response = io.BytesIO(json.dumps(pages[request.full_url.split("/razzant/ouroboros/")[1].split("?")[0]]).encode())
        response.headers = {}
        return response

    monkeypatch.setattr(book_growth_mode.urllib.request, "urlopen", urlopen)
    environ = {"REPO": "razzant/ouroboros", "OWNER": "razzant"}
    assert pmc.approved_merges([sha], environ, pmc._book_growth_mode) == []
    events.append({"event": "labeled", "actor": {"login": "razzant"}, "label": {"name": "book-growth"}})
    assert pmc.approved_merges([sha], environ, pmc._book_growth_mode) == [sha]


# --- sessions --------------------------------------------------------------------------------

def test_a_session_without_its_final_export_is_read_from_its_journal(tmp_path):
    events = [{"event": "start", "nodeid": "tests/test_a.py::test_hang"},
              {"event": "start", "nodeid": "tests/test_a.py::test_red"},
              {"event": "report", "nodeid": "tests/test_a.py::test_red", "phase": "call", "outcome": "failed"},
              {"event": "finish", "nodeid": "tests/test_a.py::test_red"},
              {"event": "start", "nodeid": "tests/test_b.py::test_crash"},
              {"event": "report", "nodeid": "tests/test_b.py::test_crash", "phase": "crash", "outcome": "failed"}]
    (tmp_path / "events.jsonl").write_text("\n".join(map(json.dumps, events)) + '\n{"event": "sta', encoding="utf-8")
    session = pmc.read_session(tmp_path)
    assert session["final"] is False and session["exit"] is None
    assert pmc.candidates(session) == {"tests/test_a.py::test_red": "failed in call",
                                       "tests/test_b.py::test_crash": "its worker crashed",
                                       "tests/test_a.py::test_hang": "was running when the session ended"}
    # Without a completed session nothing is known to be absent.
    assert pmc.outcomes(session, ["tests/test_a.py::test_hang", "tests/test_c.py::test_x"]) == {
        "tests/test_a.py::test_hang": "failed", "tests/test_c.py::test_x": "unknown"}


def test_outcomes_name_each_id_and_a_module_stands_for_its_tests(tmp_path):
    _results(tmp_path, [_row("tests/t.py::test_ok"), _row("tests/t.py::test_ok", phase="teardown"),
                        _row("tests/t.py::test_red", "failed"), _row("tests/t.py::test_skip", "skipped", "setup"),
                        _row("tests/u.py::test_late", "failed", "teardown"), _row("tests/v.py::test_ok")],
             collection=["tests/w.py"])
    session = pmc.read_session(tmp_path)
    assert pmc.outcomes(session, ["tests/t.py::test_ok", "tests/t.py::test_red", "tests/t.py::test_skip",
                                  "tests/u.py::test_late", "tests/t.py::test_gone", "tests/w.py::test_any",
                                  "tests/v.py", "tests/u.py", "tests/w.py"]) == {
        "tests/t.py::test_ok": "passed", "tests/t.py::test_red": "failed", "tests/t.py::test_skip": "skipped",
        "tests/u.py::test_late": "failed", "tests/t.py::test_gone": "unknown", "tests/w.py::test_any": "failed",
        "tests/v.py": "passed", "tests/u.py": "failed", "tests/w.py": "failed"}


def test_a_class_or_function_id_stands_for_its_tests_and_parametrizations(tmp_path):
    _results(tmp_path, [_row("tests/t.py::test_p[1]"), _row("tests/t.py::test_p[2]"), _row("tests/t.py::test_pp", "failed"),
                        _row("tests/t.py::TestK::test_a"), _row("tests/t.py::test_q[1]"), _row("tests/t.py::test_q[2]", "failed")])
    ids = ["tests/t.py::test_p", "tests/t.py::TestK", "tests/t.py::test_q", "tests/t.py::test_p[1]",
           "tests/t.py::TestK::test_b"]
    assert pmc.outcomes(pmc.read_session(tmp_path), ids) == dict(zip(ids, [
        "passed", "passed", "failed", "passed", "unknown"]))  # test_pp is not one of test_p's parametrizations


def test_an_aggregate_id_passes_only_when_each_test_its_session_selected_ran_whole(tmp_path):
    module, one, two = "tests/t.py", "tests/t.py::test_one", "tests/t.py::test_two"
    ids = [module, one, two, "tests/t.py::test_none"]
    # A budget that ended the session before test_two leaves the module unknown, never passed on test_one alone.
    _results(tmp_path, _whole(one), selected=[one, two], exit_code=1)
    assert pmc.outcomes(pmc.read_session(tmp_path), ids) == dict(zip(ids, ["unknown", "passed", "unknown", "unknown"]))
    _results(tmp_path, [*_whole(one), *_whole(two)], selected=[one, two])
    assert pmc.outcomes(pmc.read_session(tmp_path), ids) == dict(zip(ids, ["passed", "passed", "passed", "unknown"]))
    # test_two exists, but the session's markers left it out (the lane's exclusions, the size check's selection): it
    # runs in no session that judges the module, while test_two itself stays unknown, neither passed nor absent.
    _results(tmp_path, _whole(one), selected=[one])
    assert pmc.outcomes(pmc.read_session(tmp_path), ids) == dict(zip(ids, ["passed", "passed", "unknown", "unknown"]))
    _results(tmp_path / "size", _whole("tests/test_smoke.py::test_sized"), selected=["tests/test_smoke.py::test_sized"])
    assert pmc.size_status(pmc.read_session(tmp_path / "size"), ["size tests/test_smoke.py"]) == {
        "size tests/test_smoke.py": "passed"}
    # A class that did not collect fails the module and its own ids; the module's other tests keep their verdict.
    _results(tmp_path, [*_whole(one), *_whole(two)], selected=[one, two], collection=["tests/t.py::TestK"])
    assert pmc.outcomes(pmc.read_session(tmp_path), [module, "tests/t.py::TestK::test_k", one]) == {
        module: "failed", "tests/t.py::TestK::test_k": "failed", one: "passed"}
    # A class that skipped at collection held tests that never ran: no red and no pass of its module or itself, with a
    # selection or without one (an xdist controller's export). Beneath it nothing is known; its siblings keep theirs.
    klass, inner = "tests/t.py::TestK", "tests/t.py::TestK::test_k"
    for selection in ([one, two], None):
        _results(tmp_path, [*_whole(one), *_whole(two)], selected=selection, skips=[klass])
        assert pmc.candidates(pmc.read_session(tmp_path)) == {}
        assert pmc.outcomes(pmc.read_session(tmp_path), [module, klass, inner, one, two]) == dict(zip(
            [module, klass, inner, one, two], ["skipped", "skipped", "unknown", "passed", "passed"]))
    # A failure beside it still fails the module, and a skip in one module touches no other.
    _results(tmp_path, [*_whole(one), _row(two, "failed")], selected=[one, two], skips=[klass, "tests/u.py::TestX"])
    assert pmc.outcomes(pmc.read_session(tmp_path), [module]) == {module: "failed"}
    _results(tmp_path, _whole(one), selected=[one], skips=["tests/u.py::TestX"])
    assert pmc.outcomes(pmc.read_session(tmp_path), [module]) == {module: "passed"}
    # A whole module that skipped at collection: exit 5, nothing selected, its ids no pass and none of them absent.
    _results(tmp_path, [], selected=[], skips=[module], exit_code=5)
    assert pmc.outcomes(pmc.read_session(tmp_path), [module, one]) == {module: "skipped", one: "unknown"}
    # Without a selection (an older producer) an aggregate counts only on an export that ended (exit 0 or 1).
    _results(tmp_path, _whole(one), exit_code=2)
    assert pmc.outcomes(pmc.read_session(tmp_path), [module, one]) == {module: "unknown", one: "passed"}


def _xfail(nodeid, phase="call", reason="known breach", **facts):
    """A report row of pytest's XFAIL as tests/ci_evidence.py keeps it: "skipped", with the marker's reason."""
    return {**_row(nodeid, "skipped", phase), "wasxfail": reason, **facts}


def _whole(nodeid, call=None):
    """The setup, call and teardown a pass reports of a test that ran whole."""
    return [_row(nodeid, phase="setup"), call or _row(nodeid), _row(nodeid, phase="teardown")]


def test_an_expected_failure_that_ran_is_neither_a_pass_nor_a_skip(tmp_path):
    def ran(nodeid, call):
        return [_row(nodeid, phase="setup"), call, _row(nodeid, phase="teardown")]
    _results(tmp_path, [*ran("tests/t.py::test_breach", _xfail("tests/t.py::test_breach")),
                        _xfail("tests/t.py::test_notrun", "setup", "[NOTRUN] kept"), _row("tests/t.py::test_notrun", phase="teardown"),
                        _row("tests/t.py::test_late", phase="setup"), _xfail("tests/t.py::test_late"),
                        _xfail("tests/t.py::test_late", "teardown"),
                        *ran("tests/t.py::test_opted", _xfail("tests/t.py::test_opted", error_type="XFailed")),
                        *ran("tests/t.py::test_legacy", _row("tests/t.py::test_legacy", "skipped")),
                        *ran("tests/u.py::test_ok", _row("tests/u.py::test_ok")),
                        *ran("tests/u.py::test_breach", _xfail("tests/u.py::test_breach")),
                        _row("tests/t.py::test_torn", phase="setup"), _xfail("tests/t.py::test_torn")])
    ids = ["tests/t.py::test_breach", "tests/t.py::test_notrun", "tests/t.py::test_late", "tests/t.py::test_opted",
           "tests/t.py::test_legacy", "tests/u.py", "tests/t.py", "tests/t.py::test_torn"]
    # Only the call that ran whole and failed as its marker expects; an older row without the field proves nothing.
    assert pmc.outcomes(pmc.read_session(tmp_path), ids) == dict(zip(ids, [
        "xfailed", "skipped", "skipped", "skipped", "skipped", "xfailed", "skipped", "skipped"]))


def test_container_args_never_name_a_node_the_tree_lacks():
    sources = {
        "tests/test_a.py": "def test_plain(): pass\n\ndef test_param(x): pass\n\nclass TestK:\n"
                           "    def test_m(self): pass\n\nclass TestInherited(Base):\n    pass\n",
        "tests/test_b.py": "def test_one(): pass\n",
        "tests/test_broken.py": "def broken(:\n",
    }
    ids = ["tests/test_a.py::test_plain", "tests/test_a.py::test_param[case-1]", "tests/test_a.py::test_param[2]",
           "tests/test_a.py::TestK::test_m", "tests/test_a.py::TestInherited::test_m", "tests/test_a.py::test_new",
           "tests/test_a.py::TestGone::test_m", "tests/test_b.py::test_one", "tests/test_gone.py::test_x",
           "tests/test_broken.py::test_y"]
    args, absent = pmc.container_args(ids, sources.get)
    # A missing function or class falls back to the module, which then covers every node in it.
    assert args == ["tests/test_a.py", "tests/test_b.py::test_one", "tests/test_broken.py"]
    assert absent == ["tests/test_gone.py::test_x"]
    args, _ = pmc.container_args(ids[:5], sources.get)
    assert args == ["tests/test_a.py::TestInherited", "tests/test_a.py::TestK::test_m",
                    "tests/test_a.py::test_param", "tests/test_a.py::test_plain"]


# --- classify --------------------------------------------------------------------------------

def test_one_retry_separates_failed_from_flaky(tmp_path):
    _selection(tmp_path, parallel="failure", serial="failure")
    _results(tmp_path / "parallel", [_row("tests/t.py::test_flaky", "failed"), _row("tests/t.py::test_broken", "failed"),
                                     _row("tests/t.py::test_fine")])
    _results(tmp_path / "serial", [_row("tests/s.py::test_skips_now", "failed", "setup")])
    seen = []

    def retry(args, directory):
        seen.append((list(args), Path(directory)))
        _results(directory, [_row("tests/t.py::test_flaky"), _row("tests/t.py::test_broken", "failed"),
                             _row("tests/s.py::test_skips_now", "skipped", "setup")])

    provisional = []
    record = pmc.classify(tmp_path, run=retry, node=lambda: pytest.fail("the Node lane was green"),
                          checkpoint=lambda early: provisional.append((len(seen), early)))
    # Before the retry starts, every red is on record as FAILED: what a killed step leaves behind.
    [(retries_so_far, early)] = provisional
    assert retries_so_far == 0 and early["complete"] is False and early["flaky"] == []
    assert [(row["id"], row["retry"]) for row in early["failed"]] == [
        ("tests/s.py::test_skips_now", "not run"), ("tests/t.py::test_broken", "not run"),
        ("tests/t.py::test_flaky", "not run")]
    assert record["complete"] is True
    # Exactly the failures, once, in one session.
    assert seen == [(["tests/s.py::test_skips_now", "tests/t.py::test_broken", "tests/t.py::test_flaky"],
                     tmp_path / "retry")]
    assert [row["id"] for row in record["flaky"]] == ["tests/t.py::test_flaky"]
    # Only a pass is a pass: a failure skipped on its retry stays FAILED.
    assert [(row["id"], row["retry"]) for row in record["failed"]] == [
        ("tests/s.py::test_skips_now", "skipped"), ("tests/t.py::test_broken", "failed")]
    assert record["failed"][0]["pass"] == "serial" and record["problems"] == []


LINTED = "ruff F811 tests/t.py fake_route"


def test_lint_findings_name_code_file_and_name_and_say_when_ruff_could_not_run(tmp_path, monkeypatch):
    """The release's lint gate on ruff's JSON rows, so no ruff is needed here; a line number is not
    part of the id, it moves with edits. A ruff that cannot run or answers anything but a list is None."""
    asked = []

    def ruff(answer, code=0, stderr=""):
        def run(argv, **kwargs):
            asked.append((argv, kwargs["cwd"]))
            if isinstance(answer, BaseException):
                raise answer
            return subprocess.CompletedProcess(argv, code, answer, stderr)
        monkeypatch.setattr(pmc.subprocess, "run", run)

    module = str(tmp_path / "pkg" / "m.py")
    for line in (5, 9):  # the same two findings before and after an edit above them
        ruff(json.dumps([{"code": "F401", "filename": module, "location": {"row": 1, "column": 8},
                          "message": "`os` imported but unused"},
                         {"code": "F811", "filename": module, "location": {"row": line, "column": 5},
                          "message": f"Redefinition of unused `loads` from line {line - 3}: `loads` redefined here"}]))
        found = pmc.lint_findings(tmp_path)
        assert sorted(found) == ["ruff F401 pkg/m.py os", "ruff F811 pkg/m.py loads"]
        assert f"from line {line - 3}" in found["ruff F811 pkg/m.py loads"]
    assert asked[-1] == ([sys.executable, "-m", "ruff", *pmc.LINT_ARGS, "."], tmp_path.resolve())
    ruff("[]")
    assert pmc.lint_findings(tmp_path, ["pkg/clean.py"]) == {}
    assert asked[-1][0][-1] == "pkg/clean.py"

    for answer, code, stderr in (("", 1, f"{sys.executable}: No module named ruff\n"),  # the packaged runtime
                                 (OSError("no ruff"), 0, ""), (subprocess.TimeoutExpired("ruff", 600), 0, ""),
                                 ("not json", 0, ""), ('{"code": "F401"}', 0, ""), ("", 0, ""),
                                 ("[]", 2, "ruff failed: invalid configuration\n")):
        ruff(answer, code, stderr)
        assert pmc.lint_findings(tmp_path) is None, (answer, code)


def test_lint_findings_read_the_installed_ruff(tmp_path):
    """The same ids from a real ruff run; only an interpreter without ruff (the packaged runtime) skips."""
    pytest.importorskip("ruff", reason="ruff is a development dependency; the packaged runtime has none")
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "m.py").write_text("import os\nfrom json import loads\n\n\ndef loads():\n    return 1\n",
                                           encoding="utf-8")
    (tmp_path / "pkg" / "clean.py").write_text("VALUE = 1\n", encoding="utf-8")
    assert sorted(pmc.lint_findings(tmp_path)) == ["ruff F401 pkg/m.py os", "ruff F811 pkg/m.py loads"]
    assert pmc.lint_findings(tmp_path, ["pkg/clean.py"]) == {}


def test_a_lint_finding_is_failed_without_a_retry_and_ruff_that_cannot_run_is_a_problem(tmp_path):
    _selection(tmp_path)
    no_retry = dict(run=lambda *a: pytest.fail("a lint finding is deterministic: no retry"), node=lambda: 0)
    record = pmc.classify(tmp_path, lint=lambda: {LINTED: "Redefinition of unused `fake_route` from line 3"},
                          **no_retry)
    assert record["failed"] == [{"id": LINTED, "pass": "lint", "retry": "failed",
                                 "why": "Redefinition of unused `fake_route` from line 3"}]
    assert record["lint_checked"] is True and record["problems"] == []
    record = pmc.classify(tmp_path, lint=lambda: None, **no_retry)
    assert record["failed"] == [] and record["lint_checked"] is False
    assert record["problems"] == ["ruff could not run, so the lint gate went unchecked"]
    assert pmc.classify(tmp_path, **no_retry)["lint_checked"] is False  # a caller that does not lint


def test_a_lint_finding_is_probed_on_its_file_as_the_commit_has_it(tmp_path):
    tree = tmp_path / "tree"
    (tree / "tests").mkdir(parents=True)
    (tree / "tests" / "t.py").write_text("x = 1\n", encoding="utf-8")
    absent = "ruff F401 tests/gone.py os"
    asked = []

    def lint(cwd, paths):
        asked.append((Path(cwd), list(paths)))
        return {LINTED: "Redefinition"}

    assert pmc.lint_answers(tree, [LINTED, absent], lint) == {LINTED: "fail", absent: "pass"}
    assert asked == [(tree, ["tests/t.py"])]  # only the files the commit has
    assert pmc.lint_answers(tree, [LINTED], lambda cwd, paths: {}) == {LINTED: "pass"}
    assert pmc.lint_answers(tree, [LINTED], lambda cwd, paths: None) == {LINTED: "error"}
    assert pmc.lint_answers(tree, [], lambda cwd, paths: pytest.fail("no finding, no ruff")) == {}


SIZED = "size tests/test_smoke.py::test_no_oversized_modules"


def test_the_size_check_is_the_release_lane_over_the_modules_that_mark_it(tmp_path):
    repo = tmp_path / "repo"
    (repo / "tests" / "nested").mkdir(parents=True)
    files = {"tests/test_sized.py": "@pytest.mark.size_ratchet\ndef test_a(): pass\n",
             "tests/nested/test_deep.py": "pytestmark = [pytest.mark.serial, pytest.mark.size_ratchet]\n",
             "tests/test_class.py": "class TestK:\n    pytestmark: list = [pytest.mark.size_ratchet]\n",
             "tests/test_plain.py": "def test_b(): pass  # the size_ratchet marker lane\n",
             # Mentions, not declarations: a fixture string, a comment, a longer marker name.
             "tests/test_fixture.py": 'SOURCE = "@pytest.mark.size_ratchet\\ndef test_x(): pass"\n'
                                      "# @pytest.mark.size_ratchet goes on each size test\n"
                                      "@pytest.mark.size_ratchet_extra\ndef test_y(): pass\n",
             "tests/helper.py": "@pytest.mark.size_ratchet\n"}  # not a test module
    for path, text in files.items():
        (repo / path).write_text(text, encoding="utf-8")
    _git(repo, "init", "-q")
    _git(repo, "add", "-A")
    assert pmc.size_modules(repo) == ["tests/nested/test_deep.py", "tests/test_class.py", "tests/test_sized.py"]
    _git(repo, "rm", "-q", "--cached", "tests/test_class.py")
    seen = []

    def run(args, directory, *, cwd, env):
        seen.append((list(args), Path(directory), Path(cwd), env))
        _results(directory, [_row("tests/test_sized.py::test_a", "failed")])

    session = pmc.size_session(tmp_path / "size", base="b" * 40, mode="approved", cwd=repo, run=run)
    assert seen == [(["-m", "size_ratchet", "tests/nested/test_deep.py", "tests/test_sized.py"], tmp_path / "size", repo,
                     {"OURO_SIZE_RATCHET_BASE_REF": "b" * 40, "OURO_BOOK_GROWTH_MODE": "approved"})]
    assert pmc.size_findings(session) == ({"size tests/test_sized.py::test_a": "failed in call"}, [])
    _git(repo, "rm", "-q", "--cached", "tests/test_sized.py", "tests/nested/test_deep.py")
    assert pmc.size_modules(repo) == []  # an untracked module is not the tree's
    assert pmc.size_session(tmp_path / "none", base="b" * 40, cwd=repo, run=run) is None and len(seen) == 1


def test_the_release_size_modules_are_found_in_this_checkout():
    if not (REPO / ".git").exists():
        pytest.skip("this tree is not a git checkout")
    release = {"tests/test_reference_book_budgets.py", "tests/test_repo_health_smoke.py", "tests/test_smoke.py"}
    found = set(pmc.size_modules(REPO))
    assert release <= found and all(Path(path).name.startswith("test_") for path in found)
    # This module writes the marker into its fixtures; it declares no size test of its own.
    assert "tests/test_post_merge_check.py" not in found


def test_a_size_check_that_skipped_or_did_not_finish_measured_nothing(tmp_path):
    _results(tmp_path / "skip", [_row("tests/test_smoke.py::test_a"), _row("tests/test_b.py::test_books", "skipped", "setup")])
    found, problems = pmc.size_findings(pmc.read_session(tmp_path / "skip"))
    assert found == {} and problems == ["the size check skipped `tests/test_b.py::test_books`, so it measured nothing there"]
    # A module id of the size check is an aggregate like any other: a skip at collection within it is no pass.
    _results(tmp_path / "uncollected", _whole("tests/test_b.py::test_a"), skips=["tests/test_b.py::TestK"])
    assert pmc.size_status(pmc.read_session(tmp_path / "uncollected"), ["size tests/test_b.py"]) == {
        "size tests/test_b.py": "skipped"}
    _results(tmp_path / "green", [_row("tests/test_smoke.py::test_a")])
    assert pmc.size_findings(pmc.read_session(tmp_path / "green")) == ({}, [])
    unfinished = ["the size check could not run or finish, so it went unchecked"]
    _results(tmp_path / "interrupted", [_row("tests/test_smoke.py::test_a", "failed")], exit_code=2)
    assert pmc.size_findings(pmc.read_session(tmp_path / "interrupted")) == (None, unfinished)
    assert pmc.size_findings(pmc.read_session(tmp_path / "missing")) == (None, unfinished)
    assert pmc.size_findings(None) == (None, unfinished)  # no module marks a size test
    assert pmc.size_status(None, [SIZED]) == {SIZED: "unknown"}


def test_a_red_size_test_is_failed_without_a_retry(tmp_path):
    _selection(tmp_path, parallel="failure")
    _results(tmp_path / "parallel", [_row("tests/t.py::test_red", "failed")])
    _results(tmp_path / "serial", [])
    _results(tmp_path / "size", [_row(SIZED.removeprefix("size "), "failed"), _row("tests/test_smoke.py::test_ok")])
    retried = []

    def retry(args, directory):
        retried.append(list(args))
        _results(directory, [_row("tests/t.py::test_red", "failed")])

    record = pmc.classify(tmp_path, run=retry, node=lambda: 0, size=lambda: pmc.read_session(tmp_path / "size"))
    assert retried == [["tests/t.py::test_red"]]  # the size test is deterministic: never in the retry
    assert [(row["id"], row["pass"], row["retry"]) for row in record["failed"]] == [
        (SIZED, "size", "failed"), ("tests/t.py::test_red", "parallel", "failed")]
    assert record["size_checked"] is True and record["problems"] == []
    record = pmc.classify(tmp_path, run=retry, node=lambda: 0, size=lambda: None)
    assert record["size_checked"] is False and record["problems"] == [
        "the size check could not run or finish, so it went unchecked"]
    record = pmc.classify(tmp_path, run=retry, node=lambda: 0)  # a leg without the size check
    assert "size_checked" not in record and [row["id"] for row in record["failed"]] == ["tests/t.py::test_red"]


def test_classify_runs_the_gates_its_leg_names_and_measures_books_by_the_decided_mode(tmp_path, monkeypatch):
    _selection(tmp_path)
    for label in ("parallel", "serial"):
        _results(tmp_path / label, [_row(f"tests/{label}.py::test_ok")])
    sessions, linted = [], []
    monkeypatch.setattr(pmc, "size_session", lambda directory, **kwargs: sessions.append(kwargs) or None)
    monkeypatch.setattr(pmc, "lint_findings", lambda: linted.append(True) or {})
    monkeypatch.setattr(pmc, "run_pytest", lambda *a, **k: pytest.fail("nothing to retry"))
    (tmp_path / "growth.json").write_text(json.dumps({"mode": "approved", "approved": ["a" * 40]}), encoding="utf-8")
    assert pmc.main(["classify", "--evidence-dir", str(tmp_path), "--head", HEAD, "--lint", "--size", "--base", "b" * 40]) == 0
    record = json.loads((tmp_path / "classification.json").read_text(encoding="utf-8"))
    assert sessions == [{"base": "b" * 40, "mode": "approved"}] and linted == [True]
    assert record["gates"] == ["lint", "size"] and record["size"] == {
        "base": "b" * 40, "mode": "approved", "decided": True, "approved": ["a" * 40]}
    (tmp_path / "growth.json").unlink()  # the growth step did not run: books are measured strictly
    assert pmc.main(["classify", "--evidence-dir", str(tmp_path), "--head", HEAD, "--size", "--base", "b" * 40]) == 0
    record = json.loads((tmp_path / "classification.json").read_text(encoding="utf-8"))
    assert sessions[-1] == {"base": "b" * 40, "mode": "block"} and record["size"]["decided"] is False
    assert record["gates"] == ["size"] and linted == [True]  # this leg did not lint
    assert pmc.main(["classify", "--evidence-dir", str(tmp_path), "--head", HEAD]) == 0  # the Windows leg
    record = json.loads((tmp_path / "classification.json").read_text(encoding="utf-8"))
    assert record["gates"] == [] and "size" not in record and len(sessions) == 2 and linted == [True]
    with pytest.raises(SystemExit):
        pmc.main(["classify", "--evidence-dir", str(tmp_path), "--size"])


def test_a_red_node_lane_is_retried_as_one_target(tmp_path):
    for code, verdict in ((0, "flaky"), (1, "failed")):
        evidence = tmp_path / str(code)
        _selection(evidence, node="failure")
        _results(evidence / "parallel", [_row("tests/t.py::test_ok")])
        _results(evidence / "serial", [])
        record = pmc.classify(evidence, run=lambda *a: pytest.fail("no pytest red to retry"), node=lambda: code)
        assert [row["id"] for row in record[verdict]] == [pmc.NODE_LANE]


def test_a_retry_pytest_refuses_runs_the_nodes_the_tree_defines(tmp_path):
    _selection(tmp_path, parallel="failure")
    _results(tmp_path / "parallel", [_row("tests/t.py::test_p[redacted-value]", "failed")])
    _results(tmp_path / "serial", [])
    sessions = []

    def run(args, directory):
        sessions.append(list(args))
        if len(sessions) == 1:  # "ERROR: not found": nothing ran
            _results(directory, [], exit_code=4)
        else:
            _results(directory, [_row("tests/t.py::test_p[redacted-value]")])

    record = pmc.classify(tmp_path, run=run, node=lambda: 0, read_source={"tests/t.py": "def test_p(v): pass\n"}.get)
    assert sessions == [["tests/t.py::test_p[redacted-value]"], ["tests/t.py::test_p"]]
    assert [row["id"] for row in record["flaky"]] == ["tests/t.py::test_p[redacted-value]"]


def test_a_check_that_could_not_see_its_failures_says_so(tmp_path):
    assert pmc.classify(tmp_path)["problems"] == [
        "the selected run left no selection.json (its selection failed or it died early)"]
    _selection(tmp_path, parallel="failure", serial="skipped",
               inventory={"parallel": {"expected": ["tests/t.py::a", "tests/t.py::b"], "executed": ["tests/t.py::a"]}})
    _results(tmp_path / "parallel", [_row("tests/t.py::a")])
    record = pmc.classify(tmp_path, run=lambda *a: pytest.fail("nothing to retry"), node=lambda: 0)
    assert record["problems"] == ["the parallel pass was red without a failing test (read its log)",
                                  "the serial pass did not run"]
    assert record["not_run"] == {"parallel": ["tests/t.py::b"]} and record["failed"] == []
    # A pass killed before it recorded its outcome still left its journal.
    (tmp_path / "serial").mkdir()
    (tmp_path / "serial" / "events.jsonl").write_text(json.dumps({"event": "start", "nodeid": "tests/s.py::test_slow"}),
                                                      encoding="utf-8")
    record = pmc.classify(tmp_path, run=lambda args, directory: _results(directory, [_row("tests/s.py::test_slow")]),
                          node=lambda: 0)
    assert record["problems"][1] == "the serial pass did not finish"
    assert [(row["id"], row["why"]) for row in record["flaky"]] == [
        ("tests/s.py::test_slow", "was running when the session ended")]


def test_a_successful_retry_cannot_hide_an_unrun_selected_test(tmp_path, monkeypatch, capsys):
    red, missing = "tests/t.py::test_retry", "tests/t.py::test_unrun"
    _selection(tmp_path, parallel="failure", inventory={
        "parallel": {"expected": [red, missing], "executed": [red]}})
    _results(tmp_path / "parallel", [_row(red, "failed")], exit_code=2)
    _results(tmp_path / "serial", [])
    retried = []

    def retry(ids, directory):
        retried.extend(ids)
        _results(directory, [_row(red)])

    record = pmc.classify(tmp_path, run=retry, node=lambda: 0)
    assert retried == [red] and not record["failed"]
    assert record["not_run"] == {"parallel": [missing]}
    (tmp_path / "classification.json").write_text(json.dumps(record), encoding="utf-8")
    text, is_red = pmc.render_summary(tmp_path)
    assert is_red and "— RED" in text and missing in text
    assert "passed on their one retry" in text  # the good observation is not discarded
    monkeypatch.delenv("GITHUB_STEP_SUMMARY", raising=False)
    assert pmc.main(["summary", "--evidence-dir", str(tmp_path), "--os", "windows"]) == 1
    assert missing in capsys.readouterr().out
    # Already retained classification records have the same meaning without reclassification.
    record["problems"] = []
    (tmp_path / "classification.json").write_text(json.dumps(record), encoding="utf-8")
    assert pmc.render_summary(tmp_path)[1]


@pytest.mark.parametrize("cost_outcome,session,expected_red", [
    ("failure", "failed", True), ("failure", "passed", True),
    ("failure", "unfinished", True), ("skipped", "missing", True),
    ("success", "missing", True), ("success", "failed", True),
    ("success", "skipped", True), ("success", "passed", False),
    ("", "missing", True),
])
def test_cost_outcome_reaches_the_real_summary_command(tmp_path, monkeypatch, capsys,
                                                      cost_outcome, session, expected_red):
    (tmp_path / "classification.json").write_text(json.dumps({
        "complete": True, "failed": [], "flaky": [], "problems": [], "not_run": {}}), encoding="utf-8")
    nodeid = "tests/test_cost_contract.py::test_rest_law"
    if session not in ("missing", "unfinished"):
        _results(tmp_path / "cost-contract", [_row(nodeid, session,
                 "setup" if session == "skipped" else "call")])
    elif session == "unfinished":
        directory = tmp_path / "cost-contract"
        directory.mkdir()
        (directory / "events.jsonl").write_text(json.dumps({"event": "start", "nodeid": nodeid}), encoding="utf-8")
    monkeypatch.setenv("COST_CONTRACT_OUTCOME", cost_outcome)
    monkeypatch.delenv("GITHUB_STEP_SUMMARY", raising=False)
    code = pmc.main(["summary", "--evidence-dir", str(tmp_path), "--os", "linux"])
    text = capsys.readouterr().out
    assert code == int(expected_red) and "Cost contract" in text
    assert ("— RED" in text) is expected_red
    if session == "failed":
        assert nodeid in text
    # Windows deliberately does not run the cost contract, even if another leg failed it.
    assert pmc.main(["summary", "--evidence-dir", str(tmp_path), "--os", "windows"]) == 0
    assert "Cost contract: not selected on Windows" in capsys.readouterr().out


BREACH = "tests/test_cost_contract.py::test_room_history_is_flat[k30-R_late]"


@pytest.mark.parametrize("case,expected_red,said", [
    ("executed", False, "Known breach `" + BREACH + "`, measured as its xfail marker expects; it does not make this "
                        "leg red: reads one more tools segment (k=1 -> k=30)."),
    ("not_run", True, f"Cost check `{BREACH}` is an xfail that never ran its test."),
    ("setup_error", True, f"Cost check `{BREACH}` is an xfail that never ran its test."),
    ("teardown", True, f"Cost check `{BREACH}` failed in teardown, which its xfail marker absorbed."),
    ("opted_out", True, f"Cost check `{BREACH}` ended in pytest.xfail() instead of measuring."),
    ("missing_teardown", True, f"Cost check `{BREACH}` is an xfail without its setup or teardown on record"),
    ("missing_setup", True, f"Cost check `{BREACH}` is an xfail without its setup or teardown on record"),
    ("legacy", True, "by evidence older than the xfail record: whether it ran is unknown."),
    ("skip", True, f"Cost check `{BREACH}` was skipped."),
    ("strict_xpass", True, f"Cost check `{BREACH}`: failed in call."),
    ("unreported", True, "Cost contract not fully verified"),
    ("red_session", True, "Cost contract not fully verified"),
])
def test_only_a_known_breach_that_ran_leaves_the_cost_contract_verified(tmp_path, monkeypatch, capsys,
                                                                        case, expected_red, said):
    """#2972: a gap row's strict xfail that ran is measured, listed with its reason, and not red by itself."""
    (tmp_path / "classification.json").write_text(json.dumps({
        "complete": True, "failed": [], "flaky": [], "problems": [], "not_run": {}}), encoding="utf-8")
    passing = "tests/test_cost_contract.py::test_room_history_is_flat[k30-R_active]"
    breach = {
        "executed": [_row(BREACH, phase="setup"), _xfail(BREACH), _row(BREACH, phase="teardown")],
        "not_run": [_xfail(BREACH, "setup", "[NOTRUN] reads one more tools segment"), _row(BREACH, phase="teardown")],
        "setup_error": [_xfail(BREACH, "setup", error_type="RuntimeError"), _row(BREACH, phase="teardown")],
        "teardown": [_row(BREACH, phase="setup"), _xfail(BREACH), _xfail(BREACH, "teardown")],
        "opted_out": [_row(BREACH, phase="setup"), _xfail(BREACH, error_type="XFailed"), _row(BREACH, phase="teardown")],
        "missing_teardown": [_row(BREACH, phase="setup"), _xfail(BREACH)],
        "missing_setup": [_xfail(BREACH), _row(BREACH, phase="teardown")],
        "legacy": [_row(BREACH, phase="setup"), _row(BREACH, "skipped"), _row(BREACH, phase="teardown")],
        "skip": [{**_row(BREACH, "skipped", "setup"), "wasxfail": None}, _row(BREACH, phase="teardown")],
        "strict_xpass": [_row(BREACH, phase="setup"), _row(BREACH, "failed"), _row(BREACH, phase="teardown")],
    }.get(case, [_row(BREACH, phase="setup"), _xfail(BREACH), _row(BREACH, phase="teardown")])
    for row in breach:
        if row.get("wasxfail") == "known breach":
            row["wasxfail"] = "reads one more tools segment (k=1 -> k=30)."
    directory = tmp_path / "cost-contract"
    directory.mkdir()
    (directory / "results.json").write_text(json.dumps({
        "format": 1, "session_exit_code": int(case in ("strict_xpass", "red_session")),
        "tests_collected": 3 if case == "unreported" else 2,
        "reports": [_row(passing, phase="setup"), _row(passing), _row(passing, phase="teardown"), *breach],
        "collection_failures": []}), encoding="utf-8")
    monkeypatch.setenv("COST_CONTRACT_OUTCOME", "success")
    monkeypatch.delenv("GITHUB_STEP_SUMMARY", raising=False)
    code = pmc.main(["summary", "--evidence-dir", str(tmp_path), "--os", "linux"])
    text = capsys.readouterr().out
    assert code == int(expected_red) and ("— RED" in text) is expected_red and said in text
    assert ("Cost contract not fully verified" in text) is expected_red
    notice = f"::notice title=Known cost-contract breach on Linux::{BREACH} (an expected failure, measured)"
    assert (notice in text) is (case in ("executed", "unreported", "red_session"))
    if case == "executed":
        assert "1 passed, 0 failed/unfinished, 0 skipped, 1 known breach(es) measured." in text


@pytest.mark.parametrize("stage", ["classify", "attribute", "route"])
@pytest.mark.parametrize("outcome", ["failure", "skipped", "", "success"])
def test_workflow_stage_outcome_reaches_summary_without_a_new_retry(tmp_path, monkeypatch, capsys, stage, outcome):
    (tmp_path / "classification.json").write_text(json.dumps({
        "complete": True, "failed": [], "flaky": [], "problems": []}), encoding="utf-8")
    monkeypatch.setenv(f"POST_MERGE_{stage.upper()}_OUTCOME", outcome)
    monkeypatch.delenv("GITHUB_STEP_SUMMARY", raising=False)
    code = pmc.main(["summary", "--evidence-dir", str(tmp_path), "--os", "windows"])
    assert code == (0 if outcome == "success" else 1)
    text = capsys.readouterr().out
    if outcome != "success":
        assert f"the {stage} step is {outcome or 'unknown'}" in text


def test_a_missing_report_does_not_establish_removal_and_a_skip_does_not_establish_a_pass(tmp_path):
    ids = ["tests/t.py", "tests/t.py::test_deselected", "tests/gone.py::test_removed"]
    def run(args, directory):
        assert args == ["tests/t.py"]
        _results(directory, [_row("tests/t.py::test_ok"), _row("tests/t.py::test_skip", "skipped", "setup")])
    found = pmc.check_ids(ids, tmp_path, read_source={"tests/t.py": "def test_deselected(): pass\n"}.get, run=run)
    assert found == dict(zip(ids, ["skipped", "unknown", "absent"]))


def test_a_recheck_runs_only_the_listed_tests_and_absence_needs_their_modules_complete_collection(tmp_path):
    ids = ["tests/t.py::test_ok", "tests/t.py::test_param", "tests/t.py::test_renamed", "tests/t.py::test_deselected",
           "tests/u.py::test_x", "tests/v.py::TestK::test_y", "tests/gone.py::test_z"]
    sources, seen = {path: "" for path in ("tests/t.py", "tests/u.py", "tests/v.py")}, {}

    def run(args, directory):
        seen["args"], seen["listed"] = args, Path(args[-2].split("=", 1)[1]).read_text(encoding="utf-8").split()
        _results(directory, [*_whole("tests/t.py::test_ok"), *_whole("tests/t.py::test_param[1]"),
                             *_whole("tests/t.py::test_param[2]")], collection=["tests/u.py", "tests/v.py::TestK"],
                 selected=["tests/t.py::test_ok", "tests/t.py::test_param[1]", "tests/t.py::test_param[2]"])
        data = json.loads((Path(directory) / "results.json").read_text(encoding="utf-8"))
        data["listed"] = {"unmatched": ["tests/t.py::test_renamed", "tests/u.py::test_x", "tests/v.py::TestK::test_y"]}
        (Path(directory) / "results.json").write_text(json.dumps(data), encoding="utf-8")

    found = pmc.recheck_ids(ids, tmp_path / "recheck", read_source=sources.get, run=run)
    assert seen["args"][:-2] == ["tests/t.py", "tests/u.py", "tests/v.py"] and seen["args"][-2].startswith("--ci-listed=")
    assert seen["args"][-1] == f"--session-timeout={pmc.RECHECK_SECONDS}"
    assert seen["listed"] == ids[:-1]  # a module the head lacks is not collected: its ids are absent
    # Gone only where the module collected cleanly; a module or class that did not collect fails what it holds, and the
    # test a marker left out (test_deselected: it exists, so it was never unmatched) stays unknown.
    assert found == dict(zip(ids, ["passed", "passed", "absent", "unknown", "failed", "failed", "absent"]))

    def killed(args, directory):  # no final export: its journal names what ran, never what is gone
        Path(directory).mkdir(parents=True, exist_ok=True)
        (Path(directory) / "events.jsonl").write_text("".join(json.dumps(row) + "\n" for row in [
            {"event": "start", "nodeid": ids[0]}, {"event": "report", **_row(ids[0])}, {"event": "finish", "nodeid": ids[0]},
            {"event": "start", "nodeid": "tests/t.py::test_param[1]"},
            {"event": "report", **_row("tests/t.py::test_param[1]")}]), encoding="utf-8")
    assert pmc.recheck_ids(ids[:3], tmp_path / "killed", read_source=sources.get, run=killed) == dict(zip(
        ids[:3], ["passed", "failed", "unknown"]))  # test_param[1] was still running when the session ended


def test_the_head_picks_the_module_a_recheck_starts_from_so_a_slow_red_front_holds_back_no_fixed_tail(tmp_path):
    """#2834: a recheck runs its modules in argument order until its budget ends, so the module that comes first is
    the one reached. The head's CRC-32 rotates the sorted modules: one head keeps one order, and successive heads bring
    every module to the front. The listed ids are the same whatever the order, and a test without a report stays unknown."""
    import zlib

    modules = [f"tests/test_{name}.py" for name in "abcde"]
    ids = [f"{module}::test_x" for module in reversed(modules)]
    heads, seen = [f"{index:x}" * 40 for index in range(16)], []

    def run(args, directory):
        assert Path(args[-2].split("=", 1)[1]).read_text(encoding="utf-8").splitlines() == ids
        seen.append(args[:-2])
        _results(directory, [])

    for head in heads:
        for again in range(2):
            assert pmc.recheck_ids(ids, tmp_path / f"{head[0]}{again}", head=head, run=run,
                                   read_source=dict.fromkeys(modules, "").get) == dict.fromkeys(ids, "unknown")
    for head, first, again in zip(heads, seen[::2], seen[1::2]):
        start = zlib.crc32(head.encode()) % len(modules)
        assert first == again == modules[start:] + modules[:start]
    assert {order[0] for order in seen} == set(modules)


def test_windows_shaped_ids_and_paths_reach_the_session_whole(tmp_path, monkeypatch):
    """The Windows leg: a parameter keeps its drive letter, backslashes and `::` in the nodeid, the evidence path has a
    drive letter and a space, and text mode writes CRLF. Each id is one listed line, each argument one argsfile line
    that pytest's @file reader (argparse: `splitlines`) returns whole, and the results map back to the exact ids."""
    import argparse
    import zlib

    ids = ["tests/sub/test_w.py::test_path[C:\\\\work\\\\a b.txt]", "tests/sub/test_w.py::TestK::test_m[x::y]",
           "tests/test_v.py::test_v"]
    seen, directory = {}, tmp_path / "D:" / "a b" / "recheck"

    def run(args, directory):
        seen["args"] = args
        _results(directory, [*_whole(ids[0]), *_whole(ids[1])])

    found = pmc.recheck_ids(ids, directory, head="h", run=run,
                            read_source=dict.fromkeys(["tests/sub/test_w.py", "tests/test_v.py"], "").get)
    assert found == dict(zip(ids, ["passed", "passed", "unknown"]))
    modules, start = ["tests/sub/test_w.py", "tests/test_v.py"], zlib.crc32(b"h") % 2
    assert seen["args"][:2] == modules[start:] + modules[:start]
    assert seen["args"][2] == f"--ci-listed={directory / 'listed.txt'}"
    text = (directory / "listed.txt").read_text(encoding="utf-8")
    assert text.replace("\n", "\r\n").splitlines() == text.splitlines() == ids  # what Windows text mode writes

    commands = []
    monkeypatch.setattr(pmc.subprocess, "call", lambda command, **kwargs: commands.append(command) or 0)
    windows = ["tests/test_v.py", "--ci-listed=D:\\a\\_temp\\post-merge\\recheck\\listed.txt", "--session-timeout=225"]
    assert pmc.run_pytest(windows, tmp_path / "a b", cwd=tmp_path) == 0
    assert commands[0][3:5] == ["--ci-evidence-dir", str(tmp_path / "a b")]
    argsfile = Path(commands[0][2].removeprefix("@"))
    argsfile.write_bytes(argsfile.read_text(encoding="utf-8").replace("\n", "\r\n").encode("utf-8"))
    parser = argparse.ArgumentParser(fromfile_prefix_chars="@", allow_abbrev=False)
    parser.add_argument("paths", nargs="*")
    known, options = parser.parse_known_args([commands[0][2]])
    assert [*known.paths, *options] == windows


def test_the_recheck_budget_leaves_a_whole_test_bound_inside_the_route_steps_limit():
    """Two copies of the route step's limit and of the per-test bound must not drift apart (#2834)."""
    import re

    text = (pmc.REPO_ROOT / ".github/workflows/post-merge.yml").read_text(encoding="utf-8")
    step = text.split("- name: Hand each red to its author", 1)[1].split("- name:", 1)[0]
    assert int(re.search(r"timeout-minutes: (\d+)", step).group(1)) * 60 == pmc.ROUTE_STEP_SECONDS
    assert f"--timeout={pmc.TEST_SECONDS}" in pmc.SESSION_FLAGS and "python scripts/post_merge_check.py route" in step
    assert 0 < pmc.RECHECK_SECONDS and pmc.RECHECK_SECONDS + pmc.TEST_SECONDS + 75 == pmc.ROUTE_STEP_SECONDS


def test_source_read_failure_is_not_module_removal(tmp_path, monkeypatch):
    def unavailable(*args, **kwargs):
        raise subprocess.CalledProcessError(128, args)
    monkeypatch.setattr(pmc, "_git", unavailable)
    with pytest.raises(subprocess.CalledProcessError):
        pmc.tree_source("bad-revision")("tests/t.py")
    monkeypatch.setattr(pmc, "REPO_ROOT", tmp_path)
    assert pmc.working_source("tests/missing.py") is None
    (tmp_path / "bad.py").write_bytes(b"\xff")
    with pytest.raises(UnicodeError):
        pmc.working_source("bad.py")


def test_the_runner_final_inventory_failure_is_not_a_green_selected_run(tmp_path):
    _selection(tmp_path)
    selection = json.loads((tmp_path / "selection.json").read_text(encoding="utf-8"))
    selection["outcome"] = "failure"  # the final floor check failed after both passes succeeded
    (tmp_path / "selection.json").write_text(json.dumps(selection), encoding="utf-8")
    for label in ("parallel", "serial"):
        _results(tmp_path / label, [_row(f"tests/{label}.py::test_ok")])
    result = pmc.classify(tmp_path, run=lambda *a: pytest.fail("no retry of the whole selection"), node=lambda: 0)
    assert not result["failed"] and "inventory/floor" in result["problems"][0]
    (tmp_path / "classification.json").write_text(json.dumps(result), encoding="utf-8")
    assert pmc.render_summary(tmp_path)[1]


def _identified(directory, reports, *, sha=HEAD, run_id="77", platform="Linux", collection=()):
    """A selected pass's export as tests/ci_evidence.py writes it: its reports with the run and system it names."""
    _results(directory, reports, collection=collection)
    data = json.loads((Path(directory) / "results.json").read_text(encoding="utf-8"))
    data.update(github={"sha": sha, "run_id": run_id, "run_attempt": "1"}, environment={"platform": platform})
    (Path(directory) / "results.json").write_text(json.dumps(data), encoding="utf-8")


def test_only_tests_this_run_passed_whole_at_its_head_on_its_system_spare_a_recheck(tmp_path):
    ids = ["tests/t.py::test_a", "tests/t.py::test_flaky", "tests/t.py::test_skip", "tests/s.py::test_serial",
           "tests/t.py::test_breach", "tests/t.py::test_not_selected", "tests/t.py::test_torn", "tests/u.py",
           "tests/u.py::test_whole", "tests/b.py::test_b", "tests/t.py::test_param",
           "tests/test_post_merge_check.py::test_of_a_module_these_passes_did_not_run", "tests/gone.py::test_x"]
    _selection(tmp_path)
    _identified(tmp_path / "parallel", [*_whole(ids[0]), _row(ids[1], "failed"), _row(ids[2], "skipped", "setup"),
                                        *_whole(ids[4], _xfail(ids[4])), _row(ids[6], phase="setup"), _row(ids[6]),
                                        *_whole(ids[8]), *_whole("tests/t.py::test_param[1]")], collection=["tests/b.py"])
    _identified(tmp_path / "serial", [*_whole(ids[3]), *_whole(ids[9])])
    # Kept without a recheck: an id whose own report was no whole pass (a failure, even one its retry passed, a skip,
    # an xfail, a test without its teardown, a test of a module that did not collect in another pass) and a test of a
    # module these passes ran nothing of, while its file exists. Left for the recheck: a test, module or function id
    # without a report in a module they ran, and a test whose file is gone.
    kept = dict.fromkeys([ids[1], ids[2], ids[4], ids[6], ids[9], ids[11]], "kept")
    assert pmc.same_run_verdicts(tmp_path, ids, head=HEAD, system="linux", run_id="77") == {
        ids[0]: "passed", ids[3]: "passed", ids[8]: "passed", **kept}
    for other in ({"sha": "e" * 40}, {"run_id": "78"}, {"platform": "Windows"}):
        _identified(tmp_path / "parallel", _whole(ids[0]), **other)
        _identified(tmp_path / "serial", _whole(ids[3]), **other)
        # a pass of another commit, run or system ran nothing here: only the existing module's test is kept
        assert pmc.same_run_verdicts(tmp_path, ids, head=HEAD, system="linux", run_id="77") == {ids[11]: "kept"}, other
    for label in ("parallel", "serial"):  # a pass killed before its export names no commit, run or system
        (tmp_path / label / "results.json").unlink()
        (tmp_path / label / "events.jsonl").write_text(json.dumps(
            {"event": "report", "nodeid": ids[0], "phase": "call", "outcome": "passed"}) + "\n", encoding="utf-8")
    assert pmc.same_run_verdicts(tmp_path, ids, head=HEAD, system="linux", run_id="77") == {ids[11]: "kept"}


def test_route_records_its_hand_over_before_the_recheck_and_reuses_this_runs_passes(tmp_path, monkeypatch, capsys):
    _selection(tmp_path)
    _identified(tmp_path / "parallel", _whole("tests/t.py::test_here"))
    seen = {}

    def fake_route(github, **kwargs):
        kwargs["checkpoint"]({"handed": "before the recheck"})
        seen["saved"] = json.loads((tmp_path / "routing.json").read_text(encoding="utf-8"))
        seen["same_run"] = kwargs["same_run"](["tests/t.py::test_here", "tests/t.py::test_other"])
        return {"comments": [], "created": [], "updated": [], "narrowed": [{"issue": 5, "ids": ["x"]}], "closed": [],
                "warnings": []}

    monkeypatch.setattr(pmc, "route", fake_route)
    monkeypatch.setenv("GITHUB_RUN_ID", "77")
    assert pmc.main(["route", "--evidence-dir", str(tmp_path), "--run-url", "u", "--head", HEAD]) == 0
    assert seen == {"saved": {"handed": "before the recheck"}, "same_run": {"tests/t.py::test_here": "passed"}}
    assert json.loads((tmp_path / "routing.json").read_text(encoding="utf-8"))["narrowed"] == [{"issue": 5, "ids": ["x"]}]
    assert "narrowed [5]" in capsys.readouterr().out


def test_the_summary_names_narrowed_issues_and_a_recheck_that_did_not_finish(tmp_path):
    (tmp_path / "classification.json").write_text(json.dumps({
        "failed": [], "flaky": [], "problems": [], "selection": {}}), encoding="utf-8")
    routing = {"comments": [], "created": [], "updated": [], "closed": [], "warnings": [],
               "narrowed": [{"issue": 2464, "ids": ["tests/t.py::a", "tests/t.py::b"]}], "complete": True}
    (tmp_path / "routing.json").write_text(json.dumps(routing), encoding="utf-8")
    text, red = pmc.render_summary(tmp_path, system="windows")
    assert not red and "updated: none; narrowed: #2464 (2 passing again); closed as green: none." in text
    assert "did not finish" not in text and "got no result" not in text
    (tmp_path / "routing.json").write_text(json.dumps({**routing, "rechecked": {"a": "passed", "b": "unknown"}}),
                                           encoding="utf-8")
    assert "- 1 tracked test(s) got no result in the recheck (it starts none after 225 s" in pmc.render_summary(
        tmp_path, system="windows")[0]
    (tmp_path / "routing.json").write_text(json.dumps({**routing, "narrowed": [], "complete": False}), encoding="utf-8")
    text, _ = pmc.render_summary(tmp_path, system="windows")
    assert "the recheck of the open issues did not finish" in text and "narrowed: #" not in text


def test_skipped_base_probe_is_not_a_good_revision(tmp_path, monkeypatch):
    monkeypatch.setattr(pmc, "_git", lambda *args, **kwargs: "")
    monkeypatch.setattr(pmc, "remove_worktree", lambda tree: None)
    nodeid = "tests/t.py::test_case"
    monkeypatch.setattr(pmc, "check_ids", lambda *a, **k: {nodeid: "skipped"})
    prober = pmc.Prober(tmp_path / "evidence", tmp_path / "work", HEAD)
    assert prober("a" * 40, [nodeid]) == {nodeid: "error"}
    assert pmc.bisect(["base", "head"], [nodeid], prober)[0]["verdict"] == "unattributed"


# --- attribute -------------------------------------------------------------------------------

CHAIN = ["base"] + [f"m{index}" for index in range(1, 9)]  # base, then eight merges; m8 is the head


def _prober(first_bad, *, errors=()):
    """A fake probe: an id fails from its first bad commit on (None: already at the base)."""
    calls = []

    def probe(sha, ids):
        calls.append((sha, sorted(ids)))
        index = CHAIN.index(sha)
        return {nodeid: "error" if sha in errors else
                "fail" if first_bad[nodeid] is None or index >= CHAIN.index(first_bad[nodeid]) else "pass"
                for nodeid in ids}
    return probe, calls


def test_ids_broken_by_one_merge_share_one_bisection():
    probe, calls = _prober({"a": "m3", "b": "m3", "c": "m3"})
    verdicts = pmc.bisect(CHAIN, ["c", "a", "b"], probe)
    assert {row["id"]: (row["verdict"], row["culprit"], row["last_good"]) for row in verdicts} == {
        nodeid: ("culprit", "m3", "m2") for nodeid in "abc"}
    assert calls[0] == ("base", ["a", "b", "c"]) and len(calls) == 4  # the base, then log2(8) probes
    assert all(ids == ["a", "b", "c"] for _, ids in calls)


def test_two_culprits_and_a_preexisting_red_are_told_apart():
    probe, calls = _prober({"early": "m1", "late": "m8", "old": None})
    verdicts = {row["id"]: row for row in pmc.bisect(CHAIN, ["early", "late", "old"], probe)}
    assert (verdicts["early"]["culprit"], verdicts["early"]["last_good"]) == ("m1", "base")
    assert (verdicts["late"]["culprit"], verdicts["late"]["last_good"]) == ("m8", "m7")
    assert verdicts["old"]["verdict"] == "pre-existing" and "culprit" not in verdicts["old"]
    assert all("old" not in ids for _, ids in calls[1:])  # a pre-existing red is never bisected


def test_the_probe_budget_leaves_the_rest_unattributed():
    probe, calls = _prober({"a": "m5"})
    verdicts = pmc.bisect(CHAIN, ["a"], probe, max_probes=2)
    assert len(calls) == 2 and verdicts[0]["verdict"] == "unattributed"
    assert "probe budget (2)" in verdicts[0]["why"]


def test_a_probe_that_could_not_run_attributes_nothing():
    probe, _ = _prober({"a": "m6"}, errors={"m4"})
    assert pmc.bisect(CHAIN, ["a"], probe)[0]["verdict"] == "unattributed"
    assert pmc.bisect(["base"], ["a"], probe)[0]["verdict"] == "pre-existing"  # nothing landed since the base


def _git(repo, *args):
    return subprocess.run(["git", *args], cwd=repo, capture_output=True, text=True, check=True).stdout.strip()


@pytest.mark.skipif(os.name == "nt", reason="the probe runner is POSIX CI only")
def test_attribute_probes_detached_worktrees_of_the_commits_and_removes_them(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "ouroboros")
    _git(repo, "config", "user.email", "t@example.invalid")
    _git(repo, "config", "user.name", "t")
    shas = []
    for value in ("1", "1", "2", "2"):  # the third commit changes the value the test checks
        (repo / "tests").mkdir(exist_ok=True)
        (repo / "tests" / "test_v.py").write_text("def test_value(): pass\n", encoding="utf-8")
        (repo / "VALUE").write_text(value + str(len(shas)), encoding="utf-8")
        _git(repo, "add", "-A")
        _git(repo, "commit", "-q", "-m", f"commit {len(shas)}")
        shas.append(_git(repo, "rev-parse", "HEAD"))
    monkeypatch.setattr(pmc, "REPO_ROOT", repo)
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    sized = "size tests/test_v.py::test_size"
    (evidence / "classification.json").write_text(json.dumps(
        {"failed": [{"id": "tests/test_v.py::test_value"}, {"id": pmc.NODE_LANE}, {"id": sized}],
         "size": {"base": shas[0], "mode": "approved"}}), encoding="utf-8")
    probed, measured = [], []

    def run(args, directory, *, cwd):
        probed.append(Path(cwd))
        red = (Path(cwd) / "VALUE").read_text(encoding="utf-8").startswith("2")
        _results(directory, [_row("tests/test_v.py::test_value", "failed" if red else "passed")])

    def size_session(directory, *, base, mode, cwd):  # each probe runs its own tree's size lane
        measured.append((base, mode, Path(cwd)))
        red = (Path(cwd) / "VALUE").read_text(encoding="utf-8").startswith("2")
        _results(directory, [_row("tests/test_v.py::test_size", "failed" if red else "passed")])
        return pmc.read_session(directory)

    monkeypatch.setattr(pmc, "run_pytest", run)
    monkeypatch.setattr(pmc, "size_session", size_session)
    monkeypatch.setattr(pmc, "run_node_lane", lambda cwd: int((Path(cwd) / "VALUE").read_text(encoding="utf-8").startswith("2")))
    args = type("Args", (), {"evidence_dir": evidence, "base": shas[0], "head": shas[-1], "max_probes": 16,
                             "work_dir": tmp_path})
    assert pmc.command_attribute(args) == 0
    record = json.loads((evidence / "attribution.json").read_text(encoding="utf-8"))
    assert {row["id"]: (row["verdict"], row["culprit"], row["last_good"]) for row in record["verdicts"]} == {
        nodeid: ("culprit", shas[2], shas[1]) for nodeid in ("tests/test_v.py::test_value", pmc.NODE_LANE, sized)}
    # The base, then the two commits between it and the head: each probe in its own worktree, all removed.
    assert [probe["sha"] for probe in record["probes"]] == shas[:3] and len(set(probed)) == 3
    # The size test is measured against the check's base with its growth mode, in the probed tree.
    assert measured == [(shas[0], "approved", path) for path in probed]
    assert record["commits"] == shas[1:] and not any(path.exists() for path in probed)
    assert _git(repo, "worktree", "list", "--porcelain").count("worktree ") == 1
    assert record["notes"] == [] and not any(probe["dependencies_differ"] for probe in record["probes"])


def test_a_probe_answers_a_size_test_from_the_size_check_of_its_tree(tmp_path, monkeypatch):
    monkeypatch.setattr(pmc, "_git", lambda *args, **kwargs: "")  # no worktree: the tree is the fake's
    monkeypatch.setattr(pmc, "remove_worktree", lambda tree: None)
    ids = [f"size tests/s.py::test_{name}" for name in ("red", "green", "new", "skip")]

    def size(tree, directory):
        _results(directory, [_row("tests/s.py::test_red", "failed"), _row("tests/s.py::test_green"),
                             _row("tests/s.py::test_skip", "skipped", "setup")])
        return pmc.read_session(directory)

    prober = pmc.Prober(tmp_path / "evidence", tmp_path / "work", HEAD, size=size,
                        run=lambda *a, **k: pytest.fail("a size test is not rerun as a plain test"))
    assert prober("a" * 40, ids) == {"size tests/s.py::test_red": "fail", "size tests/s.py::test_green": "pass",
                                     "size tests/s.py::test_new": "error",  # no report is not proof of removal
                                     "size tests/s.py::test_skip": "error"}  # it measured nothing there
    for unfinished in (lambda tree, directory: None, None):  # no size session there; a Prober without one
        prober = pmc.Prober(tmp_path / "evidence", tmp_path / "work", HEAD, size=unfinished)
        assert prober("a" * 40, ids[:1]) == {ids[0]: "error"}


# --- summary ---------------------------------------------------------------------------------

@pytest.mark.parametrize("verdict", ["culprit", "pre-existing", "unattributed"])
def test_attribution_explains_a_failed_check_without_changing_red(tmp_path, verdict):
    nodeid = "tests/t.py::test_red"
    (tmp_path / "classification.json").write_text(json.dumps({"complete": True,
        "failed": [{"id": nodeid, "why": "failed in call", "pass": "serial", "retry": "skipped"}],
        "flaky": [], "problems": []}), encoding="utf-8")
    (tmp_path / "attribution.json").write_text(json.dumps({"verdicts": [
        {"id": nodeid, "verdict": verdict}]}), encoding="utf-8")
    text, red = pmc.render_summary(tmp_path)
    assert red and "— RED" in text and verdict in text
    assert "does not establish the same cause" in text
    assert "red again on retry" not in text


def test_the_summary_is_red_only_for_failures_left_after_their_retry(tmp_path):
    text, red = pmc.render_summary(tmp_path)
    assert red and "did not finish" in text
    (tmp_path / "base.json").write_text(json.dumps({"base": "b" * 40, "head": HEAD, "why": "the branch tip"}), encoding="utf-8")
    (tmp_path / "classification.json").write_text(json.dumps({
        "failed": [], "flaky": [{"id": "tests/t.py::test_f", "why": "failed in call", "pass": "parallel"}],
        "problems": [], "selection": {"tier": "selected", "file_count": 3, "available": 40, "node": "success"}}), encoding="utf-8")
    text, red = pmc.render_summary(tmp_path, [("1" * 40, "Merge pull request #2146 from x"), ("2" * 40, "direct")])
    assert not red and "GREEN" in text and "2 commit(s) (#2146)" in text and "`tests/t.py::test_f`" in text
    (tmp_path / "classification.json").write_text(json.dumps({
        "failed": [{"id": "tests/t.py::test_a", "why": "failed in call", "pass": "serial"}], "flaky": [],
        "problems": [], "selection": {}}), encoding="utf-8")
    (tmp_path / "attribution.json").write_text(json.dumps({"verdicts": [
        {"id": "tests/t.py::test_a", "verdict": "culprit", "culprit": "c" * 40, "last_good": "b" * 40}]}), encoding="utf-8")
    (tmp_path / "routing.json").write_text(json.dumps({
        "comments": [2067], "created": [100], "updated": [], "closed": [5], "warnings": ["comment on #9: HTTP 403"],
        "handed": [{"ids": ["tests/t.py::test_a"], "pr": 2067, "issue": 100}]}), encoding="utf-8")
    text, red = pmc.render_summary(tmp_path)
    assert red and ("| `tests/t.py::test_a` | failed in call | culprit | `cccccccccccc` | `bbbbbbbbbbbb` "
                    "| PR #2067, issue #100 |") in text
    assert "closed as green: #5" in text and "**Warning:** comment on #9: HTTP 403" in text
    web = "https://github.com/o/r"
    text, _ = pmc.render_summary(tmp_path, [("1" * 40, "Merge pull request #2146 from x")], web)
    for link in (f"[`cccccccccccc`]({web}/commit/{'c' * 40})", f"PR [#2067]({web}/pull/2067)",
                 f"issue [#100]({web}/issues/100)", f"([#2146]({web}/pull/2146))", f"[#5]({web}/issues/5)"):
        assert link in text
    # A retry cut short by its timeout left only its provisional record: red, and said so.
    (tmp_path / "classification.json").write_text(json.dumps({"complete": False, "failed": [], "flaky": [],
                                                              "problems": []}), encoding="utf-8")
    text, red = pmc.render_summary(tmp_path)
    assert red and "the retry did not finish" in text


def test_each_leg_names_its_system_and_the_gates_it_ran(tmp_path, monkeypatch, capsys):
    (tmp_path / "classification.json").write_text(json.dumps({
        "failed": [{"id": SIZED, "why": "failed in call", "pass": "size"}], "flaky": [], "problems": [],
        "gates": ["lint", "size"], "size": {"base": "a" * 40, "mode": "approved", "decided": True, "approved": ["1" * 40]},
        "lint_checked": True, "size_checked": True,
        "selection": {"tier": "selected", "file_count": 3, "available": 40, "node": "success"}}), encoding="utf-8")
    text, red = pmc.render_summary(tmp_path)
    assert red and text.startswith("## Post-merge check on Linux — RED")
    assert "Gates: ruff F, the size check against `aaaaaaaaaaaa` (book growth: approved by the owner at `111111111111`)." in text
    (tmp_path / "classification.json").write_text(json.dumps({
        "failed": [], "flaky": [], "problems": [], "gates": [],
        "selection": {"tier": "selected", "file_count": 3, "available": 40, "node": "success", "deferred_full": True}}),
        encoding="utf-8")
    text, red = pmc.render_summary(tmp_path, system="windows")
    assert not red and text.startswith("## Post-merge check on Windows — GREEN")
    assert "Gates: none on this leg (ruff F and the size check run on the Linux leg)." in text
    assert ("A full escalation was deferred to the release matrix: this leg ran the contract floor and what the other "
            "changes select.") in text
    assert pmc.render_summary(tmp_path / "nothing", system="windows")[0].startswith("## Post-merge check on Windows — RED")
    (tmp_path / "classification.json").write_text(json.dumps({
        "failed": [{"id": "tests/t.py::test_a", "why": "failed in call", "pass": "serial"}], "flaky": [], "problems": [],
        "gates": ["size"], "size": {"base": "a" * 40, "mode": "block", "decided": False}, "selection": {}}), encoding="utf-8")
    text, _ = pmc.render_summary(tmp_path)
    assert "book growth: block, as the merges' labels were not read) (unchecked)." in text  # the gate never ran
    assert "A full escalation" not in text and "none on this leg" not in text
    monkeypatch.delenv("GITHUB_STEP_SUMMARY", raising=False)
    monkeypatch.delenv("GITHUB_REPOSITORY", raising=False)
    assert pmc.main(["summary", "--evidence-dir", str(tmp_path), "--os", "windows"]) == 1
    assert "::error title=Post-merge red on Windows::tests/t.py::test_a (failed in call)" in capsys.readouterr().out


# --- run_tests.py --keep-going ---------------------------------------------------------------

@pytest.fixture
def run_tests():
    spec = importlib.util.spec_from_file_location("run_tests_keep_going", REPO / "scripts" / "run_tests.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("keep_going", [False, True])
def test_keep_going_runs_every_lane_after_a_red_one(run_tests, monkeypatch, tmp_path, keep_going):
    from ouroboros import preflight_inventory, test_selection

    chosen = test_selection.TestSelection("selected", "base", ("tests/test_floor.py",), (), (), "policy", 3)
    monkeypatch.setattr(test_selection, "select_tests", lambda *a, **k: chosen)
    monkeypatch.setattr(preflight_inventory, "read_inventory", lambda root, label: ({}, ""))
    monkeypatch.setattr(preflight_inventory, "inventory_problem", lambda selection, inventory: "")
    monkeypatch.delenv("GITHUB_OUTPUT", raising=False)
    monkeypatch.delenv("OUROBOROS_PREFLIGHT_SERIAL", raising=False)  # two passes, not the legacy single one
    for node, parallel in ((1, 0), (0, 1)):  # a red Node lane; a red parallel pass
        ran = []
        monkeypatch.setattr(run_tests, "_node_lane", lambda: node)
        monkeypatch.setattr(run_tests, "_run", lambda label, argv, cwd: ran.append(label) or
                            (parallel if label == "parallel" else 0))
        evidence = tmp_path / f"{keep_going}-{node}"
        code = run_tests.main(["--selected", "--evidence-dir", str(evidence), *(["--keep-going"] * keep_going)])
        record = json.loads((evidence / "selection.json").read_text(encoding="utf-8"))
        assert code == 1 and record["outcome"] == "failure"
        assert record["node"] == ("failure" if node else "success")
        if keep_going:
            assert ran == ["warm", "parallel", "serial"]
        else:  # the unchanged default stops at the first red lane
            assert ran == ([] if node else ["warm", "parallel"])
    with pytest.raises(SystemExit):
        run_tests.main(["--keep-going", "tests/test_x.py"])
