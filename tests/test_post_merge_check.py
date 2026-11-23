"""The post-merge check: base choice, book-growth mode, one retry, the size check, bisection to the
culprit merge and the summary (the hand-over itself: tests/test_post_merge_routing.py).

Deterministic: GitHub is a fake, test sessions are fakes that write tests/ci_evidence.py's files,
and git is a throwaway local repository without remotes, beside one read-only `git grep` of this
checkout for the modules of its size check.
"""
from __future__ import annotations

from fnmatch import fnmatchcase
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import urllib.error

import pytest

from scripts import post_merge_check as pmc
from scripts.post_merge_bisect import PROTECTED
from ouroboros.update_qualification import CONTINUOUS_STEP, MEASURED_STEP, RANGE_TITLE

REPO = Path(__file__).resolve().parents[1]
HEAD = "f" * 40


def _row(nodeid, outcome="passed", phase="call"):
    return {"nodeid": nodeid, "phase": phase, "outcome": outcome, "is_subtest": False}


def _results(directory, reports, *, exit_code=None, collection=(), selected=None, skips=None, github=None,
             collected=None):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    failed = any(row["outcome"] == "failed" for row in reports) or collection
    result = {
        "format": 1, "session_exit_code": (1 if failed else 0) if exit_code is None else exit_code,
        # `collected`: what an xdist controller exports, the tests its workers collected, whether or not they reported
        "reports": reports, "tests_collected": collected if collected is not None else len(
            selected if selected is not None else {r["nodeid"] for r in reports}),
        "collection_failures": [{"nodeid": path, "outcome": "failed"} for path in collection],
    }
    if selected is not None:  # what tests/ci_evidence.py's session left to run after every filter
        result["selected_nodeids"] = list(selected)
    if skips is not None:  # the collectors that skipped at collection; a producer older than the field names none
        result["collection_skips"] = list(skips)
    if github is not None:
        result["github"] = github
    (directory / "results.json").write_text(json.dumps(result), encoding="utf-8")
    return result["session_exit_code"]


def _selection(evidence, *, parallel="success", serial="success", node="success", inventory=None, reports=None, base="b" * 40):
    evidence.mkdir(parents=True, exist_ok=True)
    outcomes = {"parallel": parallel, "serial": serial}
    if inventory is None:
        inventory = {}
        if reports is None:
            reports = {"parallel": _whole("tests/t.py::test_ok")}
        for label, outcome in outcomes.items():
            if outcome == "skipped":
                continue
            rows = reports.get(label, [])
            ids = [r["nodeid"] for r in rows if r["phase"] == "setup"]
            code = (1 if outcome == "failure" else 0) if ids else 5
            inventory[label] = {"collection_reports": 1, "expected": ids, "exitcode": code,
                                "executed": [r["nodeid"] for r in rows if r["phase"] == "teardown"]}
            _results(evidence / label, rows, selected=ids, exit_code=code)
    (evidence / "selection.json").write_text(json.dumps({
        "tier": "selected", "file_count": 3, "available": 40, "base": base, "node": node,
        "outcomes": outcomes, "inventory": inventory,
        "outcome": "success" if all(value == "success" for value in (*outcomes.values(), node)) else "failure",
    }), encoding="utf-8")


# --- base ------------------------------------------------------------------------------------

def _run(sha, conclusion="success", status="completed", number=1, attempt=1):
    return {"id": number, "run_attempt": attempt, "head_sha": sha, "status": status, "conclusion": conclusion,
            "display_title": RANGE_TITLE, "head_branch": "ouroboros", "event": "push"}


def _job(run, system, marker="success", *, base="1" * 40, scope="chain"):
    return {"name": f"post-merge ({system}) from {base}", "run_id": run["id"], "run_attempt": run["run_attempt"],
            "head_sha": run["head_sha"], "status": "completed", "conclusion": run["conclusion"],
            "steps": [{"name": CONTINUOUS_STEP if scope == "chain" else MEASURED_STEP,
                       "status": "completed", "conclusion": marker}]}


def _window_job(run, window):
    source = "confirmed" if window["continuous"] else "provisional"
    return {**_job(run, "linux"), "name": "window", "conclusion": "success",
            "steps": [{"name": f"Post-merge {source} base {window['base']}", "status": "completed", "conclusion": "success"}]}


def test_the_base_is_the_head_of_the_newest_check_that_ran_on_both_systems():
    ancestors = {"a" * 40, "c" * 40}
    runs = [_run(HEAD, number=9), _run("d" * 40, status="in_progress", conclusion=None, number=8),
            _run("e" * 40, conclusion="cancelled", number=7), _run("x" * 40, number=6),
            _run("c" * 40, conclusion="failure", number=5, attempt=2), _run("a" * 40, number=4)]

    def get(path):
        if path == f"{pmc.POST_MERGE_RUNS}?branch=ouroboros&head_sha={'c' * 40}&per_page=100&page=1":
            return {"total_count": 1, "workflow_runs": [runs[4]]}  # the one exact list of the qualified head's checks
        assert path == "actions/runs/5/attempts/2/jobs?per_page=100&page=1"
        return {"total_count": 101, "jobs": [_job(runs[4], "linux"), _job(runs[4], "windows"), *[{}] * 98]}

    base, why, continuous = pmc.choose_base(runs, "a" * 40, HEAD, get=get, ancestor=lambda sha, head: sha in ancestors,
                                            parent=lambda head: pytest.fail("no fallback"))
    assert continuous and base == "c" * 40 and "run 5, attempt 2, failure" in why


def test_a_partial_classification_keeps_the_selected_step_disagreement(tmp_path, monkeypatch):
    _selection(tmp_path, parallel="failure", reports={"parallel": _whole("tests/t.py::test_red",
                                                                        _row("tests/t.py::test_red", "failed"))})
    def interrupted(*a, **kw):
        raise RuntimeError("retry stopped after the checkpoint")
    monkeypatch.setattr(pmc.subprocess, "call", interrupted)
    with pytest.raises(RuntimeError, match="after the checkpoint"):
        pmc.main(["classify", "--evidence-dir", str(tmp_path), "--head", HEAD, "--selected-outcome", "skipped"])
    record = json.loads((tmp_path / "classification.json").read_text())
    assert not record["selection_executed"] and record["failed"][0]["retry"] == "not run"
    assert "selected workflow step was skipped" in record["problems"][0]


@pytest.mark.parametrize("gap", ["no marker", "old marker", "skipped", "other attempt", "other run", "other head",
                                 "other base", "push only", "unavailable"])
def test_the_base_reaches_past_a_run_without_both_execution_witnesses(tmp_path, monkeypatch, gap):
    newer = _run("b" * 40, number=30, attempt=2)
    older = _run("a" * 40, number=29)
    windows = _job(newer, "windows")
    if gap == "no marker":
        windows["steps"] = [{"name": "Upload the evidence", "conclusion": "success"}]
    elif gap == "old marker":
        windows["steps"][0]["name"] = "Selected test execution confirmed"
    elif gap == "skipped":
        windows["steps"][0]["conclusion"] = "skipped"
    elif gap == "other attempt":
        windows["run_attempt"] = 1
    elif gap == "other run":
        windows["run_id"] = 28
    elif gap == "other head":
        windows["head_sha"] = older["head_sha"]
    elif gap == "other base":
        windows = _job(newer, "windows", base="2" * 40)
    elif gap == "push only":
        windows = _job(newer, "windows", scope="push")

    def get(self, path):
        if path.startswith("actions/workflows/post-merge.yml/runs?"):
            if "head_sha=" in path:  # the one exact list of the qualified head's checks
                assert path == f"{pmc.POST_MERGE_RUNS}?branch=ouroboros&head_sha={older['head_sha']}&per_page=100&page=1"
                return {"total_count": 1, "workflow_runs": [older]}
            return {"total_count": 2, "workflow_runs": [newer, older]}
        if path == "actions/runs/30/attempts/2/jobs?per_page=100&page=1":
            if gap == "unavailable":
                raise urllib.error.HTTPError(path, 404, "Not Found", None, None)
            return {"total_count": 2, "jobs": [_job(newer, "linux", scope="push" if gap == "push only" else "chain"), windows]}
        if path == "actions/runs/30/attempts/1/jobs?per_page=100&page=1":  # no window job chose its input either
            return {"total_count": 0, "jobs": []}
        assert path == "actions/runs/29/attempts/1/jobs?per_page=100&page=1"
        return {"total_count": 2, "jobs": [_job(older, "linux"), _job(older, "windows")]}

    monkeypatch.setattr(pmc.GitHub, "get", get)
    monkeypatch.setattr(pmc, "is_ancestor", lambda *_: True)
    monkeypatch.setenv("BEFORE", newer["head_sha"])
    monkeypatch.setenv("GITHUB_OUTPUT", str(tmp_path / "output"))
    assert pmc.main(["base", "--head", HEAD, "--evidence-dir", str(tmp_path)]) == 0
    receipt = json.loads((tmp_path / "base.json").read_text(encoding="utf-8"))
    assert receipt["base"] == older["head_sha"] and "run 29, attempt 1" in receipt["why"]
    # A bare 404 on the newest run's job list proves no loss, but withholds confirmation: the established base is
    # kept and the range stays provisional.
    assert receipt["continuous"] == (gap != "unavailable")
    assert ("; earlier history unreadable" in receipt["why"]) == (gap == "unavailable")


def test_an_oversized_handover_reaches_the_next_window_by_its_exact_receipt(tmp_path, monkeypatch):
    first, head, following = "a" * 40, "c" * 40, "f" * 40
    pending = [f"tests/test_leaf.py::test_case[{n:03d}]" for n in range(40)]
    window = {"base": first, "head": head, "continuous": True}
    (tmp_path / "base.json").write_text(json.dumps(window), encoding="utf-8")
    (tmp_path / "classification.json").write_text(json.dumps({
        "base": first, "head": head, "selection_executed": True, "failed": [],
    }), encoding="utf-8")
    owed = [{"head": head, "run": 7, "attempt": 1, "base": first, "ids": pending}]
    (tmp_path / "routing.json").write_text(json.dumps({"os": "windows", "untracked": pending, "awaiting_handoff": owed}),
                                           encoding="utf-8")
    output = tmp_path / "output"
    monkeypatch.setenv("GITHUB_OUTPUT", str(output))
    monkeypatch.setenv("GITHUB_RUN_ID", "7")
    monkeypatch.setenv("GITHUB_RUN_ATTEMPT", "1")
    monkeypatch.delenv("GITHUB_STEP_SUMMARY", raising=False)
    monkeypatch.setattr(pmc, "merges", lambda *_: [])
    assert pmc.main(["summary", "--evidence-dir", str(tmp_path), "--os", "windows"]) == 1
    emitted = dict(line.split("=", 1) for line in output.read_text(encoding="utf-8").splitlines())
    assert emitted["handoff_file"] == "yes" and emitted["window_scope"] == "push"
    ref = json.loads(emitted["untracked"])["artifact"]
    assert len(emitted["untracked"]) < 800 and (ref["run"], ref["attempt"]) == (7, 1)

    run = _run(head, "failure", number=7)
    windows = _job(run, "windows", base=first, scope="push")
    windows["steps"].append({"name": "Post-merge untracked " + emitted["untracked"], "status": "completed",
                             "conclusion": "success"})
    jobs = [_window_job(run, window), _job(run, "linux", base=first), windows]

    def read(run_id, attempt, part, digest):
        saved = (tmp_path / "handoff.json").read_bytes()
        assert (run_id, attempt, part, digest) == (7, 1, "windows", pmc.hashlib.sha256(saved).hexdigest())
        return json.loads(saved)

    gap = {"run": 7, "attempt": 1, "part": "windows", "base": first}
    unavailable = urllib.error.HTTPError("artifact", 503, "Service Unavailable", None, None)
    for fault in (None, FileNotFoundError("expired"), unavailable):
        replay = {}
        def handoff(*args):
            if fault is not None:
                raise fault
            return read(*args)
        base, _, continuous = pmc.choose_base([run], head, following, get=lambda _: {"total_count": 3, "jobs": jobs},
                                              ancestor=lambda a, b: a < b, replay=replay, handoff=handoff)
        # An expired receipt is a labelled gap, and this check measures that range again. One that could not be read
        # now says nothing about what it holds, so the window it seeds is no confirmed range.
        assert base == first and continuous == (fault is not unavailable)
        assert replay["awaiting_handoff"] == ({"windows": owed} if fault is None else {})
        assert replay["history_gaps"] == ([gap] if isinstance(fault, FileNotFoundError) else [])
        assert replay["history_read"] is continuous


def test_an_explicit_initial_window_seeds_the_next_push_across_run_and_job_pages(tmp_path, monkeypatch):
    initial, before, head = "1" * 40, "2" * 40, "3" * 40
    seed = {**_run(head, number=7, attempt=2), "event": "workflow_dispatch"}
    env = {"GITHUB_SHA": head, "GITHUB_REPOSITORY": "razzant/ouroboros", "GITHUB_REF": "refs/heads/ouroboros",
           "GITHUB_EVENT_NAME": "workflow_dispatch", "INITIAL_BASE": initial}
    args = type("Args", (), {"head": None, "evidence_dir": tmp_path})
    monkeypatch.setattr(pmc, "is_ancestor", lambda a, b: a in (head, initial, before) and a != b)
    monkeypatch.setattr(pmc.GitHub, "get", lambda *a: {"total_count": 0, "workflow_runs": []})
    assert pmc.command_base(args, env) == 0
    first = json.loads((tmp_path / "base.json").read_text())
    assert first["base"] == initial and first["continuous"]
    path = pmc.POST_MERGE_RUNS + "?per_page=100&page="
    jobs = "actions/runs/7/attempts/2/jobs?per_page=100&page="
    pages = {
        path + "1": {"total_count": 201, "workflow_runs": [{"display_title": "old"}] * 100},
        path + "2": {"total_count": 201, "workflow_runs": [seed, *[{}] * 99]},
        jobs + "1": {"total_count": 101, "jobs": [_job(seed, "linux", base=first["base"]), *[{}] * 99]},
        jobs + "2": {"total_count": 101, "jobs": [_job(seed, "windows", base=first["base"])]},
        f"{pmc.POST_MERGE_RUNS}?branch=ouroboros&head_sha={head}&per_page=100&page=1": {
            "total_count": 1, "workflow_runs": [seed]},  # the one exact list of the qualified head's checks
    }
    monkeypatch.setattr(pmc.GitHub, "get", lambda self, url: pages.pop(url))
    assert pmc.command_base(args, {"GITHUB_SHA": HEAD, "BEFORE": before}) == 0
    second = json.loads((tmp_path / "base.json").read_text())
    assert not pages and second["base"] == head and second["continuous"]


@pytest.mark.parametrize("fault", ["before selection", "missing Windows", "cancelled dispatch", "window cleanup failed",
                                   "wrong attempt", "invalid base"])
def test_first_dispatch_keeps_its_base_without_certifying_a_missing_leg(fault):
    from ouroboros.update_qualification import post_merge_checks

    base, missed, head, docs = (str(i) * 40 for i in range(1, 5))
    run = {**_run(head, conclusion="cancelled" if fault == "cancelled dispatch" else "failure"),
           "event": "workflow_dispatch"}
    window = _window_job(run, {"base": base, "continuous": True})
    jobs = [window, _job(run, "linux", base=base)]
    if fault == "wrong attempt":
        window["run_attempt"] += 1
    if fault == "before selection":
        jobs.append(_job(run, "windows", marker="skipped", base=base))
    if fault == "window cleanup failed":
        window["conclusion"] = "failure"
    if fault in ("cancelled dispatch", "window cleanup failed"):
        jobs = [window]
    if fault == "invalid base":
        jobs[0] = _window_job(run, {"base": head, "continuous": True})

    def get(path):
        return ({"workflow_runs": [run], "total_count": 1} if "/workflows/" in path
                else {"jobs": jobs, "total_count": len(jobs)})

    # Only the previous starting point can be reused, including when repeating a dispatch at its own SHA.
    for tip, initial in ((docs, ""), (head, missed)):
        following, _, confirmed = pmc.choose_base([run], head, tip, get=get, initial=initial,
                                                  ancestor=lambda a, b: a < b, parent=lambda _: head)
        if fault in ("wrong attempt", "invalid base"):
            assert following == (initial or head) and confirmed == bool(initial)
        else:
            assert confirmed and following == base
    fact = post_merge_checks({"base_sha": base, "commits": [{"sha": head}, {"sha": missed}]},
                             "https://github.com/test/repo", get_json=get)
    assert fact["missing_shas"] == [missed] and not fact["runs"][0].get("covered_shas")


@pytest.mark.parametrize("windows", ["failure", "success"])
def test_a_rerun_of_failed_jobs_replays_the_base_its_first_attempt_chose_and_certifies_nothing(windows):
    """Re-running only the failed Windows leg of the first dispatch lists that leg alone in the new attempt: its window
    job and Linux leg stay in the first attempt. The next push replays the dispatch's confirmed base instead of starting
    at its head, and no attempt joins one attempt's Linux leg to another's Windows leg."""
    from ouroboros.update_qualification import post_merge_checks

    base, missed, head, docs = (str(i) * 40 for i in range(1, 5))
    run = {**_run(head, conclusion=windows, attempt=2), "event": "workflow_dispatch"}
    first = {**run, "run_attempt": 1, "conclusion": "failure"}
    attempts = {1: [_window_job(first, {"base": base, "continuous": True}), _job(first, "linux", base=base),
                    _job(first, "windows", marker="skipped", base=base)],
                2: [_job(run, "windows", marker="success" if windows == "success" else "skipped", base=base)]}
    requests = []

    def get(path):
        requests.append(path)
        if "/workflows/" in path:
            return {"workflow_runs": [run], "total_count": 1}
        jobs = attempts[int(path.split("/")[4])]
        return {"jobs": jobs, "total_count": len(jobs)}

    following, why, confirmed = pmc.choose_base([run], head, docs, get=get, ancestor=lambda a, b: a < b,
                                                parent=lambda _: pytest.fail("no fallback to the dispatch's head"))
    assert (following, confirmed) == (base, True) and "retained starting base of run 1, attempt 1" in why
    assert requests == [f"actions/runs/1/attempts/{n}/jobs?per_page=100&page=1" for n in (2, 1)]
    fact = post_merge_checks({"base_sha": base, "commits": [{"sha": head}, {"sha": missed}]}, "https://github.com/test/repo",
                             get_json=lambda url: get(url.removeprefix("https://api.github.com/repos/test/repo/")))
    assert fact["missing_shas"] == [missed] and not fact["runs"][0].get("covered_shas")


@pytest.mark.parametrize("conclusion,outage", [("failure", False), ("failure", True), ("cancelled", False)])
def test_unseeded_red_replays_the_earliest_known_base_without_promoting_it(conclusion, outage):
    base, before, docs = (str(i) * 40 for i in range(1, 4))
    run = _run(before, conclusion=conclusion)
    jobs = [_window_job(run, {"base": base, "continuous": False}),
            _job(run, "linux", base=base, scope="push"), _job(run, "windows", base=base, scope="push")]
    requests = []

    def get(path):
        requests.append(path)
        return {"jobs": jobs, "total_count": len(jobs)}

    def runs():
        yield run
        if outage:
            raise OSError("older history offline")

    following, _, confirmed = pmc.choose_base(runs(), before, docs, get=get, ancestor=lambda a, b: a < b)
    assert following == base and not confirmed
    assert requests == ["actions/runs/1/attempts/1/jobs?per_page=100&page=1"]


@pytest.mark.parametrize("red_start", ["same", "earlier"])
def test_a_green_provisional_check_settles_the_window_it_measured(red_start):
    earlier, base, red_head, green_head, docs = (str(i) * 40 for i in range(1, 6))
    red, green = _run(red_head, conclusion="failure", number=1), _run(green_head, number=2)
    starts = {1: base if red_start == "same" else earlier, 2: base}
    jobs = {run["id"]: [_window_job(run, {"base": starts[run["id"]], "continuous": False}),
                        *(_job(run, leg, base=starts[run["id"]], scope="push") for leg in ("linux", "windows"))]
            for run in (red, green)}

    def get(path):
        return {"jobs": jobs[int(path.split("/")[2])], "total_count": 3}

    following, _, confirmed = pmc.choose_base([green, red], green_head, docs, get=get, ancestor=lambda a, b: a < b)
    # The green check re-measured the red one's window on both systems; a red window starting earlier is still a gap.
    assert (following, confirmed) == ((green_head, False) if red_start == "same" else (earlier, False))


@pytest.mark.parametrize("checkpoint", [False, True])
def test_contiguous_provisional_greens_settle_the_older_red_without_certifying_earlier_history(checkpoint):
    base, red, repaired, last, head = [str(i) * 40 for i in range(1, 6)]
    runs = [_run(last, number=3), _run(repaired, number=2), _run(red, "failure", number=1)]
    starts, asked = {3: repaired, 2: base, 1: base}, []

    def get(path):
        run = runs[3 - int(path.split("/")[2])]
        asked.append(run["id"])
        start = starts[run["id"]]
        window = _window_job(run, {"base": start, "continuous": False})
        if checkpoint and run == runs[0]:
            window["steps"].append({"name": 'Post-merge replay input {"history_read":true,"awaiting_handoff":{}}',
                                    "status": "completed", "conclusion": "success"})
        jobs = [window, *(_job(run, leg, base=start, scope="push") for leg in ("linux", "windows"))]
        return {"jobs": jobs, "total_count": len(jobs)}

    following, _, confirmed = pmc.choose_base(runs, last, head, get=get, ancestor=lambda a, b: a < b)
    assert following == last and not confirmed
    # A provisional window that read its history whole is a checkpoint, no boundary: only a qualified descendant closes
    # the walk behind it, so the older runs are read whether or not the newest one carries that record.
    assert asked == [3, 2, 1]


def test_push_runs_from_before_range_receipts_end_the_search():
    initial, before, head = (str(i) * 40 for i in range(1, 4))
    legacy = {**_run(before), "display_title": "Merge pull request #1 from someone/change"}

    def runs():
        yield legacy
        pytest.fail("history older than range receipts was read")

    for start, expected in (("", (before, False)), (initial, (initial, True))):
        base, _, continuous = pmc.choose_base(runs(), before, head, get=lambda path: pytest.fail(f"read {path}"),
                                              ancestor=lambda a, b: a < b, initial=start)
        assert (base, continuous) == expected


@pytest.mark.parametrize("fault", ["short page", "missing metadata", "offline", "other branch", "other repo", "push", "side parent"])
def test_initialization_cannot_hide_missing_history_or_cross_dispatch_authority(tmp_path, monkeypatch, fault):
    env = {"GITHUB_SHA": HEAD, "GITHUB_REPOSITORY": "razzant/ouroboros", "GITHUB_REF": "refs/heads/ouroboros",
           "GITHUB_EVENT_NAME": "workflow_dispatch", "INITIAL_BASE": "b" * 40}
    env.update({"other branch": {"GITHUB_REF": "refs/heads/feature"}, "other repo": {"GITHUB_REPOSITORY": "fork/r"},
                "push": {"GITHUB_EVENT_NAME": "push"}}.get(fault, {}))
    monkeypatch.setattr(pmc, "is_ancestor", lambda *_: fault != "side parent")
    def get(self, url):
        if fault == "offline":
            raise OSError("offline")
        if fault == "missing metadata":
            return {"total_count": 1, "workflow_runs": [{"display_title": RANGE_TITLE}]}
        assert fault == "short page"
        return {"total_count": 101, "workflow_runs": []}
    monkeypatch.setattr(pmc.GitHub, "get", get)
    args = type("Args", (), {"head": None, "evidence_dir": tmp_path})
    if fault in ("short page", "missing metadata", "offline"):
        assert pmc.command_base(args, env) == 0
        window = json.loads((tmp_path / "base.json").read_text())
        assert window["base"] == env["INITIAL_BASE"] and not window["continuous"]
    else:
        assert pmc.command_base(args, env) != 0 and not (tmp_path / "base.json").exists()


def test_a_branch_without_the_recovery_base_can_still_check_the_push():
    parent = lambda head: "p" * 40  # noqa: E731
    ancestor = lambda sha, head: sha == "a" * 40  # noqa: E731
    base, why, continuous = pmc.choose_base([], "a" * 40, HEAD, ancestor=ancestor, parent=parent)
    assert base == "a" * 40 and not continuous and "tip before this push" in why
    for before in ("", pmc.ZERO_SHA, "n" * 40):  # missing, a new branch, not an ancestor
        base, why, continuous = pmc.choose_base([], before, HEAD, ancestor=ancestor, parent=parent)
        assert base == "p" * 40 and not continuous and "first parent" in why
    # A comparison Git cannot make names no tip either: the push alone is checked, from its first parent.
    for fault in (subprocess.TimeoutExpired("git", 300), OSError("cannot start git")):
        def stuck(sha, head, fault=fault):
            raise fault
        base, why, continuous = pmc.choose_base([], "a" * 40, HEAD, ancestor=stuck, parent=parent)
        assert base == "p" * 40 and not continuous and "first parent" in why


@pytest.mark.parametrize("unreadable", ["runs", "jobs"])
def test_the_base_command_survives_an_api_error_and_publishes_its_choice(tmp_path, monkeypatch, capsys, unreadable):
    def broken(self, path):
        if unreadable == "jobs" and path.startswith("actions/workflows/"):
            return {"total_count": 1, "workflow_runs": [_run("c" * 40)]}
        raise urllib.error.HTTPError(path, 503, "Service Unavailable", None, None)

    monkeypatch.setattr(pmc.GitHub, "get", broken)
    monkeypatch.setattr(pmc, "is_ancestor", lambda sha, head: True)
    output = tmp_path / "github-output"
    environ ={"GITHUB_SHA": HEAD, "BEFORE": "a" * 40, "GITHUB_OUTPUT": str(output), "GITHUB_REPOSITORY": "o/r"}
    parsed = type("Args", (), {"head": None, "evidence_dir": tmp_path / "evidence"})
    assert pmc.command_base(parsed, environ) == 0
    outputs = dict(line.split("=", 1) for line in output.read_text(encoding="utf-8").splitlines())
    receipt = json.loads((tmp_path / "evidence" / "base.json").read_text(encoding="utf-8"))
    assert outputs["base"] == receipt["base"] == "a" * 40 and not receipt["continuous"]
    assert json.loads(outputs["window"]) == receipt
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


def test_a_recheck_retires_a_red_only_by_a_whole_pass_while_the_bisection_reader_keeps_the_legacy_call(tmp_path):
    """A call report alone (a session that names no selection: a journal, an older producer) is a pass for the bisection
    reader, as it always was. The recheck that takes a red off its issue needs the setup, call and teardown of the exact
    id from a session that names its selection or reported every test it collected; a test still running, red or skipped
    keeps its answer, and a journal alone, which names neither, retires none."""
    ids = [f"tests/t.py::test_{name}" for name in ("whole", "call_only", "running", "red", "skipped")]
    whole, call_only, running, red, skipped = ids

    def report(*rows):
        return [{"event": "report", **row} for row in rows]

    events = [{"event": "start", "nodeid": whole}, *report(*_whole(whole)), {"event": "finish", "nodeid": whole},
              {"event": "start", "nodeid": call_only}, *report(_row(call_only)),
              {"event": "finish", "nodeid": call_only},
              {"event": "start", "nodeid": running}, *report(*_whole(running)[:2]),
              *report(*_whole(red, _row(red, "failed"))),
              *report(_row(skipped, "skipped", "setup"), _row(skipped, phase="teardown"))]

    def killed(args, directory):  # no final export and no selection: only its journal
        Path(directory).mkdir(parents=True, exist_ok=True)
        (Path(directory) / "events.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in events), encoding="utf-8")

    killed(None, tmp_path / "journal")
    assert pmc.outcomes(pmc.read_session(tmp_path / "journal"), ids) == dict(zip(ids, [
        "passed", "passed", "failed", "failed", "skipped"]))
    # The recheck reads that journal and retires nothing from it: the journal names no selection and counted nothing, so a
    # sibling of one public id that never started would leave no row there. Its reds and skips stand.
    source = dict.fromkeys(["tests/t.py"], "").get
    found = pmc.recheck_ids(ids, tmp_path / "recheck", read_source=source, run=killed)
    assert found == dict(zip(ids, ["unknown", "unknown", "failed", "failed", "skipped"]))

    def exported(args, directory):  # the same tests in a final export that names its selection
        rows = [*_whole(whole), _row(call_only), *_whole(running)[:2], *_whole(red, _row(red, "failed")),
                _row(skipped, "skipped", "setup"), _row(skipped, phase="teardown")]
        return _results(directory, rows, selected=ids)

    found = pmc.recheck_ids(ids, tmp_path / "final", read_source=source, run=exported)
    assert found == dict(zip(ids, ["passed", "unknown", "unknown", "failed", "skipped"]))
    for counted, verdict in ((1, "passed"), (2, "unknown")):  # an xdist export names none: it counts only if complete
        def counting(args, directory, counted=counted):
            return _results(directory, _whole(whole), collected=counted)

        assert pmc.recheck_ids([whole], tmp_path / f"xdist{counted}", read_source=source, run=counting) == {whole: verdict}


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
    # A parameter's own `::` or `@` names no class or function: the container stays the test.
    structured = ["tests/test_a.py::test_param[x@y::z]", "tests/test_a.py::TestK::test_m[a::b@c]"]
    args, _ = pmc.container_args(ids[:5] + structured, sources.get)
    assert args == ["tests/test_a.py::TestInherited", "tests/test_a.py::TestK::test_m",
                    "tests/test_a.py::test_param", "tests/test_a.py::test_plain"]


# --- classify --------------------------------------------------------------------------------

def test_one_retry_separates_failed_from_flaky(tmp_path):
    flaky, broken, setup = "tests/t.py::test_flaky", "tests/t.py::test_broken", "tests/s.py::test_skips_now"
    _selection(tmp_path, parallel="failure", serial="failure", reports={
        "parallel": [*_whole(flaky, _row(flaky, "failed")), *_whole(broken, _row(broken, "failed")),
                     *_whole("tests/t.py::test_fine")],
        "serial": [_row(setup, "failed", "setup"), _row(setup, phase="teardown"), *_whole("tests/s.py::test_ok")]})
    seen = []

    def retry(args, directory):
        seen.append((list(args), Path(directory)))
        return _results(directory, [*_whole(flaky), *_whole(broken, _row(broken, "failed")),
                                    _row(setup, "skipped", "setup"), _row(setup, phase="teardown")])

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


def test_a_module_that_does_not_collect_again_leaves_the_retry_complete_and_the_other_outcomes_kept(tmp_path):
    """The retry runs on past a module that does not import (--continue-on-collection-errors, exit 1): that module
    is FAILED and goes to bisection, while a red that ran whole and passed on the same retry is FLAKY."""
    broken, flaky = "tests/test_broken.py", "tests/t.py::test_flaky"
    _selection(tmp_path, parallel="failure", reports={"parallel": _whole(flaky, _row(flaky, "failed"))})
    _results(tmp_path / "parallel", _whole(flaky, _row(flaky, "failed")), collection=[broken], selected=[flaky])
    retried = []

    def retry(ids, directory):
        retried.append(list(ids))
        return _results(directory, _whole(flaky), collection=[broken], selected=[flaky])

    record = pmc.classify(tmp_path, run=retry, node=lambda: 0)
    assert retried == [[flaky, broken]] and record["complete"] is True
    assert [(row["id"], row["retry"]) for row in record["failed"]] == [(broken, "failed")]
    assert [row["id"] for row in record["flaky"]] == [flaky]
    assert not any("retry" in problem for problem in record["problems"])


@pytest.mark.parametrize("initial, later", [("collection", "call"), ("call", "collection"),
                                             ("call", "setup"), ("call", "teardown")])
def test_retry_does_not_confirm_a_different_phase_or_collector(tmp_path, monkeypatch, initial, later):
    module = "tests/t.py"
    test = module + "::test_case"
    initial_id = module if initial == "collection" else test
    later_id = module if later == "collection" else test
    _selection(tmp_path, parallel="failure", reports={"parallel": _whole(test, _row(test, "failed"))})
    if initial == "collection":
        _results(tmp_path / "parallel", [], collection=[module], selected=[])

    def retry(ids, directory):
        assert list(ids) == [initial_id]
        if later == "collection":
            return _results(directory, [], collection=[module], selected=[])
        if later == "setup":
            reports = [_row(test, "failed", "setup"), _row(test, phase="teardown")]
        elif later == "teardown":
            reports = [*_whole(test)[:-1], _row(test, "failed", "teardown")]
        else:
            reports = _whole(test, _row(test, "failed"))
        return _results(directory, reports, selected=[test])

    record = pmc.classify(tmp_path, run=retry)
    assert record["complete"] and record["flaky"] == []
    found = {row["id"]: row for row in record["failed"]}
    assert set(found) == {initial_id, later_id} and found[initial_id]["retry"] == "unknown"
    initial_why = "did not collect" if initial == "collection" else "failed in call"
    later_why = "did not collect" if later == "collection" else f"failed in {later}"
    if later_id == initial_id:
        assert found[initial_id]["why"] == f"{initial_why}; on retry: {later_why}"
    else:
        assert found[initial_id]["why"] == initial_why
        assert found[later_id] == {"id": later_id, "pass": "retry", "why": later_why, "retry": "first seen on retry"}
    (tmp_path / "classification.json").write_text(json.dumps(record), encoding="utf-8")
    monkeypatch.setattr(pmc, "first_parent_chain", lambda *_: pytest.fail("no exact repeat to bisect"))
    assert pmc.main(["attribute", "--evidence-dir", str(tmp_path), "--base", "b" * 40, "--head", HEAD]) == 0
    verdicts = json.loads((tmp_path / "attribution.json").read_text(encoding="utf-8"))["verdicts"]
    assert {row["id"]: row["verdict"] for row in verdicts} == dict.fromkeys(found, "unattributed")


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
        nodeid = "tests/test_sized.py::test_a"
        _results(directory, _whole(nodeid, _row(nodeid, "failed")))
        return 1

    session = pmc.size_session(tmp_path / "size", base="b" * 40, mode="approved", cwd=repo, run=run)
    assert seen == [(["-m", "size_ratchet", "tests/nested/test_deep.py", "tests/test_sized.py"], tmp_path / "size", repo,
                     {"OURO_SIZE_RATCHET_BASE_REF": "b" * 40, "OURO_BOOK_GROWTH_MODE": "approved"})]
    assert pmc.size_findings(session) == ({"size tests/test_sized.py::test_a": "failed in call"}, [])
    assert pmc.size_session(tmp_path / "size", base="b" * 40, cwd=repo, run=lambda *a, **k: 2) is None
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
    skipped = "tests/test_b.py::test_books"
    _results(tmp_path / "skip", [*_whole("tests/test_smoke.py::test_a"), _row(skipped, "skipped", "setup"),
                                 _row(skipped, phase="teardown")])
    found, problems = pmc.size_findings(pmc.read_session(tmp_path / "skip"))
    assert found == {} and problems == ["the size check skipped `tests/test_b.py::test_books`, so it measured nothing there"]
    # A module id of the size check is an aggregate like any other: a skip at collection within it is no pass.
    _results(tmp_path / "uncollected", _whole("tests/test_b.py::test_a"), skips=["tests/test_b.py::TestK"])
    assert pmc.size_status(pmc.read_session(tmp_path / "uncollected"), ["size tests/test_b.py"]) == {
        "size tests/test_b.py": "skipped"}
    _results(tmp_path / "green", _whole("tests/test_smoke.py::test_a"))
    assert pmc.size_findings(pmc.read_session(tmp_path / "green")) == ({}, [])
    _results(tmp_path / "wrong-exit", _whole(skipped, _row(skipped, "failed")), exit_code=0)
    assert pmc.size_findings(pmc.read_session(tmp_path / "wrong-exit")) == (
        {"size " + skipped: "failed in call"}, ["the size check reported a failure despite a zero exit"])
    unfinished = ["the size check could not run or finish, so it went unchecked"]
    _results(tmp_path / "interrupted", [_row("tests/test_smoke.py::test_a", "failed")], exit_code=2)
    assert pmc.size_findings(pmc.read_session(tmp_path / "interrupted")) == (None, unfinished)
    assert pmc.size_findings(pmc.read_session(tmp_path / "missing")) == (None, unfinished)
    assert pmc.size_findings(None) == (None, unfinished)  # no module marks a size test
    assert pmc.size_status(None, [SIZED]) == {SIZED: "unknown"}
    for rows in ([], [_row("tests/t.py::test_a", phase="setup")], [_row("tests/t.py::test_a", "failed", "setup"),
                                                                   _row("tests/t.py::test_a", phase="teardown")]):
        _results(tmp_path / "incomplete", rows)
        assert "did not measure every collected test" in " ".join(pmc.size_findings(pmc.read_session(tmp_path / "incomplete"))[1])


def test_a_red_size_test_is_failed_without_a_retry(tmp_path):
    red, sized = "tests/t.py::test_red", SIZED.removeprefix("size ")
    _selection(tmp_path, parallel="failure", reports={"parallel": _whole(red, _row(red, "failed"))})
    _results(tmp_path / "size", [*_whole(sized, _row(sized, "failed")), *_whole("tests/test_smoke.py::test_ok")])
    retried = []

    def retry(args, directory):
        retried.append(list(args))
        return _results(directory, _whole(red, _row(red, "failed")))

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


@pytest.mark.parametrize("pytest_exit", [0, 1])
def test_size_wrapper_mismatch_keeps_fresh_red_without_certifying_measurement(tmp_path, monkeypatch, pytest_exit):
    _selection(tmp_path)
    nodeid = SIZED.removeprefix("size ")
    monkeypatch.setattr(pmc, "size_modules", lambda _: [nodeid.split("::", 1)[0]])

    def run(args, directory, **kwargs):
        _results(directory, _whole(nodeid, _row(nodeid, "failed" if pytest_exit else "passed")))
        return pytest_exit + 1  # A cleanup failure changes the wrapper status after pytest exported.

    session = pmc.size_session(tmp_path / "size", base="b" * 40, run=run)
    assert session["exit"] == pytest_exit and session["runner_exit"] == pytest_exit + 1
    record = pmc.classify(tmp_path, size=lambda: session, run=lambda *_: pytest.fail("a size test is not retried"))
    assert record["selection_executed"] and not record["size_checked"]
    assert record["problems"] == [f"the size check runner exit {pytest_exit + 1} disagrees with pytest exit {pytest_exit}"]
    assert record["failed"] == ([{"id": SIZED, "pass": "size", "why": "failed in call", "retry": "failed"}]
                                if pytest_exit else [])
    assert pmc.size_status(session, [SIZED]) == {SIZED: "unknown"}  # not evidence to retire a known red


def test_classify_runs_the_gates_its_leg_names_and_measures_books_by_the_decided_mode(tmp_path, monkeypatch):
    _selection(tmp_path)
    sessions, linted = [], []
    monkeypatch.setattr(pmc, "size_session", lambda directory, **kwargs: sessions.append(kwargs) or None)
    monkeypatch.setattr(pmc, "lint_findings", lambda: linted.append(True) or {})
    monkeypatch.setattr(pmc, "run_pytest", lambda *a, **k: pytest.fail("nothing to retry"))
    growth = {"mode": "approved", "approved": ["a" * 40], "base": "b" * 40, "head": HEAD}
    (tmp_path / "growth.json").write_text(json.dumps(growth), encoding="utf-8")
    assert pmc.main(["classify", "--evidence-dir", str(tmp_path), "--head", HEAD, "--lint", "--size", "--base", "b" * 40]) == 0
    record = json.loads((tmp_path / "classification.json").read_text(encoding="utf-8"))
    assert sessions == [{"base": "b" * 40, "mode": "approved"}] and linted == [True]
    assert record["gates"] == ["lint", "size"] and record["size"] == {
        "base": "b" * 40, "mode": "approved", "decided": True, "approved": ["a" * 40]}
    for field in ("base", "head"):
        (tmp_path / "growth.json").write_text(json.dumps({**growth, field: "c" * 40}), encoding="utf-8")
        assert pmc.main(["classify", "--evidence-dir", str(tmp_path), "--head", HEAD, "--size", "--base", "b" * 40]) == 0
        assert sessions[-1] == {"base": "b" * 40, "mode": "block"}
    (tmp_path / "growth.json").unlink()  # the growth step did not run: books are measured strictly
    assert pmc.main(["classify", "--evidence-dir", str(tmp_path), "--head", HEAD, "--size", "--base", "b" * 40]) == 0
    record = json.loads((tmp_path / "classification.json").read_text(encoding="utf-8"))
    assert sessions[-1] == {"base": "b" * 40, "mode": "block"} and record["size"]["decided"] is False
    assert record["gates"] == ["size"] and linted == [True]  # this leg did not lint
    assert pmc.main(["classify", "--evidence-dir", str(tmp_path), "--head", HEAD]) == 0  # the Windows leg
    record = json.loads((tmp_path / "classification.json").read_text(encoding="utf-8"))
    assert record["gates"] == [] and "size" not in record and len(sessions) == 4 and linted == [True]
    with pytest.raises(SystemExit):
        pmc.main(["classify", "--evidence-dir", str(tmp_path), "--size"])


def test_a_red_node_lane_is_retried_as_one_target(tmp_path):
    for code, verdict in ((0, "flaky"), (1, "failed"), (2, "failed")):
        evidence = tmp_path / str(code)
        _selection(evidence, node="failure")
        record = pmc.classify(evidence, run=lambda *a: pytest.fail("no pytest red to retry"), node=lambda: code)
        assert [row["id"] for row in record[verdict]] == [pmc.NODE_LANE]
        assert record["complete"] is (code != 2)
        if code == 2:
            assert record["failed"][0]["retry"] == "unknown"


@pytest.mark.parametrize("sibling_red", [False, True])
def test_a_retry_pytest_refuses_runs_the_nodes_the_tree_defines(tmp_path, sibling_red):
    red = "tests/t.py::test_p[redacted-value]"
    _selection(tmp_path, parallel="failure", reports={"parallel": _whole(red, _row(red, "failed"))})
    sessions = []

    def run(args, directory):
        sessions.append(list(args))
        if len(sessions) == 1:  # "ERROR: not found": nothing ran
            return _results(directory, [], exit_code=4)
        sibling = "tests/t.py::test_p[sibling]"
        return _results(directory, _whole(red) + (_whole(sibling, _row(sibling, "failed")) if sibling_red else []))

    record = pmc.classify(tmp_path, run=run, node=lambda: 0, read_source={"tests/t.py": "def test_p(v): pass\n"}.get)
    assert [(row["id"], row["retry"]) for row in record["failed"]] == (
        [("tests/t.py::test_p[sibling]", "first seen on retry")] if sibling_red else [])
    assert sessions == [["tests/t.py::test_p[redacted-value]"], ["tests/t.py::test_p"]]
    assert [row["id"] for row in record["flaky"]] == ["tests/t.py::test_p[redacted-value]"]


@pytest.mark.parametrize("retry", ["red", "green"])
def test_two_raw_parameters_with_one_public_id_confirm_neither_a_repeat_nor_a_flake(tmp_path, monkeypatch, retry):
    """The inventory holds two raw ids (a parameter with `@`, one with `::`) that every public report spells the same.
    The fallback runs their function, so a red or a pass under that spelling cannot be told from the other test's: the
    red stays on record with an unknown retry, is neither flaky nor bisected, and no raw id leaves the inventory."""
    first, second, masked = (f"tests/t.py::test_name[{part}]" for part in ("first@x", "second::y", "redacted"))
    _selection(tmp_path, parallel="failure",
               reports={"parallel": [*_whole(first, _row(first, "failed")), *_whole(second)]})
    _results(tmp_path / "parallel", [*_whole(masked, _row(masked, "failed")), *_whole(masked)],
             selected=[masked, masked])
    sessions = []

    def run(args, directory):
        sessions.append(list(args))
        if len(sessions) == 1:  # pytest finds no node under the public spelling: nothing ran
            return _results(directory, [], exit_code=4)
        again = _row(masked, "failed") if retry == "red" else None
        return _results(directory, [*_whole(masked), *_whole(masked, again)], selected=[masked, masked])

    record = pmc.classify(tmp_path, run=run, node=lambda: 0, read_source={"tests/t.py": "def test_name(v): pass\n"}.get)
    assert sessions == [[masked], ["tests/t.py::test_name"]]
    assert [(row["id"], row["retry"]) for row in record["failed"]] == [(masked, "unknown")] and not record["flaky"]
    assert record["complete"] and record["selection_executed"]
    (tmp_path / "classification.json").write_text(json.dumps(record), encoding="utf-8")
    monkeypatch.setattr(pmc, "first_parent_chain", lambda *_: pytest.fail("no confirmed repeat to bisect"))
    assert pmc.main(["attribute", "--evidence-dir", str(tmp_path), "--base", "b" * 40, "--head", HEAD]) == 0
    verdicts = json.loads((tmp_path / "attribution.json").read_text(encoding="utf-8"))["verdicts"]
    assert [(row["id"], row["verdict"]) for row in verdicts] == [(masked, "unattributed")]
    published = "".join((tmp_path / name).read_text(encoding="utf-8") for name in ("classification.json", "attribution.json"))
    assert "first@x" not in published and "second::y" not in published


def test_bisection_only_receives_the_ids_that_failed_again(tmp_path, monkeypatch):
    gone, again, sibling = (f"tests/t.py::test_p[{name}]" for name in ("gone", "again", "sibling"))
    _selection(tmp_path, parallel="failure", reports={"parallel": [
        *_whole(gone, _row(gone, "failed")), *_whole(again, _row(again, "failed"))]})

    def retry(args, directory):
        if list(args) == ["tests/t.py::test_p"]:
            return _results(directory, [*_whole(sibling), *_whole(again, _row(again, "failed"))])
        return _results(directory, [], exit_code=4)

    record = pmc.classify(tmp_path, run=retry, read_source={"tests/t.py": "def test_p(v): pass"}.get)
    assert {row["id"]: row["retry"] for row in record["failed"]} == {gone: "unknown", again: "failed"}
    (tmp_path / "classification.json").write_text(json.dumps(record), encoding="utf-8")
    monkeypatch.setattr(pmc, "first_parent_chain", lambda a, b: [a, b])
    def bisect(chain, ids, probe, **kw):
        assert ids == [again]
        return [{"id": again, "verdict": "culprit", "culprit": HEAD, "last_good": chain[0]}]
    monkeypatch.setattr(pmc, "bisect", bisect)
    assert pmc.main(["attribute", "--evidence-dir", str(tmp_path), "--base", "b" * 40, "--head", HEAD]) == 0
    verdicts = json.loads((tmp_path / "attribution.json").read_text())["verdicts"]
    assert {row["id"]: row["verdict"] for row in verdicts} == {gone: "unattributed", again: "culprit"}
    text, red = pmc.render_summary(tmp_path, system="windows")
    assert red and gone in text and "retry: unknown" in text


@pytest.mark.parametrize("fault", ["lost export", "runner failure"])
def test_a_passing_retry_without_a_matching_final_result_stays_unconfirmed(tmp_path, monkeypatch, fault):
    red = "tests/t.py::test_retry"
    _selection(tmp_path, parallel="failure", reports={"parallel": _whole(red, _row(red, "failed"))})

    def retry(ids, directory):
        code = _results(directory, _whole(red))
        if fault == "lost export":
            (directory / "results.json").unlink()
            (directory / "events.jsonl").write_text("\n".join(json.dumps({"event": "report", **row})
                                                             for row in _whole(red)), encoding="utf-8")
            return code
        return 2

    record = pmc.classify(tmp_path, run=retry)
    assert record["selection_executed"] and not record["complete"]
    assert record["flaky"] == [] and record["failed"][0]["retry"] == "unknown"
    (tmp_path / "classification.json").write_text(json.dumps(record), encoding="utf-8")
    assert pmc.render_summary(tmp_path, system="windows")[1]
    monkeypatch.setattr(pmc, "first_parent_chain", lambda *_: pytest.fail("no bisection of an unconfirmed retry"))
    assert pmc.main(["attribute", "--evidence-dir", str(tmp_path), "--base", "b" * 40, "--head", HEAD]) == 0


def test_a_check_that_could_not_see_its_failures_says_so(tmp_path):
    record = pmc.classify(tmp_path)
    assert record["problems"] == ["the selected run left no selection.json (its selection failed or it died early)"]
    assert record["complete"] and not record["selection_executed"]
    _selection(tmp_path, parallel="skipped", serial="skipped")  # the plan was written before a pass started
    assert not pmc.classify(tmp_path)["selection_executed"]
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
    record = pmc.classify(tmp_path, run=lambda args, directory: _results(directory, _whole("tests/s.py::test_slow")),
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
        return _results(directory, _whole(red))

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


@pytest.mark.parametrize("case,counts,problem", [
    ("empty", {"parallel": 1, "serial": 0}, ""),
    ("call skip", {"parallel": 1, "serial": 1}, ""),
    ("mixed setup skip", {"parallel": 1, "serial": 1}, ""),
    ("all setup skip", {"parallel": 1}, "in setup before their call"),
    ("subtest only", {"parallel": 1}, "missing selected test reports"),
    ("collector skipped", {"parallel": 1}, "after a collector skipped"),
    ("both empty", {"parallel": 0, "serial": 0}, "both selected passes collected zero tests"),
])
def test_an_empty_pass_and_a_setup_skip_have_distinct_workflow_results(tmp_path, case, counts, problem):
    one, two = "tests/t.py::test_a", "tests/s.py::test_b"
    setup_skip = [_row(two, "skipped", "setup"), _row(two, phase="teardown")]
    reports = {"parallel": [] if case == "both empty" else _whole(one)}
    if case == "call skip":
        reports["serial"] = _whole(two, _row(two, "skipped"))
    if case in ("mixed setup skip", "all setup skip"):
        reports["serial"] = setup_skip + (_whole(one) if case == "mixed setup skip" else [])
    if case == "subtest only":
        reports["serial"] = _whole(two, {**_row(two), "is_subtest": True})
    _selection(tmp_path, reports=reports)
    if case == "collector skipped":
        _results(tmp_path / "serial", [], selected=[], exit_code=5, skips=["tests/s.py"])
    record = pmc.classify(tmp_path)
    assert record["selection"]["calls"] == counts and record["selection_executed"] is (not problem)
    (tmp_path / "classification.json").write_text(json.dumps(record), encoding="utf-8")
    text, red = pmc.render_summary(tmp_path, system="windows")
    assert red == bool(problem) and (problem or "— GREEN") in text


def test_selection_rejects_an_older_result_even_if_its_journal_also_survived(tmp_path):
    node = "tests/t.py::test_current"
    _selection(tmp_path, reports={"parallel": _whole(node)})
    selection = json.loads((tmp_path / "selection.json").read_text())
    selection["inventory"]["parallel"]["session"] = "current"
    (tmp_path / "selection.json").write_text(json.dumps(selection))
    directory = tmp_path / "parallel"
    assert _results(directory, _whole("tests/t.py::stale")) == 0
    export = directory / "results.json"
    export.write_text(json.dumps({**json.loads(export.read_text()), "session": "older"}))
    (directory / "events.jsonl").write_text(json.dumps({"event": "session", "session": "older"}) + "\n")
    record = pmc.classify(tmp_path)
    assert not record["selection_executed"] and record["selection"]["calls"] == {"serial": 0}
    assert record["diagnostics_incomplete"] == ["parallel"]


def test_distinct_inventory_ids_can_share_a_redacted_id_in_the_public_reports(tmp_path):
    first, second, masked = (f"tests/t.py::test_name[{p}]" for p in ("first@x", "second::y", "redacted"))
    # The serial pass runs without xdist, so its export names the selection it ran (an xdist controller's names none).
    _selection(tmp_path, reports={"serial": [*_whole(first), *_whole(second)]})
    _results(tmp_path / "serial", [*_whole(masked), *_whole(masked)], selected=[masked, masked])
    record = pmc.classify(tmp_path)
    assert record["selection_executed"] and record["selection"]["calls"] == {"parallel": 0, "serial": 2}
    _results(tmp_path / "serial", [*_whole(masked), *_whole(masked)], selected=[masked, "tests/t.py::other"])
    assert not pmc.classify(tmp_path)["selection_executed"]


@pytest.fixture
def offline_cost(monkeypatch):
    for name in ("GITHUB_SHA", "GITHUB_RUN_ID", "GITHUB_RUN_ATTEMPT"):
        monkeypatch.delenv(name, raising=False)


@pytest.mark.parametrize("case,system,red,scope", [
    ("complete", "windows", False, "chain"), ("complete", "linux", False, "chain"),
    ("different selection", "windows", True, ""), ("selected step failed", "windows", True, ""),
    ("push only", "windows", False, "push"),
    ("unmeasured size", "linux", True, ""), ("wrong size base", "linux", True, ""),
    ("tracked red", "linux", True, "chain"), ("untracked red", "linux", True, "push"),
    ("no routing record", "windows", True, "push"), ("other OS", "windows", True, "push"),
    ("missing tracking list", "windows", True, "push"), ("cost failed", "linux", True, "chain"),
    # A cost test that failed is a red no issue tracks; a lint gate that never ran measured nothing.
    ("missing cost reports", "linux", True, ""), ("partial cost", "linux", True, ""),
    ("cost red", "linux", True, "push"), ("lint unchecked", "linux", True, ""),
    ("old classification", "windows", False, ""),
])
def test_summary_marks_only_a_recoverable_window_independently_of_green(
        tmp_path, monkeypatch, offline_cost, case, system, red, scope):
    base = "b" * 40
    output = tmp_path / "outputs"
    monkeypatch.setenv("GITHUB_OUTPUT", str(output))
    monkeypatch.delenv("GITHUB_STEP_SUMMARY", raising=False)
    monkeypatch.setattr(pmc, "merges", lambda *_: [])
    (tmp_path / "base.json").write_text(json.dumps({
        "base": base, "head": HEAD, "continuous": case != "push only"}), encoding="utf-8")
    _selection(tmp_path)
    assert pmc.main(["classify", "--evidence-dir", str(tmp_path), "--head", HEAD,
                     "--base", ("d" * 40 if case == "different selection" else base),
                     "--selected-outcome", "failure" if case == "selected step failed" else "success"]) == 0
    path = tmp_path / "classification.json"
    record = json.loads(path.read_text(encoding="utf-8"))
    if system == "linux":
        record.update(gates=["lint", "size"], size={"base": "c" * 40 if case == "wrong size base" else base},
                      size_checked=case != "unmeasured size", lint_checked=case != "lint unchecked")
        if case == "unmeasured size":
            record["problems"].append("the size check did not measure every collected test")
        if case == "lint unchecked":
            record["problems"].append("ruff could not run, so the lint gate went unchecked")
        # "cost failed": the step failed while its report passed; "cost red": the report holds the failure.
        monkeypatch.setenv("COST_CONTRACT_OUTCOME", "failure" if case in ("cost failed", "cost red") else "success")
        cost = "tests/test_cost_contract.py::test_cost"
        if case != "missing cost reports":
            rows = _whole(cost, _row(cost, "failed")) if case == "cost red" else _whole(cost)
            _results(tmp_path / "cost-contract", rows[:-1] if case == "partial cost" else rows)
    if case in ("tracked red", "untracked red", "no routing record", "other OS", "missing tracking list"):
        node = "tests/t.py::test_red"
        record["failed"] = [{"id": node, "pass": "parallel", "why": "failed in call", "retry": "failed"}]
        if case != "no routing record":
            routing = {"os": "linux" if case == "other OS" else system}
            if case != "missing tracking list":
                routing["untracked"] = [node] if case == "untracked red" else []
                routing["awaiting_handoff"] = [{"head": HEAD, "run": 7, "attempt": 1, "base": base, "ids": ids}
                                               for ids in [routing["untracked"]] if ids]
            (tmp_path / "routing.json").write_text(json.dumps(routing), encoding="utf-8")
    if case == "old classification":
        del record["selection_executed"]
    path.write_text(json.dumps(record), encoding="utf-8")
    assert pmc.main(["summary", "--evidence-dir", str(tmp_path), "--os", system]) == int(red)
    assert output.read_text(encoding="utf-8").splitlines()[-1] == f"window_scope={scope}"


@pytest.mark.parametrize("handoff", ["tracked", "refused extension", "interrupted retry"])
def test_docs_only_push_selects_the_red_module_until_its_os_has_an_issue(tmp_path, monkeypatch, offline_cost, handoff):
    from ouroboros import test_selection, update_qualification
    from tests._post_merge_github import FakeGitHub, tracking_issue

    module = "tests/test_leaf.py"
    first, second = (f"{module}::test_{name}" for name in ("one", "two"))
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "ouroboros")
    base = _commit(repo, {"leaf.py": "value = 1\n", "tests/test_floor.py": "def test_floor(): pass\n",
                          module: "from leaf import value\ndef test_one(): assert value == 1\ndef test_two(): assert value < 2\n"})
    monkeypatch.setattr(test_selection, "FLOOR_TESTS", {"tests/test_floor.py": "fixture"})
    monkeypatch.setattr(pmc, "REPO_ROOT", repo)
    missed = _commit(repo, {"leaf.py": "value = 2\n"})
    head = _commit(repo, {"README.md": "First push\n"})
    run = {**_run(head, conclusion="failure", number=7), "event": "workflow_dispatch"}
    run_page = f"{pmc.POST_MERGE_RUNS}?per_page=100&page=1"
    history = {run_page: {"workflow_runs": [], "total_count": 0}}
    monkeypatch.setattr(pmc.GitHub, "get", lambda self, path: history[path])
    args = type("Args", (), {"head": None, "evidence_dir": tmp_path})
    assert pmc.command_base(args, {"GITHUB_SHA": head, "GITHUB_EVENT_NAME": "workflow_dispatch", "INITIAL_BASE": base,
                                   "GITHUB_REPOSITORY": "razzant/ouroboros", "GITHUB_REF": "refs/heads/ouroboros"}) == 0
    window = json.loads((tmp_path / "base.json").read_text(encoding="utf-8"))
    assert window["base"] == base and window["continuous"]
    jobs = [_window_job(run, window)]
    github = FakeGitHub(label=True, issues=[tracking_issue(5, {"kind": "unattributed", "os": "windows", "ids": [first]}, head=head)],
                        fail=[("PATCH", "issues/5")] if handoff == "refused extension" else [])
    failures = _whole(first, _row(first, "failed")) + _whole(second, _row(second, "failed"))

    def retry(command, **kwargs):
        if handoff == "interrupted retry":
            raise RuntimeError("interrupted after the first measurements")
        return _results(Path(command[command.index("--ci-evidence-dir") + 1]), failures)

    monkeypatch.setattr(pmc.subprocess, "call", retry)
    monkeypatch.setattr(pmc, "size_session", lambda directory, **kw: pmc.read_session(directory))
    monkeypatch.setattr(pmc, "lint_findings", lambda: {})
    monkeypatch.setenv("COST_CONTRACT_OUTCOME", "success")
    for system in ("linux", "windows"):
        evidence = tmp_path / system
        linux = system == "linux"
        _selection(evidence, base=base, parallel="success" if linux else "failure",
                   reports={"parallel": _whole(first) + _whole(second) if linux else failures})
        (evidence / "base.json").write_bytes((tmp_path / "base.json").read_bytes())
        output = evidence / "output"
        monkeypatch.setenv("GITHUB_OUTPUT", str(output))
        monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(evidence / "summary"))
        classify = ["classify", "--evidence-dir", str(evidence), "--head", head, "--base", base,
                    "--selected-outcome", "success" if linux else "failure"]
        if linux:
            _results(evidence / "size", _whole("tests/test_size.py::test_size"))
            _results(evidence / "cost-contract", _whole("tests/test_cost.py::test_cost"))
            classify += ["--lint", "--size"]
        if not linux and handoff == "interrupted retry":
            with pytest.raises(RuntimeError, match="after the first measurements"):
                pmc.main(classify)
        else:
            assert pmc.main(classify) == 0
        with monkeypatch.context() as patch:
            patch.setattr(pmc, "GitHub", lambda *_: github)
            assert pmc.main(["route", "--evidence-dir", str(evidence), "--os", system, "--head", head,
                             "--run-url", "https://example.invalid/run/7"]) == 0
        held = not linux and handoff == "refused extension"
        routing = json.loads((evidence / "routing.json").read_text(encoding="utf-8"))
        assert routing["untracked"] == ([second] if held else [])
        assert pmc.main(["summary", "--evidence-dir", str(evidence), "--os", system]) == int(not linux)
        summary = (evidence / "summary").read_text(encoding="utf-8")
        assert ("Keep the starting base" in summary) == held
        assert ("retry: not run" in summary) == (not linux and handoff == "interrupted retry")
        scope = dict(line.split("=", 1) for line in output.read_text().splitlines())["window_scope"]
        assert scope == ("push" if held else "chain")
        leg = _job(run, system, base=base, scope=scope)
        debt = dict(line.split("=", 1) for line in output.read_text().splitlines())["untracked"]
        leg["steps"].append({"name": "Post-merge untracked " + debt, "status": "completed", "conclusion": "success"})
        jobs.append(leg)

    history[run_page] = {"workflow_runs": [run], "total_count": 1}
    history[f"{pmc.POST_MERGE_RUNS}?branch=ouroboros&head_sha={head}&per_page=100&page=1"] = history[run_page]
    history["actions/runs/7/attempts/1/jobs?per_page=100&page=1"] = {"jobs": jobs, "total_count": len(jobs)}
    docs_head = _commit(repo, {"README.md": "Documentation only\n"})
    assert pmc.command_base(args, {"GITHUB_SHA": docs_head, "BEFORE": head}) == 0
    following = json.loads((tmp_path / "base.json").read_text(encoding="utf-8"))
    assert following["continuous"] and following["base"] == (base if handoff == "refused extension" else head)
    selection = test_selection.select_tests(repo, base=following["base"])
    assert (module in selection.files) == (handoff == "refused extension")
    if handoff == "refused extension":
        assert following["replay"] == {"history_read": True, "history_gaps": [], "awaiting_handoff": {"windows": [
            {"head": head, "run": None, "attempt": 1, "base": base, "ids": [second]}]}}
        monkeypatch.setenv("GITHUB_RUN_ID", "8")
        for outcome in ("skipped", "passed"):
            replay = tmp_path / outcome
            rows = _whole(first) + (_whole(second) if outcome == "passed" else [
                _row(second, "skipped", "setup"), _row(second, phase="teardown")])
            _selection(replay, base=base, reports={"parallel": rows})
            _identified(replay / "parallel", rows, sha=docs_head, run_id="8", platform="Windows")
            (replay / "base.json").write_text(json.dumps(following))
            monkeypatch.setenv("GITHUB_OUTPUT", str(replay / "outputs"))
            assert pmc.main(["classify", "--evidence-dir", str(replay), "--head", docs_head, "--base", base]) == 0
            with monkeypatch.context() as patch:
                patch.setattr(pmc, "GitHub", lambda *_: github)
                assert pmc.main(["route", "--evidence-dir", str(replay), "--os", "windows", "--head", docs_head,
                                 "--run-url", "run/8"]) == 0
            assert pmc.main(["summary", "--evidence-dir", str(replay), "--os", "windows"]) == int(outcome == "skipped")
            expected = [second] if outcome == "skipped" else []
            assert json.loads((replay / "routing.json").read_text())["untracked"] == expected
            assert (replay / "outputs").read_text().splitlines()[-1] == (
                "window_scope=push" if expected else "window_scope=chain")
    # The Update reader still sees the original direct red and the range both systems measured.
    observed = update_qualification.post_merge_checks({"base_sha": base, "commits": [{"sha": head}, {"sha": missed}]},
        "https://github.com/test/repo", get_json=lambda url: history[url.removeprefix("https://api.github.com/repos/test/repo/")])
    [direct] = observed["runs"]
    assert direct["conclusion"] == "failure" and direct["covered_shas"] == [missed] and observed["missing_shas"] == []


@pytest.mark.parametrize("cost_outcome,session,expected_red", [
    ("failure", "failed", True), ("failure", "passed", True),
    ("failure", "unfinished", True), ("skipped", "missing", True),
    ("success", "missing", True), ("success", "failed", True),
    ("success", "skipped", True), ("success", "passed", False),
    ("", "missing", True),
])
def test_cost_outcome_reaches_the_real_summary_command(tmp_path, monkeypatch, capsys,
                                                      offline_cost, cost_outcome, session, expected_red):
    (tmp_path / "classification.json").write_text(json.dumps({
        "complete": True, "failed": [], "flaky": [], "problems": [], "not_run": {}}), encoding="utf-8")
    nodeid = "tests/test_cost_contract.py::test_rest_law"
    if session not in ("missing", "unfinished"):
        rows = ([_row(nodeid, "skipped", "setup"), _row(nodeid, phase="teardown")] if session == "skipped"
                else _whole(nodeid, _row(nodeid, session)))
        _results(tmp_path / "cost-contract", rows)
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
                                                                        offline_cost, case, expected_red, said):
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


@pytest.mark.parametrize("foreign", [{}, {"sha": "e" * 40}, {"run_id": "78"}, {"run_attempt": "1"}, None])
def test_cost_evidence_must_name_the_current_actions_run_and_attempt(tmp_path, monkeypatch, capsys, foreign):
    base = "b" * 40
    for name, value in {
        "base": {"base": base, "head": HEAD, "continuous": True},
        "classification": {"base": base, "head": HEAD, "selection_executed": True, "size_checked": True,
                           "size": {"base": base}},
        "routing": {"os": "linux", "untracked": [], "awaiting_handoff": []},
    }.items():
        (tmp_path / f"{name}.json").write_text(json.dumps(value), encoding="utf-8")
    output = tmp_path / "outputs"
    monkeypatch.setattr(pmc, "merges", lambda *_: [])
    monkeypatch.setenv("GITHUB_OUTPUT", str(output))
    monkeypatch.delenv("GITHUB_STEP_SUMMARY", raising=False)
    monkeypatch.setenv("COST_CONTRACT_OUTCOME", "success")
    identity = {"sha": HEAD, "run_id": "77", "run_attempt": "2"}
    for key, value in identity.items():
        monkeypatch.setenv(f"GITHUB_{key.upper()}", value)
    _results(tmp_path / "cost-contract", [*_whole("tests/test_cost_contract.py::test_ok"), *_whole(BREACH, _xfail(BREACH))],
             github=None if foreign is None else {**identity, **foreign})
    accepted = foreign == {}
    assert pmc.main(["summary", "--evidence-dir", str(tmp_path), "--attempt", "2"]) == int(not accepted)
    assert output.read_text(encoding="utf-8").splitlines()[-1] == ("window_scope=chain" if accepted else "window_scope=")
    text = capsys.readouterr().out
    assert ("1 known breach(es) measured" in text) is accepted
    assert ("::notice title=Known cost-contract breach" in text) is accepted
    assert ("no result matches this GitHub commit, run and attempt" in text) is (not accepted)


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
        return _results(directory, [_row("tests/t.py::test_ok"), _row("tests/t.py::test_skip", "skipped", "setup")])
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
            {"event": "start", "nodeid": ids[0]}, *({"event": "report", **row} for row in _whole(ids[0])),
            {"event": "finish", "nodeid": ids[0]},
            {"event": "start", "nodeid": "tests/t.py::test_param[1]"},
            {"event": "report", **_row("tests/t.py::test_param[1]")}]), encoding="utf-8")
    assert pmc.recheck_ids(ids[:3], tmp_path / "killed", read_source=sources.get, run=killed) == dict(zip(
        ids[:3], ["unknown", "failed", "unknown"]))  # test_param[1] was still running when the session ended; test_ok's
    # whole pass in a journal alone retires nothing: its public id may hide a sibling that never started


def test_route_collects_unselected_owed_ids_without_retiring_a_reported_skip(tmp_path, monkeypatch):
    from tests._post_merge_github import FakeGitHub

    module = "tests/test_owed_fixture.py"
    renamed, deselected, skipped = [f"{module}::test_{name}" for name in ("old_name", "deselected", "skipped")]
    _selection(tmp_path)
    new = "tests/test_floor.py::test_floor"
    rows = _whole(new) + [_row(skipped, "skipped", "setup"), _row(skipped, phase="teardown")]
    _identified(tmp_path / "parallel", rows, platform="Windows")
    (tmp_path / "base.json").write_text(json.dumps({"replay": {
        "awaiting_handoff": {"windows": [{"head": HEAD, "run": 60, "attempt": 1, "base": None,
                                          "ids": [renamed, deselected, skipped]}]},
    }}), encoding="utf-8")
    monkeypatch.setenv("GITHUB_RUN_ID", "77")
    github = FakeGitHub(label=True)
    monkeypatch.setattr(pmc, "GitHub", lambda *_: github)
    monkeypatch.setattr(pmc, "working_source", lambda _: "def test_new_name(): pass\n")
    calls = []

    def collect(args, directory):
        calls.append(args)
        listed = Path(next(arg.split("=", 1)[1] for arg in args if arg.startswith("--ci-listed=")))
        assert listed.read_text(encoding="utf-8").splitlines() == [deselected, renamed]
        code = _results(directory, [], selected=())
        path = directory / "results.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        data["listed"] = {"unmatched": [renamed]}
        path.write_text(json.dumps(data), encoding="utf-8")
        return code

    recheck = pmc.recheck_ids
    monkeypatch.setattr(pmc, "recheck_ids", lambda ids, directory, **kw: recheck(
        ids, directory, read_source=pmc.working_source, run=collect, **kw))
    assert pmc.main(["route", "--evidence-dir", str(tmp_path), "--head", HEAD,
                     "--run-url", "run/77", "--os", "windows"]) == 0
    record = json.loads((tmp_path / "routing.json").read_text(encoding="utf-8"))
    # The recorded reds get their tracker before the recheck, which then takes off only the test that is gone:
    # the skip and the test no session ran stay on the issue.
    assert len(calls) == 1 and record["created"] == [100] and record["untracked"] == []
    assert record["narrowed"] == [{"issue": 100, "ids": [renamed]}]
    body = github.issues[100]["body"]
    assert deselected in body and skipped in body and renamed not in body
    assert record["rechecked"] == {deselected: "unknown", renamed: "absent"}


@pytest.mark.parametrize("ran", ["one of two", "both"])
def test_a_recheck_never_passes_a_public_id_that_names_several_tests(tmp_path, ran):
    """`--ci-listed` keeps every test whose public id it lists, so two raw parameters that redact to one id both run. Their
    phases and selected entries pool under that id: one whole instance (the session stopped before the other) would pass
    it. The recheck leaves such an id unknown, and a unique id beside it still passes."""
    masked, unique = "tests/t.py::test_name[redacted]", "tests/t.py::test_other"

    def run(args, directory):
        rows = [*_whole(masked), *([] if ran == "one of two" else _whole(masked)), *_whole(unique)]
        _results(directory, rows, selected=[masked, masked, unique], exit_code=2 if ran == "one of two" else 0)

    found = pmc.recheck_ids([masked, unique], tmp_path, read_source=dict.fromkeys(["tests/t.py"], "").get, run=run)
    assert found == {masked: "unknown", unique: "passed"}


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
    red, unknown, gone = "tests/t.py::test_red", "tests/unknown.py::test_red", "tests/gone.py::test_red"

    def source(path):
        if path == "tests/unknown.py":
            return pmc.tree_source("bad-revision")(path)
        return "def test_red(): pass\n" if path == "tests/t.py" else None

    def run(args, directory):
        assert args == [red]
        return _results(directory, [_row(red, "failed")])

    assert pmc.check_ids([red, unknown, gone], tmp_path, read_source=source, run=run) == {
        red: "failed", unknown: "unknown", gone: "absent"}


def test_the_runner_final_inventory_failure_is_not_a_green_selected_run(tmp_path):
    _selection(tmp_path)
    selection = json.loads((tmp_path / "selection.json").read_text(encoding="utf-8"))
    selection["outcome"] = "failure"  # the final floor check failed after both passes succeeded
    (tmp_path / "selection.json").write_text(json.dumps(selection), encoding="utf-8")
    result = pmc.classify(tmp_path, run=lambda *a: pytest.fail("no retry of the whole selection"), node=lambda: 0)
    assert not result["failed"] and "inventory/floor" in result["problems"][0]
    (tmp_path / "classification.json").write_text(json.dumps(result), encoding="utf-8")
    assert pmc.render_summary(tmp_path)[1]


def _identified(directory, reports, *, sha=HEAD, run_id="77", platform="Linux", collection=(), selected=None,
                collected=None):
    """A selected pass's export as tests/ci_evidence.py writes it: its reports with the run and system it names. The pass in
    one process (the serial one) names its selection; an xdist controller (the parallel one) names none and exports
    `collected`, the count of every test its workers collected, those it never saw report included."""
    _results(directory, reports, collection=collection, selected=selected, collected=collected)
    data = json.loads((Path(directory) / "results.json").read_text(encoding="utf-8"))
    data.update(github={"sha": sha, "run_id": run_id, "run_attempt": "1"}, environment={"platform": platform})
    (Path(directory) / "results.json").write_text(json.dumps(data), encoding="utf-8")


def test_only_tests_this_run_passed_whole_at_its_head_on_its_system_spare_a_recheck(tmp_path):
    ids = ["tests/t.py::test_a", "tests/t.py::test_flaky", "tests/t.py::test_skip", "tests/s.py::test_serial",
           "tests/t.py::test_breach", "tests/t.py::test_not_selected", "tests/t.py::test_torn", "tests/u.py",
           "tests/u.py::test_whole", "tests/b.py::test_b", "tests/t.py::test_param",
           "tests/test_post_merge_check.py::test_of_a_module_these_passes_did_not_run", "tests/gone.py::test_x"]
    _selection(tmp_path)
    # The parallel pass is xdist: its controller names no selection and counts every test its workers collected, each
    # reported as the producer writes it (a red and a skip with their setup and teardown). The serial pass runs in one
    # process and names its selection, a test it left without its teardown included.
    parallel = [*_whole(ids[0]), *_whole(ids[1], _row(ids[1], "failed")), _row(ids[2], "skipped", "setup"),
                _row(ids[2], phase="teardown"), *_whole(ids[4], _xfail(ids[4])), *_whole(ids[8]),
                *_whole("tests/t.py::test_param[1]")]
    _identified(tmp_path / "parallel", parallel, collection=["tests/b.py"])
    _identified(tmp_path / "serial", [*_whole(ids[3]), *_whole(ids[9]), _row(ids[6], phase="setup"), _row(ids[6])],
                selected=[ids[3], ids[9], ids[6]])
    # Kept without a recheck: an id whose own report was no whole pass (a failure, even one its retry passed, a skip,
    # an xfail, a test without its teardown, a test of a module that did not collect in another pass) and a test of a
    # module these passes ran nothing of, while its file exists. Left for the recheck: a test, module or function id
    # without a report in a module they ran, and a test whose file is gone.
    kept = dict.fromkeys([ids[1], ids[2], ids[4], ids[6], ids[9], ids[11]], "kept")
    assert pmc.same_run_verdicts(tmp_path, ids, head=HEAD, system="linux", run_id="77") == {
        ids[0]: "passed", ids[3]: "passed", ids[8]: "passed", **kept}
    # The same controller counting a test it never saw report (a worker lost before its setup) can no longer say that a whole
    # pass of its was one test's: what it reported stays owed, and the serial pass still passes its own.
    _identified(tmp_path / "parallel", parallel, collection=["tests/b.py"], collected=7)
    assert pmc.same_run_verdicts(tmp_path, ids, head=HEAD, system="linux", run_id="77") == {
        ids[3]: "passed", ids[0]: "kept", ids[8]: "kept", **kept}
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


@pytest.mark.parametrize("shape,ran", [("serial", "one of two"), ("serial", "both"), ("xdist", "one of two"),
                                       ("xdist", "both"), ("xdist", "one test per id")])
def test_a_public_id_that_several_tests_share_is_never_passed_by_the_same_run_reader_or_retired_by_the_route(
        tmp_path, monkeypatch, shape, ran):
    """Two raw parameters redact to one public id (`shared`), and the tests of a red id are two, one failing. The serial pass
    runs in one process: its export names its selection, which lists such an id twice, one instance reported or both. The
    parallel pass is xdist: its controller names no selection and exports `tests_collected` over every test its workers
    collected, so a twin that never reported leaves the count above the reports. A session that cannot tell its tests
    apart clears nothing: what it reported stays owed (kept), the unique test too, and the route asks for no recheck. Where
    the count equals the reports, the shared and red ids stay kept, a unique test passes, and so does an id that names one
    test (a lone red stays red). The route takes only what passed off the issue."""
    from tests._post_merge_github import FakeGitHub, tracking_issue

    shared, red, unique = "tests/t.py::test_name[redacted]", "tests/t.py::test_red[redacted]", "tests/t.py::test_other"
    rows = [*_whole(shared), *(_whole(shared) if ran == "both" else []), *_whole(red), *_whole(red, _row(red, "failed")),
            *_whole(unique)]
    counted = [shared, shared, red, red, unique]  # the tests the pass collected: each shared id names two
    if ran == "one test per id":  # no collision: each id names one test, the red one failing
        rows, counted = [*_whole(shared), *_whole(red, _row(red, "failed")), *_whole(unique)], [shared, red, unique]
    _selection(tmp_path)
    if shape == "serial":  # one process: the export names its selection
        _identified(tmp_path / "serial", rows, platform="Windows", selected=counted)
    else:  # xdist: the controller names none and counts what its workers collected
        _identified(tmp_path / "parallel", rows, platform="Windows", collected=len(counted))
    passed = {("xdist", "one of two"): set(), ("xdist", "one test per id"): {shared, unique}}.get((shape, ran), {unique})
    ids = [shared, red, unique]
    expected = {node: "passed" if node in passed else "kept" for node in ids}
    assert pmc.same_run_verdicts(tmp_path, ids, head=HEAD, system="windows", run_id="77") == expected
    issue = tracking_issue(5, {"os": "windows", "kind": "unattributed", "ids": [shared, red, unique]})
    github = FakeGitHub(label=True, issues=[issue])
    monkeypatch.setattr(pmc, "GitHub", lambda *_: github)
    monkeypatch.setattr(pmc, "recheck_ids", lambda *args, **kwargs: pytest.fail("each id is settled or kept here"))
    monkeypatch.setenv("GITHUB_RUN_ID", "77")
    assert pmc.main(["route", "--evidence-dir", str(tmp_path), "--head", HEAD, "--run-url", "run/77",
                     "--os", "windows"]) == 0
    record = json.loads((tmp_path / "routing.json").read_text(encoding="utf-8"))
    assert record["same_run"] == expected and not record["closed"]
    assert sorted(node for row in record["narrowed"] for node in row["ids"]) == sorted(passed)
    body = github.issues[5]["body"]
    assert github.issues[5]["state"] == "open" and all((node in body) is (node not in passed) for node in ids)


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


@pytest.mark.parametrize("filed", [True, False])
def test_a_rerun_inherits_the_failed_attempt_and_only_the_branch_tip_retires_issues(tmp_path, monkeypatch, filed):
    """The first attempt could not hand `second` over. Its rerun takes that record over: it files the red under the
    attempt that saw it fail, keeps it owed while the issue refuses the write and, no longer at the branch tip,
    takes nothing off an issue whose test passes again."""
    from tests._post_merge_github import FakeGitHub, tracking_issue

    first, second = "tests/t.py::test_fixed", "tests/t.py::test_skips"
    run = _run(HEAD, conclusion="failure", number=7)
    leg = _job(run, "windows", scope="push")
    owed = [{"head": HEAD, "run": 7, "attempt": 1, "base": "1" * 40, "ids": [second]}]
    leg["steps"].append({"name": "Post-merge untracked " + json.dumps(owed),
                         "status": "completed", "conclusion": "success"})
    jobs = [leg, _window_job(run, {"base": "1" * 40, "continuous": True})]
    github = FakeGitHub(label=True, issues=[tracking_issue(5, {"os": "windows", "kind": "unattributed", "ids": [first]})],
                        fail=[] if filed else [("PATCH", "issues/5")])
    github.repo = "razzant/ouroboros"  # the link of a recorded failure names the repository
    pages = {"actions/runs/7/attempts/1/jobs?per_page=100&page=1": {"jobs": jobs, "total_count": 2},
             "git/ref/heads/ouroboros": {"object": {"sha": "a" * 40}}}
    monkeypatch.setattr(github, "get", lambda path: pages.get(path, {}))
    monkeypatch.setattr(pmc, "GitHub", lambda *_: github)
    monkeypatch.setattr(pmc, "working_source", lambda _: "def test_skips(): pass")
    monkeypatch.setenv("GITHUB_RUN_ID", "7")
    rows = _whole(first) + [_row(second, "skipped", "setup"), _row(second, phase="teardown")]
    _selection(tmp_path, reports={"parallel": rows})
    _identified(tmp_path / "parallel", rows, run_id="7", platform="Windows")
    assert pmc.main(["route", "--attempt", "2", "--os", "windows", "--evidence-dir", str(tmp_path),
                     "--run-url", "run/7/attempts/2", "--head", HEAD]) == 0
    record = json.loads((tmp_path / "routing.json").read_text())
    assert record["awaiting_handoff"] == ([] if filed else owed) and record["untracked"] == ([] if filed else [second])
    assert record["updated"] == ([5] if filed else []) and not record["closed"] and not record["narrowed"]
    assert github.issues[5]["state"] == "open" and [number for number, _ in github.comments] == ([5] if filed else [])
    # The record is filed under the attempt that saw it fail, never under the rerun that carried it.
    link = "https://github.com/razzant/ouroboros/actions/runs/7/attempts/1"
    assert all(link in text and "run/7/attempts/2" not in text for _, text in github.comments)
    assert (second in github.issues[5]["body"]) is filed and "run/7/attempts/2" not in github.issues[5]["body"]


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


def _commit(repo, files):
    """Write `files` into the fixture repository and commit them on its first-parent line: the new SHA."""
    for path, text in files.items():
        (repo / path).parent.mkdir(parents=True, exist_ok=True)
        (repo / path).write_text(text, encoding="utf-8")
    _git(repo, "add", "-A")
    _git(repo, "-c", "user.name=t", "-c", "user.email=t@example.invalid", "commit", "-q", "-m", "change")
    return _git(repo, "rev-parse", "HEAD")


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
        (repo / "tests" / "test_v.py").write_text(
            "import pytest\n\ndef test_value(): pass\n\n@pytest.mark.size_ratchet\ndef test_size(): pass\n",
            encoding="utf-8")
        (repo / "VALUE").write_text(value + str(len(shas)), encoding="utf-8")
        _git(repo, "add", "-A")
        _git(repo, "commit", "-q", "-m", f"commit {len(shas)}")
        shas.append(_git(repo, "rev-parse", "HEAD"))
    monkeypatch.setattr(pmc, "REPO_ROOT", repo)
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    sized = "size tests/test_v.py::test_size"
    (evidence / "classification.json").write_text(json.dumps(
        {"failed": [{"id": n, "retry": "failed"} for n in ("tests/test_v.py::test_value", pmc.NODE_LANE, sized)],
         "size": {"base": shas[0], "mode": "approved"}}), encoding="utf-8")
    probed, measured = [], []

    def run(args, directory, *, cwd):
        assert args == ["tests/test_v.py::test_value"]
        probed.append(Path(cwd))
        red = (Path(cwd) / "VALUE").read_text(encoding="utf-8").startswith("2")
        return _results(directory, [_row(args[0], "failed" if red else "passed")])

    def size_session(directory, *, base, mode, cwd, modules=None):  # each probe runs its own tree's size lane
        assert modules == ["tests/test_v.py"]
        measured.append((base, mode, Path(cwd)))
        red = (Path(cwd) / "VALUE").read_text(encoding="utf-8").startswith("2")
        code = _results(directory, [_row("tests/test_v.py::test_size", "failed" if red else "passed")])
        return {**pmc.read_session(directory), "runner_exit": code}

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


def test_a_probe_answers_a_size_test_from_the_size_check_of_its_tree(tmp_path, monkeypatch, no_process):
    path = "tests/test_sized.py"
    monkeypatch.setattr(pmc, "_git", _git_reader({}, size_modules={"a" * 40: [path]}))
    monkeypatch.setattr(pmc, "remove_worktree", lambda tree: None)
    ids = [f"size {path}::test_{name}" for name in ("red", "green", "new", "skip")]

    def size(tree, directory, modules):
        assert modules == [path]
        code = _results(directory, [_row(ids[0].removeprefix("size "), "failed"), _row(ids[1].removeprefix("size ")),
                                    _row(ids[3].removeprefix("size "), "skipped", "setup")])
        return {**pmc.read_session(directory), "runner_exit": code}

    prober = pmc.Prober(tmp_path / "evidence", tmp_path / "work", HEAD, size=size,
                        run=lambda *a, **k: pytest.fail("a size test is not rerun as a plain test"))
    assert prober("a" * 40, ids) == dict(zip(ids, ("fail", "pass", "error", "error")))  # unreported or skipped is unknown
    for unfinished in (lambda tree, directory, modules: None, None):  # no size session there; a Prober without one
        prober = pmc.Prober(tmp_path / "evidence", tmp_path / "work", HEAD, size=unfinished)
        assert prober("a" * 40, ids[:1]) == {ids[0]: "error"}


# --- the protected path a historical probe may answer -----------------------------------------

def _git_reader(trees, *, importers=None, size_modules=None, unreadable=None, calls=None, sources=None):
    """A Git that answers only the reads a probe makes, in the shape the real Git answers them.

    A tree search names its files `REV:path`, NUL-separated under `-z`; a search of the checkout
    names plain relative paths. `ls-tree --name-only` names the file a tree holds, `ls-tree` names
    its blob, and `show` returns the file's source. Every other read answers as an empty result,
    and `calls` records what was asked.
    """
    def git(*args, cwd=None, strip=True):
        if calls is not None:
            calls.append((args, cwd))
        if args[:1] == ("ls-tree",):
            named = "--name-only" in args
            sha, path = (args[2] if named else args[1]), args[-1]
            if path == PROTECTED and unreadable == "timeout":
                raise subprocess.TimeoutExpired(args, 300)
            if path == PROTECTED and unreadable == "entry":
                raise subprocess.CalledProcessError(128, args)
            oid = trees.get(sha, {}).get(path)
            if named:
                return path if oid else ""
            return f"100644 blob {oid}\t{path}" if oid else ""
        if args[:1] == ("show",):
            path = args[1].partition(":")[2]
            assert path != PROTECTED, "the protected source must not be read"
            return (sources or {})[path]
        if args[:1] == ("grep",):
            index = args.index("--")
            rev = "" if args[index - 2] == "-e" else args[index - 1]
            if "-F" in args:  # the importer search: the module's name and the questioned revision
                scope = args[index + 1:]
                if unreadable == "grep" and ("tests" in scope or ":(glob)tests/**/*.py" in scope):
                    raise subprocess.CalledProcessError(128, args)
                candidates = {path for i, arg in enumerate(args) if arg == "-e"
                              for path in (importers or {}).get((rev, args[i + 1]), ())}
                names = sorted(path for path in candidates if path in scope or path.startswith("tests/") and (
                    "tests" in scope or any(spec.startswith(":(glob)tests/**/")
                                            and fnmatchcase(path.rsplit("/", 1)[-1], spec.rsplit("/", 1)[-1])
                                            for spec in scope)))
            else:
                names = (size_modules or {}).get(rev, ())
            return "".join(f"{rev}:{name}\0" if rev else f"{name}\0" for name in names)
        return ""
    return git


@pytest.fixture
def no_process(monkeypatch):
    """A process this test did not plan for fails before it starts."""
    def refuse(*args, **kwargs):
        raise AssertionError(f"unexpected process: {args}")
    for name in ("run", "call", "check_output", "Popen"):
        monkeypatch.setattr(pmc.subprocess, name, refuse)


def _tree_prober(tmp_path, monkeypatch, trees, *, ids=(), importers=None, size_modules=None, measured=None,
                 outcome="failed", unreadable=None, calls=None, sources=None, ran=None, runner_exit=None):
    """A Prober over fake trees: its runner reports only selected IDs, its size lane records the lane it was
    asked to measure, and nothing it does reaches a process."""
    suite = [nodeid for nodeid in ids if nodeid != pmc.NODE_LANE and not nodeid.startswith((pmc.LINT_PREFIX, pmc.SIZE_PREFIX))]
    fixtures = dict(sources or {})
    for nodeid in suite:
        path, _, test = nodeid.partition("::")
        if test and path != PROTECTED:
            fixtures[path] = fixtures.get(path, "") + f"\ndef {test}(): pass\n"
    monkeypatch.setattr(pmc, "_git", _git_reader(trees, importers=importers, size_modules=size_modules,
                                                 unreadable=unreadable, calls=calls, sources=fixtures))
    monkeypatch.setattr(pmc, "remove_worktree", lambda tree: None)

    def run(args, directory, cwd=None):
        allowed = set(suite) | {nodeid.split("::", 1)[0] for nodeid in suite}
        assert args and set(args) <= allowed and all(arg.split("::", 1)[0] != PROTECTED for arg in args)
        if ran is not None:
            ran.append(list(args))
        selected = [nodeid for nodeid in suite if nodeid in args or nodeid.split("::", 1)[0] in args]
        code = _results(directory, [_row(nodeid, outcome) for nodeid in selected])
        return code if runner_exit is None else runner_exit

    def size(tree, directory, modules):
        if measured is not None:
            measured.append(list(modules))
        return None

    return pmc.Prober(tmp_path / "evidence", tmp_path / "work", HEAD, run=run, size=size, node=lambda tree: 0)


def test_a_probe_answers_where_the_protected_file_is_the_head_s_own(tmp_path, monkeypatch, no_process):
    """The protected file is the head's own at this revision: every id keeps the previous route, even
    the one whose own file differs there."""
    importer = "tests/test_importer.py::test_red"
    trees = {HEAD: {PROTECTED: "a" * 40, "tests/test_importer.py": "b" * 40},
             "r" * 40: {PROTECTED: "a" * 40, "tests/test_importer.py": "c" * 40}}
    prober = _tree_prober(tmp_path, monkeypatch, trees, ids=[importer])
    assert prober("r" * 40, [importer]) == {importer: "fail"}


def test_inventory_mentions_do_not_block_independent_pytest_and_size(tmp_path, monkeypatch, no_process):
    """Two imports belong to the family; data mentions in both searches do not."""
    revision = "r" * 40
    paths = [f"tests/test_{name}.py" for name in ("importer", "peer", "inventory", "neighbour")]
    ids = [f"{path}::test_red" for path in paths]
    module = "tests/test_sized.py"
    size_id = f"{module}::test_measure"
    inventory = "tests/protected_tests.json"
    trees = {HEAD: {**dict.fromkeys(paths, "b" * 40), PROTECTED: "a" * 40},
             revision: {**dict.fromkeys(paths, "b" * 40), PROTECTED: "z" * 40}}
    ran, measured = [], []
    prober = _tree_prober(tmp_path, monkeypatch, trees, ids=ids, ran=ran,
                         importers={(revision, "test_repo_health_smoke"): [*paths[:3], inventory],
                                    (revision, "test_importer"): [inventory, "tests/fixtures/notes.md"]},
                         size_modules={revision: [PROTECTED, *paths[:2], module]}, sources={
                             paths[0]: "from tests.test_repo_health_smoke import value\n",
                             paths[1]: "from . import test_repo_health_smoke as fixture\n",
                             paths[2]: f"INVENTORY = ({PROTECTED!r},)\n",
                         })

    def size_run(args, directory, *, cwd, env):
        measured.append(args)
        return _results(directory, [_row(size_id)])

    prober.size = lambda tree, directory, modules: pmc.size_session(
        directory, base=revision, cwd=tree, modules=modules, run=size_run)
    answers = prober(revision, [*ids, f"size {size_id}"])
    assert ran == [ids[2:]] and all(answers[nodeid] == "fail" for nodeid in ids[2:])
    assert measured == [["-m", "size_ratchet", module]] and answers[f"size {size_id}"] == "pass"
    for nodeid, path in zip(ids[:2], paths):
        assert answers[nodeid][0] == "error" and PROTECTED in answers[nodeid][1] and path in answers[nodeid][1]


def test_an_excluded_id_is_unattributed_with_the_probe_that_could_not_answer_it(tmp_path, monkeypatch, no_process):
    importer = "tests/test_importer.py::test_red"
    path = importer.split("::", 1)[0]
    trees = {HEAD: {PROTECTED: "a" * 40}, "b" * 40: {PROTECTED: "a" * 40, path: "c" * 40},
             "m" * 40: {PROTECTED: "z" * 40, path: "c" * 40}}
    prober = _tree_prober(tmp_path, monkeypatch, trees, ids=[importer], outcome="passed",
                         importers={("m" * 40, "test_repo_health_smoke"): [path]},
                         sources={path: "import tests.test_repo_health_smoke\n"})
    [verdict] = pmc.bisect(["b" * 40, "m" * 40, HEAD], [importer], prober)
    assert verdict["verdict"] == "unattributed"
    assert importer in verdict["why"] and "m" * 12 in verdict["why"] and PROTECTED in verdict["why"]


@pytest.mark.parametrize("hook", ["conftest.py", "tests/conftest.py", "tests/helper.py", "plugin", "computed plugin",
                                  "pyproject.toml", "tests/nested/pytest.ini"])
def test_a_shared_dependency_withholds_only_python_answers(tmp_path, monkeypatch, no_process, hook):
    neighbour = ("tests/nested/" if hook.startswith("tests/nested/") else "tests/") + "test_neighbour.py::test_red"
    sized = f"size {neighbour}"
    lint = "ruff F401 tests/gone.py name.py"  # even with a .py suffix this is a lint ID
    trees = {HEAD: {PROTECTED: "a" * 40}, "r" * 40: {PROTECTED: "z" * 40}}
    if hook.endswith("plugin"):
        declaration = ('pytest_plugins = ["tests.test_importer"]\n' if hook == "plugin" else
                       'PLUGIN = "tests.test_importer"\npytest_plugins = [PLUGIN]\n')
        carrier, hook = "tests/test_importer.py", "tests/conftest.py"
        importers = {("r" * 40, "test_repo_health_smoke"): [carrier], ("r" * 40, "test_importer"): [hook]}
        sources = {carrier: "from . import test_repo_health_smoke\n", hook: declaration}
    else:
        importers = {("r" * 40, "test_repo_health_smoke"): [hook]}
        sources = {hook: "from tests import test_repo_health_smoke\n"} if hook.endswith(".py") else {}
    ran, measured = [], []
    prober = _tree_prober(tmp_path, monkeypatch, trees, ids=[neighbour], importers=importers, sources=sources,
                         ran=ran, measured=measured, size_modules={"r" * 40: [neighbour.split("::", 1)[0]]})
    answers = prober("r" * 40, [neighbour, sized, lint, pmc.NODE_LANE])
    assert answers[neighbour] == answers[sized] and answers[neighbour][0] == "error"
    assert hook in answers[neighbour][1] and not (ran or measured)
    assert answers[lint] == answers[pmc.NODE_LANE] == "pass"


@pytest.mark.parametrize("fault", ["entry", "timeout", "grep"])
def test_a_git_read_gap_withholds_only_python_answers(tmp_path, monkeypatch, no_process, fault):
    neighbour = "tests/test_neighbour.py::test_red"
    lint = "ruff F401 tests/gone.py name.py"
    trees = {HEAD: {PROTECTED: "a" * 40}, "r" * 40: {PROTECTED: "z" * 40}}
    calls, ran = [], []
    prober = _tree_prober(tmp_path, monkeypatch, trees, ids=[neighbour], unreadable=fault, calls=calls, ran=ran)
    assert prober("r" * 40, [lint, pmc.NODE_LANE]) == {lint: "pass", pmc.NODE_LANE: "pass"}
    assert any(args[0] == "ls-tree" for args, _ in calls)  # all lanes share the path-omission decision
    assert not any(args[0] == "show" or ":(glob)tests/**/*.py" in args for args, _ in calls)
    verdicts = {row["id"]: row for row in pmc.bisect(["r" * 40, HEAD], [neighbour, lint, pmc.NODE_LANE], prober)}
    assert verdicts[neighbour]["verdict"] == "unattributed" and PROTECTED in verdicts[neighbour]["why"] and not ran
    assert verdicts[lint]["verdict"] == verdicts[pmc.NODE_LANE]["verdict"] == "culprit"


@pytest.mark.parametrize("launcher", ["scripts/run_tests.py", "scripts/safe_test.py"])
def test_a_node_launcher_that_loads_the_family_does_not_run(tmp_path, monkeypatch, no_process, launcher):
    revision = "r" * 40
    lint = "ruff F401 tests/gone.py name"
    trees = {HEAD: {PROTECTED: "a" * 40}, revision: {PROTECTED: "z" * 40}}
    calls = []
    prober = _tree_prober(tmp_path, monkeypatch, trees, calls=calls,
                          importers={(revision, "test_repo_health_smoke"): [launcher]},
                          sources={launcher: "from tests import test_repo_health_smoke\n"})
    prober.node = lambda tree: pytest.fail("a dependent launcher must not run")
    answers = prober(revision, [pmc.NODE_LANE])
    assert answers[pmc.NODE_LANE][0] == "error" and launcher in answers[pmc.NODE_LANE][1]
    assert not any(args[0] == "worktree" for args, _ in calls)
    assert prober(revision, [pmc.NODE_LANE, lint]) == {**answers, lint: "pass"}


@pytest.mark.parametrize("unreadable", [None, "entry"], ids=["changed", "unreadable"])
def test_protected_lint_does_not_depend_on_neighbouring_probe_ids(tmp_path, monkeypatch, no_process, unreadable):
    revision = "r" * 40
    lint = f"ruff F401 {PROTECTED} name"
    path = "tests/test_neighbour.py"
    test = f"{path}::test_green"
    entries = {HEAD: {PROTECTED: "a" * 40}, revision: {PROTECTED: "z" * 40, path: "b" * 40}}
    calls = []
    prober = _tree_prober(tmp_path, monkeypatch, entries, ids=[test], unreadable=unreadable, calls=calls)
    prober.lint = lambda *args: pytest.fail("the protected file must not be linted")
    assert prober(revision, [lint]) == {lint: "error"}
    assert prober(revision, [lint, test])[lint] == "error"
    assert all("--no-checkout" in args for args, _ in calls if args[:2] == ("worktree", "add"))


def test_an_unparseable_test_candidate_does_not_erase_its_neighbour(tmp_path, monkeypatch, no_process):
    broken, plain = "tests/test_broken.py", "tests/test_plain.py"
    neighbour = f"{plain}::test_red"
    trees = {HEAD: {PROTECTED: "a" * 40}, "r" * 40: {PROTECTED: "z" * 40, plain: "c" * 40}}
    ran = []
    prober = _tree_prober(tmp_path, monkeypatch, trees, ids=[neighbour], ran=ran,
                         importers={("r" * 40, "test_repo_health_smoke"): [broken]},
                         sources={broken: "from .test_repo_health_smoke import (\n"})
    answers = prober("r" * 40, [broken, neighbour])
    assert answers[broken][0] == "error" and "could not be read" in answers[broken][1]
    assert answers[neighbour] == "fail" and ran == [[neighbour]]


@pytest.mark.parametrize("outcome,pytest_exit,runner_exit,stale,expected", [
    pytest.param("passed", 0, 0, False, "pass", id="fresh-pass"),
    pytest.param("failed", 1, 1, False, "fail", id="fresh-red"),
    pytest.param("passed", 1, 1, False, "error", id="red-exit-no-red-row"),
    pytest.param("failed", 0, 0, False, "error", id="green-exit-red-row"),
    pytest.param("passed", 0, 1, False, "error", id="wrapper-disagrees"),
    pytest.param("failed", 2, 2, False, "error", id="abnormal-exit"),
    pytest.param("passed", 0, 0, True, "error", id="stale-pass"),
    pytest.param("failed", 1, 1, True, "error", id="stale-red"),
])
def test_historical_pytest_and_size_need_a_fresh_consistent_export(
        tmp_path, monkeypatch, no_process, outcome, pytest_exit, runner_exit, stale, expected):
    revision = "r" * 40
    path, module = "tests/test_neighbour.py", "tests/test_sized.py"
    nodeid, sized = f"{path}::test_case", f"{module}::test_measure"
    trees = {HEAD: {PROTECTED: "a" * 40}, revision: {PROTECTED: "a" * 40, path: "c" * 40}}
    ran = []
    prober = _tree_prober(tmp_path, monkeypatch, trees, ids=[nodeid], size_modules={revision: [module]})
    if stale:  # the same evidence directories a second `attribute` invocation would find
        prefix = prober.evidence / "probes" / f"01-{revision[:12]}"
        _results(prefix, [_row(nodeid, outcome)], exit_code=pytest_exit)
        _results(prefix.with_name(prefix.name + "-size"), [_row(sized, outcome)], exit_code=pytest_exit)

    def run(args, directory, *, cwd, env=None):
        ran.append(args)
        if not stale:
            _results(directory, [_row(sized if env else nodeid, outcome)], exit_code=pytest_exit)
        return runner_exit

    prober.run = run
    prober.size = lambda tree, directory, modules: pmc.size_session(
        directory, base=revision, cwd=tree, modules=modules, run=run)
    ids = [nodeid, f"size {sized}"]
    assert prober(revision, ids) == dict.fromkeys(ids, expected)
    assert ran == [["-m", "size_ratchet", module], [nodeid]]


def test_the_size_lane_of_a_revision_is_measured_whole_and_leaves_the_family(tmp_path, monkeypatch, no_process):
    sized = "size tests/test_other.py::test_size"
    trees = {HEAD: {PROTECTED: "a" * 40}, "r" * 40: {PROTECTED: "z" * 40}}
    measured = []
    prober = _tree_prober(tmp_path, monkeypatch, trees, ids=[sized], measured=measured,
                     importers={("r" * 40, "test_repo_health_smoke"): [PROTECTED]},
                     size_modules={"r" * 40: [PROTECTED, "tests/test_other.py"]})
    prober("r" * 40, [sized])
    assert measured == [["tests/test_other.py"]]  # the whole lane of that revision, without the family


def test_the_reader_asks_git_about_the_tree_it_questions(tmp_path, monkeypatch, no_process):
    """The reads name the tree they ask about, and the names Git writes come back relative."""
    importer = "tests/test_importer.py::test_red"
    sized = "size tests/test_other.py::test_size"
    calls = []
    trees = {HEAD: {PROTECTED: "a" * 40}, "r" * 40: {PROTECTED: "z" * 40}}
    prober = _tree_prober(tmp_path, monkeypatch, trees, ids=[importer, sized], calls=calls,
                     importers={("r" * 40, "test_repo_health_smoke"): ["tests/test_importer.py"]},
                     sources={"tests/test_importer.py": "from tests import test_repo_health_smoke\n"},
                     size_modules={"r" * 40: ["tests/test_other.py"]})
    prober("r" * 40, [importer, sized])
    asked = [args for args, _ in calls if args[0] == "grep"]
    assert [args[3] for args in asked] == ["-F", "-F", "-E"]
    assert all(args[args.index("--") - 1] == "r" * 40 and f":(exclude,literal){PROTECTED}" in args for args in asked)
    assert [args[1] for args, _ in calls if args[0] == "show"] == [f"{'r' * 40}:tests/test_importer.py"]


def test_a_changed_protected_file_leaves_its_family_unanswered(tmp_path, monkeypatch, no_process):
    """The protected file changed there: its own module, the module that imports it and that
    module's whole-module size id are not answered, and the neighbour is still probed."""
    importer = "tests/test_importer.py::test_red"
    importer_module = "size tests/test_importer.py"
    direct = [f"{PROTECTED}::test_case", f"size {PROTECTED}"]
    outside = "size tests/test_other.py::test_size"
    neighbour = "tests/test_neighbour.py::test_red"
    trees = {HEAD: {PROTECTED: "a" * 40, "tests/test_importer.py": "b" * 40,
                     "tests/test_neighbour.py": "c" * 40},
             "r" * 40: {PROTECTED: "z" * 40, "tests/test_importer.py": "b" * 40,
                         "tests/test_neighbour.py": "c" * 40}}
    measured = []
    prober = _tree_prober(tmp_path, monkeypatch, trees, ids=[importer, neighbour, outside], measured=measured,
                     importers={("r" * 40, "test_repo_health_smoke"): ["tests/test_importer.py"]},
                     sources={"tests/test_importer.py": "from .test_repo_health_smoke import value\n"},
                     size_modules={"r" * 40: [PROTECTED, "tests/test_importer.py", "tests/test_other.py"]})
    answers = prober("r" * 40, [importer, importer_module, *direct, outside, neighbour])
    assert answers[neighbour] == "fail"  # the neighbour is probed, not absent
    for nodeid in (importer, importer_module, *direct):
        assert answers[nodeid][0] == "error"
    assert PROTECTED in answers[importer][1] and "tests/test_importer.py" in answers[importer][1]
    assert measured == [["tests/test_other.py"]]  # the lane without the family it may not answer
    alone = prober("r" * 40, [importer])  # a single id is no answer either
    assert alone[importer][0] == "error"


@pytest.mark.parametrize("case", ["tracked", "absent", "restore-refused", "unexpected-path"])
def test_a_partial_probe_requires_the_protected_path_to_stay_off_disk(tmp_path, monkeypatch, no_process, case):
    revision = "r" * 40
    tree = tmp_path / "work" / f"pmc-01-{revision[:12]}"
    path, content = "tests/test_neighbour.py", "\ndef test_green(): pass\n"
    nodeid, linted, protected_lint = f"{path}::test_green", f"ruff F401 {path} name", f"ruff F401 {PROTECTED} name"
    tracked = case != "absent"
    entries = {path: "b" * 40, **({PROTECTED: "z" * 40} if tracked else {})}
    calls, ran, removed, lint_calls, node_calls = [], [], [], [], []
    prober = _tree_prober(tmp_path, monkeypatch, {HEAD: {PROTECTED: "a" * 40}, revision: entries},
                          ids=[nodeid], outcome="passed", calls=calls, ran=ran)
    read, run = pmc._git, prober.run

    def git(*args, cwd=None, strip=True):
        result = read(*args, cwd=cwd, strip=strip)
        if args[:2] == ("worktree", "add"):
            assert args[2:] == ("--detach", "--quiet", "--no-checkout", str(tree), revision)
            (tree / "tests").mkdir(parents=True)
        elif args[0] == "read-tree":
            assert cwd == tree and args[1:] == (revision,)
        elif args[0] == "restore":
            assert cwd == tree and args[1:] == (f"--source={revision}", "--worktree", "--", ".",
                                                f":(exclude,literal){PROTECTED}")
            if case == "restore-refused":
                raise subprocess.CalledProcessError(128, args)
            (tree / path).write_text(content, encoding="utf-8")
            if case == "unexpected-path":
                (tree / PROTECTED).mkdir()
        elif args[0] == "ls-files":
            assert cwd == tree and args[1:] == ("--", PROTECTED)
            return PROTECTED if tracked else ""
        elif args[0] == "update-index":
            assert cwd == tree and tracked and args[1:] == ("--skip-worktree", "--", PROTECTED)
        return result

    def run_projected(args, directory, *, cwd):
        assert cwd == tree and not os.path.lexists(tree / PROTECTED)
        assert (tree / path).read_text(encoding="utf-8") == content
        return run(args, directory, cwd=cwd)

    def lint(cwd, paths):
        lint_calls.append((cwd, paths))
        return {linted: "unused name"}

    monkeypatch.setattr(pmc, "_git", git)
    monkeypatch.setattr(pmc, "remove_worktree", removed.append)
    prober.run, prober.lint = run_projected, lint
    prober.node = lambda cwd: node_calls.append(cwd) or 0
    ids = [nodeid, linted, protected_lint, pmc.NODE_LANE]
    answers = prober(revision, ids)
    assert removed == [tree]
    setup = {"worktree", "read-tree", "restore", "ls-files", "update-index"}
    want = ["worktree", "read-tree", "restore"]
    if case != "restore-refused":
        want += ["ls-files", "update-index"] if tracked else ["ls-files"]
    assert [args[0] for args, _ in calls if args[0] in setup] == want
    if case in ("restore-refused", "unexpected-path"):
        assert answers == dict.fromkeys(ids, "error") and not (ran or lint_calls or node_calls)
    else:
        assert ran == [[nodeid]] and lint_calls == [(tree, [path])] and node_calls == [tree]
        assert answers == {nodeid: "pass", linted: "fail", protected_lint: "error" if tracked else "pass",
                           pmc.NODE_LANE: "pass"}


def test_the_reader_names_what_git_names_and_refuses_what_it_cannot_place():
    """A tree's names carry the tree they were asked for; a foreign tree is no answer at all."""
    assert pmc.grep_names("abc", "abc:tests/one.py\0abc:tests/two.py\0") == ["tests/one.py", "tests/two.py"]
    assert pmc.grep_names("", "tests/one.py\0") == ["tests/one.py"]  # the checkout's own names stay
    assert pmc.grep_names("abc", "") == []  # nothing matched there
    assert pmc.grep_names("abc", "def:tests/one.py\0") is None  # a name of another tree


def test_size_session_measures_the_modules_it_is_given_and_none_when_the_list_is_empty(tmp_path):
    seen = []

    def run(args, directory, *, cwd, env):
        seen.append(list(args))
        _results(directory, [])
        return 0

    session = pmc.size_session(tmp_path / "size", base="b" * 40, cwd=tmp_path, run=run,
                               modules=["tests/test_a.py"])
    assert session["final"] and seen == [["-m", "size_ratchet", "tests/test_a.py"]]
    assert pmc.size_session(tmp_path / "empty", base="b" * 40, cwd=tmp_path, run=run, modules=[]) is None
    assert len(seen) == 1  # an empty lane measures nothing


# --- summary ---------------------------------------------------------------------------------

@pytest.mark.parametrize("verdict", ["culprit", "pre-existing", "unattributed"])
def test_attribution_explains_a_failed_check_without_changing_red(tmp_path, verdict):
    nodeid = "tests/t.py::test_red"
    (tmp_path / "classification.json").write_text(json.dumps({"complete": True,
        "failed": [{"id": nodeid, "why": "failed in call", "pass": "serial", "retry": "failed"}],
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
        "failed": [{"id": "tests/t.py::test_a", "why": "failed in call", "pass": "serial", "retry": "failed"}], "flaky": [],
        "problems": [], "selection": {}}), encoding="utf-8")
    (tmp_path / "attribution.json").write_text(json.dumps({"verdicts": [
        {"id": "tests/t.py::test_a", "verdict": "culprit", "culprit": "c" * 40, "last_good": "b" * 40}]}), encoding="utf-8")
    (tmp_path / "routing.json").write_text(json.dumps({
        "comments": [2067], "created": [100], "updated": [], "closed": [5], "warnings": ["comment on #9: HTTP 403"],
        "handed": [{"ids": ["tests/t.py::test_a"], "pr": 2067, "issue": 100}]}), encoding="utf-8")
    text, red = pmc.render_summary(tmp_path)
    assert red and ("| `tests/t.py::test_a` | failed in call (retry: failed) | culprit | `cccccccccccc` | `bbbbbbbbbbbb` "
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


@pytest.mark.parametrize(("fault", "problem"), [
    ("none", ""), ("persistent test failure", ""), ("unconfirmed empty", "test inventory recorded no collection"),
    ("missing", "expected and executed test ids differ (missing or duplicate execution)"),
    ("duplicate", "expected and executed test ids differ (missing or duplicate execution)"),
    ("disagreement", "worker collections disagree"), ("unfinished", "test inventory has no terminal exit code"),
    ("unreadable", "test inventory unavailable"), ("floor", "floor modules collected no tests"),
    # Without its export a pass that ran tests is unmeasured, whatever its phase journal shows; an empty pass's
    # inventory still proves it.
    ("lost results", "no valid results.json"), ("damaged results", "no valid results.json"), ("lost empty results", ""),
    ("no empty diagnostics", ""), ("partial journal with intact results", ""),
    ("partial journal", "no valid results.json"), ("torn journal", "no valid results.json"),
    ("wrong journal IDs", "no valid results.json"), ("collector skipped", "collector skipped"),
    ("subtest only", "no valid results.json"), ("partial results", "missing selected test reports"),
    ("interrupted", "inconsistent pytest exit/collection"), ("runner mismatch", "runner exit 2 disagrees with pytest exit 1"),
    ("missing Node", "the complete Node lane did not run to completion"),
])
def test_selected_inventory_and_an_independent_test_retry_reach_the_workflow_result(
        run_tests, tmp_path, monkeypatch, fault, problem):
    from ouroboros import test_selection

    module = "tests/test_example.py"
    flaky, other = f"{module}::test_retry[x@y::z]", f"{module}::test_other"
    files = (module, "tests/test_empty_floor.py") if fault == "floor" else (module,)
    monkeypatch.setattr(test_selection, "FLOOR_TESTS", dict.fromkeys(files, "fixture"))
    chosen = test_selection.TestSelection("selected", HEAD, files, (), (), "policy", len(files))
    monkeypatch.setattr(test_selection, "select_tests", lambda *a, **k: chosen)
    monkeypatch.setattr(run_tests, "_node_lane", lambda: (int(fault == "missing Node"), ""))
    monkeypatch.delenv("OUROBOROS_PREFLIGHT_SERIAL", raising=False)
    output, summary = tmp_path / "output", tmp_path / "summary"
    monkeypatch.setenv("GITHUB_OUTPUT", str(output))
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary))

    def first_pass(label, argv, cwd):
        if label == "warm":
            return 0
        expected = [flaky, other] if label == "parallel" else []
        reports = _whole(flaky, _row(flaky, "failed")) if expected else []
        if expected and fault != "missing":
            reports.extend(_whole(other))
        code = (2 if fault == "interrupted" else 1) if expected else 5
        directory = tmp_path / label
        exported = reports[:-1] if expected and fault == "partial results" else reports
        _results(directory, exported, exit_code=code, selected=expected)
        if expected and fault == "partial journal with intact results":
            (directory / "events.jsonl").write_text(json.dumps({"event": "start", "nodeid": other}))
        if not expected and fault == "no empty diagnostics":
            (directory / "results.json").unlink()
        if (expected and fault in ("lost results", "damaged results", "partial journal", "torn journal",
                                   "wrong journal IDs", "subtest only")) or (not expected and fault in ("lost empty results", "collector skipped")):
            journal = [dict(row) for row in reports]
            if fault == "partial journal":
                journal.pop()
            if fault == "wrong journal IDs":
                for row in journal:
                    if row["nodeid"] == other:
                        row["nodeid"] = "tests/test_example.py::test_unrelated"
            if fault == "subtest only":
                journal[-2]["is_subtest"] = True
            text = "\n".join(json.dumps({"event": "report", **row}) for row in journal)
            (directory / "events.jsonl").write_text(text + ("\n{" if fault == "torn journal" else ""), encoding="utf-8")
            if fault == "damaged results":
                (directory / "results.json").write_text("{", encoding="utf-8")
            else:
                (directory / "results.json").unlink()
        rows = [{"event": "start"}]
        if not expected and fault == "collector skipped":
            rows.append({"event": "collector", "outcome": "skipped"})
        if expected or fault != "unconfirmed empty":
            rows.append({"event": "collection", "ids": expected})
        if expected and fault == "disagreement":
            rows.append({"event": "collection", "ids": [other, flaky]})
        rows.extend({"event": "report", "nodeid": row["nodeid"], "when": row["phase"], "outcome": row["outcome"]}
                    for row in reports if row["phase"] == "teardown" or row["outcome"] == "failed")
        if expected and fault == "duplicate":
            rows.append(rows[-1])  # the other test's second teardown; its id is still in both sets
        if not expected or fault != "unfinished":
            rows.append({"event": "finish", "exitcode": code})
        data = "\n".join(map(json.dumps, rows)) if not expected or fault != "unreadable" else "{"
        (tmp_path / f"inventory-{label}.jsonl").write_text(data, encoding="utf-8")
        return 2 if expected and fault == "runner mismatch" else code

    monkeypatch.setattr(run_tests, "_run", first_pass)
    expected_code = 2 if fault in ("interrupted", "runner mismatch") else 1
    assert run_tests.main(["--selected", "--keep-going", "--evidence-dir", str(tmp_path)]) == expected_code

    retries = []
    persistent = fault == "persistent test failure"

    def retry(command, *, cwd, env):
        directory = Path(command[command.index("--ci-evidence-dir") + 1])
        retries.append((directory / "argsfile.txt").read_text(encoding="ascii").splitlines())
        _results(directory, _whole(flaky, _row(flaky, "failed" if persistent else "passed")), selected=[flaky])
        return int(persistent)

    monkeypatch.setattr(pmc.subprocess, "call", retry)
    assert pmc.main(["classify", "--evidence-dir", str(tmp_path), "--head", HEAD]) == 0
    assert retries == [[flaky]]
    classified = json.loads((tmp_path / "classification.json").read_text(encoding="utf-8"))
    assert [row["id"] for row in classified["flaky"]] == ([] if persistent else [flaky])
    assert [row["id"] for row in classified["failed"]] == ([flaky] if persistent else [])
    assert pmc.main(["summary", "--evidence-dir", str(tmp_path), "--os", "windows"]) == int(persistent or bool(problem))
    text = summary.read_text(encoding="utf-8")
    assert flaky in text and ("Failed checks (1)" if persistent else "Flaky (1)") in text
    selection = json.loads((tmp_path / "selection.json").read_text(encoding="utf-8"))
    label = "serial" if fault == "unconfirmed empty" else "parallel"
    cause = (selection.get("floor_problem") or selection.get("node_problem") or selection["inventory"][label].get("problem"))
    if problem and fault not in ("partial results", "interrupted", "lost results", "damaged results", "partial journal",
                                  "torn journal", "wrong journal IDs", "subtest only", "collector skipped"):
        assert problem in cause and problem in text
    elif problem:
        assert problem in text
    else:
        assert not cause
    assert ("— RED" if problem or persistent else "— GREEN") in text
    assert classified["selection_executed"] is (not problem)
    outputs = dict(line.split("=", 1) for line in output.read_text(encoding="utf-8").splitlines())
    assert outputs["window_scope"] == ""  # this fixture has no shared base receipt


def test_each_commit_reaches_the_selection_and_needs_selected(run_tests, monkeypatch, tmp_path):
    from ouroboros import test_selection

    asked = []

    def select(repo, **options):
        asked.append(options)
        raise ValueError("stop after reading the selection inputs")

    monkeypatch.setattr(test_selection, "select_tests", select)
    for flags, each in (([], False), (["--each-commit"], True)):
        assert run_tests.main(["--selected", "--base", "HEAD", *flags, "--evidence-dir", str(tmp_path)]) == 2
        assert asked[-1]["each_commit"] is each
    with pytest.raises(SystemExit):
        run_tests.main(["--each-commit"])


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
        monkeypatch.setattr(run_tests, "_node_lane", lambda: (node, ""))
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


@pytest.mark.parametrize(("node", "parallel", "serial", "exit_code"), [
    (0, 1, 137, 137),  # a later abnormal status is not hidden by an earlier normal red
    (1, 0, 2, 2), (0, 137, 1, 137), (0, 2, 137, 2),  # the first abnormal status stands
    (0, 1, 5, 1), (0, 0, 5, 0),  # an empty pass's 5 is no error
    (0, 1, 0, 1), (1, 1, 1, 1),
])
def test_keep_going_never_lets_a_normal_red_hide_a_later_abnormal_exit(
        run_tests, monkeypatch, tmp_path, node, parallel, serial, exit_code):
    from ouroboros import preflight_inventory, test_selection

    chosen = test_selection.TestSelection("selected", "base", ("tests/test_floor.py",), (), (), "policy", 3)
    monkeypatch.setattr(test_selection, "select_tests", lambda *a, **k: chosen)
    monkeypatch.setattr(preflight_inventory, "read_inventory", lambda root, label: ({}, ""))
    monkeypatch.setattr(preflight_inventory, "inventory_problem", lambda selection, inventory: "")
    monkeypatch.delenv("GITHUB_OUTPUT", raising=False)
    monkeypatch.delenv("OUROBOROS_PREFLIGHT_SERIAL", raising=False)
    codes = {"parallel": parallel, "serial": serial}
    monkeypatch.setattr(run_tests, "_node_lane", lambda: (node, ""))
    monkeypatch.setattr(run_tests, "_run", lambda label, argv, cwd: codes.get(label, 0))
    assert run_tests.main(["--selected", "--keep-going", "--evidence-dir", str(tmp_path)]) == exit_code


# --- what a selected pass reported (counts) ---------------------------------------------------

IDENTITY = {"sha": HEAD, "run_id": "7", "run_attempt": "1"}
SESSION = "6f1c0de"


def _export(directory, rows, *, exit_code=None, github=None, system="Linux", session=SESSION, **options):
    """A pass's final export and its journal header as tests/ci_evidence.py writes them: one session, run and system."""
    code = _results(directory, rows, exit_code=exit_code, github=IDENTITY if github is None else github, **options)
    path = Path(directory) / "results.json"
    path.write_text(json.dumps({**json.loads(path.read_text(encoding="utf-8")), "session": session,
                                "environment": {"platform": system}}), encoding="utf-8")
    (Path(directory) / "events.jsonl").write_text(json.dumps({"event": "session", "session": session}) + "\n", encoding="utf-8")
    return code


def _private(evidence, *rows, label="parallel"):
    """inventory-<lane>.jsonl as the inventory plugin writes it: one JSON object per line."""
    Path(evidence).mkdir(parents=True, exist_ok=True)
    (Path(evidence) / f"inventory-{label}.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _written(github=IDENTITY, system="linux", session=SESSION, label="parallel"):
    """selection.json as run_tests.py leaves it: the run it wrote under (a legacy one names none) and each read pass's session."""
    row = {"outcomes": {"parallel": "failure", "serial": "skipped"}, "inventory": {label: {"session": session}} if session else {}}
    if github is not None:
        row.update(github=github, system=system)
    return row


def _counts(evidence, written=None, *, whole=False, label="parallel"):
    return pmc.pass_counts(evidence, label, _written() if written is None else written, identity=IDENTITY, system="linux", whole=whole)


def test_unfinished_starts_are_counted_per_start_and_are_no_failed_report(tmp_path):
    shared, other, crashed = "tests/t.py::test_shared[masked]", "tests/t.py::test_other", "tests/t.py::test_crashed"
    events = [{"event": "start", "nodeid": shared}, {"event": "start", "nodeid": shared},  # two raw tests, one public id
              {"event": "finish", "nodeid": shared}, {"event": "start", "nodeid": other},
              {"event": "start", "nodeid": crashed},
              {"event": "report", "nodeid": crashed, "phase": "crash", "outcome": "failed"}]
    (tmp_path / "events.jsonl").write_text("".join(json.dumps(row) + "\n" for row in events), encoding="utf-8")
    session = pmc.read_session(tmp_path)
    # `running` names a public id once, so the shared id's second start is not in it; the count keeps it.
    assert session["running"] == [other] and session["unfinished_starts"] == 2
    assert [row["phase"] for row in session["rows"]] == ["crash"]  # a start is no report, and a crash row ends its start
    # An export that ended shows the tests that finished even where their journal rows were lost; a start it does not show stays.
    exported = tmp_path / "exported"
    _results(exported, [*_whole(shared), _row(other, "failed")], exit_code=1)
    (exported / "events.jsonl").write_text("".join(json.dumps({"event": "start", "nodeid": node}) + "\n" for node in (shared, other)),
                                           encoding="utf-8")
    session = pmc.read_session(exported)
    assert session["running"] == [other] and session["unfinished_starts"] == 1


@pytest.mark.parametrize(("case", "exit_code", "whole", "failed", "complete", "any_red"), [
    ("complete green", 0, True, False, True, False),
    ("complete red", 1, True, True, True, True),
    ("interrupted export", 2, True, False, False, None),  # an export that ended is not pass coverage
    ("interrupted red", 2, True, True, False, True),  # its positive survives, no zero is asserted
    ("abnormal release", 0, False, False, False, None),  # the pass did not validate (the runner's exit disagrees)
    ("abnormal release red", 1, False, True, False, True),
])
def test_an_export_names_complete_counts_only_for_a_pass_that_ended_normally_and_validated(
        tmp_path, case, exit_code, whole, failed, complete, any_red):
    one = "tests/t.py::test_one"
    _export(tmp_path / "parallel", _whole(one, _row(one, "failed") if failed else None), exit_code=exit_code)
    seen = _counts(tmp_path, whole=whole)
    assert seen["reports"] == {"source": "export", "complete": complete, "failed": {"setup": 0, "call": int(failed), "teardown": 0},
                               "failed_subtests": 0, "crashed": 0, "unfinished_starts": 0}
    assert seen["collectors"] == {"source": "export", "complete": complete, "failed": 0, "skipped": 0}
    assert seen["any_red"] is any_red and pmc.valid_pass_counts(seen)


def test_a_session_that_only_started_has_an_unfinished_start_and_no_failure(tmp_path):
    directory = tmp_path / "parallel"
    directory.mkdir()
    events = [{"event": "session", "session": SESSION}, {"event": "start", "nodeid": "tests/t.py::test_slow"}]
    (directory / "events.jsonl").write_text("".join(json.dumps(row) + "\n" for row in events), encoding="utf-8")
    _private(tmp_path, {"event": "start"}, {"event": "evidence_session", "session": SESSION})
    seen = _counts(tmp_path)
    assert seen["reports"] == {"source": "journal", "complete": False, "failed": {"setup": 0, "call": 0, "teardown": 0},
                               "failed_subtests": 0, "crashed": 0, "unfinished_starts": 1}
    assert seen["collectors"] == {"source": "inventory", "complete": False, "failed": 0, "skipped": 0}
    assert seen["any_red"] is None and pmc.valid_pass_counts(seen)  # nothing failed, and nothing proves it complete


def test_a_reported_worker_crash_is_a_failure_and_an_unreported_start_is_not(tmp_path):
    directory = tmp_path / "parallel"
    directory.mkdir()
    events = [{"event": "session", "session": SESSION}, {"event": "start", "nodeid": "tests/t.py::test_lost"},
              {"event": "report", "nodeid": "tests/t.py::test_lost", "phase": "crash", "outcome": "failed"},
              {"event": "start", "nodeid": "tests/t.py::test_slow"}]
    (directory / "events.jsonl").write_text("".join(json.dumps(row) + "\n" for row in events), encoding="utf-8")
    _private(tmp_path, {"event": "start"}, {"event": "evidence_session", "session": SESSION})
    seen = _counts(tmp_path)
    assert seen["reports"]["crashed"] == 1 and seen["reports"]["unfinished_starts"] == 1
    assert seen["reports"]["failed"] == {"setup": 0, "call": 0, "teardown": 0}  # neither is a failed setup, call or teardown report
    assert seen["any_red"] is True and seen["reports"]["complete"] is False and pmc.valid_pass_counts(seen)


def test_two_raw_tests_with_one_public_id_keep_both_failed_reports(tmp_path):
    masked = "tests/t.py::test_name[redacted]"
    rows = [*_whole(masked, _row(masked, "failed")), *_whole(masked, _row(masked, "failed"))]
    _export(tmp_path / "parallel", rows, exit_code=1, selected=[masked, masked])
    assert list(pmc._failure_phases(pmc.read_session(tmp_path / "parallel", session_id=SESSION))) == [masked]  # the routing map names it once
    seen = _counts(tmp_path, whole=True)
    assert seen["reports"]["failed"] == {"setup": 0, "call": 2, "teardown": 0} and seen["any_red"] is True


def test_collector_failures_survive_a_lost_public_journal_and_export_only_when_the_pass_binds(tmp_path):
    clean = [{"event": "start"}, {"event": "evidence_session", "session": SESSION}, {"event": "collector", "outcome": "failed"},
             {"event": "collector", "outcome": "skipped"}, {"event": "collector", "outcome": "failed"},
             {"event": "finish", "exitcode": 1}]
    _private(tmp_path, *clean)  # no pass directory at all: neither the public journal nor the export could be written
    seen = _counts(tmp_path)
    assert seen["collectors"] == {"source": "inventory", "complete": True, "failed": 2, "skipped": 1}
    assert seen["reports"]["source"] == "none" and seen["reports"]["failed"] is None  # the reports stay unknown
    assert seen["any_red"] is True and pmc.valid_pass_counts(seen)
    # A torn tail loses only its own line: the positives stand, completeness does not.
    (tmp_path / "inventory-parallel.jsonl").write_text("".join(json.dumps(row) + "\n" for row in clean[:4]) + '{"event": "coll',
                                                       encoding="utf-8")
    assert _counts(tmp_path)["collectors"] == {"source": "inventory", "complete": False, "failed": 1, "skipped": 1}
    # What cannot be bound to this run, attempt and system, or names another session, is unknown.
    _private(tmp_path, *clean)
    for written in (_written(github=None),  # a legacy selection.json names no run
                    _written(github={**IDENTITY, "run_attempt": "2"}),  # another attempt
                    _written(system="windows"),  # another system
                    _written(session="older")):  # the pass recorded another session
        seen = _counts(tmp_path, written)
        assert seen["collectors"]["source"] == "none" and seen["any_red"] is None and pmc.valid_pass_counts(seen)
    # A public journal of an older session counts nothing; the private one still binds through this run's own record.
    (tmp_path / "parallel").mkdir()
    stale = [{"event": "session", "session": "older"}, {"event": "start", "nodeid": "tests/t.py::test_old"}]
    (tmp_path / "parallel" / "events.jsonl").write_text("".join(json.dumps(row) + "\n" for row in stale), encoding="utf-8")
    seen = _counts(tmp_path)
    assert seen["reports"]["source"] == "none" and seen["collectors"]["failed"] == 2


def test_a_foreign_export_counts_no_reports_but_leaves_the_private_collector_facts(tmp_path):
    one = "tests/t.py::test_one"
    _export(tmp_path / "parallel", _whole(one, _row(one, "failed")), exit_code=1, github={**IDENTITY, "run_attempt": "2"})
    _private(tmp_path, {"event": "start"}, {"event": "evidence_session", "session": SESSION},
             {"event": "collector", "outcome": "failed"}, {"event": "finish", "exitcode": 1})
    seen = _counts(tmp_path, whole=True)
    assert seen["reports"] == {"source": "none", "complete": False, "failed": None, "failed_subtests": None, "crashed": None,
                               "unfinished_starts": None}  # an explicitly foreign final export is not relabelled a journal
    assert seen["collectors"] == {"source": "inventory", "complete": True, "failed": 1, "skipped": 0} and seen["any_red"] is True


def test_the_count_reader_refuses_what_pass_counts_never_writes(tmp_path):
    import copy

    _export(tmp_path / "parallel", _whole("tests/t.py::test_one"), exit_code=0)
    seen = _counts(tmp_path, whole=True)
    assert pmc.valid_pass_counts(seen) and seen["any_red"] is False
    assert pmc.valid_pass_counts(pmc.pass_counts(tmp_path / "missing", "parallel", {}, identity=IDENTITY, system="linux"))

    def refused(change):
        value = copy.deepcopy(seen)
        change(value)
        return not pmc.valid_pass_counts(value)

    assert refused(lambda v: v["reports"].update(unfinished_starts=1))  # complete counts with an unfinished start
    assert refused(lambda v: v["reports"].update(source="journal"))  # a journal is never complete
    assert refused(lambda v: v.update(any_red=None))  # a complete zero names no red: False, not unknown
    assert refused(lambda v: v["reports"]["failed"].update(call=True))  # a bool is no count
    assert refused(lambda v: v["collectors"].update(source="none"))  # unknown counts are all None
    assert not pmc.valid_pass_counts(None) and not pmc.valid_pass_counts({})


def test_a_deferral_that_selected_nothing_past_the_floor_is_a_problem_and_promotes_no_window(tmp_path):
    _selection(tmp_path)
    selection = json.loads((tmp_path / "selection.json").read_text(encoding="utf-8"))
    window = {"base": "b" * 40, "head": HEAD, "continuous": True}
    genuine = pmc.classify(tmp_path)  # a genuine deferral, or none, carries no problem: measured as before
    assert genuine["problems"] == [] and pmc._window_measured(window, {**genuine, "head": HEAD}, "windows")
    fixed = "the deferral could not select the other changes: this leg ran only the contract floor"
    (tmp_path / "selection.json").write_text(json.dumps({**selection, "deferred_full": True, "selection_problem": fixed}), encoding="utf-8")
    record = pmc.classify(tmp_path)
    assert record["problems"] == [fixed] and record["selection_executed"] is False
    assert not pmc._window_measured(window, {**record, "head": HEAD}, "windows")
    (tmp_path / "classification.json").write_text(json.dumps({**record, "head": HEAD}), encoding="utf-8")
    text, red = pmc.render_summary(tmp_path, system="windows")
    assert red and fixed in text
