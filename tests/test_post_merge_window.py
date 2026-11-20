"""The post-merge window (scripts/post_merge_window.py): the base both legs measure from and the hand-over
records one attempt leaves the next, through the `base`, `route` and `summary` steps of
scripts/post_merge_check.py that carry them.

Deterministic: GitHub is run rows and job pages in memory, the fixture's commits are named in first-parent
order, and no test starts a process or reads the network.
"""
from __future__ import annotations

import json
import urllib.error

import pytest

from ouroboros.update_qualification import CONTINUOUS_STEP, MEASURED_STEP, RANGE_TITLE
from scripts import post_merge_check as pmc
from scripts import post_merge_window as window

# First-parent order: OLD, then K, then A, then B, then the head.
OLD, K, A, B, HEAD = "0" * 39 + "1", "1" * 40, "2" * 40, "3" * 40, "f" * 40
X, Y = "tests/t.py::test_x", "tests/t.py::test_y"
SIZED = "size tests/test_smoke.py::test_no_oversized_modules"
ABSENT = object()  # a leg without the step that carries its receipt
TESTS_STEP = "Run the affected tests (contract floor + affected modules and the complete Node lane)"


def _run(sha, number, conclusion="failure", *, attempt=1, event="push"):
    return {"id": number, "run_attempt": attempt, "head_sha": sha, "status": "completed", "conclusion": conclusion,
            "display_title": RANGE_TITLE, "head_branch": "ouroboros", "event": event}


def _record(ids, *, head, run, attempt=1, base=K):
    return {"head": head, "run": run, "attempt": attempt, "base": base, "ids": sorted(ids)}


def _step(name, conclusion="success"):
    return {"name": name, "status": "completed", "conclusion": conclusion}


def _job(run, name, steps, *, conclusion, attempt=None):
    return {"name": name, "run_id": run["id"], "run_attempt": attempt or run["run_attempt"], "head_sha": run["head_sha"],
            "status": "completed", "conclusion": conclusion, "steps": steps}


def _window(run, base, *, confirmed=True, carried=None, attempt=None):
    """The window job of `run`: the base it chose and, with `carried`, the record of what it carried in for each
    system (or the reference to the artifact that record travelled in)."""
    steps = [_step(f"Post-merge {'confirmed' if confirmed else 'provisional'} base {base}")]
    if carried is not None:
        record = carried if "artifact" in carried else {"history_read": True, "awaiting_handoff": carried, "history_gaps": []}
        steps.insert(0, _step(window.REPLAY_STEP + json.dumps(record)))
    return _job(run, "window", steps, conclusion="success", attempt=attempt)


def _leg(run, system, base, receipt=(), *, scope="chain", ran=False, attempt=None):
    """A leg of `run` that measured from `base`. `receipt` is what its hand-over step carries (`ABSENT`: no such
    step), `scope` the range marker it published ("chain", "push" or None), `ran` whether it reached its tests."""
    steps = [_step(TESTS_STEP)] if ran else []
    if receipt is not ABSENT:
        steps.append(_step(window.OWED_STEP + json.dumps(list(receipt) if isinstance(receipt, tuple) else receipt)))
    if scope:
        steps.append(_step(CONTINUOUS_STEP if scope == "chain" else MEASURED_STEP))
    return _job(run, f"post-merge ({system}) from {base}", steps, conclusion=run["conclusion"], attempt=attempt)


class Jobs:
    """`get` for the job pages of each (run, attempt), remembering which were asked for."""

    def __init__(self, pages):
        self.pages, self.asked = pages, []

    def __call__(self, path):
        number, attempt = (int(part) for part in path.split("/")[2:5:2])
        self.asked.append((number, attempt))
        jobs = self.pages[number, attempt]
        if isinstance(jobs, Exception):
            raise jobs
        return {"jobs": jobs, "total_count": len(jobs)}


def _unavailable():
    return urllib.error.HTTPError("https://api.github.invalid", 503, "Service Unavailable", None, None)


def _choose(runs, jobs, *, handoff=None, initial="", before=""):
    replay = {}
    base, why, continuous = pmc.choose_base(runs, before, HEAD, get=jobs, ancestor=lambda older, newer: older < newer,
                                            parent=lambda sha: pytest.fail("no fallback to the first parent"),
                                            initial=initial, replay=replay, handoff=handoff)
    return base, why, continuous, replay


# --- which commit the next window starts from ------------------------------------------------

@pytest.mark.parametrize("descendant", ["qualified", "cancelled"])
def test_only_a_qualified_descendant_closes_the_walk_behind_it(descendant):
    """A's first attempt qualified K..A, B started from A, and then a rerun of A (its second attempt) could not hand X
    over on Windows. Once B qualified, the next check starts at B and never reads A again: the rerun keeps its own
    red. While B only chose A and was cancelled, the next check reads the rerun as well: it keeps the rerun's base K
    and X with the attempt that saw it fail, and promotes neither B nor A."""
    late = _record([X], head=A, run=1, attempt=2)
    rerun = _run(A, 1, attempt=2)
    b = _run(B, 2, "success" if descendant == "qualified" else "cancelled")
    jobs = Jobs({
        (1, 2): [_window(rerun, K, carried={}), _leg(rerun, "linux", K), _leg(rerun, "windows", K, [late], scope="push")],
        (2, 1): [_window(b, A, carried={})] + ([_leg(b, "linux", A), _leg(b, "windows", A)]
                                               if descendant == "qualified" else []),
    })
    base, why, continuous, replay = _choose([b, rerun], jobs)
    if descendant == "qualified":
        assert (base, continuous) == (B, True) and why == "eligible on both systems (run 2, attempt 1, success)"
        assert jobs.asked == [(2, 1)] and replay["awaiting_handoff"] == {}
    else:
        assert (base, continuous) == (K, True) and why == "retained starting base of run 1, attempt 2"
        assert jobs.asked == [(2, 1), (1, 2)] and replay["awaiting_handoff"] == {"windows": [late]}
    assert replay["history_gaps"] == [] and replay["history_read"] is True


@pytest.mark.parametrize("rerun", ["handed over", "left no record", "could not take over"])
def test_one_legs_rerun_is_no_joint_measurement_and_only_its_whole_receipt_replaces_the_earlier_one(rerun):
    """The first attempt measured K..A on both systems and could not hand X over on Windows. Re-running only that leg
    lists it alone in the second attempt: no attempt holds both votes, so the head is never qualified and the base
    stays. What the rerun's own receipt says replaces the first attempt's only when it is whole."""
    first, again = _record([X], head=A, run=1), _record([Y], head=A, run=1, attempt=2)
    run = _run(A, 1, attempt=2)
    receipt = {"handed over": (), "left no record": ABSENT, "could not take over": {"partial": [again]}}[rerun]
    jobs = Jobs({
        (1, 2): [_leg(run, "windows", K, receipt, scope="chain" if rerun == "handed over" else None, ran=True)],
        (1, 1): [_window(run, K, carried={}, attempt=1), _leg(run, "linux", K, attempt=1),
                 _leg(run, "windows", K, [first], scope="push", attempt=1)],
    })
    base, why, continuous, replay = _choose([run], jobs)
    assert (base, continuous) == (K, True) and why == "retained starting base of run 1, attempt 2"
    assert jobs.asked == [(1, 2), (1, 1)]
    assert replay["awaiting_handoff"] == {"handed over": {}, "left no record": {"windows": [first]},
                                          "could not take over": {"windows": [again, first]}}[rerun]
    assert replay["history_gaps"] == ([{"run": 1, "attempt": 2, "part": "windows", "base": K}]
                                      if rerun == "left no record" else [])


def test_an_unqualified_window_keeps_its_base_behind_a_newer_qualified_head():
    """The owner's dispatch measured from K and did not qualify; an older run qualified A, which is after K. The next
    check starts where the unqualified window started, not at the qualified head it reaches behind it."""
    kept, qualified = _run(B, 2, event="workflow_dispatch"), _run(A, 1, "success")
    jobs = Jobs({(2, 1): [_window(kept, K, carried={})],
                 (1, 1): [_leg(qualified, "linux", OLD), _leg(qualified, "windows", OLD)]})
    base, why, continuous, _ = _choose([kept, qualified], jobs)
    assert (base, continuous) == (K, True) and why == "retained starting base of run 2, attempt 1"


def test_a_provisional_window_behind_a_confirmed_one_widens_nothing():
    """An explicit seed started a confirmed range at A; the provisional checks before it measured from K. A later
    check keeps the seed's base: what came before that guarantee is not measured in retrospect."""
    seeded, earlier = _run(B, 2, event="workflow_dispatch"), _run(A, 1)
    jobs = Jobs({(2, 1): [_window(seeded, A, carried={})],
                 (1, 1): [_window(earlier, K, confirmed=False, carried={}), _leg(earlier, "linux", K, scope="push"),
                          _leg(earlier, "windows", K, scope="push")]})
    base, why, continuous, _ = _choose([seeded, earlier], jobs)
    assert (base, continuous) == (A, True) and why == "retained starting base of run 2, attempt 1"


def test_the_walk_reads_no_run_that_ended_before_the_base_it_keeps():
    """A check reads history only as far back as the range it measures: a run whose head lies before the base an
    unqualified window keeps closed with that base, whatever it left."""
    kept, behind = _run(B, 2), _run(K, 1)
    jobs = Jobs({(2, 1): [_window(kept, A, carried={})]})  # asking for the older run's jobs would fail the lookup
    base, _, continuous, _ = _choose([kept, behind], jobs)
    assert (base, continuous) == (A, True) and jobs.asked == [(2, 1)]


# --- what the next window carries in ---------------------------------------------------------

@pytest.mark.parametrize("older", ["carried", "late rerun", "carry unread"])
def test_a_record_a_newer_window_already_carried_is_that_attempts_to_report(older):
    """P could not hand X over on Windows. Q carried that record in, settled it and stayed unqualified for a red of
    its own on Linux. The next window reads Q's receipts and does not take X from P again: Q's own receipt says it is
    settled. A rerun of P that Q never saw is read, and so is P's record when what Q carried in cannot be shown."""
    attempt = 2 if older == "late rerun" else 1
    x_p, y_q = _record([X], head=A, run=1), _record([Y], head=B, run=2)
    left = _record([X], head=A, run=1, attempt=attempt)
    p, q = _run(A, 1, attempt=attempt), _run(B, 2)
    reference = {"artifact": {"run": 2, "attempt": 1, "sha256": "0" * 64}}

    def handoff(*asked):
        assert asked == (2, 1, "window", "0" * 64)
        raise _unavailable()

    jobs = Jobs({
        (2, 1): [_window(q, K, carried=reference if older == "carry unread" else {"windows": [x_p]}),
                 _leg(q, "linux", K, [y_q], scope="push"), _leg(q, "windows", K)],
        (1, attempt): [_window(p, K, carried={}), _leg(p, "linux", K), _leg(p, "windows", K, [left], scope="push")],
    })
    base, _, continuous, replay = _choose([q, p], jobs, handoff=handoff)
    assert (base, continuous) == (K, True) and replay["history_gaps"] == []
    assert replay["awaiting_handoff"] == ({"linux": [y_q]} if older == "carried" else {"linux": [y_q], "windows": [left]})


@pytest.mark.parametrize("fault, gap, confirmed", [
    ("expired", True, True), ("left no record", True, True), ("cannot say", True, True), ("not a receipt", True, True),
    ("never ran", False, True), ("unavailable", False, False)])
def test_a_lost_receipt_is_a_labelled_gap_and_an_unreadable_one_confirms_nothing(fault, gap, confirmed):
    """An earlier attempt could not hand Y over on Windows; this run's window carried that record in, and its own
    Windows leg then left no receipt that can be read. The known record stays beside the gap and the base stays, so
    the next check measures K..head itself. A receipt that only could not be read now is no gap, and the window it
    seeds is no confirmed range."""
    known = _record([Y], head=A, run=1)
    run = _run(B, 2)
    reference = {"artifact": {"run": 2, "attempt": 1, "sha256": "0" * 64}}
    receipt = {"expired": reference, "unavailable": reference, "left no record": ABSENT, "never ran": ABSENT,
               "cannot say": None, "not a receipt": [{"head": A, "ids": []}]}[fault]

    def handoff(*asked):
        assert asked == (2, 1, "windows", "0" * 64)
        raise _unavailable() if fault == "unavailable" else FileNotFoundError("the artifact has expired")

    jobs = Jobs({(2, 1): [_window(run, K, carried={"windows": [known]}), _leg(run, "linux", K),
                          _leg(run, "windows", K, receipt, scope=None, ran=fault != "never ran")]})
    base, why, continuous, replay = _choose([run], jobs, handoff=handoff)
    assert base == K and continuous is confirmed and replay["history_read"] is confirmed
    assert why == "retained starting base of run 2, attempt 1" + ("" if confirmed else "; earlier history unreadable")
    assert replay["awaiting_handoff"] == {"windows": [known]}
    assert replay["history_gaps"] == ([{"run": 2, "attempt": 1, "part": "windows", "base": K}] if gap else [])


@pytest.mark.parametrize("bound", ["qualified ancestor", "explicit base", "none"])
@pytest.mark.parametrize("code", [410, 404])
def test_jobs_that_are_gone_are_a_gap_and_the_range_starts_at_an_established_bound_or_an_explicit_seed(bound, code):
    """The jobs of a red run answer 410 Gone: what it could not hand over is unknown for good, a labelled gap. A bare
    404 proves no loss — it may be transient — so it is no gap, but it withholds confirmation. Either way a qualified
    ancestor behind it is an established bound and an explicit first-parent base starts a new guarantee; without
    either the push is checked provisionally and confirms nothing."""
    lost, qualified = _run(B, 2), _run(A, 1, "success")
    jobs = Jobs({(2, 1): urllib.error.HTTPError("https://api.github.invalid", code,
                                                "Gone" if code == 410 else "Not Found", None, None),
                 (1, 1): [_leg(qualified, "linux", K), _leg(qualified, "windows", K)]})
    runs = [lost, qualified] if bound == "qualified ancestor" else [lost]
    base, why, continuous, replay = _choose(runs, jobs, initial=K if bound == "explicit base" else "", before=B)
    gone = code == 410
    assert replay["history_gaps"] == ([{"run": 2, "attempt": 1, "part": "jobs", "base": None}] if gone else [])
    assert replay["history_read"] == gone and replay["awaiting_handoff"] == {}
    assert (base, why, continuous) == {
        "qualified ancestor": (A, "eligible on both systems (run 1, attempt 1, success)"
                                 + ("" if gone else "; earlier history unreadable"), gone),
        "explicit base": (K, "explicit first-parent window" if gone else "explicit range; earlier history unreadable",
                          gone),
        "none": (B, "the branch tip before this push (earlier coverage remains pending)", False)}[bound]


def test_history_that_could_not_be_read_now_seeds_no_confirmed_range():
    """A row of the run list that cannot be read may hide an attempt that left a red owed, and an outage ends the
    walk early. What was established still chooses the base; the range stays provisional."""
    qualified, kept = _run(A, 1, "success"), _run(B, 2)
    jobs = Jobs({(1, 1): [_leg(qualified, "linux", K), _leg(qualified, "windows", K)],
                 (2, 1): [_window(kept, A, carried={})]})
    base, why, continuous, replay = _choose([{}, qualified], jobs)
    assert (base, continuous, replay["history_read"]) == (A, False, False)
    assert why == "eligible on both systems (run 1, attempt 1, success); earlier history unreadable"

    def outage():
        yield kept
        raise OSError("the run list went away")

    base, why, continuous, replay = _choose(outage(), jobs)
    assert (base, continuous, replay["history_read"]) == (A, False, False)
    assert why == "retained starting base of run 2, attempt 1; earlier history unreadable"


def test_a_receipt_is_hand_over_records_or_nothing():
    whole = {"head": A, "run": 7, "attempt": 2, "base": K, "ids": [Y, X, X]}
    same = {"head": A, "run": 7, "attempt": 2, "base": K, "ids": [X, Y]}
    bare = {"head": None, "run": None, "attempt": None, "base": None, "ids": [X]}
    assert window.records([whole, {"ids": [X]}]) == [same, bare] and window.records([]) == []
    for broken in (None, {"partial": []}, [X], [{**whole, "ids": []}], [{**whole, "ids": [""]}], [{**whole, "head": A[:12]}],
                   [{**whole, "run": "7"}], [{**whole, "run": True}], [{**whole, "attempt": 0}], [{**whole, "base": 5}]):
        assert window.records(broken) is None, broken
    # One record per source: the same attempt at another base is another measurement.
    other = {**same, "base": A, "ids": [X]}
    assert window.merge_records([bare], [{**same, "ids": [Y]}, other], [{**same, "ids": [X]}]) == [bare, same, other]
    assert window.carried_records({"awaiting_handoff": {"linux": [whole]}}, "linux") == [same]
    assert window.carried_records({}, "windows") == []
    for unreadable in ({"artifact": {}}, {"awaiting_handoff": []}, {"awaiting_handoff": {"linux": [X]}}, None,
                       window.LOST, window.UNREAD):
        assert window.carried_records(unreadable, "linux") is None, unreadable


# --- the steps that carry the records --------------------------------------------------------

def _passed(directory, nodeid):
    """A finished pytest session as tests/ci_evidence.py exports it, in which `nodeid` ran whole and passed."""
    directory.mkdir(parents=True)
    rows = [{"nodeid": nodeid, "phase": phase, "outcome": "passed", "is_subtest": False}
            for phase in ("setup", "call", "teardown")]
    (directory / "results.json").write_text(json.dumps({
        "format": 1, "session_exit_code": 0, "reports": rows, "tests_collected": 1, "collection_failures": []}),
        encoding="utf-8")


def _routed(seen, **record):
    """A stand-in for the hand-over that remembers what it was given and answers `record`."""
    def fake_route(github, **kwargs):
        seen.update(kwargs)
        return {"os": kwargs["system"], "comments": [], "created": [], "updated": [], "narrowed": [], "closed": [],
                "warnings": [], "awaiting_handoff": [], "untracked": [], **record}
    return fake_route


@pytest.mark.parametrize("mode", ["block", "approved"])
def test_route_takes_over_its_own_systems_records_with_their_source_and_reads_its_size_gate_strictly(
        tmp_path, monkeypatch, mode):
    """Each leg receives the records of its own system, never the other's, and names itself as the origin of what it
    will leave owed. Its own size gate settles a recorded size red only when it measured strictly: a pass under an
    approved growth is no pass against that record."""
    linux, windows = [_record([SIZED, X], head=A, run=5)], [_record([Y], head=A, run=5)]
    (tmp_path / "base.json").write_text(json.dumps({"base": K, "head": HEAD, "continuous": True, "replay": {
        "history_read": True, "awaiting_handoff": {"linux": linux, "windows": windows}, "history_gaps": []}}),
        encoding="utf-8")
    (tmp_path / "classification.json").write_text(json.dumps({
        "failed": [], "flaky": [], "size_checked": True, "size": {"base": K, "mode": mode}}), encoding="utf-8")
    _passed(tmp_path / "size", SIZED.removeprefix("size "))
    seen = {}
    monkeypatch.setattr(pmc, "route", _routed(seen))
    monkeypatch.setenv("GITHUB_RUN_ID", "77")
    for system, own in (("linux", linux), ("windows", windows)):
        assert pmc.main(["route", "--evidence-dir", str(tmp_path), "--run-url", "u", "--head", HEAD, "--os", system]) == 0
        assert seen["system"] == system and seen["prior_failures"] == own
        assert seen["origin"] == {"head": HEAD, "run": 77, "attempt": 1, "base": K}
        assert seen["classification"].get("size_passed") == ([SIZED] if (system, mode) == ("linux", "block") else None)
        assert json.loads((tmp_path / "routing.json").read_text(encoding="utf-8"))["untracked"] == []


@pytest.mark.parametrize("carry", ["read", "unavailable", "gone"])
def test_a_leg_that_could_not_take_over_the_carried_records_says_so_in_its_receipt(tmp_path, monkeypatch, capsys, carry):
    """The window's record travelled as an artifact. A leg that read it publishes the whole list it still owes and
    may qualify; one that could not publishes a partial list beside the earlier ones, is red for it and keeps the
    base."""
    old = _record([Y], head=A, run=5)
    (tmp_path / "base.json").write_text(json.dumps({"base": K, "head": HEAD, "continuous": True, "replay": {
        "artifact": {"run": 77, "attempt": 1, "sha256": "0" * 64}}}), encoding="utf-8")
    (tmp_path / "classification.json").write_text(json.dumps({
        "base": K, "head": HEAD, "selection_executed": True, "failed": [], "flaky": [], "problems": []}), encoding="utf-8")

    def handoff(self, *asked):
        assert asked == (77, 1, "window", "0" * 64)
        if carry == "read":
            return {"history_read": True, "awaiting_handoff": {"windows": [old]}, "history_gaps": []}
        raise _unavailable() if carry == "unavailable" else FileNotFoundError("never delivered")

    seen, output = {}, tmp_path / "outputs"
    monkeypatch.setattr(pmc.GitHub, "handoff_json", handoff)
    monkeypatch.setattr(pmc, "route", _routed(seen))
    monkeypatch.setattr(pmc, "merges", lambda *_: [])
    monkeypatch.setenv("GITHUB_OUTPUT", str(output))
    monkeypatch.delenv("GITHUB_STEP_SUMMARY", raising=False)
    for name, value in (("GITHUB_RUN_ID", "77"), ("GITHUB_RUN_ATTEMPT", "1")):
        monkeypatch.setenv(name, value)
    where = ["--evidence-dir", str(tmp_path), "--os", "windows"]
    assert pmc.main(["route", *where, "--run-url", "u", "--head", HEAD]) == 0
    took_over = carry == "read"
    assert seen["prior_failures"] == ([old] if took_over else [])
    assert json.loads((tmp_path / "routing.json").read_text(encoding="utf-8"))["untracked"] == ([] if took_over else None)
    assert pmc.main(["summary", *where]) == int(not took_over)
    emitted = dict(line.split("=", 1) for line in output.read_text(encoding="utf-8").splitlines())
    assert json.loads(emitted["untracked"]) == ([] if took_over else {"partial": []})
    assert emitted["window_scope"] == ("chain" if took_over else "push")
    assert ("earlier untracked failures could not be read" in capsys.readouterr().out) is not took_over


@pytest.mark.parametrize("state", ["first attempt", "rerun", "nothing classified"])
def test_without_the_route_receipt_a_leg_still_names_what_it_knows_it_owes(tmp_path, monkeypatch, state):
    """The route step left no record. Every FAILED of this run is then owed under this run's name beside what the
    window carried in. A rerun never reconciled its earlier attempts, so its list is partial; a leg without any
    classification cannot say what failed."""
    old = _record([Y], head=A, run=5)
    attempt = 2 if state == "rerun" else 1
    own = _record([X], head=HEAD, run=77, attempt=attempt)
    (tmp_path / "base.json").write_text(json.dumps({"base": K, "head": HEAD, "continuous": True, "replay": {
        "history_read": True, "awaiting_handoff": {"windows": [old]}, "history_gaps": []}}), encoding="utf-8")
    if state != "nothing classified":
        (tmp_path / "classification.json").write_text(json.dumps({
            "base": K, "head": HEAD, "selection_executed": True, "flaky": [], "problems": [],
            "failed": [{"id": X, "why": "failed in call", "pass": "parallel", "retry": "failed"}]}), encoding="utf-8")
    output = tmp_path / "outputs"
    monkeypatch.setattr(pmc, "merges", lambda *_: [])
    monkeypatch.setenv("GITHUB_OUTPUT", str(output))
    monkeypatch.delenv("GITHUB_STEP_SUMMARY", raising=False)
    for name, value in (("GITHUB_RUN_ID", "77"), ("GITHUB_RUN_ATTEMPT", str(attempt))):
        monkeypatch.setenv(name, value)
    assert pmc.main(["summary", "--evidence-dir", str(tmp_path), "--os", "windows", "--attempt", str(attempt)]) == 1
    emitted = dict(line.split("=", 1) for line in output.read_text(encoding="utf-8").splitlines())
    assert json.loads(emitted["untracked"]) == {"first attempt": [old, own], "rerun": {"partial": [old, own]},
                                                "nothing classified": None}[state]
    assert emitted["window_scope"] == ("" if state == "nothing classified" else "push")


@pytest.mark.parametrize("code", [410, 404])
def test_a_gap_is_named_in_the_window_log_and_in_the_summary_of_the_system_it_concerns(tmp_path, monkeypatch, capsys, code):
    lost = _run(B, 2)

    def get(self, path):
        if path.startswith("actions/workflows/"):
            return {"total_count": 1, "workflow_runs": [lost]}
        raise urllib.error.HTTPError(path, code, "Gone" if code == 410 else "Not Found", None, None)

    monkeypatch.setattr(pmc.GitHub, "get", get)
    monkeypatch.setattr(pmc, "is_ancestor", lambda older, newer: older < newer)
    args = type("Args", (), {"head": None, "evidence_dir": tmp_path})
    assert pmc.command_base(args, {"GITHUB_SHA": HEAD, "BEFORE": B}) == 0
    record = json.loads((tmp_path / "base.json").read_text(encoding="utf-8"))
    assert (record["base"], record["continuous"]) == (B, False)
    gone = code == 410  # only a proven loss is a gap; a bare 404 leaves the history unread, not gone
    assert record["replay"] == {"history_read": gone, "awaiting_handoff": {},
                                "history_gaps": ([{"run": 2, "attempt": 1, "part": "jobs", "base": None}]
                                                 if gone else [])}
    logged = capsys.readouterr().out
    assert ("::warning title=Post-merge history gap::The job list of run 2, attempt 1 is gone" in logged) == gone
    # A leg names the loss that concerns its system; the gap alone makes no leg red.
    record["replay"]["history_gaps"] = [{"run": 7, "attempt": 1, "part": "windows", "base": K}]
    (tmp_path / "base.json").write_text(json.dumps({"base": K, "head": HEAD, "why": "retained", "replay": record["replay"]}),
                                        encoding="utf-8")
    (tmp_path / "classification.json").write_text(json.dumps({
        "failed": [], "flaky": [], "problems": [],
        "selection": {"tier": "selected", "file_count": 3, "available": 40, "node": "success"}}), encoding="utf-8")
    text, red = pmc.render_summary(tmp_path, system="windows")
    assert not red and "- **History gap:** The hand-over record on Windows of run 7, attempt 1 is gone" in text
    assert "History gap" not in pmc.render_summary(tmp_path, system="linux")[0]


@pytest.mark.parametrize("code", [410, 404])
def test_an_earlier_attempts_job_listing_distinguishes_gone_from_unread(code):
    """A rerun reads its first attempt's jobs: a 410 there is a labelled gap, a bare 404 proves no loss and only
    leaves what that attempt recorded unread, so nothing it left is settled."""
    run = _run(A, 1, attempt=2)
    jobs = Jobs({(1, 2): [_leg(run, "windows", K, (), scope="chain")],
                 (1, 1): urllib.error.HTTPError("https://api.github.invalid", code,
                                                "Gone" if code == 410 else "Not Found", None, None)})
    prior = window.prior_window(run, jobs)
    assert prior.gaps == ([{"run": 1, "attempt": 1, "part": "jobs", "base": None}] if code == 410 else [])
    assert prior.unread == (code == 404)


def test_the_route_step_carries_the_gaps_it_read_into_the_leg_record_and_the_summary_shows_them(tmp_path, monkeypatch):
    """The window's replay record travelled as an artifact, so base.json names only its reference and the per-OS
    summary cannot read gaps from there. The route step already read the artifact (and the earlier attempt of a
    rerun) and leaves what it found as `history_gaps` in its own record; the summary then names its system's gaps
    with no new read."""
    old = _record([Y], head=A, run=5)
    carried = {"history_read": True, "awaiting_handoff": {"windows": [old]},
               "history_gaps": [{"run": 5, "attempt": 1, "part": "window", "base": K}]}
    (tmp_path / "base.json").write_text(json.dumps({"base": K, "head": HEAD, "continuous": True, "replay": {
        "artifact": {"run": 77, "attempt": 1, "sha256": "0" * 64}}}), encoding="utf-8")
    (tmp_path / "classification.json").write_text(json.dumps({
        "base": K, "head": HEAD, "selection_executed": True, "failed": [], "flaky": [], "problems": []}),
        encoding="utf-8")

    def handoff(self, *asked):
        assert asked == (77, 1, "window", "0" * 64)
        return carried

    run = _run(HEAD, number=77, attempt=2)
    earlier_gap = {"run": 77, "attempt": 1, "part": "windows", "base": K}

    def get(self, path):
        if path == "git/ref/heads/ouroboros":
            return {"object": {"sha": HEAD}}
        number, attempt = (int(part) for part in path.split("/")[2:5:2])
        assert (number, attempt) == (77, 1)
        # The earlier attempt's Windows leg ran its tests and left no readable hand-over receipt.
        return {"jobs": [_leg(run, "windows", K, ABSENT, ran=True, attempt=1)], "total_count": 1}

    monkeypatch.setattr(pmc.GitHub, "handoff_json", handoff)
    monkeypatch.setattr(pmc.GitHub, "get", get)
    monkeypatch.setattr(pmc, "route", _routed({}))
    monkeypatch.delenv("GITHUB_STEP_SUMMARY", raising=False)
    for name, value in (("GITHUB_RUN_ID", "77"), ("GITHUB_RUN_ATTEMPT", "2")):
        monkeypatch.setenv(name, value)
    assert pmc.main(["route", "--evidence-dir", str(tmp_path), "--os", "windows", "--run-url", "u", "--head", HEAD,
                     "--attempt", "2"]) == 0
    record = json.loads((tmp_path / "routing.json").read_text(encoding="utf-8"))
    assert record["history_gaps"] == carried["history_gaps"] + [earlier_gap]
    text, red = pmc.render_summary(tmp_path, system="windows")
    assert not red
    assert "- **History gap:** The record of what the window carried in of run 5, attempt 1 is gone" in text
    assert "- **History gap:** The hand-over record on Windows of run 77, attempt 1 is gone" in text
    assert "History gap" not in pmc.render_summary(tmp_path, system="linux")[0]
