"""The post-merge window (scripts/post_merge_window.py): the base both legs measure from and the hand-over
records one attempt leaves the next, through the `base`, `route` and `summary` steps of
scripts/post_merge_check.py that carry them.

Deterministic: GitHub is run rows and job pages in memory, the fixture's commits are named in first-parent
order, and no test reads the network. Only the ancestry tests start a process: they ask the real Git about a
throwaway local repository.
"""
from __future__ import annotations

import json
import subprocess
import urllib.error

import pytest

from ouroboros.update_qualification import (CONTINUOUS_STEP, MEASURED_STEP, RANGE_TITLE, attempt_job_pages,
                                            measured_window)
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


def _stopped(run, carried, *, attempt=None):
    """A window job cancelled between its two steps: it published what it carries in and named no base."""
    job = _window(run, K, carried=carried, attempt=attempt)
    return {**job, "conclusion": "cancelled", "steps": job["steps"][:1]}


def _leg(run, system, base, receipt=(), *, scope="chain", ran=False, attempt=None):
    """A leg of `run` that measured from `base`. `receipt` is what its hand-over step carries (`ABSENT`: no such
    step), `scope` the range marker it published ("chain", "push" or None), `ran` whether it reached its tests."""
    steps = [_step(TESTS_STEP)] if ran else []
    if receipt is not ABSENT:
        steps.append(_step(window.OWED_STEP + json.dumps(list(receipt) if isinstance(receipt, tuple) else receipt)))
    if scope:
        steps.append(_step(CONTINUOUS_STEP if scope == "chain" else MEASURED_STEP))
    return _job(run, f"post-merge ({system}) from {base}", steps, conclusion=run["conclusion"], attempt=attempt)


class Short(list):
    """The job rows of a listing that announces one more than it shows: the page is not whole."""


class Jobs:
    """`get` for the job pages of each (run, attempt), remembering which were asked for."""

    def __init__(self, pages, siblings=()):
        self.pages, self.asked, self.siblings, self.listed = pages, [], siblings, []

    def __call__(self, path):
        if path.startswith("actions/workflows/"):  # the exact list of one commit's checks: `siblings`
            self.listed.append(path)
            if isinstance(self.siblings, Exception):
                raise self.siblings
            rows = list(self.siblings)
            return {"workflow_runs": rows, "total_count": len(rows) + (1 if isinstance(self.siblings, Short) else 0)}
        number, attempt = (int(part) for part in path.split("/")[2:5:2])
        self.asked.append((number, attempt))
        jobs = self.pages[number, attempt]
        if isinstance(jobs, Exception):
            raise jobs
        if isinstance(jobs, dict):  # the answer itself, as a broken endpoint gives it
            return jobs
        return {"jobs": jobs, "total_count": len(jobs) + (1 if isinstance(jobs, Short) else 0)}


def _unavailable():
    return urllib.error.HTTPError("https://api.github.invalid", 503, "Service Unavailable", None, None)


def _choose(runs, jobs, *, handoff=None, initial="", before=""):
    replay = {}
    base, why, continuous = pmc.choose_base(runs, before, HEAD, get=jobs, ancestor=lambda older, newer: older < newer,
                                            parent=lambda sha: pytest.fail("no fallback to the first parent"),
                                            initial=initial, replay=replay, handoff=handoff)
    return base, why, continuous, replay


def _walk(runs, jobs, *, handoff=None):
    """`choose_window` itself, for a case that needs none of the `base` step's fallbacks around it."""
    replay = {}
    base, why, continuous = window.choose_window(runs, HEAD, get=jobs, ancestor=lambda older, newer: older < newer,
                                                 replay=replay, handoff=handoff)
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
    assert (base, continuous) == (K, True) and why == "retained starting base of run 1, attempt 1"  # its window job's
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


@pytest.mark.parametrize("windows", ["took over", "never ran", "could not take over"])
def test_a_whole_receipt_takes_over_only_what_a_window_of_its_attempt_or_an_earlier_one_carried_in(windows):
    """P could not hand X over on Windows. Q's first window did not carry that record in, and both its legs left whole
    receipts. Then every job of Q was re-run and the second window carried X in. A whole Windows receipt of that
    second attempt took X over: Q answers for it and P's record is not read again. A Windows leg that never ran there,
    or could only add to the earlier lists, leaves the first attempt's receipt as the newest whole one. It was written
    before X was carried in and settles nothing about it: X stays owed under its own source."""
    x_p, y_q = _record([X], head=A, run=1), _record([Y], head=B, run=2, attempt=2)
    p, q = _run(A, 1), _run(B, 2, attempt=2)
    receipt = {"took over": (), "never ran": ABSENT, "could not take over": {"partial": [y_q]}}[windows]
    jobs = Jobs({
        (2, 2): [_window(q, K, carried={"windows": [x_p]}), _leg(q, "linux", K, [y_q], scope="push"),
                 _leg(q, "windows", K, receipt, scope="push" if windows == "took over" else None,
                      ran=windows != "never ran")],
        (2, 1): [_window(q, K, confirmed=False, carried={}, attempt=1), _leg(q, "linux", K, scope="push", attempt=1),
                 _leg(q, "windows", K, scope="push", attempt=1)],
        (1, 1): [_window(p, K, carried={}), _leg(p, "linux", K), _leg(p, "windows", K, [x_p], scope="push")],
    })
    base, why, continuous, replay = _choose([q, p], jobs)
    assert (base, why, continuous) == (K, "retained starting base of run 2, attempt 2", True)
    assert replay == {"history_read": True, "history_gaps": [], "awaiting_handoff": {
        "took over": {"linux": [y_q]}, "never ran": {"linux": [y_q], "windows": [x_p]},
        "could not take over": {"linux": [y_q], "windows": [y_q, x_p]}}[windows]}
    assert jobs.asked == ([(2, 2), (1, 1)] if windows == "took over" else [(2, 2), (2, 1), (1, 1)])


@pytest.mark.parametrize("second", ["cancelled before its base", "re-run to the end"])
def test_a_window_job_cancelled_before_its_base_step_still_names_what_it_carried_in(second):
    """P could not hand X over on Windows, and that receipt has since expired. Q's first attempt kept K and both its
    legs left whole receipts. Every job of Q was then re-run: the second window job published what it carries in, X
    with it, and was cancelled before the step that names its base, so no leg ran. The base is still the first
    attempt's and the record of X is the second's: the first attempt's whole Windows receipt was written before X was
    carried in and settles nothing about it, and the first window's own record is never asked for. Once a later
    attempt runs to its end, its whole Windows receipt stands for what its own window carried in: X is settled."""
    x_p = _record([X], head=A, run=1)
    ended = second == "re-run to the end"
    y_q = _record([Y], head=B, run=2, attempt=3 if ended else 1)
    p, q = _run(A, 1), _run(B, 2, "failure" if ended else "cancelled", attempt=3 if ended else 2)
    stopped = {**_window(q, K, carried={"windows": [x_p]}), "conclusion": "cancelled"}
    stopped["steps"] = stopped["steps"][:1]  # the replay step succeeded; the base step never ran
    expired = {"artifact": {"run": 1, "attempt": 1, "sha256": "1" * 64}}
    asked = []

    def handoff(*reference):
        asked.append(reference)
        raise FileNotFoundError("expired") if reference[0] == 1 else _unavailable()

    jobs = Jobs({
        (2, 3): [_window(q, K, carried={"windows": [x_p]}), _leg(q, "linux", K, [y_q], scope="push"),
                 _leg(q, "windows", K, scope="push")],
        (2, 2): [stopped],
        (2, 1): [_window(q, K, carried={"artifact": {"run": 2, "attempt": 1, "sha256": "0" * 64}}, attempt=1),
                 _leg(q, "linux", K, [y_q], scope="push", attempt=1), _leg(q, "windows", K, scope="push", attempt=1)],
        (1, 1): [_window(p, K, carried={}), _leg(p, "linux", K),
                 _leg(p, "windows", K, expired, scope="push", ran=True)],
    })
    base, why, continuous, replay = _choose([q, p], jobs, handoff=handoff)
    assert (base, why, continuous) == (K, f"retained starting base of run 2, attempt {3 if ended else 1}", True)
    assert replay == {"history_read": True, "history_gaps": [{"run": 1, "attempt": 1, "part": "windows", "base": K}],
                      "awaiting_handoff": {"linux": [y_q]} if ended else {"linux": [y_q], "windows": [x_p]}}
    assert jobs.asked == ([(2, 3), (1, 1)] if ended else [(2, 2), (2, 1), (1, 1)])
    assert asked == [(1, 1, "windows", "1" * 64)]


@pytest.mark.parametrize("first", ["cancelled before its base", "ran to its end"])
def test_every_window_job_read_keeps_what_it_carried_in_until_a_whole_receipt_takes_it_over(first):
    """P could not hand X over on Windows, and that receipt has since expired: only a window that read it in time
    still names X. Q's first window job did. Every job of Q was then re-run, and the second window job, cancelled
    before its base and its legs, carried in what it still found: nothing, or Z alone. A newer window that carried in
    less retires nothing. Where the first attempt was cancelled the same way, no leg ever took X over and X stays
    owed. Where the first attempt ran to its end, its whole Windows receipt took X over, and only Z, which no receipt
    at or after its window answers for, stays owed."""
    x_p, z = _record([X], head=A, run=1), _record([Y], head=A, run=4)
    ended = first == "ran to its end"
    p, q = _run(A, 1), _run(B, 2, "cancelled", attempt=2)
    expired = {"artifact": {"run": 1, "attempt": 1, "sha256": "1" * 64}}

    def handoff(*reference):
        raise FileNotFoundError("expired")

    jobs = Jobs({
        (2, 2): [_stopped(q, {"windows": [z]} if ended else {})],
        (2, 1): ([_window(q, K, carried={"windows": [x_p]}, attempt=1), _leg(q, "linux", K, scope="push", attempt=1),
                  _leg(q, "windows", K, scope="push", attempt=1)] if ended
                 else [_stopped(q, {"windows": [x_p]}, attempt=1)]),
        (1, 1): [_window(p, K, carried={}), _leg(p, "linux", K),
                 _leg(p, "windows", K, expired, scope="push", ran=True)],
    })
    base, why, continuous, replay = _choose([q, p], jobs, handoff=handoff)
    assert (base, why, continuous) == (K, f"retained starting base of run {2 if ended else 1}, attempt 1", True)
    assert replay == {"history_read": True, "history_gaps": [{"run": 1, "attempt": 1, "part": "windows", "base": K}],
                      "awaiting_handoff": {"windows": [z if ended else x_p]}}
    assert jobs.asked == [(2, 2), (2, 1), (1, 1)]


@pytest.mark.parametrize("newest", ["unavailable", "gone"])
@pytest.mark.parametrize("windows", ["could not take over", "took over"])
def test_a_replay_record_that_can_be_read_stands_beside_a_newer_one_that_cannot_and_lends_it_nothing(newest, windows):
    """Every job of A was re-run, and the run was cancelled after its second window job, whose record of what it
    carried in travelled as an artifact that cannot be read now or is gone. The first window's record can be read and
    names Y. It still stands for a Windows leg that could not take it over, and a whole Windows receipt of that first
    attempt still settles it. Either way the newer record stays what it is, unread or a named gap: the older one
    lends it no completeness and suppresses no source."""
    known, own = _record([Y], head=K, run=3, base=OLD), _record([X], head=A, run=5)
    run = _run(A, 5, "cancelled", attempt=2)
    reference = {"artifact": {"run": 5, "attempt": 2, "sha256": "0" * 64}}
    asked = []

    def handoff(*part):
        asked.append(part)
        raise _unavailable() if newest == "unavailable" else FileNotFoundError("expired")

    took_over = windows == "took over"
    jobs = Jobs({(5, 2): [_window(run, K, carried=reference)],
                 (5, 1): [_window(run, K, carried={"windows": [known]}, attempt=1), _leg(run, "linux", K, attempt=1),
                          _leg(run, "windows", K, [own] if took_over else {"partial": [own]}, scope="push",
                               attempt=1)]})
    prior = window.prior_window(run, jobs, handoff)
    assert prior.chosen == (K, True, 2) and prior.carried is None and prior.searched is False
    assert prior.awaiting == {"windows": [own] if took_over else [own, known]}
    assert prior.unread is (newest == "unavailable")
    assert prior.gaps == ([] if newest == "unavailable" else [{"run": 5, "attempt": 2, "part": "window", "base": K}])
    assert asked == [(5, 2, "window", "0" * 64)] and jobs.asked == [(5, 2), (5, 1)]


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


@pytest.mark.parametrize("descendant", ["provisional checkpoint", "qualified"])
def test_a_provisional_checkpoint_does_not_close_a_seeded_walk_but_a_qualified_descendant_does(descendant):
    """A dispatch seeds K. The newest run B (the checkpoint) chose A and owes nothing, yet a rerun of A, newer than that
    read, could not hand X over on Windows. The seeded window reads each run between the seed and the checkpoint, so the
    rerun's red is carried under its own source; a qualified B closes the walk behind it instead. Nothing behind the
    base the rerun retains is read in either case."""
    late = _record([X], head=A, run=1, attempt=2)
    rerun, behind = _run(A, 1, attempt=2), _run(OLD, 3)
    qualified = descendant == "qualified"
    newest, scope = _run(B, 2, "success" if qualified else "failure"), "chain" if qualified else "push"
    jobs = Jobs({
        (1, 2): [_window(rerun, K, carried={}), _leg(rerun, "linux", K), _leg(rerun, "windows", K, [late], scope="push")],
        (2, 1): [_window(newest, A, confirmed=qualified, carried={}), _leg(newest, "linux", A, scope=scope),
                 _leg(newest, "windows", A, scope=scope)],
    })
    base, why, continuous, replay = _choose([newest, rerun, behind], jobs, initial=K)
    if qualified:
        assert (base, why, continuous) == (K, "widened first-parent window", True)
        assert jobs.asked == [(2, 1)] and replay["awaiting_handoff"] == {}
    else:
        assert (base, why, continuous) == (K, "retained starting base of run 1, attempt 2", True)
        assert jobs.asked == [(2, 1), (1, 2)] and replay["awaiting_handoff"] == {"windows": [late]}
    assert replay["history_gaps"] == [] and replay["history_read"] is True


@pytest.mark.parametrize("confirmed", [True, False], ids=["hold first", "checkpoint first"])
def test_a_later_explicit_seed_does_not_drop_an_attempt_inside_a_known_held_range(confirmed):
    """The seed B lies inside a range that K already holds, and the attempt that owes X ended before B. The seed is no
    evidence that anything before it is closed: the walk still reaches the hold (from a provisional checkpoint, too)
    and carries X under its own source."""
    late = _record([X], head=A, run=1)
    newer, owing = _run(B, 2), _run(A, 1)
    jobs = Jobs({
        (2, 1): [_window(newer, K, confirmed=confirmed, carried={}), _leg(newer, "linux", K, scope="push"),
                 _leg(newer, "windows", K, scope="push")],
        (1, 1): [_window(owing, K, carried={}), _leg(owing, "linux", K), _leg(owing, "windows", K, [late], scope="push")],
    })
    base, why, continuous, replay = _choose([newer, owing], jobs, initial=B)
    assert (base, why, continuous) == (K, f"retained starting base of run {2 if confirmed else 1}, attempt 1", True)
    assert jobs.asked == [(2, 1), (1, 1)] and replay["awaiting_handoff"] == {"windows": [late]}


@pytest.mark.parametrize("descendant", ["held range", "qualified"])
def test_a_checkpoint_inside_a_held_range_hides_no_older_rerun_but_a_qualified_descendant_closes_the_walk(descendant):
    """Unseeded. C's run kept K as its confirmed base without qualifying; B's run is a provisional checkpoint that already
    searched its history; a rerun of A, inside K's range, could not hand X over. With a base held, the checkpoint closes
    nothing, so the rerun's red is carried under its own source. A truly qualified C closes the walk behind it instead."""
    mid = "4" * 40
    late = _record([X], head=A, run=1, attempt=2)
    rerun, checkpoint = _run(A, 1, attempt=2), _run(B, 3)
    qualified = descendant == "qualified"
    newest = _run(mid, 4, "success" if qualified else "failure")
    jobs = Jobs({
        (4, 1): ([_window(newest, A, carried={}), _leg(newest, "linux", A), _leg(newest, "windows", A)] if qualified
                 else [_window(newest, K, carried={}), _leg(newest, "linux", K, scope="push"),
                       _leg(newest, "windows", K, scope="push")]),
        (3, 1): [_window(checkpoint, A, confirmed=False, carried={}), _leg(checkpoint, "linux", A, scope="push"),
                 _leg(checkpoint, "windows", A, scope="push")],
        (1, 2): [_window(rerun, K, carried={}), _leg(rerun, "linux", K), _leg(rerun, "windows", K, [late], scope="push")],
    })
    base, why, continuous, replay = _choose([newest, checkpoint, rerun], jobs)
    if qualified:
        assert (base, why, continuous) == (mid, "eligible on both systems (run 4, attempt 1, success)", True)
        assert jobs.asked == [(4, 1)] and replay["awaiting_handoff"] == {}
    else:
        assert (base, why, continuous) == (K, "retained starting base of run 4, attempt 1", True)
        assert jobs.asked == [(4, 1), (3, 1), (1, 2)] and replay["awaiting_handoff"] == {"windows": [late]}
    assert replay["history_gaps"] == [] and replay["history_read"] is True


@pytest.mark.parametrize("listing", ["whole", "short"])
@pytest.mark.parametrize("kind", ["qualified run", "owing run"])
def test_a_short_run_listing_keeps_the_runs_it_shows_but_confirms_no_range(tmp_path, monkeypatch, kind, listing):
    """`command_base` reads the run list page by page. A page that is not whole (101 runs announced, one listed) may lack
    a newer attempt that owes a red, so even a qualified run on it, which closes the walk, confirms no range: the base
    and the debts of the runs shown stand and the history is marked unread. The same run on a whole listing confirms."""
    late = _record([X], head=A, run=1)
    if kind == "qualified run":
        run = _run(A, 1, "success")
        jobs = Jobs({(1, 1): [_leg(run, "linux", K), _leg(run, "windows", K)]})
        chosen, owed = (A, "eligible on both systems (run 1, attempt 1, success)"), {}
    else:
        run = _run(A, 1)
        jobs = Jobs({(1, 1): [_window(run, K, carried={}), _leg(run, "linux", K),
                              _leg(run, "windows", K, [late], scope="push")]})
        chosen, owed = (K, "retained starting base of run 1, attempt 1"), {"windows": [late]}
    whole = listing == "whole"

    def get(self, path):
        if path.startswith("actions/workflows/"):
            return {"total_count": 1 if whole else 101, "workflow_runs": [run]}
        return jobs(path)

    monkeypatch.setattr(pmc.GitHub, "get", get)
    monkeypatch.setattr(pmc, "is_ancestor", lambda older, newer: older < newer)
    args = type("Args", (), {"head": None, "evidence_dir": tmp_path})
    assert pmc.command_base(args, {"GITHUB_SHA": HEAD, "BEFORE": B}) == 0
    record = json.loads((tmp_path / "base.json").read_text(encoding="utf-8"))
    assert (record["base"], record["why"], record["continuous"]) == (
        chosen[0], chosen[1] + ("" if whole else "; earlier history unreadable"), whole)
    assert record["replay"]["history_read"] is whole and record["replay"]["awaiting_handoff"] == owed


@pytest.mark.parametrize("listing", ["whole", "short"])
def test_a_job_listing_that_is_not_whole_promotes_nothing_but_keeps_the_receipts_it_shows(listing):
    """Both legs and the window job of A's only attempt are listed, and a whole listing qualifies A. The same rows on a
    page that announces one job more are no joint witness (`measured_window` stays strict): the window's base is kept and
    stays provisional, A is not promoted, and no request beyond that page is made."""
    run = _run(A, 1, "success")
    rows = [_window(run, K, carried={}), _leg(run, "linux", K), _leg(run, "windows", K)]
    jobs = Jobs({(1, 1): rows if listing == "whole" else Short(rows)})
    base, why, continuous, replay = _choose([run], jobs)
    if listing == "whole":
        assert (base, why, continuous) == (A, "eligible on both systems (run 1, attempt 1, success)", True)
    else:
        assert (base, why, continuous) == (
            K, "retained starting base of run 1, attempt 1; earlier history unreadable", False)
    assert replay["history_read"] is (listing == "whole") and replay["history_gaps"] == [] and jobs.asked == [(1, 1)]


@pytest.mark.parametrize("missing", ["the linux leg", "the window job"])
def test_the_receipt_a_short_listing_shows_stays_owed_and_the_missing_job_confirms_nothing(missing):
    """The Windows leg of A's attempt is listed and still owes X, but the page announces one job more than it lists: the
    other leg's vote or the window's base is missing. The record stays owed under its own source, A is not promoted and
    the history stays provisional; without the window job there is no base to keep, so the push alone is checked."""
    late = _record([X], head=A, run=1)
    run = _run(A, 1)
    windows = _leg(run, "windows", K, [late], scope="push")
    shown = [_window(run, K, carried={}), windows] if missing == "the linux leg" else [_leg(run, "linux", K), windows]
    jobs = Jobs({(1, 1): Short(shown)})
    base, why, continuous, replay = _choose([run], jobs, before=B)
    assert (base, why, continuous) == (
        (K, "retained starting base of run 1, attempt 1; earlier history unreadable", False) if missing == "the linux leg"
        else (B, "the branch tip before this push (earlier coverage remains pending)", False))
    assert replay == {"history_read": False, "awaiting_handoff": {"windows": [late]}, "history_gaps": []}
    assert jobs.asked == [(1, 1)]


def test_receipts_of_earlier_pages_stand_when_a_later_page_of_the_listing_changes():
    """A hundred jobs fill the first page, which carries the window job and the Windows record of X. The second page lists
    the Linux leg, but the total moved while the listing was read: the page is not whole. What both pages show stands, the
    head is not qualified, the history stays provisional, and each page is requested once for both readers."""
    late = _record([X], head=A, run=1)
    run = _run(A, 1)
    pages = {1: [_window(run, K, carried={}), _leg(run, "windows", K, [late], scope="push"),
                 *[_job(run, f"setup {n}", [], conclusion="success") for n in range(98)]],
             2: [_leg(run, "linux", K)]}
    asked = []

    def get(path):
        asked.append(int(path.rsplit("page=", 1)[1]))
        return {"jobs": pages[asked[-1]], "total_count": 102}  # 101 jobs on two pages, 102 announced

    base, why, continuous, replay = _choose([run], get)
    assert (base, why, continuous) == (K, "retained starting base of run 1, attempt 1; earlier history unreadable", False)
    assert replay["awaiting_handoff"] == {"windows": [late]} and replay["history_read"] is False and asked == [1, 2]


def test_the_strict_readers_refuse_a_job_listing_that_is_not_whole_even_when_both_legs_show():
    """Only `prior_window` reads a short page's rows. `measured_window`, which qualifies a head, takes one whole listing of
    one attempt however many legs the page shows, and `attempt_job_pages` raises before the page unless `partial` asks for
    its rows first."""
    run = _run(A, 1, "success")
    rows = Short([_leg(run, "linux", K), _leg(run, "windows", K)])
    jobs = Jobs({(1, 1): rows})
    for strict in (lambda: measured_window(run, jobs), lambda: list(attempt_job_pages(run, jobs))):
        with pytest.raises(ValueError, match="incomplete jobs listing"):
            strict()
    shown = []
    with pytest.raises(ValueError, match="incomplete jobs listing"):
        for page in attempt_job_pages(run, jobs, partial=True):
            shown += page
    assert [job["name"] for job in shown] == [job["name"] for job in rows]


def test_a_rerun_takes_over_what_a_short_listing_of_its_first_attempt_shows_but_cannot_call_its_list_whole(
        tmp_path, monkeypatch):
    """The second attempt reads the first one's jobs, and that page announces one more than it lists. The Windows record it
    shows is taken over, so it is filed again, but the leg cannot say it took over everything: its receipt adds to the
    earlier ones instead of replacing them."""
    old = _record([Y], head=HEAD, run=77)
    first = _run(HEAD, 77)
    shown = [_leg(first, "windows", K, [old], scope="push")]

    def get(self, path):
        if path == "git/ref/heads/ouroboros":
            return {"object": {"sha": HEAD}}
        assert path == "actions/runs/77/attempts/1/jobs?per_page=100&page=1"
        return {"jobs": shown, "total_count": len(shown) + 1}

    seen = {}
    monkeypatch.setattr(pmc.GitHub, "get", get)
    monkeypatch.setattr(pmc, "route", _routed(seen))
    monkeypatch.setenv("GITHUB_RUN_ID", "77")
    assert pmc.main(["route", "--evidence-dir", str(tmp_path), "--os", "windows", "--run-url", "u", "--head", HEAD,
                     "--attempt", "2"]) == 0
    assert seen["prior_failures"] == [old]
    assert json.loads((tmp_path / "routing.json").read_text(encoding="utf-8"))["untracked"] is None


@pytest.mark.parametrize("code", [410, 404, 503])
def test_a_third_attempt_takes_over_the_first_ones_record_behind_a_middle_attempt_that_is_gone_or_unread(
        tmp_path, monkeypatch, code):
    """The Windows leg of a third attempt reads the attempts before it. The job list of the second is gone for good
    (410) or cannot be read now (a bare 404, an outage); the first survives and still owes Y. The leg takes Y over
    either way. Only the 410 is a named gap beside a whole list: an attempt that could not be read leaves the leg
    unable to call its list whole."""
    old = _record([Y], head=HEAD, run=77)
    first = _run(HEAD, 77)
    asked = []

    def get(self, path):
        if path == "git/ref/heads/ouroboros":
            return {"object": {"sha": HEAD}}
        asked.append(path)
        if path == "actions/runs/77/attempts/2/jobs?per_page=100&page=1":
            raise urllib.error.HTTPError(path, code, "middle attempt", None, None)
        assert path == "actions/runs/77/attempts/1/jobs?per_page=100&page=1"
        return {"jobs": [_leg(first, "windows", K, [old], scope="push")], "total_count": 1}

    seen = {}
    monkeypatch.setattr(pmc.GitHub, "get", get)
    monkeypatch.setattr(pmc, "route", _routed(seen))
    monkeypatch.setenv("GITHUB_RUN_ID", "77")
    assert pmc.main(["route", "--evidence-dir", str(tmp_path), "--os", "windows", "--run-url", "u", "--head", HEAD,
                     "--attempt", "3"]) == 0
    assert seen["prior_failures"] == [old]
    record = json.loads((tmp_path / "routing.json").read_text(encoding="utf-8"))
    gone = code == 410
    assert record.get("history_gaps", []) == ([{"run": 77, "attempt": 2, "part": "jobs", "base": None}] if gone else [])
    assert record["untracked"] == ([] if gone else None)
    assert asked == [f"actions/runs/77/attempts/{n}/jobs?per_page=100&page=1" for n in (2, 1)]


def test_a_later_attempt_takes_over_what_each_window_job_cancelled_before_it_carried_in(tmp_path, monkeypatch):
    """Two attempts of this check were cancelled after their window job published what it carries in and before any
    leg ran. The first named Y; by the second its source had expired, so that window carried in nothing. The third
    attempt's Windows leg reads both job lists: the emptier, newer record retires nothing, so the leg takes Y over,
    and with every earlier record read its own list is whole."""
    old = _record([Y], head=A, run=5)
    run = _run(HEAD, 77, "cancelled")
    pages = {2: [_stopped(run, {}, attempt=2)], 1: [_stopped(run, {"windows": [old]}, attempt=1)]}
    asked = []

    def get(self, path):
        if path == "git/ref/heads/ouroboros":
            return {"object": {"sha": HEAD}}
        number, attempt = (int(part) for part in path.split("/")[2:5:2])
        asked.append((number, attempt))
        return {"jobs": pages[attempt], "total_count": 1}

    seen = {}
    monkeypatch.setattr(pmc.GitHub, "get", get)
    monkeypatch.setattr(pmc, "route", _routed(seen))
    monkeypatch.setenv("GITHUB_RUN_ID", "77")
    assert pmc.main(["route", "--evidence-dir", str(tmp_path), "--os", "windows", "--run-url", "u", "--head", HEAD,
                     "--attempt", "3"]) == 0
    assert seen["prior_failures"] == [old] and asked == [(77, 2), (77, 1)]
    assert json.loads((tmp_path / "routing.json").read_text(encoding="utf-8"))["untracked"] == []


@pytest.mark.parametrize("descendant", ["provisional checkpoints", "qualified"])
def test_a_provisional_checkpoint_closes_nothing_and_only_a_qualified_descendant_ends_the_unseeded_walk(descendant):
    """Nothing is confirmed yet. The newest run's window read its history whole and its legs measured what it chose, as
    did the run before it: matched provisional checkpoints. The oldest run was re-run after them and its second attempt
    could not hand X over. A checkpoint closes nothing, so the walk reads on, one job page per run, to the pre-receipt
    run that ends the search, and carries the rerun's red under its own source from its own base. A qualified newest run
    closes the walk behind it instead: its page is the only one read."""
    mid = "2" * 39 + "3"  # between A and B
    late = _record([X], head=A, run=1, attempt=2)
    qualified = descendant == "qualified"
    newest, middle, rerun = _run(B, 3, "success"), _run(mid, 2, "success"), _run(A, 1, attempt=2)
    legacy = {**_run(OLD, 4), "display_title": "Merge pull request #1 from someone/change"}
    scope = "chain" if qualified else "push"
    jobs = Jobs({
        (3, 1): [_window(newest, mid, confirmed=qualified, carried={}), _leg(newest, "linux", mid, scope=scope),
                 _leg(newest, "windows", mid, scope=scope)],
        (2, 1): [_window(middle, A, confirmed=False, carried={}), _leg(middle, "linux", A, scope="push"),
                 _leg(middle, "windows", A, scope="push")],
        (1, 2): [_window(rerun, K, confirmed=False, carried={}), _leg(rerun, "linux", K, scope="push"),
                 _leg(rerun, "windows", K, [late], scope="push")],
    })
    base, why, continuous, replay = _choose([newest, middle, rerun, legacy], jobs)
    if qualified:
        assert (base, why, continuous) == (B, "eligible on both systems (run 3, attempt 1, success)", True)
        assert jobs.asked == [(3, 1)] and replay["awaiting_handoff"] == {}
    else:
        assert (base, why, continuous) == (K, "earlier range still pending", False)
        assert jobs.asked == [(3, 1), (2, 1), (1, 2)] and replay["awaiting_handoff"] == {"windows": [late]}
    assert replay["history_gaps"] == [] and replay["history_read"] is True


@pytest.mark.parametrize("fault", ["outage", "transport", "broken answer"])
def test_a_receipt_already_read_survives_an_earlier_attempt_that_cannot_be_read(fault):
    """Every job of A was re-run. The second attempt's job page shows the window job and the Windows leg, which could
    not hand X over, and announces one job more than it lists. Looking for the Linux leg, the walk asks for the first
    attempt, which cannot be read now. Nothing already read is taken back: the base the page shows is kept, X stays
    owed under its own source, no loss is named and the range stays provisional."""
    late = _record([X], head=A, run=1, attempt=2)
    run = _run(A, 1, attempt=2)
    earlier = {"outage": _unavailable(), "transport": TimeoutError("timed out"),
               "broken answer": {"message": "Server Error"}}[fault]
    jobs = Jobs({(1, 2): Short([_window(run, K, carried={}), _leg(run, "windows", K, [late], scope="push")]),
                 (1, 1): earlier})
    base, why, continuous, replay = _choose([run], jobs)
    assert (base, why, continuous) == (
        K, "retained starting base of run 1, attempt 2; earlier history unreadable", False)
    assert replay == {"history_read": False, "awaiting_handoff": {"windows": [late]}, "history_gaps": []}
    assert jobs.asked == [(1, 2), (1, 1)]


@pytest.mark.parametrize("code", [410, 404, 503])
def test_a_middle_attempt_that_is_gone_or_unread_hides_no_attempt_that_survives_behind_it(code):
    """Three attempts of A. The third re-ran the Windows leg alone, which still could not hand X over. The second's
    job list is gone for good (410) or cannot be read now (a bare 404, an outage). The first survives: it holds the
    window job and the Linux leg, which could not hand Y over. The walk reads on behind the middle attempt, so the
    base stays and both records stay owed under their own sources; the newest whole Windows receipt still stands for
    that leg. Only the 410 is a named gap, and only it leaves the range confirmed."""
    late, early = _record([X], head=A, run=1, attempt=3), _record([Y], head=A, run=1)
    run = _run(A, 1, attempt=3)
    jobs = Jobs({(1, 3): [_leg(run, "windows", K, [late], scope="push")],
                 (1, 2): urllib.error.HTTPError("https://api.github.invalid", code, "middle attempt", None, None),
                 (1, 1): [_window(run, K, carried={}, attempt=1),
                          _leg(run, "linux", K, [early], scope="push", attempt=1),
                          _leg(run, "windows", K, [_record([X], head=A, run=1)], scope="push", attempt=1)]})
    base, why, continuous, replay = _choose([run], jobs)
    gone = code == 410
    assert (base, continuous) == (K, gone)
    # The base is credited to the attempt whose window job chose it, not to the newest attempt read.
    assert why == "retained starting base of run 1, attempt 1" + ("" if gone else "; earlier history unreadable")
    assert replay == {"history_read": gone, "awaiting_handoff": {"linux": [early], "windows": [late]},
                      "history_gaps": [{"run": 1, "attempt": 2, "part": "jobs", "base": None}] if gone else []}
    assert jobs.asked == [(1, 3), (1, 2), (1, 1)]


@pytest.mark.parametrize("code", [410, 404])
def test_the_attempts_behind_a_lost_newest_attempt_keep_their_base_and_records(code):
    """A red check of A was re-run, and the job list of that second attempt is gone (410) or answers a bare 404. The
    first attempt survives with the base it kept and the Windows record of X. The loss is named as before, and the
    walk still reads the first attempt: K stays the base and X stays owed. `prior_window` itself still leaves the
    failure of its own newest attempt to its caller."""
    early = _record([X], head=A, run=1)
    run = _run(A, 1, attempt=2)
    jobs = Jobs({(1, 2): urllib.error.HTTPError("https://api.github.invalid", code, "newest attempt", None, None),
                 (1, 1): [_window(run, K, carried={}, attempt=1), _leg(run, "linux", K, attempt=1),
                          _leg(run, "windows", K, [early], scope="push", attempt=1)]})
    base, why, continuous, replay = _choose([run], jobs)
    gone = code == 410
    assert (base, continuous) == (K, gone)
    assert why == "retained starting base of run 1, attempt 1" + ("" if gone else "; earlier history unreadable")
    assert replay == {"history_read": gone, "awaiting_handoff": {"windows": [early]},
                      "history_gaps": [{"run": 1, "attempt": 2, "part": "jobs", "base": None}] if gone else []}
    assert jobs.asked == [(1, 2), (1, 1)]
    with pytest.raises(urllib.error.HTTPError):
        window.prior_window(run, jobs)


@pytest.mark.parametrize("fault", ["not found", "outage", "transport", "broken answer"])
def test_a_newest_attempt_that_cannot_be_read_now_hides_no_finished_attempt_and_only_an_outage_ends_the_search(fault):
    """A red check of A was re-run, and the job list of that second attempt cannot be read now. Its first attempt keeps
    exact endpoints: one request recovers the base it kept and the Windows record of X, and the range stays
    provisional. A bare 404 answers for that one list, so the walk reads on to the older run and its record of Y. An
    outage or a broken answer ends the search where it ended before: no older run is asked for."""
    mid = "1" * 39 + "2"  # between K and A
    early, older = _record([X], head=A, run=2), _record([Y], head=mid, run=1)
    run, behind = _run(A, 2, attempt=2), _run(mid, 1)
    unread = {"not found": urllib.error.HTTPError("https://api.github.invalid", 404, "Not Found", None, None),
              "outage": _unavailable(), "transport": TimeoutError("timed out"),
              "broken answer": {"message": "Server Error"}}[fault]
    jobs = Jobs({(2, 2): unread,
                 (2, 1): [_window(run, K, carried={}, attempt=1), _leg(run, "linux", K, attempt=1),
                          _leg(run, "windows", K, [early], scope="push", attempt=1)],
                 (1, 1): [_window(behind, K, carried={}), _leg(behind, "linux", K),
                          _leg(behind, "windows", K, [older], scope="push")]})
    base, why, continuous, replay = _choose([run, behind], jobs)
    assert (base, why, continuous) == (
        K, "retained starting base of run 2, attempt 1; earlier history unreadable", False)
    walked_on = fault == "not found"
    assert replay == {"history_read": False, "history_gaps": [],
                      "awaiting_handoff": {"windows": [early, older] if walked_on else [early]}}
    assert jobs.asked == [(2, 2), (2, 1)] + ([(1, 1)] if walked_on else [])


@pytest.mark.parametrize("sibling", ["rerun after it qualified", "debt it took over", "green", "on a short exact list",
                                     "exact list unavailable", "qualified by re-run legs", "of an older commit"])
def test_a_qualified_check_closes_the_range_before_its_head_and_not_a_sibling_check_of_that_head(sibling):
    """The push check of A (run 1) could not hand X over on Windows. The owner's later dispatch at A (run 2) carried
    that record in, settled it and qualified A. A is no descendant of A: the dispatch closes the range before A, not
    the other check of A itself. The activity listing shows an older commit's check between the two, so the next
    window stops at the dispatch and asks once for the checks of A by branch and commit. A re-run of the push check
    that failed after the dispatch qualified leaves its own record owed, under its own attempt. The record carried in
    by the window of the exact attempt that qualified is not owed again, and a record of an older commit stays
    closed. A dispatch that qualified in a re-run of its legs lists no window of that attempt: its first attempt is
    read, and as that one did not itself qualify, nothing is subtracted. A green sibling costs no request. A list
    that is not whole keeps what it shows, a list that fails
    shows nothing, and neither confirms a range. A dispatch that qualified a later commit never reaches the push
    check, and no check of an older commit is read in any case."""
    early, late = _record([X], head=A, run=1), _record([X], head=A, run=1, attempt=2)
    closed = _record([Y], head=K, run=7, base=OLD)
    at = B if sibling == "of an older commit" else A
    relegged, rerun = sibling == "qualified by re-run legs", sibling not in ("debt it took over", "green")
    push = _run(A, 1, "success" if sibling == "green" else "failure", attempt=2 if rerun else 1)
    dispatch = _run(at, 2, "success", attempt=2 if relegged else 1, event="workflow_dispatch")
    between, behind = _run(K, 8), _run(K, 7)
    legs = [_leg(dispatch, "linux", K), _leg(dispatch, "windows", K)]
    listed = [dispatch] if at == B else [dispatch, push]
    exact = {"on a short exact list": Short(listed), "exact list unavailable": _unavailable()}.get(sibling, listed)
    jobs = Jobs({
        (2, dispatch["run_attempt"]): legs if relegged else [_window(dispatch, K, carried={"windows": [early]}), *legs],
        (1, 2): [_leg(push, "windows", K, [early, late, closed], scope="push")],
        (1, 1): [_window(push, K, carried={}, attempt=1), _leg(push, "linux", K, attempt=1),
                 _leg(push, "windows", K, [early], scope="push", attempt=1)],
    }, siblings=exact)
    if relegged:  # its first attempt carried the record in and stayed one system short of the range: no qualifier
        jobs.pages[2, 1] = [_window(dispatch, K, carried={"windows": [early]}, attempt=1),
                            _leg(dispatch, "linux", K, attempt=1),
                            _leg(dispatch, "windows", K, scope="push", attempt=1)]
    base, why, continuous, replay = _choose([dispatch, between, push, behind], jobs)
    whole = sibling not in ("on a short exact list", "exact list unavailable")
    assert (base, continuous) == (at, whole)
    assert why == f"eligible on both systems (run 2, attempt {dispatch['run_attempt']}, success)" + (
        "" if whole else "; earlier history unreadable")
    owed = {"rerun after it qualified": [late], "on a short exact list": [late],
            "qualified by re-run legs": [early, late]}.get(sibling)
    assert replay == {"history_read": whole, "history_gaps": [], "awaiting_handoff": {"windows": owed} if owed else {}}
    read = [(1, 2), (1, 1)] if owed else [(1, 1)] if sibling == "debt it took over" else []
    assert jobs.asked == [(2, dispatch["run_attempt"])] + read + ([(2, 1)] if relegged else [])
    assert jobs.listed == [f"actions/workflows/post-merge.yml/runs?branch=ouroboros&head_sha={at}&per_page=100&page=1"]


@pytest.mark.parametrize("again", ["as it was read", "queued for a re-run", "unfinished in the same attempt",
                                   "re-run to a red", "re-run and qualified again", "skipped on a re-run"])
def test_a_run_listed_again_is_one_observation_for_each_attempt_and_state(tmp_path, monkeypatch, again):
    """The run list shows the dispatch at A in its first attempt, finished and qualified. The list of that commit's
    checks, read a moment later, shows the same run again. As it was read, it is the same observation and the range is
    confirmed. Queued for a second attempt, or unfinished in the attempt already read as finished, it confirms no
    range: the open attempt does not vanish behind the first, and the first attempt's pages are not fetched again. A
    second attempt that finished red qualifies nothing by its one leg and borrows none from the first: its own record
    at that head is carried in and the range stays unconfirmed. A second attempt that measured both systems itself
    qualifies the head again, by its own two legs. One that GitHub concluded `skipped` ran nothing and is not read;
    as the run's newest listed state it takes back the first attempt's confirmation, so the base is named and the
    range unconfirmed, at no request."""
    late = _record([X], head=A, run=2, attempt=2)
    first = _run(A, 2, "success", event="workflow_dispatch")
    second = {**first, "run_attempt": 2}
    listed = {"as it was read": first, "queued for a re-run": {**second, "status": "queued", "conclusion": None},
              "unfinished in the same attempt": {**first, "status": "in_progress", "conclusion": None},
              "re-run to a red": {**second, "conclusion": "failure"}, "re-run and qualified again": second,
              "skipped on a re-run": {**second, "conclusion": "skipped"}}[again]
    jobs = Jobs({(2, 1): [_window(first, K, carried={}), _leg(first, "linux", K), _leg(first, "windows", K)],
                 (2, 2): ([_leg(listed, "windows", K, [late], scope="push")] if again == "re-run to a red" else
                          [_window(second, K, carried={}), _leg(second, "linux", K), _leg(second, "windows", K)])})
    exact = []

    def get(self, path):
        if "head_sha=" in path:
            exact.append(path)
            return {"total_count": 1, "workflow_runs": [listed]}
        if path.startswith("actions/workflows/"):
            return {"total_count": 2, "workflow_runs": [first, _run(K, 7)]}
        return jobs(path)

    monkeypatch.setattr(pmc.GitHub, "get", get)
    monkeypatch.setattr(pmc, "is_ancestor", lambda older, newer: older < newer)
    args = type("Args", (), {"head": None, "evidence_dir": tmp_path})
    assert pmc.command_base(args, {"GITHUB_SHA": HEAD, "BEFORE": B}) == 0
    record = json.loads((tmp_path / "base.json").read_text(encoding="utf-8"))
    requalified = again == "re-run and qualified again"
    confirmed = requalified or again == "as it was read"
    assert (record["base"], record["continuous"]) == (A, confirmed)
    assert record["why"] == f"eligible on both systems (run 2, attempt {2 if requalified else 1}, success)" + (
        "" if confirmed else "; earlier history unreadable")
    assert record["replay"] == {"history_read": confirmed, "history_gaps": [],
                                "awaiting_handoff": {"windows": [late]} if again == "re-run to a red" else {}}
    assert jobs.asked == [(2, 1)] + ([(2, 2)] if again.startswith("re-run") else [])
    assert exact == [f"actions/workflows/post-merge.yml/runs?branch=ouroboros&head_sha={A}&per_page=100&page=1"]


@pytest.mark.parametrize("listed", ["before the qualified check", "after it"])
def test_a_check_beside_a_qualified_one_adds_the_same_wherever_the_listing_put_it(listed):
    """Two checks of A: the dispatch (run 2) qualified it, and run 3, created after the dispatch, did not. Run 3's
    window carried in a record of an older commit and the push check's record at A; its re-run left both owed beside a
    red of its own. The dispatch's own window had carried the push check's record in and settled it. Wherever the
    listing puts run 3, the next window starts at A: run 3 keeps no older base open, the older commit's record stays
    closed, the exact record the qualifying attempt took over is not owed again, and run 3's own later red is. Listed
    first, with an older commit's check between the two, that check is read before A is known to be qualified and
    then counts for nothing, as a check behind a qualified head never does."""
    mid = "1" * 39 + "2"  # between K and A
    early, late = _record([X], head=A, run=1), _record([X], head=A, run=3, attempt=2)
    closed, stale = _record([Y], head=K, run=7, base=OLD), _record([Y], head=mid, run=8)
    later, between = _run(A, 3, attempt=2), _run(mid, 8)
    dispatch = _run(A, 2, "success", event="workflow_dispatch")
    jobs = Jobs({
        (3, 2): [_leg(later, "windows", K, [closed, early, late], scope="push")],
        (3, 1): [_window(later, K, carried={"windows": [closed, early]}, attempt=1), _leg(later, "linux", K, attempt=1),
                 _leg(later, "windows", K, [closed, early], scope="push", attempt=1)],
        (8, 1): [_window(between, K, carried={}), _leg(between, "linux", K),
                 _leg(between, "windows", K, [stale], scope="push")],
        (2, 1): [_window(dispatch, K, carried={"windows": [early]}), _leg(dispatch, "linux", K),
                 _leg(dispatch, "windows", K)],
    }, siblings=[later, dispatch])
    first = listed == "before the qualified check"
    runs = [later, between, dispatch] if first else [dispatch, between, later]
    base, why, continuous, replay = _choose([*runs, _run(K, 7)], jobs)
    assert (base, why, continuous) == (A, "eligible on both systems (run 2, attempt 1, success)", True)
    assert replay == {"history_read": True, "history_gaps": [], "awaiting_handoff": {"windows": [late]}}
    assert jobs.asked == ([(3, 2), (3, 1), (8, 1), (2, 1)] if first else [(2, 1), (3, 2), (3, 1)])
    assert jobs.listed == [f"actions/workflows/post-merge.yml/runs?branch=ouroboros&head_sha={A}&per_page=100&page=1"]


@pytest.mark.parametrize("second", ["queued for a re-run", "re-run to a red of its own",
                                    "re-run that names the record again"])
@pytest.mark.parametrize("listed", ["sibling first", "sibling behind"])
def test_what_an_attempt_settled_when_it_qualified_stays_settled_under_a_newer_attempt_of_its_run(listed, second):
    """The push check of A (run 1) could not hand X over on Windows. The dispatch at A (run 2) carried that record in,
    settled it and qualified A in its first attempt, and the run list shows it so. The list of A's checks, read a
    moment later, shows run 2 in a second attempt. Whatever that attempt is, the range is no longer confirmed, and
    the first attempt's settlement stands as the observation it was: the push check's record is not owed again,
    wherever the listing put the push check. A second attempt still queued adds nothing. One that re-ran its Windows
    leg to a red of its own leaves that record owed under its own source, though it names the same test. One whose
    receipt names the push check's record again is the newest word of the run that took it over: the record is owed,
    once."""
    early, late = _record([X], head=A, run=1), _record([X], head=A, run=2, attempt=2)
    push, first = _run(A, 1), _run(A, 2, "success", event="workflow_dispatch")
    again = ({**first, "run_attempt": 2, "status": "queued", "conclusion": None} if second == "queued for a re-run"
             else {**first, "run_attempt": 2, "conclusion": "failure"})
    owed = {"queued for a re-run": [], "re-run to a red of its own": [late],
            "re-run that names the record again": [early, late]}[second]
    behind = listed == "sibling behind"
    jobs = Jobs({
        (2, 1): [_window(first, K, carried={"windows": [early]}), _leg(first, "linux", K), _leg(first, "windows", K)],
        (2, 2): [_leg(again, "windows", K, owed, scope="push")],
        (1, 1): [_window(push, K, carried={}), _leg(push, "linux", K), _leg(push, "windows", K, [early], scope="push")],
    }, siblings=[again, push] if behind else [push, again])
    runs = [first, _run(K, 7)] if behind else [push, first, _run(K, 7)]
    base, why, continuous, replay = _choose(runs, jobs)
    assert (base, why, continuous) == (
        A, "eligible on both systems (run 2, attempt 1, success); earlier history unreadable", False)
    assert replay == {"history_read": False, "history_gaps": [], "awaiting_handoff": {"windows": owed} if owed else {}}
    rerun = [(2, 2)] if owed else []  # the first attempt's pages are never asked for twice
    assert jobs.asked == ([(2, 1), *rerun, (1, 1)] if behind else [(1, 1), (2, 1), *rerun])
    assert jobs.listed == [f"actions/workflows/post-merge.yml/runs?branch=ouroboros&head_sha={A}&per_page=100&page=1"]


@pytest.mark.parametrize("git", ["answers", "one comparison times out", "stops answering"])
def test_a_comparison_git_cannot_make_takes_back_no_record_and_confirms_no_range(tmp_path, monkeypatch, capsys, git):
    """A check of an older commit (run 8) was listed before the dispatch that qualified A: its Windows leg could not
    hand Y over and its Linux leg ran without leaving a record. Whether that commit lies behind A is Git's to say.
    When Git answers, the qualified head closes what lies behind it and nothing is carried in. When that comparison
    times out, or Git stops answering from there on, nothing is closed on a guess: the record and the gap already
    read are carried in as they were read, no base is taken on a comparison that was not made, and the range is
    unconfirmed. The history is unread, never gone: no loss is invented for it. The Windows leg then takes the
    record over like any other, its list is whole, and it measures no continuous range."""
    mid = "1" * 39 + "2"  # between K and A
    stale = _record([Y], head=mid, run=8)
    gap = {"run": 8, "attempt": 1, "part": "linux", "base": K}
    older, dispatch = _run(mid, 8), _run(A, 2, "success", event="workflow_dispatch")
    jobs = Jobs({
        (8, 1): [_window(older, K, carried={}), _leg(older, "linux", K, ABSENT, scope=None, ran=True),
                 _leg(older, "windows", K, [stale], scope="push")],
        (2, 1): [_window(dispatch, K, carried={}), _leg(dispatch, "linux", K), _leg(dispatch, "windows", K)],
    })
    down = []

    def ancestor(low, high):
        # Only the weighing asks whether the older commit lies behind the qualified head.
        if git != "answers" and (down or (low, high) == (mid, A)):
            if git == "stops answering":
                down.append((low, high))
                raise OSError("cannot start git")
            raise subprocess.TimeoutExpired("git", 300)
        return low < high

    def get(self, path):
        if "head_sha=" in path:
            return {"total_count": 1, "workflow_runs": [dispatch]}
        if path.startswith("actions/workflows/"):
            return {"total_count": 3, "workflow_runs": [older, dispatch, _run(K, 7)]}
        return jobs(path)

    monkeypatch.setattr(pmc.GitHub, "get", get)
    monkeypatch.setattr(pmc, "is_ancestor", ancestor)
    args = type("Args", (), {"head": None, "evidence_dir": tmp_path})
    assert pmc.command_base(args, {"GITHUB_SHA": HEAD, "BEFORE": B}) == 0
    record = json.loads((tmp_path / "base.json").read_text(encoding="utf-8"))
    answered, qualified = git == "answers", "eligible on both systems (run 2, attempt 1, success)"
    assert (record["base"], record["why"], record["continuous"]) == {
        "answers": (A, qualified, True),
        "one comparison times out": (
            K, "retained starting base of run 8, attempt 1; earlier history unreadable", False),
        "stops answering": (A, qualified + "; earlier history unreadable", False)}[git]
    assert record["replay"] == {"history_read": answered, "history_gaps": [] if answered else [gap],
                                "awaiting_handoff": {} if answered else {"windows": [stale]}}
    # The steps that carry the record: the leg takes over what the window named, and calls no range continuous.
    (tmp_path / "classification.json").write_text(json.dumps({
        "base": record["base"], "head": HEAD, "selection_executed": True, "failed": [], "flaky": [], "problems": []}),
        encoding="utf-8")
    seen, output = {}, tmp_path / "outputs"
    monkeypatch.setattr(pmc, "route", _routed(seen))
    monkeypatch.setattr(pmc, "merges", lambda *_: [])
    monkeypatch.setenv("GITHUB_OUTPUT", str(output))
    monkeypatch.delenv("GITHUB_STEP_SUMMARY", raising=False)
    for name, value in (("GITHUB_RUN_ID", "77"), ("GITHUB_RUN_ATTEMPT", "1")):
        monkeypatch.setenv(name, value)
    where = ["--evidence-dir", str(tmp_path), "--os", "windows"]
    assert pmc.main(["route", *where, "--run-url", "u", "--head", HEAD]) == 0
    assert seen["prior_failures"] == ([] if answered else [stale])
    routing = json.loads((tmp_path / "routing.json").read_text(encoding="utf-8"))
    assert routing["untracked"] == [] and routing.get("history_gaps", []) == ([] if answered else [gap])
    assert pmc.main(["summary", *where]) == 0
    emitted = dict(line.split("=", 1) for line in output.read_text(encoding="utf-8").splitlines())
    assert json.loads(emitted["untracked"]) == [] and emitted["window_scope"] == ("chain" if answered else "push")
    logged = capsys.readouterr().out
    assert ("Earlier coverage remains pending." in logged) is not answered
    assert ("Git could not compare" in logged) is not answered


def test_a_weighing_that_fails_carries_in_what_was_read_unweighed(monkeypatch):
    """Whatever goes wrong while the facts are weighed (here the reading of what the qualifying attempt settled), no
    record already read is taken back: the push check's record is carried in as it was read, unweighed, where a
    sound weighing finds it settled, and the range stays unconfirmed."""
    early = _record([X], head=A, run=1)
    push, dispatch = _run(A, 1), _run(A, 2, "success", event="workflow_dispatch")
    jobs = Jobs({
        (2, 1): [_window(dispatch, K, carried={"windows": [early]}), _leg(dispatch, "linux", K),
                 _leg(dispatch, "windows", K)],
        (1, 1): [_window(push, K, carried={}), _leg(push, "linux", K), _leg(push, "windows", K, [early], scope="push")],
    }, siblings=[dispatch, push])

    def broken(*_):
        raise RuntimeError("the weighing broke")

    monkeypatch.setattr(window, "_took_over", broken)
    base, why, continuous, replay = _choose([dispatch, _run(K, 7)], jobs)
    assert (base, why, continuous) == (
        A, "eligible on both systems (run 2, attempt 1, success); earlier history unreadable", False)
    assert replay == {"history_read": False, "history_gaps": [], "awaiting_handoff": {"windows": [early]}}


@pytest.mark.parametrize("fault", [410, 404, 503], ids=["gone", "not found", "outage"])
@pytest.mark.parametrize("listed", ["sibling first", "sibling behind"])
def test_an_attempt_that_qualified_confirms_nothing_once_a_newer_one_of_its_run_is_listed_and_cannot_be_read(
        listed, fault):
    """The run list shows the dispatch at A (run 2) in its first attempt, which carried the push check's record X in,
    settled it and qualified A. The list of A's checks, read a moment later, shows run 2 finished in a second attempt
    whose job list cannot be read. That attempt is the run's word on A now, so the first one confirms no range: it
    still names the base, and what it took over stays settled, wherever the listing put the push check. A list that
    is gone is a named loss; a bare 404 and an outage prove none and name none, and an outage ends the reading where
    it happened."""
    early = _record([X], head=A, run=1)
    push, first = _run(A, 1), _run(A, 2, "success", event="workflow_dispatch")
    again = {**first, "run_attempt": 2, "conclusion": "failure"}
    lost = urllib.error.HTTPError("https://api.github.invalid", fault, "newest attempt", None, None)
    behind = listed == "sibling behind"
    jobs = Jobs({
        (2, 1): [_window(first, K, carried={"windows": [early]}), _leg(first, "linux", K), _leg(first, "windows", K)],
        (2, 2): lost,
        (1, 1): [_window(push, K, carried={}), _leg(push, "linux", K), _leg(push, "windows", K, [early], scope="push")],
    }, siblings=[again, push] if behind else [push, again])
    base, why, continuous, replay = _choose([first, _run(K, 7)], jobs)
    assert (base, why, continuous) == (
        A, "eligible on both systems (run 2, attempt 1, success); earlier history unreadable", False)
    assert replay == {"history_read": False, "awaiting_handoff": {},
                      "history_gaps": [{"run": 2, "attempt": 2, "part": "jobs", "base": None}] if fault == 410 else []}
    unread = behind and fault == 503  # the outage ended the reading before the push check was reached
    assert jobs.asked == ([(2, 1), (2, 2)] + ([] if unread else [(1, 1)]) if behind else [(2, 1), (1, 1), (2, 2)])
    assert jobs.listed == [f"actions/workflows/post-merge.yml/runs?branch=ouroboros&head_sha={A}&per_page=100&page=1"]


@pytest.mark.parametrize("newer", ["queued", "red of its own", "gone"])
@pytest.mark.parametrize("listed", ["the re-run first", "the sibling first"])
def test_an_earlier_attempt_that_qualified_settles_what_it_took_over_and_qualifies_nothing_now(listed, newer):
    """One listing, which shows a run by its newest attempt only. The dispatch at A (run 2) qualified A in its first
    attempt and took the push check's record X over. It has since been re-run, and the listing shows that second
    attempt: still queued, finished with a red of its own on the one leg it re-ran, or finished with a job list that
    is gone. The first attempt turns up in the job lists the reading goes through anyway. Both its legs are its own,
    so that it qualified is a fact, and the push check's record is not owed again, whichever of the two the listing
    shows first. It is no more than that: nothing makes the newer attempt qualify, A closes no range, and the window
    starts where the older qualified check ends, with the dispatch's own red or its lost job list named."""
    early, late = _record([X], head=A, run=1), _record([X], head=A, run=2, attempt=2)
    push, older = _run(A, 1), _run(K, 7, "success")
    first = _run(A, 2, "success", event="workflow_dispatch")  # the dispatch as its first attempt left it
    again = ({**first, "run_attempt": 2, "status": "queued", "conclusion": None} if newer == "queued"
             else {**first, "run_attempt": 2, "conclusion": "failure"})
    gone = urllib.error.HTTPError("https://api.github.invalid", 410, "Gone", None, None)
    jobs = Jobs({
        (2, 1): [_window(first, K, carried={"windows": [early]}), _leg(first, "linux", K), _leg(first, "windows", K)],
        (2, 2): gone if newer == "gone" else [_leg(again, "windows", K, [late], scope="push")],
        (1, 1): [_window(push, K, carried={}), _leg(push, "linux", K), _leg(push, "windows", K, [early], scope="push")],
        (7, 1): [_leg(older, "linux", OLD), _leg(older, "windows", OLD)],
    }, siblings=[older])
    rerun_first = listed == "the re-run first"
    base, why, continuous, replay = _choose([again, push, older] if rerun_first else [push, again, older], jobs)
    assert (base, continuous) == (K, newer != "queued")
    assert why == "eligible on both systems (run 7, attempt 1, success)" + (
        "; earlier history unreadable" if newer == "queued" else "")
    assert replay == {
        "history_read": newer != "queued", "awaiting_handoff": {"windows": [late]} if newer == "red of its own" else {},
        "history_gaps": [{"run": 2, "attempt": 2, "part": "jobs", "base": None}] if newer == "gone" else []}
    dispatch = [(2, 1)] if newer == "queued" else [(2, 2), (2, 1)]  # no job list is asked for twice
    assert jobs.asked == ([*dispatch, (1, 1), (7, 1)] if rerun_first else [(1, 1), *dispatch, (7, 1)])
    assert jobs.listed == [f"actions/workflows/post-merge.yml/runs?branch=ouroboros&head_sha={K}&per_page=100&page=1"]


@pytest.mark.parametrize("later", [False, True], ids=["nothing since", "a red since"])
@pytest.mark.parametrize("seen_first", ["the later qualifier", "the one that took it over", "a qualified descendant"])
def test_what_one_qualifier_took_over_stays_settled_beside_a_later_qualifier_that_took_nothing(seen_first, later):
    """Three checks of A. The push check (run 1) could not hand X over on Windows. A dispatch (run 2) carried that
    record in, settled it and qualified A, green. A later dispatch (run 3) found nothing owed, carried nothing in and
    qualified A too. Whichever of the two the listing shows first closes the range before A, and the other, being
    green, is set aside at no request. While the push check still names a record at A that no attempt read so far
    took over, the green check is read after all: the first dispatch took X over, so X is not owed again, though the
    later qualifier never carried it. A red the push check left in a re-run after both is owed under its own source.
    With no record left to ask about, the green check costs no request. Behind a check that qualified a later commit
    none of this is read at all."""
    early, late = _record([X], head=A, run=1), _record([Y], head=A, run=1, attempt=2)
    push = _run(A, 1, attempt=2 if later else 1)
    took, nothing = (_run(A, number, "success", event="workflow_dispatch") for number in (2, 3))
    after = _run(B, 4, "success")
    descendant = seen_first == "a qualified descendant"
    jobs = Jobs({
        (1, 2): [_leg(push, "windows", K, [early, late], scope="push")],
        (1, 1): [_window(push, K, carried={}, attempt=1), _leg(push, "linux", K, attempt=1),
                 _leg(push, "windows", K, [early], scope="push", attempt=1)],
        (2, 1): [_window(took, K, carried={"windows": [early]}), _leg(took, "linux", K), _leg(took, "windows", K)],
        (3, 1): [_window(nothing, K, carried={}), _leg(nothing, "linux", K), _leg(nothing, "windows", K)],
        (4, 1): [_window(after, A, carried={}), _leg(after, "linux", A), _leg(after, "windows", A)],
    }, siblings=[after] if descendant else [nothing, took, push])
    order = {"the later qualifier": [nothing, took, push], "the one that took it over": [took, nothing, push],
             "a qualified descendant": [after, nothing, took, push]}[seen_first]
    base, why, continuous, replay = _choose([*order, _run(K, 7)], jobs)
    at, number = (B, 4) if descendant else (A, 3 if seen_first == "the later qualifier" else 2)
    assert (base, why, continuous) == (at, f"eligible on both systems (run {number}, attempt 1, success)", True)
    owed = later and not descendant
    assert replay == {"history_read": True, "history_gaps": [], "awaiting_handoff": {"windows": [late]} if owed else {}}
    red = [(1, 2), (1, 1)] if later else [(1, 1)]
    assert jobs.asked == {"the later qualifier": [(3, 1), *red, (2, 1)],
                          "the one that took it over": [(2, 1), *red] + ([(3, 1)] if later else []),
                          "a qualified descendant": [(4, 1)]}[seen_first]
    assert jobs.listed == [f"actions/workflows/post-merge.yml/runs?branch=ouroboros&head_sha={at}&per_page=100&page=1"]


@pytest.mark.parametrize("listed", ["first as a skipped re-run", "first as a skipped re-run of a qualified attempt",
                                    "a first attempt skipped", "again as a skipped re-run"])
def test_a_skipped_attempt_qualifies_nothing_and_hides_no_receipt_of_the_attempts_before_it(listed):
    """The push check of A (run 2) kept K and could not hand X over on Windows; an older check qualified K. GitHub
    concluded a re-run of run 2 `skipped`: that attempt ran nothing. A listing shows a run by its newest attempt
    only, so where run 2 is first met in that state its first attempt is read behind it: X stays owed under its own
    source, the window keeps K, and nothing is left unread. Had the first attempt qualified A, the skipped one is
    still the run's word on A, and A starts nothing. A check skipped in its first attempt recorded nothing and costs
    no request. Where the first attempt was read before a dispatch qualified A, the list of A's checks showing the
    run again, skipped, takes nothing back and asks for nothing: X is owed beside the qualified check."""
    early = _record([X], head=A, run=2)
    finished, older = _run(A, 2), _run(K, 1, "success")
    dispatch = _run(A, 3, "success", event="workflow_dispatch")
    skipped = {**finished, "run_attempt": 1 if listed == "a first attempt skipped" else 2, "conclusion": "skipped"}
    again, qualified = listed == "again as a skipped re-run", listed.endswith("qualified attempt")
    jobs = Jobs({
        (2, 1): [_window(finished, K, carried={}), _leg(finished, "linux", K),
                 _leg(finished, "windows", K) if qualified else _leg(finished, "windows", K, [early], scope="push")],
        (3, 1): [_window(dispatch, K, carried={}), _leg(dispatch, "linux", K), _leg(dispatch, "windows", K)],
        (1, 1): [_leg(older, "linux", OLD), _leg(older, "windows", OLD)],
    }, siblings=[dispatch, skipped] if again else [older])
    base, why, continuous, replay = _walk([finished, dispatch, older] if again else [skipped, older], jobs)
    at, number = (A, 3) if again else (K, 1)
    assert (base, why, continuous) == (at, f"eligible on both systems (run {number}, attempt 1, success)", True)
    owed = again or listed == "first as a skipped re-run"
    assert replay == {"history_read": True, "history_gaps": [],
                      "awaiting_handoff": {"windows": [early]} if owed else {}}
    assert jobs.asked == {"a first attempt skipped": [(1, 1)], "again as a skipped re-run": [(2, 1), (3, 1)]}.get(
        listed, [(2, 1), (1, 1)])
    assert jobs.listed == [f"actions/workflows/post-merge.yml/runs?branch=ouroboros&head_sha={at}&per_page=100&page=1"]


@pytest.mark.parametrize("first", ["qualified", "qualified, a red since", "one system short", "job list not whole",
                                   "behind a qualified descendant"])
def test_what_a_check_took_over_when_it_first_qualified_stays_settled_when_its_re_run_legs_qualify_again(first):
    """The push check of A (run 1) could not hand X over on Windows. The dispatch at A (run 2) carried that record in
    and qualified A in its first attempt, its legs red for tests an issue already carries. Re-running the failed jobs
    lists both legs and no window job in the second attempt, which qualifies A again by its own two legs: it is the
    run's word on A and confirms the range, and it carried nothing in itself. While the push check still names X, the
    dispatch's first attempt is read. It qualified by its own job list, so what its own window carried in is not owed
    again, and a red the push check left in a re-run since is, under its own attempt. A first attempt one system
    short of the range did not qualify, and its Linux leg is not joined to the second attempt's Windows leg: X stays
    owed, as it does behind a job list that is not whole. Behind a check that qualified a later commit none of this
    is read."""
    early, late = _record([X], head=A, run=1), _record([X], head=A, run=1, attempt=2)
    since, descendant = first == "qualified, a red since", first == "behind a qualified descendant"
    push, after = _run(A, 1, attempt=2 if since else 1), _run(B, 4, "success")
    dispatch = _run(A, 2, attempt=2, event="workflow_dispatch")
    took = [_window(dispatch, K, carried={"windows": [early]}, attempt=1), _leg(dispatch, "linux", K, attempt=1),
            _leg(dispatch, "windows", K, scope="push" if first == "one system short" else "chain", attempt=1)]
    jobs = Jobs({
        (2, 2): [_leg(dispatch, "linux", K), _leg(dispatch, "windows", K)],
        (2, 1): Short(took) if first == "job list not whole" else took,
        (1, 2): [_leg(push, "windows", K, [early, late], scope="push")],
        (1, 1): [_window(push, K, carried={}, attempt=1), _leg(push, "linux", K, attempt=1),
                 _leg(push, "windows", K, [early], scope="push", attempt=1)],
        (4, 1): [_window(after, A, carried={}), _leg(after, "linux", A), _leg(after, "windows", A)],
    }, siblings=[after] if descendant else [dispatch, push])
    runs = [dispatch, push, _run(K, 7)]
    base, why, continuous, replay = _walk([after, *runs] if descendant else runs, jobs)
    at, word = (B, "run 4, attempt 1, success") if descendant else (A, "run 2, attempt 2, failure")
    assert (base, why, continuous) == (at, f"eligible on both systems ({word})", True)
    owed = {"qualified, a red since": [late], "one system short": [early], "job list not whole": [early]}.get(first)
    assert replay == {"history_read": True, "history_gaps": [],
                      "awaiting_handoff": {"windows": owed} if owed else {}}
    assert jobs.asked == ([(4, 1)] if descendant else [(2, 2), *([(1, 2)] if since else []), (1, 1), (2, 1)])
    assert jobs.listed == [f"actions/workflows/post-merge.yml/runs?branch=ouroboros&head_sha={at}&per_page=100&page=1"]


@pytest.mark.parametrize("sibling", ["qualifies", "job list not whole", "job list not found", "one system short"])
@pytest.mark.parametrize("owed", [False, True], ids=["nothing owed", "a red of its own"])
def test_a_green_check_that_qualifies_keeps_the_head_confirmed_when_the_first_qualifier_is_re_run(owed, sibling):
    """The run list shows the dispatch at A (run 2) qualified in its first attempt. The list of A's checks, read a
    moment later, shows run 2 in a second attempt that re-ran one leg, so it qualifies nothing, and a green check of
    A (run 3) that was never read. Run 2 confirms no range any more. Run 3 is then read for its own newest attempt,
    whether or not a record is left: measuring both systems by its own whole job list it qualifies A, and the range
    stays confirmed with run 2's own red carried in under its source. A job list of run 3 that is not whole or
    answers 404 qualifies nothing, and neither does one system's measurement: the base is named, the range
    unconfirmed. No job list is asked for twice."""
    late = _record([X], head=A, run=2, attempt=2)
    first, other = (_run(A, number, "success", event="workflow_dispatch") for number in (2, 3))
    again = {**first, "run_attempt": 2, "conclusion": "failure"}
    green = [_window(other, K, carried={}), _leg(other, "linux", K),
             _leg(other, "windows", K, scope="push" if sibling == "one system short" else "chain")]
    lost = urllib.error.HTTPError("https://api.github.invalid", 404, "Not Found", None, None)
    jobs = Jobs({
        (2, 1): [_window(first, K, carried={}), _leg(first, "linux", K), _leg(first, "windows", K)],
        (2, 2): [_leg(again, "windows", K, [late] if owed else (), scope="push" if owed else "chain")],
        (3, 1): {"job list not whole": Short(green), "job list not found": lost}.get(sibling, green),
    }, siblings=[other, again])
    base, why, continuous, replay = _walk([first, _run(K, 7)], jobs)
    confirmed = sibling == "qualifies"
    assert (base, continuous) == (A, confirmed)
    assert why == ("eligible on both systems (run 3, attempt 1, success)" if confirmed else
                   "eligible on both systems (run 2, attempt 1, success); earlier history unreadable")
    assert replay == {"history_read": confirmed, "history_gaps": [],
                      "awaiting_handoff": {"windows": [late]} if owed else {}}
    assert jobs.asked == [(2, 1), (2, 2), (3, 1)]
    assert jobs.listed == [f"actions/workflows/post-merge.yml/runs?branch=ouroboros&head_sha={A}&per_page=100&page=1"]


def _repository(path, commits):
    """A throwaway Git repository with `commits` commits on its first-parent line: its `git` and their SHAs, oldest
    first. The ancestry tests ask the real Git about it."""
    def git(*args):
        return subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@example.invalid",
                               "-c", "commit.gpgsign=false", *args], cwd=path, capture_output=True, text=True,
                              check=True).stdout.strip()

    path.mkdir()
    git("init", "-q", "-b", "ouroboros")
    shas = []
    for number in range(commits):
        (path / "line").write_text(str(number), encoding="utf-8")
        git("add", "-A")
        git("commit", "-q", "-m", f"commit {number}")
        shas.append(git("rev-parse", "HEAD"))
    return git, shas


def test_real_git_says_no_only_for_what_it_knows_and_raises_when_it_cannot_answer(tmp_path, monkeypatch):
    """`is_ancestor` on a real repository. Yes is a commit further down the first-parent line. No is an answer Git
    gave: the same commit, a descendant, a commit merged in from a side line, a line with no common history (the line
    ends before reaching it), a commit this clone does not hold, a name that is no commit. When Git cannot answer
    although it holds both commits (the commit between them cannot be read; there is no repository to ask), that is
    an error and never a no."""
    repo = tmp_path / "repo"
    git, (k, m, a, h) = _repository(repo, 4)
    monkeypatch.setattr(pmc, "REPO_ROOT", repo)
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path.parent))  # no repository above the fixture answers instead
    assert pmc.is_ancestor(k, h) and pmc.is_ancestor(a, h) and pmc.is_ancestor(k, m)
    assert not pmc.is_ancestor(h, h) and not pmc.is_ancestor(h, k)
    git("checkout", "-q", "-b", "side", k)
    (repo / "side").write_text("side", encoding="utf-8")
    git("add", "-A")
    git("commit", "-q", "-m", "side")
    side = git("rev-parse", "HEAD")
    git("checkout", "-q", "ouroboros")
    git("merge", "-q", "--no-ff", "-m", "merge", "side")
    merged = git("rev-parse", "HEAD")
    assert pmc.is_ancestor(h, merged) and pmc.is_ancestor(k, merged) and not pmc.is_ancestor(side, merged)
    git("checkout", "-q", "--orphan", "apart")
    git("commit", "-q", "-m", "apart")
    apart = git("rev-parse", "HEAD")
    assert not pmc.is_ancestor(apart, merged) and not pmc.is_ancestor(k, apart)
    absent = "d" * 40
    assert not pmc.is_ancestor(absent, merged) and not pmc.is_ancestor(k, absent)
    assert not pmc.is_ancestor("HEAD", merged)
    # Both commits are held and the one between them cannot be read: Git cannot count the line.
    lost = repo / ".git" / "objects" / a[:2] / a[2:]
    lost.chmod(0o644)
    lost.unlink()
    with pytest.raises(subprocess.CalledProcessError):
        pmc.is_ancestor(k, h)
    assert pmc.is_ancestor(k, m) and not pmc.is_ancestor(absent, h)  # what it can still answer, it answers
    monkeypatch.setattr(pmc, "REPO_ROOT", tmp_path)  # no repository here at all
    with pytest.raises(subprocess.CalledProcessError):
        pmc.is_ancestor(k, h)


@pytest.mark.parametrize("unplaced", [None, "the older check", "the base the newer check kept"])
def test_a_real_git_that_cannot_place_a_commit_keeps_what_was_read_and_confirms_no_range(
        tmp_path, monkeypatch, unplaced):
    """A real repository: K, then the commit of an older check, then that of a newer one, then this head. The newer
    check kept K, owes Y on Linux and left no Windows record; the older one owes X on Windows. When Git answers, both
    are carried in from K and the range is confirmed. When the count Git is asked for really fails (a nonzero exit
    although it holds both commits), that is no "not on this line": the older check is not dropped as foreign with K
    confirmed over it, and a base whose place is unknown is not taken. What was read before stays, the gap with it,
    and the range is unconfirmed."""
    repo = tmp_path / "repo"
    _, (k, p, n, h) = _repository(repo, 4)
    monkeypatch.setattr(pmc, "REPO_ROOT", repo)
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path.parent))
    newer, older = _run(n, 2), _run(p, 1)
    y, x = _record([Y], head=n, run=2, base=k), _record([X], head=p, run=1, base=k)
    gap = {"run": 2, "attempt": 1, "part": "windows", "base": k}
    jobs = Jobs({
        (2, 1): [_window(newer, k, carried={}), _leg(newer, "linux", k, [y], scope="push"),
                 _leg(newer, "windows", k, ABSENT, scope=None, ran=True)],
        (1, 1): [_window(older, k, carried={}), _leg(older, "linux", k), _leg(older, "windows", k, [x], scope="push")],
    })
    failing = {None: None, "the older check": f"{p}..{h}", "the base the newer check kept": f"{k}..{n}"}[unplaced]
    real = pmc._git

    def git(*args, **kwargs):
        if args[:3] == ("rev-list", "--first-parent", "--count") and args[3] == failing:
            kwargs["cwd"] = tmp_path  # the real Git, asked where it has no repository: a real nonzero exit
        return real(*args, **kwargs)

    def get(self, path):
        if path.startswith("actions/workflows/"):
            return {"total_count": 2, "workflow_runs": [newer, older]}
        return jobs(path)

    monkeypatch.setattr(pmc, "_git", git)
    monkeypatch.setattr(pmc.GitHub, "get", get)
    args = type("Args", (), {"head": None, "evidence_dir": tmp_path})
    assert pmc.command_base(args, {"GITHUB_SHA": h, "BEFORE": n}) == 0
    record = json.loads((tmp_path / "base.json").read_text(encoding="utf-8"))
    kept = "retained starting base of run 2, attempt 1"
    assert (record["base"], record["why"], record["continuous"]) == {
        None: (k, kept, True),
        "the older check": (k, kept + "; earlier history unreadable", False),
        "the base the newer check kept": (
            n, "the branch tip before this push (earlier coverage remains pending)", False)}[unplaced]
    assert record["replay"] == {"history_read": unplaced is None, "history_gaps": [gap],
                                "awaiting_handoff": {"linux": [y], **({"windows": [x]} if unplaced is None else {})}}
    assert jobs.asked == ([(2, 1), (1, 1)] if unplaced is None else [(2, 1)])


@pytest.mark.parametrize("open_attempt", [2, 1], ids=["a queued rerun", "a first attempt not yet run"])
def test_an_open_rerun_keeps_what_its_finished_attempt_recorded_and_confirms_no_range(open_attempt):
    """A's first attempt kept K and could not hand X over on Windows; an older run qualified K. The run list shows a
    run by its newest attempt only, and A's is still queued. As a re-run it stands in front of a finished attempt:
    the walk reads that one, keeps X owed under its own source and confirms no range over an attempt that has not
    reported. The open attempt is no measurement and is not read. A first attempt that has not run has recorded
    nothing: it costs no request and changes nothing."""
    late = _record([X], head=A, run=2)
    finished, qualified = _run(A, 2), _run(K, 1, "success")
    waiting = {**finished, "run_attempt": open_attempt, "status": "queued", "conclusion": None}
    jobs = Jobs({(2, 1): [_window(finished, K, carried={}), _leg(finished, "linux", K),
                          _leg(finished, "windows", K, [late], scope="push")],
                 (1, 1): [_leg(qualified, "linux", OLD), _leg(qualified, "windows", OLD)]})
    base, why, continuous, replay = _choose([waiting, qualified], jobs)
    rerun = open_attempt == 2
    assert (base, continuous) == (K, not rerun)
    assert why == "eligible on both systems (run 1, attempt 1, success)" + (
        "; earlier history unreadable" if rerun else "")
    assert replay == {"history_read": not rerun, "awaiting_handoff": {"windows": [late]} if rerun else {},
                      "history_gaps": []}
    assert jobs.asked == ([(2, 1), (1, 1)] if rerun else [(1, 1)])


@pytest.mark.parametrize("own", ["9", "2", ""], ids=["another run", "this check's own run", "no run number"])
def test_a_seeded_check_tells_its_own_run_from_another_runs_open_rerun_at_the_same_head(tmp_path, monkeypatch, own):
    """The owner seeds K at a head whose push check, run 2, kept K, could not hand X over on Windows and waits for a
    re-run. Run 9 is the seeded check: the open attempt at its own head is another run's, so that run's finished
    attempt is read, X is carried in under its own source and the seed confirms no range over an attempt that has not
    reported. When the open attempt is this check's own run it is no history: no request, and the seed stands.
    Without the run's number an open attempt at this head is still taken for this check's own."""
    late = _record([X], head=HEAD, run=2)
    finished = _run(HEAD, 2)
    waiting = {**finished, "run_attempt": 2, "status": "queued", "conclusion": None}
    seeded = {**_run(HEAD, 9, event="workflow_dispatch"), "status": "in_progress", "conclusion": None}
    jobs = Jobs({(2, 1): [_window(finished, K, carried={}), _leg(finished, "linux", K),
                          _leg(finished, "windows", K, [late], scope="push")]})

    def get(self, path):
        if path.startswith("actions/workflows/"):
            return {"total_count": 2, "workflow_runs": [seeded, waiting]}
        return jobs(path)

    monkeypatch.setattr(pmc.GitHub, "get", get)
    monkeypatch.setattr(pmc, "is_ancestor", lambda older, newer: older < newer)
    args = type("Args", (), {"head": None, "evidence_dir": tmp_path})
    environ = {"GITHUB_SHA": HEAD, "GITHUB_EVENT_NAME": "workflow_dispatch", "INITIAL_BASE": K, "GITHUB_RUN_ID": own,
               "GITHUB_REPOSITORY": "razzant/ouroboros", "GITHUB_REF": "refs/heads/ouroboros"}
    assert pmc.command_base(args, environ) == 0
    record = json.loads((tmp_path / "base.json").read_text(encoding="utf-8"))
    other = own == "9"
    assert (record["base"], record["why"], record["continuous"]) == (
        (K, "retained starting base of run 2, attempt 1; earlier history unreadable", False) if other
        else (K, "explicit first-parent window", True))
    assert record["replay"] == {"history_read": not other, "history_gaps": [],
                                "awaiting_handoff": {"windows": [late]} if other else {}}
    assert jobs.asked == ([(2, 1)] if other else [])


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
