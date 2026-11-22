"""The roster's open-run tails speak the work projection's one meaning.

``peer_roster._tree_tails`` and ``gateway/work_projection._project`` read the same executor fact —
``delegate_feed.current_activities``, the body observer's current in-process observation (a
measurement, never a new provider request): True working ("live"), False waiting ("open"), None
unanswered/stale/refused ("unknown"). For a run of a LIVE census member an open registration alone
is no longer "live"; what the projection never addressed keeps its own typing (review work, unbound
starts, runs of tasks that already answered, runs with no task an observation could answer for),
and a fresh observation outranks the registration's stop typing exactly as the projection's member
mapping does. Synthetic entries and a scripted observer: no daemon, no network, no action on any
task.

A worker holds no observer, so its current observation answers nothing: what the body's dated record
(``Feed.record``, read back by ``delegate_feed.recorded_activity``) last saw of such a run is said in past
words only, never as the live or open of the present, and only for a run the registration typed live.
"""
import json
import time
from types import SimpleNamespace

import pytest


def _entry(run_id, task_id="nanny-1", source="delegated_subagent", invocation_id="inv-1"):
    return SimpleNamespace(run_id=run_id, task_id=task_id, source=source,
                           invocation_id=invocation_id, terminal_state="")


@pytest.fixture()
def observer(monkeypatch):
    """Isolate ``_tree_tails`` and script the one executor observation it now reads.

    ``by_run`` scripts the answer per run id; ``stops`` are the run ids whose stop the live Pause
    could not confirm; ``keys`` records the exact (task, run, invocation) addressing the work
    projection uses, so a regression in the key shape fails here, not silently in production.
    """
    scripted = {"by_run": {}, "stops": frozenset(), "keys": None}

    def activities(_root, keys):
        scripted["keys"] = list(keys)
        return {key: scripted["by_run"].get(key[1]) for key in keys}

    monkeypatch.setattr("ouroboros.task_results.load_task_result", lambda root, task_id: None)
    monkeypatch.setattr("ouroboros.gateway.work_projection.unconfirmed_stops", lambda *a, **k: scripted["stops"])
    monkeypatch.setattr("ouroboros.delegate_feed.current_activities", activities)
    monkeypatch.setattr("supervisor.continuation_admission.model_handoffs", lambda root, task_id: [])
    return scripted


def _tails(entries, running=(), members="census", root="."):
    from ouroboros import peer_roster

    if members == "census":  # every addressed run's task is a live member, as its queue row would make it
        members = {str(entry.task_id) for entry in entries if entry.task_id} | set(running)
    tree = {"running": list(running), "runs": entries, "members": set(members or ()),
            "starts": [("inv-unbound", True)]}
    return peer_roster._tree_tails(root, tree, {}, "")


def _codes(tails):
    return [(tail["code"], tail["state"], tail["ids"]) for tail in tails]


def test_an_observed_working_run_stays_the_live_tail(observer):
    observer["by_run"] = {"run-1": True}
    assert _codes(_tails([_entry("run-1")])) == [
        ("delegated_running", "live", ["run-1"]),
        ("external_runs_unconfirmed", "unknown", ["inv-unbound"]),
    ]


def test_a_waiting_executor_reads_open_never_working(observer):
    """The gap this repair closes: the registration said "live" while the executor only waits."""
    observer["by_run"] = {"run-1": False}
    assert _codes(_tails([_entry("run-1")])) == [
        ("external_runs_unconfirmed", "unknown", ["inv-unbound"]),
        ("external_runs_open", "open", ["run-1"]),
    ]


def test_an_unanswered_observation_stays_unconfirmed_and_never_falls_back_to_live(observer):
    """None — stale, unanswered or a refused observer read — is an honest unknown, not the old
    registered "live" (unknown is not no, and it is not yes either)."""
    observer["by_run"] = {}
    assert _codes(_tails([_entry("run-1")])) == [
        ("external_runs_unconfirmed", "unknown", ["run-1", "inv-unbound"]),
    ]
    assert observer["keys"] == [("nanny-1", "run-1", "inv-1")]


def test_one_read_speaks_each_run_and_a_fresh_fact_outranks_the_stop_typing(observer):
    observer["by_run"] = {"run-live": True, "run-wait": False, "run-gone": None}
    observer["stops"] = {"run-live"}  # its stop unconfirmed, yet the executor is observed working
    codes = _codes(_tails([_entry("run-live"), _entry("run-wait"), _entry("run-gone")], running=["helper-1"]))
    assert codes == [
        ("helpers_running", "live", ["helper-1"]),
        ("delegated_running", "live", ["run-live"]),
        ("external_runs_unconfirmed", "unknown", ["run-gone", "inv-unbound"]),
        ("external_runs_open", "open", ["run-wait"]),
    ]
    assert observer["keys"] == [("nanny-1", "run-live", "inv-1"), ("nanny-1", "run-wait", "inv-1"),
                                ("nanny-1", "run-gone", "inv-1")]


def test_review_work_and_unaddressable_runs_keep_their_own_typing(observer):
    """What the projection never asks about, the roster does not ask about either: review work keeps
    its finishing word; a run with no task is unconfirmed. Neither requests an observation."""
    tails = _tails([_entry("review-run", source="review_substrate"), _entry("orphan-run", task_id="")])
    assert _codes(tails) == [
        ("review_work_finishing", "live", ["review-run"]),
        ("external_runs_unconfirmed", "unknown", ["orphan-run", "inv-unbound"]),
    ]
    assert observer["keys"] is None, "nothing addressable, so nothing was asked"


def test_a_run_whose_task_already_answered_keeps_its_registration_typing(observer):
    """Not a live census member: the projection never asks the observer about it, and neither does
    the roster — a run that outlived its answered task keeps the card's rule ("live"), never the
    observation's absence ("unknown"). The paired control to the member cases above."""
    observer["by_run"] = {}  # even with the observer answering None for everything
    tails = _tails([_entry("r2", task_id="answered-root")], members=())
    assert _codes(tails) == [
        ("delegated_running", "live", ["r2"]),
        ("external_runs_unconfirmed", "unknown", ["inv-unbound"]),
    ]
    assert observer["keys"] is None


@pytest.fixture()
def record(tmp_path, monkeypatch):
    """The body's real dated record of a held run under ``tmp_path``: ``Feed.record``, its one writer, as of ``ago``
    seconds back (only the body's clock moves)."""
    from ouroboros import delegate_feed

    now = [0.0]
    monkeypatch.setattr(delegate_feed, "time", SimpleNamespace(
        time=lambda: now[0], strftime=time.strftime, gmtime=time.gmtime, monotonic=time.monotonic))

    def write(run_id, *, task_id="nanny-1", invocation_id="inv-1", state="running", asks=False, answered=True, ago=0):
        now[0] = time.time() - ago
        entry = SimpleNamespace(run_id=run_id, task_id=task_id, invocation_id=invocation_id)
        error = None if answered else SimpleNamespace(observation_reason="daemon_unreachable")
        delegate_feed.Feed(tmp_path).record(
            entry, {"run_state": state, "waiting_on_user": asks, "answered": answered}, error)
    return write


def test_this_process_holds_no_body_so_the_real_current_observation_answers_nothing(tmp_path):
    """The worker's premise, measured and not scripted: no body runtime lives here for this root."""
    from ouroboros.delegate_collection import body_runtime
    from ouroboros.delegate_feed import current_activities

    key = ("nanny-1", "run-1", "inv-1")
    assert body_runtime(tmp_path) is None and current_activities(tmp_path, [key]) == {key: None}


def test_a_worker_says_what_the_body_last_recorded_in_past_words_only(observer, record, tmp_path):
    """Writer -> reader -> phrase. The scripted current observation answers nothing, as a worker's own does (the test
    above); each member run then reads the body's dated record of its own task, run and invocation. A run the record
    saw working or waiting is said as LAST OBSERVED whatever the record's age (it is written only on a change); one
    it saw end, no record and an engine that was not answering say what they said before. Reading writes nothing."""
    from ouroboros import peer_roster

    record("run-new")
    record("run-old", ago=24 * 3600)
    record("run-ask", asks=True)  # paused on a blocking question: waiting
    record("run-queue", state="queued")
    record("run-done", state="succeeded")  # a finished run is not a waiting one
    record("run-silent", answered=False)  # the engine was not answering when it was written
    folder = tmp_path / "state" / "delegate_observation"
    before = {path.name: path.read_bytes() for path in folder.iterdir()}
    ids = ("run-new", "run-old", "run-ask", "run-queue", "run-done", "run-none", "run-silent")
    tails = _tails([_entry(run_id) for run_id in ids], root=tmp_path)
    assert _codes(tails) == [
        ("external_runs_last_working", "unknown", ["run-new", "run-old"]),
        ("external_runs_last_waiting", "unknown", ["run-ask", "run-queue"]),
        ("external_runs_unconfirmed", "unknown", ["run-done", "run-none", "run-silent", "inv-unbound"]),
    ]
    row = {"task_id": "root-1", "title": "Waits", "chat_id": 7, "project_id": "", "status": "waiting", "tails": tails}
    assert ("  tails: 2 external runs last observed working; 2 external runs last observed waiting; "
            "4 external runs may still be running") in peer_roster.render_roster_note({"roots": [row]}).splitlines()
    assert {path.name: path.read_bytes() for path in folder.iterdir()} == before


@pytest.mark.parametrize("field", ["task_id", "run_id", "invocation_id"])
def test_a_record_naming_another_task_run_or_invocation_says_nothing(observer, record, tmp_path, field):
    """Only the exact task, run and invocation addressed may speak: the record that says "last observed working" in
    the test above, with one of the three changed, leaves the run unconfirmed."""
    from ouroboros import delegate_feed

    record("run-1")
    path = delegate_feed.observation_path(tmp_path, "run-1")
    path.write_text(json.dumps({**json.loads(path.read_text(encoding="utf-8")), field: "other"}), encoding="utf-8")
    assert _codes(_tails([_entry("run-1")], root=tmp_path)) == [
        ("external_runs_unconfirmed", "unknown", ["run-1", "inv-unbound"])]


def test_a_record_never_lifts_a_stopped_or_failed_run_nor_outranks_a_current_observation(
        observer, record, tmp_path, monkeypatch):
    """Past words fill only the gap of a run its registration typed live: a run whose stop its Pause could not
    confirm, one whose own task failed, one the current observation answers and one outside the live members keep
    what they said before, each with a record on disk that would say otherwise."""
    monkeypatch.setattr("ouroboros.task_results.load_task_result",
                        lambda root, task_id: {"status": "failed"} if task_id == "dead-owner" else None)
    observer["by_run"], observer["stops"] = {"run-live": True}, {"run-stop"}
    for run_id, task_id in (("run-live", "nanny-1"), ("run-stop", "nanny-1"), ("run-fail", "dead-owner"),
                            ("run-away", "answered-root")):
        record(run_id, task_id=task_id, asks=run_id == "run-live")
    entries = [_entry("run-live"), _entry("run-stop"), _entry("run-fail", task_id="dead-owner"),
               _entry("run-away", task_id="answered-root")]
    assert _codes(_tails(entries, members={"nanny-1", "dead-owner"}, root=tmp_path)) == [
        ("delegated_running", "live", ["run-live", "run-away"]),
        ("external_runs_unconfirmed", "unknown", ["run-stop", "run-fail", "inv-unbound"]),
    ]
