"""The neighbour roster names each recorded wait by what its record says it is.

Both carriers are read: the owner-wait row (a question, a review park, or the
model's warm sleep) and the exact-pause row (a cold or restart-retained sleep,
the owner's Pause, or money). The records are written by the real producers
where one exists -- ``owner_wait.checkpoint_owner_wait`` and
``restart_retention.retain_sleep_checkpoint`` -- so the roster is checked
against the shapes the runtime actually stores, and the model-visible note and
the paginated ``live_roots`` JSON must agree.
"""

from __future__ import annotations

from tests._shared import write_queue_document

import types

from ouroboros.task_results import STATUS_RUNNING, write_task_result
from ouroboros.utils import utc_now_iso

_PAUSED_AT = 1790409600.0  # 2026-09-26T08:00:00+00:00
_PAUSED_ISO = "2026-09-26T08:00:00+00:00"
_WAKE_AT = "2026-09-26T09:30:00+00:00"


def _snapshot(root, running=(), pending=()):
    write_queue_document(root / "state" / "queue_snapshot.json", {
        "ts": utc_now_iso(),
        "running": [{"id": task_id, "attempt": 1, "task": {"id": task_id, "title": task_id, "_attempt": 1}}
                    for task_id in running],
        "pending": [{"id": task_id, "attempt": 1, "task": {"id": task_id, "title": task_id, "_attempt": 1,
                                                            "_budget_pause": {"exact_continuation": True}}}
                    for task_id in pending],
    })


def _waits(root):
    from ouroboros.peer_roster import independent_roots
    return {row["task_id"]: row.get("waiting") for row in independent_roots(root)["roots"]}


def _note(root):
    from ouroboros.peer_roster import independent_roots, render_roster_note
    return render_roster_note(independent_roots(root))


def _sleep(mode):
    """The selected sources as ``model_sleep.request_sleep`` arms them (a service pin included)."""
    return {"sleep_id": f"s-{mode}", "mode": mode, "senders": ["sender-a"], "tasks": ["child-1", "child-2"],
            "runs": ["run-9"], "wake_at": _WAKE_AT, "any_mail": False,
            "services": [{"name": "api", "service_id": "warm:api", "started_at": 1790400000.0,
                          "pid": 4242, "pgid": 4242}]}


def _park(root, task_id, *, sleep=None, quiz="", review_binding=""):
    """Write a waiting owner-wait row through the real checkpoint producer."""
    from ouroboros.owner_wait import checkpoint_owner_wait, set_owner_wait

    write_task_result(root, task_id, STATUS_RUNNING, started_at="2026-09-26T07:00:00+00:00")
    ctx = types.SimpleNamespace(
        task_id=task_id, task_attempt=1, drive_root=root, budget_drive_root=str(root),
        task_started_at=1790406000.0, _model_sleep=sleep, _model_sleep_started=_PAUSED_AT,
        _owner_wait_requested=(f"sleep:{sleep['sleep_id']}" if sleep else quiz),
        _owner_wait_deadline_at=(sleep or {}).get("wake_at", ""),
    )
    checkpoint = checkpoint_owner_wait(ctx, [], {}, {}, 3, [], set(), review_binding=review_binding)
    return set_owner_wait(root, task_id, {**checkpoint, "state": "waiting"})


def test_a_warm_sleep_is_a_sleep_with_its_recorded_wake_sources_not_an_owner_answer(tmp_path):
    from ouroboros.peer_roster import live_root_catalogue

    stored = _park(tmp_path, "sleeper", sleep=_sleep("warm"))
    assert stored["reason"] == "sleep" and stored["quiz_id"] == ""  # the producer's own shape
    _park(tmp_path, "asker", quiz="q7")
    _park(tmp_path, "reviewed", review_binding="plan-review-1")
    _snapshot(tmp_path, running=["sleeper", "asker", "reviewed"])

    waits = _waits(tmp_path)
    sleep = waits["sleeper"][0]
    assert sleep["kind"] == "sleep" and sleep["mode"] == "warm"
    assert sleep["senders"] == ["sender-a"] and sleep["tasks"] == ["child-1", "child-2"] and sleep["runs"] == ["run-9"]
    assert sleep["wake_at"] == _WAKE_AT and sleep["services"] == ["api"] and "any_mail" not in sleep
    assert "quiz_id" not in sleep and "until" not in sleep and sleep["since"]
    assert waits["asker"][0]["kind"] == "owner" and waits["asker"][0]["quiz_id"] == "q7"
    assert waits["reviewed"][0]["kind"] == "review" and "quiz_id" not in waits["reviewed"][0]

    note = _note(tmp_path)
    sleeper_line = next(line for line in note.splitlines() if line.startswith("  waiting") and "sleep (" in line)
    assert ("sleep (mode=warm, wake_at=2026-09-26T09:30:00+00:00, senders=sender-a, tasks=child-1,child-2, "
            "runs=run-9, services=api, since=") in sleeper_line
    assert "owner answer" not in sleeper_line
    assert "owner answer (quiz_id=q7, since=" in note and "review (since=unknown)" in note
    # Service pins are execution custody, not a wake condition: no pid anywhere.
    assert "4242" not in note
    page = live_root_catalogue(tmp_path, limit=10)
    json_waits = {row["task_id"]: row["waiting"] for row in page["roots"]}
    assert json_waits == waits and "4242" not in str(json_waits)


def test_legacy_and_unrecognized_owner_waits_keep_their_evidence(tmp_path):
    """A reasonless row with a quiz predates the field and stays an owner question;
    a reasonless row without one is unknown; a stated reason the roster does not
    know is unknown WITH that reason, and its quiz id never turns it into owner."""
    rows = {
        "legacy-quiz": {"quiz_id": "q0"},
        "legacy-bare": {"quiz_id": ""},
        "blank-reason": {"quiz_id": "", "reason": "  "},
        "future": {"quiz_id": "q9", "reason": "handoff", "wait_deadline_at": _WAKE_AT},
    }
    for task_id, extra in rows.items():
        write_task_result(tmp_path, task_id, STATUS_RUNNING, owner_wait={
            "wait_id": f"w-{task_id}", "state": "waiting", "task_attempt": 1, **extra})
    _snapshot(tmp_path, running=list(rows))

    waits = _waits(tmp_path)
    assert waits["legacy-quiz"] == [{"kind": "owner", "since": None, "quiz_id": "q0"}]
    assert waits["legacy-bare"] == [{"kind": "unknown", "since": None}]
    assert waits["blank-reason"] == [{"kind": "unknown", "since": None}]
    assert waits["future"] == [{"kind": "unknown", "since": None, "reason": "handoff", "quiz_id": "q9",
                                "until": _WAKE_AT}]
    note = _note(tmp_path)
    assert "owner answer (quiz_id=q0, since=unknown)" in note
    assert "unknown wait (since=unknown)" in note
    assert f"unknown wait (quiz_id=q9, reason=handoff, until={_WAKE_AT}, since=unknown)" in note


def test_the_pause_carrier_tells_sleep_owner_pause_and_money_apart(tmp_path):
    """A cold sleep, the owner's Pause and a monetary rail share one carrier; a
    reasonless (pre-field) row stays money and a newer reason is never money."""
    pauses = {
        "cold": {"reason": "sleep", "rail": "model_sleep", "sleep": _sleep("cold")},
        "paused-by-owner": {"reason": "owner", "rail": "owner_pause", "settlement": "external_running"},
        "money": {"reason": "budget", "rail": "global_exhausted"},
        "legacy-money": {"rail": "graceful_ceiling"},
        "future": {"reason": "maintenance", "rail": "maintenance_window"},
    }
    for task_id, extra in pauses.items():
        write_task_result(tmp_path, task_id, "scheduled", reason_code="budget_paused", budget_pause={
            "pause_id": f"p-{task_id}", "state": "paused", "task_attempt": 1, "paused_at": _PAUSED_AT, **extra})
    _snapshot(tmp_path, pending=list(pauses))

    waits = _waits(tmp_path)
    cold = waits["cold"][0]
    assert cold["kind"] == "sleep" and cold["mode"] == "cold" and cold["state"] == "paused"
    assert cold["services"] == ["api"] and "rail" not in cold and "retained" not in cold
    assert waits["paused-by-owner"][0]["reason"] == "awaiting_selection"  # no closed tree fence (R3)
    assert waits["paused-by-owner"][0]["release"] == "resume"
    for task in ("money", "legacy-money"):
        assert waits[task][0]["reason"] == "spending_limit"
        assert waits[task][0]["kind"] == "budget" and waits[task][0]["since"] == _PAUSED_ISO
        assert waits[task][0]["rail"] == pauses[task]["rail"]
    assert waits["future"][0]["reason"] == "restore_failed" and waits["future"][0]["release"] == "resume"
    note = _note(tmp_path)
    assert "sleep (mode=cold, state=paused, wake_at=" in note
    assert "Waiting for Ouroboros or you to resume it" in note
    assert "Spending limit reached" in note
    assert "Its saved state could not be restored" in note


def test_a_parked_pause_keeps_the_author_its_tree_fence_declared(tmp_path):
    """The exact pause a member parks as names no author; its tree's fence does, read as the owner's card reads it
    (``control_actor.pause_author``): the owner's press, Ouroboros with its own sentence (a fence written before
    the declaration existed keeps its mind record), and a press that declared nobody names nobody."""
    fences = {
        "pressed": {"by": {"actor": "owner", "text": "", "at": _PAUSED_ISO}},
        # An agent's press declared for the owner: the head is the declared author, then its recorded words.
        "declared": {"by": {"actor": "owner", "text": "Anton asked to stop the export", "at": _PAUSED_ISO}},
        "mind": {"by": {"actor": "ouroboros", "text": "pausing for the update", "at": _PAUSED_ISO}},
        "legacy-mind": {"actor": "mind", "reason": "before the restart"},
        "anonymous": {},  # requested_by=owner alone is the door, not a declaration
    }
    for task_id, declared in fences.items():
        write_task_result(tmp_path, task_id, "scheduled", reason_code="budget_paused", budget_pause={
            "pause_id": f"p-{task_id}", "state": "paused", "task_attempt": 1, "paused_at": _PAUSED_AT,
            "reason": "owner", "rail": "owner_pause"}, owner_pause={
            "state": "paused", "fence_id": f"f-{task_id}", "requested_by": "owner", "requested_at": _PAUSED_ISO,
            **declared})
    _snapshot(tmp_path, pending=list(fences))

    waits = _waits(tmp_path)
    assert [{k: v for k, v in w.items() if k not in {"reason", "text", "release"}} for w in waits["pressed"]] == [{"kind": "owner_pause", "state": "paused", "by": "owner", "since": _PAUSED_ISO}]
    assert waits["declared"][0]["by"] == "owner: Anton asked to stop the export"
    assert waits["mind"][0]["by"] == "Ouroboros: pausing for the update"
    assert waits["legacy-mind"][0]["by"] == "Ouroboros: before the restart"
    assert [{k: v for k, v in w.items() if k not in {"reason", "text", "release"}} for w in waits["anonymous"]] == [{"kind": "owner_pause", "state": "paused", "since": _PAUSED_ISO}]
    note = _note(tmp_path)
    assert "You paused it" in note and "Ouroboros paused it" in note
    assert "by=Ouroboros: pausing for the update" in note and "release=resume" in note
    assert all(w[0]["release"] == "resume" for w in waits.values())
    from ouroboros.peer_roster import live_root_catalogue
    assert {row["task_id"]: row["waiting"] for row in live_root_catalogue(tmp_path, limit=10)["roots"]} == waits


def test_a_restart_retained_warm_sleep_keeps_its_warm_mode_and_says_it_was_retained(tmp_path):
    """The real retention moves a warm sleep into the exact pause: the roster reads
    the saved mode (warm, never inferred cold from the carrier), marks it retained
    (its stack ended with the stop) and no longer lists the retained owner wait."""
    from supervisor.restart_retention import retain_sleep_checkpoint

    _park(tmp_path, "kept", sleep=_sleep("warm"))
    retain_sleep_checkpoint(tmp_path, "kept", 1)
    _snapshot(tmp_path, pending=["kept"])

    waits = _waits(tmp_path)["kept"]
    assert len(waits) == 1
    kept = waits[0]
    assert kept["kind"] == "sleep" and kept["mode"] == "warm" and kept["retained"] is True
    assert kept["state"] == "paused" and kept["tasks"] == ["child-1", "child-2"] and kept["since"] == _PAUSED_ISO
    assert "sleep (mode=warm, retained=true, state=paused, wake_at=" in _note(tmp_path)


def test_a_queued_root_names_the_other_rooms_work_that_keeps_its_folder(tmp_path):
    """Owner 15.11.2026, one queue per recorded folder: the chooser's ``_folder_wait`` reaches the roster the mind
    reads, naming the holder and its room; a running root and a queued one without it say nothing of a folder."""
    from ouroboros.peer_roster import independent_roots, roster_fingerprint

    wait = {"task_id": "holder", "project_id": "room-a", "folder": "/work/shared"}
    write_queue_document(tmp_path / "state" / "queue_snapshot.json", {
        "ts": utc_now_iso(),
        "running": [{"id": "holder", "attempt": 1, "task": {"id": "holder", "title": "Weekly review", "_attempt": 1,
                                                            "project_id": "room-a"}}],
        "pending": [{"id": "waiting", "attempt": 1, "task": {"id": "waiting", "title": "Reading notes", "_attempt": 1,
                                                             "project_id": "room-b", "_folder_wait": wait}},
                    {"id": "free", "attempt": 1, "task": {"id": "free", "title": "Other", "_attempt": 1}}],
    })
    rows = {row["task_id"]: row for row in independent_roots(tmp_path)["roots"]}
    assert rows["waiting"]["folder_wait"] == wait and "folder_wait" not in rows["free"] | rows["holder"]
    note = _note(tmp_path)
    assert '  folder_wait={"folder": "/work/shared", "project_id": "room-a", "task_id": "holder"}' in note
    roster = independent_roots(tmp_path)
    freed = {**roster, "roots": [{**row, "folder_wait": {}} for row in roster["roots"]]}
    assert roster_fingerprint(roster) != roster_fingerprint(freed), "a change of holder is a roster change"


def test_a_held_or_paused_queued_root_is_not_said_to_wait_for_a_folder(tmp_path):
    """Opus 15.11.2026: the chooser notes ``_folder_wait`` only on a pass that reaches the row's lane, after its hold
    and pause gates (``worker_assignment._candidate_ineligible``), and the snapshot keeps that note. A row paused,
    asleep or held since then waits for that, not for a holder that may long have finished: the roster names its
    hold and no folder. The same row, eligible again, names its folder wait as the chooser last noted it."""
    from ouroboros.peer_roster import independent_roots

    wait = {"task_id": "holder", "project_id": "room-a", "folder": "/work/shared"}
    write_task_result(tmp_path, "paused", "scheduled", reason_code="budget_paused", budget_pause={
        "pause_id": "p-paused", "state": "paused", "task_attempt": 1, "paused_at": _PAUSED_AT,
        "reason": "owner", "rail": "owner_pause"}, owner_pause={
        "state": "paused", "fence_id": "f-paused", "requested_by": "owner", "requested_at": _PAUSED_ISO})
    write_task_result(tmp_path, "asleep", "scheduled", reason_code="budget_paused", budget_pause={
        "pause_id": "p-asleep", "state": "paused", "task_attempt": 1, "paused_at": _PAUSED_AT,
        "reason": "sleep", "rail": "model_sleep", "sleep": _sleep("cold")})
    held = {"reason": "project_routing_fence_lookup_failed", "detail": "The task result is unreadable."}
    markers = {"eligible": {}, "paused": {"_budget_pause": {"exact_continuation": True}},
               "asleep": {"_budget_pause": {"exact_continuation": True}},
               "held": {"_project_admission_restore_hold": held}, "owner-held": {"_owner_hold": {"reason": "owner"}}}

    def publish(marked):
        write_queue_document(tmp_path / "state" / "queue_snapshot.json", {
            "ts": utc_now_iso(),
            "running": [{"id": "holder", "attempt": 1, "task": {"id": "holder", "title": "Weekly review", "_attempt": 1,
                                                                "project_id": "room-a"}}],
            "pending": [{"id": task_id, "attempt": 1, "task": {"id": task_id, "title": task_id, "_attempt": 1,
                                                               "project_id": "room-b", "_folder_wait": wait, **extra}}
                        for task_id, extra in marked.items()],
        })
        return {row["task_id"]: row for row in independent_roots(tmp_path)["roots"]}

    rows = publish(markers)
    assert rows["eligible"]["folder_wait"] == wait
    for task_id in ("paused", "asleep", "held", "owner-held"):
        assert "folder_wait" not in rows[task_id], task_id
    assert rows["paused"]["waiting"][0]["kind"] == "owner_pause" and rows["asleep"]["waiting"][0]["kind"] == "sleep"
    assert rows["held"]["project_admission_hold"]["reason"] == held["reason"]
    assert rows["owner-held"]["refusal"] == {"reason": "_owner_hold"}
    assert _note(tmp_path).count("folder_wait=") == 1
    # The hold lifted, the same row is eligible again: its noted folder wait is said once more.
    assert publish({"eligible": {}, "held": {}})["held"]["folder_wait"] == wait
