"""The browser reads the server's real work projection: one shared fixture for both sides.

``web/tests/fixtures/work_projection_cases.json`` holds the ``/api/state`` work fields
(``works``, ``rooms``, ``work_summary``, ``works_complete``, ``works_gaps``) that
``gateway/work_projection.works_snapshot`` builds for the rooms below, and every reason
sentence the server can send (a counted one at 1 and at 3). This test rebuilds both from the
server and fails on any difference, so the web tests (``web/tests/work_status.test.js`` and
the chat/room fixtures) never test a shape or a sentence the server does not send.
Regenerate after an intended projection change by running this module as a script from the
repository root (``python -m tests.test_work_projection_web_fixture``).
"""
from __future__ import annotations

import json
import pathlib

import pytest

from ouroboros.gateway import work_status
from tests._work_projection_shared import work_fields

FIXTURE = pathlib.Path(__file__).resolve().parents[1] / "web" / "tests" / "fixtures" / "work_projection_cases.json"


def _row(task_id, phase="working", **extra):
    return {"activity_id": task_id, "phase": phase, "kind": extra.pop("kind", "managed_task"),
            "chat_id": extra.pop("chat_id", 42), "task_attempt": 1, "started_at": extra.pop("started_at", 100.0), **extra}


_HOLD = {"_budget_pause_hold": {"reason": "continuation_writer_unsettled"}}
_RUNNING = {"running": True}  # the census's RUNNING fact (gateway/state.py): the member executes now

# name -> (census owner rows, raw queue facts, stored task results, held delegated runs)
CASES = {
    # A held Continue waits for three helpers of its failed earlier attempt.
    "held_continue": (
        [_row("c", "budget_paused", pause_cause="unknown", can_resume=False),
         *[_row(f"h{index}", parent_task_id="p", root_task_id="p") for index in range(3)]],
        {"c": {**_HOLD, "title": "Continue: release", "continuation": {"predecessor_task_id": "p"}}},
        {"p": {"task_id": "p", "status": "failed", "suggested_name": "Ouroboros 7.7.0 — finish the saved release"}},
        {},
    ),
    "working_call": (
        [_row("w", model_activity={"phase": "awaiting_response"}, suggested_name="Compare the cited pages")],
        {"w": {"suggested_name": "Compare the cited pages"}}, {}, {},
    ),
    "direct_question": (
        [_row("d", kind="direct_chat", chat_id=1, client_message_id="cm-ask",
              owner_wait={"quiz_id": "q1", "owner_wait_state": "waiting", "quiz_state": "open"})],
        {}, {}, {},
    ),
    # A direct turn still preparing says the start step it stands at beside Preparing.
    "direct_preparing": (
        [_row("dp", "preparing", kind="direct_chat", chat_id=1, client_message_id="cm-prep",
              turn_phase="context_building")],
        {}, {}, {},
    ),
    # The same turn held by a conversation-lane carrier, the ordinary path since #2654: the lane row reads `working`
    # once carried, and its frame-fed start step still says Preparing · Building context.
    "carried_preparing": (
        [_row("cp", "working", kind="direct_chat", chat_id=1, client_message_id="cm-carried", carrier="worker",
              turn_phase="context_building")],
        {}, {}, {},
    ),
    # Its own Main call started before its census phase moved: the word that move gives (Working), no start step.
    "direct_requested": (
        [_row("dr", "preparing", kind="direct_chat", chat_id=1, client_message_id="cm-call", turn_phase="requested")],
        {}, {}, {},
    ),
    "paused_by_owner": (
        [_row("o", "budget_paused", pause_cause="owner", can_resume=False)],
        {"o": {"_budget_pause": {"reason": "owner", "paused_at": "2026-10-17T15:00:00Z"}}}, {}, {},
    ),
    "paused_by_mind": (
        [_row("m", "budget_paused", pause_cause="owner", can_resume=False)],
        {"m": {"_budget_pause": {"reason": "owner"}}}, {}, {},
    ),
    "spending_limit": (
        [_row("s", "budget_paused", pause_cause="budget", can_resume=True)],
        {"s": {"_budget_pause": {"reason": "budget"}}}, {}, {},
    ),
    "model_access": (
        [_row("a", model_waits={"w1": {"state": "waiting", "task_attempt": 1, "revision": 1,
                                       "requires_owner_action": True}})],
        {}, {}, {},
    ),
    "queued_and_pausing": (
        [_row("q", "queued", started_at=90.0), _row("z", "budget_pausing", pause_cause="owner")],
        {"z": _RUNNING}, {}, {},
    ),
    # The owner's Pause still settles: the root and one helper finish their step; a helper that never started is
    # queued, not working, so only the running one is said.
    "pausing_with_queued_helper": (
        [_row("pr", "budget_pausing", pause_cause="owner"),
         *[_row(task, "budget_pausing", pause_cause="owner", parent_task_id="pr", root_task_id="pr")
           for task in ("pc-run", "pc-new")]],
        {"pr": _RUNNING, "pc-run": _RUNNING}, {}, {},
    ),
    # Two works in one room: the room says Working for the one that runs, never the newer one's Pause reason.
    "two_works_one_pausing": (
        [_row("wb", suggested_name="Write the notes"), _row("wa", "budget_pausing", pause_cause="owner", started_at=200.0)],
        {"wb": _RUNNING, "wa": _RUNNING}, {}, {},
    ),
    # A running task's own review is its own work: no tail beside it.
    "working_with_review": (
        [_row("wr")], {"wr": _RUNNING}, {}, {"wr": [{"run_id": "run-own-review", "source": "review_substrate"}]},
    ),
    # The owner's Pause could not confirm the stop of the task's own run: Paused, with that run said unconfirmed.
    "paused_stop_unknown": (
        [_row("so", "budget_paused", pause_cause="owner", can_resume=False)], {}, {}, {"so": ["run-s"]},
    ),
    "finalizing_failed": (
        [_row("f", "finalizing", status="failed", outcome_axes={"lifecycle": {"status": "failed"}})],
        {}, {}, {},
    ),
    "sleeping": (
        [_row("sl", "budget_paused", pause_cause="sleep")],
        {"sl": {"_budget_pause": {"reason": "sleep"}}}, {}, {},
    ),
    # After a root-only Resume the root sleeps warm on the helper it chose, which still awaits its own selection:
    # the root keeps its word and controls; its line names that helper and who resumes it. No owner's turn.
    "sleeping_on_paused_helper": (
        [_row("sw", suggested_name="Repair the release notes"),
         _row("sw-c", "budget_paused", pause_cause="unknown", can_resume=False, parent_task_id="sw", root_task_id="sw")],
        {"sw": {**_RUNNING, "owner_wait": {"state": "waiting", "reason": "sleep", "task_attempt": 1,
                                            "sleep": {"mode": "warm", "tasks": ["sw-c"], "senders": ["sw-c"]}}},
         "sw-c": {"_budget_pause": {"reason": "owner", "exact_continuation": True}}},
        {}, {},
    ),
    # A warm sleep says what it waits for by the census's own name of the helper it chose, never its id.
    "sleeping_on_named_helper": (
        [_row("sn", suggested_name="Publish the release"),
         _row("sn-c", parent_task_id="sn", root_task_id="sn", suggested_name="Draft the release notes")],
        {"sn": {**_RUNNING, "owner_wait": {"state": "waiting", "reason": "sleep", "task_attempt": 1,
                                            "sleep": {"mode": "warm", "tasks": ["sn-c"], "senders": ["sn-c"]}}},
         "sn-c": {**_RUNNING, "suggested_name": "Draft the release notes"}},
        {}, {},
    ),
    # A cold sleep that chose only a time says it; the browser reads the instant in the viewer's clock.
    "sleeping_until_time": (
        [_row("st", "budget_paused", pause_cause="sleep")],
        {"st": {"_budget_pause": {"reason": "sleep", "sleep": {"mode": "cold", "wake_at": "2026-10-17T18:05:00+00:00",
                                                               "any_mail": True}}}}, {}, {},
    ),
    "unknown": ([_row("u", "unknown")], {}, {}, {}),
    "ended_with_runs": ([], {}, {"r": {"task_id": "r", "status": "done", "chat_id": 42}}, {"r": ["run-1", "run-2"]}),
    # The parent finished while its helper still works: its episode is not live, its tree's Stop still stands.
    "helper_after_parent": (
        [_row("h", parent_task_id="p", root_task_id="p")], {},
        {"p": {"task_id": "p", "status": "completed", "chat_id": 42}}, {},
    ),
    # The 20.10 card: the owner's Pause settled while a review it lets finish still runs. Paused, never Working.
    "paused_review_finishing": (
        [_row("rv", "budget_paused", pause_cause="owner", can_resume=False, finishing_reviews=True)], {}, {},
        {"rv": [{"run_id": "run-review", "source": "review_substrate"}]},
    ),
    # A run whose own task failed and that nobody can confirm: unconfirmed, listed with its Stop.
    "unconfirmed_run": (
        [], {}, {"x": {"task_id": "x", "status": "failed", "chat_id": 42}, "x-h": {"task_id": "x-h", "status": "failed"}},
        {"x": [{"run_id": "run-u", "task_id": "x-h"}]},
    ),
    # A Pause pressed through the API with no declared author names nobody.
    "paused_unattributed": (
        [_row("n", "budget_paused", pause_cause="owner", can_resume=False)], {}, {}, {},
    ),
    "paused_model_handoff": (
        [_row("mh", "budget_pausing", pause_cause="owner", live=False)], {}, {}, {},
    ),
    # The owner's Resume of a work they paused was refused after the press: the line keeps who paused it and says
    # that refusal last, dated; its words go to the card's details. Its Resume control stays the door's own.
    "paused_resume_refused": (
        [_row("rr", "budget_paused", pause_cause="owner", can_resume=False)], {}, {}, {},
    ),
    # Saved when the app stopped, then Ouroboros's own Resume was refused: said as Ouroboros's, never the owner's.
    "saved_resume_refused": (
        [_row("sr", "budget_paused", pause_cause="restart", can_resume=True)],
        {"sr": {"_budget_pause_hold": {"reason": "saved_work_hold", "release": "resume", "held_at": "2026-10-17T14:00:00Z",
                                       "by": {"actor": "host", "task_id": "", "text": "", "at": "2026-10-17T14:00:00Z"}}}},
        {}, {},
    ),
    # A refusal about an earlier Pause (another fence) belongs to that stop: nothing is said of it.
    "paused_resume_refused_earlier": (
        [_row("re", "budget_paused", pause_cause="owner", can_resume=False)], {}, {}, {},
    ),
    # A Continue whose earlier work settled, its own start refused for another reason: that refusal, dated, in place
    # of the wait's sentence; it still waits and tries again by itself (no Resume).
    "continue_start_refused": (
        [_row("cs", "budget_paused", pause_cause="unknown", can_resume=False)],
        {"cs": {**_HOLD, "title": "Continue: release", "continuation": {"predecessor_task_id": "ps"}}},
        {"ps": {"task_id": "ps", "status": "failed", "suggested_name": "Finish the saved release"}}, {},
    ),
}

# The owner fence records who declared the Pause: the owner's own press, or Ouroboros with its sentence (here a
# fence written before ``by`` existed). An undeclared press ("n") names nobody.
_PRESSED = "2026-10-17T15:00:00Z"
FENCES = {"o": {"by": {"actor": "owner", "text": "", "at": _PRESSED}, "requested_at": _PRESSED},
          "m": {"requested_by": "owner", "actor": "mind", "actor_task_id": "mind-1",
                "reason": "pause before the update you asked for", "requested_at": _PRESSED},
          "z": {"by": {"actor": "owner", "text": "", "at": _PRESSED}, "requested_at": _PRESSED},
          "rv": {"by": {"actor": "owner", "text": "", "at": _PRESSED}, "requested_at": _PRESSED},
          "n": {"requested_by": "owner", "requested_at": _PRESSED},
          **{task: {"by": {"actor": "owner", "text": "", "at": _PRESSED}, "requested_at": _PRESSED,
                    **({"fence_id": f"fence-{task}"} if task in {"rr", "re"} else {})}
             for task in ("pr", "wa", "so", "mh", "rr", "re")}}
# The runs whose stop a task's live Pause could not confirm (its pause row's ``external_runs``).
STOPS = {"so": ["run-s"]}
# The latest Resume attempt each stored record keeps (``work_stop.attempt_record``), as its door wrote it: bound to
# the control it answered about (``work_stop.control_identity``), the door's whole answer kept.
_REFUSED_AT = "2026-10-17T15:05:00Z"
_QUEUED = [{"kind": "dispatchable_member", "task_id": "h1"}, {"kind": "dispatchable_member", "task_id": "h2"}]
_OWNER_PAUSE = {"cause": "pause", "scope": "tree", "ui_code": "owner_paused", "release": "resume"}


def _attempt(by, control, answer, at=_REFUSED_AT, **extra):
    return {"at": at, "answered_at": at, "by": by, "control": control, "ok": False, **extra,
            "answer": {"ok": False, **answer}}


ATTEMPTS = {
    "rr": _attempt({"actor": "owner"}, {**_OWNER_PAUSE, "fence_id": "fence-rr"}, {
        "error": "predecessor_writers_unsettled", "action": "wait_or_stop_the_previous_work", "blockers": _QUEUED,
        "message": "Resume refused: task h1 is queued to run; task h2 is queued to run"}),
    "sr": _attempt({"actor": "ouroboros", "task_id": "coordinator"}, {
        "cause": "app_stop", "scope": "member", "ui_code": "saved_work_hold", "reason": "saved_work_hold",
        "release": "resume", "held_at": "2026-10-17T14:00:00Z"}, {
        "error": "predecessor_writers_unsettled", "action": "wait_or_stop_the_previous_work",
                               "blockers": _QUEUED[:1], "message": "Resume refused: task h1 is queued to run"}),
    "re": _attempt({"actor": "owner"}, {**_OWNER_PAUSE, "fence_id": "fence-re-earlier"}, {
        "error": "selection_authority_changed", "message": "Resume refused: selection authority changed"},
        at="2026-10-17T14:55:00Z"),
    "cs": _attempt({"actor": "host"}, {
        "cause": "wait", "ui_code": "continuation_writer_unsettled",
        "reason": "continuation_writer_unsettled", "release": "none"}, {
        "error": "accounting_unavailable", "action": "cancel_or_new_run",
        "message": "Resume refused: accounting unavailable"}, selected_by="reconciliation"),
}


def project(name):
    """The ``/api/state`` work fields the server builds for one case."""
    owners, raw, stored, held = CASES[name]
    return work_fields(owners, raw, results=stored, fences={k: {"state": "paused", **v} for k, v in FENCES.items()},
                       held=held, stops=STOPS, models={"mh": [{"kind": "model_handoff", "attempt_id": "attempt-mh"}]},
                       attempts=ATTEMPTS)


def reason_sentences():
    """Each reason code's English sentence as the server words it; a counted code at 1 and at 3."""
    return {code: ({"1": work_status.reason(code, count=1)["text"], "3": work_status.reason(code, count=3)["text"]}
                   if isinstance(phrase, tuple) else {"": work_status.reason(code)["text"]})
            for code, phrase in work_status.REASON_TEXT.items()}


# One saved sleep per clause the server says (gateway/work_status.py SLEEP_CLAUSES), named by a census name where it has
# one, and sleeps that chose several sources: each is said, joined by "or", its time last.
_AT = "2026-10-17T18:05:00+00:00"
SLEEPS = {
    "task": {"tasks": ["named"]}, "task_unnamed": {"tasks": ["elsewhere"]}, "tasks": {"tasks": ["a", "b", "c"]},
    "all_tasks": {"tasks": ["a", "b"], "task_wait": {"mode": "all_terminal"}},
    "run": {"runs": ["r1"]}, "runs": {"runs": ["r1", "r2", "r3"]}, "service": {"services": [{"name": "preview"}]},
    "reviewer": {"reviewer_slots": ["s1"], "any_mail": False}, "sender": {"senders": ["named"]},
    "sender_unnamed": {"senders": ["elsewhere"]}, "mail": {"any_mail": True}, "time_only": {"wake_at": _AT},
    "time": {"wake_at": _AT, "any_mail": True}, "task_and_time": {"tasks": ["named"], "wake_at": _AT},
    "task_and_sender": {"tasks": ["named"], "senders": ["named"]},
    "task_and_sender_and_time": {"tasks": ["named"], "senders": ["named"], "wake_at": _AT},
    "every_kind": {"tasks": ["a", "b"], "task_wait": {"mode": "all_terminal"}, "runs": ["r1"],
                   "services": [{"name": "preview"}], "reviewer_slots": ["s1"], "senders": ["named"], "wake_at": _AT},
    "unrecorded": {},
}


def sleep_sentences():
    """Each kind of sleep as the server says it: its reason with ``wake`` and English sentence."""
    return {name: work_status.sleep_reason(sleep, {"named": "Draft the release notes"}) for name, sleep in SLEEPS.items()}


def test_the_web_fixture_is_the_servers_projection():
    built = {name: project(name) for name in CASES}
    recorded = json.loads(FIXTURE.read_text(encoding="utf-8"))
    assert sorted(recorded["cases"]) == sorted(built)
    for name in CASES:
        assert recorded["cases"][name] == built[name], name
    assert recorded["reasons"] == reason_sentences()
    assert recorded["sleeps"] == sleep_sentences()


@pytest.mark.parametrize("name, activity", [
    ("held_continue", "working"), ("direct_question", "waiting"), ("paused_by_owner", "paused"),
])
def test_each_fixture_room_is_the_state_its_name_says(name, activity):
    """The fixture is meaningful, not merely equal: each room is the case the web tests read it as."""
    assert [room["activity"] for room in project(name)["rooms"].values()] == [activity]


if __name__ == "__main__":  # regenerate the fixture after an intended projection change
    FIXTURE.write_text(json.dumps({"_note": "Generated by tests/test_work_projection_web_fixture.py (run it as a script).",
                                   "cases": {name: project(name) for name in CASES},
                                   "reasons": reason_sentences(), "sleeps": sleep_sentences()},
                                  ensure_ascii=False, indent=1) + "\n",
                       encoding="utf-8")
