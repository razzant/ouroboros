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
          **{task: {"by": {"actor": "owner", "text": "", "at": _PRESSED}, "requested_at": _PRESSED}
             for task in ("pr", "wa", "so", "mh")}}
# The runs whose stop a task's live Pause could not confirm (its pause row's ``external_runs``).
STOPS = {"so": ["run-s"]}


def project(name):
    """The ``/api/state`` work fields the server builds for one case."""
    owners, raw, stored, held = CASES[name]
    return work_fields(owners, raw, results=stored, fences={k: {"state": "paused", **v} for k, v in FENCES.items()},
                       held=held, stops=STOPS, models={"mh": [{"kind": "model_handoff", "attempt_id": "attempt-mh"}]})


def reason_sentences():
    """Each reason code's English sentence as the server words it; a counted code at 1 and at 3."""
    return {code: ({"1": work_status.reason(code, count=1)["text"], "3": work_status.reason(code, count=3)["text"]}
                   if isinstance(phrase, tuple) else {"": work_status.reason(code)["text"]})
            for code, phrase in work_status.REASON_TEXT.items()}


def test_the_web_fixture_is_the_servers_projection():
    built = {name: project(name) for name in CASES}
    recorded = json.loads(FIXTURE.read_text(encoding="utf-8"))
    assert sorted(recorded["cases"]) == sorted(built)
    for name in CASES:
        assert recorded["cases"][name] == built[name], name
    assert recorded["reasons"] == reason_sentences()


@pytest.mark.parametrize("name, activity", [
    ("held_continue", "working"), ("direct_question", "waiting"), ("direct_preparing", "working"),
    ("direct_requested", "working"), ("paused_by_owner", "paused"),
    ("spending_limit", "paused"), ("queued_and_pausing", "working"), ("unknown", "unknown"),
    ("helper_after_parent", "working"), ("paused_review_finishing", "paused"), ("unconfirmed_run", "unknown"),
    ("paused_unattributed", "paused"), ("pausing_with_queued_helper", "working"), ("two_works_one_pausing", "working"),
    ("working_with_review", "working"), ("paused_stop_unknown", "paused"),
    ("paused_model_handoff", "paused"), ("sleeping_on_paused_helper", "waiting"),
])
def test_each_fixture_room_is_the_state_its_name_says(name, activity):
    """The fixture is meaningful, not merely equal: each room is the case the web tests read it as."""
    assert [room["activity"] for room in project(name)["rooms"].values()] == [activity]


if __name__ == "__main__":  # regenerate the fixture after an intended projection change
    FIXTURE.write_text(json.dumps({"_note": "Generated by tests/test_work_projection_web_fixture.py (run it as a script).",
                                   "cases": {name: project(name) for name in CASES},
                                   "reasons": reason_sentences()}, ensure_ascii=False, indent=1) + "\n",
                       encoding="utf-8")
