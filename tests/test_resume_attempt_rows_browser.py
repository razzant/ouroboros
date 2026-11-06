"""#2380/#2411 in the real SPA: a refused Resume stays dated on its work's row, in Main and in a Project.

The page (index.html, app.js, room_work.js, work_status.js and their CSS) runs in Chromium and WebKit over the
controlled transport of ``test_room_work_browser``; its ``/api/state`` work fields are the server's own projection
(``gateway/work_projection.works_snapshot``) of these census rows with the latest attempt each stored record keeps,
bound to the control it answered about (``work_stop.attempt_record``). The owner's refused Resume ends its Pause's
line, dated; Ouroboros's is said as its own beside the app stop; an attempt about another Pause says nothing; a
Continue whose own start was refused after its earlier work settled says that, dated, in place of its wait, and
offers no Resume. Each row's details give the door's words. Screenshots at 1148x808 and 390x844, in both themes,
go to ``OUROBOROS_UI_EVIDENCE_DIR`` (``room-work/<engine>``), with the facts read in the page.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from tests import test_room_work_browser as room_work
from tests._work_projection_shared import work_fields
from tests.test_room_work_browser import (  # noqa: F401 - pytest fixtures (room_ui, subscription_ui)
    PROJECT, Transport, activity, capture, evidence, facts, open_app, open_project, room_ui, row, row_main,
    subscription_ui, trigger, wait_facts,
)

pytestmark = [pytest.mark.ui_browser]

_NOW = datetime.now(timezone.utc).replace(microsecond=0)
PRESSED = (_NOW - timedelta(minutes=24)).isoformat()
REFUSED = (_NOW - timedelta(minutes=9)).isoformat()
OWNER_PAUSE = {"cause": "pause", "scope": "tree", "ui_code": "owner_paused", "release": "resume"}
SAVED = {"reason": "saved_work_hold", "release": "resume", "held_at": PRESSED,
         "by": {"actor": "host", "task_id": "", "text": "", "at": PRESSED}}
WAIT = {"reason": "continuation_writer_unsettled", "release": "none", "held_at": PRESSED,
        "by": {"actor": "owner", "task_id": "", "text": "", "at": PRESSED}}
QUEUED = "task h1 is queued to run; task h2 is queued to run"


def _attempt(by, control, answer, **extra):
    return {"at": REFUSED, "answered_at": REFUSED, "by": by, "control": control, "ok": False, **extra,
            "answer": {"ok": False, **answer}}


ATTEMPTS = {
    "ra-owner": _attempt({"actor": "owner"}, {**OWNER_PAUSE, "fence_id": "fence-ra-owner"}, {
        "error": "predecessor_writers_unsettled", "message": f"Resume refused: {QUEUED}"}),
    "ra-mind": _attempt({"actor": "ouroboros", "task_id": "coordinator"}, {
        "cause": "app_stop", "scope": "member", "ui_code": "saved_work_hold", "reason": "saved_work_hold",
        "release": "resume", "held_at": PRESSED}, {
        "error": "predecessor_writers_unsettled", "message": "Resume refused: task h1 is queued to run"}),
    "ra-earlier": _attempt({"actor": "owner"}, {**OWNER_PAUSE, "fence_id": "fence-before"}, {
        "error": "selection_authority_changed", "message": "Resume refused: selection authority changed"}),
    "ra-start": _attempt({"actor": "host"}, {
        "cause": "wait", "ui_code": "continuation_writer_unsettled", "reason": "continuation_writer_unsettled",
        "release": "none", "held_at": PRESSED}, {"error": "accounting_unavailable", "action": "cancel_or_new_run",
                             "message": "Resume refused: accounting unavailable"}, selected_by="reconciliation"),
    "ra-project": _attempt({"actor": "owner"}, {**OWNER_PAUSE, "fence_id": "fence-ra-project"}, {
        "error": "predecessor_writers_unsettled", "message": "Resume refused: delegated run run-leaf is running; a "
                                                             "delegated start has an unknown outcome (no run was recorded)"}),
}
NAMES = {"ra-owner": "Compare the saved release notes", "ra-mind": "Rebuild the citation index",
         "ra-earlier": "Check the archive mirror", "ra-start": "Finish the saved release",
         "ra-project": "Collect the appendix evidence"}


def _fence(task_id):
    return {"state": "paused", "fence_id": f"fence-{task_id}", "parked_members": [task_id],
            "by": {"actor": "owner", "text": "", "at": PRESSED}, "requested_at": PRESSED}


def _transport(page):
    transport = Transport(page, projects=[PROJECT], history={1: [], PROJECT["chat_id"]: []}, activities=[
        activity("ra-owner", "budget_paused", pause_cause="owner", can_resume=False),
        activity("ra-mind", "budget_paused", pause_cause="restart", can_resume=True),
        activity("ra-earlier", "budget_paused", pause_cause="owner", can_resume=False),
        activity("ra-start", "budget_paused", pause_cause="unknown", can_resume=False),
        activity("ra-project", "budget_paused", pause_cause="owner", can_resume=False, chat_id=PROJECT["chat_id"],
                 project_id=PROJECT["id"]),
    ])
    transport.raw = {task_id: {"suggested_name": name} for task_id, name in NAMES.items()}
    transport.raw["ra-mind"]["_budget_pause_hold"] = SAVED
    transport.raw["ra-start"].update(_budget_pause_hold=WAIT, title="Continue: release",
                                     continuation={"predecessor_task_id": "ra-pred"})
    transport.results = {"ra-pred": {"task_id": "ra-pred", "status": "failed", "suggested_name": NAMES["ra-start"]}}
    transport.fences = {task_id: _fence(task_id) for task_id in ("ra-owner", "ra-earlier", "ra-project")}
    return transport


def _open_details(page, task_id, scope="main"):
    row_main(page, task_id, scope).click()
    return wait_facts(page, scope, lambda f: (row(f, task_id) or {}).get("open"), f"{task_id}'s details")


def _shots(page, engine, stem, scope="main"):
    """Both themes at the owner's window and at phone width, the list open."""
    shots = {}
    for theme in ("light", "dark"):
        page.evaluate("(t) => { localStorage.setItem('ouroboros.theme', t); document.documentElement.dataset.theme = t; }",
                      theme)
        for width, height in ((1148, 808), (390, 844)):
            page.set_viewport_size({"width": width, "height": height})
            page.wait_for_timeout(300)
            current = facts(page, scope)
            overflow = page.evaluate("""() => [...document.querySelectorAll('.rw-row')].filter(
                (r) => r.offsetParent && r.scrollWidth > r.clientWidth + 1).map((r) => r.dataset.id)""")
            assert not overflow, (theme, width, overflow)
            shots[f"{theme}-{width}"] = current
            capture(page, f"{stem}-{theme}-{width}")
    page.set_viewport_size({"width": 1148, "height": 808})
    return shots


def test_refused_resumes_and_a_refused_start_stay_dated_on_their_rows(room_ui, monkeypatch):  # noqa: F811
    monkeypatch.setattr(room_work, "work_fields",
                        lambda rows, raw=None, **reads: work_fields(rows, raw, attempts=ATTEMPTS, **reads))
    transport = _transport(room_ui["page"])
    page = open_app(room_ui, transport, theme="light", viewport={"width": 1148, "height": 808})
    clock = page.evaluate("""(at) => { const d = new Date(at);
        return d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }); }""", REFUSED)
    pressed = page.evaluate("""(at) => new Date(at).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })""",
                            PRESSED)
    trigger(page).click()
    opened = wait_facts(page, "main", lambda f: len(f["rows"]) == 4 and all(item["name"] for item in f["rows"]),
                        "four named rows in Main")
    by = {item["id"]: item for item in opened["rows"]}
    assert (by["ra-owner"]["word"], by["ra-owner"]["secondary"]) == (
        "Paused", f"You paused it · {pressed} · Resume refused at {clock}"), by["ra-owner"]
    assert (by["ra-mind"]["word"], by["ra-mind"]["secondary"]) == (
        "Paused", f"Saved when the app stopped · Resume by Ouroboros refused at {clock}"), by["ra-mind"]
    assert (by["ra-earlier"]["word"], by["ra-earlier"]["secondary"]) == ("Paused", f"You paused it · {pressed}"), \
        "an attempt about another Pause says nothing here"
    # The Continue's row is its work's, which its predecessor's id names.
    assert (by["ra-pred"]["word"], by["ra-pred"]["secondary"]) == (
        "Waiting", f"Continue tries to start again by itself · Start refused at {clock}"), by["ra-pred"]
    assert by["ra-owner"]["resume"] and by["ra-mind"]["resume"], "the stops keep their own Resume"
    assert not by["ra-pred"]["resume"], "a Continue that tries again by itself offers no Resume"
    details = {}
    for work_id in ("ra-owner", "ra-mind", "ra-pred"):
        details[work_id] = row(_open_details(page, work_id), work_id)["detail"]
    assert f"Last ResumeResume refused at {clock}: {QUEUED}" in details["ra-owner"], details
    assert f"Last ResumeResume by Ouroboros refused at {clock}: task h1 is queued to run" in details["ra-mind"], details
    assert f"Last startStart refused at {clock}: accounting unavailable" in details["ra-pred"], details
    main_shots = _shots(page, room_ui["engine"], "resume-attempts-main")
    page.keyboard.press("Escape")
    wait_facts(page, "main", lambda f: f["surfaceHidden"], "Main's list closed")
    open_project(page, PROJECT)
    trigger(page, "project").click()
    project = wait_facts(page, "project", lambda f: row(f, "ra-project") and row(f, "ra-project")["name"],
                         "the Project's row")
    assert row(project, "ra-project")["secondary"] == f"You paused it · {pressed} · Resume refused at {clock}", project
    detail = row(_open_details(page, "ra-project", "project"), "ra-project")["detail"]
    assert f"Last ResumeResume refused at {clock}: delegated run run-leaf is running" in detail, detail
    project_shots = _shots(page, room_ui["engine"], "resume-attempts-project", "project")
    assert transport.posts == [], "reading the rows presses nothing"
    evidence("resume-attempt-rows", {"main": opened, "details": details, "project": project, "project_detail": detail,
                                     "main_shots": main_shots, "project_shots": project_shots, "clock": clock})
