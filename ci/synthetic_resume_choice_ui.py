"""The approved B flow in one real Chromium, on C's card: ended predecessor → held Continue → addressed Resume.

Scratch CI module for the successor-start repair (A+B+C). The overlay workflow copies this
file into `tests/` only for a run through `scripts/safe_test.py`; it is never part of the
tested checkout. Nothing here touches a real installation: the server binds an ephemeral
loopback socket on the fixture's own drive root, the engine is a counting stand-in, and
every address comes from B's fixture — all synthetic.

What this scenario proves, and what it deliberately does NOT.

 * PROVED HERE (browser, end to end, on the REAL projected card and the REAL gateway door):
   POST `/api/tasks/<successor>/resume` answering 409 whose `acceptable_unknown` names the
   exact read-only `delegate_start:…` key of the predecessor's unconfirmed start; the real
   task-control menu rendering C's singular dialog ("Resume without this result?") with a
   Cancel that sends nothing further and leaves the successor held; a recheck press asking
   again with the SAME choice and read-only wording although a historical risk record now
   exists; a confirm ("Resume anyway") whose 200 releases the SAME never-started successor id
   exactly once, queues that same id, records the decision on that successor, and is refused
   when pressed again. The synthetic case is set up the way B's own merged tests set it up —
   `_held_continue(..., before=_own_earlier_choice)` — so the changed historical risk record
   is part of what the UI must still call read-only.
 * PROVED BY THE PYTHON JOB, not by this file: the whole B module runs `events_mind_controls`'
   list refusal and the twelve `test_without_positive_no_write_proof…` parameters; C's whole
   `test_work_status.py` / `test_work_projection_pause_producers.py` and the two Node suites
   hold the card/menu/Stop geometry; A's definite pre-admission rejection lives in A's modules.
   The merged real browser case `tests/test_held_continue_browser.py::
   test_held_continue_toast_activity_reason_and_resume_check` runs in the same job and is the
   geometry/toast witness — this file does not duplicate its checks.
 * Product interfaces this fixture expects:
   `tests/test_successor_start_choice.py` exposes PRED/LOST/STORE, `_held_continue(root,
   monkeypatch, *, head=None, lost=None, before=None)` and `_own_earlier_choice(root)`;
   `delegate_resume` builds the blocker `{**row, "choice_id": …, "read_only": True}`;
   `gateway/tasks.py::api_task_resume` passes error/blockers/acceptable_unknown/
   accepted_start_risks through a JSONResponse (409 unless task_not_pending);
   `web/modules/task_control_menu.js` keys on `kind === 'delegated_run' && b.choice_id`,
   reads `b.read_only === true`, and with exactly one start titles the dialog
   "Resume without this result?", says "The earlier request was read-only, without permission
   to change files" and "Resuming will not send this request again.", confirmLabel
   "Resume anyway"; `web/modules/api_client.js` POSTs `{accept_unknown, reason}`.
   If the final merged source drifts from any of this, the scenario fails naming what it
   could not find — never vacuously.
"""
import hashlib
import json
import os
import re
import socket
import threading
import time
from pathlib import Path
from urllib.parse import quote

import pytest


def pytest_sessionstart(session):
    """CI overlay launcher policy only; target product test sources are not edited."""
    from functools import wraps
    from playwright.sync_api import BrowserType

    original = BrowserType.launch
    session._successor_browser_launch = (BrowserType, original)

    @wraps(original)
    def launch(self, *args, **kwargs):
        if self.name == "chromium":
            kwargs["chromium_sandbox"] = True
        return original(self, *args, **kwargs)

    BrowserType.launch = launch


def pytest_sessionfinish(session, exitstatus):
    browser_type, original = session._successor_browser_launch
    browser_type.launch = original


@pytest.mark.ui_browser
def test_addressed_resume_of_a_never_started_successor(tmp_path, monkeypatch):
    import uvicorn
    from playwright.sync_api import expect, sync_playwright
    from starlette.applications import Starlette
    from starlette.responses import HTMLResponse, JSONResponse
    from starlette.routing import Mount, Route
    from starlette.staticfiles import StaticFiles

    from ouroboros import delegate_custody as dc
    from ouroboros import delegate_resume as dr
    from ouroboros.gateway.tasks import api_task_resume
    from ouroboros.outcomes import public_task_result
    from ouroboros.task_results import load_task_result
    from ouroboros.work_stop import hold_fact
    from tests._budget_pause_exact_helpers import _install_queue  # noqa: F401 — the queue seam _held_continue installs
    from tests._owner_pause_ended_helpers import _Engine
    from tests._work_projection_shared import state_fields
    from tests.test_activity_outcomes_browser import HTML, WEB
    from tests.test_resume_attempt_history import _work
    from tests.test_successor_start_choice import (
        LOST, PRED, STORE, _held_continue, _own_earlier_choice)

    assert os.environ.get("OUROBOROS_PYTEST_ACTIVE") == "1", "safe_test owns this run"
    evidence_dir = os.environ.get("OUROBOROS_UI_EVIDENCE_DIR")
    assert evidence_dir, "the launcher must declare OUROBOROS_UI_EVIDENCE_DIR before the scenario runs"
    evidence = Path(evidence_dir).resolve()
    evidence.mkdir(exist_ok=True, parents=True)

    # ── the synthetic case, exactly as the merged B suite builds it: its own earlier choice first ──
    queue, workers, successor = _held_continue(tmp_path, monkeypatch, before=_own_earlier_choice)
    held = hold_fact(next(row for row in workers.PENDING if row["id"] == successor)) or {}
    blockers = held.get("blockers") or []
    assert held.get("predecessor_task_id") == PRED and any(
        b.get("kind") == "delegated_run" and b.get("invocation_id") == LOST for b in blockers
    ), ("the held Continue does not name the fixture's lost start", held)

    # The engine as a loud counter: nothing in this lifecycle may start or cancel a run.
    from ouroboros.gateways import claudexor as gw
    engine = _Engine()
    monkeypatch.setattr(gw, "ClaudexorGateway", engine)

    queue.persist_queue_snapshot(reason="fixture for a real browser request")
    pred_before = json.dumps(load_task_result(tmp_path, PRED), sort_keys=True, default=str)
    before_row = dc.invocation_record(tmp_path, LOST)
    assert (before_row["state"], before_row["daemon_store_id"], before_row["handshake_store_id"],
            before_row["store_identity_unproven"]) == ("pending", "", STORE, True), before_row

    # What the merged projection offers before any press: one addressed Resume on the successor
    # card, activity waiting, and no pause control over a wait (C's rule, via the product helper).
    work = _work(tmp_path, successor)
    assert (work["activity"], work["because"][0]["code"]) == ("waiting", "earlier_work_unconfirmed"), \
        (work["activity"], work["because"])
    assert {"action": "resume", "target": successor} in work["controls"], work["controls"]
    assert not any(control["action"] == "pause" for control in work["controls"]), \
        "the projection pauses a wait — C's rule broke before the browser"
    fields = state_fields(tmp_path, direct_turns=[])
    offered = [c for w in fields.get("works", []) for c in (w.get("controls") or [])
               if c.get("action") == "resume" and c.get("target") == successor]
    assert offered, ("the work projection offers no addressed Resume for the held successor; it offered "
                     + json.dumps([[w.get("work_id"), w.get("controls")] for w in fields.get("works", [])]))

    posts, pressed = [], []

    async def activity(request):
        return HTMLResponse(HTML)

    async def state(request):
        return JSONResponse({"supervisor_ready": True, "bg_consciousness_enabled": False,
                             "projects": [], "project_chat_ids": [],
                             **state_fields(tmp_path, direct_turns=[]), "active_chat_activities_complete": True})

    async def tasks(request):
        from ouroboros.gateway.tasks import _queue_snapshot
        return JSONResponse({"tasks": [], "queue": _queue_snapshot(tmp_path)})

    async def detail(request):
        r = load_task_result(tmp_path, request.path_params["task_id"])
        return JSONResponse(public_task_result(r) if r else {}, status_code=200 if r else 404)

    async def schedules(request):
        return JSONResponse({"tasks": []})

    async def resume(request):
        sent = await request.body()
        result = await api_task_resume(request)   # the real owner door, not a mocked reply
        posts.append({"task": request.path_params["task_id"], "press": pressed[-1],
                      "sent": json.loads(sent) if sent else None,
                      "status": result.status_code, "reply": json.loads(result.body)})
        return result

    app = Starlette(routes=[
        Route("/fixture", activity), Route("/api/state", state), Route("/api/tasks", tasks),
        Route("/api/tasks/{task_id}/resume", resume, methods=["POST"]),
        Route("/api/tasks/{task_id}", detail), Route("/api/schedules", schedules),
        Mount("/", app=StaticFiles(directory=WEB))])
    sock = socket.socket(); sock.bind(("127.0.0.1", 0)); sock.listen(16)
    origin = f"http://127.0.0.1:{sock.getsockname()[1]}"
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=0, ws="none",
                                           lifespan="off", log_level="warning", access_log=False))
    worker = threading.Thread(target=server.run, kwargs={"sockets": [sock]}, daemon=True)
    worker.start()
    page_errors, repeat, choice, words = [], None, "", ""
    try:
        for _ in range(100):
            if server.started or not worker.is_alive():
                break
            time.sleep(.05)
        assert server.started
        with sync_playwright() as pw:
            browser = pw.chromium.launch(headless=True, chromium_sandbox=True)
            try:
                page = browser.new_page(viewport={"width": 1100, "height": 750}, reduced_motion="reduce")
                page.route("**/*", lambda route: route.continue_()
                           if route.request.url.startswith(origin + "/") else route.abort())
                page.on("pageerror", lambda error: page_errors.append(str(error)))
                page.goto(origin + "/fixture?theme=light")
                page.wait_for_function("window.ready === true")

                def open_dialog():
                    # The trigger is the ACTUAL projected Resume on the successor's own card:
                    # the ENDED predecessor (with its recorded stop and earlier choice) must not
                    # be the press target. C renders one title per count — this case has one.
                    trigger = page.locator(f'[data-act="task-control"][data-id="{successor}"]')
                    expect(trigger).to_be_visible()
                    trigger.click()
                    page.get_by_role("menuitem", name="Resume", exact=True).click()
                    d = page.get_by_role("dialog").filter(
                        has=page.get_by_text(re.compile(r"Resume without th(is|ese) re(sult|sults)\?")))
                    expect(d).to_be_visible()
                    return d

                # 1 · a bare press must refuse, naming only the one read-only start of the
                #     predecessor, addressed to this successor — and release nothing.
                pressed.append("bare")
                dialog = open_dialog()
                words = dialog.inner_text()
                page.screenshot(path=str(evidence / "successor-resume-ask.png"), animations="disabled")
                (evidence / "successor-resume-dialog.txt").write_text(words, encoding="utf-8")
                refused = posts[-1]
                assert refused["status"] == 409, refused
                assert refused["reply"].get("error") == "predecessor_writers_unsettled", refused["reply"]
                offered_keys = refused["reply"].get("acceptable_unknown") or []
                start_rows = [b for b in (refused["reply"].get("blockers") or [])
                              if b.get("kind") == "delegated_run" and b.get("choice_id")]
                choice = offered_keys[0] if len(offered_keys) == 1 else ""
                choice_targets = (len(offered_keys) == 1 and len(start_rows) == 1
                                  and start_rows[0].get("read_only") is True
                                  and start_rows[0]["choice_id"] == offered_keys[0]
                                  and PRED in offered_keys[0] and LOST in offered_keys[0]
                                  and quote(successor, safe="") in offered_keys[0])
                # Custody proved this start read-only: C says it in the SINGULAR for one start
                # and never claims file changes; the earlier recorded choice did not change that.
                readOnly_words = ("read-only" in words and "changing files" not in words
                                  and "The earlier request was read-only" in words
                                  and "Resuming will not send this request again." in words)
                posts_after_cancel_probe = len(posts)
                dialog.get_by_role("button", name="Cancel", exact=True).click()
                expect(dialog).to_have_count(0)
                page.wait_for_function(
                    "async id => !(await import('/modules/task_control_menu.js')).resumePending(id)", arg=successor)
                cancel_sent_nothing = len(posts) == posts_after_cancel_probe
                held_after_cancel = hold_fact(next(i for i in workers.PENDING if i["id"] == successor)) is not None
                assert cancel_sent_nothing, "Cancel sent another POST"
                assert held_after_cancel, "Cancel released the writer hold"

                # 2 · the recheck press: the door counts afresh — same choice, still read-only
                #     with the earlier choice still in history — and still releases nothing.
                pressed.append("recheck")
                dialog = open_dialog()
                recheck = posts[-1]
                assert recheck["status"] == 409, recheck
                recheck_same_choice = recheck["reply"].get("acceptable_unknown") == [choice]
                recheck_read_only = all(
                    b.get("read_only") is True for b in (recheck["reply"].get("blockers") or [])
                    if b.get("kind") == "delegated_run" and b.get("choice_id"))
                assert recheck_same_choice and recheck_read_only, recheck["reply"]

                # 3 · the confirmed press: same successor, released once, decision recorded on it.
                pressed.append("confirm")
                with page.expect_response(lambda r: r.url.endswith(f"/api/tasks/{successor}/resume")
                                          and r.request.method == "POST") as ack:
                    dialog.get_by_role("button", name="Resume anyway", exact=True).click()
                assert ack.value.status == 200, (ack.value.status, ack.value.body())
                page.get_by_text("Resuming: the task returns to the queue.", exact=True).wait_for()
                ok = posts[-1]
                risks = ok["reply"].get("accepted_start_risks") or []
                assert len(risks) == 1, ok["reply"]
                risk = risks[0]
                released_same_successor = (ok["task"] == successor
                                           and risk.get("resuming_task_id") == successor
                                           and risk.get("successor_of") == PRED
                                           and risk.get("creation") == "lookup_only"
                                           and not any(k in risk for k in ("daemon_store_id",
                                                                           "handshake_store_id")))
                assert ok["sent"].get("reason", "").strip(), ok["sent"]
                page.screenshot(path=str(evidence / "successor-resume-confirmed.png"), animations="disabled")

                # 4 · the same decision pressed again must refuse: the release was one-shot.
                pressed.append("repeat")
                repeat = page.evaluate(
                    """async ([task, body]) => (await fetch(`/api/tasks/${task}/resume`, {
                        method: 'POST', headers: {'content-type': 'application/json'},
                        body: JSON.stringify(body)})).status""",
                    [successor, {"accept_unknown": [choice], "reason": "the same decision again"}])
            finally:
                browser.close()
    finally:
        server.should_exit = True
        worker.join(10)
        sock.close()
        (evidence / "successor-resume-http.json").write_text(
            json.dumps({"posts": posts, "page_errors": page_errors}, indent=2), encoding="utf-8")

    assert not worker.is_alive() and not page_errors, page_errors

    # ── durable truths, read from the fixture's own drive root ────────────────────────────
    saved = load_task_result(tmp_path, successor) or {}
    acceptance_recorded = (saved.get("accepted_predecessor_starts") or {}) == {
        LOST: {"root_task_id": PRED, "task_id": PRED, "invocation_id": LOST}}
    same_rows = [i for i in workers.PENDING if i["id"] == successor]
    same_id_queued_once = len(same_rows) == 1 and hold_fact(same_rows[0]) is None
    row_released = same_id_queued_once
    pred_after = json.dumps(load_task_result(tmp_path, PRED), sort_keys=True, default=str)
    after = dc.invocation_record(tmp_path, LOST)
    unknown_kept = (after["state"], after["daemon_store_id"], after["handshake_store_id"],
                    after["store_identity_unproven"]) == ("pending", "", STORE, True)
    lookup_only = False
    try:
        with dr._admit_start(tmp_path, PRED, LOST):
            pass
    except dr.StartHeld as held:
        lookup_only = held.args[0] == dr.LOOKUP_ONLY

    dialog_posts = [p for p in posts if p["press"] != "repeat"]
    dialog_presses = [p["press"] for p in dialog_posts]
    statuses = [p["status"] for p in dialog_posts]
    repeat_statuses = [p["status"] for p in posts if p["press"] == "repeat"]
    facts = {
        "flow": "ended-predecessor -> held Continue -> addressed successor Resume",
        "predecessor_id": PRED,
        "expected_successor_id": successor,          # from admit_continuation, not from the press
        "resumed_task_id": saved.get("task_id"),     # no fallback: absent saves as None and fails
        "saved_row_released": row_released,
        "same_id_queued_once": same_id_queued_once,  # the release queues the SAME successor row
        "dialog_presses": dialog_presses,
        "statuses": statuses,                        # 409 (bare) · 409 (recheck) · 200 (confirm)
        "repeat_statuses": repeat_statuses,
        "posts_after_cancel": posts_after_cancel_probe,
        "recheck_same_choice": bool(recheck_same_choice),
        "recheck_read_only": bool(recheck_read_only),
        "repeat_status": repeat,
        "choice_targets_lost_start": bool(choice_targets),
        "refusal_read_only_words": bool(readOnly_words),
        "cancel_left_it_held": bool(held_after_cancel),
        "released_same_successor": bool(released_same_successor),
        "acceptance_recorded_on_successor": bool(acceptance_recorded),
        "unknown_start_kept_lookup_only": bool(unknown_kept and lookup_only),
        "grant_uses": statuses.count(200),
        "engine_starts_delta": len(engine.starts),
        "engine_cancels_delta": len(engine.cancels),
        # The dialog may SAY money may be spending; no number may be presented as known, and
        # nothing here reads provider balances — pending stays pending, money stays unclaimed.
        "money_amount_claimed": bool(re.search(r"\$\s?\d|\d+(\.\d+)?\s*(USD|usd|\$)", words)),
        "predecessor_unchanged": pred_before == pred_after,
        "predecessor_sha256": hashlib.sha256(pred_before.encode()).hexdigest()[:16],
        "page_errors": page_errors,
        "rows": {LOST: after},
    }
    (evidence / "successor-resume-facts.json").write_text(json.dumps(facts, indent=2), encoding="utf-8")

    # The scenario's own gates; the CI prove step re-reads the same file independently.
    assert dialog_presses == ["bare", "recheck", "confirm"] and statuses == [409, 409, 200], (dialog_presses, statuses)
    assert repeat in (409, 404) and repeat_statuses == [repeat], (repeat, repeat_statuses)
    assert facts["expected_successor_id"] == facts["resumed_task_id"] == successor, facts
    assert facts["saved_row_released"] and facts["same_id_queued_once"], "the release is not the same queue row"
    assert facts["posts_after_cancel"] == 1, "Cancel triggered another request"
    assert facts["recheck_same_choice"] and facts["recheck_read_only"]
    assert facts["choice_targets_lost_start"] and facts["refusal_read_only_words"]
    assert facts["cancel_left_it_held"] and facts["released_same_successor"]
    assert facts["acceptance_recorded_on_successor"] and facts["unknown_start_kept_lookup_only"]
    assert facts["engine_starts_delta"] == 0 and facts["engine_cancels_delta"] == 0
    assert not facts["money_amount_claimed"], "the flow presented a money figure as known"
    assert facts["predecessor_unchanged"]
