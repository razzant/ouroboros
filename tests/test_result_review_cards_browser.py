"""Result-review cards in the served app (owner, 8 Nov 2026; docs/DESIGN.md "Status marks").

Four delivered answers carry review and objective axes made by the production result-review binder
(``review_result._bind_review_run`` / ``_acceptance`` / ``_finish_pending``) and outcome readers
(``outcomes._review_axis`` / ``_objective_axis``). The author's informed finish over a FAIL and an explicit
finish before the reviewers answered read Done with warnings and say the author's own reason. The author's
acceptance of an earlier approved version and a clean PASS read Done; the earlier one also says which version
was approved. Done with warnings keeps the green mark and says the warning in words, with no recolour. The
check runs at the owner's 1100x750 desktop size in both engines: live in the light theme, then after a reload in
light and in dark; Settings › Behavior says the same rules in both themes. Screenshots go to
``OUROBOROS_BROWSER_EVIDENCE_OUT``.
"""
import re
from types import SimpleNamespace

import pytest

from tests.settings_browser_helpers import open_settings, setting_row
from tests._shared import producer_evidence_dir
from tests.test_ui_smoke_playwright import direct_server_with_data as direct_server_with_data  # noqa: F401
from tests.ui_console_errors import PageErrors

pytestmark = [pytest.mark.browser]

LONG_REASON = ("The owner asked for the corrected answer before the evening call; the fix is verified by the focused "
               "tests and by reading the produced file back from its retained copy, and the one remaining finding is "
               "about wording that the answer already explains in its second paragraph.")
PENDING_REASON = ("The owner needs this report for a call that starts now; the reviewers' verdict is read when it "
                  "arrives and any finding goes to the owner as a follow-up.")
EARLIER_REASON = "Only the wording of the summary changed after the approved version."
WITHOUT = "Ouroboros delivered this answer without the reviewers' sign-off"
BEFORE = "Ouroboros delivered this answer before the reviewers answered; their verdict is kept when it arrives"
EARLIER = "The reviewers approved an earlier version of this answer; the current version was not re-reviewed."
# name: (headline, words the expanded card says, words it must not say)
CASES = {
    "author-fail": ("Done with warnings", [WITHOUT, LONG_REASON], [EARLIER]),
    "author-pending": ("Done with warnings", [BEFORE, PENDING_REASON], [WITHOUT]),
    "earlier-pass": ("Done", [EARLIER], [WITHOUT, BEFORE]),
    "clean-pass": ("Done", [], [WITHOUT, BEFORE, EARLIER]),
}


def _record(name, aggregate, reviewed):
    seat = {"seat_id": "s1", "operation_id": f"op-{name}", "status": "complete",
            "outcome": {"outcome_id": f"out-{name}", "status": "complete"}}
    return {"record_id": f"rec-{name}", "state": "complete", "subject": {"diff_sha": reviewed},
            "verdict": {"aggregate": aggregate, "reason": "", "critical_findings": []}, "rows": [seat]}


def _axes(name):
    """The outcome axes the delivery step writes for each case, through its own producers."""
    from ouroboros.outcomes import _objective_axis, _review_axis
    from ouroboros.tools.review_result import _acceptance, _bind_review_run, _finish_pending

    trace = {}
    ctx = SimpleNamespace(_completion_selected={"rationale": {"author-fail": LONG_REASON, "author-pending": PENDING_REASON,
                                                              "earlier-pass": EARLIER_REASON}.get(name, "")},
                          task_metadata={}, task_id=f"card-{name}")
    if name == "author-pending":
        _finish_pending(ctx, trace, f"rec-{name}", SimpleNamespace(diff_sha="current"))
    elif name == "author-fail":
        record = _record(name, "FAIL", "current")
        _bind_review_run(trace, record, "current", "advisory")["informed_finish"] = _acceptance(ctx, "current", record)
    elif name == "earlier-pass":
        record = _record(name, "PASS", "earlier")
        run = _bind_review_run(trace, record, "current", "advisory", earlier_revision=True)
        run["informed_finish"] = _acceptance(ctx, "current", record)
    else:
        _bind_review_run(trace, _record(name, "PASS", "current"), "current", "advisory")
    review = _review_axis(trace)
    return {"execution": {"status": "ok"}, "review": review, "objective": _objective_axis(review)}


MARK_JS = """(card) => {
    const probe = document.createElement('span');
    document.body.append(probe);
    const ink = (token) => { probe.style.color = `var(${token})`; return getComputedStyle(probe).color; };
    const tokens = { ok: ink('--status-ok-fg'), green: ink('--green') };
    probe.remove();
    const phase = card.querySelector('[data-live-phase]');
    const mark = phase.querySelector('.status-mark');
    return { tokens, tone: phase.dataset.chipTone, hue: mark?.dataset.hue, shape: mark?.dataset.shape,
             color: getComputedStyle(phase).color, markColor: mark ? getComputedStyle(mark).color : '',
             overflow: card.scrollWidth - card.clientWidth };
}"""


@pytest.mark.parametrize("engine", ["webkit", "chromium"])
def test_result_review_cards_say_the_authors_acceptance_in_both_themes(direct_server_with_data, tmp_path, engine):
    from playwright.sync_api import expect, sync_playwright
    from ouroboros import dialogue_store
    from ouroboros.task_results import write_task_result
    from ouroboros.terminal_projection import _project_row
    from supervisor.message_bus import log_chat
    from tests.ui_chat_viewport_smoke import (
        _CAPTURE_TEST_SOCKET, _OBSERVE_STATE_READS, _emit_ws_frame, _wait_socket_open_quiescent,
    )

    fixture = direct_server_with_data
    root = fixture["data_dir"]
    evidence = producer_evidence_dir(tmp_path / "screenshots", "result-review-cards", env="OUROBOROS_BROWSER_EVIDENCE_OUT")
    evidence.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as pw:
        browser = getattr(pw, engine).launch()
        try:
            seeded = False
            for theme in ("light", "dark"):
                context = browser.new_context(viewport={"width": 1100, "height": 750}, color_scheme=theme,
                                              reduced_motion="reduce")
                page = context.new_page()
                errors = PageErrors(page, engine)
                page.add_init_script(f"({_CAPTURE_TEST_SOCKET})(); ({_OBSERVE_STATE_READS})();")
                page.goto(fixture["url"], wait_until="domcontentloaded")
                _wait_socket_open_quiescent(page)
                if not seeded:  # the live frames once; the second theme reads the same history back
                    for name in CASES:
                        tid = f"card-{name}"
                        answer = f"The delivered answer for {name}."
                        result = write_task_result(root, tid, "completed", chat_id=1, root_task_id=tid,
                            delegation_role="root", result=answer, terminal_origin="model_final",
                            reason_code="final_message", outcome_axes=_axes(name))
                        summary = _project_row(tid, result, {}, {"chat_id": 1})
                        progress = {"task_id": tid, "chat_id": 1, "content": f"Checking {name}", "ts": summary["ts"],
                                    "narration": True}
                        dialogue_store.append(root, "progress", progress)
                        _emit_ws_frame(page, {**progress, "type": "chat", "role": "assistant", "is_progress": True})
                        log_chat("out", 1, 1, answer, task_id=tid, drive_root=root)
                        _emit_ws_frame(page, {"type": "chat", "role": "assistant", "chat_id": 1, "task_id": tid,
                                             "content": answer})
                        dialogue_store.append(root, "chat", summary)
                        _emit_ws_frame(page, {**summary, "type": "chat", "role": "system",
                                             "system_type": "task_summary", "content": summary["text"]})
                    seeded = True
                visits = ("live", "reloaded") if theme == "light" else ("reloaded",)
                for visit in visits:
                    if visit == "reloaded":
                        page.reload(wait_until="domcontentloaded")
                    for name, (headline, says, never) in CASES.items():
                        card = page.locator(f'.chat-live-card[data-task-id="card-{name}"]')
                        expect(card.locator('[data-live-phase]')).to_have_text(headline, timeout=30000)
                        expect(card).to_have_attribute("data-finished", "1")
                        if card.get_attribute("data-expanded") != "1":
                            card.locator('[data-live-disclosure]').click()
                        for words in says:
                            expect(card).to_contain_text(words)
                        text = card.inner_text()
                        assert not [words for words in never if words in text], (name, text)
                        facts = card.evaluate(MARK_JS)
                        # One outcome family: green dot and ok words for Done and Done with warnings alike.
                        assert (facts["tone"], facts["hue"], facts["shape"]) == ("ok", "ok", "dot"), (name, facts)
                        assert facts["color"] == facts["tokens"]["ok"], (name, facts)
                        assert facts["markColor"] == facts["tokens"]["green"], (name, facts)
                        assert facts["overflow"] <= 1, (name, facts)
                        # Clear of the floating header and composer, which a bottom-edge scroll leaves over it.
                        card.evaluate("el => el.scrollIntoView({block: 'center'})")
                        card.screenshot(path=str(evidence / f"{engine}-{theme}-{visit}-{name}.png"), animations="disabled")
                    page.screenshot(path=str(evidence / f"{engine}-{theme}-{visit}.png"), animations="disabled")
                assert not errors.unexpected(), errors
                context.close()
        finally:
            browser.close()


RECEIPT = ("No recorded check or result review examined this result after its last change; "
           "this does not establish whether other checks ran.")
# name: (independent warning at the end, late verdict, headline after, words the collapsed line never says after)
LATE = {
    "late-pass": ("", "PASS", "Done", BEFORE),
    # Nothing examined the result is a detail: the headline is Done and the line still says it (9 Nov 2026).
    "late-pass-receipt": ("receipt_absent", "PASS", "Done", BEFORE),
    "late-fail": ("", "FAIL", "Done with warnings", RECEIPT),
    # A newer state (a custody debt recorded after the late PASS) is shown first; the late frame, published after it,
    # still carries the older revision.
    "late-stale": ("", "PASS", "Done with warnings", BEFORE),
}


def _pending_axes(name, warning):
    """An explicit finish before the reviewers answered (``_finish_pending``), with the warning ``apply_receipt_absent_flag``
    adds when the turn had effects and no host verification."""
    from ouroboros.outcomes import _merge_objective_warning, _objective_axis, _review_axis
    from ouroboros.tools.review_result import _finish_pending

    trace, ctx = {}, SimpleNamespace(_completion_selected={"rationale": PENDING_REASON}, task_metadata={},
                                     task_id=f"card-{name}")
    _finish_pending(ctx, trace, f"rec-{name}", SimpleNamespace(diff_sha="current"))
    review = _review_axis(trace)
    objective = _objective_axis(review)
    if warning:
        _merge_objective_warning(objective, warning)
    return {"execution": {"status": "ok"}, "review": review, "objective": objective}


@pytest.mark.parametrize("engine", ["webkit", "chromium"])
def test_a_late_pass_repaints_the_open_card_and_its_reload_reads_the_same(direct_server_with_data, tmp_path, monkeypatch,
                                                                         engine):
    """Owner, 8 Nov 2026: Ouroboros finished before its reviewers answered, so each card reads Done with warnings. The
    late verdict then settles: the real resolver (``resolve_pending_result_review``) writes the stored outcome and the
    existing ``review_late_result`` event carries its CURRENT outcome, delivered here as ``bridge.push_log`` frames it
    (``address_task_event``; the route itself: tests/test_log_forwarding.py). The open card repaints only on a PASS of
    exactly the delivered answer and drops only the warning that wait caused: an independent ``receipt_absent`` stays
    as a detail (Done, its sentence still in the line),
    a late FAIL keeps its warning, a replay changes nothing, a seat's result without an outcome touches no card, a task
    with no card here gets none, and a late outcome older than a newer state the card shows never paints over it, even
    published after it: its revision orders it, not its later ``ts`` (the card's clock; Astra, 10 Nov 2026). A reload
    reads the same outcome and line from the stored row; the ending row stays as written."""
    import datetime
    import json
    import time

    from playwright.sync_api import expect, sync_playwright
    from ouroboros import dialogue_store, review_ledger
    from ouroboros.late_review_marks import resolve_pending_result_review
    from ouroboros.task_results import load_task_result, write_task_result
    from ouroboros.terminal_projection import _project_row, terminal_projection_owed
    from supervisor.log_addressing import address_task_event
    from supervisor.message_bus import log_chat
    from tests.ui_chat_viewport_smoke import (
        _CAPTURE_TEST_SOCKET, _OBSERVE_STATE_READS, _emit_ws_frame, _wait_socket_open_quiescent,
    )

    root = direct_server_with_data["data_dir"]
    evidence = producer_evidence_dir(tmp_path / "screenshots", "result-review-late", env="OUROBOROS_BROWSER_EVIDENCE_OUT")
    evidence.mkdir(parents=True, exist_ok=True)
    verdicts = {f"rec-{name}": verdict for name, (_warning, verdict, *_rest) in LATE.items()}
    monkeypatch.setattr(review_ledger, "load_record", lambda _root, record_id: {
        "record_id": record_id, "subject": {"diff_sha": "current"}, "verdict": {"aggregate": verdicts[record_id]},
        "ts": "2026-11-08T23:00:00+00:00"})

    def look(page, name, headline, step):
        card = page.locator(f'.chat-live-card[data-task-id="card-{name}"]')
        expect(card.locator('[data-live-phase]')).to_have_text(headline, timeout=30000)
        card.evaluate("el => el.scrollIntoView({block: 'center'})")
        card.screenshot(path=str(evidence / f"{engine}-{step}-{name}.png"), animations="disabled")
        return card.locator('[data-live-activity]').inner_text()

    def log_frame(page, event):  # what bridge.push_log broadcasts for one addressed event
        addressed = address_task_event(None, root, dict(event))
        _emit_ws_frame(page, {"type": "log", "data": addressed, "chat_id": int(addressed.get("chat_id") or 0)})

    reads = {"open": 0, "at": 0.0}

    def watch_history(page):  # a card that first finishes schedules its own history read (scheduleHistorySync)
        def note(request, step):
            if "/api/chat/history" in request.url:
                reads.update(open=reads["open"] + step, at=time.monotonic())
        page.on("request", lambda request: note(request, 1))
        page.on("requestfinished", lambda request: note(request, -1))
        page.on("requestfailed", lambda request: note(request, -1))

    def settle_history(page):  # those reads land before the scenario's frames, never between them
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline and (reads["open"] or time.monotonic() - reads["at"] < 1.5):
            page.wait_for_timeout(250)
        assert not reads["open"], "a history read stayed open"

    lines = {}
    with sync_playwright() as pw:
        browser = getattr(pw, engine).launch()
        try:
            for theme in ("light", "dark"):
                context = browser.new_context(viewport={"width": 1100, "height": 750}, color_scheme=theme,
                                              reduced_motion="reduce")
                page = context.new_page()
                errors = PageErrors(page, engine)
                page.add_init_script(f"({_CAPTURE_TEST_SOCKET})(); ({_OBSERVE_STATE_READS})();")
                watch_history(page)
                page.goto(direct_server_with_data["url"], wait_until="domcontentloaded")
                _wait_socket_open_quiescent(page)
                if theme == "light":  # the live story once; the dark theme reads it back after the fact
                    for name, (warning, *_rest) in LATE.items():
                        tid, answer = f"card-{name}", f"The delivered answer for {name}."
                        result = write_task_result(root, tid, "completed", chat_id=1, root_task_id=tid,
                            delegation_role="root", result=answer, terminal_origin="model_final",
                            reason_code="final_message", outcome_axes=_pending_axes(name, warning))
                        summary = _project_row(tid, result, {}, {"chat_id": 1})
                        progress = {"task_id": tid, "chat_id": 1, "content": f"Checking {name}", "ts": summary["ts"],
                                    "narration": True}
                        dialogue_store.append(root, "progress", progress)
                        _emit_ws_frame(page, {**progress, "type": "chat", "role": "assistant", "is_progress": True})
                        log_chat("out", 1, 1, answer, task_id=tid, drive_root=root)
                        _emit_ws_frame(page, {"type": "chat", "role": "assistant", "chat_id": 1, "task_id": tid,
                                             "content": answer})
                        dialogue_store.append(root, "chat", summary)
                        _emit_ws_frame(page, {**summary, "type": "chat", "role": "system",
                                             "system_type": "task_summary", "content": summary["text"]})
                    # The server publishes each new ending row itself (its terminal-projection sweep), as at a real
                    # task's end; the late review comes after that, so the scenario waits for it.
                    deadline = time.monotonic() + 120
                    while time.monotonic() < deadline and any(terminal_projection_owed(
                            f"card-{name}", load_task_result(root, f"card-{name}") or {}) for name in LATE):
                        page.wait_for_timeout(500)
                    assert not any(terminal_projection_owed(f"card-{name}", load_task_result(root, f"card-{name}") or {})
                                   for name in LATE), "the server published every ending row"
                    settle_history(page)
                    for name in LATE:
                        assert BEFORE in look(page, name, "Done with warnings", "light-before"), name
                    page.screenshot(path=str(evidence / f"{engine}-light-before.png"), animations="disabled")
                    assert [resolve_pending_result_review(root, f"card-{name}", f"rec-{name}") for name in LATE] == [
                        verdict == "PASS" for _warning, verdict, *_rest in LATE.values()]
                    events = {event["task_id"]: event for event in (
                        json.loads(line) for line in (root / "logs" / "events.jsonl").read_text(encoding="utf-8")
                        .splitlines() if '"review_late_result"' in line)}
                    assert sorted(events) == ["card-late-pass", "card-late-pass-receipt", "card-late-stale"]
                    # A newer durable state of card-late-stale (a custody debt recorded after its late PASS) reaches the
                    # card first, through its existing terminal consumer, one minute after the late event was made.
                    newer = write_task_result(root, "card-late-stale", "completed", delegated_runs_unreconciled=["run-x"])
                    later = (datetime.datetime.fromisoformat(events["card-late-stale"]["ts"])
                             + datetime.timedelta(minutes=1)).isoformat()
                    newer_row = {**_project_row("card-late-stale", newer, {}, {"chat_id": 1}), "ts": later}
                    assert newer_row["outcome_phase"] == "warn" and newer_row["delegated_runs_unreconciled"] == ["run-x"]
                    _emit_ws_frame(page, {**newer_row, "type": "chat", "role": "system", "system_type": "task_summary",
                                         "content": newer_row["text"]})
                    # Its late frame is published after that newer state reached the card, still carrying the older
                    # revision it read: the revision orders it, never this later ts.
                    published = (datetime.datetime.fromisoformat(later) + datetime.timedelta(minutes=1)).isoformat()
                    stale = {**events["card-late-stale"], "ts": published}
                    assert stale["current_outcome"]["revision"] < newer["updated_at"] < later < published
                    seat = {"type": "review_late_result", "task_id": "card-late-fail", "surface": "result",
                            "slot_id": "s1", "status": "complete", "operation_state": "complete"}
                    ghost = {**events["card-late-pass"], "task_id": "card-ghost", "record_id": "rec-ghost"}
                    for event in (*{**events, "card-late-stale": stale}.values(), events["card-late-pass"], seat,
                                  ghost):  # PASS twice: a replay
                        log_frame(page, event)
                    visits = ("light-live", "light-reloaded")
                else:
                    visits = ("dark-reloaded",)
                for step in visits:
                    if step.endswith("reloaded"):
                        page.reload(wait_until="domcontentloaded")
                    for name, (_warning, _verdict, headline, never) in LATE.items():
                        line = look(page, name, headline, step)
                        assert never not in line, (step, name, line)
                        lines.setdefault(name, {})[step] = line
                    if step.endswith("reloaded"):
                        log_frame(page, stale)  # a delayed snapshot must also lose to the revision history just showed
                        assert look(page, "late-stale", "Done with warnings", step + "-delayed") == lines["late-stale"][step]
                    assert page.locator('.chat-live-card[data-task-id="card-ghost"]').count() == 0, step
                    page.screenshot(path=str(evidence / f"{engine}-{step}.png"), animations="disabled")
                assert not errors.unexpected(), errors
                context.close()
        finally:
            browser.close()
    for name, seen in lines.items():  # one current outcome: the live card's line is the line its reloads read
        assert seen["light-live"] == seen["light-reloaded"] == seen["dark-reloaded"], (name, seen)
    assert RECEIPT in lines["late-pass-receipt"]["light-live"] and BEFORE in lines["late-fail"]["light-live"]


CHILD = {"parent_task_id": "card-parent", "root_task_id": "card-parent", "delegation_role": "subagent",
         "subagent_role": "researcher"}


@pytest.mark.parametrize("engine", ["webkit", "chromium"])
def test_a_late_pass_repaints_a_finished_child_card_through_its_own_terminal_path(direct_server_with_data, tmp_path,
                                                                                 monkeypatch, engine):
    """Astra, 9 Nov 2026: helpers that finished before their reviewers answered read Done with warnings under their
    working parent. The real resolver then writes each stored outcome and the existing ``review_late_result`` event
    carries its CURRENT outcome as ``bridge.push_log`` frames it: a finished child card takes it through the child's
    own terminal path (Done) and keeps its report line. The parent card is untouched, a late frame older than a newer
    child state the card shows never paints over it, a child with no card here gets none, and a reload reads the
    same outcome from the stored rows."""
    import datetime
    import json
    import time

    from playwright.sync_api import expect, sync_playwright
    from ouroboros import dialogue_store, review_ledger
    from ouroboros.late_review_marks import resolve_pending_result_review
    from ouroboros.task_results import load_task_result, write_task_result
    from ouroboros.terminal_projection import terminal_projection_owed
    from supervisor.log_addressing import address_task_event
    from tests.ui_chat_viewport_smoke import (
        _CAPTURE_TEST_SOCKET, _OBSERVE_STATE_READS, _emit_ws_frame, _wait_socket_open_quiescent,
    )

    root = direct_server_with_data["data_dir"]
    evidence = producer_evidence_dir(tmp_path / "screenshots", "result-review-child-late",
                                     env="OUROBOROS_BROWSER_EVIDENCE_OUT")
    evidence.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(review_ledger, "load_record", lambda _root, record_id: {
        "record_id": record_id, "subject": {"diff_sha": "current"}, "verdict": {"aggregate": "PASS"},
        "ts": "2026-11-09T01:00:00+00:00"})
    write_task_result(root, "card-parent", "running", chat_id=1, root_task_id="card-parent", delegation_role="root")

    def stored(row, frame):  # a progress row as ``message_bus`` keeps it, and the frame its live socket carried
        dialogue_store.append(root, "progress", row)
        return {**row, **frame}

    def log_frame(page, event):  # what bridge.push_log broadcasts for one addressed event
        addressed = address_task_event(None, root, dict(event))
        _emit_ws_frame(page, {"type": "log", "data": addressed, "chat_id": int(addressed.get("chat_id") or 0)})

    def phase(page, task_id):
        return page.locator(f'.chat-live-card[data-task-id="{task_id}"] [data-live-phase]').first

    reads = {"open": 0, "at": 0.0}

    def note_history(request, step):  # a card that first finishes schedules its own history read
        if "/api/chat/history" in request.url:
            reads.update(open=reads["open"] + step, at=time.monotonic())

    def settle(page):  # the server's own ending rows and those reads land before the late frames, never between
        deadline = time.monotonic() + 120
        while time.monotonic() < deadline and (any(terminal_projection_owed(tid, load_task_result(root, tid) or {})
                                                   for tid in ("card-child", "card-child-stale"))
                                               or reads["open"] or time.monotonic() - reads["at"] < 1.5):
            page.wait_for_timeout(250)
        assert not reads["open"], "a history read stayed open"

    def look(page, step):
        for tid, headline in (("card-child", "Done"), ("card-child-stale", "Done with warnings")):
            card = page.locator(f'.chat-live-card[data-task-id="{tid}"]')
            expect(card.locator('[data-live-phase]').first).to_have_text(headline, timeout=30000)
            assert f"Report from {tid}" in card.locator('[data-live-activity]').first.inner_text(), (step, tid)
            card.evaluate("el => el.scrollIntoView({block: 'center'})")
            card.screenshot(path=str(evidence / f"{engine}-{step}-{tid}.png"), animations="disabled")
        assert page.locator('.chat-live-card[data-task-id="card-child-ghost"]').count() == 0, step
        page.screenshot(path=str(evidence / f"{engine}-{step}.png"), animations="disabled")

    with sync_playwright() as pw:
        browser = getattr(pw, engine).launch()
        try:
            for theme in ("light", "dark"):
                context = browser.new_context(viewport={"width": 1100, "height": 750}, color_scheme=theme,
                                              reduced_motion="reduce")
                page = context.new_page()
                errors = PageErrors(page, engine)
                page.add_init_script(f"({_CAPTURE_TEST_SOCKET})(); ({_OBSERVE_STATE_READS})();")
                page.on("request", lambda request: note_history(request, 1))
                page.on("requestfinished", lambda request: note_history(request, -1))
                page.on("requestfailed", lambda request: note_history(request, -1))
                page.goto(direct_server_with_data["url"], wait_until="domcontentloaded")
                _wait_socket_open_quiescent(page)
                if theme == "light":  # the live story once; the dark theme reads it back after the fact
                    start = datetime.datetime(2026, 11, 9, 0, 0, tzinfo=datetime.timezone.utc)
                    at = lambda minutes: (start + datetime.timedelta(minutes=minutes)).isoformat()  # noqa: E731
                    _emit_ws_frame(page, stored({"task_id": "card-parent", "chat_id": 1, "ts": at(0), "is_progress": True,
                                                 "content": "Parent gathers its helpers' reports", "narration": True},
                                                {"type": "chat", "role": "assistant"}))
                    for minute, tid in enumerate(("card-child", "card-child-stale"), start=1):
                        axes = _pending_axes(tid, "")
                        write_task_result(root, tid, "completed", chat_id=1, result=f"Report from {tid}",
                                          terminal_origin="model_final", reason_code="final_message", outcome_axes=axes,
                                          **{k: v for k, v in CHILD.items() if k != "subagent_role"})
                        lineage = {"subagent_task_id": tid, **CHILD}
                        _emit_ws_frame(page, stored({"task_id": tid, "chat_id": 1, "ts": at(minute), "is_progress": True,
                                                     "content": f"Checking {tid}", "narration": True, "status": "running",
                                                     "subagent_event": "running", **lineage},
                                                    {"type": "chat", "role": "assistant"}))
                        notice = {"task_id": tid, "chat_id": 1, "ts": at(minute + 2), "is_progress": True,
                                  "role": "system", "system_type": "subagent_terminal_notice",
                                  "content": f"Subagent {tid} completed (researcher).", "subagent_event": "completed",
                                  "status": "completed", "result": f"Report from {tid}", "outcome_axes": axes,
                                  "reason_code": "final_message", **lineage}
                        _emit_ws_frame(page, stored(notice, {"type": "chat"}))
                        log_frame(page, {"type": "task_done", "task_id": tid, "chat_id": 1, "status": "completed",
                                         "outcome_axes": axes, "reason_code": "final_message", "ts": at(minute + 2),
                                         **{k: CHILD[k] for k in ("parent_task_id", "root_task_id")}})
                    settle(page)
                    for tid in ("card-child", "card-child-stale"):
                        expect(phase(page, tid)).to_have_text("Done with warnings", timeout=30000)
                    parent_before = page.locator('.chat-live-card[data-task-id="card-parent"]').first.evaluate(
                        "el => [el.querySelector('[data-live-phase]')?.textContent, el.dataset.finished || '']")
                    page.screenshot(path=str(evidence / f"{engine}-light-before.png"), animations="disabled")
                    assert all(resolve_pending_result_review(root, tid, f"rec-{tid}")
                               for tid in ("card-child", "card-child-stale"))
                    events = {event["task_id"]: event for event in (
                        json.loads(line) for line in (root / "logs" / "events.jsonl").read_text(encoding="utf-8")
                        .splitlines() if '"review_late_result"' in line)}
                    assert events["card-child"]["parent_task_id"] == "card-parent"
                    assert events["card-child"]["current_outcome"]["outcome_phase"] == "done"
                    # A newer state of card-child-stale (a custody debt recorded after its late PASS) reaches its card
                    # first, through its own terminal path, one minute after its late event was made.
                    write_task_result(root, "card-child-stale", "completed", delegated_runs_unreconciled=["run-x"])
                    later = (datetime.datetime.fromisoformat(events["card-child-stale"]["ts"])
                             + datetime.timedelta(minutes=1)).isoformat()
                    stale_axes = _pending_axes("card-child-stale", "")
                    log_frame(page, {"type": "task_done", "task_id": "card-child-stale", "chat_id": 1,
                                     "status": "completed", "outcome_axes": stale_axes, "reason_code": "final_message",
                                     "delegated_runs_unreconciled": ["run-x"], "ts": later,
                                     **{k: CHILD[k] for k in ("parent_task_id", "root_task_id")}})
                    ghost = {**events["card-child"], "task_id": "card-child-ghost", "record_id": "rec-ghost"}
                    for event in (events["card-child"], events["card-child-stale"], events["card-child"], ghost):
                        log_frame(page, event)  # card-child twice: a replay changes nothing
                    parent_after = page.locator('.chat-live-card[data-task-id="card-parent"]').first.evaluate(
                        "el => [el.querySelector('[data-live-phase]')?.textContent, el.dataset.finished || '']")
                    assert parent_after == parent_before, "the parent card is not the child's"
                    visits = ("light-live", "light-reloaded")
                else:
                    visits = ("dark-reloaded",)
                for step in visits:
                    if step.endswith("reloaded"):
                        page.reload(wait_until="domcontentloaded")
                    look(page, step)
                assert not errors.unexpected(), errors
                context.close()
        finally:
            browser.close()


@pytest.mark.parametrize("engine", ["webkit", "chromium"])
def test_a_late_outcome_keeps_its_microsecond_order_on_root_and_child_cards(direct_server_with_data, tmp_path, monkeypatch,
                                                                           engine):
    """Astra, 10 Nov 2026 (C1): a stored revision carries microseconds, and a custody warning can land inside the
    millisecond of the late PASS before it. Read to the millisecond, the delayed older PASS repainted Done over it. On a
    finished root and a finished child, an outcome older by 688 µs changes nothing on the open card whose clock took the
    newer frame, and after a reload whose clock is the stored row's revision, on roots and children alike. The same
    revision again repaints the same outcome; a truly later outcome inside that millisecond paints Done, and the older
    warning cannot return. The stored rows and announcements are production; only the writer's clock is set, to make
    the millisecond exact."""
    import datetime
    import json
    import time

    from playwright.sync_api import expect, sync_playwright
    from ouroboros import dialogue_store, late_review_marks as late, review_ledger, task_results
    from ouroboros.task_results import load_task_result, write_task_result
    from ouroboros.terminal_projection import _project_row, terminal_projection_owed
    from supervisor.log_addressing import address_task_event
    from supervisor.message_bus import log_chat
    from tests.ui_chat_viewport_smoke import (
        _CAPTURE_TEST_SOCKET, _OBSERVE_STATE_READS, _emit_ws_frame, _wait_socket_open_quiescent,
    )

    root = direct_server_with_data["data_dir"]
    evidence = producer_evidence_dir(tmp_path / "screenshots", "result-review-microseconds",
                                     env="OUROBOROS_BROWSER_EVIDENCE_OUT")
    evidence.mkdir(parents=True, exist_ok=True)
    root_id, child_id = "card-us-root", "card-us-child"
    tasks = (root_id, child_id)
    report = f"Report from {child_id}"
    warned = {root_id: "Some delegated work was never reconciled.", child_id: report}
    cleared = {root_id: "", child_id: report}
    monkeypatch.setattr(review_ledger, "load_record", lambda _root, record_id: {
        "record_id": record_id, "subject": {"diff_sha": "current"}, "verdict": {"aggregate": "PASS"},
        "ts": "2026-11-10T00:00:00+00:00"})
    now = lambda: datetime.datetime.now(datetime.timezone.utc).isoformat()  # noqa: E731

    def stored(row, frame):  # a progress row as ``message_bus`` keeps it, and the frame its live socket carried
        dialogue_store.append(root, "progress", row)
        return {**row, **frame}

    def log_frame(page, event):  # what bridge.push_log broadcasts for one addressed event
        addressed = address_task_event(None, root, dict(event))
        _emit_ws_frame(page, {"type": "log", "data": addressed, "chat_id": int(addressed.get("chat_id") or 0)})

    def card(page, task_id):
        return page.locator(f'.chat-live-card[data-task-id="{task_id}"]').first

    def view(page, task_id):  # everything a frame the card rejects must leave as it was, and where the card stands
        return card(page, task_id).evaluate("""el => ({
            phase: el.querySelector('[data-live-phase]').textContent,
            activity: el.querySelector('[data-live-activity]').textContent,
            finished: el.dataset.finished || '',
            timeline: [...el.querySelector('[data-live-timeline]').children].map(row => row.textContent),
            place: [...el.parentNode.children].indexOf(el)})""")

    def picture(page, step):
        for task_id in tasks:
            card(page, task_id).evaluate("el => el.scrollIntoView({block: 'center'})")
            card(page, task_id).screenshot(path=str(evidence / f"{engine}-{step}-{task_id}.png"), animations="disabled")

    reads = {"open": 0, "at": 0.0}

    def note_history(request, step):  # a card that first finishes schedules its own history read
        if "/api/chat/history" in request.url:
            reads.update(open=reads["open"] + step, at=time.monotonic())

    def settle(page):  # the server's own ending rows and those reads land before the late frames, never between
        deadline = time.monotonic() + 120
        while time.monotonic() < deadline and (any(terminal_projection_owed(tid, load_task_result(root, tid) or {})
                                                   for tid in tasks) or reads["open"]
                                               or time.monotonic() - reads["at"] < 1.5):
            page.wait_for_timeout(250)
        assert not reads["open"], "a history read stayed open"

    def stamped(moment, write, *args, **kwargs):  # the production writer, at an exact moment of its own clock
        with monkeypatch.context() as patch:
            patch.setattr(task_results, "utc_now_iso", lambda: moment)
            return write(*args, **kwargs)

    frames, kept, painted, changes = {}, {}, {}, {}

    def facts(state):  # what a repaint of the same outcome must keep; the outcome row's time is the revision it took
        return {**state, "timeline": [re.sub(r"\b\d{1,2}:\d{2}(?::\d{2})?\b", "", row) for row in state["timeline"]]}

    def keeps(page, task_id, frame, step, replay=False):  # recorded per case: the first red case must not hide the others
        before = view(page, task_id)
        log_frame(page, frame)
        after = view(page, task_id)
        kept[(task_id, step)] = (facts(after) == facts(before)) if replay else after == before
        if not kept[(task_id, step)]:
            changes[f"{task_id} / {step}"] = {"before": before, "after": after}

    def paints(page, task_id, frame, headline, line, step):
        log_frame(page, frame)
        expect(card(page, task_id).locator('[data-live-phase]').first).to_have_text(headline, timeout=30000)
        painted[(task_id, step)] = card(page, task_id).locator('[data-live-activity]').first.inner_text() == line

    def reloaded(page, headline):
        page.reload(wait_until="domcontentloaded")
        _wait_socket_open_quiescent(page)
        for task_id in tasks:
            expect(card(page, task_id).locator('[data-live-phase]').first).to_have_text(headline, timeout=30000)
        settle(page)

    def parent(page):
        return card(page, "card-parent").evaluate(
            "el => [el.querySelector('[data-live-phase]')?.textContent, el.dataset.finished || '']")

    with sync_playwright() as pw:
        browser = getattr(pw, engine).launch()
        try:
            context = browser.new_context(viewport={"width": 1100, "height": 750}, reduced_motion="reduce")
            page = context.new_page()
            errors = PageErrors(page, engine)
            page.add_init_script(f"({_CAPTURE_TEST_SOCKET})(); ({_OBSERVE_STATE_READS})();")
            page.on("request", lambda request: note_history(request, 1))
            page.on("requestfinished", lambda request: note_history(request, -1))
            page.on("requestfailed", lambda request: note_history(request, -1))
            page.goto(direct_server_with_data["url"], wait_until="domcontentloaded")
            _wait_socket_open_quiescent(page)
            answer = "The report whose order is checked."
            ended = write_task_result(root, root_id, "completed", chat_id=1, root_task_id=root_id, delegation_role="root",
                                      result=answer, terminal_origin="model_final", reason_code="final_message",
                                      outcome_axes=_pending_axes(root_id, ""))
            summary = _project_row(root_id, ended, {}, {"chat_id": 1})
            _emit_ws_frame(page, stored({"task_id": root_id, "chat_id": 1, "content": "Checking the order",
                                         "ts": summary["ts"], "narration": True},
                                        {"type": "chat", "role": "assistant", "is_progress": True}))
            log_chat("out", 1, 1, answer, task_id=root_id, drive_root=root)
            _emit_ws_frame(page, {"type": "chat", "role": "assistant", "chat_id": 1, "task_id": root_id, "content": answer})
            dialogue_store.append(root, "chat", summary)
            _emit_ws_frame(page, {**summary, "type": "chat", "role": "system", "system_type": "task_summary",
                                  "content": summary["text"]})
            write_task_result(root, "card-parent", "running", chat_id=1, root_task_id="card-parent", delegation_role="root")
            _emit_ws_frame(page, stored({"task_id": "card-parent", "chat_id": 1, "ts": now(), "is_progress": True,
                                         "content": "Parent gathers its helpers' reports", "narration": True},
                                        {"type": "chat", "role": "assistant"}))
            axes = _pending_axes(child_id, "")
            write_task_result(root, child_id, "completed", chat_id=1, result=report, terminal_origin="model_final",
                              reason_code="final_message", outcome_axes=axes,
                              **{k: v for k, v in CHILD.items() if k != "subagent_role"})
            lineage = {"subagent_task_id": child_id, **CHILD}
            _emit_ws_frame(page, stored({"task_id": child_id, "chat_id": 1, "ts": now(), "is_progress": True,
                                         "content": "Checking the order", "narration": True, "status": "running",
                                         "subagent_event": "running", **lineage},
                                        {"type": "chat", "role": "assistant"}))
            finished = now()
            _emit_ws_frame(page, stored({"task_id": child_id, "chat_id": 1, "ts": finished, "is_progress": True,
                                         "role": "system", "system_type": "subagent_terminal_notice",
                                         "content": f"Subagent {child_id} completed (researcher).",
                                         "subagent_event": "completed", "status": "completed", "result": report,
                                         "outcome_axes": axes, "reason_code": "final_message", **lineage},
                                        {"type": "chat"}))
            log_frame(page, {"type": "task_done", "task_id": child_id, "chat_id": 1, "status": "completed",
                             "outcome_axes": axes, "reason_code": "final_message", "ts": finished,
                             **{k: CHILD[k] for k in ("parent_task_id", "root_task_id")}})
            settle(page)
            for task_id in tasks:
                expect(card(page, task_id).locator('[data-live-phase]').first).to_have_text("Done with warnings", timeout=30000)
            parent_before = parent(page)

            # One millisecond, later than anything these cards have shown: the PASS, the custody warning 688 µs after
            # it (the reproduction's real interval), and a clean state 112 µs after that.
            base = (datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(seconds=2)).replace(microsecond=147169)
            older, newer, latest = (base + datetime.timedelta(microseconds=by) for by in (0, 688, 800))
            assert older.isoformat()[:23] == latest.isoformat()[:23]
            for task_id in tasks:
                assert stamped(older.isoformat(), late.resolve_pending_result_review, root, task_id, f"rec-{task_id}")
                stale = next(row for row in map(json.loads, (root / "logs" / "events.jsonl").read_text(encoding="utf-8")
                                                .splitlines()) if row.get("task_id") == task_id)
                row = stamped(newer.isoformat(), write_task_result, root, task_id, "completed",
                              delegated_runs_unreconciled=["run-x"])
                warning = {**stale, "current_outcome": late.late_current_outcome(root, row)}
                assert [frame["current_outcome"]["outcome_phase"] for frame in (stale, warning)] == ["done", "warn"]
                assert [frame["current_outcome"]["revision"] for frame in (stale, warning)] == [older.isoformat(),
                                                                                                newer.isoformat()]
                frames[task_id] = {"stale": stale, "warning": warning}

                # The open card first takes the newer outcome, then the delayed older one arrives.
                paints(page, task_id, warning, "Done with warnings", warned[task_id], "live: the newer warning")
                keeps(page, task_id, warning, "live: the same revision again", replay=True)
                keeps(page, task_id, stale, "live: older by 688 microseconds")
                keeps(page, task_id, {**stale, "current_outcome": {**stale["current_outcome"], "revision": (
                    older - datetime.timedelta(milliseconds=2)).isoformat()}}, "live: older by 2 milliseconds")
            assert parent(page) == parent_before, "the parent card is not the child's"
            picture(page, "live-after-stale")

            # A reload's clock is the stored row's revision; the delayed older outcome must lose to it too.
            reloaded(page, "Done with warnings")
            for task_id in tasks:
                keeps(page, task_id, frames[task_id]["stale"], "reloaded: older by 688 microseconds")
                keeps(page, task_id, frames[task_id]["warning"], "reloaded: the same revision again", replay=True)
            picture(page, "reloaded-after-stale")

            # A later state of the row inside that millisecond is current, and an older warning cannot return over it.
            for task_id in tasks:
                row = stamped(latest.isoformat(), write_task_result, root, task_id, "completed",
                              delegated_runs_unreconciled=[])
                clean = {**frames[task_id]["stale"], "current_outcome": late.late_current_outcome(root, row)}
                assert clean["current_outcome"]["outcome_phase"] == "done" and clean["current_outcome"]["revision"] == latest.isoformat()
                paints(page, task_id, clean, "Done", cleared[task_id], "later: inside the same millisecond")
                keeps(page, task_id, frames[task_id]["warning"], "later: the older warning arrives after it")
            picture(page, "later-pass")
            reloaded(page, "Done")
            picture(page, "later-pass-reloaded")
            assert not errors.unexpected(), errors
            context.close()
        finally:
            browser.close()
    (evidence / f"{engine}-cases.json").write_text(json.dumps(
        {"kept": {" / ".join(key): value for key, value in kept.items()},
         "painted": {" / ".join(key): value for key, value in painted.items()}, "changes": changes}, indent=1),
        encoding="utf-8")
    assert all(kept.values()) and all(painted.values()), json.dumps(
        {"kept": {" / ".join(key): value for key, value in kept.items() if not value},
         "painted": {" / ".join(key): value for key, value in painted.items() if not value}, "changes": changes})


# Settings › Behavior says the same rules: the answer a task requires checked, Advisory's re-review choice after a
# review, Blocking's hold, and result review counted per root task (``review_cycles``), plans outside the cap.
SETTINGS = {
    "OUROBOROS_TASK_REVIEW_MODE": ["an answer its task requires checked",
                           "With Advisory, Ouroboros decides after a review whether a revised result needs another",
                           "with Blocking, a result that did not pass is held until a review passes it"],
    "OUROBOROS_REVIEW_MAX_CYCLES": ["result review and commit review per task"],
}


@pytest.mark.parametrize("engine", ["webkit", "chromium"])
def test_settings_say_the_result_review_rules_in_both_themes(direct_server_with_data, tmp_path, engine):
    from playwright.sync_api import expect, sync_playwright

    fixture = direct_server_with_data
    evidence = producer_evidence_dir(tmp_path / "screenshots", "result-review-cards", env="OUROBOROS_BROWSER_EVIDENCE_OUT")
    evidence.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as pw:
        browser = getattr(pw, engine).launch()
        try:
            for theme in ("light", "dark"):
                context = browser.new_context(viewport={"width": 1100, "height": 750}, color_scheme=theme,
                                              reduced_motion="reduce")
                page = context.new_page()
                errors = PageErrors(page, engine)
                page.goto(fixture["url"], wait_until="domcontentloaded")
                page.locator('[data-nav-page="settings"]').click()
                page.wait_for_selector(".settings-shell", timeout=30_000)
                for row_id, says in SETTINGS.items():
                    open_settings(page, "permissions", row=row_id)
                    section = setting_row(page, row_id)
                    section.locator('.settings-row-more > summary').click()
                    for words in says:
                        expect(section).to_contain_text(words)
                    section.evaluate("el => el.scrollIntoView({block: 'center'})")
                    assert page.evaluate("() => { const s = document.querySelector('[data-settings-page=permissions]');"
                                         " return s.scrollWidth - s.clientWidth; }") <= 1
                    slug = row_id.lower()
                    section.screenshot(path=str(evidence / f"settings-{engine}-{theme}-{slug}.png"), animations="disabled")
                page.screenshot(path=str(evidence / f"settings-{engine}-{theme}.png"), animations="disabled")
                assert not errors.unexpected(), errors
                context.close()
        finally:
            browser.close()
