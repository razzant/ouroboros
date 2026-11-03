"""#1585, owner decision 13.10: a failed check keeps its own row in the task card.

Runs through the REAL server, worker pool, tool loop and SPA: the shared
``direct_server_with_data`` fixture (a copied candidate checkout, a disposable data
root, one worker) with its in-process mock model scripted here. The owner asks for a
report fix; the model calls ``verify_and_record`` on the report (the host runs the
check; its criterion fails), then ``run_command`` to fix the file, then the same check
again (it passes), then answers. Every row is produced by the product: live frames
from the worker over the WebSocket, then, after a reload, the durable tools.jsonl
replayed through ``/api/chat/history``. The disposable settings switch the safety LLM
review and the task-result review off, because the mock model is neither reviewer (its
"OK" makes a review DEGRADED, which is a task outcome, not this row's); nothing else differs.

The owner-visible states are captured to ``OUROBOROS_UI_EVIDENCE_DIR`` (light and
dark, desktop and phone width): mid-task with the failed check, the row expanded,
the finished task, and the replayed card after a reload.

The second test is the compound card: a real subagent tree (root and child, each with
its own failed check) through the same server and SPA, then late and threshold frames
on the page's own socket, then the durable replay of both cards.
"""
from __future__ import annotations

import json
import os
import pathlib
import re
import threading
from types import SimpleNamespace

import pytest

from tests import test_ui_smoke_playwright as smoke
from tests.test_owner_wait_integration import wait_clone as clone_fixture

direct_server_with_data = smoke.direct_server_with_data
wait_clone = clone_fixture

pytestmark = [pytest.mark.ui_browser, pytest.mark.serial]

MARKER = "CHECK-1585"
CRITERION = "report-final"


class _CheckModel:
    """Main-model answers in order: failing check, (held) fix, passing check, finish."""

    def __init__(self, monkeypatch, report: pathlib.Path):
        from tests import fixtures_mock_llm

        self.report = report
        self.held, self.release = threading.Event(), threading.Event()
        self.lock = threading.Lock()
        self.main = 0
        monkeypatch.setattr(fixtures_mock_llm._Handler, "do_POST", lambda handler: self.handle(handler))

    def _call(self, name, arguments):
        return {"role": "assistant", "content": "", "tool_calls": [{
            "id": f"call_{self.main}", "type": "function",
            "function": {"name": name, "arguments": json.dumps(arguments)}}]}, "tool_calls"

    def handle(self, handler):
        from tests.test_batch4_owner_controls_browser import _reply, _user_text

        length = int(handler.headers.get("Content-Length", 0) or 0)
        payload = json.loads(handler.rfile.read(length) or b"{}")
        users = [_user_text(m) for m in payload.get("messages") or [] if isinstance(m, dict) and m.get("role") == "user"]
        message, finish = {"role": "assistant", "content": "OK"}, "stop"
        # The task's own conversation opens with the owner's words; review agents and
        # other helpers quote them inside their own prompts and keep the stock answer.
        if payload.get("tools") and users and users[0].startswith(MARKER):
            with self.lock:
                self.main += 1
                ordinal = self.main
            check = {"contract_kind": "explicit_command", "check": ["cat", str(self.report)],
                     "expected": "status: final", "criterion_id": CRITERION, "criterion_source": "task_stated"}
            if ordinal in (1, 3):
                message, finish = self._call("verify_and_record", check)
            elif ordinal == 2:
                self.held.set()
                self.release.wait(240)
                message, finish = self._call("run_command", {
                    "cmd": ["sh", "-c", "printf 'status: final\\n' > report.txt"],
                    "cwd": str(self.report.parent), "outputs": ["report.txt"]})
            else:  # every later turn (e.g. after the acceptance review) selects completion again
                message, finish = self._call("finish_task", {
                    "action": "finish", "answer": "The report now says final; the check passed."})
        _reply(handler, payload, message, finish)


def _launch(pw, engine):
    """The engine, or a skip when this machine has no such build (unless engines are required)."""
    from playwright.sync_api import Error

    try:
        return getattr(pw, engine).launch(headless=True)
    except Error as exc:
        if "Executable doesn't exist" in str(exc) and not os.environ.get("OUROBOROS_EXPECT_BROWSER_ENGINES"):
            pytest.skip(f"{engine} is not installed here: {str(exc).splitlines()[0]}")
        raise


def _tool_rows(data_dir):
    path = data_dir / "logs" / "tools.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _evidence(data_dir):
    root = pathlib.Path(os.environ.get("OUROBOROS_UI_EVIDENCE_DIR", str(data_dir.parent))) / "verification-check-row"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _open(card):
    if card.get_attribute("data-expanded") != "1":
        card.locator(":scope > [data-live-summary-button] [data-live-disclosure]").click()


def _assert_running(card):
    """A failed check leaves the task running. The chip's exact word while a model call is held
    follows the host's census (`Working`, `Preparing request`, …), so the invariant is "not over"."""
    phase = card.locator("[data-live-phase]").first.inner_text()
    assert phase not in {"Done", "Done with warnings", "Failed", "Stopped", ""}, phase


def _check_row(card):
    return card.locator(".chat-live-line.warn", has_text="Check did not pass")


def _assert_row(card):
    row = _check_row(card)
    assert row.count() == 1, card.inner_text()
    head = row.locator(".chat-live-line-title").inner_text()
    assert head == "Check did not pass: expected text not found", head
    text = row.inner_text()
    assert "The check ran successfully (exit 0); its criterion was not met." in text, text
    evidence = card.locator(".chat-live-line", has_text="tool call").first.inner_text()
    assert "error" not in evidence, evidence
    return row


@pytest.mark.parametrize("engine", ["chromium", "webkit"])
def test_failed_check_row_live_expanded_finished_and_replayed(direct_server_with_data, monkeypatch, engine):
    from playwright.sync_api import sync_playwright

    fixture = direct_server_with_data
    data_dir, url = fixture["data_dir"], fixture["url"]
    # The server's isolated home is the task's user_files root, so the fix may write it.
    report = data_dir.parent / "home" / "report.txt"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("status: draft\n", encoding="utf-8")
    fixture["stop_server"]()
    settings_path = data_dir / "settings.json"
    settings = json.loads(settings_path.read_text(encoding="utf-8"))
    settings.update(OUROBOROS_SAFETY_MODE="off", OUROBOROS_TASK_REVIEW_MODE="off")
    settings_path.write_text(json.dumps(settings), encoding="utf-8")
    model = _CheckModel(monkeypatch, report)
    fixture["start_server"]()
    shots = _evidence(data_dir)
    with sync_playwright() as pw:
        browser = _launch(pw, engine)
        try:
            page = browser.new_page(viewport={"width": 1280, "height": 900}, color_scheme="light")
            page.goto(url, wait_until="domcontentloaded")
            page.fill("#chat-input", f"{MARKER}: make the report final and verify it.")
            page.click("#chat-send")
            assert model.held.wait(120), "the model never received the failed check's result"
            card = page.locator(".chat-live-card", has=page.locator(".chat-live-line.warn", has_text="Check did not pass"))
            page.wait_for_function("() => [...document.querySelectorAll('.chat-live-card')].some(card => "
                                   "card.textContent.includes('Check did not pass'))", timeout=60_000)
            _open(card)
            task_id = card.get_attribute("data-task-id")
            row = _assert_row(card)
            _assert_running(card)
            page.screenshot(path=str(shots / f"{engine}-light-1280-live.png"), full_page=True)
            row.locator("[data-live-line-toggle]").click()
            body = row.locator(".chat-live-line-body").inner_text()
            assert "FAIL: exit=0 expected='status: final'" in body and "status: draft" in body, body
            assert f"Criterion: {CRITERION}" in body
            page.screenshot(path=str(shots / f"{engine}-light-1280-live-expanded.png"), full_page=True)

            model.release.set()
            page.wait_for_function(
                f"() => ['Done', 'Done with warnings', 'Failed', 'Stopped'].includes(document.querySelector("
                f"'.chat-live-card[data-task-id=\"{task_id}\"] [data-live-phase]')?.textContent)", timeout=180_000)
            assert card.locator("[data-live-phase]").first.inner_text() == "Done", "a failed check is not a failed task"
            _assert_row(card)
            # two checks, the fix and the finish_task completion act, each counted once
            assert "4 tool calls" in card.inner_text()
            page.screenshot(path=str(shots / f"{engine}-light-1280-done.png"), full_page=True)

            # Durable facts behind the cards: one completed call per check, both verdicts kept.
            verifies = [r for r in _tool_rows(data_dir) if r.get("task_id") == task_id
                        and r.get("tool") == "verify_and_record" and r.get("type") == "tool_call"]
            assert [(r["is_error"], r["status"], r["tool_result_meta"]["verification"]["status"]) for r in verifies] \
                == [(False, "ok", "fail"), (False, "ok", "pass")]

            for scheme, width in (("light", 1280), ("dark", 1280), ("dark", 390), ("light", 390)):
                context = browser.new_context(viewport={"width": width, "height": 900}, color_scheme=scheme,
                                              is_mobile=width == 390, has_touch=width == 390)
                replay = context.new_page()
                replay.goto(url, wait_until="domcontentloaded")
                replayed = replay.locator(f'.chat-live-card[data-task-id="{task_id}"]')
                replayed.wait_for(state="visible", timeout=60_000)
                _open(replayed)
                replayed.locator(".chat-live-line.warn", has_text="Check did not pass").wait_for(timeout=30_000)
                again = _assert_row(replayed)
                assert replayed.locator("[data-live-phase]").first.inner_text() == "Done"
                assert replay.locator("#chat-messages").evaluate("el => el.scrollWidth <= el.clientWidth + 1")
                replay.screenshot(path=str(shots / f"{engine}-{scheme}-{width}-replayed.png"), full_page=True)
                again.locator("[data-live-line-toggle]").click()
                body = again.locator(".chat-live-line-body").inner_text()
                assert "expected='status: final'" in body and f"Criterion: {CRITERION}" in body
                replay.screenshot(path=str(shots / f"{engine}-{scheme}-{width}-replayed-expanded.png"), full_page=True)
                context.close()
        finally:
            model.release.set()
            browser.close()


# ---------------------------------------------------------------------------
# The compound card: a real subagent tree, late frames and threshold rows, replay
# ---------------------------------------------------------------------------

ROOT_MARKER = "CHECK-ROOT-1585"
CHILD_MARKER = "CHECK-CHILD-1585"
CHILD_RESULT = "CHILD_RESULT_CHECK_1585"
ROOT_FINAL = "ROOT_FINAL_CHECK_1585"
CHILD_SLUG = "check-child"
_CHILD_ID = re.compile(r"Subagent request queued ([0-9a-f]{8})")
_CHILD_SHA = re.compile(r"child_result_sha256[\"'=:\s]+([0-9a-f]{64})")


class _TreeScript:
    """Root and child answers by actor, in order; the child's second round is held."""

    def __init__(self, root_report, child_report):
        self.reports = {"root": root_report, "child": child_report}
        self.child_root = child_report.parent
        self.held, self.release = threading.Event(), threading.Event()
        self.lock = threading.Lock()
        self.counts = {"root": 0, "child": 0}

    @staticmethod
    def _check(path, criterion):
        return {"contract_kind": "explicit_command", "check": ["cat", str(path)], "expected": "status: final",
                "criterion_id": criterion, "criterion_source": "task_stated"}

    def __call__(self, body):
        from tests.system_e2e.harness import body_text

        actor = "child" if body.get("model") == CHILD_SLUG else "root"
        with self.lock:
            self.counts[actor] += 1
            turn = self.counts[actor]
        text = body_text(body)
        if actor == "child":
            if turn == 1:
                return {"tool": "verify_and_record", "arguments": self._check(self.reports["child"], "child-report-final")}
            self.held.set()
            assert self.release.wait(240), "the scenario never released the child's held round"
            # An acting child completes through finish_task; a plain final is only narration for it.
            return {"tool": "finish_task", "arguments": {
                "action": "finish", "answer": f"{CHILD_RESULT}: the child report is still a draft."}}
        if turn == 1:
            return {"tool": "schedule_subagent", "arguments": {
                "subagent_id": CHILD_SLUG, "objective": f"{CHILD_MARKER}: check the child report and report what you saw.",
                "expected_output": "One sentence about the child report.",
                # Only an acting child may run a check: a read-only scout is refused verify_and_record.
                "write_surface": "external_workspace", "write_root": str(self.child_root)}}
        if turn == 2:
            return {"tool": "verify_and_record", "arguments": self._check(self.reports["root"], "root-report-final")}
        ids, shas = _CHILD_ID.findall(text), _CHILD_SHA.findall(text)
        if turn == 3:
            return {"tool": "wait_tasks", "arguments": {"task_ids": ids[-1:], "timeout_sec": 240, "mode": "all_terminal"}}
        if turn == 4 and ids and shas and CHILD_RESULT in text:
            return {"tool": "tree_note", "arguments": {"kind": "decision", "text": "Absorbed the child's report.",
                "payload": {"type": "child_result_disposition", "child_task_id": ids[-1],
                            "disposition": "integrated", "child_result_sha256": shas[-1]}}}
        return {"tool": "finish_task", "arguments": {
            "action": "finish", "answer": f"{ROOT_FINAL}: both checks ran; neither report is final yet."}}


def _threshold_frames(task_id, chat_id):
    """The checkpoints the real producer emits for each bound, as the worker publishes them."""
    import datetime as dt

    from ouroboros import task_pacing

    now = dt.datetime.now(dt.timezone.utc)
    stamp = lambda offset: (now + dt.timedelta(seconds=offset)).isoformat().replace("+00:00", "Z")  # noqa: E731
    notes = []
    ceiling = SimpleNamespace()
    for cost in (5.1, 7.6, 8.1):
        notes.append(task_pacing.build_cost_budget_note(ceiling, cost_ceiling_usd=10.0, task_cost=cost))
    pool = SimpleNamespace()
    for remaining in (19.5, 9.0, 7.5):
        notes.append(task_pacing.build_wallet_budget_note(pool, baseline_usd=40.0, task_cost=0.5, wallet={
            "status": "known", "remaining_usd": remaining, "limit_usd": 100.0, "observed_at": stamp(0)}))
    for elapsed, left in ((35, 25), (50, 10)):
        notes.append(task_pacing.build_time_budget_note(SimpleNamespace(task_metadata={
            "created_at": stamp(-elapsed), "deadline_at": stamp(left)})))
    payloads = [note.checkpoint for note in notes if note is not None]
    assert len(payloads) == 8, payloads
    # A row from before the producer named its bound keeps the label it always had.
    payloads.append({"checkpoint_kind": "cost_budget_wrapup", "deciding_spend_usd": 8.2, "base_usd": 10.0})
    return [{"type": "log", "chat_id": chat_id, "data": {
        "type": "task_checkpoint", "task_id": task_id, "ts": stamp(index), **payload}}
        for index, payload in enumerate(payloads)]


def _late_check_frames(task_id, chat_id, report):
    """A check whose wait ended and that settled, handled FAIL, after its task was done."""
    identity = {"task_id": task_id, "tool": "verify_and_record", "invocation_id": "late-inv-1", "tool_call_id": "late-call-1"}
    fact = {"status": "fail", "contract_kind": "explicit_command", "expected_match": "substring", "matched": False,
            "check": f"cat {report}", "check_rendering": "shlex_join", "criterion_id": "late-report-final",
            "criterion_source": "task_stated", "receipt_ts": "2026-10-13T00:00:00Z"}
    preview = "verify_and_record [explicit_command] FAIL: exit=0 expected='status: final'\n\nstatus: draft"
    settled = {"exit_code": 0, "status": "ok", "is_error": False, "result_preview": preview}
    rows = [{"type": "tool_call_started", "_live_tool_frame": True, **identity},
            {"type": "tool_call_timeout", **identity},
            {"type": "tool_call", **identity, **settled, "tool_result_meta": {"verification": fact}},
            {"type": "tool_call_finished", **identity, **settled, "verification": fact}]
    import datetime as dt

    now = dt.datetime.now(dt.timezone.utc)
    return [{"type": "log", "chat_id": chat_id, "data": {
        **row, "ts": (now + dt.timedelta(milliseconds=index)).isoformat().replace("+00:00", "Z")}}
        for index, row in enumerate(rows)]


def _cards(page, root_id, child_id):
    root = page.locator(f'.chat-live-card[data-task-id="{root_id}"]')
    child = root.locator(f':scope > .chat-subagents > .chat-live-card.subagent[data-task-id="{child_id}"]')
    return root, child


def _lines(card):
    return card.locator(":scope > [data-live-timeline] > .chat-live-line")


def _titles(card):
    return [text.strip() for text in _lines(card).locator(".chat-live-line-title").all_inner_texts()]


def _checks(card):
    return _lines(card).filter(has_text="Check did not pass")


def _time_of(card, title_pattern):
    """The time text of the first timeline row whose title matches."""
    for line in _lines(card).all():
        if re.match(title_pattern, line.locator(".chat-live-line-title").inner_text().strip()):
            return line.locator(".chat-live-line-time").inner_text().strip()
    raise AssertionError(f"no row titled {title_pattern!r}: {_titles(card)}")


@pytest.mark.parametrize("engine", ["chromium", "webkit"])
def test_compound_card_child_check_late_frames_thresholds_and_replay(wait_clone, tmp_path, monkeypatch, request, engine):
    from playwright.sync_api import sync_playwright

    from devtools.benchmarks.common.server_runner import _api
    from tests.system_e2e.harness import (
        ArtifactOracle,
        KeylessIsolatedServer,
        keyless_settings,
        start_server,
        wait_durable_result,
    )
    from tests.test_chat_addressing_browser import ToolCallOnlyModel
    from tests.ui_chat_viewport_smoke import _CAPTURE_TEST_SOCKET, _emit_ws_frame

    home = tmp_path / "home"
    home.mkdir()
    original_env = KeylessIsolatedServer._env
    monkeypatch.setattr(KeylessIsolatedServer, "_env", lambda server: {
        **original_env(server), "HOME": str(home), "USERPROFILE": str(home), "XDG_CONFIG_HOME": str(home / ".config")})
    (home / "child-work").mkdir()
    reports = {"root": home / "root-report.txt", "child": home / "child-work" / "child-report.txt"}
    for report in reports.values():
        report.write_text("status: draft\n", encoding="utf-8")
    script = _TreeScript(reports["root"], reports["child"])
    catalog = json.dumps({"enabled": True, "items": [{
        "subagent_id": CHILD_SLUG, "recommended_use": "A child that checks its own report.",
        "route": {"kind": "api_model", "target_id": f"openai-compatible::{CHILD_SLUG}"}, "effort": "low"}]})
    with ToolCallOnlyModel([script] * 40, model_ids=["mock-model", CHILD_SLUG]) as stub:
        server = start_server(wait_clone, tmp_path / "instance", keyless_settings(
            stub, OUROBOROS_MAX_WORKERS=2, OUROBOROS_SUBAGENTS=catalog))
        oracle = ArtifactOracle(server.data_root)
        shots = _evidence(server.data_root)
        try:
            with sync_playwright() as pw:
                browser = _launch(pw, engine)
                try:
                    page = browser.new_page(viewport={"width": 1280, "height": 1500}, color_scheme="light")
                    page.add_init_script(f"({_CAPTURE_TEST_SOCKET})()")
                    errors = []
                    page.on("pageerror", lambda error: errors.append(str(error)))
                    page.goto(server.base_url, wait_until="domcontentloaded")
                    page.wait_for_function("() => window.__testSockets?.some(s => s.readyState === WebSocket.OPEN)", timeout=60_000)
                    created = _api(server.base_url, "POST", "/api/tasks", {
                        "description": f"{ROOT_MARKER}: check your report, have your scout check its own, then report.",
                        "title": "Compound checks", "chat_id": 1, "source": "web", "metadata": {"delegation_role": "root"}})
                    root_id = created["task_id"]
                    root = page.locator(f'.chat-live-card[data-task-id="{root_id}"]')
                    root.wait_for(state="attached", timeout=90_000)
                    _open(root)
                    kids = root.locator(":scope > .chat-subagents > .chat-live-card.subagent")
                    kids.first.wait_for(state="attached", timeout=90_000)
                    assert kids.count() == 1
                    child_id = kids.first.get_attribute("data-task-id")
                    assert child_id and child_id != root_id
                    child = kids.first
                    _open(child)
                    assert script.held.wait(180), "the child never reached its held round"
                    _checks(root).first.wait_for(timeout=60_000)
                    _checks(child).first.wait_for(timeout=60_000)
                    # Live, both tasks still running: each card owns exactly its own check row.
                    assert _checks(root).count() == 1 and _checks(child).count() == 1, (_titles(root), _titles(child))
                    for card, criterion in ((root, "root-report-final"), (child, "child-report-final")):
                        row = _checks(card)
                        row.locator("[data-live-line-toggle]").click()
                        assert criterion in row.locator(".chat-live-line-body").inner_text()
                        row.locator("[data-live-line-toggle]").click()
                    _assert_running(root)
                    _assert_running(child)
                    page.screenshot(path=str(shots / f"{engine}-compound-light-1280-live.png"), full_page=True)
                    # The producer's threshold notices as the worker publishes them, while the root works:
                    # every distinct crossing is its own row, each stating its fact.
                    for frame in _threshold_frames(root_id, 1):
                        _emit_ws_frame(page, frame)
                    crossings = [t for t in _titles(root) if re.search(r"remaining|spent|Cost budget", t)]
                    assert crossings == [
                        "Cost ceiling: 50% remaining", "Cost ceiling: 25% remaining", "Cost ceiling: 81% spent",
                        "Shared wallet: 50% of starting headroom remaining", "Shared wallet: 25% of starting headroom remaining",
                        "Shared wallet: 81% of starting headroom spent",
                        "Time budget: 50% remaining", "Time budget: 25% remaining", "Cost budget wrap-up"], crossings
                    assert _checks(root).count() == 1
                    page.screenshot(path=str(shots / f"{engine}-compound-light-1280-live-thresholds.png"), full_page=True)
                    script.release.set()
                    page.wait_for_function(
                        f"() => document.querySelector('.chat-live-card[data-task-id=\"{root_id}\"] [data-live-phase]')"
                        "?.textContent === 'Done'", timeout=180_000)
                    page.wait_for_function(
                        f"() => document.querySelector('.chat-live-card[data-task-id=\"{child_id}\"] [data-live-phase]')"
                        "?.textContent === 'Done'", timeout=60_000)
                    for card in (root, child):
                        assert card.locator("[data-live-phase]").first.inner_text() == "Done", "a failed check is not a failed task"
                        assert _checks(card).count() == 1
                        assert "error" not in _lines(card).filter(has_text="tool call").first.inner_text()
                    results = {task: wait_durable_result(oracle, task) for task in (root_id, child_id)}
                    assert all(row["status"] == "completed" for row in results.values()), results

                    # Durable facts: one completed call per check, the verdict beside it.
                    def checks_of(task_id):
                        rows = [json.loads(line) for line in (server.data_root / "logs" / "tools.jsonl")
                                .read_text(encoding="utf-8").splitlines() if line.strip()]
                        return [(r["is_error"], r["status"], r["tool_result_meta"]["verification"]["status"])
                                for r in rows if r.get("task_id") == task_id and r.get("tool") == "verify_and_record"
                                and r.get("type") == "tool_call"]
                    assert checks_of(root_id) == [(False, "ok", "fail")], checks_of(root_id)
                    assert checks_of(child_id) == [(False, "ok", "fail")], checks_of(child_id)

                    # The replayed evidence stands at each task's FIRST call: every observation on the wire
                    # carries the host time its tools.jsonl row recorded, and the earliest is the first start.
                    rows = [json.loads(line) for line in (server.data_root / "logs" / "tools.jsonl")
                            .read_text(encoding="utf-8").splitlines() if line.strip()]
                    first_started = {task: next(r["ts"] for r in rows if r.get("task_id") == task and r.get("type") == "tool_call_started")
                                     for task in (root_id, child_id)}
                    carried = {m["task_id"]: m["tool_evidence"] for m in _api(server.base_url, "GET", "/api/chat/history?chat_id=1")
                               .get("messages", []) if m.get("task_id") in first_started and m.get("tool_evidence")}
                    for task in (root_id, child_id):
                        stamps = [o.get("ts") for o in carried[task]["observations"]]
                        assert stamps and all(isinstance(ts, str) and ts for ts in stamps), stamps
                        assert min(stamps) == first_started[task], (stamps, first_started[task])
                    # A check whose wait ended settles, handled FAIL, after the child is done (a scripted
                    # backend on the page's own socket): one more row, nothing reopened.
                    for frame in _late_check_frames(child_id, 1, reports["child"]):
                        _emit_ws_frame(page, frame)
                    assert _checks(child).count() == 2, _titles(child)
                    assert child.locator("[data-live-phase]").first.inner_text() == "Done", "a late check reopens nothing"
                    assert root.locator("[data-live-phase]").first.inner_text() == "Done"
                    assert _checks(root).count() == 1
                    assert "error" not in _lines(child).filter(has_text="tool call").first.inner_text()
                    page.screenshot(path=str(shots / f"{engine}-compound-light-1280-late-check.png"), full_page=True)
                    assert not errors, errors

                    # The durable replay of both cards, light and dark, desktop and phone.
                    for scheme, width in (("light", 1280), ("dark", 1280), ("dark", 390), ("light", 390)):
                        # Tall, so the whole compound card is one image (the feed scrolls inside the page).
                        context = browser.new_context(viewport={"width": width, "height": 1500 if width > 400 else 2800},
                                                      color_scheme=scheme, is_mobile=width == 390, has_touch=width == 390)
                        replay = context.new_page()
                        replay.goto(server.base_url, wait_until="domcontentloaded")
                        r_root, r_child = _cards(replay, root_id, child_id)
                        r_root.wait_for(state="visible", timeout=60_000)
                        _open(r_root)
                        r_child.wait_for(state="attached", timeout=30_000)
                        _open(r_child)
                        _checks(r_root).wait_for(timeout=30_000)
                        _checks(r_child).wait_for(timeout=30_000)
                        assert _checks(r_root).count() == 1 and _checks(r_child).count() == 1
                        assert r_root.locator("[data-live-phase]").first.inner_text() == "Done"
                        assert r_child.locator("[data-live-phase]").first.inner_text() == "Done"
                        # The evidence row stands before the check it summarises, as it did live.
                        for card in (r_root, r_child):
                            titles = _titles(card)
                            fold = next(i for i, t in enumerate(titles) if re.fullmatch(r"\d+ tool calls?", t))
                            assert fold < titles.index(next(t for t in titles if t.startswith("Check did not pass"))), titles
                            assert "error" not in titles[fold]
                        for card, task in ((r_root, root_id), (r_child, child_id)):
                            first_call = replay.evaluate("ts => new Date(ts).toLocaleTimeString([], {hour12: false})", first_started[task])
                            assert _time_of(card, r"\d+ tool calls?") == first_call, "the evidence row stands at the first call"
                            assert _time_of(card, r"\d+ tool calls?") <= _time_of(card, "Check did not pass")
                        assert replay.locator("#chat-messages").evaluate("el => el.scrollWidth <= el.clientWidth + 1")
                        replay.screenshot(path=str(shots / f"{engine}-compound-{scheme}-{width}-replayed.png"), full_page=True)
                        for card, criterion in ((r_root, "root-report-final"), (r_child, "child-report-final")):
                            _checks(card).locator("[data-live-line-toggle]").click()
                            body = _checks(card).locator(".chat-live-line-body").inner_text()
                            assert criterion in body and "status: draft" in body, body
                        replay.screenshot(path=str(shots / f"{engine}-compound-{scheme}-{width}-replayed-expanded.png"), full_page=True)
                        context.close()
                finally:
                    script.release.set()
                    browser.close()
        finally:
            script.release.set()
            server.stop()
