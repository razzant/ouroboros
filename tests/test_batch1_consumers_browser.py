"""Batch1 (#1307/#1316) through the real candidate server, gateway, WS and SPA.

A byte-faithful candidate checkout serves its own Python and JS (origin proof).
Its isolated data root is initialized by the production state initializer and
one confirmed owner decision, then its primary copy is torn: the server's own
boot recovery must show every control as unknown, never as the backup's old
"on" and never as "off", while Chat keeps working.

Tool facts come from the real loop wrapper inside that server: two rounds reuse
one provider ``tool_call_id``; a ``journal_write`` appends to a FIFO, so its
15 s wait really ends and the call really settles once the test opens a reader;
a crash during a second blocked append leaves a start-only call. Only the
reversed arrival order (settlement row before the wait-end row) is produced
in-process by the same production wrapper: the server cannot make that race
deterministic. Its frames reach the served page over the page's own socket.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import signal
import subprocess
import threading
import time
import urllib.request
import uuid
from pathlib import Path

import pytest

from tests.candidate_checkout import candidate_checkout
from tests._shared import producer_evidence_dir
from tests.system_e2e.harness import (
    ArtifactOracle,
    KeylessIsolatedServer,
    ScriptedStubModel,
    keyless_settings,
    message_text,
    wait_durable_result,
    wait_until,
    write_settings_file,
)
from tests.ui_chat_viewport_smoke import _CAPTURE_TEST_SOCKET

pytestmark = [pytest.mark.serial, pytest.mark.browser,
              pytest.mark.skipif(os.name == "nt", reason="FIFO/crash qualification requires POSIX")]

REPO = Path(__file__).resolve().parents[1]
REPEATED_PROVIDER_ID = "call_repeated"
SERVED_MODULES = ("chat.js", "chat_activity.js", "log_events.js", "evolution.js", "logs.js", "activity.js")


def _launch(pw):
    """Playwright's own Chromium when installed, else the machine's Chrome channel."""
    from playwright.sync_api import Error

    channel = os.environ.get("OUROBOROS_BROWSER_CHANNEL", "")
    if channel:
        return pw.chromium.launch(channel=channel), channel
    try:
        return pw.chromium.launch(), "playwright-chromium"
    except Error:
        return pw.chromium.launch(channel="chrome"), "chrome"


@pytest.fixture
def batch1_clone(tmp_path):
    pytest.importorskip("playwright.sync_api")
    from playwright.sync_api import Error, sync_playwright

    from ouroboros.tools.browser import _set_playwright_browsers_path_if_bundled

    _set_playwright_browsers_path_if_bundled()
    try:
        with sync_playwright() as pw:
            browser, _engine = _launch(pw)
            browser.close()
    except Error as exc:
        if os.environ.get("OUROBOROS_EXPECT_BROWSER_ENGINES"):
            pytest.fail(str(exc))
        pytest.skip(str(exc))
    with candidate_checkout(REPO, tmp_path / "clone", origin_proof=True) as candidate:
        yield candidate


def _seed_recovering_state(root: Path) -> dict:
    """Production initializer plus one confirmed decision, then a torn primary."""
    from supervisor import state as state_module

    previous = state_module.DRIVE_ROOT
    state_module.init(root)
    try:
        assert state_module.init_state(origin="first_boot").quality == "current"
        state_module.update_state(
            lambda st: st.update(evolution_mode_enabled=True, bg_consciousness_enabled=True),
            confirm=("evolution_mode_enabled", "bg_consciousness_enabled"))
    finally:
        state_module.init(previous)
    backup = json.loads((root / "state" / "state.last_good.json").read_text(encoding="utf-8"))
    assert backup["evolution_mode_enabled"] is True and backup["bg_consciousness_enabled"] is True
    torn = b'{"evolution_mode_enabled": true, "bg_consciousness_ena'
    (root / "state" / "state.json").write_bytes(torn)
    return {"torn": torn, "initialization_id": backup["initialization_id"]}


def _turn_of(body: dict, markers: dict) -> tuple:
    """(turn, round) of a chat-turn body: the LAST owner row naming a marker, and
    the tool results the loop has fed back after it."""
    messages = body.get("messages") or []
    for index in range(len(messages) - 1, -1, -1):
        if messages[index].get("role") != "user":
            continue
        text = message_text(messages[index])
        for name, marker in markers.items():
            if marker in text:
                return name, sum(1 for row in messages[index + 1:] if row.get("role") == "tool")
    return None, 0


class _Holds:
    """Event-gated holds of named (turn, round) model calls, outside the stub lock."""

    def __init__(self, markers):
        self.markers = markers
        self.rules = {}
        self._lock = threading.Lock()

    def add(self, key):
        self.rules[key] = {"arrived": threading.Event(), "release": threading.Event(), "taken": False}
        return self.rules[key]

    def __call__(self, body):
        if not body.get("tools"):
            return
        rule = self.rules.get(_turn_of(body, self.markers))
        if rule is None:
            return
        with self._lock:
            if rule["taken"]:
                return
            rule["taken"] = True
        rule["arrived"].set()
        if not rule["release"].wait(300):
            raise TimeoutError("scenario never released a held model round")


class _Batch1Model(ScriptedStubModel):
    """Scripted turns keyed by owner marker; a step may pin its provider tool_call_id."""

    def __init__(self, respond, gate):
        super().__init__([respond] * 400, gate=gate)
        self._last_step = None

    def _next_step(self, body):
        step = super()._next_step(body)
        self._last_step = step(body) if callable(step) else step
        return self._last_step

    def _answer(self, body, seq):
        self._last_step = None
        kind, message = super()._answer(body, seq)
        step = self._last_step or {}
        if step.get("provider_id") and message.get("tool_calls"):
            message = {**message, "tool_calls": [{**message["tool_calls"][0], "id": step["provider_id"]}]}
        return kind, message


def _get(url, timeout=15):
    with urllib.request.urlopen(url, timeout=timeout) as response:  # noqa: S310 - loopback test server
        return response.read()


def _api(server, path):
    return json.loads(_get(server.base_url + path))


def _tool_rows(server, task_id):
    return [row for row in _api(server, f"/api/logs/tools?task_id={task_id}&limit=400")["entries"]
            if row.get("task_id") == task_id]


def _git(*args):
    return subprocess.run(["git", *args], cwd=REPO, capture_output=True, check=True).stdout


def _candidate_facts(server, candidate):
    """Bind served modules to this captured working candidate, without requiring a commit."""
    served = {}
    for name in SERVED_MODULES:
        body = _get(f"{server.base_url}/static/modules/{name}")
        expected = (REPO / "web/modules" / name).read_bytes()
        assert body == expected, f"served {name} differs from captured source"
        served[name] = hashlib.sha256(body).hexdigest()
    return {"source_head": candidate.state.head.decode().strip(),
            "candidate_identity": candidate.identity, "checkout_identity": candidate.checkout_identity,
            "served_runtime_version": _api(server, "/api/health")["runtime_version"],
            "served_module_sha256": served, "server_pid": server.proc.pid, "server_url": server.base_url}


def _open_fifo_reader(path: Path, timeout=30) -> bytes:
    """Become the blocked writer's reader, drain its one line, and return it."""
    fd = os.open(str(path), os.O_RDONLY | os.O_NONBLOCK)
    data = b""
    deadline = time.monotonic() + timeout
    try:
        while time.monotonic() < deadline:
            try:
                chunk = os.read(fd, 65536)
            except BlockingIOError:
                chunk = None
            if chunk:
                data += chunk
            elif chunk == b"" and data.endswith(b"\n"):
                return data  # the writer wrote its line and closed
            time.sleep(0.05)
    finally:
        os.close(fd)
    raise AssertionError(f"FIFO writer never completed: {data!r}")


def _order_b_rows(data_root: Path, task_id: str, monkeypatch) -> tuple:
    """Settlement BEFORE the wait end, through the production wrapper and writer.

    The only instrumentation: the timeout path waits until the late worker's
    settlement row is durable, making the reversed race deterministic."""
    import ouroboros.loop_tool_execution as execution
    from ouroboros.tools.tool_result import ToolResult
    from tests.test_tool_call_log import _Registry

    release = threading.Event()

    def handler(_name, _args):
        release.wait(timeout=20)
        return ToolResult(status="ok", code="OK", text="Order-B late read result")

    registry = _Registry(data_root, handler, round_id="orderb:round:1")
    registry._ctx.task_metadata = {"budget_drive_root": str(data_root)}
    logs = data_root / "logs"
    original = execution._make_timeout_result

    def settle_first(*args, **kwargs):
        release.set()
        invocation = kwargs["invocation"]["invocation_id"]
        wait_until(lambda: any(row.get("invocation_id") == invocation and row.get("type") == "tool_call"
                               for row in _jsonl(logs / "tools.jsonl")), 20, 0.05)
        return original(*args, **kwargs)

    monkeypatch.setattr(execution, "_make_timeout_result", settle_first)
    try:
        tc = {"id": REPEATED_PROVIDER_ID, "function": {"name": "read_file", "arguments": json.dumps({"path": "VERSION"})}}
        result = execution._execute_with_timeout(registry, tc, logs, 1, task_id)
    finally:
        monkeypatch.setattr(execution, "_make_timeout_result", original)
    assert result["tool_result"].code == "TOOL_TIMEOUT"
    frames = [frame for frame in registry.frames
              if frame.get("type") in {"tool_call_started", "tool_call", "tool_call_timeout"}]
    return frames


def _jsonl(path: Path) -> list:
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return []
    return [json.loads(line) for line in text.splitlines() if line.strip()]


class _Observer:
    """Every page error, console error, failed request and HTTP error, by scenario phase."""

    def __init__(self, page):
        self.phase = "boot1"
        self.rows = []
        self.statuses = {}
        page.on("pageerror", lambda error: self._add("pageerror", str(error)))
        page.on("console", lambda msg: msg.type == "error" and self._add("console", msg.text))
        page.on("requestfailed", lambda req: self._add("requestfailed", f"{req.method} {req.url} {req.failure}", url=req.url))
        page.on("response", self._response)

    def _response(self, resp):
        self.statuses.setdefault(resp.url, []).append(resp.status)
        if resp.status >= 400:
            self._add("http", f"{resp.status} {resp.url}")

    def _add(self, kind, text, url=""):
        self.rows.append({"phase": self.phase, "kind": kind, "text": text, "url": url})

    def classified(self):
        """Chromium reports a body-less 204 fetch as ERR_ABORTED although the page got it."""
        for row in self.rows:
            if (row["kind"] == "requestfailed" and "ERR_ABORTED" in row["text"]
                    and 204 in self.statuses.get(row["url"], [])):
                row["benign"] = "response 204 received for the same URL"
        return self.rows

    def outside(self, phases):
        return [row for row in self.classified() if row["phase"] not in phases and not row.get("benign")]


def _card(page, task_id):
    return page.locator(f'#page-chat .chat-live-card[data-task-id="{task_id}"]')


def _card_text(page, task_id):
    return page.evaluate("id => document.querySelector(`#page-chat .chat-live-card[data-task-id=\"${id}\"]`)?.textContent || ''",
                         task_id)


def _card_lines(page, task_id):
    """Every rendered line of one live card (hidden ones included) with its phase class."""
    return page.evaluate("""id => {
        const card = document.querySelector(`#page-chat .chat-live-card[data-task-id="${id}"]`);
        if (!card) return null;
        return {phase: card.querySelector('[data-live-phase]')?.textContent?.trim() || '',
                lines: [...card.querySelectorAll('.chat-live-line')].map(line => ({
                    cls: line.className, text: line.textContent.replace(/\\s+/g, ' ').trim()}))};
    }""", task_id)


def _expand(page, task_id):
    """Open a live card's timeline, so a screenshot shows its rows, not only the summary."""
    button = _card(page, task_id).locator(":scope > [data-live-summary-button] [data-live-disclosure]")
    button.wait_for(state="attached", timeout=30000)
    if button.get_attribute("aria-expanded") != "true":
        button.click()
    page.wait_for_function("id => document.querySelector(`#page-chat .chat-live-card[data-task-id=\"${id}\"] "
                           "> [data-live-summary-button] [data-live-disclosure]`)?.getAttribute('aria-expanded') === 'true'",
                           arg=task_id, timeout=15000)


def _fold(view):
    return next((line for line in (view or {}).get("lines", []) if " tool call" in line["text"]), None)


def _chip(page):
    # The bar's consciousness chip (consciousness_status.js): its word, its one mark and the words that say it all.
    return page.locator("#page-chat .cx-chip").evaluate(
        "b => ({label: b.getAttribute('aria-label') || '', word: b.querySelector('.cx-chip-word').textContent,"
        " off: b.dataset.off === 'true', shape: b.querySelector('.status-mark').dataset.shape,"
        " hue: b.querySelector('.status-mark').dataset.hue})")


def _wait_chip(page, predicate, timeout=45000):
    page.wait_for_function(
        "source => { const chip = document.querySelector('#page-chat .cx-chip');"
        " return Boolean(chip && !chip.hidden) && new Function('chip', `return ${source}`)(chip); }",
        arg=predicate, timeout=timeout)
    return _chip(page)


# The chip says the host's own sentence for its state (consciousness_view.py).
UNKNOWN_CHIP = "chip.getAttribute('aria-label').startsWith('Consciousness: The stored consciousness control is unknown')"


def _evolution_pills(page):
    page.locator('[data-nav-page="dashboard"]').click()
    page.locator('[data-dashboard-tab="evolution"]').click()
    page.locator("#evo-campaign-chip").wait_for(state="visible", timeout=30000)  # shown once the state answered
    return {"evolution": page.locator("#evo-campaign-chip").inner_text(),
            "consciousness": page.locator("#evo-consciousness").inner_text(),
            "evolution_tone": page.locator("#evo-campaign-chip").get_attribute("data-tone"),
            "consciousness_tone": page.locator("#evo-consciousness").get_attribute("data-tone")}


def _to_chat(page):
    page.locator('[data-nav-page="chat"]').click()
    page.locator("#page-chat.active").wait_for(timeout=15000)


def _send(page, text):
    page.locator("#chat-input").fill(text)
    page.locator("#chat-send").click()


def _direct_task(oracle, marker, timeout=120):
    return wait_until(lambda: next((row["task"] for row in oracle.events("task_received")
                                    if marker in str(row.get("task", {}).get("text", ""))), None), timeout)


def test_batch1_unknown_controls_and_tool_history_reach_the_real_spa(batch1_clone, tmp_path, monkeypatch):
    from playwright.sync_api import sync_playwright

    from ouroboros.tool_call_log import logical_calls

    evidence = producer_evidence_dir(tmp_path / "evidence", "batch1-consumers",
                                     env="OUROBOROS_BROWSER_EVIDENCE_OUT")
    evidence.mkdir(parents=True, exist_ok=True)
    receipt: dict = {"assertions": [], "red": []}

    def passed(name, **facts):
        receipt["assertions"].append({"name": name, **facts})

    def red(name, **facts):
        """A product consumer defect: kept as evidence while the other branches still run."""
        receipt["red"].append({"name": name, **facts})

    root = tmp_path / "instance" / "data"
    root.mkdir(parents=True)
    seeded = _seed_recovering_state(root)
    home = tmp_path / "home"
    home.mkdir()
    original_env = KeylessIsolatedServer._env
    monkeypatch.setattr(KeylessIsolatedServer, "_env", lambda server: {
        **original_env(server), "HOME": str(home), "USERPROFILE": str(home),
        "XDG_CONFIG_HOME": str(home / ".config")})

    markers = {name: f"B1_{name}_{uuid.uuid4().hex}" for name in ("A", "B", "C")}
    fifo = {name: root / "projects" / f"batch1-fifo-{name.lower()}" / "journal.jsonl" for name in ("A", "B")}
    scenario = {"crashed": False, "closed": False}

    def journal(name):
        fifo[name].parent.mkdir(parents=True, exist_ok=True)
        if not fifo[name].exists():
            os.mkfifo(fifo[name])
        return {"tool": "journal_write", "arguments": {
            "kind": "note", "text": f"Batch1 FIFO milestone {name}", "project_id": fifo[name].parent.name}}

    steps = {
        "A": [lambda: {"tool": "read_file", "arguments": {"root": "system_repo", "path": "VERSION"},
                       "provider_id": REPEATED_PROVIDER_ID},
              lambda: {"tool": "read_file", "arguments": {"root": "system_repo", "path": "README.md"},
                       "provider_id": REPEATED_PROVIDER_ID},
              lambda: journal("A"),
              lambda: {"final": "Batch1 turn A is complete."}],
        "B": [lambda: journal("B"), lambda: {"final": "Batch1 turn B is complete."}],
        "C": [lambda: {"final": "Batch1 turn C is complete."}],
    }

    def respond(body):
        turn, index = _turn_of(body, markers)
        if scenario["closed"] or turn is None or (turn == "B" and scenario["crashed"]):
            return {"final": "Nothing further is needed."}
        return steps[turn][index]() if index < len(steps[turn]) else {"final": f"Batch1 turn {turn} is complete."}

    holds = _Holds(markers)
    hold_a = holds.add(("A", 3))
    hold_c = holds.add(("C", 0))
    shots = evidence
    with _Batch1Model(respond, holds) as stub:
        settings_path = root / "settings.json"
        # The owner-settable global cap is only a floor over each tool's own
        # timeout (max(setting, per-tool)); 1 s leaves journal_write its 15 s.
        write_settings_file(settings_path, keyless_settings(stub, OUROBOROS_MAX_WORKERS=1,
                                                            OUROBOROS_TOOL_TIMEOUT_SEC=1))
        server = KeylessIsolatedServer(batch1_clone, root, settings_path)
        server.start(ready_timeout=300)
        oracle = ArtifactOracle(root)
        try:
            receipt["candidate"] = _candidate_facts(server, batch1_clone)
            # -- Boot recovery: the torn primary is preserved, the backup's "on" is not promoted.
            api = _api(server, "/api/state")
            assert api["evolution_enabled"] is None and api["bg_consciousness_enabled"] is None, api
            assert api["state_quality"]["quality"] == "recovered", api["state_quality"]
            assert {"evolution_mode_enabled", "bg_consciousness_enabled"} <= set(api["state_quality"]["unconfirmed"])
            corrupt = sorted((root / "state").glob("state.corrupt-*.json"))
            assert [path.read_bytes() for path in corrupt] == [seeded["torn"]]
            passed("boot_recovery_unknown", api_state={k: api[k] for k in ("evolution_enabled", "bg_consciousness_enabled", "state_quality")},
                   evolution_status=api["evolution_state"].get("status"), bg_status=api["bg_consciousness_state"].get("status"),
                   corrupt_copy=corrupt[0].name)
            with sync_playwright() as pw:
                browser, engine = _launch(pw)
                receipt["browser"] = {"engine": engine, "version": browser.version}
                page = browser.new_page(viewport={"width": 1440, "height": 1000}, reduced_motion="reduce")
                observer = _Observer(page)
                page.add_init_script(f"({_CAPTURE_TEST_SOCKET})()")
                try:
                    page.goto(server.base_url, wait_until="domcontentloaded")
                    page.wait_for_function("() => window.__testSockets?.some(s => s.readyState === WebSocket.OPEN)", timeout=60000)
                    chip = _wait_chip(page, UNKNOWN_CHIP)
                    # An unknown control is never Off: a faint ring and no word, the words say unknown.
                    assert not chip["off"] and chip["shape"] == "ring" and chip["word"] == "", chip
                    page.screenshot(path=str(shots / "01-boot1-unknown-consciousness-chip.png"), animations="disabled")
                    passed("chat_chip_unknown", chip=chip)
                    pills = _evolution_pills(page)
                    assert pills["evolution"] == "Unknown" and pills["consciousness"] == "Consciousness unknown", pills
                    page.screenshot(path=str(shots / "02-boot1-unknown-evolution-panel.png"), animations="disabled")
                    passed("evolution_panel_unknown", pills=pills)
                    # A narrow viewport: the bar keeps the chip whole beside the title and Panic.
                    mobile = browser.new_page(viewport={"width": 390, "height": 844}, has_touch=True, is_mobile=True)
                    mobile_observer = _Observer(mobile)
                    try:
                        mobile.goto(server.base_url, wait_until="domcontentloaded")
                        mobile_chip = _wait_chip(mobile, UNKNOWN_CHIP)
                        fit = mobile.evaluate("""() => [...document.querySelectorAll('#page-chat .cx-chip, '
                            + '#page-chat [data-chat-command="panic"]')].map(b => { const r = b.getBoundingClientRect();
                            return {left: r.left, right: r.right, width: r.width, visible: r.width > 0 && r.height > 0,
                                    viewport: innerWidth}; })""")
                        mobile.screenshot(path=str(shots / "02m-boot1-unknown-consciousness-chip-390.png"), animations="disabled")
                        assert len(fit) == 2 and all(item["visible"] and item["left"] >= 0 and item["right"] <= item["viewport"]
                                                     for item in fit), fit
                        passed("mobile_chip_unknown_fits", chip=mobile_chip, geometry=fit,
                               console_network=mobile_observer.outside(set()))
                        assert not mobile_observer.outside(set()), mobile_observer.rows
                    finally:
                        mobile.close()
                    _to_chat(page)

                    # -- Turn A: repeated provider id, a real wait end, then a real late success.
                    _send(page, f"{markers['A']} read two files and record a journal milestone")
                    task_a = _direct_task(oracle, markers["A"])
                    assert task_a, "the owner message never reached a turn under unknown controls"
                    a_id = task_a["id"]
                    assert hold_a["arrived"].wait(180), "turn A never reached the round after its wait end"
                    rows = _tool_rows(server, a_id)
                    calls = logical_calls(rows)
                    assert [call["tool"] for call in calls] == ["read_file", "read_file", "journal_write"], rows
                    assert [call["state"] for call in calls] == ["settled", "settled", "wait_ended"], calls
                    assert {call["started"]["tool_call_id"] for call in calls[:2]} == {REPEATED_PROVIDER_ID}
                    assert len({call["invocation_id"] for call in calls}) == 3
                    assert calls[2]["started"]["timeout_sec"] == 15 and calls[2]["wait_ended"]["waited_ms"] >= 14000
                    card = _card(page, a_id)
                    card.wait_for(state="attached", timeout=30000)
                    page.wait_for_function("id => (document.querySelector(`#page-chat .chat-live-card[data-task-id=\"${id}\"]`)"
                                           "?.textContent || '').includes('3 tool calls · wait ended')", arg=a_id, timeout=30000)
                    held_text = _card_text(page, a_id)
                    assert "Tool wait ended; operation may still settle · journal_write" in held_text, held_text
                    held = _card_lines(page, a_id)
                    fold = _fold(held)
                    assert fold and "error" not in fold["text"] and " warn" not in fold["cls"] and "calling" not in fold["cls"], held
                    _expand(page, a_id)
                    card.scroll_into_view_if_needed()
                    page.screenshot(path=str(shots / "03-boot1-wait-ended-before-settlement.png"), animations="disabled")
                    passed("live_wait_end_is_not_failure", invocation_ids=[c["invocation_id"] for c in calls],
                           provider_ids=[c["started"]["tool_call_id"] for c in calls])
                    line = _open_fifo_reader(fifo["A"])
                    assert json.loads(line)["text"] == "Batch1 FIFO milestone A"
                    settled = wait_until(lambda: (lambda c: c if c[2]["state"] == "settled" else None)(
                        logical_calls(_tool_rows(server, a_id))), 30)
                    assert settled and settled[2]["settled"]["is_error"] is False
                    assert settled[2]["settled"]["result_preview"].startswith("OK: journal[batch1-fifo-a]")
                    assert settled[2]["wait_ended"]["invocation_id"] == settled[2]["invocation_id"]
                    page.wait_for_function("id => !(document.querySelector(`#page-chat .chat-live-card[data-task-id=\"${id}\"]`)"
                                           "?.textContent || '').includes('operation may still settle')", arg=a_id, timeout=30000)
                    late = _card_lines(page, a_id)
                    assert "3 tool calls · wait ended" in _fold(late)["text"] and "error" not in _fold(late)["text"], late
                    assert not any(" warn" in line["cls"] or " error" in line["cls"] for line in late["lines"]), late
                    page.screenshot(path=str(shots / "04-boot1-late-settlement-while-turn-continues.png"), animations="disabled")
                    passed("late_success_retires_notice_while_task_continues",
                           settlement=settled[2]["settled"]["result_preview"][:80],
                           waited_ms=settled[2]["wait_ended"].get("waited_ms"), elapsed_ms=settled[2]["settled"].get("elapsed_ms"))
                    hold_a["release"].set()
                    result_a = wait_durable_result(oracle, a_id, timeout=120)
                    page.get_by_text("Batch1 turn A is complete.", exact=True).wait_for(timeout=60000)
                    page.wait_for_timeout(1500)
                    done = _card_lines(page, a_id)
                    assert "3 tool calls" in _fold(done)["text"] and "error" not in _fold(done)["text"], done
                    _expand(page, a_id)
                    fold = card.locator(".chat-live-line.expandable").filter(has_text="3 tool calls")
                    if fold.count() and fold.first.is_visible():
                        fold.first.click()
                        page.wait_for_timeout(500)
                    receipt["turn_a_expanded"] = _card_lines(page, a_id)
                    page.screenshot(path=str(shots / "05-boot1-turn-a-terminal-expanded.png"), animations="disabled")
                    execution_a = (result_a.get("outcome_axes") or {}).get("execution") or {}
                    passed("terminal_live_card_counts_one_invocation_each", status=result_a.get("status"),
                           card=done, held=held, late=late,
                           execution_axis={"status": execution_a.get("status"),
                                           "unresolved_tool_errors": execution_a.get("unresolved_tool_errors")})

                    # -- An owner decision makes one control known; the other stays unknown.
                    observer.phase = "owner-commands"
                    _send(page, "/evolve stop")
                    wait_until(lambda: _api(server, "/api/state")["evolution_enabled"] is False, 60)
                    chip = _wait_chip(page, UNKNOWN_CHIP)
                    assert "evolution campaign running" not in chip["label"] and chip["word"] != "Evolving", chip
                    page.screenshot(path=str(shots / "06-boot1-known-disabled-beside-unknown.png"), animations="disabled")
                    passed("known_disabled_distinct_from_unknown", chip=chip,
                           api={k: _api(server, "/api/state")[k] for k in ("evolution_enabled", "bg_consciousness_enabled")})

                    # -- A transiently unreadable primary: display-only backup, no known control.
                    observer.phase = "transient-unreadable"
                    primary = root / "state" / "state.json"
                    primary.chmod(0)
                    try:
                        wait_until(lambda: _api(server, "/api/state")["state_quality"]["quality"] == "recovered_transient", 30)
                        transient = _api(server, "/api/state")
                        assert transient["evolution_enabled"] is None and transient["bg_consciousness_enabled"] is None
                        chip = _wait_chip(page, UNKNOWN_CHIP)
                        page.screenshot(path=str(shots / "07-boot1-transient-unreadable-unknown.png"), animations="disabled")
                        passed("transient_unreadable_is_unknown", state_quality=transient["state_quality"], chip=chip)
                    finally:
                        primary.chmod(0o600)
                    wait_until(lambda: _api(server, "/api/state")["evolution_enabled"] is False, 30)
                    _wait_chip(page, UNKNOWN_CHIP)
                    passed("returning_primary_restores_known_control")
                    observer.phase = "boot1"

                    # -- Turn C, live: the reversed arrival order from the production wrapper.
                    _send(page, f"{markers['C']} answer after a short wait")
                    task_c = _direct_task(oracle, markers["C"])
                    c_id = task_c["id"]
                    assert hold_c["arrived"].wait(120)
                    frames = _order_b_rows(root, c_id, monkeypatch)
                    assert [frame["type"] for frame in frames] == ["tool_call_started", "tool_call", "tool_call_timeout"]
                    durable_c = logical_calls(_tool_rows(server, c_id))
                    assert len(durable_c) == 1 and durable_c[0]["state"] == "settled"
                    order = [row["type"] for row in _tool_rows(server, c_id)]
                    assert order == ["tool_call_started", "tool_call", "tool_call_timeout"], order
                    page.evaluate("""frames => {
                        const socket = window.__testSockets.find(s => s.readyState === WebSocket.OPEN);
                        for (const data of frames) socket.dispatchEvent(new MessageEvent('message',
                            {data: JSON.stringify({type: 'log', chat_id: 1, data: {...data, chat_id: 1}})}));
                    }""", frames)
                    page.wait_for_function("id => (document.querySelector(`#page-chat .chat-live-card[data-task-id=\"${id}\"]`)"
                                           "?.textContent || '').includes('1 tool call · wait ended')", arg=c_id, timeout=30000)
                    c_text = _card_text(page, c_id)
                    c_view = _card_lines(page, c_id)
                    receipt["order_b_card"] = c_view
                    assert "1 tool call · wait ended" in _fold(c_view)["text"] and "error" not in _fold(c_view)["text"], c_view
                    _expand(page, c_id)
                    _card(page, c_id).scroll_into_view_if_needed()
                    page.screenshot(path=str(shots / "08-boot1-order-b-settled-before-wait-end.png"), animations="disabled")
                    passed("order_b_one_invocation_not_failed", durable_order=order,
                           invocation_id=durable_c[0]["invocation_id"], card=c_view)
                    if "operation may still settle" in c_text:
                        (shots / "08-order-b-card-dom.html").write_text(_card(page, c_id).evaluate("n => n.outerHTML"),
                                                                        encoding="utf-8")
                        red("order_b_stale_wait_notice", card=c_view, frames=[f["type"] for f in frames],
                            detail="settlement arrived first; the later wait-end frame still adds the provisional "
                                   "'operation may still settle' warn row, which nothing retires")
                    hold_c["release"].set()
                    wait_durable_result(oracle, c_id, timeout=120)
                    page.get_by_text("Batch1 turn C is complete.", exact=True).wait_for(timeout=60000)
                    page.wait_for_timeout(1500)
                    receipt["order_b_card_after_terminal"] = _card_lines(page, c_id)

                    # -- Turn B: a crash while its call is blocked leaves a start-only invocation.
                    _send(page, f"{markers['B']} record one more journal milestone")
                    task_b = _direct_task(oracle, markers["B"])
                    b_id = task_b["id"]
                    started = wait_until(lambda: [row for row in _tool_rows(server, b_id) if row["type"] == "tool_call_started"], 60, 0.2)
                    assert started and started[0]["tool"] == "journal_write" and started[0]["timeout_sec"] == 15, started
                    page.wait_for_function("id => (document.querySelector(`#page-chat .chat-live-card[data-task-id=\"${id}\"]`)"
                                           "?.textContent || '').includes('1 tool call')", arg=b_id, timeout=10000)
                    _expand(page, b_id)
                    receipt["b_before_crash"] = _card_lines(page, b_id)
                    page.screenshot(path=str(shots / "09-boot1-turn-b-blocked-before-crash.png"), animations="disabled")
                    observer.phase = "crash-restart"
                    scenario["crashed"] = True
                    os.kill(server.proc.pid, signal.SIGKILL)
                    server.proc.wait(timeout=30)
                    rows_b = _jsonl(root / "logs" / "tools.jsonl")
                    assert [row["type"] for row in rows_b if row.get("task_id") == b_id] == ["tool_call_started"], rows_b
                    server.stop()
                    fifo["B"].unlink()
                    passed("crash_left_start_only", invocation_id=started[0]["invocation_id"])

                    # -- Boot 2 on the same root: reconnect, reload, Logs backfill.
                    server.start(ready_timeout=300)
                    receipt["candidate_boot2"] = _candidate_facts(server, batch1_clone)
                    page.wait_for_function("() => window.__testSockets?.some(s => s.readyState === WebSocket.OPEN)", timeout=120000)
                    api2 = _api(server, "/api/state")
                    assert api2["evolution_enabled"] is False and api2["bg_consciousness_enabled"] is None, api2
                    observer.phase = "boot2"
                    receipt["reconnected_b_initial"] = _card_lines(page, b_id)
                    # The restore hands the caught direct turn to the cancel-intent watchdog
                    # (20 s cadence, intents >= 10 s old): wait for ITS terminal, then read the card.
                    booted = time.monotonic()
                    stored_b = wait_until(lambda: (lambda r: r if r.get("status") not in ("running", "", None) else None)(
                        oracle.task_result(b_id)), 150, 1.0)
                    receipt["b_durable_after_restart"] = {
                        "status": (stored_b or oracle.task_result(b_id)).get("status"),
                        "seconds_after_boot2": round(time.monotonic() - booted, 1)}
                    assert stored_b, receipt["b_durable_after_restart"]
                    try:
                        page.wait_for_function("id => !['Working', 'Cancelling…'].includes((document.querySelector("
                                               "`#page-chat .chat-live-card[data-task-id=\"${id}\"] [data-live-phase]`)"
                                               "?.textContent || '').trim())", arg=b_id, timeout=60000)
                    except Exception:
                        pass
                    reconnected_b = _card_lines(page, b_id)
                    if _card(page, b_id).count():
                        _card(page, b_id).screenshot(path=str(shots / "10b-boot2-reconnected-turn-b-card.png"), animations="disabled")
                    if reconnected_b and (reconnected_b["phase"] in ("Working", "Cancelling…")
                                          or any("calling" in line["cls"] for line in reconnected_b["lines"])):
                        red("kept_open_terminal_card_keeps_interrupted_call_calling", card=reconnected_b,
                            durable=receipt["b_durable_after_restart"])
                    page.screenshot(path=str(shots / "10-boot2-reconnected-same-page.png"), animations="disabled")
                    page.reload(wait_until="domcontentloaded")
                    page.get_by_text("Batch1 turn A is complete.", exact=True).wait_for(timeout=60000)
                    _wait_chip(page, UNKNOWN_CHIP)
                    page.wait_for_timeout(3000)
                    for label, task in (("a", a_id), ("b", b_id), ("c", c_id)):
                        if _card(page, task).count():
                            _expand(page, task)
                            _card(page, task).scroll_into_view_if_needed()
                            _card(page, task).screenshot(path=str(shots / f"11{label}-boot2-reloaded-turn-{label}-card.png"),
                                                         animations="disabled")
                    reloaded = {task: _card_lines(page, task) for task in (a_id, b_id, c_id)}
                    page.screenshot(path=str(shots / "11-boot2-reloaded-chat.png"), full_page=True, animations="disabled")
                    receipt["reload"] = {"chip": _chip(page), "cards": reloaded, "reconnected_b": reconnected_b}
                    a_reload_text = _card_text(page, a_id)
                    if re.search(r"\b\d+ errors?\b", a_reload_text) or (_fold(reloaded[a_id]) or {}).get("cls", "").find("warn") >= 0:
                        red("reload_turn_a_timeout_counted_as_error", card=reloaded[a_id],
                            detail="after reload the card carries only host round totals; the durable late success "
                                   "in tools.jsonl does not reach Chat")
                    view_b = reloaded[b_id]
                    if view_b and (view_b["phase"] in ("Working", "Cancelling…")
                                   or any("calling" in line["cls"] for line in view_b["lines"])):
                        red("reloaded_start_only_turn_still_live", card=view_b, durable=receipt["b_durable_after_restart"])
                    served_calls = {task: logical_calls(_tool_rows(server, task)) for task in (a_id, b_id, c_id)}
                    assert [call["state"] for call in served_calls[a_id]] == ["settled"] * 3
                    assert [call["state"] for call in served_calls[b_id]] == ["unknown"]
                    assert [call["state"] for call in served_calls[c_id]] == ["settled"]
                    passed("gateway_history_logical_calls",
                           states={task: [(c["tool"], c["state"], "wait_ended" in c) for c in calls] for task, calls in served_calls.items()})
                    page.locator('[data-nav-page="dashboard"]').click()
                    page.locator('[data-dashboard-tab="logs"]').click()
                    page.wait_for_function("ids => ids.every(id => document.querySelector(`[data-task-group=\"${id}\"]`))",
                                           arg=[a_id, b_id, c_id], timeout=60000)
                    logs_view = page.evaluate("""ids => Object.fromEntries(ids.map(id => {
                        const card = document.querySelector(`[data-task-group="${id}"]`);
                        return [id, {headline: card.querySelector('[data-task-headline]').textContent,
                                     phase: card.querySelector('[data-task-phase]').textContent,
                                     timeline: [...card.querySelectorAll('.log-task-event .log-headline, .log-task-event .log-main')]
                                        .map(node => node.textContent.replace(/\\s+/g, ' ').trim())}];
                    }))""", [a_id, b_id, c_id])
                    for task in (a_id, b_id):
                        group = page.locator(f'[data-task-group="{task}"]')
                        group.locator("details.log-task-details").evaluate("d => { d.open = true; }")
                    page.locator(f'[data-task-group="{b_id}"]').scroll_into_view_if_needed()
                    page.screenshot(path=str(shots / "12-boot2-logs-backfill.png"), full_page=True, animations="disabled")
                    receipt["logs_view"] = logs_view
                    _to_chat(page)

                    # -- Known enabled and known disabled after a real owner decision.
                    observer.phase = "owner-commands"
                    _send(page, "/bg start")
                    wait_until(lambda: _api(server, "/api/state")["bg_consciousness_enabled"] is True, 60)
                    scenario["closed"] = True
                    chip_on = _wait_chip(page, "chip.dataset.off === 'false' && !chip.getAttribute('aria-label').includes('is unknown')")
                    assert chip_on["word"] != "Off" and "evolution campaign running" not in chip_on["label"], chip_on
                    page.screenshot(path=str(shots / "13-boot2-known-enabled-and-disabled.png"), animations="disabled")
                    _send(page, "/bg stop")
                    wait_until(lambda: _api(server, "/api/state")["bg_consciousness_enabled"] is False, 60)
                    chip_off = _wait_chip(page, "chip.dataset.off === 'true'")
                    assert chip_off["word"] == "Off", chip_off
                    assert chip_off["label"] == "Consciousness: Background consciousness is off.", chip_off
                    pills2 = _evolution_pills(page)
                    page.screenshot(path=str(shots / "14-boot2-known-controls-evolution-panel.png"), animations="disabled")
                    passed("known_enabled_and_disabled_distinct", chip_on=chip_on, chip_off=chip_off, pills=pills2)
                    observer.phase = "boot2"
                    receipt["observer"] = observer.classified()
                    unexpected = observer.outside({"crash-restart"})
                    receipt["unexpected_console_network"] = unexpected
                    assert not unexpected, unexpected
                    assert not receipt["red"], json.dumps(receipt["red"], ensure_ascii=False)[:4000]
                except Exception:
                    page.screenshot(path=str(shots / "failure.png"), full_page=True, animations="disabled")
                    (shots / "failure-dom.html").write_text(page.content(), encoding="utf-8")
                    receipt["observer"] = observer.rows
                    raise
                finally:
                    for rule in (hold_a, hold_c):
                        rule["release"].set()
                    browser.close()
        finally:
            for rule in (hold_a, hold_c):
                rule["release"].set()
            for path in fifo.values():
                if path.exists() and not path.is_file():
                    try:
                        _open_fifo_reader(path, timeout=2)
                    except Exception:
                        pass
            server.stop()
            (evidence / "receipt.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2, default=str),
                                                   encoding="utf-8")
