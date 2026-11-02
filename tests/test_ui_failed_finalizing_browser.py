"""A REAL root actor fails on a provider refusal while its post-task synthesis
is still open (#1110/#1011): the Main card reads the outcome first ("Failed")
and "Finalizing…" only as the secondary phase — live, after a reload, after a
real WebSocket reconnect and on a narrow light page — then settles once.

Only model judgment is a fixture. The provider refusal is a real HTTP 401 on
the model wire, the host's provider-unavailable rail preserves intermediate work,
and the one event-held call is the root's post-task evolution chooser, which
this fixture's owner settings enable (identified by the production prompt's first
line — there is no paid summary, no reflection and, since #918, no scratchpad
compression), so synthesis is provably open while the browser looks. Nothing is
timed: every wait is an event or a bounded DOM poll.
"""
import json

import pytest

from devtools.benchmarks.common.server_runner import _api
from ouroboros.contracts.chat_id_policy import WEB_UI_CHAT_ID
from tests._shared import producer_evidence_dir
from tests.test_owner_wait_integration import wait_clone as clone_fixture
from tests.system_e2e.harness import (
    ArtifactOracle, KeylessIsolatedServer, ModelGate, ScriptedStubModel, body_text,
    keyless_settings, wait_until, write_settings_file,
)

wait_clone = clone_fixture
pytestmark = [pytest.mark.serial, pytest.mark.browser]

MARKER = "FAILED_FINALIZING_REAL_ACTOR"
CHOOSER_MARKER = "You decide whether Ouroboros should run ONE reviewed self-improvement"  # post_task_evolution._DECISION_PROMPT
SALVAGE_MARKER = "[PROVIDER_UNAVAILABLE]"  # ouroboros/loop.py::_provider_unavailable_result
NEUTRAL = "Nothing further to record for this fixture."
TITLE = "Provider outage proof"
CARD = '#chat-messages .chat-live-card[data-task-id="{}"]'

STATE_JS = """tid => {
    const cards = document.querySelectorAll(`#chat-messages .chat-live-card[data-task-id="${tid}"]`);
    const card = cards[0];
    if (!card) return null;
    const chip = card.querySelector('[data-live-phase]');
    const second = card.querySelector('[data-live-phase-secondary]');
    const style = el => el ? getComputedStyle(el) : null;
    const mark = chip.querySelector(':scope > .status-mark');
    return {count: cards.length, finished: card.dataset.finished, phase: chip.dataset.phase,
            chip: chip.textContent, chipHidden: chip.hidden, label: chip.getAttribute('aria-label'),
            mark: mark ? `${mark.dataset.shape}/${mark.dataset.hue}` : '', tone: chip.dataset.chipTone,
            secondary: second.textContent, secondaryHidden: second.hidden,
            title: card.querySelector('[data-live-title]')?.textContent || '',
            theme: document.documentElement.dataset.theme || '',
            chipColor: style(chip).color, chipBackground: style(chip).backgroundColor,
            secondaryColor: style(second).color,
            overflow: card.scrollWidth > card.clientWidth + 1
                || second.getBoundingClientRect().right > innerWidth + 1};
}"""
HOLDING_JS = """tid => { const c = document.querySelector(`#chat-messages .chat-live-card[data-task-id="${tid}"]`);
    const chip = c?.querySelector('[data-live-phase]'), s = c?.querySelector('[data-live-phase-secondary]');
    return Boolean(c && chip.textContent === 'Failed' && !s.hidden && s.textContent === 'Finalizing…'); }"""
SETTLED_JS = """tid => { const c = document.querySelector(`#chat-messages .chat-live-card[data-task-id="${tid}"]`);
    const s = c?.querySelector('[data-live-phase-secondary]');
    return Boolean(c && c.dataset.finished === '1' && s.hidden); }"""
# Settled work claims no activity: the header returns to idle with the card.
HEADER_IDLE_JS = "() => document.querySelector('#chat-status')?.textContent === 'Online'"
OBSERVE_JS = """tid => { const card = document.querySelector(`#chat-messages .chat-live-card[data-task-id="${tid}"]`);
    const chip = card.querySelector('[data-live-phase]'), second = card.querySelector('[data-live-phase-secondary]');
    window.__proofCard = card; window.__phaseLog = [];
    const note = () => window.__phaseLog.push([chip.textContent, second.hidden ? '' : second.textContent, card.dataset.finished]);
    window.__phaseObserver = new MutationObserver(note);
    window.__phaseObserver.observe(card, {subtree: true, childList: true, characterData: true,
                                          attributes: true, attributeFilter: ['hidden', 'data-finished', 'data-phase']}); }"""


class _OutageModel(ScriptedStubModel):
    """One real (failing) tool round, then a provider refusal (HTTP 401, a permanent
    class, so no backoff retries) on every later tool round of the marked task AND on
    the host's forced outage final (``[PROVIDER_UNAVAILABLE]``): the provider is down
    for that call too, so the host's terminal incident preserves the intermediate
    output (``host_salvage``). The post-task evolution chooser is never refused —
    it is the one call the gate holds."""

    def __init__(self, gate):
        super().__init__([{"tool": "run_command", "arguments": {
            "cmd": ["python", "-c", "import sys; sys.exit(7)"]}}],
                         final_answer=NEUTRAL, gate=gate)
        self.refused = 0
        outer, base = self, self._server.RequestHandlerClass

        class Handler(base):
            def do_POST(self):  # noqa: N802 - stdlib callback name
                raw = self.rfile.read(int(self.headers.get("Content-Length") or 0))
                try:
                    body = json.loads((raw or b"{}").decode("utf-8"))
                except ValueError:
                    body = {}
                body = body if isinstance(body, dict) else {}
                if outer._refuse(body):
                    return self._send({"error": {"message": "Invalid API key (fixture provider outage)",
                                                 "type": "invalid_request_error", "code": "invalid_api_key"}},
                                      status=401)
                outer.gate(body)
                return self._send(outer._completion(body), stream=bool(body.get("stream")))

        self._server.RequestHandlerClass = Handler

    def _answer(self, body, seq):
        if CHOOSER_MARKER in body_text(body):
            # The held late-phase call answers as the production prompt asks, so the
            # phase completes instead of recording a failed chooser stage.
            return "chooser", {"role": "assistant", "content": json.dumps(
                {"promote": False, "objective": "", "requires_plan_review": False, "backlog_id": ""})}
        return super()._answer(body, seq)

    def _refuse(self, body):
        text = body_text(body)
        if MARKER not in text or CHOOSER_MARKER in text:
            return False
        later_tool_round = bool(body.get("tools")) and any(
            isinstance(m, dict) and m.get("role") == "tool" for m in body.get("messages") or [])
        if not (later_tool_round or SALVAGE_MARKER in text):
            return False
        with self._lock:
            self.refused += 1
            self.calls.append(("refused_401", body))
        return True


def _open(browser, url, *, theme, viewport):
    context = browser.new_context(viewport=viewport, has_touch=viewport["width"] < 500,
                                  reduced_motion="reduce")
    context.add_init_script(f"try {{ localStorage.setItem('ouroboros.theme', '{theme}'); }} catch {{}}")
    page = context.new_page()
    page.goto(url, wait_until="domcontentloaded")
    return context, page


def _synthesis(row):
    return str((row.get("root_phase_checkpoint") or {}).get("post_task_synthesis") or "")


def _rows(oracle, task_id):
    return oracle.task_result(task_id), oracle.task_drive(task_id).task_result(task_id)


def _brief(row):
    return {key: row.get(key) for key in ("status", "reason_code", "root_phase_checkpoint", "outcome_axes")}


def _holding(page, task_id, stage):
    page.wait_for_function(HOLDING_JS, arg=task_id, timeout=60000)
    state = page.evaluate(STATE_JS, task_id)
    # Outcome owns the chip; the running finalization is only the secondary fact.
    assert state["count"] == 1 and state["finished"] == "0", (stage, state)
    assert state["chip"] == "Failed" and not state["chipHidden"], (stage, state)
    # The outcome keeps its failed dot in its red body; the running finalization moves nothing beside it.
    assert (state["mark"], state["tone"]) == ("dot/failed", "failed"), (stage, state)
    assert state["label"] == "Task status: Failed, Finalizing…", (stage, state)
    # The name stays the task's own; the outcome is never smuggled into it.
    assert state["title"] == TITLE, (stage, state)
    assert page.locator(CARD.format(task_id) + " [data-live-phase-secondary]").is_visible(), stage
    # The header names the wrap-up of ended work, never "Working" (12.10: ~19 min of "Working").
    page.wait_for_function("() => document.querySelector('#chat-status')?.textContent === 'Finalizing…'",
                           timeout=60000)
    return state


def test_failed_root_reads_failed_then_finalizing(wait_clone, tmp_path, monkeypatch):
    from playwright.sync_api import sync_playwright

    root = tmp_path / "instance" / "data"
    root.mkdir(parents=True)
    fake_home = tmp_path / "home"
    fake_home.mkdir()
    original_env = KeylessIsolatedServer._env
    monkeypatch.setattr(KeylessIsolatedServer, "_env", lambda server: {
        **original_env(server), "HOME": str(fake_home), "USERPROFILE": str(fake_home),
        "XDG_CONFIG_HOME": str(fake_home / ".config")})
    # Only diagnostics leave the managed scratch; the fake HOME and server inputs do not.
    shots = producer_evidence_dir(tmp_path / "screenshots", "failed-finalizing")
    # The owner-enabled post-task chooser: the root's late phase buys that one call.
    gate = ModelGate(lambda body: CHOOSER_MARKER in body_text(body), timeout=300)
    facts = {}
    with _OutageModel(gate) as model:
        settings_path = root / "settings.json"
        # The chooser runs only outside Light access, with the owner's post-task evolution on.
        write_settings_file(settings_path, keyless_settings(model, OUROBOROS_MAX_WORKERS=2, OUROBOROS_RUNTIME_MODE="advanced",
                                                            OUROBOROS_POST_TASK_EVOLUTION="true"))
        server = KeylessIsolatedServer(wait_clone, root, settings_path)
        server.start(ready_timeout=120)
        oracle = ArtifactOracle(root)
        try:
            with sync_playwright() as pw:
                browser = pw.chromium.launch()
                try:
                    desk_ctx, desk = _open(browser, server.base_url, theme="dark",
                                           viewport={"width": 1440, "height": 1000})
                    # An owner's open Main: the admission name frame must reach a live socket.
                    desk.wait_for_function("() => window.__ouroWs?.ws?.readyState === 1", timeout=60000)
                    created = _api(server.base_url, "POST", "/api/tasks", {
                        "description": f"{MARKER}: run one diagnostic command, then report its outcome.",
                        "title": TITLE, "chat_id": WEB_UI_CHAT_ID, "source": "web",
                        "memory_mode": "forked", "metadata": {"delegation_role": "root"}})
                    task_id = str(created.get("task_id") or "")
                    assert task_id, created
                    # The held chooser call IS the open synthesis: the early final already left.
                    assert gate.arrived.wait(180), (model.kinds(), oracle.task_result(task_id))
                    # The worker's own (forked) row already settled Failed; the canonical
                    # row stays live until task_done, and carries the open checkpoint.
                    canonical, forked = _rows(oracle, task_id)
                    facts["held_rows"] = {"canonical": _brief(canonical), "forked": _brief(forked)}
                    assert forked.get("status") == "failed", facts["held_rows"]
                    assert "running" in (_synthesis(canonical), _synthesis(forked)), facts["held_rows"]
                    assert model.refused >= 1
                    card = desk.locator(CARD.format(task_id))
                    facts["live"] = _holding(desk, task_id, "live")
                    card.screenshot(animations="disabled", path=str(shots / "chromium-failed-finalizing-live.png"))

                    desk.evaluate("window.__beforeReload = true")
                    desk.reload(wait_until="domcontentloaded")
                    facts["reload"] = _holding(desk, task_id, "reload")
                    assert desk.evaluate("window.__beforeReload === undefined"), "reload kept the old document"
                    card.screenshot(animations="disabled", path=str(shots / "chromium-failed-finalizing-reload.png"))

                    # A real socket close through the production client: first bind the
                    # serving SHA after reload, so an unknown-SHA recovery is not
                    # mistaken for a same-document reconnect (the #1196 test seam).
                    desk.wait_for_function("() => Boolean(window.__ouroWs?._lastSha)", timeout=30000)
                    desk.wait_for_function("() => window.__ouroWs?.ws?.readyState === 1", timeout=30000)
                    desk.evaluate("window.__sameDocument = true; window.__oldSocket = window.__ouroWs.ws")
                    with desk.expect_response(lambda r: "/api/chat/history" in r.url, timeout=60000):
                        desk.evaluate("window.__oldSocket.close()")
                    desk.wait_for_function("() => window.__ouroWs.ws && window.__ouroWs.ws !== window.__oldSocket"
                                           " && window.__ouroWs.ws.readyState === 1", timeout=30000)
                    facts["reconnect"] = _holding(desk, task_id, "reconnect")
                    assert desk.evaluate("window.__sameDocument === true"), "reconnect reloaded the window"
                    card.screenshot(animations="disabled", path=str(shots / "chromium-failed-finalizing-reconnect.png"))

                    narrow_ctx, narrow = _open(browser, server.base_url, theme="light",
                                               viewport={"width": 390, "height": 844})
                    facts["narrow_light"] = light = _holding(narrow, task_id, "narrow-light")
                    dark = facts["reconnect"]
                    assert (dark["theme"], light["theme"]) == ("dark", "light"), (dark, light)
                    assert (light["chipColor"], light["chipBackground"]) != (dark["chipColor"], dark["chipBackground"])
                    assert light["secondaryColor"] != dark["secondaryColor"], (dark, light)
                    assert not light["overflow"], light
                    narrow.locator(CARD.format(task_id)).scroll_into_view_if_needed()
                    narrow.screenshot(animations="disabled", path=str(shots / "chromium-390-failed-finalizing.png"))
                    narrow_ctx.close()

                    # Only now may synthesis close; the live card must settle exactly once.
                    assert "running" in (_synthesis(r) for r in _rows(oracle, task_id)), _rows(oracle, task_id)
                    desk.evaluate(OBSERVE_JS, task_id)
                    gate.release.set()
                    # Synthesis closes its checkpoint before the supervisor records task_done
                    # (through the event queue), so the settled row is the one carrying both.
                    settled = wait_until(lambda: (row if _synthesis(row := oracle.task_result(task_id)) == "completed"
                                                  and row.get("status") != "running" else None), 120)
                    assert settled and settled["status"] == "failed", oracle.task_result(task_id)
                    desk.wait_for_function(SETTLED_JS, arg=task_id, timeout=60000)
                    desk.wait_for_function(HEADER_IDLE_JS, timeout=60000)
                    facts["settled"] = state = desk.evaluate(STATE_JS, task_id)
                    assert state["count"] == 1 and state["chip"] == "Failed" and state["secondary"] == "", state
                    assert state["title"] == TITLE, state
                    assert state["label"] == "Task status: Failed", state
                    facts["phase_log"] = log = desk.evaluate("window.__phaseLog")
                    assert desk.evaluate("window.__proofCard.isConnected"), "the card was replaced, not settled"
                    assert {chip for chip, _second, _finished in log} <= {"Failed"}, log
                    finished = [f for _c, _s, f in log]
                    assert "1" in finished and "0" not in finished[finished.index("1"):], log
                    # Provider death preserves intermediate work as a host incident,
                    # not as a model-authored salvage answer. The fixture does not
                    # assert a forced call returned when the provider cannot answer.
                    assert settled.get("terminal_origin") == "host_salvage", settled
                    assert not settled.get("final_answer"), settled
                    incidents = [row for row in oracle._jsonl("logs/chat.jsonl", type_filter="terminal_incident")
                                 if row.get("task_id") == task_id]
                    assert len(incidents) == 1 and "intermediate output" in incidents[0]["text"], incidents
                    card.screenshot(animations="disabled", path=str(shots / "chromium-failed-settled.png"))

                    # Replay after synthesis closed settles too: no stale Finalizing….
                    desk.reload(wait_until="domcontentloaded")
                    desk.wait_for_function(SETTLED_JS, arg=task_id, timeout=60000)
                    desk.wait_for_function(HEADER_IDLE_JS, timeout=60000)
                    facts["settled_reload"] = again = desk.evaluate(STATE_JS, task_id)
                    assert again["chip"] == "Failed" and again["secondary"] == "" and again["count"] == 1, again
                    assert again["title"] == TITLE, again
                    desk_ctx.close()
                finally:
                    browser.close()
            facts.update(task_id=task_id, refused=model.refused, gate={"matched": gate.matched, "held": gate.held,
                         "timed_out": gate.timed_out}, kinds=model.kinds(), server_pid=server.proc.pid,
                         execution=settled.get("outcome_axes", {}).get("execution"),
                         reason_code=settled.get("reason_code"), status=settled.get("status"))
            (shots / "journey.json").write_text(json.dumps(facts, indent=2, default=str))
            assert gate.held == 1 and not gate.timed_out
        finally:
            gate.release.set()
            server.stop()
            assert server.proc.poll() is not None
