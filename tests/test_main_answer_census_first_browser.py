"""#1993/#1921: a Main question copy answered there while the census beats every confirmation.

The real isolated server answers, persists and discharges a Project's required question while the
page holds only transport: the press's own reply and this quiz's live frames wait in the page, the
exact detail read waits until the census has applied, and the census goes out once the backend has
discharged the answer and finished the task. Nothing is fabricated: every held body is the server's
own, released unchanged and in order. The copy that took the press must stay the same node through
the census without claiming an answer it has not seen, show the recorded winner and words for its
five seconds once real evidence arrives, then leave once; the Project keeps the record, also after a
fresh reopen. The ACK-before-census order of the same journey stays in tests/test_ui_result_browser.py.
"""
import json
import uuid

import pytest

from devtools.benchmarks.common.server_runner import _api
from tests.test_owner_wait_integration import wait_clone as clone_fixture
from tests.test_native_owner_wait_browser import chat_connection
from tests._shared import producer_evidence_dir
from tests.ui_console_errors import PageErrors
from tests.test_chat_history_paging_browser import _OBSERVE_HISTORY
from tests.test_ui_result_browser import _journey_history_painted
from tests.system_e2e.harness import (
    ArtifactOracle, KeylessIsolatedServer, ScriptedStubModel, keyless_settings,
    wait_durable_result, wait_until, write_settings_file,
)

wait_clone = clone_fixture
pytestmark = [pytest.mark.serial, pytest.mark.browser]

WINNER = "Both sources"
COMMENT = "Use the replication data as well."
ANSWER_LINE = f"Owner's answer: {COMMENT}"

# Transport holds for one armed task/quiz, installed before the SPA; it composes with
# _OBSERVE_HISTORY's fetch. A request is only delayed, never answered here, and every held
# body is released unchanged and in order. `segments` is the copy as each animation frame saw it.
_INVERSE_ORDER = r"""() => {
    const nativeFetch = window.fetch.bind(window);
    const NativeSocket = window.WebSocket;
    const now = () => performance.now();
    const latch = () => {
        const gate = { openedAt: null };
        gate.promise = new Promise((resolve) => {
            gate.open = () => { if (gate.openedAt === null) { gate.openedAt = now(); resolve(); } };
        });
        return gate;
    };
    // Census reads sent before arming: one answered after the press could carry its answer.
    window.__ungatedStateReads = 0;
    let s = null;
    window.__inverseState = () => s;
    const parse = (text) => { try { return JSON.parse(text); } catch { return null; } };
    const urlOf = (input) => new URL(typeof input === 'string' ? input : input.url, location.href);
    const methodOf = (input, init) => String(init?.method || (typeof input === 'object' && input?.method) || 'GET').toUpperCase();

    async function ack(input, init) {
        const entry = { requestedAt: now(), request: parse(init?.body) };
        s.acks.push(entry);
        let response;
        try { response = await nativeFetch(input, init); } catch (error) { entry.error = String(error); throw error; }
        Object.assign(entry, { status: response.status, respondedAt: now() });
        try { entry.reply = await response.clone().json(); } catch { entry.reply = null; }
        // The reply has not "arrived" until released; an abort meanwhile rejects as a real fetch would.
        await new Promise((resolve, reject) => {
            const signal = init?.signal;
            const abort = () => { entry.abortedAt = now(); reject(new DOMException('The operation was aborted.', 'AbortError')); };
            if (signal?.aborted) { abort(); return; }
            signal?.addEventListener?.('abort', abort, { once: true });
            s.gates.ack.promise.then(() => { signal?.removeEventListener?.('abort', abort); resolve(); });
        });
        entry.deliveredAt = now();
        return response;
    }

    // What one census says about the armed question: its row, or its task's live activity row.
    const facts = (row) => row && { quiz_state: row.quiz_state ?? null, answered_index: row.answered_index ?? null,
        comment: row.comment ?? null, question_delivery: row.question_delivery ?? null,
        owner_wait_state: row.owner_wait_state ?? null };
    function summarize(body) {
        const ours = (row) => row && String(row.task_id) === s.tid && String(row.quiz_id) === s.qid;
        const rows = Array.isArray(body?.current_questions) ? body.current_questions : [];
        const activities = Array.isArray(body?.active_chat_activities) ? body.active_chat_activities : [];
        return { complete: body?.current_questions_complete === true, gaps: body?.current_questions_gaps ?? null,
            named: facts(rows.find(ours) || null), required: activities.map((row) => row?.required_question).filter(ours).map(facts),
            rows: rows.length, activities: activities.length };
    }

    async function census(input, init) {
        const entry = { seq: s.census.length + 1, requestedAt: now() };
        s.census.push(entry);
        await s.gates.census.promise;
        entry.sentAt = now();
        const response = await nativeFetch(input, init);
        Object.assign(entry, { status: response.status, respondedAt: now() });
        if (!response.ok) return response;
        let body = null;
        try { body = await response.clone().json(); } catch { /* the reader reports it */ }
        entry.summary = summarize(body);
        entry.omits = entry.summary.complete && !entry.summary.named;
        const json = response.json.bind(response);
        response.json = () => json().then((value) => {
            entry.consumedAt = now();
            // The reader applies in this same turn; look once that turn and two frames have passed.
            setTimeout(() => requestAnimationFrame(() => requestAnimationFrame(() => {
                entry.after = s.describe(); entry.afterAt = now();
            })), 0);
            return value;
        });
        return response;
    }

    async function detail(input, init) {
        const entry = { requestedAt: now() };
        s.details.push(entry);
        await s.gates.detail.promise;
        entry.sentAt = now();
        try {
            const response = await nativeFetch(input, init);
            Object.assign(entry, { status: response.status, respondedAt: now() });
            try {
                const body = await response.clone().json();
                const block = body?.owner_quiz?.[s.qid] || null;
                entry.block = block && { state: block.state, answered_index: block.answered_index, comment: block.comment,
                    delivery: block.delivery?.state ?? null };
                entry.room = body?.question_rooms?.[s.qid] ?? null;
            } catch { /* the reader reports it */ }
            return response;
        } catch (error) { entry.error = String(error); throw error; }
    }

    window.fetch = async (input, init) => {
        const url = urlOf(input);
        if (!s) {
            if (url.pathname !== '/api/state') return nativeFetch(input, init);
            window.__ungatedStateReads += 1;
            try { return await nativeFetch(input, init); } finally { window.__ungatedStateReads -= 1; }
        }
        if (url.pathname === '/api/decisions' && methodOf(input, init) === 'POST') return ack(input, init);
        if (url.pathname === '/api/state') return census(input, init);
        if (url.pathname === `/api/tasks/${encodeURIComponent(s.tid)}`) return detail(input, init);
        return nativeFetch(input, init);
    };

    window.WebSocket = class extends NativeSocket {
        constructor(...args) {
            super(...args);
            this.addEventListener('open', () => s?.sockets.push({ at: now(), event: 'open' }));
            this.addEventListener('close', () => s?.sockets.push({ at: now(), event: 'close' }));
        }
        set onmessage(handler) {
            const socket = this;
            super.onmessage = typeof handler !== 'function' ? handler : function (event) {
                const msg = s ? parse(event.data) : null;
                if (msg && String(msg.task_id || '') === s.tid) {
                    const record = { at: now(), type: msg.type ?? null, system_type: msg.system_type ?? null,
                        quiz_id: msg.quiz_id ?? null, state: msg.state ?? msg.quiz_state ?? null,
                        question_delivery: msg.question_delivery ?? null,
                        held: String(msg.quiz_id || '') === s.qid && s.gates.frames.openedAt === null };
                    s.frames.push(record);
                    if (record.held) {
                        record.msg = msg;
                        s.held.push(() => { record.deliveredAt = now(); handler.call(socket, event); });
                        return undefined;
                    }
                }
                return handler.call(this, event);
            };
        }
        get onmessage() { return super.onmessage; }
    };

    window.__armInverseOrder = (tid, qid, winner, answerLine) => {
        const selector = `#chat-messages .project-question-card[data-task-id="${CSS.escape(tid)}"][data-quiz-id="${CSS.escape(qid)}"]`;
        s = { tid, qid, winner, answerLine, selector, armedAt: now(),
            node: document.querySelector(selector)?.closest('.chat-bubble') || null,
            gates: { ack: latch(), census: latch(), detail: latch(), frames: latch() }, held: [], released: {},
            acks: [], census: [], details: [], frames: [], sockets: [], detaches: [], reattaches: [], remounts: [],
            segments: [], youAnsweredAt: null, stopped: false, refs: null, beforeDetach: null, afterDetach: null };
        const feed = document.querySelector('#chat-messages');
        const active = () => { const el = document.activeElement;
            return { tag: el?.tagName ?? null, id: el?.id ?? '', connected: el?.isConnected === true,
                body: !el || el === document.body, composer: el?.id === 'chat-input',
                card: el?.closest?.('.chat-quiz-card')?.dataset.quizId ?? null }; };
        // Where the rows beside the copy stand, and where the reader is.
        const place = () => ({ at: now(), tops: (s.refs || []).map((ref) => ref.isConnected ? ref.getBoundingClientRect().top : null),
            scrollTop: feed.scrollTop, overflow: feed.scrollHeight > feed.clientHeight + 1, active: active() });
        s.describe = () => {
            const card = document.querySelector(selector);
            const bubble = card?.closest('.chat-bubble') || null;
            const rect = s.node?.isConnected ? s.node.getBoundingClientRect() : null;
            const options = card ? [...card.querySelectorAll('.chat-quiz-option')] : [];
            const field = card?.querySelector('.chat-quiz-comment');
            return { present: Boolean(card), same: Boolean(bubble) && bubble === s.node, connected: s.node?.isConnected === true,
                state: card?.dataset.state ?? null, submission: card?.dataset.submission ?? null,
                status: card?.querySelector('.chat-quiz-status-text')?.textContent ?? null,
                chosen: options.filter((o) => o.classList.contains('chosen'))
                    .map((o) => o.querySelector('.chat-quiz-option-label')?.textContent ?? ''),
                answer: card?.querySelector('.chat-quiz-answer')?.textContent ?? null,
                focusIn: Boolean(s.node?.isConnected) && s.node.contains(document.activeElement),
                actionable: options.filter((o) => !o.disabled && o.getAttribute('aria-disabled') !== 'true').length
                    + (field && !field.disabled && !field.readOnly ? 1 : 0),
                visible: Boolean(rect) && rect.bottom > 0 && rect.top < innerHeight };
        };
        s.release = (which) => {
            s.released[which] = now();
            s.gates[which].open();
            if (which === 'frames') for (const deliver of s.held.splice(0)) deliver();
        };
        s.snapshot = () => {
            const { node, gates, held, describe, release, snapshot, refs, ...plain } = s;
            return JSON.parse(JSON.stringify({ ...plain, view: describe(), heldCount: held.length,
                gates: Object.fromEntries(Object.entries(gates).map(([key, gate]) => [key, gate.openedAt])) }));
        };
        new MutationObserver((records) => {
            for (const record of records) {
                for (const node of record.removedNodes) if (node === s.node || node.contains?.(s.node)) {
                    s.detaches.push(now());
                    requestAnimationFrame(() => requestAnimationFrame(() => { s.afterDetach ||= place(); }));
                }
                for (const node of record.addedNodes) {
                    if (node.nodeType !== 1) continue;
                    if (node === s.node || node.contains?.(s.node)) s.reattaches.push(now());
                    else if (node.matches?.(selector) || node.querySelector?.(selector)) s.remounts.push(now());
                }
            }
        }).observe(document.querySelector('#chat-messages'), { subtree: true, childList: true });
        const paint = () => {
            if (s.stopped) return;
            const view = s.describe();
            const at = now();
            const confirmed = view.same && view.status === 'You answered' && view.chosen.length === 1
                && view.chosen[0] === s.winner && view.answer === s.answerLine;
            if (confirmed && s.youAnsweredAt === null) {
                s.youAnsweredAt = at;
                // Rendered neighbours only: a hidden row (the idle typing bubble) has no place to keep.
                s.refs = [s.node.previousElementSibling, s.node.nextElementSibling]
                    .filter((ref) => ref && ref.getBoundingClientRect().height > 0);
                s.refNames = s.refs.map((ref) => ref.className || ref.tagName);
            }
            if (confirmed && !s.detaches.length) s.beforeDetach = place();
            const row = { present: view.present, same: view.same, connected: view.connected, confirmed,
                visible: view.visible, focusIn: view.focusIn, status: view.status, submission: view.submission, state: view.state };
            const key = JSON.stringify(row);
            const last = s.segments.at(-1);
            if (last?.key === key) { last.to = at; last.frames += 1; }
            else s.segments.push({ key, from: at, to: at, frames: 1, ...row });
            requestAnimationFrame(paint);
        };
        requestAnimationFrame(paint);
        return Boolean(s.node);
    };
}"""

_SETTLE_FRAMES = "() => new Promise(done => requestAnimationFrame(() => requestAnimationFrame(done)))"
_MAIN_READS = "() => window.__historyReads.filter(read => read.chatId === 1).length"


def _ours(row, task_id, quiz_id):
    return isinstance(row, dict) and row.get("task_id") == task_id and row.get("quiz_id") == quiz_id


def _confirmed_interval(segments, detached_at):
    """The frames from the first one that showed the recorded answer until the copy left: every one of them
    must still show it, on screen (a gap would be an answer taken back and shown again)."""
    first = next((i for i, segment in enumerate(segments) if segment["confirmed"] and segment["visible"]), None)
    stretch = [segment for segment in segments[first:] if segment["from"] < detached_at] if first is not None else []
    return stretch, [segment for segment in stretch if not (segment["confirmed"] and segment["visible"])]


def _answer_evidence(snapshot):
    """When real evidence of the recorded answer reached the page: a census naming it answered, or a detail read."""
    named = lambda facts: (facts or {}).get("quiz_state") == "answered"
    times = [entry["consumedAt"] for entry in snapshot["census"] if entry.get("consumedAt") and (
        named((entry.get("summary") or {}).get("named")) or any(map(named, (entry.get("summary") or {}).get("required", []))))]
    times += [entry["respondedAt"] for entry in snapshot["details"]
              if entry.get("respondedAt") and (entry.get("block") or {}).get("state") == "answered"]
    return sorted(times)


def _census_after(server, task_id, quiz_id):
    state = _api(server.base_url, "GET", "/api/state")
    return state, [row.get("required_question") for row in state.get("active_chat_activities") or []
                   if _ours(row.get("required_question"), task_id, quiz_id)]


def _wire(server, task_id, quiz_id, state, required):
    """The real bodies this journey's client met, for the unit replay (selected keys only, never edited)."""
    detail = _api(server.base_url, "GET", f"/api/tasks/{task_id}")
    return {"detail_after": {**{key: detail.get(key) for key in ("id", "task_id", "status", "owner_wait")},
                             "owner_quiz": {quiz_id: detail["owner_quiz"][quiz_id]},
                             "question_rooms": {quiz_id: (detail.get("question_rooms") or {}).get(quiz_id)}},
            "census_after": {**{key: state.get(key) for key in (
                "current_questions", "current_questions_complete", "current_questions_gaps")},
                "required_questions": required}}


def _fresh_reopen(pw, server, project, task_id, quiz_id, theme, final_phase, evidence, prefix):
    """A fresh page in each engine: no Main copy comes back, and the Project replays the record."""
    from playwright.sync_api import expect
    model_facts = []
    for engine, viewport in (("webkit", {"width": 1100, "height": 750}), ("chromium", {"width": 390, "height": 844})):
        browser = getattr(pw, engine).launch()
        page = browser.new_page(viewport=viewport, device_scale_factor=2, color_scheme=theme,
                                has_touch=viewport["width"] < 500, reduced_motion="reduce")
        page.add_init_script(f"({_OBSERVE_HISTORY})()")
        replay_errors = PageErrors(page, engine)
        replay_posts = []
        page.on("request", lambda request: replay_posts.append(request.url)
                if request.method == "POST" and request.url.endswith("/api/decisions") else None)
        try:
            page.goto(server.base_url, wait_until="domcontentloaded")
            page.wait_for_function("() => window.__historyReads.some(read => read.chatId === 1 && read.done)",
                                   timeout=30000)
            page.evaluate(_SETTLE_FRAMES)
            assert page.locator("#chat-messages .project-question-card").count() == 0
            since = page.evaluate("window.__historyReads.length")
            page.evaluate("""([project, task, quiz]) => window.dispatchEvent(new CustomEvent('ouro:open-project',
                {detail: {project, task_id: task, quiz_id: quiz}}))""", [project, task_id, quiz_id])
            replay_quiz = page.locator(".chat-quiz-card").filter(has_text="Which evidence should the report use?")
            replay_quiz.locator(".chat-quiz-option.chosen").filter(has_text=WINNER).wait_for(timeout=30000)
            replay_quiz.get_by_text(ANSWER_LINE, exact=True).wait_for()
            replay_card = page.locator(f'.chat-live-card[data-task-id="{task_id}"]')
            replay_history = _journey_history_painted(page, project, task_id, since)
            expect(replay_card).to_have_attribute("data-finished", "1", timeout=30000)
            expect(replay_card.locator("[data-live-phase]")).to_have_attribute("data-phase", final_phase)
            assert replay_card.locator("[data-live-title]").inner_text() == "still working"
            model_facts.append(replay_card.locator("[data-live-facts]").filter(has_text="Last solve response").count())
            page.screenshot(animations="disabled", full_page=True,
                            path=str(evidence / f'{prefix}-{engine}-{viewport["width"]}-replay.png'))
            assert not replay_posts and not replay_errors.unexpected()
            (evidence / f'{prefix}-{engine}-{viewport["width"]}-replay.json').write_text(json.dumps({
                "history_proof": replay_history, "posts": replay_posts, "page_errors": list(replay_errors)}, indent=2))
        finally:
            browser.close()
    return model_facts


# `activity`: the task is held inside its next model round, so the complete census omits the question
# from `current_questions` and names it answered only through the task's live activity row (the order
# the original reports met 1-3 s after the press). `silent`: the task has finished and left the live
# census, which then names the question nowhere; only an exact read can tell the page what happened.
@pytest.mark.parametrize("census_shape", ["activity", "silent"])
@pytest.mark.parametrize("live_engine", ["chromium", "webkit"])
def test_main_copy_survives_census_before_ack_and_shows_recorded_answer(
        wait_clone, tmp_path, monkeypatch, live_engine, census_shape):
    from playwright.sync_api import expect, sync_playwright
    from ouroboros.gateway.history import _DEFAULT_N_HUMAN
    from supervisor.message_bus import log_chat
    from tests.system_e2e.harness import ModelGate, body_text

    root = tmp_path / "instance" / "data"
    root.mkdir(parents=True)
    fake_home = tmp_path / "home"
    fake_home.mkdir()
    original_env = KeylessIsolatedServer._env

    def isolated_env(server):
        return {**original_env(server), "HOME": str(fake_home), "USERPROFILE": str(fake_home),
                "XDG_CONFIG_HOME": str(fake_home / ".config")}
    monkeypatch.setattr(KeylessIsolatedServer, "_env", isolated_env)
    evidence = producer_evidence_dir(tmp_path / "evidence", "main-answer-census-first",
                                     env="OUROBOROS_BROWSER_EVIDENCE_OUT")
    evidence.mkdir(parents=True, exist_ok=True)
    prefix = f"{live_engine}-{census_shape}"
    steps = [
        {"tool": "escalate", "arguments": {"question": "Which evidence should the report use?",
            "options": [{"label": "Primary sources", "detail": "Use the complete original measurements."},
                        {"label": WINNER, "detail": "Include the independent replication too."}],
            "stake": "The answer determines the report's evidence.", "wait_for_answer": True}},
        {"final": "The report will include both sources, as requested."},
    ]
    # Only the round that reads the delivered answer waits, and only in the `activity` shape.
    round_gate = (ModelGate(lambda body: bool(body.get("tools")) and f"Owner comment (verbatim): {COMMENT}" in body_text(body),
                            timeout=240) if census_shape == "activity" else None)
    wire, record = {}, {"engine": live_engine, "census_shape": census_shape,
                        "delivery_order": "census before the press reply and its quiz_state frames"}
    with ScriptedStubModel(steps, gate=round_gate) as stub:
        settings_path = root / "settings.json"
        write_settings_file(settings_path, keyless_settings(stub, OUROBOROS_MAX_WORKERS=1))
        server = KeylessIsolatedServer(wait_clone, root, settings_path)
        server.start(ready_timeout=120)
        oracle = ArtifactOracle(root)
        try:
            project = _api(server.base_url, "POST", "/api/projects", {"name": "Evidence review"})["project"]
            with sync_playwright() as pw:
                browser = getattr(pw, live_engine).launch()
                theme = "light" if live_engine == "webkit" else "dark"
                context = browser.new_context(viewport={"width": 1100, "height": 750}, device_scale_factor=2,
                                              color_scheme=theme, reduced_motion="reduce")
                context.tracing.start(screenshots=True, snapshots=True)
                page = context.new_page()
                page.add_init_script(f"({_OBSERVE_HISTORY})()")
                page.add_init_script(f"({_INVERSE_ORDER})()")
                errors = PageErrors(page, live_engine)
                console, posts = [], []
                page.on("console", lambda message: console.append({"type": message.type, "text": message.text}))
                page.on("request", lambda request: posts.append(request.post_data_json)
                        if request.method == "POST" and request.url.endswith("/api/decisions") else None)
                try:
                    page.goto(server.base_url, wait_until="domcontentloaded")
                    with chat_connection(server) as ws:
                        message_id = uuid.uuid4().hex
                        ws.send(json.dumps({"type": "chat", "content": "Ask which evidence to use, then wait for my answer.",
                            "client_message_id": message_id, "chat_id": project["chat_id"], "project_id": project["id"]}))
                        task = wait_until(lambda: next((row["task"] for row in oracle.events("task_received")
                            if (row.get("task", {}).get("metadata", {}).get("origin_message_ref") or {}).get("client_message_id") == message_id), None), 90)
                        assert task
                        wait = wait_until(lambda: (block if (block := oracle.task_result(task["id"]).get("owner_wait", {})).get("state") == "waiting" else None), 90)
                        assert wait
                        task_id, quiz_id = task["id"], wait["quiz_id"]
                        record.update(task_id=task_id, quiz_id=quiz_id, project=project)
                        pointer = page.locator(f'#chat-messages .project-question-card[data-task-id="{task_id}"]')
                        pointer.get_by_text("Waiting for your answer", exact=True).wait_for(timeout=30000)
                        main_question = page.locator("#chat-messages .chat-bubble.project-question").filter(
                            has=page.locator(f'.project-question-card[data-task-id="{task_id}"]'))
                        assert main_question.locator(".chat-quiz-option").count() == 2
                        assert main_question.locator(".chat-quiz-comment").count() == 1
                        main_question.get_by_text("Include the independent replication too.", exact=True).wait_for()
                        # The Project's own question row exists before its window is aged past the history page.
                        assert wait_until(lambda: any(row.get("msg_type") == "quiz" for row in _api(
                            server.base_url, "GET", f"/api/chat/history?chat_id={project['chat_id']}")["messages"]), 30)
                        for index in range(_DEFAULT_N_HUMAN + 1):
                            log_chat("in", project["chat_id"], 1, f"Retained later project note {index}", drive_root=root)
                        wire["census_before"] = [row for row in _api(server.base_url, "GET", "/api/state")["current_questions"]
                                                 if _ours(row, task_id, quiz_id)]
                        # The chip opens this exact question in its Project, rebuilt from its detail.
                        main_question.locator(".chat-bubble-voice > .chat-quiz-project").click()
                        quiz = page.locator(f'.chat-quiz-card[data-task-id="{task_id}"][data-quiz-id="{quiz_id}"]'
                                            ":not(.project-question-card)")
                        quiz.get_by_text("Include the independent replication too.", exact=True).wait_for()
                        page.locator("#project-panel-close").click()
                        main_question.scroll_into_view_if_needed()
                        page.screenshot(animations="disabled", path=str(evidence / f"{prefix}-1-main-question.png"))

                        # Arm the inverse order, then press once with the owner's words.
                        main_reads = page.evaluate(_MAIN_READS)
                        assert page.evaluate("a => window.__armInverseOrder(...a)", [task_id, quiz_id, WINNER, ANSWER_LINE])
                        page.wait_for_function("() => window.__ungatedStateReads === 0", timeout=30000)
                        main_question.locator(".chat-quiz-comment").fill(COMMENT)
                        main_question.get_by_role("button", name=WINNER).click()
                        page.wait_for_function("() => window.__inverseState().acks.length === 1"
                                               " && window.__inverseState().acks[0].respondedAt", timeout=30000)
                        page.screenshot(animations="disabled", path=str(evidence / f"{prefix}-2-press-held.png"))

                        # The backend really persists and discharges the answer.
                        def discharged():
                            block = _api(server.base_url, "GET", f"/api/tasks/{task_id}").get("owner_quiz", {}).get(quiz_id) or {}
                            return block if block.get("state") == "answered" and (block.get("delivery") or {}).get("state") == "delivered" else None
                        block = wait_until(discharged, 90)
                        assert block, "the backend never discharged the recorded answer"
                        assert block["answered_index"] == 1 and block["comment"] == COMMENT, block
                        page.wait_for_function("""() => window.__inverseState().frames.some(f => f.held && f.type === 'quiz_state'
                            && f.question_delivery?.state === 'delivered')""", timeout=30000)
                        if round_gate is not None:
                            assert round_gate.arrived.wait(90), "the round reading the answer never started"
                            state, required = _census_after(server, task_id, quiz_id)
                            assert [row.get("quiz_state") for row in required] == ["answered"], required
                        else:
                            result = wait_durable_result(oracle, task_id, timeout=90)
                            assert wait_until(lambda: not _census_after(server, task_id, quiz_id)[1], 90), \
                                "the finished task never left the live census"
                            state, required = _census_after(server, task_id, quiz_id)
                        assert state["current_questions_complete"] is True
                        assert not any(_ours(row, task_id, quiz_id) for row in state["current_questions"])
                        wire.update(_wire(server, task_id, quiz_id, state, required))
                        held = page.evaluate("() => window.__inverseState().snapshot()")
                        wire["post"] = {"request": held["acks"][0]["request"], "status": held["acks"][0]["status"],
                                        "reply": held["acks"][0]["reply"]}
                        wire["held_frames"] = [frame["msg"] for frame in held["frames"] if frame["held"]]
                        record["before_census"] = held
                        # Nothing but the census may reach the copy now: its reply, frames and reads all wait.
                        assert not held["acks"][0].get("deliveredAt") and held["heldCount"] >= 1, held
                        assert not any(entry.get("sentAt") for entry in held["details"]), held["details"]
                        assert not held["sockets"], "the socket cycled; its reopen census is a different journey"
                        assert page.evaluate(_MAIN_READS) == main_reads, "Main re-read its history: another answer source"
                        assert held["view"]["same"] and held["view"]["connected"], held["view"]
                        assert held["view"]["status"] != "You answered", held["view"]

                        # The fresh census: complete, and without this question among its current questions.
                        page.evaluate("() => window.__inverseState().release('census')")
                        page.wait_for_function("() => window.__inverseState().census.some(c => c.omits && c.after)", timeout=30000)
                        page.screenshot(animations="disabled", path=str(evidence / f"{prefix}-3-after-census.png"))
                        applied = page.evaluate("() => window.__inverseState().snapshot()")
                        record["after_census"] = applied
                        census = next(entry for entry in applied["census"] if entry.get("omits") and entry.get("after"))
                        after = census["after"]
                        assert after["connected"] and after["same"], (
                            "#1993: the census removed the Main copy that took the press before any recorded answer "
                            f"was shown (its reply and quiz_state frames still held): {census}")
                        if census_shape == "silent":
                            assert after["status"] != "You answered", f"no evidence of the answer reached the page yet: {census}"
                        elif after["status"] == "You answered":
                            assert after["chosen"] == [WINNER] and after["answer"] == ANSWER_LINE, census

                        # The exact detail read may now answer; the reply and frames stay held.
                        page.evaluate("() => window.__inverseState().release('detail')")
                        page.wait_for_function("() => window.__inverseState().youAnsweredAt !== null", timeout=40000)
                        page.screenshot(animations="disabled", path=str(evidence / f"{prefix}-4-recorded-answer.png"))
                        page.wait_for_function("() => window.__inverseState().afterDetach", timeout=20000)
                        settled = page.evaluate("() => window.__inverseState().snapshot()")
                        record["settled"] = settled
                        shown_after = _answer_evidence(settled)
                        assert shown_after and shown_after[0] <= settled["youAnsweredAt"], (
                            "You answered appeared before any real evidence of the answer", settled["census"], settled["details"])
                        assert not settled["acks"][0].get("deliveredAt") and settled["heldCount"] >= 1, settled
                        assert len(settled["detaches"]) == 1 and not settled["reattaches"] and not settled["remounts"], settled
                        detached_at = settled["detaches"][0]
                        stretch, broken = _confirmed_interval(settled["segments"], detached_at)
                        assert stretch and not broken, ("the recorded answer showed with a gap", settled["segments"])
                        shown_for = detached_at - stretch[0]["from"]
                        assert 4500 <= shown_for <= 7000, ("one five-second interval, neither cut short nor restarted", shown_for)
                        assert detached_at - stretch[-1]["to"] <= 500, ("frames showed the answer until it left", stretch[-1])
                        # DESIGN: focus remains in the copy during the interval, then moves only to the next
                        # Main question or a keyboard owner's composer; retiring it does not move the viewport.
                        # WebKit gives a clicked button no focus, so a pointer press may leave it outside the copy.
                        index = settled["segments"].index(stretch[0])
                        prior = settled["segments"][index - 1] if index else None
                        assert len({segment["focusIn"] for segment in stretch}) == 1, ("focus moved during the interval", stretch)
                        assert stretch[0]["focusIn"] or not (prior or {}).get("focusIn"), ("the settle took focus out", prior, stretch[0])
                        before, after = settled["beforeDetach"], settled["afterDetach"]
                        assert after["active"]["connected"] and (after["active"]["body"] or after["active"]["composer"]), after
                        moves = [abs(a - b) for a, b in zip(after["tops"], before["tops"]) if a is not None and b is not None]
                        assert moves and min(moves) <= 2, ("retiring the copy moved the reading position",
                                                           settled["refNames"], before, after)

                        # The held reply and frames arrive late: nothing returns, nothing is sent again.
                        page.evaluate("() => { window.__inverseState().release('ack'); window.__inverseState().release('frames'); }")
                        page.wait_for_function("() => window.__inverseState().acks[0].deliveredAt || window.__inverseState().acks[0].abortedAt",
                                               timeout=10000)
                        for _ in range(30):
                            page.evaluate(_SETTLE_FRAMES)
                        late = page.evaluate("() => window.__inverseState().snapshot()")
                        record["late"] = late
                        assert not late["view"]["present"] and len(late["detaches"]) == 1 and not late["remounts"], late
                        assert len(posts) == 1 and posts[0]["option_index"] == 1 and posts[0]["comment"] == COMMENT, posts
                        page.evaluate("() => { window.__inverseState().stopped = true; }")
                        if round_gate is not None:
                            round_gate.release.set()
                            result = wait_durable_result(oracle, task_id, timeout=90)
                        rounds = [body_text(body) for _kind, body in list(stub.calls)
                                  if body.get("tools") and f"Owner comment (verbatim): {COMMENT}" in body_text(body)]
                        assert rounds and f"The owner chose option 2: {WINNER}" in rounds[0], stub.kinds()
                        record["answer_round"] = rounds[0][rounds[0].index("[Owner quiz answer]"):][:600]

                        # Only the Main copy went: the exact question still opens in its Project, with the record.
                        final_phase = {"completed": "done", "failed": "error", "cancelled": "cancelled"}[result["status"]]
                        since = page.evaluate("window.__historyReads.length")
                        page.evaluate("""([project, task, quiz]) => window.dispatchEvent(new CustomEvent('ouro:open-project',
                            {detail: {project, task_id: task, quiz_id: quiz}}))""", [project, task_id, quiz_id])
                        quiz.locator(".chat-quiz-option.chosen").filter(has_text=WINNER).wait_for(timeout=30000)
                        quiz.get_by_text(ANSWER_LINE, exact=True).wait_for()
                        task_card = page.locator(f'.chat-live-card[data-task-id="{task_id}"]')
                        history_proof = _journey_history_painted(page, project, task_id, since)
                        # The durable result can precede the card's finalization (Finalizing…): wait, bounded.
                        expect(task_card).to_have_attribute("data-finished", "1", timeout=30000)
                        expect(task_card.locator("[data-live-phase]")).to_have_attribute("data-phase", final_phase)
                        assert task_card.locator("[data-live-title]").inner_text() == "still working"
                        model_facts = [task_card.locator("[data-live-facts]").filter(has_text="Last solve response").count()]
                        page.screenshot(animations="disabled", path=str(evidence / f"{prefix}-5-project-record.png"), full_page=True)
                        assert not errors.unexpected(), errors
                        record.update(history_proof=history_proof, status=result["status"])
                finally:
                    if round_gate is not None:
                        round_gate.release.set()
                    try:
                        page.screenshot(animations="disabled", path=str(evidence / f"{prefix}-end.png"))
                        record["final"] = page.evaluate("() => window.__inverseState?.()?.snapshot?.() ?? null")
                        record["history_reads"] = page.evaluate("window.__historyReads")
                    finally:
                        record.update(console=console, page_errors=list(errors), posts=posts)
                        (evidence / f"{prefix}-journey.json").write_text(json.dumps(record, indent=2))
                        (evidence / f"{prefix}-wire.json").write_text(json.dumps(wire, indent=2))
                        context.tracing.stop(path=str(evidence / f"{prefix}-trace.zip"))
                        browser.close()
                model_facts += _fresh_reopen(pw, server, project, task_id, quiz_id, theme, final_phase, evidence, prefix)
            assert result["status"] == "completed", result
            assert model_facts == [1, 1, 1], model_facts
        finally:
            server.stop()
            (evidence / f"{prefix}-server.log").write_bytes((root / "logs/server.log").read_bytes())
            assert server.proc.poll() is not None
