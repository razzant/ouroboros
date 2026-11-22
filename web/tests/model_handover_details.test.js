// A host model handover reads in its card's Details, never on its head. Only settled stored detail says what
// handover history is available. The durable terminal signals refresh open Details even after an early answer.
import assert from 'node:assert/strict';
import test from 'node:test';
import { createChatInstance } from '../modules/chat.js';
import { modelHandoverLines } from '../modules/model_handover.js';
import { installDom, restoreDom, walkCard } from './chat_dom_fixture.js';

const ASTRA = 'codex=gpt-6-astra', FABLE = 'claude-fable-5-1', OPUS = 'claude-opus-5-5';
const TS = '2026-11-18T09:00:00Z';
// A row of the declared shape (loop_model_call._record_authoring_handover, loop_nudges._authoring_handover_step).
const row = (status, extra = {}) => ({ from_model: ASTRA, to_model: FABLE, reason: 'fallback', tool_calls_at_handover: 1,
    recovery_prompted: true, status, ...extra });
const OK = { execution: { status: 'ok' }, objective: { status: 'not_evaluated' } };
const drain = () => new Promise((resolve) => setImmediate(resolve));  // the stub's fetch and paint are promises

test('a route pair is not a reply; only a recorded selection or successful tool call says what the successor did', () => {
    const pair = `${ASTRA} → ${FABLE}`;
    const unknown = `${pair} · successor reply not recorded`;
    assert.deepEqual(modelHandoverLines([
        row('selected', { selection: 'answer_sha256' }), row('selected', { selection: 'answer', incomplete_observed: true }),
        row('selected', { selection: 'silent' }), row('selected', { selection: 'stop' }),
        // The same response can work and then choose: one fact does not hide the other (`accept_handover_selection`).
        row('selected', { selection: 'stop', work_observed: true }), row('pending', { work_observed: true }),
        row('recovered', { work_observed: true }),
        row('recovered'),  // an older row: its `recovered` may be a selection, so it proves no work
        row('incomplete', { incomplete_observed: true }),
        row('incomplete', { selection: 'answer_sha256', incomplete_observed: true }),  // legacy status does not settle what it did
        row('pending'), row('pending', { reply_observed: true }), row('selected'),
        row('selected', { selection: 'later', reply_observed: true }), row('a_later_status'), row('selected.stop'),
    ]), [
        `${pair} · confirmed a saved answer`, `${pair} · submitted an answer`, `${pair} · finished without a final message`,
        `${pair} · chose to stop`, `${pair} · completed a tool call · chose to stop`,
        `${unknown} · completed a tool call`, `${pair} · completed a tool call`, unknown,
        `${unknown} · handover recorded as incomplete`, `${unknown} · handover recorded as incomplete`,
        unknown, `${pair} · no choice recorded`, unknown, pair, unknown, unknown,
    ]);
    // Different routes can share a compact model name; a row without both models is no line.
    assert.deepEqual(modelHandoverLines([row('pending', { from_model: 'openrouter::lab/model', to_model: 'other::model', reply_observed: true }),
        { status: 'selected', selection: 'answer_sha256' }]), ['openrouter::lab/model → other::model · no choice recorded']);
    assert.deepEqual(modelHandoverLines(null), []);
});

/** One root card on a page whose GET /api/tasks/{id} answers `f.detail` at the time of the read. */
function taskCard(taskId, detail, chatId = 1) {
    const reads = [];
    const f = { detail, reads };
    const { prior, mount } = installDom(async (url) => {
        const own = String(url) === `/api/tasks/${taskId}`, body = f.detail;  // the store as the request found it
        if (own) { reads.push(body); await f.gate; if (f.fail) return { ok: false, status: 500, json: async () => ({}) }; }  // `gate`: a read still in flight; `fail`: a failed read
        return { ok: true, json: async () => (own ? body : String(url).startsWith('/api/chat/history')
            ? { messages: [], window: { complete: true } } : { active_direct_turns: [] }) };
    });
    const handlers = new Map();
    let instance;
    f.close = () => { instance?.destroy(); restoreDom(prior); };
    try {
        instance = createChatInstance({
            ws: { on(type, fn) { handlers.set(type, fn); return () => handlers.delete(type); }, isConnected: () => true, send() {} },
            state: { activePage: 'chat', projectChatIds: new Set(chatId === 1 ? [] : [chatId]), unreadCount: 0 },
            updateUnreadBadge() {}, chatId, idPrefix: 'chat', mountEl: mount,
            stateSnapshots: { begin: () => ({ generation: 1, requestedAt: Date.now() }), gate() { return Promise.resolve(this.begin()); },
                isCurrent: () => true, apply() {} },
        });
        handlers.get('chat')({ chat_id: chatId, role: 'assistant', is_progress: true, task_id: taskId, ts: TS,
            content: 'Reading the predecessor\'s checks', narration: true });
    } catch (error) { f.close(); throw error; }
    f.card = walkCard(globalThis.document.byId.get('chat-messages'), taskId);
    f.chat = (data) => handlers.get('chat')({ chat_id: chatId, task_id: taskId, ts: TS, ...data });
    f.log = (data) => handlers.get('log')({ chat_id: chatId, data: { task_id: taskId, ...data } });
    // One /api/state body listing the task's census row as gateway/state.py publishes a managed task's.
    f.census = (attempt) => instance.hydrateStateSnapshot({ supervisor_ready: true, active_chat_activities_complete: true,
        active_chat_activities: attempt === null ? [] : [{ activity_id: taskId, chat_id: chatId, project_id: '', client_message_id: '',
            kind: 'managed_task', phase: 'running', detached: false, can_resume: false, cancelable: true,
            started_at: 1795000000, task_attempt: attempt }] });
    f.facts = () => f.card.querySelector('[data-live-facts]').innerHTML;
    f.reveal = instance.revealSource;
    f.until = async (ok, what) => {
        for (let i = 0; i < 200 && !ok(); i += 1) await new Promise((r) => setTimeout(r, 5));
        assert.ok(ok(), `${what}: ${f.facts()}`);
    };
    f.final = () => f.chat({ role: 'assistant', content: 'The report stands as written.', task_terminal_status: 'completed',
        reason_code: f.detail.reason_code, outcome_axes: f.detail.outcome_axes ?? OK });
    f.done = () => f.log({ type: 'task_done', status: 'completed',
        reason_code: f.detail.reason_code, outcome_axes: f.detail.outcome_axes ?? OK });
    f.summary = (extra = {}) => f.chat({ role: 'system', system_type: 'task_summary', summary_kind: 'terminal_root_projection',
        outcome_final: true, status: 'completed', outcome_axes: OK, content: 'Done.', ...extra });
    f.settle = () => { f.done(); f.final(); };
    f.toggle = async () => {  // the disclosure; opening it is what reads the stored detail
        const before = reads.length;
        const opening = f.card.dataset.expanded !== '1';
        f.card.querySelector('[data-live-disclosure]').click();
        if (opening) await f.until(() => reads.length > before, 'opening Details reads the detail');
        await drain();  // the read applies after its body resolves
    };
    return f;
}

const settled = (taskId, extra) => ({ task_id: taskId, status: 'completed', reason_code: 'final_message', task_attempt: 1,
    outcome_axes: OK, ...extra });

test('Details open while the work runs show the settled history when it settles, without a second click', async () => {
    const id = 'handover-open';
    const f = taskCard(id, { task_id: id, status: 'running', task_attempt: 1 });
    try {
        let release;
        f.gate = new Promise((resolve) => { release = resolve; });
        await f.toggle();  // its read of the running detail is still in flight when the work settles
        f.detail = settled(id, { authoring_handovers: [row('incomplete', { incomplete_observed: true, reply_observed: true }),
            row('selected', { from_model: FABLE, to_model: OPUS, selection: 'answer_sha256' })] });
        f.settle();
        f.gate = null;
        release();
        await f.until(() => f.facts().includes('Model handover'), 'the settled read paints the open row');
        assert.equal(f.reads.length, 2, 'one more read, at settlement');
        assert.ok(f.facts().includes(`<dt>Model handover</dt><dd>${ASTRA} → ${FABLE} · handover recorded as incomplete<br>`
            + `${FABLE} → ${OPUS} · confirmed a saved answer</dd>`), f.facts());
        assert.equal(f.card.querySelector('[data-live-phase]').textContent, 'Done', 'a settled switch adds no warning');
        for (const part of ['[data-live-title]', '[data-live-activity]', '[data-live-meta]']) {
            const node = f.card.querySelector(part);
            assert.doesNotMatch(`${node?.innerHTML || ''} ${node?.textContent || ''}`, /handover|→/, part);
        }
        // Partial frames without the history (a cost, a replayed task_done) leave the row as the detail said it.
        f.log({ type: 'task_cost_finalized', accounted_upper_bound_usd: 1, cost_final: true, task_attempt: 2 });
        f.log({ type: 'task_done', status: 'completed', reason_code: 'final_message', outcome_axes: OK });
        f.census(1);  // the same attempt, listed again
        assert.ok(f.facts().includes('<dt>Model handover</dt>'), f.facts());
        // An omitted field on a stored read of the same attempt is not a deletion.
        f.detail = settled(id, {});
        await f.toggle();
        await f.toggle();
        assert.match(f.facts(), /confirmed a saved answer/);
        f.detail = settled(id, { authoring_handovers: [] });
        await f.toggle();
        await f.toggle();
        assert.doesNotMatch(f.facts(), /Model handover/, 'an explicit empty history clears the row');
    } finally { f.close(); }
});

for (const [room, chatId, publish] of [['Main', 1, 'done'], ['Project', 23, 'summary'], ['Project replay', 23, 'replaySummary']]) {
    test(`${room}: an early answer can close the card before the stored handover is readable`, async () => {
        const id = 'handover-late-store';
        const f = taskCard(id, { task_id: id, status: 'running', task_attempt: 1 }, chatId);
        const replaySummary = () => f.summary({ outcome_final: undefined, status: undefined,
            historical_terminal: { status: 'completed', phase: 'done', provenance: 'canonical_task_result_after_finalization' } });
        try {
            await f.toggle();
            f.final();  // live publication before the store (test_finalization_timing.py)
            f.summary({ outcome_final: false, task_phase: 'finalizing' });
            f.log({ type: 'task_cost_finalized', status: 'completed', cost_final: true });
            await drain();
            assert.equal(f.card.dataset.finished, '1');
            assert.equal(f.reads.length, 1, 'early answer and partial updates did not spend a settled-detail read');
            assert.doesNotMatch(f.facts(), /Model handover/);

            f.detail = settled(id, { authoring_handovers: [row('selected', { selection: 'answer_sha256' })] });
            let release;
            f.gate = new Promise((resolve) => { release = resolve; });
            (publish === 'replaySummary' ? replaySummary : f[publish])();  // post-store, with no second click or timer
            await drain();
            assert.equal(f.reads.length, 2, 'its single stored read started');
            f[publish === 'done' ? 'summary' : 'done']();  // a second terminal signal during that read
            release();
            await drain();
            assert.match(f.facts(), /confirmed a saved answer/);
            assert.equal(f.reads.length, 2);
            f.done(); f.summary();  // duplicate/following durable publications need no more requests
            await drain();
            assert.equal(f.reads.length, 2);
        } finally { f.close(); }
    });
}

test('an exact quote opens Details once and rereads only when the opened card lacks current detail', async () => {
    const id = 'handover-quote';
    const first = settled(id, { authoring_handovers: [row('selected', { selection: 'answer' })] });
    const f = taskCard(id, first);
    try {
        f.settle();
        const target = document.createElement('div');
        target.dataset.rowAddress = `row:1@${TS}#0123456789ab`;
        f.card.append(target);  // the exact timeline row already held by this card
        const jump = async () => {
            assert.deepEqual(await f.reveal(target.dataset.rowAddress), { state: 'found', target });
            assert.equal(document.activeElement, target);
            await drain();
        };
        assert.equal(f.card.dataset.expanded, '0');
        assert.equal(f.reads.length, 0);
        await jump();
        assert.equal(f.card.dataset.expanded, '1');
        assert.match(f.facts(), /submitted an answer/);
        assert.deepEqual(f.reads, [first]);
        await jump();
        assert.deepEqual(f.reads, [first], 'jumping twice to the same row does not read current detail twice');

        f.census(2);
        assert.doesNotMatch(f.facts(), /Model handover/);
        const running = { task_id: id, status: 'running', task_attempt: 2 };
        f.detail = running;
        await jump();
        assert.doesNotMatch(f.facts(), /Model handover/, 'a running read supplies no settled detail');
        const next = settled(id, { task_attempt: 2, authoring_handovers: [row('selected', { selection: 'answer_sha256' })] });
        f.detail = next;
        await jump();
        assert.match(f.facts(), /confirmed a saved answer/);
        assert.deepEqual(f.reads, [first, running, next], 'neither the first attempt nor a running read hides missing detail');
    } finally { f.close(); }
});

for (const via of ['quote', 'button']) test(`reopening with ${via} joins the running read and obtains settled Details`, async () => {
    const id = 'handover-reopen', running = { task_id: id, task_attempt: 1, status: 'running' };
    const f = taskCard(id, running);
    let release;
    try {
        f.gate = new Promise(resolve => { release = resolve; });
        await f.toggle();  // the first GET captured the running result, but hasn't answered
        await f.toggle();  // folded during that GET, so settlement alone won't request Details
        const terminal = settled(id, { authoring_handovers: [row('selected', { selection: 'answer_sha256' })] });
        f.detail = terminal;
        f.settle();
        if (via === 'button') f.card.querySelector('[data-live-disclosure]').click();
        else {
            const target = document.createElement('div');
            target.dataset.rowAddress = `row:1@${TS}#0123456789ab`;
            f.card.append(target);
            assert.equal((await f.reveal(target.dataset.rowAddress)).target, target);
        }
        assert.equal(f.card.dataset.expanded, '1');
        assert.deepEqual(f.reads, [running]);
        f.gate = null;
        release();
        await f.until(() => f.facts().includes('confirmed a saved answer'), 'reopened card received terminal detail');
        assert.deepEqual(f.reads, [running, terminal]);
    } finally { release?.(); f.close(); }
});

test('a census reconciliation applies the terminal detail it already fetched', async () => {
    const id = 'handover-reconciled';
    const f = taskCard(id, { task_id: id, status: 'running', task_attempt: 1 });
    try {
        f.census(1);
        await f.toggle();
        f.detail = settled(id, { authoring_handovers: [row('selected', { selection: 'answer' })] });
        f.census(null);
        await drain();
        assert.match(f.facts(), /submitted an answer/);
        assert.equal(f.card.dataset.finished, '1');
        assert.equal(f.reads.length, 2, 'one on open, one on census loss');
        f.done(); f.summary();
        await drain();
        assert.equal(f.reads.length, 2, 'the stored read was already applied');
    } finally { f.close(); }
});

test('an early answer arriving during the census read does not discard its stored handover', async () => {
    const id = 'handover-census-early-answer';
    const f = taskCard(id, { task_id: id, status: 'running', task_attempt: 1 });
    try {
        f.census(1);
        await f.toggle();
        let release;
        f.gate = new Promise((resolve) => { release = resolve; });
        f.detail = settled(id, { authoring_handovers: [row('selected', { selection: 'answer' })] });
        f.census(null);
        await drain();
        assert.equal(f.reads.length, 2, 'the stored read is already in flight');
        f.final();
        assert.equal(f.card.dataset.finished, '1');
        assert.doesNotMatch(f.facts(), /Model handover/);
        release();
        await drain();
        assert.match(f.facts(), /submitted an answer/);
        f.done();
        await drain();
        assert.equal(f.reads.length, 2, 'no extra read after the census one');
    } finally { f.close(); }
});

for (const first of ['completed', 'running']) test(`settlement overlaps a census GET of ${first} detail`, async () => {
    const id = 'handover-census-settlement';
    const running = { task_id: id, status: 'running', task_attempt: 1 };
    const terminal = settled(id, { authoring_handovers: [row('selected', { selection: 'answer' })],
        review_overrides: { count: 1, latest: { reason: 'detail_only' } } });
    const f = taskCard(id, running);
    try {
        f.census(1);
        await f.toggle();
        let release;
        f.gate = new Promise(resolve => { release = resolve; });
        f.detail = first === 'completed' ? terminal : running;
        f.census(null);
        await drain();
        assert.equal(f.reads.length, 2, 'the census read started');
        f.detail = terminal;
        f.done();
        f.summary();
        await drain();
        const whileHeld = f.reads.length;
        f.gate = null;
        release();
        assert.equal(whileHeld, 2, 'the terminal signal waits for the matching census read');
        await f.until(() => f.facts().includes('submitted an answer'), 'terminal detail paints the open disclosure');
        await drain();
        assert.deepEqual(f.reads, first === 'completed' ? [running, terminal] : [running, running, terminal]);
        assert.match(f.card.querySelector('[data-live-meta]').innerHTML, /latest: detail only/);
    } finally { f.close(); }
});

test('a terminal hint for a new attempt does not join an older census GET', async () => {
    const id = 'handover-census-old-attempt';
    const old = settled(id, { authoring_handovers: [row('selected', { selection: 'answer' })] });
    const f = taskCard(id, { task_id: id, status: 'running', task_attempt: 1 });
    let release;
    try {
        f.census(1);
        await f.toggle();
        f.gate = new Promise(resolve => { release = resolve; });
        f.detail = old;
        f.census(null);
        await drain();
        f.gate = null;
        f.census(2);
        f.detail = settled(id, { task_attempt: 2, authoring_handovers: [row('selected', { selection: 'answer_sha256' })] });
        f.summary({ task_attempt: 2 });
        await drain();
        assert.match(f.facts(), /confirmed a saved answer/, 'attempt 2 reads while attempt 1 is still held');
        release();
        await drain();
        assert.equal(f.reads.length, 3);
        assert.doesNotMatch(f.facts(), /submitted an answer/, 'the released attempt 1 is not drawn');
    } finally { release?.(); f.close(); }
});

test('a reveal during the census read joins it; a settled answer needs no second GET', async () => {
    const id = 'handover-census-join';
    const terminal = settled(id, { authoring_handovers: [row('selected', { selection: 'answer_sha256' })] });
    const f = taskCard(id, { task_id: id, status: 'running', task_attempt: 1 });
    let release;
    try {
        f.census(1);
        f.gate = new Promise((resolve) => { release = resolve; });
        f.detail = terminal;
        f.census(null);  // the task's disappearance starts the census read...
        await drain();
        assert.equal(f.reads.length, 1, 'the census read is in flight');
        f.settle();  // ...while terminal frames finish the card
        await drain();
        assert.equal(f.reads.length, 1, 'the closed disclosure spent no read');
        f.card.querySelector('[data-live-disclosure]').click();  // the reveal joins the in-flight read
        await drain();
        assert.equal(f.reads.length, 1, 'no concurrent second GET for the same attempt');
        f.gate = null;
        release();
        await f.until(() => f.facts().includes('confirmed a saved answer'), 'the joined read settles the open Details');
        assert.deepEqual(f.reads, [terminal], 'one read total: the census answer supplied the disclosure');
        f.done(); f.summary();
        await drain();
        assert.equal(f.reads.length, 1, 'following terminal frames add no read');
    } finally { release?.(); f.close(); }
});

test('a joined census read that lands visibly incomplete earns exactly one top-up read', async () => {
    const id = 'handover-census-topup';
    const running = { task_id: id, status: 'running', task_attempt: 1 };
    const terminal = settled(id, { authoring_handovers: [row('selected', { selection: 'answer' })] });
    const f = taskCard(id, running);
    let release;
    try {
        f.census(1);
        f.gate = new Promise((resolve) => { release = resolve; });
        f.census(null);  // the census GET captured the running store...
        await drain();
        f.settle();
        f.card.querySelector('[data-live-disclosure]').click();  // the reveal joins it
        await drain();
        assert.equal(f.reads.length, 1, 'no concurrent second GET');
        f.detail = terminal;  // ...and the store settles only after that capture
        f.gate = null;
        release();
        await f.until(() => f.reads.length > 1, 'the incomplete answer is followed by one top-up read');
        await f.until(() => f.facts().includes('submitted an answer'), 'the top-up paints the settled row');
        assert.deepEqual(f.reads, [running, terminal], 'exactly one top-up after the joined read');
        f.done(); f.summary();
        await drain();
        assert.equal(f.reads.length, 2, 'later terminal frames add no read');
    } finally { release?.(); f.close(); }
});

test('a census read failing under an open card costs one top-up, and a reopen rereads once', async () => {
    const id = 'handover-census-failed';
    const running = { task_id: id, status: 'running', task_attempt: 1 };
    const terminal = settled(id, { authoring_handovers: [row('selected', { selection: 'answer' })] });
    const f = taskCard(id, running);
    let release;
    try {
        f.census(1);
        f.gate = new Promise((resolve) => { release = resolve; });
        f.fail = true;
        f.census(null);  // the census read is in flight and will fail
        await drain();
        f.settle();
        f.card.querySelector('[data-live-disclosure]').click();  // the reveal joins it
        await drain();
        assert.equal(f.reads.length, 1, 'the reveal spent no concurrent GET');
        f.gate = null;
        release();
        await f.until(() => f.reads.length > 1, 'the failed census read is followed by one top-up');
        await drain();
        assert.equal(f.reads.length, 2, 'a failed top-up is no retry loop; its card offers Retry');
        f.fail = false;
        f.detail = terminal;
        await f.toggle();  // close...
        await f.toggle();  // ...reopen: one fresh read paints the settled row
        assert.deepEqual(f.reads, [running, running, terminal]);
        assert.match(f.facts(), /submitted an answer/);
    } finally { release?.(); f.close(); }
});

const retry = (f) => f.card.querySelector('[data-review-hydrate-retry]');
// The DOM stub's click does not bubble or match data selectors; hand the review host the press as a browser would.
function pressRetry(f) {
    const button = retry(f);
    let host = button.parentElement;
    while (host && !host.listeners.get('click')?.length) host = host.parentElement;
    for (const fn of host.listeners.get('click')) fn({ target: { closest: (sel) => (sel === '[data-review-hydrate-retry]' ? button : null) } });
}

test('an open running card whose joined census read fails shows Retry; Retry reads once and a success retires it', async () => {
    const id = 'handover-census-open-failed';
    const running = { task_id: id, status: 'running', task_attempt: 1 };
    const terminal = settled(id, { authoring_handovers: [row('selected', { selection: 'answer' })] });
    const f = taskCard(id, running);
    try {
        f.census(1);
        await f.toggle();  // open while running: its own read answers
        assert.equal(retry(f), null, 'a good read offers no Retry');
        let release;
        f.gate = new Promise((resolve) => { release = resolve; });
        f.fail = true;
        f.census(null);  // the census read starts under the open, still unfinished card
        await drain();
        assert.equal(f.reads.length, 2, 'one census read');
        assert.equal(retry(f), null, 'no failure before the read answers');
        f.gate = null;
        release();
        await f.until(() => retry(f) !== null, 'the failed census read leaves the ordinary Retry');
        assert.equal(f.reads.length, 2, 'a failure spends no extra GET');
        f.fail = false;
        f.detail = terminal;
        pressRetry(f);
        await f.until(() => f.facts().includes('submitted an answer'), 'Retry reads the settled detail');
        await drain();
        assert.equal(retry(f), null, 'a successful read retires the Retry');
        assert.equal(f.reads.length, 3, 'Retry is exactly one GET');
    } finally { f.close(); }
});

test('a successful joined census read retires the Retry an earlier failed read left', async () => {
    const id = 'handover-census-stale-retry';
    const terminal = settled(id, { authoring_handovers: [row('selected', { selection: 'answer' })] });
    const f = taskCard(id, { task_id: id, status: 'running', task_attempt: 1 });
    try {
        f.census(1);
        f.fail = true;
        await f.toggle();  // the card's own read fails
        await f.until(() => retry(f) !== null, 'the first failure offers Retry');
        f.fail = false;
        f.detail = terminal;
        f.census(null);  // the census read now succeeds with the terminal answer
        await f.until(() => f.facts().includes('submitted an answer'), 'the census applied the terminal detail');
        await drain();
        assert.equal(f.card.dataset.finished, '1');
        assert.equal(retry(f), null, 'the answer and a stale Retry never show together');
        assert.equal(f.reads.length, 2, 'the census read was the only extra GET');
    } finally { f.close(); }
});

for (const why of ['the card is closed', 'the attempt moved on']) test(`a failed census read offers no Retry when ${why}`, async () => {
    const id = 'handover-census-no-retry';
    const f = taskCard(id, { task_id: id, status: 'running', task_attempt: 1 });
    let release;
    try {
        f.census(1);
        f.gate = new Promise((resolve) => { release = resolve; });
        f.fail = true;
        f.census(null);  // the census read of attempt 1 starts under a closed card
        await drain();
        if (why !== 'the card is closed') {
            f.card.querySelector('[data-live-disclosure]').click();  // opened, then the attempt moves on
            f.census(2);
        }
        assert.equal(f.reads.length, 1);
        f.gate = null;
        release();
        await drain();
        assert.equal(retry(f), null, 'a read that cannot apply to an open card reports nothing');
        assert.equal(f.reads.length, 1, 'the failed read spent no other GET');
    } finally { release?.(); f.close(); }
});

test('an earlier Continue episode reads the handover into its original Details, not the current episode', async () => {
    const id = 'handover-predecessor', successor = 'handover-successor';
    const f = taskCard(id, settled(id, { authoring_handovers: [row('selected', { selection: 'answer' })] }));
    try {
        const original = f.card.querySelector('.chat-live-details');
        const facts = f.card.querySelector('[data-live-facts]');
        original.appendChild(facts);  // the test DOM flattens the template; the browser nests these
        f.chat({ task_id: successor, role: 'assistant', is_progress: true, narration: true, content: 'Checking a new item' });
        const next = walkCard(document.byId.get('chat-messages'), successor);
        const current = next.querySelector('.chat-live-details');
        const nextFacts = next.querySelector('[data-live-facts]');
        current.appendChild(nextFacts);
        f.summary({ continued_by: { state: 'admitted', successor_task_id: successor } });
        const earlier = next.querySelector(`[data-episode-task="${id}"]`);
        assert.equal(f.card.hidden, true);
        assert.equal(original.parentNode, earlier);
        assert.equal(f.reads.length, 0, 'a closed episode did not fetch its detail');

        next.querySelector('[data-live-disclosure]').click();
        earlier.open = true;
        for (const toggle of earlier.listeners.get('toggle')) toggle();
        await drain();
        assert.equal(f.reads.length, 1, 'opening the earlier episode reads its own task');
        assert.match(facts.innerHTML, /submitted an answer/);
        assert.doesNotMatch(nextFacts.innerHTML, /Model handover/);
    } finally { f.close(); }
});

// Either card can arrive first: the host's admission can precede the successor's card, so the chain may form when that
// card appears and nothing else paints the earlier work afterwards.
for (const order of ['the successor\'s card first', 'the admission first']) test(`an earlier episode says its closing sentence once: in its heading, not again in its moved Details (${order})`, async () => {
    const id = 'handover-episode-words', successor = 'handover-episode-next', sentence = 'Checked the disk; all fine.';
    const stamp = { initiator: 'consciousness', wake_summary: sentence };
    const f = taskCard(id, settled(id, {}));
    try {
        f.chat({ role: 'assistant', is_progress: true, narration: true, content: 'Reading the disk', ...stamp });
        f.summary(stamp);
        const original = f.card.querySelector('.chat-live-details');
        const facts = f.card.querySelector('[data-live-facts]');
        original.appendChild(facts);  // the test DOM flattens the template; the browser nests these
        assert.match(facts.innerHTML, /<dt>In its words<\/dt><dd><span data-i18n-authored>Checked the disk; all fine\.<\/span><\/dd>/,
            'a work with no later episode says its sentence in its own Details');
        const open = () => f.chat({ task_id: successor, role: 'assistant', is_progress: true, narration: true, content: 'Checking a new item' });
        const admit = () => f.summary({ ...stamp, continued_by: { state: 'admitted', successor_task_id: successor } });
        if (order === 'the admission first') { admit(); open(); } else { open(); admit(); }
        const next = walkCard(document.byId.get('chat-messages'), successor);
        const earlier = next.querySelector(`[data-episode-task="${id}"]`);
        assert.equal(original.parentNode, earlier, 'the original Details node is the one disclosed');
        assert.ok(earlier.firstElementChild.textContent.endsWith(` · ${sentence}`), earlier.firstElementChild.textContent);
        assert.doesNotMatch(facts.innerHTML, /In its words/, 'the heading already says it');
        assert.match(facts.innerHTML, /<dt>Started by<\/dt><dd>Consciousness<\/dd>/, 'its other facts stay');
        // Opening it reads its stored detail and paints its facts again: that paint does not bring the second display back.
        f.detail = settled(id, { ...stamp, authoring_handovers: [row('selected', { selection: 'answer' })] });
        next.querySelector('[data-live-disclosure]').click();
        earlier.open = true;
        for (const toggle of earlier.listeners.get('toggle')) toggle();
        await drain();
        assert.match(facts.innerHTML, /submitted an answer/, 'the stored read painted this episode\'s facts');
        assert.doesNotMatch(facts.innerHTML, /In its words/, 'and still says the sentence once');
        assert.equal(f.reads.length, 1);
    } finally { f.close(); }
});

test('an old attempt\'s GET that finishes after the census names a newer attempt neither revives nor replaces it', async () => {
    const id = 'handover-race';
    const old = settled(id, { authoring_handovers: [row('selected', { selection: 'answer_sha256' })] });
    const f = taskCard(id, old);
    try {
        f.settle();
        await f.toggle();
        assert.match(f.facts(), /<dt>Model handover<\/dt>/, 'attempt 1 says its handover');
        await f.toggle();  // closed
        let release;
        f.gate = new Promise((resolve) => { release = resolve; });
        await f.toggle();  // this read of attempt 1 is still in flight...
        f.census(2);  // ...when the census lists the same task's next attempt
        assert.doesNotMatch(f.facts(), /Model handover/, 'a newer attempt keeps no earlier handover');
        f.gate = null;
        release();
        await drain();
        assert.doesNotMatch(f.facts(), /Model handover/, 'the late read of attempt 1 does not bring its row back');
        await f.toggle();
        await f.toggle();  // nor does it lower the attempt the card knows: attempt 1 read once more stays old
        assert.doesNotMatch(f.facts(), /Model handover/);
        // An unnumbered read after a newer attempt superseded one cannot say which attempt it is.
        f.detail = { ...old, task_attempt: undefined };
        await f.toggle();
        await f.toggle();
        assert.doesNotMatch(f.facts(), /Model handover/);
        // Attempt 2's own stored detail says its history; the census naming it again, or attempt 1, changes nothing.
        f.detail = settled(id, { task_attempt: 2, authoring_handovers: [row('pending', { from_model: FABLE, to_model: OPUS, reply_observed: true })] });
        await f.toggle();
        await f.toggle();
        assert.match(f.facts(), new RegExp(`<dd>${FABLE} → ${OPUS} · no choice recorded</dd>`), f.facts());
        f.census(2);
        f.census(1);
        assert.match(f.facts(), /no choice recorded/);
    } finally { f.close(); }
});

test('an unnumbered stored detail stays readable while no newer attempt superseded one', async () => {
    const legacy = taskCard('legacy', settled('legacy', { task_attempt: undefined,
        authoring_handovers: [row('recovered')] }));
    try {
        legacy.settle();
        await legacy.toggle();
        assert.ok(legacy.facts().includes(`<dt>Model handover</dt><dd>${ASTRA} → ${FABLE} · successor reply not recorded</dd>`), legacy.facts());
        legacy.census(2);  // the first number we observe is already a later attempt
        assert.doesNotMatch(legacy.facts(), /Model handover/);
        legacy.detail = settled('legacy', { task_attempt: undefined, _attempt: 2,
            authoring_handovers: [row('selected', { selection: 'answer' })] });
        await legacy.toggle();
        await legacy.toggle();
        assert.match(legacy.facts(), /submitted an answer/);
    } finally { legacy.close(); }
});

test('a legacy incomplete handover keeps its warning without inventing a Stop or other act; models stay text', async () => {
    const id = 'handover-stopped';
    const stopped = taskCard(id, settled(id, { reason_code: 'authoring_handover_incomplete',
        outcome_axes: { execution: { status: 'degraded' }, objective: { status: 'not_evaluated' } },
        authoring_handovers: [row('incomplete', { selection: 'answer_sha256', incomplete_observed: true })] }));
    try {
        stopped.settle();
        await stopped.toggle();
        assert.equal(stopped.card.querySelector('[data-live-phase]').textContent, 'Done with warnings');
        assert.match(stopped.facts(), /<dt>Model handover<\/dt><dd>[^<]* · handover recorded as incomplete<\/dd>/);
        assert.match(stopped.card.querySelector('[data-live-activity]').textContent, /records an incomplete model handover/);
    } finally { stopped.close(); }
    const plain = taskCard('no-handover', settled('no-handover'));
    try {
        plain.settle();
        await plain.toggle();
        assert.equal(plain.reads.length, 1);
        assert.doesNotMatch(plain.facts(), /Model handover/);
    } finally { plain.close(); }
    const odd = taskCard('odd-model', settled('odd-model', {
        authoring_handovers: [row('recovered', { to_model: 'mock<b>', work_observed: true })] }));
    try {
        odd.settle();
        await odd.toggle();
        assert.ok(odd.facts().includes(`${ASTRA} → mock&lt;b&gt; · completed a tool call`), odd.facts());
    } finally { odd.close(); }
});
