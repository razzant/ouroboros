// #1585, owner decision 13.10 («Показывать отдельную строку неудачной проверки»): a check
// that ran but whose criterion failed keeps its own expandable diagnostic row in the task
// card, live and on replay, while its call counts once as an ordinary completed call — no
// error count, no task status change. Every assertion below drives the real Chat module.
import assert from 'node:assert/strict';
import test, { after } from 'node:test';
import { createChatInstance } from '../modules/chat.js';
import { noteToolCall, toolEvidenceView } from '../modules/chat_activity.js';
import { summarizeChatLiveEvent } from '../modules/log_events.js';
import { replayedEvidenceTs } from '../modules/chat_render_batch.js';
import { checkFailureCause, checkFailureView, checkProcessSentence } from '../modules/verification_presentation.js';
import { ElementStub, installDom, restoreDom, walkCard } from './chat_dom_fixture.js';

const TS = '2026-10-13T01:00:00Z';
const TASK = 'check-turn';
const KEY = (id = 'inv-1', task = TASK) => `tool:${task}:${id}`;
const verdict = (patch = {}) => ({
    status: 'fail', contract_kind: 'explicit_command', expected_match: 'substring', matched: false,
    check: 'pytest -q tests/test_report.py', check_rendering: 'shlex_join', criterion_id: 'C1',
    criterion_source: 'task_stated', receipt_ts: TS, ...patch,
});
const preview = "verify_and_record [explicit_command] FAIL: exit=0 expected='3 passed'; root=active_workspace";
const call = (id, patch = {}) => ({ task_id: TASK, tool: 'verify_and_record', invocation_id: id, tool_call_id: `c-${id}`, ...patch });
// The loop's two live frames for one settled call: the durable row, then the finished frame.
const settledFrames = (id, fact, extra = {}) => [
    { type: 'tool_call', ...call(id), status: 'ok', is_error: false, exit_code: 0, tool_result_meta: { verification: fact },
        result_preview: preview, ...extra },
    { type: 'tool_call_finished', ...call(id), status: 'ok', is_error: false, exit_code: 0, verification: fact,
        result_preview: preview, ...extra },
];

const originalQuery = ElementStub.prototype.querySelector;
after(() => { ElementStub.prototype.querySelector = originalQuery; });
ElementStub.prototype.querySelector = function (selector) {
    const direct = originalQuery.call(this, selector);
    if (direct) return direct;
    for (const child of this.children) { const found = child.querySelector(selector); if (found) return found; }
    return null;
};
const innerHTMLDescriptor = Object.getOwnPropertyDescriptor(ElementStub.prototype, 'innerHTML');
Object.defineProperty(ElementStub.prototype, 'innerHTML', {
    configurable: true,
    get() { return innerHTMLDescriptor.get.call(this); },
    set(value) {
        innerHTMLDescriptor.set.call(this, value);
        const html = String(value || '');
        let cursor = 0;
        for (const child of this.children) {
            const index = html.indexOf(`<${child.tagName.toLowerCase()}`, cursor);
            if (index < 0) continue;
            child._innerHTML = html.slice(index);
            cursor = index + 1;
        }
    },
});

function fixture(history = []) {
    const env = installDom(async (url) => ({ ok: true, json: async () => String(url).startsWith('/api/chat/history')
        ? { messages: history, window: { complete: true } } : { active_direct_turns: [] } }));
    const handlers = new Map();
    let generation = 0;
    const instance = createChatInstance({
        ws: { on(type, fn) { handlers.set(type, fn); return () => handlers.delete(type); }, isConnected: () => true, send() {} },
        state: { activePage: 'chat', projectChatIds: new Set(), unreadCount: 0 },
        updateUnreadBadge() {}, chatId: 1, idPrefix: 'chat', mountEl: env.mount,
        stateSnapshots: { begin: () => ({ generation: ++generation, requestedAt: Date.now() }), gate() { return Promise.resolve(this.begin()); },
            isCurrent: () => true, apply() {} },
    });
    const messages = document.byId.get('chat-messages');
    // The rows a card itself owns: a nested child card's rows belong to the child.
    const owned = (node, card) => (node?.children || []).flatMap((child) => (child !== card && child.classList?.contains('chat-live-card') ? []
        : [...(child.classList?.contains('chat-live-line') ? [child] : []), ...owned(child, card)]));
    const rows = (id = TASK) => owned(walkCard(messages, id), walkCard(messages, id));
    const card = (id = TASK) => walkCard(messages, id);
    return {
        instance,
        card,
        rows,
        text: (id = TASK) => rows(id).map((row) => row.innerHTML).join(' '),
        checks: (id = TASK) => rows(id).filter((row) => /Check did not pass/.test(row.innerHTML)),
        titles: (id = TASK) => rows(id).map((row) => (row.innerHTML.match(/chat-live-line-title[^>]*>([^<]*)</) || [])[1]),
        meta: (id = TASK) => card(id)?.querySelector('[data-live-meta]')?.innerHTML || '',
        // The card's labelled facts (the tool-call count among them) live in its details list.
        facts: (id = TASK) => card(id)?.querySelector('[data-live-facts]')?.innerHTML || '',
        phase: (id = TASK) => card(id)?.querySelector('[data-live-phase]')?.textContent,
        open: (id = TASK) => {
            if (card(id) && card(id).dataset.expanded !== '1') card(id).querySelector('[data-live-disclosure]').listeners.get('click')[0]({ detail: 0 });
        },
        emit: (type, row) => handlers.get(type)({ chat_id: 1, ts: TS, ...row }),
        log: (row) => handlers.get('log')({ chat_id: 1, data: { task_id: TASK, ts: TS, ...row } }),
        census: (activities) => instance.hydrateStateSnapshot({
            active_chat_activities: activities, active_chat_activities_complete: true, supervisor_ready: true,
        }, Infinity, ++generation),
        close() { instance.destroy(); restoreDom(env.prior); },
    };
}

const done = { task_id: TASK, role: 'assistant', content: 'Report written.', text: 'Report written.',
    task_terminal_status: 'completed', outcome_axes: { execution: { status: 'ok' } },
    accounted_upper_bound_usd: 0.05, cost_final: true, cost_accounting_status: 'available' };

// ---------------------------------------------------------------------------
// The row's words come from typed facts only
// ---------------------------------------------------------------------------

test('cause follows the typed facts in precedence order', () => {
    const cause = (evt, fact) => checkFailureCause(evt, verdict(fact));
    assert.equal(cause({}, {}), 'expected text not found');
    assert.equal(cause({}, { expected_match: 'exact' }), 'output did not match exactly');
    assert.equal(cause({}, { expected_match: 'exact_line' }), 'no output line matched exactly');
    assert.equal(cause({}, { expected_match: 'json_equals' }), 'JSON output not equal');
    // Absent `matched` is no recorded miss: the criterion is unconfirmed, not unmet.
    assert.equal(cause({ exit_code: 0 }, { matched: undefined }), 'criterion not confirmed');
    // A failing exit is the verdict even when the text also missed.
    assert.equal(cause({ exit_code: 1 }, {}), 'exit code 1');
    assert.equal(cause({ signal: 'SIGKILL', exit_code: -9 }, {}), 'killed by SIGKILL');
    // A timeout ran no comparison: no "matched", and the word is the timeout.
    assert.equal(cause({ timed_out: true }, { matched: undefined, detail: 'check timed out after 300s' }), 'timed out');
    // bytes_equal: the producer's own first line. Without a byte verdict there is no `matched`
    // and the line names the missing file or refusal, never "bytes differ".
    assert.equal(cause({ exit_code: 0 }, { expected_match: 'bytes_equal', matched: undefined,
        detail: 'bytes_equal: file not found: b.bin' }), 'bytes_equal: file not found: b.bin');
    assert.equal(cause({ exit_code: 0 }, { expected_match: 'bytes_equal', matched: undefined, detail: '' }), 'comparison not confirmed');
    assert.equal(cause({ exit_code: 0 }, { expected_match: 'bytes_equal',
        detail: 'bytes differ at offset 3 (sizes 6 vs 6).\na.bin @0: 61 62 63' }), 'bytes differ at offset 3 (sizes 6 vs 6).');
});

test('the first sentence says what the process did before the criterion', () => {
    assert.equal(checkProcessSentence({ exit_code: 0 }, { matched: false }), 'The check ran successfully (exit 0); its criterion was not met.');
    assert.equal(checkProcessSentence({ exit_code: 0 }, {}), 'The check ran successfully (exit 0); its comparison reached no verdict, so the criterion is not confirmed.');
    assert.equal(checkProcessSentence({ exit_code: 2 }), 'The check ran and exited with code 2.');
    assert.equal(checkProcessSentence({ exit_code: -9, signal: 'SIGKILL' }), 'The check process was killed by SIGKILL.');
    assert.equal(checkProcessSentence({ timed_out: true }), 'The host stopped the check at its time limit; no comparison ran.');
    assert.equal(checkProcessSentence({}), '', 'no process facts, no claim');
});

test('only a failed verdict on a completed call draws a row', () => {
    assert.equal(checkFailureView({ verification: verdict({ status: 'pass', matched: true }) }, KEY()), null);
    assert.equal(checkFailureView({ verification: verdict(), is_error: true }, KEY()), null, 'a failed call keeps its error row');
    assert.equal(checkFailureView({ result_preview: preview }, KEY()), null, 'text is never read for the verdict');
    assert.equal(checkFailureView({ verification: verdict() }, ''), null);
    const view = checkFailureView({ tool_result_meta: { verification: verdict() }, exit_code: 0, result_preview: preview, ts: TS }, KEY());
    assert.deepEqual({ phase: view.phase, headline: view.headline, dedupeKey: view.dedupeKey }, {
        phase: 'warn', headline: 'Check did not pass: expected text not found', dedupeKey: `${KEY()}:check` });
    assert.equal(view.body, 'The check ran successfully (exit 0); its criterion was not met. '
        + 'Check: pytest -q tests/test_report.py · Criterion: C1');
    assert.ok(view.fullBody.endsWith(preview));
});

test('the live frame keeps the call ok in the fold and gives the verdict its own key', () => {
    const [settledRow, finishedRow] = settledFrames('inv-1', verdict());
    for (const frame of [settledRow, finishedRow]) {
        const view = summarizeChatLiveEvent({ ...frame, _live_tool_frame: true });
        assert.equal(view.phase, 'warn');
        assert.equal(view.headline, 'Check did not pass: expected text not found');
        assert.equal(view.dedupeKey, `${KEY()}:check`);
        assert.deepEqual([view.toolCall.key, view.toolCall.status], [KEY(), 'ok']);
    }
    const passed = summarizeChatLiveEvent({ ...settledFrames('inv-2', verdict({ status: 'pass', matched: true }))[1], _live_tool_frame: true });
    assert.equal(passed.dedupeKey, `tools|${TASK}`, 'a passing check folds like any successful call');
    const record = {};
    noteToolCall(record, summarizeChatLiveEvent(finishedRow).toolCall);
    noteToolCall(record, passed.toolCall);
    const fold = toolEvidenceView(record.toolFold);
    assert.deepEqual([fold.calls, fold.errors, fold.phase, fold.headline], [2, 0, 'result', '2 tool calls']);
});

// ---------------------------------------------------------------------------
// Real Chat: live, reload, ordering and settlement
// ---------------------------------------------------------------------------

test('live: one row per failed check, counted once, a later pass keeps both facts', () => {
    const f = fixture();
    try {
        f.emit('chat', { task_id: TASK, role: 'assistant', is_progress: true, narration: true, content: 'Writing the report' });
        f.log({ type: 'tool_call_started', ...call('inv-1'), _live_tool_frame: true });
        for (const frame of settledFrames('inv-1', verdict())) f.log(frame);
        assert.equal(f.checks().length, 1, 'the settled row and the finished frame are one row');
        assert.match(f.checks()[0].innerHTML, /Check did not pass: expected text not found/);
        assert.match(f.checks()[0].className, /\bwarn\b/);
        f.log({ type: 'tool_call_started', ...call('inv-2'), _live_tool_frame: true });
        for (const frame of settledFrames('inv-2', verdict({ status: 'pass', matched: true }))) f.log(frame);
        assert.equal(f.checks().length, 1, 'the later pass never erases the earlier failure');
        // The header carries the live count (this DOM stub keeps a stale fold node after
        // any diagnostic row, an existing failure row included; the browser test reads it).
        assert.match(f.meta(), /2 tool calls/);
        assert.doesNotMatch(f.meta() + f.text(), /error|One of the steps failed/);
        f.log({ ...done, type: 'task_done', status: 'completed', tool_calls: 2, tool_errors: 0, _is_direct_chat: true });
        assert.equal(f.phase(), 'Done', 'a failed check is not a failed task');
        assert.equal(f.checks().length, 1);
        assert.doesNotMatch(f.meta(), /error/);
    } finally { f.close(); }
});

test('expand shows the check, criterion and the call preview', () => {
    const f = fixture();
    try {
        f.emit('chat', { task_id: TASK, role: 'assistant', is_progress: true, narration: true, content: 'Checking' });
        for (const frame of settledFrames('inv-1', verdict())) f.log(frame);
        const card = f.card();
        if (card.dataset.expanded !== '1') card.querySelector('[data-live-disclosure]').listeners.get('click')[0]({ detail: 0 });
        const row = f.checks()[0];
        assert.match(row.innerHTML, /The check ran successfully \(exit 0\); its criterion was not met\. Check: pytest -q tests\/test_report.py · Criterion: C1/);
        // Expandable like every diagnostic row; the expanded body (check, criterion, the
        // call's own preview) is pinned on the view above and opened in the browser test.
        assert.match(row.className, /\bexpandable\b/);
        assert.match(row.innerHTML, /data-live-line-toggle=.*aria-expanded="false"/s);
    } finally { f.close(); }
});

const carrier = (observations, extra = {}) => ({ ...done, role: 'system', system_type: 'task_summary', text: 'Done',
    ts: TS, chat_id: 1, tool_calls: observations.filter((o) => o.fact === 'settled').length, tool_errors: 0,
    tool_evidence: { observations }, ...extra });
const replayed = (id, fact) => {
    const base = { key: KEY(id), tool: 'verify_and_record', receipt: false, live: false };
    return [{ ...base, fact: 'started', status: 'unknown', hostError: false },
        { ...base, fact: 'settled', status: 'ok', hostError: false, verification: fact, ts: TS, exit_code: 0,
            ...(fact.status === 'fail' ? { result_preview: preview } : {}) }];
};

test('reload draws the same row from replayed evidence, and a live duplicate stays one row', async () => {
    const f = fixture([carrier([...replayed('inv-1', verdict()), ...replayed('inv-2', verdict({ status: 'pass', matched: true }))])]);
    try {
        await f.instance.refreshHistory({ revision: 1 });
        const card = f.card();
        if (card.dataset.expanded !== '1') card.querySelector('[data-live-disclosure]').listeners.get('click')[0]({ detail: 0 });
        assert.equal(f.checks().length, 1);
        assert.match(f.checks()[0].innerHTML, /Check did not pass: expected text not found/);
        assert.match(f.text() + f.meta(), /2 tool calls/);
        assert.doesNotMatch(f.text() + f.meta(), /error/);
        assert.equal(f.phase(), 'Done');
        for (const frame of settledFrames('inv-1', verdict())) f.log(frame);
        await f.instance.refreshHistory({ revision: 2 });
        assert.equal(f.checks().length, 1, 'live, replay and a second replay share one key');
    } finally { f.close(); }
});

test('a check settled after the terminal row still gets its row, without reopening the task', () => {
    const f = fixture();
    try {
        f.emit('chat', { task_id: TASK, role: 'assistant', is_progress: true, narration: true, content: 'Finishing' });
        f.log({ type: 'tool_call_started', ...call('inv-1'), _live_tool_frame: true });
        f.log({ type: 'tool_call_timeout', ...call('inv-1') });
        f.log({ ...done, type: 'task_done', status: 'completed', tool_calls: 1, tool_errors: 1, _is_direct_chat: true });
        for (const frame of settledFrames('inv-1', verdict({ matched: undefined, detail: 'check timed out after 300s' }),
            { timed_out: true, exit_code: undefined })) f.log(frame);
        const card = f.card();
        if (card.dataset.expanded !== '1') card.querySelector('[data-live-disclosure]').listeners.get('click')[0]({ detail: 0 });
        assert.equal(f.checks().length, 1);
        assert.match(f.checks()[0].innerHTML, /Check did not pass: timed out/);
        // Late success settlement retires the provisional wait notice; no error is resurrected.
        assert.doesNotMatch(f.text() + f.meta(), /operation may still settle|One of the steps failed|1 error/);
        assert.match(f.meta(), /1 tool call/);
        assert.equal(f.phase(), 'Done');
    } finally { f.close(); }
});

test('a check whose receipt was not recorded is a failed call, not a failed criterion', () => {
    const f = fixture();
    try {
        f.emit('chat', { task_id: TASK, role: 'assistant', is_progress: true, narration: true, content: 'Checking' });
        f.log({ type: 'tool_call_finished', ...call('inv-1'), status: 'error', is_error: true,
            result_preview: '⚠️ TOOL_ERROR (verify_and_record): receipt_custody_failed for explicit_command' });
        assert.equal(f.checks().length, 0);
        assert.match(f.text(), /One of the steps failed · verify_and_record/);
        assert.match(f.meta(), /1 error/);
    } finally { f.close(); }
});

// ---------------------------------------------------------------------------
// A comparison that reached no verdict is not a missed criterion
// ---------------------------------------------------------------------------

test('a bytes_equal comparison without a byte verdict is "not confirmed", never "not met" or "differ"', () => {
    const infra = 'bytes_equal infra error (cmp exit 127, not a byte verdict) (a.bin vs b.bin): cmp: not found';
    const view = checkFailureView({ verification: verdict({ expected_match: 'bytes_equal', matched: undefined, detail: infra }),
        exit_code: 0, result_preview: preview, ts: TS }, KEY());
    assert.equal(view.headline, `Check did not pass: ${infra}`);
    assert.match(view.body, /^The check ran successfully \(exit 0\); its comparison reached no verdict, so the criterion is not confirmed\./);
    assert.doesNotMatch(`${view.headline} ${view.body}`, /was not met|bytes differ/);
    // The same exit with a byte verdict that the files differ keeps the plain statement.
    const differ = checkFailureView({ verification: verdict({ expected_match: 'bytes_equal', matched: false,
        detail: 'bytes differ at offset 3 (sizes 6 vs 6).' }), exit_code: 0, ts: TS }, KEY());
    assert.match(differ.body, /^The check ran successfully \(exit 0\); its criterion was not met\./);
    assert.equal(differ.headline, 'Check did not pass: bytes differ at offset 3 (sizes 6 vs 6).');
    // A host kill or failing exit is still the cause: no comparison is claimed either way.
    assert.match(checkFailureView({ verification: verdict({ expected_match: 'bytes_equal', matched: undefined }),
        exit_code: 2, ts: TS }, KEY()).body, /^The check ran and exited with code 2\./);
});

// ---------------------------------------------------------------------------
// Order: the evidence row first, then its diagnostic — live and on reload
// ---------------------------------------------------------------------------

const T = (clock) => `2026-10-13T01:${clock}Z`;
const settled = (id, fact, ts, extra = {}) => settledFrames(id, fact, { ts, ...extra });
const summaryRow = (observations, ts, extra = {}) => ({ ...carrier(observations, { ts }), ...extra,
    history_id: `chat:${ts}`, history_position: { source: 'chat', offset: Date.parse(ts) } });
const progressRow = (taskId, text, ts, extra = {}) => ({ task_id: taskId, role: 'assistant', is_progress: true, narration: true,
    content: text, text, ts, chat_id: 1, history_id: `progress:${taskId}:${ts}`,
    history_position: { source: 'progress', offset: Date.parse(ts) }, ...extra });

test('reload orders the evidence row before its check, as the live card does', async () => {
    const live = fixture();
    let liveTitles;
    try {
        live.emit('chat', { task_id: TASK, role: 'assistant', is_progress: true, narration: true, content: 'Writing the report', ts: T('00:00') });
        live.log({ type: 'tool_call_started', ...call('inv-1'), _live_tool_frame: true, ts: T('00:05') });
        for (const frame of settled('inv-1', verdict(), T('00:20'))) live.log(frame);
        live.log({ ...done, type: 'task_done', status: 'completed', tool_calls: 1, tool_errors: 0, _is_direct_chat: true, ts: T('01:00') });
        liveTitles = live.titles();
    } finally { live.close(); }
    assert.deepEqual(liveTitles, ['Writing the report', '1 tool call', 'Check did not pass: expected text not found', 'Done']);
    // The same facts as the gateway replays them: the terminal summary carries the evidence at
    // ITS time (01:00), after the check settled (00:20) — the check must not sort ahead of it.
    const replay = fixture([progressRow(TASK, 'Writing the report', T('00:00')),
        summaryRow(replayed('inv-1', verdict()).map((o) => (o.fact === 'settled' ? { ...o, ts: T('00:20') } : o)), T('01:00'))]);
    try {
        await replay.instance.refreshHistory({ revision: 1 });
        replay.open();
        assert.deepEqual(replay.titles(), liveTitles);
        // This producer's replayed observations carry a time only on a settled check, so that is
        // the earliest valid host time the row can stand at (see the started-time test below).
        const times = replay.rows().map((row) => row.innerHTML.match(/chat-live-line-time[^>]*>([^<]*)</)?.[1]);
        assert.equal(times[1], times[2], 'the evidence row stands at the check\'s settlement, not at the terminal');
    } finally { replay.close(); }
});

test('a check that settled after the terminal row stays behind it, live and on reload', async () => {
    const live = fixture();
    let liveTitles;
    try {
        live.emit('chat', { task_id: TASK, role: 'assistant', is_progress: true, narration: true, content: 'Finishing', ts: T('00:00') });
        live.log({ type: 'tool_call_started', ...call('inv-1'), _live_tool_frame: true, ts: T('00:05') });
        live.log({ type: 'tool_call_timeout', ...call('inv-1'), ts: T('00:10') });
        live.log({ ...done, type: 'task_done', status: 'completed', tool_calls: 1, tool_errors: 1, _is_direct_chat: true, ts: T('01:00') });
        for (const frame of settled('inv-1', verdict(), T('01:30'))) live.log(frame);
        liveTitles = live.titles();
        assert.equal(live.phase(), 'Done');
    } finally { live.close(); }
    assert.deepEqual(liveTitles, ['Finishing', '1 tool call', 'Done', 'Check did not pass: expected text not found']);
    const replay = fixture([progressRow(TASK, 'Finishing', T('00:00')),
        summaryRow(replayed('inv-1', verdict()).map((o) => (o.fact === 'settled' ? { ...o, ts: T('01:30') } : o)), T('01:00'))]);
    try {
        await replay.instance.refreshHistory({ revision: 1 });
        replay.open();
        assert.deepEqual(replay.titles(), liveTitles);
        assert.equal(replay.phase(), 'Done');
        assert.doesNotMatch(replay.meta() + replay.text(), /error/);
    } finally { replay.close(); }
});

// ---------------------------------------------------------------------------
// A child card: the same row, live, after its terminal row, and on reload
// ---------------------------------------------------------------------------

const CHILD = 'check-child';
const managed = () => [{ activity_id: TASK, chat_id: 1, kind: 'managed_task', phase: 'working' }];
const childNote = (patch = {}) => ({ role: 'assistant', is_progress: true, task_id: CHILD, subagent_task_id: CHILD,
    parent_task_id: TASK, root_task_id: TASK, delegation_role: 'subagent', subagent_event: 'progress', subagent_role: 'scout',
    model: 'm', ts: T('00:01'), ...patch });
const childCall = (id) => call(id, { task_id: CHILD });
const childSettled = (id, fact, ts, extra = {}) => settledFrames(id, fact, { ts, ...extra })
    .map((frame) => ({ ...frame, ...childCall(id) }));

function startChild(f) {
    f.census(managed());
    f.emit('chat', { task_id: TASK, role: 'assistant', is_progress: true, narration: true, content: 'Planning', ts: T('00:00') });
    f.emit('chat', childNote({ content: 'Scouting', narration: true }));
    assert.ok(f.card(CHILD), 'the child frame mints the child card');
    assert.equal(f.card(CHILD).dataset.parentTaskId, TASK);
}

test('child live: its own row, counted once on the child, nothing on the root, a later pass keeps both', () => {
    const f = fixture();
    try {
        startChild(f);
        f.log({ type: 'tool_call_started', ...childCall('c-1'), _live_tool_frame: true, ts: T('00:05') });
        for (const frame of childSettled('c-1', verdict({ criterion_id: 'CHILD-1' }), T('00:20'))) f.log(frame);
        f.log({ type: 'tool_call_started', ...childCall('c-2'), _live_tool_frame: true, ts: T('00:25') });
        for (const frame of childSettled('c-2', verdict({ status: 'pass', matched: true }), T('00:30'))) f.log(frame);
        f.open(TASK); f.open(CHILD);
        assert.equal(f.checks(CHILD).length, 1, 'one failed check, one row, on the child');
        assert.match(f.checks(CHILD)[0].innerHTML, /Criterion: CHILD-1/);
        assert.equal(f.checks(TASK).length, 0, 'the root card draws nothing for its child\'s check');
        assert.match(f.meta(CHILD), /2 tool calls/);
        assert.doesNotMatch(f.meta(CHILD) + f.text(CHILD), /error|One of the steps failed/);
        assert.doesNotMatch(f.meta(TASK), /error|tool call/);
        assert.equal(f.phase(TASK), 'Working', 'the child\'s check changes no task status');
        f.emit('chat', childNote({ subagent_event: 'completed', status: 'completed', result: 'Scouted.', content: 'Scouted.',
            narration: false, ts: T('00:40') }));
        assert.equal(f.phase(CHILD), 'Done');
        assert.equal(f.checks(CHILD).length, 1, 'the child\'s terminal keeps the row');
        assert.equal(f.checks(TASK).length, 0);
    } finally { f.close(); }
});

test('child: a wait that ended, the child\'s terminal, then the handled FAIL settles late', () => {
    const f = fixture();
    try {
        startChild(f);
        f.log({ type: 'tool_call_started', ...childCall('c-1'), _live_tool_frame: true, ts: T('00:05') });
        f.log({ type: 'tool_call_timeout', ...childCall('c-1'), ts: T('00:10') });
        f.emit('chat', childNote({ subagent_event: 'completed', status: 'completed', result: 'Scouted.', content: 'Scouted.',
            narration: false, ts: T('00:30') }));
        f.open(TASK); f.open(CHILD);
        assert.equal(f.phase(CHILD), 'Done');
        assert.equal(f.checks(CHILD).length, 0);
        for (const frame of childSettled('c-1', verdict({ matched: undefined, detail: 'check timed out after 300s' }), T('00:40'),
            { timed_out: true, exit_code: undefined })) f.log(frame);
        assert.equal(f.checks(CHILD).length, 1, 'the late settlement still draws its row on the finished child');
        assert.match(f.checks(CHILD)[0].innerHTML, /Check did not pass: timed out/);
        // The provisional notice retires on a late success; no wait error is resurrected.
        assert.doesNotMatch(f.text(CHILD) + f.meta(CHILD), /operation may still settle|One of the steps failed|\d error/);
        assert.equal(f.phase(CHILD), 'Done', 'a late check never reopens the child');
        assert.equal(f.checks(TASK).length, 0);
        assert.equal(f.phase(TASK), 'Working');
        // The late frames repeat (settled row, finished frame, a second delivery): still one row.
        for (const frame of childSettled('c-1', verdict({ matched: undefined, detail: 'check timed out after 300s' }), T('00:40'),
            { timed_out: true, exit_code: undefined })) f.log(frame);
        assert.equal(f.checks(CHILD).length, 1);
    } finally { f.close(); }
});

test('child reload: the carrier replays the row on the child, and live frames share its key', async () => {
    const childCarrier = (ts) => ({ task_id: CHILD, role: 'system', system_type: 'task_evidence', text: '', ts, chat_id: 1,
        narration: false, tool_calls: 1, tool_errors: 0, history_id: `evidence:${CHILD}`,
        history_position: { source: 'evidence', offset: 9 },
        tool_evidence: { observations: replayed('c-1', verdict({ criterion_id: 'CHILD-1' })).map((o) => ({
            ...o, key: KEY('c-1', CHILD), ...(o.fact === 'settled' ? { ts: T('00:20') } : {}) })) } });
    const history = [progressRow(TASK, 'Planning', T('00:00')),
        { ...childNote({ content: 'scheduled', text: 'scheduled', narration: false, subagent_event: 'scheduled' }), chat_id: 1,
            history_id: 'progress:child:1', history_position: { source: 'progress', offset: 1 } },
        { ...childNote({ content: 'Scouted.', text: 'Scouted.', narration: false, subagent_event: 'completed',
            status: 'completed', result: 'Scouted.', ts: T('00:30') }), chat_id: 1,
            history_id: 'progress:child:2', history_position: { source: 'progress', offset: 2 } },
        childCarrier(T('00:01'))];
    const f = fixture(history);
    try {
        f.census(managed());
        await f.instance.refreshHistory({ revision: 1 });
        f.open(TASK); f.open(CHILD);
        assert.equal(f.card(CHILD).dataset.parentTaskId, TASK, 'the child nests under its root');
        assert.equal(f.checks(CHILD).length, 1);
        assert.equal(f.checks(TASK).length, 0);
        assert.deepEqual(f.titles(CHILD).filter((title) => /tool call|Check did not pass/.test(title)),
            ['1 tool call', 'Check did not pass: expected text not found'], 'the evidence row, then its check');
        assert.match(f.meta(CHILD), /1 tool call/);
        assert.doesNotMatch(f.meta(CHILD) + f.text(CHILD), /error/);
        assert.equal(f.phase(CHILD), 'Done');
        // A live duplicate after the reload, and a second replay, change nothing.
        for (const frame of childSettled('c-1', verdict({ criterion_id: 'CHILD-1' }), T('00:20'))) f.log(frame);
        await f.instance.refreshHistory({ revision: 2 });
        assert.equal(f.checks(CHILD).length, 1);
    } finally { f.close(); }
});

test('child reload: a check that settled after the child ended is still one row behind its terminal', async () => {
    const history = [progressRow(TASK, 'Planning', T('00:00')),
        { ...childNote({ content: 'Scouted.', text: 'Scouted.', narration: false, subagent_event: 'completed',
            status: 'completed', result: 'Scouted.', ts: T('00:30') }), chat_id: 1,
            history_id: 'progress:child:2', history_position: { source: 'progress', offset: 2 } },
        { task_id: CHILD, role: 'system', system_type: 'task_evidence', text: '', ts: T('00:01'), chat_id: 1, narration: false,
            tool_calls: 1, tool_errors: 0, history_id: `evidence:${CHILD}`, history_position: { source: 'evidence', offset: 9 },
            tool_evidence: { observations: replayed('c-1', verdict()).map((o) => ({ ...o, key: KEY('c-1', CHILD),
                ...(o.fact === 'settled' ? { ts: T('00:40') } : {}) })) } }];
    const f = fixture(history);
    try {
        f.census(managed());
        await f.instance.refreshHistory({ revision: 1 });
        f.open(TASK); f.open(CHILD);
        assert.equal(f.checks(CHILD).length, 1);
        const titles = f.titles(CHILD);
        assert.ok(titles.indexOf('Check did not pass: expected text not found') > titles.indexOf('1 tool call'),
            `the evidence row precedes its check: ${titles}`);
        assert.equal(f.phase(CHILD), 'Done');
        assert.equal(f.checks(TASK).length, 0);
    } finally { f.close(); }
});

// ---------------------------------------------------------------------------
// The evidence row stands at its FIRST call's time (DESIGN: "at the first call's position and
// time"), from the earliest valid host observation of ANY call — not from a failed check
// ---------------------------------------------------------------------------

const obs = (id, patch = {}, tool = 'verify_and_record') => ({ key: KEY(id), tool, receipt: false, live: false, hostError: false, ...patch });
// A call as replay delivers it once EVERY observation carries its host time: started, then settled.
const timedCall = (id, started, settledAt, extra = {}, tool = 'verify_and_record') => [
    obs(id, { fact: 'started', status: 'unknown', ts: started }, tool),
    obs(id, { fact: 'settled', status: 'ok', ts: settledAt, ...extra }, tool)];
const bareTitles = (titles) => titles.filter((title) => /tool call|Check did not pass|^Writing|^Still|^Done$/.test(title));

test('a time on every observation: the row stands at the started call, narration around it stays in order', async () => {
    // narration 00:00 · first call started 00:05 · narration 00:10 · failed check settled 00:20 · terminal 01:00
    const live = fixture();
    let liveTitles;
    try {
        live.emit('chat', { task_id: TASK, role: 'assistant', is_progress: true, narration: true, content: 'Writing the report', ts: T('00:00') });
        live.log({ type: 'tool_call_started', ...call('inv-1'), _live_tool_frame: true, ts: T('00:05') });
        live.emit('chat', { task_id: TASK, role: 'assistant', is_progress: true, narration: true, content: 'Still working', ts: T('00:10') });
        for (const frame of settled('inv-1', verdict(), T('00:20'))) live.log(frame);
        live.log({ ...done, type: 'task_done', status: 'completed', tool_calls: 1, tool_errors: 0, _is_direct_chat: true, ts: T('01:00') });
        liveTitles = bareTitles(live.titles());
    } finally { live.close(); }
    assert.deepEqual(liveTitles, ['Writing the report', '1 tool call', 'Still working', 'Check did not pass: expected text not found', 'Done']);
    const observations = timedCall('inv-1', T('00:05'), T('00:20'), { verification: verdict(), exit_code: 0, result_preview: preview });
    for (const rows of [
        [progressRow(TASK, 'Writing the report', T('00:00')), progressRow(TASK, 'Still working', T('00:10')), summaryRow(observations, T('01:00'))],
        // History arrives in another order: the carrier first, the narration after it.
        [summaryRow(observations, T('01:00')), progressRow(TASK, 'Still working', T('00:10')), progressRow(TASK, 'Writing the report', T('00:00'))],
    ]) {
        const replay = fixture(rows);
        try {
            await replay.instance.refreshHistory({ revision: 1 });
            replay.open();
            assert.deepEqual(bareTitles(replay.titles()), liveTitles, 'reload draws the live order');
            assert.equal(replay.checks().length, 1);
        } finally { replay.close(); }
    }
});

test('any call\'s time places the row, with no failed check at all', async () => {
    const observations = [...timedCall('r-1', T('00:05'), T('00:06'), {}, 'read_file'), ...timedCall('r-2', T('00:12'), T('00:13'), {}, 'read_file')];
    const replay = fixture([progressRow(TASK, 'Writing the report', T('00:00')), progressRow(TASK, 'Still working', T('00:10')),
        summaryRow(observations, T('01:00'))]);
    try {
        await replay.instance.refreshHistory({ revision: 1 });
        replay.open();
        assert.equal(replay.checks().length, 0);
        assert.deepEqual(bareTitles(replay.titles()), ['Writing the report', '2 tool calls', 'Still working', 'Done'],
            'the row stands at its first call (00:05), before the narration at 00:10');
    } finally { replay.close(); }
});

test('without a valid observation time the carrier\'s placement is unchanged', async () => {
    for (const ts of [undefined, null, '', 'not-a-time', 12, '2026', {}]) {
        const observations = timedCall('r-1', ts, ts, {}, 'read_file');
        const replay = fixture([progressRow(TASK, 'Writing the report', T('00:00')), progressRow(TASK, 'Still working', T('00:10')),
            summaryRow(observations, T('01:00'))]);
        try {
            await replay.instance.refreshHistory({ revision: 1 });
            replay.open();
            assert.deepEqual(bareTitles(replay.titles()), ['Writing the report', 'Still working', '1 tool call', 'Done'],
                `time ${JSON.stringify(ts)} is no host time: the row keeps the carrier's place`);
        } finally { replay.close(); }
    }
});

test('replayedEvidenceTs: the earliest valid host time, never later than the carrier, else the carrier', () => {
    const at = (observations, carrier = T('01:00')) => replayedEvidenceTs({ observations }, carrier);
    assert.equal(at([{ ts: T('00:20') }, { ts: T('00:05') }, { ts: T('00:12') }]), T('00:05'), 'any order, any fact');
    assert.equal(at([{ fact: 'wait_ended', ts: T('00:07') }, { fact: 'settled', ts: T('00:20') }]), T('00:07'));
    assert.equal(at([{ ts: 'x' }, { ts: T('00:30') }, {}, { ts: null }, { ts: 5 }]), T('00:30'), 'invalid times are skipped, not the call');
    assert.equal(at([{ ts: T('01:30') }]), T('01:00'), 'a settlement after the carrier never drags the row behind it');
    assert.equal(at([{ ts: T('01:00') }]), T('01:00'));
    assert.equal(at([{}, { ts: undefined }, { ts: '' }, { ts: 'tomorrow' }, { ts: '2026' }]), T('01:00'), 'no valid time: the carrier');
    assert.equal(at([]), T('01:00'));
    assert.equal(replayedEvidenceTs(undefined, T('01:00')), T('01:00'));
    assert.equal(replayedEvidenceTs({}, T('01:00')), T('01:00'));
    // A carrier with no usable time leaves the host's own time as the best there is.
    assert.equal(at([{ ts: T('00:05') }], ''), T('00:05'));
    assert.equal(at([{ ts: T('00:05') }], undefined), T('00:05'));
    assert.equal(at([], ''), '', 'nothing valid anywhere: unchanged');
});

test('a late-settled check keeps the row before the terminal, and its own row after it', async () => {
    // started 00:05 (host time), wait ended, terminal 01:00, handled FAIL settles 01:30
    const observations = [obs('inv-1', { fact: 'started', status: 'unknown', ts: T('00:05') }),
        obs('inv-1', { fact: 'wait_ended', status: 'unknown', ts: T('00:10') }),
        obs('inv-1', { fact: 'settled', status: 'ok', ts: T('01:30'), verification: verdict(), exit_code: 0, result_preview: preview })];
    const replay = fixture([progressRow(TASK, 'Writing the report', T('00:00')), progressRow(TASK, 'Still working', T('00:20')),
        summaryRow(observations, T('01:00'))]);
    try {
        await replay.instance.refreshHistory({ revision: 1 });
        replay.open();
        assert.deepEqual(bareTitles(replay.titles()),
            ['Writing the report', '1 tool call · wait ended', 'Still working', 'Done', 'Check did not pass: expected text not found']);
    } finally { replay.close(); }
});


test('a recorded child start outranks an earlier inert history carrier', async () => {
    const observations = timedCall('c-1', T('00:05'), T('00:20'), {
        verification: verdict({ criterion_id: 'CHILD-1' }), exit_code: 0, result_preview: preview,
    }).map((row) => ({ ...row, key: KEY('c-1', CHILD) }));
    assert.equal(replayedEvidenceTs({ observations }, T('00:01')), T('00:05'),
        'an inert carrier is not evidence that the call already started');
    const history = [progressRow(TASK, 'Planning', T('00:00')),
        { ...childNote({ content: 'Still preparing', text: 'Still preparing', narration: true, ts: T('00:03') }), chat_id: 1,
            history_id: 'progress:child:1', history_position: { source: 'progress', offset: 1 } },
        { ...childNote({ content: 'Scouted.', text: 'Scouted.', narration: false, subagent_event: 'completed',
            status: 'completed', result: 'Scouted.', ts: T('00:30') }), chat_id: 1,
            history_id: 'progress:child:2', history_position: { source: 'progress', offset: 2 } },
        { task_id: CHILD, role: 'system', system_type: 'task_evidence', text: '', ts: T('00:01'), chat_id: 1,
            narration: false, tool_calls: 1, tool_errors: 0, tool_evidence: { observations },
            history_id: `evidence:${CHILD}`, history_position: { source: 'evidence', offset: 9 } }];
    const f = fixture(history);
    try {
        f.census(managed());
        await f.instance.refreshHistory({ revision: 1 });
        f.open(TASK); f.open(CHILD);
        const titles = f.titles(CHILD);
        assert.ok(titles.indexOf('Still preparing') < titles.indexOf('1 tool call'), titles.join(' | '));
        assert.ok(titles.indexOf('1 tool call') < titles.indexOf('Check did not pass: expected text not found'));
        assert.equal(f.checks(CHILD).length, 1);
    } finally { f.close(); }
});
