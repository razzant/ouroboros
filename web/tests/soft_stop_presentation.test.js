// S3 stream-gate fixes: MAJOR-A (owner decision №8/Q3) — an owner-requested
// finalization renders as factual SUCCESS "Done", never as a warning — while
// the owner-request marker stays in details. MINOR 7 (Q4): cancel_receipt
// keeps the System render style, never assistant-styled.

import assert from 'node:assert/strict';
import test from 'node:test';
import { readFileSync } from 'node:fs';

import {
    OWNER_STOP_DETAIL_MARKER,
    summarizeChatLiveEvent,
    summarizeLogEvent,
    taskOutcomeSeverity,
    taskStoppedWithSummary,
    taskTerminalPhase,
    taskTerminalSummary,
} from '../modules/log_events.js';
import { senderLabel, buildTimelineItemHtml } from '../modules/chat_activity.js';
import { appendProjectRow } from '../modules/project_answer.js';

const chat = readFileSync(new URL('../modules/chat.js', import.meta.url), 'utf8');

test('Project row delegation preserves the actual producer metadata and terminal options', () => {
    const delegation = chat.match(/\) decorateProjectRow\(bubble, (\{ \.\.\.meta[\s\S]*?\})\);/);
    assert.ok(delegation, 'the chat renderer delegates Project decoration');
    const forward = new Function('meta', 'opts', 'role', 'ts', 'fetchTaskDetailStrict', 'chatMedia', 'withStableViewport',
        'projectRow', `return (${delegation[1]});`);
    const frame = { system_type: 'project_completion_summary', project_id: 'p', project_name: 'Project',
        task_id: 't', completion_answer: 'Result', completion_reason: 'Windows verification remains',
        terminal_time: '2026-10-09T10:00:00Z', ts: '2026-10-09T10:01:00Z', delivery_id: 'delivery' };
    appendProjectRow((_text, role, _markdown, ts, _progress, opts) => {
        const fetchDetail = () => {}, renderFile = () => {}, stableWrite = () => {};
        const forwarded = forward({ systemType: opts.systemType, projectName: 'stale' }, opts, role, ts,
            fetchDetail, { buildDocumentBubble: renderFile }, stableWrite, true);
        assert.equal(forwarded.systemType, frame.system_type);
        assert.equal(forwarded.projectId, frame.project_id);
        assert.equal(forwarded.projectName, frame.project_name);
        assert.equal(forwarded.taskId, frame.task_id);
        assert.equal(forwarded.completionLimitation, frame.completion_reason);
        assert.equal(forwarded.deliveryId, frame.delivery_id);
        assert.equal(forwarded.terminalTime, frame.terminal_time);
        assert.equal(forwarded.addedAt, frame.ts);
        assert.equal(forwarded.role, 'assistant');
        assert.equal(forwarded.fetchDetail, undefined, 'the Main mirror does not fetch a Files inventory');
        assert.equal(forwarded.renderFile, undefined);
        assert.equal(forwarded.onDomWrite, undefined, 'the fold has no in-place expansion');
        assert.equal(forwarded.fold, true, 'a completion mirror in Ouroboros\'s voice folds');
    }, frame, '', {});
});

// The terminal frame shape a soft-stopped task actually publishes: best_effort
// execution (the generic warn trigger) plus the typed owner-requested reason.
const softStop = {
    type: 'task_done',
    status: 'done',
    reason_code: 'owner_requested_finalization',
    outcome_axes: { execution: { status: 'best_effort' } },
};

// --- MAJOR-A: success severity, never warn-styled ---

test('owner-requested finalization classifies as done, not warn', () => {
    assert.equal(taskStoppedWithSummary(softStop), true);
    assert.equal(taskOutcomeSeverity(softStop), 'done');
    assert.equal(taskTerminalPhase(softStop), 'done');
    // Without the owner-requested reason the same best_effort axes still warn —
    // the special case is scoped to exactly this reason code.
    assert.equal(taskOutcomeSeverity({ ...softStop, reason_code: 'deadline' }), 'warn');
});

test('chat live card headline reads factual "Done" with the owner marker', () => {
    const view = summarizeChatLiveEvent(softStop);
    assert.equal(view.headline, 'Done');
    assert.equal(view.phase, 'done');                     // NOT warn-styled
    assert.equal(view.terminal, true);
    assert.ok(view.body.includes(OWNER_STOP_DETAIL_MARKER));
    assert.match(OWNER_STOP_DETAIL_MARKER, /owner's request/);
    assert.match(OWNER_STOP_DETAIL_MARKER, /best available result/);
    assert.doesNotMatch(view.headline, /Finished with warnings/);
});

test('logs surface shows the same factual headline and marker instead of the raw code', () => {
    const view = summarizeLogEvent(softStop);
    assert.equal(view.headline, 'Done');
    assert.equal(view.phase, 'done');                     // NOT warn-styled
    assert.ok(view.meta.includes(OWNER_STOP_DETAIL_MARKER));
    assert.ok(!view.meta.includes('owner_requested_finalization'));
});

test('an expiry kill still reads Stopped — honesty outranks the soft-stop label', () => {
    // When the grace ran out and custody hard-killed, lifecycle=cancelled must
    // win over the soft-stop presentation (the summary was NOT delivered).
    const expired = { ...softStop, status: 'cancelled' };
    assert.equal(taskOutcomeSeverity(expired), 'cancelled');
    assert.equal(summarizeChatLiveEvent(expired).headline, 'Stopped');
});

test('the chat terminal seam shares one note with the owner marker in its body', () => {
    assert.match(chat, /const summary = taskTerminalSummary\(\{ \.\.\.msg, task_id: taskId \}\)/);
    const summary = taskTerminalSummary(softStop);
    assert.equal(summary.phase, 'done');
    assert.equal(summary.headline, 'Done');
    assert.equal(summary.body, OWNER_STOP_DETAIL_MARKER);
    assert.equal(summary.visible, true);
    assert.equal(summary.terminal, true);
    assert.deepEqual(summarizeChatLiveEvent(softStop), summary);
    const priorDocument = globalThis.document;
    globalThis.document = { createElement: () => ({ textContent: '', get innerHTML() { return this.textContent; } }) };
    try {
        const html = buildTimelineItemHtml({ ...summary, lineKey: 'terminal' }, { expandedLineKeys: new Set() });
        assert.match(html, /owner's request/);
        assert.match(html, /best available result/);
        assert.match(html, /chat-live-line done/);
    } finally { globalThis.document = priorDocument; }
});

// --- MINOR 7 (Q4): cancel_receipt rendered as System, not assistant ---

test('a system cancel_receipt row renders the plain System sender label', () => {
    // The receipt is transported role="system", system_type="cancel_receipt"
    // (supervisor/terminal_delivery.py). The extracted shared mapping has no
    // receipt special case, so it must fall through to the generic system label.
    assert.equal(senderLabel('system', false, 'cancel_receipt'), 'System');
});

test('the bubble keeps the system style class and the system_type marker', () => {
    // Rendered style, not just transported role: the bubble class is derived
    // from the role (`chat-bubble system`, never the assistant class), and the
    // system_type lands on the dataset for targeted styling.
    assert.match(chat, /bubble\.className = `chat-bubble \$\{role\}`/);
    // Exercise the renderer's shared metadata assignment, rather than pinning
    // the former one-off spelling of the same system_type propagation.
    const assignment = chat.match(/for \(const key of \[[^\]]*'systemType'[^\]]*\]\) \{\s*if \(meta\[key\]\) bubble\.dataset\[key\] = meta\[key\];\s*\}/)?.[0];
    assert.ok(assignment, 'the bubble renderer must carry systemType metadata');
    const assign = new Function('bubble', 'meta', assignment);
    const bubble = { dataset: {} };
    assign(bubble, { systemType: 'cancel_receipt', taskId: 'stopped-task', deliveryId: 'delivery-1' });
    assert.deepEqual(bubble.dataset, { systemType: 'cancel_receipt', taskId: 'stopped-task', deliveryId: 'delivery-1' });
    const ordinary = { dataset: {} };
    assign(ordinary, { systemType: '' });
    assert.deepEqual(ordinary.dataset, {});
    // History replay forwards system_type through to the renderer.
    assert.match(chat, /systemType: msg\.system_type \|\| ''/);
});
