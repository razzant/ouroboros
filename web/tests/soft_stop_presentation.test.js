// S3 stream-gate fixes: MAJOR-A (owner decision №8/Q3) — an owner-requested
// finalization renders as the SUCCESS "Stopped with summary", never as
// "Finished with warnings" — and MINOR 7 (Q4) — the cancel_receipt system row
// keeps the 📋 System render style, never assistant-styled.

import assert from 'node:assert/strict';
import test from 'node:test';
import { readFileSync } from 'node:fs';

import {
    OWNER_STOP_DETAIL_MARKER,
    OWNER_STOP_DONE_HEADLINE,
    summarizeChatLiveEvent,
    summarizeLogEvent,
    taskOutcomeSeverity,
    taskStoppedWithSummary,
    taskTerminalPhase,
} from '../modules/log_events.js';
import { createMessageIdentity } from '../modules/chat_message_identity.js';

const chat = readFileSync(new URL('../modules/chat.js', import.meta.url), 'utf8');
const messageIdentity = readFileSync(new URL('../modules/chat_message_identity.js', import.meta.url), 'utf8');

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

test('chat live card headline reads "Stopped with summary" with the owner marker', () => {
    const view = summarizeChatLiveEvent(softStop);
    assert.equal(view.headline, OWNER_STOP_DONE_HEADLINE);
    assert.equal(view.headline, 'Stopped with summary');
    assert.equal(view.phase, 'done');                     // NOT warn-styled
    assert.equal(view.terminal, true);
    assert.ok(view.meta.includes(OWNER_STOP_DETAIL_MARKER));
    assert.match(OWNER_STOP_DETAIL_MARKER, /owner's request/);
    assert.match(OWNER_STOP_DETAIL_MARKER, /best available result/);
    assert.doesNotMatch(view.headline, /Finished with warnings/);
});

test('logs surface shows the same headline and marker instead of the raw code', () => {
    const view = summarizeLogEvent(softStop);
    assert.equal(view.headline, OWNER_STOP_DONE_HEADLINE);
    assert.equal(view.phase, 'done');                     // NOT warn-styled
    assert.ok(view.meta.includes(OWNER_STOP_DETAIL_MARKER));
    assert.ok(!view.meta.includes('owner_requested_finalization'));
});

test('an expiry kill still reads Cancelled — honesty outranks the soft-stop label', () => {
    // When the grace ran out and custody hard-killed, lifecycle=cancelled must
    // win over the soft-stop presentation (the summary was NOT delivered).
    const expired = { ...softStop, status: 'cancelled' };
    assert.equal(taskOutcomeSeverity(expired), 'cancelled');
    assert.equal(summarizeChatLiveEvent(expired).headline, 'Cancelled');
});

test('the chat.js terminal seam consumes the shared soft-stop branch', () => {
    // Pinned at source: the live-card done headline branches through the SAME
    // shared predicate/constants (no divergent inline string), and the details
    // panel body carries the owner-request marker. The seam moved with
    // appendTaskSummaryToLiveCard into the task-frame router (W3 wave D).
    const frames = readFileSync(new URL('../modules/chat_task_frames.js', import.meta.url), 'utf8');
    assert.match(frames, /taskStoppedWithSummary\(msg \|\| \{\}\)/);
    assert.match(frames, /\? OWNER_STOP_DONE_HEADLINE/);
    assert.match(frames, /softStopped \? OWNER_STOP_DETAIL_MARKER : ''/);
    assert.match(frames, /\[softStopDetail, reviewDetails\]\.filter\(Boolean\)\.join\('\\n'\)/);
    assert.match(frames, /visible: Boolean\(softStopDetail \|\| reviewDetails\)/);
});

// --- MINOR 7 (Q4): cancel_receipt rendered as 📋 System, not assistant ---

test('a system cancel_receipt row renders the 📋 System sender label', () => {
    // The receipt is transported role="system", system_type="cancel_receipt"
    // (supervisor/terminal_delivery.py). getSenderLabel has no special case for
    // it, so it must fall through to the generic system label — pin the branch
    // so a future mapping cannot silently restyle receipts as assistant text.
    // getSenderLabel is owned by web/modules/chat_message_identity.js.
    const { getSenderLabel } = createMessageIdentity({
        chatSessionId: 'session-a', seenMessageKeys: new Set(), messageKeyOrder: [],
    });
    assert.equal(getSenderLabel('system', false, 'cancel_receipt'), '📋 System');
    const senderFn = messageIdentity.slice(messageIdentity.indexOf('function getSenderLabel'));
    const systemBranch = senderFn.slice(0, senderFn.indexOf("if (isProgress)"));
    assert.match(systemBranch, /if \(role === 'system'\) \{/);
    assert.match(systemBranch, /return '📋 System';/);
    // No cancel_receipt→assistant diversion anywhere in the sender mapping.
    assert.doesNotMatch(senderFn.slice(0, 600), /cancel_receipt/);
});

test('the bubble keeps the system style class and the system_type marker', () => {
    // Rendered style, not just transported role: the bubble class is derived
    // from the role (`chat-bubble system`, never the assistant class), and the
    // system_type lands on the dataset for targeted styling.
    // addMessage and the history replay moved with the feed owner (W3 wave D).
    const feed = readFileSync(new URL('../modules/chat_history_sync.js', import.meta.url), 'utf8');
    assert.match(feed, /bubble\.className = `chat-bubble \$\{role\}`/);
    assert.match(feed, /if \(systemType\) bubble\.dataset\.systemType = systemType;/);
    // History replay forwards system_type through to the renderer.
    assert.match(feed, /systemType: msg\.system_type \|\| ''/);
});
