// S3 (Q2/HQ1): the shared three-action task stop/hurry control — exact owner
// wording, action gating around a pending cancel, stable hurry request-id
// reuse, and the no-chat-bubble contract pinned at source for both surfaces.

import assert from 'node:assert/strict';
import test from 'node:test';
import { readFileSync } from 'node:fs';

import {
    ACTION_FINALIZE,
    ACTION_HURRY,
    ACTION_STOP_NOW,
    TASK_CONTROL_LABELS,
    TASK_CONTROL_TRIGGER_LABEL,
    hurryRequestId,
    stopPolicyFor,
    taskControlActions,
} from '../modules/task_control_menu.js';
import { ownerHurryProjection, summarizeChatLiveEvent, taskSoftStopPending } from '../modules/log_events.js';

const chat = readFileSync(new URL('../modules/chat.js', import.meta.url), 'utf8');
const cardActions = readFileSync(new URL('../modules/chat_card_actions.js', import.meta.url), 'utf8');
const activity = readFileSync(new URL('../modules/activity.js', import.meta.url), 'utf8');
const menuSrc = readFileSync(new URL('../modules/task_control_menu.js', import.meta.url), 'utf8');

// --- the frozen owner dropdown (Q2/HQ1) ---

test('the dropdown offers exactly the three owner-decided actions, in order', () => {
    assert.deepEqual(taskControlActions(), [ACTION_FINALIZE, ACTION_HURRY, ACTION_STOP_NOW]);
    assert.equal(TASK_CONTROL_LABELS[ACTION_FINALIZE], 'Wrap up');
    assert.equal(TASK_CONTROL_LABELS[ACTION_HURRY], 'Hurry up');
    assert.equal(TASK_CONTROL_LABELS[ACTION_STOP_NOW], 'Stop now');
});

test('a pending cancel offers ONLY the hard escalation — hurry is never shown then', () => {
    // Q1: the hard stop stays reachable DURING the soft-stop wait as the
    // monotonic escalation of the SAME intent; HQ1: a pending cancel refuses
    // hurry, so the menu does not offer it.
    assert.deepEqual(taskControlActions({ cancelPending: true }), [ACTION_STOP_NOW]);
});

test('stop actions map to the wire stop_policy; hurry is not a stop', () => {
    assert.equal(stopPolicyFor(ACTION_FINALIZE), 'finalize_then_cancel');
    assert.equal(stopPolicyFor(ACTION_STOP_NOW), 'immediate');
    assert.equal(stopPolicyFor(ACTION_HURRY), '');
});

// --- stable request-id reuse (HQ1 idempotent retry) ---

test('hurryRequestId is stable per task and distinct across tasks', () => {
    const first = hurryRequestId('t-1');
    assert.equal(hurryRequestId('t-1'), first, 'a retry reuses the SAME id');
    assert.match(first, /^hurry-/);
    assert.notEqual(hurryRequestId('t-2'), first);
});

// --- no chat message, ever (HQ1) ---

test('owner_hurry is hidden from the chat timeline (visible=false)', () => {
    const view = summarizeChatLiveEvent({ type: 'owner_hurry', task_id: 't1', phase: 'applied' });
    assert.equal(view.visible, false);
    assert.equal(view.promote, false);
});

test('ownerHurryProjection is the shared card/detail projection', () => {
    const proj = ownerHurryProjection({ type: 'owner_hurry', task_id: 't1', phase: 'applied' });
    assert.equal(proj.applied, true);
    assert.equal(proj.taskId, 't1');
    assert.match(proj.label, /applied/);
});

test('the hurry path never creates a chat bubble in either surface', () => {
    // Pinned at source: the chat handler consumes owner_hurry BEFORE the
    // timeline summarizer and returns; the shared flow acknowledges via toast
    // only; neither surface routes hurry anywhere near addMessage.
    // The log-event consumer moved into the task-frame router (W3 wave D).
    const frames = readFileSync(new URL('../modules/chat_task_frames.js', import.meta.url), 'utf8');
    assert.match(frames, /eventType === 'owner_hurry'/);
    assert.match(frames, /ownerHurryProjection\(evt\)\.applied/);
    assert.match(menuSrc, /showToast\(/);
    assert.doesNotMatch(menuSrc, /addMessage|send_message|chat\.jsonl/);
});

// --- both surfaces share the ONE control module (owner parity) ---

test('Chat and Activity both wire the shared dropdown', () => {
    for (const source of [cardActions, activity]) {
        assert.match(source, /openTaskControlMenu\(/);
        assert.match(source, /hurryTaskAction\(/);
        assert.match(source, /requestStop\(/);
    }
    // The trigger renders the shared label on both surfaces.
    assert.match(cardActions, /TASK_CONTROL_TRIGGER_LABEL/);
    assert.match(activity, /TASK_CONTROL_TRIGGER_LABEL/);
    assert.equal(typeof TASK_CONTROL_TRIGGER_LABEL, 'string');
});

test('the dropdown replaced the old cancel confirm dialogs (dismiss = continue)', () => {
    // Q2: dismissing the menu continues the run — no separate confirm dialog
    // remains on either cancel path (Activity keeps its schedule-delete one).
    assert.doesNotMatch(chat, /Cancel this run and all its subagents\?/);
    assert.doesNotMatch(cardActions, /Cancel this run and all its subagents\?/);
    assert.doesNotMatch(activity, /Cancel this task and all its subagents\?/);
    assert.match(menuSrc, /Escape/);
});

// --- pending soft stop presentation (Q1) ---

test('taskSoftStopPending distinguishes the soft episode from a hard cancel', () => {
    assert.equal(taskSoftStopPending({
        status: 'running', cancel_state: 'pending', stop_policy: 'finalize_then_cancel',
    }), true);
    assert.equal(taskSoftStopPending({ status: 'running', cancel_state: 'pending' }), false);
    assert.equal(taskSoftStopPending({
        status: 'cancelled', cancel_state: 'pending', stop_policy: 'finalize_then_cancel',
    }), false);
});

test('the chat card re-offers the escalation during a pending soft stop', () => {
    // Q1 pinned at source: after a soft 202 the trigger is re-enabled (the
    // pending menu offers only "Stop now"), while an immediate
    // stop keeps the button disabled until the terminal frame.
    assert.match(cardActions, /record\.cancelPendingPolicy === 'finalize'/);
    assert.match(cardActions, /cancelPending: Boolean\(record\.cancelPendingPolicy\)/);
});
