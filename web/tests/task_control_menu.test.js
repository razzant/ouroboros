// S3 (Q2/HQ1): the shared three-action task stop/hurry control — exact owner
// wording, action gating around a pending cancel, stable hurry request-id
// reuse, and the no-chat-bubble contract pinned at source for both surfaces.

import assert from 'node:assert/strict';
import test from 'node:test';
import { readFileSync } from 'node:fs';

import {
    ACTION_FINALIZE,
    ACTION_HURRY,
    ACTION_PAUSE,
    ACTION_RESUME,
    ACTION_STOP_EARLIER,
    ACTION_STOP_NOW,
    TASK_CONTROL_LABELS,
    TASK_CONTROL_TRIGGER_LABEL,
    heldWorkFacts,
    hurryRequestId,
    isRootTaskRow,
    pauseTaskAction,
    requestHurry,
    requestStop,
    resumeEligible,
    resumeTaskAction,
    stopPolicyFor,
    taskControlActions,
    taskControlBusy,
    taskControlTriggerLabel,
} from '../modules/task_control_menu.js';
import { ownerHurryProjection, summarizeChatLiveEvent, taskSoftStopPending } from '../modules/log_events.js';

const chat = readFileSync(new URL('../modules/chat.js', import.meta.url), 'utf8');
const activity = readFileSync(new URL('../modules/activity.js', import.meta.url), 'utf8');
const menuSrc = readFileSync(new URL('../modules/task_control_menu.js', import.meta.url), 'utf8');
const workSrc = readFileSync(new URL('../modules/work_status.js', import.meta.url), 'utf8');

// --- the frozen owner dropdown (Q2/HQ1) ---

test('the dropdown offers exactly the owner-decided actions, in order', () => {
    // Q2/HQ1 froze Wrap up / Hurry up / Stop now; owner Batch4 (5A) added the
    // whole-tree Pause beside them, before the hard stop.
    assert.deepEqual(taskControlActions(), [ACTION_FINALIZE, ACTION_HURRY, ACTION_PAUSE, ACTION_STOP_NOW]);
    assert.equal(TASK_CONTROL_LABELS[ACTION_FINALIZE], 'Wrap up');
    assert.equal(TASK_CONTROL_LABELS[ACTION_HURRY], 'Hurry up');
    assert.equal(TASK_CONTROL_LABELS[ACTION_PAUSE], 'Pause');
    assert.equal(TASK_CONTROL_LABELS[ACTION_STOP_NOW], 'Stop now');
    // A held Continue's earlier attempt has its own Stop.
    assert.equal(TASK_CONTROL_LABELS[ACTION_STOP_EARLIER], 'Stop earlier attempt');
    // A paused member never offers a second Pause; Resume only where its work's Resume control asks for it.
    assert.deepEqual(taskControlActions({ budgetPaused: true }), [ACTION_STOP_NOW]);
    assert.deepEqual(taskControlActions({ budgetPaused: true, resume: true }), [ACTION_RESUME, ACTION_STOP_NOW]);
    assert.equal(stopPolicyFor(ACTION_PAUSE), '', 'Pause is not a stop');
});

test('Pause retries an unknown answer with the same id, then a new action gets a fresh id', async () => {
    const calls = [];
    const toasts = [];
    const pause = async (id, requestId) => {
        calls.push([id, requestId]);
        if (calls.length === 1) throw new Error('response lost');
        return { ok: true, state: calls.length === 2 ? 'requested' : 'paused' };
    };
    const toast = (text, kind) => toasts.push([text, kind]);
    assert.equal(await pauseTaskAction('root-1', { pause, toast }), false);
    assert.equal(await pauseTaskAction('root-1', { pause, toast }), true);
    await resumeTaskAction('root-1', { resume: async () => ({ ok: true }), toast });
    assert.equal(await pauseTaskAction('root-1', { pause, toast }), true);
    assert.equal(calls[0][1], calls[1][1], 'a retry reuses the SAME request id');
    assert.notEqual(calls[1][1], calls[2][1], 'a new owner Pause after Resume is a NEW action');
    assert.match(calls[0][1], /^pause-/);
    // Owner S1: Pause abandons a model answer still awaited; it never promises sent work finishes.
    assert.match(toasts[1][0], /^Pausing: new work is stopped; running work stops at its last saved point\. Reviews already started finish separately\.$/);
    assert.doesNotMatch(toasts[1][0], /already sent|then the tree is saved/);
    assert.match(toasts[3][0], /^Paused: the whole task tree is saved/);
    assert.equal(await pauseTaskAction('root-2', { pause: async () => { throw new Error('cancel_pending'); }, toast }), false);
    assert.match(toasts.at(-1)[0], /^Pause refused: cancel_pending/);
    // Both surfaces reach it through the shared menu, resolved from the anchor.
    assert.match(menuSrc, /anchor\.dataset\?\.id \|\| anchor\.closest\?\.\('\[data-task-id\]'\)/);
});

test("the owner's Pause press declares the owner through the shared client, the same on its retry", async () => {
    const prior = globalThis.fetch;
    const posts = [];
    globalThis.fetch = async (url, init) => {
        posts.push([url, init.method, JSON.parse(init.body)]);
        const latch = posts.length === 1;
        return { ok: true, status: latch ? 202 : 200, json: async () => ({ ok: true, state: 'requested', latch_pending: latch }) };
    };
    try {
        assert.equal(await pauseTaskAction('owner-root', { toast: () => {}, wait: async () => {} }), true);
    } finally { globalThis.fetch = prior; }
    assert.deepEqual(posts.map(([url, method]) => [url, method]),
        [['/api/tasks/owner-root/pause', 'POST'], ['/api/tasks/owner-root/pause', 'POST']]);
    const [[, , body], [, , retry]] = posts;
    assert.match(body.request_id, /^pause-/);
    assert.deepEqual(body, { request_id: body.request_id, actor: 'owner' }, 'the owner declares itself, with no text');
    assert.deepEqual(retry, body, 'completing the same action repeats the same declared press');
});

test('Resume retires an unconfirmed Pause id, but a refused Resume leaves it retryable', async () => {
    const ids = [];
    const pause = async (_id, requestId) => { ids.push(requestId); throw new Error('response lost'); };
    const toast = () => {};
    await pauseTaskAction('uncertain-pause', { pause, toast });
    await resumeTaskAction('uncertain-pause', { resume: async () => { throw new Error('still pausing'); }, toast });
    await pauseTaskAction('uncertain-pause', { pause, toast });
    assert.equal(ids[1], ids[0]);
    await resumeTaskAction('uncertain-pause', { resume: async () => null, toast });
    await pauseTaskAction('uncertain-pause', { pause, toast });
    assert.equal(ids[2], ids[0], 'an unknown Resume response cannot retire the pending Pause');
    await resumeTaskAction('uncertain-pause', { resume: async () => ({ ok: true }), toast });
    await pauseTaskAction('uncertain-pause', { pause, toast });
    assert.notEqual(ids[3], ids[0]);
});

test('tree Resume keeps helpers for selection and source refusals use the server sentence', async () => {
    const seen = [], toast = (text) => seen.push(text);
    await resumeTaskAction('answered', { resume: async () => ({ ok: true, tree_only: true }), toast });
    assert.equal(seen.pop(), 'Resumed. Ouroboros or you choose which subagents continue.');
    const refusal = Object.assign(new Error('pause_source_unreadable'), {
        body: { message: 'Resume refused: Its saved state could not be restored' },
    });
    await resumeTaskAction('saved', { resume: async () => { throw refusal; }, toast });
    assert.equal(seen.pop(), refusal.body.message);
});

test('a null Pause acknowledgement keeps its id; a released replay never claims Pausing', async () => {
    const ids = [];
    const toasts = [];
    const pause = async (_id, requestId) => {
        ids.push(requestId);
        return ids.length === 1 ? null : { ok: true, duplicate: true, state: 'released' };
    };
    const toast = (...args) => toasts.push(args);
    assert.equal(await pauseTaskAction('late-pause', { pause, toast }), false);
    assert.equal(await pauseTaskAction('late-pause', { pause, toast }), true);
    assert.equal(ids[0], ids[1]);
    assert.match(toasts.at(-1)[0], /already resumed/i);
    assert.equal(toasts.at(-1)[1], 'info');
    assert.equal(await pauseTaskAction('late-pause', { pause, toast }), true);
    assert.notEqual(ids[1], ids[2]);
});

test('a durable Pause with a pending queue latch is never reported refused; the same id completes it', async () => {
    const toasts = [];
    const toast = (...args) => toasts.push(args);
    const wait = async () => {};
    const pending = { ok: true, state: 'requested', latch_pending: true, duplicate: false };
    const ids = [];
    const completes = async (_id, requestId) => {
        ids.push(requestId);
        if (ids.length === 2) throw new Error('response lost');
        return ids.length < 3 ? pending : { ok: true, state: 'requested', duplicate: true };
    };
    assert.equal(await pauseTaskAction('latch-1', { pause: completes, toast, wait }), true);
    assert.equal(ids.length, 3);
    assert.equal(new Set(ids).size, 1, 'every completion retry reuses the SAME request id');
    assert.match(toasts.at(-1)[0], /^Pausing: new work is stopped; running work stops at its last saved point/);
    assert.equal(toasts.length, 1);
    // Never confirmed within the bound: still accepted and truthful; the id is kept.
    const stuck = [];
    const always = async (_id, requestId) => { stuck.push(requestId); return pending; };
    assert.equal(await pauseTaskAction('latch-2', { pause: always, toast, wait }), true);
    assert.match(toasts.at(-1)[0], /^Pausing: new work is stopped\. The queue part .* not confirmed yet/);
    assert.doesNotMatch(toasts.at(-1)[0], /refused/i);
    await pauseTaskAction('latch-2', { pause: always, toast, wait });
    assert.equal(new Set(stuck).size, 1, 'the owner\'s next press finishes the SAME action');
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
    assert.match(chat, /eventType === 'owner_hurry'/);
    assert.match(chat, /ownerHurryProjection\(evt\)\.applied/);
    assert.match(menuSrc, /showToast\(/);
    assert.doesNotMatch(menuSrc, /addMessage|send_message|chat\.jsonl/);
});

// --- both surfaces share the ONE control module (owner parity) ---

test('Chat and Activity both wire the shared dropdown', () => {
    // Chat cards and the room's Shade open it through work_status.js on their work's controls (a card
    // no work lists keeps the record-aware adapter). Activity still calls the same lower-level dropdown.
    assert.match(chat, /openCardControls\(/);
    assert.match(workSrc, /openTaskControlMenu\(/);
    assert.match(workSrc, /openTaskControls\(/);
    // Exact adapter effects are exercised in test_room_work_review_browser.py;
    // finding its own function names inside the adapter would prove nothing.
    assert.match(activity, /openTaskControlMenu\(/);
    assert.match(activity, /hurryTaskAction\(/);
    assert.match(activity, /requestStop\(/);
    // The trigger renders the shared label on both surfaces; Activity names a single offered action itself.
    assert.match(chat, /TASK_CONTROL_TRIGGER_LABEL/);
    assert.match(activity, /taskControlTriggerLabel\(/);
    assert.equal(typeof TASK_CONTROL_TRIGGER_LABEL, 'string');
});

test('a compensated feed scroll keeps an anchored menu; moving or losing its trigger still dismisses it', async (t) => {
    const { closeTaskControlMenu, openTaskControlMenu } = await import('../modules/task_control_menu.js');
    const { installDomDouble } = await import('./dom_double.js');
    const env = installDomDouble();
    const prior = globalThis.IntersectionObserver;
    let observed;
    globalThis.IntersectionObserver = class {
        constructor(callback) { observed = callback; }
        observe() {} disconnect() {} takeRecords() { return []; }
    };
    t.after(() => { closeTaskControlMenu(); globalThis.IntersectionObserver = prior; env.restore(); });
    const feed = env.el('div'), other = env.el('div'), anchor = env.el('button');
    feed.nodeType = other.nodeType = 1;
    env.doc.body.append(feed, other);
    feed.append(anchor);
    const open = () => openTaskControlMenu(anchor, { actions: [ACTION_HURRY, ACTION_STOP_NOW] });
    const scroll = (target) => {
        for (const listener of [...env.doc.captures.get('scroll') || []]) listener({ type: 'scroll', target });
    };
    let menu = open();
    feed.scrollTop = 26;  // a late card fact grew above the trigger; the feed restored its place
    scroll(feed);
    assert.equal(menu.isConnected, true, 'the same on-screen trigger still owns the open choice');
    scroll(menu);
    assert.equal(menu.isConnected, true, 'a constrained menu can scroll inside itself');
    anchor.rect = { ...anchor.rect, top: 10, bottom: 40 };
    scroll(other);
    assert.equal(menu.isConnected, true, 'an unrelated scroller does not own dismissal');
    scroll(feed);
    assert.equal(menu.isConnected, false, 'a scroll that moved the trigger dismisses');
    assert.equal(env.doc.activeElement, anchor);
    menu = open();
    anchor.rect = { ...anchor.rect, left: 12, right: 112 };
    scroll(feed);
    assert.equal(menu.isConnected, false, 'horizontal movement dismisses too');
    menu = open();
    scroll(env.doc);
    assert.equal(menu.isConnected, false, 'the document keeps its page-level dismissal contract');
    menu = open();
    env.doc.defaultView.emit('resize');
    assert.equal(menu.isConnected, false, 'a changed viewport still dismisses');
    menu = open();
    observed([{ target: anchor, isIntersecting: false }]);
    assert.equal(menu.isConnected, false, 'an off-screen trigger cannot own a menu');
    menu = open();
    anchor.remove();
    observed([{ target: anchor, isIntersecting: true }]);
    assert.equal(menu.isConnected, false, 'a detached trigger cannot keep the old menu');
    assert.equal(env.doc.captures.get('scroll').size, 0, 'dismissal releases its listener');
});

test('a single offered action is the trigger itself: it names the action and a stop asks first', async (t) => {
    const { closeTaskControlMenu, openTaskControlMenu } = await import('../modules/task_control_menu.js');
    const { installDomDouble } = await import('./dom_double.js');
    const env = installDomDouble();
    const prior = globalThis.IntersectionObserver;
    globalThis.IntersectionObserver = class { observe() {} disconnect() {} takeRecords() { return []; } };
    t.after(() => { closeTaskControlMenu(); globalThis.IntersectionObserver = prior; env.restore(); });
    assert.equal(taskControlTriggerLabel([ACTION_STOP_NOW]), TASK_CONTROL_LABELS[ACTION_STOP_NOW]);
    assert.equal(taskControlTriggerLabel([ACTION_RESUME]), 'Resume');
    for (const many of [[ACTION_RESUME, ACTION_STOP_NOW], [], undefined]) {
        assert.equal(taskControlTriggerLabel(many), TASK_CONTROL_TRIGGER_LABEL, 'a menu keeps its ellipsis');
    }
    const anchor = env.doc.createElement('button');
    anchor.dataset.taskControlDirect = '1';
    env.doc.body.append(anchor);
    const settle = () => new Promise((resolve) => setImmediate(resolve));
    const done = [];
    const asked = [];
    for (const [action, answer, ran] of [[ACTION_STOP_NOW, true, true], [ACTION_STOP_EARLIER, false, false],
        [ACTION_RESUME, null, true]]) {
        done.length = 0;
        asked.length = 0;
        const menu = openTaskControlMenu(anchor, { actions: [action], onAction: (a) => done.push(a),
            askConfirm: async (dialog) => { asked.push(dialog); return answer; } });
        await settle();
        assert.equal(menu, null, 'no menu of one item');
        assert.equal(env.doc.body.querySelector('[role="menu"]'), null);
        assert.equal(anchor.getAttribute('aria-haspopup'), null, 'the trigger is a plain button');
        assert.deepEqual(done, ran ? [action] : [], action);
        if (answer === null) assert.equal(asked.length, 0, 'only a stop asks');
        else assert.deepEqual([asked.length, asked[0].confirmLabel, asked[0].danger], [1, TASK_CONTROL_LABELS[action], true]);
    }
    done.length = 0;
    openTaskControlMenu(anchor, { actions: [ACTION_STOP_NOW], busy: true, onAction: (a) => done.push(a), askConfirm: async () => true });
    await settle();
    assert.deepEqual(done, [], 'a busy task takes no second press');
    // Two actions are a real choice: the menu opens and nothing runs until an item is chosen.
    const menu = openTaskControlMenu(anchor, { actions: [ACTION_HURRY, ACTION_STOP_NOW], onAction: (a) => done.push(a),
        askConfirm: async () => { throw new Error('a menu item never asks again'); } });
    assert.ok(menu);
    assert.equal(anchor.getAttribute('aria-expanded'), 'true');
    menu.querySelector(`[data-task-control="${ACTION_STOP_NOW}"]`).click();
    await settle();
    assert.deepEqual(done, [ACTION_STOP_NOW]);
    closeTaskControlMenu();
    // A trigger that always says Stop… (a chat card's) keeps its menu even for one action: its label promised one.
    const plain = env.doc.createElement('button');
    env.doc.body.append(plain);
    assert.ok(openTaskControlMenu(plain, { actions: [ACTION_STOP_NOW], onAction: (a) => done.push(a),
        askConfirm: async () => { throw new Error('no dialog on a menu trigger'); } }));
    assert.deepEqual(done, [ACTION_STOP_NOW], 'nothing runs before an item is chosen');
});

test('the Stop confirmation of an accepted unbound start says the request, never an immediate end (#3428)', async (t) => {
    const { closeTaskControlMenu, openTaskControlMenu } = await import('../modules/task_control_menu.js');
    const { installDomDouble } = await import('./dom_double.js');
    const env = installDomDouble();
    const prior = globalThis.IntersectionObserver;
    globalThis.IntersectionObserver = class { observe() {} disconnect() {} takeRecords() { return []; } };
    t.after(() => { closeTaskControlMenu(); globalThis.IntersectionObserver = prior; env.restore(); });
    const settle = () => new Promise((resolve) => setImmediate(resolve));
    const anchor = env.doc.createElement('button');
    anchor.dataset.taskControlDirect = '1';
    env.doc.body.append(anchor);
    const asked = [];
    const askConfirm = async (dialog) => { asked.push(dialog); return false; };
    const open = () => openTaskControlMenu(anchor, { actions: [ACTION_STOP_NOW],
        onAction: () => { throw new Error('a declined confirm never runs'); }, askConfirm });

    open();
    await settle();
    assert.equal(asked.length, 1);
    assert.equal(asked[0].body, 'It stops at once, with the subagents it started.', 'a bound run keeps its words');

    // The same trigger on a work the server marks with an accepted unbound start (activity.js renders the fact):
    // no engine run answers it yet, so the confirmation says the recorded request, not an immediate end
    // (tests/test_settled_tree_run_stop.py: HTTP 202 pending, zero engine cancels while the start is unbound).
    anchor.dataset.unboundStart = '1';
    open();
    await settle();
    assert.equal(asked.length, 2);
    assert.equal(asked[1].title, 'Stop this work now?');
    assert.equal(asked[1].danger, true);
    assert.equal(asked[1].confirmLabel, TASK_CONTROL_LABELS[ACTION_STOP_NOW]);
    assert.equal(asked[1].body, 'Stop is requested at once. A subagent whose start was never confirmed may keep '
        + 'working until it is identified and its stop is confirmed.');
});

test('the dropdown replaced the old cancel confirm dialogs (dismiss = continue)', () => {
    // Q2: dismissing the menu continues the run — no separate confirm dialog
    // remains on either cancel path (Activity keeps its schedule-delete one).
    assert.doesNotMatch(chat, /Cancel this run and all its subagents\?/);
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
    assert.match(chat, /record\.cancelPendingPolicy === 'finalize'/);
    assert.match(workSrc, /const cancelPending = Boolean\(record\?\.cancelPendingPolicy\);/);
});

// --- budget-paused resume offer (#322) ---

test('a budget-paused member offers Resume (where its work asks for it) and the stop escalation only', () => {
    assert.deepEqual(taskControlActions({ budgetPaused: true, resume: true }), [ACTION_RESUME, ACTION_STOP_NOW]);
    assert.deepEqual(taskControlActions({ budgetPaused: true }), [ACTION_STOP_NOW],
        'a paused phase alone offers no Resume (a held Continue is never the owner\'s to resume)');
    // A pending cancel outranks the pause: only the hard escalation remains.
    assert.deepEqual(
        taskControlActions({ budgetPaused: true, cancelPending: true, resume: true }),
        [ACTION_STOP_NOW],
    );
    assert.equal(TASK_CONTROL_LABELS[ACTION_RESUME], 'Resume');
});

// --- Batch4 F14: the whole-tree Pause is a ROOT's action ---

test('a child row never offers the whole-tree Pause; its other actions stay', () => {
    assert.deepEqual(taskControlActions({ wholeTree: false }), [ACTION_FINALIZE, ACTION_HURRY, ACTION_STOP_NOW]);
    assert.deepEqual(taskControlActions({ wholeTree: false, budgetPaused: true }), [ACTION_STOP_NOW]);
    assert.deepEqual(taskControlActions({ wholeTree: false, budgetPaused: true, resume: true }), [ACTION_RESUME, ACTION_STOP_NOW]);
    assert.deepEqual(taskControlActions({ wholeTree: false, cancelPending: true }), [ACTION_STOP_NOW]);
    assert.deepEqual(taskControlActions({ wholeTree: true }), [ACTION_FINALIZE, ACTION_HURRY, ACTION_PAUSE, ACTION_STOP_NOW]);
});

test('isRootTaskRow mirrors the server lineage rule (task_results.resolve_task_lineage)', () => {
    assert.equal(isRootTaskRow({ root_task_id: 'r1' }, 'r1'), true);
    assert.equal(isRootTaskRow({}, 'r1'), true, 'no lineage fields: its own root');
    assert.equal(isRootTaskRow({ root_task_id: 'r1', parent_task_id: 'r1', delegation_role: 'subagent' }, 'c1'), false);
    assert.equal(isRootTaskRow({ parent_task_id: 'r1' }, 'c1'), false, 'any parent makes it a child');
    assert.equal(isRootTaskRow({ root_task_id: 'c1', delegation_role: 'subagent' }, 'c1'), false);
    assert.equal(isRootTaskRow({ metadata: { parent_task_id: 'r1' } }, 'c1'), false, 'metadata carries lineage');
    assert.equal(isRootTaskRow({ parent_task_id: '', metadata: { parent_task_id: 'stale' } }, 'r1'), true,
        'an explicit empty parent overrides stale metadata, like the server');
    // A top-level hard-timeout retry: a root attempt only when both host markers agree.
    const retry = { root_task_id: 'orig', delegation_role: 'root', original_task_id: 'orig', timeout_retry_from: 'orig' };
    assert.equal(isRootTaskRow(retry, 'retry-1'), true);
    assert.equal(isRootTaskRow({ ...retry, timeout_retry_from: 'other' }, 'retry-1'), false);
    assert.equal(isRootTaskRow({ root_task_id: 'orig' }, 'retry-1'), false);
});

test('Activity marks root rows and hides Pause for a child; Chat cards stay root-only', () => {
    // A compact retry row can omit lineage; its own root-only census identity still counts.
    assert.match(activity, /phases\.has\(ownId\) \|\| isRootTaskRow\(t, ownId\)/);
    assert.match(activity, /wholeTree: btn\.dataset\.root === '1'/);
    // activity_work_truth.test.js checks the root and target on rendered census and tail-only buttons.
    // A chat card offers the menu only for a root (never a subagent card).
    assert.match(chat, /cancelRunEligibility\(\{\s*groupId: record\.groupId, isSubagent: record\.isSubagent/);
});

test('Stop producer reuses uncertain action identity and distinguishes acknowledged later Stop', async () => {
    const { requestStop } = await import('../modules/task_control_menu.js');
    const oldFetch = globalThis.fetch;
    const calls = [];
    globalThis.fetch = async (url, init) => {
        calls.push({ url, ...JSON.parse(init.body) });
        if (calls.length === 1) throw new Error('response lost after acceptance');
        return { ok: true, status: 202, json: async () => ({ ok: true, cancel_state: 'pending' }) };
    };
    try {
        await assert.rejects(requestStop('retry-stop', ACTION_FINALIZE), /response lost/);
        await requestStop('retry-stop', ACTION_STOP_NOW);
        await requestStop('retry-stop', ACTION_FINALIZE);
        await requestStop('retry-stop', ACTION_FINALIZE);
        assert.equal(calls.length, 4);
        assert.match(calls[0].stop_action_id, /^stop-/);
        assert.equal(calls[0].stop_action_id, calls[2].stop_action_id);
        assert.notEqual(calls[0].stop_action_id, calls[1].stop_action_id);
        assert.notEqual(calls[0].stop_action_id, calls[3].stop_action_id);
        assert.equal(calls[0].stop_policy, 'finalize_then_cancel');
        assert.equal(calls[1].stop_policy, undefined); // immediate legacy encoding
        assert.ok(calls.every(call => call.cascade === true));
    } finally { globalThis.fetch = oldFetch; }
});

// --- #1721: a held Continue names what the census observed ---

const UNBOUND = { kind: 'delegated_run', task_id: 'pred-1', run_id: '', invocation_id: 'inv-1', review_owned: false,
    state: 'stop_unknown', stop_outcome: 'pending_invocation_unbound', source: 'delegated_subagent' };
const RUNNING = { kind: 'delegated_run', task_id: 'pred-1', run_id: 'run-leaf', invocation_id: '', review_owned: false,
    state: 'running', stop_outcome: '', source: 'delegated_subagent' };

test('held-work facts say what is known, and an unknown start is never called running', () => {
    assert.equal(heldWorkFacts([UNBOUND]), 'a delegated start has an unknown outcome (no run was recorded)');
    assert.equal(heldWorkFacts([RUNNING]), 'delegated run run-leaf is running');
    assert.equal(heldWorkFacts([{ kind: 'running_member', task_id: 'c-1' }, { kind: 'custody_unreadable', task_id: 'x' },
        { kind: 'attempt_custody_unreadable' }]), 'task c-1 is still running; its custody record could not be read');
    assert.equal(heldWorkFacts([{ kind: 'some_future_kind' }]), 'some_future_kind', 'a kind with no sentence stays raw');
    assert.equal(heldWorkFacts([1, 2, 3, 4].map((n) => ({ kind: 'running_member', task_id: `c-${n}` }))),
        'task c-1 is still running; task c-2 is still running; task c-3 is still running; and 1 more');
    // The server sends the whole census: a model request listed after 21 running runs is counted.
    const census = [...Array.from({ length: 21 }, (_, n) => ({ kind: 'delegated_run', run_id: `run-${n}`, state: 'running' })),
        { kind: 'model_handoff' }];
    assert.equal(heldWorkFacts(census),
        'delegated run run-0 is running; delegated run run-1 is running; delegated run run-2 is running; and 19 more');
    assert.ok(heldWorkFacts(census, 22).endsWith('; a model request has no recorded result'));
    assert.equal(heldWorkFacts(null), '');
});

test('a held tool call is worded by what is known, never as a call that has not returned', () => {
    const base = { kind: 'tool_handoff', task_id: '96ac2f7c981b4f54', tool: 'run_command' };
    assert.equal(heldWorkFacts([{ ...base, reason: 'local_completion_unknown' }]),
        'it is unknown whether the run_command call of task 96ac2f7c981b4f54 finished: its worker cannot be proven ended');
    assert.equal(heldWorkFacts([{ ...base, reason: 'local_call_live' }]),
        'the run_command call of task 96ac2f7c981b4f54 is in a live worker: still running, or its finished result is not recorded yet');
    assert.equal(heldWorkFacts([{ ...base, tool: 'mcp_gitlab__get_commit', reason: 'owner_ended_effect_unknown',
        effects: ['MCP request'] }]),
    'the mcp_gitlab__get_commit call of task 96ac2f7c981b4f54 ended with its worker; the outcome of its MCP request is unknown');
    assert.equal(heldWorkFacts([{ ...base, reason: 'returned_effect_unconfirmed', effects: [] }]),
        'the run_command call of task 96ac2f7c981b4f54 returned; its outside effect is not confirmed');
    assert.equal(heldWorkFacts([{ ...base }]), 'a run_command call of task 96ac2f7c981b4f54 is held');
    assert.ok(!menuSrc.includes('has not returned'), 'the false sentence is gone');
});

test('a held merge request is worded by what its receipt shows', () => {
    const base = { kind: 'merge_operation', task_id: 't-1', receipt_id: 'r-1' };
    assert.equal(heldWorkFacts([{ ...base, reason: 'merge_effect_unknown' }]),
        'the outcome of a merge request of task t-1 is unknown');
    assert.equal(heldWorkFacts([{ ...base, reason: 'merge_queued' }]), 'a merge of task t-1 is queued on GitHub');
    assert.equal(heldWorkFacts([{ ...base, reason: 'merge_call_unreturned' }]), 'a merge of task t-1 is not settled');
    assert.equal(heldWorkFacts([base]), 'a merge of task t-1 is not settled');
});

test('a refused Resume of a held Continue toasts its facts, with wait-or-stop only for running work', async () => {
    const toasts = [];
    const toast = (text, kind) => toasts.push([text, kind]);
    const refuse = (body) => async () => { throw Object.assign(new Error(body.error), { status: 409, body }); };
    await resumeTaskAction('succ-1', { resume: refuse({ error: 'predecessor_writers_unsettled', blockers: [UNBOUND] }), toast });
    await resumeTaskAction('succ-2', { resume: refuse({ error: 'predecessor_writers_unsettled', blockers: [RUNNING],
        action: 'wait_or_stop_the_previous_work' }), toast });
    await resumeTaskAction('succ-3', { resume: refuse({ error: 'replay_unsafe' }), toast });
    assert.deepEqual(toasts, [
        ['Resume refused: a delegated start has an unknown outcome (no run was recorded).', 'error'],
        ['Resume refused: delegated run run-leaf is running. Wait for that work or stop it.', 'error'],
        ['Resume refused: replay_unsafe', 'error'],
    ]);
});

test('a Resume over earlier work that still runs is accepted into its wait and says what it waits for', async () => {
    const toasts = [];
    await resumeTaskAction('succ-w', { resume: async () => ({ ok: true, task_id: 'succ-w', held: true, blockers: [RUNNING] }),
        toast: (text, kind) => toasts.push([text, kind]) });
    assert.deepEqual(toasts, [["Resume accepted; it starts by itself when the earlier attempt's work finishes: "
        + 'delegated run run-leaf is running.', 'info']]);
});

test('a Resume held only by calls of unknown outcome asks once, then resumes naming them', async () => {
    const held = { kind: 'tool_handoff', task_id: '96ac2f7c981b4f54', tool: 'run_command',
        reason: 'local_completion_unknown', operation_id: 'op-1' };
    const toasts = [];
    const calls = [];
    const asked = [];
    const toast = (text, kind) => toasts.push([text, kind]);
    const resume = async (id, body) => {
        calls.push(body ?? null);
        if (!body) {
            throw Object.assign(new Error('owner_pause_effects_unsettled'), { status: 409,
                body: { error: 'owner_pause_effects_unsettled', blockers: [held], acceptable_unknown: ['op-1'] } });
        }
        return { ok: true };
    };
    await resumeTaskAction('root-u', { resume, toast, askOwner: async (dialog) => { asked.push(dialog); return true; } });
    assert.deepEqual(calls, [null, { accept_unknown: ['op-1'], reason: 'The owner resumed without these results.' }]);
    assert.equal(asked.length, 1);
    assert.match(asked[0].body, /^It is unknown whether the run_command call of task 96ac2f7c981b4f54 finished/);
    assert.match(asked[0].body, /this call is never run again/);
    assert.deepEqual(toasts, [['Resuming: the task returns to the queue.', 'info']]);

    calls.length = 0;
    toasts.length = 0;
    await resumeTaskAction('root-u', { resume, toast, askOwner: async () => false });
    assert.deepEqual(calls, [null], 'declining keeps the task paused');
    assert.deepEqual(toasts, []);

    const live = { ...held, operation_id: 'op-2', reason: 'local_call_live' };
    await resumeTaskAction('root-l', {
        resume: async () => { throw Object.assign(new Error('x'), { status: 409,
            body: { error: 'owner_pause_effects_unsettled', blockers: [held, live] } }); },
        toast, askOwner: async () => { throw new Error('a live call is never offered'); } });
    assert.equal(toasts.length, 1);
    assert.match(toasts[0][0], /^Resume refused: .*is in a live worker/);
});

test('the same confirmation exposes every addressed task and invocation before accepting running and spending risk', async () => {
    const { renderConfirmDialogDetails } = await import('../modules/confirm_dialog.js');
    const blockers = Array.from({ length: 7 }, (_, i) => ({
        kind: 'delegated_run', task_id: `helper-${i}`, invocation_id: `lost-${i}`, run_id: '',
        state: 'stop_unknown', stop_outcome: 'pending_invocation_unbound',
        choice_id: `delegate_start:tree/helper-${i}/lost-${i}/tree/fence-id/park-id`,
    }));
    blockers.push({ kind: 'tool_handoff', task_id: 'tree', reason: 'local_completion_unknown', operation_id: 'op-8' });
    const ids = blockers.map(b => b.choice_id || b.operation_id);
    const calls = [];
    const resume = async (task, body) => {
        calls.push(body);
        if (!body) throw Object.assign(new Error('held'), { status: 409, body: {
            error: 'owner_pause_effects_unsettled', blockers, acceptable_unknown: ids,
        } });
        return { ok: true };
    };
    for (const confirmed of [false, true]) {
        calls.length = 0;
        await resumeTaskAction('tree', { resume, toast: () => {}, askOwner: async (dialog) => {
            assert.match(dialog.body, /still be running, changing files or spending money/);
            assert.match(renderConfirmDialogDetails(dialog.details), /<details[^>]* open>/);
            assert.doesNotMatch(renderConfirmDialogDetails({ rows: dialog.details.rows }), /<details[^>]* open>/);
            assert.equal(dialog.details.rows.length, ids.length);
            dialog.details.rows.forEach(({ label, value }, i) => {
                assert.ok(label.includes(blockers[i].task_id));
                if (blockers[i].kind === 'delegated_run') {
                    assert.equal(value, `Start request ${blockers[i].invocation_id}`);
                } else assert.ok(value.includes(ids[i]));
            });
            return confirmed;
        } });
        assert.equal(calls.length, confirmed ? 2 : 1);
        if (confirmed) {
            assert.deepEqual(calls[1].accept_unknown, ids);
            assert.match(calls[1].reason, /spending/);
        }
    }
});

// The one registry of a task's pending controls (docs/DESIGN.md "Work status": a press is pending until its answer and its
// reconciling read are in). A Resume door reads `resumeEligible`, so a census older than the press is no leave to send
// again; a menu reads `taskControlBusy` (a request awaiting its answer), which a Resume only awaiting its read is not.
test('a Resume keeps its task pending until the read that reconciles it has settled, whatever the answer was', async () => {
    const unknownOutcome = Object.assign(new Error('x'), { status: 409, body: { error: 'owner_pause_effects_unsettled',
        blockers: [{ kind: 'tool_handoff', task_id: 'held', tool: 'run_command', reason: 'local_completion_unknown' }],
        acceptable_unknown: ['op-1'] } });
    const answers = [
        ['accepted', async () => ({ ok: true })],
        ['refused', async () => { throw Object.assign(new Error('replay_unsafe'), { status: 409, body: { message: 'Not safe to replay.' } }); }],
        ['a network failure', async () => { throw new TypeError('Failed to fetch'); }],
        ['an unknown acknowledgement', async () => null],
        ['a declined Resume anyway', async () => { throw unknownOutcome; }],
    ];
    for (const [name, answer] of answers) {
        const id = `pending-${name.replaceAll(' ', '-')}`;
        const sent = [];
        const resume = async (task) => { sent.push(task); return answer(); };
        let reread;
        const settle = () => new Promise((resolve) => { reread = resolve; });
        const toast = () => {};
        const first = resumeTaskAction(id, { resume, toast, askOwner: async () => false, settle });
        for (let waited = 0; !reread && waited < 50; waited += 1) await new Promise((resolve) => setTimeout(resolve, 0));
        assert.ok(reread, `${name}: its reconciling read is out`);
        assert.equal(resumeEligible(id, () => true), false, `${name}: pending while that read is out`);
        assert.equal(taskControlBusy(id), false, `${name}: but no request is out: a menu stays enabled`);
        await resumeTaskAction(id, { resume, toast, settle: () => { throw new Error('a pending task reads nothing for a second press'); } });
        assert.deepEqual(sent, [id], `${name}: a second press of the pending task sends nothing`);
        reread();
        await first;
        assert.equal(resumeEligible(id, () => true), true, `${name}: the pending ends with its read`);
        assert.equal(resumeEligible(id, () => false), false, `${name}: and still follows the caller's own authority`);
        assert.equal(await resumeTaskAction(id, { resume: async () => ({ ok: true }), toast }), undefined);
    }
});

test('a Resume whose reconciling read fails still releases its task, and the failure reaches the door', async () => {
    const sent = [];
    const resume = async (id) => { sent.push(id); return { ok: true }; };
    await assert.rejects(resumeTaskAction('read-fails', { resume, toast: () => {}, settle: async () => { throw new Error('read lost'); } }),
        /read lost/);
    assert.equal(resumeEligible('read-fails', () => true), true, 'a read that ended badly is no lock');
    await resumeTaskAction('read-fails', { resume, toast: () => {} });
    assert.deepEqual(sent, ['read-fails', 'read-fails'], 'the owner\'s retry sends');
});

// Every pending control is a lease its own operation gives back, and only that one. A Stop or a Hurry that ends (a
// refusal too) must not free the wait of a Resume for its reconciling read: a census read before the Resume's answer
// would then unlock a second Resume. Stop stays available throughout; nothing forbids it to keep the wait.
test("a Stop, a Hurry or a second Stop that ends — refused too — frees only its own lease, never a Resume's wait", async (t) => {
    const prior = globalThis.fetch;
    t.after(() => { globalThis.fetch = prior; });
    const posts = [];
    globalThis.fetch = (url, init) => new Promise((resolve) => { posts.push({ url, method: init.method, resolve }); });
    const reply = (index, ok = true) => posts[index].resolve({ ok, status: ok ? 200 : 409,
        json: async () => (ok ? { ok: true } : { error: 'refused by the server' }) });
    const until = async (ready) => {
        for (let waited = 0; !ready() && waited < 100; waited += 1) await new Promise((resolve) => setImmediate(resolve));
        assert.ok(ready());
    };
    const id = 'lease-task';
    let reread;
    const waiting = resumeTaskAction(id, { resume: async () => ({ ok: true }), toast: () => {},
        settle: () => new Promise((resolve) => { reread = resolve; }) });
    await until(() => reread);
    const state = () => [resumeEligible(id, () => true), taskControlBusy(id)];
    assert.deepEqual(state(), [false, false], 'a Resume awaiting its read holds the task; no request is out');

    const refused = requestStop(id, 'stop_now');
    await until(() => posts.length === 1);
    assert.deepEqual(state(), [false, true], 'a Stop awaiting its answer is a request out');
    reply(0, false);
    await assert.rejects(refused, /refused by the server/);
    assert.deepEqual(state(), [false, false], 'a refused Stop frees nothing of the Resume');

    const accepted = requestStop(id, 'stop_now');
    await until(() => posts.length === 2);
    reply(1);
    assert.equal((await accepted).ok, true);
    assert.deepEqual(state(), [false, false], 'an accepted Stop frees nothing of the Resume');

    const hurried = requestHurry(id);
    await until(() => posts.length === 3);
    assert.match(posts[2].url, /\/hurry$/);
    reply(2);
    await hurried;
    assert.deepEqual(state(), [false, false], 'a Hurry frees nothing of the Resume');
    const hurryRefused = requestHurry(id);
    await until(() => posts.length === 4);
    reply(3, false);
    await assert.rejects(hurryRefused);
    assert.deepEqual(state(), [false, false], 'a refused Hurry neither');

    // Two Stops at once (a Wrap up, then its hard escalation): the first to end leaves the second's request out.
    const soft = requestStop(id, 'finalize');
    const hard = requestStop(id, 'stop_now');
    await until(() => posts.length === 6);
    reply(4);
    await soft;
    assert.deepEqual(state(), [false, true], 'the first Stop ended; the second still awaits its answer');
    reply(5);
    await hard;
    assert.deepEqual(state(), [false, false], 'both ended: only the Resume still holds the task');

    const sentAgain = [];
    await resumeTaskAction(id, { resume: async (task) => { sentAgain.push(task); return { ok: true }; }, toast: () => {} });
    assert.deepEqual(sentAgain, [], 'throughout, a second Resume of the pending task sent nothing');
    reread();
    await waiting;
    assert.deepEqual(state(), [true, false], 'the Resume\'s read settled: the task is free again');
});

test("a work menu keeps Stop available while its Resume awaits its read: neither that Stop nor the read's end frees the wait early", async (t) => {
    const { closeTaskControlMenu } = await import('../modules/task_control_menu.js');
    const { openWorkControls } = await import('../modules/work_status.js');
    const { installDomDouble } = await import('./dom_double.js');
    const env = installDomDouble();
    const prior = { observer: globalThis.IntersectionObserver, fetch: globalThis.fetch };
    globalThis.IntersectionObserver = class { observe() {} disconnect() {} takeRecords() { return []; } };
    const posts = [];
    globalThis.fetch = (url, init) => new Promise((resolve) => { posts.push({ url, method: init.method, resolve }); });
    t.after(() => { closeTaskControlMenu(); Object.assign(globalThis, { IntersectionObserver: prior.observer, fetch: prior.fetch }); env.restore(); });
    const reply = (index, ok = true) => posts[index].resolve({ ok, status: ok ? 200 : 409, json: async () => (ok ? { ok: true } : { error: 'refused' }) });
    const until = async (ready) => {
        for (let waited = 0; !ready() && waited < 100; waited += 1) await new Promise((resolve) => setImmediate(resolve));
        assert.ok(ready());
    };
    const anchor = env.doc.createElement('button');
    env.doc.body.append(anchor);
    const id = 'held-task';
    const work = { episode: { task_id: id }, controls: [{ action: 'resume', target: id },
        { action: 'stop', target: id, scope: 'cascade' }] };
    const rereads = [];
    let end;
    const afterAction = (action) => {
        rereads.push(action);
        return action === 'resume' ? new Promise((resolve) => { end = resolve; }) : undefined;
    };
    const items = (menu) => menu.querySelectorAll('[data-task-control]').map((item) => [item.dataset.taskControl, item.disabled]);
    const free = () => resumeEligible(id, () => true);

    const first = openWorkControls(anchor, work, { withResume: true, afterAction });
    assert.deepEqual(items(first), [['resume', false], ['stop_now', false]]);
    first.querySelector('[data-task-control="resume"]').click();
    await until(() => posts.length === 1);
    assert.match(posts[0].url, /\/resume$/);
    reply(0);
    await until(() => end);
    assert.deepEqual(rereads, ['resume'], "its read is the Resume's own, asked once");
    assert.deepEqual([free(), taskControlBusy(id)], [false, false], 'the answer is in: the Resume awaits its read, no request is out');

    // Reopened meanwhile: Stop is there and enabled, Resume is not offered again.
    const again = openWorkControls(anchor, work, { withResume: true, afterAction });
    assert.deepEqual(items(again), [['stop_now', false]]);
    again.querySelector('[data-task-control="stop_now"]').click();
    await until(() => posts.length === 2);
    assert.match(posts[1].url, /\/cancel$/);
    assert.deepEqual([free(), taskControlBusy(id)], [false, true], 'the Stop awaits its answer');
    reply(1, false);  // refused
    await until(() => !taskControlBusy(id));
    assert.equal(free(), false, 'a refused Stop frees nothing of the Resume');
    assert.equal(posts.filter((post) => post.url.endsWith('/resume')).length, 1, 'one Resume POST so far');

    end();  // the Resume's read settles
    await until(free);
    const later = openWorkControls(anchor, work, { withResume: true, afterAction });
    assert.deepEqual(items(later), [['resume', false], ['stop_now', false]], 'the task is free: Resume is offered again');
    assert.equal(rereads.filter((action) => action === 'resume').length, 1, 'one read for the one Resume');
});

// Activity's own dispatch (activity.js `afterTaskControl`): a Resume's settle is its bounded state read alone, and the
// page's schedules/services reads — started beside the confirmation — never enter it. The menu awaits exactly what
// `afterAction('resume')` returns, so this pins the seam that keeps an unrelated read from prolonging the pending.
test("Activity's Resume confirmation is its own bounded state read: an independent schedules read never prolongs it", async (t) => {
    const { closeTaskControlMenu } = await import('../modules/task_control_menu.js');
    const { openWorkControls } = await import('../modules/work_status.js');
    const { installDomDouble } = await import('./dom_double.js');
    const env = installDomDouble();
    const prior = { observer: globalThis.IntersectionObserver, fetch: globalThis.fetch };
    globalThis.IntersectionObserver = class { observe() {} disconnect() {} takeRecords() { return []; } };
    const posts = [];
    globalThis.fetch = (url, init) => new Promise((resolve) => { posts.push({ url, method: init.method, resolve }); });
    t.after(() => { closeTaskControlMenu(); Object.assign(globalThis, { IntersectionObserver: prior.observer, fetch: prior.fetch }); env.restore(); });
    const until = async (ready) => {
        for (let waited = 0; !ready() && waited < 100; waited += 1) await new Promise((resolve) => setImmediate(resolve));
        assert.ok(ready());
    };
    const anchor = env.doc.createElement('button');
    env.doc.body.append(anchor);
    const id = 'act-task';
    const work = { episode: { task_id: id }, controls: [{ action: 'resume', target: id },
        { action: 'stop', target: id, scope: 'cascade' }] };
    const items = (menu) => menu.querySelectorAll('[data-task-control]').map((item) => [item.dataset.taskControl, item.disabled]);
    const free = () => resumeEligible(id, () => true);

    // A Resume's settle is the bounded state read alone; an independent schedules read stays out the whole time.
    // afterAction runs only when something is pressed, so the healthy menu below can carry it from the start.
    let stateRead = null;
    const afterAction = (action) => (action === 'resume'
        ? new Promise((resolve) => { stateRead = resolve; })
        : new Promise(() => {}));  // an unrelated read that never settles: it must not be awaited

    // The healthy control first: free, both offered and enabled.
    const menu = openWorkControls(anchor, work, { withResume: true, afterAction });
    assert.deepEqual(items(menu), [['resume', false], ['stop_now', false]], 'nothing pending: Resume and Stop are offered, both enabled');
    menu.querySelector('[data-task-control="resume"]').click();
    await until(() => posts.length === 1);
    assert.match(posts[0].url, /\/resume$/);
    posts[0].resolve({ ok: true, status: 200, json: async () => ({ ok: true }) });
    await until(() => stateRead);
    assert.deepEqual([free(), taskControlBusy(id)], [false, false], 'the answer is in: only its own bounded read is awaited');
    stateRead();  // the bounded read settles while the independent one is still out
    await until(free);
    assert.equal(posts.filter((post) => post.url.endsWith('/resume')).length, 1, 'one Resume POST, one confirmation');
    const later = openWorkControls(anchor, work, { withResume: true, afterAction });
    assert.ok(later.querySelector('[data-task-control="resume"]'), 'free again with the independent read still out: Resume is offered');
});

test('a held model answer receiver is worded by what is known', () => {
    assert.equal(heldWorkFacts([{ kind: 'model_handoff', task_id: 't-1', reason: 'local_completion_unknown' }]),
        'it is unknown whether the model answer receiver of task t-1 ended: its worker cannot be proven ended');
    assert.equal(heldWorkFacts([{ kind: 'model_handoff', task_id: 't-1', reason: 'local_call_live' }]),
        'a model answer for task t-1 is awaited by a live worker');
    assert.equal(heldWorkFacts([{ kind: 'model_handoff', reason: 'no_local_receiver' }]),
        'a model request has no recorded result');
});

test('an accepted Resume keeps its own words and names the gap its Pause did not wait for', async () => {
    const { membershipGapWords } = await import('../modules/task_control_menu.js');
    const toasts = [];
    const toast = (...args) => toasts.push(args);
    const gaps = { rows: [{ task_id: 'stray', reason: 'body_unreadable' }, { task_id: 'stray-2', reason: 'body_not_object' }],
        count: 2 };
    // The server's own answer (queue_transitions.resume_budget_paused_task): its message words the same fact.
    await resumeTaskAction('gap-root', { resume: async () => ({ ok: true, membership_gaps: gaps,
        message: 'Resume accepted. The Pause did not wait for 2 saved task records ...' }), toast });
    assert.deepEqual(toasts[0], ['Resuming: the task returns to the queue.', 'info']);
    assert.deepEqual(toasts[1], ['The Pause did not wait for 2 saved task records that could not be read and that '
        + 'nothing ties to this task (stray, stray-2); whether they belong here is unknown.', 'warn',
    { ttl: 0, dismissLabel: 'Dismiss notice' }]);
    toasts.length = 0;
    await resumeTaskAction('tree-gap', { resume: async () => ({ ok: true, tree_only: true,
        membership_gaps: { rows: [{ task_id: 'x', reason: 'identity_mismatch' }], count: 1 } }), toast });
    assert.equal(toasts[0][0], 'Resumed. Ouroboros or you choose which subagents continue.');
    assert.match(toasts[1][0], /^The Pause did not wait for 1 saved task record .* \(x\); whether it belongs here is unknown\.$/);
    toasts.length = 0;
    await resumeTaskAction('plain', { resume: async () => ({ ok: true }), toast });
    assert.deepEqual(toasts, [['Resuming: the task returns to the queue.', 'info']], 'no gap, no second notice');
    // Malformed or empty gaps say nothing rather than guess.
    for (const bad of [null, {}, { count: 0, rows: [] }, { count: '2', rows: [] }, { count: -1 }]) {
        assert.equal(membershipGapWords(bad, 'resume'), '');
    }
    assert.equal(membershipGapWords({ count: 1, rows: [{ task_id: 'a' }] }, 'pause'), '', 'only the two actions');
    assert.match(membershipGapWords({ count: 3, rows: [] }, 'continue'), /\(…\)/, 'a count without names still counts');
});

test('a gap notice keeps all its words beside its close button; the page stack it cuts is a keyboard region', async (t) => {
    const { announceMembershipGap } = await import('../modules/task_control_menu.js');
    const { showToast } = await import('../modules/toast.js');
    const { installDomDouble } = await import('./dom_double.js');
    const env = installDomDouble();
    t.after(() => env.restore());
    const make = env.doc.createElement;
    let cut = true;  // the double has no layout: the stack's measured height is given (the browser lane measures it)
    env.doc.createElement = (tag) => Object.defineProperties(make(tag), {
        scrollHeight: { get: () => (cut ? 900 : 40) }, clientHeight: { value: 40 } });
    const rows = Array.from({ length: 20 }, (_, n) => ({ task_id: `${String(n).padStart(2, '0')}-${'f'.repeat(30)}`,
        reason: 'body_unreadable' }));
    const words = announceMembershipGap({ rows, count: 23 }, 'resume');
    const stack = env.doc.getElementById('toast-stack');
    const toast = stack.querySelector('.toast-warn');
    const [box, close] = toast.children;
    assert.equal(box.className, 'toast-words');
    assert.equal(box.textContent, words, 'every name stays in the notice');
    assert.match(words, /\(00-f+, 01-f+, .*, 19-f+, …\); whether they belong here is unknown\.$/);
    assert.ok(!box.hasAttribute('tabindex'), 'the words never scroll on their own: the stack does');
    assert.deepEqual([stack.tabIndex, stack.getAttribute('role'), stack.getAttribute('aria-label')],
        [0, 'region', 'Scrollable notices']);
    assert.equal(close.getAttribute('aria-label'), 'Dismiss notice');
    cut = false;
    toast.dispose();
    assert.ok(!stack.hasAttribute('tabindex') && !stack.hasAttribute('role') && !stack.hasAttribute('aria-label'),
        'once everything fits again the stack is no keyboard stop');
    const short = showToast('One short notice.', 'warn', { ttl: 0, dismissLabel: 'Dismiss notice' }).children[0];
    assert.equal(short.className, 'toast-words');
    assert.ok(!stack.hasAttribute('tabindex'), 'words that fit make no keyboard stop');
    const plain = showToast('Saved.', 'ok', { ttl: 0 });
    assert.equal(plain.querySelector('.toast-words')?.textContent, 'Saved.');
    assert.equal(plain.querySelector('.toast-dismiss')?.getAttribute('aria-label'), 'Dismiss notice',
        'a non-expiring toast always has a keyboard-reachable close');
    cut = true;  // a room's stack keeps its own place: never the page's scrolling region
    const host = env.doc.createElement('div');
    env.doc.body.append(host);
    showToast('A room notice.', 'warn', { ttl: 0, dismissLabel: 'Dismiss notice', host });
    assert.ok(!host.querySelector('.toast-stack--room').hasAttribute('tabindex'));
});

test('two unconfirmed effects are worded in the plural, as on the server', () => {
    assert.equal(heldWorkFacts([{ kind: 'tool_handoff', task_id: 't', tool: 'run_command',
        reason: 'returned_effect_unconfirmed', effects: ['started process', 'MCP request'] }]),
    'the run_command call of task t returned; its started process and MCP request are not confirmed');
});
