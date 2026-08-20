// Behavioural characterization of the live-card owner-actions module, exercised
// where the code now lives. The actions are DOM-light — they read and write a
// card record, mount or remove one button, and settle against the durable task
// record — so a small element model plus a stubbed fetch reaches every branch:
// the honest "Cancelling…"/"Finalizing…" interim, the 404 completion race, the
// unproven-detail guard, and the one-way project conversion.

import assert from 'node:assert/strict';
import test from 'node:test';

import { createCardActions, projectIdFromTask } from '../modules/chat_card_actions.js';

function makeElement(tag = 'div') {
    const el = {
        tagName: tag.toUpperCase(),
        type: '',
        className: '',
        textContent: '',
        title: '',
        disabled: false,
        innerHTML: '',
        dataset: {},
        attributes: {},
        children: [],
        listeners: {},
        removed: false,
        parentElement: null,
        appendChild(child) { el.children.push(child); child.parentElement = el; return child; },
        append(...nodes) { nodes.forEach((node) => el.appendChild(node)); },
        insertBefore(child) { el.children.unshift(child); child.parentElement = el; return child; },
        replaceChildren(...nodes) { el.children = []; nodes.forEach((node) => el.appendChild(node)); },
        remove() { el.removed = true; el.parentElement = null; },
        classList: { add() {}, remove() {}, toggle() {} },
        setAttribute(name, value) { el.attributes[name] = String(value); },
        addEventListener(type, fn) { (el.listeners[type] ||= []).push(fn); },
        click(event = { stopPropagation() {} }) { (el.listeners.click || []).forEach((fn) => fn(event)); },
        querySelector(selector) { return el.stubbedNodes?.[selector] ?? null; },
    };
    return el;
}

function makeRecord(overrides = {}) {
    const root = makeElement('div');
    root.stubbedNodes = { '.chat-live-actions': null, '[data-cancel-run]': null };
    return {
        groupId: 'task-1',
        isSubagent: false,
        finished: false,
        cancelPendingPolicy: '',
        cancelRunBtn: null,
        timelineEl: null,
        root,
        phaseEl: Object.assign(makeElement('span'), {
            className: 'chat-live-phase working',
            textContent: 'Working',
            dataset: { phase: 'working' },
        }),
        ...overrides,
    };
}

function actions({ cancelable = ['task-1'], responses = [] } = {}) {
    const liveCardRecords = new Map();
    const cancelableTaskIds = new Set(cancelable);
    const finished = [];
    const freed = [];
    const calls = [];

    const priorDocument = globalThis.document;
    const priorFetch = globalThis.fetch;
    globalThis.document = {
        createElement: (tag) => makeElement(tag),
        getElementById: () => null,
        body: makeElement('body'),
    };
    globalThis.fetch = async (url, init = {}) => {
        calls.push({ url, method: init.method || 'GET' });
        const next = responses.shift();
        if (typeof next === 'function') return next();
        return next ?? { ok: true, status: 200, json: async () => ({}) };
    };

    const api = createCardActions({
        liveCardRecords,
        cancelableTaskIds,
        // The viewport wrapper is chat.js's; here it must simply run the mutation.
        withStableViewport: (mutate) => mutate(),
        finishLiveCard: (taskId, phase) => finished.push({ taskId, phase }),
        signalChatFreed: () => freed.push(true),
    });

    return {
        ...api,
        liveCardRecords,
        cancelableTaskIds,
        finished,
        freed,
        calls,
        restore() { globalThis.document = priorDocument; globalThis.fetch = priorFetch; },
    };
}

test('a project id is derived from the task id, slug-safe and bounded', () => {
    assert.equal(projectIdFromTask('Task_42.a'), 'task-task_42.a');
    assert.equal(projectIdFromTask('  weird//id  '), 'task-weird-id');
    assert.equal(projectIdFromTask('x'.repeat(200)).length, 64);
    assert.match(projectIdFromTask(''), /^task-[a-z0-9]+$/);
});

test('a pending cancel keeps the card honestly live: Cancelling… or Finalizing…', () => {
    const a = actions();
    const record = makeRecord();
    a.liveCardRecords.set('task-1', record);
    a.markLiveCardCancelPending('task-1', false);
    assert.equal(record.phaseEl.textContent, 'Cancelling…');
    assert.equal(record.phaseEl.dataset.phase, 'working', 'the card stays live, never an instant "Cancelled"');
    assert.equal(record.cancelPendingPolicy, 'immediate');
    a.markLiveCardCancelPending('task-1', true);
    assert.equal(record.phaseEl.textContent, 'Finalizing…');
    assert.equal(record.cancelPendingPolicy, 'finalize');
    // A finished card is never re-marked as pending.
    record.finished = true;
    record.phaseEl.textContent = 'Done';
    a.markLiveCardCancelPending('task-1', false);
    assert.equal(record.phaseEl.textContent, 'Done');
    a.restore();
});

test('the phase snapshot round-trips, but never onto a finished card', () => {
    const a = actions();
    const record = makeRecord();
    const snapshot = a.captureLiveCardPhase(record);
    assert.deepEqual(snapshot, { phase: 'working', text: 'Working', className: 'chat-live-phase working' });
    record.phaseEl.textContent = 'Cancelling…';
    a.restoreLiveCardPhase(record, snapshot);
    assert.equal(record.phaseEl.textContent, 'Working');
    record.finished = true;
    record.phaseEl.textContent = 'Done';
    a.restoreLiveCardPhase(record, snapshot);
    assert.equal(record.phaseEl.textContent, 'Done');
    assert.equal(a.captureLiveCardPhase({}), null);
    a.restore();
});

test('reconciliation reads the typed pending projection before the terminal statuses', () => {
    const a = actions();
    const record = makeRecord();
    a.liveCardRecords.set('task-1', record);
    // A task wedged in the legacy cancel_requested STATUS latch with an OPEN
    // intent is pending, not terminal: the card keeps the interim.
    a.reconcileCancelCardFromDetail(record, 'task-1', {
        status: 'cancel_requested',
        cancel_state: 'pending',
    });
    assert.deepEqual(a.finished, []);
    assert.equal(record.phaseEl.textContent, 'Cancelling…');
    // An intent-free legacy latch is history awaiting migration: it resolves,
    // and a cancelled root says "Cancelled", never a generic "Done".
    a.reconcileCancelCardFromDetail(record, 'task-1', { status: 'cancelled' });
    assert.deepEqual(a.finished, [{ taskId: 'task-1', phase: 'cancelled' }]);
    // A still-running record resolves nothing.
    a.reconcileCancelCardFromDetail(record, 'task-1', { status: 'running' });
    assert.equal(a.finished.length, 1);
    // The same seam renders a pending SOFT stop as "Finalizing…".
    record.finished = false;
    a.reconcileCancelCardFromDetail(record, 'task-1', {
        status: 'running',
        cancel_state: 'pending',
        stop_policy: 'finalize_then_cancel',
    });
    assert.equal(record.phaseEl.textContent, 'Finalizing…');
    assert.equal(a.finished.length, 1);
    a.restore();
});

test('the Cancel trigger is mounted only while the card is eligible, and never twice', () => {
    const a = actions();
    const record = makeRecord();
    const mounted = a.ensureLiveActionsEl(record);
    record.root.stubbedNodes['.chat-live-actions'] = mounted;
    a.syncCancelRunButton(record);
    assert.notEqual(record.cancelRunBtn, null);
    assert.equal(record.cancelRunBtn.dataset.cancelRun, '1');
    assert.equal(mounted.children.length, 1);
    // An already-rendered trigger is adopted, not duplicated.
    record.root.stubbedNodes['[data-cancel-run]'] = record.cancelRunBtn;
    a.syncCancelRunButton(record);
    assert.equal(mounted.children.length, 1);
    // Losing the host-attested marker removes it.
    a.cancelableTaskIds.delete('task-1');
    a.syncCancelRunButton(record);
    assert.equal(record.cancelRunBtn, null);
    a.restore();
});

test('a converted card refuses new action chrome (its task belongs to the panel)', () => {
    const a = actions();
    const record = makeRecord();
    record.root.dataset.projectCreated = '1';
    assert.equal(a.ensureLiveActionsEl(record), null);
    a.restore();
});

test('markTaskCancelable learns the marker once and resyncs an existing card', () => {
    const a = actions({ cancelable: [] });
    const record = makeRecord();
    const mounted = makeElement('div');
    record.root.stubbedNodes['.chat-live-actions'] = mounted;
    a.liveCardRecords.set('task-1', record);
    a.markTaskCancelable('  task-1  ');
    assert.equal(a.cancelableTaskIds.has('task-1'), true);
    assert.notEqual(record.cancelRunBtn, null);
    a.markTaskCancelable('');
    assert.equal(a.cancelableTaskIds.size, 1);
    a.restore();
});

test('a 404 cancel drops the eligibility authority and reconciles from the durable record', async () => {
    const a = actions({
        responses: [
            { ok: false, status: 404, json: async () => ({ error: 'gone' }) },
            { ok: true, status: 200, json: async () => ({ status: 'completed' }) },
        ],
    });
    const record = makeRecord();
    record.cancelRunBtn = makeElement('button');
    a.liveCardRecords.set('task-1', record);
    await a.cancelRunFromCard(record, 'stop_now');
    assert.equal(a.cancelableTaskIds.has('task-1'), false, 'the eligibility AUTHORITY is cleared, not just the flag');
    assert.equal(record.cancelable, false);
    assert.deepEqual(a.finished, [{ taskId: 'task-1', phase: 'done' }]);
    a.restore();
});

test('a failed cancel whose detail fetch also fails proves nothing: no restore, no re-enable', async () => {
    const a = actions({
        responses: [
            { ok: false, status: 503, json: async () => ({ error: 'busy' }) },
            () => { throw new TypeError('network down'); },
        ],
    });
    const record = makeRecord();
    record.cancelRunBtn = makeElement('button');
    a.liveCardRecords.set('task-1', record);
    await a.cancelRunFromCard(record, 'stop_now');
    assert.equal(record.cancelRunBtn.disabled, true);
    assert.equal(record.phaseEl.textContent, 'Cancelling…', 'the pending presentation survives an unproven failure');
    assert.deepEqual(a.finished, []);
    a.restore();
});

test('conversion turns the whole card into a terminal project chip and frees the composer', () => {
    const a = actions();
    const record = makeRecord();
    record.root.dataset.projectCreating = '1';
    globalThis.requestAnimationFrame = (fn) => fn();
    a.markCardConverted(record, { id: 'proj-1', name: 'Refactor' });
    assert.equal(record.root.dataset.projectCreating, undefined);
    assert.equal(record.root.dataset.projectCreated, '1');
    assert.equal(record.root.dataset.projectId, 'proj-1');
    assert.equal(record.root.children.length, 1, 'the live timeline is swapped for the chip in one paint');
    assert.equal(record.finished, true);
    assert.equal(record.cancelRunBtn, null);
    assert.deepEqual(a.freed, [true]);
    a.restore();
});
