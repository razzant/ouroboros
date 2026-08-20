// Behavioural characterization of the three chat primitives owners, exercised
// where the code now lives: the page-wide once-per-incident toast keys
// (chat_notices.js), the header control projection (chat_header_controls.js),
// and the per-frame thread routing that decides which column sees a frame and
// whether it may raise the global unread badge (chat_frame_routing.js).

import assert from 'node:assert/strict';
import test from 'node:test';

import { createFrameRouting } from '../modules/chat_frame_routing.js';
import { createHeaderControls } from '../modules/chat_header_controls.js';
import { showContextFitToast, showTaskIncidentToast } from '../modules/chat_notices.js';

// --- transient notices ---------------------------------------------------

function notices() {
    const toasts = [];
    const priorDocument = globalThis.document;
    const stack = { id: '', className: '', appendChild() {}, setAttribute() {} };
    globalThis.document = {
        getElementById: () => stack,
        createElement: () => ({
            className: '',
            set textContent(value) { toasts.push(value); },
            get textContent() { return toasts[toasts.length - 1] ?? ''; },
            setAttribute() {},
            addEventListener() {},
            classList: { add() {} },
            remove() {},
        }),
        body: { appendChild() {} },
    };
    return { toasts, restore() { globalThis.document = priorDocument; } };
}

test('a task incident toasts once per key, even when Main mirrors a Project frame', () => {
    const n = notices();
    const frame = { task_incident: 'worker_crashed', task_id: 'task-1', content: 'Worker crashed' };
    showTaskIncidentToast(frame);
    showTaskIncidentToast({ ...frame });  // the Main mirror of the same incident
    assert.deepEqual(n.toasts, ['Worker crashed']);
    // No incident, no toast.
    showTaskIncidentToast({ task_id: 'task-2' });
    assert.equal(n.toasts.length, 1);
    // A different task with the same incident is a different key.
    showTaskIncidentToast({ ...frame, task_id: 'task-2', content: 'Worker crashed again' });
    assert.equal(n.toasts.length, 2);
    n.restore();
});

test('only the context-fit checkpoint toasts, and only once per round', () => {
    const n = notices();
    showContextFitToast({ checkpoint_kind: 'something_else', task_id: 'task-3' });
    assert.deepEqual(n.toasts, []);
    const evt = { checkpoint_kind: 'context_fit_low_retry', task_id: 'task-3', round: 4 };
    showContextFitToast(evt);
    showContextFitToast({ ...evt });
    assert.equal(n.toasts.length, 1);
    assert.match(n.toasts[0], /Retrying the same model once with the task-local Low view/);
    n.restore();
});

// --- header controls -----------------------------------------------------

function headerControls({ activePage = 'chat' } = {}) {
    const nodes = {
        evolve: { dataset: { chatCommand: 'evolve' }, title: '', classList: { on: null, toggle(_n, v) { nodes.evolve.classList.on = v; } } },
        bg: { dataset: { chatCommand: 'bg' }, title: '', classList: { on: null, toggle(_n, v) { nodes.bg.classList.on = v; } } },
        more: { classList: { active: null, toggle(_n, v) { nodes.more.classList.active = v; } } },
        ctx: { dataset: {} },
        text: { textContent: '' },
        fill: { style: {} },
    };
    const headerActions = {
        querySelectorAll: () => [nodes.evolve, nodes.bg],
        querySelector: () => nodes.more,
    };
    const byId = (suffix) => ({ 'context-mode': nodes.ctx, 'budget-text': nodes.text, 'budget-bar-fill': nodes.fill }[suffix] ?? null);
    const api = createHeaderControls({ byId, headerActions, state: { activePage } });
    return { ...api, nodes };
}

test('the header projects both toggles, the More dot, the context mode and the budget', () => {
    const h = headerControls();
    h.syncHeaderControlState({
        evolution_enabled: true,
        evolution_state: { detail: 'campaign running' },
        bg_consciousness_enabled: false,
        context_mode: 'low',
        accounting: { available: true },
        budget_total_usd: 100,
        budget_spent_usd: 25,
    });
    assert.equal(h.nodes.evolve.classList.on, true);
    assert.equal(h.nodes.evolve.title, 'campaign running');
    assert.equal(h.nodes.bg.classList.on, false);
    assert.equal(h.nodes.more.classList.active, true, 'an active mode stays visible without opening the menu');
    assert.equal(h.nodes.ctx.dataset.contextMode, 'low');
    assert.notEqual(h.nodes.text.textContent, '');
    assert.match(h.nodes.fill.style.width, /%$/);
});

test('an unknown context mode falls back to max, and neither mode active clears the dot', () => {
    const h = headerControls();
    h.syncHeaderControlState({ context_mode: 'nonsense', accounting: { available: false } });
    assert.equal(h.nodes.ctx.dataset.contextMode, 'max');
    assert.equal(h.nodes.more.classList.active, false);
    // A frame without a context_mode string does not touch the segment.
    h.nodes.ctx.dataset.contextMode = 'low';
    h.syncHeaderControlState({ accounting: { available: false } });
    assert.equal(h.nodes.ctx.dataset.contextMode, 'low');
});

test('a refresh off the chat page is skipped unless forced, and a failure renders unavailable', async () => {
    const priorFetch = globalThis.fetch;
    let calls = 0;
    globalThis.fetch = async () => { calls += 1; throw new TypeError('offline'); };
    const away = headerControls({ activePage: 'skills' });
    await away.refreshHeaderControlState();
    assert.equal(calls, 0, 'a background page does not poll');
    await away.refreshHeaderControlState(true);
    assert.equal(calls, 1);
    assert.equal(away.nodes.more.classList.active, false, 'an unreachable backend renders as unavailable, not stale');
    globalThis.fetch = priorFetch;
});

// --- per-frame thread routing --------------------------------------------

function frameRouting({ isMain = true, chatId = 1, activePage = 'chat', projectChatIds = [7] } = {}) {
    const state = { activePage, unreadCount: 0, projectChatIds: new Set(projectChatIds) };
    const badges = [];
    const api = createFrameRouting({ state, isMain, chatId, updateUnreadBadge: () => badges.push(state.unreadCount) });
    return { ...api, state, badges };
}

test('a project panel takes only its own thread', () => {
    const panel = frameRouting({ isMain: false, chatId: 7 });
    assert.equal(panel.isMyThread({ chat_id: 7 }), true);
    assert.equal(panel.isMyThread({ chat_id: 1 }), false);
    assert.equal(panel.isMyThread({}), false, 'an unstamped legacy frame defaults to Main');
});

test('Main keeps ordinary traffic and mirrors project progress, but never project chat', () => {
    const main = frameRouting();
    assert.equal(main.isMyThread({ chat_id: 1 }), true);
    // A registered Project frame reaches Main ONLY as a mirror, and only for the
    // mirrorable families.
    assert.equal(main.isMyThread({ chat_id: 7, is_progress: true }), false, 'no mirror requested');
    assert.equal(main.isMyThread({ chat_id: 7, is_progress: true }, { mirrorProject: true }), true);
    assert.equal(main.isMyThread({ chat_id: 7, type: 'log' }, { mirrorProject: true }), true);
    assert.equal(main.isMyThread({ chat_id: 7, system_type: 'task_summary' }, { mirrorProject: true }), true);
    assert.equal(main.isMyThread({ chat_id: 7, system_type: 'project_digest' }, { mirrorProject: true }), true);
    assert.equal(main.isMyThread({ chat_id: 7, role: 'user' }, { mirrorProject: true }), false);
    assert.equal(main.isProjectMirrorFrame(null), false);
});

test('the presentation mirror never creates a second Main unread', () => {
    const main = frameRouting({ activePage: 'skills' });
    main.incrementUnreadIfNeeded({ chat_id: 1 });
    assert.equal(main.state.unreadCount, 1);
    assert.deepEqual(main.badges, [1]);
    // A Project-origin frame is the Project's own unread authority.
    main.incrementUnreadIfNeeded({ chat_id: 7, is_progress: true });
    assert.equal(main.state.unreadCount, 1);
    // An open chat page is already read.
    const open = frameRouting({ activePage: 'chat' });
    open.incrementUnreadIfNeeded({ chat_id: 1 });
    assert.equal(open.state.unreadCount, 0);
    // The global badge tracks Main only.
    const panel = frameRouting({ isMain: false, chatId: 7, activePage: 'skills' });
    panel.incrementUnreadIfNeeded({ chat_id: 7 });
    assert.equal(panel.state.unreadCount, 0);
});
