import assert from 'node:assert/strict';
import test from 'node:test';
import { createChatDecision } from '../modules/chat_decision.js';
import { createChatInstance } from '../modules/chat.js';
import { stampNodeTimestamp } from '../modules/chat_activity.js';
import { createMainWaiting } from '../modules/main_waiting.js';
import { fixture as mediaFixture, NodeStub } from './helpers/media_dom.js';
import { ElementStub, installDom, restoreDom } from './chat_dom_fixture.js';

const turn = () => new Promise(resolve => setTimeout(resolve, 0));
const find = (node, predicate) => predicate(node) ? node : node.children.reduce((found, child) => found || find(child, predicate), null);
const all = (node, predicate) => [...(predicate(node) ? [node] : []), ...node.children.flatMap(child => all(child, predicate))];
const question = (id, state = 'open') => ({ task_id: 'task-1', quiz_id: id, project_id: 'project-1',
    project_chat_id: 12, project_name: 'Build', quiz_state: state, question: 'Choose?', options: ['Yes', 'No'],
    materials: [{ task_id: 'task-1', name: 'brief.txt', label: 'brief.txt', sha256: 'a'.repeat(64) }] });

test('replacing an unknown mirror releases its material listeners and reader while retaining fresh materials and drafts', async t => {
    const fx = mediaFixture();
    const changes = {
        focus() { document.activeElement = this; },
        replaceChildren(...nodes) { for (const node of [...this.children]) node.remove(); this.append(...nodes); },
        replaceWith(node) { this.before(node); this.remove(); },
    };
    const prior = Object.fromEntries(Object.keys(changes).map(key => [key, Object.getOwnPropertyDescriptor(NodeStub.prototype, key)]));
    prior.ownerDocument = Object.getOwnPropertyDescriptor(NodeStub.prototype, 'ownerDocument');
    Object.assign(NodeStub.prototype, changes);
    Object.defineProperty(NodeStub.prototype, 'ownerDocument', { configurable: true, get() { return document; } });
    const classes = Object.getPrototypeOf(document.body.classList);
    const remove = classes.remove;
    classes.remove = function(...names) { names.forEach(name => this.toggle(name, false)); };
    t.after(() => {
        for (const [key, descriptor] of Object.entries(prior)) {
            if (descriptor) Object.defineProperty(NodeStub.prototype, key, descriptor); else delete NodeStub.prototype[key];
        }
        if (remove) classes.remove = remove; else delete classes.remove;
        fx.controller.destroy(); fx.restore();
    });
    t.mock.method(globalThis, 'fetch', async () => new Response('Exact retained text'));
    // Main's feed: a copy enters through the chat's insertion seam; the census only confirms it.
    const column = document.createElement('div'); document.body.append(column);
    const released = [], markdownDisposed = [];
    const decision = createChatDecision({ isMain: true, insertMessageNode: node => { column.append(node); return true; },
        apiFetch: async () => assert.fail('replacement never answers a question'),
        showToast() {}, mountMarkdown: (node, text) => { node.textContent = text; },
        enhanceMarkdown: node => () => markdownDisposed.push(node),
        renderMaterial: fx.controller.buildDocumentBubble,
        releaseMaterials: node => { released.push(node); fx.controller.release(node); },
        // The production frame: a copy's Project reference joins its voice line, its host facts its time line.
        frameNode: fx.controller.bubbleFrameNode,
    });
    t.after(() => decision.destroy());
    const rows = [question('recover', 'unknown'), question('draft')];
    for (const row of rows) decision.appendQuestionPointer(row);
    const old = column.children[0], sibling = column.children[1];
    const draft = sibling.querySelector('.chat-quiz-comment');
    draft.value = 'Keep my unfinished answer';
    const oldFile = old.querySelector('.chat-file-card');
    await oldFile.click(); await turn();
    const reader = document.body.querySelector('.document-reader');
    assert.ok(reader.attributes.has('open'));
    draft.focus();
    decision.syncCurrentQuestions([question('recover'), rows[1]], true);
    assert.deepEqual(released, [old]);
    assert.equal(markdownDisposed.length, 1);
    assert.equal(reader.attributes.has('open'), false, 'the released material cannot keep its reader open');
    assert.equal(oldFile.listeners.get('click').size, 0, 'the detached material has no live listener');
    assert.ok([...oldFile.listeners.values()].every(listeners => listeners.size === 0));
    assert.equal(document.activeElement, draft);
    assert.equal(draft.value, 'Keep my unfinished answer');
    assert.equal(column.children[1], sibling);
    const fresh = column.children[0];
    assert.notEqual(fresh, old);
    assert.ok(fresh.querySelector('.chat-quiz-comment'));
    await oldFile.click(); await turn();
    assert.equal(reader.attributes.has('open'), false);
    await fresh.querySelector('.chat-file-card').click(); await turn();
    const freshReader = document.body.querySelector('.document-reader');
    assert.ok(freshReader?.attributes.has('open'), 'the replacement owns usable fresh listeners');
    fx.controller.release(fresh);
    assert.equal(freshReader.attributes.has('open'), false);
    decision.appendQuestionPointer(question('focused', 'unknown'));
    const focusedOld = column.children[2];
    focusedOld.querySelector('.chat-file-card').focus();
    decision.syncCurrentQuestions([question('recover'), rows[1], question('focused')], true);
    assert.equal(document.activeElement, column.children[2].querySelector('.chat-quiz-project'));
    assert.deepEqual(released, [old, focusedOld]);
    assert.equal(draft.value, 'Keep my unfinished answer');
});

test('Main renders the existing Try again control for a wait with no proven automatic route', async () => {
    const calls = [];
    const env = installDom(async (url, init) => {
        calls.push({ url, body: JSON.parse(init.body) });
        return { ok: true, status: 202, json: async () => ({ accepted: true }) };
    });
    // The waiting task is one owner-action System message in Main's feed.
    const feed = document.createElement('div');
    env.mount.append(feed);
    const waiting = createMainWaiting({ page: env.mount, messagesDiv: feed, toolbar: null, stampNodeTimestamp,
        insertMessageNode: node => { feed.append(node); return true; } });
    try {
        waiting.snapshot({ main_attention: [{ task_id: 'task-1', activity_id: 'task-1', project_id: 'project-1', phase: 'working',
            task_attempt: 1, attention_reason: 'model_access', model_waits: { w: {
                wait_id: 'w', state: 'waiting', reason: 'quota', role: 'main', source: 'codex',
                task_attempt: 1, revision: 7, auto_continue: true, requires_owner_action: true,
            } } }], main_attention_complete: true, current_questions_complete: true });
        const message = find(feed, node => node.dataset.ownerAction === 'task-1');
        const button = find(message, node => Object.hasOwn(node.dataset, 'waitRetry'));
        assert.ok(button && !button.disabled);
        for (const handler of button.listeners.get('click')) await handler({ target: button });
        assert.equal(calls.length, 1);
        assert.equal(calls[0].url, '/api/decisions');
        assert.equal(calls[0].body.action, 'retry');
        assert.equal(calls[0].body.revision, 7);
    } finally { waiting.destroy(); restoreDom(env.prior); }
});

// The panel's sentences are gone with it: a question whose original chat no import could prove
// has no copy to show, and Main neither invents one nor says anything about it.
test('a question whose original chat is unknown leaves no card and no sentence anywhere in Main', async t => {
    const original = ElementStub.prototype.querySelector;
    ElementStub.prototype.querySelector = function(selector) {
        return original.call(this, selector) || this.children.map(child => child.querySelector(selector)).find(Boolean) || null;
    };
    const priorCss = globalThis.CSS;
    globalThis.CSS = { escape: value => String(value) };  // the browser global Main's navigator looks cards up with
    t.after(() => { ElementStub.prototype.querySelector = original; globalThis.CSS = priorCss; });
    const env = installDom(async url => ({ ok: true, json: async () => (String(url).startsWith('/api/chat/history')
        ? { messages: [], window: { complete: true } } : {}) }));
    const handlers = new Map();
    const instance = createChatInstance({
        ws: { on(type, fn) { handlers.set(type, [...(handlers.get(type) || []), fn]); return () => {}; },
            isConnected: () => true, send() {} },
        state: { activePage: 'chat', projectChatIds: new Set(), unreadCount: 0 }, updateUnreadBadge() {},
        stateSnapshots: { begin: () => ({}), gate() { return Promise.resolve({}); }, isCurrent: () => true,
            apply() {}, fail() {}, latest: () => null },
        chatId: 1, idPrefix: 'chat', mountEl: env.mount });
    const emit = (type, msg = {}) => (handlers.get(type) || []).forEach(fn => fn(msg));
    // The instance's own follow-up reads (state refresh on reopen, child holds) finish before teardown.
    const settle = async () => { for (let i = 0; i < 8; i += 1) await turn(); };
    const routing = { task_id: 'task-1', quiz_id: 'q1', reason: 'publication_address_unknown' };
    const rendered = () => all(env.mount, () => true)
        .flatMap(node => [node.textContent, node.innerHTML, ...node.attributes.values()]).join('\n');
    const cards = () => all(env.mount, node => node.classList.contains('chat-quiz-card'));
    const pill = () => find(env.mount, node => node.classList.contains('chat-question-nav'));
    const census = patch => instance.hydrateStateSnapshot({ main_attention: [], main_attention_complete: true,
        current_questions: [], current_questions_complete: false, current_questions_gaps: [routing], ...patch }, Infinity);
    try {
        await turn();
        assert.ok(pill(), 'Main keeps its navigator in the composer row');
        for (const gaps of [[routing], [routing, { ...routing, quiz_id: 'q2' }, { reason: 'question_index_unavailable' }]]) {
            census({ current_questions_gaps: gaps });
            assert.doesNotMatch(rendered(), /original chat is unknown|can't be shown|questions and actions could not be refreshed/);
            assert.deepEqual(cards(), [], 'a gap is no card');
            assert.equal(pill().hidden, true, 'a gap is nothing to go to');
        }
        emit('close');  // a lost connection says nothing about questions either
        assert.doesNotMatch(rendered(), /questions and actions could not be refreshed/);
        emit('open', { previouslyConnected: true });
        await settle();
        // A real unanswered question beside the gap is the one target; the census still mounts no card.
        census({ current_questions: [question('real')] });
        assert.equal(pill().hidden, false);
        assert.equal(pill().querySelector('.chat-question-nav-count').textContent, '1');
        assert.deepEqual(cards(), []);
        assert.doesNotMatch(rendered(), /original chat is unknown|can't be shown/);
        await settle();
    } finally { instance.destroy(); restoreDom(env.prior); }
});

// Such gaps can last for good, so an incomplete census still speaks for every question they do
// not name: Main's copies and its navigator move on instead of freezing.
test('through the real Main wiring, a gap withholds only what it names: an omitted copy waits for its read, its own stays unknown', async t => {
    const original = ElementStub.prototype.querySelector;
    ElementStub.prototype.querySelector = function(selector) {
        return original.call(this, selector) || this.children.map(child => child.querySelector(selector)).find(Boolean) || null;
    };
    const priorCss = globalThis.CSS;
    globalThis.CSS = { escape: value => String(value) };
    t.after(() => { ElementStub.prototype.querySelector = original; globalThis.CSS = priorCss; });
    const env = installDom(async url => ({ ok: true, json: async () => (String(url).startsWith('/api/chat/history')
        ? { messages: [], window: { complete: true } } : {}) }));
    const handlers = new Map();
    const instance = createChatInstance({
        ws: { on(type, fn) { handlers.set(type, [...(handlers.get(type) || []), fn]); return () => {}; },
            isConnected: () => true, send() {} },
        state: { activePage: 'chat', projectChatIds: new Set(), unreadCount: 0 }, updateUnreadBadge() {},
        stateSnapshots: { begin: () => ({}), gate() { return Promise.resolve({}); }, isCurrent: () => true,
            apply() {}, fail() {}, latest: () => null },
        chatId: 1, idPrefix: 'chat', mountEl: env.mount });
    const settle = async () => { for (let i = 0; i < 8; i += 1) await turn(); };
    const row = (id, taskId) => ({ task_id: taskId, quiz_id: id, project_id: 'project-1', project_chat_id: 12,
        project_name: 'Build', quiz_state: 'open', question: `Question ${id}?`, options: ['Yes', 'No'], ts: '2026-10-13T10:00:00Z' });
    const pointer = r => (handlers.get('chat') || []).forEach(fn => fn({ ...r, type: 'chat', chat_id: 1, role: 'system',
        system_type: 'project_question_pointer' }));
    const census = (questions, patch = {}) => instance.hydrateStateSnapshot({ current_questions: questions,
        current_questions_complete: true, current_questions_gaps: [], main_attention: [], main_attention_complete: true, ...patch }, Infinity);
    const card = id => find(env.mount, node => node.classList.contains('chat-quiz-card') && node.dataset.quizId === id);
    const count = () => find(env.mount, node => node.classList.contains('chat-question-nav'))
        .querySelector('.chat-question-nav-count').textContent;
    const elsewhere = { task_id: 'task-3', quiz_id: 'q9', reason: 'publication_address_unknown' };
    const leaving = row('leaving', 'task-1'), kept = row('kept', 'task-2');
    try {
        await settle();
        pointer(leaving); pointer(kept);
        census([leaving, kept]);
        assert.deepEqual([card('leaving').dataset.state, card('kept').dataset.state, count()], ['open', 'open', '2']);
        const leavingCard = card('leaving');
        census([kept], { current_questions_complete: false, current_questions_gaps: [elsewhere] });
        await settle();
        // Vouched for and omitted: no longer counted, but absence is no outcome. Its exact read decides;
        // this unreadable one leaves the same copy withheld (#1993).
        assert.deepEqual([card('leaving') === leavingCard, card('leaving')?.dataset.state, count()], [true, 'unknown', '1']);
        assert.deepEqual([card('kept').dataset.state, count()], ['open', '1']);
        pointer(leaving);
        census([leaving, kept]);
        census([kept], { current_questions_complete: false,
            current_questions_gaps: [{ ...elsewhere, task_id: 'task-1', quiz_id: 'leaving' }] });
        assert.deepEqual([card('leaving').dataset.state, count()], ['unknown', '2'], 'its own gap: it stays, awaiting proof');
        census([leaving, kept]);
        census([kept], { current_questions_complete: false, current_questions_gaps: [elsewhere, { reason: 'question_index_unavailable' }] });
        assert.deepEqual([card('leaving').dataset.state, count()], ['unknown', '2'], 'a gap naming no task withholds everything');
        await settle();
    } finally { instance.destroy(); restoreDom(env.prior); }
});
