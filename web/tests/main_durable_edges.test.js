import assert from 'node:assert/strict';
import test from 'node:test';
import { readFileSync } from 'node:fs';
import { createChatInstance } from '../modules/chat.js';
import { taskArtifactDownloadUrl } from '../modules/api_client.js';
import { stampNodeTimestamp } from '../modules/chat_activity.js';
import { createMainWaiting } from '../modules/main_waiting.js';
import { installDom, restoreDom, ElementStub, walkCard } from './chat_dom_fixture.js';
import { fixture as mediaFixture, NodeStub } from './helpers/media_dom.js';

const turn = () => new Promise(resolve => setTimeout(resolve, 0));
const find = (node, predicate) => predicate(node) ? node : node.children.reduce((found, child) => found || find(child, predicate), null);
const all = (node, predicate) => [...(predicate(node) ? [node] : []), ...node.children.flatMap(child => all(child, predicate))];
const page = (messages, next = null) => ({ messages, page_cursor: 'recent', next_cursor: next,
    has_more: Boolean(next), window: { complete: !next, truncated_by: next ? ['quota'] : [] } });
function nestedQueries(t) {
    const original = ElementStub.prototype.querySelector;
    ElementStub.prototype.querySelector = function(selector) {
        return original.call(this, selector) || this.children.map(child => child.querySelector(selector)).find(Boolean) || null;
    };
    t.after(() => { ElementStub.prototype.querySelector = original; });
}

test('actual two-page history controller folds closure after canonical GC without an empty bubble', async t => {
    nestedQueries(t);
    const calls = [];
    const closure = { role: 'system', text: '', system_type: 'quiz_closed', task_id: 'task-1',
        ts: '2026-10-09T01:00:00Z', history_id: 'chat:200', quiz: { quiz_id: 'q', state: 'closed',
            closed_reason: 'The action was withdrawn', closed_actor: 'owner', closed_at: '2026-10-09T01:00:00Z' } };
    const ask = { role: 'assistant', text: 'Proceed?', msg_type: 'quiz', task_id: 'task-1',
        ts: '2026-10-09T00:00:00Z', history_id: 'chat:100', quiz: { quiz_id: 'q', state: 'open', options: ['Yes', 'No'] } };
    const env = installDom(async url => {
        if (String(url).startsWith('/api/chat/history')) {
            const cursor = new URL(url, 'http://local').searchParams.get('cursor'); calls.push(cursor);
            return { ok: true, json: async () => cursor ? page([ask]) : page([closure], 'older') };
        }
        return { ok: true, json: async () => ({ active_direct_turns: [], owner_quiz: {} }) };
    });
    const priorSocket = globalThis.WebSocket; globalThis.WebSocket = { OPEN: 1 };
    const instance = createChatInstance({ ws: { on() { return () => {}; }, isConnected: () => true, send() {} },
        state: { activePage: 'chat', projectChatIds: new Set(), unreadCount: 0 }, updateUnreadBadge() {},
        stateSnapshots: { begin: () => ({}), gate() { return Promise.resolve({}); }, isCurrent: () => true, apply() {} },
        chatId: 2, idPrefix: 'chat', mountEl: env.mount, asPanel: true });
    try {
        await instance.refreshHistory({ revision: 1 });
        const messages = document.byId.get('chat-messages');
        const button = messages.querySelector('.chat-load-older-btn');
        for (const handler of button.listeners.get('click')) await handler({ target: button });
        await turn();
        const card = find(messages, node => node.classList.contains('chat-quiz-card'));
        assert.deepEqual(calls, [null, 'older']);
        assert.equal(card.dataset.state, 'closed');
        assert.equal(card.querySelector('.chat-quiz-closed-reason').textContent, closure.quiz.closed_reason);
        assert.equal(card.querySelector('.chat-quiz-comment'), null);
        assert.ok(card.querySelectorAll('.chat-quiz-option').every(option => option.disabled));
        assert.equal(find(messages, node => node.dataset.historyId === closure.history_id), null);
    } finally { instance.destroy(); restoreDom(env.prior); globalThis.WebSocket = priorSocket; }
});

test('nested text opens the existing document dialog and noncanonical addresses cannot enter its reader', async t => {
    const focus = NodeStub.prototype.focus;
    const replaceChildren = NodeStub.prototype.replaceChildren;
    const ownerDocument = Object.getOwnPropertyDescriptor(NodeStub.prototype, 'ownerDocument');
    Object.defineProperty(NodeStub.prototype, 'ownerDocument', { configurable: true, get() { return globalThis.document; } });
    NodeStub.prototype.focus = function() { document.activeElement = this; };
    NodeStub.prototype.replaceChildren = function(...nodes) { this.children = []; this.append(...nodes); };
    t.after(() => {
        if (focus) NodeStub.prototype.focus = focus; else delete NodeStub.prototype.focus;
        if (replaceChildren) NodeStub.prototype.replaceChildren = replaceChildren; else delete NodeStub.prototype.replaceChildren;
        if (ownerDocument) Object.defineProperty(NodeStub.prototype, 'ownerDocument', ownerDocument);
        else delete NodeStub.prototype.ownerDocument;
    });
    const fx = mediaFixture();
    const priorFetch = globalThis.fetch;
    globalThis.fetch = async () => new Response('Exact nested document', { headers: { 'content-length': '21' } });
    try {
        const url = taskArtifactDownloadUrl('task-1', 'brief.txt', 'research/brief.txt');
        const bubble = fx.controller.buildDocumentBubble({ task_id: 'task-1', filename: 'brief.txt', download_url: url });
        document.body.append(bubble);
        assert.match(bubble.innerHTML, /is-read/);
        await bubble.querySelector('.chat-file-card').click(); await turn();
        const reader = document.body.querySelector('.document-reader');
        assert.ok(reader && reader.attributes.has('open'));
        for (const bad of [url + '&extra=1', url.replace('task-1', 'foreign'), url.replace('research', '..')]) {
            assert.doesNotMatch(fx.controller.buildDocumentBubble({ task_id: 'task-1', filename: 'brief.txt', download_url: bad }).innerHTML, /is-read/);
        }
    } finally { fx.controller.destroy(); fx.restore(); globalThis.fetch = priorFetch; }
});

test('Main child wait uses child revision for auto-off and the existing sign-in entry', async () => {
    const calls = [], settings = [];
    const projected = process.env.MAIN_REVIEW_HISTORY_PAYLOAD
        ? JSON.parse(readFileSync(process.env.MAIN_REVIEW_HISTORY_PAYLOAD, 'utf8')).main_attention?.[0] : null;
    const wait = projected?.model_waits.login || { wait_id: 'login', revision: 3, task_attempt: 3, role: 'main', reason: 'auth', state: 'waiting',
        model: 'claudexor::codex=test-model', auto_continue: true, credential_harness: 'codex', credential_profile_id: 'fixture' };
    const env = installDom(async (url, init) => { calls.push({ url, body: JSON.parse(init.body) });
        return { ok: true, json: async () => ({ ok: true, wait }) }; });
    // The child's wait is one owner-action System message in Main's feed.
    const feed = document.createElement('div');
    env.mount.append(feed);
    const waiting = createMainWaiting({ page: env.mount, messagesDiv: feed, toolbar: null, stampNodeTimestamp,
        insertMessageNode: node => { feed.append(node); return true; },
        openSettings: page => { settings.push(page); return false; } });
    try {
        waiting.snapshot({ main_attention_complete: true, main_attention: [projected || { task_id: 'child', root_task_id: 'root',
            project_id: 'build', phase: 'working', task_attempt: 3, model_waits: { login: wait } }] });
        const message = find(feed, node => node.dataset.ownerAction === 'child');
        assert.ok(message?.classList.contains('owner-action'), 'the child task, not its root, owns the message');
        const login = find(message, node => Object.hasOwn(node.dataset, 'waitLogin'));
        assert.ok(login); login.click(); await turn(); assert.deepEqual(settings, ['providers']);
        const auto = find(message, node => Object.hasOwn(node.dataset, 'waitAuto'));
        auto.checked = false;
        for (const handler of auto.listeners.get('change')) await handler({ target: auto });
        assert.equal(calls[0].body.decision_id, 'model_wait:child:login');
        assert.equal(calls[0].body.revision, 3); assert.equal(calls[0].body.auto_continue, false);
    } finally { waiting.destroy(); restoreDom(env.prior); }
});

test('native Project nonclean final keeps one answer and limitation live and after actual history assembly', async t => {
    if (!process.env.MAIN_REVIEW_HISTORY_PAYLOAD) return t.skip('The paired pytest generates the production history payload.');
    const rows = JSON.parse(readFileSync(process.env.MAIN_REVIEW_HISTORY_PAYLOAD, 'utf8')).messages;
    const completion = rows.find(row => row.system_type === 'project_completion_summary');
    assert.ok(completion.completion_reason.includes('The Windows verification was not performed.'));
    nestedQueries(t);
    const originalAfter = ElementStub.prototype.after;
    ElementStub.prototype.after = function(node) { return this.insertAdjacentElement('afterend', node); };
    t.after(() => { if (originalAfter) ElementStub.prototype.after = originalAfter; else delete ElementStub.prototype.after; });
    for (const live of [true, false]) {
        const env = installDom(async url => ({ ok: true, json: async () => String(url).startsWith('/api/chat/history')
            ? page(rows) : { active_direct_turns: [] } }));
        const handlers = new Map();
        const instance = createChatInstance({ ws: { on(type, fn) { handlers.set(type, fn); return () => handlers.delete(type); },
            isConnected: () => true, send() {} }, state: { activePage: 'chat', projectChatIds: new Set(), unreadCount: 0 },
            updateUnreadBadge() {}, chatId: 1, idPrefix: 'chat', mountEl: env.mount,
            stateSnapshots: { begin: () => ({}), gate() { return Promise.resolve({}); }, isCurrent: () => true, apply() {} } });
        try {
            if (live) handlers.get('chat')({ ...completion, history_id: undefined, history_position: undefined, chat_id: 1, content: completion.text });
            else await instance.refreshHistory({ revision: 1 });
            const messages = document.byId.get('chat-messages');
            const answers = all(messages, node => node.dataset.systemType === 'project_completion_summary');
            assert.equal(answers.length, 1);
            assert.ok(answers[0].innerHTML.includes(completion.completion_answer));
            const limits = all(messages, node => node.classList.contains('project-answer-limitation'));
            assert.equal(limits.length, 1); assert.equal(limits[0].textContent, completion.completion_reason);
            assert.equal(Boolean(walkCard(messages, completion.task_id)), false, `${live ? 'live' : 'reload'}: no duplicate Failed card`);
        } finally { instance.destroy(); restoreDom(env.prior); }
    }
});
