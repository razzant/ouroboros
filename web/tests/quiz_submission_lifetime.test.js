import assert from 'node:assert/strict';
import test from 'node:test';
import { fixture, turn, WS_MSG, NodeStub } from './chat_decision_fixture.js';
const key = `ouro_quiz_reply_v1:${JSON.stringify(['t-1', 'qz-1'])}`;
const store = () => { const rows = new Map(); return { rows, getItem: k => rows.get(k) || null,
    setItem: (k, v) => rows.set(k, v), removeItem: k => rows.delete(k) }; };
const edit = (card, text, selector = '.chat-quiz-comment') => {
    const field = card.querySelector(selector); field.value = text; field.listeners.get('input')();
};
const choose = card => card.querySelectorAll('.chat-quiz-option')[0].click();
const detail = (state = 'open') => ({ task_id: 't-1', chat_id: 1,
    owner_quiz: { 'qz-1': { ...WS_MSG, state } } });
const publication = (events, record, phase = '') => events.dispatchEvent(new CustomEvent('ouro:quiz-reply',
    { detail: { key, record, phase } }));

test('draft edit before the other window receives its storage event cannot erase submitted words', async () => {
    const storage = store(); let finish;
    const a = fixture({ storage, fetchImpl: () => new Promise(r => { finish = r; }) });
    const sender = a.decision.buildQuizCard(WS_MSG);
    const b = fixture({ storage });
    try {
        const stale = b.decision.buildQuizCard(WS_MSG);
        edit(sender, 'immutable words'); choose(sender);
        const body = JSON.parse(a.calls[0].init.body);
        edit(stale, 'typed before queued event');
        assert.equal(stale.querySelector('.chat-quiz-comment').value, 'immutable words');
        assert.deepEqual(JSON.parse(storage.getItem(key + ':attempt')).attempt, body);
        finish({ ok: true, status: 200, json: async () => ({ ok: true, state: 'answered', answered_index: 0, comment: body.comment }) });
        await turn();
        assert.equal(sender.dataset.state, 'answered'); assert.equal(storage.rows.size, 0);
    } finally { b.restore(); a.restore(); }
});

test('sender disposal publishes recoverable unknown to its mounted sibling, without a POST', async () => {
    const events = new EventTarget(), storage = store();
    const a = fixture({ events, storage, fetchImpl: () => new Promise(() => {}) });
    const sender = a.decision.buildQuizCard(WS_MSG);
    const b = fixture({ events, storage });
    try {
        const mirror = b.decision.buildQuizCard(WS_MSG); choose(sender);
        assert.equal(mirror.dataset.submission, 'sending');
        a.decision.destroy(); await turn();
        assert.equal(a.calls[0].init.signal.aborted, true);
        assert.equal(mirror.dataset.submission, 'unknown');
        assert.equal(mirror.querySelector('.chat-quiz-retry').disabled, false);
        assert.equal(mirror.querySelector('.chat-quiz-retry').hidden, false);
        assert.equal(b.calls.length, 0);
    } finally { b.restore(); a.restore(); }
});

test('a late open read begun for a cleared attempt cannot relabel a newer POST', async () => {
    const events = new EventTarget(), storage = store(), reads = []; let count = 0;
    const fx = fixture({ events, storage, isMain: true,
        fetchImpl: async () => { if (++count === 1) throw Error('unknown'); return new Promise(() => {}); },
        fetchDetail: () => new Promise(r => reads.push(r)) });
    try {
        const card = fx.decision.buildQuizCard(WS_MSG); choose(card); await turn();
        storage.rows.clear(); publication(events, { v: 1, draft: '', reason: '', attempt: null }); await turn();
        choose(card); assert.equal(fx.calls.length, 2); assert.equal(card.dataset.submission, 'sending');
        reads.forEach(r => r(detail())); await turn(); await turn();
        assert.equal(card.dataset.submission, 'sending');
        assert.equal(card.querySelector('.chat-quiz-retry').hidden, true);
    } finally { fx.restore(); }
});

test('remount releases the old complete-but-unknown view: no detached POST or duplicate read', async () => {
    const events = new EventTarget(); let reads = 0;
    const fx = fixture({ events, isMain: true, fetchDetail: async () => { reads++; return {
        task_id: 't-1', project_id: 'p1', owner_quiz: { 'qz-1': WS_MSG } }; } });
    try {
        const host = new NodeStub();
        const old = fx.decision.buildQuestionPointer({ ...WS_MSG, quiz_state: 'unknown', state: 'unknown', project_id: 'p1', project_chat_id: 23 }, false, true);
        host.append(old); await turn(); await turn();
        assert.notEqual(host.children[0], old);
        reads = 0;
        choose(old); assert.equal(fx.calls.length, 0, 'detached control has no active submission view');
        const record = { v: 1, draft: 'old attempt', reason: '', attempt: { request_id: 'unresolved', decision_id: 'quiz:t-1:qz-1', option_index: 0 } };
        publication(events, record, 'unknown'); publication(events, { ...record, attempt: null });
        await turn(); assert.equal(reads, 1);
    } finally { fx.restore(); }
});

test('close and answer share admission in both orders; refused close permits a later answer', async () => {
    let finish;
    const fx = fixture({ fetchImpl: () => new Promise(r => { finish = r; }) });
    try {
        const card = fx.decision.buildQuizCard(WS_MSG);
        card.querySelector('.chat-quiz-close').click(); choose(card);
        assert.equal(fx.calls.length, 1); assert.equal(JSON.parse(fx.calls[0].init.body).action, 'close');
        finish({ ok: false, status: 422, json: async () => ({}) }); await turn();
        choose(card); card.querySelector('.chat-quiz-close').click();
        assert.equal(card.querySelector('.chat-quiz-close').getAttribute('aria-disabled'), 'true');
        assert.equal(fx.calls.length, 2); assert.equal(JSON.parse(fx.calls[1].init.body).action, undefined);
    } finally { fx.restore(); }
});

test('partial unknown form neither validates nor exposes a valid stored attempt as a storage failure', async () => {
    const storage = store(); const record = { v: 1, draft: 'words', reason: '', attempt: {
        request_id: 'kept', decision_id: 'quiz:t-1:qz-1', option_index: 1 } };
    storage.setItem(key, JSON.stringify(record));
    const fx = fixture({ storage, isMain: true });
    try {
        const card = fx.decision.buildQuestionPointer({ task_id: 't-1', quiz_id: 'qz-1', project_id: 'p1', project_chat_id: 23, quiz_state: 'unknown' });
        assert.equal(card.querySelector('.chat-quiz-feedback'), null);
        assert.deepEqual(JSON.parse(storage.getItem(key)), record);
    } finally { fx.restore(); }
});

for (const source of ['restore', 'storage']) test(`quiet Close protects ${source} Project draft before a lifecycle refresh`, () => {
    const storage = store(), events = new EventTarget();
    const stored = draft => JSON.stringify({ v: 1, draft, reason: '', attempt: null });
    if (source === 'restore') storage.setItem(key, stored('  kept words\n'));
    const fx = fixture({ storage, events, projectId: 'p1', chatId: 23 });
    const storageChanged = () => {
        const event = new Event('storage'); Object.defineProperty(event, 'key', { value: key }); events.dispatchEvent(event);
    };
    try {
        const card = fx.decision.buildQuizCard(WS_MSG);
        if (source === 'storage') { storage.setItem(key, stored('  kept words\n')); storageChanged(); }
        assert.equal(card.querySelector('.chat-quiz-comment').value, '  kept words\n');
        assert.equal(card.querySelector('.chat-quiz-close').disabled, true);
        card.querySelector('.chat-quiz-close').click(); assert.equal(fx.calls.length, 0);
        storage.removeItem(key); storageChanged();
        assert.equal(card.querySelector('.chat-quiz-close').disabled, false);
        card.querySelector('.chat-quiz-close').click(); assert.equal(fx.calls.length, 1);
        assert.equal(JSON.parse(fx.calls[0].init.body).action, 'close');
    } finally { fx.restore(); }
});
