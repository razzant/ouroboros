import assert from 'node:assert/strict';
import test, { mock } from 'node:test';
import { fixture, turn, WS_MSG, NodeStub } from './chat_decision_fixture.js';

const key = `ouro_quiz_reply_v1:${JSON.stringify(['t-1', 'qz-1'])}:attempt`;
const store = () => { const rows = new Map(); return { rows, getItem: k => rows.get(k) || null,
    setItem: (k, v) => rows.set(k, v), removeItem: k => rows.delete(k) }; };
const detail = async () => ({ task_id: 't-1', chat_id: 1, owner_quiz: { 'qz-1': WS_MSG } });
const choose = card => card.querySelectorAll('.chat-quiz-option')[0].click();
const refusal = () => ({ ok: false, status: 401, json: async () => ({ error: 'auth' }) });
const answer = () => ({ ok: true, status: 200, json: async () => ({ ok: true, state: 'answered', answered_index: 0, comment: 'winner' }) });
const retry = async card => { card.querySelector('.chat-quiz-retry').click(); await turn(); };
const held = (card, storage, body) => {
    assert.deepEqual(JSON.parse(storage.getItem(key) || '{}').attempt, JSON.parse(body));
    assert.equal(card.querySelector('.chat-quiz-comment').readOnly, true);
};

for (const refused of [0, 1]) test(`physical refusal ${refused + 1} cannot retire two unresolved sends of the same answer`, async () => {
    mock.timers.enable({ apis: ['setTimeout'] });
    const storage = store(), sends = [];
    const fx = fixture({ storage, isMain: true, fetchDetail: detail,
        fetchImpl: () => new Promise(resolve => sends.push(resolve)) });
    try {
        const card = fx.decision.buildQuizCard(WS_MSG); choose(card);
        const body = fx.calls[0].init.body;
        mock.timers.tick(15001); await turn();
        assert.equal(card.dataset.submission, 'unknown');
        await retry(card); assert.equal(fx.calls.length, 2);
        assert.equal(fx.calls[1].init.body, body, 'retry keeps exact request identity and words');
        sends[refused](refusal()); await turn();
        held(card, storage, body);
        if (refused === 0) assert.equal(card.dataset.submission, 'sending', 'old refusal cannot relabel active retry');
        choose(card); assert.equal(fx.calls.length, 2, 'refusal cannot permit a changed new answer');
        sends[1 - refused](answer()); await turn();
        assert.equal(card.dataset.state, 'answered'); assert.equal(storage.rows.size, 0);
    } finally { fx.restore(); mock.timers.reset(); }
});

test('a first definite refusal still unlocks editing and a new deliberate answer', async () => {
    const storage = store();
    const fx = fixture({ storage, fetchImpl: async () => refusal() });
    try {
        const card = fx.decision.buildQuizCard(WS_MSG); choose(card); await turn();
        assert.equal(storage.getItem(key), null);
        assert.equal(card.querySelector('.chat-quiz-comment').readOnly, false);
        choose(card); await turn(); assert.equal(fx.calls.length, 2);
    } finally { fx.restore(); }
});

test('reload preserves old send uncertainty even when the new explicit retry is refused', async () => {
    const storage = store();
    let fx = fixture({ storage, isMain: true, fetchDetail: detail,
        fetchImpl: async () => { throw Error('unknown old response'); } });
    let card = fx.decision.buildQuizCard(WS_MSG); choose(card); await turn();
    const body = fx.calls[0].init.body; fx.restore();
    fx = fixture({ storage, isMain: true, fetchDetail: detail, fetchImpl: async () => refusal() });
    try {
        card = fx.decision.buildQuizCard(WS_MSG); await turn();
        await retry(card); held(card, storage, body);
        assert.equal(fx.calls.length, 1); assert.equal(card.dataset.submission, 'unknown');
    } finally { fx.restore(); }
});

test('a sibling retry is protected before its storage event reaches the old sender', async () => {
    mock.timers.enable({ apis: ['setTimeout'] });
    const storage = store(); let oldSend;
    const a = fixture({ storage, isMain: true, fetchDetail: detail,
        fetchImpl: () => new Promise(resolve => { oldSend = resolve; }) });
    const first = a.decision.buildQuizCard(WS_MSG); choose(first);
    const body = a.calls[0].init.body;
    mock.timers.tick(15001); await turn();
    const b = fixture({ storage, isMain: true, fetchDetail: detail, fetchImpl: () => new Promise(() => {}) });
    try {
        const sibling = b.decision.buildQuizCard(WS_MSG); await turn(); await retry(sibling);
        assert.equal(b.calls.length, 1);
        oldSend(refusal()); await turn();
        held(first, storage, body); held(sibling, storage, body);
    } finally { b.restore(); a.restore(); mock.timers.reset(); }
});

test('adopting a stored submission during an edit resets stale refusal phase and exposes recovery', async () => {
    const storage = store();
    const a = fixture({ storage, isMain: true, fetchDetail: detail, fetchImpl: async () => refusal() });
    const stale = a.decision.buildQuizCard(WS_MSG); choose(stale); await turn();
    assert.match(stale.querySelector('.chat-quiz-submission-note').textContent, /not accepted/);
    const b = fixture({ storage, isMain: true, fetchDetail: detail, fetchImpl: () => new Promise(() => {}) });
    try {
        const sender = b.decision.buildQuizCard(WS_MSG); choose(sender);
        const field = stale.querySelector('.chat-quiz-comment'); field.value = 'stale typing'; field.listeners.get('input')();
        await turn();
        assert.equal(stale.dataset.submission, 'unknown');
        assert.equal(stale.querySelector('.chat-quiz-retry').hidden, false);
        assert.doesNotMatch(stale.querySelector('.chat-quiz-submission-note').textContent, /not accepted/);
        assert.equal(field.readOnly, true);
        stale.querySelector('.chat-quiz-close').click(); assert.equal(a.calls.length, 1);
    } finally { b.restore(); a.restore(); }
});

test('a sibling clearing the attempt also clears its obsolete retry note', async () => {
    const storage = store(), events = new EventTarget();
    const fx = fixture({ storage, events, isMain: true, fetchDetail: detail,
        fetchImpl: async () => { throw Error('unknown'); } });
    try {
        const card = fx.decision.buildQuizCard(WS_MSG); choose(card); await turn();
        assert.match(card.querySelector('.chat-quiz-submission-note').textContent, /retry the same answer/);
        storage.rows.clear();
        events.dispatchEvent(new CustomEvent('ouro:quiz-reply', { detail: {
            key: key.slice(0, -8), record: { v: 1, draft: '', reason: '', attempt: null }, phase: '',
        } }));
        await turn();
        assert.equal(card.querySelector('.chat-quiz-comment').readOnly, false);
        assert.equal(card.querySelector('.chat-quiz-submission-note').textContent, '');
    } finally { fx.restore(); }
});

// The shared fixture omits HTML parsing. Only supply the dialog container/focus
// surface here; its actual asynchronous confirmation handler remains production.
function dialogSurface() {
    const doc = globalThis.document, create = doc.createElement;
    doc.body = new NodeStub('body'); doc.body.appendChild = node => doc.body.append(node);
    doc.addEventListener = doc.removeEventListener = () => {};
    doc.createElement = tag => {
        const backdrop = create(tag), dialog = new NodeStub();
        dialog.ownerDocument = doc; dialog.removeEventListener = () => {};
        dialog.querySelectorAll = () => [];
        backdrop.append(dialog);
        backdrop.querySelector = selector => selector === '[role="dialog"]' ? dialog : null;
        return backdrop;
    };
    return () => doc.body.children[0].click({ target: { closest: selector => selector === '[data-confirm-ok]' ? {} : null } });
}

for (const replaced of [false, true]) test(`Forget confirmation ${replaced ? 'adopts the newer shared answer' : 'deletes the unchanged selected answer'}`, async () => {
    const storage = store();
    let reads = 0;
    const fx = fixture({ storage, isMain: true, fetchDetail: async () => { reads++; return detail(); },
        fetchImpl: async () => { throw Error('unknown'); } });
    try {
        const card = fx.decision.buildQuizCard(WS_MSG); choose(card); await turn();
        const beforeReads = reads;
        const confirm = dialogSurface();
        card.querySelector('.chat-quiz-forget').click();
        const replacement = { v: 1, draft: 'other window words', reason: '', attempt: {
            request_id: 'new-attempt', decision_id: 'quiz:t-1:qz-1', option_index: 1, comment: 'other window words',
        } };
        if (replaced) storage.setItem(key, JSON.stringify(replacement)); // storage event has not arrived
        confirm(); await turn();
        assert.equal(reads, beforeReads, 'Forget revalidation is a local exact-question read, not network I/O');
        if (replaced) {
            assert.deepEqual(JSON.parse(storage.getItem(key)), replacement);
            assert.equal(card.querySelector('.chat-quiz-comment').value, 'other window words');
            assert.equal(card.querySelector('.chat-quiz-comment').readOnly, true);
            assert.equal(card.dataset.submission, 'unknown');
        } else {
            assert.equal(storage.getItem(key), null);
            assert.equal(card.querySelector('.chat-quiz-comment').readOnly, false);
            assert.equal(card.querySelector('.chat-quiz-submission-note').textContent, '');
        }
    } finally { fx.restore(); }
});

test('failed persistence with working empty reads retains the in-memory press and still sends', async () => {
    const storage = { getItem: () => null, setItem: () => { throw Error('QuotaExceededError'); }, removeItem: () => {} };
    let finish;
    const fx = fixture({ storage, fetchImpl: () => new Promise(resolve => { finish = resolve; }) });
    try {
        const card = fx.decision.buildQuizCard(WS_MSG); choose(card);
        assert.equal(fx.calls.length, 1);
        assert.equal(card.dataset.submission, 'sending');
        assert.equal(card.querySelector('.chat-quiz-comment').readOnly, true);
        assert.match(card.querySelector('.chat-quiz-storage-note').textContent, /Local saving is unavailable/);
        finish(answer()); await turn(); assert.equal(card.dataset.state, 'answered');
    } finally { fx.restore(); }
});

test('retry dispatch metadata write failure cannot let a refusal erase the persisted unknown', async () => {
    mock.timers.enable({ apis: ['setTimeout'] });
    const storage = store(), sends = [];
    const fx = fixture({ storage, isMain: true, fetchDetail: detail,
        fetchImpl: () => new Promise(resolve => sends.push(resolve)) });
    try {
        const card = fx.decision.buildQuizCard(WS_MSG); choose(card);
        const body = fx.calls[0].init.body;
        mock.timers.tick(15001); await turn();
        storage.setItem = () => { throw Error('QuotaExceededError'); };
        await retry(card); assert.equal(fx.calls.length, 2);
        sends[0](refusal()); sends[1](refusal()); await turn();
        held(card, storage, body); assert.equal(card.dataset.submission, 'unknown');
    } finally { fx.restore(); mock.timers.reset(); }
});

const storageEvent = (events, changedKey = key) => {
    const event = new Event('storage'); Object.defineProperty(event, 'key', { value: changedKey });
    events.dispatchEvent(event);
};

for (const draftOnly of [false, true]) test(`failed-write memory attempt survives sibling ${draftOnly ? 'draft-only' : 'empty'} storage, then canonical settlement`, async () => {
    const storage = store(), events = new EventTarget(); let finish;
    storage.setItem = () => { throw Error('QuotaExceededError'); };
    const fx = fixture({ storage, events, isMain: true, fetchDetail: detail,
        fetchImpl: () => new Promise(resolve => { finish = resolve; }) });
    try {
        const card = fx.decision.buildQuizCard(WS_MSG), field = card.querySelector('.chat-quiz-comment');
        field.value = '  original\nwords  '; field.listeners.get('input')(); choose(card);
        const body = fx.calls[0].init.body;
        if (draftOnly) storage.rows.set(key.slice(0, -8), JSON.stringify({ v: 1, draft: 'sibling draft', reason: '', attempt: null }));
        storageEvent(events, key.slice(0, -8)); await turn();
        assert.equal(field.readOnly, true, 'a storage event cannot erase a press whose persistence failed');
        assert.equal(field.value, '  original\nwords  ');
        assert.match(card.querySelector('.chat-quiz-storage-note').textContent, /Local saving is unavailable/);
        card.querySelectorAll('.chat-quiz-option')[1].click();
        assert.equal(fx.calls.length, 1, 'no changed answer or automatic resend');
        fx.decision.applyQuizStateFrame({}, { task_id: 't-1', quiz_id: 'qz-1', state: 'answered', answered_index: 0, comment: 'canonical winner' });
        finish(refusal()); await turn();
        assert.equal(card.dataset.state, 'answered'); assert.equal(card.dataset.ownerComment, 'canonical winner');
        assert.equal(card.dataset.submission, ''); assert.equal(JSON.parse(body).comment, '  original\nwords  ');
    } finally { fx.restore(); }
});

test('sibling failed retry metadata survives old sole refusal removing the shared record', async () => {
    mock.timers.enable({ apis: ['setTimeout'] });
    const storage = store(), events = new EventTarget(); let oldSend, retrySend;
    const a = fixture({ storage, isMain: true, fetchDetail: detail,
        fetchImpl: () => new Promise(resolve => { oldSend = resolve; }) });
    const first = a.decision.buildQuizCard(WS_MSG); choose(first);
    const body = a.calls[0].init.body, oldWindow = globalThis.window;
    mock.timers.tick(15001); await turn();
    const b = fixture({ storage, events, isMain: true, fetchDetail: detail,
        fetchImpl: () => new Promise(resolve => { retrySend = resolve; }) });
    try {
        const sibling = b.decision.buildQuizCard(WS_MSG); await turn();
        const setItem = storage.setItem;
        storage.setItem = () => { throw Error('QuotaExceededError'); };
        await retry(sibling); assert.equal(b.calls[0].init.body, body);
        storage.setItem = setItem; // the old sender can retire only its persisted sole-send record
        // The fixture shares globals, whereas real windows have separate local events.
        globalThis.window = oldWindow;
        oldSend(refusal()); await turn(); assert.equal(storage.getItem(key), null);
        globalThis.window = events;
        storageEvent(events); await turn();
        assert.equal(sibling.querySelector('.chat-quiz-comment').readOnly, true);
        assert.equal(sibling.dataset.submission, 'sending');
        assert.match(sibling.querySelector('.chat-quiz-storage-note').textContent, /Local saving is unavailable/);
        choose(sibling); assert.equal(b.calls.length, 1);
        retrySend(answer()); await turn(); assert.equal(sibling.dataset.state, 'answered');
    } finally { b.restore(); a.restore(); mock.timers.reset(); }
});

test('explicit Forget still releases a memory-only unconfirmed attempt after an empty storage event', async () => {
    const storage = store(), events = new EventTarget();
    storage.setItem = () => { throw Error('QuotaExceededError'); };
    const fx = fixture({ storage, events, isMain: true, fetchDetail: detail,
        fetchImpl: async () => { throw Error('unknown'); } });
    try {
        const card = fx.decision.buildQuizCard(WS_MSG); choose(card); await turn();
        storageEvent(events); await turn();
        assert.equal(card.querySelector('.chat-quiz-forget').hidden, false);
        const confirm = dialogSurface(); card.querySelector('.chat-quiz-forget').click(); confirm(); await turn();
        assert.equal(card.querySelector('.chat-quiz-comment').readOnly, false);
        choose(card); await turn(); assert.equal(fx.calls.length, 2, 'only the new explicit press sends again');
    } finally { fx.restore(); }
});

test('a positively stored replacement still replaces a failed-write memory-only attempt', async () => {
    const storage = store(), events = new EventTarget();
    storage.setItem = () => { throw Error('QuotaExceededError'); };
    const fx = fixture({ storage, events, isMain: true, fetchDetail: detail,
        fetchImpl: async () => { throw Error('unknown'); } });
    try {
        const card = fx.decision.buildQuizCard(WS_MSG); choose(card); await turn();
        storage.rows.set(key, JSON.stringify({ v: 1, draft: 'replacement words', reason: '', attempt: {
            request_id: 'replacement', decision_id: 'quiz:t-1:qz-1', option_index: 1, comment: 'replacement words',
        } }));
        storageEvent(events); await turn();
        assert.equal(card.querySelector('.chat-quiz-comment').value, 'replacement words');
        assert.equal(card.querySelector('.chat-quiz-comment').readOnly, true);
        assert.equal(card.querySelectorAll('.chat-quiz-option')[1].classList.contains('pending-choice'), true);
        assert.equal(card.querySelector('.chat-quiz-storage-note').textContent, '');
        assert.equal(fx.calls.length, 1, 'adoption only reads the canonical question');
    } finally { fx.restore(); }
});
