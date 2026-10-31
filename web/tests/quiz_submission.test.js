import assert from 'node:assert/strict';
import test, { mock } from 'node:test';
import { fixture as baseFixture, turn, WS_MSG } from './chat_decision_fixture.js';

const fixture = options => baseFixture({ isMain: true, ...options });
const store = () => {
    const rows = new Map();
    return { rows, getItem: key => rows.get(key) || null, setItem: (key, value) => rows.set(key, value), removeItem: key => rows.delete(key) };
};
const detail = (block = WS_MSG) => ({ task_id: 't-1', chat_id: 1, owner_quiz: { 'qz-1': block } });
const choose = card => card.querySelectorAll('.chat-quiz-option')[1].click();
const field = card => card.querySelector('.chat-quiz-comment');
const edit = (card, text) => { field(card).value = text; field(card).listeners.get('input')(); };
const sent = fx => fx.calls.map(call => JSON.parse(call.init.body));

test('one press paints an expanded, neutral pending selection before transport answers and retains focus', async () => {
    let finish;
    const fx = fixture({ fetchImpl: () => new Promise(resolve => { finish = resolve; }) });
    try {
        const card = fx.decision.buildQuizCard(WS_MSG);
        const button = card.querySelectorAll('.chat-quiz-option')[1]; button.focus();
        edit(card, '  literal\nanswer  '); choose(card);
        assert.equal(card.dataset.state, 'open');
        assert.equal(card.dataset.submission, 'sending');
        assert.equal(card.querySelector('.chat-quiz-status-text').textContent, 'Sending…');
        assert.ok(button.classList.contains('pending-choice'));
        assert.equal(button.classList.contains('chosen'), false);
        assert.equal(button.disabled, false);
        assert.equal(button.getAttribute('aria-disabled'), 'true');
        assert.equal(globalThis.document.activeElement, button);
        assert.equal(field(card).readOnly, true);
        assert.ok(card.querySelector('.chat-quiz-question'));
        choose(card); card.querySelectorAll('.chat-quiz-option')[0].click();
        assert.equal(fx.calls.length, 1);
        finish({ ok: true, status: 200, json: async () => ({ ok: true, state: 'answered', answered_index: 1, comment: '  literal\nanswer  ' }) });
        await turn();
        assert.equal(card.dataset.state, 'answered');
        assert.ok(button.classList.contains('chosen'));
        assert.equal(field(card), null);
        assert.equal(card.dataset.ownerComment, '  literal\nanswer  ');
        assert.equal(globalThis.document.activeElement, card.querySelector('.chat-quiz-question'));
    } finally { fx.restore(); }
});

test('server-known answered + owed delivery 503 is a recorded answer, not a failed save', async () => {
    const fx = fixture({ fetchImpl: async () => ({ ok: false, status: 503, json: async () => ({
        state: 'answered', answered_index: 0, comment: 'winner',
        question_delivery: { state: 'owed', kind: 'quiz_answer', reason_code: 'delivery_pending' },
    }) }) });
    try {
        const card = fx.decision.buildQuizCard(WS_MSG); choose(card); await turn();
        assert.equal(card.dataset.state, 'answered');
        assert.equal(card.dataset.ownerComment, 'winner');
        assert.match(card.querySelector('.chat-quiz-status-text').textContent, /recorded.*pending/i);
        assert.equal(fx.toasts.some(toast => /Could not record/.test(toast.text)), false);
    } finally { fx.restore(); }
});

test('positive WS beats a later HTTP error and no recovery can overwrite the winner', async () => {
    let finish;
    const fx = fixture({ fetchImpl: () => new Promise(resolve => { finish = resolve; }) });
    try {
        const card = fx.decision.buildQuizCard(WS_MSG); choose(card);
        fx.decision.applyQuizStateFrame({}, { task_id: 't-1', quiz_id: 'qz-1', state: 'answered', answered_index: 0, comment: 'other device' });
        finish({ ok: false, status: 500, json: async () => ({}) }); await turn();
        assert.equal(card.dataset.state, 'answered');
        assert.equal(card.dataset.ownerComment, 'other device');
        assert.equal(fx.toasts.length, 0);
    } finally { fx.restore(); }
});

test('verbatim unsent draft survives full instance destruction and is keyed by task plus quiz', () => {
    const storage = store();
    let fx = fixture({ storage });
    let card = fx.decision.buildQuizCard(WS_MSG); edit(card, '  preserved\nwords  '); fx.restore();
    fx = fixture({ storage });
    try {
        card = fx.decision.buildQuizCard(WS_MSG);
        assert.equal(field(card).value, '  preserved\nwords  ');
        assert.equal(fx.calls.length, 0);
        const foreign = fx.decision.buildQuizCard({ ...WS_MSG, task_id: 'different' });
        assert.equal(field(foreign).value, '');
        // No separate control clears a draft (owner 19.10, A): emptying the field releases it.
        assert.equal(card.querySelector('.chat-quiz-clear'), null);
        edit(card, '');
        assert.equal(field(card).value, ''); assert.equal(storage.rows.size, 0);
    } finally { fx.restore(); }
});

test('emptying the unsent words releases a refusal about them, as the retired Clear did', async () => {
    const storage = store();
    const fx = fixture({ storage, fetchImpl: async () => ({ ok: false, status: 422, json: async () => ({ error: 'refused' }) }) });
    try {
        const card = fx.decision.buildQuizCard(WS_MSG);
        const note = card.querySelector('.chat-quiz-submission-note');
        edit(card, 'my words'); choose(card); await turn();
        assert.match(note.textContent, /Answer not accepted/);
        assert.equal(field(card).value, 'my words', 'a refused answer keeps its draft');
        edit(card, 'my words, edited');
        assert.match(note.textContent, /Answer not accepted/, 'an edit keeps the refusal beside the words it describes');
        edit(card, '');
        assert.equal(note.textContent, '');
        assert.equal(card.querySelector('.chat-quiz-feedback').hidden, true, 'nothing left to say takes no line');
        assert.equal(storage.rows.size, 0);
    } finally { fx.restore(); }
});

test('typing never clears a frozen attempt: an unconfirmed answer keeps its words, note and recovery', async () => {
    mock.timers.enable({ apis: ['setTimeout'] });
    const fx = fixture({ fetchImpl: () => new Promise(() => {}), fetchDetail: async () => detail() });
    try {
        const card = fx.decision.buildQuizCard(WS_MSG);
        edit(card, 'frozen words'); choose(card);
        mock.timers.tick(15001); await turn(); await turn();
        assert.equal(card.dataset.submission, 'unknown');
        const note = card.querySelector('.chat-quiz-submission-note').textContent;
        assert.ok(note);
        edit(card, '');
        assert.equal(field(card).value, 'frozen words');
        assert.deepEqual([card.dataset.submission, card.querySelector('.chat-quiz-submission-note').textContent], ['unknown', note]);
        assert.equal(card.querySelector('.chat-quiz-forget').hidden, false);
    } finally { fx.restore(); mock.timers.reset(); }
});

test('the feedback follows the answer it describes: Forget first and quiet, the server actions last', async () => {
    const fx = fixture({ fetchImpl: async () => { throw new Error('lost'); }, fetchDetail: async () => detail() });
    try {
        const card = fx.decision.buildQuizCard(WS_MSG);
        const box = card.querySelector('.chat-quiz-comment-box');
        const feedback = card.querySelector('.chat-quiz-feedback');
        // One frame holds the words and Send; the field is named and grows through the composer's sizing.
        assert.deepEqual(box.children.map((node) => node.className), ['chat-quiz-comment', 'chat-quiz-send ui-hit-area']);
        assert.equal(field(card).getAttribute('aria-label'), 'Your answer');
        assert.ok(['content', 'fit'].includes(field(card).dataset.sizing));
        assert.equal(box.nextElementSibling, feedback);
        assert.equal(feedback.hidden, true, 'nothing to say takes no line');
        assert.deepEqual(feedback.children.map((node) => node.className), ['chat-quiz-submission-note', 'chat-quiz-storage-note',
            'chat-quiz-forget btn btn-xs btn-ghost', 'chat-quiz-check btn btn-xs btn-default', 'chat-quiz-retry btn btn-xs btn-default']);
        choose(card); await turn(); await turn();
        assert.equal(card.dataset.submission, 'unknown');
        assert.equal(feedback.hidden, false);
        assert.deepEqual(['forget', 'check', 'retry'].map((name) => card.querySelector(`.chat-quiz-${name}`).hidden), [false, false, false]);
    } finally { fx.restore(); }
});

test('unconfirmed attempt survives reload, reads before explicit retry, and replays identical payload despite attempted edits', async () => {
    const storage = store();
    let fx = fixture({ storage, fetchImpl: async () => { throw new Error('lost response'); }, fetchDetail: async () => detail() });
    let card = fx.decision.buildQuizCard(WS_MSG); edit(card, '  first\nwords  '); choose(card); await turn();
    const original = sent(fx)[0]; fx.restore();
    let reads = 0;
    fx = fixture({ storage, fetchDetail: async () => { reads++; return detail(); } });
    try {
        card = fx.decision.buildQuizCard(WS_MSG); await turn();
        assert.equal(fx.calls.length, 0, 'reload only reads; never resends'); assert.equal(reads, 1);
        assert.equal(field(card).value, '  first\nwords  ');
        edit(card, 'replacement must not be sent');
        card.querySelectorAll('.chat-quiz-option')[0].click(); await turn();
        assert.equal(fx.calls.length, 0);
        assert.equal(field(card).value, '  first\nwords  ');
        card.querySelector('.chat-quiz-retry').click(); await turn();
        assert.equal(reads, 2, 'the retry gets a new canonical read');
        assert.deepEqual(sent(fx)[0], original);
        assert.equal(storage.rows.size, 0);
    } finally { fx.restore(); }
});

test('a fresh recovery read finding a winning answer or closure suppresses the retry', async () => {
    for (const state of ['answered', 'closed', 'superseded']) {
        let winner = false;
        const fx = fixture({ fetchImpl: async () => { throw new Error('unknown'); },
            fetchDetail: async () => detail(winner ? { ...WS_MSG, state, answered_index: 0, comment: 'saved elsewhere', closed_reason: 'not needed' } : WS_MSG) });
        try {
            const card = fx.decision.buildQuizCard(WS_MSG); choose(card); await turn(); winner = true;
            card.querySelector('.chat-quiz-retry').click(); await turn();
            assert.equal(fx.calls.length, 1); assert.equal(card.dataset.state, state);
        } finally { fx.restore(); }
    }
});

test('failed or wrong-room canonical read cannot authorize a retry', async () => {
    const fx = fixture({ fetchImpl: async () => ({ ok: false, status: 500 }),
        fetchDetail: async () => ({ ...detail(), chat_id: 23 }) });
    try {
        const card = fx.decision.buildQuizCard(WS_MSG); choose(card); await turn();
        card.querySelector('.chat-quiz-retry').click(); await turn();
        assert.equal(fx.calls.length, 1); assert.equal(card.dataset.submission, 'unknown');
        assert.match(card.querySelector('.chat-quiz-submission-note').textContent, /could not be confirmed/);
    } finally { fx.restore(); }
});

test('local storage failure never blocks immediate feedback or overrides a later server answer', async () => {
    const storage = { getItem() { throw new Error('denied'); }, setItem() { throw new Error('full'); }, removeItem() { throw new Error('denied'); } };
    let finish;
    const fx = fixture({ storage, fetchImpl: () => new Promise(resolve => { finish = resolve; }) });
    try {
        const card = fx.decision.buildQuizCard(WS_MSG); choose(card);
        assert.equal(card.dataset.submission, 'sending');
        assert.match(card.querySelector('.chat-quiz-storage-note').textContent, /Local saving is unavailable/);
        finish({ ok: true, status: 200, json: async () => ({ ok: true, state: 'answered', answered_index: 1 }) }); await turn();
        assert.equal(card.dataset.state, 'answered'); assert.equal(card.querySelector('.chat-quiz-storage-note').textContent, '');
    } finally { fx.restore(); }
});

test('a hung POST becomes unknown, may reconcile, and destroy releases its transport without automatic resend', async () => {
    mock.timers.enable({ apis: ['setTimeout'] });
    const fx = fixture({ fetchImpl: () => new Promise(() => {}), fetchDetail: async () => detail() });
    try {
        const card = fx.decision.buildQuizCard(WS_MSG); choose(card);
        mock.timers.tick(15001); await turn();
        assert.equal(card.dataset.submission, 'unknown'); assert.equal(fx.calls.length, 1);
        assert.equal(card.querySelector('.chat-quiz-retry').disabled, false);
        fx.decision.destroy(); assert.equal(fx.calls[0].init.signal.aborted, true);
    } finally { fx.restore(); mock.timers.reset(); }
});

test('two mounted Main/Project forms sync drafts without GETs; one shared pending attempt gates both', async () => {
    const storage = store(), events = new EventTarget();
    let reads = 0, finish, winner = false;
    const read = async () => { reads++; return { task_id: 't-1', project_id: 'p1', owner_quiz: { 'qz-1': {
        ...WS_MSG, ...(winner ? { state: 'answered', answered_index: 1, comment: 'shared draft' } : {}),
    } } }; };
    const a = baseFixture({ storage, events, isMain: true, fetchDetail: read,
        fetchImpl: () => new Promise(resolve => { finish = resolve; }) });
    const main = a.decision.buildQuestionPointer({ ...WS_MSG, quiz_state: 'open', project_id: 'p1', project_chat_id: 23 });
    const b = baseFixture({ storage, events, projectId: 'p1', chatId: 23, fetchDetail: read });
    try {
        const project = b.decision.buildQuizCard(WS_MSG);
        edit(main, 'shared draft'); await turn();
        assert.equal(field(project).value, 'shared draft'); assert.equal(reads, 0, 'typing never GETs');
        choose(main); choose(project); await turn();
        assert.equal(a.calls.length, 1); assert.equal(b.calls.length, 0);
        assert.equal(project.dataset.submission, 'sending');
        assert.equal(field(project).readOnly, true);
        winner = true;
        finish({ ok: true, status: 200, json: async () => ({ ok: true, state: 'answered', answered_index: 1, comment: 'shared draft' }) });
        await turn(); await turn();
        assert.equal(main.dataset.state, 'answered'); assert.equal(project.dataset.state, 'answered');
        assert.equal(reads, 1, 'only learning settlement asks the other view to reconcile');
    } finally { b.restore(); a.restore(); }
});

test('Project recovery requires its existing project binding; foreign and unknown room proofs never authorize POST', async () => {
    for (const room of [{ state: 'known', project_id: 'foreign', chat_id: 99 }, { state: 'unknown' }]) {
        const storage = store();
        let fx = baseFixture({ storage, projectId: 'p1', chatId: 23, fetchImpl: async () => { throw new Error('lost'); } });
        let card = fx.decision.buildQuizCard(WS_MSG); choose(card); await turn(); fx.restore();
        fx = baseFixture({ storage, projectId: 'p1', chatId: 23, rawDetail: true, fetchDetail: async () => ({
            task_id: 't-1', question_rooms: { 'qz-1': room }, owner_quiz: { 'qz-1': WS_MSG },
        }) });
        try {
            card = fx.decision.buildQuizCard(WS_MSG); await turn();
            card.querySelector('.chat-quiz-retry').click(); await turn();
            assert.equal(fx.calls.length, 0); assert.equal(card.dataset.submission, 'unknown');
        } finally { fx.restore(); }
    }
});

test('a late negative response of an older attempt cannot clear a different attempt learned from another view', async () => {
    const events = new EventTarget(), storage = store();
    let finish;
    const fx = fixture({ events, storage, fetchImpl: () => new Promise(resolve => { finish = resolve; }) });
    try {
        const card = fx.decision.buildQuizCard(WS_MSG); choose(card);
        const first = sent(fx)[0];
        const replacement = { v: 1, draft: 'newer attempt', reason: '', attempt: {
            ...first, request_id: 'another-owner-attempt', comment: 'newer attempt', option_index: 0,
        } };
        events.dispatchEvent(new CustomEvent('ouro:quiz-reply', { detail: {
            key: `ouro_quiz_reply_v1:${JSON.stringify(['t-1', 'qz-1'])}`, record: replacement, phase: 'unknown',
        } }));
        finish({ ok: false, status: 422, json: async () => ({ error: 'old refusal' }) }); await turn();
        assert.equal(card.dataset.submission, 'unknown'); assert.equal(field(card).value, 'newer attempt');
        assert.equal(fx.toasts.length, 0); assert.equal(fx.calls.length, 1);
    } finally { fx.restore(); }
});
