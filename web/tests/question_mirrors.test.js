// Project questions in Main (docs/DESIGN.md "Project question mirror"): every open, passed,
// finished or replaced Project question is the Project's own form in Main, plus one chip that
// opens it in its Project. The first confirmed answer from any source shows the result for five
// seconds and removes only the Main copy; an answered question never enters Main again.
import assert from 'node:assert/strict';
import test, { mock } from 'node:test';
import { NodeStub, fixture, turn } from './chat_decision_fixture.js';

// The rows the producer emits (ouroboros/project_dialogue.py::project_question_pointer): complete
// for display, `ts` from the question's own asked_at. One batch is asked in sequence.
const ASKED = ['2026-09-18T22:00:00+00:00', '2026-09-18T22:00:07+00:00'];
const ROW = { role: 'system', system_type: 'project_question_pointer', task_id: 't-1', quiz_id: 'qz-1',
    project_id: 'p1', project_chat_id: 23, project_name: 'Storage', ts: ASKED[0], quiz_state: 'open',
    question: 'Merge **now**?', options: ['Yes', 'No'], option_details: ['Ship today', 'Wait for CI'],
    stake: 'Release timing', recommended_index: 0 };
const WAITING = { ...ROW, wait_for_answer: true, owner_wait_state: 'waiting' };
const SETTLE_MS = 5000;
const text = (node, name) => node.querySelector(`.${name}`)?.textContent ?? null;
const options = (node) => node.querySelectorAll('.chat-quiz-option');
const labels = (node) => options(node).map((button) => text(button, 'chat-quiz-option-label'));

// A Main column: bubbles are framed and mounted like chat.js does, and retirement goes through
// the one removal seam the chat instance hands over.
function mainFixture(extra = {}) {
    const column = new NodeStub();
    const removed = [];
    const fx = fixture({ isMain: true,
        frameNode: (_msg, card) => { const bubble = new NodeStub(); bubble.append(card); return bubble; },
        insertMessageNode: (node) => { column.append(node); return true; },
        removeMessageNode: (node) => { removed.push(node); node.remove(); },
        ...extra });
    const cards = () => column.children.map((bubble) => bubble.children[0]);
    return { ...fx, column, removed, cards, add: (row) => fx.decision.appendQuestionPointer(row) };
}

function withTimers(run) {
    mock.timers.enable({ apis: ['setTimeout'] });
    return Promise.resolve().then(run).finally(() => mock.timers.reset());
}

test('a realistic burst: every unanswered question is the full Project form, an answered one never enters Main', () => {
    const fx = mainFixture();
    try {
        fx.add({ ...ROW, quiz_id: 'q-answered', quiz_state: 'answered', answered_index: 0 });
        fx.add({ ...WAITING, quiz_id: 'q-resumed', owner_wait_state: 'resumed' });
        fx.add({ ...ROW, quiz_id: 'q-optional', assumption: 'Yes meanwhile' });
        fx.add({ ...ROW, quiz_id: 'q-finished', quiz_state: 'expired_terminal', wait_for_answer: true });
        fx.add({ ...ROW, quiz_id: 'q-replaced', quiz_state: 'superseded' });
        fx.add({ ...WAITING, quiz_id: 'q-waiting', ts: ASKED[1] });
        const cards = fx.cards();
        assert.deepEqual(cards.map((card) => card.dataset.quizId), ['q-resumed', 'q-optional', 'q-finished', 'q-replaced', 'q-waiting']);
        assert.deepEqual(cards.map((card) => text(card, 'chat-quiz-status-text')), [
            'Unanswered · the task continued; an answer is still accepted',
            'Unanswered · an answer is still accepted',
            'Unanswered · the task finished; a late answer is accepted as your message',
            'Replaced by a newer question',
            'Waiting for your answer',
        ]);
        for (const card of cards) {
            assert.ok(card.classList.contains('chat-quiz-card') && card.classList.contains('project-question-card'));
            assert.equal(text(card, 'chat-quiz-chip'), 'Question');
            assert.equal(text(card, 'chat-live-project-name'), 'Storage');
            assert.equal(text(card, 'chat-quiz-question'), 'Merge **now**?');
            assert.equal(text(card, 'chat-quiz-stake'), 'At stake: Release timing');
            assert.deepEqual(labels(card), ['Yes', 'No']);
            assert.deepEqual(options(card).map((button) => text(button, 'chat-quiz-option-detail')), ['Ship today', 'Wait for CI']);
            assert.ok(options(card)[0].querySelector('.chat-quiz-option-recommended'), 'index zero is a recommendation too');
            assert.equal(card.querySelector('.chat-quiz-details-unavailable'), null);
        }
        // Answerable questions carry the own-answer field; the replaced one is a read-only record.
        assert.deepEqual(cards.map((card) => Boolean(card.querySelector('.chat-quiz-comment'))), [true, true, true, false, true]);
        assert.deepEqual(cards.map((card) => options(card)[0].disabled), [false, false, false, true, false]);
        assert.equal(text(cards[1], 'chat-quiz-assumption'), 'Continuing meanwhile: Yes meanwhile');
        assert.equal(text(cards[4], 'chat-quiz-assumption'), 'Waiting for your answer; Stop and the task deadline still apply.');
        assert.match(text(cards[0], 'chat-quiz-assumption'), /^The wait ended without an answer/);
    } finally { fx.restore(); }
});

test('the mirror is the Project form node for node; its one addition is the Project chip', () => {
    const shape = (node) => [node.className, node.textContent, node.disabled, node.children.map(shape)];
    const room = fixture();
    const main = mainFixture();
    try {
        main.add({ ...WAITING, assumption: 'Yes meanwhile' });
        const mirror = main.cards()[0];
        // The Project room replays the same ask from its own chat row (option objects, stake).
        const own = room.decision.buildQuizCard({ role: 'assistant', task_id: 't-1', ts: ASKED[0], text: 'Merge **now**?',
            msg_type: 'quiz', quiz: { quiz_id: 'qz-1', state: 'open', wait_for_answer: true, owner_wait_state: 'waiting',
                stake: 'Release timing', assumption: 'Yes meanwhile', options: [
                    { label: 'Yes', detail: 'Ship today', recommended: true }, { label: 'No', detail: 'Wait for CI' }] } });
        const [head, ...body] = mirror.children;
        const [ownHead, ...ownBody] = own.children;
        assert.deepEqual(body.map(shape), ownBody.map(shape));
        assert.deepEqual(head.children.map((node) => node.className),
            ['chat-quiz-chip', 'chat-live-project-card-btn chat-quiz-project', 'chat-quiz-status']);
        assert.deepEqual(ownHead.children.map((node) => node.className), ['chat-quiz-chip', 'chat-quiz-status']);
        assert.deepEqual(shape(head.children[2]), shape(ownHead.children[1]));
        assert.equal(own.classList.contains('project-question-card'), false);
        const chip = head.children[1];
        assert.equal(chip.querySelector('.chat-live-project-status').getAttribute('aria-hidden'), 'true');
        chip.click();
        assert.deepEqual(main.opened.map((detail) => [detail.project.id, detail.project.chat_id, detail.task_id, detail.quiz_id]),
            [['p1', 23, 't-1', 'qz-1']]);
    } finally { main.restore(); room.restore(); }
});

test('an answer pressed in Main shows the result for five seconds, then only the Main copy goes', () => withTimers(async () => {
    const fx = mainFixture();
    try {
        fx.add(WAITING);
        const [card] = fx.cards();
        options(card)[1].click();
        options(card)[0].click();
        await turn();
        const sent = JSON.parse(fx.calls[0].init.body);
        assert.deepEqual([fx.calls.length, sent.decision_id, sent.option_index, 'comment' in sent], [1, 'quiz:t-1:qz-1', 1, false]);
        assert.deepEqual([card.dataset.state, text(card, 'chat-quiz-status-text')], ['answered', 'You answered']);
        assert.ok(options(card)[1].classList.contains('chosen') && options(card).every((button) => button.disabled));
        assert.equal(card.querySelector('.chat-quiz-comment'), null, 'a settled copy takes no more input');
        mock.timers.tick(SETTLE_MS - 1);
        assert.equal(fx.column.children.length, 1, 'the result stays readable for the whole moment');
        mock.timers.tick(1);
        assert.deepEqual([fx.column.children.length, fx.removed.length], [0, 1]);
        // Neither the stale open snapshot nor the answered one brings it back.
        fx.add(WAITING);
        fx.add({ ...ROW, quiz_state: 'answered', answered_index: 1 });
        fx.decision.appendActivityQuestion({ ...WAITING, owner_wait_state: 'waiting' });
        assert.deepEqual([fx.column.children.length, fx.calls.length], [0, 1]);
    } finally { fx.restore(); }
}));

test('an answer from any other source settles the copy once; repeated observations never restart the countdown', () => withTimers(async () => {
    const fx = mainFixture();
    try {
        fx.add({ ...WAITING, quiz_id: 'from-project' });
        fx.add({ ...ROW, quiz_id: 'from-history' });
        fx.add({ ...WAITING, quiz_id: 'from-census', ts: ASKED[1] });
        const [project, history, census] = fx.cards();
        // The Project form (or another device) answered: the live quiz_state frame.
        fx.decision.applyQuizStateFrame({}, { type: 'quiz_state', task_id: 't-1', quiz_id: 'from-project', state: 'answered',
            answered_index: 0, comment: 'From the Project form.', ts: '2026-09-18T23:59:59+00:00' });
        assert.deepEqual([text(project, 'chat-quiz-status-text'), text(project, 'chat-quiz-answer')],
            ['You answered', "Owner's answer: From the Project form."]);
        mock.timers.tick(3000);
        // A history re-read carries the same answer, and so does the next census tick.
        fx.add({ ...ROW, quiz_id: 'from-project', quiz_state: 'answered', answered_index: 0 });
        fx.add({ ...ROW, quiz_id: 'from-history', quiz_state: 'answered', comment: 'Neither — use the archive.' });
        fx.decision.appendActivityQuestion({ ...WAITING, quiz_id: 'from-census', ts: ASKED[1], quiz_state: 'answered',
            answered_index: 1, owner_wait_state: 'resumed' });
        assert.equal(text(history, 'chat-quiz-answer'), "Owner's answer: Neither — use the archive.");
        assert.ok(options(census)[1].classList.contains('chosen'));
        mock.timers.tick(1999);
        fx.decision.applyQuizStateFrame({}, { task_id: 't-1', quiz_id: 'from-project', state: 'answered', answered_index: 0 });
        mock.timers.tick(1);
        assert.deepEqual(fx.cards().map((card) => card.dataset.quizId), ['from-history', 'from-census'],
            'the first answer started the only countdown');
        mock.timers.tick(3000);
        assert.deepEqual([fx.column.children.length, fx.removed.length], [0, 3]);
        assert.equal(fx.calls.length, 0, 'observing an answer never sends one');
    } finally { fx.restore(); }
}));

test('own words answer from Main; a failed attempt stays answerable and a lost race settles into the winner', () => withTimers(async () => {
    const replies = [
        { ok: false, status: 500, json: async () => ({}) },
        { ok: false, status: 409, json: async () => ({ state: 'answered', answered_index: 1, comment: 'Winner.' }) },
    ];
    const fx = mainFixture({ fetchImpl: async (_url, init) => replies.shift()
        || { ok: true, status: 200, json: async () => ({ ok: true, state: 'answered', comment: JSON.parse(init.body).comment }) } });
    try {
        fx.add(WAITING);
        fx.add({ ...ROW, quiz_id: 'own-words' });
        const [raced, own] = fx.cards();
        options(raced)[0].click();
        await turn();
        assert.deepEqual([raced.dataset.state, fx.toasts[0].text], ['open', 'Could not record the answer (500).']);
        mock.timers.tick(SETTLE_MS);
        assert.equal(fx.column.children.length, 2, 'a refused attempt starts no countdown');
        options(raced)[0].click();
        await turn();
        assert.deepEqual([raced.dataset.state, text(raced, 'chat-quiz-answer'), fx.toasts[1].text],
            ['answered', "Owner's answer: Winner.", 'Already answered.']);
        assert.ok(options(raced)[1].classList.contains('chosen'));
        const field = own.querySelector('.chat-quiz-comment');
        field.value = 'Neither — keep both.';
        field.listeners.get('input')();
        own.querySelector('.chat-quiz-send').click();
        await turn();
        const sent = JSON.parse(fx.calls.at(-1).init.body);
        assert.deepEqual([sent.comment, 'option_index' in sent], ['Neither — keep both.', false]);
        assert.equal(text(own, 'chat-quiz-answer'), "Owner's answer: Neither — keep both.");
        mock.timers.tick(SETTLE_MS);
        assert.equal(fx.column.children.length, 0);
    } finally { fx.restore(); }
}));

test('an unknown source keeps what is known; otherwise the copy is a safe record with its way to the Project', () => {
    const fx = mainFixture();
    try {
        fx.add(WAITING);
        fx.add({ ...ROW, quiz_state: 'unknown', source_status: 'unavailable', question: '', options: [] });
        const [known] = fx.cards();
        assert.deepEqual([known.dataset.state, text(known, 'chat-quiz-status-text')], ['open', 'Waiting for your answer']);
        assert.equal(options(known)[0].disabled, false, 'an unavailable read never disables a known question');
        fx.add({ ...ROW, quiz_id: 'cold', quiz_state: 'unknown', source_status: 'unavailable' });
        const cold = fx.cards()[1];
        assert.deepEqual([cold.dataset.state, text(cold, 'chat-quiz-status-text')], ['unknown', 'Status unavailable']);
        assert.ok(options(cold).every((button) => button.disabled) && !cold.querySelector('.chat-quiz-comment'),
            'no invented invitation to answer');
        // The source becomes readable: the whole answerable form mounts where the record was.
        fx.add({ ...ROW, quiz_id: 'cold', quiz_state: 'open' });
        const upgraded = fx.cards()[1];
        assert.notEqual(upgraded, cold);
        assert.deepEqual([upgraded.dataset.state, Boolean(upgraded.querySelector('.chat-quiz-comment'))], ['open', true]);
        // A census row that names a wait but not its form: a partial copy, then the form in place.
        fx.decision.appendActivityQuestion({ task_id: 't-9', quiz_id: 'narrow', project_id: 'p1', project_chat_id: 23,
            quiz_state: 'open', owner_wait_state: 'waiting', ts: ASKED[1] });
        const partial = fx.cards()[2];
        assert.deepEqual([text(partial, 'chat-quiz-question'), options(partial).length, Boolean(partial.querySelector('.chat-quiz-comment'))],
            ['Open the original question for its text.', 0, false]);
        partial.querySelector('.chat-quiz-project').click();
        assert.equal(fx.opened.at(-1).quiz_id, 'narrow');
        fx.add({ ...WAITING, task_id: 't-9', quiz_id: 'narrow', ts: ASKED[1], project_name: 'Storage v2' });
        const full = fx.cards()[2];
        assert.deepEqual([labels(full), text(full, 'chat-live-project-name'), fx.column.children.length], [['Yes', 'No'], 'Storage v2', 3]);
    } finally { fx.restore(); }
});

test('a partial copy that learns its form together with the answer still shows the result, then goes', () => withTimers(() => {
    const fx = mainFixture();
    const narrow = (quizId) => fx.decision.appendActivityQuestion({ task_id: 't-9', quiz_id: quizId, project_id: 'p1',
        project_chat_id: 23, quiz_state: 'open', owner_wait_state: 'waiting', ts: ASKED[1] });
    try {
        // The census names the wait before any row carries its form; the next history read is
        // complete and already answered (the Project form or another device answered meanwhile).
        narrow('at-once');
        const partial = fx.cards()[0];
        assert.equal(options(partial).length, 0);
        fx.add({ ...WAITING, task_id: 't-9', quiz_id: 'at-once', ts: ASKED[1], quiz_state: 'answered', answered_index: 1,
            owner_wait_state: 'resumed' });
        const full = fx.cards()[0];
        assert.notEqual(full, partial, 'the whole form mounts in place of the partial copy');
        assert.deepEqual([full.dataset.state, text(full, 'chat-quiz-status-text'), labels(full)], ['answered', 'You answered', ['Yes', 'No']]);
        assert.ok(options(full)[1].classList.contains('chosen') && options(full).every((button) => button.disabled));
        mock.timers.tick(SETTLE_MS - 1);
        assert.equal(fx.column.children.length, 1, 'the result stays readable for the whole moment');
        mock.timers.tick(1);
        assert.deepEqual([fx.column.children.length, fx.removed.length], [0, 1]);
        // An answer observed while the copy was still partial keeps its one countdown across the remount.
        narrow('answered-first');
        fx.decision.applyQuizStateFrame({}, { task_id: 't-9', quiz_id: 'answered-first', state: 'answered', answered_index: 0 });
        mock.timers.tick(2000);
        fx.add({ ...WAITING, task_id: 't-9', quiz_id: 'answered-first', ts: ASKED[1], quiz_state: 'answered', answered_index: 0 });
        assert.deepEqual([labels(fx.cards()[0]), options(fx.cards()[0])[0].classList.contains('chosen')], [['Yes', 'No'], true]);
        mock.timers.tick(SETTLE_MS - 2001);
        assert.equal(fx.column.children.length, 1);
        mock.timers.tick(1);
        assert.deepEqual([fx.column.children.length, fx.removed.length], [0, 2], 'never restarted by the remount');
    } finally { fx.restore(); }
}));

test('a removed copy stays removed after the bounded memory forgets it: the canonical source decides', () => withTimers(async () => {
    // The task detail the owner-quiz store serves (GET /api/tasks/{id}): the canonical lifecycle.
    const canonical = { 't-1': { 'qz-1': 'open', 'fresh': 'open', 'kept': 'open' } };
    const reads = [];
    let unavailable = false;
    const fetchDetail = async (taskId) => {
        reads.push(taskId);
        if (unavailable) throw new Error('detail read failed');
        return { task_id: taskId, project_id: 'p1', owner_quiz: Object.fromEntries(Object.entries(canonical[taskId] || {})
            .map(([quizId, state]) => [quizId, { quiz_id: quizId, state, question: 'Merge **now**?', options: ['Yes', 'No'],
                asked_at: ASKED[0], ...(state === 'answered' ? { answered_index: 0 } : {}) }])) };
    };
    const fx = mainFixture({ fetchDetail });
    let noise = 0;
    const forget = () => {
        // A long-lived Main observes many other questions (every quiz_state frame is one) past
        // the memory's bound (OBSERVATION_LIMIT, 2000).
        for (let i = 0; i < 2001; i += 1)
            fx.decision.applyQuizStateFrame({}, { task_id: 'noise', quiz_id: `n-${noise += 1}`, state: 'open' });
    };
    try {
        fx.add(WAITING);
        fx.add({ ...WAITING, quiz_id: 'kept' });
        fx.decision.applyQuizStateFrame({}, { task_id: 't-1', quiz_id: 'qz-1', state: 'answered', answered_index: 0 });
        canonical['t-1']['qz-1'] = 'answered';
        mock.timers.tick(SETTLE_MS);
        assert.deepEqual(fx.cards().map((card) => card.dataset.quizId), ['kept']);
        assert.equal(reads.length, 0, 'while this tab still remembers, nothing is re-read');
        forget();
        // A stale open snapshot (a history page or census read begun before the answer): the
        // copy is not resurrected, neither at once nor after the canonical read.
        fx.add(WAITING);
        fx.decision.appendActivityQuestion({ ...WAITING });
        assert.deepEqual(fx.cards().map((card) => card.dataset.state), ['open', 'unknown']);
        await turn();
        assert.deepEqual([fx.cards().map((card) => card.dataset.quizId), reads], [['kept'], ['t-1']],
            'one canonical read for the question, however many snapshots named it');
        fx.add(WAITING);
        assert.deepEqual([fx.cards().length, reads.length], [1, 1], 'the canonical answer is remembered again');
        // The copy still shown keeps its own lifecycle: a stale snapshot cannot move it backwards.
        fx.decision.applyQuizStateFrame({}, { task_id: 't-1', quiz_id: 'kept', state: 'expired_terminal' });
        forget();
        fx.add({ ...WAITING, quiz_id: 'kept' });
        assert.equal(fx.cards()[0].dataset.state, 'expired_terminal');
        // A question this tab never saw still arrives, once the canonical read confirms it is unanswered.
        fx.add({ ...WAITING, quiz_id: 'fresh', ts: ASKED[1] });
        assert.equal(fx.cards()[1].dataset.state, 'unknown');
        await turn();
        assert.deepEqual(fx.cards().map((card) => [card.dataset.quizId, card.dataset.state]), [['kept', 'expired_terminal'], ['fresh', 'open']]);
        assert.equal(Boolean(fx.cards()[1].querySelector('.chat-quiz-comment')), true);
        // An unavailable canonical read proves nothing: do not turn the stale open
        // snapshot into an answerable card. Its chip remains a recovery path.
        forget();
        unavailable = true;
        fx.add({ ...WAITING, quiz_id: 'unreadable' });
        await turn();
        const unknown = fx.cards()[2];
        assert.equal(unknown.dataset.state, 'unknown');
        assert.ok(options(unknown).every(button => button.disabled));
        unknown.querySelector('.chat-quiz-project').click();
        assert.equal(fx.opened.at(-1).quiz_id, 'unreadable');
        unavailable = false;
        canonical['t-1'].unreadable = 'open';
        fx.add({ ...WAITING, quiz_id: 'unreadable' });
        assert.equal(unknown.dataset.state, 'unknown', 'a stale re-delivery cannot unlock the form');
        const bubble = fx.column.children[2];
        fx.decision.releaseViews(bubble); bubble.remove();
        fx.add({ ...WAITING, quiz_id: 'unreadable' });
        await turn();
        assert.equal(fx.cards()[2].dataset.state, 'open', 'revisiting the history row can recover too');
        // An answer observed while the canonical read is in flight wins; so does a destroyed Main.
        unavailable = false;
        forget();
        fx.add({ ...WAITING, quiz_id: 'answered-meanwhile' });
        fx.decision.applyQuizStateFrame({}, { task_id: 't-1', quiz_id: 'answered-meanwhile', state: 'answered', answered_index: 1 });
        fx.add({ ...WAITING, quiz_id: 'torn-down' });
        fx.decision.destroy();
        await turn();
        assert.equal(fx.cards().length, 5, 'destroy never lets the pending safe copy become answerable');
        assert.equal(fx.cards()[3].dataset.state, 'answered');
        assert.equal(fx.cards()[4].dataset.state, 'unknown');
        assert.equal(fx.calls.length, 0, 'revalidation never answers anything');
    } finally { fx.restore(); }
}));

test('failed, missing and foreign canonical reads keep a safe Project recovery path after eviction', () => withTimers(async () => {
    for (const failure of ['network', '404', 'missing', 'wrong-project', 'wrong-task', 'wrong-quiz', 'unknown']) {
        let recovered = false;
        const detail = () => ({ task_id: 't-1', project_id: 'p1', owner_quiz: {
            stale: { quiz_id: 'stale', state: 'open', question: 'Canonical source', options: ['A', 'B'] },
        } });
        const fx = mainFixture({ fetchDetail: async () => {
            if (recovered) return detail();
            if (failure === 'network' || failure === '404') throw new Error(failure);
            const row = detail();
            if (failure === 'missing') row.owner_quiz = {};
            if (failure === 'wrong-project') row.project_id = 'p2';
            if (failure === 'wrong-task') row.task_id = 'other';
            if (failure === 'wrong-quiz') row.owner_quiz.stale.quiz_id = 'other';
            if (failure === 'unknown') row.owner_quiz.stale.state = 'unknown';
            return row;
        } });
        try {
            fx.add({ ...WAITING, quiz_id: 'stale' });
            fx.decision.applyQuizStateFrame({}, { task_id: 't-1', quiz_id: 'stale', state: 'answered', answered_index: 1 });
            mock.timers.tick(SETTLE_MS);
            assert.equal(fx.cards().length, 0);
            for (let i = 0; i < 2001; i += 1)
                fx.decision.applyQuizStateFrame({}, { task_id: 'noise', quiz_id: `n-${i}`, state: 'open' });
            fx.add({ ...WAITING, quiz_id: 'stale', question: 'Snapshot source' });
            await turn();
            const card = fx.cards()[0];
            assert.equal(text(card, 'chat-quiz-status-text'), 'Status unavailable', failure);
            assert.ok(options(card).every(button => button.disabled) && !card.querySelector('.chat-quiz-comment'));
            fx.add({ ...WAITING, quiz_id: 'stale' });
            await turn();
            fx.decision.applyQuizStateFrame({}, { task_id: 't-1', quiz_id: 'stale', state: 'open' });
            assert.equal(card.dataset.state, 'unknown', 'snapshots and non-answer live facts cannot unlock an unverified copy');
            card.querySelector('.chat-quiz-project').click();
            assert.equal(fx.opened.at(-1).quiz_id, 'stale');
            // An explicit retry of the owned pointer can recover the canonical Project form.
            recovered = true;
            fx.add({ ...WAITING, quiz_id: 'stale' });
            await turn();
            assert.equal(fx.cards()[0].dataset.state, 'open');
            assert.equal(text(fx.cards()[0], 'chat-quiz-question'), 'Canonical source');
        } finally { fx.decision.destroy(); fx.restore(); }
    }
}));

test('an answered pointer settles a pending safe copy for five seconds', () => withTimers(async () => {
    const pending = [];
    const fx = mainFixture({ fetchDetail: () => new Promise(resolve => pending.push(resolve)) });
    try {
        for (let i = 0; i < 2001; i += 1)
            fx.decision.applyQuizStateFrame({}, { task_id: 'noise', quiz_id: `n-${i}`, state: 'open' });
        fx.add({ ...WAITING, quiz_id: 'forgotten' });
        await turn();
        const unknown = fx.cards()[0];
        assert.equal(unknown.dataset.state, 'unknown');
        // This could be an already-removed answer whose observation was evicted. A positive
        // answered pointer shows its recorded result for the normal Main-only settlement.
        fx.add({ ...WAITING, quiz_id: 'forgotten', quiz_state: 'answered', answered_index: 1 });
        assert.equal(unknown.dataset.state, 'answered');
        assert.ok(options(unknown).every(button => button.disabled));
        pending[0]({ task_id: 't-1', project_id: 'p1', owner_quiz: {
            forgotten: { quiz_id: 'forgotten', state: 'answered', answered_index: 1,
                question: 'Canonical source', options: ['A', 'B'] },
        } });
        await turn();
        assert.equal(fx.column.children.length, 1, 'canonical answer cannot cut the live result short');
        mock.timers.tick(SETTLE_MS - 1);
        assert.equal(fx.column.children.length, 1);
        mock.timers.tick(1);
        assert.equal(fx.column.children.length, 0);
    } finally { fx.decision.destroy(); fx.restore(); }
}));

test('revalidation bypasses an already-started navigation detail read', () => withTimers(async () => {
    const pending = [];
    const fx = mainFixture({ fetchDetail: () => new Promise(resolve => pending.push(resolve)) });
    const detail = (state) => ({ task_id: 't-1', project_id: 'p1', owner_quiz: {
        stale: { quiz_id: 'stale', state, answered_index: state === 'answered' ? 1 : null,
            question: 'Canonical source', options: ['A', 'B'] },
    } });
    try {
        for (let i = 0; i < 2001; i += 1)
            fx.decision.applyQuizStateFrame({}, { task_id: 'noise', quiz_id: `n-${i}`, state: 'open' });
        const old = fx.decision.readQuestion('t-1', 'stale', 'p1');
        await turn();
        fx.add({ ...WAITING, quiz_id: 'stale' });
        await turn();
        assert.equal(pending.length, 2, 'a read already on the wire predates the revalidation');
        pending[0](detail('open'));
        await old;
        await turn();
        assert.equal(fx.cards().at(-1).dataset.state, 'unknown', 'the old open observation cannot authorize the copy');
        pending[1](detail('answered'));
        await turn();
        assert.ok(!fx.cards().some(card => card.dataset.quizId === 'stale'));
    } finally { fx.decision.destroy(); fx.restore(); }
}));

test('a copy mounted after a sibling started its validation never inherits that older read', () => withTimers(async () => {
    const pending = [];
    const fx = mainFixture({ fetchDetail: () => new Promise(resolve => pending.push(resolve)) });
    const detail = (states) => ({ task_id: 't-1', project_id: 'p1', owner_quiz: Object.fromEntries(Object.entries(states)
        .map(([quizId, state]) => [quizId, { quiz_id: quizId, state, question: 'Canonical source', options: ['A', 'B'],
            ...(state === 'answered' ? { answered_index: 1 } : {}) }])) });
    try {
        for (let i = 0; i < 2001; i += 1)
            fx.decision.applyQuizStateFrame({}, { task_id: 'noise', quiz_id: `n-${i}`, state: 'open' });
        fx.add({ ...WAITING, quiz_id: 'sibling' });
        await turn();
        // Meanwhile 'late' is answered and this tab misses the frame; a stale open snapshot of it
        // arrives while the sibling's read, begun before that answer, is still on the wire.
        fx.add({ ...WAITING, quiz_id: 'late', ts: ASKED[1] });
        await turn();
        assert.equal(pending.length, 2, 'the newer copy validates with its own read');
        pending[0](detail({ sibling: 'open', late: 'open' }));
        await turn();
        const [sibling, late] = fx.cards();
        assert.deepEqual([sibling.dataset.state, late.dataset.state], ['open', 'unknown']);
        assert.ok(options(late).every(button => button.disabled) && !late.querySelector('.chat-quiz-comment'),
            'the older read cannot unlock the newer copy');
        pending[1](detail({ sibling: 'open', late: 'answered' }));
        await turn();
        assert.deepEqual(fx.cards().map(card => [card.dataset.quizId, card.dataset.state]), [['sibling', 'open']]);
        fx.add({ ...WAITING, quiz_id: 'late', ts: ASKED[1] });
        assert.deepEqual([fx.cards().length, pending.length], [1, 2], 'the canonical answer is remembered again');
    } finally { fx.decision.destroy(); fx.restore(); }
}));

test('a canonical answer removes the copy in its own viewport transaction, never anchored on it', () => withTimers(async () => {
    const pending = [], anchors = [];
    let column = null, depth = 0;
    // chat.js withStableViewport: only the outermost write captures the anchor (the first node it
    // does not exclude) and restores it afterwards; a nested write joins that transaction.
    const viewport = (mutate, { excludeAnchorNode = null } = {}) => {
        if (depth > 0) return mutate();
        const anchor = column?.children.find(node => node !== excludeAnchorNode) || null;
        depth = 1;
        try { return mutate(); } finally { depth = 0; anchors.push(anchor); }
    };
    const fx = mainFixture({ onDomWrite: viewport, fetchDetail: () => new Promise(resolve => pending.push(resolve)),
        removeMessageNode: node => viewport(() => { node.remove(); return true; }, { excludeAnchorNode: node }) });
    column = fx.column;
    try {
        for (let i = 0; i < 2001; i += 1)
            fx.decision.applyQuizStateFrame({}, { task_id: 'noise', quiz_id: `n-${i}`, state: 'open' });
        fx.add({ ...WAITING, quiz_id: 'gone' });
        fx.add({ ...WAITING, quiz_id: 'below', ts: ASKED[1] });
        await turn();
        const [gone, below] = fx.column.children;
        anchors.length = 0;
        pending[0]({ task_id: 't-1', project_id: 'p1', owner_quiz: {
            gone: { quiz_id: 'gone', state: 'answered', answered_index: 0, question: 'Canonical source', options: ['A', 'B'] } } });
        await turn();
        assert.deepEqual([fx.column.children, gone.parentNode], [[below], null]);
        assert.deepEqual(anchors, [below], 'the reader stays on the node below, not on the one that left');
    } finally { fx.decision.destroy(); fx.restore(); }
}));

test('history retirement cancels pending revalidation and an old response cannot change a replacement copy', () => withTimers(async () => {
    const pending = [], queued = [];
    let queueNext = false;
    const fx = mainFixture({ onDomWrite: mutate => {
        if (queueNext) { queueNext = false; queued.push(mutate); return; }
        return mutate();
    }, fetchDetail: () => new Promise(resolve => pending.push(resolve)) });
    const detail = state => ({ task_id: 't-1', project_id: 'p1', owner_quiz: {
        stale: { quiz_id: 'stale', state, question: 'Canonical source', options: ['A', 'B'] },
    } });
    try {
        for (let i = 0; i < 2001; i += 1)
            fx.decision.applyQuizStateFrame({}, { task_id: 'noise', quiz_id: `n-${i}`, state: 'open' });
        fx.add({ ...WAITING, quiz_id: 'stale', history_id: 'page-row' });
        await turn();
        const old = fx.column.children[0];
        // Exactly the existing chat.js releaseMessageNode seam used by history eviction.
        fx.decision.releaseViews(old); old.remove();
        fx.add({ ...WAITING, quiz_id: 'stale', history_id: 'page-row' });
        await turn();
        pending[0](detail('open'));
        await turn();
        assert.equal(fx.cards()[0].dataset.state, 'unknown', 'retired read cannot unlock the replacement');
        queueNext = true;
        pending[1](detail('open'));
        await turn();
        assert.equal(queued.length, 1);
        const replacement = fx.column.children[0];
        fx.decision.releaseViews(replacement); replacement.remove();
        for (const mutate of queued.splice(0)) mutate();
        assert.equal(fx.column.children.length, 0, 'even a queued DOM continuation respects retirement');
        fx.decision.destroy();
        fx.add({ ...WAITING, quiz_id: 'stale' });
        assert.equal(fx.column.children.length, 0);
    } finally { fx.restore(); }
}));

test('an unchanged row writes nothing and a narrower re-delivery never blanks the form', () => {
    let writes = 0;
    const fx = mainFixture({ onDomWrite: (mutate) => { writes += 1; return mutate(); } });
    try {
        fx.add({ ...WAITING, assumption: 'Yes meanwhile' });
        const painted = writes;
        fx.add({ ...WAITING, assumption: 'Yes meanwhile', ts: ASKED[0], history_id: 'later' });
        // The activity census re-delivers the waiting question without its display fields.
        fx.decision.buildQuestionPointer({ ...WAITING, question: '', options: [], option_details: [], stake: '',
            assumption: '', recommended_index: null, project_name: '' });
        assert.equal(writes, painted + 1, 'only the append attempt itself; the card is untouched');
        const [card] = fx.cards();
        assert.deepEqual([labels(card), text(card, 'chat-quiz-stake'), text(card, 'chat-live-project-name')],
            [['Yes', 'No'], 'At stake: Release timing', 'Storage']);
        fx.decision.buildQuestionPointer({ ...WAITING, project_name: 'Storage v2' });
        assert.equal(text(card, 'chat-live-project-name'), 'Storage v2', 'a renamed Project renames the chip in place');
    } finally { fx.restore(); }
});

test('a fresh single-wait census ends older waits in place without settling or removing them', () => withTimers(() => {
    const fx = mainFixture();
    try {
        fx.add(WAITING);
        fx.add({ ...WAITING, quiz_id: 'q2', ts: ASKED[1] });
        fx.add({ ...WAITING, task_id: 'other' });
        fx.decision.appendActivityQuestion({ ...WAITING, quiz_id: 'q2', ts: ASKED[1] }, 0);
        const [older, newer, foreign] = fx.cards();
        assert.equal(text(older, 'chat-quiz-status-text'), 'Waiting for your answer', 'a pre-arrival read cannot end a newer wait');
        fx.decision.appendActivityQuestion({ ...WAITING, quiz_id: 'q2', ts: ASKED[1] });
        assert.deepEqual([older, newer, foreign].map((card) => text(card, 'chat-quiz-status-text')), [
            'Unanswered · the task continued; an answer is still accepted', 'Waiting for your answer', 'Waiting for your answer']);
        assert.match(text(older, 'chat-quiz-assumption'), /^The wait ended without an answer/);
        assert.deepEqual([older.dataset.state, Boolean(older.querySelector('.chat-quiz-comment'))], ['open', true]);
        // Neither a missing stamp nor a sub-millisecond tie is an order.
        fx.add({ ...WAITING, quiz_id: 'q3', ts: '' });
        fx.add({ ...WAITING, quiz_id: 'q4', ts: '2026-09-18T22:00:07.000100+00:00' });
        fx.decision.appendActivityQuestion({ ...WAITING, quiz_id: 'q5', ts: '2026-09-18T22:00:07.000900+00:00' });
        assert.deepEqual(fx.cards().slice(3, 5).map((card) => text(card, 'chat-quiz-status-text')),
            ['Waiting for your answer', 'Waiting for your answer']);
        fx.add(WAITING);
        assert.equal(text(older, 'chat-quiz-status-text'), 'Unanswered · the task continued; an answer is still accepted',
            'stale history cannot restore the closed wait');
        mock.timers.tick(SETTLE_MS * 2);
        assert.equal(fx.removed.length, 0, 'an unanswered question is never removed');
    } finally { fx.restore(); }
}));

test('focus stays with the owner: inside the settled copy, then on to the next question or the composer', () => withTimers(async () => {
    let composer = 0;
    const fx = mainFixture({ focusAfterRemoval: () => { composer += 1; } });
    try {
        fx.add(WAITING);
        fx.add({ ...WAITING, quiz_id: 'q2', ts: ASKED[1] });
        const [first, second] = fx.cards();
        const pressed = options(first)[0];
        pressed.focus();
        pressed.click();
        await turn();
        // The pressed option is now a disabled record: focus stays in this copy.
        assert.equal(globalThis.document.activeElement, first.querySelector('.chat-quiz-question'));
        mock.timers.tick(SETTLE_MS);
        assert.equal(globalThis.document.activeElement, second.querySelector('.chat-quiz-question'));
        // The last copy hands a keyboard owner's focus to the composer...
        options(second)[1].focus();
        options(second)[1].click();
        await turn();
        mock.timers.tick(SETTLE_MS);
        assert.equal(composer, 1);
        // ...and leaves focus alone when it was elsewhere, or when a pointer put it there.
        fx.add({ ...WAITING, quiz_id: 'q3' });
        fx.add({ ...WAITING, quiz_id: 'q4' });
        const [, fourth] = fx.cards();
        const elsewhere = new NodeStub();
        elsewhere.focus();
        fx.decision.applyQuizStateFrame({}, { task_id: 't-1', quiz_id: 'q3', state: 'answered', answered_index: 0 });
        mock.timers.tick(SETTLE_MS);
        assert.equal(globalThis.document.activeElement, elsewhere);
        const tapped = options(fourth)[0];
        globalThis.document.pointerModality = true;
        tapped.focus();
        fx.decision.applyQuizStateFrame({}, { task_id: 't-1', quiz_id: 'q4', state: 'superseded' });
        fx.decision.applyQuizStateFrame({}, { task_id: 't-1', quiz_id: 'q4', state: 'answered', answered_index: 0 });
        mock.timers.tick(SETTLE_MS);
        assert.deepEqual([composer, fx.column.children.length], [1, 0], 'a tap never summons the keyboard');
    } finally { fx.restore(); }
}));

test('a replaced question stays as a read-only record; a native Main quiz is never removed', () => withTimers(async () => {
    const fx = mainFixture();
    try {
        fx.add({ ...WAITING, quiz_id: 'replaced' });
        fx.decision.applyQuizStateFrame({}, { task_id: 't-1', quiz_id: 'replaced', state: 'superseded' });
        const native = fx.decision.buildQuizCard({ type: 'quiz', role: 'assistant', quiz_id: 'native', task_id: 't-main',
            question: 'Main question?', options: [{ label: 'A' }, { label: 'B' }], assumption: 'A', state: 'open', ts: ASKED[0] });
        fx.column.append(native);
        const nativeCard = native.children[0];
        assert.equal(nativeCard.querySelector('.chat-quiz-project'), null);
        options(nativeCard)[0].click();
        await turn();
        mock.timers.tick(SETTLE_MS * 2);
        const [replaced] = fx.cards();
        assert.deepEqual([replaced.dataset.state, text(replaced, 'chat-quiz-status-text'), options(replaced)[0].disabled],
            ['superseded', 'Replaced by a newer question', true]);
        assert.deepEqual([nativeCard.dataset.state, fx.column.children.length, fx.removed.length], ['answered', 2, 0]);
    } finally { fx.restore(); }
}));

test('released and destroyed copies take their countdown and rendered markdown with them', () => withTimers(async () => {
    const released = [];
    const fx = mainFixture({ renderMarkdown: (value) => `<p>${value}</p>`, enhanceMarkdown: (node) => () => released.push(node) });
    try {
        fx.add(WAITING);
        fx.add({ ...WAITING, quiz_id: 'q2' });
        const [first, second] = fx.cards();
        assert.equal(first.querySelector('.chat-quiz-question').innerHTML, '<p>Merge **now**?</p>');
        fx.decision.applyQuizStateFrame({}, { task_id: 't-1', quiz_id: 'qz-1', state: 'answered', answered_index: 0 });
        // History eviction releases the node mid-countdown through the ordinary lifecycle.
        fx.decision.releaseViews(first.parentNode);
        assert.deepEqual(released, [first]);
        mock.timers.tick(SETTLE_MS);
        assert.equal(fx.removed.length, 0, 'a released copy has no countdown left');
        fx.decision.applyQuizStateFrame({}, { task_id: 't-1', quiz_id: 'q2', state: 'answered', answered_index: 0 });
        fx.decision.destroy();
        mock.timers.tick(SETTLE_MS);
        assert.deepEqual([fx.removed.length, released], [0, [first, second]]);
    } finally { fx.restore(); }
}));

test('a Project room never mirrors its own questions', () => {
    const fx = fixture({ insertMessageNode: () => { throw new Error('no Main copy in a Project room'); } });
    try {
        assert.equal(fx.decision.appendQuestionPointer(WAITING), false);
        assert.equal(fx.decision.appendActivityQuestion(WAITING), false);
    } finally { fx.restore(); }
});

// Retained census regressions from question_rows: layout changed, ordering did not.
test('a census without positive wait evidence leaves earlier waits untouched', () => {
    const fx = mainFixture();
    try {
        fx.add(WAITING);
        fx.add({ ...WAITING, quiz_id: 'new', ts: ASKED[1] });
        fx.decision.appendActivityQuestion({ ...WAITING, quiz_id: 'new', ts: ASKED[1], owner_wait_state: undefined });
        assert.deepEqual(fx.cards().map(card => text(card, 'chat-quiz-status-text')),
            ['Waiting for your answer', 'Waiting for your answer']);
    } finally { fx.decision.destroy(); fx.restore(); }
});

test('a census naming an older resumed wait cannot end the newer wait', () => {
    const fx = mainFixture();
    try {
        fx.add(WAITING);
        fx.add({ ...WAITING, quiz_id: 'new', ts: ASKED[1] });
        fx.decision.appendActivityQuestion({ ...WAITING, owner_wait_state: 'resumed' }, Date.now() + 1000);
        const newer = fx.cards()[1];
        assert.equal(text(newer, 'chat-quiz-status-text'), 'Waiting for your answer');
        assert.ok(options(newer).every(button => !button.disabled));
    } finally { fx.decision.destroy(); fx.restore(); }
});

test('an optional canonical ask inherits no resumed fact from another quiz wait', async () => {
    const fx = mainFixture({ fetchDetail: async () => ({ task_id: 't-1', project_id: 'p1',
        owner_quiz: { opt: { quiz_id: 'opt', state: 'open', question: 'Format?', options: ['A', 'B'], assumption: 'A' } },
        owner_wait: { quiz_id: 'new', state: 'waiting' } }) });
    try {
        const detail = await fx.decision.readQuestion('t-1', 'opt', 'p1');
        assert.equal(detail.owner_wait_state, undefined);
        const card = fx.decision.buildQuizCard(detail);
        assert.equal(text(card, 'chat-quiz-status-text'), 'Unanswered · an answer is still accepted');
        assert.ok(options(card).every(button => !button.disabled));
    } finally { fx.decision.destroy(); fx.restore(); }
});
