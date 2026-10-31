// #1993/#1921: Main's copy of a Project question when the census arrives before the press's reply and
// its quiz_state frames, or before any frame of an answer or closure made elsewhere. The bodies are
// ones the real isolated server sent this page (fixtures/main_answer_census_first_wire.json, from
// tests/test_main_answer_census_first_browser.py); only their order is the scenario's, and the clock
// is fake. A census that stops naming a copy proves nothing about its outcome: the copy the owner was
// offered stays the same node until its exact canonical read shows that outcome for the one
// five-second interval. A copy mounted withheld from an older read and never offered keeps the
// existing suppression: proven answered, it leaves at once.
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test, { mock } from 'node:test';
import { NodeStub, fixture, frameBubble, turn } from './chat_decision_fixture.js';

const WIRE = JSON.parse(readFileSync(new URL('./fixtures/main_answer_census_first_wire.json', import.meta.url), 'utf8'));
const { task_id: TASK, quiz_id: QUIZ, pointer: POINTER, post: POST, frames: FRAMES } = WIRE;
const DETAIL = WIRE.detail_after;
const BLOCK = DETAIL.owner_quiz[QUIZ];
// The same record as the decision wrote it when it accepted the answer, before the task read it: still owed.
const OWED_DETAIL = { ...DETAIL, owner_quiz: { [QUIZ]: WIRE.owed_block } };
// Closed elsewhere instead of answered: the unanswered record with the closure fields owner_quiz.record_closed
// writes (its owner sentence OWNER_CLOSE_REASON) and its delivery made.
const { answered_index: _index, comment: _comment, answered_at: closedAt, ...UNANSWERED } = BLOCK;
const CLOSED_DETAIL = { ...DETAIL, owner_quiz: { [QUIZ]: { ...UNANSWERED, state: 'closed',
    closed_reason: 'Closed by the owner without an answer.', closed_actor: 'owner', closed_at: closedAt,
    delivery: { kind: 'closure', state: 'delivered' } } } };
const ACTIVITY_ROW = WIRE.census_activity.required_questions[0];
const SETTLE_MS = 5000;
const COMMENT = POST.request.comment;
const status = (card) => card.querySelector('.chat-quiz-status-text')?.textContent ?? null;
const chosen = (card) => card.querySelectorAll('.chat-quiz-option').map((button, index) => button.classList.contains('chosen') && index)
    .filter((index) => index !== false);
const answerLine = (card) => card.querySelector('.chat-quiz-answer')?.textContent ?? null;
const recorded = (card) => [status(card), chosen(card), answerLine(card)];
const RECORDED = ['You answered', [POST.reply.answered_index], `Owner's answer: ${COMMENT}`];

const deferred = () => { let resolve, reject; const promise = new Promise((ok, fail) => { resolve = ok; reject = fail; }); return { promise, resolve, reject }; };

// One Main column: copies enter through the feed's insertion seam and leave through its one removal seam.
// The press's reply and each exact read are held until the scenario answers them.
function mainColumn() {
    const column = new NodeStub();
    const removed = [], reads = [], replies = [];
    const fx = fixture({ isMain: true, frameNode: frameBubble, rawDetail: true,
        insertMessageNode: (node) => { column.append(node); return true; },
        removeMessageNode: (node) => { removed.push(node); node.remove(); },
        fetchImpl: () => { const reply = deferred(); replies.push(reply); return reply.promise; },
        fetchDetail: (taskId) => { const read = deferred(); reads.push({ taskId, ...read }); return read.promise; } });
    const bubble = () => column.children[0] || null;
    const card = () => bubble()?.querySelector('.chat-quiz-card') || null;
    const reply = (index = 0) => replies[index].resolve({ ok: true, status: POST.status, json: async () => POST.reply });
    return { ...fx, column, removed, reads, replies, bubble, card, reply };
}

// The owner's words, then the winning option: one press, one POST, its reply held.
async function press(main) {
    main.decision.appendQuestionPointer(POINTER);
    const card = main.card();
    const field = card.querySelector('.chat-quiz-comment');
    field.value = COMMENT;
    field.listeners.get('input')();
    card.querySelectorAll('.chat-quiz-option')[POST.request.option_index].click();
    await turn();
    assert.equal(main.calls.length, 1);
    assert.deepEqual(JSON.parse(main.calls[0].init.body).comment, COMMENT);
    assert.equal(card.dataset.submission, 'sending');
    return { bubble: main.bubble(), card };
}

function withTimers(run) {
    mock.timers.enable({ apis: ['setTimeout'] });
    return Promise.resolve().then(run).finally(() => mock.timers.reset());
}

const omittedBy = (main, census) => main.decision.syncCurrentQuestions(census.current_questions, census.current_questions_complete === true);
// After an omission: still the copy that took the press, with `reads` exact reads asked so far.
function heldThrough(main, bubble, reads) {
    assert.equal(main.bubble(), bubble, '#1993: the census removed the copy before its answer was shown');
    assert.equal(main.reads.length, reads, 'one exact read per omission, never more than one in flight');
}

test('the census before the press reply keeps the same copy; its exact read shows the recorded winner for five seconds', () => withTimers(async () => {
    const main = mainColumn();
    try {
        const { bubble, card } = await press(main);
        omittedBy(main, WIRE.census_silent);
        await turn();
        assert.equal(main.bubble(), bubble, 'the copy that took the press is the same node');
        assert.equal(main.card(), card);
        assert.deepEqual(main.removed, []);
        assert.notEqual(status(card), 'You answered', 'absence is no answer');
        assert.deepEqual(main.reads.map((read) => read.taskId), [TASK], 'one exact read of this question');
        main.reads[0].resolve(DETAIL);
        await turn();
        assert.deepEqual(recorded(card), RECORDED, 'the canonical winner and the owner\'s own words');
        assert.equal(card.querySelector('.chat-quiz-comment'), null, 'a settled copy takes no more input');
        assert.equal(card.dataset.submission, '', 'the answer in flight settled into the record');
        mock.timers.tick(SETTLE_MS - 1);
        assert.equal(main.bubble(), bubble, 'the result stays for the whole interval');
        mock.timers.tick(1);
        assert.deepEqual([main.column.children.length, main.removed], [0, [bubble]]);
        // The held reply and frames arrive late; the stale pointer is re-delivered: nothing comes back.
        main.reply(); await turn();
        for (const frame of FRAMES) main.decision.applyQuizStateFrame(main.column, frame);
        main.decision.appendQuestionPointer(POINTER);
        mock.timers.tick(60000);
        assert.deepEqual([main.column.children.length, main.removed.length, main.calls.length, main.reads.length], [0, 1, 1, 1]);
    } finally { main.restore(); }
}));

test('a census whose task activity already names the answer confirms the same copy at once', () => withTimers(async () => {
    const main = mainColumn();
    try {
        const { bubble, card } = await press(main);
        // chat.js hydrateStateSnapshot: Main's waiting census first, then each activity's required question.
        omittedBy(main, WIRE.census_activity);
        main.decision.appendActivityQuestion(ACTIVITY_ROW);
        await turn();
        assert.equal(main.bubble(), bubble, '#1993: the census removed the copy before its activity row could confirm it');
        assert.deepEqual(recorded(card), RECORDED);
        mock.timers.tick(2000);
        // Its exact read answers later with the same record: no second interval.
        main.reads[0]?.resolve(DETAIL);
        await turn();
        mock.timers.tick(SETTLE_MS - 2000 - 1);
        assert.equal(main.bubble(), bubble);
        mock.timers.tick(1);
        assert.deepEqual([main.column.children.length, main.removed], [0, [bubble]]);
        assert.ok(main.reads.length <= 1);
        assert.equal(main.calls.length, 1);
    } finally { main.restore(); }
}));

test('an owed answer the census drops keeps its pending record until a read proves its delivery', () => withTimers(async () => {
    const main = mainColumn();
    try {
        const { bubble, card } = await press(main);
        main.reply(); await turn();
        const pending = status(card);
        assert.equal(card.dataset.state, 'answered');
        assert.notEqual(pending, 'You answered', 'the reply says delivery is still owed');
        omittedBy(main, WIRE.census_silent);
        await turn();
        assert.equal(main.bubble(), bubble, 'complete withdrawal alone never retires an owed record');
        main.reads[0].resolve(OWED_DETAIL);
        await turn();
        mock.timers.tick(60000);
        assert.deepEqual([main.bubble(), main.removed.length], [bubble, 0], 'still owed: it stays');
        assert.match(status(card), /^Answer (saved|recorded)/, `a pending record, not a result (was: ${pending})`);
        // The next omission reads again; this read proves the delivery.
        omittedBy(main, WIRE.census_silent);
        await turn();
        heldThrough(main, bubble, 2);
        main.reads[1].resolve(DETAIL);
        await turn();
        assert.deepEqual(recorded(card), RECORDED);
        mock.timers.tick(SETTLE_MS - 1);
        assert.equal(main.bubble(), bubble);
        mock.timers.tick(1);
        assert.deepEqual(main.removed, [bubble]);
    } finally { main.restore(); }
}));

test('repeated omissions, the late reply and duplicate frames neither restart the interval nor read twice at once', () => withTimers(async () => {
    const main = mainColumn();
    try {
        const { bubble, card } = await press(main);
        omittedBy(main, WIRE.census_silent);
        omittedBy(main, WIRE.census_silent);
        await turn();
        omittedBy(main, WIRE.census_silent);
        await turn();
        assert.equal(main.reads.length, 1, 'one exact read in flight per copy');
        main.reads[0].resolve(DETAIL);
        await turn();
        assert.deepEqual(recorded(card), RECORDED);
        mock.timers.tick(2000);
        omittedBy(main, WIRE.census_silent);
        main.reply(); await turn();
        for (const frame of FRAMES) main.decision.applyQuizStateFrame(main.column, frame);
        main.decision.applyQuizStateFrame(main.column, FRAMES.at(-1));
        await turn();
        mock.timers.tick(SETTLE_MS - 2000 - 1);
        assert.equal(main.bubble(), bubble, 'the first confirmation started the only interval');
        mock.timers.tick(1);
        assert.deepEqual([main.removed, main.reads.length, main.calls.length], [[bubble], 1, 1]);
    } finally { main.restore(); }
}));

test('an unavailable or foreign exact read retains the copy; a later omission reads again', () => withTimers(async () => {
    for (const fail of [(read) => read.reject(new Error('task detail read failed (HTTP 503)')),
        (read) => read.resolve({ ...DETAIL, question_rooms: { [QUIZ]: { state: 'unknown' } } })]) {
        const main = mainColumn();
        try {
            const { bubble, card } = await press(main);
            main.reply(); await turn();
            omittedBy(main, WIRE.census_silent);
            await turn();
            heldThrough(main, bubble, 1);
            fail(main.reads[0]);
            await turn();
            mock.timers.tick(60000);
            assert.deepEqual([main.bubble(), main.removed.length], [bubble, 0], 'no proof: the record stays');
            assert.notEqual(status(card), 'You answered');
            // A failed read is no outcome: the owed answer keeps its state, winner and words, still pending.
            assert.deepEqual([card.dataset.state, chosen(card), answerLine(card)], ['answered', [POST.reply.answered_index], RECORDED[2]]);
            assert.match(status(card), /^Answer (saved|recorded)/);
            omittedBy(main, WIRE.census_silent);
            await turn();
            heldThrough(main, bubble, 2);
            main.reads[1].resolve(DETAIL);
            await turn();
            assert.deepEqual(recorded(card), RECORDED);
            mock.timers.tick(SETTLE_MS);
            assert.deepEqual([main.removed, main.calls.length], [[bubble], 1]);
        } finally { main.restore(); }
    }
}));

// The census's read is in flight when the same census's activity row, or the delivered frame, confirms the
// answer; the read then fails. That failure proves nothing against the confirmed record or its one interval.
const CONFIRMATIONS = {
    'activity row': [WIRE.census_activity, (main) => main.decision.appendActivityQuestion(ACTIVITY_ROW)],
    'delivered frame': [WIRE.census_silent, (main) => { for (const frame of FRAMES) main.decision.applyQuizStateFrame(main.column, frame); }],
};
for (const [name, [census, confirm]] of Object.entries(CONFIRMATIONS)) {
    test(`a read failing after the ${name} confirmed the answer keeps that winner and words for the original interval`, () => withTimers(async () => {
        const main = mainColumn();
        try {
            const { bubble, card } = await press(main);
            omittedBy(main, census);
            confirm(main);
            await turn();
            heldThrough(main, bubble, 1);
            assert.deepEqual(recorded(card), RECORDED, name);
            mock.timers.tick(1000);
            main.reads[0].reject(new Error('task detail read failed (HTTP 503)'));
            await turn();
            assert.deepEqual([card.dataset.state, recorded(card)], ['answered', RECORDED], `${name}: the failed read left the record`);
            mock.timers.tick(SETTLE_MS - 1000 - 1);
            assert.deepEqual([main.bubble(), recorded(card)], [bubble, RECORDED], `${name}: through the whole first interval`);
            mock.timers.tick(1);
            assert.deepEqual([main.column.children.length, main.removed], [0, [bubble]], `${name}: then it leaves once`);
            main.reply(); await turn();
            mock.timers.tick(60000);
            assert.deepEqual([main.column.children.length, main.reads.length, main.calls.length], [0, 1, 1], name);
        } finally { main.restore(); }
    }));
}

test('control: an offered open copy whose exact read fails keeps its words and withholds its controls', () => withTimers(async () => {
    const main = mainColumn();
    try {
        main.decision.appendQuestionPointer(POINTER);
        const bubble = main.bubble(), card = main.card();
        omittedBy(main, WIRE.census_silent);
        await turn();
        heldThrough(main, bubble, 1);
        main.reads[0].reject(new Error('task detail read failed (HTTP 503)'));
        await turn();
        mock.timers.tick(60000);
        assert.deepEqual([main.bubble(), card.dataset.state, status(card)], [bubble, 'unknown', 'Status unavailable']);
        assert.equal(card.querySelector('.chat-quiz-question').textContent, POINTER.question, 'its words stay');
        assert.equal(card.querySelector('.chat-quiz-comment').disabled, true);
        assert.ok(card.querySelectorAll('.chat-quiz-option').every((button) => button.disabled), 'no option takes a press');
        omittedBy(main, WIRE.census_silent);
        await turn();
        heldThrough(main, bubble, 2);
        main.reads[1].resolve(DETAIL);
        await turn();
        assert.deepEqual(recorded(card), RECORDED);
        mock.timers.tick(SETTLE_MS);
        assert.deepEqual([main.removed, main.calls.length], [[bubble], 0]);
    } finally { main.restore(); }
}));

test('a typed gap naming the task vouches nothing: the copy waits for proof without a read of its own', () => withTimers(async () => {
    const main = mainColumn();
    try {
        const { bubble, card } = await press(main);
        main.reply(); await turn();
        main.decision.syncCurrentQuestions([], (taskId) => taskId !== TASK);
        await turn();
        mock.timers.tick(60000);
        assert.deepEqual([main.bubble(), main.removed.length, main.reads.length], [bubble, 0, 0]);
        assert.notEqual(status(card), 'You answered');
        omittedBy(main, WIRE.census_silent);
        await turn();
        heldThrough(main, bubble, 1);
        main.reads[0].resolve(DETAIL);
        await turn();
        assert.deepEqual(recorded(card), RECORDED);
        mock.timers.tick(SETTLE_MS);
        assert.deepEqual(main.removed, [bubble]);
    } finally { main.restore(); }
}));

test('a socket close voids the exact read in flight; the reopen census reads again', () => withTimers(async () => {
    const main = mainColumn();
    try {
        const { bubble, card } = await press(main);
        omittedBy(main, WIRE.census_silent);
        await turn();
        heldThrough(main, bubble, 1);
        main.decision.setConnected(false);
        main.reads[0].resolve(DETAIL);
        await turn();
        mock.timers.tick(SETTLE_MS * 2);
        assert.deepEqual([main.bubble(), main.removed.length], [bubble, 0], 'a read begun before the close applies nothing');
        assert.notEqual(status(card), 'You answered');
        main.decision.setConnected(true);
        omittedBy(main, WIRE.census_silent);
        await turn();
        heldThrough(main, bubble, 2);
        main.reads[1].resolve(DETAIL);
        await turn();
        assert.deepEqual(recorded(card), RECORDED);
        mock.timers.tick(SETTLE_MS);
        assert.deepEqual([main.removed, main.calls.length], [[bubble], 1]);
    } finally { main.restore(); }
}));

test('destroying Main during the exact read schedules and removes nothing', () => withTimers(async () => {
    const main = mainColumn();
    try {
        const { bubble } = await press(main);
        omittedBy(main, WIRE.census_silent);
        await turn();
        heldThrough(main, bubble, 1);
        main.decision.destroy();
        main.reads[0].resolve(DETAIL);
        await turn();
        mock.timers.tick(60000);
        assert.deepEqual([main.bubble(), main.removed.length], [bubble, 0]);
    } finally { main.restore(); }
}));

test('after the read settled it, no stale pointer, frame or activity row brings the copy back', () => withTimers(async () => {
    const main = mainColumn();
    try {
        const { bubble } = await press(main);
        omittedBy(main, WIRE.census_silent);
        await turn();
        heldThrough(main, bubble, 1);
        main.reads[0].resolve(DETAIL);
        await turn();
        mock.timers.tick(SETTLE_MS);
        assert.deepEqual([main.column.children.length, main.removed], [0, [bubble]]);
        // The read's canonical receipt is this tab's memory now: an owed snapshot cannot outrank it.
        assert.deepEqual(main.decision.observedQuestion(TASK, QUIZ)?.state, 'answered');
        assert.equal(main.decision.observedQuestion(TASK, QUIZ)?.question_delivery?.state, 'delivered');
        main.decision.appendQuestionPointer(POINTER);
        main.decision.appendQuestionPointer({ ...POINTER, quiz_state: POST.reply.state, answered_index: POST.reply.answered_index,
            comment: POST.reply.comment, question_delivery: POST.reply.question_delivery });
        main.decision.applyQuizStateFrame(main.column, FRAMES[0]);
        main.decision.appendActivityQuestion(ACTIVITY_ROW);
        assert.deepEqual([main.column.children.length, main.calls.length], [0, 1]);
    } finally { main.restore(); }
}));

test('an exposed copy another device answered: the omission paints that winner and its words for five seconds', () => withTimers(async () => {
    const main = mainColumn();
    try {
        main.decision.appendQuestionPointer(POINTER);
        const bubble = main.bubble(), card = main.card();
        omittedBy(main, WIRE.census_silent);
        await turn();
        heldThrough(main, bubble, 1);
        assert.notEqual(status(card), 'You answered');
        main.reads[0].resolve(DETAIL);
        await turn();
        assert.deepEqual(recorded(card), RECORDED, 'the remote winner, not a guess');
        mock.timers.tick(SETTLE_MS - 1);
        assert.equal(main.bubble(), bubble);
        mock.timers.tick(1);
        assert.deepEqual([main.removed, main.calls.length], [[bubble], 0], 'observing an answer never sends one');
    } finally { main.restore(); }
}));

test('an exposed copy closed elsewhere shows its closure for five seconds, then leaves', () => withTimers(async () => {
    const main = mainColumn();
    try {
        main.decision.appendQuestionPointer(POINTER);
        const bubble = main.bubble(), card = main.card();
        omittedBy(main, WIRE.census_silent);
        await turn();
        heldThrough(main, bubble, 1);
        main.reads[0].resolve(CLOSED_DETAIL);
        await turn();
        assert.deepEqual([status(card), card.querySelector('.chat-quiz-closed-reason')?.textContent, chosen(card)],
            ['Question closed', 'Closed by the owner without an answer.', []]);
        assert.equal(card.querySelector('.chat-quiz-comment'), null, 'a closed copy takes no answer');
        mock.timers.tick(SETTLE_MS - 1);
        assert.equal(main.bubble(), bubble);
        mock.timers.tick(1);
        assert.deepEqual([main.removed, main.calls.length], [[bubble], 0]);
    } finally { main.restore(); }
}));

test('a copy mounted withheld from an older read keeps the suppression: proven answered, it leaves at once', () => withTimers(async () => {
    const main = mainColumn();
    try {
        main.decision.appendQuestionPointer(POINTER, { stale: true });
        const bubble = main.bubble(), card = main.card();
        await turn();
        assert.equal(main.reads.length, 1, 'its own canonical read');
        omittedBy(main, WIRE.census_silent);
        await turn();
        heldThrough(main, bubble, 1);
        assert.equal(card.dataset.state, 'unknown', 'never offered: nothing to answer yet');
        main.reads[0].resolve(DETAIL);
        await turn();
        assert.deepEqual([main.column.children.length, main.removed], [0, [bubble]], 'no five seconds of an answer never offered here');
        assert.notEqual(status(card), 'You answered');
    } finally { main.restore(); }
}));

test('control: a replaced record a complete census omits still leaves at once, without a read', () => withTimers(async () => {
    const main = mainColumn();
    try {
        main.decision.appendQuestionPointer({ ...POINTER, quiz_state: 'superseded' });
        const bubble = main.bubble();
        omittedBy(main, WIRE.census_silent);
        await turn();
        assert.deepEqual([main.column.children.length, main.removed, main.reads.length, main.calls.length], [0, [bubble], 0, 0]);
    } finally { main.restore(); }
}));
