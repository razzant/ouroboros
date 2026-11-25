import assert from 'node:assert/strict';
import test from 'node:test';
import { NodeStub, countPropertyWrites, fixture, frameBubble, turn, WS_MSG } from './chat_decision_fixture.js';
import { questionPresentation } from '../modules/question_presentation.js';
// The head's shared status mark (web/modules/status_mark.js): [shape, hue, hidden].
const markOf = (card) => { const mark = card.querySelector('.chat-quiz-mark'); return [mark.dataset.shape, mark.dataset.hue, mark.hidden]; };
// Settled options: the recorded choice stays inert yet focusable (aria-disabled, out of the tab order) so a Reply
// can land on it (DESIGN "Replies"); every other option is disabled.
const settled = (buttons) => buttons.every((btn) => (btn.classList.contains('chosen')
    ? !btn.disabled && btn.getAttribute('aria-disabled') === 'true' && btn.tabIndex === -1 : btn.disabled));
test('targeted detail preserves normalized option details, single-flight, and the existing answer form', async () => {
    let calls = 0;
    const block = { ...WS_MSG, options: ['Yes', 'No'], option_details: ['Immediate release', 'Wait for CI'], asked_at: WS_MSG.ts };
    const fx = fixture({ fetchDetail: async () => {
        calls += 1;
        return { task_id: 't-1', project_id: 'p1', owner_quiz: { 'qz-1': block } };
    } });
    try {
        const [question, same] = await Promise.all([
            fx.decision.readQuestion('t-1', 'qz-1', 'p1'), fx.decision.readQuestion('t-1', 'qz-1', 'p1'),
        ]);
        assert.equal(calls, 1);
        assert.deepEqual(question, same);
        const card = fx.decision.buildQuizCard(question);
        assert.deepEqual(card.querySelectorAll('.chat-quiz-option-detail').map((node) => node.textContent),
            ['Immediate release', 'Wait for CI']);
        const field = card.querySelector('.chat-quiz-comment');
        field.value = 'keep my draft';
        assert.equal(fx.decision.buildQuizCard({ ...question, ts: 'later-history' }), null);
        assert.equal(card.querySelector('.chat-quiz-comment'), field);
        card.querySelectorAll('.chat-quiz-option')[1].click();
        await new Promise((resolve) => setImmediate(resolve));
        assert.equal(JSON.parse(fx.calls[0].init.body).option_index, 1);
        assert.equal(JSON.parse(fx.calls[0].init.body).comment, 'keep my draft');
        assert.equal(card.dataset.state, 'answered');
        assert.equal(await fx.decision.readQuestion('t-1', 'qz-1', 'wrong-project'), null);
    } finally { fx.restore(); }
});

test('closing a native card with its × keeps focus on the question, not the page', async () => {
    const fx = fixture({ fetchImpl: async () => ({ ok: true, status: 200, json: async () => ({ ok: true, state: 'closed',
        closed_reason: 'Closed by the owner without an answer.', closed_actor: 'owner' }) }) });
    try {
        const card = fx.decision.buildQuizCard({ ...WS_MSG, options: ['Yes', 'No'] });
        const close = card.querySelector('.chat-quiz-close');
        close.focus();
        close.click();
        await turn();
        assert.equal(card.dataset.state, 'closed');
        assert.equal(card.querySelector('.chat-quiz-close'), null, 'the × leaves with the settled state');
        assert.equal(document.activeElement, card.querySelector('.chat-quiz-question'));
        assert.deepEqual(markOf(card), ['dot', 'ink', false], 'a closed question wears the head\'s disabled ink');
    } finally { fx.restore(); }
});

test('a finished task keeps its card answerable and legacy labels disclose missing details', async () => {
    const fx = fixture({ fetchDetail: async () => ({ task_id: 't-1', project_id: 'p1', owner_quiz: {
        'qz-1': { ...WS_MSG, state: 'expired_terminal', options: ['Yes', 'No'] },
    } }) });
    try {
        const question = await fx.decision.readQuestion('t-1', 'qz-1', 'p1');
        const card = fx.decision.buildQuizCard(question);
        assert.ok(card.querySelectorAll('.chat-quiz-option').every((button) => !button.disabled));
        assert.match(card.querySelector('.chat-quiz-details-unavailable').textContent, /not retained/);
    } finally { fx.restore(); }
});

test('a late targeted question read cannot steal a newer navigation or a hidden room', async () => {
    const pending = new Map();
    const fx = fixture({ fetchDetail: (taskId) => new Promise((resolve) => pending.set(taskId, resolve)) });
    const appended = [];
    const append = (msg) => { appended.push(msg.task_id); fx.decision.buildQuizCard(msg); };
    let visible = true;
    const detail = (taskId) => ({ task_id: taskId, project_id: 'p1', owner_quiz: {
        'qz-1': { ...WS_MSG, options: ['Yes', 'No'], option_details: ['', 'Wait for CI'] },
    } });
    try {
        const old = fx.decision.revealQuestion('old', 'qz-1', 'p1', 23, append, () => visible);
        const next = fx.decision.revealQuestion('new', 'qz-1', 'p1', 23, append, () => visible);
        await Promise.resolve();
        pending.get('new')(detail('new'));
        assert.equal(await next, true);
        pending.get('old')(detail('old'));
        assert.equal(await old, false);
        assert.deepEqual(appended, ['new']);
        const hidden = fx.decision.revealQuestion('hidden', 'qz-1', 'p1', 23, append, () => visible);
        await Promise.resolve();
        visible = false;
        pending.get('hidden')(detail('hidden'));
        assert.equal(await hidden, false);
        assert.deepEqual(appended, ['new']);
        // Hidden while its read is held, then shown again by a plain reopen (no question):
        // that showing is the newer navigation, so the late detail cannot act.
        visible = true;
        const reopened = fx.decision.revealQuestion('reopened', 'qz-1', 'p1', 23, append, () => visible);
        await Promise.resolve();
        assert.equal(await fx.decision.revealQuestion('', '', 'p1', 23, append, () => visible), false);
        pending.get('reopened')(detail('reopened'));
        assert.equal(await reopened, false);
        assert.deepEqual(appended, ['new']);
    } finally { fx.restore(); }
});

test('question intent cancels the bookmark before I/O and yields to a later viewport intent', async () => {
    let finish, started = 0, current = true;
    const fx = fixture({ fetchDetail: () => new Promise(resolve => { finish = resolve; }) });
    const appended = [];
    const begin = () => { started++; return () => current; };
    try {
        const pending = fx.decision.revealQuestion('t-1', 'qz-1', 'p1', 23,
            msg => appended.push(msg), () => true, begin);
        assert.equal(started, 1, 'saved restoration is cancelled synchronously');
        await Promise.resolve();
        current = false; // wheel or latest while detail is in flight
        finish({ task_id: 't-1', project_id: 'p1', owner_quiz: { 'qz-1': WS_MSG } });
        assert.equal(await pending, false);
        assert.deepEqual(appended, []);
        current = true;
        const unavailable = fx.decision.revealQuestion('missing', 'qz-1', 'p1', 23,
            msg => appended.push(msg), () => true, begin);
        assert.equal(started, 2);
        await Promise.resolve(); finish(null);
        assert.equal(await unavailable, false);
    } finally { fx.restore(); }
});

test('required question renders waiting identically from live and stored frames', () => {
    const fx = fixture();
    try {
        const live = { ...WS_MSG, quiz_id: 'required-live', wait_for_answer: true, assumption: '' };
        const card = fx.decision.buildQuizCard(live);
        assert.match(card.querySelector('.chat-quiz-assumption').textContent, /Waiting for your answer/);
        assert.deepEqual(markOf(card), ['dot', 'owner', false], 'the owner\'s amber dot while the task waits');
        const stored = { task_id: 't-1', text: live.question, quiz: { ...live, quiz_id: 'required-replay' } };
        const replay = fx.decision.buildQuizCard(stored);
        assert.equal(replay.querySelector('.chat-quiz-assumption').textContent,
                     card.querySelector('.chat-quiz-assumption').textContent);
        assert.deepEqual(markOf(replay), markOf(card));
    } finally { fx.restore(); }
});

test('required waiting copy ends when the answer settles, including history replay', async () => {
    const fx = fixture();
    try {
        const required = { ...WS_MSG, wait_for_answer: true, assumption: '' };
        const card = fx.decision.buildQuizCard(required);
        card.querySelectorAll('.chat-quiz-option')[0].click();
        await new Promise((resolve) => setTimeout(resolve, 0));
        assert.equal(card.dataset.state, 'answered');
        assert.equal(card.querySelector('.chat-quiz-assumption'), null);
        assert.ok(settled(card.querySelectorAll('.chat-quiz-option')));
        assert.deepEqual(markOf(card), ['dot', 'ok', false]);
        // Still unanswered after its task ended, Main's pill counts it: amber, like every question it counts.
        const hues = { answered: 'ok', expired_terminal: 'owner', superseded: 'ink' };
        for (const state of ['answered', 'expired_terminal', 'superseded']) {
            const replay = fx.decision.buildQuizCard({
                task_id: 't-1', text: required.question,
                quiz: { ...required, state, wait_for_answer: false, wait_ended_at: "earlier", quiz_id: `closed-${state}` },
            });
            assert.equal(replay.querySelector('.chat-quiz-assumption'), null);
            assert.deepEqual(markOf(replay), ['dot', hues[state], false], state);
        }
    } finally { fx.restore(); }
});

test('quiz card renders full anatomy from a WS frame', () => {
    const fx = fixture();
    try {
        const card = fx.decision.buildQuizCard(WS_MSG);
        assert.ok(card);
        assert.equal(card.dataset.state, 'open');
        assert.equal(card.querySelector('.chat-quiz-question').textContent, 'Merge now?');
        assert.match(card.querySelector('.chat-quiz-stake').textContent, /At stake: release timing/);
        assert.match(card.querySelector('.chat-quiz-assumption').textContent, /Continuing meanwhile: continuing with the merge/);
        assert.equal(card.querySelector('.chat-quiz-status-text').textContent, 'Unanswered');
        assert.deepEqual(markOf(card), ['dot', 'owner', false], 'unanswered: the pill counts it, so amber');
        const buttons = card.querySelectorAll('.chat-quiz-option');
        assert.equal(buttons.length, 2);
        assert.equal(buttons[1].querySelector('.chat-quiz-option-detail').textContent, 'wait for CI');
        assert.ok(buttons.every((btn) => !btn.disabled));
    } finally { fx.restore(); }
});

test('a finished task leaves its card answerable; only a settled one is a record', async () => {
    const fx = fixture({ fetchImpl: async () => ({
        ok: true, status: 200,
        json: async () => ({ ok: true, state: 'answered', answered_index: 1,
            answered_after_terminal: true, forwarded: true }),
    }) });
    try {
        const replayMsg = {
            msg_type: 'quiz', role: 'assistant', task_id: 't-1',
            text: 'Merge now?', ts: 'x',
            quiz: {
                quiz_id: 'qz-2', state: 'expired_terminal',
                options: [{ label: 'Yes' }, { label: 'No' }],
                stake: '', assumption: 'merging meanwhile',
            },
        };
        const card = fx.decision.buildQuizCard(replayMsg);
        assert.ok(card);
        assert.equal(card.dataset.state, 'expired_terminal');
        assert.equal(card.querySelector('.chat-quiz-status-text').textContent, 'Unanswered · task finished');
        assert.ok(card.querySelectorAll('.chat-quiz-option').every((btn) => !btn.disabled));
        const field = card.querySelector('.chat-quiz-comment');
        assert.ok(field);
        // The assumption line survives settlement: it is the record of the path taken.
        assert.match(card.querySelector('.chat-quiz-assumption').textContent, /merging meanwhile/);

        field.value = 'late but decided';
        card.querySelectorAll('.chat-quiz-option')[1].click();
        await new Promise((resolve) => setTimeout(resolve, 0));
        assert.equal(JSON.parse(fx.calls[0].init.body).option_index, 1);
        assert.equal(JSON.parse(fx.calls[0].init.body).comment, 'late but decided');
        // Answered IS settled: the draft field goes and the buttons close.
        assert.equal(card.dataset.state, 'answered');
        assert.equal(card.querySelector('.chat-quiz-comment'), null);
        assert.ok(settled(card.querySelectorAll('.chat-quiz-option')));

        const superseded = fx.decision.buildQuizCard({
            ...replayMsg, quiz: { ...replayMsg.quiz, quiz_id: 'qz-3', state: 'superseded' },
        });
        assert.ok(superseded.querySelectorAll('.chat-quiz-option').every((btn) => btn.disabled));
        assert.equal(superseded.querySelector('.chat-quiz-comment'), null);
    } finally { fx.restore(); }
});

test('a required card that outlived its task stops claiming the task is waiting', () => {
    const fx = fixture();
    try {
        const card = fx.decision.buildQuizCard({
            ...WS_MSG, quiz_id: 'qz-wait-expired', wait_for_answer: true,
            assumption: '', state: 'expired_terminal',
        });
        assert.equal(card.querySelector('.chat-quiz-wait'), null);
        assert.ok(card.querySelector('.chat-quiz-comment'));
        assert.ok(card.querySelectorAll('.chat-quiz-option').every((btn) => !btn.disabled));
        assert.deepEqual(markOf(card), ['dot', 'owner', false], 'its task ended, the question still waits for the owner');
    } finally { fx.restore(); }
});

test('an accepted answer marks the chosen option; degenerate cards refuse to render', async () => {
    const fx = fixture();
    try {
        const card = fx.decision.buildQuizCard(WS_MSG);
        card.querySelectorAll('.chat-quiz-option')[1].click();
        await new Promise((resolve) => setTimeout(resolve, 0));
        assert.equal(fx.calls.length, 1);
        assert.equal(fx.calls[0].url, '/api/decisions');
        const body = JSON.parse(fx.calls[0].init.body);
        assert.equal(body.decision_id, 'quiz:t-1:qz-1');
        assert.equal(body.option_index, 1);
        assert.ok(body.request_id);
        assert.equal(card.dataset.state, 'answered');
        const buttons = card.querySelectorAll('.chat-quiz-option');
        assert.ok(buttons[1].classList.contains('chosen'));
        assert.ok(settled(buttons));

        assert.equal(fx.decision.buildQuizCard({ ...WS_MSG, quiz_id: 'one', options: [{ label: 'only' }] })
            .querySelectorAll('.chat-quiz-option').length, 1);
        assert.equal(fx.decision.buildQuizCard({ ...WS_MSG, quiz_id: '' }), null);
        // An anonymous quiz has no answer address: refuse to render buttons.
        assert.equal(fx.decision.buildQuizCard({ ...WS_MSG, task_id: '' }), null);
    } finally { fx.restore(); }
});

test('an open question offers a free-text answer without fabricated options', async () => {
    const fx = fixture();
    try {
        const card = fx.decision.buildQuizCard({ ...WS_MSG, quiz_id: 'open', options: [] });
        assert.ok(card);
        assert.equal(card.querySelectorAll('.chat-quiz-option').length, 0);
        const answer = card.querySelector('.chat-quiz-comment');
        assert.ok(answer);
        answer.value = 'I would take a different route.';
        answer.listeners.get('input')();
        card.querySelector('.chat-quiz-send').click();
        await new Promise((resolve) => setTimeout(resolve, 0));
        const body = JSON.parse(fx.calls[0].init.body);
        assert.equal(body.comment, 'I would take a different route.');
        assert.equal(Object.hasOwn(body, 'option_index'), false);
        assert.equal(fx.decision.buildQuizCard({ ...WS_MSG, quiz_id: 'absent', options: undefined }), null);
        assert.equal(fx.decision.buildQuizCard({ ...WS_MSG, quiz_id: 'too-many',
            options: Array.from({ length: 7 }, (_, i) => ({ label: `Choice ${i}` })) }), null);
    } finally { fx.restore(); }
});

test('one corrupt option refuses that card only, preserving index integrity', () => {
    const fx = fixture();
    try {
        // Filtering would shift option_index against the producer's original
        // list and submit a silently WRONG answer once the ingress exists.
        assert.equal(fx.decision.buildQuizCard({ ...WS_MSG, options: [null, null] }), null);
        assert.equal(fx.decision.buildQuizCard({
            ...WS_MSG, options: [null, 'Plain', { label: 'Real' }, { detail: 'no label' }],
        }), null);
        // String options remain a legal producer shorthand.
        const card = fx.decision.buildQuizCard({ ...WS_MSG, options: ['Plain', { label: 'Real' }] });
        assert.ok(card);
        assert.equal(card.querySelectorAll('.chat-quiz-option').length, 2);
    } finally { fx.restore(); }
});

// What the injected Markdown is handed: the question, and the stake after its label. The stub DOM has no text nodes,
// so the label's own span (question_presentation.js mountQuizStake) is pinned by the real renderer instead
// (tests/test_history_reply_browser.py, the Project question's search scene).
test('question and stake reach the injected markdown pipeline', () => {
    const fx = fixture({ renderMarkdown: (text) => `<md>${text}</md>` });
    try {
        const card = fx.decision.buildQuizCard(WS_MSG);
        assert.equal(card.querySelector('.chat-quiz-question').innerHTML, '<md>Merge now?</md>');
        assert.equal(card.querySelector('.chat-quiz-stake').innerHTML, '<md>At stake: release timing</md>');
    } finally { fx.restore(); }
});

test('a second click while the first answer is in flight is ignored', async () => {
    let resolveFetch;
    const fx = fixture({ fetchImpl: () => new Promise((resolve) => { resolveFetch = resolve; }) });
    try {
        const card = fx.decision.buildQuizCard(WS_MSG);
        const buttons = card.querySelectorAll('.chat-quiz-option');
        buttons[0].click();
        buttons[1].click();
        resolveFetch({ ok: true, status: 200, json: async () => ({ ok: true, state: 'answered', answered_index: 0 }) });
        await new Promise((resolve) => setTimeout(resolve, 0));
        assert.equal(fx.calls.length, 1);
        assert.equal(card.dataset.state, 'answered');
    } finally { fx.restore(); }
});

test('a non-409 failure keeps the card open with an honest toast', async () => {
    const fx = fixture({ fetchImpl: async () => ({ ok: false, status: 404 }) });
    try {
        const card = fx.decision.buildQuizCard(WS_MSG);
        card.querySelectorAll('.chat-quiz-option')[0].click();
        await new Promise((resolve) => setTimeout(resolve, 0));
        assert.equal(card.dataset.state, 'open');
        assert.match(fx.toasts[0].text, /Could not record the answer \(404\)/);
        // The pending latch is released: a later click retries.
        card.querySelectorAll('.chat-quiz-option')[0].click();
        await new Promise((resolve) => setTimeout(resolve, 0));
        assert.equal(fx.calls.length, 2);
    } finally { fx.restore(); }
});

test('a 409 settles the card by the BODY state — a lost race reads answered, not expired', async () => {
    // The refusal body carries the true lifecycle state: the loser of a
    // first-wins race must see the winning answer, never a false expiry.
    const fx = fixture({ fetchImpl: async () => ({
        ok: false, status: 409,
        json: async () => ({ ok: false, error: 'quiz_closed', state: 'answered', answered_index: 1 }),
    }) });
    try {
        const card = fx.decision.buildQuizCard(WS_MSG);
        card.querySelectorAll('.chat-quiz-option')[0].click();
        await new Promise((resolve) => setTimeout(resolve, 0));
        assert.equal(card.dataset.state, 'answered');
        assert.ok(card.querySelectorAll('.chat-quiz-option')[1].classList.contains('chosen'));
        assert.match(fx.toasts[0].text, /Already answered/);
    } finally { fx.restore(); }
});

test('a bodyless 409 no longer invents an expiry the card would obey', async () => {
    // An expired card is still answerable, so a refusal without a state is not
    // evidence of anything: report the failed attempt and keep the card as it is.
    const fx = fixture({ fetchImpl: async () => ({ ok: false, status: 409 }) });
    try {
        const card = fx.decision.buildQuizCard(WS_MSG);
        card.querySelectorAll('.chat-quiz-option')[0].click();
        await new Promise((resolve) => setTimeout(resolve, 0));
        assert.equal(card.dataset.state, 'open');
        assert.match(fx.toasts[0].text, /Answer not confirmed/);
        // An unknown outcome forbids a different/automatic second answer.
        card.querySelectorAll('.chat-quiz-option')[0].click();
        await new Promise((resolve) => setTimeout(resolve, 0));
        assert.equal(fx.calls.length, 1);
    } finally { fx.restore(); }
});

test('a retry reuses the SAME request_id (stable idempotency key)', async () => {
    const seen = [];
    let failFirst = true;
    const fx = fixture({ isMain: true, fetchDetail: async () => ({ task_id: 't-1', chat_id: 1, owner_quiz: { 'qz-1': WS_MSG } }), fetchImpl: async (url, init) => {
        seen.push(JSON.parse(init.body).request_id);
        if (failFirst) { failFirst = false; return { ok: false, status: 503, json: async () => ({}) }; }
        return { ok: true, status: 200, json: async () => ({ ok: true, state: 'answered', answered_index: 0 }) };
    } });
    try {
        const card = fx.decision.buildQuizCard(WS_MSG);
        const btn = card.querySelectorAll('.chat-quiz-option')[0];
        btn.click();
        await new Promise((resolve) => setTimeout(resolve, 0));
        assert.equal(card.dataset.state, 'open'); // 503 leaves the card open for retry
        card.querySelector('.chat-quiz-retry').click();
        await new Promise((resolve) => setTimeout(resolve, 0));
        assert.equal(card.dataset.state, 'answered');
        assert.equal(seen.length, 2);
        assert.equal(seen[0], seen[1]);
    } finally { fx.restore(); }
});

test('an answer whose delivery is unknown is amber on a Project question\'s cards and not on Main\'s own', () => {
    const delivery = { kind: 'answer', state: 'owed', reason_code: 'late_answer_delivery_unknown' };
    const answered = { ...WS_MSG, state: 'answered', answered_index: 0, question_delivery: delivery };
    // Main's own question: the pill does not count it, its head is not amber.
    const main = fixture({ isMain: true });
    try {
        assert.deepEqual(markOf(main.decision.buildQuizCard(answered)), ['dot', 'ok', false]);
    } finally { main.restore(); }
    // The same answer of a Project's question, in its own room: the pill counts it, the head is amber.
    const room = fixture({ projectId: 'p1', chatId: 23 });
    try {
        assert.deepEqual(markOf(room.decision.buildQuizCard({ ...answered, project_id: 'p1' })), ['dot', 'owner', false]);
    } finally { room.restore(); }
    // And on its Main copy, which offers Remove from pending.
    const column = new NodeStub();
    const mirror = fixture({ isMain: true, frameNode: frameBubble, insertMessageNode: (node) => { column.append(node); return true; } });
    try {
        mirror.decision.appendQuestionPointer({ role: 'system', system_type: 'project_question_pointer', task_id: 't-1',
            quiz_id: 'qz-1', project_id: 'p1', project_chat_id: 23, project_name: 'Storage', ts: '2026-09-18T22:00:00+00:00',
            quiz_state: 'answered', question: 'Merge now?', options: ['Yes', 'No'], answered_index: 0, question_delivery: delivery });
        const card = column.querySelector('.chat-quiz-card');
        assert.equal(card.querySelector('.chat-quiz-dismiss')?.textContent, 'Remove from pending');
        assert.deepEqual(markOf(card), ['dot', 'owner', false]);
    } finally { mirror.restore(); }
});

test('a Telegram Reply\'s files ride the answered pointer into the Main copy of a Project question as host facts distinct from owner words, and narrower deliveries never blank them', () => {
    const files = [{ name: 'shot.png', kind: 'image', size: 12, sha256: 'a'.repeat(64) },
        { name: 'clip.mp4', kind: 'file', unavailable: 'download_failed' }];
    const column = new NodeStub();
    const mirror = fixture({ isMain: true, frameNode: frameBubble, insertMessageNode: (node) => { column.append(node); return true; } });
    try {
        const row = { role: 'system', system_type: 'project_question_pointer', task_id: 't-1', quiz_id: 'qz-1', project_id: 'p1',
            project_chat_id: 23, project_name: 'Storage', ts: '2026-09-18T22:00:00+00:00', quiz_state: 'answered',
            question: 'Merge now?', options: ['Yes', 'No'], comment: 'this one', attachments: files,
            question_delivery: { kind: 'answer', state: 'owed', reason_code: 'late_answer_delivery_unknown' } };  // owed: the copy stays
        mirror.decision.appendQuestionPointer(row);
        const answer = () => column.querySelector('.chat-quiz-answer')?.textContent;
        const fileLines = () => column.querySelector('.chat-quiz-answer-files')?.textContent;
        const expectedWords = "Owner's answer: this one";
        const expectedFiles = '\u{1F4CE} shot.png\n\u{1F4CE} clip.mp4: not received, the download failed';
        assert.equal(answer(), expectedWords, 'owner words stay separate from host file facts');
        assert.equal(fileLines(), expectedFiles);
        // The census or a lifecycle frame names the same winner without its files: the copy keeps what it shows.
        const narrower = { ...row };
        delete narrower.attachments;
        mirror.decision.appendQuestionPointer(narrower);
        assert.equal(answer(), expectedWords);
        assert.equal(fileLines(), expectedFiles);
        mirror.decision.applyQuizStateFrame({}, { quiz_id: 'qz-1', task_id: 't-1', state: 'answered' });
        assert.equal(answer(), expectedWords);
        assert.equal(fileLines(), expectedFiles);
    } finally { mirror.restore(); }
});

test('a full read without a delivery fact drops an owed observation older than its start; a live one stated since survives', async () => {
    const delivery = { kind: 'answer', state: 'owed', reason_code: 'late_answer_delivery_unknown' };
    const address = 'row:23@2024-04-09T10:14:00+00:00#0123456789ab';
    const row = (extra = {}) => ({ ...WS_MSG, state: 'answered', answered_index: 0, project_id: 'p1', ...extra });
    // A history row is a full read as of `requestedAt`, when its read was asked for (api_client chatHistory).
    const room = fixture({ projectId: 'p1', chatId: 23 });
    try {
        const card = room.decision.buildQuizCard(row({ question_delivery: { ...delivery } }), null, Date.now());
        assert.deepEqual(markOf(card), ['dot', 'owner', false]);
        // A later read of the same row whose block names no delivery: the answer row's own address
        // proves recording, not receipt — the owed observation leaves as unknown, never delivered.
        assert.equal(room.decision.buildQuizCard(row({ answer_row_address: address }), null, Date.now()), null);
        assert.equal(room.decision.observedQuestion('t-1', 'qz-1').question_delivery, undefined);
        assert.deepEqual(markOf(card), ['dot', 'ok', false]);
        // Delivered is durable: a later full read without the fact never downgrades it to unknown.
        room.decision.buildQuizCard({ ...row({ answer_row_address: address }), quiz_id: 'qz-2',
            question_delivery: { kind: 'answer', state: 'delivered' } }, null, Date.now());
        assert.equal(room.decision.buildQuizCard(row({ quiz_id: 'qz-2', answer_row_address: address }), null, Date.now()), null);
        assert.equal(room.decision.observedQuestion('t-1', 'qz-2').question_delivery.state, 'delivered');
        // A live question frame, a copy's repaint or a row no read dates is partial: its missing fact decides nothing.
        room.decision.buildQuizCard({ ...row({ question_delivery: { ...delivery } }), quiz_id: 'qz-6' }, null, Date.now());
        assert.equal(room.decision.buildQuizCard(row({ quiz_id: 'qz-6' })), null);
        assert.equal(room.decision.observedQuestion('t-1', 'qz-6').question_delivery.state, 'owed');
        // A closure's owed delivery follows the same rule.
        room.decision.buildQuizCard({ ...row({ question_delivery: { kind: 'closure', state: 'owed' } }),
            quiz_id: 'qz-5', state: 'closed', closed_reason: 'Closed by the owner without an answer.' }, null, Date.now());
        assert.equal(room.decision.observedQuestion('t-1', 'qz-5').question_delivery.state, 'owed');
        room.decision.buildQuizCard(row({ quiz_id: 'qz-5', state: 'closed', closed_reason: 'Closed by the owner without an answer.' }),
            null, Date.now());
        assert.equal(room.decision.observedQuestion('t-1', 'qz-5').question_delivery, undefined);
    } finally { room.restore(); }
    // Main's own card, open in the feed, and the two orders of one live delivery and one canonical read of its record
    // whose block names none; the read's start is its own (readQuestion), its answer arrives after the frame both times.
    const replies = [];
    const main = fixture({ isMain: true, fetchDetail: () => new Promise((resolve) => replies.push(resolve)) });
    const record = (quizId) => ({ task_id: 't-1', chat_id: 1,
        owner_quiz: { [quizId]: { ...WS_MSG, quiz_id: quizId, state: 'answered', answered_index: 0, asked_at: WS_MSG.ts } } });
    const liveOwed = (quizId) => main.decision.applyQuizStateFrame(new NodeStub(), { ...WS_MSG, quiz_id: quizId,
        state: 'answered', answered_index: 0, question_delivery: { ...delivery } });
    const statusOf = (card) => card.querySelector('.chat-quiz-status-text').textContent;
    try {
        const [first, second] = ['qz-7', 'qz-8'].map((quizId) => main.decision.buildQuizCard({ ...WS_MSG, quiz_id: quizId,
            state: 'answered', answered_index: 0, question_delivery: { ...delivery } }, null, Date.now()));
        assert.deepEqual([statusOf(first), statusOf(second)], Array(2).fill('Answer recorded · delivery unknown'));
        // 1. The live frame came first; a read asked for after it finds no delivery: that older owed fact goes.
        liveOwed('qz-7');
        const stated = Date.now();
        while (Date.now() <= stated) await new Promise((resolve) => setTimeout(resolve, 1));
        const early = main.decision.readQuestion('t-1', 'qz-7', '', { fresh: true });
        await turn();
        replies.shift()(record('qz-7'));
        assert.equal((await early).question_delivery, undefined);
        assert.equal(main.decision.observedQuestion('t-1', 'qz-7').question_delivery, undefined);
        assert.equal(statusOf(first), 'You answered', 'the open card drops its stale pending words');
        // 2. A read already asked for when the live frame came answers late without it: the newer fact stays.
        const late = main.decision.readQuestion('t-1', 'qz-8', '', { fresh: true });
        await turn();
        liveOwed('qz-8');
        replies.shift()(record('qz-8'));
        assert.equal((await late).question_delivery.state, 'owed');
        assert.equal(main.decision.observedQuestion('t-1', 'qz-8').question_delivery.state, 'owed');
        assert.equal(statusOf(second), 'Answer recorded · delivery unknown', 'the late snapshot clears no newer fact');
    } finally { main.restore(); }
    // A census pointer is partial: its missing fact changes nothing, while the canonical read drops the owed
    // observation — Main's copy keeps its Remove control until that read, never by the pointer alone.
    const column = new NodeStub();
    const answers = [];
    const mirror = fixture({ isMain: true, frameNode: frameBubble, insertMessageNode: (node) => { column.append(node); return true; },
        fetchDetail: () => new Promise((resolve) => answers.push(resolve)) });
    const canonicalRecord = { task_id: 't-1', project_id: 'p1',
        owner_quiz: { 'qz-4': { ...WS_MSG, quiz_id: 'qz-4', state: 'answered', answered_index: 0, asked_at: WS_MSG.ts } },
        question_rooms: { 'qz-4': { state: 'known', chat_id: 23, project_id: 'p1' } } };
    const past = async (at) => { while (Date.now() <= at) await new Promise((resolve) => setTimeout(resolve, 1)); };
    try {
        const pointer = (extra = {}) => ({ role: 'system', system_type: 'project_question_pointer', task_id: 't-1',
            quiz_id: 'qz-4', project_id: 'p1', project_chat_id: 23, project_name: 'Storage', ts: '2026-09-18T22:00:00+00:00',
            quiz_state: 'answered', question: 'Merge now?', options: ['Yes', 'No'], answered_index: 0,
            question_delivery: { ...delivery }, ...extra });
        mirror.decision.appendQuestionPointer(pointer());
        const card = column.querySelector('.chat-quiz-card');
        assert.equal(card.querySelector('.chat-quiz-dismiss')?.textContent, 'Remove from pending');
        const partial = pointer();
        delete partial.question_delivery;
        mirror.decision.appendQuestionPointer(partial);
        assert.equal(card.querySelector('.chat-quiz-dismiss')?.textContent, 'Remove from pending');
        // An earlier live frame states the debt; the canonical read is asked for after it, and while it waits a raw
        // partial frame (answered, no delivery fact) arrives: the copy is observed live then, the debt it repaints is not.
        const frame = { ...WS_MSG, quiz_id: 'qz-4', state: 'answered', answered_index: 0 };
        mirror.decision.applyQuizStateFrame(new NodeStub(), { ...frame, question_delivery: { ...delivery } });
        await past(Date.now());
        const reading = mirror.decision.readQuestion('t-1', 'qz-4', 'p1');
        await turn();
        const censusAskedAt = Date.now();
        await past(censusAskedAt);
        mirror.decision.applyQuizStateFrame(new NodeStub(), frame);
        mirror.decision.syncCurrentQuestions([], () => true, censusAskedAt);
        await turn();
        assert.equal(answers.length, 1, 'a census asked for before that frame reads no copy observed live since');
        answers.shift()(canonicalRecord);
        const canonical = await reading;
        assert.equal(canonical.question_delivery, undefined);
        // One source: the copy's row loses that delivery with its observation, so its next repaint (here a renamed
        // Project) neither brings the owed fact back through mirrorQuiz nor keeps Remove from pending.
        mirror.decision.appendQuestionPointer({ ...partial, project_name: 'Storage room' });
        assert.equal(mirror.decision.observedQuestion('t-1', 'qz-4').question_delivery, undefined);
        assert.equal(card.querySelector('.chat-quiz-dismiss'), null);
    } finally { mirror.restore(); }
});

test('a full read asked for before a live delivery never replaces it: neither by the older fact it states nor by its silence', async (t) => {
    const unknown = { kind: 'answer', state: 'owed', reason_code: 'late_answer_delivery_unknown' };
    const saved = { kind: 'answer', state: 'owed', reason_code: 'awaiting_recipient_read' };
    const pending = { kind: 'answer', state: 'owed' };
    const words = { unknown: 'Answer recorded · delivery unknown', saved: 'Answer saved · the task reads it when it continues',
        pending: 'Answer recorded · delivery pending' };
    const statusOf = (card) => card.querySelector('.chat-quiz-status-text').textContent;
    const later = async () => { const at = Date.now(); while (Date.now() <= at) await new Promise((resolve) => setTimeout(resolve, 1)); };
    // Main's own cards; each canonical read is a full read as of its own start (readQuestion) and answers last.
    const replies = [];
    const main = fixture({ isMain: true, fetchDetail: () => new Promise((resolve) => replies.push(resolve)) });
    const record = (quizId, delivery) => ({ task_id: 't-1', chat_id: 1, owner_quiz: { [quizId]: { ...WS_MSG, quiz_id: quizId,
        state: 'answered', answered_index: 0, asked_at: WS_MSG.ts, ...(delivery ? { delivery: { ...delivery } } : {}) } } });
    const live = (quizId, delivery) => main.decision.applyQuizStateFrame(new NodeStub(), { ...WS_MSG, quiz_id: quizId,
        state: 'answered', answered_index: 0, question_delivery: { ...delivery } });
    const ask = (quizId) => main.decision.readQuestion('t-1', quizId, '', { fresh: true });
    const card = (quizId, delivery) => main.decision.buildQuizCard({ ...WS_MSG, quiz_id: quizId, state: 'answered',
        answered_index: 0, question_delivery: { ...delivery } }, null, Date.now());
    try {
        // 1. A new unknown, and 2. a new saved answer, each stated live while the read waits: the read's older pending
        // fact keeps neither from the card (Remove from pending of a Project's copy rides on the first, below).
        for (const [quizId, fresh] of [['qz-11', unknown], ['qz-12', saved]]) {
            const shown = card(quizId, pending);
            assert.equal(statusOf(shown), words.pending);
            const reading = ask(quizId);
            await turn();
            live(quizId, fresh);
            replies.shift()(record(quizId, pending));
            assert.deepEqual((await reading).question_delivery, fresh, 'the read hands on the newer fact, never its own');
            assert.deepEqual(main.decision.observedQuestion('t-1', quizId).question_delivery, fresh);
            assert.equal(statusOf(shown), fresh === unknown ? words.unknown : words.saved);
        }
        // 3. Equal moments: a delivery stated live in the same millisecond the read was asked for is still the newer.
        const sameMoment = card('qz-13', pending);
        const clock = t.mock.method(Date, 'now', () => 1_800_000_000_000);
        const reading = ask('qz-13');
        await turn();
        live('qz-13', unknown);
        clock.mock.restore();
        replies.shift()(record('qz-13', saved));
        await reading;
        assert.equal(statusOf(sameMoment), words.unknown);
        // 4. Two reads around one live frame: the newer read finds no delivery and the frame's fact goes; the older one,
        // answering last with its pending fact, brings back nothing — not through what it hands on either.
        const both = card('qz-14', pending);
        const older = ask('qz-14');
        await turn();
        await later();
        live('qz-14', unknown);
        await later();
        const newer = ask('qz-14');
        await turn();
        replies.pop()(record('qz-14'));
        assert.equal((await newer).question_delivery, undefined);
        assert.equal(statusOf(both), 'You answered');
        replies.shift()(record('qz-14', pending));
        assert.equal((await older).question_delivery, undefined);
        assert.equal(main.decision.observedQuestion('t-1', 'qz-14').question_delivery, undefined);
        assert.equal(statusOf(both), 'You answered');
        // 5. A read asked for after the live frame is the newer fact itself, whatever it states.
        const replaced = card('qz-15', pending);
        live('qz-15', unknown);
        await later();
        const after = ask('qz-15');
        await turn();
        replies.shift()(record('qz-15', saved));
        assert.deepEqual((await after).question_delivery, saved);
        assert.equal(statusOf(replaced), words.saved);
    } finally { main.restore(); }
    // A Project question's Main copy, revalidated by a census that vouches for it: the read is asked for before the live
    // unknown, and its older saved fact neither replaces the words nor takes Remove from pending away.
    const column = new NodeStub();
    const answers = [];
    const mirror = fixture({ isMain: true, frameNode: frameBubble, insertMessageNode: (node) => { column.append(node); return true; },
        fetchDetail: () => new Promise((resolve) => answers.push(resolve)) });
    try {
        mirror.decision.appendQuestionPointer({ role: 'system', system_type: 'project_question_pointer', task_id: 't-1',
            quiz_id: 'qz-4', project_id: 'p1', project_chat_id: 23, project_name: 'Storage', ts: '2026-09-18T22:00:00+00:00',
            quiz_state: 'answered', question: 'Merge now?', options: ['Yes', 'No'], answered_index: 0, question_delivery: { ...saved } });
        const copy = column.querySelector('.chat-quiz-card');
        assert.equal(statusOf(copy), words.saved);
        assert.equal(copy.querySelector('.chat-quiz-dismiss'), null);
        await later();
        mirror.decision.syncCurrentQuestions([], () => true, Date.now());
        await turn();
        assert.equal(answers.length, 1, 'the vouched-for copy reads its own record');
        mirror.decision.applyQuizStateFrame(new NodeStub(), { ...WS_MSG, quiz_id: 'qz-4', state: 'answered', answered_index: 0,
            question_delivery: { ...unknown } });
        assert.equal(copy.querySelector('.chat-quiz-dismiss')?.textContent, 'Remove from pending');
        answers.shift()({ task_id: 't-1', project_id: 'p1', owner_quiz: { 'qz-4': { ...WS_MSG, quiz_id: 'qz-4', state: 'answered',
            answered_index: 0, asked_at: WS_MSG.ts, delivery: { ...saved } } },
            question_rooms: { 'qz-4': { state: 'known', chat_id: 23, project_id: 'p1' } } });
        await turn();
        assert.equal(statusOf(copy), words.unknown);
        assert.equal(copy.querySelector('.chat-quiz-dismiss')?.textContent, 'Remove from pending');
    } finally { mirror.restore(); }
});

test('a private Presence question is Main\'s wherever it is drawn: its unknown delivery is not amber in its work\'s Project', async () => {
    const delivery = { kind: 'answer', state: 'owed', reason_code: 'late_answer_delivery_unknown' };
    const answered = { ...WS_MSG, state: 'answered', answered_index: 0, question_delivery: delivery };
    // From history: the row names its work's Project only as a reference; its contact says it is Main's.
    const history = fixture({ projectId: 'p1', chatId: 23 });
    try {
        const card = history.decision.buildQuizCard({ ...answered, project_id: 'p1',
            presence_contact: { task_id: 't-1', binding_id: 'binding', source_event_id: 'event' } });
        assert.deepEqual(markOf(card), ['dot', 'ok', false]);
    } finally { history.restore(); }
    // Read canonically first (Main's own room, shown in the Project too): building the room's card keeps that fact.
    const block = { ...WS_MSG, state: 'answered', answered_index: 0, asked_at: WS_MSG.ts,
        delivery: { kind: 'answer', state: 'owed', reason_code: 'late_answer_delivery_unknown' } };
    const read = fixture({ projectId: 'p1', chatId: 23, fetchDetail: async () => ({ task_id: 't-1', owner_quiz: { 'qz-1': block },
        question_rooms: { 'qz-1': { state: 'known', chat_id: 1, project_id: '', also_project_ids: ['p1'] } } }) });
    try {
        const question = await read.decision.readQuestion('t-1', 'qz-1', 'p1');
        assert.equal(question?.project_id, '');
        assert.deepEqual(markOf(read.decision.buildQuizCard({ ...answered, project_id: 'p1' })), ['dot', 'ok', false]);
    } finally { read.restore(); }
    // An ordinary question of that Project drawn in its room stays amber, as the pill counts it.
    const own = fixture({ projectId: 'p1', chatId: 23 });
    try {
        assert.deepEqual(markOf(own.decision.buildQuizCard(answered)), ['dot', 'owner', false]);
    } finally { own.restore(); }
});

test('applyQuizStateFrame settles an existing card and ignores unknown ids', async () => {
    const fx = fixture({ fetchImpl: async () => ({ ok: true, status: 200 }) });
    try {
        const card = fx.decision.buildQuizCard(WS_MSG);
        const quizCard = card;
        const root = {};
        // The live lifecycle frame (WS quiz_state) settles the card in place.
        const applied = fx.decision.applyQuizStateFrame(root, {
            quiz_id: 'qz-1', task_id: 't-1', state: 'answered', answered_index: 0,
        });
        assert.equal(applied, true);
        assert.equal(quizCard.dataset.state, 'answered');
        const buttons = quizCard.querySelectorAll('chat-quiz-option');
        assert.ok(buttons.length >= 2);
        assert.ok(settled(buttons));
        assert.ok(buttons[0].classList.contains('chosen'));
        const stateWrites = countPropertyWrites(quizCard.dataset, 'state');
        const disabledWrites = buttons.map((btn) => countPropertyWrites(btn, 'disabled'));
        let chosenWrites = 0;
        for (const btn of buttons) {
            const toggle = btn.classList.toggle.bind(btn.classList);
            btn.classList.toggle = (...args) => { chosenWrites += 1; return toggle(...args); };
        }
        assert.equal(fx.decision.applyQuizStateFrame(root, {
            quiz_id: 'qz-1', task_id: 't-1', state: 'answered', answered_index: 0,
        }), false, 'an identical acknowledgement performs no DOM mutation');
        assert.equal(stateWrites(), 0);
        assert.ok(disabledWrites.every((writes) => writes() === 0));
        assert.equal(chosenWrites, 0);

        // Unknown id: no card found, nothing thrown, honest false.
        assert.equal(fx.decision.applyQuizStateFrame(root, { quiz_id: 'other', state: 'answered' }), false);
    } finally {
        fx.restore();
    }
});

// The owner's recorded answer names itself (question_presentation.paintChosenAnswer): after its label a spoken name and an
// aria-hidden word, both `Your choice`; ui.css shows the word only where the theme draws it.
const CHOSEN = [['chat-quiz-option-chosen-name', 'chat-quiz-option-chosen-cue'], 'Your choice', 'Your choice', 'true'];
const chosenParts = (card) => card.querySelectorAll('.chat-quiz-option').map((button) => {
    const name = button.querySelector('.chat-quiz-option-chosen-name');
    const cue = button.querySelector('.chat-quiz-option-chosen-cue');
    if (!name && !cue) return null;
    const after = button.children.map((child) => child.className);
    const label = after.indexOf('chat-quiz-option-label');
    return [after.slice(label + 1, label + 3), name?.textContent, cue?.textContent, cue?.getAttribute('aria-hidden')];
});

test('the recorded answer says `Your choice` after its label, moves with a corrected record and repaints nothing unchanged', () => {
    const fx = fixture();
    try {
        const card = fx.decision.buildQuizCard({ ...WS_MSG, options: [{ label: 'Yes', recommended: true }, { label: 'No', detail: 'wait for CI' }] });
        const root = {};
        assert.deepEqual(chosenParts(card), [null, null], 'an open question names no choice');
        assert.equal(fx.decision.applyQuizStateFrame(root, { quiz_id: 'qz-1', task_id: 't-1', state: 'answered', answered_index: 1 }), true);
        assert.deepEqual(chosenParts(card), [null, CHOSEN]);
        const no = card.querySelectorAll('.chat-quiz-option')[1];
        assert.deepEqual(no.children.map((child) => child.className),
            ['chat-quiz-option-label', 'chat-quiz-option-chosen-name', 'chat-quiz-option-chosen-cue', 'chat-quiz-option-detail']);
        const cue = no.querySelector('.chat-quiz-option-chosen-cue');
        assert.equal(fx.decision.applyQuizStateFrame(root, { quiz_id: 'qz-1', task_id: 't-1', state: 'answered', answered_index: 1 }), false,
            'the same record writes nothing');
        assert.equal(no.querySelector('.chat-quiz-option-chosen-cue'), cue);
        // The canonical record names another option: the words leave the old one and follow the new one.
        assert.equal(fx.decision.applyQuizStateFrame(root, { quiz_id: 'qz-1', task_id: 't-1', state: 'answered', answered_index: 0 }), true);
        assert.deepEqual(chosenParts(card), [CHOSEN, null]);
        assert.deepEqual(card.querySelectorAll('.chat-quiz-option')[0].children.map((child) => child.className),
            ['chat-quiz-option-label', 'chat-quiz-option-chosen-name', 'chat-quiz-option-chosen-cue', 'chat-quiz-option-recommended'],
            'the owner\'s choice before the asker\'s recommendation');
    } finally { fx.restore(); }
});

test('a comment-only answer and an answer still being sent name no chosen option', async () => {
    let finish;
    const fx = fixture({ fetchImpl: () => new Promise((resolve) => { finish = resolve; }) });
    try {
        const own = fx.decision.buildQuizCard({ ...WS_MSG, quiz_id: 'qz-own' });
        fx.decision.applyQuizStateFrame({}, { quiz_id: 'qz-own', task_id: 't-1', state: 'answered', comment: 'neither — use duckdb' });
        assert.equal(own.dataset.state, 'answered');
        assert.deepEqual(chosenParts(own), [null, null], 'the owner\'s own words chose no option');
        const card = fx.decision.buildQuizCard(WS_MSG);
        const button = card.querySelectorAll('.chat-quiz-option')[1];
        button.click();
        assert.ok(button.classList.contains('pending-choice'));
        assert.deepEqual(chosenParts(card), [null, null], 'a pending choice is not a recorded one');
        finish({ ok: true, status: 200, json: async () => ({ ok: true, state: 'answered', answered_index: 1 }) });
        await turn();
        assert.equal(card.dataset.state, 'answered');
        assert.deepEqual(chosenParts(card), [null, CHOSEN], 'the recorded answer names itself once confirmed');
    } finally { fx.restore(); }
});

test('a live quiz_state frame carries the owner comment onto the card like replay does (#471)', () => {
    const fx = fixture();
    try {
        const card = fx.decision.buildQuizCard(WS_MSG);
        const quizCard = card;
        const root = { querySelector: (sel) => (sel.includes('qz-1') ? quizCard : null) };
        // The owner rejected every option and answered in their own words: the
        // frame carries no answered_index and the recorded comment.
        assert.equal(fx.decision.applyQuizStateFrame(root, {
            quiz_id: 'qz-1', task_id: 't-1', state: 'answered', comment: 'neither — use duckdb',
        }), true);
        assert.equal(quizCard.dataset.state, 'answered');
        assert.equal(quizCard.dataset.ownerComment, 'neither — use duckdb');
        assert.equal(quizCard.querySelector('.chat-quiz-answer').textContent,
            "Owner's answer: neither — use duckdb");
        assert.ok(quizCard.querySelectorAll('.chat-quiz-option').every((btn) => !btn.classList.contains('chosen')));
        // A later lifecycle frame without a comment never wipes the recorded one.
        fx.decision.applyQuizStateFrame(root, { quiz_id: 'qz-1', task_id: 't-1', state: 'superseded' });
        assert.equal(quizCard.dataset.ownerComment, 'neither — use duckdb');
        assert.equal(quizCard.dataset.state, 'superseded');
    } finally {
        fx.restore();
    }
});

// ---- free answer (the owner's own option) ----

function commentParts(card) {
    return {
        field: card.querySelector('.chat-quiz-comment'),
        send: card.querySelector('.chat-quiz-send'),
    };
}

test('a typed remark rides WITH an option click as one answer', async () => {
    const fx = fixture();
    try {
        const card = fx.decision.buildQuizCard(WS_MSG);
        const { field, send } = commentParts(card);
        assert.ok(field && send);
        field.value = 'yes, but only after CI';
        field.listeners.get('input')();
        assert.equal(send.disabled, false);
        card.querySelectorAll('.chat-quiz-option')[0].click();
        await new Promise((resolve) => setTimeout(resolve, 0));
        const body = JSON.parse(fx.calls[0].init.body);
        assert.equal(body.option_index, 0);
        assert.equal(body.comment, 'yes, but only after CI');
        // Settled: the draft field is gone and the words are the record.
        assert.equal(card.querySelector('.chat-quiz-comment-box'), null);
        assert.equal(card.querySelector('.chat-quiz-answer').textContent,
            "Owner's answer: yes, but only after CI");
    } finally { fx.restore(); }
});

test('the send button answers WITHOUT an option_index', async () => {
    const fx = fixture();
    try {
        const card = fx.decision.buildQuizCard(WS_MSG);
        const { field, send } = commentParts(card);
        field.value = '  neither — do C  ';
        field.listeners.get('input')();
        send.click();
        await new Promise((resolve) => setTimeout(resolve, 0));
        assert.equal(fx.calls.length, 1);
        const body = JSON.parse(fx.calls[0].init.body);
        assert.equal('option_index' in body, false, 'no option was taken — the key must be absent');
        // VERBATIM: the owner's exact characters, edges included.
        assert.equal(body.comment, '  neither — do C  ');
        assert.equal(card.dataset.state, 'answered');
        // No option is highlighted: the owner chose none of them.
        assert.ok(card.querySelectorAll('.chat-quiz-option').every((btn) => !btn.classList.contains('chosen')));
    } finally { fx.restore(); }
});

test('an empty field leaves the send button disabled and sends nothing', async () => {
    const fx = fixture();
    try {
        const card = fx.decision.buildQuizCard(WS_MSG);
        const { field, send } = commentParts(card);
        assert.equal(send.disabled, true);
        field.value = '   ';
        field.listeners.get('input')();
        assert.equal(send.disabled, true);
        send.click();
        await new Promise((resolve) => setTimeout(resolve, 0));
        assert.equal(fx.calls.length, 0);
        assert.equal(card.dataset.state, 'open');
    } finally { fx.restore(); }
});

test('an over-long answer is refused client-side, not truncated', async () => {
    const fx = fixture();
    try {
        const card = fx.decision.buildQuizCard(WS_MSG);
        const { field, send } = commentParts(card);
        field.value = 'x'.repeat(2001);
        field.listeners.get('input')();
        assert.equal(send.disabled, true, 'the cap is the ingress limit, mirrored');
        send.click();
        await new Promise((resolve) => setTimeout(resolve, 0));
        assert.equal(fx.calls.length, 0);
        assert.match(fx.toasts[0].text, /under 2000 characters/);
        assert.equal(card.dataset.state, 'open');
    } finally { fx.restore(); }
});

const pressEnter = (field, init = {}) => {
    const event = { key: 'Enter', preventDefault() { event.defaultPrevented = true; }, ...init };
    field.listeners.get('keydown')(event);
    return event;
};

test('Enter sends an optionless answer through Send; Shift+Enter, composition and an empty field do not', async () => {
    const fx = fixture();
    try {
        const card = fx.decision.buildQuizCard({ ...WS_MSG, options: [] });
        const { field } = commentParts(card);
        assert.equal(field.enterKeyHint, 'send');
        assert.equal(pressEnter(field).defaultPrevented, true, 'an empty answer neither sends nor becomes a line');
        field.value = 'Friday';
        field.listeners.get('input')();
        assert.equal(pressEnter(field, { shiftKey: true }).defaultPrevented, undefined, 'Shift+Enter is the line break');
        field.value = 'Friday\nand say it is provisional';
        field.listeners.get('input')();
        assert.equal(pressEnter(field, { isComposing: true }).defaultPrevented, undefined);
        assert.equal(pressEnter(field, { keyCode: 229 }).defaultPrevented, undefined, 'WebKit commits IME this way');
        await turn();
        assert.equal(fx.calls.length, 0);
        pressEnter(field);
        await turn();
        const body = JSON.parse(fx.calls[0].init.body);
        assert.deepEqual([fx.calls.length, body.comment, 'option_index' in body], [1, 'Friday\nand say it is provisional', false]);
        assert.equal(card.dataset.state, 'answered');
        assert.equal(card.querySelector('.chat-quiz-comment-box'), null);
        assert.equal(pressEnter(field).defaultPrevented, undefined, 'the settled card takes no key');
        await turn();
        assert.equal(fx.calls.length, 1);
    } finally { fx.restore(); }
});

test('held Enter and an Enter/click race record one answer; a refusal keeps the draft for the same retry', async () => {
    const replies = [];
    const fx = fixture({ isMain: true, fetchDetail: async () => ({ task_id: 't-1', chat_id: 1, owner_quiz: { 'qz-1': WS_MSG } }),
        fetchImpl: () => new Promise((resolve) => replies.push(resolve)) });
    try {
        const card = fx.decision.buildQuizCard(WS_MSG);
        const { field, send } = commentParts(card);
        field.value = 'wait for CI, then merge';
        field.listeners.get('input')();
        pressEnter(field);
        assert.equal(pressEnter(field, { repeat: true }).defaultPrevented, undefined, 'the submitted field is read-only while confirmation is unknown');
        pressEnter(field);
        send.click();
        await turn();
        assert.equal(fx.calls.length, 1, "the card's pending answer is the one gate");
        replies.shift()({ ok: false, status: 503, json: async () => ({}) });
        await turn();
        assert.deepEqual([card.dataset.state, fx.toasts.at(-1).text], ['open', 'Answer not confirmed. Your original answer is kept.']);
        assert.equal(card.querySelector('.chat-quiz-comment'), field);
        assert.equal(field.value, 'wait for CI, then merge', 'the draft survives the refusal');
        card.querySelector('.chat-quiz-retry').click();
        await turn();
        const [first, retry] = fx.calls.map((call) => JSON.parse(call.init.body));
        assert.deepEqual([fx.calls.length, retry.request_id], [2, first.request_id], 'the retry replays the same request');
        replies.shift()({ ok: true, status: 200, json: async () => ({ ok: true, state: 'answered', comment: retry.comment }) });
        await turn();
        assert.equal(card.dataset.state, 'answered');
    } finally { fx.restore(); }
});

test('a settled replayed card shows the owner answer and offers no field', () => {
    const fx = fixture();
    try {
        const card = fx.decision.buildQuizCard({
            msg_type: 'quiz', role: 'assistant', task_id: 't-1', text: 'Merge now?', ts: 'x',
            quiz: {
                quiz_id: 'qz-3', state: 'answered', stake: '', assumption: 'merging meanwhile',
                options: [{ label: 'Yes' }, { label: 'No' }],
                comment: 'neither — revert first',
            },
        });
        assert.equal(card.querySelector('.chat-quiz-comment-box'), null);
        assert.equal(card.querySelector('.chat-quiz-answer').textContent,
            "Owner's answer: neither — revert first");
        assert.ok(card.querySelectorAll('.chat-quiz-option').every((btn) => btn.disabled));
        assert.ok(card.querySelectorAll('.chat-quiz-option').every((btn) => !btn.classList.contains('chosen')));
        const replay = { ...WS_MSG, quiz_id: 'qz-3', state: 'answered' };
        fx.decision.buildQuizCard(replay);
        assert.match(card.querySelector('.chat-quiz-answer').textContent, /neither — revert first/);
        fx.decision.buildQuizCard({ ...replay, answered_index: 0, comment: '' });
        assert.equal(card.querySelector('.chat-quiz-answer'), null);
        assert.ok(card.querySelectorAll('.chat-quiz-option')[0].classList.contains('chosen'));
    } finally { fx.restore(); }
});

test('a live state frame for an open card never wipes the typed draft', () => {
    const fx = fixture();
    try {
        const card = fx.decision.buildQuizCard(WS_MSG);
        const { field } = commentParts(card);
        field.value = 'half-written thought';
        field.listeners.get('input')();
        const root = { querySelector: () => card };
        assert.equal(fx.decision.applyQuizStateFrame(root, {
            quiz_id: 'qz-1', task_id: 't-1', state: 'open',
        }), false, 'an open→open frame is a no-op');
        assert.equal(card.querySelector('.chat-quiz-comment').value, 'half-written thought');
        assert.ok(card.querySelector('.chat-quiz-comment-box'));
    } finally { fx.restore(); }
});

// ---- routing picker (#198) ----

function routingBubble(cmid = 'cm-1') {
    const bubble = new NodeStub('div');
    bubble.dataset.clientMessageId = cmid;
    return bubble;
}

const ROUTING_ANNOTATION = {
    status: 'needs_manual_target', routing_token: 'tok-1',
    options: [
        { action: 'steer_task', task_id: 't1', title: 'Fix CI' },
        { action: 'new_task_in_project', project_id: 'p1', project_name: 'Web' },
    ],
};

test('an actionable refusal renders the picker card; other statuses fall back to text', () => {
    const fx = fixture();
    try {
        const bubble = routingBubble();
        assert.equal(fx.decision.renderRoutingDecision(bubble, ROUTING_ANNOTATION), true);
        assert.equal(fx.decision.renderRoutingDecision(bubble, ROUTING_ANNOTATION), false);
        const card = bubble.querySelector('.chat-routing-card');
        assert.ok(card);
        assert.equal(card.dataset.state, 'open');
        assert.deepEqual(markOf(card), ['dot', 'owner', false], 'a destination only the owner can choose is the owner\'s move');
        const buttons = card.querySelectorAll('.chat-quiz-option');
        assert.equal(buttons.length, 2);
        assert.equal(buttons[0].querySelector('.chat-quiz-option-label').textContent, 'Fix CI');
        assert.equal(buttons[1].querySelector('.chat-quiz-option-label').textContent, 'New task in Web');
        // A later settled annotation (the dispatch ack) REPLACES the card
        // with the plain text line — the card never lingers past its attempt.
        const settled = {
            status: 'delivered', action: 'steer_task', target: 't1', target_label: 'Fix CI',
        };
        assert.equal(fx.decision.renderRoutingDecision(bubble, settled), true);
        const note = bubble.querySelector('.msg-routing-annotation');
        const textWrites = countPropertyWrites(note, 'textContent');
        const noteStatusWrites = countPropertyWrites(note.dataset, 'annotationStatus');
        const bubbleStatusWrites = countPropertyWrites(bubble.dataset, 'chatAnnotationStatus');
        assert.equal(fx.decision.renderRoutingDecision(bubble, settled), false);
        assert.deepEqual(
            [textWrites(), noteStatusWrites(), bubbleStatusWrites()], [0, 0, 0],
        );
        assert.equal(bubble.querySelector('.chat-routing-card'), null);
        assert.match(bubble.querySelector('.msg-routing-annotation').textContent, /Steered task/);
        assert.equal(fx.decision.renderRoutingDecision(bubble, null), true);
        assert.equal(bubble.querySelector('.msg-routing-annotation'), null);
        assert.equal(bubble.dataset.chatAnnotationStatus, undefined);
        assert.equal(fx.decision.renderRoutingDecision(bubble, null), false);
    } finally { fx.restore(); }
});

test('a refusal with a token but no options renders the host cause as the plain line, never a card', () => {
    // The incident shape (client_message_id + routing_token, NO options): the
    // host sends `cause`, the owner reads a sentence, and nothing is clickable.
    const cause = 'Not started: the working folder can\'t be used';
    const fx = fixture();
    try {
        const bubble = routingBubble('cm-refused');
        const refused = {
            action: 'promote_chat_to_task', status: 'needs_manual_target', routing_token: 'tok-r',
            target: 'never-started', target_label: 'Аудит', cause,
        };
        assert.equal(fx.decision.renderRoutingDecision(bubble, refused), true);
        assert.equal(bubble.querySelector('.chat-routing-card'), null);
        const note = bubble.querySelector('.msg-routing-annotation');
        assert.equal(note.textContent, cause);
        assert.equal(note.dataset.annotationStatus, 'needs_manual_target');
        assert.equal(bubble.dataset.chatAnnotationStatus, 'needs_manual_target');
        assert.equal(fx.decision.renderRoutingDecision(bubble, refused), false);
    } finally { fx.restore(); }
});

test('a routing 409 that reopens the card names the host cause before the raw reason', async () => {
    const cause = 'Not started: the working folder can\'t be used';
    const fx = fixture({
        fetchImpl: async () => ({
            ok: false, status: 409,
            json: async () => ({ state: 'open', reason: 'workspace_unusable', cause }),
        }),
    });
    try {
        const bubble = routingBubble('cm-5');
        fx.decision.renderRoutingDecision(bubble, ROUTING_ANNOTATION);
        const card = bubble.querySelector('.chat-routing-card');
        card.querySelectorAll('.chat-quiz-option')[0].click();
        await new Promise((resolve) => setTimeout(resolve, 0));
        assert.equal(card.dataset.state, 'open');
        assert.equal(fx.toasts.length, 1);
        assert.equal(fx.toasts[0].text, `Not routed: ${cause} — pick again.`);
        assert.equal(fx.toasts[0].text.includes('workspace_unusable'), false);
    } finally { fx.restore(); }
});

test('a routing click posts the routing decision id with a STABLE request id', async () => {
    const fx = fixture();
    try {
        const bubble = routingBubble('cm-2');
        fx.decision.renderRoutingDecision(bubble, ROUTING_ANNOTATION);
        const card = bubble.querySelector('.chat-routing-card');
        card.querySelectorAll('.chat-quiz-option')[1].click();
        await new Promise((resolve) => setTimeout(resolve, 0));
        assert.equal(fx.calls.length, 1);
        const body = JSON.parse(fx.calls[0].init.body);
        assert.equal(body.decision_id, 'routing:cm-2:tok-1');
        assert.equal(body.option_index, 1);
        assert.ok(body.request_id);
        assert.equal(card.dataset.state, 'answered');
        assert.ok(card.querySelectorAll('.chat-quiz-option')[1].classList.contains('chosen'));
        assert.deepEqual(markOf(card), ['dot', 'ok', false]);
        assert.deepEqual(chosenParts(card), [null, CHOSEN], 'the routed destination is the owner\'s choice');
    } finally { fx.restore(); }
});

test('a destination another click is still routing is marked, and named `Your choice` only once it is recorded', async () => {
    const fx = fixture({ fetchImpl: async () => ({ ok: false, status: 409, json: async () => ({ state: 'pending', answered_index: 1 }) }) });
    try {
        const bubble = routingBubble('cm-7');
        fx.decision.renderRoutingDecision(bubble, ROUTING_ANNOTATION);
        const card = bubble.querySelector('.chat-routing-card');
        card.querySelectorAll('.chat-quiz-option')[0].click();
        await turn();
        assert.equal(card.dataset.state, 'pending');
        assert.ok(card.querySelectorAll('.chat-quiz-option')[1].classList.contains('chosen'), 'the destination being routed');
        assert.deepEqual(chosenParts(card), [null, null], 'not yet the owner\'s recorded choice');
    } finally { fx.restore(); }
});

test('a superseded routing card keeps the head\'s disabled ink', async () => {
    const fx = fixture({ fetchImpl: async () => ({ ok: false, status: 409, json: async () => ({ state: 'superseded' }) }) });
    try {
        const bubble = routingBubble('cm-6');
        fx.decision.renderRoutingDecision(bubble, ROUTING_ANNOTATION);
        const card = bubble.querySelector('.chat-routing-card');
        card.querySelectorAll('.chat-quiz-option')[0].click();
        await turn();
        assert.equal(card.dataset.state, 'superseded');
        assert.deepEqual(markOf(card), ['dot', 'ink', false]);
    } finally { fx.restore(); }
});

test('a routing 409 settles the card from the body state, never a false expiry', async () => {
    const fx = fixture({
        fetchImpl: async () => ({
            ok: false, status: 409,
            json: async () => ({ state: 'answered', answered_index: 0 }),
        }),
    });
    try {
        const bubble = routingBubble('cm-3');
        fx.decision.renderRoutingDecision(bubble, ROUTING_ANNOTATION);
        const card = bubble.querySelector('.chat-routing-card');
        card.querySelectorAll('.chat-quiz-option')[1].click();
        await new Promise((resolve) => setTimeout(resolve, 0));
        assert.equal(card.dataset.state, 'answered');
        assert.ok(card.querySelectorAll('.chat-quiz-option')[0].classList.contains('chosen'));
        assert.equal(fx.toasts.length, 1);
    } finally { fx.restore(); }
});

test('more than eight options hide behind a show-all control', () => {
    let writes = 0;
    const fx = fixture({ onDomWrite: (mutate) => { writes += 1; return mutate(); } });
    try {
        const bubble = routingBubble('cm-4');
        const wide = {
            ...ROUTING_ANNOTATION,
            options: Array.from({ length: 11 }, (_, i) => (
                { action: 'steer_task', task_id: `t${i}`, title: `Task ${i}` })),
        };
        fx.decision.renderRoutingDecision(bubble, wide);
        const card = bubble.querySelector('.chat-routing-card');
        const buttons = card.querySelectorAll('.chat-quiz-option');
        assert.equal(buttons.length, 11);
        assert.equal(buttons.filter((btn) => btn.hidden).length, 3);
        const more = card.querySelector('.chat-quiz-more');
        assert.match(more.textContent, /11/);
        const beforeClick = writes;
        more.click();
        assert.equal(writes, beforeClick + 1);
        assert.equal(buttons.filter((btn) => btn.hidden).length, 0);
        assert.equal(card.querySelector('.chat-quiz-more'), null);
    } finally { fx.restore(); }
});

test('quiz success and duplicate retry use recorded confirmation, never the attempted answer', async () => {
    for (const body of [null, 'malformed', {}, { ok: true, state: 'answered' },
        { ok: true, state: 'answered', answered_index: 99 },
        { ok: true, state: 'answered', answered_index: 0 },
        { ok: true, state: 'answered', duplicate: true, comment: 'my own plan' }]) {
        const fx = fixture({ fetchImpl: async () => ({ ok: true, status: 200,
            json: async () => { if (body === 'malformed') throw new Error('bad JSON'); return body; } }) });
        try {
            const card = fx.decision.buildQuizCard(WS_MSG);
            card.querySelector('.chat-quiz-comment').value = 'unconfirmed retry draft';
            card.querySelectorAll('.chat-quiz-option')[1].click();
            await new Promise((resolve) => setTimeout(resolve, 0));
            const valid = body?.answered_index === 0 || body?.duplicate === true;
            assert.deepEqual([card.dataset.state, card.dataset.ownerComment, fx.toasts.length],
                [valid ? 'answered' : 'open', body?.comment, valid ? 0 : 1]);
            assert.deepEqual(card.querySelectorAll('.chat-quiz-option').map(b => b.classList.contains('chosen')),
                [body?.answered_index === 0, false]);
        } finally { fx.restore(); }
    }
});

test('a 409 loser adopts the winning comment over its local draft', async () => {
    const fx = fixture({ fetchImpl: async () => ({
        ok: false, status: 409,
        json: async () => ({ ok: false, error: 'quiz_closed', state: 'answered', answered_index: 0, comment: 'winning note' }),
    }) });
    try {
        const card = fx.decision.buildQuizCard(WS_MSG);
        const area = card.querySelector('.chat-quiz-comment');
        area.value = 'losing draft';
        card.querySelectorAll('.chat-quiz-option')[1].click();
        await new Promise((resolve) => setTimeout(resolve, 0));
        assert.equal(card.dataset.state, 'answered');
        assert.equal(card.dataset.ownerComment, 'winning note');
    } finally { fx.restore(); }
});

test('the recommended option carries a badge on the live card and on the targeted-detail replay', async () => {
    const fx = fixture({ fetchDetail: async () => ({ task_id: 't-1', project_id: 'p1', owner_quiz: {
        'qz-2': { ...WS_MSG, quiz_id: 'qz-2', options: ['Yes', 'No'], option_details: ['', 'wait for CI'],
            recommended_index: 1, asked_at: WS_MSG.ts },
    } }) });
    const badges = (card) => card.querySelectorAll('.chat-quiz-option')
        .map((button) => button.querySelectorAll('.chat-quiz-option-recommended').length);
    try {
        const live = fx.decision.buildQuizCard({ ...WS_MSG, options: [{ label: 'Yes', recommended: true }, { label: 'No' }] });
        assert.deepEqual(badges(live), [1, 0]);
        assert.equal(live.querySelector('.chat-quiz-option-recommended').textContent, 'recommended');
        // A history row that lost the option details carries no badge; the targeted detail
        // (durable recommended_index) adds it in place, exactly like the option details.
        const stale = fx.decision.buildQuizCard({ ...WS_MSG, quiz_id: 'qz-2', options: ['Yes', 'No'] });
        assert.deepEqual(badges(stale), [0, 0]);
        const question = await fx.decision.readQuestion('t-1', 'qz-2', 'p1');
        assert.equal(fx.decision.buildQuizCard(question), null);
        assert.deepEqual(badges(stale), [0, 1]);
        assert.equal(fx.decision.buildQuizCard(question), null);
        assert.deepEqual(badges(stale), [0, 1], 'the badge is added once');
        // A fresh card built straight from the projection carries the same badge; a plain card none.
        assert.deepEqual(badges(fx.decision.buildQuizCard({ ...question, quiz_id: 'qz-5' })), [0, 1]);
        assert.deepEqual(badges(fx.decision.buildQuizCard({ ...WS_MSG, quiz_id: 'qz-3' })), [0, 0]);
    } finally { fx.restore(); }
});

test('a bounded wait that closed stops saying waiting while the card stays answerable', () => {
    const fx = fixture({ fetchImpl: async () => ({ ok: true, status: 200 }) });
    try {
        const card = fx.decision.buildQuizCard({ ...WS_MSG, quiz_id: 'qz-w', wait_for_answer: true });
        const quizCard = card;
        assert.match(quizCard.querySelector('.chat-quiz-wait').textContent, /Waiting for your answer/);
        assert.deepEqual(markOf(quizCard), ['dot', 'owner', false]);
        const root = { querySelector: (sel) => (sel.includes('qz-w') ? quizCard : null) };
        const applied = fx.decision.applyQuizStateFrame(root, { quiz_id: 'qz-w', task_id: 't-1', state: 'open', wait_for_answer: false });
        assert.equal(applied, true);
        assert.equal(quizCard.querySelector('.chat-quiz-wait'), null);
        assert.match(quizCard.querySelector('.chat-quiz-assumption').textContent, /wait ended/);
        assert.equal(quizCard.dataset.state, 'open');
        assert.deepEqual(markOf(quizCard), ['dot', 'owner', false], 'the bound closed; the question still waits for the owner');
        // History replay renders the ended wait from the projection (wait_ended_at, no wait_for_answer).
        const replayed = fx.decision.buildQuizCard({ ...WS_MSG, quiz_id: 'qz-r', wait_ended_at: '2026-08-31T10:05:00Z' });
        const rcard = replayed;
        assert.match(rcard.querySelector('.chat-quiz-assumption').textContent, /wait ended/);
        assert.equal(rcard.querySelector('.chat-quiz-wait'), null);
        assert.deepEqual(markOf(rcard), ['dot', 'owner', false]);
    } finally { fx.restore(); }
});

test('reconciling a replayed row into an existing card projects the closed bound', () => {
    const fx = fixture({ fetchImpl: async () => ({ ok: true, status: 200 }) });
    try {
        const card = fx.decision.buildQuizCard({ ...WS_MSG, quiz_id: 'qz-m', wait_for_answer: true });
        const quizCard = card;
        assert.match(quizCard.querySelector('.chat-quiz-wait').textContent, /Waiting for your answer/);
        const mark = quizCard.querySelector('.chat-quiz-mark');
        assert.deepEqual(markOf(quizCard), ['dot', 'owner', false]);
        // The owner was disconnected during the timeout: no frame arrived, history replays the row
        // with the closed bound and reconciles it into the SAME card.
        assert.equal(fx.decision.buildQuizCard({ ...WS_MSG, quiz_id: 'qz-m', wait_ended_at: '2026-08-31T10:05:00Z' }), null);
        assert.equal(quizCard.querySelector('.chat-quiz-wait'), null);
        assert.match(quizCard.querySelector('.chat-quiz-assumption').textContent, /wait ended/);
        assert.deepEqual(markOf(quizCard), ['dot', 'owner', false], 'unanswered: still the owner\'s move');
        assert.equal(quizCard.querySelector('.chat-quiz-mark'), mark, 'the same mark is repainted, never replaced');
    } finally { fx.restore(); }
});

test('a detail read begun before a live answer enriches the source, never the state', async () => {
    let resolve;
    const fx = fixture({ fetchDetail: () => new Promise((done) => { resolve = done; }) });
    try {
        const read = fx.decision.readQuestion('t-1', 'qz-1', 'p1');
        await turn();
        fx.decision.applyQuizStateFrame({}, { task_id: 't-1', quiz_id: 'qz-1', state: 'answered', answered_index: 0, comment: 'Actual answer' });
        resolve({ task_id: 't-1', project_id: 'p1', owner_quiz: { 'qz-1': { ...WS_MSG, wait_for_answer: true } },
            owner_wait: { quiz_id: 'qz-1', state: 'waiting' } });
        const question = await read;
        assert.equal(question.state, 'answered');
        assert.equal(question.comment, 'Actual answer');
        assert.equal(question.question, 'Merge now?');
        const card = fx.decision.buildQuizCard(question);
        assert.equal(card.querySelector('.chat-quiz-status-text').textContent, 'You answered');
        assert.ok(card.querySelectorAll('.chat-quiz-option')[0].classList.contains('chosen'));
        assert.equal(card.querySelector('.chat-quiz-wait'), null);
        fx.decision.destroy();
        assert.equal(await fx.decision.readQuestion('t-1', 'qz-1', 'p1'), null);
    } finally { fx.restore(); }
});

test('the Project card reads the wait facts history attaches: a wait resumed by owner input stops waiting', async () => {
    const fx = fixture({ fetchDetail: async () => ({ task_id: 't-1', project_id: 'p1',
        owner_quiz: { 'qz-d': { ...WS_MSG, quiz_id: 'qz-d', wait_for_answer: true, options: ['Yes', 'No'] } },
        owner_wait: { quiz_id: 'another', state: 'waiting' } }) });
    try {
        const replayed = fx.decision.buildQuizCard({ task_id: 't-1', text: 'Merge now?',
            quiz: { ...WS_MSG, quiz_id: 'qz-o', wait_for_answer: true, owner_wait_state: 'resumed', assumption: '' } });
        assert.equal(replayed.querySelector('.chat-quiz-wait'), null);
        assert.match(replayed.querySelector('.chat-quiz-assumption').textContent, /wait ended without an answer/);
        assert.equal(replayed.querySelector('.chat-quiz-status-text').textContent, 'Unanswered · task continued');
        assert.ok(replayed.querySelectorAll('.chat-quiz-option').every((btn) => !btn.disabled));
        // A detail read whose wait record moved on to another quiz proves this wait ended too.
        const detail = fx.decision.buildQuizCard(await fx.decision.readQuestion('t-1', 'qz-d', 'p1'));
        assert.equal(detail.querySelector('.chat-quiz-status-text').textContent, 'Unanswered · task continued');
        // A bounded wait that ended under an assumption names the path the task took.
        const assumed = fx.decision.buildQuizCard({ ...WS_MSG, quiz_id: 'qz-a', wait_for_answer: true, wait_ended_at: 'x' });
        assert.match(assumed.querySelector('.chat-quiz-assumption').textContent, /under its assumption \(continuing with the merge\)/);
    } finally { fx.restore(); }
});

test('a late answer says where it went', async () => {
    for (const forwarded of [true, false]) {
        const fx = fixture({ fetchImpl: async () => ({ ok: true, status: 200, json: async () => ({
            ok: true, state: 'answered', answered_index: 0, answered_after_terminal: true, forwarded }) }) });
        try {
            const card = fx.decision.buildQuizCard({ ...WS_MSG, state: 'expired_terminal' });
            card.querySelectorAll('.chat-quiz-option')[0].click();
            await turn();
            assert.equal(card.dataset.state, 'answered');
            assert.deepEqual(fx.toasts.map((toast) => toast.tone), ['info']);
            assert.match(fx.toasts[0].text, forwarded ? /delivered to its chat as your message/ : /nothing is waiting on it/);
        } finally { fx.restore(); }
    }
});

// Owner 21.10 (refined A): the question bubble's foot is its time alone. The host's sentence (asking task, run start,
// the owner's last message) no longer renders anywhere; everything inside the card — question, options with their
// details and recommendation, the recorded choice (`Your choice`), the owner's own words and the signature line —
// stays exactly as it was, live and on a replayed history row.
test('the question bubble ends on its time alone; the host sentence never renders and the card keeps every part', () => {
    const fx = fixture({ frameNode: frameBubble });
    const facts = 'Asked by task t-1, started by your message of 2026-09-25 00:21 UTC; '
        + 'your last message in this chat: 2026-09-25 00:21 UTC (47 minutes before this question).';
    const answered = { ...WS_MSG, state: 'answered', answered_index: 1, comment: 'after CI, please', host_facts: facts };
    const parts = (bubble) => {
        const card = bubble.querySelector('.chat-quiz-card');
        const options = card.querySelectorAll('.chat-quiz-option');
        return {
            question: card.querySelector('.chat-quiz-question')?.textContent,
            options: options.map((button) => button.querySelector('.chat-quiz-option-label')?.textContent),
            detail: options[1]?.querySelector('.chat-quiz-option-detail')?.textContent,
            chosen: options.map((button) => Boolean(button.querySelector('.chat-quiz-option-chosen-name'))),
            answer: card.querySelector('.chat-quiz-answer')?.textContent,
            assumption: card.querySelector('.chat-quiz-assumption')?.textContent,
            state: card.dataset.state,
        };
    };
    try {
        const live = fx.decision.buildQuizCard({ ...answered, quiz_id: 'qz-facts' });
        assert.equal(live.querySelector('.chat-quiz-host-facts'), null);
        assert.equal(live.querySelector('.chat-bubble-foot'), null, 'no part joins the time: the plain frame');
        assert.deepEqual(live.children.map((node) => node.className), ['sender', 'chat-quiz-card', 'msg-time']);
        const expected = { question: 'Merge now?', options: ['Yes', 'No'], detail: 'wait for CI', chosen: [false, true],
            answer: "Owner's answer: after CI, please", assumption: 'Continuing meanwhile: continuing with the merge',
            state: 'answered' };
        assert.deepEqual(parts(live), expected);
        // A stored history row nests the quiz: the replay paints the same card, its choice included, and no sentence.
        const replay = fx.decision.buildQuizCard({ task_id: 't-1', text: WS_MSG.question,
            quiz: { ...answered, quiz_id: 'qz-facts-replay' } });
        assert.equal(replay.querySelector('.chat-quiz-host-facts'), null);
        assert.deepEqual(parts(replay), expected);
        // Free words with no chosen option are the whole answer: no option is marked as chosen.
        const words = fx.decision.buildQuizCard({ ...WS_MSG, quiz_id: 'qz-words', state: 'answered', comment: 'neither yet',
            host_facts: facts });
        assert.deepEqual(parts(words).chosen, [false, false]);
        assert.equal(parts(words).answer, "Owner's answer: neither yet");
    } finally { fx.restore(); }
});

test('a Main mirror of a Project question shows no host sentence, from its pointer row or a later delivery', () => {
    const column = new NodeStub();
    const fx = fixture({ isMain: true, frameNode: frameBubble,
        insertMessageNode: (node) => { column.append(node); return true; } });
    const row = { role: 'system', system_type: 'project_question_pointer', task_id: 't-9', quiz_id: 'qz-9',
        project_id: 'p1', project_chat_id: 23, project_name: 'Storage', ts: '2026-09-25T00:00:00+00:00',
        quiz_state: 'open', question: 'Merge now?', options: ['Yes', 'No'] };
    try {
        assert.ok(fx.decision.appendQuestionPointer({ ...row, host_facts: 'Asked by task t-9, origin unknown.' }));
        const bubble = column.children[0];
        assert.equal(bubble.querySelector('.chat-quiz-host-facts'), null);
        assert.equal(bubble.querySelector('.chat-quiz-question').textContent, 'Merge now?');
        assert.deepEqual(bubble.querySelectorAll('.chat-quiz-option').map((button) =>
            button.querySelector('.chat-quiz-option-label').textContent), ['Yes', 'No']);
        fx.decision.appendQuestionPointer({ ...row, host_facts: 'Asked by task t-9, a later sentence.' });
        assert.equal(bubble.querySelector('.chat-quiz-host-facts'), null);
    } finally { fx.restore(); }
});

test('Main reveals its own question by exact task and room, never a Project or another chat record', async () => {
    const records = {
        main: { task_id: 'main', chat_id: 1, owner_quiz: { 'qz-1': WS_MSG } },
        unstamped: { task_id: 'unstamped', owner_quiz: { 'qz-1': WS_MSG } },
        project: { task_id: 'project', project_id: 'p1', owner_quiz: { 'qz-1': WS_MSG } },
        other: { task_id: 'other', chat_id: 7, owner_quiz: { 'qz-1': WS_MSG } },
    };
    const fx = fixture({ isMain: true, fetchDetail: async (taskId) => records[taskId] || null });
    const appended = [];
    const append = (msg) => { appended.push(msg.task_id); fx.decision.buildQuizCard(msg); };
    try {
        assert.equal((await fx.decision.readQuestion('main', 'qz-1', ''))?.task_id, 'main');
        assert.equal(await fx.decision.readQuestion('unstamped', 'qz-1', ''), null);
        assert.equal(await fx.decision.readQuestion('project', 'qz-1', ''), null);
        assert.equal(await fx.decision.readQuestion('other', 'qz-1', ''), null);
        const revealed = await fx.decision.revealQuestion('main', 'qz-1', '', 1, append, () => true,
            () => {}, () => {}, { focus: 'question' });
        assert.equal(revealed, true);
        assert.deepEqual(appended, ['main']);
        // The work list lands on the question itself, never the answer field, and posts nothing.
        assert.ok(globalThis.document.activeElement?.classList.contains('chat-quiz-question'));
        assert.equal(fx.calls.length, 0);
        assert.equal(await fx.decision.revealQuestion('other', 'qz-1', '', 1, append, () => true), false);
        assert.deepEqual(appended, ['main']);
    } finally { fx.restore(); }
    const project = fixture({ fetchDetail: async () => records.main });
    try {
        // A Project room still needs its Project: an empty one names nothing there.
        assert.equal(await project.decision.revealQuestion('main', 'qz-1', '', 23, () => {}, () => true), false);
    } finally { project.restore(); }
});

test('Main Header reads crossing a socket close cannot observe or reveal stale authority after reopen', { timeout: 3000 }, async () => {
    const pending = [];
    const fx = fixture({ isMain: true, fetchDetail: () => new Promise(resolve => pending.push(resolve)) });
    const detail = state => ({ task_id: 'main', chat_id: 1, owner_quiz: { 'qz-1': { ...WS_MSG, state } } });
    const appended = [];
    try {
        const read = fx.decision.revealQuestion('main', 'qz-1', '', 1,
            row => { appended.push(row); fx.decision.buildQuizCard(row); }, () => true);
        await Promise.resolve();
        fx.decision.setConnected(false);
        let offlineSettled = false;
        const offline = fx.decision.readQuestion('main', 'qz-1', '').then(value => { offlineSettled = true; return value; });
        await Promise.resolve(); await Promise.resolve();
        assert.equal(offlineSettled, true, 'offline reads return without joining an old request');
        assert.equal(await offline, null);
        fx.decision.setConnected(true);
        const fresh = fx.decision.readQuestion('main', 'qz-1', '');
        await Promise.resolve();
        assert.equal(pending.length, 2, 'the reopen does not join the retired read');
        pending[0](detail('closed'));
        assert.equal(await read, false);
        assert.deepEqual(appended, []);
        pending[1](detail('open'));
        assert.equal((await fresh).state, 'open', 'the obsolete read never enters lifecycle memory');
        assert.equal(fx.calls.length, 0);
    } finally { fx.decision.destroy(); fx.restore(); }
});

test('Main native question withdraws every input on close and canonical reread keeps its draft and node', { timeout: 3000 }, async () => {
    const fx = fixture({ isMain: true, fetchDetail: async () => ({ task_id: 't-1', chat_id: 1,
        owner_quiz: { 'qz-1': WS_MSG } }) });
    try {
        const bubble = fx.decision.buildQuizCard(WS_MSG);
        const card = bubble.querySelector('.chat-quiz-card') || bubble;
        const field = card.querySelector('.chat-quiz-comment');
        field.value = 'Keep this draft';
        fx.decision.setConnected(false);
        assert.equal(field.disabled, true);
        assert.ok(card.querySelectorAll('.chat-quiz-option').every(button => button.disabled));
        fx.decision.buildQuizCard(WS_MSG);
        assert.equal(field.disabled, true, 'an older history row cannot restore controls');
        fx.decision.setConnected(true);
        assert.equal(field.disabled, true, 'the reopen alone is no observation');
        await fx.decision.readQuestion('t-1', 'qz-1', '');
        assert.equal(card.querySelector('.chat-quiz-comment'), field);
        assert.equal(field.value, 'Keep this draft');
        assert.equal(field.disabled, false);
        assert.equal(fx.calls.length, 0);
    } finally { fx.decision.destroy(); fx.restore(); }
});

test('settled native history needs no recovery reads, and missing exact room proof authorizes nothing', async () => {
    const reads = [];
    const fx = fixture({ isMain: true, rawDetail: true, fetchDetail: async (taskId) => {
        reads.push(taskId);
        return { task_id: taskId, chat_id: 1, owner_quiz: { 'qz-1': { ...WS_MSG, task_id: taskId } } };
    } });
    try {
        fx.decision.buildQuizCard({ ...WS_MSG, state: 'answered', answered_index: 0 });
        // An unanswered native card beside it: the census after the reopen does reconfirm that one.
        fx.decision.buildQuizCard({ ...WS_MSG, task_id: 't-2' });
        fx.decision.setConnected(false);
        fx.decision.setConnected(true);
        for (let n = 0; n < 3; n++) { fx.decision.syncCurrentQuestions([], true); await turn(); }
        assert.ok(reads.includes('t-2'), 'the census ran and reconfirms the unanswered native card');
        assert.equal(reads.filter(id => id === 't-1').length, 0, 'the settled native card is never read');
        assert.equal(await fx.decision.readQuestion('t-1', 'qz-1', ''), null);
        assert.equal(fx.calls.length, 0);
    } finally { fx.decision.destroy(); fx.restore(); }
});

test('a question card says its work is paused from the instance\'s works and returns to its words after Resume', () => {
    const fx = fixture();
    try {
        const card = fx.decision.buildQuizCard({ ...WS_MSG, options: ['Yes', 'No'], owner_wait_state: 'waiting' });
        const status = () => card.querySelector('.chat-quiz-status-text').textContent;
        assert.equal(status(), 'Waiting for your answer');
        const works = (activity) => ({ episodes: new Map([['root', { work_id: 'root', confirmed: true, episode: { task_id: 'root' },
            members: { ids: ['root', 't-1'], states: { 't-1': { activity } } } }]]) });
        fx.decision.syncWorks(works('paused'));
        // The card shows the paused reading of its own question, which names Resume (question_presentation.test.js).
        assert.equal(status(), questionPresentation({ state: 'open', owner_wait_state: 'waiting' }, { paused: true }).status);
        assert.match(status(), /\bResume\b/);
        assert.deepEqual(markOf(card), ['dot', 'owner', false], 'the owner\'s answer is still wanted');
        fx.decision.syncWorks(works('working'));
        assert.equal(status(), 'Waiting for your answer');
        // An answer in flight keeps its own words until it settles.
        fx.decision.syncWorks(works('paused'));
        card.dataset.submission = 'sending';
        card.querySelector('.chat-quiz-status-text').textContent = 'Sending…';
        fx.decision.syncWorks(works('working'));
        assert.equal(status(), 'Sending…');
    } finally { fx.restore(); }
});

// DESIGN "Replies": an owner's answer to a card has no row of its own in the feed. History names its row
// (`answer_row_address`); the home card puts that address on the one visible answer, so a Reply lands there.
const ANSWER_ROW = 'row:23@2024-04-09T10:14:00+00:00#0123456789ab';
const legendFrame = { ...WS_MSG, question: 'Use blue and gold in the table legend?', ts: '2024-04-09T10:05:00+00:00',
    stake: 'Table readability', assumption: 'Keep the current colors', options: [{ label: 'Yes' }, { label: 'No' }] };
const answeredRow = (quiz) => ({ msg_type: 'quiz', role: 'assistant', task_id: 't-1', text: legendFrame.question, ts: legendFrame.ts,
    quiz: { stake: '', assumption: '', options: legendFrame.options, state: 'answered', ...quiz } });

test('the answer row\'s address marks the recorded choice, else the owner\'s words, on its own card', () => {
    const fx = fixture();
    try {
        // A choice with words beside it is one answer: the choice carries the address.
        const both = fx.decision.buildQuizCard(answeredRow({ quiz_id: 'qz-both', answered_index: 1, comment: 'and soon',
            answer_row_address: ANSWER_ROW }));
        const [yes, no] = both.querySelectorAll('.chat-quiz-option');
        assert.deepEqual([yes.dataset.rowAddress, no.dataset.rowAddress, both.querySelector('.chat-quiz-answer').dataset.rowAddress],
            [undefined, ANSWER_ROW, undefined]);
        assert.deepEqual([no.disabled, no.getAttribute('aria-disabled'), no.tabIndex, yes.disabled], [false, 'true', -1, true],
            'the recorded choice is inert yet focusable; the other option is disabled');
        assert.equal(no.dataset.historyId, undefined, 'an address alone: the card is not a row of the feed');
        // Free words without a choice: the answer line carries it and can take focus.
        const words = fx.decision.buildQuizCard(answeredRow({ quiz_id: 'qz-words', comment: 'neither', answer_row_address: ANSWER_ROW }));
        const line = words.querySelector('.chat-quiz-answer');
        assert.deepEqual([line.dataset.rowAddress, line.tabIndex], [ANSWER_ROW, -1]);
        assert.ok(words.querySelectorAll('.chat-quiz-option').every((btn) => btn.disabled && !btn.dataset.rowAddress));
        // Without an address (an older row) nothing is marked; the choice still records the answer.
        const old = fx.decision.buildQuizCard(answeredRow({ quiz_id: 'qz-old', answered_index: 0 }));
        assert.ok(old.querySelectorAll('.chat-quiz-option').every((btn) => !btn.dataset.rowAddress));
        assert.ok(settled(old.querySelectorAll('.chat-quiz-option')));
    } finally { fx.restore(); }
});

test('a narrower live frame keeps the answer row\'s address and a card answered live receives it from history', () => {
    const fx = fixture();
    try {
        const card = fx.decision.buildQuizCard(answeredRow({ quiz_id: 'qz-kept', answered_index: 1, answer_row_address: ANSWER_ROW }));
        const chosen = card.querySelectorAll('.chat-quiz-option')[1];
        fx.decision.applyQuizStateFrame({}, { quiz_id: 'qz-kept', task_id: 't-1', state: 'answered', answered_index: 1 });
        assert.equal(chosen.dataset.rowAddress, ANSWER_ROW, 'the live quiz_state frame names no row and erases none');
        const live = fx.decision.buildQuizCard({ ...legendFrame, quiz_id: 'qz-live' });
        fx.decision.applyQuizStateFrame({}, { quiz_id: 'qz-live', task_id: 't-1', state: 'answered', answered_index: 0 });
        const first = live.querySelectorAll('.chat-quiz-option')[0];
        assert.equal(first.dataset.rowAddress, undefined, 'answered live: no row is known yet');
        assert.equal(fx.decision.buildQuizCard({ task_id: 't-1', text: legendFrame.question, ts: legendFrame.ts,
            quiz: { ...legendFrame, quiz_id: 'qz-live', state: 'answered', answered_index: 0, answer_row_address: ANSWER_ROW } }), null);
        assert.equal(first.dataset.rowAddress, ANSWER_ROW, 'the next history row names it on the same card');
    } finally { fx.restore(); }
});

test('the owner\'s own press still hands focus to the question; a landing on the recorded choice keeps it', async () => {
    const fx = fixture();
    try {
        const card = fx.decision.buildQuizCard({ ...legendFrame, quiz_id: 'qz-focus' });
        const first = card.querySelectorAll('.chat-quiz-option')[0];
        first.focus();
        first.click();
        await turn();
        assert.equal(card.dataset.state, 'answered');
        assert.equal(globalThis.document.activeElement, card.querySelector('.chat-quiz-question'),
            'settling made the pressed option inert: focus stays in the card, on its question');
        first.focus();  // a Reply's landing (chat_reply.markTarget)
        fx.decision.applyQuizStateFrame({}, { quiz_id: 'qz-focus', task_id: 't-1', state: 'answered', answered_index: 0 });
        assert.equal(globalThis.document.activeElement, first, 'a repeated settlement does not move a landing');
    } finally { fx.restore(); }
});

test('a file-only settled answer keeps host file facts out of Owner\'s answer and carries the Reply landing address', async () => {
    const files = [
        { name: 'shot.png', kind: 'image', size: 12, sha256: 'a'.repeat(64) },
        { name: 'clip.mp4', kind: 'file', unavailable: 'too_large', limit_bytes: 10485760 },
        { name: 'gone.pdf', kind: 'file', unavailable: 'missing' },
    ];
    const fx = fixture({
        isMain: true,
        fetchDetail: async () => ({
            task_id: 't-1', chat_id: 1,
            owner_quiz: { 'qz-detail': { ...WS_MSG, quiz_id: 'qz-detail', state: 'answered', asked_at: WS_MSG.ts, attachments: files } },
        }),
    });
    try {
        // History row with file-only answer: no Owner's answer: label wrapping host facts; Reply lands on file block.
        const card = fx.decision.buildQuizCard(answeredRow({ quiz_id: 'qz-files', attachments: files, answer_row_address: ANSWER_ROW }));
        assert.equal(card.querySelector('.chat-quiz-answer'), null, 'no Owner\'s answer block when there is no caption');
        const fileBlock = card.querySelector('.chat-quiz-answer-files');
        assert.equal(fileBlock?.textContent,
            '📎 shot.png\n📎 clip.mp4: not received, too large\n📎 gone.pdf: not received, unavailable');
        assert.deepEqual([fileBlock.dataset.rowAddress, fileBlock.tabIndex], [ANSWER_ROW, -1]);
        assert.equal(card.querySelectorAll('a').length, 0, 'answer files never open a download link');

        // Task-detail re-read (GET /api/tasks/{id}) also settles a card with projected answer files.
        const detailCard = fx.decision.buildQuizCard({ ...WS_MSG, quiz_id: 'qz-detail' });
        await fx.decision.readQuestion('t-1', 'qz-detail', '');
        assert.equal(detailCard.dataset.state, 'answered');
        assert.equal(detailCard.querySelector('.chat-quiz-answer'), null);
        assert.equal(detailCard.querySelector('.chat-quiz-answer-files')?.textContent, fileBlock.textContent);
    } finally { fx.restore(); }
});

test('a known secret allow and its reader snapshot outlive older replies, task-detail reads and history', async () => {
    const ask = { task_id: 't-1', quiz_id: 'secret-race', project_id: 'p1', state: 'open',
        question: 'Secret needed for demo', options: [], secret_request: { plugin_name: 'demo', fields: [
            { id: 'primary', setting_key: 'SYNTH_PRIMARY' }, { id: 'other', setting_key: 'SYNTH_OTHER' },
        ] } };
    const old = { event: 'secret_filled', pluginName: 'demo', at: 'save-time', fields: ['primary', 'other'],
        allow: { state: 'deferred', skill: 'demo', keys: ['SYNTH_PRIMARY', 'SYNTH_OTHER'], enable: true },
        consumers: [{ name: 'mailer', enabled: true, reads: ['SYNTH_PRIMARY', 'SYNTH_OTHER'], granted: [] }] };
    const final = { ...old, allow: { ...old.allow, state: 'granted', granted_keys: ['SYNTH_PRIMARY'],
        resolved_at: 'resolution-time', enabled: false, enable_reason: 'owner_off' },
        consumers: [{ ...old.consumers[0], granted: ['SYNTH_OTHER'] }] };
    const update = (secret_outcome) => ({ ...ask, state: 'answered', secret_outcome });
    let returnDetail;
    const fx = fixture({ projectId: 'p1', chatId: 23, fetchDetail: () => new Promise((resolve) => { returnDetail = resolve; }) });
    const rows = (card, selector) => card.querySelectorAll(selector).map((row) => row.textContent).join('\n');
    try {
        const card = fx.decision.buildQuizCard(ask);
        fx.decision.applyQuizStateFrame(card, update(old));
        const olderRead = fx.decision.readQuestion(ask.task_id, ask.quiz_id, ask.project_id);
        await Promise.resolve();
        fx.decision.applyQuizStateFrame(card, update(final));
        // The detail readQuestion checks: its block and the room the question is shown in (question_rooms).
        returnDetail({ task_id: ask.task_id, project_id: 'p1', owner_quiz: { [ask.quiz_id]: update(old) },
            question_rooms: { [ask.quiz_id]: { state: 'known', project_id: 'p1', chat_id: 23 } } });
        assert.deepEqual((await olderRead).secret_outcome, final);
        for (const state of ['requested', 'running', 'deferred', 'unknown']) {
            fx.decision.applyQuizStateFrame(card, update({ ...old, allow: { ...old.allow, state } }));
            fx.decision.buildQuizCard(update(old));
            assert.equal(card.dataset.state, 'answered');
            assert.match(rows(card, '.chat-secret-outcome-line'), /Granted SYNTH_PRIMARY to demo/);
            assert.match(rows(card, '.chat-secret-outcome-line'), /stayed off/);
            assert.doesNotMatch(rows(card, '.chat-secret-outcome-line'), /waits|Granted SYNTH_OTHER/);
            assert.match(rows(card, '.chat-secret-reader'), /was allowed to read SYNTH_OTHER, not SYNTH_PRIMARY/);
            assert.equal(card.querySelector('.chat-secret-actions'), null);
        }
        fx.decision.releaseViews(card);
        const reopened = fx.decision.buildQuizCard(update(old));
        assert.match(rows(reopened, '.chat-secret-outcome-line'), /Granted SYNTH_PRIMARY to demo/);
        assert.match(rows(reopened, '.chat-secret-reader'), /was allowed to read SYNTH_OTHER/);
        assert.equal(reopened.querySelector('.chat-secret-actions'), null);
    } finally { fx.restore(); }
});
