import assert from 'node:assert/strict';
import test, { mock } from 'node:test';
import { NodeStub, fixture, frameBubble, turn } from './chat_decision_fixture.js';
import { fixture as mediaFixture, NodeStub as MediaNode } from './helpers/media_dom.js';
import { ElementStub, installDom, restoreDom } from './chat_dom_fixture.js';
import { createChatInstance } from '../modules/chat.js';
import { stampNodeTimestamp } from '../modules/chat_activity.js';
import { createMainWaiting } from '../modules/main_waiting.js';
import { decorateProjectRow } from '../modules/project_answer.js';
import { syncQuestionMaterials } from '../modules/result_files.js';

const question = (id, patch = {}) => ({ task_id: 'task-1', quiz_id: id, project_id: 'project-1',
    project_chat_id: 12, project_name: 'Build', quiz_state: 'open', question: `Question ${id}?`,
    options: ['Yes', 'No'], ts: '2020-01-01T00:00:00Z', ...patch });
const find = (node, predicate) => predicate(node) ? node
    : node.children.reduce((found, child) => found || find(child, predicate), null);

// Main's feed (docs/DESIGN.md "Main questions and actions"): a Project question's copy takes its
// place among the messages through the chat's one insertion seam; the census only confirms or
// settles copies already there. `recounts` is what Main's navigator sees on each change.
function feedFixture(extra = {}) {
    const feed = new NodeStub();
    const inserted = [], recounts = [];
    const fx = fixture({ isMain: true, frameNode: frameBubble,
        insertMessageNode: node => { inserted.push(node); feed.append(node); return true; },
        removeMessageNode: node => node.remove(),
        onQuestionsChanged: () => recounts.push(feed.children.length), ...extra });
    return { ...fx, feed, inserted, recounts, add: row => fx.decision.appendQuestionPointer(row),
        cards: () => feed.children.map(bubble => bubble.querySelector('.chat-quiz-card')) };
}

// Main's waiting navigator over a stub page: owner-action messages enter the feed, the pill the
// composer row. `CSS.escape` is the browser global its question lookup uses.
function mainWaiting(env, options = {}) {
    const priorCss = globalThis.CSS;
    globalThis.CSS = { escape: value => String(value) };
    const page = document.createElement('div');
    const messagesDiv = document.createElement('div');
    const toolbar = document.createElement('div');
    toolbar.className = 'chat-toolbar-row';
    page.append(messagesDiv, toolbar);
    env.mount.append(page);
    const waiting = createMainWaiting({ page, messagesDiv, toolbar, stampNodeTimestamp,
        insertMessageNode: node => { messagesDiv.append(node); return true; }, ...options });
    return { waiting, messagesDiv, destroy: () => { waiting.destroy(); globalThis.CSS = priorCss; } };
}

test('Main waiting forwards census, failed read and connection to its question owner', () => {
    const env = installDom();
    const calls = [], links = [];
    const main = mainWaiting(env, { decision: () => ({ syncCurrentQuestions: (...args) => calls.push(args),
        setConnected: value => links.push(value) }) });
    try {
        const rows = [question('current')];
        main.waiting.snapshot({ current_questions: rows, current_questions_complete: true }, 120);
        main.waiting.snapshot(null, 130);
        main.waiting.setConnected(false);
        // The census's vouching travels as a predicate: a complete one vouches for every question, a failed read for none.
        assert.deepEqual(calls.map(([sent, vouched, at]) => [sent, vouched('task-1', 'current'), vouched('other', 'q'), at]),
            [[rows, true, true, 120], [[], false, false, 130]]);
        assert.deepEqual(links, [false]);
        main.waiting.setConnected(true);
        assert.equal(calls.length, 2, 'reconnection alone is not a new question census');
        assert.deepEqual(links, [false, true]);
    } finally { main.destroy(); restoreDom(env.prior); }
});

test('owner closes a Main question with one tap: no reason to type, a stable retry identity, never over a draft', async () => {
    let attempts = 0;
    const fx = feedFixture({ fetchImpl: async () => {
        attempts++;
        if (attempts === 1) throw new Error('lost response');
        return { ok: true, status: 200, json: async () => ({ ok: true, state: 'closed',
            closed_reason: 'Closed by the owner without an answer.', closed_actor: 'owner',
            question_delivery: { kind: 'closure', state: 'owed' } }) };
    } });
    try {
        fx.add(question('close-from-main', { quiz_state: 'expired_terminal' }));
        const card = fx.cards()[0], field = card.querySelector('.chat-quiz-comment');
        const close = card.querySelector('.chat-quiz-close');
        assert.equal(card.querySelector('.chat-quiz-head-end').children.at(-1), close, 'the × ends the card head');
        assert.equal(close.getAttribute('aria-label'), 'Close question without answering');
        assert.equal(close.disabled, false, 'an answerable card closes with one tap');
        // A drafted answer holds the ×: a mis-tap never discards the owner's words.
        field.value = 'Half an answer'; field.listeners.get('input')();
        assert.equal(close.disabled, true);
        close.click(); await turn();
        assert.equal(fx.calls.length, 0, 'a held × sends nothing');
        field.value = '   '; field.listeners.get('input')();
        assert.equal(close.disabled, false, 'blank text is no draft');
        field.value = ''; field.listeners.get('input')();
        close.click(); await turn();
        assert.equal(card.dataset.state, 'expired_terminal', 'a lost response never invents closure');
        close.click(); await turn();
        const [first, retry] = fx.calls.map(call => JSON.parse(call.init.body));
        assert.deepEqual(retry, first);
        assert.deepEqual(Object.keys(first).sort(), ['action', 'decision_id', 'reason', 'request_id'],
            'closure never becomes an answer');
        assert.equal(first.action, 'close'); assert.equal(first.reason, '');
        assert.equal(first.decision_id, 'quiz:task-1:close-from-main');
        assert.equal(card.dataset.state, 'closed');
        assert.equal(card.querySelector('.chat-quiz-closed-reason').textContent, 'Closed by the owner without an answer.');
        assert.equal(card.querySelector('.chat-quiz-comment'), null);
        assert.equal(card.querySelector('.chat-quiz-close'), null, 'a settled card keeps no ×');
        assert.match(card.querySelector('.chat-quiz-status-text').textContent, /update pending/);
    } finally { fx.decision.destroy(); fx.restore(); }
});

test('a one-tap close losing an answer race displays the canonical answer and cannot revive the ×', async () => {
    const fx = feedFixture({ fetchImpl: async () => ({ ok: false, status: 409,
        json: async () => ({ state: 'answered', answered_index: 1, comment: 'Keep the original.' }) }) });
    try {
        const row = question('race'); fx.add(row);
        const card = fx.cards()[0];
        card.querySelector('.chat-quiz-close').click(); await turn();
        assert.equal(card.dataset.state, 'answered');
        assert.equal(card.querySelectorAll('.chat-quiz-option')[1].classList.contains('chosen'), true);
        fx.decision.syncCurrentQuestions([row], true);
        fx.add(row);
        assert.equal(card.dataset.state, 'answered');
        assert.equal(card.querySelector('.chat-quiz-close'), null);
        assert.equal(fx.cards().length, 1);
    } finally { fx.decision.destroy(); fx.restore(); }
});

test('the census never mounts a copy: the feed places copies, and the census confirms or settles only those', async () => {
    const fx = feedFixture();
    try {
        const rows = Array.from({ length: 25 }, (_, id) => question(`q-${id}`,
            { quiz_state: id % 2 ? 'expired_terminal' : 'open' }));
        fx.decision.syncCurrentQuestions(rows, true);
        assert.equal(fx.feed.children.length, 0, 'the canonical census alone mounts nothing');
        for (const row of rows.slice(0, 3)) fx.add(row);
        assert.deepEqual(fx.cards().map(card => card.dataset.quizId), ['q-0', 'q-1', 'q-2']);
        assert.equal(fx.inserted.length, 3, 'each copy enters through the feed insertion seam');
        fx.add(rows[0]);
        assert.equal(fx.cards().length, 3, 'a later delivery of the same question names the same copy');
        assert.deepEqual(fx.recounts, [0, 1, 2, 3], 'the navigator recounts after the census and each insert');
        assert.ok(fx.cards().every(card => card.querySelector('.chat-quiz-comment')));
        const draft = fx.cards()[0].querySelector('.chat-quiz-comment');
        draft.value = 'Keep this unfinished answer';
        fx.decision.syncCurrentQuestions(rows, true);
        assert.equal(fx.cards().length, 3, 'rows without a copy in the feed stay unmounted');
        assert.equal(fx.cards()[0].querySelector('.chat-quiz-comment'), draft);
        assert.equal(draft.value, 'Keep this unfinished answer');
        fx.decision.syncCurrentQuestions([], false);
        assert.equal(fx.cards().length, 3, 'an incomplete source is not an empty current set');
        assert.deepEqual(fx.cards().map(card => card.dataset.state), ['unknown', 'unknown', 'unknown']);
        assert.equal(draft.disabled, true);
        assert.equal(fx.cards()[1].querySelector('.chat-quiz-close').disabled, true, 'an unknown copy keeps its × inert');
        assert.equal(draft.value, 'Keep this unfinished answer');
        fx.decision.syncCurrentQuestions(rows, true);
        assert.equal(fx.cards()[0].querySelector('.chat-quiz-comment'), draft);
        assert.equal(draft.disabled, false);
        assert.equal(fx.cards()[1].querySelector('.chat-quiz-close').disabled, false, 'the canonical confirmation restores the ×');
        assert.equal(fx.cards()[0].querySelector('.chat-quiz-close').disabled, true, 'a restored draft still holds its ×');
        fx.decision.syncCurrentQuestions(rows.slice(1), true);
        await turn();
        // No longer current says nothing of its outcome (#1993): with no readable record the copy stays, withheld.
        assert.deepEqual(fx.cards().map(card => [card.dataset.quizId, card.dataset.state]),
            [['q-0', 'unknown'], ['q-1', 'expired_terminal'], ['q-2', 'open']]);
    } finally { fx.decision.destroy(); fx.restore(); }
});

test('a census vouching per question reads the unnamed copies it vouches for and leaves the rest unknown', async () => {
    mock.timers.enable({ apis: ['setTimeout'] });
    // The vouched copy's own record: closed elsewhere, its closure delivered.
    const fx = feedFixture({ fetchDetail: async (taskId) => (taskId === 'task-1' ? { task_id: taskId, project_id: 'project-1',
        owner_quiz: { left: { quiz_id: 'left', question: 'Question left?', options: ['Yes', 'No'], state: 'closed',
            closed_reason: 'Not needed any more.', delivery: { kind: 'closure', state: 'delivered' } } } } : null) });
    try {
        const named = question('named', { task_id: 'task-3' });
        fx.add(question('left'));
        fx.add(question('withheld', { task_id: 'task-2' }));
        fx.add(named);
        const asked = [];
        // The read's gaps name task-2's question: about that one it says nothing.
        fx.decision.syncCurrentQuestions([named], (taskId, quizId) => {
            asked.push([taskId, quizId]);
            return !(taskId === 'task-2' && quizId === 'withheld');
        });
        await turn();
        assert.deepEqual(fx.cards().map(card => [card.dataset.quizId, card.dataset.state]),
            [['left', 'closed'], ['withheld', 'unknown'], ['named', 'open']], 'the vouched copy shows what its read proved');
        mock.timers.tick(5000);
        assert.deepEqual(fx.cards().map(card => [card.dataset.quizId, card.dataset.state]),
            [['withheld', 'unknown'], ['named', 'open']], 'then it leaves; the withheld one awaits proof');
        assert.deepEqual(asked, [['task-1', 'left'], ['task-2', 'withheld']], 'asked only about the unnamed copies');
        assert.equal(fx.cards()[0].querySelector('.chat-quiz-comment').disabled, true);
    } finally { fx.decision.destroy(); fx.restore(); mock.timers.reset(); }
});

test('a history copy is the full form at once while nothing is forgotten; the census confirms it without a read', async () => {
    let reads = 0;
    const fx = feedFixture({ fetchDetail: async () => { reads++; return null; } });
    try {
        fx.add(question('old'));
        await turn();
        const card = fx.cards()[0];
        assert.equal(card.dataset.state, 'open');
        assert.ok(card.querySelector('.chat-quiz-comment'));
        assert.ok(card.parentNode.querySelector('.chat-live-project-name'), 'the voice line names the Project');
        assert.equal(reads, 0, 'only an evicted memory sends a copy to its canonical read');
        fx.decision.syncCurrentQuestions([question('old', { quiz_state: 'expired_terminal' })], true);
        assert.equal(card.dataset.state, 'expired_terminal');
        assert.ok(card.querySelector('.chat-quiz-comment'));
        assert.equal(reads, 0);
    } finally { fx.decision.destroy(); fx.restore(); }
});

test('explicit closure shows its reason, accepts no answer, and stale snapshots cannot revive it', () => {
    mock.timers.enable({ apis: ['setTimeout'] });
    const fx = feedFixture();
    try {
        const row = question('close');
        fx.add(row);
        fx.decision.syncCurrentQuestions([], false);
        fx.decision.applyQuizStateFrame(fx.feed, { task_id: row.task_id, quiz_id: row.quiz_id,
            state: 'closed', closed_reason: 'The owner cancelled this output.' });
        assert.equal(fx.cards()[0].dataset.state, 'closed');
        assert.equal(fx.cards()[0].querySelector('.chat-quiz-closed-reason').textContent, 'The owner cancelled this output.');
        assert.equal(fx.cards()[0].querySelector('.chat-quiz-comment'), null);
        assert.equal(fx.cards()[0].querySelector('.chat-quiz-answer'), null);
        assert.equal(fx.cards()[0].querySelector('.chat-quiz-close'), null);
        fx.decision.syncCurrentQuestions([row], true);
        assert.equal(fx.cards()[0].dataset.state, 'closed');
        mock.timers.tick(5000);
        assert.equal(fx.cards().length, 0);
        fx.decision.syncCurrentQuestions([row], true);
        fx.add(row);
        assert.equal(fx.cards().length, 0, 'neither the census nor a stale history row brings it back');
    } finally { fx.decision.destroy(); fx.restore(); mock.timers.reset(); }
});

test('a settled pointer never enters Main unless its delivery is owed: a reload shows no closed flash', () => {
    mock.timers.enable({ apis: ['setTimeout'] });
    const fx = feedFixture();
    try {
        fx.add(question('closed', { quiz_state: 'closed', closed_reason: 'Not needed any more.' }));
        fx.add(question('answered', { quiz_state: 'answered', answered_index: 0 }));
        assert.equal(fx.inserted.length, 0, 'nothing settled and discharged mounts, not even for five seconds');
        fx.add(question('owed', { quiz_state: 'closed', closed_reason: 'Not needed any more.',
            question_delivery: { kind: 'closure', state: 'owed' } }));
        assert.deepEqual(fx.cards().map(card => [card.dataset.quizId, card.dataset.state]), [['owed', 'closed']],
            'an owed closure keeps its place until it is delivered');
        mock.timers.tick(60000);
        assert.equal(fx.cards().length, 1);
        // A live closure this tab already observed keeps a stale open delivery out as well.
        fx.decision.applyQuizStateFrame(fx.feed, { task_id: 'task-1', quiz_id: 'seen', state: 'closed', closed_reason: 'Done.' });
        fx.add(question('seen'));
        assert.deepEqual(fx.cards().map(card => card.dataset.quizId), ['owed']);
    } finally { fx.decision.destroy(); fx.restore(); mock.timers.reset(); }
});

test('owed answers and closures survive reload and time until their delivery is confirmed', () => {
    mock.timers.enable({ apis: ['setTimeout'] });
    const fx = feedFixture();
    try {
        const rows = [question('answered', { quiz_state: 'answered', answered_index: 0,
            question_delivery: { kind: 'answer', state: 'owed' } }),
        question('closed', { quiz_state: 'closed', closed_reason: 'The requested work was cancelled.',
            question_delivery: { kind: 'closure', state: 'owed' } })];
        for (const row of rows) fx.add(row);
        assert.equal(fx.cards().length, 2, 'a fresh Main includes already settled but undischarged records');
        assert.match(fx.cards()[0].querySelector('.chat-quiz-status-text').textContent, /Answer recorded.*delivery pending/);
        assert.match(fx.cards()[1].querySelector('.chat-quiz-status-text').textContent, /Question closed.*update pending/);
        assert.ok(fx.cards().every(card => !card.querySelector('.chat-quiz-comment')));
        mock.timers.tick(60000);
        assert.equal(fx.cards().length, 2);
        fx.decision.syncCurrentQuestions([], false);
        mock.timers.tick(60000);
        assert.equal(fx.cards().length, 2, 'source failure cannot discharge the records');
        fx.decision.syncCurrentQuestions(rows.map(row => ({ ...row,
            question_delivery: { ...row.question_delivery, state: 'delivered' } })), true);
        mock.timers.tick(5000);
        assert.equal(fx.cards().length, 0);
        fx.decision.syncCurrentQuestions(rows, true);
        for (const row of rows) fx.add(row);
        assert.equal(fx.cards().length, 0, 'stale owed delivery cannot undo a canonical receipt');
    } finally { fx.decision.destroy(); fx.restore(); mock.timers.reset(); }
});

test('owner removes an unknown late answer from pending once, without answering again or resending', async () => {
    mock.timers.enable({ apis: ['setTimeout'] });
    const unknown = { kind: 'answer', state: 'owed', reason_code: 'late_answer_delivery_unknown' };
    let attempts = 0;
    const fx = feedFixture({ fetchImpl: async () => {
        attempts++;
        if (attempts === 1) throw new Error('lost response');
        return { ok: true, status: 200, json: async () => ({ ok: true, state: 'answered', answered_index: 1,
            duplicate: false, question_delivery: { ...unknown, dismissed: true } }) };
    } });
    try {
        const rows = [question('unknown', { quiz_state: 'answered', answered_index: 1, question_delivery: unknown }),
            question('pending', { quiz_state: 'answered', answered_index: 0, question_delivery: { kind: 'answer', state: 'owed' } })];
        for (const row of rows) fx.add(row);
        const [card, pending] = fx.cards();
        assert.match(card.querySelector('.chat-quiz-status-text').textContent, /Answer recorded.*delivery unknown/);
        const remove = card.querySelector('.chat-quiz-dismiss');
        assert.equal(remove.textContent, 'Remove from pending');
        assert.equal(pending.querySelector('.chat-quiz-dismiss'), null, 'only an unknown delivery is removable');
        remove.click(); await turn();
        assert.ok(card.querySelector('.chat-quiz-dismiss'), 'a lost response removes nothing');
        card.querySelector('.chat-quiz-dismiss').click(); await turn();
        const [first, retry] = fx.calls.map(call => JSON.parse(call.init.body));
        assert.deepEqual(retry, first, 'the retry replays the same removal request');
        assert.deepEqual(Object.keys(first).sort(), ['action', 'decision_id', 'request_id']);
        assert.equal(first.action, 'dismiss'); assert.equal(first.decision_id, 'quiz:task-1:unknown');
        assert.equal(card.querySelector('.chat-quiz-dismiss'), null);
        assert.match(card.querySelector('.chat-quiz-status-text').textContent, /delivery unknown/, 'the record stays honest');
        mock.timers.tick(5000);
        assert.deepEqual(fx.cards().map(view => view.dataset.quizId), ['pending']);
        fx.decision.syncCurrentQuestions(rows, true);
        fx.add(rows[0]);
        assert.deepEqual(fx.cards().map(view => view.dataset.quizId), ['pending'], 'a stale census or history row cannot bring it back');
    } finally { fx.decision.destroy(); fx.restore(); mock.timers.reset(); }
});

test('a removal recorded by another client settles this Main copy and drops its control', () => {
    mock.timers.enable({ apis: ['setTimeout'] });
    const unknown = { kind: 'answer', state: 'owed', reason_code: 'late_answer_delivery_unknown' };
    const fx = feedFixture({ fetchImpl: async () => assert.fail('this client never posts') });
    try {
        fx.add(question('elsewhere', { quiz_state: 'answered', answered_index: 0, question_delivery: unknown }));
        assert.ok(fx.cards()[0].querySelector('.chat-quiz-dismiss'));
        fx.decision.applyQuizStateFrame(fx.feed, { task_id: 'task-1', quiz_id: 'elsewhere', state: 'answered',
            answered_index: 0, question_delivery: { ...unknown, dismissed: true } });
        assert.equal(fx.cards()[0].querySelector('.chat-quiz-dismiss'), null);
        mock.timers.tick(5000);
        assert.equal(fx.cards().length, 0);
        assert.equal(fx.calls.length, 0);
    } finally { fx.decision.destroy(); fx.restore(); mock.timers.reset(); }
});

test('answer acknowledgement with pending delivery remains recorded and visible', async () => {
    mock.timers.enable({ apis: ['setTimeout'] });
    // The exact read's canonical block, with its delivery as the server records it at that moment.
    let delivery = { kind: 'answer', state: 'owed' };
    const fx = feedFixture({ fetchImpl: async () => ({ ok: true, json: async () => ({ ok: true,
        state: 'answered', answered_index: 0, answered_after_terminal: true, forwarded: false,
        question_delivery: { kind: 'answer', state: 'owed' } }) }),
        fetchDetail: async (taskId) => ({ task_id: taskId, project_id: 'project-1', owner_quiz: { ack: {
            quiz_id: 'ack', question: 'Question ack?', options: ['Yes', 'No'], state: 'answered', answered_index: 0, delivery } } }) });
    try {
        fx.add(question('ack'));
        fx.cards()[0].querySelector('.chat-quiz-option').click();
        await turn();
        assert.equal(fx.cards()[0].dataset.state, 'answered');
        assert.match(fx.toasts.at(-1).text, /delivery.*pending/);
        mock.timers.tick(60000);
        assert.equal(fx.cards().length, 1);
        // Complete withdrawal proves no delivery (#1993): the copy's own exact read decides.
        fx.decision.syncCurrentQuestions([], true);
        await turn();
        assert.equal(fx.cards().length, 1, 'an owed record never retires from absence');
        assert.match(fx.cards()[0].querySelector('.chat-quiz-status-text').textContent, /delivery pending/);
        delivery = { kind: 'answer', state: 'delivered' };
        fx.decision.syncCurrentQuestions([], true);
        await turn();
        assert.equal(fx.cards()[0].querySelector('.chat-quiz-status-text').textContent, 'You answered');
        mock.timers.tick(5000);
        assert.equal(fx.cards().length, 0, 'the read that proves delivery shows the recorded answer, then the copy leaves');
    } finally { fx.decision.destroy(); fx.restore(); mock.timers.reset(); }
});

test('question materials use validated artifact identities and display their original labels', () => {
    const fx = feedFixture();
    try {
        fx.add(question('material', { materials: [
            { task_id: 'task-1', name: 'material-a.csv', label: 'Schedule.csv', sha256: 'abc', download_url: 'javascript:alert(1)' },
            { task_id: 'task-1', name: '../secrets', sha256: 'abc' },
            { name: 'unbound.csv', sha256: 'abc' },
        ] }));
        const links = fx.cards()[0].querySelector('.chat-quiz-materials').children;
        assert.equal(links.length, 1);
        assert.equal(links[0].textContent, 'Schedule.csv');
        assert.equal(links[0].href, '/api/tasks/task-1/artifacts/material-a.csv');
    } finally { fx.decision.destroy(); fx.restore(); }
});

test('question photos reuse the existing file preview and the reader, from their custody address', async t => {
    // The media stub hosts the reader the way main_durable_edges does: a document, focus, replaceChildren.
    const ownerDocument = Object.getOwnPropertyDescriptor(MediaNode.prototype, 'ownerDocument');
    const replaceChildren = MediaNode.prototype.replaceChildren;
    Object.defineProperty(MediaNode.prototype, 'ownerDocument', { configurable: true, get() { return globalThis.document; } });
    MediaNode.prototype.replaceChildren = function(...nodes) { this.children = []; this.append(...nodes); };
    t.after(() => {
        if (ownerDocument) Object.defineProperty(MediaNode.prototype, 'ownerDocument', ownerDocument);
        else delete MediaNode.prototype.ownerDocument;
        if (replaceChildren) MediaNode.prototype.replaceChildren = replaceChildren; else delete MediaNode.prototype.replaceChildren;
    });
    const fx = mediaFixture();
    try {
        const card = document.createElement('div');
        document.body.append(card);
        let rendered = '';
        syncQuestionMaterials(card, [{ task_id: 'task-1', name: 'material-safe.png', label: 'Design.png',
            size: 24, sha256: 'abc', download_url: '/api/files/download?path=arbitrary' }], row => {
            const bubble = fx.controller.buildDocumentBubble(row);
            rendered = bubble.innerHTML;
            return bubble;
        });
        const content = card.querySelector('.message');
        assert.ok(content.querySelector('.chat-file-thumb'));
        assert.match(rendered, /\/api\/tasks\/task-1\/artifacts\/material-safe.png/);
        assert.doesNotMatch(rendered, /arbitrary/);
        await card.querySelector('.chat-file-card').click();
        const reader = document.body.querySelector('.document-reader');
        assert.ok(reader?.attributes.has('open'), 'the photo opens in the reader, not outside the app');
        assert.ok(reader.querySelector('[data-reader-action="download"]'), 'Download stays');
        const image = fx.tracker.created.filter((node) => node.tagName === 'IMG').at(-1);
        assert.equal(image.getAttribute('src'), '/api/tasks/task-1/artifacts/material-safe.png');
        fx.controller.release(card);
        assert.equal(document.body.querySelector('.document-reader'), null, 'retiring the question closes its reader');
    } finally { fx.controller.destroy(); fx.restore(); }
});

test('Main model access uses the existing revision-bound recovery decision and refreshes selection', async () => {
    const calls = [];
    const wait = { wait_id: 'login', revision: 1, task_attempt: 1, role: 'main', reason: 'auth',
        model: 'claudexor::route=model', state: 'waiting', auto_continue: true };
    const env = installDom(async (url, init) => {
        calls.push({ url, body: JSON.parse(init.body) });
        return { ok: true, json: async () => ({ ok: true, wait: { ...wait, revision: 2, state: 'resolved' } }) };
    });
    const main = mainWaiting(env);
    try {
        const row = { task_id: 'access-task', project_id: 'build', project_name: 'Build', phase: 'working',
            task_attempt: 1, model_waits: { login: wait }, attention_reason: 'model_access' };
        main.waiting.snapshot({ main_attention: [row], main_attention_complete: true });
        const message = find(main.messagesDiv, node => node.dataset.ownerAction === 'access-task');
        assert.ok(message, 'the blocked task is one System message in the feed');
        assert.equal(find(message, node => node.classList.contains('owner-action-reason')).textContent,
            'Needs you to restore model access.');
        const retry = find(message, node => Object.hasOwn(node.dataset, 'waitRetry'));
        assert.ok(retry, 'the model-wait controller mounted its actual recovery controls in the message');
        retry.click();
        await turn();
        assert.equal(calls.length, 1);
        assert.equal(calls[0].url, '/api/decisions');
        assert.equal(calls[0].body.decision_id, 'model_wait:access-task:login');
        assert.equal(calls[0].body.revision, 1);
        assert.equal(calls[0].body.action, 'retry');
        assert.equal(find(env.mount, node => node.classList.contains('model-wait-row')), null);
        main.waiting.snapshot({ main_attention: [{ ...row, model_waits: {} }], main_attention_complete: true });
        assert.equal(find(env.mount, node => node.classList.contains('model-wait-row')), null);
    } finally { main.destroy(); restoreDom(env.prior); }
});

test('current Main pause offers the existing real Resume action and withdraws it with current authority', async () => {
    mock.timers.enable({ apis: ['setTimeout'] });
    const calls = [];
    const env = installDom(async (url, init) => {
        calls.push({ url, init });
        return { ok: true, json: async () => ({ ok: true }) };
    });
    const main = mainWaiting(env);
    try {
        document.body = env.mount;
        const row = { task_id: 'paused', activity_id: 'paused', project_id: 'build', project_name: 'Build',
            phase: 'budget_paused', pause_cause: 'budget', can_resume: true, attention_reason: 'budget', model_waits: {} };
        main.waiting.snapshot({ main_attention: [row], main_attention_complete: true, current_questions_complete: true });
        const resume = find(env.mount, node => node.dataset.resumeRun === '1');
        assert.ok(resume);
        await resume.listeners.get('click')[0]({ stopPropagation() {} });
        assert.equal(calls[0].url, '/api/tasks/paused/resume');
        assert.equal(calls[0].init.method, 'POST');
        main.waiting.setConnected(false);
        assert.equal(resume.disabled, true);
        main.waiting.snapshot({ main_attention: [], main_attention_complete: false });
        assert.ok(find(env.mount, node => node.dataset.taskId === 'paused'));
        main.waiting.setConnected(true);
        assert.equal(resume.disabled, true, 'reconnection alone cannot restore stale control authority');
        main.waiting.snapshot({ main_attention: [], main_attention_complete: true, current_questions_complete: true });
        assert.equal(find(env.mount, node => node.dataset.taskId === 'paused'), null);
        assert.equal(find(main.messagesDiv, node => node.dataset.ownerAction === 'paused'), null, 'its System message leaves the feed');
    } finally { main.destroy(); restoreDom(env.prior); mock.timers.reset(); }
});

test('a complete read that lands while disconnected restores nothing; only the census after reopening does', () => {
    const env = installDom(async () => assert.fail('withdrawn authority makes no request'));
    const calls = [], links = [];
    const main = mainWaiting(env, { decision: () => ({ syncCurrentQuestions: (...args) => calls.push(args),
        setConnected: value => links.push(value) }) });
    try {
        document.body = env.mount;
        const wait = { wait_id: 'login', revision: 1, task_attempt: 1, role: 'main', reason: 'auth',
            model: 'claudexor::route=model', state: 'waiting', auto_continue: true };
        const rows = [question('current')];
        const census = { current_questions: rows, current_questions_complete: true, main_attention_complete: true,
            main_attention: [{ task_id: 'paused', activity_id: 'paused', project_id: 'build', project_name: 'Build',
                phase: 'budget_paused', pause_cause: 'budget', can_resume: true, cancelable: true,
                attention_reason: 'budget', task_attempt: 1, model_waits: { login: wait } }] };
        main.waiting.snapshot(census, 100);
        const resume = find(env.mount, node => node.dataset.resumeRun === '1');
        const stop = find(env.mount, node => node.textContent === 'Stop…');
        const waits = find(env.mount, node => node.classList.contains('model-waits'));
        const timeline = [];
        const observe = stage => timeline.push({ stage, resume: !resume.disabled, stop: !stop.disabled,
            waits: !waits.inert, questions: calls.length, connected: links.at(-1) ?? true });
        observe('census');
        main.waiting.setConnected(false);
        observe('disconnected');
        main.waiting.snapshot(census, 200);  // HTTP still answers while the socket is down
        observe('offline body');
        main.waiting.setConnected(true);
        observe('reopened');
        main.waiting.snapshot(census, 300);
        observe('fresh census');
        const restored = { resume: true, stop: true, waits: true };
        const withdrawn = { resume: false, stop: false, waits: false };
        assert.deepEqual(timeline, [{ stage: 'census', ...restored, questions: 1, connected: true },
            { stage: 'disconnected', ...withdrawn, questions: 1, connected: false },
            { stage: 'offline body', ...withdrawn, questions: 1, connected: false },
            { stage: 'reopened', ...withdrawn, questions: 1, connected: true },
            { stage: 'fresh census', ...restored, questions: 2, connected: true }]);
        assert.deepEqual(calls.map(([sent, vouched, at]) => [sent, vouched('task-1', 'current'), at]),
            [[rows, true, 100], [rows, true, 300]]);
    } finally { main.destroy(); restoreDom(env.prior); }
});

test('a Main question regains its form only after the reopen: no offline pointer or read crossing the close restores it', async t => {
    const original = ElementStub.prototype.querySelector;
    ElementStub.prototype.querySelector = function(selector) {
        return original.call(this, selector) || this.children.map(child => child.querySelector(selector)).find(Boolean) || null;
    };
    ElementStub.prototype.replaceWith = function(node) { this.parentNode.replaceChild(node, this); };
    const priorCss = globalThis.CSS;
    globalThis.CSS = { escape: value => String(value) };  // the browser global Main's navigator looks cards up with
    t.after(() => { ElementStub.prototype.querySelector = original; delete ElementStub.prototype.replaceWith; globalThis.CSS = priorCss; });
    const asked = { task_id: 'question-task', quiz_id: 'cached', quiz_state: 'open', project_id: 'build', project_name: 'Build',
        project_chat_id: 12, question: 'Keep this output?', options: ['Yes', 'No'], ts: '2026-10-13T10:00:00Z' };
    const newer = { ...asked, quiz_id: 'newer', question: 'A newer live question?', ts: '2026-10-13T10:05:00Z' };
    const detail = { task_id: 'question-task', project_id: 'build',
        question_rooms: Object.fromEntries([asked, newer].map(row => [row.quiz_id, { state: 'known', chat_id: 12, project_id: 'build' }])),
        owner_quiz: Object.fromEntries(
        [asked, newer].map(row => [row.quiz_id, { ...row, state: 'open', asked_at: row.ts }])) };
    const reads = [];
    let started = 0;
    const env = installDom(url => String(url) === '/api/tasks/question-task'
        ? new Promise(resolve => { started += 1; reads.push(() => resolve({ ok: true, json: async () => detail })); })
        : Promise.resolve({ ok: true, json: async () => (String(url).startsWith('/api/chat/history')
            ? { messages: [], window: { complete: true } } : {}) }));
    const handlers = new Map();
    let generation = 0;
    const instance = createChatInstance({
        ws: { on(type, fn) { handlers.set(type, [...(handlers.get(type) || []), fn]); return () => {}; },
            isConnected: () => true, send() {} },
        state: { activePage: 'chat', projectChatIds: new Set(), unreadCount: 0 }, updateUnreadBadge() {},
        // The sequencer is app.js's; here each body is handed to the real Main consumer directly.
        stateSnapshots: { begin: () => ({ generation: ++generation, requestedAt: Date.now() }),
            gate() { return Promise.resolve(this.begin()); }, isCurrent: () => true, apply() {}, fail() {}, latest: () => null },
        chatId: 1, idPrefix: 'chat', mountEl: env.mount });
    const emit = (type, msg = {}) => (handlers.get(type) || []).forEach(fn => fn(msg));
    // Reads start a turn after their request; each lands only once everything requested so far was asked.
    const land = async (count = Infinity) => {
        await turn();
        while (reads.length && count-- > 0) reads.shift()();
        for (let i = 0; i < 8; i += 1) await turn();
    };
    // A real waiting Project question: the census names it and its activity carries the pointer (gateway/state.py).
    const census = (patch = {}) => instance.hydrateStateSnapshot({ current_questions: [asked], current_questions_complete: true,
        main_attention: [], main_attention_complete: true, active_chat_activities_complete: true,
        active_chat_activities: [{ activity_id: 'question-task', chat_id: 12, project_id: 'build',
            required_question: { ...asked, owner_wait_state: 'waiting' } }], ...patch }, Infinity);
    const card = id => find(env.mount, node => node.classList?.contains('chat-quiz-card') && node.dataset.quizId === id);
    const comment = () => card('cached').querySelector('.chat-quiz-comment');
    const timeline = [];
    const view = stage => timeline.push({ stage, state: card('cached').dataset.state, comment: !comment().disabled, reads: started });
    try {
        census();
        await land();
        assert.ok(document.byId.get('chat-messages').contains(card('cached')), 'the copy takes its place in the feed');
        comment().value = 'Draft kept across the drop';
        view('census');
        emit('close');
        view('closed');
        census();  // HTTP still answers while the socket is down, and its pointer's read too
        await land();
        view('offline body');
        emit('open', { previouslyConnected: true });
        view('reopened, census held');
        census();
        view('fresh census');
        // An incomplete census leaves the copy to its own canonical read, still in flight at the next close.
        census({ current_questions: [], current_questions_complete: false });
        await turn();  // its request leaves before the close
        view('read in flight');
        emit('close');
        emit('open', { previouslyConnected: true });
        // Connected again, the copy's next pointer reads anew while the older read is still out.
        census({ current_questions: [], current_questions_complete: false });
        await land(1);
        view('read begun before the close lands after the reopen');
        census({ current_questions: [], current_questions_complete: false });
        await turn();
        view('a further pointer while the newer read is out');
        await land();
        view('the read begun after the reopen gives the form back');
        // A pointer arriving on the reopened socket is a live delivery: nothing was forgotten, so its
        // form needs no canonical read.
        emit('chat', { ...newer, type: 'chat', chat_id: 1, role: 'system', system_type: 'project_question_pointer' });
        await land();
        const live = card('newer').dataset.state;
        census();
        view('fresh census again');
        const on = { state: 'open', comment: true }, off = { state: 'unknown', comment: false };
        assert.deepEqual(timeline, [{ stage: 'census', ...on, reads: 0 }, { stage: 'closed', ...off, reads: 0 },
            { stage: 'offline body', ...off, reads: 0 }, { stage: 'reopened, census held', ...off, reads: 0 },
            { stage: 'fresh census', ...on, reads: 0 }, { stage: 'read in flight', ...off, reads: 1 },
            { stage: 'read begun before the close lands after the reopen', ...off, reads: 2 },
            { stage: 'a further pointer while the newer read is out', ...off, reads: 2 },
            { stage: 'the read begun after the reopen gives the form back', ...on, reads: 2 },
            { stage: 'fresh census again', ...on, reads: 2 }]);
        assert.equal(live, 'open', 'a live pointer on the reopened socket mounts its own form');
        assert.equal(comment().value, 'Draft kept across the drop');
    } finally { instance.destroy(); restoreDom(env.prior); }
});

test('an interrupted owner Resume of a member offers the same Resume; the owner\'s own Pause never does', async () => {
    const calls = [];
    const env = installDom(async (url, init) => {
        calls.push({ url, init });
        return { ok: true, json: async () => ({ ok: true }) };
    });
    const main = mainWaiting(env);
    try {
        document.body = env.mount;
        // The census rows of a member whose consumed owner Resume a stop interrupted (#1716):
        // a held boot restore says restart, an acknowledged return has no cause of its own.
        const member = (id, cause, reason) => ({ task_id: id, activity_id: id, project_id: 'build',
            project_name: 'Build', root_task_id: 'root', phase: 'budget_paused', pause_cause: cause,
            can_resume: true, cancelable: false, attention_reason: reason, model_waits: {} });
        main.waiting.snapshot({ main_attention: [member('held', 'restart', 'restart'),
            member('returned', 'unknown', 'resume'), member('own', 'owner', 'resume')],
            main_attention_complete: true, current_questions_complete: true });
        const card = id => find(env.mount, node => node.dataset.taskId === id);
        const resume = id => find(card(id), node => node.dataset.resumeRun === '1');
        assert.equal(find(card('held'), node => node.classList.contains('owner-action-reason')).textContent,
            'Needs Resume after the restart.');
        assert.equal(find(card('returned'), node => node.classList.contains('owner-action-reason')).textContent,
            'Needs an explicit Resume.');
        assert.equal(resume('own'), null, 'the owner\'s own Pause is never asked for again');
        assert.equal(find(card('held'), node => node.textContent === 'Stop…'), null, 'Stop addresses roots only');
        for (const id of ['held', 'returned']) {
            await resume(id).listeners.get('click')[0]({ stopPropagation() {} });
            assert.equal(calls.at(-1).url, `/api/tasks/${id}/resume`);
            assert.equal(calls.at(-1).init.method, 'POST');
        }
        assert.equal(calls.length, 2);
    } finally { main.destroy(); restoreDom(env.prior); }
});

test('Main completion has one Project reference, its caveat and time, without result controls or reads', t => {
    const originalAfter = ElementStub.prototype.after;
    ElementStub.prototype.after = function(node) { return this.insertAdjacentElement('afterend', node); };
    t.after(() => { if (originalAfter) ElementStub.prototype.after = originalAfter; else delete ElementStub.prototype.after; });
    const env = installDom();
    try {
        for (const role of ['assistant', 'system']) {
            const bubble = document.createElement('div');
            const message = document.createElement('div');
            message.className = 'message';
            bubble.append(message);
            env.mount.append(bubble);
            const actions = decorateProjectRow(bubble, { role, projectId: 'project-1', projectName: 'Build',
                systemType: 'project_completion_summary', taskId: 'task-1', limitation: 'Windows is not verified.',
                fetchDetail() { assert.fail('The Main mirror must not read the task inventory'); } });
            assert.equal(actions.children.length, 1);
            assert.equal(actions.children[0].dataset.intent, 'open-project');
            assert.equal(bubble.querySelector('.project-answer-files'), null);
            assert.equal(bubble.querySelector('[data-answer-fold]'), null);
            assert.equal(Boolean(bubble.querySelector('.project-answer-limitation')), role === 'assistant');
            // One compact time line (owner 17.10, Q7 A): the row's own time carries the end, no second note.
            const time = bubble.querySelector('.msg-time');
            assert.equal(time.title, 'Task end time not recorded');
            assert.equal(time.children[0].textContent, 'End time not recorded');
            assert.equal(bubble.children.filter(node => node.classList.contains('msg-provenance')).length,
                role === 'assistant' ? 1 : 0);
        }
    } finally { restoreDom(env.prior); }
});
