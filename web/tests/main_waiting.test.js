// Main's waiting navigator (docs/DESIGN.md "Project question mirror"): questions keep their
// place in the feed, a task that cannot continue without the owner is one System message there,
// and one amber pill in the composer row reaches whichever of them is off screen.
import assert from 'node:assert/strict';
import test from 'node:test';
import { ElementStub, installDom, restoreDom } from './chat_dom_fixture.js';
import { stampNodeTimestamp } from '../modules/chat_activity.js';
import { createMainWaiting } from '../modules/main_waiting.js';

const turn = () => new Promise(resolve => setTimeout(resolve, 0));
const find = (node, predicate) => predicate(node) ? node : node.children.reduce((found, child) => found || find(child, predicate), null);
const question = (id, patch = {}) => ({ task_id: 'task-1', quiz_id: id, project_id: 'project-1', project_chat_id: 12,
    project_name: 'Build', quiz_state: 'open', question: `Question ${id}?`, options: ['Yes', 'No'],
    ts: '2020-01-01T09:00:00Z', ...patch });
const action = (id, patch = {}) => ({ task_id: id, activity_id: id, project_id: 'build', project_name: 'Build',
    task_name: `Task ${id}`, phase: 'budget_paused', pause_cause: 'budget', can_resume: true, cancelable: true,
    attention_reason: 'budget', model_waits: {}, ...patch });
const census = ({ questions = [], actions = [], complete = true } = {}) => ({ current_questions: questions,
    current_questions_complete: complete, main_attention: actions, main_attention_complete: complete });

// The lookups main_waiting.js makes, answered the browser's way: descendants by class and data
// attributes (`.chat-quiz-card[data-task-id="t"][data-quiz-id="q"]`) and `closest('.chat-bubble')`.
function browserQueries(t) {
    const { querySelector, closest } = ElementStub.prototype;
    const parse = selector => selector.match(/^\.([\w-]+)((?:\[data-[\w-]+="[^"]*"\])*)$/);
    const matches = (node, [, name, attrs]) => node.classList.contains(name)
        && [...attrs.matchAll(/\[data-([\w-]+)="([^"]*)"\]/g)].every(([, key, value]) =>
            node.dataset[key.replace(/-([a-z])/g, (_all, char) => char.toUpperCase())] === value);
    ElementStub.prototype.querySelector = function(selector) {
        const parsed = parse(selector);
        if (!parsed) return querySelector.call(this, selector);
        for (const child of this.children) {
            if (matches(child, parsed)) return child;
            const nested = child.querySelector(selector);
            if (nested) return nested;
        }
        return null;
    };
    ElementStub.prototype.closest = function(selector) {
        const parsed = parse(selector);
        if (!parsed) return closest.call(this, selector);
        for (let node = this; node; node = node.parentElement) if (matches(node, parsed)) return node;
        return null;
    };
    t.after(() => Object.assign(ElementStub.prototype, { querySelector, closest }));
}

// Main as a browser lays it out: the feed spans 0..600 under a floating header that ends at 60
// and over the composer row that starts at 560, so the reading area is 60..560. Feed nodes are
// 200px tall, 1000px apart in time order (the timeline inserts by `data-ts`), and move up as the
// feed scrolls down. `CSS.escape` is the browser global the card lookup uses.
const ROW = 1000;
const box = (top, height) => ({ top, bottom: top + height, height, left: 0, right: 600, width: 600 });
function mainPage(t, { fetchImpl, observedQuestion = () => null, ...options } = {}) {
    browserQueries(t);
    const priorCss = globalThis.CSS;
    globalThis.CSS = { escape: value => String(value) };
    const env = installDom(fetchImpl);
    document.body = env.mount;  // toasts from the task controls land here
    t.after(() => { restoreDom(env.prior); globalThis.CSS = priorCss; });
    const page = document.createElement('div');
    const header = document.createElement('div');
    header.className = 'chat-page-header';
    const feed = document.createElement('div');
    const toolbar = document.createElement('div');
    toolbar.className = 'chat-toolbar-row';
    toolbar.append(document.createElement('button'));  // the composer's own pills come first
    page.append(header, feed, toolbar);
    env.mount.append(page);
    header.getBoundingClientRect = () => box(0, 60);
    feed.getBoundingClientRect = () => box(0, 600);
    toolbar.getBoundingClientRect = () => box(560, 40);
    Object.defineProperty(feed, 'scrollHeight', { configurable: true, get: () => feed.children.length * ROW, set() {} });
    const insert = node => {
        const later = feed.children.find(el => Number(el.dataset.ts) > Number(node.dataset.ts));
        feed.insertBefore(node, later || null);
        node.getBoundingClientRect = () => box(feed.children.indexOf(node) * ROW - feed.scrollTop, 200);
        return true;
    };
    const message = ts => {
        const bubble = document.createElement('div');
        bubble.className = 'chat-bubble';
        stampNodeTimestamp(bubble, ts);
        return bubble;
    };
    // A question's feed copy: the chat bubble around its card, whose question text takes focus.
    const card = row => {
        const bubble = message(row.ts);
        const quiz = document.createElement('div');
        quiz.className = 'chat-quiz-card';
        quiz.dataset.taskId = row.task_id;
        quiz.dataset.quizId = row.quiz_id;
        const text = document.createElement('div');
        text.className = 'chat-quiz-question';
        quiz.append(text);
        bubble.append(quiz);
        insert(bubble);
        return bubble;
    };
    const calls = { sync: [], connect: [], inserted: [], stamps: [], mounted: [], authority: [], revealed: [], reading: 0 };
    // The question owner's surface the navigator uses (chat_decision.js), recorded.
    const owner = { syncCurrentQuestions: (...args) => calls.sync.push(args), setConnected: value => calls.connect.push(value),
        appendQuestionPointer: (row, authority) => { calls.mounted.push(row); calls.authority.push(authority); card(row); return true; },
        observedQuestion };
    const waiting = createMainWaiting({ page, messagesDiv: feed, toolbar,
        insertMessageNode: node => { calls.inserted.push(node); return insert(node); },
        stampNodeTimestamp: (node, raw, opts) => { calls.stamps.push(raw); return stampNodeTimestamp(node, raw, opts); },
        decision: () => owner, reading: () => ({ claim: () => { calls.reading += 1; }, stick: true }), ...options });
    t.after(() => waiting.destroy());
    const pill = toolbar.children.at(-1);
    const shown = () => pill.hidden ? null : {
        text: pill.querySelector('.chat-question-nav-count').textContent + pill.querySelector('.chat-question-nav-word').textContent,
        label: pill.getAttribute('aria-label'),
        down: pill.querySelector('.chat-question-nav-arrow').classList.contains('down'),
    };
    const scrollTo = top => { feed.scrollTop = top; for (const fn of feed.listeners.get('scroll') || []) fn(); };
    const press = () => Promise.all(pill.listeners.get('click').map(fn => fn({ target: pill })));
    const actionNode = id => find(feed, node => node.dataset.ownerAction === id);
    const filler = ts => insert(message(ts));
    return { env, page, feed, toolbar, waiting, pill, calls, card, filler, shown, scrollTo, press, actionNode };
}

test('the pill is one of the composer pills: a button at the end of its row, hidden while nothing waits', t => {
    const main = mainPage(t);
    assert.equal(main.pill.tagName, 'BUTTON');
    assert.equal(main.pill.type, 'button');
    assert.equal(main.pill.className, 'chat-question-nav');
    assert.equal(main.toolbar.children.indexOf(main.pill), 1, 'after the row\'s own pills');
    assert.deepEqual(main.pill.children.map(node => node.className).filter(Boolean),
        ['chat-question-nav-glyph', 'chat-question-nav-count', 'chat-question-nav-word', 'chat-question-nav-arrow']);
    assert.equal(main.pill.hidden, true);
    main.waiting.snapshot(census());
    assert.equal(main.shown(), null, 'no question and no action: nothing to go to');
    main.waiting.snapshot(census({ questions: [question('q1')] }));
    assert.notEqual(main.shown(), null);
    main.waiting.snapshot(census());
    assert.equal(main.shown(), null);
});

test('a snapshot hands its questions to their owner with what it vouches for and its request time; none while disconnected', t => {
    const main = mainPage(t);
    const rows = [question('q1')];
    main.waiting.snapshot(census({ questions: rows }), 100);
    main.waiting.snapshot({ current_questions: rows }, 110);
    main.waiting.snapshot(null, 120);
    main.waiting.snapshot(census({ questions: rows }));
    assert.deepEqual(main.calls.sync.map(([sent, vouched, at]) => [sent, vouched('task-1', 'q1'), at]),
        [[rows, true, 100], [rows, false, 110], [[], false, 120], [rows, true, Infinity]]);
    main.waiting.setConnected(false);
    assert.deepEqual(main.calls.connect, [false], 'the question owner withdraws its forms with the socket');
    main.waiting.snapshot(census({ questions: [question('q2'), question('q3')], actions: [action('late')] }), 200);
    assert.equal(main.calls.sync.length, 4, 'a body read while the socket is down restores nothing');
    assert.equal(main.actionNode('late'), null);
    assert.equal(main.shown().text, '1 question', 'nor does it change what the pill counts');
    main.waiting.setConnected(true);
    assert.deepEqual(main.calls.connect, [false, true]);
    assert.equal(main.calls.sync.length, 4, 'reconnection alone is not a census');
    main.waiting.snapshot(census({ questions: [question('q2'), question('q3')], actions: [action('late')] }), 300);
    assert.equal(main.calls.sync.at(-1)[2], 300);
    assert.ok(main.actionNode('late'));
    assert.equal(main.shown().text, '3 waiting');
});

test('the pill counts unanswered questions and owner actions, in words that name them', t => {
    const main = mainPage(t);
    main.feed.scrollTop = 100 * ROW;  // reading far below: every message stays off screen
    const cases = [
        [{ questions: [question('q1')] }, '1 question', '1 unanswered: show the next question'],
        [{ questions: [question('q1'), question('q2')] }, '2 questions', '2 unanswered: show the next question'],
        [{ actions: [action('a1')] }, '1 action', '1 waiting for you: show the next one'],
        [{ actions: [action('a1'), action('a2')] }, '2 actions', '2 waiting for you: show the next one'],
        [{ questions: [question('q1'), question('q2')], actions: [action('a1')] }, '3 waiting', '3 waiting for you: show the next one'],
    ];
    for (const [snapshot, text, label] of cases) {
        main.waiting.snapshot(census(snapshot));
        assert.deepEqual([main.shown().text, main.shown().label, main.pill.title], [text, label, label], text);
    }
    // Only an unanswered question waits: a finished task's question still does, a settled one
    // whose delivery is merely owed does not.
    main.waiting.snapshot(census({ questions: [question('open'), question('finished', { quiz_state: 'expired_terminal' }),
        question('answered', { quiz_state: 'answered', answered_index: 0, question_delivery: { kind: 'answer', state: 'owed' } }),
        question('closed', { quiz_state: 'closed', question_delivery: { kind: 'closure', state: 'owed' } }),
        question('replaced', { quiz_state: 'superseded' })] }));
    assert.equal(main.shown().text, '2 questions');
});

// A restart can leave a late answer's delivery unknown; only the owner can take it out of pending,
// from its Main copy, so it waits on them like an action until they do.
test('a late answer whose delivery is unknown is an owner action the pill reaches, until it is removed', async t => {
    const main = mainPage(t);
    main.feed.scrollTop = 100 * ROW;
    const unknown = (patch = {}) => question('late', { quiz_state: 'answered', answered_index: 1,
        question_delivery: { kind: 'answer', state: 'owed', reason_code: 'late_answer_delivery_unknown', ...patch } });
    main.waiting.snapshot(census({ questions: [unknown()] }));
    assert.deepEqual([main.shown().text, main.shown().label], ['1 action', '1 waiting for you: show the next one']);
    const earlier = question('q1', { ts: '2020-01-01T08:00:00Z' });
    main.waiting.snapshot(census({ questions: [earlier, unknown()] }));
    assert.equal(main.shown().text, '2 waiting');
    await main.press();
    await main.press();
    assert.deepEqual(main.calls.mounted.map(row => row.quiz_id), ['q1', 'late'], 'it mounts its read-only copy at its own date');
    main.waiting.snapshot(census({ questions: [earlier, unknown({ dismissed: true })] }));
    main.feed.scrollTop = 100 * ROW;
    main.waiting.refresh();
    assert.equal(main.shown().text, '1 question', 'once removed from pending it waits no more');
    main.waiting.snapshot(census({ questions: [unknown({ reason_code: 'publication_address_unknown' })] }));
    assert.equal(main.shown(), null, 'an owed delivery the core still retries needs nobody');
});

// The census arrives every few seconds; an answer or close this tab already saw must not leave the
// pill counting it, or pointing a press at a card that has left the feed.
test('what this tab already saw settle leaves the count at once, before the next census', async t => {
    const seen = new Map();
    const main = mainPage(t, { observedQuestion: (task, quiz) => seen.get(`${task}:${quiz}`) || null });
    main.feed.scrollTop = 100 * ROW;
    const late = question('late', { quiz_state: 'answered', ts: '2020-01-01T10:00:00Z',
        question_delivery: { kind: 'answer', state: 'owed', reason_code: 'late_answer_delivery_unknown' } });
    main.waiting.snapshot(census({ questions: [question('done', { ts: '2020-01-01T08:00:00Z' }), question('next'), late] }));
    assert.equal(main.shown().text, '3 waiting');
    seen.set('task-1:done', { state: 'closed' });
    main.waiting.refresh();
    assert.equal(main.shown().text, '2 waiting', 'the close this tab saw leaves the count');
    await main.press();
    assert.equal(main.calls.mounted.at(-1).quiz_id, 'next', 'and a press never goes to it');
    seen.set('task-1:next', { state: 'open' });
    // A long-lived tab remembers the question's own publication receipt from before the answer
    // came from another device: that receipt says nothing about the answer's delivery.
    seen.set('task-1:late', { state: 'open', question_delivery: { kind: 'publication', state: 'delivered' } });
    main.feed.scrollTop = 100 * ROW;
    main.waiting.refresh();
    assert.equal(main.shown().text, '2 waiting', 'another delivery\'s receipt never settles the pending answer');
    // An answer this tab saw while the census still says "open" leaves the count, unless a restart
    // left its delivery unknown: then it still waits on the owner.
    seen.set('task-1:next', { state: 'answered', question_delivery: { kind: 'answer', state: 'owed',
        reason_code: 'late_answer_delivery_unknown' } });
    main.waiting.refresh();
    assert.equal(main.shown().text, '2 waiting', 'its own unknown delivery keeps it counted, with no gap');
    seen.set('task-1:next', { state: 'open' });
    seen.set('task-1:late', { state: 'answered', question_delivery: { kind: 'answer', state: 'owed', dismissed: true } });
    main.feed.scrollTop = 100 * ROW;
    main.waiting.refresh();
    assert.equal(main.shown().text, '1 question', 'a pending answer the owner removed waits no more; an open one still does');
});

test('the pill mounts a row the latest census named live, and one kept from an older read withheld', async t => {
    const main = mainPage(t);
    main.feed.scrollTop = 100 * ROW;
    const pressFor = async (id) => {
        await main.press();
        assert.equal(main.calls.mounted.at(-1).quiz_id, id);
        return main.calls.authority.at(-1);
    };
    main.waiting.snapshot(census({ questions: [question('named', { ts: '2020-01-01T09:00:00Z' })] }));
    assert.deepEqual(await pressFor('named'), { canonical: true }, 'the current census named it');
    main.waiting.snapshot(census({ questions: [question('kept', { ts: '2020-01-01T09:30:00Z' })] }));
    main.waiting.snapshot({ current_questions: [], current_questions_complete: false, main_attention: [],
        main_attention_complete: true, current_questions_gaps: [{ task_id: 'task-1', quiz_id: 'kept', reason: 'question_source_unavailable' }] });
    assert.equal(main.shown().text, '1 question', 'its gap keeps it counted');
    assert.deepEqual(await pressFor('kept'), { stale: true }, 'kept from an older read: withheld until its own read');
    main.waiting.snapshot(census({ questions: [question('reopened', { ts: '2020-01-01T10:00:00Z' })] }));
    main.waiting.setConnected(false);
    main.waiting.setConnected(true);
    assert.deepEqual(await pressFor('reopened'), { stale: true }, 'a reopen before the next census confirms nothing');
    main.waiting.snapshot(census({ questions: [question('again', { ts: '2020-01-01T10:30:00Z' })] }));
    assert.deepEqual(await pressFor('again'), { canonical: true });
});

test('an incomplete snapshot keeps the questions it cannot see; only a complete one lets them go', t => {
    const main = mainPage(t);
    main.waiting.snapshot(census({ questions: [question('q1'), question('q2')] }));
    assert.equal(main.shown().text, '2 questions');
    main.waiting.snapshot(census({ complete: false }));
    assert.equal(main.shown().text, '2 questions', 'a partial read never drops the count to zero');
    main.waiting.snapshot(null);
    assert.equal(main.shown().text, '2 questions', 'a failed read is a partial one');
    main.waiting.snapshot(census({ questions: [question('q3')], complete: false }));
    assert.equal(main.shown().text, '3 questions', 'what it does see joins what it keeps');
    main.waiting.snapshot(census({ questions: [question('q3')] }));
    assert.equal(main.shown().text, '1 question', 'a complete census is the whole current set');
    main.waiting.snapshot(census());
    assert.equal(main.shown(), null);
});

// An install can carry gaps for good (an old question whose room no import proved): an incomplete
// census still speaks for every question its gaps do not name.
test('an incomplete census vouches for every question its gaps do not name, and for nothing when a gap names no task', t => {
    const main = mainPage(t);
    const gap = (patch = {}) => ({ task_id: 'task-1', quiz_id: 'q1', reason: 'publication_address_unknown', ...patch });
    const partial = (gaps, questions = []) => ({ current_questions: questions, current_questions_complete: false,
        current_questions_gaps: gaps, main_attention: [], main_attention_complete: true });
    const probes = [['task-1', 'q1'], ['task-1', 'q2'], ['task-2', 'q1']];
    const vouching = data => { main.waiting.snapshot(data); const [, vouched] = main.calls.sync.at(-1); return probes.map(([task, quiz]) => vouched(task, quiz)); };
    assert.deepEqual(vouching(census()), [true, true, true], 'a complete census vouches for everything');
    assert.deepEqual(vouching(partial([gap()])), [false, true, true], 'a gap naming a question withholds that one');
    assert.deepEqual(vouching(partial([gap({ quiz_id: undefined })])), [false, false, true], 'a gap naming a task withholds all of its');
    assert.deepEqual(vouching(partial([gap(), { reason: 'question_index_unavailable' }])), [false, false, false],
        'a gap naming no task withholds everything');
    assert.deepEqual(vouching(partial([])), [false, false, false], 'unexplained incompleteness withholds everything');
    assert.deepEqual(vouching(null), [false, false, false], 'so does a failed read');
    // The pill drops an omitted question exactly when the census vouches for it.
    const known = [question('q1'), question('q2', { task_id: 'task-2' })];
    const counted = gaps => {
        main.waiting.snapshot(census({ questions: known }));
        main.waiting.snapshot(partial(gaps, [known[1]]));
        return main.shown().text;
    };
    assert.equal(counted([gap({ task_id: 'task-3', quiz_id: 'q9' })]), '1 question', 'gaps elsewhere: the omitted question has left');
    assert.equal(counted([gap()]), '2 questions', 'its own gap: it may still wait');
    assert.equal(counted([gap({ quiz_id: undefined })]), '2 questions');
    assert.equal(counted([gap({ task_id: 'task-3', quiz_id: 'q9' }), { reason: 'registry_unavailable' }]), '2 questions');
});

test('each owner-required action is one System message in the feed, first stamped when seen', t => {
    const main = mainPage(t);
    const before = Date.now();
    main.waiting.snapshot(census({ actions: [action('paused', { task_name: 'Nightly build' })] }));
    const after = Date.now();
    const message = main.actionNode('paused');
    assert.deepEqual(main.calls.inserted, [message], 'it enters through the feed insertion seam');
    assert.ok(['chat-bubble', 'system', 'owner-action'].every(name => message.classList.contains(name)));
    const [sender, body] = message.children;
    assert.deepEqual([sender.className, sender.textContent], ['sender', 'System']);
    assert.deepEqual([body.tagName, body.className, body.dataset.taskId], ['SECTION', 'owner-action-body', 'paused']);
    const [heading, reason, controls] = body.children;
    assert.equal(heading.className, 'model-wait-heading');
    assert.equal(heading.children[0].textContent, 'Nightly build');
    assert.deepEqual([reason.className, reason.textContent],
        ['owner-action-reason', 'Paused at the spending limit. Change the limit if needed, then Resume.']);
    assert.equal(controls.className, 'system-message-actions');
    assert.deepEqual(controls.children.map(node => node.dataset.intent || (node.dataset.resumeRun ? 'resume' : node.textContent)),
        ['open-project', 'resume', 'Stop…'], 'its Project, Resume and the task control');
    const stamped = Number(message.dataset.ts);
    assert.ok(stamped >= before && stamped <= after, 'it stands at the live edge: the time it was first seen');
    main.waiting.snapshot(census({ actions: [action('paused', { task_name: 'Nightly build' })] }));
    assert.deepEqual([main.calls.inserted.length, main.calls.stamps.length], [1, 1], 'a later census names the same message');
    // One message per task, named by the task or else its activity; a row naming neither has no message.
    main.waiting.snapshot(census({ actions: [action('paused'), { ...action(''), activity_id: 'activity-only' }, action('', { activity_id: '' })] }));
    assert.deepEqual(main.feed.children.map(node => node.dataset.ownerAction), ['paused', 'activity-only']);
});

test('the reason names what the task needs from the owner', t => {
    const main = mainPage(t);
    const rows = [action('restart', { attention_reason: 'restart' }), action('resume', { attention_reason: 'resume' }),
        action('access', { attention_reason: 'model_access' }), action('own', { attention_reason: '', reason: 'A reason of its own.' }),
        action('held', { attention_reason: '', project_admission_hold: { label: 'Waiting for a Project slot.' } })];
    main.waiting.snapshot(census({ actions: rows }));
    const reason = id => find(main.actionNode(id), node => node.classList.contains('owner-action-reason')).textContent;
    assert.deepEqual(rows.map(row => reason(row.task_id)), ['Needs Resume after the restart.', 'Needs an explicit Resume.',
        'Needs you to restore model access.', 'A reason of its own.', 'Waiting for a Project slot.']);
});

test('Resume is offered only for a resumable pause the owner did not make; the task control only when cancelable', t => {
    const main = mainPage(t);
    main.waiting.snapshot(census({ actions: [action('budget'), action('owner', { pause_cause: 'owner' }),
        action('blocked', { can_resume: false }), action('unstated', { can_resume: undefined }),
        action('restart', { pause_cause: 'restart', cancelable: false })] }));
    const has = (id, predicate) => Boolean(find(main.actionNode(id), predicate));
    const resume = node => node.dataset.resumeRun === '1';
    const control = node => node.textContent === 'Stop…';
    assert.deepEqual(['budget', 'owner', 'blocked', 'unstated', 'restart'].map(id => [id, has(id, resume), has(id, control)]), [
        ['budget', true, true], ['owner', false, true], ['blocked', false, true], ['unstated', false, true], ['restart', true, false]]);
});

test('the task control carries Resume only where the server allows it and no Resume button of its own stands', t => {
    const main = mainPage(t);
    const priorObserver = globalThis.IntersectionObserver;
    globalThis.IntersectionObserver = class { observe() {} disconnect() {} takeRecords() { return []; } };
    t.after(() => { globalThis.IntersectionObserver = priorObserver; });
    main.waiting.snapshot(census({ actions: [action('budget'), action('owner', { pause_cause: 'owner' }),
        action('blocked', { can_resume: false })] }));
    const menuOf = id => {
        const control = find(main.actionNode(id), node => node.textContent === 'Stop…');
        for (const fn of control.listeners.get('click') || []) fn({ target: control, stopPropagation() {}, preventDefault() {} });
        const items = [];
        const walk = node => { if (node?.dataset?.taskControl) items.push(node.dataset.taskControl); (node?.children || []).forEach(walk); };
        walk(main.env.mount);
        return items;
    };
    // The owner's own Pause has no Resume button: the menu carries it; a standing button or a refused Resume keeps it out.
    assert.ok(menuOf('owner').includes('resume'));
    assert.ok(!menuOf('budget').includes('resume') && menuOf('budget').includes('stop_now'));
    assert.ok(!menuOf('blocked').includes('resume') && menuOf('blocked').includes('stop_now'));
});

test('a complete census removes a message whose task no longer waits; an incomplete one keeps it with its controls withdrawn', t => {
    const main = mainPage(t);
    const row = action('paused');
    main.waiting.snapshot(census({ actions: [row] }));
    const message = main.actionNode('paused');
    const resume = find(message, node => node.dataset.resumeRun === '1');
    const control = find(message, node => node.textContent === 'Stop…');
    assert.deepEqual([resume.disabled, control.disabled], [false, false]);
    main.waiting.snapshot(census({ complete: false }));
    assert.equal(main.actionNode('paused'), message, 'a partial read is no withdrawal');
    assert.deepEqual([resume.disabled, control.disabled], [true, true], 'but its authority is not current');
    main.waiting.snapshot(census({ actions: [row] }));
    assert.deepEqual([main.actionNode('paused'), resume.disabled, control.disabled], [message, false, false]);
    assert.equal(main.calls.inserted.length, 1);
    main.waiting.snapshot(census());
    assert.equal(main.actionNode('paused'), null, 'a complete census without the task takes its message away');
    assert.equal(message.isConnected, false);
    assert.equal(main.shown(), null);
});

test('a lost connection withdraws every control and tells the question owner; only the next census restores them', async t => {
    const posts = [];
    const main = mainPage(t, { fetchImpl: async (url, init) => { posts.push({ url, init }); return { ok: true, json: async () => ({ ok: true }) }; } });
    main.waiting.snapshot(census({ actions: [action('paused')] }));
    const resume = find(main.actionNode('paused'), node => node.dataset.resumeRun === '1');
    const control = find(main.actionNode('paused'), node => node.textContent === 'Stop…');
    main.waiting.setConnected(false);
    assert.deepEqual([resume.disabled, control.disabled, main.calls.connect], [true, true, [false]]);
    await resume.listeners.get('click')[0]({ stopPropagation() {} });
    assert.equal(posts.length, 0, 'a withdrawn Resume sends nothing, even when pressed past its disabled state');
    main.waiting.setConnected(true);
    assert.deepEqual([resume.disabled, control.disabled, main.calls.connect], [true, true, [false, true]],
        'reconnection alone cannot restore control authority');
    main.waiting.snapshot(census({ actions: [action('paused')] }));
    assert.deepEqual([resume.disabled, control.disabled], [false, false]);
    await resume.listeners.get('click')[0]({ stopPropagation() {} });
    assert.deepEqual(posts.map(post => [post.url, post.init.method]), [['/api/tasks/paused/resume', 'POST']]);
});

test('destroy takes the pill and every owner-action message away', t => {
    const main = mainPage(t);
    main.waiting.snapshot(census({ questions: [question('q1')], actions: [action('a1'), action('a2')] }));
    assert.deepEqual([main.toolbar.children.includes(main.pill), main.feed.children.length], [true, 2]);
    main.waiting.destroy();
    assert.equal(main.toolbar.children.includes(main.pill), false);
    assert.deepEqual(main.feed.children, []);
});

test('the pill goes to the oldest target off screen, then onward past what is on screen, cycling', async t => {
    const main = mainPage(t);
    main.filler('2020-01-01T08:00:00Z');
    const first = main.card(question('first'));
    main.filler('2020-01-01T10:00:00Z');
    const last = main.card(question('last', { ts: '2999-01-01T00:00:00Z' }));
    main.feed.scrollTop = 2 * ROW - 72;  // reading the middle message
    main.waiting.snapshot(census({ questions: [question('first'), question('last', { ts: '2999-01-01T00:00:00Z' })],
        actions: [action('paused')] }));
    const paused = main.actionNode('paused');
    assert.equal(main.feed.children.indexOf(paused), 3, 'the action stands at the time it was first seen');
    assert.deepEqual(main.shown(), { text: '3 waiting', label: '3 waiting for you: show the next one', down: false });
    const landed = [];
    const step = async () => {
        await main.press();
        const node = [first, paused, last].find(candidate => candidate.getBoundingClientRect().top === 72);
        landed.push([node === first ? 'first' : node === paused ? 'paused' : node === last ? 'last' : '?',
            document.activeElement === (node.querySelector('.chat-quiz-question') || node)]);
    };
    await step();
    assert.equal(main.shown().down, true, 'next: the action below');
    main.scrollTo(3 * ROW - 200);  // the owner scrolls the action into view himself
    assert.equal(main.shown().down, true, 'the action on screen is passed over: next is the last question');
    await step();
    assert.equal(main.shown().down, false, 'after the last target the pill cycles back up to the first');
    await step();
    await step();
    assert.deepEqual(landed, [['first', true], ['last', true], ['first', true], ['paused', true]],
        'each lands just under the header with focus on the question text (the message itself for an action)');
    assert.equal(paused.tabIndex, -1, 'the action message is a programmatic focus target, not a tab stop');
    assert.equal(main.calls.reading, 4, 'every landing claims the reading position');
    assert.deepEqual([main.calls.mounted, main.calls.revealed], [[], []], 'loaded cards need no mounting');
});

test('an off-window question mounts at its own date (Project) or opens through its canonical read (Main\'s own)', async t => {
    const revealed = [];
    const native = { task_id: 'main-task', quiz_id: 'q-native', chat_id: 1, native: true, quiz_state: 'open',
        ts: '2020-01-01T11:00:00Z' };
    let main = null;
    main = mainPage(t, { revealQuestion: async (taskId, quizId, options) => {
        revealed.push([taskId, quizId, options]);
        await turn();
        main.card(native);
        return true;
    } });
    main.filler('2020-01-01T08:00:00Z');
    main.filler('2020-01-01T12:00:00Z');
    main.feed.scrollTop = ROW - 72;
    const project = question('project', { ts: '2020-01-01T10:00:00Z' });
    main.waiting.snapshot(census({ questions: [project, native] }));
    assert.equal(main.shown().text, '2 questions');
    await main.press();
    assert.deepEqual([main.calls.mounted, revealed], [[project], []], 'the Project row mounts its feed copy');
    const copy = find(main.feed, node => node.dataset.quizId === 'project');
    assert.equal(main.feed.children.indexOf(copy.parentNode), 1, 'at its own date');
    assert.deepEqual([copy.parentNode.getBoundingClientRect().top, document.activeElement], [72, copy.children[0]]);
    await main.press();
    assert.deepEqual([main.calls.mounted.length, revealed], [1, [['main-task', 'q-native', { focus: 'question' }]]],
        'Main\'s own question opens through its canonical read, focusing the question rather than its answer field');
    const own = find(main.feed, node => node.dataset.quizId === 'q-native');
    assert.deepEqual([own.parentNode.getBoundingClientRect().top, document.activeElement], [72, own.children[0]]);
    await main.press();
    assert.deepEqual([main.calls.mounted.length, revealed.length], [1, 1], 'a copy already in the feed is only scrolled to');
    assert.equal(copy.parentNode.getBoundingClientRect().top, 72);
});

test('the arrow points where the next target stands, or where it will mount by its time', t => {
    const main = mainPage(t);
    for (const ts of ['2020-01-01T08:00:00Z', '2020-01-01T10:00:00Z', '2020-01-01T11:00:00Z']) main.filler(ts);
    const mounted = question('mounted');
    main.card(mounted);  // 09:00, between the first two messages
    const arrow = (rows, top) => { main.scrollTo(top); main.waiting.snapshot(census({ questions: rows })); return main.shown().down; };
    assert.equal(arrow([mounted], 2 * ROW - 72), false, 'a loaded card above points up');
    assert.equal(arrow([mounted], 0), true, 'a loaded card below points down');
    // After landing on the oldest question at the top, a slightly newer one not loaded yet will
    // mount just below it: before the first later message, which is below the reading area.
    assert.equal(arrow([question('newer', { ts: '2020-01-01T09:30:00Z' })], ROW - 72), true, 'an unloaded question below points down');
    assert.equal(arrow([question('older', { ts: '2019-12-31T00:00:00Z' })], 2 * ROW - 72), false, 'an unloaded question above points up');
    assert.equal(arrow([question('newest', { ts: '2020-01-02T00:00:00Z' })], 0), true,
        'one newer than everything loaded will mount at the end of the feed');
});

test('the pill hides while every target is on screen, and the feed\'s own changes recount it', t => {
    const main = mainPage(t);
    main.filler('2020-01-01T08:00:00Z');
    main.card(question('seen'));
    main.feed.scrollTop = ROW - 200;
    main.waiting.snapshot(census({ questions: [question('seen')] }));
    assert.equal(main.shown(), null, 'the one question is on screen');
    main.scrollTo(0);
    assert.deepEqual(main.shown(), { text: '1 question', label: '1 unanswered: show the next question', down: true });
    main.scrollTo(ROW - 200);
    assert.equal(main.shown(), null);
    // A copy the feed mounts on screen (chat_decision's onQuestionsChanged) hides the pill at the next recount.
    main.waiting.snapshot(census({ questions: [question('seen'), question('arriving', { ts: '2020-01-01T09:30:00Z' })] }));
    assert.equal(main.shown().text, '2 questions');
    const arriving = main.card(question('arriving', { ts: '2020-01-01T09:30:00Z' }));
    // Two short copies share the reading area (60..560).
    main.feed.children[1].getBoundingClientRect = () => box(100, 200);
    arriving.getBoundingClientRect = () => box(320, 200);
    main.waiting.refresh();
    assert.equal(main.shown(), null);
});

// A task that cannot continue without the owner is one System message whose chip is the owner's move (status marks:
// the amber body and the amber dot), whatever the move is: Resume after the spending limit, an interrupted Resume the
// census still names, or model access only the owner can restore. Resume stays the crimson primary, Stop… neutral;
// a closed socket proves nothing, so the chip says Activity unconfirmed with its ring and no amber.
test('an owner-action message says its task\'s state as the owner\'s move, beside Resume and a neutral Stop…', t => {
    const main = mainPage(t);
    const chipOf = id => find(main.actionNode(id), node => node.classList?.contains('chat-live-phase'));
    const look = id => {
        const chip = chipOf(id), mark = chip.firstElementChild;
        return [chip.textContent, chip.dataset.chipTone, mark.hidden ? 'none' : `${mark.dataset.shape}/${mark.dataset.hue}`];
    };
    const wait = { wait_id: 'w', revision: 1, task_attempt: 1, state: 'waiting', reason: 'auth', role: 'main',
        model: 'codex::gpt', requires_owner_action: true };
    main.waiting.snapshot(census({ actions: [action('limit'),
        action('resumed', { pause_cause: 'owner', attention_reason: 'resume' }),
        action('access', { phase: 'working', pause_cause: undefined, can_resume: false, attention_reason: 'model_access',
            model_waits: { w: wait } })] }));
    // The chip says its word; why it waits is the message's own sentence.
    assert.deepEqual(look('limit'), ['Paused', 'owner', 'dot/owner']);
    assert.deepEqual(look('resumed'), ['Paused', 'owner', 'dot/owner']);
    assert.deepEqual(look('access'), ['Waiting for access', 'owner', 'dot/owner']);
    const resume = find(main.actionNode('limit'), node => node.dataset?.resumeRun === '1');
    // Both keep their look and carry the one touch hit area (ui.css .ui-hit-area), like a task card's.
    assert.equal(resume.className, 'btn btn-sm btn-primary ui-hit-area');
    const stop = find(main.actionNode('limit'), node => node.tagName === 'BUTTON' && node.textContent === 'Stop…');
    assert.equal(stop.className, 'btn btn-xs btn-default ui-hit-area');
    main.waiting.setConnected(false);
    for (const id of ['limit', 'resumed', 'access']) assert.deepEqual(look(id), ['Activity unconfirmed', 'neutral', 'ring/ink'], id);
});
