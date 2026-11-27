// A park past its hard deadline (docs/DESIGN.md "Room work status"): `Deadline expired` is the status, `Close` a
// separate offer for one exact task, and a Stop the server still grants stays. These pin each surface over the
// server's own shapes: the word and its line, a room row's one Close, Activity's late row, Panic's choice and Main's
// pill and cards.
import assert from 'node:assert/strict';
import test from 'node:test';
import { installDom, restoreDom } from './chat_dom_fixture.js';
import { fixture as decisionFixture } from './chat_decision_fixture.js';
import { stampNodeTimestamp } from '../modules/chat_activity.js';
import { cancelCauseClauses } from '../modules/cancel_presentation.js';
import { taskReasonDetail } from '../modules/log_events.js';
import { createMainWaiting } from '../modules/main_waiting.js';
import { createPanicRecovery } from '../modules/panic_recovery.js';
import { createRoomWork } from '../modules/room_work.js';
import { indexCloseOffers, indexDeadlineExpired, initActivity, liveActivityRowHtml } from '../modules/activity.js';
import { closeExpiredAction, closeTaskControlMenu } from '../modules/task_control_menu.js';
import { activityWord, clock, emptyWorkState, memberDeadlineExpired, memberPaused, reconcileWorkState, syncCardControls, workLine } from '../modules/work_status.js';
import { setup as setupDashboard } from './dashboard_dom_fixture.js';
import { projection, room, work } from './work_fixture.js';

const TOKEN = 'a'.repeat(64);
const OTHER = 'b'.repeat(64);
const SINCE = '2026-11-23T07:06:00Z';
const stopOf = (target, scope = 'cascade') => ({ action: 'stop', target, scope });
const closeOf = (target, token = TOKEN, scope = 'single') => ({ action: 'close', target, scope, close_expired: token });
const EXPIRED = { code: 'deadline_expired', text: 'Deadline expired' };
const PAUSED = { code: 'paused_by_owner', text: 'You paused it', since: SINCE };
const WAITING = { code: 'awaiting_selection', text: 'Waiting for Ouroboros or you to resume it' };
const all = (node, predicate) => [...(predicate(node) ? [node] : []), ...node.children.flatMap(child => all(child, predicate))];
const one = (node, predicate) => all(node, predicate)[0] || null;
const hasData = key => node => Object.hasOwn(node.dataset || {}, key);
const turn = () => new Promise(resolve => setTimeout(resolve, 0));
const settle = async () => { for (let i = 0; i < 20; i += 1) await turn(); };

test('Deadline expired is the status whether the server says it as a reason or a boolean; the line keeps the pause, never its wait', () => {
    const reasonFirst = { activity: 'paused', because: [EXPIRED, PAUSED] };
    assert.equal(activityWord(reasonFirst), 'Deadline expired');
    // The time is read in the viewer's clock, whose shape depends on the day and the viewer: today's is time-only,
    // another day's starts with the month. `clock` on the same instant is the reference; the pause's own words and
    // a non-empty time are checked apart from any fixed date or timezone.
    const pauseLine = workLine(reasonFirst);
    assert.ok(pauseLine.startsWith('You paused it · '), 'the pause keeps its author');
    const pressed = pauseLine.slice('You paused it · '.length);
    assert.ok(pressed.length > 0, 'the pause keeps its time, never empty');
    assert.equal(pressed, clock(SINCE), 'the time is the viewer\'s clock for the same instant, in whichever of its two shapes');
    // The release instruction beside Close is no longer true.
    assert.equal(workLine({ activity: 'paused', because: [EXPIRED, WAITING] }), '');
    // A status without an offer, in any order or as the boolean alone, reads the same.
    const reasonLast = { activity: 'paused', because: [PAUSED, EXPIRED] };
    assert.equal(activityWord(reasonLast), 'Deadline expired');
    assert.match(workLine(reasonLast), /^You paused it/);
    assert.equal(activityWord({ activity: 'paused', deadline_expired: true, because: [WAITING] }), 'Deadline expired');
    assert.equal(workLine({ activity: 'paused', deadline_expired: true, because: [WAITING] }), '');
    // Without the status nothing changes: the held word keeps its own line.
    assert.equal(workLine({ activity: 'paused', because: [WAITING] }), 'Waiting for Ouroboros or you to resume it');
});

test('a room row carries Close only for its own episode: a helper\'s Close never stands beside another name', async () => {
    const posts = [];
    const { prior, mount } = installDom(async (url, init) => {
        if (init?.method === 'POST') posts.push([String(url), init.body ? JSON.parse(init.body) : null]);
        return { ok: true, json: async () => ({ ok: true }) };
    });
    document.body = mount;  // the shared toasts land here
    let shade = null;
    try {
        const node = (tag, cls, parent) => { const el = document.createElement(tag); el.className = cls; parent.appendChild(el); return el; };
        const page = node('div', 'page', mount);
        const header = node('div', 'chat-page-header', page);
        const titleRow = node('div', 'app-page-title-row', node('div', 'app-page-title-block', header));
        const group = node('div', 'app-page-actions ui-chrome-group', header);
        const badge = node('span', 'status-badge online', group);
        Object.assign(badge.dataset, { tone: 'wait', mark: 'pause', owner: '', amber: '' });
        const messages = node('div', 'chat-messages', page);
        const parked = (controls, extra = {}) => work({ activity_id: 'root', chat_id: 1, kind: 'managed_task' },
            { activity: 'paused', release: 'none', because: [EXPIRED, PAUSED], controls, ...extra });
        let works = [parked([closeOf('helper-a'), closeOf('helper-b', OTHER)], { activity: 'working', because: [] })];
        const state = () => reconcileWorkState(emptyWorkState(), { supervisor_ready: true, ...projection(1, works, room('paused')) }, 1);
        shade = createRoomWork({ page, statusBadge: badge, messagesDiv: messages, ws: { isConnected: () => true },
            reading: { stick: true, cancel() {}, scroll() {} }, works: state, liveCardRecords: new Map(), pendingSubmissions: new Map(),
            revealQuestion: async () => true });
        const trigger = titleRow.children.find((child) => child.classList.contains('rw-trigger'));
        const list = header.children.find((child) => child.classList.contains('rw-shade-anchor')).children
            .find((child) => child.classList.contains('rw-surface')).querySelector('.rw-list');
        const closes = () => all(list, item => item.tagName === 'BUTTON' && item.textContent === 'Close');
        trigger.click();
        assert.equal(closes().length, 0, 'two helpers\' Close, neither named here: none is drawn under the root');
        // Its Stop comes first in the controls: the exact Close of the same row is still found and pressed.
        works = [parked([stopOf('root'), closeOf('root', TOKEN, 'cascade'), closeOf('helper-a', OTHER)])];
        shade.update();
        assert.equal(closes().length, 1, 'the root\'s own Close, once');
        assert.equal(closes()[0].offer, TOKEN);
        assert.ok(closes()[0].attrs?.['aria-describedby'] || closes()[0].getAttribute?.('aria-describedby'), 'described by the row\'s name');
        for (const fn of closes()[0].listeners.get('click') || []) await fn({ stopPropagation() {}, target: closes()[0] });
        await settle();
        assert.deepEqual(posts.map(([url, body]) => [url, body]),
            [['/api/tasks/root/cancel', { stop_policy: 'close_expired', close_expired: TOKEN }]]);
        assert.ok(closes()[0].classList.contains('rw-pill-resume'), 'the same neutral capsule as Resume');
        works = [parked([])];
        shade.update();
        assert.equal(closes().length, 0, 'a repaint without the offer removes the mounted button');
    } finally {
        shade?.destroy();
        restoreDom(prior);
    }
});

test('Activity\'s late row says Deadline expired beside the answer and offers Close with the Stop it still grants', () => {
    const now = 1_700_000_100_000;
    const late = { title: 'Weekly report', answered_at: '2026-11-23T06:00:00Z' };
    const expiredWork = work({ activity_id: 'late-root', chat_id: 1, kind: 'managed_task' }, {
        activity: 'paused', release: 'none', because: [EXPIRED, PAUSED, WAITING],
        controls: [closeOf('late-root', TOKEN, 'cascade'), { action: 'stop', target: 'late-root', scope: 'cascade' }] });
    const html = liveActivityRowHtml({ activity_id: 'late-root', kind: 'direct_chat', phase: 'budget_paused', started_at: 0,
        late_work: late, close_expired: closeOf('late-root', TOKEN, 'cascade') }, now, null, false, expiredWork);
    assert.match(html, /Deadline expired · Remaining work paused · answer delivered/);
    assert.match(html, /You paused it/, 'the dated pause stays');
    assert.doesNotMatch(html, /Waiting for Ouroboros or you to resume/);
    assert.equal((html.match(/data-act="close-expired"/g) || []).length, 1);
    assert.match(html, /data-act="task-control"/, 'the Stop the work still grants stays beside Close');
    // A census-only row with Close and no Stop of its own keeps just Close; with no expiry nothing changes.
    const bare = liveActivityRowHtml({ activity_id: 'late-root', kind: 'direct_chat', phase: 'budget_paused', started_at: 0,
        late_work: late, close_expired: closeOf('late-root', TOKEN, 'cascade'), cancelable: false }, now);
    assert.match(bare, /Deadline expired/);
    assert.doesNotMatch(bare, /data-act="task-control"/);
    const plain = liveActivityRowHtml({ activity_id: 'late-root', kind: 'direct_chat', phase: 'budget_paused', started_at: 0,
        late_work: late }, now);
    assert.doesNotMatch(plain, /Deadline expired|close-expired/);
    // No work at all: the census row's own fact says the word, and a legacy action set is the granted Stop alone.
    const census = liveActivityRowHtml({ activity_id: 'census-root', kind: 'managed_task', phase: 'budget_paused', started_at: 0,
        deadline_expired: true, cancelable: true }, now, null, true);
    assert.match(census, /<span data-i18n-skip>Deadline expired/);
    assert.match(census, /data-act="task-control"[^>]*data-budget-paused="1"/, 'a parked set: Stop only, never Hurry or Pause');
    assert.doesNotMatch(census, /data-resume/, 'no dead Resume, even where the caller offered one');
    const ungranted = liveActivityRowHtml({ activity_id: 'census-root', kind: 'managed_task', phase: 'budget_paused', started_at: 0,
        deadline_expired: true, cancelable: false }, now, null, true);
    assert.doesNotMatch(ungranted, /data-act="task-control"/, 'no Stop is inferred from the expiry');
});

// ---- Panic's choice
const ROWS = {
    plain: { task_id: 'task-plain', task_name: 'Weekly digest', panic_hold_id: 'hold-plain', chat_id: 1 },
    late: { task_id: 'task-late', task_name: 'Linked helper', panic_hold_id: 'hold-late', chat_id: 1, close_expired: closeOf('task-late') },
    old: { task_id: 'task-old', task_name: 'Older helper', panic_hold_id: 'hold-old', chat_id: 1, close_expired: closeOf('task-old', OTHER) },
};
const card = (id, patch = {}) => ({ task_id: id, activity_id: id, phase: 'budget_paused', can_resume: true,
    cancelable: true, attention_reason: 'resume', model_waits: {}, ...patch });
const body = (rows, cards = rows.map(row => card(row.task_id)), extra = {}) => ({ panic_recovery: rows,
    panic_recovery_complete: true, main_attention: cards, main_attention_complete: true, projects: [], ...extra });

function panicRig(t) {
    const env = installDom();
    document.body = env.mount;
    t.after(() => restoreDom(env.prior));
    const feed = document.createElement('div');
    env.mount.append(feed);
    const resumes = [];
    const panic = createPanicRecovery({
        insertMessageNode: node => { feed.append(node); return true; }, stampNodeTimestamp, refresh: async () => {},
        resume: async (taskId, body) => { resumes.push([taskId, body]); return { ok: true, task_id: taskId }; },
        stopControl: (taskId) => { const button = document.createElement('button'); button.textContent = 'Stop…'; button.dataset.stopFor = taskId; return button; },
        now: () => 1000,
    });
    t.after(() => panic.destroy());
    const message = () => one(feed, node => node.classList?.contains('panic-recovery'));
    const row = id => one(message(), node => node.classList?.contains('panic-recovery-row') && node.dataset.taskId === id);
    return { panic, resumes, message, row, part: key => one(message(), hasData(key)),
        actions: () => one(message(), node => node.classList?.contains('system-message-actions')),
        stop: id => one(row(id), hasData('stopFor')), box: id => one(row(id), node => node.tagName === 'INPUT'),
        close: id => one(row(id), node => node.tagName === 'BUTTON' && node.textContent === 'Close'),
        outcome: id => one(row(id), node => node.classList?.contains('panic-recovery-outcome')).textContent };
}

test('Panic leaves an expired row out of every choice while its granted Stop stays; all expired leaves no empty actions row', (t) => {
    const rig = panicRig(t);
    rig.panic.snapshot(body([ROWS.plain, ROWS.late, ROWS.old]));
    assert.equal(rig.box('task-late').hidden, true);
    assert.equal(rig.box('task-plain').hidden, false);
    assert.equal(rig.outcome('task-late'), 'Deadline expired');
    assert.ok(rig.close('task-late') && rig.close('task-old'), 'one Close in each expired row, inside that row');
    assert.equal(rig.stop('task-late').hidden, false, 'the normal Stop it is granted stays');
    assert.equal(rig.part('panicAll').hidden, true, 'one choosable row: no Choose all');
    assert.equal(rig.part('panicResume').hidden, false);
    assert.equal(rig.actions().hidden, false);
    // Close can be offered while the card no longer grants Stop: then only Close stands.
    rig.panic.snapshot(body([ROWS.plain, ROWS.late, ROWS.old],
        [card('task-plain'), card('task-late', { cancelable: false }), card('task-old', { cancelable: false })]));
    assert.equal(rig.stop('task-late').hidden, true);
    // Every row expired: nothing to resume, so the footer takes no room and the pill does not count the choice.
    rig.panic.snapshot(body([ROWS.late, ROWS.old]));
    assert.equal(rig.part('panicResume').hidden, true);
    assert.equal(rig.actions().hidden, true);
    assert.equal(rig.panic.target(), null);
    // The status can arrive without an offer (the card's own reason): still a status line with no box and no Resume.
    rig.panic.snapshot(body([ROWS.plain, { ...ROWS.plain, task_id: 'task-bare', panic_hold_id: 'hold-bare' }],
        [card('task-plain'), card('task-bare', { can_resume: false, attention_reason: 'deadline_expired' })]));
    assert.equal(rig.box('task-bare').hidden, true);
    assert.equal(rig.close('task-bare'), null, 'no offer, no Close');
    assert.equal(rig.stop('task-bare').hidden, false);
    assert.equal(rig.outcome('task-bare'), 'Deadline expired');
});

// ---- Main
function mainRig(t) {
    const resumes = [];
    const env = installDom(async (url, init) => {
        const match = String(url).match(/^\/api\/tasks\/([^/]+)\/resume$/);
        if (match && init?.method === 'POST') {
            resumes.push([decodeURIComponent(match[1]), init.body ? JSON.parse(init.body) : null]);
            return { ok: true, json: async () => ({ ok: true, task_id: decodeURIComponent(match[1]) }) };
        }
        return { ok: true, json: async () => ({}) };
    });
    document.body = env.mount;
    const priorCss = globalThis.CSS;
    globalThis.CSS = { escape: value => String(value) };
    t.after(() => { restoreDom(env.prior); globalThis.CSS = priorCss; });
    const box = (top, height) => ({ top, bottom: top + height, height, left: 0, right: 600, width: 600 });
    const page = document.createElement('div');
    const header = document.createElement('div');
    header.className = 'chat-page-header';
    const feed = document.createElement('div');
    const toolbar = document.createElement('div');
    toolbar.className = 'chat-toolbar-row';
    page.append(header, feed, toolbar);
    env.mount.append(page);
    header.getBoundingClientRect = () => box(0, 60);
    feed.getBoundingClientRect = () => box(0, 600);
    toolbar.getBoundingClientRect = () => box(560, 40);
    Object.defineProperty(feed, 'scrollHeight', { configurable: true, get: () => feed.children.length * 1000, set() {} });
    const insert = node => { feed.append(node); node.getBoundingClientRect = () => box(feed.children.indexOf(node) * 1000 - feed.scrollTop, 200); return true; };
    const waiting = createMainWaiting({ page, messagesDiv: feed, toolbar, insertMessageNode: insert, stampNodeTimestamp,
        decision: () => ({ syncCurrentQuestions() {}, setConnected() {}, observedQuestion: () => null }),
        reading: () => ({ claim() {}, stick: true }) });
    t.after(() => waiting.destroy());
    const pill = toolbar.children.at(-1);
    const named = id => one(feed, node => node.dataset?.taskId === id && node.classList?.contains('owner-action-body'));
    const button = (id, text) => one(named(id), node => node.tagName === 'BUTTON' && node.textContent === text);
    return { waiting, feed, pill, named, button, resumes };
}
const attention = (id, patch = {}) => ({ task_id: id, activity_id: id, task_name: `Task ${id}`, phase: 'budget_paused',
    pause_cause: 'budget', can_resume: true, cancelable: true, attention_reason: 'budget', model_waits: {}, ...patch });
const mainBody = (rows) => ({ current_questions: [], current_questions_complete: true, main_attention: rows,
    main_attention_complete: true, panic_recovery: [], panic_recovery_complete: true });

test('Main: a card that crosses its deadline drops its Resume and an ungranted Stop, and a Close-only card is never "waiting for you"', (t) => {
    const rig = mainRig(t);
    rig.waiting.snapshot(mainBody([attention('t-cross'), attention('t-need', { task_name: 'Needs you' })]));
    rig.waiting.refresh();
    assert.ok(rig.button('t-cross', 'Stop…') && rig.button('t-cross', 'Resume'));
    assert.equal(rig.pill.querySelector('.chat-question-nav-count').textContent, '2');
    // The same task a poll later: past its deadline, Close offered, Stop no longer granted.
    rig.waiting.snapshot(mainBody([
        attention('t-cross', { can_resume: false, cancelable: false, attention_reason: 'deadline_expired', close_expired: closeOf('t-cross', TOKEN, 'cascade') }),
        attention('t-need', { task_name: 'Needs you' })]));
    rig.waiting.refresh();
    assert.equal(rig.button('t-cross', 'Stop…'), null, 'no greyed Stop left behind');
    assert.equal(rig.button('t-cross', 'Resume'), null, 'no dead Resume');
    assert.ok(rig.button('t-cross', 'Close'));
    assert.equal(rig.pill.querySelector('.chat-question-nav-count').textContent, '1', 'the optional Close is not an action waiting for the owner');
    // Only the Close-only card remains: nothing waits, so the pill goes.
    rig.waiting.snapshot(mainBody([attention('t-cross', { can_resume: false, cancelable: false, attention_reason: 'deadline_expired', close_expired: closeOf('t-cross', TOKEN, 'cascade') })]));
    rig.waiting.refresh();
    assert.equal(rig.pill.hidden, true);
    // A Stop the server still grants stays on an expired card.
    rig.waiting.snapshot(mainBody([attention('t-cross', { can_resume: false, cancelable: true, attention_reason: 'deadline_expired', close_expired: closeOf('t-cross', TOKEN, 'cascade') })]));
    assert.ok(rig.button('t-cross', 'Stop…') && rig.button('t-cross', 'Close'));
    assert.equal(rig.button('t-cross', 'Resume'), null);
});

test('Panic reads the host\'s own expiry fact: no Close, no card, still no box, no Resume and no count', (t) => {
    const rig = panicRig(t);
    const plain = ROWS.plain;
    const bare = (id, patch = {}) => ({ task_id: id, task_name: `Task ${id}`, panic_hold_id: `hold-${id}`, chat_id: 1, ...patch });
    const member = { confirmed: true, activity: 'paused', transition: 'none', call: '', release: 'none', turn: [], tails: [],
        because: [EXPIRED, PAUSED], mark: 'pause' };
    const rows = [plain, bare('t-own', { deadline_expired: true }), bare('t-act'), bare('t-mem'), bare('t-card')];
    // Only the first row has a card (the host omits a held task offering nothing): the others' expiry rides the row's
    // own boolean, the same body's exact activity, or its member state, never a Close offer or a card.
    rig.panic.snapshot(body(rows, [card('task-plain')], {
        active_chat_activities: [{ activity_id: 't-act', deadline_expired: true }, { activity_id: 'other', deadline_expired: true }],
        works: [{ work_id: 'w', members: { ids: ['t-mem'], states: { 't-mem': member } } }] }));
    for (const id of ['t-own', 't-act', 't-mem']) {
        assert.equal(rig.box(id).hidden, true, `${id}: no box`);
        assert.equal(rig.outcome(id), 'Deadline expired', id);
        assert.equal(rig.close(id), null, `${id}: no token, no Close`);
        assert.equal(rig.stop(id), null, `${id}: no card grants a Stop, and none is inferred from the expiry`);
    }
    assert.equal(rig.box('t-card').hidden, false, 'an exact identity: another row\'s expiry reaches no other');
    assert.equal(rig.box('task-plain').hidden, false);
    assert.equal(rig.part('panicAll').textContent, 'Choose all 2', 'only the two rows that still wait for Resume');
    // All rows expired by facts alone: nothing to resume, no footer, and the pill has nothing to count.
    rig.panic.snapshot(body([bare('t-own', { deadline_expired: true }), bare('t-act')], [], {
        active_chat_activities: [{ activity_id: 't-act', deadline_expired: true }] }));
    assert.equal(rig.actions().hidden, true);
    assert.equal(rig.panic.target(), null);
});

test('Main: a distinct sign-in request on an expired task still waits for the owner, and the pause keeps its dated author', (t) => {
    const rig = mainRig(t);
    const member = { confirmed: true, activity: 'paused', transition: 'none', call: '', release: 'none', turn: [], tails: [],
        because: [EXPIRED, PAUSED, WAITING], mark: 'pause' };
    const body = (rows) => ({ ...mainBody([attention('t-first', { task_name: 'Needs you' }), ...rows]),
        works: [{ work_id: 'w', members: { ids: ['t-line'], states: { 't-line': member } } }] });
    rig.waiting.snapshot(body([attention('t-line', { attention_reason: 'deadline_expired', can_resume: false, cancelable: false,
        close_expired: closeOf('t-line', TOKEN, 'cascade') })]));
    rig.waiting.refresh();
    const reason = one(rig.named('t-line'), node => node.classList?.contains('owner-action-reason'));
    assert.match(reason.textContent, /^You paused it/, 'the dated pause stays');
    assert.doesNotMatch(reason.textContent, /Waiting for Ouroboros or you to resume/);
    assert.ok(rig.button('t-line', 'Close').getAttribute('aria-describedby'), 'described by the task\'s name');
    assert.equal(rig.pill.hidden, true, 'the off-screen Close-only card is not counted: nothing else is away');
    rig.waiting.snapshot(body([attention('t-line', { attention_reason: 'model_access', can_resume: false, cancelable: false,
        close_expired: closeOf('t-line', TOKEN, 'cascade') })]));
    rig.waiting.refresh();
    assert.ok(rig.button('t-line', 'Close'));
    assert.equal(rig.pill.hidden, false, 'a distinct sign-in request is still the owner\'s move');
    assert.equal(rig.pill.querySelector('.chat-question-nav-count').textContent, '2');
});

test('a loaded card\'s Close is the neutral capsule described by its card title, removes ungranted legacy Stop, and keeps granted Stop', (t) => {
    const env = installDom();
    document.body = env.mount;
    t.after(() => restoreDom(env.prior));
    const makeCard = (groupId, titleText, withStop = true) => {
        const root = document.createElement('div');
        const titleEl = document.createElement('span');
        titleEl.textContent = titleText;
        const actions = document.createElement('div');
        let stopBtn = null;
        if (withStop) {
            stopBtn = document.createElement('button');
            stopBtn.dataset.cancelRun = '1';
            actions.append(stopBtn);
        }
        root.append(titleEl, actions);
        const query = (selector) => (selector === '[data-cancel-run]'
            ? one(actions, hasData('cancelRun')) : selector === '[data-live-title]' ? titleEl : null);
        root.querySelector = query;
        actions.querySelector = query;
        return { record: { root, titleEl, groupId, finished: false }, actions, titleEl };
    };
    const rootCard = makeCard('root-1', 'Release check');
    const authWork = work({ activity_id: 'root-1', chat_id: 1, kind: 'managed_task' },
        { activity: 'paused', because: [EXPIRED, PAUSED], controls: [closeOf('root-1', TOKEN, 'cascade')] });
    // Authoritative work offering only Close drops a previously mounted legacy Stop… even if legacy=true.
    let offer = { work: authWork, close: TOKEN, resume: false, menu: false, legacy: true };
    const sync = (card, read) => syncCardControls(card.record, { read, actionsOf: () => card.actions, stop() {}, afterAction() {} });
    sync(rootCard, () => offer);
    const close = one(rootCard.actions, hasData('closeExpiredRun'));
    assert.ok(close && close.classList.contains('btn-default') && !close.classList.contains('btn-primary'), 'neutral, never urgent');
    assert.equal(close.offer, TOKEN, 'the offer it stands for');
    assert.equal(one(rootCard.actions, hasData('cancelRun')), null, 'no dummy legacy Stop when authoritative work omits it');
    assert.ok(rootCard.titleEl.id, 'card title gets an instance-unique id');
    assert.equal(close.getAttribute('aria-describedby'), rootCard.titleEl.id, 'Close is described by its own card title');
    // When the authoritative work DOES grant Stop (menu=true), Stop… returns beside Close.
    const authWithStop = work({ activity_id: 'root-1', chat_id: 1, kind: 'managed_task' },
        { activity: 'paused', because: [EXPIRED, PAUSED], controls: [closeOf('root-1', TOKEN, 'cascade'), stopOf('root-1')] });
    offer = { work: authWithStop, close: TOKEN, resume: false, menu: true, legacy: false };
    sync(rootCard, () => offer);
    assert.ok(one(rootCard.actions, hasData('closeExpiredRun')) && one(rootCard.actions, hasData('cancelRun')), 'granted Stop stays beside Close');
    // Two expired helper cards get distinct instance-unique title ids wired to their Close buttons.
    const helperA = makeCard('helper-1', 'Verifier (helper-1)', false);
    const helperB = makeCard('helper-2', 'Verifier (helper-2)', false);
    sync(helperA, () => ({ work: null, close: TOKEN, resume: false, menu: false, legacy: false }));
    sync(helperB, () => ({ work: null, close: OTHER, resume: false, menu: false, legacy: false }));
    const closeA = one(helperA.actions, hasData('closeExpiredRun'));
    const closeB = one(helperB.actions, hasData('closeExpiredRun'));
    assert.ok(helperA.titleEl.id && helperB.titleEl.id && helperA.titleEl.id !== helperB.titleEl.id, 'distinct title ids');
    assert.equal(closeA.getAttribute('aria-describedby'), helperA.titleEl.id);
    assert.equal(closeB.getAttribute('aria-describedby'), helperB.titleEl.id);
    // A repaint without the offer removes the mounted Close button.
    offer = { work: null, close: '', resume: false, menu: false, legacy: false };
    sync(rootCard, () => offer);
    assert.equal(one(rootCard.actions, hasData('closeExpiredRun')), null, 'a repaint without the offer removes the mounted button');
});

test('Panic on a partial census applies authoritative expiry from the same response without touching neighbors', async (t) => {
    const rig = panicRig(t);
    const rowA = { task_id: 'task-a', task_name: 'Alpha check', panic_hold_id: 'hold-a', chat_id: 1 };
    const rowB = { task_id: 'task-b', task_name: 'Beta check', panic_hold_id: 'hold-b', chat_id: 1 };
    const rowC = { task_id: 'task-c', task_name: 'Gamma check', panic_hold_id: 'hold-c', chat_id: 1 };
    rig.panic.snapshot(body([rowA, rowB, rowC]));
    // Owner selects Alpha and Beta.
    for (const id of ['task-a', 'task-b']) {
        const box = rig.box(id);
        box.checked = true;
        for (const fn of box.listeners.get('change') || []) fn({ target: box });
    }
    assert.equal(rig.part('panicResume').textContent, 'Resume 2 selected');
    // Partial census omits Alpha and Gamma, while main_attention in the same response positively marks Alpha expired.
    const partialBody = {
        panic_recovery: [rowB],
        panic_recovery_complete: false,
        main_attention: [
            card('task-b'),
            card('task-a', { can_resume: false, cancelable: false, attention_reason: 'deadline_expired',
                deadline_expired: true, close_expired: closeOf('task-a', TOKEN, 'cascade') }),
        ],
        main_attention_complete: false,
        projects: [],
    };
    rig.panic.snapshot(partialBody);
    assert.equal(rig.box('task-a').hidden, true, 'expired unseen row loses its checkbox immediately');
    assert.equal(rig.outcome('task-a'), 'Deadline expired');
    assert.equal(rig.stop('task-a').hidden, true, 'old Stop is withdrawn when its attention card drops cancelable');
    assert.equal(rig.box('task-b').checked, true, 'listed neighbor keeps its selection');
    assert.equal(rig.box('task-c').hidden, false, 'unseen non-expired neighbor stays retained');
    assert.equal(rig.box('task-c').disabled, true, 'unseen neighbor stays unconfirmed/disabled');
    assert.equal(rig.part('panicResume').textContent, 'Resume 1 selected', 'counts exclude the expired row');
    assert.equal(one(rig.message(), node => node.classList?.contains('panic-recovery-count')).textContent, '1 of 1 chosen');
    // Pressing Resume sends ONLY task-b; the now-expired task-a never enters Resume.
    rig.part('panicResume').click();
    await settle();
    assert.deepEqual(rig.resumes, [['task-b', { expected_panic_hold_id: 'hold-b' }]]);

    // End-to-end Main integration: omitted expired task-a mounts its standalone Close card while Resume sends only task-b.
    const main = mainRig(t);
    main.waiting.snapshot({ current_questions: [], current_questions_complete: true, ...body([rowA, rowB, rowC]) });
    const panicMsg = () => one(main.feed, node => node.classList?.contains('panic-recovery'));
    const panicRow = (id) => one(panicMsg(), node => node.classList?.contains('panic-recovery-row') && node.dataset.taskId === id);
    for (const id of ['task-a', 'task-b']) {
        const box = one(panicRow(id), node => node.tagName === 'INPUT');
        box.checked = true;
        for (const fn of box.listeners.get('change') || []) fn({ target: box });
    }
    main.waiting.snapshot({ current_questions: [], current_questions_complete: true, ...partialBody });
    assert.equal(one(panicRow('task-a'), node => node.tagName === 'INPUT').hidden, true);
    assert.equal(one(panicRow('task-b'), node => node.tagName === 'INPUT').checked, true);
    assert.equal(one(panicRow('task-c'), node => node.tagName === 'INPUT').disabled, true);
    assert.ok(main.button('task-a', 'Close'), 'Main mounts standalone Close card for omitted expired task-a');
    one(panicMsg(), hasData('panicResume')).click();
    await settle();
    assert.deepEqual(main.resumes, [['task-b', { expected_panic_hold_id: 'hold-b' }]]);

    // If Alpha was the only Panic row and becomes unseen+expired on a partial read, the pill drops it immediately.
    const sole = panicRig(t);
    sole.panic.snapshot(body([rowA]));
    assert.ok(sole.panic.target(), 'initially waiting on the owner');
    sole.panic.snapshot({
        panic_recovery: [],
        panic_recovery_complete: false,
        main_attention: [card('task-a', { can_resume: false, cancelable: false, attention_reason: 'deadline_expired', deadline_expired: true })],
        main_attention_complete: false,
        projects: [],
    });
    assert.equal(sole.box('task-a').hidden, true);
    assert.equal(sole.part('panicResume').hidden, true);
    assert.equal(sole.actions().hidden, true);
    assert.equal(sole.panic.target(), null, 'waiting pill drops the expired-only partial list');
});

test('Main expired card renders a dated Resume refusal once and preserves pause attribution', (t) => {
    const rig = mainRig(t);
    const attempt = { at: '2026-11-23T08:00:00Z', by: { actor: 'owner' }, error: 'deadline_passed', text: 'Deadline passed.' };
    const member = { confirmed: true, activity: 'paused', transition: 'none', call: '', release: 'none', turn: [], tails: [],
        because: [EXPIRED, PAUSED, WAITING], last_resume: attempt, mark: 'pause' };
    rig.waiting.snapshot({
        ...mainBody([attention('t-ref', { attention_reason: 'deadline_expired', can_resume: false, cancelable: false,
            close_expired: closeOf('t-ref', TOKEN, 'cascade') })]),
        works: [{ work_id: 'w', members: { ids: ['t-ref'], states: { 't-ref': member } } }],
    });
    const reason = one(rig.named('t-ref'), node => node.classList?.contains('owner-action-reason')).textContent;
    assert.match(reason, /^You paused it · /, 'keeps original pause attribution');
    assert.equal((reason.match(/Resume refused at/g) || []).length, 1, `refusal must appear once, got: ${reason}`);
    assert.match(reason, /Deadline passed\.$/);
    assert.doesNotMatch(reason, /Waiting for Ouroboros or you to resume/);
    // Without pause attribution (only waiting + refusal), refusal still appears once and no Resume instruction leaks in.
    const bareMember = { ...member, because: [EXPIRED, WAITING] };
    rig.waiting.snapshot({
        ...mainBody([attention('t-ref', { attention_reason: 'resume', deadline_expired: true, can_resume: false, cancelable: false,
            close_expired: closeOf('t-ref', TOKEN, 'cascade') })]),
        works: [{ work_id: 'w', members: { ids: ['t-ref'], states: { 't-ref': bareMember } } }],
    });
    const bareReason = one(rig.named('t-ref'), node => node.classList?.contains('owner-action-reason')).textContent;
    assert.equal((bareReason.match(/Resume refused at/g) || []).length, 1, `refusal must appear once, got: ${bareReason}`);
    assert.doesNotMatch(bareReason, /Needs (?:an explicit )?Resume/);
});

test('Activity grants expired helper Stop from exact child queue row + confirmed paused member with equal attempt, never from root Stop', async (t) => {
    const response = (data, status = 200) => ({ ok: status < 400, status, json: async () => data });
    let activity = null;
    t.after(() => activity?.destroy());
    const { mount, routes, ws, document } = setupDashboard(t);
    const expiredMember = (extra = {}) => ({
        confirmed: true, activity: 'paused', transition: 'none', call: '', release: 'none',
        turn: [], tails: [], because: [EXPIRED, PAUSED], queue_attempt: 1, mark: 'pause', ...extra,
    });
    const unknownMember = { confirmed: false, activity: 'unknown', transition: 'none', call: '', release: 'none',
        turn: [], tails: [], because: [], queue_attempt: 1, mark: 'ring' };
    // Owning root has NO stop grant (controls: []), yet its confirmed paused helper queue row keeps Stop.
    const noStopWork = work({ activity_id: 'root-no-stop', chat_id: 1, kind: 'managed_task' }, {
        activity: 'paused', controls: [],
        members: {
            ids: ['root-no-stop', 'helper-stop'],
            states: {
                'root-no-stop': expiredMember(),
                'helper-stop': expiredMember({ queue_attempt: 1 }),
            },
        },
    });
    // Owning root DOES have Stop, yet negative neighbors with unknown/mismatched/missing attempt or non-child lineage get NO Stop.
    const stoppableWork = work({ activity_id: 'root-with-stop', chat_id: 1, kind: 'managed_task' }, {
        activity: 'working', controls: [stopOf('root-with-stop')],
        members: {
            ids: ['root-with-stop', 'helper-unknown', 'helper-census-behind', 'helper-queue-behind',
                'helper-no-queue-attempt', 'helper-no-member-attempt', 'generic-non-root'],
            states: {
                'root-with-stop': { confirmed: true, activity: 'working', transition: 'none', call: 'working', live: true,
                    release: 'self', turn: [], tails: [], because: [], queue_attempt: 1, mark: 'moving' },
                'helper-unknown': unknownMember,
                'helper-census-behind': expiredMember({ queue_attempt: 1 }),
                'helper-queue-behind': expiredMember({ queue_attempt: 2 }),
                'helper-no-queue-attempt': expiredMember({ queue_attempt: 1 }),
                'helper-no-member-attempt': expiredMember({ queue_attempt: undefined }),
                'generic-non-root': expiredMember({ queue_attempt: 1 }),
            },
        },
    });
    const helperRow = (id, rootId, name, attempt, taskExtra = {}) => ({
        id, type: 'task', ...(attempt === undefined ? {} : { attempt }),
        task: { id, root_task_id: rootId, parent_task_id: rootId, delegation_role: 'subagent', suggested_name: name, ...taskExtra },
    });
    const cancels = [];
    routes.set('/api/tasks?queue_only=1', response({
        queue: {
            running: [{ id: 'root-with-stop', type: 'task', attempt: 1, runtime_sec: 15,
                task: { id: 'root-with-stop', root_task_id: 'root-with-stop', suggested_name: 'Release coordinator' } }],
            pending: [
                helperRow('helper-stop', 'root-no-stop', 'Windows window check', 1),
                helperRow('helper-unknown', 'root-with-stop', 'Unconfirmed neighbor', 1),
                helperRow('helper-census-behind', 'root-with-stop', 'Census behind helper', 2),
                helperRow('helper-queue-behind', 'root-with-stop', 'Queue behind helper', 1),
                helperRow('helper-no-queue-attempt', 'root-with-stop', 'Missing queue attempt helper', undefined),
                helperRow('helper-no-member-attempt', 'root-with-stop', 'Missing member attempt helper', 1),
                // Non-root by root_task_id alone, without parent_task_id or delegation_role='subagent'.
                { id: 'generic-non-root', type: 'task', attempt: 1,
                    task: { id: 'generic-non-root', root_task_id: 'root-with-stop', suggested_name: 'Ambiguous non-root' } },
            ],
        },
    }));
    routes.set('/api/schedules', response({ tasks: [] }));
    routes.set('/api/services/kept', response({ services: [], complete: true }));
    routes.set('/api/state', response({
        bg_consciousness_enabled: false,
        // Public census is root-only: none of the helpers appears in active_chat_activities.
        active_chat_activities: [{ activity_id: 'root-with-stop', kind: 'managed_task', phase: 'working', cancelable: true, task_attempt: 1 }],
        active_chat_activities_complete: true,
        main_attention: [
            attention('helper-unknown', { can_resume: false, cancelable: false, attention_reason: 'deadline_expired', deadline_expired: true }),
        ],
        works: [noStopWork, stoppableWork],
        works_complete: true,
    }));
    routes.set('/api/tasks/helper-stop', response({ id: 'helper-stop', status: 'running' }));
    routes.set('/api/tasks/helper-stop/cancel', (_url, init) => {
        cancels.push(JSON.parse(init.body));
        return response({ ok: true, task_id: 'helper-stop' });
    });
    activity = initActivity({ mount, ws });
    await activity.refresh();
    const rows = mount.querySelector('[data-activity-section="queue"]').querySelectorAll('.activity-row');
    const rowByName = (label) => rows.find((row) => row.querySelector('.activity-name')?.textContent === label);

    const grantedRow = rowByName('Windows window check');
    assert.ok(grantedRow, 'expired non-close helper row rendered');
    assert.match(grantedRow.textContent, /Deadline expired/);
    assert.equal(grantedRow.querySelector('[data-act="close-expired"]'), null, 'Close is unavailable');
    const stopBtn = grantedRow.querySelector('[data-act="task-control"]');
    assert.ok(stopBtn, 'confirmed paused expired helper keeps Stop even when its owning root has no Stop');
    assert.equal(stopBtn.dataset.id, 'helper-stop');
    assert.equal(stopBtn.dataset.budgetPaused, '1');
    assert.equal(stopBtn.dataset.root, undefined, 'menu-shape wholeTree flag stays false for helper');
    assert.equal(stopBtn.dataset.resume, undefined, 'no dead Resume');
    assert.equal(stopBtn.textContent, 'Stop now');

    // Healthy root keeps its normal Stop control unchanged.
    const rootRow = rowByName('Release coordinator');
    assert.ok(rootRow?.querySelector('[data-act="task-control"][data-id="root-with-stop"][data-root="1"]'));

    for (const label of [
        'Unconfirmed neighbor', 'Census behind helper', 'Queue behind helper',
        'Missing queue attempt helper', 'Missing member attempt helper', 'Ambiguous non-root',
    ]) {
        const deniedRow = rowByName(label);
        assert.ok(deniedRow, `${label} rendered`);
        assert.equal(deniedRow.querySelector('[data-act="task-control"]'), null, `${label} must not get Stop`);
        assert.equal(deniedRow.querySelector('[data-act="close-expired"]'), null, `${label} has no Close`);
    }

    // Confirming Stop sends the exact ordinary immediate-Stop cancel endpoint/body for helper-stop only
    // (cancelTask omits stop_policy for 'immediate' -> { cascade: true, stop_action_id }, never close_expired).
    const clicking = mount.fire('click', { target: stopBtn });
    await settle();
    const dialog = document.body.querySelector('[role="dialog"]');
    assert.ok(dialog, 'confirmation dialog opens for direct Stop now');
    await dialog.parentElement.fire('click', { target: dialog.querySelector('[data-confirm-ok]') });
    await clicking;
    await settle();
    assert.equal(cancels.length, 1);
    assert.equal(cancels[0].cascade, true);
    assert.equal(cancels[0].stop_policy, undefined, 'api_client.cancelTask omits stop_policy for immediate Stop');
    assert.match(cancels[0].stop_action_id, /^stop-/);
    assert.deepEqual(Object.keys(cancels[0]).sort(), ['cascade', 'stop_action_id']);
});

test('Activity rechecks expired helper Stop on open/confirm against latest accepted queue, census and connected state', async (t) => {
    const response = (data, status = 200) => ({ ok: status < 400, status, json: async () => data });
    let activity = null;
    t.after(() => activity?.destroy());
    const { mount, routes, ws, document } = setupDashboard(t);
    let connected = true;
    ws.isConnected = () => connected;
    const mkState = (attempt, confirmed = true) => ({
        bg_consciousness_enabled: false,
        active_chat_activities: [],
        active_chat_activities_complete: true,
        works: [work({ activity_id: 'root-no-stop', chat_id: 1, kind: 'managed_task' }, {
            activity: 'paused', controls: [],
            members: {
                ids: ['root-no-stop', 'helper-stop'],
                states: {
                    'helper-stop': { confirmed, activity: confirmed ? 'paused' : 'unknown', transition: 'none', call: '',
                        release: 'none', turn: [], tails: [], because: confirmed ? [EXPIRED, PAUSED] : [],
                        queue_attempt: attempt, mark: confirmed ? 'pause' : 'ring' },
                },
            },
        })],
        works_complete: true,
    });
    const mkQueue = (attempt, present = true) => ({
        queue: {
            running: [],
            pending: present ? [{ id: 'helper-stop', type: 'task', attempt,
                task: { id: 'helper-stop', root_task_id: 'root-no-stop', parent_task_id: 'root-no-stop',
                    delegation_role: 'subagent', suggested_name: 'Windows window check' } }] : [],
        },
    });
    let queueBody = mkQueue(1, true);
    let stateBody = mkState(1, true);
    const cancels = [];
    routes.set('/api/tasks?queue_only=1', () => response(queueBody));
    routes.set('/api/state', () => response(stateBody));
    routes.set('/api/schedules', response({ tasks: [] }));
    routes.set('/api/services/kept', response({ services: [], complete: true }));
    routes.set('/api/tasks/helper-stop', response({ id: 'helper-stop', status: 'running' }));
    routes.set('/api/tasks/helper-stop/cancel', (_url, init) => {
        cancels.push(JSON.parse(init.body));
        return response({ ok: true, task_id: 'helper-stop' });
    });

    activity = initActivity({ mount, ws });
    await activity.refresh();
    const stopBtn = mount.querySelector('[data-act="task-control"][data-id="helper-stop"]');
    assert.ok(stopBtn, 'initially offers Stop now on confirmed attempt 1');

    // Disconnected socket blocks opening the dialog.
    connected = false;
    await mount.fire('click', { target: stopBtn });
    await settle();
    assert.equal(document.body.querySelector('[role="dialog"]'), null, 'disconnected socket refuses open');

    // Reconnected: open dialog, then crossed repaint replaces attempt 1 -> 2 before user confirms.
    connected = true;
    const clicking = mount.fire('click', { target: stopBtn });
    await settle();
    const dialog = document.body.querySelector('[role="dialog"]');
    assert.ok(dialog, 'dialog opens while connected and confirmed');
    queueBody = mkQueue(2, true);
    stateBody = mkState(1, true); // crossed repaint: queue moved to attempt 2 while census still at 1
    await activity.refresh();
    assert.equal(mount.querySelector('[data-act="task-control"][data-id="helper-stop"]'), null,
        'crossed repaint drops Stop now on mismatched attempt');
    await dialog.parentElement.fire('click', { target: dialog.querySelector('[data-confirm-ok]') });
    await clicking;
    await settle();
    assert.equal(cancels.length, 0, 'confirmed action recheck blocks submission after attempt changed');

    // Restore attempt 1, open dialog, then row disappears before confirm.
    queueBody = mkQueue(1, true);
    stateBody = mkState(1, true);
    await activity.refresh();
    const stopBtn2 = mount.querySelector('[data-act="task-control"][data-id="helper-stop"]');
    assert.ok(stopBtn2);
    const clicking2 = mount.fire('click', { target: stopBtn2 });
    await settle();
    const dialog2 = document.body.querySelector('[role="dialog"]');
    assert.ok(dialog2);
    queueBody = mkQueue(1, false);
    await activity.refresh();
    await dialog2.parentElement.fire('click', { target: dialog2.querySelector('[data-confirm-ok]') });
    await clicking2;
    await settle();
    assert.equal(cancels.length, 0, 'confirmed action recheck blocks submission after row disappeared');
});

test('Activity refresh where both queue_only and state reads fail expires helper Stop recheck while keeping known root exact controls', async (t) => {
    const response = (data, status = 200) => ({ ok: status < 400, status, json: async () => data });
    const prevIO = globalThis.IntersectionObserver;
    let activity = null;
    t.after(() => closeTaskControlMenu());
    t.after(() => activity?.destroy());
    t.after(() => { globalThis.IntersectionObserver = prevIO; });
    const { mount, routes, ws, document, window } = setupDashboard(t);
    const rect = () => ({ top: 40, left: 40, right: 160, bottom: 70, width: 120, height: 30 });
    const origCreate = document.createElement;
    window.innerWidth = 1024;
    window.innerHeight = 768;
    window.setTimeout = globalThis.setTimeout;
    window.clearTimeout = globalThis.clearTimeout;
    globalThis.IntersectionObserver = class { observe() {} takeRecords() { return []; } disconnect() {} };
    document.createElement = (tag) => {
        const el = origCreate(tag);
        el.getBoundingClientRect = rect;
        el.style = { setProperty() {} };
        return el;
    };
    let failReads = false;
    const cancels = [];
    const helperWork = work({ activity_id: 'root-no-stop', chat_id: 1, kind: 'managed_task' }, {
        activity: 'paused', controls: [],
        members: {
            ids: ['root-no-stop', 'helper-stop'],
            states: {
                'helper-stop': { confirmed: true, activity: 'paused', transition: 'none', call: '',
                    release: 'none', turn: [], tails: [], because: [EXPIRED, PAUSED],
                    queue_attempt: 1, mark: 'pause' },
            },
        },
    });
    const healthyRootWork = work({ activity_id: 'root-healthy', chat_id: 1, kind: 'managed_task' }, {
        activity: 'working',
        // Exact work controls omit wrap_up and hurry; legacy fallback would invent both.
        controls: [{ action: 'pause', target: 'root-healthy', scope: 'cascade' }, stopOf('root-healthy')],
        members: {
            ids: ['root-healthy'],
            states: {
                'root-healthy': { confirmed: true, activity: 'working', transition: 'none', call: 'working', live: true,
                    release: 'self', turn: [], tails: [], because: [], queue_attempt: 1, mark: 'moving' },
            },
        },
    });
    routes.set('/api/tasks?queue_only=1', () => {
        if (failReads) throw new Error('queue read unavailable');
        return response({
            queue: {
                running: [{ id: 'root-healthy', type: 'task', attempt: 1, runtime_sec: 12,
                    task: { id: 'root-healthy', root_task_id: 'root-healthy', suggested_name: 'Healthy root' } }],
                pending: [{ id: 'helper-stop', type: 'task', attempt: 1,
                    task: { id: 'helper-stop', root_task_id: 'root-no-stop', parent_task_id: 'root-no-stop',
                        delegation_role: 'subagent', suggested_name: 'Windows window check' } }],
            },
        });
    });
    routes.set('/api/state', () => {
        if (failReads) throw new Error('state read unavailable');
        return response({
            bg_consciousness_enabled: false,
            active_chat_activities: [{ activity_id: 'root-healthy', kind: 'managed_task', phase: 'working', cancelable: true, task_attempt: 1 }],
            active_chat_activities_complete: true,
            works: [helperWork, healthyRootWork],
            works_complete: true,
        });
    });
    routes.set('/api/schedules', response({ tasks: [] }));
    routes.set('/api/services/kept', response({ services: [], complete: true }));
    routes.set('/api/tasks/helper-stop', response({ id: 'helper-stop', status: 'running' }));
    routes.set('/api/tasks/root-healthy', response({ id: 'root-healthy', status: 'running' }));
    routes.set('/api/tasks/helper-stop/cancel', (_url, init) => {
        cancels.push(JSON.parse(init.body));
        return response({ ok: true, task_id: 'helper-stop' });
    });

    activity = initActivity({ mount, ws });
    await activity.refresh();
    const helperStopBtn = mount.querySelector('[data-act="task-control"][data-id="helper-stop"]');
    const rootBtn = mount.querySelector('[data-act="task-control"][data-id="root-healthy"][data-root="1"]');
    assert.ok(helperStopBtn, 'expired helper initially offers Stop now');
    assert.ok(rootBtn, 'healthy root initially offers its work controls');
    rootBtn.getBoundingClientRect = rect;

    // Open the helper Stop dialog before both reads fail, then fail both /api/tasks?queue_only=1 and /api/state.
    const clickingHelper = mount.fire('click', { target: helperStopBtn });
    await settle();
    const openDialog = document.body.querySelector('[role="dialog"]');
    assert.ok(openDialog, 'helper Stop dialog opens before reads fail');

    failReads = true;
    await activity.refresh();

    // Confirming the previously open helper dialog submits nothing once the refresh reads failed.
    await openDialog.parentElement.fire('click', { target: openDialog.querySelector('[data-confirm-ok]') });
    await clickingHelper;
    await settle();
    assert.equal(cancels.length, 0, 'in-flight expired helper confirm submits nothing after both reads fail');

    // Re-clicking the retained expired helper button also refuses to open/submit.
    await mount.fire('click', { target: helperStopBtn });
    await settle();
    assert.equal(document.body.querySelector('[role="dialog"]'), null, 'retained expired helper button refuses open after both reads fail');
    assert.equal(cancels.length, 0, 're-clicking expired helper submits nothing after both reads fail');

    // Clicking the retained healthy root button keeps its prior exact work controls (Pause, Stop now),
    // never falling through to the broad legacy menu with Wrap up / Hurry up.
    await mount.fire('click', { target: rootBtn });
    await settle();
    const menu = document.body.querySelector('.task-control-menu');
    assert.ok(menu, 'healthy root still opens its prior exact work controls menu');
    const items = menu.querySelectorAll('[role="menuitem"]').map((el) => ({ action: el.dataset.taskControl, label: el.textContent }));
    assert.deepEqual(items, [
        { action: 'pause', label: 'Pause' },
        { action: 'stop_now', label: 'Stop now' },
    ], 'known healthy root keeps exact controls and never offers generic Wrap up / Hurry up');
});

test('closeExpiredAction distinguishes closed, preserved result_kept, pending 202, and refuses to invent success on 409/503/unknown', async () => {
    const cases = [
        { name: '200 closed', reply: async () => ({ ok: true, task_id: 't1', request_id: 'r1' }),
            expected: [['Closed.', 'info']] },
        { name: '200 result_kept', reply: async () => ({ ok: true, task_id: 't2', request_id: 'r2', reason_code: 'result_kept' }),
            expected: [['The saved result was kept.', 'info']] },
        { name: '202 pending', reply: async () => ({ ok: true, task_id: 't3', request_id: 'r3', cancel_state: 'pending' }),
            expected: [['Close requested. The saved work is still being checked.', 'info']] },
        { name: '409 stale', reply: async () => { const err = new Error('The saved work changed. Nothing was closed.'); err.status = 409; throw err; },
            expected: [['The saved work changed. Nothing was closed.', 'warn']] },
        { name: '409 partial', reply: async () => { const err = new Error('Some parked members were closed; changed work was left alone.'); err.status = 409; throw err; },
            expected: [['Some parked members were closed; changed work was left alone.', 'warn']] },
        { name: '503 unavailable', reply: async () => { const err = new Error('The saved work could not be verified. Nothing was closed.'); err.status = 503; throw err; },
            expected: [['The saved work could not be verified. Nothing was closed.', 'warn']] },
        { name: 'unknown non-ok', reply: async () => ({ ok: false }),
            expected: [['Close was not confirmed.', 'warn']] },
    ];
    for (const item of cases) {
        const toasts = [];
        let calls = 0;
        let settled = 0;
        await closeExpiredAction(`task-${item.name}`, TOKEN, {
            close: async () => { calls += 1; return item.reply(); },
            toast: (msg, level) => toasts.push([msg, level]),
            settle: async () => { settled += 1; },
        });
        assert.equal(calls, 1, `${item.name}: exactly one request, no blanket retries`);
        assert.equal(settled, 1, `${item.name}: settles once`);
        assert.deepEqual(toasts, item.expected, item.name);
    }
});

test('expired quiz reader uses authoritative per-member expiry without promising Resume, surviving an unrelated unknown sibling', () => {
    const fx = decisionFixture();
    try {
        const makeCard = (taskId, quizId, extra) => fx.decision.buildQuizCard({
            task_id: taskId, quiz_id: quizId, question: `Question for ${taskId}?`,
            options: ['Keep', 'Discard'], ts: '2026-11-23T07:05:00Z', ...extra,
        });
        const savedDelivery = { state: 'answered', answered_index: 0,
            question_delivery: { state: 'owed', reason_code: 'awaiting_recipient_read' } };
        const expiredSaved = makeCard('t-exp', 'qz-1', savedDelivery);
        const expiredWait = makeCard('t-exp', 'qz-2', { state: 'open', owner_wait_state: 'waiting' });
        const healthySaved = makeCard('t-healthy', 'qz-3', savedDelivery);
        const unknownSaved = makeCard('t-unknown', 'qz-4', savedDelivery);
        const statusOf = (card) => card.querySelector('.chat-quiz-status-text').textContent;

        // Real server shape (gateway/work_status.py:661): an unrelated unknown sibling (`t-unknown`) makes the
        // aggregate work `confirmed: false`, while `t-exp` keeps its own `confirmed: true` member state.
        const body = {
            supervisor_ready: true, works_complete: true, rooms: {},
            works: [
                work({ activity_id: 't-exp', chat_id: 2, kind: 'managed_task' }, {
                    confirmed: false, activity: 'unknown', because: [],
                    members: {
                        ids: ['t-exp', 't-unknown'],
                        states: {
                            't-exp': { confirmed: true, activity: 'paused', because: [EXPIRED, PAUSED] },
                            't-unknown': { confirmed: false, activity: 'paused', because: [EXPIRED] },
                        },
                    },
                }),
                work({ activity_id: 't-healthy', chat_id: 2, kind: 'managed_task' }, {
                    confirmed: true, activity: 'paused', because: [PAUSED],
                    members: {
                        ids: ['t-healthy'],
                        states: { 't-healthy': { confirmed: true, activity: 'paused', because: [PAUSED] } },
                    },
                }),
            ],
        };
        const state = reconcileWorkState(emptyWorkState(), body, 1);
        assert.deepEqual([memberPaused(state, 't-exp'), memberDeadlineExpired(state, 't-exp')], [false, true]);
        assert.deepEqual([memberPaused(state, 't-healthy'), memberDeadlineExpired(state, 't-healthy')], [true, false]);
        assert.deepEqual([memberPaused(state, 't-unknown'), memberDeadlineExpired(state, 't-unknown')], [false, false]);

        fx.decision.syncWorks(state);
        assert.equal(statusOf(expiredSaved), 'Answer saved · task deadline expired');
        assert.doesNotMatch(statusOf(expiredSaved), /\bResume\b|when it continues/i);
        assert.equal(statusOf(expiredWait), 'Unanswered · task deadline expired');
        assert.doesNotMatch(statusOf(expiredWait), /\bResume\b/i);
        assert.equal(statusOf(healthySaved), 'Answer saved · read after Resume');
        assert.equal(statusOf(unknownSaved), 'Answer saved · the task reads it when it continues');

        // Disconnected, omitted-work partial read, and failed read revert to plain words rather than holding stale expiry.
        for (const stale of [
            reconcileWorkState(state, body, 1, false),
            reconcileWorkState(state, { supervisor_ready: true, works_complete: false, rooms: {}, works: [] }, 1, true),
            reconcileWorkState(state, null, 1, true),
        ]) {
            fx.decision.syncWorks(stale);
            assert.equal(statusOf(expiredSaved), 'Answer saved · the task reads it when it continues');
            assert.equal(statusOf(expiredWait), 'Waiting for your answer');
            assert.equal(statusOf(healthySaved), 'Answer saved · the task reads it when it continues');
        }
    } finally {
        fx.decision.destroy();
        fx.restore();
    }
});

test('cancel_presentation attributes parent-issued deadline_close as requested by parent without altering other sources', () => {
    const child = { task_id: 'child', parent_task_id: 'parent', root_task_id: 'parent', status: 'cancelled' };
    // Exact direct parent proven by typed agent_task request_origin: emit "Requested by its parent task" once (no generic "Requested by a task").
    assert.deepEqual(
        cancelCauseClauses({
            source: 'deadline_close',
            reason: 'the deadline expired',
            requested_by: 'parent',
            request_origin: { kind: 'agent_task', task_id: 'parent' },
        }, child).filter(Boolean),
        ['Closed after deadline expired', 'Requested by its parent task'],
    );
    assert.equal(
        taskReasonDetail({ ...child, cancel_origin: { source: 'deadline_close', reason: 'the deadline expired',
            requested_by: 'parent', request_origin: { kind: 'agent_task', task_id: 'parent' } } }),
        'Closed after deadline expired · Requested by its parent task.',
    );
    // Negative cases on deadline_close: no typed origin, a non-agent one, or one with no actor states NO relation
    // at all. A Close is never "with" its parent, so an ordinary Stop's lineage wording is excluded here — the
    // twin of supervisor.cancel_publication.cancel_cause_clauses keeping the same words.
    const unproven = [
        ['no typed origin at all', { source: 'deadline_close', reason: 'the deadline expired', requested_by: 'parent' }],
        ['a non-agent typed origin', { source: 'deadline_close', reason: 'the deadline expired', requested_by: 'parent',
            request_origin: { kind: 'http', task_id: 'parent' } }],
        ['an agent origin with no actor', { source: 'deadline_close', reason: 'the deadline expired', requested_by: 'parent',
            request_origin: { kind: 'agent_task', task_id: '' } }],
    ];
    for (const [name, origin] of unproven) {
        const clauses = cancelCauseClauses(origin, child).filter(Boolean);
        assert.deepEqual(clauses, ['Closed after deadline expired'], `${name}: claims no lineage the record cannot prove`);
        assert.ok(!clauses.some(clause => clause.includes('Stopped with')),
            `${name}: never "with" its parent, as an ordinary Stop of the same record reads`);
    }
    // A mismatched typed actor is a real asker, but not this record's parent: the task is named, its lineage is not.
    assert.deepEqual(
        cancelCauseClauses({
            source: 'deadline_close',
            reason: 'the deadline expired',
            requested_by: 'parent',
            request_origin: { kind: 'agent_task', task_id: 'other' },
        }, child).filter(Boolean),
        ['Closed after deadline expired', 'Requested by a task'],
    );
    // Owner-issued deadline_close (no requested_by) and root-close reason keep their exact wording.
    assert.equal(
        taskReasonDetail({ ...child, cancel_origin: { source: 'deadline_close', reason: 'the deadline expired' } }),
        'Closed after deadline expired.',
    );
    assert.equal(
        taskReasonDetail({ ...child, cancel_origin: { source: 'deadline_close', reason: "the work's deadline expired" } }),
        "Closed after deadline expired · the work's deadline expired.",
    );
    // A non-parent ancestor on deadline_close claims neither that it requested the Close nor that the task was
    // stopped with it; legacy agent_tool and cascade_descendant keep their existing wording unchanged.
    const grandchild = { task_id: 'grandchild', parent_task_id: 'child', root_task_id: 'root', status: 'cancelled' };
    const ancestorClose = cancelCauseClauses({
        source: 'deadline_close',
        reason: 'the deadline expired',
        requested_by: 'root',
        request_origin: { kind: 'agent_task', task_id: 'root' },
    }, grandchild).filter(Boolean);
    assert.deepEqual(ancestorClose, ['Closed after deadline expired', 'Requested by a task']);
    assert.ok(!ancestorClose.some(clause => clause.includes('ancestor task')), 'an ancestor gets no lineage wording of its own');
    assert.deepEqual(
        cancelCauseClauses({ source: 'cascade_descendant', requested_by: 'parent' }, child).filter(Boolean),
        ['Stopped with the task tree it belongs to', 'Stopped with its parent task'],
    );
    assert.deepEqual(
        cancelCauseClauses({ source: 'agent_tool', requested_by: 'parent' }, child).filter(Boolean),
        ['Stopped by Ouroboros', 'Requested by its parent task'],
    );
});

test('Activity indexes Close offers and expiry once per accepted census, groups expired apart from Paused, and proves clicks against latest census', async (t) => {
    const response = (data, status = 200) => ({ ok: status < 400, status, json: async () => data });
    let activity = null;
    t.after(() => activity?.destroy());
    const { mount, routes, ws } = setupDashboard(t);
    let controlReads = 0;
    let activityArrayReads = 0;
    let attentionArrayReads = 0;
    const mkWork = (id, controls, extra = {}) => {
        const w = work({ activity_id: id, chat_id: 1, kind: 'managed_task' }, { activity: 'paused', because: [EXPIRED, PAUSED], ...extra });
        Object.defineProperty(w, 'controls', {
            enumerable: true,
            get() { controlReads += 1; return controls; },
        });
        return w;
    };
    const works = [
        mkWork('w-1', [closeOf('w-1', TOKEN, 'cascade')]),
        mkWork('w-2', [closeOf('h-2', OTHER, 'single')]),
        mkWork('w-healthy', [stopOf('w-healthy')], { because: [PAUSED] }),
        mkWork('w-4', []),
    ];
    const makeState = (workList) => {
        const activities = [
            { activity_id: 'a-only', kind: 'managed_task', phase: 'budget_paused', cancelable: false, close_expired: closeOf('a-only', OTHER, 'cascade') },
        ];
        const attentionRows = [
            { task_id: 'exp-no-close', attention_reason: 'deadline_expired', deadline_expired: true },
        ];
        const body = { bg_consciousness_enabled: false, active_chat_activities_complete: true, works: workList, works_complete: true };
        Object.defineProperty(body, 'active_chat_activities', { enumerable: true, get() { activityArrayReads += 1; return activities; } });
        Object.defineProperty(body, 'main_attention', { enumerable: true, get() { attentionArrayReads += 1; return attentionRows; } });
        return body;
    };
    let stateBody = makeState(works);
    // Direct unit check of source precedence in indexCloseOffers and token-independence in indexDeadlineExpired.
    const indexed = indexCloseOffers(stateBody);
    const expiredIndexed = indexDeadlineExpired(stateBody);
    assert.equal(indexed.get('w-1'), TOKEN);
    assert.equal(indexed.get('h-2'), OTHER);
    assert.equal(indexed.get('a-only'), OTHER);
    assert.equal(indexed.has('exp-no-close'), false, 'Close index never invents tokens from deadline flags');
    assert.equal(expiredIndexed.has('exp-no-close'), true, 'expiry index captures deadline-only tasks apart from Close tokens');
    assert.equal(expiredIndexed.has('w-1'), true);
    assert.equal(expiredIndexed.has('w-healthy'), false);
    controlReads = activityArrayReads = attentionArrayReads = 0;

    const mkPending = (n) => Array.from({ length: n }, (_, i) => {
        const id = i === 0 ? 'w-1' : i === 1 ? 'h-2' : i === 2 ? 'w-healthy' : i === 3 ? 'exp-no-close' : `q-${i}`;
        return { id, type: 'task', attempt: 1, task: { id, suggested_name: `Task ${id}` } };
    });
    let pending = mkPending(12);
    const cancels = [];
    routes.set('/api/tasks?queue_only=1', () => response({ queue: { running: [], pending } }));
    routes.set('/api/state', () => response(stateBody));
    routes.set('/api/schedules', response({ tasks: [] }));
    routes.set('/api/services/kept', response({ services: [], complete: true }));
    routes.set('/api/tasks/w-1/cancel', (_url, init) => { cancels.push(['w-1', JSON.parse(init.body)]); return response({ ok: true, task_id: 'w-1' }); });
    routes.set('/api/tasks/h-2/cancel', (_url, init) => { cancels.push(['h-2', JSON.parse(init.body)]); return response({ ok: true, task_id: 'h-2' }); });

    activity = initActivity({ mount, ws });
    await activity.refresh();
    const readsFor12 = { controls: controlReads, activities: activityArrayReads, attention: attentionArrayReads };
    // Doubling queue rows from 12 to 24 adds ZERO extra work.controls, active_chat_activities, or main_attention reads.
    controlReads = activityArrayReads = attentionArrayReads = 0;
    pending = mkPending(24);
    await activity.refresh();
    assert.deepEqual({ controls: controlReads, activities: activityArrayReads, attention: attentionArrayReads }, readsFor12,
        'per-census indices must not rescan arrays per queue row');
    assert.equal(attentionArrayReads, 1, 'main_attention read once per accepted census');

    // Healthy paused rows stay in Paused (1); expired rows group separately under neutral Deadline expired (4: w-1, h-2, exp-no-close, a-only).
    const queueSec = mount.querySelector('[data-activity-section="queue"]');
    const pausedGroup = queueSec.querySelector('details[data-open-key="paused"]');
    const expiredGroup = queueSec.querySelector('details[data-open-key="expired"]');
    assert.equal(pausedGroup.querySelector('summary').textContent, 'Paused (1)');
    assert.match(pausedGroup.textContent, /Task w-healthy/);
    assert.doesNotMatch(pausedGroup.textContent, /Deadline expired/);
    assert.equal(expiredGroup.querySelector('summary').textContent, 'Deadline expired (4)');
    assert.match(expiredGroup.textContent, /Task w-1[\s\S]*Task h-2[\s\S]*Task exp-no-close/);
    // Folding the expired group persists across refresh independently of Paused.
    expiredGroup.open = false;
    await activity.refresh();
    assert.equal(mount.querySelector('details[data-open-key="expired"]').hasAttribute('open'), false);
    assert.equal(mount.querySelector('details[data-open-key="paused"]').hasAttribute('open'), true);

    const btnW1 = mount.querySelector('[data-act="close-expired"][data-id="w-1"]');
    const btnH2 = mount.querySelector('[data-act="close-expired"][data-id="h-2"]');
    assert.ok(btnW1 && btnH2);

    // Valid click against latest accepted census submits exact close_expired payload.
    await mount.fire('click', { target: btnW1 });
    await settle();
    assert.deepEqual(cancels, [['w-1', { stop_policy: 'close_expired', close_expired: TOKEN }]]);

    // A newer accepted census rotates h-2's token: clicking the stale button submits nothing.
    const rotated = 'c'.repeat(64);
    stateBody = makeState([mkWork('w-2', [closeOf('h-2', rotated, 'single')])]);
    await activity.refresh();
    await mount.fire('click', { target: btnH2 });
    await settle();
    assert.equal(cancels.length, 1, 'stale button token rejected by latest accepted census index');

    // Clicking the newly rendered button with the rotated token submits that exact rotated token.
    const freshH2 = mount.querySelector('[data-act="close-expired"][data-id="h-2"]');
    assert.equal(freshH2.dataset.closeExpired, rotated);
    await mount.fire('click', { target: freshH2 });
    await settle();
    assert.deepEqual(cancels[1], ['h-2', { stop_policy: 'close_expired', close_expired: rotated }]);
});
