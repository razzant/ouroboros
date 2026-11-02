// Host notes never claim the card title or the collapsed activity line (owner
// decision 16.09, Q1=A). The worker stamps `narration` on every progress frame;
// only the model's own round narration is promoted, so the projection reads a
// typed fact and never the note's wording (BIBLE P5). Both voices stay visible
// rows, live and on replay, and a frame without the key is an older worker's
// (or a supervisor's) and keeps the promotion it always had.
//
// The card's title slot belongs to a coined name when there is one, and the
// collapsed line is blanked whenever it would repeat the title, so these turns
// are named: that is the only shape where the activity line is observable.
import assert from 'node:assert/strict';
import test, { after } from 'node:test';
import { createChatInstance } from '../modules/chat.js';
import { summarizeChatLiveEvent } from '../modules/log_events.js';
import { buildTimelineItemHtml } from '../modules/chat_activity.js';
import { formatMsgTime } from '../modules/utils.js';
import { ElementStub, installDom, restoreDom, walkCard } from './chat_dom_fixture.js';

const originalQuery = ElementStub.prototype.querySelector;
after(() => { ElementStub.prototype.querySelector = originalQuery; });
ElementStub.prototype.querySelector = function (selector) {
    const direct = originalQuery.call(this, selector);
    if (direct) return direct;
    for (const child of this.children) { const found = child.querySelector(selector); if (found) return found; }
    return null;
};

const TS = '2026-09-16T12:00:00Z';
const TASK = 'turn-voice';
const NAME = 'Flake hunt';
const SPOKEN = 'Reading the failing test first.';
const NOTE = 'Checkpoint 3 at round 12';

function fixture(history = []) {
    const env = installDom(async (url) => ({ ok: true, json: async () =>
        String(url).startsWith('/api/chat/history')
            ? { messages: history, window: { complete: true } }
            : { active_direct_turns: [] } }));
    const handlers = new Map();
    let generation = 0;
    const instance = createChatInstance({
        ws: { on(type, fn) { handlers.set(type, fn); return () => handlers.delete(type); },
            isConnected: () => true, send() {} },
        state: { activePage: 'chat', projectChatIds: new Set(), unreadCount: 0 },
        updateUnreadBadge() {}, chatId: 1, idPrefix: 'chat', mountEl: env.mount,
        stateSnapshots: { begin: () => ({ generation: ++generation, requestedAt: Date.now() }), gate() { return Promise.resolve(this.begin()); },
            isCurrent: () => true, apply() {} },
    });
    const messages = document.byId.get('chat-messages');
    const nodes = (node) => [node, ...(node?.children || []).flatMap(nodes)];
    const card = (id = TASK) => walkCard(messages, id);
    return {
        instance, messages, card, handlers,
        rows: (id = TASK) => nodes(card(id)).filter((n) => n.classList?.contains('chat-live-line')),
        title: (id = TASK) => card(id)?.querySelector('[data-live-title]')?.textContent ?? null,
        activity: (id = TASK) => card(id)?.querySelector('[data-live-activity]')?.textContent ?? null,
        census: (rows) => instance.hydrateStateSnapshot({
            active_chat_activities: rows, active_chat_activities_complete: true, supervisor_ready: true,
        }, Infinity, ++generation),
        emit: (row) => handlers.get('chat')({ chat_id: 1, ts: TS, task_id: TASK,
            role: 'assistant', is_progress: true, ...row }),
        close() { instance.destroy(); restoreDom(env.prior); },
    };
}

const direct = (id = TASK) => [{ activity_id: id, chat_id: 1, kind: 'direct_chat', phase: 'thinking' }];
const row = (patch) => ({ task_id: TASK, role: 'assistant', is_progress: true, chat_id: 1,
    ts: '2026-09-16T12:00:01Z', ...patch });

test('the projection reads the voice, not the wording: only narration is promoted', () => {
    const note = { type: 'send_message', is_progress: true, task_id: TASK, content: NOTE, narration: false };
    const host = summarizeChatLiveEvent(note);
    assert.deepEqual([host.promote, host.human, host.visible], [false, false, true],
        'a host note is still a visible row');
    assert.deepEqual([host.phase, host.headline], ['working', NOTE], 'with its phase and its text');
    const spoken = summarizeChatLiveEvent({ ...note, content: SPOKEN, narration: true });
    assert.deepEqual([spoken.promote, spoken.human], [true, true]);
    // Identical wording, opposite voice: no text rule could tell these apart.
    const same = summarizeChatLiveEvent({ ...note, content: SPOKEN, narration: false });
    assert.deepEqual([same.headline, same.promote, same.human], [spoken.headline, false, false]);
    const { narration, ...legacy } = note;
    const older = summarizeChatLiveEvent(legacy);
    assert.deepEqual([older.promote, older.human], [true, true],
        'a frame from before the fact keeps the legacy promotion');
});

test('host notes are rows the card shows, never its title', () => {
    const f = fixture();
    try {
        f.census(direct());
        f.emit({ content: NOTE, narration: false });
        f.emit({ content: 'Task acceptance review: PASS (clean acceptance).', narration: false });
        assert.ok(f.card(), 'the notes are content: the block exists');
        assert.equal(f.rows().length, 2, 'both notes are visible rows');
        assert.equal(f.title(), '', 'no title: neither the last host note nor a placeholder repeating the chip (#1369)');
    } finally { f.close(); }
});

test('only the host placeholder left: authored words that say Working keep the title (#1369)', () => {
    const f = fixture();
    try {
        f.census(direct());
        f.emit({ content: 'Working…', narration: true });
        assert.equal(f.title(), 'Working…', 'the model narrated these words; they are its own');
    } finally { f.close(); }
    const g = fixture();
    try {
        g.census(direct());
        g.emit({ content: NOTE, narration: false, suggested_name: 'Working...' });
        assert.equal(g.title(), 'Working...', 'a coined name is the author\'s, whatever it says');
    } finally { g.close(); }
});

test('a host-notes-only turn keeps its coined name and an empty activity line', () => {
    const f = fixture();
    try {
        f.census(direct());
        f.emit({ content: NOTE, narration: false, suggested_name: NAME });
        f.emit({ content: '📐 Plan review: wave 1 dispatched', narration: false });
        assert.equal(f.rows().length, 2);
        assert.equal(f.title(), NAME);
        assert.equal(f.activity(), '', 'no host note ever reaches the collapsed line');
    } finally { f.close(); }
});

test('narration owns the activity line and a later host note cannot take it', () => {
    const f = fixture();
    try {
        f.census(direct());
        f.emit({ content: SPOKEN, narration: true, suggested_name: NAME });
        assert.equal(f.title(), NAME);
        assert.equal(f.activity(), SPOKEN, 'the turn voice reaches the collapsed line');
        f.emit({ content: '⚡ Fallback: switching model lane', narration: false });
        assert.equal(f.rows().length, 2, 'the fallback note is still shown');
        assert.equal(f.title(), NAME);
        assert.equal(f.activity(), SPOKEN, 'the host note does not overwrite it');
        f.emit({ content: 'Running the suite now.', narration: true });
        assert.equal(f.activity(), 'Running the suite now.', 'the next narration does');
    } finally { f.close(); }
});

test('a frame without the key is a legacy frame and still leads the card', () => {
    const f = fixture();
    try {
        f.census(direct());
        f.emit({ content: 'Working on the big thing', suggested_name: NAME });
        f.emit({ content: 'Still working through it' });
        assert.equal(f.title(), NAME);
        assert.equal(f.activity(), 'Still working through it');
    } finally { f.close(); }
});

test('replay reads the voice exactly as live did', async () => {
    const opening = { role: 'user', text: 'fix the flake', ts: TS, chat_id: 1 };
    const f = fixture([opening,
        row({ text: NOTE, content: NOTE, narration: false, suggested_name: NAME }),
        row({ text: '📐 Plan review: wave 1 dispatched', content: '📐 Plan review: wave 1 dispatched',
            narration: false, ts: '2026-09-16T12:00:02Z' })]);
    try {
        await f.instance.refreshHistory({ revision: 1 });
        assert.ok(f.card(), 'the replayed notes are content');
        assert.equal(f.rows().length, 2, 'and stay visible rows');
        assert.equal(f.title(), NAME);
        assert.equal(f.activity(), '', 'a reloaded host note does not become the activity line');
    } finally { f.close(); }

    const g = fixture([opening,
        row({ text: SPOKEN, content: SPOKEN, narration: true, suggested_name: NAME }),
        row({ text: NOTE, content: NOTE, narration: false, ts: '2026-09-16T12:00:02Z' })]);
    try {
        await g.instance.refreshHistory({ revision: 1 });
        assert.equal(g.title(), NAME);
        assert.equal(g.activity(), SPOKEN, 'narration still leads after a reload');
    } finally { g.close(); }

    // A row stored before the fact existed carries no key and replays promoted.
    const h = fixture([opening,
        row({ text: NOTE, content: NOTE, suggested_name: NAME }),
        row({ text: 'Still working through it', content: 'Still working through it',
            ts: '2026-09-16T12:00:02Z' })]);
    try {
        await h.instance.refreshHistory({ revision: 1 });
        assert.equal(h.activity(), 'Still working through it');
    } finally { h.close(); }
});

test('finalizing outcome overlay keeps narration as title while publishing phase and model', () => {
    const f = fixture();
    try {
        f.census(direct());
        f.emit({ content: '💬 still working', narration: true });
        f.emit({ content: '💬 still working', narration: true, task_phase: 'finalizing',
            outcome_final: false, outcome_axes: { execution: { status: 'ok' } },
            model_execution: { source: 'usable_solve_response', used_model: 'actual-solver' } });
        assert.equal(f.title(), 'still working');
        assert.equal(f.card().querySelector('[data-live-phase]').textContent, 'Finalizing…');
        // The observed solver is the compact line's model; its full solve evidence is a row of the details.
        assert.match(f.card().querySelector('[data-live-meta]').innerHTML, />actual-solver</);
        assert.match(f.card().querySelector('[data-live-facts]').innerHTML, /<dt>Last solve response<\/dt><dd>actual-solver<\/dd>/);
        f.emit({ is_progress: false, role: 'system', system_type: 'task_summary', content: 'Done.',
            outcome_final: true, outcome_phase: 'done', outcome_axes: { execution: { status: 'ok' } } });
        assert.equal(f.title(), 'still working');
    } finally { f.close(); }
});

test('the collapsed line states the terminal cause, and a clean ending keeps the last narration', () => {
    const f = fixture();
    try {
        f.census(direct());
        f.emit({ content: '💬 reading the failing test first', narration: true, suggested_name: 'Flake hunt' });
        assert.equal(f.activity(), 'reading the failing test first');
        f.emit({ is_progress: false, role: 'system', system_type: 'task_summary', content: 'Done with warnings.',
            outcome_final: true, outcome_phase: 'warn', reason_code: 'plan_review_advisory',
            outcome_axes: { execution: { status: 'degraded', reason_code: 'plan_review_advisory', plan_review: 'unanswered' } } });
        assert.equal(f.activity(), 'Only some of the plan reviewers answered; the work went on with their notes.');
    } finally { f.close(); }
    const g = fixture();
    try {
        g.census(direct());
        g.emit({ content: '💬 reading the failing test first', narration: true, suggested_name: 'Flake hunt' });
        g.emit({ is_progress: false, role: 'system', system_type: 'task_summary', content: 'Done.',
            outcome_final: true, outcome_phase: 'done', outcome_axes: { execution: { status: 'ok' } } });
        assert.equal(g.activity(), 'reading the failing test first');
    } finally { g.close(); }
});

// Display reasoning is the model's own thinking, typed `reasoning` on its narration (owner 01.11 10:53, A): a Details
// row whose preview tells one thought from the next, Expand holding the whole text; never the title or collapsed line.
const THOUGHT = ['I keep weighing whether the edit — a body change on an already merged change — needs a reply.',
    'It does not: **nobody asked**, the note `#42` is linked, and "quiet" means quiet.'].join('\n\n').repeat(40);
// The flat fixture parses innerHTML without text; read the row markup the renderer built.
function renderedRows() {
    const create = document.createElement.bind(document), built = [];
    document.createElement = (tag) => { const node = create(tag); built.push(node); return node; };
    return () => built.map((node) => node.innerHTML).filter((html) => html.includes('class="chat-live-line '));
}
const titles = (markup) => markup.map((html) => (html.match(/chat-live-line-title[^>]*>(?:<p>)?([^<]*)/) || [])[1]);

test('display reasoning is a Thinking row holding the whole text, never the title or the collapsed line', () => {
    const frame = { type: 'send_message', is_progress: true, task_id: TASK, content: `💬 ${THOUGHT}`, narration: true };
    const view = summarizeChatLiveEvent({ ...frame, reasoning: true });
    assert.deepEqual([view.promote, view.human, view.visible, view.headline, view.activityPreview],
        [false, false, true, 'Thinking', '']);
    assert.equal(view.fullBody, THOUGHT.trim(), 'the reasoning is kept whole for its disclosure');
    const child = summarizeChatLiveEvent({ ...frame, reasoning: true, delegation_role: 'subagent',
        subagent_task_id: 'child-1', parent_task_id: TASK, subagent_event: 'running' });
    assert.deepEqual([child.promote, child.human, child.headline, child.activityPreview], [false, false, 'Thinking', '']);
    const speech = summarizeChatLiveEvent(frame);
    assert.deepEqual([speech.promote, speech.human, speech.reasoning], [true, true, undefined],
        'visible speech, like a row stored before the fact, keeps leading');
    const f = fixture();
    try {
        const item = { ...view, lineKey: 'k', count: 1 };
        const open = buildTimelineItemHtml(item, { expandedLineKeys: new Set(['k']), groupId: TASK });
        assert.ok(open.includes('quiet') && open.includes('Collapse'), 'Expand shows the reasoning itself');
        assert.ok(open.length > THOUGHT.length, 'and all of it');
        const closed = buildTimelineItemHtml(item, { expandedLineKeys: new Set(), groupId: TASK });
        assert.ok(closed.includes('I keep weighing') && closed.includes('Expand'), 'a closed row shows its own opening words');
        assert.ok(view.body.length <= 240 && closed.length < THOUGHT.length, 'and only those');
    } finally { f.close(); }
});

test('live and replay: reasoning joins the timeline while the spoken line and title stay', async () => {
    const f = fixture();
    try {
        const rendered = renderedRows();
        f.census(direct());
        f.emit({ content: `💬 ${THOUGHT}`, narration: true, reasoning: true });
        assert.equal(f.title(), '', 'an unnamed turn is not titled by its thinking');
        f.emit({ content: SPOKEN, narration: true });
        f.emit({ content: `💬 ${THOUGHT} later`, narration: true, reasoning: true, ts: '2026-09-16T12:00:05Z' });
        assert.equal(f.title(), SPOKEN);
        assert.equal(f.activity(), '', 'the spoken title is not repeated, and thinking does not take the line');
        assert.equal(f.rows().length, 3);
        assert.deepEqual([...new Set(titles(rendered()))].sort(), [SPOKEN, 'Thinking'].sort());
    } finally { f.close(); }
    const g = fixture([{ role: 'user', text: 'fix the flake', ts: TS, chat_id: 1 },
        row({ text: SPOKEN, content: SPOKEN, narration: true, suggested_name: NAME }),
        row({ text: `💬 ${THOUGHT}`, content: `💬 ${THOUGHT}`, narration: true, reasoning: true, ts: '2026-09-16T12:00:02Z' })]);
    try {
        const rendered = renderedRows();
        await g.instance.refreshHistory({ revision: 1 });
        assert.equal(g.title(), NAME);
        assert.equal(g.activity(), SPOKEN, 'after a reload the spoken words keep the line');
        assert.equal(g.rows().length, 2);
        assert.ok(titles(rendered()).includes('Thinking'));
    } finally { g.close(); }
});

// A thought's whole words, read through the card's own disclosure: the row's real click, then the row it renders.
function expandedRowText(f, id, row, rendered) {
    const nodes = (node) => [node, ...(node?.children || []).flatMap(nodes)];
    const timeline = nodes(f.card(id)).find((n) => n.listeners?.has('click') && n.classList?.contains('chat-live-timeline'));
    const target = { closest: (sel) => (sel.includes('.chat-live-line') ? row : null) };
    timeline.listeners.get('click')[0]({ target, stopPropagation() {} });
    return rendered().filter((html) => html.includes('data-expanded="1"')).at(-1);
}
const CHILD = 'child-voice';
const THOUGHTS = ['First I weigh the flake against timing.', 'Second I weigh the flake against test order.',
    'Third I weigh the fixture cleanup.'].map((opening, n) => `${opening} ${'More thinking on it. '.repeat(30)}END-${n}`);
const childRow = (words, ts, extra = {}) => ({ chat_id: 1, task_id: CHILD, subagent_task_id: CHILD, parent_task_id: TASK,
    root_task_id: TASK, delegation_role: 'subagent', subagent_role: 'reader', model: 'm', role: 'assistant',
    is_progress: true, narration: true, content: `💬 ${words}`, text: `💬 ${words}`, ts, ...extra });
const CHILD_SPEECH = 'Reading the fixture.';
const CHILD_NOTE = 'Read the fixture; the flake is the cleanup order.';
const bodies = (markup) => markup.map((html) => (html.match(/class="chat-live-line-body[^"]*"[^>]*>(?:<p>)?([^<]*)/) || [])[1] || '');

test('a child keeps every thought and its next note as their own rows, live and on replay', async () => {
    const frames = (extra = {}) => [
        childRow('Starting.', '2026-09-16T12:00:01Z', { subagent_event: 'running', narration: false, role: 'system', system_type: 'host_progress' }),
        childRow(CHILD_SPEECH, '2026-09-16T12:00:02Z', { subagent_event: 'progress', ...extra }),
        ...THOUGHTS.map((words, n) => childRow(words, `2026-09-16T12:00:0${n + 3}Z`, { subagent_event: 'progress', reasoning: true, ...extra })),
        childRow(CHILD_NOTE, '2026-09-16T12:00:07Z', { subagent_event: 'progress', ...extra }),
    ];
    const check = (f, rendered, where) => {
        // A child's timeline renders when its Details open (the owner's own gesture).
        f.card(CHILD).querySelector('[data-live-disclosure]').click();
        const rows = f.rows(CHILD), markup = rendered();
        // The guard under test: without it the second thought overwrites the first through `subagent-progress:`.
        const thoughts = [...new Set(markup.filter((html) => html.includes('>Thinking')))];
        for (const [n, words] of THOUGHTS.entries()) {
            const opening = words.split(' More')[0];
            assert.ok(thoughts.some((html) => html.includes(opening)), `${where}: thought ${n} keeps its own row`);
        }
        assert.ok(rows.length >= THOUGHTS.length + 1, `${where}: ${rows.length} rows`);
        // Rows stand in the order they happened, each with its own words: speech, thoughts, then the next note.
        const inOrder = rows.map((node) => markup.findLast((html) => html.includes(`data-live-line-key="${node.dataset.liveLineKey}"`)) || '');
        const at = (words) => inOrder.findIndex((html) => html.includes(words));
        assert.deepEqual([at(CHILD_SPEECH) < at(THOUGHTS[0].split(' More')[0]), at(THOUGHTS[2].split(' More')[0]) < at(CHILD_NOTE)], [true, true],
            `${where}: rows in order, the speech and the next note both kept: ${[at(CHILD_SPEECH), at(CHILD_NOTE)]}`);
        assert.equal(new Set(bodies(markup.filter((html) => html.includes('>Thinking')))).size, THOUGHTS.length, `${where}: rows tell apart`);
        const line = f.activity(CHILD);
        assert.equal(line, CHILD_NOTE, `${where}: only the spoken note takes the child's line`);
        assert.ok(!markup.some((html) => /chat-live-activity[^<]*Thinking/.test(html)), `${where}: no Thinking line`);
        // Expand holds the whole words of a thought, not a neighbour's.
        const first = rows.find((row) => row.dataset.liveLineKey && row.classList.contains('expandable'));
        assert.ok(first, `${where}: a thought is expandable`);
        const open = expandedRowText(f, CHILD, first, rendered);
        assert.ok(/END-\d/.test(open) && open.includes('More thinking on it.'), `${where}: Expand shows the whole thought`);
    };
    const live = fixture();
    try {
        const rendered = renderedRows();
        live.census(direct());
        live.emit({ content: SPOKEN, narration: true });
        for (const frame of frames()) live.handlers.get('chat')(frame);
        check(live, rendered, 'live');
    } finally { live.close(); }
    const stored = (frame, n) => ({ ...frame, history_id: `progress:${n + 10}`, history_position: { source: 'progress', offset: n + 10 } });
    const saved = fixture([row({ text: SPOKEN, content: SPOKEN, narration: true, suggested_name: NAME }), ...frames().map(stored)]);
    try {
        const rendered = renderedRows();
        await saved.instance.refreshHistory({ revision: 1 });
        check(saved, rendered, 'replay');
    } finally { saved.close(); }
});

test('reasoning moves the card clock like the speech it was; a host note or a cost-only frame does not', () => {
    const f = fixture();
    const at = (minute) => `2026-09-16T12:${minute}:00Z`;
    const time = (minute) => `updated ${formatMsgTime(at(minute)).short}`;
    try {
        f.census(direct());
        f.emit({ content: SPOKEN, narration: true, suggested_name: NAME, ts: at('01') });
        // The flat fixture parses the head's markup onto the card itself; the head reaches its clock through the card.
        const card = f.card();
        card.querySelector('[data-live-summary-button]').querySelector = (selector) => card.querySelector(selector);
        const clock = () => card.querySelector('[data-live-when]').textContent;
        f.emit({ content: `💬 ${THOUGHT}`, narration: true, reasoning: true, ts: at('05') });
        assert.equal(clock(), time('05'), 'thinking is activity');
        assert.equal(f.activity(), SPOKEN, 'its words still stay out of the line');
        f.emit({ content: NOTE, narration: false, role: 'system', ts: at('09') });
        assert.equal(clock(), time('05'), 'a host note is not');
        f.handlers.get('log')({ chat_id: 1, data: { type: 'task_metrics', task_id: TASK, cost_usd: 1.5, ts: at('12') } });
        assert.equal(clock(), time('05'), 'a cost-only frame is not');
        // A child keeps its clock on its meta line, and its spoken line under the same rule.
        f.handlers.get('chat')(childRow('Starting.', at('20'), { subagent_event: 'progress' }));
        f.handlers.get('chat')(childRow(THOUGHTS[0], at('25'), { subagent_event: 'progress', reasoning: true }));
        const meta = f.card(CHILD).querySelector('[data-live-meta]');
        assert.ok(meta.innerHTML.includes(time('25')), meta.innerHTML);
        assert.equal(f.activity(CHILD), 'Starting.', 'the child keeps its spoken line');
    } finally { f.close(); }
});

// One headline owner: naming goes through `syncCardHeadline`, which writes only on change (a rewrite loses a selection).
function countWrites(el) {
    const proto = Object.getOwnPropertyDescriptor(ElementStub.prototype, 'textContent');
    const seen = { writes: 0 };
    Object.defineProperty(el, 'textContent', { configurable: true, get() { return proto.get.call(this); },
        set(value) { seen.writes += 1; proto.set.call(this, value); } });
    return seen;
}
const SENTENCE = 'Checked the disk: 30 GiB free, the cause is still unknown.';
const done = (task_id, extra = {}) => ({ chat_id: 1, data: { type: 'task_done', task_id, status: 'completed', ts: '2026-09-16T12:00:30Z', ...extra } });

test('naming has one headline owner: every kind of card keeps its line, and a repeated name writes nothing', () => {
    const named = (f, id, name) => f.handlers.get('task_named')({ task_id: id, suggested_name: name });
    const repeat = (f, id, name) => {
        const card = f.card(id), title = countWrites(card.querySelector('[data-live-title]')), line = countWrites(card.querySelector('[data-live-activity]'));
        for (let n = 0; n < 3; n += 1) named(f, id, name);
        assert.deepEqual([title.writes, line.writes], [0, 0], `${id}: a repeated name rewrites nothing`);
    };
    const f = fixture();
    try {
        f.census([...direct('ordinary'), ...direct('wake'), ...direct('sched')]);
        // A running named root: the deferred narration is the line under its name.
        f.emit({ task_id: 'ordinary', content: SPOKEN, narration: true });
        assert.deepEqual([f.title('ordinary'), f.activity('ordinary')], [SPOKEN, ''], 'an unnamed root is named by its words, once');
        named(f, 'ordinary', NAME);
        assert.deepEqual([f.title('ordinary'), f.activity('ordinary')], [NAME, SPOKEN]);
        repeat(f, 'ordinary', NAME);
        // A wake whose summary is its name: the sentence alone, whichever frame arrives first, however often.
        f.emit({ task_id: 'wake', content: `💬 ${THOUGHT}`, narration: true });
        f.handlers.get('log')(done('wake', { wake_summary: SENTENCE }));
        named(f, 'wake', SENTENCE);
        assert.deepEqual([f.title('wake'), f.activity('wake')], [SENTENCE, '']);
        repeat(f, 'wake', SENTENCE);
        // A schedule's named run: the name above, its sentence below.
        f.emit({ task_id: 'sched', content: SPOKEN, narration: true });
        f.handlers.get('log')(done('sched', { wake_summary: SENTENCE }));
        named(f, 'sched', 'Daily disk check');
        assert.deepEqual([f.title('sched'), f.activity('sched')], ['Daily disk check', SENTENCE]);
        repeat(f, 'sched', 'Daily disk check');
    } finally { f.close(); }
});

test('a continuation takes the work\'s name through the same owner: its line, or its finished summary', async () => {
    const claim = { continued_by: { state: 'admitted', successor_task_id: 'next' } };
    const history = [
        row({ task_id: 'first', text: 'Earlier words', content: 'Earlier words', narration: true, suggested_name: 'Audit the projects', ts: '2026-09-16T12:00:01Z' }),
        row({ task_id: 'first', role: 'system', is_progress: false, system_type: 'task_summary', text: 'Ended', content: 'Ended', task_terminal_status: 'completed',
            outcome_phase: 'done', outcome_final: true, outcome_axes: { lifecycle: { status: 'completed' } }, ts: '2026-09-16T12:00:02Z', ...claim }),
        row({ task_id: 'next', text: 'Going on with it', content: 'Going on with it', narration: true, ts: '2026-09-16T12:00:03Z' }),
    ];
    const f = fixture(history);
    try {
        await f.instance.refreshHistory({ revision: 1 });
        assert.deepEqual([f.title('next'), f.activity('next')], ['Audit the projects', 'Going on with it'],
            'the unnamed successor\'s own line stood in for its title; under the work\'s name it is the line again');
        f.handlers.get('log')(done('next', { wake_summary: SENTENCE }));
        assert.deepEqual([f.title('next'), f.activity('next')], ['Audit the projects', SENTENCE], 'the owner\'s rule, not the raw last narration');
        const title = countWrites(f.card('next').querySelector('[data-live-title]')), line = countWrites(f.card('next').querySelector('[data-live-activity]'));
        f.handlers.get('task_named')({ task_id: 'next', suggested_name: 'Continue: Audit the projects' });
        assert.deepEqual([f.title('next'), f.activity('next'), title.writes, line.writes], ['Audit the projects', SENTENCE, 0, 0]);
    } finally { f.close(); }
});

// A child's rows as its owner reads them, each named by the words it shows and moved only by real frames and refreshes.
// A frame's source occurrence (child, source time, words) names its row across live and history; a note replaces the one
// before it only inside a stretch no thought divides (chat_render_batch.js::childProgressKey).
const WORDS = { A: 'Note A: reading the fixture.', B: 'Note B: the flake is the cleanup order.', C: 'Note C: writing the fix.',
    C2: 'Note C2: fix written.', D: 'Note D: running the suite.', S: 'Starting.',
    H: 'Provider connection restored after 0.1 min — resuming.' };
const nameOf = (text) => Object.keys(WORDS).find((name) => text.includes(WORDS[name]))
    ?? `T${THOUGHTS.findIndex((words) => text.includes(words.split(' More')[0]))}`;
const at = (second) => `2026-09-16T12:00:${String(second).padStart(2, '0')}Z`;
const note = (name, ts, extra = {}) => childRow(WORDS[name], ts, { subagent_event: 'progress', ...extra });
const thought = (n, ts) => childRow(THOUGHTS[n], ts, { subagent_event: 'progress', reasoning: true });
// The source order names a saved row: its microseconds into the minute, plus `tie` for two frames with one source time.
const saved = (frame, tie = 0) => {
    const offset = Math.round(parseFloat(frame.ts.slice(17)) * 1e6) + tie;
    return { ...frame, history_id: `progress:${offset}`, history_position: { source: 'progress', offset } };
};

function childTimeline() {
    const history = [row({ text: SPOKEN, content: SPOKEN, narration: true, suggested_name: NAME })];
    const f = fixture(history), rendered = renderedRows();
    let revision = 1, opened = false;
    // A child's timeline renders when its Details open (the owner's own gesture).
    const markup = () => {
        if (!opened) { f.card(CHILD).querySelector('[data-live-disclosure]').click(); opened = true; }
        const built = rendered();
        return f.rows(CHILD).map((node) => built.findLast((html) => html.includes(`data-live-line-key="${node.dataset.liveLineKey}"`)) || '');
    };
    f.census(direct());
    f.emit({ content: SPOKEN, narration: true });
    return {
        f, markup,
        rows: () => markup().map(nameOf),
        send: (...frames) => frames.forEach((frame) => f.handlers.get('chat')(frame)),
        // The saved window now holds these frames, in this order; a real refresh reads it.
        page: async (...frames) => {
            history.splice(1, Infinity, ...frames.map((frame) => (frame.history_id ? frame : saved(frame))));
            await f.instance.refreshHistory({ revision: revision += 1 });
        },
    };
}
const childTest = (name, body) => test(name, async () => {
    const timeline = childTimeline();
    try { await body(timeline); } finally { timeline.f.close(); }
});

childTest('live, refresh, live: a child keeps one row per note and thought, in order, and later notes stay after them', async ({ f, send, page, rows }) => {
    const start = childRow(WORDS.S, at(1), { subagent_event: 'running', narration: false, role: 'system', system_type: 'host_progress' });
    const [A, T0, B] = [note('A', at(2)), thought(0, at(3)), note('B', at(4))];
    send(start, A, T0, B);
    assert.deepEqual(rows(), ['S', 'A', 'T0', 'B'], 'live');
    // Delayed and overlapping pages: first the saved rows only up to the thought, then all of them, twice.
    await page(start, A, T0);
    assert.deepEqual(rows(), ['S', 'A', 'T0', 'B'], 'a page ending at the thought');
    for (const refresh of [1, 2]) {
        await page(start, A, T0, B);
        assert.deepEqual(rows(), ['S', 'A', 'T0', 'B'], `refresh ${refresh}: each saved row adopts its own`);
    }
    const [C, C2, T1, D] = [note('C', at(5)), note('C2', at(6)), thought(1, at(7)), note('D', at(8))];
    send(C);
    assert.deepEqual(rows(), ['S', 'A', 'T0', 'B', 'C'], 'a later live note follows, overwriting nothing');
    send(C2);
    assert.deepEqual(rows(), ['S', 'A', 'T0', 'B', 'C2'], 'and replaces its own line in place, as live');
    send(T1, D);
    assert.deepEqual(rows(), ['S', 'A', 'T0', 'B', 'C2', 'T1', 'D'], 'a note after a new thought stands after it');
    assert.equal(f.activity(CHILD), WORDS.D, 'the child line is its latest spoken note');
    await page(start, A, T0, B, C, C2, T1, D);
    assert.deepEqual(rows(), ['S', 'A', 'T0', 'B', 'C', 'C2', 'T1', 'D'], 'the caught-up page: every saved note, in its own words');
    send(note('D', at(9)), thought(2, at(10)), note('A', at(11)));
    assert.deepEqual(rows().slice(8), ['D', 'T2', 'A'], 'incoming progress after the caught-up page keeps its order');
});

childTest('earlier history changes no note, whether it comes with the rows it adopts or after them', async ({ send, page, rows }) => {
    const [A, T0, B] = [note('A', at(2)), thought(0, at(3)), note('B', at(4))];
    const [earlierNote, earlierThought] = [note('D', at(0)), thought(1, at(1))];
    send(A, T0, B);
    assert.deepEqual(rows(), ['A', 'T0', 'B']);
    await page(earlierThought, A, T0, B);
    assert.deepEqual(rows(), ['T1', 'A', 'T0', 'B'], 'a page with an unseen earlier thought adopts each row, not a copy');
    await page(earlierNote, earlierThought, A, T0, B);
    assert.deepEqual(rows(), ['D', 'T1', 'A', 'T0', 'B'], 'an older row after those adoptions moves nothing');
    send(note('C', at(5)), note('C2', at(6)));
    assert.deepEqual(rows(), ['D', 'T1', 'A', 'T0', 'B', 'C2'], 'and the next notes overwrite nothing');
});

childTest('a partial page leaves the open note replaceable, and the full page gives each note back its own words', async ({ send, page, rows }) => {
    const [A, T0, B, C] = [note('A', at(2)), thought(0, at(3)), note('B', at(4)), note('C', at(5))];
    const earlier = thought(1, at(1));
    send(A, T0, B);
    await page(earlier, A);
    assert.deepEqual(rows(), ['T1', 'A', 'T0', 'B'], 'the page ends before the open note');
    send(C);
    assert.deepEqual(rows(), ['T1', 'A', 'T0', 'C'], 'C and B stand after the same thought: C replaces B in place');
    for (const refresh of [1, 2]) {
        await page(earlier, A, T0, B, C);
        assert.deepEqual(rows(), ['T1', 'A', 'T0', 'B', 'C'], `refresh ${refresh}: B returns, C stays one row`);
    }
    send(note('D', at(6)));
    assert.deepEqual(rows(), ['T1', 'A', 'T0', 'B', 'C', 'D'], 'a saved row is never replaced');
});

childTest('a saved thought divides as a live one does: a window starting at it leaves the note before it standing', async ({ send, page, rows }) => {
    const [A, T0, B] = [note('A', at(2)), thought(0, at(3)), note('B', at(4))];
    send(A, T0);
    await page(T0);
    send(B);
    assert.deepEqual(rows(), ['A', 'T0', 'B']);
    await page(A, T0, B);
    assert.deepEqual(rows(), ['A', 'T0', 'B'], 'and the older page adopts it where it stands');
});

childTest('source order finer than a millisecond keeps every frame its own row', async ({ send, page, rows }) => {
    // The sequence transport recovery emits: its host note for a response already received, then that response's reasoning.
    const recovery = childRow(`🌐 ${WORDS.H}`, '2026-09-16T12:00:02.123100+00:00',
        { subagent_event: 'progress', narration: false, role: 'system', system_type: 'host_progress' });
    const reasoning = thought(0, '2026-09-16T12:00:02.123900+00:00');
    const B = note('B', at(4));
    assert.equal(Date.parse(recovery.ts), Date.parse(reasoning.ts), 'one parsed millisecond');
    send(recovery, reasoning, B);
    assert.deepEqual(rows(), ['H', 'T0', 'B'], 'live');
    for (const refresh of [1, 2]) {
        await page(recovery, reasoning, B);
        assert.deepEqual(rows(), ['H', 'T0', 'B'], `refresh ${refresh}`);
    }
    send(note('C', at(5)));
    assert.deepEqual(rows(), ['H', 'T0', 'B', 'C']);
});

childTest('a thought and the note after it in one millisecond, and two notes with one source time', async ({ send, page, rows }) => {
    const reasoning = thought(0, '2026-09-16T12:00:02.123100+00:00');
    const [C, C2] = [note('C', '2026-09-16T12:00:02.123900+00:00'), note('C2', '2026-09-16T12:00:02.123900+00:00')];
    const D = note('D', at(3));
    send(reasoning, C, C2);
    assert.deepEqual(rows(), ['T0', 'C2'], 'after the thought, one line replaced in place');
    await page(reasoning, saved(C, 0), saved(C2, 1));
    assert.deepEqual(rows(), ['T0', 'C', 'C2'], 'one source time, two saved rows: each keeps its own words');
    send(D);
    assert.deepEqual(rows(), ['T0', 'C', 'C2', 'D']);
});

childTest('a saved frame delivered live again finds the row it adopted', async ({ f, markup, send, page, rows }) => {
    const [A, T0, B] = [note('A', at(2)), thought(0, at(3)), note('B', at(4))];
    const lines = () => f.rows(CHILD).map((node) => node.dataset.liveLineKey);
    send(A, T0, B);
    rows();
    const before = lines();
    await page(A, T0, B);
    assert.deepEqual(lines(), before, 'adoption keeps each live line, its disclosure with it');
    send(A, T0, B);
    assert.deepEqual(rows(), ['A', 'T0', 'B'], 'the frames once more, after adoption');
    assert.ok(!markup().some((html) => />\d+x</.test(html)), 'and they do not count as repeats');
    send(note('C', at(5)));
    assert.deepEqual(rows(), ['A', 'T0', 'B', 'C']);
});

childTest('history may come first: the same frames live add no row', async ({ send, page, rows }) => {
    const [A, T0, B] = [note('A', at(2)), thought(0, at(3)), note('B', at(4))];
    await page(A, T0, B);
    assert.deepEqual(rows(), ['A', 'T0', 'B'], 'history first');
    send(A, T0, B);
    assert.deepEqual(rows(), ['A', 'T0', 'B'], 'then the same frames live');
    send(note('C', at(5)), note('C2', at(6)));
    assert.deepEqual(rows(), ['A', 'T0', 'B', 'C2'], 'and the next notes stand after them, replacing in their own stretch');
});

childTest('inside one stretch a note replaces the line before it, in any voice, and a thought ends the stretch', async ({ f, send, rows }) => {
    const { narration, ...legacy } = note('A', at(2));
    const line = () => f.rows(CHILD).map((node) => node.dataset.liveLineKey);
    send(legacy);
    assert.deepEqual(rows(), ['A'], 'a frame from before the voice fact');
    const first = line();
    send(note('B', at(3), { narration: false, role: 'system', system_type: 'host_progress' }));
    assert.deepEqual(rows(), ['B'], 'a host note');
    send(note('C', at(4)));
    assert.deepEqual(rows(), ['C'], 'spoken words');
    assert.deepEqual(line(), first, 'all in the same line');
    assert.equal(f.activity(CHILD), WORDS.C, 'and only spoken words take the child line');
    send(thought(0, at(5)));
    assert.deepEqual(rows(), ['C', 'T0'], 'a thought is its own row');
    send(note('D', at(6)));
    assert.deepEqual(rows(), ['C', 'T0', 'D'], 'the next note stands after it');
    send(note('A', at(7)));
    assert.deepEqual(rows(), ['C', 'T0', 'A'], 'and replaces within its own stretch');
});

childTest('a repeated thought is its own row and still divides the notes around it', async ({ send, page, rows }) => {
    const [A, T0, B, C, D] = [note('A', at(2)), thought(0, at(3)), note('B', at(4)), note('C', at(6)), note('D', at(8))];
    const [again, third, sameMillisecond] = [thought(0, at(5)), thought(0, at(7)), thought(0, `${at(7).slice(0, -1)}.000100Z`)];
    assert.equal(Date.parse(third.ts), Date.parse(sameMillisecond.ts));
    send(A, T0, B, again, C);
    assert.deepEqual(rows(), ['A', 'T0', 'B', 'T0', 'C'], 'live');
    await page(A, T0, B, again, C);
    assert.deepEqual(rows(), ['A', 'T0', 'B', 'T0', 'C'], 'after the refresh');
    send(third, sameMillisecond, D);
    assert.deepEqual(rows(), ['A', 'T0', 'B', 'T0', 'C', 'T0', 'T0', 'D'], 'saved first, then the same words live, twice inside one millisecond');
});

test('a refresh ends where a cold load starts, whatever the live card went through', async () => {
    const reasoning = thought(0, '2026-09-16T12:00:02.123900+00:00');
    const recovery = childRow(WORDS.H, '2026-09-16T12:00:02.123100+00:00',
        { subagent_event: 'progress', narration: false, role: 'system', system_type: 'host_progress' });
    const stories = {
        'note, thought, note': [note('A', at(2)), thought(0, at(3)), note('B', at(4))],
        'notes replacing one line': [note('C', at(5)), note('C2', at(6))],
        'replacements around thoughts': [note('A', at(2)), note('B', at(3)), thought(0, at(4)), note('C', at(5)), note('C2', at(6)), thought(1, at(7)), note('D', at(8))],
        'one millisecond': [recovery, reasoning, note('B', at(4))],
        'a repeated thought': [note('A', at(2)), thought(0, at(3)), note('B', at(4)), thought(0, at(5)), note('C', at(6))],
    };
    // One chat at a time: the fixture's DOM is global.
    const afterRefresh = async (frames, live) => {
        const { f, send, page, rows } = childTimeline();
        try {
            if (live) send(...frames);
            await page(...frames);
            return rows();
        } finally { f.close(); }
    };
    for (const [story, frames] of Object.entries(stories)) {
        const cold = await afterRefresh(frames, false);
        assert.deepEqual(cold, frames.map((frame) => nameOf(frame.content)), `${story}: a cold load shows every saved frame`);
        assert.deepEqual(await afterRefresh(frames, true), cold, `${story}: live, then refreshed`);
    }
});
