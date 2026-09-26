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
        instance, messages, card,
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
        assert.equal(f.title(), 'Working...', 'the running placeholder, not the last host note');
    } finally { f.close(); }
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
        assert.match(f.card().querySelector('[data-live-meta]').innerHTML, /Last solve response: actual-solver/);
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
