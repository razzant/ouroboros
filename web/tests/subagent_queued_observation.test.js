// A scheduler-accepted child shows its queue place at once. The host's own `accepted` fact on the
// live `scheduled` frame is the child card's current Queued state — the census's own word, before
// the census holds the row — while a requested/legacy row without the fact, a replayed history row
// and a late or duplicate admission prove nothing current and downgrade nothing. A refused read
// stays unconfirmed until a positive fact speaks; an explicit working fact replaces the gap the
// admission filled (docs/DESIGN.md "Status marks": still = queued; the words already exist).
import assert from 'node:assert/strict';
import test, { after } from 'node:test';
import { createChatInstance } from '../modules/chat.js';
import { summarizeChatLiveEvent } from '../modules/log_events.js';
import { desiredLiveCardPhase, frameObservesPhase, observeChildPhase } from '../modules/task_phase_chip.js';
import { ElementStub, installDom, restoreDom, walkCard } from './chat_dom_fixture.js';
import { projection, room, work } from './work_fixture.js';

const TS = '2026-11-18T12:00:00Z';
const TASK = 'root-1';
const KID = 'kid-1';
const lineage = { delegation_role: 'subagent', parent_task_id: TASK, root_task_id: TASK, subagent_role: 'scout' };
// The scheduler's real admission frame (supervisor/events_schedule_task.py): the child's own task
// id, system_type task_scheduled, and the host's progress_meta facts flattened onto the frame.
const admission = (extra = {}) => ({
    type: 'send_message', is_progress: true, role: 'system', system_type: 'task_scheduled',
    chat_id: 1, task_id: KID, ts: TS, content: '🗓️ Scheduled subagent kid-1 (scout): reads the registry',
    subagent_event: 'scheduled', subagent_task_id: KID, accepted: true, ...lineage, ...extra,
});

// ---- the projection: one host fact decides between a bare start and the current queued state ----

test('an accepted admission projects queued; a bare scheduled row stays a start', () => {
    assert.equal(summarizeChatLiveEvent(admission()).phase, 'queued');
    const requested = admission();
    delete requested.accepted;
    assert.equal(summarizeChatLiveEvent(requested).phase, 'start', 'a request proves nothing current');
    assert.equal(summarizeChatLiveEvent(admission({ accepted: false })).phase, 'start');
    assert.equal(summarizeChatLiveEvent(admission({ accepted: 'true' })).phase, 'start', 'the fact is the boolean');
    assert.equal(summarizeChatLiveEvent({ ...admission(), subagent_event: 'running' }).phase, 'working');
});

// ---- the gate: what a live frame may observe, and what it can never downgrade ----

test('frameObservesPhase keeps the working/thinking gate and admits queued only into a gap', () => {
    const summary = (phase, extra = {}) => ({ phase, ...extra });
    // the two established live phases keep their exact old gate semantics
    assert.equal(frameObservesPhase({ isSubagent: true }, summary('working'), true), true);
    assert.equal(frameObservesPhase({}, summary('thinking'), true), true);
    assert.equal(frameObservesPhase({ isSubagent: true }, summary('working', { activity: {} }), true), false,
        'a delegated activity frame never doubles as a phase fact');
    assert.equal(frameObservesPhase({}, summary('working'), false), false, 'a replayed row is not current');
    // the admission's gap cases: a card with no fact yet, or only a refused read's unknown
    assert.equal(frameObservesPhase({ isSubagent: true }, summary('queued'), true), true);
    assert.equal(frameObservesPhase({ isSubagent: true, censusPhase: 'unknown', parkedPhase: 'unknown' },
        summary('queued'), true), true);
    assert.equal(frameObservesPhase({ isSubagent: true, workView: { phase: 'unknown' } }, summary('queued'), true), true);
    // never a downgrade of a phase the card already knows
    for (const known of [{ censusPhase: 'working' }, { censusPhase: 'queued' }, { censusPhase: 'budget_paused' },
        { workView: { phase: 'working' } }, { finished: true }]) {
        assert.equal(frameObservesPhase({ isSubagent: true, ...known }, summary('queued'), true), false,
            JSON.stringify(known));
    }
    assert.equal(frameObservesPhase({ isSubagent: true }, summary('queued'), false), false);
    assert.equal(frameObservesPhase({}, summary('queued'), true), false,
        'a root-shaped card keeps the exact old gate: only working/thinking');
});

test('the observed queued fact paints the chosen chip, and an explicit working fact replaces it', () => {
    const record = { isSubagent: true };
    assert.equal(observeChildPhase(record, 'queued'), true);
    assert.deepEqual(['phase', 'text', 'mark'].map((key) => desiredLiveCardPhase(record)[key]),
        ['queued', 'Queued', 'still']);
    assert.equal(observeChildPhase(record, 'working'), true);
    assert.deepEqual(['phase', 'text', 'mark'].map((key) => desiredLiveCardPhase(record)[key]),
        ['working', 'Working', 'moving']);
    assert.equal(frameObservesPhase(record, { phase: 'queued' }, true), false, 'the late admission is refused');
    assert.equal(desiredLiveCardPhase(record).text, 'Working');
});

// ---- the card: the live frame, a stale parent-only census, a late admission, a finished card ----

// The flat fixture's querySelector does not descend; card internals need a real descendant lookup
// (chat_activity_block.test.js's proven patch).
const originalQuery = ElementStub.prototype.querySelector;
after(() => { ElementStub.prototype.querySelector = originalQuery; });
ElementStub.prototype.querySelector = function (selector) {
    const direct = originalQuery.call(this, selector);
    if (direct) return direct;
    for (const child of this.children) { const found = child.querySelector(selector); if (found) return found; }
    return null;
};

function fixture(history = [], chatId = 1) {
    const env = installDom(async (url) => ({ ok: true, json: async () => (String(url).startsWith('/api/chat/history')
        ? { messages: history, window: { complete: true } }
        : { active_direct_turns: [] }) }));
    const handlers = new Map();
    let generation = 0;
    const instance = createChatInstance({
        ws: { on(type, fn) { handlers.set(type, fn); return () => handlers.delete(type); },
            isConnected: () => true, send() {} },
        state: { activePage: 'chat', projectChatIds: new Set(), unreadCount: 0 },
        updateUnreadBadge() {}, chatId, idPrefix: 'chat', mountEl: env.mount,
        stateSnapshots: { begin: () => ({ generation: ++generation, requestedAt: Date.now() }), gate() { return Promise.resolve(this.begin()); },
            isCurrent: () => true, apply() {} },
    });
    const messages = document.byId.get('chat-messages');
    const managed = [{ activity_id: TASK, chat_id: chatId, kind: 'managed_task', phase: 'working' }];
    return {
        instance, messages,
        chip: (id = KID) => walkCard(messages, id)?.querySelector('[data-live-phase]')?.textContent,
        emit: (row) => handlers.get('chat')({ chat_id: chatId, ts: TS, ...row }),
        // the earlier snapshot: the parent alone, as the queue was before the admission
        census: (rows = managed, requestedAt = Date.now()) => instance.hydrateStateSnapshot({
            active_chat_activities: rows, active_chat_activities_complete: true, supervisor_ready: true,
            ...projection(chatId, rows.map((row) => work(row)), room('working', { call: 'working' })),
        }, requestedAt, ++generation),
        close() { instance.destroy(); restoreDom(env.prior); },
    };
}

test('a live accepted admission shows the child Queued at once; a stale parent-only census cannot downgrade it', () => {
    const f = fixture();
    try {
        const started = Date.now() - 1;
        f.census();  // the earlier read: only the parent
        f.emit({ task_id: TASK, role: 'assistant', is_progress: true, content: 'Planning the swarm.' });
        f.emit({ ...admission(), ts: '2026-11-18T12:00:01Z' });
        assert.equal(f.chip(), 'Queued');
        f.census(undefined, started);  // this read began before the admission and did not list the child
        assert.equal(f.chip(), 'Queued', 'an older incomplete absence is not a new fact');
    } finally { f.close(); }
});

test('without the host acceptance the honest gap stays: Activity unconfirmed', () => {
    const f = fixture();
    try {
        f.census();
        const requested = admission();
        delete requested.accepted;
        f.emit({ ...requested, ts: '2026-11-18T12:00:01Z' });
        assert.equal(f.chip(), 'Activity unconfirmed');
    } finally { f.close(); }
});

test('an explicit working fact replaces the admission; a late admission downgrades nothing, a finished card never reopens', () => {
    const f = fixture();
    try {
        f.census();
        f.emit({ ...admission(), ts: '2026-11-18T12:00:01Z' });
        assert.equal(f.chip(), 'Queued');
        f.emit({ ...lineage, task_id: KID, role: 'system', system_type: 'host_progress', narration: false,
            is_progress: true, content: 'Reading the registry.', subagent_event: 'running',
            subagent_task_id: KID, ts: '2026-11-18T12:00:02Z' });
        assert.equal(f.chip(), 'Working');
        f.emit({ ...admission(), ts: '2026-11-18T12:00:01Z' });
        assert.equal(f.chip(), 'Working', 'the late admission is history, not a phase');
        f.emit({ ...lineage, task_id: KID, role: 'assistant', is_progress: true, subagent_event: 'completed',
            status: 'completed', subagent_task_id: KID, content: 'Done.', result: 'Done.',
            ts: '2026-11-18T12:00:03Z' });
        assert.equal(f.chip(), 'Done');
        f.emit({ ...admission(), ts: '2026-11-18T12:00:04Z' });
        assert.equal(f.chip(), 'Done');
    } finally { f.close(); }
});

test('a replayed admission is history: it paints no current queued fact', async () => {
    const f = fixture([{ role: 'system', system_type: 'task_scheduled', is_progress: true,
        text: '🗓️ Scheduled subagent kid-1 (scout): reads the registry',
        content: '🗓️ Scheduled subagent kid-1 (scout): reads the registry', task_id: KID,
        ts: '2026-11-18T12:00:01Z', chat_id: 1, subagent_event: 'scheduled', subagent_task_id: KID,
        parent_task_id: TASK, root_task_id: TASK, delegation_role: 'subagent', subagent_role: 'scout',
        accepted: true }]);
    try {
        await f.instance.refreshHistory({ revision: 1 });
        assert.equal(f.chip(), 'Activity unconfirmed', 'a saved row proves no current state');
    } finally { f.close(); }
});
