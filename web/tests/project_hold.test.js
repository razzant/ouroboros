import test from 'node:test';
import assert from 'node:assert/strict';
import { computeHydratedDirectActivities } from '../modules/chat_activity.js';
import { summarizeRoom } from '../modules/project_activity.js';
import { censusTaskPhase, desiredLiveCardPhase, setHistoricalUnavailable, setLiveCardPhase, syncParkedPhase } from '../modules/task_phase_chip.js';
import { emptyWorkState, roomStatusView, workRowView } from '../modules/work_status.js';
import { readFileSync } from 'node:fs';
import { ElementStub, TextStub } from './chat_dom_fixture.js';
import { projection, room, work } from './work_fixture.js';

// A chip as the page has it: the shared status mark first, its words one text node beside it.
const chipDoc = { createElement: (tag) => new ElementStub(tag, chipDoc), createTextNode: (data) => new TextStub(data) };
const markOf = (chip) => {
    const mark = chip.firstElementChild;
    return mark && !mark.hidden ? `${mark.dataset.shape}/${mark.dataset.hue}` : 'none';
};

const hold = { label: 'Waiting for Project verification', reason: 'project_routing_fence_lookup_failed', detail: 'Authority is unreadable.' };
const held = { activity_id: 'same-id', chat_id: 7, kind: 'managed_task', phase: 'queued', project_admission_hold: hold };
// The server classifies a Project or task-scope hold as a wait whose reason is the hold's own label.
const heldWork = (row, label = hold.label) => work(row, { activity: 'waiting',
    because: [{ code: 'project_admission', text: label }] });
const inRoom = (activity, extra = {}) => ({ ...emptyWorkState(), room: room(activity, extra) });

test('a known non-Project wait keeps its scope label, on its row: the header says Waiting', () => {
    const scopeHold = { ...hold, label: 'Waiting for task scope verification' };
    const row = { ...held, chat_id: 1, project_admission_hold: scopeHold };
    assert.equal(roomStatusView(inRoom('waiting')).text, 'Waiting');
    assert.deepEqual([workRowView(heldWork(row, scopeHold.label)).word, workRowView(heldWork(row, scopeHold.label)).secondary],
        ['Waiting', scopeHold.label]);
});

test('a held task is stationary in Chat and the sidebar, its label beside the word', () => {
    assert.deepEqual(roomStatusView(inRoom('waiting')),
        { kind: 'online', text: 'Waiting', tone: 'held', mark: 'still', owner: false, amber: '' });
    const sidebar = summarizeRoom({ ...room('waiting'), works: ['same-id'] }, [heldWork(held)]);
    assert.deepEqual(sidebar, { mark: 'still', owner: false, word: 'Waiting', note: '', label: `Waiting · ${hold.label}` });
});

for (const [activity, extra, text, kind, mark, tone] of [
    ['working', { transition: 'pausing' }, 'Pausing…', 'thinking', 'moving', 'held'],
    ['paused', {}, 'Paused', 'online', 'pause', 'held'],
    ['unknown', {}, 'Activity unconfirmed', 'online', 'ring', 'unknown'],
    ['waiting', {}, 'Waiting', 'online', 'still', 'held'],
]) {
    test(`a held task's room says ${text}; a card's hold never sets the header`, () => {
        const view = roomStatusView(inRoom(activity, extra));
        assert.deepEqual([view.kind, view.text, view.tone, view.mark], [kind, text, tone, mark]);
        // Same-id recovery: the server's room works again.
        assert.equal(roomStatusView(inRoom('working')).text, 'Working');
    });
}

test('same-ID recovery clears the hold on its row; independent work and a pause remain truthful', () => {
    const activities = computeHydratedDirectActivities(new Map(), [held], 7);
    const recovered = computeHydratedDirectActivities(activities, [{ ...held, phase: 'working', project_admission_hold: undefined }], 7);
    assert.equal(recovered.get('same-id').project_admission_hold, undefined);
    assert.equal(roomStatusView(inRoom('working')).text, 'Working');
    const sibling = { ...held, activity_id: 'sibling', phase: 'working', project_admission_hold: undefined };
    const mixed = summarizeRoom({ ...room('working'), works: ['sibling', 'same-id'] }, [work(sibling), heldWork(held)]);
    assert.equal(mixed.mark, 'moving');
    // One room word: a budget pause, the owner's Pause and a Restart hold all read Paused; the row says which.
    assert.equal(roomStatusView(inRoom('paused')).text, 'Paused');
});

test('the census phase (a Main portal\'s chip source) keeps a budget pause beside the Project wait, as the sidebar does', () => {
    const paused = { ...held, phase: 'budget_paused' };
    // The chip says the pause's word; the Project wait is the line beside it.
    assert.deepEqual([censusTaskPhase(paused, null).text, censusTaskPhase(paused, null).secondary], ['Paused', hold.label]);
    assert.equal(censusTaskPhase(paused, null).mark, 'pause');
    const pausing = censusTaskPhase({ ...held, phase: 'budget_pausing' }, null);
    assert.deepEqual([pausing.text, pausing.secondary], ['Pausing…', hold.label]);
    assert.equal(pausing.mark, 'moving');
    assert.equal(censusTaskPhase({ ...paused, finishing_reviews: true }, null).secondary, `${hold.label} · Review work finishing`);
    assert.equal(censusTaskPhase(held, null).text, hold.label);
    assert.deepEqual([censusTaskPhase(held, null).mark, censusTaskPhase(held, null).tone], ['still', 'neutral']);
    assert.equal(censusTaskPhase(paused, { status: 'cancelled' }).text, 'Stopped');
});

test('a held card with retained progress waits statically; Stop, terminal and same-ID recovery outrank it', () => {
    const card = { finished: false, isSubagent: false, root: { dataset: {} },
        phaseEl: new ElementStub('span', chipDoc),
        phaseSecondaryEl: { hidden: true, textContent: '', isConnected: true } };
    setLiveCardPhase(card, 'working', 'Working', 'chat-live-phase working');  // replayed progress
    assert.equal(markOf(card.phaseEl), 'moving/ink');
    assert.equal(setHistoricalUnavailable(card, false, hold.label), true);  // census restore
    assert.equal(card.phaseEl.textContent, hold.label);
    assert.equal(card.phaseEl.className, 'chat-live-phase warn');
    // A hold goes on by itself: neutral and still, never the owner's amber, nothing that moves.
    assert.deepEqual([markOf(card.phaseEl), card.phaseEl.dataset.chipTone], ['still/ink', 'neutral']);
    assert.equal(card.phaseEl.dataset.phase, 'working');  // still unfinished, never a terminal phase
    assert.equal(setHistoricalUnavailable(card, false), false);  // no census fact: the hold stays
    assert.equal(desiredLiveCardPhase({ ...card, cancelPendingPolicy: 'immediate' }).text, 'Cancelling…');
    assert.equal(desiredLiveCardPhase({ ...card, finished: true }, 'error').phase, 'error');
    assert.equal(setHistoricalUnavailable(card, false, ''), true);  // same-ID recovery
    assert.equal(card.phaseEl.textContent, 'Working');
    assert.deepEqual([markOf(card.phaseEl), card.phaseEl.dataset.chipTone], ['moving/ink', 'live']);
    const chat = readFileSync(new URL('../modules/chat.js', import.meta.url), 'utf8');
    assert.match(chat, /restoreCardActivity\(liveCardRecords\.get\(k\), v\.project_admission_hold\)/);
    assert.match(chat, /function restoreCardActivity\(record, held = \{\}\) \{\r?\n\s+if \(!setHistoricalUnavailable\(record, false, held\)\)/);
});


for (const [parkedPhase, phase, label, mark] of [
    ['budget_paused', 'paused', 'Paused', 'pause/ink'], ['unknown', 'unknown', 'Activity unconfirmed', 'ring/ink'],
]) {
    for (const releaseFirst of ['Project', 'parked']) {
        test(`Project hold and ${parkedPhase} recover independently (${releaseFirst} first)`, () => {
            const card = { finished: false, root: { dataset: {} }, phaseEl: new ElementStub('span', chipDoc) };
            setLiveCardPhase(card, 'working', 'Working', 'chat-live-phase working');
            assert.equal(markOf(card.phaseEl), 'moving/ink');
            syncParkedPhase(card, parkedPhase);
            setHistoricalUnavailable(card, false, hold);
            assert.equal(card.parkedPhase, parkedPhase);
            assert.equal(card.projectHold, hold.label);
            assert.equal(card.projectHoldDetail, hold.detail);
            assert.equal(card.phaseEl.dataset.phase, phase);
            assert.ok(card.phaseEl.textContent.includes(label), 'the parked fact remains visible');
            assert.equal(markOf(card.phaseEl), mark);

            if (releaseFirst === 'Project') {
                setHistoricalUnavailable(card, false, {});
                assert.equal(card.projectHold, '');
                assert.equal(card.projectHoldDetail, '');
                assert.equal(card.parkedPhase, parkedPhase, 'Project recovery cannot release another hold');
                assert.equal(card.phaseEl.dataset.phase, phase);
                assert.equal(card.phaseEl.textContent, label);
                assert.equal(markOf(card.phaseEl), mark, 'the remaining parked phase keeps its own mark');
                syncParkedPhase(card, 'working');
            } else {
                syncParkedPhase(card, 'working');
                assert.equal(card.parkedPhase, '');
                assert.equal(card.projectHold, hold.label, 'a positive task phase cannot clear the Project hold');
                assert.equal(card.phaseEl.textContent, hold.label);
                assert.equal(card.phaseEl.className, 'chat-live-phase warn');
                assert.equal(markOf(card.phaseEl), 'still/ink', 'the remaining Project hold stays still');
                setHistoricalUnavailable(card, false, {});
            }
            assert.equal(card.parkedPhase, '');
            assert.equal(card.projectHold, '');
            assert.equal(card.phaseEl.dataset.phase, 'working');
            assert.equal(card.phaseEl.textContent, 'Working');
            assert.equal(card.phaseEl.className, 'chat-live-phase working');
            assert.equal(markOf(card.phaseEl), 'moving/ink', 'clearing both causes restores real activity');
        });
    }
}

test('nested wait keeps the host cause as text and clears it only on a recovery fact', () => {
    const card = { isSubagent: true };
    setHistoricalUnavailable(card, false, hold);
    assert.equal(card.projectHoldDetail, hold.detail);
    assert.equal(card.projectHold, hold.label);
    setHistoricalUnavailable(card, false);
    assert.equal(card.projectHoldDetail, hold.detail);
    const unconfirmed = { label: 'Waiting for previous run verification', reason: 'project_dispatch_unconfirmed',
        detail: 'The previous run cannot be confirmed; automatic recovery is not authorized.' };
    setHistoricalUnavailable(card, false, unconfirmed);
    assert.equal(desiredLiveCardPhase(card).text, unconfirmed.label);
    assert.equal(card.projectHoldDetail, unconfirmed.detail);
    setHistoricalUnavailable(card, false, {});
    assert.equal(card.projectHoldDetail, '');
    assert.equal(desiredLiveCardPhase(card).text, 'Working');
});

test('held children never vote on the room: the server room speaks whatever the cards hold', () => {
    const child = { groupId: 'child', isSubagent: true, root: { isConnected: true }, projectHold: hold.label };
    const state = (activity) => ({ ...inRoom(activity), works: [] });
    assert.equal(roomStatusView(state('waiting')).text, 'Waiting');
    assert.equal(roomStatusView(state('working')).text, 'Working', 'a held child under working work');
    assert.equal(child.projectHold, hold.label, 'its hold stays its own card\'s fact');
});

test('actual child queue hydration refreshes room status once and reconciles departure without polling reviews', async () => {
    const { createChatInstance } = await import('../modules/chat.js');
    const { installDom, restoreDom, walkCard } = await import('./chat_dom_fixture.js');
    const settle = () => new Promise(resolve => setImmediate(resolve));
    let queue = { pending: [], running: [{ id: 'child', task: { project_admission_hold: {} } }] };
    let connected = true;
    let detailStatus = 'running';
    const reads = [];
    const { prior, mount } = installDom(async url => {
        url = String(url);
        if (url === '/api/tasks?queue_only=1') return { ok: true, json: async () => ({ queue }) };
        if (url.startsWith('/api/tasks/')) {
            reads.push(url);
            return { ok: true, json: async () => ({ task_id: url.split('/').at(-1), status: detailStatus }) };
        }
        return { ok: true, json: async () => url.startsWith('/api/chat/history')
            ? { messages: [] } : { active_direct_turns: [] } };
    });
    const handlers = new Map();
    const ws = { on(type, fn) { handlers.set(type, fn); return () => handlers.delete(type); },
        isConnected: () => connected, send() {} };
    let instance;
    try {
        instance = createChatInstance({ ws, state: { activePage: 'chat', projectChatIds: new Set(), unreadCount: 0 },
            updateUnreadBadge() {}, stateSnapshots: { gate: async () => ({}), isCurrent: () => true, apply() {} },
            chatId: 7, idPrefix: 'chat', mountEl: mount, asPanel: true });
        await instance.refreshHistory({ revision: 1 });
        const frame = (id, fields = {}) => handlers.get('chat')({ chat_id: 7, role: 'assistant', is_progress: true,
            task_id: id, content: 'Inspecting sources', ts: '2026-09-30T10:00:00Z', ...fields });
        frame('same-id');
        const childFrame = id => frame(id, { delegation_role: 'subagent', subagent_event: 'scheduled',
            subagent_task_id: id, parent_task_id: 'same-id', root_task_id: 'same-id', subagent_role: 'reader' });
        childFrame('child');
        // Work membership names all queued and running children, unlike the root census.
        const members = () => ['same-id', ...[...queue.pending, ...queue.running].map(row => row.id)];
        const snapshot = () => instance.hydrateStateSnapshot({ active_chat_activities: [held],
            active_chat_activities_complete: true, supervisor_ready: true,
            ...projection(7, [{ ...heldWork(held), members: { ids: members() } }], room('waiting')) });
        const status = () => globalThis.document.byId.get('chat-status').textContent;
        snapshot(); await settle();
        assert.equal(status(), 'Waiting', 'the room says what the server says');
        queue.pending = queue.running.splice(0);
        queue.pending[0].task.project_admission_hold = hold;
        snapshot();
        assert.equal(status(), 'Waiting', 'the snapshot finishes before its queue read');
        await settle();
        assert.equal(status(), 'Waiting', 'the child\'s queue read paints its card, never the room\'s header');
        const messages = globalThis.document.byId.get('chat-messages');
        const child = walkCard(messages, 'child');
        assert.equal(child.querySelector('[data-live-phase]').textContent, hold.label);
        connected = false; handlers.get('close')();
        snapshot(); await settle();
        assert.equal(child.querySelector('[data-live-phase]').textContent, 'Activity unconfirmed');
        assert.equal(markOf(child.querySelector('[data-live-phase]')), 'ring/ink');
        connected = true;
        snapshot(); await settle();
        assert.equal(child.querySelector('[data-live-phase]').textContent, hold.label,
            'a fresh child queue read restores its own hold after reconnect');
        assert.equal(markOf(child.querySelector('[data-live-phase]')), 'still/ink');
        queue.pending[0].task.project_admission_hold = {};
        snapshot(); await settle();
        assert.equal(child.querySelector('[data-live-phase]').textContent, 'Queued');
        assert.deepEqual([markOf(child.querySelector('[data-live-phase]')),
            child.querySelector('[data-live-phase]').dataset.chipTone], ['still/ink', 'neutral']);
        queue.pending[0].task.project_admission_hold = hold;
        snapshot(); await settle();
        frame('working-sibling');
        assert.equal(status(), 'Waiting', 'a card the projection does not list never sets the header');
        handlers.get('log')({ chat_id: 7, data: { type: 'task_done', task_id: 'working-sibling', status: 'cancelled' } });

        childFrame('review-child');
        queue.running.push({ id: 'review-child', task: { project_admission_hold: {} } });
        handlers.get('chat')({ chat_id: 7, role: 'system', system_type: 'skill_review', task_id: 'review-child',
            review_group: { surface: 'skill', id: 'task:review-child:alpha', presentation_owner_task_id: 'review-child',
                skill: 'alpha', status: 'clean', attempts: [{ job_id: 'job-child', skill: 'alpha', status: 'clean' }] } });
        for (let pass = 0; pass < 3; pass++) { snapshot(); await settle(); }
        assert.equal(reads.filter(url => url === '/api/tasks/review-child').length, 0,
            'an ordinary child review never joins root-census absence reconciliation');

        queue.pending = [];
        snapshot(); await settle();
        assert.equal(reads.filter(url => url === '/api/tasks/child').length, 1, 'held queue departure reads detail once');
        for (let pass = 0; pass < 3; pass++) { snapshot(); await settle(); }
        assert.equal(reads.filter(url => url === '/api/tasks/child').length, 4,
            'later complete bodies retry an absent nonterminal child without inventing hold recovery');
        assert.equal(child.querySelector('[data-live-phase]').textContent, hold.label);
        queue.pending = [{ id: 'child', task: { project_admission_hold: hold } }];
        snapshot(); await settle();
        queue.pending = [];
        detailStatus = 'cancelled';
        snapshot(); await settle();
        assert.equal(reads.filter(url => url === '/api/tasks/child').length, 5, 'a new queue departure is a new observation');
        assert.equal(child.dataset.finished, '1', 'terminal detail heals a missing terminal frame');
        for (let pass = 0; pass < 3; pass++) { snapshot(); await settle(); }
        assert.equal(reads.filter(url => url === '/api/tasks/child').length, 5);
    } finally { instance?.destroy(); restoreDom(prior); }
});
