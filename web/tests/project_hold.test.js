import test from 'node:test';
import assert from 'node:assert/strict';
import { computeHydratedDirectActivities, chatStatusCounts, computeDerivedChatStatus } from '../modules/chat_activity.js';
import { summarizeProjectActivities } from '../modules/project_activity.js';
import { handoffPhase } from '../modules/project_handoff.js';
import { desiredLiveCardPhase, setHistoricalUnavailable, setLiveCardPhase, setLiveCardTypingVisible, syncParkedPhase } from '../modules/task_phase_chip.js';
import { readFileSync } from 'node:fs';

const hold = { label: 'Waiting for Project verification', reason: 'project_routing_fence_lookup_failed', detail: 'Authority is unreadable.' };
const held = { activity_id: 'same-id', chat_id: 7, kind: 'managed_task', phase: 'queued', project_admission_hold: hold };

test('known non-Project wait keeps the scope label in Chat and Main handoff', () => {
    const scopeHold = { ...hold, label: 'Waiting for task scope verification' };
    const row = { ...held, chat_id: 1, project_admission_hold: scopeHold };
    const activities = computeHydratedDirectActivities(new Map(), [row], 1);
    assert.equal(computeDerivedChatStatus(chatStatusCounts(activities, [])).text, scopeHold.label);
    assert.equal(handoffPhase(row, null).text, scopeHold.label);
});

test('held task is stationary across hydrated Chat, Project and Main receipt', () => {
    const activities = computeHydratedDirectActivities(new Map(), [held], 7);
    const card = { root: { isConnected: true }, groupId: 'same-id', finished: false };
    const status = computeDerivedChatStatus(chatStatusCounts(activities, [card]));
    assert.deepEqual(status, { kind: 'online', text: hold.label, showDots: false });
    assert.equal(summarizeProjectActivities([{ ...held, required_question_unavailable: true }]).label, hold.label);
    assert.equal(summarizeProjectActivities([held]).motion, false);
    assert.equal(handoffPhase(held, null).text, hold.label);
    assert.equal(handoffPhase(held, null).motion, false);
    assert.equal(handoffPhase(held, { status: 'failed' }).text, 'Failed');
    assert.equal(handoffPhase(held, null, false).text, 'Activity unconfirmed');
});

for (const [phase, text, kind, showDots] of [
    ['budget_pausing', 'Pausing…', 'thinking', true],
    ['budget_paused', 'Paused', 'online', false],
    ['unknown', 'Activity unconfirmed', 'online', false],
    ['queued', hold.label, 'online', false],
]) {
    test(`Project hold preserves the ${phase} Chat header`, () => {
        const row = { ...held, phase };
        const activities = computeHydratedDirectActivities(new Map(), [row], 7);
        const card = { root: { isConnected: true }, groupId: 'same-id', finished: false,
            parkedPhase: phase === 'queued' ? '' : phase, projectHold: hold.label };
        assert.deepEqual(computeDerivedChatStatus(chatStatusCounts(activities, [card])),
            { kind, text, showDots });
        const resumed = computeHydratedDirectActivities(activities,
            [{ ...row, phase: 'working', project_admission_hold: undefined }], 7);
        card.parkedPhase = '';
        card.projectHold = '';
        assert.equal(computeDerivedChatStatus(chatStatusCounts(resumed, [card])).text, 'Working...');
    });
}

test('same-ID recovery clears the hold; independent work and budget remain truthful', () => {
    const activities = computeHydratedDirectActivities(new Map(), [held], 7);
    const recovered = computeHydratedDirectActivities(activities, [{ ...held, phase: 'working', project_admission_hold: undefined }], 7);
    assert.equal(recovered.get('same-id').project_admission_hold, undefined);
    assert.equal(computeDerivedChatStatus(chatStatusCounts(recovered, [])).text, 'Working...');
    const sibling = { ...held, activity_id: 'sibling', phase: 'working', project_admission_hold: undefined };
    assert.equal(summarizeProjectActivities([held, sibling]).motion, true);
    assert.match(summarizeProjectActivities([held, sibling]).label, /Waiting for Project/);
    const paused = computeHydratedDirectActivities(new Map(), [{ ...held, phase: 'budget_paused' }], 7);
    // Batch4: one census phase covers budget pause, owner Pause and Restart hold, so no cause is claimed.
    assert.equal(computeDerivedChatStatus(chatStatusCounts(paused, [])).text, 'Paused');
});

test('Main handoff keeps a budget pause beside the Project wait, as the sidebar does', () => {
    const paused = { ...held, phase: 'budget_paused' };
    assert.equal(handoffPhase(paused, null).text, `Paused · ${hold.label}`);
    assert.equal(handoffPhase(paused, null).motion, false);
    assert.equal(handoffPhase(paused, null).text, summarizeProjectActivities([paused]).label);
    assert.equal(handoffPhase({ ...held, phase: 'budget_pausing' }, null).text, `Pausing… · ${hold.label}`);
    assert.equal(handoffPhase({ ...held, phase: 'budget_pausing' }, null).motion, false);
    assert.equal(handoffPhase(held, null).text, hold.label);
    assert.equal(handoffPhase(paused, { status: 'cancelled' }).text, 'Cancelled');
});

test('a held card with retained progress waits statically; Stop, terminal and same-ID recovery outrank it', () => {
    const card = { finished: false, isSubagent: false, root: { dataset: {} },
        phaseEl: { hidden: false, dataset: {}, attrs: {}, textContent: '', className: '',
            getAttribute(key) { return this.attrs[key]; }, setAttribute(key, value) { this.attrs[key] = value; } },
        phaseSecondaryEl: { hidden: true, textContent: '', isConnected: true },
        inlineTypingEl: { style: { display: '' }, isConnected: true } };
    setLiveCardPhase(card, 'working', 'Working', 'chat-live-phase working');  // replayed progress
    assert.equal(card.inlineTypingEl.style.display, '');
    assert.equal(setHistoricalUnavailable(card, false, hold.label), true);  // census restore
    assert.equal(card.phaseEl.textContent, hold.label);
    assert.equal(card.phaseEl.className, 'chat-live-phase warn');  // static amber, no pulse class
    assert.equal(card.phaseEl.dataset.phase, 'working');  // still unfinished, never a terminal phase
    assert.equal(card.inlineTypingEl.style.display, 'none');
    setLiveCardTypingVisible(card, true);  // a later typing writer cannot animate the wait
    assert.equal(card.inlineTypingEl.style.display, 'none');
    assert.equal(setHistoricalUnavailable(card, false), false);  // no census fact: the hold stays
    assert.equal(desiredLiveCardPhase({ ...card, cancelPendingPolicy: 'immediate' }).text, 'Cancelling…');
    assert.equal(desiredLiveCardPhase({ ...card, finished: true }, 'error').phase, 'error');
    assert.equal(setHistoricalUnavailable(card, false, ''), true);  // same-ID recovery
    assert.equal(card.phaseEl.textContent, 'Working');
    assert.equal(card.inlineTypingEl.style.display, '');
    const chat = readFileSync(new URL('../modules/chat.js', import.meta.url), 'utf8');
    assert.match(chat, /restoreCardActivity\(liveCardRecords\.get\(k\), v\.project_admission_hold\)/);
    assert.match(chat, /function restoreCardActivity\(record, held = \{\}\) \{\r?\n\s+if \(!setHistoricalUnavailable\(record, false, held\)\)/);
});


for (const [parkedPhase, phase, label] of [
    ['budget_paused', 'paused', 'Paused'], ['unknown', 'unknown', 'Activity unconfirmed'],
]) {
    for (const releaseFirst of ['Project', 'parked']) {
        test(`Project hold and ${parkedPhase} recover independently (${releaseFirst} first)`, () => {
            const attrs = new Map();
            const card = { finished: false, root: { dataset: {} },
                phaseEl: { hidden: false, dataset: {}, textContent: '', className: '',
                    getAttribute: key => attrs.get(key), setAttribute: (key, value) => attrs.set(key, value) },
                inlineTypingEl: { style: { display: '' }, isConnected: true } };
            setLiveCardPhase(card, 'working', 'Working', 'chat-live-phase working');
            assert.equal(card.inlineTypingEl.style.display, '');
            syncParkedPhase(card, parkedPhase);
            setHistoricalUnavailable(card, false, hold);
            assert.equal(card.parkedPhase, parkedPhase);
            assert.equal(card.projectHold, hold.label);
            assert.equal(card.projectHoldDetail, hold.detail);
            assert.equal(card.phaseEl.dataset.phase, phase);
            assert.ok(card.phaseEl.textContent.includes(label), 'the parked fact remains visible');
            setLiveCardTypingVisible(card, true);
            assert.equal(card.inlineTypingEl.style.display, 'none');

            if (releaseFirst === 'Project') {
                setHistoricalUnavailable(card, false, {});
                assert.equal(card.projectHold, '');
                assert.equal(card.projectHoldDetail, '');
                assert.equal(card.parkedPhase, parkedPhase, 'Project recovery cannot release another hold');
                assert.equal(card.phaseEl.dataset.phase, phase);
                assert.equal(card.phaseEl.textContent, label);
                setLiveCardTypingVisible(card, true);
                assert.equal(card.inlineTypingEl.style.display, 'none', 'the remaining parked phase stays still');
                syncParkedPhase(card, 'working');
            } else {
                syncParkedPhase(card, 'working');
                assert.equal(card.parkedPhase, '');
                assert.equal(card.projectHold, hold.label, 'a positive task phase cannot clear the Project hold');
                assert.equal(card.phaseEl.textContent, hold.label);
                assert.equal(card.phaseEl.className, 'chat-live-phase warn');
                setLiveCardTypingVisible(card, true);
                assert.equal(card.inlineTypingEl.style.display, 'none', 'the remaining Project hold stays still');
                setHistoricalUnavailable(card, false, {});
            }
            assert.equal(card.parkedPhase, '');
            assert.equal(card.projectHold, '');
            assert.equal(card.phaseEl.dataset.phase, 'working');
            assert.equal(card.phaseEl.textContent, 'Working');
            assert.equal(card.phaseEl.className, 'chat-live-phase working');
            assert.equal(card.inlineTypingEl.style.display, '', 'clearing both causes restores real activity');
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

test('held children do not claim room activity, while a real working sibling still does', () => {
    const activities = computeHydratedDirectActivities(new Map(), [held], 7);
    const root = { groupId: 'same-id', root: { isConnected: true }, projectHold: hold.label };
    const child = { groupId: 'child', isSubagent: true, root: { isConnected: true }, projectHold: hold.label };
    assert.equal(computeDerivedChatStatus(chatStatusCounts(activities, [root, child])).text, hold.label);
    assert.equal(computeDerivedChatStatus(chatStatusCounts(new Map(), [child])).text, hold.label);
    const sibling = { groupId: 'sibling', root: { isConnected: true } };
    assert.equal(computeDerivedChatStatus(chatStatusCounts(activities, [root, child, sibling])).text, 'Working...');
    child.projectHold = '';
    assert.equal(computeDerivedChatStatus(chatStatusCounts(activities, [root, child])).text, 'Working...');
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
        const snapshot = () => instance.hydrateStateSnapshot({ active_chat_activities: [held],
            active_chat_activities_complete: true, supervisor_ready: true });
        const status = () => globalThis.document.byId.get('chat-status').textContent;
        snapshot(); await settle();
        assert.equal(status(), 'Working...', 'the child has not entered its hold yet');
        queue.pending = queue.running.splice(0);
        queue.pending[0].task.project_admission_hold = hold;
        snapshot();
        assert.equal(status(), 'Working...', 'the snapshot finishes before its queue read');
        await settle();
        assert.equal(status(), hold.label, 'the async queue response refreshes the header without another snapshot');
        const messages = globalThis.document.byId.get('chat-messages');
        const child = walkCard(messages, 'child');
        assert.equal(child.querySelector('[data-live-phase]').textContent, hold.label);
        connected = false; handlers.get('close')();
        snapshot(); await settle();
        assert.equal(child.querySelector('[data-live-phase]').textContent, 'Activity unconfirmed');
        assert.equal(child.querySelector('[data-live-phase]').dataset.motion, '0');
        connected = true;
        snapshot(); await settle();
        assert.equal(child.querySelector('[data-live-phase]').textContent, hold.label,
            'a fresh child queue read restores its own hold after reconnect');
        assert.equal(child.querySelector('[data-live-phase]').dataset.motion, '0');
        queue.pending[0].task.project_admission_hold = {};
        snapshot(); await settle();
        assert.equal(child.querySelector('[data-live-phase]').textContent, 'Queued');
        assert.equal(child.querySelector('[data-live-phase]').dataset.motion, '0');
        queue.pending[0].task.project_admission_hold = hold;
        snapshot(); await settle();
        frame('working-sibling');
        assert.equal(status(), 'Working...', 'independent working card keeps room activity');
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
        assert.equal(reads.filter(url => url === '/api/tasks/child').length, 1,
            'a current nonterminal answer retires missing membership without inventing hold recovery');
        assert.equal(child.querySelector('[data-live-phase]').textContent, hold.label);
        queue.pending = [{ id: 'child', task: { project_admission_hold: hold } }];
        snapshot(); await settle();
        queue.pending = [];
        detailStatus = 'cancelled';
        snapshot(); await settle();
        assert.equal(reads.filter(url => url === '/api/tasks/child').length, 2, 'a new queue departure is a new observation');
        assert.equal(child.dataset.finished, '1', 'terminal detail heals a missing terminal frame');
        for (let pass = 0; pass < 3; pass++) { snapshot(); await settle(); }
        assert.equal(reads.filter(url => url === '/api/tasks/child').length, 2);
    } finally { instance?.destroy(); restoreDom(prior); }
});
