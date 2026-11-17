// A room away draws nothing (room_retire.js). The full output of a line the owner expanded can answer after the room
// left the screen: it waits, kept, and the room's return (chat.js `stay`) draws that card's timeline once from it, with
// no second read. A room on screen draws it as soon as it answers, as before.
import assert from 'node:assert/strict';
import test from 'node:test';
import { createChatInstance } from '../modules/chat.js';
import { createRoomRetirement, roomSocket } from '../modules/room_retire.js';
import { ElementStub, installDom, restoreDom, walkCard } from './chat_dom_fixture.js';

const settle = async () => { for (let i = 0; i < 12; i += 1) await new Promise((resolve) => setTimeout(resolve, 0)); };
const RESULT = 'REPORT_HEAD\n' + 'The retained answer has a complete paragraph.\n'.repeat(120) + 'REPORT_TAIL';
const TRACE = 'TRACE_HEAD\n' + 'Read the recorded evidence.\n'.repeat(160) + 'TRACE_TAIL';
// A child's completed report, its result and trace capped at 4000 characters on the wire: Show full reads the rest.
const DETAIL = {
    type: 'send_message', is_progress: true, chat_id: 7, task_id: 'child-1', delegation_role: 'subagent',
    parent_task_id: 'root-1', root_task_id: 'root-1', subagent_task_id: 'child-1', subagent_role: '',
    subagent_event: 'completed', status: 'completed', model: 'x-ai/grok-4.7', result: RESULT, trace_summary: TRACE,
    ts: '2026-10-25T14:12:58Z',
};
const ROWS = [
    { chat_id: 7, role: 'system', is_progress: true, content: 'Reading the brief.', narration: true,
        ts: '2026-10-25T13:35:40Z', task_id: 'root-1' },
    { ...DETAIL, role: 'system', content: '', text: '', result: RESULT.slice(0, 4000), result_truncated: true,
        trace_summary: TRACE.slice(0, 4000), trace_summary_truncated: true },
];

async function fixture() {
    const fullReads = [];  // the line's full-output reads (chat.js fetchFullLineOutput), answered by the test
    const { prior, mount } = installDom(async (url, init = {}) => {
        const path = String(url);
        if (path === '/api/tasks/child-1' && init.cache === 'no-store') return new Promise((resolve) => fullReads.push(resolve));
        return { ok: true, json: async () => {
            if (path.startsWith('/api/chat/history')) return { messages: ROWS, page_cursor: 'recent', has_more: false,
                next_cursor: null, window: { complete: true } };
            if (path === '/api/tasks/child-1') return DETAIL;  // the card's review detail read
            return { active_chat_activities: [], active_chat_activities_complete: true, supervisor_ready: true };
        } };
    });
    // The timeline writes this stub keeps flat: a line drawn with the full output is a write holding its tail.
    const html = Object.getOwnPropertyDescriptor(ElementStub.prototype, 'innerHTML');
    let drawn = 0;
    Object.defineProperty(ElementStub.prototype, 'innerHTML', { ...html, set(value) {
        if (/^\s*<div\s+class="chat-live-line\s/.test(String(value)) && String(value).includes('REPORT_TAIL')) drawn += 1;
        html.set.call(this, value);
    } });
    const socket = { on() { return () => {}; }, isConnected: () => true, send: () => ({ status: 'sent' }), ws: { readyState: 1 } };
    let generation = 0;
    const instance = createChatInstance({
        ws: roomSocket(socket),
        state: { activePage: 'chat', projectChatIds: new Set([7]), unreadCount: 0 },
        updateUnreadBadge() {}, chatId: 7, projectId: 'p1', idPrefix: 'chat', mountEl: mount, asPanel: true,
        stateSnapshots: { begin: () => ({ generation: ++generation, requestedAt: Date.now() }),
            gate() { return Promise.resolve(this.begin()); }, isCurrent: () => true, apply() {}, fail() {}, latest: () => null },
    });
    const rooms = createRoomRetirement({ instances: new Map([['p1', instance]]) });
    await instance.refreshHistory({ revision: 1 });
    const card = walkCard(globalThis.document.byId.get('chat-messages'), 'child-1');
    assert.ok(card, 'the child card is drawn');
    card.querySelector('[data-live-disclosure]').click();
    await settle();
    const timeline = card.querySelector('[data-live-timeline]');
    return {
        instance, rooms, timeline, fullReads,
        get drawn() { return drawn; },
        // Show full on the card's one expandable line. The fixture has no general closest/bubbling: the real listener.
        showFull() {
            const line = timeline.querySelector('.chat-live-line.expandable');
            const target = { closest: (selector) => (selector === '.chat-live-line.expandable' ? line : null) };
            for (const fn of timeline.listeners.get('click') || []) fn({ target, stopPropagation() {} });
        },
        answer: () => fullReads.shift()({ ok: true, json: async () => DETAIL }),
        close() {
            instance.destroy();
            restoreDom(prior);
            Object.defineProperty(ElementStub.prototype, 'innerHTML', html);
        },
    };
}

test('a full output that answers while the room is away is drawn once on its return, from what it read', async () => {
    const f = await fixture();
    try {
        f.showFull();
        assert.equal(f.fullReads.length, 1, 'Show full reads the task once');
        f.rooms.putAway('p1');
        f.answer();
        await settle();
        assert.equal(f.drawn, 0, 'away, nothing is drawn into the hidden room');
        assert.doesNotMatch(f.timeline.innerHTML, /REPORT_TAIL/);
        f.instance.page.hidden = false;
        f.rooms.show('p1');  // stay(): its return
        assert.equal(f.drawn, 1, 'back, the card\'s timeline is drawn once, with the full output');
        await settle();
        assert.equal(f.fullReads.length, 0, 'what the read returned is kept: no second read');
    } finally { f.close(); }
});

test('a full output that answers while the room is on screen is drawn at once', async () => {
    const f = await fixture();
    try {
        f.showFull();
        f.answer();
        await settle();
        assert.equal(f.drawn, 1);
        f.rooms.putAway('p1');
        f.instance.page.hidden = false;
        f.rooms.show('p1');
        assert.equal(f.drawn, 1, 'its return owes nothing more');
    } finally { f.close(); }
});
