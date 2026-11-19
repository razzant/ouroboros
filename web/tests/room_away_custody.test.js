// A room away (room_retire.js) is hidden and quiet, yet nothing it owes is lost on its return. Through the real chat
// consumer: a disconnect while it is away still leaves its unsaved sends in doubt (Send again once it is shown, never
// drawn into the hidden room, and only for what was written before that disconnect), and a review detail read that
// answers while it is away is drawn on its return, never into the hidden room.
import assert from 'node:assert/strict';
import test from 'node:test';
import { createChatInstance } from '../modules/chat.js';
import { createRoomRetirement, roomSocket } from '../modules/room_retire.js';
import { installDom, restoreDom, walkCard } from './chat_dom_fixture.js';

const ME = 'session-me';
const TS = '2026-11-04T09:40:00Z';
const settle = async () => { for (let i = 0; i < 12; i += 1) await new Promise((resolve) => setTimeout(resolve, 0)); };

// The page's one socket (ws.js), as a room's view of it sees it: subscriptions, connection and sends.
function pageSocket() {
    const handlers = new Map();
    let connected = true, coined = 0, status = 'sent';
    const sent = [];
    return {
        sent,
        set status(value) { status = value; },
        set connected(value) { connected = value; },
        ws: { readyState: 1 },
        on(type, fn) {
            if (!handlers.has(type)) handlers.set(type, new Set());
            handlers.get(type).add(fn);
            return () => handlers.get(type).delete(fn);
        },
        isConnected: () => connected,
        send(frame, options) {
            const id = frame.client_message_id || `cm-${++coined}`;
            sent.push({ frame: { ...frame, client_message_id: id }, options });
            return { status, clientMessageId: id };
        },
        fire: (type, payload) => [...(handlers.get(type) || [])].forEach((fn) => fn(payload)),
    };
}

function fixture({ detail = null } = {}) {
    const reads = [];
    const env = installDom(async (url) => {
        const path = String(url);
        if (path.startsWith('/api/tasks/') && detail) return new Promise((resolve) => reads.push(resolve));
        return { ok: true, json: async () => (path.startsWith('/api/chat/history')
            ? { messages: [], window: { complete: true } }
            : { active_chat_activities: [], active_chat_activities_complete: true, supervisor_ready: true }) };
    });
    globalThis.sessionStorage.setItem('ouro_chat_session_id', ME);
    const socket = pageSocket();
    let generation = 0;
    const instance = createChatInstance({
        ws: roomSocket(socket),
        state: { activePage: 'chat', projectChatIds: new Set([7]), unreadCount: 0 },
        updateUnreadBadge() {}, chatId: 7, projectId: 'p1', idPrefix: 'chat', mountEl: env.mount, asPanel: true,
        stateSnapshots: { begin: () => ({ generation: ++generation, requestedAt: Date.now() }),
            gate() { return Promise.resolve(this.begin()); }, isCurrent: () => true, apply() {}, fail() {}, latest: () => null },
    });
    const instances = new Map([['p1', instance]]);
    const rooms = createRoomRetirement({ instances });
    const messages = document.byId.get('chat-messages');
    const bubbleOf = (cmid) => messages.children.find((node) => node.dataset.clientMessageId === cmid);
    const actions = (node) => (node?.querySelectorAll('.msg-pending') || [])
        .flatMap((note) => note.children.filter((child) => child.tagName === 'BUTTON')).map((button) => button.textContent);
    return {
        instance, socket, rooms, messages, bubbleOf, actions, reads,
        async send(text) {
            document.byId.get('chat-input').value = text;
            for (const fn of document.byId.get('chat-send').listeners.get('click')) fn();
            await settle();
            return socket.sent.at(-1).frame.client_message_id;
        },
        away() { rooms.putAway('p1'); },
        async back() { instance.page.hidden = false; rooms.show('p1'); await settle(); },
        close() { instance.destroy(); restoreDom(env.prior); },
    };
}

test('a disconnect while the room is away leaves its unsaved send in doubt on its return, not in the hidden room', async () => {
    const f = fixture();
    try {
        const cmid = await f.send('look at the logs');
        f.away();
        f.socket.connected = false;
        f.socket.fire('close');
        assert.deepEqual(f.actions(f.bubbleOf(cmid)), [], 'nothing is drawn into the hidden room');
        f.socket.connected = true;  // the socket came back while the room was away; the host never saved the row
        await f.back();
        assert.deepEqual(f.actions(f.bubbleOf(cmid)), ['Send again', 'Discard'],
            'back, after its history caught up, the room says the send was not confirmed');
        assert.equal(f.instance.hasPendingWork(), true);
    } finally { f.close(); }
});

test('back while still disconnected, the room shows the outage and the doubt at once', async () => {
    const f = fixture();
    try {
        const cmid = await f.send('still offline?');
        f.away();
        f.socket.connected = false;
        f.socket.fire('close');
        await f.back();
        assert.deepEqual(f.actions(f.bubbleOf(cmid)), ['Send again', 'Discard']);
    } finally { f.close(); }
});

test('a send saved while the room was away is not put in doubt by an earlier disconnect', async () => {
    const f = fixture();
    try {
        const cmid = await f.send('saved after all');
        f.away();
        f.socket.connected = false;
        f.socket.fire('close');
        f.socket.connected = true;
        // The custody listener hears the host's saved row while the room is away.
        f.socket.fire('chat', { type: 'chat', role: 'user', chat_id: 7, ts: TS, source: 'web', content: 'saved after all',
            sender_session_id: ME, client_message_id: cmid, ingress_accepted: true, ingress_dispatched: true });
        await f.back();
        assert.deepEqual(f.actions(f.bubbleOf(cmid)), []);
        assert.equal(f.instance.hasPendingWork(), false);
    } finally { f.close(); }
});

test('a frame the socket queue wrote after the disconnect is not in doubt for that disconnect', async () => {
    const f = fixture();
    try {
        f.socket.status = 'queued';
        f.socket.connected = false;
        const cmid = await f.send('sent while offline');
        f.away();
        f.socket.fire('close');                       // a failed reconnect attempt: the frame is still the queue's
        f.socket.connected = true;
        f.socket.fire('outbound_sent', { clientMessageId: cmid, queued: true });  // written once the socket came back
        await f.back();
        assert.deepEqual(f.actions(f.bubbleOf(cmid)), [], 'written after that disconnect: it waits for its saved row');
        f.socket.fire('close');
        assert.deepEqual(f.actions(f.bubbleOf(cmid)), ['Send again', 'Discard'], 'a disconnect after it was written does');
    } finally { f.close(); }
});

test('a review detail that answers while the room is away is drawn on its return, not into the hidden room', async () => {
    const taskId = 'review-task';
    const detail = { task_id: taskId, status: 'running', review_projection: { panels: [], acceptance_incident: {
        incident_id: 'acceptance-preparation:1', status: 'failed', stage: 'preparation', attempts: 1, source_known: true,
        failure_kind: 'UnicodeDecodeError', failure_detail: 'bad byte', feedback_delivered: true } } };
    const f = fixture({ detail });
    try {
        f.socket.fire('chat', { chat_id: 7, role: 'system', is_progress: true, task_id: taskId, content: 'assembling',
            ts: '2026-11-04T09:00:00Z' });
        f.socket.fire('chat', { task_id: taskId, chat_id: 7, presentation_owner_task_id: taskId, role: 'system',
            is_progress: true, system_type: 'review_reference', surface: 'task_acceptance', state_revision: 'a'.repeat(64),
            ts: '2026-11-04T09:00:01Z', content: '' });
        await settle();
        assert.equal(f.reads.length, 1, 'the card reads its review detail');
        const host = walkCard(f.messages, taskId).querySelector('[data-live-reviews-host]');
        const before = host.innerHTML;
        f.away();
        f.reads.shift()({ ok: true, json: async () => detail });
        await settle();
        assert.equal(host.innerHTML, before, 'the answer waits: nothing is drawn into the hidden room');
        await f.back();
        assert.match(host.innerHTML, /acceptance evidence · host attempt 1/, 'back, the room draws what its read returned');
    } finally { f.close(); }
});

test('a room away draws no status when another room\'s send is written; back, it shows its own again', async () => {
    const f = fixture();
    try {
        await settle();
        const status = () => document.byId.get('chat-status')?.textContent;
        const shown = status();
        assert.notEqual(shown, 'Reconnecting…');
        f.away();
        // Every room's custody listener hears every written frame; a paused socket view is not an outage.
        f.socket.fire('outbound_sent', { clientMessageId: 'another-room-send' });
        assert.equal(status(), shown, 'nothing is drawn into the hidden room, least of all a false outage');
        await f.back();
        assert.equal(status(), shown);
    } finally { f.close(); }
});
