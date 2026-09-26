/* The client-level subscription: what `attach()` lets through, what it refuses,
   and what it learns. These are the rules a per-room wiring got wrong — a
   closed Project panel destroys its chat instance, so the room can never be the
   subscription's owner. */
import test from 'node:test';
import assert from 'node:assert/strict';

import {
    DEFAULT_NOTIFY_PREFS,
    NOTIFY_PREFS_KEY,
    createNotifier,
} from '../modules/notifications.js';

const ON = { ...DEFAULT_NOTIFY_PREFS, enabled: true, main_reply: true };

function fixture({ prefs = ON, projects = [2] } = {}) {
    const handlers = new Map();
    const disposed = [];
    const ws = {
        on(event, fn) {
            handlers.set(event, fn);
            return () => { disposed.push(event); handlers.delete(event); };
        },
    };
    const built = [];
    class FakeNotification {
        static permission = 'granted';
        constructor(title, options) { this.title = title; this.options = options; built.push(this); }
        close() {}
    }
    const documentRef = {
        addEventListener() {}, removeEventListener() {}, querySelectorAll: () => [],
    };
    const notifier = createNotifier({
        storage: {
            getItem: () => JSON.stringify(prefs),
            setItem: () => {},
        },
        notificationCtor: FakeNotification,
        audioContextCtor: null,
        showToast: () => {},
        documentRef,
    });
    const release = notifier.attach({
        ws,
        ownerVisibleChat: (id) => id === 1 || projects.includes(id),
    });
    const send = (event, frame) => handlers.get(event)?.(frame);
    return { notifier, ws, handlers, disposed, built, release, send, titles: () => built.map((n) => n.title) };
}

test('a closed Project room still notifies: the socket carries it, not the room', () => {
    const fx = fixture({ projects: [42] });
    // Nothing in this test mounts a chat instance for chat 42 — that is the
    // point. The panel may be closed or never opened.
    fx.send('quiz', {
        task_id: 't-p', chat_id: 42,
        quiz: { quiz_id: 'q-p', state: 'open', wait_for_answer: true, question: 'Which one?' },
    });
    assert.deepEqual(fx.titles(), ['Ouroboros is waiting for your answer']);
});

test('machine traffic never notifies', () => {
    const fx = fixture();
    fx.send('chat', { role: 'system', system_type: 'task_summary', task_id: 'h', chat_id: 0 });
    fx.send('chat', { role: 'system', system_type: 'task_summary', task_id: 'a', chat_id: -7 });
    assert.deepEqual(fx.titles(), [], 'the hidden partition and A2A are machine traffic');
    // The owner's own threads do.
    fx.send('chat', { role: 'system', system_type: 'task_summary', task_id: 'm', chat_id: 1 });
    fx.send('chat', { role: 'system', system_type: 'task_summary', task_id: 'p', chat_id: 2 });
    assert.deepEqual(fx.titles(), ['Task finished', 'Task finished']);
});

test('an interrupted terminal neither rings nor burns the key', () => {
    const fx = fixture();
    // Update/restart teardown: pushed under the same task id, then requeued.
    fx.send('log', { chat_id: 1, data: { type: 'task_done', task_id: 't-r', status: 'interrupted' } });
    assert.deepEqual(fx.titles(), []);
    // The real completion of that same task still reaches the owner.
    fx.send('log', { chat_id: 1, data: { type: 'task_done', task_id: 't-r', status: 'completed' } });
    assert.deepEqual(fx.titles(), ['Task finished']);
});

test('an external owner transport notifies; another Project room does not leak', () => {
    const fx = fixture({ projects: [2] });
    // Main accepts any positive chat not stamped as a Project's (an external
    // owner transport), and so must the notifier.
    fx.send('chat', { role: 'system', system_type: 'task_summary', task_id: 'tg',
                      chat_id: 994321, content: 'done' });
    assert.deepEqual(fx.titles(), ['Task finished']);
    // A frame stamped as some other Project's room, which this client does not
    // know, stays out.
    fx.send('chat', { role: 'system', system_type: 'task_summary', task_id: 'other',
                      chat_id: 777, project_thread: true });
    assert.equal(fx.built.length, 1);
});

test('a throw inside the notifier cannot break message rendering', () => {
    const handlers = new Map();
    const ws = { on(event, fn) { handlers.set(event, fn); return () => handlers.delete(event); } };
    const notifier = createNotifier({
        storage: { getItem: () => JSON.stringify(ON), setItem: () => {} },
        notificationCtor: undefined,
        audioContextCtor: null,
        showToast: () => {},
        documentRef: { addEventListener() {}, removeEventListener() {}, querySelectorAll: () => [] },
    });
    notifier.attach({ ws, ownerVisibleChat: () => { throw new Error('boom'); } });
    // This listener runs BEFORE the chat instances' own handlers and ws.emit
    // does not isolate them, so it must absorb its own failure.
    assert.doesNotThrow(() => handlers.get('chat')({ role: 'assistant', chat_id: 5 }));
});

test('a log frame is unwrapped and keyed by its own task identity', () => {
    const fx = fixture();
    fx.send('log', { chat_id: 1, data: { type: 'task_done', task_id: 't-log', status: 'completed' } });
    assert.deepEqual(fx.titles(), ['Task finished']);
    // The same conclusion arriving later as the authored summary is silent.
    fx.send('chat', { role: 'system', system_type: 'task_summary', task_id: 't-log', chat_id: 1 });
    assert.equal(fx.built.length, 1);
});

test('lineage is learned from the wire, so a child terminal stays with its parent', () => {
    const fx = fixture();
    // A subagent's ordinary traffic declares the lineage the terminal omits.
    fx.send('chat', {
        role: 'assistant', is_progress: true, chat_id: 1,
        delegation_role: 'subagent', parent_task_id: 'root-1', subagent_task_id: 'child-1',
    });
    fx.send('log', { chat_id: 1, data: { type: 'task_done', task_id: 'child-1', status: 'completed' } });
    assert.deepEqual(fx.titles(), [], 'the child escalates to its parent, it does not ring');
    // Its parent's conclusion does notify.
    fx.send('log', { chat_id: 1, data: { type: 'task_done', task_id: 'root-1', status: 'completed' } });
    assert.deepEqual(fx.titles(), ['Task finished']);
});

test('a delegation fact on the terminal itself marks a child', () => {
    // Second, independent signal: a subagent terminal carries the executor /
    // substrate enrichment an ordinary root terminal does not, so a child whose
    // earlier traffic this client never saw is still recognised.
    const fx = fixture();
    fx.send('log', { chat_id: 1, data: { type: 'task_done', task_id: 'child-x',
                                        status: 'completed', actual_substrate: 'harness_used' } });
    assert.deepEqual(fx.titles(), []);
    fx.send('log', { chat_id: 1, data: { type: 'task_done', task_id: 'child-y',
                                        status: 'completed',
                                        execution_evidence: { dispatch_executor: 'harness' } } });
    assert.deepEqual(fx.titles(), []);
});

test('a bare parent_task_id is enough to mark a child', () => {
    const fx = fixture();
    fx.send('chat', {
        role: 'assistant', is_progress: true, chat_id: 1,
        parent_task_id: 'root-2', task_id: 'child-2',
    });
    fx.send('log', { chat_id: 1, data: { type: 'task_done', task_id: 'child-2', status: 'completed' } });
    assert.deepEqual(fx.titles(), []);
});

test('lineage is learned even while notifications are off', () => {
    const fx = fixture({ prefs: DEFAULT_NOTIFY_PREFS });
    fx.send('chat', {
        role: 'assistant', is_progress: true, chat_id: 1,
        delegation_role: 'subagent', parent_task_id: 'root-3', subagent_task_id: 'child-3',
    });
    // Switching on mid-session must not start from an empty map.
    fx.notifier.setPref('enabled', true);
    fx.send('log', { chat_id: 1, data: { type: 'task_done', task_id: 'child-3', status: 'completed' } });
    assert.deepEqual(fx.titles(), []);
});

test('a conversation turn is not a finished task', () => {
    // Every direct turn — an ordinary Main answer, a consciousness wake-up —
    // ends with the SAME log frame a managed task does. Reading that as "a task
    // finished" would ring on every reply while the owner's ordinary-reply
    // toggle is off, which is the opposite of what that toggle promises.
    const off = fixture({ prefs: { ...ON, main_reply: false } });
    off.send('chat', { role: 'assistant', chat_id: 1, content: 'Here it is.', client_message_id: 'c1' });
    off.send('log', { chat_id: 1, data: { type: 'task_done', task_id: 't-direct',
                                          status: 'completed', _is_direct_chat: true } });
    assert.deepEqual(off.titles(), [], 'a conversation ending is not a task terminal');

    // With the ordinary-reply category on, the SAME turn rings once, as itself.
    const on = fixture({ prefs: { ...ON, main_reply: true } });
    on.send('chat', { role: 'assistant', chat_id: 1, task_id: 't-direct2', content: 'Here it is.' });
    on.send('log', { chat_id: 1, data: { type: 'task_done', task_id: 't-direct2',
                                        status: 'completed', _is_direct_chat: true } });
    assert.deepEqual(on.titles(), ['Ouroboros replied']);

    // A managed task's terminal still rings.
    const managed = fixture();
    managed.send('log', { chat_id: 1, data: { type: 'task_done', task_id: 't-managed',
                                              status: 'completed', _is_direct_chat: false } });
    assert.deepEqual(managed.titles(), ['Task finished']);
});

test('an ordinary reply in a Project room is not the Main-reply category', () => {
    const fx = fixture();
    fx.send('chat', { role: 'assistant', chat_id: 2, content: 'done', client_message_id: 'c1' });
    assert.deepEqual(fx.titles(), []);
    fx.send('chat', { role: 'assistant', chat_id: 1, content: 'done', client_message_id: 'c2' });
    assert.deepEqual(fx.titles(), ['Ouroboros replied']);
});

test('a frame with no chat id is legacy Main traffic', () => {
    const fx = fixture();
    fx.send('chat', { role: 'system', system_type: 'task_summary', task_id: 'legacy' });
    assert.deepEqual(fx.titles(), ['Task finished']);
});

test('attach returns a disposer, and destroy releases the subscription', () => {
    const fx = fixture();
    assert.deepEqual([...fx.handlers.keys()], ['chat', 'quiz', 'log']);
    fx.release();
    assert.deepEqual(fx.disposed.sort(), ['chat', 'log', 'quiz']);
    assert.equal(fx.handlers.size, 0);

    const second = fixture();
    second.notifier.destroy();
    assert.equal(second.handlers.size, 0, 'destroy releases what attach took');
});

test('attach without a socket is a no-op disposer, never a throw', () => {
    const notifier = createNotifier({
        storage: { getItem: () => JSON.stringify(ON), setItem: () => {} },
        notificationCtor: undefined,
        audioContextCtor: null,
        documentRef: { addEventListener() {}, removeEventListener() {}, querySelectorAll: () => [] },
    });
    const release = notifier.attach({});
    assert.equal(typeof release, 'function');
    release();
});

test('an owner notification rings from the log lane for the owner chat, never the hidden partition', () => {
    const { notifier, handlers, built } = fixture();
    const release = notifier.attach({ ws: { on: (event, fn) => { handlers.set(event, fn); return () => {}; } } });
    const row = { type: 'owner_notification', text: 'Meeting in 15 min', source: 'skill:calendar', key: 'k1', ts: '2026-09-25T14:30:00+00:00' };
    handlers.get('log')({ type: 'log', data: { ...row, chat_id: 0 }, chat_id: 0 });
    assert.equal(built.length, 0, 'chat 0 is machine traffic');
    handlers.get('log')({ type: 'log', data: { ...row, chat_id: 1 }, chat_id: 1 });
    assert.equal(built.length, 1);
    assert.equal(built[0].title, 'Reminder from calendar');
    handlers.get('log')({ type: 'log', data: { ...row, chat_id: 1 }, chat_id: 1 });
    assert.equal(built.length, 1, 'the same producer key rings once');
    release();
});
