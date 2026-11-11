// A Project room that is left keeps its content while it leaves and stays whole while it is away (room_retire.js):
// out of the live instances, marked leaving and inert while its exit runs, then hidden and quiet but kept; back as the
// same room, its selection given back and its frames flowing again. Every visited room stays (parent decision
// 2026-11-04): only its project going away lets one go; its socket view carries only its own sends' custody while away.
import assert from 'node:assert/strict';
import test from 'node:test';

import { createCatchUp, createRoomRetirement, roomSocket } from '../modules/room_retire.js';

function room({ hidden = false, pending = false } = {}) {
    const page = { hidden, inert: false, dataset: {}, contains: (node) => node?.room === r };
    const r = { page, calls: [],
        leave() { this.calls.push('leave'); }, stay() { this.calls.push('stay'); }, destroy() { this.calls.push('destroy'); },
        closeTransient() { this.calls.push('transient'); }, hasPendingWork: () => pending,
        ws: { paused: false, pause() { this.paused = true; }, resume() { this.paused = false; } } };
    return r;
}

function world() {
    const instances = new Map();
    const cancelled = [];
    const rooms = createRoomRetirement({ instances, cancelPaint: (pid) => cancelled.push(pid) });
    return { instances, cancelled, rooms };
}

test('a room on screen leaves whole for its exit, then is kept away: hidden, quiet, not destroyed', () => {
    const w = world();
    const r = room();
    w.instances.set('p1', r);
    w.rooms.retire('p1');
    assert.equal(w.instances.has('p1'), false);
    assert.equal(w.rooms.retiring.get('p1'), r);
    assert.deepEqual([r.page.inert, 'leaving' in r.page.dataset, r.calls, r.ws.paused], [true, true, ['transient', 'leave'], true],
        'another place replaced it: its reader closes and its playback pauses before it leaves');
    assert.deepEqual(w.cancelled, ['p1'], 'its history paint is cancelled');
    w.rooms.settle('p1', r);  // its exit has run
    assert.deepEqual([r.page.inert, 'leaving' in r.page.dataset, r.page.hidden], [false, false, true]);
    assert.deepEqual(r.calls, ['transient', 'leave'], 'still left: no stay, no destroy');
    assert.equal(w.instances.get('p1'), r, 'kept among the instances, away');
    assert.equal(w.rooms.retiring.size, 0);
    w.rooms.show('p1');
    assert.deepEqual([r.calls.at(-1), r.ws.paused], ['stay', false], 'back: it stays and hears its frames again');
});

test('back, a room takes the last census at once, dated before anything it saw live; a new room is not treated as back', () => {
    const instances = new Map();
    const census = { works: [] };
    const hydrated = [];
    const rooms = createRoomRetirement({ instances, latestState: () => census });
    const r = room();
    r.hydrateStateSnapshot = (...args) => hydrated.push(args);
    instances.set('p1', r);
    rooms.show('p1');
    assert.deepEqual(hydrated, [], 'never away: nothing to catch up');
    rooms.putAway('p1');
    r.page.hidden = false;
    rooms.show('p1');
    assert.deepEqual(hydrated, [[census, 0]]);
});

test('a room merely suspended (Main behind another place, a hidden window) keeps its reader and its playback', () => {
    const w = world();
    const main = room();
    w.rooms.suspend(main);
    assert.deepEqual([main.calls, main.ws.paused], [['leave'], true], 'quiet, but nothing it opened or plays is touched');
    w.rooms.restore(main);
    assert.deepEqual([main.calls.at(-1), main.ws.paused], ['stay', false]);
});

test('a project going away while its room leaves is let go once the exit settles, not kept hidden', () => {
    const w = world();
    const r = room();
    w.instances.set('p1', r);
    w.rooms.retire('p1');     // closing the panel of a project being deleted …
    w.rooms.forget('p1');     // … and the registry's word arrives during its exit
    assert.equal(r.calls.includes('destroy'), false, 'still leaving: its exit runs first');
    w.rooms.settle('p1', r);
    assert.deepEqual([r.calls.at(-1), w.instances.has('p1')], ['destroy', false], 'its exit settled: let go');
    w.rooms.forget();
    assert.equal(r.calls.filter((c) => c === 'destroy').length, 1, 'once');
});

test('a room going away that holds work only it can finish is let go on a later read once that work has ended', () => {
    const w = world();
    let pending = true;
    const r = room();
    r.hasPendingWork = () => pending;
    w.instances.set('p1', r);
    w.rooms.putAway('p1');
    w.rooms.forget('p1');
    assert.equal(w.instances.get('p1'), r, 'its staged files or unsaved send keep it');
    pending = false;
    w.rooms.forget();  // the next projects read (app.js names the projects still deleting, here none new)
    assert.deepEqual([r.calls.at(-1), w.instances.has('p1')], ['destroy', false]);
});

test('a room shown again is wanted: an earlier going-away word no longer lets it go', () => {
    const w = world();
    const r = room();
    w.instances.set('p1', r);
    w.rooms.forget('p1');  // the word arrives while it is on screen: nothing to let go yet
    assert.equal(w.instances.get('p1'), r);
    w.rooms.show('p1');    // the owner opens it again (the deletion was undone or refused)
    w.rooms.putAway('p1');
    w.rooms.forget();
    assert.deepEqual([w.instances.get('p1'), r.calls.includes('destroy')], [r, false]);
});

test('the registry retires a room kept away; only work only it can finish keeps it', () => {
    const w = world();
    const plain = room(), pending = room({ pending: true }), shown = room();
    w.instances.set('plain', plain); w.rooms.putAway('plain');
    w.instances.set('pending', pending); w.rooms.putAway('pending');
    w.instances.set('shown', shown);
    for (const pid of ['plain', 'pending', 'shown']) w.rooms.forget(pid);
    assert.deepEqual([...w.instances.keys()], ['pending', 'shown']);
    assert.equal(plain.calls.at(-1), 'destroy');
});

test('a hidden room has nothing to show leaving: it stays away as it is', () => {
    const w = world();
    const r = room({ hidden: true });
    w.instances.set('p1', r);
    w.rooms.retire('p1');
    assert.equal(w.instances.get('p1'), r);
    assert.equal(w.rooms.retiring.size, 0);
    assert.equal(r.calls.includes('destroy'), false);
});

test('reopened during its exit it is the same room; a late put-away of a replaced room destroys only itself', () => {
    const w = world();
    const r = room();
    w.instances.set('p1', r);
    w.rooms.retire('p1');
    w.rooms.settle('p1', undefined, false);  // reopened
    assert.equal(w.instances.get('p1'), r);
    assert.equal(r.calls.includes('destroy'), false);
    w.rooms.retire('p1');
    const next = room();
    w.instances.set('p1', next);  // a new room took its place meanwhile
    w.rooms.settle('p1', r);
    assert.deepEqual(r.calls.at(-1), 'destroy', 'the old room destroys itself');
    assert.equal(w.instances.get('p1'), next, 'the new room is untouched');
    w.rooms.settle('p1', r);
    assert.equal(r.calls.filter((c) => c === 'destroy').length, 1, 'once');
});

test('left, reopened and left again: the first exit ending late settles nothing, the second puts the room away', () => {
    const w = world();
    const r = room();
    w.instances.set('p1', r);
    w.rooms.retire('p1');
    const [first] = w.rooms.exits();
    w.rooms.settle('p1', undefined, false);  // reopened during its exit …
    w.rooms.retire('p1');                     // … and left again
    const [second] = w.rooms.exits();
    assert.deepEqual([first[0], first[1], second[0], second[1]], ['p1', r, 'p1', r], 'the same room, leaving twice');
    w.rooms.settle(...first.slice(0, 2), true, first[2]);  // the first exit's motion ends during the second
    assert.equal(w.rooms.retiring.get('p1'), r, 'still leaving: its words stay while the second exit runs');
    assert.deepEqual([r.page.hidden, r.page.inert, 'leaving' in r.page.dataset], [false, true, true]);
    w.rooms.settle(...second.slice(0, 2), true, second[2]);
    assert.deepEqual([r.page.hidden, w.instances.get('p1')], [true, r], 'the second exit puts it away');
    assert.deepEqual(w.rooms.exits(), []);
});

test('every visited room stays away whole, however many: A→…→A returns the same room at once, none evicted', () => {
    const w = world();
    const visited = Array.from({ length: 24 }, (_, index) => [`p${index}`, room({ pending: index === 3 })]);
    for (const [pid, r] of visited) { w.instances.set(pid, r); w.rooms.putAway(pid); }
    assert.equal(w.instances.size, 24, 'stateless or not, no room is let go');
    assert.equal(visited.some(([, r]) => r.calls.includes('destroy')), false);
    assert.equal(visited[3][1].page.dataset.pendingWork, '1', 'one holding work only it can finish is marked');
    const [first, a] = visited[0];
    a.page.hidden = false;
    w.rooms.show(first);
    assert.deepEqual([w.instances.get(first), a.calls.at(-1), a.ws.paused], [a, 'stay', false], 'the same room, at once');
    // Only the registry's word that a project is going away lets a room go, never one holding work only it can finish.
    for (const [pid] of visited.slice(1, 5)) w.rooms.forget(pid);
    assert.deepEqual([1, 2, 3, 4].map((index) => visited[index][1].calls.includes('destroy')), [true, true, false, true]);
    assert.equal(w.instances.size, 21);
});

test("each room's own selection is kept when it leaves and given back when it returns", (t) => {
    const ranges = [];
    let selection = null;
    const previous = globalThis.document;
    globalThis.document = undefined;
    t.after(() => { globalThis.document = previous; });
    const doc = {
        getSelection: () => ({
            get rangeCount() { return selection ? 1 : 0; },
            get isCollapsed() { return !selection; },
            get anchorNode() { return selection?.start; },
            get focusNode() { return selection?.end; },
            getRangeAt: () => ({ startContainer: selection.start, startOffset: selection.from,
                endContainer: selection.end, endOffset: selection.to }),
            removeAllRanges: () => { selection = null; },
            addRange: (range) => { selection = range; ranges.push(range); },
        }),
        createRange: () => ({ setStart(node, offset) { this.start = node; this.from = offset; },
            setEnd(node, offset) { this.end = node; this.to = offset; } }),
    };
    const w = world();
    const a = room(), b = room();
    for (const r of [a, b]) r.page.ownerDocument = doc;
    const inA = { room: a, isConnected: true }, inB = { room: b, isConnected: true };
    w.instances.set('a', a);
    selection = { start: inA, from: 2, end: inA, to: 9 };  // the owner selects in A …
    w.rooms.putAway('a');
    w.instances.set('b', b);
    selection = { start: inB, from: 0, end: inB, to: 4 };  // … then something else in B
    w.rooms.putAway('b');
    assert.equal(a.keptRanges.length, 1);
    w.rooms.show('a');
    w.rooms.restoreSelection(a);
    assert.deepEqual([selection.start, selection.from, selection.end, selection.to], [inA, 2, inA, 9], 'A gets its own back');
    w.rooms.show('b');
    w.rooms.restoreSelection(b);
    assert.deepEqual([selection.start, selection.from, selection.to], [inB, 0, 4], 'and B its own');
    assert.equal(w.instances.size, 2, 'both rooms stay');
});

test('a room away hears only its own sends\' custody; back, it hears everything and catches up as after a reconnect', () => {
    const listeners = new Map();
    let connected = true;
    const ws = {
        on(event, fn) { if (!listeners.has(event)) listeners.set(event, new Set()); listeners.get(event).add(fn); return () => listeners.get(event).delete(fn); },
        emit(event, data) { [...(listeners.get(event) || [])].forEach((fn) => fn(data)); },
        isConnected: () => connected, send: (frame) => ({ status: 'sent', frame }), ws: { readyState: 1 },
    };
    const socket = roomSocket(ws);
    const heard = [];
    for (const event of ['chat', 'log', 'outbound_sent', 'outbound_dropped', 'open', 'close']) {
        socket.on(event, (msg) => heard.push([event, msg]));
    }
    const count = () => [...listeners.values()].reduce((total, set) => total + set.size, 0);
    assert.equal(count(), 6);
    socket.pause();
    assert.equal(count(), 4, 'away, only custody listens: chat (filtered), outbound_sent, outbound_dropped, close');
    assert.equal(socket.isConnected(), false, 'and the room treats its live facts as not current');
    ws.emit('log', { data: {} });
    ws.emit('chat', { role: 'assistant', content: 'later' });
    ws.emit('chat', { role: 'user', client_message_id: 'm1' });
    ws.emit('chat', { role: 'system', system_type: 'initialization_notice' });
    // A task's incident still reaches it (it toasts, chat.js); its ordinary progress does not.
    ws.emit('chat', { role: 'system', is_progress: true, task_id: 't1', task_incident: 'deadline' });
    ws.emit('chat', { role: 'assistant', is_progress: true, task_id: 't1', content: 'narration' });
    ws.emit('chat', { role: 'system', task_incident: 'deadline' });  // not raised in a task's progress
    ws.emit('outbound_sent', { clientMessageId: 'm1' });
    ws.emit('outbound_dropped', { clientMessageId: 'm2' });
    ws.emit('close');  // a disconnect puts the room's unsaved sends in doubt; chat.js draws it on the return
    ws.emit('open', { previouslyConnected: true });  // the real socket reconnects while the room is away
    assert.deepEqual(heard.map(([event, msg]) => [event, msg?.role || msg?.clientMessageId || '']),
        [['chat', 'user'], ['chat', 'system'], ['chat', 'system'], ['outbound_sent', 'm1'], ['outbound_dropped', 'm2'], ['close', '']]);
    assert.equal(heard[2][1].task_incident, 'deadline');
    heard.length = 0;
    socket.resume();
    assert.equal(count(), 6);
    assert.deepEqual(heard, [['open', { previouslyConnected: true }]], 'back: one catch-up, as after a reconnect');
    ws.emit('log', { data: {} });
    assert.equal(heard.at(-1)[0], 'log');
    assert.equal(socket.send({ type: 'chat' }).status, 'sent');
    assert.equal(socket.ws.readyState, 1);
    // Away while the socket is down: back, it waits for the real open instead of claiming one. The socket's own
    // observation reaches the room only while it is back, as its other live facts do.
    socket.pause();
    connected = false;
    ws.serverAnswersWhileDown = () => true;
    assert.equal(socket.serverAnswersWhileDown(), false, 'away, the room claims nothing about the socket');
    heard.length = 0;
    socket.resume();
    assert.equal(socket.serverAnswersWhileDown(), true);
    assert.deepEqual(heard, []);
    ws.emit('open', { previouslyConnected: true });
    assert.deepEqual(heard.map(([event]) => event), ['open']);
});

test('a catch-up read still out after its bound is said once and unsaid when it settles; a room leaving says nothing', async () => {
    const wait = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
    let changes = 0;
    const catchUp = createCatchUp(() => { changes += 1; }, 20);
    catchUp.begin(() => true);
    catchUp.end();  // settled within the bound: no Updating at all
    await wait(40);
    assert.deepEqual([catchUp.active, changes], [false, 0]);
    catchUp.begin(() => true);
    await wait(40);
    assert.deepEqual([catchUp.active, changes], [true, 1], 'still out after the bound: said');
    catchUp.end();
    assert.deepEqual([catchUp.active, changes], [false, 2], 'settled: unsaid');
    catchUp.end();
    assert.equal(changes, 2, 'an end with nothing said changes nothing');
    catchUp.begin(() => false);  // the room left before the bound
    await wait(40);
    assert.deepEqual([catchUp.active, changes], [false, 2]);
    catchUp.begin(() => true);
    await wait(40);
    catchUp.cancel();  // leaving: its return reads, and says so, anew
    assert.deepEqual([catchUp.active, changes], [false, 3]);
    catchUp.begin(() => true);
    catchUp.cancel();
    await wait(40);
    assert.deepEqual([catchUp.active, changes], [false, 3], 'a cancelled bound never fires');
});
