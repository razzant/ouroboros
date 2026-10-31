// A Project room that is left keeps its content while it leaves (room_retire.js): out of the live instances, marked
// leaving and inert, its reads stopped; put away once its exit has run, back as the same room when reopened, and a
// room another one replaced meanwhile only destroys itself.
import assert from 'node:assert/strict';
import test from 'node:test';

import { createRoomRetirement } from '../modules/room_retire.js';

function room(hidden = false) {
    return { page: { hidden, inert: false, dataset: {} }, calls: [], leave() { this.calls.push('leave'); },
        stay() { this.calls.push('stay'); }, destroy() { this.calls.push('destroy'); } };
}

function world() {
    const instances = new Map();
    const destroyed = [], cancelled = [];
    const rooms = createRoomRetirement({ instances, cancelPaint: (pid) => cancelled.push(pid),
        destroy: (pid) => { destroyed.push(pid); instances.delete(pid); } });
    return { instances, destroyed, cancelled, rooms };
}

test('a room on screen leaves whole: out of the instances, inert and marked, its reads stopped', () => {
    const w = world();
    const r = room();
    w.instances.set('p1', r);
    w.rooms.retire('p1');
    assert.equal(w.instances.has('p1'), false);
    assert.equal(w.rooms.retiring.get('p1'), r);
    assert.deepEqual([r.page.inert, 'leaving' in r.page.dataset, r.calls], [true, true, ['leave']]);
    assert.deepEqual(w.cancelled, ['p1'], 'its history paint is cancelled');
    assert.deepEqual(w.destroyed, []);
    // Its exit has run: put away the way any room is.
    w.rooms.settle('p1', r);
    assert.deepEqual([r.page.inert, 'leaving' in r.page.dataset, r.calls], [false, false, ['leave', 'stay']]);
    assert.deepEqual(w.destroyed, ['p1']);
    assert.equal(w.rooms.retiring.size, 0);
});

test('a hidden room has nothing to show leaving: destroyed at once', () => {
    const w = world();
    w.instances.set('p1', room(true));
    w.rooms.retire('p1');
    assert.deepEqual(w.destroyed, ['p1']);
    assert.equal(w.rooms.retiring.size, 0);
});

test('reopened during its exit it is the same room; a late put-away of a replaced room destroys only itself', () => {
    const w = world();
    const r = room();
    w.instances.set('p1', r);
    w.rooms.retire('p1');
    w.rooms.settle('p1', undefined, false);  // reopened
    assert.equal(w.instances.get('p1'), r);
    assert.deepEqual(w.destroyed, []);
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
    assert.deepEqual([w.destroyed, r.page.inert, 'leaving' in r.page.dataset], [[], true, true]);
    w.rooms.settle(...second.slice(0, 2), true, second[2]);
    assert.deepEqual(w.destroyed, ['p1'], 'the second exit puts it away');
    assert.deepEqual(w.rooms.exits(), []);
});
