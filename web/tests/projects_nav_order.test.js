import test from 'node:test';
import assert from 'node:assert/strict';
import { listOrder } from '../modules/projects_nav.js';

// The Projects list's order (docs/DESIGN.md "Navigation", owner 2026-11-07): the pins, then the newest message, the
// server's own order (projects_summary: time, then id, newest first) also where times are equal; unread is no rank.

const order = (rows) => [...rows].sort(listOrder).map((row) => row.id);

test('pins lead, then the newest message; equal times by id as the server lists them, unread no rank', () => {
    const archived = '2026-10-01T00:00:00Z';  // imported history keeps second-resolution times: equal times are real
    assert.deepEqual(order([
        { id: 'room-a', last_message_at: archived }, { id: 'room-c', last_message_at: archived },
        { id: 'pin', pinned: true, last_message_at: '2020-01-01T00:00:00Z' }, { id: 'room-b', last_message_at: archived },
        { id: 'newer', last_message_at: '2026-10-02T00:00:00Z' },
        { id: 'unread', last_message_at: '2026-09-01T00:00:00Z', _unread: true, visible_revision: 9 },
    ]), ['pin', 'newer', 'room-c', 'room-b', 'room-a', 'unread']);
});

test('an undated store orders by activity; rows with no time at all follow, by id as well', () => {
    assert.deepEqual(order([
        { id: 'b', last_active_at: '2026-01-02T00:00:00Z' }, { id: 'x' }, { id: 'a', last_active_at: '2026-01-03T00:00:00Z' },
        { id: 'y' },
    ]), ['a', 'b', 'y', 'x']);
});
