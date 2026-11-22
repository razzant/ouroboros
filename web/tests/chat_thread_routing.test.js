import assert from 'node:assert/strict';
import test from 'node:test';

import { chatLogThreadAccepts, chatThreadAccepts, mainLogFrameAccepts, mainThreadAccepts } from '../modules/chat_activity.js';

const known = new Set([60193030]);

test('main adopts its own and transport-shaped frames but rejects explicit panel zero', () => {
    assert.equal(mainThreadAccepts({ chat_id: 1 }, known), true);
    assert.equal(mainThreadAccepts({}, known), true);               // missing chat_id -> main
    assert.equal(mainThreadAccepts({ chat_id: 0 }, known), false);
    assert.equal(mainThreadAccepts({ chat_id: -1001 }, known), false);
    // External transport (Telegram-shaped big id) is never a registry member
    // and is never stamped: Main keeps adopting it.
    assert.equal(mainThreadAccepts({ chat_id: 197422551 }, known), true);
    assert.equal(mainThreadAccepts(null, known), true);
});

test('legacy absent log identity stays Main while explicit inner panel zero is rejected', () => {
    assert.equal(mainLogFrameAccepts({ chat_id: 0, data: { type: 'task_started' } }, known), true);
    assert.equal(mainLogFrameAccepts({ chat_id: 0, data: { type: 'review_reference', chat_id: 0 } }, known), false);
    assert.equal(mainLogFrameAccepts({ chat_id: 1, data: { type: 'review_reference', chat_id: 0 } }, known), false);
    assert.equal(mainLogFrameAccepts({ chat_id: 1, data: { type: 'task_started', chat_id: -1001 } }, known), false);
    assert.equal(mainLogFrameAccepts({ chat_id: -1001, data: { type: 'task_started' } }, known), false);
    assert.equal(mainLogFrameAccepts({ chat_id: 0, data: { type: 'task_started', chat_id: 1 } }, known), true);
    assert.equal(mainLogFrameAccepts({ project_thread: true, data: { chat_id: 1, project_thread: false } }, known), false);
});

test('main rejects frames of a project it already knows', () => {
    assert.equal(mainThreadAccepts({ chat_id: 60193030 }, known), false);
});

test('server stamp rejects a project frame the client has not learned yet', () => {
    // The race: fresh project, projectChatIds still empty (or stale).
    assert.equal(mainThreadAccepts({ chat_id: 77777777, project_thread: true }, new Set()), false);
    assert.equal(mainThreadAccepts({ chat_id: 77777777, project_thread: true }, null), false);
    assert.equal(mainThreadAccepts({ chat_id: 77777777 }, new Set()), true); // unstamped unknown id: unchanged behaviour
});

test('an accepted room echo reaches its primary room and explicit additional Projects', () => {
    const echo = { type: 'chat', role: 'user', source: 'web', ingress_accepted: true,
        kind: 'room_echo', chat_id: 7, also_rooms: [23] };
    for (const projects of [new Set(), new Set([41])]) {
        assert.equal(chatThreadAccepts(echo, true, 1, projects), false);
        assert.equal(chatThreadAccepts(echo, false, 7, projects), true);
        assert.equal(chatThreadAccepts(echo, false, 23, projects), true);
        assert.equal(chatThreadAccepts(echo, false, 41, projects), false);
        assert.equal(chatThreadAccepts({ ...echo, chat_id: 1 }, true, 1, projects), true);
    }
});

test('only an accepted web user room echo narrows the implicit Main mirror', () => {
    const echo = { type: 'chat', role: 'user', source: 'web', ingress_accepted: true, chat_id: 197422551 };
    assert.equal(mainThreadAccepts(echo, known), true, 'older web receipts keep their meaning');
    assert.equal(mainThreadAccepts({ ...echo, project_id: 'garden' }, known), true, 'a Project hint is not a stamp');
    for (const other of [{ source: 'skill:telegram' }, { ingress_accepted: false }, { role: 'system' }, { type: 'log' }]) {
        assert.equal(mainThreadAccepts({ ...echo, kind: 'room_echo', ...other }, known), true);
    }
});

test('stamp is authoritative regardless of chat_id shape', () => {
    assert.equal(mainThreadAccepts({ chat_id: 1, project_thread: true }, known), false);
    assert.equal(mainThreadAccepts({ project_thread: true }, known), false);
});

test('a Project instance also adopts the one frame of its work written to Main, and no other room does', () => {
    const notice = { chat_id: 1, also_rooms: [60193030], system_type: 'main_notice' };
    assert.equal(chatThreadAccepts(notice, false, 60193030, known), true);   // its work's Project
    assert.equal(chatThreadAccepts(notice, true, 1, known), true);           // Main, where it was written
    assert.equal(chatThreadAccepts(notice, false, 77777777, known), false);  // an unrelated Project
    assert.equal(chatThreadAccepts({ chat_id: 1, system_type: 'main_notice' }, false, 60193030, known), false);
    assert.equal(chatLogThreadAccepts({ chat_id: 1, data: { chat_id: 1, also_rooms: [60193030] } }, false, 60193030, known), true);
});
