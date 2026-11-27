// An agent's note delivered to a running reader names its declared program author as data,
// never the owner and never «Message from task unknown» (owner_mailbox stamps agent_note).
import assert from 'node:assert/strict';
import test from 'node:test';

import { agentNoteSignature, summarizeChatLiveEvent, summarizeLogEvent } from '../modules/log_events.js';
import { createChatInstance } from '../modules/chat.js';
import { installDom, PREVIEW_SCOPE, restoreDom } from './chat_dom_fixture.js';

const note = {
    type: 'task_message_injected', task_id: 'receiver-1', source_task_id: '', provenance: 'agent_note',
    agent_note: { author: { kind: 'agent', label: 'Build agent', basis: 'self_declared' }, row_address: 'row:1@t#abc' },
    text_preview: 'the example comparison is ready', ts: '2026-11-23T00:00:00Z',
};

test('an agent note names its declared author in both the log and the chat block', () => {
    assert.equal(agentNoteSignature(note), 'Agent note from agent "Build agent" (declared, not verified)');
    assert.equal(summarizeLogEvent(note).headline, 'Agent note from agent "Build agent" (declared, not verified)');
    const chat = summarizeChatLiveEvent(note);
    assert.equal(chat.visible, true);
    assert.equal(chat.headline, 'Agent note from agent "Build agent" (declared, not verified)');
    assert.equal(chat.body, 'the example comparison is ready');
    assert.ok(!chat.headline.includes('unknown'), 'never «Message from task unknown»');
});

test('a note with no recorded label still reads as a declared agent, not the owner', () => {
    const anonymous = { ...note, agent_note: { author: { kind: 'agent', label: '', basis: 'self_declared' } } };
    assert.equal(summarizeLogEvent(anonymous).headline, 'Agent note from agent "not recorded" (declared, not verified)');
});

test('a host system message and a real task sender keep their own readings', () => {
    const host = { type: 'task_message_injected', task_id: 'receiver-1', source_task_id: '', provenance: 'system' };
    assert.equal(summarizeLogEvent(host).headline, 'System message');
    assert.equal(summarizeChatLiveEvent(host).headline, 'System message');
    const task = { type: 'task_message_injected', task_id: 'receiver-1', source_task_id: 'sender-9', provenance: 'independent_task' };
    assert.equal(summarizeLogEvent(task).headline, 'Message from task sender-9');
    assert.equal(agentNoteSignature(task), '');
});

test('composer recalls the human input, not a recorded program message', async () => {
    const messages = [
        { role: 'user', text: 'Could you check the table?', history_id: 'chat:1',
            ts: '2026-11-23T00:00:00Z', history_position: { source: 'chat', offset: 1 } },
        { role: 'user', text: 'The table has six rows.', declared_author: note.agent_note.author,
            history_id: 'chat:2', ts: '2026-11-23T00:01:00Z', history_position: { source: 'chat', offset: 2 } },
    ]; // historical rows may have the user skin and a separate program signature
    const env = installDom(async (url) => ({ ok: true, json: async () => String(url).startsWith('/api/chat/history')
        ? { profile_scope: PREVIEW_SCOPE, messages, window: { complete: true } }
        : { active_chat_activities: [], active_chat_activities_complete: true, supervisor_ready: true } }));
    let generation = 0;
    const instance = createChatInstance({ mountEl: env.mount, chatId: 1, idPrefix: 'chat',
        ws: { on: () => () => {}, isConnected: () => true },
        state: { activePage: 'chat', projectChatIds: new Set(), unreadCount: 0 }, updateUnreadBadge() {},
        stateSnapshots: { begin: () => ({ generation: ++generation, requestedAt: Date.now() }),
            gate() { return Promise.resolve(this.begin()); }, isCurrent: () => true, apply() {}, fail() {}, latest: () => null },
    });
    try {
        await instance.refreshHistory({ revision: 1 });
        assert.equal(instance.hasPaintedHistory(), true, 'the fixture must load the room history before recall');
        const field = instance.composer.input;
        field.setSelectionRange = (start, end) => { field.selectionStart = start; field.selectionEnd = end; };
        field.setSelectionRange(0, 0);
        for (const listener of field.listeners.get('keydown')) listener({ key: 'ArrowUp', target: field, preventDefault() {} });
        assert.equal(field.value, 'Could you check the table?');
        assert.deepEqual(JSON.parse(sessionStorage.getItem('ouro_chat_input_history')), ['Could you check the table?']);
    } finally { instance.destroy(); restoreDom(env.prior); }
});