// Task-authored messages are visible where they land and where they were sent
// (owner 5=A / 7A): the receiving task's timeline names the sender by value,
// the sender's timeline shows the written/refused outcome with the host's
// reason, and both file under the task family so the Logs task card groups them.
import assert from 'node:assert/strict';
import test from 'node:test';

import { categorizeLogEvent, getLogTaskGroupId, isGroupedTaskEvent, summarizeChatLiveEvent, summarizeLogEvent } from '../modules/log_events.js';

test('a task message landing in a task names its sender in that task timeline', () => {
    const evt = { type: 'task_message_injected', task_id: 'receiver-1', source_task_id: 'sender-9', provenance: 'independent_task' };
    const view = summarizeLogEvent(evt);
    assert.equal(view.headline, 'Message from task sender-9');
    assert.ok(view.meta.includes('provenance=independent_task'));
    assert.equal(getLogTaskGroupId(evt), 'receiver-1');
    assert.equal(isGroupedTaskEvent(evt), true);
    assert.equal(categorizeLogEvent(evt, view), 'tasks');
});

test('a written task-authored message shows in the sender timeline with its target', () => {
    const evt = { type: 'task_message_routed', task_id: 'sender-9', target_task_id: 'receiver-1', status: 'written' };
    const view = summarizeLogEvent(evt);
    assert.equal(view.headline, 'Message sent to task receiver-1');
    assert.equal(view.phase, 'info');
    assert.ok(view.meta.includes('target=receiver-1'));
    assert.equal(getLogTaskGroupId(evt), 'sender-9');
});

test('a refused task-authored message carries the host reason and warns', () => {
    const evt = { type: 'task_message_routed', task_id: 'sender-9', target_task_id: 'gone-2', status: 'refused', reason: 'target_finished' };
    const view = summarizeLogEvent(evt);
    assert.equal(view.headline, 'Message to task gone-2 refused');
    assert.equal(view.phase, 'warn');
    assert.equal(view.body, 'target_finished');
    assert.equal(categorizeLogEvent(evt, view), 'tasks');
});


test('a task message landing in a task is a visible row in the receiver\'s chat block, with its preview', () => {
    const evt = { type: 'task_message_injected', task_id: 'receiver-1', source_task_id: 'sender-9',
        provenance: 'independent_task', text_preview: 'the PR is ready; please review it', ts: '2026-09-15T12:00:00Z' };
    const view = summarizeChatLiveEvent(evt);
    assert.equal(view.visible, true);
    assert.equal(view.headline, 'Message from task sender-9');
    assert.equal(view.body, 'the PR is ready; please review it');
    assert.ok(summarizeLogEvent(evt).body.includes('the PR is ready'));
});

