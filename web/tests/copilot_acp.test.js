import assert from 'node:assert/strict';
import test from 'node:test';
import { summarizeChatLiveEvent, summarizeLogEvent } from '../modules/log_events.js';
import { renderSettingsPage } from '../modules/settings_ui.js';

test('Copilot ACP uses the shared settings controls, not a model provider row', () => {
    const page = renderSettingsPage();
    for (const id of ['s-task-backend', 's-copilot-model', 's-copilot-bin', 's-copilot-permissions']) {
        assert.equal(page.split(`id="${id}"`).length - 1, 1);
        assert.ok(page.includes(`for="${id}"`));
    }
    assert.ok(page.includes('value="copilot_acp"'));
    assert.ok(page.includes('not an OS sandbox'));
});

test('ACP activity is nonterminal in the existing task card and full text is retained', () => {
    for (const kind of ['plan', 'diff', 'permission_request', 'permission_resolved', 'agent_message_chunk', 'error']) {
        const event = {
            type: 'task_runtime_update', execution_backend: 'copilot_acp',
            task_id: 'root', execution_id: 'run', sequence: 12,
            acp_update_type: kind, text: 'exact activity '.repeat(50),
            ...(kind === 'error' ? { level: 'error' } : {}),
        };
        const card = summarizeChatLiveEvent(event);
        assert.equal(card.visible, true);
        assert.equal(card.terminal, false);
        assert.equal(card.executorChip.label, 'Copilot ACP');
        assert.equal(card.fullBody, event.text.trim());
        assert.equal(summarizeLogEvent(event).phase, kind === 'error' ? 'error' : 'progress');
        const replay = summarizeChatLiveEvent({ ...event, type: 'send_message', is_progress: true });
        assert.equal(replay.executorChip.label, card.executorChip.label);
        assert.equal(replay.fullBody, card.fullBody);
    }
    assert.equal(summarizeChatLiveEvent({ type: 'task_runtime_protocol', sequence: 1 }).visible, false);
});

test('identical streamed text at different ACP sequence positions is not deduplicated', () => {
    const event = { type: 'task_runtime_update', task_id: 'root', text: 'same', acp_update_type: 'agent_message_chunk' };
    assert.notEqual(
        summarizeChatLiveEvent({ ...event, sequence: 1 }).dedupeKey,
        summarizeChatLiveEvent({ ...event, sequence: 2 }).dedupeKey,
    );
});

test('ACP tool metadata does not bypass the ordinary tool failure projection', () => {
    const event = {
        type: 'tool_call_finished', execution_backend: 'copilot_acp',
        acp_update_type: 'tool_call_update', task_id: 'root',
        tool: 'Copilot execute', is_error: true, result_preview: 'Check failed',
    };
    assert.equal(summarizeLogEvent(event).phase, 'error');
    assert.match(summarizeLogEvent(event).headline, /failed/);
});
