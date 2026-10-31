import test from 'node:test';
import assert from 'node:assert/strict';
import { classifyLiveFrame } from '../modules/notifications.js';

for (const kind of ['log', 'chat']) {
    test(`${kind} completion rings only when the run is not effectively quiet`, () => {
        const frame = {
            type: 'task_done', system_type: 'task_summary', role: 'system',
            task_id: 'scheduled-check', root_task_id: 'scheduled-check',
            chat_id: 1000001, status: 'completed', content: 'Done · no message',
        };
        assert.equal(classifyLiveFrame({ ...frame, wake_silent: true, quiet_effective: true }, { kind }), null);
        for (const flags of [{}, { wake_silent: false, quiet_effective: false },
            { wake_silent: true, quiet_effective: false }]) {
            assert.equal(classifyLiveFrame({ ...frame, ...flags }, { kind })?.category, 'task_done');
        }
    });
}

test('a quiet finish does not suppress an earlier proactive message', () => {
    assert.equal(classifyLiveFrame({
        role: 'system', system_type: 'proactive_message', chat_id: 1,
        task_id: 'scheduled-check', wake_silent: true, quiet_effective: true,
        content: 'A finding worth sharing.',
    }, { kind: 'chat' })?.category, 'important');
});
