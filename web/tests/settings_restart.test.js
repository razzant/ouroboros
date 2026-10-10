import test from 'node:test';
import assert from 'node:assert/strict';

import { readFileSync } from 'node:fs';

import { confirmAndSendRestart } from '../modules/settings.js';
import { confirmAndSendRestart as sharedRestart, restartConfirmBody } from '../modules/chat_activity.js';
import { summarizeChatLiveEvent } from '../modules/log_events.js';

const noActivities = async () => [];

// #285 decision 16=A: the settings "Restart now" flow — confirm dialog, the
// exact /restart command, queue:false (a disconnected page must not queue a
// destructive command for a later reconnect), and honest outcomes.

test('Restart now confirm-and-send sends exactly one non-queueable /restart', async () => {
    const sent = [];
    const seenOptions = [];
    const outcome = await confirmAndSendRestart({
        openConfirmDialog: async (options) => {
            seenOptions.push(options);
            return true;
        },
        ws: { send: (msg, options) => { sent.push([msg, options]); return { status: 'sent' }; } },
        readActivities: noActivities,
    });
    assert.equal(outcome, 'sent');
    assert.equal(sent.length, 1);
    assert.deepEqual(sent[0][0], { type: 'command', cmd: '/restart' });
    assert.deepEqual(sent[0][1], { queue: false });
    assert.equal(seenOptions[0].danger, true);
    assert.equal(seenOptions[0].confirmLabel, 'Restart');
    // Truthful about the server's owner Restart (restart_retention.py):
    // eligible saved work and the previously runnable queue return; existing
    // pauses, holds and limits are not released by Restart.
    assert.match(seenOptions[0].body, /Running tasks stop, then eligible saved work resumes after the restart/);
    assert.match(seenOptions[0].body, /already paused stay paused/);
    assert.match(seenOptions[0].body, /Runnable queued tasks return to the queue/);
    assert.match(seenOptions[0].body, /Existing holds and limits still apply/);
    assert.doesNotMatch(seenOptions[0].body, /queued tasks stop/);
    assert.doesNotMatch(seenOptions[0].body, /still pausing/);
});

test('Settings and the chat header share ONE restart confirmation (owner quiz 285597)', () => {
    assert.equal(confirmAndSendRestart, sharedRestart);
    const chat = readFileSync(new URL('../modules/chat.js', import.meta.url), 'utf8');
    const header = chat.slice(chat.indexOf("if (command === 'restart')"), chat.indexOf("if (command === 'panic')"));
    // The header no longer sends /restart directly: it runs the same dialog.
    assert.match(header, /await confirmAndSendRestart\(\{ openConfirmDialog, ws \}\)/);
    assert.doesNotMatch(header, /ws\.send/);
});

test('Restart confirmation warns about every still-pausing task, and about an unread census', async () => {
    let body = '';
    const outcome = await confirmAndSendRestart({
        openConfirmDialog: async (options) => { body = options.body; return false; },
        ws: { send: () => ({ status: 'sent' }) },
        readActivities: async () => [{ phase: 'budget_pausing' }, { phase: 'pausing' }, { phase: 'working' }],
    });
    assert.equal(outcome, 'cancelled');
    assert.match(body, /2 tasks are still pausing: a pause not saved when the restart stops it is interrupted/);
    assert.match(restartConfirmBody([{ phase: 'budget_pausing' }]), /1 task is still pausing/);
    // A census that cannot be read never reads as "nothing is pausing".
    let unread = '';
    await confirmAndSendRestart({
        openConfirmDialog: async (options) => { unread = options.body; return false; },
        ws: { send: () => ({ status: 'sent' }) },
        readActivities: async () => { throw new Error('offline'); },
    });
    assert.match(unread, /Pause status could not be read/);
});

test('Restart now cancel sends nothing', async () => {
    for (const resolution of [false, undefined, null, 0]) {
        const sent = [];
        const outcome = await confirmAndSendRestart({
            openConfirmDialog: async () => resolution,
            ws: { send: (msg) => { sent.push(msg); return { status: 'sent' }; } },
            readActivities: noActivities,
        });
        assert.equal(outcome, 'cancelled');
        assert.deepEqual(sent, []);
    }
});

test('Restart now on a disconnected socket reports not_connected, never queues', async () => {
    const outcome = await confirmAndSendRestart({
        openConfirmDialog: async () => true,
        ws: { send: (_msg, options) => (options?.queue === false ? { status: 'failed' } : { status: 'queued' }) },
        readActivities: noActivities,
    });
    assert.equal(outcome, 'not_connected');
});

// #285 loud disclosure: the reload-failure event must be VISIBLE in the chat
// timeline (the default projection hides unknown types) with the honest story.
test('task_start_settings_reload_failed renders a visible warning row in chat', () => {
    const view = summarizeChatLiveEvent({
        type: 'task_start_settings_reload_failed',
        task_id: 't1',
        error: 'RuntimeError: settings.json unreadable',
    });
    assert.equal(view.visible, true);
    assert.equal(view.phase, 'warn');
    assert.match(view.headline, /Settings reload failed/);
    assert.match(view.body, /previously applied configuration/);
});

test('a per-task unknown pause is disclosed even in a returned census array', () => {
    assert.match(restartConfirmBody([{ phase: 'unknown' }]), /Pause status could not be read/);
});
