import test from 'node:test';
import assert from 'node:assert/strict';

import { readFileSync } from 'node:fs';

import { confirmAndSendRestart, restartConfirmBody } from '../modules/chat_activity.js';
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
    assert.deepEqual(sent[0][0], { type: 'command', cmd: '/restart', ui_control: { name: 'restart' } });
    assert.deepEqual(sent[0][1], { queue: false });
    assert.equal(seenOptions[0].danger, true);
    assert.equal(seenOptions[0].confirmLabel, 'Restart');
    // One line in the main view (U022); the policy is the dialog's Details, truthful about the server's owner
    // Restart (restart_retention.py): eligible saved work and the previously runnable queue return; existing pauses,
    // holds and limits are not released by Restart.
    assert.equal(seenOptions[0].body, 'Running tasks stop and continue after the restart.');
    const facts = seenOptions[0].details.rows.map((row) => `${row.label}: ${row.value}`).join('\n');
    assert.match(facts, /Running tasks: They stop, then eligible saved work resumes after the restart/);
    assert.match(facts, /already paused stay paused/);
    // What this door does that an Update does not: a sleeping task is held, and the tasks' own engine runs end.
    assert.match(facts, /Running tasks: [^\n]*Sleeping tasks wait for Resume\./);
    assert.match(facts, /Running tasks: [^\n]*Runs they started in Claudexor end\./);
    assert.match(facts, /Queued tasks: Runnable ones return to the queue/);
    assert.match(facts, /Existing holds and limits still apply/);
    // Live routing: resumed work that follows a role takes its current model; an explicit choice stays.
    assert.match(facts, /work that follows a model setting resumes on the current setting; task-specific choices stay\./);
    assert.doesNotMatch(`${seenOptions[0].body}\n${facts}`, /queued tasks stop|still pausing/);
});

test('Settings and the chat header share ONE restart confirmation (owner quiz 285597)', () => {
    const bar = readFileSync(new URL('../modules/chat_header.js', import.meta.url), 'utf8');
    const header = bar.slice(bar.indexOf("if (command === 'restart')"), bar.indexOf("if (command === 'panic')"));
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

const QUIET_RESTART_LINES = 1;  // the one line every confirmation says

test('Restart counts live inline Presence turns as interrupted, promising no return; nothing else adds the line', () => {
    const quiet = restartConfirmBody([]);
    assert.equal(quiet.split('\n').length, QUIET_RESTART_LINES);
    assert.doesNotMatch(quiet, /place conversation/);
    // Ordinary work reads as before: only a Presence row names this behaviour.
    for (const row of [{ phase: 'working', kind: 'direct_chat' }, { phase: 'working', kind: 'managed_task' }]) {
        assert.equal(restartConfirmBody([row]), quiet);
    }
    const one = restartConfirmBody([{ phase: 'working', kind: 'presence' }]);
    assert.notEqual(one, quiet);
    assert.ok(one.startsWith(quiet));
    assert.match(one, /^1 place conversation is mid-turn: the restart interrupts it and it does not resume by itself; a new message can start another turn\.$/m);
    // A turn still preparing is as live, and a second turn is counted, never a second line.
    const two = restartConfirmBody([{ phase: 'preparing', kind: 'presence' }, { phase: 'working', kind: 'presence' }]);
    assert.match(two, /^2 place conversations are mid-turn: the restart interrupts them and they do not resume by themselves;/m);
    assert.equal(two.split('\n').length, QUIET_RESTART_LINES + 1);
    // An unreadable census claims no count at all.
    assert.doesNotMatch(restartConfirmBody(null), /place conversation/);
    // Unknown and pausing work keep their own warnings beside it.
    const mixed = restartConfirmBody([{ phase: 'unknown', kind: 'presence' }, { phase: 'budget_pausing' }]);
    assert.match(mixed, /Pause status could not be read/);
    assert.match(mixed, /1 task is still pausing/);
    assert.match(mixed, /1 place conversation is mid-turn/);
    // A turn moved to a Project as text is saved and returned by the restart: no interruption warning for it,
    // while an unmoved one beside it is still counted.
    assert.equal(restartConfirmBody([{ phase: 'working', kind: 'presence', presence_pause: true }]), quiet);
    const beside = restartConfirmBody([{ phase: 'working', kind: 'presence', presence_pause: true },
        { phase: 'working', kind: 'presence' }]);
    assert.match(beside, /^1 place conversation is mid-turn/m);
    // Paused warm it stays that same work: its saved pause survives the restart, and no interruption is claimed.
    assert.equal(restartConfirmBody([{ phase: 'budget_paused', kind: 'presence', presence_pause: true }]), quiet);
});
