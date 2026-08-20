// v6.90.3 dialog-class migration (owner decision Б2-2): pure decision helpers
// behind the openConfirmDialog sites, node-tested with injected dialogs. The
// class-wide ban on native prompt/confirm/alert lives in the python static
// test (tests/test_web_dialogs_static.py) so quick CI enforces it.

import assert from 'node:assert/strict';
import test from 'node:test';

import { promptUpdateVersion } from '../modules/marketplace.js';
import { promptCampaignObjective } from '../modules/evolution.js';
import { confirmAndSendPanic, shouldFirePanic } from '../modules/chat_controls.js';
import { shouldPollStatus } from '../modules/claudexor_status_store.js';
import {
    JOB_POLL_GIVE_UP_FAILURES,
    JOB_POLL_MAX_DELAY_MS,
    nextJobPollDelay,
} from '../modules/harness_login_cards.js';
import { reviewerSlotsSavePayload } from '../modules/reviewer_slots.js';

// ---------------------------------------------------------------------------
// marketplace: update-to-version prompt (window.prompt was dead on desktop).
// ---------------------------------------------------------------------------

test('promptUpdateVersion preserves the prompt() contract: cancel skips, empty means latest', async () => {
    const calls = [];
    const confirmed = await promptUpdateVersion('my-skill', '2.1.0', {
        dialogImpl: async (options) => { calls.push(options); return { confirmed: true, value: ' 2.0.0 ' }; },
    });
    assert.deepEqual(confirmed, { confirmed: true, version: '2.0.0' });
    assert.equal(calls.length, 1);
    assert.equal(calls[0].input, true);
    // The latest version is pre-filled exactly as window.prompt pre-filled it.
    assert.equal(calls[0].initialValue, '2.1.0');
    assert.ok(calls[0].body.includes('Leave empty for latest (2.1.0)'));

    // Confirmed-empty = latest (the caller then POSTs {} — server contract).
    const latest = await promptUpdateVersion('my-skill', '2.1.0', {
        dialogImpl: async () => ({ confirmed: true, value: '' }),
    });
    assert.deepEqual(latest, { confirmed: true, version: '' });

    // Cancel (and Escape/backdrop — the dialog resolves the same shape) skips.
    const cancelled = await promptUpdateVersion('my-skill', '2.1.0', {
        dialogImpl: async () => ({ confirmed: false, value: '2.0.0' }),
    });
    assert.deepEqual(cancelled, { confirmed: false, version: '' });

    // An unknown latest is said, not rendered as an empty parenthesis.
    const unknown = [];
    await promptUpdateVersion('my-skill', '', {
        dialogImpl: async (options) => { unknown.push(options); return { confirmed: false }; },
    });
    assert.ok(unknown[0].body.includes('latest (unknown)'));
    assert.equal(unknown[0].initialValue, '');
});

// ---------------------------------------------------------------------------
// evolution: campaign objective (cancel = do NOT start, owner-approved Б1-8).
// ---------------------------------------------------------------------------

test('promptCampaignObjective: cancel starts nothing, confirmed-empty starts the default objective', async () => {
    // The old `window.prompt(...) || ''` flow started a PAID campaign even on
    // Cancel (null coerced to ''). Cancel is now a real decision.
    const cancelled = await promptCampaignObjective({
        dialogImpl: async () => ({ confirmed: false, value: 'ignored' }),
    });
    assert.deepEqual(cancelled, { confirmed: false, objective: '' });

    // Confirmed-empty preserves the contract: /evolve on with no text lets the
    // backend pick its default autonomous objective.
    const empty = await promptCampaignObjective({
        dialogImpl: async () => ({ confirmed: true, value: '   ' }),
    });
    assert.deepEqual(empty, { confirmed: true, objective: '' });

    const typed = await promptCampaignObjective({
        dialogImpl: async () => ({ confirmed: true, value: '  improve review latency  ' }),
    });
    assert.deepEqual(typed, { confirmed: true, objective: 'improve review latency' });

    // A dialog that never resolves a shape (glitch) reads as cancel, not start.
    const glitch = await promptCampaignObjective({ dialogImpl: async () => undefined });
    assert.deepEqual(glitch, { confirmed: false, objective: '' });
});

// ---------------------------------------------------------------------------
// chat: /panic — CRITICAL CONTROL. Fires on explicit boolean true, only.
// ---------------------------------------------------------------------------

test('/panic fires on an explicit confirm and on NOTHING else', () => {
    assert.equal(shouldFirePanic(true), true);
    // Cancel, backdrop, and Escape all resolve false from openConfirmDialog.
    assert.equal(shouldFirePanic(false), false);
    // Defensive strictness: a dialog API drift toward object results (the
    // input mode's {confirmed, value} shape), or any truthy junk, must NOT
    // kill all workers.
    assert.equal(shouldFirePanic({ confirmed: true, value: '' }), false);
    assert.equal(shouldFirePanic('true'), false);
    assert.equal(shouldFirePanic(1), false);
    assert.equal(shouldFirePanic(undefined), false);
    assert.equal(shouldFirePanic(null), false);
});

test('/panic confirm-and-send flow sends exactly one /panic command', async () => {
    // Consumer-level coverage (adversarial round 3): this is the REAL flow the
    // header action runs — confirmAndSendPanic({openConfirmDialog, ws}) — not
    // just the boolean gate. A broken await, option drift, or command typo
    // fails here instead of leaving the live Panic button silently inert.
    const sent = [];
    const seenOptions = [];
    const fired = await confirmAndSendPanic({
        openConfirmDialog: async (options) => {
            seenOptions.push(options);
            return true;
        },
        ws: { send: (msg) => sent.push(msg) },
    });

    assert.equal(fired, true);
    assert.deepEqual(sent, [{ type: 'command', cmd: '/panic' }]);
    assert.equal(seenOptions.length, 1);
    assert.equal(seenOptions[0].danger, true);
    assert.match(seenOptions[0].title, /Panic/);
    assert.equal(seenOptions[0].confirmLabel, 'Kill all workers');
});

test('/panic cancel/backdrop/Escape resolutions send NOTHING', async () => {
    for (const resolution of [false, undefined, null, 'true', 1, { confirmed: true, value: '' }]) {
        const sent = [];
        const fired = await confirmAndSendPanic({
            openConfirmDialog: async () => resolution,
            ws: { send: (msg) => sent.push(msg) },
        });
        assert.equal(fired, false, `resolution ${JSON.stringify(resolution)} must not fire`);
        assert.deepEqual(sent, [], `resolution ${JSON.stringify(resolution)} must send nothing`);
    }
});

// ---------------------------------------------------------------------------
// claudexor status store (#125, phase 2): polling gate + job-poll pacing.
// ---------------------------------------------------------------------------

test('the status POLL gate needs a subscriber, a visible page, and a reason', () => {
    // Visible surface, tab shown, someone listening: polls.
    assert.equal(shouldPollStatus({ hasSubscribers: true, surfaceVisible: true }), true);
    // Nobody listening: never — the old app-lifetime interval ran on every page.
    assert.equal(shouldPollStatus({ hasSubscribers: false, surfaceVisible: true }), false);
    // Listening but the surface is on another page/subtab: no reason to poll.
    assert.equal(shouldPollStatus({ hasSubscribers: true, surfaceVisible: false }), false);
    // A live login job HOLDS the poll open even off-surface…
    assert.equal(shouldPollStatus({ hasSubscribers: true, held: true }), true);
    // …but a hidden browser tab pauses every reason, the hold included.
    assert.equal(shouldPollStatus({ hasSubscribers: true, surfaceVisible: true, hidden: true }), false);
    assert.equal(shouldPollStatus({ hasSubscribers: true, held: true, hidden: true }), false);
    assert.equal(shouldPollStatus({}), false);
});

test('job polling backs off on consecutive failures and gives up at the bound', () => {
    // Healthy: the plain 3s cadence.
    assert.deepEqual(nextJobPollDelay(0), { delayMs: 3000, giveUp: false });
    // Failures: 6/12/24/30…s, capped.
    assert.deepEqual(nextJobPollDelay(1), { delayMs: 6000, giveUp: false });
    assert.deepEqual(nextJobPollDelay(2), { delayMs: 12000, giveUp: false });
    assert.deepEqual(nextJobPollDelay(3), { delayMs: 24000, giveUp: false });
    assert.deepEqual(nextJobPollDelay(4), { delayMs: JOB_POLL_MAX_DELAY_MS, giveUp: false });
    assert.deepEqual(nextJobPollDelay(9), { delayMs: JOB_POLL_MAX_DELAY_MS, giveUp: false });
    // The 10th consecutive failure stops the chain (honest "unconfirmed"
    // verdict — the sign-in may still have completed; never active.error,
    // whose Try-again would start a second login beside a live jobId).
    assert.deepEqual(nextJobPollDelay(JOB_POLL_GIVE_UP_FAILURES), { delayMs: 0, giveUp: true });
    assert.deepEqual(nextJobPollDelay(JOB_POLL_GIVE_UP_FAILURES + 5), { delayMs: 0, giveUp: true });
    // Junk input degrades to the healthy cadence, never to give-up.
    assert.deepEqual(nextJobPollDelay(-3), { delayMs: 3000, giveUp: false });
    assert.deepEqual(nextJobPollDelay(NaN), { delayMs: 3000, giveUp: false });
});

// ---------------------------------------------------------------------------
// reviewer_slots (#126): the save payload is honest about an empty set.
// ---------------------------------------------------------------------------

test('an unloaded or unreachable view never authors the reviewer-slot setting', () => {
    assert.deepEqual(reviewerSlotsSavePayload({ loaded: false }), {});
    assert.deepEqual(reviewerSlotsSavePayload({
        loaded: false, loadError: 'could not load reviewer slots: HTTP 502',
    }), {});
    // A load error outranks loaded=true (a stale flag from a previous load).
    assert.deepEqual(reviewerSlotsSavePayload({
        loaded: true, loadError: 'network gone', triad: [], scope: [],
    }), {});
});

test('a LOADED empty set SENDS the key, so the backend 400 surfaces instead of a silent success', () => {
    // The old guard returned {} for an empty triad/scope, so deleting every
    // row reported "Settings saved" while saving nothing. The key now rides
    // with triad:[] and the backend's «triad needs at least one slot» 400
    // lands in the existing failed-save status. No client-side duplicate
    // validation — the backend stays the SSOT.
    const payload = reviewerSlotsSavePayload({
        loaded: true, loadError: '', triad: [], scope: [],
        advisory: { enabled: true, route: { kind: 'api', target_id: '' }, effort: 'low' },
    });
    assert.ok('OUROBOROS_REVIEWER_SLOTS' in payload);
    const parsed = JSON.parse(payload.OUROBOROS_REVIEWER_SLOTS);
    assert.deepEqual(parsed.triad, []);
    assert.deepEqual(parsed.scope, []);
    assert.equal(parsed.advisory.enabled, true);
});

test('a loaded populated set serializes exactly as the setting builder emits it', () => {
    const payload = reviewerSlotsSavePayload({
        loaded: true,
        loadError: '',
        triad: [{ slot_id: 't_1', route: { kind: 'api_chat', target_id: 'openai/gpt-5.6-sol' }, effort: 'high' }],
        scope: [{ slot_id: 's_1', route: { kind: 'agent_session', target_id: 'codex=gpt-5.6-sol', profile_id: 'koshak' }, effort: '' }],
        advisory: { enabled: false, route: { kind: 'api', target_id: 'sonnet' }, effort: 'low' },
    });
    const parsed = JSON.parse(payload.OUROBOROS_REVIEWER_SLOTS);
    assert.deepEqual(parsed.triad, [{
        slot_id: 't_1', route: { kind: 'api_chat', target_id: 'openai/gpt-5.6-sol' }, effort: 'high',
    }]);
    assert.equal(parsed.scope[0].route.profile_id, 'koshak');
    assert.equal('effort' in parsed.scope[0], false);
    assert.equal(parsed.advisory.enabled, false);
});
