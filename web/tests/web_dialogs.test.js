// v6.90.3 dialog-class migration (owner decision Б2-2): pure decision helpers
// behind the openConfirmDialog sites, node-tested with injected dialogs. The
// class-wide ban on native prompt/confirm/alert lives in the python static
// test (tests/test_web_dialogs_static.py) so quick CI enforces it.

import assert from 'node:assert/strict';
import test from 'node:test';

import { promptUpdateVersion } from '../modules/marketplace.js';
import { promptCampaignObjective } from '../modules/evolution.js';
import { confirmAndSendPanic, shouldFirePanic } from '../modules/chat_activity.js';
import { chooseAndSendReview } from '../modules/review_command.js';
import { shouldPollStatus } from '../modules/claudexor_status_store.js';
import {
    JOB_POLL_GIVE_UP_FAILURES,
    JOB_POLL_MAX_DELAY_MS,
    nextJobPollDelay,
} from '../modules/harness_login_cards.js';
import { ALLOW_EMPTY_REVIEW_POOL, availableSubagentsSavePayload } from '../modules/subagents_settings.js';

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
// chat: /review names its executor (decision 3A): one enabled catalog row, else Main.
// ---------------------------------------------------------------------------

const REVIEW_ROSTER = { enabled: true, items: [
    { subagent_id: 'sol', name: 'stale label', effort: 'high', route: { kind: 'api_model', target_id: 'openai/gpt-5.6-sol' } },
    { subagent_id: 'off', enabled: false, route: { kind: 'agent_session', target_id: 'claude=claude-fable-5-1' } },
    { subagent_id: 'r1', effort: 'xhigh', route: { kind: 'agent_session', target_id: 'codex=gpt-6-astra' } },
    { subagent_id: 'r2', effort: 'xhigh', route: { kind: 'agent_session', target_id: 'codex=gpt-6-astra' } },
] };
const REVIEW_SETTINGS = { OUROBOROS_SUBAGENTS: JSON.stringify(REVIEW_ROSTER), OUROBOROS_PROCESSING_PREFERENCE: 'fast' };
const REVIEW_CHOICES = [
    { value: '', label: 'Main model (default)' },
    { value: 'sol', label: 'Subagent 1 — openai/gpt-5.6-sol/high/fast' },
    { value: 'r1', label: 'Subagent 3 — codex=gpt-6-astra/xhigh/fast~r1' },
    { value: 'r2', label: 'Subagent 4 — codex=gpt-6-astra/xhigh/fast~r2' },
];
const UNREAD_CATALOG = 'could not be read, so only Main is available';

test('/review labels each enabled row the way Settings and the model name it, and sends the stored id', async () => {
    const sent = [];
    const seen = [];
    const fired = await chooseAndSendReview({
        openConfirmDialog: async (options) => { seen.push(options); return { confirmed: true, value: 'r2' }; },
        ws: { send: (msg) => sent.push(msg) },
        readSettings: async () => REVIEW_SETTINGS,
    });
    assert.equal(fired, true);
    assert.deepEqual(sent, [{ type: 'command', cmd: '/review r2' }]);
    assert.equal(seen[0].input, true);
    assert.equal(seen[0].body, 'Who reviews the whole system?');
    // The ordinal is the card's ("Subagent N" counts switched-off rows too); twins
    // carry the roster's ~<stored id>; a stale `name` never labels a row.
    assert.deepEqual(seen[0].choices, REVIEW_CHOICES);
});

test('/review tells an unreadable catalog apart from an empty one; Main stays offered either way', async () => {
    const realFetch = globalThis.fetch;
    const open = async (readSettings) => {
        const seen = [];
        await chooseAndSendReview({
            ws: { send() {} }, readSettings,
            openConfirmDialog: async (options) => { seen.push(options); return false; },
        });
        return seen[0];
    };
    try {
        // The last two run the handler's real reader (no injected readSettings).
        for (const [why, readSettings, fetchImpl] of [
            ['a thrown read', async () => { throw new Error('offline'); }],
            ['an unparseable catalog', async () => ({ OUROBOROS_SUBAGENTS: '{"items": [' })],
            ['a catalog without rows', async () => ({ OUROBOROS_SUBAGENTS: { enabled: true } })],
            ['a refused /api/settings', undefined, async () => ({ ok: false, status: 503, json: async () => ({ error: 'down' }) })],
            ['a failed fetch', undefined, async () => { throw new TypeError('Failed to fetch'); }],
        ]) {
            globalThis.fetch = fetchImpl || realFetch;
            const dialog = await open(readSettings);
            assert.deepEqual(dialog.choices, [{ value: '', label: 'Main model (default)' }], why);
            assert.match(dialog.body, new RegExp(UNREAD_CATALOG), why);
        }
        for (const settings of [{}, { OUROBOROS_SUBAGENTS: '' }, { OUROBOROS_SUBAGENTS: { enabled: true, items: [] } }]) {
            const dialog = await open(async () => settings);
            assert.deepEqual(dialog.choices, [{ value: '', label: 'Main model (default)' }]);
            assert.equal(dialog.body, 'Who reviews the whole system?');
        }
        globalThis.fetch = async (url) => ({ ok: url === '/api/settings', status: 200, json: async () => REVIEW_SETTINGS });
        assert.deepEqual((await open(undefined)).choices, REVIEW_CHOICES);
    } finally {
        globalThis.fetch = realFetch;
    }
});

test('/review default sends the bare command (Main); cancel sends nothing', async () => {
    const sent = [];
    const ws = { send: (msg) => sent.push(msg) };
    const readSettings = async () => ({ OUROBOROS_SUBAGENTS: REVIEW_ROSTER });
    assert.equal(await chooseAndSendReview({
        ws, readSettings, openConfirmDialog: async () => ({ confirmed: true, value: '' }),
    }), true);
    assert.deepEqual(sent, [{ type: 'command', cmd: '/review' }]);
    for (const resolution of [false, null, { confirmed: false, value: 'sol' }]) {
        assert.equal(await chooseAndSendReview({ ws, readSettings, openConfirmDialog: async () => resolution }), false);
    }
    assert.equal(sent.length, 1);
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
// Review pool (#126 lineage): the save payload is honest about an empty pool.
// ---------------------------------------------------------------------------

const ROW = { subagent_id: 'r1', name: 'Reviewer one', recommended_use: '',
    route: { kind: 'api_model', target_id: 'openai::gpt-5.6-sol' } };

test('an unloaded or unparseable catalog never authors the subagent setting', () => {
    assert.deepEqual(availableSubagentsSavePayload({ loaded: false, setting: { enabled: true, items: [ROW] } }), {});
    assert.deepEqual(availableSubagentsSavePayload({ loaded: true, parseError: 'bad JSON',
        setting: { enabled: true, items: [ROW] } }), {});
});

test('a catalog with no reviewer SENDS the rows, so the server refusal surfaces instead of a silent success', () => {
    const payload = availableSubagentsSavePayload({ loaded: true, setting: { enabled: true, items: [ROW] } });
    assert.deepEqual(payload.OUROBOROS_SUBAGENTS.items.map((row) => row.subagent_id), ['r1']);
    assert.ok(!(ALLOW_EMPTY_REVIEW_POOL in payload));
    // Only the owner's explicit confirmation rides as a request flag, never inside the stored setting.
    const confirmed = availableSubagentsSavePayload({ loaded: true, allowEmptyReviewPool: true,
        setting: { enabled: true, items: [ROW] } });
    assert.equal(confirmed[ALLOW_EMPTY_REVIEW_POOL], true);
    assert.ok(!(ALLOW_EMPTY_REVIEW_POOL in confirmed.OUROBOROS_SUBAGENTS));
    const paused = availableSubagentsSavePayload({ loaded: true, allowEmptyReviewPool: true,
        setting: { enabled: true, items: [{ ...ROW, review_eligible: true, enabled: false }] } });
    assert.equal(paused[ALLOW_EMPTY_REVIEW_POOL], true, 'a pool of switched-off reviewers is empty too');
});

test('a stale confirmation never rides once a row is marked, nor on an empty catalog', () => {
    const marked = availableSubagentsSavePayload({ loaded: true, allowEmptyReviewPool: true,
        setting: { enabled: true, items: [{ ...ROW, review_eligible: true }] } });
    assert.equal(marked.OUROBOROS_SUBAGENTS.items[0].review_eligible, true);
    assert.ok(!(ALLOW_EMPTY_REVIEW_POOL in marked));
    const empty = availableSubagentsSavePayload({ loaded: true, allowEmptyReviewPool: true,
        setting: { enabled: true, items: [] } });
    assert.ok(!(ALLOW_EMPTY_REVIEW_POOL in empty));
});
