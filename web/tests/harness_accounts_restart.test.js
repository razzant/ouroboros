// Restart Claudexor (Accounts): the button, what its confirmation says, and the refusal of a
// preview that no longer holds. Each guard asserts the case that fires and the case that stays quiet.
import assert from 'node:assert/strict';
import test from 'node:test';

import {
    RESTART_LABEL,
    restartClaudexor,
    restartConfirmCopy,
    restartFactSentence,
    restartOffered,
    restartOutcomeNote,
} from '../modules/claudexor_restart.js';
import { RESTART_ENDPOINT, createClaudexorStatusStore } from '../modules/claudexor_status_store.js';
import {
    destroyHarnessAccounts,
    initHarnessAccounts,
} from '../modules/harness_accounts.js';

const fakeResponse = (status, body) => ({ ok: status >= 200 && status < 300, status, json: async () => body });
const STATUS_UP = {
    daemon: { state: 'running', engine_version: '5.0.1', runtime: {} },
    config_dir: '/home/agent', harnesses: [], profiles: {}, quota: [],
    reads: { catalog: 'ok', accounts: 'ok', quota: 'ok' },
};
const ENGINE = { state: 'serving', serving: { version: '5.0.1', build_sha: 'b'.repeat(40) },
    pin: { version: '5.0.1', build_sha: 'b'.repeat(40) }, direction: 'unchanged' };
const item = (run_id, owner, extra = {}) => ({ run_id, state: 'running', owner, task_id: owner === 'ouroboros' ? 'task-1' : '',
    kind: owner === 'ouroboros' ? 'subagent' : '', harness: 'codex', project: 'site', started_at: '2026-11-06T10:00:00Z', ...extra });

function preview({ total = 3, ouroboros = 2, other = 1, status = 'known', token = 'token-a', ...rest } = {}) {
    const items = status === 'known'
        ? [...Array(ouroboros || 0).keys()].map((n) => item(`own-${n}`, 'ouroboros'))
            .concat([...Array(other || 0).keys()].map((n) => item(`other-${n}`, 'other')))
        : [];
    return { engine: ENGINE, runs: { status, total, ouroboros, other, error: null, items, items_omitted: 0 },
        model_calls: { status: 'known', active: 1, queued: 0 }, warnings: [], token, ...rest };
}

const PIN_WARNING = { code: 'engine_pin_below_serving', pin_version: '4.9.0', serving_version: '5.0.1',
    message: 'This Ouroboros version pins Claudexor 4.9.0, older than the serving 5.0.1. Engine data only moves forward: '
        + 'after the restart 4.9.0 may refuse the data 5.0.1 wrote, and then Claudexor does not start.' };

test('the confirmation says one fact: how many runs end and whose, and an uncounted engine is never zero', () => {
    assert.equal(restartFactSentence(preview()),
        'Restarting Claudexor ends 3 runs: 2 of Ouroboros, 1 of other programs.');
    assert.equal(restartFactSentence(preview({ total: 1, ouroboros: 0, other: 1 })),
        'Restarting Claudexor ends 1 run: 0 of Ouroboros, 1 of other programs.');
    assert.equal(restartFactSentence(preview({ total: 0, ouroboros: 0, other: 0 })),
        'Claudexor has no runs right now, so restarting it ends none.');

    const silent = preview({ status: 'unknown', total: null, ouroboros: 2, other: null,
        engine: { state: 'not_answering', serving: null, pin: ENGINE.pin, direction: 'unknown' } });
    assert.equal(restartFactSentence(silent), 'Claudexor is not answering, so its runs could not be counted. '
        + "If it is running, restarting ends every run on it, other programs' included. "
        + "Ouroboros's own records hold 2 runs open.");
    const unlisted = restartFactSentence(preview({ status: 'unknown', total: null, ouroboros: null, other: null }));
    assert.match(unlisted, /^Claudexor did not list its runs\. If it is running, restarting ends every run on it/);
    for (const text of [restartFactSentence(silent), unlisted]) assert.doesNotMatch(text, /\b0 runs|ends none|no runs/i);
    assert.match(restartFactSentence(preview({ status: 'unattributed', total: 4, ouroboros: null, other: null })),
        /^Restarting Claudexor ends 4 runs\. Ouroboros could not read its own run records/);
    assert.equal(restartFactSentence(preview({ engine: { state: 'not_started' }, total: 0, ouroboros: 0, other: 0 })),
        'Claudexor has not been started yet, so no runs end. This starts it.');
});

test('the dialog adds the pin warning only when the server sent one, and keeps the rest under details', () => {
    const plain = restartConfirmCopy(preview());
    assert.equal(plain.title, 'Restart Claudexor?');
    assert.equal(plain.confirmLabel, 'Restart Claudexor');
    assert.equal(plain.body, 'Restarting Claudexor ends 3 runs: 2 of Ouroboros, 1 of other programs.');
    assert.equal(plain.danger, true, 'ending runs is the destructive tone');
    assert.doesNotMatch(plain.body, /pins|5\.0\.1|task-1/, 'versions and run ids are details, not the sentence');

    const warned = restartConfirmCopy(preview({ warnings: [PIN_WARNING] }));
    assert.equal(warned.body, `${plain.body} ${PIN_WARNING.message}`);

    const rows = Object.fromEntries(plain.details.rows.map((row) => [row.label, row.value]));
    assert.equal(plain.details.summary, 'Engine and runs');
    assert.equal(rows['Serving engine'], '5.0.1 (bbbbbbbbbbbb)');
    assert.equal(rows['Engine this Ouroboros starts'], '5.0.1 (bbbbbbbbbbbb)');
    assert.equal(rows['Model calls in flight'], '1 active, 0 queued; they end too');
    const named = plain.details.rows.filter((row) => /run$/.test(row.label));
    assert.deepEqual(named.map((row) => row.label), ['Ouroboros run', 'Ouroboros run', 'Other program’s run']);
    assert.match(named[0].value, /^codex · site · running since .+ · subagent · task task-1 · own-0$/);
    assert.match(named[2].value, /^codex · site · running since .+ · other-0$/, 'another program has no task of ours');

    const idle = restartConfirmCopy(preview({ total: 0, ouroboros: 0, other: 0 }));
    assert.equal(idle.danger, false, 'nothing ends: an ordinary confirmation');
    const many = restartConfirmCopy(preview({ total: 20, ouroboros: 20, other: 0 }));
    assert.equal(many.details.rows.at(-1).value, '8 runs not listed here');
    assert.equal(restartConfirmCopy(preview({ status: 'unknown', total: null, ouroboros: null, other: null })).danger, true);
});

test('the outcome note keeps the stop and the next start apart', () => {
    const runs = { status: 'known', total: 3, ouroboros: 2, other: 1 };
    assert.deepEqual(restartOutcomeNote({ stop: 'stopped', serving: true, error: null, runs }),
        { tone: 'ok', text: 'Claudexor restarted. 3 runs ended.' });
    assert.deepEqual(restartOutcomeNote({ stop: 'nothing_to_stop', serving: true, error: null, runs }),
        { tone: 'ok', text: 'Claudexor was not running. It is started now.' });
    assert.deepEqual(restartOutcomeNote({ stop: 'unconfirmed', serving: true, error: null, runs }),
        { tone: 'warn', text: 'Claudexor did not confirm that it stopped. An engine is answering; it may be the same process.' });
    assert.deepEqual(restartOutcomeNote({ stop: 'stopped', serving: false, error: 'daemon_spawn_failed: engine floor', runs }),
        { tone: 'error', text: 'Claudexor stopped and did not start again: daemon_spawn_failed: engine floor.' });
});

function restartStore(answers) {
    const requests = [];
    const store = createClaudexorStatusStore({
        fetchImpl: async (url, init = {}) => {
            requests.push({ method: init.method || 'GET', url, body: init.body ? JSON.parse(init.body) : null });
            if (url === RESTART_ENDPOINT && init.method === 'POST') return answers.shift();
            if (url.endsWith('/quota/refresh')) return fakeResponse(200, { snapshots: [], absences: [], refreshed_at: null });
            return fakeResponse(200, STATUS_UP);
        },
        doc: { hidden: false, addEventListener() {}, removeEventListener() {} },
    });
    const posts = () => requests.filter((r) => r.url === RESTART_ENDPOINT && r.method === 'POST').map((r) => r.body);
    return { store, posts };
}

test('a stale preview is refused and asked again with the fresh facts; nothing is sent without a confirmation', async () => {
    const fresh = preview({ total: 4, ouroboros: 2, other: 2, token: 'token-b' });
    const receipt = { stop: 'stopped', serving: true, error: null, runs: { status: 'known', total: 4, ouroboros: 2, other: 2 } };
    const { store, posts } = restartStore([
        fakeResponse(409, { error: 'What a restart would end has changed since it was shown.', code: 'restart_preview_stale', preview: fresh }),
        fakeResponse(200, { status: STATUS_UP, restart: receipt }),
    ]);
    const asked = [];
    const busy = [];
    try {
        const note = await restartClaudexor({ store, fetchImpl: async () => fakeResponse(200, preview()),
            dialogImpl: async (copy) => { asked.push(copy.body); return true; }, onBusy: (value) => busy.push(value) });
        assert.deepEqual(posts(), [{ token: 'token-a' }, { token: 'token-b' }], 'each POST echoes the preview on screen');
        assert.deepEqual(asked, [
            'Restarting Claudexor ends 3 runs: 2 of Ouroboros, 1 of other programs.',
            'This changed while the question was open. Restarting Claudexor ends 4 runs: 2 of Ouroboros, 2 of other programs.',
        ]);
        assert.deepEqual(note, { tone: 'ok', text: 'Claudexor restarted. 4 runs ended.' });
        assert.deepEqual(busy, [true, false, true, false]);
        assert.equal(store.snapshot.daemon.engine_version, '5.0.1', 'the store committed the status the answer carried');
    } finally {
        store.dispose();
    }

    // Quiet side: the owner declines the fresh facts, so the old consent restarts nothing.
    const second = restartStore([
        fakeResponse(409, { error: 'changed', code: 'restart_preview_stale', preview: fresh }),
        fakeResponse(200, { status: STATUS_UP, restart: receipt }),
    ]);
    const answers = [true, false];
    try {
        const note = await restartClaudexor({ store: second.store, fetchImpl: async () => fakeResponse(200, preview()),
            dialogImpl: async () => answers.shift() });
        assert.equal(note, null);
        assert.deepEqual(second.posts(), [{ token: 'token-a' }], 'the refused token was never resent');
    } finally {
        second.store.dispose();
    }

    const none = restartStore([]);
    try {
        // Only a strict `true` confirms: the dialog's cancel, close and supersession all resolve false.
        for (const answer of [false, undefined, { confirmed: true }]) {
            assert.equal(await restartClaudexor({ store: none.store, fetchImpl: async () => fakeResponse(200, preview()),
                dialogImpl: async () => answer }), null);
        }
        assert.deepEqual(none.posts(), []);
        const unread = await restartClaudexor({ store: none.store, fetchImpl: async () => fakeResponse(503, { error: 'offline' }),
            dialogImpl: async () => assert.fail('no dialog without a preview') });
        assert.deepEqual(unread, { tone: 'error', text: 'Could not check what a restart would end: offline. Nothing was stopped.' });
        assert.deepEqual(none.posts(), []);
    } finally {
        none.store.dispose();
    }
});

test('a typed refusal says nothing was restarted; a lost answer does not claim that', async () => {
    const typed = restartStore([fakeResponse(503, { error: 'owner_restart: The owning generation is restarting.', code: 'owner_restart' })]);
    const lost = restartStore([{ ok: false, status: 502, json: async () => { throw new Error('not json'); } }]);
    try {
        const deps = { fetchImpl: async () => fakeResponse(200, preview()), dialogImpl: async () => true };
        assert.deepEqual(await restartClaudexor({ store: typed.store, ...deps }),
            { tone: 'error', text: 'Claudexor was not restarted: owner_restart: The owning generation is restarting.' });
        assert.deepEqual(await restartClaudexor({ store: lost.store, ...deps }),
            { tone: 'error', text: 'The restart did not answer: HTTP 502. Claudexor may have stopped; its status is above.' });
    } finally {
        typed.store.dispose();
        lost.store.dispose();
    }
});

// ---- the button in the Accounts section ------------------------------------------------------

function fakeElement(id) {
    const el = { id, innerHTML: '', textContent: '', dataset: {}, offsetParent: {}, hidden: false, disabled: false,
        listeners: [], addEventListener(type, fn) { el.listeners.push([type, fn]); }, removeEventListener() {},
        querySelector: () => null, querySelectorAll: () => [], contains: () => false, closest: () => null };
    return el;
}

function mountSection() {
    const elements = {};
    for (const id of ['agents-service-banner', 'agents-service-version', 'harness-login-card', 'btn-harness-refresh',
        'btn-claudexor-restart', 'harness-accounts-error', 'harness-accounts-groups']) elements[id] = fakeElement(id);
    const doc = { hidden: false, activeElement: null, getElementById: (id) => elements[id] || null,
        addEventListener() {}, removeEventListener() {} };
    return { elements, doc, win: { addEventListener() {}, removeEventListener() {} } };
}

// The frame that holds the two buttons is the provider's page (web/tests/settings_accounts.test.js proves it there).
test('Restart Claudexor is offered, in words, once an owned engine was ever started', () => {
    assert.equal(RESTART_LABEL, 'Restart Claudexor');

    assert.equal(restartOffered(STATUS_UP), true);
    assert.equal(restartOffered({ daemon: { state: 'stale' } }), true, 'a sick or stopped engine is what the action is for');
    assert.equal(restartOffered({ daemon: { state: 'not_provisioned' } }), false);
    assert.equal(restartOffered({ daemon: { state: 'running', ownership_problem: 'foreign home' } }), false);
    assert.equal(restartOffered(null), false, 'not before this page read the engine');
});

test('a press previews, asks in the in-app dialog and restarts only on confirm', async () => {
    const { elements, doc, win } = mountSection();
    const prior = { document: globalThis.document, window: globalThis.window, confirm: globalThis.confirm };
    globalThis.document = doc;
    globalThis.window = win;
    globalThis.confirm = () => assert.fail('the native confirm is never used');
    const receipt = { stop: 'stopped', serving: true, error: null, runs: { status: 'known', total: 3, ouroboros: 2, other: 1 } };
    const { store, posts } = restartStore([fakeResponse(200, { status: STATUS_UP, restart: receipt })]);
    const asked = [];
    let answer = false;
    const restartDeps = {
        fetchImpl: async (url) => { assert.equal(url, RESTART_ENDPOINT); return fakeResponse(200, preview({ warnings: [PIN_WARNING] })); },
        dialogImpl: async (copy) => { asked.push(copy); return answer; },
    };
    const button = elements['btn-claudexor-restart'];
    const note = elements['harness-accounts-error'];
    try {
        assert.equal(await initHarnessAccounts({ store, restartDeps }), true);
        assert.equal(button.hidden, true, 'hidden until the engine was read');
        await store.refresh();
        assert.equal(button.hidden, false);
        assert.equal(button.textContent, 'Restart Claudexor');
        const click = button.listeners.filter(([type]) => type === 'click').map(([, fn]) => fn).at(-1);

        await click(); // the owner cancels
        assert.equal(asked.length, 1);
        assert.match(asked[0].body, /^Restarting Claudexor ends 3 runs: 2 of Ouroboros, 1 of other programs\. This Ouroboros version pins Claudexor 4\.9\.0/);
        assert.deepEqual(posts(), [], 'a cancelled question restarts nothing');
        assert.equal(note.hidden, true);

        answer = true;
        await click(); // the owner confirms
        assert.deepEqual(posts(), [{ token: 'token-a' }]);
        assert.equal(note.hidden, false);
        assert.equal(note.textContent, 'Claudexor restarted. 3 runs ended.');
        assert.equal(note.dataset.tone, 'ok');
        assert.equal(button.textContent, 'Restart Claudexor');
        assert.equal(button.disabled, false);
        assert.equal(elements['agents-service-banner'].textContent, 'Claudexor ready', 'the panel shows the status read after the restart');

        // The note is the section's last outcome: the owner's next action here retires it.
        const refresh = elements['btn-harness-refresh'].listeners.filter(([type]) => type === 'click').map(([, fn]) => fn).at(-1);
        await refresh();
        assert.equal(note.hidden, true);
    } finally {
        await destroyHarnessAccounts();
        store.dispose();
        Object.assign(globalThis, prior);
    }
});
