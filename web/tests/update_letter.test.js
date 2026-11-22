// The update description: Ouroboros's own account of the pending official
// update, written as ordinary work (the owner's Check and describe, Write or
// Rewrite, or its own initiative), delivered inside the ordinary status payload
// and KEPT after the update lands.
//
// Two kinds of assertion live here, for two different failure modes. The
// projector cases pin the pure function (what the panel decides to say);
// the source pins guard the facts a pure test cannot see — where the section
// sits in the card, that the markdown pipeline is the sanitizing one, that
// its disposer runs before every re-render, and which request each control
// sends beside the authored body.

import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';

import { attemptTime } from '../modules/update_status.js';
import { headlineAction, requestFailureText, updateLetterView, updateVerdict } from '../modules/updates.js';

const SOURCE = readFileSync(new URL('../modules/updates.js', import.meta.url), 'utf8')
    .replace(/\r\n?/g, '\n');

const CURRENT = {
    managed: true,
    check_ok: true,
    available: false,
    current_version: '6.114.0',
    current_short_sha: 'abcd1234',
    current_sha: 'b'.repeat(40), running_sha: 'b'.repeat(40),
    checked_target_sha: 'b'.repeat(40),
};
const AVAILABLE = {
    managed: true,
    check_ok: true,
    available: true,
    safe_to_apply: true,
    current_version: '6.113.5',
    current_short_sha: 'abcd1234',
    latest_version: '6.114.0',
    latest_short_sha: 'ef567890',
    current_sha: 'a'.repeat(40), running_sha: 'a'.repeat(40),
    latest_sha: 'b'.repeat(40),
};

function letter(overrides = {}) {
    return {
        state: 'ready',
        relation: 'pending',
        description_current: true,
        text: 'This update makes the Updates panel explain itself.',
        author_version: '6.113.5',
        target_version: '6.114.0',
        written_at: new Date(Date.now() - 3 * 3600 * 1000).toISOString(),
        error_kind: '',
        error_text: '',
        key: { base_sha: 'a'.repeat(40), target_sha: 'b'.repeat(40), update_channel: 'stable', target_ref: 'managed/main' },
        has_last_good: false,
        ...overrides,
    };
}


const REQUEST_KEY = { base_sha: 'a'.repeat(40), target_sha: 'b'.repeat(40) };
const DESCRIBE = { id: 'describe', label: 'Check and describe', disabled: false };

test('a payload without a description: hidden with nothing on offer, Check and describe when an update is', () => {
    for (const data of [CURRENT, { ...CURRENT, letter: null }, { ...CURRENT, letter: 'nope' }]) {
        const view = updateLetterView(data, '');
        assert.equal(view.state, 'none');
        assert.equal(view.markdown, '');
        assert.equal(view.action, null);
        assert.equal(view.rewrite, null);
    }
    assert.equal(updateLetterView().state, 'none');
    // An update on offer with nothing written yet keeps the section, with its one control: the server checks and
    // asks for the range that check finds, so the press names no range of its own.
    for (const letterPart of [null, { state: 'none', text: '', checked_key: REQUEST_KEY, revision: 0 }]) {
        const view = updateLetterView({ ...AVAILABLE, letter: letterPart }, '');
        assert.equal(view.state, 'empty');
        assert.equal(view.label, "What's new");
        assert.equal(view.note, 'No description of this update yet.');
        assert.deepEqual(view.action, DESCRIBE);
        assert.equal(view.rewrite, null);
        assert.deepEqual(view.writeKey, { ...(letterPart?.checked_key || {}), ...REQUEST_KEY });
        assert.equal(view.covers, '');
    }
});


test('a pending description is "What\'s new" with its provenance, its range and a quiet Rewrite', () => {
    const view = updateLetterView({ ...AVAILABLE, letter: letter() }, '');
    assert.equal(view.state, 'ready');
    assert.equal(view.relation, 'pending');
    assert.equal(view.label, "What's new");
    assert.match(view.markdown, /explain itself/);
    assert.equal(view.note, '');
    assert.equal(view.failure, null);
    assert.equal(view.meta.authorVersion, '6.113.5');
    assert.equal(view.meta.targetVersion, '6.114.0');
    assert.equal(view.covers, `${'a'.repeat(8)} → ${'b'.repeat(8)}`);
    // Check and describe stays the one action (the server reuses this text unless the channel moved on);
    // asking again for this very range is Rewrite, in Details.
    assert.deepEqual(view.action, DESCRIBE);
    assert.equal(view.rewrite.mode, 'rewrite');
    assert.equal(view.rewrite.label, 'Rewrite');
    assert.equal(view.rewrite.key.target_sha, 'b'.repeat(40));
    // Same four buckets as the action row's "checked N ago" — one vocabulary.
    assert.equal(view.meta.ageText, '3 h ago');
    assert.equal(
        updateVerdict({ ...AVAILABLE, letter: letter() }, '').checkedAgo,
        '',
        'the description age is not the check age',
    );
});


test('an applied description is relabelled, never deleted, and may be rephrased for its own range', () => {
    const view = updateLetterView({ ...CURRENT, letter: letter({ relation: 'applied', description_current: false }) }, '');
    assert.equal(view.state, 'ready');
    assert.equal(view.label, 'What changed in this version');
    assert.equal(view.note, '', 'the running version IS the target: nothing to disclaim');
    assert.match(view.markdown, /explain itself/);
    assert.deepEqual(view.rewrite.key, letter().key, 'Rewrite names the text\'s own range, never the empty one');
    assert.equal(view.rewrite.mode, 'rewrite');
    assert.equal(view.action, null, 'nothing on offer and nothing retained: the headline\'s own check is the next step');
    // A kept description about a version this one has moved past keeps its provenance in Details.
    const older = updateLetterView({ ...CURRENT, current_version: '6.115.0', letter: letter({ relation: 'applied' }) }, '');
    assert.equal(older.meta.targetVersion, '6.114.0');
});


test('a newer target keeps the old text with its own range and offers Check and describe for the new one', () => {
    const view = updateLetterView({
        ...AVAILABLE,
        latest_version: '6.115.0', latest_sha: 'c'.repeat(40),
        letter: letter({ relation: 'superseded', description_current: false,
            checked_key: { base_sha: 'a'.repeat(40), target_sha: 'c'.repeat(40) } }),
    }, '');
    assert.equal(view.state, 'ready');
    // The heading answers which text this is before it is read: not the account of the update on offer, for which
    // nothing is being written (no request is held).
    assert.equal(view.label, 'Earlier update');
    assert.equal(view.note, `This description covers an earlier update (${'a'.repeat(8)} → ${'b'.repeat(8)}). `
        + 'No request to describe the available update is open.');
    assert.equal(view.covers, `${'a'.repeat(8)} → ${'b'.repeat(8)}`);
    assert.match(view.markdown, /explain itself/);
    assert.deepEqual(view.action, DESCRIBE);
    assert.equal(view.writeKey.target_sha, 'c'.repeat(40));
    // Asking again for the shown text keeps its own, earlier range beside the update on offer.
    assert.deepEqual(view.rewrite, { mode: 'rewrite', key: letter().key, label: 'Rewrite', disabled: false });
});


test('a same-range post-merge change never makes the description stale', () => {
    // The server compares ranges only; new qualification rides beside the text.
    const view = updateLetterView({ ...AVAILABLE, letter: letter({
        qualification: { ...REQUEST_KEY, github: { status: 'observed', runs: [{ sha: 'b'.repeat(40), conclusion: 'failure' }] } },
    }) }, '');
    assert.equal(view.descriptionCurrent, true);
    assert.equal(view.note, '');
});


test('a description whose HEAD moved elsewhere is marked, and an unnamed relation lands there too', () => {
    const moved = updateLetterView({ ...CURRENT, letter: letter({ relation: 'other', description_current: false }) }, '');
    assert.equal(moved.relation, 'other');
    assert.equal(moved.label, 'Earlier update', 'the heading says what the note says');
    assert.equal(moved.note, 'This description was written for an earlier update.');
    // A relation this client does not know is treated as the honest "other".
    const unnamed = updateLetterView({ ...CURRENT, letter: letter({ relation: 'sideways', description_current: false }) }, '');
    assert.equal(unnamed.relation, 'other');
    assert.equal(unnamed.label, 'Earlier update');
    assert.equal(unnamed.note, 'This description was written for an earlier update.');
});


test('a held request turns the action into a check that writes nothing and says what its task really does', () => {
    const CHECK = { id: 'check', label: 'Check', disabled: false };
    // Only a running task with no Stop being carried out writes: one waiting to run may be queued, paused or held (its
    // door shows which), an interrupted one says only that, and a Stop is the host's pending cancel beside the honest
    // status (or a legacy cancel_requested record).
    const states = [['requested', {}, 'is waiting to run'], ['scheduled', {}, 'is waiting to run'], ['running', {}, ''],
        ['interrupted', {}, 'was interrupted'], ['cancel_requested', {}, 'has been asked to stop'],
        ['running', { cancel_state: 'pending' }, 'has been asked to stop'],
        ['scheduled', { cancel_state: 'pending' }, 'has been asked to stop']];
    for (const [status, stop, state] of states) {
        const request = { task_id: 'd3adb33f', mode: 'rewrite', key: REQUEST_KEY, status, saved: false, ...stop };
        const view = updateLetterView({ ...AVAILABLE, letter: letter({ request }) }, '');
        assert.deepEqual(view.action, CHECK, status);
        assert.equal(view.rewrite, null, 'nothing else is asked while its task can still save');
        const line = state ? `The description is not saved yet; its task ${state} (task d3adb33f).`
            : 'Ouroboros is writing the description (task d3adb33f).';
        assert.equal(view.note, `${line} The displayed text stays until its replacement is saved.`, status);
        if (state) assert.doesNotMatch(view.note, /writing|retries|Resume/, `${status} is neither writing nor promised back`);
        assert.match(view.markdown, /explain itself/);
        const empty = updateLetterView({ ...AVAILABLE, letter: { state: 'none', text: '', checked_key: REQUEST_KEY, request } }, '');
        assert.equal(empty.state, 'empty');
        assert.equal(empty.note, line);
        // Its own save landed while the task finishes: the shown text is the new one, and nothing more is prepared.
        const saved = updateLetterView({ ...AVAILABLE, letter: letter({ request: { ...request, saved: true } }) }, '');
        assert.equal(saved.note, `Ouroboros saved the description; its task (d3adb33f) ${state || 'is still running'}.`);
        assert.doesNotMatch(saved.note, /is writing|not saved yet|until its replacement/);
        assert.deepEqual(saved.action, CHECK, status);
        // A press finishes the package it found: a writer busy with another range names it, and the update on offer waits.
        for (const [base, target] of [['a', 'c'], ['c', 'b']]) {
            const key = { base_sha: base.repeat(40), target_sha: target.repeat(40) };
            const changed = updateLetterView({ ...AVAILABLE, running_sha: key.base_sha, latest_sha: key.target_sha,
                letter: { state: 'none', text: '', checked_key: key, request } });
            assert.equal(changed.note, `${state ? `The description of aaaaaaaa → bbbbbbbb is not saved yet; its task ${state}`
                : 'Ouroboros is writing the description of aaaaaaaa → bbbbbbbb'} (task d3adb33f). The available update `
                + `(${base.repeat(8)} → ${target.repeat(8)}) can be described once that task ends.`);
            assert.deepEqual(changed.action, CHECK);
        }
        // When the headline's own action already checks, the section adds no second check.
        const retained = { ...CURRENT, letter: { state: 'none', text: '', request,
            material: { key: { base_sha: 'a'.repeat(40), target_sha: 'b'.repeat(40) } } } };
        assert.equal(updateVerdict(retained, '').action.id, 'check');
        assert.equal(updateLetterView(retained, '').action, null);
    }
});


test('a writer of an earlier range stays visible beside an independently saved checked description', () => {
    const key = { base_sha: AVAILABLE.running_sha, target_sha: 'c'.repeat(40) };
    const request = { task_id: 'original', current_task_id: 'successor', status: 'running', key: REQUEST_KEY };
    const view = updateLetterView({ ...AVAILABLE, latest_sha: key.target_sha,
        letter: letter({ key, checked_key: key, request, text: 'Already saved for the new update.' }) });
    assert.equal(view.markdown, 'Already saved for the new update.');
    assert.equal(view.action.id, 'check');
    assert.equal(view.request.taskId, 'successor');
    assert.equal(view.label, "What's new");
    assert.equal(view.note, 'This description covers the available update. Ouroboros is writing the description of '
        + 'aaaaaaaa → bbbbbbbb (task successor).');
    assert.doesNotMatch(view.note, /once that task ends|until its replacement is saved/);
});


test('which text is shown and who writes stay two facts for exact ranges, a writer never hiding either', () => {
    const newer = { base_sha: 'a'.repeat(40), target_sha: 'c'.repeat(40) };
    const offer = { ...AVAILABLE, latest_sha: newer.target_sha, checked_target_sha: newer.target_sha };
    const earlier = `This description covers an earlier update (${'a'.repeat(8)} → ${'b'.repeat(8)}).`;
    // The owner presses Check and describe over the old text; the request names the update on offer.
    for (const [status, writer] of [['scheduled', 'The description is not saved yet; its task is waiting to run (task w1).'],
        ['running', 'Ouroboros is writing the description (task w1).'],
        ['cancel_requested', 'The description is not saved yet; its task has been asked to stop (task w1).']]) {
        const request = { task_id: 'w1', mode: 'write', key: newer, status, saved: false };
        const view = updateLetterView({ ...offer, letter: letter({ relation: 'superseded', description_current: false,
            checked_key: newer, request }) });
        assert.equal(view.label, 'Earlier update', status);
        assert.equal(view.note, `${earlier} ${writer}`, status);
        assert.match(view.markdown, /explain itself/, 'the old text stays readable until the new one is saved');
    }
    // The same release version is no proof of the same range: the SHAs decide.
    const sameVersion = updateLetterView({ ...offer, latest_version: letter().target_version,
        letter: letter({ relation: 'superseded', description_current: false, checked_key: newer }) });
    assert.deepEqual([sameVersion.label, sameVersion.note],
        ['Earlier update', `${earlier} No request to describe the available update is open.`]);
    // Busy with the earlier package: its line names that range, never calls the earlier text current, and the update
    // on offer is asked for after it.
    const busy = updateLetterView({ ...offer, letter: letter({ relation: 'superseded', description_current: false,
        checked_key: newer, request: { task_id: 'w0', mode: 'rewrite', key: REQUEST_KEY, status: 'running' } }) });
    assert.equal(busy.label, 'Earlier update');
    assert.equal(busy.note, `${earlier} Ouroboros is writing the description of ${'a'.repeat(8)} → ${'b'.repeat(8)} `
        + `(task w0). The displayed text stays until its replacement is saved. The available update (${'a'.repeat(8)} → `
        + `${'c'.repeat(8)}) can be described once that task ends.`);
    assert.doesNotMatch(busy.note, /current/i);
    // The same target from another starting version keeps its own heading beside a live writer.
    const moved = { base_sha: 'd'.repeat(40), target_sha: 'b'.repeat(40) };
    const otherBase = updateLetterView({ ...AVAILABLE, running_sha: moved.base_sha, checked_target_sha: moved.target_sha,
        letter: letter({ relation: 'other', description_current: false, checked_key: moved,
            request: { task_id: 'w2', mode: 'write', key: moved, status: 'requested' } }) });
    assert.equal(otherBase.label, 'Different range');
    assert.equal(otherBase.note, `This description covers the same target from another starting version (${'a'.repeat(8)} `
        + `→ ${'b'.repeat(8)}). The description is not saved yet; its task is waiting to run (task w2).`);
    // A failed refresh and an unknown writer: each axis says its own failure.
    const failed = updateLetterView({ ...AVAILABLE, letter: letter({ state: 'failed', description_current: false,
        has_last_good: true, request: { task_id: 'w3', mode: 'write', key: REQUEST_KEY, status: 'unknown' } }) });
    assert.equal(failed.note, 'Refresh failed. Previous description kept. The description task’s status is unknown '
        + '(task w3); no new author will be requested until its status can be read.');
    // An ended request of another range says nothing about this update's empty section.
    const endedElsewhere = updateLetterView({ ...offer, letter: { state: 'none', text: '', checked_key: newer,
        request: { task_id: 'w4', mode: 'write', key: REQUEST_KEY, status: 'failed', saved: false } } });
    assert.equal(endedElsewhere.note, 'No description of this update yet.');
    // Author A finished and saved; nothing writes the update found since, and the panel says so in words.
    const finishedA = updateLetterView({ ...offer, letter: letter({ relation: 'superseded', description_current: false,
        checked_key: newer, request: { task_id: 'w5', mode: 'write', key: REQUEST_KEY, status: 'completed', saved: true } }) });
    assert.deepEqual([finishedA.label, finishedA.note],
        ['Earlier update', `${earlier} No request to describe the available update is open.`]);
    // A saved account of exactly the update on offer is current: no such line, and no writer claims it.
    const current = updateLetterView({ ...offer, letter: letter({ key: newer, checked_key: newer,
        request: { task_id: 'w6', mode: 'write', key: newer, status: 'running', saved: true } }) });
    assert.deepEqual([current.label, current.note], ["What's new", 'Ouroboros saved the description; its task (w6) is still running.']);
});


test('a running request says when the owner asked, a fixed time, and names its current task', () => {
    // The owner's press is the one recorded instant: no elapsed counter, nothing that ticks.
    const requested_at = '2026-10-29T22:03:25.827937+00:00';
    const asked = attemptTime(requested_at);
    assert.match(asked, /\d{2}:\d{2}/);
    const request = { task_id: 'd3adb33f', current_task_id: 'c0ffee42', mode: 'rewrite', key: REQUEST_KEY,
        requested_at, status: 'running', saved: false };
    const view = updateLetterView({ ...AVAILABLE, letter: letter({ request }) }, '');
    assert.equal(view.request.taskId, 'c0ffee42', 'a Continue or host retry is the task the card follows');
    assert.equal(view.request.requestedAt, requested_at);
    assert.equal(view.note, `Ouroboros is writing the description you asked for on ${asked} (task c0ffee42). `
        + 'The displayed text stays until its replacement is saved.');
    const empty = updateLetterView({ ...AVAILABLE, letter: { state: 'none', text: '', checked_key: REQUEST_KEY, request } }, '');
    assert.equal(empty.note, `Ouroboros is writing the description you asked for on ${asked} (task c0ffee42).`);
    // An unreadable time says nothing about when rather than something false.
    const unreadable = updateLetterView({ ...AVAILABLE, letter: { state: 'none', text: '', checked_key: REQUEST_KEY,
        request: { ...request, requested_at: 'not a time' } } }, '');
    assert.equal(unreadable.note, 'Ouroboros is writing the description (task c0ffee42).');
});


test('an unknown writer keeps its task and prose without claiming it is running or offering another author', () => {
    for (const saved of [false, true]) {
        const request = { task_id: 'original', current_task_id: 'successor', mode: 'write',
            key: REQUEST_KEY, status: 'unknown', saved };
        const data = { ...AVAILABLE, letter: letter({ request }) };
        const view = updateLetterView(data);
        assert.equal(view.request.taskId, 'successor');
        assert.equal(view.request.live, false);
        assert.match(view.note, /status is unknown \(task successor\)/);
        assert.doesNotMatch(view.note, /is writing|still running|nothing was saved/);
        assert.match(view.markdown, /explain itself/);
        assert.deepEqual(view.action, { id: 'check', label: 'Check', disabled: false });
        assert.equal(view.rewrite, null);
        assert.equal(headlineAction(updateVerdict(CURRENT), data.letter).id, 'check');
        const empty = updateLetterView({ ...CURRENT, letter: { state: 'none', request } });
        assert.notEqual(empty.state, 'none', 'unknown work keeps the task door even without text or an offer');
        assert.match(empty.note, /status is unknown/);
    }
});


test('an ended request that saved nothing says so once; a saved one says nothing; nothing repeats', () => {
    const ended = (status, saved = false) => updateLetterView({ ...AVAILABLE, letter: { state: 'none', text: '',
        checked_key: REQUEST_KEY, request: { task_id: 'd3adb33f', mode: 'write', key: REQUEST_KEY, status, saved } } }, '');
    assert.equal(ended('completed').note, 'The last description task finished without saving a description.');
    assert.equal(ended('failed').note, 'The last description task failed without saving a description.');
    assert.equal(ended('cancelled').note, 'The last description task was stopped without saving a description.');
    assert.equal(ended('missing').note, 'The last description task left no readable result; nothing was saved.');
    for (const status of ['completed', 'failed', 'cancelled']) {
        assert.deepEqual(ended(status).action, DESCRIBE, 'the owner may ask again; the panel never does');
    }
    const saved = updateLetterView({ ...AVAILABLE, letter: letter({ request: {
        task_id: 'd3adb33f', mode: 'write', key: REQUEST_KEY, status: 'completed', saved: true } }) }, '');
    assert.equal(saved.note, '');
    const stoppedRewrite = updateLetterView({ ...AVAILABLE, letter: letter({ request: {
        task_id: 'd3adb33f', mode: 'rewrite', key: REQUEST_KEY, status: 'failed', saved: false } }) }, '');
    assert.equal(stoppedRewrite.note, 'The last description task failed without saving a description.');
    assert.match(stoppedRewrite.markdown, /explain itself/);
});


test('the headline\'s ordinary check is Check and describe; recovery and a live writer keep the plain check', () => {
    // Up to date, never checked, unknown, or a check that failed: one press finds an update and asks for its text.
    const unchecked = { managed: true, check_ok: null, available: false, warnings: ['official_status_requires_check'] };
    const failed = { ...AVAILABLE, check_ok: false, warnings: ['status_error:offline'] };
    for (const [state, data] of [['current', CURRENT], ['unchecked', unchecked], ['check_failed', failed],
        ['unknown', { managed: true, check_ok: null, available: false }]]) {
        const verdict = updateVerdict(data, '');
        assert.equal(verdict.state, state, `fixture no longer produces ${state}`);
        assert.equal(verdict.action.id, 'check', 'the verdict itself is unchanged');
        assert.deepEqual(headlineAction(verdict, data.letter), { id: 'describe', label: 'Check and describe' }, state);
    }
    // The section adds no second Check and describe when the headline carries it (a failed read kept the offer).
    assert.equal(updateLetterView({ ...failed, letter: letter() }, '').action, null);
    assert.deepEqual(updateLetterView({ ...failed, letter: letter() }, '').rewrite.key, letter().key);
    // While a requested task can still save, the headline's check writes nothing.
    const writing = { ...CURRENT, letter: { state: 'none', text: '', request: { task_id: 'd3adb33f', mode: 'write',
        key: REQUEST_KEY, status: 'running', saved: false }, material: { key: REQUEST_KEY } } };
    assert.deepEqual(headlineAction(updateVerdict(writing, ''), writing.letter), updateVerdict(writing, '').action);
    // Recovery, a check already running, an update on offer and a restart keep their own action.
    const recovery = { managed: true, update_tx: { active: true, local_work_recovery: true, phase: 'pending_boot_smoke' } };
    for (const [data, phase] of [[recovery, ''], [CURRENT, 'checking'], [AVAILABLE, ''], [AVAILABLE, 'restart_required']]) {
        const verdict = updateVerdict(data, phase);
        assert.deepEqual(headlineAction(verdict, data.letter), verdict.action, `${verdict.state} keeps its action`);
    }
});


test('a press without a usable answer says refused only when the answer establishes it, else unknown', () => {
    const refused = [
        { status: 400, message: 'mode write or rewrite with base_sha and target_sha, or mode latest alone, is required.', body: {} },
        { status: 409, message: 'The check found an update, but its exact range cannot be described.',
            body: { reason: 'range_unavailable' } },
        { status: 503, message: 'The writing task was not accepted (worker_readiness_exhausted).',
            body: { reason: 'worker_readiness_exhausted' } },
        { status: 503, message: 'The update description record is busy; nothing was requested.', body: { reason: 'record_busy' } },
        { status: 503, message: 'The description record is missing or changed; nothing was requested. Check and describe again.',
            body: { reason: 'request_not_recorded' } },
    ];
    for (const err of refused) {
        assert.equal(requestFailureText(err), `Could not request a description: ${err.message}`);
    }
    // A 500, a gateway's 502 or 503 without the endpoint's facts, an unreadable 200 and a lost answer may all follow
    // an admission: the outcome is unknown and nothing is asked again.
    const unknown = [
        [{ status: 500, message: 'database is locked', body: { error: 'database is locked' } }, 'database is locked'],
        [{ status: 502, message: 'HTTP 502', body: { error: 'non-json response (HTTP 502)' } }, 'HTTP 502'],
        [{ status: 503, message: 'Service Unavailable', body: { error: 'Service Unavailable' } }, 'Service Unavailable'],
        [Object.assign(new Error(''), { unreadable: true, responseStatus: 200 }), 'no readable answer'],
        [new TypeError('Failed to fetch'), 'Failed to fetch'],
    ];
    for (const [err, detail] of unknown) {
        assert.equal(requestFailureText(err), `This press received no confirmation (${detail}); whether it requested a description is unknown.`);
    }
});


test('a busy update phase disables the action and Rewrite but keeps the text', () => {
    for (const phase of ['preflighting', 'updating']) {
        const view = updateLetterView({ ...AVAILABLE, letter: letter() }, phase);
        assert.equal(view.state, 'ready', phase);
        assert.equal(view.action.disabled, true, phase);
        assert.equal(view.rewrite.disabled, true, phase);
    }
});


test('an old failed letter with a last good text shows the text plus the failure reason', () => {
    const view = updateLetterView({
        ...AVAILABLE,
        latest_sha: 'c'.repeat(40),
        letter: letter({
            state: 'failed',
            description_current: false,
            error_kind: 'provider_unavailable',
            error_text: 'openrouter 503',
            has_last_good: true,
        }),
    }, '');
    assert.equal(view.state, 'failed');
    assert.match(view.markdown, /explain itself/, 'the last good letter survives the failed rewrite');
    assert.deepEqual(view.failure, { kind: 'provider_unavailable', text: 'openrouter 503', failedAt: '', key: null });
    // Each axis says its own fact: the kept text after the failed refresh, and that nothing writes the update on offer.
    assert.equal(view.note, 'Refresh failed. Previous description kept. No request to describe the available update is open.');
    assert.equal(view.label, 'Earlier update');
});


test('an old failed letter with no text still names why there is nothing to read', () => {
    const view = updateLetterView({
        ...AVAILABLE,
        letter: letter({ state: 'failed', text: '', description_current: false, error_kind: 'no_credentials', error_text: '', has_last_good: false }),
    }, '');
    assert.equal(view.state, 'failed');
    assert.equal(view.markdown, '');
    assert.deepEqual(view.failure, { kind: 'no_credentials', text: '', failedAt: '', key: null });
    assert.equal(view.note, 'Refresh failed. No description is available yet.');
});


test('the description hides wherever it could only mislead', () => {
    const hiddenByVerdict = [
        ['unmanaged', { managed: false, letter: letter() }],
    ];
    for (const [expected, data] of hiddenByVerdict) {
        assert.equal(updateVerdict(data, '').state, expected, `fixture no longer produces ${expected}`);
        assert.equal(updateLetterView(data, '').state, 'none', `${expected} must not carry a description`);
    }
    assert.equal(updateLetterView({ ...AVAILABLE, letter: letter() }, 'restarting').state, 'none');
    for (const phase of ['', 'loading', 'checking', 'preflighting', 'updating', 'restart_required', 'restart_needed']) {
        assert.equal(updateLetterView({ ...AVAILABLE, letter: letter() }, phase).state, 'ready', phase);
    }
});


test('an unchecked channel keeps historical text and the known writer without claiming its offer', () => {
    const key = letter().key;
    for (const unknown of [
        { warnings: ['official_status_requires_check'] },
        { warnings: ['status_error:offline'] },
    ]) {
        for (const text of ['', letter().text]) {
            const data = { managed: true, check_ok: null, available: false, ...unknown,
                update_channel: 'development', target_ref: 'managed/ouroboros',
                letter: letter({ text, description_current: false, request: {
                    task_id: 'writer-a', status: 'running', mode: 'write', key, saved: false,
                } }) };
            const view = updateLetterView(data);
            assert.equal(view.markdown, text);
            assert.equal(view.request.taskId, 'writer-a');
            assert.equal(view.request.live, true);
            assert.equal(view.action, null, 'the headline already offers the free check');
            assert.equal(headlineAction(updateVerdict(data), data.letter).id, 'check');
            assert.equal(view.rewrite, null, 'no implicit restart or duplicate writer');
            assert.equal(view.over, '');
            assert.match(view.note, /^Ouroboros is writing the description of aaaaaaaa → bbbbbbbb \(task writer-a\)/,
                'with no known offer the writer names the range it writes');
            assert.doesNotMatch(view.note, /available update|once that task ends/);
        }
        const history = updateLetterView({ managed: true, check_ok: null, available: false, ...unknown,
            letter: letter({ description_current: false }) });
        assert.equal(history.markdown, letter().text);
        assert.equal(history.label, 'Saved description');
        assert.doesNotMatch(history.note, /earlier update/);
        assert.deepEqual(history.rewrite.key, key);
    }
});


test('stale check facts cannot remove Check and describe or label another range as qualified', () => {
    for (const text of ['', letter().text]) {
        const oldKey = { ...letter().key, base_sha: 'c'.repeat(40) };
        const view = updateLetterView({ ...AVAILABLE, letter: letter({ text, checked_key: oldKey,
            qualification: { ...oldKey, landed_over_review: [{ sha: 'd'.repeat(40) }] },
            description_current: false }) });
        assert.deepEqual(view.action, DESCRIBE);
        assert.equal(view.over, '');
        assert.deepEqual(view.writeKey, text ? letter().key : null, 'stale facts name no current offer');
    }
    const unavailable = updateLetterView({ ...AVAILABLE, running_sha: '', letter: letter({
        text: '', description_current: false, checked_key: REQUEST_KEY }) });
    assert.equal(unavailable.state, 'empty');
    assert.equal(unavailable.writeKey, null, 'an explicitly unknown runtime is never checkout HEAD');
    assert.deepEqual(unavailable.action, DESCRIBE, 'the server checks the range on this press');
});


test('an available update whose recorded check facts are stale still compares the text with the checked pair', () => {
    // The status names a..c; the record's checked_key is another range, so nothing is on offer for actions (the server
    // checks on the press), yet a text applied to this version is not that update's account.
    const status = { ...AVAILABLE, latest_sha: 'c'.repeat(40), checked_target_sha: 'c'.repeat(40) };
    const stale = { base_sha: 'd'.repeat(40), target_sha: 'c'.repeat(40) };
    const applied = updateLetterView({ ...status, letter: letter({ key: { base_sha: 'e'.repeat(40), target_sha: 'a'.repeat(40) },
        relation: 'applied', description_current: false, checked_key: stale }) });
    assert.deepEqual([applied.label, applied.note], ['Earlier update', 'This description covers an earlier update (eeeeeeee → aaaaaaaa).']);
    assert.deepEqual([applied.action, applied.writeKey.target_sha], [DESCRIBE, 'a'.repeat(40)], 'actions keep their offer rule');
    // A text of exactly that checked pair stays its account, never "written for an earlier update".
    const own = updateLetterView({ ...status, letter: letter({ key: { base_sha: 'a'.repeat(40), target_sha: 'c'.repeat(40) },
        description_current: false, checked_key: stale }) });
    assert.deepEqual([own.label, own.note], ["What's new", '']);
});


test('the host\'s description_current settles which text this is; without it the retained cell keeps its reading', () => {
    // Its meaning: the text's range is exactly the running base to the checked target. Retained material of another
    // base does not make such a text an earlier one; the same text the host does not confirm reads as another range.
    const merge = 'd'.repeat(40);
    const retained = { ...CURRENT, current_sha: merge, running_sha: merge, checked_target_sha: 'c'.repeat(40),
        letter: letter({ key: { base_sha: merge, target_sha: 'c'.repeat(40) }, relation: 'pending', description_current: true,
            material: { key: { base_sha: 'b'.repeat(40), target_sha: 'c'.repeat(40) } } }) };
    assert.deepEqual([updateLetterView(retained).label, updateLetterView(retained).note], ["What's new", '']);
    const unconfirmed = updateLetterView({ ...retained, letter: { ...retained.letter, description_current: false } });
    assert.deepEqual([unconfirmed.label, unconfirmed.note], ['Different range',
        'This description covers the same target from another starting version (dddddddd → cccccccc).']);
});


test('the host line about the update on offer shows with or without a description, only from that range\'s facts', () => {
    const range = { base_sha: 'a'.repeat(40), target_sha: 'b'.repeat(40) };
    const receipt = { checks: [], review: { observed: 'FAIL', declared: 'PASS' } };
    const facts = { ...range, landed_over_review: [], commits: [{ sha: 'm'.repeat(40), subject: 'merge', receipt }] };
    const line = 'Went in over a review FAIL or no verdict: 1 merge.';
    const described = updateLetterView({ ...AVAILABLE, letter: letter({ qualification: facts, checked_key: range }) });
    assert.equal(described.over, line);
    assert.equal(described.overDetail, 'Went in over a review FAIL or no verdict: mmmmmmmm (merge; review FAIL).');
    // No description yet: the section shows for the update on offer, and the line with it.
    const bare = updateLetterView({ ...AVAILABLE, letter: { state: 'none', text: '', checked_key: range, qualification: facts } });
    assert.equal(bare.state, 'empty');
    assert.equal(bare.over, line);
    // Facts of another range, a range that is not on offer, or facts that name nothing: no line.
    const other = { ...facts, target_sha: 'c'.repeat(40) };
    assert.equal(updateLetterView({ ...AVAILABLE, letter: letter({ qualification: other, checked_key: range }) }).over, '');
    assert.equal(updateLetterView({ ...CURRENT, letter: letter({ relation: 'applied', qualification: facts }) }).over, '');
    const quiet = { ...facts, commits: [{ sha: 'm'.repeat(40), subject: 'merge', receipt: { checks: [], review: { observed: 'PASS' } } }] };
    assert.equal(updateLetterView({ ...AVAILABLE, letter: letter({ qualification: quiet, checked_key: range }) }).over, '');
    // Rows of this computer that were not read are said in Details, with or without a line; what is painted and
    // where is the browser test's (tests/test_update_description_browser.py).
    const { landed_over_review: _unread, ...blind } = quiet;
    const unreadView = updateLetterView({ ...AVAILABLE, letter: letter({ qualification: blind, checked_key: range }) });
    assert.equal(unreadView.over, '');
    assert.match(unreadView.overDetail, /^Made on this computer: Not read for this range/);
});

test('a description-bearing payload leaves the verdict byte-for-byte identical', () => {
    for (const base of [CURRENT, AVAILABLE, { managed: true, update_tx: { active: true, phase: 'rolling_back' } }]) {
        for (const phase of ['', 'checking', 'updating', 'restart_required']) {
            const without = updateVerdict(base, phase);
            const withLetter = updateVerdict({ ...base, letter: letter() }, phase);
            assert.equal(withLetter.state, without.state);
            assert.equal(withLetter.headline, without.headline);
            assert.equal(withLetter.tone, without.tone);
            assert.deepEqual(withLetter.action, without.action);
            assert.deepEqual(withLetter.chips, without.chips);
            assert.deepEqual(withLetter.warnings, without.warnings);
            assert.equal(withLetter.hint, without.hint);
        }
    }
});

test('currentness comes from the server range fact, never VERSION or mutable checkout', () => {
    const data = { ...AVAILABLE, current_sha: 'x'.repeat(40), letter: letter() };
    assert.equal(updateLetterView(data).descriptionCurrent, true, 'disk movement is not adoption');
    const baseChanged = updateLetterView({ ...data, running_sha: 'c'.repeat(40),
        letter: letter({ description_current: false }) });
    assert.equal(baseChanged.descriptionCurrent, false);
    assert.equal(updateLetterView({ ...data, letter: letter({ description_current: undefined }) }).descriptionCurrent, false);
});

test('applied old text keeps a current failure distinct from historical failure', () => {
    const failed = letter({ state: 'failed', relation: 'applied', has_last_good: true, description_current: false,
        failed_at: '2026-10-03T13:07:00Z', error_text: 'Capacity unavailable',
        latest_failed_key: { base_sha: 'b'.repeat(40), target_sha: 'c'.repeat(40) } });
    const status = { ...AVAILABLE, running_sha: 'b'.repeat(40), latest_sha: 'c'.repeat(40), letter: failed };
    const current = updateLetterView(status);
    assert.match(current.note, /^Refresh failed on/);
    assert.equal(current.failure.key.base_sha, 'b'.repeat(40));
    assert.equal(current.meta.writtenAt, failed.written_at);
    assert.equal(current.markdown, failed.text);
    const adopted = updateLetterView({ ...status, available: false, running_sha: 'c'.repeat(40) });
    assert.equal(adopted.failure, null);
    assert.equal(adopted.note, '');
    assert.equal(adopted.markdown, failed.text);
});

test('failed check preserves the visible prior description', () => {
    const view = updateLetterView({ ...AVAILABLE, check_ok: false,
        warnings: ['fetch_error:offline'], letter: letter() });
    assert.equal(view.markdown, letter().text);
});


// --- Source pins: the DOM contract a pure projector cannot see --------------

test('the description section keeps one compact action and a quiet Rewrite outside its authored body', () => {
    const actionRow = SOURCE.indexOf('<div class="dashboard-hero-actions"><button type="button" class="btn btn-primary" id="btn-update-primary" hidden>');
    const section = SOURCE.indexOf('<section class="updates-letter" id="updates-letter" aria-labelledby="updates-letter-label" hidden>');
    const recovery = SOURCE.indexOf('<details class="ui-disclosure updates-recovery">');
    assert.ok(actionRow > -1 && section > -1 && recovery > -1, 'the card template moved');
    assert.ok(actionRow < section, 'the letter belongs BELOW the single primary action');
    assert.ok(section < recovery, 'the letter belongs ABOVE the Recovery disclosure');

    const card = SOURCE.slice(section, recovery);
    assert.match(card, /class="updates-letter-head"/);
    // A real heading names the section (aria-labelledby above), like Recovery's h4.
    assert.match(card, /<h4 class="updates-letter-label" id="updates-letter-label">/);
    assert.match(card, /class="updates-letter-meta"/);
    assert.match(card, /class="updates-letter-note"/);
    assert.match(card, /class="updates-letter-body ui-rich-content" id="updates-letter-body">/);
    // The enhancer marks the body itself; a static or duplicate mark would claim
    // an un-enhanced node is enhanced.
    assert.doesNotMatch(card, /data-chat-markdown-enhanced/);
    assert.doesNotMatch(SOURCE, /chatMarkdownEnhanced/);
    assert.match(card, /class="btn btn-default btn-sm updates-letter-action" id="updates-letter-action"/);
    const head = card.slice(card.indexOf('class="updates-letter-actions"'), card.indexOf('class="updates-letter-note"'));
    assert.equal((head.match(/<button/g) || []).length, 1, 'the head holds the section\'s one action');
    // Asking again for the shown range waits in Details, after the sentence that says what writing costs.
    const details = card.slice(card.indexOf('<details class="ui-disclosure updates-letter-details"'));
    assert.ok(details.indexOf('id="updates-letter-resources"') < details.indexOf('id="updates-letter-rewrite"'));
    assert.match(details, /class="btn btn-default btn-sm updates-letter-rewrite" id="updates-letter-rewrite"/);
    // The body's one length control sits under it: the first part reads at once, Read all opens the rest.
    assert.match(card, /<button type="button" class="btn btn-ghost btn-sm ui-list-more" id="updates-letter-more" hidden>Read all<\/button>/);
    // The note's words are the live region; the door to the running task's card sits beside it, outside it.
    assert.match(card, /<div class="updates-letter-note" id="updates-letter-note-line" hidden><span id="updates-letter-note" role="status" aria-live="polite"><\/span> <span id="updates-letter-task"><\/span><\/div>/);
    // Which text this is and who writes are read before the text itself: the note line precedes the body.
    assert.ok(card.indexOf('id="updates-letter-note-line"') < card.indexOf('id="updates-letter-body"'));
    // Which notes draw it and where a press lands: tests/test_update_description_browser.py, through app.js.
    assert.doesNotMatch(SOURCE, /ouro:open-task/, 'the one door module raises it');
    // Check and describe is ONE request whose server checks first; the check while a task writes never authors;
    // Write and Rewrite name their range to the same endpoint and never check.
    const fn = (name) => SOURCE.slice(SOURCE.indexOf(`async function ${name}(`)).split('\n    }\n')[0];
    assert.match(fn('describeUpdate'), /apiClient\.updateDescribeLatest\(\)/);
    assert.doesNotMatch(fn('describeUpdate'), /updateCheck|fetchRemote: true/);
    assert.match(fn('checkChannel'), /loadStatus\(\{ fetchRemote: true, preservePhase: true \}\)/);
    assert.doesNotMatch(fn('checkChannel'), /updateDescri/);
    assert.match(fn('requestDescription'), /apiClient\.updateDescription\(choice\.mode, choice\.key\)/);
    assert.doesNotMatch(fn('requestDescription'), /fetchRemote: true|updateDescribeLatest/);
    assert.match(card, /<details class="ui-disclosure updates-letter-details"/);
});


test('the letter body goes through the sanitizing markdown pipeline and is disposed before re-render', () => {
    assert.match(SOURCE, /import \{ destroyChatMarkdown, enhanceChatMarkdown, mountChatMarkdown \} from '\.\/chat_markdown\.js'/);
    assert.match(SOURCE, /mountChatMarkdown\(letterBody, view\.markdown \|\| ''\)/);
    assert.match(SOURCE, /letterDisposer = enhanceChatMarkdown\(letterBody, \{[\s\S]*?onDomWrite:/);
    // Controls the pipeline adds are scrubbed after EVERY write it makes: a fenced block
    // gets its Copy button at render, a degrading mermaid block gets one asynchronously.
    assert.match(SOURCE, /function stripLetterControls\(root = letterBody\)[\s\S]*?querySelectorAll\('button'\)[\s\S]*?remove\(\)/);
    const enhanceCall = SOURCE.slice(SOURCE.indexOf('letterDisposer = enhanceChatMarkdown'));
    assert.match(enhanceCall.slice(0, 400), /onDomWrite: \(mutate\) => \{\s*mutate\(\);\s*stripLetterControls\(\);/);
    // The disposer (or the module-level destroyer, when none was kept) runs
    // BEFORE the innerHTML that would orphan its charts and timers — on the
    // content path and on the hide path alike.
    assert.match(SOURCE, /function releaseLetterBody\(\)[\s\S]*?letterDisposer\(\)[\s\S]*?destroyChatMarkdown\(letterBody\)/);
    const render = SOURCE.slice(SOURCE.indexOf('function renderLetter()'), SOURCE.indexOf('function render()'));
    assert.ok(render.indexOf('releaseLetterBody()') < render.indexOf('letterBody.innerHTML'), 'release before write');
    assert.equal((render.match(/releaseLetterBody\(\)/g) || []).length, 2, 'hide path and content path both release');
    // Unchanged content keeps its DOM (and the owner's selection with it); a "Read all" measure taken while the tab
    // was hidden (0 x 0, unknown) is still owed and runs once the body has a height.
    assert.match(render, /const nextKey = letterContentKey\(view\);\s*if \(nextKey === letterKey\) \{\s*if \(letterMeasureOwed\) requestAnimationFrame\(measureLetter\);\s*return;\s*\}/);
    assert.match(SOURCE, /function measureLetter\(\) \{\s*if \(!letterMeasureOwed \|\| letterBody\.hidden \|\| !letterBody\.clientHeight\) return;/);
    assert.match(SOURCE, /'ouro:dashboard-subtab-shown', \(event\) => \{\s*if \(event\.detail\?\.tab !== 'updates'[^\n]*\n\s*requestAnimationFrame\(measureLetter\);/);
    // Whichever door shows the body, its own box takes the owed measure: the page opened from the Dashboard's list by
    // the row already chosen changes no tab and announces nothing (tests/test_update_description_browser.py, step 7).
    assert.match(SOURCE, /new ResizeObserver\(\(\) => measureLetter\(\)\)\.observe\(letterBody\)/);
    // Content identity is the CONTENT: two different paragraphs of equal length must not
    // share a key, and a rewrite that produced the same text must not throw the DOM away.
    assert.match(SOURCE, /function letterContentKey\(view\) \{[\s\S]*?return view\.markdown;/);
    assert.doesNotMatch(SOURCE, /view\.markdown\.length/);
});


test('the letter rides the existing render path and never writes the verdict surfaces', () => {
    // No listener, timer or poll of its own: render() already runs on every
    // phase change and status load.
    const renderBody = SOURCE.slice(SOURCE.indexOf('function render()'), SOURCE.indexOf('async function reconcileRestartStatus'));
    assert.match(renderBody, /renderLetter\(\);/); // Same render owner; attempt details may render before the letter.
    const letterCode = SOURCE.slice(SOURCE.indexOf('function releaseLetterBody()'), SOURCE.indexOf('function render()'));
    for (const forbidden of ['dot.dataset.tone', '#updates-summary', 'summary.textContent', 'primaryBtn']) {
        assert.ok(!letterCode.includes(forbidden), `letter code must not touch ${forbidden}`);
    }
    // The apply flow reads the exact preflight description through its own helpers, never the
    // panel's letter state: the legacy-path slice that tests/test_packaged_runtime_and_lifecycle.py
    // reads must not gain letter code (or the 'replace'/'stash' literals its own guard forbids).
    const applyFn = SOURCE.split('async function applyUpdate')[1].split('\n    }')[0];
    for (const forbidden of ['letter', 'Letter', "'replace'", "'stash'"]) {
        assert.ok(!applyFn.includes(forbidden), `applyUpdate must not contain ${forbidden}`);
    }
});


test('a level check after apply writes the retained update, not an older description', () => {
    for (const channel of ['stable', 'qa', 'development']) {
        const recent = { base_sha: 'b'.repeat(40), target_sha: 'c'.repeat(40),
            update_channel: channel, target_ref: `managed/${channel}` };
        const status = { ...CURRENT, current_sha: recent.target_sha, running_sha: recent.target_sha,
            checked_target_sha: recent.target_sha, update_channel: channel, target_ref: recent.target_ref,
            letter: letter({ relation: 'applied', description_current: false,
                checked_key: { ...recent, base_sha: recent.target_sha }, material: { key: recent } }) };
        const view = updateLetterView(status);
        assert.deepEqual(view.action, { id: 'write', mode: 'write', key: recent, label: 'Write description', disabled: false });
        // The Earlier update keeps its own Rewrite beside Write for the update just applied.
        assert.deepEqual(view.rewrite, { mode: 'rewrite', key: letter().key, label: 'Rewrite', disabled: false });
        assert.equal(view.label, 'Earlier update');
        assert.match(view.note, /covers an earlier update/);
        assert.equal(view.markdown, letter().text);
        const empty = updateLetterView({ ...status, letter: { ...status.letter, text: '', state: 'none' } });
        assert.equal(empty.state, 'empty');
        assert.deepEqual(empty.action.key, recent);
        assert.deepEqual([empty.label, empty.note], ["What's new", 'No description of this update yet.']);
        // Before restart the disk holds the target while the server still runs its base: the check that offered
        // the update kept its material, so Write names that range and no word says this version runs it.
        const waiting = { ...status, running_sha: recent.base_sha, letter: { ...status.letter, checked_key: recent } };
        assert.deepEqual(updateLetterView(waiting).action.key, recent);
        assert.equal(updateLetterView(waiting).label, 'Earlier update');
        const waitingEmpty = updateLetterView({ ...waiting, letter: { ...waiting.letter, text: '', state: 'none' } });
        assert.deepEqual([waitingEmpty.state, waitingEmpty.label, waitingEmpty.action.key], ['empty', "What's new", recent]);
        // Without that material the checked pair alone proves no incoming change: only the shown text's Rewrite.
        const unproven = { ...waiting, letter: { ...waiting.letter, material: null } };
        assert.deepEqual(updateLetterView(unproven).rewrite,
            { mode: 'rewrite', key: letter().key, label: 'Rewrite', disabled: false });
        assert.equal(updateLetterView(unproven).action, null);
        assert.equal(updateLetterView({ ...unproven, letter: { ...unproven.letter, text: '', state: 'none' } }).state,
            'none');
        // A new pending offer always wins; another target's, channel's or ref's retained material never does.
        const pending = { ...status, available: true, latest_sha: 'd'.repeat(40),
            checked_target_sha: 'd'.repeat(40), letter: { ...status.letter,
                checked_key: { ...recent, base_sha: recent.target_sha, target_sha: 'd'.repeat(40) } } };
        assert.equal(updateLetterView(pending).writeKey.target_sha, 'd'.repeat(40));
        assert.equal(updateLetterView(pending).action.id, 'describe');
        for (const other of [{ update_channel: 'other' }, { target_ref: 'managed/elsewhere' }, { checked_target_sha: 'e'.repeat(40) }]) {
            assert.equal(updateLetterView({ ...status, ...other }).rewrite.mode, 'rewrite');
            assert.equal(updateLetterView({ ...status, ...other }).action, null);
        }
    }
});


test('a merged install never writes its bare checked pair; a text just saved for the material reads neutrally', () => {
    const merge = 'd'.repeat(40);
    for (const channel of ['stable', 'qa', 'development']) {
        const recent = { base_sha: 'b'.repeat(40), target_sha: 'c'.repeat(40),
            update_channel: channel, target_ref: `managed/${channel}` };
        // The install serves its own merge M with the official target inside it: checkout and server are both M,
        // and the check records the pair M..T, which names no incoming commit.
        const merged = { ...CURRENT, current_sha: merge, running_sha: merge, checked_target_sha: recent.target_sha,
            update_channel: channel, target_ref: recent.target_ref,
            letter: letter({ relation: 'applied', description_current: false,
                checked_key: { ...recent, base_sha: merge }, material: { key: recent } }) };
        const view = updateLetterView(merged);
        assert.deepEqual(view.action, { id: 'write', mode: 'write', key: recent, label: 'Write description', disabled: false });
        assert.equal(view.label, 'Earlier update', 'a technical checked pair keeps no current-version heading');
        assert.equal(view.note, `This description covers an earlier update (${'a'.repeat(8)} → ${'b'.repeat(8)}).`);
        // Without material nothing proves an incoming range, whether the checkout is still M or a local commit C
        // moved it past M: no new Write; the shown text keeps its own Rewrite and no text hides the section.
        for (const checkout of [merge, 'e'.repeat(40)]) {
            const level = { ...merged, current_sha: checkout, letter: { ...merged.letter, material: null } };
            assert.equal(updateLetterView(level).action, null);
            assert.deepEqual(updateLetterView(level).rewrite,
                { mode: 'rewrite', key: letter().key, label: 'Rewrite', disabled: false });
            assert.equal(updateLetterView({ ...level, letter: { ...level.letter, text: '', state: 'none' } }).state,
                'none');
        }
        // The text just saved for that material keeps its predecessor's ancestry (relation other) until the next
        // check: a neutral heading, no earlier note, and Rewrite of its own range.
        const fresh = updateLetterView({ ...merged, letter: { ...merged.letter, key: recent, relation: 'other',
            text: 'What b..c brings.' } });
        assert.deepEqual([fresh.label, fresh.note], ["What's new", '']);
        assert.deepEqual(fresh.rewrite, { mode: 'rewrite', key: recent, label: 'Rewrite', disabled: false });
        assert.equal(fresh.action, null);
    }
});


test('the same target from another base: Write keeps the material, the old text reads as a different range', () => {
    // The text was written for A..C before the install moved to B; B..C was then retained and applied.
    const recent = { base_sha: 'b'.repeat(40), target_sha: 'c'.repeat(40), update_channel: 'stable',
        target_ref: 'managed/main' };
    const old = { ...recent, base_sha: 'a'.repeat(40) };
    const view = updateLetterView({ ...CURRENT, current_sha: recent.target_sha, running_sha: recent.target_sha,
        checked_target_sha: recent.target_sha, update_channel: 'stable', target_ref: 'managed/main',
        letter: letter({ key: old, relation: 'applied', description_current: false, material: { key: recent } }) });
    assert.deepEqual(view.action, { id: 'write', mode: 'write', key: recent, label: 'Write description', disabled: false });
    // Rewrite keeps the shown text's own recorded range, never the material's.
    assert.deepEqual(view.rewrite, { mode: 'rewrite', key: old, label: 'Rewrite', disabled: false });
    assert.deepEqual([view.markdown, view.covers], [letter().text, `${'a'.repeat(8)} → ${'c'.repeat(8)}`]);
    // A shared target makes the text neither earlier nor this version's own account: a neutral heading and note.
    assert.equal(view.label, 'Different range');
    assert.equal(view.note,
        `This description covers the same target from another starting version (${'a'.repeat(8)} → ${'c'.repeat(8)}).`);
});
