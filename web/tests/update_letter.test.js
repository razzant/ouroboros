// The update description: Ouroboros's own account of the pending official
// update, written as ordinary work (the owner's Write/Rewrite or its own
// initiative), delivered inside the ordinary status payload and KEPT after the
// update lands.
//
// Two kinds of assertion live here, for two different failure modes. The
// projector cases pin the pure function (what the panel decides to say);
// the source pins guard the facts a pure test cannot see — where the section
// sits in the card, that the markdown pipeline is the sanitizing one, that
// its disposer runs before every re-render, and that Check and Write stay two
// distinct requests beside the authored body.

import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';

import { attemptTime } from '../modules/update_status.js';
import { updateLetterView, updateVerdict } from '../modules/updates.js';

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

test('a payload without a description: hidden with nothing on offer, Write when an update is', () => {
    for (const data of [CURRENT, { ...CURRENT, letter: null }, { ...CURRENT, letter: 'nope' }]) {
        const view = updateLetterView(data, '');
        assert.equal(view.state, 'none');
        assert.equal(view.markdown, '');
        assert.equal(view.write, null);
    }
    assert.equal(updateLetterView().state, 'none');
    // An update on offer with nothing written yet keeps the section, with the one authoring control.
    for (const letterPart of [null, { state: 'none', text: '', checked_key: REQUEST_KEY, revision: 0 }]) {
        const view = updateLetterView({ ...AVAILABLE, letter: letterPart }, '');
        assert.equal(view.state, 'empty');
        assert.equal(view.label, "What's new");
        assert.equal(view.note, 'No description of this update yet.');
        assert.deepEqual(view.write, { mode: 'write', key: { ...(letterPart?.checked_key || {}), ...REQUEST_KEY },
            label: 'Write description', disabled: false });
        assert.equal(view.covers, '');
    }
});


test('a pending description is "What\'s new" with its provenance, its range and Rewrite', () => {
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
    assert.equal(view.write.mode, 'rewrite');
    assert.equal(view.write.label, 'Rewrite');
    assert.equal(view.write.key.target_sha, 'b'.repeat(40));
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
    assert.deepEqual(view.write.key, letter().key, 'Rewrite names the text\'s own range, never the empty one');
    assert.equal(view.write.mode, 'rewrite');
    // A kept description about a version this one has moved past keeps its provenance in Details.
    const older = updateLetterView({ ...CURRENT, current_version: '6.115.0', letter: letter({ relation: 'applied' }) }, '');
    assert.equal(older.meta.targetVersion, '6.114.0');
});


test('a newer target keeps the old text with its own range and offers Write for the new one', () => {
    const view = updateLetterView({
        ...AVAILABLE,
        latest_version: '6.115.0', latest_sha: 'c'.repeat(40),
        letter: letter({ relation: 'superseded', description_current: false,
            checked_key: { base_sha: 'a'.repeat(40), target_sha: 'c'.repeat(40) } }),
    }, '');
    assert.equal(view.state, 'ready');
    assert.equal(view.label, "What's new");
    assert.equal(view.note, `This description covers an earlier update (${'a'.repeat(8)} → ${'b'.repeat(8)}).`);
    assert.equal(view.covers, `${'a'.repeat(8)} → ${'b'.repeat(8)}`);
    assert.match(view.markdown, /explain itself/);
    assert.equal(view.write.mode, 'write');
    assert.equal(view.write.key.target_sha, 'c'.repeat(40));
    assert.equal(view.write.label, 'Write description');
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
    assert.equal(moved.label, "What's new");
    assert.equal(moved.note, 'This description was written for an earlier update.');
    // A relation this client does not know is treated as the honest "other".
    const unnamed = updateLetterView({ ...CURRENT, letter: letter({ relation: 'sideways', description_current: false }) }, '');
    assert.equal(unnamed.relation, 'other');
    assert.equal(unnamed.note, 'This description was written for an earlier update.');
});


test('a running request disables Write and says which task is writing; the old text stays', () => {
    for (const status of ['scheduled', 'running', 'interrupted']) {
        const request = { task_id: 'd3adb33f', mode: 'rewrite', key: REQUEST_KEY, status, saved: false };
        const view = updateLetterView({ ...AVAILABLE, letter: letter({ request }) }, '');
        assert.equal(view.write.disabled, true, status);
        assert.equal(view.write.label, 'Writing…');
        assert.equal(view.note, 'Ouroboros is writing the description (task d3adb33f). The current text stays until the new one is saved.');
        assert.match(view.markdown, /explain itself/);
        const empty = updateLetterView({ ...AVAILABLE, letter: { state: 'none', text: '', checked_key: REQUEST_KEY, request } }, '');
        assert.equal(empty.state, 'empty');
        assert.equal(empty.note, 'Ouroboros is writing the description (task d3adb33f).');
        // Its own save landed while the task finishes: the shown text is the new one, not one awaiting replacement.
        const saved = updateLetterView({ ...AVAILABLE, letter: letter({ request: { ...request, saved: true } }) }, '');
        assert.equal(saved.note, 'Ouroboros saved the description; its task (d3adb33f) is still running.');
        assert.equal(saved.write.disabled, true, status);
    }
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
        + 'The current text stays until the new one is saved.');
    const empty = updateLetterView({ ...AVAILABLE, letter: { state: 'none', text: '', checked_key: REQUEST_KEY, request } }, '');
    assert.equal(empty.note, `Ouroboros is writing the description you asked for on ${asked} (task c0ffee42).`);
    // An unreadable time says nothing about when rather than something false.
    const unreadable = updateLetterView({ ...AVAILABLE, letter: { state: 'none', text: '', checked_key: REQUEST_KEY,
        request: { ...request, requested_at: 'not a time' } } }, '');
    assert.equal(unreadable.note, 'Ouroboros is writing the description (task c0ffee42).');
});


test('an ended request that saved nothing says so once; a saved one says nothing; nothing repeats', () => {
    const ended = (status, saved = false) => updateLetterView({ ...AVAILABLE, letter: { state: 'none', text: '',
        checked_key: REQUEST_KEY, request: { task_id: 'd3adb33f', mode: 'write', key: REQUEST_KEY, status, saved } } }, '');
    assert.equal(ended('completed').note, 'The last writing task finished without saving a description.');
    assert.equal(ended('failed').note, 'The last writing task failed without saving a description.');
    assert.equal(ended('cancelled').note, 'The last writing task was stopped without saving a description.');
    assert.equal(ended('missing').note, 'The last writing task left no readable result; nothing was saved.');
    for (const status of ['completed', 'failed', 'cancelled']) {
        assert.equal(ended(status).write.disabled, false, 'the owner may ask again; the panel never does');
    }
    const saved = updateLetterView({ ...AVAILABLE, letter: letter({ request: {
        task_id: 'd3adb33f', mode: 'write', key: REQUEST_KEY, status: 'completed', saved: true } }) }, '');
    assert.equal(saved.note, '');
    const stoppedRewrite = updateLetterView({ ...AVAILABLE, letter: letter({ request: {
        task_id: 'd3adb33f', mode: 'rewrite', key: REQUEST_KEY, status: 'failed', saved: false } }) }, '');
    assert.equal(stoppedRewrite.note, 'The last writing task failed without saving a description.');
    assert.match(stoppedRewrite.markdown, /explain itself/);
});


test('a busy update phase disables Write but keeps the text', () => {
    for (const phase of ['preflighting', 'updating']) {
        const view = updateLetterView({ ...AVAILABLE, letter: letter() }, phase);
        assert.equal(view.state, 'ready', phase);
        assert.equal(view.write.disabled, true, phase);
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
    assert.equal(view.note, 'Refresh failed. Previous description kept.');
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
        ['unknown', { managed: true, warnings: ['status_error:boom'], check_ok: null, available: false, letter: letter() }],
        ['unchecked', {
            managed: true, check_ok: null, available: false,
            warnings: ['official_status_requires_check'], letter: letter(),
        }],
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

test('the description section keeps Check and Write as two compact actions outside its authored body', () => {
    const actionRow = SOURCE.indexOf('class="settings-action-row updates-action-row"');
    const section = SOURCE.indexOf('<section class="updates-letter" id="updates-letter" aria-labelledby="updates-letter-label" hidden>');
    const recovery = SOURCE.indexOf('<details class="updates-recovery">');
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
    assert.match(card, /class="btn btn-ghost btn-sm updates-letter-refresh"/);
    assert.match(card, /aria-label="Check for updates"/);
    assert.match(card, /class="btn btn-default btn-sm updates-letter-write" id="updates-letter-write"/);
    assert.equal((card.match(/<button/g) || []).length, 2);
    // The note's words are the live region; the door to the running task's card sits beside it, outside it.
    assert.match(card, /<div class="updates-letter-note" id="updates-letter-note-line" hidden><span id="updates-letter-note" role="status" aria-live="polite"><\/span> <span id="updates-letter-task"><\/span><\/div>/);
    // Which notes draw it and where a press lands: tests/test_update_description_browser.py, through app.js.
    assert.doesNotMatch(SOURCE, /ouro:open-task/, 'the one door module raises it');
    // Check never authors; Write is the owner's explicit request to the one endpoint.
    const check = SOURCE.slice(SOURCE.indexOf("letterRefresh.addEventListener('click'"), SOURCE.indexOf("letterWrite.addEventListener('click'"));
    assert.match(check, /loadStatus\(\{ fetchRemote: true, preservePhase: true \}\)/);
    assert.doesNotMatch(check, /updateDescription/);
    const write = SOURCE.slice(SOURCE.indexOf("letterWrite.addEventListener('click'"));
    assert.match(write.slice(0, 1200), /apiClient\.updateDescription\(view\.write\.mode, view\.write\.key\)/);
    assert.doesNotMatch(write.slice(0, 1200), /fetchRemote: true/);
    assert.match(card, /<details class="updates-letter-details"/);
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
    // Unchanged content keeps its DOM (and the owner's selection with it).
    assert.match(render, /const nextKey = letterContentKey\(view\);\s*if \(nextKey === letterKey\) return;/);
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
    assert.equal((SOURCE.match(/renderLetter\(\)/g) || []).length, 4, 'ordinary render, Check and Write share one renderer');

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
