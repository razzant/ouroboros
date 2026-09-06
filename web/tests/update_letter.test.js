// The update letter: Ouroboros's own short paragraph about the pending
// official update, delivered inside the ordinary status payload and KEPT
// after the update lands.
//
// Two kinds of assertion live here, for two different failure modes. The
// projector cases pin the pure function (what the panel decides to say);
// the source pins guard the facts a pure test cannot see — where the section
// sits in the card, that the markdown pipeline is the sanitizing one, that
// its disposer runs before every re-render, and that none of this leaked
// into the apply flow or grew a second button.

import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';

import { updateLetterView, updateVerdict } from '../modules/updates.js';

const SOURCE = readFileSync(new URL('../modules/updates.js', import.meta.url), 'utf8')
    .replace(/\r\n?/g, '\n');

const CURRENT = {
    managed: true,
    check_ok: true,
    available: false,
    current_version: '6.114.0',
    current_short_sha: 'abcd1234',
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
};

function letter(overrides = {}) {
    return {
        state: 'ready',
        relation: 'pending',
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


test('a payload without a letter leaves the section hidden', () => {
    for (const data of [AVAILABLE, { ...AVAILABLE, letter: null }, { ...AVAILABLE, letter: 'nope' }]) {
        const view = updateLetterView(data, '');
        assert.equal(view.state, 'none');
        assert.equal(view.markdown, '');
        assert.equal(view.label, '');
        assert.equal(view.note, '');
        assert.equal(view.failure, null);
    }
    assert.equal(updateLetterView().state, 'none');
});


test('a pending letter is "What\'s new" with its provenance and no note', () => {
    const view = updateLetterView({ ...AVAILABLE, letter: letter() }, '');
    assert.equal(view.state, 'ready');
    assert.equal(view.relation, 'pending');
    assert.equal(view.label, "What's new");
    assert.match(view.markdown, /explain itself/);
    assert.equal(view.note, '');
    assert.equal(view.failure, null);
    assert.equal(view.meta.authorVersion, '6.113.5');
    assert.equal(view.meta.targetVersion, '6.114.0');
    // Same four buckets as the action row's "checked N ago" — one vocabulary.
    assert.equal(view.meta.ageText, '3 h ago');
    assert.equal(
        updateVerdict({ ...AVAILABLE, letter: letter() }, '').checkedAgo,
        '',
        'the letter age is not the check age',
    );
});


test('an applied letter about an older version than the running one says so', () => {
    // The kept letter describes a version this one includes but has moved past.
    const view = updateLetterView({ ...CURRENT, current_version: '6.115.0', letter: letter({ relation: 'applied' }) }, '');
    assert.equal(view.label, 'What changed in this version');
    assert.equal(view.note, 'written about 6.114.0; the running version is 6.115.0');
});


test('an applied letter is relabelled, never deleted', () => {
    const view = updateLetterView({ ...CURRENT, letter: letter({ relation: 'applied' }) }, '');
    assert.equal(view.state, 'ready');
    assert.equal(view.label, 'What changed in this version');
    assert.equal(view.note, '', 'the running version IS the target: nothing to disclaim');
    assert.match(view.markdown, /explain itself/);
});


test('a superseded letter keeps its text and says which range it was written for', () => {
    const view = updateLetterView({
        ...AVAILABLE,
        latest_version: '6.115.0',
        letter: letter({ relation: 'superseded' }),
    }, '');
    assert.equal(view.state, 'ready');
    assert.equal(view.label, "What's new");
    assert.equal(view.note, 'written for 6.113.5 → 6.114.0');
    assert.match(view.markdown, /explain itself/);
});


test('a letter whose HEAD moved elsewhere is marked, and an unnamed relation lands there too', () => {
    const moved = updateLetterView({ ...CURRENT, letter: letter({ relation: 'other' }) }, '');
    assert.equal(moved.relation, 'other');
    assert.equal(moved.label, "What's new");
    assert.equal(moved.note, 'written for 6.113.5 → 6.114.0');

    // A relation this client does not know is treated as the honest "other":
    // keep the text, mark it — never claim it describes the update on offer.
    const unnamed = updateLetterView({ ...CURRENT, letter: letter({ relation: 'sideways' }) }, '');
    assert.equal(unnamed.relation, 'other');
    assert.equal(unnamed.note, 'written for 6.113.5 → 6.114.0');

    // Versionless provenance degrades instead of printing "undefined".
    const bare = updateLetterView({
        ...CURRENT,
        letter: letter({ relation: 'other', author_version: '', target_version: '' }),
    }, '');
    assert.equal(bare.note, 'written for an earlier update');
});


test('a failed letter with a last good text shows the text plus the failure reason', () => {
    const view = updateLetterView({
        ...AVAILABLE,
        letter: letter({
            state: 'failed',
            error_kind: 'provider_unavailable',
            error_text: 'openrouter 503',
            has_last_good: true,
        }),
    }, '');
    assert.equal(view.state, 'failed');
    assert.match(view.markdown, /explain itself/, 'the last good letter survives the failed rewrite');
    assert.deepEqual(view.failure, { kind: 'provider_unavailable', text: 'openrouter 503' });
    assert.match(view.note, /rewriting this letter failed \(openrouter 503\)/);
    assert.match(view.note, /showing the last one that succeeded/);
});


test('a kept letter about an earlier target is labelled by its own range, with the failure beside it', () => {
    // The backend relates a kept letter by ITS range (update_letter.py::project_letter),
    // so a letter about 6.114.0 kept through a failed rewrite for 6.115.0 arrives as
    // superseded: the card offering 6.115.0 must not present it as that update's letter.
    const view = updateLetterView({
        ...AVAILABLE,
        latest_version: '6.115.0',
        letter: letter({
            state: 'failed', relation: 'superseded',
            error_kind: 'provider_unavailable', error_text: '503', has_last_good: true,
        }),
    }, '');
    assert.equal(view.state, 'failed');
    assert.equal(view.label, "What's new");
    assert.equal(view.meta.targetVersion, '6.114.0');
    assert.equal(
        view.note,
        'written for 6.113.5 → 6.114.0 · rewriting this letter failed (503); showing the last one that succeeded',
    );
});


test('a failed letter with no text still names why there is nothing to read', () => {
    const view = updateLetterView({
        ...AVAILABLE,
        letter: letter({ state: 'failed', text: '', error_kind: 'no_credentials', error_text: '', has_last_good: false }),
    }, '');
    assert.equal(view.state, 'failed');
    assert.equal(view.markdown, '');
    assert.deepEqual(view.failure, { kind: 'no_credentials', text: '' });
    assert.equal(view.note, 'Ouroboros could not write an update letter (no_credentials)');
});


test('an empty or unnamed letter state renders nothing rather than an empty block', () => {
    assert.equal(updateLetterView({ ...AVAILABLE, letter: letter({ text: '   ' }) }, '').state, 'none');
    assert.equal(updateLetterView({ ...AVAILABLE, letter: letter({ state: 'writing' }) }, '').state, 'none');
});


test('the letter hides wherever it could only mislead', () => {
    // Verdict states with no trustworthy update story to attach a letter to.
    const hiddenByVerdict = [
        ['unmanaged', { managed: false, letter: letter() }],
        ['check_failed', { managed: true, check_ok: false, warnings: ['fetch_error:down'], letter: letter() }],
        ['unknown', { managed: true, warnings: ['status_error:boom'], check_ok: null, available: false, letter: letter() }],
        ['unchecked', {
            managed: true, check_ok: null, available: false,
            warnings: ['official_status_requires_check'], letter: letter(),
        }],
    ];
    for (const [expected, data] of hiddenByVerdict) {
        assert.equal(updateVerdict(data, '').state, expected, `fixture no longer produces ${expected}`);
        assert.equal(updateLetterView(data, '').state, 'none', `${expected} must not carry a letter`);
    }
    // The restart phase: the served-SHA reload owns the card.
    assert.equal(updateLetterView({ ...AVAILABLE, letter: letter() }, 'restarting').state, 'none');
    // …and the phases that keep it: a passive refresh (tab reopen) or a running
    // check must not blank the last known paragraph, and an owner mid-update is
    // exactly who wants to read what the update brings.
    for (const phase of ['', 'loading', 'checking', 'preflighting', 'updating', 'restart_required', 'restart_needed']) {
        assert.equal(updateLetterView({ ...AVAILABLE, letter: letter() }, phase).state, 'ready', phase);
    }
});


test('a letter-bearing payload leaves the verdict byte-for-byte identical', () => {
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


// --- Source pins: the DOM contract a pure projector cannot see --------------

test('the letter section sits between the action row and Recovery, and adds no control', () => {
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
    // The letter is a fact, not an action: no button of its own, and above all
    // no Retry (a failed write is the backend's to retry, not a control here).
    assert.doesNotMatch(card, /<button/);
    assert.doesNotMatch(SOURCE, /Retry/);
});


test('the letter body goes through the sanitizing markdown pipeline and is disposed before re-render', () => {
    assert.match(SOURCE, /import \{ destroyChatMarkdown, enhanceChatMarkdown, renderChatMarkdown \} from '\.\/chat_markdown\.js'/);
    assert.match(SOURCE, /letterBody\.innerHTML = view\.markdown \? renderChatMarkdown\(view\.markdown\) : ''/);
    assert.match(SOURCE, /letterDisposer = enhanceChatMarkdown\(letterBody, \{[\s\S]*?onDomWrite:/);
    // Controls the pipeline adds are scrubbed after EVERY write it makes: a fenced block
    // gets its Copy button at render, a degrading mermaid block gets one asynchronously.
    assert.match(SOURCE, /function stripLetterControls\(\)[\s\S]*?querySelectorAll\('button'\)[\s\S]*?remove\(\)/);
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
    assert.match(SOURCE, /function letterContentKey\(view\) \{[\s\S]*?return \[view\.state, view\.relation, view\.markdown\]/);
    assert.doesNotMatch(SOURCE, /view\.markdown\.length/);
});


test('the letter rides the existing render path and never writes the verdict surfaces', () => {
    // No listener, timer or poll of its own: render() already runs on every
    // phase change and status load.
    assert.match(SOURCE, /\]\.includes\(verdict\.state\);\s*\n\s*renderLetter\(\);/);
    assert.equal((SOURCE.match(/renderLetter\(\)/g) || []).length, 2, 'defined once, called from render() once');

    const letterCode = SOURCE.slice(SOURCE.indexOf('function releaseLetterBody()'), SOURCE.indexOf('function render()'));
    for (const forbidden of ['dot.dataset.tone', '#updates-summary', 'summary.textContent', 'primaryBtn']) {
        assert.ok(!letterCode.includes(forbidden), `letter code must not touch ${forbidden}`);
    }
    // The apply flow stays exactly what it was: the legacy-path slice that
    // tests/test_packaged_runtime_and_lifecycle.py reads must not gain letter
    // code (or the 'replace'/'stash' literals its own guard forbids).
    const applyFn = SOURCE.split('async function applyUpdate')[1].split('\n    }')[0];
    for (const forbidden of ['letter', 'Letter', "'replace'", "'stash'"]) {
        assert.ok(!applyFn.includes(forbidden), `applyUpdate must not contain ${forbidden}`);
    }
});
