// What a private compatible key says of the server at its address, and when it offers its consent box
// (connection_server_check.js): the words come from typed facts and a Test's typed code, and no version is compared here.
import assert from 'node:assert/strict';
import test from 'node:test';

import {
    CONSENT_CODE, consentEdit, consentMatters, guardChip, guardCode, nextOutcome, paintServerCheck, providerTestTone,
    restoreConsent, retargetConsent, serverNote, testVerdict,
} from '../modules/connection_server_check.js';
import { PROVIDER_CARDS, connectionConsentHtml } from '../modules/settings_provider_cards.js';
import { setup } from './dashboard_dom_fixture.js';
import { Stub } from './helpers/settings_dom.js';

const COMPATIBLE = PROVIDER_CARDS.find((card) => card.kind === 'openai-compatible');
const RISK = 'A server that does not show which program it runs may shorten a long conversation without saying so.';
const AT = '2026-11-19T10:00:00Z';
const DAY = /\S+ \d+/;

test('the note names the server as known, from typed facts alone', () => {
    const state = (unidentified, options) => serverNote({ unidentified }, options).state;
    assert.deepEqual(serverNote({ id: 'openai-compatible@cloud' }), { state: '', text: '', admission: '' }, 'a key the check does not concern says nothing');
    assert.equal(state({ admission: '' }), 'unchecked');
    assert.equal(state({ admission: '', checked: { type: 'unknown' } }), 'unconfirmed');
    assert.equal(state({ admission: '', checked: { type: 'llama.cpp' } }), 'identified');
    assert.equal(state({ admission: '', checked: { type: 'ollama', version: '0.40.16' } }), 'identified');
    assert.equal(state({ admission: '', ollama: { type: 'ollama', version: '0.40.16' }, checked: { type: 'unknown' } }), 'held');
    assert.equal(state({ admission: '', ollama: { type: 'ollama' } }), 'held', 'Ollama seen at the address and this key never checked: held');
    assert.equal(state({ admission: 'consent', ollama: { type: 'ollama' }, checked: { type: 'LM Studio' } }), 'identified', 'a program that named itself is the key\'s own word');
    // An address typed that the facts do not speak of: the earlier reading is not about it.
    const draft = serverNote({ unidentified: { admission: 'prior', checked: { type: 'unknown', observed_at: AT } } }, { moved: true });
    assert.deepEqual([draft.state, draft.admission], ['draft', '']);
    assert.match(draft.text, /^This address has not been checked: what was read at the saved one does not apply here\. Test checks it\.$/);
    assert.doesNotMatch(draft.text, /consent|set up before|Identified|shorten/, 'none of the saved facts survives');
});

test('an admission is told for the server that names no program, never for a named one or a held address', () => {
    const checked = { type: 'unknown', observed_at: AT };
    const said = (admission, extra = {}) => serverNote({ unidentified: { admission, checked, ...extra } });
    assert.ok(said('').text.endsWith(`${RISK} A request goes to one only with your consent.`));
    assert.ok(said('prior').text.endsWith(`${RISK} This key may send to one: its server is on record answering a request before this check existed.`));
    assert.ok(said('consent').text.endsWith(RISK), 'the ticked box says the rest');
    assert.deepEqual(['', 'prior', 'consent'].map((admission) => said(admission).admission), ['', 'prior', 'consent'], 'the admission rides along for a reader that binds to it');
    const seen = { type: 'ollama', version: '0.40.16', observed_at: AT };
    for (const admission of ['', 'prior', 'consent']) {
        const held = said(admission, { ollama: seen });
        assert.match(held.text, /^Ollama 0\.40\.16 was seen at this address on \S+ \d+\. Server type not confirmed at the last check on \S+ \d+\. Nothing is sent there until this key's server identifies itself: let its metadata be read \(GET \/api\/version\) or use another key\.$/);
        assert.doesNotMatch(held.text, /consent|set up before|is Ollama|may shorten/, 'a prior route or a consent changes nothing there, and what was seen is not what this server is');
    }
});

test('an Ollama the backend says is too old is told with its version and the version to update to, and nothing else decides it', () => {
    const own = { type: 'ollama', version: '0.12.5', update_to: '0.12.6', observed_at: AT };
    const update = serverNote({ unidentified: { admission: 'consent', checked: own, ollama: { ...own } } });
    assert.equal(update.state, 'update');
    assert.match(update.text, /^Identified as Ollama 0\.12\.5 at the last check on \S+ \d+\. It can silently truncate history: update Ollama to 0\.12\.6 or later\. Nothing is sent until then\.$/);
    assert.doesNotMatch(update.text, /consent|seen at this address/, 'no consent opens it, and the address-level word is not repeated for the key\'s own check');
    // The same version with no `update_to` is no verdict of this module: it compares nothing.
    const bare = serverNote({ unidentified: { admission: '', checked: { type: 'ollama', version: '0.12.5', observed_at: AT } } });
    assert.equal(bare.state, 'identified');
    assert.match(bare.text, /^Identified as Ollama 0\.12\.5 at the last check on \S+ \d+\.$/);
    assert.doesNotMatch(bare.text, /update|truncate/);
    // An old Ollama seen at the address, by whichever key: the hold's own words, with what it needs, never "this server is Ollama".
    const held = serverNote({ unidentified: { admission: '', ollama: { type: 'ollama', version: '0.12.5', update_to: '0.12.6', observed_at: AT }, checked: { type: 'unknown', observed_at: AT } } });
    assert.equal(held.state, 'held');
    assert.match(held.text, /^Ollama 0\.12\.5 was seen at this address on \S+ \d+\. It can silently truncate history until updated to 0\.12\.6 or later\. Server type not confirmed/);
    assert.doesNotMatch(held.text, /Identified as/, 'an address-level observation is no identification of this key\'s server');
});

test('a consent can matter only to a private server that names no program and has no Ollama seen at its address', () => {
    assert.equal(consentMatters(null), false, 'a public address keeps no such fact');
    assert.equal(consentMatters({ admission: '' }), true);
    assert.equal(consentMatters({ checked: { type: 'unknown' } }), true);
    assert.equal(consentMatters({ checked: { type: 'llama.cpp' } }), false);
    assert.equal(consentMatters({ checked: { type: 'unknown' }, ollama: { type: 'ollama' } }), false);
});

/** A key's box and note as a host draws them. */
function scope(t) {
    setup(t);
    const root = new Stub('div');
    root.innerHTML = `<div>${connectionConsentHtml(COMPATIBLE)}<p class="settings-card-note" data-connection-unidentified hidden></p></div>`;
    return { root, label: () => root.querySelector('.accounts-key-consent'), box: () => root.querySelector('[data-connection-consent]'),
        note: () => root.querySelector('[data-connection-unidentified]') };
}

test('the box is offered by the facts, by a Test that asked for it, or while it is ticked; a hold takes the offer back', (t) => {
    const here = scope(t);
    const entry = (unidentified) => (unidentified ? { unidentified } : null);
    const paint = (options) => { paintServerCheck(here.root, options); return [here.label().hidden, here.label().dataset.consentOffer ?? '']; };
    restoreConsent(here.box(), { stands: false, url: 'http://lab.lan/v1' });
    assert.deepEqual(paint({}), [true, ''], 'a draft nobody knows anything of offers nothing');
    assert.deepEqual(paint({ entry: entry({ admission: '', checked: { type: 'unknown' } }) }), [false, 'facts']);
    assert.deepEqual(paint({ entry: entry({ admission: '', checked: { type: 'llama.cpp' } }) }), [true, ''], 'an identified server needs none');
    // A previously identified server that no longer names its program: the Test that was refused for want of the consent opens the box.
    const known = entry({ admission: '', checked: { type: 'llama.cpp', observed_at: AT } });
    assert.deepEqual(paint({ entry: known, outcome: CONSENT_CODE }), [false, 'test']);
    assert.equal(here.box().checked, false, 'it opens unticked');
    // The address changed: the saved facts are not about it, and the Test of the old one is gone with it.
    assert.deepEqual(paint({ entry: entry({ admission: '', checked: { type: 'unknown' } }), moved: true }), [true, '']);
    // A hold or an old Ollama is found by a Test: no consent lifts either, so nothing is offered (the facts did not know yet).
    for (const code of ['ollama_guard_held', 'ollama_guard_unavailable']) {
        assert.deepEqual(paint({ entry: entry({ admission: '', checked: { type: 'unknown' } }), outcome: code }), [true, ''], code);
    }
    // A consent that stands is shown whatever else is known, to be withdrawn.
    here.box().checked = true;
    assert.deepEqual(paint({ entry: entry({ admission: 'consent', ollama: { type: 'ollama' }, checked: { type: 'unknown' } }) }), [false, 'ticked']);
    assert.deepEqual(paint({ entry: known, outcome: 'ollama_guard_held' }), [false, 'ticked']);
});

test('the note carries its state and admission as stable hooks, and is gone when it has nothing to say', (t) => {
    const here = scope(t);
    const unidentified = { admission: 'prior', checked: { type: 'unknown', observed_at: AT } };
    paintServerCheck(here.root, { entry: { unidentified } });
    assert.deepEqual([here.note().hidden, here.note().dataset.serverState, here.note().dataset.admission], [false, 'unconfirmed', 'prior']);
    paintServerCheck(here.root, { entry: { unidentified }, moved: true });
    assert.deepEqual([here.note().hidden, here.note().dataset.serverState, here.note().dataset.admission], [false, 'draft', undefined]);
    paintServerCheck(here.root, { entry: null });
    assert.deepEqual([here.note().hidden, here.note().textContent, here.note().dataset.serverState], [true, '', undefined]);
});

test('the box stands for the address shown: a new one clears it, the stored one brings it back, and each is an edit only against what was read', (t) => {
    const { box } = scope(t);
    restoreConsent(box(), { stands: true, url: 'http://lab.lan/v1' });
    assert.deepEqual([box().checked, consentEdit(box(), 'http://lab.lan/v1')], [true, null], 'as read: no edit');
    retargetConsent(box(), 'http://other.lan/v1');
    assert.deepEqual([box().checked, consentEdit(box(), 'http://other.lan/v1')], [false, ''], 'a consent that stood is withdrawn');
    retargetConsent(box(), 'http://lab.lan/v1');
    assert.deepEqual([box().checked, consentEdit(box(), 'http://lab.lan/v1')], [true, null], 'typed back, the consent that stands covers it again');
    restoreConsent(box(), { stands: false, url: '' });
    assert.equal(consentEdit(box(), 'http://lab.lan/v1'), null, 'consent starts false: nothing is given until the box is ticked');
    box().checked = true;
    assert.equal(consentEdit(box(), 'http://lab.lan/v1'), 'http://lab.lan/v1', 'ticked, it names the address shown');
    assert.equal(consentEdit(null, 'http://lab.lan/v1'), null, 'a key with no box has none');
});

test('a box keeps two readings of its consent: what is stored, which a host that shows a draft re-bases, and the draft as it was read', (t) => {
    const { box } = scope(t);
    const edits = () => [consentEdit(box(), 'http://lab.lan/v1'), consentEdit(box(), 'http://lab.lan/v1', 'mounted')];
    restoreConsent(box(), { stands: true, url: 'http://lab.lan/v1' });
    assert.deepEqual(edits(), [null, null], 'read once, both readings are the same');
    Object.assign(box().dataset, { applied: '0', appliedUrl: '' });  // what is stored holds none: the draft's consent is one a Test adds
    assert.deepEqual(edits(), ['http://lab.lan/v1', null], 'ticked as the draft held it: an edit of what is stored, none of the draft');
    box().checked = false;
    assert.deepEqual(edits(), [null, ''], 'cleared: nothing to take from what is stored, a consent the draft held is withdrawn');
});

test('a Test\'s typed code decides its tone, its chip and what the box does next', () => {
    const refused = { ok: false, code: CONSENT_CODE, error: 'The server at http://lab.lan does not show which program it runs; nothing was sent.' };
    assert.deepEqual(testVerdict(refused), { text: `Not ready — ${refused.error}`, tone: 'warn', code: CONSENT_CODE });
    assert.equal(providerTestTone({ ok: false, error: 'Invalid key' }), 'danger');
    assert.equal(providerTestTone({ ok: false, code: 'ollama_guard_held', error: 'held' }), 'warn');
    assert.deepEqual(['ollama_guard_consent', 'ollama_guard_held', 'ollama_guard_unavailable'].map(guardChip),
        ['Needs your consent', 'Ollama was seen at this address', 'Ollama needs an update']);
    assert.equal(guardCode({ code: 'rate_limited' }), '', 'a code outside the closed set is no answer of the check');
    assert.equal(guardChip('rate_limited'), '');
    // What the draft's Test left for its box.
    assert.equal(nextOutcome(refused), CONSENT_CODE);
    assert.equal(nextOutcome({ ok: true }, CONSENT_CODE), '', 'passed on a server that named its program: nothing is asked');
    assert.equal(nextOutcome({ ok: true, caution: 'Server type not confirmed' }, CONSENT_CODE), CONSENT_CODE, 'passed on the consent: it still matters');
    assert.equal(nextOutcome(null, CONSENT_CODE), CONSENT_CODE, 'a Test that failed otherwise says nothing of it');
    assert.equal(nextOutcome({ ok: false, error: 'Invalid key' }, ''), '');
});
