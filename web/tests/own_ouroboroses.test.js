// Your Ouroboroses (own_ouroboroses.js, peer_intro.js, peer_letter_card.js, peer_links_api.js, ouroboros_face.js; DESIGN
// "Circles and Add"): a linked place reads its link from /api/places; its row says the state and who may give whom work
// in words with names, and its conversation opens in the place sheet like any place's; the introduction suggests a rule
// for the owner's own; a letter's card at its sender has one status line and one next line for each state of the
// contract, and no button.
import assert from 'node:assert/strict';
import test from 'node:test';

import { normalizePlacesResponse } from '../modules/consciousness_api.js';
import { FACES, faceOf, faceSvg } from '../modules/ouroboros_face.js';
import { peerPlaceFacts } from '../modules/peer_links_api.js';
import { circlePlaces, codeWords, createOwnOuroboroses, directionWords, linkStateOf, ownBlockHtml, ownOrder, ownRowHtml, rowLine, ruleNote, workRowHtml, wroteRoom } from '../modules/own_ouroboroses.js';
import { consequenceWords, introFailure, introView, inviteParts, ruleSuggestion } from '../modules/peer_intro.js';
import { endLetters, heldWords, letterCardView, letterHeadHtml, letterRow, letterStatusHtml, refusalWords, usd } from '../modules/peer_letter_card.js';

const strip = (html) => html.replace(/<[^>]+>/g, ' ').replace(/&#39;/g, "'").replace(/&quot;/g, '"').replace(/&amp;/g, '&').replace(/\s+/g, ' ').trim();
const today = (h, m) => { const d = new Date(); d.setHours(h, m, 0, 0); return d.toISOString(); };
const peerRaw = (fields = {}) => ({ place_id: 'p-tarel', connector: 'ouroboros', provider: 'ouroboros', label: 'Tarel', state: 'active',
    account_id: 'install-tarel', revision: 4, notes: [{ note_id: 'n1', text: 'Tarel is my own Ouroboros.', author: 'owner', at: today(13, 41) }],
    wakes: [], owner_accounts: [], peer: { mine: true, may_start: true }, peer_facts: 'Host facts…',
    peer_effective: { messages: true, assignments: true, reason: '' },
    link: { link_id: 'link-1', code: '418207', code_check: 'confirmed', peer_install_id: 'install-tarel', created_at: today(13, 41), revoked_at: '', remote_may_start: true, remote_unknown_at: '' },
    rooms: [], last_activity_at: '', ...fields });
const view = (...places) => normalizePlacesResponse({ connectors: [], circles: [], places });
const place = (fields) => view(peerRaw(fields)).places[0];
// The host's reading when my human's switch says no: messages still saved, work refused (peer_links.effective_permissions).
const NO_WORK = { peer: { mine: true, may_start: false }, peer_effective: { messages: true, assignments: false, reason: 'may_not_start' } };
const UI = { open: '', editing: '', drafts: {}, busy: '', pending: null };

test('a face is the app icon in one colour of the free palette, the same for the same install id, never the icon itself', () => {
    assert.ok(FACES.length > 20 && FACES.every((face) => face.body.startsWith('#')));
    assert.equal(faceOf('install-tarel').id, faceOf('install-tarel').id);
    assert.notEqual(faceOf('install-tarel').id, faceOf('install-velm').id);
    assert.ok(!FACES.some((face) => face.body === '#130d11'));
    assert.match(faceSvg('', 36, { own: true }), /fill="#130d11"/);
    assert.match(faceSvg('install-tarel', 20), /width="20" height="20"/);
});

test('only a linked place reads its link; every other place keeps its shape', () => {
    assert.deepEqual(peerPlaceFacts({ connector: 'telegram', link: { link_id: 'x' } }), {});
    const tarel = place({ link: { ...peerRaw().link, remote_unknown_at: today(15, 2) } });
    assert.equal(tarel.link.code, '418207');
    assert.equal(tarel.peer.mayStart, true);
    assert.deepEqual(tarel.peerEffective, { messages: true, assignments: true, reason: '' }, 'the host\'s reading, carried as said');
    assert.deepEqual(place({ peer_effective: { messages: null, assignments: null, reason: 'unknown' } }).peerEffective,
        { messages: null, assignments: null, reason: 'unknown' }, 'unknown stays unknown, never a refusal or a permission');
    assert.equal(tarel.link.remoteUnknownAt, today(15, 2));
    assert.ok(!('letters' in tarel) && !('address' in tarel), 'its conversation is the place\'s own, no feed of its own');
    const telegram = view({ place_id: 't', connector: 'telegram', label: 'T', state: 'active' }).places[0];
    assert.ok(!('link' in telegram));
    assert.deepEqual(circlePlaces([tarel, telegram]).map((p) => p.id), ['t']);
});

test('the link state on this side: unknown, removed, no longer known there, waiting for the check, paused, connected', () => {
    assert.equal(linkStateOf(place({ link: null })), 'unknown');
    assert.equal(linkStateOf(place({ link: { ...peerRaw().link, revoked_at: today(15, 2) } })), 'removed');
    assert.equal(linkStateOf(place({ link: { ...peerRaw().link, remote_unknown_at: today(15, 2) } })), 'lost');
    assert.equal(linkStateOf(place({ state: 'paused', link: { ...peerRaw().link, code_check: 'pending' } })), 'check');
    assert.equal(linkStateOf(place({ state: 'paused' })), 'paused');
    assert.equal(linkStateOf(place({ state: 'silent' })), 'silent', 'Silent is its own state, never «Connected»');
    assert.equal(linkStateOf(place()), 'active');
});

test('Silent, Paused and a removed link read apart: each its own word, and Silent keeps the link and its messages', () => {
    const open = { ...UI, open: 'p-tarel' };
    const silent = place({ state: 'silent', peer_effective: { messages: true, assignments: false, reason: 'silent' } });
    const closed = strip(ownRowHtml(silent, UI));
    assert.match(closed, /Tarel no captured event Silent$/, 'no direction promised while Silent');
    const body = ownRowHtml(silent, open);
    assert.match(strip(body), /Requests from Tarel Tarel is Silent here, so work it would start could not answer: its requests are refused\./);
    assert.match(strip(body), /Requests to Tarel This side sends nothing while the link is Silent or Paused\./);
    assert.match(strip(body), /Link Silent: what Tarel writes is saved here, nothing is sent to it\. The keys stay\. Active Silent Paused/);
    assert.match(body, /class="ui-segment active" data-effort-value="silent" aria-pressed="true"/, 'the place\'s own three states');
    const words = [place({ state: 'paused' }), silent, place({ link: { ...peerRaw().link, revoked_at: today(15, 2) } })]
        .map((one) => strip(ownRowHtml(one, UI)).split(' ').slice(-2).join(' '));
    assert.equal(new Set(words).size, 3, words.join(' | '));
});

test('incoming work uses the live reading; a past choice abroad cannot fill an unknown local reading', () => {
    assert.equal(directionWords(place()), 'Can ask for work here');
    assert.equal(directionWords(place(NO_WORK)), 'Cannot start work here');
    assert.equal(directionWords(place({ link: { ...peerRaw().link, remote_may_start: false } })), 'Can ask for work here');
    assert.equal(directionWords(place({ peer_effective: { messages: null, assignments: null, reason: 'unknown' } })), 'Permission to start work here is unknown');
});

test('a closed row: face, name, its last captured event as every place row says it, the state word and the direction', () => {
    const recent = new Date(Date.now() - 5 * 60_000).toISOString();
    const tarel = place({ last_activity_at: recent });
    assert.equal(rowLine(tarel), '5 min ago');
    const html = ownRowHtml(tarel, UI);
    assert.match(html, /<details class="ui-disclosure own-link" data-own-row="p-tarel" data-state="active">/);
    assert.match(strip(html), /Tarel 5 min ago Connected Can ask for work here/);
    assert.match(html, /class="peer-face"/);
    assert.equal(rowLine(place()), 'no captured event', 'never «no activity» and never a letter count it cannot see');
    assert.match(strip(ownRowHtml(place({ state: 'paused', link: { ...peerRaw().link, code_check: 'pending' } }), UI)), /introduced today 13:41 Waiting for your check$/);
});

test('an open connected row: the owner\'s rule with Edit, requests from as a switch, requests to as a fact, the link strip', () => {
    const html = ownRowHtml(place(), { ...UI, open: 'p-tarel' });
    const text = strip(html);
    assert.match(text, /Your rule for Tarel Edit Tarel is my own Ouroboros\./);
    assert.match(html, /<div class="place-note circle-words-text is-owner"><p data-i18n-authored>Tarel is my own Ouroboros\.<\/p><\/div>/,
        'the rule on record is the owner\'s words in their voice, the circle words\' part');
    assert.doesNotMatch(html, /chat-quote/);
    // Each result has its line beside its control, written after the paint: none is a status paragraph of the body.
    for (const slot of ['p-tarel:rule', 'p-tarel:from', 'p-tarel:state', 'p-tarel:links']) {
        assert.match(html, new RegExp(`class="ui-result[^"]*" data-own-result="${slot}"`), slot);
    }
    assert.doesNotMatch(html, /<p class="ui-status"/);
    assert.match(text, /Requests from Tarel Tarel can ask this Ouroboros for work\./);
    assert.match(text, /Requests to Tarel Allowed at introduction\. The current choice there may differ\./);
    assert.match(text, /Link Code 418 207 · introduced today 13:41 Active Silent Paused Open conversation Remove link/);
    assert.match(html, /data-own-act="conversation" data-place="p-tarel">Open conversation</, 'the place sheet\'s own door, no feed of its own');
    assert.match(html, /role="switch" data-own-switch="p-tarel"[^>]* aria-label="Requests from Tarel" aria-checked="true" checked>/);
    assert.match(html, /class="ui-segment active" data-effort-value="active" aria-pressed="true"/);
    assert.match(html, /role="group" aria-label="Link"/);
    // Requests from it off forbid only starting work here: no «only answers».
    const off = strip(ownRowHtml(place(NO_WORK), { ...UI, open: 'p-tarel' }));
    assert.match(off, /Requests from Tarel Tarel cannot start work here\./);
    assert.doesNotMatch(off, /only answers/);
    const words = (effect) => strip(ownRowHtml(place({ peer_effective: { messages: true, assignments: false, ...effect } }), { ...UI, open: 'p-tarel' }));
    assert.match(words({ reason: 'silent' }), /Tarel is Silent here, so work it would start could not answer: its requests are refused\./);
    assert.match(words({ reason: 'panic' }), /A Panic stopped this Ouroboros: requests from Tarel are refused until you return\./);
    assert.match(strip(ownRowHtml(place({ peer_effective: { messages: null, assignments: null, reason: 'unknown' } }), { ...UI, open: 'p-tarel' })),
        /Whether Tarel can start work here is not known now\./);
    const paused = strip(ownRowHtml(place({ state: 'paused', peer_effective: { messages: false, assignments: false, reason: 'paused' } }), { ...UI, open: 'p-tarel' }));
    assert.match(paused, /While the link is paused, requests from Tarel are refused\./);
    assert.match(paused, /Nothing goes either way until you choose Active\. The keys stay\./);
    assert.match(ownRowHtml(place({ state: 'paused' }), { ...UI, open: 'p-tarel' }), /data-own-switch="p-tarel"[^>]*disabled>/);
});

test('a row that is not simply connected says what is going on and its one next step first', () => {
    const open = { ...UI, open: 'p-tarel' };
    const check = strip(ownRowHtml(place({ state: 'paused', link: { ...peerRaw().link, code_check: 'pending' } }), open));
    assert.match(check, /The link with Tarel starts after you check the code on both sides\. Check the code Remove link Your rule for Tarel/);
    const removed = strip(ownRowHtml(place({ link: { ...peerRaw().link, revoked_at: today(15, 2) } }), open));
    assert.match(removed, /removed today 15:02 Link removed The link is removed on this side.*Introduce again Open conversation/);
    const lost = strip(ownRowHtml(place({ link: { ...peerRaw().link, remote_unknown_at: today(15, 2) } }), open));
    assert.match(lost, /Needs a new introduction Tarel no longer knows this link: it was removed there, or a data folder moved\./);
    assert.match(lost, /Introduce again Open conversation Remove link/);
    assert.equal(ruleNote(place({ notes: [] })), null);
});

test('the block: its title and one action; with none yet, what it is for', () => {
    assert.match(strip(ownBlockHtml([], UI)), /^Your Ouroboroses Introduce an Ouroboros None yet\. Another Ouroboros of yours can write here and, if allowed, ask for work\.$/);
    assert.equal((ownBlockHtml([place(), place({ place_id: 'p-velm', label: 'Velm' })], UI).match(/<li>/g) || []).length, 2);
    assert.equal(codeWords('905132'), '905 132');
});

test('the rows keep the order they were introduced in, the oldest first; an unread link comes last', () => {
    const at = (h) => ({ ...peerRaw().link, created_at: today(h, 0) });
    const velm = place({ place_id: 'p-velm', label: 'Velm', link: at(15) }), tarel = place({ link: at(13) });
    const lost = place({ place_id: 'p-0', label: 'Lost', link: null });
    assert.deepEqual(ownOrder([lost, velm, tarel]).map((p) => p.label), ['Tarel', 'Velm', 'Lost']);
    const html = ownBlockHtml([velm, tarel], UI);
    assert.ok(html.indexOf('data-own-row="p-tarel"') < html.indexOf('data-own-row="p-velm"'), 'a new one never jumps above');
});

test('while a write is out its control shows the value it writes, held; the others stay as stored', () => {
    const open = { ...UI, open: 'p-tarel' };
    const switching = ownRowHtml(place(NO_WORK), { ...open, busy: 'p-tarel', pending: { field: 'mayStart', value: true } });
    assert.match(switching, /data-own-switch="p-tarel"[^>]*aria-checked="true" checked disabled aria-busy="true">/);
    assert.match(strip(switching), /Tarel cannot start work here\./, 'the words stay the host\'s last reading until it is read again');
    assert.match(switching, /class="ui-segment active" data-effort-value="active" aria-pressed="true">/, 'the strip keeps its stored choice');
    const pausing = ownRowHtml(place(), { ...open, busy: 'p-tarel', pending: { field: 'state', value: 'paused' } });
    assert.match(pausing, /aria-label="Link" aria-busy="true"/);
    assert.match(pausing, /class="ui-segment active" data-effort-value="paused" aria-pressed="true" disabled>/);
    assert.match(strip(pausing), /Tarel can ask this Ouroboros for work\./, 'write still in flight: host facts have not changed');
    assert.doesNotMatch(ownRowHtml(place(), open), /aria-busy/);
});

test('Open conversation asks the page to open the place sheet for that place; nothing else is read or sent', async () => {
    const asked = [];
    const listeners = {};
    const body = { addEventListener: (type, fn) => { listeners[type] = fn; }, removeEventListener: () => {},
        contains: () => true, querySelector: () => null };
    const tarel = place({ rooms: [{ room_chat_id: 3381 }] });
    globalThis.CSS = globalThis.CSS || { escape: (value) => String(value) };
    const own = createOwnOuroboroses(body, { places: () => [tarel], paint: () => {}, refresh: async () => {},
        openIntro: () => {}, openCodeCheck: () => {}, openConversation: (subject, options) => asked.push([subject.id, typeof options.returnFocus]) });
    const button = { dataset: { ownAct: 'conversation', place: 'p-tarel' }, disabled: false, closest: (selector) => (selector === '[data-own-act]' ? button : null) };
    listeners.click({ target: { closest: (selector) => (selector === '[data-own-act]' ? button : null) } });
    assert.deepEqual(asked, [['p-tarel', 'function']]);
    assert.deepEqual(tarel.rooms, [{ chatId: 3381 }], 'the room the page opens it at is the place\'s own');
    own.dispose();
});

test('the introduction suggests a rule for the owner\'s own Ouroboros: what it may ask for and what it may be told', () => {
    const own = ruleSuggestion({ name: 'Tarel' });
    assert.match(own, /^Tarel is my linked Ouroboros\. If allowed to ask for work here, do its ordinary requests; its requests are its own, not mine\. Tell it what its task needs; private matters stay inside their circles\. Rights, keys and money come only from me\.$/);
    assert.doesNotMatch(own, /as if I had asked/, 'its requests never pass for the owner\'s own words');
    assert.match(ruleSuggestion({}), /^It is my linked Ouroboros\./);
    assert.equal(consequenceWords({ name: '' }), 'The one you invite can write but cannot start work here.');
    assert.equal(consequenceWords({ name: 'Velm', mayStart: true }), 'Velm can write and ask for work here.');
    assert.equal(introFailure({ body: { refusal_code: 'invite_expired' } }), 'This invitation has expired. Ask for a new one.');
});

test('the sheet\'s steps: the rule and Create invitation, Connect only after the invitation was read, the code check', () => {
    const form = { mode: 'invite', name: '', rule: ruleSuggestion({}), ruleEdited: false, invite: '', preview: null, previewError: '' };
    const invite = introView({ step: 'compose', form });
    assert.match(strip(invite.body), /Who can log in to the other Ouroboros’s computer\? Only me Other people too/);
    assert.match(invite.actions, /data-intro-go disabled>Create invitation/, 'no assumed direction');
    for (const [mayStart, words] of [[true, 'can write and ask for work here.'], [false, 'can write but cannot start work here.']]) {
        const chosen = introView({ step: 'compose', form: { ...form, name: 'Tarel', mayStart } });
        assert.ok(strip(chosen.body).includes(`Tarel ${words}`));
        assert.match(chosen.actions, /data-intro-go>Create invitation/);
    }
    // Ouroboros's suggestion is not the owner's words until they edit it: the field's own ink, a help that says whose.
    assert.match(invite.body, /<textarea class="ui-control peer-rule-field" rows="4" data-intro-rule/);
    assert.match(strip(invite.body), /Suggested by Ouroboros\. Edit it freely\./);
    const edited = introView({ step: 'compose', form: { ...form, ruleEdited: true } });
    assert.match(edited.body, /class="ui-control peer-rule-field consciousness-owner-words"/);
    assert.match(strip(edited.body), /Your words become the place’s first note\./);
    // What the introduction lets the other do here is a setting's value: a help line, no status tone.
    assert.match(invite.body, /<p class="ui-field-help" data-intro-then data-i18n-skip>Both of you can write\./);
    // Its fields are named by their labels.
    assert.match(invite.body, /<label class="ui-field"><span class="peer-field-label">Name<\/span><input class="ui-control"/);
    // A failed step says so once, however often it fails.
    assert.equal((introView({ step: 'compose', form, error: 'The other Ouroboros did not answer. Is it running?' }).body.match(/role="alert"/g) || []).length, 1);
    // The invitation on one line beside Copy, its value whole in the page; the rule, sent, is the owner's words.
    const invitation = { invite: 'ouroboros-invite:1:' + 'x'.repeat(388) + 'tail-1234567', linkId: 'l', expiresAt: '' };
    const invited = introView({ step: 'invited', invitation, form: { ...form, name: 'Tarel', rule: 'Tarel is mine.' } });
    const line = invited.body.match(/<div class="ui-action-row peer-invite-row"><output class="ui-control ui-control-readonly peer-invite-field" tabindex="0" data-intro-token data-i18n-authored aria-labelledby="peer-intro-invite-label"[^>]*><span class="peer-invite-head">([^<]*)<\/span><span class="peer-invite-tail">([^<]*)<\/span><\/output>\s*<button type="button" class="btn" data-primary data-intro-copy>Copy<\/button><\/div><span class="ui-result" data-intro-copied/);
    assert.ok(line, 'one line: the value, then Copy as the row\'s primary (crimson), its result under them');
    assert.equal(line[1] + line[2], invitation.invite, 'the page holds the value whole, character for character');
    assert.equal(line[2], 'tail-1234567', 'its last characters stay in view after the ellipsis');
    assert.deepEqual(inviteParts('ouroboros-invite:1:short'), ['ouroboros-invite:1:short', ''], 'a short one is one part');
    assert.match(invited.body, /<span class="peer-field-label" id="peer-intro-invite-label">Invitation<\/span>/);
    assert.doesNotMatch(invited.body, /title="/, 'nothing in a title');
    assert.match(invited.body, /<div class="place-note circle-words-text is-owner"><p data-i18n-authored>Tarel is mine\.<\/p>/);
    assert.doesNotMatch(invited.actions, /data-primary/);
    const expired = introView({ step: 'invited', invitation, expired: true, form: { ...form, name: 'Tarel' } });
    assert.match(expired.body, /class="btn btn-default" data-intro-copy disabled>Copy/);
    assert.match(expired.actions, /data-primary data-intro-close>Close/, 'expired, Close is the one way on and focus finds it');
    const accept = introView({ step: 'compose', form: { ...form, mode: 'accept' } });
    assert.match(accept.actions, /data-intro-go disabled>Connect/);
    const code = introView({ step: 'code', link: { name: 'Tarel', code: '418207', seed: 'install-tarel' }, form });
    assert.equal(code.title, 'Check the code');
    assert.match(strip(code.body), /Code here 418 207 Tarel shows a code too\. Do both show 418 207\? The link starts only after you say they match\./);
    assert.match(code.actions, /They differ.*They match/);
    assert.match(code.body, /<span class="peer-pair-link" aria-hidden="true"><svg [^>]*data-glyph="fluent"/, 'the link between the faces is the house glyph');
    assert.doesNotMatch(code.body, /[↔⟷]/);
});

test('a letter at its sender: one status line and one next line for each state, and no button', () => {
    const base = { letter_id: 'L1', to: 'Tarel', to_seed: 'install-tarel', text: 'Найди причину.', at: today(14, 2), ceiling_usd: 5 };
    const words = (fact) => { const v = letterCardView({ ...base, ...fact }); return [v.word, v.rest, v.next].join(' | '); };
    assert.equal(words({ state: 'accepted_running', phase: 'running' }), 'Tarel is working on it | limit $5.00 of Tarel\'s budget | The answer will come here.');
    assert.match(words({ state: 'accepted_running', phase: 'queued' }), /^Queued at Tarel \|/);
    assert.match(words({ state: 'needs_input' }), /^Tarel asked about the letter \| limit \$5\.00 of Tarel's budget \| Ouroboros answers it in this work/);
    assert.match(words({ state: 'waiting_owner' }), /^Waiting for you at Tarel \|/);
    assert.equal(words({ state: 'unconfirmed' }), 'Not confirmed |  | Tarel did not confirm that it got the letter. Nothing is sent again on its own.');
    assert.equal(words({ state: 'unknown' }), 'State unknown |  | ', 'a state not stated is never «working»');
    assert.equal(words({ state: 'refused', refusal_code: 'link_paused' }), 'Refused | no work was started | The link is paused at Tarel, or its code is not checked there yet. Turn it on in Tarel\'s Circles.');
    assert.equal(words({ state: 'not_sent', refusal_code: 'place_paused' }), 'Not sent | nothing left this computer | The link with Tarel is paused here. Turn it on in Circles.');
    assert.equal(words({ state: 'not_sent', refusal_code: 'owner_paused' }), 'Not sent | nothing left this computer | Nothing was sent (owner_paused).');
    assert.equal(refusalWords('peer_may_not_start', 'Tarel'), 'Tarel does not take requests from Ouroboros. Turn them on in Tarel\'s Circles.');
    assert.equal(refusalWords('no_budget', 'Tarel'), 'Tarel has no budget left. Raise it in Tarel\'s Costs.');
    assert.equal(refusalWords('protocol_unsupported', 'Tarel'), 'Tarel\'s version cannot read this letter. Update Tarel.');
    // The end: a reply only on the receiver's explicit word (`reply` from `result_available`), the total only when final.
    assert.equal(words({ state: 'done', reply: 'given', answer: 'ok', spent_usd: 1.2, spent_final: true }), 'Answered | spent $1.20 of $5.00 from Tarel\'s budget | ');
    assert.match(words({ state: 'done', reply: 'given', spent_usd: null, answer: null }), /^Answered \| spend unknown \| The answer is too long for this card/);
    assert.equal(words({ state: 'done', reply: 'given', answer: '', spent_usd: 0.3, spent_final: false }), 'Answered | spent $0.30 so far from Tarel\'s budget | The reply is empty.');
    assert.equal(words({ state: 'done', reply: 'none', spent_usd: 0.3, spent_final: true }), 'Finished | spent $0.30 of $5.00 from Tarel\'s budget | Work finished without a reply for this Ouroboros.');
    assert.equal(words({ state: 'done', spent_usd: null }), 'Finished | reply unknown · spend unknown | ', 'no stated reply is neither an answer nor a refusal');
    assert.equal(words({ state: 'done', reply: 'unknown', answer: 'B\'s own final', spent_usd: null }), 'Finished | reply unknown · spend unknown | ');
    assert.equal(usd(5), '$5.00');
    assert.match(strip(letterHeadHtml(base, { ts: base.at })), /^Ouroboros Tarel \d\d:\d\d$/);
    assert.match(strip(letterHeadHtml(base, { answer: true, ts: base.at })), /^Tarel Ouroboros \d\d:\d\d$/, 'the answer comes back');
    const status = letterStatusHtml({ ...base, state: 'refused', refusal_code: 'link_paused' });
    assert.match(status, /data-tone="error"/);
    assert.doesNotMatch(status + letterHeadHtml(base), /<button/, 'a letter stops with its work: the card has no button');
});

/** A room's feed and its addMessage, just enough for letterRow: parts mount in order and are found by their data. */
function letterRoom() {
    const nodes = [];
    const node = (text) => {
        const el = { text, html: '', classes: new Set(), dataset: {}, attrs: {}, host: null, isConnected: true };
        el.classList = { add: (...names) => names.forEach((name) => el.classes.add(name)) };
        el.setAttribute = (key, value) => { el.attrs[key] = String(value); };
        el.insertAdjacentHTML = (_where, html) => { el.html = html + el.html; if (html.includes('peer-letter-state')) el.host = { innerHTML: '' }; };
        el.querySelector = (selector) => (selector === '.peer-letter-state' ? el.host : null);
        return el;
    };
    const feed = { querySelectorAll: (selector) => (selector === '.peer-letter-card' ? nodes.filter((el) => el.classes.has('peer-letter-card')) : []) };
    const calls = [];
    const add = (text, role, markdown, ts, isProgress, opts) => { calls.push({ text, role, ts, isProgress, opts }); const el = node(text); nodes.push(el); return el; };
    return { feed, add, calls, nodes };
}

test('a room draws a letter once where its first fact stands, rewrites it in place and mounts its answer once', () => {
    const room = letterRoom();
    const fact = { letter_id: 'L1', to: 'Velm', to_seed: 'install-velm', text: 'Count the files.', ceiling_usd: 1, spent_usd: null };
    let tick = 0;
    const row = (extra, history = '') => ({ task_id: 't-a', ts: new Date(Date.parse(today(14, 3)) + 1000 * tick++).toISOString(), ...(history ? { history_id: history } : {}), peer_letter_card: { ...fact, ...extra } });
    assert.equal(letterRow({ task_id: 't-a', text: 'plain progress' }, room.feed, room.add), false, 'any other row is the room\'s');
    assert.equal(letterRow(row({ state: 'accepted_running', phase: 'queued', at: today(14, 2) }, 'h1'), room.feed, room.add), true);
    assert.equal(room.calls.length, 1);
    assert.deepEqual([room.calls[0].text, room.calls[0].role, room.calls[0].isProgress, room.calls[0].opts.systemType, room.calls[0].opts.historyId],
        ['Count the files.', 'assistant', false, 'peer_letter', 'h1']);
    const card = room.nodes[0];
    assert.ok(card.classes.has('chat-relay-note') && card.dataset.letterId === 'L1' && card.dataset.letterPart === 'letter');
    assert.match(strip(card.host.innerHTML), /^Queued at Velm limit \$1\.00 of Velm's budget The answer will come here\.$/);
    letterRow(row({ state: 'accepted_running', phase: 'running' }, 'h2'), room.feed, room.add);
    letterRow(row({ state: 'accepted_running', phase: 'running' }), room.feed, room.add);
    assert.equal(room.calls.length, 1, 'a later fact rewrites the card in place');
    assert.match(card.attrs['aria-label'], /^Letter from Ouroboros to Velm: Velm is working on it, limit \$1\.00 of Velm's budget$/);
    // The receiver's question and this side's answer to it: a note each, once, in the order their facts came.
    letterRow(row({ state: 'needs_input', question: { question_id: 'q1', text: 'Which folder?' } }, 'h2q'), room.feed, room.add);
    letterRow(row({ state: 'needs_input', question: { question_id: 'q1', text: 'Which folder?' } }), room.feed, room.add);
    letterRow(row({ state: 'needs_input', answered: { question_id: 'q1', text: '/tmp' } }, 'h2a'), room.feed, room.add);
    assert.deepEqual(room.nodes.map((node) => node.dataset.letterPart), ['letter', 'question:q1', 'answered:q1']);
    assert.match(strip(room.nodes[1].html), /^Velm Ouroboros \d\d:\d\d Question about the letter Which folder\?$/);
    assert.match(strip(room.nodes[2].html), /^Ouroboros Velm \d\d:\d\d Answer to the question \/tmp$/);
    assert.equal(room.nodes[2].attrs['aria-label'], 'Answer from Ouroboros to Velm to its question');
    // Once this side's answer was delivered (its note stands), the card no longer asks, also on the next read.
    assert.match(strip(card.host.innerHTML), /^Answered Velm's question limit \$1\.00 of Velm's budget The answer will come here\.$/);
    letterRow(row({ state: 'needs_input', question: { question_id: 'q1', text: 'Which folder?' } }), room.feed, room.add);
    assert.match(strip(card.host.innerHTML), /^Answered Velm's question /, 'the next read of the same question');
    letterRow(row({ state: 'needs_input', question: { question_id: 'q2', text: 'And the subfolders?' } }), room.feed, room.add);
    assert.match(strip(card.host.innerHTML), /^Velm asked about the letter /, 'a new question asks again');
    assert.deepEqual(room.nodes.map((node) => node.dataset.letterPart), ['letter', 'question:q1', 'answered:q1', 'question:q2']);
    room.nodes.pop(); room.calls.pop();  // the second question is this check's own; the story below goes on without it
    letterRow(row({ state: 'done', reply: 'given', spent_usd: 0.4, spent_final: true, answer: 'There are 3 files.' }, 'h3'), room.feed, room.add);
    letterRow(row({ state: 'done', reply: 'given', spent_usd: 0.4, spent_final: true, answer: 'There are 3 files.' }, 'h3'), room.feed, room.add);
    assert.equal(room.calls.length, 4, 'the answer mounts once');
    const answer = room.nodes[3];
    assert.equal(answer.dataset.letterPart, 'answer');
    assert.equal(room.calls[3].opts.historyId, undefined, 'the answer note borrows no row identity');
    assert.match(strip(answer.html), /^Velm Ouroboros \d\d:\d\d There are 3 files\.$/);
    assert.match(strip(card.host.innerHTML), /^Answered spent \$0\.40 of \$1\.00 from Velm's budget$/);
    const other = letterRoom();
    for (const reply of ['none', 'unknown']) {
        letterRow({ task_id: 't-b', ts: today(14, 9), peer_letter_card: { ...fact, letter_id: `L-${reply}`, state: 'done', reply, answer: 'B\'s own final words' } }, other.feed, other.add);
    }
    assert.deepEqual(other.nodes.map((node) => node.dataset.letterPart), ['letter', 'letter'], 'no reply for this Ouroboros: no answer note');
    // A page loading above replays the letter's earlier facts: an older fact never repaints the newer one.
    letterRow({ ...row({ state: 'accepted_running', phase: 'running' }, 'h0'), ts: today(14, 1) }, room.feed, room.add);
    assert.match(strip(card.host.innerHTML), /^Answered /);
    assert.equal(room.calls.length, 4);
    letterRow({ ...row({ state: 'unknown' }), ts: today(15, 0) }, room.feed, room.add);
    assert.match(strip(card.host.innerHTML), /^State unknown$/, 'a newer fact still repaints');
});

test('a letter of a work that ended no longer promises an answer; a settled one keeps its words', () => {
    const base = { letter_id: 'L1', to: 'Tarel', text: 'Найди причину.', ceiling_usd: 5 };
    const words = (fact) => { const v = letterCardView({ ...base, ...fact }, { ended: true }); return [v.word, v.rest, v.next].join(' | '); };
    assert.equal(words({ state: 'accepted_running', phase: 'queued' }), 'This work ended | last read: Queued at Tarel | No answer will come here.');
    assert.equal(words({ state: 'needs_input' }), 'This work ended | last read: Tarel asked about the letter | No answer will come here.');
    assert.match(words({ state: 'done', reply: 'given', answer: 'ok', spent_usd: 1, spent_final: true }), /^Answered \|/);
    assert.match(words({ state: 'unconfirmed' }), /^Not confirmed \|/, 'an unknown acceptance stays what it is');
    assert.match(words({ state: 'refused', refusal_code: 'link_paused' }), /^Refused \|/);
    // The room's own end of that work repaints its open letters (chat.js settleLiveCard → endLetters).
    const room = letterRoom();
    letterRow({ task_id: 't-a', ts: today(14, 3), peer_letter_card: { ...base, state: 'accepted_running', phase: 'running' } }, room.feed, room.add);
    const card = room.nodes[0];
    assert.equal(card.dataset.letterTask, 't-a');
    assert.match(strip(card.host.innerHTML), /^Tarel is working on it /);
    const saved = { document: globalThis.document, CSS: globalThis.CSS };
    globalThis.CSS = { escape: (value) => String(value) };
    globalThis.document = { querySelector: (selector) => (selector.includes('"t-a"') ? { dataset: { finished: '1' } } : null) };
    try {
        room.feed.querySelectorAll = (selector) => (selector.startsWith('.peer-letter-card') ? room.nodes : []);
        endLetters(room.feed, 't-b');
        assert.match(strip(card.host.innerHTML), /^Tarel is working on it /, 'another work\'s end changes nothing');
        endLetters(room.feed, 't-a');
        assert.equal(strip(card.host.innerHTML), 'This work ended last read: Tarel is working on it No answer will come here.');
        assert.match(card.attrs['aria-label'], /^Letter from Ouroboros to Tarel: This work ended, last read: Tarel is working on it$/);
    } finally {
        globalThis.document = saved.document;
        globalThis.CSS = saved.CSS;
    }
});

test('only a card-shaped fact under its own key is a letter: the core\'s receiver metadata `peer_letter` is never one', () => {
    const added = [];
    const feed = { querySelectorAll: () => [] };
    const add = (...args) => { added.push(args); return null; };
    // The receiving side's task metadata names its letter the same way; a history row carrying it stays its own row.
    assert.equal(letterRow({ task_id: 't-r', peer_letter: { place_id: 'p', link_id: 'l', letter_id: 'L-1' } }, feed, add), false);
    assert.equal(letterRow({ task_id: 't-r', peer_letter_card: { letter_id: 'L-1', place_id: 'p' } }, feed, add), false,
        'a fact that states no state is no card');
    assert.equal(added.length, 0);
});

test('the work a letter of it started here shows in its row with who sent it, and opens in the room its record names', () => {
    // The receiving side, before any ordinary message: its arrival index is empty, its acceptance record is not. The
    // record keeps no acceptance time, so no time is said.
    const work = { task_id: 'letter-1', title: 'Installer test', status: 'running', phase: 'working', from: 'Mac', room_chat_id: 3381 };
    const mac = place({ label: 'Mac', peer_work: work });
    assert.deepEqual(mac.peerWork, { taskId: 'letter-1', title: 'Installer test', status: 'running', phase: 'working', from: 'Mac', roomChatId: 3381 });
    assert.equal(rowLine(mac), 'work from Mac', 'the row says who sent work, never «no captured event»');
    assert.equal(rowLine(place({ label: 'Mac', peer_work: work, last_activity_at: new Date(Date.now() - 5 * 60_000).toISOString() })), '5 min ago',
        'a captured event keeps its own line');
    const open = strip(ownRowHtml(mac, { ...UI, open: 'p-tarel' }));
    assert.match(open, /Work from Mac Installer test · Working Open work/);
    assert.equal(strip(workRowHtml(mac)), 'Work from Mac Installer test · Working Open work', 'no date the record cannot vouch for');
    assert.match(ownRowHtml(mac, { ...UI, open: 'p-tarel' }), /data-own-act="open-work" data-place="p-tarel">Open work</);
    assert.match(strip(workRowHtml(place({ label: 'Mac', peer_work: { ...work, status: 'completed', phase: 'warn' } }))), /Installer test · Done with warnings/);
    assert.match(strip(workRowHtml(place({ peer_work: { ...work, phase: 'paused' } }))), /Installer test · Paused/);
    assert.match(strip(workRowHtml(place({ peer_work: { ...work, phase: null } }))), /Installer test · State unknown/);
    assert.equal(workRowHtml(place()), '', 'no accepted work: no row, the empty link stays as it was');
    assert.match(strip(workRowHtml(place({ peer_work: { unknown: true } }))), /Its accepted work could not be read now\./);
    assert.doesNotMatch(strip(workRowHtml(place({ peer_work: { ...work, room_chat_id: null } }))), /Open work/, 'no room named: no door');
    const asked = [];
    const listeners = {};
    const body = { addEventListener: (type, fn) => { listeners[type] = fn; }, removeEventListener: () => {}, contains: () => true, querySelector: () => null };
    globalThis.CSS = globalThis.CSS || { escape: (value) => String(value) };
    const own = createOwnOuroboroses(body, { places: () => [mac], paint: () => {}, refresh: async () => {}, openIntro: () => {}, openCodeCheck: () => {},
        openConversation: () => asked.push('conversation'), openWork: (subject) => asked.push(['work', subject.peerWork.roomChatId]) });
    const button = { dataset: { ownAct: 'open-work', place: 'p-tarel' }, disabled: false, closest: (selector) => (selector === '[data-own-act]' ? button : null) };
    listeners.click({ target: { closest: (selector) => (selector === '[data-own-act]' ? button : null) } });
    assert.deepEqual(asked, [['work', 3381]]);
    own.dispose();
});

test('where this side wrote first, before anything arrived, is the row\'s news and the room its conversation opens at', () => {
    // Three things the row tells apart: nothing yet, a work its letter started here, and this side's own first message.
    const wrote = new Date(Date.now() - 3 * 60_000).toISOString();
    const nothing = place({ label: 'Mac' });
    assert.equal(nothing.peerRoom, null);
    assert.equal(rowLine(nothing), 'no captured event');
    const mac = place({ label: 'Mac', peer_room: { room_chat_id: 1099511628800, wrote_at: wrote } });
    assert.deepEqual(mac.peerRoom, { roomChatId: 1099511628800, wroteAt: wrote }, 'the room as the host names it');
    assert.equal(rowLine(mac), 'Ouroboros wrote 3 min ago');
    assert.deepEqual(wroteRoom(mac), { roomChatId: 1099511628800, wroteAt: wrote });
    const work = { task_id: 'letter-1', title: 'Installer test', status: 'running', from: 'Mac', room_chat_id: 3381 };
    assert.equal(rowLine(place({ label: 'Mac', peer_work: work })), 'work from Mac');
    assert.equal(rowLine(place({ label: 'Mac', peer_work: work, peer_room: { room_chat_id: 1099511628800, wrote_at: wrote } })),
        'Ouroboros wrote 3 min ago', 'a dated message is the news over a work its record keeps no date for');
    const arrived = place({ label: 'Mac', rooms: [{ room_chat_id: 3381 }], last_activity_at: wrote, peer_room: { room_chat_id: 1099511628800, wrote_at: wrote } });
    assert.equal(wroteRoom(arrived), null, 'once something arrived, its listed room is the conversation');
    assert.equal(rowLine(arrived), '3 min ago');
    assert.equal(place({ peer_room: { unknown: true } }).peerRoom.unknown, true);
    assert.equal(wroteRoom(place({ peer_room: { unknown: true } })), null, 'an unread history names no room');
    assert.equal(place({ peer_room: { room_chat_id: 'x', wrote_at: wrote } }).peerRoom, null, 'a room id must be the host\'s number');
});

/** A page for the focus checks: a body whose repaint replaces the held control, a document that knows who has focus. */
function focusPage({ shown = true } = {}) {
    const listeners = {}, docListeners = {};
    const page = { nodeName: 'BODY' };
    const node = (attrs) => ({ ...attrs, focused: 0, focus() { this.focused += 1; globalThis.document.activeElement = this; },
        closest(selector) { return selector === '[data-own-switch]' && this.dataset?.ownSwitch ? this : selector.includes('button') ? this : null; } });
    const fresh = node({ dataset: { ownSwitch: 'p-tarel' } });
    const other = node({ dataset: {}, nodeName: 'BUTTON' });
    const body = { addEventListener: (type, fn) => { listeners[type] = fn; }, removeEventListener: () => {}, contains: () => true,
        isConnected: true, getClientRects: () => (shown ? [{}] : []),
        querySelector: (selector) => (selector === '[data-own-switch="p-tarel"]' ? fresh : null) };
    const control = { dataset: { ownSwitch: 'p-tarel' }, checked: false, setAttribute: () => {},
        closest: (selector) => (selector === '[data-own-switch]' || selector.includes('input') ? control : null) };
    const doc = { activeElement: control, body: page,
        addEventListener: (type, fn) => { (docListeners[type] ||= []).push(fn); },
        removeEventListener: (type, fn) => { docListeners[type] = (docListeners[type] || []).filter((f) => f !== fn); } };
    const fire = (type, target) => (docListeners[type] || []).forEach((fn) => fn({ target }));
    return { listeners, docListeners, page, fresh, other, body, control, doc, fire };
}

async function writeOnce(view, during = () => {}) {
    const saved = globalThis.document;
    globalThis.document = view.doc;
    globalThis.CSS = globalThis.CSS || { escape: (value) => String(value) };
    let own = null, release = null;
    const written = [];
    const held = new Promise((resolve) => { release = resolve; });
    try {
        // Each paint replaces the held control, so focus falls to the page; the page then calls painted().
        own = createOwnOuroboroses(view.body, { places: () => [place(NO_WORK)],
            paint: () => { if (globalThis.document.activeElement === view.control) globalThis.document.activeElement = view.page; own?.painted(); },
            refresh: async () => { during(); }, openIntro: () => {}, openCodeCheck: () => {}, openConversation: () => {},
            writes: { updatePlace: async (id, change) => { written.push([id, change]); await held; }, revokeLink: async () => {} } });
        view.listeners.change({ target: view.control });
        release();
        for (let i = 0; i < 5; i += 1) await new Promise((resolve) => setImmediate(resolve));
        return { written, own };
    } finally {
        globalThis.document = saved;
        own?.dispose();
    }
}

test('rule save checks the version seen when editing, preserves the draft on conflict, and retries against the shown rule', async () => {
    const page = focusPage();
    const saved = globalThis.document;
    globalThis.document = page.doc;
    globalThis.CSS ??= { escape: String };
    const changes = [];
    let observed = place();
    const own = createOwnOuroboroses(page.body, { places: () => [observed], paint: () => {}, refresh: async () => {},
        writes: { async updatePlace(id, change) {
            changes.push([id, change]);
            if (changes.length === 1) throw Object.assign(new Error('Changed elsewhere'), { status: 409,
                body: { place: { revision: 5, notes: [{ note_id: 'n2', text: 'Other window rule', author: 'owner' }] } } });
            return place({ revision: 6, notes: [{ note_id: 'n3', text: 'My draft', author: 'owner' }] });
        } } });
    const click = (action) => page.listeners.click({ target: { closest: (selector) => selector === '[data-own-act]'
        ? { dataset: { ownAct: action, place: 'p-tarel' } } : null } });
    const wait = () => new Promise((resolve) => setImmediate(resolve));
    try {
        const fold = { open: true, dataset: { ownRow: 'p-tarel' } };
        page.listeners.toggle({ target: { closest: () => fold } });
        click('edit-rule');
        page.listeners.input({ target: { closest: () => ({ dataset: { ownDraft: 'p-tarel' }, value: 'My draft' }) } });
        observed = place({ revision: 5 });
        click('save-rule'); await wait();
        assert.deepEqual(changes[0], ['p-tarel', { expected_revision: 4, remove_note: 'n1', add_note: 'My draft' }]);
        assert.match(strip(own.html([observed])), /My draft Save Cancel Current rule Other window rule/);
        click('save-rule'); await wait();
        assert.deepEqual(changes[1], ['p-tarel', { expected_revision: 5, remove_note: 'n2', add_note: 'My draft' }]);
    } finally { own.dispose(); globalThis.document = saved; }
});

test('a switch written from the keyboard has focus again once the place is read again: the same switch, drawn anew', async () => {
    const view = focusPage();
    const { written } = await writeOnce(view);
    assert.deepEqual(written, [['p-tarel', { peer: { mine: true, may_start: false }, expected_revision: 4 }]]);
    assert.equal(view.fresh.focused, 1, 'the switch has focus again after its write');
    assert.deepEqual(view.docListeners.focusin, [], 'nothing keeps listening once the write is done');
});

test('a write never takes focus back from where the owner put it meanwhile: another control, a press, or another page', async () => {
    const tabbed = focusPage();
    await writeOnce(tabbed, () => { tabbed.fire('focusin', tabbed.other); tabbed.doc.activeElement = tabbed.page; });
    assert.equal(tabbed.fresh.focused, 0, 'Tab to another control: the switch does not take focus back, even if that one was redrawn');
    const pressed = focusPage();
    await writeOnce(pressed, () => pressed.fire('pointerdown', pressed.other));
    assert.equal(pressed.fresh.focused, 0, 'a press on another control (WebKit gives a pressed button no focus) is the owner\'s choice too');
    const same = focusPage();
    await writeOnce(same, () => same.fire('pointerdown', same.control));
    assert.equal(same.fresh.focused, 1, 'a press on the held switch itself is no move elsewhere');
    const left = focusPage({ shown: false });
    await writeOnce(left);
    assert.equal(left.fresh.focused, 0, 'the page was left: nothing is focused there, now or at a later paint');
});

test('a question whose words the receiver holds back says so, without words, until a later fact carries them', () => {
    const fact = { letter_id: 'L1', to: 'Velm', to_seed: 'install-velm', text: 'Check the weather notes.', ceiling_usd: 1, spent_usd: null };
    const held = letterCardView({ ...fact, state: 'needs_input', question: { question_id: 'q1', text: null, held: 'peer_silent' } });
    assert.deepEqual(held.mark, { shape: 'still', hue: 'ink' }, 'nothing moves here while its words are held');
    assert.equal(held.word, 'Velm asked about the letter');
    assert.equal(held.next, 'Velm has not released the question\'s words yet, so Ouroboros cannot answer it.');
    const done = letterCardView({ ...fact, state: 'done', reply: 'held', held: 'peer_silent', spent_usd: 0.2, spent_final: true });
    assert.equal(done.word, 'Finished');
    assert.equal(done.next, 'Velm has not released its reply yet.', 'a held reply is neither none nor an empty answer');
    assert.equal(letterCardView({ ...fact, state: 'done', reply: 'held', held: 'rules_changed' }).next,
        'Velm\'s rules for this link changed since, so its reply does not come here.', 'one that does not come by itself says so');
    assert.equal(letterCardView({ ...fact, state: 'needs_input', question: { question_id: 'q1', text: null, held: 'rules_changed' } }).next,
        'Velm\'s rules for this link changed since it asked, so its words are not shown here.');
    const room = letterRoom();
    let tick = 0;
    const row = (extra) => ({ task_id: 't-a', ts: new Date(Date.parse(today(14, 3)) + 1000 * tick++).toISOString(), peer_letter_card: { ...fact, ...extra } });
    letterRow(row({ state: 'accepted_running', phase: 'running', at: today(14, 2) }), room.feed, room.add);
    letterRow(row({ state: 'needs_input', question: { question_id: 'q1', text: null, held: 'peer_silent' } }), room.feed, room.add);
    assert.deepEqual(room.nodes.map((node) => node.dataset.letterPart), ['letter', 'question:q1']);
    assert.equal(room.nodes[1].dataset.held, '1');
    assert.equal(heldWords('Velm', 'rules_changed'), 'Not shown under Velm\'s current rules for this link.');
    assert.match(strip(room.nodes[1].html), new RegExp(`Question about the letter ${heldWords('Velm')}$`));
    letterRow(row({ state: 'needs_input', question: { question_id: 'q1', text: 'Which shelf?' } }), room.feed, room.add);
    assert.equal(room.nodes.length, 2, 'its words come into the same note');
    assert.equal(room.nodes[1].dataset.held, undefined);
    letterRow(row({ state: 'done', reply: 'held', held: 'peer_silent', spent_usd: 0.2, spent_final: true }), room.feed, room.add);
    assert.deepEqual(room.nodes.map((node) => node.dataset.letterPart), ['letter', 'question:q1'], 'a held reply mounts no answer');
    assert.match(strip(room.nodes[0].host.innerHTML), /^Finished spent \$0\.20 of \$1\.00 from Velm's budget Velm has not released its reply yet\.$/);
});
