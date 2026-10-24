// The place sheet and the Connect flow against the contract's PlaceSummary: the state control with each
// state's note (Silent's host card verbatim), what wakes it and who added each item, the owner's accounts,
// notes with their authors, the folder and the model, and the one owner write each press asks for.
import assert from 'node:assert/strict';
import fs from 'node:fs';
import test from 'node:test';
import { setup } from './dashboard_dom_fixture.js';

import { normalizePlacesResponse } from '../modules/consciousness_api.js';
import { SILENT_CARD, placeStatusText } from '../modules/consciousness_words.js';
import { connectListHtml, openPlaceSheet, placeChangeFor, placeSheetBodyHtml, placeSheetMetaText } from '../modules/place_sheet.js';

const FIXTURE = JSON.parse(fs.readFileSync(new URL('./fixtures/consciousness_contract.json', import.meta.url), 'utf8'));
const { places, connectors } = normalizePlacesResponse(FIXTURE.places);
const [slack, community, mail] = places;
const strip = (html) => html.replace(/<[^>]+>/g, ' ').replace(/\s+/g, ' ').trim();

for (const action of ['save-names', 'clear-names']) {
    for (const newer of [undefined, 'Newer name', '']) {
        test(`${action} response preserves ${newer === undefined ? 'unchanged' : JSON.stringify(newer)} draft in the open sheet`, async (t) => {
            let sheet;
            t.after(() => sheet?.close());
            const { document } = setup(t);
            const place = { ...slack, attentionNames: ['Saved name'] };
            let resolveUpdate;
            const requests = [];
            sheet = openPlaceSheet(place, { update: (id, change) => {
                requests.push({ id, change });
                return new Promise((resolve) => { resolveUpdate = resolve; });
            } });
            const body = document.body.querySelector('[data-place-body]');
            const input = async (value) => {
                const field = body.querySelector('[data-place-draft="attentionNames"]');
                field.value = value;
                await body.fire('input', { target: field });
            };
            await input(' Submitted name ');
            const pending = body.fire('click', { target: body.querySelector(`[data-place-act="${action}"]`) });
            assert.deepEqual(requests, [{ id: place.id, change: { attention_names: action === 'save-names' ? ['Submitted name'] : [] } }]);
            assert.match(body.innerHTML, /Saving…/);
            const field = body.querySelector('[data-place-draft="attentionNames"]');
            assert.equal(field.disabled, false, 'typing stays available while the update is pending');
            if (newer !== undefined) await input(newer);
            resolveUpdate({ ...place, attentionNames: requests[0].change.attention_names });
            await pending;
            assert.deepEqual(sheet.place.attentionNames, requests[0].change.attention_names, 'the saved record is the confirmed response, not the newer draft');
            assert.equal(body.querySelector('[data-place-draft="attentionNames"]').textContent,
                newer ?? (action === 'save-names' ? 'Submitted name' : ''));
            const status = body.querySelector('.place-sheet-status');
            assert.equal(status.textContent, newer === undefined
                ? (action === 'save-names' ? 'Names saved.' : 'Name rule off.')
                : `${action === 'save-names' ? 'Earlier names saved.' : 'Name rule off.'} Your newer edits are not saved.`);
            assert.equal(status.getAttribute('data-tone'), newer === undefined ? 'ok' : 'muted');
            if (newer !== undefined) {
                const resave = body.fire('click', { target: body.querySelector('[data-place-act="save-names"]') });
                assert.deepEqual(requests[1].change, { attention_names: newer ? [newer] : [] }, 'the next Save submits the retained draft');
                resolveUpdate({ ...place, attentionNames: requests[1].change.attention_names });
                await resave;
                assert.equal(body.querySelector('.place-sheet-status').textContent, 'Names saved.');
            }
        });
    }
}

test('name rule defaults empty, edits and clears separately from provider wakes', () => {
    assert.deepEqual(slack.attentionNames, []);
    const place = { ...slack, wakes: [], attentionNames: ['Вик', 'Ouroboros'], attentionNamesAuthor: 'owner' };
    const html = placeSheetBodyHtml(place);
    assert.match(html, /data-place-draft="attentionNames"[^>]*>Вик\nOuroboros<\/textarea>/);
    assert.match(html, /not a platform @mention/);
    assert.match(html, /Bots still need Messages from bots/);
    assert.match(strip(html), /added by you/);
    assert.match(placeStatusText(place), /wakes for names and aliases/);
    assert.deepEqual(placeChangeFor('save-names', place, { drafts: { attentionNames: ' Вик \n\n Broboros, Vic ' } }),
        { attention_names: ['Вик', 'Broboros, Vic'] }, 'commas remain literal; one alias per line');
    assert.deepEqual(placeChangeFor('clear-names', place), { attention_names: [] });
    assert.deepEqual(placeChangeFor('save-names', place, { drafts: { attentionNames: '' } }), { attention_names: [] });
    assert.doesNotMatch(placeSheetBodyHtml({ ...place, attentionNames: ['<unsafe>'] }), /<unsafe>/);
});

test('the head names the connector and how much of it the place covers', () => {
    assert.equal(placeSheetMetaText(slack), 'Slack · whole account · T024BE7LD');
    assert.equal(placeSheetMetaText(community), 'Telegram · one conversation · -100123');
});

test('Active · Silent · Paused in one control; the note says what the chosen state does', () => {
    const html = placeSheetBodyHtml(slack);
    assert.deepEqual([...html.matchAll(/data-effort-value="(\w+)">(\w+)</g)].map((m) => [m[1], m[2]]),
        [['active', 'Active'], ['silent', 'Silent'], ['paused', 'Paused']]);
    assert.match(html, /data-place-state/);
    assert.match(html, /What it does here/);
    assert.match(html, /Lives here: listens, wakes when its list says so, speaks and acts\./);
    assert.doesNotMatch(html, /place-silent-card/);
    assert.match(placeSheetBodyHtml(mail), /Away from this place: no wakes, no actions\. Still connected\./);
});

test('Silent shows the host\'s card: the note and, verbatim, what it blocks and what it cannot', () => {
    const html = placeSheetBodyHtml(community);
    const card = html.match(/<p class="place-silent-card" role="note">([^<]*)<\/p>/)?.[1] || '';
    assert.equal(card.replace(/&#39;/g, "'"),
        `Hears everything here and changes nothing: no messages, reactions, edits or moderation. ${SILENT_CARD}`);
    assert.doesNotMatch(html, /class="place-state-note"/, 'the note is said once, in the card');
});

test('what wakes it, with who added each item; more from the connector\'s suggestions only', () => {
    const html = placeSheetBodyHtml(slack);
    assert.match(strip(html), /Direct messages added by connector Remove/);
    assert.match(strip(html), /Replies added by you Remove/);
    const offers = [...html.matchAll(/data-place-act="add-wake" data-fact="(\w+)"/g)].map((m) => m[1]);
    assert.deepEqual(offers, ['group_dm', 'my_thread'], 'only suggestions not yet listed');
    assert.match(strip(placeSheetBodyHtml(mail)), /Every message carried over/);
    assert.match(placeSheetBodyHtml(slack), /Everything else here is heard and waits for its own clock\./);
});

test('Silent still wakes for its list and holds what it would say or change; Paused is not woken', () => {
    const silent = placeSheetBodyHtml(community);
    assert.match(silent, /Silent: these still wake it to read and think; what it would say or change here is held\./);
    assert.doesNotMatch(silent, /nothing here wakes it/);
    assert.match(placeSheetBodyHtml(mail), /While this place is not Active, nothing here wakes it\./);
    assert.doesNotMatch(placeSheetBodyHtml({ ...mail, state: 'gone' }), /wakes it\.|its own clock\./, 'an unreadable state claims nothing');
});

test('your accounts here: confirmed ones with Remove, Add mine asks for the account, people seen offer «This is me»', () => {
    let html = placeSheetBodyHtml(slack);
    assert.match(strip(html), /@anton\.r you · confirmed on 12 Oct · U01ANTON Remove/);
    assert.match(html, /data-place-act="add-mine">Add mine</);
    html = placeSheetBodyHtml(slack, { adding: true, drafts: { actorId: 'U02' } });
    assert.match(html, /data-place-draft="actorId" value="U02"/);
    assert.match(html, /data-place-act="confirm-typed">This is me</);
    const seen = { ...community, accountCandidates: [{ actorId: 'U77', label: 'Anton R' }] };
    assert.match(placeSheetBodyHtml(seen), /data-actor="U77" data-actor-label="Anton R">This is me</);
    assert.match(html, /Anyone else here, whatever name they use, is talked to like anyone/);
});

test('Add mine names the identifier the place\'s service knows you by', () => {
    const help = (place) => placeSheetBodyHtml(place, { adding: true }).match(/<div class="ui-field-help" id="place-account-help">([^<]*)<\/div>/)?.[1];
    assert.equal(help(slack), 'Your member ID (U…).');
    assert.equal(help(community), 'Your numeric user ID.');
    assert.equal(help(mail), 'Your email address.', 'a mail place (imap)');
    assert.equal(help({ ...mail, provider: 'email' }), 'Your email address.');
    assert.equal(help({ ...slack, provider: 'agentrq' }), 'The account ID this service uses.');
    const form = placeSheetBodyHtml(slack, { adding: true });
    assert.match(form, /<label for="place-account-id">Your account here<\/label>\s*<input class="ui-control" id="place-account-id" data-place-draft="actorId"[^>]*aria-describedby="place-account-help">\s*<div class="ui-field-help" id="place-account-help">/,
        'the help sits under its own field');
    assert.doesNotMatch(placeSheetBodyHtml(slack), /place-account-help/, 'only while adding');
});

test('notes keep their authors: the owner\'s in the owner\'s voice, Ouroboros\'s own as its own', () => {
    const html = placeSheetBodyHtml(slack);
    assert.match(html, /<li class="place-note is-owner">\s*<p>Отвечай коротко и по делу\.<\/p>/);
    assert.match(html, /<li class="place-note">\s*<p>Лена ведёт релизы\.<\/p>/);
    assert.match(strip(html), /Лена ведёт релизы\. added by Ouroboros Remove/);
    assert.match(placeSheetBodyHtml(mail), /<li class="place-note is-owner">/, 'a carried-over note was the owner\'s');
    assert.match(html, /the Safety Supervisor treats yours as your instructions/);
});

test('working folder is an address; the model lives under More; Disconnect asks in place', () => {
    const html = placeSheetBodyHtml(slack);
    assert.match(html, /data-place-draft="folder" value="~\/work\/acme"/);
    assert.match(html, /An address, not a fence: work for this place starts here\./);
    assert.match(html, /<details class="consciousness-more place-more">/);
    assert.match(placeSheetBodyHtml(mail), /data-place-draft="model" value="anthropic\/claude-sonnet-5\.5"/);
    assert.match(html, /data-place-act="open-connector">Open connector page</);
    assert.match(html, /data-place-act="disconnect">Disconnect</);
    const asking = placeSheetBodyHtml(slack, { confirmDisconnect: true });
    assert.match(asking, /Disconnect Slack · Acme\? It stops hearing and acting here; the room's history stays\./);
    assert.match(asking, /data-place-act="disconnect-yes">Disconnect</);
});

test('each press asks for exactly one contract change, never a guess', () => {
    assert.deepEqual(placeChangeFor('state', slack, { value: 'silent' }), { state: 'silent' });
    assert.equal(placeChangeFor('state', slack, { value: 'active' }), null, 'the state it already has');
    assert.deepEqual(placeChangeFor('add-wake', slack, { fact: 'group_dm' }),
        { wakes: ['direct_message', 'mention', 'reply_to_me', 'group_dm'] });
    assert.deepEqual(placeChangeFor('remove-wake', slack, { fact: 'mention' }), { wakes: ['direct_message', 'reply_to_me'] });
    assert.equal(placeChangeFor('add-note', slack, { drafts: { note: '   ' } }), null);
    assert.deepEqual(placeChangeFor('add-note', slack, { drafts: { note: ' Be brief. ' } }), { add_note: 'Be brief.' });
    assert.deepEqual(placeChangeFor('remove-note', slack, { note: 'n2' }), { remove_note: 'n2' });
    assert.deepEqual(placeChangeFor('confirm', community, { actor: 'U77', actorLabel: 'Anton R' }),
        { confirm_owner_account: { platform_actor_id: 'U77', label: 'Anton R' } });
    assert.deepEqual(placeChangeFor('confirm-typed', slack, { drafts: { actorId: ' U02 ', actorLabel: '' } }),
        { confirm_owner_account: { platform_actor_id: 'U02', label: '' } });
    assert.equal(placeChangeFor('confirm-typed', slack, { drafts: {} }), null);
    assert.deepEqual(placeChangeFor('revoke', slack, { actor: 'U01ANTON' }), { revoke_owner_account: 'U01ANTON' });
    assert.deepEqual(placeChangeFor('save-folder', slack, { drafts: { folder: ' ~/work/acme-2 ' } }), { working_folder: '~/work/acme-2' });
    assert.deepEqual(placeChangeFor('save-folder', slack, {}), { working_folder: '~/work/acme' });
    assert.deepEqual(placeChangeFor('save-model', mail, { drafts: { model: '' } }), { model: '' }, 'empty uses the consciousness model');
    assert.equal(placeChangeFor('unknown', slack, {}), null);
});

test('Connect lists the installed connectors not yet connected; none sends the owner to Browse › Connectors', () => {
    assert.match(connectListHtml({}), /Reading the installed connectors…/);
    const html = connectListHtml({ connectors });
    assert.equal((html.match(/data-connect-act="connect"/g) || []).length, 1);
    assert.match(strip(html), /AgentRQ Connects AgentRQ · suggests waking for direct messages Connect/);
    assert.doesNotMatch(html, /data-connector="slack-bridge"/, 'a connected one is not offered again');
    const none = connectListHtml({ connectors: connectors.map((entry) => ({ ...entry, connected: true })) });
    assert.match(none, /Every installed connector is connected\./);
    assert.match(none, /data-connect-act="browse">Find connectors in Skills</);
    assert.match(connectListHtml({ connectors: [] }), /No installed connector can connect a place yet\./);
    assert.match(connectListHtml({ error: 'Couldn’t read the connectors: HTTP 404' }), /data-connect-act="retry"/);
    assert.match(connectListHtml({ connectors, status: { 'agentrq-bridge': { text: 'Refused', tone: 'error' } } }), /data-tone="error" role="status">Refused</);
});
