// A skill's service sign-ins on its card and sheet (docs/DESIGN.md "Sign-ins on a skill's card"): the words of each
// account state and failure, the floor sentence, the client preselection order, the secret never painted back, and
// the card's one tray only while a sign-in needs the owner. DOM-free: the renderers are pure over the status shape.
import assert from 'node:assert/strict';
import test from 'node:test';

import {
    accountState, clientFormHtml, clientsEditorHtml, confirmSentence, failureWords, floorShown, preselectedClient, signInEditorHtml,
    signInMetaHtml, signInRowsHtml, signInTray,
} from '../modules/skill_sign_in.js';
import { renderInstalledSkillCard } from '../modules/skill_card_renderer.js';
import { shortDate } from '../modules/skill_presentation.js';

const EVENTS = 'https://www.googleapis.com/auth/calendar.events';
const GMAIL = 'https://www.googleapis.com/auth/gmail.settings.basic';
const RETURN = 'http://127.0.0.1:8765/onboarding/static/provider-return.html';
const text = (html) => String(html).replace(/<[^>]+>/g, '').replace(/&#39;/g, "'").replace(/&quot;/g, '"').replace(/&lt;/g, '<')
    .replace(/&gt;/g, '>').replace(/&#96;/g, '`').replace(/&amp;/g, '&').replace(/\s+/g, ' ').trim();
const client = (overrides = {}) => ({ id: 'c-1', label: 'Personal', issuer: 'https://accounts.google.com', client_id_tail: 'a1b2', return: 'loopback',
    source: 'owner', secret_set: false, used_by: [], ...overrides });
const account = (overrides = {}) => ({ key: 'a-1', label: 'personal@example.test', subject: '1', verified: true, state: 'signed_in',
    since: '2026-11-14T12:00:00+00:00', scope: ['openid', 'email', EVENTS], missing_required: [], not_allowed: [],
    client: { id: 'c-1', label: 'Personal' }, via: 'card', siblings: [], ...overrides });
const service = (overrides = {}) => ({
    skill: 'calendar', service: 'google', label: 'Google', issuer: 'https://accounts.google.com', scopes: ['openid', 'email', EVENTS],
    scope_labels: { [EVENTS]: 'events' }, optional_scopes: [{ scope: GMAIL, label: 'Manage Gmail settings', default: true }],
    accounts_mode: 'many', optional: true, accounts: [], pending: null, clients: [client()],
    notes: { client: 'Create a Desktop app client.', refused: 'admin_policy_enforced: only some permissions are approved.',
        refresh_rejected: 'If your Google Cloud app is in Testing, Google ends sign-ins after 7 days.' },
    ...overrides,
});
const status = (...services) => ({ ok: true, services, return_address: RETURN });
const REFUSED = { state: 'sign_in_required', reason: 'refresh_rejected', since: '2026-11-21T12:00:00+00:00' };

test('each account state says its own words, and only a state that needs the owner asks for a sign-in', () => {
    const signed = accountState(service(), account());
    assert.deepEqual([text(signed.status), signed.tone, signed.needs], ['Signed in', 'ok', false]);
    const refused = accountState(service(), account(REFUSED));
    assert.deepEqual([text(refused.status), refused.tone, refused.needs], ['Sign in again', 'danger', true]);
    assert.equal(text(refused.sentence), `Google no longer accepts the sign-in for personal@example.test (refused ${shortDate(REFUSED.since)}). Sign in again.`);
    assert.equal(text(refused.note), 'If your Google Cloud app is in Testing, Google ends sign-ins after 7 days.');
    const missing = accountState(service(), account({ missing_required: [EVENTS] }));
    assert.deepEqual([text(missing.status), missing.tone, missing.needs], ['Needs more access', 'warn', true]);
    assert.equal(text(missing.sentence), 'Signed in, but Google didn’t allow: events. calendar can’t use this account until you sign in again and allow it.');
    const stepUp = accountState(service(), account({ reason: 'insufficient_scope',
        required_scope: `openid email ${EVENTS} ${GMAIL}`, missing_step_up: [GMAIL] }));
    assert.equal(text(stepUp.sentence), 'calendar needs more access for what was asked: Manage Gmail settings. Sign in again to allow it.');
    const moved = accountState(service(), account({ via: 'handover', verified: false }));
    assert.deepEqual([text(moved.status), moved.tone, moved.needs], ['Moved from calendar’s own sign-in · not confirmed', 'muted', false]);
    // The skill's words stay as written; the sentence around them is the host's.
    assert.match(refused.sentence, /<span data-i18n-authored>Google<\/span>/);
    assert.match(refused.note, /^<span data-i18n-authored>/);
});

test('a row names the account, its client and what it allows; Sign in again and the result line sit in the row', () => {
    const rows = signInRowsHtml(service({ accounts: [account({ not_allowed: [GMAIL] }), account({ key: 'a-2', label: 'work@example.test', ...REFUSED })] }),
        { results: new Map([['a-1', { text: 'Signed in as personal@example.test via Personal.', tone: 'ok' }]]), confirm: null });
    const [first, second] = rows.split('data-sign-in-row=').slice(1);
    assert.match(text(first), /personal@example\.test via Personal/);
    assert.match(text(first), new RegExp(`Allowed: openid, email, events · signed in ${shortDate('2026-11-14T12:00:00+00:00')}`));
    assert.match(text(first), /Not allowed: Manage Gmail settings/);
    assert.match(first, /data-sign-in-result="a-1"[^>]*data-tone="ok">Signed in as personal@example\.test via Personal\.<\/span>/);
    assert.doesNotMatch(first, /data-sign-in-act="begin"/, 'a row that needs nothing offers no Sign in again of its own');
    assert.match(second, /data-sign-in-act="begin" data-sign-in-account="a-2">Sign in again</);
    assert.match(second, /aria-label="More for work@example\.test" data-sign-in-act="menu"/);
    assert.equal(text(signInRowsHtml(service())), 'No account is signed in.');
});

test('failures and confirmations use the host words around the declaration\'s label and host', () => {
    const google = service();
    const words = (result, options) => failureWords(google, result, options);
    assert.equal(words({ reason: 'denied', error: 'access_denied' }), 'Google refused the sign-in (access_denied). Start again.');
    assert.equal(words({ reason: 'exchange_rejected', error: 'invalid_client' }, { client: 'Work (IT)' }),
        'Google refused the client «Work (IT)» (invalid_client). Check its client ID and secret in Clients, or choose another client.');
    assert.equal(words({ reason: 'exchange_rejected', error: 'invalid_grant' }), 'This sign-in link was already used or has expired; start again.');
    assert.equal(words({ code: 'SIGN_IN_UNAVAILABLE', reason: 'provider_unreachable', error: 'ConnectError: [Errno 8] nodename nor servname' }),
        'Couldn’t reach accounts.google.com (ConnectError), so no sign-in page was opened. Open the sign-in page again once this computer is online.');
    assert.equal(words({ reason: 'sign_in_superseded' }), 'A newer sign-in for calendar started meanwhile; finish that one or start again.');
    assert.equal(words({ reason: 'no_client', error: 'add a client' }, { returnAddress: RETURN }), `Add a client for Google first; it returns to ${RETURN}.`);
    assert.equal(words({ reason: 'http_400', error: 'the address carries no complete sign-in return' }, { complete: true }),
        'This address carries no sign-in return; paste the whole address your browser stopped on.');
    assert.equal(words({ reason: 'http_400', error: 'supply action' }), 'It didn’t go through: supply action', 'a bad request outside a paste keeps its cause');
    const lost = 'Ouroboros has no confirmed result for this sign-in, and it sends each one-time code only once. If this window still waits, choose Start again.';
    for (const result of [{ reason: 'provider_unreachable', error: 'ReadTimeout: no answer' }, { reason: 'exchange_not_reusable' }, { reason: 'unconfirmed' },
        { reason: 'exchange_rejected', error: 'HTTP 503' }]) assert.equal(words(result, { complete: true }), lost, result.reason);
    assert.equal(words({ reason: 'exchange_rejected', error: 'invalid_grant' }, { complete: true }), 'This sign-in link was already used or has expired; start again.');
    assert.equal(words({ reason: 'answer_saved' }, { complete: true }), 'Google’s answer to this sign-in is saved, but Ouroboros couldn’t read which account it names yet. Choose Finish sign-in again; its one-time code isn’t sent a second time.');
    assert.equal(words({ reason: 'unconfirmed' }), 'Ouroboros didn’t confirm the result. Check the sign-in before trying again.', 'a begin has no code to protect');
    assert.equal(words({ reason: 'registration_unconfirmed' }), 'Google may have registered a client that never reached Ouroboros, so no new client is available here. Open the sign-in page again to ask it to register another.');
    assert.equal(words({ reason: 'identity_unavailable' }), 'Google didn’t say which account signed in, so no account was confirmed. Start again.');
    const mail = account({ siblings: [{ skill: 'mail', service: 'google' }] });
    assert.equal(text(confirmSentence(google, mail, 'disconnect')),
        'calendar stops using personal@example.test and this installation forgets the sign-in. Google keeps the permission until you remove it in your Google account; to end it there now, cancel and choose Revoke instead.');
    assert.equal(text(confirmSentence(google, mail, 'revoke')), 'Revoke at Google? Google ends this sign-in, and may also end other sign-ins made with the same '
        + 'client for personal@example.test: mail (personal@example.test). Skills that keep their own sign-in may lose access too. You can sign in again later. '
        + 'The sign-in is forgotten here only once Google confirms.');
    const alone = text(confirmSentence(google, account(), 'revoke'));
    assert.doesNotMatch(alone, /may also end/, 'no sibling is named when the host sees none');
    assert.match(alone, /Skills that keep their own sign-in may lose access too\./);
});

test('the floor sentence shows only on a page that is not on a loopback host with a client that returns to loopback', () => {
    assert.equal(floorShown('phone.example', 'loopback'), true);
    for (const host of ['127.0.0.1', 'localhost', '[::1]', '::1']) assert.equal(floorShown(host, 'loopback'), false, host);
    assert.equal(floorShown('phone.example', 'exact:https://ouro.example/onboarding/static/provider-return.html'), false);
    assert.equal(floorShown('phone.example', ''), false, 'no chosen client, no claim');
    const LOCAL = 'exact:http://127.0.0.1:8765/onboarding/static/provider-return.html';
    assert.equal(floorShown('phone.example', LOCAL), true, 'an exact address on the computer where Ouroboros runs is no way back for another browser either');
    assert.equal(floorShown('127.0.0.1', LOCAL), false);
    const editor = { kind: 'sign_in', account: '', draft: { hint: '', client: 'c-1', optional: [] } };
    const FLOOR = 'This sign-in comes back to the computer where Ouroboros runs.';
    assert.ok(text(signInEditorHtml(service(), editor, 'phone.example')).includes(FLOOR));
    assert.ok(!text(signInEditorHtml(service(), editor, '127.0.0.1')).includes(FLOOR));
    const registered = service({ clients: [client({ return: LOCAL, source: 'registered' })] });
    assert.ok(text(signInEditorHtml(registered, editor, 'phone.example')).includes(`Google sends you back to ${LOCAL.slice(6)}; this client must list exactly this address. ${FLOOR}`));
    assert.ok(!text(signInEditorHtml(registered, editor, '127.0.0.1')).includes(FLOOR), 'on the computer where it runs there is nothing to explain');
    const web = service({ clients: [client({ return: 'exact:https://ouro.example/onboarding/static/provider-return.html' })] });
    const webText = text(signInEditorHtml(web, editor, 'phone.example'));
    assert.ok(!webText.includes(FLOOR) && webText.includes('Google sends you back to https://ouro.example/onboarding/static/provider-return.html; '
        + 'this client must list exactly this address.'));
    const waiting = (pageHost) => signInEditorHtml(service(), { kind: 'waiting', url: 'https://accounts.google.com/o/oauth2/auth?state=s', clientReturn: 'loopback' }, pageHost);
    assert.match(waiting('phone.example'), /<details class="ui-disclosure" open>/, 'where the browser cannot come back, the paste is open');
    assert.match(waiting('127.0.0.1'), /<details class="ui-disclosure">/);
    assert.match(waiting('127.0.0.1'), /admin_policy_enforced: only some permissions are approved\./, 'the skill\'s refused note waits beside the paste');
});

test('the client a sign-in starts with: the account\'s own, else the one a sibling skill used for that address, else the only one', () => {
    const personal = client({ used_by: [{ skill: 'mail', account: 'personal@example.test' }] });
    const work = client({ id: 'c-2', label: 'Work', client_id_tail: 'c3d4', used_by: [{ skill: 'calendar', account: 'other@example.test' }] });
    const both = service({ clients: [personal, work] });
    assert.deepEqual(preselectedClient(both, { account: account({ client: { id: 'c-2', label: 'Work' } }) }), { client: work, why: 'own' });
    assert.deepEqual(preselectedClient(both, { account: account({ client: { id: 'c-gone', label: 'Old' } }) }), { client: personal, why: 'sibling', by: 'mail' });
    assert.equal(preselectedClient(both, { address: 'Personal@example.test' }).client, personal, 'the address matches as an address');
    assert.deepEqual(preselectedClient(both, { address: 'other@example.test' }), { client: null, why: '' }, 'this skill\'s own use is no sibling');
    assert.deepEqual(preselectedClient(both, {}), { client: null, why: '' }, 'two clients and nothing to go by: the owner chooses');
    assert.deepEqual(preselectedClient(service(), {}), { client: service().clients[0], why: 'only' });
    const editor = signInEditorHtml(both, { kind: 'sign_in', account: '', draft: { hint: 'personal@example.test', client: 'c-1', optional: [GMAIL] } }, '127.0.0.1');
    assert.match(text(editor), /Personal · …a1b2 · returns to this computer — mail uses it for this account/);
    assert.match(editor, /value="c-1" checked>/);
    assert.doesNotMatch(editor, /value="c-2" checked/);
    assert.match(editor, new RegExp(`data-sign-in-optional value="${GMAIL.replace(/[./]/g, '\\$&')}" checked>`));
    assert.ok(text(editor).includes('Always: openid; email; events (https://www.googleapis.com/auth/calendar.events)'), 'the Always line names what every sign-in sends, by the skill label and the exact scope');
});

test('only discovery offers registration without a saved client; a partial grant shows the actual permissions', () => {
    const editor = { kind: 'sign_in', draft: { hint: '', client: '', optional: [] } };
    const endpoints = { authorization: 'https://auth.example.test/authorize', token: 'https://tokens.example.test/token',
        identity: 'https://profile.example.test/who', revocation: 'https://tokens.example.test/revoke' };
    const discovery = service({ clients: [] });
    const explicit = service({ clients: [], issuer: 'https://auth.example.test', endpoints });
    assert.match(signInEditorHtml(discovery, editor), /data-sign-in-act="open"[^>]*>Register a client and sign in</);
    assert.ok(text(signInEditorHtml(discovery, editor)).includes('This asks Google (https://accounts.google.com) to register a client, if it offers that option.'), 'the provider is named by its address');
    assert.doesNotMatch(signInEditorHtml(explicit, editor), /data-sign-in-act="open"/);
    for (const address of Object.values(endpoints)) assert.ok(text(clientFormHtml(explicit)).includes(address));
    const fixture = account({ label: 'work@example.test', state: 'sign_in_required', reason: 'insufficient_scope',
        scope: ['openid', 'email'], missing_required: [EVENTS] });
    const rows = text(signInRowsHtml(service({ accounts: [fixture] })));
    assert.ok(rows.includes('Allowed: openid, email'));
    assert.ok(rows.includes('didn’t allow: events'));
});

test('a saved client never shows its secret: the password field starts empty and a row carries only its ID tail', () => {
    const edit = clientFormHtml(service(), { id: 'c-1', label: 'Personal', return: 'loopback', tail: 'a1b2', secret_set: true, client_secret: 'S3CR3T-value' }, RETURN);
    assert.ok(!edit.includes('S3CR3T'));
    assert.match(edit, /type="password" data-client-field="client_secret" value="" autocomplete="new-password"/);
    assert.match(text(edit), /Blank keeps the saved secret\./);
    assert.match(edit, /<code data-sign-in-register>http:\/\/127\.0\.0\.1:8765\/onboarding\/static\/provider-return\.html<\/code>/);
    const added = clientFormHtml(service(), {}, RETURN);
    assert.doesNotMatch(text(added), /Blank keeps|Type its client ID again/, 'a new client keeps no secret to keep');
    const saved = service({ clients: [client({ secret_set: true, used_by: [{ skill: 'mail', account: 'personal@example.test' }] })] });
    const list = clientsEditorHtml(saved, { form: null, result: { text: 'Saved Personal.', tone: 'ok' } }, RETURN);
    assert.ok(!list.includes('S3CR3T') && !list.includes('client_secret'));
    assert.match(text(list), /Personal · …a1b2 · returns to this computer/);
    assert.match(text(list), /Used by: mail \(personal@example\.test\)/);
    assert.match(list, /data-sign-in-result="clients"[^>]*data-tone="ok">Saved Personal\.</);
    assert.match(text(list), /Create a Desktop app client\./);
});

test('the card has one sign-in tray only while a sign-in needs the owner, and one meta line under its who line', () => {
    const tray = (...accounts) => signInTray('calendar', status(service({ accounts })));
    for (const needing of [account(REFUSED), account({ missing_required: [EVENTS] }),
        account({ reason: 'insufficient_scope', required_scope: GMAIL, missing_step_up: [GMAIL] })]) {
        const html = tray(needing);
        assert.match(html, /data-sign-in-tray/);
        assert.match(html, /data-skill="calendar" data-skill-action="open_detail" data-sign-in-service="google" data-sign-in-account="a-1" data-sign-in-begin>Sign in again</);
    }
    assert.match(signInTray('calendar', status(service({ optional: false }))), /data-sign-in-account="" data-sign-in-begin>Sign in</);
    assert.equal(tray(account()), '');
    assert.equal(tray(account({ via: 'handover', verified: false })), '');
    assert.equal(signInTray('calendar', status(service())), '', 'an optional service with no account asks nothing');
    assert.equal(signInTray('mail', status(service({ accounts: [account(REFUSED)] }))), '', 'another skill\'s sign-in is not this card\'s');
    assert.equal(signInTray('calendar', null), '');

    const two = status(service({ accounts: [account(), account({ key: 'a-2', label: 'team@example.test' })] }));
    assert.equal(text(signInMetaHtml('calendar', two)), 'Signed in to Google: personal@example.test · team@example.test');
    assert.match(signInMetaHtml('calendar', two), /data-skill-action="open_detail" data-sign-in-service="google">Signed in to <span data-i18n-authored>Google<\/span><\/button>/);
    assert.equal(text(signInMetaHtml('calendar', status(service({ accounts: [account(REFUSED)] })))), 'Not signed in to Google');
    assert.equal(signInMetaHtml('weather', two), '', 'a skill that declares no service has no line');

    const row = { name: 'calendar', type: 'extension', version: '2.1.0', description: 'Calendar', enabled: true, source: 'external',
        payload_root: 'skills/external/calendar', review_status: 'clean', review_stale: false, review_gate: { executable_review: true },
        executable_review: true, permissions: [], grants: { all_granted: true }, owner_attestable: true };
    const signIn = status(service({ accounts: [account(REFUSED)] }));
    const card = renderInstalledSkillCard(row, new Set(), new Set(), {}, { signIn });
    assert.match(card.split('data-skill-trays>')[1], /data-sign-in-tray/);
    assert.match(card, /<div class="skills-card-who">[^<]*<p class="skills-card-facts" data-skill-sign-ins>/);
    assert.doesNotMatch(renderInstalledSkillCard({ ...row, enabled: false }, new Set(), new Set(), {}, { signIn }).split('data-skill-trays>')[1],
        /data-sign-in-tray/, 'an Off skill owes no sign-in now');
    assert.doesNotMatch(renderInstalledSkillCard(row), /data-sign-in|data-skill-sign-ins/, 'without a status the card is as before');
});

test('an account whose renewal the provider refused says so on its row and keeps the sign-in', () => {
    const refused = account({ last_error: { reason: 'provider_refused', error: 'admin_policy_enforced', at: new Date().toISOString() } });
    const rows = text(signInRowsHtml(service({ accounts: [refused] })));
    assert.ok(rows.includes('Last check: accounts.google.com refused the renewal (admin_policy_enforced)'), rows);
    assert.ok(rows.includes('The sign-in is kept; ask the account’s administrator to allow this client, then sign in again.'), rows);
    const unreachable = account({ last_error: { reason: 'provider_unreachable', at: new Date().toISOString() } });
    assert.ok(text(signInRowsHtml(service({ accounts: [unreachable] }))).includes('couldn’t reach accounts.google.com'));
});
