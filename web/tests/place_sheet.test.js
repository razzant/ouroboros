// The place sheet and the Connect flow against the contract's PlaceSummary: the state control with each
// state's note (Silent's host card verbatim), what wakes it and who added each item, the owner's accounts,
// notes with their authors, the folder and the model, and the one owner write each press asks for.
import assert from 'node:assert/strict';
import fs from 'node:fs';
import test from 'node:test';

import { normalizeConnector, normalizePlace, normalizePlacesResponse, normalizePlaceStatus } from '../modules/consciousness_api.js';
import { SILENT_CARD, WAKE_FACT_WORDS } from '../modules/consciousness_words.js';
import { inheritedModelWords, placeChangeFor, placeSheetBodyHtml, placeSheetMetaText, openPlaceSheet, recentRooms } from '../modules/place_sheet.js';
import { applyPayload, setMissTransport } from '../modules/i18n.js';

import { setup, NodeStub } from './dashboard_dom_fixture.js';
import { installCardDom } from './helpers/card_dom.js';

const FIXTURE = JSON.parse(fs.readFileSync(new URL('./fixtures/consciousness_contract.json', import.meta.url), 'utf8'));
const { places } = normalizePlacesResponse(FIXTURE.places);
const [slack, community, mail] = places;
const strip = (html) => html.replace(/<[^>]+>/g, ' ').replace(/\s+/g, ' ').trim();

test('the subtitle says only what the title does not: the service and the state, never the label again or a raw id', () => {
    assert.equal(placeSheetMetaText(slack, 'Acme'), 'Slack · active');
    assert.equal(placeSheetMetaText(slack, 'Slack'), 'active', 'a title that is the service is not said twice');
    assert.equal(placeSheetMetaText(community, 'Ouroboros community'), 'telegram · silent');
    assert.doesNotMatch(placeSheetMetaText(community, 'Ouroboros community'), /Ouroboros community|telegram-bot|-100123|ouroboros_bot/);
});

test('the service is the matching connector declaration, independent of the editable place label', () => {
    const raw = { ...FIXTURE.places.places[0], label: 'My team · urgent', provider: 'slack' };
    for (const connectors of [
        [{ connector: 'different', label: 'Wrong service' }, { connector: raw.connector, label: 'Declared service' }], [],
    ]) {
        const [place] = normalizePlacesResponse({ places: [raw], connectors }).places;
        const name = connectors.length ? 'Declared service' : 'slack';
        assert.equal(place.serviceLabel, name);
        const html = placeSheetBodyHtml(place);
        assert.ok(html.includes(`when <span data-i18n-authored>${name}</span> reports it.`));
        assert.doesNotMatch(html, /when .*My team|when .*Wrong service/);
    }
});

test('readiness on and off use the projection; notes and account names stay authored', () => {
    assert.match(placeSheetBodyHtml(slack), /data-tone="ok" role="status">Connector on/);
    assert.match(placeSheetBodyHtml({ ...slack, readiness: { text: 'Off in Skills', tone: 'warn' } }), /data-tone="warn" role="status">Off in Skills/);
    assert.doesNotMatch(placeSheetBodyHtml({ ...slack, readiness: null }), /Connector on|Off in Skills/);
    assert.match(placeSheetBodyHtml(slack), /data-i18n-authored>@anton.r</);
    assert.match(placeSheetBodyHtml(slack), /data-i18n-authored>U01ANTON</);
});

test('a confirmation saved without a name shows the provider name its senders here carry, else the id; a saved name stays', () => {
    const confirmed = (label) => ({ actorId: '144346271', label, confirmedAt: '2026-11-07T13:09:35Z', accountId: '9143838397' });
    const seen = [{ actorId: '144346271', label: 'razzant' }];
    const named = strip(placeSheetBodyHtml({ ...slack, ownerAccounts: [confirmed('')], accountCandidates: seen }));
    assert.match(named, /razzant you · confirmed on [^·]+ · 144346271 Remove/);
    assert.doesNotMatch(named, /This is me/, 'the confirmed sender is not offered again');
    const bare = strip(placeSheetBodyHtml({ ...slack, ownerAccounts: [confirmed('')], accountCandidates: [] }));
    assert.match(bare, /144346271 you · confirmed on [^·]+ Remove/);
    assert.doesNotMatch(bare, /razzant|· 144346271/, 'no name known: the id alone, once');
    const saved = strip(placeSheetBodyHtml({ ...slack, ownerAccounts: [confirmed('Anton R.')], accountCandidates: seen }));
    assert.match(saved, /Anton R\. you · confirmed on [^·]+ · 144346271/);
    assert.doesNotMatch(saved, /razzant/, 'the name the owner saved is never replaced');
});

test('an account another place of this bot confirms counts here: named with that place, removed there, never offered again', () => {
    const shared = (placeLabel) => ({ actorId: '144346271', label: '', confirmedAt: '2026-11-04T10:39:34Z',
        accountId: '9143838397', placeLabel });
    const place = { ...slack, accountId: '9143838397', ownerAccounts: [], accountCandidates: [{ actorId: '144346271', label: 'razzant' }],
        sharedOwnerAccounts: [shared('Household'), shared('FusionBrain')] };
    const html = strip(placeSheetBodyHtml(place));
    assert.match(html, /razzant you · confirmed in Household and 1 more · 144346271/);
    assert.doesNotMatch(html, /This is me|Remove Add mine/, 'counted already: no confirm offer and no Remove here');
    const without = strip(placeSheetBodyHtml({ ...place, sharedOwnerAccounts: [] }));
    assert.match(without, /razzant 144346271 This is me/, 'confirmed nowhere on this bot: offered as before');
    const sender = { id: '144346271', label: 'razzant', kind: 'user' };
    assert.match(strip(placeSheetBodyHtml(place, { identity: { sends_as: sender } })), /It writes here as you \(\s*razzant\s*\)/);
    assert.doesNotMatch(strip(placeSheetBodyHtml({ ...place, sharedOwnerAccounts: [] }, { identity: { sends_as: sender } })),
        /It writes here as you/, 'confirmed nowhere on this bot: not you');
});

// The shared DOM fixture parses HTML and dispatches handlers; these additions supply only the
// window disposal and descendant selector operations used by the production sheet/picker.
function sheetDom(t) {
    const cleanups = [];
    const env = setup({ after: (cleanup) => cleanups.push(cleanup) });
    t.after(async () => {
        for (const dialog of env.document.body.querySelectorAll('dialog')) await dialog.fire('click', { target: dialog.querySelector('[data-place-close]') });
        cleanups.forEach((cleanup) => cleanup());
    });
    Object.assign(env.document.defaultView, {
        removeEventListener: env.window.removeEventListener.bind(env.window),
        addEventListener: env.window.addEventListener.bind(env.window),
    });
    const matches = NodeStub.prototype.matches;
    t.mock.method(NodeStub.prototype, 'matches', function (selector) {
        if (selector === '[data-place-state] [data-effort-value]') return matches.call(this, '[data-effort-value]') && Boolean(this.parentElement?.closest('[data-place-state]'));
        return matches.call(this, selector);
    });
    const priorCss = globalThis.CSS;
    globalThis.CSS = { escape: (value) => value };
    t.after(() => { globalThis.CSS = priorCss; });
    return env;
}
const settled = () => new Promise((resolve) => setImmediate(resolve));
const input = async (body, key, value) => {
    const target = body.querySelector(`[data-place-draft="${key}"]`);
    target.value = value;
    await body.fire('input', { target });
};
const click = (body, selector) => body.fire('click', { target: body.querySelector(selector) });

// The status path also verifies mixed authored/host text in DOM order, including decoded entities.
function statusSheetDom(t) {
    const dom = installCardDom(); const { doc, win } = dom;
    doc.defaultView = win; win.getComputedStyle = () => ({ visibility: 'visible' });
    doc.getElementById = (id) => doc.body.querySelector(`[id="${id}"]`);
    doc.addEventListener = win.addEventListener; doc.removeEventListener = win.removeEventListener;
    const create = doc.createElement;
    doc.createElement = (tag) => {
        const node = create(tag);
        Object.defineProperty(node, 'isConnected', { get: () => doc.body.contains(node) });
        node.getClientRects = () => node.isConnected ? [{}] : [];
        const matches = node.matches.bind(node);
        node.matches = (selector) => matches(selector.replace(/:disabled/g, '[disabled]'));
        return node;
    };
    const routes = new Map(), calls = [];
    t.mock.method(globalThis, 'fetch', async (url, init = {}) => {
        calls.push(url);
        if (!routes.has(url)) throw new Error(`Unexpected request ${url}`);
        const value = routes.get(url);
        if (value instanceof Error) throw value;
        return typeof value === 'function' ? value(url, init) : value;
    });
    t.after(() => { doc.body.querySelectorAll('[data-place-close]').forEach((button) => button.click()); dom.restore(); });
    return { document: doc, routes, calls };
}

const sendingAccount = { id: 'mailbox-id', label: 'mail@example.test', kind: 'user' };
const declaredStatusUrl = `/api/extensions/${slack.connector}/status`;
const statusUrl = `${declaredStatusUrl}?${new URLSearchParams({ account_id: slack.accountId })}`;
const identityPlace = (fields = {}) => normalizePlace({ ...FIXTURE.places.places[0], source: { url: declaredStatusUrl }, ...fields });
const identityPayload = (identity = { sends_as: sendingAccount,
    reads_as: [{ ...sendingAccount, id: 'reader-id', label: 'Reader account' }] }) => ({ text: 'Mailbox ready', tone: 'ok', identity });
const jsonResponse = (data) => ({ ok: true, status: 200, json: async () => data });

test('status GET supplies the four identity states; a confirmed bot never becomes you', async (t) => {
    const { document, routes, calls } = statusSheetDom(t);
    for (const [sender, owner, sentence, confirm] of [
        [sendingAccount, false, 'It writes here as mail@example.test', true],
        [sendingAccount, true, 'It writes here as you (mail@example.test)', false],
        [{ ...sendingAccount, kind: 'bot' }, false, 'It writes here as the Ouroboros bot (mail@example.test)', false],
        [{ ...sendingAccount, kind: 'bot' }, true, 'It writes here as the Ouroboros bot (mail@example.test)', false],
        [null, false, '', false],
    ]) {
        routes.set(statusUrl, (_url, init) => {
            assert.equal(init.method || 'GET', 'GET'); assert.equal(init.cache, 'no-store');
            return jsonResponse(identityPayload(sender ? { sends_as: sender, reads_as: [{ ...sender, label: 'Reader account' }] } : undefined));
        });
        if (!sender) routes.set(statusUrl, jsonResponse({ text: 'Mailbox ready', tone: 'ok' }));
        const place = identityPlace({ owner_accounts: [...FIXTURE.places.places[0].owner_accounts,
            ...(owner ? [{ platform_actor_id: sendingAccount.id, label: sendingAccount.label }] : [])] });
        const sheet = openPlaceSheet(place); await settled();
        const accounts = document.body.querySelector('[data-place-section="accounts"]');
        const accountText = accounts.textContent;
        if (sentence) assert.ok(accountText.includes(`${sentence} · reads as Reader account`));
        else assert.doesNotMatch(accounts.textContent, /It writes here as|reads as|This is my account/);
        assert.equal(accounts.querySelectorAll('[data-place-act="confirm"]').length, Number(confirm));
        if (sender) {
            assert.ok(accountText.indexOf(sentence) < accountText.indexOf('@anton.r'));
            assert.equal(accounts.querySelector('[data-i18n-authored]').textContent, sender.label);
        }
        if (sender?.kind === 'bot') assert.doesNotMatch(accounts.textContent, /It writes here as you/);
        assert.ok(accounts.querySelector('[data-place-act="revoke"]'), 'the existing confirmed list keeps its revoke');
        assert.match(document.body.textContent, /Mailbox ready/);
        sheet.close();
    }
    assert.equal(calls.filter((url) => url === statusUrl).length, 5);
});

test('a confirmation scoped to another account is not this sender; the same or no scope is', async (t) => {
    const { document, routes } = statusSheetDom(t);
    routes.set(statusUrl, jsonResponse(identityPayload({ sends_as: sendingAccount })));
    const base = FIXTURE.places.places[0];
    for (const [scope, owner] of [['someone-elses-mailbox', false], [base.account_id, true], [undefined, true]]) {
        const place = identityPlace({ owner_accounts: [...base.owner_accounts,
            { platform_actor_id: sendingAccount.id, label: sendingAccount.label, ...(scope === undefined ? {} : { account_id: scope }) }] });
        const sheet = openPlaceSheet(place); await settled();
        const accounts = document.body.querySelector('[data-place-section="accounts"]');
        assert.equal(/It writes here as you/.test(accounts.textContent), owner, `scope ${scope}`);
        assert.equal(accounts.querySelectorAll('[data-place-act="confirm"]').length, owner ? 0 : 1, `scope ${scope}`);
        sheet.close();
    }
});

test('pending or failed status and malformed identities add no account row', async (t) => {
    const { document, routes } = statusSheetDom(t);
    let resolve;
    routes.set(statusUrl, () => new Promise((done) => { resolve = done; }));
    const place = identityPlace({ identity: { sends_as: sendingAccount }, can_confirm: true });
    const sheet = openPlaceSheet(place);
    assert.doesNotMatch(document.body.textContent, /It writes here as|This is my account/);
    resolve(jsonResponse(identityPayload())); await settled();
    assert.match(document.body.textContent, /It writes here as/);
    sheet.close();
    for (const identity of [null, [], 'mail', { unknown: true }, { sends_as: null }, { sends_as: [] },
        { sends_as: { ...sendingAccount, id: 7 } }, { sends_as: { ...sendingAccount, label: null } },
        { sends_as: { ...sendingAccount, kind: 'service' } }, { sends_as: { ...sendingAccount, owner: true } },
        { sends_as: { ...sendingAccount, id: '' } }, { sends_as: { ...sendingAccount, label: 'x'.repeat(201) } },
        { sends_as: sendingAccount, reads_as: {} }, { sends_as: sendingAccount, reads_as: [null] },
        { sends_as: sendingAccount, reads_as: [{ ...sendingAccount, kind: false }] }]) {
        routes.set(statusUrl, jsonResponse(identityPayload(identity)));
        const invalid = openPlaceSheet(place); await settled();
        assert.doesNotMatch(document.body.textContent, /It writes here as|It reads here as|This is my account/);
        assert.match(document.body.textContent, /Mailbox ready/, 'a bad identity does not hide readiness');
        invalid.close();
    }
    routes.set(statusUrl, new Error('Unavailable'));
    const failed = openPlaceSheet(place); await settled();
    assert.doesNotMatch(document.body.textContent, /It writes here as|This is my account/);
    assert.match(document.body.textContent, /The connector status could not be read/);
    failed.close();
});

test('status identities preserve reader-only, empty, escaped and disconnected rendering', async (t) => {
    const { document, routes } = statusSheetDom(t);
    for (const [identity, words] of [[{ reads_as: [sendingAccount] }, 'It reads here as'], [{}, ''],
        [{ sends_as: { ...sendingAccount, label: '<img>& account' } }, 'It writes here as']]) {
        routes.set(statusUrl, jsonResponse(identityPayload(identity)));
        const sheet = openPlaceSheet(identityPlace()); await settled();
        const accounts = document.body.querySelector('[data-place-section="accounts"]');
        if (words) assert.ok(accounts.textContent.includes(words));
        else assert.doesNotMatch(accounts.textContent, /It reads here as|It writes here as/);
        if (identity.sends_as) {
            assert.equal(accounts.querySelector('[data-i18n-authored]').textContent, '<img>& account');
            assert.equal(accounts.querySelector('[data-place-act="confirm"]').dataset.actorLabel, '<img>& account');
            assert.equal(accounts.querySelector('img'), null);
        }
        sheet.close();
    }
    assert.doesNotMatch(placeSheetBodyHtml(identityPlace(), { ...normalizePlaceStatus(identityPayload()), removed: true }), /It writes here as|This is my account/);
});

test('This is my account uses the existing confirm writer with the sending id, label and revision', async (t) => {
    const { document, routes } = statusSheetDom(t);
    routes.set(statusUrl, jsonResponse(identityPayload()));
    const place = identityPlace(); const writes = [];
    const sheet = openPlaceSheet(place, { update: async (id, change) => {
        writes.push({ id, change });
        return identityPlace({ revision: 5, owner_accounts: change.revoke_owner_account ? []
            : [{ platform_actor_id: sendingAccount.id, label: sendingAccount.label }] });
    } });
    t.after(() => sheet.close());
    const body = document.body.querySelector('[data-place-body]');
    await settled();
    assert.deepEqual(writes, [], 'opening does not confirm');
    body.querySelector('[data-place-act="confirm"]').click(); await settled();
    assert.deepEqual(writes, [{ id: place.id, change: { confirm_owner_account: {
        platform_actor_id: sendingAccount.id, label: sendingAccount.label }, expected_revision: place.revision } }]);
    assert.match(body.textContent, /It writes here as you \(mail@example.test\)/);
    assert.equal(body.querySelector('[data-place-act="confirm"]'), null);
    assert.equal(body.querySelector('[data-place-act="revoke"]').dataset.actor, sendingAccount.id);
    body.querySelector('[data-place-act="revoke"]').click(); await settled();
    assert.deepEqual(writes[1].change, { revoke_owner_account: sendingAccount.id, expected_revision: 5 });
    assert.doesNotMatch(body.textContent, /It writes here as you/);
    assert.ok(body.querySelector('[data-place-act="confirm"]'));
});

test('place and connector normalization keep labels and Safety without projected identity or confirm flags', () => {
    const labels = { direct_message: 'Mail addressed to you' };
    const place = identityPlace({ identity: { sends_as: sendingAccount }, can_confirm: true, labels, safety_mode: 'full' });
    assert.equal(Object.hasOwn(place, 'identity'), false);
    assert.equal(Object.hasOwn(place, 'canConfirm'), false);
    assert.deepEqual(place.labels, labels); assert.equal(place.safetyMode, 'full');
    assert.deepEqual(normalizeConnector({ labels }).labels, labels);
    for (const labels of [undefined, [], 'label', { direct_message: 9 }]) {
        assert.equal(normalizeConnector({ labels }).labels, null);
        assert.equal(identityPlace({ labels }).labels, null);
        assert.equal(normalizePlaceStatus({ labels }).labels, null);
    }
});

test('notes invite a letter without saving the placeholder and disclose each known Safety level', (t) => {
    const { document } = sheetDom(t); const host = document.createElement('div');
    const questions = 'Who am I here? What may I do? What calls me at once? What matters here? What must I never do? What do you want to see afterwards?';
    for (const [safety_mode, sentence] of [
        ['full', 'Ouroboros reads your words every time; the Safety Supervisor reads them before actions.'],
        ['light', 'Ouroboros reads your words every time; the Safety Supervisor reads them before some actions.'],
        ['off', 'Ouroboros reads your words every time; the Safety Supervisor is off.'],
    ]) {
        host.innerHTML = placeSheetBodyHtml(identityPlace({ safety_mode, notes: [] }));
        const notes = host.querySelector('[data-place-section="notes"]');
        const field = notes.querySelector('textarea');
        assert.equal(field.getAttribute('placeholder'), questions);
        assert.equal(field.getAttribute('rows'), '6', 'the empty letter has room to show its questions');
        assert.equal(field.textContent, '');
        assert.ok(field.hasAttribute('data-i18n-authored'));
        assert.ok(notes.textContent.includes(`${sentence} Silent and Paused are locks; words are not.`));
        assert.ok(notes.textContent.indexOf(sentence) > notes.textContent.indexOf('Add note'));
        assert.doesNotMatch(notes.textContent, /treats yours as your instructions/);
    }
    host.innerHTML = placeSheetBodyHtml(slack, { drafts: { note: 'My own words' } });
    assert.equal(host.querySelector('textarea').textContent, 'My own words');
    assert.equal(host.querySelector('textarea').getAttribute('rows'), '2', 'existing notes keep the compact editor');
    assert.equal(placeChangeFor('add-note', slack, { drafts: {} }), null);
    assert.match(host.textContent, /the Safety Supervisor treats yours as your instructions/);
    assert.doesNotMatch(host.textContent, /Silent and Paused are locks/);
});

test('declared core labels keep their checkbox spelling; a core fact without one reads its core word', () => {
    const place = identityPlace({ facts: ['direct_message'], labels: { direct_message: 'Mail addressed to you' },
        wakes: [{ fact: 'direct_message', author: 'connector' }] });
    assert.match(placeSheetBodyHtml(place), /data-i18n-authored>Mail addressed to you</);
    const fallback = { ...place, labels: null };
    assert.match(placeSheetBodyHtml(fallback), /<span>Direct messages<\/span>/);
    const declared = { ...fallback, facts: [{ id: 'direct_message', label: 'Mail addressed to you' }] };
    assert.match(placeSheetBodyHtml(declared), /data-i18n-authored>Mail addressed to you</);
});

test('status labels override sheet labels and are never written onto the PlaceSummary', async (t) => {
    const { document, routes, calls } = statusSheetDom(t);
    for (const manifest of [{ direct_message: 'Declared mail' }, null]) {
        for (const labels of [{ direct_message: 'Mail addressed to you' }, undefined, { direct_message: 7 }]) {
            routes.set(statusUrl, jsonResponse({ text: 'Ready', tone: 'ok', labels }));
            const place = identityPlace({ facts: ['direct_message', 'mention', { id: 'signal.words', label: 'PR words' }],
                labels: manifest, wakes: [{ fact: 'direct_message', author: 'connector' }] });
            const before = JSON.stringify(place);
            const sheet = openPlaceSheet(place); await settled();
            const label = typeof labels?.direct_message === 'string' ? labels.direct_message : manifest?.direct_message;
            const wakes = document.body.querySelector('[data-place-section="wakes"]');
            const caption = wakes.querySelector('[data-fact="direct_message"]').parentElement.querySelector('span');
            assert.equal(caption.textContent, label || 'Direct messages');
            assert.equal(caption.hasAttribute('data-i18n-authored'), Boolean(label));
            assert.match(wakes.textContent, /Mentions/);
            assert.match(wakes.textContent, /PR words/);
            assert.equal(JSON.stringify(sheet.place), before, 'sheet status is never copied onto the PlaceSummary');
            sheet.close();
        }
    }
    assert.equal(calls.filter((url) => url === statusUrl).length, 6, 'only opening a sheet reads status');
});

test('identity and notes host strings use catalog entries while account labels and notes stay authored', (t) => {
    setMissTransport(() => Promise.resolve(null));
    applyPayload({ language: 'ru', english: false, entries: {
        'It writes here as {label}': { text: 'Пишет как {label}' }, 'reads as {labels}': { text: 'читает как {labels}' },
        'code:places.account.confirm_identity': { text: 'Это мой аккаунт' },
        'code:places.notes.questions': { text: 'Кто я здесь? Что мне можно?' },
        'code:places.notes.safety.off': { text: 'Уроборос читает; проверка выключена.' },
        'code:places.notes.locks': { text: 'Слова не являются замками.' },
    } });
    t.after(() => applyPayload({ language: '', english: true, entries: {} }));
    const html = placeSheetBodyHtml(identityPlace({ safety_mode: 'off' }), normalizePlaceStatus(identityPayload()));
    assert.match(html, /Пишет как <span data-i18n-authored>mail@example.test<\/span>/);
    assert.match(html, /читает как <span data-i18n-authored>Reader account<\/span>/);
    assert.match(html, /Это мой аккаунт/);
    assert.match(html, /placeholder="Кто я здесь\? Что мне можно\?"/);
    assert.match(html, /Уроборос читает; проверка выключена\. Слова не являются замками\./);
    assert.match(html, /data-i18n-authored>Отвечай коротко и по делу\./);
});

for (const removed of [false, true]) test(`a 409 ${removed ? 'disconnect' : 'changed record'} repaints and retains every owner draft`, async (t) => {
    const { document, routes } = sheetDom(t);
    const calls = [];
    const latest = { ...FIXTURE.places.places[0], revision: 9, state: 'paused' };
    routes.set('/api/places/slack-acme', (_url, init) => {
        calls.push(JSON.parse(init.body));
        return { status: 409, ok: false, json: async () => ({ code: 'place_revision_conflict', place: removed ? null : latest }) };
    });
    const sheet = openPlaceSheet(slack);
    t.after(() => sheet.close());
    const body = document.body.querySelector('[data-place-body]');
    await click(body, '[data-place-act="add-mine"]');
    for (const [key, value] of Object.entries({ note: 'Keep this note', folder: '~/new', model: 'my/model', actorId: 'U99', actorLabel: 'My name' })) await input(body, key, value);
    await click(body, '[data-place-act="add-note"]');
    assert.deepEqual(calls, [{ add_note: 'Keep this note', expected_revision: 4 }]);
    assert.equal(sheet.place?.revision ?? null, removed ? null : 9);
    assert.match(body.textContent, /Changed elsewhere; your text is kept\./);
    assert.equal(body.querySelector('[data-place-draft="note"]').textContent, 'Keep this note');
    for (const [key, value] of Object.entries({ folder: '~/new', model: 'my/model', actorId: 'U99', actorLabel: 'My name' })) {
        assert.equal(body.querySelector(`[data-place-draft="${key}"]`).getAttribute('value'), value);
    }
    if (removed) {
        assert.match(body.textContent, /This place is disconnected/);
        assert.equal(body.querySelector('[data-place-state]'), null);
        assert.ok(body.querySelector('[data-place-act="add-note"]').disabled);
    } else assert.equal(sheet.place.state, 'paused');
});

test('a fresh successful write uses CAS and clears only the draft it actually sent', async (t) => {
    const { document } = sheetDom(t);
    let finish;
    const requests = [];
    const sheet = openPlaceSheet(slack, { update: async (id, change) => {
        requests.push({ id, change });
        return new Promise((resolve) => { finish = resolve; });
    } });
    t.after(() => sheet.close());
    const body = document.body.querySelector('[data-place-body]');
    await input(body, 'note', 'First note');
    const writing = click(body, '[data-place-act="add-note"]');
    await input(body, 'note', 'The next note');
    finish({ ...slack, revision: 10 }); await writing;
    assert.equal(body.querySelector('[data-place-draft="note"]').textContent, 'The next note');
    assert.doesNotMatch(body.textContent, /Changed elsewhere/);
    const savingNext = click(body, '[data-place-act="add-note"]');
    finish({ ...slack, revision: 11 }); await savingNext;
    assert.equal(body.querySelector('[data-place-draft="note"]').textContent, '');
    assert.deepEqual(requests.map(({ change }) => change), [
        { add_note: 'First note', expected_revision: 4 }, { add_note: 'The next note', expected_revision: 10 },
    ]);
});

test('state and attention writes carry the record revision and keep connector-selected wakes', async (t) => {
    const { document } = sheetDom(t);
    const requests = [];
    const sheet = openPlaceSheet(slack, { update: async (_id, change) => { requests.push(change); return { ...slack, state: change.state || 'silent', revision: 8 }; } });
    t.after(() => sheet.close());
    const body = document.body.querySelector('[data-place-body]');
    await click(body, '[data-effort-value="silent"]');
    await click(body, '[data-fact="any_message"]');
    assert.deepEqual(requests, [{ state: 'silent', expected_revision: 4 },
        { wakes: ['direct_message', 'mention', 'reply_to_me', 'any_message'], expected_revision: 8 }]);
});

test('a stale disconnect is refused with drafts kept; the fresh disconnect sends the answered revision', async (t) => {
    const { document, routes } = sheetDom(t);
    const requests = [];
    routes.set('/api/places/slack-acme', (_url, init) => {
        requests.push({ method: init.method, headers: init.headers, body: JSON.parse(init.body) });
        return requests.length === 1 ? { ok: false, status: 409, json: async () => ({ code: 'place_revision_conflict', place: { ...FIXTURE.places.places[0], revision: 12 } }) }
            : { ok: true, json: async () => ({ ok: true }) };
    });
    const sheet = openPlaceSheet(slack);
    t.after(() => sheet.close());
    const body = document.body.querySelector('[data-place-body]');
    await input(body, 'note', 'Still here');
    await click(body, '[data-place-act="disconnect"]');
    await click(body, '[data-place-act="disconnect-yes"]');
    assert.equal(sheet.closed, false);
    assert.equal(body.querySelector('[data-place-draft="note"]').textContent, 'Still here');
    assert.match(body.textContent, /Changed elsewhere/);
    await click(body, '[data-place-act="disconnect"]');
    await click(body, '[data-place-act="disconnect-yes"]');
    assert.equal(sheet.closed, true);
    assert.deepEqual(requests.map(({ body }) => body), [{ expected_revision: 4 }, { expected_revision: 12 }]);
    assert.ok(requests.every((request) => request.method === 'DELETE' && request.headers['Content-Type'] === 'application/json'));
});

test('a non-conflict refusal keeps the current record and shows its reason', async (t) => {
    const { document } = sheetDom(t);
    const sheet = openPlaceSheet(slack, { update: async () => { throw new Error('Cannot save while offline.'); } });
    t.after(() => sheet.close());
    const body = document.body.querySelector('[data-place-body]');
    await input(body, 'note', 'Keep me');
    await click(body, '[data-place-act="add-note"]');
    assert.equal(sheet.place, slack);
    assert.match(body.textContent, /Cannot save while offline/);
    assert.doesNotMatch(body.textContent, /Changed elsewhere/);
    assert.equal(body.querySelector('[data-place-draft="note"]').textContent, 'Keep me');
});

for (const failed of [false, true]) test(`model catalog ${failed ? 'failure keeps the saved value' : 'success enriches the editable picker'} and disposes bindings`, async (t) => {
    const { document } = sheetDom(t);
    let reads = 0;
    const sheet = openPlaceSheet(mail, { readModels: async (init) => {
        ++reads; assert.equal(init.cache, 'no-store');
        if (failed) throw new Error('offline');
        return { items: [{ value: 'another/model', label: 'Another model' }] };
    } });
    const body = document.body.querySelector('[data-place-body]');
    const originalInput = body.querySelector('[data-model-chooser]');
    assert.equal(reads, 0);
    const more = body.querySelector('.place-more'); more.open = true;
    await body.fire('toggle', { target: more });
    assert.equal(reads, 1);
    assert.equal(body.querySelector('[data-model-chooser]').getAttribute('value'), mail.model);
    assert.equal(body.querySelector('[data-model-chooser]').getAttribute('role'), 'combobox');
    if (failed) assert.match(body.textContent, /Model suggestions could not be read; your model is kept/);
    else {
        assert.doesNotMatch(body.textContent, /Model suggestions could not be read/);
        await click(body, '[data-place-act="add-mine"]');
        assert.ok(body.querySelector('[data-model-value="another/model"]'));
    }
    sheet.close();
    assert.ok([...originalInput.events.values()].every((listeners) => !listeners.length));
});

test('a live connector status is read once and authored; an off connector is not probed', async (t) => {
    const { document } = sheetDom(t);
    let reads = 0;
    const readStatus = async () => { ++reads; return { text: 'Ready from GitHub', tone: 'ok' }; };
    const off = openPlaceSheet(slack, { readStatus }); off.close();
    assert.equal(reads, 0);
    const live = openPlaceSheet({ ...slack, source: { url: '/api/extensions/github/status' } }, { readStatus });
    await settled();
    assert.equal(reads, 1);
    assert.equal(document.body.querySelector('[role="status"][data-i18n-authored]').textContent, 'Ready from GitHub');
    live.close();
});

test('SVG specs paint their geometry and an empty icon paints the house plug', (t) => {
    const prior = globalThis.DOMPurify;
    globalThis.DOMPurify = { sanitize: (value) => value };
    t.after(() => { globalThis.DOMPurify = prior; });
    const { document } = sheetDom(t);
    const icon = { kind: 'svg', svg: '<svg viewBox="0 0 24 24"><circle r="7"/></svg>' };
    const svgSheet = openPlaceSheet({ ...slack, icon });
    assert.equal(document.body.querySelector('circle').getAttribute('r'), '7');
    svgSheet.close();
    const fallback = openPlaceSheet({ ...slack, icon: {} });
    assert.equal(document.body.querySelector('circle'), null);
    assert.match(document.body.querySelector('path').getAttribute('d'), /M9 3v5M15 3v5/);
    fallback.close();
});

test('Active · Silent · Paused in one control; the note says what the chosen state does', () => {
    const html = placeSheetBodyHtml(slack);
    assert.deepEqual([...html.matchAll(/data-effort-value="(\w+)">(\w+)</g)].map((m) => [m[1], m[2]]),
        [['active', 'Active'], ['silent', 'Silent'], ['paused', 'Paused']]);
    assert.match(html, /data-place-state/);
    assert.match(html, /What it does here/);
    assert.match(html, /Lives here: listens, wakes when its list says so, speaks and acts\./);
    assert.doesNotMatch(html, /place-silent-card/);
    assert.match(placeSheetBodyHtml(mail), /Away from this place: no wakes, no actions\. Still connected; reported events are saved\./);
});

test('while consciousness is off the sheet says the place only hears: Active\'s note and Silent\'s wake note', () => {
    // Owner 06.11: «Lives here: listens, wakes…» stayed on every sheet for the 28 hours Off left the places hearing.
    const off = (place) => ({ ...place, onlyHears: true });
    const active = placeSheetBodyHtml(off(slack));
    assert.match(active, /<p class="place-state-note">Consciousness is off: it only hears here\.<\/p>/);
    assert.doesNotMatch(active, /Lives here: listens/);
    // «A checked kind calls it here at once» stays the rule; the exception under it says Off suspends it, as Paused does.
    assert.match(active, /While consciousness is off, nothing here wakes it\./);
    assert.doesNotMatch(placeSheetBodyHtml(slack), /While consciousness is off/);
    // Beside it the one way to Start: the door to Dashboard › Consciousness, never a second Start; only while Off.
    assert.match(active, /data-place-act="open-consciousness"[^>]*>Consciousness ›</);
    assert.doesNotMatch(active, /data-place-act="start"|>Start</);
    assert.doesNotMatch(placeSheetBodyHtml(slack), /open-consciousness/);
    const silent = placeSheetBodyHtml(off(community));
    assert.match(silent, /While consciousness is off, nothing here wakes it\./);
    assert.doesNotMatch(silent, /these still wake it to read and think/);
    assert.match(silent, /class="place-silent-card"/, 'what Silent blocks is still said, once');
    assert.doesNotMatch(silent, /class="place-state-note"/);
    // With the switch on the stored choice says what it does; the host's fact is a strict boolean.
    assert.match(placeSheetBodyHtml(slack), /Lives here: listens, wakes when its list says so, speaks and acts\./);
    assert.match(placeSheetBodyHtml(community), /Silent: these still wake it to read and think/);
    assert.doesNotMatch(placeSheetBodyHtml(community), /While consciousness is off/);
    assert.deepEqual([true, 'true', undefined].map((only_hears) => normalizePlace({ place_id: 'x', state: 'active', only_hears }).onlyHears),
        [true, false, false]);
});

test('Silent shows the host\'s card: the note and, verbatim, what it blocks and what it cannot', () => {
    const html = placeSheetBodyHtml(community);
    const card = html.match(/<p class="place-silent-card" role="note">([^<]*)<\/p>/)?.[1] || '';
    assert.equal(card.replace(/&#39;/g, "'"),
        `Hears everything here and changes nothing: no messages, reactions, edits or moderation. ${SILENT_CARD}`);
    assert.doesNotMatch(html, /class="place-state-note"/, 'the note is said once, in the card');
});

test('the attention vocabulary comes from the host when absent and honors a declared subset and signal labels', () => {
    const html = placeSheetBodyHtml(slack);
    assert.match(html, /What calls it here/);
    assert.match(strip(html), /Direct messages added by connector/);
    assert.match(strip(html), /Replies added by you/);
    assert.equal((html.match(/type="checkbox"/g) || []).length, Object.keys(WAKE_FACT_WORDS).length - 1);
    assert.match(html, /data-fact="any_message"/);
    assert.match(strip(html), /Every message/);
    assert.match(strip(html), /Also wakes on bot messages/);
    assert.match(html, /without another checked kind/);
    assert.match(placeSheetBodyHtml(mail), /carried over/);
    const declared = { ...slack, facts: ['mention', { id: 'signal.words', label: 'Words you chose' }],
        wakes: [{ fact: 'signal.words', author: 'ouroboros' }] };
    const limited = placeSheetBodyHtml(declared);
    assert.equal((limited.match(/type="checkbox"/g) || []).length, 2);
    assert.match(limited, /data-fact="signal.words" checked/);
    assert.match(limited, /data-i18n-authored>Words you chose</);
    assert.match(limited, /added by Ouroboros/);
    assert.doesNotMatch(limited, /data-fact="any_message"/);
    assert.doesNotMatch(placeSheetBodyHtml({ ...slack, facts: [] }), /type="checkbox"/);
    assert.match(html, /Everything else here it sees at its next own wake and decides what to read\./);
});

test('Silent still wakes for its list and holds what it would say or change; Paused is not woken', () => {
    const silent = placeSheetBodyHtml(community);
    assert.match(silent, /Silent: these still wake it to read and think; what it would say or change here is held\./);
    assert.doesNotMatch(silent, /nothing here wakes it/);
    assert.match(placeSheetBodyHtml(mail), /While this place is Paused, nothing here wakes it\./);
    assert.doesNotMatch(placeSheetBodyHtml({ ...mail, state: 'gone' }), /Silent: these still|While this place is Paused/, 'an unreadable state claims nothing');
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
    assert.match(html, /<li class="place-note is-owner">\s*<p data-i18n-authored>Отвечай коротко и по делу\.<\/p>/);
    assert.match(html, /<li class="place-note">\s*<p data-i18n-authored>Лена ведёт релизы\.<\/p>/);
    assert.match(strip(html), /Лена ведёт релизы\. added by Ouroboros Remove/);
    assert.match(placeSheetBodyHtml(mail), /<li class="place-note is-owner">/, 'a carried-over note was the owner\'s');
    assert.match(html, /the Safety Supervisor treats yours as your instructions/);
});

test('working folder is an address; the model lives under More; Disconnect asks in place', () => {
    const html = placeSheetBodyHtml(slack);
    assert.match(html, /data-place-draft="folder" value="~\/work\/acme"/);
    assert.match(html, /An address, not a fence: work for this place starts here\./);
    assert.match(html, /<details class="consciousness-more place-more">/);
    assert.match(placeSheetBodyHtml(mail), /data-place-draft="model"[^>]*value="anthropic\/claude-sonnet-5\.5"/);
    assert.match(html, /data-place-act="open-connector">Open connector page</);
    assert.match(html, /data-place-act="disconnect">Disconnect</);
    const asking = placeSheetBodyHtml(slack, { confirmDisconnect: true });
    assert.match(strip(asking), /Disconnect Slack · Acme \? It stops hearing and acting here; the room's history stays\./);
    assert.match(asking, /data-i18n-authored>Slack · Acme<\/span>/);
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

test('openRoomChatId opens the captured conversation once the index knows it; an unknown room keeps the editor', async (t) => {
    const { document } = sheetDom(t);
    // The reader clears its host with replaceChildren, which this bookkeeping stub lacks.
    NodeStub.prototype.replaceChildren = function (...nodes) { this.children = []; nodes.forEach((node) => this.appendChild(node)); };
    t.after(() => { delete NodeStub.prototype.replaceChildren; });
    const room = { room_chat_id: 1099511627777, address: 'Slack #general', unanswered_count: 0, label: null, last_at: '', conversation_id: 'C1', thread_id: '' };
    const index = { place_id: slack.id, rooms: [room], coverage: { discovery: 'captured_place_events', importing: false, gaps: [] } };
    const sheet = openPlaceSheet(slack, { readRooms: async () => index, openRoomChatId: 1099511627777 });
    t.after(() => sheet.close());
    await settled(); await settled();
    const body = document.body.querySelector('[data-place-body]');
    const editor = body.querySelector('.place-sheet-editor');
    assert.equal(editor.hidden, true, 'the reader shows the named conversation');
    assert.ok(body.querySelector('[data-history-back]'), 'Back returns to the place');
    const other = openPlaceSheet(slack, { readRooms: async () => index, openRoomChatId: 42 });
    t.after(() => other.close());
    await settled(); await settled();
    const bodies = document.body.querySelectorAll('[data-place-body]');
    const second = bodies[bodies.length - 1];
    assert.equal(second.querySelector('.place-sheet-editor').hidden, false, 'an unknown room leaves the place editor open');
    assert.match(second.textContent, /Slack #general/, 'the conversations list still names the captured room');
});

test('the sheet is an explicit modal whose native close disposes it (the shared Back relies on both)', async (t) => {
    const { document } = sheetDom(t);
    let closed = 0;
    const sheet = openPlaceSheet(slack, { onClose: () => { closed += 1; } });
    const dialog = document.body.querySelector('dialog');
    assert.equal(dialog.getAttribute('aria-modal'), 'true', 'layers.back answers only an explicit modal');
    await dialog.fire('close');
    assert.equal(sheet.closed, true, 'a native close runs the same disposal as Close');
    assert.equal(closed, 1);
    assert.equal(document.body.querySelector('dialog'), null, 'the closed dialog leaves the DOM');
});

test('the reader state keeps the authored place name out of the translation overlay', async (t) => {
    const { document } = sheetDom(t);
    NodeStub.prototype.replaceChildren = function (...nodes) { this.children = []; nodes.forEach((node) => this.appendChild(node)); };
    t.after(() => { delete NodeStub.prototype.replaceChildren; });
    const room = { room_chat_id: 1099511627777, address: 'Slack #general', unanswered_count: 0, label: null, last_at: '', conversation_id: 'C1', thread_id: '' };
    const sheet = openPlaceSheet(slack, { readRooms: async () => ({ place_id: slack.id, rooms: [room], coverage: { discovery: 'captured_place_events', importing: false, gaps: [] } }), openRoomChatId: 1099511627777 });
    t.after(() => sheet.close());
    await settled(); await settled();
    const meta = document.body.querySelector('[data-place-meta]');
    const authored = meta.querySelector('[data-i18n-authored]');
    assert.ok(authored, 'the place\'s name sits in an authored span');
    // The place as its row names it (placeTitle, then the service), never the state word.
    assert.equal(authored.textContent, 'Acme · Slack');
    assert.equal(meta.textContent, 'Acme · Slack', 'the reader state shows the place alone, no state word');
});

// ---- the circle context: the first section ------------------------------------------------------------------------
const CIRCLES = [
    { id: 'team', name: 'Team', text: 'Help the team by the facts.\nKeep money out.', textAuthor: 'owner', textAt: '', revision: 3 },
    { id: 'home', name: 'Home', text: 'Words Ouroboros wrote.', textAuthor: 'ouroboros', textAt: '', revision: 5 },
];
const inCircle = (circle, placedBy = 'owner', extra = {}) => ({ ...slack, circle: { id: circle, placedBy, placedAt: '' }, ...extra });

test('the context section: the circle as a choice, its words with both origins, the place\'s own words beside them', () => {
    const html = placeSheetBodyHtml(inCircle('team'), { circles: CIRCLES });
    const first = html.slice(0, html.indexOf('data-place-section="conversations"'));
    assert.match(first, /data-place-section="circle"/, 'the context leads the sheet, before its conversations');
    assert.match(first, /<option value="team" selected data-i18n-authored>Team<\/option>/);
    assert.match(first, /<option value="">Not in a circle<\/option>/);
    assert.match(first, /<p class="place-circle-text" data-i18n-authored>Help the team by the facts\.\nKeep money out\.<\/p>/, 'the words whole in the DOM; CSS clamps them');
    assert.match(strip(first), /written by you · placed here by you/);
    assert.doesNotMatch(first, /keep-circle/, 'my human\'s own placement needs no keeping');
    assert.match(first, /data-place-act="edit-circle">Edit in circle ›</);
    // Ouroboros placed it: both origins named, and my human may make the placement theirs without a pending state.
    const placed = placeSheetBodyHtml(inCircle('team', 'ouroboros'), { circles: CIRCLES });
    assert.match(strip(placed), /written by you · placed here by Ouroboros · Keep it in Team/);
    assert.deepEqual(placeChangeFor('keep-circle', inCircle('team', 'ouroboros')), { circle_id: 'team' });
    const theirs = strip(placeSheetBodyHtml(inCircle('home'), { circles: CIRCLES }));
    assert.match(theirs, /written by Ouroboros · placed here by you/);
    // No circle: its own notes only; a reading of circles that is not back yet says so instead of a choice.
    assert.match(strip(placeSheetBodyHtml(slack, { circles: CIRCLES })), /Only its own notes apply here/);
    assert.match(placeSheetBodyHtml(slack, { circles: null }), /Reading circles…/);
    assert.doesNotMatch(placeSheetBodyHtml(slack, { circles: null }), /data-place-circle/);
});

test('the place\'s own notes in brief keep each author\'s voice; the full Notes stay below', () => {
    const notes = [{ id: 'a', text: 'Owner words', author: 'owner', at: '' }, { id: 'b', text: 'My own record', author: 'ouroboros', at: '' }];
    const html = placeSheetBodyHtml({ ...slack, notes }, { circles: CIRCLES });
    const brief = html.slice(0, html.indexOf('data-place-section="conversations"'));
    assert.match(brief, /<div class="place-note place-note-preview is-owner"><p data-i18n-authored>Owner words<\/p><\/div>/);
    assert.match(brief, /<div class="place-note place-note-preview"><p data-i18n-authored>My own record<\/p><div class="place-note-foot">/,
        'Ouroboros\'s note is never painted in my human\'s voice');
    assert.match(strip(brief), /2 notes of its own · All notes ›/);
    assert.match(html, /data-place-section="notes"/);
    assert.match(strip(placeSheetBodyHtml({ ...slack, notes: [] }, { circles: CIRCLES })), /No notes of its own yet\. Add a note ›/);
});

test('the description is its own optional line: empty until written, never a note\'s first line, saved and cleared alone', () => {
    const notes = [{ id: 'a', text: 'First line of a note\nmore', author: 'owner', at: '' }];
    const empty = placeSheetBodyHtml({ ...slack, notes, description: null }, { circles: CIRCLES });
    assert.match(empty, /<input class="ui-control" data-place-draft="description" data-i18n-authored value=""/);
    assert.match(strip(empty), /It says what the place is for and adds no rule\./);
    const written = { ...slack, description: { text: 'Where the team asks', author: 'ouroboros', at: '' } };
    const html = placeSheetBodyHtml(written, { circles: CIRCLES });
    assert.match(html, /value="Where the team asks"/);
    assert.match(strip(html), /adds no rule\. Written by Ouroboros\./, 'who wrote it starts its own sentence');
    assert.doesNotMatch(strip(html), /\. written by/);
    assert.deepEqual(placeChangeFor('save-description', written, { drafts: { description: '  A   new\nline ' } }), { description: 'A new line' });
    assert.deepEqual(placeChangeFor('save-description', written, { drafts: { description: '' } }), { description: '' });
    assert.equal(placeChangeFor('save-description', written, { drafts: { description: 'Where the team asks' } }), null);
    assert.equal(placeChangeFor('save-description', { ...slack, description: null }, { drafts: {} }), null);
    assert.deepEqual(placeChangeFor('circle', inCircle('team'), { value: 'home' }), { circle_id: 'home' });
    assert.deepEqual(placeChangeFor('circle', inCircle('team'), { value: '' }), { circle_id: '' });
    assert.equal(placeChangeFor('circle', inCircle('team'), { value: 'team' }), null);
});

test('conversations: one door to the newest, three recent rows, the rest on request', () => {
    const room = (id, at) => ({ room_chat_id: id, address: `room ${id}`, unanswered_count: 0, label: `Room ${id}`, last_at: at, conversation_id: `c${id}`, thread_id: '' });
    const rooms = [room(1, '2026-11-01T00:00:00Z'), room(2, '2026-11-03T00:00:00Z'), room(3, '2026-11-02T00:00:00Z'), room(4, '2026-11-04T00:00:00Z')];
    assert.deepEqual(recentRooms(rooms).map((r) => r.room_chat_id), [4, 2, 3, 1]);
    const coverage = { discovery: 'arrivals', importing: false, gaps: false };
    const html = placeSheetBodyHtml(slack, { circles: CIRCLES, roomIndex: { rooms, coverage } });
    const section = html.slice(html.indexOf('data-place-section="conversations"'), html.indexOf('data-place-section="readiness"'));
    assert.match(section, /data-primary data-place-act="open-latest">Open conversation</);
    assert.match(section, /data-place-act="all-rooms">All 4 ›</);
    assert.equal((section.match(/data-place-room=/g) || []).length, 3);
    const all = placeSheetBodyHtml(slack, { circles: CIRCLES, roomIndex: { rooms, coverage }, allRooms: true });
    assert.equal((all.match(/data-place-room=/g) || []).length, 4);
    assert.match(placeSheetBodyHtml(slack, { circles: CIRCLES, roomIndex: { rooms: [], coverage } }), /data-place-act="open-latest" data-disabled>/);
});

test('the live sheet: its title is the row\'s name, a circle choice and a kept placement write once with the revision', async (t) => {
    const { document } = sheetDom(t);
    const requests = [];
    const place = inCircle('team', 'ouroboros', { label: 'Slack · Acme · Team' });
    const sheet = openPlaceSheet(place, { circles: CIRCLES, readRooms: async () => ({ rooms: [], coverage: { discovery: 'arrivals', importing: false, gaps: false } }),
        update: async (_id, change) => { requests.push(change); return { ...place, revision: place.revision + requests.length,
            circle: change.circle_id ? { id: change.circle_id, placedBy: 'owner', placedAt: '' } : null,
            description: change.description ? { text: change.description, author: 'owner', at: '' } : place.description }; } });
    t.after(() => sheet.close());
    await settled();
    const label = document.body.querySelector('[data-place-label]');
    assert.equal(label.textContent, 'Acme', 'the service and the circle\'s own name are not repeated in the title');
    // The stub DOM reads text before elements, so the meta's parts are read apart (the real-browser pass sees order).
    const meta = document.body.querySelector('[data-place-meta]');
    assert.equal(meta.querySelector('[data-i18n-authored]').textContent, 'Slack');
    assert.match(meta.textContent, /· active/);
    assert.doesNotMatch(meta.textContent, /Acme|Team/);
    const body = document.body.querySelector('[data-place-body]');
    await click(body, '[data-place-act="keep-circle"]');
    assert.deepEqual(requests.at(-1), { circle_id: 'team', expected_revision: 4 });
    assert.match(body.textContent, /Your choice now\./);
    const select = body.querySelector('[data-place-circle]');
    select.value = 'home';
    await body.fire('change', { target: select });
    assert.deepEqual(requests.at(-1), { circle_id: 'home', expected_revision: 5 });
    assert.match(body.textContent, /Now in Home\./);
    await input(body, 'description', 'Where the team asks');
    await click(body, '[data-place-act="save-description"]');
    assert.deepEqual(requests.at(-1), { description: 'Where the team asks', expected_revision: 6 });
    assert.match(body.textContent, /Description saved\./);
    assert.equal(requests.length, 3, 'no write the owner did not ask for');
});

test('a sheet opened without circles reads them once from the places list; a failed read says so and offers no choice', async (t) => {
    const { document } = sheetDom(t);
    let reads = 0;
    const sheet = openPlaceSheet(inCircle('team'), { readRooms: async () => ({ rooms: [], coverage: { discovery: 'arrivals', importing: false, gaps: false } }),
        readList: async () => { reads += 1; return { places: [inCircle('team', 'owner', { revision: 9 })], circles: CIRCLES, connectors: [] }; } });
    t.after(() => sheet.close());
    await settled(); await settled();
    const body = document.body.querySelector('[data-place-body]');
    assert.equal(reads, 1);
    assert.ok(body.querySelector('[data-place-circle]'));
    assert.match(body.textContent, /Help the team by the facts\./);
    assert.equal(sheet.place.revision, 9, 'the list\'s newer record replaces the opener\'s older one');
    const failed = openPlaceSheet(slack, { readRooms: async () => ({ rooms: [], coverage: { discovery: 'arrivals', importing: false, gaps: false } }),
        readList: async () => { throw new Error('offline'); } });
    t.after(() => failed.close());
    await settled(); await settled();
    const bodies = document.body.querySelectorAll('[data-place-body]');
    const second = bodies[bodies.length - 1];
    assert.match(second.textContent, /Circles could not be read\./);
    assert.equal(second.querySelector('[data-place-circle]'), null);
    assert.doesNotMatch(second.textContent, /Reading circles/, 'a failed read is not still reading');
    assert.equal(second.querySelector('[data-place-act="read-circles"]').textContent, 'Try again');
});

test('Try again reads the circles again; circles read before stay chosen beside a later failure', async (t) => {
    const env = sheetDom(t);
    let fail = true, reads = 0;
    const readList = async () => {
        reads += 1;
        if (fail) throw new Error('offline');
        return { places: [inCircle('team')], circles: CIRCLES, connectors: [] };
    };
    const sheet = openPlaceSheet(inCircle('team'), { readRooms: async () => ({ rooms: [], coverage: { discovery: 'arrivals', importing: false, gaps: false } }), readList });
    t.after(() => sheet.close());
    await settled(); await settled();
    const body = env.document.body.querySelector('[data-place-body]');
    fail = false;
    await click(body, '[data-place-act="read-circles"]');
    assert.equal(reads, 2);
    assert.equal(env.document.activeElement, body.querySelector('[data-place-circle]'), 'the choice it brought takes the focus');
    assert.doesNotMatch(body.textContent, /could not be read/);
    // A circle changed elsewhere and that read fails: the choice stays, with the failure and Try again beside it.
    fail = true;
    env.window.dispatchEvent(new CustomEvent('ouro:places-changed', { detail: { circleId: 'home', removed: true } }));
    await settled(); await settled();
    assert.ok(body.querySelector('[data-place-circle]'));
    assert.match(body.textContent, /Circles could not be read\./);
    fail = false;
    await click(body, '[data-place-act="read-circles"]');
    assert.equal(body.querySelector('[data-place-act="read-circles"]'), null);
    assert.equal(reads, 4);
});

// ---- replacing drafts: a description, folder or model draft is written on the revision it was typed against ----------
// The server's side of a place write, as places._write does it: compare-and-set on the revision, and a refused write
// answers the place as it stands. `edit` is another client's write, or the one the compare let through.
function placeServer(raw = FIXTURE.places.places[0]) {
    const server = { raw: structuredClone(raw), writes: [] };
    server.edit = (change) => {
        if ('description' in change) server.raw.description = change.description ? { text: change.description, author: 'owner', at: '' } : null;
        for (const key of ['working_folder', 'model', 'state']) if (key in change) server.raw[key] = change[key];
        if ('wakes' in change) server.raw.wakes = change.wakes.map((fact) => ({ fact, author: 'owner' }));
        server.raw.revision += 1;
        return normalizePlace(server.raw);
    };
    server.update = async (_id, { expected_revision: expected, ...change }) => {
        server.writes.push({ ...change, expected_revision: expected });
        if (expected !== server.raw.revision) {
            throw Object.assign(new Error(`Current revision: ${server.raw.revision}`),
                { status: 409, body: { code: 'place_revision_conflict', place: structuredClone(server.raw) } });
        }
        return server.edit(change);
    };
    return server;
}
const noRooms = async () => ({ rooms: [], coverage: { discovery: 'arrivals', importing: false, gaps: false } });
const valueOf = (body, key) => body.querySelector(`[data-place-draft="${key}"]`).getAttribute('value');
const deferredList = () => { let answer; const promise = new Promise((resolve) => { answer = resolve; }); return { read: () => promise, answer }; };

for (const [key, action, wire, mine, theirs] of [
    ['description', 'save-description', 'description', 'Mine: where the team asks', 'Theirs: the release desk'],
    ['folder', 'save-folder', 'working_folder', '~/work/mine', '~/work/theirs'],
    ['model', 'save-model', 'model', 'mine/model', 'theirs/model'],
]) test(`a ${key} draft keeps the revision it was typed on across a newer places read; that Save meets the conflict, the next is the choice`, async (t) => {
    const { document } = sheetDom(t);
    const server = placeServer();
    const list = deferredList();
    // Opened without circles (a Dashboard door, Add): the sheet reads the places list, and that read is slow.
    const sheet = openPlaceSheet(slack, { update: server.update, readRooms: noRooms, readList: list.read });
    t.after(() => sheet.close());
    const body = document.body.querySelector('[data-place-body]');
    const saved = () => (wire === 'description' ? server.raw.description?.text : server.raw[wire]);
    await input(body, key, mine);  // typed on revision 4
    server.edit({ [wire]: theirs });  // another client saves revision 5
    list.answer({ places: [normalizePlace(server.raw)], circles: CIRCLES, connectors: [] });
    await settled(); await settled();
    assert.equal(sheet.place.revision, 5, 'the newer read is shown');
    assert.equal(valueOf(body, key), mine, 'the draft stays over it');
    await click(body, `[data-place-act="${action}"]`);
    assert.deepEqual(server.writes, [{ [wire]: mine, expected_revision: 4 }], 'the draft\'s own revision, never the newer read\'s');
    assert.equal(saved(), theirs, 'the other client\'s words stay');
    assert.match(body.textContent, /Changed elsewhere; your text is kept\./);
    assert.equal(valueOf(body, key), mine);
    await click(body, `[data-place-act="${action}"]`);
    assert.deepEqual(server.writes[1], { [wire]: mine, expected_revision: 5 }, 'the Save after the conflict is the deliberate one');
    assert.equal(saved(), mine);
    assert.doesNotMatch(body.textContent, /Changed elsewhere/);
});

test('a place another surface announces shows at once and leaves a draft on its own revision', async (t) => {
    const env = sheetDom(t);
    const server = placeServer();
    const sheet = openPlaceSheet(slack, { circles: CIRCLES, update: server.update, readRooms: noRooms });
    t.after(() => sheet.close());
    const body = env.document.body.querySelector('[data-place-body]');
    await input(body, 'folder', '~/work/mine');
    env.window.dispatchEvent(new CustomEvent('ouro:places-changed', { detail: { place: server.edit({ working_folder: '~/work/theirs' }) } }));
    assert.equal(sheet.place.revision, 5);
    await click(body, '[data-place-act="save-folder"]');
    assert.deepEqual(server.writes, [{ working_folder: '~/work/mine', expected_revision: 4 }]);
    assert.equal(server.raw.working_folder, '~/work/theirs');
    await click(body, '[data-place-act="save-folder"]');
    assert.deepEqual(server.writes[1], { working_folder: '~/work/mine', expected_revision: 5 });
});

test('the sheet\'s own writes carry an open draft forward; a conflict re-bases only the draft whose Save met it', async (t) => {
    const { document } = sheetDom(t);
    const server = placeServer();
    const sheet = openPlaceSheet(slack, { circles: CIRCLES, update: server.update, readRooms: noRooms });
    t.after(() => sheet.close());
    const body = document.body.querySelector('[data-place-body]');
    await input(body, 'description', 'Where the team asks');  // on revision 4
    await click(body, '[data-fact="any_message"]');  // the owner's own press: 4 → 5
    await input(body, 'folder', '~/work/acme-2');  // on revision 5
    await click(body, '[data-effort-value="silent"]');  // 5 → 6
    await click(body, '[data-place-act="save-description"]');
    assert.deepEqual(server.writes.map((write) => write.expected_revision), [4, 5, 6], 'its own presses never conflict with its draft');
    assert.doesNotMatch(body.textContent, /Changed elsewhere/);
    // Another client writes (revision 8); the folder draft (carried to 7) meets the conflict on its own Save, and the
    // model draft typed before it keeps its revision: its first Save meets the conflict too.
    await input(body, 'model', 'mine/model');  // on revision 7
    server.edit({ state: 'paused' });
    await click(body, '[data-place-act="save-folder"]');
    await click(body, '[data-place-act="save-model"]');
    await click(body, '[data-place-act="save-folder"]');
    await click(body, '[data-place-act="save-model"]');
    assert.deepEqual(server.writes.slice(3).map((write) => [Object.keys(write)[0], write.expected_revision]),
        [['working_folder', 7], ['model', 7], ['working_folder', 8], ['model', 9]]);
    assert.equal(server.raw.working_folder, '~/work/acme-2');
    assert.equal(server.raw.model, 'mine/model');
    assert.equal(server.raw.state, 'paused', 'nothing the other client wrote was lost');
});

test('words typed while a description Save is out stay a draft on the revision that Save made', async (t) => {
    const { document } = sheetDom(t);
    const server = placeServer();
    let release;
    const held = new Promise((resolve) => { release = resolve; });
    const sheet = openPlaceSheet(slack, { circles: CIRCLES, readRooms: noRooms,
        update: async (id, change) => { if (!server.writes.length) await held; return server.update(id, change); } });
    t.after(() => sheet.close());
    const body = document.body.querySelector('[data-place-body]');
    await input(body, 'description', 'First words');
    const saving = click(body, '[data-place-act="save-description"]');
    await input(body, 'description', 'First words, and more');
    release(); await saving;
    assert.equal(server.raw.description.text, 'First words');
    assert.equal(valueOf(body, 'description'), 'First words, and more', 'the newer words stay');
    assert.doesNotMatch(body.textContent, /Changed elsewhere/);
    await click(body, '[data-place-act="save-description"]');
    assert.deepEqual(server.writes.map((write) => write.expected_revision), [4, 5], 'on the revision the first Save made');
    assert.equal(server.raw.description.text, 'First words, and more');
});

// A place inherits its circle's model and may override it; the default is the consciousness model (owner 10.11).
const GLM = 'z-ai/glm-5.3';
const ASTRA = 'claudexor::codex=gpt-6-astra';
const TEAM = { id: 'team', name: 'Team', text: '', textAuthor: '', textAt: '', model: GLM, revision: 3 };
const HOME = { id: 'home', name: 'Home', text: '', textAuthor: '', textAt: '', model: '', revision: 4 };
const placed = (circle, model, route) => normalizePlace({ ...FIXTURE.places.places[0], model, model_route: route,
    circle: circle ? { circle_id: circle, placed_by: 'owner', placed_at: '' } : null });
const modelField = (html) => ({ value: /data-place-draft="model"[^>]*value="([^"]*)"/.exec(html)[1],
    placeholder: /data-place-draft="model"[^>]*placeholder="([^"]*)"/.exec(html)[1].replaceAll('&#39;', "'"),
    help: /id="place-model-help"[^>]*>([^<]*)</.exec(html)[1].replaceAll('&#39;', "'") });

test('an empty Model field names what the place uses: its circle\'s model, else the consciousness model', () => {
    const circles = [TEAM, HOME];
    const inherits = modelField(placeSheetBodyHtml(placed('team', '', { value: GLM, layer: 'circle' }), { circles }));
    assert.deepEqual(inherits, { value: '', placeholder: "Uses Team's model: glm-5.3",
        help: "The model it thinks with when it wakes here; empty uses Team's model, glm-5.3." });
    const follows = modelField(placeSheetBodyHtml(placed('home', '', { value: ASTRA, layer: 'consciousness' }), { circles }));
    assert.deepEqual(follows, { value: '', placeholder: 'Uses the consciousness model: gpt-6-astra',
        help: 'The model it thinks with when it wakes here; empty uses the Consciousness model in Models, gpt-6-astra.' });
    // In no circle the default is the same; a server that does not say names no model.
    assert.equal(modelField(placeSheetBodyHtml(placed('', '', { value: ASTRA, layer: 'consciousness' }), { circles })).placeholder,
        'Uses the consciousness model: gpt-6-astra');
    assert.deepEqual(modelField(placeSheetBodyHtml(placed('home', '', undefined), { circles })), { value: '',
        placeholder: 'Empty uses the consciousness model', help: 'The model it thinks with when it wakes here; empty uses the Consciousness model in Models.' });
    assert.match(placeSheetBodyHtml(placed('team', '', { value: GLM, layer: 'circle' }), { circles }),
        /aria-describedby="place-model-help"[\s\S]*id="place-model-help" data-i18n-fmt="places\.model\.help">/, 'the help describes the field and names a model, not chrome');
});

test('the field holds only the place\'s own model; under an override the help says what clearing it returns to', () => {
    const circles = [TEAM, HOME];
    const own = modelField(placeSheetBodyHtml(placed('team', 'openai::gpt-6', { value: 'openai::gpt-6', layer: 'place' }), { circles }));
    assert.deepEqual(own, { value: 'openai::gpt-6', placeholder: "Uses Team's model: glm-5.3",
        help: "The model it thinks with when it wakes here; empty uses Team's model, glm-5.3." }, 'the circle is read for the inherited value');
    // The place's row carries its own model then, so the consciousness model is not named by this read.
    assert.equal(modelField(placeSheetBodyHtml(placed('home', 'main', { value: ASTRA, layer: 'place' }), { circles })).help,
        'The model it thinks with when it wakes here; empty uses the Consciousness model in Models.');
    // A draft is the field's value until it is saved or cleared; the inherited words stay what the circle says.
    assert.equal(modelField(placeSheetBodyHtml(placed('team', 'openai::gpt-6', { value: 'openai::gpt-6', layer: 'place' }), { circles, drafts: { model: '' } })).value, '');
});

test('a circle\'s alias is named by what it resolves to for a place that inherits it, by its role otherwise', () => {
    const aliased = [{ ...TEAM, model: 'main' }, HOME];
    assert.equal(inheritedModelWords(placed('team', '', { value: ASTRA, layer: 'circle' }), aliased).placeholder, "Uses Team's model: gpt-6-astra");
    assert.equal(inheritedModelWords(placed('team', GLM, { value: GLM, layer: 'place' }), aliased).placeholder, "Uses Team's model: Main");
    // A circle whose model was edited does not revise its places' rows: the circle as last read is what is named.
    assert.equal(inheritedModelWords(placed('team', '', { value: ASTRA, layer: 'circle' }), [{ ...TEAM, model: 'openai::gpt-6' }]).placeholder,
        "Uses Team's model: gpt-6");
    assert.equal(inheritedModelWords(placed('team', '', { value: GLM, layer: 'circle' }), null).placeholder, 'Empty uses the consciousness model',
        'circles not read yet: nothing is claimed for them');
});

test('clearing the field returns the place to inheriting: one write of the empty model, then what it uses now', async (t) => {
    const { document } = sheetDom(t);
    const requests = [];
    const start = placed('team', 'openai::gpt-6', { value: 'openai::gpt-6', layer: 'place' });
    const sheet = openPlaceSheet(start, { circles: [TEAM, HOME], update: async (id, change) => {
        requests.push(change);
        return change.model ? { ...start, model: change.model, modelRoute: { value: change.model, layer: 'place' }, revision: start.revision + requests.length }
            : { ...start, model: '', modelRoute: { value: GLM, layer: 'circle' }, revision: start.revision + requests.length };
    } });
    t.after(() => sheet.close());
    const body = document.body.querySelector('[data-place-body]');
    await input(body, 'model', '  ');
    await click(body, '[data-place-act="save-model"]');
    assert.deepEqual(requests, [{ model: '', expected_revision: start.revision }]);
    const plain = (text) => text.replaceAll('&#39;', "'");  // the stub DOM keeps an escaped apostrophe as written
    assert.match(plain(body.textContent), /Now uses Team's model: glm-5\.3\./);
    assert.equal(plain(body.querySelector('[data-place-draft="model"]').getAttribute('placeholder')), "Uses Team's model: glm-5.3");
    assert.equal(body.querySelector('[data-place-draft="model"]').getAttribute('value'), '');
    // And an override again is saved as the place's own.
    await input(body, 'model', GLM);
    await click(body, '[data-place-act="save-model"]');
    assert.deepEqual(requests[1], { model: GLM, expected_revision: start.revision + 1 });
    assert.match(body.textContent, /Model saved\./);
});
