// The place sheet and the Connect flow against the contract's PlaceSummary: the state control with each
// state's note (Silent's host card verbatim), what wakes it and who added each item, the owner's accounts,
// notes with their authors, the folder and the model, and the one owner write each press asks for.
import assert from 'node:assert/strict';
import fs from 'node:fs';
import test from 'node:test';

import { normalizeConnector, normalizePlace, normalizePlacesResponse, normalizePlaceStatus } from '../modules/consciousness_api.js';
import { SILENT_CARD, WAKE_FACT_WORDS, placeStatusText, sentenceLabel, wakesForText } from '../modules/consciousness_words.js';
import { dashboardPlaceRowHtml } from '../modules/consciousness_dashboard.js';
import { placeChangeFor, placeSheetBodyHtml, placeSheetMetaText, openPlaceSheet } from '../modules/place_sheet.js';
import { applyPayload, setMissTransport } from '../modules/i18n.js';

import { setup, NodeStub } from './dashboard_dom_fixture.js';
import { installCardDom } from './helpers/card_dom.js';

const FIXTURE = JSON.parse(fs.readFileSync(new URL('./fixtures/consciousness_contract.json', import.meta.url), 'utf8'));
const { places } = normalizePlacesResponse(FIXTURE.places);
const [slack, community, mail] = places;
const strip = (html) => html.replace(/<[^>]+>/g, ' ').replace(/\s+/g, ' ').trim();

test('the subtitle uses the projected service and instance label with its state, never raw ids', () => {
    assert.equal(placeSheetMetaText(slack), 'Slack · Acme · active');
    assert.equal(placeSheetMetaText(community), 'Telegram · Ouroboros community · silent');
    assert.doesNotMatch(placeSheetMetaText(community), /telegram-bot|-100123|ouroboros_bot/);
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
const statusUrl = `/api/extensions/${slack.connector}/status`;
const identityPlace = (fields = {}) => normalizePlace({ ...FIXTURE.places.places[0], source: { url: statusUrl }, ...fields });
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

test('declared core labels keep their checkbox spelling and lowercase only in wake sentences', () => {
    const place = identityPlace({ facts: ['direct_message'], labels: { direct_message: 'Mail addressed to you' },
        wakes: [{ fact: 'direct_message', author: 'connector' }] });
    assert.match(placeSheetBodyHtml(place), /data-i18n-authored>Mail addressed to you</);
    assert.equal(placeStatusText(place), 'active · wakes for mail addressed to you · 2 open turns');
    const fallback = { ...place, labels: null };
    assert.match(placeSheetBodyHtml(fallback), /<span>Direct messages<\/span>/);
    assert.equal(placeStatusText(fallback), 'active · wakes for direct messages · 2 open turns');
    const declared = { ...fallback, facts: [{ id: 'direct_message', label: 'Mail addressed to you' }] };
    assert.equal(placeStatusText(declared), placeStatusText(place));
});

test('sentence labels lower only an initial capital in a lowercase word, preserving mixed case and acronyms', () => {
    for (const [label, sentence] of [['Mail addressed to you', 'mail addressed to you'], ['GitHub', 'GitHub'],
        ['GitHub mentions', 'GitHub mentions'], ['PR', 'PR'], ['PR reviews', 'PR reviews'],
        ['Mail from GitHub', 'mail from GitHub'], ['Émail urgent', 'émail urgent'], ['mail', 'mail'],
        ['1 PR', '1 PR'], ['A reply', 'A reply'], ['', '']]) {
        assert.equal(sentenceLabel(label), sentence);
        if (!label) continue;
        const named = { direct_message: label, from_bot: label };
        assert.equal(wakesForText([{ fact: 'direct_message' }], null, named), `wakes for ${sentence}`);
        assert.equal(wakesForText([{ fact: 'from_bot' }], null, named), sentence);
    }
    assert.equal(wakesForText([{ fact: 'direct_message' }]), 'wakes for direct messages');
});

test('status labels override sheet labels while Dashboard rows retain manifest labels or core fallback', async (t) => {
    const { document, routes, calls } = statusSheetDom(t);
    for (const manifest of [{ direct_message: 'Declared mail' }, null]) {
        for (const labels of [{ direct_message: 'Mail addressed to you' }, undefined, { direct_message: 7 }]) {
            routes.set(statusUrl, jsonResponse({ text: 'Ready', tone: 'ok', labels }));
            const place = identityPlace({ facts: ['direct_message', 'mention', { id: 'signal.words', label: 'PR words' }],
                labels: manifest, wakes: [{ fact: 'direct_message', author: 'connector' }] });
            const before = dashboardPlaceRowHtml(place);
            const sheet = openPlaceSheet(place); await settled();
            const label = typeof labels?.direct_message === 'string' ? labels.direct_message : manifest?.direct_message;
            const wakes = document.body.querySelector('[data-place-section="wakes"]');
            const caption = wakes.querySelector('[data-fact="direct_message"]').parentElement.querySelector('span');
            assert.equal(caption.textContent, label || 'Direct messages');
            assert.equal(caption.hasAttribute('data-i18n-authored'), Boolean(label));
            assert.match(wakes.textContent, /Mentions/);
            assert.match(wakes.textContent, /PR words/);
            assert.equal(dashboardPlaceRowHtml(sheet.place), before, 'sheet status is never copied onto the PlaceSummary');
            assert.match(before, manifest ? /wakes for declared mail/ : /wakes for direct messages/);
            assert.doesNotMatch(before, /Mail addressed to you|mail addressed to you/);
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
        assert.doesNotMatch(body.textContent, /could not be read/);
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
    assert.ok(authored, 'the label sits in an authored span');
    assert.equal(authored.textContent, slack.label);
    assert.equal(meta.textContent, slack.label, 'the reader state shows the label alone, no state word');
});
