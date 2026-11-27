import assert from 'node:assert/strict';
import test from 'node:test';
import {
    confirmedTelegramBot, connectChoices, connectListHtml, openConnectPlace, openConnectorSettings,
    parseOwnerBotStatus, parseSeparateBotStatus, parseTelegramGroupFeed, telegramGroupChoices,
    validatedTelegramInviteUrl,
} from '../modules/place_connect.js';
import {
    connectPlace, connectorKey, connectorMatchesPlace, normalizeConnector, normalizePlace,
    normalizePlacesResponse, readPlaceConnector, readPlaces,
} from '../modules/consciousness_api.js';
import { initCircles } from '../modules/circles.js';
import { openPlaceSheet } from '../modules/place_sheet.js';
import { telegramFacts, telegramGroupsHtml } from '../modules/telegram_groups_connect.js';
import { setup, NodeStub } from './dashboard_dom_fixture.js';

const raw = { connector: 'github', label: 'GitHub', icon: { kind: 'svg', svg: '<svg viewBox="0 0 24 24"><circle r="7"/></svg>' },
    source: { url: '/api/extensions/github/status', refresh_s: 60 }, settings: 'connection',
    readiness: { text: 'Connector on', tone: 'ok' }, wakes: ['mention'], facts: ['mention', { id: 'signal.words', label: 'Words you chose' }] };
const github = normalizeConnector(raw);
const rows = [{ account_id: 'writer', conversation_id: 'repo.123', label: 'example/project' }];
const place = normalizePlace({ ...raw, place_id: 'gh-1', label: 'GitHub · example/project', account_id: 'writer', conversation_id: 'repo.123', state: 'silent', revision: 7 });
const strip = (html) => html.replace(/<[^>]*>/g, ' ').replace(/\s+/g, ' ').trim();
const choicesHtml = (data, extra = {}) => connectListHtml({ connectors: [github], selected: github, choices: connectChoices(data), ...extra });

test('Telegram Groups counts only bot group places, including legacy Telegram Bot rows', () => {
    const tg = (id, contribution_id, conversation_id, connector = 'telegram-bot') => normalizePlace({
        place_id: id, provider: 'telegram', connector, contribution_id, conversation_id,
    });
    const places = [tg('a', 'owner_bot', '-1001'), tg('b', undefined, '-1002'),
        tg('dm', 'place', '1004'), tg('account', 'place', ''), tg('user-session', 'place', '-1003', 'telegram-user-session'),
        tg('other-contribution', 'other', '-1005')];
    assert.deepEqual(telegramFacts({ places }).groupPlaces.map((p) => p.id), ['a', 'b']);
    assert.match(connectListHtml({ places, connectors: [] }), /data-tone="muted">Connected · 2 groups/);
});

test('a pending modern group intake never offers an invitation; an absent legacy field still uses exact Hub status', () => {
    const identity = { sends_as: { id: '101', label: '@mybot', kind: 'bot' } };
    const feed = { v: 1, state: 'ready', owner_bound: true, capable: true, enabled: true,
        account_id: '101', consent_account_id: '101', identity, invite_url: 'https://t.me/mybot?startgroup=true' };
    const empty = { v: 1, enabled: true, reason: '', account_id: '101', identity, connect: [] };
    const connector = { connector: 'telegram-bot', contributionId: 'owner_bot' };
    const html = (wire) => telegramGroupsHtml({ connectors: [connector], tgBranch: 'personal', tg: {
        groupFeed: parseTelegramGroupFeed(wire), ownerStatus: parseOwnerBotStatus(empty, '101'),
    } }, 'It starts Silent.');
    for (const value of [false, null, 'true']) {
        const waiting = html({ ...feed, intake_ready: value });
        assert.match(waiting, /Groups are allowed for @mybot/);
        assert.match(waiting, /data-connect-act="tg-refresh"/);
        assert.match(waiting, /data-connect-act="tg-opt-out"/);
        assert.match(waiting, /data-branch="separate"/);
        assert.doesNotMatch(waiting, /data-connect-invite|Send a message|data-connect-act="tg-opt-in"/);
    }
    assert.match(html(feed), /data-connect-invite="personal"/);
    assert.match(html({ ...feed, intake_ready: true }), /data-connect-invite="personal"/);
});

test('the reader keeps the single projection: service label, revision, sanitized icon, readiness and declarations', () => {
    assert.equal(github.label, 'GitHub');
    assert.equal(github.connectorLabel, undefined);
    assert.equal(place.revision, 7);
    assert.equal(place.icon.kind, 'svg');
    assert.deepEqual(place.readiness, raw.readiness);
    assert.deepEqual(place.facts, raw.facts);
    assert.deepEqual(place.source, raw.source);
    assert.equal(place.settings, 'connection');
    const missing = normalizePlace({ connector: 'raw-skill-id', connector_label: 'old title', icon: {} });
    assert.equal(missing.label, '');
    assert.equal(missing.revision, null);
    assert.equal(missing.readiness, null);
    assert.equal(missing.facts, null, 'absence differs from a declared empty vocabulary');
    assert.equal(missing.spent24hUsd, null);
    assert.equal(normalizePlace({ spent_24h_usd: 0 }).spent24hUsd, 0);
});

test('Where is it: every connector row says its state in one family; its verb follows the state', () => {
    const html = connectListHtml({ connectors: [github], places: [place, { ...place, id: 'gh-2' }] });
    assert.match(strip(html), /It starts Silent: it hears and thinks there and sends nothing until you make it Active\./);
    assert.match(html, /<div class="harness-account-meta"><span class="ui-status" data-tone="ok">Connected · 2 places<\/span><\/div>/);
    assert.match(strip(html), /Connected · 2 places Choose where/);
    assert.match(html, /data-connect-act="back-words"/, 'Back returns to the words');
    // A connector that is simply on and has no place says so; one that is off says its readiness and opens Skills.
    assert.match(connectListHtml({ connectors: [github] }), /data-tone="muted">Not connected<.*data-connect-act="choose"[^>]*>Connect</s);
    const off = connectListHtml({ connectors: [{ ...github, readiness: { text: 'Off in Skills', tone: 'warn' } }], places: [place] });
    assert.match(off, /data-tone="warn">Off in Skills</);
    assert.match(off, /data-connect-act="skills" data-connector="github"[^>]*>Enable in Skills</);
    assert.match(connectListHtml({ connectors: [{ ...github, source: null }] }), /data-tone="muted">Not connected<.*data-connect-act="choose"/s);
    assert.doesNotMatch(connectListHtml({ connectors: [{ ...github, readiness: { text: 'Connector on', tone: 'ok' } }] }), /Connector on/);
    assert.match(html, /data-i18n-authored>GitHub</);
    assert.match(connectListHtml({ connectors: [] }), /No installed connector/);
    assert.match(connectListHtml({}), /Reading the installed connectors/);
    assert.match(connectListHtml({ error: 'offline' }), /data-connect-act="retry"/);
});

test('the four instance-list states remain distinct and every state keeps Whole account', () => {
    const absent = choicesHtml({ text: 'Ready' });
    assert.match(absent, /Whole account/);
    assert.doesNotMatch(absent, /None offered|Could not read|retry-choices/);
    const known = choicesHtml({ connect: rows });
    assert.match(known, /data-i18n-authored>example\/project</);
    assert.match(known, /data-choice="0"/);
    assert.match(known, /Whole account/);
    const empty = choicesHtml({ connect: [] });
    assert.match(empty, /None offered yet/);
    assert.match(empty, /data-connect-act="settings"/);
    assert.doesNotMatch(empty, /retry-choices/);
    for (const data of [null, [], { connect: null }, { connect: {} }, { connect: 'bad' }]) {
        const html = choicesHtml(data);
        assert.match(html, /Could not read the connector’s list/);
        assert.match(html, /retry-choices/);
        assert.match(html, /Whole account/);
        assert.doesNotMatch(html, /None offered yet/);
    }
});

test('malformed rows are dropped without inventing an account-wide wildcard', () => {
    const bad = [null, { ...rows[0], account_id: null }, { ...rows[0], account_id: '' }, { ...rows[0], label: 5 }, { ...rows[0], conversation_id: null }];
    assert.deepEqual(connectChoices({ connect: [...bad, ...rows] }), { state: 'known', rows: [{ accountId: 'writer', conversationId: 'repo.123', label: 'example/project' }] });
    assert.equal(connectChoices({ connect: bad }).state, 'unavailable');
    const connected = choicesHtml({ connect: rows }, { places: [place] });
    assert.match(strip(connected), /example\/project Connected Choose/);
    assert.match(connected, /<div class="harness-account-meta"><span class="ui-status" data-tone="ok">Connected<\/span><\/div>/, 'its state under its name');
    assert.match(connected, /data-connect-act="adopt" data-place-id="gh-1"/, 'a connected identity is chosen, never connected twice');
    assert.match(connected, /data-choice="whole"/, 'the existing instance does not consume Whole account');
    const broad = choicesHtml({ connect: rows }, { places: [{ ...place, accountId: '', conversationId: '' }] });
    assert.match(broad, /data-choice="0"/, 'a broad place does not consume an instance');
});

test('an unloaded connector explains the missing chooser and still connects Whole account', () => {
    const html = connectListHtml({ connectors: [github], selected: { ...github, source: null, readiness: { text: 'Off in Skills', tone: 'warn' } } });
    assert.match(html, /Turn the connector on to choose where/);
    assert.match(html, /Off in Skills/);
    assert.match(html, /data-choice="whole"/);
    assert.doesNotMatch(html, /repository|channel|Reading where/);
});

test('the declared route uses the extension helper and no-store; unrelated routes never get fetched', async (t) => {
    const { calls, routes } = setup(t);
    routes.set('/api/extensions/github/status', (_url, init) => {
        assert.equal(init.cache, 'no-store');
        return { ok: true, json: async () => ({ text: 'Ready', connect: rows }) };
    });
    assert.deepEqual((await readPlaceConnector(github)).connect, rows);
    for (const url of ['https://other/status', '/api/extensions/another/status', '/api/extensions/github/../status']) {
        await assert.rejects(readPlaceConnector({ ...github, source: { url } }), /route could not be read/);
    }
    assert.equal(calls.length, 1);
});

test('a concrete Place asks for its own account identity and readiness; Connect retains the account list', async (t) => {
    const { routes, calls } = setup(t);
    for (const account of ['first+tag@example.test', 'second']) {
        routes.set(`/api/extensions/github/status?account_id=${encodeURIComponent(account)}`, () => ({ ok: true,
            json: async () => ({ account_id: account, text: `Ready ${account}`, identity: { sends_as: { id: account, kind: 'user', label: account } } }) }));
        const status = await readPlaceConnector({ ...place, accountId: account });
        assert.equal(status.identity.sends_as.id, account);
        assert.equal(status.text, `Ready ${account}`);
    }
    routes.set('/api/extensions/github/status?account_id=legacy', () => ({ ok: true, json: async () => ({ text: 'Legacy ready' }) }));
    assert.equal((await readPlaceConnector({ ...place, accountId: 'legacy' })).text, 'Legacy ready');
    routes.set('/api/extensions/github/status?account_id=unknown', () => ({ ok: true, json: async () => ({ account_id: 'neighbor' }) }));
    await assert.rejects(readPlaceConnector({ ...place, accountId: 'unknown' }), /different account/);
    assert.equal(calls.length, 4);
});

test('a typed account_mismatch passes through only for the exact telegram-bot + owner_bot pair', async (t) => {
    const { routes } = setup(t);
    const body = (reason) => ({ ok: true, json: async () => ({ account_id: 'other', reason }) });
    const owner = { connector: 'telegram-bot', contributionId: 'owner_bot', accountId: '101',
        source: { url: '/api/extensions/telegram-bot/owner_bot_status' } };
    routes.set('/api/extensions/telegram-bot/owner_bot_status?account_id=101', () => body('account_mismatch'));
    assert.equal((await readPlaceConnector(owner)).reason, 'account_mismatch');
    routes.set('/api/extensions/telegram-bot/owner_bot_status?account_id=101', () => body('identity_unconfirmed'));
    assert.equal((await readPlaceConnector(owner)).reason, 'identity_unconfirmed');
    routes.set('/api/extensions/telegram-bot/owner_bot_status?account_id=101', () => body('feed_off'));
    await assert.rejects(readPlaceConnector(owner), /different account/);
    // The same word under another connector, or the same connector's ordinary place, stays strict.
    routes.set('/api/extensions/github/status?account_id=101', () => body('account_mismatch'));
    await assert.rejects(readPlaceConnector({ ...github, contributionId: 'owner_bot', accountId: '101' }), /different account/);
    routes.set('/api/extensions/telegram-bot/status?account_id=101', () => body('account_mismatch'));
    await assert.rejects(readPlaceConnector({ ...owner, contributionId: 'place', source: { url: '/api/extensions/telegram-bot/status' } }), /different account/);
});

test('an unreadable Places response is an error; a known empty projection is empty', async (t) => {
    const { routes } = setup(t);
    routes.set('/api/places', { ok: true, json: async () => ({ error: 'not JSON' }) });
    await assert.rejects(readPlaces(), /Places could not be read/);
    routes.set('/api/places', { ok: true, json: async () => ({ places: [], connectors: [] }) });
    assert.deepEqual((await readPlaces()).places, []);
});

test('a failed status body never looks like an absent chooser, while an empty status object may omit it', async (t) => {
    const { routes } = setup(t);
    for (const data of [null, [], { error: 'not JSON' }, { ok: false, error: 'offline' }]) {
        routes.set('/api/extensions/github/status', { ok: true, json: async () => data });
        await assert.rejects(readPlaceConnector(github));
    }
    routes.set('/api/extensions/github/status', { ok: true, json: async () => ({}) });
    assert.equal(connectChoices(await readPlaceConnector(github)).state, 'absent');
});

test('the empty-list settings door opens the declared section; without a declaration it opens Skills', async (t) => {
    const { document, window } = setup(t);
    const host = document.createElement('div'); host.id = 'extension-settings-sections';
    host.innerHTML = '<form data-extension-settings-form data-skill="github" data-extension-settings-key="github:other:0"></form><form data-extension-settings-form data-skill="github" data-extension-settings-key="github:connection:0"></form>';
    document.body.appendChild(host);
    const forms = host.querySelectorAll('[data-extension-settings-form]');
    const scrolled = [], tabs = [], skills = [];
    forms.forEach((form, index) => { form.scrollIntoView = () => scrolled.push(index); });
    window.addEventListener('ouro:open-skills', (event) => skills.push(event.detail.skill));
    await openConnectorSettings(github, async (path) => tabs.push(path));
    assert.deepEqual(tabs, ['skill/github']);
    assert.deepEqual(scrolled, [1]);
    await openConnectorSettings({ ...github, settings: '' }, async (path) => tabs.push(path));
    assert.deepEqual(tabs, ['skill/github']);
    assert.deepEqual(skills, ['github']);
});

test('connecting an instance carries its identity and connector-authored label; an explicitly named contribution is never dropped, an unambiguous connect names none', async (t) => {
    const { routes } = setup(t);
    const calls = [];
    routes.set('/api/places', (_url, init) => {
        calls.push(JSON.parse(init.body));
        return { ok: true, json: async () => ({ ok: true, place: { ...raw, place_id: 'gh-1', state: 'silent' } }) };
    });
    assert.equal((await connectPlace({ connector: 'github', ...connectChoices({ connect: rows }).rows[0] })).state, 'silent');
    // The named contribution is the disambiguation when a connector declares several; connectNow omits it when only one exists.
    await connectPlace({ connector: 'github', contributionId: 'place' });
    await connectPlace({ connector: 'telegram-bot', contributionId: 'owner_bot', accountId: '101', conversationId: '-10042', label: 'Book club (-10042)' });
    assert.deepEqual(calls, [
        { connector: 'github', account_id: 'writer', conversation_id: 'repo.123', label: 'example/project' },
        { connector: 'github', contribution_id: 'place' },
        { connector: 'telegram-bot', contribution_id: 'owner_bot', account_id: '101', conversation_id: '-10042', label: 'Book club (-10042)' },
    ]);
});

const typeWords = async (sheet, words, description = '') => {
    const field = sheet.body.querySelector('[data-connect-draft="words"]');
    field.value = words; await sheet.body.fire('input', { target: field });
    if (description) {
        const line = sheet.body.querySelector('[data-connect-draft="description"]');
        line.value = description; await sheet.body.fire('input', { target: line });
    }
};

test('Add starts with the words: what the place is and what Ouroboros is for; the source comes after', async (t) => {
    const { document } = setup(t);
    let reads = 0;
    const sheet = openConnectPlace({ circle: { id: 'team', name: 'Team' }, read: async () => { reads += 1; return { connectors: [github], places: [] }; } });
    t.after(() => sheet.close());
    assert.match(sheet.body.textContent, /What is this place, and what am I here for\?/);
    assert.ok([...sheet.body.querySelectorAll('[data-i18n-authored]')].some((node) => node.textContent === 'Team'), 'the circle named, authored');
    assert.match(sheet.body.textContent, /’s words apply there too\./);
    assert.equal(sheet.body.querySelector('[data-connect-draft="description"]').getAttribute('value'), '', 'the description starts empty');
    assert.equal(reads, 0, 'nothing is read before the words');
    const click = async (selector) => sheet.body.fire('click', { target: sheet.body.querySelector(selector) });
    await click('[data-connect-act="next"]');
    assert.match(sheet.body.textContent, /Say in a few words what this place is/);
    await typeWords(sheet, 'The team room: answer when asked.');
    await click('[data-connect-act="next"]');
    await new Promise((resolve) => setImmediate(resolve));
    assert.equal(reads, 1);
    assert.match(sheet.body.textContent, /Where is it\?/);
    await click('[data-connect-act="back-words"]');
    assert.equal(sheet.body.querySelector('[data-connect-draft="words"]').textContent, 'The team room: answer when asked.', 'Back keeps the words');
    assert.ok(document.body.querySelector('dialog'));
});

test('the live Add flow fetches choices lazily, retries a failure, and lands the new place with its words in one write', async (t) => {
    const { document } = setup(t);
    let reads = 0;
    const writes = [], opened = [];
    const sheet = openConnectPlace({ circle: { id: 'team', name: 'Team' }, read: async () => ({ connectors: [github], places: [] }),
        readStatus: async () => { if (++reads === 1) throw new Error('offline'); return { text: 'Ready for you', connect: rows }; },
        connect: async (request) => { writes.push(request); return place; }, openPlace: (value) => opened.push(value) });
    t.after(() => sheet.close());
    const click = async (selector) => sheet.body.fire('click', { target: sheet.body.querySelector(selector) });
    await typeWords(sheet, 'Our project repository.', 'Where the code lives');
    await click('[data-connect-act="next"]');
    await new Promise((resolve) => setImmediate(resolve));
    assert.equal(reads, 0);
    await click('[data-connect-act="choose"]');
    assert.equal(reads, 1);
    assert.match(sheet.body.textContent, /Could not read the connector’s list/);
    assert.ok(sheet.body.querySelector('[data-choice="whole"]'));
    await click('[data-connect-act="retry-choices"]');
    assert.equal(reads, 2);
    assert.equal(sheet.body.querySelector('[data-i18n-authored][role="status"]').textContent, 'Ready for you');
    await click('[data-choice="0"]');
    assert.deepEqual(writes, [{ connector: 'github', accountId: 'writer', conversationId: 'repo.123', label: 'example/project',
        note: 'Our project repository.', description: 'Where the code lives', circleId: 'team' }]);
    assert.deepEqual(opened, [{ ...place, serviceLabel: 'GitHub' }], 'the sheet opens with the declared service name');
    assert.equal(document.body.querySelector('dialog'), null);
});

test('Where only when a choice is needed: a ready connector without instances connects its whole account at once', async (t) => {
    setup(t);
    const writes = [];
    const sheet = openConnectPlace({ read: async () => ({ connectors: [github], places: [] }), readStatus: async () => ({ text: 'Ready' }),
        connect: async (request) => { writes.push(request); return place; }, openPlace: () => {} });
    t.after(() => sheet.close());
    await typeWords(sheet, 'Words.');
    await sheet.body.fire('click', { target: sheet.body.querySelector('[data-connect-act="next"]') });
    await new Promise((resolve) => setImmediate(resolve));
    await sheet.body.fire('click', { target: sheet.body.querySelector('[data-connect-act="choose"]') });
    assert.deepEqual(writes, [{ connector: 'github', note: 'Words.', description: '', circleId: '' }], 'no circle: Not in a circle');
});

test('choosing a connected place shows its own notes and saves only the ticked changes under its revision', async (t) => {
    setup(t);
    const updates = [], opened = [];
    const existing = { ...place, notes: [{ id: 'n1', text: 'An older owner note', author: 'owner', at: '' },
        { id: 'n2', text: 'My record', author: 'ouroboros', at: '' }], description: { text: 'Old line', author: 'owner', at: '' } };
    const sheet = openConnectPlace({ circle: { id: 'team', name: 'Team' }, read: async () => ({ connectors: [github], places: [existing] }),
        readStatus: async () => ({ text: 'Ready', connect: rows }), update: async (id, change) => { updates.push({ id, change }); return existing; },
        openPlace: (value) => opened.push(value) });
    t.after(() => sheet.close());
    const click = async (selector) => sheet.body.fire('click', { target: sheet.body.querySelector(selector) });
    await typeWords(sheet, 'The project room.', 'Where the code lives');
    await click('[data-connect-act="next"]');
    await new Promise((resolve) => setImmediate(resolve));
    await click('[data-connect-act="choose"]');
    await click('[data-connect-act="adopt"]');
    assert.match(sheet.body.textContent, /An older owner note/);
    assert.match(sheet.body.textContent, /My record/);
    assert.match(sheet.body.textContent, /its state, conversations and notes stay as they are/);
    for (const key of ['circle', 'note', 'description']) assert.ok(sheet.body.querySelector(`[data-connect-adopt="${key}"]`).hasAttribute('checked'), key);
    const note = sheet.body.querySelector('[data-connect-adopt="note"]');
    note.checked = false; await sheet.body.fire('change', { target: note });
    await click('[data-connect-act="adopt-save"]');
    assert.deepEqual(updates, [{ id: 'gh-1', change: { circle_id: 'team', description: 'Where the code lives', expected_revision: 7 } }]);
    assert.equal(opened.length, 1);
});

test('a place connected meanwhile is never written twice: the Add shows it and asks what to change', async (t) => {
    setup(t);
    const connects = [], updates = [];
    const conflict = Object.assign(new Error('place_exists'), { status: 409, body: { ok: false, code: 'place_exists',
        place: { ...raw, place_id: 'gh-1', label: 'GitHub · example/project', account_id: 'writer', conversation_id: 'repo.123',
            state: 'active', revision: 12, notes: [{ note_id: 'x', text: 'The same words', author: 'owner', at: '' }] } } });
    const sheet = openConnectPlace({ circle: { id: 'team', name: 'Team' }, read: async () => ({ connectors: [github], places: [] }),
        readStatus: async () => ({ text: 'Ready', connect: rows }), connect: async (request) => { connects.push(request); throw conflict; },
        update: async (id, change) => { updates.push({ id, change }); return place; }, openPlace: () => {} });
    t.after(() => sheet.close());
    const click = async (selector) => sheet.body.fire('click', { target: sheet.body.querySelector(selector) });
    await typeWords(sheet, 'The same words');
    await click('[data-connect-act="next"]');
    await new Promise((resolve) => setImmediate(resolve));
    await click('[data-connect-act="choose"]');
    await click('[data-choice="0"]');
    assert.equal(connects.length, 1);
    assert.match(sheet.body.textContent, /It is connected already: choose what to change\./);
    assert.equal(sheet.body.querySelector('[data-connect-adopt="note"]'), null, 'words it holds already are not offered again');
    assert.ok(sheet.body.querySelector('[data-connect-adopt="circle"]'));
    await click('[data-connect-act="adopt-save"]');
    assert.deepEqual(updates, [{ id: 'gh-1', change: { circle_id: 'team', expected_revision: 12 } }]);
});

test('an enabled connector without an optional status source still connects from Add', async (t) => {
    setup(t);
    const connector = { ...github, source: null };
    const writes = [];
    const sheet = openConnectPlace({ read: async () => ({ connectors: [connector], places: [] }),
        readStatus: async () => { throw new Error('a connector without a source must not request status'); },
        connect: async (request) => { writes.push(request); return place; }, openPlace: () => {} });
    t.after(() => sheet.close());
    await typeWords(sheet, 'A place without a status route.');
    await sheet.body.fire('click', { target: sheet.body.querySelector('[data-connect-act="next"]') });
    const choose = sheet.body.querySelector('[data-connect-act="choose"]');
    assert.ok(choose, 'source is optional for provides: place, not evidence that the connector is off');
    await sheet.body.fire('click', { target: choose });
    assert.deepEqual(writes, [{ connector: 'github', note: 'A place without a status route.', description: '', circleId: '' }]);
});

test('Back during a connector status read ignores a late answer or failure without writing', async (t) => {
    for (const fails of [false, true]) await t.test(fails ? 'rejected read' : 'successful read', async (sub) => {
        setup(sub);
        let resolve, reject;
        const status = new Promise((yes, no) => { resolve = yes; reject = no; });
        const writes = [];
        const sheet = openConnectPlace({ read: async () => ({ connectors: [github], places: [place] }),
            readStatus: () => status, connect: async (request) => { writes.push(request); return place; }, openPlace: () => {} });
        sub.after(() => sheet.close());
        const click = (selector) => sheet.body.fire('click', { target: sheet.body.querySelector(selector) });
        await typeWords(sheet, 'Keep this draft.');
        await click('[data-connect-act="next"]');
        const reading = click('[data-connect-act="choose"]');
        await click('[data-connect-act="back"]');
        const completion = assert.doesNotReject(reading);
        if (fails) reject(new Error('disconnected')); else resolve({ text: 'Ready' });
        await completion;
        assert.deepEqual(writes, [], 'Back cancels the pending choice, never creates a broad connection');
        assert.match(sheet.body.textContent, /Where is it\?/);
        await click('[data-connect-act="back-words"]');
        assert.equal(sheet.body.querySelector('[data-connect-draft="words"]').textContent, 'Keep this draft.');
    });
});

test('«Choose where» on a connector with places shows Where and its explicit Whole account; it never connects by itself', async (t) => {
    for (const [name, connector] of [['with a status route', github], ['without a status route', { ...github, source: null }]]) {
        await t.test(name, async (sub) => {
            setup(sub);
            const writes = [];
            const sheet = openConnectPlace({ read: async () => ({ connectors: [connector], places: [place] }),
                readStatus: async () => ({ text: 'Ready' }), connect: async (request) => { writes.push(request); return { ...place, id: 'gh-whole' }; },
                openPlace: () => {} });
            sub.after(() => sheet.close());
            const click = (selector) => sheet.body.fire('click', { target: sheet.body.querySelector(selector) });
            await typeWords(sheet, 'The whole account.');
            await click('[data-connect-act="next"]');
            await new Promise((resolve) => setImmediate(resolve));
            assert.equal(sheet.body.querySelector('[data-connect-act="choose"]').textContent, 'Choose where');
            await click('[data-connect-act="choose"]');
            assert.deepEqual(writes, [], 'the press says where; nothing broad is written by it');
            assert.match(sheet.body.textContent, /Where · GitHub/);
            assert.doesNotMatch(sheet.body.textContent, /Turn the connector on/, 'the connector is on');
            await click('[data-connect-act="connect"][data-choice="whole"]');
            assert.deepEqual(writes, [{ connector: 'github', note: 'The whole account.', description: '', circleId: '' }],
                'Whole account is the owner\'s explicit choice, with the words as its first note');
        });
    }
});

test('a connector that is on but has no status route never says to turn it on; its whole account is chosen, not connected twice', async (t) => {
    setup(t);
    const whole = { ...place, id: 'gh-whole', accountId: '', conversationId: '', label: 'GitHub' };
    const writes = [];
    const sheet = openConnectPlace({ read: async () => ({ connectors: [{ ...github, source: null }], places: [whole] }),
        readStatus: async () => { throw new Error('a connector without a source must not request status'); },
        connect: async (request) => { writes.push(request); return whole; }, openPlace: () => {} });
    t.after(() => sheet.close());
    const click = (selector) => sheet.body.fire('click', { target: sheet.body.querySelector(selector) });
    await typeWords(sheet, 'Words.');
    await click('[data-connect-act="next"]');
    await new Promise((resolve) => setImmediate(resolve));
    await click('[data-connect-act="choose"]');
    assert.doesNotMatch(sheet.body.textContent, /Turn the connector on/);
    assert.ok(sheet.body.querySelector('[data-connect-act="adopt"][data-place-id="gh-whole"]'), 'its whole account: Connected, with Choose');
    assert.deepEqual(writes, []);
});

const flushAsync = async (turns = 6) => {
    for (let i = 0; i < turns; i += 1) await new Promise((resolve) => setImmediate(resolve));
};

const rawTgPlaceConnector = {
    connector: 'telegram-bot', contribution_id: 'place', connect_mode: 'any', scope: 'conversation_or_account',
    label: 'Telegram', destination: 'vendor:Telegram',
    source: { url: '/api/extensions/telegram-bot/status', refresh_s: 15 }, settings: 'transport',
    readiness: { text: 'Connector on', tone: 'ok' }, wakes: ['mention', 'direct_message', 'reply_to_me'],
};

const rawTgOwnerBotConnector = {
    connector: 'telegram-bot', contribution_id: 'owner_bot', connect_mode: 'exact', scope: 'conversation',
    label: 'Telegram', destination: 'vendor:Telegram',
    source: { url: '/api/extensions/telegram-bot/owner_bot_status', refresh_s: 15 }, settings: '',
    readiness: { text: 'Connector on', tone: 'ok' }, wakes: ['mention', 'reply_to_me'],
};

test('connect_mode: exact never offers Whole account or auto-connects on loading, missing source, errors, empty, malformed or success', async (t) => {
    const exactConnector = normalizeConnector({ ...raw, connector: 'matrix_rooms', label: 'Matrix Rooms', connect_mode: 'exact' });
    assert.equal(exactConnector.connectMode, 'exact');
    assert.equal(normalizeConnector({ ...raw, connect_mode: 'any' }).connectMode, 'any');
    assert.equal(github.connectMode, '');

    for (const view of [
        { connectors: [exactConnector], selected: exactConnector, choices: null },
        { connectors: [exactConnector], selected: { ...exactConnector, source: null }, choices: connectChoices({}) },
        { connectors: [exactConnector], selected: { ...exactConnector, source: null, readiness: { text: 'Off in Skills', tone: 'warn' } }, choices: null },
        { connectors: [exactConnector], selected: exactConnector, choices: connectChoices({ text: 'Ready' }) },
        { connectors: [exactConnector], selected: exactConnector, choices: connectChoices({ connect: [] }) },
        { connectors: [exactConnector], selected: exactConnector, choices: connectChoices({ connect: null }) },
        { connectors: [exactConnector], selected: exactConnector, choices: connectChoices({ connect: [{ bad: true }] }) },
        { connectors: [exactConnector], selected: exactConnector, choices: connectChoices({ connect: [{ account_id: '777', conversation_id: '-1001', label: 'Ops' }] }) },
    ]) {
        const html = connectListHtml(view);
        assert.doesNotMatch(html, /Whole account|data-choice="whole"/);
    }

    for (const [label, connector, statusReply] of [
        ['absent connect', exactConnector, async () => ({ text: 'Ready' })],
        ['no source', { ...exactConnector, source: null }, async () => ({})],
        ['status error', exactConnector, async () => { throw new Error('boom'); }],
        ['empty connect', exactConnector, async () => ({ connect: [] })],
    ]) {
        await t.test(label, async (sub) => {
            setup(sub);
            const writes = [];
            const sheet = openConnectPlace({
                read: async () => ({ connectors: [connector], places: [] }),
                readStatus: statusReply,
                connect: async (req) => { writes.push(req); return place; },
                openPlace: () => {},
            });
            sub.after(() => sheet.close());
            await typeWords(sheet, 'Exact room rules.');
            await sheet.body.fire('click', { target: sheet.body.querySelector('[data-connect-act="next"]') });
            await flushAsync();
            await sheet.body.fire('click', { target: sheet.body.querySelector('[data-connect-act="choose"]') });
            await flushAsync();
            assert.deepEqual(writes, []);
            assert.equal(sheet.body.querySelector('[data-choice="whole"]'), null);
        });
    }
});

test('Core v2 + Hub wire parsers enforce contribution keying, startgroup invite validation, owner_bound, account match, and exact negative group IDs', () => {
    const ownerConn = normalizeConnector(rawTgOwnerBotConnector);
    const placeConn = normalizeConnector(rawTgPlaceConnector);
    const legacyConn = normalizeConnector({ ...rawTgPlaceConnector, contribution_id: undefined });
    const ownerPlace = normalizePlace({ place_id: 'p-a', connector: 'telegram-bot', contribution_id: 'owner_bot', account_id: '101', conversation_id: '-10042', label: 'Club', state: 'silent', revision: 1 });
    const sepPlace = normalizePlace({ place_id: 'p-b', connector: 'telegram-bot', contribution_id: 'place', account_id: '555', conversation_id: '-10099', label: 'Ops', state: 'silent', revision: 1 });
    const legacySepPlace = normalizePlace({ place_id: 'p-legacy', connector: 'telegram-bot', account_id: '555', conversation_id: '-10099', label: 'Ops Legacy', state: 'silent', revision: 1 });

    assert.equal(connectorKey(ownerConn), 'telegram-bot:owner_bot');
    assert.equal(connectorKey(placeConn), 'telegram-bot:place');
    assert.equal(connectorKey(legacyConn), 'telegram-bot');
    assert.equal(connectorMatchesPlace(ownerConn, ownerPlace), true);
    assert.equal(connectorMatchesPlace(ownerConn, sepPlace), false);
    assert.equal(connectorMatchesPlace(ownerConn, legacySepPlace), false, 'legacy B place never collapses into A');
    assert.equal(connectorMatchesPlace(placeConn, sepPlace), true);
    assert.equal(connectorMatchesPlace(placeConn, legacySepPlace), true, 'legacy B place matches new B projection');
    assert.equal(connectorMatchesPlace(legacyConn, legacySepPlace), true);
    assert.equal(connectorMatchesPlace(legacyConn, ownerPlace), false, 'saved A place never rebinds to B on old core');

    // `GET /api/extensions/telegram/group_feed`
    const readyFeed = parseTelegramGroupFeed({
        v: 1, state: 'ready', account_id: '101',
        identity: { sends_as: { id: '101', label: '@mybot', kind: 'bot' } },
        owner_bound: true, invite_url: 'https://t.me/mybot?startgroup=true',
        enabled: false, capable: true, pending: 0, write_gaps: 0, last_write_gap_at: '',
    });
    assert.deepEqual(readyFeed, {
        state: 'ready', accountId: '101',
        bot: { id: '101', label: '@mybot', username: 'mybot' },
        ownerBound: true, enabled: false, capable: true, separateHeld: false,
        inviteUrl: 'https://t.me/mybot?startgroup=true',
    });
    assert.equal(parseTelegramGroupFeed({ ...readyFeed, v: 1, state: 'ready', account_id: '101', identity: { sends_as: { id: '101', label: '@mybot', kind: 'bot' } }, owner_bound: false, enabled: false, capable: true }).state, 'unpaired');
    assert.equal(parseTelegramGroupFeed({ v: 1, state: 'ready', account_id: '101', identity: { sends_as: { id: '999', label: '@mybot', kind: 'bot' } }, owner_bound: true, enabled: false, capable: true }).state, 'identity_unconfirmed');
    assert.equal(parseTelegramGroupFeed({ v: 1, state: 'ready', account_id: '101', identity: { sends_as: { id: '101', label: 'Display Name Only', kind: 'bot' } }, owner_bound: true, enabled: false, capable: true }).state, 'identity_unconfirmed');
    assert.equal(parseTelegramGroupFeed({ v: 1, state: 'no_token', owner_bound: false, enabled: false, capable: false }).state, 'no_token');
    assert.equal(parseTelegramGroupFeed({ v: 1, state: 'bridge_not_ready', owner_bound: true, enabled: false, capable: true }).state, 'bridge_not_ready');
    assert.equal(parseTelegramGroupFeed({ v: 2, state: 'ready' }).state, 'unavailable');

    // Strict `startgroup` invite URL validation
    assert.equal(validatedTelegramInviteUrl('https://t.me/mybot?startgroup=true', '@mybot'), 'https://t.me/mybot?startgroup=true');
    assert.equal(validatedTelegramInviteUrl('https://t.me/MyBot?startgroup=ouroboros', '@mybot'), 'https://t.me/MyBot?startgroup=ouroboros');
    for (const bad of [
        '', null, 'https://t.me/mybot', 'https://t.me/mybot?startgroup=',
        'https://t.me/mybot?startgroup=true&extra=1', 'http://t.me/mybot?startgroup=true',
        'https://evil.example/mybot?startgroup=true', 'https://t.me/otherbot?startgroup=true',
        'https://user:pass@t.me/mybot?startgroup=true', 'https://t.me:8443/mybot?startgroup=true',
        'https://t.me/mybot/sub?startgroup=true',
    ]) {
        assert.equal(validatedTelegramInviteUrl(bad, '@mybot'), '', `must reject ${bad}`);
    }

    // `GET /api/extensions/telegram-bot/owner_bot_status`
    const okOwner = parseOwnerBotStatus({
        v: 1, enabled: true, reason: '', account_id: '101',
        identity: { sends_as: { id: '101', label: '@mybot', kind: 'bot' } },
        invite_url: 'https://t.me/mybot?startgroup=true',
        connect: [{ account_id: '101', conversation_id: '-10042', label: 'Book club (-10042)' }],
    }, '101');
    assert.equal(okOwner.state, 'known');
    assert.deepEqual(okOwner.choices.rows, [{ accountId: '101', conversationId: '-10042', label: 'Book club (-10042)' }]);
    assert.equal(parseOwnerBotStatus({ v: 1, enabled: false, reason: 'feed_off', connect: null }, '101').state, 'feed_off');
    assert.equal(parseOwnerBotStatus({ v: 1, enabled: null, reason: 'update_required', connect: null }, '101').reason, 'update_required');
    assert.equal(parseOwnerBotStatus({
        v: 1, enabled: true, reason: '', account_id: '202',
        identity: { sends_as: { id: '202', label: '@otherbot', kind: 'bot' } },
        connect: [{ account_id: '202', conversation_id: '-10042', label: 'Stale' }],
    }, '101').state, 'account_mismatch', 'stale status with wrong account never yields Connect');
    assert.equal(parseOwnerBotStatus({
        v: 1, enabled: true, reason: '', account_id: '101',
        identity: { sends_as: { id: '101', label: '@mybot', kind: 'bot' } },
        connect: [{ account_id: '999', conversation_id: '-10042', label: 'Foreign' }],
    }, '101').state, 'unavailable', 'foreign row in owner_bot_status fails closed');
    assert.equal(telegramGroupChoices({ connect: [{ account_id: '101', conversation_id: '*', label: 'Wildcard' }] }, '101').state, 'unavailable');

    // `GET /api/extensions/telegram-bot/status` (Branch B): authoritative `link` + `Bot token missing` + unknown vs `[]`
    assert.equal(parseSeparateBotStatus({ text: 'Not set up: the bot token is missing', tone: 'warn', link: 'waiting_for_token', connect: null }).state, 'unconfigured');
    assert.equal(parseSeparateBotStatus({ text: 'Not set up', tone: 'warn', connect: null, connect_unavailable: 'Bot token missing' }).state, 'unconfigured');
    assert.equal(parseSeparateBotStatus({ text: 'Connecting…', tone: 'muted', link: 'connecting', connect: null }).state, 'connecting');
    assert.equal(parseSeparateBotStatus({ text: 'Connecting…', tone: 'warn', bot_unconfirmed: true, identity: { sends_as: { id: '555', label: '@old_bot', kind: 'bot' } }, connect: null }).state, 'connecting');
    assert.equal(parseSeparateBotStatus({ text: 'Connection error', tone: 'error', link: 'provider_error', connect: null }).state, 'unavailable', 'provider error must not claim missing token');
    assert.equal(parseSeparateBotStatus({ text: 'Something else warn', tone: 'warn' }).state, 'unavailable', 'generic warn without typed missing-token link must not tell owner to create a bot');
    assert.equal(parseSeparateBotStatus({
        text: 'Connected', tone: 'ok', link: 'connected',
        identity: { sends_as: { id: '555', label: '@team_bot', kind: 'bot' } },
        connect: null,
    }).state, 'unavailable', 'connect: null is unknown, never []');
    assert.equal(parseSeparateBotStatus({
        text: 'Connected', tone: 'ok',
        identity: { sends_as: { id: '555', label: '@team_bot', kind: 'bot' } },
    }).state, 'unavailable', 'omitted connect is unknown, never []');
    const dmOnlyB = parseSeparateBotStatus({
        text: 'Ready', tone: 'ok', link: 'connected',
        identity: { sends_as: { id: '555', label: '@team_bot', kind: 'bot' } },
        connect: [{ account_id: '555', conversation_id: '77777', label: 'Private DM' }],
    });
    assert.equal(dmOnlyB.state, 'empty', 'positive private chat IDs are ignored in the group chooser');
    assert.equal(confirmedTelegramBot({ sends_as: { id: '555', label: 'team_bot', kind: 'bot' } })?.label, '@team_bot');
});

test('Branch A end-to-end: passive discovery, 409 account_changed without auto-retry, explicit opt-in, empty invite, observed group connect with owner_bot, and idempotent reopen', async (t) => {
    const { calls, routes, document } = setup(t);
    let currentAccount = '101';
    let currentHandle = '@mybot';
    let feedEnabled = false;
    let groupObserved = false;
    let trigger409Once = true;
    const optInPosts = [], placePosts = [], openedPlaces = [];

    routes.set('/api/places', (_url, init) => {
        if (init?.method === 'POST') {
            const body = JSON.parse(init.body);
            placePosts.push(body);
            return {
                ok: true,
                json: async () => ({
                    ok: true,
                    place: {
                        ...rawTgOwnerBotConnector,
                        place_id: 'tg-owner-1',
                        label: `Telegram · ${body.label}`,
                        provider: 'telegram',
                        account_id: body.account_id,
                        conversation_id: body.conversation_id,
                        state: 'silent',
                        revision: 1,
                    },
                }),
            };
        }
        return {
            ok: true,
            json: async () => ({
                places: [],
                circles: [{ circle_id: 'books', name: 'Books', text: 'Reading group', text_author: 'owner', text_at: '', revision: 1 }],
                connectors: [rawTgPlaceConnector, rawTgOwnerBotConnector],
            }),
        };
    });
    routes.set('/api/extensions', () => ({
        ok: true,
        json: async () => ({
            skills: [
                { name: 'telegram', type: 'extension', enabled: true },
                { name: 'telegram-bot', type: 'extension', enabled: true, live_loaded: true, live_reason: 'ready' },
            ],
        }),
    }));
    routes.set('/api/extensions/telegram/group_feed', (_url, init) => {
        if (init?.method === 'POST') {
            const body = JSON.parse(init.body);
            optInPosts.push(body);
            if (trigger409Once) {
                trigger409Once = false;
                currentAccount = '102';
                currentHandle = '@mybot_v2';
                return { status: 409, ok: false, json: async () => ({ ok: false, code: 'account_changed', error: 'Paired bot changed.' }) };
            }
            assert.equal(body.expected_account_id, currentAccount);
            assert.equal(body.enabled, true);
            feedEnabled = true;
            return {
                ok: true,
                json: async () => ({
                    v: 1, ok: true, state: 'ready', account_id: currentAccount,
                    identity: { sends_as: { id: currentAccount, label: currentHandle, kind: 'bot' } },
                    owner_bound: true, invite_url: `https://t.me/${currentHandle.slice(1)}?startgroup=true`,
                    enabled: true, capable: true, pending: 0, write_gaps: 0, last_write_gap_at: '',
                }),
            };
        }
        return {
            ok: true,
            json: async () => ({
                v: 1, state: 'ready', account_id: currentAccount,
                identity: { sends_as: { id: currentAccount, label: currentHandle, kind: 'bot' } },
                owner_bound: true, invite_url: `https://t.me/${currentHandle.slice(1)}?startgroup=true`,
                enabled: feedEnabled, capable: true, pending: 0, write_gaps: 0, last_write_gap_at: '',
            }),
        };
    });
    routes.set('/api/extensions/telegram-bot/owner_bot_status?account_id=102', () => ({
        ok: true,
        json: async () => ({
            v: 1, enabled: feedEnabled, reason: feedEnabled ? '' : 'feed_off',
            account_id: currentAccount,
            identity: { sends_as: { id: currentAccount, label: currentHandle, kind: 'bot' } },
            invite_url: `https://t.me/${currentHandle.slice(1)}?startgroup=true`,
            connect: feedEnabled
                ? (groupObserved ? [{ account_id: currentAccount, conversation_id: '-10042', label: 'Book club (-10042)' }] : [])
                : null,
        }),
    }));

    const sheet = openConnectPlace({
        circle: { id: 'books', name: 'Books' },
        openPlace: (placeValue, opts) => openedPlaces.push([placeValue, opts]),
    });
    t.after(() => sheet.close());
    const click = async (selector) => sheet.body.fire('click', { target: sheet.body.querySelector(selector) });

    await typeWords(sheet, 'Book club: answer only when asked.', 'Monthly reading');
    await click('[data-connect-act="next"]');
    await flushAsync();

    // One Telegram Groups entry; both status routes are lazy.
    assert.equal(calls.some((u) => u.includes('group_feed') || u.includes('owner_bot_status') || u.includes('telegram_groups')), false);
    assert.equal(sheet.body.querySelectorAll('[data-connector="telegram-bot"]').length, 0, 'neither group route appears twice');
    assert.ok(sheet.body.querySelector('[data-connector-entry="telegram-groups"]'));

    await click('[data-connect-act="open-tg-groups"]');
    await flushAsync();
    assert.equal(optInPosts.length, 0, 'opening Telegram Groups never auto-POSTs opt-in');
    assert.equal(calls.some((u) => u.includes('owner_bot_status')), false, 'does not read owner_bot_status while group_feed.enabled is false');
    assert.match(sheet.body.textContent, /Group members see @mybot; your private chat stays private\./);
    assert.match(sheet.body.textContent, /Allow @mybot in groups/);

    // First explicit opt-in hits 409 `account_changed`: must NOT auto-retry; re-reads and shows `@mybot_v2`!
    await click('[data-connect-act="tg-opt-in"]');
    await flushAsync(8);
    assert.deepEqual(optInPosts, [{ enabled: true, expected_account_id: '101' }], 'no automatic second POST on 409');
    assert.match(sheet.body.textContent, /paired Telegram bot changed/i);
    assert.match(sheet.body.textContent, /Allow @mybot_v2 in groups/);

    // Second explicit opt-in confirms `@mybot_v2`, reads `owner_bot_status` (`connect: []`), and shows validated invite URL.
    await click('[data-connect-act="tg-opt-in"]');
    await flushAsync(8);
    assert.deepEqual(optInPosts, [
        { enabled: true, expected_account_id: '101' },
        { enabled: true, expected_account_id: '102' },
    ]);
    assert.equal(placePosts.length, 0, 'opt-in alone never connects a place');
    const invite = sheet.body.querySelector('[data-connect-invite="personal"]');
    assert.ok(invite);
    assert.equal(invite.getAttribute('href'), 'https://t.me/mybot_v2?startgroup=true');

    // Group message arrives -> Check again -> Connect sends `contribution_id: 'owner_bot'` and lands Silent!
    groupObserved = true;
    await click('[data-connect-act="tg-refresh"]');
    await flushAsync(8);
    assert.equal(optInPosts.length, 2, 'Check again when already enabled never re-POSTs opt-in');
    assert.match(sheet.body.textContent, /Book club \(-10042\)/);
    assert.doesNotMatch(sheet.body.textContent, /Whole account/);

    await click('[data-connect-act="connect"][data-choice="0"]');
    await flushAsync(6);
    assert.deepEqual(placePosts, [{
        contribution_id: 'owner_bot',
        connector: 'telegram-bot',
        account_id: '102',
        conversation_id: '-10042',
        label: 'Book club (-10042)',
        note: 'Book club: answer only when asked.',
        description: 'Monthly reading',
        circle_id: 'books',
    }]);
    assert.equal(openedPlaces.length, 1);
    assert.equal(openedPlaces[0][0].state, 'silent');
    assert.equal(openedPlaces[0][0].contributionId, 'owner_bot');
    assert.equal(openedPlaces[0][1].connectorStatus.identity.sends_as.label, '@mybot_v2');
    assert.equal(document.body.querySelector('dialog'), null);

    // Reopening Add -> Telegram Groups when `group_feed.enabled === true` makes zero opt-in POSTs!
    const sheet2 = openConnectPlace({ openPlace: () => {} });
    t.after(() => sheet2.close());
    await typeWords(sheet2, 'Second group.');
    await sheet2.body.fire('click', { target: sheet2.body.querySelector('[data-connect-act="next"]') });
    await flushAsync();
    await sheet2.body.fire('click', { target: sheet2.body.querySelector('[data-connect-act="open-tg-groups"]') });
    await flushAsync(8);
    assert.equal(optInPosts.length, 2, 'reopening when already enabled sends zero opt-in POSTs');
    assert.match(sheet2.body.textContent, /Book club \(-10042\)/);
});

test('Branch B works on both old and new connector projections, preserves ordinary Where row, and never sends contribution_id', async (t) => {
    for (const [modeName, connectorRaw] of [
        ['new projection (contribution_id: place)', rawTgPlaceConnector],
        ['old core projection (no contribution_id)', { ...rawTgPlaceConnector, contribution_id: undefined, connect_mode: undefined }],
    ]) {
        await t.test(modeName, async (sub) => {
            const { routes } = setup(sub);
            let bConfigured = false;
            let bGroupSeen = false;
            const placePosts = [], settingsOpened = [];

            routes.set('/api/places', (_url, init) => {
                if (init?.method === 'POST') {
                    const body = JSON.parse(init.body);
                    placePosts.push(body);
                    return {
                        ok: true,
                        json: async () => ({
                            ok: true,
                            place: { ...connectorRaw, place_id: 'tg-sep-1', provider: 'telegram', label: body.label, account_id: body.account_id, conversation_id: body.conversation_id, state: 'silent', revision: 1 },
                        }),
                    };
                }
                return { ok: true, json: async () => ({ places: [], circles: [], connectors: [connectorRaw] }) };
            });
            routes.set('/api/extensions', () => ({
                ok: true,
                json: async () => ({
                    skills: [
                        { name: 'telegram', enabled: true, live_loaded: true },
                        { name: 'telegram-bot', enabled: true, live_loaded: true },
                    ],
                }),
            }));
            routes.set('/api/extensions/telegram/group_feed', () => ({
                status: 404, ok: false, json: async () => ({ error: 'not found' }),
            }));
            routes.set('/api/extensions/telegram-bot/status', () => ({
                ok: true,
                json: async () => (bConfigured
                    ? {
                        text: 'Polling ok', tone: 'ok', link: 'connected',
                        identity: { sends_as: { id: '555', label: '@team_ops_bot', kind: 'bot' } },
                        connect: bGroupSeen
                            ? [
                                { account_id: '555', conversation_id: '7001', label: 'Alice DM' },
                                { account_id: '555', conversation_id: '-100888', label: 'Engineering (-100888)' },
                            ]
                            : [{ account_id: '555', conversation_id: '7001', label: 'Alice DM' }],
                    }
                    : { text: 'Not set up: the bot token is missing', tone: 'warn', link: 'waiting_for_token', connect: null }),
            }));

            const sheet = openConnectPlace({
                openSettings: async (section, opts) => { settingsOpened.push([section, opts]); return true; },
                openPlace: () => {},
            });
            sub.after(() => sheet.close());
            const click = async (selector) => sheet.body.fire('click', { target: sheet.body.querySelector(selector) });

            await typeWords(sheet, 'Engineering room.');
            await click('[data-connect-act="next"]');
            await flushAsync();

            // The B branch also preserves Whole account and DMs without a second group entry.
            bConfigured = true;
            await click('[data-connect-act="open-tg-groups"]');
            await click('[data-connect-act="tg-branch"][data-branch="separate"]');
            await click('[data-connect-act="tg-other-chats"]');
            await flushAsync();
            assert.ok(sheet.body.querySelector('[data-choice="whole"]'), 'ordinary B connector keeps Whole account');
            assert.match(sheet.body.textContent, /Alice DM/);
            await click('[data-connect-act="back"]');

            // A changed configuration does not choose another bot on the owner's behalf.
            bConfigured = false;
            await click('[data-connect-act="tg-refresh"]');
            await flushAsync(8);
            assert.match(sheet.body.textContent, /TELEGRAM_PUBLIC_BOT_TOKEN/);
            assert.match(sheet.body.textContent, /never reuse your personal bot token/i);

            await click('[data-connect-act="tg-branch"][data-branch="personal"]');
            await flushAsync();
            assert.match(sheet.body.textContent, /needs an Ouroboros update/i);
            assert.ok(sheet.body.querySelector('[data-connect-act="tg-open-updates"]'));

            // Configure B token -> empty group list gives manual `@team_ops_bot` instructions without inventing a t.me link!
            bConfigured = true;
            await click('[data-connect-act="tg-branch"][data-branch="separate"]');
            await flushAsync(6);
            assert.match(sheet.body.textContent, /add @team_ops_bot as a member/);
            assert.equal(sheet.body.querySelector('[data-connect-invite]'), null, 'no fabricated invite link when status omits invite_url');

            // Group message arrives -> Check again -> Connect sends legacy payload WITHOUT `contribution_id`!
            bGroupSeen = true;
            await click('[data-connect-act="tg-refresh"]');
            await flushAsync(8);
            assert.match(sheet.body.textContent, /Engineering \(-100888\)/);
            assert.doesNotMatch(sheet.body.textContent, /Alice DM|Whole account/);

            await click('[data-connect-act="connect"][data-choice="0"]');
            await flushAsync(6);
            assert.deepEqual(placePosts, [{
                connector: 'telegram-bot',
                account_id: '555',
                conversation_id: '-100888',
                label: 'Engineering (-100888)',
                note: 'Engineering room.',
            }]);
            assert.equal(Object.hasOwn(placePosts[0], 'contribution_id'), false);
        });
    }
});

test('negative, queued, disabled, grant-blocked, and partial-upgrade states route to Skills/Browse/Updates cleanly and cancel stale reads on Back', async (t) => {
    const { routes, window } = setup(t);
    const skillsOpened = [], dashboardsOpened = [];
    window.addEventListener('ouro:open-skills', (e) => skillsOpened.push(e.detail));
    window.addEventListener('ouro:open-dashboard', (e) => dashboardsOpened.push(e.detail));

    let scenario = 'absent_bot';
    routes.set('/api/places', () => ({
        ok: true,
        json: async () => ({
            places: [],
            circles: [],
            connectors: scenario === 'old_hub' ? [rawTgPlaceConnector] : [],
        }),
    }));
    routes.set('/api/extensions', () => ({
        ok: true,
        json: async () => {
            if (scenario === 'tg_disabled') return { skills: [{ name: 'telegram', enabled: false }] };
            if (scenario === 'queued_bot') return { skills: [{ name: 'telegram', enabled: true }, { name: 'telegram-bot', lifecycle_virtual: true }] };
            if (scenario === 'grants_bot') {
                return {
                    skills: [
                        { name: 'telegram', enabled: true },
                        { name: 'telegram-bot', enabled: false, grants: { blocking: true, missing_permissions: ['presence'] } },
                    ],
                };
            }
            if (scenario === 'old_hub') return { skills: [{ name: 'telegram', enabled: true },
                { name: 'telegram-bot', enabled: true, live_loaded: true, load_error: null, review_status: 'pending',
                    review_gate: { executable_review: false }, grants: { blocked: false } }] };
            return { skills: [{ name: 'telegram', enabled: true }] };
        },
    }));
    routes.set('/api/extensions/telegram/group_feed', () => ({
        ok: true,
        json: async () => ({
            v: 1,
            state: scenario === 'unpaired' ? 'ready' : 'ready',
            account_id: '101',
            identity: { sends_as: { id: '101', label: '@mybot', kind: 'bot' } },
            owner_bound: scenario !== 'unpaired',
            invite_url: 'https://t.me/mybot?startgroup=true',
            enabled: false,
            capable: scenario !== 'old_hub',
        }),
    }));
    routes.set('/api/extensions/telegram-bot/status', () => ({
        ok: true,
        json: async () => ({ text: 'Ready', tone: 'ok', identity: { sends_as: { id: '555', label: '@sep_bot', kind: 'bot' } }, connect: [] }),
    }));

    const openTelegramStep = async () => {
        const sheet = openConnectPlace({ openPlace: () => {} });
        t.after(() => sheet.close());
        await typeWords(sheet, 'Group rules.');
        await sheet.body.fire('click', { target: sheet.body.querySelector('[data-connect-act="next"]') });
        await flushAsync();
        await sheet.body.fire('click', { target: sheet.body.querySelector('[data-connect-act="open-tg-groups"]') });
        await flushAsync(8);
        return sheet;
    };

    // 1. `telegram-bot` absent -> routes to Skills Browse with `{ query: 'telegram-bot' }`.
    let sheet = await openTelegramStep();
    assert.match(sheet.body.textContent, /Install telegram-bot from Browse to use @mybot in groups/);
    await sheet.body.fire('click', { target: sheet.body.querySelector('[data-connect-act="tg-browse-bot"]') });
    assert.deepEqual(skillsOpened.at(-1), { query: 'telegram-bot' });

    // 2. Queued install (`lifecycle_virtual: true`) -> never implies ready!
    scenario = 'queued_bot';
    sheet = await openTelegramStep();
    assert.match(sheet.body.textContent, /still installing or updating in Skills for @mybot/);
    assert.equal(sheet.body.querySelector('[data-connect-act="tg-opt-in"]'), null);
    sheet.close();

    // 3. Personal `telegram` disabled in Skills -> offers `Enable in Skills` (`{ skill: 'telegram' }`), never `old_core`.
    scenario = 'tg_disabled';
    sheet = await openTelegramStep();
    assert.match(sheet.body.textContent, /personal Telegram bridge is off in Skills/);
    await sheet.body.fire('click', { target: sheet.body.querySelector('[data-connect-act="tg-open-skill"]') });
    assert.deepEqual(skillsOpened.at(-1), { skill: 'telegram' });

    // 4. Unpaired personal bot (`owner_bound: false`) -> never offers opt-in; directs to `telegram` in Skills.
    scenario = 'unpaired';
    sheet = await openTelegramStep();
    assert.match(sheet.body.textContent, /Pair your personal Telegram bot in a private chat first/);
    assert.equal(sheet.body.querySelector('[data-connect-act="tg-opt-in"]'), null);
    sheet.close();

    // 5. Grant-blocked / disabled `telegram-bot` -> opens `{ skill: 'telegram-bot' }`.
    scenario = 'grants_bot';
    sheet = await openTelegramStep();
    await sheet.body.fire('click', { target: sheet.body.querySelector('[data-connect-act="tg-open-skill"]') });
    assert.deepEqual(skillsOpened.at(-1), { skill: 'telegram-bot' });

    // 6. New core + older Hub `telegram-bot` (`capable: false`, no `owner_bot` projection) -> `Update in Skills` on A, while B works!
    scenario = 'old_hub';
    sheet = await openTelegramStep();
    await sheet.body.fire('click', { target: sheet.body.querySelector('[data-connect-act="tg-branch"][data-branch="personal"]') });
    await flushAsync();
    assert.match(sheet.body.textContent, /Telegram Bot needs an update in Skills before @mybot can connect groups here/);
    assert.doesNotMatch(sheet.body.textContent, /Review in Skills/);
    await sheet.body.fire('click', { target: sheet.body.querySelector('[data-connect-act="tg-open-skill"]') });
    assert.deepEqual(skillsOpened.at(-1), { skill: 'telegram-bot' });
    assert.deepEqual(dashboardsOpened, []);
});

test('Circles statusReads and Place sheet isolate telegram-bot:owner_bot from telegram-bot:place and pass ?account_id=<A>', async (t) => {
    const cleanups = [];
    const env = setup({ after: (fn) => cleanups.push(fn) });
    let circles = null, realSheet = null;
    t.after(() => { realSheet?.close(); circles?.destroy(); cleanups.forEach((fn) => fn()); });
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
    globalThis.CSS = { escape: (v) => v };
    t.after(() => { globalThis.CSS = priorCss; });

    const content = env.document.createElement('div');
    content.id = 'content';
    env.document.body.appendChild(content);

    const statusCalls = [];
    const placesPayload = {
        connectors: [rawTgPlaceConnector, rawTgOwnerBotConnector],
        circles: [{ circle_id: 'home', name: 'Home', text: '', text_author: '', text_at: '', revision: 1 }],
        places: [
            {
                ...rawTgOwnerBotConnector,
                place_id: 'p-owner',
                provider: 'telegram',
                account_id: '101',
                conversation_id: '-10042',
                label: 'Telegram · Family',
                state: 'silent',
                revision: 1,
                circle: { circle_id: 'home', placed_by: 'owner', placed_at: '' },
                notes: [], wakes: [], owner_accounts: [], rooms: [],
            },
            {
                ...rawTgPlaceConnector,
                contribution_id: '', // legacy migrated B place without contribution_id
                place_id: 'p-legacy-b',
                provider: 'telegram',
                account_id: '',
                conversation_id: '',
                label: 'Telegram',
                state: 'active',
                revision: 1,
                circle: { circle_id: 'home', placed_by: 'owner', placed_at: '' },
                notes: [], wakes: [{ fact: 'mention', author: 'owner' }], owner_accounts: [], rooms: [],
            },
        ],
    };

    const openedSheets = [];
    circles = initCircles({
        state: { activePage: 'circles' },
        showPage: async () => true,
        openSettings: async () => true,
        read: async () => normalizePlacesResponse(placesPayload),
        readModels: async () => ({ items: [] }),
        readStatus: async (target) => {
            statusCalls.push([target.connector, target.contributionId || '', target.accountId || '', target.source?.url || '']);
            if (target.source?.url === '/api/extensions/telegram-bot/owner_bot_status') {
                // Simulate unscoped owner_bot_status currently bound to a different bot `202`!
                return { v: 1, enabled: true, reason: '', account_id: '202', account_label: '@newer_owner_bot', identity: { sends_as: { id: '202', label: '@newer_owner_bot', kind: 'bot' } }, connect: [] };
            }
            return { text: 'Ready', tone: 'ok', account_label: '@separate_bot', identity: { sends_as: { id: '555', label: '@separate_bot', kind: 'bot' } } };
        },
        openSheet: (p, opts) => openedSheets.push([p, opts]),
        contributions: { onLifecycle: () => () => {} },
    });
    await circles.refresh();
    await flushAsync(8);

    // Both declared connector status routes were read independently (`telegram-bot:place` and `telegram-bot:owner_bot`).
    assert.deepEqual(statusCalls, [
        ['telegram-bot', 'place', '', '/api/extensions/telegram-bot/status'],
        ['telegram-bot', 'owner_bot', '', '/api/extensions/telegram-bot/owner_bot_status'],
    ]);

    // Legacy B whole-account place (`contribution_id: ""`) gets `@separate_bot` title from `telegram-bot:place`,
    // while clicking `p-owner` (`account_id: "101"`) NEVER receives unscoped B status or mismatched `202` status!
    const circlesPage = content.querySelector('#page-circles');
    assert.match(circlesPage.textContent, /@separate_bot/);
    assert.doesNotMatch(circlesPage.textContent, /@newer_owner_bot/);
    await circlesPage.fire('click', {
        target: circlesPage.querySelector('[data-place-open="p-legacy-b"]'),
    });
    await circlesPage.fire('click', {
        target: circlesPage.querySelector('[data-place-open="p-owner"]'),
    });
    assert.equal(openedSheets.length, 2);
    assert.equal(openedSheets[0][1].connectorStatus?.account_label, '@separate_bot');
    assert.equal(openedSheets[1][1].connectorStatus, null, 'mismatched unscoped status is never passed to saved A place');

    // Opening `openPlaceSheet` for `p-owner` requests `/api/extensions/telegram-bot/owner_bot_status?account_id=101`
    // via `readPlaceConnector`, shows `@my_saved_bot` when matched, and drops mismatched identities!
    env.routes.set('/api/extensions/telegram-bot/owner_bot_status?account_id=101', () => ({
        ok: true,
        json: async () => ({
            v: 1, enabled: true, reason: '', account_id: '101',
            identity: { sends_as: { id: '101', label: '@my_saved_bot', kind: 'bot' } },
        }),
    }));
    env.routes.set('/api/places/p-owner/rooms', () => ({
        ok: true,
        json: async () => ({ place_id: 'p-owner', rooms: [], coverage: { discovery: 'arrivals', importing: false, gaps: false } }),
    }));
    realSheet = openPlaceSheet(openedSheets[1][0], { circles: normalizePlacesResponse(placesPayload).circles });
    await flushAsync(8);
    const dialog = env.document.body.querySelector('dialog');
    assert.match(dialog.textContent, /@my_saved_bot/);
    assert.doesNotMatch(dialog.textContent, /@separate_bot|@newer_owner_bot/);
});

test('connecting the first personal Telegram group refreshes a cached Off only for that bot', async (t) => {
    const cleanups = [];
    const { routes, calls, document } = setup({ after: (fn) => cleanups.push(fn) });
    const priorCss = globalThis.CSS;
    globalThis.CSS = { escape: (value) => value };
    t.after(() => { globalThis.CSS = priorCss; });
    const content = document.createElement('div');
    content.id = 'content';
    document.body.appendChild(content);

    const account_id = '101';
    const identity = { sends_as: { id: account_id, label: '@mybot', kind: 'bot' } };
    const group = { account_id, conversation_id: '-10042', label: 'Book group' };
    const ownerStatus = '/api/extensions/telegram-bot/owner_bot_status';
    const separateStatus = '/api/extensions/telegram-bot/status';
    const feed = '/api/extensions/telegram/group_feed';
    const places = [];
    let allowed = false, intake = false;
    const ok = (data) => ({ ok: true, json: async () => data });
    const readLabels = () => intake
        ? { v: 1, enabled: true, account_id, identity, connect: [group] }
        : { v: 1, enabled: false, reason: 'feed_off', account_id: '', connect: null };
    const callsTo = (url) => calls.filter((entry) => entry === url).length;
    routes.set('/api/places', (_url, init) => {
        if (init?.method === 'POST') {
            const created = { ...rawTgOwnerBotConnector, ...group, place_id: 'book-group', provider: 'telegram', state: 'silent', revision: 1 };
            places.push(created);
            return ok({ ok: true, place: created });
        }
        return ok({ circles: [], places, connectors: [rawTgPlaceConnector, rawTgOwnerBotConnector] });
    });
    routes.set(ownerStatus, () => ok(readLabels()));
    routes.set(`${ownerStatus}?account_id=${account_id}`, () => ok(readLabels()));
    routes.set(separateStatus, () => ok({ text: 'Ready', tone: 'ok', connect: [] }));
    routes.set('/api/extensions', () => ok({ skills: [{ name: 'telegram', enabled: true, live_loaded: true },
        { name: 'telegram-bot', enabled: true, live_loaded: true }] }));
    routes.set(feed, (_url, init) => {
        if (init?.method === 'POST') {
            allowed = JSON.parse(init.body).enabled;
            if (!allowed) intake = false;
        }
        return ok({ ok: true, v: 1, state: 'ready', owner_bound: true, capable: true, account_id, identity,
            enabled: allowed, intake_ready: intake, consent_account_id: allowed ? account_id : '' });
    });
    let add;
    const circles = initCircles({ state: { activePage: 'circles' },
        openConnect: (options) => { add = openConnectPlace({ ...options, openPlace: () => {} }); },
        contributions: { onLifecycle: () => () => {} } });
    t.after(() => { add?.close(); circles.destroy(); cleanups.forEach((fn) => fn()); });
    const click = async (act) => {
        const target = add.body.querySelector(`[data-connect-act="${act}"]`);
        assert.ok(target, act);
        await add.body.fire('click', { target });
        await flushAsync(8);
    };
    const startAdd = async () => {
        await circles.page.fire('click', { target: circles.page.querySelector('[data-circles-act="add-place"]') });
        await typeWords(add, 'Answer when asked.');
        await click('next');
        await click('open-tg-groups');
    };
    const row = () => circles.page.querySelector('[data-place-open="book-group"]')?.textContent;

    await flushAsync(8);
    await startAdd();
    await click('tg-opt-in');
    assert.equal(callsTo(ownerStatus), 2, 'Allow triggered a read while intake was still off');
    intake = true;
    await click('tg-refresh');
    assert.equal(callsTo(ownerStatus), 2, 'Add read its own scoped status, not the list cache');
    await click('connect');
    assert.match(row(), /silent/);
    assert.doesNotMatch(row(), /feed off in Telegram/i);
    assert.equal(callsTo(ownerStatus), 3, 'Connect asked for one fresh personal bot status');
    assert.equal(callsTo(separateStatus), 1, 'the separate bot keeps its cached status');

    await circles.refresh();
    assert.equal(callsTo(ownerStatus), 3, 'an ordinary refresh does not reread the status');
    await startAdd();
    await click('tg-opt-out');
    assert.match(row(), /Group feed off in Telegram/);
});

test('A -> B -> A race while initial group_feed is held: B loads without waiting for A, late A does not overwrite B, and returning to A recovers and connects once', async (t) => {
    const { routes } = setup(t);
    let releaseFirstGroupFeed = null;
    let releaseSecondGroupFeed = null;
    let groupFeedCalls = 0;
    const placePosts = [];

    routes.set('/api/places', (_url, init) => {
        if (init?.method === 'POST') {
            const body = JSON.parse(init.body);
            placePosts.push(body);
            return {
                ok: true,
                json: async () => ({
                    ok: true,
                    place: {
                        ...rawTgOwnerBotConnector,
                        place_id: 'tg-a-race',
                        provider: 'telegram',
                        label: body.label,
                        account_id: body.account_id,
                        conversation_id: body.conversation_id,
                        state: 'silent',
                        revision: 1,
                    },
                }),
            };
        }
        return { ok: true, json: async () => ({ places: [], circles: [], connectors: [rawTgPlaceConnector, rawTgOwnerBotConnector] }) };
    });
    // Also keep `/api/extensions` pending the whole time to prove optional skills discovery never blocks B or A!
    routes.set('/api/extensions', () => new Promise(() => {}));
    routes.set('/api/extensions/telegram/group_feed', () => {
        groupFeedCalls += 1;
        const callIndex = groupFeedCalls;
        return new Promise((resolve) => {
            const finish = (accountId, handle) => resolve({
                ok: true,
                json: async () => ({
                    v: 1, state: 'ready', account_id: accountId,
                    identity: { sends_as: { id: accountId, label: handle, kind: 'bot' } },
                    owner_bound: true, invite_url: `https://t.me/${handle.slice(1)}?startgroup=true`,
                    enabled: true, capable: true, pending: 0, write_gaps: 0, last_write_gap_at: '',
                }),
            });
            if (callIndex === 1) releaseFirstGroupFeed = () => finish('999', '@stale_first_bot');
            else releaseSecondGroupFeed = () => finish('101', '@current_personal_bot');
        });
    });
    routes.set('/api/extensions/telegram-bot/status', () => ({
        ok: true,
        json: async () => ({
            text: 'Connected', tone: 'ok', link: 'connected',
            identity: { sends_as: { id: '555', label: '@separate_bot', kind: 'bot' } },
            connect: [{ account_id: '555', conversation_id: '-100555', label: 'Separate Group (-100555)' }],
        }),
    }));
    routes.set('/api/extensions/telegram-bot/owner_bot_status?account_id=101', () => ({
        ok: true,
        json: async () => ({
            v: 1, enabled: true, reason: '', account_id: '101',
            identity: { sends_as: { id: '101', label: '@current_personal_bot', kind: 'bot' } },
            invite_url: 'https://t.me/current_personal_bot?startgroup=true',
            connect: [{ account_id: '101', conversation_id: '-100101', label: 'Personal Group (-100101)' }],
        }),
    }));

    const sheet = openConnectPlace({ openPlace: () => {} });
    t.after(() => sheet.close());
    const click = async (selector) => sheet.body.fire('click', { target: sheet.body.querySelector(selector) });

    await typeWords(sheet, 'Race check rules.');
    await click('[data-connect-act="next"]');
    await flushAsync();
    assert.doesNotMatch(sheet.body.textContent, /TELEGRAM_PUBLIC_BOT_TOKEN/);

    // 1. Open Telegram Groups: starts initial `group_feed` (call #1, held); branch B row uses plain copy without env key name.
    const openPromise = click('[data-connect-act="open-tg-groups"]');
    await flushAsync();
    assert.equal(groupFeedCalls, 1);
    assert.match(sheet.body.textContent, /Checking your personal Telegram bot/);
    assert.doesNotMatch(sheet.body.textContent, /TELEGRAM_PUBLIC_BOT_TOKEN/, 'branch B summary row stays plain while A is selected');

    // 2. Switch to B while `group_feed` #1 is still pending: B resolves its group immediately without waiting for A!
    await click('[data-connect-act="tg-branch"][data-branch="separate"]');
    await flushAsync(8);
    assert.match(sheet.body.textContent, /Separate Group \(-100555\)/);

    // 3. Release cancelled `group_feed` #1 (`@stale_first_bot`): must NOT overwrite B or cache stale feed!
    releaseFirstGroupFeed();
    await openPromise;
    await flushAsync(6);
    assert.match(sheet.body.textContent, /Separate Group \(-100555\)/);
    assert.doesNotMatch(sheet.body.textContent, /@stale_first_bot/);

    // 4. Switch back to A: starts fresh `group_feed` #2, resolves `@current_personal_bot` + `Personal Group (-100101)`.
    const backToAPromise = click('[data-connect-act="tg-branch"][data-branch="personal"]');
    await flushAsync();
    assert.equal(groupFeedCalls, 2, 'returning to A starts a fresh group_feed discovery');
    releaseSecondGroupFeed();
    await backToAPromise;
    await flushAsync(8);
    assert.match(sheet.body.textContent, /Personal Group \(-100101\)/);
    assert.doesNotMatch(sheet.body.textContent, /@stale_first_bot|Separate Group/);

    // 5. Connect once -> single `POST /api/places` with `contribution_id: 'owner_bot'`.
    await click('[data-connect-act="connect"][data-choice="0"]');
    await flushAsync(6);
    assert.deepEqual(placePosts, [{
        contribution_id: 'owner_bot',
        connector: 'telegram-bot',
        account_id: '101',
        conversation_id: '-100101',
        label: 'Personal Group (-100101)',
        note: 'Race check rules.',
    }]);
});

test('link: connecting vs waiting_for_token, enabled-but-not-live vs old_core 404, and stale Hub feed_off without resubmitting enable', async (t) => {
    const { routes } = setup(t);
    let nativeSkill = { name: 'telegram', enabled: true, live_loaded: false };
    let feedMode = '404';
    let bStatus = { text: 'Connecting…', tone: 'muted', link: 'connecting', connect: null };
    const optInPosts = [];

    routes.set('/api/places', () => ({
        ok: true,
        json: async () => ({ places: [], circles: [], connectors: [rawTgPlaceConnector, rawTgOwnerBotConnector] }),
    }));
    routes.set('/api/extensions', () => ({
        ok: true,
        json: async () => ({
            skills: [nativeSkill, { name: 'telegram-bot', enabled: true, live_loaded: true }],
        }),
    }));
    routes.set('/api/extensions/telegram/group_feed', (_url, init) => {
        if (init?.method === 'POST') {
            optInPosts.push(JSON.parse(init.body));
            return { ok: true, json: async () => ({ ok: true }) };
        }
        if (feedMode === '404') return { status: 404, ok: false, json: async () => ({ error: 'not found' }) };
        return {
            ok: true,
            json: async () => ({
                v: 1, state: 'ready', account_id: '101',
                identity: { sends_as: { id: '101', label: '@mybot', kind: 'bot' } },
                owner_bound: true, invite_url: 'https://t.me/mybot?startgroup=true',
                enabled: true, capable: true, pending: 0, write_gaps: 0, last_write_gap_at: '',
            }),
        };
    });
    routes.set('/api/extensions/telegram-bot/owner_bot_status?account_id=101', () => ({
        ok: true,
        json: async () => ({ v: 1, enabled: false, reason: 'feed_off', connect: null }),
    }));
    routes.set('/api/extensions/telegram-bot/status', () => ({
        ok: true,
        json: async () => bStatus,
    }));

    const sheet = openConnectPlace({ openPlace: () => {} });
    t.after(() => sheet.close());
    const click = async (selector) => sheet.body.fire('click', { target: sheet.body.querySelector(selector) });

    await typeWords(sheet, 'Edge states.');
    await click('[data-connect-act="next"]');
    await flushAsync();
    await click('[data-connect-act="open-tg-groups"]');
    await flushAsync(8);

    // 1. B `link: 'connecting'` must NOT tell owner to create a new @BotFather bot or paste TELEGRAM_PUBLIC_BOT_TOKEN!
    assert.equal(sheet.body.querySelector('[data-tg-branch-panel]').dataset.tgBranchPanel, 'personal');
    await click('[data-connect-act="tg-branch"][data-branch="separate"]');
    assert.match(sheet.body.textContent, /connecting and confirming its @name/i);
    assert.doesNotMatch(sheet.body.textContent, /Create a new bot|@BotFather|TELEGRAM_PUBLIC_BOT_TOKEN/);

    // Switch to `waiting_for_token`: now B explicitly shows `@BotFather` + `TELEGRAM_PUBLIC_BOT_TOKEN`!
    bStatus = { text: 'Not set up: the bot token is missing', tone: 'warn', link: 'waiting_for_token', connect: null };
    await click('[data-connect-act="tg-refresh"]');
    await flushAsync(8);
    assert.match(sheet.body.textContent, /@BotFather/);
    assert.match(sheet.body.textContent, /TELEGRAM_PUBLIC_BOT_TOKEN/);

    // 2. Branch A with `enabled: true, live_loaded: false` + 404 must NOT claim `Open Updates` (`old_core`)!
    await click('[data-connect-act="tg-branch"][data-branch="personal"]');
    await flushAsync(6);
    assert.match(sheet.body.textContent, /enabled in Skills, but it is not running/i);
    assert.equal(sheet.body.querySelector('[data-connect-act="tg-open-updates"]'), null);

    // Ambiguous 404 (no `live_loaded` boolean): truthful generic unavailable state, still no `Open Updates`.
    nativeSkill = { name: 'telegram', enabled: true };
    await click('[data-connect-act="tg-refresh"]');
    await flushAsync(8);
    assert.match(sheet.body.textContent, /Could not read your personal Telegram bot status/);
    assert.equal(sheet.body.querySelector('[data-connect-act="tg-open-updates"]'), null);

    // Confirmed live `telegram` (`enabled: true, live_loaded: true`) + 404 -> `old_core` (`Open Updates`)!
    nativeSkill = { name: 'telegram', enabled: true, live_loaded: true };
    await click('[data-connect-act="tg-refresh"]');
    await flushAsync(8);
    assert.match(sheet.body.textContent, /needs an Ouroboros update/i);
    assert.ok(sheet.body.querySelector('[data-connect-act="tg-open-updates"]'));

    // 3. Native `group_feed.enabled === true` while Hub returns stale `feed_off`: must NOT offer `tg-opt-in` or resubmit POST!
    feedMode = 'enabled';
    await click('[data-connect-act="tg-refresh"]');
    await flushAsync(8);
    assert.match(sheet.body.textContent, /waiting for Telegram Bot to pick up the group feed/i);
    assert.equal(sheet.body.querySelector('[data-connect-act="tg-opt-in"]'), null, 'never resubmits opt-in when native says enabled');
    assert.equal(optInPosts.length, 0);
});


test('a lost Allow, Off or Connect reply leaves only a read until the matching saved fact is known', async (t) => {
    setup(t);
    const accountId = '101', conversationId = '-10042';
    const identity = { sends_as: { id: accountId, label: '@mybot', kind: 'bot' } };
    const writes = [], placeWrites = [], opened = [];
    const connector = normalizeConnector(rawTgOwnerBotConnector);
    let enabled = false, nativeReadOK = true, identityKnown = true, unreadableConsent = false, savedPlace = null;
    const native = () => ({ v: 1, state: identityKnown ? 'ready' : 'identity_unconfirmed', owner_bound: true, capable: true,
        enabled: identityKnown && enabled, account_id: identityKnown ? accountId : '', ...(identityKnown ? { identity } : {}),
        consent_account_id: unreadableConsent ? null : enabled ? accountId : '' });
    const sheet = openConnectPlace({
        circle: { id: 'family', name: 'Family' },
        resumeTelegram: { branch: 'personal', words: 'Reply when asked.', description: 'Family plans' },
        read: async () => ({ connectors: [connector], places: savedPlace ? [savedPlace] : [] }),
        readStatus: async (target) => {
            assert.equal(target.accountId, accountId);
            return { v: 1, enabled: true, reason: '', account_id: accountId, identity,
                connect: [{ account_id: accountId, conversation_id: conversationId, label: 'Family group' }] };
        },
        telegram: {
            readExtensions: async () => ({ skills: [] }),
            readGroupFeed: async () => {
                if (!nativeReadOK) throw new TypeError('read unavailable');
                return native();
            },
            writeGroupFeed: async (value, expectedAccount) => {
                writes.push([value, expectedAccount]);
                enabled = value;
                if (!value) nativeReadOK = false;
                throw new TypeError('reply disconnected');
            },
        },
        connect: async (payload) => { placeWrites.push(payload); throw new TypeError('reply disconnected'); },
        openPlace: (value) => opened.push(value),
    });
    t.after(() => sheet.close());
    const find = (action) => sheet.body.querySelector(`[data-connect-act="${action}"]`);
    const click = async (action) => sheet.body.fire('click', { target: find(action) });
    await flushAsync(8);

    await click('tg-opt-in');
    assert.deepEqual(writes, [[true, accountId]]);
    assert.equal(find('tg-opt-out')?.textContent, 'Turn off groups for @mybot', 'read-back of successful Allow shows Off, never a second Allow');
    assert.match(sheet.body.textContent, /stops receiving group messages and new Ouroboros replies in all groups using that bot\. Your private chat is unchanged\./);
    identityKnown = false;
    await click('tg-refresh');
    assert.match(sheet.body.textContent, /Turn off groups for bot ID 101/);
    assert.doesNotMatch(sheet.body.textContent, /previous bot/);
    assert.doesNotMatch(sheet.body.textContent, /@mybot/);
    assert.equal(find('connect'), null);
    await click('tg-opt-out');
    assert.deepEqual(writes, [[true, accountId], [false, accountId]]);
    assert.equal(find('tg-opt-out'), null);
    assert.equal(find('tg-opt-in'), null);
    assert.ok(find('tg-refresh'));
    await click('tg-refresh');
    assert.equal(writes.length, 2);
    assert.ok(find('tg-refresh'), 'a failed read cannot settle Off');
    nativeReadOK = true;
    identityKnown = true;
    unreadableConsent = true;
    await click('tg-refresh');
    assert.match(sheet.body.textContent, /group discovery change is not confirmed/);
    assert.equal(find('tg-opt-in'), null, 'a ready username plus null consent is not confirmed Off');
    unreadableConsent = false;
    await click('tg-refresh');
    assert.ok(find('tg-opt-in'));
    assert.equal(writes.length, 2);

    await click('tg-opt-in');  // a new owner choice after the old write was reconciled
    await click('connect');
    assert.ok(find('tg-check-connect'));
    assert.equal(find('connect'), null);
    await click('back');
    await click('back-words');
    assert.match(sheet.body.innerHTML, /Reply when asked\./);
    assert.equal(sheet.body.querySelector('[data-connect-draft="description"]').getAttribute('value'), 'Family plans');
    await click('next');
    await click('open-tg-groups');
    assert.ok(find('tg-check-connect'), 'going Back and reopening retains the unresolved action');
    assert.equal(find('connect'), null);
    await click('tg-check-connect');
    assert.equal(placeWrites.length, 1);

    savedPlace = normalizePlace({ place_id: 'exact-family', connector: 'telegram-bot', contribution_id: 'owner_bot',
        account_id: accountId, conversation_id: conversationId, label: 'Family group', state: 'silent',
        circle: { circle_id: 'family' }, description: { text: 'Family plans' }, notes: [{ text: 'Reply when asked.' }] });
    await click('tg-check-connect');
    assert.equal(placeWrites.length, 1);
    assert.equal(placeWrites[0].circleId, 'family');
    assert.equal(placeWrites[0].note, 'Reply when asked.');
    assert.equal(placeWrites[0].description, 'Family plans');
    assert.deepEqual(opened.map((value) => [value.id, value.state]), [['exact-family', 'silent']]);
});

test('previous-bot Off names the saved bot ID, updates the prerequisite wording, explains Off scope, and revokes the saved ID', async (t) => {
    setup(t);
    const writes = [];
    let consentId = '101';
    const identity = { sends_as: { id: '102', label: '@newerbot', kind: 'bot' } };
    const connector = normalizeConnector(rawTgOwnerBotConnector);
    const feed = () => ({ v: 1, state: 'ready', owner_bound: true, capable: true, enabled: false,
        account_id: '102', identity, consent_account_id: consentId });
    const sheet = openConnectPlace({
        circle: { id: 'family', name: 'Family' },
        resumeTelegram: { branch: 'personal', words: 'Reply when asked.', description: '' },
        read: async () => ({ connectors: [connector], places: [] }),
        readStatus: async () => ({ v: 1, enabled: false, reason: 'feed_off', connect: null }),
        telegram: {
            readExtensions: async () => ({ skills: [] }),
            readGroupFeed: async () => feed(),
            writeGroupFeed: async (value, expectedAccount) => {
                writes.push([value, expectedAccount]);
                if (!value && expectedAccount === '101') consentId = '';
                return feed();
            },
        },
        openPlace: () => {},
    });
    t.after(() => sheet.close());
    await flushAsync(8);

    const find = (action) => sheet.body.querySelector(`[data-connect-act="${action}"]`);
    assert.equal(find('tg-opt-in'), null, 'Allow stays unavailable while previous bot consent is still on');
    assert.match(sheet.body.textContent, /Turn off groups for the previous bot before allowing this bot\./);
    assert.equal(find('tg-opt-out')?.textContent, 'Turn off groups for the previous bot (ID 101)');
    assert.match(sheet.body.textContent, /Turning groups off stops receiving group messages and new Ouroboros replies in all groups using that bot\. Your private chat is unchanged\./);
    await sheet.body.fire('click', { target: find('tg-opt-out') });
    await flushAsync(6);
    assert.deepEqual(writes, [[false, '101']]);
    assert.equal(find('tg-opt-out'), null);
    assert.equal(find('tg-opt-in')?.textContent, 'Allow @newerbot in groups');
});

test('deleted circle (400 circle_not_found or vanished on return) is a definitive refusal that keeps the draft and lets the owner pick another circle or none', async (t) => {
    setup(t);
    const accountId = '101', conversationId = '-10042';
    const identity = { sends_as: { id: accountId, label: '@mybot', kind: 'bot' } };
    const connector = normalizeConnector(rawTgOwnerBotConnector);
    let circles = [{ id: 'family', name: 'Family' }, { id: 'work', name: 'Work' }];
    const placeWrites = [], opened = [], savedDrafts = [];
    const sheet = openConnectPlace({
        circle: { id: 'family', name: 'Family' },
        resumeTelegram: { branch: 'personal', words: 'Answer when mentioned.', description: 'Family room' },
        read: async () => ({ connectors: [connector], places: [], circles }),
        readStatus: async () => ({
            v: 1, enabled: true, reason: '', account_id: accountId, identity,
            connect: [{ account_id: accountId, conversation_id: conversationId, label: 'Family group' }],
        }),
        telegram: {
            readExtensions: async () => ({ skills: [] }),
            readGroupFeed: async () => ({
                v: 1, state: 'ready', owner_bound: true, capable: true, enabled: true,
                account_id: accountId, identity, consent_account_id: accountId,
            }),
        },
        connect: async (payload) => {
            placeWrites.push(payload);
            if (payload.circleId === 'family') {
                circles = [{ id: 'work', name: 'Work' }];
                throw Object.assign(new Error('No circle with id family.'), { status: 400, body: { ok: false, code: 'circle_not_found' } });
            }
            return normalizePlace({
                place_id: 'exact-work', connector: 'telegram-bot', contribution_id: 'owner_bot',
                account_id: accountId, conversation_id: conversationId, label: 'Family group', state: 'silent',
                circle: payload.circleId ? { circle_id: payload.circleId } : null,
                description: { text: payload.description }, notes: [{ text: payload.note }],
            });
        },
        openPlace: (value) => opened.push(value),
        onTelegramSetup: (draft, nextCircle) => savedDrafts.push([draft, nextCircle]),
    });
    t.after(() => sheet.close());
    const find = (sel) => sheet.body.querySelector(sel);
    await flushAsync(8);

    // 400 circle_not_found must be a definitive refusal, never entering tgConnectPending.
    await sheet.body.fire('click', { target: find('[data-connect-act="connect"]') });
    await flushAsync(6);
    assert.equal(find('[data-connect-act="tg-check-connect"]'), null, 'real 400 circle_not_found never becomes unconfirmed pending');
    assert.equal((sheet.body.textContent.match(/no longer exists/gi) || []).length, 1, 'deleted circle reason shown once, never duplicated');
    const connectBlocked = find('[data-connect-act="connect"]');
    assert.equal(connectBlocked.disabled, true, 'Connect stays disabled until a valid circle or Not in a circle is chosen');
    assert.equal(connectBlocked.classList.contains('btn-primary'), false, 'disabled Connect is not bright primary crimson');
    assert.equal(find('[data-connect-act="back"]').disabled, false, 'Back stays enabled when circle is deleted');
    assert.equal(find('[data-connect-act="tg-branch"][data-branch="separate"]').disabled, false, 'Branch B stays enabled when circle is deleted');
    const select = find('[data-connect-circle]');
    assert.ok(select, 'circle chooser appears so owner can pick another circle or Not in a circle');
    assert.equal(select.disabled, false, 'circle chooser stays enabled');
    assert.match(select.innerHTML, /Family \(deleted\)/);
    assert.match(select.innerHTML, /Not in a circle/);
    assert.match(select.innerHTML, /Work/);

    // Switch to Work -> dialog subtitle updates, Connect re-enables, and Connect succeeds once keeping words & description.
    select.value = 'work';
    await sheet.body.fire('change', { target: select });
    await flushAsync(4);
    assert.match(sheet.body.parentElement.querySelector('[data-place-meta]').textContent, /To Work/);
    assert.equal(find('[data-connect-act="connect"]').disabled, false, 'Connect re-enables once a valid circle is selected');
    await sheet.body.fire('click', { target: find('[data-connect-act="connect"]') });
    await flushAsync(6);
    assert.equal(placeWrites.length, 2);
    assert.equal(placeWrites[1].circleId, 'work');
    assert.equal(placeWrites[1].note, 'Answer when mentioned.');
    assert.equal(placeWrites[1].description, 'Family room');
    assert.equal(opened.length, 1);

    // Returning from setup after the original circle vanished: chooser appears immediately, Connect is disabled, and Not in a circle connects cleanly.
    const retWrites = [];
    const resumed = openConnectPlace({
        circle: { id: 'gone', name: 'Gone' },
        resumeTelegram: { branch: 'personal', words: 'Keep quiet.', description: 'No circle now' },
        read: async () => ({ connectors: [connector], places: [], circles: [{ id: 'work', name: 'Work' }] }),
        readStatus: async () => ({
            v: 1, enabled: true, reason: '', account_id: accountId, identity,
            connect: [{ account_id: accountId, conversation_id: conversationId, label: 'Family group' }],
        }),
        telegram: {
            readExtensions: async () => ({ skills: [] }),
            readGroupFeed: async () => ({
                v: 1, state: 'ready', owner_bound: true, capable: true, enabled: true,
                account_id: accountId, identity, consent_account_id: accountId,
            }),
        },
        connect: async (payload) => {
            retWrites.push(payload);
            return normalizePlace({ place_id: 'exact-none', connector: 'telegram-bot', contribution_id: 'owner_bot',
                account_id: accountId, conversation_id: conversationId, label: 'Family group', state: 'silent' });
        },
        openPlace: () => {},
    });
    t.after(() => { if (!resumed.closed) resumed.close(); });
    await flushAsync(8);
    assert.ok(resumed.body.querySelector('[data-connect-circle]'));
    assert.equal((resumed.body.textContent.match(/no longer exists/gi) || []).length, 1);
    assert.equal(resumed.body.querySelector('[data-connect-act="connect"]').disabled, true);
    assert.equal(resumed.body.querySelector('[data-connect-act="back"]').disabled, false);
    assert.equal(resumed.body.querySelector('[data-connect-act="tg-branch"][data-branch="separate"]').disabled, false);
    // Pressing disabled Connect before picking a valid circle sends nothing.
    await resumed.body.fire('click', { target: resumed.body.querySelector('[data-connect-act="connect"]') });
    assert.equal(retWrites.length, 0);
    const sel2 = resumed.body.querySelector('[data-connect-circle]');
    sel2.value = '';
    await resumed.body.fire('change', { target: sel2 });
    await flushAsync(4);
    assert.match(resumed.body.parentElement.querySelector('[data-place-meta]').textContent, /Not in a circle/);
    assert.equal(resumed.body.querySelector('[data-connect-act="connect"]').disabled, false);
    await resumed.body.fire('click', { target: resumed.body.querySelector('[data-connect-act="connect"]') });
    await flushAsync(6);
    assert.equal(retWrites.length, 1);
    assert.equal(retWrites[0].circleId, '');
    assert.equal(retWrites[0].note, 'Keep quiet.');
    assert.equal(retWrites[0].description, 'No circle now');
    assert.equal(savedDrafts.length, 0);
    sheet.close();
    resumed.close();
});

test('consent and place POST only treat typed route refusals as proven no-change; arbitrary 4xx/5xx stay read-only and never block Back or Branch B', async (t) => {
    setup(t);
    const accountId = '101';
    const identity = { sends_as: { id: accountId, label: '@mybot', kind: 'bot' } };
    const sepIdentity = { sends_as: { id: '202', label: '@sep_bot', kind: 'bot' } };
    const ownerConn = normalizeConnector(rawTgOwnerBotConnector);
    const sepConn = normalizeConnector(rawTgPlaceConnector);
    const readyOff = () => ({ v: 1, state: 'ready', owner_bound: true, capable: true, enabled: false,
        account_id: accountId, identity, consent_account_id: '' });
    const readyOn = () => ({ v: 1, state: 'ready', owner_bound: true, capable: true, enabled: true,
        account_id: accountId, identity, consent_account_id: accountId });
    let feed = readyOff();
    let writeBehavior = null;
    let connectBehavior = null;
    const writes = [];
    const placeWrites = [];
    const sheet = openConnectPlace({
        circle: { id: 'family', name: 'Family' },
        resumeTelegram: { branch: 'personal', words: 'Reply when asked.', description: 'Family' },
        read: async () => ({ connectors: [ownerConn, sepConn], places: [] }),
        readStatus: async (target) => (target.contributionId === 'owner_bot'
            ? { v: 1, enabled: true, reason: '', account_id: accountId, identity,
                connect: [{ account_id: accountId, conversation_id: '-100101', label: 'Family group' }] }
            : { text: 'Polling ok', tone: 'ok', link: 'connected', identity: sepIdentity,
                connect: [{ account_id: '202', conversation_id: '-100202', label: 'Separate group' }] }),
        telegram: {
            readExtensions: async () => ({ skills: [] }),
            readGroupFeed: async () => feed,
            writeGroupFeed: async (value, expected) => {
                writes.push([value, expected]);
                return writeBehavior(value, expected);
            },
        },
        connect: async (payload) => {
            placeWrites.push(payload);
            return connectBehavior(payload);
        },
        openPlace: () => {},
    });
    t.after(() => sheet.close());
    const find = (sel) => sheet.body.querySelector(sel);
    const click = async (sel) => { await sheet.body.fire('click', { target: find(sel) }); await flushAsync(6); };
    await flushAsync(8);

    // 1. 503 unwritable with v:1 unchanged status -> confirmed no-change refusal (NOT consentPending).
    writeBehavior = async () => {
        throw Object.assign(new Error('Could not save group discovery setting.'), {
            status: 503, body: { ok: false, code: 'unwritable', ...readyOff() },
        });
    };
    await click('[data-connect-act="tg-opt-in"]');
    assert.match(sheet.body.textContent, /Could not save group discovery setting/);
    assert.doesNotMatch(sheet.body.textContent, /not confirmed yet/);
    assert.ok(find('[data-connect-act="tg-opt-in"]'), 'confirmed no-change keeps Allow available');

    // 1b. Typed 400 { ok: false, code: 'invalid' } from group_feed route -> proven refusal (NOT consentPending).
    writeBehavior = async () => {
        throw Object.assign(new Error('Send enabled and the bot you saw.'), {
            status: 400, body: { ok: false, code: 'invalid', error: 'Send enabled and the bot you saw.' },
        });
    };
    await click('[data-connect-act="tg-opt-in"]');
    assert.match(sheet.body.textContent, /Send enabled and the bot you saw/);
    assert.doesNotMatch(sheet.body.textContent, /not confirmed yet/);
    assert.ok(find('[data-connect-act="tg-opt-in"]'));

    // 1c. Untyped 403 / 400 (no ok:false / route code) -> NOT proven refused; enters consentPending (read-only Check again).
    writeBehavior = async () => {
        throw Object.assign(new Error('Forbidden by proxy'), { status: 403, body: { error: 'Forbidden by proxy' } });
    };
    const writesBefore403 = writes.length;
    await click('[data-connect-act="tg-opt-in"]');
    assert.equal(writes.length, writesBefore403 + 1);
    assert.match(sheet.body.textContent, /group discovery change is not confirmed yet/);
    assert.equal(find('[data-connect-act="tg-opt-in"]'), null, 'untyped 403 must not leave Allow clickable');
    await click('[data-connect-act="tg-refresh"]');
    assert.equal(writes.length, writesBefore403 + 1, 'Check again only reads after untyped 403');
    // Settle via observed match, then return to off for the next sub-check.
    feed = readyOn();
    await click('[data-connect-act="tg-refresh"]');
    assert.ok(find('[data-connect-act="tg-opt-out"]'));
    feed = readyOff();
    await click('[data-connect-act="tg-refresh"]');

    // 2. 503 consent_unknown with v:1 consent_account_id: null -> shows unreadable consent, keeps Back & Branch B!
    writeBehavior = async () => {
        throw Object.assign(new Error('Saved consent could not be read.'), {
            status: 503, body: { ok: false, code: 'consent_unknown', ...readyOff(), consent_account_id: null },
        });
    };
    await click('[data-connect-act="tg-opt-in"]');
    assert.match(sheet.body.textContent, /Could not read the saved group discovery setting/);
    assert.equal(find('[data-connect-act="tg-opt-in"]'), null);
    assert.equal(find('[data-connect-act="tg-opt-out"]'), null);
    assert.ok(find('[data-connect-act="back"]'), 'Back stays reachable when consent is unreadable');
    assert.ok(find('[data-connect-act="tg-branch"][data-branch="separate"]'), 'Branch B stays reachable when consent is unreadable');

    // Switching to Branch B while A's consent is unreadable works cleanly!
    await click('[data-connect-act="tg-branch"][data-branch="separate"]');
    assert.match(sheet.body.textContent, /Separate group/);

    // Switch back to A with valid off state -> 503 unwritable that actually persisted settles cleanly.
    feed = readyOff();
    await click('[data-connect-act="tg-branch"][data-branch="personal"]');
    await click('[data-connect-act="tg-refresh"]');
    writeBehavior = async () => {
        feed = readyOn();
        throw Object.assign(new Error('Disk sync warning'), {
            status: 503, body: { ok: false, code: 'unwritable', ...readyOn() },
        });
    };
    await click('[data-connect-act="tg-opt-in"]');
    assert.ok(find('[data-connect-act="tg-opt-out"]'), 'post-write v:1 status showing consent landed reconciles cleanly');

    // 2b. POST /api/places typed refusal (400 place_connect_invalid) shows reason without pending; untyped 403/400 enters read-only tgConnectPending.
    connectBehavior = async () => {
        throw Object.assign(new Error('Choose a group your own bot is in now.'), {
            status: 400, body: { ok: false, code: 'place_connect_invalid', error: 'Choose a group your own bot is in now.' },
        });
    };
    await click('[data-connect-act="connect"]');
    assert.equal(find('[data-connect-act="tg-check-connect"]'), null, 'typed place_connect_invalid is a proven refusal');
    assert.match(sheet.body.textContent, /Choose a group your own bot is in now/);

    connectBehavior = async () => {
        throw Object.assign(new Error('Proxy rejected'), { status: 403, body: { error: 'Proxy rejected' } });
    };
    const placeWritesBefore403 = placeWrites.length;
    await click('[data-connect-act="connect"]');
    assert.equal(placeWrites.length, placeWritesBefore403 + 1);
    assert.ok(find('[data-connect-act="tg-check-connect"]'), 'untyped 403 on POST /api/places must stay read-only pending');
    await click('[data-connect-act="tg-check-connect"]');
    assert.equal(placeWrites.length, placeWritesBefore403 + 1, 'Check again never re-POSTs /api/places');
    sheet.close();

    // 3. Generic 502 without proven no-change on Off -> enters consentPending (only rereads, never resends), Back & B stay reachable!
    const sheet2 = openConnectPlace({
        circle: { id: 'family', name: 'Family' },
        resumeTelegram: { branch: 'personal', words: 'Reply when asked.', description: 'Family' },
        read: async () => ({ connectors: [ownerConn, sepConn], places: [] }),
        readStatus: async () => ({ v: 1, enabled: true, reason: '', account_id: accountId, identity, connect: [] }),
        telegram: {
            readExtensions: async () => ({ skills: [] }),
            readGroupFeed: async () => readyOn(),
            writeGroupFeed: async (value, expected) => {
                writes.push([value, expected]);
                throw Object.assign(new Error('Bad Gateway'), { status: 502, body: { error: 'upstream reset' } });
            },
        },
        openPlace: () => {},
    });
    t.after(() => sheet2.close());
    await flushAsync(8);
    const beforeWrites = writes.length;
    await sheet2.body.fire('click', { target: sheet2.body.querySelector('[data-connect-act="tg-opt-out"]') });
    await flushAsync(6);
    assert.equal(writes.length, beforeWrites + 1);
    assert.match(sheet2.body.textContent, /group discovery change is not confirmed yet/);
    assert.ok(sheet2.body.querySelector('[data-connect-act="back"]'), 'Back stays reachable during consentPending');
    assert.ok(sheet2.body.querySelector('[data-connect-act="tg-branch"][data-branch="separate"]'), 'Branch B stays reachable during consentPending');
    await sheet2.body.fire('click', { target: sheet2.body.querySelector('[data-connect-act="tg-refresh"]') });
    await flushAsync(6);
    assert.equal(writes.length, beforeWrites + 1, 'Check again never resends POST');
    sheet2.close();
});

test('Branch B checks connector_token_held only on waiting_for_token; healthy Branch B never reads group_feed', async (t) => {
    setup(t);
    const sepConn = normalizeConnector(rawTgPlaceConnector);
    const sepIdentity = { sends_as: { id: '202', label: '@sep_bot', kind: 'bot' } };
    let groupFeedReads = 0;
    let bMode = 'connected';
    let feedState = { v: 1, state: 'bridge_not_ready', owner_bound: true, capable: true, enabled: false, connector_token_held: true };
    const sheet = openConnectPlace({
        circle: { id: 'family', name: 'Family' },
        resumeTelegram: { branch: 'separate', words: 'Group notes', description: '' },
        read: async () => ({ connectors: [sepConn], places: [] }),
        readStatus: async () => (bMode === 'connected'
            ? { text: 'Polling ok', tone: 'ok', link: 'connected', identity: sepIdentity,
                connect: [{ account_id: '202', conversation_id: '-100202', label: 'Ops group' }] }
            : { text: 'Not set up: the bot token is missing', tone: 'warn', link: 'waiting_for_token', connect: null }),
        telegram: {
            readExtensions: async () => ({ skills: [{ name: 'telegram-bot', enabled: true, live_loaded: true }] }),
            readGroupFeed: async () => { groupFeedReads += 1; return feedState; },
        },
        openPlace: () => {},
    });
    t.after(() => sheet.close());
    await flushAsync(8);

    // Healthy B works immediately and NEVER touches native group_feed.
    assert.match(sheet.body.textContent, /Ops group/);
    assert.equal(groupFeedReads, 0, 'healthy Branch B must not depend on Branch A group_feed');

    // When B reports waiting_for_token after Check again, native group_feed connector_token_held explains the held token.
    bMode = 'waiting';
    await sheet.body.fire('click', { target: sheet.body.querySelector('[data-connect-act="tg-refresh"]') });
    await flushAsync(8);
    assert.equal(groupFeedReads, 1);
    assert.match(sheet.body.textContent, /held because its token belongs to your personal bot/i);

    // When connector_token_held is false, waiting_for_token shows the normal BotFather setup steps.
    feedState = { ...feedState, connector_token_held: false };
    await sheet.body.fire('click', { target: sheet.body.querySelector('[data-connect-act="tg-refresh"]') });
    await flushAsync(8);
    assert.match(sheet.body.textContent, /@BotFather/);
    assert.match(sheet.body.textContent, /TELEGRAM_PUBLIC_BOT_TOKEN/);
    sheet.close();
});

test('owner_bot is not special on non-telegram connectors, checkTelegramConnect dispatches places-changed, Other Telegram chats preserves draft on Open settings, and StateReadEnded uses owner copy', async (t) => {
    const env = setup(t);
    // 1. Non-telegram connector with contribution_id: 'owner_bot' matches legacy place with empty contribution_id and is not treated as Telegram.
    const customConn = normalizeConnector({ connector: 'custom-bot', contribution_id: 'owner_bot', label: 'Custom Bot', readiness: { text: 'Connector on', tone: 'ok' } });
    const customPlace = normalizePlace({ place_id: 'c-1', connector: 'custom-bot', contribution_id: '', label: 'Custom DM', conversation_id: '123' });
    assert.equal(connectorMatchesPlace(customConn, customPlace), true);
    const listHtml = connectListHtml({ connectors: [customConn], places: [customPlace] });
    assert.match(listHtml, /Custom Bot/);
    assert.doesNotMatch(listHtml, /Telegram Groups/);

    // 2. checkTelegramConnect dispatches ouro:places-changed when confirming a lost POST /api/places.
    const accountId = '101', conversationId = '-10042';
    const identity = { sends_as: { id: accountId, label: '@mybot', kind: 'bot' } };
    const ownerConn = normalizeConnector(rawTgOwnerBotConnector);
    const sepConn = normalizeConnector({ ...rawTgPlaceConnector, settings: 'connection' });
    let savedPlace = null;
    const changedEvents = [];
    env.window.addEventListener('ouro:places-changed', (ev) => changedEvents.push(ev.detail));
    const sheet = openConnectPlace({
        circle: { id: 'family', name: 'Family' },
        resumeTelegram: { branch: 'personal', words: 'Notes', description: 'Desc' },
        read: async () => ({ connectors: [ownerConn, sepConn], places: savedPlace ? [savedPlace] : [] }),
        readStatus: async () => ({ v: 1, enabled: true, reason: '', account_id: accountId, identity,
            connect: [{ account_id: accountId, conversation_id: conversationId, label: 'Family group' }] }),
        telegram: {
            readExtensions: async () => ({ skills: [] }),
            readGroupFeed: async () => ({ v: 1, state: 'ready', owner_bound: true, capable: true, enabled: true,
                account_id: accountId, identity, consent_account_id: accountId }),
        },
        connect: async () => { throw new TypeError('network reset'); },
        openPlace: () => {},
    });
    t.after(() => sheet.close());
    await flushAsync(8);
    await sheet.body.fire('click', { target: sheet.body.querySelector('[data-connect-act="connect"]') });
    await flushAsync(6);
    assert.equal(changedEvents.length, 0);
    savedPlace = normalizePlace({ place_id: 'reconciled-1', connector: 'telegram-bot', contribution_id: 'owner_bot',
        account_id: accountId, conversation_id: conversationId, label: 'Family group', state: 'silent',
        circle: { circle_id: 'family' }, description: { text: 'Desc' }, notes: [{ text: 'Notes' }] });
    await sheet.body.fire('click', { target: sheet.body.querySelector('[data-connect-act="tg-check-connect"]') });
    await flushAsync(6);
    assert.equal(changedEvents.length, 1);
    assert.equal(changedEvents[0].place.id, 'reconciled-1');
    sheet.close();

    // 3. Other Telegram chats -> Open connector settings saves words, description, circle, branch, and other: true.
    const savedSetup = [];
    const openedSettings = [];
    const otherSheet = openConnectPlace({
        circle: { id: 'family', name: 'Family' },
        resumeTelegram: { branch: 'separate', words: 'DM words', description: 'DM desc' },
        read: async () => ({ connectors: [sepConn], places: [] }),
        readStatus: async () => ({ text: 'Polling ok', tone: 'ok', link: 'connected',
            identity: { sends_as: { id: '202', label: '@sep_bot', kind: 'bot' } }, connect: [] }),
        telegram: { readExtensions: async () => ({ skills: [] }) },
        openSettings: async (tab) => { openedSettings.push(tab); return true; },
        onTelegramSetup: (draft, nextCircle) => savedSetup.push([draft, nextCircle]),
        openPlace: () => {},
    });
    await flushAsync(8);
    await otherSheet.body.fire('click', { target: otherSheet.body.querySelector('[data-connect-act="tg-other-chats"]') });
    await flushAsync(6);
    await otherSheet.body.fire('click', { target: otherSheet.body.querySelector('[data-connect-act="settings"]') });
    await flushAsync(4);
    assert.deepEqual(openedSettings, ['skill/telegram-bot']);
    assert.deepEqual(savedSetup, [[{ words: 'DM words', description: 'DM desc', branch: 'separate', other: true }, { id: 'family', name: 'Family' }]]);
    otherSheet.close();

    // Returning with resumeTelegram.other reopens Other Telegram chats with words, description, and circle intact.
    const resumedOther = openConnectPlace({
        circle: { id: 'family', name: 'Family' },
        resumeTelegram: { branch: 'separate', other: true, words: 'DM words', description: 'DM desc' },
        read: async () => ({ connectors: [sepConn], places: [] }),
        readStatus: async () => ({ text: 'Polling ok', tone: 'ok', link: 'connected',
            identity: { sends_as: { id: '202', label: '@sep_bot', kind: 'bot' } },
            connect: [{ account_id: '202', conversation_id: '999', label: 'Private DM' }] }),
        telegram: { readExtensions: async () => ({ skills: [] }) },
        openPlace: () => {},
    });
    await flushAsync(8);
    assert.match(resumedOther.body.textContent, /Private DM/);
    assert.match(resumedOther.body.parentElement.querySelector('[data-place-meta]').textContent, /To Family/);
    resumedOther.close();

    // 4. StateReadEnded displays owner-readable timeout copy instead of leaking "state read expired".
    const timedOut = openConnectPlace({
        circle: { id: 'family', name: 'Family' },
        resumeTelegram: { branch: 'personal', words: 'Notes', description: '' },
        read: async () => { throw Object.assign(new Error('state read expired'), { name: 'StateReadEnded', reason: 'expired' }); },
        telegram: { readExtensions: async () => ({ skills: [] }) },
        openPlace: () => {},
    });
    await flushAsync(6);
    assert.match(timedOut.body.textContent, /The request took too long\./);
    assert.doesNotMatch(timedOut.body.textContent, /state read expired/i);
    timedOut.close();
});

test('unconfirmed A Allow/Off survives A -> B -> Accounts/Settings -> Circles -> B -> A without allowing a duplicate POST', async (t) => {
    const cleanups = [];
    const env = setup({ after: (fn) => cleanups.push(fn) });
    let circles = null;
    t.after(() => {
        env.document.body.querySelector('dialog')?.close?.();
        circles?.destroy();
        cleanups.forEach((fn) => fn());
    });
    const priorCss = globalThis.CSS;
    globalThis.CSS = { escape: (v) => v };
    t.after(() => { globalThis.CSS = priorCss; });
    const content = env.document.createElement('div');
    content.id = 'content';
    env.document.body.appendChild(content);

    const accountId = '101';
    const identity = { sends_as: { id: accountId, label: '@mybot', kind: 'bot' } };
    const sepIdentity = { sends_as: { id: '555', label: '@sep_bot', kind: 'bot' } };
    let feedEnabled = false;
    let sepReady = false;
    const feedPosts = [];
    const settingsOpened = [];

    env.routes.set('/api/places', () => ({
        ok: true,
        json: async () => ({
            places: [],
            circles: [{ circle_id: 'family', name: 'Family', text: 'Close family', text_author: 'owner', text_at: '', revision: 1 }],
            connectors: [rawTgPlaceConnector, rawTgOwnerBotConnector],
        }),
    }));
    env.routes.set('/api/extensions', () => ({
        ok: true,
        json: async () => ({
            skills: [
                { name: 'telegram', enabled: true, live_loaded: true },
                { name: 'telegram-bot', enabled: true, live_loaded: true },
            ],
        }),
    }));
    env.routes.set('/api/extensions/telegram/group_feed', (_url, init) => {
        if (init?.method === 'POST') {
            feedPosts.push(JSON.parse(init.body));
            throw new TypeError('reply disconnected');
        }
        return {
            ok: true,
            json: async () => ({
                v: 1, state: 'ready', account_id: accountId, identity, owner_bound: true,
                invite_url: 'https://t.me/mybot?startgroup=true',
                enabled: feedEnabled, capable: true, consent_account_id: feedEnabled ? accountId : '',
            }),
        };
    });
    env.routes.set('/api/extensions/telegram-bot/status', () => ({
        ok: true,
        json: async () => (sepReady
            ? { text: 'Polling ok', tone: 'ok', link: 'connected', identity: sepIdentity,
                connect: [{ account_id: '555', conversation_id: '-100555', label: 'Ops (-100555)' }] }
            : { text: 'Not set up: the bot token is missing', tone: 'warn', link: 'waiting_for_token', connect: null }),
    }));
    env.routes.set('/api/extensions/telegram-bot/owner_bot_status', () => ({
        ok: true,
        json: async () => ({ v: 1, enabled: feedEnabled, reason: feedEnabled ? '' : 'feed_off',
            account_id: accountId, identity, connect: feedEnabled ? [] : null }),
    }));
    env.routes.set('/api/extensions/telegram-bot/owner_bot_status?account_id=101', () => ({
        ok: true,
        json: async () => ({ v: 1, enabled: feedEnabled, reason: feedEnabled ? '' : 'feed_off',
            account_id: accountId, identity, connect: feedEnabled ? [] : null }),
    }));

    const state = { activePage: 'circles' };
    circles = initCircles({
        state,
        showPage: async () => true,
        openSettings: async (tab, opts) => { settingsOpened.push([tab, opts]); state.activePage = 'settings'; return true; },
        readModels: async () => ({ items: [] }),
        contributions: { onLifecycle: () => () => {} },
    });
    await circles.refresh();
    await flushAsync(8);

    const dialogBody = () => env.document.body.querySelector('dialog')?.querySelector('.place-sheet-body');
    const clickSheet = async (selector) => {
        const body = dialogBody();
        await body.fire('click', { target: body.querySelector(selector) });
        await flushAsync(8);
    };

    // 1. Open Add a place -> enter words -> open Telegram Groups -> Allow POST loses reply while GET still shows Off.
    const circlesPage = content.querySelector('#page-circles');
    await circlesPage.fire('click', { target: circlesPage.querySelector('[data-circles-act="add-place"][data-circle="family"]') });
    await flushAsync(4);
    const wordsField = dialogBody().querySelector('[data-connect-draft="words"]');
    wordsField.value = 'Keep family private.';
    await dialogBody().fire('input', { target: wordsField });
    await clickSheet('[data-connect-act="next"]');
    await clickSheet('[data-connect-act="open-tg-groups"]');

    await clickSheet('[data-connect-act="tg-opt-in"]');
    assert.deepEqual(feedPosts, [{ enabled: true, expected_account_id: '101' }]);
    assert.match(dialogBody().textContent, /group discovery change is not confirmed yet/);
    assert.equal(dialogBody().querySelector('[data-connect-act="tg-opt-in"]'), null);

    // 2. Switch A -> B -> Open Accounts and keys (closes sheet, saves setup draft) -> return to Circles -> B -> A.
    await clickSheet('[data-connect-act="tg-branch"][data-branch="separate"]');
    await clickSheet('[data-connect-act="tg-open-keys"]');
    assert.equal(env.document.body.querySelector('dialog'), null, 'sheet closes when navigating to Accounts and keys');
    assert.deepEqual(settingsOpened.at(-1), ['accounts', { row: 'TELEGRAM_PUBLIC_BOT_TOKEN' }]);

    state.activePage = 'circles';
    env.window.dispatchEvent(new CustomEvent('ouro:page-shown', { detail: { page: 'circles' } }));
    await flushAsync(10);
    assert.ok(dialogBody(), 'sheet reopens on return to Circles');
    assert.equal(dialogBody().querySelector('[data-connect-act="tg-branch"][data-branch="separate"]').getAttribute('aria-pressed'), 'true');

    // Return B -> A while GET still shows old consent (enabled: false): POST must remain blocked; Back, B, Check again stay usable.
    await clickSheet('[data-connect-act="tg-branch"][data-branch="personal"]');
    assert.match(dialogBody().textContent, /group discovery change is not confirmed yet/);
    assert.equal(dialogBody().querySelector('[data-connect-act="tg-opt-in"]'), null, 'Allow stays blocked after B -> Settings -> Circles -> B -> A');
    assert.equal(dialogBody().querySelector('[data-connect-act="tg-opt-out"]'), null);
    assert.ok(dialogBody().querySelector('[data-connect-act="back"]'));
    assert.ok(dialogBody().querySelector('[data-connect-act="tg-branch"][data-branch="separate"]'));
    await clickSheet('[data-connect-act="tg-refresh"]');
    assert.equal(feedPosts.length, 1, 'Check again only reads and never repeats Allow POST');

    // Going Back -> Telegram Groups while consent is still unconfirmed also never unlocks a duplicate POST.
    await clickSheet('[data-connect-act="back"]');
    await clickSheet('[data-connect-act="open-tg-groups"]');
    assert.match(dialogBody().textContent, /group discovery change is not confirmed yet/);
    assert.equal(dialogBody().querySelector('[data-connect-act="tg-opt-in"]'), null);
    assert.equal(feedPosts.length, 1);

    // Once the server observation matches (enabled: true), Check again settles cleanly without another POST.
    feedEnabled = true;
    await clickSheet('[data-connect-act="tg-refresh"]');
    assert.ok(dialogBody().querySelector('[data-connect-act="tg-opt-out"]'));
    assert.equal(feedPosts.length, 1);

    // 3. Same guarantee for Off across A -> B -> Open connector settings -> Circles -> B -> A.
    sepReady = true;
    await clickSheet('[data-connect-act="tg-opt-out"]');
    assert.equal(feedPosts.length, 2);
    assert.deepEqual(feedPosts[1], { enabled: false, expected_account_id: '101' });
    assert.match(dialogBody().textContent, /group discovery change is not confirmed yet/);
    await clickSheet('[data-connect-act="tg-branch"][data-branch="separate"]');
    await clickSheet('[data-connect-act="tg-bot-settings"]');
    assert.equal(env.document.body.querySelector('dialog'), null);
    assert.deepEqual(settingsOpened.at(-1), ['skill/telegram-bot', undefined]);

    state.activePage = 'circles';
    env.window.dispatchEvent(new CustomEvent('ouro:page-shown', { detail: { page: 'circles' } }));
    await flushAsync(10);
    await clickSheet('[data-connect-act="tg-branch"][data-branch="personal"]');
    assert.match(dialogBody().textContent, /group discovery change is not confirmed yet/);
    assert.equal(dialogBody().querySelector('[data-connect-act="tg-opt-out"]'), null, 'Off stays blocked after B -> connector settings -> Circles -> B -> A');
    assert.equal(dialogBody().querySelector('[data-connect-act="tg-opt-in"]'), null);
    assert.equal(feedPosts.length, 2);

    feedEnabled = false;
    await clickSheet('[data-connect-act="tg-refresh"]');
    assert.ok(dialogBody().querySelector('[data-connect-act="tg-opt-in"]'));
    assert.equal(feedPosts.length, 2, 'settling Off sends no extra POST');
});

test('POST /api/places ambiguous 400s (place_invalid, places_unreadable) reconcile read-only without duplicate POST; strictly pre-write 400s stay refusals and circle_not_found keeps picker', async (t) => {
    const env = setup(t);
    const accountId = '101';
    const conversationId = '-100777';
    const identity = { sends_as: { id: accountId, label: '@mybot', kind: 'bot' } };
    const ownerConn = normalizeConnector(rawTgOwnerBotConnector);
    const sepConn = normalizeConnector(rawTgPlaceConnector);
    const readyOn = { v: 1, state: 'ready', owner_bound: true, capable: true, enabled: true,
        account_id: accountId, identity, consent_account_id: accountId };
    const groupStatus = { v: 1, enabled: true, reason: '', account_id: accountId, identity,
        connect: [{ account_id: accountId, conversation_id: conversationId, label: 'Family group' }] };
    const createdPlace = normalizePlace({
        place_id: 'tg-created-1', connector: 'telegram-bot', contribution_id: 'owner_bot',
        account_id: accountId, conversation_id: conversationId, label: 'Family group', state: 'silent',
        circle: { circle_id: 'family' }, description: { text: 'Family chat' }, notes: [{ text: 'Reply when asked.' }],
    });

    // 1. Post-write place_view failure returning HTTP 400 place_invalid reconciles immediately via GET /api/places with ZERO duplicate POST.
    let storedPlaces = [];
    const postCalls = [];
    const opened = [];
    const changed = [];
    env.window.addEventListener('ouro:places-changed', (ev) => changed.push(ev.detail));
    const sheet1 = openConnectPlace({
        circle: { id: 'family', name: 'Family' },
        resumeTelegram: { branch: 'personal', words: 'Reply when asked.', description: 'Family chat' },
        read: async () => ({ connectors: [ownerConn, sepConn], circles: [{ id: 'family', name: 'Family' }], places: storedPlaces }),
        readStatus: async () => groupStatus,
        telegram: { readExtensions: async () => ({ skills: [] }), readGroupFeed: async () => readyOn },
        connect: async (payload) => {
            postCalls.push(payload);
            storedPlaces = [createdPlace];
            throw Object.assign(new Error('place_view projection failed after write'), {
                status: 400, body: { ok: false, code: 'place_invalid', error: 'place_view projection failed after write' },
            });
        },
        openPlace: (place) => opened.push(place),
    });
    t.after(() => sheet1.close());
    await flushAsync(8);
    await sheet1.body.fire('click', { target: sheet1.body.querySelector('[data-connect-act="connect"]') });
    await flushAsync(8);
    assert.equal(postCalls.length, 1, 'place_invalid after persistence must never send a second POST');
    assert.equal(sheet1.closed, true, 'sheet closes once read-only GET confirms the created group');
    assert.equal(opened.length, 1);
    assert.equal(opened[0].id, 'tg-created-1');
    assert.equal(changed.length, 1);
    assert.equal(changed[0].place.id, 'tg-created-1');

    // 2. Post-write 400 places_unreadable when GET initially misses stays read-only pending and lands on Check again without re-POSTing.
    let visible = false;
    const postCalls2 = [];
    const opened2 = [];
    const sheet2 = openConnectPlace({
        circle: { id: 'family', name: 'Family' },
        resumeTelegram: { branch: 'personal', words: 'Reply when asked.', description: 'Family chat' },
        read: async () => ({ connectors: [ownerConn, sepConn], circles: [{ id: 'family', name: 'Family' }], places: visible ? [createdPlace] : [] }),
        readStatus: async () => groupStatus,
        telegram: { readExtensions: async () => ({ skills: [] }), readGroupFeed: async () => readyOn },
        connect: async (payload) => {
            postCalls2.push(payload);
            throw Object.assign(new Error('Place records need repair'), {
                status: 400, body: { ok: false, code: 'places_unreadable', error: 'Place records need repair' },
            });
        },
        openPlace: (place) => opened2.push(place),
    });
    t.after(() => sheet2.close());
    await flushAsync(8);
    await sheet2.body.fire('click', { target: sheet2.body.querySelector('[data-connect-act="connect"]') });
    await flushAsync(8);
    assert.equal(postCalls2.length, 1);
    assert.ok(sheet2.body.querySelector('[data-connect-act="tg-check-connect"]'), 'places_unreadable enters read-only confirmation');
    assert.equal(sheet2.body.querySelector('[data-connect-act="connect"]'), null, 'Connect stays unavailable while unconfirmed');
    visible = true;
    await sheet2.body.fire('click', { target: sheet2.body.querySelector('[data-connect-act="tg-check-connect"]') });
    await flushAsync(8);
    assert.equal(postCalls2.length, 1, 'Check again reconciles via GET only');
    assert.equal(sheet2.closed, true);
    assert.equal(opened2.length, 1);
    assert.equal(opened2[0].id, 'tg-created-1');

    // 3. Strictly pre-write 400 refusals (place_description_invalid, place_declaration_unavailable) stay proven refusals,
    // and circle_not_found preserves the circle picker flow.
    let circles = [{ id: 'family', name: 'Family' }];
    let mode = 'desc';
    const postCalls3 = [];
    const opened3 = [];
    const sheet3 = openConnectPlace({
        circle: { id: 'family', name: 'Family' },
        resumeTelegram: { branch: 'personal', words: 'Reply when asked.', description: 'Family chat' },
        read: async () => ({ connectors: [ownerConn, sepConn], circles, places: [] }),
        readStatus: async () => groupStatus,
        telegram: { readExtensions: async () => ({ skills: [] }), readGroupFeed: async () => readyOn },
        connect: async (payload) => {
            postCalls3.push(payload);
            if (mode === 'desc') {
                throw Object.assign(new Error('Description is too long.'), {
                    status: 400, body: { ok: false, code: 'place_description_invalid', error: 'Description is too long.' },
                });
            }
            if (mode === 'decl') {
                throw Object.assign(new Error('Enable a reviewed connector with one place declaration.'), {
                    status: 400, body: { ok: false, code: 'place_declaration_unavailable', error: 'Enable a reviewed connector with one place declaration.' },
                });
            }
            if (mode === 'circle') {
                circles = [{ id: 'friends', name: 'Friends' }];
                throw Object.assign(new Error('circle_not_found'), {
                    status: 400, body: { ok: false, code: 'circle_not_found', error: 'circle_not_found' },
                });
            }
            return { ...createdPlace, circleId: payload.circleId };
        },
        openPlace: (place) => opened3.push(place),
    });
    t.after(() => sheet3.close());
    await flushAsync(8);
    await sheet3.body.fire('click', { target: sheet3.body.querySelector('[data-connect-act="connect"]') });
    await flushAsync(6);
    assert.equal(sheet3.body.querySelector('[data-connect-act="tg-check-connect"]'), null);
    assert.match(sheet3.body.textContent, /Description is too long/);
    mode = 'decl';
    await sheet3.body.fire('click', { target: sheet3.body.querySelector('[data-connect-act="connect"]') });
    await flushAsync(6);
    assert.equal(sheet3.body.querySelector('[data-connect-act="tg-check-connect"]'), null);
    assert.match(sheet3.body.textContent, /Enable a reviewed connector/);
    mode = 'circle';
    await sheet3.body.fire('click', { target: sheet3.body.querySelector('[data-connect-act="connect"]') });
    await flushAsync(8);
    assert.equal(sheet3.body.querySelector('[data-connect-act="tg-check-connect"]'), null, 'circle_not_found must not enter tgConnectPending');
    const picker = sheet3.body.querySelector('[data-connect-circle]');
    assert.ok(picker, 'circle_not_found shows the circle picker');
    picker.value = 'friends';
    await sheet3.body.fire('change', { target: picker });
    mode = 'ok';
    await sheet3.body.fire('click', { target: sheet3.body.querySelector('[data-connect-act="connect"]') });
    await flushAsync(6);
    assert.equal(postCalls3.at(-1).circleId, 'friends');
    assert.equal(opened3.length, 1);
});
