import assert from 'node:assert/strict';
import test from 'node:test';
import { connectChoices, connectListHtml, openConnectPlace, openConnectorSettings } from '../modules/place_connect.js';
import { connectPlace, normalizePlace, normalizeConnector, readPlaceConnector, readPlaces } from '../modules/consciousness_api.js';
import { setup } from './dashboard_dom_fixture.js';

const raw = { connector: 'github', label: 'GitHub', icon: { kind: 'svg', svg: '<svg viewBox="0 0 24 24"><circle r="7"/></svg>' },
    source: { url: '/api/extensions/github/status', refresh_s: 60 }, settings: 'connection',
    readiness: { text: 'Connector on', tone: 'ok' }, wakes: ['mention'], facts: ['mention', { id: 'signal.words', label: 'Words you chose' }] };
const github = normalizeConnector(raw);
const rows = [{ account_id: 'writer', conversation_id: 'repo.123', label: 'example/project' }];
const place = normalizePlace({ ...raw, place_id: 'gh-1', label: 'GitHub · example/project', account_id: 'writer', conversation_id: 'repo.123', state: 'silent', revision: 7 });
const strip = (html) => html.replace(/<[^>]*>/g, ' ').replace(/\s+/g, ' ').trim();
const choicesHtml = (data, extra = {}) => connectListHtml({ connectors: [github], selected: github, choices: connectChoices(data), ...extra });

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
    assert.match(off, /data-connect-act="skills" data-connector="github">Enable in Skills</);
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

test('connecting an instance carries only its identity and connector-authored label', async (t) => {
    const { routes } = setup(t);
    const calls = [];
    routes.set('/api/places', (_url, init) => {
        calls.push(JSON.parse(init.body));
        return { ok: true, json: async () => ({ ok: true, place: { ...raw, place_id: 'gh-1', state: 'silent' } }) };
    });
    assert.equal((await connectPlace({ connector: 'github', ...connectChoices({ connect: rows }).rows[0] })).state, 'silent');
    await connectPlace({ connector: 'github' });
    assert.deepEqual(calls, [
        { connector: 'github', account_id: 'writer', conversation_id: 'repo.123', label: 'example/project' },
        { connector: 'github' },
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
    // A narrower place that came from configure_presence or the migration: the connector lists no instances.
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
