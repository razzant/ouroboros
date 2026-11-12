// The Model capsule (composer_model.js; DESIGN "Composer owner controls"): the first capsule of the row. A room's
// capsule follows Main until the room chooses, and says so; closed it shows the room's choice, never a running work's
// model; open, its strip is Main · the recent models · Search, and Search's suggestions float outside the row. A
// room's pick saves at once for that room; the Main chat's pick asks first and writes Main itself. Engines, geometry
// and the list's place are the browser lane's.
import assert from 'node:assert/strict';
import test from 'node:test';
import { MAIN_RULE, MODEL_RULE, confirmMainChange, modelsRunningElsewhere } from '../modules/composer_model.js';
import { createModelCatalogStore } from '../modules/model_catalog_store.js';
import { MAIN_FOLLOW_NOTE } from '../modules/model_roles.js';
import { ElementStub } from './chat_dom_fixture.js';
import { mountPair } from './composer_controls_fixture.js';

const ASTRA = 'claudexor::codex=gpt-6-astra';
const SOL = 'claudexor::codex=gpt-6.1-sol';
const GLM = 'z-ai/glm-5.3';
const KIMI = 'moonshotai/kimi-k3';
const route = (model, local = false) => ({ model, use_local: local });
const codex = (model, extra = {}) => ({ provider_id: 'claudexor', provider: 'Codex', source_id: 'codex', id: model, name: model,
    value: `claudexor::codex=${model}`, credential_profile_id: 'home', availability: 'available', ...extra });
const openrouter = (id) => ({ provider_id: 'openrouter', provider: 'OpenRouter', id, name: id, value: id });
const CATALOG = {
    model_sources: [{ id: 'codex', label: 'Codex', credentialHarness: 'codex' }],
    items: [codex('gpt-6-astra', { max_context_window: 872000 }), codex('gpt-6.1-sol'),
        codex('gpt-5-old', { availability: 'unavailable' }), openrouter(GLM), openrouter(KIMI), openrouter('x-ai/grok-4.7')],
};
const storeOf = ({ catalog = CATALOG, recents = [], connections = [], settings = {} } = {}) => {
    const reads = { catalog: 0, recents: 0, settings: 0 };
    const answer = { recents };
    const store = createModelCatalogStore({
        readCatalog: async () => { reads.catalog += 1; if (catalog instanceof Error) throw catalog; return catalog; },
        readPreferences: async () => { reads.recents += 1; return answer.recents === undefined ? {} : { model_recents: answer.recents }; },
        readSettings: async () => { reads.settings += 1; return { _meta: { connections }, ...settings }; },
    });
    return { store, reads, answer };
};
const flush = async () => { for (let i = 0; i < 4; i += 1) await new Promise((resolve) => setImmediate(resolve)); };
const COGNITION = { context_mode: 'max', subagents_mode: 'auto', effort_range: { min: 'low', recommended: 'high', max: 'max' } };

function room({ projectId = 'p7', chatId = 7, choice = { kind: 'main' }, main = ASTRA, model = {}, storeOptions, ...rest } = {}) {
    const kept = storeOf(storeOptions);
    const asked = [];
    const confirmed = { value: true };
    const m = mountPair({ projectId, chatId, subagents: true, ...rest, model: { store: kept.store,
        askMain: async (request) => { asked.push(request); return confirmed.value; }, ...model } });
    const state = (extra = {}) => ({ ...(projectId ? {} : COGNITION), main_route: route(main),
        projects: projectId ? [{ id: projectId, chat_id: chatId, cognition: COGNITION, model_choice: choice }] : [], ...extra });
    m.controls.syncState(state());
    const el = m.modelEl;
    const part = (name) => el.querySelector(`.chat-capsule-${name}`);
    const strip = el.querySelector('.chat-model-strip');
    const segs = () => strip.children.filter((node) => node.classList.contains('chat-seg'));
    const words = () => segs().map((seg) => (seg.classList.contains('chat-model-find') ? 'Search' : seg.children[0].textContent));
    const checked = () => segs().filter((seg) => seg.getAttribute('aria-checked') === 'true').map((seg) => seg.children[0].textContent);
    const press = (text) => m.fire(strip, 'click', { target: { closest: (selector) => (selector === '.chat-seg'
        ? segs().find((seg) => (text === 'Search' ? seg.classList.contains('chat-model-find') : seg.children[0].textContent === text)) : null) } });
    const hint = () => Object.fromEntries(['word', 'text', 'fact-word', 'fact-text', 'rule'].map((name) => {
        const node = m.modelHint.querySelector(`.ui-hint-${name}`);
        return [name, node.hidden ? '' : node.textContent];
    }));
    const list = () => m.doc.body.children.find((node) => node.classList.contains('chat-model-list')) || null;
    const input = () => strip.querySelector('.chat-model-query');
    // The suggestions as they are written: each row's key, whether it is the chosen one, and its words.
    const rows = () => [...list().innerHTML.matchAll(/<div class="chat-model-row[^"]*" role="option" data-row="([^"]*)" aria-selected="(true|false)">(.*?)<\/div>/g)]
        .map(([, key, selected, inner]) => ({ key, selected: selected === 'true',
            text: inner.replace(/<svg.*?<\/svg>/g, '').replace(/<[^>]+>/g, ' ').replace(/\s+/g, ' ').trim() }));
    const notes = () => [...list().innerHTML.matchAll(/<p class="chat-model-(?:note|section)"[^>]*>(.*?)<\/p>/g)]
        .map((match) => match[1].replaceAll('&#39;', '\''));
    const type = (text) => { input().value = text; return m.fire(input(), 'input', {}); };
    const key = (name) => {
        const event = { key: name, defaultPrevented: false, preventDefault() { this.defaultPrevented = true; }, stopPropagation() {} };
        return m.fire(input(), 'keydown', event).then(() => event);
    };
    const take = (rowKey) => m.fire(list(), 'click', { target: { closest: () => list().querySelectorAll('.chat-model-row').find((node) => node.dataset.row === rowKey) } });
    const modelPosts = () => m.posts.filter(([url]) => url === '/api/owner/chat-model').map(([, body]) => body);
    return { ...m, ...kept, el, strip, state, asked, confirmed, segs, words, checked, press, hint, list, input, rows, notes, type, key, take, modelPosts,
        name: () => part('name').textContent, value: () => part('value').textContent,
        head: (head) => m.press(head),
        open: async () => { await m.press(m.modelHead); await flush(); } };
}

test('a room follows Main until it chooses, and says so: the name word is Main, the value Main\'s model', () => {
    const r = room();
    assert.deepEqual([r.name(), r.value(), r.el.dataset.known, r.el.dataset.own], ['Main', 'gpt-6-astra', 'true', 'false']);
    assert.match(r.modelHead.getAttribute('aria-label'), /^Model for new work here: gpt-6-astra, Codex\. Follows Main\.$/);
    // Main moves: a follower reads it live.
    r.controls.syncState(r.state({ main_route: route(GLM) }));
    assert.deepEqual([r.name(), r.value()], ['Main', 'glm-5.3']);
    // The room's own model: its source names it, and only the mark is lit.
    r.controls.syncState({ main_route: route(ASTRA), projects: [{ id: 'p7', cognition: COGNITION, model_choice: { kind: 'model', model: GLM, use_local: false } }] });
    assert.deepEqual([r.name(), r.value(), r.el.dataset.own], ['OpenRouter', 'glm-5.3', 'true']);
    assert.match(r.modelHead.getAttribute('aria-label'), /^Model for new work here: glm-5\.3, OpenRouter\.$/);
    r.controls.destroy();
});

test('it is the first capsule of the row, before Subagents, Context and Effort, and one of their family', async () => {
    const r = room();
    assert.equal(r.row.children[0], r.el);
    assert.ok(r.el.classList.contains('chat-capsule') && r.controls.model.capsule.kind === 'model');
    assert.deepEqual(r.controls.model.capsule.peers.map((peer) => peer.kind), ['subagents', 'context', 'effort'], 'one capsule is open at a time');
    await r.open();
    await r.head(r.contextHead);
    assert.deepEqual([r.el.dataset.open, r.contextEl.dataset.open], ['false', 'true'], 'opening another closes it at once');
    r.controls.destroy();
});

test('before Main\'s model is read nothing is named, and a room never borrows another room\'s choice', () => {
    const kept = storeOf();
    const m = mountPair({ projectId: 'p7', chatId: 7, model: { store: kept.store } });
    const value = () => m.modelEl.querySelector('.chat-capsule-value').textContent;
    assert.deepEqual([value(), m.modelEl.dataset.known], ['—', 'false']);
    assert.equal(m.modelHead.getAttribute('aria-label'), 'Model for new work: not read yet');
    m.controls.syncState({ main_route: route(ASTRA), projects: [{ id: 'other', model_choice: { kind: 'model', model: GLM, use_local: false } }] });
    assert.equal(value(), 'gpt-6-astra', 'another room\'s model is never this room\'s');
    // A body without the contract's fields (a server that does not publish them yet) keeps what was read.
    m.controls.syncState({ projects: [] });
    assert.equal(value(), 'gpt-6-astra');
    m.controls.destroy();
});

test('closed it always shows the room\'s choice, never a running work\'s model; open, the hint adds one fact line', async () => {
    const works = [
        { work_id: 'w1', room: 7, live: true, lead: { model: GLM, use_local: false, basis: 'owner' } },
        { work_id: 'w2', room: 7, live: true, lead: { model: ASTRA, use_local: false, basis: 'main' } },
        { work_id: 'w3', room: 7, live: false, lead: { model: KIMI, use_local: false, basis: 'room' } },
        { work_id: 'w4', room: 9, live: true, lead: { model: SOL, use_local: false, basis: 'room' } },
        { work_id: 'w5', room: 7, live: true, lead: { model: GLM, use_local: false, basis: 'fallback' } },
    ];
    const r = room();
    r.controls.syncState(r.state({ works }));
    assert.deepEqual([r.name(), r.value()], ['Main', 'gpt-6-astra'], 'the closed capsule does not change with the works');
    assert.equal(r.el.querySelector('.chat-model-differs'), null, 'and wears no mark for them');
    await r.open();
    assert.deepEqual(r.hint(), { word: 'Main', text: 'gpt-6-astra · Codex · subscription · 872K window.',
        'fact-word': 'Running here', 'fact-text': 'on other models: glm-5.3', rule: MODEL_RULE });
    // A server that does not publish the fact yet: no line, never an error.
    r.controls.syncState(r.state({ works: works.map(({ lead, ...work }) => work) }));
    assert.deepEqual([r.hint()['fact-word'], r.hint()['fact-text']], ['', '']);
    assert.deepEqual(modelsRunningElsewhere(works, 7, route(ASTRA)), ['glm-5.3']);
    assert.deepEqual(modelsRunningElsewhere(works, 9, route(ASTRA)), ['gpt-6.1-sol']);
    assert.deepEqual(modelsRunningElsewhere(undefined, 7, route(ASTRA)), []);
    assert.equal(MODEL_RULE, 'Applies to new work here; running work keeps its model.');
    r.controls.destroy();
});

test('open, the strip is Main, the two most recent models and Search; a stack layout keeps one recent', async () => {
    const recents = [route(GLM), route(ASTRA), route(KIMI), route(SOL)];
    const r = room({ storeOptions: { recents } });
    await r.open();
    assert.deepEqual([r.name(), r.reads.catalog, r.reads.recents], ['New work here', 1, 1], 'opening reads the catalog and the recents once');
    assert.deepEqual(r.words(), ['Main', 'glm-5.3', 'kimi-k3', 'Search'], 'Main\'s own model is the Main segment, never a recent');
    assert.deepEqual(r.checked(), ['Main']);
    assert.equal(r.segs().filter((seg) => seg.tabIndex === 0).length, 1, 'one tab stop');
    r.controls.destroy();
    // The room's own model leads the recents, once.
    const own = room({ choice: { kind: 'model', model: KIMI, use_local: false }, storeOptions: { recents } });
    await own.open();
    assert.deepEqual([own.words(), own.checked()], [['Main', 'kimi-k3', 'glm-5.3', 'Search'], ['kimi-k3']]);
    own.controls.destroy();
    const phone = room({ storeOptions: { recents }, model: { isStack: () => true } });
    await phone.open();
    assert.deepEqual(phone.words(), ['Main', 'glm-5.3', 'Search']);
    phone.controls.destroy();
    // Until the server carries the field, no recents: the strip is Main and Search.
    const bare = room({ storeOptions: { recents: undefined } });
    await bare.open();
    assert.deepEqual(bare.words(), ['Main', 'Search']);
    bare.controls.destroy();
});

test('a room\'s pick saves at once for that room; Main in the strip clears the room\'s choice; Send can wait for it', async () => {
    const r = room({ storeOptions: { recents: [route(GLM)] } });
    await r.open();
    let answer;
    r.modelAnswer.value = new Promise((resolve) => { answer = resolve; });
    const saving = r.press('glm-5.3');
    assert.deepEqual(r.modelPosts(), [{ kind: 'model', model: GLM, use_local: false, project_id: 'p7' }]);
    assert.deepEqual([r.controls.hasPendingSave(), r.el.dataset.disabled, r.checked()], [true, 'true', ['Main']], 'unsaved, it is not shown as saved');
    r.controls.syncState(r.state());
    answer({ ok: true, project_id: 'p7', model_choice: { kind: 'model', model: GLM, use_local: false }, previous_model_choice: { kind: 'main' } });
    await saving;
    assert.equal(await r.controls.model.pendingSave(), true);
    assert.deepEqual([r.checked(), r.el.dataset.disabled, r.el.dataset.open, r.refreshes.at(-1)], [['glm-5.3'], 'false', 'true', true],
        'like its neighbours the strip stays open after a pick');
    assert.equal(r.hint().word, 'glm-5.3');
    assert.equal(r.reads.recents, 2, 'the server wrote the act into the recents: they are read again');
    r.modelAnswer.value = null;
    await r.press('glm-5.3');
    assert.equal(r.modelPosts().length, 1, 'the chosen model again saves nothing');
    await r.press('Main');
    assert.deepEqual(r.modelPosts().at(-1), { kind: 'main', project_id: 'p7' });
    assert.deepEqual([r.checked(), r.el.dataset.own], [['Main'], 'false']);
    r.controls.destroy();
});

test('a refused save keeps the saved value and says why; an answer for another room or without the choice is no save', async () => {
    const r = room({ storeOptions: { recents: [route(GLM)] } });
    await r.open();
    r.modelAnswer.value = Promise.reject(new Error('Not a model route'));
    await r.press('glm-5.3');
    assert.deepEqual([r.checked(), r.toasts.at(-1), await r.controls.model.pendingSave()], [['Main'], ['Not a model route', 'error'], false]);
    r.modelAnswer.value = { ok: true, project_id: 'other', model_choice: { kind: 'model', model: GLM, use_local: false } };
    await r.press('glm-5.3');
    assert.deepEqual([r.checked(), r.toasts.at(-1)[0]], [['Main'], 'The answer belongs to another room.']);
    r.modelAnswer.value = { ok: true, project_id: 'p7' };
    await r.press('glm-5.3');
    assert.deepEqual([r.checked(), r.toasts.at(-1)[0]], [['Main'], 'The server did not confirm the model.']);
    r.controls.destroy();
});

test('in the Main chat the capsule is Main itself: a pick asks first, then writes Main and says what else changed', async () => {
    const r = room({ projectId: '', chatId: 1, storeOptions: { recents: [route(GLM)] } });
    assert.deepEqual([r.name(), r.value(), r.el.dataset.own], ['Main', 'gpt-6-astra', 'false']);
    assert.match(r.modelHead.getAttribute('aria-label'), /^Main model: gpt-6-astra, Codex\.$/);
    await r.open();
    assert.deepEqual([r.name(), r.words(), r.checked()], ['Main model', ['gpt-6-astra', 'glm-5.3', 'Search'], ['gpt-6-astra']]);
    assert.deepEqual([r.hint().word, r.hint().rule], ['Main', MAIN_RULE]);
    r.confirmed.value = false;
    await r.press('glm-5.3');
    assert.deepEqual([r.el.dataset.open, r.modelPosts(), r.value()], ['false', [], 'gpt-6-astra'], 'cancelled: nothing is written');
    assert.deepEqual([r.asked[0].next.short, r.asked[0].next.sourceLabel, r.asked[0].previous.short], ['glm-5.3', 'OpenRouter', 'gpt-6-astra']);
    r.confirmed.value = true;
    r.modelAnswer.value = { ok: true, project_id: '', main_route: route(GLM), previous_main_route: route(ASTRA), account: 'auto', window_reset: true };
    await r.open();
    await r.press('glm-5.3');
    await flush();
    assert.deepEqual(r.modelPosts(), [{ kind: 'model', model: GLM, use_local: false }], 'no project: the write is Main\'s own');
    assert.deepEqual([r.value(), r.toasts.at(-1)], ['glm-5.3', ['Main\'s account is Auto now. Main\'s context window set by hand was cleared.', 'info']]);
    // Nothing else changed: nothing more is said.
    r.modelAnswer.value = { ok: true, project_id: '', main_route: route(ASTRA), previous_main_route: route(GLM), account: 'kept', window_reset: false };
    const toasts = r.toasts.length;
    await r.controls.model.pick(route(ASTRA));
    await flush();
    assert.deepEqual([r.value(), r.toasts.length], ['gpt-6-astra', toasts]);
    r.controls.destroy();
});

test('the confirming sheet says what Main becomes, what running work does, and in one line what else is reset', async () => {
    const calls = [];
    const dialog = async (options) => { calls.push(options); return true; };
    const doc = { createElement: (tag) => new ElementStub(tag, doc), byId: new Map() };
    const next = { short: 'glm-5.3', id: GLM, source: 'openrouter', lane: 'api', kind: 'openrouter', sourceLabel: 'OpenRouter', pay: 'paid by key', window: '1M' };
    const previous = { short: 'gpt-6-astra', source: 'subscription:codex', lane: 'subscription' };
    assert.deepEqual(await confirmMainChange({ next, previous, dialog, doc }), { moves: [] });
    const [sheet] = calls;
    assert.deepEqual([sheet.title, sheet.confirmLabel], ['Change Main', 'Change Main']);
    assert.equal(sheet.body, 'If Main\'s account is pinned it becomes Auto, and a context window set by hand for Main is cleared.');
    const content = doc.createElement('section');
    sheet.content.mount(content);
    assert.match(content.children[0].innerHTML, />z-ai\/glm-5\.3<\/b><span data-i18n-skip>was gpt-6-astra<\/span>$/);
    assert.deepEqual(content.children.slice(1).map((node) => node.textContent),
        ['OpenRouter · paid by key · 1M window.', `${MAIN_RULE} ${MAIN_FOLLOW_NOTE}`,
            'How many running works follow Main is unknown right now: the list of running work is incomplete.']);
    assert.equal(content.children.length, 4, 'no census was given: the count is unknown and there is nothing to list (main_sheet.test.js)');
    // The same subscription keeps Main's account, so only the window is said.
    await confirmMainChange({ next: { ...next, source: 'subscription:codex', lane: 'subscription' }, previous, dialog, doc });
    assert.equal(calls[1].body, 'A context window set by hand for Main is cleared.');
});

test('Search is a field in the row and its suggestions stand outside it: Main, the recents, then the sources', async () => {
    const r = room({ storeOptions: { recents: [route(GLM)] } });
    await r.open();
    await r.press('Search');
    assert.deepEqual([r.el.dataset.search, r.row.dataset.ownerSearch, r.el.dataset.open], ['true', 'true', 'true']);
    assert.equal(r.list().parentNode, r.doc.body, 'the list is not in the composer\'s flow');
    assert.deepEqual([r.input().placeholder, r.doc.activeElement], ['Model or source', r.input()], 'a mouse\'s press focuses the field');
    assert.equal(r.input().getAttribute('aria-controls'), r.list().id);
    assert.deepEqual(r.rows(), [
        { key: `route|${ASTRA}`, selected: true, text: 'gpt-6-astra Main 872K · Codex · subscription' },
        { key: `route|${GLM}`, selected: false, text: 'z-ai/glm-5.3 OpenRouter · paid by key' },
        { key: 'source:subscription:codex', selected: false, text: 'Codex 3 models · subscription' },
        { key: 'source:openrouter', selected: false, text: 'OpenRouter 3 models · paid by key' },
    ]);
    assert.deepEqual(r.notes(), ['Recent', 'Sources']);
    assert.match(r.list().innerHTML, /<p class="chat-model-rule" role="presentation">Applies to new work here; running work keeps its model\.<\/p>$/);
    // A source's own list, and the way back.
    await r.take('source:subscription:codex');
    assert.deepEqual(r.rows().map((row) => row.text), ['All sources Codex', 'gpt-6.1-sol Codex · subscription',
        'gpt-6-astra Main 872K · Codex · subscription', 'gpt-5-old Codex · subscription Unavailable']);
    assert.equal(r.input().placeholder, 'Codex: type a model');
    await r.take('source:');
    assert.equal(r.rows().length, 4);
    r.controls.destroy();
    assert.equal(r.list(), null, 'the list goes with its capsule');
});

test('where the catalog states prices the list reads the processing tiers once: on Fast or Economy it names none', async () => {
    const priced = { ...CATALOG, items: CATALOG.items.map((item) => (item.value === KIMI ? { ...item, price: { input: 0.6, output: 2.5 } } : item)) };
    const kimi = async (settings) => {
        const r = room({ storeOptions: { catalog: priced, settings, connections: [{ id: 'openrouter', enabled: true, configured: true }] } });
        await r.open();
        await flush();
        await r.press('Search');
        await r.type('kimi');
        const said = [r.rows()[0].text, r.reads.settings];
        r.controls.destroy();
        return said;
    };
    assert.deepEqual(await kimi({}), ['moonshotai/kimi-k3 OpenRouter · $0.60 in / $2.50 out', 1], 'a price to qualify: the tier is read once');
    assert.deepEqual(await kimi({ OUROBOROS_PROCESSING_PREFERENCE: 'fast' }), ['moonshotai/kimi-k3 OpenRouter · paid by key', 1]);
    assert.deepEqual(await kimi({ OUROBOROS_MODEL_PROCESSING_PREFERENCES: '{"main":"economy"}' }), ['moonshotai/kimi-k3 OpenRouter · paid by key', 1]);
    // New work here runs as a lead on the default tier, whatever Main's own one is.
    assert.deepEqual(await kimi({ OUROBOROS_PROCESSING_PREFERENCE: 'fast', OUROBOROS_MODEL_PROCESSING_PREFERENCES: '{"main":"standard"}' }),
        ['moonshotai/kimi-k3 OpenRouter · paid by key', 1]);
});

test('Settings\' own document reaches the list without a read, and a failed read is not repeated by every paint', async () => {
    const reads = { settings: 0 };
    const priced = { items: [{ ...CATALOG.items[3], price: { input: 1, output: 2 } }] };
    const store = createModelCatalogStore({ readCatalog: async () => priced, readPreferences: async () => ({}),
        readSettings: async () => { reads.settings += 1; throw new Error('offline'); } });
    await store.read();
    await store.readConnections();
    await store.readConnections();
    assert.equal(reads.settings, 1, 'one failed read, not one per paint');
    const price = () => store.index().row({ model: GLM }).price;
    assert.equal(price(), null, 'a tier never read names no price');
    store.adoptSettings({ OUROBOROS_PROCESSING_PREFERENCE: 'economy' });
    assert.equal(price(), null);
    store.adoptSettings({ OUROBOROS_PROCESSING_PREFERENCE: 'standard' });
    assert.deepEqual(price(), { input: 1, output: 2 });
    assert.equal(reads.settings, 1);
});

test('typing searches every source; a pick saves and closes the capsule; any id can still be used as typed', async () => {
    const r = room();
    await r.open();
    await r.press('Search');
    await r.type('k');
    assert.deepEqual(r.rows().map((row) => row.text), ['moonshotai/kimi-k3 OpenRouter · paid by key',
        'x-ai/grok-4.7 OpenRouter · paid by key', 'Use “k” as typed OpenRouter']);
    await r.type('vendor/brand-new');
    assert.deepEqual(r.rows(), [{ key: 'typed:route|vendor/brand-new', selected: false, text: 'Use “vendor/brand-new” as typed OpenRouter' }]);
    await r.type('kimi');
    assert.equal((await r.key('Enter')).defaultPrevented, true, 'Enter takes the first match');
    await flush();
    assert.deepEqual(r.modelPosts(), [{ kind: 'model', model: KIMI, use_local: false, project_id: 'p7' }]);
    assert.deepEqual([r.el.dataset.open, r.list(), r.row.dataset.ownerSearch, r.name(), r.value()],
        ['false', null, undefined, 'OpenRouter', 'kimi-k3'], 'the closed capsule says what was set');
    // Main's own row follows Main again.
    await r.open();
    await r.press('Search');
    await r.take(`route|${ASTRA}`);
    await flush();
    assert.deepEqual(r.modelPosts().at(-1), { kind: 'main', project_id: 'p7' });
    r.controls.destroy();
});

test('the field\'s keys: arrows move among the rows, Escape clears, then leaves the search, then closes the capsule', async () => {
    const r = room();
    await r.open();
    await r.press('Search');
    await r.type('g');
    const active = () => r.list().querySelectorAll('.chat-model-row').filter((node) => node.dataset.active === 'true').map((node) => node.dataset.row);
    assert.deepEqual(active(), [`route|${SOL}`], 'typing marks the first match');
    await r.key('ArrowDown');
    assert.deepEqual(active(), [`route|${ASTRA}`]);
    assert.equal(r.input().getAttribute('aria-activedescendant'), r.list().querySelectorAll('.chat-model-row')[1].id);
    await r.key('ArrowUp'); await r.key('ArrowUp');
    assert.deepEqual(active(), ['typed:route|g'], 'and wraps');
    assert.equal((await r.key('Escape')).defaultPrevented, true);
    assert.deepEqual([r.input().value, r.el.dataset.search], ['', 'true'], 'Escape first clears what was typed');
    assert.equal((await r.key('Escape')).defaultPrevented, true);
    assert.deepEqual([r.el.dataset.search, r.el.dataset.open, r.list(), r.words()], [undefined, 'true', null, ['Main', 'Search']], 'then leaves the search for the strip');
    assert.equal(r.doc.activeElement, r.segs().at(-1), 'with the focus on Search');
    const escape = { key: 'Escape', defaultPrevented: false, preventDefault() { this.defaultPrevented = true; } };
    r.doc.fire('keydown', escape);
    assert.equal(r.el.dataset.open, 'false', 'the capsule\'s own Escape closes it');
    r.controls.destroy();
});

test('a press on the suggestions is a press in the capsule; a press elsewhere closes it and takes the list away', async () => {
    const r = room({ hover: true });
    await r.open();
    await r.press('Search');
    r.docDown(r.list());
    assert.deepEqual([r.el.dataset.open, r.el.dataset.search], ['true', 'true']);
    r.docDown(new ElementStub('div', r.doc));
    assert.deepEqual([r.el.dataset.open, r.el.dataset.search, r.list()], ['false', undefined, null]);
    r.controls.destroy();
});

test('under a finger, opening the capsule and pressing Search raise no keyboard: the field takes focus on its own press', async () => {
    const r = room({ model: { isCoarse: () => true } });
    await r.open();
    await r.press('Search');
    assert.ok(r.input() && r.list(), 'the field and the list are there');
    assert.notEqual(r.doc.activeElement, r.input());
    r.controls.destroy();
});

test('every count the capsule prints agrees with its noun: one model, several models, and how many more', async () => {
    const many = Array.from({ length: 70 }, (_, i) => openrouter(`vendor/model-${String(i).padStart(2, '0')}`));
    const r = room({ storeOptions: { catalog: { model_sources: CATALOG.model_sources,
        items: [codex('gpt-6-astra'), ...many, { provider_id: 'openai', provider: 'OpenAI', id: 'gpt-6', name: 'gpt-6', value: 'openai::gpt-6' }] } } });
    await r.open();
    await r.press('Search');
    assert.deepEqual(r.rows().slice(1).map((row) => row.text),
        ['Codex 1 model · subscription', 'OpenRouter 70 models · paid by key', 'OpenAI 1 model · paid by key']);
    await r.take('source:openrouter');
    assert.equal(r.rows().length, 61, 'the way back and sixty models');
    assert.deepEqual(r.notes(), ['10 more · keep typing'], 'the rest is a number with no noun to disagree with');
    r.controls.destroy();
});

test('one source is no source step; nothing but Main says so and offers the way to Settings › Accounts and keys', async () => {
    const one = room({ storeOptions: { catalog: { model_sources: CATALOG.model_sources, items: CATALOG.items.slice(0, 3) } } });
    await one.open();
    await one.press('Search');
    assert.deepEqual(one.rows().map((row) => row.text), ['gpt-6-astra Main 872K · Codex · subscription', 'gpt-6.1-sol Codex · subscription',
        'gpt-5-old Codex · subscription Unavailable']);
    assert.deepEqual(one.notes(), ['Codex'], 'the source names its models, without a Sources step');
    await one.type('gpt-9');
    assert.deepEqual(one.rows().at(-1), { key: 'typed:route|claudexor::codex=gpt-9', selected: false, text: 'Use “gpt-9” as typed Codex' });
    one.controls.destroy();

    const alone = room({ storeOptions: { catalog: { model_sources: CATALOG.model_sources, items: [codex('gpt-6-astra')] } } });
    await alone.open();
    await alone.press('Search');
    assert.deepEqual(alone.notes(), ['Only Main\'s model can lead work here.']);
    assert.deepEqual(alone.rows().map((row) => row.key), [`route|${ASTRA}`, 'accounts']);
    assert.equal(alone.rows()[1].text, 'Add a subscription or a key in Settings › Accounts and keys');
    await alone.take('accounts');
    assert.deepEqual([alone.settingsOpened, alone.el.dataset.open], [['accounts'], 'false']);
    alone.controls.destroy();

    const failed = room({ storeOptions: { catalog: new Error('offline') } });
    await failed.open();
    await failed.press('Search');
    assert.deepEqual(failed.notes(), ['The model catalog could not be read.']);
    failed.controls.destroy();
});

test('a saved choice whose key cannot send keeps Settings\' words; the settings document is read only then', async () => {
    const connections = [{ id: 'openrouter', enabled: true, configured: true }, { id: 'deepseek', enabled: false, configured: true }];
    const r = room({ choice: { kind: 'model', model: 'deepseek::deepseek-v4', use_local: false }, storeOptions: { connections } });
    assert.deepEqual([r.name(), r.reads.settings], ['DeepSeek', 0], 'nothing is read for a closed capsule');
    await r.open();
    await flush();
    assert.equal(r.reads.settings, 1);
    assert.equal(r.hint().text, 'DeepSeek (off) · paid by key.');
    await r.head(r.modelHead);
    assert.deepEqual([r.name(), r.value()], ['DeepSeek (off)', 'deepseek-v4']);
    r.controls.destroy();
    const fine = room({ choice: { kind: 'model', model: GLM, use_local: false } });
    await fine.open();
    await flush();
    assert.equal(fine.reads.settings, 0, 'a source the catalog lists needs no settings read');
    fine.controls.destroy();
});
