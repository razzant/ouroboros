// The one list of models (model_list.js; DESIGN "Composer owner controls"), the body of the Model sheet in a composer
// and on a work's card. Without a query: Main's row (the way to follow Main, or Main itself in the Main chat), the
// pinned models, the recent ones, then the sources (one: its models at once; more: folded under `All sources`). It is a
// grid the field drives: arrows move among rows and between a row's cells, Enter presses the active cell, a pointer
// presses either; a Pin chooses nothing. The sheet around it, its tier strip and the saves are composer_model.test.js
// and card_model.test.js; the markup is built into nodes here (dom_double.js `parseHtml`) so a row has real cells.
import assert from 'node:assert/strict';
import test from 'node:test';
import { buildModelIndex, describeRoute, sameRoute } from '../modules/chat_model_routes.js';
import { createModelList, modelSearchField, routeFacts } from '../modules/model_list.js';
import { modelSheetOutline, modelSheetRows } from './composer_controls_fixture.js';
import { domDouble } from './dom_double.js';

const route = (model, local = false) => ({ model, use_local: local });
const item = (id, provider = 'openai', extra = {}) => ({ id, name: id, provider_id: provider,
    value: provider === 'openrouter' ? id : `${provider}::${id}`, ...extra });
const codex = (id, extra = {}) => ({ provider_id: 'claudexor', provider: 'Codex', source_id: 'codex', id, name: id,
    value: `claudexor::codex=${id}`, credential_profile_id: 'home', availability: 'available', ...extra });

// One source: its models stand at once under its name.
const MAIN = route('openai::model-main');
const RECENT = route('openai::model-recent');
const PINNED = route('openai::model-pinned');
const OTHER = route('openai::model-other');
const GONE = route('openai::model-gone');  // pinned once; the catalog no longer lists it
const ONE_SOURCE = { items: [item('model-main'), item('model-recent'), item('model-pinned'), item('model-other')] };
// Four sources: a subscription, two keys and the local model.
const ASTRA = route('claudexor::codex=gpt-6-astra');
const KIMI = route('moonshotai/kimi-k3');
const SOURCES = { model_sources: [{ id: 'codex', label: 'Codex', credentialHarness: 'codex' }],
    items: [codex('gpt-6-astra'), codex('gpt-5-old', { availability: 'unavailable' }), item('z-ai/glm-5.3', 'openrouter'),
        item('moonshotai/kimi-k3', 'openrouter'), item('gpt-6'),
        { provider_id: 'local', provider: 'Local', id: 'qwen3.gguf', name: 'qwen3.gguf', value: 'local::qwen3.gguf', use_local: true, context_window: 32768 }] };

/* The window's catalog store as the list reads it: the index, the read status and the owner's pins. */
function storeOf({ catalog = ONE_SOURCE, pins = [], status = { read: true } } = {}) {
    const index = buildModelIndex({ catalog });
    const store = { pinned: [...pins], index: () => index, status: () => status, pins: () => store.pinned,
        isPinned: (seen) => store.pinned.some((pinned) => sameRoute(pinned, seen)) };
    return store;
}

/* The field and the grid on a page, the list driving them, and every press it hands back. */
function mountList({ store = storeOf(), main = MAIN, current = main, recents = [], follows = true, ...options } = {}) {
    const { doc } = domDouble({ parseHtml: true });
    const input = doc.createElement('input');
    const grid = doc.createElement('div');
    doc.body.append(input, grid);
    const said = { routes: [], pins: [], leaves: [], settings: 0, paints: 0 };
    const state = { main, current, recents, follows };
    const list = createModelList({ el: grid, input, id: 'models', store,
        main: () => state.main, current: () => state.current, recents: () => state.recents, follows: () => state.follows,
        onRoute: (picked) => said.routes.push(picked), onPin: (pinned, on) => said.pins.push([pinned, on]),
        onSettings: () => { said.settings += 1; }, onLeave: (how) => said.leaves.push(how), onPaint: () => { said.paints += 1; },
        ...options });
    list.paint();
    const row = (key) => grid.querySelectorAll('.chat-model-row').find((node) => node.dataset.row === key) || null;
    return {
        doc, grid, input, list, store, state, said, row,
        outline: () => modelSheetOutline(grid.innerHTML),
        rows: () => modelSheetRows(grid.innerHTML),
        entry: (key) => modelSheetRows(grid.innerHTML).find((entry) => entry.key === key),
        key: (name, extra = {}) => input.emit('keydown', { key: name, ...extra }),
        type: (text) => { input.value = text; input.emit('input'); },
        press: (key) => row(key).querySelector('.chat-model-cell').click(),
        pin: (key) => row(key).querySelector('.chat-model-pin').click(),
        // The active row and its active cell, and the cell the field names as active.
        active: () => {
            const at = grid.querySelectorAll('.chat-model-row').find((node) => node.dataset.active === 'true');
            return at ? [at.dataset.row, at.dataset.activeCell] : null;
        },
        named: () => input.getAttribute('aria-activedescendant'),
    };
}
const key = (seen) => `route|${seen.model}`;

test('the field is a combobox over the grid; the grid names itself and every cell of a row', () => {
    const { doc } = domDouble({ parseHtml: true });
    const { label, input } = modelSearchField(doc, 'models-list');
    assert.ok(label.classList.contains('chat-model-search') && label.contains(input));
    assert.ok(input.classList.contains('chat-model-query'));
    assert.deepEqual(['role', 'aria-haspopup', 'aria-controls', 'aria-expanded', 'aria-autocomplete', 'aria-label', 'autocomplete', 'spellcheck']
        .map((name) => input.getAttribute(name)), ['combobox', 'grid', 'models-list', 'true', 'list', 'Search models', 'off', 'false']);
    assert.equal(input.dataset.i18nSkip, '', 'its placeholder may name a source: the overlay leaves it to the list');
    const l = mountList();
    assert.deepEqual([l.grid.getAttribute('role'), l.grid.getAttribute('aria-label'), l.grid.id], ['grid', 'Models', 'models-list']);
    assert.equal(l.input.placeholder, 'Model or source');
    const rowNode = l.row(key(OTHER));
    assert.equal(rowNode.getAttribute('role'), 'row');
    assert.deepEqual(rowNode.querySelectorAll('.chat-model-gridcell').map((cell) => [cell.getAttribute('role'), cell.className]),
        [['gridcell', 'chat-model-gridcell chat-model-cell'], ['gridcell', 'chat-model-gridcell chat-model-pin-cell']]);
    const at = l.rows().findIndex((entry) => entry.key === key(OTHER));
    assert.deepEqual(rowNode.querySelectorAll('.chat-model-gridcell').map((cell) => cell.id), [`models-r${at}c0`, `models-r${at}c1`]);
    const pin = rowNode.querySelector('.chat-model-pin');
    assert.deepEqual([pin.getAttribute('aria-pressed'), pin.getAttribute('aria-label'), pin.getAttribute('title'), pin.tabIndex],
        ['false', 'Pin model-other', 'Pin', -1], 'a Pin is a toggle the field reaches, never a tab stop of its own');
    // A grid that has an id keeps it.
    const own = domDouble({ parseHtml: true }).doc.createElement('div');
    own.id = 'kept';
    createModelList({ el: own, input: own.ownerDocument.createElement('input'), id: 'x', store: storeOf(), main: () => MAIN,
        current: () => MAIN, recents: () => [], onRoute() {} });
    assert.equal(own.id, 'kept');
});

test('without a query a room\'s list is Main to follow, the pinned models, the recent ones, then its one source\'s models', () => {
    const l = mountList({ store: storeOf({ pins: [PINNED, GONE] }), recents: [RECENT, PINNED] });
    assert.deepEqual(l.outline(), [
        'main *',
        '§ Pinned', `${key(PINNED)} +pin`, `${key(GONE)} +pin`,
        '§ Recent', `${key(RECENT)} -pin`,
        '§ OpenAI', `${key(MAIN)} -pin`, `${key(OTHER)} -pin`,
    ], 'a model is listed once; Main\'s own model is a model row of its source too');
    assert.deepEqual(l.entry('main'), { key: 'main', kind: 'main', selected: true, name: 'Main', tags: [],
        facts: 'Follows Main · model-main', pin: null }, 'Main\'s row follows Main: it has no Pin of its own');
    assert.deepEqual([l.entry(key(GONE)).name, l.entry(key(GONE)).facts], ['model-gone', 'OpenAI · paid by key'],
        'a pinned model the catalog does not list stays in view, said from its route');
    assert.match(l.grid.innerHTML, /<p class="chat-model-section" role="presentation">Pinned<\/p>/);
    assert.match(l.grid.innerHTML, /<p class="chat-model-section" role="presentation">Recent<\/p>/);
    assert.match(l.grid.innerHTML, /<p class="chat-model-section" role="presentation" data-i18n-skip>OpenAI<\/p>/,
        'a source\'s name is not translated; the headings are');
    // The caller's facts for Main's row say what a press does there (a work's card).
    const card = mountList({ mainFacts: () => 'this work will follow Main', follows: false });
    assert.deepEqual([card.entry('main').facts, card.entry('main').selected], ['this work will follow Main', false]);
    // A room on its own model: that row is the chosen one, Main's is not.
    const own = mountList({ follows: false, current: OTHER });
    assert.deepEqual(own.rows().filter((entry) => entry.selected).map((entry) => entry.key), [key(OTHER)]);
    // Without pins and recents there is no heading for them.
    assert.deepEqual(mountList().outline(), ['main *', '§ OpenAI', `${key(MAIN)} -pin`, `${key(OTHER)} -pin`,
        `${key(PINNED)} -pin`, `${key(RECENT)} -pin`]);
});

test('in the Main chat Main\'s row is Main\'s own model, chosen, with its Pin, and stands nowhere else', () => {
    const l = mountList({ mainRow: 'main', follows: false, store: storeOf({ pins: [PINNED] }), recents: [MAIN, RECENT] });
    assert.deepEqual(l.outline(), ['main * -pin', '§ Pinned', `${key(PINNED)} +pin`, '§ Recent', `${key(RECENT)} -pin`,
        '§ OpenAI', `${key(OTHER)} -pin`]);
    assert.deepEqual(l.entry('main'), { key: 'main', kind: 'main', selected: true, name: 'model-main', tags: ['Main'],
        facts: 'OpenAI · paid by key', pin: false });
    l.pin('main');
    assert.deepEqual(l.said.pins, [[MAIN, true]], 'Main\'s own model can be pinned from its row');
    l.press('main');
    assert.deepEqual(l.said.routes, [null], 'its row is Main: a press is no other model');
    // No Main row at all: Main's model is a row of its source like any other.
    const bare = mountList({ mainRow: false });
    assert.deepEqual(bare.outline(), ['§ OpenAI', `${key(MAIN)} -pin`, `${key(OTHER)} -pin`, `${key(PINNED)} -pin`, `${key(RECENT)} -pin`]);
});

test('a heading stands only over rows: Main\'s pinned model already on Main\'s row leaves no empty Pinned', () => {
    const l = mountList({ mainRow: 'main', follows: false, store: storeOf({ pins: [MAIN] }) });
    assert.deepEqual(l.outline(), ['main * +pin', '§ OpenAI', `${key(OTHER)} -pin`, `${key(PINNED)} -pin`, `${key(RECENT)} -pin`]);
});

test('several sources fold under All sources; opened, each is a step into its own models, and Escape comes back', () => {
    const l = mountList({ store: storeOf({ catalog: SOURCES }), main: ASTRA });
    assert.deepEqual(l.outline(), ['main *', 'sources']);
    assert.deepEqual(l.entry('sources'), { key: 'sources', kind: 'sources', selected: false, name: 'All sources', tags: [],
        facts: 'Codex, OpenRouter + 2', pin: null, expanded: false }, 'two sources named, the rest counted');
    l.press('sources');
    assert.equal(l.entry('sources').expanded, true);
    assert.deepEqual(l.rows().slice(2).map((entry) => [entry.key, entry.kind, entry.name, entry.facts, entry.pin]), [
        ['source:subscription:codex', 'source', 'Codex', '2 models · subscription', null],
        ['source:openrouter', 'source', 'OpenRouter', '2 models · paid by key', null],
        ['source:openai', 'source', 'OpenAI', '1 model · paid by key', null],
        ['source:local', 'source', 'Local', '1 model · on this computer', null],
    ]);
    l.press('source:openrouter');
    assert.deepEqual(l.outline(), ['source:', `${key(KIMI)} -pin`, 'route|z-ai/glm-5.3 -pin'], 'its models, newer first, and the way back');
    assert.deepEqual([l.entry('source:').name, l.entry('source:').facts, l.input.placeholder],
        ['All sources', 'OpenRouter', 'OpenRouter: type a model']);
    assert.equal(l.key('Escape').defaultPrevented, true);
    assert.deepEqual([l.outline().slice(0, 2), l.said.leaves, l.input.placeholder], [['main *', 'sources'], [], 'Model or source'],
        'Escape leaves the source first; the fold stays open');
    assert.equal(l.entry('sources').expanded, true);
    l.press('source:subscription:codex');
    assert.deepEqual(l.rows().slice(1).map((entry) => [entry.key, entry.tags]), [[key(ASTRA), []], ['route|claudexor::codex=gpt-5-old', ['Unavailable']]]);
    l.press('source:');
    l.press('sources');
    assert.deepEqual(l.outline(), ['main *', 'sources'], 'pressed again, the fold closes');
    assert.equal(l.key('Escape').defaultPrevented, true);
    assert.deepEqual(l.said.leaves, [{ focus: true }], 'with nothing typed or chosen Escape leaves the list to its home');
    // Three sources are all named; a source whose list was not read says so.
    const three = mountList({ main: ASTRA, store: storeOf({ catalog: { ...SOURCES, items: SOURCES.items.slice(0, 5),
        errors: [{ provider_id: 'claudexor', source_id: 'codex', error: 'sign in again' }] } }) });
    assert.equal(three.entry('sources').facts, 'Codex, OpenRouter, OpenAI');
    three.press('sources');
    assert.equal(three.entry('source:subscription:codex').facts, '2 models · subscription · list not read');
});

test('the field drives the grid: Up and Down move among rows, Right and Left between a row\'s cells, Enter presses the active cell', () => {
    const l = mountList({ store: storeOf({ pins: [PINNED, GONE] }), recents: [RECENT] });
    const rows = l.rows().map((entry) => entry.key);
    assert.deepEqual(rows, ['main', key(PINNED), key(GONE), key(RECENT), key(MAIN), key(OTHER)]);
    assert.deepEqual([l.active(), l.input.hasAttribute('aria-activedescendant')], [null, false], 'nothing is active before a key');
    assert.equal(l.key('ArrowRight').defaultPrevented, undefined, 'with no active row the arrows move the caret');
    assert.equal(l.key('ArrowDown').defaultPrevented, true);
    assert.deepEqual([l.active(), l.named()], [['main', '0'], 'models-r0c0']);
    l.key('ArrowRight');
    assert.deepEqual(l.active(), ['main', '0'], 'Main\'s row has one cell');
    l.key('ArrowDown');
    l.key('ArrowRight');
    assert.deepEqual([l.active(), l.named()], [[key(PINNED), '1'], 'models-r1c1']);
    assert.ok(l.doc.getElementById(l.named()).classList.contains('chat-model-pin-cell'), 'the field names the Pin\'s cell');
    l.key('ArrowRight');
    assert.deepEqual(l.active(), [key(PINNED), '1'], 'the last cell is the last');
    const enter = l.key('Enter');
    assert.deepEqual([l.said.pins, l.said.routes, enter.defaultPrevented], [[[PINNED, false]], [], true], 'Enter on a Pin pins, it picks nothing');
    l.key('ArrowDown');
    assert.deepEqual(l.active(), [key(GONE), '1'], 'Down keeps the column');
    l.key('ArrowUp');
    l.key('ArrowUp');
    assert.deepEqual(l.active(), ['main', '0'], 'a row without that cell takes its first');
    l.key('ArrowDown');
    assert.deepEqual(l.active(), [key(PINNED), '0']);
    l.key('ArrowLeft');
    assert.deepEqual(l.active(), [key(PINNED), '0']);
    l.key('Enter');
    assert.deepEqual(l.said.routes, [PINNED]);
    l.key('ArrowUp');
    l.key('ArrowUp');
    assert.deepEqual(l.active(), [key(OTHER), '0'], 'Up from the first row wraps to the last');
    l.key('ArrowDown');
    assert.deepEqual(l.active(), ['main', '0'], 'and Down from the last to the first');
    l.key('Enter');
    assert.deepEqual(l.said.routes, [PINNED, null], 'Enter on Main\'s row follows Main');
    const composing = l.key('ArrowDown', { isComposing: true });
    assert.deepEqual([l.active(), composing.defaultPrevented], [['main', '0'], undefined], 'a composition owns its keys');
    assert.equal(l.key('Tab').defaultPrevented, undefined, 'Tab is the page\'s');
    assert.equal(l.key('Escape').defaultPrevented, true);
    assert.deepEqual(l.said.leaves, [{ focus: true }]);
});

test('a pointer presses a row to pick its model and a Pin to pin it; a Pin\'s press goes no further', () => {
    const l = mountList({ store: storeOf({ pins: [PINNED] }), recents: [RECENT] });
    const reached = [];
    l.doc.addEventListener('click', (event) => reached.push(event.target));
    l.pin(key(RECENT));
    l.row(key(PINNED)).querySelector('.chat-model-pin path').click();
    assert.deepEqual([l.said.pins, l.said.routes, reached], [[[RECENT, true], [PINNED, false]], [], []],
        'its glyph is the Pin too; the row is not picked and the press stops there');
    l.press(key(RECENT));
    assert.deepEqual(l.said.routes, [RECENT]);
    l.press(key(MAIN));
    l.press('main');
    assert.deepEqual(l.said.routes, [RECENT, MAIN, null], 'Main\'s model as a row is that model; only Main\'s row follows Main');
    assert.equal(reached.length, 3, 'a row\'s press is the page\'s too');
    // A press on the list keeps the caret in the field.
    const down = l.row(key(RECENT)).querySelector('.chat-model-row-id').emit('mousedown');
    assert.equal(down.defaultPrevented, true);
    assert.equal(l.grid.querySelector('.chat-model-section').emit('mousedown').defaultPrevented, undefined, 'a heading is no row');
    // A Pin's state rides its row.
    assert.deepEqual([l.row(key(PINNED)).dataset.pinned, l.row(key(RECENT)).dataset.pinned], ['true', undefined]);
    assert.equal(l.row(key(PINNED)).querySelector('.chat-model-pin').getAttribute('title'), 'Unpin');
});

test('a catalog price is said only where the row\'s context pays the catalog\'s Standard price', () => {
    const OPUS = item('anthropic/claude-opus-5.5', 'openrouter', { price: { input: 4, output: 20 } });
    const GPT = item('openai/gpt-6', 'openrouter', { price: { input: 1.25, output: 10 } });
    const store = storeOf({ catalog: { items: [OPUS, GPT, item('z-ai/glm-5.3', 'openrouter')] } });
    const opus = route(OPUS.value);
    const facts = (tierOf) => {
        const asked = [];
        const l = mountList({ store, main: opus, mainRow: 'main', follows: false,
            tierOf: (seen, options) => { asked.push([seen.model, options]); return tierOf(seen, options); } });
        return { facts: Object.fromEntries(l.rows().map((entry) => [entry.key, entry.facts])), asked };
    };
    const standard = facts(() => 'standard');
    assert.deepEqual(standard.facts, { main: 'OpenRouter · $4 in / $20 out', 'route|openai/gpt-6': 'OpenRouter · $1.25 in / $10 out',
        'route|z-ai/glm-5.3': 'OpenRouter · paid by key' });
    assert.deepEqual(standard.asked.find(([model]) => model === OPUS.value), [OPUS.value, { main: true }],
        'Main\'s row asks the tier of what follows Main');
    assert.deepEqual(standard.asked.find(([model]) => model === GPT.value), [GPT.value, {}]);
    for (const tier of ['fast', 'economy', 'ultrafast']) {
        assert.deepEqual(Object.values(facts(() => tier).facts), Array(3).fill('OpenRouter · paid by key'), `${tier} names no catalog price`);
    }
    assert.equal(facts(() => '').facts.main, 'OpenRouter · Catalog rate · $4 in / $20 out');
    // Each row in its own context: Main on Fast, a lead on Standard.
    assert.deepEqual(facts((_seen, { main = false } = {}) => (main ? 'fast' : 'standard')).facts, { ...standard.facts, main: 'OpenRouter · paid by key' });
    assert.equal(routeFacts(describeRoute(store.index(), opus)), 'OpenRouter · $4 in / $20 out');
});

test('typed words search every source; what was typed can be used as it is, through the one source or the chosen one', () => {
    const l = mountList({ store: storeOf({ catalog: SOURCES }), main: ASTRA });
    l.type('kimi');
    assert.deepEqual(l.outline(), [`${key(KIMI)} -pin`, 'typed:route|kimi'], 'Main, the pins and the sources wait for an empty field');
    assert.deepEqual(l.entry('typed:route|kimi'), { key: 'typed:route|kimi', kind: 'typed', selected: false,
        name: 'Use “kimi” as typed', tags: [], facts: 'OpenRouter', pin: null }, 'a bare id is OpenRouter\'s');
    assert.deepEqual([l.active(), l.named()], [[key(KIMI), '0'], 'models-r0c0'], 'typing marks the first match');
    l.key('Enter');
    assert.deepEqual(l.said.routes, [KIMI]);
    l.type('vendor/brand-new');
    assert.deepEqual(l.outline(), ['typed:route|vendor/brand-new']);
    l.key('Enter');
    assert.deepEqual(l.said.routes.at(-1), route('vendor/brand-new'));
    // Escape clears what was typed first, then leaves.
    l.key('Escape');
    assert.deepEqual([l.input.value, l.outline(), l.said.leaves], ['', ['main *', 'sources'], []]);
    // Through a chosen source the typed id is that source's.
    l.press('sources');
    l.press('source:subscription:codex');
    l.type('gpt-9');
    assert.deepEqual(l.outline(), ['source:', 'typed:route|claudexor::codex=gpt-9']);
    assert.equal(l.entry('typed:route|claudexor::codex=gpt-9').facts, 'Codex');
    // One source needs no source step; without OpenRouter a bare id names no source, and the list says how to go on.
    const one = mountList({ store: storeOf({ catalog: { model_sources: SOURCES.model_sources, items: SOURCES.items.slice(0, 2) } }), main: ASTRA });
    one.type('gpt-9');
    assert.deepEqual(one.outline(), ['typed:route|claudexor::codex=gpt-9']);
    const keyless = mountList({ store: storeOf({ catalog: { model_sources: SOURCES.model_sources, items: [codex('gpt-6-astra'), item('gpt-6')] } }), main: ASTRA });
    keyless.type('nothing-like-it');
    assert.deepEqual(keyless.outline(), ['note: No model matches. Choose a source to use a model as typed.']);
    keyless.key('Enter');
    assert.deepEqual(keyless.said.routes, [], 'Enter with nothing to take takes nothing');
});

test('typed in a chosen source, Enter takes its first model, never the way back to all sources', () => {
    const l = mountList({ store: storeOf({ catalog: SOURCES }), main: ASTRA });
    l.press('sources');
    l.press('source:subscription:codex');
    l.type('gpt');
    l.key('Enter');
    assert.deepEqual([l.said.routes, l.input.value], [[ASTRA], 'gpt']);
});

test('nothing but Main to lead says so and offers the way to Settings; a read in flight and a failed one say so too', () => {
    const empty = { model_sources: SOURCES.model_sources, items: [] };
    const l = mountList({ store: storeOf({ catalog: empty }), main: ASTRA });
    assert.deepEqual(l.outline(), ['main *', 'note: Only Main\'s model can lead work here.', 'accounts']);
    assert.deepEqual(l.entry('accounts'), { key: 'accounts', kind: 'door', selected: false,
        name: 'Add a subscription or a key in Settings › Accounts and keys', tags: [], facts: '', pin: null });
    l.press('accounts');
    assert.deepEqual([l.said.settings, l.said.routes], [1, []]);
    assert.deepEqual(mountList({ store: storeOf({ catalog: {}, status: { read: false, reading: true } }), main: ASTRA }).outline(),
        ['main *', 'note: Reading models…']);
    assert.deepEqual(mountList({ store: storeOf({ catalog: {}, status: { read: true, failed: true } }), main: ASTRA }).outline(),
        ['main *', 'note: The model catalog could not be read.', 'accounts']);
    // Searching keeps what was listed and says what could not be read.
    const failed = mountList({ store: storeOf({ catalog: SOURCES, status: { read: true, failed: true } }), main: ASTRA });
    failed.type('kimi');
    assert.equal(failed.outline().at(-1), 'note: The model catalog could not be read; earlier suggestions are kept.');
    const gap = mountList({ store: storeOf({ catalog: { ...SOURCES, errors: [{ provider_id: 'anthropic', error: 'HTTP 401' }] } }), main: ASTRA });
    gap.type('kimi');
    assert.equal(gap.outline().at(-1), 'note: Could not read every list: Anthropic.');
    const engine = mountList({ store: storeOf({ catalog: { ...SOURCES,
        errors: [{ provider_id: 'claudexor', error: 'Subscription catalogs did not answer in time.' }] } }), main: ASTRA });
    assert.equal(engine.outline().at(-1), 'note: Subscription catalogs did not answer in time.');
    engine.type('kimi');
    assert.equal(engine.outline().at(-1), 'note: Subscription catalogs did not answer in time.');
});

test('every count agrees with its noun: one model, several models, and how many more', () => {
    const many = Array.from({ length: 70 }, (_, i) => item(`vendor/model-${String(i).padStart(2, '0')}`, 'openrouter'));
    const l = mountList({ store: storeOf({ catalog: { model_sources: SOURCES.model_sources, items: [codex('gpt-6-astra'), ...many, item('gpt-6')] } }), main: ASTRA });
    l.press('sources');
    assert.deepEqual(l.rows().slice(2).map((entry) => entry.facts), ['1 model · subscription', '70 models · paid by key', '1 model · paid by key']);
    l.press('source:openrouter');
    assert.equal(l.rows().length, 61, 'the way back and sixty models');
    assert.equal(l.outline().at(-1), '§ 10 more · keep typing', 'the rest is a number with no noun to disagree with');
});

test('a repaint that changes nothing writes nothing; one that does keeps the rows\' scroll and names a cell that is there', () => {
    const l = mountList({ recents: [RECENT] });
    l.key('ArrowDown');
    l.key('ArrowDown');
    const node = l.row(key(RECENT));
    const paints = l.said.paints;
    l.list.paint();
    assert.ok(l.row(key(RECENT)) === node, 'the same row nodes');
    assert.deepEqual([l.active(), l.said.paints], [[key(RECENT), '0'], paints + 1], 'every repaint is said to the sheet');
    const group = l.grid.querySelector('.chat-model-rows');
    group.scrollTop = 120;
    l.store.pinned.push(OTHER);  // a pin made in another window arrives
    l.list.paint();
    assert.ok(l.grid.querySelector('.chat-model-rows') !== group, 'written again');
    assert.equal(l.grid.querySelector('.chat-model-rows').scrollTop, 120, 'where it was scrolled');
    assert.equal(l.entry(key(OTHER)).pin, true);
    const named = l.doc.getElementById(l.named());
    assert.ok(named && l.grid.querySelector('.chat-model-rows').contains(named), 'the field never names a cell that was written over');
    assert.equal(named.closest('.chat-model-row').dataset.active, 'true');
});

test('the active row is its model\'s: another window\'s pin keeps Enter on it, and a model that left the list leaves none', () => {
    const l = mountList({ recents: [RECENT] });
    l.key('ArrowDown');
    l.key('ArrowDown');
    assert.deepEqual(l.active(), [key(RECENT), '0']);
    l.store.pinned.push(OTHER);  // another window pins a model: a Pinned section now stands above the recent row
    l.list.paint();
    assert.deepEqual(l.active(), [key(RECENT), '0'], 'the same model, though its place moved down');
    assert.ok(l.doc.getElementById(l.named()).closest('.chat-model-row') === l.row(key(RECENT)));
    l.key('Enter');
    assert.deepEqual(l.said.routes, [RECENT], 'Enter takes the model the keys chose, never what took its place');
    // The active model leaves the list (unpinned in another window, listed by no source): nothing stays active.
    const p = mountList({ store: storeOf({ pins: [PINNED, GONE] }) });
    for (let i = 0; i < 3; i += 1) p.key('ArrowDown');
    assert.deepEqual(p.active(), [key(GONE), '0']);
    p.store.pinned.splice(1, 1);
    p.list.paint();
    assert.deepEqual([p.active(), p.input.hasAttribute('aria-activedescendant')], [null, false]);
    p.key('Enter');
    assert.deepEqual(p.said.routes, [], 'Enter takes nothing rather than the next row');
    p.key('ArrowDown');
    assert.deepEqual(p.active(), ['main', '0'], 'the keys start again from the top');
});

test('typed words leave the arrows to the caret until Up or Down chooses a row; a repaint never scrolls the list back', () => {
    const l = mountList({ store: storeOf({ catalog: SOURCES }), main: ASTRA });
    l.type('kimi');
    assert.deepEqual(l.active(), [key(KIMI), '0'], 'typing marks the first match');
    const moves = [l.key('ArrowRight'), l.key('ArrowLeft'), l.key('ArrowRight')].map((event) => event.defaultPrevented);
    assert.deepEqual([moves, l.active()], [[undefined, undefined, undefined], [key(KIMI), '0']], 'the caret moves to mend the id');
    l.key('Enter');
    assert.deepEqual([l.said.routes, l.said.pins], [[KIMI], []], 'Enter after the arrows picks the model, never its Pin');
    // Down and Up choose the row: now Right reaches its Pin.
    l.key('ArrowDown');
    l.key('ArrowUp');
    assert.equal(l.key('ArrowRight').defaultPrevented, true);
    assert.deepEqual(l.active(), [key(KIMI), '1']);
    l.type('kimi-k');  // typing again gives the arrows back to the caret
    assert.equal(l.key('ArrowRight').defaultPrevented, undefined);
    // Only a change of row or cell scrolls: a state read's repaint leaves a scrolled list where it is.
    const scrolled = [];
    for (const node of l.grid.querySelectorAll('.chat-model-row')) node.scrollIntoView = () => scrolled.push(node.dataset.row);
    l.list.paint();
    l.list.paint();
    assert.deepEqual(scrolled, [], 'the same row and cell: no scroll');
    l.key('ArrowDown');
    assert.equal(scrolled.length, 1, 'a key that moves the row scrolls it into view');
});

test('destroy takes the list\'s listeners off the field and the grid', () => {
    const l = mountList();
    assert.deepEqual([l.input.listenerCount(), l.grid.listenerCount()], [2, 2]);
    l.list.destroy();
    assert.deepEqual([l.input.listenerCount(), l.grid.listenerCount()], [0, 0]);
    l.type('model');
    assert.deepEqual(l.outline().slice(0, 1), ['main *'], 'a destroyed list answers nothing');
});
