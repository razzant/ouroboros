// The Model capsule (composer_model.js; DESIGN "Composer owner controls"): the first capsule of the row. A room's
// capsule follows Main until the room chooses, and says so; closed it shows the room's choice, never a running work's
// model, and its processing tier when that is not Standard. A press, never hover, opens the Model sheet (model_sheet.js)
// over the page: Main's row, the pinned and recent models, all sources, and the tier strip at its foot. A room's pick
// saves at once for that room; the Main chat's pick asks first and writes Main itself; a tier saves at once and leaves
// the sheet open; a Pin chooses nothing. The list's own grammar is model_list.test.js; engines, geometry and the sheet's
// place are the browser lane's.
import assert from 'node:assert/strict';
import test from 'node:test';
import { describeRoute } from '../modules/chat_model_routes.js';
import { MAIN_CACHE_NOTE, MAIN_RULE, confirmMainChange } from '../modules/composer_model.js';
import { createModelCatalogStore } from '../modules/model_catalog_store.js';
import { closeModelSheet } from '../modules/model_sheet.js';
import { MAIN_FOLLOW_NOTE } from '../modules/model_roles.js';
import { ElementStub } from './chat_dom_fixture.js';
import { modelSheetOutline, modelSheetRows, mountPair, sleep } from './composer_controls_fixture.js';

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
const STANDARD = { main: '', default: 'standard' };
const MAIN_FALLBACK_WORK = { work_id: 'fallback-work', live: true, activity: 'working',
    lead: { ...route(SOL), basis: 'fallback', role: 'fallback:0', primary: { ...route(ASTRA), basis: 'main' } },
    processing: { choice: '', resolved: 'standard' } };
const MAIN_TOKENS = { OUROBOROS_MODEL: 'a'.repeat(24), USE_LOCAL_MAIN: 'b'.repeat(24),
    OUROBOROS_MODEL_ACCOUNTS: 'c'.repeat(24), OUROBOROS_MODEL_PROCESSING_PREFERENCES: 'd'.repeat(24) };
// Each API connection's opaque revision (`_meta.connections[].revision`, its picker id first).
const OR_SOURCE = 'openrouter#r1', OPENAI_SOURCE = 'openai#r1';
// The window's catalog store with its reads counted; its pins are written as the server keeps them (`savePins` answers
// instead when given).
const storeOf = ({ catalog = CATALOG, recents = [], pins = [], connections = [{ id: 'openrouter', configured: true, revision: OR_SOURCE }], main = ASTRA, settings = {}, savePins = null, processing = null } = {}) => {
    // `processing`: a key model's own endpoints by its routed value (none read: the read fails, as offline).
    const reads = { catalog: 0, recents: 0, settings: 0, processing: [] };
    const answer = { recents, pins };
    const pinWrites = [];
    const store = createModelCatalogStore({
        readCatalog: async () => { reads.catalog += 1; if (catalog instanceof Error) throw catalog; return typeof catalog === 'function' ? catalog() : catalog; },
        readPreferences: async () => { reads.recents += 1; return answer.recents === undefined ? {} : { model_recents: answer.recents, model_pins: answer.pins }; },
        readSettings: async () => {
            reads.settings += 1;
            if (settings instanceof Error) throw settings;
            const doc = { OUROBOROS_MODEL: main, USE_LOCAL_MAIN: false, OUROBOROS_MODEL_ACCOUNTS: {}, OUROBOROS_MODEL_PROCESSING_PREFERENCES: {},
                _meta: { connections, tokens: { ...MAIN_TOKENS, OUROBOROS_MODEL_LIGHT: 'unrelated' } } };
            return typeof settings === 'function' ? settings(doc) : { ...doc, ...settings };
        },
        readProcessing: async (value) => { reads.processing.push(value); if (!processing) throw new Error('not read'); return processing(value); },
        savePreferences: async (body) => {
            pinWrites.push(body);
            if (savePins) return savePins(body);
            const { pin = [], unpin = [] } = body.model_pins;
            answer.pins = [...answer.pins.filter((seen) => !unpin.some((gone) => gone.model === seen.model)), ...pin];
            return { model_pins: answer.pins };
        },
    });
    return { store, reads, settings, answer, pinWrites };
};
const flush = async () => { for (let i = 0; i < 4; i += 1) await new Promise((resolve) => setImmediate(resolve)); };
const COGNITION = { context_mode: 'max', subagents_mode: 'auto', effort_range: { min: 'low', recommended: 'high', max: 'max' } };

/* A room's composer row (or the Main chat's, `projectId: ''`) with its Model capsule, fed one state body: Main's route,
   the tiers (`processing_tiers`), and the room's own model (`choice`) and tier (`tier`). */
function room({ projectId = 'p7', chatId = 7, choice = { kind: 'main' }, tier = '', main = ASTRA, tiers = STANDARD, model = {},
    storeOptions, ...rest } = {}) {
    const kept = storeOf({ main, settings: { OUROBOROS_MODEL_PROCESSING_PREFERENCES: { main: tiers?.main || '' },
        OUROBOROS_PROCESSING_PREFERENCE: tiers?.default || '' }, ...storeOptions });
    const asked = [];
    const confirmed = { value: true };
    const m = mountPair({ projectId, chatId, subagents: true, ...rest, model: { store: kept.store,
        askMain: async (request) => { asked.push(request); return confirmed.value; }, ...model } });
    const state = (extra = {}) => ({ ...(projectId ? {} : COGNITION), main_route: route(main), ...(tiers ? { processing_tiers: tiers } : {}),
        projects: projectId ? [{ id: projectId, chat_id: chatId, cognition: COGNITION, model_choice: choice, processing_choice: tier }] : [], ...extra });
    m.controls.syncState(state());
    const el = m.modelEl;
    const part = (name) => el.querySelector(`.chat-capsule-${name}`);
    const sheet = () => m.doc.body.children.find((node) => node.classList.contains('model-sheet')) || null;
    const list = () => sheet()?.querySelector('.model-sheet-list') || null;
    const input = () => sheet()?.querySelector('.chat-model-query') || null;
    const rowNode = (key) => list().querySelectorAll('.chat-model-row').find((node) => node.dataset.row === key);
    const pinNode = (key) => list().querySelectorAll('.chat-model-pin').find((node) => node.dataset.pin === key);
    // A press on a row's first cell, and one on its Pin (the list asks the press's target what it is in).
    const take = (key) => m.fire(list(), 'click', { target: { closest: (selector) => (selector === '.chat-model-row' ? rowNode(key) : null) } });
    const pin = (key) => m.fire(list(), 'click', { stopPropagation() {}, target: { closest: (selector) => (selector === '.chat-model-row' ? rowNode(key)
        : selector === '.chat-model-pin' ? pinNode(key) : null) } });
    const strip = () => sheet()?.querySelector('.model-tier') || null;
    const segs = () => strip().querySelectorAll('.model-tier-seg');
    const tierNode = (which) => (which === '' ? strip().querySelector('.model-tier-default') : segs().find((seg) => seg.dataset.tier === which));
    const chooseTier = (which) => m.fire(strip(), 'click', { target: { closest: (selector) => (selector === '[data-tier]' ? tierNode(which) : null) } });
    // The strip as it is painted: the chosen tier, the one the scope takes without a choice (its ring), whether a save
    // holds it, whether Default stands beside it, and what it applies to.
    const tierView = () => ({
        checked: segs().filter((seg) => seg.getAttribute('aria-checked') === 'true').map((seg) => seg.dataset.tier),
        ring: segs().filter((seg) => seg.dataset.resolved === 'true').map((seg) => seg.dataset.tier),
        busy: segs().every((seg) => seg.getAttribute('aria-disabled') === 'true'),
        reset: !strip().querySelector('.model-tier-default').hidden,
        scope: strip().querySelector('.model-tier-scope').textContent,
    });
    const type = (text) => { input().value = text; return m.fire(input(), 'input', {}); };
    const key = (name) => {
        const event = { key: name, defaultPrevented: false, preventDefault() { this.defaultPrevented = true; }, stopPropagation() {} };
        return m.fire(input(), 'keydown', event).then(() => event);
    };
    const posted = (url) => () => m.posts.filter(([to]) => to === url).map(([, body]) => body);
    return { ...m, ...kept, el, state, asked, confirmed, sheet, list, input, take, pin, strip, chooseTier, tierView, type, key,
        modelPosts: posted('/api/owner/chat-model'), tierPosts: posted('/api/owner/chat-processing'),
        name: () => part('name').textContent, value: () => part('value').textContent,
        tierWord: () => (part('tier').hidden ? '' : part('tier').textContent),
        rows: () => modelSheetRows(list().innerHTML), outline: () => modelSheetOutline(list().innerHTML),
        open: async () => { if (!sheet()) await m.press(m.modelHead); await flush(); } };
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

test('a source confirmed late wears its harness mark: the capsule repaints the mark when the catalog names it', async () => {
    const r = room();
    const glyph = () => r.el.querySelector('.chat-capsule-glyph').innerHTML;
    // The catalog has not been read: the source is an unknown one, so only its initial stands.
    assert.match(glyph(), /model-source-initial" aria-hidden="true">C</, 'an unknown source wears its initial, never an invented mark');
    await r.store.read();  // the read brings the source's descriptor (id codex, credentialHarness codex)
    await flush();
    // The same words (Codex) now stand beside the harness's own mark: the mark is what changed.
    assert.match(glyph(), /^<svg class="model-source-mark"/, 'the harness mark stands where the descriptor names it');
    r.controls.destroy();
});

test('it is the first capsule of the row and one of its family: a press opens it, hover never does, and one is open at a time', async () => {
    const r = room({ hover: true });
    assert.equal(r.row.children[0], r.el);
    const capsule = r.controls.model.capsule;
    assert.ok(r.el.classList.contains('chat-capsule') && capsule.kind === 'model' && capsule.hoverCapable === false);
    assert.deepEqual(capsule.peers.map((peer) => peer.kind), ['subagents', 'context', 'effort']);
    await r.fire(r.el, 'pointerenter', { pointerType: 'mouse' });
    await r.fire(r.modelHead, 'pointerenter', { pointerType: 'mouse' });
    await sleep(220);
    assert.deepEqual([r.sheet(), capsule.isOpen()], [null, false], 'a mouse resting on it opens nothing');
    await r.open();
    assert.ok(r.sheet() && capsule.isOpen());
    await r.press(r.contextHead);
    assert.deepEqual([r.contextEl.dataset.open, r.sheet(), r.modelHead.getAttribute('aria-expanded'), r.el.dataset.sheetOpen],
        ['true', null, 'false', undefined], 'another capsule opening closes the sheet at once');
    await r.open();
    assert.deepEqual([r.contextEl.dataset.open, Boolean(r.sheet())], ['false', true], 'and the sheet opening closes the others');
    capsule.snapClosed();
    assert.equal(r.sheet(), null, 'the row closes it like any of its capsules');
    capsule.setOpen(true);
    assert.ok(r.sheet());
    capsule.setOpen(false);
    assert.equal(r.sheet(), null);
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

test('closed it always shows the room\'s choice, never a running work\'s model', () => {
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
    r.controls.destroy();
});

test('closed, the capsule names the tier its new work takes beside the model, only when that is not Standard', () => {
    const closed = (options) => {
        const r = room(options);
        const said = [r.tierWord(), r.el.dataset.tier];
        r.controls.destroy();
        return said;
    };
    assert.deepEqual(closed({ tiers: { main: 'fast', default: 'standard' } }), ['Fast', 'fast'], 'a follower takes Main\'s own tier');
    assert.deepEqual(closed({ tiers: { main: '', default: 'economy' } }), ['Flex', 'economy'], 'Main without one takes the default');
    assert.deepEqual(closed({ tiers: { main: 'ultrafast', default: 'standard' } }), ['Ultrafast', 'ultrafast']);
    assert.deepEqual(closed({ tiers: { main: 'fast', default: 'standard' }, choice: { kind: 'model', model: GLM, use_local: false } }),
        ['', 'standard'], 'a room on its own model takes the default tier, not Main\'s');
    assert.deepEqual(closed({ tiers: { main: 'fast', default: 'standard' }, tier: 'economy' }), ['Flex', 'economy'], 'the room\'s own tier first');
    assert.deepEqual(closed({ tiers: { main: 'fast', default: 'economy' }, tier: 'standard' }), ['', 'standard'],
        'an explicit Standard is a choice of its own, and is not named');
    assert.deepEqual(closed({ tiers: null }), ['', 'default'], 'no tiers read: nothing named');
    assert.deepEqual(closed({ projectId: '', chatId: 1, tiers: { main: '', default: 'fast' } }), ['Fast', 'fast'], 'the Main chat is Main\'s own tier');
    assert.deepEqual(closed({ projectId: '', chatId: 1, tiers: { main: 'standard', default: 'fast' } }), ['', 'standard']);
    // Read live, and said to a screen reader with the model.
    const r = room({ tiers: { main: 'fast', default: 'standard' } });
    assert.equal(r.modelHead.getAttribute('aria-label'), 'Model for new work here: gpt-6-astra, Fast, Codex. Follows Main.');
    r.controls.syncState(r.state({ processing_tiers: STANDARD }));
    assert.deepEqual([r.tierWord(), r.el.querySelector('.chat-capsule-tier').hidden], ['', true]);
    r.controls.syncState(r.state({ projects: [{ id: 'p7', cognition: COGNITION, model_choice: { kind: 'main' }, processing_choice: 'fast' }] }));
    assert.equal(r.tierWord(), 'Fast', 'the room\'s own tier from its project row');
    r.controls.destroy();
    const main = room({ projectId: '', chatId: 1, tiers: { main: 'economy', default: 'standard' } });
    assert.equal(main.modelHead.getAttribute('aria-label'), 'Main model: gpt-6-astra, Flex, Codex.');
    main.controls.destroy();
});

test('a press opens the Model sheet over the page, by the composer row; a second press closes it', async () => {
    const r = room({ storeOptions: { recents: [route(GLM)] } });
    assert.equal(r.sheet(), null);
    await r.press(r.modelHead);
    await flush();
    const sheet = r.sheet();
    assert.deepEqual([sheet.className, sheet.getAttribute('role'), sheet.getAttribute('aria-label'), sheet.parentNode === r.doc.body],
        ['ui-popup model-sheet', 'dialog', 'Model for new work here', true], 'not in the composer\'s flow');
    assert.deepEqual(sheet.children.map((node) => node.className), ['chat-model-search', 'model-sheet-list', 'model-tier'],
        'the field, the list, and the tier strip at its foot');
    assert.equal(r.input().getAttribute('aria-controls'), r.list().id);
    assert.match(r.el.innerHTML, /^<button type="button" class="chat-capsule-head ui-hit-area chat-model-head" aria-expanded="false" aria-haspopup="dialog">/,
        'the head says it opens a dialog');
    assert.deepEqual([r.modelHead.getAttribute('aria-expanded'), r.el.dataset.sheetOpen, r.row.dataset.ownerSheet],
        ['true', 'true', 'true'], 'the row knows a capsule is worked (composer_fold.js)');
    assert.deepEqual(r.places.map(([placed, anchor]) => [placed === sheet, anchor === r.row]), [[true, true]], 'it stands by the row');
    assert.deepEqual([r.reads.catalog, r.reads.recents], [1, 1], 'opening reads the catalog and the recents once');
    assert.equal(r.el.querySelector('.chat-model-strip'), null, 'nothing opens inside the row');
    await r.press(r.modelHead);
    assert.deepEqual([r.sheet(), r.modelHead.getAttribute('aria-expanded'), r.el.dataset.sheetOpen, r.row.dataset.ownerSheet],
        [null, 'false', undefined, undefined]);
    await r.open();
    r.controls.destroy();
    assert.equal(r.sheet(), null, 'the sheet goes with its capsule');
});

test('a room\'s sheet: Main\'s row to follow Main, the pinned and recent models, then all sources; its own model leads the recents', async () => {
    const r = room({ tiers: { main: 'fast', default: 'standard' }, storeOptions: { recents: [route(GLM), route(ASTRA)], pins: [route(KIMI)] } });
    await r.open();
    assert.deepEqual(r.outline(), ['main *', '§ Pinned', `route|${KIMI} +pin`, '§ Recent', `route|${GLM} -pin`, `route|${ASTRA} -pin`, 'sources']);
    assert.deepEqual([r.rows()[0].name, r.rows()[0].facts], ['Main', 'Follows Main · gpt-6-astra · Fast'], 'Main\'s row says what it follows');
    r.controls.destroy();
    const own = room({ choice: { kind: 'model', model: SOL, use_local: false }, storeOptions: { recents: [route(GLM), route(SOL)] } });
    await own.open();
    assert.deepEqual(own.outline(), ['main', '§ Recent', `route|${SOL} * -pin`, `route|${GLM} -pin`, 'sources'], 'once, first');
    assert.equal(own.rows()[0].facts, 'Follows Main · gpt-6-astra');
    own.controls.destroy();
});

test('a model row is always that model: Main\'s model picked as a row becomes the room\'s own; only Main\'s row follows Main', async () => {
    const r = room({ storeOptions: { recents: [route(ASTRA)] } });
    await r.open();
    await r.take(`route|${ASTRA}`);
    assert.equal(r.sheet(), null, 'a pick closes the sheet');
    assert.equal(await r.controls.model.pendingSave(), true);
    assert.deepEqual(r.modelPosts(), [{ kind: 'model', model: ASTRA, use_local: false, project_id: 'p7' }], 'never a follow of Main');
    assert.deepEqual([r.name(), r.value(), r.el.dataset.own], ['Codex', 'gpt-6-astra', 'true'], 'it stays when Main moves');
    r.controls.syncState(r.state({ main_route: route(GLM), projects: [{ id: 'p7', model_choice: { kind: 'model', model: ASTRA, use_local: false } }] }));
    assert.deepEqual([r.name(), r.value()], ['Codex', 'gpt-6-astra']);
    await r.open();
    assert.deepEqual(r.rows().filter((entry) => entry.selected).map((entry) => entry.key), [`route|${ASTRA}`]);
    await r.take('main');
    assert.equal(await r.controls.model.pendingSave(), true);
    assert.deepEqual(r.modelPosts().at(-1), { kind: 'main', project_id: 'p7' });
    assert.deepEqual([r.name(), r.value(), r.el.dataset.own], ['Main', 'glm-5.3', 'false']);
    // Already following: Main's row saves nothing and closes the sheet.
    await r.open();
    await r.take('main');
    assert.deepEqual([r.modelPosts().length, r.sheet()], [2, null]);
    r.controls.destroy();
});

test('a room\'s pick saves at once for that room and Send can wait for it; the chosen model again saves nothing', async () => {
    const r = room({ storeOptions: { recents: [route(GLM)] } });
    r.modelHead.isConnected = true;  // on the page: the focus can come back to it
    await r.open();
    let answer;
    r.modelAnswer.value = new Promise((resolve) => { answer = resolve; });
    await r.take(`route|${GLM}`);
    assert.deepEqual(r.modelPosts(), [{ kind: 'model', model: GLM, use_local: false, project_id: 'p7' }]);
    assert.deepEqual([r.controls.hasPendingSave(), r.el.dataset.disabled, r.name()], [true, 'true', 'Main'], 'unsaved, it is not shown as saved');
    assert.equal(r.doc.activeElement, r.modelHead, 'the sheet closed with the pick, the focus is back on the capsule');
    r.controls.syncState(r.state());
    answer({ ok: true, project_id: 'p7', model_choice: { kind: 'model', model: GLM, use_local: false }, previous_model_choice: { kind: 'main' } });
    assert.equal(await r.controls.pendingSave(), true);
    assert.deepEqual([r.name(), r.value(), r.el.dataset.disabled, r.refreshes.at(-1)], ['OpenRouter', 'glm-5.3', 'false', true]);
    assert.equal(r.reads.recents, 2, 'the server wrote the act into the recents: they are read again');
    r.modelAnswer.value = null;
    await r.open();
    await r.take(`route|${GLM}`);
    assert.deepEqual([r.modelPosts().length, r.sheet()], [1, null], 'the chosen model again saves nothing');
    r.controls.destroy();
});

test('a refused save keeps the saved value and says why; an answer for another room or without the choice is no save', async () => {
    const r = room({ storeOptions: { recents: [route(GLM)] } });
    // The answer is made as the press is: a refusal waits for no one. Send asks while the save is out.
    const pick = async (answer) => {
        await r.open();
        r.modelAnswer.value = answer();
        const pressed = r.take(`route|${GLM}`);
        const send = r.controls.model.pendingSave();
        await pressed;
        return send;
    };
    assert.equal(await pick(() => Promise.reject(new Error('Not a model route'))), false);
    assert.deepEqual([r.name(), r.value(), r.toasts.at(-1)], ['Main', 'gpt-6-astra', ['Not a model route', 'error']]);
    await pick(() => ({ ok: true, project_id: 'other', model_choice: { kind: 'model', model: GLM, use_local: false } }));
    assert.deepEqual([r.name(), r.toasts.at(-1)[0]], ['Main', 'The answer belongs to another room.']);
    await pick(() => ({ ok: true, project_id: 'p7' }));
    assert.deepEqual([r.name(), r.toasts.at(-1)[0]], ['Main', 'The server did not confirm the model.']);
    r.controls.destroy();
});

test('in the Main chat the capsule is Main itself: a pick asks first, then writes Main and says what else changed', async () => {
    const r = room({ projectId: '', chatId: 1, storeOptions: { recents: [route(GLM)] } });
    assert.deepEqual([r.name(), r.value(), r.el.dataset.own], ['Main', 'gpt-6-astra', 'false']);
    assert.match(r.modelHead.getAttribute('aria-label'), /^Main model: gpt-6-astra, Codex\.$/);
    await r.open();
    assert.equal(r.sheet().getAttribute('aria-label'), 'Main model');
    assert.deepEqual(r.outline(), ['main * -pin', '§ Recent', `route|${GLM} -pin`, 'sources'], 'Main\'s own model, chosen, with its Pin');
    assert.deepEqual([r.rows()[0].name, r.rows()[0].tags, r.rows()[0].facts], ['gpt-6-astra', ['Main'], '872K · Codex · subscription']);
    await r.take('main');
    await flush();
    assert.deepEqual([r.asked, r.modelPosts()], [[], []], 'Main\'s own row changes nothing');
    r.confirmed.value = false;
    await r.open();
    await r.take(`route|${GLM}`);
    await flush();
    assert.deepEqual([r.sheet(), r.modelPosts(), r.value()], [null, [], 'gpt-6-astra'], 'cancelled: nothing is written');
    assert.deepEqual([r.asked[0].next.short, r.asked[0].next.sourceLabel, r.asked[0].previous.short], ['glm-5.3', 'OpenRouter', 'gpt-6-astra']);
    r.confirmed.value = true;
    r.modelAnswer.value = { ok: true, project_id: '', main_route: route(GLM), previous_main_route: route(ASTRA), account: 'auto', window_reset: true };
    await r.open();
    await r.take(`route|${GLM}`);
    await flush();
    assert.deepEqual(r.modelPosts(), [{ kind: 'model', model: GLM, use_local: false, if_unchanged: MAIN_TOKENS }],
        'no project: the write is Main\'s own, on the binding it was shown');
    assert.deepEqual([r.value(), r.toasts.at(-1)], ['glm-5.3', ['Main\'s account is Auto now. Main\'s context window set by hand was cleared.', 'info']]);
    // Nothing else changed: nothing more is said.
    r.settings.OUROBOROS_MODEL = GLM;  // the server's Main now
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
        ['OpenRouter · paid by key · 1M window.', `${MAIN_RULE} ${MAIN_FOLLOW_NOTE}`, MAIN_CACHE_NOTE,
            'How many running works follow Main is unknown right now: the list of running work is incomplete.']);
    assert.equal(content.children.length, 5, 'no census was given: the count is unknown and there is nothing to list (main_sheet.test.js)');
    // The catalog's cache-write price for the install's cache tier follows the fact when the catalog states one.
    await confirmMainChange({ next: { ...next, cacheWrite: 'Cache write: $16 per 1M tokens.' }, previous, dialog, doc });
    const priced = doc.createElement('section');
    calls[1].content.mount(priced);
    assert.equal(priced.children[3].textContent, `${MAIN_CACHE_NOTE} Cache write: $16 per 1M tokens.`);
    assert.deepEqual([content.children[3].dataset.i18nFmt, priced.children[3].dataset.i18nFmt], [undefined, 'main cache note'],
        'only the line that carries a figure is composed; the plain sentence stays the overlay\'s to translate');
    // The same subscription keeps Main's account, so only the window is said.
    await confirmMainChange({ next: { ...next, source: 'subscription:codex', lane: 'subscription' }, previous, dialog, doc });
    assert.equal(calls[2].body, 'A context window set by hand for Main is cleared.');
});

test('a Pin chooses nothing: it writes the pins alone, keeps the sheet open, and the pinned model stands under Pinned', async () => {
    const r = room({ storeOptions: { recents: [route(GLM)] } });
    await r.open();
    const recentReads = r.reads.recents;
    await r.pin(`route|${GLM}`);
    await flush();
    assert.deepEqual(r.pinWrites, [{ model_pins: { pin: [route(GLM)] } }], 'one addressed write of the pins');
    assert.deepEqual([r.modelPosts(), r.tierPosts(), r.reads.recents, r.store.recents()], [[], [], recentReads, [route(GLM)]],
        'no model is chosen, so no recent is written or read');
    assert.ok(r.sheet(), 'the sheet stays open');
    assert.deepEqual([r.name(), r.el.dataset.own], ['Main', 'false']);
    assert.deepEqual(r.outline(), ['main *', '§ Pinned', `route|${GLM} +pin`, 'sources'], 'a model listed once: under Pinned');
    await r.pin(`route|${GLM}`);
    await flush();
    assert.deepEqual(r.pinWrites.at(-1), { model_pins: { unpin: [route(GLM)] } });
    assert.deepEqual(r.outline(), ['main *', '§ Recent', `route|${GLM} -pin`, 'sources']);
    r.controls.destroy();
    // A refused pin is said; the list is read again as the server has it.
    const refused = room({ storeOptions: { recents: [route(GLM)], savePins: async () => { throw new Error('Preferences are read-only.'); } } });
    await refused.open();
    await refused.pin(`route|${GLM}`);
    await flush();
    assert.deepEqual([refused.toasts.at(-1), refused.store.pins(), Boolean(refused.sheet())], [['Preferences are read-only.', 'error'], [], true]);
    assert.equal(refused.outline()[1], '§ Recent');
    refused.controls.destroy();
});

test('a touch inside the floating Model list keeps its sheet and chooses nothing', async () => {
    const r = room();
    await r.open();
    const sheet = r.sheet();
    assert.equal(sheet.parentNode, r.doc.body, 'the sheet is outside the composer flow');
    r.doc.fire('pointerdown', { target: r.list(), pointerType: 'touch' });
    assert.equal(r.sheet(), sheet);
    assert.equal(r.el.dataset.sheetOpen, 'true');
    assert.deepEqual([r.modelPosts(), r.tierPosts()], [[], []]);
    r.controls.destroy();
});

test('a tier press saves the room\'s own tier at once and keeps the sheet open; Send waits for it', async () => {
    const r = room({ tiers: { main: 'fast', default: 'standard' } });
    await r.open();
    assert.deepEqual(r.tierView(), { checked: [], ring: ['fast'], busy: false, reset: false,
        scope: 'Default · Fast as Main · for new work here and its new subagents without a tier of their own; running work keeps its own' }, 'a follower takes Main\'s tier: it wears the ring');
    let answer;
    r.tierAnswer.value = new Promise((resolve) => { answer = resolve; });
    await r.chooseTier('economy');
    assert.deepEqual(r.tierPosts(), [{ processing: 'economy', project_id: 'p7' }]);
    assert.deepEqual([r.controls.hasPendingSave(), r.modelPosts(), r.asked], [true, [], []], 'a tier moves nothing else');
    answer({ ok: true, project_id: 'p7', processing_choice: 'economy', previous_processing_choice: '' });
    assert.equal(await r.controls.pendingSave(), true);
    assert.ok(r.sheet(), 'a tier is no pick: the sheet stays open');
    assert.deepEqual(r.tierView(), { checked: ['economy'], ring: [], busy: false, reset: true,
        scope: 'Flex · for new work here and its new subagents without a tier of their own; running work keeps its own' });
    assert.deepEqual([r.tierWord(), r.refreshes.at(-1), r.controls.hasPendingSave()], ['Flex', true, false]);
    // The chosen tier again is no act; the answer's own tier is the one shown.
    await r.chooseTier('economy');
    assert.equal(r.tierPosts().length, 1);
    r.tierAnswer.value = { ok: true, project_id: 'p7', processing_choice: 'standard' };
    await r.chooseTier('fast');
    await r.controls.pendingSave();
    assert.deepEqual([r.tierPosts().at(-1), r.tierView().checked, r.tierWord()], [{ processing: 'fast', project_id: 'p7' }, ['standard'], '']);
    r.controls.destroy();
});

test('while a tier saves, the strip says so: its tiers wait for the answer', async () => {
    const r = room();
    await r.open();
    let answer;
    r.tierAnswer.value = new Promise((resolve) => { answer = resolve; });
    await r.chooseTier('fast');
    assert.equal(r.tierView().busy, true);
    answer({ ok: true, project_id: 'p7', processing_choice: 'fast' });
    await r.controls.pendingSave();
    assert.equal(r.tierView().busy, false);
    r.controls.destroy();
});

test('Default returns the room to the tier it would take without one; a refused tier keeps the one it had', async () => {
    const r = room({ tier: 'fast', tiers: { main: 'economy', default: 'standard' } });
    assert.equal(r.tierWord(), 'Fast');
    await r.open();
    assert.deepEqual([r.tierView().checked, r.tierView().reset], [['fast'], true], 'Default stands beside a choice');
    await r.chooseTier('');
    await r.controls.pendingSave();
    assert.deepEqual(r.tierPosts(), [{ processing: '', project_id: 'p7' }]);
    assert.deepEqual(r.tierView(), { checked: [], ring: ['economy'], busy: false, reset: false,
        scope: 'Default · Flex as Main · for new work here and its new subagents without a tier of their own; running work keeps its own' });
    assert.equal(r.tierWord(), 'Flex');
    r.tierAnswer.value = { ok: true, project_id: 'other', processing_choice: 'fast' };
    await r.chooseTier('fast');
    await r.controls.pendingSave();
    assert.deepEqual([r.toasts.at(-1), r.tierView().checked, r.tierWord()], [['The server did not confirm the tier.', 'error'], [], 'Flex']);
    r.tierAnswer.value = Promise.reject(new Error('Settings are being written.'));
    await r.chooseTier('fast');
    assert.equal(await r.controls.pendingSave(), true, 'a refused tier holds no message back: the model is saved');
    assert.deepEqual([r.toasts.at(-1), r.tierView().checked, Boolean(r.sheet())], [['Settings are being written.', 'error'], [], true]);
    r.controls.destroy();
});

test('in the Main chat a tier is Main\'s own: no confirmation, and its answer is the tiers every room reads', async () => {
    const r = room({ projectId: '', chatId: 1, tiers: STANDARD });
    await r.open();
    assert.deepEqual([r.tierView().ring, r.tierView().scope.split(' · ')[0]], [['standard'], 'Default'], 'no tier of its own: the default');
    await r.chooseTier('fast');
    await r.controls.pendingSave();
    assert.deepEqual(r.tierPosts(), [{ processing: 'fast' }], 'no project: Main\'s own tier');
    assert.deepEqual([r.asked, r.modelPosts()], [[], []], 'a tier is no change of Main\'s model: nothing to confirm');
    assert.deepEqual(r.store.tiers(), { main: 'fast', default: 'standard' }, 'as the answer says them');
    assert.deepEqual([r.tierWord(), r.tierView().checked, Boolean(r.sheet())], ['Fast', ['fast'], true]);
    r.tierAnswer.value = { ok: true, project_id: '', main_processing: '', previous_main_processing: 'fast', processing_default: 'economy' };
    await r.chooseTier('');
    await r.controls.pendingSave();
    assert.deepEqual([r.tierPosts().at(-1), r.store.tiers(), r.tierView().ring, r.tierWord()],
        [{ processing: '' }, { main: '', default: 'economy' }, ['economy'], 'Flex'], 'Default: Main takes the default tier again');
    r.controls.destroy();
});

test('the Main tier strip explains inheritance even when the only visible work is on a fallback', async () => {
    const main = room({ projectId: '', chatId: 1 });
    main.controls.syncState(main.state({ works: [MAIN_FALLBACK_WORK], works_complete: true, supervisor_ready: true }));
    await main.open();
    assert.match(main.tierView().scope, /^Default · Standard · .*inherits its tier, even on a fallback model/);
    const said = main.tierView().scope;
    main.controls.syncState(main.state({ works: [], works_complete: false, supervisor_ready: true }));
    assert.equal(main.tierView().scope, said, 'the rule does not change with the completeness of the model census');
    main.controls.destroy();
    // A room's tier is for its new work only.
    const r = room({ choice: { kind: 'model', model: GLM, use_local: false } });
    r.controls.syncState(r.state({ works: [MAIN_FALLBACK_WORK], works_complete: true, supervisor_ready: true }));
    await r.open();
    assert.equal(r.tierView().scope, 'Default · Standard · for new work here and its new subagents without a tier of their own; running work keeps its own');
    r.controls.destroy();
});

test('under a finger opening raises no keyboard; a mouse or a keyboard starts in the field', async () => {
    const fine = room();
    await fine.open();
    assert.equal(fine.doc.activeElement, fine.input());
    fine.controls.destroy();
    const finger = room({ model: { isCoarse: () => true } });
    await finger.open();
    assert.ok(finger.input() && finger.list(), 'the field and the list are there');
    assert.notEqual(finger.doc.activeElement, finger.input(), 'the field takes focus on its own press');
    finger.controls.destroy();
});

test('one Model sheet in the window: another one opening closes this one, and closeModelSheet closes any', async () => {
    const first = room();
    const second = room({ projectId: 'p8', chatId: 8 });
    await first.open();
    await second.open();
    assert.deepEqual([first.sheet(), Boolean(second.sheet()), first.modelHead.getAttribute('aria-expanded'), first.row.dataset.ownerSheet],
        [null, true, 'false', undefined], 'the first capsule hears its sheet go');
    closeModelSheet();
    assert.deepEqual([second.sheet(), second.controls.model.capsule.isOpen()], [null, false]);
    first.controls.destroy();
    second.controls.destroy();
});

test('in the stack layout the sheet rises from the bottom over a scrim, modal; the scrim\'s own click closes it', async () => {
    const r = room({ model: { isStack: () => true } });
    await r.open();
    const sheet = r.sheet();
    const scrim = r.doc.body.children.find((node) => node.classList.contains('model-sheet-scrim'));
    assert.equal(sheet.dataset.sheet, 'bottom');
    assert.equal(sheet.getAttribute('aria-modal'), 'true', 'what is on top: Back gives it its Escape first (layers.js back())');
    assert.ok(scrim && r.doc.body.children.indexOf(scrim) < r.doc.body.children.indexOf(sheet), 'under the sheet');
    assert.equal(scrim.getAttribute('aria-hidden'), 'true');
    assert.ok(sheet.children[0].classList.contains('model-sheet-grab'));
    assert.equal(r.places.length, 0, 'a bottom sheet is not placed by its anchor');
    r.docDown(scrim);
    assert.ok(r.sheet() && r.doc.body.children.includes(scrim), 'a finger down on the scrim leaves it there, so its click lands on it');
    let prevented = false;
    await r.fire(scrim, 'click', { preventDefault() { prevented = true; } });
    assert.deepEqual([r.sheet(), r.doc.body.children.includes(scrim), prevented], [null, false, true],
        'the dismissing click closes it and goes no further');
    r.controls.destroy();
    const desk = room();
    await desk.open();
    assert.equal(desk.sheet().getAttribute('aria-modal'), '', 'the desk\'s popover is no modal');
    desk.controls.destroy();
});

test('a room leaving the screen takes its own sheet with it, never another room\'s', async () => {
    const r = room();
    await r.open();
    closeModelSheet({ within: new ElementStub('div', r.doc) });
    assert.ok(r.sheet(), 'another room left: this sheet stays');
    const page = new ElementStub('div', r.doc);
    page.appendChild(r.row);
    closeModelSheet({ within: page });
    assert.deepEqual([r.sheet(), r.modelHead.getAttribute('aria-expanded')], [null, 'false'], 'its own room took it');
    r.controls.destroy();
});

test('Send waits for a tier save in flight, and an earlier refused model save does not stop it', async () => {
    const r = room({ storeOptions: { recents: [route(GLM)] } });
    await r.open();
    r.modelAnswer.value = Promise.reject(new Error('Not a model route'));
    const pressed = r.take(`route|${GLM}`);
    const refusedSend = r.controls.model.pendingSave();
    await pressed;
    assert.equal(await refusedSend, false, 'that refusal answers the Send that waited for it');
    r.modelAnswer.value = null;
    await r.open();
    let answer;
    r.tierAnswer.value = new Promise((resolve) => { answer = resolve; });
    await r.chooseTier('fast');
    const waiting = r.controls.model.pendingSave();
    answer({ ok: true, project_id: 'p7', processing_choice: 'fast', previous_processing_choice: '' });
    assert.equal(await waiting, true, 'the tier saved: Send goes on');
    r.controls.destroy();
});

test('a press elsewhere, Escape and focus leaving for the page close the sheet; a press in it or on the row does not', async () => {
    const r = room();
    r.modelHead.isConnected = true;  // on the page: the focus can come back to it
    const listening = () => [r.doc.count('pointerdown'), r.doc.count('keydown')];
    const before = listening();
    await r.open();
    r.docDown(r.input());
    r.docDown(r.modelHead);
    assert.ok(r.sheet(), 'a press in the sheet is its own, the row\'s are the capsules\'');
    r.docDown(new ElementStub('div', r.doc));
    assert.deepEqual([r.sheet(), r.modelHead.getAttribute('aria-expanded')], [null, 'false']);
    assert.notEqual(r.doc.activeElement, r.modelHead, 'a pointer elsewhere keeps its own focus');
    await r.open();
    await r.type('gl');
    assert.equal((await r.key('Escape')).defaultPrevented, true);
    assert.deepEqual([r.input().value, Boolean(r.sheet())], ['', true], 'Escape first clears what was typed');
    await r.key('Escape');
    assert.deepEqual([r.sheet(), r.doc.activeElement], [null, r.modelHead], 'then closes, the focus back on the capsule');
    await r.open();
    const escape = { key: 'Escape', defaultPrevented: false, preventDefault() { this.defaultPrevented = true; }, stopPropagation() {} };
    r.doc.fire('keydown', escape);
    assert.deepEqual([r.sheet(), escape.defaultPrevented], [null, true], 'Escape from anywhere on the page');
    await r.open();
    await r.fire(r.sheet(), 'focusout', { relatedTarget: r.input() });
    await r.fire(r.sheet(), 'focusout', { relatedTarget: null });
    assert.ok(r.sheet(), 'focus moving within, or to nothing, keeps it');
    await r.fire(r.sheet(), 'focusout', { relatedTarget: new ElementStub('button', r.doc) });
    assert.equal(r.sheet(), null, 'focus leaving for the page closes it');
    assert.deepEqual(listening(), before, 'each close takes the sheet\'s listeners with it');
    r.controls.destroy();
});

test('typing searches every source; a pick saves and closes the sheet; any id can still be used as typed', async () => {
    const r = room();
    await r.open();
    await r.type('k');
    assert.deepEqual(r.rows().map((entry) => [entry.key, entry.facts]), [[`route|${KIMI}`, 'OpenRouter · paid by key'],
        ['route|x-ai/grok-4.7', 'OpenRouter · paid by key'], ['typed:route|k', 'OpenRouter']]);
    await r.type('kimi');
    assert.equal((await r.key('Enter')).defaultPrevented, true, 'Enter takes the first match');
    await flush();
    assert.deepEqual(r.modelPosts(), [{ kind: 'model', model: KIMI, use_local: false, project_id: 'p7' }]);
    assert.deepEqual([r.sheet(), r.row.dataset.ownerSheet, r.name(), r.value()], [null, undefined, 'OpenRouter', 'kimi-k3'],
        'the closed capsule says what was set');
    await r.open();
    await r.type('vendor/brand-new');
    assert.deepEqual(r.outline(), ['typed:route|vendor/brand-new']);
    await r.key('Enter');
    await flush();
    assert.deepEqual(r.modelPosts().at(-1), { kind: 'model', model: 'vendor/brand-new', use_local: false, project_id: 'p7' });
    r.controls.destroy();
});

test('the tiers come with the state body: a price is said where the scope\'s new work pays the catalog\'s, without a settings read', async () => {
    const priced = { ...CATALOG, items: CATALOG.items.map((item) => (item.value === KIMI ? { ...item, price: { input: 0.6, output: 2.5 } } : item)) };
    const kimi = async (tiers, options = {}) => {
        const r = room({ tiers, main: GLM, storeOptions: { catalog: priced, recents: [route(KIMI)] }, ...options });
        await r.open();
        const said = [r.rows().find((entry) => entry.key === `route|${KIMI}`).facts, r.reads.settings];
        r.controls.destroy();
        return said;
    };
    assert.deepEqual(await kimi(STANDARD), ['OpenRouter · $0.60 in / $2.50 out', 0]);
    assert.deepEqual(await kimi({ main: '', default: 'fast' }), ['OpenRouter · paid by key', 0], 'a Fast call is priced by its endpoint');
    assert.deepEqual(await kimi({ main: 'fast', default: 'standard' }), ['OpenRouter · $0.60 in / $2.50 out', 0],
        'a model row is the room\'s own model: it runs on the default tier, whatever Main\'s');
    assert.deepEqual(await kimi(STANDARD, { tier: 'economy' }), ['OpenRouter · paid by key', 0], 'the room\'s own tier first');
    assert.deepEqual(await kimi({ main: 'economy', default: 'standard' }, { projectId: '', chatId: 1 }), ['OpenRouter · paid by key', 0],
        'the Main chat pays Main\'s own tier');
    assert.deepEqual(await kimi(STANDARD, { projectId: '', chatId: 1 }), ['OpenRouter · $0.60 in / $2.50 out', 0]);
});

test('Main reads the cache tier for a catalog figure and takes a fresh binding for confirmation', async () => {
    const cached = { ...CATALOG, items: CATALOG.items.map((item) => (item.value === GLM
        ? { ...item, price: { input: 1, output: 4, cache_write: 1.25, cache_write_ttl: '1h' } } : item)) };
    // The settings document says the same tiers as the state body, in its own keys.
    const asked = async ({ mainTier = '', ...options }) => {
        const settings = { OUROBOROS_PROMPT_CACHE_TTL: '1h', OUROBOROS_PROCESSING_PREFERENCE: 'standard',
            OUROBOROS_MODEL_PROCESSING_PREFERENCES: JSON.stringify(mainTier ? { main: mainTier } : {}) };
        const r = room({ main: KIMI, tiers: { main: mainTier, default: 'standard' }, storeOptions: { catalog: cached, recents: [route(GLM)], settings }, ...options });
        await r.open();
        await flush();
        const readsAtOpen = r.reads.settings;
        await r.take(`route|${GLM}`);
        await flush();
        const said = [readsAtOpen, r.reads.settings - readsAtOpen, r.asked[0]?.next.cacheWrite ?? null];
        r.controls.destroy();
        return said;
    };
    assert.deepEqual(await asked({ projectId: '', chatId: 1 }), [1, 1, 'Cache write: $1.25 per 1M tokens.']);
    assert.deepEqual(await asked({ projectId: '', chatId: 1, mainTier: 'fast' }), [1, 1, ''],
        'Main on Fast: the figure is Standard\'s, so it is not said');
    assert.deepEqual(await asked({}), [0, 0, null], 'a room asks no confirmation and needs no cache tier');
});

test('nothing but Main to lead offers the way to Settings › Accounts and keys; a failed catalog read says so', async () => {
    const alone = room({ storeOptions: { catalog: { model_sources: CATALOG.model_sources, items: [] } } });
    await alone.open();
    assert.deepEqual(alone.outline(), ['main *', 'note: Only Main\'s model can lead work here.', 'accounts']);
    await alone.take('accounts');
    assert.deepEqual([alone.settingsOpened, alone.sheet()], [['accounts'], null]);
    alone.controls.destroy();
    const failed = room({ storeOptions: { catalog: new Error('offline') } });
    await failed.open();
    assert.deepEqual(failed.outline(), ['main *', 'note: The model catalog could not be read.', 'accounts']);
    failed.controls.destroy();
});

test('a saved choice whose key cannot send keeps Settings\' words; the settings document is read only then', async () => {
    const connections = [{ id: 'openrouter', enabled: true, configured: true }, { id: 'deepseek', enabled: false, configured: true }];
    const r = room({ choice: { kind: 'model', model: 'deepseek::deepseek-v4', use_local: false }, storeOptions: { connections } });
    assert.deepEqual([r.name(), r.reads.settings], ['DeepSeek', 0], 'nothing is read for a closed capsule');
    await r.open();
    await flush();
    assert.equal(r.reads.settings, 1);
    assert.deepEqual([r.name(), r.value()], ['DeepSeek (off)', 'deepseek-v4']);
    assert.deepEqual(r.rows().find((entry) => entry.key === 'route|deepseek::deepseek-v4'),
        { key: 'route|deepseek::deepseek-v4', kind: '', selected: true, name: 'deepseek-v4', tags: [], facts: 'DeepSeek (off) · paid by key', pin: false },
        'the room\'s own model leads the recents, in Settings\' words');
    r.controls.destroy();
    const fine = room({ choice: { kind: 'model', model: GLM, use_local: false } });
    await fine.open();
    await flush();
    assert.equal(fine.reads.settings, 0, 'a source the catalog lists needs no settings read');
    fine.controls.destroy();
});

/* --- the strip follows the scope's own route as it is bound: its transport, its model, its account --- */

// Codex with two accounts as one read gives them: `home` lists Ultrafast for gpt-6-astra, `work` does not; the
// transport carries no Flex.
const envelope = (id, models) => ({ credentialProfileId: id, availability: 'available', problem: null,
    catalog: { source: 'codex', credentialProfileId: id, accountFingerprint: 'fp-' + id, provenance: 'live', models: Object.entries(models).map(([model, modes]) => ({ id: model, processing: { modes } })) } });
const TWO_ACCOUNTS = { ...CATALOG, store_id: 'store-a', model_sources: [{ ...CATALOG.model_sources[0], processingPreferences: ['standard', 'fast', 'ultrafast'] }],
    account_catalogs: [{ source: 'codex', partial: false, accounts: [
        envelope('home', { 'gpt-6-astra': ['standard', 'fast', 'ultrafast'] }), envelope('work', { 'gpt-6-astra': ['standard', 'fast'] })] }] };
const tierSegs = (r) => r.strip().querySelectorAll('.model-tier-seg');
const offered = (r) => tierSegs(r).filter((seg) => !seg.hidden).map((seg) => seg.dataset.tier);
const supports = (r) => Object.fromEntries(tierSegs(r).map((seg) => [seg.dataset.tier, seg.dataset.support]));
const means = (r) => r.strip().querySelector('.model-tier-means').textContent;
const opened = async (r) => { await r.open(); await flush(); await flush(); return r; };

test('what follows Main reads Main\'s own account, its pin or Auto; a room\'s own model reads Auto and never asks for Main\'s pin', async () => {
    const pinned = (main) => ({ catalog: TWO_ACCOUNTS, settings: main === undefined ? {} : { OUROBOROS_MODEL_ACCOUNTS: { main } } });
    // Auto: Fast on every account, Ultrafast on some of them; Flex is beyond the transport and is not shown.
    const auto = await opened(room({ storeOptions: pinned() }));
    assert.deepEqual([offered(auto), supports(auto), auto.reads.settings],
        [['standard', 'fast', 'ultrafast'], { economy: 'unlisted', standard: 'listed', fast: 'listed', ultrafast: 'some' }, 1],
        'several accounts may list different tiers: the settings say which one Main is pinned to, once');
    auto.controls.destroy();
    // Main pinned to the account that does not list Ultrafast: nothing confirms it there, least of all the other
    // account's word; the transport carries it, so it can still be tried, said as unconfirmed.
    const work = await opened(room({ storeOptions: pinned('work') }));
    assert.deepEqual([offered(work), supports(work).fast, supports(work).ultrafast], [['standard', 'fast', 'ultrafast'], 'listed', 'unknown']);
    work.controls.destroy();
    const home = await opened(room({ storeOptions: pinned('home'), projectId: '', chatId: 1 }));
    assert.deepEqual([offered(home), supports(home).ultrafast], [['standard', 'fast', 'ultrafast'], 'listed'], 'the Main chat is Main itself');
    home.controls.destroy();
    // The room's own model, the same one: it runs on Auto whatever Main's pin, and no settings are read for it.
    const own = await opened(room({ choice: { kind: 'model', model: ASTRA, use_local: false }, storeOptions: pinned('work') }));
    assert.deepEqual([offered(own), supports(own).ultrafast, own.reads.settings], [['standard', 'fast', 'ultrafast'], 'some', 0]);
    own.controls.destroy();
    // Even without model evidence the sheet reads the pin. An unknown transport offers no new Ultra.
    const single = await opened(room());
    assert.deepEqual([offered(single), single.reads.settings], [['economy', 'standard', 'fast'], 1], 'unconfirmed tiers stand; Ultrafast nothing lists does not');
    single.controls.destroy();
});

test('a pin that is not read, could not be read, or names an account that is gone is never taken for Auto, even with one listing account', async () => {
    // One account in the whole pool, and it lists every tier the transport carries.
    const one = { ...TWO_ACCOUNTS, account_catalogs: [{ source: 'codex', partial: false, accounts: [TWO_ACCOUNTS.account_catalogs[0].accounts[0]] }] };
    const said = async (settings) => {
        const r = await opened(room({ storeOptions: { catalog: one, settings } }));
        const seen = [supports(r).fast, supports(r).ultrafast, offered(r), r.reads.settings];
        r.controls.destroy();
        return seen;
    };
    assert.deepEqual(await said({}), ['listed', 'listed', ['standard', 'fast', 'ultrafast'], 1], 'Main on Auto, as the settings say: the one account is the pool');
    assert.deepEqual(await said({ OUROBOROS_MODEL_ACCOUNTS: { main: 'home' } }), ['listed', 'listed', ['standard', 'fast', 'ultrafast'], 1]);
    assert.deepEqual(await said({ OUROBOROS_MODEL_ACCOUNTS: { main: 'removed' } }), ['unknown', 'unknown', ['standard', 'fast', 'ultrafast'], 1],
        'a saved pin to an account this read does not hold: the one that is left confirms nothing');
    assert.deepEqual(await said(new Error('offline')), ['unknown', 'unknown', ['standard', 'fast', 'ultrafast'], 1],
        'the settings could not be read: the pin is unknown, asked for once, and nothing is confirmed');
});

test('an open follower never borrows the previous Main pin, and a route handover does not poll Settings', async (t) => {
    const replies = [];
    const settings = (doc) => new Promise((resolve) => replies.push((model, main) => resolve({ ...doc,
        OUROBOROS_MODEL: model, OUROBOROS_MODEL_ACCOUNTS: { main } })));
    const r = room({ storeOptions: { catalog: TWO_ACCOUNTS, settings } });
    t.after(() => r.controls.destroy());
    await opened(r);
    replies.shift()(ASTRA, 'home');
    await flush();
    assert.deepEqual([r.reads.settings, supports(r).ultrafast], [1, 'listed']);
    assert.match(means(r), /^Pinned account home · /);

    r.controls.syncState(r.state({ main_route: route(SOL) }));
    assert.match(means(r), /^Account not known · /);
    await flush();
    assert.equal(r.reads.settings, 2, 'a fresh read for the new route even with a cached settings document');
    const next = 'claudexor::codex=gpt-5-old';
    r.controls.syncState(r.state({ main_route: route(next) }));
    await flush();
    assert.equal(r.reads.settings, 2, 'a single read is in flight');
    replies.shift()(SOL, 'work');
    await flush();
    assert.deepEqual([r.reads.settings, supports(r).ultrafast], [3, 'unknown'], 'the latest route gets its read after the older one');
    assert.match(means(r), /^Account not known · /);
    replies.shift()(SOL, 'work'); // Settings still reports the previous route.
    await flush();
    r.controls.syncState(r.state({ main_route: route(next) }));
    await flush();
    assert.equal(r.reads.settings, 3, 'a mismatched read and repaint do not re-request the same binding');
    assert.match(means(r), /^Account not known · /);

    await r.press(r.modelHead);
    await opened(r);
    replies.shift()(next, '');
    await flush();
    assert.deepEqual([r.reads.settings, r.store.mainAccount(route(next))], [4, '']);
    assert.match(means(r), /^Account: Auto · /);
});

test('an open follower asks for Main\'s pin again when Main returns to a route it left, never from a repaint', async () => {
    // Another window moves Main away and back while this sheet stays open; the server drops a pin with its source.
    // What was read before Main left is not the returned Main's: with a key's model between (it has no pin to read),
    // or with another subscription model, whose own read came between.
    for (const between of [GLM, SOL]) {
        const replies = [];
        const settings = (doc) => new Promise((resolve) => replies.push((model, main) => resolve({ ...doc,
            OUROBOROS_MODEL: model, OUROBOROS_MODEL_ACCOUNTS: { main } })));
        const r = room({ storeOptions: { catalog: TWO_ACCOUNTS, settings } });
        await opened(r);
        replies.shift()(ASTRA, 'home');
        await flush();
        assert.deepEqual([r.reads.settings, supports(r).ultrafast], [1, 'listed']);
        assert.match(means(r), /^Pinned account home · /);

        r.controls.syncState(r.state({ main_route: route(between) }));
        await flush();
        const asked = between === SOL ? 2 : 1;
        assert.equal(r.reads.settings, asked, between);
        if (between === SOL) { replies.shift()(SOL, 'home'); await flush(); }

        r.controls.syncState(r.state({ main_route: route(ASTRA) }));
        assert.match(means(r), /^Account not known · /, `${between}: the pin read before Main left is not the returned Main's`);
        await flush();
        assert.equal(r.reads.settings, asked + 1, `${between}: one read for the route Main returned to`);
        replies.shift()(ASTRA, '');
        await flush();
        assert.match(means(r), /^Account: Auto · /);
        assert.equal(supports(r).ultrafast, 'some');

        r.controls.syncState(r.state({ main_route: route(ASTRA) }));
        await flush();
        assert.equal(r.reads.settings, asked + 1, `${between}: a repaint on the same route asks nothing`);
        r.controls.destroy();
    }
});

test('reopening a follower refreshes Main’s pin on the same model, even behind an older GET', async (t) => {
    const replies = [];
    const settings = (doc) => new Promise((resolve, reject) => replies.push({
        pin: (main) => resolve({ ...doc, OUROBOROS_MODEL_ACCOUNTS: { main } }), reject,
    }));
    const r = room({ storeOptions: { catalog: TWO_ACCOUNTS, settings } });
    t.after(() => r.controls.destroy());
    await opened(r);
    replies.shift().pin('home');
    await flush();
    assert.deepEqual([r.reads.settings, supports(r).ultrafast], [1, 'listed']);
    assert.match(means(r), /^Pinned account home · /);

    // An earlier read (for example, a binding retry) has already sampled home. The other window moves only the pin.
    await r.press(r.modelHead);
    const old = r.store.readConnections({ fresh: true });
    await flush();
    const stale = replies.shift();
    await opened(r);
    const sheet = r.sheet();
    const field = r.input();
    await r.type('astra');
    assert.deepEqual([r.reads.settings, supports(r).ultrafast, r.doc.activeElement], [2, 'unknown', field]);
    assert.match(means(r), /^Account not known · /);
    stale.pin('home');
    await flush();
    assert.deepEqual([r.reads.settings, supports(r).ultrafast], [3, 'unknown'], 'the older GET does not repaint home');
    replies.shift().pin('work');
    await old;
    assert.match(means(r), /^Pinned account work · /);
    assert.deepEqual([supports(r).fast, supports(r).ultrafast], ['listed', 'unknown']);
    assert.equal(r.value(), 'gpt-6-astra');
    assert.equal(r.sheet(), sheet);
    assert.equal(r.doc.activeElement, field);
    assert.equal(field.value, 'astra');
    r.controls.syncState(r.state());
    await flush();
    assert.equal(r.reads.settings, 3, 'typing and state paints do not refresh the pin');

    await r.press(r.modelHead);
    await opened(r);
    assert.match(means(r), /^Account not known · /, 'even a cached pin stays unconfirmed during the opening read');
    replies.shift().reject(new Error('offline'));
    await flush();
    assert.match(means(r), /^Account not known · /);
    assert.deepEqual([r.reads.settings, supports(r).fast, supports(r).ultrafast], [4, 'unknown', 'unknown'], 'a failed read does not restore work or retry');
});

test('a route proven to have only its ordinary service has no strip; an inherited or chosen tier stays in view and can be left', async () => {
    const onlyStandard = { ...CATALOG, model_sources: [{ ...CATALOG.model_sources[0], processingPreferences: ['standard'] }] };
    const plain = await opened(room({ storeOptions: { catalog: onlyStandard } }));
    assert.deepEqual([plain.strip().querySelector('.model-tier-row').hidden, offered(plain), plain.tierWord()], [true, [], ''], 'nothing to choose: no strip');
    plain.controls.destroy();
    // Main is on Fast: the follower inherits a wish its route does not serve. It is said, ringed, and can be overridden.
    const inherited = await opened(room({ tiers: { main: 'fast', default: 'standard' }, storeOptions: { catalog: onlyStandard } }));
    assert.deepEqual([inherited.strip().hidden, offered(inherited), inherited.tierView().ring, inherited.tierView().checked, inherited.tierWord()],
        [false, ['standard', 'fast'], ['fast'], [], 'Fast'], 'Default is not Standard');
    assert.equal(means(inherited), 'Account: Auto · Requests faster processing. It may cost more. Codex does not list Fast for gpt-6-astra: its calls may go as Standard.');
    await inherited.chooseTier('standard');
    await inherited.controls.pendingSave();
    assert.deepEqual([inherited.tierPosts(), offered(inherited), inherited.tierView().checked, inherited.tierView().reset],
        [[{ processing: 'standard', project_id: 'p7' }], ['standard'], ['standard'], true],
        'an explicit Standard is a choice, with Default beside it; the tier this route does not serve leaves with the wish for it');
    inherited.controls.destroy();
    // The room's own stored Flex on such a route is not wiped or hidden.
    const kept = await opened(room({ tier: 'economy', storeOptions: { catalog: onlyStandard } }));
    assert.deepEqual([offered(kept), kept.tierView().checked, supports(kept).economy, kept.tierWord()], [['economy', 'standard'], ['economy'], 'unlisted', 'Flex']);
    kept.controls.destroy();
});

test('Main and a room can request Standard over a workspace default even when their key lists only Standard', async () => {
    const options = { main: GLM, tiers: { main: '', default: '' }, storeOptions: {
        catalog: { ...CATALOG, api_sources: [{ id: 'openrouter', processing_transport: ['standard', 'economy', 'fast', 'ultrafast'] }] },
        processing: (value) => ({ value, processing: { modes: ['standard'], complete: true, endpoints: [] } }),
    } };
    for (const projectId of ['', 'p7']) {
        const r = await opened(room({ ...options, projectId }));
        try {
            assert.deepEqual([offered(r), r.tierView().checked, r.tierView().reset], [['standard'], [], false]);
            assert.match(means(r), /No tier requested.*OpenRouter may apply a workspace default/);
            if (!projectId) r.tierAnswer.value = { ok: true, project_id: '', main_processing: 'standard', processing_default: '' };
            await r.chooseTier('standard');
            await r.controls.pendingSave();
            assert.deepEqual([offered(r), r.tierView().checked, r.tierView().reset, r.tierWord()], [['standard'], ['standard'], true, '']);
            if (!projectId) r.tierAnswer.value = { ok: true, project_id: '', main_processing: '', processing_default: '' };
            await r.chooseTier('');
            await r.controls.pendingSave();
            assert.deepEqual([offered(r), r.tierView().checked, r.tierView().reset], [['standard'], [], false]);
            assert.deepEqual(r.tierPosts(), ['standard', ''].map((processing) => ({ processing, ...(projectId ? { project_id: projectId } : {}) })));
        } finally { r.controls.destroy(); }
    }
});

test('a key model\'s own endpoints are read for the route the sheet is on, when it opens or that route changes, never for the list\'s rows', async () => {
    const tiersOf = { [GLM]: ['standard', 'fast'], [KIMI]: ['standard', 'economy', 'fast'] };
    const processing = (value) => ({ value, processing: { modes: tiersOf[value], complete: true, endpoints: [
        { tag: 'a', mode: 'standard', price: { input: 1, output: 4 } }, { tag: 'a:fast', service_tier: 'priority', mode: 'fast', price: { input: 2, output: 8 } },
        { tag: 'b:fast', service_tier: 'priority', mode: 'fast', price: { input: 2, output: 8 } }] } });
    const r = room({ choice: { kind: 'model', model: GLM, use_local: false }, storeOptions: { recents: [route(KIMI), route('x-ai/grok-4.7')], processing } });
    assert.deepEqual(r.reads.processing, [], 'a closed capsule asks nothing');
    await opened(r);
    assert.deepEqual(r.reads.processing, [GLM], 'the open sheet\'s own route, once; no recent, pinned or listed row');
    assert.deepEqual([offered(r), supports(r)], [['standard', 'fast'], { economy: 'unlisted', standard: 'listed', fast: 'listed', ultrafast: 'unlisted' }],
        'the whole endpoint list leaves Flex and Ultrafast out');
    await r.fire(r.strip(), 'focusin', { type: 'focusin', target: { closest: () => tierSegs(r).find((seg) => seg.dataset.tier === 'fast') } });
    assert.equal(means(r), 'Requests faster processing. It may cost more. Listed at $2 in / $8 out per 1M tokens.', 'what the tier is listed at, the same at every provider');
    r.controls.syncState(r.state());
    await flush();
    assert.deepEqual(r.reads.processing, [GLM], 'a state body and a repaint ask nothing again');
    // Another model is picked: the sheet closes with the pick; opened again it asks for the new route alone.
    await r.take(`route|${KIMI}`);
    await r.controls.pendingSave();
    await opened(r);
    assert.deepEqual([r.reads.processing, offered(r)], [[GLM, KIMI], ['economy', 'standard', 'fast']]);
    r.controls.destroy();
    // A subscription route has no such read, and a failed one leaves the tiers unconfirmed, never Standard only.
    const codex = await opened(room({ storeOptions: { processing } }));
    assert.deepEqual(codex.reads.processing, []);
    codex.controls.destroy();
    const offline = await opened(room({ choice: { kind: 'model', model: GLM, use_local: false } }));
    assert.deepEqual([offline.reads.processing, offered(offline), supports(offline).fast], [[GLM], ['economy', 'standard', 'fast'], 'unknown']);
    assert.equal(means(offline), 'Request Standard processing. Price unknown.');
    offline.controls.destroy();
});

test('Ultrafast for Main asks first, for the route and account Main has: a cancel writes nothing, a Main that moved is not written', async () => {
    const questions = [];
    const answer = { value: false };
    const main = (options = {}) => room({ projectId: '', chatId: 1, storeOptions: { catalog: TWO_ACCOUNTS,
        settings: { OUROBOROS_MODEL_ACCOUNTS: { main: 'home' }, OUROBOROS_PROCESSING_PREFERENCE: 'standard' } },
        model: { askTier: async (facts) => { questions.push(facts); return typeof answer.value === 'function' ? answer.value() : answer.value; } }, ...options });
    const r = await opened(main());
    assert.deepEqual(offered(r), ['standard', 'fast', 'ultrafast'], 'Main is offered Ultrafast where its own route and account list it');
    // Cancel: nothing is written, Main keeps its tier, and the question stood alone (the sheet closed for it).
    await r.chooseTier('ultrafast');
    await r.controls.pendingSave();
    assert.equal(questions.length, 1);
    const { doc: _doc, ...facts } = questions[0];
    assert.deepEqual(facts, { model: 'gpt-6-astra', source: 'Codex · Pinned account home', pinned: true, priceSaid: false,
        explain: 'Requests the fastest processing this source offers. It may cost more.' }, 'the route, what it says of Ultrafast, and no price it was not told');
    assert.deepEqual([r.tierPosts(), r.store.tiers(), r.sheet(), r.tierWord(), r.toasts, r.controls.hasPendingSave()],
        [[], STANDARD, null, '', [], false], 'a cancel writes nothing');
    // Fast and Flex still save at once, without a question.
    await opened(r);
    await r.chooseTier('fast');
    await r.controls.pendingSave();
    assert.deepEqual([questions.length, r.tierPosts()], [1, [{ processing: 'fast' }]]);
    // Confirmed: one write of Main's own tier. Even a complete model census cannot narrow the question's scope.
    r.settings.OUROBOROS_MODEL_PROCESSING_PREFERENCES = { main: 'fast' };
    const fastTokens = { ...MAIN_TOKENS, OUROBOROS_MODEL_PROCESSING_PREFERENCES: 'e'.repeat(24) };
    r.settings._meta = { tokens: fastTokens };
    answer.value = () => {
        // Another read after the dialog would describe a different saved model.
        r.settings.OUROBOROS_MODEL = SOL;
        r.settings._meta = { tokens: { ...fastTokens, OUROBOROS_MODEL: 'f'.repeat(24) } };
        return true;
    };
    const readsBefore = r.reads.settings;
    r.controls.syncState(r.state({ processing_tiers: { main: 'fast', default: 'standard' }, works_complete: true, supervisor_ready: true,
        works: [MAIN_FALLBACK_WORK] }));
    await r.chooseTier('ultrafast');
    await r.controls.pendingSave();
    assert.equal(Object.hasOwn(questions[1], 'scope'), false, 'the complete but fallback-only census must not override the shared disclosure');
    assert.equal(r.reads.settings, readsBefore + 1);
    assert.equal(questions.length, 2, 'one question per Ultrafast press, including the cancelled one');
    assert.deepEqual(r.tierPosts(), [{ processing: 'fast' }, { processing: 'ultrafast', if_unchanged: { ...fastTokens, ultrafast: JSON.stringify([ASTRA, false, 'home', 'store-a', 'fp-home']) } }], 'a subscription has no API connection to bind');
    assert.deepEqual([r.store.tiers().main, r.tierWord(), r.modelPosts()], ['ultrafast', 'Ultrafast', []]);
    // The server now holds what was written: a sheet opening reads Main's pin fresh (model_sheet.js), and that read is
    // the stored document, not the one the question was asked on.
    r.settings.OUROBOROS_MODEL = ASTRA;
    r.settings.OUROBOROS_MODEL_PROCESSING_PREFERENCES = { main: 'ultrafast' };
    await opened(r);
    assert.deepEqual([r.tierView().checked, offered(r)], [['ultrafast'], ['standard', 'fast', 'ultrafast']], 'reopened: the choice is there');
    r.controls.destroy();
    // Main moved to another model while the question stood: it was not asked about that one, so nothing is written.
    const moved = await opened(main());
    answer.value = () => { moved.controls.syncState(moved.state({ main_route: route(GLM) })); return true; };
    await moved.chooseTier('ultrafast');
    await moved.controls.pendingSave();
    assert.deepEqual([moved.tierPosts(), moved.toasts], [[], [['Main changed while you were asked: Ultrafast was not set.', 'info']]]);
    moved.controls.destroy();
    // Its account moved under the question (a pin dropped elsewhere): the same.
    const unpinned = await opened(main());
    answer.value = () => { unpinned.store.adoptMainAccount(''); return true; };
    await unpinned.chooseTier('ultrafast');
    await unpinned.controls.pendingSave();
    assert.deepEqual([unpinned.tierPosts(), unpinned.toasts.length], [[], 1]);
    unpinned.controls.destroy();
    // The catalog was read again from another engine store under the question: the account's word it was asked on
    // is not this store's.
    const restored = await opened(main());
    answer.value = () => { restored.store.adopt({ ...TWO_ACCOUNTS, store_id: 'another-store' }); return true; };
    await restored.chooseTier('ultrafast');
    await restored.controls.pendingSave();
    assert.deepEqual([restored.tierPosts(), restored.toasts.length], [[], 1]);
    restored.controls.destroy();
    const reauthed = await opened(main());
    const catalog = structuredClone(TWO_ACCOUNTS);
    const home = catalog.account_catalogs[0].accounts[0].catalog;
    home.accountFingerprint = 'first';
    reauthed.store.adopt(catalog);
    answer.value = () => { home.accountFingerprint = 'replacement'; reauthed.store.adopt(catalog); return true; };
    await reauthed.chooseTier('ultrafast');
    await reauthed.controls.pendingSave();
    assert.deepEqual([reauthed.tierPosts(), reauthed.toasts.length], [[], 1], 'the same pin in the same store changed identity');
    reauthed.controls.destroy();
    // A room's Ultrafast is the room's own row: no question, as before.
    const own = await opened(room({ choice: { kind: 'model', model: ASTRA, use_local: false }, storeOptions: { catalog: TWO_ACCOUNTS },
        model: { askTier: async (facts) => { questions.push(facts); return false; } } }));
    const before = questions.length;
    await own.chooseTier('ultrafast');
    await own.controls.pendingSave();
    assert.deepEqual([questions.length, own.tierPosts()], [before, [{ processing: 'ultrafast', project_id: 'p7' }]]);
    own.controls.destroy();
});

test('Main Ultrafast needs Settings tokens for the binding actually on screen', async () => {
    for (const changed of [{ OUROBOROS_MODEL: SOL }, { USE_LOCAL_MAIN: true }, { OUROBOROS_MODEL_ACCOUNTS: { main: 'work' } },
        { OUROBOROS_MODEL_PROCESSING_PREFERENCES: { main: 'standard' } }, { _meta: { tokens: { OUROBOROS_MODEL: MAIN_TOKENS.OUROBOROS_MODEL } } }]) {
        const questions = [];
        const r = await opened(room({ projectId: '', storeOptions: { catalog: TWO_ACCOUNTS, settings: { OUROBOROS_MODEL_ACCOUNTS: { main: 'home' } } },
            model: { askTier: async (question) => { questions.push(question); return true; } } }));
        Object.assign(r.settings, changed);
        await r.chooseTier('ultrafast');
        await r.controls.pendingSave();
        assert.deepEqual([questions, r.tierPosts(), r.toasts.length], [[], [], 1], JSON.stringify(changed));
        if (changed.OUROBOROS_MODEL_ACCOUNTS) assert.equal(r.store.mainAccount(), 'work', 'the new pin is available for the next choice');
        r.controls.destroy();
    }
});

test('on a transport known to carry it, Main may try an Ultrafast nothing confirms: said so in the strip and in its question, with no price it was not told', async () => {
    const responses = { provider_id: 'openai', provider: 'OpenAI', id: 'gpt-6', name: 'gpt-6', value: 'openai::gpt-6' };
    const main = (transport) => room({ projectId: '', chatId: 1, main: 'openai::gpt-6', model: { askTier: async (facts) => { questions.push(facts); return true; } },
        storeOptions: { connections: [{ id: 'openai', configured: true, revision: OPENAI_SOURCE }],
            catalog: { ...CATALOG, items: [...CATALOG.items, { ...responses, ...(transport ? { processing_transport: transport } : {}) }] } } });
    const questions = [];
    const r = await opened(main(['standard', 'fast', 'economy', 'ultrafast']));
    assert.deepEqual([offered(r), supports(r).ultrafast, r.reads.settings, r.reads.processing], [['economy', 'standard', 'fast', 'ultrafast'], 'unknown', 0, []],
        'no per-model or per-account word, and none is asked for: the tier is offered as unconfirmed');
    // An empty census does not override the shared disclosure either.
    r.controls.syncState(r.state({ works: [], works_complete: true, supervisor_ready: true }));
    await r.chooseTier('ultrafast');
    await r.controls.pendingSave();
    const { doc: _doc, ...facts } = questions[0];
    assert.deepEqual(facts, { model: 'gpt-6', source: 'OpenAI', priceSaid: true, pinned: false,
        explain: 'Requests the fastest processing this source offers. It may cost more. Not confirmed for gpt-6 here: a provider may serve it as Standard. Price unknown.' });
    assert.deepEqual(r.tierPosts(), [{ processing: 'ultrafast', if_unchanged: { ...MAIN_TOKENS, connection: OPENAI_SOURCE, ultrafast: JSON.stringify(['openai::gpt-6', false, '', '', '']) } }],
        'on the key\'s connection as the question was shown it');
    r.controls.destroy();
    // A transport that says nothing, and one that carries no Ultrafast (Chat Completions): no new Ultrafast; a saved one stays.
    for (const transport of [null, ['standard', 'fast', 'economy']]) {
        const plain = await opened(main(transport));
        assert.deepEqual(offered(plain), ['economy', 'standard', 'fast']);
        plain.controls.destroy();
        const kept = await opened(room({ projectId: '', chatId: 1, main: 'openai::gpt-6', tiers: { main: 'ultrafast', default: 'standard' },
            storeOptions: { catalog: { ...CATALOG, items: [...CATALOG.items, { ...responses, ...(transport ? { processing_transport: transport } : {}) }] } } }));
        assert.deepEqual([offered(kept), kept.tierView().checked, kept.tierView().reset, kept.tierWord()], [['economy', 'standard', 'fast', 'ultrafast'], ['ultrafast'], true, 'Ultrafast']);
        kept.controls.destroy();
    }
});

test('a change of Main says what the new route makes of Main\'s own tier, before it is written', async () => {
    const r = await opened(room({ projectId: '', chatId: 1, tiers: { main: 'ultrafast', default: 'standard' }, storeOptions: { recents: [route(GLM)] } }));
    await r.take(`route|${GLM}`);
    await flush();
    assert.ok(r.asked[0].tierNote.startsWith('Main keeps its tier, Ultrafast. Requests the fastest processing this source offers. It may cost more. '
        + 'Not confirmed for glm-5.3 here: a provider may serve it as Standard. Price unknown.'));
    assert.match(r.asked[0].tierNote, /Direct reviews using Main and Safety falling back to Main/);
    r.controls.destroy();
    const plain = await opened(room({ projectId: '', chatId: 1, storeOptions: { recents: [route(GLM)] } }));
    await plain.take(`route|${GLM}`);
    await flush();
    assert.equal(plain.asked[0].tierNote, '', 'Standard and Default go with any route: nothing to say');
    plain.controls.destroy();
    // The sheet's own line, on the shared dialog.
    const doc = { createElement: (tag) => new ElementStub(tag, doc), byId: new Map() };
    const content = doc.createElement('section');
    await confirmMainChange({ next: { ...describeRoute(null, route(GLM)) }, previous: null, tierNote: 'Main keeps its tier, Fast.', doc,
        dialog: async (options) => { options.content.mount(content, null); return false; } });
    assert.ok(content.children.some((line) => line.textContent === 'Main keeps its tier, Fast.'));
});

// A thenable the writer awaits: the server's typed refusal, without an unhandled rejection before it is read.
const refusedWith = (code) => ({ then: (_resolve, reject) => reject(Object.assign(new Error('Changed on another device.'), { body: { code } })) });

test('a pinned Main Ultrafast refusal rereads the account catalog; the owner\'s next choice is saved on what that read names, and nothing is written for them', async () => {
    for (const writer of ['tier', 'model']) {
        const first = structuredClone(TWO_ACCOUNTS);
        const again = structuredClone(first);
        again.store_id = 'store-after-conflict';
        again.account_catalogs[0].accounts[0].catalog.accountFingerprint = 'new-home';
        let catalog = first;
        // The server's own comparison (owner_effort.check_main_ultrafast): the confirmed pin, store and fingerprint
        // against its last catalog observation, which is what it last served and holds only while the engine has not
        // moved on. So the second write is saved for the identity the reread brought, never for a repeat of the first.
        let served = null;
        const observed = () => (served === catalog ? ['home', served.store_id, served.account_catalogs[0].accounts[0].catalog.accountFingerprint] : null);
        const tiers = { main: writer === 'model' ? 'ultrafast' : '', default: 'standard' };
        // Each writer reaches its write as the owner does: the question about Ultrafast answered yes, the picked row in view.
        const r = await opened(room({ projectId: '', chatId: 1, tiers, model: { askTier: async () => true },
            storeOptions: { catalog: () => { served = catalog; return catalog; }, recents: [route(SOL)],
                settings: { OUROBOROS_MODEL_ACCOUNTS: { main: 'home' }, OUROBOROS_MODEL_PROCESSING_PREFERENCES: { main: tiers.main } } } }));
        const choose = () => writer === 'model' ? r.take(`route|${SOL}`) : r.chooseTier('ultrafast');
        const requests = writer === 'model' ? r.modelPosts : r.tierPosts;
        const answer = writer === 'model' ? r.modelAnswer : r.tierAnswer;
        const sent = (at) => JSON.parse(requests().at(at).if_unchanged.ultrafast).slice(2);
        answer.value = { then: (resolve, reject) => (JSON.stringify(sent(-1)) === JSON.stringify(observed())
            ? resolve(null) : refusedWith('settings_changed_elsewhere').then(resolve, reject)) };
        const before = r.reads.catalog;
        catalog = again;  // The saved model, tier and pin name did not change; its engine observation did.
        await choose();
        await flush();
        assert.equal(requests().length, 1);
        assert.deepEqual(sent(0), ['home', 'store-a', 'fp-home']);
        assert.equal(r.store.index().storeId, 'store-after-conflict');
        assert.equal(r.reads.catalog, before + 1);
        assert.match(r.toasts.at(-1)[0], /changed/i);
        assert.deepEqual([r.store.tiers().main, r.value()], [tiers.main, 'gpt-6-astra'], 'refused: nothing was changed');

        const said = r.toasts.length;
        await opened(r);
        await choose();
        await flush();
        assert.equal(requests().length, 2, 'only a second owner choice sends a second write');
        assert.deepEqual(sent(1), ['home', 'store-after-conflict', 'new-home']);
        assert.deepEqual([r.store.tiers().main, r.tierWord(), r.value()], ['ultrafast', 'Ultrafast', writer === 'model' ? 'gpt-6.1-sol' : 'gpt-6-astra'],
            'saved: the same choice, on the store and fingerprint the reread named');
        assert.deepEqual([r.reads.catalog, r.toasts.length], [before + 1, said], 'on that one reread, with nothing more to say');
        r.controls.destroy();
    }
});

test('two windows: a change of Main\'s model stands on Main as stored; a tier set elsewhere is said first, a move after the question is refused', async () => {
    // Window 2 set Main's Ultrafast after this page last read the tiers: the confirming sheet says what the new model
    // keeps, and the write carries the binding the owner was shown (the read's four tokens).
    const r = await opened(room({ projectId: '', chatId: 1, storeOptions: { recents: [route(GLM)] } }));
    const elsewhere = { ...MAIN_TOKENS, OUROBOROS_MODEL_PROCESSING_PREFERENCES: 'u'.repeat(24) };
    Object.assign(r.settings, { OUROBOROS_MODEL_PROCESSING_PREFERENCES: { main: 'ultrafast' }, _meta: { tokens: elsewhere, connections: [{ id: 'openrouter', configured: true, revision: OR_SOURCE }] } });
    await r.take(`route|${GLM}`);
    await flush();
    assert.match(r.asked[0].tierNote, /^Main keeps its tier, Ultrafast\. /, 'the tier the new model keeps, as stored now');
    assert.deepEqual([r.store.tiers().main, r.modelPosts()], ['ultrafast', [{ kind: 'model', model: GLM, use_local: false, if_unchanged: { ...elsewhere, connection: OR_SOURCE, ultrafast: JSON.stringify([GLM, false, '', '', '']) } }]],
        'the Ultrafast it carries stands on the picked key as shown');
    r.controls.destroy();
    // Window 2 changed Main again while the question stood: the server refuses on the tokens, nothing is said saved.
    const late = await opened(room({ projectId: '', chatId: 1, storeOptions: { recents: [route(GLM)] } }));
    late.modelAnswer.value = refusedWith('settings_changed_elsewhere');
    const reads = late.reads.settings;
    await late.take(`route|${GLM}`);
    await flush();
    assert.deepEqual([late.value(), late.toasts.at(-1)], ['gpt-6-astra',
        ['Main changed on another device. Check its model and tier, then choose again: nothing was changed.', 'info']]);
    assert.ok(late.reads.settings > reads + 1, 'read again after the refusal');
    late.controls.destroy();
    // Main itself moved before the question: nothing is asked about the Main that is gone.
    const gone = await opened(room({ projectId: '', chatId: 1, storeOptions: { recents: [route(GLM)] } }));
    gone.settings.OUROBOROS_MODEL = SOL;
    await gone.take(`route|${GLM}`);
    await flush();
    assert.deepEqual([gone.asked, gone.modelPosts(), gone.toasts.at(-1)?.[1]], [[], [], 'info']);
    gone.controls.destroy();
});

test('a pin this page had not read is learnt at the press, never a change of Main; a change after the question still refuses', async () => {
    const questions = [];
    const answer = { value: true };
    const main = () => {
        const made = room({ projectId: '', chatId: 1, storeOptions: { catalog: TWO_ACCOUNTS, settings: { OUROBOROS_MODEL_ACCOUNTS: { main: 'home' } } },
            model: { askTier: async (facts) => { questions.push(facts); return typeof answer.value === 'function' ? answer.value(made) : answer.value; } } });
        made.store.readConnections = async () => {};  // the page never read Main's pin before the press
        return made;
    };
    const r = await opened(main());
    assert.equal(r.store.mainAccount(), null);
    await r.chooseTier('ultrafast');
    await r.controls.pendingSave();
    assert.deepEqual([r.toasts, questions.length, questions[0].source], [[], 1, 'Codex · Pinned account home'], 'no "Main changed": the pin was only not known');
    assert.deepEqual(r.tierPosts(), [{ processing: 'ultrafast', if_unchanged: { ...MAIN_TOKENS, ultrafast: JSON.stringify([ASTRA, false, 'home', 'store-a', 'fp-home']) } }]);
    r.controls.destroy();
    const moved = await opened(main());
    answer.value = (made) => { made.store.adoptMainAccount('work'); return true; };
    await moved.chooseTier('ultrafast');
    await moved.controls.pendingSave();
    assert.deepEqual([moved.tierPosts(), moved.toasts], [[], [['Main changed while you were asked: Ultrafast was not set.', 'info']]]);
    moved.controls.destroy();
});

test('a local Main takes what the local adapter can send: Fast is no new choice there, and an inherited one says it is not sent', async () => {
    const local = { provider_id: 'local', provider: 'Local', id: 'qwen3', name: 'qwen3', value: 'qwen3', use_local: true, processing_transport: ['standard'] };
    const r = await opened(room({ projectId: '', chatId: 1, main: 'qwen3', tiers: { main: 'fast', default: 'standard' },
        storeOptions: { catalog: { ...CATALOG, items: [...CATALOG.items, local] } } }));
    r.controls.syncState(r.state({ main_route: route('qwen3', true) }));
    await flush();
    assert.deepEqual([offered(r), supports(r).fast, r.tierView().checked], [['standard', 'fast'], 'unlisted', ['fast']],
        'the inherited choice stays in view to be left; nothing else is offered');
    assert.match(means(r), /does not send Fast\./);
    r.controls.destroy();
});

test('a key that changed between the sheet and the press is read again before any question; the yes stands on the key shown', async () => {
    const questions = [];
    const responses = { provider_id: 'openai', provider: 'OpenAI', id: 'gpt-6', name: 'gpt-6', value: 'openai::gpt-6', processing_transport: ['standard', 'fast', 'economy', 'ultrafast'] };
    const r = await opened(room({ projectId: '', chatId: 1, main: 'openai::gpt-6', model: { askTier: async (facts) => { questions.push(facts); return true; } },
        storeOptions: { connections: [{ id: 'openai', configured: true, revision: OPENAI_SOURCE }], catalog: { ...CATALOG, items: [...CATALOG.items, responses] } } }));
    await r.store.readConnections({ fresh: true });  // the sheet shows the key as this read has it
    assert.equal(r.store.revisionOf(route('openai::gpt-6')), OPENAI_SOURCE);
    // Another tab rotates the key (or its API surface) under the same routed name.
    r.settings._meta = { tokens: MAIN_TOKENS, connections: [{ id: 'openai', configured: true, revision: 'openai#r2' }] };
    const catalogReads = r.reads.catalog;
    await r.chooseTier('ultrafast');
    await r.controls.pendingSave();
    assert.deepEqual([questions, r.tierPosts()], [[], []], 'nothing is asked or written on a key the owner was not shown');
    assert.match(r.toasts.at(-1)[0], /changed on another device\. Its tiers were read again/);
    assert.equal(r.store.revisionOf(route('openai::gpt-6')), 'openai#r2');
    assert.ok(r.reads.catalog > catalogReads, 'the transport this source carries is read again, never the earlier one\'s');
    // Pressed again, the question is about the key now shown, and the write stands on it.
    await opened(r);
    await r.chooseTier('ultrafast');
    await r.controls.pendingSave();
    assert.equal(questions.length, 1);
    assert.deepEqual(r.tierPosts(), [{ processing: 'ultrafast', if_unchanged: { ...MAIN_TOKENS, connection: 'openai#r2', ultrafast: JSON.stringify(['openai::gpt-6', false, '', '', '']) } }]);
    r.controls.destroy();
});

test('a change of Main that carries Ultrafast onto a key the sheet showed otherwise asks nothing until that key is read again', async () => {
    const r = await opened(room({ projectId: '', chatId: 1, tiers: { main: 'ultrafast', default: 'standard' }, storeOptions: { recents: [route(GLM)] } }));
    await r.store.readConnections({ fresh: true });
    r.settings._meta = { tokens: MAIN_TOKENS, connections: [{ id: 'openrouter', configured: true, revision: 'openrouter#r2' }] };
    const asked = r.reads.processing.length;
    await r.take(`route|${GLM}`);
    await flush();
    assert.deepEqual([r.asked, r.modelPosts()], [[], []]);
    assert.ok(r.reads.processing.length > asked, 'the picked model\'s endpoints are read under the key it has now');
    await r.open();
    await r.take(`route|${GLM}`);
    await flush();
    assert.equal(r.asked.length, 1);
    assert.deepEqual(r.modelPosts(), [{ kind: 'model', model: GLM, use_local: false, if_unchanged: { ...MAIN_TOKENS, connection: 'openrouter#r2', ultrafast: JSON.stringify([GLM, false, '', '', '']) } }]);
    // Without Ultrafast nothing of the key is bound: an ordinary change of Main stays as it was.
    const plain = await opened(room({ projectId: '', chatId: 1, storeOptions: { recents: [route(GLM)] } }));
    await plain.take(`route|${GLM}`);
    await flush();
    assert.deepEqual(plain.modelPosts(), [{ kind: 'model', model: GLM, use_local: false, if_unchanged: MAIN_TOKENS }]);
    r.controls.destroy();
    plain.controls.destroy();
});

test('a sheet dismissed while its key\'s endpoints are read opens no question afterwards', async () => {
    const questions = [];
    let release = () => {};
    const glm = { ...openrouter(GLM), processing_transport: ['standard', 'fast', 'ultrafast'] };
    const r = await opened(room({ projectId: '', chatId: 1, main: GLM, model: { askTier: async (facts) => { questions.push(facts); return true; } },
        storeOptions: { catalog: { ...CATALOG, items: [...CATALOG.items.filter((row) => row.value !== GLM), glm] },
            processing: (value) => new Promise((resolve) => { release = () => resolve({ value, processing: { modes: ['standard', 'ultrafast'], complete: true, endpoints: [] } }); }) } }));
    await r.store.readConnections({ fresh: true });
    r.store.readDetail(route(GLM)).catch(() => {});
    await r.chooseTier('ultrafast');
    await flush();
    r.controls.model.capsule.snapClosed();  // dismissed while the endpoints are read
    release();
    await r.controls.pendingSave();
    assert.deepEqual([questions, r.tierPosts()], [[], []], 'the dismissed press asks nothing and writes nothing');
    r.controls.destroy();
});
