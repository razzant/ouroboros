// The Model capsule (composer_model.js; DESIGN "Composer owner controls"): the first capsule of the row. A room's
// capsule, the Main chat's included, holds the model its next work starts on; it follows Main until the room chooses,
// and says so; closed it shows the room's choice, never a running work's model, and its processing tier when that is
// not Standard. A press, never hover, opens the Model sheet (model_sheet.js) over the page: the pinned and recent
// models, all sources, and the tier strip at its foot; no row follows Main again. A pick saves at once for its room
// and never writes Main; one that leaves the room on another model arms this window's next message as new work; a
// tier saves at once, leaves the sheet open and arms nothing; a Pin chooses nothing. The list's own grammar is
// model_list.test.js; engines, geometry and the sheet's place are the browser lane's.
import assert from 'node:assert/strict';
import test from 'node:test';
import { GENERIC_HARNESS_MARK, HARNESS_MARKS } from '../modules/harness_presentation.js';
import { CHAT_NOT_READ, SERVER_OLDER, sameChoice } from '../modules/composer_model.js';
import { createModelCatalogStore } from '../modules/model_catalog_store.js';
import { closeModelSheet } from '../modules/model_sheet.js';
import { CHAT_ULTRAFAST_FOLLOWS, CHAT_ULTRAFAST_PROFILE_SCOPE, CHAT_ULTRAFAST_SCOPE, confirmChatUltrafast } from '../modules/processing_choice.js';
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
// An API connection's opaque revision (`_meta.connections[].revision`, its picker id first).
const OR_SOURCE = 'openrouter#r1';
// The token the fixture's writer answers a saved choice with (composer_controls_fixture.js), as the host's own does.
const tokenOf = (projectId, model) => `token:${projectId}:${model}`;
// What a sheet says above its rows while its room still follows Main, before any pick (the fixture's outline form).
const OWN_ROUTE = 'New tasks here take the model you pick, on Auto across its source’s accounts and with its own context window; running tasks keep theirs.';
const FOLLOWS = `lead: This model is set in Settings › Models. ${OWN_ROUTE}`;
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
                _meta: { connections, tokens: { OUROBOROS_MODEL_LIGHT: 'unrelated' } } };
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
const MAIN_CHAT = { scope: 'main_chat' };  // how a save names the Main chat (a room is named by its project_id)

/* A room's composer row (or the Main chat's, `projectId: ''`) with its Model capsule, fed one state body: Main's route,
   the tiers (`processing_tiers`), and the room's own model (`choice`) and tier (`tier`): a Project's on its row, the
   Main chat's as `main_chat`. `asked` holds the Ultrafast questions put (`confirmed.value` answers them), `armings`
   counts the capsule's words that what the next message is sent as may have changed. */
function room({ projectId = 'p7', chatId = 7, choice = { kind: 'main' }, tier = '', main = ASTRA, tiers = STANDARD, model = {},
    storeOptions, ...rest } = {}) {
    const kept = storeOf({ main, settings: { OUROBOROS_MODEL_PROCESSING_PREFERENCES: { main: tiers?.main || '' },
        OUROBOROS_PROCESSING_PREFERENCE: tiers?.default || '' }, ...storeOptions });
    const asked = [];
    const confirmed = { value: true };
    const armings = { count: 0 };
    const m = mountPair({ projectId, chatId, subagents: true, ...rest, model: { store: kept.store,
        askTier: async (request) => { asked.push(request); return confirmed.value; },
        onNewWork: () => { armings.count += 1; }, ...model } });
    const state = (extra = {}) => ({ ...(projectId ? {} : COGNITION), main_route: route(main), ...(tiers ? { processing_tiers: tiers } : {}),
        ...(projectId ? {} : { main_chat: { model_choice: choice, processing_choice: tier } }),
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
    return { ...m, ...kept, el, state, asked, confirmed, armings, sheet, list, input, take, pin, strip, chooseTier, tierView, type, key,
        newWork: m.controls.model.newWork,
        modelPosts: posted('/api/owner/chat-model'), tierPosts: posted('/api/owner/chat-processing'),
        name: () => part('name').textContent, value: () => part('value').textContent,
        tierWord: () => (part('tier').hidden ? '' : part('tier').textContent),
        rows: () => modelSheetRows(list().innerHTML), outline: () => modelSheetOutline(list().innerHTML),
        open: async () => { if (!sheet()) await m.press(m.modelHead); await flush(); } };
}

test('a room follows Main until it chooses, and says so: the name word is Main, the value Main\'s model', () => {
    const r = room();
    assert.deepEqual([r.name(), r.value(), r.el.dataset.known, r.el.dataset.own], ['Main', 'gpt-6-astra', 'true', 'false']);
    assert.match(r.modelHead.getAttribute('aria-label'), /^Model for new work here: gpt-6-astra, Codex\. Set in Settings › Models\.$/);
    // Main moves: a follower reads it live.
    r.controls.syncState(r.state({ main_route: route(GLM) }));
    assert.deepEqual([r.name(), r.value()], ['Main', 'glm-5.3']);
    // The room's own model: its source names it, and only the mark is lit.
    r.controls.syncState({ main_route: route(ASTRA), projects: [{ id: 'p7', cognition: COGNITION, model_choice: { kind: 'model', model: GLM, use_local: false } }] });
    assert.deepEqual([r.name(), r.value(), r.el.dataset.own], ['OpenRouter', 'glm-5.3', 'true']);
    assert.match(r.modelHead.getAttribute('aria-label'), /^Model for new work here: glm-5\.3, OpenRouter\.$/);
    r.controls.destroy();
});

test('a source wears its program\'s mark before the catalog, and the capsule repaints it when the record names none', async () => {
    const unbound = { ...CATALOG, model_sources: [{ id: 'codex', label: 'Codex', credentialHarness: '' }] };
    const r = room({ storeOptions: { catalog: unbound } });
    const glyph = () => r.el.querySelector('.chat-capsule-glyph').innerHTML;
    // Not read yet (or timed out): the engine's own declaration names Codex, so its mark stands at once, never a letter.
    assert.ok(glyph().includes(HARNESS_MARKS.codex.path), 'Codex before any read');
    await r.store.read();  // a record that binds the source to no program
    await flush();
    // The same words (Codex) now stand beside the generic mark: the mark is what changed.
    assert.ok(glyph().includes(GENERIC_HARNESS_MARK.path), 'the record wins: the generic mark');
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
    assert.deepEqual(closed({ projectId: '', chatId: 1, tiers: { main: '', default: 'fast' } }), ['Fast', 'fast'], 'the Main chat follows Main like a room');
    assert.deepEqual(closed({ projectId: '', chatId: 1, tiers: { main: 'standard', default: 'fast' } }), ['', 'standard']);
    assert.deepEqual(closed({ projectId: '', chatId: 1, tiers: { main: 'fast', default: 'standard' }, tier: 'economy' }), ['Flex', 'economy'],
        'and its own tier comes first, as a room\'s does');
    assert.deepEqual(closed({ projectId: '', chatId: 1, tiers: { main: 'fast', default: 'standard' }, choice: { kind: 'model', model: GLM, use_local: false } }),
        ['', 'standard'], 'on a model of its own it takes the default tier, not Main\'s');
    // Read live, and said to a screen reader with the model.
    const r = room({ tiers: { main: 'fast', default: 'standard' } });
    assert.equal(r.modelHead.getAttribute('aria-label'), 'Model for new work here: gpt-6-astra, Fast, Codex. Set in Settings › Models.');
    r.controls.syncState(r.state({ processing_tiers: STANDARD }));
    assert.deepEqual([r.tierWord(), r.el.querySelector('.chat-capsule-tier').hidden], ['', true]);
    r.controls.syncState(r.state({ projects: [{ id: 'p7', cognition: COGNITION, model_choice: { kind: 'main' }, processing_choice: 'fast' }] }));
    assert.equal(r.tierWord(), 'Fast', 'the room\'s own tier from its project row');
    r.controls.destroy();
    const main = room({ projectId: '', chatId: 1, tiers: { main: 'economy', default: 'standard' } });
    assert.equal(main.modelHead.getAttribute('aria-label'), 'Model for new work here and in new rooms: gpt-6-astra, Flex, Codex. Set in Settings › Models.');
    main.controls.syncState(main.state({ main_chat: { model_choice: { kind: 'model', model: GLM, use_local: false }, processing_choice: '' } }));
    assert.equal(main.modelHead.getAttribute('aria-label'), 'Model for new work here and in new rooms: glm-5.3, OpenRouter.');
    assert.deepEqual([main.name(), main.value(), main.el.dataset.own], ['OpenRouter', 'glm-5.3', 'true']);
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
    assert.deepEqual(r.outline(), [FOLLOWS, '§ Pinned', `route|${GLM} +pin`, '§ Recent', `route|${ASTRA} * -pin`, 'sources'],
        'a model listed once: under Pinned');
    await r.pin(`route|${GLM}`);
    await flush();
    assert.deepEqual(r.pinWrites.at(-1), { model_pins: { unpin: [route(GLM)] } });
    assert.deepEqual(r.outline(), [FOLLOWS, '§ Recent', `route|${ASTRA} * -pin`, `route|${GLM} -pin`, 'sources']);
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
    assert.deepEqual([r.controls.hasPendingSave(), r.modelPosts(), r.asked, r.newWork.armed()], [true, [], [], null],
        'a tier moves nothing else and arms no new work');
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
    assert.deepEqual(await kimi({ main: 'economy', default: 'standard' }, { projectId: '', chatId: 1 }), ['OpenRouter · $0.60 in / $2.50 out', 0],
        'a model picked in the Main chat is its own too: the default tier, never Main\'s');
    assert.deepEqual(await kimi(STANDARD, { projectId: '', chatId: 1, tier: 'fast' }), ['OpenRouter · paid by key', 0], 'the chat\'s own tier first');
    // The row current now is Main's model, which this room follows: pressed it changes nothing, so it says the tier
    // the room's work runs on (Main's), never the one a pick of that model would take.
    const mainPriced = { ...CATALOG, items: CATALOG.items.map((item) => (item.value === GLM ? { ...item, price: { input: 0.6, output: 2.5 } } : item)) };
    const current = async (tiers, options = {}) => {
        const r = room({ tiers, main: GLM, storeOptions: { catalog: mainPriced }, ...options });
        await r.open();
        const said = r.rows().find((entry) => entry.key === `route|${GLM}`).facts;
        r.controls.destroy();
        return said;
    };
    assert.equal(await current({ main: 'fast', default: 'standard' }), 'OpenRouter · paid by key', 'Main\'s Fast is priced by its endpoint');
    assert.equal(await current({ main: 'standard', default: 'fast' }), 'OpenRouter · $0.60 in / $2.50 out', 'Main\'s Standard pays the catalog\'s');
    assert.equal(await current({ main: 'fast', default: 'standard' }, { projectId: '', chatId: 1 }), 'OpenRouter · paid by key', 'the Main chat\'s too');
    assert.equal(await current({ main: 'fast', default: 'standard' }, { tier: 'standard' }), 'OpenRouter · $0.60 in / $2.50 out', 'the room\'s own tier first');
});

test('nothing but Main to lead offers the way to Settings › Accounts and keys; a failed catalog read says so', async () => {
    const alone = room({ storeOptions: { catalog: { model_sources: CATALOG.model_sources, items: [] } } });
    await alone.open();
    assert.deepEqual(alone.outline(), [FOLLOWS, '§ Recent', `route|${ASTRA} * -pin`, 'note: Only Main\'s model can lead work here.', 'accounts']);
    await alone.take('accounts');
    assert.deepEqual([alone.settingsOpened, alone.sheet()], [['accounts'], null]);
    alone.controls.destroy();
    const failed = room({ storeOptions: { catalog: new Error('offline') } });
    await failed.open();
    assert.deepEqual(failed.outline(), [FOLLOWS, '§ Recent', `route|${ASTRA} * -pin`, 'note: The model catalog could not be read.', 'accounts']);
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
    assert.deepEqual([offered(home), supports(home).ultrafast], [['standard', 'fast', 'ultrafast'], 'listed'],
        'the Main chat that follows Main runs on Main\'s account like any follower');
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
            await r.chooseTier('standard');
            await r.controls.pendingSave();
            assert.deepEqual([offered(r), r.tierView().checked, r.tierView().reset, r.tierWord()], [['standard'], ['standard'], true, '']);
            await r.chooseTier('');
            await r.controls.pendingSave();
            assert.deepEqual([offered(r), r.tierView().checked, r.tierView().reset], [['standard'], [], false]);
            assert.deepEqual(r.tierPosts(), ['standard', ''].map((processing) => ({ processing, ...(projectId ? { project_id: projectId } : MAIN_CHAT) })));
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

test('a local Main takes what the local adapter can send: Fast is no new choice there, and an inherited one says it is not sent', async () => {
    const local = { provider_id: 'local', provider: 'Local', id: 'qwen3', name: 'qwen3', value: 'qwen3', use_local: true, processing_transport: ['standard'] };
    const r = await opened(room({ projectId: '', chatId: 1, main: 'qwen3', tiers: { main: 'fast', default: 'standard' },
        storeOptions: { catalog: { ...CATALOG, items: [...CATALOG.items, local] } } }));
    r.controls.syncState(r.state({ main_route: route('qwen3', true) }));
    await flush();
    assert.deepEqual([offered(r), supports(r).fast, r.tierView().checked, r.tierView().ring], [['standard', 'fast'], 'unlisted', [], ['fast']],
        'the tier it takes from Main stays in view to be left; nothing else is offered');
    assert.match(means(r), /does not send Fast\./);
    r.controls.destroy();
});

/* --- the choice a room remembers, and the new work a pick arms --- */

test('the sheet has no row to follow Main: the room\'s model now leads the recents, marked, and every row is that route', async () => {
    const r = room({ tiers: { main: 'fast', default: 'standard' }, storeOptions: { recents: [route(GLM), route(ASTRA)], pins: [route(KIMI)] } });
    await r.open();
    assert.deepEqual(r.outline(), [FOLLOWS, '§ Pinned', `route|${KIMI} +pin`, '§ Recent', `route|${ASTRA} * -pin`, `route|${GLM} -pin`, 'sources']);
    assert.equal(r.rows().some((entry) => entry.key === 'main'), false);
    r.controls.destroy();
    const own = room({ choice: { kind: 'model', model: SOL, use_local: false }, storeOptions: { recents: [route(GLM), route(SOL)] } });
    await own.open();
    assert.deepEqual(own.outline(), ['§ Recent', `route|${SOL} * -pin`, `route|${GLM} -pin`, 'sources'], 'a room on its own model says no such line');
    own.controls.destroy();
    const main = room({ projectId: '', chatId: 1, storeOptions: { recents: [route(GLM)] } });
    await main.open();
    assert.equal(main.sheet().getAttribute('aria-label'), 'Model for new work here and in new rooms');
    assert.deepEqual(main.outline(), [FOLLOWS, '§ Recent', `route|${ASTRA} * -pin`, `route|${GLM} -pin`, 'sources'], 'the Main chat\'s sheet is a room\'s');
    main.controls.destroy();
});

test('before a pick the sheet names where its model is set, the pinned account, and what a pick does for new and running tasks', async () => {
    const pinned = room({ storeOptions: { settings: { OUROBOROS_MODEL_ACCOUNTS: { main: 'work' } } } });
    await pinned.open();
    await flush();
    assert.equal(pinned.outline()[0], `lead: This model is set in Settings › Models, on its pinned account work. ${OWN_ROUTE}`);
    pinned.controls.destroy();
    // An account not read is never said as pinned, nor as Auto.
    const unread = room({ storeOptions: { settings: new Error('offline') } });
    await unread.open();
    await flush();
    assert.equal(unread.outline()[0], FOLLOWS);
    unread.controls.destroy();
});

test('the row current now changes nothing: a room that follows Main keeps Main\'s binding, and nothing is armed', async () => {
    for (const projectId of ['p7', '']) {
        const r = room({ projectId, storeOptions: { recents: [route(GLM)] } });
        await r.open();
        await r.take(`route|${ASTRA}`);  // Main's own model, the row marked current
        assert.deepEqual([r.modelPosts(), r.sheet(), r.name(), r.el.dataset.own, r.newWork.armed()], [[], null, 'Main', 'false', null],
            'never a covert move from following Main to a route of the same name');
        r.controls.destroy();
    }
});

test('a pick saves at once for its room, the Main chat\'s included, asks nothing and never writes Main', async () => {
    for (const projectId of ['p7', '']) {
        const r = room({ projectId, storeOptions: { recents: [route(GLM)] } });
        r.modelHead.isConnected = true;  // on the page: the focus can come back to it
        await r.open();
        let answer;
        r.modelAnswer.value = new Promise((resolve) => { answer = resolve; });
        await r.take(`route|${GLM}`);
        // Each save names its one scope: a room by its id, the Main chat by `scope` (a body with neither is an older
        // page's write of Main the role, which the host refuses).
        assert.deepEqual(r.modelPosts(), [{ kind: 'model', model: GLM, use_local: false, ...(projectId ? { project_id: projectId } : MAIN_CHAT) }]);
        assert.deepEqual([r.controls.hasPendingSave(), r.el.dataset.disabled, r.name(), r.asked], [true, 'true', 'Main', []],
            'unsaved, it is not shown as saved; no question stands before a pick');
        assert.equal(r.doc.activeElement, r.modelHead, 'the sheet closed with the pick, the focus is back on the capsule');
        r.controls.syncState(r.state());
        answer({ ok: true, project_id: projectId, model_choice: { kind: 'model', model: GLM, use_local: false },
            previous_model_choice: { kind: 'main' }, choice_token: tokenOf(projectId, GLM) });
        assert.equal(await r.controls.pendingSave(), true);
        assert.deepEqual([r.name(), r.value(), r.el.dataset.disabled, r.refreshes.at(-1)], ['OpenRouter', 'glm-5.3', 'false', true]);
        assert.equal(r.reads.recents, 2, 'the server wrote the act into the recents: they are read again');
        r.modelAnswer.value = null;
        await r.open();
        await r.take(`route|${GLM}`);
        assert.deepEqual([r.modelPosts().length, r.sheet()], [1, null], 'the chosen model again saves nothing');
        // It stays when Main moves: the choice is the room's until its owner changes it.
        const chosen = { model_choice: { kind: 'model', model: GLM, use_local: false }, processing_choice: '' };
        r.controls.syncState(r.state({ main_route: route(KIMI), ...(projectId ? { projects: [{ id: projectId, ...chosen }] } : { main_chat: chosen }) }));
        assert.deepEqual([r.name(), r.value()], ['OpenRouter', 'glm-5.3']);
        r.controls.destroy();
    }
});

test('a pick of another model arms this window\'s next message as new work, on the saved choice\'s token', async () => {
    const r = room({ storeOptions: { recents: [route(GLM), route(KIMI)] } });
    assert.equal(r.newWork.armed(), null, 'nothing picked: an ordinary message');
    await r.open();
    const before = r.armings.count;
    await r.take(`route|${GLM}`);
    await r.controls.pendingSave();
    assert.deepEqual(r.newWork.armed(), { token: tokenOf('p7', GLM), model: 'glm-5.3', changed: false });
    assert.ok(r.armings.count > before, 'the composer hears it: Send says so before the press');
    // Another pick moves the choice it names; the room it started from is still what it was before the first.
    await r.open();
    await r.take(`route|${KIMI}`);
    await r.controls.pendingSave();
    assert.deepEqual(r.newWork.armed(), { token: tokenOf('p7', KIMI), model: 'kimi-k3', changed: false });
    // The message goes out naming it: no later message names it too, until the host says how it took that one.
    const kimi = { token: tokenOf('p7', KIMI), model: 'kimi-k3', changed: false };
    r.newWork.sent('msg-0', tokenOf('p7', GLM));
    assert.deepEqual(r.newWork.armed(), kimi, 'a message that named an earlier choice is not this one\'s');
    r.newWork.sent('msg-1', kimi.token);
    assert.equal(r.newWork.armed(), null);
    r.newWork.settled('another', true);
    r.newWork.settled('another', false);
    r.newWork.refused('another');
    r.newWork.released('another');
    assert.equal(r.newWork.armed(), null, 'only that message\'s own word settles it');
    r.newWork.settled('msg-1', false);  // a Reply the host gave to its work: the choice is not spent
    assert.deepEqual(r.newWork.armed(), kimi);
    r.newWork.sent('msg-2', kimi.token);
    r.newWork.settled('msg-2', true);  // taken as new work
    assert.equal(r.newWork.armed(), null);
    r.newWork.settled('msg-2', false);
    assert.equal(r.newWork.armed(), null, 'a spent choice does not come back');
    r.controls.destroy();
});

/* A room on Sol whose window picked `first` and sent `msg-a` naming it; the host has not said how it took it. */
async function inFlight(first = GLM) {
    const r = room({ choice: { kind: 'model', model: SOL, use_local: false }, storeOptions: { recents: [route(GLM), route(KIMI), route(SOL)] } });
    const pick = async (model) => { await r.open(); await r.take(`route|${model}`); await r.controls.pendingSave(); };
    const armed = (model) => ({ token: tokenOf('p7', model), model: model.split(/[/=]/).pop(), changed: false });
    await pick(first);
    r.newWork.sent('msg-a', tokenOf('p7', first));
    assert.equal(r.newWork.armed(), null);
    return { r, pick, armed };
}

test('a pick made while an earlier message waits for the host\'s word is its own choice: that word never takes it', async () => {
    // Taken as new work, the room stands on the first choice: the later pick still starts its own, and a return to
    // the first is an ordinary message of the room as it is now.
    const spent = await inFlight();
    await spent.pick(KIMI);
    assert.deepEqual(spent.r.newWork.armed(), spent.armed(KIMI));
    spent.r.newWork.settled('msg-a', true);
    assert.deepEqual(spent.r.newWork.armed(), spent.armed(KIMI), 'the first message\'s success spends its own choice alone');
    await spent.pick(GLM);
    assert.equal(spent.r.newWork.armed(), null, 'back on the model the new work took');
    spent.r.controls.destroy();
    // Given to a work as its Reply, nothing started on the first choice: the room is where it was before both picks.
    const kept = await inFlight();
    await kept.pick(KIMI);
    kept.r.newWork.settled('msg-a', false);
    assert.deepEqual(kept.r.newWork.armed(), kept.armed(KIMI));
    await kept.pick(GLM);
    assert.deepEqual(kept.r.newWork.armed(), kept.armed(GLM), 'the first choice was never used: a pick of it is a new task again');
    await kept.pick(SOL);
    assert.equal(kept.r.newWork.armed(), null, 'and the model the room had is an ordinary message');
    kept.r.controls.destroy();
    // The later pick returned to what the room had: once the first message proves unspent, nothing is armed.
    const back = await inFlight();
    await back.pick(SOL);
    assert.deepEqual(back.r.newWork.armed(), back.armed(SOL), 'while the first is read as taken, Sol is another model');
    back.r.newWork.settled('msg-a', false);
    assert.equal(back.r.newWork.armed(), null);
    back.r.controls.destroy();
    // Refused because the choice changed: nothing started on it and it does not wait; the later pick stands.
    const refused = await inFlight();
    await refused.pick(KIMI);
    refused.r.newWork.refused('msg-a');
    assert.deepEqual(refused.r.newWork.armed(), refused.armed(KIMI));
    await refused.pick(SOL);
    assert.equal(refused.r.newWork.armed(), null, 'the room never left Sol');
    refused.r.controls.destroy();
    const alone = await inFlight();
    alone.r.newWork.refused('msg-a');
    assert.equal(alone.r.newWork.armed(), null, 'a refused choice does not come back');
    alone.r.newWork.settled('msg-a', false);
    assert.equal(alone.r.newWork.armed(), null);
    alone.r.controls.destroy();
});

test('a copy the owner gave up frees its choice for the next message; a row that proves it taken still settles it', async () => {
    const given = await inFlight();
    given.r.newWork.released('msg-a');  // Discard, or its words went back to the composer
    assert.deepEqual(given.r.newWork.armed(), given.armed(GLM), 'the choice waits for a message again');
    given.r.newWork.released('msg-a');
    given.r.newWork.settled('msg-a', false);  // the row says a work took it as its Reply: it already waits
    assert.deepEqual(given.r.newWork.armed(), given.armed(GLM));
    given.r.controls.destroy();
    // History later shows the given-up message was taken as new work after all: its choice is spent.
    const taken = await inFlight();
    taken.r.newWork.released('msg-a');
    taken.r.newWork.settled('msg-a', true);
    assert.equal(taken.r.newWork.armed(), null);
    taken.r.newWork.settled('msg-a', false);
    assert.equal(taken.r.newWork.armed(), null, 'one word per message');
    taken.r.controls.destroy();
    // A pick made after the give-up is its own choice there too; the late word only moves what it started from.
    const later = await inFlight();
    later.r.newWork.released('msg-a');
    await later.pick(KIMI);
    later.r.newWork.settled('msg-a', true);
    assert.deepEqual(later.r.newWork.armed(), later.armed(KIMI));
    await later.pick(GLM);
    assert.equal(later.r.newWork.armed(), null, 'the room stands on the model the taken message started');
    later.r.controls.destroy();
    // The next message names the freed choice again: the first message's late word is not about it.
    const again = await inFlight();
    again.r.newWork.released('msg-a');
    again.r.newWork.sent('msg-b', tokenOf('p7', GLM));
    again.r.newWork.settled('msg-a', false);
    assert.equal(again.r.newWork.armed(), null, 'msg-b still waits for its own word');
    again.r.newWork.settled('msg-b', false);
    assert.deepEqual(again.r.newWork.armed(), again.armed(GLM));
    again.r.controls.destroy();
    // A pick made while the given-up message flew was read from its choice; given up, it is read from the room as it was.
    const during = await inFlight();
    await during.pick(SOL);
    during.r.newWork.released('msg-a');
    assert.equal(during.r.newWork.armed(), null, 'Sol is what the room had');
    during.r.controls.destroy();
});

test('back on the model the room had, the next message is an ordinary one again', async () => {
    const r = room({ choice: { kind: 'model', model: SOL, use_local: false }, storeOptions: { recents: [route(GLM), route(SOL)] } });
    await r.open();
    await r.take(`route|${GLM}`);
    await r.controls.pendingSave();
    assert.equal(r.newWork.armed().token, tokenOf('p7', GLM));
    await r.open();
    await r.take(`route|${SOL}`);
    await r.controls.pendingSave();
    assert.deepEqual([r.modelPosts().length, r.value(), r.newWork.armed()], [2, 'gpt-6.1-sol', null], 'the room is as it was');
    r.controls.destroy();
    // A room that followed Main has no row to return to: a model of the same name is that route, and stays armed.
    const follower = room({ storeOptions: { recents: [route(GLM), route(ASTRA)] } });
    await follower.open();
    await follower.take(`route|${GLM}`);
    await follower.controls.pendingSave();
    await follower.open();
    await follower.take(`route|${ASTRA}`);
    await follower.controls.pendingSave();
    assert.deepEqual(follower.modelPosts().at(-1), { kind: 'model', model: ASTRA, use_local: false, project_id: 'p7' }, 'never a follow of Main');
    assert.deepEqual([follower.name(), follower.el.dataset.own, follower.newWork.armed().token], ['Codex', 'true', tokenOf('p7', ASTRA)]);
    follower.controls.destroy();
});

test('a refused pick arms nothing, and an answer without a token is no word to send on', async () => {
    const r = room({ storeOptions: { recents: [route(GLM)] } });
    await r.open();
    r.modelAnswer.value = Promise.reject(new Error('Not a model route'));
    const pressed = r.take(`route|${GLM}`);
    const send = r.controls.model.pendingSave();  // Send asks while the save is out
    await pressed;
    assert.equal(await send, false);
    assert.deepEqual([r.newWork.armed(), r.name()], [null, 'Main']);
    r.modelAnswer.value = { ok: true, project_id: 'p7', model_choice: { kind: 'model', model: GLM, use_local: false } };
    await r.open();
    await r.take(`route|${GLM}`);
    await r.controls.pendingSave();
    assert.deepEqual([r.value(), r.newWork.armed()], ['glm-5.3', null], 'saved, and the next message is an ordinary one');
    r.controls.destroy();
});

test('another window\'s pick voids the one made here: it is said as changed, never sent on, and dropped once told', async () => {
    const r = room({ storeOptions: { recents: [route(GLM)] } });
    await r.open();
    await r.take(`route|${GLM}`);
    await r.controls.pendingSave();
    const before = r.armings.count;
    r.controls.syncState(r.state({ projects: [{ id: 'p7', model_choice: { kind: 'model', model: KIMI, use_local: false }, processing_choice: '' }] }));
    assert.deepEqual([r.value(), r.newWork.armed()], ['kimi-k3', { token: tokenOf('p7', GLM), model: 'glm-5.3', changed: true }],
        'the capsule shows what is saved now; the pick made here still names its own choice');
    assert.ok(r.armings.count > before);
    r.newWork.drop();
    assert.equal(r.newWork.armed(), null);
    // A pick made while the overtaken one still stands unsaid starts from the room as the other window left it: back
    // on the model this window began from is a pick of another model there, never a return to an ordinary message.
    const again = room({ storeOptions: { recents: [route(GLM), route(ASTRA)] }, choice: { kind: 'model', model: ASTRA, use_local: false } });
    await again.open();
    await again.take(`route|${GLM}`);
    await again.controls.pendingSave();
    again.controls.syncState(again.state({ projects: [{ id: 'p7', model_choice: { kind: 'model', model: KIMI, use_local: false }, processing_choice: '' }] }));
    assert.equal(again.newWork.armed().changed, true);
    await again.open();
    await again.take(`route|${ASTRA}`);
    await again.controls.pendingSave();
    assert.deepEqual(again.newWork.armed(), { token: tokenOf('p7', ASTRA), model: 'gpt-6-astra', changed: false });
    again.controls.destroy();
    // Its own tier moving there is no change of the choice.
    const tiered = room({ storeOptions: { recents: [route(GLM)] } });
    await tiered.open();
    await tiered.take(`route|${GLM}`);
    await tiered.controls.pendingSave();
    tiered.controls.syncState(tiered.state({ projects: [{ id: 'p7', model_choice: { kind: 'model', model: GLM, use_local: false }, processing_choice: 'fast' }] }));
    assert.equal(tiered.newWork.armed().changed, false);
    for (const made of [r, tiered]) made.controls.destroy();
});

test('two saved choices are one when both follow Main or both name the same route', () => {
    const glm = { kind: 'model', model: GLM, use_local: false };
    assert.deepEqual([sameChoice({ kind: 'main' }, { kind: 'main' }), sameChoice(glm, { ...glm }), sameChoice(glm, { kind: 'main' }),
        sameChoice(glm, { ...glm, use_local: true }), sameChoice(glm, { kind: 'model', model: KIMI, use_local: false }), sameChoice(null, glm)],
    [true, true, false, false, false, false]);
});

test('the Main chat\'s tier is its own like a room\'s; Ultrafast asks first there, and a cancel writes nothing', async () => {
    const r = await opened(room({ projectId: '', chatId: 1, tiers: { main: 'fast', default: 'standard' }, storeOptions: { catalog: TWO_ACCOUNTS } }));
    assert.deepEqual([r.tierView().checked, r.tierView().ring], [[], ['fast']], 'no tier of its own: Main\'s, as a follower');
    assert.equal(r.tierView().scope, 'Default · Fast as Main · for new work here and its new subagents without a tier of their own; a new room copies it, running work keeps its own');
    await r.chooseTier('standard');
    await r.controls.pendingSave();
    assert.deepEqual([r.tierPosts(), r.asked, r.modelPosts(), r.newWork.armed()], [[{ processing: 'standard', ...MAIN_CHAT }], [], [], null],
        'its own setting, named as the Main chat\'s; no question, no model, no new work');
    assert.deepEqual([r.tierView().checked, r.store.tiers().main, Boolean(r.sheet())], [['standard'], 'fast', true],
        'Main the role\'s own tier is not this one');
    // Ultrafast: asked first, for the route and account the sheet shows.
    r.confirmed.value = false;
    await r.chooseTier('ultrafast');
    await r.controls.pendingSave();
    assert.deepEqual([r.asked.length, r.tierPosts().length, r.tierWord()], [1, 1, ''], 'a cancel writes nothing');
    assert.deepEqual([r.asked[0].model, r.asked[0].source, r.asked[0].follows, r.asked[0].account],
        ['gpt-6-astra', 'Codex · Account: Auto', 'auto', ''], 'Auto is said as Auto, its source\'s pool: no one account is named for it');
    r.confirmed.value = true;
    await r.open();
    await r.chooseTier('ultrafast');
    await r.controls.pendingSave();
    // The write names what was asked about, by name: the model, its locality and the profile ('' on Auto). The
    // host saves the tier for that route and answers on it alone.
    assert.deepEqual([r.asked.length, r.tierPosts().at(-1), r.tierWord()],
        [2, { processing: 'ultrafast', ...MAIN_CHAT, confirmed: { model: ASTRA, use_local: false, account: '' } }, 'Ultrafast']);
    await r.open();  // the question closed the sheet
    // New work carries the confirmation; the sheet describes its destination, not later picks in the room.
    assert.equal(r.tierView().scope, 'Ultrafast · for new work here, on calls to this model through the confirmed destination; no new room copies it, running work keeps its own');
    r.controls.destroy();
    // While the chat follows Main, the route asked about is Main's own, its pinned profile included: by its name
    // alone. The catalog's reading of who is signed in to it (TWO_ACCOUNTS names a store and fingerprints) is
    // neither asked about nor sent, and the profile that does not list Ultrafast may still be asked for.
    const pinned = await opened(room({ projectId: '', chatId: 1, storeOptions: { catalog: TWO_ACCOUNTS, settings: { OUROBOROS_MODEL_ACCOUNTS: { main: 'work' } } } }));
    await pinned.chooseTier('ultrafast');
    await pinned.controls.pendingSave();
    assert.deepEqual([pinned.asked.map((one) => [one.source, one.follows, one.account]), pinned.tierPosts()],
        [[['Codex · Pinned account work', 'profile', 'work']],
            [{ processing: 'ultrafast', ...MAIN_CHAT, confirmed: { model: ASTRA, use_local: false, account: 'work' } }]]);
    // A pinned profile alone carries the tier from the first call, whatever is chosen here later: its own words.
    await pinned.open();
    assert.equal(pinned.tierView().scope, 'Ultrafast · for new work here, on its own calls to this model and profile from the first one; no new room copies it, running work keeps its own');
    pinned.controls.destroy();
    // An account not read yet was not asked about: no question, no write.
    const unread = await opened(room({ projectId: '', chatId: 1, storeOptions: { catalog: TWO_ACCOUNTS, settings: new Error('offline') } }));
    await unread.chooseTier('ultrafast');
    await unread.controls.pendingSave();
    assert.deepEqual([unread.asked, unread.tierPosts()], [[], []]);
    assert.match(unread.toasts.at(-1)?.[0] || '', /account or this connection is not read yet/);
    unread.controls.destroy();
    // An API route names its connection's revision as the settings read it: the host compares it under its lock, so
    // the same model on a connection saved since then is not what was asked about.
    const OFFERS = { catalog: { ...CATALOG, api_sources: [{ id: 'openrouter', processing_transport: ['standard', 'economy', 'fast', 'ultrafast'] }] },
        processing: (value) => ({ value, processing: { modes: ['standard', 'ultrafast'], complete: true, endpoints: [] } }) };
    const keyed = await opened(room({ projectId: '', chatId: 1, main: GLM, storeOptions: { ...OFFERS,
        connections: [{ id: 'openrouter', configured: true, revision: 'openrouter#0123456789abcdef01234567' }] } }));
    await keyed.chooseTier('ultrafast');
    await keyed.controls.pendingSave();
    assert.deepEqual(keyed.tierPosts(), [{ processing: 'ultrafast', ...MAIN_CHAT, confirmed: { model: GLM, use_local: false, account: '',
        connection: 'openrouter#0123456789abcdef01234567' } }]);
    assert.deepEqual([keyed.asked[0].source, keyed.asked[0].follows], ['OpenRouter', 'api'], 'a key\'s connection has no name to follow');
    await keyed.open();
    assert.match(keyed.tierView().scope, /^Ultrafast · for new work here, on calls to this model through the confirmed destination; /);
    keyed.controls.destroy();
    // A connection the settings gave no revision for cannot be bound: nothing is asked, nothing written.
    const unbound = await opened(room({ projectId: '', chatId: 1, main: GLM, storeOptions: { ...OFFERS, connections: [{ id: 'openrouter', configured: true }] } }));
    await unbound.chooseTier('ultrafast');
    await unbound.controls.pendingSave();
    assert.deepEqual([unbound.asked, unbound.tierPosts()], [[], []]);
    assert.match(unbound.toasts.at(-1)?.[0] || '', /account or this connection is not read yet/);
    unbound.controls.destroy();
    // A room takes it without a question, as before.
    const own = await opened(room({ storeOptions: { catalog: TWO_ACCOUNTS } }));
    await own.chooseTier('ultrafast');
    await own.controls.pendingSave();
    assert.deepEqual([own.asked, own.tierPosts()], [[], [{ processing: 'ultrafast', project_id: 'p7' }]]);
    own.controls.destroy();
});

test('a saved Ultrafast that names another model or account reads Default in the Main chat, and the strip says why', async () => {
    const r = await opened(room({ projectId: '', chatId: 1, tiers: { main: 'fast', default: 'standard' }, storeOptions: { catalog: TWO_ACCOUNTS } }));
    // The host withholds it (new work now runs on another model or profile than it was saved for): Default.
    r.controls.syncState(r.state({ main_chat: { model_choice: { kind: 'main' }, processing_choice: '', processing_withheld: 'ultrafast' } }));
    assert.deepEqual([r.tierView().checked, r.tierView().ring], [[], ['fast']], 'Default, on Main\'s tier: never shown as Ultrafast');
    assert.notEqual(r.tierWord(), 'Ultrafast');
    assert.equal(r.tierView().scope, 'Default · Fast as Main · The saved Ultrafast no longer matches the destination of new work here. New work takes Default until you confirm again.');
    // In force again (the route it names is the one new work runs on), it is Ultrafast with its own words.
    r.controls.syncState(r.state({ main_chat: { model_choice: { kind: 'main' }, processing_choice: 'ultrafast' } }));
    assert.equal(r.tierWord(), 'Ultrafast');
    assert.match(r.tierView().scope, /^Ultrafast · for new work here, on calls to this model through the confirmed destination; /);
    // A room has no such state: its row never reads the Main chat's field.
    const own = await opened(room({ storeOptions: { catalog: TWO_ACCOUNTS } }));
    own.controls.syncState(own.state({ main_chat: { model_choice: { kind: 'main' }, processing_choice: '', processing_withheld: 'ultrafast' } }));
    assert.doesNotMatch(own.tierView().scope, /Ultrafast is saved here/);
    for (const made of [r, own]) made.controls.destroy();
});

test('a window newer than its server sends the Main chat\'s choice nowhere: an older server would change Main the role', async () => {
    // An update not finished, or rolled back: the page is this version, the server an older one. Its state body
    // names Main's route but no `main_chat`, and it reads a write without a room as Main the role's own.
    const OLDER = { main_route: route(ASTRA), processing_tiers: STANDARD, projects: [] };
    const r = room({ projectId: '', chatId: 1, storeOptions: { recents: [route(GLM)], catalog: TWO_ACCOUNTS } });
    r.controls.syncState(OLDER);
    await r.open();
    await r.take(`route|${GLM}`);
    await r.controls.pendingSave();
    assert.deepEqual([r.modelPosts(), r.newWork.armed(), r.toasts.at(-1)?.[0]], [[], null, SERVER_OLDER],
        'nothing was sent, nothing is armed, and the owner is told what to do');
    await r.open();
    await r.chooseTier('fast');  // a press on the strip
    assert.equal(await r.controls.model.chooseTier('fast'), false, 'the tier path answers that nothing was saved');
    assert.deepEqual([r.tierPosts(), r.toasts.at(-1)?.[0]], [[], SERVER_OLDER]);
    // The server of this version answers: the same presses save, each naming the Main chat.
    r.controls.syncState(r.state());
    await r.take(`route|${GLM}`);
    await r.controls.pendingSave();
    assert.deepEqual(r.modelPosts(), [{ kind: 'model', model: GLM, use_local: false, ...MAIN_CHAT }]);
    assert.equal(r.newWork.armed().token, tokenOf('', GLM));
    // Rolled back under an armed pick: an older server takes no new task, so the next message is an ordinary one.
    r.controls.syncState(OLDER);
    assert.equal(r.newWork.armed(), null);
    r.controls.destroy();
    // Before any state body is read the server is not known: nothing is sent either, with its own words.
    const early = mountPair({ projectId: '', chatId: 1, subagents: true, model: { store: storeOf({ recents: [route(GLM)] }).store } });
    assert.equal(early.controls.model.pick(route(GLM)), undefined);
    assert.equal(await early.controls.model.chooseTier('fast'), false);
    assert.deepEqual([early.posts, early.toasts.map(([text]) => text)], [[], [CHAT_NOT_READ, CHAT_NOT_READ]]);
    early.controls.destroy();
    // A room's writes name their project, which every server version reads the same way: they are never held back.
    const own = room({ storeOptions: { recents: [route(GLM)] } });
    own.controls.syncState({ main_route: route(ASTRA), processing_tiers: STANDARD, projects: [{ id: 'p7', chat_id: 7, cognition: COGNITION, model_choice: { kind: 'main' } }] });
    await own.open();
    await own.take(`route|${GLM}`);
    await own.controls.pendingSave();
    assert.deepEqual(own.modelPosts(), [{ kind: 'model', model: GLM, use_local: false, project_id: 'p7' }]);
    own.controls.destroy();
});

test('the question before the Main chat\'s Ultrafast says what it follows, what it reaches and what the route says of it', async () => {
    const asked = [];
    const doc = { createElement: () => ({ dataset: {}, className: '', textContent: '' }) };
    const dialog = async (question) => {
        const lines = [];
        question.content.mount({ append: (...nodes) => lines.push(...nodes.map((node) => node.textContent)) });
        asked.push({ title: question.title, body: question.body, confirm: question.confirmLabel, lines });
        return true;
    };
    const about = { model: 'gpt-6-astra', explain: 'Requests the fastest processing.', dialog, doc };
    // A pinned profile: the tier follows its name, and the owner reads what a new sign-in there does before agreeing.
    assert.equal(await confirmChatUltrafast({ ...about, source: 'Codex · Pinned account work', follows: 'profile', account: 'work' }), true);
    const signIn = 'Ultrafast follows the profile work by its name, not the subscription signed in to it. After a new sign-in to that profile, later calls, in work already started too, can spend the new subscription’s quota or money.';
    assert.deepEqual(asked.pop(), { title: 'Ultrafast for new work', body: CHAT_ULTRAFAST_PROFILE_SCOPE, confirm: 'Use Ultrafast',
        lines: ['gpt-6-astra · Codex · Pinned account work', signIn, 'Requests the fastest processing.', 'What Ultrafast costs on this route is not known here.'] });
    // Its reach is said in profiles: this model on this profile alone, from the first call, never a fallback, another
    // model or profile given to the work, or its subagents; and a later choice here is new work's alone. "Another
    // account" is never said: a new sign-in to the same profile is another account, and it keeps the tier.
    assert.match(CHAT_ULTRAFAST_PROFILE_SCOPE, /requests Ultrafast from its first call, only on its own calls to this model on this profile\./);
    assert.match(CHAT_ULTRAFAST_PROFILE_SCOPE, /A call that goes anywhere else \(another model, another profile, a fallback\) and the work’s subagents take Default\./);
    assert.match(CHAT_ULTRAFAST_PROFILE_SCOPE, /Another model or tier chosen here later changes only work started after it: work already started keeps Ultrafast on this model and profile until you change its tier on its card\./);
    assert.match(CHAT_ULTRAFAST_PROFILE_SCOPE, /a new room does not copy Ultrafast/);
    assert.doesNotMatch(CHAT_ULTRAFAST_PROFILE_SCOPE, /account/);
    // Auto names no individual account: it is the source's pool.
    await confirmChatUltrafast({ ...about, source: 'Codex · Account: Auto', follows: 'auto' });
    assert.deepEqual(asked.pop(), { title: 'Ultrafast for new work', body: CHAT_ULTRAFAST_SCOPE, confirm: 'Use Ultrafast',
        lines: ['gpt-6-astra · Codex · Account: Auto', CHAT_ULTRAFAST_FOLLOWS.auto, 'Requests the fastest processing.', 'What Ultrafast costs on this route is not known here.'] });
    assert.match(CHAT_ULTRAFAST_FOLLOWS.auto, /^Auto is this source’s pool of accounts, not one account: /);
    assert.deepEqual(Object.keys(CHAT_ULTRAFAST_FOLLOWS), ['profile', 'auto', 'api']);
    // A key's connection, with the price its endpoints list said in the route's own words: no "not known" line, no
    // unknown price, but an explicit warning about changing the connection.
    await confirmChatUltrafast({ ...about, model: 'glm-5.3', source: 'OpenRouter', follows: 'api',
        explain: 'Requests the fastest processing. Listed at $4 in / $20 out per 1M tokens.', priceSaid: true });
    assert.deepEqual(asked.pop(), { title: 'Ultrafast for new work', body: CHAT_ULTRAFAST_SCOPE, confirm: 'Use Ultrafast',
        lines: ['glm-5.3 · OpenRouter', CHAT_ULTRAFAST_FOLLOWS.api, 'Requests the fastest processing. Listed at $4 in / $20 out per 1M tokens.'] });
    assert.match(CHAT_ULTRAFAST_SCOPE, /on its own calls to this destination\./);
    assert.match(CHAT_ULTRAFAST_SCOPE, /A call that goes elsewhere and the work’s subagents take Default\./);
    assert.match(CHAT_ULTRAFAST_SCOPE, /Later picks in this chat change only future work\./);
    assert.match(CHAT_ULTRAFAST_SCOPE, /a new room does not copy Ultrafast/);
    assert.equal(await confirmChatUltrafast({ dialog: async () => false, doc }), false);
});
