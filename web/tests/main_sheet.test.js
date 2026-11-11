// The Main sheet's lists (composer_model.js; DESIGN "Composer owner controls"): before Main changes the sheet says how
// many running works follow it and lists what is not on Main (works on other models, rooms with their own), each
// with its own tick, none ticked beforehand. They are computed from the state the page has (CLIENT_CONTRACT §5
// `works[].lead`, §9 `projects[].model_choice`); every tick is one act after the Main write, with its own result.
import assert from 'node:assert/strict';
import test from 'node:test';
import { APPLIED_WORDS } from '../modules/card_model.js';
import { confirmMainChange, mainSheetLists, showMainMoves } from '../modules/composer_model.js';
import { createModelCatalogStore } from '../modules/model_catalog_store.js';
import { describeRoute } from '../modules/chat_model_routes.js';
import { clock } from '../modules/work_status.js';
import { ElementStub } from './chat_dom_fixture.js';
import { mountPair } from './composer_controls_fixture.js';

const ASTRA = 'claudexor::codex=gpt-6-astra';
const OPUS = 'anthropic::claude-opus-5.5';
const GLM = 'z-ai/glm-5.3';
const KIMI = 'moonshotai/kimi-k3';
const SINCE = 1794362040;  // an instant the fact carries; said in the viewer's clock
const route = (model) => ({ model, use_local: false });
const lead = (model, basis, extra = {}) => ({ ...route(model), basis, ...extra });
const work = (id, name, fact, extra = {}) => ({ work_id: `w-${id}`, room: 7, name, kind: 'task', project_id: 'p7',
    episode: { task_id: id, live: true }, activity: 'working', live: true, confirmed: true,
    controls: [{ action: 'model', target: id }], lead: fact, ...extra });
const PROJECTS = [
    { id: 'p7', name: 'Research', chat_id: 7, model_choice: { kind: 'model', model: KIMI, use_local: false } },
    { id: 'p8', name: 'Site', chat_id: 8, model_choice: { kind: 'main' } },
    { id: 'p9', name: 'Old', chat_id: 9, model_choice: { kind: 'model', model: GLM, use_local: false }, closed: { at: 'x', by: 'owner', note: '' } },
];
const WORKS = [
    work('t1', 'Fix the parser', lead(ASTRA, 'main')),
    work('t2', 'Draft the letter', lead(ASTRA, 'main'), { activity: 'queued', live: false }),
    work('t3', 'Compare prices', lead(GLM, 'owner')),
    work('t4', 'Read the logs', lead(OPUS, 'fallback', { role: 'fallback:0', primary: lead(ASTRA, 'main'), since: SINCE }), { room: 1, project_id: '' }),
    work('t5', 'Sort the mail', lead(KIMI, 'room'), { controls: [] }),
    work('t6', 'Unconfirmed', lead(GLM, 'owner'), { confirmed: false }),
    work('t7', 'No fact', undefined),
];
const census = (extra = {}) => ({ works: WORKS, complete: true, projects: PROJECTS, ...extra });
const flush = async () => { for (let i = 0; i < 6; i += 1) await new Promise((resolve) => setImmediate(resolve)); };
const NEXT = { ...describeRoute(null, route(GLM)) };
const PREVIOUS = { ...describeRoute(null, route(ASTRA)) };

/* The sheet as the shared dialog mounts it; `act(content)` is what the owner does before confirming. */
async function sheet({ lists = census(), act = () => {}, confirm = true } = {}) {
    const doc = { createElement: (tag) => new ElementStub(tag, doc), byId: new Map() };
    const content = doc.createElement('section');
    // The shared dialog's seam (confirm_dialog.test.js): what the sheet asks the confirming button to say.
    const labels = [];
    const seam = { setConfirmLabel: (words, params = null) => labels.push(params ? words.replace('{n}', params.n) : words) };
    const dialog = async (options) => {
        options.content.mount(content, seam);
        await act({ content, labels, fire: () => (content.listeners.get('change') || []).forEach((fn) => fn({})) });
        return confirm;
    };
    const answer = await confirmMainChange({ next: NEXT, previous: PREVIOUS, census: lists, dialog, doc });
    const count = content.children.find((node) => node.dataset.mainFollowers !== undefined);
    const rows = content.querySelectorAll('.chat-model-off');
    const words = (row) => [row.children.find((node) => node.classList.contains('chat-model-off-name')).textContent,
        row.querySelector('.chat-model-off-meta').children.map((node) => node.textContent).join('')];
    const boxes = content.querySelectorAll('.ui-checkbox');
    return { answer, content, labels, count, rows, words, boxes };
}

test('the count is a number when the census is complete and unknown when it is not, never zero by default', () => {
    assert.equal(mainSheetLists(census()).followers, 1, 'the running follower; a queued one starts on Main anyway');
    assert.equal(mainSheetLists(census({ complete: false })).followers, null);
    assert.equal(mainSheetLists(census({ works: null })).followers, null, 'a body without the works');
    assert.equal(mainSheetLists(census({ works: [] })).followers, 0);
    assert.equal(mainSheetLists().followers, null, 'no state at all');
});

test('the lists are what is not on Main: works on other models with why, rooms with their own; a fallback is never a follower', () => {
    const lists = mainSheetLists(census());
    assert.deepEqual(lists.works.map((entry) => [entry.id, entry.name, entry.model, entry.why, entry.room, entry.fallback]), [
        ['t3', 'Compare prices', 'OpenRouter · glm-5.3', 'you chose it', 'Research', false],
        ['t4', 'Read the logs', 'Anthropic · claude-opus-5.5', `fallback since ${clock(SINCE)}`, 'Main Chat', true],
        ['', 'Sort the mail', 'OpenRouter · kimi-k3', 'its room\'s model', 'Research', false],
    ], 'the fallback that left Main is listed as a fallback, not counted as one that will switch');
    assert.deepEqual(lists.rooms, [{ kind: 'room', id: 'p7', name: 'Research', model: 'OpenRouter · kimi-k3' }],
        'a room that follows Main and a closed room are not listed');
    // What an incomplete census still knows is shown; only the count is unknown.
    assert.equal(mainSheetLists(census({ complete: false })).works.length, 3);
});

test('the sheet says the count in words: unknown, none, one, several', async () => {
    const said = async (lists) => (await sheet({ lists })).count;
    const unknown = await said(census({ complete: false }));
    assert.equal(unknown.dataset.mainFollowers, 'unknown');
    assert.equal(unknown.textContent, 'How many running works follow Main is unknown right now: the list of running work is incomplete.');
    assert.equal((await said(census({ works: [] }))).textContent, 'No running work follows Main now.');
    assert.equal((await said(census())).textContent, '1 running work follows Main now.');
    const two = await said(census({ works: [...WORKS, work('t8', 'Another', lead(ASTRA, 'main'))] }));
    assert.deepEqual([two.textContent, two.dataset.mainFollowers], ['2 running works follow Main now.', '2']);
});

test('every movable row has its own tick and nothing is ticked beforehand; a work without the control has none', async () => {
    const s = await sheet();
    assert.deepEqual(s.rows.map(s.words), [
        ['Compare prices', 'OpenRouter · glm-5.3 · you chose it · Research'],
        ['Read the logs', `Anthropic · claude-opus-5.5 · fallback since ${clock(SINCE)} · Main Chat`],
        ['Sort the mail', 'OpenRouter · kimi-k3 · its room\'s model · Research'],
        ['Research', 'OpenRouter · kimi-k3'],
    ]);
    assert.deepEqual(s.boxes.map((box) => [box.dataset.move, box.checked]), [['work:t3', false], ['work:t4', false], ['room:p7', false]],
        'the fallback is not ticked for him either');
    assert.deepEqual(s.rows.map((row) => row.tagName), ['LABEL', 'LABEL', 'DIV', 'LABEL'], 'the server offered no control for one work: no tick');
    const amber = s.rows.map((row) => row.querySelector('.chat-model-off-meta').children.filter((node) => node.classList.contains('model-fallback')).length);
    assert.deepEqual(amber, [0, 1, 0, 0], 'only the fallback\'s word is amber');
    assert.deepEqual(s.answer, { moves: [] }, 'confirmed with no tick: Main alone changes');
    const texts = s.content.children.map((node) => node.textContent);
    assert.ok(texts.includes('Not on Main now · tick to move too'));
    assert.ok(texts.includes('Rooms that start new work on their own model · 1'));
});

test('the ticked rows are what the sheet answers with, the button says how many, and Cancel answers nothing', async () => {
    const s = await sheet({ act: ({ content, fire }) => {
        const boxes = content.querySelectorAll('.ui-checkbox');
        boxes[0].checked = true; fire();
        boxes[2].checked = true; fire();
        boxes[0].checked = false; fire();
        boxes[2].checked = false; fire();
        boxes[1].checked = true; boxes[2].checked = true; fire();
    } });
    assert.deepEqual(s.labels, ['Change Main and move 1 more', 'Change Main and move 2 more', 'Change Main and move 1 more', 'Change Main',
        'Change Main and move 2 more'], 'said through the dialog\'s own seam, on every tick');
    assert.deepEqual(s.answer.moves.map((move) => [move.kind, move.id]), [['work', 't4'], ['room', 'p7']]);
    const cancelled = await sheet({ confirm: false, act: ({ content }) => { content.querySelectorAll('.ui-checkbox')[0].checked = true; } });
    assert.equal(cancelled.answer, false);
});

test('a sheet with nothing off Main has no lists and no ticks', async () => {
    const s = await sheet({ lists: census({ works: [WORKS[0]], projects: [PROJECTS[1]] }) });
    assert.equal(s.rows.length, 0);
    assert.equal(s.content.children.length, 4, 'what Main becomes, its facts, the rule and the count');
});

/* --- the Main chat's capsule: Main is written first, then one act per tick --- */

function mainChat({ moves, moveWork, model = {} } = {}) {
    const said = [];
    const asked = [];
    const moved = [];
    const store = createModelCatalogStore({ readCatalog: async () => ({ items: [] }), readPreferences: async () => ({}),
        readSettings: async () => ({ _meta: { connections: [] } }) });
    const m = mountPair({ projectId: '', chatId: 1, subagents: true, model: { store,
        askMain: async (request) => { asked.push({ ...request, readsBefore: m.refreshes.length }); return { moves }; },
        moveWork: moveWork || (async (taskId) => { moved.push(taskId); return { ok: true, task_id: taskId, applied: taskId === 't2' ? 'at_start' : 'next_step' }; }),
        sayMoves: async ({ results }) => { said.push(results); }, ...model } });
    m.controls.syncState({ main_route: route(ASTRA), works: WORKS, works_complete: true, supervisor_ready: true, projects: PROJECTS,
        context_mode: 'max', subagents_mode: 'auto', effort_range: { min: 'low', recommended: 'high', max: 'max' } });
    const posts = () => m.posts.filter(([url]) => url === '/api/owner/chat-model').map(([, body]) => body);
    return { ...m, said, asked, moved, posts };
}

test('the sheet is given the page\'s census; each tick is its own act after the Main write, each with its own result', async () => {
    const c = mainChat({ moves: [{ kind: 'work', id: 't3', name: 'Compare prices' }, { kind: 'work', id: 't2', name: 'Draft the letter' },
        { kind: 'room', id: 'p7', name: 'Research' }] });
    await c.controls.model.pick(route(GLM));
    await flush();
    assert.deepEqual([c.asked[0].census.complete, c.asked[0].census.works.length, c.asked[0].census.projects.length], [true, WORKS.length, 3]);
    assert.deepEqual([c.asked[0].readsBefore, c.refreshes[0]], [1, true], 'the state is read once more before the sheet lists it');
    assert.deepEqual(c.posts(), [{ kind: 'model', model: GLM, use_local: false }, { project_id: 'p7', kind: 'main' }],
        'Main itself first, then the room: one write each, no batch');
    assert.deepEqual(c.moved, ['t3', 't2']);
    assert.deepEqual(c.said, [[
        { kind: 'work', id: 't3', name: 'Compare prices', ok: true, words: APPLIED_WORDS.next_step },
        { kind: 'work', id: 't2', name: 'Draft the letter', ok: true, words: APPLIED_WORDS.at_start },
        { kind: 'room', id: 'p7', name: 'Research', ok: true, words: 'New work here starts on Main.' },
    ]]);
    c.controls.destroy();
});

test('a refused tick says the server\'s words and stops none of the others; without ticks nothing more is said', async () => {
    const c = mainChat({ moves: [{ kind: 'work', id: 't3', name: 'Compare prices' }, { kind: 'room', id: 'p7', name: 'Research' }],
        moveWork: async () => { throw Object.assign(new Error('task not live'), { status: 404, body: { reason_code: 'task_not_live' } }); } });
    await c.controls.model.pick(route(GLM));
    await flush();
    assert.deepEqual(c.said[0].map((result) => [result.id, result.ok, result.words]), [
        ['t3', false, 'Not moved: This work has already ended.'], ['p7', true, 'New work here starts on Main.']],
    'a refusal in the card\'s own words for that code');
    c.controls.destroy();

    const quiet = mainChat({ moves: [] });
    await quiet.controls.model.pick(route(GLM));
    await flush();
    assert.deepEqual([quiet.said.length, quiet.posts().length], [0, 1]);
    quiet.controls.destroy();
});

test('when Main itself is not written no tick is acted on', async () => {
    const c = mainChat({ moves: [{ kind: 'work', id: 't3', name: 'Compare prices' }, { kind: 'room', id: 'p7', name: 'Research' }] });
    c.modelAnswer.value = { ok: false, project_id: '' };
    await c.controls.model.pick(route(GLM));
    await flush();
    assert.deepEqual([c.moved, c.said, c.posts().length], [[], [], 1]);
    assert.equal(c.toasts.at(-1)[1], 'error');
    c.controls.destroy();
});

test('the results sheet says each row\'s own result and how many were moved', async () => {
    const doc = { createElement: (tag) => new ElementStub(tag, doc), byId: new Map() };
    const calls = [];
    const results = [{ kind: 'work', id: 't3', name: 'Compare prices', ok: true, words: APPLIED_WORDS.next_step },
        { kind: 'room', id: 'p7', name: 'Research', ok: false, words: 'Not moved: The room is closed.' }];
    await showMainMoves({ results, doc, dialog: async (options) => { calls.push(options); } });
    assert.deepEqual([calls[0].title, calls[0].alert, calls[0].body], ['Main changed', true, '1 of 2 were moved to Main.']);
    const content = doc.createElement('section');
    calls[0].content.mount(content);
    const rows = content.querySelectorAll('.chat-model-off');
    assert.deepEqual(rows.map((row) => [row.dataset.result, row.children[0].textContent, row.children[1].children[0].textContent]), [
        ['moved', 'Compare prices', APPLIED_WORDS.next_step], ['refused', 'Research', 'Not moved: The room is closed.']]);
    assert.equal(content.querySelectorAll('.ui-checkbox').length, 0, 'a result is not a tick');
    await showMainMoves({ results: [results[0]], doc, dialog: async (options) => { calls.push(options); } });
    assert.equal(calls[1].body, '1 more was moved to Main.');
});
