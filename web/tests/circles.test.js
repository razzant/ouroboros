// Circles (circles.js; DESIGN "Circles"): the one home of the places. Its rows say each place's own name, service,
// state, readiness only when not simply on, the time of its last captured event and its optional description; a
// circle's words read in the voice of whoever wrote them; the places in no circle stay visible; every write goes
// through the circle and place clients with the revision the page read, and keeps the owner's draft on a refusal.
import assert from 'node:assert/strict';
import test from 'node:test';

import {
    circleFoldHtml, circleWordsHtml, groupPlaces, initCircles, lastEventText, migrationProblemLines, ownModelsText, placeCountText,
    placeRowHtml, placeRowsHtml, unassignedFoldHtml,
} from '../modules/circles.js';
import { normalizePlacesResponse } from '../modules/consciousness_api.js';
import { setup, NodeStub } from './dashboard_dom_fixture.js';

const strip = (html) => html.replace(/<[^>]+>/g, ' ').replace(/\s+/g, ' ').trim();
const raw = (id, fields = {}) => ({ place_id: id, connector: 'chat-bridge', provider: 'chat', label: `Chat · ${id}`, state: 'active',
    revision: 2, notes: [], wakes: [], owner_accounts: [], readiness: { text: 'Connector on', tone: 'ok' }, ...fields });
const RESPONSE = {
    connectors: [{ connector: 'chat-bridge', label: 'Chat', destination: 'vendor:chat', readiness: { text: 'Connector on', tone: 'ok' } }],
    circles: [
        { circle_id: 'team', name: 'Team', text: 'Help the team by the facts.\nKeep money out.', text_author: 'owner', text_at: '', revision: 3 },
        { circle_id: 'home', name: 'Home', text: '', text_author: '', text_at: '', revision: 4 },
    ],
    places: [
        raw('alpha', { circle: { circle_id: 'team', placed_by: 'owner', placed_at: '' }, description: { text: 'Where the team asks', author: 'owner', at: '' },
            last_activity_at: new Date(Date.now() - 39 * 60000).toISOString() }),
        raw('beta', { state: 'paused', circle: { circle_id: 'team', placed_by: 'ouroboros', placed_at: '' }, readiness: { text: 'Off in Skills', tone: 'warn' } }),
        raw('gamma', { label: 'Chat · Team · gamma', circle: { circle_id: 'gone', placed_by: 'owner', placed_at: '' } }),
    ],
};
const VIEW = normalizePlacesResponse(RESPONSE);
const [alpha, beta, gamma] = VIEW.places;
const [team, home] = VIEW.circles;

test('a place row: its own name, the service, the state word, readiness only when not simply on, the last event, its description', () => {
    const row = placeRowHtml(alpha, { title: 'alpha', source: 'Chat' });
    assert.match(strip(row), /^alpha Chat active · 39 min ago Where the team asks$/);
    assert.match(row, /<span class="ui-status" data-tone="ok">active<\/span>/);
    assert.match(row, /<span class="place-row-meta" data-i18n-authored>Where the team asks<\/span>/);
    assert.match(row, /aria-label="Open alpha \(Chat\): active; last event 39 min ago"/);
    assert.doesNotMatch(row, /Connector on/, 'a connector that is simply on says nothing on its row');
    const off = placeRowHtml(beta);
    assert.match(off, /data-tone="muted">paused</);
    assert.match(strip(off), /paused · Off in Skills · no captured event/, 'never «no activity»: what was captured');
    assert.doesNotMatch(off, /place-row-meta/, 'an empty description leaves no line; nothing is copied from a note');
    assert.equal(lastEventText({ lastActivityAt: 'not a time' }), 'last event unknown');
    assert.doesNotMatch(placeRowHtml(alpha), /wakes for|spent|\$/, 'no wake list or spend on the row');
});

test('while consciousness is off a place that is not Paused says it only hears, muted, to the eye and to a screen reader', () => {
    // Owner 06.11: Off silenced every place for 28 hours while each row went on reading «active» with the ok dot.
    const [active, silent, paused] = normalizePlacesResponse({ ...RESPONSE, places: [raw('a', { only_hears: true }),
        raw('s', { state: 'silent', only_hears: true }), raw('p', { state: 'paused' })] }).places;
    const row = placeRowHtml(active, { title: 'a', source: 'Chat' });
    assert.match(row, /<span class="ui-status" data-tone="muted">active · consciousness off<\/span>/);
    assert.match(row, /aria-label="Open a \(Chat\): active · consciousness off; last event /);
    assert.match(placeRowHtml(silent), /data-tone="muted">silent · consciousness off</);
    assert.match(placeRowHtml(paused), /data-tone="muted">paused</);
    assert.doesNotMatch(placeRowHtml(paused), /consciousness off/, 'a Paused place hears nothing new to say');
    assert.equal(alpha.onlyHears, false, 'without the host\'s fact the row keeps its stored word and the ok dot');
    assert.match(placeRowHtml(alpha), /data-tone="ok">active</);
    assert.equal(normalizePlacesResponse({ ...RESPONSE, places: [raw('t', { only_hears: 'true' })] }).places[0].onlyHears, false);
});

test('a row whose name is its service says the service once, to the eye and to a screen reader', () => {
    const [whole] = normalizePlacesResponse({ ...RESPONSE, places: [raw('whole', { label: 'Chat' })] }).places;
    const row = placeRowHtml(whole, { title: 'Chat', source: 'Chat' });
    assert.doesNotMatch(row, /place-row-source/);
    assert.match(row, /aria-label="Open Chat: active; last event no captured event"/);
    assert.equal((strip(row).match(/Chat/g) || []).length, 1);
    // Two such rows in one list keep the instance that tells them apart.
    const twins = placeRowsHtml([whole, { ...whole, id: 'twin', accountId: 'acct-2' }]);
    assert.match(twins, /place-row-source" data-i18n-authored>Chat · acct-2</);
});

test('rows drop the service and the circle\'s own name from titles; the instance tells two same titles apart', () => {
    const html = placeRowsHtml([gamma, { ...gamma, id: 'delta', conversationId: 'room-2' }], { circle: team });
    assert.equal((html.match(/consciousness-place-name" data-i18n-authored>gamma</g) || []).length, 2);
    assert.match(html, /place-row-source" data-i18n-authored>Chat · room-2</);
    assert.match(placeRowsHtml([alpha]), /place-row-source" data-i18n-authored>Chat</);
    assert.equal(placeCountText([alpha, beta]), '2 places · 1 paused');
    assert.equal(placeCountText([alpha]), '1 place');
});

test('a circle\'s words read in the voice of whoever wrote them; empty words say what Edit is for', () => {
    assert.match(circleWordsHtml(team), /<div class="place-note circle-words-text is-owner"><p data-i18n-authored>Help the team by the facts\.\nKeep money out\.<\/p><\/div>/);
    const mine = circleWordsHtml({ ...team, textAuthor: 'ouroboros' });
    assert.match(mine, /<div class="place-note circle-words-text"><p/, 'Ouroboros\'s words are never painted in the owner\'s tint');
    assert.match(strip(mine), /written by Ouroboros/);
    assert.match(circleWordsHtml(home), /No words yet\. Edit says who is here and what Ouroboros is here for\./);
});

test('the directory: places grouped by their circle in the owner\'s order; a lost circle\'s place is in no circle', () => {
    const { groups, unassigned } = groupPlaces(VIEW.circles, VIEW.places);
    assert.deepEqual(groups.map(({ circle, places }) => [circle.id, places.map((p) => p.id)]), [['team', ['alpha', 'beta']], ['home', []]]);
    assert.deepEqual(unassigned.map((p) => p.id), ['gamma']);
    const fold = circleFoldHtml(team, [alpha, beta], { open: 'team' });
    assert.match(fold, /<details class="ui-disclosure circle-fold" data-circle="team" open>/);
    assert.match(strip(fold), /^Team Help the team by the facts\. 2 places · 1 paused/);
    assert.match(fold, /data-circles-act="edit" data-circle="team"/);
    assert.match(fold, /data-circles-act="add-place" data-circle="team"/);
    const editing = circleFoldHtml(team, [], { open: 'team', editing: 'team', drafts: { 'team:text': 'A draft' } });
    assert.match(editing, /data-circles-draft="team:name" data-i18n-authored value="Team"/);
    assert.match(editing, />A draft<\/textarea>/, 'the owner\'s draft, not the saved words');
    assert.match(editing, /data-circles-act="delete" data-circle="team"[^>]*>Delete circle</);
    // No circle yet: the places in no circle are open at once; with none at all, the next step is said.
    assert.match(unassignedFoldHtml([gamma], {}, {}, { alone: true }), /data-circle="none" open>/);
    assert.equal(unassignedFoldHtml([], {}), '', 'with circles and every place in one, no empty row');
    assert.match(strip(unassignedFoldHtml([], {}, {}, { alone: true })), /No place yet\. Add one to start; a circle gathers places that share words\. Add a place/);
    assert.deepEqual(migrationProblemLines([{ code: 'connector_missing', detail: 'gone.' }, { code: 'connector_missing', detail: 'gone' }]),
        ['Could not move an old connection: gone. Connect it again.']);
});

const MODELS = { items: [
    { provider_id: 'openrouter', provider: 'Z.ai (GLM)', id: 'z-ai/glm-5.3', name: 'GLM 5.3', value: 'z-ai/glm-5.3', label: 'OpenRouter · GLM 5.3' },
    { provider_id: 'local', provider: 'Local', id: 'qwen3.gguf', name: 'qwen3.gguf', value: 'local::qwen3.gguf', label: 'Local · qwen3.gguf', use_local: true },
] };

// A circle may have a model (owner 10.11): its places think with it unless one has its own; the default is the
// consciousness model. Team has one here; alpha inherits it, beta keeps its own, and Home has none.
const GLM = 'z-ai/glm-5.3';
const WITH_MODELS = {
    ...RESPONSE,
    circles: [{ ...RESPONSE.circles[0], model: GLM }, { ...RESPONSE.circles[1], model: '' }],
    places: [
        { ...RESPONSE.places[0], model: '', model_route: { value: GLM, layer: 'circle' } },
        { ...RESPONSE.places[1], model: 'openai::gpt-6', model_route: { value: 'openai::gpt-6', layer: 'place' } },
        raw('delta', { circle: { circle_id: 'home', placed_by: 'owner', placed_at: '' }, model: '', model_route: { value: 'claudexor::codex=gpt-6-astra', layer: 'consciousness' } }),
        raw('epsilon', { circle: { circle_id: 'home', placed_by: 'owner', placed_at: '' }, model: 'main', model_route: { value: 'claudexor::codex=gpt-6-astra', layer: 'place' } }),
    ],
};
const MODELLED = normalizePlacesResponse(WITH_MODELS);

test('a circle\'s head names its model only when it has one; an alias is named by its role', () => {
    const [teamModel, homePlain] = MODELLED.circles;
    assert.deepEqual([teamModel.model, homePlain.model, team.model], [GLM, '', ''], 'a server that does not say is no model');
    const head = (circle, places) => strip(circleFoldHtml(circle, places).match(/<summary[\s\S]*?<\/summary>/)[0]);
    const members = MODELLED.places.slice(0, 2);
    assert.match(head(teamModel, members), /2 places · 1 paused · model glm-5\.3$/);
    assert.match(circleFoldHtml(teamModel, members), /<span data-i18n-fmt="model \{model\}">model glm-5\.3<\/span><\/span>\s*<\/summary>/, 'a model\'s name is not chrome');
    assert.match(head(homePlain, []), /0 places$/, 'a circle without a model says nothing: its places follow the consciousness model');
    assert.doesNotMatch(circleFoldHtml(team, [alpha, beta]), /model/);
    assert.match(head({ ...teamModel, model: 'main' }, members), /· model Main$/);
    assert.match(head({ ...teamModel, model: 'claudexor::codex=gpt-6-astra' }, members), /· model gpt-6-astra$/);
});

test('a place row shows a model only when the place overrides; an inheriting place shows none', () => {
    const [inherits, overrides, follows, aliased] = MODELLED.places;
    assert.deepEqual([inherits.modelRoute, overrides.modelRoute], [{ value: GLM, layer: 'circle' }, { value: 'openai::gpt-6', layer: 'place' }]);
    const own = placeRowHtml(overrides, { title: 'beta', source: 'Chat' });
    assert.match(strip(own), /^beta Chat paused · Off in Skills · no captured event · model gpt-6$/);
    assert.match(own, /aria-label="Open beta \(Chat\): paused; Off in Skills; last event no captured event; model gpt-6"/);
    assert.match(strip(placeRowHtml(aliased, { title: 'epsilon', source: 'Chat' })), /· model Main$/, 'a kept alias is named as it was chosen');
    for (const place of [inherits, follows, alpha]) {
        assert.doesNotMatch(placeRowHtml(place, { title: 'x', source: 'Chat' }), /model/, 'the circle\'s, the consciousness model, or a server that does not say');
    }
    // A layer the contract does not name is no layer.
    assert.equal(normalizePlacesResponse({ ...RESPONSE, places: [{ ...RESPONSE.places[0], model_route: { value: GLM, layer: 'elsewhere' } }] }).places[0].modelRoute, null);
});

test('an open circle says how many of its places keep a model of their own, since the circle\'s does not reach them', () => {
    const [teamModel, homePlain] = MODELLED.circles;
    const [inherits, overrides, follows, aliased] = MODELLED.places;
    assert.equal(ownModelsText(teamModel, [inherits, overrides]), '1 place here keeps a model of its own and does not follow the circle\'s.');
    assert.equal(ownModelsText(teamModel, [overrides, aliased]), '2 places here keep a model of their own and do not follow the circle\'s.');
    assert.equal(ownModelsText(homePlain, [follows, aliased]), '1 place here uses a model of its own.');
    assert.equal(ownModelsText(homePlain, [aliased, overrides]), '2 places here use a model of their own.');
    assert.equal(ownModelsText(teamModel, [inherits]), '', 'every place follows: nothing to say');
    assert.equal(ownModelsText(teamModel, [alpha, beta]), '');
    const open = circleFoldHtml(teamModel, [inherits, overrides], { open: 'team' });
    assert.match(open, /<p class="consciousness-copy" data-circle-own-models>1 place here keeps a model of its own and does not follow the circle&#39;s\.<\/p>/);
    assert.doesNotMatch(circleFoldHtml(teamModel, [inherits], { open: 'team' }), /data-circle-own-models/);
    // The editor says it under the Model field, where the circle's model is chosen.
    const editing = strip(circleFoldHtml(teamModel, [inherits, overrides], { open: 'team', editing: 'team' }));
    assert.match(editing, /The model its places think with when they wake; empty uses the Consciousness model in Models\. A place may keep its own\. 1 place here keeps a model of its own and does not follow the circle&#39;s\./);
});

function page(t, { response = RESPONSE, writes = {}, askConfirm = async () => true, openSheet = () => {}, openConnect = () => {},
    readStatus = async () => ({}), readModels = async () => MODELS } = {}) {
    const cleanups = [];
    const env = setup({ after: (cleanup) => cleanups.push(cleanup) });  // restored after the page is destroyed
    // The shared header binder toggles its desk attribute; the stub DOM has no toggleAttribute of its own.
    NodeStub.prototype.toggleAttribute = function (name, on) { if (on) this.attrs[name] = ''; else delete this.attrs[name]; return Boolean(on); };
    NodeStub.prototype.scrollIntoView = function () { this.scrolledIntoView = true; };
    t.after(() => { delete NodeStub.prototype.toggleAttribute; delete NodeStub.prototype.scrollIntoView; });
    const priorCss = globalThis.CSS;
    globalThis.CSS = { escape: (value) => value };
    t.after(() => { globalThis.CSS = priorCss; });
    env.mount.innerHTML = '<div id="content"></div>';
    const state = { activePage: 'circles' };
    let reads = 0, current = response, gate = null;
    const opened = [], connects = [], confirms = [];
    const controller = initCircles({ state, showPage: async () => true, read: async () => { reads += 1; await gate; return normalizePlacesResponse(current); },
        readStatus, readModels, openSheet: (place, options) => opened.push([place.id, options]), askConfirm: async (options) => { confirms.push(options); return askConfirm(options); },
        openConnect: (options) => { connects.push(options); openConnect(options); },
        writes: { createCircle: async () => { throw new Error('unexpected'); }, updateCircle: async () => { throw new Error('unexpected'); },
            deleteCircle: async () => { throw new Error('unexpected'); }, ...writes },
        contributions: { onLifecycle: () => () => {} } });
    let destroyed = false;
    const destroy = () => { if (!destroyed) { destroyed = true; controller.destroy(); } };
    t.after(() => { destroy(); cleanups.forEach((cleanup) => cleanup()); });
    const body = env.mount.querySelector('[data-circles-body]');
    const press = (selector) => controller.page.fire('click', { target: body.querySelector(selector) || controller.page.querySelector(selector) });
    const type = async (key, value) => {
        const field = body.querySelector(`[data-circles-draft="${key}"]`);
        field.value = value;
        await body.fire('input', { target: field });
    };
    // hold(): the next reads wait until the returned release() (a read still in flight when the browser delivers events).
    const hold = () => { let release; gate = new Promise((resolve) => { release = resolve; }); return () => { gate = null; release(); }; };
    return { env, state, controller, destroy, body, press, type, hold, opened, connects, confirms, reads: () => reads, serve: (next) => { current = next; } };
}
const settled = () => new Promise((resolve) => setImmediate(resolve));

test('the page reads places and circles once and opens the first circle; a row opens its sheet with the circles', async (t) => {
    const view = page(t);
    await settled();
    assert.equal(view.reads(), 1);
    assert.match(view.body.textContent, /Help the team by the facts\./);
    const folds = view.body.querySelectorAll('.circle-fold');
    assert.deepEqual(folds.map((fold) => fold.getAttribute('data-circle')), ['team', 'home', 'none']);
    assert.ok(folds[0].hasAttribute('open') && !folds[2].hasAttribute('open'));
    await view.press('[data-place-open="alpha"]');
    assert.equal(view.opened[0][0], 'alpha');
    assert.deepEqual(view.opened[0][1].circles.map((c) => c.id), ['team', 'home'], 'the sheet gets the circles the page read');
    // A write in the sheet repaints the page: the sheet's way back is the row the page holds when it closes.
    NodeStub.prototype.getClientRects = function () { return [{}]; };
    t.after(() => { delete NodeStub.prototype.getClientRects; });
    await view.controller.refresh();
    const row = view.body.querySelector('[data-place-open="alpha"]');
    assert.equal(view.opened[0][1].returnFocus(), row);
    await view.press('[data-circles-act="add-place"]');
    assert.equal(view.connects[0].returnFocus(), view.body.querySelector('[data-circles-act="add-place"]'));
});

test('a new install shows its places at once, open, with Add a place and New circle', async (t) => {
    const view = page(t, { response: { ...RESPONSE, circles: [], places: [raw('only')] } });
    await settled();
    const [fold] = view.body.querySelectorAll('.circle-fold');
    assert.equal(fold.getAttribute('data-circle'), 'none');
    assert.ok(fold.hasAttribute('open'), 'not an apparently empty folded page');
    assert.match(view.body.textContent, /Chat/);
    await view.press('[data-circles-act="add-place"]');
    assert.equal(view.connects.length, 1);
    assert.equal(view.connects[0].circle, null, 'an Add from here lands in no circle');
    assert.ok(view.controller.page.querySelector('[data-circles-act="new"]'), 'New circle sits in the page header');
});

test('New circle: a name first; the words are the owner\'s; a refusal keeps both drafts and says why', async (t) => {
    const made = [];
    let refuse = true;
    const view = page(t, { writes: { createCircle: async (request) => {
        made.push(request);
        if (refuse) { refuse = false; throw Object.assign(new Error('circle_name_taken: another circle has this name.'), { status: 400 }); }
        return { id: 'work', name: request.name, text: request.text, textAuthor: 'owner', textAt: '', revision: 9 };
    } } });
    await settled();
    await view.press('[data-circles-act="new"]');
    await view.press('[data-circles-act="create"]');
    assert.match(view.body.textContent, /Name the circle first\./);
    assert.equal(made.length, 0);
    await view.type('new:name', 'Team');
    await view.type('new:text', 'Words for the team');
    await view.press('[data-circles-act="create"]');
    assert.match(view.body.textContent, /another circle has this name/);
    assert.equal(view.body.querySelector('[data-circles-draft="new:name"]').getAttribute('value'), 'Team', 'the name draft stays');
    await view.type('new:name', 'Work');
    view.serve({ ...RESPONSE, circles: [...RESPONSE.circles, { circle_id: 'work', name: 'Work', text: 'Words for the team', text_author: 'owner', text_at: '', revision: 9 }] });
    await view.press('[data-circles-act="create"]');
    assert.deepEqual(made.at(-1), { name: 'Work', text: 'Words for the team' });
    assert.equal(view.body.querySelector('[data-circle-new]'), null);
    const work = view.body.querySelectorAll('.circle-fold').find((fold) => fold.getAttribute('data-circle') === 'work');
    assert.ok(work.hasAttribute('open'), 'the new circle opens');
});

test('a fold a paint inserted open echoes its own toggle later; the echo does not undo the circle the page chose', async (t) => {
    const work = { circle_id: 'work', name: 'Work', text: '', text_author: 'owner', text_at: '', revision: 1 };
    const view = page(t, { writes: { createCircle: async (request) => ({ id: 'work', name: request.name, text: '', textAuthor: 'owner', textAt: '', revision: 1 }) } });
    await settled();
    await view.press('[data-circles-act="new"]');
    await view.type('new:name', 'Work');
    view.serve({ ...RESPONSE, circles: [...RESPONSE.circles, work] });
    const release = view.hold();
    const created = view.press('[data-circles-act="create"]');
    await settled();  // written; the page chose Work and reads it, while its last paint still shows Team open
    const team = view.body.querySelectorAll('.circle-fold').find((fold) => fold.getAttribute('data-circle') === 'team');
    team.open = true;  // what the browser reports for the fold it inserted open, delivered now (WebKit, Chromium)
    await view.body.fire('toggle', { target: team });
    release();
    await created;
    await settled();
    const open = view.body.querySelectorAll('.circle-fold').filter((fold) => fold.hasAttribute('open'));
    assert.deepEqual(open.map((fold) => fold.getAttribute('data-circle')), ['work'], 'the new circle opens');
    // The owner's own toggles still count: closing the open fold, then opening another.
    const opened = open[0];
    opened.open = false;
    await view.body.fire('toggle', { target: opened });
    const home = view.body.querySelectorAll('.circle-fold').find((fold) => fold.getAttribute('data-circle') === 'home');
    home.open = true;
    await view.body.fire('toggle', { target: home });
    view.serve({ ...RESPONSE, circles: [...RESPONSE.circles, work] });
    await view.controller.refresh();
    const after = view.body.querySelectorAll('.circle-fold').filter((fold) => fold.hasAttribute('open'));
    assert.deepEqual(after.map((fold) => fold.getAttribute('data-circle')), ['home']);
});

test('Edit renames or rewords with the circle\'s revision; a concurrent change keeps the draft and says so', async (t) => {
    const updates = [];
    let conflict = true;
    const view = page(t, { writes: { updateCircle: async (id, change, revision) => {
        updates.push([id, change, revision]);
        if (conflict) { conflict = false; throw Object.assign(new Error('Current revision: 7'), { status: 409, body: { code: 'circle_revision_conflict' } }); }
        return { ...team, ...change, revision: 8 };
    } } });
    await settled();
    await view.press('[data-circles-act="edit"][data-circle="team"]');
    await view.type('team:text', 'New words');
    await view.press('[data-circles-act="save"][data-circle="team"]');
    assert.deepEqual(updates[0], ['team', { text: 'New words' }, 3], 'only what changed, with the revision read');
    assert.match(view.body.textContent, /Changed elsewhere; your text is kept\./);
    assert.match(view.body.querySelector('[data-circles-draft="team:text"]').textContent, /New words/);
    await view.type('team:name', 'Team room');
    await view.press('[data-circles-act="save"][data-circle="team"]');
    assert.deepEqual(updates[1][1], { name: 'Team room', text: 'New words' });
    assert.match(view.body.textContent, /Saved\./);
    await view.press('[data-circles-act="edit"][data-circle="team"]');
    await view.press('[data-circles-act="save"][data-circle="team"]');
    assert.equal(updates.length, 2, 'unchanged words write nothing');
});

test('Delete names how many places stay connected and that they lose the words; it detaches them in one write', async (t) => {
    const deletes = [];
    let answer = false;
    const view = page(t, { askConfirm: async () => answer, writes: { deleteCircle: async (id, revision) => {
        deletes.push([id, revision]);
        return { circle: team, detached: [{ placeId: 'alpha', label: 'Chat · alpha' }, { placeId: 'beta', label: 'Chat · beta' }] };
    } } });
    await settled();
    await view.press('[data-circles-act="edit"][data-circle="team"]');
    await view.press('[data-circles-act="delete"][data-circle="team"]');
    assert.equal(view.confirms[0].body, 'Its 2 places stay connected and keep their own notes, but lose these words.');
    assert.equal(view.confirms[0].details.rows[0].value, team.text, 'the words that would be lost are shown before');
    assert.ok(view.confirms[0].danger);
    assert.equal(deletes.length, 0, 'Cancel deletes nothing');
    answer = true;
    view.serve({ ...RESPONSE, circles: [RESPONSE.circles[1]], places: RESPONSE.places.map((place) => ({ ...place, circle: null })) });
    await view.press('[data-circles-act="delete"][data-circle="team"]');
    assert.deepEqual(deletes, [['team', 3]]);
    assert.match(view.body.textContent, /Deleted Team; its 2 places are in no circle now\./);
    assert.deepEqual(view.body.querySelectorAll('.circle-fold').map((fold) => fold.getAttribute('data-circle')), ['home', 'none']);
});

test('Add a place from a circle carries the circle; another surface can ask for one circle in its edit state', async (t) => {
    const view = page(t);
    await settled();
    await view.press('[data-circles-act="add-place"][data-circle="team"]');
    assert.deepEqual(view.connects[0].circle, { id: 'team', name: 'Team' });
    view.env.window.dispatchEvent(new CustomEvent('ouro:open-circles', { detail: { circle: 'home', edit: true } }));
    await settled(); await settled();
    const fold = view.body.querySelectorAll('.circle-fold').find((node) => node.getAttribute('data-circle') === 'home');
    assert.ok(fold.hasAttribute('open'));
    assert.ok(fold.scrolledIntoView, 'the page scrolls to it');
    assert.ok(view.body.querySelector('[data-circles-draft="home:text"]'), 'the circle opens in its edit state');
});

test('a failed read says so and paints no directory; a place write elsewhere reads again only while shown', async (t) => {
    const view = page(t);
    await settled();
    view.serve(null);
    view.env.window.dispatchEvent(new CustomEvent('ouro:places-changed'));
    await settled();
    assert.equal(view.reads(), 2);
    assert.match(view.body.textContent, /Places could not be read/);
    view.serve(RESPONSE);
    view.env.window.dispatchEvent(new CustomEvent('ouro:page-shown', { detail: { page: 'circles' } }));
    await settled();
    assert.doesNotMatch(view.body.textContent, /could not be read/);
    view.destroy();
    view.env.window.dispatchEvent(new CustomEvent('ouro:places-changed'));
    await settled();
    assert.equal(view.reads(), 3, 'the disposer unsubscribes');
});

// A write the browser has sent and not yet heard back from: the page keeps working (typing, reads, page switches).
const deferred = () => { let resolve, reject; const promise = new Promise((yes, no) => { resolve = yes; reject = no; }); return { promise, resolve, reject }; };
const draftOf = (view, key) => view.body.querySelector(`[data-circles-draft="${key}"]`)?.textContent ?? null;
const shownAgain = async (view) => { view.env.window.dispatchEvent(new CustomEvent('ouro:page-shown', { detail: { page: 'circles' } })); await settled(); };

test('words typed while Save is out stay: what was sent is saved, the newer words stay a draft on the revision it made', async (t) => {
    const updates = [];
    const answer = deferred();
    const view = page(t, { writes: { updateCircle: async (id, change, revision) => {
        updates.push([id, change, revision]);
        return updates.length === 1 ? answer.promise : { ...team, ...change, revision: revision + 1 };
    } } });
    await settled();
    await view.press('[data-circles-act="edit"][data-circle="team"]');
    await view.type('team:text', 'First words');
    const saving = view.press('[data-circles-act="save"][data-circle="team"]');
    await settled();
    await view.type('team:text', 'First words, and more');
    // The write landed; a page switch reads the circle again before its answer comes.
    view.serve({ ...RESPONSE, circles: [{ ...RESPONSE.circles[0], text: 'First words', revision: 4 }, RESPONSE.circles[1]] });
    await shownAgain(view);
    answer.resolve({ ...team, text: 'First words', revision: 4 });
    await saving;
    await settled();
    assert.deepEqual(updates[0], ['team', { text: 'First words' }, 3]);
    assert.equal(draftOf(view, 'team:text'), 'First words, and more', 'the newer words stay in the editor');
    assert.match(view.body.textContent, /Saved\. Newer edits are still unsaved\./);
    await view.press('[data-circles-act="save"][data-circle="team"]');
    assert.deepEqual(updates[1], ['team', { text: 'First words, and more' }, 4], 'saved on the revision the first Save made');
    assert.equal(draftOf(view, 'team:text'), null, 'nothing newer: the editor closes');
});

test('a draft keeps the revision it was written on across reads; a stale Save meets the conflict, saving again is a choice', async (t) => {
    const updates = [];
    const theirs = { ...RESPONSE.circles[0], text: 'Their words', revision: 4 };
    const view = page(t, { writes: { updateCircle: async (id, change, revision) => {
        updates.push([id, change, revision]);
        if (revision !== 4) throw Object.assign(new Error('Current revision: 4'), { status: 409, body: { code: 'circle_revision_conflict', circle: theirs } });
        return { ...team, ...change, revision: 5 };
    } } });
    await settled();
    await view.press('[data-circles-act="edit"][data-circle="team"]');
    await view.type('team:text', 'My words');  // written on revision 3
    view.serve({ ...RESPONSE, circles: [theirs, RESPONSE.circles[1]] });  // another client saves revision 4
    await shownAgain(view);
    assert.equal(draftOf(view, 'team:text'), 'My words', 'the draft survives the read');
    await view.press('[data-circles-act="save"][data-circle="team"]');
    assert.deepEqual(updates[0], ['team', { text: 'My words' }, 3], 'the draft\'s own revision, never the newer read\'s');
    assert.match(view.body.textContent, /Changed elsewhere; your text is kept\./);
    assert.equal(draftOf(view, 'team:text'), 'My words');
    await view.press('[data-circles-act="save"][data-circle="team"]');
    assert.deepEqual(updates[1], ['team', { text: 'My words' }, 4], 'the Save after the conflict is the deliberate one');
    assert.match(view.body.textContent, /Saved\./);
});

test('words read before any typing are the ones a draft starts from; Cancel after a conflict shows the other words', async (t) => {
    const updates = [];
    const theirs = { ...RESPONSE.circles[0], text: 'Their words', revision: 4 };
    const view = page(t, { writes: { updateCircle: async (id, change, revision) => {
        updates.push([id, change, revision]);
        throw Object.assign(new Error('Current revision: 5'), { status: 409, body: { code: 'circle_revision_conflict', circle: { ...theirs, text: 'Later words', revision: 5 } } });
    } } });
    await settled();
    await view.press('[data-circles-act="edit"][data-circle="team"]');
    view.serve({ ...RESPONSE, circles: [theirs, RESPONSE.circles[1]] });
    await view.controller.refresh();
    assert.equal(draftOf(view, 'team:text'), 'Their words', 'no draft yet: the editor shows what was read');
    await view.type('team:text', 'Their words, mine after');
    view.serve({ ...RESPONSE, circles: [{ ...theirs, text: 'Later words', revision: 5 }, RESPONSE.circles[1]] });
    await view.press('[data-circles-act="save"][data-circle="team"]');
    assert.deepEqual(updates, [['team', { text: 'Their words, mine after' }, 4]]);
    await settled();
    await view.press('[data-circles-act="cancel"][data-circle="team"]');
    assert.equal(draftOf(view, 'team:text'), null);
    assert.match(view.body.textContent, /Later words/, 'Cancel drops the draft and shows the words saved elsewhere');
});

test('words typed while a new circle is being made stay: it is made once and the newer words open as its edit', async (t) => {
    const made = [], updates = [];
    const answer = deferred();
    const work = { circle_id: 'work', name: 'Work', text: 'Words', text_author: 'owner', text_at: '', revision: 1 };
    const view = page(t, { writes: {
        createCircle: async (request) => { made.push(request); return answer.promise; },
        updateCircle: async (id, change, revision) => { updates.push([id, change, revision]); return { id, name: 'Work', text: '', ...change, textAuthor: 'owner', textAt: '', revision: revision + 1 }; } } });
    await settled();
    await view.press('[data-circles-act="new"]');
    await view.type('new:name', 'Work');
    await view.type('new:text', 'Words');
    const creating = view.press('[data-circles-act="create"]');
    await settled();
    await view.type('new:text', 'Words, and more');
    view.serve({ ...RESPONSE, circles: [...RESPONSE.circles, work] });
    answer.resolve({ id: 'work', name: 'Work', text: 'Words', textAuthor: 'owner', textAt: '', revision: 1 });
    await creating;
    await settled();
    assert.deepEqual(made, [{ name: 'Work', text: 'Words' }]);
    assert.equal(view.body.querySelector('[data-circle-new]'), null, 'a made circle is never offered to be made twice');
    assert.equal(draftOf(view, 'work:text'), 'Words, and more', 'the newer words stay, as an edit of the circle made');
    assert.equal(view.env.document.activeElement, view.body.querySelector('[data-circles-draft="work:name"]'),
        'the focus moves from the New circle field to the same field of the circle made');
    await view.press('[data-circles-act="save"][data-circle="work"]');
    assert.deepEqual(updates, [['work', { text: 'Words, and more' }, 1]]);
    assert.equal(made.length, 1);
});

test('nothing typed while a write is out: a made circle opens as it was saved, a saved edit closes its editor', async (t) => {
    const view = page(t, { writes: {
        createCircle: async (request) => ({ id: 'work', name: request.name, text: request.text, textAuthor: 'owner', textAt: '', revision: 1 }),
        updateCircle: async (id, change, revision) => ({ ...team, ...change, revision: revision + 1 }) } });
    await settled();
    await view.press('[data-circles-act="new"]');
    await view.type('new:name', 'Work');
    view.serve({ ...RESPONSE, circles: [...RESPONSE.circles, { circle_id: 'work', name: 'Work', text: '', text_author: 'owner', text_at: '', revision: 1 }] });
    await view.press('[data-circles-act="create"]');
    assert.equal(draftOf(view, 'work:text'), null, 'no editor: nothing was typed after the press');
    await view.press('[data-circles-act="edit"][data-circle="team"]');
    await view.type('team:name', 'Team room');
    await view.press('[data-circles-act="save"][data-circle="team"]');
    assert.equal(draftOf(view, 'team:name'), null);
    assert.match(view.body.textContent, /Saved\./);
});

// ---- the connectors' status: a whole-account place's row takes the account's own name from it ----------------------
const STATUS_URL = '/api/extensions/chat-bridge/status';
const withStatus = (url = STATUS_URL) => ({ ...RESPONSE,
    connectors: [{ ...RESPONSE.connectors[0], ...(url ? { source: { url } } : {}) }],
    places: [...RESPONSE.places, raw('whole', { label: 'Chat', circle: { circle_id: 'team', placed_by: 'owner', placed_at: '' } })] });
const titleOf = (view) => view.body.querySelector('[data-place-open="whole"]')?.querySelector('.consciousness-place-name')?.textContent;
// Every status read the page asks for, answered when the test says.
function statusReads() {
    const calls = [];
    return { calls, readStatus: (connector) => { const call = { url: connector.source?.url, ...deferred() }; calls.push(call); return call.promise; } };
}
const changedElsewhere = async (view) => { view.env.window.dispatchEvent(new CustomEvent('ouro:places-changed')); await settled(); };

test('a failed status read is asked again at the next refresh; a completed one is never asked twice', async (t) => {
    const status = statusReads();
    const view = page(t, { response: withStatus(), readStatus: status.readStatus });
    await settled();
    assert.equal(status.calls.length, 1);
    status.calls[0].reject(new Error('The connector status could not be read.'));
    await settled();
    assert.equal(titleOf(view), 'Chat', 'no name yet: the service');
    await changedElsewhere(view);  // an existing refresh event; no timer
    assert.equal(status.calls.length, 2, 'asked again');
    status.calls[1].resolve({ team_name: 'Acme team' });
    await settled();
    assert.equal(titleOf(view), 'Acme team');
    await shownAgain(view);
    await changedElsewhere(view);
    assert.equal(status.calls.length, 2, 'a completed status is not read again');
    assert.equal(titleOf(view), 'Acme team');
});

test('a status answer a newer refresh overtook still lands; the read still out is not asked twice', async (t) => {
    const status = statusReads();
    const view = page(t, { response: withStatus(), readStatus: status.readStatus });
    await settled();
    await shownAgain(view);  // the owner left and came back while the status read was out: a newer refresh
    assert.equal(status.calls.length, 1, 'the read still out is not asked again');
    status.calls[0].resolve({ team_name: 'Acme team' });
    await settled();
    assert.equal(titleOf(view), 'Acme team', 'its answer still lands');
    await shownAgain(view);
    assert.equal(status.calls.length, 1);
});

test('a status answer that comes while the page is hidden waits until it is shown again', async (t) => {
    const status = statusReads();
    const view = page(t, { response: withStatus(), readStatus: status.readStatus });
    await settled();
    view.state.activePage = 'dashboard';
    const reads = view.reads();
    status.calls[0].resolve({ team_name: 'Acme team' });
    await settled();
    assert.equal(titleOf(view), 'Chat', 'a hidden page is not repainted');
    assert.equal(view.reads(), reads, 'and reads nothing while hidden');
    view.state.activePage = 'circles';
    await shownAgain(view);
    assert.equal(titleOf(view), 'Acme team', 'shown again, the answer is there');
    assert.equal(status.calls.length, 1);
});

test('a status answer in a backgrounded tab marks the page behind; the tab\'s return reads it again', async (t) => {
    const status = statusReads();
    const view = page(t, { response: withStatus(), readStatus: status.readStatus });
    await settled();
    view.env.document.hidden = true;
    status.calls[0].resolve({ team_name: 'Acme team' });
    await settled();
    assert.equal(titleOf(view), 'Chat');
    view.env.document.hidden = false;
    view.env.document.dispatchEvent(new CustomEvent('visibilitychange'));
    await settled();
    assert.equal(titleOf(view), 'Acme team');
});

test('a disposed page keeps what it showed when a status answer comes late', async (t) => {
    const status = statusReads();
    const view = page(t, { response: withStatus(), readStatus: status.readStatus });
    await settled();
    view.destroy();
    status.calls[0].resolve({ team_name: 'Acme team' });
    await settled();
    assert.equal(titleOf(view), 'Chat');
});

test('a connector that changes or loses its status route never revives an older answer', async (t) => {
    const status = statusReads();
    const view = page(t, { response: withStatus(), readStatus: status.readStatus });
    await settled();
    view.serve(withStatus('/api/extensions/chat-bridge/status2'));
    await changedElsewhere(view);
    assert.deepEqual(status.calls.map((call) => call.url), [STATUS_URL, '/api/extensions/chat-bridge/status2'], 'the new route is asked');
    status.calls[0].resolve({ team_name: 'Old team' });
    await settled();
    assert.equal(titleOf(view), 'Chat', 'an answer from the route it no longer declares is dropped');
    status.calls[1].resolve({ team_name: 'New team' });
    await settled();
    assert.equal(titleOf(view), 'New team');
    view.serve(withStatus(''));
    await changedElsewhere(view);
    assert.equal(titleOf(view), 'Chat', 'no route, no status');
    assert.equal(status.calls.length, 2);
});

test('Edit holds the circle\'s model in the place sheet\'s chooser; it saves with the circle under its revision, and shows', async (t) => {
    const calls = [];
    let catalogReads = 0;
    const view = page(t, { response: WITH_MODELS, readModels: async () => { catalogReads += 1; return MODELS; },
        writes: { updateCircle: async (id, change, revision) => { calls.push([id, change, revision]); return { ...MODELLED.circles[1], ...change, id, revision: revision + 1 }; } } });
    await settled();
    assert.equal(catalogReads, 0, 'the catalog is not read for a page nobody edits');
    await view.press('[data-circles-act="edit"][data-circle="home"]');
    await settled();
    const field = () => view.body.querySelector('[data-circles-draft="home:model"]');
    assert.equal(field().getAttribute('value'), '', 'a circle without a model holds none');
    assert.equal(field().getAttribute('placeholder'), 'Empty uses the consciousness model');
    assert.ok(field().hasAttribute('data-model-chooser') && field().getAttribute('role') === 'combobox', 'the shared editable chooser');
    // Suggestions come once, a route each: the local model has none and is not offered.
    await view.controller.refresh();
    assert.deepEqual(view.body.querySelectorAll('.ui-model-option').map((node) => node.getAttribute('data-model-value')), [GLM]);
    await view.press('[data-circles-act="cancel"][data-circle="home"]');
    await view.press('[data-circles-act="edit"][data-circle="home"]');
    assert.equal(catalogReads, 1);
    // Typed and saved: only what changed is sent, under the revision the page read.
    await view.type('home:model', ` ${GLM} `);
    view.serve({ ...WITH_MODELS, circles: [WITH_MODELS.circles[0], { ...WITH_MODELS.circles[1], model: GLM, revision: 5 }] });
    await view.press('[data-circles-act="save"][data-circle="home"]');
    await settled();
    assert.deepEqual(calls, [['home', { model: GLM }, 4]]);
    const head = () => strip(view.body.querySelectorAll('.circle-fold').find((node) => node.getAttribute('data-circle') === 'home').querySelector('summary').innerHTML);
    assert.match(head(), /2 places · model glm-5\.3$/, 'a circle with a model names it');
    // Emptied and saved: the circle has no model again and its head says nothing.
    await view.press('[data-circles-act="edit"][data-circle="home"]');
    assert.equal(field().getAttribute('value'), GLM);
    await view.type('home:model', '');
    view.serve(WITH_MODELS);
    await view.press('[data-circles-act="save"][data-circle="home"]');
    await settled();
    assert.deepEqual(calls[1], ['home', { model: '' }, 5]);
    assert.match(head(), /2 places$/);
    // Unchanged, nothing is written.
    await view.press('[data-circles-act="edit"][data-circle="home"]');
    await view.press('[data-circles-act="save"][data-circle="home"]');
    assert.equal(calls.length, 2);
    assert.match(view.body.textContent, /Nothing to save\./);
});

test('a new circle may be made with a model; suggestions that cannot be read say so and keep what was typed', async (t) => {
    const made = [];
    const view = page(t, { readModels: async () => { throw new Error('offline'); },
        writes: { createCircle: async (circle) => { made.push(circle); return { id: 'work', name: circle.name, text: '', model: circle.model || '', revision: 1 }; } } });
    await settled();
    await view.press('[data-circles-act="new"]');
    await settled();
    assert.match(view.body.textContent, /Model suggestions could not be read; your model is kept\./);
    await view.type('new:name', 'Work');
    await view.type('new:model', 'openai::gpt-6');
    await view.controller.refresh();
    assert.equal(view.body.querySelector('[data-circles-draft="new:model"]').getAttribute('value'), 'openai::gpt-6', 'a repaint keeps the draft');
    await view.press('[data-circles-act="create"]');
    await settled();
    assert.deepEqual(made, [{ name: 'Work', text: '', model: 'openai::gpt-6' }]);
    assert.equal(view.body.querySelector('[data-circles-draft="new:model"]'), null, 'made once: no second New circle');
});

for (const count of [1, 18]) test(`circle editors share the catalog producer at ${count} accounts without another read`, async (t) => {
    const source = { path: '/vendor/codex', version: '2.0.0' };
    const items = Array.from({ length: count }, (_, index) => ({ id: 'choice', value: 'claudexor::transport=choice',
        source_id: 'transport', source: 'Claudexor', credential_profile_id: String(index) }));
    const catalog = { items, model_sources: [{ id: 'transport', credentialHarness: 'codex' }],
        account_catalogs: [{ source: 'transport', accounts: items.map((item) => ({ catalog: {
            source: 'transport', credentialProfileId: item.credential_profile_id, sourceProgram: source, models: [{ id: item.id }],
        } })) }] };
    let reads = 0;
    const view = page(t, { readModels: async () => { reads++; return catalog; } });
    await settled();
    assert.equal(reads, 0);
    await view.press('[data-circles-act="edit"][data-circle="home"]');
    await settled();
    const field = view.body.querySelector('[data-circles-draft="home:model"]');
    const programs = (input) => JSON.parse(input.dataset.modelPrograms);
    assert.equal(programs(field).length, 1);
    assert.deepEqual(programs(field)[0].programs, [source]);
    assert.equal(programs(field)[0].mixed, false);
    await view.type('home:model', 'owner/private');
    const publish = (detail) => view.env.document.dispatchEvent(new CustomEvent('settings-model-catalog:updated', { detail }));
    const api = { id: 'api', value: 'openai::api' };
    publish({ ...catalog, items: [...items, api] });
    assert.equal(view.body.querySelector('[data-circles-draft="home:model"]'), field, 'catalog updates keep the real field');
    assert.equal(field.value, 'owner/private');
    assert.equal(programs(field)[0].mixed, true);
    publish({ read_state: 'transport' });  // a Refresh that failed: settings_catalog.js broadcasts no items
    assert.equal(programs(field)[0].mixed, true, 'a failed refresh keeps the suggestions read before it');
    await view.press('[data-circles-act="new"]');
    await settled();
    const created = view.body.querySelector('[data-circles-draft="new:model"]');
    assert.deepEqual(programs(created), programs(field), 'new and saved editors use the same producer');
    assert.equal(reads, 1, 'the existing catalog read supplies every editor and refresh event');
    // A failed read Settings broadcasts (a transport error has no list) changes nothing and throws nothing.
    const before = programs(created);
    publish({ read_state: 'transport', errors: [{ provider_id: 'catalog', error: 'Timed out' }] });
    publish({ items: [], read_state: 'failed' });
    assert.deepEqual(programs(created), before, 'a failed read keeps the suggestions shown');
    publish({ items: [api], model_sources: [], account_catalogs: [] });
    for (const input of view.body.querySelectorAll('[data-model-chooser]')) assert.deepEqual(programs(input), []);
    view.destroy();
    publish(catalog);
    assert.deepEqual(programs(created), [], 'destroy releases the catalog listener');
    assert.equal(reads, 1);
});

test('a model typed while Save is out stays a draft on the revision that Save made', async (t) => {
    let finish;
    const calls = [];
    const view = page(t, { response: WITH_MODELS, writes: { updateCircle: (id, change, revision) => { calls.push([change, revision]);
        return new Promise((resolve) => { finish = () => resolve({ ...MODELLED.circles[1], ...change, id, revision: revision + 1 }); }); } } });
    await settled();
    await view.press('[data-circles-act="edit"][data-circle="home"]');
    await view.type('home:model', GLM);
    const saving = view.press('[data-circles-act="save"][data-circle="home"]');
    await view.type('home:model', 'openai::gpt-6');
    finish(); await saving; await settled();
    assert.equal(view.body.querySelector('[data-circles-draft="home:model"]').getAttribute('value'), 'openai::gpt-6');
    assert.match(view.body.textContent, /Saved\. Newer edits are still unsaved\./);
    view.serve({ ...WITH_MODELS, circles: [WITH_MODELS.circles[0], { ...WITH_MODELS.circles[1], model: GLM, revision: 5 }] });
    await view.controller.refresh();
    const next = view.press('[data-circles-act="save"][data-circle="home"]');
    await settled(); finish(); await next;
    assert.deepEqual(calls, [[{ model: GLM }, 4], [{ model: 'openai::gpt-6' }, 5]]);
});

test('a circle editor names a model\'s catalog price while its places pay it, and says it again on a tier or language change', async (t) => {
    const { modelCatalog } = await import('../modules/model_catalog_store.js');
    const { applyPayload, setMissTransport } = await import('../modules/i18n.js');
    setMissTransport(() => Promise.resolve(null));
    const priced = { items: [{ ...MODELS.items[0], price: { input: 1, output: 3 } }] };
    modelCatalog.adoptSettings({ OUROBOROS_PROCESSING_PREFERENCE: 'fast' });
    const view = page(t, { response: WITH_MODELS, readModels: async () => priced });
    await settled();
    await view.press('[data-circles-act="edit"][data-circle="home"]');
    await settled();
    const said = async () => { await view.controller.refresh(); return view.body.querySelectorAll('.ui-model-option-meta').map((node) => node.textContent); };
    assert.deepEqual(await said(), [], "a circle's places run as leads on the default tier, here Fast");
    modelCatalog.adoptSettings({ OUROBOROS_PROCESSING_PREFERENCE: 'standard' });
    assert.deepEqual(await said(), ['$1 in / $3 out'], 'a tier Settings saves reaches the open page');
    // The words alone: without a document i18n.js leaves the fixture's page unwalked.
    const language = (payload) => { const kept = globalThis.document; globalThis.document = undefined; try { applyPayload(payload); } finally { globalThis.document = kept; } };
    language({ language: 'ru', english: false, revision: 51, entries: { '{input} in / {output} out': { text: '{input} вход / {output} выход' } } });
    try {
        view.env.window.dispatchEvent(new CustomEvent('ouro:language-changed'));
        assert.deepEqual(await said(), ['$1 вход / $3 выход']);
    } finally {
        language({ language: '', english: true, revision: 52, entries: {} });
    }
});
