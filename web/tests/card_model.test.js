// A work's model on its card (card_model.js; DESIGN "A work's model on its card"): the first fact of the card's meta
// line in a node of its own, said from the server's published fact (CONTRACT §5), pressable only where the server
// offers the work's `model` control, amber while the work runs on a fallback, and one POST per act (CONTRACT §6).
// Fixtures are in the contract's shapes; geometry and the real page are the browser lane's.
import assert from 'node:assert/strict';
import test from 'node:test';
import { APPLIED_WORDS, basisWords, closeCardModel, leadView, openCardModel, paintCardModel } from '../modules/card_model.js';
import { createModelCatalogStore } from '../modules/model_catalog_store.js';
import { applyCardWorkViews, emptyWorkState, reconcileWorkState } from '../modules/work_status.js';
import { stubDocument } from './composer_controls_fixture.js';

const ASTRA = 'claudexor::codex=gpt-6-astra';
const OPUS = 'anthropic::claude-opus-5.5';
const GLM = 'z-ai/glm-5.3';
const route = (model, local = false) => ({ model, use_local: local });
const lead = (model, basis, extra = {}) => ({ ...route(model), basis, ...extra });
const CATALOG = {
    model_sources: [{ id: 'codex', label: 'Codex', credentialHarness: 'codex' }],
    items: [{ provider_id: 'claudexor', provider: 'Codex', source_id: 'codex', id: 'gpt-6-astra', name: 'gpt-6-astra',
        value: ASTRA, credential_profile_id: 'home', availability: 'available' },
    { provider_id: 'openrouter', provider: 'OpenRouter', id: GLM, name: GLM, value: GLM },
    { provider_id: 'openrouter', provider: 'OpenRouter', id: 'moonshotai/kimi-k3', name: 'moonshotai/kimi-k3', value: 'moonshotai/kimi-k3' }],
};
const flush = async () => { for (let i = 0; i < 4; i += 1) await new Promise((resolve) => setImmediate(resolve)); };

/* One card's meta line as chat_activity.js leaves it, with everything the fact's acts reach injected. */
function card({ recents = [route(GLM)], answer = null } = {}) {
    const doc = stubDocument();
    doc.body = doc.createElement('body');
    const meta = doc.createElement('div');
    meta.className = 'chat-live-meta';
    meta.isConnected = true;  // on the page: what is put into it is too
    const tools = doc.createElement('span');
    tools.className = 'chat-live-meta-part';
    tools.textContent = '3 tool calls';
    meta.append(tools);
    const record = { groupId: 't1', metaEl: meta };
    const calls = [];
    const toasts = [];
    const rerenders = [];
    const reply = { value: answer };
    const store = createModelCatalogStore({ readCatalog: async () => CATALOG,
        readPreferences: async () => ({ model_recents: recents }), readSettings: async () => ({ _meta: { connections: [] } }) });
    const deps = {
        store, coarse: () => false, place: () => ({ reposition() {}, destroy() {} }),
        toast: (message, tone) => toasts.push([message, tone]),
        rerender: (seen) => rerenders.push(seen),
        act: async (taskId, requestId, act) => {
            calls.push([taskId, act]);
            assert.ok(requestId, 'every act carries a request id');
            const said = await reply.value;
            if (said instanceof Error) throw said;
            return said || { ok: true, task_id: taskId, applied: 'next_step', lead: lead(act.model || ASTRA, act.action === 'follow' ? 'main' : 'owner') };
        },
    };
    const paint = (view) => paintCardModel(record, view && { target: 't1', main: route(ASTRA), since: '', queued: false, ...view }, deps);
    const part = () => meta.children.find((node) => node.classList.contains('card-model')) || null;
    const fact = () => part()?.children.find((node) => node.classList.contains('card-model-fact')) || null;
    // A part's words as a reader gets them; the name is its three spans, joined by their own (no-break) separators.
    const said = (name) => {
        const node = part().querySelector(`.card-model-${name}`);
        if (node.hidden) return '';
        return (name === 'name' ? node.children.map((child) => child.textContent).join('') : node.textContent).replaceAll('\u00a0', ' ');
    };
    const popup = () => doc.body.children.find((node) => node.classList.contains('card-model-pop')) || null;
    const fire = (node, type, event) => Promise.all((node.listeners.get(type) || []).map((fn) => fn(event)));
    const acts = () => popup().querySelector('.card-model-acts').children
        .map((node) => [node.dataset.act, node.children[0].textContent, node.children[1].textContent]);
    const pressAct = (name) => {
        const box = popup().querySelector('.card-model-acts');
        return fire(box, 'click', { target: { closest: () => box.children.find((node) => node.dataset.act === name) } });
    };
    const list = () => popup().querySelector('.card-model-list');
    const rows = () => [...list().innerHTML.matchAll(/<div class="chat-model-row[^"]*" role="option" data-row="([^"]*)" aria-selected="(true|false)">(.*?)<\/div>/g)]
        .map(([, key, selected, inner]) => ({ key, selected: selected === 'true',
            text: inner.replace(/<svg.*?<\/svg>/g, '').replace(/<[^>]+>/g, ' ').replace(/\s+/g, ' ').trim() }));
    const take = (key) => fire(list(), 'click', { target: { closest: () => list().querySelectorAll('.chat-model-row').find((node) => node.dataset.row === key) } });
    return { doc, meta, record, tools, calls, toasts, rerenders, reply, store, paint, part, fact, said, popup, acts, pressAct, list, rows, take, fire };
}

test('the words of a fact follow its basis; a follower says nothing beside its model', () => {
    const words = (basis, extra) => leadView(lead(GLM, basis, extra)).why;
    assert.equal(words('main'), '');
    assert.equal(words('room'), 'this room\'s model');
    assert.equal(words('owner'), 'you chose it');
    assert.equal(words('owner_words'), 'you asked for it');
    assert.equal(words('mind'), 'Ouroboros switched to it');
    assert.equal(words('light'), 'follows Light', 'any other basis is the role the work follows');
    assert.equal(words('circle'), 'its circle\'s model');
    assert.equal(basisWords('main'), 'follows Main', 'the popover and the sheet say it in words');
    assert.equal(leadView(lead(GLM, 'owner')).name, 'OpenRouter · glm-5.3');
    assert.equal(leadView({ basis: 'main' }), null, 'a fact without a model is no fact');
});

test('a fallback names where it came from, since when and what it left', () => {
    const fell = lead(OPUS, 'fallback', { role: 'fallback:1', primary: lead(ASTRA, 'owner'), since: 1, cause: 'gpt-6-astra did not answer' });
    const said = leadView(fell, { since: '02:14' });
    assert.equal(said.name, 'fallback from Settings · claude-opus-5.5 · since 02:14');
    assert.equal(said.why, 'left gpt-6-astra');
    assert.equal(said.fallback, true);
    assert.equal(said.cause, 'gpt-6-astra did not answer');
    assert.equal(leadView({ ...fell, primary: lead(ASTRA, 'main') }, { since: '02:14' }).why, 'left Main (gpt-6-astra)');
    assert.equal(leadView(fell).name, 'fallback from Settings · claude-opus-5.5', 'no time is said when the fact gives none');
    assert.equal(leadView({ ...fell, role: 'main' }).name, 'fallback to Main · claude-opus-5.5');
    assert.doesNotMatch(said.name + said.why, /\$|\d+\.\d\d/, 'no money on the amber line');
});

test('a pending route shows both models and when the change lands; the same route shows none', () => {
    const moving = lead(ASTRA, 'main', { pending: route(GLM) });
    assert.equal(leadView(moving).pending, '→ glm-5.3 at the next step');
    assert.equal(leadView(moving, { queued: true }).pending, '→ glm-5.3 when it starts');
    assert.equal(leadView(lead(ASTRA, 'main', { pending: route(ASTRA) })).pending, '');
    assert.equal(leadView(lead(ASTRA, 'main')).pending, '');
});

test('the fact is the first node of the line and a repaint of the line never touches it', () => {
    const c = card();
    assert.equal(c.paint({ lead: lead(ASTRA, 'main') }), true);
    assert.ok(c.meta.children[0] === c.part(), 'first in the line');
    const rest = c.meta.children[1];
    assert.equal(rest.className, 'card-model-rest');
    assert.ok(c.record.metaEl === rest, 'the line\'s writer now writes the rest');
    assert.ok(rest.children.length === 1 && rest.children[0] === c.tools, 'what the line held moved with it');
    assert.equal(c.rerenders.length, 1, 'the line is written once more: it no longer names the model itself');
    assert.equal(c.said('name'), 'Codex · gpt-6-astra');
    assert.equal(c.said('why'), '');
    const node = c.fact();
    c.record.metaEl.innerHTML = '<span class="chat-live-meta-part">4 tool calls</span>';  // chat_activity.js repaints
    c.paint({ lead: lead(ASTRA, 'main') });
    assert.ok(c.fact() === node, 'the same node: an open popover keeps its anchor');
    assert.equal(c.rerenders.length, 1, 'an unchanged fact asks for nothing');
    assert.equal(c.part().querySelector('.chat-live-meta-sep').hidden, false, 'a separator while something follows');
    c.record.metaEl.innerHTML = '';
    c.paint({ lead: lead(ASTRA, 'main') });
    assert.equal(c.part().querySelector('.chat-live-meta-sep').hidden, true, 'none when the fact stands alone');
});

test('the fact is a button where the server offers the model control and plain text where it does not', () => {
    const c = card();
    c.paint({ lead: lead(GLM, 'owner') });
    assert.equal(c.fact().tagName, 'BUTTON');
    assert.equal(c.fact().getAttribute('aria-haspopup'), 'dialog');
    assert.equal(c.fact().getAttribute('aria-label'), 'This work\'s model: glm-5.3, OpenRouter, you chose it');
    assert.equal(c.said('why'), 'you chose it');
    assert.equal(c.part().dataset.basis, 'own');
    c.paint({ lead: lead(GLM, 'owner'), target: '' });
    assert.equal(c.fact().tagName, 'SPAN', 'without the control it is text');
    assert.equal(c.fact().listeners.size, 0, 'and nothing to press');
    assert.equal(openCardModel(c.record), null, 'no popover without the control');
    assert.equal(c.popup(), null);
    assert.equal(c.said('name'), 'OpenRouter · glm-5.3', 'the words stay');
});

test('a fallback is amber and a follower is not; amber ends when the server\'s fact says so', () => {
    const c = card();
    c.paint({ lead: lead(ASTRA, 'main') });
    assert.equal(c.fact().classList.contains('model-fallback'), false);
    assert.equal(c.part().dataset.basis, 'main');
    c.paint({ lead: lead(OPUS, 'fallback', { role: 'fallback:0', primary: lead(ASTRA, 'main'), since: 5 }), since: '02:14' });
    assert.equal(c.fact().classList.contains('model-fallback'), true);
    assert.equal(c.part().dataset.basis, 'fallback');
    assert.equal(c.said('name'), 'fallback from Settings · claude-opus-5.5 · since 02:14');
    assert.equal(c.said('why'), 'left Main (gpt-6-astra)');
    c.paint({ lead: lead(ASTRA, 'main') });
    assert.equal(c.fact().classList.contains('model-fallback'), false, 'the next fact is not a fallback: no amber');
    assert.equal(c.said('why'), '');
});

test('a work whose fact is gone gets back the line it had', () => {
    const c = card();
    c.paint({ lead: lead(GLM, 'room') });
    assert.equal(c.record.lead.model, GLM);
    assert.equal(c.paint(null), false);
    assert.ok(c.meta.children.length === 1 && c.meta.children[0] === c.tools, 'the line as it was');
    assert.ok(c.record.metaEl === c.meta);
    assert.equal(c.record.lead, null);
    assert.equal(c.rerenders.length, 2, 'written again: the line names the model itself');
    assert.equal(c.paint(null), false, 'a card that never had a fact is left alone');
    assert.equal(c.rerenders.length, 2);
});

test('the popover offers Main, the recent models and Follow Main only to a work with a model of its own', async () => {
    const c = card();
    c.paint({ lead: lead(GLM, 'owner') });
    c.fact().click();
    await flush();
    assert.equal(c.fact().getAttribute('aria-expanded'), 'true');
    assert.equal(c.popup().querySelector('.card-model-pop-title').textContent, 'Model for this work · you chose it');
    assert.deepEqual(c.acts(), [['follow', 'Follow Main', 'Codex · gpt-6-astra']]);
    assert.deepEqual(c.rows().slice(0, 2).map((row) => [row.key, row.selected]), [[`route|${ASTRA}`, false], [`route|${GLM}`, true]]);
    assert.match(c.rows()[0].text, /^gpt-6-astra Main/);
    assert.match(c.list().innerHTML, new RegExp(APPLIED_WORDS.next_step.replace(/[.;]/g, '\\$&')));
    assert.ok(c.doc.activeElement === c.popup().querySelector('.chat-model-query'), 'a keyboard types at once');
    c.fact().click();
    assert.equal(c.popup(), null, 'a second press closes it');
    assert.equal(c.fact().getAttribute('aria-expanded'), 'false');

    c.paint({ lead: lead(ASTRA, 'main'), queued: true });
    c.fact().click();
    await flush();
    assert.deepEqual(c.acts(), [], 'a follower has nothing to return to');
    assert.equal(c.popup().querySelector('.card-model-pop-title').textContent, 'Model for this work · follows Main');
    assert.equal(c.rows()[0].selected, true, 'Main\'s row is its own');
    assert.match(c.list().innerHTML, /Applies when it starts\./, 'a queued work says when its change lands');
    closeCardModel();
    assert.equal(c.popup(), null);
});

test('a pick sets this work\'s model, Main\'s row follows Main, and the answer says when it applies', async () => {
    const c = card();
    c.paint({ lead: lead(ASTRA, 'main') });
    c.fact().click();
    await flush();
    await c.take(`route|${GLM}`);
    await flush();
    assert.deepEqual(c.calls, [['t1', { action: 'set', model: GLM, use_local: false }]]);
    assert.deepEqual(c.toasts, [[APPLIED_WORDS.next_step, 'ok']]);
    assert.equal(c.popup(), null, 'the act closes its popover');
    assert.equal(c.said('name'), 'OpenRouter · glm-5.3', 'the answer\'s own fact stands on the card');
    assert.equal(c.said('why'), 'you chose it');

    c.reply.value = { ok: true, task_id: 't1', applied: 'at_start', lead: lead(GLM, 'owner', { pending: route(ASTRA) }) };
    c.fact().click();
    await flush();
    await c.take(`route|${ASTRA}`);
    await flush();
    assert.deepEqual(c.calls[1], ['t1', { action: 'follow' }], 'Main\'s model is Main: the work follows it');
    assert.deepEqual(c.toasts[1], [APPLIED_WORDS.at_start, 'ok']);
    assert.equal(c.said('pending'), '→ gpt-6-astra at the next step');

    c.fact().click();
    await flush();
    await c.pressAct('follow');
    await flush();
    assert.deepEqual(c.calls[2], ['t1', { action: 'follow' }]);
});

test('a refusal says the server\'s words and leaves the popover and the fact as they were', async () => {
    const c = card({ answer: new Error('This work has already finished.') });
    c.paint({ lead: lead(GLM, 'owner') });
    c.fact().click();
    await flush();
    await c.pressAct('follow');
    await flush();
    assert.deepEqual(c.toasts, [['The model was not changed: This work has already finished.', 'error']]);
    assert.ok(c.popup(), 'still open: another model can be picked');
    assert.equal(c.said('name'), 'OpenRouter · glm-5.3');
    closeCardModel();
});

test('a fallback\'s popover offers Return, Wait and Keep, each its own act', async () => {
    const c = card();
    const fell = lead(OPUS, 'fallback', { role: 'fallback:0', primary: lead(GLM, 'owner'), since: 5, cause: 'glm-5.3 did not answer after 3 tries' });
    for (const [action, toast] of [['return', APPLIED_WORDS.next_step], ['wait', 'Waits for glm-5.3: no paid calls until it answers.'],
        ['keep', 'Stays on claude-opus-5.5: your choice from now on.']]) {
        c.paint({ lead: fell, since: '02:14' });
        c.fact().click();
        await flush();
        assert.equal(c.popup().querySelector('.card-model-pop-title').textContent, 'Left glm-5.3 at 02:14 · glm-5.3 did not answer after 3 tries');
        assert.deepEqual(c.acts(), [['return', 'Return to glm-5.3', 'at the next step'],
            ['wait', 'Wait for glm-5.3', 'no paid calls until it answers'], ['keep', 'Keep on claude-opus-5.5', 'your choice from now on']]);
        await c.pressAct(action);
        await flush();
        assert.deepEqual(c.calls.at(-1), ['t1', { action }]);
        assert.deepEqual(c.toasts.at(-1), [toast, 'ok']);
    }
    c.paint({ lead: { ...fell, primary: lead(ASTRA, 'main') }, since: '02:14' });
    c.fact().click();
    await flush();
    assert.deepEqual(c.acts()[0], ['return', 'Return to Main', 'gpt-6-astra · at the next step'], 'a work that followed Main returns to Main');
    closeCardModel();
});

test('an open popover follows its work\'s fact and closes with the control, with Escape or with a press outside', async () => {
    const c = card();
    c.paint({ lead: lead(GLM, 'owner') });
    c.fact().click();
    await flush();
    c.paint({ lead: lead(ASTRA, 'main') });
    assert.deepEqual(c.acts(), [], 'the work follows Main now: nothing to return to');
    c.paint({ lead: lead(ASTRA, 'main'), target: '' });
    assert.equal(c.popup(), null, 'the control is gone: so is its popover');

    c.paint({ lead: lead(GLM, 'owner') });
    c.fact().click();
    await flush();
    const event = { key: 'Escape', defaultPrevented: false, preventDefault() { this.defaultPrevented = true; }, stopPropagation() {} };
    await c.fire(c.popup().querySelector('.chat-model-query'), 'keydown', event);
    assert.equal(c.popup(), null);
    assert.ok(c.doc.activeElement === c.fact(), 'focus returns to the fact');

    c.fact().click();
    await flush();
    c.doc.fire('pointerdown', { target: c.meta });
    assert.equal(c.popup(), null, 'a press outside closes it');
    assert.deepEqual([c.doc.count('pointerdown'), c.doc.count('keydown')], [0, 0], 'and takes its listeners with it');

    // Under a finger nothing in the popover holds focus: Escape closes it from the page too.
    c.fact().click();
    await flush();
    c.doc.fire('keydown', { key: 'Escape', target: c.meta, defaultPrevented: false, preventDefault() {}, stopPropagation() {} });
    assert.equal(c.popup(), null);
});

/* --- from one state body to the card (work_status.js applyCardWorkViews) --- */

const work = (extra = {}) => ({ work_id: 'w1', room: 42, name: 'Fix the parser', kind: 'task', project_id: 'p42',
    episode: { task_id: 't1', live: true }, activity: 'working', transition: 'none', call: '', because: [], tails: [], live: true,
    turn: [], release: 'none', members: { ids: ['t1'] }, controls: [], since: 1, confirmed: true, mark: 'moving', ...extra });
const body = (extra = {}) => ({ supervisor_ready: true, works_complete: true, works: [], rooms: {}, main_route: route(ASTRA), ...extra });
function liveCard(t) {
    const doc = stubDocument();
    doc.body = doc.createElement('body');
    // The line's own writer (chat_activity.js) escapes through the page's document.
    const prior = globalThis.document;
    globalThis.document = doc;
    t.after(() => { globalThis.document = prior; });
    const meta = doc.createElement('div');
    meta.className = 'chat-live-meta';
    const record = { groupId: 't1', metaEl: meta, root: { dataset: {} }, agentModel: 'z-ai/glm-5.3', toolCalls: 3 };
    const read = (data, prior = emptyWorkState()) => { const state = reconcileWorkState(prior, data, 42); applyCardWorkViews(new Map([['t1', record]]), state); return state; };
    const part = () => meta.children.find((node) => node.classList.contains('card-model')) || null;
    const name = () => part().querySelector('.card-model-name').children.map((child) => child.textContent).join('').replaceAll('\u00a0', ' ');
    return { record, meta, read, part, name };
}

test('a card shows its work\'s published fact, pressable only with the offered control; without a fact it shows what it shows today', (t) => {
    const c = liveCard(t);
    c.read(body({ works: [work()] }));
    assert.equal(c.part(), null, 'a work that publishes no fact');
    assert.ok(c.record.metaEl === c.meta);

    c.read(body({ works: [work({ lead: lead(GLM, 'room') })] }));
    assert.equal(c.name(), 'OpenRouter · glm-5.3');
    assert.equal(c.part().querySelector('.card-model-fact').tagName, 'SPAN', 'the server offered no control');
    assert.doesNotMatch(c.record.metaEl.innerHTML, /glm-5\.3/, 'the line no longer names the model itself');
    assert.match(c.record.metaEl.innerHTML, /3 tool calls/);

    const state = c.read(body({ works: [work({ lead: lead(GLM, 'room'), controls: [{ action: 'model', target: 't1' }] })] }));
    assert.equal(c.part().querySelector('.card-model-fact').tagName, 'BUTTON');
    assert.deepEqual(state.main, route(ASTRA));

    // An incomplete body that omits the work keeps it unconfirmed: its fact stays, its control does not.
    c.read({ supervisor_ready: true, works_complete: false, works: [], rooms: {} }, state);
    assert.equal(c.part().querySelector('.card-model-fact').tagName, 'SPAN');

    c.read(body({ works: [work()] }));
    assert.equal(c.part(), null);
    assert.match(c.meta.innerHTML, /glm-5\.3/, 'the line names the model again, as today');
});

test('a direct turn\'s card takes its fact from its activity row; a Main portal shows none', (t) => {
    const c = liveCard(t);
    const direct = body({ active_chat_activities: [{ activity_id: 't1', chat_id: 42, lead: lead(ASTRA, 'main') }] });
    const state = c.read(direct);
    assert.equal(state.leads.get('t1').model, ASTRA);
    assert.equal(c.name(), 'Codex · gpt-6-astra');
    assert.equal(c.part().querySelector('.card-model-fact').tagName, 'SPAN', 'no work, no control');
    c.record.root.dataset.portal = '1';
    c.read(direct);
    assert.equal(c.part(), null);
    assert.deepEqual(reconcileWorkState(state, { ...direct, main_route: undefined }, 42).main, route(ASTRA), 'a body that does not say Main keeps the last one said');
});
