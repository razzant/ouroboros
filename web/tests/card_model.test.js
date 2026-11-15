// A work's model on its card (card_model.js; DESIGN "A work's model on its card"): the first fact of the card's meta
// line in a node of its own, said from the server's published fact (CLIENT_CONTRACT §5), pressable only where the server
// offers the work's `model` control, amber while the work runs on a fallback, with the work's processing tier beside it
// when that is not Standard, and one POST per act (CLIENT_CONTRACT §5). The control opens the Model sheet the composer
// opens (model_sheet.js), this work's own words and acts above its field and its tier at its foot. Fixtures are in the
// contract's shapes; geometry and the real page are covered by the browser tests.
import assert from 'node:assert/strict';
import test from 'node:test';
import { APPLIED_WORDS, basisWords, causeWords, closeCardModel, leadView, openCardModel, paintCardModel, refusalWords } from '../modules/card_model.js';
import { createModelCatalogStore } from '../modules/model_catalog_store.js';
import { applyCardWorkViews, emptyWorkState, reconcileWorkState } from '../modules/work_status.js';
import { modelSheetOutline, modelSheetRows, stubDocument } from './composer_controls_fixture.js';

const ASTRA = 'claudexor::codex=gpt-6-astra';
const OPUS = 'anthropic::claude-opus-5.5';
const GLM = 'z-ai/glm-5.3';
const route = (model, local = false) => ({ model, use_local: local });
const lead = (model, basis, extra = {}) => ({ ...route(model), basis, ...extra });
const STANDARD = { main: '', default: 'standard' };
const CATALOG = {
    model_sources: [{ id: 'codex', label: 'Codex', credentialHarness: 'codex' }],
    items: [{ provider_id: 'claudexor', provider: 'Codex', source_id: 'codex', id: 'gpt-6-astra', name: 'gpt-6-astra',
        value: ASTRA, credential_profile_id: 'home', availability: 'available' },
    { provider_id: 'openrouter', provider: 'OpenRouter', id: GLM, name: GLM, value: GLM },
    { provider_id: 'openrouter', provider: 'OpenRouter', id: 'moonshotai/kimi-k3', name: 'moonshotai/kimi-k3', value: 'moonshotai/kimi-k3' }],
};
const flush = async () => { for (let i = 0; i < 4; i += 1) await new Promise((resolve) => setImmediate(resolve)); };

/* One card's meta line as chat_activity.js leaves it, with everything the fact's acts reach injected. */
function card({ recents = [route(GLM)], pins = [], answer = null, catalog = CATALOG } = {}) {
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
    const ids = [];  // each act's request id, in order
    const toasts = [];
    const rerenders = [];
    const reply = { value: answer };
    const pinned = { routes: pins, writes: [] };
    const store = createModelCatalogStore({ readCatalog: async () => catalog,
        readPreferences: async () => ({ model_recents: recents, model_pins: pinned.routes }), readSettings: async () => ({ _meta: { connections: [] } }),
        savePreferences: async (body) => {
            pinned.writes.push(body);
            const { pin = [], unpin = [] } = body.model_pins;
            pinned.routes = [...pinned.routes.filter((seen) => !unpin.some((gone) => gone.model === seen.model)), ...pin];
            return { model_pins: pinned.routes };
        } });
    const deps = {
        store, coarse: () => false, place: () => ({ reposition() {}, destroy() {} }),
        toast: (message, tone) => toasts.push([message, tone]),
        rerender: (seen) => rerenders.push(seen),
        act: async (taskId, requestId, act) => {
            calls.push([taskId, act]);
            ids.push(requestId);
            assert.ok(requestId, 'every act carries a request id');
            const said = await reply.value;
            if (said instanceof Error) throw said;
            if (said) return said;
            // The endpoint's own answers: a tier act's is the work's tier, every other act's its new fact.
            return act.action === 'processing' ? { ok: true, task_id: taskId, applied: 'next_step', processing: { choice: act.processing, source: 'owner' } }
                : { ok: true, task_id: taskId, applied: 'next_step', lead: lead(act.model || ASTRA, act.action === 'follow' ? 'main' : 'owner') };
        },
    };
    const paint = (view) => paintCardModel(record, view && { target: 't1', main: route(ASTRA), since: '', queued: false, ...view }, deps);
    const part = () => meta.children.find((node) => node.classList.contains('card-model')) || null;
    const fact = () => part()?.querySelector('.card-model-fact') || null;
    // A part's words as a reader gets them; the name is its three spans, joined by their own (no-break) separators.
    const said = (name) => {
        const node = part().querySelector(`.card-model-${name}`);
        if (node.hidden) return '';
        return (name === 'name' ? node.children.map((child) => child.textContent).join('') : node.querySelector('.card-model-words').textContent).replaceAll('\u00a0', ' ');
    };
    const popup = () => doc.body.children.find((node) => node.classList.contains('model-sheet')) || null;
    const fire = (node, type, event) => Promise.all((node.listeners.get(type) || []).map((fn) => fn(event)));
    const acts = () => popup().querySelector('.card-model-acts').children
        .map((node) => [node.dataset.act, node.children[0].textContent, node.children[1].textContent]);
    const pressAct = (name) => {
        const box = popup().querySelector('.card-model-acts');
        return fire(box, 'click', { target: { closest: () => box.children.find((node) => node.dataset.act === name) } });
    };
    const list = () => popup().querySelector('.model-sheet-list');
    const rows = () => modelSheetRows(list().innerHTML);
    const rowNode = (key) => list().querySelectorAll('.chat-model-row').find((node) => node.dataset.row === key);
    // A press on a row's first cell, and one on its Pin (the list asks the press's target what it is in).
    const take = (key) => fire(list(), 'click', { target: { closest: (selector) => (selector === '.chat-model-row' ? rowNode(key) : null) } });
    const pin = (key) => fire(list(), 'click', { stopPropagation() {}, target: { closest: (selector) => (selector === '.chat-model-row' ? rowNode(key)
        : selector === '.chat-model-pin' ? list().querySelectorAll('.chat-model-pin').find((node) => node.dataset.pin === key) : null) } });
    // The tier strip at the sheet's foot: a press on a tier ('' is Default), and the strip as it is painted.
    const strip = () => popup().querySelector('.model-tier');
    const segs = () => strip().querySelectorAll('.model-tier-seg');
    const chooseTier = (tier) => fire(strip(), 'click', { target: { closest: (selector) => (selector !== '[data-tier]' ? null
        : tier === '' ? strip().querySelector('.model-tier-default') : segs().find((seg) => seg.dataset.tier === tier)) } });
    const tierView = () => ({
        checked: segs().filter((seg) => seg.getAttribute('aria-checked') === 'true').map((seg) => seg.dataset.tier),
        ring: segs().filter((seg) => seg.dataset.resolved === 'true').map((seg) => seg.dataset.tier),
        busy: segs().every((seg) => seg.getAttribute('aria-disabled') === 'true'),
        reset: !strip().querySelector('.model-tier-default').hidden,
        scope: strip().querySelector('.model-tier-scope').textContent,
        means: strip().querySelector('.model-tier-means').textContent,
    });
    return { doc, meta, record, tools, calls, ids, toasts, rerenders, reply, store, pinned, paint, part, fact, said, popup, acts, pressAct, list,
        rows, outline: () => modelSheetOutline(list().innerHTML), take, pin, chooseTier, tierView, fire };
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
    const fell = lead(OPUS, 'fallback', { role: 'fallback:1', primary: lead(ASTRA, 'owner'), since: '2026-11-11T02:14:00Z', cause: 'subscription_window_exhausted' });
    const said = leadView(fell, { since: '02:14' });
    assert.equal(said.name, 'fallback from Settings · claude-opus-5.5 · since 02:14');
    assert.equal(said.why, 'left gpt-6-astra');
    assert.equal(said.fallback, true);
    assert.equal(said.cause, 'Waiting for quota', 'the wait card\'s own words for that condition');
    assert.equal(leadView({ ...fell, primary: lead(ASTRA, 'main') }, { since: '02:14' }).why, 'left Main (gpt-6-astra)');
    assert.equal(leadView(fell).name, 'fallback from Settings · claude-opus-5.5', 'no time is said when the fact gives none');
    assert.doesNotMatch(said.name + said.why, /\$|\d+\.\d\d/, 'no money on the amber line');
});

test('cited words keep their exact text and published clipping fact, including away from the primary', () => {
    const cited = { text: '  Use GPT-6-astra\nfor <THIS> work 🙂  ', clipped: false };
    assert.equal(leadView(lead(ASTRA, 'owner_words')).cited, null);
    assert.deepEqual(leadView(lead(ASTRA, 'owner_words', { cited })).cited, cited);
    assert.deepEqual(leadView(lead(ASTRA, 'owner_words', { cited: { ...cited, clipped: true } })).cited,
        { ...cited, clipped: true });
    assert.deepEqual(leadView(lead(OPUS, 'fallback', { role: 'fallback:0', primary: lead(ASTRA, 'owner_words', { cited }) })).cited, cited);
});

test('a wait-card move names its unchanged primary in ordinary ink, with no inferred pending route', async () => {
    const moved = lead(OPUS, 'owner', { primary: lead(ASTRA, 'main'), cause: 'wait_card', since: '2026-11-11T02:14:00Z' });
    const said = leadView(moved);
    assert.equal(said.fallback, false);
    assert.equal(said.name, 'Anthropic · claude-opus-5.5');
    assert.equal(said.why, 'you chose it · left Main (gpt-6-astra)');
    assert.equal(said.pending, '');
    assert.equal(leadView({ ...moved, primary: lead(GLM, 'room') }).why, 'you chose it · left glm-5.3');
    const c = card();
    c.paint({ lead: moved });
    assert.equal(c.part().dataset.basis, 'own');
    assert.equal(c.fact().classList.contains('model-fallback'), false);
    assert.equal(c.said('pending'), '');
    c.fact().click();
    await flush();
    assert.equal(c.popup().querySelector('.card-model-pop-words').textContent,
        'Model for this work · you chose it · left Main (gpt-6-astra)');
    assert.deepEqual(c.acts(), []);
    assert.deepEqual([c.rows()[0].key, c.rows()[0].name, c.rows()[0].facts], ['main', 'Main', 'this work will follow Main']);
    c.paint({ lead: { ...moved, pending: route(GLM) } });
    assert.equal(c.said('pending'), '→ glm-5.3 at the next step', 'a published pending route is still shown');
    assert.equal(c.said('why'), 'you chose it · left Main (gpt-6-astra)', 'a different pending route does not hide the primary left');
    closeCardModel();
});

test('an accepted cited primary awaiting its first call is named as pending, not as a past departure', async () => {
    const c = card();
    const cited = { text: 'Use glm-5.3 next.', clipped: false };
    const primary = lead(GLM, 'owner_words', { cited });
    const moving = lead(ASTRA, 'main', { primary, pending: route(GLM) });
    assert.equal(leadView(moving).awaitingPrimary, true);
    assert.equal(leadView(moving).why, '');
    c.paint({ lead: moving });
    c.fact().click();
    await flush();
    assert.equal(c.popup().querySelector('.card-model-pop-words').textContent,
        'Model for this work · follows Main · → glm-5.3 at the next step · you asked for it');
    assert.equal(c.popup().querySelector('.card-model-cited').textContent, cited.text);
    c.paint({ lead: lead(OPUS, 'fallback', { role: 'fallback:0', primary, pending: route(GLM), cause: 'auth_error' }), since: '02:14' });
    assert.equal(c.said('why'), '');
    assert.equal(c.popup().querySelector('.card-model-pop-words').textContent,
        'fallback from Settings · claude-opus-5.5 · since 02:14 · → glm-5.3 at the next step · Sign-in required · you asked for it');
    assert.equal(c.popup().querySelector('.card-model-cited').textContent, cited.text);
    closeCardModel();
});

test('the popover quotes authored words intact, reuses Show all, and labels a published excerpt', async () => {
    const c = card();
    const cited = { text: '  Use GPT-6-astra\nfor <THIS> work 🙂  ', clipped: true };
    const view = { lead: lead(ASTRA, 'owner_words', { cited }) };
    c.paint(view);
    c.fact().click();
    await flush();
    const quote = c.popup().querySelector('.card-model-cited');
    const line = c.popup().querySelector('.card-model-pop-line');
    const toggle = c.popup().querySelector('.chat-clamp-toggle');
    assert.equal(c.popup().querySelector('.card-model-pop-words').textContent, 'Model for this work · you asked for it');
    assert.equal(line.children[1].textContent, ' “');
    assert.equal(line.children.at(-1).textContent, '”');
    assert.equal(quote.dataset.i18nAuthored, '');
    assert.equal(quote.textContent, cited.text);
    assert.equal(quote.children.length, 0, 'the owner\'s angle brackets never become markup');
    assert.equal(c.popup().querySelector('.card-model-cited-note').textContent, 'Excerpt; the original quotation is longer.');
    line.scrollHeight = 500;
    line.clientHeight = 80;
    toggle.click();
    assert.equal(line.dataset.expanded, '1');
    assert.equal(toggle.textContent, 'Show less');
    c.paint(view);
    assert.equal(c.popup().querySelector('.card-model-cited'), quote, 'an unchanged read preserves the authored node and expansion');
    assert.equal(line.dataset.expanded, '1');
    line.scrollTop = 300;
    toggle.click();
    assert.equal(line.dataset.expanded, undefined);
    assert.equal(line.scrollTop, 0, 'Show less returns to the start of the quotation');
    assert.equal(toggle.textContent, 'Show all');
    assert.equal(toggle.hidden, false);
    assert.equal(quote.textContent, cited.text, 'neither expansion nor collapse rewrites a byte');

    c.paint({ lead: lead(OPUS, 'fallback', { role: 'fallback:0', primary: lead(ASTRA, 'owner_words', { cited: { ...cited, clipped: false } }) }) });
    assert.equal(c.popup().querySelector('.card-model-pop-words').textContent, 'Left gpt-6-astra · you asked for it');
    assert.equal(c.popup().querySelector('.card-model-cited').textContent, cited.text);
    assert.equal(c.popup().querySelector('.card-model-cited-note'), null, 'a whole published span has no excerpt label');
    assert.equal(c.acts().length, 3);
    c.paint({ lead: lead(GLM, 'owner') });
    assert.equal(c.popup().querySelector('.card-model-cited'), null);
    assert.equal(c.popup().querySelector('.chat-clamp-toggle'), null, 'no empty quotation or toggle without a cite');
    closeCardModel();
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

test('the meta separator stays with the last visible model fact across repaints', () => {
    const c = card();
    const separator = () => c.part().querySelector('.chat-live-meta-sep');
    c.paint({ lead: lead(ASTRA, 'main') });
    assert.equal(separator().parentNode, c.fact().parentNode, 'the model and dot share a group');
    c.paint({ lead: lead(GLM, 'owner') });
    assert.equal(separator().parentNode, c.part().querySelector('.card-model-why'));
    c.paint({ lead: lead(GLM, 'owner', { pending: route(ASTRA) }) });
    assert.equal(separator().parentNode, c.part().querySelector('.card-model-pending'));
    const words = c.part().querySelector('.card-model-pending').children[0];
    c.paint({ lead: lead(GLM, 'owner', { pending: route(ASTRA) }) });
    assert.equal(c.part().querySelector('.card-model-pending').children[0], words, 'unchanged words stay mounted');
    c.paint({ lead: lead(ASTRA, 'main') });
    assert.equal(separator().parentNode, c.fact().parentNode);
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

test('the sheet is the composer\'s Model sheet; its Main row is the one way to follow Main and says so', async () => {
    const c = card();
    c.paint({ lead: lead(GLM, 'owner') });
    c.fact().click();
    await flush();
    assert.equal(c.fact().getAttribute('aria-expanded'), 'true');
    assert.deepEqual([c.popup().getAttribute('role'), c.popup().getAttribute('aria-label'), c.popup().parentNode === c.doc.body],
        ['dialog', 'Model for this work', true]);
    assert.deepEqual(c.popup().children.map((node) => node.className), ['card-model-head', 'chat-model-search', 'model-sheet-list', 'model-tier'],
        'this work\'s own words and acts above the field, its tier at the foot');
    assert.equal(c.popup().querySelector('.card-model-pop-words').textContent, 'Model for this work · you chose it');
    assert.deepEqual(c.acts(), [], 'no separate action: the list\'s Main row is the way');
    assert.equal(c.popup().querySelector('.card-model-acts').hidden, true);
    assert.deepEqual(c.outline(), ['main', '§ Recent', `route|${GLM} * -pin`, 'sources'], 'the work\'s own model is the chosen row');
    assert.deepEqual([c.rows()[0].name, c.rows()[0].facts, c.rows()[0].pin], ['Main', 'this work will follow Main', null]);
    assert.equal(c.rows()[1].facts, 'OpenRouter · paid by key', 'any other row keeps its route\'s facts');
    assert.equal(c.tierView().scope, 'Default · Standard · for this work from its next call and its new subagents without a tier of their own; the running call and started subagents keep theirs');
    assert.ok(c.doc.activeElement === c.popup().querySelector('.chat-model-query'), 'a keyboard types at once');
    c.fact().click();
    assert.equal(c.popup(), null, 'a second press closes it');
    assert.equal(c.fact().getAttribute('aria-expanded'), 'false');

    c.paint({ lead: lead(ASTRA, 'main'), queued: true });
    c.fact().click();
    await flush();
    assert.deepEqual(c.acts(), []);
    assert.equal(c.popup().querySelector('.card-model-pop-words').textContent, 'Model for this work · follows Main');
    assert.deepEqual([c.rows()[0].selected, c.rows()[0].facts], [true, 'this work follows Main'], 'Main\'s row is its own');
    assert.equal(c.tierView().scope, 'Default · Standard · for this work when it starts, and its new subagents without a tier of their own', 'a queued work says when a change lands');
    // A follower whose next step is another route has not arrived yet: the row still says what a press does.
    c.paint({ lead: lead(GLM, 'main', { pending: route(ASTRA) }) });
    assert.deepEqual([c.rows()[0].selected, c.rows()[0].facts], [false, 'this work will follow Main']);
    closeCardModel();
    assert.equal(c.popup(), null);
});

test('a pick sets this work\'s model, Main\'s row follows Main, and the answer says when it applies', async () => {
    const c = card({ recents: [route(GLM), route(ASTRA)] });
    c.paint({ lead: lead(ASTRA, 'main') });
    c.fact().click();
    await flush();
    await c.take(`route|${GLM}`);
    await flush();
    assert.deepEqual(c.calls, [['t1', { action: 'set', model: GLM, use_local: false }]]);
    assert.deepEqual(c.toasts, [[APPLIED_WORDS.next_step, 'ok']]);
    assert.equal(c.popup(), null, 'the act closes its sheet');
    assert.equal(c.said('name'), 'OpenRouter · glm-5.3', 'the answer\'s own fact stands on the card');
    assert.equal(c.said('why'), 'you chose it');

    c.reply.value = { ok: true, task_id: 't1', applied: 'at_start', lead: lead(GLM, 'owner', { pending: route(ASTRA) }) };
    c.fact().click();
    await flush();
    await c.take('main');
    await flush();
    assert.deepEqual(c.calls[1], ['t1', { action: 'follow' }], 'Main\'s row: the work follows Main');
    assert.deepEqual(c.toasts[1], [APPLIED_WORDS.at_start, 'ok']);
    assert.equal(c.said('pending'), '→ gpt-6-astra at the next step');
    assert.equal(c.calls.length, 2, 'one write per press');

    // Main's model as a row is that model: it becomes this work's own, never a follow of Main.
    c.reply.value = null;
    c.paint({ lead: lead(ASTRA, 'main') });
    c.fact().click();
    await flush();
    await c.take(`route|${ASTRA}`);
    await flush();
    assert.deepEqual(c.calls[2], ['t1', { action: 'set', model: ASTRA, use_local: false }]);
    assert.equal(c.said('why'), 'you chose it');

    // A work that already follows Main: its Main row is no act; a work's own model again is none either.
    c.paint({ lead: lead(ASTRA, 'main') });
    c.fact().click();
    await flush();
    await c.take('main');
    await flush();
    assert.deepEqual([c.calls.length, c.popup()], [3, null], 'nothing is sent; the sheet closes');
    c.paint({ lead: lead(GLM, 'owner') });
    c.fact().click();
    await flush();
    await c.take(`route|${GLM}`);
    await flush();
    assert.deepEqual([c.calls.length, c.popup()], [3, null]);
});

test('a pending switch is this work\'s choice in its sheet, and the route still serving cancels it before the next step', async () => {
    const c = card({ recents: [route(GLM), route(OPUS)] });
    const title = () => c.popup().querySelector('.card-model-pop-words').textContent;
    // Model A (GLM) serves the sent call; the owner chose B (Astra), which lands at the next step.
    c.paint({ lead: lead(GLM, 'owner', { pending: route(ASTRA) }) });
    c.fact().click();
    await flush();
    assert.equal(title(), 'Model for this work · you chose it · → gpt-6-astra at the next step');
    assert.deepEqual(c.outline().slice(0, 4), ['main', '§ Recent', `route|${GLM} -pin`, `route|${OPUS} -pin`], 'A is no longer the chosen row');
    assert.equal(c.rows()[0].facts, 'this work will follow Main');
    c.reply.value = { ok: true, task_id: 't1', applied: 'next_step', lead: lead(GLM, 'owner') };
    await c.take(`route|${GLM}`);
    await flush();
    assert.deepEqual(c.calls, [['t1', { action: 'set', model: GLM, use_local: false }]], 'A again is an act: it cancels B');
    assert.equal(c.said('pending'), '', 'the card stays on A');
    // A work that follows Main with a chosen B pending: Main's row is the cancel, B the chosen row.
    c.paint({ lead: lead(GLM, 'main', { pending: route(OPUS) }) });
    c.fact().click();
    await flush();
    assert.deepEqual([c.rows()[0].selected, c.outline().includes(`route|${OPUS} * -pin`)], [false, true]);
    c.reply.value = { ok: true, task_id: 't1', applied: 'next_step', lead: lead(GLM, 'main') };
    await c.take('main');
    await flush();
    assert.deepEqual(c.calls[1], ['t1', { action: 'follow' }]);
    // A work on its own A asked to follow Main (Main is Astra): A cancels that too.
    c.paint({ lead: lead(GLM, 'owner', { pending: route(ASTRA) }) });
    c.fact().click();
    await flush();
    let answer;
    c.reply.value = new Promise((resolve) => { answer = resolve; });
    const press = c.take(`route|${GLM}`);
    await flush();
    // A stale answer: the switch landed in a newer body while the press was on its way; the card keeps that body.
    c.paint({ lead: lead(ASTRA, 'owner') });
    answer({ ok: true, task_id: 't1', applied: 'next_step', lead: lead(GLM, 'owner') });
    await press;
    await flush();
    assert.deepEqual(c.calls[2], ['t1', { action: 'set', model: GLM, use_local: false }]);
    assert.equal(c.said('name'), 'Codex · gpt-6-astra', 'the newer body stands, not the stale answer');
    // A withdrawn control takes the open sheet with it; nothing is sent from it.
    c.paint({ lead: lead(GLM, 'owner', { pending: route(ASTRA) }) });
    c.fact().click();
    await flush();
    assert.ok(c.popup());
    c.paint({ lead: lead(GLM, 'owner', { pending: route(ASTRA) }), target: null });
    assert.equal(c.popup(), null, 'the control went: so did its sheet');
    assert.equal(c.calls.length, 3);
});

test('a Pin in a work\'s sheet acts on no work: it pins the model for every sheet and leaves this one open', async () => {
    const c = card();
    c.paint({ lead: lead(ASTRA, 'main') });
    c.fact().click();
    await flush();
    await c.pin(`route|${GLM}`);
    await flush();
    assert.deepEqual([c.pinned.writes, c.calls, c.toasts], [[{ model_pins: { pin: [route(GLM)] } }], [], []]);
    assert.ok(c.popup(), 'still open');
    assert.deepEqual(c.outline(), ['main *', '§ Pinned', `route|${GLM} +pin`, 'sources']);
    closeCardModel();
});

test('a refusal is said in the sheet in the owner\'s words; the fact and the sheet stay as they were', async () => {
    const refused = (code, status, message = code.replaceAll('_', ' ')) => Object.assign(new Error(message), { status, body: { error: message, reason_code: code, task_id: 't1' } });
    const c = card({ answer: refused('task_not_live', 404) });
    const status = () => c.popup().querySelector('.card-model-refusal');
    const words = () => (status().hidden ? '' : status().children[0].textContent);
    c.paint({ lead: lead(GLM, 'owner') });
    c.fact().click();
    await flush();
    assert.equal(status().hidden, true, 'nothing is said before an act is refused');
    await c.take('main');
    await flush();
    assert.deepEqual([words(), status().children[1].hidden], ['This work has already ended.', true], 'a decided refusal offers no retry');
    assert.deepEqual(c.toasts, [], 'said on the card, not beside it');
    assert.ok(c.popup(), 'still open: another model can be picked');
    assert.equal(c.said('name'), 'OpenRouter · glm-5.3');

    // A 503: the decision is saved; Retry sends the same request id again.
    c.reply.value = refused('model_wake_failed', 503);
    await c.take('main');
    await flush();
    assert.deepEqual([words(), status().children[1].hidden, status().children[1].textContent],
        ['Your choice is saved, but the work could not be told yet.', false, 'Retry']);
    c.reply.value = { ok: true, task_id: 't1', applied: 'next_step', lead: lead(GLM, 'main', { pending: route(ASTRA) }) };
    await c.fire(status().children[1], 'click', {});
    await flush();
    assert.deepEqual(c.calls.slice(-2), [['t1', { action: 'follow' }], ['t1', { action: 'follow' }]]);
    assert.equal(c.ids.length, 3);
    assert.ok(c.ids[1] === c.ids[2] && c.ids[0] !== c.ids[1], 'the retry carries the refused request\'s id; another press has its own');
    assert.equal(c.popup(), null, 'accepted: the sheet closes');
    assert.deepEqual(c.toasts.at(-1), [APPLIED_WORDS.next_step, 'ok']);
});

test('every refusal the endpoint names has words of its own; an unknown one keeps the server\'s', () => {
    const refused = (code, status, message) => Object.assign(new Error(message || code), { status, body: { reason_code: code } });
    const said = Object.fromEntries(['task_model_invalid', 'request_id_required', 'task_not_live', 'presence_model_unsupported',
        'not_a_root_task', 'model_control_unavailable', 'cancel_pending', 'model_action_state', 'model_recents_pending',
        'model_queue_publish_failed', 'model_wake_failed'].map((code) => [code, refusalWords(refused(code, code.startsWith('model_') && code !== 'model_action_state' && code !== 'model_control_unavailable' ? 503 : 409))]));
    for (const [code, { words }] of Object.entries(said)) {
        assert.ok(words && !words.includes('_') && words !== code.replaceAll('_', ' '), `${code} is said in words: ${words}`);
    }
    assert.equal(new Set(Object.values(said).map((entry) => entry.words)).size, 11, 'each its own');
    assert.deepEqual(Object.entries(said).filter(([, entry]) => entry.retry).map(([code]) => code),
        ['model_recents_pending', 'model_queue_publish_failed', 'model_wake_failed'], 'only a saved decision offers the retry');
    assert.deepEqual(refusalWords(refused('migrating', 503, 'the task store is importing; retry shortly')),
        { words: 'the task store is importing; retry shortly', retry: true });
    assert.deepEqual(refusalWords(new Error('Network unreachable')), { words: 'Network unreachable', retry: false });
});

test('a cause is said in the wait card\'s words where it has them, as its code where it has none, and not twice', () => {
    assert.equal(causeWords('subscription_window_exhausted'), 'Waiting for quota');
    assert.equal(causeWords('quota_exhausted'), 'Waiting for quota');
    assert.equal(causeWords('auth_error'), 'Sign-in required');
    assert.equal(causeWords('connection_unavailable'), 'Key unavailable');
    assert.equal(causeWords('provider_transient'), 'provider_transient', 'a code this page has no words for is shown as it is');
    assert.equal(causeWords('configured fallback'), '', 'the amber line already says "fallback from Settings"');
    assert.equal(causeWords('switch_model'), '', 'the card already says "Ouroboros switched to it"');
    assert.equal(causeWords(undefined), '');
});

test('a fallback\'s popover offers Return, Wait and Keep, each its own act', async () => {
    const c = card();
    const fell = lead(OPUS, 'fallback', { role: 'fallback:0', primary: lead(GLM, 'owner'), since: '2026-11-11T02:14:00Z', cause: 'auth_error' });
    for (const [action, toast] of [['return', APPLIED_WORDS.next_step], ['wait', 'Waits for glm-5.3: no paid calls until it answers.'],
        ['keep', 'Stays on claude-opus-5.5: your choice from now on.']]) {
        c.paint({ lead: fell, since: '02:14' });
        c.fact().click();
        await flush();
        assert.equal(c.popup().querySelector('.card-model-pop-words').textContent, 'Left glm-5.3 at 02:14 · Sign-in required');
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

test('an open sheet follows its work\'s fact and closes with the control, with Escape or with a press outside', async () => {
    const c = card();
    c.paint({ lead: lead(GLM, 'owner') });
    c.fact().click();
    await flush();
    c.paint({ lead: lead(ASTRA, 'main') });
    assert.deepEqual(c.acts(), [], 'the work follows Main now: nothing to return to');
    c.paint({ lead: lead(ASTRA, 'main'), target: '' });
    assert.equal(c.popup(), null, 'the control is gone: so is its sheet');

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

    // Under a finger nothing in the sheet holds focus: Escape closes it from the page too.
    c.fact().click();
    await flush();
    c.doc.fire('keydown', { key: 'Escape', target: c.meta, defaultPrevented: false, preventDefault() {}, stopPropagation() {} });
    assert.equal(c.popup(), null);
});

/* --- the work's processing tier: beside its model on the card, and at the foot of its sheet --- */

test('beside its model the fact names the work\'s tier when it is not Standard, and what its newest call was served as when that differs', () => {
    const c = card();
    const tierFact = (view) => { c.paint(view); return c.said('tier'); };
    assert.equal(tierFact({ lead: lead(ASTRA, 'main'), tiers: { main: 'fast', default: 'standard' } }), 'Fast', 'a follower takes Main\'s tier');
    assert.equal(c.part().querySelector('.chat-live-meta-sep').parentNode, c.part().querySelector('.card-model-tier'),
        'the line\'s dot stays with the last fact');
    assert.equal(tierFact({ lead: lead(GLM, 'owner'), tiers: { main: 'fast', default: 'standard' } }), '', 'its own model runs as a lead on the default tier');
    assert.equal(c.part().querySelector('.chat-live-meta-sep').parentNode, c.part().querySelector('.card-model-why'));
    // After the words of why ("you chose it") the tier follows a real separator.
    assert.equal(tierFact({ lead: lead(GLM, 'owner'), tiers: { main: '', default: 'economy' } }), '· Flex');
    assert.equal(tierFact({ lead: lead(GLM, 'owner'), processing: { choice: 'fast', source: 'owner' }, tiers: STANDARD }), '· Fast', 'its own tier first');
    assert.equal(tierFact({ lead: lead(GLM, 'owner'), processing: { choice: 'ultrafast', source: 'owner' }, tiers: STANDARD }), '· Ultrafast');
    assert.equal(tierFact({ lead: lead(ASTRA, 'main'), processing: { choice: 'standard', source: 'room' }, tiers: { main: 'fast', default: 'fast' } }), '',
        'an explicit Standard is not named');
    assert.equal(tierFact({ lead: lead(GLM, 'owner'), processing: { choice: 'fast', last: { requested: 'fast', submitted: 'fast', observed: 'standard' } }, tiers: STANDARD }),
        '· Fast · Fast asked · served Standard', 'what was served, beside what was chosen');
    assert.equal(tierFact({ lead: lead(GLM, 'owner'), processing: { choice: '', last: { requested: 'fast', submitted: null, observed: 'unknown',
        reason: 'processing_not_submitted' } }, tiers: STANDARD }), '· Fast asked · sent as Standard');
    assert.equal(tierFact({ lead: lead(GLM, 'owner'), processing: { choice: 'fast', last: { requested: 'fast', submitted: 'fast', observed: 'fast' } }, tiers: STANDARD }),
        '· Fast', 'served as asked: nothing more');
    assert.equal(tierFact({ lead: lead(GLM, 'owner') }), '', 'no tier fact and no tiers: nothing');
    // On a fallback a work still asks the tier of what it left (model_slots.task_processing_preference): Main's, when
    // it followed Main.
    const fell = (primary) => lead('moonshotai/kimi-k3', 'fallback', { role: 'fallback:1', primary });
    assert.equal(tierFact({ lead: fell(lead(ASTRA, 'main')), tiers: { main: 'fast', default: 'standard' } }), '· Fast');
    assert.equal(c.record.processingDetail, '', 'no receipt yet: Details say nothing of a last call');
    assert.equal(tierFact({ lead: fell(lead(ASTRA, 'owner')), tiers: { main: 'fast', default: 'standard' } }), '',
        'a work that left its own model takes the default');
});

test('a tier press is this work\'s own act from its next call: the sheet stays open and the answer says when it applies', async () => {
    const c = card();
    c.paint({ lead: lead(GLM, 'owner'), tiers: STANDARD });
    c.fact().click();
    await flush();
    assert.deepEqual(c.tierView(), { checked: [], ring: ['standard'], busy: false, reset: false,
        scope: 'Default · Standard · for this work from its next call and its new subagents without a tier of their own; the running call and started subagents keep theirs',
        means: 'The ordinary service at the catalog price.' });
    let answer;
    c.reply.value = new Promise((resolve) => { answer = resolve; });
    await c.chooseTier('fast');
    assert.deepEqual(c.calls, [['t1', { action: 'processing', processing: 'fast' }]], 'a tier moves nothing else');
    assert.equal(c.ids.length, 1);
    assert.deepEqual([c.tierView().busy, c.popup().getAttribute('aria-busy')], [true, 'true'], 'the strip waits for the answer');
    answer({ ok: true, task_id: 't1', applied: 'next_step', processing: { choice: 'fast', source: 'owner' } });
    await flush();
    assert.ok(c.popup(), 'a tier is no pick: the sheet stays open');
    assert.deepEqual(c.toasts, [['Fast for this work. Switches at the next step; the running call finishes.', 'ok']]);
    assert.deepEqual([c.tierView().checked, c.tierView().ring, c.tierView().busy, c.tierView().reset, c.tierView().scope],
        [['fast'], [], false, true, 'Fast · for this work from its next call and its new subagents without a tier of their own; the running call and started subagents keep theirs']);
    assert.equal(c.popup().hasAttribute('aria-busy'), false);
    assert.deepEqual([c.said('tier'), c.said('name')], ['· Fast', 'OpenRouter · glm-5.3'], 'the answer\'s own tier stands on the card; its model is untouched');
    c.reply.value = { ok: true, task_id: 't1', applied: 'at_start', processing: { choice: '', source: 'owner' } };
    await c.chooseTier('');
    await flush();
    assert.deepEqual(c.calls.at(-1), ['t1', { action: 'processing', processing: '' }], 'Default is a tier act too');
    assert.deepEqual(c.toasts.at(-1), ['Default tier for this work. Applies when it starts.', 'ok']);
    assert.deepEqual([c.tierView().checked, c.tierView().ring, c.said('tier'), Boolean(c.popup())], [[], ['standard'], '', true]);
    closeCardModel();
});

test('the strip says before a press what a tier means for this work\'s route; the sheet\'s head says what the newest call was served as', async () => {
    const c = card();
    c.paint({ lead: lead(GLM, 'owner'), queued: true, tiers: STANDARD,
        processing: { choice: 'economy', source: 'room', last: { requested: 'economy', submitted: 'economy', observed: 'standard', observedNative: ['default'] } } });
    c.fact().click();
    await flush();
    const served = () => { const node = c.popup().querySelector('.card-model-served'); return node.hidden ? '' : node.textContent; };
    // Above the field, so a receipt arriving while the sheet is open never moves the strip at its foot.
    assert.deepEqual([c.tierView().checked, c.tierView().scope, served()],
        [['economy'], 'Flex · for this work when it starts, and its new subagents without a tier of their own', 'Flex asked · served Standard']);
    c.paint({ lead: lead(GLM, 'owner'), queued: true, tiers: STANDARD,
        processing: { choice: 'economy', source: 'room', last: { requested: 'economy', submitted: 'economy', observed: 'economy' } } });
    assert.deepEqual([c.tierView().scope.endsWith('of their own'), served()], [true, ''], 'served as asked: the head says nothing');
    c.paint({ lead: lead(GLM, 'owner'), queued: true, tiers: STANDARD,
        processing: { choice: 'economy', source: 'room', last: { requested: 'economy', submitted: 'economy', observed: 'standard', observedNative: ['default'] } } });
    assert.equal(c.tierView().means, 'Lower price, slower; a provider may be out of capacity for it. Not confirmed for glm-5.3 here: a provider may serve it as Standard.',
        'the catalog does not say this route serves Flex');
    assert.deepEqual(c.popup().querySelector('.model-tier').querySelectorAll('.model-tier-seg').map((seg) => seg.dataset.support), ['unknown', 'listed', 'unknown', 'unknown']);
    assert.equal(c.said('tier'), '· Flex · Flex asked · served Standard');
    // The whole receipt (asked, sent, served) stands in the card's Details, not in the sheet or the meta line.
    assert.equal(c.record.processingDetail, 'Flex · last call: asked Flex · sent Flex · served Standard');
    closeCardModel();
});

test('a work on a named tier holds its receipt line in an open sheet, so the receipt\'s arrival moves nothing', async () => {
    const c = card();
    const node = () => c.popup().querySelector('.card-model-served');
    c.paint({ lead: lead(GLM, 'owner'), tiers: STANDARD, processing: { choice: 'fast', source: 'owner' } });
    c.fact().click();
    await flush();
    const held = node();
    assert.deepEqual([held.hidden, held.textContent], [false, ''], 'held empty: CSS keeps its one line');
    c.paint({ lead: lead(GLM, 'owner'), tiers: STANDARD,
        processing: { choice: 'fast', source: 'owner', last: { requested: 'fast', submitted: null, reason: 'processing_not_submitted' } } });
    assert.deepEqual([node() === held, held.hidden, held.textContent], [true, false, 'Fast asked · sent as Standard']);
    closeCardModel();
    // A Standard work holds nothing: no receipt of another tier is coming.
    c.paint({ lead: lead(GLM, 'owner'), tiers: STANDARD, processing: { choice: '', source: '' } });
    c.fact().click();
    await flush();
    assert.equal(node().hidden, true);
    closeCardModel();
});

test('the strip offers Ultrafast where the work\'s source and model list it, or where it is already the work\'s tier', async () => {
    const listed = { ...CATALOG, model_sources: [{ ...CATALOG.model_sources[0], processingPreferences: ['standard', 'fast', 'economy', 'ultrafast'] }],
        items: CATALOG.items.map((item) => (item.value === ASTRA ? { ...item, processing: { modes: ['standard', 'fast', 'ultrafast'] } } : item)) };
    const offered = async (view) => {
        const c = card({ catalog: listed });
        c.paint({ tiers: STANDARD, ...view });
        c.fact().click();
        await flush();
        const shown = c.popup().querySelector('.model-tier').querySelectorAll('.model-tier-seg').filter((seg) => !seg.hidden).map((seg) => seg.dataset.tier);
        closeCardModel();
        return shown;
    };
    assert.deepEqual(await offered({ lead: lead(ASTRA, 'owner') }), ['economy', 'standard', 'fast', 'ultrafast']);
    assert.deepEqual(await offered({ lead: lead(GLM, 'owner') }), ['economy', 'standard', 'fast'], 'nothing lists it: not offered');
    assert.deepEqual(await offered({ lead: lead(GLM, 'owner'), processing: { choice: 'ultrafast', source: 'owner' } }),
        ['economy', 'standard', 'fast', 'ultrafast'], 'the work\'s own Ultrafast can be seen and left');
});

test('a refused Fast or Flex the host continued on Standard reads as that deviation on the card and in Details', () => {
    // The receipt of a call the host retried on Standard after the engine refused its tier before any generation
    // (owner 13.11: continue on Standard and say so): it keeps what the work asked beside what was sent and served.
    for (const [tier, word] of [['fast', 'Fast'], ['economy', 'Flex']]) {
        const c = card();
        c.paint({ lead: lead(GLM, 'owner'), tiers: STANDARD, processing: { choice: tier, source: 'owner', last: { requested: tier,
            submitted: 'standard', submittedNative: 'default', observed: 'standard', observedNative: ['default'], reason: 'submitted_mode_differs' } } });
        assert.equal(c.said('tier'), `· ${word} · ${word} asked · served Standard`);
        assert.equal(c.record.processingDetail, `${word} · last call: asked ${word} · sent Standard · served Standard`);
    }
});

test('a refused tier is said in the sheet; one the server saved offers Retry with the same request', async () => {
    const c = card({ answer: Object.assign(new Error('model wake failed'), { status: 503, body: { reason_code: 'model_wake_failed' } }) });
    c.paint({ lead: lead(GLM, 'owner'), tiers: STANDARD });
    c.fact().click();
    await flush();
    await c.chooseTier('economy');
    await flush();
    const status = c.popup().querySelector('.card-model-refusal');
    assert.deepEqual([status.hidden, status.children[0].textContent, status.children[1].hidden],
        [false, 'Your choice is saved, but the work could not be told yet.', false]);
    assert.deepEqual([c.toasts, c.tierView().checked, c.tierView().busy, c.said('tier')], [[], [], false, '']);
    c.reply.value = null;
    await c.fire(status.children[1], 'click', {});
    await flush();
    assert.deepEqual(c.calls, [['t1', { action: 'processing', processing: 'economy' }], ['t1', { action: 'processing', processing: 'economy' }]]);
    assert.equal(c.ids[0], c.ids[1], 'the retry carries the refused request\'s id');
    assert.deepEqual([status.hidden, Boolean(c.popup()), c.said('tier'), c.tierView().checked], [true, true, '· Flex', ['economy']]);
    assert.deepEqual(c.toasts.at(-1), ['Flex for this work. Switches at the next step; the running call finishes.', 'ok']);
    closeCardModel();
});

test('a tier act\'s answer lands on the card as it stands now: a newer body\'s model, receipt, tier and control stay', async () => {
    const KIMI = 'moonshotai/kimi-k3';
    const older = { requested: 'economy', submitted: 'economy', observed: 'standard' };
    const newer = { requested: 'standard', submitted: 'standard', observed: 'fast' };
    const pending = (c) => { let answer; c.reply.value = new Promise((resolve) => { answer = resolve; }); return (tier) => answer({ ok: true,
        task_id: 't1', applied: 'next_step', processing: { choice: tier, source: 'owner' } }); };

    // A body with another model and a newer receipt arrives while the press is on its way: both stay, the tier lands.
    let c = card();
    c.paint({ lead: lead(GLM, 'owner'), tiers: STANDARD, processing: { choice: '', last: older } });
    c.fact().click();
    await flush();
    let answer = pending(c);
    await c.chooseTier('fast');
    c.paint({ lead: lead(KIMI, 'owner'), tiers: STANDARD, processing: { choice: '', last: newer } });
    answer('fast');
    await flush();
    assert.deepEqual([c.said('name'), c.said('tier'), c.record.processingDetail],
        ['OpenRouter · kimi-k3', '· Fast · Standard asked · served Fast', 'Fast · last call: asked Standard · sent Standard · served Fast']);
    assert.deepEqual([Boolean(c.popup()), c.tierView().checked], [true, ['fast']]);
    closeCardModel();

    // The control is withdrawn meanwhile: the answer brings back neither it nor the view the press was made in.
    c = card();
    c.paint({ lead: lead(GLM, 'owner'), tiers: STANDARD, processing: { choice: '', last: older } });
    c.fact().click();
    await flush();
    answer = pending(c);
    await c.chooseTier('fast');
    c.paint({ lead: lead(KIMI, 'owner'), tiers: STANDARD, target: '', processing: { choice: '', last: newer } });
    assert.equal(c.popup(), null, 'the sheet goes with its control');
    answer('fast');
    await flush();
    assert.deepEqual([c.fact().tagName, c.fact().hasAttribute('aria-expanded'), c.said('name'), c.said('tier')],
        ['SPAN', false, 'OpenRouter · kimi-k3', '· Standard asked · served Fast']);
    assert.deepEqual(c.toasts, [['Fast for this work. Switches at the next step; the running call finishes.', 'ok']], 'the server took it');

    // Another window set this work's tier meanwhile: the newer body's tier stands until the next body.
    c = card();
    c.paint({ lead: lead(GLM, 'owner'), tiers: STANDARD, processing: { choice: '' } });
    c.fact().click();
    await flush();
    answer = pending(c);
    await c.chooseTier('fast');
    c.paint({ lead: lead(GLM, 'owner'), tiers: STANDARD, processing: { choice: 'economy', source: 'owner' } });
    answer('fast');
    await flush();
    assert.deepEqual([c.said('tier'), c.tierView().checked], ['· Flex', ['economy']]);
    closeCardModel();
});

test('a model act\'s answer lands on the card as it stands now: its model only where no newer body named one', async () => {
    const KIMI = 'moonshotai/kimi-k3';
    const newer = { requested: 'fast', submitted: 'fast', observed: 'standard' };
    const pending = (c) => { let answer; c.reply.value = new Promise((resolve) => { answer = resolve; }); return () => answer({ ok: true,
        task_id: 't1', applied: 'next_step', lead: lead(GLM, 'owner') }); };
    // A body with a newer receipt and the same model: the answer's model lands beside that receipt.
    let c = card();
    c.paint({ lead: lead(ASTRA, 'main'), tiers: STANDARD, processing: { choice: 'fast', source: 'owner' } });
    c.fact().click();
    await flush();
    let answer = pending(c);
    await c.take(`route|${GLM}`);
    c.paint({ lead: lead(ASTRA, 'main'), tiers: STANDARD, processing: { choice: 'fast', source: 'owner', last: newer } });
    answer();
    await flush();
    assert.deepEqual([c.said('name'), c.said('why'), c.said('tier')], ['OpenRouter · glm-5.3', 'you chose it', '· Fast · Fast asked · served Standard']);
    assert.equal(c.popup(), null);

    // A body that already names another model (the mind switched meanwhile): the answer's older model is not put back.
    c = card();
    c.paint({ lead: lead(ASTRA, 'main'), tiers: STANDARD });
    c.fact().click();
    await flush();
    answer = pending(c);
    await c.take(`route|${GLM}`);
    c.paint({ lead: lead(KIMI, 'mind'), tiers: STANDARD });
    answer();
    await flush();
    assert.deepEqual([c.said('name'), c.said('why')], ['OpenRouter · kimi-k3', 'Ouroboros switched to it']);

    // The control is withdrawn meanwhile: the card stays plain text.
    c = card();
    c.paint({ lead: lead(ASTRA, 'main'), tiers: STANDARD });
    c.fact().click();
    await flush();
    answer = pending(c);
    await c.take(`route|${GLM}`);
    c.paint({ lead: lead(ASTRA, 'main'), tiers: STANDARD, target: '' });
    answer();
    await flush();
    assert.deepEqual([c.fact().tagName, c.said('name')], ['SPAN', 'Codex · gpt-6-astra']);
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

test('a work\'s tier and its newest call\'s receipt reach its card from the state body', (t) => {
    const c = liveCard(t);
    const tier = () => { const node = c.part().querySelector('.card-model-tier'); return node.hidden ? '' : node.querySelector('.card-model-words').textContent; };
    const state = c.read(body({ works: [work({ lead: lead(ASTRA, 'main') })], processing_tiers: { main: 'economy', default: 'standard' } }));
    assert.equal(tier(), 'Flex', 'a follower of Main takes Main\'s tier');
    assert.deepEqual(state.tiers, { main: 'economy', default: 'standard' });
    c.read(body({ works: [work({ lead: lead(GLM, 'room'), processing: { choice: 'fast', source: 'room',
        last: { requested: 'fast', submitted: 'fast', observed: 'standard' } } })] }), state);
    assert.equal(tier(), '·\u00a0Fast · Fast asked · served Standard', 'its own tier, and what was served (after why, a real dot)');
    c.read(body({ works: [work({ lead: lead(GLM, 'room') })] }), state);
    assert.equal(tier(), '', 'a lead on the default tier says nothing more');
});

test('the server-resolved fallback tier drives both the card and its open footer, separately from the receipt', async () => {
    const c = card();
    const fallback = { ...lead(GLM, 'fallback'), role: 'fallback:0', primary: lead(ASTRA, 'main') };
    c.paint({ lead: fallback, tiers: { main: 'fast', default: 'standard' },
        processing: { choice: '', resolved: 'economy', last: { requested: 'fast', submitted: null,
            observed: 'unknown', reason: 'processing_not_submitted' } } });
    assert.equal(c.said('tier'), '· Flex · Fast asked · sent as Standard');
    c.fact().click();
    await flush();
    assert.match(c.tierView().scope, /^Default · Flex/);
    assert.equal(c.record.processingDetail, 'Default · Flex · last call: asked Fast · sent with no tier · served as not reported');
    c.paint({ lead: fallback, tiers: { main: 'fast', default: 'standard' },
        processing: { choice: 'standard', resolved: 'standard', source: 'owner' } });
    assert.equal(c.said('tier'), '');
    assert.match(c.tierView().scope, /^Standard ·/);
    closeCardModel();
});

test('a model act drops only the previous route resolution; Default awaits a newer body if its route raced', async () => {
    let c = card();
    const fallback = { ...lead(GLM, 'fallback'), role: 'fallback:0', primary: lead(ASTRA, 'main') };
    c.paint({ lead: fallback, tiers: { main: 'fast', default: 'standard' }, processing: { choice: '', resolved: 'economy' } });
    c.fact().click();
    await flush();
    c.reply.value = { ok: true, applied: 'next_step', lead: lead(ASTRA, 'main') };
    await c.take('main');
    await flush();
    assert.equal(c.said('tier'), 'Fast');
    closeCardModel();
    c = card();
    c.paint({ lead: fallback, tiers: STANDARD, processing: { choice: 'fast', resolved: 'fast', source: 'owner' } });
    c.fact().click();
    await flush();
    let resolve;
    c.reply.value = new Promise((done) => { resolve = done; });
    await c.chooseTier('');
    c.paint({ lead: lead(ASTRA, 'main'), tiers: STANDARD, processing: { choice: 'fast', resolved: 'fast', source: 'owner' } });
    resolve({ ok: true, applied: 'next_step', processing: { choice: '', resolved: 'economy' } });
    await flush();
    assert.equal(c.said('tier'), 'Fast', 'a stale Default resolution cannot repaint the newer route');
    closeCardModel();
});
