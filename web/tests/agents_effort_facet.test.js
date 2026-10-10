// Settings → Agents, the effort facet of a row (DECISIONS v3 §6): the select's empty option is
// Auto (the chat range; a Reviewer reviews at its top); a row whose level is in the model name
// — a session slug `cursor=…-xhigh[-fast]` or the API-wrapped `claudexor::agy=…-high` — shows the
// read-only "<Level> · in the model name" instead. The facet follows the model IN PLACE (select
// <-> read-only) on every model or source edit, keeping the caret, and the owner's pick of a
// named model clears the pin in the draft so the row still saves.
import assert from 'node:assert/strict';
import test from 'node:test';

import { ROUTE_KIND_AGENT_SESSION, ROUTE_KIND_API_MODEL, compoundSessionEffort, compoundSessionEffortConflict } from '../modules/route_editor_primitives.js';
import {
    availableSubagentRowMarkup, availableSubagentsSavePayload, createAvailableSubagentsEditor, validateAvailableSubagentsSetting,
} from '../modules/subagents_settings.js';

const QUIET = { catalogKnown: false, accountsKnown: false, quotaKnown: false, statusError: '', snapshot: null };
const api = (extra = {}) => ({ subagent_id: 'scout', recommended_use: 'Research.',
    route: { kind: ROUTE_KIND_API_MODEL, target_id: 'openai::gpt-x' }, ...extra });
const session = (extra = {}) => ({ subagent_id: 'builder', recommended_use: 'Build.', access: 'full',
    route: { kind: ROUTE_KIND_AGENT_SESSION, target_id: 'codex=gpt-test' }, ...extra });

/* A row's elements are found in its markup; a node's innerHTML can be rewritten, after which
   its own children are found in the new markup — enough to watch the facet swap in place. */
function fakeNode(html) {
    const children = new Map();
    return {
        _html: html, textContent: '', hidden: false, dataset: {}, listeners: {},
        get innerHTML() { return this._html; },
        set innerHTML(value) { this._html = String(value); children.clear(); },
        setAttribute() {}, toggleAttribute() {}, removeAttribute() {},
        addEventListener(type, fn) { this.listeners[type] = fn; },
        querySelector(selector) {
            const key = selector.match(/^\[(data-[a-z-]+)(?:="([^"]+)")?\]/);
            if (!key) return null;
            const needle = key[2] ? `${key[1]}="${key[2]}"` : key[1];
            const at = this._html.indexOf(needle);
            if (at < 0) return null;
            if (!children.has(selector)) children.set(selector, fakeNode(this._html.slice(at)));
            return children.get(selector);
        },
        querySelectorAll: () => [],
    };
}
function fakeEditorDom() {
    let rows = [];
    const toolbar = fakeNode('data-subagents-enabled data-subagents-intent data-review-pool-stays data-review-pool-count data-subagents-validation');
    const container = {
        scrollTop: 0,
        set innerHTML(html) {
            rows = [...html.matchAll(/<article[^>]*data-subagent-row="([^"]+)"[^>]*>([\s\S]*?)<\/article>/g)]
                .map(([, key, body]) => Object.assign(fakeNode(body), { dataset: { subagentRow: key } }));
        },
        querySelector(selector) {
            const row = selector.match(/^\[data-subagent-row="([^"]+)"\]$/)?.[1];
            return row ? rows.find((item) => item.dataset.subagentRow === row) || null : toolbar.querySelector(selector);
        },
        querySelectorAll: (selector) => (selector === '[data-subagent-row]' ? rows : []),
    };
    return { doc: { getElementById: () => container }, rows: () => rows };
}
const facetOf = (row) => row.querySelector('[data-subagent-effort-facet]');
const typeModel = (row, value) => row.querySelector('[data-subagent-field="model"]').listeners.input({ target: { value } });
const chooseEffort = (row, value) => row.querySelector('[data-subagent-effort-facet] [data-subagent-field="effort"]').listeners.change({ target: { value } });

test('the level in a model name is read for session targets and API-wrapped Claudexor models alike', () => {
    assert.equal(compoundSessionEffort('cursor=gpt-5.6-sol-high-fast'), 'high');
    assert.equal(compoundSessionEffort('claudexor::cursor=gpt-5.6-sol-high-fast'), 'high');
    assert.equal(compoundSessionEffort('claudexor::agy=gemini-3.7-flash-xhigh'), 'xhigh');
    assert.equal(compoundSessionEffort('claudexor::codex=gpt-6-astra-high'), '', 'only Cursor and Agy encode a level');
    assert.equal(compoundSessionEffort('openai::gpt-5.6-terra-high'), '');
    assert.equal(compoundSessionEffortConflict('claudexor::agy=gemini-3.7-flash-xhigh', 'low'), 'xhigh');
    assert.equal(compoundSessionEffortConflict('claudexor::agy=gemini-3.7-flash-xhigh', 'xhigh'), '', 'the same level is no conflict');
});

test('a stored pin beside a named model: a session row is refused, an API-wrapped row still saves', () => {
    const setting = (row) => ({ enabled: true, items: [row] });
    const sessionNamed = session({ route: { kind: ROUTE_KIND_AGENT_SESSION, target_id: 'cursor=grok-4.7-xhigh-fast' }, effort: 'low' });
    assert.match(validateAvailableSubagentsSetting(setting(sessionNamed)).join(' '), /conflicts with compound route effort “xhigh”/);
    // A factory reviewer seat or an older row may carry an explicit level beside `claudexor::cursor=…-xhigh`:
    // the name wins when it runs, and the rest of the catalog must still save.
    const apiNamed = api({ route: { kind: ROUTE_KIND_API_MODEL, target_id: 'claudexor::cursor=grok-4.7-xhigh-fast' }, effort: 'low' });
    assert.deepEqual(validateAvailableSubagentsSetting(setting(apiNamed)), []);
});

test('markup: Auto reads the chat range, a Reviewer the top of it, a named row the level in the model name', () => {
    const plain = availableSubagentRowMarkup(api(), QUIET, 0);
    assert.match(plain, /data-subagent-effort-facet[^>]*><select class="ui-control" data-subagent-field="effort"/);
    assert.match(plain, /<option value="" selected>Auto \(chat range\)<\/option>/);
    assert.match(plain, /title="Preferred reasoning effort — default: the chat range"/);
    assert.match(availableSubagentRowMarkup(api({ review_eligible: true }), QUIET, 0),
        /<option value="" selected>Auto \(reviews at the top of the chat range\)<\/option>/);
    for (const row of [session({ route: { kind: ROUTE_KIND_AGENT_SESSION, target_id: 'cursor=gpt-5.6-sol-xhigh-fast' } }),
        api({ route: { kind: ROUTE_KIND_API_MODEL, target_id: 'claudexor::agy=gemini-3.7-flash-xhigh' }, review_eligible: true })]) {
        const html = availableSubagentRowMarkup(row, QUIET, 0);
        assert.match(html, /data-subagent-effort-named="xhigh"[^>]*>X-High · in the model name<\/span>/);
        assert.doesNotMatch(html, /data-subagent-field="effort"/, 'no select for a named row');
        assert.doesNotMatch(html, /reviews at/, 'the name decides, not the pool default');
    }
    // All eight raw tiers stay selectable on an ordinary row.
    const select = plain.slice(plain.indexOf('data-subagent-field="effort"'));
    const options = [...select.slice(0, select.indexOf('</select>')).matchAll(/<option value="([a-z]*)"/g)].map((m) => m[1]);
    assert.deepEqual(options, ['', 'none', 'minimal', 'low', 'medium', 'high', 'xhigh', 'max', 'ultra']);
});

// A Cursor session row and a Cursor subscription (API-wrapped `claudexor::cursor=…`) row: the
// Source select fixes the harness, the model field types the slug.
for (const [kind, make, prefix] of [
    ['session', () => session({ effort: 'low', route: { kind: ROUTE_KIND_AGENT_SESSION, target_id: 'cursor=gpt-test' } }), 'cursor='],
    ['API', () => api({ effort: 'low', route: { kind: ROUTE_KIND_API_MODEL, target_id: 'claudexor::cursor=gpt-test' } }), 'claudexor::cursor='],
]) {
    test(`a ${kind} row: ordinary -> named -> named -> ordinary swaps the facet in place, clears the pin, and saves`, () => {
        const dom = fakeEditorDom();
        const editor = createAvailableSubagentsEditor({ doc: dom.doc, win: null });
        editor.load({ enabled: true, items: [make()] });
        const [row] = dom.rows();
        const facet = facetOf(row);
        const named = `${prefix}gpt-5.6-sol-high-fast`;
        assert.match(facet.innerHTML, /data-subagent-field="effort"/);
        assert.equal(editor.setting.items[0].effort, 'low');
        // The owner picks a model whose name carries the level: read-only facet, pin gone.
        typeModel(row, 'gpt-5.6-sol-high-fast');
        assert.equal(editor.setting.items[0].route.target_id, named);
        assert.equal(facet.dataset.named, 'high');
        assert.match(facet.innerHTML, /High · in the model name/);
        assert.doesNotMatch(facet.innerHTML, /data-subagent-field="effort"/);
        assert.equal('effort' in editor.setting.items[0], false, 'a named model retires the pin in the draft');
        assert.equal(dom.rows()[0], row, 'the row was patched, not rebuilt');
        let saved = availableSubagentsSavePayload({ loaded: true, setting: editor.setting }).OUROBOROS_SUBAGENTS.items[0];
        assert.equal('effort' in saved, false);
        assert.equal(saved.route.target_id, named);
        // Named -> another named: the facet follows the new level, still in place.
        typeModel(row, 'grok-4.7-xhigh-fast');
        assert.equal(editor.setting.items[0].route.target_id, `${prefix}grok-4.7-xhigh-fast`);
        assert.equal(facet.dataset.named, 'xhigh');
        assert.match(facet.innerHTML, /X-High · in the model name/);
        assert.equal(dom.rows()[0], row);
        // Back to an ordinary model: the select returns, Auto, no stale pin, and it edits again.
        typeModel(row, 'gpt-test');
        assert.equal(editor.setting.items[0].route.target_id, `${prefix}gpt-test`);
        assert.equal(facet.dataset.named, undefined);
        assert.match(facet.innerHTML, /<option value="" selected>Auto \(chat range\)<\/option>/);
        chooseEffort(row, 'medium');
        assert.equal(editor.setting.items[0].effort, 'medium');
        saved = availableSubagentsSavePayload({ loaded: true, setting: editor.setting }).OUROBOROS_SUBAGENTS.items[0];
        assert.equal(saved.effort, 'medium');
        editor.destroy();
    });
}

test('named -> another named: a session slug under a new level re-reads the facet without a rebuild', () => {
    const dom = fakeEditorDom();
    const editor = createAvailableSubagentsEditor({ doc: dom.doc, win: null });
    editor.load({ enabled: true, items: [session({ route: { kind: ROUTE_KIND_AGENT_SESSION, target_id: 'cursor=gpt-5.6-sol-high-fast' } })] });
    const [row] = dom.rows();
    const facet = facetOf(row);
    assert.match(facet.innerHTML, /High · in the model name/);
    typeModel(row, 'grok-4.7-xhigh-fast');
    assert.equal(editor.setting.items[0].route.target_id, 'cursor=grok-4.7-xhigh-fast');
    assert.equal(facet.dataset.named, 'xhigh');
    assert.match(facet.innerHTML, /X-High · in the model name/);
    assert.equal(dom.rows()[0], row);
    assert.deepEqual(availableSubagentsSavePayload({ loaded: true, setting: editor.setting }).OUROBOROS_SUBAGENTS.items[0].route,
        { kind: ROUTE_KIND_AGENT_SESSION, target_id: 'cursor=grok-4.7-xhigh-fast' });
    editor.destroy();
});
