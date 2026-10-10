// The open Available-subagents card (docs/DESIGN.md §6): current facts stay on the
// card, history and provenance wait in Details & history, and a Reviewer or row switch
// patches the card in place. Real geometry, focus and disclosures: the browser test
// tests/test_subagent_card_quiet_browser.py.
import assert from 'node:assert/strict';
import test from 'node:test';

import { ROUTE_KIND_AGENT_SESSION, ROUTE_KIND_API_MODEL, profileOptionsFor } from '../modules/route_editor_primitives.js';
import { availableSubagentRowMarkup, createAvailableSubagentsEditor } from '../modules/subagents_settings.js';
import { rowStatus, rowStatusReason, sessionRouteVerdict } from '../modules/subagent_status_primitives.js';

const QUIET = { catalogKnown: false, accountsKnown: false, quotaKnown: false, statusError: '', snapshot: null };
const api = (extra = {}) => ({ subagent_id: 'scout', recommended_use: 'Research.',
    route: { kind: ROUTE_KIND_API_MODEL, target_id: 'openai::gpt-x' }, ...extra });
const session = (extra = {}) => ({ subagent_id: 'builder', recommended_use: 'Build.', access: 'full',
    route: { kind: ROUTE_KIND_AGENT_SESSION, target_id: 'codex=gpt-test' }, ...extra });

// Every element the painter may patch, created on first query when the row's markup carries it.
function fakeEditorDom() {
    let rebuilds = 0;
    let rows = [];
    const node = () => ({ textContent: '', hidden: false, dataset: {}, listeners: {},
        setAttribute() {}, toggleAttribute() {}, addEventListener(type, fn) { this.listeners[type] = fn; },
        dd: { textContent: '' }, querySelector() { return this.dd; } });
    const scope = (html) => {
        const nodes = new Map();
        return (selector) => {
            const key = selector.match(/^\[(data-[a-z-]+)(?:="([^"]+)")?\]/);
            if (!key || !html.includes(key[2] ? `${key[1]}="${key[2]}"` : key[1])) return null;
            if (!nodes.has(selector)) nodes.set(selector, node());
            return nodes.get(selector);
        };
    };
    const toolbar = scope('data-subagents-enabled data-subagents-intent data-review-pool-stays data-review-pool-count');
    const container = {
        scrollTop: 0,
        set innerHTML(html) {
            rebuilds += 1;
            rows = [...html.matchAll(/<article[^>]*data-subagent-row="([^"]+)"[^>]*>([\s\S]*?)<\/article>/g)]
                .map(([, key, body]) => {
                    const find = scope(body);
                    return { dataset: { subagentRow: key }, toggleAttribute() {}, querySelector: find,
                        querySelectorAll: () => [] };
                });
        },
        querySelector(selector) {
            const row = selector.match(/^\[data-subagent-row="([^"]+)"\]$/)?.[1];
            return row ? rows.find((item) => item.dataset.subagentRow === row) || null : toolbar(selector);
        },
        querySelectorAll: (selector) => (selector === '[data-subagent-row]' ? rows : []),
    };
    return { doc: { getElementById: () => container }, rows: () => rows, toolbar, rebuilds: () => rebuilds };
}

test('a Reviewer mark and a row switch patch the card in place, never rebuilding it', () => {
    const dom = fakeEditorDom();
    const editor = createAvailableSubagentsEditor({ doc: dom.doc, win: null });
    editor.load({ enabled: true, items: [api({ subagent_id: 'one' }), api({ subagent_id: 'two' }), session()] });
    assert.equal(dom.rebuilds(), 1);
    const [one, two] = dom.rows();
    const intent = dom.toolbar('[data-subagents-intent]');
    assert.equal(intent.textContent, 'Saved');
    assert.equal(one.querySelector('[data-subagent-delivery-field]').hidden, true, 'an unmarked API row keeps its delivery hidden');

    one.querySelector('[data-subagent-field="review_eligible"]').listeners.change({ target: { checked: true } });
    assert.equal(dom.rebuilds(), 1, 'the box under the pointer is never replaced');
    assert.equal(one.querySelector('[data-subagent-delivery-field]').hidden, false);
    assert.equal(one.querySelector('[data-subagent-field="effort"] option[value=""]').textContent, 'Default (reviews at high)');
    assert.equal(intent.textContent, 'Unsaved changes');
    // The twin of a marked row hears it in place too: one engine marked twice is a repeat.
    two.querySelector('[data-subagent-field="review_eligible"]').listeners.change({ target: { checked: true } });
    assert.deepEqual([two.querySelector('[data-subagent-review-notes]').textContent, dom.rebuilds()],
        ['Repeat of Subagent 1: another independent run of the same model, not a different reviewer.', 1]);
    one.querySelector('[data-subagent-field="enabled"]').listeners.change({ target: { checked: false } });
    assert.equal(one.querySelector('[data-subagent-review-exception]').textContent, 'Switched off, so not in the review pool.');
    one.querySelector('[data-subagent-field="review_eligible"]').listeners.change({ target: { checked: false } });
    assert.deepEqual([one.querySelector('[data-subagent-delivery-field]').hidden,
        one.querySelector('[data-subagent-review-exception]').hidden, dom.rebuilds()], [true, true, 1]);
    assert.equal(one.querySelector('[data-subagent-field="effort"] option[value=""]').textContent, 'Default effort');
    assert.equal(editor.setting.items[0].enabled, false);
    assert.equal('review_eligible' in editor.setting.items[0], false);
    editor.destroy();
});

test('the head holds identity, one availability word and a fixed Reviewer/actions slot', () => {
    const html = availableSubagentRowMarkup(session({ review_eligible: true, minted_from: 'review_lane' }), QUIET, 0);
    const head = html.slice(html.indexOf('class="available-subagent-head"'), html.indexOf('data-subagent-status-reason'));
    const slot = head.slice(head.indexOf('class="available-subagent-actions"'));
    assert.match(slot, /^class="available-subagent-actions">\s*<label class="available-subagent-reviewer">/);
    assert.deepEqual([...slot.matchAll(/data-subagent-(duplicate|remove)/g)].map((m) => m[1]), ['duplicate', 'remove']);
    assert.doesNotMatch(head, /review-facts|minted|Saved|Draft|In the review pool|session seat/);
    assert.match(head, /data-subagent-status data-tone="neutral" title="Agent session · live availability not checked">Not checked</);
});

for (const baseline of ['saved', 'generated']) {
    test(`a page dirty indicator owns Unsaved changes while the editor keeps its ${baseline} meaning`, () => {
        const dom = fakeEditorDom();
        const editor = createAvailableSubagentsEditor({ doc: dom.doc, win: null,
            hasPageDirtyIndicator: true, baseline });
        const setting = { enabled: true, items: [session()] };
        editor.load(setting);
        const intent = dom.toolbar('[data-subagents-intent]');
        const label = baseline === 'saved' ? 'Saved' : 'Generated draft';
        assert.equal(intent.textContent, label);
        dom.rows()[0].querySelector('[data-subagent-field="review_eligible"]')
            .listeners.change({ target: { checked: true } });
        assert.equal(editor.dirty, true);
        assert.equal(intent.textContent, '');
        assert.equal(intent.title, '');
        assert.equal(intent.hidden, false, 'the reserved toolbar slot remains in the layout');
        editor.load(setting);
        assert.equal(intent.textContent, label);
        editor.destroy();
    });
}

test('fields are labelled Source and wide Model, then Account, Effort and Access; Description, meta, Processing, Details follow', () => {
    const html = availableSubagentRowMarkup(session({ route: { kind: ROUTE_KIND_AGENT_SESSION, target_id: 'codex=gpt-test', credential_profile_id: 'koshak' } }), QUIET, 0);
    const grid = html.slice(html.indexOf('class="available-subagent-route"'), html.indexOf('available-subagent-purpose'));
    const labels = [...grid.matchAll(/class="ui-field available-subagent-field[^"]*">(\w[\w ]*?) </g)].map((m) => m[1]);
    assert.deepEqual(labels, ['Source', 'Model', 'Account', 'Reasoning effort', 'Access']);
    assert.match(grid, /available-subagent-field-model">Model /);
    assert.match(grid, /<option value="full" selected>Full system access<\/option><option value="workspace_write">Working files<\/option>/);
    assert.match(grid, />koshak \(not checked\)</, 'the labelled field names the account alone and keeps the saved pin');
    const order = ['available-subagent-route', 'available-subagent-purpose', 'data-subagent-meta', 'data-processing-details', 'data-subagent-details']
        .map((marker) => html.indexOf(marker));
    assert.deepEqual(order, [...order].sort((a, b) => a - b));
    assert.equal(profileOptionsFor([], '', { labelled: false })[0].label, 'Account: automatic rotation', 'unlabelled consumers keep the prefix');
    assert.equal(profileOptionsFor([], '', { labelled: true })[0].label, 'Automatic rotation');
});

test('Details & history keeps review cost, both histories, the stored spelling, origin and access help', () => {
    const state = { ...QUIET, reviewPool: { pool: [{ subagent_id: 'scout', last_execution: {
        effective: { route: 'api_model', model: 'gpt-x' }, review_record_id: 'rev_9' } }] } };
    const html = availableSubagentRowMarkup(api({ review_eligible: true, minted_from: 'factory_default' }), state, 0);
    const details = html.slice(html.indexOf('data-subagent-details'));
    assert.match(details, /<summary>Details &amp; history<\/summary>/);
    assert.match(details, /<dt>Review cost<\/dt><dd data-subagent-review-facts>price appears after saving<\/dd>/);
    assert.match(details, /<div data-subagent-last-review><dt>Last review<\/dt><dd>API model · gpt-x · record rev_9<\/dd>/);
    assert.match(details, /<div hidden data-subagent-last-task><dt>Last task run<\/dt>/, 'no receipt, nothing claimed');
    assert.match(details, /<dt>Stored as<\/dt><dd>openai::gpt-x<\/dd>/);
    assert.match(details, /<dt>Origin<\/dt><dd>Factory reviewer<\/dd>/);
    assert.doesNotMatch(html.slice(0, html.indexOf('data-subagent-details')), /rev_9|openai::gpt-x<|Factory reviewer/);
    const sessionHtml = availableSubagentRowMarkup(session(), QUIET, 0);
    assert.match(sessionHtml, /<dd id="actor-builder-access-help">Full system access \(the default\) can reach outside the working folder/);
});

test('a status reason line says only what the head word does not', () => {
    const gone = { catalogKnown: true, accountsKnown: true, quotaKnown: true, snapshot: { harnesses: [] } };
    assert.equal(rowStatusReason(rowStatus(session(), gone)), '', 'a bare "currently unavailable" repeats the word');
    const missingModel = { ...gone, snapshot: { harnesses: [{ id: 'codex', status: 'ok', models: [{ id: 'other' }] }] } };
    assert.equal(rowStatusReason(rowStatus(session(), missingModel)), 'codex · selected model gpt-test currently unavailable');
    assert.equal(sessionRouteVerdict(session(), missingModel).reason, 'codex · selected model gpt-test currently unavailable');
});

test('the toolbar names Delegation and one intent word; edit-toggled pool lines sit below the rows', () => {
    const dom = fakeEditorDom();
    let html = '';
    const container = dom.doc.getElementById();
    const doc = { getElementById: () => new Proxy(container, { set(target, key, value) {
        if (key === 'innerHTML') html = value;
        target[key] = value; return true;
    } }) };
    const editor = createAvailableSubagentsEditor({ doc, win: null, baseline: 'generated' });
    editor.load({ enabled: false, items: [api({ review_eligible: true })] }, { source: 'onboarding_default' });
    assert.match(html, /aria-label="Delegation to these subagents"[^>]*>\s*Delegation\s*<\/label>/);
    assert.match(html, /data-subagents-intent data-tone="neutral" title="Generated draft">Generated draft</);
    assert.match(html, /data-review-pool-stays >Delegation is off; rows marked Reviewer still review\.</);
    const list = html.indexOf('class="available-subagents-list"');
    assert.ok(html.indexOf('data-review-pool-empty') > list && html.indexOf('data-subagents-validation') > list,
        'a line a row edit toggles never sits above the rows it would push');
    assert.ok(html.indexOf('data-review-pool-note') < list, 'read failures stay at the top');
    editor.destroy();
});
