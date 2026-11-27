// Dashboard › Costs, the widgets their authors place there (costs_widgets.js): only a live tab whose `render.page` is
// `costs` is drawn, a list without one draws and says nothing; a frame runs only while Costs is shown and stops in
// order when it is left, one frame per widget, and a return never joins the start it left behind; the owner's launch
// policy (only as this read brought it, never the author's or an older read's in its place, deciding a start it made
// until that frame has its document, never the owner's Start) and Stop hold, as does a
// start the page did not outlive; a list that answered stands when only the launch choices miss the deadline; a
// failed read keeps the last cards as they are (a running frame runs on) and offers Retry; a changed widget stops
// before its fresh card mounts; the owner's widget order places the cards without moving a frame, and the document
// follows it, moving only a card out of its place, while nothing runs (a quick return after its stops settle); a
// declarative widget leaves the screen on leave, its state kept across leaving and a new revision, not its placement
// being taken off, and only a read mounts it again; the door opens its own tab; faults are Costs'; lifecycle and
// reconnect read only while Costs is shown. Real frames are the browser lane's.
import assert from 'node:assert/strict';
import test from 'node:test';

import { createHolds, HELD_LINES, HOLDS_KEY } from '../modules/holds.js';
import { mountCostsWidgets, mountWidgetTab, placedOnCosts } from '../modules/costs_widgets.js';
import { WIDGET_REQUEST_TIMEOUT_MS } from '../modules/widget_job.js';
import { installCardDom } from './helpers/card_dom.js';

const flush = async () => { for (let i = 0; i < 12; i += 1) await new Promise((resolve) => setImmediate(resolve)); };

const placed = (skill, { revision = 'r1', start = 'auto', page = 'costs', pinned = false } = {}) => ({
    key: `${skill}:main`, skill, tab_id: 'main', title: `${skill} quotas`, icon: '◔', ws_prefix: `ext_${skill}_`,
    revision, generation: 'g1', span: 2, grid_span: 2, ...(pinned ? { pinned_card: true } : {}),
    render: { kind: 'module', entry: 'widget.js', start, ...(page ? { page } : {}) },
});

function ledger(entries = {}) {
    const map = new Map(Object.keys(entries).length ? [[HOLDS_KEY, JSON.stringify({ v: 1, entries })]] : []);
    const store = { getItem: (key) => (map.has(key) ? map.get(key) : null), setItem: (key, value) => map.set(key, String(value)),
        removeItem: (key) => map.delete(key) };
    return { map, open: () => createHolds({ win: { location: { href: 'http://local/' }, localStorage: store } }) };
}

function socket() {
    const handlers = new Map();
    return {
        on(type, fn) { if (!handlers.has(type)) handlers.set(type, new Set()); handlers.get(type).add(fn); return () => handlers.get(type).delete(fn); },
        emit(type, msg = {}) { [...(handlers.get(type) || [])].forEach((fn) => fn(msg)); },
        count: () => [...handlers.values()].reduce((sum, set) => sum + set.size, 0),
    };
}

// The page's list and preferences, and frames that say when they start and stop (a stop settles at once).
function stand(t, { tabs = [], prefs = {}, holds = ledger().open(), mountTab: mountOverride = null } = {}) {
    const dom = installCardDom();
    t.after(() => dom.restore());
    const state = { tabs, prefs, raw: undefined, fail: false, prefsFail: false, prefsHang: false, reads: 0, log: [], opened: [], places: [] };
    const client = {
        widgets: async () => {
            state.reads += 1;
            if (state.fail) throw Object.assign(new Error('HTTP 503'), { status: 503 });
            return { ui_tabs: state.tabs };
        },
        uiPreferences: async ({ signal } = {}) => {
            if (state.prefsFail) throw Object.assign(new Error('HTTP 503'), { status: 503 });
            // An answer that never comes: only the read's own deadline (its abort) ends it.
            if (state.prefsHang) await new Promise((_resolve, reject) => signal.addEventListener('abort', () => reject(Object.assign(new Error('aborted'), { name: 'AbortError' }))));
            // The server's answer always carries the map, none of the choices overridden by default
            // (ouroboros/gateway/ui_preferences.py DEFAULT_UI_PREFERENCES); `raw` is any other answer a read can bring.
            return state.raw !== undefined ? state.raw : { widget_start_mode: {}, ...state.prefs };
        },
    };
    const mountTab = mountOverride ? (...args) => mountOverride(state, ...args) : async (mount, tab, { policy }) => {
        const frame = document.createElement('iframe');
        mount.replaceChildren(frame);
        policy?.starting?.();
        state.places.push(policy?.where);
        state.log.push(`+${tab.skill}@${tab.revision}`);
        return () => { state.log.push(`-${tab.skill}@${tab.revision}`); frame.remove(); return Promise.resolve(); };
    };
    const host = document.createElement('div');
    document.body.append(host);
    const ws = socket();
    const widgets = mountCostsWidgets(host, { ws, client, holds, mountTab, openWidgets: (skill, key) => state.opened.push([skill, key]) });
    t.after(() => widgets.destroy());
    const card = (skill) => host.querySelector(`[data-widget-key="${skill}:main"]`);
    return { state, host, ws, widgets, card };
}

test('a live tab is drawn on Costs only when its author placed it there, never a card kept from the chat', () => {
    assert.equal(placedOnCosts(placed('quota')), true);
    assert.equal(placedOnCosts(placed('quota', { page: '' })), false);
    assert.equal(placedOnCosts(placed('quota', { pinned: true })), false);
    assert.equal(placedOnCosts({ skill: 'x', render: { page: 'Costs' } }), false, 'a closed value: no case folding');
    assert.equal(placedOnCosts(null), false);
});

test('only the placed widget is drawn; a list without one draws nothing and says nothing of the skill', async (t) => {
    const { state, host, widgets, card } = stand(t, { tabs: [placed('quota'), placed('other', { page: '' }), placed('kept', { pinned: true })] });
    widgets.show();
    await flush();
    assert.equal(host.querySelectorAll('[data-widget-key]').length, 1);
    assert.ok(card('quota'));
    assert.equal(card('quota').querySelector('.costs-widget-title').textContent, 'quota quotas');
    state.tabs = [];
    widgets.refresh();
    await flush();
    assert.equal(host.querySelectorAll('[data-widget-key]').length, 0);
    assert.equal(host.textContent.trim(), '', 'no "off", no empty frame: an absent card proves nothing');
});

test('a frame runs only while Costs is shown: one per widget, stopped in order on leave, started again on return', async (t) => {
    const { state, widgets, card } = stand(t, { tabs: [placed('quota')] });
    widgets.show();
    widgets.show();
    await flush();
    assert.deepEqual(state.log, ['+quota@r1'], 'two showings, one frame');
    assert.equal(card('quota').querySelector('[data-widget-power]').textContent, 'Stop');
    assert.equal(card('quota').querySelector('[data-widget-status]').textContent, 'Running', 'never "Keeps running" here');
    widgets.leave();
    await flush();
    assert.deepEqual(state.log, ['+quota@r1', '-quota@r1']);
    assert.ok(card('quota').querySelector('[data-widget-facade]'), 'the stopped card rests as its facade');
    assert.equal(card('quota').querySelector('iframe'), null);
    widgets.show();
    await flush();
    assert.deepEqual(state.log, ['+quota@r1', '-quota@r1', '+quota@r1']);
});

test('the owner’s launch policy wins over the author’s; Stop holds on this device until Start', async (t) => {
    const store = ledger();
    const { state, widgets, card } = stand(t, { tabs: [placed('quota')], prefs: { widget_start_mode: { 'quota:main': 'manual' } }, holds: store.open() });
    widgets.show();
    await flush();
    assert.deepEqual(state.log, [], 'Manual waits for Start');
    assert.equal(card('quota').querySelector('[data-widget-power]').textContent, 'Start');
    assert.ok(card('quota').querySelector('[data-widget-facade]'));
    card('quota').querySelector('[data-widget-power]').click();
    await flush();
    assert.deepEqual(state.log, ['+quota@r1']);
    card('quota').querySelector('[data-widget-power]').click();
    await flush();
    assert.deepEqual(state.log, ['+quota@r1', '-quota@r1']);
    assert.equal(JSON.parse(store.map.get(HOLDS_KEY)).entries['costs:quota:main']?.reason, 'owner');
    // The next page on this device: Auto would start it, the owner's Stop does not let it.
    const next = stand(t, { tabs: [placed('quota')], holds: store.open() });
    next.widgets.show();
    await flush();
    assert.deepEqual(next.state.log, []);
    assert.equal(next.card('quota').querySelector('[data-widget-power]').textContent, 'Start');
});

test('a start the page did not outlive holds the widget on Costs too, with its one line', async (t) => {
    const store = ledger({ 'widget:quota:main': { reason: 'start', rev: 'r1', at: 1, n: 1 } });
    const { state, widgets, card } = stand(t, { tabs: [placed('quota')], holds: store.open() });
    widgets.show();
    await flush();
    assert.deepEqual(state.log, []);
    assert.equal(card('quota').querySelector('[data-widget-status]').textContent, HELD_LINES.start);
});

test('a failed read keeps the last cards and offers Retry, which reads again', async (t) => {
    const { state, host, ws, widgets, card } = stand(t, { tabs: [placed('quota')] });
    widgets.show();
    await flush();
    state.fail = true;
    ws.emit('extension_lifecycle', { skill: 'quota' });
    await flush();
    const line = host.querySelector('.costs-widgets-error');
    assert.ok(line, 'the failure is said');
    assert.match(line.textContent, /could not be read/);
    assert.ok(card('quota'), 'the last good card stays');
    assert.deepEqual(state.log, ['+quota@r1'], 'its frame keeps running');
    state.fail = false;
    const reads = state.reads;
    line.querySelector('[data-costs-widgets-retry]').click();
    await flush();
    assert.equal(state.reads, reads + 1);
    assert.equal(host.querySelector('.costs-widgets-error'), null);
});

test('a changed widget stops in order before its fresh card mounts; a vanished one is stopped and removed', async (t) => {
    const { state, ws, widgets, card, host } = stand(t, { tabs: [placed('quota')] });
    widgets.show();
    await flush();
    state.tabs = [placed('quota', { revision: 'r2' })];
    ws.emit('extension_lifecycle', { skill: 'quota' });
    await flush();
    assert.deepEqual(state.log, ['+quota@r1', '-quota@r1', '+quota@r2']);
    assert.equal(host.querySelectorAll('[data-widget-key]').length, 1);
    state.tabs = [];
    ws.emit('extension_lifecycle', { skill: 'quota' });
    await flush();
    assert.deepEqual(state.log, ['+quota@r1', '-quota@r1', '+quota@r2', '-quota@r2']);
    assert.equal(card('quota'), null);
});

test('lifecycle and reconnect read only while Costs is shown; the door opens the widget’s board card', async (t) => {
    const { state, ws, widgets, card } = stand(t, { tabs: [placed('quota')] });
    widgets.show();
    await flush();
    widgets.leave();
    await flush();
    const reads = state.reads;
    ws.emit('extension_lifecycle', { skill: 'quota' });
    ws.emit('open', { previouslyConnected: true });
    await flush();
    assert.equal(state.reads, reads, 'hidden: no read, no frame');
    assert.deepEqual(state.log, ['+quota@r1', '-quota@r1']);
    widgets.show();
    await flush();
    ws.emit('open', { previouslyConnected: true });
    await flush();
    assert.equal(state.reads, reads + 2);
    card('quota').querySelector('[data-costs-widget-open]').click();
    assert.deepEqual(state.opened, [['quota', 'quota:main']]);
});

test('destroy stops the frames and gives back every socket listener', async (t) => {
    const { state, ws, widgets, host } = stand(t, { tabs: [placed('quota')] });
    widgets.show();
    await flush();
    assert.equal(ws.count(), 3);
    widgets.destroy();
    await flush();
    assert.deepEqual(state.log, ['+quota@r1', '-quota@r1']);
    assert.equal(ws.count(), 0);
    assert.equal(host.querySelectorAll('[data-widget-key]').length, 0);
});

// A frame whose start the test settles: `+` when its document is in, `-` when it stops.
const deferredFrames = (gates, { inBeforeAbort = false } = {}) => (state, mount, tab, { signal, policy }) => new Promise((resolve) => {
    gates.push(() => {
        if (signal.aborted && !inBeforeAbort) { state.log.push(`x${tab.skill}@${tab.revision}`); resolve(undefined); return; }
        const frame = document.createElement('iframe');
        mount.replaceChildren(frame);
        policy?.starting?.();
        state.log.push(`+${tab.skill}@${tab.revision}`);
        resolve(() => { state.log.push(`-${tab.skill}@${tab.revision}`); frame.remove(); return Promise.resolve(); });
    });
});

test('leaving while a frame starts leaves no frame behind and offers no empty one', async (t) => {
    const gates = [];
    const { state, widgets, card } = stand(t, { tabs: [placed('quota')], mountTab: deferredFrames(gates) });
    widgets.show();
    await flush();
    assert.equal(card('quota').querySelector('[data-widget-power]').disabled, true, 'no Stop of a frame not in yet');
    widgets.leave();
    gates.shift()();
    await flush();
    assert.equal(card('quota').querySelector('iframe'), null);
    assert.ok(card('quota').querySelector('[data-widget-facade]'));
    assert.equal(card('quota').querySelector('[data-widget-power]').textContent, 'Start');
    assert.ok(!state.log.some((entry) => entry.startsWith('+')) || state.log.at(-1).startsWith('-'), state.log.join(' '));
});

for (const inBeforeAbort of [false, true]) test(`a return before the start it left behind ended starts anew once that start ${inBeforeAbort ? 'stopped its frame' : 'gave up'}`, async (t) => {
    const gates = [];
    const { state, widgets, card, host } = stand(t, { tabs: [placed('quota')], mountTab: deferredFrames(gates, { inBeforeAbort }) });
    widgets.show();
    await flush();
    widgets.leave();
    widgets.show();
    await flush();
    assert.equal(gates.length, 1, 'the new showing waits for the start it left behind');
    gates.shift()();  // the old start lands after leaving: it finds itself stale and leaves no frame running
    await flush();
    assert.equal(gates.length, 1, 'then this showing starts its own');
    gates.shift()();
    await flush();
    assert.deepEqual(state.log, inBeforeAbort ? ['+quota@r1', '-quota@r1', '+quota@r1'] : ['xquota@r1', '+quota@r1']);
    assert.equal(host.querySelectorAll('iframe').length, 1, 'an Auto widget runs again, with one frame');
    assert.equal(card('quota').querySelector('[data-widget-power]').textContent, 'Stop');
    assert.equal(card('quota').querySelector('[data-widget-status]').textContent, 'Running');
});

for (const inBeforeAbort of [false, true]) test(`a widget updated while its frame starts mounts the new revision only after the old one ${inBeforeAbort ? 'stopped' : 'gave up'}`, async (t) => {
    const gates = [];
    const { state, ws, widgets, host } = stand(t, { tabs: [placed('quota')], mountTab: deferredFrames(gates, { inBeforeAbort }) });
    widgets.show();
    await flush();
    state.tabs = [placed('quota', { revision: 'r2' })];
    ws.emit('extension_lifecycle', { skill: 'quota' });
    await flush();
    assert.equal(gates.length, 1, 'the new revision waits for the start under way');
    gates.shift()();  // the old start lands late: it finds itself replaced and stops
    await flush();
    assert.equal(gates.length, 1, 'now the new revision starts');
    gates.shift()();
    await flush();
    // Aborted before its document went in, the old start puts in no frame; a frame already in stops before the new one.
    assert.deepEqual(state.log, inBeforeAbort ? ['+quota@r1', '-quota@r1', '+quota@r2'] : ['xquota@r1', '+quota@r2']);
    assert.equal(host.querySelectorAll('iframe').length, 1, 'one frame per widget, never two');
});

test('a module that cannot load keeps its words and Start, never an empty frame', async (t) => {
    const { widgets, card } = stand(t, { tabs: [placed('quota')], mountTab: async (_state, mount) => {
        mount.innerHTML = '<div class="skills-load-error">module load failed: HTTP 500</div>';
        return undefined;
    } });
    widgets.show();
    await flush();
    assert.equal(card('quota').querySelector('iframe'), null);
    assert.match(card('quota').querySelector('[data-widget-mount]').textContent, /module load failed: HTTP 500/);
    assert.equal(card('quota').querySelector('[data-widget-power]').textContent, 'Start');
    assert.equal(card('quota').querySelector('[data-widget-power]').disabled, false);
});

test('until a read brings the owner’s launch policy a framed widget waits for Start beside Retry, never on the author’s', async (t) => {
    // The author says Auto; the owner saved Manual; the first read of it fails.
    const { state, host, widgets, card } = stand(t, { tabs: [placed('quota')], prefs: { widget_start_mode: { 'quota:main': 'manual' } } });
    state.prefsFail = true;
    widgets.show();
    await flush();
    assert.deepEqual(state.log, [], 'the author’s Auto does not stand in for the unread Manual');
    assert.ok(card('quota').querySelector('[data-widget-facade]'));
    assert.equal(card('quota').querySelector('[data-widget-power]').textContent, 'Start');
    assert.equal(card('quota').querySelector('[data-widget-power]').disabled, false);
    const line = host.querySelector('.costs-widgets-error');
    assert.match(line.textContent, /launch policies could not be read/);
    state.prefsFail = false;
    line.querySelector('[data-costs-widgets-retry]').click();
    await flush();
    assert.deepEqual(state.log, [], 'read at last, Manual keeps it waiting');
    assert.equal(host.querySelector('.costs-widgets-error'), null);
    const reads = state.reads;
    card('quota').querySelector('[data-widget-power]').click();
    await flush();
    assert.deepEqual(state.log, ['+quota@r1'], 'Start needs no read');
    assert.equal(state.reads, reads);
});

test('a launch policy read before never stands in for one this read lacks: what runs runs on, nothing new starts', async (t) => {
    const { state, host, ws, widgets, card } = stand(t, { tabs: [placed('quota'), placed('other')] });
    widgets.show();
    await flush();
    assert.deepEqual(state.log, ['+quota@r1', '+other@r1'], 'Auto as read: both start');
    card('other').querySelector('[data-widget-power]').click();
    await flush();
    // A read that lacks the choices while Costs is shown: the running frame is not stopped for it, the stopped one
    // is not started by the Auto read before.
    state.prefsFail = true;
    ws.emit('extension_lifecycle', { skill: 'quota' });
    await flush();
    assert.deepEqual(state.log, ['+quota@r1', '+other@r1', '-other@r1']);
    assert.match(host.querySelector('.costs-widgets-error').textContent, /launch policies could not be read/);
    // Then the owner saves Manual on the board, and the read on return to Costs fails again: the Auto read before does
    // not start the widget.
    state.prefs = { widget_start_mode: { 'quota:main': 'manual' } };
    widgets.leave();
    widgets.show();
    await flush();
    assert.deepEqual(state.log, ['+quota@r1', '+other@r1', '-other@r1', '-quota@r1'], 'nothing starts by itself');
    assert.equal(card('quota').querySelector('[data-widget-power]').textContent, 'Start');
    assert.equal(card('quota').querySelector('[data-widget-power]').disabled, false);
    state.prefsFail = false;
    host.querySelector('[data-costs-widgets-retry]').click();
    await flush();
    assert.deepEqual(state.log.slice(4), [], 'read at last, the owner’s Manual keeps it waiting');
    assert.equal(host.querySelector('.costs-widgets-error'), null);
});

test('a list that answered stands when only the launch choices miss the deadline: no start by itself, Start works', async (t) => {
    t.mock.timers.enable({ apis: ['setTimeout'] });
    const { state, host, widgets, card } = stand(t, { tabs: [placed('quota')] });
    state.prefsHang = true;
    widgets.show();
    await flush();
    assert.equal(card('quota'), null, 'the list is in; the read still waits for the choices');
    t.mock.timers.tick(WIDGET_REQUEST_TIMEOUT_MS);
    await flush();
    assert.ok(card('quota'), 'the card the list named is drawn');
    assert.deepEqual(state.log, [], 'the owner’s choice is unknown: the author’s Auto does not start it');
    assert.match(host.querySelector('.costs-widgets-error').textContent, /launch policies could not be read/);
    card('quota').querySelector('[data-widget-power]').click();
    await flush();
    assert.deepEqual(state.log, ['+quota@r1'], 'Start needs no read');
});

test('the owner’s widget order places the cards without moving a running frame; a new widget takes its place in it', async (t) => {
    const { state, ws, widgets, host } = stand(t, { tabs: [placed('alpha'), placed('beta')], prefs: { widget_order: ['beta:main', 'alpha:main'] } });
    const keys = () => host.querySelectorAll('[data-widget-key]').map((node) => node.dataset.widgetKey);
    const places = () => Object.fromEntries(host.querySelectorAll('[data-widget-key]')
        .map((node) => [node.dataset.widgetKey, node.style.getPropertyValue('--costs-widget-order')]));
    widgets.show();
    await flush();
    assert.deepEqual(places(), { 'beta:main': '0', 'alpha:main': '1' });
    const frames = host.querySelectorAll('iframe');
    assert.equal(frames.length, 2);
    // Reordered on the board, and a widget placed between the two.
    state.prefs = { widget_order: ['alpha:main', 'gamma:main', 'beta:main'] };
    state.tabs = [placed('alpha'), placed('beta'), placed('gamma')];
    ws.emit('extension_lifecycle', { skill: 'gamma' });
    await flush();
    assert.deepEqual(places(), { 'alpha:main': '0', 'gamma:main': '1', 'beta:main': '2' });
    assert.deepEqual(keys(), ['beta:main', 'alpha:main', 'gamma:main'], 'no card node moved: the order is the cards’ places');
    assert.ok(frames.every((frame) => frame.isConnected && host.querySelectorAll('iframe').includes(frame)), 'the same frames');
    assert.deepEqual(state.log.filter((entry) => !entry.startsWith('+')), [], 'no running frame stopped');
    assert.equal(state.log.length, 3);
    // A widget the stored order does not hold yet keeps the place this page showed it in, ahead of a newer one.
    state.prefs = { widget_order: ['alpha:main', 'beta:main'] };
    state.tabs = [placed('delta'), placed('alpha'), placed('beta'), placed('gamma')];
    ws.emit('extension_lifecycle', { skill: 'delta' });
    await flush();
    assert.deepEqual(places(), { 'alpha:main': '0', 'beta:main': '1', 'gamma:main': '2', 'delta:main': '3' });
});

test('the door opens the card of its own tab; a frame here names Costs as where its faults happen', async (t) => {
    const history = { ...placed('quota'), key: 'quota:history', tab_id: 'history', title: 'Quota history' };
    const { state, widgets, host } = stand(t, { tabs: [placed('quota'), history] });
    widgets.show();
    await flush();
    host.querySelectorAll('[data-widget-key]').find((node) => node.dataset.widgetKey === 'quota:history')
        .querySelector('[data-costs-widget-open]').click();
    assert.deepEqual(state.opened, [['quota', 'quota:history']], 'the second tab, not the skill’s first card');
    assert.deepEqual(state.places, ['costs', 'costs']);
});

test('a declarative widget keeps its state across leaving and a new revision; taken off Costs, it forgets it', async (t) => {
    const notes = (tabId, { revision = 'r1', page = 'costs' } = {}) => ({ key: `notes:${tabId}`, skill: 'notes', tab_id: tabId,
        title: `Notes ${tabId}`, icon: '', ws_prefix: '', revision, generation: 'g1', span: 1, grid_span: 1,
        render: { kind: 'declarative', schema_version: 1, ...(page ? { page } : {}),
            components: [{ type: 'tabs', tabs: [{ label: 'One', components: [] }, { label: 'Two', components: [] }] }] } });
    const { state, ws, widgets, host } = stand(t, { tabs: [notes('a'), notes('b')],
        mountTab: (_state, mount, tab, options) => mountWidgetTab(mount, tab, options) });
    const cardOf = (tabId) => host.querySelectorAll('[data-widget-key]').find((node) => node.dataset.widgetKey === `notes:${tabId}`);
    const chosen = (tabId) => cardOf(tabId).querySelectorAll('[data-widget-tab-key]').map((button) => button.getAttribute('aria-selected')).join(',');
    const lifecycle = async (tabs) => { state.tabs = tabs; ws.emit('extension_lifecycle', { skill: 'notes' }); await flush(); };
    widgets.show();
    await flush();
    cardOf('a').querySelectorAll('[data-widget-tab-key]')[1].click();
    cardOf('b').querySelectorAll('[data-widget-tab-key]')[1].click();
    widgets.leave();
    widgets.show();
    await flush();
    assert.equal(chosen('a'), 'false,true', 'leaving Costs and returning keeps what the owner chose');
    await lifecycle([notes('a', { revision: 'r2' }), notes('b')]);
    assert.equal(chosen('a'), 'false,true', 'so does a new revision, as on the board');
    await lifecycle([notes('a', { revision: 'r2', page: '' }), notes('b')]);
    assert.equal(cardOf('a'), undefined, 'taken off Costs (still on the board)');
    await lifecycle([notes('a', { revision: 'r2' }), notes('b')]);
    assert.equal(chosen('a'), 'true,false', 'placed again, it starts afresh');
    assert.equal(chosen('b'), 'false,true', 'the other tab keeps its own');
});

test('a host-drawn widget leaves the screen on leave; a failed read on return keeps it off, a read mounts it again', async (t) => {
    // Inert widgets only: an empty one (it draws nothing) and one static callout; no form, no route.
    const declared = (skill, components) => ({ key: `${skill}:main`, skill, tab_id: 'main', title: skill, icon: '', ws_prefix: '',
        revision: 'r1', generation: 'g1', span: 1, grid_span: 1, render: { kind: 'declarative', schema_version: 1, page: 'costs', components } });
    const mounts = { empty: 0, notes: 0 };
    const disposals = { empty: 0, notes: 0 };
    const { state, host, widgets, card } = stand(t, { tabs: [declared('empty', []), declared('notes', [{ type: 'callout', text: 'Synthetic note' }])],
        mountTab: async (_state, mount, tab, options) => {
            mounts[tab.skill] += 1;
            const dispose = await mountWidgetTab(mount, tab, options);
            return () => { disposals[tab.skill] += 1; return dispose(); };
        } });
    const body = () => card('notes').querySelector('[data-widget-mount]');
    widgets.show();
    await flush();
    assert.deepEqual(mounts, { empty: 1, notes: 1 });
    assert.match(body().textContent, /Synthetic note/);
    assert.equal(card('notes').hasAttribute('data-widget-framed'), false, 'its body keeps the card inset (style.css)');
    widgets.leave();
    assert.deepEqual(disposals, { empty: 1, notes: 1 }, 'leaving disposes both, which keeps what each may keep');
    assert.equal(body().childNodes.length, 0, 'disposed with Costs, at once: nothing of it stays on the card');
    state.fail = true;
    widgets.show();
    await flush();
    assert.deepEqual(mounts, { empty: 1, notes: 1 }, 'no fresh mount without a read');
    assert.equal(body().childNodes.length, 0, 'the card at rest beside the failure line');
    assert.match(host.querySelector('.costs-widgets-error').textContent, /could not be read/);
    state.fail = false;
    host.querySelector('[data-costs-widgets-retry]').click();
    await flush();
    assert.deepEqual(mounts, { empty: 2, notes: 2 }, 'the read brought them back: mounted anew');
    assert.match(body().textContent, /Synthetic note/);
});

test('while nothing runs the cards take the owner’s order in the document too, so focus follows the eye', async (t) => {
    const { state, ws, widgets, host } = stand(t, { tabs: [placed('alpha'), placed('beta')], prefs: { widget_order: ['beta:main', 'alpha:main'] } });
    const keys = () => host.querySelectorAll('[data-widget-key]').map((node) => node.dataset.widgetKey);
    widgets.show();
    await flush();
    assert.deepEqual(keys(), ['beta:main', 'alpha:main']);
    const frames = host.querySelectorAll('iframe');
    state.prefs = { widget_order: ['alpha:main', 'beta:main'] };
    ws.emit('extension_lifecycle', { skill: 'alpha' });
    await flush();
    assert.deepEqual(keys(), ['beta:main', 'alpha:main'], 'running frames are never moved: the order is their places meanwhile');
    assert.ok(frames.every((frame) => frame.isConnected));
    widgets.leave();
    await flush();
    widgets.show();
    await flush();
    assert.deepEqual(keys(), ['alpha:main', 'beta:main'], 'stopped on leave, the cards were put in the order on return');
    assert.deepEqual(state.log, ['+beta@r1', '+alpha@r1', '-beta@r1', '-alpha@r1', '+alpha@r1', '+beta@r1']);
});

// Card moves in the document: every insertion into the host (append, prepend, after) is one.
const countMoves = (host) => {
    const moves = [];
    const insert = host.insertBefore.bind(host);
    host.insertBefore = (node, ref) => { moves.push(node.dataset?.widgetKey || node.className); return insert(node, ref); };
    return moves;
};

test('a read that changes no place moves no card; a changed order moves only the card out of its place', async (t) => {
    const manual = { widget_start_mode: { 'alpha:main': 'manual', 'beta:main': 'manual', 'gamma:main': 'manual' } };
    const { state, ws, widgets, host } = stand(t, { tabs: [placed('alpha'), placed('beta'), placed('gamma')],
        prefs: { ...manual, widget_order: ['alpha:main', 'beta:main', 'gamma:main'] } });
    const keys = () => host.querySelectorAll('[data-widget-key]').map((node) => node.dataset.widgetKey);
    widgets.show();
    await flush();
    assert.deepEqual(state.log, [], 'Manual: nothing runs');
    const moves = countMoves(host);
    ws.emit('extension_lifecycle', { skill: 'alpha' });
    await flush();
    assert.deepEqual(moves, [], 'the same order: a focused Start keeps its focus');
    state.prefs = { ...manual, widget_order: ['beta:main', 'alpha:main', 'gamma:main'] };
    ws.emit('extension_lifecycle', { skill: 'beta' });
    await flush();
    assert.deepEqual(keys(), ['beta:main', 'alpha:main', 'gamma:main']);
    assert.deepEqual(moves, ['beta:main'], 'only the card out of its place moved');
});

test('a return during the stops it left waits for them, then places the cards before any frame of it starts', async (t) => {
    const stops = [];
    const orderAtStart = [];
    const { state, ws, widgets, host } = stand(t, { tabs: [placed('alpha'), placed('beta')], prefs: { widget_order: ['beta:main', 'alpha:main'] },
        // An empty inert node stands for each frame; its stop settles when the test lets it (the acknowledgement).
        mountTab: async (_state, mount, tab) => {
            orderAtStart.push(host.querySelectorAll('[data-widget-key]').map((node) => node.dataset.widgetKey).join(' '));
            const frame = document.createElement('iframe');
            mount.replaceChildren(frame);
            state.log.push(`+${tab.skill}`);
            return () => new Promise((resolve) => stops.push(() => { state.log.push(`-${tab.skill}`); frame.remove(); resolve(); }));
        } });
    const keys = () => host.querySelectorAll('[data-widget-key]').map((node) => node.dataset.widgetKey);
    widgets.show();
    await flush();
    state.prefs = { widget_order: ['alpha:main', 'beta:main'] };
    ws.emit('extension_lifecycle', { skill: 'alpha' });
    await flush();
    assert.deepEqual(keys(), ['beta:main', 'alpha:main'], 'running: the frames stay where they are');
    widgets.leave();
    widgets.show();  // back before the frames acknowledged their stop
    await flush();
    assert.deepEqual(state.log, ['+beta', '+alpha'], 'no frame starts while the ones it left still stop');
    assert.deepEqual(keys(), ['beta:main', 'alpha:main'], 'nor is a card holding a stopping frame moved');
    stops.splice(0).forEach((settle) => settle());
    await flush();
    assert.deepEqual(keys(), ['alpha:main', 'beta:main']);
    assert.deepEqual(state.log, ['+beta', '+alpha', '-beta', '-alpha', '+alpha', '+beta']);
    assert.deepEqual(orderAtStart.slice(2), ['alpha:main beta:main', 'alpha:main beta:main'], 'placed before the first start');
});

test('a start left in flight keeps every card where it is: its frame may already be in its card', async (t) => {
    const gates = [];
    const { state, ws, widgets, host } = stand(t, { tabs: [placed('alpha'), placed('beta')], prefs: { widget_order: ['beta:main', 'alpha:main'] },
        mountTab: deferredFrames(gates) });
    const keys = () => host.querySelectorAll('[data-widget-key]').map((node) => node.dataset.widgetKey);
    widgets.show();
    await flush();
    const moves = countMoves(host);
    state.prefs = { widget_order: ['alpha:main', 'beta:main'] };
    widgets.leave();
    widgets.show();
    ws.emit('extension_lifecycle', { skill: 'alpha' });
    await flush();
    assert.deepEqual(moves, [], 'no card moves while a start it left behind may still put a frame in');
    assert.deepEqual(keys(), ['beta:main', 'alpha:main']);
    gates.splice(0).forEach((open) => open());
    await flush();
    gates.splice(0).forEach((open) => open());
    await flush();
    assert.equal(host.querySelectorAll('iframe').length, 2, 'both run in this showing, each card where it stood');
});

// The four cases the card owes its reader at its edges. What the preferences read can bring besides the server's own
// answer (`raw`): fetchJson hands an OK answer it cannot read over as `{ error }`, and a body of another shape is no map.
const NO_MAP = {
    'an error object': { error: 'non-json response (HTTP 200)' },
    'an answer without the map': { widget_order: [] },
    'an array': [],
    'text': 'ok',
    'a null map': { widget_start_mode: null },
    'an array for the map': { widget_start_mode: [] },
    'text for the map': { widget_start_mode: 'manual' },
    'the defaults standing in for saved preferences the core could not read': { widget_start_mode: {}, widget_order: [], unreadable: true },
};
for (const [name, raw] of Object.entries(NO_MAP)) test(`${name} is no read of the owner’s launch policy: nothing starts by itself, Start and Retry stay`, async (t) => {
    const { state, host, widgets, card } = stand(t, { tabs: [placed('quota')] });
    state.raw = raw;
    widgets.show();
    await flush();
    assert.deepEqual(state.log, [], 'the author’s Auto does not stand in for a map that was not read');
    assert.ok(card('quota').querySelector('[data-widget-facade]'));
    assert.equal(card('quota').querySelector('[data-widget-power]').textContent, 'Start');
    assert.equal(card('quota').querySelector('[data-widget-power]').disabled, false);
    assert.match(host.querySelector('.costs-widgets-error').textContent, /launch policies could not be read/);
    state.raw = undefined;  // the read that brings the map settles it: an empty one overrides nothing
    host.querySelector('[data-costs-widgets-retry]').click();
    await flush();
    assert.deepEqual(state.log, ['+quota@r1']);
    assert.equal(host.querySelector('.costs-widgets-error'), null);
});

test('an empty map is a read: no choice overridden, so each author’s launch policy decides; a later answer without it stops nothing', async (t) => {
    const { state, host, ws, widgets, card } = stand(t, { tabs: [placed('quota'), placed('other', { start: 'manual' })] });
    widgets.show();
    await flush();
    assert.deepEqual(state.log, ['+quota@r1'], 'Auto starts, the author’s Manual waits for Start');
    assert.equal(card('other').querySelector('[data-widget-power]').textContent, 'Start');
    assert.equal(host.querySelector('.costs-widgets-error'), null, 'an empty map is not a failure to read');
    state.raw = { error: 'non-json response (HTTP 200)' };
    ws.emit('extension_lifecycle', { skill: 'quota' });
    await flush();
    assert.deepEqual(state.log, ['+quota@r1'], 'what runs runs on: the owner’s Stop, a changed widget or leaving stop a frame, not this');
    assert.equal(card('quota').querySelector('[data-widget-power]').textContent, 'Stop');
    assert.match(host.querySelector('.costs-widgets-error').textContent, /launch policies could not be read/);
});

test('the defaults standing in for unreadable saved preferences place no card either: the order last read still does', async (t) => {
    const { state, ws, widgets, host } = stand(t, { tabs: [placed('alpha'), placed('beta')], prefs: { widget_order: ['gamma:main', 'beta:main', 'alpha:main'] } });
    const places = () => Object.fromEntries(host.querySelectorAll('[data-widget-key]')
        .map((node) => [node.dataset.widgetKey, node.style.getPropertyValue('--costs-widget-order')]));
    widgets.show();
    await flush();
    assert.deepEqual(places(), { 'beta:main': '0', 'alpha:main': '1' });
    state.raw = { widget_start_mode: {}, widget_order: [], unreadable: true };
    state.tabs = [placed('alpha'), placed('beta'), placed('gamma')];
    ws.emit('extension_lifecycle', { skill: 'gamma' });
    await flush();
    assert.deepEqual(places(), { 'gamma:main': '0', 'beta:main': '1', 'alpha:main': '2' }, 'the owner’s order, not the defaults’ empty one');
});

// A frame whose start rejects as an aborted fetch does; let in, it puts its document in and runs.
const abortableFrames = (gates) => (state, mount, tab, { signal, policy }) => new Promise((resolve, reject) => {
    signal.addEventListener('abort', () => { state.log.push(`x${tab.skill}@${tab.revision}`); reject(Object.assign(new Error('The operation was aborted.'), { name: 'AbortError' })); });
    gates.push(() => {
        if (signal.aborted) return;
        const frame = document.createElement('iframe');
        mount.replaceChildren(frame);
        policy?.starting?.();
        state.log.push(`+${tab.skill}@${tab.revision}`);
        resolve(() => { state.log.push(`-${tab.skill}@${tab.revision}`); frame.remove(); return Promise.resolve(); });
    });
});

for (const [ends, frames] of [['gives up', deferredFrames], ['rejects', abortableFrames]]) test(`a fresher Manual declines an automatic start whose frame has no document yet (its mount ${ends}): the card rests with Start`, async (t) => {
    const gates = [];
    const store = ledger();
    const { state, ws, widgets, host, card } = stand(t, { tabs: [placed('quota')], mountTab: frames(gates), holds: store.open() });
    const power = () => card('quota').querySelector('[data-widget-power]');
    widgets.show();
    await flush();
    assert.equal(gates.length, 1, 'Auto as read: its frame is on its way');
    state.prefs = { widget_start_mode: { 'quota:main': 'manual' } };  // chosen on the board meanwhile
    ws.emit('extension_lifecycle', { skill: 'quota' });
    await flush();
    gates.splice(0).forEach((open) => open());  // what the mount brings after the decline puts nothing in
    await flush();
    assert.deepEqual(state.log, ['xquota@r1'], 'no document went in');
    assert.equal(host.querySelectorAll('iframe').length, 0);
    assert.equal(card('quota').querySelector('.skills-load-error'), null, 'a declined start is no failure');
    assert.equal(card('quota').querySelector('[data-widget-facade]').style.getPropertyValue('--widget-frame-height'), '96px', 'it rests as a Manual card');
    assert.equal(power().textContent, 'Start');
    assert.equal(power().disabled, false);
    assert.equal(store.map.get(HOLDS_KEY), undefined, 'no start of it was written to the ledger');
    power().click();
    await flush();
    gates.splice(0).forEach((open) => open());
    await flush();
    assert.deepEqual(state.log, ['xquota@r1', '+quota@r1'], 'the owner’s Start starts it');
    assert.equal(power().textContent, 'Stop');
});

test('a read without the choices declines a start the policy made, never the owner’s Start', async (t) => {
    const gates = [];
    const { state, host, ws, widgets, card } = stand(t, { tabs: [placed('quota'), placed('asked', { start: 'manual' })], mountTab: deferredFrames(gates) });
    widgets.show();
    await flush();
    card('asked').querySelector('[data-widget-power]').click();
    await flush();
    assert.equal(gates.length, 2, 'Auto’s start and the owner’s are both on their way');
    state.prefsFail = true;
    ws.emit('extension_lifecycle', { skill: 'quota' });
    await flush();
    gates.splice(0).forEach((open) => open());
    await flush();
    assert.deepEqual(state.log, ['xquota@r1', '+asked@r1']);
    assert.equal(card('quota').querySelector('[data-widget-power]').textContent, 'Start');
    assert.equal(card('quota').querySelector('[data-widget-power]').disabled, false);
    assert.equal(card('asked').querySelector('[data-widget-status]').textContent, 'Running');
    assert.match(host.querySelector('.costs-widgets-error').textContent, /launch policies could not be read/);
    // Left and shown again with the choices read: Auto starts it as on any showing, the Manual one waits for Start.
    state.prefsFail = false;
    widgets.leave();
    widgets.show();
    await flush();
    gates.splice(0).forEach((open) => open());
    await flush();
    assert.deepEqual(state.log, ['xquota@r1', '+asked@r1', '-asked@r1', '+quota@r1']);
    assert.equal(card('quota').querySelector('[data-widget-status]').textContent, 'Running');
    assert.equal(card('asked').querySelector('[data-widget-power]').textContent, 'Start');
    assert.equal(host.querySelector('.costs-widgets-error'), null);
});

test('a frame that has its document runs: a fresher Manual neither declines its start nor stops it', async (t) => {
    const gates = [];
    const { state, ws, widgets, card } = stand(t, { tabs: [placed('quota')], mountTab: (state_, mount, tab, { policy }) => {
        const frame = document.createElement('iframe');
        mount.replaceChildren(frame);
        policy.starting();
        state_.log.push(`+${tab.skill}@${tab.revision}`);
        return new Promise((resolve) => gates.push(() => resolve(() => { state_.log.push(`-${tab.skill}@${tab.revision}`); frame.remove(); return Promise.resolve(); })));
    } });
    widgets.show();
    await flush();
    state.prefs = { widget_start_mode: { 'quota:main': 'manual' } };
    ws.emit('extension_lifecycle', { skill: 'quota' });
    await flush();
    gates.splice(0).forEach((open) => open());
    await flush();
    ws.emit('extension_lifecycle', { skill: 'quota' });
    await flush();
    assert.deepEqual(state.log, ['+quota@r1']);
    assert.equal(card('quota').querySelector('[data-widget-status]').textContent, 'Running');
    assert.equal(card('quota').querySelector('[data-widget-power]').textContent, 'Stop');
});

test('Auto read again before a declined start has ended starts anew once it has, with one frame', async (t) => {
    const gates = [];
    const { state, ws, widgets, host, card } = stand(t, { tabs: [placed('quota')], mountTab: deferredFrames(gates) });
    widgets.show();
    await flush();
    state.prefs = { widget_start_mode: { 'quota:main': 'manual' } };
    ws.emit('extension_lifecycle', { skill: 'quota' });
    await flush();
    state.prefs = {};
    ws.emit('extension_lifecycle', { skill: 'quota' });
    await flush();
    assert.equal(gates.length, 1, 'the new start waits for the declined one to end');
    gates.shift()();
    await flush();
    assert.equal(gates.length, 1);
    gates.shift()();
    await flush();
    assert.deepEqual(state.log, ['xquota@r1', '+quota@r1']);
    assert.equal(host.querySelectorAll('iframe').length, 1);
    assert.equal(card('quota').querySelector('[data-widget-status]').textContent, 'Running');
});

test('a card that joins the list while Costs is shown starts under Auto as on showing, a Manual one waits for Start', async (t) => {
    const { state, ws, widgets, card } = stand(t, { tabs: [placed('quota')] });
    widgets.show();
    await flush();
    state.tabs = [placed('quota'), placed('joined'), placed('waits', { start: 'manual' })];
    ws.emit('extension_lifecycle', { skill: 'joined' });
    await flush();
    assert.deepEqual(state.log, ['+quota@r1', '+joined@r1'], 'the joined Auto card starts; the one that ran is untouched');
    assert.equal(card('waits').querySelector('[data-widget-power]').textContent, 'Start');
    assert.ok(card('waits').querySelector('[data-widget-facade]'));
});

// A frame whose stop the test settles (the dispose handshake's acknowledgement).
const gatedStops = (stops) => (state, mount, tab, { policy }) => {
    const frame = document.createElement('iframe');
    mount.replaceChildren(frame);
    policy?.starting?.();
    state.log.push(`+${tab.skill}@${tab.revision}`);
    return Promise.resolve(() => new Promise((resolve) => stops.push(() => { state.log.push(`-${tab.skill}@${tab.revision}`); frame.remove(); resolve(); })));
};

test('a changed widget’s old card stands only as Stopping, with nothing to press, and acts on nothing of the fresh card of its key', async (t) => {
    const stops = [];
    const store = ledger();
    const { state, host, ws, widgets } = stand(t, { tabs: [placed('quota')], mountTab: gatedStops(stops), holds: store.open() });
    const cards = () => host.querySelectorAll('[data-widget-key="quota:main"]');
    widgets.show();
    await flush();
    state.tabs = [placed('quota', { revision: 'r2' })];
    ws.emit('extension_lifecycle', { skill: 'quota' });
    await flush();
    assert.equal(cards().length, 2, 'the old card is still in view while its frame stops');
    const [old, fresh] = cards();
    assert.ok(old.hasAttribute('data-widget-removed'), 'marked as the board marks a card on its way out');
    assert.equal(old.querySelector('[data-widget-status]').textContent, 'Stopping…');
    assert.equal(old.querySelector('[data-widget-power]').disabled, true);
    assert.equal(old.querySelector('[data-costs-widget-open]').disabled, true);
    assert.equal(fresh.hasAttribute('data-widget-removed'), false);
    assert.equal(fresh.querySelector('[data-widget-power]').disabled, false, 'the fresh card is the one that answers a press');
    assert.equal(host.querySelectorAll('iframe').length, 1, 'still one frame: the old one, until it has stopped');
    assert.ok(old.querySelector('[data-widget-mount] iframe'), 'it stays in its card for the dispose handshake');
    assert.equal(old.querySelector('[data-widget-mount]').hasAttribute('inert'), true, 'neither pointer nor keyboard reaches it');
    assert.equal(old.querySelector('[data-widget-status]').closest('[inert]'), null, 'Stopping stays readable');
    assert.equal(fresh.querySelector('[data-widget-mount]').hasAttribute('inert'), false);
    // A press that still reaches the old card changes nothing of the fresh one's widget: no stop, no hold, no door.
    old.querySelector('[data-widget-power]').click();
    old.querySelector('[data-costs-widget-open]').click();
    await flush();
    assert.deepEqual(state.opened, []);
    assert.deepEqual(state.log, ['+quota@r1']);
    assert.equal(stops.length, 1, 'no second stop was asked');
    assert.equal(JSON.parse(store.map.get(HOLDS_KEY) || '{"entries":{}}').entries['costs:quota:main'], undefined, 'the owner’s Stop was not made');
    stops.splice(0).forEach((settle) => settle());
    await flush();
    assert.equal(cards().length, 1, 'once stopped, the old card goes');
    assert.equal(host.querySelectorAll('iframe').length, 1, 'one frame per widget, never two');
    assert.deepEqual(state.log, ['+quota@r1', '-quota@r1', '+quota@r2']);
    assert.equal(cards()[0].querySelector('[data-widget-status]').textContent, 'Running');
    assert.equal(cards()[0].querySelector('[data-widget-power]').disabled, false);
    cards()[0].querySelector('[data-costs-widget-open]').click();
    assert.deepEqual(state.opened, [['quota', 'quota:main']], 'the live card’s door opens its own tab');
});

test('a vanished widget’s card also stands as Stopping with nothing to press until its frame has stopped, then goes', async (t) => {
    const stops = [];
    const { state, host, ws, widgets } = stand(t, { tabs: [placed('quota')], mountTab: gatedStops(stops) });
    widgets.show();
    await flush();
    state.tabs = [];
    ws.emit('extension_lifecycle', { skill: 'quota' });
    await flush();
    const [old] = host.querySelectorAll('[data-widget-key]');
    assert.equal(host.querySelectorAll('[data-widget-key]').length, 1);
    assert.equal(old.querySelector('[data-widget-status]').textContent, 'Stopping…');
    assert.equal(old.querySelector('[data-widget-power]').disabled, true);
    assert.equal(old.querySelector('[data-costs-widget-open]').disabled, true);
    stops.splice(0).forEach((settle) => settle());
    await flush();
    assert.equal(host.querySelectorAll('[data-widget-key]').length, 0);
    assert.deepEqual(state.log, ['+quota@r1', '-quota@r1']);
});

// Frames that report a content height of their own (as a document does once it is drawn) and record the height the
// card's facade stood at when they were asked for.
const heightsAtMount = (natural) => async (state, mount, tab, { policy }) => {
    state.heights = { ...state.heights, [tab.skill]: [...(state.heights?.[tab.skill] || []),
        mount.querySelector('[data-widget-facade]')?.style.getPropertyValue('--widget-frame-height')] };
    const frame = document.createElement('iframe');
    mount.replaceChildren(frame);
    policy.starting();
    policy.measured(natural, 0);
    return () => { frame.remove(); return Promise.resolve(); };
};

test('a resting card stands as tall as the frame it will get: under the author’s ceiling, as declared, compact while it waits for Start', async (t) => {
    const withRender = (skill, extra, options) => { const tab = placed(skill, options); Object.assign(tab.render, extra); return tab; };
    const { state, widgets, card } = stand(t, { mountTab: heightsAtMount(2000), tabs: [
        withRender('capped', { max_height: 480 }), withRender('roomy', {}), withRender('fixed', { height: 600 }),
        withRender('waits', {}, { start: 'manual' }) ] });
    const facade = (skill) => card(skill).querySelector('[data-widget-facade]')?.style.getPropertyValue('--widget-frame-height');
    widgets.show();
    await flush();
    assert.deepEqual([state.heights.capped, state.heights.roomy, state.heights.fixed], [['320px'], ['320px'], ['600px']], 'before a frame has reported: the loading height, or the declared one');
    assert.equal(facade('waits'), '96px', 'a card that waits for Start is a line, not a frame’s height');
    widgets.leave();
    await flush();
    // Each content reported 2000 px: a frame is given that only up to its ceiling, and so is the place held for it.
    assert.deepEqual(['capped', 'roomy', 'fixed', 'waits'].map(facade), ['480px', '2000px', '600px', '96px']);
    widgets.show();
    await flush();
    assert.deepEqual([state.heights.capped, state.heights.roomy, state.heights.fixed], [['320px', '480px'], ['320px', '2000px'], ['600px', '600px']],
        'a return starts at the height the frame will be given, with no jump to it');
});

test('each card is named by its own authored heading, including overlapping revisions of one key', async (t) => {
    const stops = [];
    const named = (key, title, revision = 'r1') => { const [skill, tab_id] = key.split(':'); return { ...placed(skill, { revision }), key, tab_id, title }; };
    const { state, host, ws, widgets } = stand(t, { tabs: [named('quota:work_day', 'Work day'), named('quota_work:day', 'Day')], mountTab: gatedStops(stops) });
    const names = () => host.querySelectorAll('[data-widget-key]').map((node) => {
        const heading = node.querySelector('.costs-widget-title');
        const id = node.getAttribute('aria-labelledby');
        assert.ok(id);
        assert.equal(heading.getAttribute('id'), id);
        assert.ok(heading.hasAttribute('data-i18n-authored'));
        assert.equal(node.hasAttribute('aria-label'), false, 'no second translatable copy of the name');
        assert.equal(node.querySelector('[data-costs-widget-open]').hasAttribute('data-i18n-authored'), false);
        return [node.dataset.widgetKey, heading.textContent];
    });
    widgets.show();
    await flush();
    assert.deepEqual(names(), [['quota:work_day', 'Work day'], ['quota_work:day', 'Day']]);
    state.tabs = [named('quota:work_day', 'Work day', 'r2'), named('quota_work:day', 'Day')];
    ws.emit('extension_lifecycle', { skill: 'quota' });
    await flush();
    assert.deepEqual(names(), [['quota:work_day', 'Work day'], ['quota:work_day', 'Work day'], ['quota_work:day', 'Day']],
        'the old card and its fresh one are both named while the frame stops');
    assert.equal(new Set(host.querySelectorAll('[id]').map((node) => node.getAttribute('id'))).size, 3);
});

test('Costs mounts through the board’s own dispatch: a kind it cannot draw is said, and nothing is left to stop', async (t) => {
    const dom = installCardDom();
    t.after(() => dom.restore());
    const mount = document.createElement('div');
    const mounted = await mountWidgetTab(mount, { skill: 'x', tab_id: 'main', render: { kind: 'module' } }, { signal: null, policy: null, messageHandlers: new Set(), key: 'costs:x:main' });
    assert.equal(mounted, null);
    assert.match(mount.textContent, /Widget render kind module is not supported yet\./);
});
