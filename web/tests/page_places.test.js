// What the owner scrolled keeps its place while hidden (page_places.js): a box shown again after `display: none`
// starts at the top in WebKit, so it is put back when its page, Dashboard tab or room is shown again. A room's feed
// and the Logs list keep their own place and are left alone.
import assert from 'node:assert/strict';
import test from 'node:test';

import { initPagePlaces } from '../modules/page_places.js';

function world() {
    const listeners = { doc: new Map(), win: new Map() };
    const on = (where) => (type, fn) => listeners[where].set(type, fn);
    const pages = new Map();
    const doc = { addEventListener: on('doc'), getElementById: (id) => pages.get(id) || null };
    const win = { addEventListener: on('win') };
    const restore = initPagePlaces({ doc, win });
    const box = (root, { owned = false } = {}) => {
        const node = { nodeType: 1, isConnected: true, scrollTop: 0, scrollLeft: 0, matches: () => owned };
        root.boxes.add(node);
        return node;
    };
    const page = (id) => {
        const root = { boxes: new Set(), contains: (node) => root.boxes.has(node) };
        pages.set(id, root);
        return root;
    };
    const scrolled = (node, top, left = 0) => {
        node.scrollTop = top;
        node.scrollLeft = left;
        listeners.doc.get('scroll')({ target: node });
    };
    return { listeners, page, box, scrolled, restore };
}

test('a box scrolled on a page goes back when the page or its tab is shown again; another page is untouched', () => {
    const w = world();
    const settings = w.page('page-settings'), tab = w.page('dashboard-panel-activity');
    const list = w.box(settings), panel = w.box(tab);
    w.scrolled(list, 400, 12);
    w.scrolled(panel, 90);
    list.scrollTop = 0; list.scrollLeft = 0; panel.scrollTop = 0;  // hidden and shown: WebKit starts at the top
    w.listeners.win.get('ouro:page-shown')({ detail: { page: 'settings' } });
    assert.deepEqual([list.scrollTop, list.scrollLeft, panel.scrollTop], [400, 12, 0]);
    w.listeners.win.get('ouro:dashboard-subtab-shown')({ detail: { tab: 'activity' } });
    assert.equal(panel.scrollTop, 90);
});

test('a kept room\'s inner box returns with the room; its feed and the Logs list own their place; a removed box is forgotten', () => {
    const w = world();
    const room = w.page('room');
    const timeline = w.box(room), feed = w.box(room, { owned: true });
    w.scrolled(timeline, 30);
    w.scrolled(feed, 700);
    timeline.scrollTop = 0; feed.scrollTop = 0;
    w.restore(room);
    assert.deepEqual([timeline.scrollTop, feed.scrollTop], [30, 0], 'the feed is its reading owner\'s');
    timeline.isConnected = false;
    timeline.scrollTop = 0;
    w.restore(room);
    assert.equal(timeline.scrollTop, 0, 'gone from the page: nothing to put back');
    assert.doesNotThrow(() => w.restore(null));
});

test('disposal releases a room\'s places even before another page restore', () => {
    const w = world(), room = w.page('room');
    const box = w.box(room);
    w.scrolled(box, 120);
    w.restore.forget(room);
    box.scrollTop = 0;
    w.restore(room);
    assert.equal(box.scrollTop, 0);
});

test('Logs owns its actual scroller; generic page restoration still restores other boxes', () => {
    const w = world(), tab = w.page('dashboard-panel-logs');
    // Match the real IDs against the production selector rather than pre-answering `matches` as owned.
    const byId = (id) => ({ nodeType: 1, isConnected: true, scrollTop: 0, scrollLeft: 0,
        matches: (selector) => selector.split(',').some(part => part.trim() === `#${id}`) });
    const logs = byId('page-logs'), other = byId('activity-list');
    tab.boxes.add(logs); tab.boxes.add(other);
    w.scrolled(logs, 900); w.scrolled(other, 40);
    logs.scrollTop = 0; other.scrollTop = 0;
    w.listeners.win.get('ouro:dashboard-subtab-shown')({ detail: { tab: 'logs' } });
    assert.deepEqual([logs.scrollTop, other.scrollTop], [0, 40], 'only logs.js restores #page-logs');
});
