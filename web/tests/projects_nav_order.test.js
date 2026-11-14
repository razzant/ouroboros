import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import vm from 'node:vm';
import { createProjectsNav, listOrder } from '../modules/projects_nav.js';
import { SHELL_ICONS, controlIcon } from '../modules/page_icons.js';
import { NodeStub, setup } from './dashboard_dom_fixture.js';

// The Projects list's order (docs/DESIGN.md "Navigation", owner 2026-11-07): the pins, then the newest message, the
// server's own order (projects_summary: time, then id, newest first) also where times are equal; unread is no rank.

const order = (rows) => [...rows].sort(listOrder).map((row) => row.id);

test('pins lead, then the newest message; equal times by id as the server lists them, unread no rank', () => {
    const archived = '2026-10-01T00:00:00Z';  // imported history keeps second-resolution times: equal times are real
    assert.deepEqual(order([
        { id: 'room-a', last_message_at: archived }, { id: 'room-c', last_message_at: archived },
        { id: 'pin', pinned: true, last_message_at: '2020-01-01T00:00:00Z' }, { id: 'room-b', last_message_at: archived },
        { id: 'newer', last_message_at: '2026-10-02T00:00:00Z' },
        { id: 'unread', last_message_at: '2026-09-01T00:00:00Z', _unread: true, visible_revision: 9 },
    ]), ['pin', 'newer', 'room-c', 'room-b', 'room-a', 'unread']);
});

test('an undated store orders by activity; rows with no time at all follow, by id as well', () => {
    assert.deepEqual(order([
        { id: 'b', last_active_at: '2026-01-02T00:00:00Z' }, { id: 'x' }, { id: 'a', last_active_at: '2026-01-03T00:00:00Z' },
        { id: 'y' },
    ]), ['a', 'b', 'y', 'x']);
});

// Where the keyboard goes when what held it inside the list is gone (a Project deleted or archived from its row menu,
// or closed while its row left a find's result): the list's first control, Find a project, or New project when the
// field is not shown; never the page body (the Projects header's toggle, the old target, is no longer drawn).
const appSource = readFileSync(new URL('../app.js', import.meta.url), 'utf8').replace(/\r\n?/g, '\n');
function liftApp(name) {
    const from = appSource.search(new RegExp(`\\nfunction ${name}\\(`));
    const until = appSource.indexOf('\n}\n', from + 1);
    assert.ok(from >= 0 && until > from, `${name} is a top-level function in app.js`);
    return appSource.slice(from + 1, until + 2);
}

test('the list\'s keyboard home is Find a project, or New project without the field; app.js sends a lost focus there', (t) => {
    const { document } = setup(t);
    const header = new NodeStub('div');
    const add = new NodeStub('button');
    add.id = 'nav-projects-add';
    header.append(add);
    header.prepend = function prepend(node) { this.insertBefore(node, this.children[0] || null); };
    const list = new NodeStub('div');
    list.id = 'nav-projects-list';
    const nav = createProjectsNav({ header, list, feed: () => [], feedOrder: () => 0, nextOrder: () => 1,
        reserved: () => [], listableCount: () => 0, activeId: () => null, isUnread: () => false,
        readList: async () => ({ projects: [] }), onListRead() {}, onShown() {}, open() {}, openMenu() {}, pin() {} });
    const input = header.querySelector('input');
    assert.equal(nav.keyboardHome(), input, 'the field leads the list');
    input.getClientRects = () => [];
    assert.equal(nav.keyboardHome(), add, 'a field not shown hands it to New project');
    input.getClientRects = () => [{ width: 1, height: 1 }];

    assert.match(appSource, /if \(shownProjectId && !activeProjectId\) focusAfterProjectClosed\(shownProjectId, focused\);/);
    const panel = new NodeStub('aside');
    const inPanel = new NodeStub('button');
    panel.append(inPanel);
    document.body.append(list, panel);
    const context = vm.createContext({ document, projectsNav: nav, projectPanel: panel, navProjectsList: list,
        focusedProjectRow: null });
    vm.runInContext(liftApp('focusAfterProjectClosed'), context);
    // No row ever held focus: a pointer reference to an unlisted room then × leaves WebKit's body alone.
    document.activeElement = document.body;
    context.focusAfterProjectClosed('p1', document.body);
    assert.equal(document.activeElement, document.body, 'a pointer reference must not summon Find or the keyboard');
    // Closed with a pointer while its row is still listed (WebKit focuses no pressed row, so the layer has no opener):
    // the focus is left where it fell, never put into the field.
    const row = new NodeStub('li');
    row.dataset.navProjectId = 'p1';
    row.setAttribute('data-nav-project-id', 'p1');
    list.append(row);
    for (const lost of [document.body, inPanel]) {
        document.activeElement = lost;
        context.focusAfterProjectClosed('p1', lost);
        assert.equal(document.activeElement, lost, `row listed, from ${lost.tagName}`);
    }
    // Establish the lost focus, rather than treating body focus as proof: this very row held it and was removed.
    const button = new NodeStub('button');
    Object.defineProperty(button, 'isConnected', { get: () => document.body.contains(button) });
    row.append(button);
    button.focus();
    assert.equal(document.activeElement, button);
    const focusBinding = appSource.match(/document.addEventListener\('focusin',[\s\S]*?\n\}\);/)[0];
    let rememberFocus;
    document.addEventListener = (type, listener) => { if (type === 'focusin') rememberFocus = listener; };
    vm.runInContext(focusBinding, context);
    rememberFocus({ target: button });
    row.remove();
    assert.equal(button.isConnected, false);
    document.activeElement = document.body; // the browser's result of removing its focused node
    context.focusAfterProjectClosed('p1', document.body);
    assert.equal(document.activeElement, input);
    context.focusedProjectRow = null;
    document.activeElement = inPanel;
    context.focusAfterProjectClosed('p1', inPanel);
    assert.equal(document.activeElement, input, 'a panel control whose opener row disappeared also goes home');
    // The focus that found a place of its own (the row back in the list, another control) is left where it is.
    const elsewhere = new NodeStub('button');
    document.body.append(elsewhere);
    document.activeElement = elsewhere;
    context.focusAfterProjectClosed('p1', inPanel);
    assert.equal(document.activeElement, elsewhere);
    // The stack layout's closed drawer is inert: nothing in it takes the focus.
    const sidebar = new NodeStub('nav');
    sidebar.setAttribute('inert', '');
    sidebar.append(header);
    document.activeElement = document.body;
    context.focusAfterProjectClosed('p1', inPanel);
    assert.equal(document.activeElement, document.body);

    // The row menu's fallback is the same place (app.js openProjectMenu hands it to project_create.js).
    let fallback;
    const menuContext = vm.createContext({
        projectsNav: { row: () => ({ id: 'p1', chat_id: 7, lifecycle: 'active' }), keyboardHome: () => input },
        projectActivityIndex: { byChat: new Map() }, IDLE_SUMMARY: { mark: '', owner: '' },
        openProjectRowMenu: (_project, options) => { fallback = options.fallbackFocus; return true; },
        apiClient: {}, markView: () => ({}), markWords: () => '', pendingProjectPins: new Set(),
    });
    vm.runInContext(liftApp('openProjectMenu'), menuContext);
    const item = new NodeStub('li');
    item.dataset.navProjectId = 'p1';
    assert.equal(menuContext.openProjectMenu(item), true);
    assert.equal(fallback, input);
    nav.destroy();
});

// Show more says how many rooms it adds wherever that is known (the owner's sweep of 14.11: one control, one wording):
// before the complete list is read too, from the state's count of listable rooms. Only a state that counts none (an
// older server) leaves the number unsaid. A find with no match offers its two different ways back each with its own
// glyph: Clear search the field's ×, Show all projects the chevron Show more opens the list with.
function listOf(t, { listableCount = 3, reserved = [] } = {}) {
    setup(t);
    // What the list's painting asks of a node beyond the shared stub, for this test's life.
    const added = {
        insertAdjacentHTML(_where, html) {
            const glyph = new NodeStub('svg');
            glyph.html = html;
            this.insertBefore(glyph, this.children[0] || null);
        },
        prepend(node) { this.insertBefore(node, this.children[0] || null); },
        toggleAttribute(name, on) { if (on) this.setAttribute(name, ''); else this.removeAttribute(name); return on; },
        getBoundingClientRect() { return { top: 0, bottom: 48, left: 0, right: 300, width: 300, height: 48 }; },
    };
    for (const [name, fn] of Object.entries(added)) {
        if (name in NodeStub.prototype) continue;
        NodeStub.prototype[name] = fn;
        t.after(() => { delete NodeStub.prototype[name]; });
    }
    const header = new NodeStub('div');
    header.prepend = function prepend(node) { this.insertBefore(node, this.children[0] || null); };
    const list = new NodeStub('div');
    list.getBoundingClientRect = () => ({ top: 0, bottom: 600, left: 0, right: 300, width: 300, height: 600 });
    const feed = ['a', 'b', 'c'].map((id, at) => ({ id, name: `Room ${id}`, chat_id: 10 + at,
        last_message_at: `2026-11-1${at}T00:00:00Z` }));
    const nav = createProjectsNav({ header, list, feed: () => feed, feedOrder: () => 0, nextOrder: () => 1,
        reserved: () => reserved, listableCount: () => listableCount, activeId: () => null, isUnread: () => false,
        readList: async () => ({ projects: feed }), onListRead() {}, onShown() {}, open() {}, openMenu() {}, pin() {},
        isPinPending: () => false });
    t.after(() => nav.destroy());
    const commands = () => list.children.filter((node) => node.className.includes('nav-projects-more'))
        .map((node) => [node.textContent.trim(), node.firstElementChild?.html || ''].filter(Boolean));
    return { nav, header, commands };
}

test('Show more names how many it adds before the list is read, as after', (t) => {
    const count = (options) => { const { nav, commands } = listOf(t, options); nav.paint(); return commands().map(([words]) => words); };
    assert.deepEqual(count({ listableCount: 60 }), ['Show 50 more'], 'a page of fifty');
    assert.deepEqual(count({ listableCount: 5 }), ['Show 2 more'], 'the last two');
    assert.deepEqual(count({ listableCount: 3 }), [], 'nothing beyond the feed');
    assert.deepEqual(count({ listableCount: null, reserved: [10, 11, 12, 13] }), ['Show more'], 'a state that counts no rooms');
});

test('a find with no match offers Clear search with ×, and Show all projects with the chevron', async (t) => {
    const { nav, header, commands } = listOf(t);
    const input = header.querySelector('input');
    input.value = 'zzqxw';
    input.dispatchEvent({ type: 'input' });
    await new Promise((resolve) => setTimeout(resolve, 0));
    nav.paint();
    const [clear, all] = commands();
    assert.equal(clear[0], 'Clear search');
    assert.equal(clear[1], SHELL_ICONS.close, 'the field\'s own ×');
    assert.equal(all[0], 'Show all projects');
    assert.equal(all[1], controlIcon('chevron', 18), 'the chevron Show more opens the list with');
});
