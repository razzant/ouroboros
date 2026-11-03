import test from 'node:test';
import assert from 'node:assert/strict';
import { createLayerStack } from '../modules/layers.js';

// The layer stack (P2): one dismiss for every layer, one history entry while any is open, inert outside a modal
// layer, focus into the layer and back to the opener. A fake DOM small enough to read: nodes with parents,
// children, `inert`, `contains` and `focus`; a window with listeners; a history that records its calls.

function makeDoc() {
    const doc = { activeElement: null };
    function node(tag, parent) {
        const n = {
            tagName: tag.toUpperCase(), children: [], parentElement: null, inert: false, isConnected: true, attrs: {},
            tabIndex: null, ownerDocument: doc,
            hasAttribute(name) { return name in this.attrs; },
            setAttribute(name, value) { this.attrs[name] = value; },
            contains(other) { for (let x = other; x; x = x.parentElement) if (x === this) return true; return false; },
            closest(selector) {
                if (selector !== '[inert]') return null;
                for (let x = this; x; x = x.parentElement) if (x.inert) return x;
                return null;
            },
            focus() { doc.activeElement = this; },
            querySelector() { return null; },
            getClientRects() { return this.shown === false ? [] : [{}]; },
            dispatchEvent(event) { for (let x = this; x; x = x.parentElement) x.onkey?.(event); return !event.defaultPrevented; },
        };
        if (parent) { n.parentElement = parent; parent.children.push(n); }
        return n;
    }
    doc.body = node('body');
    doc.body.querySelector = () => null;
    doc.querySelector = (selector) => (selector === 'dialog[open]' && doc.dialogOpen ? {} : null);
    doc.dialogs = [];
    doc.querySelectorAll = () => doc.dialogs;
    doc.activeElement = doc.body;
    return { doc, node };
}

function makeWin() {
    const listeners = new Map();
    const history = { calls: [], pushState(state, title) { this.calls.push(['push', state, title]); }, back() { this.calls.push(['back']); } };
    const win = {
        history, listeners,
        addEventListener(type, fn) { if (!listeners.has(type)) listeners.set(type, new Set()); listeners.get(type).add(fn); },
        removeEventListener(type, fn) { listeners.get(type)?.delete(fn); if (!listeners.get(type)?.size) listeners.delete(type); },
        fire(type, event = {}) { for (const fn of [...(listeners.get(type) || [])]) fn(event); },
        KeyboardEvent: class {
            constructor(type, init) { Object.assign(this, { type, defaultPrevented: false }, init); }
            preventDefault() { this.defaultPrevented = true; }
        },
    };
    return win;
}

const tick = () => new Promise((resolve) => setTimeout(resolve, 0));

function world() {
    const { doc, node } = makeDoc();
    const app = node('div', doc.body);
    const sidebar = node('nav', app);
    const content = node('main', app);
    const panel = node('aside', app);
    const toast = node('div', doc.body);
    const win = makeWin();
    const layers = createLayerStack({ win, doc, history: win.history });
    return { doc, node, app, sidebar, content, panel, toast, win, layers };
}

test('push, top, size, pop through the owner, remove is idempotent', () => {
    const w = world();
    const closed = [];
    assert.equal(w.layers.size(), 0);
    w.layers.push({ id: 'drawer', el: w.sidebar, onClose: (reason) => { closed.push(['drawer', reason]); w.layers.remove('drawer'); } });
    w.layers.push({ id: 'project', el: w.panel, modal: false, onClose: (reason) => closed.push(['project', reason]) });
    assert.deepEqual(w.layers.ids(), ['drawer', 'project']);
    assert.equal(w.layers.top(), 'project');
    assert.equal(w.layers.pop('escape'), true);
    // The owner that forgot to call remove is removed by the stack itself; the owner that did is not removed twice.
    assert.deepEqual(closed, [['project', 'escape']]);
    assert.equal(w.layers.size(), 1);
    assert.equal(w.layers.pop('scrim'), true);
    assert.deepEqual(w.layers.ids(), []);
    assert.equal(w.layers.pop(), false, 'nothing to pop');
    assert.equal(w.layers.remove('drawer'), false);
    // Pushing a known id keeps its place and takes new facts.
    w.layers.push({ id: 'a', el: w.sidebar, onClose() {} });
    w.layers.push({ id: 'b', el: w.panel, onClose() {} });
    w.layers.push({ id: 'a', el: w.sidebar, modal: false, onClose() {} });
    assert.deepEqual(w.layers.ids(), ['a', 'b']);
    w.layers.destroy();
});

test('Escape closes the top layer unless something inside took the key or a native dialog is open', () => {
    const w = world();
    const closed = [];
    w.layers.push({ id: 'drawer', el: w.sidebar, onClose: () => closed.push('drawer') });
    const prevented = { key: 'Escape', defaultPrevented: true, preventDefault() {} };
    w.win.fire('keydown', prevented);
    assert.deepEqual(closed, [], 'a menu that prevented default owns its Escape');
    w.doc.dialogOpen = true;
    w.win.fire('keydown', { key: 'Escape', defaultPrevented: false, preventDefault() {} });
    assert.deepEqual(closed, [], 'a native <dialog> owns Escape until it adapts');
    w.doc.dialogOpen = false;
    const event = { key: 'Escape', defaultPrevented: false, prevented: false, preventDefault() { this.prevented = true; } };
    w.win.fire('keydown', event);
    assert.deepEqual(closed, ['drawer']);
    assert.equal(event.prevented, true);
    assert.equal(w.layers.size(), 0);
    w.win.fire('keydown', { key: 'Escape', defaultPrevented: false, preventDefault() { throw new Error('nothing open'); } });
    w.layers.destroy();
});

test('one same-URL history entry while any layer is open; popstate pops one; a programmatic close consumes it', async () => {
    const w = world();
    const closed = [];
    const drawer = { id: 'drawer', el: w.sidebar, onClose: (reason) => { closed.push(['drawer', reason]); w.layers.remove('drawer'); } };
    const project = { id: 'project', el: w.panel, onClose: (reason) => { closed.push(['project', reason]); w.layers.remove('project'); } };
    w.layers.push(drawer);
    w.layers.push(project);
    await tick();
    assert.deepEqual(w.win.history.calls, [['push', { ouroLayers: 1 }, '']], 'two layers, one entry');
    assert.equal(w.layers.historyArmed(), true);
    // The system Back closes the top layer, and the entry is re-armed while the other stays open.
    w.win.fire('popstate');
    assert.deepEqual(closed, [['project', 'history']]);
    await tick();
    assert.deepEqual(w.win.history.calls.slice(1), [['push', { ouroLayers: 1 }, '']]);
    // Closing the last layer from a button consumes the entry; the popstate that causes is not a dismiss.
    w.layers.pop('close');
    assert.deepEqual(closed.at(-1), ['drawer', 'close']);
    await tick();
    assert.deepEqual(w.win.history.calls.at(-1), ['back']);
    assert.equal(w.layers.historyArmed(), false);
    w.win.fire('popstate');
    assert.equal(closed.length, 2, 'the swallowed popstate closed nothing');
    // "Close the drawer, open the Project" within one task leaves the one entry in place: no back, no extra push.
    const before = w.win.history.calls.length;
    w.layers.push(drawer);
    await tick();
    w.layers.remove('drawer');
    w.layers.push(project);
    await tick();
    assert.deepEqual(w.win.history.calls.slice(before), [['push', { ouroLayers: 1 }, '']]);
    assert.equal(w.layers.historyArmed(), true);
    // A document whose history refuses: layers still open and close.
    w.win.history.pushState = () => { throw new Error('sandboxed'); };
    w.layers.remove('project');
    await tick();
    w.layers.push(drawer);
    await tick();
    assert.equal(w.layers.size(), 1);
    w.layers.destroy();
});

test('a modal layer makes everything outside it inert; a non-modal one leaves the sidebar live', () => {
    const w = world();
    w.toast.inert = true;  // the page had already made this inert: the stack leaves it alone
    w.layers.push({ id: 'drawer', el: w.sidebar, modal: true, onClose() {} });
    assert.equal(w.sidebar.inert, false);
    assert.equal(w.content.inert, true);
    assert.equal(w.panel.inert, true);
    assert.equal(w.app.inert, false, 'the ancestor of the layer stays live');
    w.layers.remove('drawer');
    assert.equal(w.content.inert, false);
    assert.equal(w.panel.inert, false);
    assert.equal(w.toast.inert, true, 'what the page inerted stays inert');
    w.layers.push({ id: 'project', el: w.panel, modal: false, onClose() {} });
    assert.equal(w.sidebar.inert, false);
    assert.equal(w.content.inert, false);
    // The phone: the drawer (modal) opens over the Project layer; the panel below goes inert, the drawer stays
    // live, and so does what the layer names as live beside it (its scrim).
    const scrim = w.node('div', w.app);
    w.layers.push({ id: 'drawer', el: w.sidebar, modal: true, live: [scrim], onClose() {} });
    assert.equal(w.panel.inert, true);
    assert.equal(w.sidebar.inert, false);
    assert.equal(scrim.inert, false, 'the scrim stays tappable under the modal drawer');
    // A layer above a modal one stays live.
    w.layers.remove('drawer');
    w.layers.push({ id: 'project', el: w.panel, modal: true, onClose() {} });
    w.layers.push({ id: 'drawer', el: w.sidebar, modal: false, onClose() {} });
    assert.equal(w.sidebar.inert, false);
    assert.equal(w.content.inert, true);
    w.layers.destroy();
    assert.equal(w.content.inert, false, 'destroy releases what it inerted');
});

test('a page layer opened while the drawer is open lies under it: Escape closes the drawer, then the file', () => {
    // Files' file shown alone on a phone (files.js) is pushed when its read lands, which may be after the drawer opened.
    const w = world();
    const file = w.node('section', w.content);
    const closed = [];
    const drawer = { id: 'drawer', el: w.sidebar, modal: true, onClose: () => { closed.push('drawer'); w.layers.remove('drawer'); } };
    const page = { id: 'files-file', el: file, page: true, modal: false, focus: false, opener: null,
        onClose: () => { closed.push('file'); w.layers.remove('files-file'); } };
    w.layers.push(drawer);
    w.layers.push(page);
    assert.deepEqual(w.layers.ids(), ['files-file', 'drawer']);
    assert.equal(w.layers.top(), 'drawer');
    assert.equal(w.content.inert, true, 'the file under the modal drawer stays in the inert room');
    w.layers.push(page);  // a re-sync keeps its place
    assert.deepEqual(w.layers.ids(), ['files-file', 'drawer']);
    w.win.fire('keydown', { key: 'Escape', preventDefault() {} });
    assert.deepEqual(closed, ['drawer']);
    assert.deepEqual(w.layers.ids(), ['files-file']);
    assert.equal(w.content.inert, false);
    w.win.fire('keydown', { key: 'Escape', preventDefault() {} });
    assert.deepEqual(closed, ['drawer', 'file']);
    // Opened first, the file stays under a drawer opened over it; a layer without `page` still goes on top.
    w.layers.push(page);
    w.layers.push(drawer);
    w.layers.push({ id: 'menu', el: w.panel, modal: false, onClose() {} });
    assert.deepEqual(w.layers.ids(), ['files-file', 'drawer', 'menu']);
    w.layers.destroy();
});

// Every write of `inert` restyles the node's whole subtree (the room: ≈130 ms on a long feed in WebKit, owner 20.10 3A).
function countInertWrites(n) {
    let value = n.inert;
    const writes = [];
    Object.defineProperty(n, 'inert', { configurable: true, get: () => value, set: (v) => { writes.push(v); value = v; } });
    return writes;
}

test('a sync that keeps the same modal picture writes no inert; only what changes flips, another owner\'s stays theirs', () => {
    const w = world();
    w.toast.inert = true;  // the page's own: never written by the stack
    const content = countInertWrites(w.content);
    const panel = countInertWrites(w.panel);
    const toast = countInertWrites(w.toast);
    w.layers.push({ id: 'drawer', el: w.sidebar, modal: true, onClose() {} });
    assert.deepEqual([content, panel], [[true], [true]]);
    w.layers.push({ id: 'drawer', el: w.sidebar, modal: true, onClose() {} });  // the owner re-syncs, twice
    w.layers.push({ id: 'drawer', el: w.sidebar, modal: true, onClose() {} });
    assert.deepEqual([content, panel], [[true], [true]], 'the same picture writes nothing');
    // The Project opens over Main while the drawer closes: the room stays inert throughout, the panel turns live.
    w.layers.push({ id: 'project', el: w.panel, modal: true, onClose() {} });
    w.layers.remove('drawer');
    assert.deepEqual(content, [true], 'the room under both never flickered');
    assert.deepEqual(panel, [true, false]);
    assert.equal(w.sidebar.inert, true, 'the closed drawer is outside the Project now');
    w.layers.remove('project');
    assert.deepEqual(content, [true, false]);
    assert.deepEqual(toast, [], 'what the page inerted is never written');
    assert.equal(w.toast.inert, true);
    w.layers.destroy();
});

test('a sibling on the layer\'s own path that another owner made inert stays theirs through push, pop and destroy', () => {
    const w = world();
    const theirs = w.node('section', w.app);  // inside the shell, beside the room: the stack walks past it
    theirs.inert = true;
    const writes = countInertWrites(theirs);
    w.layers.push({ id: 'drawer', el: w.sidebar, modal: true, onClose() {} });
    assert.deepEqual([w.content.inert, w.panel.inert], [true, true], 'the ordinary siblings turn inert');
    w.layers.pop('escape');
    assert.deepEqual([w.content.inert, w.panel.inert], [false, false], 'and are released');
    assert.equal(theirs.inert, true, 'theirs stays inert');
    w.layers.push({ id: 'drawer', el: w.sidebar, modal: true, onClose() {} });
    w.layers.destroy();
    assert.deepEqual([w.content.inert, theirs.inert, writes], [false, true, []], 'never written: destroy releases only its own');
});

test('a batch settles once: one snapshot, one history entry, focus once; a replaced drawer leaves the room inert throughout', async () => {
    const w = world();
    const seen = [];
    w.layers.onChange((snapshot) => seen.push(snapshot.ids.join(',')));
    w.layers.push({ id: 'drawer', el: w.sidebar, modal: true, onClose() {} });
    await tick();
    const content = countInertWrites(w.content);
    w.layers.batch(() => {
        w.layers.remove('drawer');
        w.layers.push({ id: 'project', el: w.panel, modal: true, onClose() {} });
    });
    assert.deepEqual(seen, ['drawer', 'project'], 'one snapshot for the replacement: Telegram\'s Back never blinks');
    assert.deepEqual(content, [], 'the room under the drawer and then the Project was never released');
    assert.equal(w.doc.activeElement, w.panel, 'focus once, into the layer the batch opened');
    await tick();
    assert.deepEqual(w.win.history.calls.map((c) => c[0]), ['push'], 'one history entry throughout');
    // A batch that throws still settles what it changed, and the error reaches its caller.
    assert.throws(() => w.layers.batch(() => { w.layers.remove('project'); throw new Error('owner failed'); }), /owner failed/);
    assert.equal(seen.at(-1), '');
    assert.deepEqual(content, [false]);
    w.layers.destroy();
});

test('with inertAfter the stack, history, Back and a live focus change at once; inert and the focus it frees, after the motion', async () => {
    const w = world();
    const opener = w.node('button', w.content);
    opener.focus();
    const seen = [];
    w.layers.onChange((snapshot) => seen.push(snapshot.size));
    let opened;
    w.layers.batch(() => w.layers.push({ id: 'drawer', el: w.sidebar, modal: true, onClose() {} }),
        { inertAfter: new Promise((resolve) => { opened = resolve; }) });
    assert.deepEqual(seen, [1], 'Telegram\'s Back shows from the first frame');
    assert.equal(w.layers.has('drawer'), true, 'Escape and Back find the drawer at once');
    assert.equal(w.doc.activeElement, w.sidebar, 'the drawer is live: focus goes in at once');
    assert.equal(w.content.inert, false, 'the room is not written while the drawer moves');
    await tick();
    assert.deepEqual(w.win.history.calls.map((c) => c[0]), ['push']);
    opened();
    await tick();
    assert.equal(w.content.inert, true, 'the room is inert once the drawer has settled');
    // Closing: the opener sits in the room the pending difference releases, so focus returns after the motion.
    let closed;
    w.layers.batch(() => w.layers.remove('drawer'), { inertAfter: new Promise((resolve) => { closed = resolve; }) });
    assert.deepEqual(seen, [1, 0]);
    assert.equal(w.content.inert, true, 'still inert while the card glides home');
    w.sidebar.inert = true;  // the owner made the closed drawer inert at once; its focus is lost
    w.doc.activeElement = w.doc.body;
    closed();
    await tick();
    assert.equal(w.content.inert, false);
    assert.equal(w.doc.activeElement, opener, 'and focus is back on the opener');
    w.layers.destroy();
});

test('Tab while the inert difference waits applies it first, so focus never walks into the covered room', async () => {
    const w = world();
    let opened;
    w.layers.batch(() => w.layers.push({ id: 'drawer', el: w.sidebar, modal: true, onClose() {} }),
        { inertAfter: new Promise((resolve) => { opened = resolve; }) });
    assert.equal(w.content.inert, false, 'the room is not written while the drawer moves');
    w.win.fire('keydown', { key: 'a' });
    assert.equal(w.content.inert, false, 'other keys leave it to the motion');
    w.win.fire('keydown', { key: 'Tab' });
    assert.equal(w.content.inert, true, 'Tab applies it before the focus moves');
    const content = countInertWrites(w.content);
    opened();
    await tick();
    assert.deepEqual(content, [], 'the motion\'s end finds it applied and writes nothing again');
    w.win.fire('keydown', { key: 'Tab' });
    assert.deepEqual(content, [], 'nothing is pending any more');
    w.layers.destroy();
    assert.equal(w.win.listeners.size, 0);
});

test('a commit that moves no focus keeps the focus a superseded commit still owes; one that moves its own replaces it', async () => {
    const w = world();
    w.layers.push({ id: 'drawer', el: w.sidebar, modal: true, onClose() {} });
    const row = w.node('button', w.sidebar);
    row.focus();  // the Project's row, tapped in the drawer
    let beat;
    w.layers.batch(() => {
        w.layers.remove('drawer');
        w.layers.push({ id: 'project', el: w.panel, modal: true, onClose() {} });
    }, { inertAfter: new Promise((resolve) => { beat = resolve; }) });
    assert.equal(w.doc.activeElement, row, 'the panel lies in the room the drawer made inert: its focus waits');
    w.sidebar.inert = true;  // the owner made the closed drawer inert at once; the row's focus is lost
    w.doc.activeElement = w.doc.body;
    // The paint at the end of the beat commits the same stack again, for the close that follows.
    let close;
    w.layers.batch(() => w.layers.push({ id: 'project', el: w.panel, modal: true, onClose() {} }),
        { inertAfter: new Promise((resolve) => { close = resolve; }) });
    beat();
    await tick();
    assert.equal(w.doc.activeElement, w.doc.body, 'the superseded commit places nothing');
    close();
    await tick();
    assert.equal(w.panel.inert, false);
    assert.equal(w.doc.activeElement, w.panel, 'the focus it owed lands once the motion has ended');
    w.layers.destroy();
    // The other direction: a later commit that moves focus itself wins over the owed one.
    const v = world();
    const menuButton = v.node('button', v.content);
    menuButton.focus();
    v.layers.push({ id: 'drawer', el: v.sidebar, modal: true, onClose() {} });
    v.node('button', v.sidebar).focus();
    v.layers.batch(() => {
        v.layers.remove('drawer');
        v.layers.push({ id: 'project', el: v.panel, modal: true, onClose() {} });
    }, { inertAfter: new Promise(() => {}) });
    v.doc.activeElement = v.doc.body;
    v.layers.batch(() => v.layers.remove('project'));  // closed before its motion ended
    assert.equal(v.doc.activeElement, menuButton, 'focus goes back to the ☰, the closed Project is never focused');
    v.layers.destroy();
});

test('a layer that replaces another in one commit returns focus where the replaced one would have', () => {
    const w = world();
    const menuButton = w.node('button', w.content);
    menuButton.focus();
    w.layers.push({ id: 'drawer', el: w.sidebar, modal: true, onClose() {} });
    const row = w.node('button', w.sidebar);
    row.focus();  // the Project's row, tapped in the drawer
    w.layers.batch(() => {
        w.layers.remove('drawer');
        w.layers.push({ id: 'project', el: w.panel, modal: true, onClose() {} });
    });
    assert.equal(w.doc.activeElement, w.panel);
    w.sidebar.inert = true;  // the closed drawer, as the owner leaves it
    w.layers.remove('project');
    assert.equal(w.doc.activeElement, menuButton, 'back on the room\'s ☰, not lost in the closed drawer');
    // An ordinary push keeps the focused control as its opener.
    const other = w.node('button', w.content);
    other.focus();
    w.layers.push({ id: 'project', el: w.panel, modal: true, onClose() {} });
    w.layers.remove('project');
    assert.equal(w.doc.activeElement, other);
    w.layers.destroy();
});

test('a drawer and its replaced room hand focus back through both retired openers', () => {
    const w = world();
    const mainMenu = w.node('button', w.content);
    mainMenu.focus();
    w.layers.push({ id: 'drawer', el: w.sidebar, onClose() {} });
    const row = w.node('button', w.sidebar);
    row.focus();
    w.layers.batch(() => {
        w.layers.remove('drawer');
        w.layers.push({ id: 'project', el: w.panel, onClose() {} });
    });
    const roomMenu = w.node('button', w.panel);
    roomMenu.focus();
    w.layers.push({ id: 'drawer', el: w.sidebar, onClose() {} });
    row.focus();
    w.layers.batch(() => {
        w.layers.remove('drawer');
        w.layers.remove('project');
        w.layers.push({ id: 'project', el: w.panel, onClose() {} });
    });
    assert.equal(w.doc.activeElement, w.panel, 'the replacement room receives the keys');
    roomMenu.isConnected = false;  // the replaced chat instance is destroyed
    w.layers.pop('escape');
    assert.equal(w.doc.activeElement, mainMenu, 'the surviving opener receives them on close');
    w.layers.destroy();
});

test('a replacement keeps a focus held outside what it retired, and an opener its owner names', () => {
    const w = world();
    const row = w.node('button', w.sidebar);
    row.focus();
    w.layers.push({ id: 'project', el: w.panel, modal: false, onClose() {} });
    const field = w.node('textarea', w.content);  // Main's composer beside the docked Project
    field.focus();
    const replace = (facts = {}) => w.layers.batch(() => {
        w.layers.remove('project');
        w.layers.push({ id: 'project', el: w.panel, modal: false, onClose() {}, ...facts });
    });
    replace();
    w.layers.pop('escape');
    assert.equal(w.doc.activeElement, field, 'the live focus it was given, not the retired room\'s row');
    w.layers.push({ id: 'project', el: w.panel, modal: false, onClose() {} });
    const named = w.node('button', w.content);
    replace({ opener: named });
    w.layers.pop('escape');
    assert.equal(w.doc.activeElement, named, 'an opener the owner names wins over the chain');
    w.layers.destroy();
});

test('a later commit supersedes a pending inert difference; one without inertAfter applies now; destroy drops it', async () => {
    const w = world();
    let first;
    w.layers.batch(() => w.layers.push({ id: 'drawer', el: w.sidebar, modal: true, onClose() {} }),
        { inertAfter: new Promise((resolve) => { first = resolve; }) });
    let second;
    w.layers.batch(() => w.layers.remove('drawer'), { inertAfter: new Promise((resolve) => { second = resolve; }) });
    first();
    await tick();
    assert.equal(w.content.inert, false, 'the open that was overtaken writes nothing');
    second();
    await tick();
    assert.equal(w.content.inert, false);
    // A pending difference, then a commit with nothing to wait for: applied now, the pending one retired.
    let late;
    w.layers.batch(() => w.layers.push({ id: 'drawer', el: w.sidebar, modal: true, onClose() {} }),
        { inertAfter: new Promise((resolve) => { late = resolve; }) });
    w.layers.push({ id: 'drawer', el: w.sidebar, modal: true, onClose() {} });
    assert.equal(w.content.inert, true, 'applied at once');
    w.layers.batch(() => w.layers.remove('drawer'), { inertAfter: new Promise(() => {}) });
    w.layers.destroy();
    late();
    await tick();
    assert.equal(w.content.inert, false, 'destroy released it and the late promise applies nothing');
    // A rejected promise counts as the motion's end.
    const v = world();
    v.layers.batch(() => v.layers.push({ id: 'drawer', el: v.sidebar, modal: true, onClose() {} }),
        { inertAfter: Promise.reject(new Error('motion gone')) });
    await tick();
    assert.equal(v.content.inert, true);
    v.layers.destroy();
});

test('what is opened over the shell beside it (a dialog, a menu, a toast) stays live under a modal layer', () => {
    const w = world();
    const reconnect = w.node('div', w.doc.body);  // mounted before the layer opens
    w.layers.push({ id: 'project', el: w.panel, modal: true, onClose() {} });
    assert.equal(w.content.inert, true, 'the rest of the shell is inert');
    assert.equal(reconnect.inert, false);
    const dialog = w.node('div', w.doc.body);  // mounted while it is open, then the owner re-syncs its layer
    w.layers.push({ id: 'project', el: w.panel, modal: true, onClose() {} });
    assert.equal(dialog.inert, false);
    assert.equal(w.content.inert, true);
    w.layers.destroy();
});

test('Back gives a dialog over the layers its Escape and keeps the layer; with no dialog it pops', async () => {
    const w = world();
    const closed = [];
    w.layers.push({ id: 'drawer', el: w.sidebar, onClose: () => { closed.push('drawer'); w.layers.remove('drawer'); } });
    await tick();
    const dialog = w.node('div', w.doc.body);
    const field = w.node('input', dialog);
    const keys = [];
    dialog.onkey = (event) => { keys.push(event.key); event.preventDefault(); w.doc.dialogs = []; };
    w.doc.dialogs = [dialog];
    field.focus();
    w.win.fire('popstate');
    assert.deepEqual(keys, ['Escape'], 'the dialog, from its focused field, took the press');
    assert.deepEqual(closed, [], 'the drawer under it stays');
    await tick();
    assert.deepEqual(w.win.history.calls.at(-1), ['push', { ouroLayers: 1 }, ''], 'the entry is re-armed');
    // A hidden dialog is not on top; a bare native dialog that nothing answers is closed as its Escape would.
    const hidden = w.node('div', w.doc.body);
    hidden.shown = false;
    const native = w.node('dialog', w.doc.body);
    native.close = () => { native.closed = true; w.doc.dialogs = []; };
    w.doc.dialogs = [native, hidden];
    w.layers.back('host-back');
    assert.equal(native.closed, true);
    assert.deepEqual(closed, []);
    w.layers.back('host-back');
    assert.deepEqual(closed, ['drawer']);
    w.layers.destroy();
});

test('focus goes into the layer on open and back to the opener on close', () => {
    const w = world();
    const opener = w.node('button', w.content);
    opener.focus();
    w.layers.push({ id: 'drawer', el: w.sidebar, onClose() {} });
    assert.equal(w.sidebar.tabIndex, -1, 'a plain container becomes focusable');
    assert.equal(w.doc.activeElement, w.sidebar);
    w.layers.remove('drawer');
    assert.equal(w.doc.activeElement, opener);
    // A custom focus target, and an opener the page itself made inert meanwhile is not focused.
    const field = w.node('input', w.panel);
    w.layers.push({ id: 'project', el: w.panel, modal: false, focus: () => field.focus(), onClose() {} });
    assert.equal(w.doc.activeElement, field);
    w.content.inert = true;
    w.layers.remove('project');
    assert.equal(w.doc.activeElement, field, 'no focus into an inert opener');
    w.layers.destroy();
});

test('closing a layer leaves a focus the person moved beside it, and places a lost one', () => {
    const w = world();
    const row = w.node('button', w.sidebar);
    row.focus();
    w.layers.push({ id: 'project', el: w.panel, modal: false, onClose() {} });
    assert.equal(w.doc.activeElement, w.panel);
    const field = w.node('textarea', w.content);  // Main's composer beside the docked Project
    field.focus();
    w.layers.pop('escape');
    assert.equal(w.doc.activeElement, field, 'the Project closed; the field keeps the focus it was given');
    w.layers.push({ id: 'project', el: w.panel, modal: false, onClose() {} });
    w.doc.activeElement = w.doc.body;  // its room was put away under the focus
    w.layers.remove('project');
    assert.equal(w.doc.activeElement, field, 'a lost focus goes back to the opener');
    w.layers.destroy();
});

test('onChange reports the stack after every change and returns a disposer', () => {
    const w = world();
    const seen = [];
    const off = w.layers.onChange((snapshot) => seen.push([snapshot.size, snapshot.top]));
    w.layers.push({ id: 'a', el: w.sidebar, onClose() {} });
    w.layers.push({ id: 'b', el: w.panel, onClose() {} });
    w.layers.pop();
    w.layers.remove('a');
    assert.deepEqual(seen, [[1, 'a'], [2, 'b'], [1, 'a'], [0, null]]);
    off();
    w.layers.push({ id: 'a', el: w.sidebar, onClose() {} });
    assert.equal(seen.length, 4);
    w.layers.destroy();
    assert.equal(w.win.listeners.size, 0, 'destroy releases the window listeners');
});
