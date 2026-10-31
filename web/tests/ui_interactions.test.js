import assert from 'node:assert/strict';
import test from 'node:test';
import { bindTabStrip, renderTabStrip } from '../modules/page_header.js';
import { bindDialogFocus, bindEnterSubmit, bindMenu, bindPopoverPosition } from '../modules/ui_interactions.js';

// Small event/geometry double, following the existing DOM tests. Native Tab
// traversal, hit testing and the real confirm markup are checked in the browser.
class Events {
    listeners = new Map();
    addEventListener(type, listener) {
        if (!this.listeners.has(type)) this.listeners.set(type, new Set());
        this.listeners.get(type).add(listener);
    }
    removeEventListener(type, listener) { this.listeners.get(type)?.delete(listener); }
    emit(type, event = {}) {
        event.target ??= this;
        event.preventDefault ??= () => { event.defaultPrevented = true; };
        event.stopPropagation ??= () => { event.stopped = true; };
        for (const listener of [...(this.listeners.get(type) || [])]) listener(event);
        if (!event.stopped) this.parent?.emit(type, event);
        return event;
    }
    listenerCount() { return [...this.listeners.values()].reduce((n, set) => n + set.size, 0); }
}
class Node extends Events {
    constructor(doc, tag = 'div', attrs = {}) {
        super();
        this.nodeType = 1;
        this.ownerDocument = doc;
        this.tagName = tag;
        this.attrs = new Map(Object.entries(attrs));
        this.children = [];
        this.isConnected = true;
        this.disabled = false;
        this.rect = { left: 0, top: 0, right: 100, bottom: 30, width: 100, height: 30 };
        this.classes = new Set();
        this.classList = { toggle: (name, active) => active ? this.classes.add(name) : this.classes.delete(name) };
        this.css = new Map();
        this.style = { setProperty: (key, value) => this.css.set(key, value) };
    }
    get tabIndex() {
        return this.attrs.has('tabindex') ? Number(this.attrs.get('tabindex'))
            : ['button', 'input', 'select', 'textarea', 'summary', 'a'].includes(this.tagName) ? 0 : -1;
    }
    set tabIndex(value) { this.setAttribute('tabindex', String(value)); }
    setAttribute(key, value) { this.attrs.set(key, String(value)); }
    getAttribute(key) { return this.attrs.get(key) ?? null; }
    removeAttribute(key) { this.attrs.delete(key); }
    matches(selector) {
        return selector.split(',').some((part) => {
            part = part.trim();
            if (part === ':disabled') return this.disabled;
            const tag = part.match(/^[a-z]+/)?.[0];
            if (tag && tag !== this.tagName) return false;
            return [...part.matchAll(/\[([^=\]^]+)(\^?=)?(?:"([^"]*)")?\]/g)].every(([, key, op, value]) => {
                const actual = this.getAttribute(key);
                return op === '^=' ? actual?.startsWith(value) : op === '=' ? actual === value : actual !== null;
            });
        });
    }
    closest(selector) { return this.matches(selector) ? this : this.parent?.closest(selector) || null; }
    contains(node) { return this === node || this.children.some((child) => child.contains(node)); }
    querySelectorAll(selector) { return this.children.flatMap((child) => [...(child.matches(selector) ? [child] : []), ...child.querySelectorAll(selector)]); }
    append(...nodes) { for (const node of nodes) { node.parent = this; this.children.push(node); } }
    getClientRects() { return this.isConnected && !this.closest('[hidden]') ? [this.rect] : []; }
    getBoundingClientRect() { return this.rect; }
    // The laid-out size, as a browser reports it without the element's own transform.
    get offsetWidth() { return this.rect.width ?? 0; }
    get offsetHeight() { return this.rect.height ?? 0; }
    focus() { if (this.isConnected) this.ownerDocument.activeElement = this; }
}
function environment() {
    const doc = new Events();
    class Observer { observe() {} disconnect() {} }
    doc.defaultView = Object.assign(new Events(), {
        innerWidth: 320, innerHeight: 240,
        ResizeObserver: Observer, MutationObserver: Observer,
        requestAnimationFrame: () => 1, cancelAnimationFrame() {},
        getComputedStyle: (node) => ({ visibility: node.visibility || 'visible',
            scrollPaddingInlineStart: node.scrollPadding || '0px', scrollPaddingInlineEnd: node.scrollPadding || '0px' }),
    });
    doc.body = new Node(doc, 'body');
    doc.activeElement = doc.body;
    return { doc, node: (tag, attrs) => new Node(doc, tag, attrs) };
}
function tabFixture() {
    const { doc, node } = environment();
    const strip = node('div', { role: 'tablist' });
    const tabs = ['one', 'two', 'three'].map((value, i) => node('button', {
        role: 'tab', 'data-view': value, 'aria-selected': String(i === 0),
    }));
    strip.append(...tabs);
    const changed = [];
    const controller = bindTabStrip(strip, { dataAttr: 'data-view', onChange: (value) => changed.push(value) });
    return { doc, node, strip, tabs, changed, controller };
}

test('static tabs remain keyboard reachable until the binder owns roving focus', () => {
    const html = renderTabStrip({ dataAttr: 'data-view', active: 'absent', items: [
        { value: 'disabled', disabled: true }, { value: 'one', panelId: 'panel-one', tabId: 'tab-one' }, { value: 'two' },
    ] });
    assert.doesNotMatch(html, /tabindex=/);
    assert.equal(html.match(/aria-selected="true"/g).length, 1);
    assert.match(html, /id="tab-one"/);
    assert.match(html, /aria-controls="panel-one"/);
});

test('tabs synchronize arrows, Home/End, wrapping, ARIA, class and one load', () => {
    const { doc, strip, tabs, changed } = tabFixture();
    tabs[0].focus();
    assert.equal(strip.emit('keydown', { target: tabs[0], key: 'ArrowRight' }).defaultPrevented, true);
    assert.equal(doc.activeElement, tabs[1]);
    assert.deepEqual(tabs.map((tab) => [tab.tabIndex, tab.getAttribute('aria-selected'), tab.classes.has('active')]),
        [[-1, 'false', false], [0, 'true', true], [-1, 'false', false]]);
    strip.emit('keydown', { target: tabs[1], key: 'End' });
    strip.emit('keydown', { target: tabs[2], key: 'ArrowRight' });
    strip.emit('keydown', { target: tabs[0], key: 'Home' });
    assert.deepEqual(changed, ['two', 'three', 'one']);
});

test('native activation has one owner; external select is silent and never steals focus', () => {
    const { doc, strip, tabs, changed, controller } = tabFixture();
    tabs[0].focus();
    strip.emit('keydown', { target: tabs[1], key: 'Enter' });
    assert.deepEqual(changed, []);
    strip.emit('click', { target: tabs[1] });
    strip.emit('click', { target: tabs[1] });
    assert.deepEqual(changed, ['two']);
    assert.equal(controller.select('three'), true);
    assert.equal(doc.activeElement, tabs[0]);
    assert.deepEqual(changed, ['two']);
    strip.setAttribute('inert', '');
    assert.equal(controller.select('one'), true, 'programmatic route selection succeeds while drawer closes over inert page');
    strip.emit('click', { target: tabs[2] });
    assert.equal(tabs[0].getAttribute('aria-selected'), 'true', 'user clicks on an inert strip are ignored');
    strip.removeAttribute('inert');
    assert.equal(controller.select('unknown'), false);
    controller.destroy();
    assert.equal(controller.select('one', { focus: true }), false);
    strip.emit('click', { target: tabs[0] });
    assert.equal(strip.listenerCount(), 0);
    assert.deepEqual(changed, ['two']);
});

test('external tab selection works under a hidden or inert page without activating or focusing it', () => {
    for (const attribute of ['hidden', 'inert']) {
        const { doc, node, strip, tabs, changed, controller } = tabFixture();
        const page = node('main', { [attribute]: '' });
        page.append(strip);
        const before = doc.activeElement;
        assert.equal(controller.select('two'), true);
        assert.equal(tabs[1].getAttribute('aria-selected'), 'true');
        assert.equal(doc.activeElement, before);
        assert.equal(controller.select('three', { focus: true }), false);
        strip.emit('click', { target: tabs[2] });
        strip.emit('keydown', { target: tabs[1], key: 'ArrowRight' });
        assert.deepEqual(changed, []);
        assert.equal(tabs[1].getAttribute('aria-selected'), 'true');
        for (const attr of ['hidden', 'inert', 'aria-disabled']) {
            tabs[2].setAttribute(attr, attr === 'aria-disabled' ? 'true' : '');
            assert.equal(controller.select('three'), false);
            tabs[2].removeAttribute(attr);
        }
        tabs[2].disabled = true;
        assert.equal(controller.select('three'), false);
        controller.destroy();
    }
});

test('tab keys skip disabled/hidden choices, ignore nested strips and modifiers', () => {
    const { strip, tabs, node, changed } = tabFixture();
    tabs[1].disabled = true;
    strip.emit('keydown', { target: tabs[0], key: 'ArrowRight' });
    assert.deepEqual(changed, ['three']);
    strip.emit('keydown', { target: tabs[2], key: 'ArrowLeft', ctrlKey: true });
    assert.deepEqual(changed, ['three']);
    const nested = node('div', { role: 'tablist' });
    const other = node('button', { role: 'tab', 'data-view': 'other' });
    nested.append(other); strip.append(nested);
    strip.emit('click', { target: other });
    assert.deepEqual(changed, ['three']);
    tabs[2].setAttribute('hidden', '');
    strip.emit('keydown', { target: tabs[0], key: 'ArrowRight' });
    assert.deepEqual(changed, ['three', 'one']);
});

test('a tab domain callback may destroy its binding without a later focus or load', () => {
    const { strip, tabs, controller } = tabFixture();
    controller.destroy();
    let calls = 0;
    const binding = bindTabStrip(strip, { dataAttr: 'data-view', onChange: () => { calls++; binding.destroy(); } });
    strip.emit('keydown', { target: tabs[0], key: 'End' });
    strip.emit('keydown', { target: tabs[2], key: 'Home' });
    assert.equal(calls, 1);
    assert.equal(strip.listenerCount(), 0);
});

test('synchronous focus teardown prevents a late tab load callback', () => {
    const { strip, tabs, changed, controller } = tabFixture();
    tabs[1].focus = () => controller.destroy();
    strip.emit('keydown', { target: tabs[0], key: 'ArrowRight' });
    assert.deepEqual(changed, []);
});

test('restored/offscreen tab scrolls only its strip, and hidden-page resize observation is disposed', () => {
    const { doc, strip, tabs, controller, changed } = tabFixture();
    controller.destroy();
    let resize, disconnected = false;
    doc.defaultView.ResizeObserver = class {
        constructor(callback) { resize = callback; }
        observe() {}
        disconnect() { disconnected = true; }
    };
    strip.scrollLeft = 0; strip.scrollTop = 11; strip.clientWidth = 0; strip.clientLeft = 0;
    strip.rect = { left: 20, right: 220 };
    tabs[2].rect = { left: 300, right: 370 };
    tabs[0].focus();
    const binding = bindTabStrip(strip, { dataAttr: 'data-view', onChange: () => changed.push('load') });
    binding.select('three');
    assert.equal(strip.scrollLeft, 0);
    strip.clientWidth = 200;
    resize();
    assert.equal(strip.scrollLeft, 150);
    assert.equal(strip.scrollTop, 11);
    assert.equal(doc.activeElement, tabs[0]);
    assert.deepEqual(changed, []);
    binding.destroy();
    resize();
    assert.equal(strip.scrollLeft, 150);
    assert.equal(disconnected, true);
});

test('revealed tabs keep the strip padding clear of the edge fade', () => {
    const { strip, tabs, controller } = tabFixture();
    strip.clientWidth = 200; strip.clientLeft = 0; strip.scrollLeft = 0;
    strip.scrollPadding = '16px'; strip.rect = {left: 20, right: 220};
    tabs[2].rect = {left: 300, right: 370};
    controller.select('three');
    assert.equal(strip.scrollLeft, 166);
    controller.destroy();
});

function dialogFixture() {
    const { doc, node } = environment();
    const trigger = node('button');
    const dialog = node('div', { role: 'dialog', 'aria-modal': 'true' });
    const first = node('button'), input = node('input'), last = node('button');
    dialog.append(first, input, last);
    doc.body.append(trigger, dialog);
    trigger.focus();
    return { doc, node, trigger, dialog, first, input, last };
}

test('dialog traps both boundaries, respects native interior Tab and restores its trigger once', () => {
    const { doc, trigger, dialog, first, input, last } = dialogFixture();
    const dispose = bindDialogFocus(dialog, { initialFocus: input });
    assert.equal(doc.activeElement, input);
    assert.equal(dialog.emit('keydown', { target: input, key: 'Tab' }).defaultPrevented, undefined);
    last.focus(); dialog.emit('keydown', { target: last, key: 'Tab' });
    assert.equal(doc.activeElement, first);
    dialog.emit('keydown', { target: first, key: 'Tab', shiftKey: true });
    assert.equal(doc.activeElement, last);
    dispose();
    assert.equal(doc.activeElement, trigger);
    assert.equal(dialog.listenerCount(), 0);
    input.focus(); dispose();
    assert.equal(doc.activeElement, input);
});

test('dialog computes current enabled controls and supports an empty focus cycle', () => {
    const { doc, dialog, first, input, last } = dialogFixture();
    first.disabled = true;
    last.setAttribute('hidden', '');
    const dispose = bindDialogFocus(dialog);
    assert.equal(doc.activeElement, input);
    input.disabled = true; dialog.focus();
    assert.equal(dialog.emit('keydown', { target: dialog, key: 'Tab' }).defaultPrevented, true);
    assert.equal(doc.activeElement, dialog);
    dispose();
    assert.equal(dialog.getAttribute('tabindex'), null);
});

test('dialog Escape is caller-owned and disposal cannot steal focus from another surface', () => {
    const { doc, node, trigger, dialog, input } = dialogFixture();
    let cancelled = 0;
    const dispose = bindDialogFocus(dialog, { onEscape: () => cancelled++ });
    dialog.emit('keydown', { target: input, key: 'Escape', isComposing: true });
    assert.equal(cancelled, 0);
    const event = dialog.emit('keydown', { target: input, key: 'Escape' });
    assert.equal(cancelled, 1);
    assert.equal(event.defaultPrevented, true);
    const next = node('button'); next.focus();
    dispose();
    assert.equal(doc.activeElement, next);
    assert.notEqual(doc.activeElement, trigger);
});

test('a removed trigger is never focused on close; nested dialog owns its own keys', () => {
    const { doc, node, trigger, dialog, input } = dialogFixture();
    const dispose = bindDialogFocus(dialog, { initialFocus: input });
    const nested = node('div', { 'aria-modal': 'true' });
    const button = node('button'); nested.append(button); dialog.append(nested);
    let closed = 0;
    const disposeNested = bindDialogFocus(nested, { onEscape: () => closed++ });
    nested.emit('keydown', { target: button, key: 'Escape' });
    assert.equal(closed, 1);
    disposeNested();
    assert.equal(doc.activeElement, input);
    trigger.isConnected = false;
    dispose();
    assert.equal(doc.activeElement, input);
});

function menuFixture() {
    const { doc, node } = environment();
    const anchor = node('button');
    anchor.rect = { left: 275, right: 315, top: 208, bottom: 238, width: 40, height: 30 };
    const menu = node('div', { role: 'menu' });
    menu.rect = { width: 170, height: 100 };
    const options = ['rename', 'disabled', 'delete'].map((id) => node('button', { role: 'menuitem', id }));
    options[1].disabled = true;
    menu.append(...options); doc.body.append(anchor, menu);
    const closed = [];
    const binding = bindMenu(menu, { anchor, onClose: (event) => closed.push(event.reason) });
    return { doc, node, anchor, menu, options, closed, binding };
}

test('menu fits a short/narrow viewport and keyboard wraps over enabled actions', () => {
    const { doc, menu, options, binding } = menuFixture();
    assert.equal(menu.css.get('--ui-popup-left'), '142px');
    assert.equal(menu.css.get('--ui-popup-top'), '104px');
    assert.equal(menu.css.get('--ui-popup-max-height'), '196px');
    assert.equal(menu.css.get('--ui-popup-origin'), '153px 100px', 'opening above its anchor, it grows from the anchor\'s middle at its foot');
    assert.equal(doc.activeElement, options[0]);
    menu.emit('keydown', { target: options[0], key: 'ArrowUp' });
    assert.equal(doc.activeElement, options[2]);
    menu.emit('keydown', { target: options[2], key: 'ArrowDown' });
    assert.equal(doc.activeElement, options[0]);
    menu.emit('keydown', { target: options[0], key: 'End' });
    assert.equal(doc.activeElement, options[2]);
    binding.destroy();
});

test('Escape closes once, restores trigger, and removes every owned listener', () => {
    const { doc, anchor, menu, options, closed, binding } = menuFixture();
    menu.emit('keydown', { target: options[0], key: 'Escape' });
    binding.close();
    assert.deepEqual(closed, ['escape']);
    assert.equal(doc.activeElement, anchor);
    assert.equal(doc.listenerCount() + doc.defaultView.listenerCount() + menu.listenerCount(), 0);
    binding.reposition();
});

test('outside and owner teardown do not refocus the trigger or dispatch an action', () => {
    const { doc, node, anchor, closed, binding } = menuFixture();
    const elsewhere = node('button');
    doc.emit('pointerdown', { target: anchor });
    assert.deepEqual(closed, []);
    elsewhere.focus();
    doc.emit('pointerdown', { target: elsewhere });
    assert.deepEqual(closed, ['outside']);
    assert.equal(doc.activeElement, elsewhere);
    binding.destroy();
    assert.equal(doc.listenerCount() + doc.defaultView.listenerCount(), 0);
    const other = menuFixture();
    other.binding.destroy();
    assert.deepEqual(other.closed, []);
});

test('scroll inside the menu stays open; scrolling the page closes without a focus jump', () => {
    const { doc, anchor, menu, closed } = menuFixture();
    doc.defaultView.emit('scroll', { target: menu });
    doc.defaultView.emit('scroll');
    assert.deepEqual(closed, [], 'a queued event without movement does not close a just-opened menu');
    // A row lifting under its long-press menu grows about its middle: the feed reflowing under it keeps the menu.
    anchor.rect = { left: 274.5, right: 315.5, top: 207.8, bottom: 238.2, width: 41, height: 30.4 };
    doc.defaultView.emit('scroll');
    assert.deepEqual(closed, [], 'a lifted anchor has not moved');
    anchor.rect = { ...anchor.rect, top: anchor.rect.top - 10 };
    doc.defaultView.emit('scroll');
    assert.deepEqual(closed, ['scroll']);
});

test('window blur preserves Files dismissal without restoring focus, then removes its listener', () => {
    const { doc, menu, options, closed } = menuFixture();
    doc.defaultView.emit('blur');
    doc.defaultView.emit('blur');
    assert.deepEqual(closed, ['blur']);
    assert.equal(doc.activeElement, options[0]);
    assert.equal(doc.listenerCount() + doc.defaultView.listenerCount() + menu.listenerCount(), 0);
});

test('action handoff restores before opening the next dialog, never after it', () => {
    const { doc, node, anchor, menu, binding } = menuFixture();
    binding.destroy();
    const nextInput = node('input');
    const next = bindMenu(menu, { anchor, onClose: () => nextInput.focus() });
    next.close({ restoreFocus: true });
    assert.equal(doc.activeElement, nextInput);
    next.close({ restoreFocus: true });
    assert.equal(doc.activeElement, nextInput);
});

test('Tab leaves a portalled menu from its trigger, even when close removes the focused item', () => {
    const { doc, node, anchor, menu, options, binding } = menuFixture();
    binding.destroy();
    const before = node('button');
    const after = node('button');
    doc.body.children = [];
    doc.body.append(before, anchor, after, menu);
    const next = bindMenu(menu, { anchor, onClose: () => { menu.isConnected = false; } });
    const event = menu.emit('keydown', { target: options[0], key: 'Tab', shiftKey: true });
    assert.equal(event.defaultPrevented, true);
    assert.equal(doc.activeElement, before);
    assert.equal(menu.isConnected, false);
    next.destroy();
});

test('listbox positioning follows its input without claiming keyboard, ARIA or focus', () => {
    const { doc, node } = environment();
    const input = node('input');
    input.rect = { left: 28, right: 268, top: 30, bottom: 60, width: 240, height: 30 };
    const popup = node('div', { role: 'listbox' });
    popup.rect = { width: 240, height: 100 };
    doc.body.append(input, popup); input.focus();
    const binding = bindPopoverPosition(popup, { anchor: input });
    assert.equal(popup.css.get('--ui-popup-left'), '28px');
    assert.equal(popup.css.get('--ui-popup-top'), '64px');
    assert.equal(popup.css.get('--ui-popup-anchor-width'), '240px');
    assert.equal(doc.activeElement, input);
    assert.equal(input.listenerCount() + popup.listenerCount() + doc.listenerCount(), 0);
    assert.equal(popup.getAttribute('role'), 'listbox');
    input.rect.bottom = 80;
    doc.defaultView.emit('scroll');
    assert.equal(popup.css.get('--ui-popup-top'), '84px');
    binding.destroy();
    input.rect.bottom = 90;
    binding.reposition();
    assert.equal(popup.css.get('--ui-popup-top'), '84px');
    assert.equal(doc.defaultView.listenerCount(), 0);
});

test('popup uses visual viewport bounds and point coordinates without an input owner', () => {
    const { doc, node } = environment();
    doc.defaultView.visualViewport = Object.assign(new Events(), { offsetLeft: 20, offsetTop: 30, width: 260, height: 180 });
    const popup = node('div'); popup.rect = { width: 120, height: 80 };
    const binding = bindPopoverPosition(popup, { point: { x: 300, y: 230 } });
    assert.equal(popup.css.get('--ui-popup-left'), '152px');
    assert.equal(popup.css.get('--ui-popup-top'), '122px');
    assert.equal(popup.css.get('--ui-popup-origin'), '120px 80px', 'it grows from the press, clamped to its own box');
    binding.destroy();
    assert.equal(doc.defaultView.listenerCount() + doc.defaultView.visualViewport.listenerCount(), 0);
});

test('a resized page cannot make an offscreen input grant more popup height than the viewport', () => {
    const { doc, node } = environment();
    const input = node('input');
    const popup = node('div', { role: 'listbox' });
    popup.rect = { width: 240, height: 1120 };
    const binding = bindPopoverPosition(popup, { anchor: input });
    for (const top of [-1300, 1300]) {
        input.rect = { left: 28, right: 268, top, bottom: top + 30, width: 240, height: 30 };
        binding.reposition();
        assert.equal(popup.css.get('--ui-popup-max-height'), '224px');
        assert.equal(popup.css.get('--ui-popup-top'), '8px');
    }
    binding.destroy();
});

function messageFieldFixture() {
    const { node } = environment();
    const field = node('textarea');
    const send = node('button');
    let sent = 0;
    send.click = () => { sent += 1; };
    const dispose = bindEnterSubmit(field, send);
    const press = (init = {}) => field.emit('keydown', { key: 'Enter', ...init });
    return { field, send, dispose, press, sent: () => sent };
}

test('a message field sends on Enter through its button; Shift+Enter keeps the line break', () => {
    const { field, press, sent } = messageFieldFixture();
    assert.equal(field.enterKeyHint, 'send', 'the touch keyboard labels the key Send');
    assert.equal(press().defaultPrevented, true);
    assert.equal(sent(), 1);
    assert.equal(press({ shiftKey: true }).defaultPrevented, undefined, 'the native line break stays');
    assert.equal(sent(), 1);
    // The composer's existing chords keep sending.
    for (const chord of ['ctrlKey', 'metaKey', 'altKey']) press({ [chord]: true });
    assert.equal(sent(), 4);
    assert.equal(field.emit('keydown', { key: 'a' }).defaultPrevented, undefined);
    assert.equal(sent(), 4);
});

test('composition, a held key and an unavailable field or button send nothing', () => {
    const { field, send, dispose, press, sent } = messageFieldFixture();
    // Chromium/Firefox mark the composing keydown; WebKit ends the composition
    // first and marks the committing Enter only with keyCode 229.
    assert.equal(press({ isComposing: true }).defaultPrevented, undefined);
    assert.equal(press({ keyCode: 229 }).defaultPrevented, undefined);
    assert.equal(press({ repeat: true }).defaultPrevented, true, 'a held key adds no line breaks');
    assert.equal(press({ defaultPrevented: true, preventDefault() {} }).defaultPrevented, true);
    assert.equal(sent(), 0);
    send.disabled = true;
    assert.equal(press().defaultPrevented, true, 'an empty or pending message is not turned into a line');
    send.disabled = false;
    send.isConnected = false;
    press();
    send.isConnected = true;
    assert.equal(sent(), 0);
    for (const state of ['disabled', 'readOnly', 'isConnected']) {
        const previous = field[state];
        field[state] = state === 'isConnected' ? false : true;
        assert.equal(press().defaultPrevented, undefined, state);
        field[state] = previous;
    }
    assert.equal(sent(), 0);
    press();
    assert.equal(sent(), 1);
    dispose();
    assert.equal(field.listenerCount(), 0);
    assert.equal(field.enterKeyHint, undefined);
    press();
    assert.equal(sent(), 1);
});
