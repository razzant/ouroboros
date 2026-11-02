import './fixtures/call_modules.js';
// The live call's two sizes through the real capsule (call_capsule.js, call_stage.js, call_aura.js) on the
// DOM double, given a small HTML reader for the capsule's own markup. Geometry, the passage's motion and the
// look are checked in the browser lane (tests/test_live_conversation_browser.py).
import test from 'node:test';
import assert from 'node:assert/strict';
import { installDomDouble } from './dom_double.js';
const { createCallCapsule } = await import('../../skills/live_conversation/ui/call_capsule.js');
const { panelBox } = await import('../../skills/live_conversation/ui/call_stage.js');
const { CALL_VISUAL } = await import('../../skills/live_conversation/ui/call_aura.js');

const VOID = new Set(['path', 'rect', 'circle', 'br', 'input']);

/** Elements whose innerHTML builds children: tags, attributes, text and self-closing SVG. */
function readable(doc) {
    const make = doc.createElement;
    const dress = (node) => {
        const css = node.css;
        node.style = Object.assign(Object.create(null), {
            setProperty: (key, value) => css.set(key, value),
            getPropertyValue: key => css.get(key) ?? '',
            removeProperty: (key) => { css.delete(key); delete node.style[key]; },
        });
        node.cloneNode = (deep = false) => {
            const copy = doc.createElement(node.tagName);
            copy.className = node.className;
            Object.assign(copy.dataset, node.dataset);
            if (deep) copy.innerHTML = node.innerHTML;
            return copy;
        };
        Object.defineProperty(node, 'innerHTML', {
            configurable: true,
            get() { return this.html || ''; },
            set(value) {
                for (const child of [...this.children]) this.removeChild(child);
                this._text = '';
                this.html = String(value);
                const stack = [this];
                for (const m of this.html.matchAll(/<(\/?)([a-zA-Z][\w-]*)((?:\s+[\w:-]+(?:="[^"]*")?)*)\s*(\/?)>|([^<]+)/g)) {
                    const [, closing, tag, attrs, selfClosing, text] = m;
                    if (text !== undefined) { if (text.trim()) stack.at(-1)._text += text; continue; }
                    if (closing) { stack.pop(); continue; }
                    const child = dress(make(tag));
                    for (const [, key, raw] of attrs.matchAll(/([\w:-]+)(?:="([^"]*)")?/g)) {
                        const value = raw ?? '';
                        if (key === 'class') child.className = value;
                        else if (key === 'hidden') child.hidden = true;
                        else if (key.startsWith('data-')) child.dataset[key.slice(5).replace(/-([a-z])/g, (_a, c) => c.toUpperCase())] = value;
                        else child.setAttribute(key, value);
                    }
                    stack.at(-1).append(child);
                    if (!selfClosing && !VOID.has(tag)) stack.push(child);
                }
            },
        });
        return node;
    };
    doc.createElement = tag => dress(make(tag));
    return dress;
}

function harness({ reduce = false, width = 1280, vision = { screen: true, camera: false, cameraReason: 'No camera here' } } = {}) {
    const dom = installDomDouble();
    const { doc } = dom;
    const dress = readable(doc);
    globalThis.MutationObserver = class { observe() {} disconnect() {} };
    const timers = [];
    const env = Object.assign(doc.defaultView, {
        innerWidth: width, innerHeight: 900, devicePixelRatio: 1,
        setTimeout: (fn) => { timers.push(fn); return timers.length; },
        clearTimeout: (id) => { if (id) timers[id - 1] = null; },
        setInterval: () => 1, clearInterval() {},
        requestAnimationFrame: () => 0, cancelAnimationFrame() {},
        matchMedia: query => ({ matches: reduce && query.includes('reduce') }),
        getComputedStyle: () => ({ getPropertyValue: () => '' }),
        performance: { now: () => 0 },
    });
    const page = dress(doc.createElement('div'));
    page.innerHTML = '<div class="chat-input-wrap"><div class="chat-attachment-preview"></div><div class="chat-text-row">'
        + '<button class="chat-attach-btn"></button><textarea id="chat-input"></textarea>'
        + '<div class="chat-send-group"><button class="chat-send-inline">Send</button></div></div></div>';
    doc.body.append(page);
    const field = page.querySelector('textarea');
    Object.assign(field, { value: '', placeholder: 'Message Ouroboros...' });
    let session = { state: 'idle', muted: false, sharing: '', activity: { working: false, label: '' }, liveSince: 0, active: false };
    const calls = [];
    const voice = { heard: false, owner: false };  // what the light hears: Ouroboros's playback, the owner's gate
    const light = { heard: () => voice.heard, speaking: () => voice.owner, setMode() {} };
    const capsule = createCallCapsule({ page, env, visual: CALL_VISUAL, light, view: () => session,
        talk: () => calls.push('talk'), mute: () => calls.push('mute'), show: kind => calls.push(`show:${kind}`),
        vision: () => vision, end: () => calls.push('end'), dismiss: () => calls.push('dismiss'), act: () => {} });
    const wrap = page.querySelector('.chat-input-wrap');
    const row = page.querySelector('.chat-text-row');
    const $ = selector => page.querySelector(selector);
    const set = (facts) => {
        session = { ...session, ...facts };
        session.active = ['permission', 'admission', 'preparing', 'connecting', 'live', 'reconnecting', 'closing'].includes(session.state);
        capsule.render();
    };
    const flush = () => { while (timers.some(Boolean)) { const index = timers.findIndex(Boolean); const fn = timers[index]; timers[index] = null; fn(); } };
    const lifted = () => row.classList.contains('call-lifted');
    const spacer = () => page.querySelector('.call-spacer');
    return { dom, doc, env, page, wrap, row, field, capsule, calls, voice, $, set, flush, lifted, spacer, timers,
        done() { capsule.destroy(); dom.restore(); delete globalThis.MutationObserver; } };
}

// Every state in which the call shows its live capsule, as the session reports it.
const LIVE = {
    permission: { state: 'permission' },
    admission: { state: 'admission' },
    preparing: { state: 'preparing', phase: 'memory' },
    connecting: { state: 'connecting' },
    listening: { state: 'live' },
    muted: { state: 'live', muted: true },
    working: { state: 'live', activity: { working: true, label: 'Reading tasks' } },
    sharing: { state: 'live', sharing: 'screen' },
    reconnecting: { state: 'reconnecting', code: 'session_cap' },
    closing: { state: 'closing' },
};
const reset = { muted: false, sharing: '', phase: '', code: '', activity: { working: false, label: '' } };

test('the panel grows from the chip: over the composer on a wide window, the whole screen on a phone', () => {
    const chip = { left: 360, top: 740, width: 900, height: 50, bottom: 790 };
    assert.deepEqual(panelBox(chip, { width: 1280, height: 800 }), { left: 360, top: 290, width: 900, height: 500, radius: 28 });
    assert.deepEqual(panelBox(chip, { width: 1280, height: 600 }), { left: 360, top: 394, width: 900, height: 396, radius: 28 });
    assert.deepEqual(panelBox({ left: 16, top: 780, width: 358, height: 50, bottom: 830 }, { width: 390, height: 844 }),
        { left: 8, top: 8, width: 374, height: 828, radius: 30 });
    // A short window never pushes the panel above the screen.
    assert.equal(panelBox({ ...chip, top: 150, bottom: 200 }, { width: 1280, height: 900 }).top, 8);
});

test('expand and minimise work both ways from every live state; a sheet has no big view', () => {
    const h = harness();
    try {
        for (const [name, facts] of Object.entries(LIVE)) {
            h.set({ ...reset, liveSince: 1, ...facts });
            assert.equal(h.wrap.dataset.callView, 'voice', name);
            assert.equal(h.$('.call-size').hidden, false, name);
            assert.equal(h.$('.call-size').getAttribute('aria-label'), 'Expand the call', name);
            h.$('.call-size').click();
            assert.equal(h.wrap.dataset.callView, 'big', name);
            assert.equal(h.capsule.big, true, name);
            assert.ok(h.lifted() && h.spacer(), name);  // the row left the flow; the spacer keeps the composer's height
            assert.equal(h.$('.call-size').getAttribute('aria-expanded'), 'true', name);
            assert.equal(h.$('.call-size').getAttribute('aria-label'), 'Minimise the call', name);
            assert.equal(h.wrap.dataset.callMove, '1', name);  // the words wait while the glass grows
            h.flush();
            assert.equal(h.wrap.dataset.callMove, undefined, name);
            assert.ok(h.lifted(), name);
            h.$('.call-size').click();
            assert.equal(h.wrap.dataset.callView, 'voice', name);
            assert.equal(h.capsule.big, false, name);
            assert.ok(h.lifted(), name);  // the glass shrinks back to the chip's rectangle first
            h.flush();
            assert.ok(!h.lifted() && !h.spacer(), name);
            assert.equal(h.row.style.getPropertyValue('--call-panel-radius'), '', name);
            assert.equal(h.$('.call-size').getAttribute('aria-expanded'), 'false', name);
        }
        h.set({ ...reset, state: 'failed', code: 'socket_silent' });
        assert.equal(h.wrap.dataset.callView, 'sheet');
        assert.equal(h.$('.call-size').hidden, true);
        h.capsule.expand();
        assert.equal(h.wrap.dataset.callView, 'sheet');
        assert.ok(!h.lifted());
    } finally { h.done(); }
});

test('Escape minimises the big view and nothing else; a menu that took Escape keeps the big view', () => {
    const h = harness();
    try {
        h.set({ state: 'live', liveSince: 1 });
        h.doc.emit('keydown', { key: 'Escape' });
        assert.equal(h.wrap.dataset.callView, 'voice');
        assert.ok(!h.lifted());
        h.capsule.expand();
        h.flush();
        h.doc.emit('keydown', { key: 'Escape', defaultPrevented: true });
        assert.equal(h.wrap.dataset.callView, 'big');
        h.doc.emit('keydown', { key: 'Enter' });
        assert.equal(h.wrap.dataset.callView, 'big');
        h.doc.emit('keydown', { key: 'Escape' });
        assert.equal(h.wrap.dataset.callView, 'voice');
        h.flush();
        assert.ok(!h.lifted());
    } finally { h.done(); }
});

test('minimising keeps the composer corner even when the call has grown taller than one line', () => {
    const h = harness();
    try {
        h.env.getComputedStyle = () => ({ borderTopLeftRadius: '26px', getPropertyValue: () => '' });
        h.row.getBoundingClientRect = () => ({ left: 0, top: 600, width: 320, height: 100 });
        h.set({ state: 'live' });
        h.capsule.expand();
        h.flush();
        h.$('.call-size').click();
        assert.equal(h.row.style.getPropertyValue('--call-panel-radius'), '26px');
        h.flush();
        assert.equal(h.row.style.getPropertyValue('--call-panel-radius'), '');
    } finally { h.done(); }
});

test('Keyboard from the big view returns to the chip with the field; in the chip it still toggles', () => {
    const h = harness();
    try {
        h.set({ state: 'live', liveSince: 1 });
        h.capsule.expand();
        h.flush();
        assert.equal(h.$('.call-keyboard').hidden, false);
        h.$('.call-keyboard').click();
        assert.equal(h.wrap.dataset.callView, 'keyboard');
        assert.equal(h.capsule.big, false);
        assert.equal(h.doc.activeElement, h.field);
        assert.equal(h.field.placeholder, 'Message…');
        h.flush();
        assert.ok(!h.lifted() && !h.spacer());
        h.$('.call-keyboard').click();
        assert.equal(h.wrap.dataset.callView, 'voice');
        assert.equal(h.field.placeholder, 'Message Ouroboros...');
        h.$('.call-keyboard').click();
        assert.equal(h.wrap.dataset.callView, 'keyboard');
        // Expanding leaves the keyboard view: the big view never carries a field.
        h.capsule.expand();
        assert.equal(h.wrap.dataset.callView, 'big');
        assert.equal(h.field.placeholder, 'Message Ouroboros...');
    } finally { h.done(); }
});

test('End, Mute and Show are the same controls in the big view: End ends, Show opens its menu or stops', () => {
    const h = harness();
    try {
        h.set({ state: 'live', liveSince: 1 });
        h.capsule.expand();
        for (const control of ['.call-end', '.call-mute', '.call-show', '.call-keyboard', '.call-size'])
            assert.equal(h.$(control).hidden, false, control);
        h.$('.call-show').click();
        const menu = h.doc.body.querySelector('.call-show-menu');
        assert.ok(menu);
        assert.equal(h.$('.call-show').getAttribute('aria-expanded'), 'true');
        const camera = menu.querySelector('[data-source="camera"]');
        assert.equal(camera.disabled, true);
        assert.equal(camera.querySelector('small').textContent, 'No camera here');
        menu.querySelector('[data-source="screen"]').click();
        assert.deepEqual(h.calls, ['show:screen']);
        assert.equal(h.doc.body.querySelector('.call-show-menu'), null);
        h.set({ sharing: 'screen' });
        assert.equal(h.wrap.dataset.callView, 'big');
        assert.equal(h.$('.call-show').getAttribute('aria-pressed'), 'true');
        h.$('.call-show').click();  // a shown source: one press stops it, no menu
        assert.deepEqual(h.calls, ['show:screen', 'show:screen']);
        assert.equal(h.doc.body.querySelector('.call-show-menu'), null);
        h.$('.call-mute').click();
        h.$('.call-end').click();
        assert.deepEqual(h.calls.slice(2), ['mute', 'end']);
    } finally { h.done(); }
});

test('an ending in the big view brings the sheet back in the composer at once; the owner\'s End brings Talk', () => {
    const h = harness();
    try {
        h.set({ state: 'live', liveSince: 1 });
        h.capsule.expand();
        h.set({ state: 'closed', code: 'provider_closed', detail: '' });
        assert.equal(h.wrap.dataset.callView, 'sheet');
        assert.equal(h.capsule.big, false);
        assert.ok(!h.lifted() && !h.spacer());  // no passage for a sheet: it stands in the composer
        assert.equal(h.$('.call-close').hidden, false);
        assert.equal(h.$('.call-state').textContent, 'The voice service closed the call');
        assert.equal(h.$('.call-time').hidden, true);
        h.$('.call-close').click();
        assert.deepEqual(h.calls, ['dismiss']);
        h.set({ state: 'live', code: '', liveSince: 1 });
        h.capsule.expand();
        assert.ok(h.lifted());
        h.set({ state: 'ended', code: 'owner_end' });
        assert.equal(h.$('.call-capsule').hidden, true);
        assert.equal(h.wrap.dataset.callView, undefined);
        assert.ok(!h.lifted() && !h.spacer());
        assert.equal(h.timers.filter(Boolean).length, 0);
    } finally { h.done(); }
});

test('reduced motion switches sizes at once, with the same words', () => {
    const h = harness({ reduce: true });
    try {
        h.set({ state: 'live', liveSince: 1 });
        const words = h.$('.call-state').textContent;
        h.capsule.expand();
        assert.equal(h.wrap.dataset.callView, 'big');
        assert.equal(h.wrap.dataset.callMove, undefined);
        assert.equal(h.row.style.transition, 'none');
        assert.equal(h.$('.call-state').textContent, words);
        h.capsule.minimise();
        assert.equal(h.wrap.dataset.callView, 'voice');
        assert.ok(!h.lifted() && !h.spacer());  // landed without waiting for a passage
        assert.equal(h.timers.filter(Boolean).length, 0);
    } finally { h.done(); }
});

test('a phone\'s full-screen big view takes the rest of the page out of reach, and gives it back', () => {
    for (const width of [390, 1280]) {
        const h = harness({ width });
        try {
            const nav = h.doc.createElement('nav');
            h.doc.body.prepend(nav);
            h.set({ state: 'live', liveSince: 1 });
            h.capsule.expand();
            const preview = h.$('.chat-attachment-preview');
            // Covered on a phone (the panel is the screen); a wide window keeps the room usable above it.
            assert.equal(Boolean(nav.inert), width < 980, `nav ${width}`);
            assert.equal(Boolean(preview.inert), width < 980, `preview ${width}`);
            assert.ok(!h.row.inert && !h.page.inert && !h.wrap.inert, `the panel's own chain stays live ${width}`);
            h.capsule.minimise();
            assert.ok(!nav.inert && !preview.inert, `given back ${width}`);
            h.capsule.expand();
            h.set({ state: 'idle', liveSince: 0 });  // the call ended while big
            assert.ok(!nav.inert, `given back when the call ends ${width}`);
        } finally { h.done(); }
    }
});

test('the call\'s time runs beside the state while the call is live, and only then', () => {
    const h = harness();
    try {
        h.set({ state: 'preparing' });
        assert.equal(h.$('.call-time').hidden, true);
        h.set({ state: 'live', liveSince: Date.now() - 65_000 });
        assert.equal(h.$('.call-time').hidden, false);
        assert.equal(h.$('.call-time').textContent, '01:05');
        assert.equal(h.$('.call-time').getAttribute('aria-hidden'), 'true');  // never read out once a second
        h.set({ state: 'closed', code: 'owner_bye' });
        assert.equal(h.$('.call-time').hidden, true);
    } finally { h.done(); }
});

test('Talk wears the wave\'s small still sign: three strands, the main one strongest', async () => {
    const h = harness();
    try {
        await Promise.resolve();
        await Promise.resolve();
        const send = h.page.querySelector('.chat-send-inline');
        assert.equal(send.dataset.callEntry, 'talk');
        const mark = send.querySelector('.call-entry-mark');
        assert.ok(mark);
        assert.equal(mark.getAttribute('aria-hidden'), 'true');
        const strands = mark.querySelectorAll('path');
        assert.equal(strands.length, 3);
        assert.deepEqual(strands.map(path => path.getAttribute('stroke-width')), ['1', '1.3', '1.8']);
        assert.equal(send.querySelector('.call-entry-word').textContent, 'Talk');
        assert.equal(mark.querySelectorAll('i').length, 0);  // no bars of the old sign
    } finally { h.done(); }
});

test('the words\' second line is the wave\'s while a voice speaks or it has nothing to say, else its words', () => {
    // call.css gives a phone's chip this line for the wave; the capsule says, in every view, whose it is now.
    const h = harness({ width: 390 });
    const line = (facts, voice = {}) => {
        Object.assign(h.voice, { heard: false, owner: false }, voice);
        h.set({ ...reset, visionError: '', turn: '', ...facts });
        return [h.wrap.dataset.callLine, h.$('.call-sub').textContent];
    };
    try {
        assert.deepEqual(line({ state: 'live' }), ['text', 'Speak whenever you’re ready']);
        assert.deepEqual(line({ state: 'live' }, { owner: true }), ['wave', 'You’re speaking']);
        assert.deepEqual(line({ state: 'live' }, { heard: true }), ['wave', 'Speaking']);
        assert.deepEqual(line({ state: 'live', turn: 'awaiting' }), ['wave', '']);  // Thinking has no second line
        assert.deepEqual(line({ state: 'live', activity: { working: true, label: 'Searching the web' } }),
            ['text', 'Searching the web']);
        assert.deepEqual(line({ state: 'live', muted: true }, { owner: true }), ['text', 'Ouroboros can’t hear you']);
        // Words that need reading keep the line even while a voice speaks.
        assert.deepEqual(line({ state: 'live', visionError: 'The camera is in use by another app' }, { owner: true }),
            ['text', 'The camera is in use by another app']);
        assert.deepEqual(line({ state: 'reconnecting', code: 'session_cap' }), ['text', 'Starting a fresh voice session']);
        assert.deepEqual(line({ state: 'connecting' }), ['wave', '']);
        h.set({ state: 'idle' });
        assert.equal(h.wrap.dataset.callLine, undefined);  // the call's attributes leave with it
    } finally { h.done(); }
});

test('an image-incapable engine explains its hidden Show action; capable engines stay quiet', () => {
    for (const unsupported of [true, false]) {
        const reason = 'This voice engine does not accept images.';
        const h = harness({ vision: { screen: !unsupported, camera: !unsupported, unsupported, reason } });
        try {
            h.set({ state: 'live', callId: 'test-call' });
            assert.equal(h.$('.call-show').hidden, unsupported);
            assert.equal(h.$('.call-small').textContent, unsupported ? reason : '');
            h.capsule.expand();
            assert.equal(h.$('.call-small').textContent, unsupported ? reason : '');
            h.set({ state: 'failed', code: 'socket_silent' });
            assert.equal(h.$('.call-small').textContent, '', 'failure sheet owns its explanation');
        } finally { h.done(); }
    }
});
