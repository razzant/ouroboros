// The composer's owner controls mounted on stub elements (composer_owner_controls.test.js and
// composer_owner_haptics.test.js): a document whose listeners can be fired, one effort capsule or both capsules in
// one composer row as chat.js builds them, and the presses that drive them.
import { ElementStub } from './chat_dom_fixture.js';
import { createComposerOwnerControls, createEffortRangeControl } from '../modules/composer_owner_controls.js';

/* A document whose listeners can be fired, and a window that may or may not hover. */
export function stubDocument() {
    const listeners = new Map();
    const doc = {
        byId: new Map(), activeElement: null,
        createElement: (tag) => new ElementStub(tag, doc),
        addEventListener(type, fn) { listeners.set(type, [...(listeners.get(type) || []), fn]); },
        removeEventListener(type, fn) { listeners.set(type, (listeners.get(type) || []).filter((f) => f !== fn)); },
        fire(type, event) { for (const fn of listeners.get(type) || []) fn(event); },
        count: (type) => (listeners.get(type) || []).length,
    };
    return doc;
}
export const hoverWindow = (matches) => ({ matchMedia: (query) => ({ matches: matches && query === '(hover: hover) and (pointer: fine)' }) });
export const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

export function mount({ hover = false, save, win } = {}) {
    const doc = stubDocument();
    const row = doc.createElement('div');
    row.className = 'chat-toolbar-row';
    row.getBoundingClientRect = () => ({ top: 0, bottom: 0, left: 0, right: 0, width: 0, height: 0 });  // not laid out: no fit
    const toasts = [];
    const refreshes = [];
    const layouts = [];
    const saves = [];
    const control = createEffortRangeControl({
        row, doc, win: win === undefined ? hoverWindow(hover) : win,
        saveEffortRange: save || (async (triple) => { saves.push(triple); return { ok: true, effort_range: triple }; }),
        showToast: (message, tone) => toasts.push([message, tone]),
        refreshState: (force) => refreshes.push(force),
        onLayout: () => layouts.push(control?.isOpen?.()),
    });
    const el = control.el;
    const segs = el.querySelectorAll('.chat-effort-seg');
    // The strip has layout: seven 40px segments, so a boundary is x / 40.
    segs.forEach((seg, i) => { seg.offsetLeft = i * 40; seg.offsetWidth = 40; });
    const head = el.querySelector('.chat-effort-head');
    const strip = el.querySelector('[data-effort-strip]');
    const handles = { min: el.querySelector('.chat-effort-min'), rec: el.querySelector('.chat-effort-rec'), max: el.querySelector('.chat-effort-max') };
    const fire = (node, type, event) => { for (const fn of node.listeners.get(type) || []) fn(event); };
    const last = (node, type) => (node.listeners.get(type) || []).at(-1);
    const pointer = (clientX, { handle = null, pointerId = 1, clientY = 10 } = {}) => ({
        button: 0, clientX, clientY, pointerId, pointerType: 'mouse', preventDefault() {},
        target: { closest: (selector) => (selector === '[data-handle]' ? handle : null) },
    });
    return {
        control, el, row, doc, head, strip, handles, segs, toasts, refreshes, layouts, saves,
        headClick: (detail = 1) => fire(head, 'click', { detail }),
        key: (which, key) => fire(handles[which], 'keydown', { key, currentTarget: handles[which], preventDefault() {} }),
        down: (x, options) => fire(strip, 'pointerdown', pointer(x, options)),
        move: (x, pointerId = 1) => last(strip, 'pointermove')?.(pointer(x, { pointerId })),
        up: (x, pointerId = 1) => last(strip, 'pointerup')?.(pointer(x, { pointerId })),
        cancel: (pointerId = 1) => last(strip, 'pointercancel')?.(pointer(0, { pointerId })),
        enter: (pointerType = 'mouse') => fire(el, 'pointerenter', { pointerType }),
        // The stub control's rectangle is x 0..100, y 0..20; the default leave point is far outside the zone.
        leave: (pointerType = 'mouse', { x = 500, y = 500 } = {}) => fire(el, 'pointerleave', { pointerType, clientX: x, clientY: y }),
        docMove: (x, y) => doc.fire('pointermove', { pointerType: 'mouse', clientX: x, clientY: y }),
        outside: () => doc.fire('pointerdown', { target: new ElementStub('div', doc) }),
        inside: () => doc.fire('pointerdown', { target: head }),
        escape: () => { const event = { key: 'Escape', defaultPrevented: false, preventDefault() { this.defaultPrevented = true; } };
            doc.fire('keydown', event); return event; },
        settle: () => control.pendingSave(),
    };
}

/* The composer row as chat.js builds it: the Subagents anchor, then the context anchor; Effort mounts after Context.
   `subagents` mounts the Subagents capsule on its anchor too (its writer answers `subagentAnswer`). */
export function mountPair({ hover = false, reply, win, withBox = false, projectId = '', subagents = false } = {}) {
    const doc = stubDocument();
    const box = doc.createElement('section');
    const row = doc.createElement('div');
    box.appendChild(row);
    row.className = 'chat-toolbar-row';
    row.getBoundingClientRect = () => ({ top: 0, bottom: 0, left: 0, right: 0, width: 0, height: 0 });  // not laid out: no fit
    const subagentsEl = doc.createElement('div');
    subagentsEl.className = 'chat-subagents-mode';
    row.appendChild(subagentsEl);
    const contextEl = doc.createElement('div');
    contextEl.className = 'chat-context-mode';
    contextEl.dataset.contextMode = 'max';
    row.appendChild(contextEl);
    const posts = [];
    const toasts = [];
    const refreshes = [];
    const answer = { value: reply || { ok: true } };
    const effortAnswer = { value: null };
    const subagentAnswer = { value: null };
    const settingsOpened = [];
    const subagentsWriter = subagents ? {
        saveSubagentsMode: async (body) => {
            posts.push(['/api/owner/subagents-mode', body]);
            const reply = await subagentAnswer.value;
            return reply || { ok: true, project_id: projectId, subagents_mode: body.mode, cognition: { subagents_mode: body.mode } };
        },
        openSettings: () => settingsOpened.push('agents'),
    } : {};
    const controls = createComposerOwnerControls({
        ...subagentsWriter,
        projectId, row, box: withBox ? box : null, doc, win: win || hoverWindow(hover),
        byId: (suffix) => (suffix === 'context-mode' ? contextEl : suffix === 'subagents-mode' ? subagentsEl : null),
        apiFetch: async (url, init) => {
            const body = JSON.parse(init.body); posts.push([url, body]);
            const response = await answer.value;
            return response.json ? response : { ...response, json: async () => ({ project_id: projectId, context_mode: body.mode }) };
        },
        saveEffortRange: async (body) => { posts.push(['/api/owner/effort-range', body]); return effortAnswer.value || { project_id: projectId, effort_range: body }; },
        showToast: (message, tone) => toasts.push([message, tone]),
        refreshState: (force) => refreshes.push(force),
    });
    const fire = (node, type, event) => Promise.all((node.listeners.get(type) || []).map((fn) => fn(event)));
    const effortEl = controls.effort.el;
    const seg = (mode) => contextEl.querySelectorAll('.chat-seg').find((node) => node.dataset.mode === mode);
    const subSeg = (mode) => row.querySelectorAll('.chat-subagents-strip')[0]?.querySelectorAll('.chat-seg').find((node) => node.dataset.mode === mode);
    const subStrip = () => subagentsEl.querySelector('.chat-subagents-strip');
    const onSeg = (mode) => ({ closest: (selector) => (selector === '.chat-seg' ? subSeg(mode) : null) });
    return {
        controls, row, box, doc, contextEl, effortEl, posts, toasts, refreshes, answer, effortAnswer, seg,
        subagentsEl, subagentAnswer, settingsOpened, subSeg, subHead: subagentsEl.querySelector('.chat-subagents-head'),
        subHint: subagentsEl.querySelector('.ui-hint'),
        contextHint: contextEl.querySelector('.ui-hint'), effortHint: effortEl.querySelector('.ui-hint'),
        preview: (node, mode) => fire(node, 'pointerover', { pointerType: 'mouse', target: { closest: (selector) => (selector === '.chat-seg'
            ? node.querySelectorAll('.chat-seg').find((seg) => seg.dataset.mode === mode) : null) } }),
        previewEnd: (node) => fire(node, 'pointerleave', { pointerType: 'mouse' }),
        chooseSub: (mode) => fire(subStrip(), 'click', { target: onSeg(mode) }),
        subKey: (mode, key) => {
            const event = { key, defaultPrevented: false, preventDefault() { this.defaultPrevented = true; }, target: onSeg(mode) };
            fire(subStrip(), 'keydown', event);
            return event;
        },
        contextHead: contextEl.querySelector('.chat-context-head'), effortHead: effortEl.querySelector('.chat-effort-head'),
        press: (head, detail = 1) => fire(head, 'click', { detail }),
        choose: (mode) => fire(contextEl, 'click', { target: { closest: (selector) => (selector === '.chat-seg' ? seg(mode) : null) } }),
        key: (mode, key) => fire(contextEl.querySelector('.chat-context-strip'), 'keydown', {
            key, preventDefault() {}, target: { closest: (selector) => (selector === '.chat-seg' ? seg(mode) : null) },
        }),
        docDown: (target) => doc.fire('pointerdown', { target }),
    };
}
