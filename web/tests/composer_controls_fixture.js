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

/* The Model sheet's list as model_list.js writes it, read from its markup in order: a row is `{ key, kind, selected,
   name, tags, facts, pin }` (`pin`: null where the row has no Pin cell, else whether it is pressed; `expanded` on the
   `All sources` row), a heading `{ section }`, a note `{ note }`. Words are read as a reader gets them: no marks, no
   entities. */
const ENTITIES = { amp: '&', lt: '<', gt: '>', quot: '"', '#39': '\'' };
const plain = (html) => String(html).replace(/<svg[\s\S]*?<\/svg>/g, '').replace(/<[^>]+>/g, '')
    .replace(/&(amp|lt|gt|quot|#39);/g, (_all, name) => ENTITIES[name]);
const rowPart = (inner, name) => plain(inner.match(new RegExp(`<span class="chat-model-row-${name}"[^>]*>(.*?)</span>`))?.[1] || '');
export function modelSheetList(html) {
    const out = [];
    const grammar = /<div role="row" class="chat-model-row([^"]*)" data-row="([^"]*)" aria-selected="(true|false)"([^>]*)>(.*?)<\/div>|<p class="chat-model-(section|note)"[^>]*>(.*?)<\/p>/g;
    for (const [, kinds, key, selected, attrs, inner, part, words] of String(html).matchAll(grammar)) {
        if (part) { out.push({ [part]: plain(words) }); continue; }
        const pin = inner.match(/class="chat-model-pin[^"]*"[^>]*aria-pressed="(true|false)"/);
        const expanded = attrs.match(/aria-expanded="(true|false)"/);
        out.push({ key: plain(key), kind: kinds.trim().replace(/^chat-model-/, ''), selected: selected === 'true',
            name: rowPart(inner, 'id'), tags: [...inner.matchAll(/<span class="chat-model-row-tag">(.*?)<\/span>/g)].map((tag) => plain(tag[1])),
            facts: rowPart(inner, 'facts'), pin: pin ? pin[1] === 'true' : null, ...(expanded ? { expanded: expanded[1] === 'true' } : {}) });
    }
    return out;
}
/** Only the rows of `modelSheetList`. */
export const modelSheetRows = (html) => modelSheetList(html).filter((entry) => entry.key !== undefined);
/** The list in brief: a row as its key, `*` while it is the chosen one, and its Pin (`+pin` pressed, `-pin` not, nothing
 *  where the row has none); a heading as `§ words`, a note as `note: words`. */
export const modelSheetOutline = (html) => modelSheetList(html).map((entry) => (entry.section !== undefined ? `§ ${entry.section}`
    : entry.note !== undefined ? `note: ${entry.note}`
        : [entry.key, entry.selected ? '*' : '', entry.pin === null ? '' : entry.pin ? '+pin' : '-pin'].filter(Boolean).join(' ')));

/* The composer row as chat.js builds it: the Model anchor, the Subagents anchor, then the context anchor; Effort mounts
   after Context. `subagents` mounts the Subagents capsule on its anchor too (its writer answers `subagentAnswer`), and
   `model` the Model capsule on its own (its writers answer `modelAnswer` and `tierAnswer`; `model` holds its seams:
   `store`, `askMain`, `isStack`, `isCoarse`, `place`, `saveTier`). Each place the Model sheet is given is in `places`. */
export function mountPair({ hover = false, reply, win, withBox = false, projectId = '', chatId, subagents = false, model = null } = {}) {
    const doc = stubDocument();
    doc.body = doc.createElement('body');  // where a capsule's floating list is put
    const box = doc.createElement('section');
    const row = doc.createElement('div');
    box.appendChild(row);
    row.className = 'chat-toolbar-row';
    row.getBoundingClientRect = () => ({ top: 0, bottom: 0, left: 0, right: 0, width: 0, height: 0 });  // not laid out: no fit
    const modelEl = doc.createElement('div');
    modelEl.className = 'chat-model-mode';
    if (model) row.appendChild(modelEl);
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
    const modelAnswer = { value: null };
    const tierAnswer = { value: null };
    const places = [];
    const settingsOpened = [];
    const subagentsWriter = subagents ? {
        saveSubagentsMode: async (body) => {
            posts.push(['/api/owner/subagents-mode', body]);
            const reply = await subagentAnswer.value;
            return reply || { ok: true, project_id: projectId, subagents_mode: body.mode, cognition: { subagents_mode: body.mode } };
        },
    } : {};
    // The endpoint's own answers (CLIENT_CONTRACT §9): a room's saved choice, or Main as written.
    const modelWriter = model ? {
        saveChatModel: async (body) => {
            posts.push(['/api/owner/chat-model', body]);
            const reply = await modelAnswer.value;
            if (reply) return reply;
            const route = { model: body.model, use_local: body.use_local === true };
            return body.project_id ? { ok: true, project_id: body.project_id, previous_model_choice: { kind: 'main' },
                model_choice: body.kind === 'main' ? { kind: 'main' } : { kind: 'model', ...route } }
                : { ok: true, project_id: '', main_route: route, previous_main_route: route, account: 'none', window_reset: false };
        },
        modelDeps: {
            place: (sheet, anchor) => { places.push([sheet, anchor]); return { reposition() {}, destroy() {} }; },
            // A room's tier as saved, or Main's own with what Default resolves to (CLIENT_CONTRACT §9).
            saveTier: async (body) => {
                posts.push(['/api/owner/chat-processing', body]);
                const reply = await tierAnswer.value;
                if (reply) return reply;
                return body.project_id ? { ok: true, project_id: body.project_id, processing_choice: body.processing, previous_processing_choice: '' }
                    : { ok: true, project_id: '', main_processing: body.processing, previous_main_processing: '', processing_default: 'standard' };
            },
            ...model,
        },
    } : {};
    const controls = createComposerOwnerControls({
        ...subagentsWriter, ...modelWriter, openSettings: (tab) => settingsOpened.push(tab),
        projectId, ...(chatId === undefined ? {} : { chatId }), row, box: withBox ? box : null, doc, win: win || hoverWindow(hover),
        byId: (suffix) => (suffix === 'context-mode' ? contextEl : suffix === 'subagents-mode' ? subagentsEl
            : suffix === 'model-mode' ? modelEl : null),
        apiFetch: async (url, init) => {
            const body = JSON.parse(init.body); posts.push([url, body, init.headers]);
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
        modelEl, modelAnswer, tierAnswer, places, modelHead: modelEl.querySelector('.chat-model-head'),
        fire,
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
