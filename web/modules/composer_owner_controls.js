// Each composer's saved room settings: Model, Subagents, Context and Effort, four inline capsules of one
// family. Main and Projects read their own authoritative values and save through the owner
// endpoints (DESIGN: Composer owner controls); all four run on the capsule machine (capsule.js).
import { fmt } from './i18n.js';
import {
    EFFORT_OPTIONS, EFFORT_RANGE_DEFAULT, OWNER_TIERS, effortText, normalizeEffortRange, ownerLevelIndex,
} from './effort_levels.js';
import { createCapsule, observeLayout } from './capsule.js';
import { SEG_PAD, px, widthOf } from './capsule_fit.js';
import { haptic } from './haptics.js';
import { createComposerFold } from './composer_fold.js';
import { createModelControl } from './composer_model.js';
import { createSubagentsControl } from './composer_subagents.js';
import { CONTEXT_LABELS, CONTEXT_MODES, UNKNOWN_CONTEXT, contextShown, observeContext } from './context_observation.js';

const N = OWNER_TIERS.length - 1;
const HANDLES = ['min', 'rec', 'max'];
const CONTEXT_MEANINGS = Object.freeze({
    nano: 'The compact owner window.', low: 'Aims for about 250K tokens.', max: 'The full context window.',
});
const contextModeOf = (mode) => (['nano', 'low', 'max'].includes(mode) ? mode : 'max');
const clamp = (value, lo, hi) => Math.min(hi, Math.max(lo, value));

/** The primary level is independent; the helper brackets cannot cross each other. */
export function applyHandle(base, which, value) {
    const s = { ...base };
    const v = clamp(value, 0, N);
    if (which === 'min') s.min = Math.min(v, s.max);
    else if (which === 'max') s.max = Math.max(v, s.min);
    else s.rec = v;
    return s;
}

export function snapRange(v) {
    const s = { min: Math.round(v.min), rec: Math.round(v.rec), max: Math.round(v.max) };
    s.rec = clamp(s.rec, 0, N);
    return s;
}

const indicesOf = (range) => ({
    min: ownerLevelIndex(range.min), rec: ownerLevelIndex(range.recommended), max: ownerLevelIndex(range.max),
});
const sameIndices = (a, b) => Boolean(a && b) && HANDLES.every((key) => a[key] === b[key]);

/* The ring glyph: the range in miniature (a 300° arc; the band is min..max, the dot the recommended
   level). Context's ring is the same frame cut into three arcs, the observed modes' arcs lit:
   positions among three modes, never a fill level or a capacity. Effort's dot, on the saved
   mode's arc, keeps the owner's choice while the observation disagrees. */
const RING = { cx: 92, cy: 92, r: 70, a0: 120, sweep: 300 };
const CONTEXT_ARCS = { nano: [120, 200], low: [230, 310], max: [340, 420] };
const ringPoint = (angle) => ({
    x: RING.cx + RING.r * Math.cos((angle * Math.PI) / 180), y: RING.cy + RING.r * Math.sin((angle * Math.PI) / 180),
});
const ringAngle = (level) => RING.a0 + (level / N) * RING.sweep;
function ringArc(a0, a1) {
    const hi = Math.max(a1, a0 + 0.02);
    const p0 = ringPoint(a0);
    const p1 = ringPoint(hi);
    return `M${p0.x.toFixed(2)} ${p0.y.toFixed(2)} A${RING.r} ${RING.r} 0 ${hi - a0 > 180 ? 1 : 0} 1 ${p1.x.toFixed(2)} ${p1.y.toFixed(2)}`;
}

/* A capsule's head: the ring, the setting's name and its value (the value gives way to the strip). */
const capsuleHead = (glyph, name, headClass, valueAttrs = '') => `<button type="button" class="chat-capsule-head ui-hit-area ${headClass}" aria-expanded="false">`
    + `<span class="chat-capsule-glyph" aria-hidden="true"><svg viewBox="0 0 184 184">${glyph}</svg></span>`
    + `<span class="chat-capsule-name">${name}</span><span class="chat-capsule-value"${valueAttrs}></span></button>`;

function effortRangeMarkup() {
    const segments = EFFORT_OPTIONS.map((option) =>
        `<span class="chat-effort-seg" data-effort-seg="${option.value}"><span class="chat-effort-seg-text" data-i18n-skip>${option.label}</span></span>`).join('');
    const ring = `<path class="chat-effort-glyph-rail" data-ring-rail d="${ringArc(RING.a0, RING.a0 + RING.sweep)}" fill="none" stroke-width="22" stroke-linecap="round"></path><path class="chat-effort-glyph-band" data-ring-band fill="none" stroke-width="26" stroke-linecap="round"></path><circle class="chat-effort-glyph-dot" data-ring-rec r="22"></circle>`;
    // The value is a tier word read through the catalog seam (effortText): the overlay leaves it alone.
    return capsuleHead(ring, 'Effort', 'chat-effort-head', ' data-i18n-skip')
        + '<div class="chat-capsule-body">'
        + '<div class="chat-effort-strip ui-hit-area" data-effort-strip data-scroll="none"><div class="chat-effort-band" data-effort-band aria-hidden="true"></div>'
        + '<div class="chat-effort-handle chat-effort-rec" data-handle="rec" role="slider" tabindex="-1" aria-label="Primary effort"></div>'
        + segments
        + '<div class="chat-effort-handle chat-effort-cap chat-effort-min" data-handle="min" role="slider" tabindex="-1" aria-label="Minimum helper effort"></div>'
        + '<div class="chat-effort-handle chat-effort-cap chat-effort-max" data-handle="max" role="slider" tabindex="-1" aria-label="Maximum helper effort"></div></div>'
        + '</div>';
}

function contextModeMarkup() {
    const arcs = CONTEXT_MODES.map((mode) => `<path class="chat-context-arc" data-arc="${mode}" data-lit="false" d="${ringArc(...CONTEXT_ARCS[mode])}" fill="none" stroke-width="24" stroke-linecap="round"></path>`).join('')
        + '<circle class="chat-effort-glyph-dot chat-context-choice" r="22"></circle>';
    const modes = CONTEXT_MODES.map((mode) =>
        `<button class="chat-seg ui-hit-area" type="button" role="radio" tabindex="-1" aria-checked="false" data-mode="${mode}"><span class="chat-seg-text">${CONTEXT_LABELS[mode]}</span></button>`).join('');
    return capsuleHead(arcs, 'Context', 'chat-context-head')
        + `<div class="chat-capsule-body"><div class="chat-context-strip" role="radiogroup" aria-label="Context size mode" data-scroll="none"><span class="chat-context-thumb" aria-hidden="true"></span>${modes}</div></div>`;
}

/** Inline primary/helper effort controls. Gesture-end saves serialize; refusal restores
 * the last server value. A stored `minimal` renders at Low until that handle moves. */
export function createEffortRangeControl({
    row, saveEffortRange, showToast, refreshState = () => {}, onLayout = () => {}, onSettled = () => {},
    win = globalThis.window, doc = globalThis.document,
}) {
    const el = doc.createElement('div');
    el.className = 'chat-capsule chat-effort-range';
    el.setAttribute('data-effort-range', '');
    el.setAttribute('role', 'group');
    el.setAttribute('aria-label', 'Effort range');
    el.dataset.known = 'false';
    el.innerHTML = effortRangeMarkup();
    const q = (selector) => el.querySelector(selector);
    const head = q('.chat-effort-head');
    const value = q('.chat-capsule-value');
    const strip = q('[data-effort-strip]');
    const band = q('[data-effort-band]');
    const ring = { band: q('[data-ring-band]'), rec: q('[data-ring-rec]') };
    const segs = Array.from(el.querySelectorAll('.chat-effort-seg'));
    // Each level says the word the capsule's value says (the catalog seam), written before the marks are measured: the
    // overlay would translate the strip's words later, after the marks had landed under them.
    const words = Array.from(el.querySelectorAll('.chat-effort-seg-text'));
    const handles = { min: q('.chat-effort-min'), rec: q('.chat-effort-rec'), max: q('.chat-effort-max') };
    // After the context capsule.
    const before = row.querySelector('.chat-context-mode');
    if (before && typeof before.insertAdjacentElement === 'function') before.insertAdjacentElement('afterend', el);
    else row.appendChild(el);

    // `stored` is the server's value; `draft` the tiers the server will hold once every queued
    // save landed (equal to `stored` when nothing is pending); `shown` the owner-tier indices.
    const state = {
        stored: { ...EFFORT_RANGE_DEFAULT }, draft: { ...EFFORT_RANGE_DEFAULT }, shown: indicesOf(EFFORT_RANGE_DEFAULT),
        view: null, drag: null, saving: null, queued: null, lastOk: true, destroyed: false,
    };
    for (const which of HANDLES) { handles[which].setAttribute('role', 'slider'); handles[which].tabIndex = -1; }
    let capsule = null;

    /* --- geometry: segment edges in px along the scale, measured from the DOM (labels differ in
       width). Right to left the levels run leftwards: a position along the scale is the strip's x
       times `dir` (-1), and the same product turns it back. --- */
    const scaleDir = () => (capsule?.isRtl() ? -1 : 1);
    function edges(dir = scaleDir()) {
        // Each level's start, then the last level's end. The levels meet, so right to left a level
        // starts where the one before it ends: one whole-pixel offset per edge, as left to right.
        const left = segs.map((seg) => Number(seg.offsetLeft));
        const e = dir > 0 ? [...left, left[N] + Number(segs[N].offsetWidth)]
            : [-(left[0] + Number(segs[0].offsetWidth)), ...left.map((x) => -x)];
        return e.every(Number.isFinite) && e[N + 1] > e[0] ? e : null;
    }
    function boundaryAt(event) {
        const dir = scaleDir();
        const e = edges(dir);
        if (!e) return null;
        const x = clamp(dir * (event.clientX - strip.getBoundingClientRect().left + (Number(strip.scrollLeft) || 0)), e[0], e[N + 1]);
        let i = 0;
        while (i < N && x >= e[i + 1]) i += 1;
        return i + (x - e[i]) / (e[i + 1] - e[i]);
    }
    function valueAt(event, which) {
        const b = boundaryAt(event);
        if (b === null) return (state.view || state.shown)[which];
        return which === 'min' ? b : which === 'max' ? b - 1 : b - 0.5;
    }
    function pick(event) {
        const b = boundaryAt(event);
        const segment = b === null ? state.shown.rec : Math.min(N, Math.floor(b));
        return { which: 'rec', start: segment };  // helper bounds have their own handles
    }

    /* --- rendering: every mark sits on whole levels, a drag included (the pill always covers
       the word it names independently of the helper brackets) --- */
    function render() {
        OWNER_TIERS.forEach((tier, i) => { const word = effortText(tier); if (words[i].textContent !== word) words[i].textContent = word; });
        const s = snapRange(state.view || state.shown);
        const dir = scaleDir();
        const e = edges(dir);
        if (e) {
            // A span along the scale stands at its left side in the strip.
            const span = (node, from, to) => {
                node.style.left = `${Math.min(dir * from, dir * to)}px`;
                node.style.width = `${to - from}px`;
            };
            span(band, e[s.min], e[s.max + 1]);
            span(handles.rec, e[s.rec], e[s.rec + 1]);
            handles.min.style.left = `${dir * e[s.min]}px`;
            handles.max.style.left = `${dir * e[s.max + 1]}px`;  // each bracket draws toward the range (style.css)
        }
        segs.forEach((seg, i) => { seg.dataset.in = String(i >= s.min && i <= s.max); seg.dataset.rec = String(i === s.rec); });
        el.dataset.dragging = String(Boolean(state.drag?.moved));
        ring.band.setAttribute('d', ringArc(ringAngle(s.min), ringAngle(s.max)));
        const dot = ringPoint(ringAngle(s.rec));
        ring.rec.setAttribute('cx', dot.x.toFixed(2));
        ring.rec.setAttribute('cy', dot.y.toFixed(2));
        // Primary may go anywhere; each helper bracket stops at the other one.
        const bounds = { min: [0, s.max], rec: [0, N], max: [s.min, N] };
        const open = Boolean(capsule?.isOpen());
        for (const which of HANDLES) {
            handles[which].setAttribute('aria-valuemin', String(bounds[which][0]));
            handles[which].setAttribute('aria-valuemax', String(bounds[which][1]));
            handles[which].setAttribute('aria-valuenow', String(s[which]));
            handles[which].setAttribute('aria-valuetext', effortText(OWNER_TIERS[s[which]]));
            handles[which].dataset.dragging = String(state.drag?.which === which);
            handles[which].tabIndex = open ? 0 : -1;
        }
        // Closed, the capsule names only the recommended level; the ring and the name say the range.
        const word = effortText(state.draft.recommended);
        if (value.textContent !== word) value.textContent = word;
        const stored = state.stored;
        head.setAttribute('aria-label', fmt('Primary effort: {level}; helpers {min}–{max}.', {
            level: effortText(stored.recommended), min: effortText(stored.min), max: effortText(stored.max),
        }));
        capsule?.sayHint();
    }
    // What the hint says: the primary level and the helpers' range where the marks stand, a drag's included.
    const say = () => {
        const s = snapRange(state.view || state.shown);
        const level = (i) => effortText(OWNER_TIERS[i]);
        return { word: level(s.rec), text: fmt('Primary effort; helpers between {min} and {max}.', { min: level(s.min), max: level(s.max) }) };
    };

    /* --- the model: drag preview, commit on gesture end, serialized saves --- */
    function beginDrag(which) { state.drag = { which }; state.view = { ...state.shown }; render(); }
    // A drag builds on its preview; cancel restores the committed values.
    function dragTo(which, value) { if (!state.drag) return; state.view = applyHandle(state.view || state.shown, which, value); render(); }
    function cancelDrag() { state.drag = null; state.view = null; render(); }
    function endDrag() {
        if (!state.drag) return;
        const { which } = state.drag;
        const next = snapRange(state.view);
        state.drag = null;
        state.view = null;
        commit(next);
        revealHandle(which);
        if (capsule.isOpen() && !capsule.isPinned() && capsule.hoverCapable) capsule.closeSoon();
    }
    // A handle the owner did not move keeps its tier (`minimal` shown at Low stays `minimal`).
    function tripleFor(next) {
        const prev = state.shown;
        const draft = state.draft;
        return {
            min: next.min === prev.min ? draft.min : OWNER_TIERS[next.min],
            recommended: next.rec === prev.rec ? draft.recommended : OWNER_TIERS[next.rec],
            max: next.max === prev.max ? draft.max : OWNER_TIERS[next.max],
        };
    }
    function commit(next) {
        if (sameIndices(next, state.shown)) { render(); return; }
        const triple = tripleFor(next);
        state.shown = next;
        state.draft = triple;
        render();
        enqueue(triple);
    }
    function applyServer(range) {
        state.stored = normalizeEffortRange(range);
        state.draft = { ...state.stored };
        state.shown = indicesOf(state.stored);
        el.dataset.known = 'true';
        render();
    }
    function enqueue(triple) {
        state.queued = triple;
        el.dataset.saving = 'true';
        if (!state.saving) state.saving = drain();
    }
    async function drain() {
        let ok = true;
        while (state.queued) {
            const triple = state.queued;
            state.queued = null;
            try {
                const response = await saveEffortRange(triple);
                if (state.destroyed) return ok;
                const saved = response?.effort_range && typeof response.effort_range === 'object' ? response.effort_range : triple;
                // A newer edit is queued and was built on the draft: the draft stays ahead of this
                // answer, so the next edit never starts from an older range.
                if (state.queued) state.stored = normalizeEffortRange(saved);
                else applyServer(saved);
                ok = true;
            } catch (error) {
                if (state.destroyed) return false;
                ok = false;
                state.queued = null;  // later edits were built on the refused draft
                showToast(error?.message || 'Could not change the effort range.', 'error');
                applyServer(state.stored);  // the server's value, never the unsaved preview
            }
        }
        state.saving = null;
        state.lastOk = ok;
        el.dataset.saving = 'false';
        refreshState(true);
        return ok;
    }

    // A strip that scrolls shows where it continues and keeps the moved handle in view.
    function revealHandle(which = 'rec') {
        if (el.dataset.fit !== 'overflow') return;
        const dir = scaleDir();
        const e = edges(dir);
        if (!e) return;
        const s = snapRange(state.view || state.shown);
        const cap = widthOf(handles.min);
        const [from, to] = which === 'min' ? [e[s.min], e[s.min] + cap]
            : which === 'max' ? [e[s.max + 1] - cap, e[s.max + 1]] : [e[s.rec], e[s.rec + 1]];
        capsule.reveal(Math.min(dir * from, dir * to), Math.max(dir * from, dir * to));
    }

    capsule = createCapsule({
        el, row, kind: 'effort', strip, segs, render, focusTarget: () => handles.rec, say, onLayout, onSettled, win, doc,
        busy: () => Boolean(state.drag), onClosed: () => { if (state.drag) cancelDrag(); }, afterOpen: () => revealHandle('rec'),
        afterOverflow: () => revealHandle(HANDLES.find((which) => handles[which] === doc.activeElement) || 'rec'),
    });

    /* --- pointer: a press on the strip grabs a handle (or picks one) and keeps the grab offset --- */
    const onDown = (event) => {
        if (event.button) return;
        event.preventDefault?.();
        const handleEl = event.target?.closest?.('[data-handle]');
        let which;
        let start;
        if (handleEl) { which = handleEl.dataset.handle; start = state.shown[which]; }
        else ({ which, start } = pick(event));
        let at = state.shown[which];  // each level the finger reaches ticks once (haptics.js), a press on another too
        const reach = () => { const now = snapRange(state.view)[which]; if (now !== at) { at = now; haptic('tick'); } };
        beginDrag(which);
        if (!handleEl) dragTo(which, start);
        reach();
        const grab = valueAt(event, which) - state.view[which];
        handles[which].focus({ preventScroll: true, focusVisible: false });  // no keyboard ring after a mouse press
        try { strip.setPointerCapture(event.pointerId); } catch { /* a synthetic event */ }
        const move = (ev) => {
            if (ev.pointerId !== event.pointerId || !state.drag) return;
            state.drag.moved = true;  // a press that stays put is a tap: its marks still animate
            dragTo(which, valueAt(ev, which) - grab);
            reach();
        };
        const release = (ev) => {
            if (ev.pointerId !== event.pointerId) return;
            strip.removeEventListener('pointermove', move);
            strip.removeEventListener('pointerup', up);
            strip.removeEventListener('pointercancel', cancel);
        };
        const up = (ev) => { if (ev.pointerId !== event.pointerId) return; release(ev); endDrag(); };
        const cancel = (ev) => { if (ev.pointerId !== event.pointerId) return; release(ev); cancelDrag(); };
        strip.addEventListener('pointermove', move);
        strip.addEventListener('pointerup', up);
        strip.addEventListener('pointercancel', cancel);
    };
    const onHandleKey = (event) => {
        const which = event.currentTarget?.dataset?.handle || event.target?.dataset?.handle;
        if (!which) return;
        const step = { ArrowRight: 1, ArrowUp: 1, ArrowLeft: -1, ArrowDown: -1, PageUp: 1, PageDown: -1 }[event.key];
        let target;
        if (step) target = state.shown[which] + step;
        else if (event.key === 'Home') target = 0;
        else if (event.key === 'End') target = N;
        else return;
        event.preventDefault?.();
        commit(snapRange(applyHandle(state.shown, which, clamp(target, 0, N))));
        revealHandle(which);
    };

    strip.addEventListener('pointerdown', onDown);
    for (const which of HANDLES) handles[which].addEventListener('keydown', onHandleKey);
    render();

    return {
        el,
        capsule,
        hoverCapable: capsule.hoverCapable,
        isOpen: () => capsule.isOpen(),
        isPinned: () => capsule.isPinned(),
        shown: () => ({ ...state.shown }),
        stored: () => ({ ...state.stored }),
        /** The containing composer supplies only its room's authoritative value. */
        syncState(data) {
            const range = data?.effort_range;
            if (!range || typeof range !== 'object' || typeof range.recommended !== 'string') return;
            if (state.saving || state.queued || state.drag) return;  // the save's own echo wins over a stale poll
            applyServer(range);
        },
        hasPendingSave: () => Boolean(state.saving),
        /** Resolves once every queued save settled: true when the last one was accepted. */
        pendingSave: () => (state.saving ? state.saving : Promise.resolve(state.lastOk)),
        destroy() {
            state.destroyed = true;
            capsule.destroy();
            strip.removeEventListener('pointerdown', onDown);
            for (const which of HANDLES) handles[which].removeEventListener('keydown', onHandleKey);
            el.remove();
        },
    };
}

/** Context saves apply to future room participants; a refusal retains the saved mode. */
function createContextModeControl({ el, row, chatId, saveContextMode, showToast, refreshState, onLayout, onSettled, win, doc }) {
    el.classList.add('chat-capsule');
    el.setAttribute('role', 'group');
    el.setAttribute('aria-label', 'Context size mode');
    el.innerHTML = contextModeMarkup();
    const value = el.querySelector('.chat-capsule-value');
    const strip = el.querySelector('.chat-context-strip');
    const thumb = el.querySelector('.chat-context-thumb');
    const segs = Array.from(el.querySelectorAll('.chat-seg'));
    const head = el.querySelector('.chat-context-head');
    const arcs = Array.from(el.querySelectorAll('.chat-context-arc'));
    const dot = el.querySelector('.chat-context-choice');
    const current = () => contextModeOf(el.dataset.contextMode);
    const chosen = () => segs.find((seg) => seg.dataset.mode === current());
    let capsule = null;
    let named = '';  // the word the value says (an overlay may have translated it)
    // The modes this page last saw this room's works' requests sent in (context_observation.js); a choice never paints it.
    let observation = UNKNOWN_CONTEXT;
    let shown = contextShown(observation, current());

    function render() {
        const mode = current();
        const open = Boolean(capsule?.isOpen());
        shown = contextShown(observation, mode);
        if (named !== shown.word) { named = shown.word; value.textContent = shown.word; }
        el.dataset.contextActual = shown.actual;
        el.dataset.contextDiffers = String(shown.differs);
        for (const arc of arcs) arc.setAttribute('data-lit', String(shown.lit.includes(arc.dataset.arc)));
        const [a0, a1] = CONTEXT_ARCS[mode];
        const at = ringPoint((a0 + a1) / 2);
        dot.setAttribute('cx', at.x.toFixed(2));
        dot.setAttribute('cy', at.y.toFixed(2));
        if (shown.label) head.setAttribute('aria-label', shown.label);
        else head.removeAttribute('aria-label');
        for (const seg of segs) {
            seg.setAttribute('aria-checked', String(seg.dataset.mode === mode));
            seg.tabIndex = open && seg.dataset.mode === mode ? 0 : -1;  // one tab stop; arrows move within
        }
        const seg = chosen();
        if (Number(seg?.offsetWidth) > 0) {
            thumb.style.left = `${seg.offsetLeft}px`;
            thumb.style.width = `${seg.offsetWidth}px`;
        }
        capsule?.sayHint();
    }
    // The chosen mode, or the one a mouse rests on; under it, the dated observation,
    // with its cause when it differs from the choice.
    const say = (preview) => {
        const mode = CONTEXT_MODES.includes(preview) ? preview : current();
        return { word: CONTEXT_LABELS[mode], text: CONTEXT_MEANINGS[mode], fact: shown.fact };
    };
    let saving = null, lastOk = true, destroyed = false;
    function choose(mode) {
        const next = contextModeOf(mode);
        if (saving || destroyed || next === current()) return;
        el.dataset.disabled = 'true';
        saving = (async () => {
            try {
                const saved = await saveContextMode(next);
                if (destroyed) return false;
                el.dataset.contextMode = contextModeOf(saved.context_mode);
                lastOk = true;
            } catch (e) {
                if (destroyed) return false;
                lastOk = false;
                showToast(e.message || 'Could not change context mode.', 'error');
            } finally {
                saving = null;
                if (!destroyed) {
                    el.dataset.disabled = 'false';
                    render();
                    refreshState(true);
                }
            }
            return lastOk;
        })();
        return saving;
    }
    const onChoose = (event) => {
        const seg = event.target?.closest?.('.chat-seg');
        const started = seg ? choose(seg.dataset.mode) : undefined;
        if (started) haptic('tick');  // another mode, chosen by a press (haptics.js)
        return started;
    };
    // A strip that scrolls (long translated words on a phone) keeps the chosen or focused mode whole.
    const revealMode = (seg) => { if (seg) capsule.reveal(Number(seg.offsetLeft), Number(seg.offsetLeft) + Number(seg.offsetWidth)); };
    const onKey = (event) => {
        const at = segs.indexOf(event.target?.closest?.('.chat-seg'));
        const step = { ArrowRight: 1, ArrowDown: 1, ArrowLeft: -1, ArrowUp: -1 }[event.key];
        let to;
        if (at < 0) return;
        if (step) to = clamp(at + step, 0, segs.length - 1);
        else if (event.key === 'Home') to = 0;
        else if (event.key === 'End') to = segs.length - 1;
        else return;
        event.preventDefault?.();
        segs[to].focus({ preventScroll: true });
        revealMode(segs[to]);
        return choose(segs[to].dataset.mode);
    };

    capsule = createCapsule({
        el, row, kind: 'context', strip, segs, render, focusTarget: chosen, say, padMax: SEG_PAD.max + 3, onLayout, onSettled, win, doc,
        afterOpen: () => revealMode(chosen()),
        afterOverflow: () => revealMode(segs.find((seg) => seg === doc.activeElement) || chosen()),
    });
    el.addEventListener('click', onChoose);
    strip.addEventListener('keydown', onKey);
    render();

    return {
        el,
        capsule,
        /** A save in flight answers for itself; the composer supplies this room's value. */
        syncState(data) {
            if (typeof data?.context_mode !== 'string' || el.dataset.disabled === 'true') return;
            el.dataset.contextMode = contextModeOf(data.context_mode);
            render();
        },
        /** One `/api/state` read: this room's works' recorded requests (`context_attempt`). A body without the census
         *  keeps what was seen, except chat.js's unavailable frame (`accounting.available === false`, sent on a socket
         *  close and a failed state read): no current read then, so `now` becomes the dated `last` until a read returns,
         *  an incomplete read (no work departed, so a later release cannot take a retained earlier send with it).
         *  An explicitly incomplete census (`active_chat_activities_complete`/`supervisor_ready` false) is kept as
         *  seen, never as the room's whole current mode.
         *  The saved mode is never written from it. */
        observe(data) {
            if (destroyed) return;
            const census = data?.active_chat_activities;
            if (!Array.isArray(census) && data?.accounting?.available !== false) return;
            const complete = Array.isArray(census) && data?.active_chat_activities_complete !== false
                && data?.supervisor_ready !== false;
            observation = observeContext(observation, Array.isArray(census) ? census : [], chatId, { complete });
            render();
        },
        hasPendingSave: () => Boolean(saving),
        pendingSave: () => saving || Promise.resolve(lastOk),
        destroy() {
            destroyed = true;
            capsule.destroy();
            el.removeEventListener('click', onChoose);
            strip.removeEventListener('keydown', onKey);
        },
    };
}

/**
 * @param {object} deps
 * @param {Element} deps.row  the composer's `.chat-toolbar-row`
 * @param {(suffix: string) => Element|null} deps.byId  the instance-namespaced lookup
 * @param {typeof fetch} deps.apiFetch
 * @param {(body: object) => Promise<object>} deps.saveEffortRange  `apiClient.ownerEffortRange`
 * @param {(body: object) => Promise<object>} [deps.saveSubagentsMode]  the room's subagents mode writer
 * @param {(body: object) => Promise<object>} [deps.saveChatModel]  the room's model writer (`apiClient.ownerChatModel`)
 * @param {(path: string) => void} [deps.openSettings]  a Settings page: Agents, where subagents are turned on; Accounts
 *   and keys (`accounts`), where a model source is added
 * @param {(message: string, tone?: string) => void} deps.showToast
 * @param {(force?: boolean) => Promise<void>|void} deps.refreshState  a forced `/api/state` re-read
 * @param {() => void} [deps.onLayout]  a strip opened or closed on its own line
 * @param {number|null} [deps.chatId]  the room whose works' observed context mode the Context capsule shows (Main: 1)
 * @param {HTMLElement} [deps.box]  independently sized page the composer docks inside
 * @param {() => boolean} [deps.readingBack]  the owner's own gesture took the room off its newest row (composer_fold.js)
 * @param {object} [deps.modelDeps]  the Model capsule's own seams (its catalog store, its confirming sheet)
 */
export function createComposerOwnerControls({ row, byId, apiFetch, saveEffortRange, saveSubagentsMode, saveChatModel, openSettings = () => {}, showToast, refreshState, onLayout, projectId = '', chatId = projectId ? null : 1, box = null, win = globalThis.window, doc = globalThis.document, readingBack, modelDeps = {} }) {
    let destroyed = false;
    const scoped = body => projectId ? { ...body, project_id: projectId } : body;
    const readback = reply => {
        if ((reply?.project_id || '') !== projectId) throw new Error('Settings response belongs to another room.');
        return reply?.cognition || reply;
    };
    const saveContextMode = async mode => {
        const response = await apiFetch('/api/owner/context-mode', {
            method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(scoped({ mode })),
        });
        const reply = await response.json();
        if (!response.ok) throw new Error(reply?.error || 'Could not change context mode.');
        return readback(reply);
    };
    const saveRoomEffort = async range => readback(await saveEffortRange(scoped(range)));
    const capsules = [];
    // Measure the settled, closed row synchronously: keep names while it fits, else rings and
    // values. If long translated values still wrap, cap the widest with an ellipsis; accessible
    // names and open strips keep full words. The closed names never flicker between fits. Contributed chips take
    // only the room the capsules leave (composer_chips.js), so their slot is never part of the fit.
    const shownKids = () => Array.from(row.children || [])
        .filter((kid) => widthOf(kid) > 0 && !kid.hasAttribute?.('data-composer-chips'));
    // A phone's ☰ leads the line and never wraps, so it is not what the line is read by: while the row folds it goes
    // down beside the field (composer_fold.js) and the line it left still holds the capsules.
    function wraps() {
        const kids = shownKids().filter((kid) => !kid.classList?.contains?.('composer-menu'));
        const first = kids[0]?.getBoundingClientRect?.();
        return Boolean(first) && kids.some((kid) => kid.getBoundingClientRect().top >= first.bottom);
    }
    // The cap that takes `over` off the widest of `widths`, each cut to it (a water level).
    function waterLevel(widths, over) {
        const values = [...widths].sort((a, b) => b - a);
        let cap = 0;
        for (let k = 1, sum = 0; k <= values.length; k += 1) {
            sum += values[k - 1];
            cap = (sum - over) / k;
            if (k === values.length || cap >= values[k]) break;
        }
        return Math.max(0, Math.floor(cap));
    }
    function fitNames() {
        if (destroyed || !row?.dataset || capsules.some((capsule) => !capsule.isSettled())) return;
        delete row.dataset.ownerCompact;
        delete row.dataset.ownerGlyphs;
        row.style.setProperty('--capsule-value-cap', '');
        capsules.forEach((capsule) => { delete capsule.el.dataset.nameFit; capsule.measureName(); });
        if (!wraps()) return;
        // Names give way one at a time, the longest first, so two capsules of one value keep the words that tell them
        // apart while the line can hold them (U123: never "Swarm · Max · Max" with room to spare).
        const nameWidth = (capsule) => widthOf(capsule.el.querySelector('.chat-capsule-name'));
        for (const capsule of [...capsules].sort((a, b) => nameWidth(b) - nameWidth(a))) {
            capsule.el.dataset.nameFit = 'off';
            if (!wraps()) return;
        }
        row.dataset.ownerCompact = 'true';
        const rowStyle = win?.getComputedStyle?.(row);
        const room = widthOf(row) - px(rowStyle?.paddingLeft) - px(rowStyle?.paddingRight);
        if (!(room > 0) || !wraps()) return;  // not laid out (the unit-test DOM), or the line holds
        const over = () => {
            const kids = shownKids();
            return kids.reduce((sum, kid) => sum + widthOf(kid), 0) + px(rowStyle?.columnGap) * (kids.length - 1) - room + 1;  // 1 px of slack
        };
        const valueWidth = (capsule) => widthOf(capsule.el.querySelector('.chat-capsule-value'));
        let cap = waterLevel(capsules.map(valueWidth), over());
        // The model keeps its value (owner 10.11, «10. A»): where the cap would cut it, Subagents, Context and Effort
        // fold to their glyphs first, and only a line that still wraps cuts the model's own value.
        const model = capsules.find((capsule) => capsule.kind === 'model');
        if (model && cap < valueWidth(model)) {
            row.dataset.ownerGlyphs = 'true';
            if (!wraps()) return;
            cap = waterLevel([valueWidth(model)], over());
        }
        row.style.setProperty('--capsule-value-cap', `${cap}px`);
        row.dataset.ownerCompact = 'values';
    }
    // A capsule settled closed refits the names and asks the fold again (a short keyboard room folds the row then).
    let fold = null;
    const shared = { showToast, refreshState, onLayout, onSettled: () => { fitNames(); fold?.sync(); }, win, doc };
    const contextEl = byId('context-mode');
    const context = contextEl ? createContextModeControl({ el: contextEl, row: row || contextEl.parentElement, chatId, saveContextMode, ...shared }) : null;
    const effort = row ? createEffortRangeControl({ row, saveEffortRange: saveRoomEffort, ...shared }) : null;
    const subagentsEl = row && saveSubagentsMode ? byId('subagents-mode') : null;
    const saveRoomSubagents = async ({ mode }) => {
        const reply = await saveSubagentsMode(scoped({ mode }));
        return { subagents_mode: readback(reply)?.subagents_mode ?? reply?.subagents_mode };
    };
    const subagents = subagentsEl ? createSubagentsControl({ el: subagentsEl, row, capsule: createCapsule,
        save: saveRoomSubagents, openSettings: () => openSettings('agents'), ...shared }) : null;
    // The room's Model, first in the row. Its endpoint answers the room's own choice, so it scopes and checks itself.
    const modelEl = row && saveChatModel ? byId('model-mode') : null;
    const model = modelEl ? createModelControl({ el: modelEl, row, projectId, chatId, save: saveChatModel,
        openSettings: () => openSettings('accounts'), ...shared, ...modelDeps }) : null;
    const controls = [model, subagents, context, effort].filter(Boolean);
    for (const control of controls) capsules.push(control.capsule);
    // One capsule is open at a time.
    for (const capsule of capsules) capsule.peers = capsules.filter((other) => other !== capsule);
    win?.addEventListener?.('resize', fitNames);
    const stopObserving = observeLayout(win, [row], fitNames);
    // A draft that grows lifts the row with its open capsule: the capsule's hint finds its place again.
    const stopField = observeLayout(win, [row?.closest?.('.chat-input-wrap')?.querySelector?.('.chat-field')],
        () => capsules.forEach((capsule) => capsule.sayHint()));
    const relayout = () => { if (!destroyed) { fitNames(); capsules.forEach((capsule) => capsule.relayout()); } };
    // Descendant fitting cannot resize this docking box, so mount/show/resize settle before paint.
    // Text is observed at the microtask boundary, not our geometry attributes; unchanged words stay.
    const boxObserver = box && typeof win?.ResizeObserver === 'function' ? new win.ResizeObserver(relayout) : null;
    boxObserver?.observe(box);
    // Own values only refit the closed line: realigning an open strip would cancel its motion. A hint's words stand
    // over the page and size nothing in the row (ui_hint.js), so they refit nothing.
    const nodeOf = (record) => (record.target.nodeType === 1 ? record.target : record.target.parentNode);
    const ownValue = (record) => Boolean(nodeOf(record)?.closest?.('.chat-capsule-value'));
    const inHint = (record) => Boolean(nodeOf(record)?.closest?.('.ui-hint'));
    const onText = (records) => {
        const moving = (records || []).filter((record) => !inHint(record));
        if (!records?.length) relayout();
        else if (moving.length && moving.every(ownValue)) fitNames();
        else if (moving.length) relayout();
    };
    const textObserver = row && typeof win?.MutationObserver === 'function' ? new win.MutationObserver(onText) : null;
    textObserver?.observe(row, { childList: true, characterData: true, subtree: true });
    // On a phone the row folds away while the owner reads back through the room's history (composer_fold.js).
    fold = box ? createComposerFold({ feed: box.querySelector('#chat-messages, .chat-messages'),
        wrap: row?.closest?.('.chat-input-wrap'), row, win, doc, readingBack }) : null;

    return {
        model,
        context,
        effort,
        subagents,
        // A room's values (Main's state, or the named Project's cognition) and, for every room, whether the catalog
        // offers subagents at all (`subagents_availability`).
        syncState(data) {
            if (destroyed) return;
            const values = projectId ? data?.projects?.find(project => project.id === projectId)?.cognition : data;
            subagents?.syncState(values, data?.subagents_availability);
            model?.syncState(data);
            context?.observe(data);
            if (!values) return;
            context?.syncState(values); effort?.syncState(values); fitNames();
        },
        hasPendingSave: () => controls.some((control) => control.hasPendingSave()),
        pendingSave: async () => (await Promise.all(controls
            .filter(control => control.hasPendingSave()).map(control => control.pendingSave()))).every(ok => ok !== false),
        destroy() {
            destroyed = true;
            stopObserving();
            stopField();
            win?.removeEventListener?.('resize', fitNames);
            boxObserver?.disconnect();
            textObserver?.disconnect();
            for (const control of controls) control.destroy();
            fold?.destroy();
            if (row?.dataset) { delete row.dataset.ownerCompact; delete row.dataset.ownerGlyphs; }
            row?.style?.setProperty?.('--capsule-value-cap', '');
        },
    };
}
