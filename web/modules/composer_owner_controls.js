// The composer's controls that write GLOBAL owner settings (docs/DESIGN.md "Composer
// owner controls"): the Nano/Low/Max context mode and the effort range. Both save
// through their audited owner endpoint, show the server's refusal as a toast, and
// resync every open composer from the next `/api/state` snapshot. Swarm is not one of
// them: it is a per-message flag that never leaves the frame it rides.
import { fmt } from './i18n.js';
import {
    EFFORT_OPTIONS, EFFORT_RANGE_DEFAULT, OWNER_TIERS, effortText, normalizeEffortRange, ownerLevelIndex,
} from './effort_levels.js';

const N = OWNER_TIERS.length - 1;
const HANDLES = ['min', 'rec', 'max'];
const HOVER_OPEN_MS = 160;
// After the mouse leaves, the open strip waits this long before it closes, and a margin
// around it counts as still being there (a forgiving hover zone, like a menu's).
const HOVER_CLOSE_MS = 1000;
const HOVER_ZONE_PX = 32;
const HOVER_MEDIA = '(hover: hover) and (pointer: fine)';
// The segments' side padding (`--effort-seg-pad`, px): var(--space-2) when there is room, never
// below var(--space-1) beside the pills, down to 3 px when the strip stands alone (from 4 px down
// the pill's corners tighten with the padding, so the word's box stays inside it: `data-tight`).
// Below that the strip scrolls inside itself.
const SEG_PAD = Object.freeze({ max: 8, beside: 4, alone: 3 });
const clamp = (value, lo, hi) => Math.min(hi, Math.max(lo, value));
const px = (value) => Number.parseFloat(value) || 0;
// Fractional widths: whole-pixel offsets lose up to a pixel per element, enough to wrap the row.
const widthOf = (node) => {
    const width = node?.getBoundingClientRect?.().width;
    return Number.isFinite(width) && width > 0 ? width : Number(node?.offsetWidth) || 0;
};

/** Each handle pushes the ones it meets: min and max carry the recommended level along, and
 *  the recommended level carries a bracket past which it moves; min and max never cross. */
export function applyHandle(base, which, value) {
    const s = { ...base };
    const v = clamp(value, 0, N);
    if (which === 'min') { s.min = Math.min(v, s.max); s.rec = Math.max(s.rec, s.min); }
    else if (which === 'max') { s.max = Math.max(v, s.min); s.rec = Math.min(s.rec, s.max); }
    else { s.rec = v; s.min = Math.min(s.min, v); s.max = Math.max(s.max, v); }
    return s;
}

export function snapRange(v) {
    const s = { min: Math.round(v.min), rec: Math.round(v.rec), max: Math.round(v.max) };
    s.rec = clamp(s.rec, s.min, s.max);
    return s;
}

/** A tap off the handles: inside the range it sets the recommended level, outside it the bracket on that side moves. */
export function nearestHandle(committed, segment) {
    if (segment >= committed.min && segment <= committed.max) return 'rec';
    return segment < committed.min ? 'min' : 'max';
}

const indicesOf = (range) => ({
    min: ownerLevelIndex(range.min), rec: ownerLevelIndex(range.recommended), max: ownerLevelIndex(range.max),
});
const sameIndices = (a, b) => Boolean(a && b) && HANDLES.every((key) => a[key] === b[key]);

/* The ring glyph: the range in miniature (a 300° arc; the band is min..max, the dot the recommended level). */
const RING = { cx: 92, cy: 92, r: 70, a0: 120, sweep: 300 };
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

function effortRangeMarkup() {
    const segments = EFFORT_OPTIONS.map((option) =>
        `<span class="chat-effort-seg" data-effort-seg="${option.value}"><span class="chat-effort-seg-text">${option.label}</span></span>`).join('');
    return `<button type="button" class="chat-effort-head" aria-expanded="false" aria-label="Effort"><span class="chat-effort-glyph" aria-hidden="true"><svg viewBox="0 0 184 184"><path class="chat-effort-glyph-rail" data-ring-rail d="${ringArc(RING.a0, RING.a0 + RING.sweep)}" fill="none" stroke-width="22" stroke-linecap="round"></path><path class="chat-effort-glyph-band" data-ring-band fill="none" stroke-width="26" stroke-linecap="round"></path><circle class="chat-effort-glyph-dot" data-ring-rec r="22"></circle></svg></span></button>`
        + '<div class="chat-effort-body">'
        + '<div class="chat-effort-strip" data-effort-strip data-scroll="none"><div class="chat-effort-band" data-effort-band aria-hidden="true"></div>'
        + '<div class="chat-effort-handle chat-effort-rec" data-handle="rec" role="slider" tabindex="-1" aria-label="Recommended effort"></div>'
        + segments
        + '<div class="chat-effort-handle chat-effort-cap chat-effort-min" data-handle="min" role="slider" tabindex="-1" aria-label="Minimum effort"></div>'
        + '<div class="chat-effort-handle chat-effort-cap chat-effort-max" data-handle="max" role="slider" tabindex="-1" aria-label="Maximum effort"></div></div>'
        + '</div>';
}

/**
 * The effort range control (variant D): a round button with the ring glyph that opens
 * inline into the seven-level strip with min/max brackets and the recommended pill. The
 * open strip stays on the composer's line: beside the pills when it fits, alone (the pills
 * step aside) when it does not, scrolling inside itself only when nothing else fits.
 *
 * Hover is an accelerator only (`(hover: hover) and (pointer: fine)`): a press pins the
 * strip open and unpins it; touch and keyboard use the press. Saves happen on gesture
 * end through the owner endpoint, serialized (last wins); a failed save shows the
 * server's sentence and returns the control to the server's value. A stored tier
 * outside the seven owner tiers (`minimal`) is shown at the nearest owner tier and is
 * rewritten only when the owner moves that handle.
 */
export function createEffortRangeControl({
    row, saveEffortRange, showToast, refreshState = () => {}, onLayout = () => {},
    win = globalThis.window, doc = globalThis.document,
}) {
    const el = doc.createElement('div');
    el.className = 'chat-effort-range';
    el.setAttribute('data-effort-range', '');
    el.setAttribute('role', 'group');
    el.setAttribute('aria-label', 'Effort range');
    el.dataset.open = 'false';
    el.dataset.known = 'false';
    el.innerHTML = effortRangeMarkup();
    const q = (selector) => el.querySelector(selector);
    const head = q('.chat-effort-head');
    const body = q('.chat-effort-body');
    const strip = q('[data-effort-strip]');
    const band = q('[data-effort-band]');
    const ring = { band: q('[data-ring-band]'), rec: q('[data-ring-rec]') };
    const segs = Array.from(el.querySelectorAll('.chat-effort-seg'));
    const handles = { min: q('.chat-effort-min'), rec: q('.chat-effort-rec'), max: q('.chat-effort-max') };
    const pills = row.querySelector('.chat-composer-pills');
    if (pills && typeof pills.insertAdjacentElement === 'function') pills.insertAdjacentElement('afterend', el);
    else row.appendChild(el);

    // `hover` exists only with a mouse; the chat unit-test DOM carries no matchMedia at all.
    const hoverCapable = typeof win?.matchMedia === 'function' && win.matchMedia(HOVER_MEDIA)?.matches === true;
    // `stored` is the server's value; `draft` the tiers the server will hold once every queued
    // save landed (equal to `stored` when nothing is pending); `shown` the owner-tier indices.
    const state = {
        stored: { ...EFFORT_RANGE_DEFAULT }, draft: { ...EFFORT_RANGE_DEFAULT }, shown: indicesOf(EFFORT_RANGE_DEFAULT),
        view: null, drag: null, open: false, pinned: false, saving: null, queued: null, lastOk: true, destroyed: false,
    };
    head.setAttribute('aria-expanded', 'false');
    for (const which of HANDLES) { handles[which].setAttribute('role', 'slider'); handles[which].tabIndex = -1; }
    let hoverTimer = 0;
    let leaveTimer = 0;
    let lastPointer = null;

    /* --- geometry: segment edges in px, measured from the DOM (labels differ in width) --- */
    function edges() {
        const e = segs.map((seg) => Number(seg.offsetLeft));
        e.push(Number(segs[N].offsetLeft) + Number(segs[N].offsetWidth));
        return e.every(Number.isFinite) && e[N + 1] > e[0] ? e : null;
    }
    function boundaryAt(event) {
        const e = edges();
        if (!e) return null;
        const x = clamp(event.clientX - strip.getBoundingClientRect().left + (Number(strip.scrollLeft) || 0), e[0], e[N + 1]);
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
        return { which: nearestHandle(state.shown, segment), start: segment };
    }

    /* --- rendering: every mark sits on whole levels, a drag included (the pill always covers
       the word it names, and a pushed bracket moves with it, never behind it) --- */
    function render() {
        const s = snapRange(state.view || state.shown);
        const e = edges();
        if (e) {
            band.style.left = `${e[s.min]}px`;
            band.style.width = `${e[s.max + 1] - e[s.min]}px`;
            handles.rec.style.left = `${e[s.rec]}px`;
            handles.rec.style.width = `${e[s.rec + 1] - e[s.rec]}px`;
            handles.min.style.left = `${e[s.min]}px`;
            handles.max.style.left = `${e[s.max + 1]}px`;  // the bracket draws to the left of its edge
        }
        segs.forEach((seg, i) => { seg.dataset.in = String(i >= s.min && i <= s.max); seg.dataset.rec = String(i === s.rec); });
        el.dataset.dragging = String(Boolean(state.drag?.moved));
        ring.band.setAttribute('d', ringArc(ringAngle(s.min), ringAngle(s.max)));
        const dot = ringPoint(ringAngle(s.rec));
        ring.rec.setAttribute('cx', dot.x.toFixed(2));
        ring.rec.setAttribute('cy', dot.y.toFixed(2));
        // The recommended level may go anywhere (it pushes the brackets); a bracket stops at the other one.
        const bounds = { min: [0, s.max], rec: [0, N], max: [s.min, N] };
        for (const which of HANDLES) {
            handles[which].setAttribute('aria-valuemin', String(bounds[which][0]));
            handles[which].setAttribute('aria-valuemax', String(bounds[which][1]));
            handles[which].setAttribute('aria-valuenow', String(s[which]));
            handles[which].setAttribute('aria-valuetext', effortText(OWNER_TIERS[s[which]]));
            handles[which].dataset.dragging = String(state.drag?.which === which);
        }
        const stored = state.stored;
        head.title = fmt('Effort: {level}, range {min}–{max}. Applies to new work.', {
            level: effortText(stored.recommended), min: effortText(stored.min), max: effortText(stored.max),
        });
        head.setAttribute('aria-label', head.title);
    }

    /* --- the model: drag preview, commit on gesture end, serialized saves --- */
    function beginDrag(which) { state.drag = { which }; state.view = { ...state.shown }; render(); }
    // A drag builds on its own preview: a bracket the pill pushed stays pushed when the pointer
    // comes back; cancel restores the committed range.
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
        if (state.open && !state.pinned && hoverCapable) closeSoon();
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

    /* --- open / pin state machine --- */
    // Focus keeps a hover-opened strip open only when the keyboard put it there: browsers report
    // a handle focused after a mouse press as :focus-visible, so the last input decides instead.
    let keyboardFocus = false;
    const stillHere = () => {
        try { return Boolean(el.matches(':hover') || (keyboardFocus && el.contains(doc.activeElement))); } catch { return false; }
    };
    // The open strip forgives a mouse that drifts a little: inside the zone (the control plus a
    // margin) it stays; outside, it closes after the delay unless the mouse comes back first. A
    // pinned strip, a drag and keyboard focus keep it open regardless.
    function inZone(point) {
        const r = point && typeof el.getBoundingClientRect === 'function' ? el.getBoundingClientRect() : null;
        return Boolean(r) && point.x >= r.left - HOVER_ZONE_PX && point.x <= r.right + HOVER_ZONE_PX
            && point.y >= r.top - HOVER_ZONE_PX && point.y <= r.bottom + HOVER_ZONE_PX;
    }
    function closeSoon() {
        clearTimeout(leaveTimer);
        startTracking();
        leaveTimer = setTimeout(() => {
            leaveTimer = 0;
            if (state.open && !state.pinned && !state.drag && !stillHere() && !inZone(lastPointer)) setOpen(false);
        }, HOVER_CLOSE_MS);
    }
    let tracking = false;
    const onDocMove = (event) => {
        if (event.pointerType && event.pointerType !== 'mouse') return;
        lastPointer = { x: event.clientX, y: event.clientY };
        if (!state.open || state.pinned) { stopTracking(); return; }
        if (inZone(lastPointer)) { clearTimeout(leaveTimer); leaveTimer = 0; }
        else if (!leaveTimer) closeSoon();
    };
    const onDocLeave = () => { lastPointer = null; if (state.open && !state.pinned) closeSoon(); };
    function startTracking() {
        if (tracking || !hoverCapable) return;
        tracking = true;
        doc.addEventListener('pointermove', onDocMove);
        doc.documentElement?.addEventListener?.('pointerleave', onDocLeave);
    }
    function stopTracking() {
        if (!tracking) return;
        tracking = false;
        doc.removeEventListener('pointermove', onDocMove);
        doc.documentElement?.removeEventListener?.('pointerleave', onDocLeave);
    }

    /* --- fit: beside the pills, alone, or scrolling — measured, no breakpoint (a translation
       or another font needs none) --- */
    const styleOf = (node) => { try { return win?.getComputedStyle?.(node) || null; } catch { return null; } };
    let pillsWidth = 0;
    function measurePills() {
        // The wrapper stretches on a phone; its children are what has to fit.
        if (!pills || row.dataset?.effortSolo === 'true') return pillsWidth;
        const kids = Array.from(pills.children || []).filter((kid) => !kid.hidden);
        const gap = px(styleOf(pills)?.columnGap);
        pillsWidth = kids.reduce((sum, kid) => sum + widthOf(kid), 0) + gap * Math.max(0, kids.length - 1);
        return pillsWidth;
    }
    // Every input is independent of the padding it decides (words, button, borders, the row),
    // so a resize that follows a new padding finds the same answer.
    function measure() {
        const rowStyle = styleOf(row);
        const room = widthOf(row) - px(rowStyle?.paddingLeft) - px(rowStyle?.paddingRight) - 1;  // 1 px of slack
        if (!(room > 0)) return null;  // not laid out: a hidden pane, the unit-test DOM
        const words = segs.reduce((sum, seg) => sum + widthOf(seg.firstElementChild), 0);
        const elStyle = styleOf(el);
        const chrome = widthOf(head) + px(elStyle?.borderLeftWidth) + px(elStyle?.borderRightWidth)
            + px(styleOf(strip)?.marginRight);
        const padFor = (width) => Math.floor((width - chrome - words) / (2 * segs.length));
        const beside = pills ? padFor(room - measurePills() - px(rowStyle?.columnGap)) : padFor(room);
        let mode = 'inline';
        let pad = Math.min(SEG_PAD.max, beside);
        if (pills && beside < SEG_PAD.beside) {
            const alone = padFor(room);
            mode = alone < SEG_PAD.alone ? 'overflow' : 'solo';
            pad = clamp(alone, SEG_PAD.alone, SEG_PAD.max);
        } else if (!pills && beside < SEG_PAD.alone) {
            mode = 'overflow';
            pad = SEG_PAD.alone;
        }
        return { mode, pad };
    }
    function fit() {
        const fitted = state.open ? measure() : null;
        if (!fitted) return;
        el.style.setProperty('--effort-seg-pad', `${fitted.pad}px`);
        el.dataset.fit = fitted.mode;
        el.dataset.tight = String(fitted.pad <= SEG_PAD.beside);
        if (fitted.mode === 'inline') delete row.dataset.effortSolo;
        else row.dataset.effortSolo = 'true';
    }
    // A strip that scrolls shows where it continues and keeps the moved handle in view.
    // The fade at a scrolling strip's edge (var(--space-3)): a revealed handle stays clear of it.
    const REVEAL_MARGIN_PX = 12;
    function revealHandle(which = 'rec') {
        if (el.dataset.fit !== 'overflow') return;
        const e = edges();
        if (!e) return;
        const s = snapRange(state.view || state.shown);
        const cap = widthOf(handles.min);
        const [left, right] = which === 'min' ? [e[s.min], e[s.min] + cap]
            : which === 'max' ? [e[s.max + 1] - cap, e[s.max + 1]] : [e[s.rec], e[s.rec + 1]];
        const width = Number(strip.clientWidth) || 0;
        if (left - REVEAL_MARGIN_PX < strip.scrollLeft) strip.scrollLeft = Math.max(0, left - REVEAL_MARGIN_PX);
        else if (right + REVEAL_MARGIN_PX > strip.scrollLeft + width) strip.scrollLeft = right + REVEAL_MARGIN_PX - width;
        onStripScroll();
    }
    function onStripScroll() {
        // The levels' own end, not scrollWidth: a bracket's hit area may reach past the last level.
        const e = edges();
        const scrollable = (e ? e[N + 1] : Number(strip.scrollWidth) || 0) - (Number(strip.clientWidth) || 0);
        const at = Number(strip.scrollLeft) || 0;
        strip.dataset.scroll = scrollable <= 1 ? 'none' : at <= 1 ? 'start' : at >= scrollable - 1 ? 'end' : 'middle';
    }
    // Alone, the strip appears and disappears at once (no width animation), so the pills never
    // come back beside a strip that is still collapsing: the instant close is flushed before the
    // solo flag goes. The padding stays as fitted: beside the pills the strip is still collapsing,
    // and a wider padding now would push it onto a second line for those frames (the next open
    // fits again).
    function settleClosed() {
        if (el.dataset.fit && el.dataset.fit !== 'inline') void body.offsetWidth;
        delete el.dataset.fit;
        if (row?.dataset) delete row.dataset.effortSolo;
    }
    // A layout change (opening, a resize, a new padding) moves the marks with the words at once;
    // only an owner's gesture animates them.
    function settle(update) {
        el.dataset.settling = 'true';
        update();
        void strip.offsetWidth;
        delete el.dataset.settling;
    }
    function setOpen(open, { focus = false } = {}) {
        if (state.open === open && !focus) return;
        state.open = open;
        if (open) el.dataset.settling = 'true';
        if (open) fit();
        el.dataset.open = String(open);
        head.setAttribute('aria-expanded', String(open));
        for (const which of HANDLES) handles[which].tabIndex = open ? 0 : -1;
        if (!open) {
            state.pinned = false;
            clearTimeout(hoverTimer);
            clearTimeout(leaveTimer);
            leaveTimer = 0;
            stopTracking();
            if (state.drag) cancelDrag();
            settleClosed();
        }
        render();
        if (open) {
            void strip.offsetWidth;  // the opening frame lands without the marks' transitions
            delete el.dataset.settling;
            revealHandle('rec');
        }
        if (open && focus) handles.rec.focus({ preventScroll: true });
        onLayout();
    }
    const onHeadClick = (event) => {
        const viaKeyboard = event.detail === 0;
        if (hoverCapable) {
            if (!state.open) { state.pinned = true; setOpen(true, { focus: viaKeyboard }); }
            else if (!state.pinned) state.pinned = true;
            else { state.pinned = false; setOpen(false); }
        } else {
            setOpen(!state.open, { focus: viaKeyboard && !state.open });
        }
    };
    const onDocPointerDown = (event) => {
        keyboardFocus = false;
        if (state.open && !el.contains(event.target)) { state.pinned = false; setOpen(false); }
    };
    const onDocKeyDown = (event) => {
        keyboardFocus = true;
        if (event.key !== 'Escape' || !state.open) return;
        state.pinned = false;
        setOpen(false);
        head.focus({ preventScroll: true });
    };
    const onEnter = (event) => {
        if (event.pointerType !== 'mouse') return;
        clearTimeout(leaveTimer);
        leaveTimer = 0;
        stopTracking();
        if (!state.open) {
            clearTimeout(hoverTimer);
            hoverTimer = setTimeout(() => {
                if (state.destroyed) return;
                // Hover opens only where the round button stays under the mouse. Alone or
                // scrolling, the pills step aside and the strip moves under a still pointer, so a
                // click meant to pin would land on a level: there the press opens it, as on touch.
                const fitted = measure();
                if (fitted && fitted.mode !== 'inline') return;
                setOpen(true);
            }, HOVER_OPEN_MS);
        }
    };
    const onLeave = (event) => {
        if (event.pointerType !== 'mouse') return;
        clearTimeout(hoverTimer);
        lastPointer = { x: event.clientX, y: event.clientY };
        if (!state.open || state.pinned) return;
        startTracking();
        if (!inZone(lastPointer)) closeSoon();
    };

    /* --- pointer: a press on the strip grabs a handle (or picks one) and keeps the grab offset --- */
    const onDown = (event) => {
        if (event.button) return;
        event.preventDefault?.();
        const handleEl = event.target?.closest?.('[data-handle]');
        let which;
        let start;
        if (handleEl) { which = handleEl.dataset.handle; start = state.shown[which]; }
        else ({ which, start } = pick(event));
        beginDrag(which);
        if (!handleEl) dragTo(which, start);
        const grab = valueAt(event, which) - state.view[which];
        handles[which].focus({ preventScroll: true, focusVisible: false });  // no keyboard ring after a mouse press
        try { strip.setPointerCapture(event.pointerId); } catch { /* a synthetic event */ }
        const move = (ev) => {
            if (ev.pointerId !== event.pointerId || !state.drag) return;
            state.drag.moved = true;  // a press that stays put is a tap: its marks still animate
            dragTo(which, valueAt(ev, which) - grab);
        };
        const release = (ev) => {
            if (ev.pointerId !== event.pointerId) return;
            if (ev.pointerType === 'mouse' && Number.isFinite(ev.clientX)) lastPointer = { x: ev.clientX, y: ev.clientY };
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
        keyboardFocus = true;
        commit(snapRange(applyHandle(state.shown, which, clamp(target, 0, N))));
        revealHandle(which);
    };

    // The row and the strip change size without a window resize (a Project pane's divider, a
    // font that loads): measure again then, so the fit and the marks follow.
    const relayout = () => settle(() => { if (state.open) fit(); render(); onStripScroll(); });
    const onResize = () => { if (state.open) relayout(); };
    const observer = typeof win?.ResizeObserver === 'function' ? new win.ResizeObserver(() => {
        if (!state.destroyed) relayout();
    }) : null;
    observer?.observe(row);
    observer?.observe(strip);

    head.addEventListener('click', onHeadClick);
    strip.addEventListener('pointerdown', onDown);
    strip.addEventListener('scroll', onStripScroll, { passive: true });
    win?.addEventListener?.('resize', onResize);
    for (const which of HANDLES) handles[which].addEventListener('keydown', onHandleKey);
    doc.addEventListener('pointerdown', onDocPointerDown, true);
    doc.addEventListener('keydown', onDocKeyDown);
    if (hoverCapable) {
        el.addEventListener('pointerenter', onEnter);
        el.addEventListener('pointerleave', onLeave);
    }
    render();

    return {
        el,
        hoverCapable,
        isOpen: () => state.open,
        isPinned: () => state.pinned,
        shown: () => ({ ...state.shown }),
        stored: () => ({ ...state.stored }),
        /** Every composer shows the one global value: the `/api/state` snapshot's `effort_range`. */
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
            clearTimeout(hoverTimer);
            clearTimeout(leaveTimer);
            doc.removeEventListener('pointerdown', onDocPointerDown, true);
            doc.removeEventListener('keydown', onDocKeyDown);
            head.removeEventListener('click', onHeadClick);
            strip.removeEventListener('pointerdown', onDown);
            strip.removeEventListener('scroll', onStripScroll);
            win?.removeEventListener?.('resize', onResize);
            observer?.disconnect();
            stopTracking();
            if (row?.dataset) delete row.dataset.effortSolo;
            for (const which of HANDLES) handles[which].removeEventListener('keydown', onHandleKey);
            el.removeEventListener('pointerenter', onEnter);
            el.removeEventListener('pointerleave', onLeave);
            el.remove();
        },
    };
}

/**
 * @param {object} deps
 * @param {Element} deps.row  the composer's `.chat-toolbar-row`
 * @param {(suffix: string) => Element|null} deps.byId  the instance-namespaced lookup
 * @param {typeof fetch} deps.apiFetch
 * @param {(body: object) => Promise<object>} deps.saveEffortRange  `apiClient.ownerEffortRange`
 * @param {(message: string, tone?: string) => void} deps.showToast
 * @param {(force?: boolean) => Promise<void>|void} deps.refreshState  a forced `/api/state` re-read
 * @param {() => void} [deps.onLayout]  the strip opened or closed on its own line
 */
export function createComposerOwnerControls({ row, byId, apiFetch, saveEffortRange, showToast, refreshState, onLayout, win, doc }) {
    // Context-mode quick toggle: the owner endpoint hot-applies the setting
    // without a restart; Max -> Low is accepted only while Ouroboros is idle.
    const contextModeBtn = byId('context-mode');
    const onContextMode = async (event) => {
        const seg = event.target.closest('.chat-seg');
        if (!seg || contextModeBtn.dataset.disabled === 'true') return;
        const next = ['nano', 'low', 'max'].includes(seg.dataset.mode) ? seg.dataset.mode : 'max';
        const current = ['nano', 'low', 'max'].includes(contextModeBtn.dataset.contextMode) ? contextModeBtn.dataset.contextMode : 'max';
        if (next === current) return;
        contextModeBtn.dataset.disabled = 'true';
        const postMode = (mode) => apiFetch('/api/owner/context-mode', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ mode }),
        });
        try {
            const resp = await postMode(next);
            if (resp.ok) {
                contextModeBtn.dataset.contextMode = next;
            } else {
                let message = 'Could not change context mode.';
                try { const p = await resp.json(); if (p?.error) message = p.error; } catch {}
                showToast(message, 'error');
            }
        } catch (e) {
            showToast(`Could not change context mode: ${e.message || e}`, 'error');
            /* leave the current value; /api/state refresh will resync */
        } finally {
            contextModeBtn.dataset.disabled = 'false';
            refreshState(true);
        }
    };
    contextModeBtn?.addEventListener('click', onContextMode);
    const effort = row ? createEffortRangeControl({ row, saveEffortRange, showToast, refreshState, onLayout, win, doc }) : null;

    return {
        effort,
        syncState(data) { effort?.syncState(data); },
        hasPendingSave: () => Boolean(effort?.hasPendingSave()),
        pendingSave: () => (effort ? effort.pendingSave() : Promise.resolve(true)),
        destroy() {
            contextModeBtn?.removeEventListener('click', onContextMode);
            effort?.destroy();
        },
    };
}
