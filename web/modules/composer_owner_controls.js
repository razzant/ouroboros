// The composer's controls that write GLOBAL owner settings (docs/DESIGN.md "Composer
// owner controls"): the Nano/Low/Max context mode and the effort range. Both save
// through their audited owner endpoint, show the server's refusal as a toast, and
// resync every open composer from the next `/api/state` snapshot. Swarm is not one of
// them: it is a per-message flag that never leaves the frame it rides.
import { fmt } from './i18n.js';
import {
    EFFORT_OPTIONS, EFFORT_RANGE_DEFAULT, OWNER_TIERS, effortText, normalizeEffortRange, ownerLevelIndex, sameEffortRange,
} from './effort_levels.js';

const N = OWNER_TIERS.length - 1;
const HANDLES = ['min', 'rec', 'max'];
const HOVER_OPEN_MS = 160;
const HOVER_CLOSE_MS = 450;
const HOVER_MEDIA = '(hover: hover) and (pointer: fine)';
const clamp = (value, lo, hi) => Math.min(hi, Math.max(lo, value));

/** min and max push the recommended level along; the recommended level stays between them. */
export function applyHandle(base, which, value) {
    const s = { ...base };
    const v = clamp(value, 0, N);
    if (which === 'min') { s.min = Math.min(v, s.max); s.rec = Math.max(s.rec, s.min); }
    else if (which === 'max') { s.max = Math.max(v, s.min); s.rec = Math.min(s.rec, s.max); }
    else s.rec = clamp(v, s.min, s.max);
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

const RESET_SVG = '<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M3 12a9 9 0 1 0 9-9 9.75 9.75 0 0 0-6.74 2.74L3 8"></path><path d="M3 3v5h5"></path></svg>';

function effortRangeMarkup() {
    const segments = EFFORT_OPTIONS.map((option) =>
        `<span class="chat-effort-seg" data-effort-seg="${option.value}"><span class="chat-effort-seg-text">${option.label}</span></span>`).join('');
    return `<button type="button" class="chat-effort-head" aria-expanded="false" aria-label="Effort"><span class="chat-effort-glyph" aria-hidden="true"><svg viewBox="0 0 184 184"><path class="chat-effort-glyph-rail" data-ring-rail d="${ringArc(RING.a0, RING.a0 + RING.sweep)}" fill="none" stroke-width="22" stroke-linecap="round"></path><path class="chat-effort-glyph-band" data-ring-band fill="none" stroke-width="26" stroke-linecap="round"></path><circle class="chat-effort-glyph-dot" data-ring-rec r="22"></circle></svg></span></button>`
        + '<div class="chat-effort-body"><span class="chat-effort-label">Effort</span><i class="chat-effort-sep" aria-hidden="true"></i>'
        + '<div class="chat-effort-strip" data-effort-strip><div class="chat-effort-band" data-effort-band aria-hidden="true"></div>'
        + '<div class="chat-effort-handle chat-effort-rec" data-handle="rec" role="slider" tabindex="-1" aria-label="Recommended effort"></div>'
        + segments
        + '<div class="chat-effort-handle chat-effort-cap chat-effort-min" data-handle="min" role="slider" tabindex="-1" aria-label="Minimum effort"></div>'
        + '<div class="chat-effort-handle chat-effort-cap chat-effort-max" data-handle="max" role="slider" tabindex="-1" aria-label="Maximum effort"></div></div>'
        + `<button type="button" class="chat-effort-reset" tabindex="-1" aria-label="Reset effort range to Low, Medium, High" title="Reset to Low · Medium · High">${RESET_SVG}</button></div>`;
}

/**
 * The effort range control (variant D): a round button with the ring glyph that opens
 * inline into a labelled strip with min/max brackets and the recommended dot.
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
    const strip = q('[data-effort-strip]');
    const band = q('[data-effort-band]');
    const reset = q('.chat-effort-reset');
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

    /* --- geometry: segment edges in px, measured from the DOM (labels differ in width) --- */
    function edges() {
        const e = segs.map((seg) => Number(seg.offsetLeft));
        e.push(Number(segs[N].offsetLeft) + Number(segs[N].offsetWidth));
        return e.every(Number.isFinite) && e[N + 1] > e[0] ? e : null;
    }
    const xAt = (b, e) => { const i = clamp(Math.floor(b), 0, N); return e[i] + (e[i + 1] - e[i]) * (clamp(b, 0, N + 1) - i); };
    function boundaryAt(event) {
        const e = edges();
        if (!e) return null;
        const x = clamp(event.clientX - strip.getBoundingClientRect().left, e[0], e[N + 1]);
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

    /* --- rendering --- */
    function render() {
        const v = state.view || state.shown;
        const s = snapRange(v);
        const e = edges();
        if (e) {
            const x0 = xAt(v.min, e);
            const x1 = xAt(v.max + 1, e);
            const r0 = xAt(v.rec, e);
            const r1 = xAt(v.rec + 1, e);
            band.style.left = `${x0}px`;
            band.style.width = `${x1 - x0}px`;
            handles.rec.style.left = `${r0 + 2}px`;
            handles.rec.style.width = `${Math.max(0, r1 - r0 - 4)}px`;
            handles.min.style.left = `${x0}px`;
            handles.max.style.left = `${x1}px`;
        }
        segs.forEach((seg, i) => { seg.dataset.in = String(i >= s.min && i <= s.max); seg.dataset.rec = String(i === s.rec); });
        ring.band.setAttribute('d', ringArc(ringAngle(v.min), ringAngle(v.max)));
        const dot = ringPoint(ringAngle(v.rec));
        ring.rec.setAttribute('cx', dot.x.toFixed(2));
        ring.rec.setAttribute('cy', dot.y.toFixed(2));
        for (const which of HANDLES) {
            handles[which].setAttribute('aria-valuemin', '0');
            handles[which].setAttribute('aria-valuemax', String(N));
            handles[which].setAttribute('aria-valuenow', String(s[which]));
            handles[which].setAttribute('aria-valuetext', effortText(OWNER_TIERS[s[which]]));
            handles[which].dataset.dragging = String(state.drag?.which === which);
        }
        const atDefault = sameEffortRange(state.stored, EFFORT_RANGE_DEFAULT) && sameIndices(state.shown, indicesOf(EFFORT_RANGE_DEFAULT));
        reset.dataset.atDefault = String(atDefault);
        reset.tabIndex = state.open && !atDefault ? 0 : -1;
        const stored = state.stored;
        head.title = fmt('Effort: {level}, range {min}–{max}. Applies to new work.', {
            level: effortText(stored.recommended), min: effortText(stored.min), max: effortText(stored.max),
        });
    }

    /* --- the model: drag preview, commit on gesture end, serialized saves --- */
    function beginDrag(which) { state.drag = { which }; state.view = { ...state.shown }; render(); }
    function dragTo(which, value) { if (!state.drag) return; state.view = applyHandle(state.shown, which, value); render(); }
    function cancelDrag() { state.drag = null; state.view = null; render(); }
    function endDrag() {
        if (!state.drag) return;
        const next = snapRange(state.view);
        state.drag = null;
        state.view = null;
        commit(next);
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
    const stillHere = () => {
        try { return Boolean(el.matches(':hover') || el.querySelector(':focus-visible')); } catch { return false; }
    };
    function closeSoon() {
        clearTimeout(leaveTimer);
        leaveTimer = setTimeout(() => {
            if (state.open && !state.pinned && !state.drag && !stillHere()) setOpen(false);
        }, HOVER_CLOSE_MS);
    }
    function setOpen(open, { focus = false } = {}) {
        if (state.open === open && !focus) return;
        state.open = open;
        el.dataset.open = String(open);
        head.setAttribute('aria-expanded', String(open));
        for (const which of HANDLES) handles[which].tabIndex = open ? 0 : -1;
        if (!open) {
            state.pinned = false;
            clearTimeout(hoverTimer);
            clearTimeout(leaveTimer);
            if (state.drag) cancelDrag();
        }
        render();
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
        if (state.open && !el.contains(event.target)) { state.pinned = false; setOpen(false); }
    };
    const onDocKeyDown = (event) => {
        if (event.key !== 'Escape' || !state.open) return;
        state.pinned = false;
        setOpen(false);
        head.focus({ preventScroll: true });
    };
    const onEnter = (event) => {
        if (event.pointerType !== 'mouse') return;
        clearTimeout(leaveTimer);
        if (!state.open) { clearTimeout(hoverTimer); hoverTimer = setTimeout(() => { if (!state.destroyed) setOpen(true); }, HOVER_OPEN_MS); }
    };
    const onLeave = (event) => {
        if (event.pointerType !== 'mouse') return;
        clearTimeout(hoverTimer);
        if (state.open && !state.pinned) closeSoon();
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
        handles[which].focus({ preventScroll: true });
        try { strip.setPointerCapture(event.pointerId); } catch { /* a synthetic event */ }
        const move = (ev) => { if (ev.pointerId === event.pointerId) dragTo(which, valueAt(ev, which) - grab); };
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
    };
    const onReset = () => {
        state.shown = indicesOf(EFFORT_RANGE_DEFAULT);
        state.draft = { ...EFFORT_RANGE_DEFAULT };
        render();
        enqueue({ ...EFFORT_RANGE_DEFAULT });
        handles.rec.focus({ preventScroll: true });
    };

    head.addEventListener('click', onHeadClick);
    strip.addEventListener('pointerdown', onDown);
    for (const which of HANDLES) handles[which].addEventListener('keydown', onHandleKey);
    reset.addEventListener('click', onReset);
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
            for (const which of HANDLES) handles[which].removeEventListener('keydown', onHandleKey);
            reset.removeEventListener('click', onReset);
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
