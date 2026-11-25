import { apiFetch } from './api_client.js';
import { isStackLayout } from './layout.js';

/**
 * Resizable side sections: edge drag-handles resize the sidebar and the Project panel, whose widths live in
 * --sidebar-width / --project-panel-width on :root, persisted (debounced) to /api/ui/preferences.
 * Disabled in the stack layout (layout.js). Width 0 = keep the CSS default.
 * The sidebar's width is its column's (Ouroboros's card and the Projects): the rail beside the column adds a share of
 * its own to the box (style.css --nav-rail-extra, owner 24.11: the room pays for the rail, not the list), which every
 * width read off the box gives back, so a stored width keeps its meaning and a drag never adds the rail twice.
 * While a handle is dragged, each frame writes the measured width into the dragged box's own --drag-width, which no
 * element inherits (style.css), and everything that lines up with the box follows it by layout: one element
 * restyles. The stored width on :root is inherited by every element, so writing it per frame restyled the whole
 * document, about half of a drag frame on the owner's Main in WebKit; it is written once, when the drag ends.
 * CW10 note: the DEVELOPMENT.md "no inline styles in JS" rule targets static styling
 * that belongs in a stylesheet — the drag's transient `userSelect:none` was that, and
 * is now the `.resizing-panels` class. A measured width in a narrowly named custom property is the exception
 * DEVELOPMENT "Design System" names.
 */
/** The narrowest and widest sidebar column, in CSS px: the stylesheet's --sidebar-min-width and --sidebar-max-width
 *  (ui.css), below which Ouroboros's card, the search row and the Projects list no longer hold whole (the sections
 *  and foot live in the external rail beside the column). The sidebar's contents are drawn in CSS px and the
 *  Interface size zooms the window (interface_size.js), so the px values follow every Interface size. */
export const SIDEBAR_MIN_FALLBACK = 270;
export const SIDEBAR_MAX = 560;
const ROOM_MIN_FALLBACK = 320;
const PX_VALUE = /^\s*(-?(?:\d+\.?\d*|\.\d+))px\s*$/i;

function pxValue(raw) {
    const match = typeof raw === 'string' && PX_VALUE.exec(raw);
    return match ? Number(match[1]) : NaN;
}

export function sidebarMinWidth(root = globalThis.document?.documentElement) {
    let read = NaN;
    try { read = pxValue(globalThis.getComputedStyle(root).getPropertyValue('--sidebar-min-width')); } catch {}
    return read > 0 ? Math.ceil(read) : SIDEBAR_MIN_FALLBACK;
}

export function sidebarMaxWidth(root = globalThis.document?.documentElement) {
    let read = NaN;
    try { read = pxValue(globalThis.getComputedStyle(root).getPropertyValue('--sidebar-max-width')); } catch {}
    return read > 0 ? Math.round(read) : SIDEBAR_MAX;
}

function railGeometryExtra(sidebar, style) {
    const rail = sidebar.querySelector?.('.nav-rail');
    if (!rail) return NaN;
    const railRect = rail.getBoundingClientRect?.().width;
    if (!(railRect > 0)) return NaN;
    // The rail replaces the column's start inset while the end keeps its own (style.css: padding-inline: 0 var(--space-3)).
    // Reading the laid-out rail track and inline padding excludes border, vertical scrollbar, and column rounding, and
    // works even when --nav-rail-extra stays an unregistered calc() string.
    const gridCols = String(style.gridTemplateColumns || '').trim();
    let railWidth = railRect;
    if (gridCols && gridCols !== 'none') {
        const tracks = gridCols.split(/\s+/).map(pxValue);
        if (tracks.length < 2 || !(tracks[0] > 0)) return NaN;
        railWidth = tracks[0];
    }
    const rtl = style.direction === 'rtl';
    const padStart = pxValue(style.paddingInlineStart);
    const padEnd = pxValue(style.paddingInlineEnd);
    const start = Number.isFinite(padStart) ? padStart : pxValue(rtl ? style.paddingRight : style.paddingLeft);
    const end = Number.isFinite(padEnd) ? padEnd : pxValue(rtl ? style.paddingLeft : style.paddingRight);
    if (!Number.isFinite(end)) return NaN;
    const gap = pxValue(style.columnGap);
    return Math.max(0, railWidth + (Number.isFinite(gap) ? gap : 0) + (Number.isFinite(start) ? start : 0) - end);
}

/** What the rail adds to the sidebar's box beside its column, in CSS px: derived from the laid-out rail and padding
 *  geometry first so an unregistered calc() in --nav-rail-extra never collapses to 0, falling back to a resolved px
 *  custom property when no rendered rail is present; 0 where the stylesheet adds none. */
export function sidebarRailExtra(sidebar) {
    if (!sidebar) return 0;
    let style = null;
    try { style = globalThis.getComputedStyle(sidebar); } catch {}
    if (!style) return 0;
    const measured = railGeometryExtra(sidebar, style);
    if (Number.isFinite(measured)) return measured;
    const read = pxValue(style.getPropertyValue?.('--nav-rail-extra'));
    return read > 0 ? read : 0;
}

export function setupResizablePanels(prefs) {
    const root = document.documentElement;
    const sidebarMin = sidebarMinWidth(root);
    const sidebar = document.getElementById('primary-sidebar');
    // CSS owns the room's remaining width, including when the window narrows without a handle drag; its maximum is the
    // whole box's, so the column's is that less the rail's share.
    const railExtra = () => sidebarRailExtra(sidebar);
    const roomMinWidth = () => {
        let read = NaN;
        try { read = pxValue(globalThis.getComputedStyle(root).getPropertyValue('--room-min-width')); } catch {}
        return read > 0 ? Math.ceil(read) : ROOM_MIN_FALLBACK;
    };
    const desktopMax = () => Math.max(sidebarMin, sidebarMaxWidth(root));
    const availableAppWidth = () => {
        const app = document.getElementById('app') || sidebar?.parentElement;
        let measured = NaN;
        try { measured = app?.getBoundingClientRect?.().width; } catch {}
        if (measured > 0) return measured;
        if (app?.clientWidth > 0) return app.clientWidth;
        return root?.clientWidth || globalThis.innerWidth || 0;
    };
    const sidebarMax = () => {
        const extra = railExtra();
        const maxCol = desktopMax();
        let cssMax = NaN;
        try { cssMax = sidebar ? pxValue(globalThis.getComputedStyle(sidebar).maxWidth) : NaN; } catch {}
        // CSS max-width remains the final fractional layout cap; round drag bounds to nearest integer so subpixel
        // layout quantization (e.g. WebKit 429.984375 px) agrees with Math.round on save and no-motion checks.
        if (cssMax > 0) return Math.max(sidebarMin, Math.round(cssMax - extra));
        // Computed maxWidth can stay an expression string (min/calc) when an inner custom property is unregistered:
        // fall back to the same room-preserving cap CSS declares (style.css: min(--sidebar-max-width + extra, --app-width - --room-min-width)).
        const appW = availableAppWidth();
        const viewportCap = appW > 0 ? Math.round(appW - roomMinWidth() - extra) : maxCol;
        return Math.max(sidebarMin, Math.min(maxCol, viewportCap));
    };
    let persistTimer = 0;
    const persist = (patch) => {
        clearTimeout(persistTimer);
        persistTimer = setTimeout(() => {
            apiFetch('/api/ui/preferences', {
                method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(patch),
            }).catch(() => {});
        }, 400);
    };
    const clamp = (width, min, max) => Math.max(min, Math.min(max, width));
    // Restore within the static desktop column bounds (never clamping to a transient narrow viewport: CSS max-width
    // caps display until the window widens, so a stored desktop width recovers without posting a narrower preference).
    const storedSidebar = Number(prefs?.sidebar_width) || 0;
    const restoredSidebar = clamp(storedSidebar, sidebarMin, desktopMax());
    if (storedSidebar > 0) root.style.setProperty('--sidebar-width', `${restoredSidebar}px`);
    if (storedSidebar > 0 && storedSidebar !== restoredSidebar) persist({ sidebar_width: restoredSidebar });
    if (Number(prefs?.project_panel_width) > 0) root.style.setProperty('--project-panel-width', `${prefs.project_panel_width}px`);
    // `beside` is what the box holds beside the width the handle owns (the sidebar's rail; none for the Project panel).
    // `preserveUnchanged` keeps the sidebar's unshown desktop preference intact across no-motion taps, out-and-back
    // drags (including at a narrow viewport's CSS cap), and cancelled gestures; the Project panel keeps its original
    // end-of-gesture semantics (including flushing a pending frame and saving its displayed width on pointercancel).
    const makeHandle = (target, cssVar, dir, prefKey, min, max, beside = () => 0, { preserveUnchanged = false } = {}) => {
        if (!target) return;
        const handle = document.createElement('div');
        handle.className = `resize-handle resize-handle-${dir}`;
        handle.title = 'Drag to resize';
        target.appendChild(handle);
        let startX = 0, startW = 0, dragging = false, moved = false, frame = 0, wanted = 0;
        // One write per displayed frame (a mouse reports more often than the screen draws), on the dragged box only.
        const apply = () => {
            frame = 0;
            target.style.setProperty('--drag-width', `${wanted}px`);
        };
        handle.addEventListener('pointerdown', (e) => {
            if (isStackLayout()) return;  // the stack layout has a drawer, not a resizable column
            dragging = true; moved = false;
            startX = e.clientX;
            startW = target.getBoundingClientRect().width - beside();
            try { handle.setPointerCapture(e.pointerId); } catch {}
            document.body.classList.add('resizing-panels');  // CW10: class, not inline style
            e.preventDefault();
        });
        handle.addEventListener('pointermove', (e) => {
            if (!dragging) return;
            const delta = dir === 'right' ? (e.clientX - startX) : (startX - e.clientX);
            if (preserveUnchanged && !moved && delta === 0) return;
            moved = true;
            wanted = clamp(Math.round(startW + delta), min, max());
            if (!frame) frame = requestAnimationFrame(apply);
        });
        const finish = (e, cancelled) => {
            if (!dragging) return;
            const discard = preserveUnchanged && cancelled;
            if (frame) {
                cancelAnimationFrame(frame);
                frame = 0;
                if (!discard) apply();
            }
            dragging = false; document.body.classList.remove('resizing-panels');  // CW10
            try { handle.releasePointerCapture(e.pointerId); } catch {}
            if (!discard && (!preserveUnchanged || moved)) {
                // Keep the width that is shown: a Project panel dragged past the content area's edge stops at it. A box
                // hidden mid-drag (Escape closed the Project) shows no width, and the stored one stays.
                const box = target.getBoundingClientRect().width;
                const shown = box > 0 ? Math.round(box - beside()) : 0;
                if (shown && (!preserveUnchanged || shown !== Math.round(startW))) {
                    root.style.setProperty(cssVar, `${shown}px`);
                    persist({ [prefKey]: shown });
                }
            }
            moved = false;
            target.style.removeProperty('--drag-width');
        };
        handle.addEventListener('pointerup', (e) => finish(e, false));
        handle.addEventListener('pointercancel', (e) => finish(e, true));
    };
    makeHandle(sidebar, '--sidebar-width', 'right', 'sidebar_width', sidebarMin, sidebarMax, railExtra, { preserveUnchanged: true });
    makeHandle(document.getElementById('project-panel'), '--project-panel-width', 'left', 'project_panel_width', 320, () => 1100);
}
