import { apiFetch } from './api_client.js';
import { isStackLayout } from './layout.js';

/**
 * Resizable side sections: edge drag-handles resize the sidebar and the Project panel, whose widths live in
 * --sidebar-width / --project-panel-width on :root, persisted (debounced) to /api/ui/preferences.
 * Disabled in the stack layout (layout.js). Width 0 = keep the CSS default.
 * While a handle is dragged, each frame writes the measured width into the dragged box's own --drag-width, which no
 * element inherits (style.css), and everything that lines up with the box follows it by layout: one element
 * restyles. The stored width on :root is inherited by every element, so writing it per frame restyled the whole
 * document, about half of a drag frame on the owner's Main in WebKit; it is written once, when the drag ends.
 * CW10 note: the DEVELOPMENT.md "no inline styles in JS" rule targets static styling
 * that belongs in a stylesheet — the drag's transient `userSelect:none` was that, and
 * is now the `.resizing-panels` class. A measured width in a narrowly named custom property is the exception
 * DEVELOPMENT "Design System" names.
 */
export function setupResizablePanels(prefs) {
    const root = document.documentElement;
    let persistTimer = 0;
    const persist = (patch) => {
        clearTimeout(persistTimer);
        persistTimer = setTimeout(() => {
            apiFetch('/api/ui/preferences', {
                method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(patch),
            }).catch(() => {});
        }, 400);
    };
    if (Number(prefs?.sidebar_width) > 0) root.style.setProperty('--sidebar-width', `${prefs.sidebar_width}px`);
    if (Number(prefs?.project_panel_width) > 0) root.style.setProperty('--project-panel-width', `${prefs.project_panel_width}px`);
    const makeHandle = (target, cssVar, dir, prefKey, min, max) => {
        if (!target) return;
        const handle = document.createElement('div');
        handle.className = `resize-handle resize-handle-${dir}`;
        handle.title = 'Drag to resize';
        target.appendChild(handle);
        let startX = 0, startW = 0, dragging = false, frame = 0, wanted = 0;
        // One write per displayed frame (a mouse reports more often than the screen draws), on the dragged box only.
        const apply = () => {
            frame = 0;
            target.style.setProperty('--drag-width', `${wanted}px`);
        };
        handle.addEventListener('pointerdown', (e) => {
            if (isStackLayout()) return;  // the stack layout has a drawer, not a resizable column
            dragging = true; startX = e.clientX; startW = target.getBoundingClientRect().width;
            try { handle.setPointerCapture(e.pointerId); } catch {}
            document.body.classList.add('resizing-panels');  // CW10: class, not inline style
            e.preventDefault();
        });
        handle.addEventListener('pointermove', (e) => {
            if (!dragging) return;
            const delta = dir === 'right' ? (e.clientX - startX) : (startX - e.clientX);
            wanted = Math.max(min, Math.min(max, Math.round(startW + delta)));
            if (!frame) frame = requestAnimationFrame(apply);
        });
        const end = (e) => {
            if (!dragging) return;
            if (frame) { cancelAnimationFrame(frame); apply(); }
            dragging = false; document.body.classList.remove('resizing-panels');  // CW10
            try { handle.releasePointerCapture(e.pointerId); } catch {}
            // Keep the width that is shown: a Project panel dragged past the content area's edge stops at it. A box
            // hidden mid-drag (Escape closed the Project) shows no width, and the stored one stays.
            const shown = Math.round(target.getBoundingClientRect().width) || 0;
            if (shown) {
                root.style.setProperty(cssVar, `${shown}px`);
                persist({ [prefKey]: shown });
            }
            target.style.removeProperty('--drag-width');
        };
        handle.addEventListener('pointerup', end);
        handle.addEventListener('pointercancel', end);
    };
    makeHandle(document.getElementById('primary-sidebar'), '--sidebar-width', 'right', 'sidebar_width', 180, 560);
    makeHandle(document.getElementById('project-panel'), '--project-panel-width', 'left', 'project_panel_width', 320, 1100);
}
