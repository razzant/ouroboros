// One shell-level owner for stale document selection. Content remains selectable.
const EDITABLE = 'textarea, [contenteditable="true"], input:not([type]), input[type="text"], '
    + 'input[type="search"], input[type="url"], input[type="email"], input[type="tel"], '
    + 'input[type="password"], input[type="number"], input[type="date"]';

function editable(node) {
    if (node?.isContentEditable === true) return true;
    return Boolean(node?.closest?.(EDITABLE));
}

function chromeTarget(doc, target) {
    const element = target?.nodeType === 1 ? target : target?.parentElement;
    if (!element) return false;
    if (element.closest?.('button, a, select, summary, label, input')) return true;
    const style = doc.defaultView?.getComputedStyle?.(element);
    return style?.userSelect === 'none' || style?.webkitUserSelect === 'none';
}

function selectedEditable(selection, element) {
    if (!element || !selection) return false;
    const anchor = selection.anchorNode;
    const focus = selection.focusNode;
    return Boolean(anchor && focus && (element.contains?.(anchor) || element === anchor)
        && (element.contains?.(focus) || element === focus));
}

function selectablePoint(doc, event) {
    const x = Number(event?.clientX);
    const y = Number(event?.clientY);
    if (!Number.isFinite(x) || !Number.isFinite(y)) return true;
    const caret = doc.caretRangeFromPoint?.(x, y)
        || (() => {
            const position = doc.caretPositionFromPoint?.(x, y);
            return position ? { startContainer: position.offsetNode } : null;
        })();
    if (!caret) return true;
    return caret.startContainer?.nodeType === 3;
}

function scrollbarPress(target, event) {
    const element = target?.nodeType === 1 ? target : target?.parentElement;
    if (!element) return false;
    const x = Number(event?.offsetX);
    const y = Number(event?.offsetY);
    if (element.scrollHeight > element.clientHeight && Number.isFinite(x) && x >= element.clientWidth) return true;
    if (element.scrollWidth > element.clientWidth && Number.isFinite(y) && y >= element.clientHeight) return true;
    return false;
}

function pointInSelection(selection, event) {
    const x = Number(event?.clientX);
    const y = Number(event?.clientY);
    if (!Number.isFinite(x) || !Number.isFinite(y)) return false;
    for (let index = 0; index < (selection.rangeCount || 0); index += 1) {
        let rects;
        try { rects = selection.getRangeAt(index).getClientRects(); } catch { rects = []; }
        for (const rect of rects || []) {
            if (x >= rect.left && x <= rect.right && y >= rect.top && y <= rect.bottom) return true;
        }
    }
    return false;
}

/** A plain main-button press outside a scrollbar. Touch scrolling and handles remain native, so a touch or a coarse
 * pen is never one: this seam is mouse/pen only. */
export function ordinaryPress(event, doc = globalThis.document) {
    if (event?.isPrimary === false || (event?.button != null && event.button !== 0)) return false;
    if (event?.pointerType === 'touch'
        || (event?.pointerType === 'pen' && doc?.documentElement?.dataset?.pointer === 'coarse')) return false;
    if (event?.shiftKey || event?.ctrlKey || event?.metaKey || event?.altKey) return false;
    return !scrollbarPress(event?.target, event);
}

/** Dismiss stale selection before shell controls cancel native mousedown behavior. `onPress` hears every ordinary
 * press, also one the browser collapses natively: after it no selection kept from before may come back. */
export function bindDocumentSelectionDismissal(doc = globalThis.document, { onPress } = {}) {
    if (!doc?.addEventListener || !doc?.getSelection) return () => {};
    const handler = (event) => {
        if (!ordinaryPress(event, doc)) return;
        onPress?.();
        const selection = doc.getSelection?.();
        if (!selection || selection.isCollapsed || !selection.rangeCount) return;
        if (editable(doc.activeElement) && selectedEditable(selection, doc.activeElement)) return;
        // Selectable prose keeps native collapse/drag; chrome inside a spanning range clears.
        if (pointInSelection(selection, event) && !chromeTarget(doc, event?.target)
            && selectablePoint(doc, event)) return;
        selection.removeAllRanges?.();
    };
    doc.addEventListener('pointerdown', handler, true);
    return () => doc.removeEventListener?.('pointerdown', handler, true);
}
