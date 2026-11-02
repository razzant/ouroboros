// Shared focus and menu behavior. Domain actions and overlay markup stay with
// their callers; each binding has an explicit, synchronous lifetime.

import { leaveAsPicture, motionMs } from './motion.js';

// A closed menu's picture shrinks a little toward where it grew from as it fades (ui.css ui-popup-in, reversed).
const POPUP_LEAVES = [{ opacity: 1, transform: 'none' }, { opacity: 0, transform: 'scale(0.96)' }];

function available(element) {
    return element.isConnected && !element.matches(':disabled, [aria-disabled="true"]')
        && !element.closest('[hidden], [inert]') && element.getClientRects().length > 0
        && element.ownerDocument.defaultView.getComputedStyle(element).visibility !== 'hidden';
}

function focusable(root) {
    return Array.from(root.querySelectorAll(
        'a[href], button, input, select, textarea, summary, [tabindex], [contenteditable="true"]',
    )).filter((element) => element.tabIndex >= 0 && available(element));
}

/**
 * Install after mounting a dialog; dispose before removing it. Only boundary
 * Tab presses are intercepted, leaving native editing/radio behavior intact.
 * onEscape belongs to the caller's result contract (confirm/input/Project).
 * `preventScroll` returns focus without scrolling its container, for an opener
 * in a scrolled feed whose place must not move.
 */
export function bindDialogFocus(dialog, {
    initialFocus,
    returnFocus = dialog.ownerDocument.activeElement,
    onEscape,
    preventScroll = false,
} = {}) {
    const doc = dialog.ownerDocument;
    let disposed = false;
    const originalTabIndex = dialog.getAttribute('tabindex');
    if (originalTabIndex === null) dialog.tabIndex = -1;
    const onKey = (event) => {
        if (disposed || !dialog.isConnected || event.defaultPrevented || event.isComposing
            || !dialog.contains(event.target)) return;
        // An independently mounted nested dialog owns its own keyboard cycle.
        if (event.target.closest('[aria-modal="true"]') !== dialog) return;
        if (event.key === 'Escape' && onEscape) {
            event.preventDefault();  // layers.back() relies on it: Back closes this dialog and keeps the layer under it
            event.stopPropagation();
            onEscape();
        } else if (event.key === 'Tab') {
            const items = focusable(dialog);
            const first = items[0];
            const last = items[items.length - 1];
            if (!first || (event.shiftKey && (doc.activeElement === first || doc.activeElement === dialog))
                || (!event.shiftKey && (doc.activeElement === last || doc.activeElement === dialog))) {
                event.preventDefault();
                (event.shiftKey ? last : first)?.focus();
                if (!first) dialog.focus();
            }
        }
    };
    dialog.addEventListener('keydown', onKey);
    const first = typeof initialFocus === 'function' ? initialFocus() : initialFocus;
    (first && dialog.contains(first) && available(first) ? first : focusable(dialog)[0] || dialog).focus();
    return ({ restoreFocus = true } = {}) => {
        if (disposed) return;
        disposed = true;
        dialog.removeEventListener('keydown', onKey);
        const ownedFocus = dialog.contains(doc.activeElement) || doc.activeElement === doc.body;
        if (originalTabIndex === null) dialog.removeAttribute('tabindex');
        if (restoreFocus && ownedFocus && returnFocus && available(returnFocus)) returnFocus.focus({ preventScroll });
    };
}

/**
 * A message field's Enter presses its own Send button (DESIGN "Controls and
 * editable choices"), so that button's validation, disabled and pending state
 * decide what is sent. Shift+Enter keeps the native line break; Ctrl/Cmd/Alt
 * with Enter send like Enter. An Enter that commits an IME composition never
 * sends: WebKit ends the composition before that keydown and marks it only
 * with keyCode 229. A held key sends once. The touch keyboard labels the key
 * Send. The listener lives on the field; dispose only if the field outlives it.
 */
export function bindEnterSubmit(field, button, submit = () => { if (!button.disabled) button.click(); }) {
    const hint = field.enterKeyHint;
    field.enterKeyHint = 'send';
    const onKey = (event) => {
        if (event.key !== 'Enter' || event.shiftKey || event.defaultPrevented || event.isComposing
            || event.keyCode === 229 || !field.isConnected || field.disabled || field.readOnly) return;
        event.preventDefault();
        if (!event.repeat && button.isConnected) submit();
    };
    field.addEventListener('keydown', onKey);
    return () => {
        field.removeEventListener('keydown', onKey);
        field.enterKeyHint = hint;
    };
}

/**
 * Geometry only for a mounted, fixed-position popup. The owner provides CSS
 * consuming --ui-popup-{top,left,max-height,max-width,anchor-width}, including
 * overflow and bounded sizing, and portals outside any clipping ancestor.
 * point uses client coordinates; otherwise align to the anchor (start/end).
 * Scroll follows the anchor unless its owner supplies onAnchorScroll to close.
 * No focus, keyboard, ARIA, DOM reparenting or action dispatch is installed.
 */
export function bindPopoverPosition(popup, { anchor, point, align = 'start', onAnchorScroll } = {}) {
    const doc = popup.ownerDocument;
    const win = doc.defaultView;
    let disposed = false;
    function reposition() {
        if (disposed || !popup.isConnected || (anchor && !anchor.isConnected)) return;
        const viewport = win.visualViewport;
        const margin = 8;
        const leftEdge = (viewport?.offsetLeft || 0) + margin;
        const topEdge = (viewport?.offsetTop || 0) + margin;
        const width = Math.max(0, (viewport?.width || win.innerWidth) - margin * 2);
        const height = Math.max(0, (viewport?.height || win.innerHeight) - margin * 2);
        const target = anchor?.getBoundingClientRect();
        popup.style.setProperty('--ui-popup-max-width', `${width}px`);
        popup.style.setProperty('--ui-popup-max-height', `${height}px`);
        popup.style.setProperty('--ui-popup-anchor-width', `${target?.width || 0}px`);
        // Its laid-out size: the entrance (ui.css) scales it for a moment, which is not its size.
        const size = () => ({ width: popup.offsetWidth, height: popup.offsetHeight });
        let rect = size();
        let below = true;
        if (target && !point) {
            const spaceBelow = Math.min(height, Math.max(0, topEdge + height - target.bottom - 4));
            const spaceAbove = Math.min(height, Math.max(0, target.top - 4 - topEdge));
            below = rect.height <= spaceBelow || spaceBelow >= spaceAbove;
            popup.style.setProperty('--ui-popup-max-height', `${below ? spaceBelow : spaceAbove}px`);
            rect = size();
        }
        const x = point?.x ?? (target ? (align === 'end' ? target.right - rect.width : target.left) : leftEdge);
        const y = point?.y ?? (target ? (below ? target.bottom + 4 : target.top - rect.height - 4) : topEdge);
        const left = Math.max(leftEdge, Math.min(x, leftEdge + width - rect.width));
        const top = Math.max(topEdge, Math.min(y, topEdge + height - rect.height));
        popup.style.setProperty('--ui-popup-left', `${Math.round(left)}px`);
        popup.style.setProperty('--ui-popup-top', `${Math.round(top)}px`);
        // It grows from where it was opened: the press point, else the middle of the anchor's side it opens on.
        const clamp = (value, most) => Math.round(Math.max(0, Math.min(most, value)));
        const ox = (point?.x ?? (target ? target.left + target.width / 2 : left)) - left;
        const oy = point ? point.y - top : (below ? 0 : rect.height);
        popup.style.setProperty('--ui-popup-origin', `${clamp(ox, rect.width)}px ${clamp(oy, rect.height)}px`);
    }
    const onScroll = (event) => {
        if (event.target?.nodeType && popup.contains(event.target)) return;
        if (onAnchorScroll) onAnchorScroll(event);
        else reposition();
    };
    win.addEventListener('scroll', onScroll, true);
    win.addEventListener('resize', reposition);
    win.visualViewport?.addEventListener('resize', reposition);
    win.visualViewport?.addEventListener('scroll', reposition);
    reposition();
    return {
        reposition,
        destroy() {
            disposed = true;
            win.removeEventListener('scroll', onScroll, true);
            win.removeEventListener('resize', reposition);
            win.visualViewport?.removeEventListener('resize', reposition);
            win.visualViewport?.removeEventListener('scroll', reposition);
        },
    };
}

/**
 * Bind an already mounted menu using the popup geometry above. Adapted from
 * Project row actions; close() notifies once, destroy() only cleans up.
 * `align` is the anchor edge the menu lines up with (default end).
 * `initialFocus: false` leaves focus where it is: a menu a pointer opened from
 * beside a field must not take the field's focus (a phone keeps its keyboard).
 */
export function bindMenu(menu, { anchor, point, onClose, align = 'end', initialFocus = true } = {}) {
    const doc = menu.ownerDocument;
    let disposed = false;
    const items = () => Array.from(menu.querySelectorAll('[role^="menuitem"]')).filter(available);
    const cleanup = () => {
        disposed = true;
        doc.removeEventListener('pointerdown', onOutside, true);
        menu.removeEventListener('keydown', onKey);
        doc.defaultView.removeEventListener('blur', onBlur);
        position.destroy();
    };
    function close({ restoreFocus = false, reason = 'action', focusTarget = anchor } = {}) {
        if (disposed) return;
        // It leaves the way it came, toward where it grew from (ui.css), while the menu itself goes at once.
        const win = doc.defaultView;
        leaveAsPicture(menu, POPUP_LEAVES, { duration: motionMs('--motion-fade', win), win });
        cleanup();
        // Restore before the callback: an action may open a dialog of its own.
        if (restoreFocus && (menu.contains(doc.activeElement) || doc.activeElement === doc.body)
            && focusTarget && available(focusTarget)) focusTarget.focus();
        onClose?.({ reason });
    }
    const onOutside = (event) => {
        if (!menu.contains(event.target) && !anchor?.contains(event.target)) close({ reason: 'outside' });
    };
    const onBlur = () => close({ reason: 'blur' });
    const onKey = (event) => {
        if (event.defaultPrevented || event.altKey || event.ctrlKey || event.metaKey) return;
        const options = items();
        const index = options.indexOf(doc.activeElement);
        let target;
        if (event.key === 'Escape') {
            event.preventDefault();
            event.stopPropagation();
            close({ restoreFocus: true, reason: 'escape' });
        } else if (event.key === 'Tab') {
            // Removing a portalled menu during native Tab loses the traversal
            // origin in WebViews. Resolve the next document control first.
            const outside = focusable(doc.body).filter((element) => !menu.contains(element));
            const origin = outside.indexOf(anchor);
            const next = origin < 0 ? anchor : outside[origin + (event.shiftKey ? -1 : 1)] || anchor;
            event.preventDefault();
            close({ restoreFocus: true, focusTarget: next, reason: 'tab' });
        } else if (event.key === 'ArrowDown') target = options[(index + 1) % options.length];
        else if (event.key === 'ArrowUp') target = options[(index < 0 ? options.length : index) - 1] || options.at(-1);
        else if (event.key === 'Home') target = options[0];
        else if (event.key === 'End') target = options.at(-1);
        if (target) {
            event.preventDefault();
            target.focus();
        }
    };
    doc.addEventListener('pointerdown', onOutside, true);
    menu.addEventListener('keydown', onKey);
    doc.defaultView.addEventListener('blur', onBlur);
    // The anchor's middle: a row that lifts under its long-press menu (style.css, a scale about its middle) keeps it.
    const middle = (box) => box && { x: box.left + (box.width || 0) / 2, y: box.top + (box.height || 0) / 2 };
    const openedAt = middle(anchor?.getBoundingClientRect());
    const position = bindPopoverPosition(menu, {
        anchor, point, align, onAnchorScroll: () => {
            const now = middle(anchor?.getBoundingClientRect());
            // A queued scroll event can arrive after opening without moving the trigger, and the feed's own reflow
            // under a lifted row scrolls without moving it. Only an actual anchor move invalidates this menu.
            if (!openedAt || !now || Math.abs(now.x - openedAt.x) > 0.5 || Math.abs(now.y - openedAt.y) > 0.5) {
                close({ reason: 'scroll' });
            }
        },
    });
    if (initialFocus) items()[0]?.focus();
    return { close, reposition: position.reposition, destroy: cleanup };
}
