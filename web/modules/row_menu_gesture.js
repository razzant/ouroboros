// What opens a row's menu without hover (docs/DESIGN.md "Replies", "Shell"): a right-click, Shift+F10 or the
// ContextMenu key on the focused row, and a long press on touch (held still for LONG_PRESS_MS). One binding per root
// (a chat feed, the Projects list); the menus and their items stay their callers'.
//
// A right-click on a link, a field or words the owner selected keeps the browser's own menu, so selected text stays
// copyable. A long press is no press once the finger moves past MOVE_TOLERANCE_PX or the root, or a scroller around
// it, scrolls. The finger's end and its moves are heard on the document, so a menu that mounts under the finger, or a
// layer that takes the finger's capture (the drawer), cannot leave a press armed; the tap that ends a long press is not
// also a press of what lies under it (nor, where a host sends them, its mouse events, whose mousedown would take the
// focus out of the menu just opened), and the contextmenu Android fires for the same hold is not a second opening.

export const LONG_PRESS_MS = 450;
export const MOVE_TOLERANCE_PX = 10;
const NATIVE = 'a[href], input, textarea, select, [contenteditable="true"], video, audio';
const CLICK_GUARD_MS = 400;

/**
 * `rowOf(node)` names the row a node belongs to (null outside one); `open(row, { point } | { anchor })` opens that
 * row's menu at a pointer or under the focused element and answers synchronously whether it did; `isOpen(row)` says
 * whether the row's menu is open already. Returns the disposer.
 */
export function bindRowMenuGesture({ root, rowOf, open, isOpen = () => false }) {
    const doc = root.ownerDocument;
    let press = null;  // a touch that may become (or became) a long press
    let swallowClickUntil = 0;
    let rightPress = null;  // what was selected when a right button went down, before its own default selected a word

    const selectedIn = (row) => {
        const selection = doc.getSelection?.();
        return Boolean(selection && !selection.isCollapsed && row.contains(selection.anchorNode));
    };

    function onContextMenu(event) {
        const row = rowOf(event.target);
        if (!row) return;
        // A long press already answered (Android also fires contextmenu for it).
        if (press?.fired || Date.now() < swallowClickUntil || isOpen(row)) { event.preventDefault(); return; }
        // Words the owner selected keep the browser's menu (Copy them). A right button on macOS selects the word under
        // it before this event; that selection is the gesture's own, not the owner's, and goes with the row menu.
        const owned = rightPress?.row === row ? rightPress.selected : selectedIn(row);
        rightPress = null;
        if (event.target.closest(NATIVE) || owned) return;
        if (!open(row, { point: { x: event.clientX, y: event.clientY } })) return;
        event.preventDefault();
        if (selectedIn(row)) doc.getSelection().removeAllRanges();
        // Android may fire it for a hold before LONG_PRESS_MS: that opening is the long press, not a second one.
        if (press && !press.fired) { clearTimeout(press.timer); press.fired = true; }
    }

    function onKeydown(event) {
        if (event.key !== 'ContextMenu' && !(event.key === 'F10' && event.shiftKey)) return;
        const focused = doc.activeElement;
        const row = rowOf(focused);
        if (!row || focused.closest(NATIVE)) return;
        if (open(row, { anchor: focused })) event.preventDefault();
    }

    function cancelPress() {
        if (press) clearTimeout(press.timer);
        press = null;
    }

    function onPointerDown(event) {
        cancelPress();
        const row = rowOf(event.target);
        rightPress = event.button === 2 && row ? { row, selected: selectedIn(row) } : null;
        if (event.pointerType !== 'touch' || !event.isPrimary) return;
        if (!row || event.target.closest(NATIVE)) return;
        const { clientX: x, clientY: y } = event;
        const touch = { id: event.pointerId, x, y, fired: false };
        touch.timer = setTimeout(() => { touch.fired = open(row, { point: { x, y } }); }, LONG_PRESS_MS);
        press = touch;
    }

    function onPointerMove(event) {
        if (press && !press.fired && event.pointerId === press.id
            && Math.hypot(event.clientX - press.x, event.clientY - press.y) > MOVE_TOLERANCE_PX) cancelPress();
    }

    function onPointerEnd(event) {
        if (press?.fired && event.pointerId === press.id) swallowClickUntil = Date.now() + CLICK_GUARD_MS;
        if (event.pointerId === press?.id) cancelPress();
    }

    // Scroll events do not bubble: the document's capture hears every scroller, and only the root's own or one around
    // it (a short window scrolls the whole sidebar) moves the row under a waiting finger.
    function onScroll(event) {
        if (press && !press.fired && (event.target === doc || event.target?.contains?.(root))) cancelPress();
    }

    function onLiftCapture(event) {
        if (Date.now() >= swallowClickUntil) return;
        if (event.type === 'click') swallowClickUntil = 0;
        event.preventDefault();
        event.stopPropagation();
    }

    const onRoot = [['contextmenu', onContextMenu], ['keydown', onKeydown], ['pointerdown', onPointerDown]];
    const onDocument = [['pointermove', onPointerMove], ['pointerup', onPointerEnd], ['pointercancel', onPointerEnd],
        ['scroll', onScroll], ['mousedown', onLiftCapture], ['click', onLiftCapture]];
    for (const [type, handler] of onRoot) root.addEventListener(type, handler);
    for (const [type, handler] of onDocument) doc.addEventListener(type, handler, true);
    return () => {
        cancelPress();
        for (const [type, handler] of onRoot) root.removeEventListener(type, handler);
        for (const [type, handler] of onDocument) doc.removeEventListener(type, handler, true);
    };
}
