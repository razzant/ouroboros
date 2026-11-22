/**
 * The Model sheet (docs/DESIGN.md "Composer owner controls", owner 12.11 «A»): one floating sheet that a press on a
 * Model capsule (composer_model.js) or on a work's model fact (card_model.js) opens, never hover. From the top: what
 * the caller puts first (a work's own acts), the search field, the list of models (model_list.js: Main, Pinned,
 * Recent, All sources), and at its foot the processing tier strip (processing_choice.js) with what its choice applies
 * to; redundant Standard controls are hidden. It floats above its anchor without moving the composer or transcript; in the
 * stack layout a bottom sheet over a scrim, whose field waits for its own press (no keyboard raised by opening).
 *
 * One sheet in the window: opening another closes this one. A press outside, Escape (after the field's own), focus
 * leaving for the page, or the caller's `close()` close it; a pick closes it through the caller, a Pin and a tier
 * do not. Focus returns to `returnFocus` when the keyboard closed it. The bottom sheet is modal over its scrim
 * (DESIGN §10 "Layers and dismissal"): a press on the scrim closes it and only it, at the press's own click, so the
 * tap that dismisses reaches nothing beneath; Back gives it its Escape first (layers.js `back()`). Being modal, it
 * holds the focus (bindDialogFocus): opened by a finger the sheet itself takes it, so no keyboard rises, and Tab and
 * Shift+Tab wrap inside it, never reaching what lies under the scrim.
 */
import { routeKey } from './chat_model_routes.js';
import { createModelList, modelSearchField } from './model_list.js';
import { createTierStrip, pinDecides } from './processing_choice.js';
import { bindDialogFocus } from './ui_interactions.js';

let serial = 0;
let current = null;

/** Closes the open sheet, if any; with `within`, only one whose anchor lies there (a room leaving the screen takes its
 *  own sheet with it, never another room's). */
export function closeModelSheet({ within = null } = {}) {
    if (current && (!within || within.contains?.(current.anchor))) current.close({ focus: false });
}

/**
 * `anchor` is what it stands by (the composer row, a card's fact), `label` its accessible name, `head` an optional
 * node above the field, `list` the options of createModelList (minus its field and element), `tier` the strip's
 * `{ onChoose(tier), view(), mainRoute() }` (`view()` returns the strip's paint; `mainRoute()` names the route using
 * Main's pin, if any, so this sheet can ask for that exact binding).
 * `stack()` says whether the layout is the phone's, `coarse()` whether a finger opened it, `place(sheet, anchor)` binds
 * its position (bindPopoverPosition). `onClose()` follows its close. Returns `{ el, input, paint, reposition, close }`.
 */
export function openModelSheet({ anchor, label = 'Model', head = null, list, tier, stack = () => false,
    coarse = () => false, place, onClose = () => {}, returnFocus = null, doc = globalThis.document }) {
    current?.close({ focus: false });
    const id = `model-sheet-${++serial}`;
    const sheet = doc.createElement('div');
    sheet.className = 'ui-popup model-sheet';
    sheet.setAttribute('role', 'dialog');
    sheet.setAttribute('aria-label', label);
    const bottom = Boolean(stack());
    let scrim = null;
    if (bottom) {
        sheet.dataset.sheet = 'bottom';
        sheet.setAttribute('aria-modal', 'true');
        scrim = doc.createElement('div');
        scrim.className = 'model-sheet-scrim';
        scrim.setAttribute('aria-hidden', 'true');
        const grab = doc.createElement('span');
        grab.className = 'model-sheet-grab';
        grab.setAttribute('aria-hidden', 'true');
        sheet.append(grab);
    }
    if (head) sheet.append(head);
    const { label: field, input } = modelSearchField(doc, `${id}-list`);
    const grid = doc.createElement('div');
    grid.className = 'model-sheet-list';
    grid.id = `${id}-list`;
    const strip = createTierStrip({ doc, id: `${id}-tier`, onChoose: (choice) => tier.onChoose(choice),
        onFocusLost: () => (bottom && coarse() ? sheet : input).focus({ preventScroll: true }) });
    sheet.append(field, grid, strip.el);
    if (scrim) doc.body.append(scrim);
    doc.body.append(sheet);

    let closed = false;
    let position = null;
    let disposeFocus = null;
    // Even for the same route, a pin kept from a previous opening may have changed in Settings elsewhere. Ask once
    // each time a route comes under this sheet, before painting its support: when it opens and when Main moves, a
    // return to a route it left included (the pin may have gone with the model). A repaint, a failed or a mismatched
    // read is not a poll.
    let askedPin = '';  // the route now under the sheet, once its pin was asked for
    function readMainPin() {
        const route = tier.mainRoute?.();
        const { store } = list;
        const key = routeKey(route);
        if (key !== askedPin) askedPin = '';
        if (!route || askedPin || !pinDecides(store.index(), route)) return;
        askedPin = key;
        void store.readConnections?.({ fresh: true });
    }
    const rows = createModelList({ ...list, el: grid, input, id,
        onLeave: ({ focus }) => close({ focus }), onPaint: () => position?.reposition() });
    const paint = () => {
        if (closed) return;
        readMainPin();
        rows.paint();
        strip.paint(tier.view());
        position?.reposition();
    };

    // A press outside closes it (the anchor's own press is its caller's: it toggles); Escape from anywhere closes it
    // (inside, the field's Escape comes first); focus leaving for the page closes it. The scrim is the sheet's own:
    // its click closes it, so a finger lifting there never lands on what lies under it.
    const inside = (node) => sheet.contains(node) || Boolean(scrim?.contains(node)) || Boolean(anchor?.contains?.(node));
    const onOutside = (event) => { if (!inside(event.target)) close({ focus: false }); };
    const onScrim = (event) => { event.preventDefault?.(); close({ focus: false }); };
    const onPageKey = (event) => {
        if (event.key !== 'Escape' || event.defaultPrevented) return;
        event.preventDefault();
        event.stopPropagation?.();
        close({ focus: true });
    };
    const onFocusOut = (event) => {
        const to = event.relatedTarget;
        if (to && !inside(to)) close({ focus: false });
    };
    function close({ focus = false } = {}) {
        if (closed) return;
        closed = true;
        doc.removeEventListener('pointerdown', onOutside, true);
        doc.removeEventListener('keydown', onPageKey);
        sheet.removeEventListener('focusout', onFocusOut);
        scrim?.removeEventListener('click', onScrim);
        position?.destroy?.();
        disposeFocus?.({ restoreFocus: false });  // the focus goes back below, once, as it does for the floating sheet
        rows.destroy();
        strip.destroy();
        sheet.remove();
        scrim?.remove();
        if (current === handle) current = null;
        onClose();
        if (focus && returnFocus?.isConnected) returnFocus.focus({ preventScroll: true });
    }
    const handle = {
        el: sheet, input, paint, close, anchor,
        reposition: () => position?.reposition(),
        isOpen: () => !closed,
    };
    current = handle;
    paint();
    position = bottom ? null : place(sheet, anchor);
    doc.addEventListener('pointerdown', onOutside, true);
    doc.addEventListener('keydown', onPageKey);
    sheet.addEventListener('focusout', onFocusOut);
    scrim?.addEventListener('click', onScrim);
    // Under a finger opening raises no keyboard: the field takes focus on its own press. A keyboard or a mouse starts
    // in the field. The bottom sheet traps Tab; a finger's sheet starts on itself, so the focus is already inside.
    const finger = Boolean(coarse());
    if (bottom) disposeFocus = bindDialogFocus(sheet, { initialFocus: finger ? sheet : input, returnFocus: null });
    else if (!finger) input.focus({ preventScroll: true });
    return handle;
}
