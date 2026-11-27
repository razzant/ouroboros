// The row menu of one chat feed (docs/DESIGN.md "Replies"): what a bubble, a work card, or an addressed timeline line
// offers, reached without hover — a right-click, Shift+F10 or the ContextMenu key on the focused row, a long press (no movement)
// on touch. Its items are the row's own doors, never second ones: Reply, Copy, Open project (when the row has
// one) and, on a work card, Resume / Pause / Stop… exactly when the card itself offers them (Pause is the item of
// the card's Stop… menu, offered by the same rule). A work card's Reply lives only here (owner 17.10, Q9 B): the
// card's action row holds commands over the work; a bubble then lists the registered message actions its control
// row shows (message_actions.js), in words. A timeline line with no address of its own is no row to quote: pressed,
// it opens its nearest card's menu without that card's Reply (the card's title is not the line), and its swipe
// does nothing. Esc and an outside press close it; keyboard focus enters the
// menu and returns to the row. A right-click on a link, a field or selected words keeps the browser's own menu,
// so selected text stays copyable. On touch the row answers the press (owner 19.10, 4A): it sinks a little while the
// finger holds, and when the menu opens it lifts over the dimmed feed, with a haptic (haptics.js), until the menu
// closes (style.css: an addressed line inside a card takes a colour step instead, since the card clips a lift);
// and a row that offers Reply can be swiped left to reply (chat_row_swipe.js), with the same Reply.
import { REPLY_ICON } from './chat_reply.js';
import { createRowSwipe } from './chat_row_swipe.js';
import { createGestureTail } from './gesture_tail.js';
import { haptic } from './haptics.js';
import { tr } from './i18n.js';
import { isStackLayout, onLayoutChange } from './layout.js';
import { bindMenu } from './ui_interactions.js';
import { ACTION_PAUSE, ACTION_RESUME, pauseTaskAction } from './task_control_menu.js';
import { copyTextWithToast } from './ui_helpers.js';
import { fullText } from './text_fit.js';
import { getMessageActions, messageRow, presentAction, runMessageAction } from './message_actions.js';
import { cardControlGuard } from './work_status.js';

const LONG_PRESS_MS = 450;
const HOLD_MS = 150;  // a still finger this long starts the sink, so a tap or a scroll never shows it
const RELEASE_MS = 260;  // the lift's way back (style.css), after which the row is at rest
const MOVE_TOLERANCE_PX = 10;
const ROW = '.chat-bubble, .chat-live-card[data-task-id], .chat-live-line, [data-work-reply-row]';
const BARE_LINE = '.chat-live-line:not([data-work-reply-row])';  // a timeline line the host gave no row address
const NATIVE = 'a[href], input, textarea, select, [contenteditable="true"], video, audio';

const shown = (node) => Boolean(node?.isConnected && !node.hidden && !node.disabled);

/**
 * `feed` is the chat's transcript, `replies` its reply controller (chat_reply.js), `cardOf(taskId)` its root card
 * record, `readControls(record)` its current offer (the same facts as its Stop… menu and Resume button),
 * `afterAction()` the re-read once an action settled, `actions` the message-action registry (the page's own when
 * omitted). Returns `{ open, destroy }`.
 */
export function createChatRowMenu({ feed, replies, cardOf, readControls, afterAction = () => {},
    actions = getMessageActions() }) {
    const doc = feed.ownerDocument;
    let current = null;  // { menu, binding, row }
    let press = null;  // a touch that may become (or became) a long press
    const tail = createGestureTail(doc, { mousedown: true });  // the lift must not take focus or activate an item
    let rightPress = null;  // what was selected when a right button went down, before its own default selected a word

    const selectedIn = (row) => {
        const selection = doc.getSelection?.();
        return Boolean(selection && !selection.isCollapsed && row.contains(selection.anchorNode));
    };

    const rowOf = (node) => {
        const row = node?.closest?.(ROW);
        return row && feed.contains(row) ? row : null;
    };

    const recordOf = (row) => row.matches('.chat-live-card') ? cardOf(row.dataset.taskId || '') : null;
    // An unregistered card does not borrow another row's Reply.
    const replyRef = (row, record = recordOf(row)) => row.matches('.chat-live-card')
        ? record && replies.cardRef(record) : replies.bubbleRef(row);
    // What a press on `row` is about: the row itself, or for a line with no address (tool evidence, a lifecycle note, the
    // files line) the card it stands in, the nearest one, so a child's line is never its root's. That card offers its
    // own doors, never its Reply: nothing quotes that line.
    const subjectOf = (row) => {
        const card = row?.matches?.(BARE_LINE) ? row.closest('.chat-live-card[data-task-id]') : null;
        return card && feed.contains(card) ? card : row;
    };

    function bubbleItems(row, point, touch) {
        const items = [];
        // A revision under the open menu (a spoken line's words, chat_text_bubble.js) replaces the row's Copy, so the
        // item presses the one this row holds at the press.
        const copy = () => row.querySelector(':scope > .chat-message-copy, :scope > .chat-message-actions > .chat-message-copy');
        if (copy()) items.push(['copy', tr('code.copy', 'Copy'), () => copy()?.click(), true]);
        if (touch && replies.canSelect?.(row, point)) items.push(['select-text', tr('chat.reply.select_text', 'Select text'),
            () => { tail.disarm(); replies.selectText(row, point); }]);
        const view = row.matches('.chat-bubble') ? messageRow(row) : null;
        for (const action of view ? actions.actionsFor(view) : []) {
            // Registered actions read the row as it is at the press, like the row's own buttons do: a revision under
            // the open menu (chat_text_bubble.js) must not leave them the words or address it replaced.
            items.push([action.key, presentAction(action, view).label,
                () => { const now = messageRow(row); if (now) runMessageAction(action, now, { anchor: row }); }, true]);
        }
        return items;
    }

    function cardItems(record) {
        const items = [];
        const words = [record.titleEl?.textContent, fullText(record.activityEl)].map((text) => String(text || '').trim());
        if (words.some(Boolean)) items.push(['copy', tr('code.copy', 'Copy'),
            () => copyTextWithToast(words.filter(Boolean).join('\n')), true]);
        const { resumeRunBtn: resume, cancelRunBtn: stop } = record;
        const targetNow = cardControlGuard(record, readControls);
        if (shown(resume)) items.push(['resume', tr('task.control.resume', 'Resume'), () => resume.click(), true,
            () => shown(resume) && targetNow(ACTION_RESUME)]);
        if (shown(stop)) {
            const pause = targetNow(ACTION_PAUSE);
            if (pause) {
                items.push(['pause', tr('task.control.pause', 'Pause'),
                    () => pauseTaskAction(pause).finally(afterAction), true, () => shown(stop) && targetNow(ACTION_PAUSE)]);
            }
            items.push(['stop', stop.textContent, () => stop.click(), false, () => shown(stop)]);
        }
        return items;
    }

    function itemsFor(target, point, touch) {
        const row = subjectOf(target);
        const record = recordOf(row);
        const items = record ? cardItems(record) : bubbleItems(row, point, touch);
        const ref = row === target ? replyRef(row, record) : null;
        // A bubble's words and address can be revised under the open menu: like its own Reply, the item reads them at
        // the press.
        if (ref) items.unshift(['reply', tr('chat.reply.reply', 'Reply'),
            () => replies.start(row.matches('.chat-bubble') ? replyRef(row) : ref)]);
        // The row's own Project reference (a portal's footer, the reference under an answer or a receipt), where Reply
        // and Copy end: a card pressed through a line has no Reply, so it follows Copy.
        const reference = [...row.querySelectorAll('[data-intent="open-project"]')]
            .find((node) => shown(node) && rowOf(node) === row && !node.closest('.chat-live-card.subagent'));
        if (reference) items.splice(Math.min(items.length, row === target ? 2 : 1), 0,
            ['project', tr('chat.row.open_project', 'Open project'), () => reference.click()]);
        return items;
    }

    function close(options) {
        current?.binding.close(options);
    }

    // The row's answer to a touch press: `hold` while the finger rests, `lift` while its menu is open, `release` on
    // the way back.
    const pressOf = (row) => row?.dataset?.rowPress || '';
    const releases = new Map();  // row → the timer that puts it back at rest
    function setPress(row, phase) {
        if (!row?.dataset) return;
        clearTimeout(releases.get(row));
        releases.delete(row);
        if (phase) { row.dataset.rowPress = phase; return; }
        if (pressOf(row) !== 'lift') { delete row.dataset.rowPress; return; }
        row.dataset.rowPress = 'release';
        releases.set(row, setTimeout(() => {
            releases.delete(row);
            if (pressOf(row) === 'release') delete row.dataset.rowPress;
        }, RELEASE_MS));
    }

    /** Open the menu of the pressed `target`, at a pointer `point` or under the focused `anchor`. False when it offers
     * nothing. `lift`: a long press opened it, so what the menu is about lifts while it is open (the card, for a line
     * with no address of its own). */
    function open(target, { point, anchor = target, lift = false } = {}) {
        close();
        const row = subjectOf(target);
        const items = itemsFor(target, point, lift || doc.documentElement?.dataset.pointer === 'coarse');
        if (!items.length) return false;
        // Focus returns to the row: a bubble or card that is not itself focusable takes focus like a quote's target.
        if (anchor === target && target.tabIndex < 0 && !target.hasAttribute('tabindex')) target.tabIndex = -1;
        const menu = doc.createElement('div');
        menu.className = 'ui-popup chat-row-menu';
        menu.setAttribute('role', 'menu');
        menu.setAttribute('aria-label', row.matches('.chat-live-card') ? tr('chat.row.work_actions', 'Work actions')
            : tr('chat.row.message_actions', 'Message actions'));
        for (const [kind, label, run, keepFocus, canRun] of items) {
            const item = doc.createElement('button');
            item.type = 'button';
            item.setAttribute('role', 'menuitem');
            item.dataset.rowAction = kind;
            item.textContent = label;
            item.addEventListener('click', (event) => {
                event.stopPropagation();
                // An item that moves focus on its own (Reply to the composer, Stop… into its menu) is not given it back.
                const available = !canRun || canRun();
                close({ restoreFocus: !available || Boolean(keepFocus), reason: 'action' });
                if (available) run();
                else if (!anchor.isConnected && row.isConnected) {
                    if (!row.hasAttribute('tabindex')) row.tabIndex = -1;
                    row.focus({ preventScroll: true });
                }
            });
            menu.append(item);
        }
        doc.body.append(menu);
        // A new finger on an item is a deliberate press, even just after the long press lifted.
        menu.addEventListener('pointerdown', () => tail.disarm());
        if (lift) { setPress(row, 'lift'); haptic('press'); }
        const binding = bindMenu(menu, { anchor, point, onClose: () => {
            menu.remove();
            if (lift) setPress(row, '');
            if (current?.menu === menu) current = null;
        } });
        current = { menu, binding, row: target };
        return true;
    }

    function onContextMenu(event) {
        const row = rowOf(event.target);
        if (!row) return;
        // A long press already answered (Android also fires contextmenu for it).
        if (press?.fired || tail.armed || current?.row === row) { event.preventDefault(); return; }
        // Words the owner selected keep the browser's menu (Copy them). A right button on macOS selects the word under
        // it before this event; that selection is the gesture's own, not the owner's, and goes with the row menu.
        const owned = rightPress?.row === row ? rightPress.selected : selectedIn(row);
        rightPress = null;
        if (event.target.closest(NATIVE) || owned || (row.dataset.selectText && event.target.closest('.message, [data-reply-caption]'))) return;
        // Android may fire it before the long press's time: it answers that touch and lifts the row as the press
        // would, so the touch's timer opens nothing more and the lifting finger's click is swallowed (onPointerEnd).
        const touch = press && !press.fired && press.row === row ? press : null;
        if (!open(row, { point: { x: event.clientX, y: event.clientY }, lift: Boolean(touch) })) return;
        event.preventDefault();
        if (touch) { clearTimeout(touch.timer); clearTimeout(touch.holdTimer); touch.fired = true; }
        if (selectedIn(row)) doc.getSelection().removeAllRanges();
    }

    function onKeydown(event) {
        if (event.key !== 'ContextMenu' && !(event.key === 'F10' && event.shiftKey)) return;
        const focused = doc.activeElement;
        const row = rowOf(focused);
        if (!row || focused.closest(NATIVE)) return;
        if (open(row, { anchor: focused })) event.preventDefault();
    }

    function cancelPress() {
        if (!press) return;
        clearTimeout(press.timer);
        clearTimeout(press.holdTimer);
        if (pressOf(press.subject) === 'hold') setPress(press.subject, '');
        press = null;
        for (const [type, handler] of pointerListeners) doc.removeEventListener(type, handler, true);
    }

    function onPointerDown(event) {
        cancelPress();
        const row = rowOf(event.target);
        rightPress = (event.button === 2 || (event.ctrlKey && event.button === 0)) && row ? { row, selected: selectedIn(row) } : null;
        if (event.pointerType !== 'touch' || !event.isPrimary) return;
        if (!row || event.target.closest(NATIVE) || selectedIn(row) || (row.dataset.selectText && event.target.closest('.message, [data-reply-caption]'))) return;
        const { clientX: x, clientY: y } = event;
        const touch = { id: event.pointerId, x, y, row, subject: subjectOf(row), fired: false };
        touch.holdTimer = setTimeout(() => {
            if (press === touch && itemsFor(row).length) setPress(touch.subject, 'hold');
        }, HOLD_MS);
        touch.timer = setTimeout(() => {
            touch.fired = open(row, { point: { x, y }, lift: true });
            if (!touch.fired && pressOf(touch.subject) === 'hold') setPress(touch.subject, '');
        }, LONG_PRESS_MS);
        press = touch;
        for (const [type, handler] of pointerListeners) doc.addEventListener(type, handler, true);
    }

    function onPointerMove(event) {
        if (press && !press.fired && event.pointerId === press.id
            && Math.hypot(event.clientX - press.x, event.clientY - press.y) > MOVE_TOLERANCE_PX) cancelPress();
    }

    // Capture in the document while the touch is tracked: a portalled menu or a detached row can put the
    // release outside the feed. Only the finger that opened the menu can arm its release guard.
    function onPointerEnd(event) {
        if (!press || event.pointerId !== press.id) return;
        if (press.fired) tail.arm();
        cancelPress();
    }

    function onScroll() {
        // A waiting hold no longer points at the same place. Reflow under a lifted row still owes its release guard.
        if (!press?.fired) cancelPress();
    }

    const pointerListeners = [['pointermove', onPointerMove], ['pointerup', onPointerEnd], ['pointercancel', onPointerEnd]];
    const listeners = [['contextmenu', onContextMenu], ['keydown', onKeydown], ['pointerdown', onPointerDown],
        ['scroll', onScroll]];
    for (const [type, handler] of listeners) feed.addEventListener(type, handler);
    const win = doc.defaultView || globalThis.window;
    const disposeSwipe = createRowSwipe({ feed, rowOf, icon: REPLY_ICON, replyRef, reply: (ref) => replies.start(ref), win, doc,
        isStack: () => isStackLayout(), isCoarse: () => Boolean(win?.matchMedia?.('(pointer: coarse)')?.matches),
        onLayout: (sync) => onLayoutChange(() => sync(), { w: win }) });

    return {
        open,
        destroy() {
            cancelPress();
            tail.disarm();
            close();
            for (const [row, timer] of releases) { clearTimeout(timer); delete row.dataset.rowPress; }
            releases.clear();
            disposeSwipe();
            for (const [type, handler] of listeners) feed.removeEventListener(type, handler);
        },
    };
}
