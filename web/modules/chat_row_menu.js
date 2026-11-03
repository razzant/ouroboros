// The row menu of one chat feed (docs/DESIGN.md "Replies"): what a bubble or a root work card offers, reached
// without hover — a right-click, Shift+F10 or the ContextMenu key on the focused row, a long press (no movement)
// on touch. Its items are the row's own doors, never second ones: Reply, Copy, Open project (when the row has
// one) and, on a work card, Resume / Pause / Stop… exactly when the card itself offers them (Pause is the item of
// the card's Stop… menu, offered by the same rule). A work card's Reply lives only here (owner 17.10, Q9 B): the
// card's action row holds commands over the work; a bubble then lists the registered message actions its control
// row shows (message_actions.js), in words. Esc and an outside press close it; keyboard focus enters the
// menu and returns to the row. A right-click on a link, a field or selected words keeps the browser's own menu,
// so selected text stays copyable. On touch the row answers the press (owner 19.10, 4A): it sinks a little while the
// finger holds, and when the menu opens it lifts over the dimmed feed, with a haptic (haptics.js), until the menu
// closes; and a row that offers Reply can be swiped left to reply (chat_row_swipe.js), with the same Reply.
import { REPLY_ICON } from './chat_reply.js';
import { createRowSwipe } from './chat_row_swipe.js';
import { haptic } from './haptics.js';
import { tr } from './i18n.js';
import { isStackLayout, onLayoutChange } from './layout.js';
import { bindMenu } from './ui_interactions.js';
import { ACTION_PAUSE, pauseTaskAction } from './task_control_menu.js';
import { copyTextWithToast } from './ui_helpers.js';
import { fullText } from './text_fit.js';
import { getMessageActions, messageRow, presentAction, runMessageAction } from './message_actions.js';

const LONG_PRESS_MS = 450;
const HOLD_MS = 150;  // a still finger this long starts the sink, so a tap or a scroll never shows it
const RELEASE_MS = 260;  // the lift's way back (style.css), after which the row is at rest
const MOVE_TOLERANCE_PX = 10;
const ROW = '.chat-bubble, .chat-live-card:not(.subagent)';
const NATIVE = 'a[href], input, textarea, select, [contenteditable="true"], video, audio';

const shown = (node) => Boolean(node?.isConnected && !node.hidden && !node.disabled);

/**
 * `feed` is the chat's transcript, `replies` its reply controller (chat_reply.js), `cardOf(taskId)` its root card
 * record, `actionsOf(record)` the items that card's own Stop… menu offers now (its work's controls),
 * `afterAction()` the re-read once an action settled, `actions` the message-action registry (the page's own when
 * omitted). Returns `{ open, destroy }`.
 */
export function createChatRowMenu({ feed, replies, cardOf, actionsOf, afterAction = () => {},
    actions = getMessageActions() }) {
    const doc = feed.ownerDocument;
    let current = null;  // { menu, binding, row }
    let press = null;  // a touch that may become (or became) a long press
    let swallowClickUntil = 0;
    let rightPress = null;  // what was selected when a right button went down, before its own default selected a word

    const selectedIn = (row) => {
        const selection = doc.getSelection?.();
        return Boolean(selection && !selection.isCollapsed && row.contains(selection.anchorNode));
    };

    const rowOf = (node) => {
        const row = node?.closest?.(ROW);
        return row && feed.contains(row) ? row : null;
    };

    // What the row's Reply quotes, a work card's included (the menu's first item, and the swipe's).
    const replyRef = (row) => (row.matches('.chat-live-card') ? replies.cardRef(cardOf(row.dataset.taskId || '') || null)
        : replies.bubbleRef(row));

    function bubbleItems(row) {
        const items = [];
        const ref = replies.bubbleRef(row);
        if (ref) items.push(['reply', tr('chat.reply.reply', 'Reply'), () => replies.start(ref)]);
        const copy = row.querySelector(':scope > .chat-message-copy, :scope > .chat-message-actions > .chat-message-copy');
        if (copy) items.push(['copy', tr('code.copy', 'Copy'), () => copy.click(), true]);
        const view = messageRow(row);
        for (const action of view ? actions.actionsFor(view) : []) {
            items.push([action.key, presentAction(action, view).label, () => runMessageAction(action, view, { anchor: row }), true]);
        }
        return items;
    }

    function cardItems(row, record) {
        const items = [];
        // A work card draws no Reply of its own (its action row holds only commands): the menu is its Reply.
        if (replies.cardRef(record)) items.push(['reply', tr('chat.reply.reply', 'Reply'), () => replies.start(replies.cardRef(record))]);
        const words = [record.titleEl?.textContent, fullText(record.activityEl)].map((text) => String(text || '').trim());
        if (words.some(Boolean)) items.push(['copy', tr('code.copy', 'Copy'),
            () => copyTextWithToast(words.filter(Boolean).join('\n')), true]);
        if (shown(record.resumeRunBtn)) items.push(['resume', tr('task.control.resume', 'Resume'), () => record.resumeRunBtn.click(), true]);
        if (shown(record.cancelRunBtn)) {
            // The card's own Stop… offers Pause on the same facts; its menu re-checks before any effect.
            if (actionsOf(record).includes(ACTION_PAUSE)) {
                items.push(['pause', tr('task.control.pause', 'Pause'),
                    () => pauseTaskAction(record.groupId).finally(afterAction), true]);
            }
            items.push(['stop', record.cancelRunBtn.textContent, () => record.cancelRunBtn.click()]);
        }
        return items;
    }

    function itemsFor(row) {
        const record = row.matches('.chat-live-card') ? cardOf(row.dataset.taskId || '') : null;
        const items = record ? cardItems(row, record) : bubbleItems(row);
        // The row's own Project reference (a portal's footer, the reference under an answer or a receipt).
        const reference = [...row.querySelectorAll('[data-intent="open-project"]')]
            .find((node) => shown(node) && rowOf(node) === row && !node.closest('.chat-live-card.subagent'));
        if (reference) items.splice(Math.min(items.length, 2), 0,
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

    /** Open the menu of `row`, at a pointer `point` or under the focused `anchor`. False when the row offers nothing.
     * `lift`: a long press opened it, so the row lifts while it is open. */
    function open(row, { point, anchor = row, lift = false } = {}) {
        close();
        const items = itemsFor(row);
        if (!items.length) return false;
        // Focus returns to the row: a bubble or card that is not itself focusable takes focus like a quote's target.
        if (anchor === row && row.tabIndex < 0 && !row.hasAttribute('tabindex')) row.tabIndex = -1;
        const menu = doc.createElement('div');
        menu.className = 'ui-popup chat-row-menu';
        menu.setAttribute('role', 'menu');
        menu.setAttribute('aria-label', row.matches('.chat-live-card') ? tr('chat.row.work_actions', 'Work actions')
            : tr('chat.row.message_actions', 'Message actions'));
        for (const [kind, label, run, keepFocus] of items) {
            const item = doc.createElement('button');
            item.type = 'button';
            item.setAttribute('role', 'menuitem');
            item.dataset.rowAction = kind;
            item.textContent = label;
            item.addEventListener('click', (event) => {
                event.stopPropagation();
                // An item that moves focus on its own (Reply to the composer, Stop… into its menu) is not given it back.
                close({ restoreFocus: Boolean(keepFocus), reason: 'action' });
                run();
            });
            menu.append(item);
        }
        doc.body.append(menu);
        if (lift) { setPress(row, 'lift'); haptic('press'); }
        const binding = bindMenu(menu, { anchor, point, onClose: () => {
            menu.remove();
            if (lift) setPress(row, '');
            if (current?.menu === menu) current = null;
        } });
        current = { menu, binding, row };
        return true;
    }

    function onContextMenu(event) {
        const row = rowOf(event.target);
        if (!row) return;
        // A long press already answered (Android also fires contextmenu for it).
        if (press?.fired || Date.now() < swallowClickUntil || current?.row === row) { event.preventDefault(); return; }
        // Words the owner selected keep the browser's menu (Copy them). A right button on macOS selects the word under
        // it before this event; that selection is the gesture's own, not the owner's, and goes with the row menu.
        const owned = rightPress?.row === row ? rightPress.selected : selectedIn(row);
        rightPress = null;
        if (event.target.closest(NATIVE) || owned) return;
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
        if (press) {
            clearTimeout(press.timer);
            clearTimeout(press.holdTimer);
            if (pressOf(press.row) === 'hold') setPress(press.row, '');
        }
        press = null;
    }

    function onPointerDown(event) {
        cancelPress();
        const row = rowOf(event.target);
        rightPress = event.button === 2 && row ? { row, selected: selectedIn(row) } : null;
        if (event.pointerType !== 'touch' || !event.isPrimary) return;
        if (!row || event.target.closest(NATIVE)) return;
        const { clientX: x, clientY: y } = event;
        const touch = { id: event.pointerId, x, y, row, fired: false };
        touch.holdTimer = setTimeout(() => { if (press === touch && itemsFor(row).length) setPress(row, 'hold'); }, HOLD_MS);
        touch.timer = setTimeout(() => {
            touch.fired = open(row, { point: { x, y }, lift: true });
            if (!touch.fired && pressOf(row) === 'hold') setPress(row, '');
        }, LONG_PRESS_MS);
        press = touch;
    }

    function onPointerMove(event) {
        if (press && !press.fired && event.pointerId === press.id
            && Math.hypot(event.clientX - press.x, event.clientY - press.y) > MOVE_TOLERANCE_PX) cancelPress();
    }

    // A touch is captured by its first target, so its end arrives here even when the menu lies under the finger:
    // the tap that ends a long press is not also a press of whatever is there (a menu item included).
    function onPointerEnd(event) {
        if (press?.fired && event.pointerId === press.id) swallowClickUntil = Date.now() + 400;
        cancelPress();
    }

    function onClickCapture(event) {
        if (Date.now() >= swallowClickUntil) return;
        swallowClickUntil = 0;
        event.preventDefault();
        event.stopPropagation();
    }

    const listeners = [['contextmenu', onContextMenu], ['keydown', onKeydown], ['pointerdown', onPointerDown],
        ['pointermove', onPointerMove], ['pointerup', onPointerEnd], ['pointercancel', onPointerEnd], ['scroll', cancelPress]];
    for (const [type, handler] of listeners) feed.addEventListener(type, handler);
    doc.addEventListener('click', onClickCapture, true);
    const win = doc.defaultView || globalThis.window;
    const disposeSwipe = createRowSwipe({ feed, icon: REPLY_ICON, replyRef, reply: (ref) => replies.start(ref), win, doc,
        isStack: () => isStackLayout(), isCoarse: () => Boolean(win?.matchMedia?.('(pointer: coarse)')?.matches),
        onLayout: (sync) => onLayoutChange(() => sync(), { w: win }) });

    return {
        open,
        destroy() {
            cancelPress();
            close();
            for (const [row, timer] of releases) { clearTimeout(timer); delete row.dataset.rowPress; }
            releases.clear();
            disposeSwipe();
            for (const [type, handler] of listeners) feed.removeEventListener(type, handler);
            doc.removeEventListener('click', onClickCapture, true);
        },
    };
}
