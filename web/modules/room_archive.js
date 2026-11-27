// The read-only archive of saved rooms while Main's history refuses for the Project registry (#3294; the server's
// rule: ouroboros/room_archive.py). Main's refused read carries `project_registry: "unavailable"`: its header note
// says the project list cannot be read and the in-feed edge offers `Read saved messages` beside Retry.
// The history controls pass the effective refusal to `archiveDoor` and the newest restoration fact separately.
// The reader (`openRoomArchive`) is a modal reading surface on
// the document reader's shell (DESIGN "Document reading"): Main and each Project room the saved records name, one at
// a time, as plain words under their speakers, oldest first, `Load more history` above them while older remain.
// It sends, answers, continues, moves, links and acknowledges nothing: no composer, Reply, question, card, Continue
// or copy-link control and no read credit; a room read here is not reopened for work, and its rooms never join the
// page's Project room set. A page with nothing to show still continues; completeness is never claimed.
import { apiClient } from './api_client.js';
import { fmt, tr } from './i18n.js';
import { bindScrollFade } from './scroll_fade.js';
import { bindDialogFocus } from './ui_interactions.js';
import { formatMsgTime } from './utils.js';

/** Whether a refused read is Main's refusal for the Project registry: the archive is what it offers. */
export const archiveOffered = (error) => error?.body?.project_registry === 'unavailable';

const SPEAKERS = { user: 'You', assistant: 'Ouroboros', system: 'System' };
// A handoff stores the destination name; other pinned notices already carry their full saved words.
const PINNED_WORDS = { project_handoff: (name) => fmt('Moved to {name}', { name }) };
const RESTORED = 'Project list restored from its saved copy.';
const LIMIT = 'Only messages and notices with a confirmed room appear. A deleted project may still appear. '
    + 'Questions, task cards, file contents and some replies are not available here.';
const RECOVER = {
    archive_cursor_changed: 'The saved view changed. Reload this room to keep reading.',
    bad_cursor: 'This page can’t be continued. Reload the room to keep reading.',
    unknown_room: 'This room is no longer available in saved messages. Choose another room.',
    migrating: 'Saved history is still being prepared. Try again later.',
    unavailable: 'Saved messages couldn’t be loaded.',
};
let opened = 0;

function element(doc, tag, className, text = '') {
    const node = doc.createElement(tag);
    if (className) node.className = className;
    if (text) node.textContent = text;
    return node;
}

function rowNode(doc, message) {
    const item = element(doc, 'li', `room-archive-row is-${message.role}`);
    const time = formatMsgTime(message.ts)?.full || '';  // the chat's own absolute time words: an archive row may be old
    const speaker = element(doc, 'div', 'room-archive-speaker');
    speaker.append(element(doc, 'span', '', SPEAKERS[message.role] || 'System'));
    if (time) {
        const stamp = element(doc, 'time', '', ` · ${time}`);
        stamp.setAttribute('data-i18n-authored', '');  // formatMsgTime already localized this value
        speaker.append(stamp);
    }
    const words = element(doc, 'div', 'room-archive-text', PINNED_WORDS[message.kind]?.(message.text) || message.text);
    words.setAttribute('data-i18n-authored', '');  // the room's own words, never interface text
    item.append(speaker, words);
    if (Array.isArray(message.attachments) && message.attachments.length) {
        const files = element(doc, 'div', 'room-archive-files', fmt('Files: {names}', { names: message.attachments.join(', ') }));
        files.setAttribute('data-i18n-authored', '');
        item.append(files);
    }
    item.dataset.archiveId = String(message.id || '');
    return item;
}

/**
 * `onDismiss` is for a reader's dismissal, not navigation or teardown. A true value means the archive
 * read found the project list readable again, not that the live history was reloaded.
 */
export function openRoomArchive({ doc = globalThis.document, read = apiClient.chatArchive,
    returnFocus = doc.activeElement, pointer = false, onDismiss = () => {} } = {}) {
    const dialog = element(doc, 'dialog', 'document-reader room-archive');
    const titleId = `room-archive-title-${++opened}`;
    dialog.setAttribute('aria-modal', 'true');
    dialog.setAttribute('aria-labelledby', titleId);
    const panel = element(doc, 'div', 'document-reader-panel');
    const head = element(doc, 'div', 'document-reader-head');
    const identity = element(doc, 'div', 'document-reader-identity');
    const title = element(doc, 'h2', 'document-reader-title', 'Saved messages');
    title.id = titleId;
    const meta = element(doc, 'div', 'document-reader-meta');
    const currentRoom = element(doc, 'span', '', 'Main');
    const separator = element(doc, 'span', '', ' · ');
    separator.setAttribute('data-i18n-authored', '');
    meta.append(element(doc, 'span', '', 'Read-only'), separator, currentRoom);
    identity.append(title, meta);
    const closeButton = element(doc, 'button', 'btn btn-default document-reader-close', 'Close');
    closeButton.type = 'button';
    closeButton.dataset.archiveClose = '';
    head.append(identity, closeButton);
    // Opened by a pointer, the reading region shows its focus ring from the first key, not before.
    const body = element(doc, 'div', 'document-reader-body room-archive-body scroll-fade-y');
    body.classList.toggle('is-quiet-focus', pointer);
    body.setAttribute('tabindex', '-1');
    const column = element(doc, 'div', 'room-archive-column');  // one reading measure: every block shares its edges
    const rooms = element(doc, 'div', 'room-archive-rooms');
    rooms.setAttribute('role', 'group');
    rooms.setAttribute('aria-label', 'Saved rooms');
    const roomsNote = element(doc, 'p', 'room-archive-note', 'Some project rooms may be missing.');
    const edge = element(doc, 'div', 'room-archive-edge');
    const older = element(doc, 'button', 'chat-load-older-btn ui-hit-area', 'Load more history');
    older.type = 'button';
    older.dataset.archiveOlder = '';
    const edgeNote = element(doc, 'p', 'room-archive-note');
    edge.append(older, edgeNote);
    const feed = element(doc, 'ol', 'room-archive-feed');
    feed.setAttribute('aria-label', 'Saved messages');
    const status = element(doc, 'div', 'room-archive-status');
    status.setAttribute('role', 'status');
    const statusText = element(doc, 'p', '');
    const retry = element(doc, 'button', 'btn btn-default btn-sm', 'Try again');
    retry.type = 'button';
    retry.dataset.archiveRetry = '';
    status.append(statusText, retry);
    const limit = element(doc, 'details', 'ui-disclosure room-archive-limit');
    limit.append(element(doc, 'summary', '', 'What’s included'), element(doc, 'p', 'room-archive-note', LIMIT));
    column.append(rooms, roomsNote, edge, status, feed, limit);
    body.append(column);
    panel.append(head, body);
    dialog.append(panel);

    const state = { room: 'main', cursor: null, hasMore: false, shown: 0, loading: false, failed: null,
        restored: false, older: false, controller: null };
    const roomButtons = new Map();
    let closed = false;

    function reconcileRooms(known) {
        const focused = doc.activeElement;
        const retained = new Set(known.map(entry => String(entry.room)));
        for (const [key, button] of roomButtons) {
            if (retained.has(key)) continue;
            button.remove();
            roomButtons.delete(key);
        }
        for (const entry of known) {
            const key = String(entry.room);
            let button = roomButtons.get(key);
            if (!button) {
                button = element(doc, 'button', 'btn btn-default btn-sm room-archive-room');
                button.type = 'button';
                button.dataset.archiveRoom = key;
                if (key !== 'main') button.setAttribute('data-i18n-authored', '');
                roomButtons.set(key, button);
                rooms.append(button);
            }
            const name = entry.name || key;
            if (button.textContent !== name) button.textContent = name;
        }
        if (focused?.dataset?.archiveRoom && !retained.has(focused.dataset.archiveRoom)) {
            (roomButtons.get(state.room) || body).focus({ preventScroll: true });
        }
    }

    function paint() {
        for (const [key, button] of roomButtons) button.setAttribute('aria-pressed', String(key === state.room));
        const selected = roomButtons.get(state.room);
        if (selected) {
            if (state.room === 'main') currentRoom.removeAttribute('data-i18n-authored');
            else currentRoom.setAttribute('data-i18n-authored', '');
            if (currentRoom.textContent !== selected.textContent) currentRoom.textContent = selected.textContent;
        }
        const failed = state.failed, reload = failed === 'archive_cursor_changed' || failed === 'bad_cursor';
        older.hidden = !state.hasMore || state.restored || Boolean(failed);
        older.disabled = state.loading;
        older.textContent = state.loading && state.older ? 'Loading…' : 'Load more history';
        edgeNote.textContent = state.restored || failed || state.loading ? ''
            : state.shown === 0 && state.hasMore ? 'Nothing in this stretch could be placed here.'
                : state.shown === 0 ? 'No saved messages could be placed here.'
                    : state.hasMore ? '' : 'No older messages can be shown here.';
        edgeNote.hidden = !edgeNote.textContent;
        edge.hidden = older.hidden && edgeNote.hidden;
        const words = state.restored ? 'The project list can be read again. Close this to return to Main.'
            : RECOVER[failed] || (state.loading && !state.older ? 'Loading saved messages…' : '');
        statusText.textContent = words;
        retry.textContent = reload ? 'Reload room' : 'Try again';
        retry.hidden = !failed || failed === 'unknown_room';
        // The page action disappears during loading or failure. Keep keyboard focus inside the reader.
        if ((older.hidden && doc.activeElement === older) || (retry.hidden && doc.activeElement === retry)) body.focus();
        status.hidden = !words;
        status.setAttribute('role', failed ? 'alert' : 'status');
        body.setAttribute('aria-busy', String(state.loading));
    }

    async function load({ room = state.room, more = false } = {}) {
        state.controller?.abort();
        const controller = new AbortController();
        state.controller = controller;
        if (!more) {
            state.room = room;
            state.cursor = null;
            state.hasMore = false;
            state.shown = 0;
            feed.replaceChildren();
        }
        Object.assign(state, { loading: true, failed: null, older: more });
        paint();
        try {
            const page = await read({ room, before: more ? state.cursor : null, signal: controller.signal });
            if (closed || state.controller !== controller) return;
            state.loading = false;
            if (page?.state === 'not_needed') {
                state.restored = true;
                state.hasMore = false;
                paint();
                return;
            }
            if (page?.rooms?.length) reconcileRooms(page.rooms);
            const messages = Array.isArray(page?.messages) ? page.messages : [];
            const height = body.scrollHeight || 0;
            if (more) feed.prepend(...messages.map((message) => rowNode(doc, message)));
            else feed.append(...messages.map((message) => rowNode(doc, message)));
            state.shown += messages.length;
            state.hasMore = page?.has_more === true && Boolean(page?.next_cursor);
            state.cursor = state.hasMore ? page.next_cursor : null;
            paint();
            // Keep both a focused picker and an older passage visible. Otherwise a first page opens at its newest row.
            if (more) body.scrollTop = (body.scrollTop || 0) + (body.scrollHeight || 0) - height;
            else if (!rooms.contains(doc.activeElement)) body.scrollTop = body.scrollHeight || 0;
        } catch (error) {
            if (closed || state.controller !== controller || error?.name === 'AbortError') return;
            state.loading = false;
            const code = error?.body?.reason_code;
            state.failed = code === 'archive_cursor_changed' || code === 'migrating' ? code
                : error?.status === 404 ? 'unknown_room' : error?.status === 400 ? 'bad_cursor' : 'unavailable';
            paint();
        }
    }

    let disposeFocus = null;
    const disposeFade = bindScrollFade(body);
    function close({ restoreFocus = true } = {}) {
        if (closed) return;
        closed = true;
        state.controller?.abort();
        // Native close may restore the old opener by itself. Navigation removes the dialog instead.
        if (restoreFocus && dialog.open && typeof dialog.close === 'function') dialog.close();
        for (const [type, handler] of Object.entries(listeners)) dialog.removeEventListener(type, handler);
        disposeFade();
        disposeFocus?.({ restoreFocus });
        dialog.remove();
        if (restoreFocus) onDismiss(state.restored);
    }

    function onClick(event) {
        const target = event.target;
        if (target === dialog || target?.closest?.('[data-archive-close]')) { close(); return; }
        if (state.restored) return;
        const room = target?.closest?.('[data-archive-room]');
        if (room) { if (room.dataset.archiveRoom !== state.room) void load({ room: room.dataset.archiveRoom }); return; }
        if (target?.closest?.('[data-archive-older]') && !state.loading) { void load({ more: true }); return; }
        if (!retry.hidden && !state.loading && target?.closest?.('[data-archive-retry]')) {
            const more = state.older && !['archive_cursor_changed', 'bad_cursor'].includes(state.failed);
            void load({ more });
        }
    }
    const listeners = { click: onClick, close: () => close(), cancel: event => { event.preventDefault(); close(); },
        keydown: () => body.classList.remove('is-quiet-focus') };
    for (const [type, handler] of Object.entries(listeners)) dialog.addEventListener(type, handler);
    doc.body.appendChild(dialog);
    if (typeof dialog.showModal === 'function') dialog.showModal(); else dialog.setAttribute('open', '');
    disposeFocus = bindDialogFocus(dialog, { initialFocus: body, returnFocus, preventScroll: true, onEscape: close });
    reconcileRooms([{ room: 'main', name: 'Main' }]);
    void load();
    return { close, get closed() { return closed; }, dialog };
}

/** The history edge owns when this door is offered; navigation owns its lifetime. */
export function archiveDoor({ open = openRoomArchive, notes = null, returnFocus = null,
    readReceipt, syncHistory, isVisible } = {}) {
    let door = null, hint = null, timer = null, reader = null, disposed = false, restored = false;
    function clearHint() {
        clearTimeout(timer);
        hint?.remove();
        hint = null;
    }
    function close() {
        reader?.close({ restoreFocus: false });
        reader = null;
    }
    return {
        get active() { return Boolean(reader && !reader.closed); },
        refusalNote(error) {
            return archiveOffered(error) ? 'The project list can’t be read, so Main’s saved history isn’t shown.' : '';
        },
        offer(edge, error) {
            const offered = archiveOffered(error);
            if (disposed || (!door && !offered)) return;
            if (!door) {
                // The transcript, unlike the composer, can take focus without raising the phone's keyboard.
                if (returnFocus && !returnFocus.hasAttribute('tabindex')) returnFocus.setAttribute('tabindex', '-1');
                door = element(edge.ownerDocument, 'button', 'chat-load-older-btn ui-hit-area chat-archive-door');
                door.type = 'button';
                edge.after(door);
                door.addEventListener('click', event => {
                    if (disposed || door.hidden || (reader && !reader.closed)) return;
                    // Native close restores the focus saved by showModal. A pointer click need not focus a WebKit button.
                    door.focus({ preventScroll: true });
                    reader = open({ doc: door.ownerDocument, pointer: event.detail > 0,
                        returnFocus: () => door.hidden ? returnFocus : door,
                        onDismiss: async readable => {
                            reader = null;
                            if (disposed || !isVisible?.()) return;
                            if (readable && !door.hidden) await syncHistory({ reason: 'navigate', decision: 'archive_return' });
                            if (!disposed && !reader && isVisible()) readReceipt.note({ discrete: true });
                        } });
                    readReceipt?.cancel();
                });
            }
            if (!offered && door.ownerDocument.activeElement === door) returnFocus?.focus({ preventScroll: true });
            if (offered) {
                const label = tr('history.read_saved', 'Read saved messages');  // the transcript excludes the overlay
                if (door.textContent !== label) door.textContent = label;
            }
            door.hidden = !offered;
        },
        recent(error, page) {
            if (disposed) return;
            if (error) { restored = false; clearHint(); return; }
            if (restored || page?.project_registry !== 'restored' || !notes) return;
            restored = true;
            hint = element(notes.ownerDocument, 'p', 'chat-registry-restored', RESTORED);
            hint.setAttribute('role', 'status');
            notes.append(hint);
            timer = setTimeout(clearHint, 8000);
        },
        close,
        dispose() { disposed = true; close(); clearHint(); door?.remove(); },
    };
}
