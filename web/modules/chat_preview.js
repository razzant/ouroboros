/** This browser's bounded copy of each room's newest text, painted before the room's first canonical read
 * (docs/DESIGN.md "History edges"). It is never history: it sets no coverage, read receipt, unread credit, action or
 * `historyLoaded`, and a row's `data-preview-history-id` only locates a passage for the reading anchor
 * (chat_render_batch.js). One same-origin record keeps the rooms read most recently (MAX_ROOMS rooms, MAX_ROWS rows,
 * MAX_CHARS serialized; whole rows give way, a person's text is never cut). chat.js writes only what it admitted: its
 * newest window replaces a room's rows, an arrivals read that accounted for everything since merges into them; a
 * failed, unavailable, older, around, stale or late read writes nothing; bounded newest windows are valid. Each row
 * goes where the feed puts its bubble (chat_render_batch.js insertTimelineNode): by the time it is stamped with (a
 * missing one is the moment it is drawn, a delayed delivery keeps its earlier one), equal times by stream position
 * (chat_history_replay.js compareHistoryPosition), a row already there keeping its place; time and position are kept
 * for that alone, and the bound keeps the rows that placement puts last. Live echoes, controls, attachments and
 * session metadata never enter it; another origin or refused storage opens cold.
 *
 * The first canonical replay retires the painted copy. A passage the owner holds selected stays, labelled, until the
 * selection is released: the live selection, or the copy room_retire.js keeps on the room (`kept`) while an inert
 * layer covers it or it is away, where WebKit drops the live range. A history read refused for authentication, current
 * and not aborted (a room's own in chat.js fetchHistory, app.js's link preparation, a Place's in place_history.js),
 * removes the record and every copy painted in this document at once, selection or not (`previewsRefused`);
 * api_client.js clears the record on sign-in, the core's sign-in page on every sign-out, and the registry's word that
 * a Project is going away takes its rooms' rows (`forgetProjectPreviews`, app.js). Its one import is the feed's tie
 * rule, so the two orders cannot fork. */
import { compareHistoryPosition } from './chat_history_replay.js';

export const PREVIEW_KEY = 'ouroboros.chat-preview.v1';
const MAX_CHARS = 200000, MAX_ROOMS = 8, MAX_ROWS = 80;
const ROLES = ['user', 'assistant', 'system'];
const CARD_ROWS = ['task_evidence', 'task_summary', 'quiz', 'card'];
const refusal = error => error?.status === 401 || error?.status === 403;
// insertTimelineNode's rule over the strings a bubble's dataset holds: a row goes before the first timed row later than
// it, or as late with a greater stream position (compareHistoryPosition, as the feed compares two bubbles), else last;
// an untimed row goes last.
const position = row => (row.pos ? { source: row.pos[0], offset: row.pos[1] } : null);
const place = (rows, row) => {
    const index = Number.isFinite(row.at) ? rows.findIndex(other => Number.isFinite(other.at) && (other.at > row.at
        || (other.at === row.at && compareHistoryPosition(position(other), position(row)) > 0))) : -1;
    rows.splice(index < 0 ? rows.length : index, 0, row);
    return rows;
};
// Every copy painted in this document: a refused or ended session takes each one at once.
const painted = new Set();

export function clearChatPreviews(storage) {
    try { (storage || globalThis.localStorage)?.removeItem(PREVIEW_KEY); } catch { /* storage unavailable */ }
    for (const release of [...painted]) release();
}

/** The registry says these Projects are going away (app.js, where their rooms are let go): their rooms' rows leave the
 * record in every scope; every other room keeps its own as it was, and nothing to remove writes nothing. */
export function forgetProjectPreviews(projectIds, storage) {
    const going = new Set((projectIds || []).filter(Boolean).map(String));
    if (!going.size) return;
    try {
        const store = storage || globalThis.localStorage;
        const raw = store?.getItem(PREVIEW_KEY);
        const data = raw ? JSON.parse(raw) : null;
        if (data?.v !== 2 || !Array.isArray(data.rooms)) return;
        const project = (entry) => { try { return String(JSON.parse(entry?.key)?.[1]?.[1] ?? ''); } catch { return ''; } };
        const rooms = data.rooms.filter(entry => !going.has(project(entry)));
        if (rooms.length < data.rooms.length) store.setItem(PREVIEW_KEY, JSON.stringify({ v: 2, rooms }));
    } catch { /* storage unavailable or unreadable: nothing here to take */ }
}

/** A history read made in this document, current and refused for authentication: the record and every painted copy go,
 * selection or not. True when it was one; any other failure changes nothing here. */
export function previewsRefused(error, storage) {
    if (!refusal(error)) return false;
    clearChatPreviews(storage);
    return true;
}

const savedRow = row => Boolean(row) && ROLES.includes(row.role) && typeof row.id === 'string'
    && typeof row.text === 'string' && typeof row.speaker === 'string';

/** `label(role, isProgress, systemType, opts)` is chat_activity.js `senderLabel`: the speaker frozen as the canonical
 * bubble names it, never the row's source or session. `mutate` is the room's stable-viewport write. */
export function createChatPreview({ feed, room, label, mutate, kept = () => null,
    scope = feed.ownerDocument.querySelector?.('meta[name="ouroboros-appearance-scope"]')?.content || '',
    storage = () => globalThis.localStorage }) {
    const doc = feed.ownerDocument;
    const key = JSON.stringify([scope, room]);
    let root = null, retiring = false, disposed = false;
    const rooms = () => {
        try {
            const raw = storage()?.getItem(PREVIEW_KEY);
            const data = raw && raw.length <= MAX_CHARS ? JSON.parse(raw) : null;
            return data?.v === 2 && Array.isArray(data.rooms) ? data.rooms : [];
        } catch { return []; }
    };
    // This room's rows; null when it has no record (never read here, or it gave way to rooms read since).
    const saved = () => {
        const rows = rooms().find(entry => entry?.key === key)?.rows;
        return Array.isArray(rows) ? rows.filter(savedRow) : null;
    };
    // This room first (an empty one too: it was read empty), newest rows kept; then the other rooms, most recent
    // first, while they fit. Every row and room is serialized once and charged a separator: the record stays under
    // MAX_CHARS.
    const write = (rows) => {
        const head = `{"key":${JSON.stringify(key)},"rows":[`;
        const pieces = rows.slice(-MAX_ROWS).map(row => JSON.stringify(row));
        let budget = MAX_CHARS - '{"v":2,"rooms":[]}'.length - head.length - ']},'.length;
        let from = pieces.length;
        while (from > 0 && pieces[from - 1].length + 1 <= budget) budget -= pieces[--from].length + 1;
        const parts = [`${head}${pieces.slice(from).join(',')}]}`];
        for (const entry of rooms().filter(item => item?.key !== key).slice(0, MAX_ROOMS - 1)) {
            const text = JSON.stringify(entry);
            if (text.length + 1 > budget) break;  // it and every older room give way
            budget -= text.length + 1;
            parts.push(text);
        }
        try { storage()?.setItem(PREVIEW_KEY, `{"v":2,"rooms":[${parts.join(',')}]}`); } catch { /* the room still works */ }
    };
    // Each row keeps the time and position its bubble is stamped with (chat.js addMessage, stampNodeTimestamp,
    // stampHistoryNode): a missing time is the moment it is drawn, one that does not parse leaves it untimed.
    const fromHistory = (messages, now = new Date().toISOString()) => (Array.isArray(messages) ? messages : [])
        .filter(row => row?.history_id && typeof row.text === 'string' && row.text && ROLES.includes(row.role)
            && !row.is_progress && !CARD_ROWS.includes(row.system_type)).map(row => {
            const raw = row.ts || now, position = row.history_position;
            const at = typeof raw === 'number' ? raw : Date.parse(String(raw));
            const source = position ? String(position.source) : '';
            return { id: String(row.history_id), role: row.role, text: row.text, speaker: label(row.role, false,
                row.system_type, { source: row.source, senderLabel: row.sender_label, initiator: row.initiator }),
            ...(Number.isFinite(at) ? { at } : {}), ...(source ? { pos: [source, String(position.offset)] } : {}) };
        });
    const note = (state) => { if (root) root.firstElementChild.textContent = `Saved text preview · ${state}`; };
    const drop = () => { root?.remove(); root = null; retiring = false; };
    const covers = ([start, startOffset, end, endOffset]) => {
        try {
            const range = doc.createRange();
            range.setStart(start, startOffset);
            range.setEnd(end, endOffset);
            return range.intersectsNode(root);
        } catch { return false; }  // a detached end holds nothing here
    };
    const held = () => {
        const selection = doc.getSelection?.();
        const live = selection && !selection.isCollapsed ? Array.from({ length: selection.rangeCount }, (_, index) => {
            const range = selection.getRangeAt(index);
            return [range.startContainer, range.startOffset, range.endContainer, range.endOffset];
        }) : [];
        return [...live, ...(kept() || [])].some(covers);
    };
    const retire = () => {
        if (!root) return;
        retiring = true;
        if (held()) note('retained until selection is released');
        else mutate(() => { drop(); return true; });
    };
    const onSelection = () => { if (retiring) retire(); };
    doc.addEventListener('selectionchange', onSelection);
    const release = () => { if (root) mutate(() => { drop(); return true; }); drop(); };
    painted.add(release);
    const discard = () => {
        try { storage()?.removeItem(PREVIEW_KEY); } catch { /* storage unavailable */ }
        for (const each of [...painted]) each();
    };
    return {
        /** Paints the saved rows once, before any canonical read; false when there is nothing valid to show. */
        show() {
            const rows = disposed || root ? null : saved();
            if (!rows?.length) return false;
            root = doc.createElement('div');
            root.className = 'chat-saved-preview';
            root.dataset.savedPreview = 'true';
            root.append(doc.createElement('div'));
            root.firstElementChild.className = 'chat-load-older-note';
            note('updating');
            for (const row of rows) {
                const bubble = doc.createElement('div');
                bubble.className = `chat-bubble ${row.role}`;
                bubble.dataset.previewHistoryId = row.id;
                for (const [className, text] of [['sender', row.speaker], ['message', row.text]]) {
                    const part = doc.createElement('div');
                    part.className = className;
                    part.textContent = text;
                    bubble.append(part);
                }
                root.append(bubble);
            }
            feed.prepend(root);
            return true;
        },
        /** The room's newest window, admitted (chat.js): its rows replace this room's. A window that could not read
         * the conversation stream, or a refusal, keeps the record; a bounded `complete: false` window is a window. */
        save(messages, { window: frame = null, reason_code: refused = '' } = {}) {
            if (disposed || refused || frame?.truncated_by?.includes?.('chat_source_unavailable')) return;
            write(fromHistory(messages).reduce(place, []));
        },
        /** Rows an arrivals read accounted for since the newest window: one the room has stays where it stands, a new
         * one goes where the feed puts it. A room without a record waits for its next newest window. */
        append(messages) {
            const incoming = disposed ? [] : fromHistory(messages), rows = incoming.length ? saved() : null;
            if (!rows) return;
            for (const row of incoming) {
                const at = rows.findIndex(other => other.id === row.id);
                if (at < 0) place(rows, row);
                else rows[at] = row;
            }
            write(rows);
        },
        retire,
        /** Any read the room made was refused for authentication: discard (every painted copy). A page that merely
         * failed changes nothing. */
        refused(error) { if (refusal(error)) discard(); },
        /** The room's newest read failed: a refusal discards, anything else leaves the text labelled. */
        failed(error) {
            if (refusal(error)) discard();
            else note('not current');
        },
        destroy() {
            disposed = true;
            painted.delete(release);
            doc.removeEventListener('selectionchange', onSelection);
            drop();
        },
    };
}
