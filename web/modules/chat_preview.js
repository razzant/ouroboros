/** This browser's bounded copy of each room's newest text, painted before the room's first canonical read
 * (docs/DESIGN.md "History edges"). It is never history: it sets no coverage, read receipt, unread credit, action or
 * `historyLoaded`, and a row's `data-preview-history-id` only locates a passage for the reading anchor
 * (chat_render_batch.js). Every room read in this origin keeps, under the data profile the serving document proves,
 * an entry of its own (MAX_ROWS rows within ROOM_CHARS; whole rows give way, a person's text is never cut), all of
 * them within TOTAL_CHARS: past it the entries saved least recently give way (each newest read saves its room; a show
 * writes nothing), and a room's show reads its own entry alone. chat.js writes only what it admitted: its newest
 * window replaces a room's rows, an arrivals read that accounted for everything since merges into them; a failed,
 * unavailable, older, around, stale or late read writes nothing; bounded newest windows are valid. Each row goes where
 * the feed puts its bubble (chat_render_batch.js insertTimelineNode): by the time it is stamped with (a missing one is
 * the moment it is drawn, a delayed delivery keeps its earlier one), equal times by stream position
 * (chat_history_replay.js compareHistoryPosition), a row already there keeping its place; time and position are kept
 * for that alone, and the bound keeps the rows that placement puts last. Live echoes, controls, attachments and
 * session metadata never enter it; another origin, a document proving no data profile, or refused storage opens cold.
 *
 * The first canonical replay retires the painted copy, and no later show paints one. A passage the owner holds
 * selected stays, labelled, until the selection is released: the live selection, or the copy room_retire.js keeps on
 * the room (`kept`) while an inert layer covers it or it is away, where WebKit drops the live range. A history read
 * refused for authentication, current and not aborted (the room's fetchHistory below, app.js's link preparation, a
 * Place's in place_history.js), removes every entry and every copy painted in this document at once, selection or not
 * (`previewsRefused`); api_client.js clears them on sign-in, the core's sign-in page on every sign-out (server_auth.py,
 * by PREVIEW_FAMILY). The registry's word that a Project is gone or going takes its rooms' entries (app.js: a
 * `deleting` row, an exact lookup of it: `forgetProjectPreviews`), as its complete list takes those of the Projects
 * it no longer names (`forgetUnlistedProjectPreviews`); a Project deleted where this origin hears none of these keeps
 * its entry until one comes or the total gives it way. Its imports are the feed's tie rule and the words a bubble
 * draws (a host notice's line), so neither the order nor the words fork. */
import { compareHistoryPosition } from './chat_history_replay.js';
import { hostRowDisplayText } from './host_notice_text.js';

// The record names the form, under the one key every eraser of every earlier version removes: entries are read only
// beside it, so such an eraser (it takes this key alone) still takes their words at this version's next touch.
export const PREVIEW_KEY = 'ouroboros.chat-preview.v1';
// Every key this preview writes: the record, and each room's entry (previewEntryKey). An eraser takes them all.
export const PREVIEW_FAMILY = 'ouroboros.chat-preview.';
export const PREVIEW_SCOPE_META = 'ouroboros-chat-preview-scope';
const RECORD = '{"v":3}';  // the entries' form; {"v":2,"rooms":[…]} was every room in this one key
// v2's one record held 200,000 characters in all. A room keeps as much: its newest 80 rows within that many characters
// of entry text, so no room keeps less than it did. Every room: 1,000,000 with their keys, about 2 MB at WebKit's 2
// bytes a character, under half an origin's 5 MB localStorage, the rest left to the shell's own keys.
const V2_CHARS = 200000, MAX_ROWS = 80, ROOM_CHARS = V2_CHARS, TOTAL_CHARS = 1000000;
/** The key of one room's entry under a proven profile; `room` is chat.js's [chatId, projectId]. */
export const previewEntryKey = (scope, room) => `${PREVIEW_KEY}:${JSON.stringify([scope, room])}`;
// server_web.chat_preview_scope: a 16-hex digest of the profile identity the current state and its completed
// witness both name. Only that fact makes a saved record this profile's; an installation scope, an older document
// and any other text are no proof, so their rooms read the canonical history and save nothing here.
const PROVEN_SCOPE = /^[0-9a-f]{16}$/;
/** A proven 16-hex profile scope, '' for anything else — shared with the WS layer (ws.js). */
export const provenScope = value => (PROVEN_SCOPE.test(String(value ?? '')) ? String(value) : '');
/** The data-profile scope this document proves via the core-stamped meta; '' when it proves none.
 *  Shared with the WS layer (ws.js): the same digest the serving connection admits frames against. */
export const documentScope = doc => provenScope(doc?.querySelector?.(`meta[name="${PREVIEW_SCOPE_META}"]`)?.content);

/** Preserve the room's previous keys unless this page proves its profile. Unscoped keys are not adopted. */
export function roomStoreKey(base, scope, chatId, isMain) {
    const scoped = scope ? `${base}:${scope}` : base;
    return isMain ? scoped : `${scoped}:${chatId}`;
}
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

/** A different serving profile retires this document's painted copies without erasing saved rooms. */
export function dropPaintedPreviews() {
    for (const release of [...painted]) release();
}

const savedRow = row => Boolean(row) && ROLES.includes(row.role) && typeof row.id === 'string'
    && typeof row.text === 'string' && typeof row.speaker === 'string';
// The family's keys in `store`, as its `length` and `key` list them (a store without them lists none).
const familyOf = store => Array.from({ length: store.length }, (_, index) => store.key(index))
    .filter(name => typeof name === 'string' && name.startsWith(PREVIEW_FAMILY));
// The record first: without it no entry is read again, even where listing the rest fails.
function erase(store) {
    store.removeItem(PREVIEW_KEY);
    for (const name of familyOf(store)) store.removeItem(name);
}
// `{"read":<ms>,"rows":[…]}`: when the copy was saved, and the newest whole rows that keep its text within ROOM_CHARS.
function entryText(rows, read) {
    const head = `{"read":${read},"rows":[`;
    const pieces = rows.slice(-MAX_ROWS).map(row => JSON.stringify(row));
    let budget = ROOM_CHARS - head.length - ']}'.length;
    let from = pieces.length;
    while (from > 0 && pieces[from - 1].length + 1 <= budget) budget -= pieces[--from].length + 1;
    return `${head}${pieces.slice(from).join(',')}]}`;
}
const readOf = text => Number(/^\{"read":(\d+),/.exec(text)?.[1]) || 0;
// The origin's storage is full (QuotaExceededError; NS_ERROR_DOM_QUOTA_REACHED in older Firefox).
const full = error => ['QuotaExceededError', 'NS_ERROR_DOM_QUOTA_REACHED'].includes(error?.name);
// The profile and the Project an entry's key names (previewEntryKey); '' for Main, the record or a key of no such form.
const roomOf = (name) => {
    let key = null;
    try { key = JSON.parse(name.slice(PREVIEW_KEY.length + 1)); } catch { /* the record, or not an entry */ }
    return { scope: Array.isArray(key) ? String(key[0] ?? '') : '',
        project: Array.isArray(key?.[1]) ? String(key[1][1] ?? '') : '' };
};
// Whether `store` holds this form's record. v2's one record moves into entries once, the order it kept by time read;
// entries standing without a record (an eraser took it) are erased before anything is read or written. `create` starts
// an empty record.
function current(store, create = false) {
    const raw = store.getItem(PREVIEW_KEY);
    if (raw === RECORD) return true;
    let rooms = [];
    try {
        const data = raw && raw.length <= V2_CHARS ? JSON.parse(raw) : null;
        if (data?.v === 2 && Array.isArray(data.rooms)) rooms = data.rooms.filter(room => typeof room?.key === 'string'
            && Array.isArray(room.rows));
    } catch { /* unreadable: nothing to move */ }
    erase(store);
    if (!rooms.length && !create) return false;
    store.setItem(PREVIEW_KEY, RECORD);
    const read = Date.now();
    rooms.forEach(({ key, rows }, index) => {
        store.setItem(`${PREVIEW_KEY}:${key}`, entryText(rows.filter(savedRow), read - index));
    });
    return true;
}

export function clearChatPreviews(storage) {
    try { erase(storage || globalThis.localStorage); } catch { /* storage unavailable */ }
    dropPaintedPreviews();
}

/** The registry says these Projects are gone or going (app.js, where their rooms are let go): their rooms' entries
 * leave, in every scope; every other room keeps its own as it was, and nothing to remove writes nothing. */
export function forgetProjectPreviews(projectIds, storage) {
    const going = new Set((projectIds || []).filter(Boolean).map(String));
    if (!going.size) return;
    try {
        const store = storage || globalThis.localStorage;
        if (!current(store)) return;
        for (const name of familyOf(store)) if (going.has(roomOf(name).project)) store.removeItem(name);
    } catch { /* storage unavailable or unreadable: nothing here to take */ }
}

/** The registry's complete list of Projects (`GET /api/projects?all=1`, app.js), asked at `asked` (Date.now()) under
 * the profile `scope` this page proves: every Project still there is in it. A Project room's entry of that profile
 * read before the list was asked, whose Project it names nowhere or names `deleting`, leaves; Main, another profile's
 * entries and entries read since stay, and nothing to remove writes nothing. */
export function forgetUnlistedProjectPreviews(projects, asked, scope, storage) {
    const profile = provenScope(scope);
    if (!profile || !Array.isArray(projects)) return;
    const listed = new Set(projects.filter(row => row?.id && row.lifecycle !== 'deleting').map(row => String(row.id)));
    try {
        const store = storage || globalThis.localStorage;
        if (!current(store)) return;
        for (const name of familyOf(store)) {
            const room = roomOf(name);
            if (room.scope === profile && room.project && !listed.has(room.project)
                && readOf(store.getItem(name) || '') < asked) store.removeItem(name);
        }
    } catch { /* storage unavailable or unreadable: nothing here to take */ }
}

/** A history read made in this document, current and refused for authentication: every entry and every painted copy
 * go, selection or not. True when it was one; any other failure changes nothing here. */
export function previewsRefused(error, storage) {
    if (!refusal(error)) return false;
    clearChatPreviews(storage);
    return true;
}

/** `label(role, isProgress, systemType, opts)` is chat_activity.js `senderLabel`: the speaker frozen as the canonical
 * bubble names it, never the row's source or session. `mutate` is the room's stable-viewport write. */
export function createChatPreview({ feed, room, label, mutate, kept = () => null,
    canShow = () => true, scope = documentScope(feed.ownerDocument),
    storage = () => globalThis.localStorage }) {
    const doc = feed.ownerDocument;
    const profile = provenScope(scope);
    const entry = previewEntryKey(profile, room);
    // `landed`: a canonical replay drew the room; `stale`: its newest read failed (a copy painted since says so).
    let root = null, retiring = false, disposed = false, landed = false, stale = false;
    // This room's entry text; '' when it has none (never read here, or it gave way to rooms read since).
    const stored = store => (store && current(store) ? store.getItem(entry) || '' : '');
    const saved = () => {
        try {
            const raw = stored(storage());
            const rows = raw && raw.length <= ROOM_CHARS ? JSON.parse(raw)?.rows : null;
            return Array.isArray(rows) ? rows.filter(savedRow) : null;
        } catch { return null; }
    };
    // This room's entry first (an empty one too: it was read empty), stamped with when it was saved; then the entries
    // saved least recently give way until all fit TOTAL_CHARS. An origin too full to take it makes that room first, then
    // asks once more. One that cannot be written goes, alone: an older text never stands for a newer read. Only a save
    // writes this entry: a show or a read with nothing new never rewrites words another tab may have replaced or erased.
    const write = (rows) => {
        let store = null, written = false;
        try {
            store = storage();
            if (!store || !current(store, true)) return;
            const text = entryText(rows, Date.now());
            // The others, saved least recently first, give way until all fit and `need` more characters are free.
            const giveWay = (need) => {
                const others = familyOf(store).filter(name => name !== PREVIEW_KEY && name !== entry).map((name) => {
                    const value = store.getItem(name) || '';
                    return { name, size: name.length + value.length, read: readOf(value) };
                }).sort((a, b) => a.read - b.read);
                let total = PREVIEW_KEY.length + RECORD.length + entry.length + text.length
                    + others.reduce((sum, other) => sum + other.size, 0);
                for (const other of others) {
                    if (total <= TOTAL_CHARS && need <= 0) break;
                    store.removeItem(other.name);
                    total -= other.size;
                    need -= other.size;
                }
            };
            try { store.setItem(entry, text); } catch (error) {
                if (!full(error)) throw error;
                giveWay(text.length);
                store.setItem(entry, text);
            }
            written = true;
            giveWay(0);
        } catch {
            if (!written) try { store?.removeItem(entry); } catch { /* storage unavailable */ }
        }
    };
    // Each row keeps the time and position its bubble is stamped with (chat.js addMessage, stampNodeTimestamp,
    // stampHistoryNode): a missing time is the moment it is drawn, one that does not parse leaves it untimed.
    const fromHistory = (messages, now = new Date().toISOString()) => (Array.isArray(messages) ? messages : [])
        .filter(row => row?.history_id && typeof row.text === 'string' && row.text && ROLES.includes(row.role)
            && !row.is_progress && !CARD_ROWS.includes(row.system_type)).map(row => {
            const raw = row.ts || now, position = row.history_position;
            const at = typeof raw === 'number' ? raw : Date.parse(String(raw));
            const source = position ? String(position.source) : '';
            return { id: String(row.history_id), role: row.role, text: hostRowDisplayText(row), speaker: label(row.role, false,
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
        landed = true;
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
        try { erase(storage()); } catch { /* storage unavailable */ }
        dropPaintedPreviews();
    };
    return {
        /** Paints the saved rows once, before any canonical read; false when there is nothing valid to show. */
        show() {
            const rows = disposed || landed || !profile || root || !canShow() ? null : saved();
            if (!rows?.length) return false;
            root = doc.createElement('div');
            root.className = 'chat-saved-preview';
            root.dataset.savedPreview = 'true';
            root.append(doc.createElement('div'));
            root.firstElementChild.className = 'chat-load-older-note';
            note(stale ? 'not current' : 'updating');
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
        /** A painted copy stands in the feed: the room has words to come in with (app.js's docked panel). */
        shown: () => Boolean(root),
        /** The room's newest window, admitted (chat.js): its rows replace this room's. A window that could not read
         * the conversation stream, or a refusal, keeps its entry; a bounded `complete: false` window is a window. */
        save(messages, { window: frame = null, reason_code: refused = '' } = {}) {
            if (!profile || disposed || refused || frame?.truncated_by?.includes?.('chat_source_unavailable')) return;
            write(fromHistory(messages).reduce(place, []));
        },
        /** Rows an arrivals read accounted for since the newest window: one the room has stays where it stands, a new
         * one goes where the feed puts it; one that brought none writes nothing. A room without an entry waits for its
         * next newest window. */
        append(messages) {
            if (!profile || disposed) return;
            const incoming = fromHistory(messages);
            if (!incoming.length) return;
            const rows = saved();
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
        refused(error) {
            if (refusal(error)) discard();
            else if (error?.foreignProfile && provenScope(error.servingScope)) dropPaintedPreviews();
        },
        /** The room's newest read failed: a refusal discards, anything else labels the text, now or as it paints. */
        failed(error) {
            if (refusal(error)) discard();
            else { stale = true; note('not current'); }
        },
        destroy() {
            disposed = true;
            painted.delete(release);
            doc.removeEventListener('selectionchange', onSelection);
            drop();
        },
    };
}

const MAIN_HYDRATION_MAX_DEFER_MS = 3500;
// The reasons a newest read fails for while its room may stay what it was (gateway/history.py).
const PASSING = ['history_source_unavailable', 'migrating'];

/** A failed newest read the core answered as its room's ordinary, passing unavailability: a 503 with a passing reason
 * and no archive offer, naming this exact room as one of this page's profile's ordinary rooms now (`ordinary_room` under
 * its `profile_scope`: history_paging.failed_read_facts). Any other answer (a refusal, another profile or room, the
 * archive, a room no longer ordinary, an answer that cannot be read, none at all) confirms nothing. */
export const ordinaryUnavailability = (error, chatId, scope) => {
    const body = error?.body;
    return error?.status === 503 && PASSING.includes(body?.reason_code) && !body.project_registry && Boolean(scope)
        && provenScope(body.profile_scope) === scope && body.ordinary_room === chatId;
};

/** Profile-bound first opening and the room's canonical history read door (chat.js). */
export function createRoomProfileLifecycle({
    ws, doc, chatId, isMain, isProjectOpening,
    isDestroyed, isLeaving, readHistory, nextRecentVersion, syncHistory, loadUiPreferences,
    getPreview, onPreviewShown, reconcileSavedSends, kept = () => null,
}) {
    // Per-thread storage so a project thread never bleeds into the main chat. A document that proves
    // its data profile scopes these keys with it too: after a profile change the new profile's room
    // finds no draft/reply/unconfirmed record of the old one, and the old ones stay under their own
    // keys (kept, never conflated or adopted) — the serving connection refuses their frames anyway (#3294).
    const profileScope = documentScope(doc);
    const storeKey = base => roomStoreKey(base, profileScope, chatId, isMain);
    // A confirmed mismatch invalidates pending reads; preview and live traffic keep their own proof.
    const foreignProfile = () => (typeof ws?.foreignProfile === 'function' ? ws.foreignProfile() : false);
    // Initial hydration stays shared after success; failure resets it. Reconnect and scheduled reads bypass it.
    let initialHydrationPromise = null;
    // one-shot idle gate for Main's deferred first hydration.
    let hydrationGatePromise = null;
    // The sends this tab kept show once the room's first canonical read settles them, or while its newest read fails, on
    // the core's word alone (`ordinaryUnavailability`), read on the socket that serves this page's profile and asked
    // since the room last changed or was left. That word holds while its socket stays current and confirmed, until another
    // outcome of the newest read, the registry's `projects_changed` naming the room or the room leaving the screen (a new
    // serving generation ends every socket). Then what it showed leaves the feed, kept (chat_attachments.js `proven`); a
    // return shows it again only on a word read after it, and its Send again asks anew.
    let canonical = false, proof = null, failure = null, epoch = 0;
    const asked = new WeakMap();  // a failed read -> the socket and the room's epoch it was sent under
    // Why these reads are made (history_attribution.py): a Send again's, or a failed newest read asked again.
    const RETRY = { reason: 'retry' };
    const servingSocket = () => (profileScope && ws?.isConnected?.() !== false
        && ws?.servingProfileScope?.() === profileScope ? ws.ws || null : null);
    const lapse = () => { proof = null; kept()?.withdraw(); };
    // A press of Send again on such a copy asks the core again. A newest read begun now must answer that this exact room
    // is still one of this page's profile's ordinary rooms, the room neither changed nor left meanwhile, and the socket
    // current at the press must still serve that profile: the ordinary write path the frame then takes (ws.js `send`,
    // which checks it again), where the core's acceptance asks the room once more (gateway/ws.py). Only then `true`.
    // Readable history hands the copy to the ordinary read; another current answer of the core ends the word; an answer
    // asked before the room changed or was left, or on another socket, changes nothing, nor does no answer. A current
    // refusal for authentication takes the saved text, as one of the room's own reads does (`fetchHistory`).
    async function confirmNow() {
        const socket = servingSocket(), at = epoch;
        if (canonical || !socket || proof !== socket) return false;
        let answer = null, readable = false;
        try { await readHistory({ chatId, cursor: null, tags: RETRY }); readable = true; } catch (error) { answer = error; }
        if (isDestroyed() || isLeaving()) return false;
        if (answer) getPreview().refused(answer);
        if (readable) { void syncHistory(RETRY); return false; }
        if (servingSocket() !== socket || epoch !== at) return false;
        if (ordinaryUnavailability(answer, chatId, profileScope)) { proof = socket; return true; }
        if (answer?.status || answer?.foreignProfile) lapse();
        return false;
    }
    // `room`: the room the word names, which a Send again standing on it names to the core (`ChatInbound.ordinary_room`).
    const word = { room: chatId, holds: () => canonical || (proof !== null && proof === servingSocket()), confirm: confirmNow };
    // The opening's saved text waits for this socket's proof (ws.js canShowSavedProfile: after a disconnect only the
    // current socket proves the profile): `profileConfirmed` paints it then, once, unless a canonical read landed first.
    // A room away hears no proof (room_retire.js roomSocket): its return's newest read spends it where the proof came.
    let previewOwed = false;
    const showPreview = () => { if (getPreview().show()) onPreviewShown(); };

    // Hydration triggers share one sticky request; reconnect/resync read again.
    const awaitInitialHydration = () => isLeaving() ? Promise.resolve(false) : (initialHydrationPromise ||= syncHistory({ reason: 'open' }));

    // Main briefly yields hydration to an opening Project, with a hard bound.
    function waitForHydrationWindow() {
        if (!isMain) return Promise.resolve();
        return hydrationGatePromise ||= new Promise((resolve) => {
            const deadline = Date.now() + MAIN_HYDRATION_MAX_DEFER_MS;
            // Each look waits while a Project is still opening, until the bound; a destroyed room waits for nothing.
            const attempt = () => (!isDestroyed() && Date.now() < deadline && isProjectOpening?.()
                ? setTimeout(attempt, 200) : resolve());
            if (typeof requestIdleCallback === 'function') requestIdleCallback(attempt, { timeout: 1000 });
            else setTimeout(attempt, 50);
        });
    }

    // Every read of this room (newest, a page, ↓, arrivals): a current one refused for authentication takes the saved text.
    async function fetchHistory(cursor, options = {}) {
        if (cursor == null && previewOwed && ws?.canShowSavedProfile?.() === true) { previewOwed = false; showPreview(); }
        const recentVersion = cursor == null ? nextRecentVersion() : 0, sent = { socket: servingSocket(), epoch };
        try {
            return { ...await readHistory({ chatId, cursor, ...options }), recentVersion };
        } catch (error) {
            if (error && typeof error === 'object') asked.set(error, sent);
            if (!isDestroyed() && !isLeaving() && !options.signal?.aborted) getPreview().refused(error);
            throw error;
        }
    }

    async function startOpening() {
        // The owners below exist first: no read stands between opening and saved text.
        await null;
        if (!isDestroyed()) {
            showPreview();
            previewOwed = !getPreview().shown() && !foreignProfile() && ws?.canShowSavedProfile?.() !== true;
        }
        await loadUiPreferences();
        // Main waits for hydration; Projects pass through. History reads share one flight.
        await waitForHydrationWindow();
        if (isDestroyed()) return;
        await awaitInitialHydration();
    }

    return {
        storeKey, foreignProfile, fetchHistory, awaitInitialHydration, waitForHydrationWindow, startOpening,
        get initialHydration() { return initialHydrationPromise; },
        settleInitialHydration(promise) { initialHydrationPromise = promise; },
        resetInitialHydration() { initialHydrationPromise = null; },
        // A canonical read settles the kept sends (the healthy path, any document), and from then on stands for them.
        noteCanonicalHistory(wasFirstLoad) {
            canonical = true; proof = null; failure = null;
            kept()?.canonicalRead();
            if (wasFirstLoad) reconcileSavedSends();
        },
        /** The newest read failed (chat.js): only the core's word on this room, read on this socket, shows the kept sends. */
        noteHistoryFailure(error) {
            if (canonical) return;
            const sent = asked.get(error), ordinary = ordinaryUnavailability(error, chatId, profileScope);
            if (ordinary && sent?.socket && sent.socket === servingSocket() && sent.epoch === epoch) {
                proof = sent.socket;
                failure = null;
                if (kept()?.unreconciled) reconcileSavedSends(word);
                return;
            }
            lapse();
            // The word came before this socket proved the profile, or before the room changed: asked again on this one.
            failure = ordinary ? error : null;
            if (failure && servingSocket()) void Promise.resolve().then(() => syncHistory(RETRY));
        },
        /** A socket proved the profile it serves (ws.js `connection_profile`): a word read on another socket lapses, and one
         *  that waited for this proof is asked again. */
        profileConfirmed() {
            kept()?.withdraw();
            if (failure && !canonical) void syncHistory(RETRY);
            // An opening this proof held back paints its saved text now, once; a room away paints nothing.
            const owed = previewOwed;
            previewOwed = false;
            if (owed && !isDestroyed() && !isLeaving()) showPreview();
        },
        /** The registry's `projects_changed` naming this Project room, or no room: the core's word on it lapses, asked again. */
        roomChanged(frame) {
            const named = Number(frame?.chat_id) || 0;
            if (isMain || canonical || (named && named !== chatId)) return;
            const held = proof !== null || failure !== null;
            epoch += 1;
            lapse();
            if (held) void syncHistory({ reason: 'refresh' });
        },
        /** The room leaves the screen (chat.js `leave`): the core's word lapses and what it showed leaves the feed, kept; no
         *  answer asked before can bring it back, so a return shows it only on a word read after it. */
        left() { epoch += 1; lapse(); },
    };
}
