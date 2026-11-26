/** A Project room that is left keeps its content while it leaves and stays whole while it is away (docs/DESIGN.md §10
 * "Moving between places"; owner 2026-11-03: a return within this page session finds the room where it was left).
 *
 * `retire(pid)` takes the room on screen out of the live instances and marks it leaving: inert, `data-leaving` (the
 * stylesheet lays it over the arriving room), and `leave()` (chat.js: its place kept, no read receipt, not shown). A room
 * already hidden has nothing to show leaving and is put away at once. `settle(pid, inst, put, turn)` ends that once its
 * exit has run (`put`: put away) or when it is reopened meanwhile (`put` false: back among the instances, the same
 * room); a room another one replaced meanwhile is destroyed. Each exit is its own turn (`exits()`): a room reopened and
 * left again is leaving anew, and the end of its earlier exit settles nothing.
 *
 * Away, a room is hidden and quiet: its socket view (`roomSocket`) carries only its own sends' custody, no state
 * snapshot reaches it (chat.js), its reads have stopped and no late read is drawn (chat.js), while its DOM, draft,
 * staged files, disclosures and place stay. A Project room another place replaces (`retire`, `putAway`) also closes the
 * reader or file dialog it opened and pauses what it plays, as when such rooms were destroyed; a room merely suspended
 * (`suspend`: Main behind another page or a Project, any room in a hidden window) keeps both, its caller closing what
 * would cover the next place. The owner's selection in it is kept here when it leaves (the page has one selection: the
 * next room's would replace it) and given back when it returns. `show(pid)` brings a kept room back: the selection, its
 * frames, and a catch-up read of what it missed, as after a reconnect.
 *
 * Every room visited in this page session stays away, whole, and returns at once (parent decision 2026-11-04: no
 * eviction, no snapshot, no remount): its own owners keep its forms, draft and place. Only the registry's word that its
 * project is going away (`forget`) or the page's end lets one go: at once when it is away, when its exit settles, or on
 * a later projects read once the work only it can finish has ended (staged files, an upload in flight, a send not yet
 * saved: `hasPendingWork`, marked `data-pending-work`). Memory grows with the rooms visited, not with the rooms whose
 * work runs. app.js owns when; this owns what leaving and being away are. */

import { documentScope } from './chat_preview.js';
import { isWindowText } from './chat_row_placement.js';
import { notifyRetainedSelectionReleased } from './selection_lifecycle.js';

function keptRanges(page) {
    const selection = page.ownerDocument?.getSelection?.();
    const inside = (node) => Boolean(node && page.contains(node));
    if (!selection?.rangeCount || selection.isCollapsed || !inside(selection.anchorNode) || !inside(selection.focusNode)) return null;
    return Array.from({ length: selection.rangeCount }, (_, index) => {
        const range = selection.getRangeAt(index);
        return [range.startContainer, range.startOffset, range.endContainer, range.endOffset];
    });
}

function giveBackRanges(page, ranges) {
    const doc = page.ownerDocument;
    const selection = doc?.getSelection?.();
    if (!ranges || !selection || !ranges.every(([start, , end]) => start.isConnected && end.isConnected)) return false;
    selection.removeAllRanges();
    for (const [start, startOffset, end, endOffset] of ranges) {
        const range = doc.createRange();
        range.setStart(start, startOffset);
        range.setEnd(end, endOffset);
        selection.addRange(range);
    }
    return selection.rangeCount > 0 && !selection.isCollapsed;
}

export function createRoomRetirement({ instances, cancelPaint = () => {}, restorePlaces = () => {}, latestState = () => null }) {
    const retiring = new Map();  // pid → the instance leaving the screen
    const turns = new Map();     // pid → which exit that is
    const forgotten = new Set(); // pids the registry says are going away, until their room is let go
    let turn = 0;
    const rememberSelection = (inst) => {
        if (inst && !inst.page.hidden) inst.keptRanges = keptRanges(inst.page) || inst.keptRanges;
    };
    const releaseSelection = inst => {
        if (!inst.keptRanges) return;
        inst.keptRanges = null;
        notifyRetainedSelectionReleased(inst.page.ownerDocument);
    };
    const restoreSelection = (inst) => {
        if (!inst?.keptRanges) return;
        // The shell first exposes the room and releases its layer's inert. WebKit cannot select inside either.
        if (inst.page.closest?.('[inert]') || inst.page.getClientRects && !inst.page.getClientRects().length) return;
        if (!inst.keptRanges.every(([start, , end]) => start.isConnected && end.isConnected)
            || giveBackRanges(inst.page, inst.keptRanges)) releaseSelection(inst);
    };
    // A room on screen keeps no copy: its live selection is its own. An ordinary press (selection_lifecycle.js) ends
    // the copies of every shown room; the window shown again (`keepCovered`) those of the rooms no layer covers, as a
    // room under an inert layer holds no live selection and its copy waits for the layer's release. A room away keeps
    // the one it left with.
    const forgetShownSelections = (others = [], { keepCovered = false } = {}) => {
        for (const inst of [...instances.values(), ...others]) {
            if (!inst || inst.ws?.paused || inst.page?.hidden) continue;
            if (!keepCovered || !inst.page?.closest?.('[inert]')) releaseSelection(inst);
        }
    };
    // A room leaving the screen: the selection in it is kept, then it leaves (chat.js), its socket view goes away.
    const leave = (inst, windowHidden = false) => {
        if (!inst) return;
        if (!windowHidden) inst.forgetWindowReturn?.();  // also ends a window's suspension when the room itself leaves
        if (inst.ws?.paused) return;
        rememberSelection(inst);
        inst.leave?.(windowHidden);
        inst.ws?.pause?.();
    };
    // Main has the same lifetime as a retained Project, but the shell owns its visibility (also while covered).
    const show = (inst) => {
        if (!inst) return;
        delete inst.page.dataset.pendingWork;
        inst.stay?.();
        restorePlaces(inst.page);
        // The shell restores selection after its layer has released inert and returned focus. Restoring here
        // loses it again when WebKit handles that focus change (Main can be visible behind a docked Project).
        if (!inst.ws?.paused) return;
        inst.ws.resume();
        inst.page.dispatchEvent?.(new Event('ouro:feed-width-settled', { bubbles: true }));
        const state = latestState();
        if (state) inst.hydrateStateSnapshot?.(state, 0);
    };
    const setLeaving = (inst, leaving) => {
        // WebKit drops a range when its room becomes inert. Keep the place and selection before that write.
        if (leaving) leave(inst);
        inst.page.inert = leaving;
        if (leaving) inst.page.dataset.leaving = ''; else { delete inst.page.dataset.leaving; inst.stay?.(); }
    };
    const destroy = (pid) => {
        const inst = instances.get(pid);
        instances.delete(pid);
        forgotten.delete(pid);
        cancelPaint(pid, inst);
        restorePlaces.forget?.(inst?.page);
        inst?.destroy?.();
    };
    // Every room going away that is away now (not leaving, not shown) and holds no work only it can finish is let go;
    // one still leaving waits for its exit to settle, a pid with no room any more is done.
    const release = () => {
        for (const pid of forgotten) {
            const inst = instances.get(pid);
            if (!inst && !retiring.has(pid)) forgotten.delete(pid);
            else if (inst?.page.hidden && !inst.hasPendingWork?.()) destroy(pid);
        }
    };
    // Another place replaces the room on screen: the reader or file dialog it opened closes and what it plays pauses.
    const replaced = (inst) => {
        rememberSelection(inst);
        inst.closeTransient?.(true);
    };
    // A room that has left the screen: hidden and kept for its return.
    const away = (inst) => {
        inst.page.hidden = true;
        if (inst.hasPendingWork?.()) inst.page.dataset.pendingWork = '1';
    };
    const rooms = {
        suspend: leave,
        restore: show,
        rememberSelection,
        restoreSelection,
        forgetShownSelections,
        retiring,
        exits: () => [...retiring].map(([pid, inst]) => [pid, inst, turns.get(pid)]),
        retire(pid) {
            const inst = instances.get(pid);
            if (!inst?.page || inst.page.hidden) { rooms.putAway(pid); return; }
            instances.delete(pid);
            cancelPaint(pid, inst);
            replaced(inst);
            setLeaving(inst, true);
            retiring.set(pid, inst);
            turns.set(pid, ++turn);
        },
        settle(pid, inst = retiring.get(pid), put = true, at = turns.get(pid)) {
            if (!inst || retiring.get(pid) !== inst || turns.get(pid) !== at) return;
            retiring.delete(pid);
            turns.delete(pid);
            if (instances.has(pid)) {
                restorePlaces.forget?.(inst.page);
                inst.destroy?.();
            } else if (!put) {
                instances.set(pid, inst);
                setLeaving(inst, false);
            } else {
                instances.set(pid, inst);
                // Put away straight from its exit: still left (no `stay`), no longer drawn as leaving.
                inst.page.inert = false;
                delete inst.page.dataset.leaving;
                away(inst);
            }
            release();  // a room whose project went away during its exit is let go now
        },
        // A room on screen leaves without motion; one already away stays as it is.
        putAway(pid) {
            const inst = instances.get(pid);
            if (!inst) return;
            cancelPaint(pid, inst);
            if (inst.page.hidden) return;
            replaced(inst);
            leave(inst);
            away(inst);
        },
        // Back on screen (app.js, once it is shown): its inner boxes' scroll, its selection, then its frames and the
        // catch-up read. A room shown is wanted again: a going-away word before this no longer lets it go.
        show(pid) {
            forgotten.delete(pid);
            show(instances.get(pid));
        },
        // The registry says these projects are going away (app.js, on every projects read; never mere absence from a
        // bounded list): a room kept for one has nothing to return to. Called with none, it lets go the rooms said
        // earlier whose exit or work has ended since. The only release before the page ends.
        forget(...pids) {
            for (const pid of pids) {
                forgotten.add(pid);
                (instances.get(pid) || retiring.get(pid))?.forgetWindowReturn?.();
            }
            release();
        },
    };
    return rooms;
}

/** A room back on screen or reconnected reads what it missed (DESIGN "History edges"). A read still out after the
 * bound a loading state waits for (300 ms) is said in the room's status (`active`, then `changed()`): the last known
 * state is not passed as now. `begin(current, read)` arms the bound for `read` (`current()` false: the room left
 * meanwhile). `reading` shares that work and `settled(current)` waits it out, a later reconnect's read too; `owed`
 * separately survives failure and the interval before begin. Leaving, disconnecting and beginning a reconnect record
 * the latest requested history version. Only an admitted newer read that was not cut short (`confirmWindow` takes its
 * answer: a refused source or a projection fault leaves rows out, CLIENT_CONTRACT "Reading history") discharges that
 * absence, however a page bounds it; a late pre-disconnect response cannot. A disconnect, and the read beginning a
 * reconnect or a return (a window hidden on a live connection hears no start or end frame either), mark the sends still
 * waiting: only a later admitted read carrying the host's own row for one (`answeredOwnerIds`) ends that wait; in a
 * window that missed nothing the live start does. `confirmWindow` returns whether the read was
 * whole; a fault owes repair even without an absence. `end` ends only the matching read's
 * feedback; `cancel` records absence. `retry` distinguishes an unconfirmed attempt from absence alone. */
const cutWindow = ({ reason_code: refused, window } = {}) => Boolean(refused)
    || Boolean(window?.truncated_by?.some?.(cause => String(cause).endsWith('_projection_failed')));
const degradedWindow = (data = {}) => cutWindow(data)
    || Boolean(data.window?.truncated_by?.some?.(cause => String(cause).endsWith('_source_unavailable')));

/** Local owner rows in a drawn history window by which the host answers for a pending send: its dispatch entered
 * (`ingress_dispatched`), or, its process having ended, no dispatch fact at all, an outcome unknown that the bubble says
 * as its delivery doubt (chat_attachments.js `settle`). `ingress_pending` is no answer: the running host still owes the
 * start, and its live frame ends the wait. */
export function answeredOwnerIds(rows, pending, { chatId, sessionId, localOwnerRow }) {
    if (!pending?.size || !Array.isArray(rows)) return [];
    const matched = [];
    for (const row of rows) {
        if (row?.role !== 'user' || row.is_progress || row.declared_author) continue;
        if (row.ingress_accepted !== true || row.ingress_pending === true) continue;
        const id = String(row.client_message_id || '');
        const entry = id && pending.get(id);
        if (!entry || !localOwnerRow(row)) continue;
        if (row.chat_id != null && Number(row.chat_id) !== chatId) continue;
        if (row.source && row.source !== 'web') continue;
        if (row.sender_session_id != null && row.sender_session_id !== sessionId) continue;
        if (entry.rowAddress && row.row_address != null && row.row_address !== entry.rowAddress) continue;
        matched.push(id);
    }
    return matched;
}

export function createCatchUp(changed, bound = 300, pending = null) {
    let timer = null, active = false, reading = null, gate = null, after = null, failed = false, awaited = null;
    const stop = () => { clearTimeout(timer); timer = null; };
    const wake = () => { const current = gate; gate = null; current?.resolve(); };
    const arm = (read) => {
        wake();
        if (!read) return;
        let resolve;
        const promise = new Promise((done) => { resolve = done; });
        gate = { promise, resolve };
        if (typeof read.then === 'function') read.then(() => { if (reading === read) wake(); }, () => { if (reading === read) wake(); });
    };
    const markDisconnected = (requested) => {
        if (!pending) return;
        for (const entry of pending.values()) {
            if (!entry.queued) entry.disconnectedAfter = Math.max(entry.disconnectedAfter ?? -1, requested);
        }
    };
    const settleSends = (version, confirmedIds = null) => {
        if (!pending || !(version > 0) || !confirmedIds) return false;
        const confirmed = confirmedIds instanceof Set ? confirmedIds : new Set(confirmedIds);
        if (!confirmed.size) return false;
        let settled = false;
        for (const id of confirmed) {
            const entry = pending.get(id);
            if (entry && entry.disconnectedAfter != null && version > entry.disconnectedAfter) {
                pending.delete(id);
                settled = true;
            }
        }
        return settled;
    };
    return {
        get active() { return active; },
        get reading() { return reading; },
        get wait() { return gate?.promise || null; },
        // A real reconnect's own whole read, still to come or out: the room's other reads wait for it, then read only
        // what is left (a return shares its paint receipt's read instead).
        get reconnecting() { return awaited && awaited === reading ? gate?.promise || null : null; },
        get owed() { return after !== null; },
        get retry() { return after !== null && failed && !reading; },
        wouldConfirm(data = {}) {
            return !cutWindow(data) && (after === null || (!degradedWindow(data) && data.recentVersion > after));
        },
        trackSend(id, queued = false) {
            if (!pending || !id) return;
            pending.set(id, { clientMessageId: id, timestamp: Date.now(), queued: Boolean(queued) });
        },
        flushSend(id) {
            const entry = pending?.get(String(id || ''));
            if (entry) entry.queued = false;
        },
        confirmWindow(data = {}, confirmedIds = null) {
            const { recentVersion } = data;
            const cut = cutWindow(data);
            const degraded = degradedWindow(data);
            if (cut) after = Math.max(after ?? 0, recentVersion);
            else if (!degraded && recentVersion > after) after = null;
            failed = after !== null;
            if (!degraded) settleSends(recentVersion, confirmedIds);
            return !cut && !(degraded && after !== null);
        },
        async settled(current) { while (current() && gate) await gate.promise; },
        begin(current, read = null, requested = 0, reconnect = false) {
            stop();
            after = requested;
            reading = read;
            awaited = reconnect ? read : null;
            markDisconnected(requested);  // a return on a live connection missed the same frames
            arm(read);
            timer = setTimeout(() => { timer = null; if (current()) { active = true; changed(); } }, bound);
        },
        end(read = null) {
            if (read !== reading) return;
            stop(); reading = null; awaited = null; wake();
            if (read) failed = after !== null;
            if (active || read && failed) { active = false; changed(); }
        },
        cancel(requested = 0, disconnected = false) {
            stop(); failed ||= Boolean(reading); active = false; reading = null; awaited = null; wake(); after = requested;
            if (disconnected) markDisconnected(requested);
        },
    };
}

const WINDOW_ROWS = 12, WINDOW_TEXT_BYTES = 64 * 1024;
const rowRoom = /^row:(\d+)@[^#\s]+#[a-f0-9]{12}$/;

/** The small exception to an away room's socket custody. A shown follower that the window hides may keep the words of
 * its own ordinary assistant rows. No re-dispatch, snapshots, other appearances, or log subscription. The same bounded
 * batch can span hides until reconciliation; on loss of proof or the bound only ordinary history fills the gap. */
export function createWindowReturn({ socket, doc, chatId, projectId, accepts, knownChild, onInvalidate = () => {} }) {
    let lease = null;
    function drop() {
        const old = lease;
        lease = null;
        old?.off.forEach(dispose => dispose());
    }
    const connected = scope => Boolean(scope) && socket?.isConnected?.() === true && !socket.foreignProfile?.()
        && socket.servingProfileScope?.() === scope && socket.ws;
    function invalidate() { drop(); onInvalidate(); }
    const cost = ({ text, address, ts }) => 2 * (text.length + address.length + ts.length);
    return {
        drop,
        // A settled page releases the captured batch, other than words still held in the feed by the reader.
        // Those words keep the same proof and count towards the same bound on the next hide.
        settle(held) {
            if (!lease) return;
            if (!lease.valid()) return invalidate();
            for (const address of lease.rows.keys()) if (!held(address)) lease.rows.delete(address);
            if (!lease.rows.size) return drop();
            lease.shown = lease.rows.size;
            lease.settled = true;
        },
        release(address) {
            if (lease?.settled && lease.rows.delete(address) && --lease.shown === 0) drop();
        },
        hold(following = true, ready = following) {
            if (lease?.valid()) {
                lease.collecting = following && (lease.settled ? ready : lease.collecting);
                lease.settled = false;
                return;
            }
            if (lease) invalidate();
            const scope = documentScope(doc), connection = connected(scope);
            if (!following || !ready || !doc.hidden || !connection) return;
            const rows = new Map();
            const valid = () => connected(scope) === connection && documentScope(doc) === scope;
            function keep(msg) {
                if (!doc.hidden || !lease?.collecting || !isWindowText(msg, knownChild) || msg.chat_id !== chatId || msg.project_id && msg.project_id !== projectId
                        || !accepts(msg)) return;
                if (!valid()) return invalidate();
                const address = msg.row_address;
                if (typeof address !== 'string' || Number(rowRoom.exec(address)?.[1]) !== chatId
                        || typeof msg.ts !== 'string' || !Number.isFinite(Date.parse(msg.ts)) || typeof msg.content !== 'string') return invalidate();
                if (!msg.content || rows.has(address)) return;
                // Retain only presentation fields and the physical address, never the producer's frame.
                const row = { address, text: msg.content, ts: msg.ts, markdown: msg.markdown === true,
                    initiator: msg.initiator === 'consciousness' ? 'consciousness' : '' };
                if (rows.size >= WINDOW_ROWS || [...rows.values()].reduce((used, saved) => used + cost(saved), cost(row)) > WINDOW_TEXT_BYTES) return invalidate();
                rows.set(address, row);
            }
            lease = { rows, valid, shown: 0, collecting: true, settled: false, off: [socket.on('chat', keep), socket.on('close', invalidate),
                socket.on('connection_profile', () => { if (!valid()) invalidate(); }),
                socket.on('projects_changed', msg => { if (projectId && (!msg?.chat_id || Number(msg.chat_id) === chatId)) invalidate(); })] };
        },
        take(following = true) {
            if (doc.hidden || !lease?.valid()) { drop(); return null; }
            lease.collecting &&= following;
            const rows = lease.collecting ? [...lease.rows.values()].slice(lease.shown) : [];
            lease.shown = lease.rows.size;
            return rows;
        },
    };
}

// What a room away still hears: the outcome of its own sends (the socket queue wrote or dropped one, the host echoed
// and saved one, a restart asks for a send again, the socket closed before one was saved) and its tasks' incidents,
// which toast once page-wide and are never drawn away (chat.js). Everything else waits for its return; what it heard
// is drawn then too.
const CUSTODY = {
    outbound_sent: () => true,
    outbound_dropped: () => true,
    close: () => true,
    chat: (msg) => msg?.role === 'user' || msg?.system_type === 'initialization_notice'
        || Boolean(msg?.is_progress && msg.task_incident),
};

/** A Project room's own view of the page's one socket (app.js). Away (`pause`), its subscriptions are released but for
 * its sends' custody; back (`resume`), they are taken again and, with the socket open, the room hears `open` as after a
 * reconnect (`previouslyConnected`), marked `resumed` so the return's paint request can share its catch-up read. A
 * real socket reconnect has no such mark and still owes a read newer than one it interrupted. */
export function roomSocket(ws) {
    const subscriptions = new Set();
    let paused = false;
    const attach = (sub) => {
        const custody = paused ? CUSTODY[sub.event] : null;
        sub.off = paused && !custody ? null : ws.on(sub.event, custody ? (msg) => custody(msg) && sub.fn(msg) : sub.fn);
    };
    const reattach = () => {
        for (const sub of subscriptions) { sub.off?.(); attach(sub); }
    };
    return {
        get ws() { return ws.ws; },
        get socket() { return ws; },  // the page's own socket, for what outlives the room's view of it (a live call)
        get paused() { return paused; },
        send: (...args) => ws.send(...args),
        isConnected: () => !paused && ws.isConnected?.() !== false,
        serverAnswersWhileDown: () => !paused && ws.serverAnswersWhileDown?.() === true,
        previouslyConnected: () => ws.previouslyConnected?.() === true,  // the page's history, never paused
        // The page's profile proofs, never paused: without them the per-room gates in chat.js
        // (history paint, live frames, reads, the saved text preview) read false through the room's own socket view
        // (#3294; the preview's gate went dark when only the page socket learned it, #4149).
        foreignProfile: () => ws.foreignProfile?.() === true,
        servingProfileScope: () => ws.servingProfileScope?.() || '',
        canShowSavedProfile: () => ws.canShowSavedProfile?.() === true,
        on(event, fn) {
            const sub = { event, fn, off: null };
            subscriptions.add(sub);
            attach(sub);
            return () => { sub.off?.(); subscriptions.delete(sub); };
        },
        pause() {
            if (paused) return;
            paused = true;
            reattach();
        },
        resume() {
            if (!paused) return;
            paused = false;
            reattach();
            if (ws.isConnected?.() === false) return;
            for (const sub of [...subscriptions]) if (sub.event === 'open') sub.fn({ previouslyConnected: true, resumed: true });
        },
    };
}
