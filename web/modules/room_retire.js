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
    const restoreSelection = (inst) => {
        if (!inst?.keptRanges) return;
        // The shell first exposes the room and releases its layer's inert. WebKit cannot select inside either.
        if (inst.page.closest?.('[inert]') || inst.page.getClientRects && !inst.page.getClientRects().length) return;
        if (!inst.keptRanges.every(([start, , end]) => start.isConnected && end.isConnected)
            || giveBackRanges(inst.page, inst.keptRanges)) inst.keptRanges = null;
    };
    // A room on screen keeps no copy: its live selection is its own. An ordinary press (selection_lifecycle.js) ends
    // the copies of every shown room; the window shown again (`keepCovered`) those of the rooms no layer covers, as a
    // room under an inert layer holds no live selection and its copy waits for the layer's release. A room away keeps
    // the one it left with.
    const forgetShownSelections = (others = [], { keepCovered = false } = {}) => {
        for (const inst of [...instances.values(), ...others]) {
            if (!inst || inst.ws?.paused || inst.page?.hidden) continue;
            if (!keepCovered || !inst.page?.closest?.('[inert]')) inst.keptRanges = null;
        }
    };
    // A room leaving the screen: the selection in it is kept, then it leaves (chat.js), its socket view goes away.
    const leave = (inst) => {
        if (!inst) return;
        if (inst.ws?.paused) return;
        rememberSelection(inst);
        inst.leave?.();
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
            for (const pid of pids) forgotten.add(pid);
            release();
        },
    };
    return rooms;
}

/** A room back on screen or reconnected reads what it missed (DESIGN "History edges"). A read still out after the
 * bound a loading state waits for (300 ms) is said in the room's status (`active`, then `changed()`): the last known
 * state is not passed as now. `begin(current)` arms the bound (`current()` false: the room left meanwhile), `end()`
 * is the read settled, `cancel()` a room leaving or torn down (its return reads, and says so, anew). */
export function createCatchUp(changed, bound = 300) {
    let timer = null, active = false;
    const stop = () => { clearTimeout(timer); timer = null; };
    return {
        get active() { return active; },
        begin(current) {
            stop();
            timer = setTimeout(() => { timer = null; if (current()) { active = true; changed(); } }, bound);
        },
        end() { stop(); if (active) { active = false; changed(); } },
        cancel() { stop(); active = false; },
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
 * reconnect (`previouslyConnected`): it reads the history and state it missed through its existing reconnect path. */
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
            for (const sub of [...subscriptions]) if (sub.event === 'open') sub.fn({ previouslyConnected: true });
        },
    };
}
