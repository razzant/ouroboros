/** One stack of layers over the room (docs/DESIGN.md "Shell", plan P2).
 *
 * A layer is anything that lies over the page and is dismissed as one thing: the phone drawer and the Project panel.
 * One page part is a layer too: Files' file shown alone on a narrow page (files.js), so Escape and Back leave it the way
 * Back to folder does. It is pushed with `page: true` and lies in the page, under every overlay: opened while the
 * drawer is open (a slow read finishing), it takes its place beneath the drawer, which Escape and Back close first. It
 * is not modal and its focus is the owner's (`focus: false`, no opener): the stack neither moves focus in nor back.
 * Dialogs, menus and the work list keep their own focus traps and Escape. Every way of leaving ends in the same place:
 * Escape, the scrim, × call `pop()`, which closes exactly one layer, the top one, through the layer's own `onClose`;
 * the browser's or Android's Back and Telegram's Back call `back()`, which first gives an open dialog over the layer
 * its Escape (the dialog is what is on top) and otherwise pops. The owner of a layer's
 * state keeps owning it: it calls `push` when it opens and `remove` when it closes (both idempotent), and `onClose` is
 * where the stack asks the owner to close.
 *
 * History: while at least one layer is open there is exactly one same-URL history entry (`pushState`), so Back
 * closes a layer before it leaves the app; `popstate` pops one layer; closing the last layer programmatically
 * consumes that entry (`history.back()`) and the popstate it causes is swallowed. The entry is reconciled once per
 * task, so "close the drawer, open the Project" never leaves a stray entry (the b2 mockup's finding). No URL is ever
 * written: the desktop shell and the Telegram mini app have no address bar (tests/test_navigation_shell_static.py).
 *
 * Inert: a modal layer makes the rest of the shell inert (siblings along its ancestor path inside the shell's
 * container, never `<body>`'s other children, where dialogs, menus and toasts opened over the layer live; what the page
 * had already made inert is left alone); a non-modal layer (the docked Project panel) leaves the sidebar live. Focus goes
 * into the layer on open and returns to the opener on close, unless it was moved out to something still live. Every
 * listener has a disposer.
 *
 * One commit per change: `batch(fn, { inertAfter })` runs the owner's pushes and removes and settles once at its end:
 * one history reconcile, one snapshot to the listeners (Telegram's Back follows once, never hide-then-show), focus
 * once. A drawer replaced by a Project keeps the room under both inert throughout. Writing `inert` restyles the room's
 * whole subtree, so the owner of a motion may pass `inertAfter`, a promise of the motion's end: the stack, history,
 * Back and focus into a layer that is live change at once, and only the inert difference waits, with the focus that
 * needs it (an opener inside the room it releases). A later commit supersedes a pending one. `push` and `remove`
 * outside a batch are a batch of their own with nothing to wait for. */

const LAYER_STATE = { ouroLayers: 1 };

export function createLayerStack({ win = globalThis.window, doc = globalThis.document, history = win?.history } = {}) {
    const stack = [];
    const listeners = new Set();
    const inerted = new Set();
    const hist = { armed: false, skip: 0, queued: false };
    let disposed = false;
    let depth = 0;        // nested batches settle once, at the outermost end
    let change = null;    // that batch's layers: newly pushed ids, removed layers
    let epoch = 0;        // the latest commit; an older pending inert difference applies nothing
    let pendingInert = null;  // that commit's inert difference while it waits for the motion
    let owed = null;          // the focus that waits for it

    const top = () => stack[stack.length - 1] || null;
    const has = (id) => stack.some((layer) => layer.id === id);
    const get = (id) => stack.find((layer) => layer.id === id) || null;

    function emit() {
        const snapshot = { size: stack.length, top: top()?.id || null, ids: stack.map((layer) => layer.id) };
        listeners.forEach((fn) => { try { fn(snapshot); } catch (err) { console.error('layers: listener failed', err); } });
    }

    // ---- inert: the topmost modal layer's siblings along its path up to the shell's container, and everything below it.
    // Only what changes is written: flipping `inert` on a room restyles its whole subtree (≈130 ms on a long feed in
    // WebKit), so a sync that keeps the same modal picture (an owner re-syncing every few seconds, a drawer → Project
    // replacement) writes nothing. What another owner made inert stays theirs.
    function syncInert() {
        const wanted = new Set();
        const modalIndex = stack.map((layer) => layer.modal).lastIndexOf(true);
        if (modalIndex >= 0 && doc?.body) {
            const modal = stack[modalIndex];
            const above = new Set(stack.slice(modalIndex + 1).map((layer) => layer.el).filter(Boolean));
            for (const el of stack.slice(modalIndex).flatMap((layer) => layer.live || [])) above.add(el);
            const keep = (node) => node === modal.el || [...above].some((el) => el === node || node.contains?.(el) || el.contains?.(node));
            let node = modal.el;
            while (node?.parentElement && node.parentElement !== doc.body) {
                for (const sibling of node.parentElement.children) {
                    if (sibling === node || keep(sibling) || sibling.tagName === 'SCRIPT') continue;
                    if (sibling.inert && !inerted.has(sibling)) continue;
                    wanted.add(sibling);
                }
                node = node.parentElement;
            }
        }
        for (const el of inerted) {
            if (wanted.has(el)) continue;
            el.inert = false;
            inerted.delete(el);
        }
        for (const el of wanted) {
            if (!el.inert) el.inert = true;
            inerted.add(el);
        }
    }

    // ---- history: one entry while the stack is not empty, reconciled once per task
    function reconcileHistory() {
        if (hist.queued || !history || typeof history.pushState !== 'function') return;
        hist.queued = true;
        Promise.resolve().then(() => {
            hist.queued = false;
            if (disposed) return;
            try {
                if (stack.length && !hist.armed) {
                    history.pushState(LAYER_STATE, '');
                    hist.armed = true;
                } else if (!stack.length && hist.armed) {
                    hist.armed = false;
                    hist.skip += 1;
                    history.back();
                }
            } catch {
                hist.skip = 0;  // a sandboxed document: layers still work, Back does not reach them
            }
        });
    }
    const onPopState = () => {
        if (hist.skip > 0) { hist.skip -= 1; return; }
        hist.armed = false;
        if (stack.length) back('history');
    };

    // ---- Escape closes the top layer unless something inside already took the key (a menu, the work list, a dialog)
    const onKeydown = (event) => {
        if (event.key !== 'Escape' || event.defaultPrevented || event.isComposing || !stack.length) return;
        if (doc?.querySelector?.('dialog[open]')) return;  // a native modal dialog owns Escape until it adapts
        event.preventDefault();
        pop('escape');
    };

    function focusInto(layer) {
        if (layer.focus === false) return;
        if (typeof layer.focus === 'function') { layer.focus(); return; }
        const el = layer.el;
        if (!el || typeof el.focus !== 'function') return;
        if (!el.hasAttribute?.('tabindex')) el.tabIndex = -1;
        try { el.focus({ preventScroll: true }); } catch { /* detached */ }
    }
    // The opener takes back a focus the layer held or that is lost; one the person moved beside a live layer (Main's
    // field beside the docked Project) stays theirs.
    function restoreFocus(layer) {
        const opener = layer.opener, at = doc?.activeElement;
        if (!opener || opener.isConnected === false || typeof opener.focus !== 'function') return;
        if (opener.closest?.('[inert]')) return;
        const outside = at && at !== doc.body && !layer.el?.contains?.(at);
        if (outside && !at.closest?.('[inert]') && !at.closest?.('[hidden]')) return;
        try { opener.focus({ preventScroll: true }); } catch { /* gone */ }
    }

    // Focus after a commit: into the top layer if this commit opened it, else back to the opener of the last layer it
    // closed. `waits()` says whether that target still lies in a subtree the pending inert difference will release.
    function focusPlan(done) {
        const layer = top();
        if (layer && done.pushed.has(layer.id)) {
            return { run: () => { if (stack.includes(layer)) focusInto(layer); }, waits: () => Boolean(layer.el?.closest?.('[inert]')) };
        }
        const left = done.removed[done.removed.length - 1];
        if (!left || done.pushed.size) return null;
        return { run: () => restoreFocus(left), waits: () => Boolean(left.opener?.closest?.('[inert]')) };
    }

    function settle(done, inertAfter) {
        reconcileHistory();
        const mine = ++epoch;
        // A commit that moves no focus of its own keeps the focus a commit it supersedes still owes: the paint at the
        // end of a handover's beat commits the same stack again while the motion runs.
        const focus = focusPlan(done) || owed;
        owed = null;
        if (!inertAfter) {
            pendingInert = null;
            syncInert();
            emit();
            focus?.run();
            return;
        }
        emit();
        if (focus?.waits()) owed = focus;
        else focus?.run();
        let applied = false;
        const apply = () => {
            if (applied || mine !== epoch || disposed) return;
            applied = true;
            if (pendingInert === apply) pendingInert = null;
            syncInert();
            const freed = owed;
            owed = null;
            const at = doc?.activeElement;
            // Focus the owner moved meanwhile is the owner's; only a lost focus is placed.
            if (freed && (!at || at === doc.body || at.closest?.('[inert]'))) freed.run();
        };
        pendingInert = apply;
        Promise.resolve(inertAfter).then(apply, apply);
    }
    // The keyboard does not wait for the motion: Tab while the difference waits applies it first, so focus never
    // walks into the room the layer covers.
    const onTab = (event) => { if (event.key === 'Tab') pendingInert?.(); };

    /** One commit: `fn` pushes and removes; the stack settles once when the outermost batch ends (also when `fn`
     * throws). `inertAfter` (the outermost batch's): a promise of the motion's end, after which the inert difference
     * and the focus that needs it are applied; without it they are applied now. */
    function batch(fn, { inertAfter = null } = {}) {
        if (depth++ === 0) change = { pushed: new Set(), removed: [] };
        try { return fn(); } finally {
            if (--depth === 0) {
                const done = change;
                change = null;
                if (!disposed) settle(done, inertAfter);
            }
        }
    }

    /** Open: `{ id, el, modal = true, onClose(reason), opener = activeElement, focus, live, page }` (`live`: elements
     * beside the layer that stay usable under a modal one, its scrim; `page`: a part of the page, placed under every
     * overlay already open). A known id keeps its
     * place in the stack and takes the new facts (an owner that syncs its state calls push on every sync). */
    function push(layer) {
        if (disposed || !layer || !layer.id) return false;
        return batch(() => {
            const existing = get(layer.id);
            if (existing) {
                Object.assign(existing, layer);
                return true;
            }
            // A layer that replaces others in one commit (the Project picked in the drawer) returns focus where they
            // would have: the opener is followed out of every layer the commit retired (a row in the closed drawer,
            // then the ☰ that opened the drawer in the replaced room) to the first one that stays.
            const retired = new Set(change.removed);
            let opener = doc?.activeElement && doc.activeElement !== doc.body ? doc.activeElement : null, gone;
            while (opener && (gone = [...retired].find((left) => left.el?.contains?.(opener)))) {
                retired.delete(gone);
                opener = gone.opener;
            }
            const entry = { modal: true, opener, ...layer };
            const under = entry.page ? stack.findIndex((open) => !open.page) : -1;
            if (under < 0) stack.push(entry);
            else stack.splice(under, 0, entry);
            change.pushed.add(entry.id);
            return true;
        });
    }

    /** The owner closed a layer itself (state already updated): drop it, restore focus, reconcile history. */
    function remove(id) {
        const index = stack.findIndex((layer) => layer.id === id);
        if (index < 0) return false;
        return batch(() => {
            const [layer] = stack.splice(index, 1);
            if (!change.pushed.delete(layer.id)) change.removed.push(layer);
            return true;
        });
    }

    /** THE dismiss: exactly one layer, the top one, through its owner's onClose. */
    function pop(reason = 'close') {
        const layer = top();
        if (!layer) return false;
        try { layer.onClose?.(reason); } finally {
            if (has(layer.id)) remove(layer.id);  // an owner that did not call remove
        }
        return true;
    }

    /** Back (system, browser, Telegram): a dialog the page opened over the layers is on top, so it takes the press
     * as its Escape (the dialogs close on Escape through ui_interactions.bindDialogFocus; a bare native dialog that
     * nothing answered closes as its own Escape would close it) and the layer stays; otherwise the top layer closes. */
    function back(reason = 'history') {
        const dialogs = [...(doc?.querySelectorAll?.('dialog[open], [aria-modal="true"]') || [])]
            .filter((el) => el.getClientRects?.().length && !stack.some((layer) => layer.el === el));
        const dialog = dialogs[dialogs.length - 1];
        if (!dialog || !win?.KeyboardEvent) return pop(reason);
        const target = dialog.contains(doc.activeElement) ? doc.activeElement : dialog;
        const escape = new win.KeyboardEvent('keydown', { key: 'Escape', bubbles: true, cancelable: true });
        target.dispatchEvent(escape);
        if (!escape.defaultPrevented && dialog.tagName === 'DIALOG') dialog.close?.();
        reconcileHistory();
        return true;
    }

    function onChange(fn) {
        listeners.add(fn);
        return () => listeners.delete(fn);
    }

    win?.addEventListener?.('popstate', onPopState);
    win?.addEventListener?.('keydown', onKeydown);
    win?.addEventListener?.('keydown', onTab, true);

    function destroy() {
        disposed = true;
        epoch += 1;  // a pending inert difference applies nothing
        win?.removeEventListener?.('popstate', onPopState);
        win?.removeEventListener?.('keydown', onKeydown);
        win?.removeEventListener?.('keydown', onTab, true);
        pendingInert = null;
        owed = null;
        for (const el of inerted) el.inert = false;
        inerted.clear();
        listeners.clear();
        stack.length = 0;
    }

    return {
        push, remove, batch, pop, back, onChange, destroy,
        top: () => top()?.id || null,
        has,
        size: () => stack.length,
        ids: () => stack.map((layer) => layer.id),
        /** The history sentinel state, for tests and the host adapter. */
        historyArmed: () => hist.armed,
    };
}
