import { haptic } from './haptics.js';
import { tr } from './i18n.js';
import { SHELL_ICONS } from './page_icons.js';
import { isModalDialog, modalSurfaces, normalizeTone } from './ui_helpers.js';

const stackObservers = new WeakMap();

// While something blocks the page (ui_helpers `modalSurfaces`: a modal dialog, a portal that is `aria-modal`) a notice
// outside it is painted under its backdrop or scrim, where a press at its × lands on them (it cancels a question, it
// closes a sheet), and it would run out unread. Such a stack is held (hidden, no time running out) and shown, its time
// started again, once nothing is above it. One watch, only while a notice is mounted: the attributes that make a surface
// (`open`, `aria-modal`) anywhere, and what enters or leaves <body>, where every portal mounts: the feed's own rows are
// never watched. A portal removed without a close says nothing else: the notices it hosted leave with it.
let modalWatch = null;

function syncHeld() {
    const doc = globalThis.document;
    const stacks = doc?.querySelectorAll?.('.toast-stack');
    if (!stacks) return;
    // Only the top surface's own notices may show. A modal dialog is the top layer, over every portal; of several, the one
    // holding focus (nothing under a modal can take it), and if none does, every stack waits: a guess could bare a
    // lower surface's ×.
    // ui_helpers `modalSurfaces` owns what actively blocks: an open modal dialog, a visible `aria-modal` portal. A
    // portal that stays hidden blocks nothing and must not hold the page; one concealed only while it asks a
    // confirmation returns to `modalSurfaces` the moment `hidden` lifts, which the watch below syncs before a paint.
    const surfaces = modalSurfaces(doc);
    const dialogs = surfaces.filter((el) => el.localName === 'dialog');
    const layer = dialogs.length ? dialogs : surfaces;
    const top = layer.length > 1 ? layer.find((el) => el.contains(doc.activeElement)) : layer[0];
    let mounted = false;
    for (const stack of stacks) {
        const hold = surfaces.length > 0 && !top?.contains(stack);
        if (stack.firstElementChild) mounted = true;
        if (stack.hasAttribute?.('data-held') === hold) continue;
        stack.toggleAttribute?.('data-held', hold);
        for (const toast of stack.children || []) toast.rearm?.();
    }
    if (!mounted || typeof MutationObserver !== 'function') {
        modalWatch?.disconnect();
        modalWatch = null;
        return;
    }
    if (modalWatch) return;
    modalWatch = new MutationObserver((records) => {
        let sync = false;
        for (const record of records) {
            for (const node of record.removedNodes) if (!node.isConnected) disposeNotices(node);
            if (record.type === 'childList') { sync = true; continue; }
            if (record.attributeName !== 'hidden') { sync = true; continue; }
            const target = record.target;
            const blocker = '[aria-modal="true"],dialog[open]';
            // A `hidden` change matters on the blocker itself or a box containing it, not on a child merely inside it.
            if (target?.matches?.(blocker) || target?.querySelector?.(blocker)) sync = true;
        }
        if (sync) syncHeld();
    });
    modalWatch.observe(doc.documentElement, { subtree: true, attributes: true, attributeFilter: ['open', 'aria-modal', 'hidden'] });
    modalWatch.observe(doc.body, { childList: true });
}

const openings = new WeakMap();  // a modal dialog -> whether its current opening has ended

// One watch per opening, whatever is asked in it, ended by the first change of `open` (a close, even if reopened at once).
// A portal is made for one opening and removed with it: it has ended once it leaves the page.
function openingEnded(surface) {
    if (surface.localName !== 'dialog') return () => !surface.isConnected;
    const known = openings.get(surface);
    if (known && !known()) return known;
    let over = false;
    let watch = null;
    const end = () => { over = true; watch?.disconnect(); };
    if (typeof MutationObserver === 'function') {
        watch = new MutationObserver(end);
        watch.observe(surface, { attributes: true, attributeFilter: ['open'] });
    }
    const ended = () => {
        if (!over && (watch?.takeRecords().length || !surface.open || surface.isConnected === false)) end();
        return over;
    };
    openings.set(surface, ended);
    return ended;
}

/** What an action at `origin` answers through. A press inside a blocking surface is answered in it (the page behind it is
 * inert, and its notices are held) for as long as that same opening stands; after it, or from outside one, through the
 * page's `page`. The origin is named when the press is made, never found again after an await. */
export function modalNotice(origin, page = showToast) {
    const surface = origin?.closest?.('dialog[open], [aria-modal="true"]');
    if (!surface || (surface.localName === 'dialog' && !isModalDialog(surface))) return page;
    const ended = openingEnded(surface);
    return (message, tone, options = {}) => (ended()
        ? page(message, tone, options)
        : showToast(message, tone, { ...options, host: surface }));
}

/** Dispose the notices mounted in `host` (its modal closing): their timers and listeners leave with them. */
export function disposeNotices(host) {
    host?.querySelectorAll?.('.toast').forEach((toast) => toast.dispose?.());
}

function getStack(host = null) {
    let stack = host ? host.querySelector(':scope > .toast-stack--room') : document.getElementById('toast-stack');
    if (!stack) {
        stack = document.createElement('div');
        if (!host) stack.id = 'toast-stack';
        stack.className = host ? 'toast-stack toast-stack--room' : 'toast-stack';
        stack.setAttribute('aria-live', 'polite');
        stack.setAttribute('aria-relevant', 'additions');
        (host || document.body).appendChild(stack);
    }
    // A window that grows or shrinks cuts the column or releases it with no notice coming or going; as it
    // scrolls, its edge moves over the words. A kept notice that grows or rewraps inside a capped column
    // shifts every notice below it while the stack's own box stays fixed, so the same observer watches each
    // global notice too.
    if (!host && !stackObservers.has(stack) && typeof ResizeObserver === 'function') {
        const observer = new ResizeObserver(() => markCut(stack));
        observer.observe(stack);
        stackObservers.set(stack, observer);
        stack.addEventListener('scroll', () => markCut(stack), { passive: true });
    }
    return stack;
}

const edged = new WeakSet();  // the columns whose words wear an edge now; one that fits has nothing to read or clear

// Words the column's edge cuts fade there, as any scroller's content does (docs/DESIGN.md principle 9, the shared
// `.scroll-fade-y` recipe in ui.css), and only they: the glass, the sticky close, its ring and the shadow stay whole.
// What counts is a word actually hidden, never the padding that holds the last shadow. The shared mask is laid over
// the column's window (style.css), so it ends at the screen's edge, not where the words end out of sight; the window
// is read from layout places (a toast arriving or gliding is displaced by its own transform).
function fadeCutWords(stack, scrolls) {
    if (!scrolls && !edged.has(stack)) return;
    const from = stack.scrollTop;
    const to = from + stack.clientHeight;
    let worn = false;
    for (const toast of stack.children || []) {
        const words = toast.firstElementChild;
        if (!words?.classList?.contains('toast-words') || toast.classList.contains('is-hiding')) continue;
        const box = words.getBoundingClientRect();
        const top = toast.offsetTop + box.top - toast.getBoundingClientRect().top;
        const bottom = top + box.height;
        const above = scrolls && top < from - 1 && bottom > from;
        const below = scrolls && bottom > to + 1 && top < to;
        words.toggleAttribute('data-scroll-above', above);
        words.toggleAttribute('data-scroll-below', below);
        if (above || below) {
            words.style.setProperty('--toast-view-top', `${from - top}px`);
            words.style.setProperty('--toast-view-height', `${to - from}px`);
            worn = true;
        } else {
            words.style.removeProperty('--toast-view-top');
            words.style.removeProperty('--toast-view-height');
        }
    }
    if (worn) edged.add(stack); else edged.delete(stack);
}

// The page's stack ends above the screen's bottom and scrolls past it (style.css); cut there, it is reached by
// keyboard too, as a wide table is (docs/DESIGN.md). Measured when a notice comes or goes, when the column's box
// changes size and while it scrolls; room stacks keep their place.
function markCut(stack) {
    if (stack.id !== 'toast-stack') return;
    const cut = stack.scrollHeight > stack.clientHeight;
    fadeCutWords(stack, cut);
    if (cut === stack.hasAttribute('tabindex')) return;
    if (cut) {
        stack.tabIndex = 0;
        stack.setAttribute('role', 'region');
        stack.setAttribute('aria-label', 'Scrollable notices');
    } else {
        ['tabindex', 'role', 'aria-label'].forEach((name) => stack.removeAttribute(name));
    }
}

/**
 * One toast (docs/DESIGN.md §4, owner 26.11 «A»): its words are only words, read and selected, and a press on them does
 * nothing; its close, on every toast, timed or kept, only hides it. `ttl` 0 keeps it until then. `action` ({label, run})
 * adds one button that dismisses the toast and runs: a notification's way to its source (notifications.js), a recovery
 * notice's way back (host_modules.js). `dismissLabel` names the close. Hover or focus pauses the auto-dismiss timer.
 * An error toast answers the owner's own action and plays Telegram's error haptic (haptics.js); a notice the owner did
 * not cause passes `{ haptic: false }`.
 * Optional `host` confines the stack to a room (above its composer) or to a blocking surface, whose page behind is inert.
 * Its owner calls the returned node's `dispose()` on teardown.
 */
export function showToast(message, tone = 'info', { ttl = 6000, action = null, haptic: felt = true, dismissLabel = '', host = null } = {}) {
    const stack = getStack(host);
    const observer = host ? null : stackObservers.get(stack) || null;
    const toast = document.createElement('div');
    const cleanTone = normalizeTone(tone || 'info', 'info');
    if (felt && cleanTone === 'danger') haptic('error');
    toast.className = `toast toast-${cleanTone}`;
    toast.setAttribute('role', cleanTone === 'danger' ? 'alert' : 'status');
    // The words keep their own box beside the controls; the page column's shared edge masks only it.
    const words = document.createElement('span');
    words.className = 'toast-words scroll-fade-y';
    toast.append(words);
    words.textContent = message || '';
    let timer = 0;
    let removalTimer = 0;
    let hovered = false;
    let focused = false;
    let observed = false;
    const clear = () => { if (timer) { clearTimeout(timer); timer = 0; } };
    const dispose = () => {
        clear();
        clearTimeout(removalTimer);
        globalThis.window?.removeEventListener?.('pagehide', dispose);
        if (observed) {
            observed = false;
            observer.unobserve(toast);
        }
        toast.remove();
        syncHeld();
        markCut(stack);
    };
    toast.dispose = dispose;
    const dismiss = () => {
        clear();
        globalThis.window?.removeEventListener?.('pagehide', dispose);
        if (toast.classList.contains('is-hiding')) return;
        // A control leaves with its notice: the surface hosting it keeps the focus, and with it its trap (a Tab from the
        // page's body would reach what lies under a sheet).
        if (host && toast.contains(globalThis.document?.activeElement)) host.focus?.({ preventScroll: true });
        // It fades where it stood, out of the column, while the toasts that stay glide into its place at once: the
        // shift is measured here, the glide is the stylesheet's (none with reduced motion).
        const stay = [...(stack.children || [])].filter((el) => el !== toast);
        const before = stay.map((el) => el.offsetTop);
        Object.assign(toast.style, { position: 'absolute', top: `${toast.offsetTop}px`, left: `${toast.offsetLeft}px`, width: `${toast.offsetWidth}px` });
        toast.classList.add('is-hiding');
        const moved = stay.filter((el, i) => {
            const shift = before[i] - el.offsetTop;
            if (shift) { el.style.setProperty('--toast-shift', `${shift}px`); el.setAttribute('data-gliding', ''); }
            return shift !== 0;
        });
        void stack.offsetHeight;  // the shifted places are where the glide starts
        moved.forEach((el) => el.removeAttribute('data-gliding'));
        removalTimer = setTimeout(dispose, 180);
    };
    // A held notice has no time running: `rearm` starts it when the notice can be seen.
    const arm = () => {
        clear();
        if (ttl > 0 && !hovered && !focused && !toast.classList?.contains?.('is-hiding') && !stack.hasAttribute?.('data-held')) {
            timer = setTimeout(dismiss, ttl);
        }
    };
    if (action?.label && typeof action.run === 'function') {
        const button = document.createElement('button');
        button.type = 'button';
        button.className = 'btn btn-sm btn-default toast-action';
        button.textContent = action.label;
        // The notice leaves over 180 ms and its button takes another Enter meanwhile: one notice, one act.
        button.addEventListener('click', (event) => {
            event.stopPropagation();
            dismiss();
            action.run();
        }, { once: true });
        toast.append(button);
    }
    // Every notice has its own visible, keyboard-reachable way out: the quiet icon control (the shared small capsule
    // without a body, its reach a finger's minimum under a coarse pointer). It hides the notice and nothing else: it
    // answers no question and runs no action.
    const close = document.createElement('button');
    close.type = 'button';
    close.className = 'btn btn-ghost btn-sm toast-dismiss';
    close.setAttribute('aria-label', dismissLabel || tr('toast.dismiss', 'Dismiss notice'));
    close.innerHTML = SHELL_ICONS.close;
    close.addEventListener('click', (event) => {
        event.stopPropagation();
        dismiss();
    });
    toast.append(close);
    stack.appendChild(toast);
    if (observer) {
        observer.observe(toast);
        observed = true;
    }
    syncHeld();
    markCut(stack);
    if (ttl > 0) {
        toast.addEventListener('pointerenter', () => { hovered = true; clear(); });
        toast.addEventListener('pointerleave', () => { hovered = false; arm(); });
        toast.addEventListener('focusin', () => { focused = true; clear(); });
        toast.addEventListener('focusout', (event) => { focused = Boolean(toast.contains?.(event.relatedTarget)); arm(); });
        globalThis.window?.addEventListener?.('pagehide', dispose, { once: true });
        // Held or released: a hover or focus from before the modal says nothing about the notice now.
        toast.rearm = () => { hovered = false; focused = false; arm(); };
        arm();
    }
    return toast;
}
