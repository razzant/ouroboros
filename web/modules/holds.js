// The start ledger (docs/DESIGN.md "Skills in the page", "Living cards", "Widgets board"). Code the core did not
// write runs in the owner's page: a skill's page module, a skill's module or route frame, an app card Ouroboros
// wrote. On WebKit, Android WebView and Firefox same-site frames it shares the page's one thread, so a start that
// loops or blows up takes the whole page with it, and the next open would run it again. Before the host hands such
// code its first turn it writes the item's key here; it removes the key once the start settled and the host's own
// timer ran a second after it (`starting` → `settle`), or once the page hides: a page whose thread runs its
// `pagehide` outlived what it started. A key the next page finds is a hold: the host does not start
// that item by itself; it rests behind its own Open / Start / Start it with one line saying why, until the owner
// presses it or the item's code changes (a new revision). The same record keeps the owner's Stop on a Widgets card
// and Close on a card in a message, so they survive a reload on this device. The page that opens with `?hold=all`
// holds every item for that load. Read once per page: this page's own starts in flight are never its holds.

import { isTelegramHost } from './host_telegram.js';

export const HOLDS_KEY = 'ouroboros.holds';
export const START_SETTLE_MS = 1000;
/** The one line a held item shows beside its own Open / Start, by the ledger's reason. */
export const HELD_LINES = Object.freeze({
    start: 'Held back: the page stopped while this was starting last time.',
    all: 'Held back: this page was opened with everything held.',
});
const HOLD_ALL_PARAM = 'hold';
const PROBE_KEY = 'ouroboros.holds.probe';
const MAX_ENTRIES = 64;
const RETIRED_KEYS = ['ouroboros.hostModules.boot'];  // the page-module boot marker this ledger replaced

/** localStorage, or sessionStorage where local storage refuses a write; null when neither takes one. */
function pickStore(win) {
    for (const name of ['localStorage', 'sessionStorage']) {
        try {
            const store = win?.[name];
            if (!store) continue;
            store.setItem(PROBE_KEY, '1');
            store.removeItem(PROBE_KEY);
            return store;
        } catch { /* storage blocked or full: try the next one */ }
    }
    return null;
}

/** `?hold=all` in the page address: every item waits for its own control on this load. */
export function holdAllRequested(href) {
    try { return new URL(href).searchParams.get(HOLD_ALL_PARAM) === 'all'; } catch { return false; }
}

/** One load: the flag leaves the address once read, so a reload or the reconnect refresh starts as usual. */
function forgetHoldAll(win) {
    try {
        const url = new URL(win.location.href);
        url.searchParams.delete(HOLD_ALL_PARAM);
        win.history?.replaceState?.(win.history.state, '', url.toString());
    } catch { /* the address keeps it: the next load holds everything again */ }
}

/**
 * The ledger over one window's storage. `reason(key, rev)` says why the host must not start an item by itself:
 * `'owner'` (Stop / Close), `'start'` (its last start with this revision did not settle), `'all'` (`?hold=all`)
 * or `''`. `starting(key, rev)` records a start and returns its `settle()`; `stop(key)` and `release(key)` are the
 * owner's presses; `found()` lists the start holds this page found that no page reported yet (each `{key, rev}`), and
 * `reported(found)` marks them reported, so a hold the owner leaves in place is named once, not after every restart.
 */
export function createHolds({ win = globalThis.window, now = () => Date.now() } = {}) {
    const store = pickStore(win);
    const read = () => {
        try {
            const parsed = JSON.parse(store?.getItem(HOLDS_KEY) || 'null');
            return parsed && typeof parsed.entries === 'object' && parsed.entries ? parsed.entries : {};
        } catch { return {}; }
    };
    const write = (entries) => {
        // Each reason keeps its newest MAX_ENTRIES: a start passing through never pushes out an owner's Stop / Close.
        const byAge = Object.entries(entries).sort((a, b) => (Number(b[1]?.at) || 0) - (Number(a[1]?.at) || 0));
        const kept = ['owner', 'start'].flatMap((reason) => byAge.filter(([, entry]) => entry?.reason === reason)
            .slice(0, MAX_ENTRIES));
        try { store?.setItem(HOLDS_KEY, JSON.stringify({ v: 1, entries: Object.fromEntries(kept) })); } catch { /* full or refused */ }
    };
    const update = (change) => { const entries = read(); change(entries); write(entries); };
    for (const key of RETIRED_KEYS) {
        try { store?.removeItem(key); } catch { /* blocked */ }
    }
    const initial = read();
    const prior = new Map();
    const unreported = new Set();
    const owners = new Set();
    for (const [key, entry] of Object.entries(initial)) {
        if (entry?.reason === 'start') {
            prior.set(key, String(entry.rev ?? ''));
            if (!entry.reported) unreported.add(key);
        } else if (entry?.reason === 'owner') owners.add(key);
    }
    const holdAll = holdAllRequested(String(win?.location?.href || ''));
    if (holdAll) forgetHoldAll(win);
    const released = new Set();
    const pending = new Set();
    let starts = 0;
    win?.addEventListener?.('pagehide', () => { for (const settle of [...pending]) settle(); });
    return {
        reason(key, rev = '') {
            if (owners.has(key)) return 'owner';
            if (released.has(key)) return '';
            if (prior.has(key) && prior.get(key) === String(rev ?? '')) return 'start';
            return holdAll ? 'all' : '';
        },
        starting(key, rev = '') {
            starts += 1;
            const entry = { reason: 'start', rev: String(rev ?? ''), at: now(), n: starts };
            update((entries) => { if (entries[key]?.reason !== 'owner') entries[key] = entry; });
            let settled = false;
            const settle = () => {
                if (settled) return;
                settled = true;
                pending.delete(settle);
                update((entries) => {
                    const held = entries[key];
                    if (held?.reason === 'start' && held.at === entry.at && held.n === entry.n && held.rev === entry.rev) {
                        delete entries[key];
                    }
                });
            };
            pending.add(settle);
            return settle;
        },
        stop(key) {
            owners.add(key);
            update((entries) => { entries[key] = { reason: 'owner', at: now() }; });
        },
        release(key) {
            released.add(key);
            const wasOwner = owners.delete(key);
            update((entries) => {
                if (wasOwner || entries[key]?.reason === 'owner' || prior.has(key)) delete entries[key];
            });
        },
        found: () => [...prior].filter(([key]) => unreported.has(key) && !released.has(key)).map(([key, rev]) => ({ key, rev })),
        reported(items) {
            update((entries) => {
                for (const { key, rev } of items) {
                    unreported.delete(key);
                    if (entries[key]?.reason === 'start' && entries[key].rev === rev) entries[key] = { ...entries[key], reported: true };
                }
            });
        },
    };
}

/** Calls `settle` one host second after the first `load` of the frame `mount` holds now; returns the canceller. */
export function settleAfterFrameLoad(mount, settle, { delay = START_SETTLE_MS } = {}) {
    const frame = mount?.querySelector?.('iframe');
    if (!frame) { settle(); return () => {}; }
    let timer = 0;
    const onLoad = () => { timer = setTimeout(settle, delay); };
    frame.addEventListener('load', onLoad, { once: true });
    return () => { frame.removeEventListener('load', onLoad); clearTimeout(timer); };
}

/**
 * Tell the host once which starts this page did not outlive, with the raw facts of where it runs, so the chat that
 * owns each item names it (`POST /api/ui/holds`). The desktop shell's bridge arrives after load: wait for it briefly.
 */
export function reportFoundHolds(holds, post, { win = globalThis.window, wait = 3000 } = {}) {
    const found = holds.found();
    if (!found.length) return;
    let sent = false;
    const send = () => {
        if (sent) return;
        sent = true;
        const surface = { pywebview: Boolean(win?.pywebview?.api), telegram: isTelegramHost({ win, doc: win?.document }) };
        Promise.resolve().then(() => post({ holds: found, surface })).then(() => holds.reported(found))
            .catch(() => { /* not named: the next page reports it again; a lost report holds nothing back */ });
    };
    if (win?.pywebview?.api) { send(); return; }
    win?.addEventListener?.('pywebviewready', send, { once: true });
    setTimeout(send, wait);
}

let shared = null;

/** The page's one ledger, read when first asked. */
export function getHolds() {
    shared ||= createHolds();
    return shared;
}
