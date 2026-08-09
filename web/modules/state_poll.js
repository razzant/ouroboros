/**
 * The single `/api/state` poll owner's PURE core: subscriber fan-out, single-flight
 * coalescing, and the cadence decision.
 *
 * Before consolidation there were TWO timers — app.js polling every 20s for the
 * projects nav and every chat instance polling every 3s for its header controls — so
 * an open project panel multiplied the request rate. The rules that replaced them are
 * the part worth testing, and none of them need a DOM:
 *
 *   • ONE read is ever in flight. Concurrent callers get the SAME promise, so a
 *     startup barrier, a `projects_changed` refresh and a timer tick that land
 *     together are one request, not three.
 *   • Every consumer reads the same snapshot through `subscribe`, and a LATE
 *     subscriber (a project panel created after startup) is replayed the last
 *     snapshot immediately rather than waiting a full interval.
 *   • The cadence is ~3s on Chat (live budget/mode feedback), ~20s elsewhere, and
 *     the timer is PAUSED while the document is hidden — paused, not backed off, so
 *     a hidden tab spends nothing.
 *   • The timer re-arms when a read SETTLES, not when it starts, so a slow response
 *     can never stack requests behind itself.
 *
 * Everything impure is injected: `read` performs the fetch and returns the snapshot
 * to publish, `activePage`/`hidden` are read at decision time (never cached), and the
 * timer functions are passed in so tests drive them directly. This module contains no
 * fetch, no document and no window reference.
 *
 * A subscriber that throws is contained: it must not stop the other consumers from
 * seeing the snapshot, and it must not break the timer that keeps the app live.
 */

export const STATE_POLL_CHAT_MS = 3000;
export const STATE_POLL_IDLE_MS = 20000;

/** Cadence for one page: the Chat page is the only one that needs live feedback. */
export function statePollIntervalMs(activePage) {
    return activePage === 'chat' ? STATE_POLL_CHAT_MS : STATE_POLL_IDLE_MS;
}

/**
 * @param {{
 *   read: () => Promise<any>,
 *   activePage: () => string,
 *   hidden: () => boolean,
 *   setTimer: (fn: () => void, ms: number) => any,
 *   clearTimer: (handle: any) => void,
 * }} deps
 */
export function createStatePoll({ read, activePage, hidden, setTimer, clearTimer }) {
    const subscribers = [];
    let lastSnapshot = null;
    let hasSnapshot = false;
    let timer = 0;
    let inFlight = null;

    function publish(data) {
        lastSnapshot = data;
        hasSnapshot = true;
        // A copy, because a handler may unsubscribe itself (or another) mid-fan-out.
        for (const handler of subscribers.slice()) {
            try { handler(data); } catch { /* one bad consumer cannot starve the rest */ }
        }
    }

    function subscribe(handler) {
        if (typeof handler !== 'function') return () => {};
        subscribers.push(handler);
        if (hasSnapshot) {
            try { handler(lastSnapshot); } catch { /* see publish */ }
        }
        return () => {
            const idx = subscribers.indexOf(handler);
            if (idx >= 0) subscribers.splice(idx, 1);
        };
    }

    function stop() {
        clearTimer(timer);
        timer = 0;
    }

    function schedule() {
        stop();
        if (hidden()) return;  // paused, not backed off: no hidden-tab spend
        timer = setTimer(() => { refresh(); }, statePollIntervalMs(activePage()));
    }

    function refresh() {
        // The coalescing guard: a second caller joins the first read instead of
        // starting one, and both settle on the same snapshot.
        if (inFlight) return inFlight;
        inFlight = (async () => {
            try {
                publish(await read());
            } finally {
                inFlight = null;
                // Re-armed on SETTLE (including a rejected read), so a slow or
                // failing response never stacks requests behind itself.
                schedule();
            }
        })();
        return inFlight;
    }

    return { subscribe, refresh, schedule, stop, intervalMs: () => statePollIntervalMs(activePage()) };
}
