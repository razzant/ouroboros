import { isReplayEvidenceRow } from './chat_activity.js';
import { historyRowIds } from './chat_history_replay.js';

/** Mixed cards have no trustworthy physical boundary. Their navigation stays
 * on the common keyboard-accessible button, never a timestamp separator. */
export function historyIslandAtEdge(feed, ids, historyNodes) {
    const roots = [...ids].flatMap(historyNodes).map(node => node.closest('.chat-live-card') || node);
    const unambiguous = roots.length && roots.every(root =>
        [...root.querySelectorAll('[data-history-id]')].every(node => ids.has(node.dataset.historyId)));
    const bottom = Math.max(...roots.map(node => node.getBoundingClientRect().bottom));
    const rect = feed.getBoundingClientRect();
    return unambiguous && bottom >= rect.top && bottom <= rect.bottom + 80;
}

/** A span keeps its byte coordinates while the newest read's chain (`head`) lists
 * its own last witness: rotation appends one, a replaced or removed earlier
 * segment changes all. */
export function sameHistoryChain(head, span) {
    return typeof span?.chain === 'string' && String(head?.chain).split('.').includes(span.chain.split('.').at(-1));
}

/** Coverage is byte delivery, never chronology, shared row identity or EOF.
 * The recent read supplies the horizon. A clean overlapping re-read can heal
 * a failed span; evicted bodies cannot certify bytes. Completeness and gaps are
 * the conversation's (the chat stream): narration (progress) pages alongside it
 * but never decides coverage; an unreadable or failed narration span on the
 * recent read is still a gap.
 */
export function historyCoverage(recent, pages = []) {
    const horizon = recent?.upper?.chat, head = recent?.spans?.chat, narration = recent?.spans?.progress;
    if (recent?.v !== 1 || !Number.isSafeInteger(horizon) || !head) return { complete: false, gaps: true, horizonGap: false };
    let gaps = pages.some(value => value?.v !== 1 || value.view !== recent.view)
        || (narration !== null && !(Array.isArray(narration?.gaps) && !narration.gaps.length));
    const delivered = [recent, ...pages].filter(value => value?.v === 1 && value.view === recent.view)
        .map(value => value.spans?.chat);
    // The newest read lists prefix witnesses through its trailing segments.
    const sameChain = span => sameHistoryChain(head, span);
    const valid = span => span && sameChain(span) && Array.isArray(span.gaps)
            && Number.isSafeInteger(span.from) && Number.isSafeInteger(span.to)
            && span.from >= 0 && span.from <= span.to;
    // A formerly empty source owns no retained bytes. Its first write
    // changes the chain witness without invalidating an older fragment,
    // and its frontier stays zero: a first read starting above it is a gap.
    const knownEmpty = span => span?.chain === 'empty' && span.from === 0 && span.to === 0
        && Array.isArray(span.gaps) && !span.gaps.length;
    const known = span => valid(span) || knownEmpty(span);
    if (delivered.some(span => !known(span))) gaps = true;
    const spans = delivered.filter(span => known(span) && !span.gaps.length)
        .sort((a, b) => a.from - b.from);
    const clean = [];
    let end = null, start = null;
    for (const span of spans) {
        if (span.from > horizon) continue;
        if (end !== null && span.from > end) gaps = true;
        if (start === null) start = span.from;
        if (!clean.length || span.from > clean.at(-1).to) clean.push({ ...span });
        else clean.at(-1).to = Math.max(clean.at(-1).to, span.to);
        end = Math.max(end ?? 0, Math.min(horizon, span.to));
    }
    // A failed read remains visible until a compatible clean read actually
    // covers its bytes; simply dropping it cannot certify the fragment.
    if (delivered.some(span => valid(span) && span.gaps.length
        && !clean.some(range => range.from <= span.from && range.to >= span.to))) gaps = true;
    if (end !== horizon) gaps = true;
    const oldEnds = pages.filter(value => value?.view === recent.view)
        .map(value => value?.spans?.chat).filter(span => sameChain(span) || knownEmpty(span))
        .map(span => span.to);
    const horizonGap = oldEnds.length > 0 && head.from > Math.max(...oldEnds);
    return { complete: start === 0 && end === horizon && !gaps, gaps, horizonGap };
}

/** Chat's bounded archive-page owner. DOM, reading protection and live rows stay
 * with the chat instance; only fetchPage is asynchronous. Cursors are opaque.
 * It also owns the open window's `after_cursor` (CLIENT_CONTRACT §5 "Reading
 * history"): `read()` takes only the rows stored since the newest read and hands
 * them to `applyArrivals`, the keyed row path the newest page takes; it leaves the
 * newest page to chat's own read (`readNewest`) whenever arrivals cannot answer.
 */
export function createChatHistoryPager({
    fetchPage,
    applyPage,
    releasePage,
    applyArrivals = () => {},
    readNewest = async () => false,
    canArrive = () => false,  // a hydrated, healthy window that needs no rebuild
    newestInFlight = () => null,
    isPageProtected = () => false,
    onState = () => {},
    isAlive = () => true,
    maxPages = 3,
}) {
    const limit = Number.isInteger(maxPages) && maxPages > 0 ? maxPages : 3;
    const cache = new Map();
    let pages = [];
    const pageRows = new Map();  // page id -> the history rows it holds (a return reads its page again)
    let chain = 0;
    let jumped = false;  // the chain began at a navigation (a page around a task, a page left), not the present
    let focus = 0;
    let pending = null;
    let failure = null;
    let destroyed = false;
    // Where the next read of new arrivals starts, the newest read's row count and the
    // rows read since; the read in flight and at most one queued behind it.
    let after = null, windowRows = 0, arrived = 0, arrival = null, queued = null, landedArrivals = 0;
    const alive = () => !destroyed && isAlive();
    const rowCount = rows => (Array.isArray(rows) ? rows.filter(row => !isReplayEvidenceRow(row)).length : 0);
    // Every newest read (the page, ↓, a recent read applied by chat) restarts the arrivals.
    function newest(data) {
        after = typeof data?.after_cursor === 'string' && data.after_cursor ? data.after_cursor : null;
        windowRows = rowCount(data?.messages);
        arrived = 0;
    }

    // Protected pages can remain as islands while the ordinary reading window
    // moves. Navigation follows the contiguous window containing the last load.
    // A landed page that contributed no rows owns nothing to mount or restore, so
    // the window steps over it: it is a boundary the reader already crossed, not a
    // gap. Only a NON-EMPTY page missing from the cache is a real hole above/below.
    const held = page => cache.has(page.id) || (page.rows === 0 && page.loaded);
    function bounds() {
        if (!pages.length) return { first: 0, last: 0 };
        let first = focus, last = focus;
        while (first > 0 && held(pages[first - 1])) first -= 1;
        while (last + 1 < pages.length && held(pages[last + 1])) last += 1;
        return { first, last };
    }

    function getState() {
        const { first, last } = bounds();
        const initialized = pages.length > 0;
        const canOlder = initialized && (last + 1 < pages.length || pages[last].hasMore);
        const canNewer = initialized && first > 0;
        return {
            initialized, destroyed,
            loading: pending?.direction || '',
            error: failure?.error || null,
            retryDirection: failure?.direction || '',
            retryCursor: failure?.cursor ?? null,
            canOlder, canNewer,
            // The chain does not show the present: newer pages of it, or a navigation began it.
            away: initialized && (first > 0 || jumped),
            olderExhausted: initialized && !canOlder,
            newerExhausted: initialized && !canNewer,
            pageCount: pages.length,
            firstPage: initialized ? pages[first] : null,
            lastPage: initialized ? pages[last] : null,
            cachedPages: [...cache.values()].map(entry => entry.page),
            coverage: [...new Map([...cache.values()].map(entry => [entry.page.id, entry.page])
                .concat(pages.filter(page => page.rows === 0 && page.loaded).map(page => [page.id, page])))
                .values()].map(page => page.coverage),
        };
    }
    const publish = () => { if (alive()) onState(getState()); };

    function release(entry) {
        cache.delete(entry.page.id);
        releasePage(entry.page);
    }

    function prune() {
        const released = [];
        for (const entry of cache.values()) {
            if (entry.page.chain !== chain && !isPageProtected(entry.page)) {
                released.push(entry.page.id);
                release(entry);
            }
        }
        // An empty page mounts nothing: its descriptor alone holds the window (see
        // `held`), so its cache entry is released at once and never spends or pays
        // the budget. Only pages that contributed rows count toward the limit.
        for (const entry of [...cache.values()]) {
            if (entry.page.rows === 0 && entry.page.index !== focus) {
                released.push(entry.page.id);
                release(entry);
            }
        }
        const mounted = [...cache.values()].filter(entry => entry.page.rows !== 0);
        let size = mounted.length;
        const candidates = mounted
            .filter(entry => entry.page.chain === chain && entry.page.index !== focus)
            .sort((a, b) => Math.abs(b.page.index - focus) - Math.abs(a.page.index - focus));
        for (const entry of candidates) {
            if (size <= limit) break;
            if (isPageProtected(entry.page)) continue;
            released.push(entry.page.id);
            release(entry);
            size -= 1;
        }
        return released;
    }

    function sourceReadError(data) {
        if (!data?.page_cursor && data?.reason_code === 'history_source_unavailable') {
            const error = new Error(data.error || 'Some saved history is unavailable. Retry loading messages.');
            error.body = data;
            return error;
        }
        return null;
    }

    function land(data, { index, direction, newChain, stillWanted }, extra = {}) {
        const error = sourceReadError(data);
        if (error) throw error;
        // A read around one place that found nowhere to land answers with its anchor and no page.
        if (data?.anchor && data.anchor.state !== 'found') return { status: 'anchor', anchor: data.anchor };
        // A navigation the reader has moved on from never replaces the chain they are reading.
        if (stillWanted && !stillWanted()) return { status: 'superseded' };
        if (!Array.isArray(data?.messages) || typeof data.has_more !== 'boolean'
            || data.page_cursor == null || (data.has_more && data.next_cursor == null)) {
            throw new TypeError('History page is missing its messages or continuation boundary');
        }
        const nextChain = newChain ? chain + 1 : chain;
        const messageCount = rowCount(data.messages);
        // Re-reading a page refreshes its row count, never its frozen boundaries.
        const prior = newChain ? null : pages[index];
        const baseId = `history-page-${nextChain}-${index}`;
        let id = baseId, suffix = 0;
        while (!prior && (pages.some(page => page.id === id) || cache.has(id))) {
            id = `${baseId}-${++suffix}`;
        }
        const page = Object.freeze(prior
            ? { ...prior, rows: messageCount, loaded: true, coverage: data.coverage ?? null }
            : { id, chain: nextChain, index,
                requestCursor: data.page_cursor, nextCursor: data.next_cursor ?? null,
                hasMore: data.has_more, rows: messageCount, loaded: true, coverage: data.coverage ?? null });
        applyPage(data.messages, { ...page, direction, window: data.window ?? null, recentVersion: data.recentVersion, ...extra });
        if (!alive()) return { status: 'disposed' };
        // Only a read of the present restarts the arrivals; a page around one place is not the present.
        if (newChain) {
            chain = nextChain; pages = [page]; pageRows.clear(); failure = null; jumped = Boolean(stillWanted);
            if (direction === 'latest') newest(data);
        }
        else pages[index] = page;
        pageRows.set(page.id, new Set(data.messages.flatMap(historyRowIds)));
        cache.set(page.id, { page });
        // An empty page is never a reading position: it would make the window
        // rewind one request per click for content nobody can see.
        if (newChain || messageCount > 0) focus = index;
        const releasedPageIds = prune();
        return { status: 'applied', page, releasedPageIds, messageCount, anchor: data.anchor ?? null };
    }

    // `stillWanted` marks an explicit navigation to one place: a newer one supersedes it while it
    // loads, it waits out a window read rather than replacing it, and its failure is its own, never
    // the window's Retry.
    function request(direction, index, cursor, newChain = false, { query = null, stillWanted = null, prepared = null } = {}) {
        if (!alive()) return Promise.resolve({ status: 'disposed' });
        const newest = stillWanted && newestInFlight();
        if (newest) return newest.then(() => stillWanted()
            ? request(direction, index, cursor, newChain, { query, stillWanted, prepared }) : { status: 'superseded' });
        if (pending && stillWanted && pending.stillWanted) { pending.abort.abort(); pending = null; }
        if (pending && stillWanted) return pending.promise.then(() => (stillWanted()
            ? request(direction, index, cursor, newChain, { query, stillWanted, prepared }) : { status: 'superseded' }));
        if (pending) return pending.direction === direction
            ? pending.promise : Promise.resolve({ status: 'busy' });
        const operation = { direction, index, cursor, newChain, stillWanted, abort: new AbortController(), landed: landedArrivals };
        if (!stillWanted) failure = null;
        pending = operation;
        operation.promise = Promise.resolve()
            .then(() => alive() && pending === operation
                ? prepared || fetchPage(cursor, { signal: operation.abort.signal, ...query }) : null)
            .then(data => {
                if (!alive() || pending !== operation) return { status: 'disposed' };
                const result = land(data, operation);
                // Arrivals that landed meanwhile may lie past this newest page, which let
                // them go: they are read again from its cursor, else on the newest page. A page around one
                // place is not the present: the window's arrivals go on from their own cursor.
                if (direction === 'latest' && landedArrivals !== operation.landed && canArrive()) void read();
                return result;
            })
            .catch(error => {
                if (!alive() || pending !== operation) return { status: 'disposed' };
                if (!stillWanted) failure = { direction, index, cursor, newChain, error };
                return { status: 'error', error, cursor };
            })
            .finally(() => {
                if (pending !== operation) return;
                pending = null;
                publish();
            });
        publish();
        return operation.promise;
    }

    // One read of what arrived since `after`. Its rows cover the window's present only
    // when they account for everything stored since: a gap in either stream (narration's
    // unfinished live line excepted: the next read takes it), an unknown newest arrival, a refusal
    // (409 history_view_changed/history_delta_reset) or any failure forgets the cursor, and
    // the caller reads the newest page instead. An answer that lands after a newest read
    // restarted the arrivals is older than that window: it is dropped, and the arrivals are
    // read again from the newer cursor, so it never rolls rows or the newest message back.
    function readArrivals(ask) {
        const cursor = after, abort = new AbortController();
        const promise = Promise.resolve()
            .then(() => fetchPage(cursor, { signal: abort.signal }))
            .then(data => {
                if (!alive()) return { status: 'disposed' };
                if (after !== cursor) return startArrivals(ask);
                const gap = data?.window?.truncated_by?.some?.(cause => /^(chat|progress)_/.test(String(cause))
                    && String(cause) !== 'progress_incomplete_live_line');
                if (!Array.isArray(data?.messages) || typeof data.after_cursor !== 'string' || !data.after_cursor
                    || gap || data.window?.latest_message === null) {
                    throw new TypeError('History arrivals do not account for everything stored since the last read');
                }
                after = data.after_cursor;
                arrived += rowCount(data.messages);
                landedArrivals += 1;
                applyArrivals(data.messages, { window: data.window ?? null, fromReconnect: ask.fromReconnect });
                return alive() ? { status: 'applied', messageCount: rowCount(data.messages) } : { status: 'disposed' };
            })
            .catch(error => {
                if (after === cursor) after = null;
                return alive() ? { status: 'reset', error } : { status: 'disposed' };
            })
            .finally(() => { if (arrival?.promise === promise) arrival = null; });
        arrival = { promise, abort };
        return promise;
    }
    // A window that took in a whole newest page of rows (at least the default page:
    // 150 conversation and 60 narration rows, CLIENT_CONTRACT §5) reads the newest
    // page again, which re-anchors it and lets go of what that page no longer holds.
    const arrivable = () => Boolean(after) && arrived <= Math.max(windowRows, 210);
    const startArrivals = ask => (!alive() ? Promise.resolve({ status: 'disposed' })
        : arrivable() ? readArrivals(ask) : Promise.resolve({ status: 'unavailable' }));
    // One read in flight and at most one queued behind it, so every caller gets a read
    // begun after its call; a queued read reconciles as a reconnect when any caller it
    // serves is one.
    function arrivals(fromReconnect) {
        if (queued) { queued.fromReconnect ||= fromReconnect; return queued.promise; }
        if (!arrival) return startArrivals({ fromReconnect });
        const ask = { fromReconnect };
        ask.promise = arrival.promise.then(() => { queued = null; return startArrivals(ask); });
        queued = ask;
        return ask.promise;
    }
    // A read begun after this call, once chat's newest read in flight settled: what
    // arrived since the newest read, else the newest page (`whole`, no cursor yet, a
    // window grown full, one `canArrive` declines, a refusal or a failure). When
    // arrivals cannot answer, chat's newest read starts within this very call.
    function read({ fromReconnect = false, whole = false } = {}) {
        const inFlight = newestInFlight();
        if (inFlight) return inFlight.then(() => read({ fromReconnect, whole }));
        if (!alive()) return Promise.resolve(false);
        if (whole || !canArrive() || !(arrival || arrivable())) {
            return readNewest({ fromReconnect });
        }
        return arrivals(fromReconnect).then(result => result.status === 'applied' || (alive() && readNewest({ fromReconnect })));
    }

    return {
        getState,
        read,
        arrivalsLanded: () => landedArrivals,
        // The first recent response owns the frozen chain. Later recent/live
        // refreshes are applied by chat itself, without registering their bodies
        // here or replacing the original page-zero continuation, unless chat asks
        // to `rechain` (a window shifted under a reader following the present):
        // this read then starts the new chain as `latest` would, without reading
        // the same page again; its descriptor carries `rechain` and `rechained`.
        acceptRecent(data, rechain = null) {
            if (!alive()) return { status: 'disposed' };
            newest(data);
            const sourceError = sourceReadError(data);
            if (sourceError && !pending) {
                failure = { direction: 'recent', index: 0, cursor: null, newChain: true, error: sourceError };
                publish();
                return { status: 'error', error: sourceError, cursor: null };
            }
            if (!sourceError && failure?.direction === 'recent') { failure = null; publish(); }
            if (pending || (pages.length && !rechain)) return { status: 'ignored' };
            try {
                const result = pages.length ? land(data, { index: 0, direction: 'latest', newChain: true }, { ...rechain, rechained: true })
                    : land(data, { index: 0, direction: 'recent', newChain: true });
                failure = null;
                publish();
                return result;
            } catch (error) {
                failure = { direction: 'recent', index: 0, cursor: null, newChain: true, error };
                publish();
                return { status: 'error', error, cursor: null };
            }
        },
        older() {
            if (!pages.length) return Promise.resolve({ status: 'unavailable' });
            const index = bounds().last + 1;
            const cursor = pages[index]?.requestCursor ?? pages[index - 1].nextCursor;
            if (index === pages.length && !pages[index - 1].hasMore) {
                return Promise.resolve({ status: 'unavailable' });
            }
            return request('older', index, cursor);
        },
        newer() {
            const index = bounds().first - 1;
            return index < 0 || !pages.length
                ? Promise.resolve({ status: 'unavailable' })
                : request('newer', index, pages[index].requestCursor);
        },
        // Explicit return to latest starts a new snapshot only after success.
        // Old protected pages remain mounted until the reader unpins them.
        latest: () => request('latest', 0, null, true),
        // One explicit navigation to one place, as a new chain: the page around `address`
        // (`task:<id>`, DESIGN "Room work status"), or a page the reader left read again by its
        // `page_cursor` (`revisit`). Dropped before it applies once `stillWanted()` says no.
        // app may have read the destination before switching rooms, so a failed read leaves the old view intact.
        around: (address, stillWanted, prepared = null) => request('around', 0, null, true,
            { query: { around: address }, stillWanted, prepared }),
        revisit: (cursor, stillWanted) => request('revisit', 0, cursor, true, { stillWanted }),
        // The `page_cursor` of this chain's page that holds a row, if one does: where a return reads it again.
        pageOf: id => pages.find(page => pageRows.get(page.id)?.has(id))?.requestCursor ?? null,
        whenIdle: () => pending?.promise || Promise.resolve(),
        retry() {
            if (!failure) return Promise.resolve({ status: 'unavailable' });
            const { direction, index, cursor, newChain } = failure;
            return request(direction, index, cursor, newChain);
        },
        trim() {
            if (!alive()) return [];
            const released = prune();
            if (released.length) publish();
            return released;
        },
        destroy() {
            if (destroyed) return;
            destroyed = true;
            pending?.abort.abort();
            arrival?.abort.abort();
            pending = null;
            failure = null;
            after = null;
            for (const entry of cache.values()) release(entry);
            pages = [];
        },
    };
}
