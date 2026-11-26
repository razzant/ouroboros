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

// Where a read goes is apart from whether it is still wanted: a navigation to one place (a task, a page left) begins
// a chain away from the present and fails on its own, while the present (`latest`, also ↓'s) is the window's read.
const navigates = direction => direction === 'around' || direction === 'revisit';

// Why a physical read is made (CLIENT_CONTRACT "Reading history"): `reason`, the selector it sends (`mode`), the gate
// or fallback leg that chose it (`decision`) and what that leg met (`detail`). The server counts them; nothing here
// or there selects by them. A navigation's reason is `navigate` unless its caller names another; empty names go.
const MODES = { older: 'older', newer: 'page', revisit: 'page', around: 'around', latest: 'newest', recent: 'newest' };
const why = tags => Object.fromEntries(Object.entries(tags).filter(([, value]) => value));
// The newest read that answers for arrivals: a typed 409 (with the server's cause; an older one says only its code),
// a 200 this window cannot admit (or a profile it does not serve), or a failure (its status, else the network).
function fallbackLeg(error) {
    if (error?.status === 409) {
        return { decision: 'after_refused', detail: error.body?.detail || String(error.body?.reason_code || '').replace('history_', '') };
    }
    if (error?.rejected || error?.foreignProfile) return { decision: 'after_rejected', detail: error.rejected || 'profile_mismatch' };
    return { decision: 'after_failed', detail: error?.status ? `http_${error.status}` : 'network' };
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
    arrivalGate = () => '',  // the gate `canArrive` closed, by name: it labels the newest read sent instead
    rebaseOlderEdge = () => false,  // a horizon gap under the window may need the present before going older
    newestInFlight = () => null,
    catchingUp = () => null,  // a reconnect's whole read still to come (room_retire.js createCatchUp)
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
    let destroyed = false, visit = 0;
    // Where the next read of new arrivals starts, the newest read's row count and the
    // rows read since; the read in flight and at most one queued behind it.
    let after = null, windowRows = 0, arrived = 0, arrival = null, queued = null, landedArrivals = 0, lastDrawError = null;
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
            retryReturn: Boolean(failure?.returning),  // the failed read was ↓'s own, its Retry a fresh ↓
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

    function registryReadableAgain() {
        if (failure?.error?.body?.project_registry !== 'unavailable') return;
        // The old page still needs its own Retry. A newer successful read has answered only the
        // availability question, so retain the earlier refusal as its cause, not a current archive offer.
        failure = { ...failure, error: new Error('Some saved history could not be loaded.', { cause: failure.error }) };
        publish();
    }

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
        // A failed source has no page. A partial projection still has useful rows and continuation boundaries.
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
        lastDrawError = null;
        try {
            applyPage(data.messages, { ...page, direction, window: data.window ?? null,
                recentVersion: data.recentVersion, requestedAt: data.requestedAt, page_cursor: data.page_cursor,
                reason_code: data.reason_code, project_registry: data.project_registry, anchor: data.anchor ?? null, ...extra });
        } catch (error) {
            lastDrawError = error;
            throw error;
        }
        if (!alive()) return { status: 'disposed' };
        // Only a read of the present restarts the arrivals; a page around one place is not the present.
        if (newChain) {
            chain = nextChain; pages = [page]; pageRows.clear(); failure = null; jumped = navigates(direction);
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

    // The reader returns to the present (↓): a page still loading for older history, or a navigation to one place,
    // yields before it is drawn, so it can neither move the window's focus nor release the present under that reader.
    // A read of the present stays; the arrivals and the room's newest read are not the pager's to stop.
    function toPresent() {
        if (!pending || pending.direction === 'latest' || pending.direction === 'recent') return;
        pending.abort.abort();
        pending = null;
        publish();
    }

    // `stillWanted` marks a read made for one intent (a navigation, ↓): it is dropped before it lands once the reader
    // moved on, and a newer navigation supersedes it while it loads. A navigation waits out a window read rather
    // than replacing it, and its failure is its own, never the window's Retry; ↓'s `latest` is the window's read.
    function request(direction, index, cursor, newChain = false, { query = null, stillWanted = null, prepared = null, tags = null } = {}) {
        if (!alive()) return Promise.resolve({ status: 'disposed' });
        const navigation = navigates(direction);
        const again = () => (stillWanted()
            ? request(direction, index, cursor, newChain, { query, stillWanted, prepared, tags }) : { status: 'superseded' });
        const newest = navigation && newestInFlight();
        if (newest) return newest.then(again);
        if (pending && navigation && pending.stillWanted) { pending.abort.abort(); pending = null; }
        if (pending && navigation) return pending.promise.then(again);
        if (stillWanted && direction === 'latest') toPresent();
        // A read of the present already out serves this one too: it lands while its latest asker wants it (a ↓ by
        // its claim, not an earlier one's), and always once one without a claim joins.
        if (pending && direction === 'latest' && (stillWanted || pending.direction === 'latest')) {
            pending.stillWanted &&= stillWanted;
            return pending.promise;
        }
        if (pending) return pending.direction === direction
            ? pending.promise : Promise.resolve({ status: 'busy' });
        const operation = { direction, index, cursor, newChain, stillWanted, abort: new AbortController(), landed: landedArrivals,
            tags: { reason: 'navigate', mode: MODES[direction], ...tags } };
        if (!navigation) failure = null;
        pending = operation;
        operation.promise = Promise.resolve()
            .then(() => alive() && pending === operation
                ? prepared || fetchPage(cursor, { signal: operation.abort.signal, ...query, tags: why(operation.tags) }) : null)
            .then(data => {
                if (!alive() || pending !== operation) return { status: 'disposed' };
                const result = land(data, operation);
                // Arrivals that landed meanwhile may lie past this newest page, which let
                // them go: they are read again from its cursor, else on the newest page. A page around one
                // place is not the present: the window's arrivals go on from their own cursor.
                if (direction === 'latest' && landedArrivals !== operation.landed && canArrive()) {
                    void read({ reason: operation.tags.reason, decision: 'base_superseded' });
                }
                return result;
            })
            .catch(error => {
                if (!alive() || pending !== operation) return { status: 'disposed' };
                // The window's read keeps its failure for Retry while it is wanted; one the reader left is no failure.
                // A claim still on it at failure is ↓'s own (a read ↓ only joined is its asker's).
                if (!navigation && operation.stillWanted?.() !== false) {
                    failure = { direction, index, cursor, newChain, error, returning: Boolean(operation.stillWanted), tags: operation.tags };
                }
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
        let admitted = false;
        const promise = Promise.resolve()
            .then(() => fetchPage(cursor, { signal: abort.signal,
                tags: why({ reason: ask.reason, mode: 'after', decision: ask.decision }) }))
            .then(data => {
                if (!alive() || abort.signal.aborted) return { status: 'disposed' };
                if (after !== cursor) return startArrivals(Object.assign(ask, { decision: 'base_superseded' }));
                const gap = data?.window?.truncated_by?.some?.(cause => /^(chat|progress)_/.test(String(cause))
                    && String(cause) !== 'progress_incomplete_live_line');
                // The first fact the answer lacks to account for what was stored since; it names the newest read sent instead.
                const rejected = !Array.isArray(data?.messages) || !Array.isArray(data.repairs ?? []) ? 'bad_messages'
                    : typeof data.after_cursor !== 'string' || !data.after_cursor ? 'no_continuation'
                        : gap ? 'stream_gap' : data.window?.latest_message === null ? 'latest_unknown' : '';
                if (rejected) {
                    throw Object.assign(new TypeError('History arrivals do not account for everything stored since the last read'), { rejected });
                }
                admitted = true;
                // `repairs`: the rows the window holds of a task it shows that settled, as the newest page shows them now.
                const repairs = data.repairs ?? [], card = row => row.presentation_owner_task_id || row.task_id;
                // A card only repairs changed is no arrival: it marks nothing new (one that also gained a row does).
                const quiet = new Set(repairs.map(card).filter(id => id
                    && !data.messages.some(row => !isReplayEvidenceRow(row) && card(row) === id)));
                applyArrivals(data.messages, { window: data.window ?? null, fromReconnect: ask.fromReconnect,
                    requestedAt: data.requestedAt, repairs, quiet });
                // The arrivals go on from this answer only once it is drawn: one that failed to apply reads the newest page.
                after = data.after_cursor;
                arrived += rowCount(data.messages);
                landedArrivals += 1;
                registryReadableAgain();
                return alive() ? { status: 'applied', messageCount: rowCount(data.messages) } : { status: 'disposed' };
            })
            .catch(error => {
                if (abort.signal.aborted) return { status: 'disposed' };
                if (after === cursor) after = null;
                const leg = admitted ? { decision: 'after_rejected', detail: 'apply_failed' } : fallbackLeg(error);
                return alive() ? { status: 'reset', error, leg } : { status: 'disposed' };
            })
            .finally(() => { if (arrival?.promise === promise) arrival = null; });
        arrival = { promise, abort };
        return promise;
    }
    // A window that took in a whole newest page of rows (at least the default page:
    // 150 conversation and 60 narration rows, CLIENT_CONTRACT §5) reads the newest
    // page again, which re-anchors it and lets go of what that page no longer holds.
    const arrivable = () => Boolean(after) && arrived <= Math.max(windowRows, 210);
    const unarrivable = () => ({ decision: after ? 'window_overflow' : 'no_cursor' });  // the gate `arrivable` closed
    const startArrivals = ask => (!alive() || ask.currents?.some(ok => ok()) === false ? Promise.resolve({ status: 'disposed' })
        : arrivable() ? readArrivals(ask) : Promise.resolve({ status: 'unavailable', leg: unarrivable() }));
    // One read in flight and at most one queued behind it, so every caller gets a read
    // begun after its call; a queued read reconciles as a reconnect when any caller it
    // serves is one, and is made for the reason of the first.
    function arrivals(fromReconnect, tags, current = () => true) {
        if (queued) { queued.fromReconnect ||= fromReconnect; queued.currents.push(current); return queued.promise; }
        if (!arrival) return startArrivals({ fromReconnect, ...tags, currents: [current] });
        const ask = { fromReconnect, ...tags, currents: [current] };
        ask.promise = arrival.promise.then(() => {
            if (queued !== ask) return { status: 'disposed' };
            queued = null;
            return startArrivals(ask);
        });
        queued = ask;
        return ask.promise;
    }
    // A read begun after this call, once chat's newest read in flight settled: what
    // arrived since the newest read, else the newest page (`whole`, no cursor yet, a
    // window grown full, one `canArrive` declines, a refusal or a failure). When
    // arrivals cannot answer, chat's newest read starts within this very call. `reason`
    // and `decision` (a `whole` read's, else the arrivals read's) say why, with `detail`. While a reconnect's whole
    // read is still to come, any other read waits for it and then reads only what is left.
    function read({ fromReconnect = false, whole = false, current = () => true, reason = '', decision = '', detail = '' } = {}) {
        const started = visit;
        // A superseded reconnect leaves its queued read to the current catch-up.
        if (!alive() || !current()) return Promise.resolve(false);
        const inFlight = newestInFlight() || !fromReconnect && catchingUp();
        if (inFlight) return inFlight.then(() => started === visit && current() && read({ fromReconnect, whole, current, reason, decision, detail }));
        if (whole || !canArrive() || !(arrival || arrivable())) {
            // The first gate that chose the newest page, in the order they are weighed.
            const gate = whole ? decision || 'explicit_whole' : !canArrive() ? arrivalGate() : unarrivable().decision;
            return readNewest({ fromReconnect, ...why({ reason, decision: gate, detail }) });
        }
        return arrivals(fromReconnect, { reason, decision }, current).then(result => started === visit && current() && (result.status === 'applied'
            || (result.status !== 'disposed' && alive() && readNewest({ fromReconnect, ...why({ reason, detail, ...result.leg }) }))));
    }

    const pager = {
        getState,
        read,
        arrivalsLanded: () => landedArrivals,
        // The first recent response owns the frozen chain. Later recent/live
        // refreshes are applied by chat itself, without registering their bodies
        // here or replacing the original page-zero continuation, unless chat asks
        // to `rechain` (a window shifted under a reader following the present):
        // this read then starts the new chain as `latest` would, without reading
        // the same page again; its descriptor carries `rechain` and `rechained`.
        acceptRecent(data, rechain = null, draw = null) {
            if (!alive()) return { status: 'disposed' };
            const sourceError = sourceReadError(data);
            if (sourceError && !pending) {
                draw?.();
                newest(data);
                failure = { direction: 'recent', index: 0, cursor: null, newChain: true, error: sourceError };
                publish();
                return { status: 'error', error: sourceError, cursor: null };
            }
            if (pending || (pages.length && !rechain)) {
                draw?.();
                newest(data);
                if (!sourceError && (failure?.direction === 'recent' || failure?.direction === 'latest')) { failure = null; publish(); }
                else if (!sourceError && data?.page_cursor) registryReadableAgain();
                return { status: 'ignored' };
            }
            lastDrawError = null;
            try {
                const result = pages.length ? land(data, { index: 0, direction: 'latest', newChain: true }, { ...rechain, rechained: true })
                    : land(data, { index: 0, direction: 'recent', newChain: true });
                newest(data);
                failure = null;
                publish();
                return result;
            } catch (error) {
                const drawFailed = lastDrawError === error;
                lastDrawError = null;
                if (!drawFailed) {
                    draw?.();
                    newest(data);
                }
                failure = { direction: 'recent', index: 0, cursor: null, newChain: true, error };
                publish();
                return { status: 'error', error, cursor: null, ...(drawFailed ? { drawFailed: true } : {}) };
            }
        },
        older(tags = null) {
            if (!pages.length) return Promise.resolve({ status: 'unavailable' });
            const index = bounds().last + 1;
            const cursor = pages[index]?.requestCursor ?? pages[index - 1].nextCursor;
            if (index === pages.length && !pages[index - 1].hasMore) {
                return Promise.resolve({ status: 'unavailable' });
            }
            return request('older', index, cursor, false, { tags: { ...(pages[index] && { mode: 'page' }), ...tags } });
        },
        newer(tags = null) {
            const index = bounds().first - 1;
            return index < 0 || !pages.length
                ? Promise.resolve({ status: 'unavailable' })
                : request('newer', index, pages[index].requestCursor, false, { tags });
        },
        // Explicit return to latest starts a new snapshot only after success.
        // Old protected pages remain mounted until the reader unpins them. ↓ passes its claim (`stillWanted`):
        // pages still loading yield to it (`toPresent`), a read of the present already out is shared, and once the
        // reader has moved on its answer is dropped before it lands.
        latest: (stillWanted = null, tags = null) => request('latest', 0, null, true, { stillWanted, tags }),
        // Load more reads older, ↓ the present (owner 2026-09-14, 10-05), past empty pages that cross the server's read ceiling.
        // Growth re-anchors unless the reader is walking the older chain. `current` guards each request.
        async edge(direction, { current = () => true, tags = null } = {}) {
            if (direction === 'older' && current() && rebaseOlderEdge(getState())) {
                const rebased = await pager.latest(null, { ...tags, decision: 'horizon_gap' });
                if (destroyed || rebased.status !== 'applied') return rebased;
            }
            let result;
            do {
                result = current() ? await pager[direction](tags) : {};
            } while (!destroyed && result.status === 'applied' && result.messageCount === 0
                && getState()[direction === 'older' ? 'canOlder' : 'canNewer']);
            return result;
        },
        toPresent,
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
            const { direction, index, cursor, newChain, tags } = failure;
            return request(direction, index, cursor, newChain, { tags: { ...tags, reason: 'retry' } });
        },
        trim() {
            if (!alive()) return [];
            const released = prune();
            if (released.length) publish();
            return released;
        },
        // Abort in-flight/queued arrivals on disconnect without forgetting `after` or the reader's pages.
        cancelArrivals() {
            arrival?.abort.abort();
            arrival = null;
            queued = null;
        },
        // A retained room stops reads without releasing its pages or the reader's chain.
        suspend() {
            visit += 1;
            pending?.abort.abort();
            arrival?.abort.abort();
            pending = null;
            arrival = null;
            queued = null;
        },
        destroy() {
            if (destroyed) return;
            destroyed = true;
            visit += 1;
            pending?.abort.abort();
            arrival?.abort.abort();
            pending = null;
            arrival = null;
            queued = null;
            failure = null;
            after = null;
            for (const entry of cache.values()) release(entry);
            pages = [];
        },
    };
    return pager;
}
