import { isReplayEvidenceRow } from './chat_activity.js';

/** Chat's bounded archive-page owner. DOM, reading protection and live rows stay
 * with the chat instance; only fetchPage is asynchronous. Cursors are opaque.
 */
export function createChatHistoryPager({
    fetchPage,
    applyPage,
    releasePage,
    isPageProtected = () => false,
    onState = () => {},
    isAlive = () => true,
    maxPages = 3,
}) {
    const limit = Number.isInteger(maxPages) && maxPages > 0 ? maxPages : 3;
    const cache = new Map();
    let pages = [];
    let chain = 0;
    let focus = 0;
    let pending = null;
    let failure = null;
    let destroyed = false;
    const alive = () => !destroyed && isAlive();

    // Protected pages can remain as islands while the ordinary reading window
    // moves. Navigation follows the contiguous window containing the last load.
    // A landed page that contributed no rows owns nothing to mount or restore, so
    // the window steps over it: it is a boundary the reader already crossed, not a
    // gap. Only a NON-EMPTY page missing from the cache is a real hole above/below.
    const held = page => cache.has(page.id) || page.rows === 0;
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
            olderExhausted: initialized && !canOlder,
            newerExhausted: initialized && !canNewer,
            pageCount: pages.length,
            firstPage: initialized ? pages[first] : null,
            lastPage: initialized ? pages[last] : null,
            cachedPages: [...cache.values()].map(entry => entry.page),
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

    function land(data, { index, direction, newChain }) {
        const error = sourceReadError(data);
        if (error) throw error;
        if (!Array.isArray(data?.messages) || typeof data.has_more !== 'boolean'
            || data.page_cursor == null || (data.has_more && data.next_cursor == null)) {
            throw new TypeError('History page is missing its messages or continuation boundary');
        }
        const nextChain = newChain ? chain + 1 : chain;
        const messageCount = data.messages.filter(row => !isReplayEvidenceRow(row)).length;
        // Re-reading a page refreshes its row count, never its frozen boundaries.
        const prior = newChain ? null : pages[index];
        const page = prior?.rows === messageCount ? prior : Object.freeze(prior
            ? { ...prior, rows: messageCount }
            : { id: `history-page-${nextChain}-${index}`, chain: nextChain, index,
                requestCursor: data.page_cursor, nextCursor: data.next_cursor ?? null,
                hasMore: data.has_more, rows: messageCount });
        applyPage(data.messages, { ...page, direction, window: data.window ?? null });
        if (!alive()) return { status: 'disposed' };
        if (newChain) { chain = nextChain; pages = [page]; }
        else pages[index] = page;
        cache.set(page.id, { page });
        // An empty page is never a reading position: it would make the window
        // rewind one request per click for content nobody can see.
        if (newChain || messageCount > 0) focus = index;
        const releasedPageIds = prune();
        return { status: 'applied', page, releasedPageIds, messageCount };
    }

    function request(direction, index, cursor, newChain = false) {
        if (!alive()) return Promise.resolve({ status: 'disposed' });
        if (pending) return pending.direction === direction
            ? pending.promise : Promise.resolve({ status: 'busy' });
        const operation = { direction, index, cursor, newChain, abort: new AbortController() };
        failure = null;
        pending = operation;
        operation.promise = Promise.resolve()
            .then(() => alive() && pending === operation
                ? fetchPage(cursor, { signal: operation.abort.signal }) : null)
            .then(data => {
                if (!alive() || pending !== operation) return { status: 'disposed' };
                return land(data, operation);
            })
            .catch(error => {
                if (!alive() || pending !== operation) return { status: 'disposed' };
                failure = { direction, index, cursor, newChain, error };
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

    return {
        getState,
        exportResume(pageId = '') {
            const chosen = pages.findIndex(page => page.id === pageId);
            return pages.length ? { pages: [...pages], focus: chosen >= 0 ? chosen : focus } : null;
        },
        // A resume written before pages carried `rows` restores unchanged: an
        // unknown row count is treated as non-empty until the page is re-read.
        restore(saved) {
            if (pages.length || !Array.isArray(saved?.pages) || !saved.pages.length
                || !Number.isInteger(saved.focus) || !saved.pages[saved.focus]
                || !saved.pages.every(page => typeof page.requestCursor === 'string'
                    && typeof page.hasMore === 'boolean')) return Promise.resolve({ status: 'unavailable' });
            pages = saved.pages.map(page => Object.freeze({ ...page }));
            focus = saved.focus;
            chain = pages[focus].chain;
            return request('restore', focus, pages[focus].requestCursor);
        },
        // The first recent response owns the frozen chain. Later recent/live
        // refreshes are applied by chat itself, without registering their bodies
        // here or replacing the original page-zero continuation.
        acceptRecent(data) {
            if (!alive()) return { status: 'disposed' };
            const sourceError = sourceReadError(data);
            if (sourceError && !pending) {
                failure = { direction: 'recent', index: 0, cursor: null, newChain: true, error: sourceError };
                publish();
                return { status: 'error', error: sourceError, cursor: null };
            }
            if (pages.length || pending) return { status: 'ignored' };
            try {
                const result = land(data, { index: 0, direction: 'recent', newChain: true });
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
            pending = null;
            failure = null;
            for (const entry of cache.values()) release(entry);
            pages = [];
        },
    };
}
