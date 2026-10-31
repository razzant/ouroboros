// When a Project room has READ a visible revision (DESIGN "Unread").
//
// A revision is read only when a history read covering it has crossed a real
// browser paint while the room stayed shown AND the reader is at the newest
// message — the one that arrived last, on screen (`isAtNewestMessage`), normally
// near the bottom of the conversation. Opening an unread room lands the reader on
// that message (`newestMessageOffset`), first reading the room's older pages while
// none drawn holds it (`arrive`); refreshing a room
// while its reader is elsewhere is not reading; the chat instance reports each arrival at
// the newest message (a scroll, a page shown again, an older page applied) so the
// withheld acknowledgement is retried then, and app.js's existing state snapshots
// offer it again — no timer of this module's own. This module owns the paint
// generation, the highest covered revision, which message is the newest and that
// arrival edge; the chat instance owns reading history, and app.js alone posts
// the acknowledgement.

import { sameHistoryChain } from './chat_history.js';
import { historyNodeOnScreen } from './chat_history_replay.js';

// A read whose bounded search ran out first: none from `latest_before` up to it.
const searched = (window) => window?.latest_message === null && Number.isSafeInteger(window.latest_before);

/**
 * @param {{
 *   read: (fresh: boolean, whole: boolean) => Promise<boolean>,  // true when the read succeeded; `whole`: the newest page
 *   isShown: () => boolean,
 *   isReadingLatest: (latest: undefined|null|object) => boolean,  // the reader is at `latest`
 *   onReadingLatest?: () => void,
 *   land?: (latest: undefined|null|object, token: number) => void,  // the opening `token` lands on `latest`
 *   arriving?: () => number|null,  // the unread opening still owed: its claim's token
 *   older?: (current: () => boolean) => Promise<{status?: string}>,  // one older page, while `current()`
 *   present?: (current: () => boolean) => Promise<unknown>,  // the newest page, if not drawn, while `current()`
 *   delivered?: (id: string) => boolean,  // a drawn page holds this row
 *   onNote?: () => void,  // every reading moment (a scroll edge, a paint, an arrival): read_ahead.js samples there
 * }} facts
 */
export function createProjectReadReceipt({ read, isShown, isReadingLatest, onReadingLatest = () => {}, land = () => {},
    arriving = () => null, older = async () => ({}), present = async () => {}, delivered = () => false, onNote = () => {} }) {
    let generation = 0;
    let coveredRevision = 0;
    let readingLatest = false;
    let settledLatest;  // the newest message the previous settle named
    let recentRead = {};  // the newest admitted recent read: its window, coverage and request
    let found = null;  // what an older page of a quiet chain found (the newest message, none, or unknown) and its coverage
    let reaching = null;  // the opening whose older pages are being read
    // A history_id and a search bound are byte offsets: they mean the same bytes
    // only through the same room view on one chain of byte coordinates.
    const sameBytes = (coverage, frozen) => Boolean(coverage) && coverage.view === frozen?.view
        && sameHistoryChain(coverage.spans?.chat, frozen?.spans?.chat);
    // A read that searched from its `latest_before` up to itself and found none
    // joins what was searched below `frozen`'s upper on the same bytes: none is left between them.
    const joins = ({ window, coverage }, frozen) => searched(window)
        && window.latest_before <= frozen?.upper?.chat && sameBytes(coverage, frozen);
    // The recent read names the newest message (null: unknown). When its bounded
    // search ran out first, the older pages of its quiet chain carry it on
    // (history_paging.latest_arrival): the message they name, or the absence the
    // one reaching the start of the chat proves, still holds while this read joins
    // that chain. A newer arrival, a source gap or another membership breaks that.
    const carried = () => found !== null && joins(recentRead, found.coverage);
    const newest = () => carried() ? found.message : recentRead.window?.latest_message;
    // Still to reach: named but on no drawn page yet, or unknown only because the
    // bounded search ran out first and no page of its quiet chain has carried it on.
    // Unknown otherwise (an unreadable, unfinished or broken source) is never searched.
    const unreached = () => {
        const latest = newest();
        return latest?.history_id ? !delivered(latest.history_id)
            : latest === null && !carried() && searched(recentRead.window);
    };
    // The unread opening lands once on the newest message, wherever it lies (DESIGN
    // "Unread"). While it is still to reach, the room's older pages are
    // read for that opening alone, one at a time, re-reading what the newest recent
    // read and the pages name after each. The reader's own move, a question, the
    // room hidden or closed, a failed page or the chat's start ends it; a page
    // already asked for may still be drawn, but nothing more is asked or landed.
    // The pager keeps its `maxPages`: that bounds what stays loaded, not how far
    // this reads. A start proven clean lets the bottom decide, but the newest page
    // may be among those let go: the opening returns there once, as ↓ does, and
    // reaches what that fresh read names the same way. Every later read finds no
    // opening owed, so it never pages or jumps.
    async function arrive(readable) {
        const token = arriving();
        if (token === null || reaching === token) return;
        const owed = () => arriving() === token && isShown() && unreached();
        const returning = () => arriving() === token && isShown() && newest() === undefined;
        const reach = readable && owed();
        if (reach) {
            reaching = token;
            try {
                for (let returned = false; ; returned = true) {
                    for (let status = 'busy'; owed() && (status === 'busy' || status === 'applied');) {
                        ({ status } = await older(owed));
                    }
                    if (returned || !returning()) break;
                    await present(returning);
                }
            } finally { if (reaching === token) reaching = null; }
        }
        land(readable && isShown() ? newest() : null, token);
        // Landed after the opening read's own paint: the landing reports where the reader is.
        if (reach) note();
    }
    // Scroll edges report arrivals only; a discrete change (the page or the
    // window shown again, another newest message) reports the current position once.
    const note = ({ discrete = false } = {}) => {
        onNote();
        const latest = isReadingLatest(newest());
        if (latest && (discrete || !readingLatest)) onReadingLatest();
        readingLatest = latest;
    };
    // Another newest message than the previous settle named moves where the
    // reader must be: the edge is taken again without a scroll.
    const settle = () => {
        const identity = JSON.stringify(newest()) ?? '';
        if (settledLatest !== undefined && identity !== settledLatest) note({ discrete: true });
        settledLatest = identity;
    };
    return {
        cancel() { generation += 1; },
        // Whether the opening walks on the pager's frozen `chain` (described by its
        // own recent read) or first returns to the newest admitted read's snapshot.
        // The chain serves that read's target while it is that read, named the same
        // message on the same bytes, or itself searched (its cursors carry the
        // search on) and this read's search joins it; growth that keeps those facts
        // never restarts the walk. A chain that named a message carries no search:
        // its older pages name nothing. Unknown targets are never walked (null).
        continuation(chain = {}) {
            const { window, coverage } = recentRead, id = window?.latest_message?.history_id;
            if (!id && !searched(window)) return null;
            if (Number.isSafeInteger(chain.recentVersion) && chain.recentVersion === recentRead.recentVersion) return 'older';
            const serves = id ? chain.window?.latest_message?.history_id === id && sameBytes(coverage, chain.coverage)
                : searched(chain.window) && joins(recentRead, chain.coverage);
            return serves ? 'older' : 'latest';
        },
        // Only a revision newer than any covered one needs a fresh read; the
        // receipt is still taken after this call's own paint either way. A read
        // that cannot name the newest message covers nothing, so the same revision
        // is read again: a repaired source can name it without a new revision.
        // The current opening read consumes its landing even on a gap: a later
        // arrival must not inherit that navigation. Unknown/hidden content never lands.
        // A fresh read takes only what arrived since the newest read, except while the
        // newest message is unknown and unsearched: only the newest page can name it.
        async refresh({ revision = 0 } = {}) {
            const own = ++generation;
            const target = Math.max(0, Number(revision) || 0);
            const readable = await read(target > coveredRevision, newest() === null && !searched(recentRead.window));
            const covered = readable && newest() !== null;
            if (covered && target > coveredRevision) coveredRevision = target;
            if (own !== generation) return { painted: false, revision: target };
            void arrive(readable);
            if (!covered || !isShown()) return { painted: false, revision: target };
            // Two frames cover layout followed by paint/composite.
            await new Promise((resolve) => requestAnimationFrame(() => requestAnimationFrame(resolve)));
            const painted = own === generation && isShown();
            if (painted) onNote();
            const atLatest = painted && isReadingLatest(newest());
            // A withheld read lowers the edge, so the next arrival retries it.
            if (painted && !atLatest) readingLatest = false;
            return { painted, read: atLatest, revision: target };
        },
        note,
        // Each admitted recent read, before its rows are drawn. One that cannot
        // name the newest message leaves no revision covered, so the next refresh
        // reads again: a healed source names it without a new revision.
        recent({ window, coverage, recentVersion }) {
            recentRead = { window, coverage, recentVersion };
            if (newest() === null) coveredRevision = 0;
        },
        // Rows an `after_cursor` read added, once drawn: the newest message among them
        // (`history_paging.latest_arrival` names it) is the room's newest, on the same
        // recent read's bytes; none named leaves the newest as it was. A read that
        // cannot name it never comes here (the newest page is read instead).
        arrived(window) {
            const named = window?.latest_message;
            if (named) recentRead = { ...recentRead, window: { ...recentRead.window, latest_message: named } };
            settle();
        },
        // An older page, once drawn, can show the newest message or, on a quiet
        // chain, name it, prove there is none or end the search unknown. The recent
        // page read again (index 0) cannot see past its frozen boundary: its unknown
        // says nothing of what the chain found.
        page({ index, window, coverage }) {
            const absent = window?.latest_absent === true, named = window?.latest_message;
            if (index !== 0 && (absent || named !== undefined) && Number.isSafeInteger(coverage?.upper?.chat)) {
                found = { message: absent ? undefined : named?.history_id ? named : null, coverage };
            }
            requestAnimationFrame(() => { settle(); note(); });
        },
        settle,
    };
}

// The feed the reader can see: its viewport less the chrome drawn over it, the
// header above and the composer below; chrome covering it all leaves an empty
// band, where nothing is on screen. Eviction keeps the whole viewport.
export function readBand(viewport, header, composer) {
    let { top, bottom } = viewport.getBoundingClientRect();
    if (header?.getClientRects?.().length) top = Math.max(top, header.getBoundingClientRect().bottom);
    if (composer?.getClientRects?.().length) bottom = Math.min(bottom, composer.getBoundingClientRect().top);
    return { top, bottom };
}

// What shows a node: itself, or, inside a collapsed task card (no boxes of its
// own), the nearest enclosing card that is drawn. Collapsed cards need not be expanded.
function shownBy(node) {
    for (let shown = node; shown; shown = shown.parentElement?.closest?.('.chat-live-card')) {
        if (shown.getClientRects?.().length) return shown;
    }
    return node;
}

/**
 * How far the feed moves to show the newest message when an unread room is opened
 * (DESIGN "Unread"): it lands at the top of the read band, so what came
 * after it shows below. 0 when it is already on screen or no node of this page can
 * show it: absent (the bottom decides), unknown, or not loaded (the pager reaches it).
 *
 * @param {undefined|null|{history_id: string}} latest
 * @param {{ delivered: (id: string) => boolean, nodes: (id: string) => Array<Element>,
 *   viewport: Element, header?: Element|null, composer?: Element|null }} facts
 */
export function newestMessageOffset(latest, { delivered, nodes, viewport, header, composer }) {
    if (!latest?.history_id || !delivered(latest.history_id)) return 0;
    const band = readBand(viewport, header, composer);
    const node = nodes(latest.history_id).map(shownBy).find((shown) => shown.getClientRects?.().length);
    if (!node || band.bottom <= band.top || historyNodeOnScreen(node, viewport, band)) return 0;
    return node.getBoundingClientRect().top - band.top;
}

/**
 * Whether the reader is at the room's newest message. `latest` is the receipt's
 * newest message (the recent read's `window.latest_message`, or where an older
 * page carried its search on), the standalone message that ARRIVED last:
 * absent names nothing (the room holds no standalone message), so the
 * bottom of the conversation decides; null (its arrival unknown) is never read.
 * A named message — in its ordinary place or not — must have reached the page
 * and must itself be on screen, clear of the header and composer drawn over the
 * feed. Being at the bottom never stands in for it: later card rows and the
 * owner's own messages can push it above the fold, and a late answer keeping its
 * original time (`out_of_order`) is not at the bottom at all. A named message
 * with no node on the page is not on screen.
 *
 * @param {undefined|null|{history_id: string, out_of_order?: boolean}} latest
 * @param {{ delivered: (id: string) => boolean, nodes: (id: string) => Array<Element>,
 *   viewport: Element, header?: Element|null, composer?: Element|null, atBottom: () => boolean }} facts
 */
export function isAtNewestMessage(latest, { delivered, nodes, viewport, header, composer, atBottom }) {
    if (latest === undefined) return atBottom();
    if (!latest?.history_id || !delivered(latest.history_id)) return false;
    const band = readBand(viewport, header, composer);
    return band.bottom > band.top
        && nodes(latest.history_id).some((node) => historyNodeOnScreen(shownBy(node), viewport, band));
}
