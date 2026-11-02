// One message shown in two rooms is read once (DESIGN "Unread"). A room's crimson count is its
// revision past the shared read cursor, less the revisions read in full in another room
// (`read_ahead_revisions`, recorded by the host); no cursor moves for them, so the room's other
// unseen messages stay unread. A room instance reports a message it showed whole: every part of it
// was inside the read band at some moment while the room was shown, by one view or by parts while
// scrolling; a resize starts its reading again, a node with no box shows nothing, and a folded or
// hidden part is never shown. A part that clips what it holds (a code block or table wider or
// taller than its box) shows only its viewport of it: all of its content must have been in view, by
// parts while it scrolls too. Rows an overlay lies over are not in view (`visibleRows`).

import { fetchJson } from './api_client.js';

let spans = {};  // {chat_id: [[from, to], ...]}: the host's answers, joined (a later answer may arrive first)
const listeners = new Set();

/** Unread standalone messages: the revisions past `seen`, less those read elsewhere. */
export function unreadCount(visible, seen, read = []) {
    const top = Math.max(0, Number(visible) || 0), floor = Math.max(0, Number(seen) || 0);
    let count = Math.max(0, top - floor);
    for (const [from, to] of Array.isArray(read) ? read : []) {
        count -= Math.max(0, Math.min(top, Number(to) || 0) - Math.max(floor + 1, Number(from) || 0) + 1);
    }
    return Math.max(0, count);
}

/** The read-ahead revisions of one room (Main is chat 1). */
export const readAheadFor = (chatId) => spans[String(Number(chatId) || 0)] || [];

// Closed revision intervals, joined as the host merges them (adjacent ones join, `merge`).
const joined = (list) => merge(list.filter((item) => Array.isArray(item) && Number.isSafeInteger(item[0])
    && Number.isSafeInteger(item[1]) && item[0] <= item[1]));

/**
 * Adopt the host's recorded read-ahead from a preferences answer; listeners repaint the counts. A record is a
 * durable fact the host drops only once that room's own cursor covers it (which a count never reads below), so
 * answers join: an older answer arriving after a newer one never takes back a message read in full.
 */
export function noteReadAhead(prefs) {
    const next = prefs && typeof prefs === 'object' ? prefs.read_ahead_revisions : undefined;
    if (!next || typeof next !== 'object') return;
    const merged = { ...spans };
    for (const [chat, list] of Object.entries(next)) {
        if (Array.isArray(list)) merged[chat] = joined([...(spans[chat] || []), ...list]);
    }
    if (JSON.stringify(merged) === JSON.stringify(spans)) return;
    spans = merged;
    for (const listener of [...listeners]) listener();
}

/** Repaint on every new read-ahead answer; returns its disposer. */
export function onReadAheadChange(listener) {
    listeners.add(listener);
    return () => listeners.delete(listener);
}

// Closed spans (a node's pixels, a room's revisions), merged; `covers` allows one pixel of rounding at each end.
function merge(list) {
    const out = [];
    for (const [lo, hi] of list.sort((a, b) => a[0] - b[0])) {
        if (out.length && lo <= out.at(-1)[1] + 1) out.at(-1)[1] = Math.max(out.at(-1)[1], hi);
        else out.push([lo, hi]);
    }
    return out;
}
const covers = (list, height) => list.length === 1 && list[0][0] <= 1 && list[0][1] >= height - 1;

// What lies over the feed. A modal layer makes the room under it inert (layers.js) and a modal dialog's backdrop
// covers the page: nothing of the room is in view. A toast, a menu, a hint, the scroll-to-latest button or the
// Project's side sheet over Main conceals the rows it lies over, taken across the whole width (a cut that errs
// safe): what it covered is read at a later reading moment, once it is gone. The sheet a room is shown in is not
// over that room.
const MODAL = '[aria-modal="true"]';
const OVER_ROWS = '.rw-shade, .toast, [role="menu"], .ui-hint, .chat-scroll-bottom-btn, .project-panel.open';
const drawn = (element) => Boolean(element.getClientRects?.().length);

// `{rows, over}`: the band's rows in view and what conceals the rest (an inert ancestor, a modal, an overlay).
function visibleRows(feed, { top, bottom }) {
    if (!(bottom > top)) return { rows: [], over: [] };
    const inert = feed.closest?.('[inert]');
    if (inert) return { rows: [], over: [inert] };
    const doc = feed.ownerDocument;
    if (!doc?.querySelectorAll) return { rows: [[top, bottom]], over: [] };
    const modal = [...doc.querySelectorAll(MODAL)].filter(drawn);
    if (modal.length) return { rows: [], over: modal };
    const { left, right } = feed.getBoundingClientRect();
    let rows = [[top, bottom]];
    const over = [];
    for (const element of doc.querySelectorAll(OVER_ROWS)) {
        if (!drawn(element) || element.contains?.(feed)) continue;
        const box = element.getBoundingClientRect();
        if (box.right <= left || box.left >= right || box.bottom <= top || box.top >= bottom) continue;
        over.push(element);
        rows = rows.flatMap(([lo, hi]) => [[lo, Math.min(hi, box.top)], [Math.max(lo, box.bottom), hi]])
            .filter(([lo, hi]) => hi > lo);
    }
    return { rows, over };
}

// The parts of a message that can clip what they hold (their computed overflow is not `visible`), read once per
// node and size; whether one clips now is read at each moment (a font that loads widens a line).
function overflowing(content) {
    return [content, ...content.querySelectorAll('*')].filter((part) => {
        const { overflowX, overflowY } = getComputedStyle(part);
        return overflowX !== 'visible' || overflowY !== 'visible';
    });
}
const clips = (part) => part.scrollWidth > part.clientWidth + 1 || part.scrollHeight > part.clientHeight + 1;

// A part's client area in the viewport: its box less its border, inside the feed's width and every part that clips
// it (a scrolling table inside a scrolling block shows only what both show).
function clientArea(part, outer, feedBox) {
    const box = part.getBoundingClientRect();
    const left = box.left + (part.clientLeft || 0), top = box.top + (part.clientTop || 0);
    let area = [Math.max(left, feedBox.left), top, Math.min(left + part.clientWidth, feedBox.right), top + part.clientHeight];
    for (const other of outer) {
        const [a, b, c, d] = clientArea(other, [], feedBox);
        area = [Math.max(area[0], a), Math.max(area[1], b), Math.min(area[2], c), Math.min(area[3], d)];
    }
    return area;
}

// Rectangles [x1, y1, x2, y2] of one part's content that were in view, joined where they touch along one axis so
// scrolling one way keeps one rectangle; at most RECTS kept (dropping the oldest only ever leaves a part unread).
const RECTS = 64;
function addRect(rects, rect) {
    if (rects.some(([a, b, c, d]) => a <= rect[0] && b <= rect[1] && c >= rect[2] && d >= rect[3])) return rects;
    let next = rect, kept = rects.filter(([a, b, c, d]) => !(rect[0] <= a && rect[1] <= b && rect[2] >= c && rect[3] >= d));
    for (let i = 0; i < kept.length;) {
        const [a, b, c, d] = kept[i];
        const rows = Math.abs(b - next[1]) <= 1 && Math.abs(d - next[3]) <= 1 && a <= next[2] + 1 && next[0] <= c + 1;
        const columns = Math.abs(a - next[0]) <= 1 && Math.abs(c - next[2]) <= 1 && b <= next[3] + 1 && next[1] <= d + 1;
        if (!rows && !columns) { i += 1; continue; }
        next = [Math.min(a, next[0]), Math.min(b, next[1]), Math.max(c, next[2]), Math.max(d, next[3])];
        kept = [...kept.slice(0, i), ...kept.slice(i + 1)];
        i = 0;
    }
    return [...kept, next].slice(-RECTS);
}

// Whether the rectangles cover a width x height content: every band between their edges is covered across.
function coversArea(rects, width, height) {
    const edges = [...new Set([0, height, ...rects.flatMap(([, y1, , y2]) => [y1, y2])])]
        .filter((y) => y >= 0 && y <= height).sort((a, b) => a - b);
    for (let i = 0; i + 1 < edges.length; i += 1) {
        const lo = edges[i], hi = edges[i + 1];
        if (hi - lo <= 1) continue;  // a pixel of rounding
        const across = rects.filter(([, y1, , y2]) => y1 <= lo + 1 && y2 >= hi - 1).map(([x1, , x2]) => [x1, x2]);
        if (!covers(merge(across), width)) return false;
    }
    return edges.length > 1;
}

/**
 * A room instance's record of shared messages it showed whole. `feed` holds the bubbles (each
 * stamped `data-row-address` and, for work speech, `data-room-revisions`); `band()` is the visible
 * reading area (`project_read_state.readBand`); `shown()` says the room is the one on screen.
 * `sample()` runs at every reading moment the receipt notes (scroll edges, paints, arrivals), when a
 * part inside the feed scrolls, and when what concealed a message not read yet goes away (its own
 * listener and observer; `dispose` removes both).
 */
export function createSharedReadTracker({ chatId, feed, band, shown, post = postSharedRead }) {
    const room = String(Number(chatId) || 0);
    const seen = new Map();  // address -> {size, spans, parts: [{rects}], node, overflow}
    const sent = new Set();
    let watch = null;  // while something conceals a shared message not read yet: its end is a reading moment
    const unwatch = () => { watch?.disconnect(); watch = null; };
    function conceal(over) {
        if (watch || !over.length || typeof MutationObserver !== 'function') return;
        watch = new MutationObserver(() => {
            unwatch();
            (globalThis.requestAnimationFrame || setTimeout)(sample);  // once painted without it; still concealed: again
        });
        for (const element of over) {
            watch.observe(element, { attributes: true, attributeFilter: ['inert', 'hidden', 'class', 'style'] });
            if (element.parentNode) watch.observe(element.parentNode, { childList: true });
        }
    }
    function sample() {
        if (!feed || !shown()) return;
        const { rows, over } = visibleRows(feed, band());
        const shared = [];
        for (const node of feed.querySelectorAll('[data-room-revisions][data-row-address]')) {
            let rooms;
            try { rooms = Object.keys(JSON.parse(node.dataset.roomRevisions)); } catch { continue; }
            if (!sent.has(node.dataset.rowAddress) && rooms.some((key) => key !== room)) shared.push(node);  // else the cursor reads it
        }
        if (shared.length && over.length) conceal(over);
        if (!rows.length) return;
        const feedBox = feed.getBoundingClientRect();
        for (const node of shared) {
            const address = node.dataset.rowAddress;
            if (sent.has(address)) continue;
            if (node.closest('.is-folded, [hidden], details:not([open])')) continue;  // a folded part is not shown
            const box = node.getBoundingClientRect();
            if (!(box.height > 0)) continue;
            const inView = rows.map(([top, bottom]) => [Math.max(top, box.top), Math.min(bottom, box.bottom)])
                .filter(([lo, hi]) => hi > lo);
            if (!inView.length) continue;
            const content = node.querySelector(':scope > .message');
            let entry = seen.get(address);
            const overflow = content ? (entry?.node === node && entry.width === box.width && Math.abs(entry.height - box.height) <= 1 ? entry.overflow
                : overflowing(content)) : [];
            const clipping = overflow.filter(clips);
            const size = JSON.stringify([box.width, Math.round(box.height), clipping.map((part) => [part.scrollWidth,
                part.scrollHeight, part.clientWidth, part.clientHeight])]);
            if (!entry || entry.size !== size || entry.node !== node) {  // a resize is a new reading
                entry = { size, width: box.width, height: box.height, node, overflow, spans: [], parts: clipping.map(() => ({ rects: [] })) };
            }
            entry.spans = merge([...entry.spans, ...inView.map(([lo, hi]) => [lo - box.top, hi - box.top])]);
            clipping.forEach((part, index) => {  // the content under its client area, in its own coordinates
                const partBox = part.getBoundingClientRect();
                const originX = partBox.left + (part.clientLeft || 0) - part.scrollLeft;
                const originY = partBox.top + (part.clientTop || 0) - part.scrollTop;
                const [left, top, right, bottom] = clientArea(part, clipping.filter((other) => other !== part
                    && other.contains?.(part)), feedBox);
                for (const [rowTop, rowBottom] of inView) {
                    const lo = Math.max(top, rowTop), hi = Math.min(bottom, rowBottom);
                    if (!(hi > lo && right > left)) continue;  // horizontal scrolling never shows rows out of view
                    entry.parts[index].rects = addRect(entry.parts[index].rects,
                        [left - originX, lo - originY, right - originX, hi - originY]);
                }
            });
            seen.set(address, entry);
            const whole = covers(entry.spans, entry.height) && !content?.querySelector('[hidden], details:not([open])')
                && clipping.every((part, index) => coversArea(entry.parts[index].rects, part.scrollWidth, part.scrollHeight));
            if (whole) {
                sent.add(address);
                seen.delete(address);  // receipt identity is enough; retain no completed DOM subtree
                void post(address).then((ok) => { if (!ok) { sent.delete(address); seen.delete(address); } });
            }
        }
    }
    // A part inside the feed that scrolls (a wide table, a long code block) moves what it shows: a reading moment.
    const inner = (event) => { if (event.target !== feed) sample(); };
    feed?.addEventListener?.('scroll', inner, { capture: true, passive: true });
    return {
        sample,
        release(node) {
            const address = node?.dataset?.rowAddress;
            if (seen.get(address)?.node === node) seen.delete(address);
            sent.delete(address);
        },
        dispose() {
            feed?.removeEventListener?.('scroll', inner, { capture: true });
            unwatch();
            seen.clear();
            sent.clear();
        },
    };
}

async function postSharedRead(address) {
    try {
        noteReadAhead(await fetchJson('/api/ui/preferences', {
            method: 'POST', headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ shared_read: [address] }),
        }));
        return true;
    } catch {
        return false;  // still unread elsewhere: the next whole showing offers it again
    }
}
