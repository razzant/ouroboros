// Rows that sleep (owner 20.10, 3A step 3). The browser skips a feed row far from the screen (content-visibility:
// auto under [data-laid], style.css) only once that row was laid out at the feed's width with everything in it: the
// size it keeps while skipped (contain-intrinsic-size: auto) is then its real size, and the passage being read never
// moves when it wakes (tests/test_chat_lazy_rows_browser.py). Skipped rows keep a long history cheap to move: the room
// card sliding under the drawer no longer restyles and repaints every row.
//
// A row sleeps a moment after it arrived, after it last changed, and after every picture in it has arrived (a picture
// still loading has no size yet, so its row stays awake until it loads or fails). A row that changes wakes at once and
// is laid out like any other row; inside the reading owner's write (chat_reading_position.js `mutate`) the owner
// settles it before it restores the passage (`flush`), so the place it restores is measured with the row's new size.
// A new width, new fonts, another theme, layout or language wake every row (a row's size can depend on any of them:
// the dark theme's recorded answer has a line the light one has not); they sleep again once the width has held.
// A feed that is not shown lays nothing out: its rows wait until it is.
//
// A row holding a part sized by the window's height (a review's long detail or full body, a widget's JSON or chart:
// style.css `vh`) never sleeps: only the window's height changes it, which wakes nothing else.
//
// Only message rows sleep (a bubble, a work card), and never within a screen of the feed's view: a skipped row is
// contained (layout, style and paint), and WebKit's scroll anchoring does not hold the passage being read when rows
// around it are contained (a row above the passage growing outside a write, a work card's buttons arriving after an
// older page, moved the passage by their height); a contained row also clips what reaches past it, such as a
// control's larger touch area. Near the view the browser renders every row anyway: nothing is lost there.

const MARK = 'data-laid';
const ADDED_FRAMES = 2;     // a row added is laid out by the next frame; it sleeps after the one after
const CHANGED_FRAMES = 30;  // a row that changes often (a streaming answer, a live card) stays awake while it does
const HELD_FRAMES = 12;     // a width that held this long has stopped moving (chat_reading_position.js)
// What changes a row's size besides its text and children: its classes, a part hidden or opened, an inline size,
// a picture's source, a note a deck tucks away or brings back, the ledges a stack's front card reserves room for
// (chat_relay_deck.js).
const RESIZING = ['class', 'hidden', 'open', 'style', 'src', 'data-deck-tucked', 'data-deck-edges'];
// What on the page's root restyles every row: the theme, the layout, the language and its direction.
const RESTYLING = ['data-theme', 'data-layout', 'lang', 'dir'];
const ROWS = '.chat-bubble, .chat-live-card';  // the rows that sleep; a history control or notice never does
const NEAR = '100% 0px';  // within the feed's own height above and below its view
const WINDOW_SIZED = '.review-facts[open] > .review-facts-body, .skill-review-full, .widget-json pre, .widget-chart-canvas';

// `onRestyle`: the room's reading owner places the passage again once a restyle (theme, layout, language, fonts) has
// woken every row: rows far from the view change size at once, which WebKit's scroll anchoring does not absorb.
export function createRowSleep({ feed, win = globalThis.window, onRestyle } = {}) {
    if (!feed || typeof win?.requestAnimationFrame !== 'function' || typeof win.MutationObserver !== 'function') {
        return { flush() {}, destroy() {} };
    }
    const doc = feed.ownerDocument;
    const due = new Map();  // row → the frame it may sleep at
    const waiting = new WeakSet();  // rows awake until a picture in them arrives
    // The feed's width and whether it is shown, as its ResizeObserver last saw them (read there, after layout: a
    // frame callback never forces one). Without the observer the feed is taken as shown at its first width.
    const observes = typeof win.ResizeObserver === 'function';
    let destroyed = false, frame = 0, ticking = false, held = HELD_FRAMES;
    let width = observes ? 0 : feed.clientWidth, shown = !observes;

    const rowOf = (node) => {
        let el = node?.nodeType === 1 ? node : node?.parentElement;
        while (el && el.parentElement !== feed) el = el.parentElement;
        return el || null;
    };
    const queue = (row, frames) => {
        due.set(row, Math.max(due.get(row) || 0, frame + frames));
        if (!ticking) { ticking = true; win.requestAnimationFrame(tick); }
    };
    const wake = (row, frames = CHANGED_FRAMES) => {
        if (row.hasAttribute(MARK)) row.removeAttribute(MARK);
        queue(row, frames);
    };
    const wakeAll = () => { for (const row of feed.children) wake(row, ADDED_FRAMES); };

    // The rows near the view, which never carry the mark; one that leaves may sleep again.
    const near = new Set();
    const watch = typeof win.IntersectionObserver === 'function' ? new win.IntersectionObserver((entries) => {
        for (const entry of entries) {
            if (entry.isIntersecting) {
                near.add(entry.target);
                if (entry.target.hasAttribute(MARK)) entry.target.removeAttribute(MARK);
            } else if (near.delete(entry.target)) queue(entry.target, ADDED_FRAMES);
        }
    }, { root: feed, rootMargin: NEAR }) : null;
    const arrived = (row) => {
        if (!row.matches?.(ROWS)) return;
        watch?.observe(row);
        queue(row, ADDED_FRAMES);
    };

    function sleep(row) {
        if (near.has(row) || !row.matches?.(ROWS) || row.querySelector?.(WINDOW_SIZED)) return;
        // A picture or a video whose size has not arrived yet (a video's box is 300x150 until its metadata).
        const pictures = [...row.querySelectorAll('img, video')]
            .filter((media) => (media.tagName === 'VIDEO' ? media.readyState === 0 : !media.complete));
        if (!pictures.length) { row.setAttribute(MARK, ''); return; }
        if (waiting.has(row)) return;
        waiting.add(row);
        const loaded = () => {
            if (!waiting.delete(row) || destroyed || row.parentElement !== feed) return;
            queue(row, ADDED_FRAMES);
        };
        for (const media of pictures) {
            media.addEventListener(media.tagName === 'VIDEO' ? 'loadedmetadata' : 'load', loaded, { once: true });
            media.addEventListener('error', loaded, { once: true });
        }
    }

    function tick() {
        // A feed that is not shown lays nothing out: the count stops, and its observer starts it again when it shows.
        if (destroyed || !shown) { ticking = false; return; }
        frame += 1;
        // Nothing sleeps while the width moves (the reading owner's lazy rows): a row that was not laid out at this
        // width would keep a size that is not its own.
        const settled = !feed.hasAttribute('data-resizing') && ++held >= HELD_FRAMES;
        if (settled) {
            const ready = [];
            for (const [row, at] of due) {
                if (at > frame) continue;
                due.delete(row);
                if (row.parentElement === feed) ready.push(row);
            }
            // A row without a box (`display: none`, a deck's tucked note) has no size to remember: marked, it would wake at
            // the placeholder. Every box is read before any mark is written, so the batch costs one layout, not one per row.
            const boxed = ready.filter((row) => typeof row.getClientRects !== 'function' || row.getClientRects().length > 0);
            for (const row of boxed) sleep(row);
        }
        if (due.size) win.requestAnimationFrame(tick);
        else ticking = false;
    }

    const settle = (records) => {
        for (const record of records) {
            if (record.target === feed) {
                for (const node of record.addedNodes) if (node.nodeType === 1) arrived(node);
                continue;
            }
            const row = rowOf(record.target);
            if (row) wake(row);
        }
    };
    const changes = new win.MutationObserver(settle);
    changes.observe(feed, { childList: true, subtree: true, characterData: true, attributes: true,
        attributeFilter: RESIZING });
    const restyle = () => { wakeAll(); onRestyle?.(); };
    const restyled = doc?.documentElement ? new win.MutationObserver(restyle) : null;
    restyled?.observe(doc.documentElement, { attributes: true, attributeFilter: RESTYLING });

    // A new width wakes every row: each is laid out again at it. A hidden feed reads 0 and keeps its rows as they are;
    // shown again at its width, the rows that arrived meanwhile (never laid out) sleep once they have been.
    const resized = observes ? new win.ResizeObserver(() => {
        const now = feed.clientWidth;
        shown = now > 0;
        if (!shown) return;
        if (width && now !== width) { width = now; held = 0; wakeAll(); return; }
        width = now;
        if (due.size && !ticking) { ticking = true; win.requestAnimationFrame(tick); }
    }) : null;
    resized?.observe(feed);
    doc?.fonts?.addEventListener?.('loadingdone', restyle);

    for (const row of feed.children) arrived(row);

    return {
        // The changes written so far wake their rows now, not at the next microtask: a reader's place measured right
        // after a write sees every changed row at its new size.
        flush() { if (!destroyed) settle(changes.takeRecords()); },
        destroy() {
            if (destroyed) return;
            destroyed = true;
            changes.disconnect();
            restyled?.disconnect();
            watch?.disconnect();
            resized?.disconnect();
            doc?.fonts?.removeEventListener?.('loadingdone', restyle);
            due.clear();
        },
    };
}
