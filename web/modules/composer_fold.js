/** The composer's row of capsules folds while the owner reads back through a room's history (owner 19.10, 2A;
 * docs/DESIGN.md "Composer owner controls"), and in a keyboard room too short for it (owner 9.11).
 *
 * On a phone (the stack layout) the row (☰, Swarm, Context, Effort) takes its own line above the field. While the
 * owner scrolls back through the history it folds away: the composer keeps only the field, and ☰, still the row's
 * first control, waits beside it on its last line, beside "+". Scrolling toward the latest, reaching it, focusing the field or opening a capsule
 * brings the row back, and so does any other layout. Like Safari's bars, a direction decides: FOLD.run px one way.
 * A press on ☰, ↓, 📎 or Send is that control's own: the row stays as it is, so nothing slides under the finger
 * before the tap lands. While the owner writes, or the field holds the focus, the row never folds, but in a short
 * keyboard room (style.css --keyboard-short: Telegram fullscreen's band over a tall keyboard) the row would leave the
 * feed no line of a message: there it folds while the keyboard is up, the field keeping its focus, until the keyboard
 * goes down or the room grows. The room says so by resizing the feed, and the capsules' owner once a capsule has
 * settled closed; a capsule being worked keeps the row.
 *
 * Only the owner's own scroll is a step. It is read from where the newest row stands below the feed's visible
 * bottom, and only between two reads of the same content: older history paged in or let go, a message or a newer
 * page arriving, a row growing and the composer's own reserve change the feed's content height, so the read that
 * sees them starts again from there and neither folds the row nor brings it back by itself. Only while the owner's
 * own gesture has taken the room off its newest row (`readingBack`, chat_reading_position.js) is a scroll the owner
 * reading back: a swipe begun in a card's frame takes its touches away from the feed (a fold then would resize the
 * composer and lay the still-following feed out at its newest row again), and the app's own navigation (the waiting
 * pill's jump to a question) would fold the very pill the owner steps with.
 * style.css draws the fold from two facts on the composer: the row's measured height (`--fold-row-h`, so the line
 * collapses from what it was) and how far ☰ goes down to the field's last line, where "+" stands (`--fold-menu-dy`:
 * read at every fold, a reply staged inside the field making it taller, and again whenever the folded field's height
 * changes, as a draft of several lines wraps anew in the narrower field); `data-fold` holds from the fold until the row
 * is back, `data-folded` while it is folded. */

import { isStackLayout, onLayoutChange } from './layout.js';

export const FOLD = Object.freeze({
    run: 40,      // px read back (or forward) in one direction before the row folds (or comes back)
    away: 80,     // px the newest row lies below the visible bottom before the row may fold
    latest: 24,   // px from the scroll's end: the row is always there
    settle: 340,  // ms of the fold's own motion (style.css), after which an unfolded row lets go of its height
});

// The short keyboard room's step as the composer inherits it (style.css registers it as a length: 0px or 1px).
const shortRoom = (win, el) => parseFloat(win?.getComputedStyle?.(el)?.getPropertyValue?.('--keyboard-short')) > 0;

export function createComposerFold({ feed, wrap, row, win = globalThis.window, doc = globalThis.document,
    isStack = () => isStackLayout(), onLayout = (sync) => onLayoutChange(() => sync(), { w: win }),
    readingBack = () => true, short = () => shortRoom(win, wrap) } = {}) {
    if (!feed || !wrap || !row) return { destroy() {}, sync() {}, folded: () => false };
    let destroyed = false;
    let folded = false;
    // Folded for the short keyboard room, not for reading back: only the room's own change brings this fold back.
    let tight = false;
    // The last read: how far the newest row lay below the visible bottom, and the feed's content height then.
    let last = { away: null, height: 0 };
    let run = 0;
    let releaseTimer = 0;
    let frame = 0;
    let roomFrame = 0;
    // ☰'s way down: the depth of ☰'s centre on the line, measured when the line is first held, and, read at every fold,
    // what lies between the line and the field and how far above the field's bottom its last line runs (the centre of
    // "+", which rides on it; the field holds a staged reply's strip, chat_reply.js).
    let aim = null;
    const box = wrap.querySelector?.('.chat-field');
    const plus = wrap.querySelector?.('[data-composer-plus]');
    const field = wrap.querySelector?.('textarea');

    // Where the newest row with a box stands (the typing indicator sits last and is hidden most of the time): how far
    // it lies below the feed's visible bottom (negative: above it), and the feed's content height.
    function measure() {
        let newest = feed.lastElementChild;
        while (newest && !newest.getClientRects?.().length) newest = newest.previousElementSibling;
        const away = newest?.getBoundingClientRect
            ? newest.getBoundingClientRect().bottom - feed.getBoundingClientRect().bottom : null;
        return { away, height: feed.scrollHeight };
    }
    const atLatest = () => feed.scrollHeight - feed.clientHeight - feed.scrollTop < FOLD.latest;
    const keyboard = () => Boolean(doc?.body?.classList?.contains?.('keyboard-open'));
    // A capsule is worked: open, focused or with the line alone. ☰ stays in sight folded, so its focus (the drawer gives
    // it back on closing) keeps nothing.
    const working = () => {
        const focused = doc?.activeElement;
        const menu = row.querySelector?.(':scope > .composer-menu');
        return Boolean(focused && row.contains?.(focused) && !menu?.contains?.(focused))
            || Boolean(row.dataset?.ownerSolo) || Boolean(row.querySelector?.('[data-open="true"]'));
    };
    // While the owner writes or works a capsule the row stays: the keyboard is up (app.js's keyboard-open), the field
    // holds the focus (with or without a keyboard), or a capsule is worked.
    const busy = () => keyboard() || Boolean(field && doc?.activeElement === field) || working();
    // But a short room with the keyboard up has no line of a message to spare for the row (owner 9.11).
    const cramped = () => keyboard() && Boolean(isStack?.()) && short() && !working();

    // Folded, the line starts where what lay between it and the field does (the wrap's gap goes with the line), and ☰
    // keeps its place on the line: it goes down by that, then by the field's height less the last line's rise above the
    // field's bottom and its own centre's depth on the line, to the field's last line beside "+".
    function aimMenu(fieldHeight) {
        if (aim) wrap.style.setProperty('--fold-menu-dy', `${Math.round(aim.between + fieldHeight - aim.rise - aim.depth)}px`);
    }

    function release() {
        delete wrap.dataset.fold;
        aim = null;
        wrap.style.removeProperty('--fold-row-h');
        wrap.style.removeProperty('--fold-menu-dy');
    }

    function fold(forRoom = false) {
        tight = forRoom;
        if (folded) return;
        folded = true;
        if (releaseTimer) { win?.clearTimeout?.(releaseTimer); releaseTimer = 0; }
        const line = row.getBoundingClientRect();
        const field = box?.getBoundingClientRect?.();
        // The line collapses from the height it has: measured and held before the fold starts.
        if (!('fold' in wrap.dataset)) {
            wrap.style.setProperty('--fold-row-h', `${Math.ceil(line.height)}px`);
            const menu = row.querySelector?.(':scope > .composer-menu')?.getBoundingClientRect?.();
            aim = menu ? { between: 0, rise: 0, depth: menu.top + menu.height / 2 - line.top } : null;
            wrap.dataset.fold = '';
            void wrap.offsetHeight;  // the held height is the transition's start
        }
        // A fold inside the hold (the row back for less than FOLD.settle) keeps the held line, but a reply staged or
        // cancelled meanwhile (each brings the row back first) has changed the field: it is read again at every fold.
        // Read past the line's height and margin as they stand now, it is the same mid-motion as at rest. The last
        // line's rise is "+"'s centre above the field's bottom (half the field without "+").
        if (aim && field) {
            const gap = parseFloat(win?.getComputedStyle?.(wrap)?.rowGap) || 0;
            const margin = parseFloat(win?.getComputedStyle?.(row)?.marginBottom) || 0;
            const mark = plus?.getBoundingClientRect?.();
            aim.between = Math.max(0, field.top - (line.top + line.height + margin) - gap);
            aim.rise = mark ? field.top + field.height - (mark.top + mark.height / 2) : field.height / 2;
            aimMenu(field.height);
        }
        wrap.dataset.folded = '';
    }

    function unfold() {
        tight = false;
        if (!folded) return;
        folded = false;
        delete wrap.dataset.folded;
        // Once back, the row lets go of its held height (a strip that opens, a longer name, a larger text).
        releaseTimer = win?.setTimeout?.(() => { releaseTimer = 0; if (!folded) release(); }, FOLD.settle) || 0;
    }

    function read() {
        frame = 0;
        if (destroyed) return;
        const now = measure();
        // Between two reads of the same content only the scroll moved the newest row; otherwise this read is a new
        // start (history paged in or let go, rows arriving or growing, the reserve), never the owner's step.
        const prev = now.height === last.height ? last.away : null;
        last = now;
        if (cramped()) { run = 0; fold(true); return; }
        if (!isStack?.() || busy()) { run = 0; unfold(); return; }
        if (atLatest() || !readingBack()) { run = 0; unfold(); return; }
        if (now.away === null || prev === null) return;
        const step = now.away - prev;  // > 0: the newest row went further down, the owner reads back
        if (!step) return;
        run = Math.sign(step) === Math.sign(run) ? run + step : step;
        if (run >= FOLD.run && now.away > FOLD.away) fold();
        else if (run <= -FOLD.run) unfold();
    }
    // One read per frame, however many scroll events the frame brings.
    const onScroll = () => {
        if (frame || destroyed) return;
        frame = typeof win?.requestAnimationFrame === 'function' ? win.requestAnimationFrame(read) : (read(), 0);
    };
    // The room itself changed (the keyboard up or down, the visual viewport): the feed's box resized. A short keyboard
    // room folds the row; a roomier one brings back the fold it made, never one of the owner's reading back. Neither is
    // a step of the owner's, so the next read starts from here.
    function sync() {
        roomFrame = 0;
        if (destroyed) return;
        if (cramped()) fold(true);
        else if (tight) unfold();
        else return;
        run = 0;
        last = measure();
    }
    const onRoom = () => {
        if (roomFrame || destroyed) return;
        roomFrame = typeof win?.requestAnimationFrame === 'function' ? win.requestAnimationFrame(sync) : (sync(), 0);
    };
    // Focusing the field (the owner is about to write; a folded row means it did not hold the focus) brings the row
    // back, but in a short keyboard room. The focus or press of another control in the composer is that control's own
    // and moves nothing.
    const onFocus = (event) => {
        if (!field || event?.target !== field) return;
        run = 0;
        if (cramped()) fold(true);
        else unfold();
    };

    feed.addEventListener('scroll', onScroll, { passive: true });
    wrap.addEventListener('focusin', onFocus);
    // The feed's box tells the room's changes. Folded, the field is narrower by ☰'s place, so a draft of several lines
    // may take one more (its bottom stays where it was), and the mic, a busy Send or words put in change its height
    // too: ☰ follows its last line.
    const resized = typeof win?.ResizeObserver === 'function' ? new win.ResizeObserver((entries) => {
        for (const entry of entries || []) {
            if (entry.target === feed) onRoom();
            else if (folded && box) aimMenu(box.offsetHeight);
        }
    }) : null;
    resized?.observe(feed);
    if (box) resized?.observe(box);
    const unwatchLayout = onLayout(() => { run = 0; last = measure(); unfold(); });

    return {
        folded: () => folded,
        // A capsule closed by a press the field keeps its focus through moves neither the feed nor the focus: the
        // capsules' owner asks again once it has settled closed (composer_owner_controls.js onSettled).
        sync: onRoom,
        destroy() {
            if (destroyed) return;
            destroyed = true;
            feed.removeEventListener('scroll', onScroll);
            wrap.removeEventListener('focusin', onFocus);
            resized?.disconnect();
            if (typeof win?.cancelAnimationFrame === 'function') {
                if (frame) win.cancelAnimationFrame(frame);
                if (roomFrame) win.cancelAnimationFrame(roomFrame);
            }
            if (releaseTimer) win?.clearTimeout?.(releaseTimer);
            unwatchLayout?.();
            delete wrap.dataset.folded;
            release();
        },
    };
}
