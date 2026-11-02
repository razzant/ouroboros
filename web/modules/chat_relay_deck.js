// Notes between works (DESIGN "Replies"): a relay row's bubble is the note card itself and keeps its row (address,
// history identity, actions, reading anchors). Its head is an address, sender → receiver · time, so a reader sees at
// once that this is not a reply and which room wrote to which. Consecutive notes lie as one stack the way grouped
// notifications do: the newest card in front, up to two older cards peeking BELOW it as ledges inside the card's own
// box, ordinary air above and below; a blue mark names the tucked cards with the owner's words. The capsule `N notes`
// fans the stack downward into a column and `Show less` on the column's first card folds it, while one more / less
// control opens a card's note as written and all its quotations together inside a stable-viewport write.
import { fmt, tr } from './i18n.js';
import { formatMsgTime } from './utils.js';
import { fitText, fullText } from './text_fit.js';
import { motionMs, reducedMotion } from './motion.js';
import { projectReference } from './project_reference.js';
import { fetchJson } from './api_client.js';
import { RELAY_CARD_LINES } from './chat_reply_quote.js';

let nextNoteId = 0;
const EDGES = 2;        // at most two ledges behind the front card
const MAIN_CHAT = 1;
const LEDGE_INSET = 12; // one stack level: 12px narrower per side, 8px lower (style.css draws it; the fold animates it)
const LEDGE_DROP = 8;
// Everything that moves here rides the one sheet curve, the emerging cards of a fanned stack included (owner question
// 5, B: no overshoot anywhere).
const SHEET_CURVE = 'cubic-bezier(0.32, 0.72, 0, 1)';  // --ease-sheet's own value, should the stylesheet not answer
const ARROW = '<svg viewBox="0 0 16 16" width="14" height="14" fill="none" stroke="currentColor" stroke-width="1.5"'
    + ' stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M2.5 8h11"></path>'
    + '<path d="M9.5 4.5 13.5 8l-4 3.5"></path></svg>';

const node = (tag, cls, text) => {
    const el = document.createElement(tag);
    if (cls) el.className = cls;
    if (text != null) el.textContent = text;
    return el;
};
const control = (cls, text) => {
    const el = node('button', `ui-hit-area ${cls}`, text);
    el.type = 'button';
    return el;
};
const layer = (cls, level = 0) => {
    const el = node('span', cls);
    el.setAttribute('aria-hidden', 'true');
    if (level) el.dataset.level = String(level);
    return el;
};
const setData = (el, key, value) => {
    if (value == null || value === false) delete el.dataset[key];
    else el.dataset[key] = String(value);
};
const isNote = (el) => Boolean(el?.__relayNote);
const transparent = (el) => el.hidden || Boolean(el.classList?.contains('typing-bubble'));
const thisWork = () => tr('chat.relay.this_work', 'this work');

/**
 * One instance per room feed (chat_reply.js creates it beside the replies). `chatId` is the room (Main is 1): it decides
 * how the address reads. `onDomWrite` is the room's stable-viewport seam: every change that moves heights runs inside
 * it. `reading()` is the room's reading-position owner, `clearance()` the height of what covers the feed's top, `band()`
 * the part of the feed a reader sees (its bottom is the composer's top where the composer is drawn over the feed), and
 * `place(row, { follow })` the room's one explicit placement of a row under its header (or its return to the newest
 * message): the deck decides when a fanned column asks for it, never how the feed scrolls.
 */
export function createRelayDecks({ feed, chatId = 0, onDomWrite = (write) => write(), formatTime = formatMsgTime,
    reading = null, clearance = () => 72, band = null, place = null,
    lookupProject = async (room, signal) => {
        const data = await fetchJson(`/api/projects?chat_id=${encodeURIComponent(room)}`, { signal });
        return data?.projects?.length === 1 ? data.projects[0] : null;
    } } = {}) {
    const rooms = new Map(), lookup = new AbortController();
    const fresh = new Set();     // notes adopted since the last reconcile
    let scheduled = false, settled = false, destroyed = false;
    const inMain = Number(chatId) === MAIN_CHAT;

    const timing = () => {
        let ease = '';
        try {
            ease = globalThis.window.getComputedStyle(globalThis.window.document.documentElement).getPropertyValue('--ease-sheet').trim();
        } catch { ease = ''; }
        return { open: motionMs('--motion-open'), close: motionMs('--motion-close'), ease: ease || SHEET_CURVE };
    };
    // Position and opacity only. Reduced motion keeps a short fade of what fades and lets nothing travel; no engine
    // plays nothing.
    const play = (el, frames, duration, { delay = 0, easing = '' } = {}) => {
        if (!el || typeof el.animate !== 'function' || !duration) return null;
        let keyframes = frames;
        if (reducedMotion()) {
            keyframes = frames.filter((frame) => 'opacity' in frame).map(({ opacity }) => ({ opacity }));
            duration = motionMs('--motion-fade');
            delay = 0;
            if (keyframes.length < 2 || !duration) return null;
        }
        try { return el.animate(keyframes, { duration, delay, easing: easing || timing().ease, fill: 'backwards' }); } catch { return null; }
    };

    /* ---- a note: the relay bubble dressed as a card ------------------------------------------------------------- */
    function noteOf(row, quoted, bubble) {
        const relay = row?.relay && typeof row.relay === 'object' ? row.relay : {};
        const sender = relay.sender && typeof relay.sender === 'object' ? relay.sender : null;
        const target = relay.target && typeof relay.target === 'object' ? relay.target : null;
        // A note saved before its room was recorded: Main when it came from Main, never an invented room.
        const fromMain = sender ? Number(sender.chat_id) === MAIN_CHAT : !quoted && Number(relay.source?.chat_id) === MAIN_CHAT;
        const saved = String(sender?.name || '').trim();
        // A Main work writing where the reader is also in Main: the board's phrase, never `Main → Main`.
        const name = fromMain ? (inMain ? tr('chat.relay.another_main_work', 'Another Main work') : 'Main')
            : saved || tr('chat.relay.another_work', 'Another work');
        const time = formatTime(row?.ts) || null;
        return {
            quoted, chatId: Number(sender?.chat_id),
            words: String(row?.text ?? bubble.querySelector(':scope > .message')?.textContent ?? '').trim(),
            summary: String(relay.summary || '').trim(),
            name, kind: fromMain ? 'main' : 'project', time: time?.short || '', when: time?.full || '',
            // The work the note was written for, as the host stamped it (new rows; an old row says `this work`).
            targetId: String(target?.task_id || ''), targetTitle: String(target?.title || '').trim(),
            room: `${fromMain ? MAIN_CHAT : Number(sender?.chat_id) || ''}|${saved}`,
        };
    }

    // The receiver: `this work` in a work's room; in Main the work the note names. A closed stack says the count of
    // works when its notes name several, the shared name only when every note names that one work, else `this work`:
    // never one card's work as the stack's.
    const receiver = (note, stack = null) => {
        if (!inMain) return thisWork();
        if (stack) {
            if (stack.works > 1) return fmt('{n} works', { n: stack.works });
            if (!stack.shared) return thisWork();
        }
        return note.targetTitle || thisWork();
    };
    const worksOf = (run) => {
        const ids = run.map((bubble) => bubble.__relayNote.note.targetId);
        const known = new Set(ids.filter(Boolean));
        return { works: known.size, shared: known.size === 1 && ids.every(Boolean) };
    };

    function dress(bubble, note) {
        bubble.classList.add('chat-relay-note');
        bubble.dataset.kind = note.kind;
        bubble.dataset.raw = '0';
        // The card's material and the cards behind it: layers inside the row's own box (style.css draws them under the
        // content; nothing paints above the front card), never a second row.
        bubble.prepend(layer('chat-relay-ledge', 2), layer('chat-relay-ledge', 1), layer('chat-relay-face'));
        // The address: sender → receiver, with the time and the stack's control as one unit at the right edge.
        const head = node('div', 'chat-relay-card-head');
        const from = node('span', 'chat-relay-from', note.name);
        const extra = node('span', 'chat-relay-rooms');
        extra.hidden = true;
        const dest = node('span', 'chat-relay-dest');
        const arrow = node('span', 'chat-relay-arrow');
        arrow.setAttribute('aria-hidden', 'true');
        arrow.innerHTML = ARROW;
        const to = node('span', 'chat-relay-to', receiver(note));
        dest.append(arrow, to);
        const side = node('span', 'chat-relay-side');
        const time = node('time', 'chat-relay-time', note.time);
        if (note.when) time.title = note.when;
        const act = control('chat-relay-act btn btn-default btn-sm', '');
        act.hidden = true;
        side.append(time, act);
        head.append(from, extra, dest, side);
        bubble.__relayNote.head = { from, extra, to, time, act };
        // Only a host-resolved Project gets a reference. A missing/failed lookup stays an honest plain name.
        // Cache per room feed, and cancel pending reads when that feed is disposed.
        if (Number.isInteger(note.chatId) && note.chatId > 1) {
            if (!rooms.has(note.chatId)) rooms.set(note.chatId, Promise.resolve()
                .then(() => lookupProject(note.chatId, lookup.signal)).catch(() => null));
            rooms.get(note.chatId).then((project) => {
                if (destroyed || !project?.id || Number(project.chat_id) !== note.chatId) return;
                onDomWrite(() => { from.replaceWith(projectReference(project)); return true; });
            });
        }
        const sender = bubble.querySelector(':scope > .sender');
        bubble.insertBefore(head, sender || bubble.querySelector(':scope > .message'));
        const excerpt = node('div', 'chat-relay-excerpt');
        const gist = node('p', 'chat-relay-gist');
        if (!note.summary) gist.dataset.raw = '1';
        const more = control('chat-relay-more', tr('chat.relay.more', 'more'));
        more.setAttribute('aria-expanded', 'false');
        const id = `chat-relay-text-${++nextNoteId}`;
        gist.id = `${id}-gist`;
        const message = bubble.querySelector(':scope > .message');
        if (message) message.id ||= id;
        const quotes = [...bubble.querySelectorAll('.chat-quote[data-mode="relay"] > .chat-quote-text')];
        quotes.forEach((text, i) => { text.id ||= `${id}-quote-${i}`; });
        more.setAttribute('aria-controls', [gist.id, message?.id, ...quotes.map(text => text.id)].filter(Boolean).join(' '));
        bubble.__relayNote.more = more;
        excerpt.append(gist, more);
        const quote = bubble.querySelector(':scope > .chat-quote, :scope > .chat-relay-quotes');
        (quote || sender || head).after(excerpt);
        fitText(gist, note.summary || note.words, 3);
        if (message) bubble.insertBefore(node('div', 'chat-relay-raw-label', tr('chat.relay.as_written', 'Note as written')), message);
        bubble.addEventListener('click', (event) => onPress(bubble, event));
    }

    // `22:27–02:13` for a stack of today's notes; a shared day said once (`Oct 19, 22:27–23:13`), two days both
    // (`Oct 19, 22:27–Oct 20, 02:13`): the formatter's own compact clock, with its day before the comma.
    function range(run) {
        const a = run[0].__relayNote.note.time, b = run[run.length - 1].__relayNote.note.time;
        if (!a || !b || a === b) return b || a || '';
        const split = (clock) => { const at = clock.lastIndexOf(', '); return at < 0 ? ['', clock] : [clock.slice(0, at + 2), clock.slice(at + 2)]; };
        const [dayA, clockA] = split(a), [dayB, clockB] = split(b);
        return dayA === dayB ? `${dayA}${clockA}–${clockB}` : `${a}–${b}`;
    }

    // Everything that follows from a run and its state: who is in front, what is tucked, the ledges, the address, the
    // controls. It is idempotent: a note whose place and state did not change keeps every node, so the room's ordinary
    // reconciles write nothing here.
    function paint(run, open) {
        const n = run.length;
        const deck = n > 1;
        const closed = deck && !open;
        const quoted = closed ? run.slice(0, -1).filter(bubble => bubble.__relayNote.note.quoted).length : 0;
        const span = closed ? range(run) : '';
        const extraRooms = closed ? new Set(run.map(bubble => bubble.__relayNote.note.room)).size - 1 : 0;
        const works = closed && inMain ? worksOf(run) : null;
        const held = run.find(bubble => bubble.__relayNote.head.act === document.activeElement);
        let changed = false;
        run.forEach((bubble, i) => {
            const front = i === n - 1;
            bubble.__relayRun = run;
            if (deck) bubble.__relayOpen = open;  // a note split off on its own keeps its deck's intent for a later join
            const key = [n, i, deck && open ? 1 : 0, bubble.dataset.raw, quoted, span, extraRooms,
                works ? `${works.works}/${works.shared}` : ''].join('|');
            if (bubble.__relayPainted === key) return;
            bubble.__relayPainted = key;
            changed = true;
            const tucked = closed && !front;
            setData(bubble, 'deck', deck ? (open ? 'open' : 'closed') : null);
            setData(bubble, 'deckTucked', tucked ? '1' : null);
            setData(bubble, 'deckEdges', closed && front ? Math.min(n - 1, EDGES) : null);
            setData(bubble, 'deckFirst', deck && open && i === 0 ? '1' : null);
            const { note, head, more } = bubble.__relayNote;
            // The closed stack's front card speaks for the whole stack: the other rooms that wrote, the works
            // addressed, the time range.
            const stack = closed && front;
            head.extra.hidden = !(stack && extraRooms);
            if (stack && extraRooms) {
                head.extra.textContent = extraRooms === 1 ? tr('chat.relay.more_rooms_one', 'and 1 more room')
                    : fmt('and {n} more rooms', { n: extraRooms });
            }
            head.to.textContent = receiver(note, stack ? works : null);
            head.time.textContent = stack && span ? span : note.time;
            const first = run[0].__relayNote.note;
            head.time.title = stack && span && first.when ? `${first.when} – ${note.when}` : note.when;
            if (stack) {
                head.act.hidden = false;
                head.act.textContent = fmt('{n} notes', { n });
                head.act.setAttribute('aria-label', fmt('Show {n} notes', { n }));
                head.act.setAttribute('aria-expanded', 'false');
            } else if (deck && open && i === 0) {
                head.act.hidden = false;
                head.act.textContent = tr('chat.relay.show_less', 'Show less');
                head.act.removeAttribute('aria-label');
                head.act.setAttribute('aria-expanded', 'true');
            } else head.act.hidden = true;
            // His words in tucked cards: the blue mark after the excerpt (the 22 Oct rule), on the closed front card.
            const oldMark = bubble.querySelector(':scope > .chat-relay-words-inside');
            if (stack && quoted) {
                const label = quoted === 1 ? tr('chat.relay.earlier_words_one', 'Your words in 1 earlier note')
                    : fmt('Your words in {n} earlier notes', { n: quoted });
                if (oldMark) oldMark.textContent = label;
                else bubble.querySelector(':scope > .chat-relay-excerpt').after(node('div', 'chat-relay-words-inside', label));
            } else oldMark?.remove();
            const raw = bubble.dataset.raw === '1';
            more.textContent = raw ? tr('chat.relay.less', 'less') : tr('chat.relay.more', 'more');
            more.setAttribute('aria-expanded', String(raw));
            more.setAttribute('aria-label', raw ? fmt('Collapse note from {name}', { name: note.name })
                : fmt('Read full note and quoted words from {name}', { name: note.name }));
        });
        // A focused control that left its card (an older note became the column's first card) follows the deck's one
        // control, so the keyboard reader is not dropped on the page.
        if (held?.__relayNote.head.act.hidden) (open ? run[0] : run[n - 1]).__relayNote.head.act.focus({ preventScroll: true });
        return changed;
    }

    /* ---- presses ---------------------------------------------------------------------------------------------------- */
    function onPress(bubble, event) {
        const run = bubble.__relayRun || [bubble];
        const deck = run.length > 1;
        const open = bubble.dataset.deck === 'open';
        if (event.target.closest('.chat-relay-act')) {
            // Focus follows the deck's one control only when it had it (a keyboard reader): paint moves it.
            setOpen(run, !open);
            return;
        }
        if (event.target.closest('.chat-relay-more')) {
            setRaw(bubble, bubble.dataset.raw !== '1');
            return;
        }
        // The closed stack answers a press anywhere on its front card (never on its quote, a control, a link or a selection).
        if (deck && !open && !event.target.closest('.chat-quote, .chat-relay-quotes, button, a, .chat-message-actions')
            && !String(globalThis.document?.getSelection?.() || '')) {
            setOpen(run, true);
        }
    }

    function setRaw(bubble, on) {
        onDomWrite(() => {
            bubble.dataset.raw = on ? '1' : '0';
            const { note, more } = bubble.__relayNote;
            const held = document.activeElement === more;
            fitText(bubble.querySelector('.chat-relay-gist'), note.summary || note.words, on ? 0 : 3);
            for (const text of bubble.querySelectorAll('.chat-quote[data-mode="relay"] > .chat-quote-text')) {
                setData(text, 'expanded', on ? '1' : null);
                fitText(text, fullText(text), on ? 0 : RELAY_CARD_LINES);
            }
            if (held) more.focus({ preventScroll: true });
            paint(bubble.__relayRun || [bubble], bubble.dataset.deck === 'open');
            return true;
        });
        if (on) {
            const { open } = timing();
            for (const part of bubble.querySelectorAll(':scope > .chat-relay-raw-label, :scope > .message, :scope > .chat-message-actions')) {
                play(part, [{ opacity: 0, transform: 'translateY(-4px)' }, { opacity: 1, transform: 'none' }], open);
            }
        }
    }

    /* ---- the stack fans out and folds back ----------------------------------------------------------------------- */
    // The new heights land at once inside one viewport write; then the cards move by position and opacity. Fanning
    // out, the front card glides from where the stack lay to its place at the column's end (only when that is above:
    // a card never starts below its place, since moving content would reach past the feed's end and a write in the
    // meantime would read the reader as having left the newest message) and the older cards emerge from the stack's
    // place behind it. Folding, no card travels: only the ledges settle under the front card.
    function setOpen(run, open, { instant = false } = {}) {
        if (run.length < 2) return;
        const front = run[run.length - 1];
        const before = front.getBoundingClientRect();
        const scrolled = feed.scrollTop || 0;
        const following = Boolean(reading?.()?.stick);
        onDomWrite(() => { paint(run, open); return true; });
        if (instant) return;
        // The reader is placed first, on the column's resting geometry: a card already on its way (its start transform
        // applied) would measure where it set out from, not where it lands.
        placeColumn(run, open, following);
        // Where the stack lay, in the feed as it now stands: placing the column scrolled the feed under it.
        if (open) emerge(run, before.top - ((feed.scrollTop || 0) - scrolled));
        else settleLedges(front, timing().close);
    }

    function emerge(run, stackTop) {
        const { open, ease } = timing();
        const front = run[run.length - 1];
        const landing = front.getBoundingClientRect();
        const glide = stackTop - landing.top;
        if (glide < 0) play(front, [{ transform: `translateY(${glide}px)` }, { transform: 'none' }], open, { easing: ease });
        run.slice(0, -1).forEach((card, i) => {
            const place = card.getBoundingClientRect();
            // From the stack's place, a step lower per card, and never past the front card's end nor below its own place.
            const from = Math.min(stackTop - place.top + i * LEDGE_DROP, landing.bottom - place.bottom, 0);
            // Shrunk toward its top edge, so the scale never lowers a card that sets out at its own place.
            play(card, [{ transform: `translateY(${from}px) scale(0.96)`, transformOrigin: '50% 0', opacity: 0 },
                { transform: 'none', transformOrigin: '50% 0', opacity: 1 }],
                open, { delay: Math.min(i, 4) * 24, easing: ease });
        });
    }

    // The ledges slide in under the front card as cards tucking back behind it: each from the width and place of the
    // level above it (the front card itself for the first), on the sheet curve.
    function settleLedges(front, duration) {
        const width = front.getBoundingClientRect().width || 1;
        const shown = Number(front.dataset.deckEdges || 0);
        for (const ledge of front.querySelectorAll(':scope > .chat-relay-ledge')) {
            const level = Number(ledge.dataset.level || 1);
            if (level > shown) continue;
            const wider = (width - (level - 1) * 2 * LEDGE_INSET) / Math.max(1, width - level * 2 * LEDGE_INSET);
            play(ledge, [{ transform: `translateY(-${LEDGE_DROP}px) scaleX(${wider})`, opacity: 0 }, { transform: 'none', opacity: 1 }],
                duration, { easing: timing().ease });
        }
    }

    // Where the reader is once a stack fans out at the bottom of a room (owner question 6, A): a column that does not
    // fit, whose first card would stand above the read band, asks the room to place that card under the header (the
    // room's own navigation, which the reader's next gesture supersedes like a quote jump); folding asks for the
    // front card back, or for the newest message when the reader was following it. A column that fits asks nothing.
    // Option B is "the screen stays": return at once here.
    function placeColumn(run, open, following) {
        if (!place || typeof feed.getBoundingClientRect !== 'function') return;
        const first = run[0], front = run[run.length - 1];
        const box = feed.getBoundingClientRect();
        const top = box.top + clearance();
        const bottom = band ? band().bottom : box.bottom;
        if (open) {
            const head = first.getBoundingClientRect().top;
            const fits = front.getBoundingClientRect().bottom - head <= bottom - top;
            if (fits || head >= top) return;
            place(first);
            front.__relayPlaced = { following };
            return;
        }
        // The placement rides the card that was in front when the column opened; a note that arrived meanwhile is the
        // front now, and the reader it was following for comes back to it.
        const holder = run.find((bubble) => bubble.__relayPlaced);
        const placed = holder?.__relayPlaced;
        if (!placed) return;
        delete holder.__relayPlaced;
        const card = front.getBoundingClientRect();
        if (placed.following) place(front, { follow: true });
        else if (card.top < top || card.bottom > bottom) place(front);
    }

    /* ---- the feed ------------------------------------------------------------------------------------------------ */
    // Runs of consecutive notes in feed order: hidden rows and the typing row are transparent, any other row ends a
    // run. The presence of the owner's words never splits a run.
    function runs() {
        const out = [];
        let run = [];
        const flush = () => { if (run.length) out.push(run); run = []; };
        for (const el of [...feed.children]) {
            if (isNote(el)) {
                run.push(el);
            } else if (!transparent(el)) flush();
        }
        flush();
        return out;
    }

    // A note arriving live lands at the newest end: no visible row after it.
    const atNewest = (bubble) => {
        for (let el = bubble.nextElementSibling; el; el = el.nextElementSibling) if (!transparent(el)) return false;
        return true;
    };

    function reconcile() {
        scheduled = false;
        if (destroyed) return;
        const arrived = [...fresh].filter((bubble) => bubble.parentElement === feed);
        fresh.clear();
        if (!feed.querySelector(':scope > .chat-relay-note')) { settled = true; return; }
        // One note landing at the newest end of a room already drawn is an arrival; a history page is not.
        const live = settled && arrived.length === 1 && atNewest(arrived[0]) ? arrived[0] : null;
        settled = true;
        onDomWrite(() => {
            let changed = false;
            for (const run of runs()) {
                // A deck stays open while any of its notes was open (a run joined by a removed row keeps it).
                // A note being read on its own (its text open) opens the deck it joins; Show less keeps a deck shut.
                const open = run.length > 1 && run.some((bubble) => bubble.__relayOpen
                    || (!bubble.dataset.deck && bubble.dataset.raw === '1'));
                changed = paint(run, open) || changed;
            }
            return changed;
        });
        if (!live) return;
        // A new note slides in from the side its room lies on (Main to the left of a Project panel, a Project to the
        // right of Main) and lands on the front; the card it covers settles behind it as a ledge.
        const { open, ease } = timing();
        const fromRight = live.__relayNote.note.kind === 'project' && inMain;
        play(live, [{ transform: `translateX(${fromRight ? 24 : -24}px)`, opacity: 0 }, { transform: 'none', opacity: 1 }], open, { easing: ease });
        if (live.dataset.deck === 'closed') settleLedges(live, open);
    }

    function schedule() {
        if (scheduled || destroyed) return;
        scheduled = true;
        queueMicrotask(reconcile);
    }

    // A row inserted between two notes, or taken out between them, changes the runs: the feed's child list says so where
    // an observer exists (the browser); everywhere else each adopted note and each landed row asks for it.
    const watcher = typeof MutationObserver === 'function' ? new MutationObserver(schedule) : null;
    watcher?.observe(feed, { childList: true });

    return {
        /** A relay row's bubble once its quote is decorated: `row` is the history row or live frame, `quoted` whether it
         * passes the owner's words. The bubble is dressed as a card at once (it lands already drawn) and joins its deck
         * once it stands in the feed. */
        adopt(bubble, row, quoted) {
            if (!bubble || !row || bubble.__relayNote) return;
            const note = noteOf(row, Boolean(quoted), bubble);
            bubble.__relayNote = { note };
            dress(bubble, note);
            fresh.add(bubble);
            schedule();
        },
        schedule,
        /** A quote's jump to a tucked note opens its deck first, so the row it lands on is visible; else null. */
        surface(target) {
            if (!isNote(target) || target.dataset.deckTucked !== '1') return null;
            setOpen(target.__relayRun, true, { instant: true });
            return target;
        },
        destroy() {
            destroyed = true;
            watcher?.disconnect();
            lookup.abort();
            rooms.clear();
            fresh.clear();
        },
    };
}
