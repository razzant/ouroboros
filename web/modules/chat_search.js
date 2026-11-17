// Search in a chat (docs/DESIGN.md "Search in a chat"): one room's find surface, opened by Cmd+F on a Mac and
// Ctrl+F elsewhere (`bindChatSearchKeys`, which app.js points at the room in front) or by the composer's "+" → Search
// this chat. The host (GET /api/chat/search, chat_search.py) reads the room's saved messages in bounded pages, newest
// first, and answers each page's candidates; the surface reads every candidate's shown words as the room draws them
// (chat_search_words.js), a slice at a time with the event loop free between slices, keeps only the matches (their
// address and snippet, never a message's body), lists them and lands on one through the room's own reveal
// (chat_reply.js: the exact row, its bounded history page, the reader's claim, its first matching words placed in
// view), keeping the keyboard in the field. It never decides membership, reads the transcript's DOM to find which
// messages match (only where the words stand in the landed row) or loads a room's whole history, and nothing it does
// touches the composer's draft. Its timers, reads, reading of candidates and highlight end with new words, when it
// closes, when the room leaves the screen (chat.js closes it) and when the room is destroyed.
import { apiClient, isStoreMigrating } from './api_client.js';
import { authorOf } from './chat_reply_quote.js';
import { fmt, tr } from './i18n.js';
import { SHELL_ICONS, strokeIcon } from './page_icons.js';
import { formatMsgTime } from './utils.js';
import { candidateBox, findIn, queryWords, rangesIn, searchKey, shownRow, snippetOf, wordPieces, wordStream } from './chat_search_words.js';

const DEBOUNCE_MS = 220;
const PAGE_MATCHES = 20;
// The reads one search makes by itself once it has a match and fewer than a page of them; then the list offers
// Search older messages (a press, or Enter past the last match, goes on). Before its first match a search reads on
// until it finds one or reaches the room's beginning: a few pages without a match never read as "none".
const AUTO_READS = 4;
// The longest words the host searches (chat_search.py _MAX_QUERY_CHARS, Python code points): the field takes no more
// (`clipQuery`; an HTML maxlength counts UTF-16 units, so it would refuse the second half of an emoji query).
const MAX_QUERY_CHARS = 200;
// How long one slice of reading candidates holds the main thread before the event loop runs (typing, Esc, a scroll).
// A message is never split: its words are drawn and read in one step, as its bubble is drawn in one step.
const SLICE_MS = 12;
// One registered highlight (style.css ::highlight(ouro-chat-find)) holds every room's marks: each surface adds and
// removes only its own ranges, so closing one room's search never clears another's.
const HIGHLIGHT = 'ouro-chat-find';
const UP = strokeIcon('<path d="m18 15-6-6-6 6"/>', 16);
const DOWN = strokeIcon('<path d="m6 9 6 6 6-6"/>', 16);
let surfaces = 0;

function element(tag, className, attributes = {}) {
    const node = document.createElement(tag);
    if (className) node.className = className;
    for (const [name, value] of Object.entries(attributes)) node.setAttribute(name, value);
    return node;
}

/**
 * Keep `field` within `MAX_QUERY_CHARS` code points, as a maxlength keeps its units: the code points just typed (those
 * before the caret) give way first, a pair of surrogates never split. Whether it clipped. The host counts its words after
 * NFC and one space, so the rare letter NFC writes as two (U+0958) can still be refused there (`search_query_invalid`).
 */
function clipQuery(field) {
    const value = String(field.value ?? ''), points = Array.from(value);
    const excess = points.length - MAX_QUERY_CHARS;
    if (excess <= 0) return false;
    const caret = Number.isInteger(field.selectionStart) ? field.selectionStart : value.length;
    let at = 0;  // the code points wholly before the caret
    for (let units = 0; at < points.length && units + points[at].length <= caret; at += 1) units += points[at].length;
    const cut = Math.min(excess, at), kept = points.slice(0, at - cut).join('');
    field.value = kept + points.slice(at, points.length - (excess - cut)).join('');
    field.setSelectionRange?.(kept.length, kept.length);
    return true;
}

/** The day and time a match was written, as the chat's own clock says it. */
const timeOf = (ts) => formatMsgTime(ts)?.short || '';

/** A candidate's shown words as a stream (chat_search_words.js), drawn in `inert()`, an inert document the read owns. */
const shownWords = (candidate, inert) => wordStream(wordPieces(candidateBox(candidate, inert())));
const inertDocument = () => document.implementation.createHTMLDocument('');
const pauseTask = () => new Promise((done) => { setTimeout(done, 0); });
const clock = () => globalThis.performance?.now?.() ?? Date.now();

/**
 * `feed` is the room's transcript (the surface goes right before it, inside the room's page), `reveal(address,
 * isCurrent, locate)` the room's exact-row navigation (resolving `{state, target}`; `locate(target)`, called before
 * it places the row, names the Range it places in view), `present()` whether the room is shown and alive,
 * `onLayout()` the room's re-measure when the surface opens, closes or changes its height. `words(candidate, inert)`
 * and `pause()` are the reading of a candidate's shown words and the yield between slices (replaced only by tests).
 */
export function createChatSearch({ chatId, feed, reveal, present = () => true, api = apiClient, onLayout = () => {},
    words: readWords = shownWords, pause = pauseTask }) {
    let root = null, input = null, count = null, older = null, newer = null, list = null, note = null;
    let open = false, opener = null, destroyed = false;
    // The search: the words it is for, its matches newest first, the one landed on, and where its next read starts.
    // `key` is the words as matched; `unread` counts candidates whose shown words could not be drawn (never "no match").
    let words = '', key = '', matches = [], active = -1, cursor = null, more = false, reached = '', reading = false;
    // `begun`: a page of this search was read, so a read with no cursor would be its newest page again.
    let beginning = false, importing = false, gaps = new Set(), failure = null, steps = 0, reads = 0, unread = 0;
    let begun = false;
    let generation = 0, controller = null, timer = 0, landing = 0, folded = false, unshown = false, owned = [];
    // The list draws each match once while a search reads (a page can hold thousands): `listed` is the matches it
    // draws, a new search's starting a new list, and `chosen` the one option marked selected.
    let listed = null, chosen = null, sizes = null, sized = 0;
    const id = `chat-find-${++surfaces}`;

    function build() {
        root = element('div', 'chat-find');
        root.hidden = true;
        const bar = element('div', 'chat-find-bar', { role: 'search' });
        const glyph = element('span', 'chat-find-glyph', { 'aria-hidden': 'true' });
        glyph.innerHTML = SHELL_ICONS.search;
        input = element('input', 'ui-control chat-find-input', {
            type: 'search', role: 'combobox', placeholder: 'Search this chat', 'aria-label': 'Search this chat’s saved messages',
            'aria-autocomplete': 'list', 'aria-expanded': 'false', 'aria-controls': `${id}-results`,
            autocomplete: 'off', spellcheck: 'false', enterkeyhint: 'search',
        });
        // The count and the note are composed with the query's words (fmt/tr), never chrome for the overlay.
        count = element('span', 'chat-find-count', { 'aria-hidden': 'true', 'data-i18n-skip': '' });
        const button = (className, icon, label) => {
            const node = element('button', `chat-find-button ui-hit-area ${className}`, { type: 'button', 'aria-label': label, title: label });
            node.innerHTML = icon;
            return node;
        };
        // As a browser's find: ↓ (Enter) the next match down the newest-first list, older; ↑ (Shift+Enter) the previous.
        older = button('chat-find-older', DOWN, 'Next match, older (Enter)');
        newer = button('chat-find-newer', UP, 'Previous match, newer (Shift+Enter)');
        const close = button('chat-find-close', SHELL_ICONS.close, 'Close search (Esc)');
        bar.append(glyph, input, count, newer, older, close);
        list = element('ul', 'chat-find-results', { id: `${id}-results`, role: 'listbox', 'aria-label': 'Matching messages',
            'data-i18n-authored': '' });
        list.hidden = true;
        note = element('div', 'chat-find-note', { role: 'status', 'aria-live': 'polite', 'data-i18n-skip': '' });
        note.hidden = true;  // shown only with something to say (render): the field alone is the sheet's capsule
        root.append(bar, list, note);
        // An IME's words are clipped once its composition ends, never while it composes them.
        input.addEventListener('input', (event) => { if (!event.isComposing) clipQuery(input); schedule(); });
        input.addEventListener('compositionend', () => { if (clipQuery(input)) schedule(); });
        input.addEventListener('keydown', onKeydown);
        root.addEventListener('keydown', (event) => {
            if (event.key !== 'Escape' || event.isComposing || event.defaultPrevented) return;
            event.preventDefault();  // the room's layer keeps its own Escape for after the search
            closeSearch();
        });
        older.addEventListener('click', () => step(1));
        newer.addEventListener('click', () => step(-1));
        close.addEventListener('click', () => closeSearch());
        // A mouse press on a match or a control leaves the keyboard in the field, as a chooser's list and "+" do.
        root.addEventListener('mousedown', (event) => {
            if (event.target !== input && event.target.closest?.('.chat-find-results, .chat-find-button')) event.preventDefault();
        });
        // A phone folds the list over a landed message (below); touching or typing in the field brings it back.
        const unfold = () => { if (root.hasAttribute('data-folded')) { root.removeAttribute('data-folded'); onLayout(); } };
        input.addEventListener('pointerdown', unfold);
        input.addEventListener('input', unfold);
        list.addEventListener('click', (event) => {
            const option = event.target.closest?.('[data-find-index]');
            if (option) land(Number(option.dataset.findIndex));
        });
        note.addEventListener('click', (event) => {
            const press = event.target.closest?.('[data-find-more]');
            if (!press) return;
            input.focus({ preventScroll: true });
            if (retryable()) { reads = 0; void read(generation); } else readMore();
        });
        feed.before(root);
        // The sheet's height follows its matches, its note, a fold and the window: the room re-measures what it covers,
        // once, in the next frame (its write resizes the feed, which the room observes too; a landing measures itself).
        if (typeof ResizeObserver !== 'function') return;
        (sizes = new ResizeObserver(() => {
            if (!open || sized) return;
            sized = requestAnimationFrame(() => { sized = 0; if (open && !destroyed) onLayout(); });
        })).observe(root);
    }

    // ---- reads ---------------------------------------------------------------------------------------------------
    function stop() {
        generation += 1;
        controller?.abort();
        controller = null;
        clearTimeout(timer);
        timer = 0;
        reading = false;
    }

    function reset() {
        stop();
        landing += 1;  // a landing still on its way for the old words lands nowhere
        matches = []; active = -1; cursor = null; more = false; reached = ''; beginning = false; importing = false;
        begun = false;
        gaps = new Set(); failure = null; steps = 0; reads = 0; unread = 0; folded = false; unshown = false;
        clearHighlight();
    }

    function schedule() {
        reset();
        words = queryWords(input.value);
        key = searchKey(words);
        render();
        if (!words) return;
        const mine = generation;
        timer = setTimeout(() => { timer = 0; if (mine === generation) void read(mine); }, DEBOUNCE_MS);
    }

    async function read(mine) {
        if (!open || destroyed || !present()) { if (open && !present()) closeSearch(); return; }
        // A read continues from `cursor`, which a failed read left where it failed; once the room's beginning was
        // reached there is nothing older, and no read starts again from the newest page.
        if (begun && !cursor) { steps = 0; render(); return; }
        failure = null;  // a new read speaks for itself, never the last landing's or the last failed read's reason
        reading = true;
        controller = new AbortController();
        render();
        let page;
        try {
            page = await api.chatSearch({ chatId, q: words, cursor, signal: controller.signal });
        } catch (error) {
            if (mine !== generation) return;
            reading = false;
            if (error?.status === 409 && cursor) {  // the room's view changed under the search: read it again from now
                const keep = words;
                reset();
                words = keep;
                void read(generation);
                return;
            }
            failure = error;
            steps = 0;
            render();
            return;
        }
        if (mine !== generation) return;
        controller = null;
        if (!await scan(Array.isArray(page?.candidates) ? page.candidates : [], mine)) return;
        reading = false;
        reads += 1;
        begun = true;
        cursor = page?.next_cursor || null;
        more = Boolean(page?.has_more && cursor);
        const span = page?.searched || {};
        if (span.reached_ts) reached = span.reached_ts;
        beginning = Boolean(span.beginning);
        importing = Boolean(span.importing);
        for (const gap of span.gaps || []) gaps.add(gap);
        render();
        if (steps > 0 && active + 1 < matches.length) {  // an Enter pressed while this read ran lands now
            steps = 0;
            land(active + 1);
        } else if (more && (steps > 0 || !matches.length || (matches.length < PAGE_MATCHES && reads < AUTO_READS))) {
            void read(mine);
        } else {
            steps = 0;
        }
    }

    /** Reads `candidates` for the words, newest first, a slice at a time; false once newer words, Esc or the room's
     *  leaving ended this search. A candidate kept is its match only: address, author, time and snippet. */
    async function scan(candidates, mine) {
        let doc = null, since = clock();
        const inert = () => (doc ||= inertDocument());
        for (const candidate of candidates) {
            if (mine !== generation) return false;
            let stream = null;
            try {
                stream = readWords(candidate, inert);
            } catch (error) {
                unread += 1;
                console.warn('chat search: a saved message could not be read as shown', error);
            }
            const found = stream ? findIn(stream, key) : [];
            if (found.length) {
                const { history_id, row_address, ts, sender_label, task_id } = candidate;
                matches.push({ history_id, row_address, ts: ts || '', role: shownRow(candidate).role,
                    ...(sender_label ? { sender_label } : {}), ...(task_id ? { task_id } : {}),
                    snippet: snippetOf(stream, found) });
            }
            if (clock() - since >= SLICE_MS) {
                render();
                await pause();
                if (mine !== generation) return false;
                since = clock();
            }
        }
        return true;
    }

    /** Whether the last read failed and can be tried again (words too long to search never can). */
    function retryable() {
        return Boolean(failure && !failure.landing && failure.body?.reason_code !== 'search_query_invalid');
    }

    function readMore() {
        if (reading || !more || !words) return;
        reads = 0;
        void read(generation);
    }

    // ---- landing -------------------------------------------------------------------------------------------------
    /** This surface's marks become `ranges`; the other rooms' marks in the shared highlight stay. */
    function mark(ranges) {
        const registry = globalThis.CSS?.highlights;
        let shared = null;
        try { shared = registry?.get(HIGHLIGHT) || null; } catch { return; /* no highlight registry */ }
        for (const range of owned) shared?.delete(range);
        owned = [];
        if (ranges.length && registry && typeof globalThis.Highlight === 'function') {
            if (!shared) registry.set(HIGHLIGHT, shared = new globalThis.Highlight());
            for (const range of ranges) shared.add(range);
            owned = ranges;
        } else if (shared && !shared.size) registry.delete(HIGHLIGHT);
    }
    const clearHighlight = () => mark([]);

    async function land(index) {
        if (!open || index < 0 || index >= matches.length) return;
        active = index;
        folded = false;
        unshown = false;
        // On a phone the list would leave the landed message little room: it folds before the landing measures
        // the sheet, and opens again if nothing was landed.
        if (document.documentElement?.dataset?.layout === 'stack') root.setAttribute('data-folded', '');
        render();
        const mine = ++landing;
        // Where the words stand in the landed row, read as its candidate was, marked and handed to the reveal, which
        // places the first of them in view within its own claim (chat_reply.js).
        let located = false;
        const locate = (target) => {
            located = true;
            const ranges = rangesIn(itemOf(target, matches[index].history_id), key);
            mark(ranges);
            folded = Boolean(ranges[0]) && foldHides(ranges[0], target);
            unshown = !ranges.length;
            failure = null;
            render();  // said before the row is placed below the sheet
            return folded ? null : ranges[0] || null;
        };
        const result = await reveal(matches[index].row_address, () => open && mine === landing && present(), locate);
        if (mine !== landing || !open) return;
        if (result?.state === 'found') {
            if (!located) locate(result.target);
            failure = null;
        } else {
            root.removeAttribute('data-folded');
            if (result) failure = { landing: result };
        }
        render();
    }

    /** Enter (1, older) or Shift+Enter (-1, newer): the next match, reading older history when the loaded ones end. */
    function step(direction) {
        if (!words) return;
        const next = active + direction;
        if (direction < 0) { if (next >= 0) land(next); return; }
        if (next < matches.length) { land(next); return; }
        if (reading) { steps = 1; return; }  // lands once the running read brings it
        if (more) { steps = 1; reads = 0; void read(generation); }
    }

    function onKeydown(event) {
        if (event.isComposing) return;
        if (event.key === 'Enter' || event.key === 'ArrowDown' || event.key === 'ArrowUp') {
            event.preventDefault();
            if (timer) {  // the words just typed are read first; the press lands on their first match
                clearTimeout(timer);
                timer = 0;
                steps = 1;
                void read(generation);
                return;
            }
            step(event.key === 'ArrowUp' || (event.key === 'Enter' && event.shiftKey) ? -1 : 1);
        }
    }

    // ---- painting ------------------------------------------------------------------------------------------------
    function option(match, index) {
        const item = element('li', 'chat-find-result', { role: 'option', id: `${id}-${index}`,
            'aria-selected': String(index === active) });
        item.dataset.findIndex = String(index);
        const meta = element('div', 'chat-find-meta');
        const author = element('span', 'chat-find-author');
        author.textContent = match.sender_label || authorOf(match.role) || '';
        const time = element('time', 'chat-find-time');
        time.textContent = timeOf(match.ts);
        if (match.ts) time.dateTime = match.ts;
        meta.append(author, time);
        const text = element('div', 'chat-find-snippet');
        for (const part of Array.isArray(match.snippet) ? match.snippet : []) {
            if (part.match) {
                const mark = element('mark', 'chat-find-mark');
                mark.textContent = part.text;
                text.append(mark);
            } else text.append(document.createTextNode(part.text));
        }
        item.append(meta, text);
        return item;
    }

    function noteText() {
        if (failure?.landing) {
            const { state, error } = failure.landing;
            return state === 'absent' ? tr('chat.find.landing_absent', 'This message is no longer in saved history.')
                : state === 'ambiguous' ? tr('chat.find.landing_ambiguous', 'This reference matches more than one saved message.')
                : state === 'unavailable' ? tr('chat.find.landing_unavailable', 'This message cannot be shown in an available room.')
                    : isStoreMigrating(error) ? tr('chat.find.migrating', 'Saved history is being upgraded. Try again shortly.')
                        : error?.body?.reason_code === 'history_source_unavailable'
                            ? tr('chat.find.landing_source', 'Saved history holding this message could not be read. Try again.')
                            : tr('chat.find.landing_unknown', 'Could not open this message. Try again.');
        }
        if (failure?.body?.reason_code === 'search_query_invalid') {
            return fmt('chat.find.too_long', { n: MAX_QUERY_CHARS }, 'Search words are limited to {n} characters.');
        }
        if (failure) {
            return isStoreMigrating(failure) ? tr('chat.find.migrating', 'Saved history is being upgraded. Try again shortly.')
                : tr('chat.find.failed', 'Saved history could not be searched.');
        }
        if (!words) return '';
        if (reading) {  // with the count's `N+`, how far the reading has come; Esc or new words end it
            return reached ? fmt('chat.find.searching_back', { time: timeOf(reached) },
                'Searching saved messages… back to {time} so far.') : tr('chat.find.searching', 'Searching saved messages…');
        }
        const lines = [];
        if (folded) lines.push(tr('chat.find.folded', 'The words are in the folded part of this answer: its Project shows it in full.'));
        if (unshown) lines.push(tr('chat.find.unshown', 'The match is in the saved message, but its words are not shown here.'));
        if (!matches.length && beginning && !gaps.size && !unread) {
            lines.push(fmt('chat.find.none', { query: words }, 'No saved message in this chat contains “{query}”.'));
        } else if (!matches.length && reached) {
            lines.push(fmt('chat.find.none_back_to', { time: timeOf(reached) }, 'No match in messages back to {time}.'));
        }
        if (importing) lines.push(tr('chat.find.importing', 'Older history is still being upgraded and was not searched.'));
        if (gaps.size) lines.push(tr('chat.find.gaps', 'Some saved history could not be read, so it was not searched.'));
        if (unread) {
            lines.push(fmt('chat.find.unread', { n: unread }, '{n} saved messages could not be read as shown, so they were not searched.'));
        }
        if (more && matches.length && reached) {
            lines.push(fmt('chat.find.back_to', { time: timeOf(reached) }, 'Searched back to {time}.'));
        }
        return lines.join(' ');
    }

    function render() {
        if (!root) return;
        const total = matches.length;
        // The count is exact only when the whole room was read: its beginning reached with nothing still importing or
        // unreadable, and no read failed, runs or waits. Otherwise it says how many so far (`N+`), or nothing yet.
        const exact = beginning && !importing && !gaps.size && !unread && !(failure && !failure.landing) && !reading && !timer
            && !more;
        const all = exact ? total : `${total}+`;
        count.textContent = !words || (!total && !exact) ? ''
            : active >= 0 ? fmt('chat.find.position', { n: active + 1, total: all }, '{n} of {total}')
                : exact ? fmt('chat.find.total', { n: total, total }, '{total} found') : all;
        older.disabled = !words || (active + 1 >= total && !more);
        newer.disabled = active <= 0;
        if (listed !== matches) { list.replaceChildren(); listed = matches; chosen = null; }
        const drawn = list.children.length;
        if (drawn < total) list.append(...matches.slice(drawn).map((match, at) => option(match, drawn + at)));
        const selected = active >= 0 ? list.children[active] || null : null;
        if (chosen !== selected) {
            chosen?.setAttribute('aria-selected', 'false');
            selected?.setAttribute('aria-selected', 'true');
            chosen = selected;
        }
        list.hidden = !total;
        input.setAttribute('aria-expanded', String(Boolean(total)));
        if (active >= 0) {
            input.setAttribute('aria-activedescendant', `${id}-${active}`);
            list.children[active]?.scrollIntoView?.({ block: 'nearest' });
        } else input.removeAttribute('aria-activedescendant');
        root.toggleAttribute('data-searching', reading);
        const text = noteText();
        note.replaceChildren();
        if (text) note.append(Object.assign(element('span', 'chat-find-note-text'), { textContent: text }));
        const retry = retryable();
        if ((more && !reading && words) || retry) {
            const press = element('button', 'chat-find-more', { type: 'button', 'data-find-more': '' });
            press.textContent = retry ? tr('chat.find.retry', 'Try again') : tr('chat.find.more', 'Search older messages');
            note.append(press);
        }
        note.hidden = !note.children.length;
    }

    // ---- open and close ------------------------------------------------------------------------------------------
    function openSearch() {
        if (destroyed) return;
        if (!root) build();
        if (!open) {
            const at = document.activeElement;
            opener = at && at !== document.body && !root.contains(at) ? at : null;
            open = true;
            root.hidden = false;
            onLayout();
            if (String(input.value || '').trim()) schedule();  // the words kept from the last search, read again now
        }
        input.focus({ preventScroll: true });
        input.select();
    }

    function closeSearch() {
        if (!open) return;
        open = false;
        landing += 1;
        stop();
        clearHighlight();
        root.hidden = true;
        root.removeAttribute('data-folded');
        onLayout();
        const back = opener;
        opener = null;
        if (root.contains(document.activeElement) || document.activeElement === document.body) {
            if (back?.isConnected && !back.closest?.('[inert], [hidden]')) back.focus({ preventScroll: true });
        }
    }

    return {
        open: openSearch,
        close: closeSearch,
        get isOpen() { return open; },
        /** What the surface covers of the feed's top while it is open: a landing places its row below it. */
        cover: () => (open && root && !root.hidden ? root : null),
        destroy() {
            if (destroyed) return;
            closeSearch();
            destroyed = true;
            sizes?.disconnect();
            if (sized) cancelAnimationFrame(sized);
            root?.remove();
            root = null;
        },
    };
}

/** The photo or file of a group that is the match's own message, else the landed row: a work's consecutive photos, or
 *  files, share the first one's bubble, each item keeping its own history id (chat_media.js buildGallery), so a match
 *  marks its own caption, never the next one's. */
function itemOf(target, id) {
    if (!id || !target || target.dataset?.historyId === id) return target;
    return [...target.querySelectorAll?.('.chat-gallery-item[data-history-id], .chat-file-item[data-history-id]') || []]
        .find((item) => item.dataset.historyId === id) || target;
}

/** Whether a folded Project answer (project_answer.js: its seven-line fold, no expansion in Main) hides `words`. */
function foldHides(words, target) {
    const message = target.matches('.project-answer.is-folded') ? target.querySelector(':scope > .message') : null;
    const box = words.getClientRects()[0];
    if (!box || !message?.contains(words.startContainer)) return false;
    const style = getComputedStyle(message);
    const fold = parseFloat(style.lineHeight) * parseFloat(style.getPropertyValue('--project-answer-lines'));
    return box.bottom > message.getBoundingClientRect().top + fold + 1;
}

const LATIN = /^[a-z]$/i;
const isMac = (win) => /Mac|iPhone|iPad/.test(String(win?.navigator?.platform || win?.navigator?.userAgent || ''));

/**
 * Cmd+F on a Mac, Ctrl+F elsewhere (interface_size.js reads the platform the same way) opens the search of `room()`'s
 * room (the room in front, or null where the key stays the page's): the letter F of the layout, or the key in F's place
 * where the layout types no Latin letter. A Mac's Ctrl+F stays the text field's own (a character forward). A dialog, a
 * menu or an inert part of the page keeps its own keys. Returns the disposer.
 */
export function bindChatSearchKeys({ room, win = globalThis.window }) {
    const mac = isMac(win);
    const onKeydown = (event) => {
        if (event.defaultPrevented || event.isComposing || event.altKey || event.shiftKey) return;
        if (mac ? !event.metaKey || event.ctrlKey : !event.ctrlKey || event.metaKey) return;
        const key = String(event.key || '');
        if (LATIN.test(key) ? key.toLowerCase() !== 'f' : event.code !== 'KeyF') return;
        const doc = win.document;
        const at = doc.activeElement;
        if (at?.closest?.('[inert], dialog, [role="dialog"], [role="menu"], [aria-modal="true"]')) return;
        // Pointer-opened menus keep focus in the composer; their leaving pictures no longer own keys.
        if ([...doc.querySelectorAll('dialog[open], [aria-modal="true"], [role="menu"]')]
            .some((el) => !el.closest('[data-leaving]') && el.getClientRects?.().length)) return;
        const target = room();  // the room in front, shown and live: a stale or covered room never takes the key
        if (!target?.search || !target.page?.getClientRects?.().length || target.page.closest?.('[inert]')) return;
        event.preventDefault();
        target.search.open();
    };
    win.addEventListener('keydown', onKeydown);
    return () => win.removeEventListener('keydown', onKeydown);
}
