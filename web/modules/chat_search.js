// Search in a chat (docs/DESIGN.md "Search in a chat"): one room's find surface, opened by Ctrl+F or Cmd+F
// (`bindChatSearchKeys`, which app.js points at the room in front) or by the composer's "+" → Search this chat. Its
// words go to the host (GET /api/chat/search, chat_search.py), which reads the room's saved messages in bounded
// pages, newest first; the surface lists the matches with their snippets and lands on one through the room's own
// reveal (chat_reply.js: the exact row, its bounded history page, the reader's claim), keeping the keyboard in the
// field. It never decides membership, reads the transcript's DOM to find words or loads a room's whole history, and
// nothing it does touches the composer's draft. Its timers, reads and highlight end when it closes, when the room
// leaves the screen (chat.js closes it) and when the room is destroyed.
import { apiClient, isStoreMigrating } from './api_client.js';
import { authorOf } from './chat_reply_quote.js';
import { fmt, tr } from './i18n.js';
import { SHELL_ICONS, strokeIcon } from './page_icons.js';
import { formatMsgTime } from './utils.js';

const DEBOUNCE_MS = 220;
const PAGE_MATCHES = 20;
// The reads one search makes by itself once it has a match and fewer than a page of them; then the list offers
// Search older messages (a press, or Enter past the last match, goes on). Before its first match a search reads on
// until it finds one or reaches the room's beginning: a few pages without a match never read as "none".
const AUTO_READS = 4;
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

/** The day and time a match was written, as the chat's own clock says it. */
const timeOf = (ts) => formatMsgTime(ts)?.short || '';

/**
 * `feed` is the room's transcript (the surface goes right before it, inside the room's page), `reveal(address,
 * isCurrent)` the room's exact-row navigation (resolving `{state, target}`), `present()` whether the room is shown
 * and alive, `onLayout()` the room's re-measure when the surface opens or closes.
 */
export function createChatSearch({ chatId, feed, reveal, present = () => true, api = apiClient, onLayout = () => {} }) {
    let root = null, input = null, count = null, older = null, newer = null, list = null, note = null;
    let open = false, opener = null, destroyed = false;
    // The search: the words it is for, its matches newest first, the one landed on, and where its next read starts.
    let words = '', matches = [], active = -1, cursor = null, more = false, reached = '', reading = false;
    let beginning = false, importing = false, gaps = new Set(), failure = null, steps = 0, reads = 0;
    let generation = 0, controller = null, timer = 0, landing = 0;
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
        input.addEventListener('input', () => schedule());
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
            if (failure && !failure.landing) { failure = null; reads = 0; void read(generation); } else readMore();
        });
        feed.before(root);
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
        gaps = new Set(); failure = null; steps = 0; reads = 0;
        clearHighlight();
    }

    function schedule() {
        reset();
        words = String(input.value || '').replace(/\s+/g, ' ').trim();
        render();
        if (!words) return;
        const mine = generation;
        timer = setTimeout(() => { timer = 0; if (mine === generation) void read(mine); }, DEBOUNCE_MS);
    }

    async function read(mine) {
        if (!open || destroyed || !present()) { if (open && !present()) closeSearch(); return; }
        reading = true;
        controller = new AbortController();
        render();
        let page;
        try {
            page = await api.chatSearch({ chatId, q: words, cursor, limit: PAGE_MATCHES, signal: controller.signal });
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
        reading = false;
        controller = null;
        reads += 1;
        matches = matches.concat(Array.isArray(page?.matches) ? page.matches : []);
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

    function readMore() {
        if (reading || !more || !words) return;
        reads = 0;
        void read(generation);
    }

    // ---- landing -------------------------------------------------------------------------------------------------
    function clearHighlight() {
        try { globalThis.CSS?.highlights?.delete(HIGHLIGHT); } catch { /* no highlight registry */ }
    }

    /** Best effort: mark the query's words in the landed row (the CSS Custom Highlight API: no node is written),
     * lowercased within each single text node. It does not repeat the host's matching (Unicode case folding, NFC,
     * white space runs) or cross markup, so words the host matched may go unmarked; the landing's ring still shows
     * the row. Lowercasing is per character, so a mark maps back to its own text even where case changes a length. */
    function highlight(target) {
        clearHighlight();
        const registry = globalThis.CSS?.highlights;
        if (!target || !registry || typeof globalThis.Highlight !== 'function' || !words) return;
        const wanted = words.toLowerCase();
        const ranges = [];
        const walker = document.createTreeWalker(target, 4 /* NodeFilter.SHOW_TEXT */);
        for (let node = walker.nextNode(); node; node = walker.nextNode()) {
            if (node.parentElement?.closest('.chat-message-actions, .msg-time, .chat-quote, button')) continue;
            const text = node.data;
            let folded = '';
            const owner = [];
            for (let index = 0; index < text.length; index += 1) {
                const part = text[index].toLowerCase();
                folded += part;
                for (let unit = 0; unit < part.length; unit += 1) owner.push(index);
            }
            for (let at = folded.indexOf(wanted); at >= 0; at = folded.indexOf(wanted, at + wanted.length)) {
                const range = document.createRange();
                range.setStart(node, owner[at]);
                range.setEnd(node, owner[at + wanted.length - 1] + 1);
                ranges.push(range);
            }
        }
        if (ranges.length) registry.set(HIGHLIGHT, new globalThis.Highlight(...ranges));
    }

    async function land(index) {
        if (!open || index < 0 || index >= matches.length) return;
        active = index;
        // On a phone the list would leave the landed message little room: it folds before the landing measures
        // the sheet, and opens again if nothing was landed.
        if (document.documentElement?.dataset?.layout === 'stack') root.setAttribute('data-folded', '');
        render();
        const mine = ++landing;
        const result = await reveal(matches[index].row_address, () => open && mine === landing && present());
        if (mine !== landing || !open) return;
        if (result?.state === 'found') {
            highlight(result.target);
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
            const state = failure.landing.state;
            return state === 'absent' ? tr('chat.find.landing_absent', 'This message is no longer in saved history.')
                : state === 'unavailable' ? tr('chat.find.landing_unavailable', 'This message cannot be shown in an available room.')
                    : tr('chat.find.landing_unknown', 'Could not open this message. Try again.');
        }
        if (failure) {
            return isStoreMigrating(failure) ? tr('chat.find.migrating', 'Saved history is being upgraded. Try again shortly.')
                : tr('chat.find.failed', 'Saved history could not be searched.');
        }
        if (!words) return '';
        if (reading) {
            return matches.length ? '' : reached ? fmt('chat.find.searching_back', { time: timeOf(reached) },
                'Searching saved messages… back to {time} so far.') : tr('chat.find.searching', 'Searching saved messages…');
        }
        const lines = [];
        if (!matches.length && beginning && !gaps.size) {
            lines.push(fmt('chat.find.none', { query: words }, 'No saved message in this chat contains “{query}”.'));
        } else if (!matches.length && reached) {
            lines.push(fmt('chat.find.none_back_to', { time: timeOf(reached) }, 'No match in messages back to {time}.'));
        }
        if (importing) lines.push(tr('chat.find.importing', 'Older history is still being upgraded and was not searched.'));
        if (gaps.size) lines.push(tr('chat.find.gaps', 'Some saved history could not be read, so it was not searched.'));
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
        const exact = beginning && !importing && !gaps.size && !(failure && !failure.landing) && !reading && !timer && !more;
        const all = exact ? total : `${total}+`;
        count.textContent = !words || (!total && !exact) ? ''
            : active >= 0 ? fmt('chat.find.position', { n: active + 1, total: all }, '{n} of {total}')
                : exact ? fmt('chat.find.total', { n: total, total }, '{total} found') : all;
        older.disabled = !words || (active + 1 >= total && !more);
        newer.disabled = active <= 0;
        list.replaceChildren(...matches.map(option));
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
        const retry = failure && !failure.landing;
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
            root?.remove();
            root = null;
        },
    };
}

const LATIN = /^[a-z]$/i;

/**
 * Ctrl+F or Cmd+F opens the search of `room()`'s room (the room in front, or null where the key stays the page's):
 * the letter F of the layout, or the key in F's place where the layout types no Latin letter. A dialog, a menu or an
 * inert part of the page keeps its own keys. Returns the disposer.
 */
export function bindChatSearchKeys({ room, win = globalThis.window }) {
    const onKeydown = (event) => {
        if (event.defaultPrevented || event.isComposing || event.altKey || event.shiftKey) return;
        if (event.ctrlKey === event.metaKey) return;  // exactly one of Ctrl and Cmd
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
