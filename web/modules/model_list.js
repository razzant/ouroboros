// The one list of models (docs/DESIGN.md "Composer owner controls"): a field and its suggestions, the body of the Model
// sheet in a composer (composer_model.js) and on a work's card (card_model.js), model_sheet.js around both. Without a
// query it shows Main (`mainRow`), the pinned models, the recent ones, then the sources that can lead a work (one
// source: its models at once; more: folded under `All sources`); typed words search every source; any id can be used
// as typed. The catalog suggests and never admits (chat_model_routes.js).
//
// The list is a grid the field drives (a combobox over it, focus stays in the field): a row's first cell picks its
// model, its second is its Pin. Arrows Up and Down move among rows, then Right and Left between that row's cells
// (until then they move the caret), Enter presses the active cell; a pointer presses either. The active row is its
// model's, not a place: a repaint (another window's Pin, a state read) keeps it on that model or clears it when the
// model left the list. Main's row is the one way to follow Main: a model row is always that route, even the model Main
// runs on now. A Pin chooses nothing and leaves the list open.
import { fmt, tr } from './i18n.js';
import { sourceMark } from './model_source_mark.js';
import { SHELL_ICONS } from './page_icons.js';
import { escapeHtmlAttr as esc } from './ui_primitives.js';
import { describeRoute, routeKey, sameRoute, searchModels, typedRoute } from './chat_model_routes.js';

// How many suggestions one list shows; the rest wait for another letter.
const LIST_ROWS = 60;
// How many sources `All sources` names before `+ N`.
const NAMED_SOURCES = 2;
export const SEARCH_ICON = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round"'
    + ' stroke-linejoin="round" aria-hidden="true" focusable="false"><circle cx="11" cy="11" r="7"></circle><path d="m20 20-3.5-3.5"></path></svg>';

// Words composed into lines the translation overlay leaves alone (they name models and sources): each is asked for
// by its own code, the composed ones by their template. Words standing alone in the page are the overlay's.
const payWords = (said) => tr(`model.pay.${said.lane}`, said.pay);

/** A route's facts in the owner's words: its source, how that source is paid, its window when one is known. */
export function routeFacts(said) {
    return [said.sourceLabel, said.price || payWords(said), said.window ? fmt('{window} window', { window: said.window }) : '']
        .filter(Boolean).join(' · ');
}

/** The search field both homes of the list use: a combobox over the grid `listId`. */
export function modelSearchField(doc, listId) {
    const label = doc.createElement('label');
    label.className = 'chat-model-search';
    label.innerHTML = SEARCH_ICON;
    const input = doc.createElement('input');
    input.className = 'chat-model-query';
    for (const [key, value] of Object.entries({ type: 'text', role: 'combobox', autocomplete: 'off', autocorrect: 'off',
        autocapitalize: 'off', spellcheck: 'false', 'aria-autocomplete': 'list', 'aria-expanded': 'true',
        'aria-haspopup': 'grid', 'aria-controls': listId, 'aria-label': 'Search models' })) input.setAttribute(key, value);
    input.dataset.i18nSkip = '';  // its placeholder may name a source: written through the catalog by the list
    label.append(input);
    return { label, input };
}

/**
 * The list painted into `el` (the grid) and driven by `input`. `main()`, `current()` and `recents()` are the caller's
 * routes (Main's now, the chosen one, the recent ones to offer); `follows()` says whether the scope follows Main now.
 * `mainRow`: `follow` puts Main first as the way to follow it (its facts `mainFacts()`), `main` puts Main's own model
 * first as the chosen one (the Main chat), false none. `tierOf(route, { main })` is the processing tier a pick would
 * run on here: a catalog price is said only where that tier pays it. `onRoute(route)` takes a pick (null: follow
 * Main), `onPin(route, pinned)` a Pin, `onSettings()` the way to Settings › Accounts and keys where nothing but Main can
 * lead, `onLeave({focus})` an Escape with nothing typed; `onPaint()` follows every repaint. Returns
 * `{ paint, destroy }`.
 */
export function createModelList({ el, input, id, store, main, current, recents, follows = () => false, mainRow = 'follow',
    mainFacts = null, tierOf = () => undefined, onRoute, onPin = () => {}, onSettings = () => {}, onLeave = () => {},
    onPaint = () => {} }) {
    // `key` is the active row's model (its data-row), `nav` whether the arrows made it active, `shown` the row and cell
    // last scrolled into view.
    const search = { query: '', only: '', sources: false, active: -1, key: '', cell: 0, nav: false, rows: [],
        picks: new Map(), html: '', shown: '' };
    const describe = (route, options = {}) => describeRoute(store.index(), route, { tier: tierOf(route, options) });
    el.setAttribute('role', 'grid');
    el.setAttribute('aria-label', 'Models');
    el.id ||= `${id}-list`;

    // One row: the mark, the model and its facts in the pick cell; its Pin in a cell of its own (none for Main's row,
    // a source or a door). A model whose catalog states a price its context pays says it where "paid by key" stood.
    const rowMarkup = (key, said, { tag = '', selected = false, extra = '', facts = '', pin = null, kind = '', name = '' } = {}) => {
        const pinned = pin === true;
        const pinCell = pin === null ? '' : `<span role="gridcell" class="chat-model-gridcell chat-model-pin-cell"><button type="button"`
            + ` class="chat-model-pin ui-hit-area" tabindex="-1" data-pin="${esc(key)}" aria-pressed="${pinned}" data-i18n-authored`
            + ` aria-label="${esc(fmt('Pin {model}', { model: said?.id || name }))}" title="${esc(pinned ? tr('model.unpin', 'Unpin') : tr('model.pin', 'Pin'))}">`
            + `${SHELL_ICONS.pin}</button></span>`;
        return `<div role="row" class="chat-model-row${kind ? ` chat-model-${kind}` : ''}" data-row="${esc(key)}"`
            + ` aria-selected="${selected}"${pinned ? ' data-pinned="true"' : ''}><span role="gridcell" class="chat-model-gridcell chat-model-cell">`
            + (said ? sourceMark(said) : SEARCH_ICON)
            + `<span class="chat-model-row-id" data-i18n-skip>${esc(name || said.id)}</span>`
            + (tag ? `<span class="chat-model-row-tag">${esc(tag)}</span>` : '')
            + `<span class="chat-model-row-facts" data-i18n-skip>${esc(facts || (said ? [said.window, said.sourceLabel, said.price || payWords(said)].filter(Boolean).join(' · ') : ''))}</span>${extra}</span>${pinCell}</div>`;
    };

    function paint() {
        const index = store.index();
        const status = store.status();
        const [chosen, mainRoute, following] = [current(), main(), follows()];
        const picks = new Map();  // a row's key → what a press on it does
        const parts = [];
        const shown = new Set();
        const add = (route, options = {}) => {
            const said = describe(route);
            const key = said.key;
            if (shown.has(key)) return;
            shown.add(key);
            picks.set(key, { route });
            const unavailable = said.available === false ? '<span class="chat-model-row-tag">Unavailable</span>' : '';
            parts.push(rowMarkup(key, said, { selected: !following && sameRoute(route, chosen), ...options,
                extra: unavailable, pin: store.isPinned(route) }));
        };
        const note = (words) => parts.push(`<p class="chat-model-note" role="presentation">${esc(words)}</p>`);
        const section = (words, literal = false) => parts.push(`<p class="chat-model-section" role="presentation"${literal ? ' data-i18n-skip' : ''}>${esc(words)}</p>`);
        const listRows = (found) => {
            found.slice(0, LIST_ROWS).forEach((entry) => add(entry.route));
            if (found.length > LIST_ROWS) section(fmt('{n} more · keep typing', { n: found.length - LIST_ROWS }));
        };
        if (!search.query && !search.only) {
            if (mainRow && mainRoute) {
                // Main's row: the way to follow Main (a room, a work), or Main itself (the Main chat).
                const said = describe(mainRoute, { main: true });
                picks.set('main', { main: true, route: mainRoute });
                parts.push(rowMarkup('main', said, mainRow === 'follow'
                    ? { name: tr('model.row.main', 'Main'), selected: following, kind: 'main',
                        facts: mainFacts?.() || fmt('Follows Main · {model}', { model: said.short }) }
                    : { tag: 'Main', selected: true, kind: 'main', pin: store.isPinned(mainRoute) }));
                if (mainRow === 'main') shown.add(said.key);
            }
            // A section heads only rows it adds: a pin already shown above (Main's own model) adds none.
            const pins = store.pins().filter((route) => !shown.has(routeKey(route)));
            if (pins.length) { section(tr('model.section.pinned', 'Pinned')); pins.forEach((route) => add(route)); }
            const recent = recents().filter((route) => !shown.has(routeKey(route)));
            if (recent.length) { section(tr('model.section.recent', 'Recent')); recent.forEach((route) => add(route)); }
            const others = index.rows.filter((entry) => !shown.has(entry.key));
            if (status.reading && !status.read) note('Reading models…');
            else if (!index.rows.length && (status.failed || !index.problems?.length) && !index.sources.some((entry) => entry.problems.length)) {
                // Nothing but Main can lead: say so and offer the way to where a source is added.
                note(status.failed ? 'The model catalog could not be read.' : 'Only Main\'s model can lead work here.');
                picks.set('accounts', { settings: true });
                parts.push(`<div role="row" class="chat-model-row chat-model-door" data-row="accounts" aria-selected="false">`
                    + '<span role="gridcell" class="chat-model-gridcell chat-model-cell"><span class="chat-model-row-id">Add a subscription or a key in Settings › Accounts and keys</span></span></div>');
            } else if (index.sources.length === 1) {
                // One source: its models at once, no source step.
                section(index.sources[0].label, true);
                listRows(others.map((row) => ({ route: row.route })));
            } else if (index.sources.length > 1) {
                // More sources fold under one row; opened, each is a step into its own models.
                picks.set('sources', { sources: !search.sources });
                const names = index.sources.map((entry) => entry.label);
                const named = names.length > NAMED_SOURCES + 1
                    ? fmt('{sources} + {n}', { sources: names.slice(0, NAMED_SOURCES).join(', '), n: names.length - NAMED_SOURCES }) : names.join(', ');
                parts.push(`<div role="row" class="chat-model-row chat-model-sources" data-row="sources" aria-selected="false"`
                    + ` aria-expanded="${search.sources}"><span role="gridcell" class="chat-model-gridcell chat-model-cell"><span class="chat-model-fold" aria-hidden="true">${SHELL_ICONS.chevron}</span>`
                    + `<span class="chat-model-row-id">${esc(tr('model.all_sources', 'All sources'))}</span>`
                    + `<span class="chat-model-row-facts" data-i18n-skip>${esc(named)}</span></span></div>`);
                if (search.sources) {
                    for (const entry of index.sources) {
                        picks.set(`source:${entry.id}`, { only: entry.id });
                        const facts = [entry.count === 1 ? fmt('{n} model', { n: 1 }) : entry.count ? fmt('{n} models', { n: entry.count }) : '', payWords(entry),
                            entry.problems.length ? tr('model.list_not_read', 'list not read') : ''].filter(Boolean).join(' · ');
                        parts.push(`<div role="row" class="chat-model-row chat-model-source" data-row="${esc(`source:${entry.id}`)}" aria-selected="false">`
                            + `<span role="gridcell" class="chat-model-gridcell chat-model-cell">${sourceMark(entry)}<span class="chat-model-row-id" data-i18n-skip>${esc(entry.label)}</span>`
                            + `<span class="chat-model-row-facts" data-i18n-skip>${esc(facts)}</span></span></div>`);
                    }
                }
            }
        } else {
            if (search.only && index.sources.length > 1) {
                picks.set('source:', { only: '' });
                parts.push(`<div role="row" class="chat-model-row chat-model-source" data-row="source:" aria-selected="false">`
                    + `<span role="gridcell" class="chat-model-gridcell chat-model-cell">${SEARCH_ICON}<span class="chat-model-row-id">All sources</span>`
                    + `<span class="chat-model-row-facts" data-i18n-skip>${esc(index.source(search.only)?.label || '')}</span></span></div>`);
            }
            const found = searchModels(index, search.query, search.only);
            listRows(found.map((row) => ({ route: row.route })));
            // Suggestions assist typing and are not an allowlist: what was typed can be used as it is.
            const typed = search.query ? typedRoute(index, search.query, search.only) : null;
            if (typed && !found.some((entry) => sameRoute(entry.route, typed))) {
                const said = describe(typed);
                picks.set(`typed:${said.key}`, { route: typed });
                parts.push(rowMarkup(`typed:${said.key}`, said, { kind: 'typed', name: fmt('Use “{model}” as typed', { model: search.query }),
                    facts: said.sourceLabel }));
            } else if (search.query && !typed && !found.length) {
                note('No model matches. Choose a source to use a model as typed.');
            }
            const gaps = (search.only ? [index.source(search.only)] : index.sources).filter((entry) => entry?.problems.length);
            if (status.failed) note('The model catalog could not be read; earlier suggestions are kept.');
            else if (gaps.length) note(fmt('Could not read every list: {sources}.', { sources: gaps.map((entry) => entry.label).join(', ') }));
        }
        if (!status.failed && !search.only) for (const problem of index.problems || []) note(problem);
        // A state read that changes nothing here writes nothing: the list keeps its scroll and its active row.
        const html = `<div class="chat-model-rows" role="rowgroup">${parts.join('')}</div>`;
        search.picks = picks;
        if (html !== search.html) {
            const scrolled = el.querySelector('.chat-model-rows')?.scrollTop || 0;
            search.html = html;
            el.innerHTML = html;
            const rows = el.querySelector('.chat-model-rows');
            if (rows && scrolled) rows.scrollTop = scrolled;
            search.rows = [...el.querySelectorAll('.chat-model-row')];
            search.rows.forEach((made, at) => {
                made.querySelectorAll('.chat-model-gridcell').forEach((cell, column) => { cell.id = `${id}-r${at}c${column}`; });
            });
        }
        // A pinned row moves between sections and another window's pin inserts rows: the active row follows its model,
        // never the place it left, and a model that left the list leaves nothing active.
        setActive(search.key ? search.rows.findIndex((made) => made.dataset.row === search.key) : -1, search.cell);
        input.placeholder = search.only ? fmt('{source}: type a model', { source: index.source(search.only)?.label || '' })
            : tr('model.search.placeholder', 'Model or source');
        onPaint();
    }
    const cellsOf = (row) => (row ? [...row.querySelectorAll('.chat-model-gridcell')] : []);
    function setActive(at, cell = 0) {
        const row = search.rows[at];
        search.active = row ? at : -1;
        search.key = row?.dataset.row || '';
        if (!row) search.nav = false;
        search.cell = Math.min(cell, Math.max(0, cellsOf(row).length - 1));
        search.rows.forEach((made, order) => {
            made.dataset.active = String(order === at);
            if (order === at) made.dataset.activeCell = String(search.cell); else delete made.dataset.activeCell;
        });
        const active = cellsOf(row)[search.cell];
        if (active) input.setAttribute('aria-activedescendant', active.id);
        else input.removeAttribute('aria-activedescendant');
        // Only a change of row or cell scrolls: a repaint never snaps a list the owner scrolled back to it.
        const shown = row ? `${search.key}|${search.cell}` : '';
        if (shown !== search.shown) { search.shown = shown; row?.scrollIntoView?.({ block: 'nearest' }); }
    }
    function pin(row) {
        const act = search.picks.get(row?.dataset?.row);
        if (!act?.route) return;
        onPin(act.route, !store.isPinned(act.route));
    }
    function take(row) {
        const act = search.picks.get(row?.dataset?.row);
        if (!act) return;
        if (act.settings) { onSettings(); return; }
        if (act.sources !== undefined) { search.sources = act.sources; paint(); return; }
        if (act.only !== undefined) {
            Object.assign(search, { only: act.only, query: '', active: -1, key: '', cell: 0, nav: false });
            input.value = '';
            paint();
            return;
        }
        onRoute(act.main ? null : act.route);
    }
    // A press on the list leaves the caret where it was (a pointer's press on a row never takes the field's focus away).
    const keepFocus = (event) => { if (event.target?.closest?.('.chat-model-row')) event.preventDefault(); };
    const onQuery = () => {
        search.query = input.value.trim();
        Object.assign(search, { active: -1, key: '', cell: 0, nav: false });
        paint();
        // What was typed makes its first model active, never the way back to all sources; the arrows still move the
        // caret until Up or Down chooses a row.
        if (search.query) setActive(Math.max(0, search.rows.findIndex((made) => !made.matches('.chat-model-source, .chat-model-sources'))));
    };
    const onClick = (event) => {
        const row = event.target?.closest?.('.chat-model-row');
        if (!row) return;
        if (event.target?.closest?.('.chat-model-pin')) { event.stopPropagation?.(); pin(row); return; }
        take(row);
    };
    const onKey = (event) => {
        if (event.isComposing) return;
        const count = search.rows.length;
        const row = search.rows[search.active];
        if (event.key === 'ArrowDown' || event.key === 'ArrowUp') {
            if (count) {
                setActive(event.key === 'ArrowDown' ? (search.active + 1) % count : search.active <= 0 ? count - 1 : search.active - 1, search.cell);
                search.nav = true;
            }
        } else if ((event.key === 'ArrowRight' || event.key === 'ArrowLeft') && row && search.nav) {
            // In a row the arrows chose, they move between its cells; otherwise they move the caret.
            setActive(search.active, event.key === 'ArrowRight' ? search.cell + 1 : Math.max(0, search.cell - 1));
        } else if (event.key === 'Enter') {
            // The active cell, else the first model when something was typed or a source chosen.
            if (row && search.cell === 1) { pin(row); event.preventDefault(); return; }
            const first = search.query || search.only ? search.rows.find((made) => !made.matches('.chat-model-source, .chat-model-sources')) : null;
            const target = row || first;
            if (target) take(target);
        } else if (event.key === 'Escape') {
            // Escape clears what was typed first; with nothing typed it leaves the list to its home.
            if (search.query || search.only) {
                Object.assign(search, { query: '', only: '', active: -1, key: '', cell: 0, nav: false });
                input.value = '';
                paint();
            } else onLeave({ focus: true });
        } else return;
        event.preventDefault();
        event.stopPropagation?.();
    };
    input.addEventListener('input', onQuery);
    input.addEventListener('keydown', onKey);
    el.addEventListener('mousedown', keepFocus);
    el.addEventListener('click', onClick);

    return {
        paint,
        destroy() {
            input.removeEventListener('input', onQuery);
            input.removeEventListener('keydown', onKey);
            el.removeEventListener('mousedown', keepFocus);
            el.removeEventListener('click', onClick);
        },
    };
}
