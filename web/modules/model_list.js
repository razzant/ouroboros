// The one list of models (docs/DESIGN.md "Composer owner controls"): a field and its suggestions. The composer's
// Model capsule puts the field in its strip and floats the list above the row (composer_model.js); a work card's
// model fact puts both in its popover (card_model.js). Without a query it shows Main (where the caller has a Main
// row), the recent models, then the sources that can lead a work (one source: its models at once); typed words
// search every source; any id can be used as typed. The catalog suggests and never admits (chat_model_routes.js).
import { fmt, tr } from './i18n.js';
import { sourceMark } from './model_source_mark.js';
import { escapeHtmlAttr as esc } from './ui_primitives.js';
import { describeRoute, sameRoute, searchModels, typedRoute } from './chat_model_routes.js';

// How many suggestions one list shows; the rest wait for another letter.
const LIST_ROWS = 60;
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

/** The search field both homes of the list use: a combobox over the list `listId`. */
export function modelSearchField(doc, listId) {
    const label = doc.createElement('label');
    label.className = 'chat-model-search';
    label.innerHTML = SEARCH_ICON;
    const input = doc.createElement('input');
    input.className = 'chat-model-query';
    for (const [key, value] of Object.entries({ type: 'text', role: 'combobox', autocomplete: 'off', autocorrect: 'off',
        autocapitalize: 'off', spellcheck: 'false', 'aria-autocomplete': 'list', 'aria-expanded': 'true',
        'aria-controls': listId, 'aria-label': 'Search models' })) input.setAttribute(key, value);
    input.dataset.i18nSkip = '';  // its placeholder may name a source: written through the catalog by the list
    label.append(input);
    return { label, input };
}

/**
 * The list painted into `el` and driven by `input`. `main()`, `current()` and `recents()` are the caller's routes
 * (Main's now, the chosen one, the recent ones to offer); `mainRow` puts Main's model first, tagged, and
 * `mainFacts()` says in its facts what a press on it means where that is not Main's own route facts; `rule` is the
 * line under the rows. `onRoute(route)` takes a pick, `onSettings()` the way to Settings › Accounts and keys where nothing but
 * Main can lead, `onLeave({focus})` an Escape with nothing typed (`focus`) or a Tab out; `onPaint()` follows every
 * repaint (the list may have changed size). Returns `{ paint, destroy }`.
 */
export function createModelList({ el, input, id, store, main, current, recents, mainRow = true, mainFacts = null,
    rule = '', onRoute, onSettings = () => {}, onLeave = () => {}, onPaint = () => {} }) {
    const search = { query: '', only: '', active: -1, rows: [], picks: new Map() };
    const describe = (route) => describeRoute(store.index(), route);

    // A model whose catalog states a price says it where "paid by key" stood.
    const rowMarkup = (said, { tag = '', selected = false, extra = '', facts = '' } = {}) => `<div class="chat-model-row" role="option"`
        + ` data-row="${esc(said.key)}" aria-selected="${selected}">${sourceMark(said)}`
        + `<span class="chat-model-row-id" data-i18n-skip>${esc(said.id)}</span>`
        + (tag ? `<span class="chat-model-row-tag">${esc(tag)}</span>` : '')
        + `<span class="chat-model-row-facts" data-i18n-skip>${esc(facts || [said.window, said.sourceLabel, said.price || payWords(said)].filter(Boolean).join(' · '))}</span>${extra}</div>`;

    function paint() {
        const index = store.index();
        const status = store.status();
        const [chosen, mainRoute] = [current(), main()];
        const picks = new Map();  // a row's key → what a press on it does
        const parts = [];
        const add = (route, options = {}) => {
            const said = describe(route);
            picks.set(said.key, { route });
            const unavailable = said.available === false ? '<span class="chat-model-row-tag">Unavailable</span>' : '';
            parts.push(rowMarkup(said, { selected: sameRoute(route, chosen), ...options, extra: unavailable }));
        };
        const note = (words) => parts.push(`<p class="chat-model-note" role="presentation">${esc(words)}</p>`);
        const section = (words, literal = false) => parts.push(`<p class="chat-model-section" role="presentation"${literal ? ' data-i18n-skip' : ''}>${esc(words)}</p>`);
        const listRows = (found) => {
            found.slice(0, LIST_ROWS).forEach((entry) => add(entry.route, { tag: sameRoute(entry.route, mainRoute) ? 'Main' : '' }));
            if (found.length > LIST_ROWS) section(fmt('{n} more · keep typing', { n: found.length - LIST_ROWS }));
        };
        const others = index.rows.filter((entry) => !sameRoute(entry.route, mainRoute));
        if (!search.query && !search.only) {
            if (mainRow && mainRoute) add(mainRoute, { tag: 'Main', facts: mainFacts?.() || '' });
            const recent = recents();
            if (recent.length) { section('Recent'); recent.forEach((route) => add(route)); }
            if (status.reading && !status.read) note('Reading models…');
            else if (!others.length && !index.sources.some((entry) => entry.problems.length)) {
                // Nothing but Main can lead: say so and offer the way to where a source is added.
                note(status.failed ? 'The model catalog could not be read.' : 'Only Main\'s model can lead work here.');
                picks.set('accounts', { settings: true });
                parts.push(`<div class="chat-model-row chat-model-door" role="option" data-row="accounts" aria-selected="false">`
                    + '<span class="chat-model-row-id">Add a subscription or a key in Settings › Accounts and keys</span></div>');
            } else if (index.sources.length === 1) {
                // One source: its models at once, no source step.
                section(index.sources[0].label, true);
                listRows(mainRow && mainRoute ? others : index.rows);
            } else {
                section('Sources');
                for (const entry of index.sources) {
                    picks.set(`source:${entry.id}`, { only: entry.id });
                    const facts = [entry.count === 1 ? fmt('{n} model', { n: 1 }) : entry.count ? fmt('{n} models', { n: entry.count }) : '', payWords(entry),
                        entry.problems.length ? tr('model.list_not_read', 'list not read') : ''].filter(Boolean).join(' · ');
                    parts.push(`<div class="chat-model-row chat-model-source" role="option" data-row="${esc(`source:${entry.id}`)}" aria-selected="false">`
                        + `${sourceMark(entry)}<span class="chat-model-row-id" data-i18n-skip>${esc(entry.label)}</span>`
                        + `<span class="chat-model-row-facts" data-i18n-skip>${esc(facts)}</span></div>`);
                }
            }
        } else {
            if (search.only && index.sources.length > 1) {
                picks.set('source:', { only: '' });
                parts.push(`<div class="chat-model-row chat-model-source" role="option" data-row="source:" aria-selected="false">`
                    + `${SEARCH_ICON}<span class="chat-model-row-id">All sources</span>`
                    + `<span class="chat-model-row-facts" data-i18n-skip>${esc(index.source(search.only)?.label || '')}</span></div>`);
            }
            const found = searchModels(index, search.query, search.only);
            listRows(found);
            // Suggestions assist typing and are not an allowlist: what was typed can be used as it is.
            const typed = search.query ? typedRoute(index, search.query, search.only) : null;
            if (typed && !found.some((entry) => sameRoute(entry.route, typed))) {
                const said = describe(typed);
                picks.set(`typed:${said.key}`, { route: typed });
                parts.push(`<div class="chat-model-row chat-model-typed" role="option" data-row="${esc(`typed:${said.key}`)}" aria-selected="false">`
                    + `${sourceMark(said)}<span class="chat-model-row-id" data-i18n-skip>${esc(fmt('Use “{model}” as typed', { model: search.query }))}</span>`
                    + `<span class="chat-model-row-facts" data-i18n-skip>${esc(said.sourceLabel)}</span></div>`);
            } else if (search.query && !typed && !found.length) {
                note('No model matches. Choose a source to use a model as typed.');
            }
            const gaps = (search.only ? [index.source(search.only)] : index.sources).filter((entry) => entry?.problems.length);
            if (status.failed) note('The model catalog could not be read; earlier suggestions are kept.');
            else if (gaps.length) note(fmt('Could not read every list: {sources}.', { sources: gaps.map((entry) => entry.label).join(', ') }));
        }
        el.innerHTML = `<div class="chat-model-rows">${parts.join('')}</div>`
            + (rule ? `<p class="chat-model-rule" role="presentation">${esc(rule)}</p>` : '');
        search.rows = [...el.querySelectorAll('.chat-model-row')];
        search.rows.forEach((made, at) => { made.id = `${id}-o${at}`; });
        search.picks = picks;
        setActive(Math.min(search.active, search.rows.length - 1));
        input.placeholder = search.only ? fmt('{source}: type a model', { source: index.source(search.only)?.label || '' })
            : tr('model.search.placeholder', 'Model or source');
        onPaint();
    }
    function setActive(at) {
        search.active = at;
        search.rows.forEach((made, order) => { made.dataset.active = String(order === at); });
        const active = search.rows[at];
        if (active) input.setAttribute('aria-activedescendant', active.id);
        else input.removeAttribute('aria-activedescendant');
        active?.scrollIntoView?.({ block: 'nearest' });
    }
    function take(made) {
        const act = search.picks.get(made?.dataset?.row);
        if (!act) return;
        if (act.settings) { onSettings(); return; }
        if (act.only !== undefined) {
            Object.assign(search, { only: act.only, query: '', active: -1 });
            input.value = '';
            input.focus({ preventScroll: true });
            paint();
            return;
        }
        onRoute(act.route);
    }
    const keepFocus = (event) => event.preventDefault();  // a press on a suggestion leaves the caret in the field
    const onQuery = () => {
        search.query = input.value.trim();
        search.active = search.query ? 0 : -1;
        paint();
    };
    const onClick = (event) => take(event.target?.closest?.('.chat-model-row'));
    const onKey = (event) => {
        if (event.isComposing) return;
        const count = search.rows.length;
        if (event.key === 'ArrowDown' || event.key === 'ArrowUp') {
            if (count) setActive(event.key === 'ArrowDown' ? (search.active + 1) % count : search.active <= 0 ? count - 1 : search.active - 1);
        } else if (event.key === 'Enter') {
            // The active row, else the first model when something was typed or a source chosen.
            const first = search.query || search.only ? search.rows.find((made) => !made.classList.contains('chat-model-source')) : null;
            const target = search.rows[search.active] || first;
            if (target) take(target);
        } else if (event.key === 'Escape') {
            // Escape clears what was typed first; with nothing typed it leaves the list to its home.
            if (search.query || search.only) {
                Object.assign(search, { query: '', only: '', active: -1 });
                input.value = '';
                paint();
            } else onLeave({ focus: true });
        } else if (event.key === 'Tab') {
            onLeave({ focus: false });
            return;
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
