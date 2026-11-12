// Circles (docs/DESIGN.md "Circles"): the one home of the places Ouroboros lives in, gathered into circles of people
// whose shared words every place in a circle inherits beside its own notes. The page header carries New circle; the
// body is one .ui-card directory: each circle is a row that opens (details.ui-disclosure, the Dashboard's Details
// recipe), its head the circle's name, its words in one ellipsized line and the factual count of its places; one
// circle is open at a time. Open, a circle shows its words whole in the voice of whoever wrote them (Edit swaps them
// for its name, words and model with Save, Cancel and Delete; the words themselves are not a control), Add a place and
// its place rows; the last row holds the places in no circle, open at once while there is no circle to choose. A
// circle may have a model: its places think with it unless one has its own, and the default is the consciousness
// model (owner 10.11); the head names the circle's model only when it has one, and the open circle says how many of
// its places keep their own. placeRowHtml is the app's one place row: the place's own name, the service as a quiet
// label, the state word, readiness only when it is not simply on, the time of its last captured event, its model only
// when the place overrides, and under them its optional one-line description; a press opens the place sheet. No wake
// list, spend, summary or feed here: Main is where attention goes.
import { bindPageHeader, renderPageHeader } from './page_header.js';
import { escapeHtmlAttr as escapeHtml } from './ui_primitives.js';
import { tr, fmt } from './i18n.js';
import { chipHtml, relativeAge, skillIconHtml } from './skill_presentation.js';
import { PLACES_CHANGED_EVENT, createCircle, deleteCircle, failureText, readPlaceConnector, readPlaces, updateCircle } from './consciousness_api.js';
import { placeInstance, placeModelWord, placeService, placeStateWord, placeTitle, sameWords, writtenByWord } from './consciousness_words.js';
import { getContributions } from './contributions.js';
import { bindModelChoosers, modelChooserHtml, updateModelChooser } from './model_chooser.js';
import { modelCatalog } from './model_catalog_store.js';
import { modelSuggestions, openPlaceSheet, statusLine } from './place_sheet.js';
import { catalogPrograms } from './model_program.js';
import { openConnectPlace, ownerWordsFieldHtml } from './place_connect.js';
import { openConfirmDialog } from './confirm_dialog.js';

const SEP = ' · ';
const UNASSIGNED = 'none';
const OWNER_VOICE = new Set(['owner', 'migration']);
// The state word's tone: a live place is the ok dot, a silent or paused one the muted dot, and so is a place that only
// hears because consciousness is off (the host's `only_hears`).
const STATE_TONE = { active: 'ok', silent: 'muted', paused: 'muted' };

/** «model glm-5.3»: a model named as a fact of a row or a head. The words name a model, so the overlay leaves them. */
const modelFactHtml = (model) => `<span data-i18n-fmt="model {model}">${escapeHtml(fmt('model {model}', { model: placeModelWord(model) }))}</span>`;
/** A place that thinks with a model of its own, not its circle's or the consciousness model. */
const overrides = (place) => place.modelRoute?.layer === 'place';

/** «2 h ago» from the last captured event; a place without one says so (never «no activity»). */
export const lastEventText = (place) => (place.lastActivityAt
    ? relativeAge(place.lastActivityAt) || tr('places.event.unknown', 'last event unknown')
    : tr('places.no_event', 'no captured event'));

/** One place row for every list of places: a door to its sheet (DESIGN "Doors and containers"). `source` is the row's
 *  quiet service label (the service, with the instance when two rows of the list would read the same). */
export function placeRowHtml(place, { title = '', source = '' } = {}) {
    const readiness = place.readiness && place.readiness.tone !== 'ok'
        ? tr(`places.readiness.${place.readiness.text}`, place.readiness.text) : '';
    const word = tr(`places.state.${place.state}`, placeStateWord(place.state));
    const state = place.onlyHears ? `${word}${SEP}${tr('places.only_hears.consciousness_off', 'consciousness off')}` : word;
    const when = lastEventText(place);
    const name = title || placeTitle(place);
    // The service as a quiet label, unless the name already is it (a whole account no status has named yet).
    const service = source || placeService(place);
    const where = sameWords(name, service) ? '' : service;
    const description = place.description?.text || '';
    const tone = place.onlyHears ? 'muted' : STATE_TONE[place.state] || 'muted';
    // Its model is a fact of the row only when the place overrides: an inheriting place says nothing.
    const own = overrides(place) ? place.model || place.modelRoute.value : '';
    const facts = `<span class="ui-status" data-tone="${tone}">${escapeHtml(state)}</span>${readiness
        ? `${SEP}<span data-tone="${escapeHtml(place.readiness.tone)}">${escapeHtml(readiness)}</span>` : ''}${SEP}${escapeHtml(when)}${own
        ? `${SEP}${modelFactHtml(own)}` : ''}`;
    const label = fmt(where ? 'Open {place} ({where}): {facts}' : 'Open {place}: {facts}', { place: name, where,
        facts: [state, readiness, fmt('last event {when}', { when }), own ? fmt('model {model}', { model: placeModelWord(own) }) : '']
            .filter(Boolean).join('; ') });
    return `<li><button type="button" class="place-row" data-place-open="${escapeHtml(place.id)}" data-state="${escapeHtml(place.state || '')}" aria-label="${escapeHtml(label)}">
        <span class="place-row-main"><span class="place-glyph" aria-hidden="true">${skillIconHtml(place.icon, 18)}</span><span class="consciousness-place-name" data-i18n-authored>${escapeHtml(name)}</span>${where
            ? `<span class="place-row-source" data-i18n-authored>${escapeHtml(where)}</span>` : ''}<span class="consciousness-place-facts">${facts}</span></span>
        ${description ? `<span class="place-row-meta" data-i18n-authored>${escapeHtml(description)}</span>` : ''}
    </button></li>`;
}

/** The rows of one list: titles without the circle's name, the instance on the source label where two would read the same. */
export function placeRowsHtml(places, { circle = null, statuses = {} } = {}) {
    const titled = places.map((place) => ({ place, title: placeTitle(place, { circle, status: statuses[place.connector] }) }));
    const counts = new Map();
    for (const { title } of titled) counts.set(title, (counts.get(title) || 0) + 1);
    return titled.map(({ place, title }) => {
        const instance = counts.get(title) > 1 ? placeInstance(place) : '';
        return placeRowHtml(place, { title, source: instance ? `${placeService(place)}${SEP}${instance}` : placeService(place) });
    }).join('');
}

/** «Could not move an old connection: <detail>. Connect it again.», each distinct sentence once (the places
 *  migration's leftovers, said where the places are listed). */
export function migrationProblemLines(problems = []) {
    const lines = problems.map((problem) => {
        const why = (problem.detail || problem.code.replace(/_/g, ' ')).trim().replace(/\.+$/, '');
        return `Could not move an old connection${why ? `: ${why}` : ''}. Connect it again.`;
    });
    return [...new Set(lines)];
}

/** «10 places · 1 paused»: the count, and the places that are not simply active, by state (facts, no summary). */
export function placeCountText(places) {
    const parts = [fmt(places.length === 1 ? '{n} place' : '{n} places', { n: places.length })];
    for (const state of ['silent', 'paused']) {
        const n = places.filter((place) => place.state === state).length;
        if (n) parts.push(fmt(`{n} ${state}`, { n }));
    }
    return parts.join(SEP);
}

/** «2 places here keep a model of their own and do not follow the circle's.»: the member places that override, said
 *  in the open circle because the circle's model does not reach them; '' when none does. */
export function ownModelsText(circle, places) {
    const n = places.filter(overrides).length;
    if (!n) return '';
    if (circle.model) {
        return fmt(n === 1 ? '{n} place here keeps a model of its own and does not follow the circle\'s.'
            : '{n} places here keep a model of their own and do not follow the circle\'s.', { n });
    }
    return fmt(n === 1 ? '{n} place here uses a model of its own.' : '{n} places here use a model of their own.', { n });
}

/** A circle's words in the voice of whoever wrote them: my human's in the owner's tint, Ouroboros's neutral and named. */
export function circleWordsHtml(circle) {
    if (!circle.text) return `<p class="consciousness-copy">${tr('circles.words.empty', 'No words yet. Edit says who is here and what Ouroboros is here for.')}</p>`;
    const owner = OWNER_VOICE.has(circle.textAuthor);
    return `<div class="place-note circle-words-text${owner ? ' is-owner' : ''}"><p data-i18n-authored>${escapeHtml(circle.text)}</p>${owner
        ? '' : `<div class="place-note-foot">${chipHtml(writtenByWord(circle.textAuthor), 'neutral')}</div>`}</div>`;
}

const rowsHtml = (places, options) => (places.length
    ? `<ul class="place-rows">${placeRowsHtml(places, options)}</ul>`
    : `<p class="consciousness-copy">${tr('circles.empty', 'No place here yet.')}</p>`);
const actButton = (act, label, attrs = '', tone = 'btn-default') => `<button type="button" class="btn ${tone} btn-sm" data-circles-act="${act}" ${attrs}>${escapeHtml(label)}</button>`;

/** The name, words and model of a circle being made or edited: the owner's voice in its field; Delete only for a
 *  saved one. The model is the place sheet's own chooser: suggestions assist typing, and an empty field inherits. */
function circleEditorHtml(key, { name = '', text = '', model = '' }, { saved = false, busy = false, models = null, catalog, modelsFailed = false, own = '' } = {}) {
    const id = escapeHtml(key);
    const disabled = busy ? ' disabled' : '';
    const help = [tr('circles.model.help', 'The model its places think with when they wake; empty uses the Consciousness model in Models. A place may keep its own.'),
        own, modelsFailed ? tr('places.models.unavailable', 'Model suggestions could not be read; your model is kept.') : ''].filter(Boolean).join(' ');
    return `<div class="circle-words circle-editor">
        <label class="ui-field">${tr('circles.name', 'Name')}<input class="ui-control" data-circles-draft="${id}:name" data-i18n-authored value="${escapeHtml(name)}" autocomplete="off" spellcheck="false"></label>
        <label class="ui-field">${tr('circles.words', 'Its words: who is here and what Ouroboros is here for')}${ownerWordsFieldHtml(`data-circles-draft="${id}:text"`, text, 4)}</label>
        <label class="ui-field">${tr('circles.model', 'Model')}${modelChooserHtml(`data-circles-draft="${id}:model" aria-label="${escapeHtml(tr('circles.model', 'Model'))}" aria-describedby="circle-model-help-${id}"`,
            model, `circle-model-options-${id}`, models || [], { placeholder: tr('circles.model.empty', 'Empty uses the consciousness model'), programs: catalogPrograms(catalog) })}</label>
        <div class="ui-field-help" id="circle-model-help-${id}" data-i18n-fmt="circles.model.help">${escapeHtml(help)}</div>
        <div class="place-sheet-actions">${actButton(saved ? 'save' : 'create', tr('circles.save', 'Save'), `data-circle="${id}"${disabled}`)}${actButton(saved ? 'cancel' : 'cancel-new', tr('circles.cancel', 'Cancel'), `data-circle="${id}"${disabled}`, 'btn-ghost')}${saved
            ? `<span class="circle-editor-end">${actButton('delete', tr('circles.delete', 'Delete circle'), `data-circle="${id}"${disabled}`, 'btn-ghost place-danger')}</span>` : ''}</div>
    </div>`;
}

/** One circle: a row of the directory. Its head (the summary) is the one control that opens it; open, the body has
 *  its words, Edit, Add a place and its rows. */
export function circleFoldHtml(circle, places, ui = {}, statuses = {}) {
    const id = escapeHtml(circle.id);
    const open = ui.open === circle.id;
    const drafts = ui.drafts || {};
    const editing = open && ui.editing === circle.id;
    const own = ownModelsText(circle, places);
    const words = editing
        ? circleEditorHtml(circle.id, { name: drafts[`${circle.id}:name`] ?? circle.name, text: drafts[`${circle.id}:text`] ?? circle.text,
            model: drafts[`${circle.id}:model`] ?? circle.model }, { saved: true, busy: ui.busy, models: ui.models, catalog: ui.catalog, modelsFailed: ui.modelsFailed, own })
        : `<div class="circle-words">${circleWordsHtml(circle)}${own ? `<p class="consciousness-copy" data-circle-own-models>${escapeHtml(own)}</p>` : ''}
            <div class="place-sheet-actions">${actButton('edit', tr('circles.edit', 'Edit'), `data-circle="${id}"`, 'btn-ghost')}${actButton('add-place', tr('circles.add_place', 'Add a place'), `data-circle="${id}"`)}</div>
        </div>`;
    const line = circle.text.split('\n')[0];
    // The head names the circle's model only when it has one: without one its places follow the consciousness model.
    return `<details class="ui-disclosure circle-fold" data-circle="${id}"${open ? ' open' : ''}>
        <summary class="circle-head">
            <span class="circle-head-main"><span class="dashboard-card-title circle-head-name" data-i18n-authored>${escapeHtml(circle.name)}</span><span class="circle-head-line"${line ? ' data-i18n-authored' : ''}>${escapeHtml(line)}</span></span>
            <span class="consciousness-place-facts circle-head-count">${escapeHtml(placeCountText(places))}${circle.model ? `${SEP}${modelFactHtml(circle.model)}` : ''}</span>
        </summary>
        <div class="circle-body">${words}${statusLine(ui.status?.[circle.id])}${rowsHtml(places, { circle, statuses })}</div>
    </details>`;
}

/** The last row: places in no circle (only their own notes apply there); open while there is no circle at all, so a
 *  new install shows its places rather than a folded page. */
export function unassignedFoldHtml(places, ui = {}, statuses = {}, { alone = false } = {}) {
    if (!places.length && !alone) return '';
    const open = alone || ui.open === UNASSIGNED;
    return `<details class="ui-disclosure circle-fold" data-circle="${UNASSIGNED}"${open ? ' open' : ''}>
        <summary class="circle-head">
            <span class="circle-head-main"><span class="dashboard-card-title circle-head-name">${tr('circles.none', 'Not in a circle')}</span><span class="circle-head-line">${tr('circles.none.line', 'Only their own notes apply here')}</span></span>
            <span class="consciousness-place-facts circle-head-count">${escapeHtml(placeCountText(places))}</span>
        </summary>
        <div class="circle-body">
            <p class="consciousness-copy">${places.length ? tr('circles.none.note', 'Only its own notes apply in a place here. Open one to choose its circle.')
        : tr('circles.none.empty', 'No place yet. Add one to start; a circle gathers places that share words.')}</p>
            <div class="place-sheet-actions">${actButton('add-place', tr('circles.add_place', 'Add a place'), `data-circle="${UNASSIGNED}"`)}</div>
            ${statusLine(ui.status?.[UNASSIGNED])}${places.length ? rowsHtml(places, { statuses }) : ''}
        </div>
    </details>`;
}

/** The card a New circle opens with: name, words, Save and Cancel. */
export function newCircleCardHtml(ui = {}) {
    const drafts = ui.drafts || {};
    return `<section class="ui-card consciousness-card" aria-label="${escapeHtml(tr('circles.new', 'New circle'))}" data-circle-new>
        <div class="consciousness-card-head"><h3 class="dashboard-card-title">${tr('circles.new', 'New circle')}</h3></div>
        ${circleEditorHtml('new', { name: drafts['new:name'] || '', text: drafts['new:text'] || '', model: drafts['new:model'] || '' },
            { busy: ui.busy, models: ui.models, catalog: ui.catalog, modelsFailed: ui.modelsFailed })}
        ${statusLine(ui.status?.new)}
    </section>`;
}

/** The circles with their places, in the owner's order, then the places in no circle (or in a circle the read lacks). */
export function groupPlaces(circles, places) {
    const known = new Set((circles || []).map((circle) => circle.id));
    const groups = (circles || []).map((circle) => ({ circle, places: (places || []).filter((place) => place.circle?.id === circle.id) }));
    const unassigned = (places || []).filter((place) => !known.has(place.circle?.id));
    return { groups, unassigned };
}

// The window's one catalog read (model_catalog_store.js) as the Model field's suggestions; a read that failed with
// nothing listed earlier rejects, so the field says its suggestions could not be read.
async function readCatalog() {
    await Promise.all([modelCatalog.read(), modelCatalog.readConnections()]);  // the tiers say whether a price is paid
    if (modelCatalog.status().failed && !modelCatalog.catalog()?.items?.length) throw new Error('The model catalog could not be read.');
    return modelCatalog.catalog();
}

/** Mount the page: its shell now, its data when it is shown. */
export function initCircles({ state, showPage, openSettings, read = readPlaces, readStatus = readPlaceConnector,
    openSheet = openPlaceSheet, openConnect = openConnectPlace, askConfirm = openConfirmDialog, readModels = readCatalog,
    writes = { createCircle, updateCircle, deleteCircle }, contributions = getContributions() } = {}) {
    const page = document.createElement('div');
    page.id = 'page-circles';
    page.className = 'page app-page-glass';
    page.innerHTML = `
        ${renderPageHeader({ page: 'circles', actionsHtml: actButton('new', tr('circles.new', 'New circle')) })}
        <div class="dashboard-shell">
            <div class="dashboard-body scroll-fade-y circles-body" data-circles-body></div>
        </div>
    `;
    document.getElementById('content').appendChild(page);
    const disposeBar = bindPageHeader(page.querySelector('.app-page-header'));
    const body = page.querySelector('[data-circles-body]');
    // One circle open at a time; the first one opens on the first visit.
    const ui = { open: '', opened: false, editing: '', drafts: {}, creating: false, status: {}, busy: false, models: null, modelsFailed: false };
    let view = null, readError = '', generation = 0, behind = false;
    let disposeChoosers = () => {}, modelsAsked = false, destroyed = false;
    // The connectors' status (the account's own name for a whole-account place), read once per declared route: each
    // connector's read is {url, data}, data null while it is out. A failed read leaves nothing, so the next refresh asks
    // again; a connector gone or declaring another route forgets its read, and an answer for that read is dropped.
    const statusReads = new Map();
    const statuses = () => Object.fromEntries([...statusReads].filter(([, read]) => read.data).map(([name, read]) => [name, read.data]));
    // A fold inserted open fires its own toggle a moment later: an echo of the paint, never the owner opening it.
    const paintedOpen = new WeakSet();
    const shown = () => !document.hidden && state?.activePage === 'circles';
    const circleById = (id) => (view?.circles || []).find((entry) => entry.id === id) || null;

    function html() {
        const top = readError ? { text: readError, tone: 'error' } : !view ? { text: tr('circles.reading', 'Reading…'), tone: '' } : ui.status.page;
        const moved = migrationProblemLines(view?.migrationProblems || []).map((line) => `<p class="consciousness-foot ui-status" data-tone="warn" role="status">${escapeHtml(line)}</p>`).join('');
        let directory = '';
        if (view) {
            const { groups, unassigned } = groupPlaces(view.circles, view.places);
            const answered = statuses();
            directory = `<section class="ui-card circles-directory" aria-label="${escapeHtml(tr('circles.title', 'Circles'))}">`
                + groups.map(({ circle, places }) => circleFoldHtml(circle, places, ui, answered)).join('')
                + unassignedFoldHtml(unassigned, ui, answered, { alone: !groups.length }) + '</section>';
        }
        return `<p class="ui-status consciousness-read-status" role="status"${top?.tone ? ` data-tone="${escapeHtml(top.tone)}"` : ''}>${escapeHtml(top?.text || '')}</p>${moved}`
            + (ui.creating ? newCircleCardHtml(ui) : '') + directory;
    }

    // A repaint keeps the reader's place: the scroll, the focused control and a field's selection.
    function paint() {
        const scroll = body.scrollTop;
        const active = document.activeElement;
        const key = body.contains(active) ? focusKeyOf(active) : '';
        const selection = active?.selectionStart == null ? null : [active.selectionStart, active.selectionEnd, active.selectionDirection];
        disposeChoosers();
        body.innerHTML = html();
        disposeChoosers = bindModelChoosers(body);
        for (const fold of body.querySelectorAll('.circle-fold')) if (fold.hasAttribute('open')) paintedOpen.add(fold);
        body.scrollTop = scroll;
        if (!key) return;
        const node = body.querySelector(key);
        node?.focus({ preventScroll: true });
        if (selection && node?.setSelectionRange) node.setSelectionRange(...selection);
    }

    // Settings broadcasts every catalog read, a failed one too: that one leaves the suggestions as they are.
    const onBroadcast = (event) => { if (Array.isArray(event?.detail?.items) && event.detail.read_state !== 'failed') onCatalog(event); };
    function onCatalog(event) {
        if (destroyed) return;
        const data = event.detail;
        ui.models = modelSuggestions(data, { tierPriced: modelCatalog.tierPriced('lead') });
        ui.catalog = data;
        for (const field of body.querySelectorAll('[data-model-chooser]')) updateModelChooser(field, ui.models, catalogPrograms(data));
    }
    // The prices beside the suggestions follow the tier Settings saves and the page's language.
    const resay = () => { if (ui.catalog) onCatalog({ detail: ui.catalog }); };

    // The Model field's suggestions, read once when an editor first shows; the field works without them (any id can be
    // typed), and they reach the open field without a repaint, so the caret stays where it is.
    async function readSuggestions() {
        if (modelsAsked) return;
        modelsAsked = true;
        try {
            onCatalog({ detail: await readModels() });
        } catch {
            if (destroyed) return;
            modelsAsked = false;  // asked again by the next editor
            ui.modelsFailed = true;
            paint();
        }
    }

    // One read in flight at a time; the newest answer paints. Places and circles come from one read; the connectors'
    // status follows it (readStatuses) and repaints when it comes.
    async function refresh() {
        const current = ++generation;
        behind = false;
        try {
            const data = await read();
            if (current !== generation) return;
            if (!Array.isArray(data.circles)) throw new Error(tr('circles.unavailable', 'Circles could not be read.'));
            view = data;
            readError = '';
        } catch (error) {
            if (current !== generation) return;
            readError = fmt('Places could not be read: {reason}', { reason: failureText(error) });
        }
        if (view && !ui.opened) { ui.opened = true; ui.open = view.circles[0]?.id || UNASSIGNED; }
        readStatuses();
        paint();
    }

    // Each declared route is asked once, whichever refresh is the newest when it answers: a read still out is not asked
    // again. A shown page repaints with the answer; a hidden one is behind, and reads again when it is shown.
    function readStatuses() {
        const declared = new Map((view?.connectors || []).filter((connector) => connector.source?.url)
            .map((connector) => [connector.connector, connector]));
        for (const [name, read] of statusReads) if (declared.get(name)?.source.url !== read.url) statusReads.delete(name);
        for (const [name, connector] of declared) {
            if (statusReads.has(name)) continue;
            const read = { url: connector.source.url, data: null };
            statusReads.set(name, read);
            readStatus(connector).then((data) => {
                if (statusReads.get(name) !== read) return;  // another route since, or a disposed page
                if (!data || typeof data !== 'object') { statusReads.delete(name); return; }  // asked again, as a failure is
                read.data = data;
                if (shown()) paint(); else behind = true;
            }).catch(() => { if (statusReads.get(name) === read) statusReads.delete(name); });
        }
    }

    // A sheet gives focus back where it was opened; a write repaints this page meanwhile, so its control is named
    // at close: the row (its circle's head when that circle is folded) or the circle's Add a place.
    const current = (selector) => () => {
        const node = body.querySelector(selector);
        return node?.getClientRects().length ? node : node?.closest('.circle-fold')?.querySelector('summary') || null;
    };

    function addPlace(circle) {
        openConnect({ returnFocus: current(`[data-circles-act="add-place"][data-circle="${CSS.escape(circle?.id || UNASSIGNED)}"]`),
            openSettings, circle: circle ? { id: circle.id, name: circle.name } : null });
    }

    // The fields stay open while a write is out: `onConflict` hears the circle as it stands now, the facts a resave uses.
    async function write(key, run, onConflict = null) {
        if (ui.busy) return false;
        ui.busy = true; ui.status = { [key]: { text: tr('circles.saving', 'Saving…'), tone: 'muted' } }; paint();
        try {
            await run();
            return true;
        } catch (error) {
            const conflict = error?.status === 409 && error.body?.code === 'circle_revision_conflict';
            ui.status = { [key]: { text: conflict ? tr('places.conflict', 'Changed elsewhere; your text is kept.') : failureText(error),
                tone: conflict ? 'warn' : 'error' } };
            if (conflict) { onConflict?.(error.body.circle); void refresh(); }
            return false;
        } finally {
            ui.busy = false;
            paint();
        }
    }

    // A draft is its name and words with the revision they were written on (`{id}:revision`, set at the first key).
    // After a confirmed write, what it sent is done; words typed since stay, as an edit of `circle` on the revision
    // that write made. True when such words stay.
    function keepNewer(from, circle, sent) {
        let kept = false;
        for (const field of ['name', 'text', 'model']) {
            const draft = ui.drafts[`${from}:${field}`];
            delete ui.drafts[`${from}:${field}`];
            if (draft !== undefined && draft !== sent[field]) { ui.drafts[`${circle.id}:${field}`] = draft; kept = true; }
        }
        delete ui.drafts[`${from}:revision`];
        if (kept) ui.drafts[`${circle.id}:revision`] = circle.revision;
        return kept;
    }

    async function remove(circle, opener) {
        const members = groupPlaces(view.circles, view.places).groups.find((group) => group.circle.id === circle.id)?.places.length || 0;
        const ok = await askConfirm({
            title: fmt('Delete {circle}?', { circle: circle.name }),
            body: fmt(members === 1 ? 'Its {n} place stays connected and keeps its own notes, but loses these words.'
                : 'Its {n} places stay connected and keep their own notes, but lose these words.', { n: members }),
            details: circle.text ? { summary: tr('circles.delete.words', 'Its words'), rows: [{ label: circle.name, value: circle.text }] } : null,
            confirmLabel: tr('circles.delete.confirm', 'Delete'), danger: true,
        });
        if (!ok) { opener?.focus?.(); return; }
        let removed = null;
        const done = await write(circle.id, async () => { removed = await writes.deleteCircle(circle.id, circle.revision); });
        if (!done) return;
        ui.editing = ''; ui.open = UNASSIGNED;
        Object.keys(ui.drafts).filter((key) => key.startsWith(`${circle.id}:`)).forEach((key) => delete ui.drafts[key]);
        ui.status = { page: { text: fmt(removed.detached.length === 1 ? 'Deleted {circle}; its place is in no circle now.'
            : 'Deleted {circle}; its {n} places are in no circle now.', { circle: circle.name, n: removed.detached.length }), tone: 'ok' } };
        await refresh();
    }

    const onClick = async (event) => {
        const row = event.target.closest?.('[data-place-open]');
        if (row && body.contains(row)) {
            const place = (view?.places || []).find((entry) => entry.id === row.dataset.placeOpen);
            if (place) openSheet(place, { returnFocus: current(`[data-place-open="${CSS.escape(place.id)}"]`), circles: view.circles,
                connectorStatus: statusReads.get(place.connector)?.data || null });
            return;
        }
        const button = event.target.closest?.('[data-circles-act]');
        if (!button || button.disabled || ui.busy) return;
        const act = button.dataset.circlesAct;
        const circle = circleById(button.dataset.circle);
        if (act === 'new') {
            ui.creating = true; ui.status = {}; ui.modelsFailed = false; paint();
            void readSuggestions();
            body.scrollTop = 0;
            body.querySelector('[data-circles-draft="new:name"]')?.focus();
            return;
        }
        if (act === 'cancel-new') { ui.creating = false; ['name', 'text', 'model'].forEach((field) => delete ui.drafts[`new:${field}`]); ui.status = {}; paint(); return; }
        if (act === 'create') {
            const name = String(ui.drafts['new:name'] || '').trim();
            if (!name) { ui.status = { new: { text: tr('circles.name.first', 'Name the circle first.'), tone: 'muted' } }; paint(); body.querySelector('[data-circles-draft="new:name"]')?.focus(); return; }
            const sent = { name: ui.drafts['new:name'], text: ui.drafts['new:text'], model: ui.drafts['new:model'] };
            let made = null;
            const model = String(sent.model || '').trim();  // a circle made without one sends none
            if (!await write('new', async () => { made = await writes.createCircle({ name, text: String(sent.text || '').trim(), ...(model ? { model } : {}) }); })) return;
            // Made once: words typed after the press open as its edit (the caret where it was), never a second New circle.
            const active = document.activeElement, typing = active?.dataset?.circlesDraft || '';
            const caret = active?.selectionStart == null ? null : [active.selectionStart, active.selectionEnd, active.selectionDirection];
            const kept = keepNewer('new', made, sent);
            ui.creating = false; ui.open = made.id; ui.editing = kept ? made.id : ''; ui.status = {};
            await refresh();
            if (!kept) body.querySelector(`[data-circle="${CSS.escape(made.id)}"] [data-circles-act="add-place"]`)?.focus();
            else if (typing.startsWith('new:')) {
                const field = body.querySelector(`[data-circles-draft="${CSS.escape(`${made.id}:${typing.slice(4)}`)}"]`);
                field?.focus({ preventScroll: true });
                if (caret) field?.setSelectionRange?.(...caret);
            }
            return;
        }
        if (act === 'add-place') { addPlace(circle); return; }
        if (!circle) return;
        if (act === 'edit') {
            ui.open = circle.id; ui.editing = circle.id; ui.status = {}; ui.modelsFailed = false; paint();
            void readSuggestions();
            const field = body.querySelector(`[data-circles-draft="${CSS.escape(circle.id)}:text"]`);
            field?.focus({ preventScroll: true }); field?.setSelectionRange?.(field.value.length, field.value.length);
            return;
        }
        const editAgain = () => body.querySelector(`[data-circle="${CSS.escape(circle.id)}"] [data-circles-act="edit"]`)?.focus();
        if (act === 'cancel') {
            ui.editing = ''; ['name', 'text', 'model', 'revision'].forEach((field) => delete ui.drafts[`${circle.id}:${field}`]); ui.status = {}; paint(); editAgain();
            return;
        }
        if (act === 'save') {
            const name = String(ui.drafts[`${circle.id}:name`] ?? circle.name).trim();
            const words = String(ui.drafts[`${circle.id}:text`] ?? circle.text).trim();
            // An emptied Model field clears the circle's model: its places follow the consciousness model again.
            const model = String(ui.drafts[`${circle.id}:model`] ?? circle.model).trim();
            if (!name) { ui.status = { [circle.id]: { text: tr('circles.name.first', 'Name the circle first.'), tone: 'muted' } }; paint(); return; }
            const change = { ...(name !== circle.name ? { name } : {}), ...(words !== circle.text ? { text: words } : {}),
                ...(model !== circle.model ? { model } : {}) };
            if (!Object.keys(change).length) { ui.editing = ''; ui.status = { [circle.id]: { text: tr('circles.unchanged', 'Nothing to save.'), tone: 'muted' } }; paint(); editAgain(); return; }
            // Sent on the draft's own revision: a newer read never turns another client's edit into ours to overwrite.
            // A conflict re-bases the draft on the circle as it stands, so only a second Save writes over it.
            const sent = { name: ui.drafts[`${circle.id}:name`], text: ui.drafts[`${circle.id}:text`], model: ui.drafts[`${circle.id}:model`] };
            let saved = null;
            if (!await write(circle.id, async () => { saved = await writes.updateCircle(circle.id, change, ui.drafts[`${circle.id}:revision`] ?? circle.revision); },
                (now) => { if (Number.isSafeInteger(now?.revision)) ui.drafts[`${circle.id}:revision`] = now.revision; else delete ui.drafts[`${circle.id}:revision`]; })) return;
            const kept = keepNewer(circle.id, saved, sent);
            if (!kept) ui.editing = '';
            ui.status = { [circle.id]: { text: kept ? tr('circles.saved.newer', 'Saved. Newer edits are still unsaved.') : tr('circles.saved', 'Saved.'), tone: kept ? 'muted' : 'ok' } };
            await refresh();
            if (!kept) editAgain();
            return;
        }
        if (act === 'delete') await remove(circle, button);
    };
    // One circle open at a time: opening a row closes the row that was open (the summary is the control).
    const onToggle = (event) => {
        const fold = event.target.closest?.('.circle-fold');
        if (!fold || !body.contains(fold)) return;
        // Its echo may come after the page chose another circle (a new or deleted one) and before that repaint.
        if (paintedOpen.delete(fold) && fold.open) return;
        if (fold.open) {
            ui.open = fold.dataset.circle;
            for (const other of body.querySelectorAll('.circle-fold[open]')) if (other !== fold) other.open = false;
        } else if (ui.open === fold.dataset.circle) {
            ui.open = '';
            if (ui.editing === fold.dataset.circle) ui.editing = '';
        }
    };
    const onInput = (event) => {
        const field = event.target.closest?.('[data-circles-draft]');
        if (!field) return;
        const key = field.dataset.circlesDraft, id = key.slice(0, key.lastIndexOf(':'));
        // The first key fixes the revision the draft is written on; later reads repaint around it.
        if (!(`${id}:revision` in ui.drafts) && circleById(id)) ui.drafts[`${id}:revision`] = circleById(id).revision;
        ui.drafts[key] = field.value;
    };
    const onShown = (event) => { if (event?.detail?.page === 'circles') void refresh(); };
    const onChanged = () => { if (shown()) void refresh(); else behind = true; };
    const onVisibility = () => { if (behind && shown()) void refresh(); };
    // Another surface asks for this page ({}), or one circle ({circle}) in its edit state ({edit}): the place sheet's
    // "Edit in circle ›". That circle opens (the open one closes) and the page scrolls to it.
    const onOpen = async (event) => {
        const { circle = '', edit = false } = event?.detail || {};
        if (!await showPage?.('circles')) return;
        if (!circle) return;
        ui.open = circle; ui.opened = true;
        if (edit) { ui.editing = circle; void readSuggestions(); }
        const land = () => {
            const fold = body.querySelector(`[data-circle="${CSS.escape(circle)}"]`);
            if (!fold) return false;
            fold.scrollIntoView({ block: 'start' });
            if (edit) fold.querySelector('[data-circles-draft$=":text"]')?.focus({ preventScroll: true });
            return true;
        };
        paint();
        if (!land()) { await refresh(); land(); }
    };
    const offLifecycle = contributions.onLifecycle(onChanged);
    page.addEventListener('click', onClick);
    body.addEventListener('toggle', onToggle, true);
    body.addEventListener('input', onInput);
    window.addEventListener('ouro:page-shown', onShown);
    window.addEventListener(PLACES_CHANGED_EVENT, onChanged);
    window.addEventListener('ouro:open-circles', onOpen);
    document.addEventListener('visibilitychange', onVisibility);
    document.addEventListener('settings-model-catalog:updated', onBroadcast);
    const offTiers = modelCatalog.subscribe(resay);
    window.addEventListener('ouro:language-changed', resay);
    paint();
    if (shown()) void refresh();
    return {
        page,
        refresh,
        destroy() {
            generation += 1;
            destroyed = true;
            disposeChoosers();
            statusReads.clear();  // a late status answer finds no read of this page
            offLifecycle();
            disposeBar();
            page.removeEventListener('click', onClick);
            body.removeEventListener('toggle', onToggle, true);
            body.removeEventListener('input', onInput);
            window.removeEventListener('ouro:page-shown', onShown);
            window.removeEventListener(PLACES_CHANGED_EVENT, onChanged);
            window.removeEventListener('ouro:open-circles', onOpen);
            document.removeEventListener('visibilitychange', onVisibility);
            document.removeEventListener('settings-model-catalog:updated', onBroadcast);
            offTiers();
            window.removeEventListener('ouro:language-changed', resay);
        },
    };
}

function focusKeyOf(node) {
    const draft = node.closest?.('[data-circles-draft]');
    if (draft) return `[data-circles-draft="${CSS.escape(draft.dataset.circlesDraft)}"]`;
    const act = node.closest?.('[data-circles-act]');
    if (act) return `[data-circles-act="${act.dataset.circlesAct}"]${act.dataset.circle ? `[data-circle="${CSS.escape(act.dataset.circle)}"]` : ''}`;
    const row = node.closest?.('[data-place-open]');
    if (row) return `[data-place-open="${CSS.escape(row.dataset.placeOpen)}"]`;
    const head = node.closest?.('.circle-fold > summary');
    return head ? `[data-circle="${CSS.escape(head.parentElement.dataset.circle)}"] > summary` : '';
}
