// Add a place (owner 5 Nov: «сначала человеческое описание, источник и доступ раскрываются по необходимости»): first
// what the place is and what Ouroboros is there for, with an optional short description; then the installed
// connectors; Where only when a choice is needed; a connector's settings or Skills only when it is not ready. A new
// place lands Silent with those words as its first note, its description and its circle in one write. A place that is
// connected already shows its own notes, and only the changes ticked there are saved, under its revision.
import { escapeHtmlAttr as escapeHtml } from './utils.js';
import { tr, fmt } from './i18n.js';
import { skillIconHtml } from './skill_presentation.js';
import { connectPlace, failureText, normalizePlace, readPlaceConnector, readPlaces, updatePlace } from './consciousness_api.js';
import { authorWord } from './consciousness_words.js';
import { openPlaceSheet, openSkillsFor, sheetShell, showSheet, statusLine } from './place_sheet.js';

const silentCopy = () => tr('places.connect.silent', 'It starts Silent: it hears and thinks there and sends nothing until you make it Active.');
const wordsExample = () => tr('circles.words.example', 'A team chat: answer when asked; keep money and private matters out; bring me what matters.');
/** The owner-words field of a circle or of a new place: the owner's voice, an example as its placeholder. */
export const ownerWordsFieldHtml = (attrs, value = '', rows = 3) => `<textarea class="ui-control consciousness-owner-words" rows="${rows}" ${attrs} data-i18n-authored spellcheck="false" placeholder="${escapeHtml(wordsExample())}">${escapeHtml(value)}</textarea>`;
const button = (act, label, attrs = '', tone = 'btn-default') => `<button type="button" class="btn ${tone} btn-sm" data-connect-act="${act}" ${attrs}>${escapeHtml(label)}</button>`;
const offline = (connector) => Boolean(connector.readiness && connector.readiness.tone !== 'ok');

/** Absent is different from a known empty list and from a failed or malformed read. Bad rows never become wildcards. */
export function connectChoices(data) {
    if (!data || typeof data !== 'object' || Array.isArray(data)) return { state: 'unavailable', rows: [] };
    if (!Object.hasOwn(data, 'connect')) return { state: 'absent', rows: [] };
    if (!Array.isArray(data.connect)) return { state: 'unavailable', rows: [] };
    const rows = data.connect.filter((row) => row && typeof row.account_id === 'string' && row.account_id.trim()
        && typeof row.conversation_id === 'string' && typeof row.label === 'string' && row.label.trim())
        .map((row) => ({ accountId: row.account_id, conversationId: row.conversation_id, label: row.label }));
    return { state: rows.length ? 'known' : data.connect.length ? 'unavailable' : 'empty', rows };
}

const existingPlace = (places, connector, row) => places.find((place) => place.connector === connector.connector
    && place.accountId === row.accountId && place.conversationId === row.conversationId);

/** Step one: what the place is and what Ouroboros is there for (its first note), and an optional short description. */
export function wordsHtml(view) {
    const where = view.circle ? fmt('{circle}’s words apply there too.', { circle: `<span data-i18n-authored>${escapeHtml(view.circle.name)}</span>` }) : '';
    return `<section class="place-sheet-section">
        <h3 id="place-add-question">${tr('places.add.question', 'What is this place, and what am I here for?')}</h3>
        ${ownerWordsFieldHtml('data-connect-draft="words" aria-labelledby="place-add-question"', view.words || '', 4)}
        <p class="place-sheet-note">${tr('places.add.note', 'Your words become the place’s first note.')} ${where}</p>
        <label class="ui-field">${tr('places.description.optional', 'Description (optional)')}<input class="ui-control" data-connect-draft="description" data-i18n-authored maxlength="240" value="${escapeHtml(view.description || '')}" spellcheck="false" autocomplete="off" aria-describedby="place-add-description-help"></label>
        <div class="ui-field-help" id="place-add-description-help">${tr('places.description.help', 'Shown under its name in Circles. It says what the place is for and adds no rule.')}</div>
        <div class="place-sheet-actions">${button('next', tr('places.add.next', 'Next: where it is'))}</div>${statusLine(view.status)}
    </section>`;
}

function whereHtml(view) {
    const { selected: connector, places, choices, busy } = view;
    const rowHtml = (row, index) => {
        const existing = existingPlace(places, connector, row);
        // One family with the connector list: the row's state under its name, its verb on the right.
        const state = existing ? `<div class="harness-account-meta"><span class="ui-status" data-tone="ok">${tr('places.connected', 'Connected')}</span></div>` : '';
        const action = existing
            ? button('adopt', tr('places.choose', 'Choose'), `data-place-id="${escapeHtml(existing.id)}"${busy ? ' disabled' : ''}`)
            : button('connect', tr('places.connect.action', 'Connect'), `data-choice="${index}"${busy ? ' disabled' : ''}`);
        return `<div class="harness-account-row place-connector-row">
            <div class="harness-account-main"><strong${index === 'whole' ? '' : ' data-i18n-authored'}>${escapeHtml(row.label)}</strong></div>
            ${state}<div class="harness-account-actions">${action}</div>
        </div>`;
    };
    let note = '';
    // No status route on a connector that is on means no list to choose from (Whole account alone), not «off».
    if (!connector.source) note = offline(connector) ? tr('places.connector.off', 'Turn the connector on to choose where') : '';
    else if (!choices) note = tr('places.choices.loading', 'Reading where you can connect…');
    else if (choices.state === 'empty') note = tr('places.choices.empty', 'None offered yet.');
    else if (choices.state === 'unavailable') note = tr('places.choices.unavailable', 'Could not read the connector’s list.');
    // The connector's settings only when it is not ready, or offers nowhere to connect.
    const settings = choices?.state === 'empty' || offline(connector) ? button('settings', connector.settings
        ? tr('places.connector.settings', 'Open connector settings') : tr('places.connector.page', 'Open connector page')) : '';
    const retry = choices?.state === 'unavailable' ? button('retry-choices', tr('places.retry', 'Try again')) : '';
    const readiness = view.connectorStatus?.text ? { ...view.connectorStatus, authored: true } : connector.readiness;
    return `<section class="place-sheet-section">
        <div class="place-sheet-actions">${button('back', tr('places.connect.back', 'Back'))}</div>
        <h3>${tr('places.where', 'Where')} · <span data-i18n-authored>${escapeHtml(connector.label)}</span></h3>
        ${readiness ? statusLine(readiness.authored ? readiness : { ...readiness, text: tr(`places.readiness.${readiness.text}`, readiness.text) }) : ''}
        <p class="place-sheet-note">${silentCopy()}</p>
        ${note ? `<p class="place-sheet-note" role="status">${escapeHtml(note)}</p>` : ''}
        ${settings || retry ? `<div class="place-sheet-actions">${settings}${retry}</div>` : ''}
        <div class="place-accounts">${rowHtml({ accountId: '', conversationId: '', label: tr('places.whole', 'Whole account') }, 'whole')}
            ${(choices?.rows || []).map(rowHtml).join('')}</div>${statusLine(view.status)}
    </section>`;
}

/** The verb of a connector row: one that is off opens Skills to enable it; one with places lets Where offer them
 *  beside what can still be connected; a ready one connects. */
function connectorAction(connector, count) {
    if (offline(connector)) return button('skills', tr('places.connector.enable', 'Enable in Skills'), `data-connector="${escapeHtml(connector.connector)}"`);
    return button('choose', count ? tr('places.choose.where', 'Choose where') : tr('places.connect.action', 'Connect'), `data-connector="${escapeHtml(connector.connector)}"`);
}

export function connectListHtml({ connectors = null, places = [], error = '', selected = null, choices = null,
    status = {}, busy = false, connectorStatus = null } = {}) {
    if (error) return `<section class="place-sheet-section">${statusLine({ text: error, tone: 'error' })}${button('retry', tr('places.retry', 'Try again'))}</section>`;
    if (!connectors) return `<p class="place-sheet-section place-sheet-note" role="status">${tr('places.connect.loading', 'Reading the installed connectors…')}</p>`;
    if (selected) return whereHtml({ selected, choices, places, status, busy, connectorStatus });
    const back = `<div class="place-sheet-actions">${button('back-words', tr('places.connect.back', 'Back'))}</div>`;
    if (!connectors.length) return `<section class="place-sheet-section">${back}<p>${tr('places.connect.empty', 'No installed connector can connect a place yet.')}</p>
        ${button('browse', tr('places.connect.browse', 'Find connectors in Skills'))}</section>`;
    return `<section class="place-sheet-section">${back}<h3>${tr('places.where_from', 'Where is it?')}</h3><p class="place-sheet-note">${silentCopy()}</p>
        <div class="place-accounts">${connectors.map((connector) => {
            const n = places.filter((place) => place.connector === connector.connector).length;
            // Every row's meta line is its state, one family: connected (and how many places), not connected, or off,
            // in the readiness's own words when it has them.
            const readiness = connector.readiness && connector.readiness.tone !== 'ok' ? tr(`places.readiness.${connector.readiness.text}`, connector.readiness.text) : '';
            const state = offline(connector) ? (readiness || tr('places.connector.state.off', 'Off'))
                : n ? fmt(n === 1 ? 'Connected · {n} place' : 'Connected · {n} places', { n }) : tr('places.connector.state.none', 'Not connected');
            const tone = offline(connector) ? (readiness ? connector.readiness.tone : 'muted') : n ? 'ok' : 'muted';
            return `<div class="harness-account-row place-connector-row">
                <div class="harness-account-main"><span class="place-glyph" aria-hidden="true">${skillIconHtml(connector.icon, 18)}</span><strong data-i18n-authored>${escapeHtml(connector.label)}</strong></div>
                <div class="harness-account-meta"><span class="ui-status" data-tone="${escapeHtml(tone)}">${escapeHtml(state)}</span></div>
                <div class="harness-account-actions">${connectorAction(connector, n)}</div>
            </div>`;
        }).join('')}</div></section>`;
}

/** What choosing a connected place would change, from what the Add said; each change only where it changes something. */
export function adoptOffers({ adopt: place, circle, words = '', description = '' }) {
    const has = (text) => place.notes.some((note) => note.text.trim() === text.trim());
    return [circle && place.circle?.id !== circle.id ? 'circle' : '', words.trim() && !has(words) ? 'note' : '',
        description.trim() && description.trim() !== (place.description?.text || '') ? 'description' : ''].filter(Boolean);
}

/** A place that is connected already: its own notes as they stand, and only the changes ticked here. */
export function adoptHtml(view) {
    const { adopt: place, circle, words = '', description = '', busy } = view;
    const notes = place.notes.map((note) => `<li class="place-note${note.author === 'owner' || note.author === 'migration' ? ' is-owner' : ''}"><p data-i18n-authored>${escapeHtml(note.text)}</p><div class="place-note-foot"><span class="place-sheet-note">${escapeHtml(authorWord(note.author))}</span></div></li>`).join('');
    const label = {
        circle: escapeHtml(fmt('Put it in {circle}', { circle: circle?.name || '' })),
        note: `${escapeHtml(tr('places.adopt.note', 'Add your words as a new note:'))} <span data-i18n-authored>${escapeHtml(words.trim())}</span>`,
        description: `${escapeHtml(place.description ? tr('places.adopt.description_replace', 'Replace its description with:')
            : tr('places.adopt.description', 'Use your description:'))} <span data-i18n-authored>${escapeHtml(description.trim())}</span>`,
    };
    const options = adoptOffers(view).map((key) => `<label class="place-wake-toggle ui-hit-area"><input class="ui-checkbox" type="checkbox" data-connect-adopt="${key}"${view.picks?.[key] === false ? '' : ' checked'}><span>${label[key]}</span></label>`);
    return `<section class="place-sheet-section">
        <div class="place-sheet-actions">${button('back-where', tr('places.connect.back', 'Back'))}</div>
        <h3><span data-i18n-authored>${escapeHtml(place.label)}</span></h3>
        <p class="place-sheet-note">${tr('places.adopt.kept', 'It is connected already: its state, conversations and notes stay as they are.')}</p>
        ${notes ? `<ul class="place-note-list">${notes}</ul>` : `<p class="place-sheet-note">${tr('places.notes.none', 'No notes of its own yet.')}</p>`}
        ${options.length ? `<div class="place-wake-list">${options.join('')}</div>
        <div class="place-sheet-actions">${button('adopt-save', tr('places.adopt.save', 'Save these changes'), busy ? 'disabled' : '')}</div>`
        : `<p class="place-sheet-note">${tr('places.adopt.nothing', 'Nothing here would change it.')}</p>${button('open', tr('places.open', 'Open'), `data-place-id="${escapeHtml(place.id)}"`)}`}
        ${statusLine(view.status)}
    </section>`;
}

/** The connector's declared settings live on its skill's Settings page; fall back to its Skills entry. */
export async function openConnectorSettings(connector, openSettings) {
    if (!connector.settings || !openSettings) { openSkillsFor({ skill: connector.connector }); return; }
    if (await openSettings(`skill/${connector.connector}`) === false) return;
    const section = document.getElementById('extension-settings-sections');
    const form = [...(section?.querySelectorAll('[data-extension-settings-form]') || [])]
        .find((node) => node.dataset.skill === connector.connector && node.dataset.extensionSettingsKey?.startsWith(`${connector.connector}:${connector.settings}:`));
    (form || section)?.scrollIntoView({ block: 'start' });
}

/** `circle` = {id, name}: Add a place to that circle; null adds it to no circle. */
export function openConnectPlace({ returnFocus = document.activeElement, read = readPlaces, connect = connectPlace, update = updatePlace,
    readStatus = readPlaceConnector, openPlace = openPlaceSheet, openSettings = null, circle = null } = {}) {
    const dialog = sheetShell(escapeHtml(tr('places.add', 'Add a place')),
        circle ? fmt('To {circle}', { circle: circle.name }) : tr('circles.none', 'Not in a circle'), 'place-connect');
    const abort = new AbortController();
    const sheet = showSheet(dialog, { returnFocus, onClose: () => abort.abort() });
    const view = { connectors: null, places: [], error: '', status: {}, selected: null, choices: null, busy: false,
        circle, words: '', description: '', step: 'words', adopt: null, picks: {} };
    let generation = 0;
    const paint = () => {
        if (sheet.closed) return;
        sheet.body.innerHTML = view.step === 'words' ? wordsHtml(view) : view.step === 'adopt' ? adoptHtml(view) : connectListHtml(view);
    };
    const load = async () => {
        view.error = ''; view.connectors = null; paint();
        try { Object.assign(view, await read({ signal: abort.signal })); }
        catch (error) { view.error = fmt('Could not read the connectors: {reason}', { reason: failureText(error) }); }
        paint();
    };
    const land = (place) => {
        // A single place's answer names only its provider; the flow holds the connector's declared name.
        if (!sheet.closed) { sheet.close(); openPlace({ ...place, serviceLabel: view.selected?.label || place.serviceLabel }, { returnFocus }); }
    };
    const showAdopt = (place, status = {}) => { view.adopt = place; view.picks = {}; view.status = status; view.step = 'adopt'; paint(); };
    const connectNow = async (choice) => {
        view.busy = true; view.status = { text: tr('places.connect.busy', 'Connecting…'), tone: 'muted' }; paint();
        try {
            land(await connect({ connector: view.selected.connector, ...choice, note: view.words.trim(),
                description: view.description.trim(), circleId: circle?.id || '' }));
        } catch (error) {
            // Connected meanwhile: nothing was written; its own notes come first, the changes are chosen there.
            if (error?.status === 409 && error.body?.code === 'place_exists' && error.body.place) {
                showAdopt(normalizePlace(error.body.place), { text: tr('places.adopt.exists', 'It is connected already: choose what to change.'), tone: 'warn' });
                return;
            }
            view.status = { text: failureText(error), tone: 'error' };
        } finally { view.busy = false; paint(); }
    };
    const saveAdopt = async () => {
        const place = view.adopt;
        const offers = adoptOffers(view);  // offered changes stay ticked until the owner unticks one
        const ticked = (key) => offers.includes(key) && view.picks[key] !== false;
        const change = { ...(ticked('circle') ? { circle_id: circle.id } : {}), ...(ticked('note') ? { add_note: view.words.trim() } : {}),
            ...(ticked('description') ? { description: view.description.trim() } : {}) };
        if (!Object.keys(change).length) { view.status = { text: tr('places.adopt.none', 'Tick what to change, or go back.'), tone: 'muted' }; paint(); return; }
        view.busy = true; view.status = { text: tr('places.saving', 'Saving…'), tone: 'muted' }; paint();
        try {
            land(await update(place.id, { ...change, expected_revision: place.revision }));
        } catch (error) {
            if (error?.status === 409 && Object.hasOwn(error.body || {}, 'place')) {
                if (!error.body.place) { view.status = { text: tr('places.disconnected', 'This place is disconnected.'), tone: 'warn' }; return; }
                view.adopt = normalizePlace(error.body.place);  // what it holds now; the ticks stay as they were
                view.status = { text: tr('places.adopt.changed', 'Changed elsewhere; check it and save again.'), tone: 'warn' };
                return;
            }
            view.status = { text: failureText(error), tone: 'error' };
        } finally { view.busy = false; paint(); }
    };
    const loadChoices = async () => {
        const current = ++generation;
        view.choices = null; view.connectorStatus = null; paint();
        try {
            // A status route is optional; without one there are no declared instances to choose among.
            const data = view.selected.source ? await readStatus(view.selected, { signal: abort.signal }) : {};
            if (current !== generation || sheet.closed) return;
            view.choices = connectChoices(data);
            view.connectorStatus = typeof data.text === 'string' ? { text: data.text, tone: data.tone } : null;
        } catch { if (current === generation) view.choices = { state: 'unavailable', rows: [] }; }
        if (current !== generation || sheet.closed) return;
        // Where only when a choice is needed: a ready connector with no instances and no place yet connects its whole
        // account now. Once it has a place (one a tool or the migration made among them), «Choose where» keeps Where:
        // its Whole account is then an explicit choice beside what is connected.
        const placed = view.places.some((place) => place.connector === view.selected.connector);
        if (current === generation && view.choices?.state === 'absent' && !offline(view.selected) && !placed) { await connectNow({}); return; }
        paint();
    };
    sheet.body.addEventListener('input', (event) => {
        const field = event.target.closest?.('[data-connect-draft]');
        if (field) view[field.dataset.connectDraft] = field.value;
    });
    sheet.body.addEventListener('change', (event) => {
        const pick = event.target.closest?.('[data-connect-adopt]');
        if (pick) view.picks[pick.dataset.connectAdopt] = pick.checked;
    });
    sheet.body.addEventListener('click', async (event) => {
        const control = event.target.closest('[data-connect-act]');
        if (!control || control.disabled || view.busy) return;
        const act = control.dataset.connectAct;
        if (act === 'next') {
            if (!view.words.trim()) {
                view.status = { text: tr('places.add.words_first', 'Say in a few words what this place is and what Ouroboros is here for.'), tone: 'muted' };
                paint(); sheet.body.querySelector('[data-connect-draft="words"]')?.focus(); return;
            }
            view.status = {}; view.step = 'list'; paint();
            if (!view.connectors) await load();
            return;
        }
        if (act === 'back-words') { ++generation; view.selected = null; view.status = {}; view.step = 'words'; paint(); sheet.body.querySelector('[data-connect-draft="words"]')?.focus(); return; }
        if (act === 'back-where') { view.adopt = null; view.status = {}; view.step = 'list'; paint(); return; }
        if (act === 'retry') { await load(); return; }
        if (act === 'browse') { sheet.close(); openSkillsFor({ category: 'connectors' }); return; }
        if (act === 'skills') { sheet.close(); openSkillsFor({ skill: control.dataset.connector }); return; }
        if (act === 'back') { ++generation; view.selected = null; view.status = {}; paint(); return; }
        if (act === 'settings') { sheet.close(); await openConnectorSettings(view.selected, openSettings); return; }
        if (act === 'choose') {
            view.selected = view.connectors.find((item) => item.connector === control.dataset.connector);
            if (view.selected) await loadChoices();
            return;
        }
        if (act === 'retry-choices') { await loadChoices(); return; }
        if (act === 'adopt') {
            const place = view.places.find((item) => item.id === control.dataset.placeId);
            if (place) showAdopt(place);
            return;
        }
        if (act === 'adopt-save') { await saveAdopt(); return; }
        if (act === 'open') {
            const place = view.places.find((item) => item.id === control.dataset.placeId) || view.adopt;
            if (place) land(place);
            return;
        }
        if (act !== 'connect' || !view.selected) return;
        const choice = control.dataset.choice === 'whole' ? {} : view.choices?.rows[Number(control.dataset.choice)];
        if (choice) await connectNow(choice);
    });
    paint();
    sheet.body.querySelector('[data-connect-draft="words"]')?.focus();
    return sheet;
}
