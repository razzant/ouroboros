// A place's sheet: everything the owner decides about one place, on the delivered-document reader's geometry
// (a dialog over the page, a full sheet on a phone), like the skill detail sheet. Its state (Active · Silent ·
// Paused), what wakes it there and who added each item, the owner's accounts there, the notes with their
// authors, the working folder (an address, not a fence) and the model. Every change is one owner write through
// consciousness_api.js and repaints from the record the server answers; nothing is saved by a guess. The
// Connect flow lives here too: an installed connector's place is created from its declaration, no id typed.
import { bindDialogFocus } from './ui_interactions.js';
import { escapeHtmlAttr as escapeHtml } from './ui_primitives.js';
import { renderSegmentedField } from './page_header.js';
import { openPlaceHistory } from './place_history.js';
import { chipHtml, skillIconHtml } from './skill_presentation.js';
import { connectPlace, disconnectPlace, PLACES_CHANGED_EVENT, readPlaces, updatePlace } from './consciousness_api.js';
import {
    PLACE_STATES, SILENT_CARD, accountIdText, authorWord, capitalized, listWords, placeScopeText, placeStateOf, wakeFactWord,
} from './consciousness_words.js';

/** The one «Connect a place» control (Dashboard and Settings draw this, never their own); a press opens the flow. */
export const connectPlaceButtonHtml = () => '<button type="button" class="btn btn-default btn-sm" data-place-connect>Connect a place</button>';

/** Ask the Skills page for one installed skill's card, or Browse filtered to connectors. */
export function openSkillsFor(detail) {
    globalThis.window?.dispatchEvent?.(new CustomEvent('ouro:open-skills', { detail }));
}

const OWNER_VOICE = new Set(['owner', 'migration']);
const shortDate = (value) => {
    const time = Date.parse(String(value || ''));
    return Number.isFinite(time) ? new Date(time).toLocaleDateString('en-GB', { day: 'numeric', month: 'short' }) : '';
};
const statusLine = (status) => (status?.text
    ? `<p class="place-sheet-status ui-status" data-tone="${escapeHtml(status.tone || 'muted')}" role="status">${escapeHtml(status.text)}</p>` : '');
const section = (key, title, inner, ui) => `<section class="place-sheet-section" data-place-section="${key}">
    ${title ? `<h3>${escapeHtml(title)}</h3>` : ''}${inner}${statusLine(ui.status?.[key])}</section>`;

function stateHtml(place, ui) {
    const state = placeStateOf(place.state);
    const silent = place.state === 'silent';
    const options = PLACE_STATES.map(({ value, label }) => ({ value, label }));
    const group = renderSegmentedField({ target: `place-state-${place.id}`, options, modifier: 'data-place-state', title: 'Changes at once' });
    // What the chosen state does: a quiet note in the tray; what Silent guarantees is a host fact in the
    // System voice, under the tray (fills never stack).
    return `<div class="settings-effort-card place-state">
            <label id="place-state-label">What it does here</label>
            ${group}
            ${silent ? '' : `<p class="place-state-note">${escapeHtml(state?.note || 'Its state could not be read.')}</p>`}
        </div>
        ${silent ? `<p class="place-silent-card" role="note">${escapeHtml(`${state.note} ${SILENT_CARD}`)}</p>` : ''}`;
}

// What the wake list does in each state: Silent still wakes it (it reads and thinks; what it would say or change
// there is held), Paused does not. An unreadable state claims nothing.
const WAKE_NOTES = {
    active: 'Everything else here is heard and waits for its own clock.',
    silent: 'Silent: these still wake it to read and think; what it would say or change here is held.',
    paused: 'While this place is not Active, nothing here wakes it.',
};

function wakesHtml(place) {
    const listed = new Set(place.wakes.map((item) => item.fact));
    const items = place.wakes.map((item) => `<li class="place-wake-item">
            <span>${escapeHtml(capitalized(wakeFactWord(item.fact)))}</span>
            ${chipHtml(authorWord(item.author), 'neutral')}
            <button type="button" class="btn btn-ghost btn-sm" data-place-act="remove-wake" data-fact="${escapeHtml(item.fact)}" aria-label="Stop waking for ${escapeHtml(wakeFactWord(item.fact))}">Remove</button>
        </li>`).join('');
    const offers = place.wakeSuggestions.filter((fact) => !listed.has(fact));
    const add = offers.length ? `<div class="place-wake-offers" role="group" aria-label="Wake it also for">
            <span class="place-sheet-note">Also wake it for:</span>
            ${offers.map((fact) => `<button type="button" class="btn btn-default btn-sm" data-place-act="add-wake" data-fact="${escapeHtml(fact)}">${escapeHtml(capitalized(wakeFactWord(fact)))}</button>`).join('')}
        </div>` : '';
    const note = WAKE_NOTES[place.state] || '';
    return `${items ? `<ul class="place-wake-list">${items}</ul>` : '<p class="place-sheet-note">Nothing here wakes it at once.</p>'}
        ${add}${note ? `<p class="place-sheet-note">${escapeHtml(note)}</p>` : ''}`;
}

function accountsHtml(place, ui) {
    const rows = place.ownerAccounts.map((account) => `<div class="harness-account-row place-account">
            <div class="harness-account-main"><strong>${escapeHtml(account.label || account.actorId)}</strong></div>
            <div class="harness-account-meta">${escapeHtml(['you', account.confirmedAt ? `confirmed on ${shortDate(account.confirmedAt)}` : 'confirmed',
                account.label && account.label !== account.actorId ? account.actorId : ''].filter(Boolean).join(' · '))}</div>
            <div class="harness-account-actions"><button type="button" class="btn btn-ghost btn-sm" data-place-act="revoke" data-actor="${escapeHtml(account.actorId)}" aria-label="Not my account: remove ${escapeHtml(account.label || account.actorId)}">Remove</button></div>
        </div>`).join('');
    const mine = new Set(place.ownerAccounts.map((account) => account.actorId));
    const candidates = place.accountCandidates.filter((account) => !mine.has(account.actorId)).map((account) => `<div class="harness-account-row place-account">
            <div class="harness-account-main"><strong>${escapeHtml(account.label || account.actorId)}</strong></div>
            <div class="harness-account-meta">${escapeHtml(account.label ? account.actorId : 'seen here')}</div>
            <div class="harness-account-actions"><button type="button" class="btn btn-default btn-sm" data-place-act="confirm" data-actor="${escapeHtml(account.actorId)}" data-actor-label="${escapeHtml(account.label)}">This is me</button></div>
        </div>`).join('');
    const form = ui.adding ? `<div class="place-account-form">
            <div class="ui-field"><label for="place-account-id">Your account here</label>
                <input class="ui-control" id="place-account-id" data-place-draft="actorId" value="${escapeHtml(ui.drafts.actorId || '')}" spellcheck="false" autocomplete="off" aria-describedby="place-account-help">
                <div class="ui-field-help" id="place-account-help">${escapeHtml(`${capitalized(accountIdText(place.provider))}.`)}</div></div>
            <label class="ui-field">Name (optional)<input class="ui-control" data-place-draft="actorLabel" value="${escapeHtml(ui.drafts.actorLabel || '')}" spellcheck="false" autocomplete="off"></label>
            <div class="place-sheet-actions"><button type="button" class="btn btn-default btn-sm" data-place-act="confirm-typed">This is me</button><button type="button" class="btn btn-ghost btn-sm" data-place-act="cancel-add">Cancel</button></div>
        </div>` : '<div class="place-sheet-actions"><button type="button" class="btn btn-ghost btn-sm" data-place-act="add-mine">Add mine</button></div>';
    return `${rows || candidates ? `<div class="place-accounts">${rows}${candidates}</div>` : ''}${form}
        <p class="place-sheet-note">Anyone else here, whatever name they use, is talked to like anyone; their requests are theirs, not yours.</p>`;
}

function notesHtml(place, ui) {
    const items = place.notes.map((note) => `<li class="place-note${OWNER_VOICE.has(note.author) ? ' is-owner' : ''}">
            <p>${escapeHtml(note.text)}</p>
            <div class="place-note-foot">${chipHtml(authorWord(note.author), 'neutral')}${note.id
                ? `<button type="button" class="btn btn-ghost btn-sm" data-place-act="remove-note" data-note="${escapeHtml(note.id)}">Remove</button>` : ''}</div>
        </li>`).join('');
    return `<p class="place-sheet-note">It reads them at every wake here; the Safety Supervisor treats yours as your instructions.</p>
        ${items ? `<ul class="place-note-list">${items}</ul>` : ''}
        <label class="ui-field">New note<textarea class="ui-control consciousness-owner-words" rows="2" data-place-draft="note" spellcheck="false">${escapeHtml(ui.drafts.note || '')}</textarea></label>
        <div class="place-sheet-actions"><button type="button" class="btn btn-default btn-sm" data-place-act="add-note">Add note</button></div>`;
}

function folderHtml(place, ui) {
    const value = ui.drafts.folder ?? place.workingFolder;
    return `<div class="place-field-row">
            <input class="ui-control ui-control-code" data-place-draft="folder" value="${escapeHtml(value)}" spellcheck="false" autocomplete="off" aria-label="Working folder" aria-describedby="place-folder-help" placeholder="No folder">
            <button type="button" class="btn btn-default btn-sm" data-place-act="save-folder">Save</button>
        </div>
        <div class="ui-field-help" id="place-folder-help">An address, not a fence: work for this place starts here.</div>`;
}

function moreHtml(place, ui) {
    const value = ui.drafts.model ?? place.model;
    return `<details class="consciousness-more place-more"${ui.moreOpen ? ' open' : ''}>
            <summary><span class="consciousness-marker" aria-hidden="true">▸</span>More</summary>
            <div class="consciousness-more-body">
                <label class="ui-field">Model
                    <div class="place-field-row">
                        <input class="ui-control" data-place-draft="model" value="${escapeHtml(value)}" spellcheck="false" autocomplete="off" placeholder="Empty uses the consciousness model">
                        <button type="button" class="btn btn-default btn-sm" data-place-act="save-model">Save</button>
                    </div>
                </label>
                <div class="ui-field-help">The model it thinks with when it wakes here; empty uses the Consciousness model in Models.</div>
            </div>
        </details>`;
}

function footHtml(place, ui) {
    if (ui.confirmDisconnect) {
        return `<div class="place-disconnect-confirm" role="group" aria-label="Disconnect ${escapeHtml(place.label)}">
            <p>Disconnect ${escapeHtml(place.label)}? It stops hearing and acting here; the room's history stays.</p>
            <div class="place-sheet-actions"><button type="button" class="btn btn-danger btn-sm" data-place-act="disconnect-yes">Disconnect</button><button type="button" class="btn btn-default btn-sm" data-place-act="disconnect-no">Keep it</button></div>
        </div>`;
    }
    return `<div class="place-sheet-actions place-sheet-foot">
            <button type="button" class="btn btn-ghost btn-sm" data-place-act="open-connector">Open connector page</button>
            <button type="button" class="btn btn-ghost btn-sm place-danger" data-place-act="disconnect">Disconnect</button>
        </div>`;
}

/** The sheet's body for one normalized place and the sheet's own UI state (drafts, busy, statuses). */
export function placeSheetBodyHtml(place, ui = {}) {
    const view = { drafts: {}, status: {}, ...ui };
    return [
        section('state', '', stateHtml(place, view), view),
        section('wakes', 'Wakes it at once', wakesHtml(place), view),
        section('conversations', 'Conversations', '<button type="button" class="btn btn-default btn-sm" data-place-act="conversations">Open captured conversations</button><p class="place-sheet-note">Read the shared room history without starting work or sending a message.</p>', view),
        section('accounts', 'Your accounts here', accountsHtml(place, view), view),
        section('notes', 'Notes', notesHtml(place, view), view),
        section('folder', 'Working folder', folderHtml(place, view), view),
        section('more', '', moreHtml(place, view), view),
        section('foot', '', footHtml(place, view), view),
    ].join('');
}

export const placeSheetMetaText = (place) => [place.connectorLabel, placeScopeText(place)].filter(Boolean).join(' · ');

function sheetShell(titleHtml, metaText, modifier = '') {
    const dialog = document.createElement('dialog');
    dialog.className = `document-reader place-sheet${modifier ? ` ${modifier}` : ''}`;
    dialog.setAttribute('aria-labelledby', 'place-sheet-title');
    dialog.innerHTML = `<div class="document-reader-panel">
        <div class="document-reader-head">
            <div class="document-reader-identity">
                <h2 class="document-reader-title place-sheet-title" id="place-sheet-title">${titleHtml}</h2>
                <div class="document-reader-meta" data-place-meta>${escapeHtml(metaText)}</div>
            </div>
            <div class="document-reader-actions"></div>
            <button type="button" class="btn btn-default document-reader-close" data-place-close>Close</button>
        </div>
        <div class="document-reader-body place-sheet-body is-quiet-focus" tabindex="-1" data-place-body></div>
    </div>`;
    return dialog;
}

function showSheet(dialog, { returnFocus, onClose }) {
    const body = dialog.querySelector('[data-place-body]');
    let closed = false;
    let disposeFocus = null;
    const close = () => {
        if (closed) return;
        closed = true;
        if (dialog.open && typeof dialog.close === 'function') dialog.close();
        disposeFocus?.();
        dialog.remove();
        onClose?.();
    };
    dialog.addEventListener('cancel', (event) => { event.preventDefault(); close(); });
    dialog.addEventListener('click', (event) => {
        if (event.target === dialog || event.target.closest('[data-place-close]')) close();
    });
    dialog.addEventListener('keydown', () => body.classList.remove('is-quiet-focus'), { once: true });
    document.body.appendChild(dialog);
    if (typeof dialog.showModal === 'function') dialog.showModal(); else dialog.setAttribute('open', '');
    disposeFocus = bindDialogFocus(dialog, { initialFocus: body, returnFocus, onEscape: close });
    return { close, body, get closed() { return closed; } };
}

/** The one write a press on the sheet asks for, as the contract's change; null when it needs no write. */
export function placeChangeFor(action, place, { value = '', fact = '', note = '', actor = '', actorLabel = '', drafts = {} } = {}) {
    const facts = place.wakes.map((item) => item.fact);
    switch (action) {
    case 'state': return value && value !== place.state ? { state: value } : null;
    case 'add-wake': return facts.includes(fact) ? null : { wakes: [...facts, fact] };
    case 'remove-wake': return { wakes: facts.filter((entry) => entry !== fact) };
    case 'add-note': return String(drafts.note || '').trim() ? { add_note: String(drafts.note).trim() } : null;
    case 'remove-note': return note ? { remove_note: note } : null;
    case 'confirm': return actor ? { confirm_owner_account: { platform_actor_id: actor, label: actorLabel } } : null;
    case 'confirm-typed': {
        const id = String(drafts.actorId || '').trim();
        return id ? { confirm_owner_account: { platform_actor_id: id, label: String(drafts.actorLabel || '').trim() } } : null;
    }
    case 'revoke': return actor ? { revoke_owner_account: actor } : null;
    case 'save-folder': return { working_folder: String(drafts.folder ?? place.workingFolder).trim() };
    case 'save-model': return { model: String(drafts.model ?? place.model).trim() };
    default: return null;
    }
}

const SECTION_OF = {
    state: 'state', 'add-wake': 'wakes', 'remove-wake': 'wakes', 'add-note': 'notes', 'remove-note': 'notes',
    confirm: 'accounts', 'confirm-typed': 'accounts', revoke: 'accounts', 'save-folder': 'folder', 'save-model': 'more',
};
const DONE_WORDS = {
    state: (place) => `Now ${placeStateOf(place.state)?.label || place.state}.`, 'add-wake': () => 'Added.', 'remove-wake': () => 'Removed.',
    'add-note': () => 'Note added.', 'remove-note': () => 'Note removed.', confirm: () => 'Confirmed as you.',
    'confirm-typed': () => 'Confirmed as you.', revoke: () => 'Removed from your accounts.',
    'save-folder': () => 'Folder saved.', 'save-model': () => 'Model saved.',
};

/**
 * Open the sheet for a normalized place. `onChanged(place)` hears every confirmed write; Disconnect closes
 * the sheet. `update`, `disconnect` are injectable for tests.
 */
export function openPlaceSheet(place, { returnFocus = document.activeElement, onClose = null, update = updatePlace,
    disconnect = disconnectPlace } = {}) {
    let current = place;
    let history = null;
    const ui = { drafts: {}, status: {}, busy: false, adding: false, confirmDisconnect: false, moreOpen: false };
    const dialog = sheetShell(`<span class="place-glyph" aria-hidden="true">${skillIconHtml(place.icon, 22)}</span><span>${escapeHtml(place.label)}</span>`,
        placeSheetMetaText(place));
    const sheet = showSheet(dialog, { returnFocus, onClose: () => { history?.close(); window.removeEventListener(PLACES_CHANGED_EVENT, onExternal); onClose?.(); } });
    const { body } = sheet;

    function paint() {
        const scroll = body.scrollTop;
        const focusKey = body.contains(document.activeElement) ? focusKeyOf(document.activeElement) : '';
        body.innerHTML = placeSheetBodyHtml(current, ui);
        body.querySelectorAll(`[data-place-state] [data-effort-value]`).forEach((button) => {
            const chosen = button.dataset.effortValue === current.state;
            button.classList.toggle('active', chosen);
            button.setAttribute('aria-pressed', String(chosen));
            button.disabled = ui.busy;
        });
        body.querySelector('[data-place-state]')?.setAttribute('aria-labelledby', 'place-state-label');
        body.scrollTop = scroll;
        if (focusKey) body.querySelector(focusKey)?.focus();
        const meta = dialog.querySelector('[data-place-meta]');
        if (meta) meta.textContent = placeSheetMetaText(current);
    }

    async function write(action, options = {}) {
        if (ui.busy) return;
        const change = placeChangeFor(action, current, { ...options, drafts: ui.drafts });
        const key = SECTION_OF[action];
        if (!change) {
            ui.status[key] = { text: action === 'add-note' ? 'Write the note first.' : 'Nothing to save.', tone: 'muted' };
            paint();
            return;
        }
        ui.busy = true;
        ui.status = { [key]: { text: 'Saving…', tone: 'muted' } };
        paint();
        try {
            current = await update(current.id, change);
            if (action === 'add-note') ui.drafts.note = '';
            if (action === 'confirm-typed') { ui.drafts.actorId = ''; ui.drafts.actorLabel = ''; ui.adding = false; }
            if (action === 'save-folder') delete ui.drafts.folder;
            if (action === 'save-model') delete ui.drafts.model;
            ui.status = { [key]: { text: DONE_WORDS[action](current), tone: 'ok' } };
        } catch (error) {
            ui.status = { [key]: { text: error?.message || String(error), tone: 'error' } };
        } finally {
            ui.busy = false;
            paint();
        }
    }

    function onExternal(event) {
        const changed = event?.detail?.place;
        if (!ui.busy && changed && changed.id === current.id && changed !== current) { current = changed; paint(); }
    }
    window.addEventListener(PLACES_CHANGED_EVENT, onExternal);

    body.addEventListener('input', (event) => {
        const field = event.target.closest?.('[data-place-draft]');
        if (field) ui.drafts[field.dataset.placeDraft] = field.value;
    });
    body.addEventListener('toggle', (event) => {
        if (event.target.classList?.contains('place-more')) ui.moreOpen = event.target.open;
    }, true);
    body.addEventListener('click', async (event) => {
        const segment = event.target.closest('[data-place-state] [data-effort-value]');
        if (segment) {
            if (segment.dataset.effortValue !== current.state) await write('state', { value: segment.dataset.effortValue });
            return;
        }
        const button = event.target.closest('[data-place-act]');
        if (!button || button.disabled) return;
        const act = button.dataset.placeAct;
        if (act === 'conversations') { history?.close(); history = openPlaceHistory(current, { returnFocus: button }); return; }
        if (act === 'add-mine') { ui.adding = true; paint(); body.querySelector('[data-place-draft="actorId"]')?.focus(); return; }
        if (act === 'cancel-add') { ui.adding = false; paint(); return; }
        if (act === 'open-connector') { sheet.close(); openSkillsFor({ skill: current.connector }); return; }
        if (act === 'disconnect') { ui.confirmDisconnect = true; paint(); body.querySelector('[data-place-act="disconnect-no"]')?.focus(); return; }
        if (act === 'disconnect-no') { ui.confirmDisconnect = false; paint(); body.querySelector('[data-place-act="disconnect"]')?.focus(); return; }
        if (act === 'disconnect-yes') {
            ui.busy = true;
            try {
                await disconnect(current.id);
                sheet.close();
            } catch (error) {
                ui.busy = false;
                ui.status = { foot: { text: error?.message || String(error), tone: 'error' } };
                paint();
            }
            return;
        }
        await write(act, { fact: button.dataset.fact || '', note: button.dataset.note || '',
            actor: button.dataset.actor || '', actorLabel: button.dataset.actorLabel || '' });
    });
    paint();
    return { close: sheet.close, get closed() { return sheet.closed; }, get place() { return current; } };
}

// A repaint keeps the owner's focus on the same control (the press that asked for it, or the field).
function focusKeyOf(node) {
    const act = node.closest?.('[data-place-act]');
    if (act) {
        const extra = ['fact', 'note', 'actor'].map((name) => act.dataset[name] ? `[data-${name}="${CSS.escape(act.dataset[name])}"]` : '').join('');
        return `[data-place-act="${act.dataset.placeAct}"]${extra}`;
    }
    const draft = node.closest?.('[data-place-draft]');
    if (draft) return `[data-place-draft="${draft.dataset.placeDraft}"]`;
    const segment = node.closest?.('[data-effort-value]');
    return segment ? `[data-place-state] [data-effort-value="${segment.dataset.effortValue}"]` : '';
}

/** The Connect list: installed connectors whose place is not connected yet; none → the way to Browse. */
export function connectListHtml({ connectors = null, error = '', status = {} } = {}) {
    if (error) return `<div class="place-sheet-section"><p class="place-sheet-status ui-status" data-tone="error" role="alert">${escapeHtml(error)}</p>
        <div class="place-sheet-actions"><button type="button" class="btn btn-default btn-sm" data-connect-act="retry">Try again</button></div></div>`;
    if (!connectors) return '<p class="place-sheet-section place-sheet-note" role="status">Reading the installed connectors…</p>';
    const open = connectors.filter((connector) => !connector.connected);
    if (!open.length) {
        return `<section class="place-sheet-section">
            <p>${connectors.length ? 'Every installed connector is connected.' : 'No installed connector can connect a place yet.'} A connector is a skill that connects a service such as Slack, Telegram or email.</p>
            <div class="place-sheet-actions"><button type="button" class="btn btn-default btn-sm" data-connect-act="browse">Find connectors in Skills</button></div>
        </section>`;
    }
    return `<section class="place-sheet-section">
        <p class="place-sheet-note">Connecting a place lets it hear there and, while the place is Active, speak and act there. You choose its state after connecting.</p>
        <div class="place-accounts">${open.map((connector) => `<div class="harness-account-row place-connector-row">
            <div class="harness-account-main"><span class="place-glyph" aria-hidden="true">${skillIconHtml(connector.icon, 18)}</span><strong>${escapeHtml(connector.label)}</strong></div>
            <div class="harness-account-meta">${escapeHtml([connector.destination.replace(/^vendor:/, 'Connects '),
                connector.wakes.length ? `suggests waking for ${listWords(connector.wakes.map(wakeFactWord))}` : ''].filter(Boolean).join(' · '))}</div>
            <div class="harness-account-actions"><button type="button" class="btn btn-default btn-sm" data-connect-act="connect" data-connector="${escapeHtml(connector.connector)}">Connect</button></div>
            ${statusLine(status[connector.connector])}
        </div>`).join('')}</div>
    </section>`;
}

/** Open the Connect flow; a connected place opens its own sheet at once. */
export function openConnectPlace({ returnFocus = document.activeElement, read = readPlaces, connect = connectPlace, connector = '' } = {}) {
    const dialog = sheetShell('Connect a place', 'From the connectors installed here', 'place-connect');
    const sheet = showSheet(dialog, { returnFocus });
    const view = { connectors: null, error: '', status: {} };
    const paint = () => { sheet.body.innerHTML = connectListHtml(view); };
    const load = async () => {
        view.error = '';
        view.connectors = null;
        paint();
        try { view.connectors = (await read()).connectors.filter((row) => !connector || row.connector === connector); } catch (error) { view.error = `Couldn’t read the connectors: ${error?.message || error}`; }
        if (!sheet.closed) paint();
    };
    sheet.body.addEventListener('click', async (event) => {
        const button = event.target.closest('[data-connect-act]');
        if (!button || button.disabled) return;
        const act = button.dataset.connectAct;
        if (act === 'retry') { await load(); return; }
        if (act === 'browse') { sheet.close(); openSkillsFor({ category: 'connectors' }); return; }
        if (act !== 'connect') return;
        const name = button.dataset.connector;
        button.disabled = true;
        view.status = { [name]: { text: 'Connecting…', tone: 'muted' } };
        paint();
        try {
            const place = await connect({ connector: name });
            sheet.close();
            openPlaceSheet(place, { returnFocus });
        } catch (error) {
            view.status = { [name]: { text: error?.message || String(error), tone: 'error' } };
            if (!sheet.closed) paint();
        }
    });
    void load();
    return sheet;
}

/** Navigation only: Skills never connects, enables, or selects an account by opening this door. */
export function openConnectorPlaces(connector, { returnFocus = document.activeElement, read = readPlaces } = {}) {
    const dialog = sheetShell('Places', connector);
    const controller = new AbortController();
    const sheet = showSheet(dialog, { returnFocus, onClose: () => controller.abort() });
    let places = [];
    const load = async () => {
        sheet.body.textContent = 'Reading connected places…';
        try {
            const data = await read({ signal: controller.signal });
            if (sheet.closed) return;
            places = data.places.filter((place) => place.connector === connector);
            if (places.length === 1) { sheet.close(); openPlaceSheet(places[0], { returnFocus }); return; }
            sheet.body.innerHTML = places.length
                ? `<section class="place-sheet-section"><p>Select the account or conversation.</p><div class="place-sheet-actions">${places.map((place, index) => `<button type="button" class="btn btn-default btn-sm" data-connector-place="${index}">${escapeHtml([place.label, place.accountId, place.conversationId].filter(Boolean).join(' · '))}</button>`).join('')}</div></section>`
                : '<section class="place-sheet-section"><p>No connected place for this connector.</p><button type="button" class="btn btn-default btn-sm" data-connector-connect>Connect a place</button></section>';
        } catch (error) {
            if (!sheet.closed) sheet.body.innerHTML = `<section class="place-sheet-section"><p role="alert">${escapeHtml(error?.message || String(error))}</p><button type="button" class="btn btn-default btn-sm" data-connector-retry>Try again</button></section>`;
        }
    };
    sheet.body.addEventListener('click', (event) => {
        if (event.target.closest('[data-connector-retry]')) { void load(); return; }
        if (event.target.closest('[data-connector-connect]')) { sheet.close(); openConnectPlace({ returnFocus, connector }); return; }
        const button = event.target.closest('[data-connector-place]');
        const place = button && places[Number(button.dataset.connectorPlace)];
        if (place) { sheet.close(); openPlaceSheet(place, { returnFocus }); }
    });
    void load();
    return sheet;
}
