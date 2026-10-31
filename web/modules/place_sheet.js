// One PlaceSummary in a document-reader sheet. Owner writes repaint from the server and retain drafts.
// Connect is separate in place_connect.js; the body stays here for the place-room integration.
import { bindDialogFocus } from './ui_interactions.js';
import { escapeHtmlAttr as escapeHtml } from './ui_primitives.js';
import { renderSegmentedField } from './page_header.js';
import { chipHtml, skillIconHtml } from './skill_presentation.js';
import { tr, fmt } from './i18n.js';
import { fetchJson } from './api_client.js';
import { bindModelChoosers, modelChooserHtml, updateModelChooser } from './model_chooser.js';
import { disconnectPlace, normalizePlace, normalizePlaceStatus, PLACES_CHANGED_EVENT, readPlaceConnector, readPlaceRooms, updatePlace } from './consciousness_api.js';
import { mountPlaceHistory, placeRoomsHtml } from './place_history.js';
import {
    PLACE_STATES, SILENT_CARD, WAKE_FACT_WORDS, accountIdText, authorWord, capitalized, listWords, placeStateOf, placeStateWord, signalLabels, wakeFactWord,
} from './consciousness_words.js';

/** Ask the Skills page for one installed skill's card, or Browse filtered to connectors. */
export function openSkillsFor(detail) {
    globalThis.window?.dispatchEvent?.(new CustomEvent('ouro:open-skills', { detail }));
}

const OWNER_VOICE = new Set(['owner', 'migration']);
const shortDate = (value) => {
    const time = Date.parse(String(value || ''));
    return Number.isFinite(time) ? new Date(time).toLocaleDateString('en-GB', { day: 'numeric', month: 'short' }) : '';
};
export const statusLine = (status) => (status?.text
    ? `<p class="place-sheet-status ui-status" data-tone="${escapeHtml(status.tone || 'muted')}" role="status"${status.authored ? ' data-i18n-authored' : ''}>${escapeHtml(status.text)}</p>` : '');
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

function wakesHtml(place, ui) {
    const labels = { ...signalLabels(place.facts), ...place.labels, ...ui.labels };
    const vocabulary = place.facts ?? Object.keys(WAKE_FACT_WORDS).filter((fact) => fact !== 'legacy_addressed'
        || place.wakes.some((item) => item.fact === fact));
    const items = vocabulary.map((item) => {
        const fact = typeof item === 'string' ? item : item?.id;
        if (!fact) return '';
        const chosen = place.wakes.find((wake) => wake.fact === fact);
        const authored = Boolean(labels[fact]);
        const label = authored ? labels[fact] : capitalized(wakeFactWord(fact));
        return `<li class="place-wake-item"><label class="place-wake-toggle ui-hit-area"><input class="ui-checkbox" type="checkbox" data-place-act="${chosen ? 'remove-wake' : 'add-wake'}" data-fact="${escapeHtml(fact)}"${chosen ? ' checked' : ''}>
            <span${authored ? ' data-i18n-authored' : ''}>${escapeHtml(label)}</span></label>${chosen ? chipHtml(tr(`places.author.${chosen.author}`, authorWord(chosen.author)), 'neutral') : ''}</li>`;
    }).join('');
    const service = `<span data-i18n-authored>${escapeHtml(ui.serviceLabel || place.serviceLabel || place.provider)}</span>`;
    return `<p class="place-sheet-note">${fmt('A checked kind calls it here at once, when {service} reports it.', { service })}</p>
        <ul class="place-wake-list">${items}</ul>
        ${vocabulary.includes('from_bot') ? `<p class="place-sheet-note">${tr('places.wake.bots', 'Bot messages need this check; once checked, they can wake it without another checked kind.')}</p>` : ''}
        ${place.state === 'silent' ? `<p class="place-sheet-note">${tr('places.wake.silent', 'Silent: these still wake it to read and think; what it would say or change here is held.')}</p>` : ''}
        ${place.state === 'paused' ? `<p class="place-sheet-note">${tr('places.wake.paused', 'While this place is Paused, nothing here wakes it.')}</p>` : ''}
        <p class="place-sheet-note">${tr('places.wake.honesty', 'Everything else here it sees at its next own wake and decides what to read.')}</p>`;
}

function identityHtml(place, ui) {
    // The same account-scoped match as the server's owner_account_matches: a confirmation for another account is not this one.
    const withOwner = (entry) => ({ ...entry, owner: entry.kind === 'user' && place.ownerAccounts.some((account) =>
        account.actorId === entry.id && (!account.accountId || account.accountId === place.accountId)) });
    const sender = ui.identity?.sends_as && withOwner(ui.identity.sends_as);
    const authored = (label) => `<span data-i18n-authored>${escapeHtml(label)}</span>`;
    const readers = listWords((ui.identity?.reads_as || []).map(withOwner).map((entry) => authored(entry.label)));
    const writes = sender ? fmt(sender.kind === 'bot' ? 'It writes here as the Ouroboros bot ({label})'
        : sender.owner ? 'It writes here as you ({label})' : 'It writes here as {label}', { label: authored(sender.label) }) : '';
    const reads = readers ? fmt(writes ? 'reads as {labels}' : 'It reads here as {labels}', { labels: readers }) : '';
    if (!writes && !reads) return '';
    const canConfirm = sender?.kind === 'user' && !sender.owner;
    return `<div class="harness-account-row place-account">
        <div class="harness-account-main"><span>${[writes, reads].filter(Boolean).join(' · ')}</span></div>
        ${canConfirm ? `<div class="harness-account-actions"><button type="button" class="btn btn-default btn-sm" data-place-act="confirm" data-actor="${escapeHtml(sender.id)}" data-actor-label="${escapeHtml(sender.label)}">${escapeHtml(tr('places.account.confirm_identity', 'This is my account'))}</button></div>` : ''}
    </div>`;
}

function accountsHtml(place, ui) {
    const identity = ui.removed ? '' : identityHtml(place, ui);
    const rows = place.ownerAccounts.map((account) => `<div class="harness-account-row place-account">
            <div class="harness-account-main"><strong data-i18n-authored>${escapeHtml(account.label || account.actorId)}</strong></div>
            <div class="harness-account-meta">${escapeHtml([tr('places.account.you', 'you'), account.confirmedAt ? fmt('confirmed on {date}', { date: shortDate(account.confirmedAt) }) : tr('places.account.confirmed', 'confirmed')].join(' · '))}${account.label && account.label !== account.actorId ? ` · <span data-i18n-authored>${escapeHtml(account.actorId)}</span>` : ''}</div>
            <div class="harness-account-actions"><button type="button" class="btn btn-ghost btn-sm" data-place-act="revoke" data-actor="${escapeHtml(account.actorId)}" aria-label="Not my account: remove ${escapeHtml(account.label || account.actorId)}">Remove</button></div>
        </div>`).join('');
    const mine = new Set(place.ownerAccounts.map((account) => account.actorId));
    const candidates = place.accountCandidates.filter((account) => !mine.has(account.actorId)).map((account) => `<div class="harness-account-row place-account">
            <div class="harness-account-main"><strong data-i18n-authored>${escapeHtml(account.label || account.actorId)}</strong></div>
            <div class="harness-account-meta"${account.label ? ' data-i18n-authored' : ''}>${escapeHtml(account.label ? account.actorId : tr('places.account.seen', 'seen here'))}</div>
            <div class="harness-account-actions"><button type="button" class="btn btn-default btn-sm" data-place-act="confirm" data-actor="${escapeHtml(account.actorId)}" data-actor-label="${escapeHtml(account.label)}">This is me</button></div>
        </div>`).join('');
    const form = ui.adding ? `<div class="place-account-form">
            <div class="ui-field"><label for="place-account-id">Your account here</label>
                <input class="ui-control" id="place-account-id" data-place-draft="actorId" value="${escapeHtml(ui.drafts.actorId || '')}" spellcheck="false" autocomplete="off" aria-describedby="place-account-help">
                <div class="ui-field-help" id="place-account-help">${escapeHtml(`${capitalized(accountIdText(place.provider))}.`)}</div></div>
            <label class="ui-field">${tr('places.account.name', 'Name (optional)')}<input class="ui-control" data-place-draft="actorLabel" data-i18n-authored value="${escapeHtml(ui.drafts.actorLabel || '')}" spellcheck="false" autocomplete="off"></label>
            <div class="place-sheet-actions"><button type="button" class="btn btn-default btn-sm" data-place-act="confirm-typed">This is me</button><button type="button" class="btn btn-ghost btn-sm" data-place-act="cancel-add">Cancel</button></div>
        </div>` : '<div class="place-sheet-actions"><button type="button" class="btn btn-ghost btn-sm" data-place-act="add-mine">Add mine</button></div>';
    return `${identity || rows || candidates ? `<div class="place-accounts">${identity}${rows}${candidates}</div>` : ''}${form}
        <p class="place-sheet-note">Anyone else here, whatever name they use, is talked to like anyone; their requests are theirs, not yours.</p>`;
}

function notesHtml(place, ui) {
    const safety = {
        full: 'Ouroboros reads your words every time; the Safety Supervisor reads them before actions.',
        light: 'Ouroboros reads your words every time; the Safety Supervisor reads them before some actions.',
        off: 'Ouroboros reads your words every time; the Safety Supervisor is off.',
    }[place.safetyMode];
    const honesty = safety ? `${tr(`places.notes.safety.${place.safetyMode}`, safety)} ${tr('places.notes.locks', 'Silent and Paused are locks; words are not.')}` : '';
    const placeholder = tr('places.notes.questions', 'Who am I here? What may I do? What calls me at once? What matters here? What must I never do? What do you want to see afterwards?');
    const items = place.notes.map((note) => `<li class="place-note${OWNER_VOICE.has(note.author) ? ' is-owner' : ''}">
            <p data-i18n-authored>${escapeHtml(note.text)}</p>
            <div class="place-note-foot">${chipHtml(authorWord(note.author), 'neutral')}${note.id
                ? `<button type="button" class="btn btn-ghost btn-sm" data-place-act="remove-note" data-note="${escapeHtml(note.id)}">Remove</button>` : ''}</div>
        </li>`).join('');
    return `${honesty ? '' : '<p class="place-sheet-note">It reads them at every wake here; the Safety Supervisor treats yours as your instructions.</p>'}
        ${items ? `<ul class="place-note-list">${items}</ul>` : ''}
        <label class="ui-field">${tr('places.note.new', 'New note')}<textarea class="ui-control consciousness-owner-words" rows="${place.notes.length ? 2 : 6}" data-place-draft="note" data-i18n-authored spellcheck="false" placeholder="${escapeHtml(placeholder)}">${escapeHtml(ui.drafts.note || '')}</textarea></label>
        <div class="place-sheet-actions"><button type="button" class="btn btn-default btn-sm" data-place-act="add-note">Add note</button></div>
        ${honesty ? `<p class="place-sheet-note">${escapeHtml(honesty)}</p>` : ''}`;
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
                        ${modelChooserHtml(`data-place-draft="model" aria-label="${escapeHtml(tr('places.model', 'Model'))}"`, value, 'place-model-options', ui.models || [], { placeholder: tr('places.model.empty', 'Empty uses the consciousness model') })}
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
            <p>${fmt("Disconnect {place}? It stops hearing and acting here; the room's history stays.", { place: `<span data-i18n-authored>${escapeHtml(place.label)}</span>` })}</p>
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
    if (view.removed) place = { ...place, notes: [], ownerAccounts: [], accountCandidates: [] };
    return [
        view.conflict || view.removed || view.readiness || place.readiness ? section('readiness', '',
            (view.conflict ? statusLine({ text: tr('places.conflict', 'Changed elsewhere; your text is kept.'), tone: 'warn' }) : '')
            + (view.removed ? statusLine({ text: tr('places.disconnected', 'This place is disconnected.'), tone: 'muted' })
                : statusLine(view.readiness || (place.readiness && { ...place.readiness, text: tr(`places.readiness.${place.readiness.text}`, place.readiness.text) }))), view) : '',
        view.removed ? '' : section('state', '', stateHtml(place, view), view),
        view.removed ? '' : section('conversations', tr('places.conversations', 'Conversations'), placeRoomsHtml(view.roomIndex || { rooms: null }), view),
        view.removed ? '' : section('wakes', tr('places.calls', 'What calls it here'), wakesHtml(place, view), view),
        section('accounts', 'Your accounts here', accountsHtml(place, view), view),
        section('notes', 'Notes', notesHtml(place, view), view),
        section('folder', 'Working folder', folderHtml(place, view), view),
        section('more', '', moreHtml(place, view), view),
        view.removed ? '' : section('foot', '', footHtml(place, view), view),
    ].join('');
}

export const placeSheetMetaText = (place) => [place.label, placeStateWord(place.state)].filter(Boolean).join(' · ');

export function sheetShell(titleHtml, metaText, modifier = '') {
    const dialog = document.createElement('dialog');
    dialog.className = `document-reader place-sheet${modifier ? ` ${modifier}` : ''}`;
    dialog.setAttribute('aria-labelledby', 'place-sheet-title');
    dialog.setAttribute('aria-modal', 'true');  // bindDialogFocus answers Escape (the shared Back) only for an explicit modal
    dialog.innerHTML = `<div class="document-reader-panel">
        <div class="document-reader-head">
            <div class="document-reader-identity">
                <h2 class="document-reader-title place-sheet-title" id="place-sheet-title">${titleHtml}</h2>
                <div class="document-reader-meta" data-place-meta>${escapeHtml(metaText)}</div>
            </div>
            <div class="document-reader-actions"></div>
            <button type="button" class="btn btn-default document-reader-close" data-place-close>${tr('places.close', 'Close')}</button>
        </div>
        <div class="document-reader-body place-sheet-body is-quiet-focus" tabindex="-1" data-place-body></div>
    </div>`;
    return dialog;
}

export function showSheet(dialog, { returnFocus, onClose, onBack }) {
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
    const back = () => { if (!onBack?.()) close(); };
    dialog.addEventListener('cancel', (event) => { event.preventDefault(); back(); });
    dialog.addEventListener('close', close);  // a native close (the shared Back's fallback) disposes the sheet too
    dialog.addEventListener('click', (event) => {
        if (event.target === dialog || event.target.closest('[data-place-close]')) close();
    });
    dialog.addEventListener('keydown', () => body.classList.remove('is-quiet-focus'), { once: true });
    document.body.appendChild(dialog);
    if (typeof dialog.showModal === 'function') dialog.showModal(); else dialog.setAttribute('open', '');
    disposeFocus = bindDialogFocus(dialog, { initialFocus: body, returnFocus, onEscape: back });
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

/** Open one projected place; requests keep its revision and drafts survive a refused or concurrent write. */
export function openPlaceSheet(place, { returnFocus = document.activeElement, onClose = null, update = updatePlace,
    disconnect = disconnectPlace, readStatus = readPlaceConnector, readModels = (init) => fetchJson('/api/model-catalog', init),
    readRooms = readPlaceRooms, openRoomChatId = null } = {}) {
    let current = place;
    const ui = { drafts: {}, status: {}, busy: false, adding: false, confirmDisconnect: false, moreOpen: false,
        serviceLabel: place.serviceLabel || place.provider, roomIndex: { rooms: null } };
    let disposeModels = () => {}, modelsRead = false;
    let history = null, selected = null, editorScroll = 0, roomController = null, roomGeneration = 0;
    const abort = new AbortController();
    const dialog = sheetShell(`<span class="place-glyph" aria-hidden="true">${skillIconHtml(place.icon, 22)}</span><span data-place-label data-i18n-authored>${escapeHtml(place.label)}</span>`,
        placeSheetMetaText(place));
    const sheet = showSheet(dialog, { returnFocus, onBack: () => { if (!history) return false; backToPlace(); return true; },
        onClose: () => { abort.abort(); disposeModels(); roomGeneration += 1; roomController?.abort(); history?.close(); window.removeEventListener(PLACES_CHANGED_EVENT, onExternal); onClose?.(); } });
    const { body } = sheet;
    // The editor holds the sections; a captured conversation opens in the reader beside it (one sheet, Back returns).
    const editor = document.createElement('div'); editor.className = 'place-sheet-editor';
    const reader = document.createElement('div'); reader.hidden = true;
    body.append(editor, reader);
    const readerPlace = () => ({ ...current, labels: { ...current.labels, ...ui.labels } });

    function backToPlace() {
        const id = selected?.room_chat_id;
        history?.close(); history = null; selected = null;
        reader.hidden = true; editor.hidden = false;
        paint(); body.scrollTop = editorScroll;
        editor.querySelector(`[data-place-room="${id}"]`)?.focus({ preventScroll: true });
    }

    function openRoom(room) {
        if (ui.busy || sheet.closed || !current) return;
        editorScroll = body.scrollTop; selected = room;
        editor.hidden = true; reader.hidden = false;
        history = mountPlaceHistory(reader, readerPlace(), room, { onBack: backToPlace });
        paint(); body.scrollTop = 0;
        reader.querySelector('[data-history-back]')?.focus({ preventScroll: true });
    }

    async function loadRooms() {
        roomController?.abort(); roomController = new AbortController(); const version = ++roomGeneration;
        ui.roomIndex = { rooms: null }; paint();
        try {
            const value = await readRooms(place.id, { signal: roomController.signal });
            if (sheet.closed || version !== roomGeneration) return;
            ui.roomIndex = value;
            // A Dashboard door names the conversation; it opens once the index knows that room.
            const wanted = openRoomChatId; openRoomChatId = null;
            const room = wanted ? (value.rooms || []).find((item) => item.room_chat_id === wanted) : null;
            if (room) { openRoom(room); if (history) return; }
        } catch (error) {
            if (sheet.closed || version !== roomGeneration) return;
            ui.roomIndex = { error: error?.message || String(error) };
        }
        paint();
    }

    function paint() {
        if (sheet.closed) return;
        const scroll = body.scrollTop;
        const active = document.activeElement;
        const focusKey = editor.contains(active) ? focusKeyOf(active) : '';
        const selection = active?.selectionStart == null ? null : [active.selectionStart, active.selectionEnd, active.selectionDirection];
        disposeModels();
        editor.innerHTML = placeSheetBodyHtml(current || place, ui);
        disposeModels = bindModelChoosers(editor);
        body.querySelectorAll('[data-place-act]').forEach((control) => { control.disabled = ui.busy || ui.removed; });
        body.querySelectorAll(`[data-place-state] [data-effort-value]`).forEach((button) => {
            const chosen = button.dataset.effortValue === current?.state;
            button.classList.toggle('active', chosen);
            button.setAttribute('aria-pressed', String(chosen));
            button.disabled = ui.busy;
        });
        body.querySelector('[data-place-state]')?.setAttribute('aria-labelledby', 'place-state-label');
        body.scrollTop = scroll;
        if (focusKey) {
            const field = editor.querySelector(focusKey);
            field?.focus({ preventScroll: true });
            if (selection) field?.setSelectionRange?.(...selection);
        }
        if (current && history) history.updatePlace(readerPlace());
        const meta = dialog.querySelector('[data-place-meta]');
        if (meta) meta.innerHTML = `<span data-i18n-authored>${escapeHtml((current || place).label)}</span> · ${escapeHtml(ui.removed ? tr('places.disconnected.word', 'disconnected') : tr(`places.state.${current.state}`, placeStateWord(current.state)))}`;
        if (selected && meta) meta.innerHTML = `<span data-i18n-authored>${escapeHtml((current || place).label)}</span>`;
        dialog.querySelector('[data-place-label]').textContent = selected?.label || selected?.address || (current || place).label;
    }

    function failed(error, key) {
        if (error?.status === 409 && error.body?.code === 'place_revision_conflict' && Object.hasOwn(error.body, 'place')) {
            current = error.body.place ? normalizePlace(error.body.place) : null;
            ui.removed = !current; ui.conflict = true; ui.confirmDisconnect = false; ui.status = {}; ui.readiness = null;
        } else ui.status = { [key]: { text: error?.message || String(error), tone: 'error' } };
    }

    async function write(action, options = {}) {
        if (ui.busy || !current) return;
        const sent = { ...ui.drafts };
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
            current = await update(current.id, { ...change, expected_revision: current.revision });
            const keys = { 'add-note': ['note'], 'confirm-typed': ['actorId', 'actorLabel'], 'save-folder': ['folder'], 'save-model': ['model'] }[action] || [];
            keys.forEach((draft) => { if (ui.drafts[draft] === sent[draft]) delete ui.drafts[draft]; });
            if (action === 'confirm-typed' && !ui.drafts.actorId && !ui.drafts.actorLabel) ui.adding = false;
            ui.conflict = false;
            ui.status = { [key]: { text: DONE_WORDS[action](current), tone: 'ok' } };
        } catch (error) {
            failed(error, key);
        } finally {
            ui.busy = false;
            paint();
        }
    }

    function onExternal(event) {
        const changed = event?.detail?.place;
        if (ui.busy || sheet.closed) return;
        if (changed?.id === place.id) { current = changed; ui.removed = false; ui.readiness = null; paint(); }
        if (event?.detail?.removed && event.detail.placeId === place.id) { current = null; ui.removed = true; ui.conflict = true; paint(); }
    }
    window.addEventListener(PLACES_CHANGED_EVENT, onExternal);

    body.addEventListener('input', (event) => {
        const field = event.target.closest?.('[data-place-draft]');
        if (field) ui.drafts[field.dataset.placeDraft] = field.value;
    });
    body.addEventListener('toggle', async (event) => {
        if (!event.target.classList?.contains('place-more')) return;
        ui.moreOpen = event.target.open;
        if (!ui.moreOpen || modelsRead) return;
        modelsRead = true;
        try {
            const data = await readModels({ cache: 'no-store', signal: abort.signal });
            if (!Array.isArray(data?.items) || data.read_state === 'failed') throw new Error();
            ui.models = data.items;
            if (!sheet.closed) updateModelChooser(body.querySelector('[data-model-chooser]'), ui.models);
        } catch { ui.status.more = { text: tr('places.models.unavailable', 'Model suggestions could not be read; your model is kept.'), tone: 'warn' }; paint(); }
    }, true);
    body.addEventListener('click', async (event) => {
        if (event.target.closest('[data-place-rooms-retry]')) { void loadRooms(); return; }
        const roomButton = event.target.closest('[data-place-room]');
        if (roomButton) {
            const room = ui.roomIndex.rooms?.find((row) => String(row.room_chat_id) === roomButton.dataset.placeRoom);
            if (room) openRoom(room);
            return;
        }
        const segment = event.target.closest('[data-place-state] [data-effort-value]');
        if (segment) {
            if (segment.dataset.effortValue !== current?.state) await write('state', { value: segment.dataset.effortValue });
            return;
        }
        const button = event.target.closest('[data-place-act]');
        if (!button || button.disabled || ui.busy || !current) return;
        const act = button.dataset.placeAct;
        if (act === 'add-mine') { ui.adding = true; paint(); body.querySelector('[data-place-draft="actorId"]')?.focus(); return; }
        if (act === 'cancel-add') { ui.adding = false; paint(); return; }
        if (act === 'open-connector') { sheet.close(); openSkillsFor({ skill: current.connector }); return; }
        if (act === 'disconnect') { ui.confirmDisconnect = true; paint(); body.querySelector('[data-place-act="disconnect-no"]')?.focus(); return; }
        if (act === 'disconnect-no') { ui.confirmDisconnect = false; paint(); body.querySelector('[data-place-act="disconnect"]')?.focus(); return; }
        if (act === 'disconnect-yes') {
            ui.busy = true; paint();
            try {
                await disconnect(current.id, current.revision);
                sheet.close();
            } catch (error) {
                ui.busy = false;
                failed(error, 'foot');
                paint();
            }
            return;
        }
        await write(act, { fact: button.dataset.fact || '', note: button.dataset.note || '',
            actor: button.dataset.actor || '', actorLabel: button.dataset.actorLabel || '' });
    });
    paint();
    if (place.source) void readStatus(place, { signal: abort.signal }).then((data) => {
        if (sheet.closed || current?.source?.url !== place.source.url) return;
        const status = normalizePlaceStatus(data);
        ui.identity = status.identity; ui.labels = status.labels;
        if (status.readiness) ui.readiness = { ...status.readiness, authored: true };
        paint();
    }).catch(() => { ui.status.state = { text: tr('places.status.unavailable', 'The connector status could not be read.'), tone: 'warn' }; paint(); });
    void loadRooms();
    return { close: sheet.close, get closed() { return sheet.closed; }, get place() { return current; } };
}

// A repaint keeps the owner's focus on the same control (the press that asked for it, or the field).
function focusKeyOf(node) {
    const room = node.closest?.('[data-place-room]');
    if (room) return `[data-place-room="${room.dataset.placeRoom}"]`;
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
