// One PlaceSummary in a document-reader sheet. Owner writes repaint from the server and retain drafts. Its first
// section is the place's context: its circle (a choice: the place moves when another is chosen), the circle's words
// with who wrote them and who placed the place there, the place's own notes in brief and its optional description;
// its conversations follow. Connect is separate in place_connect.js; the body stays here for the place-room integration.
import { bindDialogFocus } from './ui_interactions.js';
import { escapeHtmlAttr as escapeHtml } from './ui_primitives.js';
import { renderSegmentedField } from './page_header.js';
import { chipHtml, skillIconHtml } from './skill_presentation.js';
import { tr, fmt } from './i18n.js';
import { fetchJson } from './api_client.js';
import { catalogPrograms } from './model_program.js';
import { bindModelChoosers, modelChooserHtml, updateModelChooser } from './model_chooser.js';
import { priceWords } from './route_editor_primitives.js';
import { modelCatalog } from './model_catalog_store.js';
import { disconnectPlace, failureText, normalizePlace, normalizePlaceStatus, PLACES_CHANGED_EVENT, readPlaceConnector, readPlaceRooms, readPlaces, updatePlace } from './consciousness_api.js';
import { mountPlaceHistory, placeRoomsHtml } from './place_history.js';
import {
    PLACE_STATES, SILENT_CARD, WAKE_FACT_WORDS, accountIdText, authorWord, capitalized, inheritedModel, listWords, placeService, placeStateOf,
    placeStateWord, placeTitle, placedByWord, sameWords, signalLabels, wakeFactWord, writtenByWord,
} from './consciousness_words.js';

/** Ask the Skills page for one installed skill's card, or Browse filtered to connectors. */
export function openSkillsFor(detail) {
    globalThis.window?.dispatchEvent?.(new CustomEvent('ouro:open-skills', { detail }));
}

/** Ask the Circles page to show itself: `{}`, or `{circle, edit}` to land on one circle, in its edit state. */
export function openCircles(detail = {}) {
    globalThis.window?.dispatchEvent?.(new CustomEvent('ouro:open-circles', { detail }));
}

/** Ask the app for one Dashboard tab (`{tab}`): the page that holds what a line here only points at. */
export function openDashboard(detail = {}) {
    globalThis.window?.dispatchEvent?.(new CustomEvent('ouro:open-dashboard', { detail }));
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

/** The latest captured conversations, newest first. */
export function recentRooms(rooms) {
    return [...(rooms || [])].sort((a, b) => String(b.last_at || '').localeCompare(String(a.last_at || '')));
}

/** Conversations: one door straight to the most recent one, a few recent rows, the rest on request. */
function conversationsHtml(ui) {
    const index = ui.roomIndex || { rooms: null };
    const rooms = index.rooms ? recentRooms(index.rooms) : null;
    const shown = rooms && !ui.allRooms ? rooms.slice(0, 3) : rooms;
    const more = rooms && rooms.length > 3 && !ui.allRooms
        ? `<span class="place-sheet-note"><button type="button" class="consciousness-inline-link ui-hit-area" data-place-act="all-rooms">${escapeHtml(fmt('All {n} ›', { n: rooms.length }))}</button></span>` : '';
    // No room yet: the list under the door says so (placeRoomsHtml), once.
    return `<div class="ui-action-row"><button type="button" class="btn btn-default btn-sm" data-primary data-place-act="open-latest"${rooms?.length ? '' : ' data-disabled'}>${tr('places.rooms.open', 'Open conversation')}</button>${more}</div>
        ${placeRoomsHtml({ ...index, rooms: shown })}`;
}

/** The place's own notes in brief, each in its author's voice (my human's in the owner's tint), three lines at most,
 *  and the way to the whole Notes section; no notes yet shows the way to writing the first one. */
function notesPreviewHtml(place) {
    const notes = place.notes || [];
    const door = (words) => `<button type="button" class="consciousness-inline-link ui-hit-area" data-place-act="show-notes">${words}</button>`;
    if (!notes.length) return `<p class="place-sheet-note">${tr('places.notes.none', 'No notes of its own yet.')} ${door(tr('places.notes.add', 'Add a note ›'))}</p>`;
    const block = (list, owner) => (list.length ? `<div class="place-note place-note-preview${owner ? ' is-owner' : ''}"><p data-i18n-authored>${list.map((note) => escapeHtml(note.text)).join('\n')}</p>${owner
        ? '' : `<div class="place-note-foot">${chipHtml(listWords([...new Set(list.map((note) => authorWord(note.author)))]), 'neutral')}</div>`}</div>` : '');
    return `${block(notes.filter((note) => OWNER_VOICE.has(note.author)), true)}${block(notes.filter((note) => !OWNER_VOICE.has(note.author)), false)}
        <p class="place-sheet-note">${escapeHtml(fmt(notes.length === 1 ? '{n} note of its own' : '{n} notes of its own', { n: notes.length }))} · ${door(tr('places.notes.all', 'All notes ›'))}</p>`;
}

/** The circle this place is in, as a choice (another circle, or none, moves the place), the circle's words (the
 *  context the place inherits, two lines here, whole in the circle) with who wrote them and who placed the place
 *  there, then the place's own notes in brief and its description. A place in no circle has only its own notes. */
function circleHtml(place, ui) {
    const circles = ui.circles;
    const circle = circles?.find((entry) => entry.id === place.circle?.id) || null;
    const options = [{ value: '', label: tr('circles.none', 'Not in a circle'), authored: false },
        ...(circles || []).map((entry) => ({ value: entry.id, label: entry.name, authored: true }))];
    const choice = circles ? `<select class="ui-control" data-place-circle aria-label="${escapeHtml(tr('places.circle', 'Circle'))}">${options.map((option) =>
        `<option value="${escapeHtml(option.value)}"${option.value === (circle?.id || '') ? ' selected' : ''}${option.authored ? ' data-i18n-authored' : ''}>${escapeHtml(option.label)}</option>`).join('')}</select>`
        : ui.circlesFailed ? '' : `<span class="place-sheet-note" role="status">${tr('circles.reading', 'Reading circles…')}</span>`;
    const edit = circle ? `<button type="button" class="consciousness-inline-link ui-hit-area" data-place-act="edit-circle">${tr('places.circle.edit', 'Edit in circle ›')}</button>` : '';
    // A failed read of the circles says so beside the way to read them again (any circles read before stay chosen).
    const failed = ui.circlesFailed ? `${statusLine({ text: tr('circles.unavailable', 'Circles could not be read.'), tone: 'warn' })}
        <div class="place-sheet-actions"><button type="button" class="btn btn-default btn-sm" data-place-act="read-circles">${escapeHtml(tr('places.retry', 'Try again'))}</button></div>` : '';
    const words = !circle ? (circles ? `<p class="place-sheet-note">${tr('places.circle.alone', 'Only its own notes apply here. A circle adds its words to every place in it.')}</p>` : '')
        : circle.text ? `<p class="place-circle-text" data-i18n-authored>${escapeHtml(circle.text)}</p>`
            : `<p class="place-sheet-note">${tr('circles.words.none', 'No words for this circle yet; Edit in circle says who is here and what Ouroboros is here for.')}</p>`;
    const origins = circle ? [circle.text ? writtenByWord(circle.textAuthor) : '', placedByWord(place.circle.placedBy)].filter(Boolean).join(' · ') : '';
    // Ouroboros's placement gives the circle's words no consent of my human's; they may make it their own choice here.
    const keep = circle && !OWNER_VOICE.has(place.circle.placedBy)
        ? ` · <button type="button" class="consciousness-inline-link ui-hit-area" data-place-act="keep-circle">${escapeHtml(fmt('Keep it in {circle}', { circle: circle.name }))}</button>` : '';
    return `${choice ? `<div class="place-field-row place-circle-row">${choice}${edit}</div>` : ''}${failed}${words}${origins ? `<p class="place-sheet-note">${escapeHtml(origins)}${keep}</p>` : ''}
        ${notesPreviewHtml(place)}${descriptionHtml(place, ui)}`;
}

/** The optional one-line description under the place's name in Circles: empty until someone writes it; never a rule. */
function descriptionHtml(place, ui) {
    const value = ui.drafts.description ?? place.description?.text ?? '';
    const author = place.description ? ` ${capitalized(writtenByWord(place.description.author))}.` : '';
    return `<div class="place-field-row">
            <input class="ui-control" data-place-draft="description" data-i18n-authored value="${escapeHtml(value)}" maxlength="240" spellcheck="false" autocomplete="off" aria-label="${escapeHtml(tr('places.description', 'Description'))}" aria-describedby="place-description-help" placeholder="${escapeHtml(tr('places.description.placeholder', 'Description: what this place is for, in one line'))}">
            <button type="button" class="btn btn-default btn-sm" data-place-act="save-description">${tr('places.description.save', 'Save')}</button>
        </div>
        <div class="ui-field-help" id="place-description-help">${escapeHtml(tr('places.description.help', 'Shown under its name in Circles. It says what the place is for and adds no rule.') + author)}</div>`;
}

function stateHtml(place, ui) {
    const state = placeStateOf(place.state);
    const silent = place.state === 'silent';
    const options = PLACE_STATES.map(({ value, label }) => ({ value, label }));
    const group = renderSegmentedField({ target: `place-state-${place.id}`, options, modifier: 'data-place-state' });
    // What the chosen state does: a quiet note in the tray; what Silent guarantees is a host fact in the
    // System voice, under the tray (fills never stack). While consciousness is off it only hears (the host's
    // `only_hears`), whatever the stored choice says it would do.
    const note = place.onlyHears ? tr('places.only_hears.note', 'Consciousness is off: it only hears here.')
        : state?.note || 'Its state could not be read.';
    return `<div class="settings-effort-card place-state">
            <label id="place-state-label">What it does here</label>
            ${group}
            ${silent ? '' : `<p class="place-state-note">${escapeHtml(note)}</p>`}
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
        ${place.onlyHears ? `<p class="place-sheet-note">${tr('places.wake.consciousness_off', 'While consciousness is off, nothing here wakes it.')}
            <button type="button" class="consciousness-inline-link ui-hit-area" data-place-act="open-consciousness">${tr('places.consciousness.door', 'Consciousness ›')}</button></p>`
        : place.state === 'silent' ? `<p class="place-sheet-note">${tr('places.wake.silent', 'Silent: these still wake it to read and think; what it would say or change here is held.')}</p>` : ''}
        ${place.state === 'paused' ? `<p class="place-sheet-note">${tr('places.wake.paused', 'While this place is Paused, nothing here wakes it.')}</p>` : ''}
        <p class="place-sheet-note">${tr('places.wake.honesty', 'Everything else here it sees at its next own wake and decides what to read.')}</p>`;
}

function identityHtml(place, ui) {
    // The same account-scoped match as the server's owner_account_matches, over the accounts that count here (its own and
    // those another place of this bot or app confirms): a confirmation for another account is not this one.
    const counted = [...place.ownerAccounts, ...(place.sharedOwnerAccounts || [])];
    const withOwner = (entry) => ({ ...entry, owner: entry.kind === 'user' && counted.some((account) =>
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
    // A confirmation saved without a name shows the provider's name its senders here carry (username or display name), else the id.
    const seen = new Map(place.accountCandidates.map((account) => [account.actorId, account.label]));
    const named = (account) => account.label || seen.get(account.actorId) || account.actorId;
    const rows = place.ownerAccounts.map((account) => `<div class="harness-account-row place-account">
            <div class="harness-account-main"><strong data-i18n-authored>${escapeHtml(named(account))}</strong></div>
            <div class="harness-account-meta">${escapeHtml([tr('places.account.you', 'you'), account.confirmedAt ? fmt('confirmed on {date}', { date: shortDate(account.confirmedAt) }) : tr('places.account.confirmed', 'confirmed')].join(' · '))}${named(account) !== account.actorId ? ` · <span data-i18n-authored>${escapeHtml(account.actorId)}</span>` : ''}</div>
            <div class="harness-account-actions"><button type="button" class="btn btn-ghost btn-sm" data-place-act="revoke" data-actor="${escapeHtml(account.actorId)}" aria-label="Not my account: remove ${escapeHtml(named(account))}">Remove</button></div>
        </div>`).join('');
    // Confirmed in another place of this bot or app: it counts here too, and is removed where it was confirmed.
    const own = new Set(place.ownerAccounts.map((account) => account.actorId));
    const elsewhere = new Map();
    for (const account of place.sharedOwnerAccounts || []) {
        if (!own.has(account.actorId)) elsewhere.set(account.actorId, [...(elsewhere.get(account.actorId) || []), account]);
    }
    const shared = [...elsewhere.values()].map(([first, ...more]) => `<div class="harness-account-row place-account">
            <div class="harness-account-main"><strong data-i18n-authored>${escapeHtml(named(first))}</strong></div>
            <div class="harness-account-meta">${escapeHtml(tr('places.account.you', 'you'))} · ${fmt('confirmed in {place}', { place: `<span data-i18n-authored>${escapeHtml(first.placeLabel)}</span>` })}${more.length ? escapeHtml(fmt(' and {count} more', { count: more.length })) : ''}${named(first) !== first.actorId ? ` · <span data-i18n-authored>${escapeHtml(first.actorId)}</span>` : ''}</div>
        </div>`).join('');
    const mine = new Set([...own, ...elsewhere.keys()]);
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
    return `${identity || rows || shared || candidates ? `<div class="place-accounts">${identity}${rows}${shared}${candidates}</div>` : ''}${form}
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

/** What an empty Model field uses, in the field's own words: the circle's model, else the consciousness model, each
 *  with the model's name when this read says it (`inheritedModel`). `placeholder` stands in the empty field, `help`
 *  under it, `done` is said once the field was cleared and saved. */
export function inheritedModelWords(place, circles) {
    const { circle, model } = inheritedModel(place, circles?.find((entry) => entry.id === place.circle?.id) || null);
    if (circle) {
        return { placeholder: fmt("Uses {circle}'s model: {model}", { circle, model }),
            help: fmt("The model it thinks with when it wakes here; empty uses {circle}'s model, {model}.", { circle, model }),
            done: fmt("Now uses {circle}'s model: {model}.", { circle, model }) };
    }
    if (model) {
        return { placeholder: fmt('Uses the consciousness model: {model}', { model }),
            help: fmt('The model it thinks with when it wakes here; empty uses the Consciousness model in Models, {model}.', { model }),
            done: fmt('Now uses the consciousness model: {model}.', { model }) };
    }
    return { placeholder: tr('places.model.empty', 'Empty uses the consciousness model'),
        help: tr('places.model.help', 'The model it thinks with when it wakes here; empty uses the Consciousness model in Models.'),
        done: tr('places.model.cleared', 'Now uses the consciousness model.') };
}

// The field holds only the place's own model; empty, the place inherits, and the placeholder and the help say from where.
function moreHtml(place, ui) {
    const value = ui.drafts.model ?? place.model;
    const words = inheritedModelWords(place, ui.circles);
    return `<details class="consciousness-more place-more"${ui.moreOpen ? ' open' : ''}>
            <summary><span class="consciousness-marker" aria-hidden="true">▸</span>More</summary>
            <div class="consciousness-more-body">
                <label class="ui-field">Model
                    <div class="place-field-row">
                        ${modelChooserHtml(`data-place-draft="model" aria-label="${escapeHtml(tr('places.model', 'Model'))}" aria-describedby="place-model-help"`, value, 'place-model-options', ui.models || [], { placeholder: words.placeholder, programs: catalogPrograms(ui.catalog, {sources: ui.catalog?.model_sources}) })}
                        <button type="button" class="btn btn-default btn-sm" data-place-act="save-model">Save</button>
                    </div>
                </label>
                <div class="ui-field-help" id="place-model-help" data-i18n-fmt="places.model.help">${escapeHtml(words.help)}</div>
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

/** The sheet's body for one normalized place and the sheet's own UI state (drafts, busy, statuses, circles). */
export function placeSheetBodyHtml(place, ui = {}) {
    const view = { drafts: {}, status: {}, ...ui };
    if (view.removed) place = { ...place, notes: [], ownerAccounts: [], sharedOwnerAccounts: [], accountCandidates: [] };
    return [
        view.removed ? '' : section('circle', '', circleHtml(place, view), view),
        // What happened here comes right after the context (owner 5 Nov: the place and its conversations first).
        view.removed ? '' : section('conversations', tr('places.conversations', 'Conversations'), conversationsHtml(view), view),
        view.conflict || view.removed || view.readiness || place.readiness ? section('readiness', '',
            (view.conflict ? statusLine({ text: tr('places.conflict', 'Changed elsewhere; your text is kept.'), tone: 'warn' }) : '')
            + (view.removed ? statusLine({ text: tr('places.disconnected', 'This place is disconnected.'), tone: 'muted' })
                : statusLine(view.readiness || (place.readiness && { ...place.readiness, text: tr(`places.readiness.${place.readiness.text}`, place.readiness.text) }))), view) : '',
        view.removed ? '' : section('state', '', stateHtml(place, view), view),
        section('notes', 'Notes', notesHtml(place, view), view),
        view.removed ? '' : section('wakes', tr('places.calls', 'What calls it here'), wakesHtml(place, view), view),
        section('accounts', 'Your accounts here', accountsHtml(place, view), view),
        section('folder', 'Working folder', folderHtml(place, view), view),
        section('more', '', moreHtml(place, view), view),
        view.removed ? '' : section('foot', '', footHtml(place, view), view),
    ].join('');
}

// The sheet's title is the place as its row names it (placeTitle); its meta line says what the title does not: the
// service it is on (unless the title is the service) and the state.
export const placeSheetMetaText = (place, title = '') => [sameWords(title, placeService(place)) ? '' : placeService(place),
    placeStateWord(place.state)].filter(Boolean).join(' · ');

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

/** What one `/api/model-catalog` answer suggests for a place's or a circle's Model field. Both store a route string,
 *  and the local model has none (its row says `use_local`), so that row is left out. Throws for an unread catalog.
 *  Each model's catalog price stands beside its name (`meta`, never part of the route) unless `tierPriced`
 *  (model_catalog_store.js): the model runs as a lead, and on a Fast or Economy tier that price is not the one paid. */
export function modelSuggestions(data, { tierPriced = true } = {}) {
    if (!Array.isArray(data?.items) || data.read_state === 'failed') throw new Error('The model catalog could not be read.');
    return data.items.filter((item) => !item?.use_local)
        .map((item) => (item?.price && !tierPriced ? { ...item, meta: priceWords(item.price) } : item));
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
    case 'circle': return value !== (place.circle?.id || '') ? { circle_id: value } : null;
    case 'keep-circle': return place.circle?.id ? { circle_id: place.circle.id } : null;
    case 'save-description': {
        const line = String(drafts.description ?? place.description?.text ?? '').replace(/\s+/g, ' ').trim();
        return line !== (place.description?.text || '') ? { description: line } : null;
    }
    default: return null;
    }
}

const SECTION_OF = {
    state: 'state', 'add-wake': 'wakes', 'remove-wake': 'wakes', 'add-note': 'notes', 'remove-note': 'notes',
    confirm: 'accounts', 'confirm-typed': 'accounts', revoke: 'accounts', 'save-folder': 'folder', 'save-model': 'more',
    circle: 'circle', 'keep-circle': 'circle', 'save-description': 'circle',
};
// The drafts whose Save writes a field over: each goes out on the revision it was first typed against. A note or an
// account is added, and every other press acts on the place as shown, so those send the shown revision.
const REPLACING = { 'save-description': 'description', 'save-folder': 'folder', 'save-model': 'model' };
const DONE_WORDS = {
    state: (place) => `Now ${placeStateOf(place.state)?.label || place.state}.`, 'add-wake': () => 'Added.', 'remove-wake': () => 'Removed.',
    'add-note': () => 'Note added.', 'remove-note': () => 'Note removed.', confirm: () => 'Confirmed as you.',
    'confirm-typed': () => 'Confirmed as you.', revoke: () => 'Removed from your accounts.',
    'save-folder': () => 'Folder saved.',
    // A cleared field says what the place inherits from now on.
    'save-model': (place, circles) => (place.model ? 'Model saved.' : inheritedModelWords(place, circles).done),
    circle: (place, circles) => (place.circle ? fmt('Now in {circle}.', { circle: circles?.find((c) => c.id === place.circle.id)?.name || '' })
        : tr('circles.left', 'In no circle now.')),
    'keep-circle': () => tr('places.circle.kept', 'Your choice now.'),
    'save-description': (place) => (place.description ? tr('places.description.saved', 'Description saved.') : tr('places.description.cleared', 'Description cleared.')),
};

/** Open one projected place; requests keep its revision and drafts survive a refused or concurrent write. `circles`
 *  are the opener's (Circles); another opener's sheet reads them, and later circle writes, from the places list. */
export function openPlaceSheet(place, { returnFocus = document.activeElement, onClose = null, update = updatePlace,
    disconnect = disconnectPlace, readStatus = readPlaceConnector, readModels = (init) => fetchJson('/api/model-catalog', init),
    readRooms = readPlaceRooms, readList = readPlaces, openRoomChatId = null, circles = null, connectorStatus = null } = {}) {
    let current = place;
    const ui = { drafts: {}, status: {}, busy: false, adding: false, confirmDisconnect: false, moreOpen: false,
        serviceLabel: place.serviceLabel || place.provider, roomIndex: { rooms: null }, circles, allRooms: false, connectorStatus };
    // The revision each replacing draft was first typed against. A newer read meanwhile (another client's write) then
    // meets the server's conflict, never a silent overwrite; the conflict re-bases that draft, so its next Save is the
    // owner's choice. A write from this sheet carries the drafts written on its revision to the one it made.
    const writtenOn = {};
    let disposeModels = () => {}, modelsRead = false;
    let history = null, selected = null, editorScroll = 0, roomController = null, roomGeneration = 0, listGeneration = 0;
    const abort = new AbortController();
    // A single place's answer (after a write) names only its provider; the service label the opener projected stays.
    const named = (subject) => ({ ...subject, serviceLabel: ui.serviceLabel || subject.serviceLabel });
    const titleOf = (subject) => placeTitle(named(subject), {
        circle: (ui.circles || []).find((entry) => entry.id === subject.circle?.id) || null, status: ui.connectorStatus });
    const dialog = sheetShell(`<span class="place-glyph" aria-hidden="true">${skillIconHtml(place.icon, 22)}</span><span data-place-label data-i18n-authored>${escapeHtml(titleOf(place))}</span>`,
        placeSheetMetaText(named(place), titleOf(place)));
    const sheet = showSheet(dialog, { returnFocus, onBack: () => { if (!history) return false; backToPlace(); return true; },
        onClose: () => { abort.abort(); document.removeEventListener('settings-model-catalog:updated', onCatalog); disposeModels(); roomGeneration += 1; listGeneration += 1; roomController?.abort(); history?.close(); window.removeEventListener(PLACES_CHANGED_EVENT, onExternal); onClose?.(); } });
    const { body } = sheet;
    function onCatalog(event) {
        const data = event.detail;
        if (sheet.closed) return;
        ui.models = modelSuggestions(data, { tierPriced: modelCatalog.tierPriced('lead') });
        ui.catalog = data;
        updateModelChooser(body.querySelector('[data-model-chooser]'), ui.models, catalogPrograms(data, {sources: data.model_sources}));
    }
    document.addEventListener('settings-model-catalog:updated', onCatalog);
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
            ui.roomIndex = { error: failureText(error) };
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
        body.querySelectorAll('[data-place-act], [data-place-circle]').forEach((control) => {
            control.disabled = ui.busy || ui.removed || control.hasAttribute('data-disabled');
        });
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
        const subject = current || place, title = titleOf(subject), service = placeService(named(subject));
        const quiet = sameWords(title, service) ? '' : service;
        const meta = dialog.querySelector('[data-place-meta]');
        if (meta) meta.innerHTML = `${quiet ? `<span data-i18n-authored>${escapeHtml(quiet)}</span> · ` : ''}${escapeHtml(ui.removed
            ? tr('places.disconnected.word', 'disconnected') : tr(`places.state.${subject.state}`, placeStateWord(subject.state)))}`;
        // A conversation open in the reader: its name is the title, the place (as its row names it) the meta line.
        if (selected && meta) meta.innerHTML = `<span data-i18n-authored>${escapeHtml([title, quiet].filter(Boolean).join(' · '))}</span>`;
        dialog.querySelector('[data-place-label]').textContent = selected?.label || selected?.address || title;
    }

    // The circles, and the place as the list holds it now, when the opener had no circles or a circle changed. A read
    // after a failed one shows itself (Reading circles…, or the circles read before), then its own result.
    async function loadList() {
        const version = ++listGeneration;
        if (ui.circlesFailed) { ui.circlesFailed = false; paint(); }
        try {
            const data = await readList({ signal: abort.signal });
            if (sheet.closed || version !== listGeneration) return;
            if (!Array.isArray(data.circles)) throw new Error();
            ui.circles = data.circles;
            const fresh = data.places.find((entry) => entry.id === place.id);
            if (fresh && current && !ui.busy && fresh.revision >= current.revision) current = { ...fresh, serviceLabel: current.serviceLabel || fresh.serviceLabel };
        } catch {
            if (sheet.closed || version !== listGeneration) return;
            ui.circlesFailed = true;
        }
        paint();
    }

    // True for the revision conflict: the place as it stands replaces the sheet's, and every draft stays.
    function failed(error, key) {
        if (error?.status === 409 && error.body?.code === 'place_revision_conflict' && Object.hasOwn(error.body, 'place')) {
            current = error.body.place ? normalizePlace(error.body.place) : null;
            ui.removed = !current; ui.conflict = true; ui.confirmDisconnect = false; ui.status = {}; ui.readiness = null;
            return true;
        }
        ui.status = { [key]: { text: failureText(error), tone: 'error' } };
        return false;
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
        const field = REPLACING[action];
        const revision = field && field in writtenOn ? writtenOn[field] : current.revision;
        ui.busy = true;
        ui.status = { [key]: { text: 'Saving…', tone: 'muted' } };
        paint();
        try {
            current = { ...await update(current.id, { ...change, expected_revision: revision }), serviceLabel: current.serviceLabel };
            const keys = { 'add-note': ['note'], 'confirm-typed': ['actorId', 'actorLabel'], 'save-folder': ['folder'], 'save-model': ['model'],
                'save-description': ['description'] }[action] || [];
            keys.forEach((draft) => { if (ui.drafts[draft] === sent[draft]) delete ui.drafts[draft]; });
            // Only this write came between its revision and the one it made: the drafts written on it move along. What it
            // sent is done; words typed since stay a draft on the new revision.
            for (const name of Object.keys(writtenOn)) if (writtenOn[name] === revision) writtenOn[name] = current.revision;
            if (field && !(field in ui.drafts)) delete writtenOn[field];
            if (action === 'confirm-typed' && !ui.drafts.actorId && !ui.drafts.actorLabel) ui.adding = false;
            ui.conflict = false;
            ui.status = { [key]: { text: DONE_WORDS[action](current, ui.circles), tone: 'ok' } };
        } catch (error) {
            if (failed(error, key) && field && field in ui.drafts && current) writtenOn[field] = current.revision;
        } finally {
            ui.busy = false;
            paint();
        }
    }

    function onExternal(event) {
        const changed = event?.detail?.place;
        if (ui.busy || sheet.closed) return;
        if (changed?.id === place.id) { current = { ...changed, serviceLabel: current?.serviceLabel || changed.serviceLabel }; ui.removed = false; ui.readiness = null; paint(); }
        if (event?.detail?.removed && event.detail.placeId === place.id) { current = null; ui.removed = true; ui.conflict = true; paint(); }
        // A circle renamed, reworded or deleted elsewhere: this place's context is read again.
        if (event?.detail?.circle || event?.detail?.circleId) void loadList();
    }
    window.addEventListener(PLACES_CHANGED_EVENT, onExternal);
    body.addEventListener('change', (event) => {
        const choice = event.target.closest?.('[data-place-circle]');
        if (choice && current && !ui.busy) void write('circle', { value: choice.value });
    });

    body.addEventListener('input', (event) => {
        const field = event.target.closest?.('[data-place-draft]');
        if (!field) return;
        const name = field.dataset.placeDraft;
        // The first key fixes the revision a replacing draft is written on; later reads repaint around it.
        if (Object.values(REPLACING).includes(name) && !(name in writtenOn) && current) writtenOn[name] = current.revision;
        ui.drafts[name] = field.value;
    });
    body.addEventListener('toggle', async (event) => {
        if (!event.target.classList?.contains('place-more')) return;
        ui.moreOpen = event.target.open;
        if (!ui.moreOpen || modelsRead) return;
        modelsRead = true;
        try {
            // The window's settings document says the tier the place's lead pays, read once (model_catalog_store.js).
            const [data] = await Promise.all([readModels({ cache: 'no-store', signal: abort.signal }), modelCatalog.readConnections()]);
            onCatalog({detail: data});
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
        // Its Start stands with the daily limit it spends under: the door goes there, it starts nothing itself.
        if (act === 'open-consciousness') { sheet.close(); openDashboard({ tab: 'consciousness' }); return; }
        if (act === 'edit-circle') { const id = current.circle?.id || ''; sheet.close(); openCircles({ circle: id, edit: Boolean(id) }); return; }
        if (act === 'open-latest') { const latest = recentRooms(ui.roomIndex?.rooms)[0]; if (latest) openRoom(latest); return; }
        if (act === 'all-rooms') { ui.allRooms = true; paint(); return; }
        if (act === 'read-circles') {
            // Read again; a focus the repaint dropped lands on the circle choice it brought, else on Try again.
            await loadList();
            if (!editor.contains(document.activeElement)) body.querySelector(ui.circles ? '[data-place-circle]' : '[data-place-act="read-circles"]')?.focus({ preventScroll: true });
            return;
        }
        if (act === 'show-notes') {
            // The brief's door: the whole Notes section, and the new-note field when there is nothing to read yet.
            editor.querySelector('[data-place-section="notes"]')?.scrollIntoView({ block: 'start' });
            if (!current.notes.length) editor.querySelector('[data-place-draft="note"]')?.focus({ preventScroll: true });
            return;
        }
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
    if (!ui.circles) void loadList();
    if (place.source) void readStatus(place, { signal: abort.signal }).then((data) => {
        if (sheet.closed || current?.source?.url !== place.source.url) return;
        const status = normalizePlaceStatus(data);
        ui.connectorStatus = data;
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
    if (node.closest?.('[data-place-circle]')) return '[data-place-circle]';
    const segment = node.closest?.('[data-effort-value]');
    return segment ? `[data-place-state] [data-effort-value="${segment.dataset.effortValue}"]` : '';
}
