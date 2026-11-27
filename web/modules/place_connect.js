// Add a place (owner 5 Nov: «сначала человеческое описание, источник и доступ раскрываются по необходимости»): first
// what the place is and what Ouroboros is there for, with an optional short description; then the installed
// connectors; Where only when a choice is needed; a connector's settings or Skills only when it is not ready.
// Connectors declaring `connect_mode: 'exact'` never offer Whole account or implicit account-wide auto-connect.
// A new place lands Silent with those words as its first note, its description and its circle in one write. A place
// that is connected already shows its own notes, and only the changes ticked there are saved, under its revision.
import { escapeHtmlAttr as escapeHtml } from './utils.js';
import { withinRequestLifetime } from './api_client.js';
import { tr, fmt } from './i18n.js';
import { skillIconHtml } from './skill_presentation.js';
import { PLACES_CHANGED_EVENT, connectPlace, connectorKey, connectorMatchesPlace, isOwnerBotPair, normalizePlace, readPlaceConnector, readPlaces, updatePlace } from './consciousness_api.js';
import { authorWord } from './consciousness_words.js';
import { openPlaceSheet, openSkillsFor, sheetShell, showSheet, statusLine } from './place_sheet.js';
import {
    confirmedTelegramBot, createTelegramGroupsController, parseOwnerBotStatus, parseSeparateBotStatus,
    parseTelegramGroupFeed, requestFailureText, telegramFacts, telegramGroupChoices, telegramGroupsEntryHtml, telegramGroupsHtml,
    validatedTelegramInviteUrl,
} from './telegram_groups_connect.js';

export {
    confirmedTelegramBot, parseOwnerBotStatus, parseSeparateBotStatus, parseTelegramGroupFeed,
    telegramGroupChoices, validatedTelegramInviteUrl,
};

// Where a new place lands, in the host's words for the switch as it stands (consciousness_view.connect_note).
const silentCopy = (view) => (view?.connectNote ? tr(view.connectNote, view.connectNote)
    : tr('places.connect.silent', 'It starts Silent: it hears and thinks there and sends nothing until you make it Active.'));
const wordsExample = () => tr('circles.words.example', 'A team chat: answer when asked; keep money and private matters out; bring me what matters.');
/** The owner-words field of a circle or of a new place: the owner's voice, an example as its placeholder. */
export const ownerWordsFieldHtml = (attrs, value = '', rows = 3) => `<textarea class="ui-control consciousness-owner-words" rows="${rows}" ${attrs} data-i18n-authored spellcheck="false" placeholder="${escapeHtml(wordsExample())}">${escapeHtml(value)}</textarea>`;
const button = (act, label, attrs = '', tone = 'btn-default') => `<button type="button" class="btn ${tone} btn-sm" data-connect-act="${act}" ${attrs}>${escapeHtml(label)}</button>`;
const offline = (connector) => Boolean(connector.readiness && connector.readiness.tone !== 'ok');
const isOwnerBot = isOwnerBotPair;

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

const existingPlace = (places, connector, row) => (Array.isArray(places) ? places : []).find((place) => connectorMatchesPlace(connector, place)
    && place.accountId === row.accountId && place.conversationId === row.conversationId);

// Only codes raised strictly before connect_declared_place writes: places_api.py runs connect_declared_place and
// post-write place_view under one except (ValueError, TypeError) -> 400 (code or "place_invalid"), so place_invalid
// and place_records/validate errors cannot prove the group was not created.
const PLACE_CONNECT_REFUSAL_CODES = new Set([
    'place_connect_invalid', 'place_declaration_unavailable', 'place_author_invalid', 'place_description_invalid',
]);
const isTypedPlaceConnectRefusal = (error) => error?.status === 400
    && error.body?.ok === false
    && typeof error.body?.code === 'string'
    && PLACE_CONNECT_REFUSAL_CODES.has(error.body.code);
const pendingConsent = (pending) => {
    const accountId = typeof pending?.accountId === 'string' ? pending.accountId.trim() : '';
    return accountId && accountId !== '*' && typeof pending?.enabled === 'boolean'
        ? { accountId, enabled: pending.enabled, legacy: pending.legacy === true }
        : null;
};

const circleIsMissing = (view) => Boolean(view?.circle?.id
    && (view.circleMissing || (Array.isArray(view.circles) && !view.circles.some((entry) => entry.id === view.circle.id))));

function circlePickerHtml(view) {
    const missing = circleIsMissing(view);
    if (!missing && !view?.chooseCircle) return '';
    const existing = (Array.isArray(view.circles) ? view.circles : [])
        .filter((entry) => !(missing && entry.id === view.circle?.id));
    const missingOpt = missing
        ? `<option value="${escapeHtml(view.circle.id)}" selected disabled data-i18n-authored>${escapeHtml(fmt('{circle} (deleted)', { circle: view.circle.name }))}</option>`
        : '';
    const selectedId = missing ? '' : (view.circle?.id || '');
    const options = [
        missingOpt,
        `<option value=""${!missing && !selectedId ? ' selected' : ''}>${escapeHtml(tr('circles.none', 'Not in a circle'))}</option>`,
        ...existing.map((entry) => `<option value="${escapeHtml(entry.id)}"${entry.id === selectedId ? ' selected' : ''} data-i18n-authored>${escapeHtml(entry.name)}</option>`),
    ].join('');
    const note = missing
        ? statusLine({ text: fmt('{circle} no longer exists. Choose another circle or Not in a circle.', { circle: view.circle.name }), tone: 'warn' })
        : '';
    return `<div class="place-connect-group" data-connect-circle-box>${note}
        <div class="place-field-row place-circle-row"><select class="ui-control" data-connect-circle aria-label="${escapeHtml(tr('places.circle', 'Circle'))}"${view.busy ? ' disabled' : ''}>${options}</select></div></div>`;
}

/** Step one: what the place is and what Ouroboros is there for (its first note), and an optional short description. */
export function wordsHtml(view) {
    const where = view.circle && !circleIsMissing(view) ? fmt('{circle}’s words apply there too.', { circle: `<span data-i18n-authored>${escapeHtml(view.circle.name)}</span>` }) : '';
    return `<section class="place-sheet-section">
        <h3 id="place-add-question">${tr('places.add.question', 'What is this place, and what am I here for?')}</h3>
        ${circlePickerHtml(view)}
        ${ownerWordsFieldHtml('data-connect-draft="words" aria-labelledby="place-add-question"', view.words || '', 4)}
        <p class="place-sheet-note">${tr('places.add.note', 'Your words become the place’s first note.')} ${where}</p>
        <label class="ui-field">${tr('places.description.optional', 'Description (optional)')}<input class="ui-control" data-connect-draft="description" data-i18n-authored maxlength="240" value="${escapeHtml(view.description || '')}" spellcheck="false" autocomplete="off" aria-describedby="place-add-description-help"></label>
        <div class="ui-field-help" id="place-add-description-help">${tr('places.description.help', 'Shown under its name in Circles. It says what the place is for and adds no rule.')}</div>
        <div class="place-sheet-actions">${button('next', tr('places.add.next', 'Next: where it is'))}</div>${statusLine(view.status)}
    </section>`;
}

function whereHtml(view) {
    const { selected: connector, places = [], choices, busy = false, tgOther = false } = view;
    const connectDisabled = busy || circleIsMissing(view);
    const exact = connector.connectMode === 'exact';
    const rowHtml = (row, index) => {
        const existing = existingPlace(places, connector, row);
        // One family with the connector list: the row's state under its name, its verb on the right.
        const state = existing ? `<div class="harness-account-meta"><span class="ui-status" data-tone="ok">${tr('places.connected', 'Connected')}</span></div>` : '';
        const action = existing
            ? button('adopt', tr('places.choose', 'Choose'), `data-place-id="${escapeHtml(existing.id)}"${busy ? ' disabled' : ''}`)
            : button('connect', tr('places.connect.action', 'Connect'), `data-choice="${index}"${connectDisabled ? ' disabled' : ''}`);
        return `<div class="harness-account-row place-connector-row">
            <div class="harness-account-main"><strong${index === 'whole' ? '' : ' data-i18n-authored'}>${escapeHtml(row.label)}</strong></div>
            ${state}<div class="harness-account-actions">${action}</div>
        </div>`;
    };
    const effectiveState = exact && (!connector.source ? !offline(connector) : choices?.state === 'absent')
        ? (!connector.source ? 'unavailable' : 'empty') : choices?.state || '';
    let note = '';
    // No status route on a generic connector that is on means no list to choose from (Whole account alone), not «off»;
    // an `exact` connector never offers Whole account, so a missing route cannot list exact places.
    if (!connector.source) {
        note = offline(connector) ? tr('places.connector.off', 'Turn the connector on to choose where')
            : exact ? tr('places.choices.unavailable', 'Could not read the connector’s list.') : '';
    } else if (!choices) note = tr('places.choices.loading', 'Reading where you can connect…');
    else if (effectiveState === 'empty') note = tr('places.choices.empty', 'None offered yet.');
    else if (effectiveState === 'unavailable') note = tr('places.choices.unavailable', 'Could not read the connector’s list.');
    // The connector's settings only when it is not ready, or offers nowhere to connect.
    const settings = effectiveState === 'empty' || offline(connector) ? button('settings', connector.settings
        ? tr('places.connector.settings', 'Open connector settings') : tr('places.connector.page', 'Open connector page')) : '';
    const retry = effectiveState === 'unavailable' && connector.source ? button('retry-choices', tr('places.retry', 'Try again')) : '';
    const readiness = view.connectorStatus?.text ? { ...view.connectorStatus, authored: true } : connector.readiness;
    const wholeHtml = exact ? '' : rowHtml({ accountId: '', conversationId: '', label: tr('places.whole', 'Whole account') }, 'whole');
    const instancesHtml = (choices?.rows || []).map((row, index) =>
        tgOther && /^-\d+$/.test(row.conversationId) ? '' : rowHtml(row, index)).join('');
    return `<section class="place-sheet-section">
        <div class="place-sheet-actions">${button('back', tr('places.connect.back', 'Back'))}</div>
        <h3>${tr('places.where', 'Where')} · <span data-i18n-authored>${escapeHtml(connector.label)}</span></h3>
        ${readiness ? statusLine(readiness.authored ? readiness : { ...readiness, text: tr(`places.readiness.${readiness.text}`, readiness.text) }) : ''}
        <p class="place-sheet-note">${silentCopy(view)}</p>
        ${circlePickerHtml(view)}
        ${note ? `<p class="place-sheet-note" role="status">${escapeHtml(note)}</p>` : ''}
        ${settings || retry ? `<div class="place-sheet-actions">${settings}${retry}</div>` : ''}
        ${wholeHtml || instancesHtml ? `<div class="place-accounts">${wholeHtml}${instancesHtml}</div>` : ''}${statusLine(view.status)}
    </section>`;
}

/** The verb of a connector row: one that is off opens Skills to enable it; one with places lets Where offer them
 *  beside what can still be connected; a ready one connects. */
function connectorAction(connector, count) {
    const attrs = `data-connector="${escapeHtml(connector.connector)}" data-connector-key="${escapeHtml(connectorKey(connector))}"`;
    if (offline(connector)) return button('skills', tr('places.connector.enable', 'Enable in Skills'), attrs);
    return button('choose', count ? tr('places.choose.where', 'Choose where') : tr('places.connect.action', 'Connect'), attrs);
}

export function connectListHtml(view = {}) {
    const { connectors = null, places = [], error = '', selected = null, choices = null,
        status = {}, busy = false, connectorStatus = null, connectNote = '', tgOpen = false, tgOther = false } = view;
    if (error) return `<section class="place-sheet-section">${statusLine({ text: error, tone: 'error' })}${button('retry', tr('places.retry', 'Try again'))}</section>`;
    if (!connectors) return `<p class="place-sheet-section place-sheet-note" role="status">${tr('places.connect.loading', 'Reading the installed connectors…')}</p>`;
    if (tgOpen && !tgOther) return telegramGroupsHtml(view, silentCopy(view), circlePickerHtml(view), circleIsMissing(view));
    if (selected) return whereHtml(view);

    const back = `<div class="place-sheet-actions">${button('back-words', tr('places.connect.back', 'Back'))}</div>`;
    const tgInfo = telegramFacts(view);
    // One group door. The ordinary Telegram Bot still offers DMs and Whole account inside it; other connector rows stay.
    const ordinaryConnectors = connectors.filter((connector) => !isOwnerBot(connector) && connector !== tgInfo.tgBotConnector);
    if (!tgInfo.shown && !ordinaryConnectors.length) {
        return `<section class="place-sheet-section">${back}<p>${tr('places.connect.empty', 'No installed connector can connect a place yet.')}</p>
            ${button('browse', tr('places.connect.browse', 'Find connectors in Extensions'))}</section>`;
    }
    const tgGroupsRowHtml = telegramGroupsEntryHtml(tgInfo, places);
    return `<section class="place-sheet-section">${back}<h3>${tr('places.where_from', 'Where is it?')}</h3><p class="place-sheet-note">${silentCopy({ connectNote })}</p>
        ${circlePickerHtml(view)}
        <div class="place-accounts">${tgGroupsRowHtml}${ordinaryConnectors.map((connector) => {
            const n = places.filter((place) => connectorMatchesPlace(connector, place)).length;
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
export function adoptOffers(view) {
    const { adopt: place, circle, words = '', description = '' } = view;
    const has = (text) => place.notes.some((note) => note.text.trim() === text.trim());
    return [circle && !circleIsMissing(view) && place.circle?.id !== circle.id ? 'circle' : '', words.trim() && !has(words) ? 'note' : '',
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
        ${circlePickerHtml(view)}
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
    readStatus = readPlaceConnector, openPlace = openPlaceSheet, openSettings = null, circle = null, telegram = undefined,
    resumeTelegram = null, onTelegramSetup = null } = {}) {
    const dialog = sheetShell(escapeHtml(tr('places.add', 'Add a place')),
        circle ? fmt('To {circle}', { circle: circle.name }) : tr('circles.none', 'Not in a circle'), 'place-connect');
    const abort = new AbortController();
    const sheet = showSheet(dialog, { returnFocus, onClose: () => abort.abort() });
    const view = {
        connectors: null, places: [], circles: null, error: '', status: {}, selected: null, choices: null, busy: false,
        circle: circle ? { id: circle.id, name: circle.name } : null, circleMissing: false, chooseCircle: false,
        words: resumeTelegram?.words || '', description: resumeTelegram?.description || '',
        step: resumeTelegram ? 'list' : 'words', adopt: null, picks: {}, connectorStatus: null, rawStatus: null,
        tgOpen: Boolean(resumeTelegram), tgBranch: resumeTelegram?.branch === 'separate' ? 'separate' : 'personal',
        tgOther: Boolean(resumeTelegram?.other),
        tg: {
            skills: null, groupFeed: null, ownerStatus: null, rawOwnerStatus: null, separateStatus: null, rawSeparateStatus: null,
            consentPending: pendingConsent(resumeTelegram?.consentPending),
        },
    };
    const syncDialogMeta = () => {
        const meta = dialog.querySelector('[data-place-meta]');
        if (!meta) return;
        meta.textContent = view.circle
            ? (circleIsMissing(view)
                ? fmt('{circle} (deleted)', { circle: view.circle.name })
                : fmt('To {circle}', { circle: view.circle.name }))
            : tr('circles.none', 'Not in a circle');
    };
    const saveTelegramSetup = () => {
        const consentPending = pendingConsent(view.tg.consentPending);
        return onTelegramSetup?.(
            {
                words: view.words,
                description: view.description,
                branch: view.tgBranch,
                ...(view.tgOther ? { other: true } : {}),
                ...(consentPending ? { consentPending } : {}),
            },
            view.circle,
        );
    };
    let generation = 0;
    let loadSeq = 0;
    const captureFocus = (body) => {
        const active = globalThis.document?.activeElement;
        if (!active || active === body || !body.contains?.(active)) return null;
        const keys = ['data-connect-act', 'data-connect-draft', 'data-connect-adopt', 'data-connect-circle', 'data-connect-invite',
            'data-connector-key', 'data-connector', 'data-branch', 'data-choice', 'data-place-id'];
        const sig = keys.filter((key) => active.hasAttribute?.(key)).map((key) => [key, active.getAttribute(key)]);
        if (!sig.length) return null;
        return () => {
            const match = [...body.querySelectorAll(`[${sig[0][0]}]`)].find((el) =>
                !el.disabled && sig.every(([key, value]) => el.hasAttribute(key) && el.getAttribute(key) === value));
            if (match && globalThis.document?.activeElement !== match) match.focus?.();
        };
    };
    const paint = () => {
        if (sheet.closed) return;
        const restoreFocus = captureFocus(sheet.body);
        sheet.body.innerHTML = view.step === 'words' ? wordsHtml(view) : view.step === 'adopt' ? adoptHtml(view) : connectListHtml(view);
        restoreFocus?.();
    };
    const enrichTelegram = () => {
        if (sheet.closed || abort.signal.aborted) return;
        if (view.step !== 'list' || view.selected || view.busy || view.error || !Array.isArray(view.connectors)) return;
        if (view.tgOpen) { paint(); return; }
        const accounts = sheet.body.querySelector('.place-accounts');
        const tgHtml = telegramGroupsEntryHtml(telegramFacts(view), view.places);
        if (!accounts) {
            if (tgHtml) paint();
            return;
        }
        const existing = accounts.querySelector('[data-connector-entry="telegram-groups"]');
        if (!tgHtml) {
            existing?.remove();
            if (!accounts.firstElementChild) paint();
            return;
        }
        const wrap = globalThis.document.createElement('div');
        wrap.innerHTML = tgHtml;
        const fresh = wrap.firstElementChild;
        if (!fresh) return;
        if (existing) {
            const restore = existing.contains?.(globalThis.document?.activeElement) ? captureFocus(sheet.body) : null;
            accounts.insertBefore(fresh, existing);
            existing.remove();
            restore?.();
        } else {
            accounts.insertBefore(fresh, accounts.firstChild);
        }
    };
    const land = (place, connectorOverride = null) => {
        // A single place's answer names only its provider; the flow holds the connector's declared name.
        if (!sheet.closed) {
            const sourceConnector = connectorOverride || view.selected;
            const raw = view.rawStatus;
            const connectorStatus = raw && !(isOwnerBotPair(place) && place.accountId && raw.account_id !== place.accountId)
                ? raw : null;
            sheet.close();
            openPlace({ ...place, serviceLabel: sourceConnector?.label || place.serviceLabel }, { returnFocus, connectorStatus });
        }
    };
    const showAdopt = (place, status = {}) => { view.adopt = place; view.picks = {}; view.status = status; view.step = 'adopt'; paint(); };
    const checkTelegramConnect = async () => {
        const pending = view.tgConnectPending;
        if (!pending) return;
        view.busy = true; paint();
        try {
            const latest = await withinRequestLifetime((signal) => read({ signal }), { signal: abort.signal, lifetimeMs: 10000 });
            if (sheet.closed) return;
            Object.assign(view, latest);
            const existing = existingPlace(view.places, pending.connector, pending.choice);
            if (!existing) return;  // Absence from this read does not settle an in-flight write.
            view.tgConnectPending = null;
            view.rawStatus = null;
            // Refresh Circles and other place listeners behind the sheet when read-only reconciliation confirms the write.
            globalThis.window?.dispatchEvent?.(new CustomEvent(PLACES_CHANGED_EVENT, { detail: { place: existing } }));
            if (adoptOffers({ ...view, adopt: existing }).length) showAdopt(existing);
            else land(existing, pending.connector);
        } catch { /* The outcome is still unknown; Check again only reads. */ }
        finally { view.busy = false; paint(); }
    };
    const refreshCirclesForMissing = async (current) => {
        try {
            const latest = await withinRequestLifetime((signal) => read({ signal }), { signal: abort.signal, lifetimeMs: 10000 });
            if (current === generation && !sheet.closed && Array.isArray(latest?.circles)) view.circles = latest.circles;
        } catch { /* Keep the refusal reason; the owner can still choose Not in a circle or retry. */ }
    };
    const connectNow = async (choice, connectorOverride = null) => {
        const targetConnector = connectorOverride || view.selected;
        const teleGroup = view.tgOpen && !view.tgOther;
        if (!targetConnector || (teleGroup && view.tgConnectPending)) return;
        if (circleIsMissing(view)) {
            view.chooseCircle = true;
            view.status = {};
            paint();
            sheet.body.querySelector('[data-connect-circle]')?.focus();
            return;
        }
        const current = generation;
        view.busy = true; view.status = { text: tr('places.connect.busy', 'Connecting…'), tone: 'muted' }; paint();
        try {
            const siblings = (view.connectors || []).filter((c) => c.connector === targetConnector.connector);
            const contrib = targetConnector.contributionId && (targetConnector.contributionId !== 'place' || siblings.length > 1)
                ? { contributionId: targetConnector.contributionId } : {};
            const body = {
                connector: targetConnector.connector, ...contrib, ...choice, note: view.words.trim(),
                description: view.description.trim(), circleId: view.circle?.id || '',
            };
            const created = await (teleGroup
                ? withinRequestLifetime((signal) => connect(body, { signal }), { signal: abort.signal, lifetimeMs: 10000 })
                : connect(body));
            if (current !== generation || sheet.closed) return;
            land(created, targetConnector);
        } catch (error) {
            if (current !== generation || sheet.closed) return;
            // Connected meanwhile: nothing was written; its own notes come first, the changes are chosen there.
            if (error?.status === 409 && error.body?.ok === false && error.body?.code === 'place_exists' && error.body.place) {
                showAdopt(normalizePlace(error.body.place), { text: tr('places.adopt.exists', 'It is connected already: choose what to change.'), tone: 'warn' });
                return;
            }
            if (error?.status === 400 && error.body?.ok === false && error.body?.code === 'circle_not_found') {
                view.circleMissing = true;
                view.chooseCircle = true;
                syncDialogMeta();
                view.status = {};
                await refreshCirclesForMissing(current);
                return;
            }
            if (teleGroup && !isTypedPlaceConnectRefusal(error)) {
                view.tgConnectPending = { connector: targetConnector, choice };
                view.status = {};
                await checkTelegramConnect();
                return;
            }
            view.status = { text: requestFailureText(error), tone: 'error' };
        } finally { view.busy = false; paint(); }
    };
    const tgController = createTelegramGroupsController({
        view, sheet, signal: abort.signal, read, readStatus, openSettings, openConnectorSettings, paint,
        enrich: enrichTelegram, nextGen: () => ++generation, getGen: () => generation, telegram, enabledByDefault: read === readPlaces,
        onSetup: saveTelegramSetup,
    });
    const load = async () => {
        const currentLoad = ++loadSeq;
        view.error = ''; view.connectors = null; paint();
        try {
            const next = await read({ signal: abort.signal });
            if (sheet.closed || abort.signal.aborted || currentLoad !== loadSeq) return;
            Object.assign(view, next);
            if (circleIsMissing(view)) {
                view.chooseCircle = true;
                syncDialogMeta();
            }
            if (view.tgOpen && view.tgOther) {
                const bot = telegramFacts(view).tgBotConnector;
                if (bot && !offline(bot)) view.selected = bot;
                else view.tgOther = false;
            }
            paint();
            void tgController.readPassive(view.connectors, { isCurrent: () => currentLoad === loadSeq });
            if (view.tgOpen && view.tgOther && view.selected) await loadChoices();
            else if (view.tgOpen && !view.tgOther) await tgController.refresh();
        } catch (error) {
            if (sheet.closed || abort.signal.aborted || currentLoad !== loadSeq) return;
            view.error = fmt('Could not read the connectors: {reason}', { reason: requestFailureText(error) });
            paint();
        }
    };
    const saveAdopt = async () => {
        const place = view.adopt;
        const offers = adoptOffers(view);
        const ticked = (key) => offers.includes(key) && view.picks[key] !== false;
        const change = { ...(ticked('circle') && view.circle ? { circle_id: view.circle.id } : {}), ...(ticked('note') ? { add_note: view.words.trim() } : {}),
            ...(ticked('description') ? { description: view.description.trim() } : {}) };
        if (!Object.keys(change).length) { view.status = { text: tr('places.adopt.none', 'Tick what to change, or go back.'), tone: 'muted' }; paint(); return; }
        view.busy = true; view.status = { text: tr('places.saving', 'Saving…'), tone: 'muted' }; paint();
        try {
            land(await update(place.id, { ...change, expected_revision: place.revision }));
        } catch (error) {
            if (error?.status === 409 && Object.hasOwn(error.body || {}, 'place')) {
                if (!error.body.place) { view.status = { text: tr('places.disconnected', 'This place is disconnected.'), tone: 'warn' }; return; }
                view.adopt = normalizePlace(error.body.place);
                view.status = { text: tr('places.adopt.changed', 'Changed elsewhere; check it and save again.'), tone: 'warn' };
                return;
            }
            if (error?.status === 400 && error.body?.ok === false && error.body?.code === 'circle_not_found') {
                view.circleMissing = true;
                view.chooseCircle = true;
                syncDialogMeta();
                view.status = {};
                await refreshCirclesForMissing(generation);
                return;
            }
            view.status = { text: requestFailureText(error), tone: 'error' };
        } finally { view.busy = false; paint(); }
    };
    const loadChoices = async () => {
        const current = ++generation;
        const target = view.selected;
        const exact = target?.connectMode === 'exact';
        view.choices = null; view.connectorStatus = null; view.rawStatus = null; paint();
        try {
            const data = target.source ? await readStatus(target, { signal: abort.signal }) : {};
            if (current !== generation || sheet.closed) return;
            view.rawStatus = data;
            view.choices = connectChoices(data);
            view.connectorStatus = typeof data.text === 'string' ? { text: data.text, tone: data.tone } : null;
        } catch { if (current === generation) view.choices = { state: 'unavailable', rows: [] }; }
        if (current !== generation || sheet.closed) return;
        const placed = view.places.some((place) => connectorMatchesPlace(target, place));
        if (current === generation && !exact && !view.tgOther && view.choices?.state === 'absent' && !offline(target) && !placed) {
            await connectNow({});
            return;
        }
        paint();
    };
    sheet.body.addEventListener('input', (event) => {
        const field = event.target.closest?.('[data-connect-draft]');
        if (field) view[field.dataset.connectDraft] = field.value;
    });
    sheet.body.addEventListener('change', (event) => {
        const pick = event.target.closest?.('[data-connect-adopt]');
        if (pick) { view.picks[pick.dataset.connectAdopt] = pick.checked; return; }
        const circleSelect = event.target.closest?.('[data-connect-circle]');
        if (!circleSelect || view.busy) return;
        const nextId = circleSelect.value;
        if (!nextId) {
            view.circle = null;
            view.circleMissing = false;
            view.chooseCircle = true;
            view.status = {};
            syncDialogMeta();
            paint();
            return;
        }
        const found = (view.circles || []).find((entry) => entry.id === nextId);
        if (found) {
            view.circle = { id: found.id, name: found.name };
            view.circleMissing = false;
            view.chooseCircle = true;
            view.status = {};
            syncDialogMeta();
            paint();
        }
    });
    sheet.body.addEventListener('click', async (event) => {
        const control = event.target?.closest?.('[data-connect-act]');
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
        if (act === 'back-words') {
            ++generation; view.selected = null; view.choices = null; view.rawStatus = null; view.tgOpen = false;
            tgController.cancelInFlight(); view.status = {}; view.step = 'words'; paint();
            sheet.body.querySelector('[data-connect-draft="words"]')?.focus(); return;
        }
        if (act === 'back-where') { view.adopt = null; view.status = {}; view.step = 'list'; paint(); return; }
        if (act === 'retry') { await load(); return; }
        if (act === 'browse') { sheet.close(); openSkillsFor({ category: 'connectors' }); return; }
        if (act === 'skills') { sheet.close(); openSkillsFor({ skill: control.dataset.connector }); return; }
        if (act === 'back') {
            if (view.tgOther) {
                view.tgOther = false; view.status = {};
                await tgController.refresh(); return;
            }
            ++generation; view.selected = null; view.choices = null; view.rawStatus = null; view.tgOpen = false;
            tgController.cancelInFlight(); view.status = {}; paint(); return;
        }
        if (act === 'settings') {
            if (view.tgOther) saveTelegramSetup();
            sheet.close();
            await openConnectorSettings(view.selected, openSettings);
            return;
        }
        if (act === 'tg-check-connect') { await checkTelegramConnect(); return; }
        if (act === 'tg-other-chats') {
            view.selected = telegramFacts(view).tgBotConnector;
            if (view.selected && !offline(view.selected)) { view.tgOther = true; await loadChoices(); }
            return;
        }
        if ((act === 'open-tg-groups' || act.startsWith('tg-')) && await tgController.handleAction(act, control)) return;
        if (act === 'choose') {
            const key = control.dataset.connectorKey;
            view.selected = view.connectors.find((item) => (key ? connectorKey(item) === key : item.connector === control.dataset.connector) && !isOwnerBot(item));
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
        if (control.dataset.choice === 'whole' && (view.selected.connectMode === 'exact' || (view.tgOpen && !view.tgOther))) return;
        const choice = control.dataset.choice === 'whole' ? {} : view.choices?.rows[Number(control.dataset.choice)];
        if (view.tgOther && /^-\d+$/.test(choice?.conversationId)) return;
        if (choice) await connectNow(choice);
    });
    paint();
    if (resumeTelegram) void load();
    else sheet.body.querySelector('[data-connect-draft="words"]')?.focus();
    return sheet;
}
