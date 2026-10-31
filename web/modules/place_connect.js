// Connect only: installed services, then Whole account or an instance from the declared status route.
import { escapeHtmlAttr as escapeHtml } from './utils.js';
import { tr, fmt } from './i18n.js';
import { skillIconHtml } from './skill_presentation.js';
import { connectPlace, readPlaceConnector, readPlaces } from './consciousness_api.js';
import { openPlaceSheet, openSkillsFor, sheetShell, showSheet, statusLine } from './place_sheet.js';

export const connectPlaceButtonHtml = () => `<button type="button" class="btn btn-default btn-sm" data-place-connect>${tr('places.connect', 'Connect a place')}</button>`;
const silentCopy = () => tr('places.connect.silent', 'It starts Silent: it hears and thinks there and sends nothing until you make it Active.');
const button = (act, label, attrs = '') => `<button type="button" class="btn btn-default btn-sm" data-connect-act="${act}" ${attrs}>${escapeHtml(label)}</button>`;

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

function whereHtml(view) {
    const { selected: connector, places, choices, busy } = view;
    const rowHtml = (row, index) => {
        const existing = existingPlace(places, connector, row);
        const action = existing ? `<span>${tr('places.connected', 'Connected')} · </span>${button('open', tr('places.open', 'Open'), `data-place-id="${escapeHtml(existing.id)}"`)}`
            : button('connect', tr('places.connect.action', 'Connect'), `data-choice="${index}"${busy ? ' disabled' : ''}`);
        return `<div class="harness-account-row place-connector-row">
            <div class="harness-account-main"><strong${index === 'whole' ? '' : ' data-i18n-authored'}>${escapeHtml(row.label)}</strong></div>
            <div class="harness-account-actions">${action}</div>
        </div>`;
    };
    let note = '';
    if (!connector.source) note = tr('places.connector.off', 'Turn the connector on to choose where');
    else if (!choices) note = tr('places.choices.loading', 'Reading where you can connect…');
    else if (choices.state === 'empty') note = tr('places.choices.empty', 'None offered yet.');
    else if (choices.state === 'unavailable') note = tr('places.choices.unavailable', 'Could not read the connector’s list.');
    const settings = choices?.state === 'empty' ? button('settings', connector.settings
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

export function connectListHtml({ connectors = null, places = [], error = '', selected = null, choices = null,
    status = {}, busy = false, connectorStatus = null } = {}) {
    if (error) return `<section class="place-sheet-section">${statusLine({ text: error, tone: 'error' })}${button('retry', tr('places.retry', 'Try again'))}</section>`;
    if (!connectors) return `<p class="place-sheet-section place-sheet-note" role="status">${tr('places.connect.loading', 'Reading the installed connectors…')}</p>`;
    if (selected) return whereHtml({ selected, choices, places, status, busy, connectorStatus });
    if (!connectors.length) return `<section class="place-sheet-section"><p>${tr('places.connect.empty', 'No installed connector can connect a place yet.')}</p>
        ${button('browse', tr('places.connect.browse', 'Find connectors in Skills'))}</section>`;
    return `<section class="place-sheet-section"><p class="place-sheet-note">${silentCopy()}</p>
        <div class="place-accounts">${connectors.map((connector) => {
            const n = places.filter((place) => place.connector === connector.connector).length;
            return `<div class="harness-account-row place-connector-row">
                <div class="harness-account-main"><span class="place-glyph" aria-hidden="true">${skillIconHtml(connector.icon, 18)}</span><strong data-i18n-authored>${escapeHtml(connector.label)}</strong></div>
                ${connector.readiness ? `<div class="harness-account-meta" data-tone="${escapeHtml(connector.readiness.tone)}">${escapeHtml(tr(`places.readiness.${connector.readiness.text}`, connector.readiness.text))}</div>` : ''}
                <div class="harness-account-actions">${n ? `<span>${escapeHtml(fmt(n === 1 ? 'Connected · {n} place ·' : 'Connected · {n} places ·', { n }))}</span>` : ''}
                    ${button('choose', n ? tr('places.connect.another', 'Connect another') : tr('places.connect.action', 'Connect'), `data-connector="${escapeHtml(connector.connector)}"`)}</div>
            </div>`;
        }).join('')}</div></section>`;
}

/** The connector's declared settings live in Settings › Advanced; fall back to its Skills entry. */
export async function openConnectorSettings(connector, openSettingsTab) {
    if (!connector.settings || !openSettingsTab) { openSkillsFor({ skill: connector.connector }); return; }
    if (await openSettingsTab('advanced') === false) return;
    const section = document.getElementById('extension-settings-sections');
    const form = [...(section?.querySelectorAll('[data-extension-settings-form]') || [])]
        .find((node) => node.dataset.skill === connector.connector && node.dataset.extensionSettingsKey?.startsWith(`${connector.connector}:${connector.settings}:`));
    (form || section)?.scrollIntoView({ block: 'start' });
}

export function openConnectPlace({ returnFocus = document.activeElement, read = readPlaces, connect = connectPlace,
    readStatus = readPlaceConnector, openPlace = openPlaceSheet, openSettingsTab = null, connector = '' } = {}) {
    const dialog = sheetShell(escapeHtml(tr('places.connect', 'Connect a place')), tr('places.connect.from', 'From the connectors installed here'), 'place-connect');
    const abort = new AbortController();
    const sheet = showSheet(dialog, { returnFocus, onClose: () => abort.abort() });
    const view = { connectors: null, places: [], error: '', status: {}, selected: null, choices: null, busy: false };
    let generation = 0;
    const paint = () => { if (!sheet.closed) sheet.body.innerHTML = connectListHtml(view); };
    const load = async () => {
        view.error = ''; view.connectors = null; paint();
        try { Object.assign(view, await read({ signal: abort.signal })); }
        catch (error) { view.error = fmt('Could not read the connectors: {reason}', { reason: error?.message || error }); }
        // A connector's own card opens the flow filtered to that connector.
        if (connector && view.connectors) view.connectors = view.connectors.filter((row) => row.connector === connector);
        paint();
    };
    const loadChoices = async () => {
        const current = ++generation;
        view.choices = null; view.connectorStatus = null; paint();
        if (!view.selected.source) return;
        try {
            const data = await readStatus(view.selected, { signal: abort.signal });
            if (current !== generation || sheet.closed) return;
            view.choices = connectChoices(data);
            view.connectorStatus = typeof data.text === 'string' ? { text: data.text, tone: data.tone } : null;
        } catch { if (current === generation) view.choices = { state: 'unavailable', rows: [] }; }
        paint();
    };
    sheet.body.addEventListener('click', async (event) => {
        const control = event.target.closest('[data-connect-act]');
        if (!control || control.disabled || view.busy) return;
        const act = control.dataset.connectAct;
        if (act === 'retry') { await load(); return; }
        if (act === 'browse') { sheet.close(); openSkillsFor({ category: 'connectors' }); return; }
        if (act === 'back') { ++generation; view.selected = null; view.status = {}; paint(); return; }
        if (act === 'settings') { sheet.close(); await openConnectorSettings(view.selected, openSettingsTab); return; }
        if (act === 'choose') {
            view.selected = view.connectors.find((item) => item.connector === control.dataset.connector);
            if (view.selected) await loadChoices();
            return;
        }
        if (act === 'retry-choices') { await loadChoices(); return; }
        if (act === 'open') {
            const place = view.places.find((item) => item.id === control.dataset.placeId);
            if (place) { sheet.close(); openPlace(place, { returnFocus }); }
            return;
        }
        if (act !== 'connect' || !view.selected) return;
        const choice = control.dataset.choice === 'whole' ? {} : view.choices?.rows[Number(control.dataset.choice)];
        if (!choice) return;
        view.busy = true; view.status = { text: tr('places.connect.busy', 'Connecting…'), tone: 'muted' }; paint();
        try {
            const place = await connect({ connector: view.selected.connector, ...choice });
            // The list projection joins the declared service name later; the flow already holds it.
            if (!sheet.closed) { sheet.close(); openPlace({ ...place, serviceLabel: place.serviceLabel || view.selected.label }, { returnFocus }); }
        } catch (error) { view.status = { text: error?.message || String(error), tone: 'error' }; }
        finally { view.busy = false; paint(); }
    });
    void load();
    return sheet;
}
