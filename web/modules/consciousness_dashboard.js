// Dashboard › Consciousness: what Ouroboros does when nobody asked, now and lately. The status with Stop
// (the same /bg command the Activity row and the chat header send), where its money went by cause (each place's
// included), the own tasks MAX_TASKS counts, its settings card (rows of settings_rows.js that save themselves) and the
// recent wakes with their cause, where, how fast it answered, what the Safety Supervisor did and what each cost. The
// places themselves live on the Circles page; this tab reads `/api/places` only to open a wake's captured conversation
// in its place sheet. Facts come from `GET /api/consciousness`, with `/api/state` as the status fallback. Wakes say
// when they are not reported; unknown money is never zero.
import { escapeHtmlAttr as escapeHtml } from './ui_primitives.js';
import { statusMarkHtml } from './status_mark.js';
import { controlIcon } from './page_icons.js';
import { chipHtml } from './skill_presentation.js';
import { tr, fmt } from './i18n.js';
import { failureText, readConsciousness, readConsciousnessFromState, readPlaces } from './consciousness_api.js';
import { autonomyOf } from './consciousness_autonomy.js';
import { CHECK_CHIPS, allowanceText, clockText, consciousnessMark, safetyLevelOf, splitWakes, durationText, sendsAsHtml, usdText, wakePhaseText, wakeFactsText, wakeCheckChip, wakeCheckText, wakeCostText } from './consciousness_words.js';
import { mountChatMarkdown, enhanceChatMarkdown, destroyChatMarkdown } from './chat_markdown.js';
import { confirmAndSendRestart, costPresentationDetail } from './chat_activity.js';
import { openConfirmDialog } from './confirm_dialog.js';
import { openCircles, openPlaceSheet } from './place_sheet.js';
import { openTaskReference, taskReferenceHtml } from './task_reference.js';
import { CODECS, GUARDED_WRITERS, WRITERS } from './settings_controls.js';
import { mountTabSettings } from './dashboard.js';
import { parseModelSource } from './route_editor_primitives.js';
import { whenName } from './evolution.js';

const SEP = ' · ';
const REFRESH_MS = 20000;
// A command waits this long for the runtime before the state is read again (the chip's Shade keeps the same beat).
const COMMAND_SETTLE_MS = 400;
// Its own time: the house brain glyph, beside the words that say it.
const OWN_GLYPH = controlIcon('brain', 14);
// Work running now: the house status mark's moving dots, in the surface's own ink.
const dots = () => statusMarkHtml('consciousness-mark', { shape: 'moving', hue: 'ink' });

// The wake range is refused before anything is sent when its shortest wait is above its longest: the server would take
// it and wake at the shortest (runtime_limits.get_bg_wakeup_max_sec).
WRITERS.wakeRange = (row, value, options) => (Number(value?.[0]) > Number(value?.[1])
    ? { state: 'refused', effect: '', withheld: '', message: 'The shortest wait is above the longest.', code: 'invalid_value' }
    : WRITERS.generic(row, value, options));
GUARDED_WRITERS.add('wakeRange');  // Undo's token and unset keys go to the generic write with the rest

/** Seconds as the words a person reads: minutes below an hour, hours from there. */
export function waitWords(seconds) {
    const value = Number(seconds);
    if (!Number.isFinite(value) || value <= 0) return '';
    const [amount, unit] = value < 3600 ? [+(value / 60).toFixed(1), 'minute'] : [+(value / 3600).toFixed(1), 'hour'];
    return `${amount} ${unit}${amount === 1 ? '' : 's'}`;
}

/** What the settings cards say from the last settings read besides their rows (dashboard.js `words`): the chosen level
 *  in its own sentence, the wake range in minutes or hours, and the values behind the doors: the allowance, the model,
 *  the Safety level of its own life in the scale's words (consciousness_words.js SAFETY_LEVELS) and the evolution
 *  choice by its name. A key the document does not carry is not said. */
export function cardWords(settings) {
    const [min, max] = [waitWords(settings.OUROBOROS_BG_WAKEUP_MIN), waitWords(settings.OUROBOROS_BG_WAKEUP_MAX)];
    const model = String(settings.OUROBOROS_MODEL_CONSCIOUSNESS ?? '').trim();
    const stored = settings.OUROBOROS_CONSCIOUSNESS_DAILY_USD;
    const allowance = stored === undefined || stored === null || stored === '' ? NaN : Number(stored);
    return {
        OUROBOROS_CONSCIOUSNESS_AUTONOMY: { caption: autonomyOf(settings.OUROBOROS_CONSCIOUSNESS_AUTONOMY)?.note || '' },
        OUROBOROS_BG_WAKEUP: { caption: !min || !max ? '' : min === max ? `It wakes every ${min}.` : `It picks a moment between ${min} and ${max}.` },
        // The row names the limit as Costs does (a rolling 24 hours, never a calendar day), so its value is the sum alone.
        'consciousness-allowance': { value: !Number.isFinite(allowance) ? '' : Number.isInteger(allowance) ? `$${allowance}` : usdText(allowance) },
        'consciousness-model': { value: model ? parseModelSource(model).model || model : 'Main model' },
        'consciousness-safety': { value: safetyLevelOf(settings.OUROBOROS_CONSCIOUSNESS_SAFETY_MODE)?.label || '' },
        'consciousness-evolution': { value: 'OUROBOROS_POST_TASK_EVOLUTION' in settings ? whenName(CODECS.evolutionWhen.read(settings)) : '' },
    };
}

/** Active rows are explicitly marked by the producer; their phase may still be waiting. */
export function liveWake(snapshot) {
    return (snapshot.recentWakes || []).find((wake) => wake.live) || null;
}

/** The alarm clock owns the headline, independently of the recent-wake list; `note` only when the host says what
 *  follows from its state (a held wake's allowance). */
export function headlineView(snapshot) {
    return { text: snapshot.detail ? tr(snapshot.detail, snapshot.detail) : tr('task.phase.unknown', 'State unknown'),
        ...(snapshot.detailNote ? { note: tr(snapshot.detailNote, snapshot.detailNote) } : {}), moving: snapshot.status === 'thinking' };
}

const causeWord = (row) => (row.cause === 'own' ? tr('task.title.own_time', 'Own time') : row.cause === 'started' ? tr('consciousness.started_work', 'Started work') : row.label || row.cause.replace(/^place:/, ''));

/** Where the money of the rolling 24 hours went, for the headline card's Details: each cause that spent something,
 *  every place included; and whether some calls have no price. */
export function spendDetails(snapshot) {
    const rows = snapshot.spendByCause.filter((row) => row.usd !== null && row.usd > 0)
        .map((row) => ({ words: causeWord(row), amount: usdText(row.usd) }));
    const unpriced = snapshot.spendByCause.some((row) => row.unmeteredCalls > 0);
    return { rows, unpriced };
}

/** The own tasks MAX_TASKS counts, each with its cause; work the owner asked for never counts. */
export function countedRootsParts(snapshot) {
    if (snapshot.maxTasks === null && !snapshot.countedRoots.length) return null;
    const running = snapshot.countedRoots.length || snapshot.tasksRunning || 0;
    const head = snapshot.maxTasks !== null ? fmt('Its own tasks: {running} of {max}', { running, max: snapshot.maxTasks })
        : fmt('Its own tasks: {running}', { running });
    const names = snapshot.countedRoots.map((root) => `${root.title || root.taskId}${root.causeLabel ? ` (${root.causeLabel})` : ''}`);
    return { head, names };
}

export function countedRootsText(snapshot) {
    const parts = countedRootsParts(snapshot);
    return parts ? [parts.head, ...parts.names].join(' · ') : '';
}

/** The tab's headline card on the Dashboard tab body: the state in the host's words with its one status mark, the money
 *  of the last 24 hours and the own-task count as its meta, Start/Stop (and Restart when it cannot wake) beside them,
 *  and while it is on, that Stop also ends a running wake; where the money went, the unpriced caveat and the own tasks'
 *  names are the card's Details. Its doors (Costs, Models, Circles) are rows of the settings cards under it.
 *  `toggling` is the command just sent (its button says so until the next read). */
export function summaryHtml(snapshot, { hasSocket = true, toggling = '', detailsOpen = false } = {}) {
    const headline = headlineView(snapshot);
    const kicker = snapshot.enabled === null ? 'Consciousness · state unknown' : `Consciousness · ${snapshot.enabled ? 'on' : 'off'}`;
    const money = allowanceText(snapshot);
    const roots = countedRootsParts(snapshot);
    const mark = consciousnessMark(snapshot);
    const markHtml = mark.shape === 'none' ? '' : statusMarkHtml('consciousness-mark', mark);
    const busy = toggling ? (toggling === 'stop' ? 'Stopping…' : 'Starting…') : '';
    const toggle = snapshot.enabled === null ? '' : `<button type="button" class="btn btn-default btn-sm" data-cx-act="toggle" data-enabled="${snapshot.enabled ? '1' : '0'}"${hasSocket && !busy ? '' : ' disabled'}>${busy || (snapshot.enabled ? 'Stop' : 'Start')}</button>`;
    // It cannot wake until Ouroboros starts again: the next step is beside the words that say so.
    const restart = snapshot.enabled && snapshot.status === 'stopped'
        ? `<button type="button" class="btn btn-default btn-sm" data-cx-act="restart"${hasSocket ? '' : ' disabled'}>Restart</button>` : '';
    const spend = spendDetails(snapshot);
    const details = [
        spend.rows.length ? `<dl class="dashboard-facts">${spend.rows.map((row) => `<dt data-i18n-authored>${escapeHtml(row.words)}</dt><dd>${escapeHtml(row.amount)}</dd>`).join('')}</dl>` : '',
        spend.unpriced ? `<p class="consciousness-foot">${escapeHtml(tr('consciousness.unpriced', 'Some calls have no price'))}.</p>` : '',
        roots?.names.length ? `<p class="consciousness-foot" data-i18n-authored>${escapeHtml(roots.names.join(' · '))}</p>` : '',
    ].join('');
    return `<div class="dashboard-hero">
            <p class="dashboard-hero-kicker">${escapeHtml(kicker)}</p>
            <p class="dashboard-hero-figure consciousness-headline">${escapeHtml(headline.text)}${markHtml}</p>
            ${headline.note ? `<p class="dashboard-hero-meta consciousness-headline-note">${escapeHtml(headline.note)}</p>` : ''}
            ${money ? `<p class="dashboard-hero-meta">${escapeHtml(money)}</p>` : ''}
            ${roots ? `<p class="dashboard-hero-meta">${escapeHtml(roots.head)}</p>` : ''}
            ${snapshot.enabled ? '<p class="dashboard-hero-meta">Stop also ends a wake that is running now.</p>' : ''}
            ${toggle || restart ? `<div class="dashboard-hero-actions">${restart}${toggle}</div>` : ''}
        </div>
        ${details ? `<details class="ui-disclosure consciousness-details"${detailsOpen ? ' open' : ''}><summary>Details</summary>${details}</details>` : ''}`;
}

/** One view of the card: authored line, whole note on request, factual phase and cost. */
export function wakeRowHtml(wake, { open = false } = {}) {
    const working = wake.phase === 'running' || wake.phase === 'working';
    const where = wake.cause === 'focus'
        ? `<span class="consciousness-wake-where" data-i18n-authored>${escapeHtml(wake.placeLabel)}</span>`
        : `<span class="consciousness-wake-own"><span class="place-glyph" aria-hidden="true">${OWN_GLYPH}</span>${escapeHtml(tr('task.title.own_time', 'Own time'))}</span>`;
    const duration = durationText(wake.startedAt, wake.endedAt);
    const sender = sendsAsHtml(wake.as);
    const counts = escapeHtml(wakeFactsText(wake));
    const checks = wake.checks.map(wakeCheckText).filter(Boolean);
    const flags = wake.checks.filter((check) => CHECK_CHIPS[check.chip]?.tone === 'warn')
        .map((check) => chipHtml(wakeCheckChip(check), 'warn')).join(' ');
    const cost = wakeCostText(wake);
    const note = wake.note ? `<p class="consciousness-wake-note-label">${escapeHtml(tr('consciousness.own_note', 'Its own note'))}</p>
        <div class="consciousness-wake-note ui-rich-content" data-i18n-authored>${escapeHtml(wake.note)}</div>` : '';
    const costDetail = costPresentationDetail(wake.cost.cost_presentation, wake.cost);
    // Main cards open in Main; a focus opens its captured conversation in the place sheet.
    const door = wake.chatId === 1 && wake.taskId ? taskReferenceHtml(wake.taskId)
        : wake.cause === 'focus' && wake.chatId ? `<button type="button" class="consciousness-inline-link ui-hit-area" data-cx-open-room="${wake.chatId}">${escapeHtml(tr('consciousness.open_room', 'Open the conversation'))} ↗</button>` : '';
    const line = wake.summary ? `<p class="consciousness-wake-summary" data-i18n-authored>${escapeHtml(wake.summary)}</p>`
        : wake.note ? `<p class="consciousness-wake-lead" data-i18n-authored>${escapeHtml(String(wake.note ?? '').trim())}</p>`
            : counts ? `<p class="consciousness-wake-facts is-line">${counts}</p>` : '';
    return `<li class="consciousness-wake${wake.live ? ' is-live' : ''}" data-wake="${escapeHtml(wake.taskId)}">
        <time class="consciousness-wake-time"${wake.startedAt ? ` datetime="${escapeHtml(wake.startedAt)}"` : ''}>${escapeHtml(clockText(wake.startedAt) || '—')}</time>
        <div class="consciousness-wake-main">
            <div class="consciousness-wake-head">${where}${sender ? SEP + sender : ''}${SEP}${escapeHtml(wakePhaseText(wake))}${working ? dots() : ''}${duration ? SEP + escapeHtml(duration) : ''}</div>
            ${line}${flags ? `<div class="consciousness-wake-checks">${flags}</div>` : ''}
            <details class="ui-disclosure consciousness-wake-more"${open ? ' open' : ''}><summary>${escapeHtml(tr('consciousness.details', 'Details'))}</summary>
                ${note}${counts ? `<p class="consciousness-wake-facts">${counts}</p>` : ''}
                ${checks.map((words) => `<p class="consciousness-wake-facts">${escapeHtml(words)}</p>`).join('')}
                ${costDetail ? `<p class="consciousness-wake-facts">${escapeHtml(tr(costDetail, costDetail))}</p>` : ''}
                ${cost ? '' : `<p class="consciousness-wake-facts">${escapeHtml(tr('consciousness.cost_unknown', 'Cost unknown'))}</p>`}${door}
            </details>
        </div>
        <span class="consciousness-wake-cost">${escapeHtml(cost)}</span>
    </li>`;
}

// A read that failed says so first, where the lists would be (the card owns the warning); the lists' scope is their
// caption, after them.
const wakesNoticeHtml = (wakes, coverage) => Array.isArray(wakes) ? ''
    : coverage ? `<p class="consciousness-foot ui-status" role="status" data-tone="warn" data-i18n-authored>${escapeHtml(coverage)}</p>`
        : `<p class="consciousness-copy">${escapeHtml(tr('consciousness.wakes_unavailable', 'This server does not report its recent wakes yet.'))}</p>`;
const wakesCaptionHtml = (wakes, coverage) => Array.isArray(wakes) && coverage
    ? `<p class="consciousness-foot" data-i18n-authored>${escapeHtml(coverage)}</p>` : '';

const wakeGroupTitle = (group) => group === 'places' ? tr('consciousness.place_wakes', 'Wakes in places')
    : tr('consciousness.own_wakes', 'Its own wakes');
const wakeGroupEmpty = (group) => group === 'places'
    ? tr('consciousness.no_place_wakes', 'No wakes in places in the last 48 h.')
    : tr('consciousness.no_own_wakes', 'No own wakes with a recorded model call in the last 48 h.');

export function wakesCardHtml(wakes, coverage = '', group = '') {
    if (!group) {
        const split = splitWakes(wakes);
        return `<div data-cx-wake-notice>${wakesNoticeHtml(wakes, coverage)}</div>`
            + ['own', 'places'].map((key) => `<section class="consciousness-wake-group" aria-label="${escapeHtml(wakeGroupTitle(key))}" data-cx-wake-group="${key}">${
            wakesCardHtml(Array.isArray(wakes) ? split[key] : null, coverage, key)}</section>`).join('')
            + `<div data-cx-wake-coverage>${wakesCaptionHtml(wakes, coverage)}</div>`;
    }
    const body = !Array.isArray(wakes) ? ''
        : !wakes.length ? `<p class="consciousness-copy">${escapeHtml(wakeGroupEmpty(group))}</p>`
            : `<ol class="consciousness-wakes">${wakes.map((wake) => wakeRowHtml(wake)).join('')}</ol>`;
    return `<h3 class="dashboard-card-title">${escapeHtml(wakeGroupTitle(group))}</h3>${body}`;
}

/** Keyed rows: an open disclosure is a reading surface, retained until its reader closes it. */
export function paintWakeRows(region, wakes, coverage, opened, put, release = destroyChatMarkdown) {
    const group = region.dataset.cxWakeGroup || 'own';
    const list = region.querySelector('.consciousness-wakes');
    // Native toggle is queued: read the open property too when a refresh wins that race.
    for (const row of list?.children || []) if (row.querySelector('details')?.open) opened.add(row.dataset.wake);
    const scroller = region.closest('.dashboard-body');
    const reading = [...(list?.children || [])].find((row) => opened.has(row.dataset.wake));
    const top = reading?.getBoundingClientRect().top;
    const keepPosition = () => { if (scroller && reading?.isConnected) scroller.scrollTop += reading.getBoundingClientRect().top - top; };
    if (list && (wakes === null && coverage || reading && (!Array.isArray(wakes) || !wakes.length))) { keepPosition(); return; }
    if (!list || !Array.isArray(wakes) || !wakes.length) { put(region, wakesCardHtml(wakes, coverage, group)); return; }
    const rows = new Map([...list.children].map((row) => [row.dataset.wake, row]));
    let before = list.firstElementChild;
    for (const wake of wakes) {
        let row = rows.get(wake.taskId);
        if (!row) {
            const template = region.ownerDocument.createElement('template');
            template.innerHTML = wakeRowHtml(wake);
            row = template.content.firstElementChild;
        } else if (!opened.has(wake.taskId)) {
            const template = region.ownerDocument.createElement('template');
            template.innerHTML = wakeRowHtml(wake);
            put(row, template.content.firstElementChild.innerHTML);
            row.className = template.content.firstElementChild.className;
        }
        // Defer reordering existing rows while reading; moving a focused node drops its selection.
        if (row !== before && (!reading || !row.isConnected)) list.insertBefore(row, before);
        before = row.nextElementSibling;
        rows.delete(wake.taskId);
    }
    for (const [id, row] of rows) if (!opened.has(id)) { release(row); row.remove(); }
    keepPosition();
}

/** The tab's settings (dashboard.js mountTabSettings): the card "On its own" with what it does as rows that save
 *  themselves and its fold, then the doors of "Around it", so what concerns consciousness is reached from here. The
 *  allowance door opens its row in Costs and the evolution door its row on the Evolution tab; the model door opens its
 *  role in Models and the Safety door its row in Permissions and checks, both leading back here; its places are the
 *  Circles page. */
function mountSettingsCards(host, { state, openSettings, openDashboardTab }) {
    if (!host) return { dispose() {} };
    const back = { label: 'Consciousness', go: () => openDashboardTab?.('consciousness') };
    const doors = {
        'consciousness-allowance': () => openDashboardTab?.('costs', { row: 'OUROBOROS_CONSCIOUSNESS_DAILY_USD' }),
        'consciousness-model': () => openSettings?.('models/role/consciousness', { back }),
        'consciousness-safety': () => openSettings?.('permissions', { row: 'OUROBOROS_CONSCIOUSNESS_SAFETY_MODE', back }),
        'consciousness-places': () => openCircles(),
        'consciousness-evolution': () => openDashboardTab?.('evolution', { row: 'OUROBOROS_POST_TASK_EVOLUTION' }),
    };
    const cards = mountTabSettings(host, 'consciousness', { state, words: cardWords, openDoor: (row) => doors[row.id]?.(),
        cards: [{ title: 'On its own', fold: 'More: conversations at once' },
            { title: 'Around it', rows: ['consciousness-model', 'consciousness-safety', 'consciousness-places', 'consciousness-evolution'] }] });
    // The range reads "[shortest] to [longest] s".
    const range = Array.from(host.querySelectorAll('[data-setting-row]')).find((node) => node.dataset.settingRow === 'OUROBOROS_BG_WAKEUP');
    const longest = range?.querySelectorAll('[data-setting-control]')[1];
    if (longest) {
        const to = host.ownerDocument.createElement('span');
        to.className = 'settings-row-unit';
        to.textContent = 'to';
        longest.parentElement.insertBefore(to, longest);
    }
    return cards;
}

/** Mount the tab into its Dashboard panel. */
export function initConsciousnessDashboard({ mount, ws, state, openSettings, openDashboardTab, read = readConsciousness,
    readFallback = readConsciousnessFromState, readPlacesFn = readPlaces, openSheet = openPlaceSheet,
    markdown = { mount: mountChatMarkdown, enhance: enhanceChatMarkdown, destroy: destroyChatMarkdown } } = {}) {
    if (!mount) return { refresh: () => {}, destroy() {} };
    // The tab's one root and scroller (style.css "The Dashboard tab body"); the settings cards are drawn once and no
    // repaint of the tab touches them.
    mount.innerHTML = `<div class="dashboard-body scroll-fade-y consciousness-root">
            <p class="ui-status consciousness-read-status" role="status" data-cx-status></p>
            <section class="ui-card" aria-label="Consciousness" data-cx-summary><p class="consciousness-copy">Reading…</p></section>
            <div class="dashboard-settings-cards" data-cx-settings></div>
            <section class="ui-card consciousness-card" aria-label="Recent wakes" data-cx-wakes></section>
        </div>`;
    const regions = {
        status: mount.querySelector('[data-cx-status]'), summary: mount.querySelector('[data-cx-summary]'),
        wakes: mount.querySelector('[data-cx-wakes]'),
    };
    const cards = mountSettingsCards(mount.querySelector('[data-cx-settings]'), { state, openSettings, openDashboardTab });
    // The page binds every `.scroll-fade-y` scroller once, after the panels mount (app.js).
    let snapshot = null;
    let generation = 0;
    let timer = 0;
    let destroyed = false;
    const shown = () => !document.hidden && state?.activePage === 'dashboard' && state?.dashboardActiveSubtab === 'consciousness';

    // A refresh that found nothing new leaves a region's nodes alone, so focus and reading position stay.
    const opened = new Set();
    const notes = new Map();
    function releaseNotes(region) {
        for (const [note, text] of notes) if (region.contains(note)) {
            markdown.destroy(note);
            note.textContent = text;
            notes.delete(note);
        }
    }
    const painted = new WeakMap();
    const put = (region, html) => {
        if (painted.get(region) === html || region.innerHTML === html) return;
        painted.set(region, html);
        releaseNotes(region);
        region.innerHTML = html;
    };
    // The Start/Stop just sent: its button says so until a read made after the runtime's beat paints the state it reached
    // (the chip's Shade keeps the same rule, consciousness_status.js).
    let toggling = '';
    let toggledAt = 0;
    let detailsOpen = false;  // the headline card's Details stays as its reader left it across repaints
    function paint() {
        if (snapshot) put(regions.summary, summaryHtml(snapshot, { hasSocket: Boolean(ws?.send), toggling, detailsOpen }));
        const wakes = snapshot?.recentWakes ?? null;
        const coverage = snapshot?.coverage || '';
        if (!regions.wakes.querySelector('[data-cx-wake-group]')) { put(regions.wakes, wakesCardHtml(wakes, coverage)); return; }
        put(regions.wakes.querySelector('[data-cx-wake-notice]'), wakesNoticeHtml(wakes, coverage));
        put(regions.wakes.querySelector('[data-cx-wake-coverage]'), wakesCaptionHtml(wakes, coverage));
        const split = splitWakes(wakes);
        for (const region of regions.wakes.querySelectorAll('[data-cx-wake-group]')) {
            paintWakeRows(region, wakes === null ? null : split[region.dataset.cxWakeGroup], coverage, opened, put, releaseNotes);
        }
    }

    async function refresh() {
        if (destroyed) return;
        const current = ++generation;
        const earlier = Boolean(snapshot);
        regions.status.textContent = earlier ? '' : 'Reading…';
        let next = null;
        let readError = '';
        try { next = await read(); } catch (error) {
            // A server without the panel (404) is read from its state; any other failure keeps the painted rows and says so.
            const served = error?.status !== 404;
            if (served) readError = failureText(error);
            try {
                const state = await readFallback();
                next = served ? { ...state, recentWakes: snapshot?.recentWakes ?? null,
                    coverage: snapshot?.recentWakes ? snapshot.coverage : fmt('Recent wakes could not be read: {error}', { error: readError }) } : state;
            } catch (fallbackError) { readError = failureText(fallbackError); }
        }
        if (destroyed || current !== generation) return;
        if (next) snapshot = next;
        if (toggling && Date.now() - toggledAt >= COMMAND_SETTLE_MS) toggling = '';
        // One sentence per fact, each ending once: a reason that already ends its sentence gets no second full stop.
        const reason = /[.!?]$/.test(readError) ? readError : `${readError}.`;
        regions.status.textContent = readError ? `Could not read consciousness: ${reason}${earlier ? ' Earlier values are shown.' : ''}` : '';
        regions.status.dataset.tone = readError ? 'error' : '';
        paint();
        schedule();
    }

    function schedule() {
        clearTimeout(timer);
        if (!destroyed && shown()) timer = setTimeout(refresh, REFRESH_MS);
    }

    const onToggle = (event) => {
        if (event.target.matches?.('.consciousness-details')) { detailsOpen = event.target.open; return; }
        if (!event.target.matches?.('.consciousness-wake-more')) return;
        const id = event.target.closest('[data-wake]').dataset.wake;
        if (event.target.open) {
            opened.add(id);
            const note = event.target.querySelector('.consciousness-wake-note');
            if (note && !notes.has(note)) {
                notes.set(note, note.textContent);
                markdown.mount(note, note.textContent);
                if (note.querySelector('.md-code-block, .md-mermaid, .md-chart, .md-table-wrap')) markdown.enhance(note);
            }
        } else { releaseNotes(event.target); opened.delete(id); paint(); }
    };
    const onClick = async (event) => {
        if (openTaskReference(event)) return;
        const roomDoor = event.target.closest('[data-cx-open-room]');
        if (roomDoor) {
            if (roomDoor.dataset.pending) return;  // one press at a time while the places are read
            // The places are read at the press: the conversation's place as it stands now, never an older list's.
            const chatId = Number(roomDoor.dataset.cxOpenRoom);
            roomDoor.dataset.pending = '1';
            let place = null, failed = '';
            try { place = (await readPlacesFn()).places.find((entry) => entry.rooms.some((room) => room.chatId === chatId)) || null; }
            catch (error) { failed = fmt('Places could not be read: {reason}', { reason: failureText(error) }); }
            finally { delete roomDoor.dataset.pending; }
            if (destroyed) return;
            if (failed) {  // the tab's status says so; the door stays for another try
                regions.status.textContent = failed;
                regions.status.dataset.tone = 'error';
                return;
            }
            if (place) openSheet(place, { returnFocus: roomDoor, openRoomChatId: chatId });
            else if (roomDoor.isConnected !== false) { roomDoor.disabled = true; roomDoor.textContent = tr('consciousness.room_unavailable', 'Conversation unavailable'); }
            return;
        }
        const button = event.target.closest('[data-cx-act]');
        if (!button || button.disabled) return;
        const act = button.dataset.cxAct;
        if (act === 'restart') {
            // The house Restart (the chat header's): it confirms first and names the work it would interrupt.
            await confirmAndSendRestart({ openConfirmDialog, ws });
            return;
        }
        if (act === 'toggle') {
            // The existing direct control command (the Activity row and the chat header send the same); the button says
            // it is on its way until the read after the runtime's beat paints the state it reached.
            toggling = button.dataset.enabled === '1' ? 'stop' : 'start';
            toggledAt = Date.now();
            paint();
            ws?.send?.({ type: 'command', cmd: `/bg ${toggling}`, ui_control: { name: 'consciousness_toggle' } });
            await new Promise((resolve) => setTimeout(resolve, COMMAND_SETTLE_MS));
            await refresh();
        }
    };
    // Shown by its subtab or by returning to the Dashboard on it.
    const onShown = (event) => {
        if (event?.detail?.tab === 'consciousness' || (event?.detail?.page === 'dashboard' && shown())) void refresh();
        else clearTimeout(timer);
    };
    mount.addEventListener('click', onClick);
    mount.addEventListener('toggle', onToggle, true);
    window.addEventListener('ouro:dashboard-subtab-shown', onShown);
    window.addEventListener('ouro:page-shown', onShown);
    // The card's frame at once, before any read: the wake rows then always arrive through the keyed per-group painter.
    paint();
    if (shown()) void refresh();
    return {
        refresh,
        destroy() {
            destroyed = true;
            cards.dispose();
            releaseNotes(mount);
            clearTimeout(timer);
            mount.removeEventListener('click', onClick);
            mount.removeEventListener('toggle', onToggle, true);
            window.removeEventListener('ouro:dashboard-subtab-shown', onShown);
            window.removeEventListener('ouro:page-shown', onShown);
        },
    };
}
