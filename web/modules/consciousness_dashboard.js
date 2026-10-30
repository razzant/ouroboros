// Dashboard › Consciousness: what Ouroboros does when nobody asked, now and lately. The status with Stop
// (the same /bg command the Activity row and the chat header send), where its money went by cause, the own
// tasks MAX_TASKS counts, its places (each opens its sheet; Connect a place) and the recent wakes with their
// cause, where, how fast it answered, what the Safety Supervisor did and what each cost. Facts come from
// `GET /api/consciousness`, with `/api/state` as the status fallback. Places always read `/api/places`.
// Wakes say when they are not reported; unknown money is never zero.
import { escapeHtmlAttr as escapeHtml } from './ui_primitives.js';
import { statusMarkHtml } from './status_mark.js';
import { controlIcon } from './page_icons.js';
import { chipHtml, skillIconHtml } from './skill_presentation.js';
import { allowanceLabel } from './utils.js';
import { tr, fmt } from './i18n.js';
import { PLACES_CHANGED_EVENT, readConsciousness, readConsciousnessFromState, readPlaces } from './consciousness_api.js';
import { getContributions } from './contributions.js';
import { CHECK_CHIPS, clockText, splitWakes, durationText, noteLeadText, placeStatusText, sendsAsHtml, usdText, wakePhaseText, wakeFactsText, wakeCheckText, wakeCostText } from './consciousness_words.js';
import { mountChatMarkdown, enhanceChatMarkdown, destroyChatMarkdown } from './chat_markdown.js';
import { costPresentationDetail } from './chat_activity.js';
import { openPlaceSheet } from './place_sheet.js';
import { connectPlaceButtonHtml, openConnectPlace } from './place_connect.js';
import { openTaskReference, taskReferenceHtml } from './task_reference.js';

const SEP = ' · ';
const REFRESH_MS = 20000;
// Its own time: the house brain glyph, beside the words that say it.
const OWN_GLYPH = controlIcon('brain', 14);
// Work running now: the house status mark's moving dots, in the surface's own ink.
const dots = () => statusMarkHtml('consciousness-mark', { shape: 'moving', hue: 'ink' });
const facts = (parts) => parts.filter(Boolean).map((part) => `<span class="consciousness-fact">${part}</span>`).join(SEP);

/** Active rows are explicitly marked by the producer; their phase may still be waiting. */
export function liveWake(snapshot) {
    return (snapshot.recentWakes || []).find((wake) => wake.live) || null;
}

/** The alarm clock owns the headline, independently of the recent-wake list. */
export function headlineView(snapshot) {
    return { text: snapshot.detail ? tr(snapshot.detail, snapshot.detail) : tr('task.phase.unknown', 'State unknown'),
        moving: snapshot.status === 'thinking' };
}

const causeWord = (row) => (row.cause === 'own' ? tr('consciousness.own', 'own') : row.cause === 'started' ? tr('consciousness.started', 'started work') : row.label || row.cause.replace(/^place:/, ''));

/** «on its own clock · 3 places · own $0.62 · Slack · Acme $0.88 · started work $0.48 · some calls have no price» (the rolling 24 hours). */
export function factsLineParts(snapshot, placeCount = null) {
    const parts = [snapshot.enabled === false ? tr('consciousness.off', 'off') : snapshot.enabled === null ? '' : tr('consciousness.clock', 'on its own clock')];
    if (Number.isInteger(placeCount)) parts.push(fmt(placeCount === 1 ? '{n} place' : '{n} places', { n: placeCount }));
    const spend = snapshot.spendByCause.filter((row) => row.usd !== null);
    spend.forEach((row) => parts.push(`<span data-i18n-authored>${escapeHtml(causeWord(row))}</span> <span class="consciousness-num">${escapeHtml(usdText(row.usd))}</span>`));
    const unmetered = snapshot.spendByCause.reduce((sum, row) => sum + row.unmeteredCalls, 0);
    if (unmetered > 0) parts.push(tr('consciousness.unpriced', 'Some calls have no price'));
    return parts.filter(Boolean);
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

export function summaryHtml(snapshot, { placeCount = null, hasSocket = true } = {}) {
    const headline = headlineView(snapshot);
    const kicker = snapshot.enabled === null ? 'Consciousness · state unknown' : `Consciousness · ${snapshot.enabled ? 'on' : 'off'}`;
    const money = allowanceLabel(snapshot.spentUsd, snapshot.dailyUsd, snapshot.unknownUnmetered, snapshot.integrityDegraded);
    const roots = countedRootsParts(snapshot);
    const toggle = snapshot.enabled === null ? '' : `<button type="button" class="btn btn-ghost btn-sm consciousness-stop" data-cx-act="toggle" data-enabled="${snapshot.enabled ? '1' : '0'}"${hasSocket ? '' : ' disabled'}>${snapshot.enabled ? 'Stop' : 'Start'}</button>`;
    return `<div class="storage-summary-head">
            <div class="storage-summary-text consciousness-summary-text">
                <div class="storage-kicker">${escapeHtml(kicker)}</div>
                <div class="consciousness-headline"><div class="storage-total">${escapeHtml(headline.text)}${headline.moving ? dots() : ''}</div>${toggle}</div>
                ${roots ? `<p class="consciousness-next">${escapeHtml(roots.head)}${roots.names.map((name) => ` · <span data-i18n-authored>${escapeHtml(name)}</span>`).join('')}</p>` : ''}
            </div>
        </div>
        <div class="consciousness-summary-foot">
            ${money ? `<span class="storage-meta consciousness-total">${escapeHtml(money)} in the last 24 hours</span>` : ''}
            <span class="consciousness-causes">${facts(factsLineParts(snapshot, placeCount))}</span>
        </div>`;
}

export function dashboardPlaceRowHtml(place, { awake = false } = {}) {
    const unknown = place.spent24hUsd == null;
    const spend = unknown ? tr('places.money.unknown', 'unknown') : usdText(place.spent24hUsd);
    const spendWords = unknown ? tr('places.money.aria', 'spend unknown') : fmt('{spend} spent', { spend });
    const readiness = place.readiness ? tr(`places.readiness.${place.readiness.text}`, place.readiness.text) : '';
    return `<li><button type="button" class="consciousness-dash-place" data-place-open="${escapeHtml(place.id)}"
            aria-label="${escapeHtml(fmt('Open {place}{awake}: {status}; {spend} in 24 hours', { place: place.label,
                awake: awake ? tr('places.awake', ', awake here now') : '', status: [placeStatusText(place), readiness].filter(Boolean).join('; '), spend: spendWords }))}">
        <span class="consciousness-dash-place-main"><span class="place-glyph" aria-hidden="true">${skillIconHtml(place.icon, 18)}</span><span class="consciousness-place-name" data-i18n-authored>${escapeHtml(place.label)}</span>${awake ? dots() : ''}</span>
        <span class="consciousness-dash-place-meta">${escapeHtml(placeStatusText(place))}${readiness ? `${SEP}<span data-tone="${escapeHtml(place.readiness.tone)}">${escapeHtml(readiness)}</span>` : ''}</span>
        <span class="consciousness-spend${unknown ? ' is-none' : ''}">${escapeHtml(spend)}</span>
    </button></li>`;
}

export function placesCardHtml(places, { error = '', awakeIn = '' } = {}) {
    const body = error ? `<p class="ui-status" data-tone="warn">${escapeHtml(error)}</p>`
        : !places ? '<p class="storage-meta">Reading places…</p>'
            : !places.length ? '<p class="consciousness-copy">No place yet. Connect one from an installed connector.</p>'
                : `<ul class="consciousness-dash-places">${places.map((place) => dashboardPlaceRowHtml(place, { awake: place.id === awakeIn })).join('')}</ul>`;
    return `<div class="consciousness-card-head">
            <div class="consciousness-card-head-text">
                <h4 class="storage-card-title">Places</h4>
                <p class="consciousness-copy">A connector you install can connect a place here: Connect a place.</p>
            </div>
            ${connectPlaceButtonHtml()}
        </div>
        ${body}
        <p class="consciousness-foot">Allowance, own tasks and concurrent conversations: <button type="button" class="consciousness-inline-link ui-hit-area" data-cx-act="settings">Settings ›</button></p>`;
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
        .map((check) => chipHtml(wakeCheckText(check), 'warn')).join(' ');
    const note = wake.note ? `<p class="consciousness-wake-note-label">${escapeHtml(tr('consciousness.own_note', 'Its own note'))}</p>
        <div class="consciousness-wake-note ui-rich-content" data-i18n-authored>${escapeHtml(wake.note)}</div>` : '';
    const costDetail = costPresentationDetail(wake.cost.cost_presentation);
    // Main cards open in Main; a focus opens its captured conversation in the place sheet.
    const door = wake.chatId === 1 && wake.taskId ? taskReferenceHtml(wake.taskId)
        : wake.cause === 'focus' && wake.chatId ? `<button type="button" class="consciousness-inline-link ui-hit-area" data-cx-open-room="${wake.chatId}">${escapeHtml(tr('consciousness.open_room', 'Open the conversation'))} ↗</button>` : '';
    const line = wake.summary ? `<p class="consciousness-wake-summary" data-i18n-authored>${escapeHtml(wake.summary)}</p>`
        : wake.note ? `<p class="consciousness-wake-lead" data-i18n-authored>${escapeHtml(noteLeadText(wake.note))}</p>`
            : counts ? `<p class="consciousness-wake-facts is-line">${counts}</p>` : '';
    return `<li class="consciousness-wake${wake.live ? ' is-live' : ''}" data-wake="${escapeHtml(wake.taskId)}">
        <time class="consciousness-wake-time"${wake.startedAt ? ` datetime="${escapeHtml(wake.startedAt)}"` : ''}>${escapeHtml(clockText(wake.startedAt) || '—')}</time>
        <div class="consciousness-wake-main">
            <div class="consciousness-wake-head">${where}${sender ? SEP + sender : ''}${SEP}${escapeHtml(wakePhaseText(wake))}${working ? dots() : ''}${duration ? SEP + escapeHtml(duration) : ''}</div>
            ${line}${flags ? `<div class="consciousness-wake-checks">${flags}</div>` : ''}
            <details class="consciousness-wake-more"${open ? ' open' : ''}><summary class="ui-hit-area">${escapeHtml(tr('consciousness.more', 'more'))}</summary>
                ${note}${counts ? `<p class="consciousness-wake-facts">${counts}</p>` : ''}
                ${checks.map((words) => `<p class="consciousness-wake-facts">${escapeHtml(words)}</p>`).join('')}
                ${costDetail ? `<p class="consciousness-wake-facts">${escapeHtml(tr(costDetail, costDetail))}</p>` : ''}${door}
            </details>
        </div>
        <span class="consciousness-wake-cost">${escapeHtml(wakeCostText(wake))}</span>
    </li>`;
}

const coverageHtml = (wakes, coverage) => coverage ? `<p class="storage-meta${wakes === null ? ' ui-status' : ''}"${wakes === null ? ' role="status" data-tone="warn"' : ''} data-i18n-authored>${escapeHtml(coverage)}</p>`
    : !Array.isArray(wakes) ? `<p class="consciousness-copy">${escapeHtml(tr('consciousness.wakes_unavailable', 'This server does not report its recent wakes yet.'))}</p>` : '';

const wakeGroupTitle = (group) => group === 'places' ? tr('consciousness.place_wakes', 'Wakes in places')
    : tr('consciousness.own_wakes', 'Its own wakes');
const wakeGroupEmpty = (group) => group === 'places'
    ? tr('consciousness.no_place_wakes', 'No wakes in places in the last 48 h.')
    : tr('consciousness.no_own_wakes', 'No own wakes with a recorded model call in the last 48 h.');

export function wakesCardHtml(wakes, coverage = '', group = '') {
    if (!group) {
        const split = splitWakes(wakes);
        return `<div data-cx-wake-coverage>${coverageHtml(wakes, coverage)}</div>`
            + ['own', 'places'].map((key) => `<section class="consciousness-wake-group" aria-label="${escapeHtml(wakeGroupTitle(key))}" data-cx-wake-group="${key}">${
            wakesCardHtml(Array.isArray(wakes) ? split[key] : null, coverage, key)}</section>`).join('');
    }
    const body = !Array.isArray(wakes) ? ''
        : !wakes.length ? `<p class="consciousness-copy">${escapeHtml(wakeGroupEmpty(group))}</p>`
            : `<ol class="consciousness-wakes">${wakes.map((wake) => wakeRowHtml(wake)).join('')}</ol>`;
    return `<div class="consciousness-card-head"><div class="consciousness-card-head-text">
            <h4 class="storage-card-title">${escapeHtml(wakeGroupTitle(group))}</h4>
        </div></div>${body}`;
}

/** Keyed rows: an open disclosure is a reading surface, retained until its reader closes it. */
export function paintWakeRows(region, wakes, coverage, opened, put, release = destroyChatMarkdown) {
    const group = region.dataset.cxWakeGroup || 'own';
    const list = region.querySelector('.consciousness-wakes');
    // Native toggle is queued: read the open property too when a refresh wins that race.
    for (const row of list?.children || []) if (row.querySelector('details')?.open) opened.add(row.dataset.wake);
    const scroller = region.closest('.costs-scroll');
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

/** Mount the tab into its Dashboard panel. */
export function initConsciousnessDashboard({ mount, ws, state, openSettingsTab, read = readConsciousness,
    readFallback = readConsciousnessFromState, readPlacesFn = readPlaces, openSheet = openPlaceSheet, contributions = getContributions(),
    markdown = { mount: mountChatMarkdown, enhance: enhanceChatMarkdown, destroy: destroyChatMarkdown } } = {}) {
    if (!mount) return { refresh: () => {}, destroy() {} };
    mount.innerHTML = `<div class="settings-embedded-content"><div class="costs-scroll scroll-fade-y">
        <div class="storage-root consciousness-root">
            <p class="ui-status consciousness-read-status" role="status" data-cx-status></p>
            <section class="storage-summary ui-card consciousness-summary" aria-label="Consciousness" data-cx-summary><p class="storage-meta">Reading…</p></section>
            <section class="ui-card consciousness-card" aria-label="Places" data-cx-places></section>
            <section class="ui-card consciousness-card" aria-label="Recent wakes" data-cx-wakes></section>
        </div></div></div>`;
    const regions = {
        status: mount.querySelector('[data-cx-status]'), summary: mount.querySelector('[data-cx-summary]'),
        places: mount.querySelector('[data-cx-places]'), wakes: mount.querySelector('[data-cx-wakes]'),
    };
    // The page binds every `.scroll-fade-y` scroller once, after the panels mount (app.js).
    let snapshot = null;
    let places = null;
    let placesError = '', placesGeneration = 0;
    let placesBehind = false;  // a skill lifecycle ended while hidden: the next show reads Places
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
    function paint() {
        if (snapshot) put(regions.summary, summaryHtml(snapshot, { placeCount: places ? places.length : null, hasSocket: Boolean(ws?.send) }));
        const awake = snapshot ? liveWake(snapshot) : null;
        put(regions.places, placesCardHtml(places, { error: placesError, awakeIn: awake?.cause === 'focus' ? places?.find((place) => place.rooms.some((room) => room.chatId === awake.chatId))?.id : '' }));
        const wakes = snapshot?.recentWakes ?? null;
        const coverage = snapshot?.coverage || '';
        if (!regions.wakes.querySelector('[data-cx-wake-group]')) { put(regions.wakes, wakesCardHtml(wakes, coverage)); return; }
        put(regions.wakes.querySelector('[data-cx-wake-coverage]'), coverageHtml(wakes, coverage));
        const split = splitWakes(wakes);
        for (const region of regions.wakes.querySelectorAll('[data-cx-wake-group]')) {
            paintWakeRows(region, wakes === null ? null : split[region.dataset.cxWakeGroup], coverage, opened, put, releaseNotes);
        }
    }

    // One read in flight at a time; a change, a show or a lifecycle end starts a fresh one over it, and the newest paints.
    let placesPending = null;
    function refreshPlaces({ force = false } = {}) {
        if (placesPending && !force) return placesPending;
        const current = ++placesGeneration;
        placesBehind = false;
        placesPending = (async () => {
            try {
                const data = await readPlacesFn();
                if (destroyed || current !== placesGeneration) return;
                places = data.places; placesError = '';
            } catch (error) {
                if (destroyed || current !== placesGeneration) return;
                placesError = fmt('Places could not be read: {reason}', { reason: error?.message || error });
            }
            paint();
        })().finally(() => { if (current === placesGeneration) placesPending = null; });
        return placesPending;
    }

    async function refresh({ rereadPlaces = false } = {}) {
        if (destroyed) return;
        const current = ++generation;
        // Places are read when the tab is shown, on their change event and at a skill lifecycle end, never by the poll;
        // a failed first read retries.
        const placesRead = places === null || rereadPlaces ? refreshPlaces({ force: rereadPlaces }) : Promise.resolve();
        const earlier = Boolean(snapshot);
        regions.status.textContent = earlier ? '' : 'Reading…';
        let next = null;
        let readError = '';
        try { next = await read(); } catch (error) {
            // A server without the panel (404) is read from its state; any other failure keeps the painted rows and says so.
            const served = error?.status !== 404;
            if (served) readError = error?.message || String(error);
            try {
                const state = await readFallback();
                next = served ? { ...state, recentWakes: snapshot?.recentWakes ?? null,
                    coverage: snapshot?.recentWakes ? snapshot.coverage : fmt('Recent wakes could not be read: {error}', { error: readError }) } : state;
            } catch (fallbackError) { readError = fallbackError?.message || String(fallbackError); }
        }
        if (destroyed || current !== generation) return;
        if (next) snapshot = next;
        regions.status.textContent = readError ? `Could not read consciousness: ${readError}. ${earlier ? 'Earlier values are shown.' : ''}` : '';
        regions.status.dataset.tone = readError ? 'error' : '';
        paint();
        schedule();
        await placesRead;
    }

    function schedule() {
        clearTimeout(timer);
        if (!destroyed && shown()) timer = setTimeout(refresh, REFRESH_MS);
    }

    const onToggle = (event) => {
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
            if (roomDoor.dataset.pending) return;  // one press at a time while Places are re-read
            const chatId = Number(roomDoor.dataset.cxOpenRoom);
            const owner = () => (places || []).find((entry) => entry.rooms.some((room) => room.chatId === chatId));
            let place = owner();
            if (!place) {
                // A conversation captured after the last Places read is not absence: read once more before saying so.
                roomDoor.dataset.pending = '1';
                try { await refreshPlaces({ force: true }); while (placesPending) await placesPending; } finally { delete roomDoor.dataset.pending; }
                place = owner();
                if (!place && placesError) return;  // the read failed: the Places card says so; the door stays
            }
            if (place) openSheet(place, { returnFocus: roomDoor, openRoomChatId: chatId });
            else if (roomDoor.isConnected !== false) { roomDoor.disabled = true; roomDoor.textContent = tr('consciousness.room_unavailable', 'Conversation unavailable'); }
            return;
        }
        const row = event.target.closest('[data-place-open]');
        if (row) {
            const place = (places || []).find((entry) => entry.id === row.dataset.placeOpen);
            if (place) openSheet(place, { returnFocus: row });
            return;
        }
        const connect = event.target.closest('[data-place-connect]');
        if (connect) { openConnectPlace({ returnFocus: connect, openSettingsTab }); return; }
        const button = event.target.closest('[data-cx-act]');
        if (!button || button.disabled) return;
        const act = button.dataset.cxAct;
        if (act === 'settings') {
            await openSettingsTab?.('behavior');
            document.getElementById('settings-consciousness')?.scrollIntoView({ block: 'start' });
            return;
        }
        if (act === 'toggle') {
            button.disabled = true;
            // The existing direct control command (the Activity row and the chat header send the same).
            ws?.send?.({ type: 'command', cmd: `/bg ${button.dataset.enabled === '1' ? 'stop' : 'start'}` });
            await new Promise((resolve) => setTimeout(resolve, 400));
            await refresh();
        }
    };
    // Shown by its subtab or by returning to the Dashboard on it, as Settings re-reads its places when shown.
    const onShown = (event) => {
        if (event?.detail?.tab === 'consciousness' || (event?.detail?.page === 'dashboard' && shown())) void refresh({ rereadPlaces: true });
        else clearTimeout(timer);
    };
    const onChanged = () => { void refreshPlaces({ force: true }); };
    // A skill lifecycle end (an update, a turn-on, whoever started it) can change a place's icon, label or readiness
    // without the change event, and the contributions projection leaves an off connector out, so `onChange` misses it.
    const onLifecycle = () => { if (shown()) void refreshPlaces({ force: true }); else placesBehind = true; };
    const onVisibility = () => { if (placesBehind && shown()) void refreshPlaces({ force: true }); };
    const offLifecycle = contributions.onLifecycle(onLifecycle);
    mount.addEventListener('click', onClick);
    mount.addEventListener('toggle', onToggle, true);
    window.addEventListener('ouro:dashboard-subtab-shown', onShown);
    window.addEventListener('ouro:page-shown', onShown);
    window.addEventListener(PLACES_CHANGED_EVENT, onChanged);
    document.addEventListener('visibilitychange', onVisibility);
    if (shown()) void refresh(); else void refreshPlaces();
    return {
        refresh,
        destroy() {
            destroyed = true;
            releaseNotes(mount);
            clearTimeout(timer);
            mount.removeEventListener('click', onClick);
            mount.removeEventListener('toggle', onToggle, true);
            window.removeEventListener('ouro:dashboard-subtab-shown', onShown);
            window.removeEventListener('ouro:page-shown', onShown);
            window.removeEventListener(PLACES_CHANGED_EVENT, onChanged);
            document.removeEventListener('visibilitychange', onVisibility);
            offLifecycle();
        },
    };
}
