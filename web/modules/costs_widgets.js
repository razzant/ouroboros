// Dashboard › Costs, the widgets its authors place there (docs/WIDGET_HOST_PROTOCOL.md §1, §7): the live tabs of
// `GET /api/widgets` whose `render.page` is `costs`, each a card of its own with the Widgets board's frame, head
// controls and start ledger (widget_module.js, widget_card.js, holds.js), in the owner's `widget_order`. The board stays
// the widget's home: its arrangement, launch policy and menu live there, and the card's door opens that very card. Read
// on show, Refresh, `extension_lifecycle` and a reconnect while Costs is shown, never polled; a failed read keeps the
// cards it last drew as they are (a running frame runs on) and offers Retry. A widget runs only while Costs is shown:
// leaving stops every frame through the dispose handshake and takes a host-drawn widget off the screen once its
// disposer kept what it may. A return starts the frames whose launch policy starts on show (the owner's
// `widget_start_mode` over the author's `start`) only when that read brought the choice, the map itself (an empty map is
// a choice: nothing overridden; an error object, another shape, saved preferences the server could not read
// (`unreadable`) or no answer is no choice): without it nothing starts by itself, a framed widget waits for Start beside
// Retry, and what runs keeps running. A start that policy made is decided again by each read until its frame gets its
// document; the owner's Start is not. The owner's Stop holds on this
// device until Start (`costs:<skill>:<tab>`); a start the page did not outlive holds the widget everywhere
// (`widget:<skill>:<tab>` at its revision). A widget the list does not name is not drawn and nothing is said of it: an
// absent card proves neither that its skill is off nor that it exists. The card a changed widget leaves behind stays in
// view while its frame stops, saying Stopping and taking no input; nothing on it reaches the fresh card of its key.
import { apiClient } from './api_client.js';
import { openSkillWidgets } from './contributions.js';
import { getHolds, HELD_LINES, settleAfterFrameLoad } from './holds.js';
import { controlIcon } from './page_icons.js';
import { errorWords, escapeHtmlAttr as escapeHtml } from './ui_primitives.js';
import {
    isFramedWidget, renderWidgetCardParts, renderWidgetFacade, setWidgetCardNote, syncWidgetCardControls, widgetCodeHoldKey,
    widgetStartsOnShow,
} from './widget_card.js';
import { assertWidgetsList, requestWidgetListPayload, widgetCardSignature, widgetKey, widgetListRequests } from './widget_list.js';
import { boardFrameGeometry } from './widget_module.js';
import { sortTabsByWidgetOrder } from './widget_reorder.js';
import { forgetWidgetSessionState, mountTabInto } from './widgets.js';

export const COSTS_PAGE = 'costs';
// A widget that waits for Start rests as its icon and title, not as tall as a frame it is not running.
const RESTING_HEIGHT = 96;

/** A live widget its author placed on Costs: a skill's own tab, never a card kept from the chat. */
export const placedOnCosts = (tab) => tab?.render?.page === COSTS_PAGE && !tab?.pinned_card;

/** Costs' mount of one placed widget: the board's own dispatch by render kind (widgets.js `mountTabInto`) with Costs'
 *  frame policy (`policy`), its message handlers and a declarative mount's session key (`key`, never the board's). */
export const mountWidgetTab = (mount, tab, { signal, policy, messageHandlers, key }) => mountTabInto(mount, tab,
    { signal, messageHandlers, board: policy, key });

// The card names itself by its words, not by an id built from the key: two valid keys (`quota:work_day`, `quota_work:day`)
// would fold into one id, and a changed widget's card and its fresh one share a key while the old one stops.
function cardHtml(tab) {
    const key = widgetKey(tab);
    const framed = isFramedWidget(tab);
    const title = escapeHtml(tab.title || tab.tab_id || tab.skill);
    return `<section class="ui-card costs-widget-card" data-widget-key="${escapeHtml(key)}"${framed ? ' data-widget-framed' : ''} aria-label="${title}">
        <div class="costs-widget-head">
            <h3 class="dashboard-card-title costs-widget-title">${title}</h3>
            <div class="costs-widget-controls">${renderWidgetCardParts(tab).run}
                <button type="button" class="costs-meter-action costs-widget-open" data-costs-widget-open aria-label="Open in Widgets" title="Open in Widgets">${controlIcon('chevron', 18)}</button>
            </div>
        </div>
        ${framed ? '<p class="costs-widget-fault" data-widget-fault-details hidden></p>' : ''}
        <div class="costs-widget-body" data-widget-mount></div>
    </section>`;
}

// The board's one list read under its one deadline (widget_list.js); here cards that answered stay when only the
// owner's launch choices missed it: those read as unknown, so nothing starts by itself (the board keeps its contract).
async function readPlacements(client, controller) {
    let cards = null;
    const seen = { widgets: (options) => client.widgets(options).then((data) => { cards = data; return data; }),
        uiPreferences: (options) => client.uiPreferences(options) };
    try {
        return await requestWidgetListPayload(seen, controller);
    } catch (error) {
        if (cards && error?.code === 'WIDGET_REQUEST_TIMEOUT') return [assertWidgetsList(cards), null];
        throw error;
    }
}

/**
 * Mount the placed widgets into `host`; returns `{ show, leave, refresh, destroy }`, the page's verbs: `show` reads the
 * list and starts what starts on show, `leave` stops every frame and ends the read out, `refresh` reads again while
 * shown. `ws` brings lifecycle, reconnect and the skills' own events; `mountTab` and `holds` are seams for tests.
 */
export function mountCostsWidgets(host, { ws = null, client = apiClient, holds = getHolds(), openWidgets = openSkillWidgets,
    mountTab = mountWidgetTab } = {}) {
    const records = new Map();  // key -> {key, tab, node, signature, dispose, starting, visit, attempt, controller, natural, held, retired}
    const stopping = new Map();  // key -> the settle of an ordered stop: the next frame of that key waits for it
    const mounting = new Map();  // key -> the start under way: the next start of that key begins once it ended (and stopped)
    const requests = widgetListRequests();
    const messageHandlers = new Set();
    let prefs = null;  // the owner's launch choices as the latest read brought them; null when it did not: unknown, not none
    let order = [];
    let shownKeys = [];
    let shown = false;
    let destroyed = false;
    let generation = 0;  // the list read: a later one, or leaving, makes an earlier one's answer stale
    let visit = 0;  // the showing of Costs: leaving begins the next, and a start counts only in its own
    let line = null;
    const stopKey = (tab) => `${COSTS_PAGE}:${widgetKey(tab)}`;
    const live = (record, at = visit) => shown && !destroyed && at === visit && records.get(record.key) === record;
    const startsOnShow = (tab) => widgetStartsOnShow(tab, prefs, holds, stopKey(tab));

    // A framed widget's frame here: as tall as its content under the author's ceiling (a fixed one as declared),
    // measured anew in each document and remembered for this page session, so a return stands at the last height while
    // its frame comes back. Its faults are reported as Costs', not the board's.
    const geometry = (record) => boardFrameGeometry(record.tab.render, { natural: record.natural });
    const framePolicy = (record, starting) => ({
        where: COSTS_PAGE,
        starting,
        geometry: () => geometry(record),
        held: () => record.held,
        measured(natural, held = 0) { record.natural = natural; record.held = held; },
        attach: () => () => {},
    });

    // One line above the cards, in the Widgets board's list-failure words and shape (widgets.js `#widgets-list-error`),
    // with Retry: the list could not be read (the last cards stay), or it came without the owner's launch choices.
    function showLine(text) {
        if (!text) { line?.remove(); line = null; return; }
        if (!line) {
            line = document.createElement('div');
            line.className = 'skills-load-error costs-widgets-error';
            line.setAttribute('role', 'status');
            line.innerHTML = '<span data-costs-widgets-error></span> <button type="button" class="btn btn-default btn-sm" data-costs-widgets-retry>Retry</button>';
            host.prepend(line);
        }
        line.querySelector('[data-costs-widgets-error]').textContent = text;
    }

    function stop(record) {
        record.controller?.abort();
        record.controller = null;
        const dispose = record.dispose;
        record.dispose = null;
        let result = null;
        try { result = dispose?.(); } catch (error) { console.warn('costs widgets: dispose failed', error); }
        // A host-drawn widget leaves the screen with its disposer, which has just kept what may be kept (widgets.js
        // session state, passwords never): no disposed control stays to be used, and only a read mounts it anew.
        if (dispose && !isFramedWidget(record.tab)) record.node.querySelector('[data-widget-mount]')?.replaceChildren();
        // A placement that left the list keeps no session state for its return; its declarative disposer has just
        // written it. Leaving Costs, or a changed widget's fresh card, keeps it.
        if (record.retired) forgetWidgetSessionState(stopKey(record.tab));
        if (result && typeof result.then === 'function') {
            const settling = result.catch(() => {}).finally(() => { if (stopping.get(record.key) === settling) stopping.delete(record.key); });
            stopping.set(record.key, settling);
        }
        return stopping.get(record.key) || null;
    }

    // The card a changed or vanished widget leaves behind stays in view until its frame has stopped, as the board's
    // retired card does: it says Stopping, is marked as on its way out (`data-widget-removed`) and takes no input: its
    // controls are disabled and its body inert, so neither pointer nor keyboard reaches the frame while the dispose
    // handshake runs on, and its head stays readable. The fresh card of its key stands beside it, and a press only ever
    // acts through the node `records` holds for the key.
    function retire(node) {
        node.setAttribute('data-widget-removed', '');
        syncWidgetCardControls(node, 'stopping');
        node.querySelectorAll('[data-widget-power], [data-costs-widget-open]').forEach((control) => { control.disabled = true; });
        node.querySelector('[data-widget-mount]')?.setAttribute('inert', '');
    }

    async function rest(record) {
        const { key, node, tab } = record;
        if (stopping.has(key) && isFramedWidget(tab)) { syncWidgetCardControls(node, 'stopping'); await stopping.get(key); }
        if (record.starting) await record.starting.catch(() => {});
        // A start begun meanwhile (a return to Costs, Start) owns the card now.
        if (record.dispose || record.starting || records.get(key) !== record || !node.isConnected) return;
        if (!isFramedWidget(tab)) return;
        const mount = node.querySelector('[data-widget-mount]');
        // A failed start keeps its words; otherwise the facade, as tall as the frame that returns on the next show (the
        // one it will be given, never the content's own height past the author's ceiling).
        if (!mount.querySelector('.skills-load-error')) {
            renderWidgetFacade(mount, tab, startsOnShow(tab) ? geometry(record).height : RESTING_HEIGHT);
        }
        syncWidgetCardControls(node, 'stopped');
        setWidgetCardNote(node, HELD_LINES[holds.reason(widgetCodeHoldKey(tab), tab.revision)] || '');
    }

    // One start per card in one showing: a second request in it joins the start under way. A start left from an earlier
    // showing (leave → return before it ended) or one for a card this one replaced is never joined: it ends first,
    // stopping a frame it put in once it finds itself stale, and this start then mounts anew (widgets.js `mountTrackedTab`).
    // Its `attempt` says whether the owner pressed Start (`explicit`), whether its frame got its document (`installed`,
    // as the start ledger is written) and whether a later read declined it before then (`declined`: it ends quietly,
    // and a start asked for afterwards begins anew once it has).
    function start(record, { explicit = false } = {}) {
        if (record.dispose) return null;
        if (record.starting && record.visit === visit && !record.attempt.declined) {
            if (explicit) record.attempt.explicit = true;
            return record.starting;
        }
        const at = visit;
        const attempt = { explicit, installed: false, declined: false, controller: null };
        const prior = mounting.get(record.key)?.promise || null;
        const run = (async () => {
            const { key, node, tab } = record;
            const mount = node.querySelector('[data-widget-mount]');
            const framed = isFramedWidget(tab);
            if (framed) {
                renderWidgetFacade(mount, tab, geometry(record).height);
                syncWidgetCardControls(node, 'starting');
            }
            // One frame per key on this page: the start before this one ends, then a stop still awaiting its acknowledgement.
            if (prior) await prior.catch(() => {});
            if (stopping.has(key)) await stopping.get(key);
            if (!live(record, at) || record.dispose || attempt.declined) return;
            const controller = new AbortController();
            record.controller = controller;
            attempt.controller = controller;
            let settle = () => {};
            const starting = () => { attempt.installed = true; settle = holds.starting(widgetCodeHoldKey(tab), tab.revision); };
            try {
                if (framed) mount.querySelector('.skills-load-error')?.remove();
                const dispose = await mountTab(mount, tab, { signal: controller.signal, messageHandlers, key: stopKey(tab),
                    policy: framed ? framePolicy(record, starting) : null });
                if (typeof dispose !== 'function') {
                    settle();
                    if (framed && live(record, at) && !attempt.declined) syncWidgetCardControls(node, 'stopped');
                    return;
                }
                const cancelSettle = framed ? settleAfterFrameLoad(mount, settle) : () => {};
                record.dispose = () => { cancelSettle(); settle(); return dispose(); };
                if (!live(record, at) || controller.signal.aborted) { stop(record); return; }
                if (framed) syncWidgetCardControls(node, 'running');
            } catch (error) {
                settle();
                if (!live(record, at) || attempt.declined) return;
                mount.innerHTML = `<div class="skills-load-error">${escapeHtml(`The widget could not start: ${errorWords(error).text}`)}</div>`;
                if (framed) syncWidgetCardControls(node, 'stopped');
            }
        })().finally(() => {
            if (record.starting === run) record.starting = null;
            if (mounting.get(record.key)?.promise === run) mounting.delete(record.key);
        });
        Object.assign(record, { starting: run, visit: at, attempt });
        mounting.set(record.key, { record, promise: run });
        return run;
    }

    function createRecord(tab, after = null) {
        const template = document.createElement('template');
        template.innerHTML = cardHtml(tab).trim();
        const node = template.content.firstElementChild;
        if (after) after.after(node); else host.append(node);
        return { key: widgetKey(tab), tab, node, signature: widgetCardSignature(tab), dispose: null, starting: null, visit: 0,
            attempt: null, controller: null, natural: 0, held: 0, retired: false };
    }

    // While no card holds a frame, a stop or a start (a return, before any start), the cards also take the owner's
    // order in the document, so keyboard focus follows what the eye reads; a card already in its place is not moved.
    function arrange(tabs) {
        if (stopping.size || mounting.size || [...records.values()].some((record) => record.dispose)) return;
        let previous = line;  // the failure line, when shown, stays first
        for (const tab of tabs) {
            const { node } = records.get(widgetKey(tab));
            if ((previous ? previous.nextSibling : host.firstChild) !== node) {
                if (previous) previous.after(node); else host.prepend(node);
            }
            previous = node;
        }
    }

    // Keyed patch: a card whose entry changed or that left the list stops in order and goes, a changed one's fresh
    // card standing beside it; the others keep their node, so a running frame is never moved (a moved frame reloads).
    // The owner's order places the cards (style.css `order`), and the document too when `arrange` may.
    function patch(tabs) {
        const next = new Map(tabs.map((tab) => [widgetKey(tab), tab]));
        for (const [key, record] of [...records]) {
            const tab = next.get(key);
            if (tab && record.signature === widgetCardSignature(tab)) { record.tab = tab; continue; }
            records.delete(key);
            record.retired = !tab;
            const fresh = tab ? createRecord(tab, record.node) : null;
            if (fresh) records.set(key, fresh);
            const settling = stop(record);
            if (settling) { retire(record.node); settling.then(() => record.node.remove()); } else record.node.remove();
        }
        for (const tab of tabs) if (!records.has(widgetKey(tab))) records.set(widgetKey(tab), createRecord(tab));
        tabs.forEach((tab, index) => records.get(widgetKey(tab)).node.style.setProperty('--costs-widget-order', String(index)));
        arrange(tabs);
        shownKeys = tabs.map(widgetKey);
    }

    async function read() {
        if (!shown || destroyed) return;
        const mine = ++generation;
        requests.abortAll();
        let data = null;
        let stored = null;
        try {
            [data, stored] = await requests.run((controller) => readPlacements(client, controller));
        } catch (error) {
            if (mine !== generation || !shown || error?.name === 'AbortError') return;
            showLine(`The widgets placed here could not be read. ${errorWords(error).text}`);
            return;
        }
        if (mine !== generation || !shown) return;
        // Launch choices count only as this read brought them: a choice read earlier may since have changed on the
        // board, so a read without them starts nothing by itself. What it brings is the map: an object of choices, empty
        // when none was overridden (nothing saved yet included); an error object, an array, a missing field or the
        // defaults that stand in for saved preferences the server could not read (`unreadable`) is no read, never "no
        // choice". The order last read then still arranges the cards.
        const saved = stored && typeof stored === 'object' && !stored.unreadable ? stored : null;
        const choices = saved?.widget_start_mode;
        prefs = choices && typeof choices === 'object' && !Array.isArray(choices) ? { widget_start_mode: choices } : null;
        if (Array.isArray(saved?.widget_order)) order = saved.widget_order;
        // A card the stored order does not hold yet keeps the place this page last showed it in (widgets.js does too).
        const tabs = sortTabsByWidgetOrder(data.ui_tabs.filter(placedOnCosts), order, shownKeys);
        showLine(prefs === null && tabs.some(isFramedWidget) ? 'Your launch policies could not be read: widgets here start only with Start.' : '');
        patch(tabs);
        // A return while the frames it left still stop: those stops settle within the dispose handshake's bound
        // (WIDGET_DISPOSE_ACK_TIMEOUT_MS), then the cards take their places before this showing starts a frame. A start
        // left from before is not waited for (its own requests bound it): its frame may already be in its card, where a
        // move would reload it, so its showing keeps the cards where they are.
        if (stopping.size && !mounting.size && ![...records.values()].some((record) => record.dispose)) {
            await Promise.allSettled([...stopping.values()]);
            if (mine !== generation || !shown) return;
            arrange(tabs);
        }
        await Promise.allSettled(tabs.map((tab) => {
            const record = records.get(widgetKey(tab));
            if (!record || record.dispose) return null;
            // A start under way in this showing goes on when the owner pressed Start or its frame already has its
            // document (a frame that runs is never stopped by a read). One the launch policy made is this read's to
            // decide until then, as on showing: a fresher Manual, a hold or an answer without the choices declines it,
            // and the card rests with Start once it has ended.
            const attempt = record.starting && record.visit === visit && !record.attempt.declined ? record.attempt : null;
            if (attempt && (attempt.explicit || attempt.installed || startsOnShow(tab))) return record.starting;
            if (attempt) { attempt.declined = true; attempt.controller?.abort(); }
            return startsOnShow(tab) ? start(record) : rest(record);
        }));
    }

    function leave() {
        if (!shown) return;
        shown = false;
        generation += 1;
        visit += 1;
        requests.abortAll();
        for (const record of records.values()) {
            if (!record.dispose && !record.starting) continue;
            stop(record);
            rest(record);
        }
    }

    host.addEventListener('click', (event) => {
        if (event.target.closest?.('[data-costs-widgets-retry]')) { read(); return; }
        const card = event.target.closest?.('[data-widget-key]');
        const record = card ? records.get(card.dataset.widgetKey) : null;
        if (!record || record.node !== card) return;
        if (event.target.closest('[data-costs-widget-open]')) { openWidgets(record.tab.skill, record.key); return; }
        if (!event.target.closest('[data-widget-power]:not(:disabled)')) return;
        if (record.dispose) {
            holds.stop(stopKey(record.tab));
            stop(record);
            rest(record);
        } else {
            holds.release(stopKey(record.tab));
            holds.release(widgetCodeHoldKey(record.tab));
            start(record, { explicit: true });
        }
    });
    // A mount that ended by itself (it left its own page, its files went away) settles stopped, its fault kept.
    host.addEventListener('ouro-widget-ended', (event) => {
        const card = event.target.closest?.('[data-widget-key]');
        const record = card ? records.get(card.dataset.widgetKey) : null;
        if (record?.node === card) { stop(record); rest(record); }
    });
    const offs = ws?.on ? [
        ws.on('message', (msg) => messageHandlers.forEach((handler) => handler(msg))),
        ws.on('extension_lifecycle', () => { read(); }),
        ws.on('open', () => { read(); }),
    ] : [];

    return {
        show() { if (destroyed) return; shown = true; read(); },
        leave,
        refresh: read,
        destroy() {
            leave();
            destroyed = true;
            offs.forEach((off) => off?.());
            for (const record of records.values()) record.node.remove();
            records.clear();
            showLine('');
        },
    };
}
