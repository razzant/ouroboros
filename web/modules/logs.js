import { escapeHtml, escapeHtmlAttr } from './utils.js';
import { apiFetch } from './api_client.js';
import { setInlineStatus } from './ui_helpers.js';
import { harnessIdentityMarkup } from './harness_presentation.js';
import {
    LOG_CATEGORIES,
    categorizeLogEvent,
    duplicateLogEventKey,
    executorChip,
    formatReviewProjection,
    getLogTaskGroupId,
    isGroupedTaskEvent,
    normalizeLogTs,
    summarizeLogEvent,
} from './log_events.js';
import { outcomeTone } from './task_phase_chip.js';
import { bindScrollFade } from './scroll_fade.js';
import { createLogRaw, LOG_RAW_HTML } from './logs_raw.js';

/** Dashboard › Logs (docs/ARCHITECTURE.md §3 "Dashboard"). The panel's entries are data first: every live frame and
 * backfilled row updates its entry (task grouping, ×N collapse, the sticky Errors filter, the settled headline) whether
 * or not the panel is on screen, while the DOM follows only a shown panel. Hidden, the panel writes no node and reads no
 * history; shown again it places what changed newest first, RENDER_CHUNK entries a frame, so a burst that arrived
 * meanwhile never becomes one long rebuild. Every event has one fold: a task's holds its timeline, whose rows exist
 * while it is open; an event's holds its record, which pins one compact row's source or received event while open,
 * pages asynchronously and offers a full download (logs_raw.js). The reader keeps their place across hide and show:
 * following the newest entry, or on the entry they were reading. The compact history says what it left out: the fields
 * a row's view omits (in its record) and the damaged records a read skipped (in the status line).
 * The panel is one Dashboard tab body, its scroller; an event is one tray of the list's one card (U026-U035). */

// The kinds of damaged record a compact read skips (`gaps`, ouroboros/gateway/logs.py); an unknown kind is shown as named.
const GAP_WORDS = { malformed_jsonl: 'malformed line', incomplete_live_line: 'unfinished live line', torn_archive_line: 'torn archive line' };

// What a compact row's view leaves out (`_raw.omitted_fields`, at most 64 names of `omitted_field_count`); its record has it.
function omittedNote(evt) {
    const names = Array.isArray(evt?._raw?.omitted_fields) ? evt._raw.omitted_fields.map(String) : [];
    const count = Math.max(names.length, Number.isSafeInteger(evt?._raw?.omitted_field_count) ? evt._raw.omitted_field_count : 0);
    if (!count) return '';
    const more = count - names.length;
    const left = names.length ? `${names.join(', ')}${more ? ` and ${more} more` : ''}` : `${count} fields`;
    return `Left out of this view: ${left}. The record below reads the whole row.`;
}

export function initLogs({ ws, state, mount }) {
    const MAX_LOGS = 500;
    const MAX_TASK_EVENTS = 30;
    // Entries placed per frame while a shown panel catches up with what changed while it was hidden.
    const RENDER_CHUNK = 40;
    const duplicateWindowMs = 5000;
    const duplicateState = new Map();
    const taskGroups = new Map();
    // Every entry the panel holds, oldest first: the panel's order. An entry that changes moves to the end.
    const entries = new Map();
    // Entries whose node does not show their latest state yet (or has none).
    const stale = new Set();
    const entryOf = new WeakMap();  // an entry's node → the entry
    const removedNodes = new Set(); // hidden trimming owes DOM removal until the next showing
    // Exact-duplicate guard so a backfilled row and its live-stream twin (or a
    // reconnect re-backfill) render once. Bounded; a rare full clear briefly
    // disables the guard, which at worst re-shows one row — never drops one.
    const renderedLogKeys = new Set();
    let backfillRevision = 0, order = 0, nextId = 0, painting = false, batching = false;
    // Recent history is owed to the panel: read when it is shown, again after every reconnect.
    let backfillOwed = true, backfillAbort = null;
    // A history read in progress keeps the empty line quiet; Clear's time is said until an event comes.
    let historyLoading = false, clearedAt = '';
    // The reader follows the newest entry (`pinned`) or reads an entry (`anchor`); `ownTop` is this module's own scroll.
    let pinned = true, anchor = null, ownTop = null, selectionInside = false, settleQueued = false;

    // The filter shows the kinds chosen, every kind while none is: All, lit at rest. The choice lives in
    // `state.activeFilters`, which meant the kinds SHOWN before (all six at rest, one turned off to narrow): read once
    // here, a kind it still shows stays chosen and all six (or none) is the rest, so what the owner saw stays.
    const KINDS = Object.keys(LOG_CATEGORIES);
    const choice = state.activeFilters || {};
    if (KINDS.every((key) => choice[key])) KINDS.forEach((key) => { choice[key] = false; });
    state.activeFilters = choice;
    const narrowed = () => KINDS.some((key) => state.activeFilters[key]);
    const shows = (category) => !narrowed() || Boolean(state.activeFilters[category]);

    // One Dashboard tab body is the tab's root and its one scroller; the strip stays at its top while the list, one card
    // of trays, scrolls under it. The strip is the filter (Skills' strip: it wraps, and on a phone it is one line that
    // scrolls sideways), then the list's own action at its end; the history line and the empty line live in the card.
    const page = document.createElement('div');
    page.id = 'page-logs';
    page.className = 'dashboard-body settings-logs-panel';
    page.innerHTML = `
        <div class="logs-filters"><div class="logs-kinds" id="log-filters" role="group" aria-label="Show" data-scroll-x>${
            Object.entries({ '': 'All', ...LOG_CATEGORIES }).map(([key, label]) => `<button type="button" class="filter-chip" data-log-kind="${key}">${label}</button>`).join('')
        }</div><button type="button" class="btn btn-ghost btn-sm logs-clear-list" id="btn-clear-logs">Clear list</button></div>
        <section class="ui-card logs-card">
            <div class="ui-status logs-history-status" role="status"></div>
            <p class="logs-empty" data-log-empty hidden></p>
            <div id="log-entries"></div>
        </section>
    `;
    mount.appendChild(page);

    const filtersDiv = page.querySelector('#log-filters');
    bindScrollFade(filtersDiv, { axis: 'x' });  // a phone's one scrolling line fades where chips hide
    const logEntries = page.querySelector('#log-entries');
    const historyStatus = page.querySelector('.logs-history-status');
    const emptyLine = page.querySelector('[data-log-empty]');
    function isLogsVisible() {
        return state.activePage === 'dashboard' && state.dashboardActiveSubtab === 'logs' && !document.hidden;
    }

    const raw = createLogRaw({ visible: isLogsVisible });

    // A live entry never yanks the view away from history the reader scrolled
    // up to: following sticks only while the reader is at the bottom of the tab
    // body, the scroller. The reader's own scroll decides it; this module's
    // placements do not. Deliberate jumps (a filter, Clear) follow the newest entry again.
    const PINNED_SLACK_PX = 48;
    function isPinnedToLatest() {
        return page.scrollHeight - page.scrollTop - page.clientHeight <= PINNED_SLACK_PX;
    }
    function noteReader() {
        if (!isLogsVisible() || (ownTop !== null && page.scrollTop === ownTop)) return;
        ownTop = null;
        pinned = isPinnedToLatest();
        anchor = pinned ? null : captureAnchor();
    }
    // The first two entries on screen and where they stand: the second keeps the place when the first moves away.
    function captureAnchor() {
        const box = page.getBoundingClientRect();
        const marks = [];
        for (const node of logEntries.children) {
            const entry = entryOf.get(node);
            if (!entry || node.hidden) continue;
            const rect = node.getBoundingClientRect();
            if (rect.bottom <= box.top) continue;
            marks.push({ entry, order: entry.order, offset: rect.top - box.top });
            if (marks.length === 2) break;
        }
        return marks;
    }
    function anchorTop() {
        for (const mark of anchor || []) {
            const node = mark.entry.node;
            if (mark.entry.removed || mark.entry.order !== mark.order || node?.parentElement !== logEntries) continue;
            return page.scrollTop + node.getBoundingClientRect().top - page.getBoundingClientRect().top - mark.offset;
        }
        pinned = true;  // the entry read left this view (trimmed or cleared): the newest one is the place
        return page.scrollHeight;
    }
    function settle() {
        if (!isLogsVisible()) return;
        page.scrollTop = pinned ? page.scrollHeight : anchorTop();
        ownTop = page.scrollTop;
    }
    function settleSoon() {
        if (settleQueued) return;
        settleQueued = true;
        requestAnimationFrame(() => { settleQueued = false; settle(); });
    }
    function follow() {
        pinned = true;
        anchor = null;
        settleSoon();
    }

    function updateVisibility(entry) {
        if (entry.node) entry.node.hidden = !shows(entry.category);
        raw.exposure();
    }

    // One line says why the list shows nothing (U033): every event filtered out (with Show all), the owner's
    // Clear, or no event yet; nothing while the history is still loading (its own line says so). It reads the
    // entries, not the nodes a shown panel may still be placing, and is written only while the panel shows.
    let emptyWords = null;
    function updateEmptyState() {
        if (!isLogsVisible()) return;
        const quiet = [...entries.values()].some((entry) => shows(entry.category))
            || (historyLoading && !entries.size);
        if (emptyLine.hidden !== quiet) emptyLine.hidden = quiet;
        if (quiet) return;
        const words = entries.size ? '' : `${clearedAt ? `Cleared at ${clearedAt}.` : 'No events yet.'} New events appear here.`;
        if (words === emptyWords) return;
        emptyWords = words;
        if (words) emptyLine.textContent = words;
        else emptyLine.innerHTML = 'The filters hide every event. <button type="button" class="btn btn-ghost btn-sm" data-log-show-all>Show all</button>';
    }

    // The shared filter chips (ui.css `.filter-chip`, as Skills' strip): All leads, lit while no kind is chosen; a kind
    // pressed is lit beside the others chosen and All goes out; All pressed clears the choice.
    const chips = new Map([...filtersDiv.querySelectorAll('[data-log-kind]')].map((chip) => [chip.dataset.logKind, chip]));
    function syncFilterChips() {
        chips.forEach((chip, key) => {
            const lit = key ? Boolean(state.activeFilters[key]) : !narrowed();
            chip.classList.toggle('active', lit);
            chip.setAttribute('aria-pressed', String(lit));
        });
    }

    function applyFilters() {
        syncFilterChips();
        entries.forEach(updateVisibility);
        updateEmptyState();
        follow();
    }

    function choose(key) {
        if (key) state.activeFilters[key] = !state.activeFilters[key];
        else KINDS.forEach((kind) => { state.activeFilters[kind] = false; });
        applyFilters();
    }

    chips.forEach((chip, key) => chip.addEventListener('click', () => choose(key)));
    syncFilterChips();

    // The empty line's Show all is All pressed: every kind shown.
    emptyLine.addEventListener('click', (event) => {
        if (event.target?.closest?.('[data-log-show-all]')) choose('');
    });

    // ---- The entries (data) --------------------------------------------------------------------------------------

    // An update moves its entry to the end and restamps it (`order`): an anchored reader lets a moved entry go. An entry
    // already last among the shown ones stays where the reader sees it, so it keeps its stamp and the reader their place.
    function touch(entry) {
        if (!lastShown(entry.node)) entry.order = ++order;
        entries.delete(entry.id);
        entries.set(entry.id, entry);
        stale.add(entry);
        if (batching) return;
        trim();
        paint();
    }

    function lastShown(node) {
        if (node?.parentElement !== logEntries) return false;
        let next = node.nextElementSibling;
        while (next?.hidden) next = next.nextElementSibling;
        return !next;
    }

    function trim() {
        while (entries.size > MAX_LOGS) drop(entries.values().next().value);
    }

    // A merged history is older than what arrived live while it was owed: every entry takes its place by time (U035),
    // the oldest past the bound leaves. Only moved nodes are written (`place`); an anchored reader's entries keep their stamp.
    function sortByTime() {
        const sorted = [...entries.values()].sort((a, b) => a.at - b.at);
        entries.clear();
        for (const entry of sorted) { entries.set(entry.id, entry); stale.add(entry); }
        trim();
    }

    function drop(entry) {
        entries.delete(entry.id);
        stale.delete(entry);
        entry.removed = true;
        if (taskGroups.get(entry.groupId) === entry) taskGroups.delete(entry.groupId);
        if (entry.dedupeKey && duplicateState.get(entry.dedupeKey)?.entry === entry) duplicateState.delete(entry.dedupeKey);
        if (entry.node) {
            raw.remove(entry.node);
            if (isLogsVisible()) entry.node.remove();
            else removedNodes.add(entry.node);
        }
    }

    function addEntry(entry) {
        entry.id = ++nextId;
        touch(entry);
        return entry;
    }

    function createStandaloneEntry(evt) {
        const dedupeKey = duplicateLogEventKey(evt);
        const now = (() => {
            const parsed = evt.ts ? Date.parse(evt.ts) : NaN;
            return Number.isFinite(parsed) ? parsed : Date.now();
        })();
        const tsText = normalizeLogTs(evt.ts || evt.timestamp);

        if (dedupeKey) {
            let last = duplicateState.get(dedupeKey);
            if (last?.entry.removed) {
                duplicateState.delete(dedupeKey);
                last = null;
            }
            if (last && Math.abs(now - last.ts) <= duplicateWindowMs) {
                last.count += 1;
                last.entry.repeat = last.count;
                // A backfilled twin may be the older one: the entry keeps showing the newest.
                if (now >= last.ts) Object.assign(last.entry, { evt, tsText, at: now });
                last.ts = Math.max(last.ts, now);
                touch(last.entry);
                return;
            }
        }

        const view = summarizeLogEvent(evt);
        const entry = addEntry({
            kind: 'standalone', evt, view, tsText, repeat: 1, dedupeKey, at: now, omitted: omittedNote(evt),
            category: categorizeLogEvent(evt, view), execution: executorChip(evt),
        });
        if (dedupeKey) duplicateState.set(dedupeKey, { entry, ts: now, count: 1 });
    }

    function updateTaskGroupCard(evt) {
        const groupId = getLogTaskGroupId(evt);
        if (!groupId) {
            createStandaloneEntry(evt);
            return;
        }

        const view = summarizeLogEvent(evt);
        const eventCategory = categorizeLogEvent(evt, view);
        // A task group stays under Errors once one of its events earned it
        // (#323): the next heartbeat must not flip the card back to amber and
        // drop it out of the Errors filter while the failure is still in its
        // timeline. The phase pill still follows the latest event.
        const earlier = taskGroups.get(groupId);
        // A wake-up's group is labelled by the origin fact its frames carry
        // (sticky once seen).
        const wake = evt.initiator === 'consciousness' || Boolean(earlier?.wake);
        const category = eventCategory === 'errors' || earlier?.category === 'errors' ? 'errors'
            : (wake ? 'consciousness' : 'tasks');
        // A task is named by its own type and its id's first eight characters, as every other row of the log names
        // one (U030); the type, once seen, stays the card's.
        const taskType = String(earlier?.taskType || evt.task_type || '').trim();
        const record = earlier || createTaskGroupCard(groupId);
        const ts = normalizeLogTs(evt.ts || evt.timestamp);
        const at = Date.parse(evt.ts || evt.timestamp || '') || Date.now();
        // A backfilled row older than what this card already shows joins its timeline in place and leaves the rest.
        const behind = batching && earlier && at < record.at;

        record.events += 1;
        record.category = category;
        record.wake = wake;
        record.taskType = taskType;
        record.kindLabel = wake ? 'Consciousness' : `${taskType || 'task'} ${String(groupId).slice(0, 8)}`;
        // The card's time is its newest event's: an older frame, live or from history, never dates it back.
        if (at >= record.at) { record.tsText = ts; record.at = at; }
        // The card's headline settles only on the typed task_done frame. The bus
        // serves control before telemetry, so a task's late tool or round frame
        // (its own `done`/`error` phase included) can arrive after its task_done:
        // it stays a timeline row and never reopens or rewrites the settled
        // headline. Until then the headline follows the latest frame.
        const settles = evt.type === 'task_done';
        if (settles || (!record.terminal && !behind)) {
            record.phase = view.phase || 'info';
            // A settled headline is the task's outcome and takes the cards' tone (Stopped neutral, Done with
            // warnings green); until then it is the latest event's phase in its level's colour.
            record.tone = settles ? outcomeTone(view.phase) : '';
            record.headline = view.headline || 'Task activity';
        }
        if (settles) record.terminal = true;
        // The card's facts under its words; the type its kind already names is not said again (U029).
        if (!behind) record.meta = view.meta.filter((item) => item !== taskType);
        const execution = executorChip(evt), review = formatReviewProjection(evt.review_projection);
        record.execution = (behind ? record.execution || execution : execution || record.execution);
        record.review = (behind ? record.review || review : review || record.review);

        const last = record.recent[record.recent.length - 1];
        const dedupeKey = duplicateLogEventKey(evt);
        // A row's body and what its view left out come from the same event (an adopted file source changes neither).
        if (!behind && last && last.dupKey && last.dupKey === dedupeKey) {
            Object.assign(last, { count: last.count + 1, ts, meta: view.meta, body: view.body, full: view.full, omitted: omittedNote(evt),
                evt, at, rev: last.rev + 1 });
        } else {
            const row = {
                ts,
                phase: view.phase || 'info',
                headline: view.headline || 'Task event',
                meta: view.meta,
                body: view.body,
                full: view.full,
                omitted: omittedNote(evt),
                evt,
                at,
                count: 1,
                dupKey: dedupeKey,
                rev: 0,
            };
            const later = behind ? record.recent.findIndex((item) => item.at > at) : -1;
            if (later >= 0) record.recent.splice(later, 0, row);
            else record.recent.push(row);
            if (record.recent.length > MAX_TASK_EVENTS) record.recent.shift();
        }
        if (behind) stale.add(record);
        else touch(record);
    }

    function createTaskGroupCard(groupId) {
        const record = { kind: 'group', groupId, events: 0, recent: [], terminal: false, meta: [], at: 0 };
        taskGroups.set(groupId, record);
        return addEntry(record);
    }

    // ---- The panel (DOM): only while shown ------------------------------------------------------------------------

    // One anatomy for every event, a task card and a timeline row alike (U027, U028): a first line with its time, what
    // it is in its kind's colour, its level chip (the default `info` says nothing, CSS hides it) and its repeat count at
    // the line's end; then its words in body text, at most three lines in the list; its facts in one meta line; and one
    // fold named for what it holds, in the same place: a task's Timeline, an event's Record. A record holds the whole
    // words where the list shows them cut, what the compact view left out, and the raw record (logs_raw.js).
    const RECORD_HTML = `<div class="log-whole" hidden></div><div class="log-note" data-log-omitted hidden></div>${LOG_RAW_HTML}`;
    const TIMELINE_HTML = '<div class="log-note log-task-review" hidden></div><div class="log-task-timeline"></div>';
    const partsOf = new WeakMap();  // an event's node → its parts, found once when it is made
    const wholeOf = new WeakMap();  // an event's node → its whole words, where its summary cut them

    function createEvent(className, fold, foldBody) {
        const node = document.createElement('div');
        node.className = className;
        node.innerHTML = `<div class="log-head"><span class="log-ts"></span><span class="log-type" hidden></span>`
            + '<span class="log-phase"></span><span class="log-repeat" hidden></span></div>'
            + '<div class="log-words"><span class="log-headline"></span><span class="log-body"></span></div><div class="log-facts" hidden></div>'
            + `<details class="ui-disclosure log-fold"><summary>${fold}</summary>`
            + `<div class="log-note log-executor-detail" hidden></div>${foldBody}</details>`;
        const find = (selector) => node.querySelector(selector);
        partsOf.set(node, { ts: find('.log-ts'), kind: find('.log-type'), phase: find('.log-phase'), repeat: find('.log-repeat'),
            words: find('.log-words'), headline: find('.log-headline'), body: find('.log-body'), facts: find('.log-facts'),
            fold: find('.log-fold'), detail: find('.log-executor-detail'),
            whole: find('.log-whole'), omitted: find('[data-log-omitted]') });
        return node;
    }

    const setText = (el, text) => { if (el && el.textContent !== text) el.textContent = text; };
    const setClass = (el, name) => { if (el && el.className !== name) el.className = name; };
    const setData = (el, key, value) => { if (el.dataset[key] !== value) el.dataset[key] = value; };
    const shownHtml = new WeakMap();  // markup a node was last given, so an unchanged one is not rewritten
    const setHtml = (el, html) => { if (el && shownHtml.get(el) !== html) { el.innerHTML = html; shownHtml.set(el, html); } };

    // An event's parts, each written only when it changed. Its words never repeat the name its kind already says (U028);
    // a delegated executor is a quiet chip among its facts, and its detail sentence is text in the fold (#360 A).
    function paintEvent(node, { ts = '', kind = null, phase = 'info', tone = '', repeat = 1, headline = '', body = '', full = '',
        meta = [], execution = null, omitted = '' }) {
        const parts = partsOf.get(node);
        setText(parts.ts, ts);
        if (kind) { setText(parts.kind, kind.label); setClass(parts.kind, `log-type ${kind.className}`); }
        parts.kind.hidden = !kind;
        setText(parts.phase, phase);
        setClass(parts.phase, `log-phase ${phase}`);
        if (tone) parts.phase.dataset.chipTone = tone;
        else delete parts.phase.dataset.chipTone;
        setText(parts.repeat, repeat > 1 ? `×${repeat}` : '');
        parts.repeat.hidden = repeat <= 1;
        // The headline, then the body on its own line: one box of at most three lines (style.css .log-words).
        const lead = kind?.label === headline ? '' : headline;
        setText(parts.headline, lead);
        setText(parts.body, lead && body ? `\n${body}` : body);
        parts.words.hidden = !lead && !body;
        wholeOf.set(node, full);
        const chip = execution ? `<span class="harness-chip log-executor-chip" title="${escapeHtmlAttr(execution.title || '')}">${
            harnessIdentityMarkup(execution.harness, { label: execution.label || '', className: 'log-executor-identity' })}</span>` : '';
        setHtml(parts.facts, chip + escapeHtml(meta.join(' · ')));
        parts.facts.hidden = !chip && !meta.length;
        setText(parts.detail, execution?.title || '');
        parts.detail.hidden = !execution?.title;
        if (parts.omitted) { setText(parts.omitted, omitted); parts.omitted.hidden = !omitted; }
        if (parts.whole && parts.fold.open) showWhole(node);
    }

    // An open record starts with the whole words where the list shows a cut: its summary shortened them, or the list
    // holds them to three lines (one measure, when the owner opens it).
    function showWhole(node) {
        const { words, whole } = partsOf.get(node);
        const full = wholeOf.get(node) || (words.scrollHeight > words.clientHeight + 1 ? words.textContent : '');
        setText(whole, full);
        whole.hidden = !full;
    }

    function renderStandalone(entry) {
        if (!entry.node) {
            entry.node = createEvent('log-entry', 'Record', RECORD_HTML);
            raw.bind(entry.node, () => entry.evt);
            entryOf.set(entry.node, entry);
        }
        const { node, view } = entry;
        setData(node, 'category', entry.category);
        setData(node, 'logTime', String(entry.at));
        paintEvent(node, { ts: entry.tsText, kind: { label: view.typeLabel, className: entry.category }, phase: view.phase || 'info',
            repeat: entry.repeat, headline: view.headline || 'Event', body: view.body, full: view.full, meta: view.meta,
            execution: entry.execution, omitted: entry.omitted });
        raw.changed(node);
    }

    function renderTaskGroup(entry) {
        if (!entry.node) {
            const node = entry.node = createEvent('log-entry log-task-card', 'Timeline', TIMELINE_HTML);
            node.dataset.taskGroup = entry.groupId;
            entry.dom = { ...partsOf.get(node), review: node.querySelector('.log-task-review'), timeline: node.querySelector('.log-task-timeline') };
            entry.rows = new Map();
            entryOf.set(node, entry);
        }
        const { node, dom } = entry;
        setData(node, 'category', entry.category);
        // A card stands where its newest event does (`at`); an older event never dates it back.
        setData(node, 'logTime', String(entry.at));
        // A task's words are its latest headline: the body of that event is its timeline row's, said once.
        paintEvent(node, { ts: entry.tsText, kind: { label: entry.kindLabel, className: entry.category }, phase: entry.phase,
            tone: entry.tone, repeat: entry.events, headline: entry.headline, meta: entry.meta, execution: entry.execution });
        // The task's review sentence opens with its timeline.
        setText(dom.review, entry.review || '');
        dom.review.hidden = !entry.review;
        if (dom.fold.open) renderTaskTimeline(entry);
    }

    // An open Timeline's rows follow the task's recent events in place: a new event appends its row, the oldest
    // past MAX_TASK_EVENTS leaves, a collapsed repeat redraws only its own row.
    function renderTaskTimeline(entry) {
        const { rows } = entry;
        for (const [item, row] of rows) {
            if (!entry.recent.includes(item)) { raw.remove(row); row.remove(); rows.delete(item); }
        }
        // Newest last, placed newest first: a row new here (backfilled ones older than those on screen too) goes right
        // before the row that follows it, already placed; a row on screen never moves.
        let next = null;
        for (let index = entry.recent.length - 1; index >= 0; index -= 1) {
            const item = entry.recent[index];
            let row = rows.get(item);
            if (!row) {
                row = createEvent('log-task-event', 'Record', RECORD_HTML);
                raw.bind(row, () => item.evt);
                rows.set(item, row);
                entry.dom.timeline.insertBefore(row, next);
            }
            next = row;
            if (row.dataset.rev === String(item.rev)) continue;
            paintEvent(row, { ts: item.ts, phase: item.phase, repeat: item.count, headline: item.headline, body: item.body,
                full: item.full, meta: item.meta, omitted: item.omitted });
            row.dataset.rev = String(item.rev);
            raw.changed(row);
        }
    }

    function closeTaskTimeline(entry) {
        raw.remove(entry.dom.timeline);
        entry.rows.clear();
        entry.dom.timeline.innerHTML = '';
    }

    function render(entry) {
        if (entry.kind === 'group') renderTaskGroup(entry);
        else renderStandalone(entry);
        updateVisibility(entry);
    }

    // Moving a node that holds the focus or the selection would drop them: they are given back.
    function place(node, before) {
        if (node.parentElement === logEntries && (node.nextElementSibling || null) === before) return;
        const doc = node.ownerDocument || document;
        const active = node.parentElement && node.contains(doc.activeElement) ? doc.activeElement : null;
        const selection = node.parentElement && selectionInside ? doc.getSelection?.() : null;
        const ranges = selection && node.contains(selection.anchorNode)
            ? Array.from({ length: selection.rangeCount }, (_, index) => {
                const range = selection.getRangeAt(index);
                return [range.startContainer, range.startOffset, range.endContainer, range.endOffset];
            }) : [];
        if (before) logEntries.insertBefore(node, before);
        else logEntries.appendChild(node);
        if (active && doc.activeElement !== active) active.focus?.({ preventScroll: true });
        if (ranges.length) {
            selection.removeAllRanges();
            for (const [start, startOffset, end, endOffset] of ranges) {
                const range = doc.createRange();
                range.setStart(start, startOffset);
                range.setEnd(end, endOffset);
                selection.addRange(range);
            }
        }
    }

    // The DOM follows the entries only while the panel is shown: newest first, RENDER_CHUNK entries now and the rest
    // on later frames, each placed right before the entry that follows it. A live event on a shown panel is one
    // entry: it is drawn at once.
    function paint() {
        if (painting || !isLogsVisible()) return;
        for (const node of removedNodes) node.remove();
        removedNodes.clear();
        updateEmptyState();
        if (!stale.size) return;
        const list = [...entries.values()];
        let budget = RENDER_CHUNK, next = null;
        for (let index = list.length - 1; index >= 0 && stale.size && budget > 0; index -= 1) {
            const entry = list[index];
            if (stale.has(entry)) {
                render(entry);
                place(entry.node, next);
                stale.delete(entry);
                budget -= 1;
            }
            next = entry.node;
        }
        // Finish this DOM batch's placement before its scroll event can be mistaken for the reader's.
        // Deferring to the next frame lets prepends/clamps turn follow-end into an old-reading anchor.
        settle();
        if (stale.size) {
            painting = true;
            requestAnimationFrame(() => { painting = false; paint(); });
        }
    }

    function show() {
        raw.exposure();
        if (!isLogsVisible()) {
            if (backfillAbort) { backfillAbort.abort(); backfillAbort = null; backfillRevision += 1; backfillOwed = true; }
            return;
        }
        if (backfillOwed) void backfillRecentLogs();
        paint();
        // A page shown again starts at scrollTop 0 in WebKit (chat.js): the reader's place is set now.
        settle();
    }

    function eventKey(evt) {
        const ts = evt?.ts || evt?.timestamp || '';
        const type = evt?.type || '';
        // Dedupe on the natural identity (timestamp has microseconds, so distinct
        // events never collide; rapid same-type repeats differ by ts and still
        // reach the x-N collapse below). Only guard when both ts and type exist.
        return (ts && type)
            ? `${ts}|${type}|${evt?.task_id || evt?.taskId || ''}|${evt?.execution_id || evt?.round || evt?.tool || ''}`
            : '';
    }
    // The entries and timeline rows holding each event, by exact key: built once per backfill, when its first twin with a
    // file source arrives (one pass, not one per row); a holder whose event was replaced or that left is skipped.
    let holders = null;
    function holdersOf(key) {
        if (!holders) {
            holders = new Map();
            const hold = (evt, holder) => {
                const at = eventKey(evt);
                if (at) holders.set(at, [...holders.get(at) || [], holder]);
            };
            for (const entry of entries.values()) {
                if (entry.kind === 'group') for (const item of entry.recent) hold(item.evt, { entry, item });
                else hold(entry.evt, { entry, item: null });
            }
        }
        return holders.get(key) || [];
    }

    function addLogEntry(evt) {
        const exactKey = eventKey(evt);
        if (exactKey && renderedLogKeys.has(exactKey)) {
            // Compact backfill can give a live twin its exact file source without counting the event twice.
            if (evt._raw) for (const { entry, item } of holdersOf(exactKey)) {
                const holder = item || entry;
                if (entry.removed || (item && !entry.recent.includes(item)) || eventKey(holder.evt) !== exactKey) continue;
                holder.evt = evt;
                if (item) item.rev += 1;
                stale.add(entry);
            }
            if (!batching) holders = null;
            return;
        }
        if (exactKey) {
            renderedLogKeys.add(exactKey);
            if (renderedLogKeys.size > 6000) {
                // Evict the oldest keys (Set preserves insertion order) instead of
                // clearing everything, so a reconnect backfill right after the cap
                // is still deduped against recent history.
                const it = renderedLogKeys.values();
                for (let i = 0; i < 1000; i += 1) renderedLogKeys.delete(it.next().value);
            }
        }
        if (isGroupedTaskEvent(evt)) {
            updateTaskGroupCard(evt);
            return;
        }
        createStandaloneEntry(evt);
    }

    // Backfill recent history so the panel is not empty when it is shown after a
    // page load or reconnect (the live WS stream only carries events from the
    // current connection). All dashboard-relevant streams are merged and
    // replayed oldest-first; addLogEntry's exact-duplicate guard collapses any
    // overlap with the live stream, so this neither drops the pre-connect window
    // nor double-renders the post-connect overlap. A hidden panel reads nothing:
    // the read is owed until it is shown.
    async function backfillRecentLogs() {
        backfillOwed = false;
        backfillAbort?.abort();
        const abort = backfillAbort = new AbortController();
        const revision = ++backfillRevision;
        historyLoading = true;
        updateEmptyState();
        setInlineStatus(historyStatus, 'Loading recent history…');
        historyStatus.hidden = false;
        const merged = [];
        const failed = [];
        const damaged = [];  // the records a read skipped as damaged, in the part of each log it read (not all history)
        for (const name of ['events', 'tools', 'progress', 'supervisor']) {
            try {
                const resp = await apiFetch(`/api/logs/${name}?view=compact&limit=150`, { cache: 'no-store', signal: abort.signal });
                if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
                const data = await resp.json();
                if (!Array.isArray(data?.entries)) throw new Error('History unavailable');
                if (revision !== backfillRevision || abort.signal.aborted) return;
                merged.push(...data.entries);
                const gaps = Array.isArray(data.gaps) ? data.gaps.map((gap) => GAP_WORDS[gap] || String(gap)) : [];
                const skipped = Number.isSafeInteger(data.skipped_rows) ? data.skipped_rows : 0;
                if (skipped > 0 || gaps.length) damaged.push(`${name} ${skipped}${gaps.length ? ` (${gaps.join(', ')})` : ''}`);
            } catch {
                if (revision !== backfillRevision || abort.signal.aborted) return;
                failed.push(name);
            }
        }
        if (revision !== backfillRevision || abort.signal.aborted) return;
        backfillAbort = null;
        const tsOf = (evt) => Date.parse(evt?.ts || evt?.timestamp || '') || 0;
        merged.sort((a, b) => tsOf(a) - tsOf(b));
        batching = true;
        try { for (const evt of merged) addLogEntry(evt); } finally { batching = false; holders = null; }
        sortByTime();
        historyLoading = false;
        paint();
        if (revision === backfillRevision) {
            const status = [
                failed.length && `Recent history is incomplete: ${failed.join(', ')} could not be loaded. Live events are shown as they arrive; reconnect reloads recent history.`,
                damaged.length && `Damaged records skipped in the recent history read: ${damaged.join('; ')}.`,
            ].filter(Boolean).join(' ');
            setInlineStatus(historyStatus, status, failed.length ? 'error' : 'warn');
            historyStatus.hidden = !status;
        }
    }

    ws.on('log', (msg) => {
        if (msg.data) addLogEntry(msg.data);
    });

    // Backfill on every (re)connect so history missed while disconnected is
    // recovered (now if the panel is shown, else when it is); the dedupe guard
    // keeps the overlap single.
    ws.on('open', () => {
        backfillOwed = true;
        if (isLogsVisible()) void backfillRecentLogs();
    });

    page.querySelector('#btn-clear-logs').addEventListener('click', () => {
        const cancelledRead = Boolean(backfillAbort);
        backfillAbort?.abort(); backfillAbort = null; backfillRevision += 1;
        backfillOwed = false;
        historyLoading = false;
        // Clear list empties this view, not the evidence that its history could not be read.
        if (cancelledRead) {
            setInlineStatus(historyStatus, 'History read cancelled. Live events continue.', 'warn');
            historyStatus.hidden = false;
        }
        raw.remove(logEntries); removedNodes.clear();
        for (const entry of entries.values()) entry.removed = true;
        entries.clear();
        stale.clear();
        duplicateState.clear();
        taskGroups.clear();
        logEntries.innerHTML = '';
        clearedAt = new Date().toLocaleTimeString('en-GB', { hour: '2-digit', minute: '2-digit' });
        updateEmptyState();
        follow();
    });

    page.addEventListener('scroll', noteReader, { passive: true });
    logEntries.addEventListener('click', (event) => {
        if (raw.click(event.target, event)) return;
        // A Timeline about to open draws its rows first, so they are there when it opens.
        const fold = event.target?.closest?.('summary')?.parentElement;
        const entry = fold && entryOf.get(fold.parentElement);
        if (entry?.kind === 'group' && fold === entry.dom.fold && !fold.open) renderTaskTimeline(entry);
    });
    // `toggle` does not bubble: caught on its way down. An event's record opens and closes with its fold (logs_raw.js),
    // its whole words with it; a task's Timeline draws its rows open and lets them go closed.
    logEntries.addEventListener('toggle', (event) => {
        const fold = event.target;
        if (!fold?.classList?.contains('log-fold')) return;
        if (raw.toggled(fold)) {
            if (fold.open) showWhole(fold.parentElement);
            return;
        }
        const entry = entryOf.get(fold.parentElement);
        if (entry?.kind !== 'group') return;
        if (fold.open) renderTaskTimeline(entry);
        else closeTaskTimeline(entry);
    }, true);
    document.addEventListener('selectionchange', () => {
        const selection = document.getSelection?.();
        selectionInside = Boolean(selection && !selection.isCollapsed
            && (logEntries.contains(selection.anchorNode) || logEntries.contains(selection.focusNode)));
    });

    window.addEventListener('ouro:dashboard-subtab-shown', show);

    // A `.page` display round-trip destroys the scroll position (documented in
    // chat.js: WebKit re-shows a hidden column at scrollTop=0). Returning to the
    // dashboard is the same class of navigation as a subtab switch (evolution.js
    // listens to both events the same way): the panel catches up and the reader's
    // place is set again, the newest entry or the one being read.
    window.addEventListener('ouro:page-shown', show);
    document.addEventListener('visibilitychange', show);
    show();
}
