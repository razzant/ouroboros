import { escapeHtml, escapeHtmlAttr, formatMsgTime, machineWords } from './utils.js';
import { apiFetch } from './api_client.js';
import { setInlineStatus } from './ui_helpers.js';
import { harnessIdentityMarkup } from './harness_presentation.js';
import { plainActivityText, selectionInside as selectionWithin } from './chat_activity.js';
import {
    ERROR_LOG_PHASES as FAILED_PHASES,
    LOG_CATEGORIES,
    categorizeLogEvent,
    duplicateLogEventKey,
    executorChip,
    formatReviewProjection,
    getLogTaskGroupId,
    isGroupedTaskEvent,
    summarizeLogEvent,
    serviceOutputLinksHtml,
    taskPresentation,
} from './log_events.js';
import { markView, statusMarkHtml } from './status_mark.js';
import { bindScrollFade } from './scroll_fade.js';
import { revealStripChoice } from './page_header.js';
import { createLogRaw, LOG_RAW_HTML } from './logs_raw.js';

/** Dashboard › Logs (docs/ARCHITECTURE.md §3 "Dashboard"). The panel's entries are data first: every live frame and
 * backfilled row updates its entry (task grouping, ×N collapse, the sticky Errors filter, the settled headline) whether
 * or not the panel is on screen, while the DOM follows only a shown panel. Hidden, the panel writes no node and reads no
 * history; shown again it places what changed newest first, RENDER_CHUNK entries a frame, so a burst that arrived
 * meanwhile never becomes one long rebuild. Every event is one shared fold whose summary is its head: a task's opens
 * its timeline, whose rows exist while it is open; an event's opens its facts and its record, which pins one compact
 * row's source or received event while open, pages asynchronously and offers a full download (logs_raw.js). The reader
 * keeps their place across hide and show: following the newest entry, or on the entry they were reading. The compact
 * history says what it left out: the fields a row's view omits (in its record) and the damaged records a read skipped
 * (in the status line). The panel is one Dashboard tab body, its scroller; an event is one tray of the list's one card
 * (U026-U035). Only a failure is coloured: the shared outcome chip, beside the failure's own words. */

// The kinds of damaged record a compact read skips (`gaps`, ouroboros/gateway/logs.py); an unknown kind is shown as named.
const GAP_WORDS = { malformed_jsonl: 'malformed line', incomplete_live_line: 'unfinished live line', torn_archive_line: 'torn archive line' };

// An event's time, said by this one function: Main's clock at its head's end (utils.js formatMsgTime), its tooltip the
// exact moment to the second, as a Main message's time carries its date (chat_text_bubble.js), which leads its facts.
const timeOf = (ts) => formatMsgTime(ts) || { short: '', exact: '' };
// A task's id said short: its first eight characters, past the `presence-` its live turns' ids begin with.
const shortTaskId = (id) => String(id || '').replace(/^presence-/, '').slice(0, 8);
// Words an author wrote (narration, an executor's message) read as text, without Markdown's marks (chat_activity.js).
const wordsOf = (view, text) => (view.markdown && text ? plainActivityText(text) : text);

// What a compact row's view leaves out (`_raw.omitted_fields`, at most 64 names of `omitted_field_count`); its record has it.
function omittedNote(evt) {
    const names = Array.isArray(evt?._raw?.omitted_fields) ? evt._raw.omitted_fields.map(String) : [];
    const count = Math.max(names.length, Number.isSafeInteger(evt?._raw?.omitted_field_count) ? evt._raw.omitted_field_count : 0);
    if (!count) return '';
    const more = count - names.length;
    const left = names.length ? `${names.join(', ')}${more ? ` and ${more} more` : ''}` : `${count} fields`;
    return `Left out of this view: ${left}. The record below reads the whole row.`;
}

export function initLogs({ ws, state, mount, setControls, setBeforePageLeave }) {
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

    // The list is one plain scroller. Its filter rail and Clear belong to the section's header; moving the nodes
    // keeps their handlers and counts. The history line and empty line belong to the card.
    const page = document.createElement('div');
    page.id = 'page-logs';
    page.className = 'dashboard-body settings-logs-panel scroll-fade-y';
    page.innerHTML = `
        <div class="logs-filters"><div class="logs-kinds app-tab-strip" id="log-filters" role="group" aria-label="Show" data-scroll-x>${
            Object.entries({ '': 'All', ...LOG_CATEGORIES }).map(([key, label]) => `<button type="button" class="filter-chip skills-category" data-log-kind="${key}">${label}<span class="skills-category-count"></span></button>`).join('')
        }</div><button type="button" class="btn btn-default logs-clear-list" id="btn-clear-logs">Clear</button></div>
        <section class="ui-card logs-card">
            <div class="ui-status logs-history-status" role="status"></div>
            <p class="logs-empty" data-log-empty hidden></p>
            <div id="log-entries"></div>
        </section>
    `;
    mount.appendChild(page);
    bindScrollFade(page);  // the Logs panel is mounted after the app-wide pass; its newest end is hidden at first paint

    const filtersDiv = page.querySelector('#log-filters');
    bindScrollFade(filtersDiv, { axis: 'x' });  // a phone's one scrolling line fades where chips hide
    const logEntries = page.querySelector('#log-entries');
    const historyStatus = page.querySelector('.logs-history-status');
    const emptyLine = page.querySelector('[data-log-empty]');
    // The section's sheets decide whether its page column is drawn (a one-column list hides it): a panel with no box
    // is not shown, whatever its tab.
    function isLogsVisible() {
        return state.activePage === 'dashboard' && state.dashboardActiveSubtab === 'logs' && !document.hidden
            && page.getClientRects().length > 0;
    }

    const raw = createLogRaw({ visible: isLogsVisible });
    setBeforePageLeave?.(() => { if (isLogsVisible()) raw.rememberSelection(); });

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
        syncFilterChips();
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

    // Skills' chips: All leads, lit while no kind is chosen; a kind pressed is lit beside the others chosen, All pressed
    // clears the choice. Each counts the entries held of it ("Errors 2"), one with none hides unless chosen; while the
    // first history read is in flight with nothing held, no count is said and nothing hides.
    const chips = new Map([...filtersDiv.querySelectorAll('[data-log-kind]')].map((chip) => [chip.dataset.logKind, chip]));
    function syncFilterChips() {
        const known = !historyLoading || entries.size > 0;
        const counts = { '': entries.size };
        for (const entry of entries.values()) counts[entry.category] = (counts[entry.category] || 0) + 1;
        chips.forEach((chip, key) => {
            const lit = key ? Boolean(state.activeFilters[key]) : !narrowed();
            chip.classList.toggle('active', lit);
            chip.setAttribute('aria-pressed', String(lit));
            setText(chip.querySelector('.skills-category-count'), known ? String(counts[key] || 0) : '');
            chip.hidden = known && Boolean(key) && !counts[key] && !lit;
        });
    }

    function choose(key) {
        if (key) state.activeFilters[key] = !state.activeFilters[key];
        else KINDS.forEach((kind) => { state.activeFilters[kind] = false; });
        entries.forEach(updateVisibility);
        updateEmptyState();
        follow();
        revealStripChoice(filtersDiv, chips.get(key));
    }

    chips.forEach((chip, key) => chip.addEventListener('click', () => choose(key)));

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
        const ts = evt.ts || evt.timestamp || '';
        const now = Date.parse(ts) || Date.now();  // its own time, as a card's (updateTaskGroupCard), else now

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
                if (now >= last.ts) Object.assign(last.entry, { evt, ts, at: now });
                last.ts = Math.max(last.ts, now);
                touch(last.entry);
                return;
            }
        }

        const view = summarizeLogEvent(evt);
        // An event of a task with no card of its own says its task after its name, as a card does: runs side by side part.
        const entry = addEntry({
            kind: 'standalone', evt, view, ts, repeat: 1, dedupeKey, at: now, omitted: omittedNote(evt),
            category: categorizeLogEvent(evt, view), execution: executorChip(evt), task: shortTaskId(evt.subagent_task_id || evt.task_id),
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
        // A task is named by its type in words and its short id, as every other row of the log names one (U030); the
        // type, once seen, stays the card's.
        const taskType = String(earlier?.taskType || evt.task_type || '').trim();
        const record = earlier || createTaskGroupCard(groupId);
        const ts = evt.ts || evt.timestamp || '';
        const at = Date.parse(ts) || Date.now();
        // A backfilled row older than what this card already shows joins its timeline in place and leaves the rest.
        const behind = batching && earlier && at < record.at;

        record.events += 1;
        record.category = category;
        record.wake = wake;
        record.taskType = taskType;
        record.name = wake ? 'Consciousness' : machineWords(taskType || 'task');
        // The card's time is its newest event's: an older frame, live or from history, never dates it back.
        if (at >= record.at) { record.ts = ts; record.at = at; }
        // The card's headline settles only on the typed task_done frame. The bus
        // serves control before telemetry, so a task's late tool or round frame
        // (its own `done`/`error` phase included) can arrive after its task_done:
        // it stays a timeline row and never reopens or rewrites the settled
        // headline. Until then the headline follows the latest frame.
        const settles = evt.type === 'task_done';
        // A task's words are its latest event's (a settled task's are its outcome). While it holds a failure they are its
        // newest failure's own words under every filter (its error, or its name where it has none; a failed ending's
        // name is its chip), kept even after the failure's row leaves the timeline.
        if (settles || (!record.terminal && !behind)) record.words = wordsOf(view, view.headline || view.body) || 'Task activity';
        if (settles) { record.terminal = true; record.failed = view.phase === 'error'; }
        const failed = FAILED_PHASES.has(view.phase);
        if (failed && !settles) record.failures = (record.failures || 0) + 1;
        if (failed && at >= (record.failure?.at || 0)) record.failure = { at, words: wordsOf(view, view.body) || (settles ? '' : view.headline) };
        const execution = executorChip(evt), review = formatReviewProjection(evt.review_projection);
        record.execution = (behind ? record.execution || execution : execution || record.execution);
        record.review = (behind ? record.review || review : review || record.review);

        const last = record.recent[record.recent.length - 1];
        const dedupeKey = duplicateLogEventKey(evt);
        // A row's view and what its view left out come from the same event (an adopted file source changes neither).
        if (!behind && last && last.dupKey && last.dupKey === dedupeKey) {
            Object.assign(last, { count: last.count + 1, ts, view, omitted: omittedNote(evt), evt, at, rev: last.rev + 1 });
        } else {
            const row = { ts, view, omitted: omittedNote(evt), evt, at, count: 1, dupKey: dedupeKey, rev: 0 };
            const later = behind ? record.recent.findIndex((item) => item.at > at) : -1;
            if (later >= 0) record.recent.splice(later, 0, row);
            else record.recent.push(row);
            if (record.recent.length > MAX_TASK_EVENTS) record.recent.shift();
        }
        if (behind) stale.add(record);
        else touch(record);
    }

    function createTaskGroupCard(groupId) {
        const record = { kind: 'group', groupId, events: 0, recent: [], terminal: false, at: 0 };
        taskGroups.set(groupId, record);
        return addEntry(record);
    }

    // ---- The panel (DOM): only while shown ------------------------------------------------------------------------

    // One anatomy for every event, a task card and a timeline row alike (U027, U028): one shared fold
    // (details.ui-disclosure) whose summary is its head: its name right after the caret, its task's short id and its count,
    // its time at the line's end as a sidebar row's; then its line of words from the name's edge, at most three lines, led
    // by a failure's chip. Open, the words are whole (their whole text where the view cut them) over its facts and its
    // record, or a task's timeline. A head without a name says its line in the name's place (style.css).
    const HEAD_HTML = '<summary class="log-head"><span class="log-title"><span class="log-name"></span>'
        + '<span class="log-task" hidden></span><span class="log-count" hidden></span></span><time class="log-ts"></time>'
        + '<span class="log-line" hidden><span class="chat-live-phase log-failure" data-chip-tone="failed" hidden>'
        + `${statusMarkHtml('', markView({ outcome: 'error' }))}<span class="log-failure-word"></span></span>`
        + '<span class="log-words"></span></span></summary>'
        + '<div class="log-facts" hidden></div><div class="log-note log-executor-detail" hidden></div>';
    const RECORD_HTML = `<div class="log-note" data-log-omitted hidden></div>${LOG_RAW_HTML}`;
    const TIMELINE_HTML = '<div class="log-note log-task-review" hidden></div><div class="log-task-timeline"></div>';
    const partsOf = new WeakMap();  // an event's node → its parts, found once when it is made
    const wordsOfNode = new WeakMap();  // an event's node → its words and their whole text, where its view cut them

    function createEvent(className, body) {
        const node = document.createElement('details');
        node.className = `ui-disclosure log-fold ${className}`;
        node.innerHTML = HEAD_HTML + body;
        const find = (selector) => node.querySelector(selector);
        partsOf.set(node, { ts: find('.log-ts'), title: find('.log-title'), name: find('.log-name'), task: find('.log-task'),
            count: find('.log-count'), line: find('.log-line'), failure: find('.log-failure'), failureWord: find('.log-failure-word'),
            words: find('.log-words'), facts: find('.log-facts'), detail: find('.log-executor-detail'), omitted: find('[data-log-omitted]') });
        return node;
    }

    const setText = (el, text) => { if (el && el.textContent !== text) el.textContent = text; };
    const setAttr = (el, key, value) => { if (el.getAttribute(key) !== value) el.setAttribute(key, value); };
    const shownHtml = new WeakMap();  // markup a node was last given, so an unchanged one is not rewritten
    const setHtml = (el, html) => { if (el && shownHtml.get(el) !== html) { el.innerHTML = html; shownHtml.set(el, html); } };
    const plural = (n, word) => `${n} ${word}${n === 1 ? '' : 's'}`;

    // An event's own head and fold: its name, a failure's chip (the outcome family's word), its words, and its facts
    // after the exact moment.
    const eventParts = (view, ts) => ({ ts, name: view.headline, failure: FAILED_PHASES.has(view.phase) ? taskPresentation(view.phase).headline : '',
        words: wordsOf(view, view.body), whole: wordsOf(view, view.full), facts: [timeOf(ts).exact, ...view.meta],
        serviceStop: view.serviceStop });

    // An event's parts, each written only when it changed. A delegated executor is a quiet chip among its facts, and its
    // detail sentence is text in the fold (#360 A).
    function paintEvent(node, { ts = '', name = '', task = '', failure = '', count = '', words = '', whole = '', facts = [],
        execution = null, omitted = '', serviceStop = null }) {
        const parts = partsOf.get(node), time = timeOf(ts);
        setText(parts.ts, time.short);
        setAttr(parts.ts, 'datetime', ts);
        setAttr(parts.ts, 'title', time.exact);
        setText(parts.name, name);
        setText(parts.task, task);
        parts.task.hidden = !task;
        setText(parts.count, count);
        parts.count.hidden = !count;
        parts.title.hidden = !name && !task && !count;
        setText(parts.failureWord, failure);
        parts.failure.hidden = !failure;
        wordsOfNode.set(node, { words, whole });
        showWords(node);
        parts.line.hidden = !words && !failure;
        const chip = execution ? `<span class="harness-chip log-executor-chip" title="${escapeHtmlAttr(execution.title || '')}">${
            harnessIdentityMarkup(execution.harness, { label: execution.label || '', className: 'log-executor-identity' })}</span>` : '';
        const said = facts.filter(Boolean);
        const links = serviceOutputLinksHtml(serviceStop?.links);
        node.classList.toggle('service-stop-result', Boolean(serviceStop));
        setHtml(parts.facts, chip + escapeHtml(said.join(' · ')) + links);
        parts.facts.hidden = !chip && !said.length && !links;
        setText(parts.detail, execution?.title || '');
        parts.detail.hidden = !execution?.title;
        if (parts.omitted) { setText(parts.omitted, omitted); parts.omitted.hidden = !omitted; }
    }

    // Open, an event's words are whole: the CSS clamp lets go, and words its view cut read in full.
    function showWords(node) {
        const { words = '', whole = '' } = wordsOfNode.get(node) || {};
        setText(partsOf.get(node).words, node.open && whole ? whole : words);
    }

    function renderStandalone(entry) {
        if (!entry.node) {
            entry.node = createEvent('log-entry', RECORD_HTML);
            raw.bind(entry.node, () => entry.evt);
            entryOf.set(entry.node, entry);
        }
        const { node, view } = entry;
        setAttr(node, 'data-category', entry.category);
        setAttr(node, 'data-log-time', String(entry.at));
        paintEvent(node, { ...eventParts(view, entry.ts), task: entry.task, count: entry.repeat > 1 ? `×${entry.repeat}` : '',
            execution: entry.execution, omitted: entry.omitted });
        raw.changed(node);
    }

    function renderTaskGroup(entry) {
        if (!entry.node) {
            const node = entry.node = createEvent('log-entry log-task-card', TIMELINE_HTML);
            node.dataset.taskGroup = entry.groupId;
            entry.dom = { review: node.querySelector('.log-task-review'), timeline: node.querySelector('.log-task-timeline') };
            entry.rows = new Map();
            entryOf.set(node, entry);
        }
        const { node, dom } = entry;
        setAttr(node, 'data-category', entry.category);
        // A card stands where its newest event does (`at`); an older event never dates it back.
        setAttr(node, 'data-log-time', String(entry.at));
        // A task that failed says the outcome's word; one that holds failed steps while it works says how many, never
        // Failed (DESIGN §4: a local failure does not relabel a working task). Its count is its events, not repeats.
        const failed = entry.failed ? taskPresentation('error').headline : entry.failures ? `${plural(entry.failures, 'step')} failed` : '';
        paintEvent(node, { ts: entry.ts, name: entry.name, task: shortTaskId(entry.groupId), failure: failed,
            count: plural(entry.events, 'event'), words: failed ? entry.failure.words : entry.words, execution: entry.execution });
        // The task's review sentence opens with its timeline.
        setText(dom.review, entry.review || '');
        dom.review.hidden = !entry.review;
        if (node.open) renderTaskTimeline(entry);
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
                row = createEvent('log-task-event', RECORD_HTML);
                raw.bind(row, () => item.evt);
                rows.set(item, row);
                entry.dom.timeline.insertBefore(row, next);
            }
            next = row;
            if (row.dataset.rev === String(item.rev)) continue;
            paintEvent(row, { ...eventParts(item.view, item.ts), count: item.count > 1 ? `×${item.count}` : '', omitted: item.omitted });
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
        if (evt?.checkpoint_kind === 'service_stop' && evt.task_id && evt.service_stop?.launch_id && evt.card_row_revision)
            return `service-stop|${evt.task_id}|${evt.service_stop.launch_id}|${evt.card_row_revision}`;
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
        clearedAt = timeOf(new Date().toISOString()).short;
        updateEmptyState();
        follow();
    });

    page.addEventListener('scroll', noteReader, { passive: true });
    logEntries.addEventListener('click', (event) => {
        if (raw.click(event.target, event)) return;
        const head = event.target?.closest?.('summary');
        // A press that selected words of a head was reading them: it neither opens nor closes the event (DESIGN §5).
        if (head && event.detail && selectionWithin(head)) { event.preventDefault(); return; }
        // A Timeline about to open draws its rows first, so they are there when it opens.
        const entry = head && entryOf.get(head.parentElement);
        if (entry?.kind === 'group' && !head.parentElement.open) renderTaskTimeline(entry);
    });
    // `toggle` does not bubble: caught on its way down. An event's words are whole while it is open, and its record
    // opens and closes with it (logs_raw.js); a task's timeline draws its rows open and lets them go closed.
    logEntries.addEventListener('toggle', (event) => {
        const fold = event.target;
        if (!fold?.classList?.contains('log-fold')) return;
        showWords(fold);
        if (raw.toggled(fold)) return;
        const entry = entryOf.get(fold);
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
    setControls?.('logs', page.querySelector('.logs-filters'));
    show();
    return { captureReading() {
        const saved = { pinned, anchor: pinned ? null : captureAnchor() };
        return () => { pinned = saved.pinned; anchor = saved.anchor; settleSoon(); };
    } };
}
