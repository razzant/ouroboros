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
import { createLogRaw, LOG_RAW_HTML, LOG_RAW_TOGGLE_HTML } from './logs_raw.js';

/** Dashboard › Logs (docs/ARCHITECTURE.md §3 "Dashboard"). The panel's entries are data first: every live frame and
 * backfilled row updates its entry (task grouping, x-N collapse, the sticky Errors filter, the settled headline) whether
 * or not the panel is on screen, while the DOM follows only a shown panel. Hidden, the panel writes no node and reads no
 * history; shown again it places what changed newest first, RENDER_CHUNK entries a frame, so a burst that arrived
 * meanwhile never becomes one long rebuild. A task's timeline rows exist while its Timeline is open. Raw pins one
 * compact row's source or received event, pages asynchronously, and offers a full download (logs_raw.js). The reader keeps their place across
 * hide and show: following the newest entry, or on the entry they were reading. The compact history says what it left
 * out: the fields a row's view omits (on that row) and the damaged records a read skipped (in the status line).
 * The panel is one Dashboard tab body, its scroller; an event is one line in one card of trays (U026-U035). */

// The kinds of damaged record a compact read skips (`gaps`, ouroboros/gateway/logs.py); an unknown kind is shown as named.
const GAP_WORDS = { malformed_jsonl: 'malformed line', incomplete_live_line: 'unfinished live line', torn_archive_line: 'torn archive line' };

// What a compact row's view leaves out (`_raw.omitted_fields`, at most 64 names of `omitted_field_count`); Raw has it.
function omittedNote(evt) {
    const names = Array.isArray(evt?._raw?.omitted_fields) ? evt._raw.omitted_fields.map(String) : [];
    const count = Math.max(names.length, Number.isSafeInteger(evt?._raw?.omitted_field_count) ? evt._raw.omitted_field_count : 0);
    if (!count) return '';
    const more = count - names.length;
    const left = names.length ? `${names.join(', ')}${more ? ` and ${more} more` : ''}` : `${count} fields`;
    return `Left out of this view: ${left}. Raw reads the whole row.`;
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

    state.activeFilters = state.activeFilters || Object.fromEntries(
        Object.keys(LOG_CATEGORIES).map((key) => [key, true]),
    );

    // One Dashboard tab body is the tab's root and its one scroller; the filters stay at its top while the list,
    // one card of tray rows, scrolls under them. The history line and the empty line live in the list's card.
    const page = document.createElement('div');
    page.id = 'page-logs';
    page.className = 'dashboard-body settings-logs-panel';
    page.innerHTML = `
        <div class="logs-filters" id="log-filters"><button type="button" class="btn btn-ghost btn-sm logs-inline-clear" id="btn-clear-logs">Clear</button></div>
        <section class="ui-card logs-card">
            <div class="ui-status logs-history-status" role="status"></div>
            <p class="logs-empty" data-log-empty hidden></p>
            <div id="log-entries"></div>
        </section>
    `;
    mount.appendChild(page);

    const filtersDiv = page.querySelector('#log-filters');
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
        if (entry.node) entry.node.hidden = !state.activeFilters[entry.category];
        raw.exposure();
    }

    // One line says why the list shows nothing (U033): every event filtered out (with Show all), the owner's
    // Clear, or no event yet; nothing while the history is still loading (its own line says so). It reads the
    // entries, not the nodes a shown panel may still be placing, and is written only while the panel shows.
    let emptyWords = null;
    function updateEmptyState() {
        if (!isLogsVisible()) return;
        const quiet = [...entries.values()].some((entry) => state.activeFilters[entry.category])
            || (historyLoading && !entries.size);
        if (emptyLine.hidden !== quiet) emptyLine.hidden = quiet;
        if (quiet) return;
        const words = entries.size ? '' : `${clearedAt ? `Cleared at ${clearedAt}.` : 'No events yet.'} New events appear here.`;
        if (words === emptyWords) return;
        emptyWords = words;
        if (words) emptyLine.textContent = words;
        else emptyLine.innerHTML = 'The filters hide every event. <button type="button" class="btn btn-ghost btn-sm" data-log-show-all>Show all</button>';
    }

    // Filters narrow the list: while every one is on nothing is narrowed and no chip takes the chosen tone; once
    // one is off, the ones still on are the chosen ones (board fork "Logs filters": quiet chips, one tone).
    const chips = new Map();
    function syncFilterChips() {
        const narrowed = Object.keys(LOG_CATEGORIES).some((key) => !state.activeFilters[key]);
        chips.forEach((chip, key) => {
            chip.classList.toggle('active', narrowed && Boolean(state.activeFilters[key]));
            chip.setAttribute('aria-pressed', String(Boolean(state.activeFilters[key])));
        });
    }

    function applyFilters() {
        syncFilterChips();
        entries.forEach(updateVisibility);
        updateEmptyState();
        follow();
    }

    function renderFilters() {
        const inlineClear = filtersDiv.querySelector('.logs-inline-clear');
        filtersDiv.innerHTML = '';
        chips.clear();
        Object.entries(LOG_CATEGORIES).forEach(([key, cat]) => {
            const chip = document.createElement('button');
            chip.type = 'button';
            chip.className = 'filter-chip';
            chip.textContent = cat.label;
            chip.addEventListener('click', () => {
                state.activeFilters[key] = !state.activeFilters[key];
                applyFilters();
            });
            chips.set(key, chip);
            filtersDiv.appendChild(chip);
        });
        if (inlineClear) filtersDiv.appendChild(inlineClear);
        syncFilterChips();
    }

    emptyLine.addEventListener('click', (event) => {
        if (!event.target?.closest?.('[data-log-show-all]')) return;
        Object.keys(LOG_CATEGORIES).forEach((key) => { state.activeFilters[key] = true; });
        applyFilters();
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
        // The card's facts beside its kind; the ids and the type its kind and lineage already are stay in its
        // Timeline (U029).
        if (!behind) {
            const lineage = new Set([`task=${groupId}`, `task=${evt.task_id || ''}`, `parent=${evt.parent_task_id || ''}`,
                `root=${evt.root_task_id || ''}`, taskType]);
            record.meta = view.meta.filter((item) => !lineage.has(item));
        }
        const execution = executorChip(evt), review = formatReviewProjection(evt.review_projection);
        record.execution = (behind ? record.execution || execution : execution || record.execution);
        record.review = (behind ? record.review || review : review || record.review);

        const last = record.recent[record.recent.length - 1];
        const dedupeKey = duplicateLogEventKey(evt);
        // A row's body and what its view left out come from the same event (an adopted file source changes neither).
        if (!behind && last && last.dupKey && last.dupKey === dedupeKey) {
            Object.assign(last, { count: last.count + 1, ts, meta: view.meta, body: view.body, omitted: omittedNote(evt), evt, at,
                rev: last.rev + 1 });
        } else {
            const row = {
                ts,
                phase: view.phase || 'info',
                headline: view.headline || 'Task event',
                meta: view.meta,
                body: view.body,
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

    function metaPills(meta) {
        if (!meta.length) return '';
        return `<div class="log-meta">${meta.map((item) => `<span class="log-pill">${escapeHtml(item)}</span>`).join('')}</div>`;
    }

    // An event's one first line (U027, U028): its time, its type only when the headline does not already say it
    // (a task card's kind always: it changes with the card), its level (the default `info` says nothing, CSS hides
    // it), the headline, the repeat count and, for an event with a record, Raw at the line's end (logs_raw.js).
    function logMainHtml({ ts = '', type = '', phase = 'info', headline = 'Event', repeat = '', attrs = {}, hasRaw = false }) {
        const attr = (key) => attrs[key] ? ` ${attrs[key]}` : '';
        const showType = type && (attrs.type || (type.label && type.label !== headline));
        return `
            <div class="log-main">
                <span class="log-ts"${attr('ts')}>${escapeHtml(ts)}</span>
                ${showType ? `<span class="log-type ${escapeHtml(type.className || '')}"${attr('type')}>${escapeHtml(type.label || '')}</span>` : ''}
                <span class="log-phase ${escapeHtml(phase)}"${attr('phase')}>${escapeHtml(phase)}</span>
                <span class="log-headline"${attr('headline')}>${escapeHtml(headline)}</span>
                <span class="log-repeat"${attr('repeat')}${repeat ? '' : ' hidden'}>${escapeHtml(repeat)}</span>
                ${hasRaw ? LOG_RAW_TOGGLE_HTML : ''}
            </div>
        `;
    }

    const setText = (el, text) => { if (el && el.textContent !== text) el.textContent = text; };
    const setClass = (el, name) => { if (el && el.className !== name) el.className = name; };
    const setData = (el, key, value) => { if (el.dataset[key] !== value) el.dataset[key] = value; };
    const shownHtml = new WeakMap();  // markup a node was last given, so an unchanged one is not rewritten
    const setHtml = (el, html) => { if (el && shownHtml.get(el) !== html) { el.innerHTML = html; shownHtml.set(el, html); } };

    function renderStandalone(entry) {
        if (!entry.node) {
            const { view, execution } = entry;
            const node = document.createElement('div');
            node.className = 'log-entry';
            node.innerHTML = `
                ${logMainHtml({
                    ts: entry.tsText,
                    type: { className: entry.category, label: view.typeLabel },
                    phase: view.phase || 'info',
                    headline: view.headline || 'Event',
                    hasRaw: true,
                })}
                ${execution ? `<div class="harness-chip log-executor-chip" title="${escapeHtmlAttr(execution.title || '')}">${harnessIdentityMarkup(execution.harness, {
                    label: execution.label || '',
                    className: 'log-executor-identity',
                })}</div>` : ''}
                ${execution?.title ? `<div class="log-body log-executor-detail">${escapeHtml(execution.title)}</div>` : ''}
                ${metaPills(view.meta)}
                ${view.body ? `<div class="log-body">${escapeHtml(view.body)}</div>` : ''}
                ${entry.omitted ? `<div class="log-body" data-log-omitted>${escapeHtml(entry.omitted)}</div>` : ''}
                ${LOG_RAW_HTML}
            `;
            raw.bind(node, () => entry.evt);
            entry.node = node;
            entryOf.set(node, entry);
        }
        const { node } = entry;
        setData(node, 'category', entry.category);
        setData(node, 'logTime', String(entry.at));
        setText(node.querySelector('.log-ts'), entry.tsText);
        const repeat = node.querySelector('.log-repeat');
        setText(repeat, entry.repeat > 1 ? `x${entry.repeat}` : '');
        if (repeat) repeat.hidden = entry.repeat <= 1;
        raw.changed(node);
    }

    function createTaskGroupNode(entry) {
        const node = document.createElement('div');
        node.className = 'log-entry log-task-card';
        node.dataset.taskGroup = entry.groupId;
        node.innerHTML = `
            ${logMainHtml({
                type: { className: entry.category, label: entry.kindLabel },
                phase: 'info',
                headline: 'Task activity',
                attrs: {
                    ts: 'data-task-ts',
                    type: 'data-task-kind',
                    phase: 'data-task-phase',
                    headline: 'data-task-headline',
                    repeat: 'data-task-count',
                },
            })}
            <div class="log-task-summary" data-task-summary></div>
            <div class="harness-chip log-executor-chip" data-task-executor hidden></div>
            <div class="log-body log-executor-detail" data-task-executor-detail hidden></div>
            <div class="log-body log-task-review" data-task-review hidden></div>
            <details class="ui-disclosure log-task-details">
                <summary>Timeline</summary>
                <div class="log-task-timeline" data-task-timeline></div>
            </details>
        `;
        const find = (selector) => node.querySelector(selector);
        entry.dom = {
            ts: find('[data-task-ts]'),
            kind: find('[data-task-kind]'),
            phase: find('[data-task-phase]'),
            headline: find('[data-task-headline]'),
            count: find('[data-task-count]'),
            summary: find('[data-task-summary]'),
            executor: find('[data-task-executor]'),
            executorDetail: find('[data-task-executor-detail]'),
            review: find('[data-task-review]'),
            details: find('details'),
            timeline: find('[data-task-timeline]'),
        };
        entry.rows = new Map();
        entry.node = node;
        entryOf.set(node, entry);
    }

    function renderTaskGroup(entry) {
        if (!entry.node) createTaskGroupNode(entry);
        const { dom } = entry;
        setData(entry.node, 'category', entry.category);
        // A card stands where its newest event does (`at`); an older event never dates it back.
        setData(entry.node, 'logTime', String(entry.at));
        setText(dom.ts, entry.tsText);
        setText(dom.kind, entry.kindLabel);
        setClass(dom.kind, `log-type ${entry.category}`);
        setText(dom.phase, entry.phase);
        setClass(dom.phase, `log-phase ${entry.phase}`);
        if (entry.tone) dom.phase.dataset.chipTone = entry.tone;
        else delete dom.phase.dataset.chipTone;
        setText(dom.headline, entry.headline);
        setText(dom.count, `x${entry.events}`);
        dom.count.hidden = entry.events <= 1;
        setHtml(dom.summary, metaPills(entry.meta));
        const execution = entry.execution;
        if (execution) {
            if (dom.executor.title !== (execution.title || '')) dom.executor.title = execution.title || '';
            setHtml(dom.executor, harnessIdentityMarkup(execution.harness, {
                label: execution.label || '',
                className: 'log-executor-identity',
            }));
            dom.executor.hidden = false;
            // The chip stays quiet; its detail sentence is text, never only a title.
            setText(dom.executorDetail, execution.title || '');
            dom.executorDetail.hidden = !execution.title;
        }
        if (entry.review) {
            setText(dom.review, entry.review);
            dom.review.hidden = false;
        }
        if (dom.details.open) renderTaskTimeline(entry);
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
                row = document.createElement('div');
                row.className = 'log-task-event';
                rows.set(item, row);
                entry.dom.timeline.insertBefore(row, next);
            }
            next = row;
            if (row.dataset.rev === String(item.rev)) continue;
            if (!row.dataset.rev) {
                row.innerHTML = `
                    ${logMainHtml({ ts: item.ts, phase: item.phase || 'info', headline: item.headline, hasRaw: true })}
                    <div data-row-meta></div><div class="log-body"></div><div class="log-body" data-log-omitted hidden></div>
                    ${LOG_RAW_HTML}`;
                raw.bind(row, () => item.evt);
            }
            setText(row.querySelector('.log-ts'), item.ts);
            const repeat = row.querySelector('.log-repeat');
            setText(repeat, item.count > 1 ? `x${item.count}` : '');
            repeat.hidden = item.count <= 1;
            setHtml(row.querySelector('[data-row-meta]'), metaPills(item.meta));
            setText(row.querySelector('.log-body'), item.body || '');
            const omitted = row.querySelector('[data-log-omitted]');
            setText(omitted, item.omitted || '');
            omitted.hidden = !item.omitted;
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

    renderFilters();

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
        // Clear empties this view, not the evidence that its history could not be read.
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
        const details = event.target?.closest?.('summary')?.parentElement;
        const entry = details?.classList?.contains('log-task-details') && entryOf.get(details.closest('.log-entry'));
        if (entry && !details.open) renderTaskTimeline(entry);
    });
    // `toggle` does not bubble: caught on its way down. Closed, a Timeline lets its rows go.
    logEntries.addEventListener('toggle', (event) => {
        const details = event.target;
        const entry = details?.classList?.contains('log-task-details') && entryOf.get(details.closest('.log-entry'));
        if (!entry) return;
        if (details.open) renderTaskTimeline(entry);
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
