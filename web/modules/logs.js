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
    prettyLogEvent,
    summarizeLogEvent,
} from './log_events.js';
import { outcomeTone } from './task_phase_chip.js';

export function initLogs({ ws, state, mount }) {
    const MAX_LOGS = 500;
    const MAX_TASK_EVENTS = 30;
    const duplicateWindowMs = 5000;
    const duplicateState = new Map();
    const taskGroups = new Map();
    // Exact-duplicate guard so a backfilled row and its live-stream twin (or a
    // reconnect re-backfill) render once. Bounded; a rare full clear briefly
    // disables the guard, which at worst re-shows one row — never drops one.
    const renderedLogKeys = new Set();
    let backfillRevision = 0;

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
    let historyLoading = false;
    let clearedAt = '';
    function isLogsVisible() {
        return state.activePage === 'dashboard' && state.dashboardActiveSubtab === 'logs';
    }

    function scrollToLatest() {
        if (!isLogsVisible()) return;
        requestAnimationFrame(() => {
            page.scrollTop = page.scrollHeight;
        });
    }

    // Live entries must not yank the view away from history the reader
    // scrolled up to: per-entry autoscroll sticks only while the reader is
    // already at the bottom. Callers capture this BEFORE appending — the
    // append itself grows scrollHeight, which would misread a pinned reader
    // as scrolled-up. Deliberate jumps (filter toggle, tab switch, backfill)
    // stay unconditional. A hidden panel measures 0/0/0 → pinned, matching
    // the tab-switch jump the reader gets on return.
    const PINNED_SLACK_PX = 48;
    function isPinnedToLatest() {
        return page.scrollHeight - page.scrollTop - page.clientHeight <= PINNED_SLACK_PX;
    }

    function updateVisibility(entry) {
        entry.hidden = !state.activeFilters[entry.dataset.category];
    }

    // One line says why the list shows nothing (U033): every event filtered out (with Show all), the owner's
    // Clear, or no event yet; nothing while the history is still loading (its own line says so).
    function updateEmptyState() {
        const entries = [...logEntries.children];
        const quiet = entries.some((entry) => !entry.hidden) || (historyLoading && !entries.length);
        emptyLine.hidden = quiet;
        if (quiet) return;
        if (entries.length) {
            emptyLine.innerHTML = 'The filters hide every event. <button type="button" class="btn btn-ghost btn-sm" data-log-show-all>Show all</button>';
        } else {
            emptyLine.textContent = clearedAt ? `Cleared at ${clearedAt}. New events appear here.` : 'No events yet. New events appear here.';
        }
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
        logEntries.querySelectorAll('.log-entry').forEach(updateVisibility);
        updateEmptyState();
        scrollToLatest();
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

    // History arrives after the live stream has started: every entry takes its place by its own time, so the list
    // reads oldest to newest whatever order the two sources delivered (U035). A live event lands at the end at once.
    function placeEntry(entry, time) {
        entry.dataset.logTime = String(time);
        entry.remove?.();
        const nodes = [...logEntries.children];
        let index = nodes.length;
        while (index > 0 && Number(nodes[index - 1].dataset.logTime) > time) index -= 1;
        if (index === nodes.length) logEntries.appendChild(entry);
        else logEntries.insertBefore(entry, nodes[index]);
    }

    function trimEntries() {
        while (logEntries.children.length > MAX_LOGS) {
            const first = logEntries.firstElementChild;
            if (!first) break;
            const removeKey = [...duplicateState.entries()].find(([, tracked]) => tracked.entry === first)?.[0];
            if (removeKey) duplicateState.delete(removeKey);
            if (first.dataset.taskGroup) taskGroups.delete(first.dataset.taskGroup);
            first.remove();
        }
    }

    function metaPills(meta) {
        if (!meta.length) return '';
        return `<div class="log-meta">${meta.map((item) => `<span class="log-pill">${escapeHtml(item)}</span>`).join('')}</div>`;
    }

    // An event's one first line (U027, U028): its time, its type only when the headline does not already say it
    // (a task card's kind always: it changes with the card), its level (the default `info` says nothing, CSS hides
    // it), the headline, the repeat count and, for an event with a record, Raw at the line's end.
    function logMainHtml({ ts = '', type = '', phase = 'info', headline = 'Event', repeat = '', attrs = {}, raw = false }) {
        const attr = (key) => attrs[key] ? ` ${attrs[key]}` : '';
        const showType = type && (attrs.type || (type.label && type.label !== headline));
        return `
            <div class="log-main">
                <span class="log-ts"${attr('ts')}>${escapeHtml(ts)}</span>
                ${showType ? `<span class="log-type ${escapeHtml(type.className || '')}"${attr('type')}>${escapeHtml(type.label || '')}</span>` : ''}
                <span class="log-phase ${escapeHtml(phase)}"${attr('phase')}>${escapeHtml(phase)}</span>
                <span class="log-headline"${attr('headline')}>${escapeHtml(headline)}</span>
                <span class="log-repeat"${attr('repeat')}${repeat ? '' : ' hidden'}>${escapeHtml(repeat)}</span>
                ${raw ? '<button class="btn btn-ghost btn-sm log-raw-toggle" type="button" aria-expanded="false">Raw</button>' : ''}
            </div>
        `;
    }

    function bindRawToggle(root) {
        root.querySelectorAll('.log-raw-toggle').forEach((rawToggle) => {
            if (rawToggle.dataset.bound === '1') return;
            // The record follows the event's first line, after its facts.
            let rawEl = rawToggle.parentElement?.nextElementSibling;
            while (rawEl && !rawEl.classList.contains('log-raw')) rawEl = rawEl.nextElementSibling;
            if (!rawEl) return;
            rawToggle.dataset.bound = '1';
            rawToggle.addEventListener('click', () => {
                const isHidden = rawEl.hasAttribute('hidden');
                if (isHidden) {
                    rawEl.removeAttribute('hidden');
                    rawToggle.textContent = 'Hide raw';
                } else {
                    rawEl.setAttribute('hidden', '');
                    rawToggle.textContent = 'Raw';
                }
                rawToggle.setAttribute('aria-expanded', String(isHidden));
            });
        });
    }

    function createStandaloneEntry(evt) {
        const view = summarizeLogEvent(evt);
        const execution = executorChip(evt);
        const cat = categorizeLogEvent(evt, view);
        const dedupeKey = duplicateLogEventKey(evt);
        const now = (() => {
            const parsed = evt.ts ? Date.parse(evt.ts) : NaN;
            return Number.isFinite(parsed) ? parsed : Date.now();
        })();

        if (dedupeKey) {
            let last = duplicateState.get(dedupeKey);
            if (last && !logEntries.contains(last.entry)) {
                duplicateState.delete(dedupeKey);
                last = null;
            }
            if (last && now - last.ts <= duplicateWindowMs) {
                last.count += 1;
                last.ts = now;
                const repeatEl = last.entry.querySelector('.log-repeat');
                if (repeatEl) {
                    repeatEl.textContent = `x${last.count}`;
                    repeatEl.hidden = false;
                }
                const tsEl = last.entry.querySelector('.log-ts');
                if (tsEl) tsEl.textContent = normalizeLogTs(evt.ts || evt.timestamp);
                const rawEl = last.entry.querySelector('.log-raw');
                if (rawEl) rawEl.textContent = prettyLogEvent(evt);
                placeEntry(last.entry, now);
                updateVisibility(last.entry);
                updateEmptyState();
                return;
            }
        }

        const entry = document.createElement('div');
        entry.className = 'log-entry';
        entry.dataset.category = cat;
        const bodyHtml = view.body
            ? `<div class="log-body">${escapeHtml(view.body)}</div>`
            : '';
        entry.innerHTML = `
            ${logMainHtml({
                ts: normalizeLogTs(evt.ts || evt.timestamp),
                type: { className: cat, label: view.typeLabel },
                phase: view.phase || 'info',
                headline: view.headline || 'Event',
                raw: true,
            })}
            ${execution ? `<div class="harness-chip log-executor-chip" title="${escapeHtmlAttr(execution.title || '')}">${harnessIdentityMarkup(execution.harness, {
                label: execution.label || '',
                className: 'log-executor-identity',
            })}</div>` : ''}
            ${execution?.title ? `<div class="log-body log-executor-detail">${escapeHtml(execution.title)}</div>` : ''}
            ${metaPills(view.meta)}
            ${bodyHtml}
            <pre class="log-raw" hidden>${escapeHtml(prettyLogEvent(evt))}</pre>
        `;
        bindRawToggle(entry);
        updateVisibility(entry);
        const pinned = isPinnedToLatest();
        placeEntry(entry, now);

        if (dedupeKey) {
            duplicateState.set(dedupeKey, { entry, ts: now, count: 1 });
        }

        trimEntries();
        updateEmptyState();
        if (state.activeFilters[cat] && pinned) scrollToLatest();
    }

    function createTaskGroupCard(groupId, category, kindLabel) {
        const entry = document.createElement('div');
        entry.className = 'log-entry log-task-card';
        entry.dataset.category = category;
        entry.dataset.taskGroup = groupId;
        entry.innerHTML = `
            ${logMainHtml({
                type: { className: category, label: kindLabel },
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
        const record = {
            entry,
            ts: entry.querySelector('[data-task-ts]'),
            kind: entry.querySelector('[data-task-kind]'),
            phase: entry.querySelector('[data-task-phase]'),
            headline: entry.querySelector('[data-task-headline]'),
            count: entry.querySelector('[data-task-count]'),
            summary: entry.querySelector('[data-task-summary]'),
            executor: entry.querySelector('[data-task-executor]'),
            executorDetail: entry.querySelector('[data-task-executor-detail]'),
            review: entry.querySelector('[data-task-review]'),
            timeline: entry.querySelector('[data-task-timeline]'),
            events: 0,
            category,
            recent: [],
            terminal: false,
        };
        taskGroups.set(groupId, record);
        return record;
    }

    function renderTaskTimeline(record) {
        record.timeline.innerHTML = record.recent.map((item) => `
            <div class="log-task-event">
                ${logMainHtml({
                    ts: item.ts,
                    phase: item.phase || 'info',
                    headline: item.headline,
                    repeat: item.count > 1 ? `x${item.count}` : '',
                    raw: true,
                })}
                ${metaPills(item.meta)}
                ${item.body ? `<div class="log-body">${escapeHtml(item.body)}</div>` : ''}
                <pre class="log-raw" hidden>${escapeHtml(item.raw || '')}</pre>
            </div>
        `).join('');
        bindRawToggle(record.timeline);
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
        const kindLabel = wake ? 'Consciousness' : `${taskType || 'task'} ${String(groupId).slice(0, 8)}`;
        // Captured before ANY record mutation: an already-mounted card grows
        // in place (summary rewrite, review unhide, timeline render) before
        // the append below, and that growth alone can push a pinned reader
        // past the slack allowance.
        const pinned = isPinnedToLatest();
        const record = taskGroups.get(groupId) || createTaskGroupCard(groupId, category, kindLabel);
        const ts = normalizeLogTs(evt.ts || evt.timestamp);

        record.events += 1;
        record.category = category;
        record.wake = wake;
        record.taskType = taskType;
        record.entry.dataset.category = category;
        // The card's time is its newest event's: an older frame that history delivers later never dates it back.
        const time = Date.parse(evt.ts || evt.timestamp || '') || Date.now();
        const newest = Math.max(time, Number(record.entry.dataset.logTime) || 0);
        if (time >= newest) record.ts.textContent = ts;
        record.kind.textContent = kindLabel;
        record.kind.className = `log-type ${category}`;
        // The card's headline settles only on the typed task_done frame. The bus
        // serves control before telemetry, so a task's late tool or round frame
        // (its own `done`/`error` phase included) can arrive after its task_done:
        // it stays a timeline row and never reopens or rewrites the settled
        // headline. Until then the headline follows the latest frame.
        const settles = evt.type === 'task_done';
        if (settles || !record.terminal) {
            record.phase.textContent = view.phase || 'info';
            record.phase.className = `log-phase ${view.phase || 'info'}`;
            // A settled headline is the task's outcome and takes the cards' tone (Stopped neutral, Done with
            // warnings green); until then it is the latest event's phase in its level's colour.
            const tone = settles ? outcomeTone(view.phase) : '';
            if (tone) record.phase.dataset.chipTone = tone;
            else delete record.phase.dataset.chipTone;
            record.headline.textContent = view.headline || 'Task activity';
        }
        if (settles) record.terminal = true;
        record.count.textContent = `x${record.events}`;
        record.count.hidden = record.events <= 1;
        // The card's facts beside its kind; the ids and the type its kind and lineage already are stay in its
        // Timeline (U029).
        const lineage = new Set([`task=${groupId}`, `task=${evt.task_id || ''}`, `parent=${evt.parent_task_id || ''}`,
            `root=${evt.root_task_id || ''}`, taskType]);
        record.summary.innerHTML = metaPills(view.meta.filter((item) => !lineage.has(item)));
        const execution = executorChip(evt);
        if (execution && record.executor) {
            record.executor.title = execution.title || '';
            record.executor.innerHTML = harnessIdentityMarkup(execution.harness, {
                label: execution.label || '',
                className: 'log-executor-identity',
            });
            record.executor.hidden = false;
            // The chip stays quiet; its detail sentence is text, never only a title.
            record.executorDetail.textContent = execution.title || '';
            record.executorDetail.hidden = !execution.title;
        }
        const reviewDetails = formatReviewProjection(evt.review_projection);
        if (reviewDetails && record.review) {
            record.review.textContent = reviewDetails;
            record.review.hidden = false;
        }

        const last = record.recent[record.recent.length - 1];
        const dedupeKey = duplicateLogEventKey(evt);
        if (last && last.dupKey && last.dupKey === dedupeKey) {
            last.count += 1;
            last.ts = ts;
            last.meta = view.meta;
            last.body = view.body;
            last.raw = prettyLogEvent(evt);
        } else {
            record.recent.push({
                ts,
                phase: view.phase || 'info',
                headline: view.headline || 'Task event',
                meta: view.meta,
                body: view.body,
                raw: prettyLogEvent(evt),
                count: 1,
                dupKey: dedupeKey,
            });
            if (record.recent.length > MAX_TASK_EVENTS) record.recent.shift();
        }

        renderTaskTimeline(record);
        updateVisibility(record.entry);
        // A card stands where its newest event does; an older event (history) never moves it later.
        placeEntry(record.entry, newest);
        trimEntries();
        updateEmptyState();
        if (state.activeFilters[category] && pinned) scrollToLatest();
    }

    function addLogEntry(evt) {
        const ts = evt?.ts || evt?.timestamp || '';
        const type = evt?.type || '';
        // Dedupe on the natural identity (timestamp has microseconds, so distinct
        // events never collide; rapid same-type repeats differ by ts and still
        // reach the x-N collapse below). Only guard when both ts and type exist.
        const exactKey = (ts && type)
            ? `${ts}|${type}|${evt?.task_id || evt?.taskId || ''}|${evt?.execution_id || evt?.round || evt?.tool || ''}`
            : '';
        if (exactKey && renderedLogKeys.has(exactKey)) return;
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

    // Backfill recent history so the panel is not empty after a page load or
    // reconnect (the live WS stream only carries events from the current
    // connection). All dashboard-relevant streams are merged and replayed
    // oldest-first; addLogEntry's exact-duplicate guard collapses any overlap
    // with the live stream, so this neither drops the pre-connect window nor
    // double-renders the post-connect overlap.
    async function backfillRecentLogs() {
        const revision = ++backfillRevision;
        historyLoading = true;
        updateEmptyState();
        setInlineStatus(historyStatus, 'Loading recent history…');
        historyStatus.hidden = false;
        const merged = [];
        const failed = [];
        for (const name of ['events', 'tools', 'progress', 'supervisor']) {
            try {
                const resp = await apiFetch(`/api/logs/${name}?limit=150`, { cache: 'no-store' });
                if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
                const data = await resp.json();
                if (!Array.isArray(data?.entries)) throw new Error('History unavailable');
                merged.push(...data.entries);
            } catch {
                failed.push(name);
            }
        }
        const tsOf = (evt) => Date.parse(evt?.ts || evt?.timestamp || '') || 0;
        merged.sort((a, b) => tsOf(a) - tsOf(b));
        for (const evt of merged) addLogEntry(evt);
        if (revision === backfillRevision) {
            setInlineStatus(historyStatus, failed.length
                ? `Recent history is incomplete: ${failed.join(', ')} could not be loaded. Live events are shown as they arrive; reconnect reloads recent history.`
                : '', 'error');
            historyStatus.hidden = failed.length === 0;
            historyLoading = false;
            updateEmptyState();
        }
        scrollToLatest();
    }

    ws.on('log', (msg) => {
        if (msg.data) addLogEntry(msg.data);
    });

    // Backfill at init and on every (re)connect so history missed while
    // disconnected is recovered; the dedupe guard keeps the overlap single.
    backfillRecentLogs();
    ws.on('open', () => { backfillRecentLogs(); });

    page.querySelector('#btn-clear-logs').addEventListener('click', () => {
        duplicateState.clear();
        taskGroups.clear();
        logEntries.innerHTML = '';
        clearedAt = new Date().toLocaleTimeString('en-GB', { hour: '2-digit', minute: '2-digit' });
        updateEmptyState();
    });

    window.addEventListener('ouro:dashboard-subtab-shown', (event) => {
        if (event.detail?.tab === 'logs') scrollToLatest();
    });

    // A `.page` display round-trip destroys the scroll position (documented in
    // chat.js: WebKit re-shows a hidden column at scrollTop=0), which would
    // read as scrolled-up and silently disarm the pinned-only autoscroll.
    // Returning to the dashboard is the same class of deliberate navigation
    // as a subtab switch (evolution.js listens to both events the same way),
    // so re-pin on it; this also covers the very first dashboard show, where
    // the backfill's own jump ran while the panel was still hidden.
    window.addEventListener('ouro:page-shown', (event) => {
        if (event.detail?.page === 'dashboard' && state.dashboardActiveSubtab === 'logs') scrollToLatest();
    });
}
