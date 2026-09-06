import { escapeHtml } from './utils.js';
import { apiFetch } from './api_client.js';
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

    state.activeFilters = state.activeFilters || Object.fromEntries(
        Object.keys(LOG_CATEGORIES).map((key) => [key, true]),
    );

    const page = document.createElement('div');
    page.id = 'page-logs';
    page.className = 'settings-embedded-content settings-logs-panel';
    page.innerHTML = `
        <div class="logs-filters" id="log-filters"><button class="btn btn-default logs-inline-clear" id="btn-clear-logs">Clear</button></div>
        <div id="log-entries"></div>
    `;
    mount.appendChild(page);

    const filtersDiv = page.querySelector('#log-filters');
    const logEntries = page.querySelector('#log-entries');
    function isLogsVisible() {
        return state.activePage === 'dashboard' && state.dashboardActiveSubtab === 'logs';
    }

    function scrollToLatest() {
        if (!isLogsVisible()) return;
        requestAnimationFrame(() => {
            logEntries.scrollTop = logEntries.scrollHeight;
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
        return logEntries.scrollHeight - logEntries.scrollTop - logEntries.clientHeight <= PINNED_SLACK_PX;
    }

    function updateVisibility(entry) {
        entry.hidden = !state.activeFilters[entry.dataset.category];
    }

    function renderFilters() {
        const inlineClear = filtersDiv.querySelector('.logs-inline-clear');
        filtersDiv.innerHTML = '';
        Object.entries(LOG_CATEGORIES).forEach(([key, cat]) => {
            const chip = document.createElement('button');
            chip.className = `filter-chip ${state.activeFilters[key] ? 'active' : ''}`;
            chip.textContent = cat.label;
            chip.addEventListener('click', () => {
                state.activeFilters[key] = !state.activeFilters[key];
                chip.classList.toggle('active');
                logEntries.querySelectorAll('.log-entry').forEach(updateVisibility);
                scrollToLatest();
            });
            filtersDiv.appendChild(chip);
        });
        if (inlineClear) filtersDiv.appendChild(inlineClear);
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

    function logMainHtml({ ts = '', type = '', phase = 'info', headline = 'Event', repeat = '', attrs = {} }) {
        const attr = (key) => attrs[key] ? ` ${attrs[key]}` : '';
        return `
            <div class="log-main">
                <span class="log-ts"${attr('ts')}>${escapeHtml(ts)}</span>
                ${type ? `<span class="log-type ${escapeHtml(type.className || '')}"${attr('type')}>${escapeHtml(type.label || '')}</span>` : ''}
                <span class="log-phase ${escapeHtml(phase)}"${attr('phase')}>${escapeHtml(phase)}</span>
                <span class="log-headline"${attr('headline')}>${escapeHtml(headline)}</span>
                <span class="log-repeat"${attr('repeat')}${repeat ? '' : ' hidden'}>${escapeHtml(repeat)}</span>
            </div>
        `;
    }

    function bindRawToggle(root) {
        root.querySelectorAll('.log-raw-toggle').forEach((rawToggle) => {
            if (rawToggle.dataset.bound === '1') return;
            const rawEl = rawToggle.parentElement?.nextElementSibling;
            if (!rawEl || !rawEl.classList.contains('log-raw')) return;
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
                logEntries.appendChild(last.entry);
                updateVisibility(last.entry);
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
            })}
            ${execution ? `<div class="harness-chip log-executor-chip" title="${escapeHtml(execution.title || '')}">${harnessIdentityMarkup(execution.harness, {
                label: execution.label || '',
                className: 'log-executor-identity',
            })}</div>` : ''}
            ${metaPills(view.meta)}
            ${bodyHtml}
            <div class="log-actions">
                <button class="log-raw-toggle" type="button">Raw</button>
            </div>
            <pre class="log-raw" hidden>${escapeHtml(prettyLogEvent(evt))}</pre>
        `;
        bindRawToggle(entry);
        updateVisibility(entry);
        const pinned = isPinnedToLatest();
        logEntries.appendChild(entry);

        if (dedupeKey) {
            duplicateState.set(dedupeKey, { entry, ts: now, count: 1 });
        }

        trimEntries();
        if (state.activeFilters[cat] && pinned) scrollToLatest();
    }

    function createTaskGroupCard(groupId, category) {
        const entry = document.createElement('div');
        entry.className = 'log-entry log-task-card';
        entry.dataset.category = category;
        entry.dataset.taskGroup = groupId;
        entry.innerHTML = `
            ${logMainHtml({
                type: { className: category, label: groupId === 'bg-consciousness' ? 'background' : 'task' },
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
            <div class="log-body log-task-review" data-task-review hidden></div>
            <details class="log-task-details">
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
            review: entry.querySelector('[data-task-review]'),
            timeline: entry.querySelector('[data-task-timeline]'),
            events: 0,
            category,
            recent: [],
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
                })}
                ${metaPills(item.meta)}
                ${item.body ? `<div class="log-body">${escapeHtml(item.body)}</div>` : ''}
                <div class="log-actions">
                    <button class="log-raw-toggle" type="button">Raw</button>
                </div>
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
        const category = groupId === 'bg-consciousness'
            ? 'consciousness'
            : (eventCategory === 'errors' || earlier?.category === 'errors' ? 'errors' : 'tasks');
        // Captured before ANY record mutation: an already-mounted card grows
        // in place (summary rewrite, review unhide, timeline render) before
        // the append below, and that growth alone can push a pinned reader
        // past the slack allowance.
        const pinned = isPinnedToLatest();
        const record = taskGroups.get(groupId) || createTaskGroupCard(groupId, category);
        const ts = normalizeLogTs(evt.ts || evt.timestamp);

        record.events += 1;
        record.category = category;
        record.entry.dataset.category = category;
        record.ts.textContent = ts;
        record.kind.textContent = groupId === 'bg-consciousness' ? 'background' : `task ${groupId}`;
        record.kind.className = `log-type ${category}`;
        record.phase.textContent = view.phase || 'info';
        record.phase.className = `log-phase ${view.phase || 'info'}`;
        record.headline.textContent = view.headline || 'Task activity';
        record.count.textContent = `x${record.events}`;
        record.count.hidden = record.events <= 1;
        record.summary.innerHTML = metaPills([
            groupId === 'bg-consciousness' ? 'background' : `task=${groupId}`,
            ...view.meta,
        ]);
        const execution = executorChip(evt);
        if (execution && record.executor) {
            record.executor.title = execution.title || '';
            record.executor.innerHTML = harnessIdentityMarkup(execution.harness, {
                label: execution.label || '',
                className: 'log-executor-identity',
            });
            record.executor.hidden = false;
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
        logEntries.appendChild(record.entry);
        trimEntries();
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
        const merged = [];
        for (const name of ['events', 'tools', 'progress', 'supervisor']) {
            try {
                const resp = await apiFetch(`/api/logs/${name}?limit=150`, { cache: 'no-store' });
                if (!resp.ok) continue;
                const data = await resp.json();
                if (Array.isArray(data.entries)) merged.push(...data.entries);
            } catch (err) {
                /* best-effort backfill: live stream still works without it */
            }
        }
        const tsOf = (evt) => Date.parse(evt?.ts || evt?.timestamp || '') || 0;
        merged.sort((a, b) => tsOf(a) - tsOf(b));
        for (const evt of merged) addLogEntry(evt);
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
