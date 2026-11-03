import { renderPageHeader } from './page_header.js';
import {
    classifyWidgetJobStatus,
    isRetryableWidgetError,
    readWidgetJobStatus,
    boundedNumber,
    withWidgetRequestTimeout,
} from './widget_job.js';
import { getPath } from './widget_chart.js';
import { applyChartTheme, onThemeChange } from './theme_palette.js';
import { mountModuleWidget, mountRouteIframeWidget } from './widget_module.js';
import {
    planWidgetListPatch, requestWidgetCards, requestWidgetListPayload, widgetKey,
    widgetListRequests, widgetTabsSignature,
} from './widget_list.js';
import {
    bindWidgetCardMenus,
    effectiveStartMode,
    isFramedWidget,
    isRetainedWidget,
    renderWidgetCardParts,
    renderWidgetFacade,
    setWidgetCardNote,
    syncWidgetCardControls,
    WIDGET_START_MODES,
    withWidgetStartMode,
} from './widget_card.js';
import { bindWidgetCardReorder, createWidgetWidths, mergeWidgetOrder, normalizeWidgetOrder, sortTabsByWidgetOrder } from './widget_reorder.js';
import {
    apiClient,
    apiFetch,
    extensionRoutePath,
    extensionRoutePrefix,
} from './api_client.js';
import { escapeHtmlAttr as escapeHtml } from './utils.js';
import { collectSafeFieldValues, downloadViaHostBridge } from './ui_helpers.js';
import { mountPinnedCard, setCardPinned } from './card_components.js';
import { getHolds, HELD_LINES, settleAfterFrameLoad } from './holds.js';
import {
    filenameFromWidgetUrl, indexComponentTree, patchWidgetChildren, renderComponent, syncWidgetCharts,
} from './widget_render.js';

export {
    WIDGET_FRAME_BORDER_RESERVE,
    WIDGET_FRAME_DEFAULT_HEIGHT,
    WIDGET_FRAME_MAX_HEIGHT,
} from './widget_module.js';

function pageTemplate() {
    return `
        <section class="page app-page-glass" id="page-widgets">
            ${renderPageHeader({
                page: 'widgets',
            })}
            <div class="widgets-scroll">
                <div id="widgets-list-error" class="skills-load-error" hidden><span data-widget-list-error></span> <button type="button" class="btn btn-default btn-sm" data-widget-list-retry>Retry</button></div>
                <div id="widgets-list" class="widgets-list"></div>
                <div class="widgets-arrange-status ui-status" data-tone="neutral" data-widget-arrange-status role="status" aria-live="polite"></div>
            </div>
        </section>
    `;
}

function renderCardHtml(tab) {
    // Avoid leaking internal "skill:tab_id"; show skill only as needed.
    const title = tab.title || tab.tab_id || tab.skill;
    const source = tab.pinned_card ? 'kept from the chat' : `from ${tab.skill}`;
    const subtitle = tab.skill && tab.skill !== title
        ? `<span class="widgets-card-source">${escapeHtml(source)}</span>`
        : '';
    const span = Number(tab.span || tab.grid_span || 1);
    const spanClass = span >= 2 ? ' widgets-card-span-2' : '';
    const { run, menu } = renderWidgetCardParts(tab);
    return `
        <article class="widgets-card${spanClass}" data-widget-key="${escapeHtml(widgetKey(tab))}">
            <div class="widgets-card-head">
                <div class="widgets-card-title">
                    <strong>${escapeHtml(title)}</strong>
                    ${subtitle}
                </div>
                <div class="widgets-card-controls">
                    ${run ? `<div class="widgets-card-run">${run}</div>` : ''}
                    <div class="widgets-card-tools">
                        ${menu}
                        <button class="widgets-card-drag" type="button" data-widget-reorder-handle title="Move widget: drag or use arrow keys" aria-label="Move widget: drag or use arrow keys">↕</button>
                    </div>
                </div>
            </div>
            <div class="widgets-card-body" data-widget-mount></div>
            <button class="widgets-card-resize" type="button" data-widget-resize-handle title="Resize width: drag or use arrow keys" aria-label="Resize width: drag or use arrow keys"></button>
        </article>
        `;
}

// Full paint — only for a list that has no cards yet.
function renderShell(host, tabs) {
    if (!tabs.length) {
        host.innerHTML = '<div class="muted">No live widgets yet. Review and enable an extension that registers a UI tab.</div>';
        return;
    }
    host.innerHTML = tabs.map(renderCardHtml).join('');
}

function createCardElement(tab) {
    const template = document.createElement('template');
    template.innerHTML = renderCardHtml(tab).trim();
    return template.content.firstElementChild;
}

const widgetDisposers = new Map();
// key → settle promise of an ordered stop still waiting for the child's
// acknowledgement (its frame is still in the card until then).
const widgetDisposing = new Map();
// key → the mount in flight for it, `{card, isCurrent, promise}`: one per key;
// a second request for the same card joins it (`mountTrackedTab`).
const widgetMounting = new Map();
const widgetMountControllers = new Set();
const widgetMessageHandlers = new Set();
const widgetSessionState = new Map();
// The start ledger (holds.js) keeps two facts for a framed card on this device:
// the owner's Stop (`board:<key>`, until Start or a launch-policy change to Auto /
// Keep running) and a start of its code the page did not outlive
// (`widget:<skill>:<tab_id>` at its revision). Either shows the facade instead
// of starting the card by itself.
const holds = getHolds();
const stopHoldKey = (tab) => `board:${widgetKey(tab)}`;
const codeHoldKey = (tab) => `widget:${tab.skill}:${tab.tab_id}`;
let widgetsWsBridgeBound = false;

async function callWidgetRoute(tab, spec, values, signal) {
    const method = String(spec.method || 'GET').toUpperCase();
    const params = new URLSearchParams();
    for (const [key, value] of Object.entries(values || {})) {
        params.set(key, String(value ?? ''));
    }
    const noBody = method === 'GET' || method === 'HEAD';
    const url = extensionRoutePath(tab.skill, spec.route || spec.api_route, noBody ? params : null);
    if (!url) throw new Error('invalid widget route');
    const init = noBody
        ? { method, signal }
        : {
            method,
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(values || {}),
            signal,
        };
    try {
        const resp = await apiFetch(url, init);
        const contentType = resp.headers.get('content-type') || '';
        const data = contentType.includes('application/json')
            ? await resp.json().catch(() => ({}))
            : { text: await resp.text() };
        if (!resp.ok || data?.error || data === null) {
            const error = new Error(data?.error || `HTTP ${resp.status}`);
            error.status = resp.status;
            error.retryable = resp.status === 408 || resp.status === 429 || (resp.status >= 500 && resp.status <= 599);
            throw error;
        }
        return data;
    } catch (error) {
        if (error?.name === 'TypeError' && typeof error.retryable !== 'boolean') error.retryable = true;
        throw error;
    }
}

// `input` (a living card showing this widget in a message: `show_card(widget={..., input})`) gives the
// widget's form fields of the same names their first values; what the owner typed in this page session wins.
export async function mountDeclarativeWidget(mount, tab, render, { input = null } = {}) {
    // A new mount is a real lifecycle boundary, including password lifetime.
    // In-place reconciliation applies only while this mounted owner is alive.
    mount.replaceChildren();
    const components = Array.isArray(render.components) ? render.components : [];
    const { entries: componentEntries, byKey: componentByKey } = indexComponentTree(components);
    const componentSpec = (key) => componentByKey.get(key)?.component || null;
    // Where the declaration renders (widget_render.js): this card's own skill routes.
    const renderContext = { routePath: (route, params) => extensionRoutePath(tab.skill, route, params),
        routePrefix: extensionRoutePrefix(tab.skill) };
    const persistenceKey = tab.key || `${tab.skill}:${tab.tab_id}`;
    const saved = widgetSessionState.get(persistenceKey) || {};
    const state = { ...(saved.state || {}) };
    const status = { ...(saved.status || {}) };
    const seeded = input && typeof input === 'object' ? componentEntries.filter(({ component }) => component.type === 'form')
        .map(({ component, key }) => [key, Object.fromEntries((component.fields || [])
            .filter((field) => Object.hasOwn(input, field?.name || '')).map((field) => [field.name, input[field.name]]))]) : [];
    const formValues = { ...Object.fromEntries(seeded), ...(saved.formValues || {}) };
    const componentState = { ...(saved.componentState || {}) };
    const timers = new Set();
    const controllers = new Set();
    const chartInstances = new Map();
    const chartShapes = new Map();
    const eventSources = new Map();
    const activePolls = new Set();
    const activeJobs = new Set();
    const autoStarted = new Set();
    const messageHandlers = new Set();
    const subscribed = new Set();
    const pendingActions = new Set();
    const interactions = new AbortController();
    controllers.add(interactions);
    let disposed = false;

    // Re-tint the charts this mount already owns. They keep their instances, so
    // the plotted data and the reuse shape in chartShapes both survive; only the
    // chrome colours move. Released by dispose(), the same owner that destroys them.
    const unsubscribeTheme = onThemeChange(() => {
        if (disposed) return;
        chartInstances.forEach((chart) => applyChartTheme(chart));
    });

    const actionFeedback = (key, outcome, data = {}) => {
        componentState[`feedback:${key}`] = {
            status: outcome,
            message: outcome === 'error' ? (data.error || 'Action failed.')
                : (data.message || (outcome === 'loading' ? 'Working…' : 'Completed.')),
        };
    };

    const downloadWidgetFile = async (url, filename) => {
        const resolvedUrl = new URL(url, window.location.origin);
        const expectedPrefix = extensionRoutePrefix(tab.skill);
        if (resolvedUrl.origin !== window.location.origin || !resolvedUrl.pathname.startsWith(expectedPrefix)) {
            throw new Error('download URL is outside this widget extension');
        }
        const safeName = filenameFromWidgetUrl(resolvedUrl.toString(), filename || 'download');
        return downloadViaHostBridge(resolvedUrl.pathname + resolvedUrl.search, safeName, { fetchOptions: { credentials: 'include' } });
    };

    const schedule = (fn, delay) => {
        if (disposed) return null;
        const timer = setTimeout(() => {
            timers.delete(timer);
            fn();
        }, delay);
        timers.add(timer);
        return timer;
    };
    const dispose = () => {
        if (disposed) return;
        rememberFormValues();
        const jobTargets = new Set(componentEntries.filter(({ key }) => componentState[`job:${key}`])
            .map(({ component }) => component.target || 'result'));
        for (const [record, feedback] of Object.entries(componentState)) {
            if (!record.startsWith('feedback:') || feedback.status !== 'loading') continue;
            const key = record.slice('feedback:'.length);
            if (!componentState[`job:${key}`]) {
                actionFeedback(key, 'error', { error: 'No result received before this widget was closed.' });
                const spec = componentSpec(key);
                const target = spec?.target || 'result';
                if (spec && status[target] === 'loading' && !jobTargets.has(target)) status[target] = 'error';
            }
        }
        widgetSessionState.set(persistenceKey, {
            state: { ...state },
            status: { ...status },
            formValues: { ...formValues },
            componentState: { ...componentState },
        });
        disposed = true;
        unsubscribeTheme();
        controllers.forEach((controller) => controller.abort());
        controllers.clear();
        chartInstances.forEach((chart) => chart.destroy());
        chartInstances.clear();
        chartShapes.clear();
        eventSources.forEach((source) => source.close());
        eventSources.clear();
        timers.forEach((timer) => clearTimeout(timer));
        timers.clear();
        activePolls.clear();
        activeJobs.clear();
        pendingActions.clear();
        messageHandlers.forEach((handler) => widgetMessageHandlers.delete(handler));
        messageHandlers.clear();
        subscribed.clear();
    };
    const callRoute = async (spec, values) => {
        if (disposed) throw new Error('widget disposed');
        const controller = new AbortController();
        controllers.add(controller);
        try {
            return await withWidgetRequestTimeout(
                (signal) => callWidgetRoute(tab, spec, values, signal),
                controller,
            );
        } finally {
            controllers.delete(controller);
        }
    };
    const rememberFormValues = () => {
        mount.querySelectorAll('[data-widget-form]').forEach((form) => {
            const key = form.dataset.widgetForm || '';
            const spec = componentSpec(key);
            if (!spec) return;
            formValues[key] = collectSafeFieldValues(form, spec.fields || [], { includePasswords: false });
        });
    };
    const startPoll = (key) => {
        if (disposed || activePolls.has(key)) return;
        const spec = componentSpec(key);
        if (!spec) return;
        const target = spec.target || 'result';
        const maxTicks = boundedNumber(spec.max_ticks, 20, 1, 100);
        const intervalMs = boundedNumber(spec.interval_ms, 2000, 1000, 30000);
        let ticks = 0;
        activePolls.add(key);
        const poll = async () => {
            if (disposed) return;
            ticks += 1;
            // SWR (v6.71.0): 'loading' only when there is nothing to show yet;
            // a background refetch keeps the content and shows a thin indicator.
            const hadData = state[target] !== undefined;
            status[target] = hadData ? 'refreshing' : 'loading';
            renderAll();
            try {
                state[target] = await callRoute(spec, {});
                if (disposed) return;
                status[target] = 'success';
            } catch (err) {
                // SWR: a failed background refetch keeps the stale content —
                // only the status reports the error; a failed FIRST load still
                // surfaces the error payload (there is nothing else to show).
                if (!hadData) state[target] = { error: err.message || String(err) };
                status[target] = 'error';
            }
            const stopValue = getPath(state[target], spec.stop_path || '', undefined);
            if (ticks < maxTicks && String(stopValue) !== String(spec.stop_value ?? 'done')) {
                schedule(poll, intervalMs);
            } else {
                activePolls.delete(key);
            }
            renderAll();
        };
        poll();
    };
    // A job's progress is fed by two writers — the status poll and the WS
    // subscription. Keep the percent monotonic per job so a stale poll tick or an
    // out-of-order WS event can never move the bar backward. Resets when the job id
    // changes. The percent key is whatever the progress component(s) read
    // (`value_key`), so this works regardless of the skill's field name.
    const progressValueKeys = (() => {
        const keys = [];
        for (const { component: c } of componentEntries) {
            if (String(c?.type || '') !== 'progress') continue;
            const k = String(c.path || c.value_key || 'progress');
            if (k && !k.includes('.') && !keys.includes(k)) keys.push(k);
        }
        return keys.length ? keys : ['progress_pct', 'progress'];
    })();
    const clampMonotonicProgress = (target, jobId, nextObj) => {
        if (!nextObj || typeof nextObj !== 'object') return nextObj;
        let pctKey = '';
        let pct;
        for (const k of progressValueKeys) {
            if (typeof nextObj[k] === 'number' && Number.isFinite(nextObj[k])) { pct = nextObj[k]; pctKey = k; break; }
        }
        if (pctKey === '') return nextObj;
        const stateKey = `progress-clamp:${target}`;
        const prev = componentState[stateKey];
        if (prev && prev.jobId === jobId && prev.pct > pct) {
            nextObj[pctKey] = prev.pct;
            return nextObj;
        }
        componentState[stateKey] = { jobId, pct: nextObj[pctKey] };
        return nextObj;
    };

    const startJobPoll = (key, jobId) => {
        if (disposed || !jobId || activeJobs.has(key)) return;
        const spec = componentSpec(key);
        if (!spec) return;
        const target = spec.target || 'result';
        const statusRoute = spec.status_route || spec.job_status_route || 'status';
        const intervalMs = boundedNumber(spec.interval_ms, 2000, 1000, 30000);
        const maxTicks = boundedNumber(spec.max_ticks, 240, 1, 1000);
        let ticks = 0;
        activeJobs.add(key);
        componentState[`job:${key}`] = { job_id: jobId, status_route: statusRoute };
        const pollJob = async () => {
            if (disposed) return;
            ticks += 1;
            try {
                const data = await callRoute({ route: statusRoute, method: 'GET' }, { job_id: jobId });
                if (disposed) return;
                if (!data || typeof data !== 'object' || Array.isArray(data)) {
                    state[target] = { error: 'invalid job status response' };
                    status[target] = 'error';
                    delete componentState[`job:${key}`];
                    activeJobs.delete(key);
                    actionFeedback(key, status[target], state[target]);
                    renderAll();
                    return;
                }
                const currentStatus = readWidgetJobStatus(data);
                const statusKind = classifyWidgetJobStatus(currentStatus);
                if (statusKind === 'success') {
                    state[target] = data.result && typeof data.result === 'object' ? data.result : data;
                    status[target] = 'success';
                    delete componentState[`job:${key}`];
                    activeJobs.delete(key);
                    actionFeedback(key, status[target], state[target]);
                    renderAll();
                    return;
                }
                if (statusKind === 'failure') {
                    state[target] = { error: data.error || 'job failed' };
                    status[target] = 'error';
                    delete componentState[`job:${key}`];
                    activeJobs.delete(key);
                    actionFeedback(key, status[target], state[target]);
                    renderAll();
                    return;
                }
                if (statusKind === 'invalid') {
                    state[target] = { error: 'invalid job status response' };
                    status[target] = 'error';
                    delete componentState[`job:${key}`];
                    activeJobs.delete(key);
                    actionFeedback(key, status[target], state[target]);
                    renderAll();
                    return;
                }
                // Merge the whole flat status payload so the renderer's value_key
                // (e.g. `progress_pct`) is surfaced — cherry-picking `data.progress`
                // dropped the percent and broke the poll fallback when WS hiccuped.
                state[target] = clampMonotonicProgress(target, jobId, {
                    ...(state[target] || {}),
                    ...data,
                    job_id: jobId,
                });
                status[target] = 'loading';
                actionFeedback(key, 'loading', data);
                renderAll();
                if (ticks < maxTicks) {
                    schedule(pollJob, intervalMs);
                } else {
                    state[target] = { error: 'job timed out waiting for result' };
                    status[target] = 'error';
                    delete componentState[`job:${key}`];
                    activeJobs.delete(key);
                    actionFeedback(key, status[target], state[target]);
                    renderAll();
                }
            } catch (err) {
                if (disposed) return;
                if (isRetryableWidgetError(err) && ticks < maxTicks) {
                    // Keep the durable job id and any useful progress while a
                    // transient transport/server failure is retried.
                    status[target] = 'loading';
                    actionFeedback(key, 'loading', componentState[`feedback:${key}`]);
                    renderAll();
                    schedule(pollJob, intervalMs);
                    return;
                }
                state[target] = { error: err.message || String(err) };
                status[target] = 'error';
                delete componentState[`job:${key}`];
                activeJobs.delete(key);
                actionFeedback(key, status[target], state[target]);
                renderAll();
            }
        };
        pollJob();
    };
    const runAction = async (key, values) => {
        const spec = componentSpec(key);
        if (disposed || !spec || spec.disabled || pendingActions.has(key) || activeJobs.has(key)) return;
        const target = spec.target || 'result';
        pendingActions.add(key);
        status[target] = 'loading';
        actionFeedback(key, 'loading');
        renderAll();
        try {
            const data = await callRoute(spec, values);
            if (disposed) return;
            if (spec.job === true || spec.mode === 'job') {
                const jobId = data.job_id || data.id;
                if (!jobId) throw new Error('job response missing job_id');
                state[target] = { job_id: jobId, message: data.message || 'Job started.' };
                status[target] = 'loading';
                actionFeedback(key, 'loading', state[target]);
                startJobPoll(key, jobId);
            } else {
                state[target] = data;
                status[target] = 'success';
                actionFeedback(key, 'success', data);
            }
        } catch (err) {
            if (disposed) return;
            state[target] = { error: err.message || String(err) };
            status[target] = 'error';
            actionFeedback(key, 'error', state[target]);
        } finally {
            if (!disposed) {
                pendingActions.delete(key);
                renderAll();
            }
        }
    };
    const moveCard = async (board, cardId, columnId) => {
        const key = board.dataset.widgetKanbanKey || '';
        const spec = componentSpec(key);
        const route = board.dataset.widgetKanbanRoute || '';
        const pendingKey = `kanban:${key}`;
        if (disposed || !spec || !route || !cardId || !columnId || pendingActions.has(pendingKey)) return;
        const target = spec.target || 'result';
        pendingActions.add(pendingKey);
        status[target] = 'loading';
        actionFeedback(key, 'loading');
        renderAll();
        try {
            const data = await callRoute({ route, method: spec.on_move?.method || 'POST' }, { card_id: cardId, column_id: columnId });
            if (disposed) return;
            state[target] = data;
            status[target] = 'success';
            actionFeedback(key, 'success', data);
        } catch (err) {
            if (disposed) return;
            state[target] = { error: err.message || String(err) };
            status[target] = 'error';
            actionFeedback(key, 'error', state[target]);
        } finally {
            if (!disposed) {
                pendingActions.delete(pendingKey);
                renderAll();
            }
        }
    };
    const renderAll = () => {
        if (disposed) return;
        rememberFormValues();
        widgetSessionState.set(persistenceKey, {
            state: { ...state },
            status: { ...status },
            formValues: { ...formValues },
            componentState: { ...componentState },
        });
        const visibleKeys = new Set();
        const view = { state, status, componentState, formValues, pendingActions, visibleKeys };
        const desired = document.createElement('template');
        desired.innerHTML = components.map((component, idx) => renderComponent(renderContext, component, view, `components.${idx}`)).join('');
        patchWidgetChildren(mount, desired.content);
        syncWidgetCharts(mount, chartInstances, chartShapes);
        visibleKeys.forEach((key) => {
            const component = componentSpec(key);
            if (!component) return;
            if (String(component.type || '') === 'poll' && component.auto_start === true && !autoStarted.has(key)) {
                autoStarted.add(key);
                queueMicrotask(() => startPoll(key));
            }
            if (component.job === true || component.mode === 'job') {
                const savedJob = componentState[`job:${key}`];
                const jobId = savedJob && savedJob.job_id;
                if (jobId) {
                    queueMicrotask(() => startJobPoll(key, jobId));
                }
            }
            if (String(component.type || '') !== 'stream' || eventSources.has(key)) return;
            const url = extensionRoutePath(tab.skill, component.route || component.api_route, new URLSearchParams());
            if (!url || typeof EventSource === 'undefined') return;
            const target = component.target || 'result';
            const source = new EventSource(url);
            eventSources.set(key, source);
            status[target] = 'loading';
            source.onmessage = (event) => {
                if (disposed) return;
                try {
                    state[target] = JSON.parse(event.data);
                } catch {
                    state[target] = { text: event.data || '' };
                }
                status[target] = 'success';
                renderAll();
            };
            source.onerror = () => {
                if (disposed) return;
                status[target] = 'error';
                renderAll();
            };
        });
        visibleKeys.forEach((key) => {
            const component = componentSpec(key);
            if (!component || String(component.type || '') !== 'subscription' || subscribed.has(key)) return;
            const event = String(component.event || component.message_type || '').trim();
            const prefix = String(tab.ws_prefix || '').trim();
            if (!event || !prefix) return;
            const expectedType = `${prefix}${event}`;
            const target = component.target || 'result';
            const handler = (msg) => {
                if (disposed || msg?.type !== expectedType) return;
                const data = msg.data || {};
                // Same monotonic guard as the poll writer: an out-of-order WS event
                // must not rewind the bar.
                state[target] = clampMonotonicProgress(target, data.job_id || '', { ...data });
                status[target] = 'success';
                renderAll();
            };
            subscribed.add(key);
            messageHandlers.add(handler);
            widgetMessageHandlers.add(handler);
        });
    };
    // One binding per mount: preserved nodes must not acquire another handler
    // on every data tick. The mount's disposer aborts all delegated listeners.
    const listen = (event, handler) => mount.addEventListener(event, handler, { signal: interactions.signal });
    listen('submit', (event) => {
        const form = event.target.closest('[data-widget-form]');
        if (!form) return;
        event.preventDefault();
        const key = form.dataset.widgetForm || '';
        const spec = componentSpec(key);
        if (spec) runAction(key, collectSafeFieldValues(form, spec.fields || []));
    });
    listen('click', async (event) => {
        const button = event.target.closest('button');
        if (!button || button.disabled) return;
        if (button.dataset.widgetAction) {
            const key = button.dataset.widgetAction;
            runAction(key, componentSpec(key)?.body || {});
        } else if (button.dataset.widgetPoll) {
            startPoll(button.dataset.widgetPoll);
        } else if (button.dataset.widgetTabKey) {
            componentState[button.dataset.widgetTabKey] = Number(button.dataset.widgetTabIdx || 0);
            renderAll();
        } else if (button.dataset.widgetDownloadUrl) {
            event.preventDefault();
            const key = button.closest('[data-widget-component]')?.dataset.widgetComponent || '';
            const pendingKey = `download:${key}`;
            if (pendingActions.has(pendingKey)) return;
            pendingActions.add(pendingKey);
            actionFeedback(key, 'loading');
            renderAll();
            try {
                const receipt = await downloadWidgetFile(button.dataset.widgetDownloadUrl, button.dataset.widgetDownloadFilename || 'download');
                if (disposed) return;
                if (receipt?.ok) actionFeedback(key, 'success', { message: 'Download requested.' });
                else if (receipt?.degraded === 'copy-link') actionFeedback(key, 'warning', { message: 'Download unavailable. Link copied.' });
                else actionFeedback(key, 'error', { error: receipt?.error || 'Download failed.' });
            } catch (err) {
                if (disposed) return;
                state.download = { error: err.message || String(err) };
                status.download = 'error';
                actionFeedback(key, 'error', state.download);
            } finally {
                if (!disposed) {
                    pendingActions.delete(pendingKey);
                    renderAll();
                }
            }
        }
    });
    listen('change', (event) => {
        const select = event.target.closest('[data-widget-kanban-move]');
        const board = select?.closest('[data-widget-kanban-key]');
        if (board) moveCard(board, select.dataset.widgetKanbanCardId || '', select.value || '');
    });
    let draggedCardId = '';
    listen('dragstart', (event) => {
        const card = event.target.closest('[data-widget-kanban-card]');
        const board = card?.closest('[data-widget-kanban-key]');
        if (!board) return;
        if (pendingActions.has(`kanban:${board.dataset.widgetKanbanKey}`)) {
            event.preventDefault();
            return;
        }
        draggedCardId = card.dataset.widgetKanbanCard || '';
        if (event.dataTransfer) {
            event.dataTransfer.effectAllowed = 'move';
            event.dataTransfer.setData('text/plain', draggedCardId);
        }
    });
    for (const type of ['dragover', 'drop']) listen(type, (event) => {
        const column = event.target.closest('[data-widget-kanban-col]');
        const board = column?.closest('[data-widget-kanban-key]');
        if (!board?.dataset.widgetKanbanRoute) return;
        event.preventDefault();
        if (type === 'dragover') {
            if (event.dataTransfer) event.dataTransfer.dropEffect = 'move';
        } else {
            moveCard(board, event.dataTransfer?.getData('text/plain') || draggedCardId, column.dataset.widgetKanbanCol || '');
        }
    });
    renderAll();
    return dispose;
}

async function mountTab(card, tab, mountSignal = null) {
    const mount = card.querySelector('[data-widget-mount]');
    const render = tab.render || {};
    if (!mount) return;
    if (render.kind === 'iframe' && render.route) {
        return mountRouteIframeWidget(mount, tab, render, mountSignal, null, widgetMessageHandlers);
    }
    // A kept living card is Ouroboros's, not a skill's: no route, its presses go to the card's decisions.
    if (tab.pinned_card) return mountPinnedCard(mount, tab);
    if (render.kind === 'declarative') {
        return mountDeclarativeWidget(mount, tab, render);
    }
    if (render.kind === 'module' && render.entry) {
        return mountModuleWidget(mount, tab, render, mountSignal, widgetMessageHandlers);
    }
    mount.innerHTML = `<div class="muted">Widget render kind <code>${escapeHtml(render.kind || 'unknown')}</code> is not supported yet.</div>`;
    return null;
}

// A framed disposer settles asynchronously (dispose → child ack, ≤ 1 s); track
// it by key so a remount of the same key waits for it — never two frames per key.
function trackSettling(key, result) {
    if (!result || typeof result.then !== 'function') return null;
    const settling = result
        .catch((err) => console.warn('widgets: dispose failed', err))
        .finally(() => {
            if (widgetDisposing.get(key) === settling) widgetDisposing.delete(key);
        });
    widgetDisposing.set(key, settling);
    return settling;
}

// Returns the settle promise of an ordered stop, or null when nothing is left to
// wait for (declarative disposers, and route pages that never joined the bridge,
// finish synchronously).
function disposeWidgetByKey(key) {
    const dispose = widgetDisposers.get(key);
    if (!dispose) return widgetDisposing.get(key) || null;
    widgetDisposers.delete(key);
    try {
        return trackSettling(key, dispose());
    } catch (err) {
        console.warn('widgets: dispose failed', err);
        return null;
    }
}

// Issue the stop for every mounted card except the keys `keep` answers true for
// (the frames the owner keeps running while Widgets is hidden; a caller that
// passes nothing stops them too); resolves once every ordered stop in flight
// has settled (they run in parallel, each bounded by the ack timeout).
function disposeMountedWidgets(keep = null) {
    widgetMountControllers.forEach((controller) => controller.abort());
    widgetMountControllers.clear();
    Array.from(widgetDisposers.keys()).forEach((key) => {
        if (!keep?.(key)) disposeWidgetByKey(key);
    });
    return Promise.allSettled(Array.from(widgetDisposing.values()));
}

// A card whose skill left the live list: its stop is still the ordered one, so
// the node (and the frame in it) stays until the acknowledgement or the timeout,
// marked so the keyed patch treats it as already gone.
function liveCardFor(list, key) {
    return list.querySelector(`[data-widget-key="${CSS.escape(key)}"]:not([data-widget-removed])`);
}

// A card whose frame is stopping in order stays in the document, marked and in
// the `stopping` state, until the acknowledgement or the timeout — both the
// vanished card and the changed card (a revision or render change while its
// frame runs) leave this way; the frame is never torn down in the same turn as
// the dispose message. The keyed patch treats a marked card as already gone.
function retireCard(card, settling) {
    card.setAttribute('data-widget-removed', '');
    syncWidgetCardControls(card, 'stopping');
    settling.then(() => card.remove());
}

// Keyed patch over the existing <article> nodes: vanished cards go (their mounted
// work disposed first, their session state evicted), cards whose own entry
// changed are replaced — a running one retires first while its fresh card is
// inserted beside it and mounts once the stop settled (`mountTrackedTab` waits
// on the same settle promise) — new cards are appended, every other card keeps
// its DOM node. No node ever moves: the visible order is the masonry key order,
// and a moved <iframe> would reload. The list's masonry relayouts on the mutation.
function patchWidgetCards(list, previousTabs, nextTabs) {
    const plan = planWidgetListPatch(previousTabs, nextTabs);
    for (const key of plan.removed) {
        const card = liveCardFor(list, key);
        const settling = disposeWidgetByKey(key);
        widgetSessionState.delete(key);
        if (!card) continue;
        if (settling) retireCard(card, settling);
        else card.remove();
    }
    for (const tab of nextTabs) {
        const key = widgetKey(tab);
        const card = liveCardFor(list, key);
        if (card && !plan.changed.includes(key)) continue;
        const settling = disposeWidgetByKey(key);
        const fresh = createCardElement(tab);
        if (!card) list.append(fresh);
        else if (!settling) card.replaceWith(fresh);
        else {
            card.after(fresh);
            retireCard(card, settling);
        }
    }
}

// One mount in flight per key: a second request for the same card while it is
// `starting` (the policy menu's Auto / Keep running) joins the mount under way
// instead of racing it for the body and the disposer. A request for another
// card node (the fresh card of a changed entry) or from a later page generation
// (leave → return inside the ack window) never joins a mount that bails as
// stale: it waits for that mount to finish, then runs its own `mountTabOnce`,
// which re-checks `isCurrent()` and the node's connection before it mounts.
function mountTrackedTab(card, tab, isCurrent = () => true) {
    const key = widgetKey(tab);
    const inFlight = widgetMounting.get(key);
    if (inFlight && inFlight.card === card && inFlight.isCurrent()) return inFlight.promise;
    const started = inFlight
        ? inFlight.promise.catch(() => {}).then(() => mountTabOnce(card, tab, key, isCurrent))
        : mountTabOnce(card, tab, key, isCurrent);
    const mounting = started.finally(() => {
        if (widgetMounting.get(key)?.promise === mounting) widgetMounting.delete(key);
    });
    widgetMounting.set(key, { card, isCurrent, promise: mounting });
    return mounting;
}

async function mountTabOnce(card, tab, key, isCurrent) {
    // One frame per key: a stop still awaiting its acknowledgement finishes first.
    const settling = disposeWidgetByKey(key);
    if (settling) await settling;
    if (!isCurrent() || !card.isConnected) return;
    const mountController = new AbortController();
    widgetMountControllers.add(mountController);
    // A frame's start goes into the ledger before its first byte runs; its load and one host second clear it.
    const settle = isFramedWidget(tab) && !tab.pinned_card ? holds.starting(codeHoldKey(tab), tab.revision) : () => {};
    try {
        const mounted = await mountTab(card, tab, mountController.signal);
        if (typeof mounted !== 'function') { settle(); return; }
        const cancelSettle = settleAfterFrameLoad(card.querySelector('[data-widget-mount]'), settle);
        const dispose = () => { cancelSettle(); settle(); return mounted(); };
        // Stale (page left, list rebuilt, card replaced meanwhile): stop it in
        // order instead of registering a frame nobody can reach.
        if (!isCurrent() || !card.isConnected) {
            trackSettling(key, dispose());
            return;
        }
        widgetDisposers.set(key, dispose);
    } catch (err) {
        settle();
        throw err;
    } finally {
        widgetMountControllers.delete(mountController);
    }
}


export function initWidgets(ctx = {}) {
    const page = document.createElement('div');
    page.innerHTML = pageTemplate();
    document.getElementById('content').appendChild(page.firstElementChild);
    const list = document.getElementById('widgets-list');
    const listError = document.getElementById('widgets-list-error');
    const retryButton = listError.querySelector('[data-widget-list-retry]');
    const listRequests = widgetListRequests();
    let renderGeneration = 0;
    let widgetsVisible = false;
    let widgetsMounted = false;
    // Last good payload keeps revisits and slow refreshes from blanking the page;
    // its order-independent signature decides whether a fetched list touches the DOM.
    let lastTabs = null;
    let lastSignature = '';
    // Generation of the list sync in flight (0 = none). A reconcile trigger landing
    // mid-sync marks the list dirty and the running sync loops once more.
    let activeSync = 0;
    let listDirty = false;
    let uiPreferences = { widget_order: [], widget_size: {}, widget_start_mode: {}, nested_subagents_expanded: false };
    if (ctx.ws && !widgetsWsBridgeBound) {
        widgetsWsBridgeBound = true;
        ctx.ws.on('message', (msg) => {
            widgetMessageHandlers.forEach((handler) => handler(msg));
        });
        // List reconcile triggers besides page entry: a skill lifecycle change, and
        // every (re)connect — events may have been missed while offline, or the
        // server restarted with the same SHA (no SPA reload then). No polling.
        ctx.ws.on('extension_lifecycle', reconcileWidgetList);
        ctx.ws.on('open', reconcileWidgetList);
    }

    const hasCards = () => Boolean(list.querySelector('[data-widget-key]:not([data-widget-removed])'));
    const isCurrentFor = (generation) => () => widgetsVisible && generation === renderGeneration;
    const tabByKey = (key) => (lastTabs || []).find((tab) => widgetKey(tab) === key) || null;
    // A framed card the owner keeps running (effective policy `retain`) holds
    // its frame while Widgets is hidden; every other mounted card is stopped.
    const retainsWhileHidden = (key) => isRetainedWidget(tabByKey(key), uiPreferences);
    const keptRunning = () => Array.from(widgetDisposers.keys()).filter(retainsWhileHidden);
    // The complete visible key order (`lastTabs` already carries `widget_order`) and the
    // owner's widths: masonry packs the cards by them; no DOM node is ever moved for it.
    const currentWidgetOrder = () => (lastTabs || []).map(widgetKey);
    const widths = createWidgetWidths(list, {
        tabs: () => lastTabs || [], prefs: () => uiPreferences, save: (payload) => apiClient.saveUiPreferences(payload),
        adopt(widgetSize) { uiPreferences = { ...uiPreferences, widget_size: widgetSize }; },
    });
    const relayout = widths.relayout;

    function paintShell(tabs) {
        renderShell(list, tabs);
        bindWidgetCardReorder(list, currentWidgetOrder, persistWidgetOrder);
        relayout();
    }

    // Page entry: the shell is on screen before the first await. Leaving
    // disposed the mounted work but kept the cards, so an entry reuses them and
    // only mounts into them again. A window reload is the only hard reset there
    // is; nothing in the page rebuilds every card behind the owner's back.
    async function render() {
        const generation = ++renderGeneration;
        widgetsVisible = true;
        if (widgetsMounted) return;
        if (!lastTabs) {
            list.innerHTML = '<div class="muted">Loading widgets…</div>';
        } else if (!hasCards()) {
            paintShell(lastTabs);
        } else {
            relayout();
        }
        await syncWidgets(generation);
    }

    // WS-driven reconcile: visible → sync now (or mark dirty while one runs);
    // hidden → dirty flag, consumed by the next entry's unconditional fetch —
    // plus the force-stop of kept-running frames whose skill left the list,
    // whether or not a sync was still in flight when the owner left.
    function reconcileWidgetList() {
        if (widgetsVisible && !activeSync) {
            syncWidgets(renderGeneration);
            return;
        }
        listDirty = true;
        if (!widgetsVisible) stopVanishedRetainedWidgets();
    }

    // Lifecycle event while Widgets is hidden and frames are kept running: a
    // skill that left the live list (disable / unload / delete) must not keep
    // its frame alive until the next visit. Stop those frames now, in order;
    // the card itself goes with the next entry's sync (the list is dirty).
    async function stopVanishedRetainedWidgets() {
        const kept = keptRunning();
        if (!kept.length) return;
        let data;
        try {
            data = await listRequests.run((controller) => requestWidgetCards(apiClient, controller));
        } catch {
            return;
        }
        if (widgetsVisible) return;
        const live = new Set((Array.isArray(data.ui_tabs) ? data.ui_tabs : []).map(widgetKey));
        kept.filter((key) => !live.has(key)).forEach(disposeWidgetByKey);
    }

    // One list sync: GET /api/widgets (+ preferences), signature compare, keyed
    // patch, then mount. Repeats while a trigger marked the list dirty.
    async function syncWidgets(generation) {
        const isCurrent = isCurrentFor(generation);
        activeSync = generation;
        retryButton.disabled = true;
        retryButton.textContent = 'Retrying…';
        try {
            do {
                listDirty = false;
                const readSizes = widths.beginRead();
                const [data, prefs] = await listRequests.run(
                    (controller) => requestWidgetListPayload(apiClient, controller));
                if (!isCurrent()) return;
                listError.hidden = true;
                if (prefs) {
                    uiPreferences = {
                        widget_order: normalizeWidgetOrder(prefs.widget_order),
                        widget_size: readSizes(prefs.widget_size),
                        widget_start_mode: prefs.widget_start_mode && typeof prefs.widget_start_mode === 'object'
                            ? prefs.widget_start_mode
                            : {},
                        nested_subagents_expanded: prefs.nested_subagents_expanded === true,
                    };
                }
                const tabs = sortTabsByWidgetOrder(
                    Array.isArray(data.ui_tabs) ? data.ui_tabs : [],
                    uiPreferences.widget_order, currentWidgetOrder(),
                );
                const signature = widgetTabsSignature(tabs);
                if (hasCards() && tabs.length) {
                    // Same signature: no card node is added, removed, replaced or
                    // moved (the sync still reconciles controls and layout below).
                    if (signature !== lastSignature) {
                        cardMenus.close();
                        patchWidgetCards(list, lastTabs, tabs);
                    }
                } else {
                    cardMenus.close();
                    // Rebuilding the shell destroys frames, so the ordered stops
                    // still in flight get their acknowledgement window first.
                    await disposeMountedWidgets();
                    if (!isCurrent()) return;
                    // Same eviction the keyed patch performs for a vanished card,
                    // for the transition the patch never sees (the last card
                    // leaving, or the first list arriving): a key that is gone must
                    // not leave its declarative session state behind, or re-enabling
                    // the skill would restore values the owner never re-entered (the
                    // owner's Stop stays: it holds until Start). It runs after
                    // disposal, because a declarative disposer writes that snapshot
                    // on its way out.
                    const live = new Set(tabs.map(widgetKey));
                    for (const key of (lastTabs || []).map(widgetKey)) {
                        if (!live.has(key)) widgetSessionState.delete(key);
                    }
                    renderShell(list, tabs);
                }
                lastTabs = tabs;
                lastSignature = signature;
                bindWidgetCardReorder(list, currentWidgetOrder, persistWidgetOrder);
                relayout();
                widgetsMounted = true;
                for (const tab of tabs) {
                    if (!isCurrent()) return;
                    const key = widgetKey(tab);
                    const card = liveCardFor(list, key);
                    if (!card) continue;
                    if (widgetDisposers.has(key)) {
                        if (isFramedWidget(tab)) syncWidgetCardControls(card, 'running', effectiveStartMode(tab, uiPreferences));
                        continue;
                    }
                    if (isFramedWidget(tab) && !startsOnShow(tab)) await settleStopped(card, tab);
                    else await startWidget(card, tab, isCurrent);
                }
            } while (listDirty && isCurrent());
        } catch (err) {
            if (!isCurrent()) return;
            // A deliberate abort (page hide, disposal) is not a list failure;
            // the deadline reports itself as its own typed timeout error.
            if (err && err.name === 'AbortError') return;
            // Error feedback is outside the keyed list. Last good cards and
            // owner Stop choices remain intact while the list can be retried.
            if (!lastTabs) list.textContent = '';
            listError.querySelector('[data-widget-list-error]').textContent = `Failed to load widgets: ${err.message || err}`;
            listError.hidden = false;
            widgetsMounted = false;
        } finally {
            if (activeSync === generation) {
                activeSync = 0;
                retryButton.disabled = false;
                retryButton.textContent = 'Retry';
            }
        }
    }

    // A reorder (handle drag / keys) hands over the next order of the shown cards: merge it
    // into the stored order (a card not on screen keeps its slot), re-sort, relayout, persist.
    function persistWidgetOrder(order) {
        const normalized = mergeWidgetOrder(uiPreferences.widget_order, order);
        uiPreferences = { ...uiPreferences, widget_order: normalized };
        if (lastTabs) lastTabs = sortTabsByWidgetOrder(lastTabs, normalized);
        relayout();
        apiClient.saveUiPreferences({ widget_order: normalized }).catch((err) => {
            console.warn('Failed to save widget order', err);
        });
    }

    // Framed card, effective policy `auto` or `retain` (both start on show; only
    // `retain` survives leaving), and the ledger holds neither the owner's Stop
    // nor a start of this revision the page did not outlive (nor `?hold=all`).
    const startsOnShow = (tab) => effectiveStartMode(tab, uiPreferences) !== 'manual'
        && !holds.reason(stopHoldKey(tab)) && !holds.reason(codeHoldKey(tab), tab.revision);

    // Mount one card and keep its head controls truthful: the facade stands in
    // while it starts (idempotent — never over a settling frame), a failed mount
    // leaves the error in the body and Start available for a retry, and a card
    // the owner left meanwhile ends stopped with its facade, never a stale Stop.
    async function startWidget(card, tab, isCurrent) {
        const mount = card.querySelector('[data-widget-mount]');
        if (isFramedWidget(tab)) renderWidgetFacade(mount, tab);
        syncWidgetCardControls(card, 'starting', effectiveStartMode(tab, uiPreferences));
        try {
            await mountTrackedTab(card, tab, isCurrent);
        } catch (err) {
            if (isCurrent() && mount) mount.innerHTML = `<div class="skills-load-error">widget failed: ${escapeHtml(err.message || err)}</div>`;
        }
        if (isCurrent()) syncWidgetCardControls(card, widgetDisposers.has(widgetKey(tab)) ? 'running' : 'stopped', effectiveStartMode(tab, uiPreferences));
        else if (isFramedWidget(tab)) await settleStopped(card, tab);
    }

    // A framed card that is not to run now: let an ordered stop still in flight
    // finish (its frame is still in the card), and let a mount under way for
    // the key (another card node, or a stale one about to bail) decide the body
    // first, then show the facade — unless the key ended up running. The card
    // never ends with an empty body behind a mount that bailed.
    async function settleStopped(card, tab) {
        const key = widgetKey(tab);
        const mode = effectiveStartMode(tab, uiPreferences);
        if (widgetDisposing.has(key)) {
            syncWidgetCardControls(card, 'stopping', mode);
            await widgetDisposing.get(key);
        }
        if (widgetMounting.has(key)) await widgetMounting.get(key).promise.catch(() => {});
        if (widgetDisposers.has(key) || !card.isConnected) return;
        renderWidgetFacade(card.querySelector('[data-widget-mount]'), tab);
        syncWidgetCardControls(card, 'stopped', mode);
        setWidgetCardNote(card, HELD_LINES[holds.reason(codeHoldKey(tab), tab.revision)] || '');
    }

    async function stopWidgetByOwner(card, tab) {
        holds.stop(stopHoldKey(tab));
        syncWidgetCardControls(card, 'stopping', effectiveStartMode(tab, uiPreferences));
        await disposeWidgetByKey(widgetKey(tab));
        await settleStopped(card, tab);
    }

    async function startWidgetByOwner(card, tab) {
        holds.release(stopHoldKey(tab));
        holds.release(codeHoldKey(tab));
        await startWidget(card, tab, isCurrentFor(renderGeneration));
    }

    // Launch-policy writes are a whole-map replace through the preferences API:
    // read the stored map first (a stale in-memory copy never drops another
    // card's choice), merge, POST. One at a time — two quick changes chain
    // behind each other, so the second reads the first's map and loses nothing.
    let startModeWrites = Promise.resolve();
    async function persistWidgetStartMode(key, mode) {
        let current = uiPreferences.widget_start_mode;
        try {
            const stored = await apiClient.uiPreferences();
            if (stored?.widget_start_mode && typeof stored.widget_start_mode === 'object') current = stored.widget_start_mode;
        } catch { /* fall back to the in-memory map */ }
        const next = withWidgetStartMode(current, key, mode);
        uiPreferences = { ...uiPreferences, widget_start_mode: next };
        await apiClient.saveUiPreferences({ widget_start_mode: next }).catch((err) => {
            console.warn('Failed to save widget start mode', err);
        });
    }

    // Launch-policy change from the card menu, applied once its write landed —
    // Auto / Keep running start a stopped card now, Manual changes nothing until Start.
    async function setWidgetStartMode(key, mode) {
        const tab = tabByKey(key);
        if (!tab || !WIDGET_START_MODES.includes(mode)) return;
        const write = startModeWrites.then(() => persistWidgetStartMode(key, mode));
        startModeWrites = write.catch(() => {});
        await write;
        const card = liveCardFor(list, key);
        if (!card) return;
        syncWidgetCardControls(card, widgetDisposers.has(key) ? 'running' : 'stopped', mode);
        if (mode !== 'manual' && !widgetDisposers.has(key) && widgetsVisible) await startWidgetByOwner(card, tab);
    }

    retryButton.addEventListener('click', reconcileWidgetList);
    // Remove on a card kept from the chat takes it off the board; the server's `extension_lifecycle` re-reads the list.
    async function removeKeptCard(key) {
        const tab = tabByKey(key);
        if (!tab?.pinned_card) return;
        const { ok, error } = await setCardPinned(tab.tab_id, false);
        const status = list.parentElement?.querySelector('[data-widget-arrange-status]');
        if (!ok && status) Object.assign(status, { textContent: `The card was not removed: ${error}` }).dataset.tone = 'error';
    }

    const cardMenus = bindWidgetCardMenus(list, setWidgetStartMode, widths, removeKeptCard);
    window.addEventListener('pagehide', (event) => {
        listRequests.abortAll();
        if (!event.persisted) { cardMenus.destroy(); widths.dispose(); }
    });
    list.addEventListener('click', (event) => {
        const power = event.target.closest('[data-widget-power]');
        if (!power || power.disabled) return;
        const card = power.closest('[data-widget-key]');
        const tab = tabByKey(card?.dataset.widgetKey || '');
        if (!card || !tab) return;
        if (widgetDisposers.has(widgetKey(tab))) stopWidgetByOwner(card, tab);
        else startWidgetByOwner(card, tab);
    });

    window.addEventListener('ouro:page-shown', (event) => {
        if (event.detail?.page === 'widgets') {
            render();
        } else {
            cardMenus.close();
            // Leaving Widgets cancels ITS sync; once hidden, an unrelated page change
            // must not cancel a skill-disable retention check (its kept frame would run on).
            if (widgetsVisible) listRequests.abortAll();
            // Leaving disposes the mounted work — except the frames the owner
            // keeps running, which stay mounted in the hidden page — and stops
            // stale paints; the cards stay in the DOM so the next entry mounts
            // into them instead of rebuilding. A stopped framed card ends with
            // its facade and Start once its stop settled, so a hidden or
            // returned card never shows a stale Stop / Running over an empty body.
            widgetsVisible = false;
            widgetsMounted = false;
            const stopping = Array.from(widgetDisposers.keys()).filter((key) => !retainsWhileHidden(key));
            disposeMountedWidgets(retainsWhileHidden);
            for (const key of stopping) {
                const card = liveCardFor(list, key);
                const tab = tabByKey(key);
                if (card && tab && isFramedWidget(tab)) settleStopped(card, tab);
            }
        }
    });
}
