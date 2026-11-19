// Family-level program maintenance. Claudexor owns install selection, supported
// targets and durable operations. This view owns only pending presses and reads:
// no updater, auth action, task retry, daemon wake or durable operation ledger.
// Accounts' existing visible status tick drives poll(); there is no second timer.
import { apiClient } from './api_client.js';
import { absoluteTime, escapeHtmlAttr as escapeHtml } from './utils.js';
import { errorWords } from './ui_primitives.js';
import { routeExternalLink } from './skill_presentation.js';

const ACTIVE = new Set(['queued', 'running']);
const TERMINAL = new Set(['succeeded', 'failed', 'cancelled', 'interrupted']);
// Survives Settings remounts, including a lost POST reply. A full page reload
// recovers accepted operation handles from the engine's inventory instead.
const sessionIntents = new Map();
const viewListeners = new WeakMap();
// The whole-inventory read's slot among them: its failure survives a remount like a row's.
const WHOLE_INVENTORY = '*';

/** What this session already read about a provider's program, or null: the list of providers says one fact of it
 *  (Accounts and keys) and reads nothing itself. */
export const knownProgram = (harness) => sessionIntents.get(String(harness))?.entry || null;

export function maintenanceOperationPending(operation) {
    return Boolean(operation?.id) && (!TERMINAL.has(operation.state)
        || operation.termination === 'unconfirmed');
}

function inspectedAfterOperation(entry, operation) {
    return TERMINAL.has(operation?.state)
        && Date.parse(entry?.observedAt) > Date.parse(operation.finishedAt);
}

export function maintenanceVersionLine(entry, { unknownCurrent = false, inspectionError = '' } = {}) {
    if (!entry) return 'Program version not checked';
    const selected = entry.selection || {};
    const version = selected.path ? `Program ${unknownCurrent ? 'version unknown' : selected.version || 'version unknown'}` : 'Not installed';
    const observed = absoluteTime(entry.observedAt);
    return version + (inspectionError ? ` · Last known${observed ? ` ${observed}` : ''}` : '');
}

export function maintenanceLatestLine(entry) {
    if (!entry) return '';
    if (entry.updateAvailable === 'no') return 'Up to date';
    if (entry.updateAvailable === 'yes' && entry.newest?.status === 'available') return `${entry.newest.version} available`;
    return 'Latest unknown';
}

export function maintenanceOperationLine(operation, { missing = false } = {}) {
    if (!operation) return '';
    const verb = missing ? 'Install' : 'Update';
    const outcome = { changed: operation.before?.version ? `Updated from ${operation.before.version}` : 'Installed',
        not_changed: 'Nothing changed', changed_elsewhere: 'Selected program unchanged', unknown: 'Result unknown' };
    let line = { queued: `${verb} queued`, running: missing ? 'Installing…' : 'Updating…',
        succeeded: outcome[operation.outcome] || 'Result unknown', failed: `${verb} failed`,
        cancelled: `${verb} cancelled`, interrupted: `${verb} interrupted` }[operation.state] || 'Operation status unknown';
    if (operation.termination === 'unconfirmed') line += ' · Installer may still be running';
    if (operation.mutation === 'unknown') line += ' · Installed files may have changed';
    return line;
}

const problemText = (problem) => String(problem?.message || problem?.code || '');
const DETAILS_CARET = '<svg class="settings-disclosure-caret" width="10" height="10" viewBox="0 0 10 10" aria-hidden="true" focusable="false"><path d="M3.5 2 7 5 3.5 8" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"></path></svg>';
const offeredAction = (entry) => entry?.actions?.[entry.selection?.path ? 'update' : 'install'];
const returnVersion = (entry, operation) => operation?.outcome === 'changed' && entry?.actions?.exactVersion?.available
    ? operation.before?.version : null;

function button(action, label, disabled = false) {
    return `<button type="button" class="btn btn-default" data-maintenance-action="${action}"${disabled ? ' disabled' : ''}>${escapeHtml(label)}</button>`;
}

function commandText(command) {
    const quote = (value) => /^[A-Za-z0-9_./:={}+-]+$/.test(String(value)) ? String(value) : JSON.stringify(value);
    const env = Object.entries(command.env || {}).map(([key, value]) => `${key}=${quote(value)}`);
    if (command.kind === 'script') return [...env, command.interpreter, ...(command.interpreterArgs || []),
        command.url, ...(command.args || [])].map(quote).join(' ');
    return [...env, command.binary || command.command, ...(command.args || [])].map(quote).join(' ');
}

export function harnessMaintenanceMarkup(entry, view = {}) {
    const operation = view.operation || entry?.operation;
    const pending = Boolean(view.request) || maintenanceOperationPending(operation);
    const busy = Boolean(view.busy);
    const missing = !entry?.selection?.path;
    const action = offeredAction(entry);
    const canAct = !pending && !busy && !view.inspectionPending && !view.inspectionError;
    const actions = [];
    const extraActions = [button('inspect', view.inspectionPending ? 'Checking…' : 'Check latest', busy || view.inspectionPending)];
    if (view.request && !operation) actions.push(button('rejoin', busy ? 'Sending…' : 'Retry same request', busy));
    else if (pending) {
        actions.push(button('latest', missing ? 'Install' : 'Update', true));
        if (view.operationError || !ACTIVE.has(operation?.state)) extraActions.push(button('rejoin', view.reading ? 'Checking…' : 'Check operation', busy || view.reading));
        if (operation?.id) extraActions.push(button('cancel', view.cancelling ? 'Cancelling…' : 'Cancel update', busy || view.cancelling));
    } else if (action?.available && !view.inspectionError && (missing || entry.updateAvailable !== 'no')) {
        actions.push(button('latest', missing ? 'Install' : 'Update', !canAct));
    }
    const prior = returnVersion(entry, operation);
    if (prior) extraActions.push(button('return', `Return to ${prior}`, !canAct));
    const error = view.operationError || (view.inspectionProblem ? '' : view.inspectionError);
    const engineProblem = problemText(operation?.problem) || problemText(entry?.newest?.problem);
    const status = `${view.operationError && operation ? 'Last observed: ' : ''}${maintenanceOperationLine(operation, { missing })}`;
    const tone = error || engineProblem || ['failed', 'interrupted'].includes(operation?.state) ? 'warn' : 'muted';
    const selected = entry?.selection || {};
    const planned = (action?.commands || []).map(commandText).join('\n');
    const ran = (operation?.commands || []).map(commandText).join('\n');
    const facts = [
        ['Program path', selected.path || 'Not installed'],
        ...(selected.realPath && selected.realPath !== selected.path ? [['Resolved path', selected.realPath]] : []),
        ...(selected.overrideValue ? [['Override', `${selected.overrideEnv}=${selected.overrideValue}`]] : []),
        ...(planned ? [['Command', planned]] : []),
        ...(entry?.newest?.observedAt ? [['Latest read', absoluteTime(entry.newest.observedAt)]] : []),
        ...(operation ? [
            ['Last operation', absoluteTime(operation.finishedAt || operation.startedAt || operation.createdAt)],
            ...(ran && ran !== planned ? [['Ran', ran]] : []),
            ['Result', `${operation.before?.version || 'Not installed'} → ${operation.after?.version || 'Unknown'}${operation.exitCode == null ? '' : ` · Exit ${operation.exitCode}`}`],
            ...(operation.requester?.source ? [['Requested by', operation.requester.source]] : []),
            ...(operation.activeRunsAtStart != null ? [['Running sessions at start', String(operation.activeRunsAtStart)]] : []),
            ...(operation.installationHome ? [['Installation home', operation.installationHome]] : []),
        ] : []),
        ...(operation?.commands || []).filter((command) => command.script).map(({ script }) =>
            ['Installer script', `${script.url} · ${script.size} bytes · SHA-256 ${script.sha256}`]),
    ];
    const panelId = `harness-maintenance-details-${String(entry?.harness || view.harness || 'program').replace(/[^A-Za-z0-9_-]/g, '-')}`;
    return `<div class="harness-maintenance-row">
        <div class="harness-maintenance-line harness-maintenance-facts"><span>${escapeHtml(maintenanceVersionLine(entry, view))}</span>
            <span class="harness-maintenance-cause">${escapeHtml(maintenanceLatestLine(entry))}</span></div>
        <div class="harness-maintenance-actions">${actions.join('')}</div>
        <div class="harness-maintenance-state">
            ${status ? `<span class="harness-maintenance-notice ui-status" data-tone="${tone}" role="status">${escapeHtml(status)}</span>` : ''}
            ${error ? `<span class="harness-maintenance-notice" data-tone="warn">${escapeHtml(error)}</span>` : ''}
            ${action && !action.available ? `<span class="harness-maintenance-notice">${escapeHtml(action.reason)}</span>` : ''}
            <button type="button" class="settings-disclosure ui-hit-area" data-maintenance-details aria-expanded="${!!view.expanded}" aria-controls="${panelId}">Details${DETAILS_CARET}</button>
        </div>
        <div class="harness-maintenance-details" id="${panelId}"${view.expanded ? '' : ' hidden'}>
            ${engineProblem ? `<p class="harness-maintenance-exact">Exact message: <code>${escapeHtml(engineProblem)}</code></p>` : ''}
            <dl>${facts.map(([label, value]) => `<div><dt>${escapeHtml(label)}</dt><dd>${escapeHtml(value)}</dd></div>`).join('')}</dl>
            ${(operation?.limitations || []).map((sentence) => `<p>${escapeHtml(sentence)}</p>`).join('')}
            ${operation?.progress?.length ? `<pre>${escapeHtml(operation.progress.join('\n'))}</pre>` : ''}
            <div class="harness-maintenance-actions">${extraActions.join('')}
                ${entry?.releaseNotesUrl ? `<a class="btn btn-default" href="${escapeHtml(entry.releaseNotesUrl)}" target="_blank" rel="noopener noreferrer" data-skill-external>Release notes</a>` : ''}</div>
        </div>
    </div>`;
}

/** View controller. reads/presses coalesce; dispose releases only the view. */
export function createHarnessMaintenanceController({
    api = apiClient, host = () => null, onInventory = () => {}, onInspectionProblem = () => {}, onSettled = () => {}, onChange = () => {},
    visible = () => true, intents = sessionIntents,
    newKey = () => globalThis.crypto?.randomUUID?.()
        || `maintenance-${Date.now().toString(36)}-${Math.random().toString(36).slice(2)}`,
} = {}) {
    if (!viewListeners.has(intents)) viewListeners.set(intents, new Set());
    const listeners = viewListeners.get(intents);
    listeners.add(onChange);
    const harnesses = () => [...intents].filter(([, state]) => state.entry).map(([id]) => id);
    const reads = new Map();
    let disposed = false;
    // Answers can cross: the later-asked read stands, whichever answers last. `observed` keeps, per slot,
    // the newest applied failure or proved success (a cached success proves nothing current, so it marks
    // nothing); `engineObserved` the newest current observation of the engine-wide fact. A released view
    // applies nothing; its successor retains the last applied row, with a fresh read sequence.
    let asked = 0;
    let engineObserved = 0;
    const observed = new Map();

    function record(harness) {
        if (!intents.has(harness)) intents.set(harness, { operation: null, request: null, expanded: false, revision: 0 });
        return intents.get(harness);
    }

    function inspectionProblem() {
        for (const state of intents.values()) if (state.inspectionProblem) return state.inspectionProblem;
        return null;
    }

    function clearInspectionFailure(state) {
        state.inspectionError = '';
        state.inspectionNeedsFresh = false;
        state.inspectionProblem = null;
    }

    function forgetMissingExecutable() {
        for (const state of intents.values()) if (state.inspectionProblem) clearInspectionFailure(state);
    }

    // "Checking…" answers a check the owner asked for: this card's own, or a fresh read of every program (Refresh);
    // a cached read changes no card's words (U106).
    const inspectionPending = (harness) => reads.has(harness) || Boolean(reads.get(WHOLE_INVENTORY)?.current);

    function render() {
        if (disposed) return;
        const problem = inspectionProblem();
        const whole = record(WHOLE_INVENTORY);
        host()?.querySelectorAll('[data-family-maintenance]').forEach((element) => {
            const harness = element.dataset.familyMaintenance;
            const state = record(harness);
            // A rebuild keeps the focused control: the same action or Details toggle takes the focus back.
            const active = element.ownerDocument?.activeElement;
            const focused = active && element.contains?.(active)
                ? (active.dataset?.maintenanceAction ? `[data-maintenance-action="${active.dataset.maintenanceAction}"]`
                    : active.hasAttribute?.('data-maintenance-details') ? '[data-maintenance-details]' : '') : '';
            element.innerHTML = harnessMaintenanceMarkup(record(harness).entry, {
                ...state, harness, inspectionProblem: problem,
                inspectionError: state.inspectionError || whole.inspectionError || problem?.message || '',
                inspectionPending: inspectionPending(harness),
            });
            if (focused) element.querySelector(focused)?.focus?.();
            element.querySelector('[data-maintenance-details]')?.addEventListener('click', (event) => {
                state.expanded = !state.expanded;
                event.currentTarget.setAttribute('aria-expanded', String(state.expanded));
                const panel = element.querySelector('.harness-maintenance-details');
                if (panel) panel.hidden = !state.expanded;
            });
            element.querySelectorAll('[data-maintenance-action]').forEach((control) => {
                control.addEventListener('click', () => act(harness, control.dataset.maintenanceAction));
            });
            element.querySelectorAll('[data-skill-external]').forEach((control) => control.addEventListener('click', routeExternalLink));
        });
        for (const changed of listeners) changed();
    }

    function accept(harness, operation, fromInventory = false, settled = onSettled) {
        if (!operation?.id || operation.harness && operation.harness !== harness) throw new Error('No matching operation receipt was returned.');
        const state = record(harness);
        const previous = state.operation;
        state.operation = operation;
        state.request = null;
        state.operationError = '';
        if (!ACTIVE.has(operation.state)) state.cancelling = false;
        // A later version probe answers what is installed now, independently
        // of a historical operation's effect or process-termination evidence.
        const inspectedAfter = inspectedAfterOperation(record(harness).entry, operation);
        if (operation.phase === 'installing' || operation.mutation === 'unknown') state.unknownCurrent = !inspectedAfter;
        if (TERMINAL.has(operation.state) && (previous?.id !== operation.id || previous.state !== operation.state
            || previous.termination !== operation.termination)) state.revision += 1;
        if (TERMINAL.has(operation.state) && !maintenanceOperationPending(operation)
            && (!fromInventory || state.inventorySeen || maintenanceOperationPending(previous))
            && (previous?.id !== operation.id || maintenanceOperationPending(previous))) {
            state.unknownCurrent = operation.mutation !== 'none' && !inspectedAfter;
            if (!fromInventory) void refresh({ harness, fresh: true });
            if (!disposed) settled(harness, operation);
        }
        return operation;
    }

    function refresh({ harness = '', fresh = false, checkLatest = false } = {}) {
        if (disposed) return Promise.resolve(null);
        const key = harness || WHOLE_INVENTORY;
        // A running read answers only a request no stronger than itself: joining a cached
        // read would report its success as the fresh inspection this caller asked for.
        const asksCurrent = fresh || checkLatest;
        const running = reads.get(key);
        if (running) {
            return (running.current || !asksCurrent) && (running.checkLatest || !checkLatest)
                ? running.promise : running.promise.then(() => refresh({ harness, fresh, checkLatest }));
        }
        const revisions = new Map([...intents].map(([id, state]) => [id, state.revision]));
        const order = ++asked;
        const older = (slot) => (observed.get(slot) || 0) > order;
        const claim = (slot) => {
            if (older(slot)) return false;
            observed.set(slot, order);
            return true;
        };
        const claimEngine = () => {
            if (engineObserved > order) return false;
            engineObserved = order;
            return true;
        };
        let superseded = false;
        const promise = Promise.resolve().then(() => api.harnessMaintenanceInventory({ harness, fresh, checkLatest }))
            .then((inventory) => {
                // A released view's late answer is older than anything its successor reads.
                if (disposed) return null;
                if (!Array.isArray(inventory?.harnesses)) throw new Error('Program inventory was not returned.');
                // Only a child that really ran proves anything current: a cached answer keeps
                // a failed fresh inspection standing, and an empty one spawned nothing.
                const proved = asksCurrent && inventory.harnesses.length > 0;
                if (proved && claimEngine()) forgetMissingExecutable();
                const settle = (slot) => {
                    const state = record(slot);
                    if (proved ? claim(slot) : !older(slot) && !state.inspectionNeedsFresh) clearInspectionFailure(state);
                };
                settle(WHOLE_INVENTORY);
                let settlement;
                for (const entry of inventory.harnesses) {
                    const state = record(entry.harness);
                    if (state.revision !== (revisions.get(entry.harness) || 0)) {
                        superseded = true;
                        continue;
                    }
                    state.entry = entry;
                    settle(entry.harness);
                    state.unknownCurrent = (ACTIVE.has(state.operation?.state) && state.operation?.phase === 'installing')
                        || (state.operation?.mutation === 'unknown' && !inspectedAfterOperation(entry, state.operation));
                    if (!state.request && entry.operation?.id && (entry.operation.id !== state.operation?.id
                        || maintenanceOperationPending(state.operation))) accept(entry.harness, entry.operation, true,
                            (...args) => { settlement ||= args; });
                    state.inventorySeen = true;
                }
                if (settlement) onSettled(...settlement);
                if (!disposed) onInventory(harnesses());
                return inventory;
            }).catch((error) => {
                // A late answer to a released view cannot overwrite what its successor has since read.
                if (disposed) return null;
                const problem = error.body?.error;
                const missing = problem?.context?.observation === 'reported_executable_missing' ? problem : null;
                const current = Boolean(asksCurrent || missing);
                const state = record(key);
                // A newer observation of this slot stands, a failed cached read cannot replace a current
                // failure, and only the newest engine-wide observation may say the file is missing.
                if (older(key) || (!current && state.inspectionNeedsFresh) || (missing && engineObserved > order)) return null;
                const before = inspectionProblem();
                claim(key);
                if (current && claimEngine()) forgetMissingExecutable();
                // A transport failure is said in owner words (a TypeError with no response: errorWords); an engine's own
                // sentence stays on the row (an unclassified failure keeps its exact sentence).
                const reason = error instanceof TypeError && error.response === undefined ? errorWords(error).text : error.message || error;
                state.inspectionError = `Program inspection unavailable: ${reason}`;
                state.inspectionNeedsFresh = current;
                state.inspectionProblem = missing;
                if (inspectionProblem() !== before) onInspectionProblem();
                return null;
            }).finally(() => {
                reads.delete(key);
                render();
                if (superseded) void refresh({ harness, fresh: true });
            });
        reads.set(key, { promise, current: asksCurrent, checkLatest });
        render();
        return promise;
    }

    function readOperation(harness) {
        const state = record(harness);
        if (!state.operation?.id || state.reading || disposed) return state.reading || Promise.resolve(null);
        const id = state.operation.id;
        const revision = state.revision;
        state.reading = Promise.resolve().then(() => api.harnessMaintenanceOperation(id))
            .then((operation) => state.operation?.id === id && state.revision === revision ? accept(harness, operation) : null)
            .catch((error) => {
                if (state.revision === revision) state.operationError = `Operation status unconfirmed: ${error.message || error}. Check the same operation before starting another update.`;
                return null;
            }).finally(() => { state.reading = null; render(); });
        render();
        return state.reading;
    }

    function submit(harness) {
        const state = record(harness);
        if (state.busy || disposed) return state.busy || Promise.resolve(null);
        const request = state.request;
        if (!request) return readOperation(harness);
        state.operationError = '';
        state.busy = Promise.resolve().then(() => api.startHarnessMaintenance(request.body, request.key))
            .then((operation) => accept(harness, operation))
            .catch(async (error) => {
                const problem = error.body?.error || error.body;
                if (problem?.code === 'maintenance_already_active' && problem.context?.operationId) {
                    state.request = null;
                    state.operation = { id: problem.context.operationId, state: 'running' };
                    return readOperation(harness);
                }
                // A typed pre-acceptance refusal has no installation effect.
                // Network/5xx/non-JSON replies retain exactly this key/body.
                const notSubmitted = problem?.context?.scope === 'this_attempt'
                    && problem.context.acceptance === 'not_submitted';
                // A local acquisition refusal knows only THIS attempt. It
                // cannot erase an earlier lost POST using the same key.
                if ((notSubmitted || [400, 409].includes(error.status) && problem?.code)
                    && !(notSubmitted && request.mayHaveSubmitted)) state.request = null;
                if (state.request && !notSubmitted) request.mayHaveSubmitted = true;
                if (state.request?.mayHaveSubmitted) state.unknownCurrent = true;
                state.operationError = state.request
                    ? `Update acceptance unconfirmed: ${error.message || error}. Retry the same request to recover its result.`
                    : `Update not started: ${error.message || error}`;
                return null;
            }).finally(() => { state.busy = null; render(); });
        render();
        return state.busy;
    }

    function start(harness, target) {
        if (disposed) return Promise.resolve(null);
        const state = record(harness);
        if (state.busy) return state.busy;
        if (state.request) return submit(harness);
        if (maintenanceOperationPending(state.operation)) return readOperation(harness);
        const entry = record(harness).entry;
        if (!(target.kind === 'latest' ? offeredAction(entry) : target.kind === 'version' ? entry?.actions?.exactVersion : null)?.available) return Promise.resolve(null);
        state.operation = null;
        state.revision += 1;
        state.request = { key: newKey(), body: { harness, target } };
        return submit(harness);
    }

    function cancel(harness) {
        const state = record(harness);
        if (state.busy || !state.operation?.id || disposed) return state.busy || Promise.resolve(null);
        state.cancelling = true;
        state.operationError = '';
        state.busy = Promise.resolve().then(() => api.cancelHarnessMaintenance(state.operation.id))
            .then((operation) => accept(harness, operation))
            .catch((error) => {
                state.cancelling = false;
                state.operationError = `Cancellation unconfirmed: ${error.message || error}. Check the operation for its result.`;
                return null;
            }).finally(() => { state.busy = null; render(); });
        render();
        return state.busy;
    }

    async function act(harness, action) {
        if (disposed) return;
        if (action === 'inspect') return refresh({ harness, fresh: true, checkLatest: true });
        if (action === 'rejoin') return record(harness).request ? submit(harness) : readOperation(harness);
        if (action === 'cancel') return cancel(harness);
        if (action === 'return') {
            const version = returnVersion(record(harness).entry, record(harness).operation || record(harness).entry?.operation);
            return version ? start(harness, { kind: 'version', version }) : null;
        }
        return start(harness, { kind: action });
    }

    return {
        refresh, render, start, cancel, act, readOperation,
        get harnesses() { return harnesses(); },
        get inspectionProblem() { return inspectionProblem(); },
        view(harness) {
            return { ...record(harness), entry: record(harness).entry, inspectionPending: inspectionPending(harness),
                inventoryError: record(WHOLE_INVENTORY).inspectionError || '' };
        },
        poll() {
            if (disposed || !visible()) return;
            for (const [harness, state] of intents) {
                if (maintenanceOperationPending(state.operation) && !state.operationError && !state.busy) void readOperation(harness);
            }
        },
        dispose() { disposed = true; listeners.delete(onChange); },
    };
}
