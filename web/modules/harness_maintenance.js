// Family-level program maintenance. Claudexor owns install selection, supported
// targets and durable operations. This view owns only pending presses and reads:
// no updater, auth action, task retry, daemon wake or durable operation ledger.
// Accounts' existing visible status tick drives poll(); there is no second timer.
import { apiClient } from './api_client.js';
import { openConfirmDialog } from './confirm_dialog.js';
import { escapeHtmlAttr as escapeHtml } from './utils.js';

const ACTIVE = new Set(['queued', 'running']);
const TERMINAL = new Set(['succeeded', 'failed', 'cancelled', 'interrupted']);
// Survives Settings remounts, including a lost POST reply. A full page reload
// recovers accepted operation handles from the engine's inventory instead.
const sessionIntents = new Map();

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
    const managed = selected.kind === 'managed';
    const version = unknownCurrent ? null : selected.version;
    const ownership = managed ? 'Managed' : selected.kind === 'missing' ? 'Not installed' : 'External install';
    const latest = entry.available?.version
        ? `Latest ${entry.available.version}` : 'Latest not checked';
    return [`Program ${version || 'version unknown'}`, ownership, latest,
        ...(inspectionError ? ['Last known'] : [])].join(' · ');
}

export function maintenanceOperationLine(operation) {
    if (!operation) return '';
    const version = operation.target?.version || operation.targetVersion;
    const target = version ? ` ${version}` : '';
    const labels = {
        queued: `Update queued${target}`,
        running: operation.phase === 'installing' ? `Installing${target}…` : `Preparing update${target}…`,
        succeeded: `Operation completed${operation.after?.version ? ` · Version ${operation.after.version}` : ''}`,
        failed: 'Update failed', cancelled: 'Update cancelled', interrupted: 'Update interrupted',
    };
    let line = labels[operation.state] || 'Operation status unknown';
    if (operation.termination === 'unconfirmed') line += ' · Installer may still be running';
    if (operation.mutation === 'unknown') line += ' · Installed files may have changed';
    if (operation.after?.selected === false) line += ' · This install is not the selected program';
    return line;
}

const LIMITATIONS = {
    in_place_replacement: 'Program files are replaced in place.',
    new_starts_may_fail: 'New sessions may fail to start until installation finishes.',
    return_requires_registry: 'Returning to an earlier version requires the package to remain available for download.',
    previous_unknown: 'The previous version is unknown.',
    vendor_resolves_latest: 'The vendor chooses the available version during the update.',
    no_version_change_observed: 'No version change was observed. The command does not confirm that the newest version is installed.',
};

function problemText(problem) {
    return String(problem?.message || problem?.code || '');
}

// What an update does, said once where the owner decides on one (it used to sit in the section's copy).
const UPDATE_NOTE = 'Program updates replace files in place and may interrupt new starts. Returning to an earlier version may require a download. Retry a waiting task separately.';
const DETAILS_CARET = '<svg class="settings-disclosure-caret" width="10" height="10" viewBox="0 0 10 10" aria-hidden="true" focusable="false"><path d="M3.5 2 7 5 3.5 8" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"></path></svg>';

function button(action, label, disabled = false) {
    return `<button type="button" class="btn btn-default" data-maintenance-action="${action}"${disabled ? ' disabled' : ''}>${escapeHtml(label)}</button>`;
}

export function harnessMaintenanceMarkup(entry, view = {}) {
    const operation = view.request ? view.operation : (view.operation || entry?.operation);
    const pending = Boolean(view.request) || maintenanceOperationPending(operation);
    const busy = Boolean(view.busy);
    const targets = entry?.maintainable && !view.inspectionError ? (entry.targets || []) : [];
    const canAct = !pending && !busy && !view.inspectionPending;
    const actions = [];
    if (view.request && !operation) {
        actions.push(button('rejoin', busy ? 'Sending…' : 'Retry same request', busy));
    } else if (pending) {
        if (view.operationError || !ACTIVE.has(operation?.state)) actions.push(button('rejoin', view.reading ? 'Checking…' : 'Check operation', busy || view.reading));
        if (operation?.id) actions.push(button('cancel', view.cancelling ? 'Cancelling…' : 'Cancel update', busy || view.cancelling));
    } else {
        const checkLabel = entry?.canCheckLatest ? 'Check latest' : 'Check version';
        actions.push(button('inspect', view.inspectionPending ? 'Checking…' : checkLabel, busy || view.inspectionPending));
        if (targets.includes('latest')) actions.push(button('latest', 'Update', !canAct));
    }
    const extraActions = [];
    if (targets.includes('version')) extraActions.push(button('version', 'Install exact version…', !canAct));
    if (targets.includes('previous') && entry.previous?.version) {
        extraActions.push(button('previous', `Return to ${entry.previous.version}`, !canAct));
    }
    if (targets.includes('baseline') && entry.releaseTested?.version) {
        extraActions.push(button('baseline', `Install release version ${entry.releaseTested.version}`, !canAct));
    }
    // This view's own sentences (an unconfirmed acceptance, an unavailable inspection) say what to do next and stay
    // on the row; the engine's exact problem text is a diagnostic, shown under Details. Only a typed source gets a
    // short visible cause: the inventory's problem with the latest version is about the latest version.
    const error = view.operationError || view.inspectionError;
    const engineProblem = problemText(operation?.problem) || problemText(entry?.availableProblem);
    const notice = error || engineProblem;
    const cause = !error && !problemText(operation?.problem) && engineProblem ? 'Latest version not determined' : '';
    const status = `${view.operationError && operation ? 'Last observed: ' : ''}${maintenanceOperationLine(operation)}`;
    const tone = notice || ['failed', 'interrupted'].includes(operation?.state) ? 'warn' : 'muted';
    const selected = entry?.selection || {};
    const facts = [
        ['Selected program', selected.binary || 'Unknown'],
        ...(selected.overrideEnv ? [['Selected by', selected.overrideEnv]] : []),
        ['Managed copy', view.unknownCurrent ? 'Version unknown until inspected' : entry?.installed?.version || 'Absent or unknown'],
        ['Bundled baseline', entry?.releaseTested?.version || 'Unknown'],
        ...(entry?.releaseTested?.verification ? [['Release verification', entry.releaseTested.verification === 'release_verified' ? 'Release verified' : 'Deterministic checks only']] : []),
        ...(entry?.available?.observedAt ? [['Latest checked', entry.available.observedAt]] : []),
        ...(operation?.id ? [['Operation', operation.id]] : []),
    ];
    const details = facts.map(([label, value]) => `<div><dt>${escapeHtml(label)}</dt><dd>${escapeHtml(value)}</dd></div>`).join('');
    const limitations = (operation?.limitations || []).map((code) => LIMITATIONS[code] || code).join(' ');
    const progress = (operation?.progress || []).join('\n');
    const panelId = `harness-maintenance-details-${String(entry?.harness || view.harness || 'program').replace(/[^A-Za-z0-9_-]/g, '-')}`;
    // One state row: the operation's status, a short typed cause or this view's own sentence, then the Details
    // toggle, whose panel holds the exact engine text, what an update does, the version facts and the rarer actions.
    return `<div class="harness-maintenance-row">
        <div class="harness-maintenance-line">${escapeHtml(maintenanceVersionLine(entry, view))}</div>
        <div class="harness-maintenance-actions">${actions.join('')}</div>
        <div class="harness-maintenance-state">
            ${status ? `<span class="harness-maintenance-notice ui-status" data-tone="${tone}" role="status">${escapeHtml(status)}</span>` : ''}
            ${error ? `<span class="harness-maintenance-notice" data-tone="warn">${escapeHtml(error)}</span>` : ''}
            ${cause ? `<span class="harness-maintenance-cause">${escapeHtml(cause)}</span>` : ''}
            ${entry && !entry.maintainable ? `<span class="harness-maintenance-notice">${escapeHtml(entry.remedy || 'Updates are unavailable for the selected installation.')}</span>` : ''}
            <button type="button" class="settings-disclosure ui-hit-area" data-maintenance-details aria-expanded="${view.expanded ? 'true' : 'false'}" aria-controls="${panelId}">Details${DETAILS_CARET}</button>
        </div>
        <div class="harness-maintenance-details" id="${panelId}"${view.expanded ? '' : ' hidden'}>
            ${engineProblem ? `<p class="harness-maintenance-exact">Exact message: <code>${escapeHtml(engineProblem)}</code></p>` : ''}
            <p>${escapeHtml(UPDATE_NOTE)}</p>
            <dl>${details}</dl>
            ${limitations ? `<p>${escapeHtml(limitations)}</p>` : ''}
            ${progress ? `<pre>${escapeHtml(progress)}</pre>` : ''}
            ${extraActions.length ? `<div class="harness-maintenance-actions">${extraActions.join('')}</div>` : ''}
        </div>
    </div>`;
}

/** View controller. reads/presses coalesce; dispose releases only the view. */
export function createHarnessMaintenanceController({
    api = apiClient, host = () => null, onInventory = () => {}, onSettled = () => {},
    visible = () => true, dialog = openConfirmDialog, intents = sessionIntents,
    newKey = () => globalThis.crypto?.randomUUID?.()
        || `maintenance-${Date.now().toString(36)}-${Math.random().toString(36).slice(2)}`,
} = {}) {
    const rows = new Map();
    const reads = new Map();
    let disposed = false;
    let inventoryError = '';

    function record(harness) {
        if (!intents.has(harness)) intents.set(harness, { operation: null, request: null, expanded: false, revision: 0 });
        return intents.get(harness);
    }

    function render() {
        if (disposed) return;
        host()?.querySelectorAll('[data-family-maintenance]').forEach((element) => {
            const harness = element.dataset.familyMaintenance;
            const state = record(harness);
            // A rebuild keeps the focused control: the same action or Details toggle takes the focus back.
            const active = element.ownerDocument?.activeElement;
            const focused = active && element.contains?.(active)
                ? (active.dataset?.maintenanceAction ? `[data-maintenance-action="${active.dataset.maintenanceAction}"]`
                    : active.hasAttribute?.('data-maintenance-details') ? '[data-maintenance-details]' : '') : '';
            element.innerHTML = harnessMaintenanceMarkup(rows.get(harness), {
                ...state, harness, inspectionError: state.inspectionError || inventoryError,
                inspectionPending: reads.has(harness) || reads.has('*'),
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
        });
    }

    function accept(harness, operation) {
        if (!operation?.id || operation.harness && operation.harness !== harness) throw new Error('No matching operation receipt was returned.');
        const state = record(harness);
        const previous = state.operation;
        state.operation = operation;
        state.request = null;
        state.operationError = '';
        if (!ACTIVE.has(operation.state)) state.cancelling = false;
        // A later version probe answers what is installed now, independently
        // of a historical operation's effect or process-termination evidence.
        const inspectedAfter = inspectedAfterOperation(rows.get(harness), operation);
        if (operation.phase === 'installing' || operation.mutation === 'unknown') state.unknownCurrent = !inspectedAfter;
        if (TERMINAL.has(operation.state) && (previous?.id !== operation.id || !TERMINAL.has(previous?.state))) {
            state.revision += 1;
            state.unknownCurrent = operation.mutation !== 'none' && !inspectedAfter;
            void refresh({ harness, fresh: true });
            if (!disposed) onSettled(harness, operation);
        }
        return operation;
    }

    function refresh({ harness = '', fresh = false, checkLatest = false } = {}) {
        if (disposed) return Promise.resolve(null);
        const key = harness || '*';
        if (reads.has(key)) return reads.get(key);
        const revisions = new Map([...intents].map(([id, state]) => [id, state.revision]));
        let superseded = false;
        const promise = Promise.resolve().then(() => api.harnessMaintenanceInventory({ harness, fresh, checkLatest }))
            .then((inventory) => {
                if (!Array.isArray(inventory?.harnesses)) throw new Error('Program inventory was not returned.');
                inventoryError = '';
                for (const entry of inventory.harnesses) {
                    const state = record(entry.harness);
                    if (state.revision !== (revisions.get(entry.harness) || 0)) {
                        superseded = true;
                        continue;
                    }
                    rows.set(entry.harness, entry);
                    state.inspectionError = '';
                    state.unknownCurrent = (ACTIVE.has(state.operation?.state) && state.operation?.phase === 'installing')
                        || (state.operation?.mutation === 'unknown' && !inspectedAfterOperation(entry, state.operation));
                    // Inventory offers a handle, not the full result. Do not
                    // replace a retained full receipt with its shorter summary.
                    if (!state.request && entry.operation?.id && entry.operation.id !== state.operation?.id) {
                        state.operation = entry.operation;
                        state.request = null;
                        state.operationError = '';
                        void readOperation(entry.harness);
                    }
                }
                if (!disposed) onInventory([...rows.keys()]);
                return inventory;
            }).catch((error) => {
                const message = `Program inspection unavailable: ${error.message || error}`;
                if (harness) record(harness).inspectionError = message;
                else inventoryError = message;
                return null;
            }).finally(() => {
                reads.delete(key);
                render();
                if (superseded) void refresh({ harness, fresh: true });
            });
        reads.set(key, promise);
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
                const preparation = problem?.context?.scope === 'this_attempt'
                    && problem.context.acceptance === 'not_submitted';
                // A local preparation response knows only THIS attempt. It
                // cannot erase an earlier lost POST using the same key.
                if ((preparation || [400, 409].includes(error.status) && problem?.code)
                    && !(preparation && request.mayHaveSubmitted)) state.request = null;
                if (state.request && !preparation) request.mayHaveSubmitted = true;
                if (state.request?.mayHaveSubmitted) state.unknownCurrent = true;
                state.operationError = state.request
                    ? `Update acceptance unconfirmed: ${error.message || error}. Retry the same request to recover its result.`
                    : preparation && ['maintenance_restart_required', 'maintenance_preparation_failed'].includes(problem?.code)
                        ? String(error.message || error) : `Update not started: ${error.message || error}`;
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
        const entry = rows.get(harness);
        if (!entry?.maintainable || !entry.targets?.includes(target.kind)) return Promise.resolve(null);
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
        if (action === 'inspect') return refresh({ harness, fresh: true,
            checkLatest: rows.get(harness)?.canCheckLatest === true });
        if (action === 'rejoin') return record(harness).request ? submit(harness) : readOperation(harness);
        if (action === 'cancel') return cancel(harness);
        if (action === 'version') {
            const answer = await dialog({
                title: 'Install exact version', input: true,
                body: 'Enter the version to install. This can upgrade or downgrade the selected managed program. Files are replaced in place and a download may be required.',
                confirmLabel: 'Install version',
            });
            if (disposed || !answer?.confirmed || !answer.value.trim()) return;
            return start(harness, { kind: 'version', version: answer.value.trim() });
        }
        return start(harness, { kind: action });
    }

    return {
        refresh, render, start, cancel, act, readOperation,
        get harnesses() { return [...rows.keys()]; },
        view(harness) { return { ...record(harness), entry: rows.get(harness), inventoryError }; },
        poll() {
            if (disposed || !visible()) return;
            for (const [harness, state] of intents) {
                if (ACTIVE.has(state.operation?.state) && !state.operationError && !state.busy) void readOperation(harness);
            }
        },
        dispose() { disposed = true; },
    };
}
