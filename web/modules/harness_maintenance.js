// Family-level program maintenance. Claudexor owns install selection, supported
// targets and durable operations. This view owns only pending presses and reads:
// no updater, auth action, task retry, daemon wake or durable operation ledger.
// Accounts' existing visible status tick drives poll(); there is no second timer.
import { apiClient } from './api_client.js';
import { absoluteTime, escapeHtmlAttr as escapeHtml } from './utils.js';
import { errorWords } from './ui_primitives.js';
import { routeExternalLink } from './skill_presentation.js';
import { restoreFocusWithoutReveal } from './scroll_fade.js';

const ACTIVE = new Set(['queued', 'running']);
const TERMINAL = new Set(['succeeded', 'failed', 'cancelled', 'interrupted']);
// Survives Settings remounts, including a lost POST reply. A full page reload
// recovers accepted operation handles from the engine's inventory instead.
const sessionIntents = new Map();
const viewListeners = new WeakMap();
// The whole-inventory read's slot among them: its failure survives a remount like a row's.
const WHOLE_INVENTORY = '*';
// The read clock lives on the shared `intents` map, so two controllers on one dispatcher compare one
// order; ANSWER_STAMP marks the inventory a read returned, with that read's order and supersession.
const READ_CLOCK = Symbol('read-clock');
const ANSWER_STAMP = Symbol('answer-stamp');
const nextReadOrder = (intents) => (intents[READ_CLOCK] = (intents[READ_CLOCK] || 0) + 1);

/** What this session already read about a provider's program, or null: the list of providers says one fact of it
 *  (Accounts and keys) and reads nothing itself. */
export const knownProgram = (harness) => sessionIntents.get(String(harness))?.entry || null;

export function maintenanceOperationPending(operation) {
    return Boolean(operation?.id) && (!TERMINAL.has(operation.state)
        || operation.termination === 'unconfirmed');
}

/** A read program row whose program is absent, as `{ installable, failed, byHand }`; null while unread or installed.
 *  The engine's install action alone says whether Ouroboros can install it here; `failed`: the last install ended
 *  without it; `byHand`: either way the owner may install it where Claudexor runs and read it again. */
export function missingProgram(entry, operation = entry?.operation) {
    if (!entry || entry.selection?.path) return null;
    const installable = entry.actions?.install?.available === true;
    const failed = !maintenanceOperationPending(operation) && ['failed', 'interrupted'].includes(operation?.state)
        && !operation.before?.path;  // an install's: an Update that failed before the program went is no install to redo
    return { installable, failed, byHand: !installable || failed };
}

/** `missingProgram` of a controller's view, or null while its last read of that program or of all failed, or the
 *  newest answer left it out: a row from an earlier read (an engine since stopped, a refused fresh read, a shorter
 *  list) is no current fact to install by. The owner's request and operation stay on the record. */
export const currentMissing = (view) => {
    if (!view || view.inspectionError || view.inventoryError || view.listed === false) return null;
    const missing = missingProgram(view.entry, view.operation || view.entry?.operation);
    if (!missing) return missing;
    // While this program's own maintenance is under way the engine's refusal is no refusal to
    // install it here: the row keeps the absence, and never advises installing it by hand.
    if (view.request || maintenanceOperationPending(view.operation || view.entry?.operation))
        return { ...missing, byHand: false };
    return missing;
};

function inspectedAfterOperation(entry, operation) {
    return TERMINAL.has(operation?.state)
        && Date.parse(entry?.observedAt) > Date.parse(operation.finishedAt);
}

const sameSelection = (a, b) => Boolean(a && b) && a.path === b.path && a.version === b.version;

/** Whether the second observation is strictly newer. The engine stamps each program row with when
 *  it observed it, and the host keeps the same per-program order (harness_program_facts.py), so two
 *  comparable stamps order the facts about one program. */
const observedAfter = (later, earlier) => {
    const a = Date.parse(later || ''), b = Date.parse(earlier || '');
    return Number.isFinite(a) && Number.isFinite(b) && a > b;
};

export function maintenanceVersionLine(entry, { unknownCurrent = false, inspectionError = '' } = {}) {
    if (!entry) return 'Program version not checked';
    const selected = entry.selection || {};
    const version = selected.path ? `Program ${unknownCurrent ? 'version unknown' : selected.version || 'version unknown'}` : 'Not installed';
    const observed = absoluteTime(entry.observedAt);
    return version + (inspectionError ? ` · Last known${observed ? ` ${observed}` : ''}` : '');
}

export function maintenanceLatestLine(entry) {
    if (!entry) return '';
    // Beside "Not installed" only a newest release that was read is a fact worth saying.
    if (!entry.selection?.path) return entry.newest?.status === 'available' ? `${entry.newest.version} available` : '';
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

/** The engine's own sentence about a failure, in its words; shared with the surfaces that show the same row. */
export const maintenanceProblemText = (problem) => String(problem?.message || problem?.code || '');
const problemText = maintenanceProblemText;
// The one verified step by hand for the line the engine runs where its maker's own install gives another: OpenCode's
// documented install is the incompatible 2.x; `opencode-ai@1` ships the 1.x builds for macOS, Linux and Windows
// (claudexor #668; npm registry, 26.11.2026). A release-notes page is no install guide, and a recorded command never
// runs again from here.
const BY_HAND_STEP = { opencode: ['npm install -g opencode-ai@1', 'the 1.x line Claudexor runs'] };

/** How an absent program is installed by hand: a verified step where one is known, else (the engine refusing to install
 *  it here, often for want of a build) the way out, never a call to install what may not run here. */
export function byHandHtml(entry, operation = entry?.operation) {
    const missing = missingProgram(entry, operation);
    if (!missing?.byHand) return '';
    const [step, line] = BY_HAND_STEP[entry.harness] || [];
    if (step) return `To install it by hand where Claudexor runs, run <code>${escapeHtml(step)}</code> (${line}), then press Check again.`;
    return missing.installable ? '' : 'If this system cannot run it, choose another program or run Ouroboros on a system that supports it.';
}

/** While the engine's programs are not known (Claudexor stopped, or its inventory refused): what a subscription's
 *  Connect does about a missing program, said once and conditionally, then the engine's refusal (`reason`) or the
 *  owner's press that lists them by starting the agent daemon (the wake of Accounts), saying so before it is pressed. */
export function programsUnknownHtml({ starts = '', busy = false, error = '', reason = '' } = {}) {
    const press = starts ? `<button type="button" class="btn btn-default btn-sm" data-program-action="find"${busy ? ' disabled' : ''}>${
        busy ? 'Starting the agent daemon…' : `Show programs to install (${escapeHtml(starts)})`}</button>` : '';
    return `<span>If a program is missing, Connect tries to install it first.</span>${press}`
        + [error, reason].filter(Boolean).map((text) => `<span data-tone="warn">${escapeHtml(text)}</span>`).join('');
}

/** What that press does to the engine, from the status as last read: a runtime not known ready is prepared first. */
export const wakeWords = (payload) => (String(payload?.daemon?.runtime?.state || '') === 'ready'
    ? 'starts the agent daemon' : 'prepares and starts the agent daemon');
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
    // A program installed by hand is found by reading it again, so for an absent one that read stands beside Install.
    // A row the newest answered list left out (`listed=false`, `currentMissing`'s own rule) keeps its receipt and a
    // fresh Check; its retained facts are no current basis for an Install, an Update or advice to install by hand.
    const stale = view.listed === false;
    const byHand = !pending && !stale && missingProgram(entry, operation)?.byHand;
    // A read that failed leaves the row without a current fact: the manual way out is advice that fact can no
    // longer support (the rule model_program keeps for the same program). The way back — Check again — stays.
    const readError = view.inventoryError || (view.inspectionProblem ? '' : view.inspectionError);
    // An unresolved Install that must not be sent again is read, not repeated: the row offers the
    // read-only check and says why, and no enabled Retry stands in its place.
    const blocked = view.retryBlocked || '';
    const inspect = button('inspect', view.inspectionPending ? 'Checking…' : byHand || stale || blocked ? 'Check again' : 'Check latest', busy || view.inspectionPending);
    // While nothing is under way the fresh Check stands beside the withheld action; under way it keeps its Details place.
    const promoteCheck = (byHand || stale) && !pending;
    const extraActions = promoteCheck || blocked ? [] : [inspect];
    // The press names what it means: `install` (only for an absent program) or `latest`, the update (installed ones).
    const press = missing ? 'install' : 'latest';
    // A pending request wins over any historical receipt: the row offers Retry same request, never
    // a Check operation labelled by an older operation's receipt.
    if (view.request && !blocked) actions.push(button('rejoin', busy ? 'Sending…' : 'Retry same request', busy));
    else if (view.request) actions.push(inspect);
    else if (pending) {
        actions.push(button(press, missing ? 'Install' : 'Update', true));
        if (view.operationError || !ACTIVE.has(operation?.state)) extraActions.push(button('rejoin', view.reading ? 'Checking…' : 'Check operation', busy || view.reading));
        if (operation?.id) extraActions.push(button('cancel', view.cancelling ? 'Cancelling…' : missing ? 'Cancel install' : 'Cancel update', busy || view.cancelling));
    } else if (!stale && action?.available && !view.inspectionError && (missing || entry.updateAvailable !== 'no')) {
        actions.push(button(press, missing ? 'Install' : 'Update', !canAct));
    }
    if (promoteCheck) actions.push(inspect);
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
            ${blocked ? `<span class="harness-maintenance-notice" data-tone="warn">${escapeHtml(blocked)}</span>` : ''}
            ${action && !action.available ? `<span class="harness-maintenance-notice">${escapeHtml(action.reason)}</span>` : ''}
            ${byHand && !readError && byHandHtml(entry, operation) ? `<span class="harness-maintenance-notice">${byHandHtml(entry, operation)}</span>` : ''}
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
    visible = () => true, intents = sessionIntents, preservePlace = paint => paint(),
    newKey = () => globalThis.crypto?.randomUUID?.()
        || `maintenance-${Date.now().toString(36)}-${Math.random().toString(36).slice(2)}`,
} = {}) {
    if (!viewListeners.has(intents)) viewListeners.set(intents, new Set());
    const listeners = viewListeners.get(intents);
    listeners.add(onChange);
    const harnesses = () => [...intents].filter(([, state]) => state.entry).map(([id]) => id);
    const reads = new Map();
    let disposed = false;
    // Answers can cross: the later-asked read stands, whichever answers last. The read clock is shared
    // through `intents`; each intent's `readStamp` keeps the newest applied read of its slot, `engineStamp`
    // the newest current observation of the engine-wide fact.
    // A released view applies nothing; its successor retains the last applied row, with a fresh read sequence.

    function record(harness) {
        if (!intents.has(harness)) intents.set(harness, { operation: null, request: null, expanded: false, revision: 0 });
        return intents.get(harness);
    }

    // The owner's press belongs with the shared record, not with one view: a remounted surface still shows
    // the row of a program the owner pressed, even when its last read no longer stands.
    function markPressed(harness) {
        record(harness).pressed = true;
    }

    const pressed = (harness) => Boolean(record(harness).pressed);

    /** Why an Install press whose acceptance is unknown may not be sent again: the program has moved
 *  on (it is installed now, so the same request would update it instead), the engine refuses to
 *  install it, or its absence is no longer confirmed. The press's own intent is no evidence. */
    function installRetryBlocked(harness) {
        const state = record(harness);
        const request = state.request;
        if (!request || request.verb !== 'Install' || !request.mayHaveSubmitted) return '';
        const entry = state.entry;
        // The engine holds this very press under its own key: its receipt is the answer to it.
        if (entry?.operation?.requester?.requestId === request.key) return '';
        // The program is installed now: the same request would update it instead of installing it.
        if (entry?.selection?.path)
            return 'This Install is not sent again: the program is installed now, and the same request would update it instead.';
        // A failed read or a newer answer that left the program out leaves no current absence to send an
        // Install on: the row keeps its reason and offers a read-only Check again.
        if (!currentMissing({ ...state, inventoryError: inventoryError(harness) })?.installable)
            return 'This Install is not sent again: a fresh read no longer finds the program absent. Check again before installing.';
        return '';
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
    // The whole-inventory failure stands for a slot until the slot's own read of the same kind answers
    // newer: a current failure is outranked only by the slot's own current read; a cached one only by a
    // newer applied fact. A program's own fresh answer therefore clears it for itself alone.
    const inventoryError = (harness = '') => {
        const whole = record(WHOLE_INVENTORY);
        if (!harness || !whole.inspectionError) return whole.inspectionError || '';
        const slot = record(harness);
        const wholeStamp = whole.errorCurrent ? (whole.currentStamp || 0) : (whole.readStamp || 0);
        const slotStamp = whole.errorCurrent ? (slot.currentStamp || 0) : (slot.readStamp || 0);
        return wholeStamp > slotStamp ? whole.inspectionError : '';
    };
    // A program's row as this view holds it, with the facts a surface reads beside it (`currentMissing` takes it).
    const viewOf = (harness) => ({ ...record(harness), inspectionPending: inspectionPending(harness),
        inventoryError: inventoryError(harness), retryBlocked: installRetryBlocked(harness) });

    function render() {
        // A released view paints nothing, but a shared transition (a retained request, a settled
        // operation) still reaches the views that are mounted on the same dispatcher.
        if (!disposed) preservePlace(paintPrograms);
        for (const changed of listeners) changed();
    }

    function paintPrograms() {
        const problem = inspectionProblem();
        host()?.querySelectorAll('[data-family-maintenance]').forEach((element) => {
            const harness = element.dataset.familyMaintenance;
            const state = record(harness);
            // A rebuild keeps the focused control without scrolling to it: the same action or Details toggle takes it back.
            const active = element.ownerDocument?.activeElement;
            const focused = active && element.contains?.(active)
                ? (active.dataset?.maintenanceAction ? `[data-maintenance-action="${active.dataset.maintenanceAction}"]`
                    : active.hasAttribute?.('data-maintenance-details') ? '[data-maintenance-details]' : '') : '';
            element.innerHTML = harnessMaintenanceMarkup(record(harness).entry, {
                ...state, harness, inspectionProblem: problem, retryBlocked: installRetryBlocked(harness),
                inspectionError: state.inspectionError || inventoryError(harness) || problem?.message || '',
                inspectionPending: inspectionPending(harness),
            });
            if (focused) restoreFocusWithoutReveal(element.querySelector(focused));
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
            // The settlement belongs to the shared record: a released view cannot deliver it, so it waits
            // for the next live view that reads this operation.
            state.pendingSettlement = { harness, operation };
            if (!disposed) {
                state.pendingSettlement = null;
                settled(harness, operation);
            }
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
        const order = nextReadOrder(intents);
        // readStamp: the newest asked read that applied a fact (an entry, or membership) to the slot, cached
        // reads included — the later-asked read stands, whichever answers last. currentStamp: the newest
        // CURRENT read (fresh/checkLatest) that applied; only a current read outranks a failure, because a
        // cached answer proves nothing current.
        const older = (slot) => (record(slot).readStamp || 0) > order;
        const currentOlder = (slot) => (record(slot).currentStamp || 0) > order;
        const claim = (slot) => {
            if (older(slot)) return false;
            record(slot).readStamp = order;
            return true;
        };
        const claimCurrent = (slot) => {
            if (currentOlder(slot)) return false;
            record(slot).currentStamp = order;
            return true;
        };
        const claimEngine = () => {
            const whole = record(WHOLE_INVENTORY);
            if ((whole.engineStamp || 0) > order) return false;
            whole.engineStamp = order;
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
                    // A passive answer keeps the slot's read order too: the later-asked read stands, so an
                    // older answer that fails later cannot overwrite what this read already said.
                    if (proved ? claim(slot) && claimCurrent(slot)
                        : !older(slot) && !state.inspectionNeedsFresh && claim(slot))
                        clearInspectionFailure(state);
                };
                // The whole-inventory slot's failure belongs to a whole read only: one program's answer must
                // not clear the refusal its siblings still stand on.
                if (!harness) settle(WHOLE_INVENTORY);
                // An empty current answer still stamps its read's order, but never the freshness
                // clock: it proves nothing current, so it neither clears nor re-dates a failure.
                // A fresh answer proves current only where its own order still stands for that slot: a newer
                // cached read of the same program keeps the selection it applied, and a stale current stamp
                // would clear a refusal that belongs to another row.
                if (asksCurrent && claim(key) && proved) claimCurrent(key);
                const listed = new Set(inventory.harnesses.map((entry) => entry.harness));
                for (const [id, state] of intents) {
                    if (id === WHOLE_INVENTORY || (harness && id !== harness) || older(id)) continue;
                    state.listed = listed.has(id);
                    // An omission from an accepted inventory stands over older reads of that program: the
                    // inventory was read and did not list it. A cached inventory proves no freshness, so it
                    // claims the read's order only — never the current clock.
                    if (!listed.has(id)) claim(id);
                }
                let settlement;
                for (const entry of inventory.harnesses) {
                    const state = record(entry.harness);
                    // A settlement whose view was released waits here: the live view that reads this
                    // operation delivers it, once.
                    const pending = state.pendingSettlement;
                    if (pending && entry.operation?.id === pending.operation.id) {
                        state.pendingSettlement = null;
                        settlement ||= [pending.harness, pending.operation];
                    }
                    if (state.revision !== (revisions.get(entry.harness) || 0)) {
                        superseded = true;
                        continue;
                    }
                    // A later-asked read stands, whichever answers last; a current read (fresh or
                    // checkLatest) also moves the slot's freshness clock.
                    if (!claim(entry.harness)) continue;
                    if (asksCurrent) claimCurrent(entry.harness);
                    // A passive read never checked the release: its `not_read` must not overwrite an
                    // explicit checkLatest already held for the same selected program.
                    let row = entry;
                    if (!checkLatest && entry.newest?.status === 'not_read' && state.newestCheckedStamp
                        && sameSelection(state.entry?.selection, entry.selection))
                        // The release reading and its update answer are one outcome: a passive read
                        // that did not check the release keeps both.
                        row = { ...entry, newest: state.entry.newest, updateAvailable: state.entry.updateAvailable };
                    if (checkLatest) state.newestCheckedStamp = order;
                    // The program's facts are the engine's newest observation of it: an older
                    // observation does not replace a newer one, whatever order the reads were asked in.
                    if (observedAfter(state.entry?.observedAt, entry.observedAt)) row = state.entry;
                    state.entry = row;
                    settle(entry.harness);
                    state.unknownCurrent = (ACTIVE.has(state.operation?.state) && state.operation?.phase === 'installing')
                        || (state.operation?.mutation === 'unknown' && !inspectedAfterOperation(entry, state.operation));
                    // The engine holds this very press under its own key: that receipt is the answer
                    // to it. Another press's receipt never clears ours.
                    const receipt = entry.operation;
                    const ours = Boolean(state.request && receipt?.requester?.requestId
                        && receipt.requester.requestId === state.request.key);
                    if (receipt?.id && (!state.request || ours) && (receipt.id !== state.operation?.id
                        || maintenanceOperationPending(state.operation))) accept(entry.harness, receipt, true,
                            (...args) => { settlement ||= args; });
                    state.inventorySeen = true;
                }
                if (checkLatest && proved) for (const entry of inventory.harnesses) {
                    // An explicit release check that answered after a later-asked passive read still
                    // holds for the identical selected program.
                    const state = record(entry.harness);
                    if (state.entry && sameSelection(state.entry.selection, entry.selection)
                        && (state.newestCheckedStamp || 0) <= order) {
                        state.entry = { ...state.entry, newest: entry.newest, updateAvailable: entry.updateAvailable };
                        state.newestCheckedStamp = order;
                    }
                }
                if (settlement) onSettled(...settlement);
                if (!disposed) onInventory(harnesses());
                Object.defineProperty(inventory, ANSWER_STAMP, { value: { order, superseded } });
                return inventory;
            }).catch((error) => {
                // A late answer to a released view cannot overwrite what its successor has since read.
                if (disposed) return null;
                // An answer with no harness list is empty evidence, not a refusal: it stamps the
                // read's order only and never moves the failure clock.
                const emptyAnswer = error.message === 'Program inventory was not returned.';
                const problem = emptyAnswer ? null : error.body?.error;
                const missing = problem?.context?.observation === 'reported_executable_missing' ? problem : null;
                const current = !emptyAnswer && Boolean(asksCurrent || missing);
                const state = record(key);
                // A newer CURRENT observation of this slot stands (a cached answer never outranks a fresh
                // failure), a failed cached read cannot replace a current failure, a cached refusal that
                // a later-asked read already answered is no fact about this slot, and only the newest
                // engine-wide observation may say the file is missing.
                if (currentOlder(key) || (!current && state.inspectionNeedsFresh)
                    || (!current && older(key))
                    || (missing && (record(WHOLE_INVENTORY).engineStamp || 0) > order)) return null;
                claim(key);
                if (emptyAnswer) return null;
                const before = inspectionProblem();
                if (current) claimCurrent(key);
                state.errorCurrent = current;
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
        // A press whose acceptance is unknown is not sent again once the program has moved on: the
        // same key and body would now mean an Update. Its result is read instead.
        if (installRetryBlocked(harness)) { render(); return Promise.resolve(null); }
        // An Install whose acceptance is unknown is re-sent only after a fresh read of its own program:
        // the program may have been installed since the lost POST, and the read may restore its receipt.
        if (request.verb === 'Install' && request.mayHaveSubmitted) return retryInstall(harness, request);
        return post(harness, request);
    }

    // The one POST of a standing request. A refusal that knows only THIS attempt clears the request; a
    // reply that may have reached the engine keeps it, key and body, for a read-only recovery.
    function post(harness, request) {
        const state = record(harness);
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
                // What may be done with the retained request is the row's own control to say: a Retry where the
                // absence still stands, a read-only Check again where it no longer does (`retryBlocked`). The saved
                // reason names the gesture and its cause, never a second instruction that can contradict that control.
                state.operationError = state.request
                    ? `${request.verb} acceptance unconfirmed: ${error.message || error}`
                    : `${request.verb} not started: ${error.message || error}`;
                return null;
            }).finally(() => { state.busy = null; render(); });
        render();
        return state.busy;
    }

    // An unknown Install's retry: read its program fresh first. A current absence sends the same key and
    // body; an installed program, a refused read or a newer omission sends nothing, and a read that
    // restored the engine's receipt under this key answers the press without a second POST.
    function retryInstall(harness, request) {
        const state = record(harness);
        const checked = installableNow(harness).then((found) => {
            if (state.busy === checked) state.busy = null;
            render();
            if (disposed || !found || state.request !== request) return null;
            return installRetryBlocked(harness) ? null : post(harness, request);
        });
        state.busy = checked;
        return checked;
    }

    // An unresolved press of this program is rejoined, never replaced: a lost POST under its own key and body, or the
    // operation the engine accepted.
    function rejoin(harness) {
        const state = record(harness);
        return state.request ? submit(harness) : maintenanceOperationPending(state.operation) ? readOperation(harness) : null;
    }

    function send(harness, target) {
        // A released view sends nothing and writes nothing: the shared request and receipt are its successor's.
        if (disposed) return Promise.resolve(null);
        const state = record(harness);
        state.operation = null;
        state.revision += 1;
        state.request = { key: newKey(), body: { harness, target },
            verb: state.entry?.selection?.path ? 'Update' : 'Install' };
        return submit(harness);
    }

    // The read an Install is sent on: one fresh read of this program that answered, named it absent and
    // installable, and the standing view still agrees — a newer omission, refusal or installed reading
    // contradicts it, a newer agreeing absence does not lose the press. Another surface's absence, an
    // answer that leaves the program out or is empty, and a refused read prove nothing.
    async function installableNow(harness) {
        const answer = await refresh({ harness, fresh: true });
        const stamp = answer?.[ANSWER_STAMP];
        const named = (answer?.harnesses || []).find((item) => item.harness === harness);
        const namedAbsent = Boolean(named) && !named.selection?.path
            && named.actions?.install?.available === true;
        return Boolean(stamp) && !stamp.superseded && namedAbsent
            && Boolean(currentMissing(viewOf(harness))?.installable);
    }

    /** `install`: the press says Install. `latest` installs a program the engine finds absent and updates one it finds,
     *  and no target says "install only", so that press never becomes an update (a program found installed sends
     *  nothing) and is sent only after `installableNow`; between that read and the engine's own start another installer
     *  can still win the race. An Update, a Return or an unresolved press of this program is sent or rejoined as it was. */
    function start(harness, target, install = false) {
        if (disposed) return Promise.resolve(null);
        const state = record(harness);
        if (state.busy) return state.busy;
        const joined = rejoin(harness);
        if (joined) return joined;
        const entry = state.entry;
        if (!(target.kind === 'latest' ? offeredAction(entry) : target.kind === 'version' ? entry?.actions?.exactVersion : null)?.available) return Promise.resolve(null);
        if (target.kind === 'latest' && (install || !entry.selection?.path)) {
            if (entry.selection?.path) return Promise.resolve(null);
            const busy = installableNow(harness).then((found) => {
                if (state.busy === busy) state.busy = null;
                // The row is free again: say so to the mounted views, this one released or not.
                render();
                // A view released while its read was in flight owns nothing: the shared request stays as it is.
                if (disposed) return null;
                if (found) return rejoin(harness) || send(harness, target);  // a press another surface began meanwhile is rejoined
                return null;
            });
            state.busy = busy;
            return busy;
        }
        return send(harness, target);
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
        return action === 'install' ? start(harness, { kind: 'latest' }, true) : start(harness, { kind: action });
    }

    return {
        refresh, render, start, cancel, act, readOperation, markPressed, pressed, inventoryError,
        get harnesses() { return harnesses(); },
        get inspectionProblem() { return inspectionProblem(); },
        view: viewOf,
        poll() {
            if (disposed || !visible()) return;
            for (const [harness, state] of intents) {
                if (maintenanceOperationPending(state.operation) && !state.operationError && !state.busy) void readOperation(harness);
            }
        },
        dispose() { disposed = true; listeners.delete(onChange); },
    };
}
