// Task-local subscription recovery. The host owns waits, revisions and the
// application of an action; this view only preserves and presents those facts.
import { fetchJson, jsonPost } from './api_client.js';
import { claudexorStatus } from './claudexor_status_store.js';
import { startLogin } from './harness_accounts.js';
import { createModelRolesEditor, MODEL_ACCOUNTS_KEY, modelRolesHost, parseModelSource } from './model_roles.js';
import { API_PROVIDER_CREDENTIAL_KEYS, apiProviderLabel } from './route_editor_primitives.js';
import { adoptConnectionLabels, connectionLabel, routeConnectionId } from './connection_labels.js';
import { desiredLiveCardPhase, setLiveCardPhase } from './task_phase_chip.js';
import { taskDoneIsTerminal } from './log_events.js';

/** Only the provider credential fields and the connection list; no other setting reaches the picker draft. */
function providerCredentials(settings = {}) {
    return { ...Object.fromEntries(API_PROVIDER_CREDENTIAL_KEYS
        .filter((key) => key in settings).map((key) => [key, settings[key]])),
    _meta: { connections: settings?._meta?.connections } };
}

export function modelWaitRoleLabel(role = '') {
    const labels = { main: 'Main', light: 'Light', vision: 'Vision', consciousness: 'Consciousness',
        safety: 'Safety Supervisor', deep_review: 'Deep self-review', websearch: 'Web search' };
    if (labels[role]) return labels[role];
    const [kind, ...identity] = String(role).split(':');
    const suffix = identity.join(':');
    if (kind === 'fallback' && /^\d+$/.test(suffix)) return `Fallback ${Number(suffix) + 1}`;
    if (kind === 'reviewer') return `Reviewer · ${suffix}`;
    if (kind === 'subagent') return `Subagent · ${suffix}`;
    return String(role || 'Model');
}

/** Host wait reasons: `unavailable` is an engine-dated pool refusal, neither quota nor sign-in;
 *  `connection_unavailable`: the API key the route names is turned off or removed in Accounts. */
export const MODEL_WAIT_REASONS = ['quota', 'auth', 'auth_quota', 'unavailable', 'connection_unavailable'];

/** A late snapshot cannot rewind a row or reopen a resolved wait episode. */
export function mergeModelWaits(previous = {}, incoming = {}) {
    const result = { ...previous };
    for (const [id, row] of Object.entries(incoming || {})) {
        if (!row || row.wait_id !== id || !Number.isInteger(row.revision) || row.revision < 1
            || !Number.isInteger(row.task_attempt) || row.task_attempt < 1
            || !['waiting', 'resolved'].includes(row.state) || !MODEL_WAIT_REASONS.includes(row.reason)) continue;
        const prior = result[id];
        if (prior && (prior.revision > row.revision || prior.state === 'resolved')) continue;
        if (prior?.revision === row.revision) {
            // Acceptance adds a pending action without pretending the worker
            // applied it. An equal-revision stale snapshot cannot erase it.
            if (!prior.pending_action && row.pending_action?.request_id) result[id] = { ...prior,
                pending_action: row.pending_action, saved_request_id: row.saved_request_id };
            else if (row.saved_request_id && prior.pending_action?.request_id === row.saved_request_id) result[id] = { ...prior,
                saved_request_id: row.saved_request_id };
            continue;
        }
        result[id] = { ...row };
    }
    return result;
}

export function activeModelWaits(waits = {}, finished = false, currentAttempt = 0) {
    const rows = Object.values(waits);
    const attempt = currentAttempt || Math.max(0, ...rows.map((row) => row.task_attempt));
    return rows.filter((row) => row.state === 'waiting' && row.task_attempt === attempt
        && (!finished || (row.review_operation?.owner_id
            && row.review_operation.owner_id === row.model_wait_owner_id
            && row.review_operation.retry_key && row.review_operation.slot_id)));
}

export function modelWaitAction(taskId, row, action, fields, requestId) {
    const body = { request_id: requestId, decision_id: `model_wait:${taskId}:${row.wait_id}`,
        revision: row.revision, action };
    if (action === 'auto_continue') {
        if (typeof fields.auto_continue !== 'boolean') throw new Error('Choose whether to continue automatically.');
        body.auto_continue = fields.auto_continue;
    } else if (action === 'switch') {
        if (!String(fields.model || '').trim()) throw new Error('Choose a replacement model.');
        Object.assign(body, { model: String(fields.model).trim(),
            credential_profile_id: String(fields.credential_profile_id || ''),
            use_local: fields.use_local === true, persist_role: fields.persist_role === true });
    } else if (action !== 'retry') throw new Error('Unknown subscription recovery action.');
    return body;
}

const requestIdentity = () => globalThis.crypto?.randomUUID?.()
    || `model-wait-${Date.now()}-${Math.random().toString(16).slice(2)}`;

/** A requested Settings save that did not land, told once its wait resolved; '' otherwise. */
export function unsavedPersistenceNotice(row, outcome) {
    if (!outcome || outcome.saved === true || row?.state !== 'resolved') return '';
    const applied = Boolean(outcome.request_id) && row.applied_request_id === outcome.request_id;
    return `${modelWaitRoleLabel(row.role)}: ${applied ? 'model change applied' : 'model wait ended'}; `
        + (outcome.saved === null ? 'whether it was saved in Settings is unconfirmed.' : 'it was not saved in Settings.');
}

export function isModelWaitReference(value) {
    return value?.type === 'task_model_wait' || value?.system_type === 'task_model_wait';
}

/**
 * One controller per chat; one keyed view per wait inside the existing card.
 * getRecord may materialize the real task card, never an invented task identity.
 * No polling, timer, login job or task lifecycle is owned by this component.
 */
export function createModelWaitController({ getRecord, onDomWrite = (fn) => fn(),
    onChange = () => {}, openSettings = () => {}, login = startLogin, notify = () => {},
    includeWait = () => true,
    read = fetchJson, send = (body) => jsonPost('/api/decisions', body, { rejectOkFalse: true }),
    store = claudexorStatus, doc = () => document } = {}) {
    const tasks = new Map();
    const getDoc = typeof doc === 'function' ? doc : () => doc;
    let destroyed = false;
    let serial = 0;
    let settingsPromise = null;
    const readAbort = new AbortController();

    // `waiting`: an admitted wait of the task's current attempt is open; `ownerAction`: one of them needs the owner
    // (the host's own revision-bound `requires_owner_action`, never read from its words).
    function syncPhase(taskId, { waiting, ownerAction = false }) {
        const record = getRecord(taskId, false);
        if (record && !record.finished) {
            record.modelWaiting = waiting;
            record.modelOwnerAction = waiting && ownerAction;
            record.root.dataset.modelWaiting = waiting ? '1' : '0';
            // A block without work carries no title placeholder while it waits: the chrome
            // follows the work it stands on, never the lane (and no lane is always shown).
            const phase = desiredLiveCardPhase(record);
            setLiveCardPhase(record, phase.phase, phase.text, phase.className, phase.secondary);
        }
        onChange(taskId, waiting);
    }

    function taskEntry(taskId) {
        if (!tasks.has(taskId)) tasks.set(taskId, { waits: {}, views: new Map(), finished: false, host: null, attempt: 0 });
        return tasks.get(taskId);
    }

    function clearViews(task) {
        for (const view of task.views.values()) { view.editor?.destroy(); view.node.remove(); }
        task.views.clear();
        task.host?.remove();
        task.host = null;
    }

    function isPending(row) {
        return Boolean(row.pending_action?.request_id && row.pending_action.request_id !== row.applied_request_id);
    }

    function updateView(taskId, view, row) {
        view.row = row;
        const node = view.node;
        const pending = view.sending || isPending(row);
        const mixed = row.reason === 'auth_quota';
        const pool = row.reason === 'unavailable';
        const keyOff = row.reason === 'connection_unavailable';
        const quota = row.reason === 'quota' || mixed;
        const observesAccess = row.availability_observation !== 'unavailable';
        const timed = (quota || pool) && observesAccess;
        node.querySelector('[data-wait-role]').textContent = modelWaitRoleLabel(row.role);
        node.querySelector('[data-wait-reason]').textContent = keyOff ? 'Key unavailable' : !observesAccess ? 'Waiting for provider access'
            : mixed ? 'Waiting for access' : pool ? 'Waiting for an account' : row.reason === 'auth' ? 'Sign-in required' : 'Waiting for quota';
        // Amber only while the wait needs the owner (the host's revision-bound fact); a wait that goes on by itself is a
        // quiet tray with a neutral sentence (docs/DESIGN.md "Status marks").
        const owner = row.requires_owner_action === true;
        if (node.dataset.owner !== (owner ? '1' : '0')) node.dataset.owner = owner ? '1' : '0';
        node.querySelector('[data-wait-reason]').dataset.tone = owner ? 'warn' : 'neutral';
        const model = parseModelSource(row.model).model || row.model;
        const key = routeConnectionId(row.model);
        node.querySelector('[data-wait-model]').textContent = `${model || 'Model'} · ${keyOff
            ? (key ? connectionLabel(key) : apiProviderLabel(parseModelSource(row.model).source))
            : !observesAccess ? 'Configured API access' : row.credential_profile_id ? `Account: ${row.credential_profile_id}` : 'Auto rotation'}`;
        const reset = row.reset_at ? new Date(row.reset_at) : null;
        const known = reset && Number.isFinite(reset.getTime());
        node.querySelector('[data-wait-reset]').textContent = keyOff
            ? 'This API key is turned off or removed in Accounts. Turn it back on, choose another model, or stop.' : !observesAccess
            ? 'Automatic access checks are unavailable. Retry after access is restored, choose another model, or stop.' : (mixed
            ? 'Some accounts need sign-in; others are waiting for quota. ' : '') + (pool
            ? `No account can serve this model now. ${known ? `The engine reported a reset at ${reset.toLocaleString()}; availability is unconfirmed.` : 'The reset time is not known.'}`
            : quota ? (known ? `Quota resets ${reset.toLocaleString()}.` : 'Quota reset time is not known.') : '');
        const automatic = node.querySelector('[data-wait-auto]');
        automatic.parentElement.hidden = !timed;
        node.querySelector('[data-wait-auto-label]').textContent = mixed || pool
            ? 'Continue automatically when access is restored' : 'Continue automatically when quota resets';
        const requested = view.sending ? view.lastRequest : row.pending_action;
        automatic.checked = pending && requested?.action === 'auto_continue'
            ? requested.auto_continue : row.auto_continue !== false;
        const auth = node.querySelector('[data-wait-login]');
        auth.hidden = row.reason !== 'auth' || !observesAccess;
        node.querySelector('[data-wait-settings]').textContent = mixed || keyOff ? 'Accounts' : 'Settings';
        const saved = view.persisted?.saved;
        node.querySelector('[data-wait-notice]').textContent = view.error
            || (view.sending ? 'Sending your choice…' : isPending(row)
                ? `${saved === true ? 'Settings saved. ' : saved === null ? 'Settings save is unconfirmed. '
                    : saved === false ? 'Settings were not saved. ' : ''}Request accepted; waiting for the task to apply it.` : '');
        node.querySelector('[data-wait-notice]').dataset.tone = view.error ? 'warn' : 'neutral';
        node.querySelector('[data-wait-repeat]').hidden = !view.error || !view.lastRequest;
        node.querySelectorAll('button, input, select').forEach((el) => { el.disabled = pending; });
        node.querySelector('[data-wait-repeat]').disabled = view.sending;
        // Settings remain reachable even when the task is applying a control.
        node.querySelector('[data-wait-settings]').disabled = false;
        node.querySelector('[data-wait-apply]').disabled = pending || !view.editor;
        const taskOnlyLocal = view.editor && row.role.startsWith('fallback:')
            && view.editor.collect().USE_LOCAL_MAIN !== view.sharedFallbackLocal;
        const persist = node.querySelector('[data-wait-persist]');
        persist.disabled = pending || taskOnlyLocal;
        if (taskOnlyLocal) persist.checked = false;
        node.querySelector('[data-wait-scope]').textContent = taskOnlyLocal
            ? 'Local applies to all fallbacks in Settings. This change is task-only; edit Models for a permanent change.'
            : row.review_operation ? 'This role changes for this review operation.' : 'This role changes until the task ends.';
    }

    // A row leaving because it resolved still owes its requested Settings outcome; disposal does not.
    function tellUnsaved(view, row) {
        const notice = unsavedPersistenceNotice(row, view.persisted);
        if (notice) { view.persisted = null; notify(notice, 'warn'); }
    }

    function paint(taskId) {
        if (destroyed) return false;
        const task = tasks.get(taskId);
        if (!task) return false;
        const record = getRecord(taskId, false);
        const active = activeModelWaits(task.waits, task.finished || record?.finished, task.attempt)
            .filter(row => includeWait(taskId, row));
        const activeIds = new Set(active.map((row) => row.wait_id));
        for (const [id, view] of task.views) if (!activeIds.has(id)) tellUnsaved(view, task.waits[id]);
        if (!active.length) { clearViews(task); syncPhase(taskId, { waiting: false }); return true; }
        const card = record || getRecord(taskId);
        if (!card?.root) { clearViews(task); return false; }
        if (task.host?.parentElement !== card.root) {
            if (!task.host) {
                task.host = getDoc().createElement('section');
                task.host.className = 'model-waits';
                task.host.setAttribute('aria-label', 'Subscription access');
                task.host.innerHTML = '<div data-wait-rows></div><div class="model-wait-footnote" data-wait-slot></div>';
            }
            card.timelineEl.before(task.host);
            const focused = task.focused;
            task.focused = null;
            if (focused) requestAnimationFrame(() => {
                if (!destroyed && focused.isConnected && getDoc().activeElement === getDoc().body) focused.focus({ preventScroll: true });
            });
        }
        for (const [id, view] of task.views) {
            if (activeIds.has(id)) continue;
            view.editor?.destroy(); view.node.remove(); task.views.delete(id);
        }
        for (const row of active) {
            let view = task.views.get(row.wait_id);
            if (!view) {
                view = createView(taskId, row);
                task.views.set(row.wait_id, view);
                task.host.querySelector('[data-wait-rows]').append(view.node);
            }
            updateView(taskId, view, row);
        }
        task.host.querySelector('[data-wait-slot]').textContent = active.some((row) => row.worker_slot_held === true)
            ? 'This task keeps its worker slot. Queued tasks may wait. Completed steps are kept.'
            : 'Completed steps are kept. Continuation is available while Ouroboros remains running.';
        syncPhase(taskId, { waiting: true, ownerAction: active.some((row) => row.requires_owner_action === true) });
        return true;
    }

    function adopt(taskId, rows, currentAttempt = 0) {
        if (destroyed || !taskId || !rows || typeof rows !== 'object') return false;
        const task = taskEntry(taskId);
        const next = mergeModelWaits(task.waits, rows);
        const nextAttempt = Math.max(task.attempt, currentAttempt,
            ...Object.values(next).map((row) => row.task_attempt));
        const attemptChanged = nextAttempt > task.attempt;
        if (attemptChanged) { task.attempt = nextAttempt; task.finished = false; }
        if (!attemptChanged && task.host?.isConnected && Object.keys(next).every((id) => next[id] === task.waits[id])) return false;
        task.waits = next;
        return onDomWrite(() => paint(taskId));
    }

    async function submit(taskId, view, action, fields = {}, retry = false) {
        if (destroyed || view.sending || view.row.state !== 'waiting'
            || (isPending(view.row) && (!retry || view.lastRequest?.request_id !== view.row.pending_action.request_id))) return;
        let body;
        try { body = retry ? view.lastRequest : modelWaitAction(taskId, view.row, action, fields, requestIdentity()); }
        catch (error) { view.error = error.message; onDomWrite(() => paint(taskId)); return; }
        view.lastRequest = body;
        if (!retry) view.persisted = null;
        const persisted = (saved) => {
            if (body.persist_role === true) view.persisted = { request_id: body.request_id, saved: typeof saved === 'boolean' ? saved : null };
        };
        view.sending = true;
        view.error = '';
        onDomWrite(() => paint(taskId));
        try {
            const answer = await send(body);
            if (destroyed) return;
            if (answer?.ok !== true || !answer.wait) throw new Error('The request was not confirmed. Check the task state before retrying.');
            persisted(answer.saved);
            adopt(taskId, { [answer.wait.wait_id]: answer.wait });
        } catch (error) {
            if (destroyed) return;
            const latest = error?.body?.wait || error?.payload?.wait;
            const failure = error?.body || error?.payload || {};
            persisted(failure.saved);
            if (latest) {
                // A failed mailbox delivery retains the accepted action. Only
                // its exact request can retry; a conflicting/stale one is gone.
                if (!['mailbox_write_failed', 'model_wait_decision_failed'].includes(failure.reason_code)
                    || latest.pending_action?.request_id !== body.request_id) view.lastRequest = null;
                adopt(taskId, { [latest.wait_id]: latest });
            }
            view.error = String(error?.message || error);
        } finally {
            view.sending = false;
            const task = destroyed ? null : tasks.get(taskId);
            // Its row may have resolved, and left the card, before this answer arrived.
            if (task && task.views.get(view.row.wait_id) !== view) tellUnsaved(view, task.waits[view.row.wait_id]);
            if (!destroyed) onDomWrite(() => paint(taskId));
        }
    }

    async function openPicker(taskId, view) {
        const panel = view.node.querySelector('[data-wait-picker]');
        panel.hidden = false;
        if (view.editor || view.loading) return;
        view.loading = true;
        try {
            settingsPromise ||= read('/api/settings', { cache: 'no-store', signal: readAbort.signal }).catch((error) => {
                settingsPromise = null; throw error;
            });
            const settings = await settingsPromise;
            adoptConnectionLabels(settings?._meta?.connections);
            if (destroyed || !view.node.isConnected || view.row.state !== 'waiting') return;
            const id = view.editorId;
            onDomWrite(() => {
                view.sharedFallbackLocal = settings.USE_LOCAL_FALLBACK === true || settings.USE_LOCAL_FALLBACK === 'true';
                view.editor = createModelRolesEditor({ hostId: id, store, doc: getDoc, showContext: false,
                    onChange: () => onDomWrite(() => paint(taskId)) });
                // The picker offers the same configured API providers as Models,
                // so the wait panel needs this document's credential fields too.
                view.editor.load({ ...providerCredentials(settings), model: view.row.model,
                    [MODEL_ACCOUNTS_KEY]: { main: view.row.credential_profile_id || '' } },
                { providerProfiles: settings?._meta?.setup_contract?.providerProfiles || {}, modelSlots: [
                    { slot: 'main', settingKey: 'model', inputId: `${id}-model`, settingsToggleId: `${id}-local`, label: modelWaitRoleLabel(view.row.role) },
                ] });
                view.editor.mount();
                panel.querySelector('[data-model-role-model]')?.focus();
                return true;
            });
            onDomWrite(() => paint(taskId));
            const catalog = await read('/api/model-catalog', { cache: 'no-store', signal: readAbort.signal });
            if (!destroyed && view.node.isConnected) onDomWrite(() => { view.editor?.adoptCatalog(catalog); return true; });
        } catch (error) {
            if (!destroyed) { view.error = String(error?.message || error); onDomWrite(() => paint(taskId)); }
        } finally { view.loading = false; }
    }

    function createView(taskId, row) {
        const node = getDoc().createElement('div');
        node.className = 'model-wait-row';
        node.dataset.waitId = row.wait_id;
        const editorId = `model-wait-editor-${++serial}-${requestIdentity()}`;
        const view = { node, row, editorId, editor: null, sending: false, loading: false,
            error: '', persisted: null, lastRequest: null };
        node.innerHTML = `<div class="model-wait-heading"><strong data-wait-role></strong><span class="ui-status" data-tone="neutral" data-wait-reason></span></div>
            <div class="model-wait-meta" data-wait-model></div><div class="model-wait-meta" data-wait-reset></div>
            <label class="model-wait-auto"><input class="ui-checkbox" type="checkbox" data-wait-auto> <span data-wait-auto-label></span></label>
            <div class="model-wait-actions"><button class="btn btn-default" type="button" data-wait-change>Change model or account</button>
                <button class="btn btn-default" type="button" data-wait-login>Sign in again</button>
                <button class="btn btn-default" type="button" data-wait-retry>Try again</button>
                <button class="btn btn-default" type="button" data-wait-settings>Settings</button></div>
            <div class="model-wait-picker" data-wait-picker hidden>${modelRolesHost(editorId)}
                <label class="model-wait-auto"><input class="ui-checkbox" type="checkbox" data-wait-persist> Also save this role in Settings</label>
                <div class="model-wait-actions"><button class="btn btn-default" type="button" data-wait-apply disabled>Apply to this task</button>
                    <span class="model-wait-meta" data-wait-scope></span></div></div>
            <div class="model-wait-notice ui-status" data-wait-notice role="status" aria-live="polite"></div>
            <button class="btn btn-default" type="button" data-wait-repeat hidden>Retry request</button>`;
        node.querySelector('[data-wait-auto]').addEventListener('change', (event) => {
            void submit(taskId, view, 'auto_continue', { auto_continue: event.target.checked });
        });
        node.querySelector('[data-wait-retry]').addEventListener('click', () => { void submit(taskId, view, 'retry'); });
        node.querySelector('[data-wait-repeat]').addEventListener('click', () => { void submit(taskId, view, '', {}, true); });
        node.querySelector('[data-wait-settings]').addEventListener('click', () => { void openSettings('providers'); });
        node.querySelector('[data-wait-login]').addEventListener('click', async () => {
            if (await openSettings('providers') === false) return;
            if (view.row.credential_harness && view.row.credential_profile_id) {
                await login(view.row.credential_harness, view.row.credential_profile_id);
            }
        });
        node.querySelector('[data-wait-change]').addEventListener('click', () => { onDomWrite(() => {
            void openPicker(taskId, view); return true;
        }); });
        node.querySelector('[data-wait-apply]').addEventListener('click', () => {
            const error = view.editor?.validate();
            if (error) { view.error = error; onDomWrite(() => paint(taskId)); return; }
            const choice = view.editor?.collect();
            if (choice) void submit(taskId, view, 'switch', { model: choice.model,
                credential_profile_id: choice[MODEL_ACCOUNTS_KEY]?.main || '',
                use_local: choice.USE_LOCAL_MAIN, persist_role: node.querySelector('[data-wait-persist]').checked });
        });
        return view;
    }

    return {
        adopt,
        refresh(taskId) { return onDomWrite(() => paint(taskId)); },
        observe(taskId, value) {
            const attempt = Number.isInteger(value?.task_attempt) ? value.task_attempt : 0;
            if (attempt && attempt < (tasks.get(taskId)?.attempt || 0)) return false;
            if (taskId && taskDoneIsTerminal(value)) {
                const changed = adopt(taskId, value?.model_waits || (isModelWaitReference(value) && value.wait_id
                    ? { [value.wait_id]: value } : {}), attempt);
                this.finish(taskId); return changed;
            }
            if (value?.model_waits) return adopt(taskId, value.model_waits, attempt);
            if (isModelWaitReference(value) && value.wait_id) return adopt(taskId, { [value.wait_id]: value }, attempt);
            if (attempt) return adopt(taskId, {}, attempt);
            return false;
        },
        waiting(taskId) { const task = tasks.get(taskId); return Boolean(task && activeModelWaits(task.waits, task.finished, task.attempt).length); },
        finish(taskId) { const task = tasks.get(taskId); if (task) { task.finished = true; onDomWrite(() => paint(taskId)); } },
        forget(taskId) { const task = tasks.get(taskId); if (task) clearViews(task); tasks.delete(taskId); },
        // History replaces task cards, not the owner's unsubmitted form or
        // pending action. Reattach the same keyed DOM/editor on the next paint.
        resetViews() { for (const task of tasks.values()) {
            const focused = getDoc().activeElement;
            task.focused = task.host?.contains(focused) ? focused : null;
            task.host?.remove();
        } },
        retainCards(records) {
            for (const [id, task] of tasks) {
                if (task.finished && !records.has(id)) { clearViews(task); tasks.delete(id); }
            }
        },
        destroy() { destroyed = true; readAbort.abort(); for (const task of tasks.values()) clearViews(task); tasks.clear(); },
    };
}
