// Task-local subscription recovery. The host owns waits, revisions and the
// application of an action; this view only preserves and presents those facts.
import { fetchJson, jsonPost } from './api_client.js';
import { claudexorStatus } from './claudexor_status_store.js';
import { startLogin } from './harness_accounts.js';
import { createModelRolesEditor, MODEL_ACCOUNTS_KEY, modelRolesHost, parseModelSource } from './model_roles.js';
import { API_PROVIDER_CREDENTIAL_KEYS } from './route_editor_primitives.js';
import { desiredLiveCardPhase, setLiveCardPhase } from './task_phase_chip.js';
import { taskDoneIsTerminal } from './log_events.js';

/** Only the provider credential fields; no other setting reaches the picker draft. */
function providerCredentials(settings = {}) {
    return Object.fromEntries(API_PROVIDER_CREDENTIAL_KEYS
        .filter((key) => key in settings).map((key) => [key, settings[key]]));
}

export function modelWaitRoleLabel(role = '') {
    const labels = { main: 'Main', light: 'Light', vision: 'Vision', consciousness: 'Background consciousness',
        deep_review: 'Deep self-review', websearch: 'Web search' };
    if (labels[role]) return labels[role];
    const [kind, ...identity] = String(role).split(':');
    const suffix = identity.join(':');
    if (kind === 'fallback' && /^\d+$/.test(suffix)) return `Fallback ${Number(suffix) + 1}`;
    if (kind === 'reviewer') return `Reviewer · ${suffix}`;
    if (kind === 'subagent') return `Subagent · ${suffix}`;
    return String(role || 'Model');
}

/** A late snapshot cannot rewind a row or reopen a resolved wait episode. */
export function mergeModelWaits(previous = {}, incoming = {}) {
    const result = { ...previous };
    for (const [id, row] of Object.entries(incoming || {})) {
        if (!row || row.wait_id !== id || !Number.isInteger(row.revision) || row.revision < 1
            || !Number.isInteger(row.task_attempt) || row.task_attempt < 1
            || !['waiting', 'resolved'].includes(row.state) || !['quota', 'auth', 'auth_quota'].includes(row.reason)) continue;
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
    if (finished) return [];
    const rows = Object.values(waits);
    const attempt = currentAttempt || Math.max(0, ...rows.map((row) => row.task_attempt));
    return rows.filter((row) => row.state === 'waiting' && row.task_attempt === attempt);
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

export function isModelWaitReference(value) {
    return value?.type === 'task_model_wait' || value?.system_type === 'task_model_wait';
}

/**
 * One controller per chat; one keyed view per wait inside the existing card.
 * getRecord may materialize the real task card, never an invented task identity.
 * No polling, timer, login job or task lifecycle is owned by this component.
 */
export function createModelWaitController({ getRecord, onDomWrite = (fn) => fn(),
    onChange = () => {}, openSettings = () => {}, login = startLogin,
    read = fetchJson, send = (body) => jsonPost('/api/decisions', body, { rejectOkFalse: true }),
    store = claudexorStatus, doc = () => document } = {}) {
    const tasks = new Map();
    let backgroundOwner = null;
    const getDoc = typeof doc === 'function' ? doc : () => doc;
    let destroyed = false;
    let serial = 0;
    let settingsPromise = null;
    const readAbort = new AbortController();

    function syncPhase(taskId, waiting) {
        const record = getRecord(taskId, false);
        if (record && !record.finished) {
            record.modelWaiting = waiting;
            record.root.dataset.modelWaiting = waiting ? '1' : '0';
            if (waiting && !record.suggestedName && !record.lastHumanHeadline && !record.direct) record.titleEl.textContent = taskId === 'bg-consciousness' ? 'Background thinking' : 'Task';
            const phase = desiredLiveCardPhase(record);
            setLiveCardPhase(record, record.backgroundPaused ? 'model_wait' : phase.phase,
                record.backgroundPaused ? 'Paused for foreground task' : phase.text, phase.className);
        }
        onChange(taskId, waiting);
    }

    function taskEntry(taskId) {
        if (!tasks.has(taskId)) tasks.set(taskId, { waits: {}, views: new Map(), finished: false, host: null, attempt: 0 });
        return tasks.get(taskId);
    }

    function syncBackground(snapshot) {
        if (!snapshot || typeof snapshot.model_wait_owner_id !== 'string') return false;
        const id = 'bg-consciousness', owner = snapshot.model_wait_owner_id;
        if (backgroundOwner !== owner) {
            const previous = tasks.get(id);
            if (previous) clearViews(previous);
            tasks.delete(id);
            backgroundOwner = owner;
        }
        const record = getRecord(id, false);
        if (record) {
            record.backgroundPaused = Boolean(owner && snapshot.paused);
            if (owner) record.finished = false;
        }
        if (!owner) { syncPhase(id, false); return true; }
        const task = taskEntry(id);
        task.finished = false;
        task.paused = Boolean(snapshot.paused);
        const changed = adopt(id, snapshot.model_waits || {});
        syncPhase(id, activeModelWaits(task.waits).length > 0);
        return changed;
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
        const quota = row.reason === 'quota' || mixed;
        node.querySelector('[data-wait-role]').textContent = modelWaitRoleLabel(row.role);
        node.querySelector('[data-wait-reason]').textContent = mixed ? 'Waiting for access'
            : row.reason === 'auth' ? 'Sign-in required' : 'Waiting for quota';
        const model = parseModelSource(row.model).model || row.model;
        node.querySelector('[data-wait-model]').textContent = `${model || 'Model'} · ${row.credential_profile_id ? `Account: ${row.credential_profile_id}` : 'Auto rotation'}`;
        const reset = row.reset_at ? new Date(row.reset_at) : null;
        node.querySelector('[data-wait-reset]').textContent = (mixed
            ? 'Some accounts need sign-in; others are waiting for quota. ' : '') + (quota
            ? (reset && Number.isFinite(reset.getTime()) ? `Quota resets ${reset.toLocaleString()}.` : 'Quota reset time is not known.') : '');
        const automatic = node.querySelector('[data-wait-auto]');
        automatic.parentElement.hidden = !quota;
        node.querySelector('[data-wait-auto-label]').textContent = mixed
            ? 'Continue automatically when access is restored' : 'Continue automatically when quota resets';
        const requested = view.sending ? view.lastRequest : row.pending_action;
        automatic.checked = pending && requested?.action === 'auto_continue'
            ? requested.auto_continue : row.auto_continue !== false;
        const auth = node.querySelector('[data-wait-login]');
        auth.hidden = row.reason !== 'auth';
        node.querySelector('[data-wait-settings]').textContent = mixed ? 'Accounts' : 'Settings';
        node.querySelector('[data-wait-notice]').textContent = view.error
            || (view.sending ? 'Sending your choice…' : isPending(row)
                ? `${view.saved === true ? 'Settings saved. ' : view.saved === null ? 'Settings save is unconfirmed. ' : ''}Request accepted; waiting for the task to apply it.` : '');
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
            : `This role changes until ${taskId === 'bg-consciousness' ? 'this wakeup cycle' : 'the task'} ends.`;
    }

    function paint(taskId) {
        if (destroyed) return false;
        const task = tasks.get(taskId);
        if (!task) return false;
        const active = activeModelWaits(task.waits, task.finished, task.attempt);
        if (!active.length) { clearViews(task); syncPhase(taskId, false); return true; }
        const record = getRecord(taskId);
        if (!record?.root || record.finished) { clearViews(task); return false; }
        if (taskId === 'bg-consciousness') record.backgroundPaused = Boolean(task.paused);
        if (task.host?.parentElement !== record.root) {
            if (!task.host) {
                task.host = getDoc().createElement('section');
                task.host.className = 'model-waits';
                task.host.setAttribute('aria-label', 'Subscription access');
                task.host.innerHTML = '<div data-wait-rows></div><div class="model-wait-footnote" data-wait-slot></div>';
            }
            record.timelineEl.before(task.host);
            const focused = task.focused;
            task.focused = null;
            if (focused) requestAnimationFrame(() => {
                if (!destroyed && focused.isConnected && getDoc().activeElement === getDoc().body) focused.focus({ preventScroll: true });
            });
        }
        const activeIds = new Set(active.map((row) => row.wait_id));
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
            : taskId === 'bg-consciousness' ? 'No worker slot is held. Foreground work may pause continuation. Changes last for this wakeup cycle.'
                : 'Completed steps are kept. Continuation is available while Ouroboros remains running.';
        syncPhase(taskId, true);
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
        if (task.finished) return false;
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
        view.sending = true;
        view.error = '';
        onDomWrite(() => paint(taskId));
        try {
            const answer = await send(body);
            if (destroyed) return;
            if (answer?.ok !== true || !answer.wait) throw new Error('The request was not confirmed. Check the task state before retrying.');
            view.saved = answer.saved;
            adopt(taskId, { [answer.wait.wait_id]: answer.wait });
        } catch (error) {
            if (destroyed) return;
            const latest = error?.body?.wait || error?.payload?.wait;
            if (latest) {
                // A failed mailbox delivery retains the accepted action. Only
                // its exact request can retry; a conflicting/stale one is gone.
                const failure = error?.body || error?.payload || {};
                if (!['mailbox_write_failed', 'model_wait_decision_failed'].includes(failure.reason_code)
                    || latest.pending_action?.request_id !== body.request_id) view.lastRequest = null;
                view.saved = failure.saved;
                adopt(taskId, { [latest.wait_id]: latest });
            }
            view.error = String(error?.message || error);
        } finally {
            view.sending = false;
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
            error: '', saved: false, lastRequest: null };
        node.innerHTML = `<div class="model-wait-heading"><strong data-wait-role></strong><span class="ui-status" data-tone="warn" data-wait-reason></span></div>
            <div class="model-wait-meta" data-wait-model></div><div class="model-wait-meta" data-wait-reset></div>
            <label class="model-wait-auto"><input class="ui-checkbox" type="checkbox" data-wait-auto> <span data-wait-auto-label></span></label>
            <div class="model-wait-actions"><button class="btn btn-default" type="button" data-wait-change>Change model or account</button>
                <button class="btn btn-default" type="button" data-wait-login>Sign in again</button>
                <button class="btn btn-default" type="button" data-wait-retry>Try again</button>
                <button class="btn btn-default" type="button" data-wait-settings>Settings</button></div>
            <div class="model-wait-picker" data-wait-picker hidden>${modelRolesHost(editorId)}
                <label class="model-wait-auto"><input class="ui-checkbox" type="checkbox" data-wait-persist> Also save this role in Settings</label>
                <div class="model-wait-actions"><button class="btn btn-default" type="button" data-wait-apply disabled>Apply to this ${taskId === 'bg-consciousness' ? 'cycle' : 'task'}</button>
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
        syncBackground,
        observe(taskId, value) {
            const attempt = Number.isInteger(value?.task_attempt) ? value.task_attempt : 0;
            if (attempt && attempt < (tasks.get(taskId)?.attempt || 0)) return false;
            if (taskId === 'bg-consciousness') {
                if (value?.model_wait_live) return syncBackground(value);
                if (isModelWaitReference(value)) {
                    if (value.type !== 'task_model_wait') return false;
                    if (backgroundOwner === null) syncBackground({ ...value, model_waits: {} });
                    if (value.model_wait_owner_id !== backgroundOwner) return false;
                }
            }
            if (taskId && taskDoneIsTerminal(value)) {
                if (value?.model_waits || isModelWaitReference(value)) taskEntry(taskId);
                this.finish(taskId); return false;
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
