// Settings → Connections: the owner's view of the remote SSH connection store.
// This module is deliberately thin. Trust decisions live server-side
// (ouroboros/gateway/connections.py + ouroboros/connection_store.py): only a
// successful Bootstrap pins a host identity, a changed identity is a typed
// refusal, and Retrust needs the exact old/new pair a live probe observed. The
// page never holds a private key, an SSH password, or the Network Password —
// unlocking goes through the ordinary login route, which returns an HttpOnly
// session cookie.
import { openConfirmDialog } from './confirm_dialog.js';
import {
    REMOTE_TRANSPORT_UNAVAILABLE_NOTE,
    isRemoteTransportUnavailable,
    remoteActionErrorText,
} from './remote_task_state.js';
import {
    collectSafeFieldValues,
    normalizeTone,
    renderSafeField,
} from './ui_helpers.js';
import { escapeHtmlAttr, escapeHtmlText } from './utils.js';

const CONNECTION_FIELDS = Object.freeze([
    { name: 'name', label: 'Name', type: 'text', required: true, placeholder: 'Production' },
    { name: 'ssh_alias', label: 'SSH alias', type: 'text', required: true, placeholder: 'production' },
]);
const AUTH_FIELDS = Object.freeze([
    {
        name: 'password',
        label: 'Network Password',
        type: 'password',
        required: true,
        help: 'Used once to create an HttpOnly owner session; it is not retained by this page.',
    },
]);
const FIELD_OPTIONS = Object.freeze({
    fieldClass: 'form-field',
    helpClass: 'settings-inline-note',
});
// The LIVE (process-local) fields of `gateway/connections._public_live_fields`, in ONE
// list, because TWO routes carry them — the WS `connection_state` merge and the REST
// reload's retain pass — and a key in one list but not the other is dropped on
// whichever route omits it. It WAS two lists and they had already drifted: the retain
// pass omitted `error_code` and `action`, so a REST reload kept a row's
// `blocker_action` while silently discarding the typed refusal's own "Next action"
// beside it. One list means that drift cannot recur.
const LIVE_FIELD_KEYS = Object.freeze([
    'status', 'phase', 'platform', 'architecture', 'build', 'completion',
    'bootstrap_compatible', 'health_fresh',
    // A server field absent from this list is dropped on the next merge, so an
    // outdated-executor badge would flicker off on the first status frame after the
    // reload that produced it.
    'execd_outdated', 'required_contract_set', 'bootstrap_contract_set',
    // The blocker travels with the evidence it was derived from, or the row would
    // keep an outdated verdict beside fresh facts.
    'blocked_by', 'blocker_action', 'blocker_hint', 'blocker_rank',
    'error_code', 'action', 'diagnostic', 'log_refs', 'warnings',
    // The totals travel with their lists or the disclosure goes stale.
    'log_refs_count', 'warnings_count',
]);
// Every one of these needs the ssh transport to be present in the build; Add and
// the list itself are pure owner-state operations and stay available without it.
const TRANSPORT_ACTIONS = Object.freeze(['test', 'bootstrap', 'reconnect', 'retrust', 'retire']);

export function isSelectableRemoteConnection(row) {
    if (!row || row.lifecycle !== 'active') return false;
    const status = String(row.status || 'unknown');
    return (
        status === 'ready'
        && row.bootstrap_compatible === true
        && row.health_fresh === true
        // An outdated execd is `ready` by every other measure and fails EVERY remote
        // tool call, so offering it in the project picker would route a task onto a
        // host that cannot run it. Bootstrap clears the flag; until then this
        // connection is not a place a new project can go.
        && row.execd_outdated !== true
    );
}

/**
 * The server's own answer to "why not, and what removes it" for one row.
 *
 * `remote_refusal_actions.connection_blocker` derives this from the same evidence
 * `isSelectableRemoteConnection` reads and ships it on the row, so a surface never
 * has to invent a remedy sentence. That is the whole fix: the New Project picker
 * used to compose "run Bootstrap (or Test to refresh health)" — every conceivable
 * cure at once — and an owner blocked by an outdated executor pressed Test, got a
 * success, and stayed blocked, because only Bootstrap writes the contract-set stamp.
 *
 * Returns null for a selectable row. A row from an older server that carries no
 * blocker fields yet returns null too: this reads the evidence, it does not re-derive
 * it, and inventing a hint here would put the second authority back.
 */
export function connectionBlocker(row) {
    if (!row || isSelectableRemoteConnection(row)) return null;
    const code = String(row.blocked_by || '');
    if (!code) return null;
    return {
        code,
        action: String(row.blocker_action || ''),
        hint: String(row.blocker_hint || ''),
        rank: Number.isInteger(row.blocker_rank) ? row.blocker_rank : -1,
    };
}

const SUMMARY_WARNING_LIMIT = 4;

/**
 * How many of a bounded list were left out, from the server's own TOTAL.
 *
 * The gateway sends `<name>_count` beside every capped list, so "four warnings" and
 * "four of nine warnings" are distinguishable — a cap that leaves no trace reads
 * exactly like data that was never there. This folds in the summary's own tighter
 * cap, so the number the owner sees is against the full total and not against
 * whatever survived the previous hop.
 */
function omittedCount(rows, total, shown) {
    const received = Array.isArray(rows) ? rows.length : 0;
    return Math.max(0, Math.max(Number(total) || 0, received) - Math.min(received, shown));
}

/** The visible trace a bounded list leaves behind, or nothing when nothing was cut. */
function omissionNote(omitted) {
    return omitted
        ? `<p class="settings-inline-note">${escapeHtmlText(
            `${omitted} more entr${omitted === 1 ? 'y' : 'ies'} omitted by the bounded live projection.`,
        )}</p>`
        : '';
}

export function connectionStatusCopy(row = {}) {
    const status = String(row.status || (row.lifecycle === 'retired' ? 'disconnected' : 'unknown'));
    const phase = String(row.phase || '');
    const error = String(row.error_code || '');
    const warnings = (Array.isArray(row.warnings) ? row.warnings : [])
        .filter((warning) => warning && typeof warning === 'object');
    const omitted = omittedCount(warnings, row.warnings_count, SUMMARY_WARNING_LIMIT);
    const blocker = connectionBlocker(row);
    return [
        status,
        phase && `phase: ${phase}`,
        error && `error: ${error}`,
        // After the error and before the warnings, because it is neither: a block is a
        // settled FACT about this host with exactly one action attached, and a row
        // showing only `ready` described a connection on which every remote tool call
        // fails. The pair comes from the server's ladder rather than from a hardcoded
        // `execd_outdated` arm here, so a row blocked for any other reason says so too
        // instead of looking merely un-green.
        blocker && `blocked: ${blocker.code} - ${blocker.action}`,
        ...warnings.slice(0, SUMMARY_WARNING_LIMIT)
            .map((warning) => `warning: ${String(warning.code || 'ssh_warning')}`),
        omitted && `+${omitted} more warning${omitted === 1 ? '' : 's'}`,
    ]
        .filter(Boolean)
        .join(' · ');
}

function observedHostId(payload) {
    for (const source of [
        payload,
        payload?.handshake,
        payload?.diagnostic,
    ]) {
        const value = String(source?.host_id || source?.observed_host_id || '').trim();
        if (value) return value;
    }
    return '';
}

function statusTone(row) {
    const status = String(row?.status || 'unknown');
    // `connecting` first: it IS a blocker, but it is the one that clears itself, and
    // painting in-flight work as a warning would make every Test look like a fault.
    if (status === 'connecting') return 'info';
    // Then any blocker, BEFORE the `ready` arm, because that is the whole point: an
    // outdated executor reports `ready` and cannot serve, so a green row would be the
    // surface agreeing with the wrong half of the evidence.
    if (connectionBlocker(row)) return 'warn';
    if (status === 'ready') return 'ok';
    if (status === 'degraded' || status === 'disconnected') return 'warn';
    return 'muted';
}

function connectionDetails(row) {
    const blocker = connectionBlocker(row);
    const facts = [
        row.expected_host_id && ['Pinned host identity', row.expected_host_id],
        row.platform && ['Platform', row.platform],
        row.architecture && ['Architecture', row.architecture],
        row.build && ['Executor build', row.build],
        ['Bootstrap compatible', row.bootstrap_compatible === true ? 'yes' : 'no'],
        ['Health fresh', row.health_fresh === true ? 'yes' : 'no'],
        row.execd_outdated === true && [
            'Executor contract set',
            `${row.bootstrap_contract_set ?? 0} installed, ${row.required_contract_set ?? 0} required`,
        ],
        row.completion && ['Completion', row.completion],
        row.action && ['Next action', row.action],
        // The blocker's OWN sentence, verbatim from the server's ladder. `Next action`
        // above is the last live answer's action, which is a different fact: an action
        // can succeed (`Test` → ok) while a block remains, and showing only the former
        // is exactly how the owner was told the problem was solved.
        blocker && ['Still blocked by', `${blocker.code} → ${blocker.action}`],
        blocker?.hint && ['What removes it', blocker.hint],
    ].filter(Boolean);
    const diagnostic = row.diagnostic && typeof row.diagnostic === 'object'
        ? JSON.stringify(row.diagnostic, null, 2)
        : '';
    const logRefs = Array.isArray(row.log_refs) && row.log_refs.length
        ? JSON.stringify(row.log_refs, null, 2)
        : '';
    // The details panel shows every warning it RECEIVED — the row summary is the
    // place that abbreviates — and names the total when the gateway itself capped.
    const warnings = Array.isArray(row.warnings) && row.warnings.length
        ? JSON.stringify(row.warnings, null, 2)
        : '';
    const warningsOmitted = omittedCount(row.warnings, row.warnings_count, Infinity);
    const logRefsOmitted = omittedCount(row.log_refs, row.log_refs_count, Infinity);
    if (!facts.length && !diagnostic && !logRefs && !warnings) return '';
    return `
        <details class="connection-details">
            <summary>Details and logs</summary>
            ${facts.length ? `<dl>${facts.map(([label, value]) => `
                <div><dt>${escapeHtmlText(label)}</dt><dd><code>${escapeHtmlText(value)}</code></dd></div>
            `).join('')}</dl>` : ''}
            ${diagnostic ? `<h4>Diagnostic</h4><pre>${escapeHtmlText(diagnostic)}</pre>` : ''}
            ${logRefs ? `<h4>Log references</h4><pre>${escapeHtmlText(logRefs)}</pre>${omissionNote(logRefsOmitted)}` : ''}
            ${warnings ? `<h4>Warnings</h4><pre>${escapeHtmlText(warnings)}</pre>${omissionNote(warningsOmitted)}` : ''}
        </details>
    `;
}

/**
 * Whether one row action can run at all, and why not — the single rule behind
 * every disabled button in this card. Exported pure for the node tests
 * (`web/tests/remote_connections.test.js`), same as chat.js does for its
 * disclosure guard: a build without the ssh transport must show a settled,
 * explained refusal instead of a button that answers 503 forever.
 */
export function connectionActionState(action, { loading = false, transportAvailable = true } = {}) {
    const blocked = !transportAvailable && TRANSPORT_ACTIONS.includes(action);
    return {
        disabled: Boolean(loading || blocked),
        reason: blocked ? REMOTE_TRANSPORT_UNAVAILABLE_NOTE : '',
    };
}

/** The card-level honest note for a build that carries no ssh transport. */
export function transportNoteMarkup(transportAvailable) {
    if (transportAvailable) return '';
    return `
            <div class="connection-transport-note" role="note" data-conn-transport-missing>
                ${escapeHtmlText(REMOTE_TRANSPORT_UNAVAILABLE_NOTE)}
            </div>
        `;
}

function actionButton(action, label, variant, flags) {
    const { disabled, reason } = connectionActionState(action, flags);
    const disabledAttr = disabled ? ' disabled' : '';
    const hint = reason ? ` title="${escapeHtmlAttr(reason)}"` : '';
    return `<button type="button" class="btn ${variant} btn-sm" data-conn-action="${escapeHtmlAttr(action)}"${disabledAttr}${hint}>${escapeHtmlText(label)}</button>`;
}

/** Pure row renderer; exported so the node tests can assert the visible state. */
export function renderConnectionRow(row, flags = {}) {
    const retired = row.lifecycle === 'retired';
    return `
        <article class="connection-row" data-connection-id="${escapeHtmlAttr(row.id)}">
            <div class="connection-row-body">
                <div class="connection-row-main">
                    <strong>${escapeHtmlText(row.name || row.id)}</strong>
                    <code>${escapeHtmlText(row.ssh_alias)}</code>
                    <span class="connection-state" data-tone="${statusTone(row)}">${escapeHtmlText(connectionStatusCopy(row))}</span>
                </div>
                ${connectionDetails(row)}
            </div>
            <div class="connection-row-actions">
                ${retired ? '<span class="settings-inline-note">Retired</span>' : [
                    actionButton('test', 'Test', 'btn-default', flags),
                    actionButton('bootstrap', 'Bootstrap', 'btn-secondary', flags),
                    actionButton('reconnect', 'Reconnect', 'btn-default', flags),
                    actionButton('retrust', 'Retrust host…', 'btn-default', flags),
                    actionButton('retire', 'Retire', 'btn-danger', flags),
                ].join('')}
            </div>
        </article>
    `;
}

export function initConnectionsUI({ root, apiClient, ws } = {}) {
    const host = root?.querySelector?.('#settings-connections-root');
    if (!host) return;
    let connections = [];
    let loading = false;
    let accessState = 'unknown';
    // Optimistic: the transport is assumed present until the gateway says
    // otherwise with its typed refusal. Any later success re-enables the row
    // actions, so a transient stand-in never latches permanently.
    let transportAvailable = true;
    let requestRevision = 0;
    let liveRevision = 0;
    const liveOverrides = new Map();

    function authMarkup() {
        if (accessState === 'required') {
            return `
                <form class="connection-auth-form" data-conn-auth>
                    ${renderSafeField(AUTH_FIELDS[0], {}, FIELD_OPTIONS)}
                    <button type="submit" class="btn btn-primary"${loading ? ' disabled' : ''}>Unlock Connections</button>
                </form>
            `;
        }
        if (accessState === 'unconfigured') {
            return `
                <div class="settings-inline-note" role="note">
                    Configure a Network Password in Settings → Providers (or
                    <code>OUROBOROS_NETWORK_PASSWORD</code> in the server environment),
                    then restart Ouroboros. The value is never shown here.
                </div>
            `;
        }
        return '';
    }

    function managementMarkup() {
        if (accessState !== 'ready') return '';
        const flags = { loading, transportAvailable };
        return `
            <form class="connections-add-form" data-conn-add>
                ${CONNECTION_FIELDS.map((field) => renderSafeField(field, {}, {
                    ...FIELD_OPTIONS,
                    disabled: loading,
                })).join('')}
                <button type="submit" class="btn btn-primary"${loading ? ' disabled' : ''}>Add</button>
            </form>
            <div class="connections-list">
                ${connections.length
                    ? connections.map((row) => renderConnectionRow(row, flags)).join('')
                    : '<div class="settings-inline-note">No saved SSH connections.</div>'}
            </div>
        `;
    }

    function render(message = '', tone = 'muted') {
        host.innerHTML = `
            <section class="settings-card connections-card" aria-busy="${loading ? 'true' : 'false'}">
                <div class="settings-card-head">
                    <div>
                        <h3>SSH Connections</h3>
                        <div class="settings-section-copy">
                            Add a name and an SSH config alias such as <code>production</code>.
                            Test verifies transport and host identity; Bootstrap installs or
                            upgrades the compatible remote executor. Password, MFA and OpenSSH
                            host-trust prompts must be completed in a normal terminal.
                        </div>
                    </div>
                    <button type="button" class="btn btn-default btn-sm" data-conn-refresh${loading ? ' disabled' : ''}>Refresh</button>
                </div>
                ${authMarkup()}
                ${accessState === 'ready' ? transportNoteMarkup(transportAvailable) : ''}
                ${managementMarkup()}
                <div class="settings-inline-status" data-tone="${normalizeTone(tone)}" data-conn-status aria-live="polite">${escapeHtmlText(message)}</div>
            </section>
        `;
        host.querySelector('[data-conn-refresh]')?.addEventListener('click', () => load());
        host.querySelector('[data-conn-auth]')?.addEventListener('submit', authenticate);
        host.querySelector('[data-conn-add]')?.addEventListener('submit', async (event) => {
            event.preventDefault();
            if (loading) return;
            const values = collectSafeFieldValues(event.currentTarget, CONNECTION_FIELDS);
            await act(
                () => apiClient.connectionAdd({
                    name: String(values.name || '').trim(),
                    ssh_alias: String(values.ssh_alias || '').trim(),
                }),
                'Connection saved. Run Test, then Bootstrap before selecting it for a Project.',
                { phase: 'save' },
            );
        });
        host.querySelectorAll('[data-conn-action]').forEach((button) => {
            button.addEventListener('click', () => {
                if (loading) return;
                handleAction(
                    button.closest('[data-connection-id]')?.dataset.connectionId || '',
                    button.dataset.connAction,
                );
            });
        });
    }

    function mergePayload(payload, connectionId = '') {
        if (!payload || typeof payload !== 'object') return;
        const stored = payload.connection && typeof payload.connection === 'object'
            ? payload.connection
            : {};
        const id = String(stored.id || payload.connection_id || connectionId || '');
        if (!id) return;
        const publicLive = {};
        for (const key of LIVE_FIELD_KEYS) {
            if (Object.prototype.hasOwnProperty.call(payload, key)) publicLive[key] = payload[key];
        }
        const existing = connections.find((row) => row.id === id) || {};
        const merged = { ...existing, ...stored, ...publicLive, id };
        const previous = liveOverrides.get(id)?.fields || {};
        liveOverrides.set(id, {
            revision: ++liveRevision,
            fields: { ...previous, ...stored, ...publicLive },
        });
        const index = connections.findIndex((row) => row.id === id);
        if (index < 0) connections.push(merged);
        else connections[index] = merged;
    }

    async function authenticate(event) {
        event.preventDefault();
        if (loading) return;
        const values = collectSafeFieldValues(event.currentTarget, AUTH_FIELDS);
        const password = String(values.password || '');
        loading = true;
        render('Creating owner session…', 'info');
        try {
            await apiClient.ownerLogin(password);
            accessState = 'ready';
            await load('Owner session established.', 'ok');
        } catch (error) {
            loading = false;
            accessState = 'required';
            render(error?.body?.error || error?.message || String(error), 'warn');
        }
    }

    async function load(message = '', tone = 'muted') {
        const revision = ++requestRevision;
        const liveAtStart = liveRevision;
        loading = true;
        render(message || 'Loading connections…', tone);
        try {
            const data = await apiClient.connections();
            if (revision !== requestRevision) return;
            connections = (Array.isArray(data?.connections) ? data.connections : []).map((row) => {
                const override = liveOverrides.get(String(row.id || ''));
                const fields = override?.fields || {};
                const retainedDetails = {};
                for (const key of LIVE_FIELD_KEYS) {
                    if (!(key in row) && key in fields) retainedDetails[key] = fields[key];
                }
                return {
                    ...row,
                    ...retainedDetails,
                    ...(override?.revision > liveAtStart ? fields : {}),
                };
            });
            accessState = 'ready';
            loading = false;
            render(message, tone);
        } catch (error) {
            if (revision !== requestRevision) return;
            loading = false;
            const code = error?.body?.error_code || '';
            if (code === 'owner_auth_required') {
                connections = [];
                accessState = 'required';
                render('Enter the existing Network Password to unlock connection administration.', 'warn');
            } else if (code === 'owner_auth_not_configured') {
                connections = [];
                accessState = 'unconfigured';
                render('Owner authentication is not configured.', 'warn');
            } else {
                render(remoteActionErrorText(error), 'warn');
            }
        }
    }

    async function act(operation, success, { connectionId = '', phase = 'connect' } = {}) {
        loading = true;
        if (connectionId) {
            mergePayload({ connection_id: connectionId, status: 'connecting', phase });
        }
        render(`Working · phase: ${phase}`, 'info');
        try {
            const result = await operation();
            mergePayload(result, connectionId);
            transportAvailable = true;
            loading = false;
            // The action SUCCEEDED — and that is not the same as "the connection is
            // usable now". A Test on a host whose executor predates this build's
            // contract set answers `ok` with fresh health and leaves the connection
            // unselectable, and this line used to say "Transport test passed. Run
            // Bootstrap to make this connection selectable." on every Test, whether or
            // not Bootstrap was the thing needed. The server's answer now names what is
            // still in the way, so the report is the truth about the STATE rather than
            // about the request.
            const blocker = connectionBlocker(
                connections.find((item) => item.id === connectionId),
            );
            if (blocker?.hint) render(blocker.hint, 'warn');
            else render(success, 'ok');
        } catch (error) {
            // Order matters: `loading` is cleared BEFORE rendering on every
            // failure path, so a typed refusal settles into a readable state
            // instead of leaving the card marked busy forever.
            if (isRemoteTransportUnavailable(error)) transportAvailable = false;
            mergePayload(error?.body, connectionId);
            loading = false;
            if (error?.body?.error_code === 'owner_auth_required') {
                accessState = 'required';
                connections = [];
            }
            render(remoteActionErrorText(error), 'warn');
        }
    }

    async function retrust(connectionId, row) {
        loading = true;
        mergePayload({ connection_id: connectionId, status: 'connecting', phase: 'connect' });
        render('Testing the currently observed host identity…', 'info');
        let probe;
        try {
            probe = await apiClient.connectionTest(connectionId);
            transportAvailable = true;
        } catch (error) {
            if (isRemoteTransportUnavailable(error)) transportAvailable = false;
            probe = error?.body || {};
            if (!transportAvailable) {
                mergePayload(probe, connectionId);
                loading = false;
                render(REMOTE_TRANSPORT_UNAVAILABLE_NOTE, 'warn');
                return;
            }
        }
        mergePayload(probe, connectionId);
        loading = false;
        const oldHost = String(row.expected_host_id || '');
        const newHost = observedHostId(probe);
        if (!oldHost || !newHost) {
            render('Could not obtain both the pinned and currently observed host identities.', 'warn');
            return;
        }
        const confirmed = await openConfirmDialog({
            title: `Trust a new host identity for “${row.name || row.id}”?`,
            body: `Pinned: ${oldHost} · Now observed: ${newHost}. Only continue if this server was intentionally replaced or reinstalled — otherwise this is the signature of a different machine answering for the alias.`,
            confirmLabel: 'Trust new host',
            danger: true,
        });
        if (!confirmed) return;
        await act(() => apiClient.connectionRetrust(connectionId, {
            confirm: true,
            old_host_id: oldHost,
            new_host_id: newHost,
        }), 'New host identity trusted.', {
            connectionId,
            phase: 'retrust',
        });
    }

    async function handleAction(connectionId, action) {
        const row = connections.find((item) => item.id === connectionId);
        if (!row) return;
        if (action === 'test') {
            await act(
                () => apiClient.connectionTest(connectionId),
                // Said only when the answer carries NO remaining blocker; `act` renders
                // the server's blocker sentence instead when one does. Test's job is to
                // refresh the evidence, and what that evidence then implies is not
                // something this call site can know.
                'Transport test passed; this connection is ready to carry a Project.',
                { connectionId, phase: 'connect' },
            );
        } else if (action === 'bootstrap') {
            await act(
                () => apiClient.connectionBootstrap(connectionId),
                'Remote executor is ready.',
                { connectionId, phase: 'bootstrap' },
            );
        } else if (action === 'reconnect') {
            await act(
                () => apiClient.connectionReconnect(connectionId),
                'Remote Project sessions reconnected and reconciled.',
                { connectionId, phase: 'reconcile' },
            );
        } else if (action === 'retire') {
            const confirmed = await openConfirmDialog({
                title: `Retire “${row.name || row.id}”?`,
                // Name the path that actually performs the rebind. The promise used
                // to be open-ended ("until they are rebound") while nothing in the
                // UI or the CLI could rebind anything, which left a retired
                // connection's Projects permanently unusable.
                body: 'Existing Projects keep their binding, but no new work can start on them until they are rebound with “ouroboros projects rebind <project> --connection <id> --remote-root <path>”. Trust history is preserved.',
                confirmLabel: 'Retire',
                danger: true,
            });
            if (!confirmed) return;
            await act(
                () => apiClient.connectionRetire(connectionId),
                'Connection retired.',
                { connectionId, phase: 'retire' },
            );
        } else if (action === 'retrust') {
            await retrust(connectionId, row);
        }
    }

    if (ws && typeof ws.on === 'function') {
        ws.on('connection_state', (event) => {
            const connectionId = String(event?.connection_id || '');
            if (!connectionId) return;
            mergePayload(event, connectionId);
            if (accessState === 'ready') render();
        });
    }
    window.addEventListener('ouro:settings-subtab-shown', (event) => {
        if (event.detail?.tab === 'connections') load();
    });
    render();
    if (root.dataset?.activeSettingsTab === 'connections') load();
}
