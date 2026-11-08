import { apiFetch, jsonPost } from './api_client.js';
/** MCP settings cards; preserves masked credentials until the user edits them. */
import { absoluteTime, escapeHtmlAttr as escapeHtml, safeExternalUrl } from './utils.js';
import { formatRelativeAge, openExternalViaHostBridge, revealNewRow } from './ui_helpers.js';
import { resetSecretReveals } from './settings_secrets.js';
import {
    renderHeaderFields, bindHeaderFields, credentialPayload, headerKeepsSavedValues, keptHeaderNames,
    headerDraftRows, headerImportFacts, resetHeaderRows, serverIdentity, validateHeaderFields, canonicalServerId,
} from './mcp_header_fields.js';
import { endpointChange, openMcpImportDialog } from './mcp_import_dialog.js';

const TRANSPORTS = [
    { value: 'streamable_http', label: 'Streamable HTTP' },
    { value: 'sse', label: 'SSE (Server-Sent Events)' },
    { value: 'stdio', label: 'Local process (stdio)' },
];
// task_session/browser_policy: set by Import and kept as saved, with no control of their own. A held stdio card
// says what they do; off stdio they are unsupported fields that Save refuses until they are removed.
const LIFETIME_FLAGS = ['task_session', 'browser_policy'];
// A remote server's sign-in (`status.sign_in`, ouroboros/mcp_oauth.py): why it is needed, in words.
const SIGN_IN_REASONS = {
    not_signed_in: 'not signed in yet',
    expired: 'the sign-in expired',
    refresh_rejected: 'the provider no longer accepts the sign-in',
    token_rejected: 'the provider no longer accepts the sign-in',
    signed_out: 'signed out',
};
const SERVER_FIELDS = new Set(['id', 'slug', 'name', 'label', 'enabled', 'transport', 'url',
    'command', 'args', 'auth_header', 'auth_token', 'headers', 'allowed_tools', 'cwd', 'env', 'env_from_settings',
    'schemas_at_start', ...LIFETIME_FLAGS]);

let mcpServers = [];
let mcpStatusByServer = {};
let mcpStatusEnvelope = null;
let onChangeCallback = null;
// Original lookup identity and masked address only; never serialize these into
// a server draft or use its editable id / current list index to reveal a value.
const savedMcpSecrets = new WeakMap();
// Card results follow the server object, not its list position: a pending
// Refresh or Test lands on its own card or nowhere, and a draft edit retires a
// Test result that no longer describes the draft.
const cardMessages = new WeakMap();
const draftGenerations = new WeakMap();
const pendingActions = new WeakMap();
const signInLinks = new WeakMap(); // the provider page of the sign-in this window began, per server
const signInBusy = new WeakSet(); // a sign-in step in flight: a second press waits for it
// A server is a row; its form opens behind Edit. A server not saved yet starts open, and a rebuild (a transport
// change redraws the card) keeps whatever the owner opened or closed.
const editOpen = new WeakSet();

function looksMasked(value) {
    const text = String(value ?? '').trim();
    if (!text) return false;
    return text === '***' || text.endsWith('...');
}

function emptyServer() {
    return {
        id: '',
        name: '',
        enabled: false,
        transport: 'streamable_http',
        url: '',
        command: '',
        args: [],
        auth_header: 'Authorization',
        auth_token: '',
        allowed_tools: [],
    };
}

function notifyChanged() {
    if (typeof onChangeCallback === 'function') {
        try { onChangeCallback(); } catch (err) { /* swallow */ }
    }
}

function cardFor(server) {
    const index = mcpServers.indexOf(server);
    return index < 0 ? null : document.querySelector(`[data-mcp-index="${index}"]`);
}

function paintCardMessage(server) {
    const el = cardFor(server)?.querySelector('[data-mcp-message]');
    if (!el) return;
    const message = cardMessages.get(server);
    el.textContent = message?.text || '';
    if (message) el.dataset.tone = message.tone;
    el.hidden = !message;
}

function setCardMessage(server, text, tone = 'muted', kind = '') {
    if (text) cardMessages.set(server, { text, tone, kind });
    else cardMessages.delete(server);
    paintCardMessage(server);
}

/** One server's draft changed: a finished Test no longer describes it. */
function serverEdited(server) {
    draftGenerations.set(server, (draftGenerations.get(server) || 0) + 1);
    if (cardMessages.get(server)?.kind === 'test') setCardMessage(server, '');
    paintEndpointNotice(server);
    notifyChanged();
}

function setPending(server, action, busy) {
    const actions = pendingActions.get(server) || new Set();
    if (busy) actions.add(action); else actions.delete(action);
    pendingActions.set(server, actions);
    const button = cardFor(server)?.querySelector(`[data-mcp-${action}]`);
    if (button) button.disabled = busy;
}

/** A deliberate address change keeps saved values; say where they will go. */
function endpointNotice(server) {
    const saved = savedMcpSecrets.get(server);
    if (!saved || server.transport === 'stdio') return '';
    const change = endpointChange(saved.url, server.url);
    const names = keptHeaderNames(server);
    if (!change || !names.length) return '';
    const where = change.sameOrigin ? `changes on ${change.to}` : `changes from ${change.from} to ${change.to}`;
    return `Address ${where}. Saved values of ${names.join(', ')} stay with this server and go to the new address when you Test or Save.`;
}

function paintEndpointNotice(server) {
    const el = cardFor(server)?.querySelector('[data-mcp-endpoint-notice]');
    if (!el) return;
    el.textContent = endpointNotice(server);
    el.hidden = !el.textContent;
}

function toolCountLabel(count) {
    return `${count} tool${count === 1 ? '' : 's'}`;
}

function tokensLabel(count) {
    const n = Number(count || 0);
    return n < 1000 ? `≈${n} tokens` : `≈${(n / 1000).toFixed(1)}k tokens`;
}

function isPinned(server) {
    return [true, 'true', 'True'].includes(server.schemas_at_start);
}

/** A call-lifetime flag the loader applies (``normalize_server_config``'s reading); false or absent is off. */
function lifetimeFlagOn(value) {
    return value === true || (typeof value !== 'boolean'
        && ['1', 'true', 'yes', 'on'].includes(String(value ?? '').trim().toLowerCase()));
}

/** Flags that are on while the transport is not stdio: the loader refuses such a server, so Save does too. */
function remoteLifetimeFlags(server) {
    return server.transport === 'stdio' ? [] : LIFETIME_FLAGS.filter((key) => lifetimeFlagOn(server[key]));
}

/**
 * Why Save refuses this card, or ''. The loader (``normalize_server_config``) refuses a server whose known
 * fields its transport does not use (a URL left from another transport included), not one with unknown fields,
 * which it keeps with a warning; ``unsupportedFields`` already lists both, so only its known names refuse.
 */
function refusedFieldsMessage(server) {
    const refused = unsupportedFields(server).filter((key) => SERVER_FIELDS.has(key));
    if (!refused.length) return '';
    const flags = refused.filter((key) => LIFETIME_FLAGS.includes(key));
    const fields = refused.filter((key) => !LIFETIME_FLAGS.includes(key));
    const label = (TRANSPORTS.find((item) => item.value === server.transport) || TRANSPORTS[0]).label;
    return [
        fields.length ? `${fields.join(', ')} ${fields.length > 1 ? 'are' : 'is'} not used by ${label} servers, `
            + `so the server would not load with ${fields.length > 1 ? 'them' : 'it'}.` : '',
        flags.length ? `${flags.join(' and ')} ${flags.length > 1 ? 'apply' : 'applies'} only to Local process `
            + '(stdio) servers.' : '',
        'Choose a transport that uses these fields, or use Remove unsupported fields.',
    ].filter(Boolean).join(' ');
}

/** The read-only line of a held stdio server: what its Import-set flags do (no control of their own). */
function lifetimeNote(server) {
    if (server.transport !== 'stdio' || !lifetimeFlagOn(server.task_session)) return '';
    return 'task_session: a task’s calls share one running server process until its tool loop ends.'
        + (lifetimeFlagOn(server.browser_policy)
            ? ' browser_policy: that process may ask Ouroboros’s browser rules about its requests.' : '');
}

/** What this server costs the request: its part of the connections map, and its call schemas. */
function weightLabel(server) {
    const status = statusOf(server);
    if (!status || status.schema_tokens == null) return '';
    const schemas = isPinned(server)
        ? `call schemas ${tokensLabel(status.schema_tokens)} in every Low/Max request from task start`
        : `call schemas ${tokensLabel(status.schema_tokens)}, loaded when Ouroboros takes up this server`;
    return `Connections map ${tokensLabel(status.map_tokens)} in every Low/Max request; ${schemas}.`;
}

function unsupportedFields(server) {
    const unused = server.transport === 'stdio' ? ['url', 'auth_token', 'headers']
        : ['command', 'args', 'cwd', 'env', 'env_from_settings'];
    // Edited header rows live in the draft even when server.headers is absent.
    const draft = { ...server, ...credentialPayload(server) };
    const fields = Object.keys(draft).filter((key) => !SERVER_FIELDS.has(key)
        || (unused.includes(key) && !['', '[]', '{}', 'null'].includes(JSON.stringify(draft[key]))
            && draft[key] !== ''));
    if (server.transport === 'stdio' && draft.auth_header && draft.auth_header !== 'Authorization') {
        fields.push('auth_header');
    }
    return [...fields, ...remoteLifetimeFlags(server)];
}

const age = (iso) => formatRelativeAge(Date.parse(String(iso || '')), 'just now');

/** The saved-status parts of a card: its chip (one typed state, relative time), its Details facts (the exact
 *  stamps and the engine's text), and its discovered tools' list items. */
function statusParts(server) {
    const status = statusOf(server);
    const toolCount = status ? Number(status.tool_count || 0) : 0;
    const lastError = status ? String(status.last_error || '') : '';
    const listedAt = status ? String(status.listed_at || '') : '';
    const attempt = status?.latest_attempt;
    const tools = status && Array.isArray(status.tools) ? status.tools : [];
    const parts = [listedAt ? `${toolCountLabel(toolCount)} · listed ${age(listedAt)}` : ''];
    if (attempt?.ok === false) parts.push(`listing failed ${age(attempt.at)}`);
    else if (attempt?.partial) parts.push(`listing partial ${age(attempt.at)}`);
    else if (!listedAt) parts.push(attempt ? 'never listed' : 'not listed yet');
    else if (lastError) parts.push('failing');
    if (status?.publication_error) parts.push('catalog not published');
    const label = parts.filter(Boolean).join(' · ');
    const tone = lastError || status?.publication_error || attempt?.ok === false ? 'danger'
        : toolCount > 0 && !attempt?.partial ? 'ok' : listedAt ? 'warn' : 'muted';
    const facts = [
        ['Listed', listedAt ? absoluteTime(listedAt) || listedAt : ''],
        ['Last attempt', attempt?.at ? `${absoluteTime(attempt.at) || attempt.at}${attempt.ok === false ? ' · failed' : attempt.partial ? ' · partial' : ''}` : ''],
        ['Error', attempt?.ok === false ? String(attempt.error || '') : lastError],
        ['Catalog publication', status?.publication_error ? String(status.publication_error) : ''],
    ].filter(([, value]) => value);
    const factItems = facts.map(([name, value]) => `<div><dt>${escapeHtml(name)}</dt><dd>${escapeHtml(value)}</dd></div>`).join('');
    const toolItems = tools.map((t) => `
                <li>
                    <strong>${escapeHtml(t.name || t.prefixed_name || '')}</strong>
                    ${t.description ? `<span class="mcp-tool-desc">${escapeHtml(String(t.description).slice(0, 220))}</span>` : ''}
                </li>
            `).join('');
    return { badgeText: label.charAt(0).toUpperCase() + label.slice(1), badgeClass: `mcp-server-status mcp-server-status-${tone}`, factItems, toolItems };
}

/** The status row of a saved server: the backend keys it by the canonical identity (id, slug or name). */
function statusOf(server) {
    return mcpStatusByServer[canonicalServerId(serverIdentity(server))] || mcpStatusByServer[String(server.id ?? '')] || null;
}

function signInKey(server) {
    const signIn = statusOf(server)?.sign_in;
    return signIn ? `${signIn.state}:${signIn.reason || ''}:${signIn.required_scope || ''}:${signInLinks.get(server) || ''}` : '';
}

/** One fact and one button in view; the paste and the owner's own app registration under Details. */
function signInRow(server) {
    const signIn = statusOf(server)?.sign_in;
    if (!signIn) return '';
    if (signIn.state === 'signed_in' && signIn.reason !== 'insufficient_scope') {
        return `<span class="mcp-server-status mcp-server-status-ok">Signed in</span>
            <button type="button" class="btn btn-default" data-mcp-sign-in-action="sign_out">Sign out</button>`;
    }
    const link = signInLinks.get(server);
    const opener = link ? `<a href="${escapeHtml(safeExternalUrl(link))}" target="_blank" rel="noopener noreferrer" data-mcp-sign-in-link>Open the sign-in page ↗</a>` : '';
    const details = `<details class="mcp-sign-in-details"><summary>Details</summary>
            <label class="ui-field-help">If your browser did not come back here after signing in, paste the address it stopped on:
                <input type="text" class="ui-control" data-mcp-sign-in-returned autocomplete="off" spellcheck="false"></label>
            <button type="button" class="btn btn-default" data-mcp-sign-in-action="complete">Finish sign-in</button>
            <label class="ui-field-help">For a provider that registers no apps itself: your app's client ID and secret, used by Sign in
                <input type="text" class="ui-control" data-mcp-sign-in-client-id placeholder="Client ID" autocomplete="off" spellcheck="false">
                <input type="password" class="ui-control" data-mcp-sign-in-client-secret placeholder="Client secret (optional)" autocomplete="off"></label>
        </details>`;
    if (signIn.state === 'signed_in') { // still allowed what it was granted; a call needs more
        return `<span class="mcp-server-status mcp-server-status-warn">Signed in; a call needs more access: ${escapeHtml(signIn.required_scope || '')}</span>
            <button type="button" class="btn btn-default" data-mcp-sign-in-action="begin">Sign in again</button>
            <button type="button" class="btn btn-default" data-mcp-sign-in-action="sign_out">Sign out</button>
            ${opener}
            ${details}`;
    }
    const reason = SIGN_IN_REASONS[signIn.reason] || String(signIn.reason || 'not signed in yet').replaceAll('_', ' ');
    return `<span class="mcp-server-status mcp-server-status-warn">Sign-in needed: ${escapeHtml(reason)}</span>
        <button type="button" class="btn btn-default" data-mcp-sign-in-action="begin">Sign in</button>
        ${opener}
        ${details}`;
}

function paintSignIn(card, server) {
    const el = card.querySelector('[data-mcp-sign-in]');
    const key = signInKey(server);
    if (!el || el.dataset.signInKey === key) return; // unchanged: a typed paste or an open Details stays
    el.innerHTML = signInRow(server);
    el.dataset.signInKey = key;
    el.hidden = !key;
}

async function signInAction(card, server, sid, action) {
    if (!sid) {
        setCardMessage(server, 'Save the server before signing in.', 'warn');
        return;
    }
    const field = (name) => String(card.querySelector(`[data-mcp-sign-in-${name}]`)?.value || '').trim();
    const body = action === 'complete' ? { action, returned_url: field('returned') } : { action, server_id: sid };
    if (action === 'begin' && field('client-id')) Object.assign(body, { client_id: field('client-id'), client_secret: field('client-secret') });
    if (action === 'complete' && !body.returned_url) {
        setCardMessage(server, 'Paste the address your browser stopped on first.', 'warn', 'sign-in');
        return;
    }
    if (signInBusy.has(server)) return;
    signInBusy.add(server);
    const buttons = [...card.querySelectorAll('[data-mcp-sign-in-action]')];
    buttons.forEach((button) => { button.disabled = true; });
    setCardMessage(server, action === 'sign_out' ? 'Signing out...' : 'Contacting the provider...', 'muted', 'sign-in');
    try {
        const data = await jsonPost('/api/mcp/sign-in', body);
        if (data.ok === false) {
            setCardMessage(server, `Sign-in refused: ${data.error || data.reason || 'unknown reason'}`, 'danger', 'sign-in');
        } else if (action === 'begin') {
            signInLinks.set(server, data.url);
            setCardMessage(server, 'Finish signing in with the provider; this card updates when you come back here.', 'muted', 'sign-in');
            window.addEventListener('focus', () => { void refreshStatus(); }, { once: true });
            // Best effort while the press is recent; the link in the card works on every surface.
            void openExternalViaHostBridge(data.url).catch(() => {});
        } else if (action === 'complete') {
            signInLinks.delete(server);
            setCardMessage(server, `Signed in — ${toolCountLabel(Number(data.listing?.tool_count || 0))} listed.`, 'ok', 'sign-in');
        } else {
            setCardMessage(server, data.revoked === true ? 'Signed out; the provider confirmed the revocation.'
                : data.revoked === false ? 'Signed out here; the provider did not confirm the revocation.'
                    : 'Signed out here; the provider offers no revocation.', 'ok', 'sign-in');
        }
    } catch (err) {
        setCardMessage(server, `Sign-in failed: ${err && err.message ? err.message : err}`, 'danger', 'sign-in');
    } finally {
        signInBusy.delete(server);
        buttons.forEach((button) => { button.disabled = false; });
    }
    await refreshStatus();
}

function toolsDisclosure(toolItems) {
    return toolItems
        ? `<details class="mcp-tools-disclosure ui-disclosure"><summary>Discovered tools</summary><ul class="mcp-tools-list">${toolItems}</ul></details>`
        : '';
}

/**
 * A status answer arrives whenever its read finishes, often while the owner is
 * typing or reading: repaint changed status without replacing editable fields
 * or the disclosure summary. Unchanged tools retain text selection; changed
 * lists replace their content. Field focus, caret and Show state stay intact.
 */
function paintStatus() {
    for (const server of mcpServers) {
        const card = isServer(server) ? cardFor(server) : null;
        if (!card) continue;
        const { badgeText, badgeClass, factItems, toolItems } = statusParts(server);
        const weight = card.querySelector('[data-mcp-weight]');
        if (weight) { weight.textContent = weightLabel(server); weight.hidden = !weight.textContent; }
        const facts = card.querySelector('[data-mcp-facts]');
        if (facts && facts.innerHTML !== factItems) facts.innerHTML = factItems;
        const badge = card.querySelector('.mcp-server-status');
        badge.className = badgeClass;
        if (badge.textContent !== badgeText) badge.textContent = badgeText;
        paintSignIn(card, server);
        const disclosure = card.querySelector('.mcp-tools-disclosure');
        if (!toolItems) {
            disclosure?.remove();
        } else if (!disclosure) {
            card.insertAdjacentHTML('beforeend', toolsDisclosure(toolItems));
        } else {
            const list = disclosure.querySelector('.mcp-tools-list');
            const next = list.cloneNode(false);
            next.innerHTML = toolItems;
            if (!list.isEqualNode(next)) list.replaceChildren(...next.childNodes);
        }
    }
}

function renderServerCard(server, index) {
    const id = String(server.id ?? '');
    const name = String(server.name ?? '');
    const transport = String(server.transport ?? 'streamable_http');
    const url = String(server.url ?? '');
    const command = String(server.command ?? '');
    const args = Array.isArray(server.args) ? server.args.map(String) : [];
    const isStdio = transport === 'stdio';
    const unsupported = unsupportedFields(server);
    const envRefs = typeof server.env_from_settings === 'string'
        ? server.env_from_settings : JSON.stringify(server.env_from_settings ?? {}, null, 2);
    const literalEnv = typeof server.env === 'string' ? server.env : JSON.stringify(server.env ?? {}, null, 2);
    const enabled = server.enabled === true || server.enabled === 'True' || server.enabled === 'true';
    const weight = weightLabel(server);
    const allowedTools = Array.isArray(server.allowed_tools) ? server.allowed_tools.join(', ') : '';
    const transportOptions = TRANSPORTS.map((opt) => {
        const selected = opt.value === transport ? ' selected' : '';
        return `<option value="${escapeHtml(opt.value)}"${selected}>${escapeHtml(opt.label)}</option>`;
    }).join('');
    const { badgeText, badgeClass, factItems, toolItems } = statusParts(server);

    const message = cardMessages.get(server);
    const pending = pendingActions.get(server) || new Set();
    const notice = endpointNotice(server);
    const lifetime = lifetimeNote(server);

    // The row: the name, one state chip, the Enabled switch and the actions; the action's result right under
    // them; the form behind Edit (open for a server not saved yet); the stamps, the engine's text and the request
    // weight behind Details; then the discovered tools (DESIGN "List editors").
    return `
        <article class="mcp-server-card" data-mcp-card data-mcp-index="${index}">
            <header class="mcp-server-card-head">
                <div class="mcp-server-card-title">
                    <strong>${escapeHtml(name || id || `MCP Server ${index + 1}`)}</strong>
                    <span class="${badgeClass}">${escapeHtml(badgeText)}</span>
                </div>
                <div class="mcp-server-card-actions">
                    <label class="mcp-server-enabled">
                        <input class="ui-checkbox" type="checkbox" aria-label="MCP server ${index + 1}: Enabled" data-mcp-field="enabled" ${enabled ? 'checked' : ''}>
                        <span>Enabled</span>
                    </label>
                    <button type="button" class="btn btn-default" data-mcp-test ${pending.has('test') ? 'disabled' : ''}>Test draft</button>
                    <button type="button" class="btn btn-default" data-mcp-refresh ${pending.has('refresh') ? 'disabled' : ''}>Refresh catalog</button>
                    <button type="button" class="btn btn-default mcp-server-remove" data-mcp-remove>Remove</button>
                </div>
            </header>
            <div class="mcp-sign-in" data-mcp-sign-in data-sign-in-key="${escapeHtml(signInKey(server))}" ${signInKey(server) ? '' : 'hidden'}>${signInRow(server)}</div>
            <div class="settings-inline-status mcp-server-message" data-mcp-message ${message ? `data-tone="${escapeHtml(message.tone)}"` : 'hidden'}>${escapeHtml(message?.text || '')}</div>
            <details class="ui-disclosure mcp-server-edit" data-mcp-edit${editOpen.has(server) ? ' open' : ''}>
            <summary>Edit</summary>
            <div class="mcp-server-form">
            <div class="form-grid two">
                <div class="form-field ui-field">
                    <label for="mcp-${index}-id">Server ID</label>
                    <input type="text" class="ui-control" id="mcp-${index}-id" aria-label="MCP server ${index + 1}: Server ID" data-mcp-field="id" value="${escapeHtml(id)}" placeholder="github" autocomplete="off" spellcheck="false">
                </div>
                <div class="form-field ui-field">
                    <label for="mcp-${index}-name">Display name</label>
                    <input type="text" class="ui-control" id="mcp-${index}-name" aria-label="MCP server ${index + 1}: Display name" data-mcp-field="name" value="${escapeHtml(name)}" placeholder="GitHub MCP" autocomplete="off" spellcheck="false">
                </div>
            </div>
            <div class="form-grid two">
                <div class="form-field ui-field">
                    <label for="mcp-${index}-transport">Transport</label>
                    <select class="ui-control" id="mcp-${index}-transport" aria-label="MCP server ${index + 1}: Transport" data-mcp-field="transport">${transportOptions}</select>
                </div>
                <div class="form-field ui-field">
                    <label for="mcp-${index}-endpoint">${isStdio ? 'Command' : 'Server URL'}</label>
                    ${isStdio
                        ? `<input type="text" class="ui-control" id="mcp-${index}-endpoint" aria-label="MCP server ${index + 1}: Command" data-mcp-field="command" value="${escapeHtml(command)}" placeholder="npx" autocomplete="off" spellcheck="false">`
                        : `<input type="text" class="ui-control" id="mcp-${index}-endpoint" aria-label="MCP server ${index + 1}: Server URL" data-mcp-field="url" value="${escapeHtml(url)}" placeholder="https://example.com/mcp" autocomplete="off" spellcheck="false">
                    <span class="ui-field-help" data-mcp-endpoint-notice ${notice ? '' : 'hidden'}>${escapeHtml(notice)}</span>`}
                </div>
            </div>
            ${isStdio ? `
            <div class="form-row">
                <div class="form-field ui-field">
                    <label for="mcp-${index}-args">Arguments (one per line)</label>
                    <textarea class="ui-control" id="mcp-${index}-args" aria-label="MCP server ${index + 1}: Arguments (one per line)" data-mcp-field="args" rows="3" placeholder="-y&#10;@modelcontextprotocol/server-filesystem&#10;/path/to/folder" autocomplete="off" spellcheck="false">${escapeHtml(args.join('\n'))}</textarea>
                    <span class="muted">Each line is passed as one argument. Ouroboros does not use a shell.</span>
                </div>
            </div>
            <div class="form-grid two">
                <div class="form-field ui-field">
                    <label for="mcp-${index}-cwd">Working directory (optional)</label>
                    <input type="text" class="ui-control" id="mcp-${index}-cwd" aria-label="MCP server ${index + 1}: Working directory (optional)" data-mcp-field="cwd" value="${escapeHtml(String(server.cwd ?? ''))}" placeholder="/path/to/project" autocomplete="off" spellcheck="false">
                    <span class="muted">Used for both discovering and calling tools. Empty uses the default directory.</span>
                </div>
                <div class="form-field ui-field">
                    <label for="mcp-${index}-env_from_settings">Environment from settings (JSON)</label>
                    <textarea class="ui-control" id="mcp-${index}-env_from_settings" aria-label="MCP server ${index + 1}: Environment from settings (JSON)" data-mcp-field="env_from_settings" rows="4" placeholder='{"API_TOKEN": "MY_MCP_KEY"}' autocomplete="off" spellcheck="false">${escapeHtml(envRefs)}</textarea>
                    <span class="muted">Map environment names to saved setting keys. Put secret values in Settings → Custom keys.</span>
                </div>
            </div>
            <div class="form-row">
                <div class="form-field ui-field">
                    <label for="mcp-${index}-env">Environment (JSON, optional)</label>
                    <textarea class="ui-control" id="mcp-${index}-env" aria-label="MCP server ${index + 1}: Environment (JSON, optional)" data-mcp-field="env" rows="3" placeholder='{"PORT": "8080", "DEBUG": "1"}' autocomplete="off" spellcheck="false">${escapeHtml(literalEnv)}</textarea>
                    <span class="muted">Ordinary values passed directly to the process. Settings references override matching names; use Custom keys for secrets.</span>
                </div>
            </div>` : renderHeaderFields(server, index)}
            <div class="form-row">
                <div class="form-field ui-field">
                    <label for="mcp-${index}-allowed_tools">Allowed tools (optional, comma-separated)</label>
                    <input type="text" class="ui-control" id="mcp-${index}-allowed_tools" aria-label="MCP server ${index + 1}: Allowed tools (optional, comma-separated)" data-mcp-field="allowed_tools" value="${escapeHtml(allowedTools)}" placeholder="search, read_repo" autocomplete="off" spellcheck="false">
                </div>
            </div>
            <label class="mcp-server-enabled" title="Off: Ouroboros sees this server's tools in its connections map and loads their call schemas when it needs them. On: in Low and Max the call schemas are in every request from the start of each task.">
                <input class="ui-checkbox" type="checkbox" aria-label="MCP server ${index + 1}: Call schemas from task start" data-mcp-field="schemas_at_start" ${isPinned(server) ? 'checked' : ''}>
                <span>Schemas from task start</span>
            </label>
            ${lifetime ? `<div class="form-row"><span class="muted" data-mcp-lifetime>${escapeHtml(lifetime)}</span></div>` : ''}
            ${unsupported.length ? `<div class="ui-action-row"><span class="muted">Fields retained but not applied: ${escapeHtml(unsupported.join(', '))}</span>
                <button type="button" class="btn btn-default" data-mcp-clear-unsupported>Remove unsupported fields</button></div>` : ''}
            </div>
            </details>
            <details class="ui-disclosure mcp-server-details"><summary>Details</summary>
                <dl class="mcp-server-facts" data-mcp-facts>${factItems}</dl>
                <span class="ui-field-help" data-mcp-weight ${weight ? '' : 'hidden'}>${escapeHtml(weight)}</span>
            </details>
            ${toolsDisclosure(toolItems)}
        </article>
    `;
}

function bindCardEvents(card) {
    const idx = Number(card.dataset.mcpIndex || 0);
    const rowServer = mcpServers[idx];
    const saved = savedMcpSecrets.get(rowServer);
    bindHeaderFields(card, {server: rowServer, savedIdentity: saved?.identity,
        onChange: () => serverEdited(rowServer), render: renderAll});

    card.querySelector('[data-mcp-sign-in]')?.addEventListener('click', (event) => {
        const link = event.target.closest('a[data-mcp-sign-in-link]');
        if (link) {
            event.preventDefault(); // through the host while the press is live (desktop shell, Telegram)
            void openExternalViaHostBridge(link.getAttribute('href')).catch(() => {});
            return;
        }
        const button = event.target.closest('[data-mcp-sign-in-action]');
        if (button) void signInAction(card, rowServer, saved?.identity || '', button.dataset.mcpSignInAction);
    });

    card.querySelector('[data-mcp-edit]')?.addEventListener('toggle', (event) => {
        if (event.target.open) editOpen.add(rowServer); else editOpen.delete(rowServer);
    });
    card.querySelectorAll('[data-mcp-field]').forEach((input) => {
        input.addEventListener('input', () => {
            const field = input.dataset.mcpField;
            const server = rowServer;
            if (field === 'enabled' || field === 'schemas_at_start') {
                server[field] = Boolean(input.checked);
            } else if (field === 'allowed_tools') {
                server.allowed_tools = String(input.value || '')
                    .split(',')
                    .map((s) => s.trim())
                    .filter(Boolean);
            } else if (field === 'args') {
                server.args = String(input.value || '')
                    .split(/\r?\n/)
                    .filter((value) => value.length > 0);
            } else if (field === 'env_from_settings' || field === 'env') {
                try {
                    server[field] = JSON.parse(input.value || '{}');
                    setCardMessage(server, '');
                } catch {
                    server[field] = input.value;
                    setCardMessage(server, 'Environment values and references must be JSON objects. Your input is retained.', 'danger');
                }
            } else {
                server[field] = input.value;
            }
            if (field === 'transport') renderAll();
            serverEdited(server);
        });
        input.addEventListener('change', () => {
            const field = input.dataset.mcpField;
            if (field === 'enabled' || field === 'schemas_at_start') {
                rowServer[field] = Boolean(input.checked);
                if (field === 'schemas_at_start') paintStatus();
                serverEdited(rowServer);
            }
        });
    });

    card.querySelector('[data-mcp-clear-unsupported]')?.addEventListener('click', () => {
        const fields = unsupportedFields(rowServer);
        const credentials = fields.includes('headers') || fields.includes('auth_token');
        // The list was judged on what the header rows would save; keep that before
        // rebuilding the rows, or an edited saved legacy token comes back.
        if (credentials) Object.assign(rowServer, credentialPayload(rowServer));
        for (const key of fields) delete rowServer[key];
        if (credentials) resetHeaderRows(rowServer);
        renderAll();
        serverEdited(rowServer);
    });

    card.querySelector('[data-mcp-remove]')?.addEventListener('click', () => {
        const index = mcpServers.indexOf(rowServer);
        if (index >= 0) mcpServers.splice(index, 1);
        renderAll();
        revealAfterRemove(index);
        notifyChanged();
    });

    card.querySelector('[data-mcp-test]')?.addEventListener('click', async () => {
        const server = rowServer;
        const errors = validateHeaderFields(card, server, saved?.identity);
        if (errors.length) {
            setCardMessage(server, errors[0].message, 'danger');
            if (errors[0].input.hidden) resetSecretReveals(card); // a shown value stands in for its input
            errors[0].input.focus();
            return;
        }
        const generation = draftGenerations.get(server) || 0;
        setPending(server, 'test', true);
        setCardMessage(server, 'Testing connection...', 'muted', 'test');
        try {
            // The saved identity lets the backend restore kept values; the
            // candidate's own identity still decides which saved server they are.
            const candidate = collectServer(server);
            const urlMasked = String(server.url || '').includes('://***@');
            const keep = saved?.identity && (looksMasked(candidate.auth_token) || urlMasked || headerKeepsSavedValues(server));
            const body = keep ? { server_id: saved.identity, server: candidate } : { server: serverForTest(candidate) };
            const data = await jsonPost('/api/mcp/test', body, { rejectOkFalse: true });
            if (!mcpServers.includes(server)) return;
            if ((draftGenerations.get(server) || 0) !== generation) {
                setCardMessage(server, 'This server changed while it was tested. Test again to check the current draft.', 'warn', 'test');
                return;
            }
            const warnings = (data.configuration_warnings || []).join(' ');
            setCardMessage(server, `Draft connection OK — ${toolCountLabel(Number(data.tool_count || 0))} discovered. No tool was called.${warnings ? ' ' + warnings : ''}`, warnings ? 'warn' : 'ok', 'test');
        } catch (err) {
            if (!mcpServers.includes(server)) return;
            const stale = (draftGenerations.get(server) || 0) !== generation;
            setCardMessage(server, `Test failed${stale ? ' for an earlier draft' : ''}: ${err && err.message ? err.message : err}`, 'danger', 'test');
        } finally {
            setPending(server, 'test', false);
            await refreshStatus(); // a draft that asked for sign-in shows its Sign in
        }
    });

    card.querySelector('[data-mcp-refresh]')?.addEventListener('click', async () => {
        const server = rowServer;
        // The saved catalog belongs to the saved identity (id, slug or name).
        const sid = saved?.identity || '';
        if (!sid) {
            setCardMessage(server, 'Save the server before refreshing its saved catalog.', 'warn');
            return;
        }
        setPending(server, 'refresh', true);
        setCardMessage(server, 'Refreshing tools...', 'muted', 'refresh');
        try {
            const data = await jsonPost('/api/mcp/refresh', { server_id: sid }, { rejectOkFalse: true });
            if (data.published === false) throw new Error(data.publication_error || 'Catalog was not published');
            await refreshStatus();
            if (mcpServers.includes(server)) setCardMessage(server, `Saved catalog refreshed — ${toolCountLabel(Number(data.tool_count || 0))} discovered. No tool was called.`, 'ok', 'refresh');
        } catch (err) {
            await refreshStatus();
            if (mcpServers.includes(server)) setCardMessage(server, `Refresh failed: ${err && err.message ? err.message : err}`, 'danger', 'refresh');
        } finally {
            setPending(server, 'refresh', false);
        }
    });
}

function serverForTest(server) {
    const out = { ...server };
    if (looksMasked(out.auth_token)) {
        // Drop literal masks so inline tests never send "***" as Bearer auth.
        out.auth_token = '';
    }
    return out;
}

function renderAll() {
    const host = document.getElementById('mcp-servers-list');
    if (!host) return;
    resetSecretReveals(host);
    if (!mcpServers.length) {
        host.innerHTML = '<div class="muted">No MCP servers configured. Click "Add Server" to start.</div>';
        return;
    }
    host.innerHTML = mcpServers.map((s, idx) => s && typeof s === 'object' && !Array.isArray(s)
        ? renderServerCard(s, idx)
        : `<article class="mcp-server-card" data-mcp-invalid="${idx}"><span class="ui-field-help">MCP entry ${idx + 1} is malformed and retained unchanged. Remove it explicitly to discard it.</span><button type="button" class="btn btn-default" data-mcp-invalid-remove>Remove</button></article>`).join('');
    host.querySelectorAll('[data-mcp-card]').forEach((card) => bindCardEvents(card));
    host.querySelectorAll('[data-mcp-invalid-remove]').forEach((button) => button.addEventListener('click', () => {
        const index = Number(button.closest('[data-mcp-invalid]').dataset.mcpInvalid);
        mcpServers.splice(index, 1);
        renderAll(); revealAfterRemove(index); notifyChanged();
    }));
}

// After a Remove the caret goes to the entry now in that place, else to Add (DESIGN "List editors"): its first field
// while its form is open, else the Edit that opens it, since a field inside a closed disclosure cannot take the caret;
// a malformed entry has no field, only its own Remove.
function revealAfterRemove(index) {
    const card = document.getElementById('mcp-servers-list')?.children[Math.min(index, mcpServers.length - 1)];
    const edit = card?.querySelector('[data-mcp-edit]');
    const field = edit && !edit.open ? edit.querySelector('summary') : card?.querySelector('[data-mcp-field="id"]');
    if (card) revealNewRow(card, field || card.querySelector('button'));
    else document.getElementById('btn-mcp-add-server')?.focus();
}

function renderEnvelopeStatus() {
    const el = document.getElementById('mcp-global-status');
    if (!el) return;
    if (!mcpStatusEnvelope) {
        el.textContent = '';
        el.dataset.tone = 'muted';
        return;
    }
    if (!mcpStatusEnvelope.sdk_available) {
        el.textContent = `MCP SDK not installed: ${mcpStatusEnvelope.sdk_error || 'install `mcp>=1.10` to enable MCP integration.'}`;
        el.dataset.tone = 'warn';
        return;
    }
    if (!mcpStatusEnvelope.enabled) {
        el.textContent = 'MCP client is disabled. Enable it above to allow tool discovery.';
        el.dataset.tone = 'muted';
        return;
    }
    const total = Array.isArray(mcpStatusEnvelope.servers) ? mcpStatusEnvelope.servers.length : 0;
    const totalTools = Array.isArray(mcpStatusEnvelope.servers)
        ? mcpStatusEnvelope.servers.reduce((sum, s) => sum + Number(s.tool_count || 0), 0)
        : 0;
    const weights = mcpStatusEnvelope.weights;
    const cost = weights ? ` Connections map ${tokensLabel(weights.map_tokens)} in every Low/Max request; call schemas from task start ${tokensLabel(weights.schema_tokens_from_task_start)}.` : '';
    el.textContent = `${total} server${total === 1 ? '' : 's'} configured, ${totalTools} tool${totalTools === 1 ? '' : 's'} discovered.${cost}`;
    el.dataset.tone = 'ok';
}

async function refreshStatus() {
    try {
        const resp = await apiFetch('/api/mcp/status', { cache: 'no-store' });
        if (!resp.ok) return;
        const data = await resp.json();
        mcpStatusEnvelope = data;
        const map = {};
        for (const entry of data.servers || []) {
            if (entry && entry.id) map[entry.id] = entry;
        }
        mcpStatusByServer = map;
        renderEnvelopeStatus();
        paintStatus();
    } catch (err) {
    }
}

function bindAddButton() {
    const btn = document.getElementById('btn-mcp-add-server');
    if (!btn) return;
    btn.addEventListener('click', () => {
        const server = emptyServer();
        mcpServers.push(server);
        editOpen.add(server);
        renderAll();
        // The Add action lives in the section head while the new card lands
        // at the list's end — show it there and hand the caret to its id.
        const card = document.getElementById('mcp-servers-list')?.lastElementChild;
        revealNewRow(card, card?.querySelector?.('[data-mcp-field="id"]'));
        notifyChanged();
    });
}

function bindRefreshAllButton() {
    const btn = document.getElementById('btn-mcp-refresh-all');
    if (!btn) return;
    btn.addEventListener('click', async () => {
        btn.disabled = true;
        const wasText = btn.textContent;
        btn.textContent = 'Refreshing...';
        try {
            await jsonPost('/api/mcp/refresh', {}, { rejectOkFalse: true });
            await refreshStatus();
        } finally {
            btn.disabled = false;
            btn.textContent = wasText;
        }
    });
}

export function initMcpSettings({ onChange } = {}) {
    onChangeCallback = typeof onChange === 'function' ? onChange : null;
    bindAddButton();
    bindRefreshAllButton();
    const importButton = document.getElementById('btn-mcp-import');
    importButton?.addEventListener('click', async () => {
        const applied = await openMcpImportDialog({
            draftServers: () => mcpServers.map((server) => isServer(server) ? {
                ...(server.id ? { id: String(server.id) } : {}), ...(server.slug ? { slug: String(server.slug) } : {}),
                ...(server.name ? { name: String(server.name) } : {}),
                transport: String(server.transport || 'streamable_http'), ...headerImportFacts(server),
            } : null),
            draftFreshness,
            draftUrl: (index) => isServer(mcpServers[index]) ? String(mcpServers[index].url || '') : '',
            onApply: (entries) => {
                const touched = [];
                for (const entry of entries) {
                    if (entry.action === 'add') {
                        const server = {...entry.patch};
                        mcpServers.push(server);
                        editOpen.add(server);
                        touched.push(server);
                    } else if (entry.action === 'update') {
                        const server = mcpServers[entry.index];
                        Object.assign(server, entry.patch);
                        if (['headers', 'auth_token'].some((key) => Object.hasOwn(entry.patch, key))) resetHeaderRows(server);
                        draftGenerations.set(server, (draftGenerations.get(server) || 0) + 1);
                        if (cardMessages.get(server)?.kind === 'test') cardMessages.delete(server);
                        touched.push(server);
                    }
                }
                renderAll(); notifyChanged();
                return touched;
            },
        });
        // The dialog has returned focus to Import; hand it to the first card it changed.
        const card = applied?.length ? cardFor(applied.find((server) => !savedMcpSecrets.has(server)) || applied[0]) : null;
        revealNewRow(card, card?.querySelector('[data-mcp-field="id"]'));
    });
    const enabled = document.getElementById('s-mcp-enabled');
    if (enabled) enabled.addEventListener('change', notifyChanged);
    const timeout = document.getElementById('s-mcp-tool-timeout');
    if (timeout) timeout.addEventListener('input', notifyChanged);
}

export function applyMcpSettings(settings) {
    const enabledCheckbox = document.getElementById('s-mcp-enabled');
    if (enabledCheckbox) {
        enabledCheckbox.checked = settings.MCP_ENABLED === true || settings.MCP_ENABLED === 'True';
    }
    const timeoutInput = document.getElementById('s-mcp-tool-timeout');
    if (timeoutInput) {
        const value = Number(settings.MCP_TOOL_TIMEOUT_SEC || 60);
        timeoutInput.value = String(Number.isFinite(value) && value > 0 ? value : 60);
    }
    const incoming = Array.isArray(settings.MCP_SERVERS) ? settings.MCP_SERVERS : [];
    // auth_configured belongs to Settings response metadata, not user config;
    // the legacy pair's masked token itself becomes a header row.
    mcpServers = incoming.map((entry) => {
        if (!entry || typeof entry !== 'object' || Array.isArray(entry)) return entry;
        const { auth_configured: _authConfigured, ...s } = entry;
        const server = {
            ...s,
            id: String(s.id ?? ''),
            name: String(s.name ?? ''),
            enabled: Boolean(s.enabled),
            transport: String(s.transport ?? 'streamable_http'),
            url: String(s.url ?? ''),
            command: String(s.command ?? ''),
            args: s.args ?? [],
            auth_header: String(s.auth_header ?? 'Authorization'),
            auth_token: String(s.auth_token ?? ''),
            allowed_tools: Array.isArray(s.allowed_tools) ? s.allowed_tools.map(String) : [],
            schemas_at_start: isPinned(s),
        };
        savedMcpSecrets.set(server, { identity: serverIdentity(s), url: server.url });
        return server;
    });
    renderAll();
    refreshStatus();
}

/** Validate the current editable MCP draft without painting or changing it. */
export function validateMcpSettings() {
    const errors = [];
    document.querySelectorAll('[data-mcp-card]').forEach((card) => {
        const server = mcpServers[Number(card.dataset.mcpIndex)];
        if (server) errors.push(...validateHeaderFields(card, server, savedMcpSecrets.get(server)?.identity));
        const refused = server ? refusedFieldsMessage(server) : '';
        if (refused) errors.push({ input: card.querySelector('[data-mcp-field="transport"]'), message: refused });
    });
    document.querySelectorAll('[data-mcp-field="env"], [data-mcp-field="env_from_settings"]').forEach((input) => {
        let value, parsed = false;
        try { value = JSON.parse(input.value || '{}'); parsed = true; } catch {}
        if (!parsed || (value !== null && (typeof value !== 'object' || Array.isArray(value)
                || Object.entries(value).some(([name, item]) => !name || name.includes('=')
                    || name.includes('\u0000') || typeof item !== 'string' || item.includes('\u0000'))))) {
            errors.push({ input, message: 'Enter a JSON object with nonempty environment names and string values, for example {"PORT":"8080"}. Your input is retained.' });
        }
    });
    const timeout = document.getElementById('s-mcp-tool-timeout');
    if (timeout && (timeout.validity.badInput || (timeout.value.trim() && (!Number.isFinite(Number(timeout.value)) || Number(timeout.value) <= 0)))) {
        errors.push({ input: timeout, message: 'Enter a positive tool timeout in seconds.' });
    }
    return errors;
}

function isServer(server) {
    return Boolean(server) && typeof server === 'object' && !Array.isArray(server);
}

/** One server as Save and Test send it; header rows decide headers and the legacy pair. */
function collectServer(s) {
    return {
        ...s,
        id: String(s.id || '').trim(),
        name: String(s.name || '').trim(),
        enabled: Boolean(s.enabled),
        transport: String(s.transport || 'streamable_http'),
        url: String(s.url || '').trim(),
        command: String(s.command || '').trim(),
        args: s.args ?? [],
        auth_header: String(s.auth_header || 'Authorization').trim() || 'Authorization',
        auth_token: String(s.auth_token || ''),
        allowed_tools: Array.isArray(s.allowed_tools) ? s.allowed_tools.map(String) : [],
        schemas_at_start: isPinned(s),
        ...credentialPayload(s),
    };
}

/** Raw draft for dirty tracking: header rows are kept even where the payload collapses them. */
export function mcpDraftSnapshot() {
    return mcpServers.map((server) => isServer(server) ? headerDraftRows(server) : null);
}

/** Every field an import Apply could overwrite or match on, raw header rows and order included. */
function draftFreshness() {
    return JSON.stringify(mcpServers.map((server) => {
        if (!isServer(server)) return server;
        const { enabled: _enabled, allowed_tools: _allowedTools, schemas_at_start: _pin, ...fields } = server;
        return [fields, headerDraftRows(server)];
    }));
}

export function collectMcpSettings() {
    const enabledCheckbox = document.getElementById('s-mcp-enabled');
    const timeoutInput = document.getElementById('s-mcp-tool-timeout');
    const enabled = enabledCheckbox ? enabledCheckbox.checked : false;
    const timeoutRaw = timeoutInput ? Number(timeoutInput.value) : 60;
    const timeout = Number.isFinite(timeoutRaw) && timeoutRaw > 0 ? Math.floor(timeoutRaw) : 60;
    return {
        MCP_ENABLED: Boolean(enabled),
        MCP_TOOL_TIMEOUT_SEC: timeout,
        MCP_SERVERS: mcpServers.map((s) => isServer(s) ? collectServer(s) : s),
    };
}
