import { apiFetch, jsonPost } from './api_client.js';
/** MCP settings cards; preserves masked credentials until the user edits them. */
import { escapeHtmlAttr as escapeHtml } from './utils.js';
import { revealNewRow } from './ui_helpers.js';
import { resetSecretReveals } from './settings_secrets.js';
import {
    renderHeaderFields, bindHeaderFields, credentialPayload, headerKeepsSavedValues, keptHeaderNames,
    headerDraftRows, headerImportFacts, resetHeaderRows, serverIdentity, validateHeaderFields,
} from './mcp_header_fields.js';
import { endpointChange, openMcpImportDialog } from './mcp_import_dialog.js';

const TRANSPORTS = [
    { value: 'streamable_http', label: 'Streamable HTTP' },
    { value: 'sse', label: 'SSE (Server-Sent Events)' },
    { value: 'stdio', label: 'Local process (stdio)' },
];
const SERVER_FIELDS = new Set(['id', 'slug', 'name', 'label', 'enabled', 'transport', 'url',
    'command', 'args', 'auth_header', 'auth_token', 'headers', 'allowed_tools', 'cwd', 'env', 'env_from_settings',
    'browser_bridge', 'browser_protocol', 'browser_skill_binding']);

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
        browser_bridge: false,
        browser_protocol: 'playwright',
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
    return fields;
}

/** The saved-status parts of a card: its badge and its discovered tools' list items. */
function statusParts(server) {
    const status = mcpStatusByServer[String(server.id ?? '')] || null;
    const toolCount = status ? Number(status.tool_count || 0) : 0;
    const lastError = status ? String(status.last_error || '') : '';
    const lastRefreshed = status ? String(status.last_refreshed || '') : '';
    const tools = status && Array.isArray(status.tools) ? status.tools : [];
    const badge = lastError ? [`Error: ${lastError}`, 'mcp-server-status-danger']
        : toolCount > 0 ? [`${toolCountLabel(toolCount)} discovered`, 'mcp-server-status-ok']
            : lastRefreshed ? ['0 tools discovered', 'mcp-server-status-warn'] : ['Not refreshed yet', 'mcp-server-status-muted'];
    const toolItems = tools.map((t) => `
                <li>
                    <strong>${escapeHtml(t.name || t.prefixed_name || '')}</strong>
                    ${t.description ? `<span class="mcp-tool-desc">${escapeHtml(String(t.description).slice(0, 220))}</span>` : ''}
                </li>
            `).join('');
    return { badgeText: badge[0], badgeClass: `mcp-server-status ${badge[1]}`, toolItems };
}

function toolsDisclosure(toolItems) {
    return toolItems
        ? `<details class="mcp-tools-disclosure"><summary>Discovered tools</summary><ul class="mcp-tools-list">${toolItems}</ul></details>`
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
        const { badgeText, badgeClass, toolItems } = statusParts(server);
        const badge = card.querySelector('.mcp-server-status');
        badge.className = badgeClass;
        if (badge.textContent !== badgeText) badge.textContent = badgeText;
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
    const allowedTools = Array.isArray(server.allowed_tools) ? server.allowed_tools.join(', ') : '';
    const transportOptions = TRANSPORTS.map((opt) => {
        const selected = opt.value === transport ? ' selected' : '';
        return `<option value="${escapeHtml(opt.value)}"${selected}>${escapeHtml(opt.label)}</option>`;
    }).join('');
    const { badgeText, badgeClass, toolItems } = statusParts(server);

    const message = cardMessages.get(server);
    const pending = pendingActions.get(server) || new Set();
    const notice = endpointNotice(server);

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
                    <button type="button" class="btn btn-default" data-mcp-test ${pending.has('test') ? 'disabled' : ''}>Test draft catalog</button>
                    <button type="button" class="btn btn-default" data-mcp-refresh ${pending.has('refresh') ? 'disabled' : ''}>Refresh saved catalog</button>
                    <button type="button" class="btn btn-default mcp-server-remove" data-mcp-remove>Remove</button>
                </div>
            </header>
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
                <label class="mcp-server-enabled">
                    <input class="ui-checkbox" type="checkbox" data-mcp-field="browser_bridge" ${server.browser_bridge === true ? 'checked' : ''}>
                    <span>Task-owned browser bridge</span>
                </label>
            </div>
            <div class="form-row">
                <div class="form-field ui-field">
                    <label for="mcp-${index}-browser-protocol">Browser protocol</label>
                    <select class="ui-control" id="mcp-${index}-browser-protocol" data-mcp-field="browser_protocol" aria-label="MCP server ${index + 1}: Browser protocol">
                        <option value="playwright" ${server.browser_protocol !== 'ouroboros_chrome' ? 'selected' : ''}>Playwright MCP</option>
                        <option value="ouroboros_chrome" ${server.browser_protocol === 'ouroboros_chrome' ? 'selected' : ''}>Ouroboros Chrome</option>
                    </select>
                    <span class="ui-field-help">Chrome requires a task-owned bridge and a connected extension.</span>
                </div>
            </div>
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
            <div class="settings-inline-status mcp-server-message" data-mcp-message ${message ? `data-tone="${escapeHtml(message.tone)}"` : 'hidden'}>${escapeHtml(message?.text || '')}</div>
            ${unsupported.length ? `<div class="form-row"><span class="muted">Fields retained but not applied: ${escapeHtml(unsupported.join(', '))}</span>
                <button type="button" class="btn btn-default" data-mcp-clear-unsupported>Remove unsupported fields</button></div>` : ''}
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

    card.querySelectorAll('[data-mcp-field]').forEach((input) => {
        input.addEventListener('input', () => {
            const field = input.dataset.mcpField;
            const server = rowServer;
            if (field === 'enabled' || field === 'browser_bridge') {
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
            if (field === 'enabled' || field === 'browser_bridge') {
                rowServer[field] = Boolean(input.checked);
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
        notifyChanged();
    });

    card.querySelector('[data-mcp-test]')?.addEventListener('click', async () => {
        const server = rowServer;
        const errors = validateHeaderFields(card, server, saved?.identity);
        if (errors.length) { setCardMessage(server, errors[0].message, 'danger'); errors[0].input.focus(); return; }
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
            await refreshStatus();
            if (mcpServers.includes(server)) setCardMessage(server, `Saved catalog refreshed — ${toolCountLabel(Number(data.tool_count || 0))} discovered. No tool was called.`, 'ok', 'refresh');
        } catch (err) {
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
        mcpServers.splice(Number(button.closest('[data-mcp-invalid]').dataset.mcpInvalid), 1);
        renderAll(); notifyChanged();
    }));
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
    el.textContent = `${total} server${total === 1 ? '' : 's'} configured, ${totalTools} tool${totalTools === 1 ? '' : 's'} discovered.`;
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
        mcpServers.push(emptyServer());
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
            browser_bridge: s.browser_bridge === true,
            browser_protocol: s.browser_protocol === 'ouroboros_chrome' ? 'ouroboros_chrome' : 'playwright',
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
        if (server) {
            errors.push(...validateHeaderFields(card, server, savedMcpSecrets.get(server)?.identity));
            if (server.browser_protocol === 'ouroboros_chrome'
                    && (server.transport !== 'stdio' || server.browser_bridge !== true)) {
                errors.push({ input: card.querySelector('[data-mcp-field="browser_protocol"]')
                    || card.querySelector('[data-mcp-field="transport"]'),
                    message: 'Ouroboros Chrome requires a local stdio server with Task-owned browser bridge enabled.' });
            }
        }
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
        browser_bridge: String(s.transport || 'streamable_http') === 'stdio' && s.browser_bridge === true,
        browser_protocol: s.browser_protocol === 'ouroboros_chrome' ? 'ouroboros_chrome' : 'playwright',
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
        const { enabled: _enabled, allowed_tools: _allowedTools, ...fields } = server;
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
