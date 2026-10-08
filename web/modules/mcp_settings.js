import { apiFetch, jsonPost } from './api_client.js';
/** MCP settings cards; preserves masked auth tokens until the user edits them. */
import { escapeHtmlAttr as escapeHtml } from './utils.js';
import { revealNewRow } from './ui_helpers.js';
import { bindSecretReveal, resetSecretReveals } from './settings_secrets.js';
import { renderHeaderFields, bindHeaderFields, headerPayload, headerHasMask, resetHeaderRows, validateHeaderFields } from './mcp_header_fields.js';
import { openMcpImportDialog } from './mcp_import_dialog.js';

const TRANSPORTS = [
    { value: 'streamable_http', label: 'Streamable HTTP' },
    { value: 'sse', label: 'SSE (Server-Sent Events)' },
    { value: 'stdio', label: 'Local process (stdio)' },
];
const SERVER_FIELDS = new Set(['id', 'slug', 'name', 'label', 'enabled', 'transport', 'url',
    'command', 'args', 'auth_header', 'auth_token', 'headers', 'allowed_tools', 'cwd', 'env', 'env_from_settings']);

let mcpServers = [];
let mcpStatusByServer = {};
let mcpStatusEnvelope = null;
let mcpDirtyTokens = new Set();
let onChangeCallback = null;
// Original lookup identity and masked value only; never serialize these into a
// server draft or use its editable id / current list index to reveal a token.
const savedMcpSecrets = new WeakMap();

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

function toolCountLabel(count) {
    return `${count} tool${count === 1 ? '' : 's'}`;
}

function unsupportedFields(server) {
    const unused = server.transport === 'stdio' ? ['url', 'auth_token', 'headers']
        : ['command', 'args', 'cwd', 'env', 'env_from_settings'];
    const fields = Object.keys(server).filter((key) => !SERVER_FIELDS.has(key)
        || (unused.includes(key) && !['', '[]', '{}', 'null'].includes(JSON.stringify(server[key]))
            && server[key] !== ''));
    if (server.transport === 'stdio' && server.auth_header && server.auth_header !== 'Authorization') {
        fields.push('auth_header');
    }
    return fields;
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
    const authHeader = String(server.auth_header ?? 'Authorization');
    const authToken = String(server.auth_token ?? '');
    const enabled = server.enabled === true || server.enabled === 'True' || server.enabled === 'true';
    const allowedTools = Array.isArray(server.allowed_tools) ? server.allowed_tools.join(', ') : '';
    const transportOptions = TRANSPORTS.map((opt) => {
        const selected = opt.value === transport ? ' selected' : '';
        return `<option value="${escapeHtml(opt.value)}"${selected}>${escapeHtml(opt.label)}</option>`;
    }).join('');
    const status = mcpStatusByServer[id] || null;
    const toolCount = status ? Number(status.tool_count || 0) : 0;
    const lastError = status ? String(status.last_error || '') : '';
    const lastRefreshed = status ? String(status.last_refreshed || '') : '';
    const tools = status && Array.isArray(status.tools) ? status.tools : [];

    let statusBadgeText = 'Not refreshed yet';
    let statusClass = 'mcp-server-status-muted';
    if (lastError) {
        statusBadgeText = `Error: ${lastError}`;
        statusClass = 'mcp-server-status-danger';
    } else if (toolCount > 0) {
        statusBadgeText = `${toolCount} tool${toolCount === 1 ? '' : 's'} discovered`;
        statusClass = 'mcp-server-status-ok';
    } else if (lastRefreshed) {
        statusBadgeText = '0 tools discovered';
        statusClass = 'mcp-server-status-warn';
    }

    const toolsHtml = tools.length
        ? `<ul class="mcp-tools-list">${tools.map((t) => `
                <li>
                    <strong>${escapeHtml(t.name || t.prefixed_name || '')}</strong>
                    ${t.description ? `<span class="mcp-tool-desc">${escapeHtml(String(t.description).slice(0, 220))}</span>` : ''}
                </li>
            `).join('')}</ul>`
        : '';

    const authPlaceholder = authToken && looksMasked(authToken)
        ? authToken
        : (authToken ? '••••••' : 'Bearer xxxxx (optional)');

    return `
        <article class="mcp-server-card" data-mcp-card data-mcp-index="${index}">
            <header class="mcp-server-card-head">
                <div class="mcp-server-card-title">
                    <strong>${escapeHtml(name || id || `MCP Server ${index + 1}`)}</strong>
                    <span class="mcp-server-status ${statusClass}">${escapeHtml(statusBadgeText)}</span>
                </div>
                <div class="mcp-server-card-actions">
                    <label class="mcp-server-enabled">
                        <input class="ui-checkbox" type="checkbox" aria-label="MCP server ${index + 1}: Enabled" data-mcp-field="enabled" ${enabled ? 'checked' : ''}>
                        <span>Enabled</span>
                    </label>
                    <button type="button" class="btn btn-default" data-mcp-test>Test draft catalog</button>
                    <button type="button" class="btn btn-default" data-mcp-refresh>Refresh saved catalog</button>
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
                        : `<input type="text" class="ui-control" id="mcp-${index}-endpoint" aria-label="MCP server ${index + 1}: Server URL" data-mcp-field="url" value="${escapeHtml(url)}" placeholder="https://example.com/mcp" autocomplete="off" spellcheck="false">`}
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
            </div>` : `
            <div class="form-grid two">
                <div class="form-field ui-field">
                    <label for="mcp-${index}-auth_header">Legacy header name</label>
                    <input type="text" class="ui-control" id="mcp-${index}-auth_header" aria-label="MCP server ${index + 1}: Auth header" data-mcp-field="auth_header" value="${escapeHtml(authHeader)}" placeholder="Authorization" autocomplete="off" spellcheck="false">
                </div>
                <div class="form-field ui-field">
                    <label for="mcp-${index}-auth_token">Legacy header value (optional)</label>
                    <div class="secret-input-row">
                        <input type="password" class="ui-control" id="mcp-${index}-auth_token" aria-label="MCP server ${index + 1}: Auth token (optional)" data-mcp-field="auth_token" value="${escapeHtml(authToken)}" placeholder="${escapeHtml(authPlaceholder)}" autocomplete="off" spellcheck="false">
                        <button type="button" class="btn btn-default" data-mcp-token-toggle>Show</button>
                        <button type="button" class="btn btn-default" data-mcp-token-clear>Clear</button>
                    </div>
                </div>
            </div>
            ${renderHeaderFields(server, index)}`}
            <div class="form-row">
                <div class="form-field ui-field">
                    <label for="mcp-${index}-allowed_tools">Allowed tools (optional, comma-separated)</label>
                    <input type="text" class="ui-control" id="mcp-${index}-allowed_tools" aria-label="MCP server ${index + 1}: Allowed tools (optional, comma-separated)" data-mcp-field="allowed_tools" value="${escapeHtml(allowedTools)}" placeholder="search, read_repo" autocomplete="off" spellcheck="false">
                </div>
            </div>
            <div class="settings-inline-status mcp-server-message" data-mcp-message hidden></div>
            ${unsupported.length ? `<div class="form-row"><span class="muted">Fields retained but not applied: ${escapeHtml(unsupported.join(', '))}</span>
                <button type="button" class="btn btn-default" data-mcp-clear-unsupported>Remove unsupported fields</button></div>` : ''}
            ${toolsHtml ? `<details class="mcp-tools-disclosure"><summary>Discovered tools</summary>${toolsHtml}</details>` : ''}
        </article>
    `;
}

function bindCardEvents(card) {
    const idx = Number(card.dataset.mcpIndex || 0);
    const rowServer = mcpServers[idx];
    bindHeaderFields(card, {server: rowServer, savedIdentity: savedMcpSecrets.get(rowServer)?.identity,
        onChange: notifyChanged, render: renderAll});
    const setMessage = (text, tone = 'muted') => {
        const el = card.querySelector('[data-mcp-message]');
        if (!el) return;
        if (!text) {
            el.hidden = true;
            el.textContent = '';
            return;
        }
        el.textContent = text;
        el.dataset.tone = tone;
        el.hidden = false;
    };

    card.querySelectorAll('[data-mcp-field]').forEach((input) => {
        input.addEventListener('input', () => {
            const field = input.dataset.mcpField;
            const server = mcpServers[idx];
            if (!server) return;
            if (field === 'enabled') {
                server.enabled = Boolean(input.checked);
            } else if (field === 'allowed_tools') {
                server.allowed_tools = String(input.value || '')
                    .split(',')
                    .map((s) => s.trim())
                    .filter(Boolean);
            } else if (field === 'args') {
                server.args = String(input.value || '')
                    .split(/\r?\n/)
                    .filter((value) => value.length > 0);
            } else if (field === 'auth_token') {
                if (looksMasked(input.value)) {
                    mcpDirtyTokens.delete(`${idx}`);
                } else {
                    mcpDirtyTokens.add(`${idx}`);
                }
                server.auth_token = input.value;
            } else if (field === 'env_from_settings' || field === 'env') {
                try {
                    server[field] = JSON.parse(input.value || '{}');
                    setMessage('');
                } catch {
                    server[field] = input.value;
                    setMessage('Environment values and references must be JSON objects. Your input is retained.', 'danger');
                }
            } else {
                server[field] = input.value;
            }
            if (field === 'transport') renderAll();
            notifyChanged();
        });
        input.addEventListener('change', () => {
            const field = input.dataset.mcpField;
            const server = mcpServers[idx];
            if (!server) return;
            if (field === 'enabled') {
                server.enabled = Boolean(input.checked);
                notifyChanged();
            }
        });
    });

    card.querySelector('[data-mcp-clear-unsupported]')?.addEventListener('click', () => {
        for (const key of unsupportedFields(mcpServers[idx])) delete mcpServers[idx][key];
        renderAll();
        notifyChanged();
    });

    const tokenInput = card.querySelector('[data-mcp-field="auth_token"]');
    const tokenToggle = card.querySelector('[data-mcp-token-toggle]');
    const tokenClear = card.querySelector('[data-mcp-token-clear]');
    let tokenReveal;
    if (tokenToggle && tokenInput) {
        const server = mcpServers[idx];
        const saved = savedMcpSecrets.get(server);
        tokenInput.dataset.appliedValue = saved?.value || '';
        tokenReveal = bindSecretReveal(tokenInput, tokenToggle, {
            savedSelector: () => saved?.identity ? { mcp_server_id: saved.identity } : null,
            savedLabel: () => saved && String(server.id || server.slug || server.name || '') !== saved.identity
                ? `Saved value for ${saved.identity}` : '',
            identityInputs: ['id', 'name'].map((field) => card.querySelector(`[data-mcp-field="${field}"]`)).filter(Boolean),
        });
    }
    if (tokenClear && tokenInput) {
        tokenClear.addEventListener('click', () => {
            tokenReveal?.reset();
            tokenInput.value = '';
            tokenInput.type = 'password';
            const server = mcpServers[idx];
            if (server) server.auth_token = '';
            mcpDirtyTokens.add(`${idx}`);
            notifyChanged();
        });
    }

    const removeBtn = card.querySelector('[data-mcp-remove]');
    if (removeBtn) {
        removeBtn.addEventListener('click', () => {
            mcpServers.splice(idx, 1);
            mcpDirtyTokens.delete(`${idx}`);
            renderAll();
            notifyChanged();
        });
    }

    const testBtn = card.querySelector('[data-mcp-test]');
    if (testBtn) {
        testBtn.addEventListener('click', async () => {
            const server = mcpServers[idx];
            if (!server) return;
            const errors = validateHeaderFields(card, server);
            if (errors.length) { setMessage(errors[0].message, 'danger'); errors[0].input.focus(); return; }
            testBtn.disabled = true;
            setMessage('Testing connection...', 'muted');
            try {
                // Saved identity lets the backend rehydrate either credential form.
                const sid = String(server.id || '').trim();
                const tokenMasked = looksMasked(server.auth_token);
                const urlMasked = String(server.url || '').includes('://***@');
                const body = sid && (tokenMasked || urlMasked || headerHasMask(server))
                    ? { server_id: sid, server: { ...server, ...headerPayload(server) } }
                    : { server: serverForTest(server) };
                const data = await jsonPost('/api/mcp/test', body, { rejectOkFalse: true });
                const warnings = (data.configuration_warnings || []).join(' ');
                setMessage(`Draft connection OK — ${toolCountLabel(Number(data.tool_count || 0))} discovered. No tool was called.${warnings ? ' ' + warnings : ''}`, warnings ? 'warn' : 'ok');
            } catch (err) {
                setMessage(`Test failed: ${err && err.message ? err.message : err}`, 'danger');
            } finally {
                testBtn.disabled = false;
            }
        });
    }

    const refreshBtn = card.querySelector('[data-mcp-refresh]');
    if (refreshBtn) {
        refreshBtn.addEventListener('click', async () => {
            const server = mcpServers[idx];
            if (!server) return;
            const sid = String(server.id || '').trim();
            if (!sid) {
                setMessage('Save the server (with an ID) before refreshing.', 'warn');
                return;
            }
            refreshBtn.disabled = true;
            setMessage('Refreshing tools...', 'muted');
            try {
                const data = await jsonPost('/api/mcp/refresh', { server_id: sid }, { rejectOkFalse: true });
                await refreshStatus();
                const current = document.querySelector(`[data-mcp-index="${idx}"] [data-mcp-message]`);
                if (current) { current.textContent = `Saved catalog refreshed — ${Number(data.tool_count || 0)} tools discovered. No tool was called.`;
                    current.dataset.tone = 'ok'; current.hidden = false; }
            } catch (err) {
                setMessage(`Refresh failed: ${err && err.message ? err.message : err}`, 'danger');
            } finally {
                refreshBtn.disabled = false;
            }
        });
    }
}

function serverForTest(server) {
    const out = { ...server, ...headerPayload(server) };
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
        el.textContent = `MCP SDK not installed: ${mcpStatusEnvelope.sdk_error || 'install `mcp>=1.6` to enable MCP integration.'}`;
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
        renderAll();
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
    importButton?.addEventListener('click', () => openMcpImportDialog({
        draftIdentity: () => mcpServers.map((server) => server && typeof server === 'object' && !Array.isArray(server)
            ? {...server, ...headerPayload(server)} : server),
        onApply: (entries) => {
            for (const entry of entries) {
                if (entry.action === 'add') mcpServers.push({...entry.patch});
                else if (entry.action === 'update') {
                    const server = mcpServers[entry.index];
                    Object.assign(server, entry.patch);
                    if (Object.hasOwn(entry.patch, 'headers')) resetHeaderRows(server);
                }
            }
            renderAll(); notifyChanged();
        },
    }));
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
    // auth_configured belongs to Settings response metadata, not user config.
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
        };
        savedMcpSecrets.set(server, {
            identity: String(s.id || s.slug || s.name || ''), value: server.auth_token,
        });
        return server;
    });
    mcpDirtyTokens = new Set();
    renderAll();
    refreshStatus();
}

/** Validate the current editable MCP draft without painting or changing it. */
export function validateMcpSettings() {
    const errors = [];
    document.querySelectorAll('[data-mcp-card]').forEach((card) => {
        const server = mcpServers[Number(card.dataset.mcpIndex)];
        if (server) errors.push(...validateHeaderFields(card, server));
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

export function collectMcpSettings() {
    const enabledCheckbox = document.getElementById('s-mcp-enabled');
    const timeoutInput = document.getElementById('s-mcp-tool-timeout');
    const enabled = enabledCheckbox ? enabledCheckbox.checked : false;
    const timeoutRaw = timeoutInput ? Number(timeoutInput.value) : 60;
    const timeout = Number.isFinite(timeoutRaw) && timeoutRaw > 0 ? Math.floor(timeoutRaw) : 60;
    const out = {
        MCP_ENABLED: Boolean(enabled),
        MCP_TOOL_TIMEOUT_SEC: timeout,
        MCP_SERVERS: mcpServers.map((s) => {
            if (!s || typeof s !== 'object' || Array.isArray(s)) return s;
            const transport = String(s.transport || 'streamable_http');
            return {
                ...s,
                ...headerPayload(s),
                id: String(s.id || '').trim(),
                name: String(s.name || '').trim(),
                enabled: Boolean(s.enabled),
                transport,
                url: String(s.url || '').trim(),
                command: String(s.command || '').trim(),
                args: s.args ?? [],
                auth_header: String(s.auth_header || 'Authorization').trim() || 'Authorization',
                auth_token: String(s.auth_token || ''),
                allowed_tools: Array.isArray(s.allowed_tools) ? s.allowed_tools.map(String) : [],
            };
        }),
    };
    return out;
}
