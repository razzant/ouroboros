/** Import an MCP client file into the tool servers of Settings: preview, select, then Apply saves the chosen ones. */
import { previewMcpImport } from './api_client.js';
import { escapeHtmlAttr as escapeHtml } from './utils.js';
import { bindDialogFocus } from './ui_interactions.js';

const TRANSPORT_LABELS = { streamable_http: 'Streamable HTTP', sse: 'SSE', stdio: 'Local process (stdio)' };

function endpoint(value) {
    try {
        const url = new URL(String(value || '').trim());
        return ['http:', 'https:'].includes(url.protocol) ? url : null;
    } catch {
        return null;
    }
}

/** An address change as origins only: never userinfo, path or query. Null when unchanged or unreadable. */
export function endpointChange(fromUrl, toUrl) {
    const from = endpoint(fromUrl), to = endpoint(toUrl);
    if (!from || !to || String(fromUrl).trim() === String(toUrl).trim()) return null;
    return { from: from.origin, to: to.origin, sameOrigin: from.origin === to.origin };
}

function detail(text) {
    return text ? `<span class="mcp-tool-desc">${escapeHtml(text)}</span>` : '';
}

function headerLines(entry) {
    const legacy = entry.legacy_header ? String(entry.legacy_header).toLowerCase() : '';
    const label = (name) => (name.toLowerCase() === legacy ? `${name} (saved single token)` : name);
    const changes = entry.header_changes;
    if (!changes) {
        return entry.action === 'add' && entry.header_names?.length ? [`Headers: ${entry.header_names.join(', ')}`] : [];
    }
    const lines = ['The file’s headers replace this server’s whole header list.'];
    if (changes.added.length) lines.push(`Adds: ${changes.added.join(', ')}`);
    if (changes.replaced.length) lines.push(`Replaces: ${changes.replaced.map(label).join(', ')}`);
    if (changes.removed.length) lines.push(`Removes: ${changes.removed.map(label).join(', ')}`);
    if (entry.unreadable_headers) lines.push('Removes the unreadable saved header list.');
    if (!changes.added.length && !changes.replaced.length && !changes.removed.length) lines.push('No headers remain.');
    return lines;
}

/** One previewed entry: names, counts and origins, never credential values, paths or queries. */
function renderEntry(entry, number, { checked, draftUrl }) {
    const blocked = !entry.action;
    const chip = blocked ? ['danger', 'Not applied'] : entry.action === 'add' ? ['ok', 'Add (off)'] : ['muted', 'Update'];
    const target = endpoint(entry.patch?.url);
    const where = entry.transport === 'stdio'
        ? (entry.arg_count ? `${entry.arg_count} argument${entry.arg_count === 1 ? '' : 's'}` : '')
        : (target ? target.origin : '');
    const summary = [entry.server_id ? `Server ID ${entry.server_id}` : '', TRANSPORT_LABELS[entry.transport] || '', where]
        .filter(Boolean).join(' · ');
    const change = entry.action === 'update' && Object.hasOwn(entry.patch || {}, 'url')
        ? endpointChange(draftUrl(entry.index), entry.patch.url) : null;
    const kept = change && !entry.header_changes ? (entry.kept_header_names || []) : [];
    const lines = (items, tone) => (items || []).map((text) => `<span class="settings-inline-status" data-tone="${tone}">${
        escapeHtml(text.charAt(0).toUpperCase() + text.slice(1))}</span>`).join('');
    const id = `mcp-import-entry-${number}`;
    const title = blocked
        ? `<strong>${escapeHtml(entry.source_name)}</strong>`
        : `<label class="mcp-import-choice" for="${id}"><input class="ui-checkbox" type="checkbox" id="${id}" data-mcp-import-select="${number}" ${checked ? 'checked' : ''}>
            <strong>${escapeHtml(entry.source_name)}</strong></label>`;
    return `<li class="mcp-import-entry">
        <span class="mcp-import-entry-head">${title}
            <span class="mcp-server-status mcp-server-status-${chip[0]}">${escapeHtml(chip[1])}</span></span>
        ${detail(summary)}
        ${headerLines(entry).map(detail).join('')}
        ${detail(entry.env_names?.length ? `Environment: ${entry.env_names.join(', ')}` : '')}
        ${detail(entry.patch?.task_session === true ? 'task_session: a task’s calls share one running server process until its tool loop ends, instead of one process per call.' : '')}
        ${detail(entry.patch?.browser_policy === true ? 'browser_policy: that process gets a private local address and token to ask Ouroboros’s browser rules about its requests. It does not limit the process otherwise.' : '')}
        ${detail(entry.action === 'update' ? `Fields supplied: ${Object.keys(entry.patch || {}).filter((key) => !['auth_header', 'auth_token'].includes(key)).join(', ')}` : '')}
        ${detail(entry.unsupported_keys?.length ? `Not imported: ${entry.unsupported_keys.join(', ')}` : '')}
        ${change ? lines([`Address ${change.sameOrigin ? `changes on ${change.to}` : `changes from ${change.from} to ${change.to}`}.${
            kept.length ? ` Saved values of ${kept.join(', ')} stay and go to the new address.` : ''}`], 'warn') : ''}
        ${lines(entry.problems, 'danger')}${lines(entry.warnings, 'warn')}
    </li>`;
}

/**
 * Open the dialog. ``draftServers()`` returns the matching metadata the
 * preview sends (identities, transports, header names — no values or
 * addresses); ``draftFreshness()`` is a local fingerprint of everything Apply
 * could overwrite; ``draftUrl(index)`` lets the preview name an address change
 * without sending the address. ``onApply(entries)`` applies the selected
 * entries to the draft and returns what it changed; the promise resolves with
 * that, or null when closed without applying.
 */
export function openMcpImportDialog({ draftServers, draftFreshness, draftUrl, onApply }) {
    return new Promise((resolve) => {
        const backdrop = document.createElement('div');
        backdrop.className = 'marketplace-modal-backdrop';
        backdrop.innerHTML = `
            <div class="marketplace-modal mcp-import-dialog" role="dialog" aria-modal="true" aria-labelledby="mcp-import-title">
                <div class="marketplace-modal-head">
                    <h3 id="mcp-import-title">Import mcp.json</h3>
                    <button type="button" class="btn btn-default btn-sm" data-mcp-import-cancel aria-label="Close">Close</button>
                </div>
                <div class="marketplace-modal-body">
                    <p class="settings-section-copy">Paste a document with a top-level <code>mcpServers</code> or <code>servers</code> object.
                        Preview shows what would change; choose the servers to apply. Apply saves the chosen servers.
                        Preview and Apply never connect or start anything; Test on a server's page checks it. New servers
                        start off. An existing server keeps its ID, its switch, its allowed tools and every field the document
                        does not name; a <code>headers</code> object replaces its whole header list.</p>
                    <div class="form-field ui-field">
                        <label for="mcp-import-text">mcp.json content</label>
                        <textarea class="ui-control" id="mcp-import-text" rows="10" spellcheck="false" autocomplete="off"
                            placeholder='{"mcpServers": {"docs": {"type": "http", "url": "https://example.com/mcp"}}}'></textarea>
                    </div>
                    <div class="settings-inline-status" data-mcp-import-status role="status" hidden></div>
                    <ul class="mcp-tools-list mcp-import-preview" data-mcp-import-preview hidden></ul>
                </div>
                <div class="marketplace-modal-actions">
                    <button type="button" class="btn btn-default" data-mcp-import-cancel>Cancel</button>
                    <button type="button" class="btn btn-default" data-mcp-import-preview-btn>Preview</button>
                    <button type="button" class="btn" data-primary data-mcp-import-apply disabled>Apply</button>
                </div>
            </div>`;
        const q = (selector) => backdrop.querySelector(selector);
        const text = q('#mcp-import-text');
        const statusEl = q('[data-mcp-import-status]');
        const list = q('[data-mcp-import-preview]');
        const apply = q('[data-mcp-import-apply]');
        let preview = null;
        let selected = new Set();
        let request = 0;
        let settled = false;
        let disposeFocus = () => {};
        let controller = null;
        const pageChanged = (event) => { if (event.detail?.page !== 'settings') finish(null); };
        const pageHidden = () => finish(null);
        const setStatus = (message, tone = 'muted') => {
            statusEl.textContent = message || '';
            statusEl.dataset.tone = tone;
            statusEl.hidden = !message;
        };
        const chosen = () => (preview?.entries || []).filter((entry, number) => entry.action && selected.has(number));
        const syncApply = () => {
            const count = chosen().length;
            apply.disabled = !count;
            apply.textContent = count ? `Apply ${count}` : 'Apply';
        };
        const invalidate = () => {
            controller?.abort(); controller = null;
            request += 1;
            preview = null;
            selected = new Set();
            list.hidden = true;
            list.innerHTML = '';
            syncApply();
            setStatus('');
        };
        const finish = (value) => {
            if (settled) return;
            settled = true;
            request += 1;
            controller?.abort();
            disposeFocus();
            window.removeEventListener('ouro:page-shown', pageChanged);
            window.removeEventListener('pagehide', pageHidden);
            text.value = ''; preview = null;
            backdrop.remove();
            resolve(value);
        };
        async function runPreview() {
            invalidate();
            const ticket = request;
            const fingerprint = draftFreshness();
            setStatus('Checking the document…');
            try {
                controller = new AbortController();
                const data = await previewMcpImport({ text: text.value, servers: draftServers() }, controller.signal);
                if (settled || ticket !== request) return;
                if (!data?.ok) {
                    setStatus(data?.error || 'The document could not be read.', 'danger');
                    return;
                }
                const entries = data.entries || [];
                preview = { ...data, fingerprint };
                selected = new Set(entries.map((entry, number) => (entry.action ? number : -1)).filter((number) => number >= 0));
                list.innerHTML = entries.map((entry, number) => renderEntry(entry, number, { checked: true, draftUrl })).join('');
                list.hidden = !entries.length;
                list.querySelectorAll('[data-mcp-import-select]').forEach((box) => box.addEventListener('change', () => {
                    const number = Number(box.dataset.mcpImportSelect);
                    if (box.checked) selected.add(number); else selected.delete(number);
                    syncApply();
                }));
                const count = selected.size;
                const blocked = entries.length - count;
                const ignored = (data.ignored_keys || []).length ? ` Other top-level keys are not imported: ${data.ignored_keys.join(', ')}.` : '';
                setStatus(!entries.length ? `The ${data.wrapper || 'mcpServers'} object is empty.${ignored}`
                    : `${count} of ${entries.length} server${entries.length === 1 ? '' : 's'} can be applied`
                        + `${blocked ? `; ${blocked} need${blocked === 1 ? 's' : ''} changes first` : ''}.${ignored}`,
                count && !blocked ? 'ok' : 'warn');
                syncApply();
            } catch (error) {
                if (settled || ticket !== request) return;
                setStatus(`Preview failed: ${error?.message || error}`, 'danger');
            }
        }
        text.addEventListener('input', invalidate);
        q('[data-mcp-import-preview-btn]').addEventListener('click', runPreview);
        apply.addEventListener('click', () => {
            if (!preview) return;
            if (draftFreshness() !== preview.fingerprint) {
                invalidate();
                setStatus('The servers changed after this preview. Preview again before applying.', 'warn');
                return;
            }
            const entries = chosen();
            if (!entries.length) return;
            const changed = onApply(entries);
            finish(changed);
        });
        backdrop.addEventListener('click', (event) => {
            if (event.target === backdrop || event.target.closest('[data-mcp-import-cancel]')) finish(null);
        });
        window.addEventListener('ouro:page-shown', pageChanged);
        window.addEventListener('pagehide', pageHidden);
        document.body.appendChild(backdrop);
        disposeFocus = bindDialogFocus(q('[role="dialog"]'), { initialFocus: text, onEscape: () => finish(null) });
    });
}
