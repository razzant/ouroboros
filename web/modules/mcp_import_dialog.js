/** Import mcp.json into the MCP Settings draft: preview first, apply to the draft, never save. */
import { previewMcpImport } from './api_client.js';
import { escapeHtmlAttr as escapeHtml } from './utils.js';
import { bindDialogFocus } from './ui_interactions.js';

const TRANSPORT_LABELS = { streamable_http: 'Streamable HTTP', sse: 'SSE', stdio: 'Local process (stdio)' };

function detail(text) {
    return text ? `<span class="mcp-tool-desc">${escapeHtml(text)}</span>` : '';
}

/** One previewed entry: names and counts, never credential or address values. */
function renderEntry(entry) {
    const blocked = !entry.action;
    const chip = blocked ? ['danger', 'Not applied'] : entry.action === 'add' ? ['ok', 'Add (disabled)'] : ['muted', 'Update'];
    const where = entry.transport === 'stdio'
        ? [entry.command, entry.arg_count ? `${entry.arg_count} argument${entry.arg_count === 1 ? '' : 's'}` : ''].filter(Boolean).join(' · ')
        : entry.url;
    const summary = [entry.server_id ? `Server ID ${entry.server_id}` : '', TRANSPORT_LABELS[entry.transport] || '', where]
        .filter(Boolean).join(' · ');
    const lines = (items, tone) => (items || []).map((text) => `<span class="settings-inline-status" data-tone="${tone}">${
        escapeHtml(text.charAt(0).toUpperCase() + text.slice(1))}</span>`).join('');
    return `<li class="mcp-import-entry">
        <span><strong>${escapeHtml(entry.source_name)}</strong>
            <span class="mcp-server-status mcp-server-status-${chip[0]}">${escapeHtml(chip[1])}</span></span>
        ${detail(summary)}
        ${detail(entry.header_names?.length ? `Headers: ${entry.header_names.join(', ')}` : '')}
        ${detail(entry.env_names?.length ? `Environment: ${entry.env_names.join(', ')}` : '')}
        ${detail(entry.action === 'update' ? `Fields supplied: ${Object.keys(entry.patch || {}).join(', ')}` : '')}
        ${detail(entry.unsupported_keys?.length ? `Not imported: ${entry.unsupported_keys.join(', ')}` : '')}
        ${lines(entry.problems, 'danger')}${lines(entry.warnings, 'warn')}
    </li>`;
}

/**
 * Open the dialog. ``draftIdentity()`` returns the current draft projection the
 * preview matches against; ``onApply(entries)`` applies previewed entries to the
 * draft. Resolves with the applied count, or null when closed without applying.
 */
export function openMcpImportDialog({ draftIdentity, onApply }) {
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
                    <p class="settings-section-copy">Paste a document with a top-level <code>mcpServers</code> object.
                        Preview shows what would change. Apply changes only this Settings draft — nothing connects,
                        runs or is saved until you press Save. New servers start disabled; an existing server keeps
                        its ID, enabled state, allowed tools and every field the document does not name.</p>
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
                    <button type="button" class="btn btn-primary" data-mcp-import-apply disabled>Apply to draft</button>
                </div>
            </div>`;
        const q = (selector) => backdrop.querySelector(selector);
        const text = q('#mcp-import-text');
        const statusEl = q('[data-mcp-import-status]');
        const list = q('[data-mcp-import-preview]');
        const apply = q('[data-mcp-import-apply]');
        let preview = null;
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
        const applicable = () => (preview?.entries || []).filter((entry) => entry.action);
        const invalidate = () => {
            controller?.abort(); controller = null;
            request += 1;
            preview = null;
            list.hidden = true;
            list.innerHTML = '';
            apply.disabled = true;
            apply.textContent = 'Apply to draft';
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
            const identity = draftIdentity();
            setStatus('Checking the document…');
            try {
                controller = new AbortController();
                const data = await previewMcpImport({ text: text.value, servers: identity }, controller.signal);
                if (settled || ticket !== request) return;
                if (!data?.ok) {
                    setStatus(data?.error || 'The document could not be read.', 'danger');
                    return;
                }
                preview = { ...data, fingerprint: JSON.stringify(identity) };
                list.innerHTML = (data.entries || []).map(renderEntry).join('');
                list.hidden = !(data.entries || []).length;
                const count = applicable().length;
                const blocked = (data.entries || []).length - count;
                const ignored = (data.ignored_keys || []).length ? ` Other top-level keys are not imported: ${data.ignored_keys.join(', ')}.` : '';
                setStatus(!(data.entries || []).length ? `The mcpServers object is empty.${ignored}`
                    : `${count} of ${data.entries.length} server${data.entries.length === 1 ? '' : 's'} can be applied`
                        + `${blocked ? `; ${blocked} need${blocked === 1 ? 's' : ''} changes first` : ''}.${ignored}`,
                count && !blocked ? 'ok' : 'warn');
                apply.disabled = !count;
                if (count) apply.textContent = `Apply ${count} to draft`;
            } catch (error) {
                if (settled || ticket !== request) return;
                setStatus(`Preview failed: ${error?.message || error}`, 'danger');
            }
        }
        text.addEventListener('input', invalidate);
        q('[data-mcp-import-preview-btn]').addEventListener('click', runPreview);
        apply.addEventListener('click', () => {
            if (!preview) return;
            if (JSON.stringify(draftIdentity()) !== preview.fingerprint) {
                invalidate();
                setStatus('The MCP draft changed after this preview. Preview again before applying.', 'warn');
                return;
            }
            const entries = applicable();
            onApply(entries);
            finish(entries.length);
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
