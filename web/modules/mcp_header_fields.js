/** Literal HTTP header rows: draft arrays preserve duplicate names until validation. */
import { escapeHtmlAttr as escapeHtml } from './utils.js';
import { bindSecretReveal } from './settings_secrets.js';
import { revealNewRow } from './ui_helpers.js';

export const HEADER_PLACEHOLDER = '***set***';
const rowsByServer = new WeakMap();
const NAME = /^[!#$%&'*+\-.^_`|~0-9A-Za-z]+$/;

export function resetHeaderRows(server) { rowsByServer.delete(server); }
function rowsFor(server) {
    if (!rowsByServer.has(server)) {
        const validMap = server.headers && typeof server.headers === 'object' && !Array.isArray(server.headers);
        rowsByServer.set(server, { touched: false, rows: validMap ? Object.entries(server.headers).map(([name, value]) =>
            ({ name, value, savedName: name, savedValue: value })) : [] });
    }
    return rowsByServer.get(server);
}
export function headerPayload(server) {
    const state = rowsFor(server);
    if (!state.touched) return Object.hasOwn(server, 'headers') ? { headers: server.headers } : {};
    return { headers: Object.fromEntries(state.rows.map(({name, value}) => [name, value])) };
}
export function headerHasMask(server) {
    const headers = headerPayload(server).headers;
    return headers === HEADER_PLACEHOLDER || (headers && typeof headers === 'object'
        && Object.values(headers).includes(HEADER_PLACEHOLDER));
}
export function renderHeaderFields(server, index) {
    const state = rowsFor(server);
    const malformed = Object.hasOwn(server, 'headers') && !state.touched && (server.headers !== null
        && (typeof server.headers !== 'object' || Array.isArray(server.headers)));
    const rows = state.rows.map((row, number) => `
        <div class="mcp-header-row" data-mcp-header-row="${number}">
            <div class="form-field ui-field">
                <label for="mcp-${index}-header-${number}-name">Header name</label>
                <input class="ui-control" type="text" id="mcp-${index}-header-${number}-name" data-mcp-header-field="name"
                    value="${escapeHtml(row.name)}" placeholder="Authorization" autocomplete="off" spellcheck="false">
            </div>
            <div class="form-field ui-field">
                <label for="mcp-${index}-header-${number}-value">Header value</label>
                <div class="secret-input-row">
                    <input class="ui-control" type="password" id="mcp-${index}-header-${number}-value" data-mcp-header-field="value"
                        value="${escapeHtml(typeof row.value === 'string' ? row.value : HEADER_PLACEHOLDER)}" autocomplete="off" spellcheck="false">
                    <button type="button" class="btn btn-default" data-mcp-header-toggle>Show</button>
                    <button type="button" class="btn btn-default" data-mcp-header-clear>Clear</button>
                </div>
            </div>
            <button type="button" class="btn btn-default" data-mcp-header-remove aria-label="Remove header ${number + 1}">Remove</button>
        </div>`).join('');
    return `<section class="mcp-headers-fields">
        <div class="settings-section-head"><h4>HTTP headers</h4><button type="button" class="btn btn-default" data-mcp-header-add ${malformed ? 'disabled' : ''}>Add header</button></div>
        <p class="ui-field-help">Full literal values: for example Authorization with Bearer &lt;token&gt;, Basic &lt;value&gt;, or X-API-Key.
            No scheme is added. Every value is protected. Clear leaves an empty value (not sent); Remove deletes the header.</p>
        ${malformed ? '<p class="ui-field-help">Saved headers are malformed. They are retained until you replace them. Remove all headers before adding new ones.</p><button type="button" class="btn btn-default" data-mcp-header-reset>Remove all headers</button>' : ''}
        ${rows}
    </section>`;
}
export function bindHeaderFields(card, {server, savedIdentity, onChange, render}) {
    const state = rowsFor(server);
    const changed = () => { state.touched = true; onChange(); };
    card.querySelectorAll('[data-mcp-header-row]').forEach((node) => {
        const number = Number(node.dataset.mcpHeaderRow);
        const row = state.rows[number];
        const name = node.querySelector('[data-mcp-header-field="name"]');
        const value = node.querySelector('[data-mcp-header-field="value"]');
        name.addEventListener('input', () => { row.name = name.value; changed(); });
        value.addEventListener('input', () => { row.value = value.value; changed(); });
        value.dataset.appliedValue = typeof row.savedValue === 'string' && row.savedValue === HEADER_PLACEHOLDER ? row.savedValue : '';
        const reveal = bindSecretReveal(value, node.querySelector('[data-mcp-header-toggle]'), {
            savedSelector: () => savedIdentity && row.savedName ? {mcp_server_id: savedIdentity, header_name: row.savedName} : null,
            savedLabel: () => name.value !== row.savedName ? `Saved value for ${row.savedName}` : '',
            identityInputs: [name, ...['id','name'].map((field) => card.querySelector(`[data-mcp-field="${field}"]`)).filter(Boolean)],
        });
        node.querySelector('[data-mcp-header-clear]').addEventListener('click', () => {
            reveal.reset(); row.value = value.value = ''; changed();
        });
        node.querySelector('[data-mcp-header-remove]').addEventListener('click', () => {
            state.rows.splice(number, 1); changed(); render();
        });
    });
    card.querySelector('[data-mcp-header-reset]')?.addEventListener('click', () => {
        state.rows = []; changed(); render();
    });
    card.querySelector('[data-mcp-header-add]')?.addEventListener('click', () => {
        state.rows.push({name: '', value: '', savedName: '', savedValue: ''}); changed(); render();
        const host = document.querySelector(`[data-mcp-index="${card.dataset.mcpIndex}"]`);
        const row = host?.querySelector(`[data-mcp-header-row="${state.rows.length - 1}"]`);
        revealNewRow(row, row?.querySelector('[data-mcp-header-field="name"]'));
    });
}
/** Same structural errors as the backend; returns fields for Settings' existing validation painter. */
export function validateHeaderFields(card, server) {
    if (server.transport === 'stdio') return [];
    const state = rowsFor(server), errors = [], seen = new Map();
    // An unrelated Save must preserve even an unreadable old header map; the
    // backend judges newly changed configuration after exact-mask restoration.
    if (!state.touched) return [];
    state.rows.forEach((row, number) => {
        const node = card.querySelector(`[data-mcp-header-row="${number}"]`);
        const nameInput = node?.querySelector('[data-mcp-header-field="name"]');
        const valueInput = node?.querySelector('[data-mcp-header-field="value"]');
        const name = String(row.name), value = row.value;
        let error = !NAME.test(name) ? 'Enter one HTTP header name (letters, digits or token punctuation).' : '';
        const folded = name.toLowerCase();
        if (!error && seen.has(folded)) error = `This name repeats header ${seen.get(folded)}; keep one.`;
        seen.set(folded, name);
        if (error) errors.push({input: nameInput, message: error});
        if (typeof value !== 'string' || /[^\x20-\x7e]/.test(value) || value !== value.trim()) {
            errors.push({input: valueInput, message: 'Enter a printable ASCII value without control characters or outer whitespace. It is sent literally.'});
        }
        if (value && server.auth_token && folded === String(server.auth_header || 'Authorization').toLowerCase()) {
            errors.push({input: valueInput, message: 'This header is also set by the legacy header value. Clear the legacy value or remove this header.'});
        }
    });
    return errors.filter(({input}) => input);
}
