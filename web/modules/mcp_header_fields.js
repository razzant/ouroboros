/**
 * One header list per HTTP/SSE server. A saved legacy auth_header/auth_token
 * pair is shown as an ordinary row; its provenance stays in this module's
 * transient row state, so an untouched row saves back to the legacy fields with
 * their own mask and an unrelated Save rewrites nothing. A legacy row whose name
 * or value is deliberately edited becomes a header-map entry and the legacy
 * fields are cleared. Draft arrays keep duplicate names until validation.
 *
 * A masked value is kept only while its row keeps the saved header name
 * (case aside) and its server keeps its saved canonical identity. ***set*** is
 * the Settings "keep" marker: typed as a value it is a field error, never a literal.
 */
import { escapeHtmlAttr as escapeHtml } from './utils.js';
import { bindSecretReveal } from './settings_secrets.js';
import { revealNewRow } from './ui_helpers.js';

export const HEADER_PLACEHOLDER = '***set***';
const rowsByServer = new WeakMap();
const NAME = /^[!#$%&'*+\-.^_`|~0-9A-Za-z]+$/;

const isMap = (value) => Boolean(value) && typeof value === 'object' && !Array.isArray(value);
const folded = (name) => String(name ?? '').toLowerCase();
/** The loader's identity expression (id, then slug, then name), before canonicalization. */
export const serverIdentity = (server) => String(server?.id || server?.slug || server?.name || '');

// Mirror of ``mcp_client.canonical_server_id``, the backend's identity authority
// (parity table: web/tests/fixtures/mcp_server_identity_parity.json).
const SERVER_SLUG_MAX = 24;
const PY_SPACE = '\\t\\n\\v\\f\\r\\x1c-\\x20\\x85\\xa0\\u1680\\u2000-\\u200a\\u2028\\u2029\\u202f\\u205f\\u3000';
const PY_STRIP = new RegExp(`^[${PY_SPACE}]+|[${PY_SPACE}]+$`, 'g');

/** SHA-1 hex of UTF-8 text (Python's ``errors="replace"`` for lone surrogates); only long identities reach it. */
function sha1Hex(text) {
    const data = new TextEncoder().encode(text.replace(/[\ud800-\udbff](?![\udc00-\udfff])|(?<![\ud800-\udbff])[\udc00-\udfff]/g, '?'));
    const padded = new Uint8Array(((data.length + 72) >> 6) << 6);
    padded.set(data);
    padded[data.length] = 0x80;
    const view = new DataView(padded.buffer);
    view.setUint32(padded.length - 8, Math.floor(data.length / 0x20000000));
    view.setUint32(padded.length - 4, (data.length * 8) >>> 0);
    const h = [0x67452301, 0xefcdab89, 0x98badcfe, 0x10325476, 0xc3d2e1f0], w = new Uint32Array(80);
    for (let offset = 0; offset < padded.length; offset += 64) {
        for (let i = 0; i < 80; i += 1) {
            const x = i < 16 ? view.getUint32(offset + i * 4) : w[i - 3] ^ w[i - 8] ^ w[i - 14] ^ w[i - 16];
            w[i] = i < 16 ? x : (x << 1) | (x >>> 31);
        }
        let [a, b, c, d, e] = h;
        for (let i = 0; i < 80; i += 1) {
            const f = i < 20 ? (b & c) | (~b & d) : i < 40 || i >= 60 ? b ^ c ^ d : (b & c) | (b & d) | (c & d);
            const k = [0x5a827999, 0x6ed9eba1, 0x8f1bbcdc, 0xca62c1d6][Math.floor(i / 20)];
            const t = (((a << 5) | (a >>> 27)) + f + e + k + w[i]) >>> 0;
            [e, d, c, b, a] = [d, c, ((b << 30) | (b >>> 2)) >>> 0, a, t];
        }
        [a, b, c, d, e].forEach((value, index) => { h[index] = (h[index] + value) >>> 0; });
    }
    return h.map((value) => value.toString(16).padStart(8, '0')).join('');
}

function slugify(value) {
    const text = String(value || '').replace(PY_STRIP, '');
    const safe = text.replace(/[^A-Za-z0-9_]/g, '_').replace(/_+/g, '_').replace(/^_|_$/g, '').toLowerCase();
    if (!safe || safe.length <= SERVER_SLUG_MAX) return safe;
    const digest = sha1Hex(text).slice(0, 12);
    return `${safe.slice(0, SERVER_SLUG_MAX - digest.length - 1)}_${digest}`;
}

/** The canonical server ID: two slug passes, exactly as the backend computes it. */
export const canonicalServerId = (value) => slugify(slugify(value));
const sameServer = (server, savedIdentity) => canonicalServerId(serverIdentity(server)) === canonicalServerId(savedIdentity);

export function resetHeaderRows(server) { rowsByServer.delete(server); }

function rowsFor(server) {
    if (!rowsByServer.has(server)) {
        const rows = [];
        const token = String(server.auth_token ?? '');
        if (token) {
            const name = String(server.auth_header || 'Authorization');
            rows.push({ source: 'legacy', name, value: token, savedName: name, savedValue: token, valueEdited: false });
        }
        const unreadable = Object.hasOwn(server, 'headers') && server.headers !== null && !isMap(server.headers);
        if (isMap(server.headers)) {
            for (const [name, value] of Object.entries(server.headers)) {
                const text = typeof value === 'string' ? value : HEADER_PLACEHOLDER;
                rows.push({ source: 'map', name, value: text, savedName: name, savedValue: text, valueEdited: false });
            }
        }
        rowsByServer.set(server, { touched: false, unreadable, rows });
    }
    return rowsByServer.get(server);
}

/** Whether this row still asks Save/Test to keep its saved value. */
function keepsSavedValue(row) {
    return row.source !== 'new' && !row.valueEdited && row.value === row.savedValue
        && (row.source === 'legacy' || row.savedValue === HEADER_PLACEHOLDER);
}
function untouchedLegacy(row) {
    return row.source === 'legacy' && keepsSavedValue(row) && folded(row.name) === folded(row.savedName);
}

/** Why a kept saved value cannot travel with this row; '' when it can. */
function keptValueProblem(row, server, savedIdentity) {
    if (!keepsSavedValue(row)) return '';
    if (folded(row.name) !== folded(row.savedName)) {
        return `The saved value belongs to header ${row.savedName}. Restore that name, or enter a new value.`;
    }
    if (savedIdentity && !sameServer(server, savedIdentity)) {
        return `The saved value belongs to server ${savedIdentity}. Restore its Server ID, or enter the value again.`;
    }
    return '';
}

/**
 * Fields this server's header rows decide. Untouched rows send the loaded
 * fields unchanged; touched rows split back into the legacy pair (an
 * untouched legacy row only) and the literal headers map.
 */
export function credentialPayload(server) {
    const state = rowsFor(server);
    if (!state.touched) return {};
    if (state.unreadable) {
        // An unreadable saved map is kept whole until it is removed; a legacy
        // row then stays in its own fields even when edited.
        const row = state.rows.find((item) => item.source === 'legacy');
        return { auth_header: row ? row.name : String(server.auth_header || 'Authorization'),
            auth_token: row ? row.value : '', headers: server.headers };
    }
    const legacy = state.rows.find((row) => untouchedLegacy(row));
    const payload = legacy ? { auth_header: legacy.name, auth_token: legacy.value }
        : { auth_header: 'Authorization', auth_token: '' };
    const mapRows = state.rows.filter((row) => row !== legacy);
    if (mapRows.length || Object.hasOwn(server, 'headers')) {
        payload.headers = Object.fromEntries(mapRows.map(({ name, value }) => [name, value]));
    }
    return payload;
}

/** Whether Test must name the saved server so the backend can restore kept values. */
export function headerKeepsSavedValues(server) {
    const state = rowsFor(server);
    return state.unreadable || state.rows.some((row) => keepsSavedValue(row));
}

/** Names whose saved values this draft keeps (for endpoint-change disclosure). */
export function keptHeaderNames(server) {
    return rowsFor(server).rows.filter((row) => keepsSavedValue(row)).map((row) => row.name);
}

/**
 * Raw rows for dirty tracking and import freshness: exact duplicates are not
 * collapsed, and an edited value differs from the untouched mask it may spell.
 */
export function headerDraftRows(server) {
    const state = rowsFor(server);
    return { unreadable: state.unreadable, rows: state.rows.map((row) => [row.source, row.name, row.value, row.valueEdited]) };
}

/** Matching metadata for an import preview: names and provenance, never values. */
export function headerImportFacts(server) {
    const state = rowsFor(server);
    const legacy = state.rows.find((row) => row.source === 'legacy');
    return {
        header_names: state.rows.map((row) => row.name),
        legacy_header: legacy ? legacy.name : null,
        ...(state.unreadable ? { unreadable_headers: true } : {}),
    };
}

export function renderHeaderFields(server, index) {
    const state = rowsFor(server);
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
                        value="${escapeHtml(row.value)}" autocomplete="off" spellcheck="false">
                    <button type="button" class="btn btn-default" data-mcp-header-toggle>Show</button>
                    <button type="button" class="btn btn-default" data-mcp-header-clear>Clear</button>
                </div>
            </div>
            <button type="button" class="btn btn-default" data-mcp-header-remove aria-label="Remove header ${number + 1}">Remove</button>
        </div>`).join('');
    return `<section class="mcp-headers-fields">
        <div class="settings-section-head"><h4>HTTP headers</h4><button type="button" class="btn btn-default" data-mcp-header-add ${state.unreadable ? 'disabled' : ''}>Add header</button></div>
        <p class="ui-field-help">Each value is sent exactly as written, for example Authorization with Bearer &lt;token&gt; or Basic &lt;value&gt;, or X-API-Key.
            No scheme is added. Values are masked here, not scrubbed from server replies. Clear keeps an empty value, which is not sent; Remove deletes it.
            ${server.transport === 'sse'
        ? 'SSE sets Accept and Cache-Control on its event stream; message requests send every row as written, so a Content-Type row replaces application/json and the server may refuse it.'
        : 'Streamable HTTP sets Accept and Content-Type on every request, and its session headers once known; those replace a row with the same name.'}</p>
        ${state.unreadable ? '<p class="ui-field-help">Saved headers are malformed. They are retained until you remove them; remove them before adding new headers.</p><button type="button" class="btn btn-default" data-mcp-header-reset>Remove malformed headers</button>' : ''}
        ${rows}
    </section>`;
}

export function bindHeaderFields(card, { server, savedIdentity, onChange, render }) {
    const state = rowsFor(server);
    const changed = () => { state.touched = true; onChange(); };
    card.querySelectorAll('[data-mcp-header-row]').forEach((node) => {
        const number = Number(node.dataset.mcpHeaderRow);
        const row = state.rows[number];
        const name = node.querySelector('[data-mcp-header-field="name"]');
        const value = node.querySelector('[data-mcp-header-field="value"]');
        const markEdited = () => { row.valueEdited = true; value.dataset.appliedValue = ''; };
        name.addEventListener('input', () => { row.name = name.value; changed(); });
        value.addEventListener('input', () => { markEdited(); row.value = value.value; changed(); });
        value.dataset.appliedValue = keepsSavedValue(row) ? row.savedValue : '';
        const reveal = bindSecretReveal(value, node.querySelector('[data-mcp-header-toggle]'), {
            // Show reads only the identity the value was saved under and labels
            // it once the row or server is renamed; Save and Test then ask for
            // the value again instead of keeping or substituting it.
            savedSelector: () => !savedIdentity ? null : row.source === 'legacy' ? { mcp_server_id: savedIdentity }
                : { mcp_server_id: savedIdentity, header_name: row.savedName },
            savedLabel: () => {
                const header = folded(name.value) !== folded(row.savedName) ? row.savedName : '';
                const owner = savedIdentity && !sameServer(server, savedIdentity) ? savedIdentity : '';
                return header || owner ? `Saved value for ${[header, owner].filter(Boolean).join(' on ')}` : '';
            },
            identityInputs: [name, ...['id', 'name'].map((field) => card.querySelector(`[data-mcp-field="${field}"]`)).filter(Boolean)],
        });
        node.querySelector('[data-mcp-header-clear]').addEventListener('click', () => {
            reveal.reset(); markEdited(); row.value = value.value = ''; changed();
        });
        node.querySelector('[data-mcp-header-remove]').addEventListener('click', () => {
            state.rows.splice(number, 1); changed(); render();
            // The caret moves to the header now in this place, else to Add (DESIGN "List editors").
            const host = document.querySelector(`[data-mcp-index="${card.dataset.mcpIndex}"]`);
            const next = host?.querySelector(`[data-mcp-header-row="${Math.min(number, state.rows.length - 1)}"]`);
            if (next) revealNewRow(next, next.querySelector('[data-mcp-header-field="name"]'));
            else host?.querySelector('[data-mcp-header-add]')?.focus();
        });
    });
    card.querySelector('[data-mcp-header-reset]')?.addEventListener('click', () => {
        state.rows = state.rows.filter((row) => row.source === 'legacy');
        state.unreadable = false;
        server.headers = {};
        changed(); render();
    });
    card.querySelector('[data-mcp-header-add]')?.addEventListener('click', () => {
        state.rows.push({ source: 'new', name: '', value: '', savedName: '', savedValue: '', valueEdited: false });
        changed(); render();
        const host = document.querySelector(`[data-mcp-index="${card.dataset.mcpIndex}"]`);
        const row = host?.querySelector(`[data-mcp-header-row="${state.rows.length - 1}"]`);
        revealNewRow(row, row?.querySelector('[data-mcp-header-field="name"]'));
    });
}

/** Same structural errors as the backend; returns fields for Settings' existing validation painter. */
export function validateHeaderFields(card, server, savedIdentity = '') {
    if (server.transport === 'stdio') return [];
    const state = rowsFor(server), errors = [], seen = new Map();
    // An unrelated Save must preserve even an unreadable old header map; the
    // backend judges newly changed configuration after exact-mask restoration.
    const renamed = Boolean(savedIdentity) && !sameServer(server, savedIdentity);
    if (!state.touched && !renamed) return [];
    state.rows.forEach((row, number) => {
        const node = card.querySelector(`[data-mcp-header-row="${number}"]`);
        const nameInput = node?.querySelector('[data-mcp-header-field="name"]');
        const valueInput = node?.querySelector('[data-mcp-header-field="value"]');
        const name = String(row.name), key = folded(name);
        let error = !NAME.test(name) ? 'Enter one HTTP header name (letters, digits or token punctuation).' : '';
        if (!error && seen.has(key)) error = `This name repeats header ${seen.get(key)}; keep one.`;
        seen.set(key, name);
        if (error) errors.push({ input: nameInput, message: error });
        const kept = keptValueProblem(row, server, savedIdentity);
        if (kept) {
            errors.push({ input: valueInput, message: kept });
        } else if (keepsSavedValue(row)) {
            // An untouched saved value keeps its own admission rules, legacy prefix masks included.
        } else if (savedIdentity && row.source === 'legacy' && row.valueEdited && row.value === row.savedValue) {
            errors.push({ input: valueInput, message: "This is the saved token's mask. Enter the real value or reload Settings." });
        } else if (row.value === HEADER_PLACEHOLDER) {
            errors.push({ input: valueInput, message: `${HEADER_PLACEHOLDER} is reserved for keeping a saved value. Enter the real value.` });
        } else if (typeof row.value !== 'string' || /[^\x20-\x7e]/.test(row.value) || row.value !== row.value.trim()) {
            errors.push({ input: valueInput, message: 'Enter a printable ASCII value without control characters or outer whitespace. It is sent literally.' });
        }
    });
    return errors.filter(({ input }) => input);
}
