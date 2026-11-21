/** Extensions › MCP servers (docs/DESIGN.md "Settings"; a page of the registry Extensions opens, extensions.js). The
 *  list (`tools`, its old Settings address kept): the tool client's switch, one head per tool
 *  server with its state, its Sign in when the status asks and its switch, Add, and the fold. A server's page
 *  (`tools/<id>`, `tools/new`): its form with Test, its own Save and Remove. A head's switch and a form's Save each
 *  write one entry of MCP_SERVERS from a fresh read (settings_controls.js writeEntry); a list that changed meanwhile
 *  is refused and shown again. A saved credential stays masked and kept until the owner edits it. The status is read
 *  when a page is shown; only Test, Refresh and Sign in start anything. */
import { fetchJson, jsonPost } from './api_client.js';
import { absoluteTime, escapeHtmlAttr as escapeHtml, safeExternalUrl } from './utils.js';
import { formatRelativeAge, openExternalViaHostBridge, revealNewRow } from './ui_helpers.js';
import { resetSecretReveals } from './settings_secrets.js';
import {
    renderHeaderFields, bindHeaderFields, credentialPayload, headerKeepsSavedValues, keptHeaderNames,
    headerDraftRows, headerImportFacts, headerRevision, headerSubmission, resetHeaderRows, settleHeaderRows, serverIdentity,
    validateHeaderFields, canonicalServerId, HEADER_PLACEHOLDER,
} from './mcp_header_fields.js';
import { endpointChange, openMcpImportDialog } from './mcp_import_dialog.js';
import { openConfirmDialog } from './confirm_dialog.js';
import { controlIcon } from './page_icons.js';
import { CODECS, CONFIRMS, bindSettingRows, commitSetting, doorSign, outcomeLine, setButtonBusy, settingsRow, writeEntry } from './settings_controls.js';
import { registerSettingsPage } from './settings_pages.js';
import { rowsOf } from './settings_rows.js';
import { setResult } from './ui_primitives.js';

const KEY = 'MCP_SERVERS';
const TRANSPORTS = [{ value: 'streamable_http', label: 'Streamable HTTP' }, { value: 'sse', label: 'SSE (Server-Sent Events)' },
    { value: 'stdio', label: 'Local process (stdio)' }];
// task_session is an owner-selected stdio call lifetime; browser_policy stays an Import-set policy; a skill-bound
// row's skill sets both (shown read-only). Off stdio they are unsupported fields that Save refuses until removed.
const LIFETIME_FLAGS = ['task_session', 'browser_policy'];
// A remote server's sign-in (`status.sign_in`, ouroboros/mcp_oauth.py): why it is needed, in words.
const SIGN_IN_REASONS = { not_signed_in: 'not signed in yet', expired: 'the sign-in expired', signed_out: 'signed out',
    refresh_rejected: 'the provider no longer accepts the sign-in', token_rejected: 'the provider no longer accepts the sign-in' };
// `skill`: written by a skill's Connect tools and kept as saved; such a stdio row runs that skill's declared server,
// whose command, arguments and folder the host derives each time it starts (no fields of their own here), so its
// transport is locked; on any other transport the loader refuses the row, and Save does too.
const SERVER_FIELDS = new Set(['id', 'slug', 'name', 'label', 'enabled', 'transport', 'url', 'command', 'args', 'auth_header',
    'auth_token', 'headers', 'allowed_tools', 'cwd', 'env', 'env_from_settings', 'schemas_at_start', 'skill', ...LIFETIME_FLAGS]);

let mcpServers = [];        // the saved servers in the document's order; each object is also the draft of its form
let draftServer = null;     // the server of tools/new, until its first Save
let loaded = false;
let reloadSerial = 0;
let savedGeneration = 0;
let [mcpStatusByServer, mcpStatusEnvelope] = [{}, undefined];
const STATUS_UNAVAILABLE = 'MCP status unavailable';
let listView = null;        // the list on screen: { el, ctx, heads }
let editorView = null;      // the form on screen: { host, path, entry, server, ctx }
// The stored identity used by writeEntry and kept secrets, the saved address and switch, and the entry as read.
const savedMcpSecrets = new WeakMap();
// A result follows the server object: a pending Refresh or Test lands on its own form or nowhere, and an edit retires
// a Test result that no longer describes the draft.
const cardMessages = new WeakMap();
const [draftGenerations, pendingActions] = [new WeakMap(), new WeakMap()];
const signInLinks = new WeakMap(); // the provider page of the sign-in this window began, per server
const signInBusy = new WeakSet(); // a sign-in step in flight: a second press waits for it
const dirty = new WeakSet();      // typed into and not saved: a read keeps such a form
const saving = new WeakMap();     // one Save (writing/reading) or Discard (discarding) per form, across mounts
// Saved canonical id → a typed draft the read did not confirm: one whose Save landed but whose entry was not read back as
// sent, or one whose entry changed elsewhere. It stays the page's form, and Save refuses, until the owner discards it.
const heldDrafts = new Map();
// A stale or unanswered write keeps its draft until Discard. A new form never claims a stored entry's ID.
const staleDrafts = new WeakMap();
const DISCARD_FAILED = 'The current settings could not be read. Press Discard changes again to load them.';
const READBACK_CONFLICT = 'Earlier changes saved. The current server settings could not be confirmed. Your latest edits are unsaved. Discard changes to load the current settings.';
const CHANGED_CONFLICT = 'Not saved: the server list changed elsewhere. Your edits are kept here, unsaved; discard them to start again from the current list.';
const UNCONFIRMED_CONFLICT = 'Not confirmed: this Save may have been stored. Your edits are kept here; discard them to load what the server holds now.';
const conflictWords = (server) => staleDrafts.get(server)
    || (savedMcpSecrets.get(server)?.submitted ? READBACK_CONFLICT : CHANGED_CONFLICT);
const moreOpen = new WeakSet();   // the form's More fold, as the owner left it
const wanted = new WeakMap();     // a switch in flight -> the value pressed, shown until the answer comes
const GONE = new Error('changed elsewhere');
const CANCELLED = new Error('save cancelled');
const NAMES = new Map();          // page id -> name, for the page's title

// The loader includes rejected configurations in its status, with tool_count: 0 just like an empty live catalog.
const rejectedConfig = (server) => server?.code === 'MCP_CONFIG_ERROR' || server?.code === 'MCP_ID_AMBIGUOUS';
export const mcpServerOn = (server) => server?.enabled === true && Object.hasOwn(server, 'tool_count') && !rejectedConfig(server);
export const configuredServerCount = (servers) => servers.reduce((n, server) => n + (server?.configured_count ?? 1), 0);
export const mcpNeedsSignIn = (server) => mcpServerOn(server) && Boolean(server.sign_in)
    && (server.sign_in.state !== 'signed_in' || server.sign_in.reason === 'insufficient_scope');

function looksMasked(value) {
    const text = String(value ?? '').trim();
    return text === '***' || (Boolean(text) && text.endsWith('...'));
}

const emptyServer = () => ({ id: '', name: '', enabled: false, transport: 'streamable_http', url: '', command: '', args: [],
    auth_header: 'Authorization', auth_token: '', allowed_tools: [] });

const isLive = (server) => savedMcpSecrets.has(server);
const serverName = (server) => String(server?.name || server?.id || server?.slug || 'MCP server');
/** The id of a server's page: its canonical identity (id, slug or name), as the status keys it. */
const pageId = (server) => canonicalServerId(isLive(server) ? savedMcpSecrets.get(server).identity : serverIdentity(server));
// The entry as a read gave it, without the switch (a head writes that at once, beside an open form).
const asRead = ({ auth_configured: _meta, enabled: _enabled, ...entry }) => JSON.stringify(entry, (_, value) =>
    isServer(value) ? Object.fromEntries(Object.keys(value).sort().map(key => [key, value[key]])) : value);
const savedEntry = (entry, token) => ({ identity: serverIdentity(entry), url: String(entry.url || ''), enabled: Boolean(entry.enabled), read: asRead(entry), token });
const heldEditor = server => staleDrafts.has(server) || heldDrafts.get(pageId(server)) === server;

const cardFor = (server) => (editorView?.server === server ? editorView.host.querySelector('[data-mcp-card]') : null);

// The form's one result line, beside Save: a Save, a Test, a Refresh or a sign-in step says its outcome there.
function setCardMessage(server, text, tone = 'muted', kind = '') {
    if (text) cardMessages.set(server, { text, tone, kind });
    else cardMessages.delete(server);
    setResult(cardFor(server)?.querySelector('[data-mcp-message]'), text ? { text, tone } : {});
}

function holdDraft(server, reason) {
    dirty.add(server);
    staleDrafts.set(server, reason);
    if (isLive(server)) {
        heldDrafts.set(pageId(server), server);
        cardFor(server)?.querySelector('[data-editor-discard]')?.removeAttribute('hidden');
    }
    setCardMessage(server, reason, 'warn', 'conflict');
    paintEditorActions(editorView);
}

/** One server's draft changed: a finished Test no longer describes it, and the form now holds unsaved text. */
function serverEdited(server) {
    draftGenerations.set(server, (draftGenerations.get(server) || 0) + 1);
    if (['test', 'save', 'discard'].includes(cardMessages.get(server)?.kind)) {
        if (heldEditor(server)) setCardMessage(server, conflictWords(server), 'warn', 'conflict');
        else setCardMessage(server, '');
    }
    paintEndpointNotice(server);
    dirty.add(server);
    if (isLive(server)) cardFor(server)?.querySelector('[data-editor-discard]')?.removeAttribute('hidden');
}

function setPending(server, action, busy) {
    const actions = pendingActions.get(server) || new Set();
    if (busy) actions.add(action); else actions.delete(action);
    pendingActions.set(server, actions);
    setButtonBusy(cardFor(server)?.querySelector(`[data-mcp-${action}]`), busy);
    paintEditorActions(editorView);
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
    if (el) Object.assign(el, { textContent: endpointNotice(server), hidden: !endpointNotice(server) });
}

const toolCountLabel = (count) => `${count} tool${count === 1 ? '' : 's'}`;
const tokensLabel = (count) => (Number(count || 0) < 1000 ? `≈${Number(count || 0)} tokens` : `≈${(Number(count || 0) / 1000).toFixed(1)}k tokens`);
const isPinned = (server) => [true, 'true', 'True'].includes(server.schemas_at_start);

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

/** The call lifetime a row runs with. A skill-bound row's is its skill's: each bind reads the reviewed manifest (the
 *  server's status reports it), and the copy Connect tools saved stands in until that status is read. */
function lifetimeOf(server) {
    const live = server.skill ? statusOf(server) : null;
    const source = live && live.skill === server.skill ? live : server;
    return { taskSession: lifetimeFlagOn(source.task_session), browserPolicy: lifetimeFlagOn(source.browser_policy) };
}

/** The stdio draft's lifetime and its browser policy, without changing either: Import sets the policy, a skill both. */
function lifetimeNote(server) {
    if (server.transport !== 'stdio') return '';
    const { taskSession, browserPolicy } = lifetimeOf(server);
    return (taskSession
        ? 'One server process serves every call in a task. It ends with the task, a Settings change or a call timeout.'
        : 'Each tool call starts a new server process.')
        + ' Test and Refresh still start separate processes.'
        + (server.skill ? `${browserPolicy ? ' That process may ask Ouroboros’s browser rules about its requests.' : ''}`
            + ` The skill ${server.skill} sets this lifetime; each start follows its reviewed files.`
            : browserPolicy ? ' browser_policy: that process may ask Ouroboros’s browser rules about its requests. '
                + 'It requires a task session; use Import to change that policy explicitly before switching this off.' : '');
}

/** The lifetime switch and its note as the row now runs (a bound row's status can arrive after the card). */
function paintLifetime(card, server) {
    const input = card.querySelector('[data-mcp-field="task_session"]');
    if (input) input.checked = lifetimeOf(server).taskSession;
    const note = card.querySelector('[data-mcp-lifetime]');
    if (note) note.textContent = lifetimeNote(server);
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
        : ['command', 'args', 'cwd', 'env', 'env_from_settings', 'skill'];
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

/** A saved server's status in parts: its state (one typed state with its tone, relative time), its facts (the exact
 *  stamps and the engine's text), and its discovered tools' list items. */
function statusParts(server) {
    const status = statusOf(server);
    const toolCount = status ? Number(status.tool_count || 0) : 0;
    const lastError = status ? String(status.skill_error || status.last_error || '') : '';
    const listedAt = status ? String(status.listed_at || '') : '';
    const attempt = status?.latest_attempt;
    const tools = status && Array.isArray(status.tools) ? status.tools : [];
    const parts = [listedAt ? `${toolCountLabel(toolCount)} · listed ${age(listedAt)}` : ''];
    if (status?.skill_error) parts.push('its skill can’t run it now');  // why in the Error fact; nothing starts
    else if (attempt?.ok === false) parts.push(`listing failed ${age(attempt.at)}`);
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
    const toolItems = tools.map((t) => `<li><strong>${escapeHtml(t.name || t.prefixed_name || '')}</strong>
                ${t.description ? `<span class="mcp-tool-desc">${escapeHtml(String(t.description).slice(0, 220))}</span>` : ''}</li>`).join('');
    return { badgeText: label.charAt(0).toUpperCase() + label.slice(1), tone, factItems, toolItems };
}

/** The status row of a saved server: the backend keys it by the canonical identity (id, slug or name). */
function statusOf(server) {
    return mcpStatusByServer[canonicalServerId(serverIdentity(server))] || mcpStatusByServer[String(server.id ?? '')] || null;
}

function signInKey(server) {
    const signIn = statusOf(server)?.sign_in;
    return signIn ? `${signIn.state}:${signIn.reason || ''}:${signIn.required_scope || ''}:${signInLinks.get(server) || ''}` : '';
}

/** A sign-in the status asks for: the word of its button, or '' when none is needed. */
function signInNeed(server) {
    const signIn = statusOf(server)?.sign_in;
    if (!signIn) return '';
    return signIn.state !== 'signed_in' ? 'Sign in' : signIn.reason === 'insufficient_scope' ? 'Sign in again' : '';
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

/** One step of a server's sign-in. `card` is its form (the paste and the app registration live there); a head's
 *  Sign in begins without one. `say` writes the step's words where the press was made. */
async function signInAction(card, server, action, say = (text, tone) => setCardMessage(server, text, tone, 'sign-in')) {
    const sid = savedMcpSecrets.get(server)?.identity || '';
    if (!sid) { say('Save the server before signing in.', 'warn'); return; }
    const field = (name) => String(card?.querySelector(`[data-mcp-sign-in-${name}]`)?.value || '').trim();
    const body = action === 'complete' ? { action, returned_url: field('returned') } : { action, server_id: sid };
    if (action === 'begin' && field('client-id')) Object.assign(body, { client_id: field('client-id'), client_secret: field('client-secret') });
    if (action === 'complete' && !body.returned_url) { say('Paste the address your browser stopped on first.', 'warn'); return; }
    if (signInBusy.has(server)) return;
    signInBusy.add(server);
    const buttons = [...(card?.querySelectorAll('[data-mcp-sign-in-action]') || [])];
    buttons.forEach((button) => { button.disabled = true; });
    say(action === 'sign_out' ? 'Signing out...' : 'Contacting the provider...', 'muted');
    try {
        const data = await jsonPost('/api/mcp/sign-in', body);
        if (data.ok === false) say(`Sign-in refused: ${data.error || data.reason || 'unknown reason'}`, 'danger');
        else if (action === 'begin') {
            signInLinks.set(server, data.url);
            say('Finish signing in with the provider. This page updates when you come back here.', 'muted');
            window.addEventListener('focus', () => { void refreshStatus(); }, { once: true });
            // Best effort while the press is recent; the link on the server's page works on every surface.
            void openExternalViaHostBridge(data.url).catch(() => {});
        } else if (action === 'complete') {
            signInLinks.delete(server);
            say(`Signed in. ${toolCountLabel(Number(data.listing?.tool_count || 0))} listed.`, 'ok');
        } else {
            say(data.revoked === true ? 'Signed out; the provider confirmed the revocation.'
                : data.revoked === false ? 'Signed out here; the provider did not confirm the revocation.'
                    : 'Signed out here; the provider offers no revocation.', 'ok');
        }
    } catch (err) { say(`Sign-in failed: ${err && err.message ? err.message : err}`, 'danger'); }
    signInBusy.delete(server);
    buttons.forEach((button) => { button.disabled = heldEditor(server); });
    await refreshStatus();
}

// A server's state in the status's own words; a warning or a failure takes its tone and dot.
const LOUD = ['danger', 'warn'];
function paintState(el, server) {
    if (!el) return;
    const { badgeText, tone } = statusParts(server);
    if (el.textContent !== badgeText) el.textContent = badgeText;
    el.classList.toggle('ui-status', LOUD.includes(tone));
    if (LOUD.includes(tone)) el.dataset.tone = tone; else delete el.dataset.tone;
}
function stateHtml(server, unavailable = false) {
    const { badgeText, tone } = unavailable ? { badgeText: STATUS_UNAVAILABLE, tone: 'warn' } : statusParts(server);
    return `<span${LOUD.includes(tone) ? ` class="ui-status" data-tone="${tone}"` : ''} data-mcp-state>${escapeHtml(badgeText)}</span>`;
}

const toolsListHtml = (toolItems) => (toolItems
    ? `<details class="mcp-tools-disclosure ui-disclosure"><summary>Discovered tools</summary><ul class="mcp-tools-list">${toolItems}</ul></details>` : '');

/**
 * A status answer arrives whenever its read finishes, often while the owner is typing or reading: the list's heads
 * and lead are drawn again only when their words change, and a form's status is painted in place, never over an
 * editable field or a disclosure's summary. Unchanged tools keep a text selection; field focus, caret and Show stay.
 */
function paintStatus() {
    paintList();
    paintEditorStatus();
}

function paintEditorStatus() {
    const server = editorView?.server;
    const card = server && editorShown(editorView) && cardFor(server);
    if (!card) return;
    const { factItems, toolItems } = statusParts(server);
    const weight = card.querySelector('[data-mcp-weight]');
    if (weight) { weight.textContent = weightLabel(server); weight.hidden = !weight.textContent; }
    const facts = card.querySelector('[data-mcp-facts]');
    if (facts && facts.innerHTML !== factItems) facts.innerHTML = factItems;
    paintState(card.querySelector('[data-mcp-state]'), server);
    if (server.skill) paintLifetime(card, server);
    paintUse();
    paintSignIn(card, server);
    paintEditorActions(editorView);
    const tools = card.querySelector('[data-mcp-tools]');
    const disclosure = tools?.querySelector('.mcp-tools-disclosure');
    if (!toolItems) disclosure?.remove();
    else if (!disclosure) tools?.insertAdjacentHTML('beforeend', toolsListHtml(toolItems));
    else {
        const list = disclosure.querySelector('.mcp-tools-list');
        const next = list.cloneNode(false);
        next.innerHTML = toolItems;
        if (!list.isEqualNode(next)) list.replaceChildren(...next.childNodes);
    }
}

const switchHtml = (attrs, on) => `<label class="ui-switch"><input type="checkbox" role="switch" aria-checked="${on}" ${attrs} ${on ? 'checked' : ''}>`
    + '<span class="ui-switch-track" aria-hidden="true"><span class="ui-switch-thumb"></span></span></label>';

/** One server's form (DESIGN "Settings", an entry's page): whether it is used, its connection with Save and Test and
 *  the rarer fields in a fold, then what it offers. A saved server's switch writes at once; a new one's is a field. */
function editorHtml(server) {
    const live = isLive(server);
    const index = live ? mcpServers.indexOf(server) : mcpServers.length;
    const transport = String(server.transport ?? 'streamable_http');
    const isStdio = transport === 'stdio';
    const skill = isStdio ? String(server.skill || '') : '';
    const unsupported = unsupportedFields(server);
    const json = (value) => (typeof value === 'string' ? value : JSON.stringify(value ?? {}, null, 2));
    const enabled = server.enabled === true || server.enabled === 'True' || server.enabled === 'true';
    const weight = weightLabel(server);
    const { factItems, toolItems } = statusParts(server);
    const message = cardMessages.get(server);
    const pending = pendingActions.get(server) || new Set();
    const notice = endpointNotice(server);
    const flags = lifetimeOf(server);
    // A skill-bound row is stdio by the loader's rule: its transport is shown, never offered as a choice.
    const transportOptions = TRANSPORTS.filter((opt) => !skill || opt.value === 'stdio').map((opt) =>
        `<option value="${escapeHtml(opt.value)}"${opt.value === transport ? ' selected' : ''}>${escapeHtml(opt.label)}</option>`).join('');
    const field = (name, label, control, help = '') => `<div class="form-field ui-field"><label for="mcp-${index}-${name}">${label}</label>${control}${help ? `<span class="ui-field-help">${help}</span>` : ''}</div>`;
    const input = (name, value, placeholder, id = name, attrs = '') => `<input type="text" class="ui-control" id="mcp-${index}-${id}" data-mcp-field="${name}" value="${escapeHtml(value)}" placeholder="${escapeHtml(placeholder)}" autocomplete="off" spellcheck="false"${attrs}>`;
    const area = (name, value, rows, placeholder) => `<textarea class="ui-control" id="mcp-${index}-${name}" data-mcp-field="${name}" rows="${rows}" placeholder="${escapeHtml(placeholder)}" autocomplete="off" spellcheck="false">${escapeHtml(value)}</textarea>`;
    const tick = (name, label, on, attrs = '') => `<label class="mcp-server-enabled"><input class="ui-checkbox" type="checkbox" data-mcp-field="${name}" ${on ? 'checked' : ''} ${attrs}><span>${label}</span></label>`;
    const use = live ? switchHtml(`data-entry-switch aria-label="Use this server"${wanted.has(server) ? ' disabled aria-busy="true"' : ''}`, wanted.has(server) ? wanted.get(server) : enabled)
        : switchHtml('data-mcp-field="enabled" aria-label="Use this server"', enabled);
    const stdioFields = () => `
        ${field('args', 'Arguments, one per line', area('args', (Array.isArray(server.args) ? server.args.map(String) : []).join('\n'), 3, '-y\n@modelcontextprotocol/server-filesystem\n/path/to/folder'),
            'Each line is passed as one argument. Ouroboros does not use a shell.')}
        ${field('cwd', 'Working folder', input('cwd', String(server.cwd ?? ''), '/path/to/project'), 'Used for both discovering and calling tools. Empty uses the default folder.')}
        ${field('env_from_settings', 'Environment from your saved keys (JSON)', area('env_from_settings', json(server.env_from_settings), 4, '{"API_TOKEN": "MY_MCP_KEY"}'),
            'Map environment names to saved setting keys. Keep a secret value as a custom key in Accounts and keys.')}
        ${field('env', 'Environment (JSON)', area('env', json(server.env), 3, '{"PORT": "8080", "DEBUG": "1"}'),
            'Ordinary values passed directly to the process. A saved key overrides the same name.')}`;
    return `
        <article class="settings-stack tools-editor" data-mcp-card data-mcp-index="${index}"${live ? ` data-entry-id="${escapeHtml(pageId(server))}"` : ''}>
            ${live ? `<p class="settings-page-lead">${stateHtml(server)}</p>` : ''}
            <section class="ui-card settings-card">
                <div class="settings-rows"><div class="ui-row settings-row" data-kind="switch">
                    <p class="ui-row-title">Use this server</p>
                    <p class="ui-row-meta"><span>${live ? 'Off keeps it here; its tools leave new requests.' : 'A new server starts off. Test it, then switch it on.'}</span></p>
                    <div class="ui-row-actions">${use}</div><span class="ui-result settings-row-result" data-entry-result role="status" aria-live="polite"></span>
                </div></div>
                <div class="mcp-sign-in" data-mcp-sign-in data-sign-in-key="${escapeHtml(signInKey(server))}" ${signInKey(server) ? '' : 'hidden'}>${signInRow(server)}</div>
            </section>
            <section class="ui-card settings-card">
                <h3 class="settings-card-title">Connection</h3>
                <div class="mcp-server-form">
                    <div class="form-grid two">
                        ${field('name', 'Name', input('name', String(server.name ?? ''), 'GitHub'))}
                        ${field('id', 'Server ID', input('id', String(server.id ?? ''), 'github', 'id', skill ? ' readonly' : ''))}
                    </div>
                    <div class="form-grid two">
                        ${field('transport', 'Transport', `<select class="ui-control" id="mcp-${index}-transport" data-mcp-field="transport"${skill ? ` disabled aria-describedby="mcp-${index}-endpoint"` : ''}>${transportOptions}</select>`)}
                        ${skill ? `<div class="form-field ui-field"><span>Runs</span><span class="ui-field-help" id="mcp-${index}-endpoint">The server the skill ${escapeHtml(skill)} declares. Its command, arguments and folder come from the installed skill each time it starts; while the skill can’t run, this server stays unavailable.</span></div>`
                            : isStdio ? field('endpoint', 'Command', input('command', String(server.command ?? ''), 'npx', 'endpoint'))
                            : `<div class="form-field ui-field"><label for="mcp-${index}-endpoint">Address</label>${input('url', String(server.url ?? ''), 'https://example.com/mcp', 'endpoint')}
                                <span class="ui-field-help" data-mcp-endpoint-notice ${notice ? '' : 'hidden'}>${escapeHtml(notice)}</span></div>`}
                    </div>
                </div>
                ${unsupported.length ? `<div class="ui-action-row"><span class="ui-field-help" data-mcp-unsupported>Fields kept but not used: ${escapeHtml(unsupported.join(', '))}</span>
                    <button type="button" class="btn btn-default btn-sm" data-mcp-clear-unsupported>Remove unsupported fields</button></div>` : ''}
                <div class="ui-action-row">
                    <button type="button" class="btn btn-save" data-editor-save ${saving.has(server) ? 'disabled' : ''}>Save</button>
                    <button type="button" class="btn btn-default" data-mcp-test ${pending.has('test') ? 'disabled' : ''}>Test</button>
                    <button type="button" class="btn btn-default" data-editor-discard${live && dirty.has(server) ? '' : ' hidden'}>Discard changes</button>
                    <span class="ui-result" data-editor-result data-mcp-message role="status" aria-live="polite"${message ? ` data-tone="${escapeHtml(message.tone)}"` : ''}>${escapeHtml(message?.text || '')}</span>
                </div>
                <details class="ui-disclosure" data-mcp-more${moreOpen.has(server) ? ' open' : ''}>
                    <summary>${skill ? 'More: tools, schemas' : isStdio ? 'More: arguments, folder, environment, tools' : 'More: headers, tools, schemas'}</summary>
                    <div class="mcp-server-form">
                        ${skill ? '' : isStdio ? stdioFields() : renderHeaderFields(server, index)}
                        ${field('allowed_tools', 'Allowed tools, comma-separated', input('allowed_tools', Array.isArray(server.allowed_tools) ? server.allowed_tools.join(', ') : '', 'search, read_repo'), 'Empty allows every tool the server lists.')}
                        <div>${tick('schemas_at_start', 'Schemas from task start', isPinned(server))}
                            <span class="ui-field-help">Off: Ouroboros sees this server’s tools in its connections map and loads their call schemas when it needs them. On: in Low and Max the call schemas are in every request from the start of each task.</span></div>
                        ${isStdio ? `<div>${tick('task_session', 'Keep one server session per task', flags.taskSession, `aria-describedby="mcp-${index}-lifetime" ${skill || flags.browserPolicy ? 'disabled' : ''}`)}
                            <span class="ui-field-help" id="mcp-${index}-lifetime" data-mcp-lifetime>${escapeHtml(lifetimeNote(server))}</span></div>` : ''}
                    </div>
                </details>
            </section>
            ${live ? `<section class="ui-card settings-card" data-mcp-tools>
                <div class="settings-card-head"><h3 class="settings-card-title">Tools it offers</h3>
                    <div class="ui-section-actions"><button type="button" class="btn btn-default btn-sm" data-mcp-refresh ${pending.has('refresh') ? 'disabled' : ''}>Refresh</button></div></div>
                <p class="settings-card-lead" data-mcp-weight ${weight ? '' : 'hidden'}>${escapeHtml(weight)}</p>
                <dl class="mcp-server-facts" data-mcp-facts>${factItems}</dl>
                ${toolsListHtml(toolItems)}
            </section>
            <div class="ui-action-row settings-foot"><button type="button" class="btn btn-danger" data-mcp-remove data-editor-remove>Remove</button></div>`
                : '<div class="ui-action-row settings-foot"><button type="button" class="btn btn-default" data-editor-remove>Discard</button></div>'}
        </article>`;
}

/** The form's own refusals, without painting: header rows, fields its transport does not use, environment JSON. */
function cardErrors(card, server) {
    const errors = [...validateHeaderFields(card, server, savedMcpSecrets.get(server)?.identity)];
    const refused = refusedFieldsMessage(server);
    if (refused) errors.push({ input: card.querySelector('[data-mcp-field="transport"]'), message: refused });
    card.querySelectorAll('[data-mcp-field="env"], [data-mcp-field="env_from_settings"]').forEach((input) => {
        let value, parsed = false;
        try { value = JSON.parse(input.value || '{}'); parsed = true; } catch {}
        if (!parsed || (value !== null && (typeof value !== 'object' || Array.isArray(value)
                || Object.entries(value).some(([name, item]) => !name || name.includes('=')
                    || name.includes('\u0000') || typeof item !== 'string' || item.includes('\u0000'))))) {
            errors.push({ input, message: 'Enter a JSON object with nonempty environment names and string values, for example {"PORT":"8080"}. Your input is retained.' });
        }
    });
    return errors;
}

/** The refusals of the form on screen, without painting: what its Save checks before it writes. */
export const validateMcpSettings = () => (editorView?.server && cardFor(editorView.server) ? cardErrors(cardFor(editorView.server), editorView.server) : []);

// A refusal is said beside Save and its field takes the caret; a field inside the closed fold opens the fold first.
function refuse(card, server, error) {
    setCardMessage(server, error.message, 'danger', 'save');
    if (error.input?.hidden) resetSecretReveals(card); // a shown value stands in for its input
    const fold = error.input?.closest?.('details');
    if (fold) fold.open = true;
    error.input?.focus?.();
}

/** One write of one entry: the fresh entry must still be the server this form or head was drawn from. */
async function entryWrite(server, change, fromEditor = false) {
    const saved = savedMcpSecrets.get(server);
    try {
        return await writeEntry(KEY, saved.identity, (before, token) => {
            if (!before || typeof before !== 'object' || canonicalServerId(serverIdentity(before)) !== canonicalServerId(saved.identity)) throw GONE;
            if (fromEditor && (heldEditor(server) || asRead(before) !== saved.read)) throw GONE;
            return change(before, token);
        }, { name: serverName(server) });
    } catch (error) {
        if (error !== GONE) throw error;
        return { state: 'changed_elsewhere', effect: 'now', withheld: '', message: '', code: '' };
    }
}

// What a write says: the helper's words (PINS section 5); `saved` is said for an editor's Save, a switch stays quiet.
const SWITCH = { id: KEY, keys: [KEY], control: { kind: 'switch' } };
function outcomeWords(outcome, saved = '') {
    const line = outcomeLine(SWITCH, outcome);
    if (line) return { text: line.note ? `${line.text}. ${line.note}` : line.text, tone: line.tone, details: line.details };
    return outcome.state === 'saved' && saved ? { text: saved, tone: 'ok' } : {};
}
const landed = (outcome) => outcome.state === 'saved' || outcome.state === 'saved_then_failed';
const sayOutcome = (server, outcome, saved = '') => { const words = outcomeWords(outcome, saved); setCardMessage(server, words.text, words.tone, 'save'); };

/** The switch of a saved server, on its head (`origin` 'list') or its page: one entry written at once, then what
 *  the server holds. While the write is in flight the switch shows the value pressed and takes no other press. */
async function setEnabled(server, on, origin) {
    if (origin === 'editor' && heldEditor(server)) return;
    const id = pageId(server);
    const view = origin === 'list' ? listView : editorView;
    const root = origin === 'list' ? view.el : view.host;
    const current = () => origin === 'list' ? listView === view && listShown(view.ctx, 'tools')
        : editorView === view && editorShown(view);
    const input = () => root.querySelector(origin === 'list' ? `[data-entry-id="${id}"] [data-entry-switch]` : '[data-entry-switch]');
    const line = () => root.querySelector('[data-entry-result]');
    const doc = input()?.ownerDocument;
    const focused = Boolean(doc) && doc.activeElement === input();
    setResult(line(), {});
    wanted.set(server, on);
    paintList(); paintUse();
    const outcome = await entryWrite(server, (before) => ({ ...before, enabled: on }), origin === 'editor');
    wanted.delete(server);
    await reload(view);
    paintList(); paintUse();
    if (current()) {
        if (focused && doc.activeElement === doc.body) input()?.focus?.({ preventScroll: true });
        setResult(line(), outcomeWords(outcome));
    }
    void refreshStatus();
}

// A saved server's switch on its page, painted in place: the form around it may hold unsaved text.
function paintUse() {
    const server = editorView?.server;
    const input = server && cardFor(server)?.querySelector('[data-entry-switch]');
    if (!input) return;
    const on = wanted.has(server) ? wanted.get(server) : Boolean(server.enabled);
    input.checked = on;
    input.setAttribute('aria-checked', String(on));
    setButtonBusy(input, wanted.has(server));
    if (heldEditor(server)) input.disabled = true;
}

function paintEditorActions(view) {
    const server = view?.server;
    if (!server) return;
    const card = view.host.querySelector('[data-mcp-card]');
    if (!card) return;
    const busy = saving.has(server);
    setButtonBusy(card.querySelector('[data-editor-save]'), busy);
    setButtonBusy(card.querySelector('[data-editor-discard]'), busy);
    const remove = card.querySelector('[data-editor-remove]');
    if (!remove.hasAttribute('aria-busy')) remove.disabled = busy || (isLive(server) && heldEditor(server));
    if (!heldEditor(server)) return;
    if (isLive(server)) card.querySelector('[data-editor-discard]')?.removeAttribute('hidden');
    resetSecretReveals(card);
    card.querySelectorAll('[data-entry-switch], [data-mcp-test], [data-mcp-refresh], [data-mcp-header-toggle], [data-mcp-sign-in-action]')
        .forEach(control => { control.disabled = true; });
    card.querySelector('[data-mcp-sign-in-link]')?.setAttribute('hidden', '');
}

// The list token is the only revision for these values. A public mask can stay identical across many writes.
function replacesMaskedValue(before, after) {
    if ((String(before.url).includes('//***@') || before.url === HEADER_PLACEHOLDER) && after.url !== before.url) return true;
    if (before.auth_token && after.auth_token !== before.auth_token) return true;
    if (before.headers === HEADER_PLACEHOLDER) return after.headers !== HEADER_PLACEHOLDER;
    const kept = new Set(Object.entries(after.headers || {}).filter(([, value]) => value === HEADER_PLACEHOLDER)
        .map(([name]) => name.toLowerCase()));
    return Object.entries(before.headers || {}).some(([name, value]) => value === HEADER_PLACEHOLDER && !kept.has(name.toLowerCase()));
}

async function saveServer(card, server) {
    if (saving.has(server)) return;
    if (heldEditor(server)) {
        setCardMessage(server, conflictWords(server), 'warn', 'conflict');
        return;
    }
    const view = editorView;
    const errors = cardErrors(card, server);
    if (errors.length) { refuse(card, server, errors[0]); return; }
    saving.set(server, 'writing');
    paintEditorActions(view);
    try {
        await writeServer(card, server, view);
    } finally {
        saving.delete(server);
        paintEditorActions(view);
        if (view !== editorView) paintEditorActions(editorView);
    }
}

async function writeServer(card, server, view) {
    const live = isLive(server);
    const collected = collectServer(server);
    // A server is found by its ID: a new one named only by its name takes the ID the loader derives from that name.
    if (!live && !collected.id) collected.id = canonicalServerId(serverIdentity(collected));
    const identity = canonicalServerId(serverIdentity(collected));
    if (!identity) { refuse(card, server, { input: card.querySelector('[data-mcp-field="name"]'), message: 'Give the server a name or an ID.' }); return; }
    if (mcpServers.some((other) => other !== server && isServer(other) && pageId(other) === identity)) {
        refuse(card, server, { input: card.querySelector('[data-mcp-field="id"]'), message: `Another server already uses the ID ${identity}.` });
        return;
    }
    // A form held under this ID (its entry may be gone) is never replaced by another form's Save.
    if (heldDrafts.has(identity) && heldDrafts.get(identity) !== server) {
        refuse(card, server, { input: card.querySelector('[data-mcp-field="id"]'),
            message: `Unsaved changes are kept for the ID ${identity}. Open them from the list and discard them first.` });
        return;
    }
    const generation = draftGenerations.get(server);
    const saved = savedMcpSecrets.get(server);
    const oldId = saved && canonicalServerId(saved.identity);
    const editedSwitch = saved && collected.enabled !== saved.enabled;
    const submitted = { inputId: server.id, inputUrl: server.url, headers: headerSubmission(server) };
    const newer = () => dirty.has(server) && draftGenerations.get(server) !== generation;
    setCardMessage(server, '');
    let outcome, observed, observedToken;
    try {
        outcome = live ? await entryWrite(server, async (before, token) => {
            observed = before;
            observedToken = token;
            if ((!saved.token || saved.token !== token) && replacesMaskedValue(before, collected)) {
                const why = !token ? 'The current list version is unavailable. Other edits made before this save finishes could be overwritten.'
                    : !saved.token ? 'The original list version is unavailable.' : 'The server list changed since you opened this draft.';
                const ask = view.ctx.ask || openConfirmDialog;
                if (!await ask({ title: 'Save changes to masked values?', confirmLabel: 'Save changes',
                    body: `${why} A masked value you edited or removed may have changed elsewhere. Apply your edits?` })) throw CANCELLED;
            }
            // The new form's switch can also have been edited during Add. Otherwise the saved head owns it.
            return { ...collected, enabled: editedSwitch ? collected.enabled : Boolean(before.enabled) };
        }, true) : await writeEntry(KEY, collected.id, (before) => { if (before) throw GONE; return collected; }, { name: serverName(collected) });
    } catch (error) {
        if (error === CANCELLED) { setCardMessage(server, 'Save cancelled. Your edits are kept here.', 'muted', 'save'); return; }
        if (error !== GONE) throw error;
        outcome = { state: 'refused', effect: 'now', withheld: '', message: `Another server already uses the ID ${collected.id}.`, code: 'taken' };
    }
    // The same form can have a new mount by the time its write answers.
    if (editorView?.server === server && editorShown(editorView)) view = editorView;
    saving.set(server, 'reading');
    const unchanged = outcome.state === 'unchanged' && outcome.code === 'no_changes';
    const confirmed = landed(outcome) || unchanged;
    const inDoubt = outcome.state === 'unknown';
    if (!confirmed && outcome.state !== 'changed_elsewhere' && !inDoubt) { sayOutcome(server, outcome); return; }
    if (confirmed && !newer()) dirty.delete(server);
    if (confirmed) {
        // A fresh GET can canonicalize the id and mask fields. Its receipt token identifies the
        // exact list we wrote. Keep edits based on that version, even across a return to this page.
        ++savedGeneration;
        let index = mcpServers.indexOf(server);
        if (index < 0) index = mcpServers.findIndex(other => isServer(other) && [oldId, identity].includes(pageId(other)));
        const projected = { ...collected, id: canonicalServerId(collected.id) };
        if (isServer(collected.headers)) {
            projected.headers = Object.fromEntries(Object.entries(collected.headers).map(([key, value]) =>
                [key, value === '' ? '' : HEADER_PLACEHOLDER]));
        }
        const maskedHeaders = Boolean(collected.auth_token) || projected.headers === HEADER_PLACEHOLDER
            || (isServer(projected.headers) && Object.values(projected.headers).includes(HEADER_PLACEHOLDER));
        const token = outcome.token || (unchanged ? observedToken : undefined);
        savedMcpSecrets.set(server, { ...savedEntry(unchanged && observed ? observed : projected, token),
            submitted: { ...submitted, maskedHeaders, unchanged } });
        heldDrafts.set(identity, server);
        if (index < 0) mcpServers.push(server);
        else mcpServers[index] = server;
        if (draftServer === server) draftServer = null;
        if (view.server === server && view.entry !== identity) view.savedAs = identity;
    } else if (dirty.has(server) || inDoubt) {
        holdDraft(server, inDoubt ? UNCONFIRMED_CONFLICT : CHANGED_CONFLICT);
    }
    if (inDoubt) return;
    const read = await reload(view);
    void refreshStatus();
    if (confirmed && read === false && !newer()) {
        setCardMessage(server, 'Saved. The current settings could not be read. Press Discard changes to load them.', 'warn', 'save');
        cardFor(server)?.querySelector('[data-editor-discard]')?.removeAttribute('hidden');
        return;
    }
    if (newer() && confirmed) {
        if (heldDrafts.get(identity) === server) setCardMessage(server, READBACK_CONFLICT, 'warn', 'conflict');
        else sayNewerEdits(server, outcome);
    }
    if (editorView !== view || !editorShown(view) || (!live && view.server !== server)) return;
    if (view.savedAs && confirmed) {
        if (live || newer()) openSavedEditor(view);
        else {
            view.savedAs = '';
            showList(view, outcomeWords(outcome, `Added ${serverName(collected)}.${collected.enabled ? '' : ' It is off until you switch it on.'}`), identity);
        }
    } else if (view.server && !dirty.has(view.server) && !newer()) {
        if (unchanged) setCardMessage(view.server, 'Already saved.', 'ok', 'save');
        else sayOutcome(view.server, outcome, 'Saved');
    }
}

function sayNewerEdits(server, outcome) {
    const note = outcome.code === 'no_changes' ? 'The submitted settings were already saved. Your latest edits are unsaved.'
        : 'Earlier changes saved. Your latest edits are unsaved.';
    setCardMessage(server, outcome.message ? `${note} ${outcome.message}` : note, 'warn', 'save');
}

function openSavedEditor(view) {
    const id = view.savedAs;
    const server = view.server;
    const focus = editorFocus(view.host);
    view.savedAs = '';
    view.server = null;  // the saved page now owns its draft; a retained tools/new can start again
    view.ctx.openSettings?.(`tools/${id}`);
    if (editorView?.server === server && editorView.entry === id) focus?.(editorView.host);
}

// Back to the list with one sentence in its result line (a server added or removed has no page of its own to say
// it); an added server's head is shown and takes the caret (DESIGN "List editors").
function showList(view, result, added = '', removedAt = -1) {
    view.ctx.openSettings?.('tools');
    if (!listView || !listShown(listView.ctx, 'tools')) return;
    setResult(listView.el.querySelector('[data-entry-result]'), result);
    if (added) {
        const head = listView.el.querySelector(`[data-entry-id="${added}"]`);
        if (head) revealNewRow(head, head.querySelector('[data-entry-open]'));
    } else {
        const next = mcpServers[Math.min(removedAt, mcpServers.length - 1)];
        listView.el.querySelector(next ? `[data-entry-id="${pageId(next)}"] [data-entry-open]` : '[data-tools-add]')?.focus?.();
    }
}

function bindCardEvents(card, rowServer) {
    const boundView = editorView;
    const saved = savedMcpSecrets.get(rowServer);
    bindHeaderFields(card, { server: rowServer, savedIdentity: saved?.identity, onChange: () => serverEdited(rowServer), render: drawEditor });
    card.querySelector('[data-mcp-sign-in]')?.addEventListener('click', (event) => {
        if (heldEditor(rowServer)) { event.preventDefault(); return; }
        const link = event.target.closest('a[data-mcp-sign-in-link]');
        if (link) {
            event.preventDefault(); // through the host while the press is live (desktop shell, Telegram)
            void openExternalViaHostBridge(link.getAttribute('href')).catch(() => {});
            return;
        }
        const button = event.target.closest('[data-mcp-sign-in-action]');
        if (button) void signInAction(card, rowServer, button.dataset.mcpSignInAction);
    });
    card.querySelector('[data-mcp-more]')?.addEventListener('toggle', (event) => {
        if (event.target.open) moreOpen.add(rowServer); else moreOpen.delete(rowServer);
    });
    card.querySelector('[data-entry-switch]')?.addEventListener('change', (event) => {
        void setEnabled(rowServer, event.target.checked, 'editor');
    });
    const changeTaskSession = (input) => {
        // A lifetime edit never silently removes the imported browser policy; a skill-bound row's is its skill's.
        if (rowServer.skill || lifetimeFlagOn(rowServer.browser_policy)) {
            input.checked = lifetimeOf(rowServer).taskSession;
            return;
        }
        rowServer.task_session = Boolean(input.checked);
        paintLifetime(card, rowServer);
        serverEdited(rowServer);
    };
    const TICKS = ['enabled', 'schemas_at_start'];
    card.querySelectorAll('[data-mcp-field]').forEach((input) => {
        input.addEventListener('input', () => {
            const field = input.dataset.mcpField;
            const server = rowServer;
            if (field === 'task_session') { changeTaskSession(input); return; }
            if (TICKS.includes(field)) server[field] = Boolean(input.checked);
            else if (field === 'allowed_tools') server.allowed_tools = String(input.value || '').split(',').map((s) => s.trim()).filter(Boolean);
            else if (field === 'args') server.args = String(input.value || '').split(/\r?\n/).filter((value) => value.length > 0);
            else if (field === 'env_from_settings' || field === 'env') {
                try {
                    server[field] = JSON.parse(input.value || '{}');
                    setCardMessage(server, '');
                } catch {
                    server[field] = input.value;
                    setCardMessage(server, 'Environment values and references must be JSON objects. Your input is retained.', 'danger');
                }
            } else server[field] = input.value;
            if (field === 'transport') drawEditor();
            serverEdited(server);
        });
        input.addEventListener('change', () => {
            const field = input.dataset.mcpField;
            if (field === 'task_session') { changeTaskSession(input); return; }
            if (!TICKS.includes(field)) return;
            rowServer[field] = Boolean(input.checked);
            if (field === 'enabled') input.setAttribute('aria-checked', String(input.checked));
            if (field === 'schemas_at_start') paintStatus();
            serverEdited(rowServer);
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
        drawEditor();
        serverEdited(rowServer);
    });
    card.querySelector('[data-editor-save]')?.addEventListener('click', () => void saveServer(card, rowServer));
    card.querySelector('[data-editor-discard]')?.addEventListener('click', async () => {
        if (saving.has(rowServer)) return;
        const generation = draftGenerations.get(rowServer);
        saving.set(rowServer, 'discarding');
        paintEditorActions(boundView);
        setCardMessage(rowServer, 'Loading the current settings...', 'muted', 'discard');
        try {
            const read = await reload(boundView, () => {
                if (draftGenerations.get(rowServer) !== generation) return false;
                dirty.delete(rowServer);
                return true;
            });
            if (read && editorView && editorShown(editorView)) {
                if (editorView.server === rowServer && editorView.savedAs) openSavedEditor(editorView);
                else if (editorView.entry === pageId(rowServer) && editorView.server && !dirty.has(editorView.server)) {
                    setCardMessage(editorView.server, 'Current settings loaded.', 'ok', 'discard');
                }
            } else if (!read && draftGenerations.get(rowServer) === generation) {
                setCardMessage(rowServer, DISCARD_FAILED, 'warn', 'discard');
            }
        } finally {
            saving.delete(rowServer);
            paintEditorActions(boundView);
            if (boundView !== editorView) paintEditorActions(editorView);
        }
    });
    card.querySelector('[data-editor-remove]')?.addEventListener('click', async (event) => {
        const view = boundView;
        if (saving.has(rowServer) || (isLive(rowServer) && heldEditor(rowServer))) return;
        if (!isLive(rowServer)) {
            dirty.delete(rowServer);
            staleDrafts.delete(rowServer);
            draftServer = null;
            view.ctx.openSettings?.('tools');
            return;
        }
        const name = serverName(rowServer), index = mcpServers.indexOf(rowServer);
        if (!(await CONFIRMS.removeEntry({ name }, null, null, view.ctx.ask))) return;
        if (editorView?.server !== rowServer || saving.has(rowServer) || heldEditor(rowServer)) return;
        setButtonBusy(event.target, true);
        const outcome = await entryWrite(rowServer, () => null, true);
        setButtonBusy(event.target, false);
        if (outcome.state === 'changed_elsewhere' && dirty.has(rowServer)) {
            holdDraft(rowServer, CHANGED_CONFLICT);
            await reload(view);
            return;
        }
        if (!landed(outcome) && outcome.state !== 'changed_elsewhere') { sayOutcome(rowServer, outcome); return; }
        dirty.delete(rowServer);
        await reload(view);
        void refreshStatus();
        if (editorView === view && editorShown(view)) showList(view, outcomeWords(outcome, `Removed ${name}.`), '', index);
    });
    card.querySelector('[data-mcp-test]')?.addEventListener('click', async () => {
        const server = rowServer;
        if (heldEditor(server)) return;
        const errors = validateHeaderFields(card, server, saved?.identity);
        if (errors.length) { refuse(card, server, errors[0]); return; }
        const generation = draftGenerations.get(server) || 0;
        const stale = () => (draftGenerations.get(server) || 0) !== generation;
        setPending(server, 'test', true);
        setCardMessage(server, 'Testing the connection...', 'muted', 'test');
        let said;
        try {
            // The saved identity lets the backend restore kept values; the
            // candidate's own identity still decides which saved server they are.
            const candidate = collectServer(server);
            const urlMasked = String(server.url || '').includes('://***@');
            const keep = saved?.identity && (looksMasked(candidate.auth_token) || urlMasked || headerKeepsSavedValues(server));
            const body = keep ? { server_id: saved.identity, server: candidate } : { server: serverForTest(candidate) };
            const data = await jsonPost('/api/mcp/test', body, { rejectOkFalse: true });
            const warnings = (data.configuration_warnings || []).join(' ');
            said = stale() ? ['This server changed while it was tested. Test again to check what the form holds now.', 'warn']
                : [`The connection works. ${toolCountLabel(Number(data.tool_count || 0))} discovered. No tool was called.${warnings ? ' ' + warnings : ''}`, warnings ? 'warn' : 'ok'];
        } catch (err) { said = [`Test failed${stale() ? ' for an earlier draft' : ''}: ${err && err.message ? err.message : err}`, 'danger']; }
        if (editorView?.server === server) setCardMessage(server, ...said, 'test');
        setPending(server, 'test', false);
        await refreshStatus(); // a draft that asked for sign-in shows its Sign in
    });
    card.querySelector('[data-mcp-refresh]')?.addEventListener('click', async () => {
        const server = rowServer;
        if (heldEditor(server)) return;
        setPending(server, 'refresh', true);
        setCardMessage(server, 'Listing its tools again...', 'muted', 'refresh');
        let said;
        try {
            // The saved catalog belongs to the saved identity (id, slug or name).
            const data = await jsonPost('/api/mcp/refresh', { server_id: saved.identity }, { rejectOkFalse: true });
            if (data.published === false) throw new Error(data.publication_error || 'Catalog was not published');
            said = [`Listed again. ${toolCountLabel(Number(data.tool_count || 0))} discovered. No tool was called.`, 'ok'];
        } catch (err) { said = [`Refresh failed: ${err && err.message ? err.message : err}`, 'danger']; }
        await refreshStatus();
        if (editorView?.server === server) setCardMessage(server, ...said, 'refresh');
        setPending(server, 'refresh', false);
    });
}

// Drop a literal mask, so a test without a saved identity never sends "***" as Bearer auth.
const serverForTest = (server) => (looksMasked(server.auth_token) ? { ...server, auth_token: '' } : { ...server });

// Remember only the control's location and selection. Header names identify their rows even when a read
// reorders the map. A revealed value belongs to its old node and is put away before the next form is bound.
function editorFocus(host) {
    const at = host.ownerDocument?.activeElement;
    if (!at || !host.contains?.(at)) return null;
    const row = at.closest('[data-mcp-header-row]');
    const headerName = node => node.querySelector('[data-mcp-header-field="name"]').value;
    const name = row && headerName(row);
    const button = ['data-mcp-header-remove', 'data-entry-switch', 'data-editor-save', 'data-editor-remove', 'data-mcp-test', 'data-mcp-refresh', 'data-mcp-header-add']
        .find(key => at.hasAttribute(key));
    const selector = at.dataset.mcpField ? `[data-mcp-field="${at.dataset.mcpField}"]`
        : at.dataset.mcpHeaderField ? `[data-mcp-header-field="${at.dataset.mcpHeaderField}"]`
        : button && `[${button}]`;
    if (!selector) return null;
    const rowIndex = row?.dataset.mcpHeaderRow;
    const range = typeof at.selectionStart === 'number' ? [at.selectionStart, at.selectionEnd, at.selectionDirection] : null;
    return target => {
        let scope = target;
        if (rowIndex !== undefined) {
            const matches = [...target.querySelectorAll('[data-mcp-header-row]')].filter(node => headerName(node) === name);
            scope = matches.length === 1 ? matches[0] : matches.find(node => node.dataset.mcpHeaderRow === rowIndex);
        }
        const field = scope?.querySelector(selector);
        field?.focus?.({ preventScroll: true });
        if (range && typeof field?.selectionStart === 'number') field.setSelectionRange(...range);
    };
}

function drawEditor({ keepFocus = true } = {}) {
    const view = editorView;
    if (!view) return;
    const more = view.host.querySelector('[data-mcp-more]');
    const focus = keepFocus && editorFocus(view.host);
    resetSecretReveals(view.host);
    if (!view.server) {
        view.host.innerHTML = loaded ? '<p class="settings-page-lead">This server is no longer on the list.</p>' : '';
        return;
    }
    if (more) moreOpen[more.open ? 'add' : 'delete'](view.server);
    view.host.innerHTML = editorHtml(view.server);
    view.boundEntry = savedMcpSecrets.get(view.server);
    view.host.querySelectorAll('[data-mcp-card]').forEach((card) => bindCardEvents(card, view.server));
    paintEditorActions(view);
    if (focus) focus(view.host);
}

const VIA = { streamable_http: 'HTTP', sse: 'SSE', stdio: 'Local program' };

/** A server's head on the list: its name, its state in the status's own words and how it is reached; Sign in when
 *  the status asks for it; its switch. Pressing the head opens the server's page. */
function headHtml(server) {
    const unavailable = mcpStatusEnvelope === null;
    const need = unavailable ? '' : signInNeed(server);
    const busy = wanted.has(server);
    return `<div class="ui-row settings-head" data-entry-id="${escapeHtml(pageId(server))}"${need ? ' data-actions' : ''}>
        <span class="settings-head-glyph" aria-hidden="true">${controlIcon('plug', 16)}</span>
        <button type="button" class="settings-head-open ui-hit-area" data-entry-open>
            <span class="ui-row-title">${escapeHtml(serverName(server))}</span>
            <span class="ui-row-meta">${stateHtml(server, unavailable)}<span> · ${VIA[server.transport] || escapeHtml(String(server.transport))}</span></span>
        </button>
        <div class="ui-row-actions">${need ? `<button type="button" class="btn btn-default btn-sm" data-entry-sign-in>${need}</button>` : ''}
            ${switchHtml(`data-entry-switch aria-label="Use ${escapeHtml(serverName(server))}"${busy ? ' disabled aria-busy="true"' : ''}`, busy ? wanted.get(server) : Boolean(server.enabled))}${doorSign()}</div>
    </div>`;
}

// The card's one line about every server: what is listed and what it weighs in a request, in the status's sentences.
function envelopeWords() {
    const envelope = mcpStatusEnvelope;
    if (envelope === null) return { text: STATUS_UNAVAILABLE, tone: 'warn' };
    if (!envelope) return { text: '', tone: '' };
    if (!envelope.sdk_available) return { text: `MCP servers cannot run: ${envelope.sdk_error || 'install `mcp>=1.10` to use them.'}`, tone: 'warn' };
    if (!envelope.enabled) return { text: 'All MCP servers are off. Their settings are kept.', tone: '' };
    const servers = Array.isArray(envelope.servers) ? envelope.servers : [];
    const tools = servers.reduce((sum, s) => sum + Number(s.tool_count || 0), 0);
    const weights = envelope.weights;
    const cost = weights ? ` Connections map ${tokensLabel(weights.map_tokens)} in every Low/Max request; call schemas from task start ${tokensLabel(weights.schema_tokens_from_task_start)}.` : '';
    const configured = configuredServerCount(servers);
    return { text: `${configured} server${configured === 1 ? '' : 's'} configured, ${toolCountLabel(tools)} discovered.${cost}`, tone: '' };
}

function paintList() {
    const view = listView;
    if (!view) return;
    const lead = view.el.querySelector('[data-mcp-global]');
    const words = envelopeWords();
    if (lead) {
        if (lead.textContent !== words.text) lead.textContent = words.text;
        lead.classList.toggle('ui-status', Boolean(words.tone));
        if (words.tone) lead.dataset.tone = words.tone; else delete lead.dataset.tone;
        lead.hidden = !words.text;
    }
    // A held form whose entry is gone has no head of its own: one opener leads to it, so its Discard stays reachable.
    const absent = [...heldDrafts].filter(([id]) => !mcpServers.some((server) => isServer(server) && pageId(server) === id));
    const keptHeads = absent.map(([id, server]) => `<div class="ui-row settings-head" data-entry-id="${escapeHtml(id)}">
        <span class="settings-head-glyph" aria-hidden="true">${controlIcon('plug', 16)}</span>
        <button type="button" class="settings-head-open ui-hit-area" data-entry-open>
            <span class="ui-row-title">${escapeHtml(serverName(server))}</span>
            <span class="ui-row-meta"><span class="ui-status" data-tone="warn">Unsaved changes · not on the saved list</span></span>
        </button></div>`).join('');
    const heads = !loaded ? '' : !mcpServers.length ? (keptHeads || '<p class="settings-card-lead">No MCP server yet. Add one, or import an mcp.json from Claude Code, Cursor or Codex.</p>')
        : keptHeads + mcpServers.map((server, index) => (isServer(server) && pageId(server) ? headHtml(server)
            : `<div class="ui-row settings-row" data-mcp-invalid="${index}"><p class="ui-row-title">Entry ${index + 1} cannot be read</p>
                <p class="ui-row-meta"><span>It is kept as it is until you remove it.</span></p>
                <div class="ui-row-actions"><button type="button" class="btn btn-default btn-sm" data-mcp-invalid-remove>Remove</button></div></div>`)).join('');
    if (view.heads === heads) return;
    const list = view.el.querySelector('[data-tools-list]');
    const doc = list.ownerDocument;
    const at = list.contains(doc.activeElement) ? doc.activeElement : null;  // a head drawn again keeps the caret on its control
    const held = at && [at.closest('[data-entry-id]')?.dataset.entryId, ['data-entry-switch', 'data-entry-sign-in', 'data-entry-open'].find((name) => at.hasAttribute(name))];
    list.innerHTML = heads;
    view.heads = heads;
    if (held?.[0] && held[1]) list.querySelector(`[data-entry-id="${held[0]}"] [${held[1]}]`)?.focus?.({ preventScroll: true });
}

let statusRead = null;
let lastReconnect;
/** Both sections hear the same socket event. Retire the pre-reconnect GET once, before either section reads. */
export function retireMcpStatusRead(event) {
    if (!event?.previouslyConnected || event === lastReconnect) return;
    lastReconnect = event;
    const previous = statusRead;
    statusRead = null;
    previous?.controller.abort();
}

/** The last completed read, including failure, for a summary returning to the screen. */
export const mcpStatusSeen = () => mcpStatusEnvelope;

export function refreshStatus({ join = false } = {}) {
    if (join && statusRead) return statusRead.done;
    const controller = new AbortController();
    const read = { controller };
    read.done = fetchJson('/api/mcp/status', { cache: 'no-store', signal: controller.signal }).catch(() => null).then((result) => {
        if (statusRead !== read) return;
        const status = Array.isArray(result?.servers) ? result : null;
        mcpStatusEnvelope = status;
        if (status) mcpStatusByServer = Object.fromEntries(status.servers.filter((entry) => entry?.id).map((entry) => [entry.id, entry]));
        // The list and summaries say failure; the mounted editor keeps its last read and sign-in fields.
        globalThis.window?.dispatchEvent?.(new CustomEvent('ouro:mcp-status-updated', { detail: status }));
        if (status) paintStatus();
        else paintList();
    }).finally(() => { if (statusRead === read) statusRead = null; });
    statusRead = read;
    return read.done;
}

/** Connect tools settled. Read the list when mounted and publish its passive status to both sections. */
export function readConnectedServers() {
    if (listView) void reload(listView);
    void refreshStatus();
}

// The settings are read again and both views show what the server holds (a typed form keeps its text).
async function reload(origin, releaseDraft) {
    const otherPage = () => origin && editorView && editorView !== origin && editorShown(editorView)
        && (!origin.server || editorView.server !== origin.server);
    if (!releaseDraft && (otherPage() || saving.get(origin?.server) === 'discarding')) return;
    const read = ++reloadSerial;
    const before = savedGeneration;
    let settings;
    try { settings = await (editorView?.ctx || listView?.ctx).readSettings(); } catch { return false; }  // the views keep what they show
    // Another form has its own read. Rebuilding it for a previous page's operation would drop its caret.
    if (read !== reloadSerial || (!releaseDraft && otherPage())) return;
    if (before !== savedGeneration) return reload(origin, releaseDraft);  // another editor saved while this read was on its way
    if (!Array.isArray(settings?.[KEY]) || (releaseDraft && !releaseDraft())) return false;
    applyMcpSettings(settings);
    return true;
}

// The list as one value: text, so an answer that never came is not mistaken for the list having landed.
CODECS.toolServers = { read: (settings) => (settings ? JSON.stringify(settings[KEY] ?? []) : undefined), body: (value) => ({ [KEY]: JSON.parse(value) }) };
const IMPORT = { id: `${KEY}:import`, keys: [KEY], writer: 'generic', codec: 'toolServers', control: { kind: 'custom' } };

/** Import a file of servers: the chosen entries are written in one write of the list, from a fresh read. */
async function importServers(ctx, result) {
    const applied = await openMcpImportDialog({
        draftServers: () => mcpServers.map((server) => isServer(server) ? {
            ...(server.id ? { id: String(server.id) } : {}), ...(server.slug ? { slug: String(server.slug) } : {}),
            ...(server.name ? { name: String(server.name) } : {}), ...(server.skill ? { skill: String(server.skill) } : {}),
            transport: String(server.transport || 'streamable_http'), ...headerImportFacts(server),
        } : null),
        draftFreshness,
        draftUrl: (index) => isServer(mcpServers[index]) ? String(mcpServers[index].url || '') : '',
        onApply: async (entries) => {
            let fresh;
            try { fresh = await ctx.readSettings(); } catch { return { state: 'refused', message: 'The servers could not be read.' }; }
            const list = Array.isArray(fresh[KEY]) ? [...fresh[KEY]] : [];
            for (const entry of entries) {
                if (entry.action === 'add') { list.push({ ...entry.patch }); continue; }
                const target = isServer(mcpServers[entry.index]) ? pageId(mcpServers[entry.index]) : '';
                const at = list.findIndex((item) => isServer(item) && canonicalServerId(serverIdentity(item)) === target);
                if (!target || at < 0) return { state: 'changed_elsewhere' };
                list[at] = { ...list[at], ...entry.patch };
            }
            return commitSetting(IMPORT, JSON.stringify(list), { ifUnchanged: fresh._meta?.tokens?.[KEY] || '' });
        },
    });
    if (!applied) return;
    await reload();
    setResult(result, outcomeWords(applied, 'Imported. A new server is off until you switch it on.'));
    void refreshStatus();
}

/** The list: the client's switch, the servers, the fold, and where else outside abilities live. */
function mountList(container, { path, ctx }) {
    const custom = (row) => `<button type="button" class="btn btn-default btn-sm" data-tools-action="${escapeHtml(row.id)}">${escapeHtml(row.control.label)}</button>`;
    const rows = (fold) => rowsOf('tools', { fold }).filter((row) => !row.id.startsWith('tools:door:')).map((row) => settingsRow(row, { custom })).join('');
    container.innerHTML = `<p class="settings-page-lead">Outside services Ouroboros can call.</p>
        <section class="ui-card settings-card"><div class="settings-rows">${rows(false)}</div></section>
        <section class="ui-card settings-card" data-setting-row="${KEY}">
            <div class="settings-card-head"><h3 class="settings-card-title">Servers</h3>
                <span class="ui-result" data-entry-result role="status" aria-live="polite"></span>
                <div class="ui-section-actions"><button type="button" class="btn btn-default btn-sm" data-tools-add>Add server</button></div></div>
            <p class="settings-card-lead" data-mcp-global hidden></p>
            <div class="settings-rows" data-tools-list></div>
        </section>
        <section class="ui-card settings-card"><h3 class="settings-card-title">Elsewhere</h3>
            <div class="settings-rows">${rowsOf('tools').filter((row) => row.id.startsWith('tools:door:')).map((row) => settingsRow(row)).join('')}</div></section>
        <section class="ui-card settings-card"><details class="ui-disclosure" data-settings-fold>
            <summary>More: timeout, import, refresh</summary>
            <div class="settings-rows">${rows(true)}</div></details></section>`;
    const view = { el: container, ctx, heads: null };
    listView = view;
    const bound = bindSettingRows(container, { ask: ctx.ask, custom, openDoor: (row, opener) => ctx.openSettings?.(row.control.to, { opener }) });
    const entryResult = () => container.querySelector('[data-entry-result]');
    const serverOf = (node) => { const id = node?.closest('[data-entry-id]')?.dataset.entryId;
        return mcpServers.find((server) => isServer(server) && pageId(server) === id) || heldDrafts.get(id); };
    const actionResult = (button) => button.closest('[data-setting-row]')?.querySelector('[data-setting-result]');
    const onClick = async (event) => {
        const hit = (selector) => event.target.closest?.(selector);
        const add = hit('[data-tools-add]');
        if (add) { ctx.openSettings('tools/new', { opener: add }); return; }
        const action = hit('[data-tools-action]');
        if (action?.dataset.toolsAction === 'tools:import') { void importServers(ctx, actionResult(action)); return; }
        if (action) {  // every saved server lists its tools again
            const say = (text, tone) => setResult(actionResult(action), { text, tone });
            setButtonBusy(action, true);
            say('Listing every server again...', 'muted');
            await jsonPost('/api/mcp/refresh', {}, { rejectOkFalse: true }).then(() => say('Listed again. No tool was called.', 'ok'),
                (err) => say(`Refresh failed: ${err?.message || err}`, 'danger'));
            setButtonBusy(action, false);
            await refreshStatus();
            return;
        }
        if (hit('[data-mcp-invalid-remove]')) {
            // An entry with no identity: the list's own write finds the first such entry.
            const outcome = await writeEntry(KEY, '', null, { name: 'this entry', ask: ctx.ask });
            await reload(view);
            if (listView === view && listShown(ctx, path)) setResult(entryResult(), outcomeWords(outcome, 'Removed.'));
            return;
        }
        const server = serverOf(event.target);
        if (!server) return;
        if (hit('[data-entry-sign-in]')) { void signInAction(null, server, 'begin', (text, tone) => setResult(entryResult(), { text, tone })); return; }
        if (!hit('.ui-switch')) ctx.openSettings(`tools/${pageId(server)}`, { opener: hit('[data-entry-open]') });
    };
    const onChange = (event) => {
        const server = event.target.matches?.('[data-entry-switch]') && serverOf(event.target);
        if (server) void setEnabled(server, event.target.checked, 'list');
    };
    container.addEventListener('click', onClick);
    container.addEventListener('change', onChange);
    // Its rows are painted by a read only while the list is the page on screen (a server's page over it, or another
    // place, waits for the shell's "shown" signal); its heads are drawn either way, so a Save or Remove lands on them.
    const read = () => { void refreshStatus({ join: true }); return reload(view).then(() => { if (listShown(ctx, path)) bound.paint(); }); };
    const onShown = (event) => { if (event.detail?.path === path) void read(); };
    // The client's own switch changes what the status says about every server.
    const onSaved = (event) => { if (event.detail?.source === 'settings-row' && event.detail.page === 'tools') void refreshStatus(); };
    globalThis.window?.addEventListener('ouro:settings-page-shown', onShown);
    globalThis.window?.addEventListener('ouro:settings-updated', onSaved);
    paintList();
    void read();
    return { unmount() {
        if (listView === view) listView = null;
        bound.dispose();
        container.removeEventListener('click', onClick);
        container.removeEventListener('change', onChange);
        globalThis.window?.removeEventListener('ouro:settings-page-shown', onShown);
        globalThis.window?.removeEventListener('ouro:settings-updated', onSaved);
    } };
}

/** A server's page: drawn at once from what the last read holds, then read again (a typed form keeps its text). */
function mountEditor(container, { path, ctx }, entry) {
    if (entry === 'new') draftServer ||= emptyServer();
    const view = { host: container, path, entry, ctx, server: entry === 'new' ? draftServer : serverOfPage(entry) };
    editorView = view;
    drawEditor();
    const read = () => { void refreshStatus({ join: true }); return reload(view); };
    // A retained origin can come back after another server occupied this section. It becomes the one editor
    // receiving reads and write outcomes again; the same form still owns its bindings and its draft.
    const onShown = (event) => {
        if (event.detail?.path !== path || !editorShown(view)) return;
        editorView = view;
        if (view.savedAs && (entry !== 'new' || dirty.has(view.server))) { openSavedEditor(view); return; }
        view.savedAs = '';
        // A jump may retain tools/new even after its draft was discarded or saved.
        if (entry === 'new' && (!draftServer || view.server !== draftServer)) {
            view.server = draftServer ||= emptyServer();
            drawEditor({ keepFocus: false });
        } else if (entry !== 'new' && (view.server !== serverOfPage(entry) || view.boundEntry !== savedMcpSecrets.get(view.server))) {
            view.server = serverOfPage(entry);
            drawEditor();
        } else {
            paintEditorStatus();
        }
        void read();
    };
    globalThis.window?.addEventListener('ouro:settings-page-shown', onShown);
    const handle = { unsaved: () => Boolean(view.server && dirty.has(view.server)), unmount() {
        if (editorView === view) editorView = null;
        resetSecretReveals(container);
        globalThis.window?.removeEventListener('ouro:settings-page-shown', onShown);
    } };
    return read().then(() => handle);  // the shell titles the page once the read names the server
}

const serverOfPage = (id) => heldDrafts.get(id) || mcpServers.find((server) => isServer(server) && pageId(server) === id) || null;
// A page mounted by a section knows whether it is the page on screen; one mounted outside a section is taken as shown.
const listShown = (ctx, path) => typeof ctx?.onScreen !== 'function' || ctx.onScreen(path);
const editorShown = (view) => Boolean(view?.host.isConnected && listShown(view.ctx, view.path));
const entryOf = (path) => { try { return decodeURIComponent(String(path).split('/')[1] || ''); } catch { return ''; } };

registerSettingsPage('tools', {
    title: (path) => { const id = entryOf(path); return !id ? 'MCP servers' : id === 'new' ? 'New server' : NAMES.get(id) || 'MCP server'; },
    mount: (container, page) => (entryOf(page.path) ? mountEditor(container, page, entryOf(page.path)) : mountList(container, page)),
});

// This is a match for the public client-ID field only, not proof of an immutable row or credential revision.
function sameClientEntry(view, next) {
    const { server, boundEntry: before } = view;
    const after = savedMcpSecrets.get(next);
    if (!server?.id || server.id !== next?.id || !before?.token || !after?.token || before !== savedMcpSecrets.get(server)
        || before.submitted || after.submitted || before.read !== after.read || before.enabled !== after.enabled
        || dirty.has(server) || heldEditor(server) || heldEditor(next) || saving.has(server) || signInBusy.has(server)
        || !['streamable_http', 'sse'].includes(server.transport) || server.auth_token || headerKeepsSavedValues(server)) return false;
    try {
        const url = new URL(before.url);
        return ['http:', 'https:'].includes(url.protocol) && !url.username && !url.password;
    } catch { return false; } // no readable address
}

// A redraw binds all actions to the current entry and slot. Only the public client ID crosses that redraw.
function clientIdDraft(host) {
    const signIn = host.querySelector('[data-mcp-sign-in]');
    const input = signIn?.querySelector('[data-mcp-sign-in-client-id]');
    if (!input || signIn.querySelector('[data-mcp-sign-in-link]')) return null;
    const key = signIn.dataset.signInKey;
    const { value, selectionStart, selectionEnd, selectionDirection } = input;
    const open = input.closest('details').open;
    const focused = host.ownerDocument?.activeElement === input;
    return () => {
        const current = host.querySelector('[data-mcp-sign-in]');
        const field = current?.querySelector('[data-mcp-sign-in-client-id]');
        if (!field || current.dataset.signInKey !== key) return;
        field.value = value || '';
        field.closest('details').open = open;
        if (!focused || !open) return;
        field.focus({ preventScroll: true });
        if (typeof selectionStart === 'number') field.setSelectionRange(selectionStart, selectionEnd, selectionDirection);
    };
}

/** Take a read of the settings: the servers become the drafts of their forms. A form that holds unsaved text keeps
 *  it while its saved entry is the one it was read from; when that entry changed or went elsewhere, the form is held
 *  by its saved identity until the owner discards it. The views on screen are drawn from the result. */
export function applyMcpSettings(settings) {
    const incoming = Array.isArray(settings?.MCP_SERVERS) ? settings.MCP_SERVERS : [];
    const token = settings?._meta?.tokens?.[KEY];
    const previousServers = mcpServers;
    const uniqueIncoming = id => incoming.filter(row => isServer(row) && canonicalServerId(serverIdentity(row)) === id).length === 1;
    const uniquePrevious = id => previousServers.filter(row => isServer(row) && pageId(row) === id).length === 1;
    for (const [id, server] of heldDrafts) {
        if (!dirty.has(server)) heldDrafts.delete(id);
    }
    // A held form is never adopted again by a read that merely looks like its old entry.
    const held = new Set(heldDrafts.values());
    const drafts = mcpServers.filter(server => isServer(server) && (dirty.has(server) || saving.get(server) === 'writing') && !held.has(server)
        && !savedMcpSecrets.get(server)?.submitted);
    const typed = new Map(drafts.map(server => [savedMcpSecrets.get(server).read, server]));
    NAMES.clear();
    // auth_configured belongs to Settings response metadata, not user config;
    // the legacy pair's masked token itself becomes a header row.
    mcpServers = incoming.map((entry, index) => {
        if (!entry || typeof entry !== 'object' || Array.isArray(entry)) return entry;
        const { auth_configured: _authConfigured, ...s } = entry;
        const id = canonicalServerId(serverIdentity(s));
        const read = asRead(entry);
        const previous = previousServers[index];
        const savedPrevious = savedMcpSecrets.get(previous);
        const unchanged = savedPrevious && savedPrevious.read === read
            && savedPrevious.enabled === Boolean(s.enabled) && !savedPrevious.submitted
            && !dirty.has(previous) && !held.has(previous) && !saving.has(previous);
        let kept = typed.get(read);
        if (!kept && unchanged) {
            if (savedPrevious.token === token) kept = previous;
            else if (editorView?.server === previous && editorShown(editorView) && id && savedPrevious.url !== HEADER_PLACEHOLDER
                && uniqueIncoming(id) && uniquePrevious(id)) {
                // Same mounted form and slot, not necessarily the same list revision. Refresh the clean write
                // baseline and put any revealed values away as a redraw would, without replacing form fields.
                resetSecretReveals(editorView.host);
                savedPrevious.token = token;
                kept = previous;
            }
        }
        const draft = heldDrafts.get(id);
        if (draft && savedMcpSecrets.get(draft).submitted && uniqueIncoming(id)) {
            const saved = savedMcpSecrets.get(draft);
            const sent = saved.submitted;
            const receipt = saved.token && saved.token === token;
            // A neighbour can change the list token. The public row can still match, but
            // masked values cannot settle overlapping edits to the headers themselves.
            const headersConfirmed = sent.unchanged || !sent.maskedHeaders || headerRevision(draft) === sent.headers.revision;
            if (receipt || (headersConfirmed && saved.read === read)) {
                kept = draft;
                heldDrafts.delete(id);
                if (cardMessages.get(draft)?.kind === 'conflict') sayNewerEdits(draft, {});
            }
        }
        const baseline = kept && savedMcpSecrets.get(kept);
        const submitted = baseline?.submitted;
        if (kept && submitted) {
            if (kept.id === submitted.inputId) kept.id = String(s.id || '');
            if (kept.url === submitted.inputUrl) kept.url = String(s.url || '');
            if (headerRevision(kept) === submitted.headers.revision) {
                kept.auth_token = String(s.auth_token || '');
                kept.auth_header = String(s.auth_header || 'Authorization');
                if ('headers' in s) kept.headers = s.headers; else delete kept.headers;
                resetHeaderRows(kept);
            } else settleHeaderRows(kept, submitted.headers, s);
        }
        if (kept && kept.enabled === baseline.enabled) kept.enabled = Boolean(s.enabled);
        const server = kept || {
            ...s, id: String(s.id ?? ''), name: String(s.name ?? ''), enabled: Boolean(s.enabled),
            transport: String(s.transport ?? 'streamable_http'), url: String(s.url ?? ''), command: String(s.command ?? ''),
            args: s.args ?? [], auth_header: String(s.auth_header ?? 'Authorization'), auth_token: String(s.auth_token ?? ''),
            allowed_tools: Array.isArray(s.allowed_tools) ? s.allowed_tools.map(String) : [], schemas_at_start: isPinned(s),
        };
        if (!kept || submitted) savedMcpSecrets.set(server, savedEntry(s, submitted ? baseline.token : token));
        else baseline.enabled = Boolean(s.enabled);
        if (pageId(server)) NAMES.set(pageId(server), serverName(server));
        return server;
    });
    loaded = true;
    for (const draft of drafts) if (!mcpServers.includes(draft)) heldDrafts.set(pageId(draft), draft);
    for (const draft of heldDrafts.values()) {
        setCardMessage(draft, conflictWords(draft), 'warn', 'conflict');
        if (editorView?.server === draft) paintEditorActions(editorView);
    }
    // A server's form is drawn again only while its page is the one on screen (section_levels.js onScreen): a read that
    // lands after the owner left or covered it leaves the form as it stands, and the shell's "shown" signal reads and
    // draws it when it comes back. The list is drawn either way: a Save or Remove lands on the list it returns to.
    if (editorView && editorView.entry !== 'new' && !editorView.savedAs && saving.get(editorView.server) !== 'writing' && editorShown(editorView)) {
        const [was, next] = [editorView.server, serverOfPage(editorView.entry)];
        if (next !== was || !next || editorView.boundEntry !== savedMcpSecrets.get(next)) {
            const before = previousServers.indexOf(was), after = mcpServers.indexOf(next);
            const id = editorView.entry;
            const restoreClientId = before >= 0 && after >= 0 && before !== after && uniquePrevious(id) && uniqueIncoming(id)
                && sameClientEntry(editorView, next) ? clientIdDraft(editorView.host) : null;
            editorView.server = next;
            drawEditor();
            restoreClientId?.();
        } else paintUse();
    }
    paintList();
}

function isServer(server) {
    return Boolean(server) && typeof server === 'object' && !Array.isArray(server);
}

/** One server as Save and Test send it; header rows decide headers and the legacy pair. */
function collectServer(s) {
    return {
        ...s, id: String(s.id || '').trim(), name: String(s.name || '').trim(), enabled: Boolean(s.enabled),
        transport: String(s.transport || 'streamable_http'), url: String(s.url || '').trim(), command: String(s.command || '').trim(),
        args: s.args ?? [], auth_header: String(s.auth_header || 'Authorization').trim() || 'Authorization',
        auth_token: String(s.auth_token || ''), allowed_tools: Array.isArray(s.allowed_tools) ? s.allowed_tools.map(String) : [],
        schemas_at_start: isPinned(s), ...credentialPayload(s),
    };
}

/** Every field an import Apply could overwrite or match on, raw header rows and order included. */
function draftFreshness() {
    return JSON.stringify(mcpServers.map((server) => {
        if (!isServer(server)) return server;
        const { enabled: _enabled, allowed_tools: _allowedTools, schemas_at_start: _pin, ...fields } = server;
        return [fields, headerDraftRows(server)];
    }));
}

/** The list as every form would save it now: a form's Save sends its one entry of this. */
export function collectMcpSettings() {
    return { MCP_SERVERS: mcpServers.map((s) => isServer(s) ? collectServer(s) : s) };
}
