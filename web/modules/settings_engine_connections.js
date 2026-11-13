/** Settings › Accounts and keys › Remote engines (docs/DESIGN.md "Settings", §6 row anatomy): the rows of
 * OUROBOROS_ENGINE_CONNECTIONS (ouroboros/remote_engines.py). A row names a Claudexor engine already running on a host
 * this computer's OpenSSH reaches: its SSH destination, the engine's home there, the folder for staged run copies and the
 * folders a run may work in directly.
 *
 * - The Accounts list (settings_accounts.js) draws a head per saved engine from `engineHeads`: its name, its SSH
 *   destination, what this computer last observed of it, a stop Emergency Stop still owes there, the switch that takes it
 *   out of new work, and the door to its page. A removed connection still owed a stop gets the card's own line.
 * - `accounts/engine/<name>` and `accounts/engine/new` (`mountEnginePage`): the engine's fields with their own Save; Check
 *   and Disconnect act at once on the SAVED engine (a form with unsaved edits is checked once it is saved); Turn off, and
 *   Remove, which asks first and says what it closes.
 * - Models and Agents read what the saved engines serve themselves (`readRemoteEngines`): a connected engine's catalog
 *   and its own accounts, else its state alone. No other page seeds them.
 *
 * Every write is one entry through the row helper's `writeEntry` (a fresh read, that key's token); an entry that changed
 * since its form read it is shown, never written over. Nothing here starts, installs or stops an engine, and no token is
 * ever read or shown. */
import { apiClient } from './api_client.js';
import { openConfirmDialog } from './confirm_dialog.js';
import { connectionNameFor } from './settings_connections.js';
import { outcomeLine, readSettings, setButtonBusy, writeEntry } from './settings_controls.js';
import { setResult } from './ui_primitives.js';
import { escapeHtmlAttr as escapeHtml } from './utils.js';

export const ENGINE_CONNECTIONS_KEY = 'OUROBOROS_ENGINE_CONNECTIONS';

// The browser twins of ``ouroboros/gateway/management_contracts.py``'s remote engine shapes.
/**
 * This installation's last observation of one saved remote engine; never the engine's token.
 * @typedef {Object} EngineConnectionObservation
 * @property {string} state
 * @property {string=} at
 * @property {string=} code
 * @property {string=} message
 * @property {string=} reason
 * @property {string=} engine_version
 * @property {string=} store
 * @property {string=} connected_at
 * @property {?number=} transport_uid
 * @property {?number=} engine_owner_uid
 */
/**
 * Stops Emergency Stop still owes on one engine, until each is read back stopped.
 * @typedef {Object} EngineOwedStops
 * @property {number} runs
 * @property {number} starts
 */
/**
 * @typedef {Object} EngineConnectionRow
 * @property {string} name
 * @property {string} label
 * @property {string} ssh_host
 * @property {string} engine_home
 * @property {string} remote_root
 * @property {string[]} workspaces
 * @property {boolean} enabled
 * @property {EngineConnectionObservation} observation
 * @property {?EngineOwedStops} owed_stops
 * @property {boolean=} removed
 */
/**
 * GET /api/engine-connections (no host contacted).
 * @typedef {Object} EngineConnectionsResponse
 * @property {EngineConnectionRow[]} connections
 */
/**
 * POST /api/engine-connections/{name}/check: the saved row proven through its tunnel, or its typed refusal.
 * @typedef {Object} EngineConnectionCheckResponse
 * @property {string} name
 * @property {boolean} ok
 * @property {string=} code
 * @property {string=} message
 * @property {string=} engine_version
 * @property {string=} store
 * @property {boolean=} user_mismatch
 * @property {string[]=} harnesses
 * @property {string[]=} model_sources
 */
/**
 * POST /api/engine-connections/{name}/disconnect: this installation's forward only.
 * @typedef {Object} EngineConnectionDisconnectResponse
 * @property {string} name
 * @property {?boolean} closed
 */
/**
 * @typedef {Object} EngineCatalogModel
 * @property {string} id
 * @property {string} label
 * @property {string} value
 */
/**
 * @typedef {Object} EngineCatalogSource
 * @property {string} id
 * @property {string} label
 * @property {string} credentialHarness
 * @property {EngineCatalogModel[]} models
 */
/**
 * @typedef {Object} EngineCatalogAccount
 * @property {string} harness_id
 * @property {string} profile_id
 * @property {string} label
 * @property {boolean} enabled
 * @property {string} verification
 * @property {string} availability
 */
/**
 * @typedef {Object} EngineCatalogError
 * @property {string} source
 * @property {string} code
 * @property {string} message
 */
/**
 * GET /api/engine-connections/{name}/catalog: what a connected engine serves; offline is a state, never empty.
 * @typedef {Object} EngineConnectionCatalogResponse
 * @property {string} name
 * @property {string} state
 * @property {string=} code
 * @property {string=} message
 * @property {EngineCatalogSource[]=} sources
 * @property {string[]=} harnesses
 * @property {EngineCatalogError[]=} errors
 * @property {string=} accounts_state
 * @property {string=} accounts_code
 * @property {EngineCatalogAccount[]=} accounts
 */

/** What the section says behind its More (DESIGN "Settings": a paragraph waits behind More). */
export const ENGINE_DETAILS = 'Ouroboros only attaches: it reads the running engine through SSH and never installs, starts, '
    + 'updates or stops it. A model you put there (Main, for example) only answers there: its requests go to that engine '
    + 'under its own accounts, quotas and limits, while the tools it calls still run on this computer. An agent you put '
    + 'there runs on that host: its tools and files are there. That engine may serve other work too. An agent works on a '
    + 'copy of the project staged in the folder for run copies; its changes come back here when it ends, and the copy '
    + 'stays on the host until that run\'s retention ends. In a declared folder it works on the host directly: its changes '
    + 'stay in that folder, and only its work product and review evidence come back here. Emergency Stop on this '
    + 'computer closes the tunnel at once, so its stop may not reach that engine: the work there can continue until this '
    + 'computer reconnects (at the next start), when the stop is delivered and shown here until confirmed. A choice that '
    + 'names an engine is never moved to another engine, this computer\'s included: switched off, removed or unreachable, '
    + 'it is refused, and a task keeps the engine it began on. Fallback models you set are separate choices and apply as '
    + 'they do for any route. Before going back to an Ouroboros '
    + 'version without remote engines, choose local sources in Models and Agents; an older version refuses a remote '
    + 'choice, and with a remote Main it cannot answer until you pick another. The access token opens the whole engine; '
    + 'Ouroboros acts only on its own runs and never shows it.';

/** The saved rows: the setting's JSON list, or that list's text. */
export function engineConnectionRows(value) {
    let rows = value;
    if (typeof rows === 'string') {
        try { rows = JSON.parse(rows.trim() || '[]'); } catch { rows = []; }
    }
    return Array.isArray(rows) ? rows.filter((row) => row && typeof row === 'object' && !Array.isArray(row)) : [];
}

const owedCount = (owed) => Number(owed?.runs || 0) + Number(owed?.starts || 0);

/** The sentence for the stops Emergency Stop still owes on one engine ('' when none; `null` owed = unread). */
export function owedStopsLine(owed) {
    if (owed === null) return 'Whether Emergency Stop still owes stops here could not be read';
    const count = owedCount(owed);
    if (!count) return '';
    return `Emergency Stop: ${count} ${count === 1 ? 'run is' : 'runs are'} not yet confirmed stopped on this engine; `
        + 'the stop is delivered when it answers this computer, and the work may continue there until then';
}

/** The stops Emergency Stop still owes on connections no longer saved ('' when none): no tunnel reaches them, so
 *  each stop waits until a connection of that name with the same SSH destination and engine home is saved again. */
export function removedOwedLine(connections) {
    return (connections || []).filter((row) => row?.removed).map((row) => {
        const count = owedCount(row.owed_stops);
        return count ? `Emergency Stop: ${count} ${count === 1 ? 'run is' : 'runs are'} not yet confirmed stopped on `
            + `${row.name}, which is no longer saved; the stop is delivered only after a connection named ${row.name} `
            + 'with the same SSH destination and engine home is saved again, and the work may continue there until then.'
            : '';
    }).filter(Boolean).join(' ');
}

/** An engine's status word, tone, meta line and owed-stop line from this installation's observation (docs/DESIGN.md
 *  §6). A form with unsaved edits is labelled as such: what was checked is the saved engine, said in the meta. */
export function engineRowFacts({ draft = false, dirty = false, enabled = true, observation = {}, check = null,
    owed } = {}) {
    if (draft) return { tone: 'neutral', label: 'Not saved', meta: 'Save to check it', owed: '' };
    const seen = check || observation || {};
    const state = check ? (check.ok ? 'connected' : (check.code === 'remote_ssh_needs_interaction' ? 'needs_interaction' : 'failed'))
        : String(seen.state || 'unknown');
    const owedLine = owedStopsLine(owed);
    const join = (...parts) => parts.filter(Boolean).join(' · ');
    const off = enabled ? '' : 'Switched off: takes no new work';
    let facts;
    if (state === 'connected') {
        const user = check && check.user_mismatch ? 'SSH logs in as another user than the engine: staged runs refuse' : '';
        facts = { tone: user ? 'warn' : 'ok', label: 'Connected', meta: join(off,
            seen.engine_version ? `engine ${seen.engine_version}` : '', seen.store ? `store …${String(seen.store).slice(-6)}` : '', user) };
    } else if (state === 'needs_interaction') {
        facts = { tone: 'warn', label: 'Needs sign-in', meta: join(off, seen.message || 'OpenSSH needs a prompt answered; Ouroboros never answers one') };
    } else if (state === 'failed') {
        facts = { tone: 'error', label: 'Unavailable', meta: join(off, seen.message || seen.code || 'The last check failed') };
    } else if (state === 'offline') {
        facts = { tone: 'neutral', label: 'Not connected', meta: join(off, 'Check opens the tunnel; nothing is started on the host') };
    } else if (state === 'stop_unconfirmed') {
        facts = { tone: 'warn', label: 'Stop unconfirmed', meta: join(off, seen.message
            || 'This computer could not confirm its tunnel stopped; it keeps the process in its custody') };
    } else if (state === 'unread') {
        facts = { tone: 'warn', label: 'Status unknown', meta: join(off, seen.message || 'This computer could not read its status') };
    } else {
        facts = { tone: 'neutral', label: 'Not checked', meta: join(off, 'Check reads the running engine through SSH') };
    }
    if (dirty) {
        facts = { tone: 'neutral', label: 'Unsaved changes', meta: join(`Saved engine: ${facts.label.toLowerCase()}`,
            'Save, then Check reads the new fields') };
    }
    if (owedLine && facts.tone !== 'error') facts.tone = 'warn';
    return { ...facts, owed: owedLine };
}

/** The roster line from a connected engine's catalog: that engine's own accounts, or why they are unknown; '' while
 *  the engine is not connected (its state already says so). */
export function engineAccountsLine(catalog) {
    if (!catalog) return '';
    if (catalog.state === 'unread') return `What this engine serves could not be read (${catalog.message || 'error'})`;
    if (catalog.state !== 'connected') return '';
    if (catalog.accounts_state !== 'ok') return `Accounts there could not be read (${catalog.accounts_code || 'engine'})`;
    const rows = catalog.accounts || [];
    if (!rows.length) return 'No accounts on this engine';
    return `Accounts there: ${rows.map((account) => {
        const facts = [account.harness_id, account.enabled === false ? 'disabled' : '',
            account.verification && account.verification !== 'passed' ? 'not verified' : ''].filter(Boolean);
        return `${account.label || account.profile_id} (${facts.join(', ')})`;
    }).join(' · ')}`;
}

/* ---- The list and the paths -------------------------------------------------------------------------------------- */

/** The page of a saved engine (`accounts/engine/<name>`), or of a new one (`accounts/engine/new`). */
export const enginePath = (name = '') => `accounts/engine/${name ? encodeURIComponent(name) : 'new'}`;

/** `accounts/engine/<name>` as `{ name, fresh: false }`, `accounts/engine/new` as `{ name: '', fresh: true }`; null for
 *  neither. A new engine is never named `new` (`engineNameFor`). */
export function parseEnginePath(path) {
    const [family, word, name = '', ...rest] = String(path || '').split('/');
    if (family !== 'accounts' || word !== 'engine' || !name || rest.length) return null;
    if (name === 'new') return { name: '', fresh: true };
    try { return { name: decodeURIComponent(name), fresh: false }; } catch { return { name, fresh: false }; }
}

/** A new engine's stored name, minted once from its label like an added key's; never `default`, `local` or `new`. */
export const engineNameFor = (label, taken = []) => connectionNameFor(label, [...taken, 'local', 'new']);

const NAMES = new Map();  // name -> label, for a page's title before and after its read
/** A page's title: the engine's label, or "New remote engine". */
export function engineTitle(path) {
    const page = parseEnginePath(path);
    return !page ? 'Remote engine' : page.fresh ? 'New remote engine' : NAMES.get(page.name) || page.name;
}

/**
 * The heads of the Accounts list: each saved engine in order, with its SSH destination, the state this computer last
 * observed (`connections`: GET /api/engine-connections, null while unread) and, as a chip, what needs the owner: a stop
 * Emergency Stop still owes there, a sign-in, an unconfirmed stop, a failed check, an unread status.
 */
export function engineHeads(settings = {}, connections = null) {
    const observed = new Map((connections || []).map((row) => [row?.name, row]));
    return engineConnectionRows(settings?.[ENGINE_CONNECTIONS_KEY]).filter((row) => row.name).map((row) => {
        const status = observed.get(row.name);
        const enabled = row.enabled !== false;
        NAMES.set(row.name, row.label || row.name);
        const facts = connections ? engineRowFacts({ enabled, observation: status?.observation || {},
            owed: status ? status.owed_stops : undefined }) : null;
        const owed = owedCount(status?.owed_stops);
        const chip = owed ? { text: `${owed} ${owed === 1 ? 'stop' : 'stops'} owed`, tone: 'warn' }
            : facts && facts.tone === 'error' ? { text: facts.label, tone: 'danger' }
                : facts && facts.tone === 'warn' ? { text: facts.label, tone: 'warn' } : null;
        return { name: row.name, title: row.label || row.name, path: enginePath(row.name), enabled, chip,
            meta: [row.ssh_host, enabled ? (facts && !chip ? facts.label : '') : 'Off: takes no new work'].filter(Boolean).join(' · ') };
    });
}

/* ---- What Models and Agents offer -------------------------------------------------------------------------------- */

/**
 * What the saved remote engines serve, for a page that offers them (Models, Agents): every enabled saved engine with
 * its state, and a connected one's sources (each model's value the stored route), harnesses and own accounts, read
 * through its open tunnel; an engine that is not connected is asked nothing. A status that could not be read keeps the
 * saved rows of `settings`, said unread, never an empty list.
 * @returns {Promise<Array<Object>>}
 */
export async function readRemoteEngines({ settings = null, api = apiClient } = {}) {
    let connections;
    try {
        connections = (await api.engineConnections())?.connections || [];
    } catch (error) {
        const message = String(error?.message || error);
        return engineConnectionRows(settings?.[ENGINE_CONNECTIONS_KEY]).filter((row) => row.name && row.enabled !== false)
            .map((row) => ({ name: row.name, label: row.label || row.name, state: 'unread', message,
                workspaces: Array.isArray(row.workspaces) ? row.workspaces : [] }));
    }
    return Promise.all(connections.filter((row) => row?.name && !row.removed && row.enabled !== false).map(async (row) => {
        const base = { name: row.name, label: row.label || row.name, state: row.observation?.state || 'unknown',
            workspaces: Array.isArray(row.workspaces) ? row.workspaces : [] };
        if (base.state !== 'connected') return base;
        try {
            const catalog = await api.engineConnectionCatalog(row.name);
            return { ...base, state: catalog.state || base.state, sources: catalog.sources || [], harnesses: catalog.harnesses || [],
                accounts: catalog.accounts || [], accounts_state: catalog.accounts_state || 'unread',
                ...(catalog.accounts_code ? { accounts_code: catalog.accounts_code } : {}) };
        } catch (error) {
            return { ...base, state: 'unread', message: String(error?.message || error) };  // never "connected, serving nothing"
        }
    }));
}

/* ---- One engine's writes ----------------------------------------------------------------------------------------- */

const GONE = new Error('The engine changed since this page read it.');
const TAKEN = new Error('Another engine already has this name.');
const CHANGED = { state: 'changed_elsewhere', effect: 'now', withheld: '', message: '', code: '' };
const landed = (outcome) => outcome?.state === 'saved' || outcome?.state === 'saved_then_failed';

/** One write of one engine entry (`change(before)` answers the entry to store, null removes it): a fresh read with that
 *  key's token. An entry gone or changed since its page read it answers `changed_elsewhere`, never written over. */
async function engineWrite(name, change, label = '') {
    try {
        return await writeEntry(ENGINE_CONNECTIONS_KEY, name, change, { name: label || name, confirmed: true });
    } catch (error) {
        if (error === GONE) return CHANGED;
        if (error === TAKEN) return { state: 'refused', effect: 'now', withheld: '', code: 'taken',
            message: `Another remote engine is already named ${name}. Give this one another name.` };
        throw error;
    }
}

/** Switch a saved engine in or out of new work at once (its recorded work is reached either way). */
export function setEngineEnabled(name, on, label = '') {
    return engineWrite(name, (before) => {
        if (!before) throw GONE;
        const next = { ...before };
        if (on) delete next.enabled; else next.enabled = false;
        return next;
    }, label);
}

/** What a write says, in the row helper's one wording: when it applies, the server's warnings, a refusal. */
export function engineSaveWords(outcome, saved = 'Saved') {
    if (!outcome) return {};
    if (outcome.state === 'unchanged') return { text: 'Nothing to save.', tone: 'muted' };
    const line = outcomeLine({ id: ENGINE_CONNECTIONS_KEY, keys: [ENGINE_CONNECTIONS_KEY], control: { kind: 'custom' } },
        { ...outcome, old: undefined, withheld: outcome.withheld || 'structure' });
    if (!line) return landed(outcome) ? { text: saved, tone: 'ok' } : {};
    const text = landed(outcome) && saved !== 'Saved' ? line.text.replace(/^Saved/, saved) : line.text;
    return { text: line.note ? `${text}. ${line.note}` : text, tone: line.tone, details: line.details };
}

/* ---- One engine's page ------------------------------------------------------------------------------------------- */

let pageSerial = 0;
const FIELDS = [
    ['label', 'Name', 'e.g. Build box', ''],
    ['ssh_host', 'SSH destination', 'alias from ~/.ssh/config, or user@host',
        'Log in as the engine\'s own user; keys, agent and known hosts stay OpenSSH\'s.'],
    ['engine_home', 'Engine home on the host', '~/.claudexor/v3', 'Where the running engine keeps daemon/control-api.json.'],
    ['remote_root', 'Folder for run copies', '/home/you/ouroboros-runs',
        'Absolute. Each run from this computer gets a copy here; it stays on the host until that run\'s retention ends.'],
];

/**
 * One remote engine's page: `accounts/engine/<name>` or `accounts/engine/new`. `ctx` is the shell's (`readSettings`,
 * `ask`); `left(note, added, warning)` is called after the engine was added or removed, with the sentence for the list.
 * Answers `{ ready, unsaved, unmount }`: `ready` settles once the first read drew the page.
 */
export function mountEnginePage(container, { path, ctx = {}, left = () => {}, api = apiClient } = {}) {
    const page = parseEnginePath(path) || { name: '', fresh: true };
    const uid = `engine-${pageSerial += 1}`;
    const read = ctx.readSettings || readSettings;
    const ask = ctx.ask || openConfirmDialog;
    let settings = null;
    let asRead = '';        // the stored entry this form was drawn from: a Save over a newer one is refused
    let status = null;      // this computer's last observation: `{ observation, owed_stops }` of the status read
    let check = null;       // the last Check of the saved engine, until an edit or another read
    let catalog = null;     // a connected engine's catalog: its roster line
    let dirty = false;
    let busy = false;
    let gone = false;
    let said = null;
    let generation = 0;     // a read, a Check or a Disconnect answers only the page state it was asked for
    const find = (selector) => container.querySelector(selector);
    const stored = () => (page.fresh ? null
        : engineConnectionRows(settings?.[ENGINE_CONNECTIONS_KEY]).find((row) => row.name === page.name) || null);
    const say = (line) => { said = line; setResult(find('[data-editor-result]'), line || {}); };
    const typed = (key) => String(find(`[data-engine-field="${key}"]`)?.value || '').trim();

    const fieldHtml = ([key, label, placeholder, help]) => `<div class="form-field ui-field">
            <label for="${uid}-${key}">${label}</label>
            <input id="${uid}-${key}" class="ui-control" type="text" data-engine-field="${key}" placeholder="${escapeHtml(placeholder)}" autocomplete="off" spellcheck="false">
            ${help ? `<div class="ui-field-help">${help}</div>` : ''}</div>`;

    function draw() {
        const row = stored();
        asRead = JSON.stringify(row);
        if (!page.fresh && !row) {
            container.innerHTML = '<p class="settings-page-lead">This remote engine is no longer saved.</p>';
            return;
        }
        if (row) NAMES.set(row.name, row.label || row.name);
        const on = row?.enabled !== false;
        container.innerHTML = `<p class="settings-page-lead" data-engine-lead><span class="ui-status" data-engine-status></span>
                <span data-engine-meta></span></p>
            <p class="settings-card-note" data-engine-owed data-tone="warn" hidden></p>
            <section class="ui-card settings-card accounts-key" data-engine-form data-setting-row="${ENGINE_CONNECTIONS_KEY}">
                <div class="accounts-key-form">
                    ${FIELDS.map(fieldHtml).join('')}
                    <div class="form-field ui-field">
                        <label for="${uid}-workspaces">Folders runs may work in directly</label>
                        <textarea id="${uid}-workspaces" class="ui-control" rows="2" data-engine-field="workspaces" placeholder="/home/you/project (one per line)" spellcheck="false"></textarea>
                        <div class="ui-field-help">A run there changes that folder on the host, where its changes stay; only its work product and review evidence return here.</div>
                    </div>
                </div>
                <div class="ui-action-row">
                    <button type="button" class="btn btn-save" data-editor-save>Save</button>
                    ${page.fresh ? '' : `<button type="button" class="btn btn-default" data-engine-check>Check</button>
                    <button type="button" class="btn btn-default" data-engine-disconnect>Disconnect</button>`}
                    <span class="ui-result" data-editor-result role="status" aria-live="polite"></span>
                </div>
                <p class="settings-card-note" data-engine-accounts hidden></p>
                <p class="settings-card-note">${page.fresh ? 'Save it, then Check reads the running engine through SSH.'
        : 'Check reads the running engine through SSH; Disconnect closes this computer\'s tunnel to it. Neither starts or stops anything on the host.'}</p>
            </section>
            ${page.fresh ? '' : `<div class="ui-action-row settings-foot">
                <button type="button" class="btn btn-default" data-engine-toggle>${on ? 'Turn off' : 'Turn on'}</button>
                <button type="button" class="btn btn-danger" data-editor-remove>Remove</button></div>`}`;
        for (const [key] of FIELDS) find(`[data-engine-field="${key}"]`).value = String(row?.[key] || '');
        find('[data-engine-field="workspaces"]').value = Array.isArray(row?.workspaces) ? row.workspaces.join('\n') : '';
        paint();
        if (said) setResult(find('[data-editor-result]'), said);
    }

    // The state line, the owed stops and the roster, and what Check and Disconnect may do now. With unsaved edits the
    // line still says what the last Check found of the SAVED engine (engineRowFacts' "Saved engine: …").
    function paint() {
        const row = stored();
        if (!page.fresh && !row) return;
        const facts = engineRowFacts({ draft: page.fresh, dirty, enabled: row?.enabled !== false,
            observation: status?.observation || {}, check, owed: status ? status.owed_stops : undefined });
        const mark = find('[data-engine-status]');
        if (!mark) return;
        mark.textContent = facts.label;
        mark.dataset.tone = facts.tone;
        find('[data-engine-meta]').textContent = facts.meta ? `· ${facts.meta}` : '';  // "Connected · engine 5.1.0 · …"
        Object.assign(find('[data-engine-owed]'), { textContent: facts.owed, hidden: !facts.owed });
        const roster = dirty ? '' : engineAccountsLine(catalog);
        Object.assign(find('[data-engine-accounts]'), { textContent: roster, hidden: !roster });
        const checkButton = find('[data-engine-check]');
        if (checkButton) checkButton.disabled = busy || dirty;
        const disconnect = find('[data-engine-disconnect]');
        if (disconnect) disconnect.disabled = busy;
    }

    // The engine's status (no host contact) and, while it is connected, its catalog through the open tunnel.
    async function readStatus(asked) {
        if (page.fresh) return;
        try {
            const answer = await api.engineConnections();
            if (gone || asked !== generation) return;
            status = (answer?.connections || []).find((row) => row?.name === page.name) || { observation: { state: 'unknown' } };
        } catch (error) {
            if (gone || asked !== generation) return;
            status = { observation: { state: 'unread', message: String(error?.message || error) }, owed_stops: null };
        }
        catalog = status.observation?.state === 'connected' ? await readCatalog() : null;
        if (!gone && asked === generation) paint();
    }

    async function readCatalog() {
        try { return await api.engineConnectionCatalog(page.name); } catch (error) {
            return { state: 'unread', message: String(error?.message || error) };
        }
    }

    async function reload({ keep = false } = {}) {
        const asked = generation += 1;
        try { settings = await read(); } catch { return; }  // the form keeps what it shows
        if (gone || asked !== generation) return;
        if (!(keep && dirty)) { check = keep ? check : null; draw(); }
        await readStatus(asked);
    }

    // What this form would store: the saved entry under the fields as shown; an empty optional field is left out.
    function entryValue(name, before = null) {
        const workspaces = typed('workspaces').split('\n').map((line) => line.trim()).filter(Boolean);
        const value = { name, ...(typed('label') ? { label: typed('label') } : {}), ssh_host: typed('ssh_host'),
            ...(typed('engine_home') ? { engine_home: typed('engine_home') } : {}),
            ...(typed('remote_root') ? { remote_root: typed('remote_root') } : {}),
            ...(workspaces.length ? { workspaces } : {}) };
        return before?.enabled === false ? { ...value, enabled: false } : value;
    }

    /** What a Save would refuse here, as `{ field, message }`; null when it may go. */
    function refusal() {
        if (page.fresh && !typed('label')) return { field: 'label', message: 'Name the new remote engine.' };
        if (!typed('ssh_host')) return { field: 'ssh_host', message: 'Enter its SSH destination.' };
        return null;
    }

    async function save(button) {
        const refused = refusal();
        if (refused) {
            const input = find(`[data-engine-field="${refused.field}"]`);
            input?.setAttribute('aria-invalid', 'true');
            input?.focus?.();
            say({ text: `Not saved. ${refused.message}`, tone: 'danger' });
            return;
        }
        setButtonBusy(button, true);
        let outcome;
        let name = page.name;
        if (page.fresh) {
            name = engineNameFor(typed('label'), engineConnectionRows(settings?.[ENGINE_CONNECTIONS_KEY]).map((row) => row.name));
            const value = entryValue(name);
            outcome = await engineWrite(name, (before) => { if (before) throw TAKEN; return value; }, typed('label'));
        } else {
            const label = typed('label') || name;
            outcome = await engineWrite(name, (before) => {
                if (!before || JSON.stringify(before) !== asRead) throw GONE;  // changed since this form read it
                return entryValue(name, before);
            }, label);
        }
        setButtonBusy(find('[data-editor-save]'), false);
        if (gone) return;
        if (landed(outcome) && page.fresh) {
            NAMES.set(name, typed('label'));
            left(`Added ${typed('label')}.`, name, outcome.message || '');
            return;
        }
        if (landed(outcome) || outcome.state === 'changed_elsewhere') { dirty = false; await reload(); }
        say(engineSaveWords(outcome));
    }

    async function runCheck(button) {
        if (dirty || busy) return;
        const asked = generation;
        busy = true;
        paint();
        setButtonBusy(button, true);
        say({ text: 'Checking the saved engine…', tone: 'muted' });
        try {
            const result = await api.engineConnectionCheck(page.name);
            if (gone || asked !== generation) return;
            check = result;
            catalog = result.ok ? await readCatalog() : null;
            if (gone || asked !== generation) return;
            say(result.ok ? { text: `Connected: ${(result.model_sources || []).join(', ') || 'no model sources'}`
                + ` · agents: ${(result.harnesses || []).join(', ') || 'none'}`, tone: 'ok' }
                : { text: String(result.message || result.code || 'Check failed'), tone: 'danger' });
        } catch (error) {
            if (!gone && asked === generation) say({ text: `Not confirmed: ${error?.message || error}`, tone: 'danger' });
        } finally {
            busy = false;
            if (!gone) { setButtonBusy(find('[data-engine-check]'), false); paint(); }
        }
    }

    async function disconnect(button) {
        if (busy) return;
        const asked = generation;
        busy = true;
        paint();
        setButtonBusy(button, true);
        try {
            const result = await api.engineConnectionDisconnect(page.name);
            if (gone || asked !== generation) return;
            check = null;
            catalog = null;
            const unconfirmed = result?.closed === null || result?.closed === undefined;  // never read as closed
            status = { ...(status || {}), observation: { state: unconfirmed ? 'stop_unconfirmed' : 'offline' } };
            say(unconfirmed ? { text: 'Unconfirmed: this computer could not confirm its tunnel stopped; it keeps the process '
                + 'in its custody. The engine and its runs on the host continue.', tone: 'danger' }
                : { text: result.closed ? 'Disconnected: this computer\'s tunnel is closed; the engine and its runs on the host continue.'
                    : 'No tunnel was open; the engine and its runs on the host are as they were.', tone: 'muted' });
        } catch (error) {
            if (!gone && asked === generation) say({ text: `Not confirmed: ${error?.message || error}`, tone: 'danger' });
        } finally {
            busy = false;
            if (!gone) { setButtonBusy(find('[data-engine-disconnect]'), false); paint(); }
        }
    }

    // Unsaved edits are kept as typed (the form is not redrawn), so the switch finishes here: its button, and the
    // form's base, which moves with this page's own write when only the switch changed, so a later Save of the
    // draft is not refused as changed elsewhere.
    async function toggle(button) {
        const before = stored();
        const on = before?.enabled === false;
        setButtonBusy(button, true);
        const outcome = await setEngineEnabled(page.name, on, before?.label);
        if (gone) return;
        await reload({ keep: true });
        if (gone) return;
        const after = stored();
        const rest = (row) => JSON.stringify({ ...row, enabled: undefined });
        if (dirty && after && landed(outcome) && rest(after) === rest(before)) asRead = JSON.stringify(after);
        const current = find('[data-engine-toggle]');
        if (current) {
            setButtonBusy(current, false);
            current.textContent = after?.enabled === false ? 'Turn on' : 'Turn off';
        }
        say(engineSaveWords(outcome, on ? 'Turned on' : 'Turned off'));
    }

    async function remove(button) {
        const row = stored();
        const label = row?.label || page.name;
        const owed = owedStopsLine(status ? status.owed_stops : undefined);
        const yes = await ask({ title: `Remove ${label}?`, confirmLabel: 'Remove', danger: true,
            body: 'This computer\'s tunnel to it closes. Work already running on that host continues there, and choices in '
                + 'Models or Agents that name it are refused until you choose again.'
                + (owed ? ` ${owed}; removing it leaves those stops undelivered until it is saved again.` : '') });
        if (!yes || gone) return;
        setButtonBusy(button, true);
        const outcome = await engineWrite(page.name, (before) => {
            if (!before || JSON.stringify(before) !== asRead) throw GONE;
            return null;
        }, label);
        if (gone) return;
        if (landed(outcome)) { left(`Removed ${label}.`, '', outcome.message || ''); return; }
        setButtonBusy(find('[data-editor-remove]'), false);
        if (outcome.state === 'changed_elsewhere') await reload();
        say(engineSaveWords(outcome));
    }

    const onClick = (event) => {
        const hit = (selector) => event.target?.closest?.(selector);
        if (hit('[data-editor-save]')) void save(hit('[data-editor-save]'));
        else if (hit('[data-engine-check]')) void runCheck(hit('[data-engine-check]'));
        else if (hit('[data-engine-disconnect]')) void disconnect(hit('[data-engine-disconnect]'));
        else if (hit('[data-engine-toggle]')) void toggle(hit('[data-engine-toggle]'));
        else if (hit('[data-editor-remove]')) void remove(hit('[data-editor-remove]'));
    };
    // An edit ends what the last Check said: it spoke for the saved engine, not the fields as typed.
    const onInput = (event) => {
        if (!event.target?.matches?.('[data-engine-field]')) return;
        dirty = true;
        event.target.removeAttribute('aria-invalid');
        say(null);
        paint();
    };
    // Shown again (a return, a reconnect): the read follows, and typed fields stay as typed.
    const onShown = (event) => { if (event.detail?.path === path) void reload({ keep: true }); };
    container.addEventListener('click', onClick);
    container.addEventListener('input', onInput);
    globalThis.window?.addEventListener?.('ouro:settings-page-shown', onShown);
    container.innerHTML = '<p class="settings-page-lead">Reading the remote engine…</p>';
    const ready = reload();
    return { ready, unsaved: () => dirty, unmount() {
        gone = true;
        container.removeEventListener('click', onClick);
        container.removeEventListener('input', onInput);
        globalThis.window?.removeEventListener?.('ouro:settings-page-shown', onShown);
    } };
}
