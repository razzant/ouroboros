// API keys (docs/DESIGN.md §6 account-row anatomy, "Settings"). A provider holds its built-in key ("Default": today's
// flat settings) and the keys added beside it, which are rows of OUROBOROS_CONNECTIONS: kind, a name derived once
// from the label and never changed, the label, `enabled`, the kind's fields. Turning a built-in key off writes the
// secret-free row `{kind, name: "default", enabled: false}`. GET masks a row's secrets and the save keeps an
// untouched mask by row identity, so a row travels with its fields as shown. A role names the key it uses: nothing
// here picks, orders or rotates keys. Two hosts: setup draws every key of a provider as a row with its form in the
// provider's card and collects them with its other answers (`createConnectionsEditor`);
// Settings › Accounts and keys draws a head per key in its list and one key's form on that key's own page, and each
// of them writes at once (`keyHeads`, `mountKeyPage`, below).
import { apiClient } from './api_client.js';
import { adoptConnectionLabels, connectionLabel } from './connection_labels.js';
import { API_PROVIDER_ORDER, apiConnections, apiProviderLabel, compatibleEndpoint, connectionReferencesText, parseModelSource } from './route_editor_primitives.js';
import { CONFIRMS, bindSettingRows, commitSetting, outcomeLine, readSettings, setButtonBusy, settingsRow } from './settings_controls.js';
import { checkerFieldsNote, checkerReads, checkerUses, deploymentOf } from './settings_policy.js';
import { rowById } from './settings_rows.js';
import { bindSecretInputs, resetSecretReveals } from './settings_secrets.js';
import { PROVIDER_CARDS, connectionFieldsHtml, connectionRowHtml } from './settings_provider_cards.js';
import { revealNewRow, setInlineStatus } from './ui_helpers.js';
import { setResult } from './ui_primitives.js';
import { escapeHtmlAttr as escapeHtml } from './utils.js';

export const CONNECTIONS_KEY = 'OUROBOROS_CONNECTIONS';

/** The rows of the setting: its JSON list, or that list's text (an environment value). */
export function connectionRows(value) {
    let rows = value;
    if (typeof rows === 'string') {
        try { rows = JSON.parse(rows.trim() || '[]'); } catch { rows = []; }
    }
    return Array.isArray(rows) ? rows.filter((row) => row && typeof row === 'object' && !Array.isArray(row)) : [];
}

/** An added key's stored name, derived once from its label: lower-case words joined by `-`, unique in its card. */
export function connectionNameFor(label, taken = []) {
    const base = String(label || '').normalize('NFKD').replace(/[\u0300-\u036f]/g, '').toLowerCase()
        .replace(/[^a-z0-9]+/g, '-').replace(/^-+/, '').slice(0, 32).replace(/-+$/, '') || 'key';
    const used = new Set(['default', ...taken]);
    if (!used.has(base)) return base;
    for (let n = 2; ; n += 1) {
        const candidate = `${base.slice(0, 32 - String(n).length - 1).replace(/-+$/, '')}-${n}`;
        if (!used.has(candidate)) return candidate;
    }
}

/**
 * Which saved roles and Agents rows name each API source (`kind` or `kind@name`), in the owner's words:
 * the roles by name ("Main", "Fallback 2"), then the Agents rows as a count ("3 agents"). A role left
 * empty uses another role and names nothing itself.
 */
export function connectionUsage(settings = {}, slots = []) {
    const used = new Map();
    const subagents = new Map();
    const sourceOf = (value) => {
        const raw = String(value || '').trim();
        const { source } = parseModelSource(raw);
        return raw && !source.startsWith('subscription:') ? source : '';
    };
    const note = (value, label) => {
        const source = sourceOf(value);
        if (!source) return;
        if (!used.has(source)) used.set(source, []);
        used.get(source).push(label);
    };
    for (const slot of slots || []) {
        const value = settings?.[slot.settingKey];
        if (slot.slot !== 'fallback') { note(value, String(slot.label || slot.slot || '').replace(/ Model$/, '')); continue; }
        String(value || '').split(',').map((entry) => entry.trim()).filter(Boolean)
            .forEach((entry, index) => note(entry, `Fallback ${index + 1}`));
    }
    let roster = settings?.OUROBOROS_SUBAGENTS;
    if (typeof roster === 'string') {
        try { roster = JSON.parse(roster); } catch { roster = null; }
    }
    for (const row of Array.isArray(roster?.items) ? roster.items : []) {
        const source = row?.route?.kind === 'agent_session' ? '' : sourceOf(row?.route?.target_id);
        if (source) subagents.set(source, (subagents.get(source) || 0) + 1);
    }
    for (const [source, count] of subagents) {
        if (!used.has(source)) used.set(source, []);
        used.get(source).push(`${count} agent${count === 1 ? '' : 's'}`);
    }
    return used;
}

/** A row's status word and meta line: what is true of this key now (docs/DESIGN.md §6). `legacyAddress`:
 * the built-in compatible row is served by the legacy OPENAI_BASE_URL pair, which only Settings edits. */
export function connectionRowFacts({ draft = false, enabled = true, configured = false, source = '', usedBy = [], legacyAddress = '', fixed = false } = {}) {
    if (draft) return { tone: 'neutral', label: 'Not saved', meta: '' };
    const users = usedBy.join(', ');
    if (!enabled) return { tone: 'neutral', label: 'Disabled', meta: `${users ? `Named by ${users}` : 'Not used by any role'} · nothing is sent through it` };
    const use = users ? `used by ${users}` : 'not used by any role';
    const legacy = legacyAddress ? `Legacy OpenAI Base URL ${legacyAddress} with the OpenAI API key${fixed ? '' : ' · change the address in Settings → Accounts and keys'}` : '';
    if (!configured) return { tone: 'neutral', label: 'No key', meta: legacy ? `${legacy} · ${use}` : `${use[0].toUpperCase()}${use.slice(1)}` };
    return { tone: 'ok', label: 'Ready', meta: `${source === 'environment' ? 'Provided by environment' : legacy || 'Enabled'} · ${use}` };
}

/** The legacy address a document's built-in compatible connection reads, or '' while it has its own. */
function legacyAddressOf(settings) {
    const endpoint = compatibleEndpoint(settings);
    return endpoint.legacy ? endpoint.baseUrl : '';
}

/** A compatible endpoint's learned refusal of prompt-cache hints, when the server reports one; else ''. */
export function cacheHintsNote(entry) {
    const fact = entry?.cache_hints;
    if (!fact || fact.refused !== true) return '';
    const when = new Date(fact.learned_at || fact.since || '');
    return `Cache hints: refused by this server${Number.isFinite(when.getTime())
        ? ` since ${when.toLocaleDateString(undefined, { month: 'short', day: 'numeric' })}` : ''}`;
}

export function providerTestStatusText(result = {}) {
    if (result?.ok === true) return 'Works';
    const reason = String(result?.error || '').trim();
    return reason ? `Not ready — ${reason}` : 'Not ready';
}

/** A failed request reads "Not ready"; only the server's typed refusal adds its own controlled reason. */
export function providerTestNetworkErrorStatus(error) {
    const body = error?.status === 400 && error?.body;
    return body?.code === 'masked_key_endpoint_changed' ? providerTestStatusText(body) : 'Not ready';
}

export function providerTestResultIsCurrent({
    sentGeneration, currentGeneration, sentFingerprint, currentFingerprint,
} = {}) {
    return sentGeneration === currentGeneration && sentFingerprint === currentFingerprint;
}

/**
 * A built-in key's Test request from its changed fields. The built-in compatible key reaches one of two
 * pairs (`compatibleEndpoint` over the page's flat draft `flat`): the request names it in `api_key_setting`,
 * and the legacy pair, edited outside the row, travels as the page holds it (a kept key stays masked), so
 * the probe is the pair a save would run.
 */
export function builtinTestRequest(kind, overrides, flat = {}) {
    if (kind !== 'openai-compatible') return { provider_id: kind, overrides };
    const pair = compatibleEndpoint(flat);
    return { provider_id: kind, api_key_setting: pair.apiKeySetting,
        overrides: pair.legacy ? { ...overrides, OPENAI_BASE_URL: pair.baseUrl, OPENAI_API_KEY: pair.apiKey } : overrides };
}

/**
 * Whether the built-in compatible key's draft reaches another pair (`compatibleEndpoint`, the pair its Test sends)
 * than the settings whose facts its row shows: another address, another key as the page holds it (a kept key stays
 * masked) or the other pair. While it does, those facts are not about what the row would send.
 */
export function reachesAnotherPair(facts = {}, draft = {}) {
    return JSON.stringify(compatibleEndpoint(draft)) !== JSON.stringify(compatibleEndpoint(facts));
}

/** The Accounts API-key rows of one Settings page: load, draft edits, collect, validate, Test. */
export function createConnectionsEditor({ root, onChange = () => {}, test = (body) => apiClient.providerTest(body) } = {}) {
    const cards = new Map(PROVIDER_CARDS.filter((card) => root.querySelector(`[data-connection-rows="${card.kind}"]`))
        .map((card) => [card.kind, card]));
    const kinds = API_PROVIDER_ORDER.filter((kind) => cards.has(kind));
    const panel = root.querySelector('[data-settings-panel="providers"]');
    let entries = [];
    let listed = [];            // the document's own `_meta.connections` entries (facts the picker does not read)
    let usage = new Map();
    let slots = [];
    let pageSettings = {};      // the page's flat settings as last loaded or refreshed
    let legacyAddress = '';     // the built-in compatible row's legacy OPENAI_BASE_URL, when that pair serves it
    let builtinOff = new Set();
    let deployment = null;      // a managed deployment's checker connections are shown, never edited (settings_policy.js)
    let originals = new Map();  // uid → the loaded row: fields this page does not show travel unchanged
    let foreign = [];           // loaded rows of a kind this page has no card for, kept as they are
    let baseline = '[]';        // the rows as loaded: the setting travels only when they differ
    let serial = 0;
    const touched = new Set();  // cards the owner opened or closed: a reload never overrides that
    const generations = new Map();
    const inFlight = new Set();
    const asked = new WeakMap();  // row → the request its Test status answers (or awaits)
    let rowBaselines = new WeakMap();

    const cardNode = (kind) => root.querySelector(`[data-provider-card="${cards.get(kind)?.id}"]`);
    const rowsHost = (kind) => root.querySelector(`[data-connection-rows="${kind}"]`);
    const addedRows = (kind) => [...(rowsHost(kind)?.querySelectorAll('[data-connection-row][data-uid]') || [])];
    const rowKey = (row) => (row.dataset.builtin === '1' ? row.dataset.kind : row.dataset.uid);
    const fieldInput = (row, field) => row.querySelector(`[data-connection-field="${field.rowField}"]`);

    function setCardOpen(kind, open) {
        const node = cardNode(kind);
        if (!node) return;
        node.querySelector('[data-provider-toggle]')?.setAttribute('aria-expanded', String(open));
        const body = node.querySelector('.settings-provider-body');
        if (body) body.hidden = !open;
    }

    // A draft takes the first free name of its card, after every saved one, in the order shown.
    function names(kind) {
        const rows = addedRows(kind);
        const taken = rows.filter((row) => row.dataset.draft !== '1').map((row) => row.dataset.name);
        return new Map(rows.map((row) => {
            if (row.dataset.draft !== '1') return [row, row.dataset.name];
            const name = connectionNameFor(row.querySelector('[data-connection-label]')?.value, taken);
            taken.push(name);
            return [row, name];
        }));
    }

    function rowValue(row, name) {
        const card = cards.get(row.dataset.kind);
        const original = originals.get(row.dataset.uid) || {};
        const value = { ...original, kind: card.kind, name };
        const label = row.querySelector('[data-connection-label]')?.value.trim() || '';
        if (label || 'label' in original) value.label = label;
        const enabled = row.dataset.enabled !== '0';
        if (!enabled || 'enabled' in original) value.enabled = enabled;
        for (const field of card.fields) {
            const input = fieldInput(row, field);
            const text = field.secret ? input?.value || '' : (input?.value || '').trim();
            if (text || field.rowField in original) value[field.rowField] = text;
        }
        return value;
    }

    function collectRows() {
        const rows = [];
        for (const kind of kinds) {
            if (builtinOff.has(kind)) rows.push({ kind, name: 'default', enabled: false });
            for (const [row, name] of names(kind)) rows.push(rowValue(row, name));
        }
        return [...rows, ...foreign];
    }

    function paintRow(row) {
        const kind = row.dataset.kind;
        const card = cards.get(kind);
        const builtin = row.dataset.builtin === '1';
        const draft = row.dataset.draft === '1';
        const id = builtin ? kind : `${kind}@${row.dataset.name}`;
        const entry = draft ? null : entries.find((item) => item.id === id);
        const enabled = builtin ? !builtinOff.has(kind) : row.dataset.enabled !== '0';
        const legacy = builtin && kind === 'openai-compatible' ? legacyAddress : '';
        // The built-in compatible row's pair is also edited beside it (Settings' legacy address, the OpenAI key a
        // legacy pair travels with): the facts loaded or refreshed stand only while the draft still reaches their pair.
        const moved = builtin && kind === 'openai-compatible' && reachesAnotherPair(pageSettings, flatDraft());
        const changed = draft || moved || rowFingerprint(row) !== rowBaselines.get(row);
        const fixed = !draft && checkerUses(deployment, id);
        const facts = connectionRowFacts({ draft: changed, enabled, fixed,
            configured: Boolean(entry?.configured), source: entry?.source, usedBy: usage.get(id) || [], legacyAddress: legacy });
        const status = row.querySelector('[data-connection-status]');
        Object.assign(status, { textContent: facts.label, hidden: false });
        status.dataset.tone = facts.tone;
        const meta = row.querySelector('[data-connection-meta]');
        Object.assign(meta, { textContent: facts.meta, hidden: !facts.meta });
        if (!builtin) {
            const label = row.querySelector('[data-connection-label]')?.value.trim();
            row.querySelector('[data-connection-title]').textContent = label || (draft ? `New ${card.title} key` : row.dataset.name);
        }
        const toggle = row.querySelector('[data-connection-toggle]');
        Object.assign(toggle, { textContent: enabled ? 'Disable' : 'Enable', hidden: draft,
            title: enabled ? 'Keep the key; nothing is sent through it until you enable it again'
                : 'Let the roles that name this key send through it again' });
        const remove = row.querySelector('[data-connection-remove]');
        // The built-in key's Remove clears its stored credential; a key the environment provides is turned off
        // instead, and a legacy pair keeps nothing in this row to clear.
        remove.hidden = builtin && !(entry?.configured && entry.source !== 'environment' && !legacy);
        remove.title = builtin ? 'Clear this key when you save' : draft ? 'Discard this key' : 'Forget this key when you save';
        const note = changed ? '' : cacheHintsNote(listed.find((item) => item?.id === id));
        Object.assign(row.querySelector('[data-connection-cache]'), { textContent: note, hidden: !note });
        // The organization's Supervisor sends through this key: it keeps Test, and nothing here changes it. A built-in
        // key it only borrows a field of (the legacy compatible pair's OpenAI key) keeps its row: only that field is fixed.
        row.querySelector('[data-connection-fixed]').hidden = !fixed;
        const read = builtin && !fixed ? card.fields.filter((field) => checkerReads(deployment, field.settingKey)) : [];
        Object.assign(row.querySelector('[data-connection-fixed-fields]'), {
            textContent: checkerFieldsNote(read.map((field) => field.label)), hidden: !read.length });
        if (read.some((field) => field.secret || field.credential)) remove.hidden = true;
        const inputs = fixed ? [...row.querySelectorAll('.connection-fields input')] : read.map((field) => fieldInput(row, field));
        for (const input of inputs.filter(Boolean)) {
            input.readOnly = true;
            input.closest('.form-field')?.querySelector('.secret-clear')?.setAttribute('hidden', '');
        }
        if (fixed) { toggle.hidden = true; remove.hidden = true; }
    }

    function paintAll() {
        root.querySelectorAll('[data-connection-row]').forEach(paintRow);
    }

    // Field edits withdraw saved readiness without duplicating provider credential rules.
    // Toggle status remains independent, and restoring the loaded fields restores their facts.
    function rowFingerprint(row) {
        return JSON.stringify([row.querySelector('[data-connection-label]')?.value.trim() || '',
            ...cards.get(row.dataset.kind).fields.map((field) => {
                const input = fieldInput(row, field);
                return [field.secret ? input?.value || '' : (input?.value || '').trim(), input?.dataset.forceClear || ''];
            })]);
    }

    function expire(row) {
        const key = rowKey(row);
        generations.set(key, (generations.get(key) || 0) + 1);
        asked.delete(row);
        setInlineStatus(row.querySelector('[data-connection-test-status]'), '', 'muted');
    }

    function load(settings = {}, { modelSlots = [] } = {}) {
        slots = modelSlots;
        pageSettings = settings;
        deployment = deploymentOf(settings?._meta);
        entries = apiConnections(settings);
        listed = Array.isArray(settings?._meta?.connections) ? settings._meta.connections : [];
        adoptConnectionLabels(settings?._meta?.connections);
        usage = connectionUsage(settings, modelSlots);
        legacyAddress = legacyAddressOf(settings);
        const rows = connectionRows(settings?.[CONNECTIONS_KEY]);
        builtinOff = new Set([
            ...rows.filter((row) => row.name === 'default' && row.enabled === false).map((row) => row.kind),
            ...entries.filter((entry) => entry.builtin && !entry.enabled).map((entry) => entry.kind),
        ]);
        foreign = rows.filter((row) => !cards.has(row.kind));
        originals = new Map();
        for (const card of PROVIDER_CARDS) {
            const host = rowsHost(card.kind);
            if (!host) continue;
            addedRows(card.kind).forEach((row) => { resetSecretReveals(row); row.remove(); });
            const added = rows.filter((row) => row.kind === card.kind && row.name !== 'default');
            host.insertAdjacentHTML('beforeend', added.map((row) => {
                const uid = `${card.id}-${serial += 1}`;
                originals.set(uid, row);
                return connectionRowHtml(card, { uid, name: row.name, label: row.label, enabled: row.enabled !== false, values: row });
            }).join(''));
            addedRows(card.kind).forEach((row) => bindSecretInputs(row));
            if (added.length && !touched.has(card.kind)) setCardOpen(card.kind, true);
        }
        root.querySelectorAll('[data-connection-row]').forEach(expire);
        rowBaselines = new WeakMap([...root.querySelectorAll('[data-connection-row]')].map((row) => [row, rowFingerprint(row)]));
        paintAll();
        baseline = JSON.stringify(collectRows());
    }

    function addRow(kind) {
        const card = cards.get(kind);
        const host = rowsHost(kind);
        if (!card || !host) return;
        host.insertAdjacentHTML('beforeend', connectionRowHtml(card, { uid: `${card.id}-${serial += 1}`, draft: true }));
        const row = host.lastElementChild;
        bindSecretInputs(row);
        paintRow(row);
        setCardOpen(kind, true);
        revealNewRow(row, row.querySelector('[data-connection-label]'));
        onChange();
    }

    function toggleRow(row) {
        const kind = row.dataset.kind;
        if (row.dataset.builtin !== '1') row.dataset.enabled = row.dataset.enabled === '0' ? '1' : '0';
        else if (builtinOff.has(kind)) builtinOff.delete(kind);
        else builtinOff.add(kind);
        paintRow(row);
        onChange();
    }

    function removeRow(row) {
        const card = cards.get(row.dataset.kind);
        if (row.dataset.builtin === '1') {
            // The built-in key stays where it is typed; Remove clears what makes it usable.
            for (const field of card.fields.filter((item) => item.secret || item.credential)) {
                const input = fieldInput(row, field);
                if (!input) continue;
                input.value = '';
                if (field.secret) input.dataset.forceClear = '1';
                input.dispatchEvent(new Event('input', { bubbles: true }));
            }
            onChange();
            return;
        }
        // The caret moves to the key now in this place, else to the one before (docs/DESIGN.md "List editors").
        const rows = [...rowsHost(card.kind).querySelectorAll('[data-connection-row]')];
        const next = rows[rows.indexOf(row) + 1] || rows[rows.indexOf(row) - 1];
        resetSecretReveals(row);
        row.remove();
        if (next) revealNewRow(next, next.querySelector('input'));
        else cardNode(card.kind)?.querySelector('[data-connection-add]')?.focus();
        onChange();
    }

    // The page's flat draft: the settings last loaded or refreshed under the built-in rows' current fields,
    // so a pair edited partly outside its row (the OpenAI key of a legacy compatible pair) reads as typed.
    function flatDraft() {
        const values = { ...pageSettings };
        root.querySelectorAll('[data-connection-row][data-builtin="1"]').forEach((row) => {
            for (const field of cards.get(row.dataset.kind).fields) {
                const input = fieldInput(row, field);
                if (input) values[field.settingKey] = input.value;
            }
        });
        return values;
    }

    // One short model request against the row's draft; only owner-edited fields travel, because a saved
    // secret arrives MASKED: an untouched field tests the saved value server-side, an emptied one the draft.
    function testRequest(row) {
        const card = cards.get(row.dataset.kind);
        const builtin = row.dataset.builtin === '1';
        const overrides = {};
        for (const field of card.fields) {
            const input = fieldInput(row, field);
            const value = (input?.value || '').trim();
            if (value !== (input?.dataset.appliedValue ?? '').trim()) overrides[builtin ? field.settingKey : field.rowField] = value;
        }
        return builtin ? builtinTestRequest(card.kind, overrides, flatDraft())
            : { provider_id: card.kind, connection: names(card.kind).get(row), overrides };
    }

    // A Test status answers one request: an edit outside the row that changes it clears it too.
    function settle() {
        root.querySelectorAll('[data-connection-row]').forEach((row) => {
            if (asked.has(row) && asked.get(row) !== JSON.stringify(testRequest(row))) expire(row);
        });
    }

    async function testRow(row) {
        const key = rowKey(row);
        if (inFlight.has(key)) return;
        const status = row.querySelector('[data-connection-test-status]');
        const button = row.querySelector('[data-connection-test]');
        const sent = testRequest(row);
        const sentGeneration = generations.get(key) || 0;
        const current = () => row.isConnected && providerTestResultIsCurrent({ sentGeneration,
            currentGeneration: generations.get(key) || 0,
            sentFingerprint: JSON.stringify(sent), currentFingerprint: JSON.stringify(testRequest(row)) });
        asked.set(row, JSON.stringify(sent));
        inFlight.add(key);
        setButtonBusy(button, true);
        setInlineStatus(status, 'Testing…', 'muted');
        try {
            const data = await test(sent);
            if (current()) setInlineStatus(status, providerTestStatusText(data), data?.ok ? 'ok' : 'danger');
        } catch (error) {
            if (current()) setInlineStatus(status, providerTestNetworkErrorStatus(error), 'danger');
        } finally {
            inFlight.delete(key);
            setButtonBusy(button, false);
        }
    }

    function onClick(event) {
        const target = event.target instanceof Element ? event.target : null;
        const toggle = target?.closest('[data-provider-toggle]');
        if (toggle) {
            const kind = toggle.closest('[data-provider-card]')?.querySelector('[data-connection-add]')?.dataset.connectionAdd;
            touched.add(kind);
            setCardOpen(kind, toggle.getAttribute('aria-expanded') !== 'true');
            return;
        }
        const add = target?.closest('[data-connection-add]');
        if (add) { addRow(add.dataset.connectionAdd); return; }
        const row = target?.closest('[data-connection-row]');
        if (!row) return;
        if (target.closest('[data-connection-test]')) void testRow(row);
        else if (target.closest('[data-connection-toggle]')) toggleRow(row);
        else if (target.closest('[data-connection-remove]')) removeRow(row);
    }

    // A displayed verdict is only good for the draft it tested: any edit of the row clears it,
    // and so does an edit beside the rows that changes what it asked (Settings' legacy address).
    // Every row repaints: an edit beside a row, or in another one, can move the pair it describes.
    function onInput(event) {
        const row = event.target instanceof Element ? event.target.closest('[data-connection-row]') : null;
        if (row) expire(row);
        paintAll();
        settle();
        if (row) onChange();
    }

    panel?.addEventListener('click', onClick);
    panel?.addEventListener('input', onInput);

    return {
        load,
        /** Re-read a draft's facts (which keys it holds, who names them) without remounting: an edit outside
         * the rows, such as the OpenAI key a legacy pair travels with, repaints them; drafts and baselines stay. */
        refresh(settings = {}) {
            pageSettings = settings;
            entries = apiConnections(settings);
            usage = connectionUsage(settings, slots);
            legacyAddress = legacyAddressOf(settings);
            paintAll();
            settle();
        },
        /** The setting travels only when the rows differ from the loaded ones. */
        collect() {
            const rows = collectRows();
            return JSON.stringify(rows) === baseline ? {} : { [CONNECTIONS_KEY]: rows };
        },
        /** `{input, message}` per field a save would refuse; painted by the page after a save attempt. */
        validate() {
            const errors = [];
            for (const card of PROVIDER_CARDS) {
                const seen = new Set();
                for (const row of addedRows(card.kind)) {
                    const labelInput = row.querySelector('[data-connection-label]');
                    const label = labelInput.value.trim();
                    const named = label || row.dataset.name || '';
                    if (row.dataset.draft === '1' && !label) errors.push({ input: labelInput, message: `${card.title}: name the new key.` });
                    for (const field of card.fields.filter((item) => item.required)) {
                        const input = fieldInput(row, field);
                        if (!input.value.trim()) errors.push({ input, message: `${card.title} · ${named || 'new key'}: enter the ${field.label}.` });
                    }
                    if (named && seen.has(named.toLowerCase())) errors.push({ input: labelInput,
                        message: `${card.title}: two keys are named ${named}. Give each its own name.` });
                    seen.add(named.toLowerCase());
                }
            }
            return errors;
        },
        connections: () => entries,
        destroy() {
            panel?.removeEventListener('click', onClick);
            panel?.removeEventListener('input', onInput);
        },
    };
}

/* ---- Settings › Accounts and keys: a head per key in the list, one key on its own page -----------------------------
   settings_accounts.js composes the list and hands a key's page its body. A head's switch and a key's Save, Turn off
   and Remove each write at once, through the row helper (settings_controls.js commitSetting). A built-in key is its
   flat settings: only the fields that changed travel, and a kept secret never does. A key added beside it is one row
   of the list, written from a fresh read with that read's token: a list that changed meanwhile is refused and shown,
   never retried. A secret is never shown back by a save: it has no "was" and no Undo. */

const LIST_ROW = { id: CONNECTIONS_KEY, keys: [CONNECTIONS_KEY], page: 'accounts', name: 'API keys', control: { kind: 'custom' }, writer: 'generic' };
const cardOf = (kind) => PROVIDER_CARDS.find((card) => card.kind === kind) || null;
const landed = (outcome) => outcome?.state === 'saved' || outcome?.state === 'saved_then_failed';
const TESTS = new Map();         // key id → what its last Test here said, `{ text, tone }`; an edit of the key ends it
let keyWrites = Promise.resolve();  // one key write at a time: each starts from its own fresh read
let pageSerial = 0;

export const keyPath = (kind, name = '') => `accounts/key/${kind}${name ? `/${encodeURIComponent(name)}` : ''}`;
const keyId = (kind, name = '') => (name ? `${kind}@${name}` : kind);

/** `accounts/key/<kind>[/<name>]` as `{ kind, name }`, `accounts/key/new` as `{ fresh: true }`; null for neither. */
export function parseKeyPath(path) {
    const [family, word, kind = '', ...rest] = String(path || '').split('/');
    if (family !== 'accounts' || word !== 'key') return null;
    if (kind === 'new' && !rest.length) return { kind: '', name: '', fresh: true };
    let name = rest.join('/');
    try { name = decodeURIComponent(name); } catch { /* a name with a stray percent is read as written */ }
    return cardOf(kind) ? { kind, name, fresh: false } : null;
}

/** A key's name, in the list and as its page's title: its provider as the pickers name it, then the key's own name. */
const keyName = (kind, name = '') => `${apiProviderLabel(kind)} · ${name ? connectionLabel(keyId(kind, name)) : 'Default'}`;
export function keyTitle(path) {
    const key = parseKeyPath(path);
    return !key ? 'API key' : key.fresh ? 'New key' : keyName(key.kind, key.name);
}

/** Who names a key, as one line: its roles and agents, or that none does. */
const usedWords = (users = []) => (users.length ? users.join(', ') : 'No role uses it');

/**
 * The keys of a settings document as the list shows them: `keys`, a head for every key that exists (switched off or
 * not), and `more`, the provider cards that have none yet. A head says who names the key; its chip speaks only of what
 * stands in the way or was just proved: no key, the organization's key, or what the last Test here said.
 */
export function keyHeads(settings = {}, { tests = TESTS } = {}) {
    const deployment = deploymentOf(settings?._meta);
    const usage = connectionUsage(settings, settings?._meta?.setup_contract?.modelSlots || []);
    const legacy = legacyAddressOf(settings);
    adoptConnectionLabels(settings?._meta?.connections);
    const keys = [];
    const more = [];
    for (const entry of apiConnections(settings)) {
        const card = cardOf(entry.kind);
        if (!card) continue;
        if (entry.builtin && !entry.configured && entry.enabled) { more.push(card); continue; }
        const fixed = checkerUses(deployment, entry.id);
        const test = tests.get(entry.id);
        const from = entry.source === 'environment' ? 'From the environment'
            : entry.builtin && entry.kind === 'openai-compatible' && legacy ? `Legacy address ${legacy}` : '';
        keys.push({
            id: entry.id, kind: entry.kind, name: entry.builtin ? '' : entry.name, card, enabled: entry.enabled, fixed,
            path: keyPath(entry.kind, entry.builtin ? '' : entry.name),
            title: keyName(entry.kind, entry.builtin ? '' : entry.name),
            meta: [from, usedWords(usage.get(entry.id))].filter(Boolean).join(' · '),
            chip: fixed ? { text: 'Set by your organization', tone: 'neutral' }
                : test && test.tone !== 'ok' ? { text: test.text, tone: 'warn' }
                    : !entry.configured ? { text: 'No key', tone: 'neutral' } : test ? { text: test.text, tone: 'ok' } : null,
        });
    }
    return { keys, more };
}

/** An editor's result line for an Outcome: the shared sentences, and `Saved` with when it applies for a landed write
 *  (never the value: a key is a secret). */
export function editorLine(outcome, saved = 'Saved', connections = []) {
    if (!outcome) return {};
    if (outcome.state === 'unchanged') return { text: 'Nothing changed.', tone: 'muted' };
    const line = outcomeLine({ keys: [], control: { kind: 'secret' } }, { ...outcome, old: undefined, withheld: outcome.withheld || 'structure' });
    if (!line) return {};
    const text = outcome.state === 'saved' ? line.text.replace(/^Saved/, saved) : line.text;
    const words = { text: line.note ? `${text}. ${line.note}` : text, tone: line.tone, details: line.details };
    const references = referencesLine(outcome, connections);
    return references.text ? { ...words, text: `${words.text}. ${references.text}`, tone: 'warn' } : words;
}

/** A key saved, turned off or removed: the rows that still name a key which cannot send, as the save's answer lists
 *  them (`connection_references`), or nothing. */
export function referencesLine(outcome, connections = []) {
    const text = connectionReferencesText(outcome?.references, connections);
    return text ? { text, tone: 'warn' } : {};
}

/** Write the list of added keys from a fresh read: `change(rows, fresh)` answers the next list, or null when the key
 *  it meant is no longer there. One write at a time. */
function writeKeyList(change, { ask } = {}) {
    const run = async () => {
        let fresh;
        try { fresh = await readSettings(); } catch (error) { return { state: 'refused', effect: 'now', withheld: '', message: String(error?.message || error), code: 'settings_unread' }; }
        const next = change(connectionRows(fresh[CONNECTIONS_KEY]), fresh);
        if (!next) return { state: 'changed_elsewhere', effect: 'now', withheld: '', message: '', code: '' };
        return commitSetting(LIST_ROW, next, { confirmed: true, ifUnchanged: fresh._meta?.tokens?.[CONNECTIONS_KEY] || '', ask });
    };
    const queued = keyWrites.then(run);
    keyWrites = queued.catch(() => {});
    return queued;
}

/** Turn a key on or off at once. A built-in key keeps its fields: off is the secret-free row that names it. */
export function setKeyEnabled({ kind, name = '' }, enabled, { ask } = {}) {
    const stored = name || 'default';
    return writeKeyList((rows) => {
        const at = rows.findIndex((row) => row.kind === kind && row.name === stored);
        if (name) return at < 0 ? null : rows.map((row, i) => (i === at ? { ...row, enabled: Boolean(enabled) } : row));
        const rest = rows.filter((_, i) => i !== at);
        return enabled ? rest : [...rest, { kind, name: 'default', enabled: false }];
    }, { ask });
}

/**
 * Mount one key's page: `accounts/key/<kind>` (the built-in key), `accounts/key/<kind>/<name>` (a key added beside
 * it) or `accounts/key/new`. Its form is the key's own fields with Test and its own Save; Turn off and Remove stand
 * under it, and Remove asks first. `ctx` is the shell's (`readSettings`, `openSettings`, `ask`); `test` sends one
 * short model request; `left(note, added, warning)` is called after the key was added or removed, with the sentence
 * for the list and the roles that still name a key which cannot send.
 */
export function mountKeyPage(container, { path, ctx = {}, test = (body) => apiClient.providerTest(body), left = () => {} } = {}) {
    const key = parseKeyPath(path) || { kind: '', name: '', fresh: true };
    const uid = `key-${pageSerial += 1}`;
    const read = ctx.readSettings || readSettings;
    let settings = null;
    let asRead = '';        // the stored row this form was drawn from: a save over a newer one is refused
    let kind = key.kind || PROVIDER_CARDS[0].kind;
    let rows = null;
    let generation = 0;     // an edit ends the Test in flight
    let gone = false;
    let said = null;        // the result line, kept across a redraw
    const builtin = !key.fresh && !key.name;
    const id = () => keyId(kind, key.name);
    const card = () => cardOf(kind);
    const find = (selector) => container.querySelector(selector);
    const fieldInput = (field) => find(`[data-connection-field="${field.rowField}"]`);
    const typed = (field) => { const input = fieldInput(field); return field.secret ? input?.value || '' : (input?.value || '').trim(); };
    const changed = (field) => { const input = fieldInput(field); return input?.dataset.forceClear === '1' || typed(field) !== (input?.dataset.appliedValue ?? ''); };
    const say = (line) => { said = line; setResult(find('[data-editor-result]'), line || {}); };
    // A built-in key's write: the flat settings of the fields that travel, as one row of the helper.
    const flatRow = (fields) => ({ id: `accounts:key:${kind}`, keys: fields.map((field) => field.settingKey), page: 'accounts',
        name: card().title, control: { kind: 'custom' }, writer: 'generic' });
    const storedRow = () => (key.name ? connectionRows(settings?.[CONNECTIONS_KEY]).find((row) => row.kind === kind && row.name === key.name) || null : null);
    const keys = () => apiConnections(settings || {});     // the read's keys, so a sentence names a key by its label
    const entry = () => keys().find((item) => item.id === id()) || null;

    function draw() {
        resetSecretReveals(container);
        rows?.dispose();
        rows = null;
        const stored = storedRow();
        asRead = JSON.stringify(stored);
        if (key.name && !stored) {
            container.innerHTML = '<p class="settings-page-lead">This key is no longer saved.</p>';
            return;
        }
        const deployment = deploymentOf(settings?._meta);
        const fixed = !key.fresh && checkerUses(deployment, id());
        const on = entry()?.enabled !== false;
        const usage = connectionUsage(settings, settings?._meta?.setup_contract?.modelSlots || []);
        const form = connectionFieldsHtml(card(), builtin ? {} : { uid, name: key.name, label: stored?.label, values: stored || {}, draft: key.fresh });
        const provider = key.fresh ? `<div class="form-field ui-field"><label for="${uid}-kind">Provider</label><select id="${uid}-kind" class="ui-control" data-key-kind>${
            PROVIDER_CARDS.map((each) => `<option value="${each.kind}"${each.kind === kind ? ' selected' : ''}>${escapeHtml(each.title)}</option>`).join('')}</select></div>` : '';
        const legacy = builtin && kind === 'openai' && rowById('OPENAI_BASE_URL');
        const legacyFixed = legacy && checkerReads(deployment, legacy.id);
        const legacyRow = legacyFixed ? { ...legacy, meta: checkerFieldsNote(['Legacy OpenAI Base URL']) } : legacy;
        container.innerHTML = `<p class="settings-page-lead">${key.fresh ? 'A second key of a provider, under a name of its own. A role or an agent names the key it uses.'
            : `${escapeHtml(usedWords(usage.get(id())))}.${on ? '' : ' It is off: nothing is sent through it.'}`}</p>
            <section class="ui-card settings-card accounts-key" data-connection-row data-kind="${kind}"${builtin ? ` data-builtin="1" data-setting-row="accounts:key:${kind}"` : ` data-uid="${uid}" data-name="${escapeHtml(key.name)}"${key.fresh ? ' data-draft="1"' : ''}`}>
                <div class="accounts-key-form">${provider}${form}</div>
                <p class="settings-card-note" data-key-fixed${fixed ? '' : ' hidden'}>Set by your organization: its Safety Supervisor sends through this key.</p>
                <p class="settings-card-note" data-connection-fixed-fields hidden></p>
                <div class="ui-action-row">
                    ${fixed ? '' : '<button type="button" class="btn btn-save" data-editor-save>Save</button>'}
                    <button type="button" class="btn btn-default" data-connection-test>Test</button>
                    <span class="ui-result" data-editor-result role="status" aria-live="polite"></span>
                </div>
                <p class="settings-card-note">Test sends one short model request. Provider charges may apply.</p>
                ${card().note ? `<p class="settings-card-note">${card().note}</p>` : ''}
            </section>
            ${legacy ? `<section class="ui-card settings-card"><details class="ui-disclosure" data-settings-fold><summary>More: legacy base address</summary>
                <div class="settings-rows">${settingsRow(legacyRow)}</div></details></section>` : ''}
            ${key.fresh || fixed ? '' : `<div class="ui-action-row settings-foot">
                <button type="button" class="btn btn-default" data-key-toggle>${on ? 'Turn off' : 'Turn on'}</button>
                ${builtin && (card().fields.some((field) => checkerReads(deployment, field.settingKey)) || !(entry()?.configured && entry().source !== 'environment' && !(kind === 'openai-compatible' && legacyAddressOf(settings))))
        ? '' : '<button type="button" class="btn btn-danger" data-editor-remove>Remove</button>'}</div>`}`;
        for (const field of card().fields) {
            const input = fieldInput(field);
            if (!input) continue;
            // The fields show what is stored: a secret as its mask, which never travels unless it is replaced.
            const value = String((builtin ? settings?.[field.settingKey] : stored?.[field.rowField]) ?? '');
            Object.assign(input, { value }).dataset.appliedValue = value;
            const held = fixed || (builtin && checkerReads(deployment, field.settingKey));
            input.readOnly = held;
            if (held) input.closest('.form-field')?.querySelector('.secret-clear')?.setAttribute('hidden', '');
        }
        const borrowed = builtin && !fixed ? card().fields.filter((field) => checkerReads(deployment, field.settingKey)) : [];
        Object.assign(find('[data-connection-fixed-fields]'), { textContent: checkerFieldsNote(borrowed.map((field) => field.label)), hidden: !borrowed.length });
        if (fixed) container.querySelectorAll('.accounts-key-form input').forEach((input) => { input.readOnly = true; });
        // A key with one secret has nothing to clear that Remove does not: its field keeps Show alone.
        if (card().fields.filter((field) => field.secret).length < 2) container.querySelectorAll('.secret-clear').forEach((button) => { button.hidden = true; });
        bindSecretInputs(container);
        if (legacy) rows = bindSettingRows(find('[data-settings-fold]'), { rows: [legacyRow], ask: ctx.ask });
        if (legacyFixed) find('[data-settings-fold] [data-setting-control]').readOnly = true;
        if (said) setResult(find('[data-editor-result]'), said);
    }

    // What this form would send as one row of the list: the stored row under the fields as shown.
    function rowValue(name) {
        const original = storedRow() || {};
        const value = { ...original, kind, name };
        const label = find('[data-connection-label]')?.value.trim() || '';
        if (label || 'label' in original) value.label = label;
        for (const field of card().fields) {
            const text = typed(field);
            if (text || field.rowField in original) value[field.rowField] = text;
        }
        return value;
    }

    // The page's flat draft: the document under the built-in key's fields as typed (the pair its Test would reach).
    function flatDraft() {
        const values = { ...settings };
        if (builtin) for (const field of card().fields) values[field.settingKey] = fieldInput(field)?.value ?? '';
        return values;
    }

    // Only owner-edited fields travel: a saved secret arrives masked, so an untouched field tests the saved value.
    function testRequest() {
        const overrides = {};
        for (const field of card().fields) if (changed(field)) overrides[builtin ? field.settingKey : field.rowField] = typed(field).trim();
        return builtin ? builtinTestRequest(kind, overrides, flatDraft())
            : { provider_id: kind, connection: key.name || connectionNameFor(find('[data-connection-label]')?.value, takenNames()), overrides };
    }
    const takenNames = () => connectionRows(settings?.[CONNECTIONS_KEY]).filter((row) => row.kind === kind).map((row) => row.name);

    async function runTest() {
        const button = find('[data-connection-test]');
        const sent = generation;
        setButtonBusy(button, true);
        say({ text: 'Testing…', tone: 'muted' });
        let verdict;
        try {
            const data = await test(testRequest());
            verdict = { text: providerTestStatusText(data), tone: data?.ok ? 'ok' : 'danger' };
        } catch (error) { verdict = { text: providerTestNetworkErrorStatus(error), tone: 'danger' }; }
        setButtonBusy(find('[data-connection-test]'), false);
        if (gone || sent !== generation) return;
        if (!key.fresh) TESTS.set(id(), verdict);
        say(verdict);
    }

    /** What a save would refuse here, as `{ input, message }`; null when it may go. */
    function refusal() {
        const label = find('[data-connection-label]');
        if (key.fresh && !label.value.trim()) return { input: label, message: `${card().title}: name the new key.` };
        for (const field of builtin ? [] : card().fields.filter((item) => item.required)) {
            if (!typed(field).trim()) return { input: fieldInput(field), message: `${card().title}: enter the ${field.label}.` };
        }
        const named = label?.value.trim().toLowerCase();
        const twin = named && connectionRows(settings?.[CONNECTIONS_KEY]).some((row) => row.kind === kind && row.name !== key.name
            && String(row.label || row.name || '').trim().toLowerCase() === named);
        return twin ? { input: label, message: `${card().title}: two keys are named ${label.value.trim()}. Give each its own name.` } : null;
    }

    async function save() {
        const refused = refusal();
        if (refused) {
            refused.input?.setAttribute('aria-invalid', 'true');
            refused.input?.focus?.();
            say({ text: `Not saved. ${refused.message}`, tone: 'danger' });
            return;
        }
        const button = find('[data-editor-save]');
        setButtonBusy(button, true);
        let outcome;
        let name = key.name;
        if (builtin) {
            const fields = card().fields.filter(changed);
            const values = fields.map((field) => (fieldInput(field).dataset.forceClear === '1' ? '' : typed(field)));
            outcome = !fields.length ? { state: 'unchanged' } : await commitSetting(flatRow(fields), fields.length > 1 ? values : values[0], { confirmed: true, ask: ctx.ask });
        } else {
            const wanted = rowValue(name);
            outcome = await writeKeyList((list) => {
                const at = list.findIndex((row) => row.kind === kind && row.name === key.name);
                if (key.fresh) {
                    name = connectionNameFor(wanted.label, list.filter((row) => row.kind === kind).map((row) => row.name));
                    return [...list, { ...wanted, name }];
                }
                // A row that changed since this form read it is shown, never written over.
                if (at < 0 || JSON.stringify(list[at]) !== asRead) return null;
                return list.map((row, i) => (i === at ? wanted : row));
            }, { ask: ctx.ask });
        }
        setButtonBusy(find('[data-editor-save]'), false);
        if (gone) return;
        if (landed(outcome) && key.fresh) {
            left(`Added ${card().title} · ${find('[data-connection-label]').value.trim()}.`, keyId(kind, name), referencesLine(outcome, keys()).text || '');
            return;
        }
        if (landed(outcome) || outcome.state === 'changed_elsewhere') await reload();
        say(editorLine(outcome, 'Saved', keys()));
    }

    async function toggle() {
        const button = find('[data-key-toggle]');
        const on = entry()?.enabled !== false;
        setButtonBusy(button, true);
        const outcome = await setKeyEnabled({ kind, name: key.name }, !on, { ask: ctx.ask });
        if (gone) return;
        await reload();
        say(landed(outcome) ? referencesLine(outcome, keys()) : editorLine(outcome, 'Saved', keys()));
    }

    async function remove() {
        const title = keyTitle(path);
        if (!(await CONFIRMS.removeEntry({ name: title }, null, null, ctx.ask))) return;
        const button = find('[data-editor-remove]');
        setButtonBusy(button, true);
        let outcome;
        if (builtin) {
            // The built-in key stays where it is typed: Remove clears what makes it usable.
            const fields = card().fields.filter((field) => field.secret || field.credential);
            outcome = await commitSetting(flatRow(fields), fields.length > 1 ? fields.map(() => '') : '', { confirmed: true, ask: ctx.ask });
        } else {
            outcome = await writeKeyList((list) => (list.some((row) => row.kind === kind && row.name === key.name)
                ? list.filter((row) => !(row.kind === kind && row.name === key.name)) : null), { ask: ctx.ask });
        }
        if (gone) return;
        TESTS.delete(id());
        if (landed(outcome)) { left(`Removed ${title}.`, '', referencesLine(outcome, keys()).text || ''); return; }
        setButtonBusy(find('[data-editor-remove]'), false);
        if (outcome.state === 'changed_elsewhere') await reload();
        say(editorLine(outcome, 'Saved', keys()));
    }

    async function reload() {
        try { settings = await read(); } catch { return; }  // the form keeps what it shows
        adoptConnectionLabels(settings?._meta?.connections);
        if (!gone) draw();
    }

    const onClick = (event) => {
        const hit = (selector) => event.target?.closest?.(selector);
        if (hit('[data-connection-test]')) void runTest();
        else if (hit('[data-editor-save]')) void save();
        else if (hit('[data-key-toggle]')) void toggle();
        else if (hit('[data-editor-remove]')) void remove();
    };
    // An edit ends what the last Test said: the verdict was about the draft it tested.
    const onInput = (event) => {
        if (!event.target?.closest?.('.accounts-key-form')) return;
        generation += 1;
        TESTS.delete(id());
        event.target.removeAttribute?.('aria-invalid');
        say(null);
    };
    const onChange = (event) => {
        if (!event.target?.matches?.('[data-key-kind]')) return;
        kind = event.target.value;
        draw();
        find('[data-key-kind]')?.focus?.();
    };
    container.addEventListener('click', onClick);
    container.addEventListener('input', onInput);
    container.addEventListener('change', onChange);
    container.innerHTML = '<p class="settings-page-lead">Reading the key…</p>';
    const ready = reload();
    return { ready, unmount() {
        gone = true;
        rows?.dispose();
        resetSecretReveals(container);
        container.removeEventListener('click', onClick);
        container.removeEventListener('input', onInput);
        container.removeEventListener('change', onChange);
    } };
}
