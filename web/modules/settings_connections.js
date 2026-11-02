// Settings → Accounts → API keys: the key rows of each provider card (docs/DESIGN.md §6 account-row anatomy,
// §7 "Accounts"). A card holds its built-in key ("Default": today's flat settings, collected by settings.js
// with the other inputs) and the keys added under it, which are rows of OUROBOROS_CONNECTIONS: kind, a name
// derived once from the label and never changed, the label, `enabled`, the kind's fields. Turning a built-in
// key off writes the secret-free row `{kind, name: "default", enabled: false}`. GET masks a row's secrets and
// the save keeps an untouched mask by row identity, so a row travels with its fields as shown. A role names
// the key it uses: nothing here picks, orders or rotates keys.
import { apiClient } from './api_client.js';
import { adoptConnectionLabels } from './connection_labels.js';
import { API_PROVIDER_ORDER, apiConnections, apiProviderLabel, parseModelSource } from './route_editor_primitives.js';
import { setButtonBusy } from './settings_controls.js';
import { bindSecretInputs, resetSecretReveals } from './settings_secrets.js';
import { PROVIDER_CARDS, connectionRowHtml } from './settings_provider_cards.js';
import { revealNewRow, setInlineStatus } from './ui_helpers.js';

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
 * the roles by name ("Main", "Fallback 2"), then the Agents rows as a count ("3 subagents"). A role left
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
        used.get(source).push(`${count} subagent${count === 1 ? '' : 's'}`);
    }
    return used;
}

/** A row's status word and meta line: what is true of this key now (docs/DESIGN.md §6). */
export function connectionRowFacts({ draft = false, enabled = true, configured = false, source = '', usedBy = [] } = {}) {
    if (draft) return { tone: 'neutral', label: 'Not saved', meta: '' };
    const users = usedBy.join(', ');
    if (!enabled) return { tone: 'neutral', label: 'Disabled', meta: `${users ? `Named by ${users}` : 'Not used by any role'} · nothing is sent through it` };
    const use = users ? `used by ${users}` : 'not used by any role';
    if (!configured) return { tone: 'neutral', label: 'No key', meta: `${use[0].toUpperCase()}${use.slice(1)}` };
    return { tone: 'ok', label: 'Ready', meta: `${source === 'environment' ? 'Provided by environment' : 'Enabled'} · ${use}` };
}

/** A compatible endpoint's learned refusal of prompt-cache hints, when the server reports one; else ''. */
export function cacheHintsNote(entry) {
    const fact = entry?.cache_hints;
    if (!fact || fact.refused !== true) return '';
    const when = new Date(fact.learned_at || fact.since || '');
    return `Cache hints: refused by this server${Number.isFinite(when.getTime())
        ? ` since ${when.toLocaleDateString(undefined, { month: 'short', day: 'numeric' })}` : ''}`;
}

/** The save receipt's roles and rows that still name a removed or disabled key; '' when none.
 * A role is named in the owner's words ("Main", "Fallback 2", "Agents · reviewer-1"), the key by its picker label. */
const REFERENCE_ROLE_LABELS = Object.freeze({
    OUROBOROS_MODEL: 'Main', OUROBOROS_MODEL_LIGHT: 'Light', OUROBOROS_MODEL_SAFETY: 'Safety Supervisor',
    OUROBOROS_MODEL_VISION: 'Vision', OUROBOROS_MODEL_CONSCIOUSNESS: 'Consciousness',
    OUROBOROS_MODEL_FALLBACKS: 'Fallback', OUROBOROS_MODEL_FALLBACK: 'Fallback', OUROBOROS_WEBSEARCH_MODEL: 'Web search',
    OUROBOROS_MODEL_DEEP_SELF_REVIEW: 'Deep review', OUROBOROS_SUBAGENTS: 'Agents',
});

export function connectionReferencesText(references, connections = []) {
    const named = (Array.isArray(references) ? references : []).map((reference) => {
        if (typeof reference === 'string') return reference.trim();
        if (!reference || typeof reference !== 'object') return '';
        const setting = String(reference.setting || '').trim();
        const role = String(reference.label || REFERENCE_ROLE_LABELS[setting] || setting).trim();
        const entry = String(reference.entry || '').trim();
        const where = !entry ? role : setting === 'OUROBOROS_SUBAGENTS' ? `${role} · ${entry}` : `${role} ${entry}`;
        const route = String(reference.route || '').trim();
        const { source } = route ? parseModelSource(route) : { source: '' };
        const id = source && !source.startsWith('subscription:') ? source : String(reference.connection || '').trim();
        return [where, id ? `(${apiProviderLabel(id, {}, connections)})` : ''].filter(Boolean).join(' ');
    }).filter(Boolean);
    return named.length ? `Still naming a removed or disabled key, so they cannot send: ${named.join(', ')}` : '';
}

/** The "More providers" fold opens by itself only for a key set up inside it: a stored credential or an added key. */
export function moreProvidersOpen(connections = [], kinds = []) {
    return (connections || []).some((entry) => kinds.includes(entry.kind) && (entry.configured || !entry.builtin));
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

/** The Accounts API-key rows of one Settings page: load, draft edits, collect, validate, Test. */
export function createConnectionsEditor({ root, onChange = () => {}, test = (body) => apiClient.providerTest(body) } = {}) {
    const cards = new Map(PROVIDER_CARDS.filter((card) => root.querySelector(`[data-connection-rows="${card.kind}"]`))
        .map((card) => [card.kind, card]));
    const kinds = API_PROVIDER_ORDER.filter((kind) => cards.has(kind));
    const advanced = PROVIDER_CARDS.filter((card) => card.advanced).map((card) => card.kind);
    const panel = root.querySelector('[data-settings-panel="providers"]');
    let entries = [];
    let listed = [];            // the document's own `_meta.connections` entries (facts the picker does not read)
    let usage = new Map();
    let builtinOff = new Set();
    let originals = new Map();  // uid → the loaded row: fields this page does not show travel unchanged
    let foreign = [];           // loaded rows of a kind this page has no card for, kept as they are
    let baseline = '[]';        // the rows as loaded: the setting travels only when they differ
    let serial = 0;
    const touched = new Set();  // cards the owner opened or closed: a reload never overrides that
    const generations = new Map();
    const inFlight = new Set();
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
        const facts = connectionRowFacts({ draft: draft || rowFingerprint(row) !== rowBaselines.get(row), enabled, configured: Boolean(entry?.configured),
            source: entry?.source, usedBy: usage.get(id) || [] });
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
        // The built-in key's Remove clears its stored credential; a key the environment provides is turned off instead.
        remove.hidden = builtin && !(entry?.configured && entry.source !== 'environment');
        remove.title = builtin ? 'Clear this key when you save' : draft ? 'Discard this key' : 'Forget this key when you save';
        const note = cacheHintsNote(listed.find((item) => item?.id === id));
        Object.assign(row.querySelector('[data-connection-cache]'), { textContent: note, hidden: !note });
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
        setInlineStatus(row.querySelector('[data-connection-test-status]'), '', 'muted');
    }

    function load(settings = {}, { modelSlots = [] } = {}) {
        entries = apiConnections(settings);
        listed = Array.isArray(settings?._meta?.connections) ? settings._meta.connections : [];
        adoptConnectionLabels(settings?._meta?.connections);
        usage = connectionUsage(settings, modelSlots);
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
        const fold = root.querySelector('#settings-more-providers');
        if (fold && moreProvidersOpen(entries, advanced)) fold.open = true;
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

    // One short model request against the row's draft; only owner-edited fields travel, because a saved
    // secret arrives MASKED: an untouched field tests the saved value server-side, an emptied one the draft.
    async function testRow(row) {
        const card = cards.get(row.dataset.kind);
        const builtin = row.dataset.builtin === '1';
        const key = rowKey(row);
        if (inFlight.has(key)) return;
        const status = row.querySelector('[data-connection-test-status]');
        const button = row.querySelector('[data-connection-test]');
        const request = () => {
            const overrides = {};
            for (const field of card.fields) {
                const input = fieldInput(row, field);
                const value = (input?.value || '').trim();
                if (value !== (input?.dataset.appliedValue ?? '').trim()) overrides[builtin ? field.settingKey : field.rowField] = value;
            }
            return builtin ? { provider_id: card.kind, overrides }
                : { provider_id: card.kind, connection: names(card.kind).get(row), overrides };
        };
        const sent = request();
        const sentGeneration = generations.get(key) || 0;
        const current = () => row.isConnected && providerTestResultIsCurrent({ sentGeneration,
            currentGeneration: generations.get(key) || 0,
            sentFingerprint: JSON.stringify(sent), currentFingerprint: JSON.stringify(request()) });
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

    // A displayed verdict is only good for the draft it tested: any edit of the row clears it.
    function onInput(event) {
        const row = event.target instanceof Element ? event.target.closest('[data-connection-row]') : null;
        if (!row) return;
        expire(row);
        paintRow(row);
        onChange();
    }

    panel?.addEventListener('click', onClick);
    panel?.addEventListener('input', onInput);

    return {
        load,
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
