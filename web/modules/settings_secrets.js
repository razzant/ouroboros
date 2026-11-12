import { apiClient } from './api_client.js';
import { CONFIRMS, collectCustomSecretDraft, commitSetting, outcomeLine, readCustomSecretDraft, readSettings, setButtonBusy } from './settings_controls.js';
import { deploymentOf, organizationSettingOwner } from './settings_policy.js';
import { setResult } from './ui_primitives.js';
import { escapeHtmlAttr as escapeHtml } from './utils.js';

// Controllers belong to mounted fields. The disclosed bytes live only in their
// text node, never in a Settings draft, applied snapshot or credential cache.
const reveals = new WeakMap();

export function resetSecretReveals(root) {
    root.querySelectorAll('[data-secret-reveal]').forEach((input) => reveals.get(input)?.reset());
}

// The toggle's words. Settings shows this English and the page overlay translates it; a surface the
// overlay never walks passes its own localized `copy` (secret_hitl.js, through `tr`).
const REVEAL_COPY = Object.freeze({ show: 'Show', hide: 'Hide' });

export function bindSecretReveal(input, button, { savedSelector, savedLabel = () => '', identityInputs = [],
    copy = REVEAL_COPY, saved = null } = {}) {
    const existing = reveals.get(input);
    if (existing) return existing;
    const row = input.closest('.secret-input-row');
    const doc = input.ownerDocument;
    const preview = doc.createElement('div');
    preview.id = `${input.id}-reveal`;
    preview.className = 'settings-secret-value ui-control';
    preview.tabIndex = 0;
    preview.setAttribute('role', 'region');
    const label = input.labels?.[0];
    if (label) {
        label.id ||= `${input.id}-label`;
        preview.setAttribute('aria-labelledby', label.id);
        button.setAttribute('aria-describedby', label.id);
    } else {
        preview.setAttribute('aria-label', input.getAttribute('aria-label') || 'Secret value');
    }
    const source = doc.createElement('div');
    source.id = `${input.id}-reveal-source`;
    source.className = 'settings-secret-source ui-field-help';
    const status = doc.createElement('div');
    status.className = 'settings-secret-status settings-inline-status';
    status.dataset.secretRevealStatus = '1';
    status.setAttribute('role', 'status');
    // The shown value takes the masked input's own place in its row; the input
    // steps aside untouched, so the draft and its value never change.
    input.after(preview);
    row.append(source, status);
    button.setAttribute('aria-controls', preview.id);
    input.dataset.secretReveal = '1';
    let generation = 0;
    let open = false;

    function reset() {
        generation += 1;
        open = false;
        preview.textContent = '';
        preview.hidden = true;
        input.hidden = false;
        source.textContent = '';
        source.hidden = true;
        preview.removeAttribute('aria-describedby');
        status.textContent = '';
        status.hidden = true;
        button.textContent = copy.show;
        button.setAttribute('aria-expanded', 'false');
        button.removeAttribute('aria-busy');
    }

    async function toggle() {
        if (open) { reset(); return; }
        reset();
        open = true;
        const request = generation;
        const value = input.value;
        const applied = input.dataset.appliedValue;
        const identities = identityInputs.map((node) => node.value);
        const current = () => open && generation === request && input.isConnected
            && value === input.value && applied === input.dataset.appliedValue
            && identityInputs.every((node, index) => node.value === identities[index]);
        button.textContent = copy.hide;
        try {
            let disclosed = value;
            // The saved value is read when the field shows it untouched (its mask), or, for a field that never holds
            // it, when the caller says a value is saved and nothing was typed over it.
            if (saved ? saved() : value && value === applied && input.dataset.forceClear !== '1') {
                const selector = savedSelector?.();
                if (!selector) throw new Error('The saved value has no lookup identity. Enter a new value.');
                const sourceLabel = savedLabel();
                button.setAttribute('aria-busy', 'true');
                status.textContent = 'Loading…';
                status.dataset.tone = 'muted';
                status.hidden = false;
                const response = await apiClient.revealSettingsSecret(selector);
                if (!current()) return;
                if (typeof response?.value !== 'string') throw new Error('The server did not return a secret value.');
                disclosed = response.value;
                source.textContent = sourceLabel;
                source.hidden = !sourceLabel;
                if (sourceLabel) preview.setAttribute('aria-describedby', source.id);
            }
            if (!current()) return;
            preview.textContent = disclosed;
            preview.hidden = false;
            input.hidden = true;
            status.textContent = '';
            status.hidden = true;
            button.setAttribute('aria-expanded', 'true');
        } catch (error) {
            if (!current()) return;
            reset();
            status.textContent = `Could not show this value: ${error.message || error}`;
            status.dataset.tone = 'danger';
            status.hidden = false;
        } finally {
            if (generation === request) button.removeAttribute('aria-busy');
        }
    }

    input.addEventListener('input', reset);
    identityInputs.forEach((node) => node.addEventListener('input', reset));
    button.addEventListener('click', toggle);
    const controller = { reset };
    reveals.set(input, controller);
    reset();
    return controller;
}

export function bindSecretInputs(root) {
    root.querySelectorAll('.secret-toggle, .secret-clear').forEach((button) => {
        const input = root.querySelector(`#${button.dataset.target}`);
        if (!input) return;
        button.setAttribute('aria-controls', input.id);
        const label = input.labels?.[0];
        if (label) {
            label.id ||= `${input.id}-label`;
            button.setAttribute('aria-describedby', label.id);
        }
    });
    root.querySelectorAll('.secret-input').forEach((input) => {
        input.addEventListener('input', () => {
            if (input.value.trim()) delete input.dataset.forceClear;
        });
    });

    root.querySelectorAll('.secret-toggle').forEach((button) => {
        const input = root.querySelector(`#${button.dataset.target}`);
        if (!input) return;
        const customRow = input.closest('[data-custom-secret-row]');
        const keyInput = customRow?.querySelector('[data-custom-secret-key]');
        bindSecretReveal(input, button, {
            savedSelector: () => {
                // A key added in a card is read by its connection and field; a draft has nothing saved.
                const added = input.closest('[data-connection-row][data-uid]');
                if (added) return added.dataset.draft === '1' ? null
                    : { connection: `${added.dataset.kind}@${added.dataset.name}`, field: input.dataset.connectionField };
                const key = customRow?.dataset.originalKey || input.dataset.secretSetting;
                return key ? { key } : null;
            },
            savedLabel: () => keyInput && keyInput.value.trim().toUpperCase() !== customRow.dataset.originalKey
                ? `Saved value for ${customRow.dataset.originalKey}` : '',
            identityInputs: keyInput ? [keyInput] : [],
        });
    });

    root.querySelectorAll('.secret-clear').forEach((button) => {
        button.addEventListener('click', () => {
            const target = root.querySelector(`#${button.dataset.target}`);
            if (!target) return;
            target.value = '';
            target.type = 'password';
            target.dataset.forceClear = '1';
            const toggle = root.querySelector(`.secret-toggle[data-target="${button.dataset.target}"]`);
            if (toggle) toggle.textContent = 'Show';
            // Programmatic value changes fire no 'input' event, but a Clear is
            // an edit like any other: the provider-test verdict-expiry listener
            // must see it, or a stale OK keeps vouching for a cleared key.
            target.dispatchEvent(new Event('input', { bubbles: true }));
        });
    });
}

/* ---- Settings › Accounts and keys: stored keys ---------------------------------------------------------------------
   A stored key (the GitHub token, a key a skill asked for, a key of the owner's own) is a settings row of kind
   `secret`: the row helper writes it when its field is left, like every typed row. At rest the row says only whether
   a value is saved, with Replace (or Add): that press opens its field. A saved value is never put into the field and
   never said back: Show reads it on the owner's press, a save says Saved with no "was" and no Undo, and Remove asks
   first. settings_accounts.js draws these rows among the page's others and binds them here. */

const landed = (outcome) => outcome?.state === 'saved' || outcome?.state === 'saved_then_failed';

/** A stored key's descriptor (settings_rows.js shape) for a key the page learns of when it reads: a skill's, or the
 *  owner's own. */
export const storedKeyRow = (key, meta) => ({ id: key, keys: [key], page: 'accounts', fold: false, name: key, meta, more: '',
    control: { kind: 'secret' }, aliases: [], writer: 'generic', codec: '', confirm: '', hiddenWhen: '', verbatim: true });

/** One stored key's row, on the row helper's hooks. A key's own name is shown as written, never translated. */
export function secretRowHtml(row) {
    const id = `setting-${row.id}`;
    return `<div class="ui-row settings-row accounts-secret" data-setting-row="${escapeHtml(row.id)}" data-kind="secret" data-secret-row>
        <label class="ui-row-title" for="${id}"${row.verbatim ? ' data-i18n-skip' : ''}>${escapeHtml(row.name)}</label>
        <div class="ui-row-meta"><span id="${id}-meta" data-setting-meta>${escapeHtml(row.meta || '')}</span><span class="ui-result settings-row-result" id="${id}-result" data-setting-result role="status" aria-live="polite"></span></div>
        <div class="ui-row-actions">
            <span class="settings-row-value" data-secret-state></span>
            <button type="button" class="btn btn-default btn-sm" data-secret-open aria-expanded="false" aria-controls="${id}-field">Add</button>
        </div>
        <div class="secret-input-row accounts-secret-field" id="${id}-field" data-secret-field hidden>
            <input id="${id}" class="ui-control secret-input" data-setting-control type="password" autocomplete="off" spellcheck="false" placeholder="New value" aria-describedby="${id}-meta ${id}-result">
            <button type="button" class="btn btn-default btn-sm" data-secret-show>Show</button>
            <button type="button" class="btn btn-danger btn-sm" data-secret-remove hidden>Remove</button>
        </div>
    </div>`;
}

/**
 * Wire the stored-key rows drawn in `container` (beside `bindSettingRows`, which commits their fields): Replace or
 * Add opens a row's field, Show reads the saved value, Remove asks and clears it. `rowOf(id)` answers a row's
 * descriptor. Answers `{ paint(settings), close(id), dispose() }`: `paint` says which keys hold a value, from a read;
 * `close` puts a row's field away once its write landed.
 */
export function bindSecretRows(container, { rowOf, ask } = {}) {
    let held = {};
    const nodes = () => [...container.querySelectorAll('[data-secret-row]')];
    const parts = (node) => ({ input: node.querySelector('[data-setting-control]'), field: node.querySelector('[data-secret-field]'),
        opener: node.querySelector('[data-secret-open]'), state: node.querySelector('[data-secret-state]'),
        show: node.querySelector('[data-secret-show]'), remove: node.querySelector('[data-secret-remove]') });
    const isSet = (node) => Boolean(held[node.dataset.settingRow]);
    const open = (node, on) => {
        const { field, opener, input } = parts(node);
        if (!on) resetSecretReveals(node);
        field.hidden = !on;
        opener.setAttribute('aria-expanded', String(on));
        if (on) input.focus?.({ preventScroll: true });
    };
    nodes().forEach((node) => {
        const { input, show } = parts(node);
        bindSecretReveal(input, show, { savedSelector: () => ({ key: node.dataset.settingRow }), saved: () => !input.value && isSet(node) });
    });
    const onClick = async (event) => {
        const node = event.target?.closest?.('[data-secret-row]');
        if (!node || !container.contains(node)) return;
        if (event.target.closest('[data-secret-open]')) { open(node, parts(node).field.hidden); return; }
        const remove = event.target.closest('[data-secret-remove]');
        const row = remove && rowOf?.(node.dataset.settingRow);
        if (!row || !(await CONFIRMS.removeEntry(row, null, null, ask))) return;
        const doc = node.ownerDocument, focused = node.contains(node.ownerDocument.activeElement);
        setButtonBusy(remove, true);
        // An empty value clears the key. It goes through the row's own queue, so its result line says what happened.
        const outcome = await commitSetting(row, '', { confirmed: true, ask });
        setButtonBusy(remove, false);
        // Disabling Remove can blur it before the saved-key readback removes its row.
        if (focused && doc.activeElement === doc.body && node.isConnected) (landed(outcome) ? parts(node).opener : remove).focus?.({ preventScroll: true });
    };
    container.addEventListener('click', onClick);
    return {
        paint(settings) {
            const deployment = deploymentOf(settings?._meta);
            held = Object.fromEntries(nodes().map((node) => [node.dataset.settingRow, Boolean(settings?.[node.dataset.settingRow])]));
            for (const node of nodes()) {
                const { opener, state, remove } = parts(node);
                // A value the organization fixes is said, never offered for change (settings_policy.js).
                const owner = organizationSettingOwner(deployment, node.dataset.settingRow);
                const words = owner ? `Set by your organization for ${owner}` : isSet(node) ? 'Saved' : '';
                if (state.textContent !== words) state.textContent = words;
                const label = isSet(node) ? 'Replace' : 'Add';
                if (opener.textContent !== label) opener.textContent = label;
                opener.hidden = Boolean(owner);
                opener.className = `btn ${isSet(node) ? 'btn-ghost' : 'btn-default'} btn-sm`;
                remove.hidden = !isSet(node);
                if (owner) open(node, false);
            }
        },
        close(id) {
            const node = nodes().find((each) => each.dataset.settingRow === id);
            if (!node || parts(node).field.hidden) return;
            // The caret stays in the row: on the press that would open its field again.
            const inside = node.contains(node.ownerDocument?.activeElement);
            open(node, false);
            if (inside) parts(node).opener.focus?.({ preventScroll: true });
        },
        dispose() {
            resetSecretReveals(container);
            container.removeEventListener('click', onClick);
        },
    };
}

/**
 * Mount the page that adds a key of the owner's own (`accounts/key/custom`): its name, its value, its own Save. The
 * name's rules are the helper's (`collectCustomSecretDraft`); `left(note)` is called once the key is saved.
 */
export function mountCustomKeyPage(container, { ctx = {}, left = () => {} } = {}) {
    const read = ctx.readSettings || readSettings;
    let gone = false;
    container.innerHTML = `<p class="settings-page-lead">A key of your own that skills may read once you grant it.</p>
        <section class="ui-card settings-card" data-custom-secret-row>
            <div class="accounts-key-form">
                <div class="form-field ui-field"><label for="custom-key-name">Name</label>
                    <input id="custom-key-name" class="ui-control" type="text" data-custom-secret-key placeholder="SLACK_WEBHOOK_URL" autocomplete="off" spellcheck="false" aria-describedby="custom-key-help"></div>
                <div class="form-field ui-field"><label for="custom-key-value">Value</label><div class="secret-input-row">
                    <input id="custom-key-value" class="secret-input ui-control" type="password" data-custom-secret-value placeholder="Secret value" autocomplete="off" spellcheck="false">
                    <button type="button" class="btn btn-default secret-toggle" data-target="custom-key-value">Show</button></div></div>
            </div>
            <p class="settings-card-note" id="custom-key-help">Capital letters, digits and underscores, starting with a letter.</p>
            <div class="ui-action-row"><button type="button" class="btn btn-save" data-editor-save>Save</button>
                <span class="ui-result" data-editor-result role="status" aria-live="polite"></span></div>
        </section>`;
    bindSecretInputs(container);
    const result = container.querySelector('[data-editor-result]');
    const save = container.querySelector('[data-editor-save]');
    const onSave = async () => {
        setButtonBusy(save, true);
        let settings = null;
        try { settings = await read(); } catch { /* the name is then judged by its own rules alone */ }
        const custom = Array.isArray(settings?._meta?.custom_secret_keys) ? settings._meta.custom_secret_keys : [];
        // The keys already stored stand beside the new one, so a name that repeats one of them is refused.
        const others = custom.map((key) => ({ key, value: '', appliedValue: '', originalKey: key, clear: false, removed: false }));
        const { values, errors } = collectCustomSecretDraft([...readCustomSecretDraft(container), ...others],
            Object.keys(settings || {}).filter((key) => !custom.includes(key)));
        const refused = errors.find((error) => error.index === 0);
        const key = Object.keys(values)[0];
        let outcome = null;
        if (!refused && key) {
            outcome = await commitSetting({ id: key, keys: [key], page: 'accounts', name: key, control: { kind: 'secret' }, writer: 'generic' },
                values[key], { confirmed: true, ask: ctx.ask });
        }
        setButtonBusy(save, false);
        if (gone) return;
        if (landed(outcome)) { left(`Added ${key}.`); return; }
        const input = container.querySelector(refused?.field === 'value' ? '[data-custom-secret-value]' : '[data-custom-secret-key]');
        if (refused) input?.focus?.();
        const line = outcome ? outcomeLine({ keys: [key], control: { kind: 'secret' } }, outcome) : null;
        setResult(result, refused ? { text: `Not saved. ${refused.message}`, tone: 'danger' } : line ? { text: line.text, tone: line.tone, details: line.details } : {});
    };
    const onInput = () => setResult(result, {});
    save.addEventListener('click', onSave);
    container.addEventListener('input', onInput);
    return { unmount() {
        gone = true;
        resetSecretReveals(container);
        save.removeEventListener('click', onSave);
        container.removeEventListener('input', onInput);
    } };
}
