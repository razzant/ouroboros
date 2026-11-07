export function bindEffortSegments(root) {
    root.querySelectorAll('[data-effort-group]').forEach((group) => {
        const targetId = group.dataset.effortTarget;
        const input = root.querySelector(`#${targetId}`);
        if (!input) return;
        const buttons = Array.from(group.querySelectorAll('[data-effort-value]'));
        // A group names itself by its own label, else by the title of the section it is the one control of.
        const label = input.previousElementSibling;
        const heading = label?.tagName === 'LABEL' ? label : group.closest('.form-section')?.querySelector('h3');
        if (heading) {
            heading.id ||= `${targetId}-label`;
            group.setAttribute('role', 'group');
            group.setAttribute('aria-labelledby', heading.id);
        }

        function sync() {
            buttons.forEach((button) => {
                button.classList.toggle('active', button.dataset.effortValue === input.value);
                button.setAttribute('aria-pressed', String(button.dataset.effortValue === input.value));
            });
        }

        buttons.forEach((button) => {
            button.addEventListener('click', () => {
                // Presence, not truthiness: '' is a real choice (inherit the Task / Chat effort).
                input.value = button.dataset.effortValue ?? input.value;
                input.dataset.effortTouched = '1';
                sync();
            });
        });

        sync();
    });
}

export function syncEffortSegments(root) {
    root.querySelectorAll('[data-effort-group]').forEach((group) => {
        const targetId = group.dataset.effortTarget;
        const input = root.querySelector(`#${targetId}`);
        if (!input) return;
        group.querySelectorAll('[data-effort-value]').forEach((button) => {
            button.classList.toggle('active', button.dataset.effortValue === input.value);
            button.setAttribute('aria-pressed', String(button.dataset.effortValue === input.value));
        });
    });
}

/** Read the complete custom-key draft, including blank and removed rows. No paint. */
export function readCustomSecretDraft(root) {
    return Array.from(root.querySelectorAll('[data-custom-secret-row]'), (row) => {
        const input = row.querySelector('[data-custom-secret-value]');
        return {
            key: row.querySelector('[data-custom-secret-key]')?.value || '',
            value: input?.value || '',
            appliedValue: input?.dataset.appliedValue || '',
            originalKey: row.dataset.originalKey || '',
            clear: input?.dataset.forceClear === '1',
            removed: row.dataset.removeCustomSecret === '1',
        };
    });
}

/** Compile the existing custom-key wire shape and return every local error. */
export function collectCustomSecretDraft(rows, knownKeys = []) {
    const values = {};
    const errors = [];
    const known = new Set(knownKeys);
    const seen = new Map();
    rows.forEach((row, index) => {
        const key = row.key.trim().toUpperCase();
        const fail = (message, field = 'key') => errors.push({ index, field, message });
        if (row.removed) {
            if (row.originalKey) values[row.originalKey] = '';
            return;
        }
        if (!key) fail('Enter a key name, or remove this row.');
        else if (!/^[A-Z][A-Z0-9_]{2,}$/.test(key)) fail('Use at least three letters, numbers or underscores, starting with a letter.');
        else if (key.startsWith('OUROBOROS_') || known.has(key)) fail('This name is a built-in setting. Use its existing field.');
        else if (seen.has(key)) {
            fail(`Duplicate key ${key}. Keep one row for this key.`);
            errors.push({ index: seen.get(key), field: 'key', message: `Duplicate key ${key}. Keep one row for this key.` });
        } else if (row.originalKey && key !== row.originalKey && row.value === row.appliedValue && !row.clear) {
            fail('Enter a value for the new key, or keep the saved name.', 'value');
        } else if (row.clear) values[key] = '';
        else if (row.value && row.value !== row.appliedValue) values[key] = row.value;
        else if (!row.originalKey) fail('Enter a value for this key, or remove this row.', 'value');
        if (key) seen.set(key, index);
    });
    return { values, errors };
}

/** Paint only errors owned by the Settings save gate; dirty reads never call this. */
export function paintSettingsFieldErrors(root, errors) {
    root.querySelectorAll('[data-settings-validation]').forEach((input) => {
        input.removeAttribute('aria-invalid');
        const hint = root.querySelector(`#${input.id}-error`);
        if (hint) { hint.hidden = true; hint.textContent = ''; }
        delete input.dataset.settingsValidation;
    });
    errors.forEach(({ input, message }) => {
        if (!input?.id) return;
        let hint = root.querySelector(`#${input.id}-error`);
        if (!hint) {
            hint = input.ownerDocument.createElement('div');
            hint.id = `${input.id}-error`;
            hint.className = 'settings-inline-note ui-field-help';
            hint.dataset.tone = 'error';
            hint.setAttribute('role', 'status');
            (input.closest('.form-field') || input.parentElement).appendChild(hint);
        }
        input.dataset.settingsValidation = '1';
        input.setAttribute('aria-invalid', 'true');
        const described = new Set((input.getAttribute('aria-describedby') || '').split(/\s+/).filter(Boolean));
        described.add(hint.id);
        input.setAttribute('aria-describedby', [...described].join(' '));
        hint.textContent = message;
        hint.hidden = false;
    });
}

/** A Settings action in flight: disabled and announced busy, the same for Save and a key's Test. */
export function setButtonBusy(button, busy) {
    if (!button) return;
    button.disabled = busy;
    if (busy) button.setAttribute('aria-busy', 'true');
    else button.removeAttribute('aria-busy');
}

/** Saved/unsaved/unknown are server receipts, never inferred from HTTP success. */
export function settingsWriteFailure(error, label = 'Settings') {
    const receipt = error?.body || error?.payload;
    const detail = error?.message || String(error);
    if (receipt?.saved === true) return { unknown: false, text: `${label} was saved, but a later step failed: ${detail}` };
    if (receipt?.saved === false) return { unknown: false, text: `${label} was not changed: ${detail}` };
    return { unknown: true, text: `${label} outcome is unknown: ${detail}` };
}

const SEARCH_GAPS = { no_credentials: 'no API key', package_missing: 'package not installed', legacy_openai_base_url: 'OPENAI_BASE_URL is set' };

/** The web search preview in owner words: the sources in the order they are tried (each named by `name`, the
 *  Source select's own word for its id) and a saved model a source ignores stay in the main line; the models, the
 *  sources not available now and the credential note are Details. */
export function webSearchPreviewWords(data, name = (id) => id) {
    if (data?.error) return { main: String(data.error), details: '' };
    const legs = data?.legs || [];
    const ignored = data?.ignored_legacy_model ? ` Anthropic does not apply the saved model ${data.ignored_legacy_model}; edit Source or Model to change it.`
        : data?.unapplied_model ? ` No built-in search serves the saved model ${data.unapplied_model}; Auto uses the recommended models.` : '';
    // provider_default names the source's recommended model (search_routes.SEARCH_MODELS), not the owner's.
    const models = legs.filter((leg) => leg.model)
        .map((leg) => `${name(leg.source)} · ${leg.model}${leg.model_source === 'provider_default' ? ' (recommended)' : ''}`);
    const unavailable = (data?.unavailable || [])
        .map((leg) => `${name(leg.source)} (${SEARCH_GAPS[leg.unavailable_reason] || 'not ready'})`);
    return {
        main: `${legs.length ? `Uses ${legs.map((leg) => name(leg.source)).join(', then ')}.` : 'No search source is ready.'}${ignored}`,
        details: [models.length ? `${models.join('; ')}.` : '', unavailable.length ? `Not available now: ${unavailable.join(', ')}.` : '',
            'Uses saved account credentials.'].filter(Boolean).join(' '),
    };
}

/* The one Settings section recipe (DESIGN.md principle 7; the 06.11 convergence): a title, one authored lead in
   owner words, right under it the one Details disclosure that holds the section's former copy unchanged (so it opens
   next to the line it explains, not screens below in a long section), then the controls. Quiet actions dock in the title row. Every tab's sections come from here, so Models, Agents and
   Accounts read as one scheme. `titleId` lets a lone control name itself by the title; `card: false` is a section
   that already sits inside a card (setup's step), which a second card would outline (DESIGN principle 2). */
export function settingsSection({ title, lead = '', details = '', body = '', actions = '', id = '', titleId = '', className = '', attrs = '', card = true }) {
    const h3 = `<h3${titleId ? ` id="${titleId}"` : ''}>${title}</h3>`;
    const head = actions ? `<div class="settings-section-head">${h3}<div class="settings-toolbar">${actions}</div></div>` : h3;
    return `
        <div class="form-section${card ? ' ui-card' : ''}${className ? ` ${className}` : ''}"${id ? ` id="${id}"` : ''}${attrs ? ` ${attrs}` : ''}>
            ${head}
            ${lead ? `<p class="settings-section-lead">${lead}</p>` : ''}
            ${details ? `<details class="ui-disclosure settings-section-details"><summary>Details</summary><div class="settings-section-copy">${details}</div></details>` : ''}
            ${body}
        </div>`;
}
