export function bindEffortSegments(root) {
    root.querySelectorAll('[data-effort-group]').forEach((group) => {
        const targetId = group.dataset.effortTarget;
        const input = root.querySelector(`#${targetId}`);
        if (!input) return;
        const buttons = Array.from(group.querySelectorAll('[data-effort-value]'));
        const label = input.previousElementSibling;
        if (label?.tagName === 'LABEL') {
            label.id = `${targetId}-label`;
            group.setAttribute('role', 'group');
            group.setAttribute('aria-labelledby', label.id);
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

/** Saved/unsaved/unknown are server receipts, never inferred from HTTP success. */
export function settingsWriteFailure(error, label = 'Settings') {
    const receipt = error?.body || error?.payload;
    const detail = error?.message || String(error);
    if (receipt?.saved === true) return { unknown: false, text: `${label} was saved, but a later step failed: ${detail}` };
    if (receipt?.saved === false) return { unknown: false, text: `${label} was not changed: ${detail}` };
    return { unknown: true, text: `${label} outcome is unknown: ${detail}` };
}
