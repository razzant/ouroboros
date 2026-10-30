/** The granted page module enhances its declared Settings form; catalogs stay server-owned. */
import { escapeHtmlAttr as esc, renderSafeField, collectSafeFieldValues } from '/static/modules/ui_primitives.js';
import { fetchJson } from '/static/modules/api_client.js';
import { bindDeviceSettings, devicesHtml } from './call_device_settings.js';

export const CONFIG_FIELDS = ['enabled', 'provider', 'model', 'voice', 'instructions', 'pause', 'interruptions', 'thinking', 'search']
    .map(name => ({ name, type: ['enabled', 'interruptions', 'search'].includes(name) ? 'checkbox' : 'text' }));
const field = (name, label, type, options, help) => ({ name, label, type, options, help });
const options = rows => rows.map(row => ({ value: row.id, label: row.label || row.id }));
const select = (name, label, rows, value) => {
    const choices = options(rows);
    if (value && !choices.some(row => row.value === value)) choices.push({ value, label: `${value} (not available)` });
    return field(name, label, 'select', choices);
};

export function callSettingsHtml(config) {
    const catalogs = config.catalogs || [];
    const catalog = catalogs.find(row => row.id === config.provider);
    const model = catalog?.models.find(row => config.model ? row.id === config.model : row.default);
    if (config.thinking && model?.thinking_levels?.length && !model.thinking_levels.includes(config.thinking)) {
        config = {...config, thinking:''};  // a level this model lacks falls back to the model's own default
    }
    const fields = [field('enabled', 'Calls (beta)', 'checkbox'),
        select('provider', 'Engine', catalogs, config.provider),
        select('model', 'Model', [{id:'',label:"Engine’s default model"}, ...(catalog?.models || [])], config.model),
        select('voice', 'Voice', [{id:'',label:"Engine’s default voice"}, ...(model?.voices || [])], config.voice)];
    if (catalog?.id === 'gemini_live:gemini') {
        fields[1].help = 'Google does not offer Gemini Live in Russia or Belarus.';
    }
    const knobs = {
        pause: field('pause', 'Pause before answering', 'select', ['short','normal','long']),
        interruptions: field('interruptions', 'Speaking interrupts Ouroboros', 'checkbox'),
        thinking: field('thinking', 'Thinking', 'select', [{value:'',label:`Default (${model?.thinking_default || model?.thinking_levels?.[0] || 'lowest'})`}, ...(model?.thinking_levels || [])]),
        search: field('search', 'Built-in search', 'checkbox'),
    };
    const notes = [];
    for (const [name, spec] of Object.entries(knobs)) {
        if (model?.honours?.includes(name)) fields.push(spec);
        const note = Object.fromEntries(model?.notes || [])[name];
        if (note) notes.push(`<div class="settings-inline-note" data-call-knob-note="${name}">${esc(note)}</div>`);
    }
    fields.push(field('instructions', 'Instructions for calls', 'textarea', undefined, 'Applies to the next call.'));
    const window = model?.facts?.conversation_window_tokens;
    const unit = model?.facts?.basis === 'estimator' ? 'tokens by Ouroboros’s own estimate'
        : model?.facts?.basis === 'provider' ? 'provider tokens' : 'tokens';
    const memory = window ? `${catalog.label}: about ${Number(window).toLocaleString('en-US')} ${unit} per session, then it renews.`
        : 'The memory window for this selection is not known.';
    return `<div class="settings-section-copy">Calls use this room’s ordinary memory and tools. Changes apply to the next call.</div>
        ${!catalog ? '<p class="settings-inline-note">The selected engine is not available. Choose another engine or check its status in Skills.</p>' : ''}
        <div class="form-grid two">${fields.map(spec => renderSafeField(spec, config, {fieldClass:'form-field'})).join('')}</div>
        ${notes.join('')}<p class="settings-inline-note" data-call-privacy>${esc(catalog?.privacy || '')}</p>
        <p class="settings-inline-note" data-call-memory>${esc(memory)} ${esc(model?.facts?.source || '')}</p>`;
}

/** A call setting saves as it changes; the Settings page's own Save Settings never reaches this form. */
export const autosaves = name => CONFIG_FIELDS.some(spec => spec.name === name);

/** Submit through the host's own Save, after a short pause, and after a Save still in flight. */
export function scheduleSave(form, save, env = globalThis, delay = 150) {
    let timer = 0;
    const submit = () => {
        if (save.disabled) { timer = env.setTimeout(submit, 300); return; }
        form.requestSubmit(save);
    };
    return {
        change() { env.clearTimeout(timer); timer = env.setTimeout(submit, delay); },
        cancel() { env.clearTimeout(timer); },
    };
}

export function mountCallSettings(form, { skill, env = globalThis, read = () => fetchJson(`/api/extensions/${encodeURIComponent(skill)}/config`) } = {}) {
    const fields = form.querySelector('.form-grid');
    const save = form.querySelector('button[type="submit"]');
    const status = form.querySelector('[data-extension-settings-status]');
    let disposed = false, draft, devices = null;
    let stopDevices = () => {};
    save.disabled = true;
    const autosave = scheduleSave(form, save, env);
    function paint() {
        fields.classList.remove('two');
        fields.innerHTML = callSettingsHtml(draft);
    }
    const changed = event => {
        if (!draft || !autosaves(event.target.name)) return;
        if (['provider', 'model'].includes(event.target.name)) {
            Object.assign(draft, collectSafeFieldValues(form, CONFIG_FIELDS));
            if (event.target.name === 'provider') draft.model = '';
            draft.voice = '';
            const catalog = draft.catalogs?.find(row => row.id === draft.provider);
            const model = catalog?.models.find(row => draft.model ? row.id === draft.model : row.default);
            draft.thinking = '';  // a new engine or model starts at its own default level
            paint();
        }
        autosave.change();
    };
    form.addEventListener('change', changed);
    const ready = read().then(config => {
        if (disposed) return;
        draft = config;
        paint();
        devices = form.ownerDocument.createElement('div');
        devices.innerHTML = devicesHtml();
        // The host groups Save with optional actions such as Edit anyway.
        form.insertBefore(devices, save.closest('.settings-extension-actions') || save);
        stopDevices = bindDeviceSettings(devices, env);
        delete form.dataset.extensionSettingsBlocked;
        save.disabled = false;
    }).catch(() => {
        if (!disposed) status.textContent = 'Call settings could not be read. Reload Settings to retry.';
    });
    return { ready, dispose() {
        disposed = true;
        autosave.cancel();
        form.removeEventListener('change', changed);
        stopDevices(); devices?.remove();
    } };
}
