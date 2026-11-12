/** Skill forms and skill sources in Settings (docs/DESIGN.md "Settings").
 *
 * `skill` lists the skills that registered a settings form, one door each; search lands on it by the names this
 * list had ("Extension Settings", under Advanced). `skill/<id>` is the form a skill registered (its reviewed,
 * host-rendered settings sections), opened from that list, the skill's card, a place's Connect, Voice and calls,
 * and search. Each opened form has its own host element, `#extension-settings-sections`, let go with its page: the
 * shell never puts a skill's form aside (settings.js), because a skill's page module binds each drawing anew and
 * only the forms in the page. Only that skill's routes are read, and `ouro:extension-settings-mounted` tells
 * a skill's own page module that its form stands (the bundled Calls skill enhances its form on it). The host draws
 * every field, reads the stored values from the route it posts to and owns the Save outcome.
 *
 * `skill-sources` is where skills come from on this computer: the skills folder and the ClawHub address, two rows
 * that save themselves (settings_rows.js). It is opened from the Skills page's sources view and from search. */

import { apiClient, apiFetch, cleanExtensionRoute, extensionRoutePath } from './api_client.js';
import { getContributions } from './contributions.js';
import { extensionActionStatus } from './extension_status_text.js';
import { bindSettingRows, settingsRow } from './settings_controls.js';
import { registerSettingsPage } from './settings_pages.js';
import { rowsOf } from './settings_rows.js';
import { skillTitle } from './skill_presentation.js';
import { collectSafeFieldValues, normalizeTone, renderSafeField, setInlineStatus } from './ui_helpers.js';
import { clearResultOnEdit, errorWords, setResult } from './ui_primitives.js';
import { escapeHtmlAttr as escapeHtml } from './utils.js';

const pendingExtensionSettings = new Set();

// A declarative extension form must show what is STORED (posting the bare schema once unbound the
// Telegram skill's owner chat id). The host reads the current values from the SAME route it posts
// to. A route with no GET handler (404/405) cannot say what is stored: its fields start locked
// behind a warning until the owner chooses Edit anyway; any other failed read keeps Save disabled,
// and a successful `{}` is a real empty document.
const EXTENSION_VALUES_UNREADABLE = 'Current values could not be read; Save is disabled so it does not overwrite them. Open this page again to retry.';
const EXTENSION_VALUES_UNAVAILABLE = 'Current values are unavailable. These are defaults. Sending this form may change saved settings.';
const EXTENSION_EDITING_UNREAD = 'You chose to edit without reading the saved values. Review every field before sending.';

const extensionFormKey = (section, component, idx) =>
    `${section.key || `${section.skill}:${section.section_id}`}:${component.id || idx}`;

/** Read one form's stored values from its own route: `read` is ok, unsupported or failed. Exported for node tests. */
export async function readExtensionFormValues(skill, route) {
    try {
        const resp = await apiFetch(extensionRoutePath(skill, route));
        if (resp.status === 404 || resp.status === 405) return { values: {}, read: 'unsupported' };
        if (!resp.ok) return { values: {}, read: 'failed' };
        const data = await resp.json();
        if (!data || typeof data !== 'object' || Array.isArray(data)) return { values: {}, read: 'failed' };
        return { values: data, read: 'ok' };
    } catch {
        return { values: {}, read: 'failed' };
    }
}

/** A form or action whose fields are all read-only: it reports a status, and its submit is a check, not a save. */
export function isStatusForm(component = {}) {
    const fields = Array.isArray(component.fields) ? component.fields : [];
    return fields.length > 0 && fields.every((field) => Boolean(field?.disabled));
}

export async function renderExtensionSettingsSections(root, sections, { isCurrent = () => true } = {}) {
    const host = root.querySelector('#extension-settings-sections');
    if (!host) return;
    const items = Array.isArray(sections) ? sections : [];
    if (!items.length) {
        host.innerHTML = '<div class="muted">No extension settings registered.</div>';
        globalThis.window?.dispatchEvent(new CustomEvent('ouro:extension-settings-mounted'));
        return;
    }
    const hydrated = new Map();
    await Promise.all(items.flatMap((section) => (Array.isArray(section.render?.components) ? section.render.components : [])
        .map(async (component, idx) => {
            const rawRoute = component.route || component.api_route || '';
            // Only a component with fields has values to read; an action's route is a
            // side effect and must not be probed on every Settings load.
            if (!['form', 'action'].includes(String(component.type || '')) || !cleanExtensionRoute(rawRoute)) return;
            if (!(Array.isArray(component.fields) && component.fields.length)) return;
            hydrated.set(extensionFormKey(section, component, idx),
                await readExtensionFormValues(section.skill || '', rawRoute));
        })));
    // A newer load or an owner edit may have landed while those reads were in
    // flight; a stale hydration must never overwrite the newer render.
    if (!isCurrent()) return;
    const formSpecs = new Map();
    const componentHtml = (section, component, idx) => {
        const type = String(component.type || '');
        if (type === 'markdown') {
            return `<div class="settings-section-copy">${escapeHtml(component.text || '')}</div>`;
        }
        if (type === 'json') {
            return `<details class="widget-json"><summary>${escapeHtml(component.label || 'JSON')}</summary><pre>${escapeHtml(JSON.stringify(component.value || component.data || {}, null, 2))}</pre></details>`;
        }
        if (type === 'form' || type === 'action') {
            const fields = Array.isArray(component.fields) ? component.fields : [];
            const rawRoute = component.route || component.api_route || '';
            if (!cleanExtensionRoute(rawRoute)) {
                return '<div class="settings-inline-note">Invalid extension settings route.</div>';
            }
            const formKey = extensionFormKey(section, component, idx);
            formSpecs.set(formKey, component);
            // Fieldless actions are never read, so they keep `ok`.
            const { values = {}, read = 'ok' } = hydrated.get(formKey) || {};
            const disabled = Boolean(component.disabled);
            // Every field read-only: the form shows what its route reports, and its submit checks again.
            const check = isStatusForm(component);
            const unread = read === 'unsupported' && !disabled;
            const failed = read === 'failed';
            const fieldOptions = {
                disabled: disabled || unread,
                fieldClass: 'form-field ui-field',
                inlineClass: 'settings-extension-checkbox ui-field ui-field-inline',
                helpClass: 'settings-inline-note ui-field-help',
            };
            return `
                <form class="settings-extension-form" data-extension-settings-form data-extension-settings-key="${escapeHtml(formKey)}" data-skill="${escapeHtml(section.skill || '')}" data-route="${escapeHtml(rawRoute)}"${failed || unread ? ' data-extension-settings-blocked="1"' : ''}>
                    ${unread ? `<div class="settings-inline-status" data-extension-settings-read data-tone="${normalizeTone('warn')}">${escapeHtml(EXTENSION_VALUES_UNAVAILABLE)}</div>` : ''}
                    <div class="form-grid two">${fields.map((field) => renderSafeField(field, values, fieldOptions)).join('')}</div>
                    <div class="settings-extension-actions ui-action-row">
                        <button class="btn${check ? ' btn-default' : type === 'form' ? ' btn-save' : ''}"${type === 'form' || check ? '' : ' data-primary'} type="submit"${disabled || failed || unread ? ' disabled' : ''}>${escapeHtml(component.submit_label || component.label || 'Save')}</button>
                        ${unread ? '<button class="btn btn-default" type="button" data-extension-settings-edit-anyway>Edit anyway</button>' : ''}
                        <div class="ui-result" data-extension-settings-status${failed ? ` data-tone="${normalizeTone('warn')}"` : ''}>${failed ? escapeHtml(EXTENSION_VALUES_UNREADABLE) : ''}</div>
                    </div>
                </form>
            `;
        }
        return `<div class="settings-inline-note">Unsupported extension settings component ${idx + 1}: ${escapeHtml(type || 'unknown')}</div>`;
    };
    host.innerHTML = items.map((section) => {
        const title = escapeHtml(section.title || section.section_id || section.key || 'Extension settings');
        const skill = escapeHtml(section.skill || '');
        const components = Array.isArray(section.render?.components) ? section.render.components : [];
        return `
            <article class="settings-extension-section">
                <div class="settings-extension-section-head">
                    <strong>${title}</strong>
                    ${skill ? `<span class="settings-inline-note">from ${skill}</span>` : ''}
                </div>
                <div class="settings-extension-components">
                    ${components.length ? components.map((component, idx) => componentHtml(section, component, idx)).join('') : '<div class="muted">No declarative components.</div>'}
                </div>
            </article>
        `;
    }).join('');
    host.querySelectorAll('[data-extension-settings-form]').forEach((form) => {
        // An edit clears the form's last result; a warning that its stored values are unreadable is a current fact and stays.
        const result = form.querySelector('[data-extension-settings-status]');
        if (!result?.textContent) clearResultOnEdit(form, result);
        // The owner's explicit choice to send this form without its stored values: it unlocks
        // the fields the schema does not disable, and Save sends every one of them as before.
        form.querySelector('[data-extension-settings-edit-anyway]')?.addEventListener('click', (event) => {
            const spec = formSpecs.get(form.dataset.extensionSettingsKey) || {};
            for (const field of spec.fields || []) {
                const input = field.disabled ? null : form.elements?.namedItem?.(String(field.name || ''));
                if (input) input.disabled = false;
            }
            delete form.dataset.extensionSettingsBlocked;
            form.querySelector('button[type="submit"]').disabled = false;
            event.currentTarget.disabled = true;
            setInlineStatus(form.querySelector('[data-extension-settings-read]'), EXTENSION_EDITING_UNREAD, 'warn');
        });
        form.addEventListener('submit', async (event) => {
            event.preventDefault();
            const status = form.querySelector('[data-extension-settings-status]');
            const skill = form.dataset.skill || '';
            const route = form.dataset.route || '';
            const formKey = form.dataset.extensionSettingsKey || `${skill}:${route}`;
            const spec = formSpecs.get(formKey) || {};
            const requestKey = `${skill}:${route}`;
            if (!skill || !route || spec.disabled || form.dataset.extensionSettingsBlocked
                || pendingExtensionSettings.has(requestKey)) return;
            const values = collectSafeFieldValues(form, spec.fields || []);
            const button = form.querySelector('button[type="submit"]');
            const idleLabel = spec.submit_label || spec.label || 'Save';
            pendingExtensionSettings.add(requestKey);
            const busy = spec.busy_label || 'Saving…';
            if (button) {
                button.disabled = true;
                button.textContent = busy;
            }
            setResult(status, { text: busy, tone: 'muted' });
            try {
                const cleanRoute = cleanExtensionRoute(route);
                if (!cleanRoute) throw new Error('invalid extension settings route');
                const resp = await apiFetch(extensionRoutePath(skill, route), {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(values),
                });
                const data = await resp.json().catch(() => ({}));
                if (!resp.ok || data.error) {
                    throw Object.assign(new Error(data.error || `HTTP ${resp.status}`), { status: resp.ok ? undefined : resp.status, body: data });
                }
                // A read-only field shows what the route answered just now, from the same source as its
                // first render; an editable one keeps what the owner typed.
                for (const field of spec.fields || []) {
                    const name = String(field?.name || '');
                    const input = field?.disabled && name && Object.prototype.hasOwnProperty.call(data, name)
                        ? form.elements?.namedItem?.(name) : null;
                    if (!input) continue;
                    if (input.type === 'checkbox') input.checked = Boolean(data[name]);
                    else input.value = String(data[name] ?? '');
                }
                const outcome = extensionActionStatus(data);
                setResult(status, { ...outcome, ends: outcome.tone === 'ok' });
                // A check saves nothing, so it announces no settings change to other pages.
                if (data.ok !== false && !isStatusForm(spec)) globalThis.window?.dispatchEvent(new CustomEvent('ouro:settings-updated', {
                    detail: { reason: 'extension settings saved', source: 'settings', skill, route },
                }));
            } catch (err) {
                setResult(status, errorWords(err));
            } finally {
                pendingExtensionSettings.delete(requestKey);
                if (button) {
                    button.disabled = Boolean(spec.disabled);
                    button.textContent = idleLabel;
                }
            }
        });
    });
    // Already-granted page modules may enhance their own declared forms. The
    // schema, hydration, write route and Save outcome remain host-owned.
    globalThis.window?.dispatchEvent(new CustomEvent('ouro:extension-settings-mounted'));
}

/* ---- The pages ----------------------------------------------------------------------------------------------- */

// The page the owner is looking at: Settings is shown, this path is its open level and the window is not hidden.
const onScreen = (ctx, path) => ctx?.state?.activePage === 'settings' && ctx.state.settingsPath === path && !globalThis.document?.hidden;
const FORMS = new Map();  // skill id → { skill, name, sections }: what the last read of the installed skills says

/** One read of the installed skills (`GET /api/extensions`, a passive read). It answers the list and remembers which
 *  skills registered a settings form, for the skill page, its title and search. */
export async function readSkills() {
    const data = await apiClient.extensions();
    const names = new Map((Array.isArray(data?.skills) ? data.skills : []).map((skill) => [String(skill?.name || ''), skill]));
    FORMS.clear();
    for (const section of Array.isArray(data?.live?.settings_sections) ? data.live.settings_sections : []) {
        const skill = String(section?.skill || '');
        if (!skill) continue;
        const known = names.get(skill);
        if (!FORMS.has(skill)) FORMS.set(skill, { skill, name: String((known && skillTitle(known)) || skill), sections: [] });
        FORMS.get(skill).sections.push(section);
    }
    return data;
}

/** The skill forms the last read knows: `{ skill, name, sections }` each. */
export const skillForms = () => [...FORMS.values()];

const skillOf = (path) => { try { return decodeURIComponent(String(path).split('/')[1] || ''); } catch { return ''; } };

/** A skill page's title: its one section's own title, else the skill's name; the id until the skills are read. The
 *  page without a skill is the list of them. */
export function skillFormTitle(path) {
    const form = FORMS.get(skillOf(path));
    const titled = form?.sections.length === 1 ? String(form.sections[0].title || '') : '';
    return titled || form?.name || skillOf(path) || 'Skill settings';
}

// `skill` alone: one door per skill that registered a form. A list that could not be read says so and claims nothing.
function mountSkillList(container, { path, ctx }) {
    let alive = true;
    let rows = null;
    let drawn = null;
    let drawing = null;
    // One read at a time: the page's first draw and the signal that it is shown ask together.
    const draw = () => (drawing ||= read().finally(() => { drawing = null; }));
    const read = async () => {
        let failure = null;
        try { await readSkills(); } catch (err) { failure = err; }
        if (!alive) return;
        const doors = failure ? [] : skillForms().map((form) => {
            const name = skillFormTitle(`skill/${encodeURIComponent(form.skill)}`);
            return { id: `skill:${form.skill}`, keys: [], page: path, name, meta: form.name === name ? '' : form.name, control: { kind: 'door' } };
        });
        // The same list is not drawn again: a door that holds focus keeps it.
        const signature = failure ? `failed:${failure?.message}` : JSON.stringify(doors.map((row) => [row.id, row.name, row.meta]));
        if (signature === drawn) return;
        drawn = signature;
        rows?.dispose();
        container.innerHTML = `<section class="ui-card settings-card" data-skill-forms>${doors.length
            ? `<div class="settings-rows">${doors.map((row) => settingsRow(row)).join('')}</div>`
            : `<p class="${failure ? 'ui-result' : 'settings-card-lead'}">${failure ? '' : 'No installed skill has settings.'}</p>`}</section>`;
        if (failure) setResult(container.querySelector('.ui-result'), errorWords(failure));
        rows = bindSettingRows(container, { rows: doors,
            openDoor: (row) => ctx.openSettings(`skill/${encodeURIComponent(row.id.slice('skill:'.length))}`) });
    };
    const onShown = (event) => { if (event.detail?.path === path) void draw(); };
    window.addEventListener('ouro:settings-page-shown', onShown);
    // Every way a skill changes ends in the one lifecycle signal (contributions.js): this page's own action, a job
    // that finished, another window, Ouroboros itself. A Settings that is not on screen reads nothing: the shell says
    // "shown" on the way back and the page reads then; so does a list under an open form.
    const offLifecycle = getContributions().onLifecycle(() => { if (onScreen(ctx, path)) void draw(); });
    return draw().then(() => ({
        unmount() {
            alive = false;
            window.removeEventListener('ouro:settings-page-shown', onShown);
            offLifecycle();
            rows?.dispose();
        },
    }));
}

function mountSkillForm(container, { path, ctx }) {
    const skill = skillOf(path);
    if (!skill) return mountSkillList(container, { path, ctx });
    const host = Object.assign(document.createElement('div'), { id: 'extension-settings-sections', className: 'settings-extension-sections' });
    container.replaceChildren(host);
    let alive = true;
    let serial = 0;
    let filling = null;
    // A field the owner is in, or has typed into, is not redrawn: not by a read that starts now and not by one that
    // was on its way when the owner began (a skill changed elsewhere starts such a read at any moment).
    const untouched = () => !host.dataset.typed && !(host.contains(document.activeElement) && host !== document.activeElement);
    // The skills are read, then only this skill's routes.
    const fill = () => (filling ||= (async () => {
        if (!untouched()) return;
        const run = ++serial;
        const current = () => alive && run === serial && host.parentElement === container && untouched();
        try { await readSkills(); } catch (err) {
            if (current() && !host.innerHTML) setResult(host.appendChild(Object.assign(document.createElement('p'), { className: 'ui-result' })), errorWords(err));
            return;
        }
        if (current()) await renderExtensionSettingsSections(container, FORMS.get(skill)?.sections || [], { isCurrent: current });
    })().finally(() => { filling = null; }));
    const onTyped = () => { host.dataset.typed = '1'; };
    const onSaved = (event) => { if (event.detail?.skill === skill) delete host.dataset.typed; };
    const onShown = (event) => { if (event.detail?.path === path) void fill(); };
    host.addEventListener('input', onTyped);
    window.addEventListener('ouro:settings-updated', onSaved);
    window.addEventListener('ouro:settings-page-shown', onShown);
    const offLifecycle = getContributions().onLifecycle(() => { if (onScreen(ctx, path)) void fill(); });  // as the list above
    return Promise.resolve(fill()).then(() => ({
        unmount() {
            alive = false;
            host.removeEventListener('input', onTyped);
            window.removeEventListener('ouro:settings-updated', onSaved);
            window.removeEventListener('ouro:settings-page-shown', onShown);
            offLifecycle();
            if (host.parentElement === container) host.remove();
        },
    }));
}

function mountSkillSources(container, { path, ctx }) {
    container.innerHTML = `<section class="ui-card settings-card">
        <p class="settings-card-lead">Skills are installed from the sources on the Skills page. These two say where else this computer looks.</p>
        <div class="settings-rows">${rowsOf(path).map((row) => settingsRow(row)).join('')}</div></section>`;
    const rows = bindSettingRows(container);
    const read = () => Promise.resolve(ctx.readSettings?.()).then(() => rows.paint(), () => {});
    const onShown = (event) => { if (event.detail?.path === path) void read(); };
    window.addEventListener('ouro:settings-page-shown', onShown);
    void read();
    return { unmount() { window.removeEventListener('ouro:settings-page-shown', onShown); rows.dispose(); } };
}

registerSettingsPage('skill', { title: skillFormTitle, mount: mountSkillForm });
registerSettingsPage('skill-sources', { title: 'Skill sources', mount: mountSkillSources });
