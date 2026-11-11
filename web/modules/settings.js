/** Settings: the shell (docs/DESIGN.md "Settings"). The home (settings_home.js) holds Ouroboros's card, search and six
 * groups of rows; a row opens a page, which its module draws into the body the shell gives it (settings_pages.js), or
 * leaves Settings by its door. Every row saves itself (settings_controls.js): nothing here keeps a draft or a Save.
 *
 * What Settings shows is `state.settingsPath` ('' is the home), in memory only: no URL is written and a reload starts
 * at the home. Pages open in levels. One opened from the home, a search result or outside Settings is the first level;
 * one opened from a page lies over that page. The first row of every level is Back and names where it leads: the
 * level under it, the home ("Settings"), or the place outside Settings that opened it (`back: { label, go }`).
 *
 * The page box's width decides the layout (settings.css `@container settingspage`). Narrow: one level alone, each one
 * `page: true` layer of the shell's stack (layers.js), so Escape, the browser's or Android's Back and Telegram's Back
 * close exactly one level, as Back does; a field that holds focus is left first, so its row commits. Wide: the home
 * beside the level, and no layer. A level under another keeps its place, the home its scroll.
 *
 * `ouro:settings-page-shown` (`detail.path`) says which page is visible: when one opens or is returned to, when
 * Settings is shown, after a reconnect and when the window is seen again. A page reads and repaints on it for its own
 * path; the row helper never rewrites a field that holds focus or typed text. */

import { renderPageHeader } from './page_header.js';
import { SHELL_ICONS } from './page_icons.js';
import { bindScrollFade } from './scroll_fade.js';
import { readSettings } from './settings_controls.js';
import { isDoorId, mountSettingsHome } from './settings_home.js';
import { settingsPageFor } from './settings_pages.js';
import { HOME_ROWS } from './settings_rows.js';
import { resetSecretReveals } from './settings_secrets.js';
import { escapeHtmlAttr as escapeHtml } from './utils.js';
import { apiFetch, cleanExtensionRoute, extensionRoutePath } from './api_client.js';
import { collectSafeFieldValues, normalizeTone, renderSafeField, setInlineStatus } from './ui_helpers.js';
import { clearResultOnEdit, errorWords, setResult } from './ui_primitives.js';
import { extensionActionStatus } from './extension_status_text.js';
// The page modules: each registers its family when it is imported (settings_pages.js).
import './settings_ui.js';
import './subagents_settings.js';
import './mcp_settings.js';
import './model_roles.js';
import './settings_local_model.js';

const pendingExtensionSettings = new Set();

/** A page's title for `path`: its module's words, else the home row that opens it, else the path. */
export function settingsTitle(path) {
    const page = settingsPageFor(path);
    const named = typeof page?.title === 'function' ? page.title(path) : page?.title;
    return String(named || HOME_ROWS.find((row) => row.to === path)?.name || path);
}

const quoted = (value) => String(value).replace(/["\\]/g, '\\$&');
const visible = (el) => Boolean(el?.isConnected && el.getClientRects?.().length);

export function initSettings(ctx = {}) {
    const { state, ws, layers = null, setBeforePageLeave, showPage, openDashboardTab } = ctx;
    state.settingsPath = '';
    const page = document.createElement('div');
    page.id = 'page-settings';
    page.className = 'page app-page-glass';
    page.innerHTML = `${renderPageHeader({ page: 'settings' })}
        <div class="settings-shell" data-settings-view="home">
            <div class="settings-home scroll-fade-y" data-settings-home></div>
            <div class="settings-levels"></div>
        </div>`;
    document.getElementById('content').appendChild(page);
    const shell = page.querySelector('.settings-shell');
    const homeEl = page.querySelector('[data-settings-home]');
    const levelsEl = page.querySelector('.settings-levels');
    const levels = [];
    const layered = new Set();
    const places = new WeakMap();  // a column → where it was scrolled while visible
    let opening = false;
    let besideBefore = false;
    let serial = 0;
    const shown = () => state.activePage === 'settings';
    const top = () => levels[levels.length - 1] || null;
    const beside = () => levels.length > 0 && visible(homeEl);
    const pageCtx = { openSettings: (path, options) => go(path, options), openDashboardTab, showPage, layers, ws, state, readSettings };
    const home = mountSettingsHome(homeEl, { open: (path, options = {}) => go(path, { ...options, fresh: true }) });

    // A door out of Settings: a Dashboard tab gets a Back to the page it was opened from; Circles, Skills and Main are
    // pages of their own (`main:context` shows Main and does nothing else: the Context capsule is the room's).
    function leave(door, { row = '' } = {}) {
        const [kind, name] = door.split(':');
        if (kind === 'dashboard') return openDashboardTab?.(name, { row, back: state.settingsPath });
        return showPage?.(kind === 'main' ? 'chat' : name);
    }

    function go(path = '', { row = '', back = null, fresh = false, opener = null } = {}) {
        const target = String(path || '');
        if (isDoorId(target)) return leave(target, { row });
        const at = back || fresh ? -1 : levels.findIndex((level) => level.path === target);
        if (!target) closeLevels(0);
        else if (at >= 0) closeLevels(at + 1, { row });
        else open(target, { row, back, under: back || fresh ? 0 : levels.length, opener });
        return true;
    }

    // `opener` is the control that opened the level, where focus returns: WebKit does not focus a pressed button.
    function open(path, { row, back, under, opener }) {
        while (levels.length > under) close(levels.pop());
        const focused = document.activeElement === document.body ? null : document.activeElement;
        const level = { path, back, id: `settings-level-${++serial}`, opener: opener || focused };
        level.el = document.createElement('section');
        level.el.className = 'settings-page';
        level.el.dataset.settingsPage = path;
        level.el.setAttribute('aria-labelledby', `${level.id}-title`);
        const leadsTo = back?.label || (levels.length ? settingsTitle(top().path) : 'Settings');
        level.el.innerHTML = `<div class="settings-back-row"><button type="button" class="settings-back ui-hit-area" data-settings-back
            aria-label="Back to ${escapeHtml(leadsTo)}">${SHELL_ICONS.back}<span>${escapeHtml(leadsTo)}</span></button></div>
            <div class="settings-column"><h2 class="settings-page-title" id="${level.id}-title">${escapeHtml(settingsTitle(path))}</h2>
            <div class="settings-page-body"></div></div>`;
        level.el.querySelector('[data-settings-back]').addEventListener('click', () => backFrom(level));
        levelsEl.append(level.el);
        levels.push(level);
        mount(level, row);
        paint({ row, opened: level });
    }

    function mount(level, row) {
        const found = settingsPageFor(level.path);
        if (!found) return;  // no page serves this path yet: the chrome with nothing inside
        const body = level.el.querySelector('.settings-page-body');
        const failed = (err) => {
            console.error('settings: a page failed to draw', level.path, err);
            const line = document.createElement('p');
            line.className = 'ui-result';
            body.replaceChildren(line);
            setResult(line, { text: `This page could not be shown: ${err?.message || err}`, tone: 'danger' });
        };
        try {
            Promise.resolve(found.mount(body, { path: level.path, row, ctx: pageCtx })).then((handle) => {
                if (level.closed) handle?.unmount?.();
                else level.handle = handle;
            }, failed);
        } catch (err) { failed(err); }
    }

    // A level closes: the field it holds is left first (its row commits on that change), then its page is let go.
    function close(level) {
        const at = document.activeElement;
        if (at && level.el.contains(at)) at.blur?.();
        level.closed = true;
        level.watch?.disconnect();
        try { level.handle?.unmount?.(); } catch (err) { console.error('settings: a page failed to let go', level.path, err); }
        level.el.remove();
        if (layered.delete(level.id)) layers?.remove(level.id);
    }

    function closeLevels(keep, { row = '', quiet = false } = {}) {
        const left = levels.length > keep ? levels[keep] : null;
        while (levels.length > keep) close(levels.pop());
        paint({ row, returned: left, quiet });
    }

    function backFrom(level) {
        if (level !== top()) return;
        if (levels.length === 1 && level.back) {
            closeLevels(0, { quiet: true });
            level.back.go?.();
            return;
        }
        closeLevels(levels.length - 1);
    }

    // The row a page was opened for: its fold opens, it scrolls into view, its control takes focus. A page drawn after
    // a read gets it when the row appears.
    function revealRow(level, id) {
        const find = () => level.el.querySelector(`[data-setting-row="${quoted(id)}"]`);
        const reveal = (node) => {
            for (let fold = node.closest('details'); fold; fold = fold.parentElement?.closest('details')) fold.open = true;
            node.scrollIntoView?.({ block: 'center' });
            const control = node.querySelector('[data-setting-control], [data-setting-value], button, input, select, textarea');
            (control || node).focus?.({ preventScroll: true });
        };
        const node = find();
        if (node || typeof MutationObserver !== 'function') { if (node) reveal(node); return; }
        level.watch = new MutationObserver(() => {
            const hit = find();
            if (!hit) return;
            level.watch.disconnect();
            reveal(hit);
        });
        level.watch.observe(level.el, { childList: true, subtree: true });
    }

    function paint({ row = '', opened = null, returned = null, quiet = false } = {}) {
        const level = top();
        state.settingsPath = level?.path || '';
        shell.dataset.settingsView = level ? 'page' : 'home';
        for (const each of levels) each.el.hidden = each !== level;
        const column = level ? level.el : homeEl;
        column.scrollTop = places.get(column) || 0;
        if (beside() || !level) homeEl.scrollTop = places.get(homeEl) || 0;
        besideBefore = beside();
        home.mark(levels[0]?.path || '');
        syncLayers();
        if (!quiet) {
            if (row && level) revealRow(level, row);
            else if (opened && (!besideBefore || !visible(document.activeElement))) {
                opened.el.querySelector('[data-settings-back]')?.focus({ preventScroll: true });
            } else if (returned) {
                const to = visible(returned.opener) ? returned.opener
                    : level ? level.el.querySelector('[data-settings-back]')
                        : homeEl.querySelector(`[data-settings-open="${quoted(returned.path.split('/')[0])}"]`);
                to?.focus?.({ preventScroll: true });
            }
            announce();
        }
    }

    // Narrow: every level is a layer, so Escape and the system's Back close the top one. Wide, or Settings put away: none.
    function syncLayers() {
        if (!layers) return;
        const wanted = shown() && !beside() ? levels : [];
        layers.batch(() => {
            for (const id of [...layered]) {
                if (wanted.some((level) => level.id === id)) continue;
                layers.remove(id);
                layered.delete(id);
            }
            for (const level of wanted) {
                layers.push({ id: level.id, el: level.el, page: true, modal: false, focus: false, opener: null,
                    onClose: () => backFrom(level) });
                layered.add(level.id);
            }
        });
    }

    function announce() {
        if (!shown()) return;
        window.dispatchEvent(new CustomEvent('ouro:settings-page-shown', { detail: { path: state.settingsPath } }));
    }

    // From outside Settings (app.js `openSettings`): Settings is shown first, and refused when the page left vetoes.
    // A path already open keeps its level and its place (a Dashboard tab's Back returns there); any other starts anew.
    async function openFromOutside(path = '', { row = '', back = null } = {}) {
        const target = String(path || '');
        if (isDoorId(target)) return leave(target, { row });
        opening = true;
        try {
            if (!await showPage('settings')) return false;
        } finally { opening = false; }
        const at = levels.findIndex((level) => level.path === target);
        if (!target || (at >= 0 && !back)) closeLevels(target ? at + 1 : 0, { row });
        else go(target, { row, back, fresh: true });
        return true;
    }

    const onScroll = (event) => {
        const box = event.target;
        if (box === homeEl || levels.some((level) => level.el === box)) places.set(box, box.scrollTop);
    };
    const onPageShown = (event) => {
        syncLayers();
        if (event.detail?.page === 'settings' && !opening) announce();
    };
    const onVisible = () => { if (!document.hidden) announce(); };
    const offReconnect = ws?.on?.('open', () => announce());
    // A width that puts the home beside the page, or takes it away: the layers follow, and the home comes back
    // where it was left.
    const resize = typeof ResizeObserver === 'function' ? new ResizeObserver(() => {
        if (beside() === besideBefore) return;
        besideBefore = beside();
        if (besideBefore) homeEl.scrollTop = places.get(homeEl) || 0;
        syncLayers();
    }) : null;
    resize?.observe(shell);
    const disposeFade = bindScrollFade(homeEl);
    shell.addEventListener('scroll', onScroll, { capture: true, passive: true });
    window.addEventListener('ouro:page-shown', onPageShown);
    document.addEventListener('visibilitychange', onVisible);
    // Leaving Settings puts every revealed secret away again.
    setBeforePageLeave?.(({ from }) => {
        if (from === 'settings') resetSecretReveals(page);
        return true;
    });
    window.addEventListener('pagehide', (event) => {
        resetSecretReveals(page);
        if (event.persisted) return;
        resize?.disconnect();
        disposeFade();
        offReconnect?.();
        window.removeEventListener('ouro:page-shown', onPageShown);
        document.removeEventListener('visibilitychange', onVisible);
        home.dispose();
    });

    // Loopback-only debug hook, like app.js's __ouroWs: Settings has no URL, so browser tests open a path through it.
    window.__ouroSettings = { open: openFromOutside };
    return { page, open: openFromOutside };
}

// A declarative extension form must show what is STORED (posting the bare schema once unbound the
// Telegram skill's owner chat id). The host reads the current values from the SAME route it posts
// to. A route with no GET handler (404/405) cannot say what is stored: its fields start locked
// behind a warning until the owner chooses Edit anyway; any other failed read keeps Save disabled,
// and a successful `{}` is a real empty document.
const EXTENSION_VALUES_UNREADABLE = 'Current values could not be read; Save is disabled so it does not overwrite them. Use Reload to retry.';
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
