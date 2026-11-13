/** The plain pages of Settings (docs/DESIGN.md "Settings"; the shell in settings.js draws each page's title and Back):
 * Permissions and checks, Your rules, Appearance, Notifications, Voice and calls, This computer with its Advanced and
 * About. A page is cards of rows. A row is described in settings_rows.js and drawn and written by settings_controls.js,
 * so it saves itself and says what happened; a row whose control belongs to another module (the theme, the interface
 * size, the notifier, the host's two switches, the language) is drawn here on the same frame and keeps that module's
 * hooks. A page reads the settings when it is shown (`ouro:settings-page-shown` for its path) and after one of its
 * rows saved. */
import { apiClient, apiFetch } from './api_client.js';
import { openConfirmDialog } from './confirm_dialog.js';
import { getContributions } from './contributions.js';
import { mountDesktopShell } from './desktop_shell.js';
import { mountInterfaceSize } from './interface_size.js';
import { getNotifier } from './notifications.js';
import { bindAutostartControl } from './settings_autostart.js';
import { CODECS, HIDDEN_WHEN, bindSettingRows, settingsRow } from './settings_controls.js';
import { readSkills, skillFormTitle, skillForms } from './settings_extension_forms.js';
import { bindLanguageSettings, languageBlockHtml } from './settings_language.js';
import { registerSettingsPage } from './settings_pages.js';
import { applyPolicyToRows } from './settings_policy.js';
import { rowById, rowsOf } from './settings_rows.js';
import { mountRulesPage, refreshSafetySkipCounter, safetyFacts } from './settings_safety.js';
import { bindSecretRows, secretRowHtml } from './settings_secrets.js';
import { getSpeech } from './speech_providers.js';
import { showToast } from './toast.js';
import { copyTextWithToast } from './ui_helpers.js';
import { setResult } from './ui_primitives.js';
import { escapeHtmlAttr as escapeHtml, loadVersion } from './utils.js';

/* ---- A page of rows ------------------------------------------------------------------------------------------------ */

const lead = (text) => `<p class="settings-page-lead">${text}</p>`;
const note = (attrs, text = '') => `<p class="settings-card-note" ${attrs}>${text}</p>`;
/** A card: a question as its title, one lead line, its rows, and what stands after them. */
function card({ title = '', lead: line = '', rows = '', after = '', attrs = '', held = false }) {
    const head = title ? `<h3 class="settings-card-title">${title}</h3>` : '';
    return `<section class="ui-card settings-card"${attrs ? ` ${attrs}` : ''}>`
        + `${held ? `<div class="settings-card-head">${head}<span class="ui-status settings-card-held" data-tone="neutral">Kept for when it is on</span></div>` : head}`
        + `${line ? `<p class="settings-card-lead">${line}</p>` : ''}<div class="settings-rows">${rows}</div>${after}</section>`;
}
/** The page's one fold, at its end: its rare rows, named by what it holds. */
const foldCard = (holds, rows) => `<section class="ui-card settings-card"><details class="ui-disclosure" data-settings-fold>`
    + `<summary>More: ${holds}</summary><div class="settings-rows">${rows}</div></details></section>`;
const rowNode = (container, id) => [...container.querySelectorAll('[data-setting-row]')].find((node) => node.dataset.settingRow === id) || null;
/** Give a part of a drawn row (`part`: a selector inside it, or '' for the row) the hook another module binds by. */
function hook(container, id, part, name) {
    const node = part ? rowNode(container, id)?.querySelector(part) : rowNode(container, id);
    node?.setAttribute(name, '');
    return node || null;
}
const switchHtml = (row, attrs) => `<label class="ui-switch"><input type="checkbox" role="switch" ${attrs} aria-labelledby="setting-${row.id}-name"`
    + ` aria-describedby="setting-${row.id}-meta setting-${row.id}-result"><span class="ui-switch-track" aria-hidden="true"><span class="ui-switch-thumb"></span></span></label>`;
const factHtml = (name) => `<span class="settings-row-value" data-fact="${name}"></span>`;
const setFact = (container, name, text) => {
    const node = container.querySelector(`[data-fact="${name}"]`);
    if (node && node.textContent !== text) node.textContent = text;
};

// What the drawn rows cannot repaint: which rows, their kind, and the options offered (a `choices` row draws each
// option's line too). A read that changes it draws the page again; any other fact is painted in place.
const shapeOf = (rows) => JSON.stringify(rows.map((row) => [row.id, row.control.kind,
    (row.control.options || []).map((option) => [option.value, option.label, row.control.kind === 'choices' ? option.note : ''])]));

/**
 * Mount a page of rows. `ids` are its rows; `draw(row, ctx)` answers the page's HTML (`row(id)` is the page's own copy
 * of a descriptor, changed in place by `tune(rows, settings, row)` for what a read shows); `bind(container, ctx)` wires
 * what is not a row and answers its disposer; `paint(container, settings)` writes the facts beside the rows;
 * `shown(container, again)` runs whenever the page is shown (`again()` paints once more, for a fact it read itself).
 * Remembers the last read, so a page opened again is drawn right.
 */
function rowsPage(path, { ids, draw, tune = null, bind = null, paint = null, shown = null }) {
    let known = null;
    return (container, { ctx }) => {
        const rows = ids().map((id) => structuredClone(rowById(id)));
        const row = (id) => rows.find((each) => each.id === id);
        let handle = null;
        let unbind = null;
        let shape = '';
        let gone = false;
        const render = () => {
            handle?.dispose();
            unbind?.();
            container.innerHTML = draw(row, ctx);
            unbind = bind?.(container, ctx) || null;
            handle = bindSettingRows(container, { rows, ask: ctx.ask, openDoor: (door) => ctx.openSettings?.(door.control.to) });
            shape = shapeOf(rows);
        };
        const apply = (settings) => {
            known = settings;
            tune?.(rows, settings, row);
            if (shapeOf(rows) !== shape) render(); else handle.paint();
            paint?.(container, settings, row);
        };
        const read = async () => {
            let settings;
            try { settings = await ctx.readSettings(); } catch { return; }  // the rows keep what they show
            if (!gone) apply(settings);
        };
        const onShown = (event) => {
            if (event.detail?.path !== path) return;
            shown?.(container, () => { if (known && !gone) apply(known); });
            void read();
        };
        // One of this page's rows saved: what the server derives from it (the level in force, what waits for a restart)
        // is read again. A result the owner saw ends when Settings is left.
        const onSaved = (event) => { if (event.detail?.source === 'settings-row' && event.detail.page === path) void read(); };
        const onPage = (event) => { if (event.detail?.page !== 'settings') handle?.leave(); };
        const win = globalThis.window;
        tune?.(rows, known, row);
        render();
        if (known) paint?.(container, known, row);
        win?.addEventListener?.('ouro:settings-page-shown', onShown);
        win?.addEventListener?.('ouro:settings-updated', onSaved);
        win?.addEventListener?.('ouro:page-shown', onPage);
        return { unmount() {
            gone = true;
            handle?.dispose();
            unbind?.();
            win?.removeEventListener?.('ouro:settings-page-shown', onShown);
            win?.removeEventListener?.('ouro:settings-updated', onSaved);
            win?.removeEventListener?.('ouro:page-shown', onPage);
        } };
    };
}

/* ---- Permissions and checks ---------------------------------------------------------------------------------------- */

/** The rows as this installation's policy and Safety facts leave them (pure; `rows` are the page's copies). */
export function tunePermissionRows(rows, settings, row = (id) => rows.find((each) => each.id === id)) {
    applyPolicyToRows(rows, settings, rowById);
    const facts = safetyFacts(settings || {});
    const supervisor = row('permissions:supervisor');
    const second = row('permissions:second-check');
    supervisor.meta = settings ? facts.supervisor : '';
    second.meta = settings ? `${facts.secondCheck}${facts.unmeasured ? ' · not measured yet: no replay of real checks exists' : ''}` : '';
    // An organization's Supervisor is said, never offered for change: its two rows lead nowhere.
    for (const each of [supervisor, second]) each.control = facts.fixed ? { kind: 'custom' } : structuredClone(rowById(each.id).control);
    return rows;
}

const mountPermissions = rowsPage('permissions', {
    ids: () => rowsOf('permissions').map((row) => row.id),
    tune: tunePermissionRows,
    draw: (row) => {
        const rows = (...ids) => ids.map((id) => settingsRow(row(id))).join('');
        return lead('What Ouroboros may change, and who looks before and after.')
            + card({ title: 'What it may change', rows: rows('OUROBOROS_RUNTIME_MODE') })
            + card({ title: 'Who watches its actions', lead: 'The Safety Supervisor is a second model that reads a step before it runs.',
                rows: rows('OUROBOROS_SAFETY_MODE', 'OUROBOROS_CONSCIOUSNESS_SAFETY_MODE', 'permissions:rules', 'permissions:supervisor', 'permissions:second-check'),
                after: note('data-safety-skips role="status"') + note('data-safety-spares role="status" hidden') + note('data-safety-exits hidden')
                    + note('data-safety-organization-rules hidden', 'Your organization’s rules apply too; its administrator sets them.') })
            + card({ title: 'Who checks its results', lead: 'Reviewers are other models that read finished work.',
                rows: rows('OUROBOROS_TASK_REVIEW_MODE', 'permissions:reviewers') })
            + card({ title: 'Trust it without asking', rows: rows('OUROBOROS_AUTO_GRANT_REVIEWED_SKILLS') })
            + foldCard('review rounds', rows('OUROBOROS_REVIEW_MAX_CYCLES'));
    },
    paint: (container, settings) => {
        const facts = safetyFacts(settings || {});
        const exits = container.querySelector('[data-safety-exits]');
        exits.textContent = facts.exits;
        exits.hidden = !facts.exits;
        // An organization's checker records a walk whenever a route of its own lacks a credential, its own answer included.
        container.querySelector('[data-safety-spares]').hidden = facts.fixed;
        container.querySelector('[data-safety-organization-rules]').hidden = !facts.organizationRules;
    },
    shown: (container) => { void refreshSafetySkipCounter(container); },
});

/* ---- Appearance ---------------------------------------------------------------------------------------------------- */

// Theme and size are this device's and their own modules write them (theme.js paints and takes the strip's presses,
// interface_size.js the slider); the language is the install's (settings_language.js). Nothing here is a settings row
// the helper writes, so the page binds none. The theme's options stand in theme.js's own order, the one its arrows,
// Home and End walk.
function mountAppearance(container) {
    const custom = (row) => (row.id === 'appearance:theme'
        ? `<div class="ui-segment-group" role="radiogroup" data-theme-control aria-labelledby="setting-${row.id}-name">${[['light', 'Light'], ['dark', 'Dark'], ['system', 'System']]
            .map(([value, label]) => `<button type="button" class="ui-segment" role="radio" data-theme-choice="${value}">${label}</button>`).join('')}</div>`
        : `<span class="settings-slider" data-interface-size-row hidden><input type="range" id="interface-size" min="50" max="200" step="5" value="100"
            data-interface-size-input aria-labelledby="setting-${row.id}-name"><output for="interface-size" data-interface-size-value data-i18n-skip>100%</output>
            <button type="button" class="btn btn-default btn-sm" data-interface-size-reset>Reset</button></span>`);
    container.innerHTML = card({ title: 'On this device', attrs: 'data-interface-size',
        rows: ['appearance:theme', 'appearance:size'].map((id) => settingsRow(rowById(id), { custom })).join(''),
        after: note('data-interface-size-note role="status" aria-live="polite"') + note('class="settings-card-note theme-status" data-theme-status role="status" aria-live="polite"')
            + note('data-shell-storage-status role="status" aria-live="polite"') })
        + languageBlockHtml();
    globalThis.ouroTheme?.mount();
    // The desktop app says where it keeps this device's choices (a fact of the app, which is not updated with the core).
    const disposers = [mountInterfaceSize(container), bindLanguageSettings(container), mountDesktopShell(container)];
    return { unmount() { disposers.forEach((dispose) => dispose()); } };
}

/* ---- Notifications ------------------------------------------------------------------------------------------------- */

// This device's choices: notifications.js keeps them, takes every press by its own hooks and paints them
// (`mountSettings`); a switch acts at once. The first row's one line is the notifier's sentence about this system.
function mountNotifications(container, { ctx }) {
    const custom = (row) => (row.control.pref ? switchHtml(row, `data-notify-pref="${row.control.pref}"`)
        : row.id === 'notify:tone_volume'
            ? `<span class="settings-slider"><input type="range" id="notify-tone-volume" min="0" max="100" step="1" value="40" data-notify-volume
                aria-labelledby="setting-${row.id}-name" aria-describedby="setting-${row.id}-meta"><output for="notify-tone-volume" data-notify-volume-value data-i18n-skip>40%</output>
                <button type="button" class="btn btn-default btn-sm" data-notify-preview>Preview tone</button></span>`
            : '<button type="button" class="btn btn-default btn-sm" data-notify-test>Send a test</button>');
    const rows = (...ids) => ids.map((id) => settingsRow(rowById(id), { custom })).join('');
    container.innerHTML = lead('When Ouroboros calls you back to this device.')
        + `<div class="settings-stack">${card({ rows: rows('notify:enabled') })
            + card({ title: 'Notify me about', held: true, rows: rows('notify:needs_answer', 'notify:task_done', 'notify:important', 'notify:main_reply') })
            + card({ title: 'How', held: true, rows: rows('notify:sound', 'notify:tone_volume', 'notify:show_text', 'notify:test') })}</div>`
        + `<div data-notify-doors>${card({ rows: rows('notify:places') })}</div>`;
    hook(container, 'notify:enabled', '[data-setting-meta]', 'data-notify-status');
    hook(container, 'notify:test', '[data-setting-meta]', 'data-notify-attention-status');
    getNotifier().mountSettings(container);
    // While the first switch is off the other choices are kept, not used: their cards say so.
    const master = container.querySelector('[data-notify-pref="enabled"]');
    const held = () => container.querySelectorAll('.settings-card-held').forEach((node) => { node.hidden = Boolean(master?.checked); });
    held();
    container.addEventListener('change', held);
    const doors = bindSettingRows(container.querySelector('[data-notify-doors]'), { openDoor: (door) => ctx.openSettings?.(door.control.to) });
    return { unmount() { container.removeEventListener('change', held); doors.dispose(); } };
}

/* ---- Voice and calls ----------------------------------------------------------------------------------------------- */

const skillFormPath = (skill) => `skill/${encodeURIComponent(skill)}`;

/** The saved dictation engine and read-aloud voice in words, and the skills that registered a live conversation and a
 *  settings form, each named by its form's own title (`store`: the contributions store; `speech`: the speech hub;
 *  `forms`: the skill forms the last read of the skills knows). Pure over the three. */
export function voiceFacts({ speech = getSpeech(), store = getContributions(), forms = skillForms(),
    title = (skill) => skillFormTitle(skillFormPath(skill)) } = {}) {
    const engine = speech.selected('stt');
    const reader = speech.readAloud();
    let voices = [];
    try { voices = reader?.available ? reader.impl?.voices?.() || [] : []; } catch { voices = []; }
    const voice = speech.voiceId();
    const formed = new Set(forms.map((form) => form.skill));
    const talking = new Set(store.providers('conversation').map((item) => item.skill).filter((skill) => skill && formed.has(skill)));
    return {
        dictation: !engine.key ? 'Not chosen' : engine.label ? `${engine.label}${engine.available ? '' : ' · not ready here'}` : 'Chosen engine is not here',
        dictationNote: engine.key && !engine.available ? engine.reason || '' : '',
        readAloud: !reader ? 'No voice on this device' : !reader.available ? `${reader.label || 'Chosen engine'} · not ready here`
            : `${voices.find((each) => each.id === voice)?.label || 'Automatic'}${reader.label ? ` · ${reader.label}` : ''}`,
        readAloudNote: reader && !reader.available ? reader.reason || '' : '',
        calls: [...talking].map((skill) => ({ skill, name: title(skill) })),
    };
}

// No setting lives here: each row says a saved choice and where it is made. A skill's own form is one press away: the
// skills are read when the page is shown (a passive read), which is how the page knows who has a form and its title.
function mountVoice(container, { ctx }) {
    const speech = getSpeech();
    const store = getContributions();
    const base = { 'voice:dictation': rowById('voice:dictation'), 'voice:read-aloud': rowById('voice:read-aloud') };
    let handle = null;
    let drawn = '';
    const render = () => {
        const facts = voiceFacts({ speech, store });
        const doors = facts.calls.map((call) => ({ id: `voice:skill:${call.skill}`, keys: [], page: 'voice', fold: false, name: call.name,
            meta: 'Its own settings, on its own page.', more: '', control: { kind: 'door', to: skillFormPath(call.skill) } }));
        const rows = [{ ...base['voice:dictation'], meta: facts.dictationNote || base['voice:dictation'].meta },
            { ...base['voice:read-aloud'], meta: facts.readAloudNote || base['voice:read-aloud'].meta }, ...doors];
        const shape = JSON.stringify([rows.map((row) => [row.id, row.name, row.meta]), facts.dictation, facts.readAloud]);
        if (shape === drawn) return;
        drawn = shape;
        handle?.dispose();
        const doorValue = (row) => factHtml(row.id);
        container.innerHTML = card({ rows: rows.slice(0, 2).map((row) => settingsRow(row, { doorValue })).join('') })
            + (doors.length ? card({ title: 'Calls', lead: 'Talk to Ouroboros by voice from the message field.', rows: doors.map((row) => settingsRow(row)).join('') }) : '');
        setFact(container, 'voice:dictation', facts.dictation);
        setFact(container, 'voice:read-aloud', facts.readAloud);
        handle = bindSettingRows(container, { rows, openDoor: (door) => ctx.openSettings?.(door.control.to) });
    };
    render();
    let gone = false;
    const read = () => readSkills().then(() => { if (!gone) render(); }, () => {});  // a failed read keeps what is drawn
    const onShown = (event) => { if (event.detail?.path === 'voice') void read(); };
    globalThis.window?.addEventListener?.('ouro:settings-page-shown', onShown);
    const offs = [speech.onChange(render), store.onChange(() => { render(); void read(); })];
    return { unmount() {
        gone = true;
        globalThis.window?.removeEventListener?.('ouro:settings-page-shown', onShown);
        offs.forEach((off) => off?.());
        handle?.dispose();
    } };
}

/* ---- This computer ------------------------------------------------------------------------------------------------- */

const HOST_KEY = 'OUROBOROS_SERVER_HOST';
const PASSWORD_KEY = 'OUROBOROS_NETWORK_PASSWORD';
// The Settings writer takes a loopback or a wildcard address only (gateway/settings.py); any other is set outside.
const hostChoice = (host) => {
    const text = String(host ?? '').trim().toLowerCase();
    return /^(|localhost|127(\.\d+){3}|::1|\[::1\])$/.test(text) ? '127.0.0.1' : /^(0\.0\.0\.0|::|\[::\])$/.test(text) ? '0.0.0.0' : text;
};
let typedPassword = '';  // the password typed on this visit of the page, until an opening write carried it
CODECS.network = {
    read: (settings) => (settings ? hostChoice(settings[HOST_KEY]) : undefined),
    // Opening to the network carries the password typed beside it in the same write: the server wants one with it.
    // (The Password row also saves itself when its field is left; the same value twice changes nothing.)
    body: (value) => {
        const password = value === '0.0.0.0' ? typedPassword : '';
        typedPassword = '';
        return { [HOST_KEY]: value, ...(password ? { [PASSWORD_KEY]: password } : {}) };
    },
    running: (settings) => {
        // The address in force while the saved one waits for a restart, as one of the two choices.
        const applied = settings?._meta?.restart_state?.running?.[HOST_KEY];
        return applied === undefined ? undefined : hostChoice(applied);
    },
};
/** The address other devices type, from a settings read's `_meta`: `{ url, copy, note, warning }` (the row's one line,
 *  and the server's warning about it), or null while only this computer reaches it. */
export function lanAddress(meta) {
    const found = meta?.reachability === 'lan_reachable';
    if (!found && meta?.reachability !== 'host_ip_unknown') return null;
    return { url: String(meta.recommended_url || ''), copy: found && Boolean(meta.recommended_url), warning: String(meta.warning || ''),
        note: found ? 'Type it in a browser on the same network.'
            : 'This computer’s address on your network could not be found. Put its address in place of your-host-ip.' };
}
HIDDEN_WHEN.noLanAddress = (settings) => !lanAddress(settings?._meta);

/** The network rows as a read leaves them (pure): the password row says what an unset one means (a set one says Saved
 *  in its place, settings_secrets.js), and a listener the way Ouroboros was started decides is said in place of what
 *  each choice would mean. */
export function tuneComputerRows(rows, settings, row = (id) => rows.find((each) => each.id === id)) {
    const host = row(HOST_KEY);
    const password = row(PASSWORD_KEY);
    const base = rowById(HOST_KEY);
    const meta = settings?._meta || {};
    const saved = hostChoice(settings?.[HOST_KEY]);
    const listens = hostChoice(meta.bind_host);
    const pending = CODECS.network.running(settings) !== undefined;
    const outside = Boolean(settings) && Boolean(meta.bind_host) && listens !== saved && !pending;
    host.control = { ...base.control, options: base.control.options.map((option) => (outside
        ? { ...option, note: `It listens on ${meta.bind_host} now: the way Ouroboros was started sets that, not this choice.` } : { ...option })) };
    host.meta = settings && !base.control.options.some((option) => option.value === saved) ? `Listens on ${saved}: set outside Settings.` : '';
    password.meta = !settings || settings[PASSWORD_KEY] ? rowById(PASSWORD_KEY).meta
        : 'Not set. Other devices are asked for it: set it before you open to your network.';
    const address = lanAddress(meta);
    row('computer:address').meta = address?.note || rowById('computer:address').meta;
    // The version the sidebar already shows (one health read, utils.js loadVersion).
    const version = globalThis.document?.getElementById?.('nav-version')?.textContent || '';
    row('computer:about').meta = version ? `Ouroboros ${version}` : '';
    return rows;
}

// One listener for as long as the page stands: a press copies the address the last read drew on its button.
export function bindNetworkHintCopy(host, { copy = copyTextWithToast } = {}) {
    const onClick = (event) => {
        const url = event.target?.closest?.('[data-copy-lan-url]')?.dataset.copyLanUrl;
        if (url) void copy(url, { okMessage: 'LAN URL copied.' });
    };
    host?.addEventListener('click', onClick);
    return () => host?.removeEventListener('click', onClick);
}

/** Paint the address row's control from a read's `_meta`: the address as a link, and Copy link carrying that address. */
export function paintNetworkHint(host, meta) {
    const address = lanAddress(meta);
    if (!host || !address) { if (host) host.innerHTML = ''; return; }
    const url = escapeHtml(address.url);
    host.innerHTML = (address.copy ? `<a class="settings-row-value settings-row-link" href="${url}" target="_blank" rel="noopener">${url}</a>`
        + `<button type="button" class="btn btn-default btn-sm ui-hit-area" data-copy-lan-url="${url}">Copy link</button>`
        : `<span class="settings-row-value">${url}</span>`);
}

// The password is a stored key's row (settings_secrets.js): Replace, Show, and Remove that asks first.
const passwordRows = new WeakMap();  // the page's container -> its bound password row
const mountComputer = rowsPage('computer', {
    ids: () => rowsOf('computer').map((row) => row.id),
    tune: tuneComputerRows,
    draw: (row) => {
        const custom = (each) => (each.id === 'computer:address' ? '<span class="settings-row-end" data-lan-address></span>'
            : switchHtml(each, each.id === 'computer:autostart' ? 'data-autostart-toggle' : 'data-background-toggle'));
        const rows = (...ids) => ids.map((id) => (id === PASSWORD_KEY ? secretRowHtml(row(id)) : settingsRow(row(id), { custom }))).join('');
        return lead('The computer Ouroboros runs on.')
            + card({ title: 'Start and background', attrs: 'data-autostart-settings hidden', rows: rows('computer:autostart', 'OUROBOROS_DESKTOP_KEEP_RUNNING'),
                after: note('id="settings-autostart-note" data-autostart-shared-note role="status" aria-live="polite" hidden') })
            + card({ title: 'Who can reach it', rows: rows(HOST_KEY, PASSWORD_KEY, 'computer:address') })
            + card({ rows: rows('computer:advanced', 'computer:about') });
    },
    bind: (container, ctx) => {
        // The host's two switches are settings_autostart.js's: it reads the host, paints and writes them. Each says
        // its state in its own row's line; a host that offers neither says why once, under both.
        hook(container, 'computer:autostart', '[data-setting-result]', 'data-autostart-status');
        hook(container, 'OUROBOROS_DESKTOP_KEEP_RUNNING', '[data-setting-result]', 'data-background-status');
        const background = hook(container, 'OUROBOROS_DESKTOP_KEEP_RUNNING', '', 'data-background-row');
        if (background) background.hidden = true;
        // Heard before the row helper takes the change and empties the field it saved.
        const onTyped = (event) => { if (event.target?.id === `setting-${PASSWORD_KEY}` && event.target.value) typedPassword = event.target.value; };
        container.addEventListener('change', onTyped, true);
        // A password write that landed puts its field away (Remove writes '' through the row's own queue).
        const password = bindSecretRows(container, { rowOf: rowById, ask: ctx.ask });
        passwordRows.set(container, password);
        const onSaved = (event) => { if (event.detail?.row === PASSWORD_KEY) password.close(PASSWORD_KEY); };
        globalThis.window?.addEventListener?.('ouro:settings-updated', onSaved);
        const disposers = [bindAutostartControl(container), bindNetworkHintCopy(container)];
        return () => {
            typedPassword = '';
            container.removeEventListener('change', onTyped, true);
            globalThis.window?.removeEventListener?.('ouro:settings-updated', onSaved);
            password.dispose();
            disposers.forEach((dispose) => dispose());
        };
    },
    paint: (container, settings) => {
        passwordRows.get(container)?.paint(settings);
        paintNetworkHint(container.querySelector('[data-lan-address]'), settings?._meta);
        // The server's warning about the open address (no password, for one) stands in that row's line, as a warning.
        const warning = lanAddress(settings?._meta)?.warning || '';
        setResult(rowNode(container, 'computer:address')?.querySelector('[data-setting-result]'), warning ? { text: warning, tone: 'warn' } : {});
    },
});

/* This computer › Advanced: what an installation rarely changes. */

// What the core serves at its own address right now (`served_ui` on /api/state),
// beside the saved choice that applies at the next restart; a refused choice
// carries the core's own sentence naming it and why. A core older than the choice
// reports nothing, and nothing is shown.
export function servedUiStatus(servedUi) {
    const label = { builtin: 'the built-in interface', directory: 'the interface directory', none: 'no main page' }[servedUi?.source];
    if (!label) return null;
    return servedUi.error
        ? { text: `Serving now: ${label}. ${servedUi.error}`, tone: 'warn' }
        : { text: `Serving now: ${label}.`, tone: 'muted' };
}

// The cap on acting tasks is optional: the server keeps "no cap" as a blank, which the row shows and sends as No limit.
CODECS.ownerCap = {
    read: (settings, row) => (!settings ? undefined : String(settings[row.keys[0]] ?? '').trim() === '' ? 'unlimited' : settings[row.keys[0]]),
    body: (value, row) => ({ [row.keys[0]]: String(value ?? '').trim().toLowerCase() === 'unlimited' ? '' : value }),
};

let servedUi = null;  // the last answer of /api/state about what is served
const mountAdvanced = rowsPage('computer/advanced', {
    ids: () => rowsOf('computer/advanced').map((row) => row.id),
    draw: (row) => lead('Most installations never change these.')
        + card({ rows: rowsOf('computer/advanced').map((each) => settingsRow(row(each.id))).join('') }),
    // What is served now stands in the Interface folder row, after its one line: a fact of the running core, in the
    // warning's tone when it refused the saved folder.
    bind: (container) => {
        const fact = globalThis.document.createElement('span');
        fact.className = 'ui-result settings-row-result';
        fact.setAttribute('data-served-ui', '');
        fact.setAttribute('role', 'status');
        rowNode(container, 'OUROBOROS_UI_DIR')?.querySelector('.ui-row-meta')?.append(fact);
        return null;
    },
    paint: (container) => setResult(container.querySelector('[data-served-ui]'), servedUiStatus(servedUi) || {}),
    shown: (container, again) => { apiClient.state().then((state) => { servedUi = state?.served_ui || null; again(); }, () => {}); },
});

/* This computer › About: the version and what Ouroboros is; Reset all data in its fold. */
function mountAbout(container) {
    const custom = () => '<button type="button" class="btn btn-danger btn-sm" data-reset-all>Reset all data…</button>';
    container.innerHTML = `<section class="ui-card settings-card"><div class="about-body">
            <img src="/static/logo.jpg" class="about-logo" alt="">
            <div><p class="about-title">Ouroboros</p><p id="about-version" class="about-version"></p></div>
            <p class="about-desc">A self-creating AI agent. Not a tool, but a becoming digital personality with its own constitution,
                persistent identity, and background consciousness. Born February 16, 2026.</p>
            <div class="about-credits"><span>Created by <strong>Anton Razzhigaev</strong> &amp; <strong>Andrew Kaznacheev</strong></span>
                <div class="about-links"><a href="https://t.me/abstractDL" target="_blank" rel="noopener noreferrer">@abstractDL</a>
                    <a href="https://github.com/razzant/ouroboros" target="_blank" rel="noopener noreferrer">GitHub</a></div></div>
        </div></section>${foldCard('reset all data', settingsRow(rowById('computer:reset'), { custom }))}`;
    void loadVersion();  // utils.js fills `#about-version` from one health read
    const button = container.querySelector('[data-reset-all]');
    const onReset = () => { void resetAllData(); };
    button.addEventListener('click', onReset);
    return { unmount() { button.removeEventListener('click', onReset); } };
}

registerSettingsPage('permissions', { title: 'Permissions and checks', mount: mountPermissions });
registerSettingsPage('rules', { title: 'Your rules', mount: mountRulesPage });
registerSettingsPage('appearance', { title: 'Appearance', mount: mountAppearance });
registerSettingsPage('notifications', { title: 'Notifications', mount: mountNotifications });
registerSettingsPage('voice', { title: 'Voice and calls', mount: mountVoice });
registerSettingsPage('computer', { title: 'This computer', mount: mountComputer });
registerSettingsPage('computer/advanced', { title: 'Advanced', mount: mountAdvanced });
registerSettingsPage('computer/about', { title: 'About', mount: mountAbout });

// The desktop app's own facts (its system notifications, where it keeps this device's choices) reach the notifier when
// the app starts, not when a page is opened: where an alert goes depends on them long before anyone opens Settings.
if (globalThis.document) mountDesktopShell(null, (shell) => getNotifier().configure({ shell }));

// Reset All Data (the Danger Zone's button): asks first, then deletes the runtime data and restarts.
export async function resetAllData() {
    const confirmedReset = await openConfirmDialog({
        title: 'Reset runtime data',
        body: 'This will delete all runtime data (state, memory, logs, settings) and restart.\nThe repo (agent code) will be preserved.\nYou will need to re-enter your provider settings.\n\nContinue?',
        confirmLabel: 'Delete and restart',
        danger: true,
    });
    if (!confirmedReset) return;
    try {
        const res = await apiFetch('/api/reset', { method: 'POST' });
        const data = await res.json();
        if (data.status === 'ok') {
            await openConfirmDialog({
                title: 'Reset complete',
                body: 'Deleted: ' + (data.deleted.join(', ') || 'nothing') + '\nRestarting...',
                alert: true,
            });
        } else {
            await openConfirmDialog({
                title: 'Reset failed',
                body: 'Error: ' + (data.error || 'unknown'),
                alert: true,
            });
        }
    } catch (e) {
        showToast('Reset failed: ' + e.message, 'error');
    }
}
