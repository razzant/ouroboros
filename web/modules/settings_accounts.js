/** Settings › Accounts and keys (docs/DESIGN.md "Settings"): what Ouroboros signs in with. This module only composes:
 * the page family `accounts` out of three sources that stay in their own modules.
 *
 * - `accounts`: the subscriptions as one head per provider (harness_accounts.js says the facts), the API keys as one
 *   head per key (settings_connections.js), then the other stored keys as rows (settings_secrets.js, the row helper).
 * - `accounts/<harness>`: one provider's page. harness_accounts.js draws and runs it inside the frame drawn here: the
 *   program's row above the accounts, then the accounts, the ones that need sign-in first.
 * - `accounts/key/<provider>[/<name>]`, `accounts/key/new`: one API key's page; `accounts/key/custom`: a key of the
 *   owner's own.
 *
 * What starts a read: showing a provider's page is the owner's act that reads the accounts and the program, and may
 * start an idle agent daemon, as opening the Accounts tab was. The home and the list never do: the list draws the
 * shared accounts store as it stands (its subscriber is never a visible surface, so it arms no poll), says "Not
 * checked" for what nobody read, and offers one Check, which only reads. */

import { RESTART_LABEL } from './claudexor_restart.js';
import { FACET_ACCOUNTS, FACET_QUOTA, READ_OK, accountRows, claudexorStatus, familyLabel } from './claudexor_status_store.js';
import { accountGroups, destroyHarnessAccounts, initHarnessAccounts, needsSignIn, quotaSummary, renderAgentsServiceBanner,
    serviceBannerLine } from './harness_accounts.js';
import { knownProgram, maintenanceLatestLine } from './harness_maintenance.js';
import { harnessPresentation } from './harness_presentation.js';
import { strokeIcon } from './page_icons.js';
import { apiConnections } from './route_editor_primitives.js';
import { editorLine, keyHeads, keyPath, keyTitle, mountKeyPage, parseKeyPath, referencesLine, setKeyEnabled } from './settings_connections.js';
import { bindSettingRows, doorSign, settingsRow } from './settings_controls.js';
import { readSkills } from './settings_extension_forms.js';
import { registerSettingsPage } from './settings_pages.js';
import { PROVIDER_SECRET_KEYS, providerIconHtml } from './settings_provider_cards.js';
import { rowById } from './settings_rows.js';
import { bindSecretRows, mountCustomKeyPage, secretRowHtml, storedKeyRow } from './settings_secrets.js';
import { skillTitle } from './skill_presentation.js';
import { statusMarkHtml } from './status_mark.js';
import { revealNewRow } from './ui_helpers.js';
import { setResult } from './ui_primitives.js';
import { escapeHtmlAttr as escapeHtml } from './utils.js';

const CUSTOM_KEY_PATH = 'accounts/key/custom';
const CHEVRON = doorSign();
const chipHtml = (chip) => (chip ? `<span class="ui-chip settings-chip" data-tone="${chip.tone}">${escapeHtml(chip.text)}</span>` : '');
const switchHtml = (attrs, on) => `<label class="ui-switch"><input type="checkbox" role="switch" aria-checked="${on}" ${attrs}${on ? ' checked' : ''}>`
    + '<span class="ui-switch-track" aria-hidden="true"><span class="ui-switch-thumb"></span></span></label>';
const landed = (outcome) => outcome?.state === 'saved' || outcome?.state === 'saved_then_failed';
// The key of the home's row for this page (settings_home.js), for a key whose provider has no mark of its own.
const KEY_GLYPH = strokeIcon('<path d="m15.5 7.5 2.3 2.3a1 1 0 0 0 1.4 0l2.1-2.1a1 1 0 0 0 0-1.4L19 4"/><path d="m21 2-9.6 9.6"/><circle cx="7.5" cy="15.5" r="5.5"/>', 16);

/** A provider's mark for a head: its brand mark, else the neutral one (harness_presentation.js). */
function providerMarkHtml(harness) {
    const mark = harnessPresentation(harness);
    return `<svg width="16" height="16" viewBox="${mark.viewBox}" fill="currentColor" aria-hidden="true"><path d="${mark.path}"/></svg>`;
}

/**
 * What a provider's head says in the list, from what the shared accounts store already holds (`group`: one of
 * `accountGroups`): how many of its accounts take turns, and, as a chip, whether any needs the owner. `asked`: the
 * store was read at least once, or is being read; `program`: what this session already read about the provider's
 * program, of which the head says a waiting update. Nothing here reads.
 */
export function providerRowFacts(group, { accountsRead = READ_OK, asked = true, program = null } = {}) {
    const update = program?.updateAvailable === 'yes' ? `program update: ${maintenanceLatestLine(program)}` : '';
    const say = (words, chip = null) => ({ meta: [words, update].filter(Boolean).join(' · '), chip });
    if (!asked) return say('Not checked');
    const total = group.rows.length;
    if (!total) return say(group.status.label);
    const known = accountsRead === READ_OK;
    const waiting = group.rows.filter((row) => needsSignIn(row, known)).length;
    const turns = group.rows.filter((row) => row.enabled !== false && !needsSignIn(row, known)).length;
    return say(`${turns} of ${total} in rotation${known ? '' : ' · last known'}`,
        waiting ? { text: `${waiting} ${waiting === 1 ? 'needs' : 'need'} sign-in`, tone: 'warn' } : null);
}

/** The share of an account's window that is left (0 to 1), by the runtime's own bar: fresh readings of exactly this
 *  account, the window used most, a model's own cap left out. null when no reading says. */
export function quotaLeft(payload, row) {
    let used = null;
    for (const snap of payload?.quota || []) {
        const subject = snap?.subject || {};
        if (String(subject.harness || '') !== String(row.harness) || String(subject.subject_id || '') !== String(row.profile_id)
            || String(snap?.freshness || '') !== 'fresh') continue;
        for (const constraint of snap.constraints || []) {
            if (Array.isArray(constraint.applies_to_models) && constraint.applies_to_models.some(Boolean)) continue;
            const ratio = constraint.used_ratio;
            if (typeof ratio === 'number' && Number.isFinite(ratio) && (used === null || ratio > used)) used = ratio;
        }
    }
    return used === null ? null : Math.max(0, Math.min(1, 1 - used));
}

/**
 * A provider's accounts at a glance: one bar each, as tall as the quota it has left, and the line under them. `state`:
 * `ok` (in rotation; `left` null when its window was not read), `spent`, `need` (sign-in), `off` (out of rotation).
 * In the picture's order: the ones that take turns by what they have left, then spent, then the ones that wait.
 */
export function quotaBars(rows, payload, { accountsRead = READ_OK, quotaRead = READ_OK } = {}) {
    const known = accountsRead === READ_OK;
    const rank = { ok: 0, spent: 1, need: 2, off: 3 };
    const bars = rows.map((row) => {
        const spent = quotaSummary(payload?.quota || [], row.harness, row.profile_id, { quotaRead, absences: payload?.quota_absences || [] }).exhausted;
        const state = needsSignIn(row, known) ? 'need' : row.enabled === false ? 'off' : spent ? 'spent' : 'ok';
        return { state, left: state === 'ok' && quotaRead === READ_OK ? quotaLeft(payload, row) : null };
    }).sort((a, b) => rank[a.state] - rank[b.state] || (b.left ?? -1) - (a.left ?? -1));
    const count = (state) => bars.filter((bar) => bar.state === state).length;
    const words = [`${count('ok') + count('spent')} in rotation`, count('spent') ? `${count('spent')} spent` : '',
        count('need') ? `${count('need')} ${count('need') === 1 ? 'needs' : 'need'} sign-in` : '', count('off') ? `${count('off')} out of rotation` : ''];
    return { bars, words: words.filter(Boolean).join(' · ') };
}

/** The lead mark of an account's row: the owner's dot while it needs sign-in, else what is left of its window as a
 *  ring (an empty one when no reading says, or it is out of rotation). */
function accountMarkHtml({ row, badge }, payload) {
    if (row.enabled !== false && badge.tone === 'error') return statusMarkHtml('', { shape: 'dot', hue: 'owner' });
    const left = row.enabled === false ? null : quotaLeft(payload, row);
    const around = 2 * Math.PI * 9;
    const arc = (share, name) => `<circle class="${name}" cx="12" cy="12" r="9" stroke-dasharray="${(share * around).toFixed(2)} ${around.toFixed(2)}" transform="rotate(-90 12 12)"/>`;
    return `<svg class="settings-gauge" width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="3" stroke-linecap="round" aria-hidden="true">`
        + `${arc(1, 'settings-gauge-rail')}${left ? arc(Math.max(0.04, left), 'settings-gauge-lit') : ''}</svg>`;
}

/* ---- The list ------------------------------------------------------------------------------------------------------ */

let list = null;  // the mounted list: a key's page says there what it added or removed

function providerHeadHtml(group, facts) {
    return `<div class="ui-row settings-head" data-entry-id="${escapeHtml(group.harness)}" data-open-path="accounts/${encodeURIComponent(group.harness)}">
        <span class="settings-head-glyph" aria-hidden="true">${providerMarkHtml(group.harness)}</span>
        <button type="button" class="settings-head-open ui-hit-area" data-entry-open>
            <span class="ui-row-title">${escapeHtml(group.label)}${chipHtml(facts.chip)}</span>
            <span class="ui-row-meta">${escapeHtml(facts.meta)}</span>
        </button>
        <div class="ui-row-actions">${CHEVRON}</div>
    </div>`;
}

function keyHeadHtml(head, busy) {
    const use = head.fixed ? '' : switchHtml(`data-entry-switch aria-label="${escapeHtml(`Use ${head.title}`)}"${busy ? ' disabled aria-busy="true"' : ''}`, head.enabled);
    return `<div class="ui-row settings-head" data-entry-id="${escapeHtml(head.id)}" data-open-path="${escapeHtml(head.path)}">
        <span class="settings-head-glyph" aria-hidden="true">${providerIconHtml(head.card) || KEY_GLYPH}</span>
        <button type="button" class="settings-head-open ui-hit-area" data-entry-open>
            <span class="ui-row-title" data-i18n-skip>${escapeHtml(head.title)}${chipHtml(head.chip)}</span>
            <span class="ui-row-meta">${escapeHtml(head.meta)}</span>
        </button>
        <div class="ui-row-actions">${use}${CHEVRON}</div>
    </div>`;
}

const doorHtml = (card) => `<button type="button" class="ui-row settings-row" data-kind="door" data-open-path="${keyPath(card.kind)}">
        <span class="ui-row-title">${escapeHtml(card.title)}</span><span class="ui-row-meta"><span>${escapeHtml(card.hint || '')}</span></span>
        <span class="ui-row-actions">${CHEVRON}</span></button>`;

/**
 * The other stored keys of a settings document, as row descriptors in the page's order: the GitHub token and its
 * repository, the keys installed skills asked for (each once, under who asked), then the owner's own keys. A provider's
 * key is never listed here: it has its own head above.
 */
export function otherKeyRows(settings = {}, skills = []) {
    const base = new Set(['GITHUB_TOKEN', ...PROVIDER_SECRET_KEYS]);
    const askers = new Map();
    for (const skill of Array.isArray(skills) ? skills : []) {
        for (const raw of skill?.grants?.requested_keys || []) {
            const key = String(raw || '').trim();
            if (key && !base.has(key)) askers.set(key, [...(askers.get(key) || []), String(skillTitle(skill) || '')]);
        }
    }
    const asked = [...askers.keys()].sort((a, b) => a.localeCompare(b))
        .map((key) => storedKeyRow(key, `Asked by ${askers.get(key).filter(Boolean).join(', ') || 'a skill'}`));
    const own = (Array.isArray(settings?._meta?.custom_secret_keys) ? settings._meta.custom_secret_keys : [])
        .filter((key) => !askers.has(key) && !base.has(key)).map((key) => storedKeyRow(key, 'Your own key'));
    return [rowById('GITHUB_TOKEN'), rowById('GITHUB_REPO'), ...asked, ...own];
}

function mountList(container, { ctx = {} }) {
    const store = claudexorStatus;
    container.innerHTML = `<p class="settings-page-lead">What Ouroboros signs in with. One subscription or one key is enough to work.</p>
        <section class="ui-card settings-card">
            <h3 class="settings-card-title">Subscriptions</h3>
            <p class="settings-card-lead">Several accounts of one kind take turns, so a spent quota does not stop the work.</p>
            <div class="settings-rows" data-provider-rows></div>
            <p class="settings-card-note accounts-engine"><span class="ui-status" data-accounts-engine></span>
                <button type="button" class="btn btn-ghost btn-sm" data-accounts-check hidden>Check</button></p>
        </section>
        <section class="ui-card settings-card" data-setting-row="OUROBOROS_CONNECTIONS">
            <div class="settings-card-head"><h3 class="settings-card-title">API keys</h3>
                <span class="ui-result" data-entry-result role="status" aria-live="polite"></span>
                <button type="button" class="btn btn-default btn-sm" data-open-path="accounts/key/new">Add key</button></div>
            <p class="settings-card-lead">Paid by use. A role or an agent names the key it uses.</p>
            <div class="settings-rows" data-key-rows></div>
            <details class="ui-disclosure" data-settings-fold hidden><summary>More providers</summary><div class="settings-rows" data-more-rows></div></details>
        </section>
        <section class="ui-card settings-card">
            <div class="settings-card-head"><h3 class="settings-card-title">Other keys</h3>
                <span class="ui-result" data-other-result role="status" aria-live="polite"></span>
                <button type="button" class="btn btn-default btn-sm" data-open-path="${CUSTOM_KEY_PATH}">Add key</button></div>
            <p class="settings-card-lead">Tokens Ouroboros and its skills use to reach your services.</p>
            <div class="settings-rows" data-other-rows></div>
        </section>`;
    const find = (selector) => container.querySelector(selector);
    const [providersEl, keysEl, moreEl, otherEl, fold] = ['[data-provider-rows]', '[data-key-rows]', '[data-more-rows]', '[data-other-rows]', '[data-settings-fold]'].map(find);
    const drawn = { providers: '', keys: '', more: '', other: '' };
    const busy = new Set();   // keys whose switch is in flight
    let settings = null;
    let skills = [];
    const otherRows = new Map();  // each row keeps its field, caret and result while the skills list arrives
    let reveal = '';
    let gone = false;
    // A list is drawn again only when its words changed, so a head that holds focus is not replaced under it.
    const put = (name, host, html) => { if (drawn[name] !== html) host.innerHTML = drawn[name] = html; };

    function paintProviders(view = { reads: store.reads, snapshot: store.snapshot }) {
        const payload = view.snapshot || store.snapshot || {};
        const accountsRead = store.facet(FACET_ACCOUNTS);
        const asked = Boolean(store.everSettled || store.loading);
        const groups = accountGroups(payload, { accountsRead, catalogKnown: store.catalogKnown, accountsUnsupported: Boolean(store.accountsUnsupported) });
        put('providers', providersEl, groups.map((group) => providerHeadHtml(group,
            providerRowFacts(group, { accountsRead, asked, program: knownProgram(group.harness) }))).join(''));
        // The engine in one line, as the store last heard it; a store nobody read says so, and Check reads it once.
        const line = asked ? serviceBannerLine(store) : { tone: 'muted', text: 'Accounts were not checked yet.' };
        const engine = find('[data-accounts-engine]');
        if (engine.textContent !== line.text) engine.textContent = line.text;
        engine.dataset.tone = line.tone === 'error' ? 'danger' : line.tone;
        find('[data-accounts-check]').hidden = accountsRead === READ_OK || Boolean(store.loading);
    }

    function paintKeys() {
        if (!settings) return;
        const { keys, more } = keyHeads(settings);
        put('keys', keysEl, keys.map((head) => keyHeadHtml(head, busy.has(head.id))).join('')
            || '<p class="settings-card-note">No API key yet.</p>');
        put('more', moreEl, more.map(doorHtml).join(''));
        fold.hidden = !more.length;
        if (!keys.length) fold.open = true;
        const added = reveal && [...keysEl.querySelectorAll('[data-entry-id]')].find((node) => node.dataset.entryId === reveal);
        if (added) { reveal = ''; revealNewRow(added, added.querySelector('[data-entry-open]')); }
    }

    function paintOther() {
        if (!settings) return;
        const next = otherKeyRows(settings, skills);
        const keep = new Set(next.map((row) => row.id));
        const before = [...otherEl.children].map((node) => otherRows.get(node.firstElementChild.dataset.settingRow));
        const focused = before.findIndex(({ node }) => node.contains(container.ownerDocument.activeElement));
        for (const [id, mounted] of otherRows) {
            if (keep.has(id)) continue;
            mounted.handle.dispose();
            mounted.secrets.dispose();
            mounted.node.remove();
            otherRows.delete(id);
        }
        next.forEach((row, index) => {
            let mounted = otherRows.get(row.id);
            if (!mounted) {
                const node = otherEl.ownerDocument.createElement('div');
                node.className = 'settings-rows';
                node.innerHTML = row.control.kind === 'secret' ? secretRowHtml(row) : settingsRow(row);
                otherEl.insertBefore(node, otherEl.children[index] || null);
                mounted = { node, row, handle: bindSettingRows(node, { rows: [row], ask: ctx.ask }),
                    secrets: bindSecretRows(node, { rowOf: () => row, ask: ctx.ask }) };
                otherRows.set(row.id, mounted);
            } else {
                Object.assign(mounted.row, row);
                if (otherEl.children[index] !== mounted.node) otherEl.insertBefore(mounted.node, otherEl.children[index] || null);
                mounted.node.querySelector('[data-setting-meta]').textContent = row.meta || '';
                mounted.handle.paint();
            }
            mounted.secrets.paint(settings);
        });
        if (focused >= 0 && !keep.has(before[focused].row.id)) {
            const neighbour = [...before.slice(focused + 1), ...before.slice(0, focused).reverse()]
                .find(({ row }) => keep.has(row.id));
            const control = neighbour?.node.querySelector('[data-secret-open]:not([hidden]), [data-setting-control]')
                || find(`[data-open-path="${CUSTOM_KEY_PATH}"]`);
            revealNewRow(neighbour?.node, control);
        }
    }
    const paint = () => { paintProviders(); paintKeys(); paintOther(); };

    async function read({ withSkills = false } = {}) {
        // Skills enrich the page independently: their read cannot hold up a key or erase a field being edited.
        if (withSkills) void readSkills().then((extensions) => {
            if (gone || !Array.isArray(extensions?.skills)) return;
            skills = extensions.skills;
            paintOther();
        }).catch(() => {});
        const document = await Promise.resolve().then(() => ctx.readSettings()).catch(() => null);
        if (gone) return;
        settings = document || settings;
        paint();
    }

    async function switchKey(input) {
        const head = input.closest('[data-entry-id]');
        const [kind, name = ''] = head.dataset.entryId.split('@');
        busy.add(head.dataset.entryId);
        Object.assign(input, { disabled: true }).setAttribute('aria-busy', 'true');
        const outcome = await setKeyEnabled({ kind, name }, input.checked, { ask: ctx.ask });
        busy.delete(head.dataset.entryId);
        if (gone) return;
        const keys = apiConnections(settings || {});
        setResult(find('[data-entry-result]'), landed(outcome) ? referencesLine(outcome, keys) : editorLine(outcome, 'Saved', keys));
        drawn.keys = '';   // the switch shows what is stored again, whatever the write answered
        await read();
    }

    const onClick = (event) => {
        const hit = (selector) => event.target?.closest?.(selector);
        if (hit('[data-accounts-check]')) { void store.refresh(); paintProviders(); return; }
        if (hit('.ui-switch')) return;
        const door = hit('[data-open-path]');
        if (door && container.contains(door)) ctx.openSettings?.(door.dataset.openPath);
    };
    const onChange = (event) => { if (event.target?.matches?.('[data-entry-switch]')) void switchKey(event.target); };
    const onShown = (event) => { if (event.detail?.path === 'accounts') void read({ withSkills: true }); };
    // One of the keys here was written (a row of this page, a key's page): the document is read again, and a stored
    // key whose write landed puts its field away.
    const onSaved = (event) => {
        if (event.detail?.source !== 'settings-row' || event.detail.page !== 'accounts') return;
        otherRows.get(event.detail.row)?.secrets.close(event.detail.row);
        void read();
    };
    const onPage = (event) => { if (event.detail?.page !== 'settings') otherRows.forEach(({ handle }) => handle.leave()); };
    const offStore = store.subscribe((view) => paintProviders(view), { visible: () => false });
    const win = globalThis.window;
    container.addEventListener('click', onClick);
    container.addEventListener('change', onChange);
    win?.addEventListener?.('ouro:settings-page-shown', onShown);
    win?.addEventListener?.('ouro:settings-updated', onSaved);
    win?.addEventListener?.('ouro:page-shown', onPage);
    paintProviders();
    const mine = list = {
        // What a key's page did, said where the key stands (or stood): the head of a key just added takes the caret.
        say(note, { added = '', other = false, warning = '' } = {}) {
            reveal = added;
            setResult(find(other ? '[data-other-result]' : '[data-entry-result]'),
                warning ? { text: `${note} ${warning}`, tone: 'warn' } : { text: note, tone: 'ok' });
        },
    };
    return { unmount() {
        gone = true;
        if (list === mine) list = null;
        offStore();
        otherRows.forEach(({ handle, secrets }) => { handle.dispose(); secrets.dispose(); });
        container.removeEventListener('click', onClick);
        container.removeEventListener('change', onChange);
        win?.removeEventListener?.('ouro:settings-page-shown', onShown);
        win?.removeEventListener?.('ouro:settings-updated', onSaved);
        win?.removeEventListener?.('ouro:page-shown', onPage);
    } };
}

/* ---- A provider's page ------------------------------------------------------------------------------------------- */

let providerMounts = 0;  // one provider's page is mounted at a time: a page let go late leaves its successor alone

function mountProvider(container, { path }) {
    const store = claudexorStatus;
    const harness = decodeURIComponent(String(path).split('/')[1] || '');
    container.innerHTML = `<p class="settings-page-lead" data-provider-lead></p>
        <div class="accounts-service">${renderAgentsServiceBanner()}
            <div class="settings-toolbar"><button type="button" class="btn btn-default btn-sm" id="btn-harness-refresh">Refresh</button>
                <button type="button" class="btn btn-default btn-sm" id="btn-claudexor-restart" hidden>${RESTART_LABEL}</button></div></div>
        <section class="ui-card settings-card" data-provider-quota hidden>
            <p class="settings-card-note">Quota left, one bar for each account</p>
            <div class="accounts-bars" aria-hidden="true" data-provider-bars></div>
            <p class="settings-card-note" data-provider-turns></p>
        </section>
        <div id="harness-accounts-error" class="settings-inline-status" data-tone="error" hidden></div>
        <div id="harness-model-marks-note" class="settings-inline-status" data-tone="muted" hidden></div>
        <div id="harness-accounts-groups" class="agent-family-list"></div>
        <div id="harness-login-card"></div>`;
    const find = (selector) => container.querySelector(selector);
    let bars = '';
    const paint = () => {
        const payload = store.snapshot || {};
        const reads = { accountsRead: store.facet(FACET_ACCOUNTS), quotaRead: store.facet(FACET_QUOTA) };
        const rows = store.accountsUnsupported ? [] : accountRows(payload).filter((row) => row.harness === harness);
        const known = reads.accountsRead === READ_OK;
        const lead = !known ? '' : rows.length > 1 ? `${rows.length} accounts take turns, so a spent quota does not stop the work.`
            : rows.length ? 'One account. Add a second one and they take turns.' : 'No account yet.';
        if (find('[data-provider-lead]').textContent !== lead) find('[data-provider-lead]').textContent = lead;
        // The glance is for many accounts, over windows that were really read.
        const glance = known && reads.quotaRead === READ_OK && rows.length > 3 ? quotaBars(rows, payload, reads) : null;
        find('[data-provider-quota]').hidden = !glance;
        if (!glance) return;
        const html = glance.bars.map((bar) => `<i class="accounts-bar" data-state="${bar.state}"><b style="--accounts-bar-left:${bar.state === 'ok' && bar.left !== null ? Math.max(6, Math.round(bar.left * 100)) : 0}%"></b></i>`).join('');
        if (bars !== html) find('[data-provider-bars]').innerHTML = bars = html;
        find('[data-provider-turns]').textContent = glance.words;
    };
    const mine = providerMounts += 1;
    void initHarnessAccounts({ family: harness, path, mark: (facts) => accountMarkHtml(facts, store.snapshot || {}) });
    const offStore = store.subscribe(paint, { visible: () => false });
    paint();
    return { unmount() {
        offStore();
        if (mine === providerMounts) destroyHarnessAccounts();
    } };
}

/* ---- The family ---------------------------------------------------------------------------------------------------- */

/** A key's page tells the list what it did and returns there. */
const backToList = (ctx, other = false) => (note, added = '', warning = '') => {
    ctx.openSettings?.('accounts');
    list?.say(note, { added, other, warning });
};

function mount(container, { path, row, ctx = {} }) {
    if (path === 'accounts') return mountList(container, { ctx, row });
    if (path === CUSTOM_KEY_PATH) return mountCustomKeyPage(container, { ctx, left: backToList(ctx, true) });
    if (path.startsWith('accounts/key/') || path === 'accounts/key') {
        if (parseKeyPath(path)) {
            const handle = mountKeyPage(container, { path, ctx, left: backToList(ctx) });
            return handle.ready.then(() => handle);  // the shell titles the page once the read names its key
        }
        container.innerHTML = '<p class="settings-page-lead">No provider has this name.</p>';
        return null;
    }
    return mountProvider(container, { path, ctx });
}

function title(path) {
    if (path === 'accounts') return 'Accounts and keys';
    if (path === CUSTOM_KEY_PATH) return 'New key';
    if (path.startsWith('accounts/key')) return keyTitle(path);
    return familyLabel(decodeURIComponent(path.split('/')[1] || ''), claudexorStatus.snapshot, { catalogKnown: claudexorStatus.catalogKnown });
}

registerSettingsPage('accounts', { title, mount });
