// Settings → Appearance → Language: the one control for the install's interface language.
//
// The choice is an install-wide SETTING (OUROBOROS_UI_LANGUAGE), unlike the device's theme
// beside it: it writes at once through its own endpoint, POST /api/ui/i18n/language, never
// through the generic settings save (which names it in `ignored_keys`). The select lists English, the device's
// language as a suggestion, every language that already has a memory on disk, and
// "Other…", which opens a free field: a code (`pt-BR`), a name, or a description of a
// language to invent. A tag saves at once; a name or a description is resolved by the
// light model into a tag and a profile (the gateway answers `language_needs_model` when
// no model is configured, and the note says so).
import { apiClient } from './api_client.js';
import { settingsSection } from './settings_controls.js';
import { showToast } from './toast.js';
import { applyPayload, currentLanguage, currentPayload, englishTag, fmt, isEnglish, localeDirection, pendingBootRead, pluralSelectMap, setLanguage, tr } from './i18n.js';

export const OTHER_VALUE = '__other__';
export const ENGLISH_VALUE = 'en';

/**
 * The Language control's markup, bound by `bindLanguageSettings`. In Settings (Appearance) it is rows: the language
 * with its status after the row's one line, the free field "Other…" opens, and the page's fold with the rare actions
 * on the translations themselves. `onboarding` is the block at the end of setup: its own lead, no Export and no
 * Regenerate (they act on a language the install does not have yet).
 */
export function languageBlockHtml({ onboarding = false } = {}) {
    if (!onboarding) {
        return `<div class="settings-stack" data-i18n-settings>
            <section class="ui-card settings-card" aria-labelledby="settings-language-title">
                <h3 class="settings-card-title" id="settings-language-title">Everywhere</h3>
                <div class="settings-rows">
                    <div class="ui-row settings-row" data-setting-row="OUROBOROS_UI_LANGUAGE" data-kind="custom">
                        <label class="ui-row-title" for="ui-language-select">Language</label>
                        <div class="ui-row-meta"><span id="ui-language-help">Menus and system messages, on every device. Ouroboros answers in the language you write.</span>
                            <span class="settings-row-fact" id="ui-language-status" data-i18n-status data-i18n-skip role="status" aria-live="polite"></span></div>
                        <div class="ui-row-actions"><select id="ui-language-select" class="ui-control" data-i18n-select data-i18n-skip aria-describedby="ui-language-help ui-language-status"></select></div>
                    </div>
                    <div class="ui-row settings-row" data-kind="custom" data-i18n-other hidden>
                        <label class="ui-row-title" for="ui-language-other">Other language</label>
                        <div class="ui-row-meta"><span id="ui-language-other-help">A code such as pt-BR, a name, or a description of a language to invent.</span>
                            <span class="ui-result settings-row-result" id="ui-language-note" data-i18n-note data-i18n-skip data-tone="warn" role="status" aria-live="polite"></span></div>
                        <div class="ui-row-actions">
                            <input id="ui-language-other" class="ui-control" type="text" data-i18n-other-input autocomplete="off" placeholder="Deutsch, pt-BR, Quenya" aria-describedby="ui-language-other-help ui-language-note">
                            <button type="button" class="btn btn-default btn-sm" data-i18n-other-apply>Use this language</button>
                        </div>
                    </div>
                </div>
            </section>
            <section class="ui-card settings-card">
                <details class="ui-disclosure" data-settings-fold>
                    <summary>More: translations</summary>
                    <div class="settings-rows">
                        <div class="ui-row settings-row" data-setting-row="appearance:translations" data-kind="custom">
                            <p class="ui-row-title">Translations</p>
                            <div class="ui-row-meta"><span>The light model writes them and they are kept with this installation. A label without one stays English.</span></div>
                            <div class="ui-row-actions">
                                <button type="button" class="btn btn-default btn-sm" data-i18n-import>Import JSON…</button>
                                <input type="file" accept="application/json,.json" data-i18n-import-file hidden>
                                <button type="button" class="btn btn-default btn-sm" data-i18n-export>Export</button>
                                <button type="button" class="btn btn-default btn-sm" data-i18n-regenerate>Regenerate</button>
                            </div>
                        </div>
                    </div>
                </details>
            </section>
        </div>`;
    }
    return settingsSection({
        title: 'Language',
        attrs: 'data-i18n-settings',
        card: false,  // setup's step is the card already
        lead: `Ouroboros can speak to you in your language. Pick one now — it is applied when setup finishes,
           for the whole installation — or later in Settings → Appearance. A code such as <code>pt-BR</code>
           needs nothing else; a language name or a description of a language to invent is worked out by
           the model you just set up.`,
        body: `<div class="settings-effort-card">
                <div class="form-field ui-field">
                    <label for="ui-language-select">Interface language</label>
                    <select id="ui-language-select" class="ui-control" data-i18n-select data-i18n-skip aria-describedby="ui-language-help"></select>
                    <div class="settings-inline-note ui-field-help" id="ui-language-help">Pick a language, or choose “Other…” and type a code such as <code>pt-BR</code>, a name, or describe a language to invent. Any language works.</div>
                </div>
                <div class="form-field ui-field" data-i18n-other hidden>
                    <label for="ui-language-other">Other language</label>
                    <div class="settings-toolbar">
                        <input id="ui-language-other" class="ui-control" type="text" data-i18n-other-input autocomplete="off" placeholder="Deutsch, pt-BR, Quenya, or a language to invent">
                        <button type="button" class="btn btn-default btn-sm" data-i18n-other-apply>Use this language</button>
                    </div>
                    <div class="settings-inline-note" data-i18n-note data-i18n-skip role="status" aria-live="polite"></div>
                </div>
                <div class="settings-inline-note" data-i18n-status data-i18n-skip role="status" aria-live="polite"></div>
                <div class="settings-toolbar">
                    <button type="button" class="btn btn-default btn-sm" data-i18n-import>Import JSON…</button>
                    <input type="file" accept="application/json,.json" data-i18n-import-file hidden>
                </div>
            </div>`,
    });
}

function displayName(tag, fallback = tag) {
    if (typeof Intl === 'undefined' || typeof Intl.DisplayNames !== 'function') return fallback;
    try {
        const name = new Intl.DisplayNames([tag, 'en'], { type: 'language' }).of(tag);
        return name && name !== tag ? name : fallback;
    } catch {
        return fallback;
    }
}

function validTag(value) {
    const text = String(value || '').trim();
    return /^[A-Za-z]{2,3}(?:-[A-Za-z0-9]{1,8}){0,8}$/.test(text) ? text : '';
}

function readiness(summary) {
    if (!summary) return '';
    if (summary.malformed) return tr('language.readiness.unreadable', 'file unreadable');
    const parts = [];
    if (summary.entries) parts.push(fmt('{n} ready', { n: summary.entries }));
    if (summary.pending) parts.push(fmt('{n} pending', { n: summary.pending }));
    return parts.join(', ');
}

/**
 * The option rows of the select, pure: English first, then the device's suggestion, then
 * every language on disk (with its readiness), the current tag when it is none of those,
 * and Other… last. Returns [{value, label}].
 */
export function languageOptions({ payload = null, navigatorLanguages = [], current = '' } = {}) {
    const rows = [{ value: ENGLISH_VALUE, label: tr('language.option.english_source', 'English (source)') }];
    const seen = new Set([ENGLISH_VALUE]);
    const onDisk = new Map();
    for (const item of payload?.languages || []) {
        const tag = validTag(item?.language);
        if (tag && !englishTag(tag)) onDisk.set(tag, item);
    }
    const device = (navigatorLanguages || []).map(validTag).find((tag) => tag && !englishTag(tag) && !onDisk.has(tag));
    if (device) {
        seen.add(device);
        rows.push({ value: device, label: fmt("{name} · this device's language", { name: displayName(device, device) }) });
    }
    for (const [tag, item] of onDisk) {
        seen.add(tag);
        const base = item?.label && item.label !== tag ? item.label : displayName(tag, tag);
        const ready = readiness(item);
        rows.push({ value: tag, label: ready ? `${base} · ${ready}` : base });
    }
    const chosen = validTag(current);
    if (chosen && !seen.has(chosen) && !englishTag(chosen)) {
        seen.add(chosen);
        rows.push({ value: chosen, label: displayName(chosen, chosen) });
    }
    rows.push({ value: OTHER_VALUE, label: tr('language.option.other', 'Other…') });
    return rows;
}

/** The status line under the control, pure. */
export function describeStatus(payload) {
    if (!payload) return '';
    if (payload.english) return payload.chosen ? 'English, the source text.' : 'English, the source text; no language chosen yet.';
    const label = payload.profile?.label && payload.profile.label !== payload.language
        ? payload.profile.label : displayName(payload.language, payload.language);
    const stats = payload.stats || {};
    const parts = [fmt('{n} translated', { n: stats.entries || 0 })];
    if (stats.pending) parts.push(fmt('{n} pending', { n: stats.pending }));
    if (typeof stats.stale === 'number' && stats.stale) parts.push(fmt('{n} stale', { n: stats.stale }));
    if (stats.refused) parts.push(fmt('{n} refused (no valid translation; Regenerate retries them)', { n: stats.refused }));
    if (payload.memory_error) parts.push(tr('language.status.unreadable', 'stored file unreadable, English shown'));
    const generator = payload.generator || {};
    if (generator.state === 'running') parts.push(tr('language.status.translating', 'translating…'));
    else if (generator.state === 'no_model') parts.push(tr('language.status.no_model', 'no model configured to translate; set one up in Models'));
    else if (generator.state === 'failed') parts.push(fmt('translation paused: {error}', { error: generator.error || tr('language.status.call_failed', 'the model call failed') }));
    return `${label}: ${parts.join(' · ')}`;
}

/**
 * Save a language choice through the one writer: a tag goes with this engine's plural rules,
 * display name and text direction; a name or a description is resolved by the gateway and then
 * completed with the plural map when the engine knows the resolved tag. The page is painted from
 * the payload the gateway answered, which is returned. Shared by the Settings control and the
 * first-run wizard's completion.
 */
export async function saveLanguageChoice(value, client = apiClient) {
    const body = { language: value };
    const tag = validTag(value);
    if (tag) {
        const plural = pluralSelectMap(tag);
        if (plural) {
            body.plural_select = { map: plural.map, period: plural.period };
            body.plural_categories = plural.categories;
        }
        const label = displayName(tag, '');
        if (label) body.label = label;
        // Only an engine that knows the tag's script has an opinion on its direction: for an
        // invented language Intl guesses `ltr` and would overwrite what the model decided.
        const direction = localeDirection(tag);
        if (direction) body.profile = { direction };
    }
    let payload = await client.saveUiLanguage(body);
    // A name the gateway resolved to a tag has no plural map yet (the browser could not compute
    // one for free text): complete the header with this engine's rules.
    if (!tag && payload && !payload.english && !payload.plural_select) {
        const plural = pluralSelectMap(payload.language);
        if (plural) {
            payload = await client.saveUiLanguage({
                language: payload.language,
                plural_select: { map: plural.map, period: plural.period },
                plural_categories: plural.categories,
            });
        }
    }
    applyPayload(payload);
    return payload;
}

/**
 * Wire the `[data-i18n-settings]` block. Returns a disposer. `client`, `toast` and
 * `navigatorLanguages` are injectable for tests. With `stage`, a choice is not saved but handed
 * to that callback (the first-run wizard applies it after its completion transaction); `staged`
 * is the draft the caller already holds, shown again when the block is re-mounted.
 */
export function bindLanguageSettings(page, {
    client = apiClient, toast = showToast, stage = null, staged: initialStaged = '',
    navigatorLanguages = (typeof navigator !== 'undefined' && navigator.languages) || [],
    openUrl = (url) => { if (typeof window !== 'undefined') window.open(url, '_blank', 'noopener'); },
} = {}) {
    const root = page?.querySelector?.('[data-i18n-settings]');
    if (!root) return () => {};
    const select = root.querySelector('[data-i18n-select]');
    const other = root.querySelector('[data-i18n-other]');
    const otherInput = root.querySelector('[data-i18n-other-input]');
    const otherApply = root.querySelector('[data-i18n-other-apply]');
    const status = root.querySelector('[data-i18n-status]');
    const note = root.querySelector('[data-i18n-note]');
    const importButton = root.querySelector('[data-i18n-import]');
    const importFile = root.querySelector('[data-i18n-import-file]');
    const exportButton = root.querySelector('[data-i18n-export]');
    const regenerateButton = root.querySelector('[data-i18n-regenerate]');
    let payload = null;
    let busy = false;
    let staged = typeof stage === 'function' ? String(initialStaged || '').trim() : '';   // the wizard's retained draft
    for (const el of [status, note]) if (el && el.dataset) el.dataset.i18nSkip = '';

    const setNote = (text) => { if (note) note.textContent = text || ''; };
    const setStatus = (text) => { if (status) status.textContent = text || ''; };

    function render() {
        if (!select) return;
        const current = payload ? (payload.english ? ENGLISH_VALUE : payload.language) : (isEnglish() ? ENGLISH_VALUE : currentLanguage());
        const rows = languageOptions({ payload, navigatorLanguages, current });
        select.replaceChildren(...rows.map((row) => {
            const option = document.createElement('option');
            option.value = row.value;
            option.textContent = row.label;
            return option;
        }));
        select.value = rows.some((row) => row.value === current) ? current : ENGLISH_VALUE;
        if (staged) {
            select.value = rows.some((row) => row.value === staged) ? staged : OTHER_VALUE;
            if (select.value === OTHER_VALUE && otherInput) otherInput.value = staged;
        }
        if (other) other.hidden = select.value !== OTHER_VALUE;
        setStatus(staged ? fmt('Will be applied when setup finishes: {language}', { language: staged }) : describeStatus(payload));
        for (const button of [exportButton, regenerateButton]) {
            if (button) button.disabled = !payload || payload.english;
        }
    }

    async function load() {
        // The SPA boot (app.js) applies the gateway payload; this binder only paints from it,
        // reading the gateway itself when it mounts before that boot read has answered (or on
        // a page without that boot, such as the first-run wizard). What it read is applied for
        // the page unless a newer read landed meanwhile, so a re-render does not read again.
        try {
            payload = currentPayload();
            if (!payload && pendingBootRead()) {
                await pendingBootRead();   // the SPA's own boot read is in flight: one read per page
                payload = currentPayload();
            }
            if (!payload) {
                payload = await client.uiI18n();
                if (!currentPayload()) applyPayload(payload);
                else payload = currentPayload();
            }
            render();
        } catch {
            setStatus('Language settings could not be loaded.');
        }
    }

    // After a write whose answer is not a full payload (an import of another language's pack, a
    // regenerate): read the memory once and apply it, so the select's language list and the overlay
    // follow — also on a page without the socket frames, such as the first-run wizard.
    async function reload() {
        try {
            const fresh = await client.uiI18n();
            const held = currentPayload();
            // A read that answers after a newer revision of the same language landed is not applied over it.
            const older = held && fresh && held.language === fresh.language && Number(fresh.revision || 0) < Number(held.revision || 0);
            if (!older) applyPayload(fresh);
            payload = currentPayload() || fresh;
            render();
        } catch {
            setStatus('Language settings could not be loaded.');
        }
    }

    async function choose(raw) {
        const value = String(raw || '').trim();
        if (!value || busy) return;
        if (typeof stage === 'function') {
            // First run: the choice is staged and applied through the same writer once the setup
            // transaction has written settings.json — a write now would create that file ahead of
            // completion and disqualify the install-time defaults and presets.
            staged = value;
            stage(value);
            setNote('');
            render();
            return;
        }
        busy = true;
        setNote('');
        try {
            payload = await saveLanguageChoice(value, client);
            render();
            if (otherInput) otherInput.value = '';
        } catch (error) {
            const code = error?.body?.code || error?.payload?.code || error?.code || '';
            if (code === 'language_not_a_tag' || code === 'language_needs_model' || code === 'language_resolve_failed') {
                setNote(code === 'language_needs_model'
                    ? tr('language.note.needs_model', 'A language name needs a model: set one up in Models, or type a code such as pt-BR.')
                    : code === 'language_resolve_failed'
                        ? tr('language.note.resolve_failed', 'Ouroboros could not work out this language right now. Try again, or type a code such as pt-BR.')
                        : tr('language.note.not_a_language', 'Not recognized as a language. Type a code such as pt-BR, a language name, or describe a language to invent.'));
                if (select) select.value = OTHER_VALUE;
                if (other) other.hidden = false;
            } else {
                toast('Language choice could not be saved.', 'error');
                render();
            }
        } finally {
            busy = false;
        }
    }

    const onSelect = () => {
        if (!select) return;
        if (select.value === OTHER_VALUE) {
            if (other) other.hidden = false;
            otherInput?.focus?.();
            return;
        }
        void choose(select.value);
    };
    const onOtherApply = () => void choose(otherInput?.value);
    const onOtherKey = (event) => { if (event.key === 'Enter') { event.preventDefault(); onOtherApply(); } };
    const onImportClick = () => importFile?.click?.();
    const onImportFile = async () => {
        const file = importFile?.files?.[0];
        if (!file) return;
        try {
            const text = await file.text();
            const doc = JSON.parse(text);
            const result = await client.importI18n(doc);
            toast(`Imported ${result?.result?.added ?? 0} new and replaced ${result?.result?.replaced ?? 0} translations.`, 'success');
            await reload();
        } catch (error) {
            toast(error?.body?.error || error?.message || 'The file is not a valid translation memory.', 'error');
        } finally {
            if (importFile) importFile.value = '';
        }
    };
    const onExport = () => { if (payload && !payload.english) openUrl(client.exportI18nUrl(payload.language)); };
    const onRegenerate = async () => {
        if (!payload || payload.english || busy) return;
        busy = true;
        try {
            await client.regenerateI18n({ language: payload.language });
            toast('Generated translations dropped; your own and imported ones stay. Rebuilding.', 'success');
            await reload();
        } catch {
            toast('Regeneration could not be started.', 'error');
        } finally {
            busy = false;
        }
    };
    // Painted from the payload that was just applied — never a second read: the read → apply →
    // event → read cycle is exactly the loop this guards against.
    const onLanguageChanged = () => { payload = currentPayload() || payload; render(); };

    select?.addEventListener('change', onSelect);
    otherApply?.addEventListener('click', onOtherApply);
    otherInput?.addEventListener('keydown', onOtherKey);
    importButton?.addEventListener('click', onImportClick);
    importFile?.addEventListener('change', onImportFile);
    exportButton?.addEventListener('click', onExport);
    regenerateButton?.addEventListener('click', onRegenerate);
    if (typeof window !== 'undefined') window.addEventListener('ouro:language-changed', onLanguageChanged);
    void load();
    return () => {
        select?.removeEventListener('change', onSelect);
        otherApply?.removeEventListener('click', onOtherApply);
        otherInput?.removeEventListener('keydown', onOtherKey);
        importButton?.removeEventListener('click', onImportClick);
        importFile?.removeEventListener('change', onImportFile);
        exportButton?.removeEventListener('click', onExport);
        regenerateButton?.removeEventListener('click', onRegenerate);
        if (typeof window !== 'undefined') window.removeEventListener('ouro:language-changed', onLanguageChanged);
    };
}

// `setLanguage` is re-exported so a caller that already has a payload (the POST body) can
// paint without a second read.
export { setLanguage };
