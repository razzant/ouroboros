// Settings → Appearance → Language: the option rows, the status line, and the binder's
// flows (choose a listed language, type an unusual one, import, export, regenerate), with
// the gateway client, the toast and the device languages injected. The control writes the
// install-wide setting at once and never through the Settings draft.
import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { bindLanguageSettings, describeStatus, languageBlockHtml, languageOptions, ENGLISH_VALUE, OTHER_VALUE } from '../modules/settings_language.js';

const settle = () => new Promise((resolve) => setTimeout(resolve, 0));

test('the option rows: English, the device suggestion, every memory on disk, the odd current tag, Other…', () => {
    const rows = languageOptions({
        payload: { languages: [{ language: 'ru', entries: 1200, pending: 3, label: 'Русский' }, { language: 'en' }] },
        navigatorLanguages: ['de-DE', 'en-US'],
        current: 'qya',
    });
    assert.deepEqual(rows.map((row) => row.value), [ENGLISH_VALUE, 'de-DE', 'ru', 'qya', OTHER_VALUE]);
    assert.equal(rows[0].label, 'English (source)');
    assert.ok(rows[1].label.endsWith("· this device's language"), rows[1].label);
    assert.equal(rows[2].label, 'Русский · 1200 ready, 3 pending');
    assert.equal(rows[4].label, 'Other…');
    // A device language that already has a memory is listed once, as the memory.
    const once = languageOptions({ payload: { languages: [{ language: 'de' }] }, navigatorLanguages: ['de'] });
    assert.deepEqual(once.map((row) => row.value), [ENGLISH_VALUE, 'de', OTHER_VALUE]);
    // Garbage in the payload never becomes an option.
    const clean = languageOptions({ payload: { languages: [{ language: 'Russian please' }, {}] } });
    assert.deepEqual(clean.map((row) => row.value), [ENGLISH_VALUE, OTHER_VALUE]);
});

test('the status line states the source, or the language with its counts and a damaged file', () => {
    assert.equal(describeStatus(null), '');
    assert.equal(describeStatus({ english: true, chosen: false }), 'English, the source text; no language chosen yet.');
    assert.equal(describeStatus({ english: true, chosen: true }), 'English, the source text.');
    assert.equal(describeStatus({ english: false, language: 'ru', profile: { label: 'Русский' }, stats: { entries: 1200, pending: 3, stale: 2 } }),
        'Русский: 1200 translated · 3 pending · 2 stale');
    assert.equal(describeStatus({ english: false, language: 'qya', stats: { entries: 0 }, memory_error: 'bad json' }),
        'qya: 0 translated · stored file unreadable, English shown');
});

// ---------------------------------------------------------------------------
// A flat DOM double: the binder reads [data-i18n-*] nodes, listens to a few events,
// fills the <select> through document.createElement, and toggles hidden/disabled.
// ---------------------------------------------------------------------------

class Stub {
    constructor(tag, attrs = {}) {
        this.tagName = tag.toUpperCase();
        this.attrs = { ...attrs };
        this.children = [];
        this.listeners = new Map();
        this.hidden = false;
        this.disabled = false;
        this.value = '';
        this.textContent = '';
        this.files = [];
        this.focused = 0;
        this.clicked = 0;
        this.dataset = {};
    }

    matches(selector) {
        const attr = /^\[([^\]=]+)\]$/.exec(selector.trim());
        if (attr) return attr[1] in this.attrs;
        if (selector.startsWith('#')) return this.attrs.id === selector.slice(1);
        return this.tagName === selector.toUpperCase();
    }

    querySelector(selector) {
        for (const child of this.children) {
            if (child.matches(selector)) return child;
            const deep = child.querySelector(selector);
            if (deep) return deep;
        }
        return null;
    }

    addEventListener(type, fn) { this.listeners.set(type, fn); }

    removeEventListener(type) { this.listeners.delete(type); }

    fire(type, extra = {}) {
        const fn = this.listeners.get(type);
        return fn ? fn({ type, target: this, preventDefault() { this.prevented = true; }, ...extra }) : undefined;
    }

    replaceChildren(...nodes) { this.children = nodes; }

    focus() { this.focused += 1; }

    click() { this.clicked += 1; }
}

function page() {
    const root = new Stub('div', { 'data-i18n-settings': '' });
    const parts = {
        select: new Stub('select', { 'data-i18n-select': '' }),
        other: new Stub('div', { 'data-i18n-other': '' }),
        otherInput: new Stub('input', { 'data-i18n-other-input': '' }),
        otherApply: new Stub('button', { 'data-i18n-other-apply': '' }),
        note: new Stub('div', { 'data-i18n-note': '' }),
        status: new Stub('div', { 'data-i18n-status': '' }),
        importButton: new Stub('button', { 'data-i18n-import': '' }),
        importFile: new Stub('input', { 'data-i18n-import-file': '' }),
        exportButton: new Stub('button', { 'data-i18n-export': '' }),
        regenerateButton: new Stub('button', { 'data-i18n-regenerate': '' }),
    };
    parts.other.hidden = true;
    parts.other.children = [parts.otherInput, parts.otherApply, parts.note];
    root.children = [parts.select, parts.other, parts.status, parts.importButton, parts.importFile, parts.exportButton, parts.regenerateButton];
    const doc = new Stub('div');
    doc.children = [root];
    return { doc, ...parts };
}

function fakeClient(initial) {
    const calls = [];
    let current = initial;
    return {
        calls,
        set(payload) { current = payload; },
        uiI18n: async () => { calls.push(['uiI18n']); return current; },
        saveUiLanguage: async (body) => {
            calls.push(['saveUiLanguage', body]);
            if (body.language === 'Quenya') throw Object.assign(new Error('bad'), { status: 400, body: { code: 'language_needs_model' } });
            if (body.language === 'invent a language') {
                current = { language: 'art-x-vael', english: false, chosen: true, entries: {}, revision: 1, plural_select: null,
                    profile: { label: 'Vaelic' }, stats: { entries: 0, pending: 0 }, languages: current.languages };
                return current;
            }
            if (body.language === 'Russian please') {
                current = { language: 'ru', english: false, chosen: true, entries: {}, revision: 1, plural_select: null,
                    profile: { label: 'Русский' }, stats: { entries: 0, pending: 0 }, languages: current.languages };
                return current;
            }
            if (body.language === 'ru' && body.plural_select && current.language === 'ru') {  // the completing second save
                current = { ...current, plural_select: body.plural_select, plural_categories: body.plural_categories || null };
                return current;
            }
            current = { language: body.language, english: false, chosen: true, entries: {}, revision: 1,
                profile: { label: body.label || body.language }, stats: { entries: 0, pending: 0 }, languages: current.languages };
            return current;
        },
        importI18n: async (doc) => {
            calls.push(['importI18n', doc]);
            // The gateway answers counts only; the memory on disk now has this language too.
            if (doc && doc.language && !(current.languages || []).some((row) => row.language === doc.language)) {
                current = { ...current, languages: [...(current.languages || []), { language: doc.language, label: doc.label || doc.language, entries: Object.keys(doc.entries || {}).length }] };
            }
            return { result: { added: 2, replaced: 1 } };
        },
        exportI18nUrl: (language) => `/api/ui/i18n/export?language=${language}`,
        regenerateI18n: async (body) => { calls.push(['regenerateI18n', body]); return { ok: true }; },
    };
}

function withDocument(fn, { body = null, window = null } = {}) {
    const keys = ['document', 'window', 'localStorage'];
    const saved = keys.map((key) => [key, key in globalThis, globalThis[key]]);
    globalThis.document = { createElement: (tag) => new Stub(tag), body, documentElement: { lang: '', dir: '', removeAttribute() {} } };
    if (window) globalThis.window = window;
    globalThis.localStorage = { getItem: () => null, setItem() {} };
    // Module state is process-wide: every test starts from "no payload applied yet".
    const reset = import('../modules/i18n.js').then(({ applyPayload, markBootRead }) => { applyPayload(null); markBootRead(null); });
    return reset.then(fn).finally(() => {
        for (const [key, existed, value] of saved) {
            if (existed) globalThis[key] = value;
            else delete globalThis[key];
        }
    });
}

/** A real synchronous event target: dispatch runs the listeners before it returns, as a browser does. */
function syncWindow() {
    const listeners = new Map();
    return {
        addEventListener(type, fn) { listeners.set(type, [...(listeners.get(type) || []), fn]); },
        removeEventListener(type, fn) { listeners.set(type, (listeners.get(type) || []).filter((f) => f !== fn)); },
        dispatchEvent(event) { for (const fn of listeners.get(event.type) || []) fn(event); return true; },
        count(type) { return (listeners.get(type) || []).length; },
    };
}

const ENGLISH = { language: '', english: true, chosen: false, entries: {}, revision: 0, stats: null,
    languages: [{ language: 'ru', entries: 1200, pending: 3, label: 'Русский' }] };

test('boot lists the languages and shows the English source; export and regenerate wait for a language', () => withDocument(async () => {
    const p = page();
    const client = fakeClient(ENGLISH);
    const toasts = [];
    const dispose = bindLanguageSettings(p.doc, { client, toast: (m, k) => toasts.push([m, k]), navigatorLanguages: ['de'] });
    await settle();
    assert.deepEqual(p.select.children.map((o) => o.value), [ENGLISH_VALUE, 'de', 'ru', OTHER_VALUE]);
    assert.equal(p.select.value, ENGLISH_VALUE);
    assert.equal(p.other.hidden, true);
    assert.equal(p.status.textContent, 'English, the source text; no language chosen yet.');
    assert.equal(p.exportButton.disabled, true);
    assert.equal(p.regenerateButton.disabled, true);
    assert.deepEqual(toasts, []);
    dispose();
    assert.equal(p.select.listeners.size, 0);
    assert.equal(p.importFile.listeners.size, 0);
}));

test('choosing a listed language saves the tag with the plural map and a display label, then paints the status', () => withDocument(async () => {
    const p = page();
    const client = fakeClient(ENGLISH);
    bindLanguageSettings(p.doc, { client, toast: () => {}, navigatorLanguages: [] });
    await settle();
    p.select.value = 'ru';
    p.select.fire('change');
    await settle();
    const save = client.calls.find(([name]) => name === 'saveUiLanguage');
    assert.ok(save, 'the choice writes through the language endpoint at once');
    const body = save[1];
    assert.equal(body.language, 'ru');
    assert.equal(body.plural_select.map['1'], 'one');
    assert.equal(body.plural_select.map['5'], 'many');
    assert.equal(body.plural_select.period, 100);
    assert.ok(body.plural_categories.includes('few'));
    assert.ok(typeof body.label === 'string' && body.label.length > 0, 'a display name travels with the tag');
    assert.equal(p.select.value, 'ru');
    assert.ok(p.status.textContent.startsWith(`${body.label}: 0 translated`), p.status.textContent);
    assert.equal(p.exportButton.disabled, false);
    assert.equal(p.regenerateButton.disabled, false);
}));

test('Other… opens the free field; a tag saves, a description the gateway cannot resolve yet is explained inline', () => withDocument(async () => {
    const p = page();
    const client = fakeClient(ENGLISH);
    bindLanguageSettings(p.doc, { client, toast: () => {}, navigatorLanguages: [] });
    await settle();
    p.select.value = OTHER_VALUE;
    p.select.fire('change');
    assert.equal(p.other.hidden, false);
    assert.equal(p.otherInput.focused, 1);
    assert.equal(client.calls.filter(([name]) => name === 'saveUiLanguage').length, 0, 'opening the field saves nothing');

    p.otherInput.value = 'Quenya';
    p.otherApply.fire('click');
    await settle();
    assert.equal(client.calls.at(-1)[1].language, 'Quenya');
    assert.ok(p.note.textContent.startsWith('A language name needs a model'), p.note.textContent);
    assert.equal(p.select.value, OTHER_VALUE);
    assert.equal(p.other.hidden, false);

    p.otherInput.value = 'pt-BR';
    const event = { key: 'Enter' };
    p.otherInput.fire('keydown', event);
    await settle();
    const save = client.calls.at(-1);
    assert.equal(save[0], 'saveUiLanguage');
    assert.equal(save[1].language, 'pt-BR');
    assert.equal(p.note.textContent, '');
    assert.equal(p.otherInput.value, '');
    assert.equal(p.select.value, 'pt-BR');
}));

test('import reads the chosen file into the import endpoint and reloads; export and regenerate act on the current language', () => withDocument(async () => {
    const p = page();
    const client = fakeClient({ ...ENGLISH, language: 'ru', english: false, chosen: true, profile: { label: 'Русский' }, stats: { entries: 10, pending: 0 } });
    const toasts = [];
    const opened = [];
    bindLanguageSettings(p.doc, { client, toast: (m, k) => toasts.push([m, k]), navigatorLanguages: [], openUrl: (url) => opened.push(url) });
    await settle();
    assert.equal(p.status.textContent, 'Русский: 10 translated');

    p.importButton.fire('click');
    assert.equal(p.importFile.clicked, 1);
    const readsBefore = client.calls.filter(([name]) => name === 'uiI18n').length;
    p.importFile.files = [{ text: async () => JSON.stringify({ schema: 1, language: 'de', label: 'Deutsch', entries: { Settings: { text: 'Einstellungen' } } }) }];
    p.importFile.fire('change');
    await settle();
    const imported = client.calls.find(([name]) => name === 'importI18n');
    assert.deepEqual(imported[1], { schema: 1, language: 'de', label: 'Deutsch', entries: { Settings: { text: 'Einstellungen' } } });
    assert.deepEqual(toasts.at(-1), ['Imported 2 new and replaced 1 translations.', 'success']);
    assert.equal(p.importFile.value, '');
    assert.equal(client.calls.filter(([name]) => name === 'uiI18n').length, readsBefore + 1, 'an import of another language\'s pack re-reads the memory once');
    assert.ok(p.select.children.some((o) => o.value === 'de'), 'the imported language appears in the select without a reload');

    p.importFile.files = [{ text: async () => 'not json' }];
    p.importFile.fire('change');
    await settle();
    assert.equal(toasts.at(-1)[1], 'error');

    p.exportButton.fire('click');
    assert.deepEqual(opened, ['/api/ui/i18n/export?language=ru']);

    const readsBeforeRegen = client.calls.filter(([name]) => name === 'uiI18n').length;
    p.regenerateButton.fire('click');
    await settle();
    const regen = client.calls.find(([name]) => name === 'regenerateI18n');
    assert.deepEqual(regen[1], { language: 'ru' });
    assert.equal(client.calls.filter(([name]) => name === 'uiI18n').length, readsBeforeRegen + 1, 'regenerate re-reads the memory once');
    assert.equal(toasts.at(-1)[1], 'success');
}));

test('with a real document and window the binder reads once and re-renders on the event without a fetch', () => {
    const win = syncWindow();
    const body = new Stub('body');
    return withDocument(async () => {
        const { applyPayload } = await import('../modules/i18n.js');
        const p = page();
        const client = fakeClient(ENGLISH);
        const dispose = bindLanguageSettings(p.doc, { client, toast: () => {}, navigatorLanguages: [] });
        await settle();
        await settle();
        assert.equal(client.calls.filter(([name]) => name === 'uiI18n').length, 1, 'one boot read, no read loop');
        assert.equal(win.count('ouro:language-changed'), 1);
        // The SPA applies a gateway payload (a ui_i18n_updated frame, a switch from another client):
        // the binder paints from it and never fetches again.
        applyPayload({ ...ENGLISH, language: 'ru', english: false, chosen: true, entries: {}, revision: 3,
            profile: { label: 'Русский' }, stats: { entries: 7, pending: 1 }, generator: { state: 'running' } });
        await settle();
        assert.equal(client.calls.filter(([name]) => name === 'uiI18n').length, 1);
        assert.equal(p.status.textContent, 'Русский: 7 translated · 1 pending · translating…');
        assert.equal(p.select.value, 'ru');
        assert.equal(p.status.dataset.i18nSkip, '', 'the status line is never an overlay key');
        dispose();
        assert.equal(win.count('ouro:language-changed'), 0);
        applyPayload(ENGLISH);
    }, { body, window: win });
});

test('choosing a tag the engine knows the direction of saves it with the profile', () => withDocument(async () => {
    const { localeDirection } = await import('../modules/i18n.js');
    const expected = localeDirection('ar');
    const p = page();
    const client = fakeClient(ENGLISH);
    bindLanguageSettings(p.doc, { client, toast: () => {}, navigatorLanguages: [] });
    await settle();
    p.select.value = OTHER_VALUE;
    p.select.fire('change');
    p.otherInput.value = 'ar';
    p.otherApply.fire('click');
    await settle();
    const body = client.calls.find(([name, b]) => name === 'saveUiLanguage' && b.language === 'ar')[1];
    if (expected) assert.deepEqual(body.profile, { direction: expected });
    else assert.equal(body.profile, undefined, 'an engine without text info sends no direction');
}));

test('a page without the block binds nothing and disposes harmlessly', () => {
    const dispose = bindLanguageSettings(new Stub('div'), { client: {}, toast: () => {} });
    assert.equal(typeof dispose, 'function');
    dispose();
});

test('a described language the gateway resolved to a known tag gets its plural map completed in a second save', () => withDocument(async () => {
    const p = page();
    const client = fakeClient(ENGLISH);
    bindLanguageSettings(p.doc, { client, toast: () => {}, navigatorLanguages: [] });
    await settle();
    p.select.value = OTHER_VALUE;
    p.select.fire('change');
    p.otherInput.value = 'Russian please';
    p.otherApply.fire('click');
    await settle();
    const saves = client.calls.filter(([name]) => name === 'saveUiLanguage').map(([, body]) => body);
    assert.equal(saves.length, 2, 'the description, then the resolved tag with this engine\'s plural rules');
    assert.equal(saves[0].language, 'Russian please');
    assert.equal(saves[0].plural_select, undefined, 'no plural map can be computed for free text');
    assert.equal(saves[1].language, 'ru');
    assert.equal(saves[1].plural_select.map['1'], 'one');
    assert.equal(p.select.value, 'ru');
    assert.ok(p.status.textContent.startsWith('Русский: 0 translated'), p.status.textContent);
}));

test('an invented language gets no plural map: the engine does not know it, so the memory keeps `other`', () => withDocument(async () => {
    const p = page();
    const client = fakeClient(ENGLISH);
    bindLanguageSettings(p.doc, { client, toast: () => {}, navigatorLanguages: [] });
    await settle();
    p.select.value = OTHER_VALUE;
    p.select.fire('change');
    p.otherInput.value = 'invent a language';
    p.otherApply.fire('click');
    await settle();
    const saves = client.calls.filter(([name]) => name === 'saveUiLanguage').map(([, body]) => body);
    assert.equal(saves.length, 1, 'no second save: Intl would silently hand an invented language the default locale\'s grammar');
    assert.equal(saves[0].language, 'invent a language');
    assert.equal(p.select.value, 'art-x-vael');
    assert.ok(p.status.textContent.startsWith('Vaelic: 0 translated'), p.status.textContent);
}));

test('the status line names the generator state when it is not idle', () => {
    const base = { english: false, language: 'ru', profile: { label: 'Русский' }, stats: { entries: 5, pending: 2 } };
    assert.equal(describeStatus({ ...base, generator: { state: 'running' } }), 'Русский: 5 translated · 2 pending · translating…');
    assert.ok(describeStatus({ ...base, generator: { state: 'no_model' } }).endsWith('no model configured to translate; set one up in Models'));
    assert.equal(describeStatus({ ...base, generator: { state: 'failed', error: 'budget exhausted' } }),
        'Русский: 5 translated · 2 pending · translation paused: budget exhausted');
    assert.equal(describeStatus({ ...base, generator: { state: 'idle' } }), 'Русский: 5 translated · 2 pending');
});


test('a control mounted while the SPA boot read is in flight waits for it instead of reading again', () => withDocument(async () => {
    const { applyPayload, markBootRead } = await import('../modules/i18n.js');
    let finish;
    markBootRead(new Promise((resolve) => { finish = resolve; }).then(() => { applyPayload({ ...ENGLISH, language: 'ru', english: false, chosen: true, entries: {}, revision: 2, profile: { label: 'Русский' }, stats: { entries: 3, pending: 0 } }); return null; }));
    const p = page();
    const client = fakeClient(ENGLISH);
    bindLanguageSettings(p.doc, { client, toast: () => {}, navigatorLanguages: [] });
    await settle();
    assert.equal(client.calls.filter(([name]) => name === 'uiI18n').length, 0, 'no second boot read');
    finish();
    await settle();
    assert.equal(client.calls.filter(([name]) => name === 'uiI18n').length, 0);
    assert.equal(p.status.textContent, 'Русский: 3 translated');
    assert.equal(p.select.value, 'ru');
}));

test('with a stage callback the control hands the choice over instead of writing it', () => withDocument(async () => {
    const p = page();
    const client = fakeClient(ENGLISH);
    const staged = [];
    bindLanguageSettings(p.doc, { client, toast: () => {}, navigatorLanguages: [], stage: (value) => staged.push(value) });
    await settle();
    p.select.value = 'ru';
    p.select.fire('change');
    await settle();
    assert.deepEqual(staged, ['ru']);
    assert.equal(client.calls.filter(([name]) => name === 'saveUiLanguage').length, 0, 'nothing is written before setup completes');
    assert.equal(p.status.textContent, 'Will be applied when setup finishes: ru');
    assert.equal(p.select.value, 'ru');
    p.select.value = OTHER_VALUE;
    p.select.fire('change');
    p.otherInput.value = 'Quenya';
    p.otherApply.fire('click');
    await settle();
    assert.deepEqual(staged, ['ru', 'Quenya']);
    assert.equal(p.status.textContent, 'Will be applied when setup finishes: Quenya');
    assert.equal(p.select.value, OTHER_VALUE);
    assert.equal(p.otherInput.value, 'Quenya');
}));

test('the status line names refused strings, and an invented tag sends no engine direction', () => withDocument(async () => {
    const p = page();
    const client = fakeClient({ ...ENGLISH, language: 'ru', english: false, chosen: true, profile: { label: 'Русский' }, stats: { entries: 10, pending: 0, refused: 2 } });
    bindLanguageSettings(p.doc, { client, toast: () => {}, navigatorLanguages: [] });
    await settle();
    assert.equal(p.status.textContent, 'Русский: 10 translated · 2 refused (no valid translation; Regenerate retries them)');
    const { saveLanguageChoice } = await import('../modules/settings_language.js');
    await saveLanguageChoice('art-x-vael', client);
    const body = client.calls.filter(([name]) => name === 'saveUiLanguage').at(-1)[1];
    assert.equal(body.language, 'art-x-vael');
    assert.equal(body.profile, undefined, 'the engine has no opinion on an invented language\'s direction');
    assert.equal(body.plural_select, undefined);
}));


test('a re-mounted staging control shows the draft its caller still holds', () => withDocument(async () => {
    const p = page();
    const client = fakeClient(ENGLISH);
    bindLanguageSettings(p.doc, { client, toast: () => {}, navigatorLanguages: [], stage: () => {}, staged: 'de' });
    await settle();
    assert.equal(p.status.textContent, 'Will be applied when setup finishes: de');
    assert.equal(p.select.value, OTHER_VALUE);
    assert.equal(p.otherInput.value, 'de');
    assert.equal(client.calls.filter(([name]) => name === 'saveUiLanguage').length, 0);
}));

test('a tag whose script the engine knows carries its direction even when the engine has no plural data for it', () => withDocument(async () => {
    const { localeDirection } = await import('../modules/i18n.js');
    const { saveLanguageChoice } = await import('../modules/settings_language.js');
    const client = fakeClient(ENGLISH);
    await saveLanguageChoice('lrc', client);   // Northern Luri: Arabic script, no CLDR plural rules
    const body = client.calls.filter(([name]) => name === 'saveUiLanguage').at(-1)[1];
    const expected = localeDirection('lrc');
    if (expected) assert.deepEqual(body.profile, { direction: expected });
    else assert.equal(body.profile, undefined);
    let scriptKnown = false;
    try { scriptKnown = Boolean(new Intl.Locale('lrc').maximize().script) && Boolean(new Intl.Locale('lrc').getTextInfo?.() || new Intl.Locale('lrc').textInfo); } catch { /* an engine without Intl.Locale */ }
    if (scriptKnown) assert.deepEqual(body.profile, { direction: 'rtl' }, 'the right-to-left script is sent, not withheld for the missing plural data');
}));

test('saving a choice paints the page from the answer; a refused save leaves the page as it was', () => withDocument(async () => {
    const { currentLanguage, currentPayload } = await import('../modules/i18n.js');
    const { saveLanguageChoice } = await import('../modules/settings_language.js');
    const client = fakeClient(ENGLISH);
    const answered = await saveLanguageChoice('de', client);   // the first-run wizard's post-completion step
    assert.equal(currentPayload(), answered);
    assert.equal(currentLanguage(), 'de');
    await assert.rejects(saveLanguageChoice('Quenya', client));
    assert.equal(currentLanguage(), 'de', 'a refused save applies nothing');
    // The wizard re-renders after a save that landed with a failed later step: the control it
    // mounts again paints the language the page now has, without another read.
    const p = page();
    bindLanguageSettings(p.doc, { client, toast: () => {}, navigatorLanguages: [], stage: () => {}, staged: '' });
    await settle();
    assert.equal(p.select.value, 'de');
    assert.notEqual(p.status.textContent, 'English, the source text; no language chosen yet.');
    assert.equal(client.calls.filter(([name]) => name === 'uiI18n').length, 0);
}));

test('the block mounted in the wizard offers Import only and hides what is hidden; Settings keeps Export and Regenerate', () => {
    const settings = languageBlockHtml();
    const wizard = languageBlockHtml({ onboarding: true });
    for (const marker of ['data-i18n-select', 'data-i18n-other hidden', 'data-i18n-other-input', 'data-i18n-status', 'data-i18n-import', 'data-i18n-import-file hidden']) {
        assert.ok(settings.includes(marker) && wizard.includes(marker), marker);
    }
    for (const marker of ['data-i18n-export', 'data-i18n-regenerate']) {
        assert.ok(settings.includes(marker), `${marker} in Settings`);
        assert.ok(!wizard.includes(marker), `${marker} acts on a language a first run does not have yet`);
    }
    // Setup's step is the card: the block in it is no second card (DESIGN principle 2). Settings draws rows in a card,
    // the status in the language row's own line, and the rare actions in the page's fold, named by what it holds.
    assert.match(settings, /<section class="ui-card settings-card"[^>]*>\s*<h3 class="settings-card-title"[^>]*>Everywhere<\/h3>/);
    assert.match(settings, /data-setting-row="OUROBOROS_UI_LANGUAGE"[\s\S]*?<label class="ui-row-title" for="ui-language-select">Language<\/label>/);
    assert.match(settings, /<details class="ui-disclosure" data-settings-fold>\s*<summary>More: translations<\/summary>/);
    assert.ok(settings.indexOf('data-settings-fold') < settings.indexOf('data-i18n-import'), 'Import, Export and Regenerate sit in the fold');
    // The select is the language block's own control: the row helper never takes its change as a second writer.
    assert.doesNotMatch(settings, /data-setting-control/);
    assert.doesNotMatch(wizard, /ui-card|settings-row/);
    // The wizard loads neither style.css (the global [hidden] rule) nor settings.css (the toolbar row):
    // its own stylesheet states both for this block, or "Other language" shows before "Other…" is chosen.
    const css = readFileSync(new URL('../onboarding.css', import.meta.url), 'utf8');
    assert.match(css, /\.wizard-content \[data-i18n-settings\] \[hidden\] \{ display: none; \}/);
    assert.match(css, /\.wizard-content \[data-i18n-settings\] \.settings-toolbar \{ display: flex;/);
});
