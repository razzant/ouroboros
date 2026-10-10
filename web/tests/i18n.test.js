// The interface language in the browser: string lookup with plural forms, the catalog seam
// (`tr`/`fmt`), the miss reporter, the DOM overlay (text, attributes, inline composites,
// scoped words, restore) and the serialized language switch. The overlay touches only the
// handful of DOM methods stubbed below, so the house node-stub idiom covers it without a
// browser. The memory arrives as a gateway payload here: no dictionary lives in the repo.
import test from 'node:test';
import assert from 'node:assert/strict';
import { readdirSync, readFileSync } from 'node:fs';
import * as acorn from './vendor/acorn.mjs';
import {
    applyPayload, createTranslator, currentPayload, englishTag, entryText, flushMisses, fmt, fmtInto,
    localeDirection, lookupString, missKeyFor, pendingMisses, pluralSelectMap, setLanguage, setMissTransport, tr, translateString, tx,
    CODE_PREFIX, EXCLUDE_SELECTOR, SCOPE_SEPARATOR, SKIP_ROOTS, USER_CONTENT,
} from '../modules/i18n.js';

// A Russian memory the way GET /api/ui/i18n answers it: exact chrome strings, a DOM-scoped
// word, number templates with plural forms, and catalog sentences by code.
const RU_ENTRIES = {
    'Settings': { text: 'Настройки' },
    'Main Chat': { text: 'Основной чат' },
    'Search': { text: 'Поиск' },
    'System': { text: 'Системная' },
    'Light': { text: 'Лёгкий' },
    ['Light' + SCOPE_SEPARATOR + '[data-theme-control]']: { text: 'Светлая' },
    '{n} errors': { forms: { one: '{n} ошибка', few: '{n} ошибки', many: '{n} ошибок', other: '{n} ошибки' } },
    '{n} B': { text: '{n} Б' },
    '{n} notes': { forms: { one: '{n} заметка', few: '{n} заметки', many: '{n} заметок', other: '{n} заметки' } },
    'Type a code such as <1>pt-BR</1> or a name.': { text: 'Введите код, например <1>pt-BR</1>, или название.' },
    'Open <1>Settings</1> now': { text: 'Откройте <1>Настройки</1> сейчас' },
    'Ouroboros': { text: 'Ouroboros' },
    'New task in {name}': { text: 'Новая задача в {name}' },
    [CODE_PREFIX + 'task.headline.done']: { text: 'Готово' },
};
const RU = { language: 'ru', english: false, revision: 7, entries: RU_ENTRIES };
const EN = { language: '', english: true, revision: 0, entries: {} };

const sent = [];
setMissTransport((payload) => { sent.push(payload); return Promise.resolve({ ok: true }); });

test('a tag is English when empty or an en-* tag, nothing else', () => {
    assert.equal(englishTag(''), true);
    assert.equal(englishTag('en'), true);
    assert.equal(englishTag('en-GB'), true);
    assert.equal(englishTag('ru'), false);
    assert.equal(englishTag('eng'), false);
});

test('an exact memory key is replaced and surrounding whitespace survives', () => {
    applyPayload(RU);
    assert.equal(translateString('Settings'), 'Настройки');
    assert.equal(translateString('Main Chat'), 'Основной чат');
    assert.equal(translateString('\n  Settings  '), '\n  Настройки  ');
    assert.equal(translateString('   '), '   ');
});

test('a string with one number uses its {n} template and the plural form Intl selects', () => {
    applyPayload(RU);
    assert.equal(translateString('1 errors'), '1 ошибка');
    assert.equal(translateString('3 errors'), '3 ошибки');
    assert.equal(translateString('11 errors'), '11 ошибок');
    assert.equal(translateString('25 errors'), '25 ошибок');
    assert.equal(translateString('2.5 errors'), '2.5 ошибки');
    assert.equal(translateString('12 B'), '12 Б');
    // Two numbers are not a template; the string stays.
    assert.equal(translateString('3 of 12 errors'), '3 of 12 errors');
});

test('an unknown string is returned unchanged, non-strings pass through, inherited keys never match', () => {
    applyPayload(RU);
    assert.equal(translateString('Ouroboros ate the tail'), 'Ouroboros ate the tail');
    assert.equal(translateString(''), '');
    assert.equal(translateString(undefined), undefined);
    assert.equal(translateString('constructor'), 'constructor');
    assert.equal(translateString('toString'), 'toString');
    applyPayload(EN);
    assert.equal(translateString('Settings'), 'Settings');
});

test('entryText picks the plural form, then other, then many, then the first form', () => {
    applyPayload(RU);
    assert.equal(entryText({ text: 'x' }, 5), 'x');
    assert.equal(entryText({ forms: { one: 'один', many: 'много' } }, 2.5), 'много');
    assert.equal(entryText({ forms: { few: 'несколько' } }, 1), 'несколько');
    assert.equal(entryText({ forms: {} }, 1), null);
    assert.equal(entryText(null, 1), null);
});

test('an entry whose text equals its source is found, not missing; a miss reports the {n} template', () => {
    applyPayload(RU);
    assert.deepEqual(lookupString('Ouroboros'), { text: 'Ouroboros', found: true });
    assert.deepEqual(lookupString('Files'), { text: 'Files', found: false });
    assert.equal(missKeyFor('12.3 KB'), '{n} KB');
    assert.equal(missKeyFor('5 minutes ago'), '{n} minutes ago');
    assert.equal(missKeyFor('3 of 12 errors'), '3 of 12 errors', 'two numbers are not one template');
    assert.equal(missKeyFor('v7.5.1'), 'v7.5.1');
    const root = el('div', {}, el('span', {}, 'Ouroboros'), el('span', {}, '12.3 KB'));
    createTranslator().applyTo(root);
    assert.deepEqual(pendingMisses(), ['{n} KB'], 'the identity entry is not re-reported; the size reports its template');
    applyPayload(EN);
});

test('the plural map is only written for a tag the engine really supports', () => {
    assert.equal(pluralSelectMap('qya'), null, 'Quenya: Intl would silently fall back to the default locale');
    assert.equal(pluralSelectMap('art-x-vael'), null);
    assert.notEqual(pluralSelectMap('ru'), null);
});

test('tr treats a code whose English moved since generation as a stale miss', async () => {
    applyPayload({ ...RU, entries: { ...RU_ENTRIES, [CODE_PREFIX + 'task.headline.warn']: { text: 'Готово с оговорками', provenance: 'generated', source: 'Done with warnings' } } });
    assert.equal(tr('task.headline.warn', 'Done with warnings'), 'Готово с оговорками');
    assert.equal(tr('task.headline.warn', 'Finished with warnings'), 'Finished with warnings', 'the reworded English shows until regenerated');
    assert.deepEqual(pendingMisses(), [CODE_PREFIX + 'task.headline.warn']);
    sent.length = 0;
    await flushMisses();
    assert.equal(sent[0].items[0].context.stale, true);
    applyPayload(EN);
});

test('applyPayload keeps the payload for binders that paint from it', () => {
    assert.equal(applyPayload(RU) && currentPayload(), RU);
    applyPayload(EN);
    assert.equal(currentPayload(), EN);
});

test('the plural map the browser hands the memory covers 0..100 with a period of 100', () => {
    const ru = pluralSelectMap('ru');
    assert.equal(ru.map['1'], 'one');
    assert.equal(ru.map['3'], 'few');
    assert.equal(ru.map['5'], 'many');
    assert.equal(ru.map['21'], 'one');
    assert.equal(ru.period, 100);
    assert.ok(ru.categories.includes('few'));
    assert.equal(Object.keys(ru.map).length, 101);
    const en = pluralSelectMap('en');
    assert.equal(en.map['1'], 'one');
    assert.equal(en.map['0'], 'other');
});

test('tr answers the English source in English and the memory sentence otherwise; a miss is reported once', async () => {
    applyPayload(EN);
    assert.equal(tr('task.headline.done', 'Done'), 'Done');
    assert.deepEqual(pendingMisses(), []);
    applyPayload(RU);
    assert.equal(tr('task.headline.done', 'Done'), 'Готово');
    assert.equal(tr('task.headline.warn', 'Done with warnings'), 'Done with warnings');
    assert.equal(tr('task.headline.warn', 'Done with warnings'), 'Done with warnings');
    assert.deepEqual(pendingMisses(), [CODE_PREFIX + 'task.headline.warn']);
    sent.length = 0;
    await flushMisses();
    assert.equal(sent.length, 1);
    assert.equal(sent[0].language, 'ru');
    assert.deepEqual(sent[0].items.map((item) => item.key), [CODE_PREFIX + 'task.headline.warn']);
    assert.equal(sent[0].items[0].context.source, 'Done with warnings');
    assert.deepEqual(pendingMisses(), []);
});

test('tx translates a host-composed sentence the memory knows and reports one it does not', async () => {
    applyPayload(EN);
    assert.equal(tx('Not started: the request was empty'), 'Not started: the request was empty');
    applyPayload({ ...RU, entries: { ...RU_ENTRIES, 'Not started: the request was empty': { text: 'Не запущено: запрос пуст' } } });
    assert.equal(tx('Not started: the request was empty'), 'Не запущено: запрос пуст');
    assert.equal(tx('  Not started: the request was empty '), 'Не запущено: запрос пуст');
    assert.equal(tx('Not moved: the project is no longer available'), 'Not moved: the project is no longer available');
    assert.deepEqual(pendingMisses(), ['Not moved: the project is no longer available']);
    sent.length = 0;
    await flushMisses();
    assert.equal(sent[0].items[0].context.role, 'host-text');
    assert.equal(tx(''), '');
});

test('fmt fills placeholders into the translated template and selects the plural form by n', () => {
    applyPayload(RU);
    assert.equal(fmt('New task in {name}', { name: 'Docs' }), 'Новая задача в Docs');
    assert.equal(fmt('{n} notes', { n: 1 }), '1 заметка');
    assert.equal(fmt('{n} notes', { n: 3 }), '3 заметки');
    assert.equal(fmt('{n} notes', { n: 11 }), '11 заметок');
    // A template the memory lacks renders the English source, placeholders filled.
    assert.equal(fmt('Message from task {source}', { source: 'abc' }), 'Message from task abc');
    assert.ok(pendingMisses().includes('Message from task {source}'));
    applyPayload(EN);
    assert.equal(fmt('New task in {name}', { name: 'Docs' }), 'New task in Docs');
    assert.equal(fmt('{n} notes', { n: 3 }), '3 notes');
});

test('the selector lists parse as CSS lists and keep the content the overlay must never touch', () => {
    for (const selector of [EXCLUDE_SELECTOR, SKIP_ROOTS, USER_CONTENT]) {
        assert.ok(selector.length > 0);
        for (const part of selector.split(',')) assert.ok(part.trim().length > 0, selector);
    }
    assert.ok(EXCLUDE_SELECTOR.includes('input'), 'user input is never translated as content');
    assert.ok(SKIP_ROOTS.includes('#chat-messages'), 'the Main transcript stays out of the overlay');
    assert.ok(SKIP_ROOTS.includes('.chat-messages'), 'every Project transcript stays out of the overlay');
    assert.ok(SKIP_ROOTS.includes('#log-entries'), 'the log stream stays out of the observer');
    assert.ok(USER_CONTENT.includes('.nav-project-row'), 'a project name is never translated');
});

// ---------------------------------------------------------------------------
// Minimal DOM. Only what the overlay calls: element/text nodes, `closest` and
// `querySelectorAll` over compound selectors with a descendant combinator, a
// text-node TreeWalker, `replaceChildren`/`createTextNode` for inline composites,
// and a `dataset` that `[data-*]` queries can see.
// ---------------------------------------------------------------------------

const TOKEN = /[.#][\w-]+|\[[^\]]*\]|[a-zA-Z][\w-]*/g;

function matchesCompound(node, compound) {
    return (compound.match(TOKEN) || []).every((token) => {
        if (token[0] === '.') return node.classList.has(token.slice(1));
        if (token[0] === '#') return node.getAttribute('id') === token.slice(1);
        if (token[0] === '[') {
            const body = token.slice(1, -1);
            const eq = body.indexOf('=');
            if (eq < 0) return node.getAttribute(body) !== null;
            const want = body.slice(eq + 1).replace(/^["']|["']$/g, '');
            return node.getAttribute(body.slice(0, eq)) === want;
        }
        return node.tagName === token.toUpperCase();
    });
}

function matchesSelector(node, selector) {
    return selector.split(',').some((part) => {
        const compounds = part.trim().split(/\s+/);
        if (!matchesCompound(node, compounds[compounds.length - 1])) return false;
        let index = compounds.length - 2;
        let ancestor = node.parentElement;
        while (index >= 0) {
            if (!ancestor) return false;
            if (matchesCompound(ancestor, compounds[index])) index -= 1;
            ancestor = ancestor.parentElement;
        }
        return true;
    });
}

const doc = {
    createTreeWalker(root) {
        const texts = [];
        (function collect(node) {
            for (const child of node.childNodes || []) {
                if (child.nodeType === 3) texts.push(child);
                else collect(child);
            }
        })(root);
        let i = 0;
        return { nextNode: () => (i < texts.length ? texts[i++] : null) };
    },
    createTextNode(value) { return new Txt(value); },
};

class Txt {
    constructor(value) {
        this.nodeType = 3;
        this.nodeValue = String(value);
        this.parentElement = null;
        this.isConnected = true;
        this.ownerDocument = doc;
    }
}

class El {
    constructor(tag, attrs) {
        this.nodeType = 1;
        this.tagName = tag.toUpperCase();
        this.attrs = new Map(Object.entries(attrs || {}).map(([k, v]) => [k, String(v)]));
        this.dataset = {};
        this.childNodes = [];
        this.parentElement = null;
        this.isConnected = true;
        this.ownerDocument = doc;
    }

    get classList() {
        return new Set(String(this.attrs.get('class') || '').split(/\s+/).filter(Boolean));
    }

    set textContent(value) { this.replaceChildren(new Txt(value)); }

    get textContent() { return this.childNodes.map((n) => (n.nodeType === 3 ? n.nodeValue : n.textContent)).join(''); }

    getAttribute(name) {
        if (name.startsWith('data-')) {
            const key = name.slice(5).replace(/-([a-z])/g, (m, c) => c.toUpperCase());
            return this.dataset[key] === undefined ? null : this.dataset[key];
        }
        return this.attrs.has(name) ? this.attrs.get(name) : null;
    }

    setAttribute(name, value) { this.attrs.set(name, String(value)); }

    replaceChildren(...nodes) {
        for (const node of this.childNodes) if (!nodes.includes(node)) node.parentElement = null;
        this.childNodes = nodes;
        for (const node of nodes) node.parentElement = this;
    }

    closest(selector) {
        for (let node = this; node; node = node.parentElement) {
            if (matchesSelector(node, selector)) return node;
        }
        return null;
    }

    querySelectorAll(selector) {
        const found = [];
        (function collect(node) {
            for (const child of node.childNodes) {
                if (child.nodeType !== 1) continue;
                if (matchesSelector(child, selector)) found.push(child);
                collect(child);
            }
        })(this);
        return found;
    }
}

function el(tag, attrs, ...children) {
    const node = new El(tag, attrs);
    for (const child of children) {
        const kid = typeof child === 'string' ? new Txt(child) : child;
        kid.parentElement = node;
        node.childNodes.push(kid);
    }
    return node;
}

/** The chrome/user-content mix the review found: a project row, a file row, a plain button. */
function sampleTree() {
    return el('div', {},
        el('div', { class: 'nav-project-item' },
            el('button', { class: 'nav-row nav-project-row', title: 'Delete old logs' },
                el('span', { class: 'nav-row-label' }, 'Delete old logs')),
            el('button', {
                class: 'nav-project-kebab', title: 'Project actions',
                'aria-label': 'Actions for Delete old logs',
            }, '⋯')),
        el('div', { class: 'files-entry' },
            el('span', { class: 'files-entry-name' }, 'Settings'),
            el('span', { class: 'files-entry-meta' }, '12 B')),
        el('h2', { id: 'project-panel-title' }, 'Delete old logs'),
        el('button', { class: 'nav-row', title: 'Settings' },
            el('span', { class: 'nav-row-label' }, 'Settings')),
        el('input', { placeholder: 'Search', title: 'Search' }));
}

const text = (node) => node.childNodes[0].nodeValue;

function snapshot(root) {
    const out = [];
    (function collect(node) {
        for (const child of node.childNodes) {
            if (child.nodeType === 3) { out.push(child.nodeValue); continue; }
            for (const attr of ['placeholder', 'title', 'aria-label']) {
                const value = child.getAttribute(attr);
                if (value !== null) out.push(`${attr}=${value}`);
            }
            collect(child);
        }
    })(root);
    return out;
}

test('owner-supplied names are left alone while the chrome around them is translated', () => {
    applyPayload(RU);
    const root = sampleTree();
    const [projectItem, filesEntry, panelTitle, plainRow, input] = root.childNodes;
    createTranslator().applyTo(root);

    // A project named "Delete old logs" must not become "Удалить old logs".
    assert.equal(text(projectItem.childNodes[0].childNodes[0]), 'Delete old logs');
    assert.equal(projectItem.childNodes[0].getAttribute('title'), 'Delete old logs');
    assert.equal(panelTitle.childNodes[0].nodeValue, 'Delete old logs');
    // A file named "Settings" stays a file name, not the Settings page.
    assert.equal(text(filesEntry.childNodes[0]), 'Settings');
    // A composed label merely embedding the name has no exact key and stays.
    assert.equal(projectItem.childNodes[1].getAttribute('aria-label'), 'Actions for Delete old logs');

    // The gate is not a blanket off-switch: siblings and metadata still translate.
    assert.equal(text(filesEntry.childNodes[1]), '12 Б');
    assert.equal(text(plainRow.childNodes[0]), 'Настройки');
    assert.equal(plainRow.getAttribute('title'), 'Настройки');
    // Inputs are excluded as content but their chrome attributes are translated.
    assert.equal(input.getAttribute('placeholder'), 'Поиск');
});

test('authored content outside a transcript keeps every text and attribute and reports no miss', () => {
    applyPayload(RU);
    assert.ok(SKIP_ROOTS.includes('[data-i18n-authored]'));
    const authored = (node) => { node.dataset.i18nAuthored = ''; return node; };
    // A delivered document in the reader: a file named like a known word, an unknown size
    // line, and Markdown whose link and image reference carry author titles.
    const identity = authored(el('div', { class: 'document-reader-identity' },
        el('h2', { class: 'document-reader-title' }, 'Settings'),
        el('div', { class: 'document-reader-meta' }, 'Quarterly numbers')));
    const link = el('a', { class: 'md-link', href: 'https://example.com/', title: 'Settings' }, 'Search');
    const content = authored(el('div', { class: 'document-reader-markdown' },
        el('p', {}, link, ' and the rest'),
        el('span', { title: 'Quarterly numbers', 'aria-label': 'Main Chat' }, 'Image: chart')));
    const body = el('div', { class: 'document-reader-body', 'aria-label': 'Search' }, content);
    const close = el('button', { class: 'btn btn-default', title: 'Settings' }, 'Settings');
    const root = el('dialog', { class: 'document-reader' }, identity, el('div', {}, close), body);
    const before = [...snapshot(identity), ...snapshot(content)];
    const translator = createTranslator();
    translator.applyTo(root);
    // What the observer hands over after a later write inside: the element, then its text.
    translator.applyTo(link);
    translator.applyTo(link.childNodes[0]);
    assert.deepEqual([...snapshot(identity), ...snapshot(content)], before, 'known and unknown authored strings stay');
    assert.deepEqual(pendingMisses(), [], 'and none is reported as a miss');
    // The reader's own chrome around it is still translated.
    assert.deepEqual([text(close), close.getAttribute('title'), body.getAttribute('aria-label')],
        ['Настройки', 'Настройки', 'Поиск']);
});

test('a help paragraph of several sentences is an ordinary key; only the memory\'s own bound refuses', () => {
    applyPayload(RU);
    const sentence = 'Interface language for this installation: the desktop window, browsers, the Telegram app and ' +
        'Ouroboros\'s own lines about tasks. Translations are generated by the light model and stored as install data.';
    const paragraph = `${sentence}\n           ${sentence}\n           ${sentence}`;   // a template literal's layout
    const key = [sentence, sentence, sentence].join(' ');
    assert.ok(key.length > 400 && key.length < 2000, String(key.length));
    const copy = el('div', { class: 'settings-section-copy' }, paragraph);
    const root = el('div', {}, copy, el('div', {}, 'x'.repeat(2001)));
    createTranslator().applyTo(root);
    assert.deepEqual(pendingMisses(), [key], 'reported as the reader sees it: one space per line break; the over-long string is not reported');
    applyPayload({ ...RU, entries: { ...RU.entries, [key]: { text: 'Абзац.' } } });
    createTranslator().applyTo(root);
    assert.equal(copy.textContent, 'Абзац.', 'the collapsed key finds the entry whatever the source layout');
    applyPayload(RU);
});

test('untranslated chrome is reported as a miss; volatile strings and user content are not', async () => {
    applyPayload(RU);
    const root = el('div', {},
        el('span', { class: 'nav-row-label' }, 'Files'),
        el('span', { class: 'files-entry-meta' }, '12 B'),
        el('span', {}, '⋯'),
        el('span', { class: 'files-entry-name' }, 'README'),
        el('span', { class: 'files-entry-meta' }, '2026-10-03T10:00:00Z'));
    createTranslator().applyTo(root);
    assert.deepEqual(pendingMisses(), ['Files']);
    sent.length = 0;
    await flushMisses();
    assert.equal(sent[0].items[0].context.role, 'span');
});

test('a scoped word wins only inside its scope and restores like any other', () => {
    applyPayload(RU);
    const themes = el('div', {}, el('button', {}, 'Light'), el('button', {}, 'System'));
    themes.dataset.themeControl = '';
    const root = el('div', {}, themes, el('button', {}, 'Light'));
    const translator = createTranslator();
    translator.applyTo(root);
    assert.equal(text(themes.childNodes[0]), 'Светлая');
    assert.equal(text(themes.childNodes[1]), 'Системная');
    assert.equal(text(root.childNodes[1]), 'Лёгкий');
    translator.restore(root);
    assert.equal(text(themes.childNodes[0]), 'Light');
    assert.equal(text(root.childNodes[1]), 'Light');
});

test('an inline composite translates as one sentence, keeps <code> verbatim, and restores its nodes', () => {
    applyPayload(RU);
    const code = el('code', {}, 'pt-BR');
    const help = el('div', { class: 'settings-inline-note' }, 'Type a code such as ', code, ' or a name.');
    const root = el('div', {}, help);
    const translator = createTranslator();
    translator.applyTo(root);
    assert.equal(help.textContent, 'Введите код, например pt-BR, или название.');
    assert.ok(help.childNodes.includes(code), 'the original <code> element is reused, not cloned');
    assert.equal(code.textContent, 'pt-BR');
    // Idempotent: a second pass starts from the stashed English, not from the translation.
    translator.applyTo(root);
    assert.equal(help.textContent, 'Введите код, например pt-BR, или название.');
    translator.restore(root);
    assert.equal(help.textContent, 'Type a code such as pt-BR or a name.');
    assert.equal(help.childNodes.length, 3);
});

test('an inline composite with a mutable slot translates and restores its exact English', () => {
    applyPayload(RU);
    const strong = el('strong', {}, 'Settings');
    const line = el('p', {}, 'Open ', strong, ' now');
    const root = el('div', {}, line);
    const translator = createTranslator();
    translator.applyTo(root);
    assert.equal(line.textContent, 'Откройте Настройки сейчас');
    assert.ok(line.childNodes.includes(strong), 'the <strong> element is reused');
    assert.equal(strong.textContent, 'Настройки', 'the slot text is translated in place');
    translator.applyTo(root);
    assert.equal(line.textContent, 'Откройте Настройки сейчас', 'idempotent');
    translator.restore(root);
    assert.equal(line.textContent, 'Open Settings now', 'the English inside the slot comes back too');
    assert.equal(strong.textContent, 'Settings');
    // The same through a language switch on the document.
    applyPayload(EN);
});

test('producer-written and skip-marked nodes are never overlay keys', () => {
    applyPayload(RU);
    const status = el('div', {}, 'Русский: 5 translated · 2 pending');
    status.dataset.i18nSkip = '';
    const root = el('div', {}, status, el('span', {}, 'Files'));
    createTranslator().applyTo(root);
    assert.deepEqual(pendingMisses(), ['Files']);
    applyPayload(EN);
});

test('a producer-written fmt result is left alone by the overlay', () => {
    applyPayload(RU);
    const target = el('span', {});
    const root = el('div', {}, target);
    assert.equal(fmtInto(target, 'New task in {name}', { name: 'Settings' }), 'Новая задача в Settings');
    assert.equal(target.dataset.i18nFmt, 'New task in {name}');
    createTranslator().applyTo(root);
    // Without the mark the overlay would see "Settings" inside and must not re-touch it.
    assert.equal(target.textContent, 'Новая задача в Settings');
});

test('restore returns every rewritten node and attribute to its exact English source', () => {
    applyPayload(RU);
    const root = sampleTree();
    const before = JSON.stringify(snapshot(root));
    const translator = createTranslator();
    translator.applyTo(root);
    assert.notEqual(JSON.stringify(snapshot(root)), before);
    translator.restore(root);
    assert.equal(JSON.stringify(snapshot(root)), before);
    root.querySelectorAll('button,span,input,h2,div').forEach((node) => {
        assert.deepEqual(node.dataset, {}, 'no bookkeeping is left behind');
        assert.equal(node.__ouroAttrs, undefined);
    });
});

test('restore never clobbers text or attributes the app rewrote while the language was on', () => {
    applyPayload(RU);
    const root = sampleTree();
    const translator = createTranslator();
    translator.applyTo(root);
    const row = root.childNodes[3];
    row.childNodes[0].childNodes[0].nodeValue = 'Live value';
    row.setAttribute('title', 'Live title');
    translator.restore(root);
    assert.equal(text(row.childNodes[0]), 'Live value');
    assert.equal(row.getAttribute('title'), 'Live title');
});

// ---------------------------------------------------------------------------
// Observer and language switch. The stubs record every MutationObserver ever
// constructed, so an orphaned one cannot hide behind the survivor.
// ---------------------------------------------------------------------------

function withBrowser(body, fn) {
    const observers = [];
    const frames = [];
    const keys = ['document', 'window', 'localStorage', 'requestAnimationFrame', 'MutationObserver'];
    const saved = keys.map((key) => [key, key in globalThis, globalThis[key]]);
    const documentElement = { lang: '', dir: '', removeAttribute(name) { this[name] = ''; } };
    const stored = new Map();
    globalThis.document = { body, documentElement };
    globalThis.window = { dispatchEvent() {} };
    globalThis.localStorage = { getItem: (k) => (stored.has(k) ? stored.get(k) : null), setItem(k, v) { stored.set(k, String(v)); } };
    globalThis.requestAnimationFrame = (cb) => frames.push(cb);
    globalThis.MutationObserver = class {
        constructor(cb) { this.cb = cb; this.live = false; observers.push(this); }
        observe() { this.live = true; }
        disconnect() { this.live = false; }
    };
    const deliver = (records) => {
        for (const observer of observers) if (observer.live) observer.cb(records);
        while (frames.length) frames.shift()();
    };
    const restore = () => {
        for (const [key, existed, value] of saved) {
            if (existed) globalThis[key] = value;
            else delete globalThis[key];
        }
    };
    return Promise.resolve(fn({ deliver, observers, documentElement, stored })).finally(restore);
}

test('the observer translates nodes added later and stops dead on disconnect', () => {
    applyPayload(RU);
    const root = sampleTree();
    return withBrowser(root, ({ deliver }) => {
        const translator = createTranslator();
        translator.observe(root);
        const added = el('span', { class: 'nav-row-label' }, 'Settings');
        added.parentElement = root;
        root.childNodes.push(added);
        deliver([{ type: 'childList', addedNodes: [added] }]);
        assert.equal(text(added), 'Настройки');

        translator.disconnect();
        const later = el('span', { class: 'nav-row-label' }, 'Files');
        later.parentElement = root;
        root.childNodes.push(later);
        deliver([{ type: 'childList', addedNodes: [later] }]);
        assert.equal(text(later), 'Files');
    });
});

test('a payload switch paints the document, sets <html lang>/dir, and remembers the tag', () => {
    const root = sampleTree();
    return withBrowser(root, async ({ documentElement, stored }) => {
        await setLanguage('ru', { ...RU, profile: { direction: 'rtl' } });
        assert.equal(text(root.childNodes[3].childNodes[0]), 'Настройки');
        assert.equal(documentElement.lang, 'ru');
        assert.equal(documentElement.dir, 'rtl');
        assert.equal(stored.get('ouro.language'), 'ru');
        await setLanguage('en', EN);
        assert.equal(documentElement.lang, 'en');
        assert.equal(documentElement.dir, '');
        assert.equal(stored.get('ouro.language'), '');
    });
});

test('concurrent switches leave one translator, so English comes back whole', () => {
    const root = sampleTree();
    return withBrowser(root, async ({ deliver, observers }) => {
        const english = JSON.stringify(snapshot(root));
        await Promise.all([setLanguage('ru', RU), setLanguage('ru', RU)]);
        assert.equal(text(root.childNodes[3].childNodes[0]), 'Настройки');
        assert.equal(observers.filter((o) => o.live).length, 1);

        await setLanguage('en', EN);
        assert.equal(JSON.stringify(snapshot(root)), english);
        assert.equal(observers.filter((o) => o.live).length, 0);
        // A stranded observer would retranslate on the next frame; none may.
        deliver([{ type: 'childList', addedNodes: [root] }]);
        assert.equal(JSON.stringify(snapshot(root)), english);
    });
});


test('an invented language renders the `other` form and a stored plural map wins over the engine', () => {
    applyPayload({ ...RU, language: 'art-x-vael', entries: { '{n} notes': { forms: { one: 'ONE {n}', other: 'OTHER {n}' } } }, plural_select: null });
    assert.equal(fmt('{n} notes', { n: 1 }), 'OTHER 1', 'no CLDR data: the agreed fallback, not the browser locale\'s grammar');
    applyPayload({ ...RU, language: 'art-x-vael', entries: { '{n} notes': { forms: { one: 'ONE {n}', other: 'OTHER {n}' } } }, plural_select: { map: { '1': 'one', '0': 'other' }, period: 10 } });
    assert.equal(fmt('{n} notes', { n: 1 }), 'ONE 1', 'an imported pack\'s own rules select the form');
    assert.equal(fmt('{n} notes', { n: 11 }), 'ONE 11', 'periodic map');
    assert.equal(fmt('{n} notes', { n: 5 }), 'OTHER 5');
    applyPayload(RU);
});

test('a pin keeps rendering after an upstream reword; only a generated entry yields to the new English', () => {
    applyPayload({ ...RU, entries: {
        'code:task.headline.done': { text: 'Готово', provenance: 'imported', source: 'Done' },
        'code:task.headline.warn': { text: 'С предупреждениями', provenance: 'generated', source: 'Done with warnings' },
        'code:cancel.reason_preview_note': { text: '(превью)', provenance: 'generated', source: '(preview; the full reason is kept with the task)' },
    } });
    assert.equal(tr('task.headline.done', 'Finished'), 'Готово', 'the import stays visible whatever the English says');
    assert.equal(tr('task.headline.warn', 'Finished with warnings'), 'Finished with warnings', 'a generated entry whose English moved shows the English');
    assert.ok(pendingMisses().includes('code:task.headline.warn') && !pendingMisses().includes('code:task.headline.done'), 'only the generated one is reported stale');
    assert.equal(tr('cancel.reason_preview_note', ' (preview; the full reason is kept with the task)'), '(превью)', 'edge whitespace is not a reword');
    applyPayload(RU);
});

test('re-applying the overlay to an unchanged inline composite writes nothing', () => {
    applyPayload(RU);
    const code = el('code', {}, 'pt-BR');
    const help = el('div', { class: 'settings-inline-note' }, 'Type a code such as ', code, ' or a name.');
    const root = el('div', {}, help);
    const translator = createTranslator();
    translator.applyTo(root);
    assert.equal(help.textContent, 'Введите код, например pt-BR, или название.');
    let writes = 0;
    const original = help.replaceChildren.bind(help);
    help.replaceChildren = (...nodes) => { writes += 1; return original(...nodes); };
    translator.applyTo(root);
    translator.applyTo(help);
    assert.equal(writes, 0, 'our output stands and nothing changed: the observer must not be woken');
    assert.equal(help.textContent, 'Введите код, например pt-BR, или название.');
    translator.restore(root);
    assert.equal(help.textContent, 'Type a code such as pt-BR or a name.');
});


test('an inline composite the app rewrote keeps the app\'s new content; the old stash is never put back', () => {
    applyPayload({ ...RU, entries: { ...RU_ENTRIES, 'LAN URL: <1>http://192.168.1.10:8765</1>': { text: 'Адрес в сети: <1>http://192.168.1.10:8765</1>' } } });
    const code = el('code', {}, 'pt-BR');
    const help = el('div', { class: 'settings-inline-note' }, 'Type a code such as ', code, ' or a name.');
    const root = el('div', {}, help);
    const translator = createTranslator();
    translator.applyTo(root);
    assert.equal(help.textContent, 'Введите код, например pt-BR, или название.');
    // The app replaces the sentence with another composite (a different inline element).
    const link = el('a', {}, 'http://192.168.1.10:8765');
    help.replaceChildren(doc.createTextNode('LAN URL: '), link);
    translator.applyTo(root);
    assert.equal(help.textContent, 'Адрес в сети: http://192.168.1.10:8765', 'the new sentence is translated');
    assert.ok(help.childNodes.includes(link), 'with the app\'s own new element, not the stale one');
    assert.ok(!help.childNodes.includes(code));
    // And one with no translation stays exactly what the app wrote.
    const other = el('a', {}, 'http://10.0.0.2:8765');
    help.replaceChildren(doc.createTextNode('Open '), other, doc.createTextNode(' on this network.'));
    translator.applyTo(root);
    assert.equal(help.textContent, 'Open http://10.0.0.2:8765 on this network.');
    assert.ok(help.childNodes.includes(other));
    applyPayload(RU);
});

test('a stored plural map answers its exact entry before its period', () => {
    applyPayload({ ...RU, language: 'art-x-vael', entries: { '{n} notes': { forms: { one: 'ONE {n}', other: 'OTHER {n}' } } }, plural_select: { map: { '0': 'other', '10': 'one' }, period: 10 } });
    assert.equal(fmt('{n} notes', { n: 10 }), 'ONE 10', 'the exact entry, as Python selects');
    assert.equal(fmt('{n} notes', { n: 20 }), 'OTHER 20', 'then the periodic one');
    applyPayload(RU);
});


test('a string that is already a translation in the memory is never reported as a miss', () => {
    applyPayload({ ...RU, entries: { ...RU_ENTRIES, 'code:task.chip.paused': { text: 'На паузе', provenance: 'generated', source: 'Paused' } } });
    // A producer wrote its label through tr(); the node happens to sit where the overlay walks.
    const root = el('div', {}, el('span', {}, tr('task.chip.paused', 'Paused')), el('span', {}, 'Files'));
    createTranslator().applyTo(root);
    assert.deepEqual(pendingMisses(), ['Files'], 'the Russian label is not queued as a new English key');
    applyPayload(RU);
});

test('the read behind the first socket open brings an update that landed before the subscription existed', async () => {
    const { markBootRead, pendingBootRead, refreshDictionary, dictionaryRevision } = await import('../modules/i18n.js');
    const { apiClient } = await import('../modules/api_client.js');
    const savedFetch = globalThis.fetch;
    let served = { ...RU, revision: 1, entries: { Working: { text: 'Работает (черновик)' } } };
    const reads = [];
    globalThis.fetch = async (url) => { reads.push(String(url)); const body = served; return { ok: true, status: 200, json: async () => body }; };
    try {
        // app.js at boot: the first answer paints, and the boot read settles only once it is applied.
        await markBootRead(apiClient.uiI18n().then((i18n) => setLanguage(i18n.language, i18n).then(() => i18n)));
        assert.equal(currentPayload().revision, 1, 'a control mounted behind the boot read finds its payload applied');
        // The generator finishes: its frame goes out before this page's socket has subscribed.
        served = { ...RU, revision: 2, entries: { Working: { text: 'Работает' } } };
        // app.js on every socket open, the first included.
        await pendingBootRead().then(refreshDictionary);
        assert.equal(dictionaryRevision(), 2);
        assert.equal(tx('Working'), 'Работает');
        assert.deepEqual(reads, ['/api/ui/i18n', '/api/ui/i18n'], 'the boot read and the read behind the subscription');
        // The socket may open first: the read behind it still lands after the boot answer, never before it.
        served = { ...RU, revision: 3, entries: { Working: { text: 'Работает!' } } };
        let answerBoot;
        const slowBoot = new Promise((resolve) => { answerBoot = resolve; });
        markBootRead(slowBoot.then((i18n) => setLanguage(i18n.language, i18n).then(() => i18n)));
        const behind = pendingBootRead().then(refreshDictionary);
        answerBoot({ ...RU, revision: 2, entries: { Working: { text: 'Работает' } } });   // the older answer arrives late
        await behind;
        assert.equal(dictionaryRevision(), 3, 'the older boot answer did not land over the newer read');
    } finally {
        globalThis.fetch = savedFetch;
        markBootRead(null);
        applyPayload(RU);
    }
    const source = readFileSync(new URL('../app.js', import.meta.url), 'utf8');
    const open = source.slice(source.indexOf("ws.on('open', () => {"), source.indexOf("ws.on('close', () => {"));
    assert.match(open, /pendingBootRead\(\)\.then\(refreshDictionary\);/, 'app.js re-reads on every open, behind the boot read');
    assert.match(source, /\.then\(\(i18n\) => setLanguage\(i18n\.language, i18n\)\.then\(\(\) => i18n\)\)/, 'and its boot read settles after the payload is applied');
});

test('the direction seam has no opinion on a language whose script the engine does not know', () => {
    for (const tag of ['art-x-vael', 'zz', 'not a tag', '']) assert.equal(localeDirection(tag), '', tag);
    const known = (tag) => { try { const locale = new Intl.Locale(tag); return Boolean(locale.maximize().script) && Boolean(locale.getTextInfo?.() || locale.textInfo); } catch { return false; } };
    if (known('ar')) assert.equal(localeDirection('ar'), 'rtl');
    if (known('de')) assert.equal(localeDirection('de'), 'ltr');
    // Northern Luri: Arabic script, and no plural data in the engines that ship its script.
    if (known('lrc')) assert.equal(localeDirection('lrc'), 'rtl', 'the script decides, not the plural data');
});

// The memory translates English source text, so Cyrillic typed into a module's literal would
// reach every English reader as-is. Comments may quote the owner verbatim and never paint; a
// glyph table spelled as \u escapes (the matrix rain) is artwork, not words.
function cyrillicLiterals(source) {
    const hits = [];
    acorn.parse(source, {
        ecmaVersion: 'latest', sourceType: 'module', locations: true,
        onToken: (token) => {
            if ([acorn.tokTypes.string, acorn.tokTypes.template].includes(token.type)
                && /[\u0400-\u04FF]/.test(source.slice(token.start, token.end))) hits.push(token.loc.start.line);
        },
    });
    return hits;
}

test('every string a module or page can paint is English source text: no Cyrillic literal', () => {
    assert.deepEqual(cyrillicLiterals('// Настройки\nconst a = "Settings";'), [], 'a comment is not a literal');
    assert.deepEqual(cyrillicLiterals('const glyphs = "\\u0430\\u0431";'), [], 'an escaped glyph table is not text');
    assert.deepEqual(cyrillicLiterals('const a = "Settings";\nconst b = `Настройки ${a}`;\nconst c = \'Поиск\';'), [2, 3]);
    const modules = new URL('../modules/', import.meta.url);
    const files = readdirSync(modules).filter((name) => name.endsWith('.js'));
    assert.ok(files.includes('subagents_settings.js') && files.length > 100, 'the scan reads the module directory');
    const hits = files.flatMap((name) => cyrillicLiterals(readFileSync(new URL(name, modules), 'utf8'))
        .map((line) => `web/modules/${name}:${line}`));
    assert.deepEqual(hits, []);
    for (const page of ['../index.html', '../onboarding_template.html']) {
        assert.doesNotMatch(readFileSync(new URL(page, import.meta.url), 'utf8'), /[\u0400-\u04FF]/, page);
    }
});
