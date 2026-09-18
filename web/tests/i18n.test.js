// The translation overlay: string lookup, the DOM walker, and the language
// switch. `createTranslator` and `setLanguage` touch only the handful of DOM
// methods stubbed below (closest/querySelectorAll/createTreeWalker/dataset), so
// the house node-stub idiom covers them without a browser.
import test from 'node:test';
import assert from 'node:assert/strict';
import {
    createTranslator, setLanguage, translateString,
    EXCLUDE_SELECTOR, SKIP_ROOTS, USER_CONTENT,
} from '../modules/i18n.js';
import { ru, ruPatterns, plural } from '../i18n/ru.js';

test('an exact dictionary key is replaced', () => {
    assert.equal(translateString('Settings', ru, ruPatterns), 'Настройки');
    assert.equal(translateString('Main Chat', ru, ruPatterns), 'Основной чат');
});

test('surrounding whitespace survives the replacement', () => {
    assert.equal(translateString('\n  Settings  ', ru, ruPatterns), '\n  Настройки  ');
    assert.equal(translateString('   ', ru, ruPatterns), '   ');
});

test('interpolated strings fall through to the pattern list, with Russian plurals', () => {
    assert.equal(translateString('1 errors', ru, ruPatterns), '1 ошибка');
    assert.equal(translateString('3 errors', ru, ruPatterns), '3 ошибки');
    assert.equal(translateString('11 errors', ru, ruPatterns), '11 ошибок');
    assert.equal(translateString('25 errors', ru, ruPatterns), '25 ошибок');
    assert.equal(plural(2, 'запуск', 'запуска', 'запусков'), 'запуска');
});

test('an unknown string is returned unchanged, and non-strings pass through', () => {
    assert.equal(translateString('Ouroboros ate the tail', ru, ruPatterns), 'Ouroboros ate the tail');
    assert.equal(translateString('', ru, ruPatterns), '');
    assert.equal(translateString(undefined, ru, ruPatterns), undefined);
    // No dictionary at all is the boot state before ru.js resolves.
    assert.equal(translateString('Settings', null, null), 'Settings');
});

test('inherited Object keys are not mistaken for translations', () => {
    assert.equal(translateString('constructor', ru, ruPatterns), 'constructor');
    assert.equal(translateString('toString', ru, ruPatterns), 'toString');
});

test('the hot-loop roots are a subset of nothing else: every selector parses as a CSS list', () => {
    for (const selector of [EXCLUDE_SELECTOR, SKIP_ROOTS, USER_CONTENT]) {
        assert.ok(selector.length > 0);
        for (const part of selector.split(',')) assert.ok(part.trim().length > 0, selector);
    }
    assert.ok(EXCLUDE_SELECTOR.includes('input'), 'user input is never translated as content');
    assert.ok(SKIP_ROOTS.includes('#log-entries'), 'the log stream stays out of the observer');
    assert.ok(USER_CONTENT.includes('.nav-project-row'), 'a project name is never translated');
});

// ---------------------------------------------------------------------------
// Minimal DOM. Only what the overlay calls: element/text nodes, `closest` and
// `querySelectorAll` over compound selectors with a descendant combinator, a
// text-node TreeWalker, and a `dataset` that `[data-*]` queries can see.
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

    getAttribute(name) {
        if (name.startsWith('data-')) {
            const key = name.slice(5).replace(/-([a-z])/g, (m, c) => c.toUpperCase());
            return this.dataset[key] === undefined ? null : this.dataset[key];
        }
        return this.attrs.has(name) ? this.attrs.get(name) : null;
    }

    setAttribute(name, value) { this.attrs.set(name, String(value)); }

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

test('owner-supplied names are left alone while the chrome around them is translated', () => {
    const root = sampleTree();
    const [projectItem, filesEntry, panelTitle, plainRow, input] = root.childNodes;
    createTranslator({ dict: ru, patterns: ruPatterns }).applyTo(root);

    // A project named "Delete old logs" must not become "Удалить old logs".
    assert.equal(text(projectItem.childNodes[0].childNodes[0]), 'Delete old logs');
    assert.equal(projectItem.childNodes[0].getAttribute('title'), 'Delete old logs');
    assert.equal(panelTitle.childNodes[0].nodeValue, 'Delete old logs');
    // A file named "Settings" stays a file name, not the Settings page.
    assert.equal(text(filesEntry.childNodes[0]), 'Settings');
    // A composed label merely embedding the name is safe: every pattern is anchored.
    assert.equal(projectItem.childNodes[1].getAttribute('aria-label'), 'Actions for Delete old logs');

    // The gate is not a blanket off-switch: siblings and metadata still translate.
    assert.equal(text(filesEntry.childNodes[1]), '12 Б');
    assert.equal(text(plainRow.childNodes[0]), 'Настройки');
    assert.equal(plainRow.getAttribute('title'), 'Настройки');
    // Inputs are excluded as content but their chrome attributes are translated.
    assert.equal(input.getAttribute('placeholder'), 'Поиск');
});

test('restore returns every rewritten node and attribute to its exact English source', () => {
    const root = sampleTree();
    const before = JSON.stringify(snapshot(root));
    const translator = createTranslator({ dict: ru, patterns: ruPatterns });
    translator.applyTo(root);
    assert.notEqual(JSON.stringify(snapshot(root)), before);
    translator.restore(root);
    assert.equal(JSON.stringify(snapshot(root)), before);
    root.querySelectorAll('button,span,input,h2,div').forEach((node) => {
        assert.deepEqual(node.dataset, {}, 'no bookkeeping is left behind');
    });
});

test('restore never clobbers text or attributes the app rewrote while Russian was on', () => {
    const root = sampleTree();
    const translator = createTranslator({ dict: ru, patterns: ruPatterns });
    translator.applyTo(root);
    const row = root.childNodes[3];
    row.childNodes[0].childNodes[0].nodeValue = 'Live value';
    row.setAttribute('title', 'Live title');
    translator.restore(root);
    assert.equal(text(row.childNodes[0]), 'Live value');
    assert.equal(row.getAttribute('title'), 'Live title');
});

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

// ---------------------------------------------------------------------------
// Observer and language switch. The stubs record every MutationObserver ever
// constructed, so an orphaned one cannot hide behind the survivor.
// ---------------------------------------------------------------------------

function withBrowser(body, fn) {
    const observers = [];
    const frames = [];
    const keys = ['document', 'window', 'localStorage', 'requestAnimationFrame', 'MutationObserver'];
    const saved = keys.map((key) => [key, key in globalThis, globalThis[key]]);
    globalThis.document = { body, documentElement: {} };
    globalThis.window = { dispatchEvent() {} };
    globalThis.localStorage = { getItem: () => null, setItem() {} };
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
    return Promise.resolve(fn({ deliver, observers })).finally(restore);
}

test('the observer translates nodes added later and stops dead on disconnect', () => {
    const root = sampleTree();
    return withBrowser(root, ({ deliver }) => {
        const translator = createTranslator({ dict: ru, patterns: ruPatterns });
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

test('concurrent switches to Russian leave one translator, so English comes back whole', () => {
    const root = sampleTree();
    return withBrowser(root, async ({ deliver }) => {
        const english = JSON.stringify(snapshot(root));
        // Both calls race the dynamic import of the dictionary; without the switch
        // queue each would build a translator and strand the loser's observer.
        await Promise.all([setLanguage('ru'), setLanguage('ru')]);
        assert.equal(text(root.childNodes[3].childNodes[0]), 'Настройки');

        await setLanguage('en');
        assert.equal(JSON.stringify(snapshot(root)), english);
        // A stranded observer would retranslate on the next frame; none may.
        deliver([{ type: 'childList', addedNodes: [root] }]);
        assert.equal(JSON.stringify(snapshot(root)), english);
    });
});
