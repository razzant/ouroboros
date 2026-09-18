/** Runtime translation overlay.
 *
 * The UI is authored in English at every call site and stays that way: this
 * module rewrites rendered text nodes and a short list of attributes after the
 * fact, so `git merge managed/ouroboros` never sees a translation diff. Unknown
 * strings are left untouched, which is also the fallback for a partial dictionary.
 */

// NodeFilter.SHOW_TEXT. The constant is spelled numerically because the
// hermetic no-undef walker (web/tests/no_undef.test.js) does not carry
// NodeFilter in its browser-global list.
const SHOW_TEXT = 4;

const ATTRS = ['placeholder', 'title', 'aria-label'];
const ATTR_DATA_KEYS = { placeholder: 'ouroPlaceholder', title: 'ouroTitle', 'aria-label': 'ouroAriaLabel' };

/** Content whose text is data, code, or user input — never translated. */
export const EXCLUDE_SELECTOR = [
    '.chat-bubble .message', '.chat-bubble .msg-time', '[data-chat-markdown-enhanced]',
    '.ui-rich-content', '.md-code-block', '.md-mermaid-stage', '.md-table-wrap', 'pre', 'code',
    '.log-raw', '.log-body', '.log-pill', '.log-ts', '.log-repeat', '.log-task-summary',
    '.log-task-timeline', '.files-editor', '.files-preview-content', '.files-preview-path',
    '.files-preview-meta', '.chat-live-line-body', '.chat-live-line-time',
    '.chat-live-line-repeat', '.chat-live-meta', 'input', 'textarea', 'select',
    '[contenteditable="true"]',
].join(',');

/** Subtrees rewritten per WebSocket frame: skipped whole, for cost not correctness. */
export const SKIP_ROOTS = [
    '#chat-messages', '.chat-live-card', '#log-entries', '.files-preview-content', '.files-editor',
].join(',');

/**
 * Roots where a BARE owner-supplied name lands — a project title or a path
 * segment. A project named "Delete old logs" would otherwise be eaten by the
 * catch-all `/^Delete (.+)$/` pattern, so these are gated like SKIP_ROOTS and
 * not like EXCLUDE_SELECTOR: the name reaches `.nav-project-row` as the `title`
 * attribute as well as as its label's text, and only a whole-element gate stops
 * both. Composed strings that merely embed a name ("Actions for X") are safe —
 * every pattern is anchored — so their elements stay translatable.
 */
export const USER_CONTENT = [
    '.nav-project-row', '#project-panel-title', '.chat-live-project-name',
    '.files-entry-name', '.files-crumb',
].join(',');

/**
 * Translate one source string, or return it unchanged.
 * Leading/trailing whitespace is preserved around the translated core so that
 * inline markup spacing survives.
 */
export function translateString(text, dict, patterns) {
    if (typeof text !== 'string') return text;
    const parts = /^(\s*)([\s\S]*?)(\s*)$/.exec(text);
    const core = parts[2];
    if (!core) return text;
    const exact = dict ? dict[core] : undefined;
    if (typeof exact === 'string') return parts[1] + exact + parts[3];
    for (const [pattern, replacer] of patterns || []) {
        if (pattern.test(core)) return parts[1] + core.replace(pattern, replacer) + parts[3];
    }
    return text;
}

/**
 * DOM side of the overlay. `applyTo` is idempotent: the English original is
 * stashed on each rewritten node, so a second pass (or a language switch back)
 * starts from the source string, never from a translation.
 */
export function createTranslator({
    dict = {}, patterns = [], excludeSelector = EXCLUDE_SELECTOR,
    skipRoots = SKIP_ROOTS, userContent = USER_CONTENT,
} = {}) {
    let observer = null;
    let pending = null;

    const roots = [skipRoots, userContent].filter(Boolean).join(',');
    /** Nothing in here is translated at all — neither text nor attributes. */
    const blocked = (el) => Boolean(roots && el.closest(roots));
    const excluded = (el) => blocked(el) || Boolean(excludeSelector && el.closest(excludeSelector));

    function applyTextNode(node) {
        const current = node.nodeValue;
        // Our own last output still in place → retranslate from the stored English;
        // anything else means the app rewrote the node, so that value is the source.
        const source = node.__ouroOut !== undefined && current === node.__ouroOut ? node.__ouroSrc : current;
        const out = translateString(source, dict, patterns);
        if (out === source) {
            delete node.__ouroSrc;
            delete node.__ouroOut;
            if (current !== source) node.nodeValue = source;
            return;
        }
        node.__ouroSrc = source;
        node.__ouroOut = out;
        if (current !== out) node.nodeValue = out;
    }

    function applyAttr(el, attr) {
        const current = el.getAttribute(attr);
        if (current === null) return;
        const key = ATTR_DATA_KEYS[attr];
        const saved = el.dataset[key];
        const source = saved !== undefined && current === translateString(saved, dict, patterns) ? saved : current;
        const out = translateString(source, dict, patterns);
        if (out === source) {
            delete el.dataset[key];
            if (current !== source) el.setAttribute(attr, source);
            return;
        }
        el.dataset[key] = source;
        if (current !== out) el.setAttribute(attr, out);
    }

    function applyElement(el) {
        // Inputs are excluded as content but their placeholder/title/aria-label
        // are chrome, so attributes are gated by the blocked roots only.
        if (blocked(el)) return;
        for (const attr of ATTRS) applyAttr(el, attr);
    }

    function applyTo(root) {
        if (!root) return;
        if (root.nodeType === 3) {
            const parent = root.parentElement;
            if (parent && !excluded(parent)) applyTextNode(root);
            return;
        }
        if (!root.querySelectorAll) return;  // comment, attribute, detached text
        if (root.nodeType === 1 && blocked(root)) return;
        const doc = root.ownerDocument || root;
        const walker = doc.createTreeWalker(root, SHOW_TEXT);
        for (let node = walker.nextNode(); node; node = walker.nextNode()) {
            const parent = node.parentElement;
            if (!parent || parent.tagName === 'SCRIPT' || parent.tagName === 'STYLE') continue;
            if (excluded(parent)) continue;
            applyTextNode(node);
        }
        if (root.nodeType === 1) applyElement(root);
        root.querySelectorAll('[placeholder],[title],[aria-label]').forEach(applyElement);
    }

    function restore(root) {
        if (!root || !root.querySelectorAll) return;
        const doc = root.ownerDocument || root;
        const walker = doc.createTreeWalker(root, SHOW_TEXT);
        for (let node = walker.nextNode(); node; node = walker.nextNode()) {
            if (node.__ouroSrc === undefined) continue;
            if (node.nodeValue === node.__ouroOut) node.nodeValue = node.__ouroSrc;
            delete node.__ouroSrc;
            delete node.__ouroOut;
        }
        for (const attr of ATTRS) {
            const key = ATTR_DATA_KEYS[attr];
            root.querySelectorAll(`[data-${key.replace(/[A-Z]/g, (c) => '-' + c.toLowerCase())}]`).forEach((el) => {
                // Same guard as the text path: only take back an attribute that still
                // holds OUR translation, never one the app rewrote since.
                const saved = el.dataset[key];
                if (el.getAttribute(attr) === translateString(saved, dict, patterns)) el.setAttribute(attr, saved);
                delete el.dataset[key];
            });
        }
    }

    function flush() {
        const batch = pending;
        pending = null;
        if (!batch) return;
        for (const node of batch) {
            if (node.isConnected === false) continue;
            applyTo(node);
        }
    }

    function queue(records) {
        if (!pending) {
            pending = new Set();
            requestAnimationFrame(flush);
        }
        for (const record of records) {
            if (record.type === 'childList') record.addedNodes.forEach((node) => pending.add(node));
            else pending.add(record.target);
        }
    }

    // ponytail: one document-wide observer with the hot subtrees excluded; scope it to
    // the sidebar/settings roots instead if a busy page ever measures as sluggish.
    function observe(root) {
        if (observer || !root) return;
        observer = new MutationObserver(queue);
        observer.observe(root, {
            childList: true, subtree: true, characterData: true,
            attributes: true, attributeFilter: ATTRS,
        });
    }

    function disconnect() {
        if (observer) observer.disconnect();
        observer = null;
        pending = null;
    }

    return { applyTo, observe, disconnect, restore };
}

export const LANGUAGE_STORAGE_KEY = 'ouro.language';

let translator = null;
let currentLang = 'en';
let months = null;
// Every switch runs on ONE chain: two concurrent setLanguage('ru') calls cannot
// each build a translator and leave the loser's MutationObserver attached and
// unreachable (the UI would then never come back to English).
let switching = Promise.resolve();

/**
 * Localized month names while a dictionary is active, else null (English source).
 * The dictionary stays loaded after a switch back, so the active language — not
 * the presence of the module — decides.
 */
export function monthNames() {
    return currentLang === 'ru' ? months : null;
}

/** Last choice this browser made; the server preference is the SSOT when it answers. */
export function storedLanguage() {
    try {
        return localStorage.getItem(LANGUAGE_STORAGE_KEY) === 'ru' ? 'ru' : 'en';
    } catch {
        return 'en';
    }
}

export function currentLanguage() {
    return currentLang;
}

/** Switch the whole document to `lang`; anything but "ru" means the English source. */
export function setLanguage(lang) {
    const run = () => applyLanguage(lang);
    switching = switching.then(run, run);
    return switching;
}

async function applyLanguage(lang) {
    const next = lang === 'ru' ? 'ru' : 'en';
    if (next === 'ru') {
        if (!translator) {
            const mod = await import('../i18n/ru.js');
            translator = createTranslator({ dict: mod.ru, patterns: mod.ruPatterns });
            // Calendar data for the few places that format dates in JS (chat_activity).
            months = mod.ruMonths;
        }
        translator.applyTo(document.body);
        translator.observe(document.body);
    } else if (translator) {
        translator.disconnect();
        translator.restore(document.body);
    }
    currentLang = next;
    document.documentElement.lang = next;
    try {
        localStorage.setItem(LANGUAGE_STORAGE_KEY, next);
    } catch {
        // Private mode / blocked storage: the server preference is still the SSOT.
    }
    window.dispatchEvent(new CustomEvent('ouro:language-changed', { detail: { language: next } }));
    return next;
}
