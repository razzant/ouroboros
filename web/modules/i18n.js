/** The interface language in the browser: overlay, catalog seam, composed strings, misses.
 *
 * The SPA is authored in English at every call site and stays that way. An install that
 * chose another language (the `OUROBOROS_UI_LANGUAGE` setting; `GET /api/ui/i18n`) paints
 * from a translation memory the install generated or imported — never from a dictionary in
 * the repository, so `git merge managed/ouroboros` sees no translation diff and no language
 * is enumerated in code.
 *
 * Three seams, one memory:
 *
 * - **Static chrome** (navigation, headers, tabs, labels, help copy, placeholders, titles,
 *   `aria-label`, `<option>` labels) is rewritten after the fact by a DOM overlay keyed by the
 *   rendered English string, with an optional DOM scope for the same word in two places
 *   ("Light" the theme vs "Light" the review lane). Unknown strings stay English and are
 *   reported as misses for the generator.
 * - **Host sentences** minted by closed code→sentence tables (task headline words, cause
 *   sentences, question status, notification titles) go through `tr(code, english)` at the
 *   place the table is read, so the chat transcript is never walked and model prose is
 *   never touched; the code survives an upstream reword of the English.
 * - **Composed strings** (`3 notes`, `New task in {name}`) go through `fmt(key, params,
 *   template)` at the producer; the memory's stored plural map chooses the form, then `Intl.PluralRules`.
 *
 * Excluded from the overlay by construction: every chat transcript (model prose and the
 * host rows, which translate through `tr` instead), logs, code, inputs, owner-supplied
 * names, menus portaled to `document.body` from excluded roots, and authored content
 * portaled there (`data-i18n-authored`).
 */
import { apiClient } from './api_client.js';

// NodeFilter.SHOW_TEXT, spelled numerically: the hermetic no-undef walker
// (web/tests/no_undef.test.js) does not carry NodeFilter in its browser-global list.
const SHOW_TEXT = 4;
const TEXT_NODE = 3;
const ELEMENT_NODE = 1;
const ATTRS = ['placeholder', 'title', 'aria-label'];
const INLINE_TAGS = new Set(['CODE', 'STRONG', 'EM', 'B', 'I', 'KBD', 'A', 'BR', 'ABBR', 'SMALL', 'SUP', 'SUB']);
const IMMUTABLE_INLINE = new Set(['CODE', 'KBD']);
const MISS_FLUSH_MS = 1500;
const MISS_BATCH = 200;

export const CODE_PREFIX = 'code:';
export const SCOPE_SEPARATOR = '\u001f';
export const LANGUAGE_STORAGE_KEY = 'ouro.language';

/** Content whose text is data, code, or user input — never translated. */
export const EXCLUDE_SELECTOR = [
    '.chat-bubble .message', '.chat-bubble .msg-time', '[data-chat-markdown-enhanced]',
    '.ui-rich-content', '.md-code-block', '.md-mermaid-stage', '.md-table-wrap', 'pre', 'code',
    '.log-raw', '.log-body', '.log-pill', '.log-ts', '.log-repeat', '.log-task-summary',
    '.log-task-timeline', '.files-editor', '.files-preview-content', '.files-preview-path',
    '.files-preview-meta', '.chat-live-line-body', '.chat-live-line-time',
    '.chat-live-line-repeat', '.chat-live-meta', 'input', 'textarea',
    '[contenteditable="true"]', '[data-i18n-fmt]', '[data-i18n-skip]',
].join(',');

/** Subtrees the overlay never enters: every chat transcript (Main's `#chat-messages` and
 *  the Project instances, whose ids are namespaced but carry the mirror class), live cards,
 *  logs, file views, the owner's empty-Main copy, the menus chat portals to <body>, and
 *  `[data-i18n-authored]`: authored content shown outside a transcript (a delivered
 *  document's text, name and size line), whose text and attributes are never chrome. */
export const SKIP_ROOTS = [
    '#chat-messages', '.chat-messages', '.chat-live-card', '#log-entries', '.log-entries',
    '.files-preview-content', '.files-editor', '.chat-empty-welcome', '.chat-photo-menu',
    '.task-control-menu', '#reconnect-overlay', '[data-i18n-authored]',
].join(',');

/** Roots where a BARE owner-supplied name lands (a project title, a path segment). Gated
 *  whole, like SKIP_ROOTS: the name reaches `.nav-project-row` as the `title` attribute as
 *  well as its label's text. Composed strings that merely embed a name are a `fmt` job. */
export const USER_CONTENT = [
    '.nav-project-row', '.nav-project-kebab', '#project-panel-title', '.chat-live-project-name',
    '.files-entry-name', '.files-crumb',
].join(',');

// ---------------------------------------------------------------------------
// state
// ---------------------------------------------------------------------------

const state = {
    language: '', english: true, revision: 0, entries: Object.create(null), scopes: [],
    plural: null, pluralMap: null, profile: null, stats: null, languages: [], payload: null,
    values: new Set(),
};
// The SPA's boot read of the memory, so a control that mounts before it answers waits for it
// instead of reading the gateway a second time.
let bootRead = null;
export function markBootRead(promise) { bootRead = Promise.resolve(promise).catch(() => null); return bootRead; }
export function pendingBootRead() { return bootRead; }
let translator = null;
// Every switch runs on ONE chain: two concurrent setLanguage calls cannot each build a
// translator and leave the loser's MutationObserver attached and unreachable.
let switching = Promise.resolve();

export function englishTag(tag) {
    const text = String(tag || '').trim();
    return !text || /^en(-|$)/i.test(text);
}

export function currentLanguage() { return state.language; }
export function isEnglish() { return state.english; }
export function dictionaryRevision() { return state.revision; }
export function languageProfile() { return state.profile; }
export function languageStats() { return state.stats; }
export function knownLanguages() { return state.languages.slice(); }
/** The last gateway payload applied (GET /api/ui/i18n or a language POST), or null before boot. */
export function currentPayload() { return state.payload; }

/** `ltr` | `rtl` when the engine knows the tag's script direction, else `` (unknown). */
export function localeDirection(tag) {
    try {
        const locale = new Intl.Locale(String(tag || ''));
        // No likely script: an invented or unknown language, for which Intl would only guess `ltr`.
        if (!locale.maximize().script) return '';
        const info = typeof locale.getTextInfo === 'function' ? locale.getTextInfo() : locale.textInfo;
        return info && (info.direction === 'rtl' || info.direction === 'ltr') ? info.direction : '';
    } catch { return ''; }
}

/** Whether this engine has CLDR data for `tag`. An invented or unknown language has none, and
 *  Intl would otherwise answer silently with the default locale's rules. */
export function engineKnowsLocale(tag) {
    try { return Intl.PluralRules.supportedLocalesOf([String(tag || '')]).length > 0; } catch { return false; }
}

function makePluralRules(tag) {
    if (!tag || typeof Intl === 'undefined' || typeof Intl.PluralRules !== 'function' || !engineKnowsLocale(tag)) return null;
    try { return new Intl.PluralRules(tag); } catch { return null; }
}

/** `{map: {"0": "other", …}, period, categories}` for a tag, as the memory stores it so Python
 *  (which has no Intl) can select the same plural form. Null when the engine does not know the tag. */
export function pluralSelectMap(tag) {
    if (!engineKnowsLocale(tag)) return null;
    const rules = makePluralRules(tag);
    if (!rules) return null;
    const map = {};
    for (let n = 0; n <= 100; n += 1) map[String(n)] = rules.select(n);
    let periodic = true;
    for (let n = 101; n <= 200 && periodic; n += 1) if (rules.select(n) !== map[String(n % 100)]) periodic = false;
    let categories = ['other'];
    try { categories = rules.resolvedOptions().pluralCategories || categories; } catch { /* keep */ }
    return { map, period: periodic ? 100 : null, categories };
}

// ---------------------------------------------------------------------------
// lookup
// ---------------------------------------------------------------------------

function pluralCategory(n) {
    if (typeof n !== 'number' || !Number.isFinite(n)) return 'other';
    // The memory's own rules first — the same map Python selects with (an imported pack may
    // carry its own); then this engine's CLDR rules; `other` for a language nobody has rules for.
    const stored = state.pluralMap;
    const value = Math.abs(n);
    if (stored && stored.map && typeof stored.map === 'object' && Number.isInteger(value)) {
        // The exact entry first, then the periodic one — the order Python selects in.
        const period = Number(stored.period) || 0;
        const exact = stored.map[String(value)];
        const picked = typeof exact === 'string' ? exact : (period ? stored.map[String(value % period)] : undefined);
        if (typeof picked === 'string') return picked;
    }
    if (!state.plural) return 'other';
    try { return state.plural.select(value); } catch { return 'other'; }
}

/** The text an entry yields: `text`, or the plural form `n` selects (then `other`, `many`, first). */
export function entryText(entry, n) {
    if (!entry || typeof entry !== 'object') return null;
    if (typeof entry.text === 'string') return entry.text;
    const forms = entry.forms;
    if (!forms || typeof forms !== 'object') return null;
    const category = pluralCategory(n);
    const pick = forms[category] ?? forms.other ?? forms.many ?? Object.values(forms)[0];
    return typeof pick === 'string' ? pick : null;
}

function matchesScope(element, scope) {
    return Boolean(element && typeof element.closest === 'function' && element.closest(scope));
}

const NUMBER_RE = /^([\s\S]*?)(-?\d+(?:[.,]\d+)?)([\s\S]*)$/;

/**
 * Translate one rendered string, or return it unchanged. Leading/trailing whitespace is
 * preserved around the core; `element` resolves DOM-scoped keys. A string with exactly one
 * number tries its `{n}` template and selects the plural form for that number.
 */
export function translateString(text, element = null, entries = state.entries) {
    return lookupString(text, element, entries).text;
}

// A key is the text as the reader sees it: runs of ASCII whitespace (a template literal's line
// breaks and indentation) collapse to one space, so the same sentence has one key in every
// source layout and in every client. The memory applies the same rule to reported keys.
const KEY_WHITESPACE_RE = /[ \t\n\r\f\v]+/g;
export function keyText(text) { return String(text ?? '').replace(KEY_WHITESPACE_RE, ' ').trim(); }

/** The single-number template a rendered string reports as its miss ("12.3 KB" → "{n} KB"),
 *  or the string itself. One memory entry then covers every value instead of one per value. */
export function missKeyFor(text) {
    const core = keyText(text);
    const numbered = NUMBER_RE.exec(core);
    if (numbered && !/\d/.test(numbered[1] + numbered[3]) && /\p{L}/u.test(numbered[1] + numbered[3])) {
        return `${numbered[1]}{n}${numbered[3]}`;
    }
    return core;
}

/** `{text, found}`: the translation (or the input) and whether the memory HAD an entry — an
 *  entry whose text equals its source ("Ouroboros") is found, not missing. */
export function lookupString(text, element = null, entries = state.entries) {
    if (typeof text !== 'string') return { text, found: false };
    const parts = /^(\s*)([\s\S]*?)(\s*)$/.exec(text);
    const core = keyText(parts[2]);
    if (!core) return { text, found: false };
    if (element && state.scopes.length) {
        for (const scope of state.scopes) {
            if (!matchesScope(element, scope)) continue;
            const scoped = entryText(entries[core + SCOPE_SEPARATOR + scope]);
            if (typeof scoped === 'string') return { text: parts[1] + scoped + parts[3], found: true };
        }
    }
    const exact = entryText(entries[core]);
    if (typeof exact === 'string') return { text: parts[1] + exact + parts[3], found: true };
    const numbered = NUMBER_RE.exec(core);
    if (numbered && !/\d/.test(numbered[1] + numbered[3])) {
        const key = `${numbered[1]}{n}${numbered[3]}`;
        const value = Number(numbered[2].replace(',', '.'));
        const templated = entryText(entries[key], value);
        if (typeof templated === 'string') return { text: parts[1] + templated.split('{n}').join(numbered[2]) + parts[3], found: true };
    }
    return { text, found: false };
}

// ---------------------------------------------------------------------------
// misses → generator
// ---------------------------------------------------------------------------

const misses = new Map();
const reportedAtRevision = new Set();
let missTimer = null;
let missTransport = (payload) => apiClient.reportI18nMissing(payload);

/** Test seam: replace the transport (default: POST /api/ui/i18n/missing). */
export function setMissTransport(fn) { missTransport = typeof fn === 'function' ? fn : missTransport; }

// The memory's own bound on a key (ouroboros/i18n_memory.py MAX_KEY_CHARS): a Settings help
// paragraph of several sentences is an ordinary key; the gateway refuses anything longer.
const MAX_KEY_CHARS = 2000;

function looksVolatile(text) {
    const value = String(text || '').trim();
    if (!value || value.length > MAX_KEY_CHARS) return true;
    if (/^[\W\d_]*$/.test(value) || /^(?:https?:\/\/|\/|~\/|[A-Za-z]:\\)/.test(value) || /^\d{4}-\d{2}-\d{2}[T ]/.test(value)) return true;
    let letters = 0;
    for (const ch of value) if (/\p{L}/u.test(ch)) letters += 1;
    return letters < 2;
}

function noteMiss(key, context = {}) {
    if (state.english || !key || reportedAtRevision.has(key) || misses.has(key)) return;
    const source = key.startsWith(CODE_PREFIX) ? key : key.split(SCOPE_SEPARATOR)[0];
    if (!key.startsWith(CODE_PREFIX) && (looksVolatile(source) || state.values.has(keyText(source)))) return;  // already a translation
    misses.set(key, { key, context: { ...context, page: pageContext() } });
    reportedAtRevision.add(key);
    if (misses.size >= MISS_BATCH) flushMisses();
    else if (!missTimer) missTimer = setTimeout(flushMisses, MISS_FLUSH_MS);
}

function pageContext() {
    if (typeof document === 'undefined') return '';
    const active = document.querySelector?.('.page.active, .app-page.active, [data-page].active');
    return active?.id || (typeof location !== 'undefined' ? String(location.hash || '').replace(/^#/, '') : '');
}

/** Send the collected misses now (also called by the timer). Never throws. */
export function flushMisses() {
    if (missTimer) { clearTimeout(missTimer); missTimer = null; }
    if (!misses.size || state.english) { misses.clear(); return Promise.resolve(null); }
    const items = Array.from(misses.values()).slice(0, MISS_BATCH);
    for (const item of items) misses.delete(item.key);
    let result;
    try { result = missTransport({ language: state.language, items }); } catch { result = null; }
    return Promise.resolve(result).catch(() => null);
}

export function pendingMisses() { return Array.from(misses.keys()); }

// ---------------------------------------------------------------------------
// tr / fmt — the seams host sentences and composed strings use
// ---------------------------------------------------------------------------

/** A catalog sentence by its stable code; `english` is the table's own text and the fallback. */
export function tr(code, english = '') {
    if (state.english) return english;
    const entry = state.entries[CODE_PREFIX + code];
    const text = entryText(entry);
    if (typeof text === 'string') {
        // A generated entry remembers the English it translated; a reworded source is a stale
        // miss and the English shows until the generator catches up. An owner's or an import's
        // pin keeps rendering whatever the English says: a pin survives an upstream reword.
        if (entry.provenance === 'generated' && typeof entry.source === 'string' && keyText(entry.source) !== keyText(english)) {
            noteMiss(CODE_PREFIX + code, { source: english, stale: true });
            return english;
        }
        return text;
    }
    noteMiss(CODE_PREFIX + code, { source: english });
    return english;
}

/**
 * A sentence the host composed in English from its own closed tables and this client merely
 * shows (a routing refusal, a Continue cause): translated when the memory knows the exact
 * sentence, otherwise shown as it is and reported, so the next one of its kind reads translated.
 * Never for model prose or owner-supplied names.
 */
export function tx(text) {
    const source = keyText(text);
    if (!source || state.english) return source;
    const found = entryText(state.entries[source]);
    if (typeof found === 'string') return found;
    noteMiss(source, { role: 'host-text' });
    return source;
}

const PLACEHOLDER_RE = /\{([A-Za-z_][A-Za-z0-9_]*)\}/g;

function fill(template, params) {
    return String(template).replace(PLACEHOLDER_RE, (match, name) => (name in params ? String(params[name]) : match));
}

/** A composed string: the translated template (plural form by `params.n`) or `template`,
 *  with `{name}` placeholders filled. The key IS the English template. */
export function fmt(key, params = {}, template = key) {
    let chosen = template;
    if (!state.english) {
        const text = entryText(state.entries[key], params.n);
        if (typeof text === 'string') chosen = text;
        else noteMiss(key, { source: template, params: Object.keys(params) });
    }
    return fill(chosen, params);
}

/** `fmt` written into an element, marked so the overlay leaves the rendered result alone. */
export function fmtInto(element, key, params = {}, template = key) {
    const text = fmt(key, params, template);
    if (element) {
        element.textContent = text;
        if (element.dataset) element.dataset.i18nFmt = key;
    }
    return text;
}

// ---------------------------------------------------------------------------
// the DOM overlay
// ---------------------------------------------------------------------------

function isInlineComposite(element) {
    if (!element || element.nodeType !== ELEMENT_NODE || !element.childNodes) return false;
    let textCount = 0;
    let inlineCount = 0;
    for (const child of element.childNodes) {
        if (child.nodeType === TEXT_NODE) {
            if (String(child.nodeValue).trim()) textCount += 1;
        } else if (child.nodeType === ELEMENT_NODE) {
            if (!INLINE_TAGS.has(child.tagName)) return false;
            if (child.tagName !== 'BR' && child.childNodes && [...child.childNodes].some((n) => n.nodeType !== TEXT_NODE)) return false;
            inlineCount += 1;
        } else {
            return false;
        }
    }
    return textCount > 0 && inlineCount > 0;
}

function inlineKey(element) {
    let key = '';
    const slots = [];
    for (const child of element.childNodes) {
        if (child.nodeType === TEXT_NODE) key += child.nodeValue;
        else {
            slots.push(child);
            const inner = child.tagName === 'BR' ? '' : [...(child.childNodes || [])].map((n) => n.nodeValue || '').join('');
            key += `<${slots.length}>${inner}</${slots.length}>`;
        }
    }
    return { key: keyText(key), slots };
}

function slotTexts(slots) {
    return slots.map((slot) => (slot.childNodes && slot.childNodes.length === 1 && slot.childNodes[0].nodeType === TEXT_NODE
        ? slot.childNodes[0].nodeValue : null));
}

function restoreSlotTexts(slots, texts) {
    slots.forEach((slot, index) => {
        if (texts && typeof texts[index] === 'string' && slot.childNodes && slot.childNodes[0]) slot.childNodes[0].nodeValue = texts[index];
    });
}

function rebuildInline(element, translated, slots, doc) {
    const nodes = [];
    const re = /<(\d+)>([\s\S]*?)<\/\1>/g;
    let last = 0;
    let match;
    while ((match = re.exec(translated)) !== null) {
        if (match.index > last) nodes.push(doc.createTextNode(translated.slice(last, match.index)));
        const slot = slots[Number(match[1]) - 1];
        if (slot) {
            const inner = match[2];
            if (!IMMUTABLE_INLINE.has(slot.tagName) && slot.tagName !== 'BR' && inner && slot.childNodes
                && slot.childNodes.length === 1 && slot.childNodes[0].nodeType === TEXT_NODE) {
                slot.childNodes[0].nodeValue = inner;
            }
            nodes.push(slot);
        } else {
            nodes.push(doc.createTextNode(match[0]));
        }
        last = re.lastIndex;
    }
    if (last < translated.length) nodes.push(doc.createTextNode(translated.slice(last)));
    element.replaceChildren(...nodes);
}

/**
 * DOM side of the overlay. `applyTo` is idempotent: the English original is stashed on each
 * rewritten node/attribute/composite, so a second pass (a dictionary update, a switch back)
 * starts from the source string, never from a translation.
 */
export function createTranslator({
    excludeSelector = EXCLUDE_SELECTOR, skipRoots = SKIP_ROOTS, userContent = USER_CONTENT,
    requestFrame = (fn) => (typeof requestAnimationFrame === 'function' ? requestAnimationFrame(fn) : setTimeout(fn, 16)),
} = {}) {
    let observer = null;
    let pending = null;
    const roots = [skipRoots, userContent].filter(Boolean).join(',');
    const blocked = (el) => Boolean(roots && el && el.closest && el.closest(roots));
    const excluded = (el) => blocked(el) || Boolean(excludeSelector && el && el.closest && el.closest(excludeSelector))
        || Boolean(el && el.dataset && el.dataset.i18nFmt !== undefined);

    function applyTextNode(node) {
        const current = node.nodeValue;
        // Our own last output still in place → retranslate from the stored English;
        // anything else means the app rewrote the node, so that value is the source.
        const source = node.__ouroOut !== undefined && current === node.__ouroOut ? node.__ouroSrc : current;
        const { text: out, found } = lookupString(source, node.parentElement);
        if (out === source) {
            if (!found && String(source).trim()) noteMiss(missKeyFor(source), { role: roleOf(node.parentElement) });
            delete node.__ouroSrc;
            delete node.__ouroOut;
            if (current !== source) node.nodeValue = source;
            return;
        }
        node.__ouroSrc = source;
        node.__ouroOut = out;
        if (current !== out) node.nodeValue = out;
    }

    function roleOf(el) {
        if (!el) return '';
        const tag = String(el.tagName || '').toLowerCase();
        if (tag === 'button' || tag === 'option' || tag === 'label' || tag === 'h1' || tag === 'h2' || tag === 'h3') return tag;
        return el.getAttribute?.('role') || tag;
    }

    function applyAttr(el, attr) {
        const current = el.getAttribute(attr);
        if (current === null) return;
        const stash = el.__ouroAttrs?.[attr];
        const source = stash && current === stash.out ? stash.src : current;
        const { text: out, found } = lookupString(source, el);
        if (out === source) {
            if (el.__ouroAttrs) delete el.__ouroAttrs[attr];
            if (!found && String(source).trim()) noteMiss(missKeyFor(source), { role: attr });
            if (current !== source) el.setAttribute(attr, source);
            return;
        }
        el.__ouroAttrs = el.__ouroAttrs || {};
        el.__ouroAttrs[attr] = { src: source, out };
        if (current !== out) el.setAttribute(attr, out);
    }

    function applyElementAttrs(el) {
        // Inputs are excluded as content but their placeholder/title/aria-label are chrome,
        // so attributes are gated by the blocked roots only.
        if (blocked(el)) return;
        for (const attr of ATTRS) applyAttr(el, attr);
    }

    function applyInline(el) {
        const doc = el.ownerDocument || (typeof document !== 'undefined' ? document : null);
        if (!doc || typeof doc.createTextNode !== 'function' || typeof el.replaceChildren !== 'function') return;
        const { key, slots } = inlineKey(el);
        const prior = el.__ouroInline;
        // Our own output still in place → the source is the stashed English. Otherwise the app
        // rewrote the composite: what stands now IS the source, and the stash (the nodes and slots
        // of the sentence that was there before) is dropped — never put back over the app's update.
        const stash = prior && key === prior.out ? prior : null;
        if (prior && !stash) delete el.__ouroInline;
        const source = stash ? stash.src : key;
        const { text: translated, found } = lookupString(source, el);
        // Idempotent: our output stands and the translation has not changed → not one DOM write
        // (a write here would wake the observer, which would bring the node back here, forever).
        if (stash && translated === stash.text) return;
        if (stash) restoreSlotTexts(stash.slots, stash.slotTexts);  // the slots' English back before any rebuild
        if (translated === source) {
            if (stash) {
                el.replaceChildren(...stash.nodes);
                delete el.__ouroInline;
            } else if (!found && source) {
                noteMiss(source, { role: 'inline', tags: slots.length });
            }
            return;
        }
        const originals = stash ? stash.nodes : [...el.childNodes];
        const originalSlots = stash ? stash.slots : slots;
        const originalTexts = stash ? stash.slotTexts : slotTexts(slots);
        if (stash) el.replaceChildren(...stash.nodes);
        rebuildInline(el, translated, originalSlots, doc);
        el.__ouroInline = { src: source, out: inlineKey(el).key, text: translated, nodes: originals, slots: originalSlots, slotTexts: originalTexts };
    }

    /** The composite a text node belongs to: its parent, or — for the text of an inline
     *  element such as <strong> — the grandparent that holds the sentence. */
    function compositeOf(node) {
        const parent = node.parentElement;
        if (!parent) return null;
        if (parent.__ouroInline || isInlineComposite(parent)) return parent;
        const holder = INLINE_TAGS.has(parent.tagName) ? parent.parentElement : null;
        if (holder && !excluded(holder) && (holder.__ouroInline || isInlineComposite(holder))) return holder;
        return null;
    }

    function applyTo(root) {
        if (!root) return;
        if (root.nodeType === TEXT_NODE) {
            const parent = root.parentElement;
            if (!parent || excluded(parent)) return;
            const composite = compositeOf(root);
            if (composite) applyInline(composite);
            else applyTextNode(root);
            return;
        }
        if (!root.querySelectorAll) return;  // comment, attribute, detached text
        if (root.nodeType === ELEMENT_NODE && blocked(root)) return;
        const doc = root.ownerDocument || root;
        const composites = new Set();
        const walker = doc.createTreeWalker(root, SHOW_TEXT);
        for (let node = walker.nextNode(); node; node = walker.nextNode()) {
            const parent = node.parentElement;
            if (!parent || parent.tagName === 'SCRIPT' || parent.tagName === 'STYLE') continue;
            if (excluded(parent)) continue;
            const composite = compositeOf(node);
            if (composite) { composites.add(composite); continue; }
            applyTextNode(node);
        }
        for (const el of composites) applyInline(el);
        if (root.nodeType === ELEMENT_NODE) applyElementAttrs(root);
        root.querySelectorAll('[placeholder],[title],[aria-label]').forEach(applyElementAttrs);
    }

    function restore(root) {
        if (!root || !root.querySelectorAll) return;
        const doc = root.ownerDocument || root;
        const walker = doc.createTreeWalker(root, SHOW_TEXT);
        const texts = [];
        for (let node = walker.nextNode(); node; node = walker.nextNode()) texts.push(node);
        for (const node of texts) {
            if (node.__ouroSrc === undefined) continue;
            if (node.nodeValue === node.__ouroOut) node.nodeValue = node.__ouroSrc;
            delete node.__ouroSrc;
            delete node.__ouroOut;
        }
        const elements = [root, ...root.querySelectorAll('*')];
        for (const el of elements) {
            if (el.__ouroInline) {
                if (inlineKey(el).key === el.__ouroInline.out) {
                    restoreSlotTexts(el.__ouroInline.slots, el.__ouroInline.slotTexts);
                    el.replaceChildren(...el.__ouroInline.nodes);
                }
                delete el.__ouroInline;
            }
            if (el.__ouroAttrs) {
                for (const [attr, stash] of Object.entries(el.__ouroAttrs)) {
                    // Only take back an attribute that still holds OUR translation, never one
                    // the app rewrote since.
                    if (el.getAttribute(attr) === stash.out) el.setAttribute(attr, stash.src);
                }
                delete el.__ouroAttrs;
            }
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
            requestFrame(flush);
        }
        for (const record of records) {
            if (record.type === 'childList') record.addedNodes.forEach((node) => pending.add(node));
            else pending.add(record.target);
        }
    }

    function observe(root) {
        if (observer || !root || typeof MutationObserver !== 'function') return;
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

// ---------------------------------------------------------------------------
// language lifecycle
// ---------------------------------------------------------------------------

function collectScopes(entries) {
    const scopes = new Set();
    for (const key of Object.keys(entries)) {
        const at = key.indexOf(SCOPE_SEPARATOR);
        if (at >= 0) scopes.add(key.slice(at + 1));
    }
    return Array.from(scopes);
}

/**
 * Take a `GET /api/ui/i18n` (or language POST) payload into effect: entries, plural rules,
 * `<html lang>`/`dir`, the overlay on or off. Synchronous; safe without a document.
 */
export function applyPayload(payload) {
    const data = payload && typeof payload === 'object' ? payload : {};
    state.language = String(data.language || '');
    state.english = typeof data.english === 'boolean' ? data.english : englishTag(state.language);
    state.revision = Number(data.revision) || 0;
    state.entries = Object.assign(Object.create(null), data.entries && typeof data.entries === 'object' ? data.entries : {});
    state.scopes = collectScopes(state.entries);
    // Every translated text the memory holds: a producer that already read its label through
    // `tr` may sit where the overlay walks, and what it wrote is a translation, not a new key.
    state.values = new Set();
    for (const entry of Object.values(state.entries)) {
        if (entry && typeof entry.text === 'string') state.values.add(keyText(entry.text));
        else if (entry && entry.forms && typeof entry.forms === 'object') for (const form of Object.values(entry.forms)) if (typeof form === 'string') state.values.add(keyText(form));
    }
    const stored = data.plural_select;
    state.pluralMap = !state.english && stored && typeof stored === 'object' && stored.map && typeof stored.map === 'object' ? stored : null;
    state.plural = state.english ? null : makePluralRules(state.language);
    state.profile = data.profile && typeof data.profile === 'object' ? data.profile : null;
    state.stats = data.stats && typeof data.stats === 'object' ? data.stats : null;
    state.languages = Array.isArray(data.languages) ? data.languages : [];
    state.payload = payload && typeof payload === 'object' ? payload : null;
    reportedAtRevision.clear();
    misses.clear();
    try {
        localStorage.setItem(LANGUAGE_STORAGE_KEY, state.english ? '' : state.language);
    } catch {
        // Private mode / blocked storage: the gateway is the SSOT anyway.
    }
    if (typeof document === 'undefined' || !document.body) return state;
    if (state.english) {
        if (translator) {
            translator.disconnect();
            translator.restore(document.body);
        }
        document.documentElement.lang = 'en';
        document.documentElement.removeAttribute('dir');
    } else {
        translator = translator || createTranslator();
        translator.applyTo(document.body);
        translator.observe(document.body);
        document.documentElement.lang = state.language;
        if (state.profile?.direction === 'rtl') document.documentElement.dir = 'rtl';
        else document.documentElement.removeAttribute('dir');
    }
    if (typeof window !== 'undefined' && typeof window.dispatchEvent === 'function' && typeof CustomEvent === 'function') {
        window.dispatchEvent(new CustomEvent('ouro:language-changed', {
            detail: { language: state.language, english: state.english, revision: state.revision },
        }));
    }
    return state;
}

/** Last tag this browser painted; the gateway is the SSOT when it answers. */
export function storedLanguage() {
    try {
        return localStorage.getItem(LANGUAGE_STORAGE_KEY) || '';
    } catch {
        return '';
    }
}

async function fetchPayload(fallbackTag) {
    try {
        return await apiClient.uiI18n();
    } catch {
        // Offline or pre-gateway: paint nothing new, remember the tag so the next read wins.
        return { language: fallbackTag || '', english: englishTag(fallbackTag), entries: {}, revision: 0 };
    }
}

/**
 * Switch the document to `tag`. With `payload` (a POST/GET body) nothing is fetched;
 * without it the gateway is read. Serialized: concurrent calls apply in order.
 */
export function setLanguage(tag, payload = null) {
    const run = async () => applyPayload(payload || await fetchPayload(tag));
    switching = switching.then(run, run);
    return switching;
}

/** Re-read the gateway (after a `ui_language_changed` / `ui_i18n_updated` frame, on reconnect). */
export function refreshDictionary() {
    return setLanguage(state.language, null);
}

/** Boot: paint from the last known tag immediately, then let the gateway decide. */
export function bootLanguage() {
    return setLanguage(storedLanguage(), null);
}
