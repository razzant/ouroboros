/** Rich, sanitized markdown rendering for assistant and system chat messages. */

import { safeExternalUrl } from './utils.js';
import { tr } from './i18n.js';
import { applyChartTheme, onThemeChange } from './theme_palette.js';

const CHART_TYPES = new Set([
    'bar', 'line', 'pie', 'doughnut', 'polarArea', 'radar', 'scatter', 'bubble',
]);
const MAX_CHART_DATASETS = 24;
const MAX_CHART_POINTS = 500;
const MAX_RICH_BLOCK_SOURCE_LENGTH = 32768;
const MERMAID_SCRIPT_ID = 'chat-mermaid-library';
const ROOT_STATE = new WeakMap();
const TABLE_BINDINGS = new WeakMap();
const CHART_AUTHORED = new WeakMap();
const CHART_THEMED = new WeakMap();
const writeDirectly = (mutate) => mutate();

// One parser per line-break reading: chat keeps every newline (`breaks: true`),
// a delivered document reads a single newline as a soft break.
const markdownParsers = new Map();
let mermaidLoadPromise = null;
let mermaidInitialized = false;

function escapeText(value) {
    return String(value ?? '')
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;');
}

/**
 * Parser input whose raw HTML stays literal. `<` is the only character that
 * opens HTML in marked's grammar, so it becomes `lessThan`: a decimal reference
 * the message does not already contain. Prose therefore still reads `<`, and
 * code maps each one back to the author's byte. `&` stays the author's: marked
 * decodes one entity layer in prose (the legacy stored-entity reading) and keeps
 * fenced and inline code literal. An entity never creates Markdown structure.
 * The one `<` kept opens an autolink to an address a link may carry
 * (`<https://…>`, `<mailto:…>`): it opens no HTML, and without it marked's bare
 * URL rule would take the closing `>` into the destination.
 * An escaped `\<` (backslashes paired left to right, as marked pairs them)
 * becomes `escapedLessThan`, a second unused reference that takes the backslash
 * too: a `\` left before a reference escapes its `&`, and prose would show the
 * reference. Prose reads `<` there, as marked reads `\<`; code maps it back.
 */
export function prepareMarkdownSource(text) {
    const raw = String(text ?? '');
    const unused = (reference) => {
        while (raw.includes(reference)) reference = reference.replace('&#', '&#0');
        return reference;
    };
    const lessThan = unused('&#60;');
    const escapedLessThan = unused(lessThan.replace('&#', '&#0'));
    const source = raw.replace(/(\\?<)(?!(?:https?|mailto):[^\s<>]*>)|\\[\s\S]/gi, (match, opener) => (
        opener === '<' ? lessThan : opener ? escapedLessThan : match));
    return { source, lessThan, escapedLessThan };
}

// Attribute text in marked's own manner: an entity (including `lessThan`) stays
// an entity for the browser to read, everything else that could end the value is
// escaped.
function attributeText(value) {
    return String(value ?? '').replace(/&(?!#?\w+;)/g, '&amp;').replace(/"/g, '&quot;').replace(/>/g, '&gt;');
}

// A forbidden image keeps its words and its address (#1368). Nothing loads: the
// renderer writes an inert placeholder (a link would nest inside a linked image
// and break it), and the link pass decides what it becomes. The address is
// encoded exactly as marked encodes a link's.
function renderImageReference({ href, title, tokens, text }) {
    const alt = tokens ? this.parser.parseInline(tokens) : attributeText(text);
    let address = '';
    try { address = encodeURI(String(href ?? '')).replace(/%25/g, '%'); } catch { address = ''; }
    const titleAttr = title ? ` title="${attributeText(title)}"` : '';
    return `<span class="md-image-ref" data-md-image-href="${attributeText(address)}"${titleAttr}>`
        + `Image${alt ? `: ${alt}` : ''}</span>`;
}

function getMarkdownParser(breaks = true) {
    if (markdownParsers.has(breaks)) return markdownParsers.get(breaks);
    const Marked = globalThis.marked?.Marked;
    if (typeof Marked !== 'function') return null;
    const parser = new Marked({ gfm: true, breaks, renderer: { image: renderImageReference } });
    markdownParsers.set(breaks, parser);
    return parser;
}

// Inside a block marked did not read as code, its one code construct: marked's own
// code span, from a whole run of backticks (its first not escaped) to the next run
// of the same length, across lines. The caller cuts the text where a span must end.
const INLINE_CODE = /(?<!(?<!\\)(?:\\\\)*[\\`])(`+)(?!`)(?:[^`]|[^`][\s\S]*?[^`])\1(?!`)/g;
// A line whose ``` could still open a fence: the run leads it, after only a quote's or
// list item's marks (marked reads a task box's text as inline), and no backtick follows.
const FENCE_LEAD = /^(?:[ \t]*(?:>|(?:[-+*]|\d{1,9}[.)])[ \t]))*[ \t]*`{3,}[^`]*$/;
// The code elements marked writes; their text holds no raw `<`.
const RENDERED_CODE = /(<code\b[^>]*>[\s\S]*?<\/code>)/;
const HTML_ESCAPES = { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' };

// Code marked read inside a quote or list item, as [first line, its lines] of the
// top-level block's raw; every other block in it, and each table row, as [first
// line], since a code span never leaves one. marked strips a container's marks
// line by line and lexes what is left, so the children spell that text and its
// line N is raw line N. false when a child cannot be placed so: the caller then
// keeps the whole block literal.
// A task item's text has lost its `[ ] ` box, which marked keeps as the first child
// (a loose item's first paragraph starts with it): the children spell box + text.
function nestedCode(token, line, found) {
    const children = token.type === 'list' ? token.items
        : token.type === 'blockquote' || token.type === 'list_item' ? token.tokens : [];
    const box = token.type === 'list_item' && token.task
        ? [children?.[0], children?.[0]?.tokens?.[0]].find((child) => child?.type === 'checkbox') : null;
    const text = String(box?.raw ?? '') + String((token.type === 'list' ? token.raw : token.text) ?? '');
    if (token.type === 'table') token.raw.split('\n').forEach((_, row) => found.push([line + row]));
    let offset = 0;
    for (const child of children || []) {
        if (typeof child?.raw !== 'string' || !text.startsWith(child.raw, offset)) return false;
        if (child.type === 'code') {
            found.push([line, child.raw.replace(/\n+$/, '').split('\n')]);
        } else {
            found.push([line]);
            if (!nestedCode(child, line, found)) return false;
        }
        offset += child.raw.length;
        line += child.raw.split('\n').length - 1;
    }
    return true;
}

// The author's text split into [text, isCode] runs: every code block (fenced or
// indented, top-level or nested in a quote or list item) as marked's own lexer
// reads it, then the code spans inside the rest, each within its own block. Adjacent
// prose runs join, so math may still span them, but never across a code block.
function splitAuthorCode(source, parser) {
    const blocks = typeof parser?.lexer === 'function' ? parser.lexer(source) : [{ raw: source }];
    const runs = [];
    const push = (text, code) => {
        if (!code && runs.length && !runs.at(-1)[1]) runs.at(-1)[0] += text;
        else runs.push([text, code]);
    };
    const pushProse = (text) => {
        let last = 0;
        for (const match of text.matchAll(INLINE_CODE)) {
            push(text.slice(last, match.index), false);
            push(match[0], true);
            last = match.index + match[0].length;
        }
        push(text.slice(last), false);
    };
    for (const block of blocks) {
        if (block.type === 'code') { push(block.raw, true); continue; }
        const found = [];
        const lines = block.raw.split('\n');
        // Each nested code line must end its raw line (after the container's marks).
        if (!nestedCode(block, 0, found)
            || found.some(([at, code = []]) => code.some((text, k) => !lines[at + k]?.endsWith(text)))) {
            push(block.raw, true);
            continue;
        }
        const starts = [0];
        for (const text of lines) starts.push(starts.at(-1) + text.length + 1);
        let last = 0;
        for (const [at, code = []] of found) {
            if (starts[at] < last) continue; // it opens on the last line of the code before it
            pushProse(block.raw.slice(last, starts[at]));
            last = code.length ? starts[at + code.length] - 1 : starts[at];
            if (code.length) push(block.raw.slice(starts[at], last), true);
        }
        pushProse(block.raw.slice(last));
    }
    return runs;
}

// Math is parked only outside the author's code, and each block comes back as
// marked would write that same text where it landed: code encodes every `&`,
// prose keeps an entity an entity (the one layer it reads). No `<` returns raw.
// Parked math skips marked's escapes, so `authorText` gives it back its `\<`.
function protectLatexDelimiters(source, parser, authorText = (text) => text) {
    const rawSource = String(source).replace(/\r\n?/g, '\n'); // marked's own first step
    let tokenStem = 'OUROBOROSLATEX';
    while (rawSource.includes(tokenStem)) tokenStem += 'X';
    const replacements = [
        ['\\(', `${tokenStem}OPENINLINE`],
        ['\\)', `${tokenStem}CLOSEINLINE`],
        ['\\[', `${tokenStem}OPENBLOCK`],
        ['\\]', `${tokenStem}CLOSEBLOCK`],
    ];
    const displayBlocks = [];
    // Without a math delimiter there is nothing to park and no second lex to pay.
    const runs = /\$\$|\\[()[\]]/.test(rawSource) ? splitAuthorCode(rawSource, parser) : [[rawSource, true]];
    // What marked will read of the line being written: the code before it and the
    // math already parked on it count, since a code span or math may cross lines.
    let line = '';
    const write = (text) => {
        const end = text.lastIndexOf('\n');
        line = end < 0 ? line + text : text.slice(end + 1);
        return text;
    };
    const protectedSource = runs
        .map(([part, code]) => {
            if (code) return write(part);
            let last = 0;
            const withProtectedBlocks = part.replace(/\$\$[\s\S]+?\$\$|\\\[[\s\S]+?\\\]/g, (block, at) => {
                write(part.slice(last, at));
                last = at + block.length;
                // After a ``` that could open a fence, math keeps that line's backticks in view:
                // marked read the line as text for one of them, and parking it would open a fence.
                if (/^[^\n]*`/.test(block) && FENCE_LEAD.test(line)) return write(block);
                const token = `${tokenStem}DISPLAY${displayBlocks.length}`;
                displayBlocks.push([token, authorText(block)]);
                return write(token);
            });
            write(part.slice(last));
            return replacements.reduce(
                (value, [delimiter, token]) => value.split(delimiter).join(token),
                withProtectedBlocks,
            );
        })
        .join('');
    return {
        protectedSource,
        restore: (html) => String(html).split(RENDERED_CODE).map((part, index) => {
            const escape = index % 2 === 1 ? /[&<>"']/g : /[<>"']|&(?!#?\w+;)/g;
            return displayBlocks.reduceRight(
                (value, [token, block]) => value.split(token).join(block.replace(escape, (char) => HTML_ESCAPES[char])),
                replacements.reduce((value, [delimiter, token]) => value.split(token).join(delimiter), part),
            );
        }).join(''),
    };
}

function validDownloadPath(path) {
    const value = String(path ?? '');
    if (!value || value.length > 4096 || value.startsWith('/') || /[\\\u0000-\u001f\u007f]/.test(value)) return false;
    const segments = value.split('/');
    return segments.every((segment) => segment && segment !== '.' && segment !== '..');
}

/** Chat-only URL policy: external links plus the exact relative file-download route. */
export function chatMarkdownUrl(value) {
    const text = String(value ?? '').trim();
    if (!text) return '';
    if (text.startsWith('/')) {
        try {
            const parsed = new URL(text, 'https://chat.invalid');
            const params = [...parsed.searchParams.entries()];
            if (parsed.origin !== 'https://chat.invalid'
                || parsed.pathname !== '/api/files/download'
                || parsed.hash
                || params.length !== 1
                || params[0][0] !== 'path'
                || !validDownloadPath(params[0][1])) return '';
            return `/api/files/download?${new URLSearchParams({ path: params[0][1] })}`;
        } catch {
            return '';
        }
    }
    const safe = safeExternalUrl(text);
    return safe === '#' ? '' : safe;
}

/** Parse and bound an untrusted JSON chart configuration. */
export function parseChartConfig(source) {
    const parsed = JSON.parse(String(source ?? ''));
    if (!parsed || typeof parsed !== 'object' || Array.isArray(parsed)) {
        throw new Error('chart configuration must be an object');
    }
    if (!CHART_TYPES.has(parsed.type)) throw new Error('unsupported chart type');
    if (!parsed.data || typeof parsed.data !== 'object' || Array.isArray(parsed.data)) {
        throw new Error('chart data must be an object');
    }
    if (!Array.isArray(parsed.data.datasets)) throw new Error('chart data.datasets must be an array');
    if (parsed.data.datasets.length > MAX_CHART_DATASETS) throw new Error('too many chart datasets');
    const datasets = parsed.data.datasets.map((dataset) => {
        if (!dataset || typeof dataset !== 'object' || Array.isArray(dataset) || !Array.isArray(dataset.data)) {
            throw new Error('each chart dataset must contain a data array');
        }
        if (dataset.data.length > MAX_CHART_POINTS) throw new Error('too many chart points');
        const safeDataset = { data: [...dataset.data] };
        for (const key of ['label', 'backgroundColor', 'borderColor', 'borderWidth', 'fill', 'tension']) {
            if (Object.hasOwn(dataset, key)) safeDataset[key] = dataset[key];
        }
        return safeDataset;
    });
    if (parsed.data.labels !== undefined && !Array.isArray(parsed.data.labels)) {
        throw new Error('chart labels must be an array');
    }
    if (parsed.data.labels?.length > MAX_CHART_POINTS) throw new Error('too many chart labels');
    const userOptions = parsed.options && typeof parsed.options === 'object' && !Array.isArray(parsed.options)
        ? parsed.options
        : {};
    const config = {
        type: parsed.type,
        data: {
            datasets,
            ...(parsed.data.labels === undefined ? {} : { labels: [...parsed.data.labels] }),
        },
        options: {
            ...userOptions,
            responsive: true,
            maintainAspectRatio: false,
        },
    };
    // What the message author actually wrote, kept beside the config rather than
    // inside it: a repaint has to know which colours are the author's to leave
    // alone, and Chart.js resolves its own defaults over everything else.
    CHART_AUTHORED.set(config, userOptions);
    return config;
}

function codeLanguage(code) {
    const languageClass = Array.from(code.classList || [])
        .find((name) => name.startsWith('language-')) || '';
    return languageClass.replace(/^language-/, '');
}

function createCodeBlock(source, language = '') {
    const block = document.createElement('div');
    block.className = 'md-code-block';
    const label = document.createElement('span');
    label.className = 'md-code-language';
    label.textContent = language || 'text';
    const copy = document.createElement('button');
    copy.type = 'button';
    copy.className = 'md-code-copy';
    copy.dataset.codeCopy = '';
    copy.setAttribute('aria-label', tr('code.copy_code', 'Copy code'));
    copy.title = tr('code.copy_code', 'Copy code');
    copy.textContent = tr('code.copy', 'Copy');
    const pre = document.createElement('pre');
    const code = document.createElement('code');
    code.className = `language-${language || 'plain'}`;
    code.textContent = String(source ?? '');
    pre.appendChild(code);
    block.append(label, copy, pre);
    return block;
}

function transformRenderedMarkdown(fragment, literal) {
    // Levels 4+ carry the smallest label class, as the compact renderMarkdown
    // demotes them; chat bubbles still size h4-h6 by element (DESIGN.md §5).
    fragment.querySelectorAll('h1, h2, h3, h4, h5, h6').forEach((heading) => {
        heading.classList.add(`md-h${Math.min(Number(heading.tagName.slice(1)), 3)}`);
    });
    fragment.querySelectorAll('blockquote').forEach((quote) => quote.classList.add('md-quote'));
    fragment.querySelectorAll('.md-image-ref').forEach((reference) => {
        const address = chatMarkdownUrl(reference.getAttribute('data-md-image-href') || '');
        reference.removeAttribute('data-md-image-href');
        // A refused address stays plain text, never a link-styled anchor without
        // a destination. Inside an authored link, or around one in its own words,
        // that link is already the destination: anchors never nest.
        if (!address || reference.closest('a') || reference.querySelector('a')) return;
        const link = document.createElement('a');
        link.className = 'md-image-ref';
        link.setAttribute('href', address);
        if (reference.title) link.title = reference.title;
        link.append(...reference.childNodes);
        reference.replaceWith(link);
    });
    fragment.querySelectorAll('a').forEach((link) => {
        const safe = chatMarkdownUrl(link.getAttribute('href') || '');
        if (safe) link.setAttribute('href', safe); else link.removeAttribute('href');
        link.classList.add('md-link');
        link.target = '_blank';
        link.rel = 'noopener noreferrer';
    });
    fragment.querySelectorAll('code:not(pre code)').forEach((code) => {
        code.textContent = literal(code.textContent);
        code.classList.add('inline-code');
    });
    fragment.querySelectorAll('input[type="checkbox"]').forEach((input) => {
        const item = input.closest('li');
        if (item) item.classList.add('task-list-item');
        item?.closest('ul, ol')?.classList.add('task-list');
        const marker = document.createElement('span');
        marker.className = `md-checkbox${input.checked ? ' is-checked' : ''}`;
        marker.setAttribute('aria-hidden', 'true');
        marker.textContent = input.checked ? '✓' : '';
        input.replaceWith(marker);
    });
    fragment.querySelectorAll('table').forEach((table) => {
        table.classList.add('md-table');
        const wrap = document.createElement('div');
        wrap.className = 'md-table-wrap';
        table.replaceWith(wrap);
        wrap.appendChild(table);
    });
    fragment.querySelectorAll('pre > code').forEach((code) => {
        const language = codeLanguage(code);
        const source = literal(code.textContent || '');
        if (language === 'mermaid' || language === 'chart') {
            const richBlock = document.createElement('div');
            richBlock.className = `md-${language}`;
            richBlock.textContent = source;
            code.parentElement.replaceWith(richBlock);
            return;
        }
        code.parentElement.replaceWith(createCodeBlock(source, language));
    });
}

/** Return sanitized, presentation-ready HTML for a chat message, or with
 * `softBreaks` for a delivered document (DESIGN "Document reading"). */
export function renderChatMarkdown(text, { softBreaks = false } = {}) {
    // Without the parser the message reads as the author's exact text.
    const plain = () => escapeText(text).replace(/\n/g, '<br>');
    const parser = getMarkdownParser(!softBreaks);
    if (!parser || !globalThis.DOMPurify || typeof document === 'undefined') return plain();
    try {
        const { source, lessThan, escapedLessThan } = prepareMarkdownSource(text);
        const authorText = (value) => String(value).split(escapedLessThan).join(`\\${lessThan}`);
        const latex = protectLatexDelimiters(source, parser, authorText);
        const parsed = latex.restore(parser.parse(latex.protectedSource, { async: false }));
        const safe = globalThis.DOMPurify.sanitize(parsed, {
            USE_PROFILES: { html: true },
            // 'input' stays sanitizable so the task-list post-pass can swap checkboxes for inert glyphs; raw HTML cannot open upstream (no author `<` reaches the parser).
            FORBID_TAGS: ['script', 'iframe', 'object', 'embed', 'form', 'img', 'video', 'audio', 'source'],
            FORBID_ATTR: ['style', 'src', 'srcset', 'srcdoc', 'onerror', 'onload'],
        });
        const template = document.createElement('template');
        template.innerHTML = safe;
        // Code text still holds `lessThan` where the author wrote `<` and
        // `escapedLessThan` where they wrote `\<`, and only there.
        transformRenderedMarkdown(template.content, (value) => authorText(value).split(lessThan).join('<'));
        return template.innerHTML;
    } catch (error) {
        console.warn('renderChatMarkdown: markdown render failed', error);
        return plain();
    }
}

/** Mount blocks with their CSS contract. Enhancement stays with the owning
 * bubble/card so replacing content does not create another resource owner. */
export function mountChatMarkdown(host, text, options = {}) {
    host.classList.add('ui-rich-content');
    host.innerHTML = renderChatMarkdown(text, options);
}

function highlightCodeIn(root) {
    root.querySelectorAll?.('.md-code-block pre > code').forEach((code) => {
        const source = code.textContent || '';
        // A block past the rich-block bound stays plain text: highlighting it would
        // hold the main thread; it still reads and copies exactly.
        if (source.length > MAX_RICH_BLOCK_SOURCE_LENGTH) return;
        const language = codeLanguage(code);
        const api = globalThis.hljs;
        if (!api) return;
        try {
            const result = language && api.getLanguage(language)
                ? api.highlight(source, { language, ignoreIllegals: true })
                : api.highlightAuto(source);
            code.innerHTML = result.value;
            code.classList.add('hljs');
        } catch {
            code.textContent = source;
        }
    });
}

function renderLatexIn(root) {
    if (typeof globalThis.renderMathInElement !== 'function') return;
    globalThis.renderMathInElement(root, {
        delimiters: [
            { left: '$$', right: '$$', display: true },
            { left: '\\[', right: '\\]', display: true },
            { left: '\\(', right: '\\)', display: false },
        ],
        ignoredTags: ['script', 'noscript', 'style', 'textarea', 'pre', 'code'],
        ignoredClasses: ['md-mermaid', 'md-chart', 'md-code-block'],
        throwOnError: false,
        strict: false,
    });
}

function loadMermaid() {
    if (globalThis.mermaid?.initialize && globalThis.mermaid?.run) {
        return Promise.resolve(globalThis.mermaid);
    }
    if (mermaidLoadPromise) return mermaidLoadPromise;
    mermaidLoadPromise = new Promise((resolve, reject) => {
        const existing = document.getElementById(MERMAID_SCRIPT_ID);
        const script = existing || document.createElement('script');
        const rejectLoad = (message) => {
            mermaidLoadPromise = null;
            script.remove();
            reject(new Error(message));
        };
        const loaded = () => {
            if (globalThis.mermaid?.initialize && globalThis.mermaid?.run) resolve(globalThis.mermaid);
            else rejectLoad('diagram library did not initialize');
        };
        const failed = () => rejectLoad('diagram library failed to load');
        script.addEventListener('load', loaded, { once: true });
        script.addEventListener('error', failed, { once: true });
        if (!existing) {
            script.id = MERMAID_SCRIPT_ID;
            script.src = '/static/mermaid.min.js';
            script.async = true;
            document.head.appendChild(script);
        }
    });
    return mermaidLoadPromise;
}

function hardenMermaidLinks(node) {
    node.querySelectorAll?.('a').forEach((link) => {
        const href = link.getAttribute('href')
            || link.getAttribute('xlink:href')
            || link.getAttributeNS?.('http://www.w3.org/1999/xlink', 'href')
            || '';
        const safe = chatMarkdownUrl(href);
        if (!safe) {
            link.removeAttribute('href');
            link.removeAttribute('xlink:href');
            link.removeAttributeNS?.('http://www.w3.org/1999/xlink', 'href');
            return;
        }
        if (link.hasAttribute('href')) link.setAttribute('href', safe);
        if (link.hasAttribute('xlink:href')) link.setAttribute('xlink:href', safe);
        if (!link.hasAttribute('href') && !link.hasAttribute('xlink:href')) link.setAttribute('href', safe);
        link.setAttribute('target', '_blank');
        link.setAttribute('rel', 'noopener noreferrer');
    });
}

function initializeMermaid(api) {
    const theme = typeof document !== 'undefined' ? document.documentElement.dataset.theme || 'dark' : 'dark';
    if (mermaidInitialized === theme) return;
    const rootStyle = typeof getComputedStyle === 'function' && typeof document !== 'undefined'
        ? getComputedStyle(document.documentElement)
        : null;
    const diagramToken = (name, fallback) => rootStyle?.getPropertyValue(name).trim() || fallback;
    api.initialize({
        startOnLoad: false,
        securityLevel: 'strict',
        theme: 'base',
        themeVariables: {
            fontFamily: diagramToken('--diagram-font', 'Inter, system-ui, sans-serif'),
            background: diagramToken('--diagram-bg', '#151318'),
            primaryColor: diagramToken('--diagram-primary', '#25222c'),
            primaryTextColor: diagramToken('--diagram-primary-text', '#f4eef7'),
            primaryBorderColor: diagramToken('--diagram-border', '#6f6678'),
            lineColor: diagramToken('--diagram-line', '#9b90a6'),
            secondaryColor: diagramToken('--diagram-secondary', '#302b39'),
            tertiaryColor: diagramToken('--diagram-tertiary', '#19171d'),
        },
    });
    mermaidInitialized = theme;
}

function degradeMermaid(node, source, message) {
    const block = createCodeBlock(source, 'mermaid');
    block.classList.add('md-mermaid-error');
    const note = document.createElement('div');
    note.className = 'md-diagram-error-note';
    note.textContent = message;
    block.prepend(note);
    node.replaceWith(block);
}

/* Mermaid replaces a diagram's text with its SVG, so after the first mount the
   node no longer knows what it draws. The source is parked on the node (and
   survives the clone, which is what actually lands in the document) purely so a
   palette switch can redraw the same diagram instead of losing it. */
function mermaidSourceOf(node) {
    const stored = node.dataset?.mermaidSource;
    return typeof stored === 'string' ? stored : (node.textContent || '');
}

/** Put rendered diagrams back to their source so a re-render repaints them. */
function resetMermaidNodes(root) {
    const nodes = Array.from(root.querySelectorAll?.('.md-mermaid[data-mermaid-source]') || []);
    if (root.matches?.('.md-mermaid[data-mermaid-source]')) nodes.unshift(root);
    for (const node of nodes) {
        node.textContent = node.dataset.mermaidSource;
        // Mermaid refuses to re-run a node it has already claimed.
        node.removeAttribute('data-processed');
    }
    return nodes.length;
}

async function renderMermaidNodes(root, state, onDomWrite, epoch = state.epoch) {
    const stale = () => state.destroyed || state.epoch !== epoch || root.isConnected === false;
    const foundNodes = Array.from(root.querySelectorAll?.('.md-mermaid') || []);
    if (root.matches?.('.md-mermaid')) foundNodes.unshift(root);
    const nodes = [];
    const oversized = [];
    for (const node of foundNodes) {
        const source = mermaidSourceOf(node);
        if (source.length > MAX_RICH_BLOCK_SOURCE_LENGTH) {
            oversized.push({ node, source });
        } else {
            // Park before awaiting the library: a theme event during its first
            // load must discover this fence and schedule a replacement pass.
            node.dataset.mermaidSource = source;
            nodes.push(node);
        }
    }
    if (oversized.length) onDomWrite(() => {
        let changed = false;
        for (const { node, source } of oversized) {
            if (node.isConnected === false) continue;
            degradeMermaid(node, source, 'Diagram could not be rendered.');
            changed = true;
        }
        return changed;
    });
    if (!nodes.length) return;
    let api;
    try {
        api = await loadMermaid();
    } catch {
        if (stale()) return;
        onDomWrite(() => {
            let changed = false;
            nodes.forEach((node) => {
                if (node.isConnected === false) return;
                degradeMermaid(node, mermaidSourceOf(node), 'Diagram library failed to load.');
                changed = true;
            });
            return changed;
        });
        return;
    }
    if (stale()) return;
    initializeMermaid(api);
    for (const node of nodes) {
        if (stale()) return;
        if (node.isConnected === false) continue;
        const source = mermaidSourceOf(node);
        node.dataset.mermaidSource = source;
        const rendered = node.cloneNode(true);
        const stage = document.createElement('div');
        stage.className = 'md-mermaid-stage';
        stage.setAttribute('aria-hidden', 'true');
        // A collapsed/hidden fence measures 0 wide; fall back to the nearest
        // visible ancestor before the 320px floor so the diagram is laid out
        // for the container it will actually mount into.
        const stageWidth = node.getBoundingClientRect?.().width
            || node.closest?.('.message')?.getBoundingClientRect?.().width
            || root.getBoundingClientRect?.().width
            || 0;
        stage.style.setProperty(
            '--md-mermaid-stage-width',
            `${Math.max(stageWidth, 320)}px`,
        );
        stage.append(rendered);
        document.body.append(stage);
        try {
            await api.run({ nodes: [rendered], suppressErrors: true });
            hardenMermaidLinks(rendered);
            stage.remove();
            // A theme switch during this await already reset the node and started
            // a fresh pass; this SVG is in the old palette and must be dropped.
            if (stale() || node.isConnected === false) return;
            onDomWrite(() => {
                if (node.isConnected === false) return false;
                node.replaceWith(rendered);
                return true;
            });
        } catch {
            stage.remove();
            if (!stale() && node.isConnected !== false) {
                onDomWrite(() => {
                    if (node.isConnected === false) return false;
                    degradeMermaid(node, source, 'Diagram could not be rendered.');
                    return true;
                });
            }
        }
    }
}

function renderChartNodes(root, state, onDomWrite) {
    const nodes = Array.from(root.querySelectorAll?.('.md-chart') || []);
    if (root.matches?.('.md-chart')) nodes.unshift(root);
    for (const node of nodes) {
        const source = node.textContent || '';
        try {
            if (source.length > MAX_RICH_BLOCK_SOURCE_LENGTH) throw new Error('chart source is too long');
            if (typeof globalThis.Chart !== 'function') throw new Error('chart library is unavailable');
            const config = parseChartConfig(source);
            const authored = CHART_AUTHORED.get(config) || {};
            const canvas = document.createElement('canvas');
            onDomWrite(() => {
                node.replaceChildren(canvas);
                const chart = new globalThis.Chart(canvas, config);
                state.charts.add(chart);
                CHART_THEMED.set(chart, authored);
                applyChartTheme(chart, authored);
                node.dataset.processed = 'true';
                return true;
            });
        } catch {
            onDomWrite(() => {
                node.classList.add('md-chart-error');
                node.textContent = source;
                node.dataset.processed = 'true';
                return true;
            });
        }
    }
}

async function copyCode(code) {
    const text = code?.textContent || '';
    if (navigator.clipboard?.writeText) {
        await navigator.clipboard.writeText(text);
        return;
    }
    const textarea = document.createElement('textarea');
    textarea.value = text;
    textarea.setAttribute('readonly', '');
    // Inside a modal dialog (the document reader) the rest of the page is inert:
    // the selection is made within that dialog, and focus returns after.
    const dialog = code.closest?.('dialog[open]') || null;
    const focused = document.activeElement;
    let copied = false;
    try {
        (dialog || document.body).appendChild(textarea);
        if (dialog) textarea.focus({ preventScroll: true });
        textarea.select();
        copied = typeof document.execCommand === 'function' && document.execCommand('copy') === true;
    } finally {
        textarea.remove();
        if (dialog && focused?.isConnected) focused.focus({ preventScroll: true });
    }
    if (!copied) throw new Error('copy command failed');
}

/* A table scrolls sideways only when its columns cannot fit even wrapped. Its
   edges follow the actually hidden columns, as `scroll_fade.js` does for the
   vertical scroll bodies, and only while it scrolls is its wrapper a named
   region the keyboard can focus and scroll. */
function markTableOverflow(wrap) {
    if (!wrap || wrap.isConnected === false) return;
    const hidden = wrap.scrollWidth - wrap.clientWidth;
    const scrolled = Math.abs(wrap.scrollLeft);
    const scrolls = hidden > 1;
    wrap.toggleAttribute('data-scroll-start', scrolls && scrolled > 1);
    wrap.toggleAttribute('data-scroll-end', scrolls && hidden - scrolled > 1);
    if (scrolls === wrap.hasAttribute('tabindex')) return;
    if (scrolls) {
        wrap.setAttribute('role', 'region');
        wrap.setAttribute('aria-label', tr('code.scrollable_table', 'Scrollable table'));
        wrap.tabIndex = 0;
    } else {
        wrap.removeAttribute('role');
        wrap.removeAttribute('aria-label');
        wrap.removeAttribute('tabindex');
    }
}

// The wrapper attributes `markTableOverflow` decides by. A keyed patch (a task
// timeline row) keeps the wrapper but copies attributes from fresh markup, which
// drops them while nothing resizes: their change re-marks the wrapper.
const TABLE_STATE = ['tabindex', 'data-scroll-start', 'data-scroll-end'];

// The table wrappers at or under a node: a mutation may add the wrapper itself.
function tableWraps(node) {
    if (node?.nodeType !== 1) return [];
    const nested = Array.from(node.querySelectorAll?.('.md-table-wrap') || []);
    return node.matches?.('.md-table-wrap') ? [node, ...nested] : nested;
}

/**
 * Keep every Markdown table under `rootEl`, from either renderer and including
 * tables written into it later, a keyboard region with fading edges exactly
 * while it scrolls. Table-only: nothing else in the root is enhanced, so a
 * compact surface's literal code stays as rendered. Returns the disposer;
 * `destroyChatMarkdown` on the root or an ancestor releases it too, and binding
 * a bound root again returns its disposer.
 */
export function bindMarkdownTables(rootEl) {
    if (!rootEl || typeof ResizeObserver !== 'function') return () => {};
    const bound = TABLE_BINDINGS.get(rootEl);
    if (bound) return bound;
    // The wrapper resizes with its column; the table with its own content.
    const sizes = new ResizeObserver((entries) => {
        for (const entry of entries) markTableOverflow(entry.target.closest('.md-table-wrap'));
    });
    const observe = (wrap) => {
        sizes.observe(wrap);
        if (wrap.firstElementChild) sizes.observe(wrap.firstElementChild);
    };
    const release = (wrap) => {
        sizes.unobserve(wrap);
        if (wrap.firstElementChild) sizes.unobserve(wrap.firstElementChild);
    };
    for (const wrap of rootEl.querySelectorAll?.('.md-table-wrap') || []) observe(wrap);
    // A surface that writes its Markdown later (a timeline row, a fetched review)
    // is followed: a new table is observed, a removed one released. The DOM as it
    // stands when the records arrive decides, so a moved table stays observed.
    const writes = typeof MutationObserver === 'function' ? new MutationObserver((records) => {
        for (const record of records) {
            if (record.type === 'attributes') {
                if (record.target.classList?.contains('md-table-wrap')) markTableOverflow(record.target);
                continue;
            }
            for (const node of record.removedNodes) {
                for (const wrap of tableWraps(node)) if (!rootEl.contains(wrap)) release(wrap);
            }
            for (const node of record.addedNodes) {
                for (const wrap of tableWraps(node)) if (rootEl.contains(wrap)) observe(wrap);
            }
        }
    }) : null;
    writes?.observe(rootEl, { childList: true, subtree: true, attributeFilter: TABLE_STATE });
    // Scroll events do not bubble; one capturing listener follows every wrapper.
    const onScroll = (event) => {
        if (event?.target?.classList?.contains('md-table-wrap')) markTableOverflow(event.target);
    };
    rootEl.addEventListener?.('scroll', onScroll, true);
    const dispose = () => {
        if (TABLE_BINDINGS.get(rootEl) !== dispose) return;
        TABLE_BINDINGS.delete(rootEl);
        sizes.disconnect();
        writes?.disconnect();
        rootEl.removeEventListener?.('scroll', onScroll, true);
        rootEl.removeAttribute?.('data-md-tables');
    };
    TABLE_BINDINGS.set(rootEl, dispose);
    // The marker lets `destroyChatMarkdown` find the binding from an ancestor.
    rootEl.setAttribute?.('data-md-tables', '');
    return dispose;
}

function cleanupState(root, state) {
    if (!state || state.destroyed) return;
    state.destroyed = true;
    state.unsubscribeTheme?.();
    state.unsubscribeTheme = null;
    state.disposeTables?.();
    state.disposeTables = null;
    root.removeEventListener('click', state.clickHandler);
    for (const chart of state.charts) {
        try { chart.destroy(); } catch {}
    }
    state.charts.clear();
    for (const timer of state.timers) clearTimeout(timer);
    state.timers.clear();
    if (state.frame !== null && typeof cancelAnimationFrame === 'function') cancelAnimationFrame(state.frame);
    state.frame = null;
    root.removeAttribute('data-chat-markdown-enhanced');
    ROOT_STATE.delete(root);
}

/** Enhance mounted markdown and return a disposer for resources acquired here. */
export function enhanceChatMarkdown(rootEl, { onDomWrite = writeDirectly, onThemeDomWrite = onDomWrite } = {}) {
    if (!rootEl) return () => {};
    destroyChatMarkdown(rootEl);
    const state = {
        charts: new Set(), timers: new Set(), clickHandler: null, frame: null, destroyed: false,
        // Bumped by every repaint so a diagram render still awaiting mermaid from
        // the previous palette discards its result instead of racing this one in.
        epoch: 0, unsubscribeTheme: null, disposeTables: null,
    };
    state.clickHandler = async (event) => {
        const button = event.target?.closest?.('[data-code-copy]');
        if (!button || !rootEl.contains(button)) return;
        const code = button.closest('.md-code-block')?.querySelector('pre > code');
        if (!code) return;
        try {
            await copyCode(code);
            if (state.destroyed) return;
            button.classList.add('is-copied');
            button.textContent = tr('code.copied', 'Copied');
            const timer = setTimeout(() => {
                state.timers.delete(timer);
                if (state.destroyed || button.isConnected === false) return;
                button.classList.remove('is-copied');
                button.textContent = tr('code.copy', 'Copy');
            }, 1200);
            state.timers.add(timer);
        } catch {
            if (!state.destroyed) button.textContent = tr('code.copy_failed', 'Copy failed');
        }
    };
    ROOT_STATE.set(rootEl, state);
    rootEl.setAttribute('data-chat-markdown-enhanced', 'true');
    rootEl.addEventListener('click', state.clickHandler);
    // The palette moved under an already-mounted message. Charts keep their
    // instances (and therefore their data); diagrams redraw from the source they
    // parked at mount. Nothing re-reads the markdown or rebuilds the bubble.
    state.unsubscribeTheme = onThemeChange(() => {
        if (state.destroyed || rootEl.isConnected === false) return;
        for (const chart of state.charts) applyChartTheme(chart, CHART_THEMED.get(chart) || {});
        state.epoch += 1;
        // The reset collapses each SVG back to a line of text, so it goes through
        // the caller's local DOM writer, preserving scroll without new activity.
        let pending = 0;
        onThemeDomWrite(() => { pending = resetMermaidNodes(rootEl); return pending > 0; });
        if (pending) void renderMermaidNodes(rootEl, state, onThemeDomWrite);
    });
    const start = () => {
        if (state.destroyed || rootEl.isConnected === false) return;
        onDomWrite(() => {
            highlightCodeIn(rootEl);
            renderLatexIn(rootEl);
            return true;
        });
        void renderMermaidNodes(rootEl, state, onDomWrite);
        if (rootEl.querySelector?.('.md-table-wrap')) state.disposeTables = bindMarkdownTables(rootEl);
        const mountCharts = () => {
            state.frame = null;
            if (!state.destroyed && rootEl.isConnected !== false) {
                renderChartNodes(rootEl, state, onDomWrite);
            }
        };
        if (typeof requestAnimationFrame === 'function') {
            state.frame = requestAnimationFrame(mountCharts);
        } else {
            mountCharts();
        }
    };
    if (rootEl.isConnected === false) queueMicrotask(start); else start();
    return () => cleanupState(rootEl, state);
}

/** Destroy markdown resources rooted at or below an element. */
export function destroyChatMarkdown(rootEl) {
    if (!rootEl) return;
    const roots = [];
    if (ROOT_STATE.has(rootEl)) roots.push(rootEl);
    roots.push(...(rootEl.querySelectorAll?.('[data-chat-markdown-enhanced]') || []));
    for (const root of new Set(roots)) cleanupState(root, ROOT_STATE.get(root));
    // A table-only binding (a compact surface) goes with the same node.
    for (const root of [rootEl, ...(rootEl.querySelectorAll?.('[data-md-tables]') || [])]) {
        TABLE_BINDINGS.get(root)?.();
    }
}

// Any future bubble-removal path in chat.js MUST call destroyChatMarkdown() or Chart
// instances leak (Chart.js keeps a static registry that only destroy() releases).
