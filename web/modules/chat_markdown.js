/** Rich, sanitized markdown rendering for assistant and system chat messages. */

import { safeExternalUrl } from './utils.js';

const CHART_TYPES = new Set([
    'bar', 'line', 'pie', 'doughnut', 'polarArea', 'radar', 'scatter', 'bubble',
]);
const MAX_CHART_DATASETS = 24;
const MAX_CHART_POINTS = 500;
const MAX_RICH_BLOCK_SOURCE_LENGTH = 32768;
const MERMAID_SCRIPT_ID = 'chat-mermaid-library';
const ROOT_STATE = new WeakMap();
const writeDirectly = (mutate) => mutate();

let markdownParser = null;
let mermaidLoadPromise = null;
let mermaidInitialized = false;

function escapeText(value) {
    return String(value ?? '')
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;');
}

function decodeHtmlEntities(value) {
    const raw = String(value ?? '');
    if (!/&[#A-Za-z]/.test(raw)) return raw;
    if (typeof document !== 'undefined') {
        const textarea = document.createElement('textarea');
        textarea.innerHTML = raw;
        return textarea.value;
    }
    return raw
        .replace(/&lt;/gi, '<')
        .replace(/&gt;/gi, '>')
        .replace(/&quot;/gi, '"')
        .replace(/&#0*39;/g, "'")
        .replace(/&#x0*27;/gi, "'")
        .replace(/&amp;/gi, '&');
}

/** Decode one stored entity layer, then escape it so raw HTML stays literal. */
export function prepareMarkdownSource(text) {
    return escapeText(decodeHtmlEntities(String(text ?? '')));
}

function getMarkdownParser() {
    if (markdownParser) return markdownParser;
    const Marked = globalThis.marked?.Marked;
    if (typeof Marked !== 'function') return null;
    markdownParser = new Marked({ gfm: true, breaks: true });
    return markdownParser;
}

function protectLatexDelimiters(source) {
    const rawSource = String(source);
    let tokenStem = 'OUROBOROSLATEX';
    while (rawSource.includes(tokenStem)) tokenStem += 'X';
    const replacements = [
        ['\\(', `${tokenStem}OPENINLINE`],
        ['\\)', `${tokenStem}CLOSEINLINE`],
        ['\\[', `${tokenStem}OPENBLOCK`],
        ['\\]', `${tokenStem}CLOSEBLOCK`],
    ];
    const displayBlocks = [];
    const protectedSource = rawSource.split(/(```[\s\S]*?```|`[^`\n]*`)/g)
        .map((part, index) => {
            if (index % 2 === 1) return part;
            const withProtectedBlocks = part.replace(/\$\$[\s\S]+?\$\$|\\\[[\s\S]+?\\\]/g, (block) => {
                const token = `${tokenStem}DISPLAY${displayBlocks.length}`;
                displayBlocks.push([token, block]);
                return token;
            });
            return replacements.reduce(
                (value, [delimiter, token]) => value.split(delimiter).join(token),
                withProtectedBlocks,
            );
        })
        .join('');
    return {
        protectedSource,
        restore: (html) => displayBlocks.reduceRight(
            (value, [token, block]) => value.split(token).join(block),
            replacements.reduce(
                (value, [delimiter, token]) => String(value).split(token).join(delimiter),
                html,
            ),
        ),
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
    return {
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
    copy.setAttribute('aria-label', 'Copy code');
    copy.title = 'Copy code';
    copy.textContent = 'Copy';
    const pre = document.createElement('pre');
    const code = document.createElement('code');
    code.className = `language-${language || 'plain'}`;
    code.textContent = String(source ?? '');
    pre.appendChild(code);
    block.append(label, copy, pre);
    return block;
}

function transformRenderedMarkdown(fragment) {
    fragment.querySelectorAll('h1, h2, h3').forEach((heading) => {
        heading.classList.add(`md-${heading.tagName.toLowerCase()}`);
    });
    fragment.querySelectorAll('blockquote').forEach((quote) => quote.classList.add('md-quote'));
    fragment.querySelectorAll('a').forEach((link) => {
        const safe = chatMarkdownUrl(link.getAttribute('href') || '');
        if (safe) link.setAttribute('href', safe); else link.removeAttribute('href');
        link.classList.add('md-link');
        link.target = '_blank';
        link.rel = 'noopener noreferrer';
    });
    fragment.querySelectorAll('code:not(pre code)').forEach((code) => {
        code.textContent = decodeHtmlEntities(code.textContent);
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
        const source = decodeHtmlEntities(code.textContent || '');
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

/** Return sanitized, presentation-ready HTML for a chat message. */
export function renderChatMarkdown(text) {
    const source = prepareMarkdownSource(text).replace(
        /^((?:[ \t]*&gt;)+)/gm,
        (markers) => markers.replaceAll('&gt;', '>'),
    );
    const parser = getMarkdownParser();
    if (!parser || !globalThis.DOMPurify || typeof document === 'undefined') {
        return source.replace(/\n/g, '<br>');
    }
    try {
        const latex = protectLatexDelimiters(source);
        const parsed = latex.restore(parser.parse(latex.protectedSource, { async: false }));
        const safe = globalThis.DOMPurify.sanitize(parsed, {
            USE_PROFILES: { html: true },
            // 'input' stays sanitizable so the task-list post-pass can swap checkboxes for inert glyphs; raw HTML is already escaped upstream.
            FORBID_TAGS: ['script', 'iframe', 'object', 'embed', 'form', 'img', 'video', 'audio', 'source'],
            FORBID_ATTR: ['style', 'src', 'srcset', 'srcdoc', 'onerror', 'onload'],
        });
        const template = document.createElement('template');
        template.innerHTML = safe;
        transformRenderedMarkdown(template.content);
        return template.innerHTML;
    } catch (error) {
        console.warn('renderChatMarkdown: markdown render failed', error);
        return source.replace(/\n/g, '<br>');
    }
}

function highlightCodeIn(root) {
    root.querySelectorAll?.('.md-code-block pre > code').forEach((code) => {
        const source = code.textContent || '';
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
    if (mermaidInitialized) return;
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
    mermaidInitialized = true;
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

async function renderMermaidNodes(root, state, onDomWrite) {
    const foundNodes = Array.from(root.querySelectorAll?.('.md-mermaid') || []);
    if (root.matches?.('.md-mermaid')) foundNodes.unshift(root);
    const nodes = [];
    const oversized = [];
    for (const node of foundNodes) {
        const source = node.textContent || '';
        if (source.length > MAX_RICH_BLOCK_SOURCE_LENGTH) {
            oversized.push({ node, source });
        } else {
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
        if (state.destroyed || root.isConnected === false) return;
        onDomWrite(() => {
            let changed = false;
            nodes.forEach((node) => {
                if (node.isConnected === false) return;
                degradeMermaid(node, node.textContent || '', 'Diagram library failed to load.');
                changed = true;
            });
            return changed;
        });
        return;
    }
    if (state.destroyed || root.isConnected === false) return;
    initializeMermaid(api);
    for (const node of nodes) {
        if (state.destroyed || root.isConnected === false) return;
        if (node.isConnected === false) continue;
        const source = node.textContent || '';
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
            if (state.destroyed || root.isConnected === false || node.isConnected === false) return;
            onDomWrite(() => {
                if (node.isConnected === false) return false;
                node.replaceWith(rendered);
                return true;
            });
        } catch {
            stage.remove();
            if (!state.destroyed && root.isConnected !== false && node.isConnected !== false) {
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
            const canvas = document.createElement('canvas');
            onDomWrite(() => {
                node.replaceChildren(canvas);
                const chart = new globalThis.Chart(canvas, config);
                state.charts.add(chart);
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
    let copied = false;
    try {
        document.body.appendChild(textarea);
        textarea.select();
        copied = typeof document.execCommand === 'function' && document.execCommand('copy') === true;
    } finally {
        textarea.remove();
    }
    if (!copied) throw new Error('copy command failed');
}

function cleanupState(root, state) {
    if (!state || state.destroyed) return;
    state.destroyed = true;
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
export function enhanceChatMarkdown(rootEl, { onDomWrite = writeDirectly } = {}) {
    if (!rootEl) return () => {};
    destroyChatMarkdown(rootEl);
    const state = {
        charts: new Set(), timers: new Set(), clickHandler: null, frame: null, destroyed: false,
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
            button.textContent = 'Copied';
            const timer = setTimeout(() => {
                state.timers.delete(timer);
                if (state.destroyed || button.isConnected === false) return;
                button.classList.remove('is-copied');
                button.textContent = 'Copy';
            }, 1200);
            state.timers.add(timer);
        } catch {
            if (!state.destroyed) button.textContent = 'Copy failed';
        }
    };
    ROOT_STATE.set(rootEl, state);
    rootEl.setAttribute('data-chat-markdown-enhanced', 'true');
    rootEl.addEventListener('click', state.clickHandler);
    const start = () => {
        if (state.destroyed || rootEl.isConnected === false) return;
        onDomWrite(() => {
            highlightCodeIn(rootEl);
            renderLatexIn(rootEl);
            return true;
        });
        void renderMermaidNodes(rootEl, state, onDomWrite);
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
}

// Any future bubble-removal path in chat.js MUST call destroyChatMarkdown() or Chart
// instances leak (Chart.js keeps a static registry that only destroy() releases).
