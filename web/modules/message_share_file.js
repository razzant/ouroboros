// The File format of Share (docs/DESIGN.md "Replies"): one saved answer or living card as one self-contained HTML
// page that opens without Ouroboros, made from what the room itself draws, never by a second renderer:
//   answer      the message as the room shows it once its drawing is done (`drawnAnswer`): its Markdown, tables,
//               formulas (their MathML), diagrams (their SVG), each chart as its picture beside its data;
//   components  the card through the Widgets renderer (widget_render.js), every tab a labelled section, pictures and
//               media inline, each chart as its picture beside its data; a control that would press the card is drawn
//               disabled, holds no value the owner typed and says that it works only in Ouroboros;
//   app         the app's whole HTML in the chat's own sandbox and policy (card_frames.js), its saved state left
//               behind and its submit sending nothing. The card's words are the page's first screen. The frame the
//               app starts in stays out of sight until the frame itself proves it runs (its bridge measured it there),
//               it loaded once, and a settle window passed without a script error or a refused script; only then does
//               it take their place. The words stay where scripts do not run (a phone's file preview), when the app
//               fails or is too slow to start, and when it leaves for another page.
// The look is the room's. What the file holds keeps the ancestor contract it is drawn under in the room (a message or
// card inside a bubble of the Ouroboros voice, that bubble's own chrome undone and the text it passes down read from
// the room's bubble), and the file carries the rules of the page's own stylesheets that draw it, with the theme's values
// where it was made. Nothing in a file loads from anywhere (its policy refuses every fetch), and it names no task,
// chat, card or row of the host.

import { tr } from './i18n.js';
import { CARD_RENDER_CONTEXT } from './card_components.js';
import { CARD_APP_SANDBOX, INLINE_FRAME_FLOOR, cardAppSrcdoc, cardThemeTokens } from './card_frames.js';
import { renderChartDataTable } from './widget_chart.js';
import { renderComponent, syncWidgetCharts } from './widget_render.js';
import { WIDGET_FRAME_MAX_HEIGHT } from './widget_module.js';

/** A file's policy: nothing from any address, no form going anywhere. An app's file adds inline script (its own shell
 *  and the frame the app runs in, which inherits this policy beside its own). */
export const SHARE_FILE_CSP = "default-src 'none'; style-src 'unsafe-inline'; img-src data: blob:; media-src data: blob:; "
    + "font-src data:; form-action 'none'; base-uri 'none'";
export const SHARE_APP_FILE_CSP = `${SHARE_FILE_CSP}; script-src 'unsafe-inline' blob: data:`;
// How long a proven, loaded app has for a script error before it is shown, and how long it has to prove it runs at all.
export const APP_SETTLE_MS = 150;
export const APP_START_MS = 10000;
// How long a File waits for the room to finish drawing an answer's diagrams and charts, and how often it looks.
export const DRAWING_WAIT_MS = 8000;
const DRAWING_STEP_MS = 100;
const STAGE_WIDTH = 640;
const COMMIT_TYPES = new Set(['action', 'form', 'poll']);
// The data attributes the page's rules draw by (a tone, a state, a phone table's cell label); every other one is the
// host's own (an identity, a route, a source, a press) and stays behind.
const DRAWN_DATA = new Set(['data-tone', 'data-state', 'data-label']);
// What a bubble passes down to what it holds, read from the room's bubble for the one the file draws it in.
const PASSED_DOWN = ['font-family', 'font-size', 'font-weight', 'font-style', 'line-height', 'letter-spacing', 'color',
    'overflow-wrap', 'word-break', 'text-align'];

// Dynamic states and pseudo-elements: a rule written for them applies to the page once its element is there.
const PSEUDO_ELEMENT = /::?(?:-webkit-|-moz-)?(?:before|after|marker|placeholder|selection|backdrop|first-line|first-letter|file-selector-button|details-marker|scrollbar(?:-[a-z]+)?|search-cancel-button|inner-spin-button|outer-spin-button|slider-[a-z-]+|progress-[a-z-]+|meter-[a-z-]+|color-swatch[a-z-]*)(?![\w-])|::part\([^)]*\)/gi;
const DYNAMIC = /:(?:-webkit-|-moz-)?(?:hover|focus-visible|focus-within|focus|active|visited|target|placeholder-shown|autofill|user-invalid|user-valid|popover-open|modal|fullscreen)(?![\w-])/gi;

const themeName = (doc) => {
    const candidate = globalThis.ouroTheme?.theme || doc?.documentElement?.dataset?.theme;
    return candidate === 'light' ? 'light' : 'dark';
};

/** The parts of a selector list, split at its top-level commas. */
export function selectorParts(text) {
    const parts = [];
    let depth = 0;
    let start = 0;
    let quote = '';
    const source = String(text || '');
    for (let index = 0; index < source.length; index += 1) {
        const char = source[index];
        if (quote) {
            if (char === '\\') index += 1;
            else if (char === quote) quote = '';
        } else if (char === '"' || char === "'") quote = char;
        else if (char === '(' || char === '[') depth += 1;
        else if (char === ')' || char === ']') depth -= 1;
        else if (char === ',' && depth === 0) { parts.push(source.slice(start, index)); start = index + 1; }
    }
    parts.push(source.slice(start));
    return parts.map((part) => part.trim()).filter(Boolean);
}

/** Whether one selector names `content` or anything inside it, in the document it lies in (so the room's ancestors the
 *  file keeps, a theme or `:root` count): first as written, then without its states and pseudo-elements. */
function partApplies(content, part) {
    const plain = part.replace(PSEUDO_ELEMENT, '').replace(DYNAMIC, '').trim();
    for (const selector of plain === part ? [part] : [part, plain]) {
        if (!selector) continue;
        try {
            if (content.matches(selector) || content.querySelector(selector)) return true;
        } catch { /* a selector this engine does not read on its own */ }
    }
    return false;
}

function sheetRules(sheet) {
    try { return sheet?.cssRules || []; } catch { return []; }  // a sheet from another origin keeps its rules
}

// A rule's address of an image, a mask or a cursor leaves as nothing: a file loads nothing and names no host path.
const withoutAddresses = (css) => css.replace(/url\(\s*(['"]?)(?!data:)[^'")]*\1\s*\)/gi, 'none');

function keptRules(rules, applies) {
    let out = '';
    for (const rule of Array.from(rules || [])) {
        if (typeof rule.selectorText === 'string') {
            // Only the selectors that draw what the file holds: one beside them for the bubble itself stays behind.
            const parts = selectorParts(rule.selectorText).filter(applies);
            if (parts.length) out += `${parts.join(', ')} { ${withoutAddresses(rule.style?.cssText || '')} }\n`;
        } else if (rule.styleSheet) out += keptRules(sheetRules(rule.styleSheet), applies);
        else if (rule.cssRules && /^@(?:media|supports|container|layer)\b/i.test(rule.cssText || '')) {
            const inner = keptRules(rule.cssRules, applies);
            if (inner) out += `${rule.cssText.slice(0, rule.cssText.indexOf('{')).trim()} {\n${inner}}\n`;
        }
        // @font-face, @keyframes, @property and the like: a file that loads nothing and moves nothing needs none.
    }
    return out;
}

const tokenValue = (value) => {
    const text = String(value || '').trim();
    return text && !/url\(|[<>{};]/i.test(text) ? text : '';
};

/**
 * The CSS a file needs for `content` (an element inside the file's own document): every rule of `doc`'s stylesheets
 * that applies to it or inside it, and before them the value of each token those rules (and `extra`) read, as it is
 * where `material` lies in the room. A token the room sets above the content (`:root`, a theme, a voice) arrives as
 * that value; one set inside it comes with its own rule.
 */
export function usedStyles(content, { doc = globalThis.document, win = globalThis.window, material = null, extra = '',
    tokens = {} } = {}) {
    const applies = (part) => partApplies(content, part);
    let rules = '';
    for (const sheet of Array.from(doc?.styleSheets || [])) rules += keptRules(sheetRules(sheet), applies);
    const names = new Set();
    for (const match of `${rules}\n${extra}`.matchAll(/var\(\s*(--[\w-]+)/g)) names.add(match[1]);
    const computed = win?.getComputedStyle?.(material?.isConnected ? material : doc.documentElement);
    const values = [];
    for (const name of [...names].sort()) {
        const value = Object.hasOwn(tokens, name) ? tokenValue(tokens[name]) : tokenValue(computed?.getPropertyValue(name));
        if (value) values.push(`${name}: ${value};`);
    }
    for (const [name, value] of Object.entries(tokens)) {
        if (!names.has(name) && tokenValue(value)) values.push(`${name}: ${tokenValue(value)};`);
    }
    return `:root { ${values.join(' ')} }\n${rules}${extra}`.replace(/<\/style/gi, '<\\/style');
}

// The page around what it holds: the room, the object on it, and an answer's foot that says who made it.
const PAGE_CSS = `
html { -webkit-text-size-adjust: 100%; text-size-adjust: 100%; background: var(--bg-primary); }
body.ouro-share { margin: 0; min-height: 100vh; background: var(--bg-primary); color: var(--text-primary);
    font-family: var(--font-sans, system-ui, -apple-system, "Segoe UI", sans-serif); }
.ouro-share [hidden] { display: none !important; }
.ouro-share-sheet { box-sizing: border-box; width: 100%; max-width: 760px; margin: 0 auto;
    padding: clamp(8px, 3vw, 32px) clamp(8px, 3vw, 24px); }
.ouro-share-object { position: relative; box-sizing: border-box; display: flex; flex-direction: column;
    gap: var(--space-3); min-width: 0; padding: clamp(16px, 4vw, 24px); border-radius: var(--radius-object);
    background: var(--share-paper, var(--surface-object)); box-shadow: var(--elev-rest); }
.ouro-share-made { margin: 0; color: var(--assistant-sender); font-size: var(--type-meta); font-weight: 600;
    line-height: var(--line-meta); }
#ouro-share-status[data-tone="error"] { color: var(--status-error-fg); }
.ouro-share-chart { display: block; max-width: 100%; height: auto; margin: 0 auto; }
.ouro-share-content .md-chart { height: auto; }
.ouro-share-content math[display="block"] { display: block; margin: 0.75em 0; overflow-x: auto; }
.ouro-share-app.is-starting { position: absolute; inset-inline: 0; top: 0; height: 2px; overflow: hidden; opacity: 0;
    pointer-events: none; }
.ouro-share-frame { display: block; width: 100%; height: var(--ouro-share-app-height, 320px); border: 0;
    border-radius: var(--radius-inner); background: var(--surface-tray); }
`;

/** The bubble the file draws what it holds in: the room's own classes, its chrome undone (the object is the file's
 *  paper), and what the room's bubble passes down, as the room computes it. */
function roomRule(bubble, win) {
    const style = bubble?.isConnected ? win?.getComputedStyle?.(bubble) : null;
    const passed = PASSED_DOWN.map((name) => [name, tokenValue(style?.getPropertyValue(name))]).filter(([, value]) => value)
        .map(([name, value]) => ` ${name}: ${value} !important;`).join('');
    return `.ouro-share-room { all: unset !important; display: block !important; min-width: 0 !important;${passed} }\n`;
}

function newPage(doc, { kind, title }) {
    const page = doc.implementation.createHTMLDocument('');
    const theme = themeName(doc);
    page.documentElement.dataset.theme = theme;
    const head = page.head;
    const meta = (attributes) => {
        const node = page.createElement('meta');
        for (const [name, value] of Object.entries(attributes)) node.setAttribute(name, value);
        head.append(node);
    };
    head.replaceChildren();
    meta({ charset: 'utf-8' });
    meta({ 'http-equiv': 'Content-Security-Policy', content: kind === 'app' ? SHARE_APP_FILE_CSP : SHARE_FILE_CSP });
    meta({ name: 'viewport', content: 'width=device-width, initial-scale=1' });
    meta({ name: 'referrer', content: 'no-referrer' });
    meta({ name: 'color-scheme', content: theme });
    const titleNode = page.createElement('title');
    titleNode.textContent = title;
    head.append(titleNode);
    page.body.className = 'ouro-share';
    const sheet = page.createElement('main');
    sheet.className = 'ouro-share-sheet';
    const object = page.createElement('article');
    object.className = `ouro-share-object is-${kind}`;
    sheet.append(object);
    page.body.append(sheet);
    return { page, object, theme };
}

/** The room's bubble around `classes` content in the file (see `roomRule`). */
function roomOf(page, classes, attributes = {}) {
    const room = page.createElement('div');
    room.className = `${classes} ouro-share-room`;
    for (const [name, value] of Object.entries(attributes)) room.setAttribute(name, value);
    return room;
}

function finishPage({ page, theme }, content, { doc, win, material, paper, bubble }) {
    const style = page.createElement('style');
    style.textContent = usedStyles(content, { doc, win, material,
        extra: `html { color-scheme: ${theme}; }\n${PAGE_CSS}${roomRule(bubble, win)}`,
        tokens: paper ? { '--share-paper': paper } : {} });
    page.head.append(style);
    return `<!doctype html>\n${page.documentElement.outerHTML}\n`;
}

/** The colour the room paints under `node` (its bubble): the object's paper in the file. */
function paperOf(node, win) {
    for (let current = node; current && current.nodeType === 1; current = current.parentElement) {
        const colour = String(win?.getComputedStyle?.(current)?.backgroundColor || '');
        if (colour && colour !== 'transparent' && !/rgba\([^)]*,\s*0\)$/.test(colour)) return colour;
    }
    return '';
}

const bubbleOf = (node) => node?.closest?.('.chat-bubble') || null;
const safeHref = (value) => (/^(?:https?:\/\/|mailto:)\S/i.test(String(value || '').trim()) ? String(value).trim() : '');
const safeMedia = (value) => /^data:(?:image|audio|video)\/[\w.+-]+;base64,[A-Za-z0-9+/=\s]+$/i.test(String(value || '').trim());

/** What remains of drawn markup in a file: no script, no control that would press anything, no host address or
 *  host attribute; a widget's inline file becomes a plain download link. */
export function portableMarkup(root) {
    const page = root.ownerDocument;
    root.querySelectorAll('[data-code-copy], .md-mermaid-stage, .katex-html').forEach((node) => node.remove());
    root.querySelectorAll('.katex').forEach((node) => {
        const math = node.querySelector('math');
        if (math) node.replaceWith(math);
    });
    root.querySelectorAll('button[data-widget-download-url]').forEach((node) => {
        const href = String(node.dataset.widgetDownloadUrl || '');
        if (!href.startsWith('data:')) { node.remove(); return; }
        const link = page.createElement('a');
        link.className = node.className;
        link.setAttribute('href', href);
        link.setAttribute('download', node.dataset.widgetDownloadFilename || 'download');
        link.textContent = node.textContent;
        node.replaceWith(link);
    });
    root.querySelectorAll('form').forEach((form) => {
        const group = page.createElement('div');
        group.className = form.className;
        group.setAttribute('role', 'group');
        if (form.getAttribute('aria-label')) group.setAttribute('aria-label', form.getAttribute('aria-label'));
        group.append(...form.childNodes);
        form.replaceWith(group);
    });
    root.querySelectorAll('script, style:not(svg style), iframe, object, embed, template, noscript, link, meta, base, canvas')
        .forEach((node) => node.remove());
    root.querySelectorAll('button, input, select, textarea, fieldset').forEach((node) => node.setAttribute('disabled', ''));
    for (const node of [root, ...root.querySelectorAll('*')]) {
        // A diagram keeps its drawing's own attributes; elsewhere only what the page's rules draw by stays.
        const drawing = Boolean(node.closest('svg'));
        for (const { name } of Array.from(node.attributes)) {
            const key = name.toLowerCase();
            if (key.startsWith('on') || (key.startsWith('data-') && !drawing && !DRAWN_DATA.has(key))
                || ['tabindex', 'contenteditable', 'draggable', 'aria-busy', 'autofocus', 'srcset', 'formaction', 'action',
                    'ping'].includes(key)) node.removeAttribute(name);
        }
        const tag = node.localName;
        if (tag === 'a' || tag === 'area') {
            const href = node.getAttribute('href');
            if (href !== null && !safeHref(href) && !(node.hasAttribute('download') && href.startsWith('data:'))) node.removeAttribute('href');
            if (node.hasAttribute('xlink:href') && !safeHref(node.getAttribute('xlink:href'))) node.removeAttribute('xlink:href');
            if (node.hasAttribute('href')) { node.setAttribute('target', '_blank'); node.setAttribute('rel', 'noopener noreferrer'); }
        } else if (['img', 'audio', 'video', 'source', 'track'].includes(tag) && node.hasAttribute('src') && !safeMedia(node.getAttribute('src'))) {
            node.removeAttribute('src');
        }
        if (node.classList?.contains('md-table-wrap')) { node.removeAttribute('role'); node.removeAttribute('aria-label'); }
    }
    return root;
}

/** The first words of a page, for its title: its first heading, else its opening words. */
function leadingWords(node) {
    const heading = node.querySelector('h1, h2, h3, h4, h5, h6');
    const text = String((heading || node).textContent || '').replace(/\s+/g, ' ').trim();
    return text.length > 80 ? `${text.slice(0, 79).trimEnd()}…` : text;
}

/** A file's name from its title: the words a person reads, without what a file system refuses. */
export function shareFileName(title, fallback = 'ouroboros-card') {
    const words = String(title || '').normalize('NFC').replace(/…$/, '').replace(/[\u0000-\u001f\u007f<>:"/\\|?*]+/g, ' ')
        .replace(/\s+/g, ' ').replace(/^[\s.]+|[\s.]+$/g, '');
    const clipped = Array.from(words).slice(0, 80).join('').trim();
    return `${clipped || fallback}.html`;
}

/**
 * What of the room's drawing of an answer `content` is still under way: each diagram not yet replaced by its SVG or its
 * error block, each chart not yet drawn or refused (chat_markdown.js marks both with `data-processed`), and each chart
 * still animating to its values. Empty once everything the room will draw is drawn.
 */
export function drawingUnderWay(content, { Chart = globalThis.Chart } = {}) {
    const pending = [];
    for (const node of content?.querySelectorAll?.('.md-mermaid') || []) if (!node.querySelector('svg')) pending.push('diagram');
    for (const node of content?.querySelectorAll?.('.md-chart') || []) {
        const canvas = node.querySelector('canvas');
        const chart = canvas && Chart?.getChart?.(canvas);
        if (node.dataset?.processed !== 'true' || (chart && Chart?.animator?.running?.(chart))) pending.push('chart');
    }
    return pending;
}

/** Wait, at most `wait` ms and while `alive()`, for the room to finish drawing `content`; resolves to what is still
 *  under way then (`drawingUnderWay`), which a file shows as its source. */
export async function drawnAnswer(content, { alive = () => true, wait = DRAWING_WAIT_MS, step = DRAWING_STEP_MS,
    underWay = drawingUnderWay } = {}) {
    const until = Date.now() + wait;
    let pending = underWay(content);
    while (pending.length && alive() && Date.now() < until) {
        await new Promise((resolve) => { setTimeout(resolve, step); });
        pending = underWay(content);
    }
    return pending;
}

function madeLine(page, words) {
    const line = page.createElement('p');
    line.className = 'ouro-share-made';
    line.textContent = words;
    return line;
}

/**
 * The page of one Ouroboros message: `content` is its drawn node in the room (read, never changed). Charts become
 * their picture with their data; formulas their MathML; diagrams keep their SVG. It lies in the room's own message
 * contract (a rich message in an Ouroboros bubble whose Markdown is enhanced), so it reads as it does there.
 */
export function answerPage(content, { doc = globalThis.document, win = globalThis.window } = {}) {
    if (!content) throw new Error(tr('share.file_nothing', 'This message has nothing to put in a file.'));
    const shell = newPage(doc, { kind: 'answer', title: 'Ouroboros' });
    const { page, object } = shell;
    const room = roomOf(page, 'chat-bubble assistant', { 'data-chat-markdown-enhanced': 'true' });
    const body = page.createElement('div');
    body.className = 'message ui-rich-content ouro-share-content';
    const copy = page.importNode(content, true);
    const live = Array.from(content.querySelectorAll('canvas'));
    Array.from(copy.querySelectorAll('canvas')).forEach((canvas, index) => {
        const drawn = live[index];
        const chart = drawn && globalThis.Chart?.getChart?.(drawn);
        let source = '';
        try { source = drawn?.toDataURL?.('image/png') || ''; } catch { source = ''; }
        if (!source) return;  // removed with the other canvases: no picture, no blank box
        const picture = page.createElement('img');
        picture.className = 'ouro-share-chart';
        picture.src = source;
        picture.alt = tr('share.chart_picture', 'Chart');
        const box = drawn.getBoundingClientRect?.();
        if (box?.width) { picture.width = Math.round(box.width); picture.height = Math.round(box.height); }
        canvas.replaceWith(picture);
        const data = chart?.data;
        if (Array.isArray(data?.datasets) && data.datasets.length) {
            const table = page.createElement('div');
            table.innerHTML = renderChartDataTable({ type: chart?.config?.type, data: {
                labels: (data.labels || []).map((label) => String(label ?? '')),
                datasets: data.datasets.map((set) => ({ label: String(set?.label ?? ''), data: Array.isArray(set?.data) ? set.data : [] })) } },
            tr('share.chart_label', 'chart'), false);
            picture.after(...table.childNodes);
        }
    });
    body.append(...copy.childNodes);
    portableMarkup(body);
    room.append(body);
    const title = leadingWords(body);
    page.head.querySelector('title').textContent = title || 'Ouroboros';
    object.append(room, madeLine(page, tr('share.card_brand', 'Ouroboros')));
    return { html: finishPage(shell, body, { doc, win, material: content, paper: paperOf(content, win), bubble: bubbleOf(content) }),
        title };
}

/**
 * A component card's components as a file shows them: each tab a labelled section, and every control that would press
 * the card (an action, a form, polling, a move between columns) drawn disabled. A copy: the card is not edited.
 */
export function portableComponents(components) {
    const visit = (component) => {
        if (!component || typeof component !== 'object') return component;
        if (component.type === 'tabs') {
            const { tabs, type: _type, ...rest } = component;
            return { ...rest, type: 'group', layout: 'stack', components: (Array.isArray(tabs) ? tabs : []).map((tab, index) => ({
                type: 'group', layout: 'stack', title: String(tab?.label || `Tab ${index + 1}`),
                components: (Array.isArray(tab?.components) ? tab.components : []).map(visit),
            })) };
        }
        const next = { ...component };
        if (COMMIT_TYPES.has(next.type)) next.disabled = true;
        if (next.type === 'kanban') delete next.on_move;
        if (Array.isArray(next.components)) next.components = next.components.map(visit);
        if (Array.isArray(next.render)) next.render = next.render.map(visit);
        return next;
    };
    return (Array.isArray(components) ? components : []).map(visit);
}

const pressesCard = (components) => JSON.stringify(components || []).match(/"type":"(?:action|form|poll)"|"on_move"/) !== null;

/** A card as the room lays it out (chat_cards.js): its title, its body, its foot saying who made it. */
function cardSection(page, title) {
    const section = page.createElement('section');
    section.className = 'chat-card';
    section.setAttribute('aria-label', title || 'Card');
    if (title) {
        const heading = page.createElement('div');
        heading.className = 'chat-card-title ouro-share-title';
        heading.setAttribute('role', 'heading');
        heading.setAttribute('aria-level', '1');
        heading.textContent = title;
        section.append(heading);
    }
    const body = page.createElement('div');
    body.className = 'chat-card-body';
    section.append(body);
    const foot = page.createElement('div');
    foot.className = 'chat-card-foot';
    const made = page.createElement('span');
    made.className = 'chat-card-made';
    made.textContent = tr('share.card_made', 'Ouroboros made this');
    foot.append(made);
    return { section, body, foot };
}

/**
 * The page of one component card `card` (its `CardResponse`): drawn by the Widgets renderer with its own data and
 * nothing the owner pressed or typed. Charts are drawn once on a stage outside the view, in the room's theme, and
 * leave as their picture. `material` is the card's node in the room, whose colours the page takes.
 */
export async function componentsPage(card, { doc = globalThis.document, win = globalThis.window, material = null } = {}) {
    const view = { state: { result: card?.data && typeof card.data === 'object' ? card.data : {} }, status: {},
        componentState: {}, formValues: {}, pendingActions: new Set(), visibleKeys: new Set() };
    const markup = portableComponents(card?.components)
        .map((component, index) => renderComponent(CARD_RENDER_CONTEXT, component, view, `components.${index}`)).join('');
    const stage = doc.createElement('div');
    stage.className = 'chat-card-components message-share-stage';
    stage.setAttribute('aria-hidden', 'true');
    const width = Math.round(material?.getBoundingClientRect?.().width || 0);
    stage.style.setProperty('--message-share-stage-width', `${width >= 280 ? width : STAGE_WIDTH}px`);
    stage.innerHTML = markup;
    doc.body.append(stage);
    const charts = new Map();
    const pictures = new Map();
    try {
        syncWidgetCharts(stage, charts, new Map());
        const ratio = Math.max(2, Number(win?.devicePixelRatio) || 1);
        for (const chart of charts.values()) {
            chart.stop();
            chart.options.animation = false;
            chart.options.devicePixelRatio = ratio;
            chart.resize();
            chart.update('none');
            pictures.set(chart.canvas, { src: chart.canvas.toDataURL('image/png'), width: chart.width, height: chart.height,
                label: chart.canvas.getAttribute('aria-label') || '' });
        }
        const title = String(card?.title || '');
        const shell = newPage(doc, { kind: 'components', title: title || 'Ouroboros' });
        const { page, object } = shell;
        const room = roomOf(page, 'chat-bubble assistant chat-card-bubble');
        const { section, body: cardBody, foot } = cardSection(page, title);
        const body = page.createElement('div');
        body.className = 'chat-card-components ouro-share-content';
        const copy = page.importNode(stage, true);
        const drawn = Array.from(stage.querySelectorAll('canvas'));
        Array.from(copy.querySelectorAll('canvas')).forEach((canvas, index) => {
            const shot = pictures.get(drawn[index]);
            const host = canvas.closest('.widget-chart-canvas') || canvas;
            if (!shot) { host.remove(); return; }
            const picture = page.createElement('img');
            picture.className = 'ouro-share-chart';
            picture.src = shot.src;
            picture.alt = shot.label || tr('share.chart_picture', 'Chart');
            picture.width = Math.round(shot.width);
            picture.height = Math.round(shot.height);
            host.replaceWith(picture);
        });
        body.append(...copy.childNodes);
        portableMarkup(body);
        cardBody.append(body);
        if (pressesCard(card?.components)) {
            const note = page.createElement('div');
            note.className = 'chat-card-status';
            note.textContent = tr('share.file_controls', 'Its buttons and fields work only in Ouroboros.');
            section.append(note);
        }
        section.append(foot);
        room.append(section);
        object.append(room);
        return { html: finishPage(shell, section, { doc, win, material, paper: paperOf(material, win), bubble: bubbleOf(material) }),
            title };
    } finally {
        charts.forEach((chart) => chart.destroy());
        stage.remove();
    }
}

/** The script an app's file runs (the parent side of card_frames.js's bridge, for a file on its own). */
const APP_SHELL = `(() => {
    const read = (id) => JSON.parse(document.getElementById(id).textContent);
    const config = read('ouro-share-config');
    const words = config.words;
    const first = document.getElementById('ouro-share-first');
    const note = document.getElementById('ouro-share-note');
    const line = document.getElementById('ouro-share-status');
    const say = (text, tone) => { line.textContent = text || ''; line.dataset.tone = tone || ''; line.hidden = !text; };
    let srcdoc = '';
    try { srcdoc = read('ouro-share-source'); } catch (error) { note.textContent = words.unreadable; return; }
    const host = document.createElement('div');
    host.className = 'ouro-share-app is-starting';
    host.setAttribute('aria-hidden', 'true');
    host.inert = true;
    const frame = document.createElement('iframe');
    frame.className = 'ouro-share-frame';
    frame.setAttribute('sandbox', config.sandbox);
    frame.setAttribute('title', config.title);
    frame.tabIndex = -1;
    // Started means proven from inside the frame: its bridge ran there and measured it (a frame whose scripts never run
    // sends nothing), the frame loaded once, and the settle window passed with no script error and no refused script.
    // Until then the card's words stay the first screen.
    let state = 'starting';
    let height = 0;
    let loads = 0;
    let measured = false;
    let settle = 0;
    let deadline = 0;
    const fit = () => host.style.setProperty('--ouro-share-app-height',
        Math.min(config.max, Math.max(config.floor, height || config.initial)) + 'px');
    const stop = (text) => {
        if (state === 'stopped') return;
        clearTimeout(settle);
        clearTimeout(deadline);
        state = 'stopped';
        host.remove();
        first.hidden = false;
        note.textContent = text;
        say('');
    };
    const ready = () => {
        if (state !== 'starting') return;
        clearTimeout(deadline);
        state = 'ready';
        fit();
        host.classList.remove('is-starting');
        host.removeAttribute('aria-hidden');
        host.inert = false;
        frame.removeAttribute('tabindex');
        first.hidden = true;
    };
    const consider = () => {
        if (state === 'starting' && measured && loads === 1 && !settle) settle = setTimeout(ready, config.settle);
    };
    // A page in a background tab measures nothing until it is seen: its time to start counts from then.
    const late = () => {
        if (state !== 'starting') return;
        if (document.hidden) {
            document.addEventListener('visibilitychange', () => { deadline = setTimeout(late, config.start); }, { once: true });
            return;
        }
        stop(words.slow);
    };
    deadline = setTimeout(late, config.start);
    window.addEventListener('message', (event) => {
        const msg = event.data;
        if (event.source !== frame.contentWindow || !msg || typeof msg !== 'object' || msg.nonce !== config.nonce) return;
        if (msg.type === 'ouro-card-resize') {
            const value = Number(msg.height);
            if (!Number.isFinite(value) || value <= 0) return;
            height = Math.ceil(value);
            measured = true;
            if (state === 'ready') fit(); else consider();
        } else if (msg.type === 'ouro-card-error') {
            const said = String(msg.message || 'unknown').slice(0, 300);
            const refusedScript = msg.kind === 'csp' && /^script-src(?:-elem|-attr)?$/.test(said);
            const text = (msg.kind === 'csp' ? words.blocked : words.error) + ': ' + said;
            if (state === 'starting' && (msg.kind !== 'csp' || refusedScript)) stop(words.failed + ' ' + text);
            else if (state !== 'stopped') say(text, 'error');
        } else if (msg.type === 'ouro-card-submit') {
            if (state === 'ready') say(words.submit, '');
        }
    });
    frame.addEventListener('load', () => {
        loads += 1;
        if (loads > 1) stop(words.left); else consider();
    });
    frame.srcdoc = srcdoc;
    host.append(frame);
    first.before(host);
    note.textContent = words.starting;
})();`;

const jsonScript = (page, id, value) => {
    const node = page.createElement('script');
    node.type = 'application/json';
    node.id = id;
    node.textContent = JSON.stringify(value).replace(/</g, '\\u003c');
    return node;
};

/**
 * The page of one app card `card` (its `CardResponse`) whose HTML is `html`: the card as the room lays it out, its own
 * words with what they are, and, where scripts run and the app proves it runs, the app itself in the chat's sandbox
 * under this file's shell, starting from no saved state. `material` is the card's node in the room: its colours are the
 * page's and the app's theme.
 */
export function appPage(card, html, { doc = globalThis.document, win = globalThis.window, material = null,
    nonce = `${Date.now().toString(36)}-${Math.random().toString(36).slice(2)}-${Math.random().toString(36).slice(2)}` } = {}) {
    const title = String(card?.title || '');
    const shell = newPage(doc, { kind: 'app', title: title || 'Ouroboros' });
    const { page, object } = shell;
    const room = roomOf(page, 'chat-bubble assistant chat-card-bubble');
    const { section, body, foot } = cardSection(page, title);
    // The first screen, and all of it where scripts do not run: the card's words and what they are, as the room's face
    // of a card not running. The app's frame is no part of this markup; the shell puts it before them once it runs.
    body.classList.add('ouro-share-content');
    const first = page.createElement('section');
    first.className = 'chat-card-face';
    first.id = 'ouro-share-first';
    const words = String(card?.fallback_text || '').trim();
    if (words) {
        const text = page.createElement('p');
        text.className = 'chat-card-fallback';
        text.textContent = words;
        first.append(text);
    }
    const note = page.createElement('p');
    note.className = 'chat-card-status';
    note.id = 'ouro-share-note';
    note.textContent = words ? tr('share.app_first', 'These are the card’s words. Its app opens in a browser with JavaScript.')
        : tr('share.app_first_bare', 'This card is an app. It opens in a browser with JavaScript.');
    first.append(note);
    body.append(first);
    const status = page.createElement('p');
    status.className = 'chat-card-status';
    status.id = 'ouro-share-status';
    status.setAttribute('role', 'status');
    status.setAttribute('aria-live', 'polite');
    status.hidden = true;
    section.append(status, foot);
    room.append(section);
    object.append(room);
    const theme = cardThemeTokens(doc, material?.isConnected ? material : doc.documentElement);
    page.body.append(
        jsonScript(page, 'ouro-share-config', {
            nonce, title: title || 'App', sandbox: CARD_APP_SANDBOX, max: WIDGET_FRAME_MAX_HEIGHT, floor: INLINE_FRAME_FLOOR,
            initial: 320, settle: APP_SETTLE_MS, start: APP_START_MS,
            words: {
                starting: tr('share.app_starting', 'Starting the app…'),
                failed: tr('share.app_failed', 'The app did not start here.'),
                slow: tr('share.app_slow', 'The app did not start here in time.'),
                left: tr('share.app_left', 'The app tried to open another page and was stopped.'),
                unreadable: tr('share.app_unreadable', 'This file’s app could not be read.'),
                error: tr('share.app_error', 'App error'),
                blocked: tr('share.app_blocked', 'Blocked by the card’s policy'),
                submit: tr('share.app_submit', 'This copy sends nothing.'),
            },
        }),
        jsonScript(page, 'ouro-share-source', cardAppSrcdoc(html, { nonce, theme, saved: null })),
    );
    const script = page.createElement('script');
    script.textContent = APP_SHELL;
    page.body.append(script);
    return { html: finishPage(shell, section, { doc, win, material, paper: paperOf(material, win), bubble: bubbleOf(material) }),
        title };
}
