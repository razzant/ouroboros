/* Frames inside a living card (docs/WIDGET_HOST_PROTOCOL.md "Frames in a message"): the opaque frame an `app` card
   runs in, and the inline size policy every frame in a message follows (an app, or a skill's module widget shown
   inline by a `widget` card).

   An app is the HTML the host serves as inert text (`GET /api/cards/{id}/app`), put into a created iframe through
   the `srcdoc` property with sandbox `allow-scripts` ONLY — never the same-origin token, forms, popups, modals or
   downloads — under a document policy that blocks every fetch, WebSocket and load (nothing from any URL). No policy
   covers a frame navigating itself (the host closes an app whose frame loads a second document) or WebRTC
   (WIDGET_HOST_PROTOCOL §9; the author is Ouroboros, which has the network anyway). Its one channel is a
   nonce-bound bridge, `window.OuroborosCard` inside the frame:
     resize(height)            the frame's content height (the bridge also reports it on its own);
     state.save(value)         JSON of at most 32 KiB, kept per card in this tab's sessionStorage and handed back
                               as `state.value` at the next mount;
     submit(value, summary)    the HOST puts one line into the owner's draft (`From «<title>»: <summary>` and the
                               value as compact JSON when it is small) and focuses the field. Nothing is sent: only
                               the owner's Send does that.
   The bridge also reports, on its own, a script error, an unhandled rejection or a policy refusal in the app (at most
   ten, each once), which the card shows on its status line. The host's theme reaches the frame as `--ouro-*` custom
   properties (the values of the material the frame lies on), `data-theme` and `OuroborosCard.onTheme`. */

import { onThemeChange } from './theme_palette.js';
import { WIDGET_FRAME_MAX_HEIGHT } from './widget_module.js';

export const CARD_APP_SANDBOX = 'allow-scripts';
export const CARD_APP_CSP = "default-src 'none'; script-src 'unsafe-inline' blob: data:; style-src 'unsafe-inline'; "
    + 'img-src data: blob:; media-src data: blob:; font-src data:';
export const CARD_STATE_MAX_BYTES = 32 * 1024;
export const INLINE_FRAME_FLOOR = 120;
const INLINE_FRAME_SHARE = 0.7;
const LINE_PART_MAX = 300;
const THEME_TOKENS = {
    '--ouro-text': '--text-primary', '--ouro-text-meta': '--text-meta', '--ouro-accent': '--accent',
    '--ouro-surface': '--surface-object', '--ouro-tray': '--surface-tray', '--ouro-border': '--surface-border',
    '--ouro-ok': '--status-ok-fg', '--ouro-warn': '--status-warn-fg', '--ouro-error': '--status-error-fg',
    '--ouro-font': '--font-sans', '--ouro-radius': '--radius-inner',
};

const byteLength = (text) => new TextEncoder().encode(text).length;

/**
 * The height a frame in a message takes for `content` px of content: never under the floor, and never over 70 % of
 * the viewport unless the owner expanded it (then up to the widget frame ceiling).
 */
export function inlineFrameHeight(content, { expanded = false, viewport = globalThis.innerHeight || 800 } = {}) {
    const cap = expanded ? WIDGET_FRAME_MAX_HEIGHT : Math.max(INLINE_FRAME_FLOOR, Math.floor(viewport * INLINE_FRAME_SHARE));
    const wanted = Number.isFinite(content) ? Math.ceil(content) : 0;
    return Math.min(cap, Math.max(INLINE_FRAME_FLOOR, wanted));
}

/**
 * The inline size policy for the frame in `host`: `fit(contentHeight)` sizes it (the height is the host's
 * `--widget-frame-height`, which the frame inherits); `toggle` (a button) shows only while the collapsed frame would
 * cut its content, and switches Expand / Collapse. Re-fits when the window resizes.
 */
export function createInlineFit(host, toggle) {
    let content = 0;
    let expanded = false;
    const apply = () => {
        host.style.setProperty('--widget-frame-height', `${inlineFrameHeight(content, { expanded })}px`);
        const clips = inlineFrameHeight(content) < Math.ceil(content);
        if (!toggle) return;
        toggle.hidden = !clips;
        toggle.textContent = expanded ? 'Collapse' : 'Expand';
        toggle.setAttribute('aria-expanded', expanded ? 'true' : 'false');
    };
    const onToggle = () => { expanded = !expanded; apply(); };
    toggle?.addEventListener('click', onToggle);
    globalThis.addEventListener?.('resize', apply);
    apply();
    return {
        fit(height) { content = Number(height) || 0; apply(); },
        dispose() { toggle?.removeEventListener('click', onToggle); globalThis.removeEventListener?.('resize', apply); },
    };
}

/** The one draft line an app's submit becomes. */
export function cardSubmitLine(title, value, summary = '') {
    const words = String(summary ?? '').replace(/\s+/g, ' ').trim().slice(0, LINE_PART_MAX);
    let json = '';
    try { json = value === undefined || value === null ? '' : JSON.stringify(value); } catch { json = ''; }
    const repeats = typeof value === 'string' && value.replace(/\s+/g, ' ').trim() === words;
    const shown = json && json.length <= LINE_PART_MAX && !repeats ? json : '';
    return `From «${String(title || 'card').replace(/\s+/g, ' ').trim()}»: ${[words, shown].filter(Boolean).join(' ') || '(no summary)'}`;
}

/**
 * The host's theme as the frame reads it: `{name, css}` (a `:root` block of `--ouro-*` values). The values are those of
 * `material`, the element the transparent frame lies on: inside a dark voice its inks and surfaces are that voice's
 * own (ui.css "voice materials"), so a card's host is passed; the document root otherwise.
 */
export function cardThemeTokens(doc = globalThis.document, material = doc?.documentElement) {
    const candidate = globalThis.ouroTheme?.theme || doc?.documentElement?.dataset?.theme;
    const name = candidate === 'light' ? 'light' : 'dark';
    const computed = material && globalThis.getComputedStyle ? globalThis.getComputedStyle(material) : null;
    const values = Object.entries(THEME_TOKENS).map(([own, host]) => {
        const value = String(computed?.getPropertyValue(host) || '').trim().replace(/[<>{};]/g, '');
        return value ? `${own}: ${value};` : '';
    }).join(' ');
    return { name, css: `:root { color-scheme: ${name}; ${values} } html { background: transparent; color: var(--ouro-text); font-family: var(--ouro-font, system-ui, sans-serif); }` };
}

// The frame side of the bridge. `saved` is the state this card kept in this tab (JSON text or null).
function cardBridgeScript(nonce, theme, saved) {
    return `(() => {
        const nonce = ${JSON.stringify(nonce)};
        const MAX = ${CARD_STATE_MAX_BYTES};
        let theme = ${JSON.stringify(theme)};
        let saved = ${JSON.stringify(saved)};
        try { saved = saved === null ? null : JSON.parse(saved); } catch { saved = null; }
        const post = (message) => window.parent.postMessage({ ...message, nonce }, '*');
        const style = document.getElementById('ouro-card-theme');
        const listeners = new Set();
        window.addEventListener('message', (event) => {
            const msg = event.source === window.parent ? event.data || {} : {};
            if (msg.nonce !== nonce || msg.type !== 'ouro-card-theme' || !msg.theme) return;
            theme = msg.theme;
            document.documentElement.dataset.theme = theme.name;
            if (style) style.textContent = theme.css;
            listeners.forEach((callback) => { try { callback(theme.name); } catch (error) { console.error(error); } });
        });
        let last = 0;
        let frame = 0;
        const report = () => {
            frame = 0;
            const height = Math.ceil(document.documentElement.offsetHeight || 0);
            if (height > 0 && height !== last) { last = height; post({ type: 'ouro-card-resize', height }); }
        };
        const schedule = () => { if (!frame) frame = requestAnimationFrame(report); };
        if (typeof ResizeObserver === 'function') new ResizeObserver(schedule).observe(document.documentElement);
        window.addEventListener('load', schedule);
        // A script error, a rejection nobody caught or a policy refusal otherwise leaves a blank or stuck app with no
        // word to the owner: each is reported once, ten at most, so a throwing animation loop cannot flood the host.
        const seen = new Set();
        const fault = (kind, message) => {
            const text = String(message ?? '').slice(0, 300);
            if (seen.size >= 10 || seen.has(kind + ':' + text)) return;
            seen.add(kind + ':' + text);
            post({ type: 'ouro-card-error', kind, message: text });
        };
        window.addEventListener('error', (event) => fault('error', event.message || event.error));
        window.addEventListener('unhandledrejection', (event) => fault('rejection', event.reason));
        window.addEventListener('securitypolicyviolation', (event) => fault('csp', event.violatedDirective || 'blocked'));
        window.OuroborosCard = Object.freeze({
            resize(height) {
                const value = Number(height);
                if (Number.isFinite(value) && value > 0) post({ type: 'ouro-card-resize', height: Math.ceil(value) });
            },
            state: Object.freeze({
                get value() { return saved; },
                save(value) {
                    const text = JSON.stringify(value === undefined ? null : value);
                    if (new TextEncoder().encode(text).length > MAX) throw new RangeError('card state is larger than 32 KiB');
                    saved = JSON.parse(text);
                    post({ type: 'ouro-card-state', state: text });
                },
            }),
            submit(value, summary = '') {
                post({ type: 'ouro-card-submit', value: value === undefined ? null : JSON.parse(JSON.stringify(value)), summary: String(summary ?? '') });
            },
            get theme() { return theme.name; },
            onTheme(callback) {
                if (typeof callback !== 'function') return () => {};
                listeners.add(callback);
                return () => listeners.delete(callback);
            },
        });
    })();`;
}

/**
 * The app's whole document: the host's head (the policy first, so it governs everything after it; the theme; the
 * bridge) and then the app's own HTML, which the parser folds into this document (an app's own doctype, html, head
 * and body tags only merge attributes there).
 */
export function cardAppSrcdoc(html, { nonce, theme = cardThemeTokens(), saved = null }) {
    const script = cardBridgeScript(nonce, theme, saved).replace(/<\/script/gi, '<\\/script').replace(/<!--/g, '<\\!--');
    return `<!doctype html><html data-theme="${theme.name}"><head><meta http-equiv="Content-Security-Policy" content="${CARD_APP_CSP}">`
        + `<meta name="viewport" content="width=device-width, initial-scale=1"><style id="ouro-card-theme">${theme.css}</style>`
        + `<script>${script}</script></head><body>${String(html ?? '')}</body></html>`;
}

/**
 * The parent side of the bridge: `handle(event)` acts only on a message from this frame's window carrying this
 * mount's nonce — `resize` (a positive height), `state` (JSON text of at most 32 KiB), `submit`, `error` (one line
 * for the card). True when handled.
 */
export function createCardAppBridge({ nonce, source, resize, save, submit, fault = () => {} }) {
    return (event) => {
        const msg = event?.data;
        if (!msg || typeof msg !== 'object' || msg.nonce !== nonce || event.source !== source()) return false;
        if (msg.type === 'ouro-card-resize') {
            const height = Number(msg.height);
            if (!Number.isFinite(height) || height <= 0) return false;
            resize(height);
            return true;
        }
        if (msg.type === 'ouro-card-state') {
            if (typeof msg.state !== 'string' || byteLength(msg.state) > CARD_STATE_MAX_BYTES) return false;
            try { JSON.parse(msg.state); } catch { return false; }
            save(msg.state);
            return true;
        }
        if (msg.type === 'ouro-card-submit') {
            submit(msg.value, typeof msg.summary === 'string' ? msg.summary : '');
            return true;
        }
        if (msg.type === 'ouro-card-error') {
            const label = msg.kind === 'csp' ? 'Blocked by the card\'s policy' : 'App error';
            fault(`${label}: ${String(msg.message || 'unknown').slice(0, 300)}`);
            return true;
        }
        return false;
    };
}

const stateKey = (cardId, sha256) => `ouro_card_state:${cardId}:${sha256 || ''}`;

function readState(storage, key) {
    try { return storage?.getItem(key) ?? null; } catch { return null; }
}

/**
 * Run one app inside `host`: `{cardId, title, sha256, html, fit, onSubmit(line), onLeave(), onFault(text), fresh}`;
 * `fit` is the frame's inline size policy (`createInlineFit`). The app's saved state (keyed by card and app bytes, so a
 * replaced app starts fresh) is handed back at mount unless `fresh`, which forgets it, so no later mount of this card
 * hands it back either. No policy stops a frame from navigating itself,
 * so a second document loading in the frame (the app left for a page) removes the frame at once and calls `onLeave`.
 * Returns the disposer.
 */
export function mountCardApp(host, { cardId, title, sha256, html, fit, onSubmit, onLeave = () => {}, onFault = () => {},
    fresh = false, storage = globalThis.sessionStorage }) {
    const nonce = `${Date.now().toString(36)}-${Math.random().toString(36).slice(2)}-${Math.random().toString(36).slice(2)}`;
    const key = stateKey(cardId, sha256);
    const iframe = document.createElement('iframe');
    iframe.className = 'widgets-frame chat-card-frame';
    iframe.setAttribute('sandbox', CARD_APP_SANDBOX);
    iframe.setAttribute('title', title || 'Card');
    // The frame is transparent over its host, so it reads the host's material, at mount and at every theme change;
    // a host not in the document has none yet and the root stands in.
    const theme = () => cardThemeTokens(document, host.isConnected ? host : document.documentElement);
    if (fresh) { try { storage?.removeItem(key); } catch { /* blocked: nothing was kept to forget */ } }
    iframe.srcdoc = cardAppSrcdoc(html, { nonce, theme: theme(), saved: fresh ? null : readState(storage, key) });
    host.replaceChildren(iframe);
    const handle = createCardAppBridge({
        nonce,
        source: () => iframe.contentWindow,
        resize: (height) => fit.fit(height),
        save: (text) => { try { storage?.setItem(key, text); } catch { /* the app keeps running; its state is not kept */ } },
        submit: (value, summary) => onSubmit(cardSubmitLine(title, value, summary)),
        fault: onFault,
    });
    window.addEventListener('message', handle);
    const stopTheme = onThemeChange(() => iframe.contentWindow?.postMessage({ type: 'ouro-card-theme', nonce, theme: theme() }, '*'));
    let disposed = false;
    let loads = 0;
    const dispose = () => {
        if (disposed) return;
        disposed = true;
        window.removeEventListener('message', handle);
        stopTheme();
        iframe.remove();
    };
    iframe.addEventListener('load', () => {
        loads += 1;
        if (loads < 2 || disposed) return;
        dispose();
        onLeave();
    });
    return dispose;
}
