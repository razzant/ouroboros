// Beautiful Context on Widgets (DESIGN, Widgets: "Beautiful Context"): its one button opens the opaque report frame
// (/static/context-report.html) in place below itself and closes it; nothing else on the page touches that frame.
import { onThemeChange } from './theme_palette.js';
import { cardThemeValues } from './card_frames.js';

// Binds the page's button to its mount; returns the close every leave of the page calls.
export function bindContextReport(button, mount) {
    let stopTheme = () => {}, stopBridge = () => {};
    let openGeneration = 0;
    const close = () => {
        ++openGeneration;
        stopTheme();
        stopBridge();
        stopTheme = () => {};  // spent: a repeated close calls nothing, and the closed frame is no longer reachable from here
        mount.replaceChildren();
        mount.hidden = true;
        button.setAttribute('aria-expanded', 'false');
    };
    button.addEventListener('click', () => {
        if (!mount.hidden) return close();
        const frame = document.createElement('iframe');
        frame.title = 'Beautiful Context';
        frame.setAttribute('sandbox', 'allow-scripts');
        frame.referrerPolicy = 'no-referrer';
        // An opaque frame reads no host token: told the palette one way, it answers nothing. Its name carries the theme
        // for the first paint; a message, the theme and its card's colours at load and at each change while open.
        const tell = () => frame.contentWindow?.postMessage(
            { type: 'ouro-context-report-theme', ...cardThemeValues(document, mount) }, '*');
        frame.name = `ouro-theme:${cardThemeValues(document, mount).name}`;
        frame.src = '/static/context-report.html';
        frame.addEventListener('load', tell);
        stopTheme = onThemeChange(tell);
        const mountedGeneration = openGeneration;
        const offerNative = () => {
            if (mountedGeneration !== openGeneration) return;
            const native = window.pywebview?.api;
            if (typeof native?.open_local_context_report !== 'function' || typeof native.shell_info !== 'function') return;
            stopBridge();
            const actions = document.createElement('div');
            actions.className = 'ui-action-row';
            actions.hidden = true;
            const local = document.createElement('button');
            local.type = 'button';
            local.className = 'btn btn-default';
            local.textContent = 'Open recent requests';
            const status = document.createElement('span');
            status.className = 'ui-result';
            status.setAttribute('role', 'status');
            local.addEventListener('click', async () => {
                const generation = openGeneration;
                local.disabled = true;
                status.textContent = '';
                let state;
                try { ({ state } = await native.open_local_context_report()); } catch { state = 'unavailable'; }
                if (generation !== openGeneration) return;
                local.disabled = false;
                status.textContent = state === 'opened' ? 'Opened in a separate local window.'
                    : state === 'already_open' ? 'The local viewer is already open.'
                        : 'The local viewer is unavailable in this window. You can still open a report below.';
            });
            actions.append(local, status);
            mount.insertBefore(actions, frame);
            Promise.resolve().then(() => native.shell_info()).then((facts) => {
                if (mountedGeneration === openGeneration && facts?.local_context_archive === true) actions.hidden = false;
            }, () => {});
        };
        mount.append(frame);
        stopBridge = () => window.removeEventListener('pywebviewready', offerNative);
        window.addEventListener('pywebviewready', offerNative);
        offerNative();
        mount.hidden = false;
        button.setAttribute('aria-expanded', 'true');
    });
    return close;
}
