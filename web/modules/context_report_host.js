// Beautiful Context on Widgets (DESIGN, Widgets: "Beautiful Context"): its one button opens the opaque report frame
// (/static/context-report.html) in place below itself and closes it; nothing else on the page touches that frame.
import { onThemeChange } from './theme_palette.js';
import { cardThemeValues } from './card_frames.js';

// Binds the page's button to its mount; returns the close every leave of the page calls.
export function bindContextReport(button, mount) {
    let stopTheme = () => {};
    const close = () => {
        stopTheme();
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
        mount.append(frame);
        mount.hidden = false;
        button.setAttribute('aria-expanded', 'true');
    });
    return close;
}
