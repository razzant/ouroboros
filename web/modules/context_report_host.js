// Beautiful Context's local requests on Dashboard › Logs (DESIGN, Widgets board: "Beautiful Context"): one action in the
// band beside Clear, offered only on a supported Windows WebView2 shell whose DesktopApi exposes
// `open_local_context_report` and reports `shell_info().local_context_archive === true`. The opener receives only a
// state; no archive data reaches this page or any skill. The press answers in the shared action result (ui.css
// .ui-action-row / .ui-result): an acknowledgement ends by itself, an unavailable viewer stays until the next press.
import { setResult } from './ui_primitives.js';

const ANSWERS = new Map([
    ['opened', { text: 'Opened in a separate local window.', tone: 'ok', ends: true }],
    ['already_open', { text: 'The local viewer is already open.', tone: 'muted', ends: true }],
]);
const UNAVAILABLE = { text: 'The local viewer is unavailable in this window.', tone: 'warn' };

export function bindLocalContextReport(mount) {
    if (!mount) return () => {};
    let stopBridge = () => {};
    let result = null;
    let generation = 0;
    const detachBridge = () => {
        const detach = stopBridge;
        stopBridge = () => {};
        detach();
    };
    const dispose = () => {
        ++generation;
        detachBridge();
        setResult(result, {});  // an acknowledgement's timer ends with its line
        mount.replaceChildren();
        mount.hidden = true;
    };
    const mountedGeneration = generation;
    const offerNative = () => {
        if (mountedGeneration !== generation) return;
        const native = window.pywebview?.api;
        if (typeof native?.open_local_context_report !== 'function' || typeof native.shell_info !== 'function') return;
        detachBridge();
        mount.hidden = true;
        const local = document.createElement('button');
        local.type = 'button';
        local.className = 'btn btn-default';
        local.textContent = 'Open recent requests';
        const line = document.createElement('span');
        line.className = 'ui-result';
        line.setAttribute('role', 'status');
        line.setAttribute('aria-live', 'polite');
        result = line;
        local.addEventListener('click', async () => {
            if (mountedGeneration !== generation) return;
            local.disabled = true;
            setResult(line, {});
            let state;
            try { ({ state } = await native.open_local_context_report()); } catch { state = 'unavailable'; }
            if (mountedGeneration !== generation) return;
            local.disabled = false;
            setResult(line, ANSWERS.get(state) || UNAVAILABLE);
        });
        mount.replaceChildren(local, line);
        Promise.resolve().then(() => mountedGeneration === generation && native.shell_info()).then((facts) => {
            if (mountedGeneration === generation && facts?.local_context_archive === true) mount.hidden = false;
        }, () => {});
        return true;
    };
    mount.hidden = true;
    if (!offerNative()) {
        window.addEventListener?.('pywebviewready', offerNative);
        stopBridge = () => window.removeEventListener?.('pywebviewready', offerNative);
    }
    return dispose;
}
