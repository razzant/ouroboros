// Beautiful Context's local requests on Dashboard › Logs (DESIGN, Widgets board: "Beautiful Context"): one action in the
// band beside Clear, offered only on a supported Windows WebView2 shell whose DesktopApi exposes
// `open_local_context_report` and reports `shell_info().local_context_archive === true`. The shell's answer, yes or no,
// is final for the binding; a read that failed is asked again by `retry`, which Logs calls at each real showing (no
// timer, and never the opener). The opener receives only a state; no archive data reaches this page or any skill. The
// shared toast keeps the answer out of the filter band's measure: an acknowledgement ends by itself, an unavailable
// viewer stays until dismissal or the next press.
import { showToast } from './toast.js';
import { SAVE_ACK_MS } from './ui_primitives.js';

const ANSWERS = new Map([
    ['opened', { text: 'Opened in a separate local window.', tone: 'ok', ttl: SAVE_ACK_MS }],
    ['already_open', { text: 'The local viewer is already open.', tone: 'muted', ttl: SAVE_ACK_MS }],
]);
const UNAVAILABLE = { text: 'The local viewer is unavailable in this window.', tone: 'warn', ttl: 0 };

export function bindLocalContextReport(mount) {
    if (!mount) return { dispose() {}, retry() {} };
    let stopBridge = () => {};
    let notice = null;
    let generation = 0;
    let native = null;     // the bridge the action was offered on
    let asking = false;    // one capability read at a time
    let answered = false;  // the shell said yes or no
    const detachBridge = () => {
        const detach = stopBridge;
        stopBridge = () => {};
        detach();
    };
    const clearNotice = () => {
        notice?.dispose();
        notice = null;
    };
    const dispose = () => {
        ++generation;
        detachBridge();
        clearNotice();
        mount.replaceChildren();
        mount.hidden = true;
    };
    const mountedGeneration = generation;
    const askShell = () => {
        asking = true;
        Promise.resolve().then(() => mountedGeneration === generation && native.shell_info()).then((facts) => {
            if (mountedGeneration !== generation) return;
            asking = false;
            if (!facts || typeof facts !== 'object') return;  // a failed read, like a rejected one, is no answer
            answered = true;
            if (facts.local_context_archive === true) mount.hidden = false;
        }, () => { asking = false; });
    };
    const offerNative = () => {
        if (mountedGeneration !== generation) return;
        const bridge = window.pywebview?.api;
        if (typeof bridge?.open_local_context_report !== 'function' || typeof bridge.shell_info !== 'function') return;
        native = bridge;
        detachBridge();
        mount.hidden = true;
        const local = document.createElement('button');
        local.type = 'button';
        local.className = 'btn btn-default';
        local.textContent = 'Open recent requests';
        local.addEventListener('click', async () => {
            if (mountedGeneration !== generation) return;
            local.disabled = true;
            clearNotice();
            let state;
            try { ({ state } = await native.open_local_context_report()); } catch { state = 'unavailable'; }
            if (mountedGeneration !== generation) return;
            local.disabled = false;
            const { text, tone, ttl } = ANSWERS.get(state) || UNAVAILABLE;
            notice = showToast(text, tone, { ttl });
        });
        mount.replaceChildren(local);
        askShell();
        return true;
    };
    // The one button stays; only a read that failed is asked again, and never while another is in flight.
    const retry = () => {
        if (native && !asking && !answered && mountedGeneration === generation) askShell();
    };
    mount.hidden = true;
    if (!offerNative()) {
        window.addEventListener?.('pywebviewready', offerNative);
        stopBridge = () => window.removeEventListener?.('pywebviewready', offerNative);
    }
    return { dispose, retry };
}
