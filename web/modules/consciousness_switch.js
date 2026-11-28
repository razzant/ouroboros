// The one switch of consciousness (owner 15.11, decision 5A): every Turn on / Turn off in the app (the Dashboard's
// headline, the chip's Shade, Activity's row) goes through here. Before the turn the product names what changes from
// this install's facts, in the host's words (`GET /api/consciousness?switch=`, consciousness_view.switch_preview): the
// places by name and how many people wrote there lately. With no place to name the turn goes at once; a preview that
// cannot be read never holds the switch (the dialog says so). An ordinary turn sends `/bg` over the command socket.
// After Panic (`preview.afterPanic`, owner 26.11 C01 Stage 1) the current desktop app turns consciousness on—or, when it
// is already on, hands over remaining Panic-stopped tasks for this server run—through one native OS confirmation dialog
// (`pywebview.api.request_panic_consciousness_start`) that may hand Ouroboros that choice. Browsers, phones, headless
// runs and older desktop shells without that bridge method ask as usual and send the plain `/bg start`, which hands over
// no choice (owner 27.11: as in 7.12.1); the handover alone they explain in an alert. Manual Resume stays everywhere.
import { fmt, tr } from './i18n.js';
import { openConfirmDialog } from './confirm_dialog.js';
import { readSwitchPreview } from './consciousness_api.js';
import { shellBridgeApi } from './ui_helpers.js';

/** The switch's two actions, as its buttons and its dialog name them. */
export const SWITCH_ACTIONS = Object.freeze({
    on: { act: 'Turn on', keep: 'Keep off', title: 'Turn on consciousness?' },
    off: { act: 'Turn off', keep: 'Keep on', title: 'Turn off consciousness?' },
});

/** A button's words while the preview is read (on a large install the first read takes seconds: no silent pause). */
export const switchReadingWords = () => tr('consciousness.switch.reading', 'Reading…');

/** The button's words for a switch whose state is `enabled` (true | false). */
export const switchActionWords = (enabled) => tr(`consciousness.switch.${enabled ? 'off' : 'on'}`,
    SWITCH_ACTIONS[enabled ? 'off' : 'on'].act);

/** «87 people» for a place's row in the dialog's details (the week the host's sentence names), «at least 2000 people»
 *  when the place's read was capped; '' when not read. */
const peopleWords = (n, atLeast = false) => (n === null || n === undefined ? ''
    : fmt(atLeast ? (n === 1 ? 'at least {n} person' : 'at least {n} people') : (n === 1 ? '{n} person' : '{n} people'), { n }));

function hasDesktopShell(win, api) {
    if (api) return true;
    try {
        return Boolean(win?.pywebview || (win?.parent && win.parent !== win && win.parent.pywebview));
    } catch {
        return false;
    }
}

/** The dialog for a preview, or null when there is nothing to name (no place concerns the turn). */
export function switchDialog(to, preview) {
    const words = SWITCH_ACTIONS[to];
    if (preview && Array.isArray(preview.places) && !preview.places.length && !preview.afterPanic) return null;
    // One sentence a line (the dialog's paragraph keeps line breaks).
    const body = preview ? preview.lines.join('\n') : tr('consciousness.switch.unread', 'What it would change could not be read.');
    // Every place the turn concerns, the busiest first: the week's people in the short column, the place's name in the
    // wide one (names run long); a Silent one says so beside its name, as it never speaks.
    // A place's own name is authored: the page's translator leaves it as the owner wrote it.
    const rows = (preview?.places || []).map((place) => ({ label: peopleWords(place.people, place.atLeast) || '—', authored: true,
        value: [place.label || place.id, place.state === 'silent' ? tr('places.state.silent', 'Silent') : ''].filter(Boolean).join(' · ') }));
    return {
        title: tr(`consciousness.switch.title.${to}`, words.title),
        body,
        details: rows.length ? { summary: fmt(rows.length === 1 ? 'The place' : 'All {n} places', { n: rows.length }), rows } : null,
        confirmLabel: tr(`consciousness.switch.${to}`, words.act),
        cancelLabel: tr(`consciousness.switch.keep.${to}`, words.keep),
    };
}

/** Ask, when there is something to name, and send the turn; true when the request was sent (never proof it was applied:
 *  Main's reply and the next state read report the outcome). `onRead` is called once the preview is read (or failed),
 *  before any question: the caller's button stops saying «Reading…». */
export async function turnConsciousness(to, { ws, read = readSwitchPreview, ask = openConfirmDialog, onRead, win = globalThis.window } = {}) {
    let preview = null;
    let note = '';
    try { preview = await read(to); } catch { preview = null; }
    onRead?.();
    if (to === 'on' && preview?.afterPanic) {
        const held = preview.afterPanic.why === 'held';
        const title = held
            ? tr('consciousness.switch.title.panic_choice', 'Allow wakes to resume Panic-stopped tasks?')
            : tr('consciousness.switch.title.on', SWITCH_ACTIONS.on.title);
        const api = shellBridgeApi(win);
        if (typeof api?.request_panic_consciousness_start === 'function') {
            let result = null;
            try {
                result = await api.request_panic_consciousness_start();
            } catch (err) {
                result = { submitted: false, error: err?.message || String(err || '') };
            }
            if (result?.submitted === true) return true;
            if (result?.cancelled) return false;
            await ask({
                title,
                body: result?.error || tr('consciousness.switch.panic_failed', 'The request to turn consciousness on could not be sent.'),
                confirmLabel: tr('dialog.ok', 'OK'),
                alert: true,
            });
            return false;
        }
        const older = hasDesktopShell(win, api);
        if (held) {  // only the handover is left to do, and only the current desktop app's own dialog does it
            const why = older
                ? tr('consciousness.switch.older_desktop',
                    'This desktop app shell is older than the running version and cannot confirm this action after Panic. Update the desktop app, or resume individual tasks manually.')
                : tr('consciousness.switch.desktop_only',
                    'Open Ouroboros in the desktop app on that computer to confirm this action after Panic. Here you can resume individual tasks manually.');
            await ask({ title, body: [...preview.lines, why].join('\n'), confirmLabel: tr('dialog.ok', 'OK'), alert: true });
            return false;
        }
        if (preview.afterPanic.choice) {  // the plain Turn on below hands over no choice: say so beside the host's words
            note = older
                ? tr('consciousness.switch.plain_after_panic_older',
                    'This desktop app shell is older than the running version, so turning it on here hands Ouroboros no choice: the tasks stopped by Panic stay stopped until you resume them. Update the desktop app to hand that choice over.')
                : tr('consciousness.switch.plain_after_panic',
                    'Turning it on here hands Ouroboros no choice: the tasks stopped by Panic stay stopped until you resume them.');
        }
    }
    const dialog = switchDialog(to, preview);
    if (dialog && note) dialog.body = `${dialog.body}\n${note}`;
    if (dialog && !(await ask(dialog))) return false;
    if (!ws?.send) return false;
    ws.send({ type: 'command', cmd: `/bg ${to === 'off' ? 'stop' : 'start'}`, ui_control: { name: 'consciousness_toggle' } });
    return true;
}
