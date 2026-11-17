// The one switch of consciousness (owner 15.11, decision 5A): every Turn on / Turn off in the app (the Dashboard's
// headline, the chip's Shade, Activity's row) goes through here. Before the turn the product names what changes from
// this install's facts, in the host's words (`GET /api/consciousness?switch=`, consciousness_view.switch_preview): the
// places by name and how many people wrote there lately. With no place to name the turn goes at once; a preview that
// cannot be read never holds the switch (the dialog says so). The command is the same `/bg` every surface sends.
import { fmt, tr } from './i18n.js';
import { openConfirmDialog } from './confirm_dialog.js';
import { readSwitchPreview } from './consciousness_api.js';

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

/** The dialog for a preview, or null when there is nothing to name (no place concerns the turn). */
export function switchDialog(to, preview) {
    const words = SWITCH_ACTIONS[to];
    if (preview && Array.isArray(preview.places) && !preview.places.length) return null;
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

/** Ask, when there is something to name, and send `/bg start|stop`; true when the command was sent. `onRead` is called
 *  once the preview is read (or failed), before any question: the caller's button stops saying «Reading…». */
export async function turnConsciousness(to, { ws, read = readSwitchPreview, ask = openConfirmDialog, onRead } = {}) {
    let preview = null;
    try { preview = await read(to); } catch { preview = null; }
    onRead?.();
    const dialog = switchDialog(to, preview);
    if (dialog && !(await ask(dialog))) return false;
    if (!ws?.send) return false;
    ws.send({ type: 'command', cmd: `/bg ${to === 'off' ? 'stop' : 'start'}`, ui_control: { name: 'consciousness_toggle' } });
    return true;
}
