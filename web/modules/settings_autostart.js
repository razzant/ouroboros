import { apiClient } from './api_client.js';
import { setInlineStatus } from './ui_primitives.js';

/* Startup & background: two immediate host controls, separate from the /api/settings draft.
   Sign-in startup is the OS registration; keep-running is the host's own settings choice
   (the desktop's first close may also set it). Both re-read when Settings opens, because the
   OS or another client can change them meanwhile. A host that cannot offer a control says so
   as a plain note (a fact about the host, not a problem); a state the owner can change is a
   warning; a failed read or write is an error (docs/DESIGN.md §3-§5). */

const STARTUP_NOTES = {
    unavailable: 'Sign-in startup is unavailable on this host.',
    on: '',
    off: '',
    other_copy: 'This host starts Ouroboros at sign-in from a different entry (another copy, or one set up by hand). Turn this on to start this copy instead.',
    disabled_by_os: 'Turned off in the host operating system. Turn this on to enable it, or check the host’s startup settings if it stays disabled.',
};
const BACKGROUND_NOTES = {
    unavailable: 'Keeping Ouroboros running after its window closes is unavailable on this host.',
    on: '',
    off: '',
};

function paintNote(status, text) {
    setInlineStatus(status, text, 'muted');
    if (status?.dataset) delete status.dataset.tone;  // no status dot: a note, not a status
}

function bindHostToggle({ container, box, status, notes, noun, read, write, onPaint = () => {} }) {
    if (!container || !box) return null;
    let destroyed = false;
    let busy = false;
    let generation = 0;
    let shown = null;  // what the row says now: { state, note } once a known state is painted

    const paint = ({ state, reason }) => {
        const known = Object.hasOwn(notes, state);
        container.hidden = !known;
        box.checked = state === 'on';
        box.disabled = state === 'unavailable';
        const note = known ? (reason || notes[state]) : '';
        if (state === 'unavailable') paintNote(status, note);
        else setInlineStatus(status, note, note ? 'warn' : 'muted');
        shown = known ? { state, note } : null;
        onPaint();
    };
    const fail = (message) => {
        setInlineStatus(status, message, 'danger');
        shown = null;
        onPaint();
    };

    const refresh = async () => {
        if (busy || destroyed) return;
        const current = ++generation;
        try {
            const snapshot = await read();
            if (!destroyed && !busy && current === generation) paint(snapshot);
        } catch (error) {
            // Shown even before any state is known: a lasting read failure stays explained.
            if (destroyed || busy || current !== generation) return;
            container.hidden = false;
            box.disabled = true;
            fail(`Could not read the ${noun}: ${error.message}`);
        }
    };

    const onChange = async () => {
        if (busy || destroyed) return;
        const wanted = box.checked;
        busy = true;
        generation += 1; // a read that started before the click must not repaint over it
        box.disabled = true;
        setInlineStatus(status, '', 'muted');
        try {
            const snapshot = await write(wanted);
            busy = false;
            if (!destroyed) paint(snapshot);
        } catch (error) {
            // A refusal may land between two writes: show what the host now holds.
            let snapshot;
            try { snapshot = await read(); } catch { /* current state is unknown */ }
            busy = false;
            if (destroyed) return;
            if (snapshot === undefined) {
                box.checked = !wanted; // last observed value, not a claim about the current host state
                box.disabled = true;
                fail(`Could not change the ${noun}: ${error.message}. Current host state could not be read; reopen Settings to retry.`);
            } else {
                paint(snapshot);
                fail(`Could not change the ${noun}: ${error.message}`);
            }
        }
    };

    box.addEventListener('change', onChange);
    return {
        box,
        status,
        shown: () => shown,
        refresh,
        dispose: () => {
            destroyed = true;
            box.removeEventListener('change', onChange);
        },
    };
}

export function bindAutostartControl(page) {
    const section = page.querySelector('[data-autostart-settings]');
    if (!section) return () => {};
    // Two rows that cannot be used for the same reason say it once, after both, for both.
    const shared = section.querySelector('[data-autostart-shared-note]');
    let controls = [];
    const shareNote = () => {
        const rows = controls.map((control) => control.shown());
        const same = rows.length === 2 && rows.every((row) => row?.state === 'unavailable' && row.note)
            && rows[0].note === rows[1].note;
        if (shared) {
            shared.hidden = !same;
            shared.textContent = same ? rows[0].note : '';
        }
        for (const control of controls) {
            if (control.status) control.status.hidden = Boolean(same && shared);
            if (same && shared) control.box.setAttribute?.('aria-describedby', shared.id);
            else control.box.removeAttribute?.('aria-describedby');
        }
    };
    controls = [
        bindHostToggle({
            container: section, box: section.querySelector('[data-autostart-toggle]'),
            status: section.querySelector('[data-autostart-status]'), notes: STARTUP_NOTES, noun: 'host startup entry',
            read: () => apiClient.desktopAutostart(), write: (enabled) => apiClient.setDesktopAutostart(enabled),
            onPaint: () => shareNote(),
        }),
        bindHostToggle({
            container: section.querySelector('[data-background-row]'), box: section.querySelector('[data-background-toggle]'),
            status: section.querySelector('[data-background-status]'), notes: BACKGROUND_NOTES, noun: 'background setting',
            read: () => apiClient.desktopBackground(), write: (enabled) => apiClient.setDesktopBackground(enabled),
            onPaint: () => shareNote(),
        }),
    ].filter(Boolean);
    if (!controls.length) return () => {};

    const refreshAll = () => controls.forEach((control) => { void control.refresh(); });
    const onPageShown = (event) => {
        if (event.detail?.page === 'settings') refreshAll();
    };
    const dispose = () => {
        controls.forEach((control) => control.dispose());
        window.removeEventListener('ouro:page-shown', onPageShown);
        window.removeEventListener('pagehide', onPageHide);
    };
    const onPageHide = (event) => {
        if (!event.persisted) dispose();
    };

    window.addEventListener('ouro:page-shown', onPageShown);
    window.addEventListener('pagehide', onPageHide);
    refreshAll();
    return dispose;
}
