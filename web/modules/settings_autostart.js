import { apiClient } from './api_client.js';
import { setInlineStatus } from './ui_primitives.js';

/* Startup & background: two immediate host controls, separate from the /api/settings draft.
   Sign-in startup is the OS registration; keep-running is the host's own settings choice
   (the desktop's first close may also set it). Both re-read when Settings opens, because the
   OS or another client can change them meanwhile. */

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

function bindHostToggle({ container, box, status, notes, noun, read, write }) {
    if (!container || !box) return null;
    let destroyed = false;
    let busy = false;
    let generation = 0;

    const paint = ({ state, reason }) => {
        const known = Object.hasOwn(notes, state);
        container.hidden = !known;
        box.checked = state === 'on';
        box.disabled = state === 'unavailable';
        const note = known ? (reason || notes[state]) : '';
        setInlineStatus(status, note, note ? 'warn' : 'muted');
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
            setInlineStatus(status, `Could not read the ${noun}: ${error.message}`, 'danger');
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
                setInlineStatus(status, `Could not change the ${noun}: ${error.message}. Current host state could not be read; reopen Settings to retry.`, 'danger');
            } else {
                paint(snapshot);
                setInlineStatus(status, `Could not change the ${noun}: ${error.message}`, 'danger');
            }
        }
    };

    box.addEventListener('change', onChange);
    return {
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
    const controls = [
        bindHostToggle({
            container: section, box: section.querySelector('[data-autostart-toggle]'),
            status: section.querySelector('[data-autostart-status]'), notes: STARTUP_NOTES, noun: 'host startup entry',
            read: () => apiClient.desktopAutostart(), write: (enabled) => apiClient.setDesktopAutostart(enabled),
        }),
        bindHostToggle({
            container: section.querySelector('[data-background-row]'), box: section.querySelector('[data-background-toggle]'),
            status: section.querySelector('[data-background-status]'), notes: BACKGROUND_NOTES, noun: 'background setting',
            read: () => apiClient.desktopBackground(), write: (enabled) => apiClient.setDesktopBackground(enabled),
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
