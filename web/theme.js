/* Client-local appearance: Light / Dark / System. Runs synchronously before CSS
   to avoid a dark flash. No server setting: the owner's WebView and each browser
   retain their own choice, so one device can differ from another on purpose.

   `ouroboros.theme` stores the CHOICE, not the painted theme. An absent value
   means System (the default for a new client); a previously saved 'light' or
   'dark' keeps its exact old meaning, so an owner who already picked Light is
   still on Light after this upgrade. */
(() => {
    'use strict';
    const KEY = 'ouroboros.theme';
    // Display order is the owner-facing order: Light / Dark / System.
    const CHOICES = ['light', 'dark', 'system'];
    const LABELS = { light: 'Light', dark: 'Dark', system: 'System' };
    const root = document.documentElement;
    const isChoice = (value) => CHOICES.includes(value);

    let choice = 'system';
    let storageAvailable = true;
    try {
        const saved = localStorage.getItem(KEY);
        if (isChoice(saved)) choice = saved;
    } catch { storageAvailable = false; }

    const query = typeof window.matchMedia === 'function'
        ? window.matchMedia('(prefers-color-scheme: light)')
        : null;
    // Dark is the fallback when the client reports no OS preference.
    const resolve = (value) => (value === 'system' ? (query?.matches ? 'light' : 'dark') : value);
    let resolved = resolve(choice);

    const statusText = () => {
        if (!storageAvailable) return 'Appearance applies until this window reloads; this device blocks storage.';
        if (choice === 'system' && !query) return 'This client reports no OS appearance, so System uses Dark.';
        return '';
    };

    const fill = (host) => {
        host.classList.add('theme-choice-group');
        host.setAttribute('role', 'radiogroup');
        host.replaceChildren(...CHOICES.map((value) => {
            const button = document.createElement('button');
            button.type = 'button';
            button.className = 'theme-choice';
            button.dataset.themeChoice = value;
            button.setAttribute('role', 'radio');
            button.textContent = LABELS[value];
            return button;
        }));
    };

    const controls = () => Array.from(document.querySelectorAll('[data-theme-control]'));

    const syncControls = () => {
        for (const host of controls()) {
            if (!host.querySelector('[data-theme-choice]')) fill(host);
            for (const button of host.querySelectorAll('[data-theme-choice]')) {
                const active = button.dataset.themeChoice === choice;
                button.classList.toggle('active', active);
                button.setAttribute('aria-checked', String(active));
                // Roving tabindex: the group is one tab stop, arrows move inside it.
                button.tabIndex = active ? 0 : -1;
            }
        }
        for (const status of document.querySelectorAll('[data-theme-status]')) {
            status.textContent = statusText();
        }
    };

    const paint = () => {
        root.dataset.theme = resolved;
        root.dataset.themeChoice = choice;
        syncControls();
    };

    /* The desktop window's ordinary system caption takes the light or dark tint of
       the painted palette (launcher_appearance.py). Only a top-level document speaks
       for its window: a framed copy (the onboarding overlay) never drives its
       parent's frame. pywebview runs each bridge call on its own thread, so calls
       can arrive out of order, even a replaced page's after its successor's: each
       carries when this document became the window's page (its time origin on the
       browser's monotonic clock, renewed when the back-forward cache restores it)
       and its count there, and the launcher keeps only a request newer by that
       pair. A browser, or a desktop app built before the bridge, has none. */
    const ownsWindow = (() => { try { return window.top === window; } catch { return false; } })();
    const clock = window.performance;
    let pageSince = clock?.timeOrigin;
    let nativeSequence = 0;
    const syncNativeFrame = () => {
        const api = ownsWindow ? window.pywebview?.api : null;
        if (typeof api?.set_native_appearance !== 'function') return;
        nativeSequence += 1;
        try {
            Promise.resolve(api.set_native_appearance(resolved, pageSince, nativeSequence)).catch(() => {});
        } catch { /* the frame keeps its tint; the page is unaffected */ }
    };
    // Restored from the back-forward cache after a newer page: this one is the window's again.
    const onPageShow = (event) => {
        if (!event.persisted || !clock) return;
        pageSince = clock.timeOrigin + clock.now();
        syncNativeFrame();
    };

    // Re-resolve, repaint, and tell mounted views only when the PAINTED theme
    // moved. Switching Dark -> System on a dark OS changes the choice without
    // changing a single colour, and must not churn charts or diagrams.
    const settle = () => {
        const next = resolve(choice);
        const changed = next !== resolved;
        resolved = next;
        paint();
        if (changed) {
            syncNativeFrame();
            window.dispatchEvent(new CustomEvent('ouro:theme-changed', { detail: { theme: resolved, choice } }));
        }
    };

    const choose = (value) => {
        if (!isChoice(value)) return;
        choice = value;
        try { localStorage.setItem(KEY, choice); storageAvailable = true; }
        catch { storageAvailable = false; }
        settle();
    };

    paint();
    // The first paint announces nothing; the bridge, injected after load, hears it then.
    syncNativeFrame();

    const onClick = (event) => {
        const button = event.target.closest?.('[data-theme-choice]');
        if (button?.closest('[data-theme-control]')) choose(button.dataset.themeChoice);
    };

    const onKeydown = (event) => {
        const button = event.target.closest?.('[data-theme-choice]');
        if (!button?.closest('[data-theme-control]')) return;
        const step = { ArrowRight: 1, ArrowDown: 1, ArrowLeft: -1, ArrowUp: -1 }[event.key];
        let next = '';
        if (step) next = CHOICES[(CHOICES.indexOf(choice) + step + CHOICES.length) % CHOICES.length];
        else if (event.key === 'Home') next = CHOICES[0];
        else if (event.key === 'End') next = CHOICES[CHOICES.length - 1];
        else return;
        event.preventDefault();
        choose(next);
        button.closest('[data-theme-control]')?.querySelector(`[data-theme-choice="${next}"]`)?.focus();
    };

    // Another window of the same client (browser tab, or the desktop shell's
    // onboarding window next to the main one) wrote a new choice.
    const onStorage = (event) => {
        if (event.key !== KEY && event.key !== null) return;
        let saved = 'system';
        try {
            const raw = localStorage.getItem(KEY);
            if (isChoice(raw)) saved = raw;
            storageAvailable = true;
        } catch {
            storageAvailable = false;
            syncControls();
            return;
        }
        choice = saved;
        settle();
    };

    const onSystemChange = () => { if (choice === 'system') settle(); };

    const ready = () => syncControls();
    const cleanup = (event) => {
        if (event.persisted) return;
        document.removeEventListener('click', onClick);
        document.removeEventListener('keydown', onKeydown);
        document.removeEventListener('DOMContentLoaded', ready);
        window.removeEventListener('storage', onStorage);
        window.removeEventListener('pywebviewready', syncNativeFrame);
        window.removeEventListener('pageshow', onPageShow);
        window.removeEventListener('pagehide', cleanup);
        query?.removeEventListener?.('change', onSystemChange);
    };
    document.addEventListener('click', onClick);
    document.addEventListener('keydown', onKeydown);
    document.addEventListener('DOMContentLoaded', ready, { once: true });
    window.addEventListener('storage', onStorage);
    if (ownsWindow) {
        window.addEventListener('pywebviewready', syncNativeFrame);
        window.addEventListener('pageshow', onPageShow);
    }
    window.addEventListener('pagehide', cleanup);
    query?.addEventListener?.('change', onSystemChange);

    /* Views rendered after boot (the Settings page is injected by app.js) call
       mount() once their markup is in the document; the delegated handlers above
       already work, mount() only paints the current state into fresh buttons. */
    window.ouroTheme = {
        get choice() { return choice; },
        get theme() { return resolved; },
        set: choose,
        mount: syncControls,
    };
})();
