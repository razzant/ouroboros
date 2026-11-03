/* Client-local appearance: Light / Dark / System, and the shell's layout switch.
   Runs synchronously before CSS to avoid a dark flash and a wrong-shell flash. No
   server setting: the owner's WebView and each browser retain their own choice, so
   one device can differ from another on purpose.

   The store keeps the CHOICE, not the painted theme. An absent value means System
   (the default for a new client); a previously saved 'light' or 'dark' keeps its
   exact old meaning, so an owner who already picked Light is still on Light.

   Where: a page the core serves itself names its installation's appearance scope
   before this script (server_web.py). There the choice is one host-only cookie of
   that scope: the server moving to another port keeps it, another installation on
   the host keeps its own. The origin's older `ouroboros.theme` is adopted only while
   that cookie is absent, and removed once the cookie holds the choice: clearing
   cookies returns every address opened since to System, while an address not opened
   since still carries its older value over once, and a selected interface directory
   or an older build reads System where it was removed. A page without a scope (the
   Telegram Mini App, a selected interface directory, a core whose identity is not
   minted yet) keeps `ouroboros.theme` itself, as before. */
(() => {
    'use strict';
    const KEY = 'ouroboros.theme';
    // Display order is the owner-facing order: Light / Dark / System.
    const CHOICES = ['light', 'dark', 'system'];
    const LABELS = { light: 'Light', dark: 'Dark', system: 'System' };
    const root = document.documentElement;
    const isChoice = (value) => CHOICES.includes(value);

    const scope = document.querySelector?.('meta[name="ouroboros-appearance-scope"]')?.content || '';
    const COOKIE = /^[0-9a-f]{16}$/.test(scope) ? `ouroboros-theme-${scope}` : '';
    // 400 days, the longest a browser keeps a cookie; each page load renews it.
    const COOKIE_TAIL = `; Max-Age=${400 * 24 * 60 * 60}; Path=/; SameSite=Lax`;
    const readCookie = () => {
        try {
            for (const pair of String(document.cookie || '').split(';')) {
                const at = pair.indexOf('=');
                if (at > 0 && pair.slice(0, at).trim() === COOKIE) {
                    const value = pair.slice(at + 1).trim();
                    return isChoice(value) ? value : '';
                }
            }
        } catch { /* this document may not read cookies */ }
        return '';
    };
    // A write counts once it reads back: a client that refuses cookies drops it silently.
    const writeCookie = (value) => {
        try { document.cookie = `${COOKIE}=${value}${COOKIE_TAIL}`; } catch { return false; }
        return readCookie() === value;
    };
    const readOrigin = () => {  // throws where the browser blocks storage
        const saved = localStorage.getItem(KEY);
        return isChoice(saved) ? saved : '';
    };
    const forgetOrigin = () => {
        try { if (localStorage.getItem(KEY) !== null) localStorage.removeItem(KEY); } catch { /* nothing to consume */ }
    };

    // 'cookie' while this page keeps the choice in its installation's cookie, else 'origin'.
    let store = COOKIE && window.navigator?.cookieEnabled !== false ? 'cookie' : 'origin';
    let choice = 'system';
    let storageAvailable = true;
    if (store === 'cookie') {
        const kept = readCookie();
        let older = '';
        if (!kept) { try { older = readOrigin(); } catch { /* blocked: nothing to adopt */ } }
        choice = kept || older || 'system';
        if (kept || older) {
            // Renewed or adopted: the origin's copy is consumed. A client that refuses
            // the cookie keeps its older copy as its store.
            if (writeCookie(choice)) forgetOrigin();
            else if (!kept) store = 'origin';
        }
    } else {
        try { choice = readOrigin() || 'system'; } catch { storageAvailable = false; }
    }

    const query = typeof window.matchMedia === 'function'
        ? window.matchMedia('(prefers-color-scheme: light)')
        : null;
    /* A host can stand in for the OS as the "System" source (the Telegram mini app follows
       Telegram's own colour scheme, which can differ from the OS: host_telegram.js). The
       source answers 'light', 'dark' or '' (defer to the OS query). Dark is the fallback when
       no one reports a preference. */
    let systemSource = null;
    const systemScheme = () => {
        const answer = systemSource ? systemSource() : '';
        if (answer === 'light' || answer === 'dark') return answer;
        return query?.matches ? 'light' : 'dark';
    };
    const resolve = (value) => (value === 'system' ? systemScheme() : value);
    let resolved = resolve(choice);

    const statusText = () => {
        if (!storageAvailable) return 'Appearance applies until this window reloads; this device blocks storage.';
        if (COOKIE && store === 'origin') {
            return 'This browser blocks cookies, so the choice stays with this address and is not kept when the port changes.';
        }
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

    /* The shell's one layout switch (docs/DESIGN.md "Shell"): `docked` keeps the
       sidebar as a column from LAYOUT.dockedMinWidth up; `stack` below it folds the
       sidebar into a drawer and every room into a full-width layer. Written here,
       before CSS, so the first paint is already the right shell, and kept current by
       the window's own media-query change (a resize, a rotation). The number lives
       here once; web/modules/layout.js is the ES-module door everything else reads it
       through (tests/test_shell_layout_static.py). */
    const LAYOUT = Object.freeze({ dockedMinWidth: 720 });
    const layoutQuery = typeof window.matchMedia === 'function'
        ? window.matchMedia(`(min-width: ${LAYOUT.dockedMinWidth}px)`)
        : null;
    // A client that cannot answer the query is a desktop window: docked.
    const resolveLayout = () => (layoutQuery && !layoutQuery.matches ? 'stack' : 'docked');
    let layout = resolveLayout();
    const paintLayout = () => { root.dataset.layout = layout; };
    const onLayoutQuery = () => {
        const next = resolveLayout();
        if (next === layout) return;
        layout = next;
        paintLayout();
        window.dispatchEvent(new CustomEvent('ouro:layout-changed', { detail: { layout } }));
    };
    paintLayout();
    layoutQuery?.addEventListener?.('change', onLayoutQuery);
    window.ouroLayout = Object.freeze({ ...LAYOUT, get layout() { return layout; } });

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
        return changed;
    };

    // Same-origin windows (another tab, the setup frame over the shell) hear a new
    // cookie at once; a page on another port reads it when it is shown or focused.
    // A document that may not open a channel still paints and keeps the choice.
    const syncs = store === 'cookie';
    const channel = (() => {
        try { return syncs && typeof BroadcastChannel === 'function' ? new BroadcastChannel(COOKIE) : null; }
        catch { return null; }
    })();

    const keep = (value) => {
        if (store === 'cookie') {
            if (writeCookie(value)) {
                forgetOrigin();
                try { channel?.postMessage(value); } catch { /* the others read it when shown */ }
                return true;
            }
            store = 'origin';  // refused now: the origin's store is what this page still has
        }
        try { localStorage.setItem(KEY, value); return true; } catch { return false; }
    };

    const choose = (value) => {
        if (!isChoice(value)) return;
        choice = value;
        storageAvailable = keep(value);
        settle();
    };

    // Another window wrote a choice: read this page's own store again, never the
    // event or message that said so. In cookie mode an `ouroboros.theme` event (a
    // page from before this store, or another tab consuming that key) moves nothing.
    const refresh = () => {
        let next;
        if (store === 'cookie') {
            next = readCookie() || 'system';
        } else {
            try { next = readOrigin() || 'system'; storageAvailable = true; }
            catch { storageAvailable = false; syncControls(); return false; }
        }
        // Unchanged, the root's attributes stay untouched: the macOS window observes them.
        if (next === choice) { syncControls(); return false; }
        choice = next;
        return settle();
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
    const onStorage = (event) => { if (event.key === KEY || event.key === null) refresh(); };
    const onFocus = () => { refresh(); };
    const onShown = () => { if (document.visibilityState !== 'hidden') refresh(); };
    const onMessage = () => { refresh(); };
    // Restored from the back-forward cache: the window's page again, and a page on
    // another port may have changed the choice meanwhile.
    const onPageShow = (event) => {
        if (!event.persisted) return;
        if (ownsWindow && clock) pageSince = clock.timeOrigin + clock.now();
        const told = syncs && refresh();  // a repaint already told the frame
        if (ownsWindow && clock && !told) syncNativeFrame();
    };

    const onSystemChange = () => { if (choice === 'system') settle(); };

    const ready = () => syncControls();
    const cleanup = (event) => {
        if (event.persisted) return;
        document.removeEventListener('click', onClick);
        document.removeEventListener('keydown', onKeydown);
        document.removeEventListener('DOMContentLoaded', ready);
        document.removeEventListener('visibilitychange', onShown);
        window.removeEventListener('storage', onStorage);
        window.removeEventListener('focus', onFocus);
        window.removeEventListener('pywebviewready', syncNativeFrame);
        window.removeEventListener('pageshow', onPageShow);
        window.removeEventListener('pagehide', cleanup);
        query?.removeEventListener?.('change', onSystemChange);
        layoutQuery?.removeEventListener?.('change', onLayoutQuery);
        channel?.removeEventListener('message', onMessage);
        channel?.close();
    };
    document.addEventListener('click', onClick);
    document.addEventListener('keydown', onKeydown);
    document.addEventListener('DOMContentLoaded', ready, { once: true });
    window.addEventListener('storage', onStorage);
    if (ownsWindow) window.addEventListener('pywebviewready', syncNativeFrame);
    if (ownsWindow || syncs) window.addEventListener('pageshow', onPageShow);
    if (syncs) {
        window.addEventListener('focus', onFocus);
        document.addEventListener('visibilitychange', onShown);
        channel?.addEventListener('message', onMessage);
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
        /* The host's System source: install (settles at once), returns its release. */
        setSystemSource(source) {
            systemSource = typeof source === 'function' ? source : null;
            settle();
            return () => { if (systemSource === source) { systemSource = null; settle(); } };
        },
        /* The installed source changed its answer (Telegram's themeChanged). */
        refreshSystem: () => { if (choice === 'system') settle(); },
    };
})();

/* Interface size on a touch screen (docs/DESIGN.md "Appearance"; web/modules/interface_size.js owns the setting).
   A phone's own page zoom is its viewport: the meta's layout width (`width = the width at 100 % ÷ scale`,
   `initial-scale = scale`) lays the page out narrower or wider and the screen shows it at the owner's size, with every
   length, query and coordinate still in one CSS pixel (CSS `zoom` on the page instead pushed the composer off screen).
   Written here, before CSS, so the first layout already has the stored size. A host that ignores the meta (a Mini App
   in Telegram for Android, a desktop engine, a frame inside another page) keeps 100 %: a size is kept only once the
   layout width really became the one asked for the first time; a later miss is a layout still moving (a rotation).
   A new screen width (a rotation, a split screen) is read from the visual viewport where it reads true, without a
   flash to 100 %, and the owner's choice reapplied as far as the narrower screen allows. Stored per origin. */
(() => {
    'use strict';
    const KEY = 'ouroboros.interface-size';
    const REFUSED = 'ouroboros.interface-size-refused';  // the user agent of a host that ignored the meta here
    const STEP = 5;
    const MIN = 75;      // smaller, a finger's targets fall far under their 44 px
    const FLOOR = 320;   // never laid out narrower than this many CSS px (WCAG 1.4.10's reflow width)
    const meta = document.querySelector?.('meta[name="viewport"]') || null;
    const original = meta?.getAttribute('content') || '';
    const coarse = typeof window.matchMedia === 'function' && window.matchMedia('(pointer: coarse)').matches;
    const topLevel = (() => { try { return window.top === window; } catch { return false; } })();
    const listeners = new Set();
    let base = 0;       // the layout width at 100 %, in CSS px: the screen's own width when it was measured
    let percent = 100;  // the size in effect
    let chosen = 100;   // the owner's choice; a screen too narrow for it shows the largest size it allows
    let honoured = false;  // the host has laid out a size other than 100 % once
    let settling = 0;   // writes of ours still being laid out
    let recheck = false;  // the window was resized while one of them was
    let watching = true;  // the visual viewport reads as the screen's own width (checked on the first size)
    const agent = () => { try { return String(navigator.userAgent || ''); } catch { return ''; } };
    // A host that ignored the meta is not offered the size again until its user agent changes (an update may honour it).
    const refusedHere = () => { try { return localStorage.getItem(REFUSED) === agent(); } catch { return false; } };
    let state = meta && coarse && topLevel && typeof requestAnimationFrame === 'function' && !refusedHere()
        ? 'pending' : 'unavailable';
    let generation = 0;

    const stored = () => {
        try {
            const value = Number(localStorage.getItem(KEY));
            return Number.isFinite(value) && value > 0 ? value : 100;
        } catch { return 100; }
    };
    const store = (value) => {
        try { if (value === 100) localStorage.removeItem(KEY); else localStorage.setItem(KEY, String(value)); }
        catch { /* the size holds until the page reloads */ }
    };
    const width = () => Number(document.documentElement.clientWidth) || Number(window.innerWidth) || 0;
    const max = () => Math.max(100, Math.floor(((base / FLOOR) * 100) / STEP) * STEP);
    const clamp = (value) => Math.min(max(), Math.max(MIN, Math.round(Number(value) / STEP) * STEP));
    const asked = (value) => Math.round(base / (value / 100));
    const frames = () => new Promise((done) => requestAnimationFrame(() => requestAnimationFrame(done)));
    // The screen's own width in CSS px at 100 %: the visual viewport at its scale, which Chromium and WebKit report as
    // the page's whole scale, so neither a pinch nor the meta moves it.
    const shownWidth = () => {
        const vv = window.visualViewport;
        return vv && Number(vv.scale) > 0 ? Number(vv.width) * Number(vv.scale) : 0;
    };
    const notify = () => { for (const listener of listeners) { try { listener(); } catch { /* one view's fault */ } } };
    const write = (value) => {
        // The scale is the ratio of the two widths, so the screen shows the whole layout width: 390 px at 80 % is
        // 488 px shown at 0.79918, where 0.8 would show 487.5 px of it.
        const scale = value === 100 ? 1 : Number((base / asked(value)).toFixed(6));
        meta.setAttribute('content', value === 100 ? original
            : `width=${asked(value)}, initial-scale=${scale}, viewport-fit=cover`);
        // The touch field guard keeps a field's text at 16 device px whatever the size, and a page laid out wider than
        // its screen keeps its type scale (style.css).
        document.documentElement.style.setProperty('--interface-scale', String(scale));
        document.documentElement.toggleAttribute?.('data-interface-smaller', value < 100);
    };

    /* Apply `value` %, keeping it only if the host laid the page out at the width asked for. */
    const apply = async (value) => {
        if (state === 'unavailable') return state;
        const mine = ++generation;
        const next = clamp(value);
        settling += 1;
        write(next);
        await frames();
        let laidOut = next === 100 || Math.abs(width() - asked(next)) <= 2;
        if (!laidOut && !honoured && mine === generation) {
            await frames();  // a first miss may be a slow first layout or a turn under way: look once more
            laidOut = Math.abs(width() - asked(next)) <= 2;
        }
        settling -= 1;
        if (mine !== generation) return state;
        if (!laidOut && !honoured) {
            write(100);
            percent = 100;
            state = 'unavailable';  // this host ignores the meta: 100 % stays, and the control says so
            try { localStorage.setItem(REFUSED, agent()); } catch { /* it is asked again next time */ }
        } else {
            if (laidOut && next !== 100 && !honoured) {
                honoured = true;
                // An engine whose visual viewport reports only a pinch's scale reads the layout's width here.
                watching = Math.abs(shownWidth() - asked(next)) > 2;
            }
            percent = next;  // a miss now is a layout still moving: the meta stands and the next resize reads it
            state = 'active';
        }
        notify();
        if (recheck) watch();
        return state;
    };

    /* A rotation or a new window width changes the width at 100 %: read it, or measure it at 100 %, then reapply. */
    const remeasure = async (measure = false) => {
        if (state === 'unavailable') return;
        const mine = ++generation;
        const before = base;
        const shown = watching && !measure ? shownWidth() : 0;
        if (shown > 0) {
            base = Math.round(shown);
        } else {
            settling += 1;
            write(100);
            await frames();
            settling -= 1;
            if (mine !== generation) return;
            base = width();
            // An ambiguous reading the measure shows was no new screen came from an engine reporting the layout.
            if (measure && Math.abs(base - before) <= 2) watching = false;
        }
        if (chosen !== 100 || percent !== 100) {
            await apply(chosen);
        } else {
            notify();
            if (recheck) watch();
        }
    };

    /* A split screen or a resized window keeps its orientation: a resize that moved the screen's own width measures
       again, and one that arrives while a write of ours is being laid out is read once it is. */
    const watch = () => {
        if (state === 'unavailable' || !watching) return;
        if (state === 'pending' || settling) { recheck = true; return; }
        recheck = false;
        // A pinch never changes the screen's width, and deep in one the reading is too coarse to trust.
        if (Number(window.visualViewport?.width) < width() - 1) return;
        const now = shownWidth();
        // A reading equal to the layout width at another size is a screen that now matches it or an engine that
        // reports the layout: measure at 100 % to tell them apart.
        if (percent !== 100 && Math.abs(now - width()) <= 2) { remeasure(true); return; }
        if (now && Math.abs(now - base) > 2) remeasure();
    };

    if (state === 'pending') {
        base = width();
        chosen = stored();
        if (chosen === 100) state = 'active';
        else apply(chosen);
        window.matchMedia('(orientation: portrait)')?.addEventListener?.('change', () => { remeasure(); });
        window.addEventListener?.('resize', watch);
    }

    window.ouroViewportScale = {
        get state() { return state; },  // 'active' | 'pending' | 'unavailable'
        get percent() { return percent; },
        get min() { return MIN; },
        get max() { return base ? max() : 100; },
        get step() { return STEP; },
        async set(value) {
            if (state === 'unavailable') return state;
            chosen = clamp(value);  // before the write: a new screen width read meanwhile reapplies this choice
            const next = await apply(chosen);
            if (next === 'active') store(chosen);  // the choice, even when a newer write took the screen first
            return next;
        },
        subscribe(listener) { listeners.add(listener); return () => listeners.delete(listener); },
    };
})();
