import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import vm from 'node:vm';

const source = fs.readFileSync(new URL('../theme.js', import.meta.url), 'utf8');

/* One host's cookie jar, shared by the pages a test opens on its ports: what a page
   writes through `document.cookie` (attributes kept for the test) and reads back. */
function cookieJar({ refuse = false } = {}) {
    const cookies = new Map();
    return {
        cookies,
        get header() { return [...cookies].map(([name, { value }]) => `${name}=${value}`).join('; '); },
        write(text) {
            if (refuse) return;  // a client that refuses cookies drops the write silently
            const [pair, ...attributes] = text.split(';').map((part) => part.trim());
            const at = pair.indexOf('=');
            cookies.set(pair.slice(0, at), { value: pair.slice(at + 1), attributes });
        },
    };
}

/* Same-origin windows' BroadcastChannel: a message reaches every other open channel of its name. */
function channelHub() {
    const open = new Set();
    class FakeChannel {
        constructor(name) { this.name = name; this.listeners = new Set(); this.closed = false; open.add(this); }
        addEventListener(type, fn) { if (type === 'message') this.listeners.add(fn); }
        removeEventListener(type, fn) { this.listeners.delete(fn); }
        postMessage(data) {
            for (const other of open) {
                if (other !== this && other.name === this.name && !other.closed) other.listeners.forEach((fn) => fn({ data }));
            }
        }
        close() { this.closed = true; open.delete(this); }
    }
    return { FakeChannel, open };
}

/* A fake just wide enough for the controller: it renders its own buttons, so the
   harness has to hand back real child nodes rather than one stubbed element. */
function boot({ saved = null, storage = true, systemLight = false, media = true, topLevel = false, framed = false, bridge = null, clock = null,
    scope = '', jar = null, cookieEnabled = undefined, hub = null } = {}) {
    const handlers = new Map();
    const events = [];
    const rootWrites = [];
    // The root's attribute writes are counted: the macOS window observes them, so a no-op must not write.
    const root = { dataset: new Proxy({}, { set(target, key, value) { rootWrites.push(key); target[key] = value; return true; } }) };
    const status = { textContent: '' };
    let host;

    const makeButton = () => {
        const classes = new Set();
        const el = {
            dataset: {}, attrs: {}, classes, tabIndex: null, textContent: '', type: '', focused: false,
            set className(value) { value.split(' ').forEach((name) => classes.add(name)); },
            get className() { return [...classes].join(' '); },
            setAttribute(key, value) { el.attrs[key] = value; },
            classList: { toggle: (name, on) => (on ? classes.add(name) : classes.delete(name)) },
            focus() { el.focused = true; },
            closest: (selector) => (selector === '[data-theme-control]' ? host : el),
        };
        return el;
    };

    host = {
        children: [], attrs: {}, classes: new Set(),
        classList: { add: (...names) => names.forEach((name) => host.classes.add(name)) },
        setAttribute(key, value) { host.attrs[key] = value; },
        replaceChildren(...nodes) { host.children = nodes; },
        querySelector: (selector) => host.children.find((child) => child.dataset.themeChoice
            === (selector.match(/"(.*)"/)?.[1] ?? child.dataset.themeChoice)) || null,
        querySelectorAll: () => host.children,
    };

    const scopeMeta = scope ? { content: scope } : null;
    const document = {
        documentElement: root,
        visibilityState: 'visible',
        createElement: makeButton,
        // The page the core serves names its installation's scope before theme.js runs.
        querySelector: (selector) => (selector === 'meta[name="ouroboros-appearance-scope"]' ? scopeMeta : null),
        querySelectorAll: (selector) => (selector === '[data-theme-control]' ? [host] : [status]),
        addEventListener: (key, fn) => handlers.set(key, fn),
        removeEventListener: (key) => handlers.delete(key),
        get cookie() { return jar ? jar.header : ''; },
        set cookie(text) { jar?.write(text); },
    };
    const query = {
        matches: systemLight,
        addEventListener: (key, fn) => handlers.set(`media:${key}`, fn),
        removeEventListener: (key) => handlers.delete(`media:${key}`),
    };
    // The layout switch asks its own query (layout.test.js covers it); keep its listener apart.
    const layoutQuery = {
        matches: true,
        addEventListener: (key, fn) => handlers.set(`layout:${key}`, fn),
        removeEventListener: (key) => handlers.delete(`layout:${key}`),
    };
    const window = {
        addEventListener: (key, fn) => handlers.set(key, fn),
        removeEventListener: (key) => handlers.delete(key),
        dispatchEvent: (event) => events.push(event),
        ...(media ? { matchMedia: (q) => (q.includes('min-width') ? layoutQuery : query) } : {}),
        ...(bridge ? { pywebview: { api: bridge } } : {}),
        ...(clock ? { performance: clock } : {}),
        ...(cookieEnabled === undefined ? {} : { navigator: { cookieEnabled } }),
    };
    // A desktop page is its window's top document; the onboarding overlay is a framed copy.
    if (topLevel) window.top = window;
    if (framed) window.top = {};
    let store = saved;
    const localStorage = {
        getItem() { if (!storage) throw new Error('disabled'); return store; },
        setItem(key, value) { if (!storage) throw new Error('disabled'); store = value; },
        removeItem() { if (!storage) throw new Error('disabled'); store = null; },
    };
    vm.runInNewContext(source, {
        document, window, localStorage,
        CustomEvent: class { constructor(type, init) { this.type = type; this.detail = init?.detail; } },
        ...(hub ? { BroadcastChannel: hub.FakeChannel } : {}),
    });
    const button = (value) => host.children.find((child) => child.dataset.themeChoice === value);
    return {
        root, host, status, handlers, events, query, window, document, rootWrites,
        saved: () => store,
        setSaved: (value) => { store = value; },
        buttons: () => host.children.map((child) => child.dataset.themeChoice),
        click: (value) => handlers.get('click')({ target: { closest: () => button(value) } }),
        key: (value, key) => handlers.get('keydown')({
            key, target: { closest: () => button(value) }, preventDefault() { this.prevented = true; },
        }),
        button,
    };
}

test('a new client defaults to System and follows the OS', () => {
    assert.equal(boot().root.dataset.theme, 'dark');
    assert.equal(boot().root.dataset.themeChoice, 'system');
    assert.equal(boot({ systemLight: true }).root.dataset.theme, 'light');
    // No OS signal at all is a real client, not a crash: System means Dark there.
    assert.equal(boot({ media: false }).root.dataset.theme, 'dark');
    assert.match(boot({ media: false }).status.textContent, /no OS appearance/);
});

test('an already saved explicit choice keeps its exact old meaning', () => {
    // The owner picked Light before Light/Dark/System existed; the upgrade must
    // not quietly re-read that as "follow the OS".
    const pinned = boot({ saved: 'light', systemLight: false });
    assert.equal(pinned.root.dataset.theme, 'light');
    assert.equal(pinned.root.dataset.themeChoice, 'light');
    assert.equal(boot({ saved: 'dark', systemLight: true }).root.dataset.theme, 'dark');
    assert.equal(boot({ saved: 'bogus' }).root.dataset.themeChoice, 'system');
});

test('the control renders Light / Dark / System and marks exactly one', () => {
    const s = boot({ saved: 'dark' });
    assert.deepEqual(s.buttons(), ['light', 'dark', 'system']);
    assert.equal(s.host.attrs.role, 'radiogroup');
    assert.deepEqual(s.host.children.map((b) => b.attrs['aria-checked']), ['false', 'true', 'false']);
    assert.deepEqual(s.host.children.map((b) => b.tabIndex), [-1, 0, -1]);
    assert.equal(s.host.children.every((b) => b.attrs.role === 'radio'), true);
});

test('choosing persists and repaints; arrows move through the group', () => {
    const s = boot({ saved: 'dark' });
    s.click('light');
    assert.equal(s.saved(), 'light');
    assert.equal(s.root.dataset.theme, 'light');
    assert.equal(s.events.length, 1);
    s.key('light', 'ArrowRight');
    assert.equal(s.saved(), 'dark');
    assert.equal(s.button('dark').focused, true);
    s.key('dark', 'End');
    assert.equal(s.saved(), 'system');
    assert.equal(boot({ saved: s.saved() }).root.dataset.themeChoice, 'system');
});

test('a choice that repaints nothing does not churn mounted views', () => {
    // Dark -> System on a dark OS changes the stored choice and the pressed
    // button, but not one pixel — charts and diagrams must not be told to redraw.
    const s = boot({ saved: 'dark', systemLight: false });
    s.click('system');
    assert.equal(s.saved(), 'system');
    assert.equal(s.root.dataset.themeChoice, 'system');
    assert.equal(s.root.dataset.theme, 'dark');
    assert.equal(s.events.length, 0);
    assert.equal(s.button('system').attrs['aria-checked'], 'true');
});

test('System tracks a later OS switch, a pinned choice ignores it', () => {
    const following = boot({ saved: 'system', systemLight: false });
    following.query.matches = true;
    following.handlers.get('media:change')();
    assert.equal(following.root.dataset.theme, 'light');
    assert.equal(following.events.length, 1);
    const pinned = boot({ saved: 'dark', systemLight: false });
    pinned.query.matches = true;
    pinned.handlers.get('media:change')();
    assert.equal(pinned.root.dataset.theme, 'dark');
    assert.equal(pinned.events.length, 0);
});

test('a host can stand in for the OS as the System source, and release it', () => {
    // Telegram's colour scheme can differ from the OS's: host_telegram.js installs it through this one hook.
    const s = boot({ saved: 'system', systemLight: true });
    assert.equal(s.root.dataset.theme, 'light');
    let scheme = 'dark';
    const release = s.window.ouroTheme.setSystemSource(() => scheme);
    assert.equal(s.root.dataset.theme, 'dark');
    assert.equal(s.events.length, 1);
    scheme = 'light';
    s.window.ouroTheme.refreshSystem();
    assert.equal(s.root.dataset.theme, 'light');
    scheme = '';  // the host defers: the OS query answers again
    s.query.matches = false;
    s.window.ouroTheme.refreshSystem();
    assert.equal(s.root.dataset.theme, 'dark');
    release();
    s.query.matches = true;
    s.handlers.get('media:change')();
    assert.equal(s.root.dataset.theme, 'light');
    // A pinned choice ignores the host, like it ignores the OS.
    const pinned = boot({ saved: 'light', systemLight: false });
    pinned.window.ouroTheme.setSystemSource(() => 'dark');
    assert.equal(pinned.root.dataset.theme, 'light');
});

test('unavailable storage stays usable and says so', () => {
    const s = boot({ storage: false });
    s.click('light');
    assert.equal(s.root.dataset.theme, 'light');
    assert.match(s.status.textContent, /until this window reloads/);
});

test('another window of the same client, and lifecycle cleanup', () => {
    const s = boot({ saved: 'dark' });
    // The handler re-reads storage rather than trusting event.newValue, so a
    // sibling window's clear() is covered by the same path as its setItem().
    s.setSaved('light');
    s.handlers.get('storage')({ key: 'ouroboros.theme' });
    assert.equal(s.root.dataset.theme, 'light');
    assert.equal(s.button('light').attrs['aria-checked'], 'true');
    s.setSaved(null);
    s.handlers.get('storage')({ key: 'ouroboros.theme' });
    assert.equal(s.root.dataset.themeChoice, 'system');
    // A bfcache-restorable hide must not tear the controller down.
    s.handlers.get('pagehide')({ persisted: true });
    assert.ok(s.handlers.has('click'));
    s.handlers.get('pagehide')({ persisted: false });
    assert.equal(s.handlers.size, 0);
});

test('mount() paints a control injected after boot', () => {
    // Settings is rendered long after theme.js ran, so the published mounter is
    // the only thing that can fill its control.
    const s = boot({ saved: 'light' });
    s.host.children = [];
    assert.equal(s.buttons().length, 0);
    s.window.ouroTheme.mount();
    assert.deepEqual(s.buttons(), ['light', 'dark', 'system']);
    assert.equal(s.button('light').attrs['aria-checked'], 'true');
});

/* The installation's scope, as the core names it on its own pages (server_web.appearance_scope). */
const SCOPE = '0123456789abcdef';
const OTHER = 'fedcba9876543210';
const COOKIE = `ouroboros-theme-${SCOPE}`;
const kept = (jar, name = COOKIE) => jar.cookies.get(name)?.value;

test('a served scope keeps the choice in one host cookie that every port of the installation reads', () => {
    const jar = cookieJar();
    const first = boot({ scope: SCOPE, jar });
    assert.equal(first.root.dataset.themeChoice, 'system');
    assert.equal(jar.cookies.size, 0, 'an untouched client writes nothing');
    first.click('light');
    assert.equal(kept(jar), 'light');
    assert.deepEqual(jar.cookies.get(COOKIE).attributes, ['Max-Age=34560000', 'Path=/', 'SameSite=Lax']);
    assert.equal(first.saved(), null, 'the origin store holds no second copy');
    // The same installation after a port change: another origin, the same host's cookie.
    const moved = boot({ scope: SCOPE, jar });
    assert.equal(moved.root.dataset.theme, 'light');
    assert.equal(moved.root.dataset.themeChoice, 'light');
    // Another installation on the host keeps its own.
    const other = boot({ scope: OTHER, jar });
    assert.equal(other.root.dataset.themeChoice, 'system');
    other.click('dark');
    assert.equal(kept(jar), 'light');
    assert.equal(kept(jar, `ouroboros-theme-${OTHER}`), 'dark');
});

test("the origin's older choice is adopted only into an absent cookie, then consumed", () => {
    const jar = cookieJar();
    const upgraded = boot({ scope: SCOPE, jar, saved: 'dark', systemLight: true });
    assert.equal(upgraded.root.dataset.theme, 'dark');
    assert.equal(kept(jar), 'dark');
    assert.equal(upgraded.saved(), null);
    // A port whose origin still holds its own older value: the cookie wins and the stale copy goes.
    const stale = boot({ scope: SCOPE, jar, saved: 'light' });
    assert.equal(stale.root.dataset.themeChoice, 'dark');
    assert.equal(stale.saved(), null);
    // Clearing cookies returns to System: no older copy is left to come back.
    jar.cookies.clear();
    const cleared = boot({ scope: SCOPE, jar, saved: stale.saved() });
    assert.equal(cleared.root.dataset.themeChoice, 'system');
    assert.equal(jar.cookies.size, 0);
});

test('an explicit System is a stored choice: saved without a repaint and never replaced by an older one', () => {
    const jar = cookieJar();
    const s = boot({ scope: SCOPE, jar, saved: 'dark', systemLight: false });
    s.click('system');  // a dark OS: not one colour changes, and the choice must still be kept
    assert.equal(kept(jar), 'system');
    assert.equal(s.events.length, 0);
    assert.equal(s.root.dataset.themeChoice, 'system');
    // Another port's origin still holds Dark from before the cookie existed.
    const elsewhere = boot({ scope: SCOPE, jar, saved: 'dark', systemLight: true });
    assert.equal(elsewhere.root.dataset.themeChoice, 'system');
    assert.equal(elsewhere.root.dataset.theme, 'light');
    assert.equal(elsewhere.saved(), null);
});

test("an older page's ouroboros.theme event never overrides the cookie; showing the page reads it again", () => {
    const jar = cookieJar();
    const s = boot({ scope: SCOPE, jar, saved: 'light' });
    assert.equal(kept(jar), 'light');
    // A tab still running the theme.js from before this store writes the origin's old key.
    s.setSaved('dark');
    s.handlers.get('storage')({ key: 'ouroboros.theme' });
    assert.equal(s.root.dataset.themeChoice, 'light');
    // A page on another port chose: focus, a shown tab or a back-forward restore reads the cookie.
    jar.write(`${COOKIE}=dark`);
    s.handlers.get('focus')({});
    assert.equal(s.root.dataset.theme, 'dark');
    jar.write(`${COOKIE}=light`);
    s.document.visibilityState = 'hidden';
    s.handlers.get('visibilitychange')({});
    assert.equal(s.root.dataset.theme, 'dark', 'a hidden page reads it once shown');
    s.document.visibilityState = 'visible';
    s.handlers.get('visibilitychange')({});
    assert.equal(s.root.dataset.theme, 'light');
    jar.write(`${COOKIE}=system`);
    s.handlers.get('pageshow')({ persisted: false });
    assert.equal(s.root.dataset.themeChoice, 'light');
    s.handlers.get('pageshow')({ persisted: true });
    assert.equal(s.root.dataset.themeChoice, 'system');
    // Nothing new: the root, which the macOS window observes, is not written again.
    const writes = s.rootWrites.length;
    s.handlers.get('focus')({});
    s.handlers.get('storage')({ key: null });
    assert.equal(s.rootWrites.length, writes);
});

test('same-origin windows hear a choice through the channel and read it from the cookie', () => {
    const jar = cookieJar();
    const hub = channelHub();
    const shell = boot({ scope: SCOPE, jar, hub, topLevel: true });
    const setup = boot({ scope: SCOPE, jar, hub, framed: true });  // the setup frame over the shell
    const other = boot({ scope: OTHER, jar, hub });  // another installation once served on this origin
    setup.click('light');
    assert.equal(shell.root.dataset.theme, 'light');
    assert.equal(shell.events.length, 1);
    assert.equal(other.root.dataset.themeChoice, 'system');
    assert.equal(hub.open.size, 3);
    // The page that leaves stops listening and closes its channel.
    shell.handlers.get('pagehide')({ persisted: false });
    assert.equal(shell.handlers.size, 0);
    assert.equal(hub.open.size, 2);
    setup.click('dark');
    assert.equal(shell.root.dataset.theme, 'light');
});

test('a client that refuses the cookie keeps the origin store and says a port change loses it', () => {
    const s = boot({ scope: SCOPE, jar: cookieJar({ refuse: true }), saved: 'dark' });
    assert.equal(s.root.dataset.themeChoice, 'dark');
    assert.equal(s.saved(), 'dark', 'the older copy stays where it is still the only store');
    s.click('light');
    assert.equal(s.saved(), 'light');
    assert.match(s.status.textContent, /blocks cookies.*not kept when the port changes/);
    // Refusing storage too, the choice lasts until a reload, and that is what it says.
    const blocked = boot({ scope: SCOPE, jar: cookieJar({ refuse: true }), storage: false });
    assert.equal(blocked.status.textContent, '', 'nothing is known to fail before a write');
    blocked.click('light');
    assert.equal(blocked.root.dataset.theme, 'light');
    assert.match(blocked.status.textContent, /until this window reloads/);
    // A browser that says it keeps no cookies is not asked to, and says so before any choice.
    const jar = cookieJar();
    const declined = boot({ scope: SCOPE, jar, cookieEnabled: false, saved: 'light' });
    assert.match(declined.status.textContent, /not kept when the port changes/);
    declined.click('dark');
    assert.equal(declined.saved(), 'dark');
    assert.equal(jar.cookies.size, 0);
    assert.match(boot({ scope: SCOPE, cookieEnabled: false, storage: false }).status.textContent, /until this window reloads/);
    // Without a scope nothing could follow the port, so nothing is said about it.
    assert.equal(boot({ jar: cookieJar({ refuse: true }), saved: 'dark' }).status.textContent, '');
});

test('a document that may not open a channel still paints, keeps and hears another port', () => {
    const jar = cookieJar();
    const refusing = { FakeChannel: class { constructor() { throw new Error('SecurityError'); } } };
    const s = boot({ scope: SCOPE, jar, hub: refusing, saved: 'dark' });
    assert.equal(s.root.dataset.theme, 'dark');
    s.click('light');
    assert.equal(kept(jar), 'light');
    jar.write(`${COOKIE}=dark`);
    s.handlers.get('focus')({});
    assert.equal(s.root.dataset.theme, 'dark');
    s.handlers.get('pagehide')({ persisted: false });
    assert.equal(s.handlers.size, 0);
    // A channel that refuses to post (closed under it) never costs the choice.
    const hub = channelHub();
    const posting = boot({ scope: SCOPE, jar, hub });
    for (const channel of hub.open) channel.postMessage = () => { throw new Error('InvalidStateError'); };
    posting.click('system');
    assert.equal(kept(jar), 'system');
    assert.equal(posting.status.textContent, '');
});

test('only a well-formed scope selects the cookie', () => {
    const jar = cookieJar();
    for (const scope of ['ABCDEF0123456789', '0123', '0123456789abcdef0', '0123456789abcdeg']) {
        const s = boot({ scope, jar, saved: 'light' });
        s.click('dark');
        assert.equal(s.saved(), 'dark', scope);
    }
    assert.equal(jar.cookies.size, 0);
});

test('both first-party documents load the controller before CSS', () => {
    for (const name of ['index.html', 'onboarding_template.html']) {
        const html = fs.readFileSync(new URL(`../${name}`, import.meta.url), 'utf8');
        assert.ok(html.indexOf('/static/theme.js') < html.indexOf('/static/ui.css'));
    }
    const index = fs.readFileSync(new URL('../index.html', import.meta.url), 'utf8');
    // The sidebar button moved to Settings -> Appearance; onboarding keeps a
    // control because it runs before Settings exists.
    assert.doesNotMatch(index, /data-theme-toggle|data-theme-control/);
    assert.match(fs.readFileSync(new URL('../onboarding_template.html', import.meta.url), 'utf8'), /data-theme-control/);
    assert.match(fs.readFileSync(new URL('../modules/settings_ui.js', import.meta.url), 'utf8'), /data-theme-control/);
});

/* The desktop caption follows the painted palette through the window's own bridge. */
function recorder({ reject = false, raise = false } = {}) {
    const calls = [];
    return {
        calls,
        set_native_appearance(...args) {
            calls.push(args);
            if (raise) throw new Error('older shell');
            return reject ? Promise.reject(new Error('refused')) : Promise.resolve({ ok: true });
        },
    };
}

/* A document's own clock: when it began (performance.timeOrigin) and how long ago. */
const documentClock = (timeOrigin, elapsed = 0) => ({ timeOrigin, now: () => elapsed });

test('the top-level desktop page tells its own window each painted palette, newest last', () => {
    const bridge = recorder();
    const s = boot({ saved: 'dark', systemLight: false, topLevel: true, bridge, clock: documentClock(1791100000000.5) });
    s.click('light');
    s.click('system');  // a dark OS: the painted palette moves back to dark
    s.click('dark');    // the same paint: nothing to tell the frame
    // Every call names the page by when it began, so a reload's calls outrank this page's.
    assert.deepEqual(bridge.calls, [['dark', 1791100000000.5, 1], ['light', 1791100000000.5, 2],
        ['dark', 1791100000000.5, 3]]);
});

test('a page the back-forward cache restores speaks from its restore time', () => {
    const bridge = recorder();
    const clock = documentClock(1791100000000.5);
    const s = boot({ saved: 'light', topLevel: true, bridge, clock });
    s.handlers.get('pagehide')({ persisted: true });  // kept in the cache: still listening
    s.handlers.get('pageshow')({ persisted: false });  // an ordinary load announces nothing more
    clock.now = () => 60000;
    s.handlers.get('pageshow')({ persisted: true });
    assert.deepEqual(bridge.calls, [['light', 1791100000000.5, 1], ['light', 1791100060000.5, 2]]);
});

test('a bridge injected after load hears the palette on pywebviewready, and cleanup releases it', () => {
    const s = boot({ saved: 'light', topLevel: true });
    assert.ok(s.handlers.has('pywebviewready'));
    const bridge = recorder();
    s.window.pywebview = { api: bridge };
    s.handlers.get('pywebviewready')({ type: 'pywebviewready' });
    assert.deepEqual(bridge.calls.map(([theme, , sequence]) => [theme, sequence]), [['light', 1]]);
    s.handlers.get('pagehide')({ persisted: false });
    assert.equal(s.handlers.size, 0);
});

test("a framed copy never drives its parent's frame", () => {
    const bridge = recorder();
    const s = boot({ saved: 'dark', framed: true, bridge });
    s.click('light');
    assert.equal(bridge.calls.length, 0);
    assert.equal(s.handlers.has('pywebviewready'), false);
    assert.equal(s.handlers.has('pageshow'), false);
    assert.equal(s.root.dataset.theme, 'light');
});

test('a refusing or older bridge never stops the page repainting', async () => {
    const refusing = boot({ saved: 'dark', topLevel: true, bridge: recorder({ reject: true }) });
    refusing.click('light');
    const older = boot({ saved: 'dark', topLevel: true, bridge: recorder({ raise: true }) });
    older.click('light');
    await new Promise((resolve) => setImmediate(resolve));  // a rejection would surface here
    assert.equal(refusing.root.dataset.theme, 'light');
    assert.equal(older.root.dataset.theme, 'light');
});
