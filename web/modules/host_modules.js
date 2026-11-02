// Skills' trusted page code (docs/CREATING_SKILLS.md "Page modules", CLIENT_CONTRACT §11): every live contribution
// with `render.kind: "host"` is imported from its revision URL and mounted with one small, versioned context, never
// the composer or the feed (placements are the way into those). A module that fails to load, throws, rejects or does
// not start within the host's module bound is off for this page: its providers are unregistered, one notice names
// its skill, and it stays off until the page reloads or the skill's revision changes. A module is disposed when its
// skill leaves or changes revision, on this page's Panic and when the page goes away.
//
// Recovery never depends on the modules: each start is written to the start ledger (holds.js) before the import,
// and a module whose last start with this revision did not settle is not imported again by itself. One notice names
// the held modules with "Start it"; a new revision of the skill starts as usual.
import { getHolds, START_SETTLE_MS } from './holds.js';
import { fmt, tr } from './i18n.js';
import { showToast } from './toast.js';
import { boundedText } from './utils.js';
import { WIDGET_REQUEST_TIMEOUT_MS } from './widget_job.js';

export const HOST_MODULE_API = 1;
const LOOPBACK_HOSTS = new Set(['localhost', '127.0.0.1', '::1', '[::1]']);

/** Where the page runs, as a module may need it: a secure context, and this computer (a loopback host). */
export function originFacts(win = globalThis.window) {
    const host = String(win?.location?.hostname || '').toLowerCase();
    return { secure: win?.isSecureContext === true, loopback: LOOPBACK_HOSTS.has(host) || host.endsWith('.localhost') };
}

/** A page module's key in the start ledger; its revision is the ledger's `rev`. */
export const moduleHoldKey = (item) => `module:${item.key}`;

/**
 * The page-module runtime over a contributions store (contributions.js). `draft` is the visible room's draft port,
 * `notice(text, tone, options)` a toast, `holds` the start ledger; `importModule`, `mountTimeoutMs` and
 * `settleMs` are seams for tests.
 */
export function createHostModules({ store, draft = null, win = globalThis.window, doc = globalThis.document,
    importModule = (url) => import(url), notice = (text, tone, options) => showToast(text, tone, options),
    holds = getHolds(), mountTimeoutMs = WIDGET_REQUEST_TIMEOUT_MS, settleMs = START_SETTLE_MS } = {}) {
    const mounts = new Map();  // contribution key -> { item, controller, status, stage, handle, settle }
    const providers = new Map();  // contribution key -> { entry, record }
    const providerListeners = new Set();
    const disposers = [];
    const announced = new Set();  // `key@revision` of held modules a notice already named
    let mode = 'idle';  // idle → running → stopped (Panic, unload, destroy)

    const live = (rec) => mounts.get(rec.item.key) === rec && (rec.status === 'mounting' || rec.status === 'mounted');

    function notifyProviders() {
        for (const listener of [...providerListeners]) {
            try { listener(); } catch (error) { console.error('provider listener failed', error); }
        }
    }

    function letGo(handle) {
        try {
            const result = handle?.dispose?.();
            if (typeof result?.then === 'function') result.then(null, (error) => console.warn('page module dispose failed', error));
        } catch (error) { console.warn('page module dispose failed', error); }
    }

    function retire(rec) {
        if (mounts.get(rec.item.key) === rec) mounts.delete(rec.item.key);
        const handle = rec.handle;
        rec.status = 'retired';
        rec.handle = null;
        rec.controller.abort();
        rec.settle();  // this page outlived the start: it is not a hold
        let dropped = false;
        for (const [key, provider] of providers) {
            if (provider.record === rec) { providers.delete(key); dropped = true; }
        }
        if (dropped) notifyProviders();
        if (handle) letGo(handle);
    }

    function fail(rec, error) {
        if (!live(rec)) return;  // retired while starting: its end is not a fault
        retire(rec);
        const { item } = rec;
        const reason = String(error?.message || error || 'unknown error');
        store.markFault(item, reason);
        const skill = item.skill_label || item.skill;
        const name = item.label && item.label !== skill ? `${item.label} (${skill})` : skill;
        notice(fmt('{name} stopped working: {reason}. It stays off until you reload or update the skill.', { name, reason }),
            'warn', { ttl: 0 });
    }

    function registerProvider(rec, kind, impl) {
        const { item } = rec;
        if (item.provides !== kind || item.runtime !== 'client') {
            throw new Error(`${item.key} declares no client ${String(kind)} provider to register`);
        }
        if (!impl || typeof impl !== 'object') throw new TypeError('providers.register(kind, impl) needs the provider object');
        if (!live(rec)) return () => {};
        const entry = Object.freeze({ key: item.key, skill: item.skill, skill_label: item.skill_label, label: item.label,
            kind, destination: item.destination || '', languages: item.languages || [], impl });
        providers.set(item.key, { entry, record: rec });
        notifyProviders();
        return () => {
            if (providers.get(item.key)?.entry !== entry) return;
            providers.delete(item.key);
            notifyProviders();
        };
    }

    // Exactly these facts and capabilities; one a future version adds is absent here, never a silent stub.
    function contextFor(rec) {
        const { item, controller } = rec;
        const root = doc?.documentElement;
        return Object.freeze({
            api: HOST_MODULE_API,
            skill: item.skill,
            key: item.key,
            signal: controller.signal,
            providers: Object.freeze({ register: (kind, impl) => registerProvider(rec, kind, impl) }),
            notice(text) {
                if (live(rec)) notice(`${item.skill_label || item.skill}: ${boundedText(String(text ?? ''), 300)}`, 'info');
            },
            draft: Object.freeze({ insert: (text) => live(rec) && Boolean(draft?.insert(String(text ?? ''))) }),
            theme: Object.freeze({
                get dark() { return root?.dataset?.theme !== 'light'; },
                get reducedMotion() {
                    try { return Boolean(win?.matchMedia?.('(prefers-reduced-motion: reduce)')?.matches); } catch { return false; }
                },
            }),
            origin: Object.freeze(originFacts(win)),
        });
    }

    function mount(item) {
        // The ledger entry stands before the module's first byte runs and goes a second after its start settled.
        const rec = { item, controller: new AbortController(), status: 'mounting', stage: 'import', handle: null,
            settle: holds.starting(moduleHoldKey(item), item.revision) };
        mounts.set(item.key, rec);
        let timer = 0;
        const starting = Promise.resolve().then(async () => {
            const module = await importModule(item.render.url);
            rec.stage = 'mount';
            if (typeof module?.mount !== 'function') throw new Error('the module exports no mount(ctx)');
            return module.mount(contextFor(rec));
        });
        const bound = new Promise((_resolve, reject) => {
            timer = setTimeout(() => reject(new Error(tr('host_modules.timeout', 'it did not start in time'))), mountTimeoutMs);
        });
        return Promise.race([starting, bound]).then((handle) => {
            if (live(rec)) { rec.status = 'mounted'; rec.handle = handle || null; }
            else letGo(handle);  // retired while starting: let its late handle go at once
        }, async (error) => {
            starting.then(letGo, () => {});  // a start that outlives the bound is let go when it lands
            // A revision replaced since the read answers 404: read again, and the next revision mounts instead.
            if (rec.stage === 'import' && live(rec)) await store.refresh();
            fail(rec, error);
        }).finally(() => {
            clearTimeout(timer);
            setTimeout(rec.settle, settleMs);
        });
    }

    // One notice per reason for the modules this page holds back, each named once per revision.
    function announce(held) {
        const unnamed = held.filter((item) => !announced.has(`${item.key}@${item.revision}`));
        unnamed.forEach((item) => announced.add(`${item.key}@${item.revision}`));
        for (const reason of ['start', 'all']) {
            const group = unnamed.filter((item) => holds.reason(moduleHoldKey(item), item.revision) === reason);
            if (!group.length) continue;
            const labels = [...new Set(group.map((item) => item.skill_label || item.skill))];
            const names = labels.join(', ');
            const text = reason === 'all' ? fmt('{names} did not start: this page was opened with everything held.', { names })
                : labels.length > 1 ? fmt('{names} did not start: the page stopped while they were starting last time.', { names })
                    : fmt('{names} did not start: the page stopped while it was starting last time.', { names });
            notice(text, 'warn', { ttl: 0, action: { label: tr('holds.start_it', 'Start it'), run() {
                group.forEach((item) => holds.release(moduleHoldKey(item)));
                sync();
            } } });
        }
    }

    function sync() {
        if (mode !== 'running') return;
        const wanted = new Map(store.hostEntries().map((item) => [item.key, item]));
        for (const rec of [...mounts.values()]) {
            const item = wanted.get(rec.item.key);
            if (!item || item.revision !== rec.item.revision || item.render.url !== rec.item.render.url) retire(rec);
        }
        const fresh = [...wanted.values()].filter((item) => !mounts.has(item.key) && !store.fault(item.key));
        const held = fresh.filter((item) => holds.reason(moduleHoldKey(item), item.revision));
        fresh.filter((item) => !held.includes(item)).forEach(mount);
        announce(held);
    }

    /** Panic, unload or teardown: every module is disposed and none mounts again in this page. */
    function stop() {
        if (mode === 'stopped') return;
        mode = 'stopped';
        for (const rec of [...mounts.values()]) retire(rec);
    }

    function onPageHide(event) {
        if (!event?.persisted) stop();
    }

    const api = {
        /** Follow the store: mount every live module the ledger does not hold. */
        start() {
            if (mode !== 'idle') return api;
            mode = 'running';
            disposers.push(store.onChange(sync));
            win?.addEventListener?.('ouro:panic', stop);
            win?.addEventListener?.('pagehide', onPageHide);
            disposers.push(() => {
                win?.removeEventListener?.('ouro:panic', stop);
                win?.removeEventListener?.('pagehide', onPageHide);
            });
            sync();
            return api;
        },
        /** `idle`, `running` or `stopped`. */
        mode: () => mode,
        /** Keys of the modules mounted now. */
        mounted: () => [...mounts.values()].filter((rec) => rec.status === 'mounted').map((rec) => rec.item.key),
        /** Providers page modules registered (`ctx.providers.register`), of one kind (`stt`, `tts`). */
        clientProviders: (kind) => [...providers.values()].map(({ entry }) => entry).filter((entry) => entry.kind === kind),
        onProvidersChange(listener) {
            providerListeners.add(listener);
            return () => providerListeners.delete(listener);
        },
        stop,
        destroy() {
            stop();
            for (const dispose of disposers.splice(0)) dispose();
            providerListeners.clear();
        },
    };
    return api;
}

let shared = null;

/** Start the page's one runtime (app.js, once); later calls return it. */
export function startHostModules(options) {
    if (!shared) shared = createHostModules(options).start();
    return shared;
}

/** The page's runtime, or null before app.js started it. */
export const getHostModules = () => shared;
