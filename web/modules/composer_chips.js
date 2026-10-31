// Contributed chips on the composer's capsule row (docs/DESIGN.md "Composer owner controls", "Skills in the page"):
// each `composer.chips` contribution is a neutral capsule after the host's capsules with the skill's label and the
// value its `source` route answers (`{text, tone?, title?}`: the first 24 characters show, the rest and the title
// describe it); a press does the chip's action on this room's draft. One feed for the page reads each source on its
// own refresh (15 seconds at least), only while a room showing chips is on screen and the page is visible; after a
// failed read it backs off and the chip shows a muted "—" with the reason in its description. A chip the owner hid is
// never read. The row stays one line: the chips take the room the host capsules leave, and those that do not fit fold
// into one "+N" chip whose list holds them.
import { fetchJson } from './api_client.js';
import { fmt } from './i18n.js';
import { draftPort, getContributions, runContributionAction } from './contributions.js';
import { setChipOverflow } from './composer_plus.js';

const MIN_REFRESH_S = 15;
const BACKOFF_STEPS = 4;  // a source that keeps failing is read at most 2^4 of its refreshes apart
const SHOWN_CHARS = 24;
const TONES = new Set(['ok', 'warn', 'bad', 'muted']);

/** A source's answer as its chip shows it. */
export function chipValue(reply) {
    const full = typeof reply?.text === 'string' ? reply.text.trim() : '';
    const chars = Array.from(full);
    return {
        text: chars.length > SHOWN_CHARS ? `${chars.slice(0, SHOWN_CHARS - 1).join('')}…` : full,
        full,
        tone: TONES.has(reply?.tone) ? reply.tone : '',
        title: typeof reply?.title === 'string' ? reply.title.trim() : '',
    };
}

/** What a chip says now: its value's words, tone and description, or a muted "—" with the reason it has none. */
export function chipFace({ item, value, error }) {
    if (error) {
        return { text: '—', tone: 'muted',
            title: fmt('{label} could not be read: {reason}', { label: item.label, reason: error }) };
    }
    return { text: value?.text || '', tone: value?.tone || '',
        title: value?.title || (value && value.full !== value.text ? value.full : '') || item.title || '' };
}

/**
 * The page's reader of chip sources over `store` (contributions.js). `watch()` says a room shows chips on screen and
 * returns its release; reading runs only while one does and the page is visible. `read`, `doc`, `timers` and `now`
 * are seams for tests.
 */
export function createChipFeed({ store = getContributions(), read = (url, init) => fetchJson(url, { cache: 'no-store', ...init }),
    doc = globalThis.document, timers = globalThis, now = () => Date.now() } = {}) {
    const entries = new Map();  // contribution key -> { item, value, error, failures, due, timer, reading }
    const listeners = new Set();
    let viewers = 0;

    const active = () => viewers > 0 && doc?.visibilityState !== 'hidden';
    const changed = () => {
        for (const listener of [...listeners]) {
            try { listener(); } catch (error) { console.error('chip listener failed', error); }
        }
    };
    const period = (entry) => Math.max(MIN_REFRESH_S, Number(entry.item.source?.refresh_s) || 0) * 1000
        * 2 ** Math.min(entry.failures, BACKOFF_STEPS);

    function plan(entry) {
        timers.clearTimeout(entry.timer);
        entry.timer = 0;
        if (!active() || entry.reading) return;
        entry.timer = timers.setTimeout(() => {
            entry.timer = 0;
            void poll(entry);
        }, Math.max(0, entry.due - now()));
    }

    async function poll(entry) {
        if (!active() || entries.get(entry.item.key) !== entry) return;
        const reading = new AbortController();
        entry.reading = reading;
        try {
            entry.value = chipValue(await read(entry.item.source.url, { signal: reading.signal }));
            entry.error = '';
            entry.failures = 0;
        } catch (error) {
            if (reading.signal.aborted) return;
            entry.failures += 1;
            entry.error = String(error?.message || error || 'unavailable');
        } finally {
            if (entry.reading === reading) entry.reading = null;
        }
        if (entries.get(entry.item.key) !== entry) return;
        entry.due = now() + period(entry);
        changed();
        plan(entry);
    }

    function forget(entry) {
        timers.clearTimeout(entry.timer);
        entry.reading?.abort();
        entries.delete(entry.item.key);
    }

    function sync() {
        const live = new Map(store.byPlacement('composer.chips').filter((item) => item.source?.url)
            .map((item) => [item.key, item]));
        for (const entry of [...entries.values()]) {
            const item = live.get(entry.item.key);
            if (!item || item.source.url !== entry.item.source.url || item.revision !== entry.item.revision) forget(entry);
            else entry.item = item;
        }
        for (const [key, item] of live) {
            if (!entries.has(key)) entries.set(key, { item, value: null, error: '', failures: 0, due: 0, timer: 0, reading: null });
        }
        for (const entry of entries.values()) plan(entry);
        changed();
    }

    const resume = () => { for (const entry of entries.values()) plan(entry); };
    doc?.addEventListener?.('visibilitychange', resume);
    const offStore = store.onChange(sync);
    sync();

    return {
        /** The chips in the owner's arrangement, each `{item, value, error}`. */
        chips: () => store.byPlacement('composer.chips').map((item) => entries.get(item.key)).filter(Boolean),
        onChange(listener) {
            listeners.add(listener);
            return () => listeners.delete(listener);
        },
        watch() {
            viewers += 1;
            resume();
            let released = false;
            return () => {
                if (released) return;
                released = true;
                viewers -= 1;
                resume();
            };
        },
        destroy() {
            offStore();
            doc?.removeEventListener?.('visibilitychange', resume);
            for (const entry of [...entries.values()]) forget(entry);
            listeners.clear();
        },
    };
}

let shared = null;

/** The page's one chip feed. */
export function getChipFeed() {
    if (!shared) shared = createChipFeed();
    return shared;
}

function observeVisibility(element, onChange) {
    if (typeof IntersectionObserver !== 'function') {
        onChange(true);
        return () => {};
    }
    const observer = new IntersectionObserver((records) => onChange(records.some((record) => record.isIntersecting)));
    observer.observe(element);
    return () => observer.disconnect();
}

function observeSize(element, onResize) {
    if (typeof ResizeObserver !== 'function') return () => {};
    const later = typeof requestAnimationFrame === 'function' ? requestAnimationFrame : (fn) => setTimeout(fn, 0);
    const cancel = typeof cancelAnimationFrame === 'function' ? cancelAnimationFrame : clearTimeout;
    let frame = 0;
    // A fit changes the slot's own size: the next measurement waits for the next frame, so it never loops in one.
    const observer = new ResizeObserver(() => {
        cancel(frame);
        frame = later(onResize);
    });
    observer.observe(element);
    return () => { cancel(frame); observer.disconnect(); };
}

const keepFocus = (event) => event.preventDefault();

/** The room `slot` keeps before its `first` chip, on its inline-start side (the right in a right-to-left page). */
function leadOf(slot, first) {
    const box = slot.getBoundingClientRect?.();
    const chip = first?.getBoundingClientRect?.();
    if (!box || !chip) return 0;
    const rtl = slot.ownerDocument?.defaultView?.getComputedStyle?.(slot)?.direction === 'rtl';
    return Math.max(0, (rtl ? box.right - chip.right : chip.left - box.left) || 0);
}

/**
 * One room's chips in `composer.chipsSlot`. `menu` is the composer's menu factory (composer.js `createPlusMenu`), used
 * for the "+N" list. Returns the disposer.
 */
export function bindComposerChips({ composer, menu, feed = getChipFeed(), notice = undefined, run = runContributionAction,
    observe = observeVisibility, resize = observeSize }) {
    const slot = composer?.chipsSlot;
    if (!slot) return () => {};
    const doc = slot.ownerDocument;
    const draft = draftPort(() => ({ composer }));
    const buttons = new Map();  // contribution key -> chip button
    let entries = [];
    let folded = [];
    let release = null;
    let showing = false;  // the slot shows these chips (it is left alone otherwise)

    const act = (key) => {
        const entry = entries.find((candidate) => candidate.item.key === key);
        if (entry) void run(entry.item, { draft, notice });
    };

    const more = doc.createElement('button');
    more.type = 'button';
    more.className = 'chat-composer-chip chat-chip-more ui-hit-area';
    more.setAttribute('aria-haspopup', 'menu');
    more.setAttribute('aria-expanded', 'false');
    more.hidden = true;
    slot.append(more);
    // The list behind "+N" is the composer's own menu sheet (keyboard, pointer and focus rules of "+").
    const list = menu({ button: more });

    function chip(key) {
        const button = doc.createElement('button');
        button.type = 'button';
        button.className = 'chat-composer-chip ui-hit-area';
        button.dataset.chipKey = key;
        for (const part of ['name', 'dot', 'value']) {
            const span = doc.createElement('span');
            span.className = `chat-chip-${part}`;
            if (part === 'dot') span.setAttribute('aria-hidden', 'true');
            button.append(span);
        }
        // A press keeps the focus where it is (a phone keeps its keyboard), like "+".
        button.addEventListener('mousedown', keepFocus);
        button.addEventListener('click', () => act(key));
        return button;
    }

    function paint(button, entry) {
        const face = chipFace(entry);
        const [name, , value] = button.children;
        name.textContent = entry.item.label;
        value.textContent = face.text;
        value.hidden = !face.text;
        button.dataset.tone = face.tone;
        button.title = face.title;
        // The name and the value are separate words to a screen reader too.
        button.setAttribute('aria-label', face.text ? `${entry.item.label} ${face.text}` : entry.item.label);
    }

    /** Show every chip that fits beside the host capsules; the rest fold into "+N". */
    function fit() {
        const shown = entries.map((entry) => buttons.get(entry.item.key));
        for (const button of shown) button.hidden = false;
        more.hidden = true;
        delete slot.dataset.folded;
        // The slot's own room before its first chip (style.css: the row's gap, carried inside the slot).
        const lead = shown.length ? leadOf(slot, shown[0]) : 0;
        let keep = shown.length;
        if (shown.length && slot.scrollWidth > slot.clientWidth + 1) {
            slot.dataset.folded = '1';  // folded, the slot takes all the room the capsules leave
            more.hidden = false;
            more.textContent = `+${shown.length}`;
            const gap = Number.parseFloat(doc.defaultView?.getComputedStyle?.(slot).columnGap) || 0;
            const room = slot.clientWidth - lead - more.offsetWidth;
            let used = 0;
            keep = 0;
            for (const button of shown) {
                const width = button.offsetWidth + gap;
                if (used + width > room) break;
                used += width;
                keep += 1;
            }
        }
        shown.forEach((button, index) => { button.hidden = index >= keep; });
        folded = entries.slice(keep);
        // No room even for "+N" (a phone's row, where ☰ leads the capsules): the chips are listed in "+" instead.
        const roomless = folded.length > 0 && !keep && slot.clientWidth - lead + 0.5 < more.offsetWidth;
        more.hidden = !folded.length || roomless;
        if (roomless) slot.dataset.roomless = '1';
        else delete slot.dataset.roomless;
        setChipOverflow(composer, roomless ? folded.map((entry) => {
            const face = chipFace(entry);
            return { key: entry.item.key, skill: entry.item.skill, skill_label: entry.item.skill_label,
                label: face.text ? `${entry.item.label} ${face.text}` : entry.item.label, title: face.title,
                onSelect: () => act(entry.item.key) };
        }) : []);
        if (!folded.length) delete slot.dataset.folded;
        more.textContent = `+${folded.length}`;
        more.setAttribute('aria-label', fmt('{count} more', { count: folded.length }));
        more.title = folded.map((entry) => entry.item.label).join(', ');
        list.setSections([{ key: 'chips', items: folded.map((entry) => {
            const face = chipFace(entry);
            return { key: entry.item.key, label: face.text ? `${entry.item.label} ${face.text}` : entry.item.label,
                title: face.title, onSelect: () => act(entry.item.key) };
        }) }]);
    }

    function render() {
        entries = feed.chips();
        const keys = new Set(entries.map((entry) => entry.item.key));
        for (const [key, button] of buttons) {
            if (!keys.has(key)) { button.remove(); buttons.delete(key); }
        }
        for (const entry of entries) {
            if (!buttons.has(entry.item.key)) buttons.set(entry.item.key, chip(entry.item.key));
            const button = buttons.get(entry.item.key);
            paint(button, entry);
            slot.insertBefore(button, more);
        }
        if (entries.length || showing) slot.hidden = !entries.length;
        showing = entries.length > 0;
        fit();
    }

    const offFeed = feed.onChange(render);
    const stopSeeing = observe(slot, (visible) => {
        if (visible && !release) release = feed.watch();
        else if (!visible && release) { release(); release = null; }
    });
    const stopSizing = resize(slot, fit);
    render();

    return () => {
        offFeed();
        stopSeeing();
        stopSizing();
        release?.();
        release = null;
        list.destroy();
        setChipOverflow(composer, []);
        for (const button of buttons.values()) button.remove();
        buttons.clear();
        more.remove();
        if (showing) slot.hidden = true;
        delete slot.dataset.folded;
    };
}

