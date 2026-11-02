/**
 * Consciousness in a room's bar (docs/DESIGN.md "Page header"): one compact chip at the head of the bar's end group,
 * and its Shade. The chip says the state at a glance — the house brain and the next wake (`14:20`, `since 14:20`, `due`), `Off`, or `Evolving`
 * while an Evolve campaign runs, even with consciousness off — and carries one status mark (status_mark.js, DESIGN
 * "Status marks"): the moving dots while it is awake, the amber dot only for the owner's move (a spending limit that
 * stopped it, a first conversation it waits for, a campaign that waits for the owner), a failure's red dot, a faint
 * ring while unconfirmed, two bars for a paused wake, a still mark when due. Its accessible name says the state and time.
 *
 * Pressing the chip opens the Shade and switches nothing: the host's state line (consciousness_dashboard.headlineView),
 * Start or Stop (the `/bg` command the Dashboard sends), the money of the last 24 hours, three own wakes and three wakes
 * in places in the Dashboard's phase words (one `GET /api/consciousness` per opening), `Dashboard ›` and, while
 * evolving, `Evolution ›`. The Shade is the room's work list material and gestures (room_work.js): it hangs under the
 * bar's capsules; an outside press, Escape or focus leaving it close it. The facts are the `/api/state` snapshot the
 * room already reads.
 */
import { headlineView } from './consciousness_dashboard.js';
import { normalizeConsciousness, readConsciousness } from './consciousness_api.js';
import { clockText, deadlineWords, splitWakes, wakePhaseText } from './consciousness_words.js';
import { tr } from './i18n.js';
import { controlIcon } from './page_icons.js';
import { createStatusMark, paintStatusMark } from './status_mark.js';
import { formatUsd2 } from './utils.js';
import { hangUnderCapsules } from './room_work.js';
import { motionMs, reducedMotion } from './motion.js';

// The owner's move (DESIGN "Status marks", amber): consciousness stopped by its spending limit or waiting for a first
// conversation, a campaign stopped by its limit or waiting for the owner's chat. A failure is an outcome's red dot,
// an unconfirmed outcome a ring, a paused wake two bars, a due wake the still mark.
const OWNER_MOVE = new Set(['allowance_exhausted', 'waiting_for_first_conversation']);
const CAMPAIGN_OWNER_MOVE = new Set(['waiting_for_first_conversation', 'waiting_for_owner_chat', 'budget_blocked', 'budget_stopped']);
const FAILED = new Set(['wake_failed', 'wake_rejected', 'stopped']);
const UNCONFIRMED = new Set(['wake_outcome_unknown', 'allowance_unknown']);
const NONE = Object.freeze({ shape: 'none', hue: 'ink' });
const WAKES_SHOWN = 3;
// A command waits this long for the runtime before the room's state clock is asked again (the Dashboard's beat).
const COMMAND_SETTLE_MS = 400;
let sequence = 0;
// The Shade folds back under the bar as it fades, like the room's (room_work.js).
const FOLDS = [{ opacity: 1, transform: 'none' }, { opacity: 0, transform: 'translateY(-6px)' }];

/** Whether a state answer carries consciousness facts at all (a failed read keeps the last chip). */
const carriesFacts = (data) => Boolean(data) && ('bg_consciousness_enabled' in data || 'bg_consciousness_state' in data);

/** The chip's view of one `/api/state` answer: its word, its one mark, and the words that say it all. */
export function chipView(data = {}) {
    const snapshot = normalizeConsciousness(data.bg_consciousness_state || {}, { enabled: data.bg_consciousness_enabled });
    const headline = headlineView(snapshot);
    const evolving = data.evolution_enabled === true;
    const off = snapshot.enabled === false;
    const unknown = snapshot.enabled === null;
    const status = snapshot.status;
    const owner = (!off && !unknown && OWNER_MOVE.has(status))
        || (evolving && CAMPAIGN_OWNER_MOVE.has(String(data.evolution_state?.status || '')));
    const word = evolving ? 'Evolving' : off ? 'Off' : headline.moving || unknown ? '' : deadlineWords(snapshot);
    const mark = owner ? { shape: 'dot', hue: 'owner' }
        : headline.moving ? { shape: 'moving', hue: 'ink' }
            : off ? NONE
                : FAILED.has(status) ? { shape: 'dot', hue: 'failed' }
                    : unknown || UNCONFIRMED.has(status) ? { shape: 'ring', hue: 'ink' }
                        : status === 'wake_paused' ? { shape: 'pause', hue: 'ink' }
                            : status === 'due' ? { shape: 'still', hue: 'ink' } : NONE;
    const said = [`Consciousness: ${headline.text}`,
        evolving ? 'evolution campaign running' : '', owner ? 'waits for you' : '']
        .filter(Boolean).join(' · ');
    return { word, mark, off, evolving, said, headline, snapshot };
}

/** The money line: «$1.50 of $20.00 in the last 24 hours»; a floor (≥) while some rows are unmetered; '' when unknown. */
export function allowanceText(snapshot) {
    const { spentUsd, dailyUsd } = snapshot;
    if (spentUsd === null || dailyUsd === null) return '';
    const floor = snapshot.unknownUnmetered > 0 ? '≥ ' : '';
    const degraded = snapshot.integrityDegraded ? ' (ledger integrity degraded)' : '';
    return `${floor}${formatUsd2(spentUsd)} of ${formatUsd2(dailyUsd)} in the last 24 hours${degraded}`;
}

/** The three last wakes, newest first: time, its one short line, and its phase in the Dashboard's words. */
export function wakeRows(wakes) {
    return (Array.isArray(wakes) ? wakes : []).slice(0, WAKES_SHOWN).map((wake) => ({
        time: clockText(wake.startedAt) || '—',
        text: wake.summary || (wake.cause === 'focus' ? wake.placeLabel || 'A place' : 'Own time'),
        outcome: wakePhaseText(wake),
    }));
}

/**
 * Mount the chip after `bar`'s title block and its Shade under `bar`. `ws` sends Start/Stop, `openDashboardTab` opens
 * the doors, `refreshState()` asks the room's state clock for a fresh answer after a command. `update(data)` takes
 * every `/api/state` answer the room reads.
 */
export function createConsciousnessStatus({ bar, ws = null, openDashboardTab = null, refreshState = () => {},
    read = readConsciousness } = {}) {
    const doc = bar?.ownerDocument;
    if (!doc) return null;
    const element = (tag, className = '', text = '') => {
        const node = doc.createElement(tag);
        if (className) node.className = className;
        if (text) node.textContent = text;
        return node;
    };
    const uid = `cx-${(sequence += 1)}`;

    // ---- the chip: a neutral capsule of the bar, like Panic
    const chip = element('button', 'chat-header-btn ui-chrome-control cx-chip');
    chip.type = 'button';
    chip.hidden = true;  // until the first answer that carries the facts
    chip.setAttribute('aria-haspopup', 'dialog');
    chip.setAttribute('aria-expanded', 'false');
    chip.setAttribute('aria-controls', `${uid}-shade`);
    const glyph = element('span', 'cx-chip-glyph');
    glyph.setAttribute('aria-hidden', 'true');
    glyph.innerHTML = controlIcon('brain', 18);
    const word = element('span', 'cx-chip-word');
    const mark = createStatusMark(doc, 'cx-chip-mark');
    chip.append(glyph, word, mark);
    const titleBlock = bar.querySelector('.app-page-title-block');
    if (titleBlock) titleBlock.after(chip);
    else bar.append(chip);

    // ---- the Shade: the room Shade's anchor and material under the bar
    const anchor = element('div', 'rw-shade-anchor cx-anchor');
    const shade = element('div', 'rw-surface rw-shade cx-shade');
    shade.id = `${uid}-shade`;
    shade.hidden = true;
    shade.tabIndex = -1;
    shade.setAttribute('role', 'dialog');
    shade.setAttribute('aria-label', 'Consciousness');
    const body = element('div', 'cx-body');
    const head = element('div', 'cx-head');
    const headWords = element('div', 'cx-head-words');
    const kicker = element('p', 'cx-kicker', 'Consciousness');
    const stateLine = element('p', 'cx-state');
    const money = element('p', 'cx-money');
    headWords.append(kicker, stateLine, money);
    const toggle = element('button', 'btn btn-default btn-sm cx-toggle');
    toggle.type = 'button';
    head.append(headWords, toggle);
    const coverageNote = element('p', 'cx-wakes-coverage');
    const groups = [
        ['own', tr('consciousness.own_wakes', 'Its own wakes'),
            tr('consciousness.no_own_wakes', 'No own wakes with a recorded model call in the last 48 h.')],
        ['places', tr('consciousness.place_wakes', 'Wakes in places'),
            tr('consciousness.no_place_wakes', 'No wakes in places in the last 48 h.')],
    ].map(([key, title, empty]) => {
        const heading = element('p', 'cx-wakes-title', title);
        const wakes = element('ol', 'cx-wakes');
        wakes.dataset.wakes = key;
        wakes.setAttribute('aria-label', title);
        wakes.setAttribute('data-i18n-authored', '');  // its own words, never the overlay's
        return { key, empty, heading, wakes, note: element('p', 'cx-wakes-note') };
    });
    const doors = element('div', 'cx-doors');
    const door = (label, tab) => {
        const button = element('button', 'cx-door ui-hit-area', label);
        button.type = 'button';
        button.dataset.tab = tab;
        return button;
    };
    const dashboardDoor = door('Dashboard ›', 'consciousness');
    const evolutionDoor = door('Evolution ›', 'evolution');
    doors.append(dashboardDoor, evolutionDoor);
    body.append(head, coverageNote, ...groups.flatMap((group) => [group.heading, group.wakes, group.note]), doors);
    shade.append(body);
    anchor.append(shade);
    bar.append(anchor);

    let view = null;
    let open = false;
    let destroyed = false;
    let readSeq = 0;
    // The command sent and when: its button waits for the first state answer read after it (a readback, never a guess).
    let pending = '';
    let pendingAt = 0;
    let folding = null;
    const connected = () => ws?.isConnected?.() !== false;

    function paintChip() {
        if (!view) return;
        chip.hidden = false;
        if (word.textContent !== view.word) word.textContent = view.word;
        word.hidden = !view.word;
        chip.dataset.off = view.off ? 'true' : 'false';
        chip.dataset.evolving = view.evolving ? 'true' : 'false';
        paintStatusMark(mark, view.mark);
        chip.setAttribute('aria-label', view.said);
        chip.title = view.said;
    }

    function paintShade() {
        if (!view) return;
        const { snapshot, headline } = view;
        stateLine.textContent = headline.text;
        const allowance = allowanceText(snapshot);
        money.textContent = allowance;
        money.hidden = !allowance;
        const on = snapshot.enabled === true;
        toggle.hidden = snapshot.enabled === null;
        toggle.textContent = on ? 'Stop' : 'Start';
        toggle.disabled = Boolean(pending) || !ws?.send || !connected();
        evolutionDoor.hidden = !view.evolving;
    }

    function paintWakes(list, coverage = '', failed = false) {
        coverageNote.textContent = coverage || (failed ? 'Recent wakes could not be read.'
            : !Array.isArray(list) ? 'Reading recent wakes…' : '');
        if (coverage) coverageNote.setAttribute('data-i18n-authored', '');  // the host's words; the client's are translated
        else coverageNote.removeAttribute('data-i18n-authored');
        coverageNote.hidden = !coverageNote.textContent;
        const split = splitWakes(list);
        for (const group of groups) {
            const rows = wakeRows(split[group.key]);
            group.wakes.replaceChildren(...rows.map((row) => {
                const item = element('li', 'cx-wake');
                const time = element('span', 'cx-wake-time', row.time);
                const text = element('span', 'cx-wake-text', row.text);
                const outcome = element('span', 'cx-wake-outcome', row.outcome);
                outcome.hidden = !row.outcome;
                item.append(time, text, outcome);
                return item;
            }));
            group.wakes.hidden = !rows.length;
            const note = Array.isArray(list) && !rows.length ? group.empty : '';
            group.note.textContent = note;
            group.note.hidden = !note;
        }
    }

    async function readWakes() {
        const seq = ++readSeq;
        paintWakes(null);
        let snapshot = null;
        try { snapshot = await read(); } catch { snapshot = null; }
        if (destroyed || seq !== readSeq || !open) return;
        if (!snapshot || !Array.isArray(snapshot.recentWakes)) paintWakes(null, snapshot?.coverage, true);
        else paintWakes(snapshot.recentWakes, snapshot.coverage);
    }

    function foldAway() {
        const ms = motionMs('--motion-fade');
        if (!ms || reducedMotion() || typeof shade.animate !== 'function') { shade.hidden = true; return; }
        shade.inert = true;
        const fold = shade.animate(FOLDS, { duration: ms, easing: 'ease-in', fill: 'forwards' });
        folding = fold;
        fold.finished.then(() => { if (folding === fold && !open) shade.hidden = true; }, () => {}).finally(() => {
            if (folding !== fold) return;
            folding = null;
            shade.inert = false;
            fold.cancel();
        });
    }

    function openShade(viaKeyboard = false) {
        if (open || destroyed || !view) return;
        open = true;
        if (folding) { const fold = folding; folding = null; shade.inert = false; fold.cancel(); }
        paintShade();
        hangUnderCapsules(bar, anchor);
        shade.hidden = false;
        chip.setAttribute('aria-expanded', 'true');
        chip.dataset.open = '1';
        bar.setAttribute('data-cx-shade', '');  // Main's history note waits behind it (room_work.css)
        void readWakes();
        // A pointer open lands on the Shade itself; a keyboard open enters its first control.
        const first = viaKeyboard ? [toggle, dashboardDoor].find((node) => !node.hidden && !node.disabled) : null;
        (first || shade).focus({ preventScroll: true });
    }

    function close({ restoreFocus = true } = {}) {
        if (!open) return;
        open = false;
        readSeq += 1;
        chip.setAttribute('aria-expanded', 'false');
        delete chip.dataset.open;
        bar.removeAttribute('data-cx-shade');
        foldAway();
        if (restoreFocus && chip.isConnected && !chip.hidden) chip.focus({ preventScroll: true });
    }

    const disposers = [];
    const listen = (target, type, fn, options) => {
        target.addEventListener(type, fn, options);
        disposers.push(() => target.removeEventListener(type, fn, options));
    };
    // A click with detail 0 is Enter or Space on the focused chip: a keyboard open lands in the Shade.
    listen(chip, 'click', (event) => (open ? close() : openShade(event.detail === 0)));
    listen(chip, 'keydown', (event) => {
        if ((event.key === 'ArrowDown' || event.key === 'ArrowUp') && !open) { event.preventDefault(); openShade(true); }
    });
    listen(toggle, 'click', () => {
        if (toggle.disabled || !view) return;
        pending = view.snapshot.enabled ? 'stop' : 'start';
        pendingAt = Date.now();
        ws.send({ type: 'command', cmd: `/bg ${pending}` });
        paintShade();
        // The runtime applies the command on its own clock: the room's next state answer shows it.
        setTimeout(() => { if (!destroyed) refreshState(true); }, COMMAND_SETTLE_MS);
    });
    listen(doors, 'click', (event) => {
        const target = event.target.closest?.('.cx-door');
        if (!target) return;
        close({ restoreFocus: false });
        openDashboardTab?.(target.dataset.tab);
    });
    // The room Shade's dismissal: an outside press, Escape, focus leaving the chip and the Shade.
    listen(doc, 'pointerdown', (event) => {
        if (open && !shade.contains(event.target) && !chip.contains(event.target)) close({ restoreFocus: false });
    }, true);
    listen(doc, 'keydown', (event) => {
        if (event.key !== 'Escape' || !open || event.defaultPrevented) return;
        const active = doc.activeElement;
        if (active && active !== doc.body && !shade.contains(active) && active !== chip) return;
        event.preventDefault();
        close();
    });
    // The open list follows the bar's capsules when the window changes the bar.
    if (doc.defaultView) listen(doc.defaultView, 'resize', () => { if (open) hangUnderCapsules(bar, anchor); });
    listen(shade, 'focusout', (event) => {
        const next = event.relatedTarget;
        if (open && next && !shade.contains(next) && next !== chip) close({ restoreFocus: false });
    });

    return {
        chip,
        shade,
        /** One `/api/state` answer; a failed read (no consciousness facts) keeps the last chip. */
        update(data) {
            if (destroyed || !carriesFacts(data)) return;
            view = chipView(data);
            if (pending && Date.now() - pendingAt >= COMMAND_SETTLE_MS) pending = '';
            paintChip();
            if (open) paintShade();
        },
        isOpen: () => open,
        close: () => close({ restoreFocus: false }),
        destroy() {
            if (destroyed) return;
            destroyed = true;
            readSeq += 1;
            for (const dispose of disposers.splice(0)) dispose();
            folding?.cancel();
            bar.removeAttribute('data-cx-shade');
            chip.remove();
            anchor.remove();
        },
    };
}
