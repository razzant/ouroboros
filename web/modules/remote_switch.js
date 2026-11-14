/**
 * Remote in a room's bar (docs/DESIGN.md §10 "Page header"): one capsule after the consciousness chip, the computer
 * glyph and its word, and the owner's one switch of where Main and new agents run (ouroboros/remote_placement.py).
 *
 * The capsule is the saved preference: `Off`, or the target engine's name in the switched-on tone, `· agents` beside
 * it while On does not move Main (an API key, the local model, a Main naming an engine of its own); the amber dot while
 * the target takes no new work (removed or switched off in Settings), the owner's move. A press acts at once: On turns
 * Off, whatever became of its target (the fastest way back to this computer is never behind a setup); Off with no
 * remote engine saved opens Settings › Accounts and keys › Remote engines; Off turns On for the one engine there is, or
 * the remembered one, and asks which engine when that choice is not made (several engines and none remembered, or the
 * remembered one is gone or switched off): never the first by default. The chevron inside the same capsule opens the
 * chooser: where Main and new agents run now (the server's projection, `/api/state` `remote_execution`, read by the same
 * resolver dispatch uses), the engines, what a press moves (Main at its next call, new agents; agents already running
 * keep their engine), what Emergency Stop does there, what a remote engine refuses, and the Remote engines door. A write
 * is the ordinary settings save of OUROBOROS_REMOTE_EXECUTION (settings_controls.commitSetting, its one queue and
 * outcome); the capsule says what the server confirms and never shows a change it did not, and a press stops,
 * disconnects or rewrites nothing.
 */
import { fmt, tr } from './i18n.js';
import { controlIcon } from './page_icons.js';
import { commitSetting } from './settings_controls.js';
import { createStatusMark, paintStatusMark } from './status_mark.js';
import { showToast } from './toast.js';
import { bindMenu } from './ui_interactions.js';

export const REMOTE_EXECUTION_KEY = 'OUROBOROS_REMOTE_EXECUTION';
export const ENGINES_ROW = 'OUROBOROS_ENGINE_CONNECTIONS';
// The settings row the switch writes through (settings_rows.js HAND_WRITTEN): one key, the generic save.
export const REMOTE_ROW = Object.freeze({ id: REMOTE_EXECUTION_KEY, keys: [REMOTE_EXECUTION_KEY], name: 'Remote',
    page: 'accounts', writer: 'generic', codec: '', confirm: '', control: { kind: 'custom' } });

/** The stored spelling: the server's canonical JSON (sorted keys, no spaces), so a read shows what a write sent. */
export const remoteValue = (on, engine) => JSON.stringify({ engine: String(engine || ''), on: Boolean(on) });

const SCOPE_MAIN = {
    api: 'Main uses an API key and stays as it is',
    local: 'Main uses the local model and stays on this computer',
    unreadable: 'Main stays as set in Models',
};

/**
 * The capsule's view of one `remote_execution` projection (null when the state carries none: an older server or an
 * unread projection, which hides the capsule rather than guessing). `press` is what a press of the capsule does:
 * `setup`, `off`, `on` (with `target`) or `choose`.
 */
export function remoteView(state) {
    if (!state || typeof state !== 'object' || !Array.isArray(state.engines)) return null;
    const engines = state.engines.filter((row) => row && row.name);
    const named = (name) => engines.find((row) => row.name === name) || null;
    const labelOf = (name) => named(name)?.label || name;
    const remembered = named(state.engine);
    const on = state.on === true && Boolean(state.engine);
    const usable = engines.filter((row) => row.enabled !== false);
    const main = state.main || {};
    const scope = String(main.scope || '');
    const mainEngine = scope === 'engine' ? String(main.model || '').match(/^claudexor@([^:]+)::/)?.[1] || '' : '';
    // On names its target even after it is removed or switched off: the work it places there is refused, never moved.
    const targetProblem = !on ? '' : !remembered ? 'removed' : remembered.enabled === false ? 'off' : '';
    const label = on ? labelOf(state.engine) : '';
    const mainMoves = on && scope === 'follows';
    const moved = scope === 'follows' ? tr('remote.main_and_agents', 'Main and new agents') : tr('remote.new_agents', 'new agents');
    const lines = [];
    if (on) {
        lines.push(mainMoves ? fmt('Main and new agents run on {engine}.', { engine: label })
            : fmt('New agents run on {engine}; {main}.', { engine: label, main: scope === 'engine'
                ? fmt('Main runs on {engine}, as set in Models', { engine: labelOf(mainEngine) })
                : SCOPE_MAIN[scope] || SCOPE_MAIN.unreadable }));
        lines.push(tr('remote.agents_kept', 'Agents set to their own engine in Settings keep it. There, Main and agents use that engine\'s own accounts (Auto).'));
        if (targetProblem === 'removed') lines.push(fmt('{engine} is no longer saved: new work for it is refused until you choose again.', { engine: state.engine }));
        if (targetProblem === 'off') lines.push(fmt('{engine} is switched off in Settings: new work for it is refused until you turn it on there or turn Remote off.', { engine: label }));
        if (!engines.length) lines.push(tr('remote.none_saved', 'No remote engine is saved now.'));
    } else {
        lines.push(scope === 'engine' ? fmt('New agents run on this computer; Main runs on {engine}, as set in Models.', { engine: labelOf(mainEngine) })
            : scope === 'follows' ? tr('remote.off_here', 'Main and new agents run on this computer.')
                : fmt('New agents run on this computer; {main}.', { main: SCOPE_MAIN[scope] || SCOPE_MAIN.unreadable }));
    }
    let press, target = null;
    if (on) press = 'off';  // the way back to this computer, whatever became of the target
    else if (!engines.length) press = 'setup';
    else if (remembered && remembered.enabled !== false) { press = 'on'; target = remembered; }
    else if (!state.engine && usable.length === 1) { press = 'on'; target = usable[0]; }
    else press = 'choose';
    const state_ = on ? fmt('Remote: on, {engine}', { engine: label })
        : !engines.length ? tr('remote.unset', 'Remote: no remote engine is set up') : tr('remote.off', 'Remote: off');
    const action = press === 'setup' ? tr('remote.press_setup', 'Press to set one up in Settings')
        : press === 'off' ? fmt('Press to run {what} on this computer', { what: moved })
            : press === 'on' ? fmt('Press to run {what} on {engine}', { what: moved, engine: target.label || target.name })
                : tr('remote.press_choose', 'Press to choose a remote engine');
    return {
        // The chooser stays reachable while On, even with every engine since removed: its first row is this computer.
        configured: engines.length > 0, chooser: engines.length > 0 || on, on, engine: String(state.engine || ''), label,
        engines, remembered, usable,
        mainMoves, scope, targetProblem, press, target, lines,
        word: on ? label : tr('remote.word_off', 'Off'),
        mixed: on && !mainMoves,  // On moves new agents only: the capsule says so beside the name
        mark: targetProblem ? { shape: 'dot', hue: 'owner' } : { shape: 'none' },
        said: [state_, ...lines, action].join('. ').replace(/\.\./g, '.'),
    };
}

/** The chooser's notes for a view: what a press moves and what it leaves, what Emergency Stop does to remote work,
 *  what an engine refuses. */
export function chooserNotes(view) {
    const notes = [...view.lines];
    if (view.on) {
        notes.push(fmt('Emergency Stop closes the tunnel at once; work already on {engine} may go on there until this computer reconnects.', { engine: view.label }));
        notes.push(tr('remote.limits', 'There, plan turns, in-place edits and folders that are not Git projects are refused.'));
    }
    notes.push(tr('remote.press_note', 'A press moves Main at its next call and the agents started after it; agents already running keep their engine, and nothing is stopped or disconnected.'));
    return notes;
}

let serial = 0;

/**
 * Mount the capsule after `bar`'s consciousness chip (or its title block). `update(data)` takes every `/api/state`
 * answer; `refreshState(true)` asks for a fresh one after a write, and `retireState()` first retires the page's reads
 * begun before a write that landed (state_snapshots.js `retire`), so none of them answers for it; `openSettings(path,
 * options)` is the shell's opener. `commit` stands in for the settings writer in tests. Returns `{ el, update, destroy }`.
 */
export function createRemoteSwitch({ bar, openSettings = null, refreshState = () => {}, retireState = () => {},
    commit = commitSetting, notify = showToast } = {}) {
    const doc = bar?.ownerDocument;
    if (!doc) return null;
    const uid = `rx-${(serial += 1)}`;
    const el = doc.createElement('span');
    el.className = 'rx-capsule';
    el.hidden = true;  // until the first answer that carries the projection
    const toggle = doc.createElement('button');
    toggle.type = 'button';
    toggle.className = 'rx-toggle';
    const glyph = doc.createElement('span');
    glyph.className = 'rx-glyph';
    glyph.setAttribute('aria-hidden', 'true');
    glyph.innerHTML = controlIcon('monitor', 18);
    const word = doc.createElement('span');
    word.className = 'rx-word';
    const scopeWord = doc.createElement('span');
    scopeWord.className = 'rx-scope';
    scopeWord.textContent = tr('remote.agents_only', '· agents');
    const mark = createStatusMark(doc, 'rx-mark');
    toggle.append(glyph, word, scopeWord, mark);
    const more = doc.createElement('button');
    more.type = 'button';
    more.className = 'rx-more';
    more.setAttribute('aria-haspopup', 'menu');
    more.setAttribute('aria-expanded', 'false');
    more.setAttribute('aria-label', tr('remote.choices', 'Remote: where Main and new agents run'));
    more.innerHTML = controlIcon('chevron', 14);
    more.firstElementChild?.setAttribute('aria-hidden', 'true');
    el.append(toggle, more);
    const chip = bar.querySelector('.cx-chip');
    if (chip) chip.after(el);
    else (bar.querySelector('.app-page-title-block') || bar.lastElementChild)?.after(el);

    let view = null;
    let pending = '';   // the value a write sent, until the server's answer and the next state read after it
    let confirming = false;  // the write landed: the next state answer, a read begun after it, is what the capsule says
    let menu = null;
    let destroyed = false;

    function paint() {
        if (!view) return;
        el.hidden = false;
        el.dataset.on = view.on ? 'true' : 'false';
        el.dataset.mixed = view.mixed ? 'true' : 'false';
        const shown = pending ? tr('remote.saving', 'Saving…') : view.word;
        if (word.textContent !== shown) word.textContent = shown;
        scopeWord.hidden = Boolean(pending) || !view.mixed;
        paintStatusMark(mark, view.mark);
        toggle.setAttribute('aria-pressed', view.on ? 'true' : 'false');
        toggle.setAttribute('aria-busy', pending ? 'true' : 'false');
        toggle.setAttribute('aria-label', pending ? tr('remote.saving_said', 'Remote: saving your change') : view.said);
        toggle.title = view.said;
        more.hidden = !view.chooser;
        el.dataset.chooser = view.chooser ? 'true' : 'false';
    }

    async function write(value) {
        if (pending || destroyed) return;
        pending = value;
        paint();
        let outcome;
        try { outcome = await commit(REMOTE_ROW, value, { confirmed: true }); } catch (error) {
            outcome = { state: 'unknown', message: String(error?.message || error) };
        }
        if (destroyed) return;
        if (outcome?.state !== 'saved' && outcome?.state !== 'saved_then_failed' && outcome?.state !== 'unchanged') {
            pending = '';
            paint();
            notify(outcome?.state === 'unknown'
                ? fmt('Remote change not confirmed: {reason}', { reason: outcome?.message || tr('remote.no_answer', 'no answer') })
                : (outcome?.message || tr('remote.refused', 'Remote was not changed.')), outcome?.state === 'unknown' ? 'warn' : 'danger');
            return;
        }
        confirming = true;
        retireState();  // a read begun before the save may answer after it with what was saved before
        refreshState(true);  // the capsule says what the next state answer shows (`pending` until then)
    }

    function openSetup() {
        closeMenu();
        void openSettings?.('accounts', { row: ENGINES_ROW });
    }

    function closeMenu({ restoreFocus = false } = {}) {
        if (!menu) return;
        const binding = menu;
        menu = null;
        binding.close({ restoreFocus, reason: 'action' });
    }

    function openMenu() {
        if (!view || destroyed) return;
        if (menu) { closeMenu({ restoreFocus: true }); return; }
        const list = doc.createElement('div');
        list.className = 'ui-popup rx-menu';
        list.id = `${uid}-menu`;
        list.setAttribute('role', 'menu');
        list.setAttribute('aria-label', tr('remote.menu', 'Run Main and new agents on'));
        const head = doc.createElement('div');
        head.className = 'rx-menu-head';
        head.textContent = tr('remote.menu', 'Run Main and new agents on');
        list.append(head);
        const item = (label, line, { checked = false, disabled = false, act = null, radio = true } = {}) => {
            const button = doc.createElement('button');
            button.type = 'button';
            button.className = 'rx-menu-item';
            button.setAttribute('role', radio ? 'menuitemradio' : 'menuitem');
            if (radio) button.setAttribute('aria-checked', checked ? 'true' : 'false');
            if (disabled) button.setAttribute('aria-disabled', 'true');
            const name = doc.createElement('span');
            name.className = 'rx-menu-name';
            name.textContent = label;
            button.append(name);
            if (line) {
                const sub = doc.createElement('span');
                sub.className = 'rx-menu-line';
                sub.textContent = line;
                button.append(sub);
            }
            button.addEventListener('click', () => {
                if (disabled) return;
                closeMenu({ restoreFocus: true });
                act?.();
            });
            list.append(button);
            return button;
        };
        item(tr('remote.here', 'This computer'), '', { checked: !view.on, act: () => { if (view.on) void write(remoteValue(false, view.engine)); } });
        for (const engine of view.engines) {
            const off = engine.enabled === false;
            const remembered = !view.on && engine.name === view.engine;
            item(engine.label || engine.name, off ? tr('remote.switched_off', 'Switched off in Settings')
                : remembered ? tr('remote.last_used', 'Last used') : '', {
                checked: view.on && engine.name === view.engine, disabled: off,
                act: () => { if (!(view.on && engine.name === view.engine)) void write(remoteValue(true, engine.name)); },
            });
        }
        if (view.on && view.targetProblem === 'removed') {
            item(view.engine, tr('remote.removed', 'No longer saved'), { checked: true, disabled: true });
        }
        for (const text of chooserNotes(view)) {
            const note = doc.createElement('p');
            note.className = 'rx-menu-note';
            note.textContent = text;
            list.append(note);
        }
        item(tr('remote.settings', 'Remote engines settings ›'), '', { radio: false, act: openSetup });
        doc.body.append(list);
        more.setAttribute('aria-expanded', 'true');
        el.dataset.open = '1';
        const binding = bindMenu(list, { anchor: more, align: 'end', onClose: () => {
            list.remove();
            more.setAttribute('aria-expanded', 'false');
            delete el.dataset.open;
            if (menu === binding) menu = null;
        } });
        menu = binding;
    }

    const onToggle = () => {
        if (!view || pending) return;
        if (view.press === 'setup') openSetup();
        else if (view.press === 'off') void write(remoteValue(false, view.engine));
        else if (view.press === 'on') void write(remoteValue(true, view.target.name));
        else openMenu();
    };
    const onMore = () => openMenu();
    // A settings save elsewhere on this page (an engine added, switched off or removed; Main changed) is read back at once.
    const onSettings = (event) => {
        const keys = event?.detail?.keys || [];
        if (keys.some((key) => [ENGINES_ROW, REMOTE_EXECUTION_KEY, 'OUROBOROS_MODEL', 'USE_LOCAL_MAIN'].includes(key))) refreshState(true);
    };
    toggle.addEventListener('click', onToggle);
    more.addEventListener('click', onMore);
    const win = doc.defaultView;
    win?.addEventListener?.('ouro:settings-updated', onSettings);

    return {
        el,
        /** One `/api/state` answer; one without the projection keeps the last capsule. */
        update(data) {
            if (destroyed || !data || !Object.hasOwn(data, 'remote_execution')) return;
            const next = remoteView(data.remote_execution);
            if (!next) return;
            view = next;
            if (confirming) { pending = ''; confirming = false; }  // read after the write landed: what the server shows
            paint();
        },
        view: () => view,
        isOpen: () => Boolean(menu),
        destroy() {
            if (destroyed) return;
            destroyed = true;
            closeMenu();
            toggle.removeEventListener('click', onToggle);
            more.removeEventListener('click', onMore);
            win?.removeEventListener?.('ouro:settings-updated', onSettings);
            el.remove();
        },
    };
}
