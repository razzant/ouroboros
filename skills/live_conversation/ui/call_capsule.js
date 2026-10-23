/** The composer face of a call. With nothing to send, the Send capsule reads Talk; a press (never
 * Enter) starts the call. During a call the text row keeps its footprint and becomes the live
 * capsule (the chip): Expand, the visual behind the state and the call's time, Mute, Show (a screen
 * or the camera, from a small menu), Keyboard, End. Expand grows the same capsule into the big view
 * (`call_stage.js`): the state and the time in a corner, Minimise, and the same controls at the
 * bottom. Keyboard brings the field back beside the call's controls (from the big view, in the chip);
 * End and the way back to voice never leave, and what is typed goes as an ordinary turn. When Talk
 * cannot work, or a call ends with something to say, a sheet takes the capsule's place with the
 * words and one next step.
 * Every word the owner reads is chosen here from the session's facts (`callPresentation`). Calls
 * use the explicitly selected engine.
 */
import { fmt, tx } from '/static/modules/i18n.js';
import { bindMenu } from '/static/modules/ui_interactions.js';
import { createCallStage } from './call_stage.js';

const KEYBOARD = '<svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" aria-hidden="true"><rect x="2.5" y="6" width="19" height="12" rx="2.5"/><path d="M6.5 10h.01M10 10h.01M14 10h.01M17.5 10h.01M7.5 14h9"/></svg>';
// The same button in the keyboard view: back to the call's voice.
const VOICE = '<svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" aria-hidden="true"><path d="M4 10v4M8 7v10M12 4v16M16 7v10M20 10v4"/></svg>';
// Show: what Ouroboros may see. The menu offers each source this surface can capture; while one is
// shown the button wears that source's icon and stops it.
const SHOW = '<svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M2.5 12S6 5.5 12 5.5 21.5 12 21.5 12 18 18.5 12 18.5 2.5 12 2.5 12z"/><path d="M12 9.25a2.75 2.75 0 1 0 0 5.5 2.75 2.75 0 0 0 0-5.5z"/></svg>';
const SCREEN = '<svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><rect x="2.5" y="4" width="19" height="13" rx="2.5"/><path d="M8.5 20.5h7M12 17v3.5"/></svg>';
const CAMERA = '<svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><rect x="2.5" y="6.5" width="13.5" height="11" rx="2.5"/><path d="M16 10.5l5.5-3v9l-5.5-3"/></svg>';
const SOURCES = {
    screen: { icon: SCREEN, label: 'Screen', hint: 'A window or the whole screen', stop: 'Stop showing your screen', on: 'Screen on' },
    camera: { icon: CAMERA, label: 'Camera', hint: '', stop: 'Stop the camera', on: 'Camera on' },
};
const TOGGLE = {
    keyboard: { icon: KEYBOARD, label: 'Type while talking' },
    voice: { icon: VOICE, label: 'Back to voice' },
};
// The field's placeholder while the call's controls share its row: one line at a phone's width, so
// opening the keyboard view never grows the composer.
const KEYBOARD_PLACEHOLDER = 'Message…';
// The way between the chip and the big view: one capsule, its icon and word follow the view.
const GROW = '<svg class="call-size-grow" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M14 4h6v6M10 20H4v-6M20 4l-7 7M4 20l7-7"/></svg>';
const SHRINK = '<svg class="call-size-shrink" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M20 10h-6V4M4 14h6v6M14 10l7-7M3 21l7-7"/></svg>';
const SIZE = { chip: 'Expand the call', big: 'Minimise the call' };
const CALL_STATE = ['call', 'callView', 'callLight', 'callVoice', 'callTone', 'callSharing', 'callWork', 'callLine'];

// Why a call reconnects, beside "Reconnecting…".
const RECONNECT_WORDS = {
    session_cap: 'Starting a fresh voice session',
    go_away: 'The voice service is moving the session',
    transport_lost: 'The voice connection dropped',
};
// Feature configuration lives in the ordinary extension settings.
// Engine keys are the secrets the call skills request (Settings → Secrets), not provider accounts.
const KEY = { tab: 'secrets', focus: 'skill-requested-secrets' };
const MODEL = { tab: 'advanced', focus: 'extension-settings-sections' };
const BUDGET = { tab: 'behavior', focus: '' };
const SWITCH = MODEL;
const SPENDING_LIMIT = { words: 'The spending limit is reached', action: 'settings', settings: BUDGET };
const REFUSED = { words: 'The voice service refused the call: “{detail}”', bare: 'The voice service refused the call', action: 'retry' };
const UNREACHABLE = { words: 'Can’t reach the selected voice service', action: 'retry' };
const NOT_STARTED = { words: 'The call could not start; the room has the details', action: 'retry' };
const NO_KEY = { words: 'Calls need the selected engine’s API key — add it in Settings → Secrets', action: 'settings', settings: KEY };
// How a call ends, in the owner's words. For Google's own closes and errors the server's `detail`
// is Google's sentence, quoted as Google's; for the host's own refusals it is the host's sentence,
// never framed as Google's. `bare` is the sentence when there is nothing to say, `action` the one
// next step; `calm` endings are facts, not problems. Codes this page meets itself, and surfaces
// that cannot hold a call (`sheet`), are words too. The server's own `action` wins over this table.
const ENDINGS = {
    owner_bye: { words: 'Call ended', calm: true },
    ouroboros_bye: { words: 'Ouroboros ended the call', calm: true },
    owner_stop: { words: 'Call ended', calm: true },
    peers_left: { words: 'Everyone left the call', calm: true },
    budget_exceeded: SPENDING_LIMIT,
    budget_exhausted: SPENDING_LIMIT,
    missing_credential: NO_KEY,
    credential: NO_KEY,
    missing_credential_google: NO_KEY,
    calls_disabled: { words: 'Calls are off — turn on Calls (beta) in Settings', action: 'settings', settings: SWITCH },
    authentication: { words: 'The voice service refused the key', action: 'settings', settings: KEY },
    rate_limit: { words: 'The voice service says: “{detail}”', bare: 'The voice service asked to slow down', action: 'retry' },
    provider_closed: { words: 'The voice service closed the call: “{detail}”', bare: 'The voice service closed the call', action: 'retry' },
    engine_unavailable: UNREACHABLE,
    reconnect_failed: UNREACHABLE,
    preparation_failed: NOT_STARTED,
    call_not_started: NOT_STARTED,
    invalid_request: { words: REFUSED.words, bare: REFUSED.bare },  // a refused setup: the same call would repeat it
    protocol: REFUSED,
    provider_error: REFUSED,
    backpressure: { words: 'The call fell behind the voice stream', action: 'retry' },
    engine_refused: { words: 'The call could not start: {detail}', bare: 'The call could not start',
        action: 'settings', settings: MODEL },
    // Admission refusals the server words itself; the table adds only the next step.
    call_model_missing: { action: 'settings', settings: MODEL },
    permission_denied: { sheet: true, words: 'Allow the microphone for Ouroboros in System Settings → Privacy & Security → '
        + 'Microphone (or in the browser’s site settings), then press Talk again.' },
    no_microphone: { sheet: true, words: 'No microphone was found. Connect one, then press Talk again.' },
    microphone_unreadable: { sheet: true, words: 'The microphone could not be opened. Close other apps that use it, then press Talk again.' },
    microphone_busy: { sheet: true, words: 'Another room is in a call. End that call first.', action: 'reveal' },
    // Something other than a call holds the microphone (dictation): the lease names it.
    microphone_in_use: { sheet: true, words: 'The microphone is in use by {detail}. Stop it, then press Talk again.',
        bare: 'The microphone is in use. Stop what uses it, then press Talk again.' },
    microphone_lost: { words: 'The microphone was disconnected', action: 'retry' },
    audio_failed: { words: 'Live audio stopped working', action: 'retry' },
    slow_uplink: { words: 'The connection is too slow to send live audio', action: 'retry' },
    socket_silent: { words: 'The call service did not answer', action: 'retry' },
    socket_closed: { words: 'The call connection closed: “{detail}”', bare: 'The call ended unexpectedly', action: 'retry' },
    protocol_mismatch: { words: 'This page could not read the call; reload Ouroboros and try again' },
    call_failed: { words: 'The call could not start', action: 'retry' },
    old_app: { sheet: true, actions: ['open_browser', 'get_app'], words: 'This version of the Ouroboros app can’t use the '
        + 'microphone. Talk works in your browser, or after you install the new app.' },
    insecure: { sheet: true, words: 'Talk needs a secure page (HTTPS) or this computer’s own address to use the microphone.' },
    no_microphone_api: { sheet: true, words: 'This browser can’t use a microphone here.' },
    no_live_audio: { sheet: true, words: 'This browser can’t play live audio. Open Ouroboros in a current browser.' },
    no_call_socket: { sheet: true, words: 'This browser can’t open call connections.' },
};
const ACTION_LABELS = { retry: 'Try again', settings: 'Open Settings', reveal: 'Go to the call',
    open_browser: 'Open in browser', get_app: 'Get the new app' };

/** A reason code as `{ kind, base, detail }`: `kind` and `base` are the code before any `:detail`. */
function reason(code) {
    const raw = String(code || '');
    const colon = raw.indexOf(':');
    const base = colon < 0 ? raw : raw.slice(0, colon);
    return { kind: base, base, detail: colon < 0 ? '' : raw.slice(colon + 1) };
}

function actionFor(kind, entry) {
    if (!ACTION_LABELS[kind]) return null;
    const action = { kind, label: ACTION_LABELS[kind] };
    if (kind === 'settings') Object.assign(action, entry?.settings || MODEL);
    return action;
}

/** What the capsule shows for the session's facts and what the page heard (`live`). */
export function callPresentation(view, live = {}) {
    const state = view?.state || 'idle';
    if (state === 'idle' || state === 'ended') return { mode: 'idle' };
    if (state === 'closed' || state === 'failed') return ending(view);
    const phase = { mode: 'live', light: 'trace', voice: 'owner', title: '', sub: '' };
    if (state === 'permission') return { ...phase, title: 'Waiting for the microphone', sub: 'Allow it if your browser asks' };
    if (state === 'admission') return { ...phase, title: 'Starting the call…' };
    if (state === 'preparing') return view.phase === 'memory'
        ? { ...phase, title: 'Preparing memory…', sub: 'Reading this room and my notes' }
        : { ...phase, title: 'Preparing the call…' };
    if (state === 'connecting') return { ...phase, title: 'Connecting…' };
    if (state === 'reconnecting') return { ...phase, title: 'Reconnecting…', sub: RECONNECT_WORDS[reason(view.code).base] || '' };
    if (state === 'closing') return { ...phase, voice: '', title: 'Ending the call…' };
    const heard = Boolean(live.heard);
    const base = { mode: 'live', light: 'listen', voice: heard ? 'ouroboros' : 'owner', sub: '' };
    if (view.muted) return { ...base, light: 'muted', title: 'Muted', sub: 'Ouroboros can’t hear you' };
    if (heard) return { ...base, title: 'Ouroboros', sub: 'Speaking', voiced: true };
    // The owner's voice, from the microphone gate at once (a caption arrives later).
    if (live.owner || live.ownerSpeaking) return { ...base, title: 'Listening', sub: 'You’re speaking', voiced: true };
    // One answer is one Speaking, its pauses included: the engine says when it ends (`turn`).
    if (view.turn === 'answering' && live.answered) return { ...base, voice: 'ouroboros', title: 'Ouroboros', sub: 'Speaking',
        voiced: true };
    // The tool's own label is the host's sentence, shown as it is (or as the install translated it).
    if (view.activity?.working) return { ...base, light: 'think', work: true, title: 'Working',
        sub: view.activity.label ? tx(view.activity.label) : 'Using a tool', subFormatted: Boolean(view.activity.label) };
    // An answer is awaited (the owner finished, a result or an addressed fact went in) or begun but not yet heard.
    if (live.thinking || view.turn === 'awaiting' || view.turn === 'answering') return { ...base, light: 'think', title: 'Thinking' };
    return { ...base, title: 'Listening', sub: 'Speak whenever you’re ready' };
}

function ending(view) {
    const { kind, detail: codeDetail } = reason(view.code);
    const entry = ENDINGS[kind];
    const detail = String(view.detail || codeDetail || '').trim();
    const sheet = { mode: 'sheet', light: 'still', voice: '', tone: '', sub: '', small: '', formatted: false, actions: [] };
    const steps = names => names.map(name => name && actionFor(name, entry)).filter(Boolean);
    // An admission refusal is already in the owner's words; this page adds the one next step.
    if (view.words) return { ...sheet, tone: 'error', light: 'error', title: tx(view.words), formatted: true,
        actions: steps([view.action || entry?.action || 'retry']) };
    if (!detail && (!view.code || kind === 'finished')) return { ...sheet, title: 'The call ended', sub: 'The room has its transcript' };
    if (!entry?.words) return { ...sheet, title: detail || 'The call ended', formatted: Boolean(detail),
        tone: detail ? 'error' : '', light: detail ? 'error' : 'still', actions: steps([view.action || entry?.action]) };
    const template = detail && entry.words.includes('{detail}') ? entry.words : entry.bare || entry.words;
    const formatted = template.includes('{detail}');
    return { ...sheet, title: formatted ? fmt(template, { detail }) : template, formatted,
        actions: steps(view.action ? [view.action] : entry.actions || [entry.action]),
        // A call that failed is amber; a fact, or why this surface cannot call, is plain words.
        tone: entry.calm ? '' : entry.sheet ? 'notice' : 'error', light: entry.calm || entry.sheet ? 'still' : 'error' };
}

/** Who is talking, from the captions and the light. Thinking runs from the owner's final words to
 * the first heard answer; an answer whose item is already named (Gemini can finish the owner's
 * caption after its answer began) leaves no thinking behind it. */
export function createTalkFacts() {
    const facts = { ownerSpeaking: false, thinking: false, answered: false };
    return {
        facts,
        caption(msg, heardNow = false) {
            if (msg.speaker === 'owner') {
                if (!msg.final && msg.text) Object.assign(facts, { ownerSpeaking: true, thinking: false, answered: false });
                if (msg.final) Object.assign(facts, { ownerSpeaking: false,
                    thinking: Boolean(String(msg.text || '').trim()) && !facts.answered && !heardNow });
            } else {
                facts.answered = true;
                if (msg.final) facts.thinking = false;  // an answer that was never heard still ends thinking
            }
        },
        /** The owner's voice began (the light's microphone onset): the next answer is a new one. */
        voice() { facts.answered = false; },
        /** The engine's turn: an awaited answer or an ended one starts the next answer unheard. */
        turn(next) { if (next !== 'answering') facts.answered = false; },
        heard(now) { if (now) Object.assign(facts, { thinking: false, answered: true }); },
        reset() { Object.assign(facts, { ownerSpeaking: false, thinking: false, answered: false }); },
    };
}

export function createCallCapsule({ page, env = globalThis, visual, light, styled = Promise.resolve(), view, talk, mute,
    show = () => {}, vision = {}, end, dismiss, act, offered = () => true }) {
    const doc = page.ownerDocument;
    const wrap = page.querySelector('.chat-input-wrap');
    const row = wrap.querySelector('.chat-text-row');
    const field = row.querySelector('textarea');
    const send = row.querySelector('.chat-send-inline');
    const group = row.querySelector('.chat-send-group');
    const preview = wrap.querySelector('.chat-attachment-preview');
    const capsule = doc.createElement('div');
    capsule.className = 'call-capsule';
    capsule.hidden = true;
    capsule.innerHTML = `<button type="button" class="call-btn call-size" aria-expanded="false">${GROW}${SHRINK}<span class="call-size-word">Minimise</span></button>
        <span class="call-visual"></span>
        <span class="call-words" role="status" aria-live="polite"><span class="call-state"></span><span class="call-time" aria-hidden="true" hidden></span><span class="call-sub"></span><span class="call-small"></span></span>
        <span class="call-buttons">
            <button type="button" class="call-btn call-action"></button>
            <button type="button" class="call-btn call-action"></button>
            <button type="button" class="call-btn call-mute" aria-pressed="false">Mute</button>
            <button type="button" class="call-btn call-show"></button>
            <button type="button" class="call-btn call-keyboard"></button>
            <button type="button" class="call-btn call-end">End</button>
            <button type="button" class="call-btn call-close">Close</button>
        </span>`;
    row.prepend(capsule);
    const $ = selector => capsule.querySelector(selector);
    const words = { state: $('.call-state'), sub: $('.call-sub'), small: $('.call-small') };
    const actions = [...capsule.querySelectorAll('.call-action')];
    const visualHandle = visual.mount({ slot: $('.call-visual'), row, wrap, doc, env });
    const mark = visual.mark(doc);
    let showing = false;     // the capsule holds the row (a call, or a sheet)
    let keyboard = false;    // the owner asked to type during this call
    let big = false;         // the owner grew the capsule into the big view
    let presented = { mode: 'idle' };
    let note = null;         // a sheet's follow-up line (a copied link)
    let dressed = false;     // Talk appears with its own look, never unstyled before call.css applies
    let lastTurn = 'idle';   // the engine's turn as the last render saw it
    let gone = false;
    const talking = createTalkFacts();
    const stage = createCallStage({ wrap, row, time: $('.call-time'), doc, env, escape: () => minimise() });

    // --- Talk in the Send capsule -------------------------------------------------------
    const empty = () => !field.value.trim() && !preview?.querySelector('.attach-badge');
    // Only the feature's explicit beta configuration offers Talk.
    const entryMode = () => (showing ? 'call' : group?.dataset.busy === '1' ? 'busy' : empty() && offered() ? 'talk' : 'send');
    function renderEntry() {
        if (!dressed) return;
        const mode = entryMode();
        if (mode === 'talk') {
            // Built once per face: the chat writes its own label over it when it needs the button.
            if (!send.querySelector(':scope > .call-entry-word')) {
                const word = doc.createElement('span');
                word.className = 'call-entry-word';
                word.textContent = 'Talk';
                send.replaceChildren(mark, word);
                send.title = 'Talk with Ouroboros';
                send.dataset.callEntry = 'talk';
                measureEntry();
            }
        } else if (send.dataset.callEntry === 'talk') {
            send.dataset.callEntry = 'send';
            if (mode !== 'busy') { send.textContent = 'Send'; send.title = 'Send message'; }
        }
    }
    // Talk and Send keep one width, so typing the first letter moves nothing. Talk is measured
    // whenever its size changes: the call's stylesheet arrives, fonts load, a hidden room is shown.
    function measureEntry() {
        if (!offered() || !send.isConnected) return;
        // Approved Talk A preview: measure an unconstrained clone, not Send's
        // previous minimum width. This can shrink after font/viewport changes.
        const probe = send.cloneNode(false);
        probe.removeAttribute('id');
        probe.dataset.callEntry = 'talk';
        probe.style.cssText = 'position:absolute;visibility:hidden;min-width:0';
        const word = doc.createElement('span');
        word.className = 'call-entry-word'; word.textContent = 'Talk';
        probe.append(mark.cloneNode(true), word);
        group.append(probe);
        const measured = probe.offsetWidth;
        probe.remove();
        if (measured > 0) {
            const width = `${measured}px`;
            if (send.style.getPropertyValue('--call-entry-width') !== width) send.style.setProperty('--call-entry-width', width);
        }
    }
    const entrySizing = typeof env.ResizeObserver === 'function' ? new env.ResizeObserver(measureEntry) : null;
    entrySizing?.observe(send);
    const entryObserver = new MutationObserver(renderEntry);
    entryObserver.observe(send, { childList: true });
    const previewObserver = preview ? new MutationObserver(renderEntry) : null;
    previewObserver?.observe(preview, { childList: true });
    const talkPress = (event) => {
        if (!send.contains(event.target) || entryMode() !== 'talk') return;
        // A Talk press belongs to the call: the chat's Send never sees it.
        event.stopPropagation();
        event.preventDefault();
        void talk();
    };
    const enterGuard = (event) => {
        // Enter on an empty field does nothing; only a press starts a call.
        if (event.target !== field || event.key !== 'Enter' || event.shiftKey || event.isComposing
            || event.keyCode === 229 || entryMode() !== 'talk') return;
        event.preventDefault();
        event.stopImmediatePropagation();
    };
    row.addEventListener('click', talkPress, true);
    row.addEventListener('keydown', enterGuard, true);
    field.addEventListener('input', renderEntry);
    field.addEventListener('change', renderEntry);  // dictation and contributed draft edits
    field.addEventListener('keydown', renderEntry);  // after the chat's own history keys
    send.addEventListener('click', renderEntry);     // after the chat's own Send cleared the field

    // --- the capsule ------------------------------------------------------------------
    const on = (selector, handler) => $(selector).addEventListener('click', handler);
    on('.call-mute', () => mute());
    let showMenu = null;     // the open Show menu's binding
    on('.call-show', () => {
        const shown = view().sharing;
        if (shown) { show(shown); return; }
        if (showMenu) { showMenu.close(); return; }
        openShowMenu();
    });
    function openShowMenu() {
        const button = $('.call-show');
        const available = typeof vision === 'function' ? vision() : vision;
        const menu = doc.createElement('div');
        menu.className = 'ui-popup call-show-menu';
        menu.setAttribute('role', 'menu');
        menu.setAttribute('aria-label', 'Show Ouroboros');
        let binding = null;
        for (const [kind, source] of Object.entries(SOURCES)) {
            const item = doc.createElement('button');
            item.type = 'button';
            item.setAttribute('role', 'menuitem');
            item.dataset.source = kind;
            item.innerHTML = `${source.icon}<span class="call-show-words"><span></span><small></small></span>`;
            const why = available[kind] ? source.hint : available[`${kind}Reason`] || available.reason || 'Not available here';
            item.querySelector('.call-show-words > span').textContent = source.label;
            item.querySelector('small').textContent = why;
            item.querySelector('small').hidden = !why;
            item.disabled = !available[kind];
            item.addEventListener('click', () => { binding?.close(); show(kind); });
            menu.append(item);
        }
        doc.body.append(menu);
        button.setAttribute('aria-expanded', 'true');
        binding = bindMenu(menu, { anchor: button, onClose: () => {
            menu.remove();
            button.setAttribute('aria-expanded', 'false');
            if (showMenu === binding) showMenu = null;
        } });
        showMenu = binding;
    }
    on('.call-end', () => end());
    on('.call-close', () => { note = null; dismiss(); });
    on('.call-keyboard', () => {
        // From the big view Keyboard returns to the chip with the field: the big view carries none.
        keyboard = big || !keyboard;
        if (big) { big = false; stage.shrink(); }
        render();
        if (keyboard) field.focus({ preventScroll: true });
    });
    on('.call-size', () => { if (big) minimise(); else expand(); });
    function expand() {
        if (big || presented.mode !== 'live') return;
        keyboard = false;
        big = true;
        stage.grow();
        render();
    }
    function minimise() {
        if (!big) return;
        big = false;
        stage.shrink();
        render();
    }
    actions.forEach((button, index) => button.addEventListener('click', () => {
        const action = presented.actions?.[index];
        if (action) void act(action);
    }));

    // Show: hidden where nothing can be captured; a menu of sources, or the shown source's own stop.
    function renderShow(sheet) {
        const current = view();
        const available = typeof vision === 'function' ? vision() : vision;
        const button = $('.call-show');
        const source = SOURCES[current.sharing];
        button.hidden = sheet || !(available.screen || available.camera);
        button.disabled = !source && current.state !== 'live';
        const face = source ? current.sharing : 'show';
        if (button.dataset.face !== face) {
            button.dataset.face = face;
            button.innerHTML = source ? source.icon : SHOW;
        }
        const label = source ? source.stop : 'Show Ouroboros your screen or camera';
        button.title = label;
        button.setAttribute('aria-label', label);
        if (source) {
            button.setAttribute('aria-pressed', 'true');
            button.removeAttribute('aria-haspopup');
            button.removeAttribute('aria-expanded');
            showMenu?.close();
        } else {
            button.removeAttribute('aria-pressed');
            button.setAttribute('aria-haspopup', 'menu');
            if (!showMenu) button.setAttribute('aria-expanded', 'false');
        }
        if ((button.hidden || button.disabled) && showMenu) showMenu.close();
        const explanation = !sheet && available.unsupported ? available.reason : '';
        words.small.classList.toggle('call-vision-note', Boolean(explanation));
        setWords(words.small, presented.small || explanation || '', !explanation);
    }

    function typingPlaceholder(on) {
        const kept = 'callPlaceholder' in field.dataset;
        if (on && !kept) {
            field.dataset.callPlaceholder = field.placeholder;
            field.placeholder = KEYBOARD_PLACEHOLDER;
        } else if (!on && kept) {
            field.placeholder = field.dataset.callPlaceholder;
            delete field.dataset.callPlaceholder;
        }
    }

    function setWords(element, text, formatted) {
        if (element.textContent !== text) element.textContent = text;
        if (formatted) element.dataset.i18nFmt = 'call'; else delete element.dataset.i18nFmt;
        element.hidden = !text;
    }

    function render() {
        const current = view();
        if (!current.active) keyboard = false;
        if (current.state === 'idle' || current.state === 'ended') {
            talking.reset();
            note = null;
        }
        if (current.turn !== lastTurn) { talking.turn(current.turn); lastTurn = current.turn; }
        presented = callPresentation(current, { ...talking.facts, heard: light.heard(), owner: light.speaking?.() });
        // A call that ends, or a room that leaves it, takes the big view back to the composer at once.
        if (big && presented.mode !== 'live') { big = false; stage.shrink({ instant: true }); }
        stage.clock(current);
        const nextShowing = presented.mode !== 'idle';
        if (nextShowing !== showing) {
            showing = nextShowing;
            capsule.hidden = !showing;
        }
        if (!showing) {
            typingPlaceholder(false);
            for (const name of CALL_STATE) delete wrap.dataset[name];
            light.setMode('still', '');
            renderEntry();
            return;
        }
        const sheet = presented.mode === 'sheet';
        Object.assign(wrap.dataset, { call: presented.mode,
            callView: sheet ? 'sheet' : big ? 'big' : keyboard ? 'keyboard' : 'voice',
            callLight: presented.light, callVoice: presented.voice || '', callTone: presented.tone || '',
            callSharing: current.sharing || '', callWork: presented.work ? '1' : '' });
        light.setMode(presented.light, presented.voice || '');
        setWords(words.state, presented.title, presented.formatted);
        words.state.title = presented.title;
        const followUp = note?.kind === 'copied' ? 'Link copied — paste it into your browser'
            : note?.kind === 'copy_failed' ? fmt('Copy this address into your browser: {url}', { url: note.url }) : '';
        const sharing = SOURCES[current.sharing]?.on || '';
        const subText = followUp || current.visionError || (sharing
            ? `${sharing}${presented.sub ? ` · ${presented.sub}` : ''}` : presented.sub);
        setWords(words.sub, subText, followUp ? note?.kind === 'copy_failed' : Boolean(presented.subFormatted));
        // The words' second line on a phone's chip (call.css): the wave's while a voice speaks (the line only
        // names who, and the wave's colour says it) or while it has nothing to say; else its words.
        wrap.dataset.callLine = !subText || (presented.voiced && !followUp && !current.visionError) ? 'wave' : 'text';
        const size = $('.call-size');
        size.hidden = sheet;
        size.setAttribute('aria-expanded', String(big));
        if (size.title !== SIZE[big ? 'big' : 'chip']) {
            size.title = SIZE[big ? 'big' : 'chip'];
            size.setAttribute('aria-label', size.title);
        }
        $('.call-mute').hidden = sheet;
        $('.call-mute').setAttribute('aria-pressed', String(Boolean(current.muted)));
        renderShow(sheet);
        typingPlaceholder(keyboard && !sheet);
        const toggle = $('.call-keyboard');
        toggle.hidden = sheet;
        const face = keyboard ? 'voice' : 'keyboard';
        if (toggle.dataset.face !== face) {
            toggle.dataset.face = face;
            toggle.innerHTML = TOGGLE[face].icon;
            toggle.setAttribute('aria-label', TOGGLE[face].label);
            toggle.title = TOGGLE[face].label;
        }
        $('.call-end').hidden = sheet;
        $('.call-close').hidden = !sheet;
        actions.forEach((button, index) => {
            const action = presented.actions?.[index];
            button.hidden = !action;
            if (action) setWords(button, action.label, false);
        });
        renderEntry();
    }

    styled.then(() => {
        if (gone) return;
        dressed = true;
        renderEntry();
    });
    return {
        wrap,
        row,
        render,
        entry: () => renderEntry(),
        draw: state => visualHandle.draw(state),
        expand,
        minimise,
        get big() { return big; },
        caption(msg) { talking.caption(msg, light.heard()); render(); },
        voice() { talking.voice(); },
        heard(now) { talking.heard(now); render(); },
        note(kind, url = '') { note = kind ? { kind, url } : null; render(); },
        destroy() {
            gone = true;
            entryObserver.disconnect();
            entrySizing?.disconnect();
            previewObserver?.disconnect();
            row.removeEventListener('click', talkPress, true);
            row.removeEventListener('keydown', enterGuard, true);
            field.removeEventListener('input', renderEntry);
            field.removeEventListener('change', renderEntry);
            field.removeEventListener('keydown', renderEntry);
            send.removeEventListener('click', renderEntry);
            showMenu?.close();
            stage.destroy();
            visualHandle.destroy();
            typingPlaceholder(false);
            capsule.remove();
            for (const name of CALL_STATE) delete wrap.dataset[name];
            if (send.dataset.callEntry === 'talk' && group?.dataset.busy !== '1') { send.textContent = 'Send'; send.title = 'Send message'; }
            delete send.dataset.callEntry;
            send.style.removeProperty('--call-entry-width');
        },
    };
}
