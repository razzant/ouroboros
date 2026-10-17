/** The composer face of a call. With nothing to send, the Send capsule reads Talk; a press (never
 * Enter) starts the call. During a call the text row keeps its footprint and becomes the live
 * capsule: the visual and the state, Mute, Keyboard, End. Keyboard brings the field back with the
 * visual docked, and what is typed goes as an ordinary turn. When Talk cannot work, or a call
 * ends with something to say, a sheet takes the capsule's place with the words and one next step.
 * Every word the owner reads is chosen here from the session's facts (`callPresentation`). Calls
 * have one engine, Google's Gemini Live.
 */
import { fmt, tx } from './i18n.js';

const KEYBOARD = '<svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" aria-hidden="true"><rect x="2.5" y="6" width="19" height="12" rx="2.5"/><path d="M6.5 10h.01M10 10h.01M14 10h.01M17.5 10h.01M7.5 14h9"/></svg>';
const CALL_STATE = ['call', 'callView', 'callLight', 'callVoice', 'callTone'];

// Why a call reconnects, beside "Reconnecting…".
const RECONNECT_WORDS = {
    session_cap: 'Starting a fresh voice session',
    go_away: 'The voice service is moving the session',
    transport_lost: 'The voice connection dropped',
};
// Where Settings opens for a step: the Google key in Accounts, the Live calls rows, or another tab.
const KEY = { tab: 'providers', focus: 's-google' };
const MODEL = { tab: 'models', focus: 's-call-model-google' };
const BUDGET = { tab: 'behavior', focus: '' };
const REFUSED = { words: 'Google refused the call: “{detail}”', bare: 'Google refused the call', action: 'retry' };
const UNREACHABLE = { words: 'Can’t reach Google’s voice service', action: 'retry' };
const NOT_STARTED = { words: 'The call could not start; the room has the details', action: 'retry' };
const NO_KEY = { words: 'Calls need a Google AI key — add it in Settings', action: 'settings', settings: KEY };
// How a call ends, in the owner's words. For Google's own closes and errors the server's `detail`
// is Google's sentence, quoted as Google's; for the host's own refusals it is the host's sentence,
// never framed as Google's. `bare` is the sentence when there is nothing to say, `action` the one
// next step; `calm` endings are facts, not problems. Codes this page meets itself, and surfaces
// that cannot hold a call (`sheet`), are words too. The server's own `action` wins over this table.
const ENDINGS = {
    owner_bye: { words: 'Call ended', calm: true },
    owner_stop: { words: 'Call ended', calm: true },
    max_minutes: { words: 'The call reached its time limit', calm: true },
    peers_left: { words: 'Everyone left the call', calm: true },
    budget_exceeded: { words: 'The spending limit is reached', action: 'settings', settings: BUDGET },
    missing_credential: NO_KEY,
    missing_credential_google: NO_KEY,
    authentication: { words: 'Google refused the key', action: 'settings', settings: KEY },
    rate_limit: { words: 'Google says: “{detail}”', bare: 'Google asked to slow down', action: 'retry' },
    provider_closed: { words: 'Google closed the call: “{detail}”', bare: 'Google closed the call', action: 'retry' },
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
    invalid_call_max_minutes: { action: 'settings', settings: MODEL },
    budget_exhausted: { action: 'settings', settings: BUDGET },
    permission_denied: { sheet: true, words: 'Allow the microphone for Ouroboros in System Settings → Privacy & Security → '
        + 'Microphone (or in the browser’s site settings), then press Talk again.' },
    no_microphone: { sheet: true, words: 'No microphone was found. Connect one, then press Talk again.' },
    microphone_unreadable: { sheet: true, words: 'The microphone could not be opened. Close other apps that use it, then press Talk again.' },
    microphone_busy: { sheet: true, words: 'Another room is in a call. End that call first.', action: 'reveal' },
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
    if (state === 'connecting') return { ...phase, title: 'Connecting to Google…' };
    if (state === 'reconnecting') return { ...phase, title: 'Reconnecting…', sub: RECONNECT_WORDS[reason(view.code).base] || '' };
    if (state === 'closing') return { ...phase, voice: '', title: 'Ending the call…' };
    const heard = Boolean(live.heard);
    const base = { mode: 'live', light: 'listen', voice: heard ? 'ouroboros' : 'owner', sub: '' };
    if (view.muted) return { ...base, light: 'muted', title: 'Muted', sub: 'Ouroboros can’t hear you' };
    if (heard) return { ...base, title: 'Ouroboros', sub: 'Speaking' };
    if (live.ownerSpeaking) return { ...base, title: 'Listening', sub: 'You’re speaking' };
    // The tool's own label is the host's sentence, shown as it is (or as the install translated it).
    if (view.activity?.working) return { ...base, light: 'think', title: 'Working',
        sub: view.activity.label ? tx(view.activity.label) : 'Using a tool', subFormatted: Boolean(view.activity.label) };
    if (live.thinking) return { ...base, light: 'think', title: 'Thinking' };
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
    if (!view.code || kind === 'finished') return { ...sheet, title: 'The call ended', sub: 'The room has its transcript' };
    if (!entry?.words) return { ...sheet, title: 'The call ended', small: String(view.code) };
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
        heard(now) { if (now) Object.assign(facts, { thinking: false, answered: true }); },
        reset() { Object.assign(facts, { ownerSpeaking: false, thinking: false, answered: false }); },
    };
}

export function createCallCapsule({ page, env = globalThis, visual, light, styled = Promise.resolve(), view, talk, mute,
    end, dismiss, act }) {
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
    capsule.innerHTML = `<span class="call-visual"></span>
        <button type="button" class="call-visual-return" aria-label="Show call controls" title="Show call controls"></button>
        <span class="call-words" role="status" aria-live="polite"><span class="call-state"></span><span class="call-sub"></span><span class="call-small"></span></span>
        <span class="call-buttons">
            <button type="button" class="call-btn call-action"></button>
            <button type="button" class="call-btn call-action"></button>
            <button type="button" class="call-btn call-mute" aria-pressed="false">Mute</button>
            <button type="button" class="call-btn call-keyboard" aria-pressed="false" aria-label="Type while talking" title="Type while talking">${KEYBOARD}</button>
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
    let presented = { mode: 'idle' };
    let note = null;         // a sheet's follow-up line (a copied link)
    let dressed = false;     // Talk appears with its own look, never unstyled before call.css applies
    let gone = false;
    const talking = createTalkFacts();

    // --- Talk in the Send capsule -------------------------------------------------------
    const empty = () => !field.value.trim() && !preview?.querySelector('.attach-badge');
    const entryMode = () => (showing ? 'call' : group?.dataset.busy === '1' ? 'busy' : empty() ? 'talk' : 'send');
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
        if (send.dataset.callEntry !== 'talk' || !(send.offsetWidth > 0)) return;
        const width = `${send.offsetWidth}px`;
        if (send.style.getPropertyValue('--call-entry-width') !== width) send.style.setProperty('--call-entry-width', width);
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
    };
    row.addEventListener('click', talkPress, true);
    row.addEventListener('keydown', enterGuard, true);
    field.addEventListener('input', renderEntry);
    field.addEventListener('keydown', renderEntry);  // after the chat's own history keys
    send.addEventListener('click', renderEntry);     // after the chat's own Send cleared the field

    // --- the capsule ------------------------------------------------------------------
    const on = (selector, handler) => $(selector).addEventListener('click', handler);
    on('.call-mute', () => mute());
    on('.call-end', () => end());
    on('.call-close', () => { note = null; dismiss(); });
    on('.call-keyboard', () => { keyboard = true; render(); field.focus({ preventScroll: true }); });
    on('.call-visual-return', () => { keyboard = false; render(); $('.call-keyboard').focus({ preventScroll: true }); });
    actions.forEach((button, index) => button.addEventListener('click', () => {
        const action = presented.actions?.[index];
        if (action) void act(action);
    }));

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
        presented = callPresentation(current, { ...talking.facts, heard: light.heard() });
        const nextShowing = presented.mode !== 'idle';
        if (nextShowing !== showing) {
            showing = nextShowing;
            capsule.hidden = !showing;
        }
        if (!showing) {
            for (const name of CALL_STATE) delete wrap.dataset[name];
            light.setMode('still', '');
            renderEntry();
            return;
        }
        const sheet = presented.mode === 'sheet';
        Object.assign(wrap.dataset, { call: presented.mode, callView: sheet ? 'sheet' : keyboard ? 'keyboard' : 'voice',
            callLight: presented.light, callVoice: presented.voice || '', callTone: presented.tone || '' });
        light.setMode(presented.light, presented.voice || '');
        setWords(words.state, presented.title, presented.formatted);
        words.state.title = presented.title;
        const followUp = note?.kind === 'copied' ? 'Link copied — paste it into your browser'
            : note?.kind === 'copy_failed' ? fmt('Copy this address into your browser: {url}', { url: note.url }) : '';
        setWords(words.sub, followUp || presented.sub, followUp ? note?.kind === 'copy_failed' : Boolean(presented.subFormatted));
        setWords(words.small, presented.small || '', true);
        $('.call-mute').hidden = sheet;
        $('.call-mute').setAttribute('aria-pressed', String(Boolean(current.muted)));
        $('.call-keyboard').hidden = sheet;
        $('.call-keyboard').setAttribute('aria-pressed', String(keyboard));
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
        draw: state => visualHandle.draw(state),
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
            field.removeEventListener('keydown', renderEntry);
            send.removeEventListener('click', renderEntry);
            visualHandle.destroy();
            capsule.remove();
            for (const name of CALL_STATE) delete wrap.dataset[name];
            if (send.dataset.callEntry === 'talk' && group?.dataset.busy !== '1') { send.textContent = 'Send'; send.title = 'Send message'; }
            delete send.dataset.callEntry;
            send.style.removeProperty('--call-entry-width');
        },
    };
}
