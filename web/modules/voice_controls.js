// The dictation mic of a room's composer (docs/DESIGN.md "Voice controls"). It stands in `[data-composer-voice]`
// only while the chosen speech-to-text engine can run here; "+" → Add dictation… is the way to choose one. A press
// records the engine's way (the host's recorder for a server engine, the browser's recognizer for a client one) and
// the slot becomes the recording face: a level dot, the elapsed time, Stop in the mic's place and Cancel. Stop
// transcribes; the words go in at the caret saved at Stop when the draft has not changed since, otherwise an offer
// above the composer holds them (Insert, Copy). Nothing here sends. Esc and Cancel discard; the room hiding, Panic
// and the engine leaving end a recording. The chevron lists the speech-to-text engines with where their audio goes,
// Get another engine… and Turn off dictation. `startVoice` (app.js, once) wires the page: page modules' providers,
// the "+" item and Read aloud.
import { fmt, tr } from './i18n.js';
import { controlIcon } from './page_icons.js';
import { showToast } from './toast.js';
import { bindMenu } from './ui_interactions.js';
import { copyTextWithToast } from './ui_helpers.js';
import { setDictationOffer } from './composer_plus.js';
import { createDictation } from './dictation.js';
import { openDictationChooser } from './dictation_chooser.js';
import { registerReadAloud } from './read_aloud.js';
import { getSpeech } from './speech_providers.js';

const ACTIVE = new Set(['permission', 'recording', 'transcribing']);
const HOLDS_MIC = new Set(['permission', 'recording']);

export const clockText = (ms) => {
    const seconds = Math.max(0, Math.floor(ms / 1000));
    return `${Math.floor(seconds / 60)}:${String(seconds % 60).padStart(2, '0')}`;
};

/** `text` placed between `value`'s `start` and `end` with one space where it would touch a word. */
export function spacedWords(value, start, text) {
    const before = value.slice(0, start);
    const after = value.slice(start);
    return `${before && !/\s$/.test(before) ? ' ' : ''}${text}${after && !/^\s/.test(after) ? ' ' : ''}`;
}

function button(doc, { className, label, icon = '', text = '', onPress }) {
    const node = doc.createElement('button');
    node.type = 'button';
    node.className = className;
    node.setAttribute('aria-label', label);
    node.title = label;
    if (icon) node.innerHTML = controlIcon(icon, 18);
    if (text) node.textContent = text;
    // A pointer leaves the field its focus and caret (like "+"); keyboard and touch still press.
    node.addEventListener('mousedown', (event) => event.preventDefault());
    node.addEventListener('click', onPress);
    return node;
}

/**
 * Keep one room's dictation mic in step with the chosen engine. `composer` is the room's composer (composer.js);
 * `speech` the page's hub (speech_providers.js `getSpeech`). Returns the disposer.
 */
export function bindVoiceControls({ composer, speech = getSpeech(), notice = (text, tone = 'info') => showToast(text, tone),
    makeDictation = createDictation, chooser = openDictationChooser, copy = copyTextWithToast,
    win = globalThis.window, lang = () => globalThis.navigator?.language || '' }) {
    const slot = composer?.voiceSlot;
    const input = composer?.input;
    if (!slot || !input) return () => {};
    const doc = slot.ownerDocument;
    const wrap = slot.closest?.('.chat-input-wrap') || slot.parentElement;
    let provider = null;  // the chosen speech-to-text engine, when it can run here
    let running = null;  // the engine of the recording in progress
    let face = 'idle';
    let startedAt = 0;
    let clock = 0;
    let meter = null;
    let offer = null;
    let menu = null;
    let revision = 0;
    let disposed = false;

    const onInput = () => { revision += 1; };
    const readDraft = () => {
        const start = input.selectionStart ?? input.value.length;
        return { value: input.value, revision, start, end: input.selectionEnd ?? start };
    };

    function insertAt(text, start, end) {
        input.setSelectionRange?.(start, end);
        composer.insertAtCursor(spacedWords(input.value.slice(0, start) + input.value.slice(end), start, text));
    }

    function closeOffer() {
        offer?.remove();
        offer = null;
    }

    /** The words of a transcription whose draft changed meanwhile: the owner puts them in, or copies them. */
    function showOffer(text) {
        closeOffer();
        const panel = doc.createElement('div');
        panel.className = 'voice-offer';
        panel.setAttribute('role', 'group');
        panel.setAttribute('aria-label', tr('voice.offer.label', 'Dictated text'));
        const note = doc.createElement('p');
        note.className = 'voice-offer-note';
        note.textContent = tr('voice.offer.note', 'The draft changed while this was transcribed.');
        const words = doc.createElement('p');
        words.className = 'voice-offer-text';
        words.setAttribute('aria-live', 'polite');
        words.textContent = text;
        const actions = doc.createElement('div');
        actions.className = 'voice-offer-actions';
        actions.append(
            button(doc, { className: 'btn btn-sm btn-primary', label: tr('voice.offer.insert', 'Insert'),
                text: tr('voice.offer.insert', 'Insert'), onPress: () => {
                    closeOffer();
                    const at = input.selectionStart ?? input.value.length;
                    insertAt(text, at, input.selectionEnd ?? at);
                } }),
            button(doc, { className: 'btn btn-sm btn-default', label: tr('voice.offer.copy', 'Copy'),
                text: tr('voice.offer.copy', 'Copy'), onPress: () => { closeOffer(); void copy(text); } }),
            button(doc, { className: 'voice-offer-close', label: tr('voice.offer.discard', 'Discard dictated text'),
                icon: 'close', onPress: closeOffer }));
        panel.append(note, words, actions);
        panel.addEventListener('keydown', (event) => {
            if (event.key === 'Escape') { event.stopPropagation(); closeOffer(); input.focus?.({ preventScroll: true }); }
        });
        wrap.append(panel);
        offer = panel;
    }

    function stopMeter() {
        meter?.();
        meter = null;
    }

    /** The level dot follows the microphone where the page holds its stream and the engine has an analyser. */
    function startMeter(stream, dot) {
        const Context = win?.AudioContext || win?.webkitAudioContext;
        if (!stream || !Context || !win?.requestAnimationFrame) return;
        try {
            const context = new Context();
            const source = context.createMediaStreamSource(stream);
            const analyser = context.createAnalyser();
            analyser.fftSize = 256;
            source.connect(analyser);
            const samples = new Uint8Array(analyser.fftSize);
            let frame = 0;
            const tick = () => {
                analyser.getByteTimeDomainData(samples);
                let peak = 0;
                for (const sample of samples) peak = Math.max(peak, Math.abs(sample - 128));
                dot.style.setProperty('--voice-level', (peak / 128).toFixed(2));
                frame = win.requestAnimationFrame(tick);
            };
            tick();
            meter = () => {
                win.cancelAnimationFrame?.(frame);
                source.disconnect();
                void context.close().catch(() => {});
            };
        } catch { stopMeter(); }
    }

    const dictation = makeDictation({
        readDraft,
        notice: (text) => notice(text, 'warn'),
        onState({ state, startedAt: at = 0, stream = null }) {
            face = state;
            startedAt = at || startedAt;
            if (state === 'idle') running = null;
            render(stream);
        },
        onTranscript(result) {
            if (result.kind === 'insert') insertAt(result.text, result.start, result.end);
            else if (result.kind === 'offer') showOffer(result.text);
        },
    });

    function record() {
        const chosen = speech.selected('stt');
        if (!chosen.available) { notice(chosen.reason || tr('voice.unavailable', 'Dictation is unavailable here.'), 'warn'); return; }
        running = chosen;
        // The engine starts inside this press: getUserMedia and recognition.start() need the gesture.
        void dictation.start(chosen, { lang: lang(), owner: `dictation:${chosen.key}` });
    }

    function openMenu(anchor) {
        menu?.close();
        const list = doc.createElement('div');
        list.className = 'ui-popup voice-menu';
        list.setAttribute('role', 'menu');
        list.setAttribute('aria-label', tr('voice.menu.label', 'Dictation engine'));
        const chosen = speech.selected('stt').key;
        const item = (role, label, line, onSelect, { checked = null, disabled = false } = {}) => {
            const node = doc.createElement('button');
            node.type = 'button';
            node.className = 'voice-menu-item';
            node.setAttribute('role', role);
            if (checked !== null) node.setAttribute('aria-checked', String(checked));
            if (disabled) node.setAttribute('aria-disabled', 'true');
            const name = doc.createElement('span');
            name.className = 'voice-menu-name';
            name.textContent = label;
            node.append(name);
            if (line) {
                const meta = doc.createElement('span');
                meta.className = 'voice-menu-line';
                meta.textContent = line;
                node.append(meta);
            }
            node.addEventListener('click', () => {
                if (disabled) return;
                menu?.close();
                onSelect();
            });
            list.append(node);
        };
        for (const row of speech.list('stt')) {
            item('menuitemradio', row.label, row.available ? row.destinationLabel : row.reason,
                () => { void speech.choose('stt', row.key).catch((error) => notice(String(error?.message || error), 'danger')); },
                { checked: row.key === chosen, disabled: !row.available });
        }
        item('menuitem', tr('voice.menu.another', 'Get another engine…'), '', () => chooser({ composer }));
        item('menuitem', tr('voice.menu.off', 'Turn off dictation'), '',
            () => { void speech.choose('stt', '').catch((error) => notice(String(error?.message || error), 'danger')); });
        doc.body.append(list);
        const binding = bindMenu(list, { anchor, onClose: () => {
            list.remove();
            if (menu?.list === list) menu = null;
        } });
        menu = { list, close: () => binding.close({ restoreFocus: true }) };
    }

    function render(stream = null) {
        if (disposed) return;
        stopMeter();
        win?.clearInterval?.(clock);
        clock = 0;
        slot.replaceChildren();
        slot.dataset.voiceState = face;
        if (face === 'idle') {
            slot.hidden = !provider;
            if (!provider) return;
            const mic = button(doc, { className: 'voice-btn voice-mic ui-hit-area', label: tr('voice.dictate', 'Dictate'), icon: 'mic',
                onPress: record });
            mic.title = fmt('Dictate · {line}', { line: provider.destinationLabel });
            const more = button(doc, { className: 'voice-btn voice-chevron', label: tr('voice.menu.open', 'Dictation engine'),
                icon: 'chevron', onPress: () => openMenu(more) });
            more.setAttribute('aria-haspopup', 'menu');
            slot.append(mic, more);
            return;
        }
        slot.hidden = false;
        const cancel = button(doc, { className: 'voice-btn voice-cancel', label: tr('voice.cancel', 'Cancel dictation'),
            icon: 'close', onPress: () => dictation.cancel('cancelled') });
        if (face === 'transcribing') {
            const status = doc.createElement('span');
            status.className = 'voice-status';
            status.setAttribute('role', 'status');
            status.textContent = tr('voice.transcribing', 'Transcribing…');
            slot.append(status, cancel);
            return;
        }
        const time = doc.createElement('span');
        time.className = 'voice-time';
        time.setAttribute('role', 'timer');
        time.setAttribute('aria-label', tr('voice.recording', 'Recording'));
        const dot = doc.createElement('span');
        dot.className = 'voice-dot';
        const elapsed = doc.createElement('span');
        elapsed.textContent = clockText(face === 'recording' ? Date.now() - startedAt : 0);
        time.append(dot, elapsed);
        const stop = button(doc, { className: 'voice-btn voice-stop', label: tr('voice.stop', 'Stop and transcribe'),
            icon: 'stop', onPress: () => { void dictation.stop(); } });
        stop.disabled = face !== 'recording';
        slot.append(time, stop, cancel);
        if (face === 'recording') {
            clock = win?.setInterval?.(() => { elapsed.textContent = clockText(Date.now() - startedAt); }, 1000) || 0;
            startMeter(stream, dot);
        }
    }

    /** The chosen engine changed, left or stopped being able to run: the mic follows; a recording on it ends. */
    function sync() {
        const chosen = speech.selected('stt');
        provider = chosen.available ? chosen : null;
        if (running && ACTIVE.has(face) && (!provider || provider.key !== running.key)) {
            dictation.cancel('engine');
            notice(chosen.reason || tr('voice.engine_changed', 'Dictation stopped: the engine changed.'), 'warn');
        }
        if (face === 'idle') render();
    }

    // The room hidden (its page left or its panel closed) releases the microphone; a transcription carries on.
    const hidden = () => !slot.isConnected || !(slot.getClientRects?.().length || input.getClientRects?.().length);
    const onPageShown = () => { if (HOLDS_MIC.has(face) && hidden()) dictation.suspend(); };
    const observer = typeof win?.IntersectionObserver === 'function'
        ? new win.IntersectionObserver(onPageShown) : null;
    observer?.observe(wrap);
    const onPanic = () => { if (ACTIVE.has(face)) dictation.cancel('panic'); };
    const onKey = (event) => {
        if (event.key !== 'Escape' || !ACTIVE.has(face)) return;
        event.preventDefault();
        event.stopPropagation();
        dictation.cancel('cancelled');
    };
    input.addEventListener('input', onInput);
    wrap.addEventListener('keydown', onKey, true);
    win?.addEventListener?.('ouro:page-shown', onPageShown);
    win?.addEventListener?.('ouro:panic', onPanic);
    const offSpeech = speech.onChange(sync);
    sync();

    return () => {
        disposed = true;
        offSpeech();
        observer?.disconnect();
        input.removeEventListener('input', onInput);
        wrap.removeEventListener('keydown', onKey, true);
        win?.removeEventListener?.('ouro:page-shown', onPageShown);
        win?.removeEventListener?.('ouro:panic', onPanic);
        dictation.destroy();
        stopMeter();
        win?.clearInterval?.(clock);
        menu?.close();
        closeOffer();
        slot.replaceChildren();
        slot.hidden = true;
    };
}

/**
 * The page's voice wiring (app.js, once, after the page-module runtime started): client providers join the speech
 * hub, "+" offers Add dictation… while no engine is chosen, and Read aloud joins the saved replies' actions.
 */
export function startVoice({ hosts = null, speech = getSpeech() } = {}) {
    // Shown while no engine can dictate here: none chosen, or the chosen one gone (the chooser says why).
    const offer = { open: ({ composer }) => openDictationChooser({ composer, speech }),
        shown: () => !speech.selected('stt').available };
    let unsetOffer = setDictationOffer(offer);
    // A page module registering its engine changes what "+" shows without a projection change: offer it again.
    const offSpeech = speech.onChange(() => { unsetOffer = setDictationOffer(offer); });
    speech.attachHosts(hosts);
    const offReadAloud = registerReadAloud({ speech });
    return () => { offSpeech(); unsetOffer(); offReadAloud(); };
}
