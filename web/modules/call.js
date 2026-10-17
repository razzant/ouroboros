/** A room call: its microphone, socket and playback live until the room is destroyed.
 * The server names every phase and every ending; the page bounds only a socket that never
 * answers at all. Captions name speech items before their audio, so playback acknowledgements
 * count heard samples only. The composer surface, the light, captions and the shell mark are
 * the sibling call_* modules; this one owns the session and puts them together for a room.
 */
import { apiClient } from './api_client.js';
import { openExternalViaHostBridge, shellBridgeApi } from './ui_helpers.js';
import { decodeFrame, encodeFrame, levelFromRms, newerEpoch } from './call_worklet.js';
import { createCallCapsule } from './call_capsule.js';
import { createCallCaptions } from './call_captions.js';
import { createCallLight } from './call_light.js';
import { CALL_VISUAL } from './call_aura.js';
import { trackCallMark } from './call_mark.js';

// Transport bound for a socket that delivers no message at all; every phase reports itself.
const SOCKET_ANSWER_MS = 15000;
export const NEW_APP_URL = 'https://ouroboros-agent.ai/install/';
const ACTIVE = new Set(['permission', 'admission', 'preparing', 'connecting', 'live', 'reconnecting', 'closing']);
const SERVER_STATES = new Set(['preparing', 'connecting', 'live', 'reconnecting', 'closing', 'closed']);
let microphoneOwner = null;
const rooms = new Set();  // every mounted room call, for the shell mark and "Go to the call"

export function callAvailability(env = globalThis) {
    let telegram = Boolean(env.Telegram?.WebApp)
        || env.document?.documentElement?.dataset?.ouroborosHost === 'telegram';
    try {
        telegram ||= Boolean(env.parent?.Telegram?.WebApp)
            || env.parent?.document?.documentElement?.dataset?.ouroborosHost === 'telegram';
    } catch { /* Cross-origin parents expose no host facts. */ }
    if (telegram) return { supported: false, hidden: true, code: 'telegram' };
    if (typeof env.navigator?.mediaDevices?.getUserMedia !== 'function') {
        // The desktop app's own WebView hides the microphone when its build cannot ask for it.
        if (shellBridgeApi(env)) return { supported: false, code: 'old_app' };
        return { supported: false, code: env.isSecureContext === false ? 'insecure' : 'no_microphone_api' };
    }
    if (!env.isSecureContext) return { supported: false, code: 'insecure' };
    if (!env.AudioContext || !env.AudioWorkletNode || !('audioWorklet' in env.AudioContext.prototype))
        return { supported: false, code: 'no_live_audio' };
    if (!env.WebSocket) return { supported: false, code: 'no_call_socket' };
    return { supported: true, code: '' };
}

/** A media error as the page's own code; the capsule says it in words. */
export function mediaFailure(error) {
    return { NotAllowedError: 'permission_denied', SecurityError: 'permission_denied', NotFoundError: 'no_microphone',
        OverconstrainedError: 'no_microphone', NotReadableError: 'microphone_unreadable' }[error?.name] || 'call_failed';
}

function outputLatency(context) {
    const seconds = Number(context?.outputLatency) || Number(context?.baseLatency) || 0;
    return seconds > 0 ? seconds : 0;
}

export class CallSession {
    constructor({ chatId, env = globalThis, open = apiClient.openCall,
        socket = path => apiClient.callSocket(path), changed = () => {}, caption = () => {},
        level = () => {}, settled = () => {} }) {
        this.chatId = chatId;
        this.env = env;
        this.open = open;
        this.makeSocket = socket;
        this.changed = changed;
        this.caption = caption;
        this.level = level;
        this.settled = settled;
        this.state = 'idle';
        this.generation = 0;
        this.captureGeneration = 0;
        this.items = new Map();
        this.played = new Map();
        this.seq = 0;
        this.epoch = 0;
        this.muted = false;
        this.clear();
    }

    /** Facts of the current attempt; a new attempt starts from none. */
    clear() {
        Object.assign(this, { callId: '', code: '', detail: '', action: '', words: '', phase: '',
            activity: { working: false, label: '' }, liveSince: 0 });
    }

    get active() { return ACTIVE.has(this.state); }

    view() {
        const { state, code, detail, action, words, phase, muted, callId, liveSince } = this;
        return { state, code, detail, action, words, phase, muted, callId, liveSince,
            activity: { ...this.activity }, active: this.active };
    }

    setState(state, facts = {}) {
        Object.assign(this, facts);
        this.state = state;
        if (state === 'live' && !this.liveSince) this.liveSince = Date.now();
        this.changed(this.view());
    }

    send(message) {
        if (this.socket?.readyState === 1) this.socket.send(JSON.stringify(message));
    }

    async start() {
        if (this.active) return;
        this.clear();
        const capability = callAvailability(this.env);
        if (!capability.supported) { this.setState('failed', { code: capability.code }); return; }
        if (microphoneOwner && microphoneOwner !== this) { this.setState('failed', { code: 'microphone_busy' }); return; }
        microphoneOwner = this;
        const generation = ++this.generation;
        const current = () => this.generation === generation;
        this.items.clear(); this.played.clear();
        this.seq = this.epoch = 0;
        this.muted = false;
        this.setState('permission');
        try {
            const context = new this.env.AudioContext({ latencyHint: 'interactive' });
            this.context = context;
            // Resume during the gesture, before permission or admission can consume it on iOS.
            const resume = context.resume();
            resume.catch(() => {});
            const media = await this.env.navigator.mediaDevices.getUserMedia({ audio: {
                channelCount: 1, echoCancellation: true, noiseSuppression: true, autoGainControl: true }, video: false });
            if (!current()) { media.getTracks().forEach(track => track.stop()); return; }
            this.media = media;
            for (const track of media.getAudioTracks()) track.onended = () => this.fail('microphone_lost');
            this.setState('admission');
            await resume;
            if (!current()) return;
            await context.audioWorklet.addModule(new URL('./call_worklet.js', import.meta.url));
            if (!current()) return;
            const node = new this.env.AudioWorkletNode(context, 'ouroboros-call-audio', {
                numberOfInputs: 1, numberOfOutputs: 1, outputChannelCount: [1], channelCount: 1 });
            this.node = node;
            node.onprocessorerror = () => this.fail('audio_failed');
            node.port.onmessage = ({ data }) => { if (current()) this.onAudio(data); };
            this.source = context.createMediaStreamSource(media);
            this.source.connect(node);
            node.connect(context.destination);
            const result = await this.open({ chat_id: this.chatId, client_message_id: this.env.crypto.randomUUID() });
            if (!result?.call_id || typeof result.ws_path !== 'string') throw new Error('The server did not return a call address.');
            if (!current()) { this.releaseAdmission(result.ws_path, context.sampleRate); return; }
            this.callId = String(result.call_id);
            const socket = this.makeSocket(result.ws_path);
            this.socket = socket;
            socket.binaryType = 'arraybuffer';
            this.timer = this.env.setTimeout(() => this.fail('socket_silent'), SOCKET_ANSWER_MS);
            socket.onopen = () => {
                if (!current()) return;
                this.send({ type: 'hello', protocol: 1, role: 'owner', mic: true, speaker: true,
                    rate: context.sampleRate, label: 'Owner' });
                this.capture();
            };
            socket.onmessage = ({ data }) => {
                if (!current()) return;
                this.env.clearTimeout(this.timer);  // the socket answers: phases now speak for themselves
                try { this.receive(data); } catch { this.fail('protocol_mismatch'); }
            };
            socket.onerror = () => { if (current()) this.fail('socket_closed'); };
            socket.onclose = (event) => {
                if (current()) this.fail('socket_closed', { detail: String(event?.reason || ''), closeCode: event?.code || 0 });
            };
        } catch (error) {
            if (!current()) return;
            // An admission refusal carries the server's code and the owner's words.
            if (error?.body?.ok === false || error?.status) this.fail(String(error.body?.code || 'call_not_started'),
                { words: String(error.body?.error || error.message || '') });
            else this.fail(mediaFailure(error));
        }
    }

    /** A leave during admission still closes the admitted turn when its address arrives. */
    releaseAdmission(path, rate) {
        const socket = this.makeSocket(path);
        const timer = this.env.setTimeout(() => socket.close(), SOCKET_ANSWER_MS);
        socket.onopen = () => {
            socket.send(JSON.stringify({ type: 'hello', protocol: 1, role: 'owner', mic: false, speaker: false, rate, label: 'Owner' }));
            socket.send(JSON.stringify({ type: 'bye', reason: 'left' }));
            socket.close();
        };
        socket.onclose = socket.onerror = () => this.env.clearTimeout(timer);
    }

    capture() {
        const enabled = !this.muted && ['live', 'reconnecting'].includes(this.state);
        for (const track of this.media?.getAudioTracks() || []) track.enabled = enabled;
        this.node?.port.postMessage({ type: 'capture', enabled, generation: ++this.captureGeneration });
    }

    setMuted(muted) {
        if (!this.active) return;
        this.muted = Boolean(muted);
        this.capture();
        this.send({ type: 'mute', muted: this.muted });
        this.changed(this.view());
    }

    onAudio(data) {
        if (data.type === 'error') { this.fail('audio_failed', { detail: String(data.reason || '') }); return; }
        if (data.type === 'level') {
            this.level({ mic: levelFromRms(data.mic), out: levelFromRms(data.out), playing: Boolean(data.playing),
                latency: outputLatency(this.context) });
            return;
        }
        if (data.type === 'played') {
            const last = this.played.get(data.item_id) || 0;
            if (data.item_id && Number.isInteger(data.played_ms) && data.played_ms > last) {
                this.played.set(data.item_id, data.played_ms);
                this.send({ type: 'played', item_id: data.item_id, played_ms: data.played_ms });
            }
        }
        if (data.type !== 'capture' || this.muted || data.generation !== this.captureGeneration
            || !['live', 'reconnecting'].includes(this.state) || this.socket?.readyState !== 1) return;
        // One second of unsent microphone PCM is a broken uplink, not useful live speech.
        if (this.socket.bufferedAmount > data.rate * 2) { this.fail('slow_uplink'); return; }
        this.socket.send(encodeFrame({ epoch: this.epoch, seq: this.seq++, rate: data.rate, pcm: data.pcm }));
    }

    receive(data) {
        if (data instanceof ArrayBuffer) {
            if (this.state === 'closing') return;
            const frame = decodeFrame(data);
            if (frame.kind !== 2) throw new Error('The server sent microphone audio as playback.');
            if (newerEpoch(frame.epoch, this.epoch)) this.flush(frame.epoch);
            if (frame.epoch !== this.epoch) return;
            const item_id = this.items.get(frame.stream);
            if (!item_id) throw new Error('Playback arrived without its speech item.');
            this.node.port.postMessage({ type: 'audio', ...frame, item_id }, [frame.pcm.buffer]);
            return;
        }
        if (typeof data !== 'string' || new TextEncoder().encode(data).length > 16 * 1024)
            throw new Error('The server sent an invalid call message.');
        const msg = JSON.parse(data);
        if (msg.type === 'state') this.receiveState(msg);
        else if (msg.type === 'activity') {
            this.activity = { working: msg.activity === 'working', label: String(msg.label || '').slice(0, 200) };
            this.changed(this.view());
        } else if (msg.type === 'flush') {
            if (!Number.isInteger(msg.epoch) || msg.epoch < 0 || msg.epoch > 65535) throw new Error('Invalid playback epoch.');
            this.flush(msg.epoch);
        } else if (msg.type === 'caption') {
            if (!['owner', 'ouroboros'].includes(msg.speaker) || typeof msg.text !== 'string'
                || typeof msg.item_id !== 'string') throw new Error('Invalid live caption.');
            if (msg.epoch !== undefined && msg.epoch !== this.epoch) return;
            // A final caption may arrive after the next item's audio has begun; it cannot rebind that stream.
            if (msg.speaker === 'ouroboros' && msg.item_id && !msg.final) {
                const stream = msg.stream ?? 0;
                if (!Number.isInteger(stream) || stream < 0 || stream > 65535) throw new Error('Invalid speech stream.');
                this.items.set(stream, msg.item_id);
            }
            this.caption({ ...msg, call_id: this.callId });
        }
        // A message type this page does not know belongs to a newer server: it changes nothing here.
    }

    receiveState(msg) {
        if (!SERVER_STATES.has(msg.state)) return;  // a newer server's phase; the next known one follows
        const facts = { code: String(msg.reason || '') };
        for (const key of ['phase', 'detail', 'action']) if (typeof msg[key] === 'string') facts[key] = msg[key];
        if (msg.state === 'closed') { this.stop('closed', facts); return; }
        if (msg.state === 'live') Object.assign(facts, { phase: facts.phase || '', detail: facts.detail || '' });
        this.setState(msg.state, facts);
        this.capture();
        if (msg.state === 'closing') this.node?.port.postMessage({ type: 'flush', epoch: (this.epoch + 1) & 65535 });
    }

    flush(epoch) {
        if (!newerEpoch(epoch, this.epoch)) return;
        this.epoch = epoch;
        this.items.clear();
        this.node?.port.postMessage({ type: 'flush', epoch });
    }

    fail(code, facts = {}) { this.stop('failed', { code, ...facts }); }

    /** The owner's own End: everything stops and the composer comes back. */
    end() { if (this.active) this.stop('ended', { code: 'owner_end' }); }

    /** Back to the composer after the owner read how the last call ended. */
    dismiss() {
        if (this.active) return;
        this.clear();
        this.setState('idle');
    }

    stop(state = 'closed', facts = {}) {
        const wasActive = this.active;
        this.generation++;
        this.env.clearTimeout(this.timer);
        for (const track of this.media?.getTracks() || []) { track.onended = null; track.enabled = false; track.stop(); }
        this.media = null;
        this.source?.disconnect(); this.source = null;
        if (this.node) {
            this.node.port.onmessage = null;
            this.node.port.close(); this.node.disconnect(); this.node = null;
        }
        if (this.context) { this.context.close().catch(() => {}); this.context = null; }
        if (this.socket) {
            this.send({ type: 'bye', reason: state === 'failed' ? 'client_error' : 'left' });
            this.socket.onopen = this.socket.onmessage = this.socket.onerror = this.socket.onclose = null;
            this.socket.close(); this.socket = null;
        }
        this.items.clear(); this.played.clear();
        if (microphoneOwner === this) microphoneOwner = null;
        this.activity = { working: false, label: '' };
        this.setState(state, facts);
        if (wasActive) this.settled(this.view());
    }
}

/** The room call: Talk in the composer, the live capsule, its light, caption bubbles and the mark. */
export function mountCall({ page, chatId, env = globalThis, ws = null, label = () => 'Main',
    onScreen = () => true, reveal = () => {}, openSettings = () => {}, onSettled = () => {} }) {
    const doc = page.ownerDocument;
    const composer = page.querySelector('.chat-input-wrap');
    const available = callAvailability(env);
    const noop = { destroy() {}, leave() {}, retains: () => false };
    if (!composer || available.hidden) return noop;
    const styled = ensureStylesheet(doc);
    let destroyed = false;
    let entry = null;
    const visible = () => !destroyed && onScreen() && doc.visibilityState !== 'hidden';
    const light = createCallLight({ env, visible, heard: now => capsule.heard(now), voice: () => capsule.voice(),
        draw: state => capsule.draw(state) });
    const captions = createCallCaptions({ feed: page.querySelector('#chat-messages, .chat-messages'), ws, doc });
    const session = new CallSession({ chatId, env,
        changed(view) { if (destroyed) return; light.setActive(view.active); capsule.render(); entry?.changed(); },
        caption(msg) { captions.update(msg); capsule.caption(msg); },
        level(levels) { light.level(levels); },
        settled(view) { captions.finish(); onSettled(view); },
    });
    // Talk starts a call; a surface that cannot hold one answers the press with its sheet.
    const capsule = createCallCapsule({ page, env, visual: CALL_VISUAL, light, styled, view: () => session.view(),
        talk: () => session.start(),
        mute: () => session.setMuted(!session.muted),
        end: () => session.end(),
        dismiss: () => session.dismiss(),
        act: action => act(action) });
    light.attach(capsule.wrap);
    entry = trackCallMark({ session, label, onScreen, reveal, page, env, sign: () => CALL_VISUAL.mark(doc),
        wake: () => light.wake() });
    rooms.add(entry);

    async function act(action) {
        if (action.kind === 'retry') return session.start();
        if (action.kind === 'reveal') return [...rooms].find(room => room.session.active && room !== entry)?.reveal();
        if (action.kind === 'settings') {
            session.dismiss();
            await openSettings(action.tab);
            const target = action.focus && doc.getElementById(action.focus);
            target?.closest('details')?.setAttribute('open', '');
            target?.scrollIntoView?.({ block: 'center' });
            target?.focus?.({ preventScroll: true });
            return undefined;
        }
        if (action.kind === 'open_browser' || action.kind === 'get_app') {
            const url = action.kind === 'get_app' ? NEW_APP_URL : String(env.location?.href || '');
            try {
                const opened = await openExternalViaHostBridge(url, { win: env, doc, toast: () => {} });
                capsule.note(opened?.degraded === 'copy-link' ? 'copied' : '');
            } catch { capsule.note('copy_failed', url); }
        }
        return undefined;
    }

    // Panic and Restart end the call only once confirmed: the server's own stop closes the hub, and a
    // reload leaves the page (pagehide). A press the owner then cancels keeps the call.
    const end = () => session.end();
    env.addEventListener?.('pagehide', end);
    return {
        session,
        leave: end,
        retains: () => session.active,
        destroy() {
            if (destroyed) return;
            end();
            destroyed = true;
            env.removeEventListener?.('pagehide', end);
            rooms.delete(entry);
            entry.destroy();
            light.destroy();
            captions.destroy();
            capsule.destroy();
        },
    };
}

/** The call's stylesheet, once per document; resolves when it applies (or failed to load). */
function ensureStylesheet(doc) {
    let link = doc.querySelector('link[data-call-style]');
    if (!link) {
        link = doc.createElement('link');
        link.rel = 'stylesheet'; link.href = new URL('../call.css', import.meta.url).href;
        link.dataset.callStyle = '';
        doc.head.append(link);
    }
    if (link.dataset.loaded || link.sheet) return Promise.resolve();
    return new Promise((resolve) => {
        const done = () => { link.dataset.loaded = '1'; resolve(); };
        link.addEventListener('load', done, { once: true });
        link.addEventListener('error', done, { once: true });
    });
}

/** A room's call keeps the room while it lasts: hiding the room keeps the call (like a phone
 * call), destroying it hangs up, and `hasPendingWork` holds a Project room for the whole
 * non-terminal lifetime. When the call ends, `release` lets go of a hidden room nothing else holds. */
export function mountRoomCall({ room, mount = mountCall, release = () => {}, ...options }) {
    const call = mount({ ...options, onSettled: () => queueMicrotask(release) });
    const { destroy, hasPendingWork } = room;
    room.destroy = () => { call.destroy(); destroy(); };
    room.hasPendingWork = () => call.retains() || Boolean(hasPendingWork?.());
    return call;
}
