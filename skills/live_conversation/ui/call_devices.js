/** The call's devices on this client: which microphone, speaker and camera a call uses, and how the
 * microphone is cleaned (echo cancellation, noise suppression, automatic volume). Device ids belong
 * to one browser on one machine, so the choice lives here, never on the server; a chosen device that
 * is gone (unplugged, or an id from another browser) falls back to the system's own, and the caller
 * says so. A choice made during a call reaches it through `DEVICES_CHANGED`.
 */

const STORAGE_KEY = 'ouroboros.call.devices';
export const DEVICES_CHANGED = 'ouro:call-devices-changed';
export const DEFAULT_DEVICE_PREFS = Object.freeze({ mic: '', speaker: '', camera: '',
    echoCancellation: true, noiseSuppression: true, autoGainControl: true });
const DEVICE_KEYS = ['mic', 'speaker', 'camera'];
const PROCESSING_KEYS = ['echoCancellation', 'noiseSuppression', 'autoGainControl'];
// Browsers' own aliases for "whatever the system uses"; the empty choice already means that.
const ALIASES = new Set(['', 'default', 'communications']);
const kept = new WeakMap();  // a client that keeps no storage holds its choice while its page lives

function clean(raw) {
    const prefs = { ...DEFAULT_DEVICE_PREFS };
    for (const key of DEVICE_KEYS) if (typeof raw?.[key] === 'string') prefs[key] = raw[key];
    for (const key of PROCESSING_KEYS) if (typeof raw?.[key] === 'boolean') prefs[key] = raw[key];
    return prefs;
}

export function loadDevicePrefs(env = globalThis) {
    try {
        const stored = env.localStorage?.getItem(STORAGE_KEY);
        if (stored) return clean(JSON.parse(stored));
    } catch { /* unreadable storage falls through to this page's own choice */ }
    return clean(kept.get(env) || {});
}

export function saveDevicePrefs(change, env = globalThis) {
    const next = clean({ ...loadDevicePrefs(env), ...change });
    if (env && typeof env === 'object') kept.set(env, next);
    try { env.localStorage?.setItem(STORAGE_KEY, JSON.stringify(next)); } catch { /* kept for this page */ }
    if (typeof env.CustomEvent === 'function') env.dispatchEvent?.(new env.CustomEvent(DEVICES_CHANGED, { detail: next }));
    return next;
}

export function microphoneConstraints(prefs) {
    const audio = { channelCount: 1 };
    for (const key of PROCESSING_KEYS) audio[key] = prefs[key];
    if (prefs.mic) audio.deviceId = { exact: prefs.mic };
    return { audio, video: false };
}

export function cameraConstraints(prefs) {
    return { video: prefs.camera ? { deviceId: { exact: prefs.camera } } : true, audio: false };
}

const gone = error => ['OverconstrainedError', 'NotFoundError'].includes(error?.name);

/** The chosen microphone, or the system's own when the chosen one is gone: `{ stream, fellBack }`. */
export async function openMicrophone(env, prefs = loadDevicePrefs(env)) {
    const media = env.navigator.mediaDevices;
    try {
        return { stream: await media.getUserMedia(microphoneConstraints(prefs)), fellBack: false };
    } catch (error) {
        if (!prefs.mic || !gone(error)) throw error;
        return { stream: await media.getUserMedia(microphoneConstraints({ ...prefs, mic: '' })), fellBack: true };
    }
}

/** The chosen camera, or the system's own when the chosen one is gone: `{ stream, fellBack }`. */
export async function openCamera(env, prefs = loadDevicePrefs(env)) {
    const media = env.navigator.mediaDevices;
    try {
        return { stream: await media.getUserMedia(cameraConstraints(prefs)), fellBack: false };
    } catch (error) {
        if (!prefs.camera || !gone(error)) throw error;
        return { stream: await media.getUserMedia(cameraConstraints({ ...prefs, camera: '' })), fellBack: true };
    }
}

/** Whether this engine lets a page choose its speaker (Chromium does; Safari and the macOS app follow the system). */
export function speakerChoice(env = globalThis) {
    return typeof env.AudioContext?.prototype?.setSinkId === 'function';
}

/** Send `context`'s sound to the chosen speaker where the engine can: 'routed', 'system' (no choice
 * here, or the system's own chosen) or 'fell_back' (the chosen speaker is gone; the system's plays). */
export async function routeSpeaker(context, prefs) {
    if (typeof context?.setSinkId !== 'function') return 'system';
    if (ALIASES.has(prefs.speaker)) {
        // The system's own speaker: a context routed to a chosen one earlier in the call goes back to it.
        if (context.sinkId) await Promise.resolve(context.setSinkId('')).catch(() => {});
        return 'system';
    }
    try {
        await context.setSinkId(prefs.speaker);
        return 'routed';
    } catch {
        await Promise.resolve(context.setSinkId('')).catch(() => {});
        return 'fell_back';
    }
}

/** This client's devices by kind. Browsers name them only after a microphone or camera permission;
 * `named` says whether they did. */
export async function listDevices(env = globalThis) {
    const out = { audioinput: [], audiooutput: [], videoinput: [], named: false };
    const devices = await env.navigator?.mediaDevices?.enumerateDevices?.() || [];
    for (const device of devices) {
        if (!(device.kind in out) || ALIASES.has(device.deviceId)) continue;
        out[device.kind].push({ id: device.deviceId, label: device.label || '' });
        if (device.label) out.named = true;
    }
    return out;
}

/** Whether a change of choice needs a new microphone track (another device or other cleaning). */
export function microphoneChanged(before, after) {
    return ['mic', ...PROCESSING_KEYS].some(key => before?.[key] !== after?.[key]);
}
