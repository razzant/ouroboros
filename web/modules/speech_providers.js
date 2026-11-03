/** Provider facts and strict saved selection; sources are injected by the page.
 * source(kind) returns live contributions; clients() returns {key, kind, impl}.
 * No skill is enabled, selected or substituted by reading this registry.
 * getSpeech() is the page's hub over the contributions store and page modules.
 */
import { apiClient, fetchJson, jsonPost } from './api_client.js';
import { getContributions } from './contributions.js';
import { captureAvailability, captureReason, isLoopback } from './media_capture.js';
import { recordingFormat } from './voice_recorder.js';

export function destinationLabel(destination, location = globalThis.location) {
    if (destination === 'device') return 'On this device';
    if (destination === 'host') return isLoopback(location?.hostname) ? 'On this computer'
        : `On your Ouroboros server (${location?.host || location?.hostname || 'unknown host'})`;
    if (destination?.startsWith('vendor:')) return `Goes to ${destination.slice(7)}`;
    return 'Audio destination unavailable';
}

export const voiceDestination = voice => voice?.local === true ? 'On this device' : 'Online voice';

function availability(provider, env) {
    if (provider.fault) return { ok: false, code: 'fault', reason: String(provider.fault) };
    if (provider.runtime === 'server') {
        if (!provider.route_url) return { ok: false, code: 'missing_route' };
        if (provider.provides !== 'stt') return { ok: false, code: 'unsupported_runtime' };
        const access = captureAvailability(env);
        return access.ok && !recordingFormat(provider.formats, env)
            ? { ok: false, code: 'unsupported_format' } : access;
    }
    const impl = provider.impl;
    if (!impl) return { ok: false, code: 'not_loaded' };
    const methods = provider.provides === 'stt' ? ['start', 'available'] : ['voices', 'speak', 'stop'];
    if (methods.some(name => typeof impl[name] !== 'function')) return { ok: false, code: 'invalid_provider' };
    if (provider.provides === 'stt') {
        const access = captureAvailability(env, { capture: false });
        if (!access.ok) return access;
    }
    try { return impl.available ? impl.available() : { ok: true }; }
    catch (error) { return { ok: false, code: 'fault', reason: error.message }; }
}

export function providerReason(code) {
    return {
        missing: 'The chosen engine is disabled, removed or awaiting review.',
        not_loaded: 'The chosen engine has not started. Check its skill or choose another engine.',
        unsupported_format: 'This browser cannot record a format the engine accepts.',
        unsupported_runtime: 'This engine cannot read aloud on this surface.',
        missing_route: 'The engine has no live speech route.',
        invalid_provider: 'The engine did not register a usable speech provider.',
        no_speech_api: 'Browser speech is unavailable in this browser or desktop app.',
        no_voice_api: 'Read aloud is unavailable in this browser or desktop app.',
    }[code] || captureReason(code);
}

export function createSpeechProviders({ source = () => [], clients = () => [],
    preferences = () => ({}), saveUiPreferences, env = globalThis } = {}) {
    const registered = new Map();
    function list(kind) {
        const liveClients = new Map([...clients(), ...registered.values()]
            .filter(row => row.kind === kind).map(row => [row.key, row.impl]));
        return source(kind).filter(row => row.provides === kind).map(row => {
            const impl = row.runtime === 'client' ? liveClients.get(row.key) : undefined;
            const provider = { ...row, impl, label: impl?.label || row.label,
                destination: impl?.destination || row.destination };
            const state = availability(provider, env);
            return { ...provider, available: state?.ok === true, code: state?.code || '',
                reason: state?.ok ? '' : state?.reason || providerReason(state?.code),
                destinationLabel: destinationLabel(provider.destination, env.location) };
        });
    }
    return {
        list,
        register(key, kind, impl) {
            if (!['stt', 'tts'].includes(kind)) throw new Error('Unknown speech provider kind');
            const record = { key, kind, impl };
            registered.set(key, record);
            return () => { if (registered.get(key) === record) registered.delete(key); };
        },
        selected(kind) {
            const key = preferences()?.voice?.[kind] || '';
            if (!key) return { key: '', available: false, code: 'not_selected', reason: '' };
            return list(kind).find(row => row.key === key)
                || { key, available: false, code: 'missing', reason: providerReason('missing') };
        },
        choose(kind, key) {
            if (!['stt', 'tts'].includes(kind)) throw new Error('Unknown speech provider kind');
            return saveUiPreferences({ voice: { [kind]: key } });
        },
    };
}

/**
 * The page's speech engines: the store's live providers (a page-module fault included) merged with the client
 * providers page modules registered, the saved choices, and one change notice for every control that shows them.
 * Saving a choice re-reads the projection, so every room sees it. `enableOffer` turns a bundled engine on through
 * the Skills toggle and then chooses it; nothing else enables or substitutes an engine.
 */
export function createSpeechHub({ store, save = payload => apiClient.saveUiPreferences(payload), post = jsonPost,
    emit = (name, result) => globalThis.dispatchEvent?.(new CustomEvent('ouro:skill-lifecycle',
        { detail: { action: 'enable', name, ...result } })), env = globalThis } = {}) {
    let hosts = null;
    let offHosts = () => {};
    const listeners = new Set();
    const notify = () => {
        for (const listener of [...listeners]) {
            try { listener(); } catch (error) { console.error('speech listener failed', error); }
        }
    };
    const registry = createSpeechProviders({ env, saveUiPreferences: save,
        source: kind => store.providers(kind).map(row => ({ ...row, fault: store.fault(row.key) })),
        clients: () => (hosts ? [...hosts.clientProviders('stt'), ...hosts.clientProviders('tts')] : []),
        preferences: () => store.preferences() });
    store.onChange(notify);
    async function remember(voice) {
        await save({ voice });
        await store.refresh();
        notify();
    }
    const voicePref = () => store.preferences()?.voice || {};
    return {
        list: registry.list,
        selected: registry.selected,
        offers: kind => store.offers(kind),
        /** Follow the page modules' client providers (host_modules.js runtime). */
        attachHosts(next) {
            offHosts();
            hosts = next || null;
            offHosts = hosts?.onProvidersChange?.(notify) || (() => {});
            notify();
        },
        onChange(listener) {
            listeners.add(listener);
            return () => listeners.delete(listener);
        },
        /** Choosing another Read aloud engine starts from its own voice pick. */
        choose: (kind, key) => remember(kind === 'tts' ? { tts: key, tts_voice: '' } : { [kind]: key }),
        chooseVoice: id => remember({ tts_voice: String(id || '') }),
        voiceId: () => String(voicePref().tts_voice || ''),
        /** The Read aloud engine: the chosen one (even when it cannot run: it says why), else a live one on this
         * device; null when there is none (an engine that sends the words away is only ever chosen). */
        readAloud() {
            const chosen = registry.selected('tts');
            if (chosen.key) return chosen;
            return registry.list('tts').find(row => row.available && row.destination === 'device') || null;
        },
        async enableOffer(offer) {
            const result = await post(`/api/skills/${encodeURIComponent(offer.skill)}/toggle`, { enabled: true });
            if (result?.extension_action === 'extension_load_error')
                throw new Error(result.extension_reason || `${offer.skill_label || offer.skill} did not start.`);
            emit(offer.skill, result);
            await remember({ stt: `${offer.skill}:${offer.id}` });
            return result;
        },
    };
}

let shared = null;

/** The page's one speech hub (app.js attaches the page-module runtime). */
export function getSpeech() {
    if (!shared) shared = createSpeechHub({ store: getContributions() });
    return shared;
}

/** Hub speech routes accept raw audio and return text or the explicit no_speech status. */
export async function transcribeAudio(provider, audio, { signal, fetchImpl } = {}) {
    if (audio.size > Math.min(provider.max_bytes || 524288, 524288))
        throw Object.assign(new Error('Recording too large. Try a shorter phrase.'), { code: 'too_large' });
    if (!audio.size) return { no_speech: true };
    const result = await fetchJson(provider.route_url, { method: 'POST',
        headers: { 'Content-Type': audio.type }, body: audio, signal, cache: 'no-store' }, { fetchImpl });
    if (result?.status === 'no_speech' && result.text === '') return { no_speech: true };
    if (typeof result?.text !== 'string' || !result.text.trim())
        throw new Error(result?.error?.message || 'The engine returned no transcript.');
    return { text: result.text.trim() };
}
