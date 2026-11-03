/** Local voices first; localService is the only authority for a local label. */
export function createDeviceVoice(env = globalThis) {
    const synth = env.speechSynthesis;
    const Utterance = env.SpeechSynthesisUtterance;
    let active;
    function voices() {
        return [...(synth?.getVoices() || [])].sort((a, b) => Number(b.localService === true) - Number(a.localService === true));
    }
    function stop() {
        if (!active) return;
        const current = active;
        // A native cancel may omit onend altogether. Settle our own promise first.
        current.finish();
        synth.cancel();
    }
    return {
        label: 'Device voice',
        available: () => synth && Utterance ? { ok: true } : { ok: false, code: 'no_voice_api' },
        voices: () => voices().map(voice => ({ id: voice.voiceURI, label: voice.name, lang: voice.lang,
            local: voice.localService === true })),
        speak(text, { voice, lang, signal } = {}) {
            stop();
            if (!synth || !Utterance) return Promise.reject(new Error('Read aloud is unavailable here.'));
            if (signal?.aborted) return Promise.resolve();
            const options = voices();
            // Without a chosen voice only a local one speaks: an online voice sends the words away, so it is chosen.
            const selected = voice ? options.find(item => item.voiceURI === voice)
                : options.find(item => item.localService === true && (!lang || item.lang === lang))
                    || options.find(item => item.localService === true);
            if (voice && !selected) return Promise.reject(new Error('The chosen voice is no longer available. Choose another voice.'));
            // Waiting for the browser to load its voices is explicit, never a guessed local voice.
            if (!options.length) return Promise.reject(new Error('No voices are available yet. Try Read aloud again.'));
            if (!selected) return Promise.reject(new Error('No voice on this device. To use an online voice, choose it in the Read aloud menu.'));
            return new Promise((resolve, reject) => {
                const utterance = new Utterance(text);
                utterance.voice = selected;
                utterance.lang = lang || selected.lang;
                const current = { finish(error) {
                    if (active !== current) return;
                    active = null;
                    utterance.onend = utterance.onerror = null;
                    signal?.removeEventListener('abort', stop);
                    if (error) reject(error); else resolve();
                } };
                active = current;
                utterance.onend = () => current.finish();
                utterance.onerror = event => current.finish(new Error(`Read aloud stopped: ${event.error || 'speech error'}.`));
                signal?.addEventListener('abort', stop, { once: true });
                try { synth.speak(utterance); }
                catch (error) { current.finish(error); }
            });
        }, stop,
    };
}

export function mount(ctx) {
    if (ctx.api !== 1) throw new Error('Device voice needs host API 1.');
    const provider = createDeviceVoice();
    const unregister = ctx.providers.register('tts', provider);
    let disposed = false;
    function dispose() {
        if (disposed) return;
        disposed = true;
        ctx.signal.removeEventListener('abort', dispose);
        provider.stop();
        unregister();
    }
    ctx.signal.addEventListener('abort', dispose, { once: true });
    if (ctx.signal.aborted) dispose();
    return { dispose };
}
