/** The browser owns recognition and its network path. No model or server route. */
export function recognitionDestination(ua = '') {
    if (/Edg(?:e|A|iOS)?\//.test(ua)) return 'vendor:Microsoft';
    if (/(?:Chrome|Chromium|CriOS)\//.test(ua)) return 'vendor:Google';
    if (/Safari\//.test(ua)) return 'vendor:Apple (on device when available)';
    return 'vendor:Browser speech service';
}

export function createRecognition(env = globalThis) {
    const Recognition = env.SpeechRecognition || env.webkitSpeechRecognition;
    let active;
    return {
        label: 'Browser speech',
        destination: recognitionDestination(env.navigator?.userAgent),
        available: () => Recognition ? { ok: true } : { ok: false, code: 'no_speech_api' },
        start({ lang, signal } = {}) {
            active?.cancel();
            if (!Recognition) throw Object.assign(new Error('Browser speech is unavailable here.'), { code: 'no_speech_api' });
            if (signal?.aborted) throw new DOMException('Recognition cancelled', 'AbortError');
            const recognition = new Recognition();
            recognition.continuous = true;
            recognition.interimResults = false;
            if (lang && lang !== 'auto') recognition.lang = lang;
            let settled = false, stopping = false, resolve, reject, deadline;
            const words = new Map();
            const result = new Promise((yes, no) => { resolve = yes; reject = no; });
            result.catch(() => {});
            function finish(error) {
                if (settled) return;
                settled = true;
                env.clearTimeout(deadline);
                recognition.onresult = recognition.onerror = recognition.onend = null;
                signal?.removeEventListener('abort', cancel);
                if (active === handle) active = null;
                if (error) reject(error);
                else {
                    const text = [...words.entries()].sort((a, b) => a[0] - b[0])
                        .map(([, word]) => word).join(' ').trim();
                    resolve(text ? { text } : { no_speech: true });
                }
            }
            function cancel() {
                if (settled) return;
                finish(new DOMException('Recognition cancelled', 'AbortError'));
                try { recognition.abort(); } catch { /* already ended */ }
            }
            const handle = {
                stop() {
                    if (!settled && !stopping) {
                        stopping = true;
                        try { recognition.stop(); }
                        catch (error) { finish(error); }
                        if (!settled) deadline = env.setTimeout(() => {
                            finish(new Error('Browser speech did not finish. Try again.'));
                            try { recognition.abort(); } catch { /* already ended */ }
                        }, 10000);
                    }
                    return result;
                }, cancel,
            };
            recognition.onresult = event => {
                for (let i = event.resultIndex; i < event.results.length; i++)
                    if (event.results[i].isFinal) words.set(i, event.results[i][0].transcript);
            };
            recognition.onerror = event => {
                if (event.error === 'no-speech') finish();
                else {
                    const reasons = {
                        'not-allowed': 'Microphone or speech recognition access was denied. Allow it in your browser or system settings.',
                        'service-not-allowed': 'The browser speech service is unavailable here.',
                        'audio-capture': 'No microphone is available. Connect one and try again.',
                        network: 'The browser could not reach its speech service.',
                        'language-not-supported': 'The browser speech service does not support this language.',
                    };
                    finish(Object.assign(new Error(reasons[event.error] || `Browser speech stopped: ${event.error || 'unknown error'}.`),
                        { code: event.error }));
                }
                try { recognition.abort(); } catch { /* terminal error */ }
            };
            recognition.onend = () => finish();
            signal?.addEventListener('abort', cancel, { once: true });
            active = handle;
            try { recognition.start(); } // synchronous: still inside the owner's gesture
            catch (error) { finish(error); throw error; }
            return handle;
        },
        dispose() { active?.cancel(); },
    };
}

export function mount(ctx) {
    if (ctx.api !== 1) throw new Error('Browser speech needs host API 1.');
    const provider = createRecognition();
    const unregister = ctx.providers.register('stt', provider);
    let disposed = false;
    function dispose() {
        if (disposed) return;
        disposed = true;
        ctx.signal.removeEventListener('abort', dispose);
        provider.dispose();
        unregister();
    }
    ctx.signal.addEventListener('abort', dispose, { once: true });
    if (ctx.signal.aborted) dispose();
    return { dispose };
}
