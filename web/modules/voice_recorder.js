/** Bounded raw-audio recording. No DOM, provider discovery or draft mutation.
 * MediaRecorder transport adapted from josephsteuerjr's PR #1827.
 */
const PREFERRED = ['audio/webm;codecs=opus', 'audio/webm', 'audio/mp4;codecs=mp4a.40.2', 'audio/mp4'];
const baseType = type => type.split(';')[0].trim().toLowerCase();
const accepts = (formats, type) => !formats?.length || formats.some(format =>
    format.includes(';') ? format.toLowerCase() === type.toLowerCase() : baseType(format) === baseType(type));

export function recordingFormat(formats, env = globalThis) {
    for (const type of [...PREFERRED, ...(formats || [])]) {
        if (accepts(formats, type) && env.MediaRecorder?.isTypeSupported?.(type)) return type;
    }
    if (accepts(formats, 'audio/wav') && (env.AudioContext || env.webkitAudioContext)) return 'audio/wav';
    return '';
}

export function wavBlob(chunks, samples, rate) {
    const bytes = new ArrayBuffer(44 + samples * 2);
    const view = new DataView(bytes);
    function word(offset, text) { [...text].forEach((char, i) => view.setUint8(offset + i, char.charCodeAt(0))); }
    word(0, 'RIFF'); view.setUint32(4, bytes.byteLength - 8, true); word(8, 'WAVE');
    word(12, 'fmt '); view.setUint32(16, 16, true); view.setUint16(20, 1, true);
    view.setUint16(22, 1, true); view.setUint32(24, rate, true); view.setUint32(28, rate * 2, true);
    view.setUint16(32, 2, true); view.setUint16(34, 16, true); word(36, 'data');
    view.setUint32(40, samples * 2, true);
    let offset = 44;
    for (const chunk of chunks) for (const sample of chunk) {
        const value = Math.max(-1, Math.min(1, sample));
        view.setInt16(offset, Math.round(value * (value < 0 ? 32768 : 32767)), true);
        offset += 2;
    }
    return new Blob([bytes], { type: 'audio/wav' });
}

/** Call from the gesture BEFORE waiting for the microphone. WAV needs resume there. */
export function prepareRecorder(provider, { env = globalThis, onLimit = () => {} } = {}) {
    const mime = recordingFormat(provider.formats, env);
    if (!mime) throw new Error('This browser cannot record a format the engine accepts.');
    const maxBytes = Math.min(provider.max_bytes || 524288, 524288);
    const maxSeconds = Math.min(provider.max_seconds || 60, 600);
    const Context = env.AudioContext || env.webkitAudioContext;
    const context = mime === 'audio/wav' ? new Context() : null;
    let resumed;
    try { resumed = context?.resume(); }
    catch (error) { void context?.close(); throw error; }
    // A microphone prompt may remain open; never leave an unhandled resume rejection.
    resumed?.catch(() => {});
    let current;
    function closeContext() { void context?.close().catch(() => {}); }
    return {
        mime,
        cancel() { if (current) current.cancel(); else closeContext(); },
        start(stream) {
            current = context ? recordWav(stream, { context, resumed, maxBytes, maxSeconds, onLimit, env })
                : recordMedia(stream, { mime, maxBytes, maxSeconds, onLimit, env });
            return current;
        },
    };
}

function recordMedia(stream, { mime, maxBytes, maxSeconds, onLimit, env }) {
    const recorder = new env.MediaRecorder(stream, { mimeType: mime, audioBitsPerSecond: 32000 });
    let chunks = [], bytes = 0, finished = false, stopping = false, limited = false, timer;
    let resolve, reject;
    const result = new Promise((yes, no) => { resolve = yes; reject = no; });
    result.catch(() => {});
    function cleanup() {
        env.clearTimeout(timer);
        recorder.ondataavailable = recorder.onerror = recorder.onstop = null;
        chunks = [];
    }
    function finish(error) {
        if (finished) return;
        finished = true;
        const blob = new Blob(chunks, { type: recorder.mimeType || mime });
        cleanup();
        if (error) reject(error);
        else if (blob.size > maxBytes) reject(Object.assign(new Error('Recording too large. Try a shorter phrase.'), { code: 'too_large' }));
        else resolve(blob);
    }
    function stop() {
        if (!finished && !stopping) {
            stopping = true;
            try { if (recorder.state !== 'inactive') recorder.stop(); else finish(); }
            catch (error) { finish(error); }
        }
        return result;
    }
    function limit(code) {
        if (finished || limited) return;
        limited = true;
        stop();
        onLimit(code);
    }
    recorder.ondataavailable = event => {
        if (finished || !event.data?.size) return;
        bytes += event.data.size;
        chunks.push(event.data);
        if (bytes >= maxBytes) limit('max_bytes');
    };
    recorder.onstop = () => finish();
    recorder.onerror = event => {
        const error = event.error || new Error('Audio recording failed.');
        finish(error);
        try { if (recorder.state !== 'inactive') recorder.stop(); } catch { /* already stopped */ }
    };
    try { recorder.start(250); }
    catch (error) { finish(error); throw error; }
    timer = env.setTimeout(() => limit('max_seconds'), maxSeconds * 1000);
    return { result, stop, cancel() {
        if (finished) return;
        finish(new DOMException('Recording cancelled', 'AbortError'));
        try { if (recorder.state !== 'inactive') recorder.stop(); } catch { /* already stopped */ }
    } };
}

function recordWav(stream, { context, resumed, maxBytes, maxSeconds, onLimit, env }) {
    let source, node, silent, timer, samples = 0, chunks = [], finished = false;
    let resolve, reject;
    const result = new Promise((yes, no) => { resolve = yes; reject = no; });
    result.catch(() => {});
    const maximum = Math.max(0, Math.min(Math.floor((maxBytes - 44) / 2), Math.floor(context.sampleRate * maxSeconds)));
    function finish(error) {
        if (finished) return result;
        finished = true;
        env.clearTimeout(timer);
        if (node) node.onaudioprocess = null;
        source?.disconnect(); node?.disconnect(); silent?.disconnect();
        void context.close().catch(() => {});
        if (error) reject(error);
        else resolve(wavBlob(chunks, samples, context.sampleRate));
        chunks = [];
        return result;
    }
    function limit(code) { finish(); onLimit(code); }
    try {
        if (maxBytes < 44) throw new Error('The engine audio limit is too small for WAV.');
        source = context.createMediaStreamSource(stream);
        // Compatibility path for engines without a usable MediaRecorder codec.
        node = context.createScriptProcessor(4096, 1, 1);
        silent = context.createGain(); silent.gain.value = 0;
        node.onaudioprocess = event => {
            if (finished) return;
            const input = event.inputBuffer.getChannelData(0);
            const count = Math.min(input.length, maximum - samples);
            if (count) { chunks.push(input.slice(0, count)); samples += count; }
            if (samples >= maximum) limit('max_bytes');
        };
        source.connect(node); node.connect(silent); silent.connect(context.destination);
        timer = env.setTimeout(() => limit('max_seconds'), maxSeconds * 1000);
        Promise.resolve(resumed).catch(error => finish(error));
    } catch (error) { finish(error); throw error; }
    return { result, stop: () => finish(), cancel: () => finish(new DOMException('Recording cancelled', 'AbortError')) };
}
