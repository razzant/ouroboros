/** Client-local device choices and short, explicitly started capture/playback tests. */
import { acquire, captureReason } from '/static/modules/media_capture.js';
import { listDevices, loadDevicePrefs, microphoneConstraints, cameraConstraints, routeSpeaker, saveDevicePrefs, speakerChoice }
    from './call_devices.js';

const DEVICES = [['mic','audioinput','Microphone'],['speaker','audiooutput','Speaker'],['camera','videoinput','Camera']];
const PROCESSING = [['echoCancellation','Echo cancellation'],['noiseSuppression','Noise suppression'],['autoGainControl','Automatic microphone volume']];

export function devicesHtml() {
    return `<div class="settings-effort-card" data-call-devices>
        <p class="settings-section-copy">On this device: choices stay in this browser or app and apply to a running call immediately.</p>
        <div class="form-grid two">${DEVICES.map(([key,,label]) => `<label class="ui-field form-field"><span>${label}</span>
            <select class="ui-control" aria-label="${label}" data-call-device="${key}"></select>
            <small class="settings-inline-note" data-call-device-note="${key}"></small></label>`).join('')}</div>
        ${PROCESSING.map(([key,label]) => `<label class="ui-field ui-field-inline"><input class="ui-checkbox" type="checkbox"
            data-call-processing="${key}"> ${label}</label>`).join('')}
        <meter data-call-meter min="0" max="1" value="0" aria-label="Microphone level" hidden></meter>
        <video data-call-camera-preview autoplay muted playsinline hidden style="max-width:100%;max-height:180px"></video>
        <div class="settings-toolbar">
            <button type="button" class="btn btn-default btn-sm" data-call-test="mic">Test microphone</button>
            <button type="button" class="btn btn-default btn-sm" data-call-test="speaker">Play a test sound</button>
            <button type="button" class="btn btn-default btn-sm" data-call-test="camera">Test camera</button>
        </div><div class="settings-inline-note" data-call-device-status role="status"></div></div>`;
}

export function bindDeviceSettings(root, env = globalThis) {
    const $ = selector => root.querySelector(selector);
    const media = env.navigator?.mediaDevices;
    const status = $('[data-call-device-status]');
    let disposed = false, generation = 0, capture = null, context = null, frame = 0, timer = 0;
    async function paint() {
        const prefs = loadDevicePrefs(env);
        const devices = await listDevices(env).catch(() => ({audioinput:[],audiooutput:[],videoinput:[]}));
        if (disposed) return;
        for (const [key,kind,label] of DEVICES) {
            const select = $(`[data-call-device="${key}"]`);
            const rows = [{id:'',label:'System default'}, ...devices[kind].map((row,i) => ({...row,label:row.label || `${label} ${i+1}`}))];
            if (prefs[key] && !rows.some(row => row.id === prefs[key])) rows.push({id:prefs[key],label:`${label} not connected`});
            select.replaceChildren(...rows.map(row => {
                const option = env.document.createElement('option'); option.value = row.id; option.textContent = row.label; return option;
            }));
            select.value = prefs[key];
            select.disabled = key === 'speaker' && !speakerChoice(env);
            $(`[data-call-device-note="${key}"]`).textContent = select.disabled
                ? 'This browser follows the system speaker. Choose it in your system’s sound settings.' : '';
        }
        for (const [key] of PROCESSING) $(`[data-call-processing="${key}"]`).checked = prefs[key];
    }
    function stop() {
        generation++;
        env.clearTimeout(timer); env.cancelAnimationFrame?.(frame);
        const lease = capture; capture = null; lease?.release();
        context?.close().catch(() => {}); context = null;
        $('[data-call-meter]').hidden = true;
        const preview = $('[data-call-camera-preview]'); preview.srcObject = null; preview.hidden = true;
        for (const [key,,label] of DEVICES) $(`[data-call-test="${key}"]`).textContent = key === 'speaker' ? 'Play a test sound' : `Test ${label.toLowerCase()}`;
    }
    async function test(kind) {
        if (capture || context) { stop(); return; }
        const token = ++generation, prefs = loadDevicePrefs(env);
        try {
            if (kind === 'speaker') {
                const audio = new env.AudioContext(); context = audio;
                await audio.resume();
                await routeSpeaker(audio, prefs);
                if (disposed || token !== generation) return;
                const oscillator = audio.createOscillator(), gain = audio.createGain();
                oscillator.frequency.value = 660; gain.gain.value = .12;
                oscillator.connect(gain).connect(audio.destination);
                oscillator.start(); oscillator.stop(audio.currentTime + .3);
                timer = env.setTimeout(stop, 500);
                status.textContent = 'If you heard the soft note, the speaker works.';
                return;
            }
            const pending = acquire({kind:kind === 'camera' ? 'camera' : 'audio', owner:root, label:'Call device test',
                constraints:kind === 'camera' ? cameraConstraints(prefs).video : microphoneConstraints(prefs).audio});
            capture = pending.release ? pending : null;
            const lease = await pending;
            if (disposed || token !== generation) { lease.release?.(); return; }
            if (lease.code) { capture = null; status.textContent = captureReason(lease.code, lease.label); return; }
            capture = lease;
            lease.signal.addEventListener('abort', stop, {once:true});
            $(`[data-call-test="${kind}"]`).textContent = 'Stop the test';
            timer = env.setTimeout(stop, 15000);
            if (kind === 'camera') {
                const preview = $('[data-call-camera-preview]'); preview.srcObject = lease.stream; preview.hidden = false;
                await preview.play().catch(() => {});
                status.textContent = 'This camera preview stays on this device.';
            } else {
                const audio = new env.AudioContext(); context = audio; await audio.resume();
                if (disposed || token !== generation) return;
                const analyser = audio.createAnalyser(); analyser.fftSize = 512;
                audio.createMediaStreamSource(lease.stream).connect(analyser);
                const samples = new Float32Array(analyser.fftSize), meter = $('[data-call-meter]'); meter.hidden = false;
                const draw = () => {
                    if (disposed || token !== generation) return;
                    analyser.getFloatTimeDomainData(samples);
                    meter.value = Math.min(1, Math.sqrt(samples.reduce((sum, value) => sum + value * value, 0) / samples.length) * 6);
                    frame = env.requestAnimationFrame(draw);
                };
                draw(); status.textContent = 'Speak: the meter follows your microphone.';
            }
            void paint();
        } catch {
            stop(); status.textContent = 'The device test could not start. Check this browser’s device permissions.';
        }
    }
    const changed = event => {
        const {callDevice, callProcessing} = event.target.dataset;
        if (callDevice) saveDevicePrefs({[callDevice]:event.target.value}, env);
        else if (callProcessing) saveDevicePrefs({[callProcessing]:event.target.checked}, env);
        else return;
        stop();
    };
    const clicked = event => { const kind = event.target.closest?.('[data-call-test]')?.dataset.callTest; if (kind) void test(kind); };
    const left = event => { if (event.detail?.page !== 'settings') stop(); };
    const hidden = () => { if (env.document.hidden) stop(); };
    root.addEventListener('change', changed); root.addEventListener('click', clicked);
    media?.addEventListener?.('devicechange', paint);
    env.addEventListener?.('ouro:page-shown', left); env.addEventListener?.('pagehide', stop); env.addEventListener?.('ouro:panic', stop);
    env.document.addEventListener?.('visibilitychange', hidden);
    void paint();
    return () => {
        disposed = true; stop();
        root.removeEventListener('change', changed); root.removeEventListener('click', clicked);
        media?.removeEventListener?.('devicechange', paint);
        env.removeEventListener?.('ouro:page-shown', left); env.removeEventListener?.('pagehide', stop); env.removeEventListener?.('ouro:panic', stop);
        env.document.removeEventListener?.('visibilitychange', hidden);
    };
}
