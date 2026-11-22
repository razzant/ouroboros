/** Settings › Models › Local model (docs/DESIGN.md "Settings"): what Start launches, as rows that save themselves, and
 *  the local model's own Start, Stop, Test and Install. A saved change applies when the local model is stopped and
 *  started again (its row says so); nothing here asks for a restart of Ouroboros. The status is read while this page
 *  is on screen. The runtime install stays the ordinary install's one intentional click; a managed deployment's
 *  administrator owns the program's interpreter (the install endpoint refuses), so there the page says who
 *  installs the runtime and never offers the control — an unread policy fact is no bypass. */
import { apiFetch, fetchJson } from './api_client.js';
import { bindSettingRows, setButtonBusy, settingsRow } from './settings_controls.js';
import { registerSettingsPage } from './settings_pages.js';
import { checkerUses, deploymentOf, LOCAL_RUNTIME_ADMIN } from './settings_policy.js';
import { rowsOf } from './settings_rows.js';
import { setResult } from './ui_primitives.js';

const PAGE = 'models/local';
const POLL_MS = 3000;
// The fields a deployment's checker binds when it answers from this model (`deployment_policy.local_launch_refusal`).
const FIXED_KEYS = ['LOCAL_MODEL_SOURCE', 'LOCAL_MODEL_FILENAME', 'LOCAL_MODEL_PORT', 'LOCAL_MODEL_CHAT_FORMAT'];

const settingControl = (root, key) => root.querySelector(`[data-setting-row="${key}"]`)?.querySelector('[data-setting-control]');

/** A deployment checker answering locally binds this server's identity: those fields read only, its note shown
 *  (the Start endpoint refuses any other launch). */
export function applyLocalModelPolicy(root, fixed) {
    for (const key of FIXED_KEYS) {
        const input = settingControl(root, key);
        if (input) input.readOnly = fixed;
    }
    const note = root.querySelector('[data-local-model-fixed]');
    if (note) note.hidden = !fixed;
}

/** The local model's state in words: `{ text, tone }` from one status answer. */
export function localModelStatusWords(d = {}) {
    const status = String(d.status || 'offline');
    if (d.runtime_status === 'installing') return { text: 'Installing the local runtime…', tone: 'muted' };
    let text = status.charAt(0).toUpperCase() + status.slice(1);
    if (status === 'ready' && d.context_length) text += ` · context ${d.context_length}`;
    if (status === 'downloading' && d.download_progress != null) text += ` · ${Math.round(d.download_progress * 100)}%`;
    if (d.runtime_status === 'install_ok') text += ' · runtime installed';
    if (d.runtime_status === 'install_error') text += ' · the runtime install failed';
    if (d.error) text += ` · ${d.error}`;
    // A qualified exit during startup is not proof of an old runtime: both sentences stay hedged facts. After a
    // runtime update the next Start is the evidence, so nothing offers the update again until that start answers.
    if (status === 'error' && d.error_stage === 'startup_exit') {
        if (d.runtime_status === 'ok') text += ' A newer runtime may support this model.';
        else if (d.runtime_status === 'install_ok') text += ' Start tries the updated runtime.';
    }
    const failed = d.error || status === 'error' || d.runtime_status === 'install_error';
    return { text, tone: failed ? 'danger' : status === 'ready' ? 'ok' : 'muted' };
}

function mountLocalModel(container, { path, ctx }) {
    const rows = (fold) => rowsOf(PAGE, { fold }).map((row) => settingsRow(row)).join('');
    container.innerHTML = `<p class="settings-page-lead">A model file that runs on this computer. Fill this in only if you want one.</p>
        <section class="ui-card settings-card">
            <div class="settings-rows">${rows(false)}</div>
            <p class="settings-card-note" data-local-model-fixed hidden>Your organization’s Safety Supervisor answers from this model: its model, file, port and chat format are set by your organization. Start and Stop stay yours.</p>
        </section>
        <section class="ui-card settings-card">
            <h3 class="settings-card-title">Run it</h3>
            <p class="settings-card-note" data-local-model-managed hidden>${LOCAL_RUNTIME_ADMIN} You can start and stop the model.</p>
            <p class="ui-status local-model-state" id="local-model-status" role="status" aria-live="polite" data-tone="muted">Offline</p>
            <p class="settings-card-lead" data-local-model-pending hidden></p>
            <div id="local-model-progress-wrap" class="local-model-progress-wrap" role="progressbar" aria-label="Download" aria-valuemin="0" aria-valuemax="100" aria-valuenow="0" hidden>
                <div id="local-model-progress-bar" class="local-model-progress-bar"></div></div>
            <div class="ui-action-row">
                <button type="button" class="btn" data-primary id="btn-local-start">Start</button>
                <button type="button" class="btn btn-default" id="btn-local-stop" disabled>Stop</button>
                <button type="button" class="btn btn-default" id="btn-local-test" disabled>Test tool calling</button>
                <button type="button" class="btn btn-default" id="btn-local-install-runtime" hidden>Install the local runtime</button>
                <span class="ui-result" id="local-model-action-status" role="status" aria-live="polite" aria-atomic="true"></span>
            </div>
            <pre class="ui-result-details" id="local-model-test-result" hidden></pre>
        </section>
        <section class="ui-card settings-card"><details class="ui-disclosure" data-settings-fold>
            <summary>More: GPU layers, port, chat format</summary>
            <div class="settings-rows">${rows(true)}</div></details></section>`;
    const el = (id) => container.querySelector(`#${id}`);
    const bound = bindSettingRows(container, { ask: ctx.ask });
    const [start, stop, test, install] = ['btn-local-start', 'btn-local-stop', 'btn-local-test', 'btn-local-install-runtime'].map(el);
    const action = (text, tone = 'muted') => setResult(el('local-model-action-status'), text ? { text, tone } : {});
    const testResult = (text) => Object.assign(el('local-model-test-result'), { textContent: text, hidden: !text });
    let alive = true;
    let stopPending = false;
    let lastStatus = {};
    let pendingPost = null;
    let startAfterInstall = null;
    let sequence = 0;
    // A successful policy projection omits deployment on ordinary installations. A partial settings read
    // can omit the whole projection, leaving the install authority unknown.
    let managedDeployment = null;
    let policyRead = 0;
    let missingRuntime = null;
    const managedNote = container.querySelector('[data-local-model-managed]');
    const onScreen = () => alive && ctx.state?.activePage === 'settings' && ctx.state?.settingsPath === path && !globalThis.document?.hidden;
    // What Start launches: the fields as they stand now (a field commits when it is left, so these are the saved ones).
    const typed = (key) => String(settingControl(container, key)?.value ?? '').trim();
    const launchBody = () => ({
        source: typed('LOCAL_MODEL_SOURCE'), filename: typed('LOCAL_MODEL_FILENAME'), port: parseInt(typed('LOCAL_MODEL_PORT'), 10) || 8766,
        n_gpu_layers: parseInt(typed('LOCAL_MODEL_N_GPU_LAYERS'), 10), n_ctx: parseInt(typed('LOCAL_MODEL_CONTEXT_LENGTH'), 10) || 65536,
        chat_format: typed('LOCAL_MODEL_CHAT_FORMAT'),
    });
    // An Update can finish on the server before its POST reply reaches this panel.
    const canCancelAfterInstall = (state) => state.runtime_status === 'install_ok' && Boolean(pendingPost?.afterInstall);

    function paintButtons() {
        const { status, error_stage, runtime_status } = lastStatus;
        const installing = runtime_status === 'installing';
        const modelBusy = ['downloading', 'loading', 'ready', 'stopping'].includes(status);
        const canStop = installing || modelBusy || canCancelAfterInstall(lastStatus)
            || (!pendingPost && status === 'error' && !error_stage);
        start.disabled = Boolean(pendingPost) || stopPending || installing || modelBusy;
        install.disabled = start.disabled;
        stop.disabled = stopPending || !canStop;
        test.disabled = stopPending || status !== 'ready';
    }

    function beginPost(afterInstall = null) {
        sequence += 1;
        pendingPost = { afterInstall };
        startAfterInstall = null;
        paintButtons();
        return pendingPost;
    }

    async function readStatus() {
        if (!onScreen()) return;
        const mine = ++sequence;
        let d;
        try { d = await fetchJson('/api/local-model/status', { cache: 'no-store' }); } catch { return; }
        if (!alive || mine !== sequence) return;
        if (canCancelAfterInstall(d) && !canCancelAfterInstall(lastStatus)) action('Press Stop to cancel the pending model start.');
        lastStatus = d;
        paintButtons();
        const updateRuntime = d.status === 'error' && d.error_stage === 'startup_exit' && d.runtime_status === 'ok';
        const downloading = d.status === 'downloading';
        if (missingRuntime && d.runtime_status !== 'missing') {
            missingRuntime = null;
            action('');
            testResult('');
        }
        const words = localModelStatusWords(d);
        Object.assign(el('local-model-status'), { textContent: words.text }).dataset.tone = words.tone;
        // Saved settings the running model does not have yet, and facts it did not report: the server's own sentence.
        const pending = el('local-model-status').parentElement.querySelector('[data-local-model-pending]');
        Object.assign(pending, { textContent: d.settings_application?.summary || '', hidden: !d.settings_application?.summary });
        const wrap = el('local-model-progress-wrap');
        wrap.hidden = !(downloading && d.download_progress != null);
        if (!wrap.hidden) {
            const share = Math.round(d.download_progress * 100);
            el('local-model-progress-bar').style.width = `${share}%`;
            wrap.setAttribute('aria-valuenow', String(share));
        }
        // The manual runtime install needs a settings read that positively says this deployment is not managed:
        // a managed deployment's administrator owns the interpreter, and an unread fact is no bypass.
        const runtimeWantsInstall = updateRuntime || d.runtime_status === 'missing' || d.runtime_status === 'install_error';
        install.textContent = updateRuntime ? 'Update the local runtime' : 'Install the local runtime';
        install.hidden = !(managedDeployment === false && runtimeWantsInstall);
        if (d.runtime_status === 'install_error' && d.runtime_install_log) testResult(`Install failed:\n${d.runtime_install_log}`);
        if (startAfterInstall && d.runtime_status !== 'installing') {
            const body = startAfterInstall;
            startAfterInstall = null;
            if (d.runtime_status === 'install_ok') void startModel(body);
        }
    }

    function showMissingRuntime() {
        const ordinary = managedDeployment === false;
        install.hidden = !ordinary;
        install.textContent = 'Install the local runtime';
        testResult(ordinary && missingRuntime.hint ? `By hand: ${missingRuntime.hint}` : '');
        if (ordinary) action('The local runtime is unavailable. Install it, and the model starts by itself.', 'danger');
        else if (managedDeployment) action(LOCAL_RUNTIME_ADMIN, 'danger');
        else action('The local runtime is unavailable. Reload Settings to check installation options.', 'danger');
    }

    async function startModel(body = launchBody()) {
        if (!alive || start.disabled) return;
        if (!body.source) { action('Name a model first: a Hugging Face repository or a file on this computer.', 'danger'); return; }
        const mine = beginPost();
        testResult('');
        action('');
        missingRuntime = null;
        try {
            const resp = await apiFetch('/api/local-model/start', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
            const data = await resp.json();
            if (!alive || mine !== pendingPost) return;
            if (resp.status === 412 && data.error === 'runtime_missing') {
                sequence += 1; // An older status poll cannot overrule this Start.
                lastStatus = { status: 'error', runtime_status: 'missing' };
                missingRuntime = data;
                showMissingRuntime();
            } else if (data.error) action(`Not started: ${data.error}`, 'danger');
            else {
                lastStatus = { status: 'loading' };
                void readStatus();
            }
        } catch (e) {
            if (alive && mine === pendingPost) action(`Not started: ${e.message}`, 'danger');
        } finally {
            if (alive && mine === pendingPost) { pendingPost = null; paintButtons(); }
        }
    }

    start.addEventListener('click', () => void startModel());
    stop.addEventListener('click', async () => {
        if (!alive || stop.disabled) return;
        sequence += 1;
        const cancelledStart = canCancelAfterInstall(lastStatus);
        startAfterInstall = null;
        pendingPost = null;
        const installing = lastStatus.runtime_status === 'installing';
        missingRuntime = null;
        testResult('');
        stopPending = true;
        setButtonBusy(stop, true);
        paintButtons();
        action('Stopping…');
        try {
            const result = await fetchJson('/api/local-model/stop', { method: 'POST' }, { rejectOkFalse: true });
            if (!alive) return;
            if (result?.status !== 'stopped' || result?.error) throw new Error(result?.error || 'The server did not confirm Stop.');
            const stopped = cancelledStart ? 'Automatic start cancelled.' : 'The local model stopped.';
            action(installing ? 'Runtime installation stop requested.' : stopped, 'ok');
            lastStatus = { ...lastStatus, status: 'offline' };
            void readStatus();
        } catch (e) {
            if (alive) {
                if (e.body?.code === 'local_model_stop_unconfirmed') lastStatus = { ...lastStatus, status: 'stopping' };
                const hint = cancelledStart ? 'Automatic start cancelled.' : 'You can press Stop again.';
                action(`Stop failed: ${e.message}. ${hint}`, 'danger');
                void readStatus();
            }
        }
        stopPending = false;
        if (alive) { setButtonBusy(stop, false); paintButtons(); }
    });
    test.addEventListener('click', async () => {
        missingRuntime = null;
        action('Testing…');
        testResult('');
        try {
            const r = await (await apiFetch('/api/local-model/test', { method: 'POST' })).json();
            if (r.error) { action(`Test failed: ${r.error}`, 'danger'); return; }
            action(r.success ? 'Chat and tool calling work.' : 'The test did not pass.', r.success ? 'ok' : 'warn');
            testResult([`${r.chat_ok ? 'Passed' : 'Failed'}: basic chat${r.tokens_per_sec ? ` (${r.tokens_per_sec} tokens a second)` : ''}`,
                `${r.tool_call_ok ? 'Passed' : 'Failed'}: tool calling`, r.details && !r.success ? r.details : ''].filter(Boolean).join('\n'));
        } catch (e) { action(`Test failed: ${e.message}`, 'danger'); }
    });
    install.addEventListener('click', async () => {
        if (!alive || managedDeployment !== false || install.hidden || install.disabled) return;
        const body = launchBody();
        const mine = beginPost(body.source ? body : null);
        missingRuntime = null;
        testResult('');
        action('Installing the local runtime. This may take a few minutes…');
        try {
            const d = await fetchJson('/api/local-model/install-runtime', { method: 'POST' });
            if (!alive || mine !== pendingPost) return;
            if (!['installing', 'already_installing'].includes(d.status)) throw new Error(d.error || 'The runtime install was not confirmed.');
            // The queued Start carries this reply's install_id: the server refuses it once a Stop or a newer install ended that install.
            if (d.status === 'installing' && d.install_id && mine.afterInstall) startAfterInstall = { ...mine.afterInstall, install_id: d.install_id };
            lastStatus = { ...lastStatus, runtime_status: 'installing' };
            action('');
            void readStatus();
        } catch (e) {
            if (alive && mine === pendingPost) action(`Install failed: ${e.message}`, 'danger');
        } finally {
            if (alive && mine === pendingPost) { pendingPost = null; paintButtons(); }
        }
    });

    const read = async () => {
        const mine = ++policyRead;
        let settings;
        try {
            settings = await ctx.readSettings();
        } catch { /* policy is unknown on a failed read or a partial 200 */ }
        if (!alive || mine !== policyRead) return;
        const policy = settings?._meta?.policy_state;
        const deployment = deploymentOf(settings?._meta);
        managedDeployment = policy && typeof policy === 'object' && !Array.isArray(policy) ? Boolean(deployment) : null;
        if (managedDeployment !== null) applyLocalModelPolicy(container, checkerUses(deployment, 'local'));
        managedNote.hidden = managedDeployment !== true;
        if (managedDeployment !== false) install.hidden = true;
        if (missingRuntime) showMissingRuntime();
        bound.paint();
        void readStatus();
    };
    const onShown = (event) => { if (event.detail?.path === path) void read(); };
    globalThis.window?.addEventListener('ouro:settings-page-shown', onShown);
    void read();
    const interval = setInterval(readStatus, POLL_MS);
    return { unmount() {
        alive = false;
        clearInterval(interval);
        globalThis.window?.removeEventListener('ouro:settings-page-shown', onShown);
        bound.dispose();
    } };
}

registerSettingsPage(PAGE, { title: 'Local model', mount: mountLocalModel });
