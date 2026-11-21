/** Loaded, revision-bound Talk A feature; all device capture starts at the owner's press. */
import { mountCall } from './call.js';
import { fetchJson } from '/static/modules/api_client.js';
import { mountCallSettings } from './call_settings.js';
export function mount(ctx) {
    const rooms = new Set();
    let config = { enabled: false }, disposed = false, reading = 0;
    // Each host owns its bindings even while a navigation return holds it off DOM. Another skill's render must
    // neither cancel this form's pending autosave nor remove its device controls. Actual release ends its bindings.
    const settings = new Map();
    const releaseSettings = root => {
        settings.get(root)?.forEach(handle => handle.dispose());
        settings.delete(root);
    };
    const mountSettings = event => {
        const root = event?.detail?.root;
        const forms = [...((root || document).querySelectorAll?.('[data-extension-settings-form]') || [])]
            .filter(form => form.dataset.skill === ctx.skill && form.dataset.route === 'config');
        const roots = root ? [root] : [...new Set(forms.map(form => form.closest('#extension-settings-sections') || form))];
        for (const host of roots) {
            releaseSettings(host);
            const handles = forms.filter(form => host === form || host.contains(form))
                .map(form => mountCallSettings(form, { skill: ctx.skill }));
            if (handles.length) settings.set(host, handles);
        }
    };
    const unmountSettings = event => releaseSettings(event.detail?.root);
    window.addEventListener('ouro:extension-settings-mounted', mountSettings);
    window.addEventListener('ouro:extension-settings-unmounted', unmountSettings);
    mountSettings();
    async function refresh() {
        const request = ++reading;
        let next;
        try { next = await fetchJson(`/api/extensions/${encodeURIComponent(ctx.skill)}/config`, { cache: 'no-store' }); }
        catch { return; }  // A failed read is not an owner's decision to end the call.
        if (disposed || request !== reading) return;
        config = next;
        for (const room of rooms) room.state(config);
    }
    const unregister = ctx.providers.register('conversation', {
        mountRoom(options) {
            const room = mountCall(options);
            rooms.add(room); room.state(config);
            const destroy = room.destroy;
            room.destroy = () => { rooms.delete(room); destroy(); };
            return room;
        },
    });
    // Settings writes do not require a page reload. This read cannot enable a
    // skill or reveal a key; it returns only the feature's explicit configuration.
    void refresh();
    const returned = () => { if (!document.hidden) void refresh(); };
    window.addEventListener('focus', refresh);
    window.addEventListener('ouro:settings-updated', refresh);
    document.addEventListener('visibilitychange', returned);
    function dispose() {
        if (disposed) return;
        disposed = true;
        window.removeEventListener('ouro:extension-settings-mounted', mountSettings);
        window.removeEventListener('ouro:extension-settings-unmounted', unmountSettings);
        for (const root of [...settings.keys()]) releaseSettings(root);
        window.removeEventListener('focus', refresh);
        window.removeEventListener('ouro:settings-updated', refresh);
        document.removeEventListener('visibilitychange', returned);
        unregister();
        for (const room of [...rooms]) room.destroy();
    }
    ctx.signal.addEventListener('abort', dispose, { once: true });
    if (ctx.signal.aborted) dispose();
    return { dispose };
}
