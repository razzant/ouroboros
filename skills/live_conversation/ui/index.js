/** Loaded, revision-bound Talk A feature; all device capture starts at the owner's press. */
import { mountCall } from './call.js';
import { fetchJson } from '/static/modules/api_client.js';
import { mountCallSettings } from './call_settings.js';
export function mount(ctx) {
    const rooms = new Set();
    let config = { enabled: false }, disposed = false, reading = 0;
    let settings = [];
    const mountSettings = () => {
        settings.forEach(handle => handle.dispose());
        settings = [...(document.querySelectorAll?.('[data-extension-settings-form]') || [])]
            .filter(form => form.dataset.skill === ctx.skill && form.dataset.route === 'config')
            .map(form => mountCallSettings(form, { skill: ctx.skill }));
    };
    window.addEventListener('ouro:extension-settings-mounted', mountSettings);
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
        settings.forEach(handle => handle.dispose()); settings = [];
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
