/* The parent controls lease lifetime; the frame can read its URL but cannot use these controls. */
import { apiFetch } from './api_client.js';
import { WIDGET_REQUEST_TIMEOUT_MS } from './widget_job.js';

const controls = '/api/widgets/asset-leases';

// `scope` 'files' (an ESM mount's declared files) or 'routes' (GET/HEAD of a classic or route-iframe
// mount's own skill routes, for the frame's own elements); one kind never reads the other.
export async function openWidgetAssetLease(tab, signal, scope = 'files') {
    const response = await apiFetch(controls, {
        method: 'POST', headers: { 'Content-Type': 'application/json' }, signal,
        body: JSON.stringify({ skill: tab.skill, tab_id: tab.tab_id, generation: tab.generation,
            ...(scope === 'routes' ? { scope } : {}) }),
    });
    if (!response.ok) throw new Error('widget publication unavailable');
    const lease = await response.json();
    return { ...lease, baseUrl: new URL(lease.base_url, window.location.origin).href };
}

// One declared file, read with the lease alone. Its bytes were captured at
// publication (MAX_WIDGET_BYTES bounds the whole set), so a save may hold them
// as a Blob; the desktop URL bridge admits only its own API prefixes.
export async function readWidgetAsset(url) {
    const response = await apiFetch(url, { credentials: 'omit', cache: 'no-store', redirect: 'error' });
    if (!response.ok) throw new Error(`widget file unavailable: HTTP ${response.status}`);
    return response.blob();
}

export function revokeWidgetAssetLease(lease) {
    if (!lease) return;
    // Keepalive permits navigation cleanup; a crash or a lost request instead
    // leaves at most expires_in seconds since the last accepted renewal.
    return apiFetch(`${controls}/revoke`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' }, keepalive: true,
        body: JSON.stringify({ token: lease.token }),
    }).catch(() => {});
}

// The same route lease again, under the token the running frame already uses (after a core restart, or an
// expiry while this page could not renew). Resolves the lease reply; null while the publication is not live
// yet; throws, marked definitive, when the owner's session or the lease itself has ended.
export async function reopenWidgetAssetLease(tab, lease, signal) {
    const response = await apiFetch(controls, {
        method: 'POST', headers: { 'Content-Type': 'application/json' }, signal,
        body: JSON.stringify({ skill: tab.skill, tab_id: tab.tab_id, generation: tab.generation, scope: 'routes',
            token: lease.token }),
    });
    if (response.ok) return response.json();
    if (response.status === 409) return null;
    const error = new Error('widget access ended');
    error.definitive = [400, 401, 403, 410].includes(response.status);
    throw error;
}

// Renews every `renew_after` seconds. A file lease ends the mount on any failure or at its expiry. With
// `reestablish` (a route lease) a failed or hanging renewal is retried with backoff and a lease the core no
// longer knows is re-established under the same URL, so the running document keeps its state; only a
// signed-out session or an ended lease (401, 403, 400, 410) ends the mount.
export function maintainWidgetAssetLease(lease, lost, { reestablish = null } = {}) {
    let stopped = false;
    let renewal = 0;
    let expiry = 0;
    let failures = 0;
    let controller = null;
    const expire = () => { if (!stopped) lost(); };
    const arm = (reply) => {
        failures = 0;
        clearTimeout(expiry);
        if (!reestablish) expiry = setTimeout(expire, reply.expires_in * 1000);
        renewal = setTimeout(renew, reply.renew_after * 1000);
    };
    const retry = () => {
        failures += 1;
        renewal = setTimeout(renew, Math.min(lease.renew_after, 2 ** failures) * 1000);
    };
    const renew = async () => {
        if (stopped) return;
        controller = new AbortController();
        const timer = setTimeout(() => controller?.abort(), WIDGET_REQUEST_TIMEOUT_MS);
        try {
            const response = await apiFetch(`${controls}/renew`, {
                method: 'POST', headers: { 'Content-Type': 'application/json' }, signal: controller.signal,
                body: JSON.stringify({ token: lease.token }),
            });
            let reply = response.ok ? await response.json() : null;
            if (!reply && !reestablish) throw new Error('widget lease ended');
            if (!reply && response.status === 409) reply = await reestablish(controller.signal);
            else if (!reply && [400, 401, 403].includes(response.status)) { expire(); return; }
            if (stopped) return;
            if (reply) arm(reply);
            else retry();
        } catch (error) {
            if (!reestablish || error?.definitive) expire();
            else if (!stopped) retry();
        } finally { clearTimeout(timer); controller = null; }
    };
    const release = () => {
        if (stopped) return;
        stopped = true;
        clearTimeout(renewal);
        clearTimeout(expiry);
        controller?.abort();
        window.removeEventListener('pagehide', onPageHide);
        revokeWidgetAssetLease(lease);
    };
    const onPageHide = () => { release(); lost(); };
    window.addEventListener('pagehide', onPageHide);
    arm(lease);
    return release;
}
