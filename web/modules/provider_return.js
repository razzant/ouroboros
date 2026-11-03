/** Passive provider navigation, followed by the owning skill's same-origin exchange.
 * The skill owns pending state, PKCE and credentials. This page owns no tokens or grants.
 */
import { apiFetch } from './api_client.js';
import { bootLanguage, tr } from './i18n.js';

const RETURN_PATH = '/onboarding/static/provider-return.html';

export function completionRequest(href) {
    const url = new URL(href);
    if (!['http:', 'https:'].includes(url.protocol) || !url.pathname.endsWith(RETURN_PATH)) {
        throw new Error('Open the sign-in return link supplied by Ouroboros.');
    }
    const state = url.searchParams.get('state');
    const code = url.searchParams.get('code');
    const error = url.searchParams.get('error');
    if (!state || Boolean(code) === Boolean(error)) throw new Error('The provider did not return a complete sign-in result.');
    let envelope;
    try {
        const bytes = Uint8Array.from(atob(state.replace(/-/g, '+').replace(/_/g, '/')), (value) => value.charCodeAt(0));
        envelope = JSON.parse(new TextDecoder('utf-8', { fatal: true }).decode(bytes));
    } catch {
        throw new Error('The sign-in return address is not valid. Start again from Ouroboros.');
    }
    const skill = envelope?.skill;
    // This is route syntax, not permission. Only the skill can verify its issued state.
    if (envelope?.v !== 1 || typeof envelope.nonce !== 'string' || !envelope.nonce
        || typeof skill !== 'string' || !/^[\p{L}\p{N}_.-]{1,64}$/u.test(skill)
        || skill !== skill.replace(/^[._]+|[._]+$/g, '')) {
        throw new Error('The sign-in return recipient is not valid.');
    }
    const prefix = url.pathname.slice(0, -RETURN_PATH.length);
    const body = error ? { state, error, error_description: url.searchParams.get('error_description') || '' } : { state, code };
    return { url: `${url.origin}${prefix}/api/extensions/${encodeURIComponent(skill)}/oauth/complete`,
        home: `${url.origin}${prefix}/`, cleanUrl: `${url.origin}${url.pathname}`,
        init: { method: 'POST', credentials: 'same-origin', redirect: 'error', cache: 'no-store',
            headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) } };
}

/** Injectable page startup so tests exercise the actual completion, not only URL construction. */
export async function mountProviderReturn(root, { href = globalThis.location.href, send = apiFetch,
    history = globalThis.history, lifecycle = globalThis.window, translate = tr } = {}) {
    const status = root.querySelector('[data-return-status]');
    const detail = root.querySelector('[data-return-detail]');
    const account = root.querySelector('[data-return-account]');
    const home = root.querySelector('[data-return-home]');
    const abort = new AbortController();
    const dispose = () => abort.abort();
    lifecycle?.addEventListener('pagehide', dispose, { once: true });
    try {
        const request = completionRequest(href);
        home.href = request.home;
        // Keep authorization codes out of this tab's later history/referrers. No automatic replay.
        try { history?.replaceState(null, '', request.cleanUrl); } catch { /* Browser history is not connection authority. */ }
        const response = await send(request.url, { ...request.init, signal: abort.signal });
        let result;
        try { result = await response.json(); } catch { /* A gateway/login page is not an exchange receipt. */ }
        if (!response.ok || result?.ok !== true) {
            const failure = new Error(translate('provider_return.unconfirmed', 'The connection was not confirmed.'));
            failure.connectorMessage = typeof result?.error === 'string' ? result.error : '';
            throw failure;
        }
        status.dataset.tone = 'ok';
        status.textContent = translate('provider_return.connected', 'Account connected');
        const label = result.email || result.label || result.account?.label;
        if (typeof label === 'string' && label) { account.textContent = label; account.hidden = false; }
        detail.textContent = translate('provider_return.done', 'Return to Ouroboros to use this account. You can close this tab.');
    } catch (error) {
        if (abort.signal.aborted) return;
        status.dataset.tone = 'warn';
        status.textContent = translate('provider_return.failed', 'Could not finish connecting');
        detail.textContent = error.connectorMessage || translate('provider_return.check',
            'Return to Ouroboros and check the account before starting sign-in again.');
        if (error.connectorMessage) detail.setAttribute('data-i18n-authored', '');
    } finally {
        lifecycle?.removeEventListener('pagehide', dispose);
    }
}

const page = globalThis.document?.querySelector('[data-provider-return]');
if (page) {
    void bootLanguage().catch(() => {});
    void mountProviderReturn(page);
}
