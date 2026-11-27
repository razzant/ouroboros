/** Passive provider navigation, followed by the owner's same-origin exchange: a skill's own, or the core's for
 * a remote MCP server or a skill's declared service. The owner holds pending state, PKCE and credentials. This
 * page owns no tokens or grants.
 */
import { apiFetch } from './api_client.js';
import { bootLanguage, tr, tx } from './i18n.js';

const RETURN_PATH = '/onboarding/static/provider-return.html';
const RETURN_FIELDS = ['state', 'code', 'error', 'error_description', 'iss'];

export function completionRequest(href) {
    const url = new URL(href);
    if (!['http:', 'https:'].includes(url.protocol) || !url.pathname.endsWith(RETURN_PATH)) {
        throw new Error('Open the sign-in return link supplied by Ouroboros.');
    }
    if (RETURN_FIELDS.some((key) => url.searchParams.getAll(key).length > 1)) {
        throw new Error('The provider returned an ambiguous sign-in result.');
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
    // A skill's own sign-in (`skill`), or the core's: a remote MCP server's (`mcp`) or a skill's declared service
    // (`sign_in` with `svc`). This is route syntax, not permission: only the recipient can verify its issued state.
    const kinds = envelope && typeof envelope === 'object' && !Array.isArray(envelope)
        ? ['mcp', 'sign_in', 'skill'].filter((key) => Object.hasOwn(envelope, key)) : [];
    const [kind] = kinds;
    const recipient = envelope?.[kind];
    const named = typeof recipient === 'string' && /^[\p{L}\p{N}_.-]{1,64}$/u.test(recipient)
        && recipient === recipient.replace(/^[._]+|[._]+$/g, '')
        && (kind !== 'sign_in' || (typeof envelope.svc === 'string' && /^[a-z0-9_]{1,40}$/.test(envelope.svc)));
    if (kinds.length !== 1 || !named || envelope.v !== 1 || typeof envelope.nonce !== 'string' || !envelope.nonce) {
        throw new Error('The sign-in return recipient is not valid.');
    }
    const prefix = url.pathname.slice(0, -RETURN_PATH.length);
    const body = error ? { state, error, error_description: url.searchParams.get('error_description') || '' } : { state, code };
    // RFC 9207: the issuer the provider names, for the recipient to compare with the one it asked.
    if (url.searchParams.has('iss')) body.iss = url.searchParams.get('iss');
    if (kind !== 'skill') body.action = 'complete';
    const target = { mcp: '/api/mcp/sign-in', sign_in: '/api/sign-in' }[kind]
        ?? `/api/extensions/${encodeURIComponent(recipient)}/oauth/complete`;
    return { kind, url: `${url.origin}${prefix}${target}`,
        home: `${url.origin}${prefix}/`, cleanUrl: `${url.origin}${url.pathname}`,
        init: { method: 'POST', credentials: 'same-origin', redirect: 'error', cache: 'no-store',
            headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) } };
}

/** Injectable page startup so tests exercise the actual completion, not only URL construction. */
export async function mountProviderReturn(root, { href = globalThis.location.href, send = apiFetch,
    history = globalThis.history, lifecycle = globalThis.window, translate = tr, translateHost = tx } = {}) {
    const status = root.querySelector('[data-return-status]');
    const detail = root.querySelector('[data-return-detail]');
    const account = root.querySelector('[data-return-account]');
    const home = root.querySelector('[data-return-home]');
    const next = root.querySelector('[data-return-next]');
    const again = root.querySelector('[data-return-again]');
    const technical = root.querySelector('[data-return-technical]');  // a no-send's own detail: a path, a host, OpenSSL's words
    const said = root.querySelector('[data-return-said]');
    const abort = new AbortController();
    const dispose = () => abort.abort();
    const show = (node, on) => { if (node) node.hidden = !on; };
    // This return stays in the page's memory only (its address is cleaned before it is sent). It goes out on load and, once the host
    // says it kept the provider's answer (`answer_saved`), again only on the owner's press: the host then reads that saved answer and
    // does not exchange the one-time code a second time. A remote MCP server's `not_sent` (stopped on this computer before
    // anything changed at the provider) is sent again the same way, only on the owner's press; meanwhile the repair opens in
    // another tab of this Ouroboros, so this one keeps the return.
    let request = null;
    async function finish() {
        lifecycle?.addEventListener('pagehide', dispose, { once: true });
        show(again, false);
        show(next, false);
        show(technical, false);
        detail.removeAttribute?.('data-i18n-authored');
        if (request) {  // a press after a stop: the page's waiting words (as it loaded) and the way home in this tab
            status.dataset.tone = 'muted';
            status.textContent = translate('provider_return.completing', 'Completing sign-in…');
            detail.textContent = translate('provider_return.waiting', 'Waiting for the connector to confirm this account.');
            home.href = request.home;
            home.removeAttribute?.('target');
            home.textContent = translate('provider_return.home', 'Return to Ouroboros');
        }
        let saved = false;
        let unsent = false;
        try {
            if (!request) {
                request = completionRequest(href);
                home.href = request.home;
                // Keep authorization codes out of this tab's later history/referrers. No automatic replay.
                try { history?.replaceState(null, '', request.cleanUrl); } catch { /* Browser history is not connection authority. */ }
            }
            const response = await send(request.url, { ...request.init, signal: abort.signal });
            let result;
            try { result = await response.json(); } catch { /* A gateway/login page is not an exchange receipt. */ }
            const serviceReturn = request.kind === 'sign_in';
            const label = serviceReturn ? result?.account?.label : result?.email || result?.label || result?.account?.label || result?.server;
            const named = typeof label === 'string' && Boolean(label.trim());
            if (!response.ok || result?.ok !== true || (serviceReturn && (!named || !result.attempt_id
                    || !['signed_in', 'sign_in_required'].includes(result.state)))) {
                const failure = new Error(translate('provider_return.unconfirmed', 'The connection was not confirmed.'));
                failure.connectorMessage = typeof result?.error === 'string' ? result.error : '';
                saved = serviceReturn && result?.reason === 'answer_saved' && Boolean(result.attempt_id);  // the host's typed receipt of this attempt
                unsent = request.kind === 'mcp' && result?.reason === 'not_sent' && Boolean(failure.connectorMessage);
                failure.technical = unsent && typeof result.detail === 'string' ? result.detail : '';
                throw failure;
            }
            request = null;
            const limited = serviceReturn && (result.state !== 'signed_in' || result.reason === 'insufficient_scope' || Boolean(result.missing_required?.length));
            status.dataset.tone = limited ? 'warn' : 'ok';
            status.textContent = limited ? translate('provider_return.limited', 'Account confirmed · access needs attention')
                : translate('provider_return.connected', 'Account connected');
            if (named) { account.textContent = label; account.hidden = false; }
            detail.textContent = limited ? translate('provider_return.review_access', 'Return to Ouroboros to review this skill’s access.')
                : translate('provider_return.done', 'Return to Ouroboros to use this account. You can close this tab.');
        } catch (error) {
            if (abort.signal.aborted) return;
            status.dataset.tone = 'warn';
            if (saved) {
                status.textContent = translate('provider_return.pending', 'Account not confirmed yet');
                detail.textContent = translate('provider_return.saved',
                    'Ouroboros has the provider’s answer but couldn’t read which account it names yet. Finish sign-in reads it again; the one-time code isn’t sent to the provider a second time.');
                show(again, true);
                return;
            }
            if (unsent) {
                status.textContent = translate('provider_return.unsent', 'Sign-in not finished yet');
                detail.textContent = translateHost(error.connectorMessage);  // the host's fixed words and next step
                if (said && error.technical) { said.textContent = error.technical; show(technical, true); }
                if (next) {
                    next.textContent = translate('provider_return.unsent_next',
                        'Keep this tab open. Once it is fixed, press Finish sign-in here.');
                    show(next, true);
                }
                home.href = `${request.home}#settings`;
                home.setAttribute('target', '_blank');  // the page's opener is cut (rel below): this tab keeps the code
                home.setAttribute('rel', 'noopener');
                home.textContent = translate('provider_return.open_settings', 'Open Settings in a new tab');
                show(again, true);
                return;
            }
            request = null;
            const check = translate('provider_return.check', 'Return to Ouroboros and check the account before starting sign-in again.');
            status.textContent = translate('provider_return.failed', 'Could not finish connecting');
            detail.textContent = error.connectorMessage || check;
            if (error.connectorMessage) {
                detail.setAttribute('data-i18n-authored', '');
                if (next) { next.textContent = check; show(next, true); }  // the host's message never replaces the owner's next step
            }
        } finally {
            lifecycle?.removeEventListener('pagehide', dispose);
        }
    }
    again?.addEventListener('click', () => (request ? finish() : undefined));
    await finish();
}

const page = globalThis.document?.querySelector('[data-provider-return]');
if (page) {
    void bootLanguage().catch(() => {});
    void mountProviderReturn(page);
}
