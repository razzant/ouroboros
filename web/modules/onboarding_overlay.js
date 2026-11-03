import { apiFetch, openSignIn } from './api_client.js';
import { escapeHtmlAttr as escapeHtml } from './utils.js';
import { pageOwnedByCore } from './ws.js';

export const OVERLAY_ID = 'onboarding-overlay';

// Pause before an unanswered readiness check repeats on its own; it grows 1.5x
// per attempt up to the cap, the curve the socket's state reads use. A core that
// is down when the page loads comes back without the owner pressing anything.
export const READINESS_RETRY_MS = 2000;
const READINESS_RETRY_MAX_MS = 30000;

// The framed wizard's sandbox policy, in one place: it is a security boundary
// AND the thing that has to change when a wizard step needs a new capability,
// so it should be one edit rather than a token buried in markup.
//
// `allow-popups` + `allow-popups-to-escape-sandbox` are LOAD-BEARING, not
// hygiene: the Agents step's primary action is the agent's own sign-in link,
// opened with target="_blank". Without them the browser blocks that click
// SILENTLY — the owner presses "Open sign-in link" and nothing happens, and
// copying the URL by hand is the only way through. The escape variant is
// required too: a popup that inherits this sandbox lands on the vendor's OAuth
// page without same-origin or scripts, which no sign-in survives. Both apply
// only to windows the framed page opens, never to the frame's own authority
// over this document.
export const FRAME_SANDBOX = 'allow-same-origin allow-scripts allow-forms '
    + 'allow-popups allow-popups-to-escape-sandbox';

function removeOverlay() {
    document.getElementById(OVERLAY_ID)?.remove();
}

/**
 * Mount the blocking shell. Called BEFORE the readiness probe, because the
 * decision this app is blocked on has not been made yet: until a 204 proves the
 * install is configured, the ordinary UI must not be reachable. Mounting only
 * after a successful probe meant a transport failure — a headless first run, a
 * server still coming up, a dropped connection — left the owner in the ordinary
 * UI with no onboarding and no way back to it, which contradicts what the
 * cancelled-wizard path promises ("the blocking overlay stays available").
 *
 * The shell starts WITHOUT its backdrop: `.onboarding-overlay` is already a
 * full-viewport fixed layer, so it intercepts the ordinary UI while transparent.
 * The dark scrim arrives with the answer, so a configured install (204) never
 * flashes a dimmed screen on its way to the app.
 */
function mountOverlayShell() {
    removeOverlay();
    const overlay = document.createElement('div');
    overlay.id = OVERLAY_ID;
    overlay.className = 'onboarding-overlay';
    overlay.setAttribute('role', 'dialog');
    overlay.setAttribute('aria-modal', 'true');
    overlay.setAttribute('aria-label', 'Ouroboros setup');
    document.body.appendChild(overlay);
    return overlay;
}

function setOverlayContent(overlay, node) {
    if (!overlay.querySelector('.onboarding-overlay-backdrop')) {
        const backdrop = document.createElement('div');
        backdrop.className = 'onboarding-overlay-backdrop';
        overlay.appendChild(backdrop);
    }
    overlay.querySelector('[data-onboarding-content]')?.remove();
    node.setAttribute('data-onboarding-content', '');
    overlay.appendChild(node);
    return node;
}

function showSetupFrame(overlay) {
    const surface = document.createElement('div');
    surface.className = 'onboarding-surface';
    const frame = document.createElement('iframe');
    frame.className = 'onboarding-frame';
    frame.title = 'Ouroboros Setup';
    frame.setAttribute('sandbox', FRAME_SANDBOX);
    // ONE onboarding host: frame the real /onboarding page rather than an
    // inlined srcdoc document. A srcdoc string cannot import web/modules/*,
    // and the wizard's steps need those ordinary ES modules.
    frame.src = '/onboarding';
    surface.appendChild(frame);
    if (!pageOwnedByCore()) {
        const actions = document.createElement('div');
        actions.className = 'onboarding-recovery';
        const recovery = document.createElement('button');
        recovery.type = 'button';
        recovery.className = 'btn btn-default';
        recovery.textContent = 'Open built-in setup';
        recovery.setAttribute('data-onboarding-recovery', '');
        recovery.addEventListener('click', () => { frame.src = '/onboarding?builtin=1'; });
        actions.appendChild(recovery);
        surface.appendChild(actions);
    }
    return setOverlayContent(overlay, surface);
}

/**
 * The readiness probe failed. This is NOT "no onboarding is due" — that answer
 * is a 204 and nothing else. The overlay stays up and says what happened, with
 * the only action that can resolve it. A repeated failure updates the same card,
 * so a focused Try again keeps its focus.
 */
function showReadinessFailure(overlay, detail, onRetry) {
    const text = `Could not check whether Ouroboros is configured (${detail}). `
        + 'Setup stays blocked until that check succeeds; it is checked again automatically.';
    const shown = overlay.querySelector('[data-onboarding-failure]');
    if (shown) {
        shown.querySelector('p').textContent = text;
        return shown;
    }
    const card = document.createElement('section');
    card.className = 'onboarding-restart-card';
    card.setAttribute('data-onboarding-failure', '');
    const heading = document.createElement('h2');
    heading.textContent = 'Setup unavailable';
    const copy = document.createElement('p');
    copy.setAttribute('role', 'status');
    copy.textContent = text;
    const retry = document.createElement('button');
    retry.type = 'button';
    retry.className = 'btn btn-primary';
    retry.setAttribute('data-onboarding-retry', '');
    retry.textContent = 'Try again';
    retry.addEventListener('click', onRetry);
    card.appendChild(heading);
    card.appendChild(copy);
    card.appendChild(retry);
    return setOverlayContent(overlay, card);
}

function showRestartRequiredOverlay(runtimeMode) {
    const mode = escapeHtml(runtimeMode || 'advanced');
    const overlay = document.getElementById(OVERLAY_ID) || document.createElement('div');
    overlay.id = OVERLAY_ID;
    overlay.className = 'onboarding-overlay';
    overlay.setAttribute('role', 'dialog');
    overlay.setAttribute('aria-modal', 'true');
    overlay.setAttribute('aria-label', 'Ouroboros restart required');
    overlay.innerHTML = `
        <div class="onboarding-overlay-backdrop"></div>
        <section class="onboarding-restart-card">
            <h2>Restart Required</h2>
            <p>Runtime mode was saved as <code>${mode}</code> for the next boot. Restart Ouroboros to apply it before continuing in that mode.</p>
            <button type="button" class="btn btn-primary" data-onboarding-continue>Continue in current mode</button>
        </section>
    `;
    if (!overlay.parentElement) document.body.appendChild(overlay);
    overlay.querySelector('[data-onboarding-continue]')?.addEventListener('click', () => {
        removeOverlay();
        window.location.reload();
    });
}

export async function initOnboardingOverlay() {
    function handleMessage(event) {
        // Same-origin only: any web page can postMessage into this window;
        // without the origin check a foreign page could dismiss onboarding or
        // spoof restart prompts.
        if (event.origin !== window.location.origin) return;
        if (event?.data?.type !== 'ouroboros:onboarding-complete') return;
        if (event.data.restart_required) {
            showRestartRequiredOverlay(event.data.runtime_mode);
            return;
        }
        removeOverlay();
        window.location.reload();
    }

    window.addEventListener('message', handleMessage);

    // Blocked first, answered second. The shell exists before the request, so
    // no outcome of the request can leave the app unblocked by accident — only
    // a verified 204 takes it down.
    const overlay = mountOverlayShell();
    let retryTimer = null;
    let retryDelay = READINESS_RETRY_MS;
    let probing = false;
    let disposed = false;
    const clearRetry = () => {
        clearTimeout(retryTimer);
        retryTimer = null;
    };

    async function probeReadiness() {
        // One check at a time: Try again during a pending check joins it.
        if (probing || disposed) return;
        probing = true;
        clearRetry();
        try {
            // Readiness probe only: 204 means the install is structurally
            // configured and no blocking overlay is due. The wizard itself is
            // served by the /onboarding page the frame loads.
            const response = await apiFetch('/api/onboarding', { cache: 'no-store' });
            if (disposed) return;
            if (response.status === 204) {
                removeOverlay();
                return;
            }
            // Only the core's network gate answers 401 here: this page reached
            // the app without signing in (a separately hosted interface, or an
            // expired session). Its sign-in returns to this page.
            if (response.status === 401) {
                openSignIn();
                return;
            }
            if (!response.ok) throw new Error(`HTTP ${response.status}`);
            showSetupFrame(overlay);
        } catch (error) {
            if (disposed) return;
            console.error('Failed to load onboarding overlay:', error);
            showReadinessFailure(overlay, String(error?.message || error || 'unknown error'), probeReadiness);
            retryTimer = setTimeout(probeReadiness, retryDelay);
            retryDelay = Math.min(Math.round(retryDelay * 1.5), READINESS_RETRY_MAX_MS);
        } finally {
            probing = false;
        }
    }

    await probeReadiness();
    return () => {
        disposed = true;
        clearRetry();
        window.removeEventListener('message', handleMessage);
    };
}
