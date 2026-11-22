// A skill's declared connection card, the first row of the setup's Skills step (owner 21.11, 3A: the QR right in the
// wizard). The declaration is the skill's own (its manifest's `connection_card`, carried on its starter item as
// `connection`: CLIENT_CONTRACT §12); nothing here knows which service a card is for. Closed, it is one row like the
// others, its box the skill's starter tick, with Connect. Open: the skill's own steps, its token field and, once the
// host's identity check names the account, a QR of the skill's link with a one-time code made here, the account's
// @name, the same link for this device and the skill's options. Nothing is saved here: the token rides the one
// completion into Settings, and the code and the options ride with it only once the step painted the QR and the link
// that carry the code for exactly this token (the host posts them to the skill once it runs): a code nobody saw would
// leave the skill waiting for a scan nobody can make, so a check that answers once the card is hidden or the step is
// left arms nothing, and nothing names the account as connected until that painting. A draft that changes takes the
// shown check of the draft before it off the page at once (an old bot's QR is never left to scan), the field untouched.
// A failed or unfinished check never blocks Continue or Finish; the token alone is saved then. A draft no service takes
// (a space or line break inside, or longer than any token) is never checked or sent. Each failure names only its own
// cause, and only the service's own refusal says it did not accept the token. The card's
// declared words are a built-in skill's, the maintainers' as the catalog's are, so they go through the interface's
// translation (owner 2026-11-21, v3.1 §5); words composed with a name go through `fmt`, and an account's @name, a
// link and the token stay as they are.
import { apiFetch } from './api_client.js';
import { fmt } from './i18n.js';
import { escapeHtmlAttr as escapeHtml } from './ui_primitives.js';
import { qrEncoder, qrSvg } from './qr_svg.js';

const ACCOUNT = /^[A-Za-z0-9_]{1,64}$/;
const MAX_TOKEN = 512;  // the host's own bound on a token (onboarding_connections._TOKEN_MAX)
// What makes a draft no token at all, before any check: a space or line break inside, or more than any token holds.
const flaw = (value) => (value.length > MAX_TOKEN ? 'long' : /\s/.test(value) ? 'spaced' : '');
const https = (value) => (/^https:\/\/[^\s"'<>]+$/.test(String(value || '')) ? String(value) : '');
const text = (value) => String(value ?? '').trim();
// A checked token whose QR the page never painted (the card hidden or the step left before the answer): no account named.
const UNSHOWN = 'Token checked · QR not shown yet';
// Words composed with a name or a reason, through the composed seam and marked so the overlay leaves them alone.
const composed = (template, params = {}) => [fmt(template, params), ` data-i18n-fmt="${escapeHtml(template)}"`];
// A declared field name in words: `botToken` → "Bot token".
const fieldWords = (name) => text(name).replace(/([a-z0-9])([A-Z])/g, '$1 $2').toLowerCase().replace(/^./, (first) => first.toUpperCase());

/** The declaration as the card reads it, or null when it lacks a skill, a token field or a link with both of its
 * places. Steps, a title, a pitch, a hint and a note beside the QR and options are the skill's own words; `allow` is
 * the host's: this install grants skills nothing on its own, so the skill starts once allowed in Extensions. */
export function normalizeConnection(raw) {
    if (!raw || typeof raw !== 'object') return null;
    const pairing = raw.pairing && typeof raw.pairing === 'object' ? raw.pairing : {};
    const link = https(pairing.link);
    const card = { skill: text(raw.skill), token: text(raw.token), link };
    if (!card.skill || !card.token || !link.includes('{username}') || !link.includes('{code}')) return null;
    return {
        ...card, title: text(raw.title), pitch: text(raw.pitch), hint: text(pairing.hint), note: text(pairing.note), allow: raw.needs_allow === true,
        steps: (Array.isArray(raw.steps) ? raw.steps : []).map((step) => ({ text: text(step?.text), link: https(step?.link), action: text(step?.action) }))
            .filter((step) => step.text),
        options: (Array.isArray(raw.options) ? raw.options : []).filter((option) => text(option?.key) && text(option?.label))
            .map((option) => ({ key: text(option.key), label: text(option.label), on: String(option.on ?? 'on'), off: String(option.off ?? 'off'), fresh: option.new_connection === true })),
    };
}

/** A one-time pairing code as a skill mints it: 24 random bytes as base64url, 32 characters (a start code holds 64). */
export function pairingCode(random = (bytes) => globalThis.crypto.getRandomValues(bytes)) {
    const bytes = random(new Uint8Array(24));
    return btoa(String.fromCharCode(...bytes)).replace(/\+/g, '-').replace(/\//g, '_').replace(/=+$/, '');
}

/** The host's identity check for a skill's token: `{status: "ok", username}`, `"refused"` or `"unreachable"` as the
 * host found them, `"unavailable"` (with why, never the token) when the check itself did not run. */
export async function checkIdentity(skill, token, fetchImpl = apiFetch) {
    try {
        const response = await fetchImpl('/api/onboarding/connection/identity', {
            method: 'POST', cache: 'no-store', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ skill, token }) });
        const data = await response.json().catch(() => null);
        if (data?.ok === true && ACCOUNT.test(String(data.username || ''))) return { status: 'ok', username: String(data.username) };
        if (data?.ok === false && ['refused', 'unreachable'].includes(data.cause)) return { status: data.cause };
        return { status: 'unavailable', reason: response.ok ? 'an answer it could not read' : `HTTP ${response.status}` };
    } catch {
        return { status: 'unavailable', reason: 'the request did not complete' };
    }
}

/** The card of one starter `item` from its normalized `declaration`. `repaint` redraws the step, `edited` tells it the
 * card changed its own markup in place; `ticked()` and `tick()` read and set the skill's starter box. */
export function createConnectionCard(item, declaration, { check = checkIdentity, random, repaint = () => {}, edited = () => {}, ticked = () => true, tick = () => {}, encoder = qrEncoder } = {}) {
    const card = { open: false, token: '', code: '', check: null, drawn: '',
        options: Object.fromEntries(declaration.options.map((option) => [option.key, option.fresh])) };
    const service = text(item.title || item.name || declaration.skill);
    const title = declaration.title || service;
    const pitch = declaration.pitch || text(item.pitch || item.description);
    // Only a bare name and a skill author's description (no declared or catalog words) stay as they are, as on rows.
    const authored = (own) => (own ? ' data-i18n-authored' : '');
    const bareTitle = !declaration.title && !text(item.title);
    const authorPitch = !declaration.pitch && !text(item.pitch);
    let sent = 0;
    let face = null;  // the check the card's last markup showed: {token, pair}; `painted()` makes a pair's `card.drawn`
    let gone = '';  // the draft whose shown check an edit took off the page, until the page shows a check again
    const token = () => card.token.trim();
    // Where the token stands: a local flaw first (never checked), else the host's answer for exactly this token.
    const settled = () => (flaw(token()) ? { token: token(), status: flaw(token()) }
        : card.check && card.check.token === token() ? card.check : null);
    const link = (account) => declaration.link.replace('{username}', encodeURIComponent(account)).replace('{code}', encodeURIComponent(card.code));
    const saved = () => Boolean(token()) && !['refused', 'spaced', 'long'].includes(settled()?.status);
    // The step painted the QR and the link of this card's code for exactly this token: the only code it may arm.
    const drawn = () => settled()?.status === 'ok' && card.drawn === token();
    // Where the install grants nothing on its own, the skill starts only once allowed: that, never «connects».
    const later = declaration.allow ? 'starts after you allow it in Extensions' : 'connects when you finish setup';

    async function run() {
        const value = token();
        if (flaw(value)) { repaint(); return; }  // no token at all: said here, never sent to the check
        if (!value || ['checking', 'ok', 'refused'].includes(settled()?.status)) return;  // a transport failure may retry
        const mine = ++sent;
        card.check = { token: value, status: 'checking' };
        repaint();
        const answer = await check(declaration.skill, value);
        if (mine !== sent || token() !== value) return;  // the field changed meanwhile: this answer is not its
        card.check = { token: value, ...answer };
        repaint();
    }

    // Each status in its own words: [the words, their composed-seam mark], fixed ones as plain text the overlay reads.
    function statusWords(state) {
        const words = {
            checking: 'Checking…',
            spaced: 'A token has no spaces or line breaks inside, so this one is not saved.',
            long: 'A token is at most 512 characters, so this one is not saved.',
        }[state?.status];
        if (words) return [words, ''];
        return {
            refused: () => composed('{service} did not accept this token; it is not saved.', { service }),
            unreachable: () => composed('Could not reach {service} from here, so there is no QR code. The token is still saved when you finish.', { service }),
            unavailable: () => composed('The token could not be checked ({reason}), so there is no QR code. The token is still saved when you finish.', { reason: state?.reason || 'no answer' }),
        }[state?.status]?.() || ['', ''];
    }

    function optionsHtml() {
        return declaration.options.map((option) => `<label class="skills-connection-option"><input type="checkbox" class="ui-checkbox" data-connection-option="${escapeHtml(option.key)}"${card.options[option.key] ? ' checked' : ''}>
            <span>${escapeHtml(option.label)}</span></label>`).join('');
    }

    function pairHtml(account) {
        const target = link(account);
        const qr = qrSvg(target, { label: `QR code for @${account}` });
        const [allowed, allowMark] = composed(`{service} ${later}.`, { service });
        const [open, openMark] = composed('Open @{account}', { account });
        return `<div class="skills-connection-pair">
            ${qr ? `<div class="skills-connection-qr" data-i18n-authored>${qr}</div>` : ''}
            <div class="skills-connection-facts">
                <strong data-connection-account data-i18n-authored>@${escapeHtml(account)}</strong>
                ${declaration.hint ? `<span>${escapeHtml(declaration.hint)}</span>` : ''}
                ${declaration.allow ? `<span data-connection-allow${allowMark}>${escapeHtml(allowed)}</span>`
                    : declaration.note ? `<span>${escapeHtml(declaration.note)}</span>` : '<span>It connects when you finish setup.</span>'}
                <span><span>On this device?</span> <a href="${escapeHtml(target)}" target="_blank" rel="noopener noreferrer" data-connection-link${openMark}>${escapeHtml(open)}</a></span>
                ${optionsHtml()}
            </div>
        </div>`;
    }

    // The token field: under the skill's last step (which says what to paste), its name then read out only; alone, named.
    function fieldHtml(field, underStep) {
        return `<label for="${field}"${underStep ? ' class="skills-starter-sr"' : ''}>${escapeHtml(fieldWords(declaration.token))}</label>
            <span class="skills-connection-field"><input id="${field}" class="ui-control" type="password" autocomplete="off" spellcheck="false" data-connection-token value="${escapeHtml(card.token)}">
            <button type="button" class="btn btn-default btn-sm" data-connection-check>Check</button></span>`;
    }

    // Closed, a set token says where it stands in the place of the pitch: [the words, their composed-seam mark].
    function standing() {
        const state = settled();
        if (!token()) return null;
        if (state?.status === 'ok') return drawn() ? composed(`@{account} · ${later}`, { account: state.username }) : [UNSHOWN, ''];
        if (state?.status === 'refused') return composed('{service} did not accept the token', { service });
        if (state?.status === 'spaced' || state?.status === 'long') return ['Not a token · not saved', ''];
        return [state?.status === 'checking' ? 'Checking the token…' : 'Token not checked · saved when you finish', ''];
    }

    return {
        item,
        html(order = 0) {
            const state = settled();
            const id = escapeHtml(item.id);
            const field = `connection-${escapeHtml(declaration.skill)}-token`;
            const shown = card.open ? null : standing();
            const [said, saidMark] = state && state.status !== 'ok' ? statusWords(state) : ['', ''];
            face = card.open && state ? { token: state.token, pair: state.status === 'ok' } : null;
            if (face) gone = '';
            return `<div class="skills-connection" data-connection="${escapeHtml(declaration.skill)}" style="--arrive: ${order}">
                <div class="skills-connection-head">
                    <label class="skills-starter-skill" data-starter-skill="${id}">
                        <input type="checkbox" class="ui-checkbox" data-starter-item="${id}"${ticked() ? ' checked' : ''}>
                        <span class="skills-starter-words"><strong${authored(bareTitle)}>${escapeHtml(title)}</strong>
                            <span class="skills-starter-pitch"${shown ? shown[1] : authored(authorPitch)}>${escapeHtml(shown ? shown[0] : pitch)}</span></span>
                    </label>
                    <button type="button" class="btn btn-default btn-sm" data-connection-toggle aria-expanded="${card.open}">${card.open ? 'Hide' : token() ? 'Change' : 'Connect'}</button>
                </div>
                ${card.open ? `<div class="skills-connection-body">
                    <ol class="skills-connection-steps">
                        ${declaration.steps.map((step, index) => `<li${index === declaration.steps.length - 1 ? ' class="skills-connection-token"' : ''}><span>${escapeHtml(step.text)}</span>${step.link ? ` <a class="btn btn-default btn-sm" href="${escapeHtml(step.link)}" target="_blank" rel="noopener noreferrer">${escapeHtml(step.action || 'Open')}</a>` : ''}${index === declaration.steps.length - 1 ? fieldHtml(field, true) : ''}</li>`).join('')}
                        ${declaration.steps.length ? '' : `<li class="skills-connection-token">${fieldHtml(field, false)}</li>`}
                    </ol>
                    ${said ? `<p class="skills-starter-sub" role="status" data-connection-status="${state.status}"${saidMark}>${escapeHtml(said)}</p>` : ''}
                    ${state?.status === 'ok' ? pairHtml(state.username) : ''}
                </div>` : ''}
            </div>`;
        },
        /** The step wrote the card's last markup into the page: a QR and link it carried were seen, and leaving the card
         * or the step keeps that. Never from building markup alone: only a written page counts. */
        painted() { if (face?.pair) card.drawn = face.token; },
        /** Handles the card's own controls; false for anything else. */
        click(target) {
            if (target.closest('[data-connection-toggle]')) {
                card.open = !card.open;
                if (card.open) {
                    card.code ||= pairingCode(random);  // once per card, kept until Finish
                    if (!ticked()) tick();
                    void encoder().then(repaint, () => {});
                }
                repaint();
                return true;
            }
            if (target.closest('[data-connection-check]')) { void run(); return true; }
            return false;
        },
        input(event) {
            const field = event.target.closest?.('[data-connection-token]');
            if (!field) return false;
            card.token = field.value;
            if (face && face.token !== token()) {  // the shown check is another draft's: off the page, the field kept
                field.closest?.('[data-connection]')?.querySelectorAll?.('.skills-connection-pair, [data-connection-status]')
                    .forEach((node) => node.remove());
                [gone, face] = [face.token, null];
                edited();
            } else if (gone && gone === token()) repaint();  // back to the draft it checked: its answer shows again
            if (event.inputType === 'insertFromPaste') void run();
            return true;
        },
        change(target) {
            if (target.closest('[data-connection-token]')) { card.token = target.value; void run(); return true; }
            const option = target.closest('[data-connection-option]');
            if (!option) return false;
            card.options[option.dataset.connectionOption] = option.checked;
            return true;
        },
        keydown(event) {
            if (event.key !== 'Enter' || !event.target.closest?.('[data-connection-token]')) return false;
            event.preventDefault();
            card.token = event.target.value;
            void run();
            return true;
        },
        /** `{skill: {fields: {token}, code, options}}` for the one completion once the step painted this token's QR and
         * link, `{skill: {fields: {token}}}` while it did not (a check that failed, is still running, never ran, or
         * answered while the card was hidden or the step left), or null: nothing typed, no token at all, or a token the
         * service refused. */
        payload() {
            if (!ticked() || !saved()) return null;
            return { [declaration.skill]: { fields: { [declaration.token]: token() }, ...(drawn() ? { code: card.code,
                options: Object.fromEntries(declaration.options.map((option) => [option.key, card.options[option.key] ? option.on : option.off])) } : {}) } };
        },
        /** Summary's row while a token will be saved: [title, where it stands, whether that is an account's @name]; the
         * account only once its QR was painted. */
        summary: () => (ticked() && saved() ? [title, ...(drawn() ? [`@${settled().username}`, true]
            : [settled()?.status === 'ok' ? UNSHOWN : 'Token not checked', false])] : null),
        snapshot: () => ({ ...card, options: { ...card.options } }),
    };
}
