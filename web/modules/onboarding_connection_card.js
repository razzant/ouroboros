// A skill's declared connection card, the first row of the setup's Skills step (owner 21.11, 3A: the QR right in the
// wizard). The declaration is the skill's own (its manifest's `connection_card`, carried on its starter item as
// `connection`: CLIENT_CONTRACT §12); nothing here knows which service a card is for. Closed, it is one row like the
// others, its box the skill's starter tick, with Connect. Open: the skill's own steps, its token field and, once the
// host's identity check names the account, a QR of the skill's link with a one-time code made here, the account's
// @name, the same link for this device and the skill's options. Nothing is saved here: the token, the code and the
// options ride the one completion, and a failed check never blocks Continue or Finish. Each failure names only its
// own cause.
import { apiFetch } from './api_client.js';
import { escapeHtmlAttr as escapeHtml } from './ui_primitives.js';
import { qrEncoder, qrSvg } from './qr_svg.js';

const ACCOUNT = /^[A-Za-z0-9_]{1,64}$/;
const https = (value) => (/^https:\/\/[^\s"'<>]+$/.test(String(value || '')) ? String(value) : '');
const text = (value) => String(value ?? '').trim();
// A declared field name in words: `botToken` → "Bot token".
const fieldWords = (name) => text(name).replace(/([a-z0-9])([A-Z])/g, '$1 $2').toLowerCase().replace(/^./, (first) => first.toUpperCase());

/** The declaration as the card reads it, or null when it lacks a skill, a token field, a code field or a link with
 * both of its places. Steps, a title, a pitch, a hint under the QR and options are the skill's own words. */
export function normalizeConnection(raw) {
    if (!raw || typeof raw !== 'object') return null;
    const pairing = raw.pairing && typeof raw.pairing === 'object' ? raw.pairing : {};
    const link = https(pairing.link);
    const card = { skill: text(raw.skill), token: text(raw.token), code: text(pairing.code), link };
    if (!card.skill || !card.token || !card.code || !link.includes('{username}') || !link.includes('{code}')) return null;
    return {
        ...card, title: text(raw.title), pitch: text(raw.pitch), hint: text(pairing.hint),
        steps: (Array.isArray(raw.steps) ? raw.steps : []).map((step) => ({ text: text(step?.text), link: https(step?.link), action: text(step?.action) }))
            .filter((step) => step.text),
        options: (Array.isArray(raw.options) ? raw.options : []).filter((option) => text(option?.key) && text(option?.label))
            .map((option) => ({ key: text(option.key), label: text(option.label), on: String(option.on ?? 'on'), off: String(option.off ?? 'off'), fresh: option.new_connection === true })),
    };
}

/** A one-time pairing code: 16 random bytes as base64url, 22 characters (a link's start code may hold 64). */
export function pairingCode(random = (bytes) => globalThis.crypto.getRandomValues(bytes)) {
    const bytes = random(new Uint8Array(16));
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

/** The card of one starter `item` from its normalized `declaration`. `repaint` redraws the step; `ticked()` and
 * `tick()` read and set the skill's starter box. */
export function createConnectionCard(item, declaration, { check = checkIdentity, random, repaint = () => {}, ticked = () => true, tick = () => {}, encoder = qrEncoder } = {}) {
    const card = { open: false, token: '', code: '', check: null,
        options: Object.fromEntries(declaration.options.map((option) => [option.key, option.fresh])) };
    const service = text(item.title || item.name || declaration.skill);
    const title = declaration.title || service;
    const pitch = declaration.pitch || text(item.pitch || item.description);
    const own = (value) => (value ? ' data-i18n-authored' : '');  // the skill's words stay its own
    let sent = 0;
    const token = () => card.token.trim();
    const settled = () => (card.check && card.check.token === token() ? card.check : null);
    const link = (account) => declaration.link.replace('{username}', encodeURIComponent(account)).replace('{code}', encodeURIComponent(card.code));
    const saved = () => Boolean(token()) && settled()?.status !== 'refused';

    async function run() {
        const value = token();
        if (!value || ['checking', 'ok', 'refused'].includes(settled()?.status)) return;  // a transport failure may retry
        const mine = ++sent;
        card.check = { token: value, status: 'checking' };
        repaint();
        const answer = await check(declaration.skill, value);
        if (mine !== sent || token() !== value) return;  // the field changed meanwhile: this answer is not its
        card.check = { token: value, ...answer };
        repaint();
    }

    function statusWords(state) {
        return {
            checking: 'Checking…',
            refused: `${service} did not accept this token; it is not saved.`,
            unreachable: `Could not reach ${service} from here. The token is still saved when you finish.`,
            unavailable: `The token could not be checked (${state?.reason || 'no answer'}). It is still saved when you finish.`,
        }[state?.status] || '';
    }

    function optionsHtml() {
        return declaration.options.map((option) => `<label class="skills-connection-option"><input type="checkbox" class="ui-checkbox" data-connection-option="${escapeHtml(option.key)}"${card.options[option.key] ? ' checked' : ''}>
            <span${own(true)}>${escapeHtml(option.label)}</span></label>`).join('');
    }

    function pairHtml(account) {
        const target = link(account);
        const qr = qrSvg(target, { label: `QR code for @${account}` });
        return `<div class="skills-connection-pair">
            ${qr ? `<div class="skills-connection-qr">${qr}</div>` : ''}
            <div class="skills-connection-facts">
                <strong data-connection-account>@${escapeHtml(account)}</strong>
                ${declaration.hint ? `<span${own(true)}>${escapeHtml(declaration.hint)}</span>` : ''}
                <span>It connects when you finish setup.</span>
                <span>On this device? <a href="${escapeHtml(target)}" target="_blank" rel="noopener noreferrer" data-connection-link>Open @${escapeHtml(account)}</a></span>
                ${optionsHtml()}
            </div>
        </div>`;
    }

    // Closed, a set token says where it stands in the place of the pitch.
    function standing() {
        const state = settled();
        if (!token()) return '';
        if (state?.status === 'ok') return `@${state.username} · connects when you finish setup`;
        if (state?.status === 'refused') return `${service} did not accept the token`;
        return state?.status === 'checking' ? 'Checking the token…' : 'Token not checked · saved when you finish';
    }

    return {
        item,
        html(order = 0) {
            const state = settled();
            const id = escapeHtml(item.id);
            const field = `connection-${escapeHtml(declaration.skill)}-token`;
            const shown = card.open ? '' : standing();
            return `<div class="skills-connection" data-connection="${escapeHtml(declaration.skill)}" style="--arrive: ${order}">
                <div class="skills-connection-head">
                    <label class="skills-starter-skill" data-starter-skill="${id}">
                        <input type="checkbox" class="ui-checkbox" data-starter-item="${id}"${ticked() ? ' checked' : ''}>
                        <span class="skills-starter-words"><strong${own(declaration.title)}>${escapeHtml(title)}</strong>
                            <span class="skills-starter-pitch"${own(!shown && declaration.pitch)}>${escapeHtml(shown || pitch)}</span></span>
                    </label>
                    <button type="button" class="btn btn-default btn-sm" data-connection-toggle aria-expanded="${card.open}">${card.open ? 'Hide' : token() ? 'Change' : 'Connect'}</button>
                </div>
                ${card.open ? `<div class="skills-connection-body">
                    <ol class="skills-connection-steps">
                        ${declaration.steps.map((step) => `<li><span${own(true)}>${escapeHtml(step.text)}</span>${step.link ? ` <a class="btn btn-default btn-sm" href="${escapeHtml(step.link)}" target="_blank" rel="noopener noreferrer"${own(Boolean(step.action))}>${escapeHtml(step.action || 'Open')}</a>` : ''}</li>`).join('')}
                        <li class="skills-connection-token"><label for="${field}">${escapeHtml(fieldWords(declaration.token))}</label>
                            <span class="skills-connection-field"><input id="${field}" class="ui-control" type="password" autocomplete="off" spellcheck="false" data-connection-token value="${escapeHtml(card.token)}">
                            <button type="button" class="btn btn-default btn-sm" data-connection-check>Check</button></span></li>
                    </ol>
                    ${state && state.status !== 'ok' ? `<p class="skills-starter-sub" role="status" data-connection-status="${state.status}">${escapeHtml(statusWords(state))}</p>` : ''}
                    ${state?.status === 'ok' ? pairHtml(state.username) : ['unreachable', 'unavailable'].includes(state?.status) ? optionsHtml() : ''}
                </div>` : ''}
            </div>`;
        },
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
        /** `{skill: {fields, options}}` for the one completion, or null: nothing typed, or a token the service refused. */
        payload() {
            if (!ticked() || !saved()) return null;
            return { [declaration.skill]: {
                fields: { [declaration.token]: token(), [declaration.code]: card.code },
                options: Object.fromEntries(declaration.options.map((option) => [option.key, card.options[option.key] ? option.on : option.off])),
            } };
        },
        /** Summary's row while a token will be saved: [title, where it stands]. */
        summary: () => (ticked() && saved() ? [title, settled()?.status === 'ok' ? `@${settled().username}` : 'Token not checked'] : null),
        snapshot: () => ({ ...card, options: { ...card.options } }),
    };
}
