// Sign-ins of a skill to the services it declares (docs/DESIGN.md "Sign-ins on a
// skill's card"). The card carries one meta line in its who block and, only while a sign-in needs the owner, one tray;
// the skill's sheet carries Sign-ins: a group per service, a DESIGN §6 row per account, and the sign-in and clients
// editors in place under the group's head, each result on the line of the control that caused it. Everything is
// painted from GET /api/sign-in/status and done with POST /api/sign-in, whose answers carry no token, secret or
// verifier; a client secret typed here is sent once and never shown back. Fixed words go through `tr`; the skill's own
// words (its service label, permission labels and notes) and account names are shown as written (data-i18n-authored).
import { fetchJson, jsonPost } from './api_client.js';
import { tr } from './i18n.js';
import { bindMenu } from './ui_interactions.js';
import { escapeHtmlAttr as escapeHtml, setResult } from './ui_primitives.js';
import { copyTextToClipboard, openExternalViaHostBridge, revealNewRow } from './ui_helpers.js';
import { safeExternalUrl } from './utils.js';
import { buttonHtml, relativeAge, shortDate, statusHtml, trayHtml } from './skill_presentation.js';

export const SIGN_IN_STATUS_EVENT = 'ouro:sign-in-status';
const LOOPBACK = new Set(['127.0.0.1', 'localhost', '::1', '[::1]']);
const ENDED = new Set(['refresh_rejected', 'token_rejected', 'expired']);  // the provider no longer honours the grant
const list = (value) => (Array.isArray(value) ? value : []);
const authored = (text) => `<span data-i18n-authored>${escapeHtml(text)}</span>`;
const providerOf = (service) => String(service?.label || service?.service || '');
const nameOf = (account) => String(account?.label || account?.subject || account?.key || '');
const servicesOf = (status, skill) => list(status?.services).filter((row) => row?.skill === skill);
const hostOf = (address) => { try { return new URL(address).hostname; } catch { return String(address || ''); } };
const hintKey = (value) => String(value || '').trim().toLowerCase();
const scopeWords = (value) => String(value || '').split(/\s+/).filter(Boolean);
const attemptOf = (row) => typeof row?.attempt_id === 'string' ? row.attempt_id : '';
const clientOf = (service, id) => list(service.clients).find((row) => row.id === id);
const signInClients = (service) => list(service?.clients).filter((row) => row && row.return !== 'unknown');
const signInClientOf = (service, id) => signInClients(service).find((row) => row.id === id);
const accountOf = (service, key) => list(service.accounts).find((row) => row.key === key);
const bindingOf = (service) => ({ issuer: service.issuer, ...(service.endpoints && { endpoints: { ...service.endpoints } }) });
const bindingKey = (row) => JSON.stringify([row.issuer, Object.entries(row.endpoints || {}).sort()]);

function optionalChoices(service) {
    const seen = new Set(list(service.scopes));
    return list(service.optional_scopes).filter((row) => {
        if (!row?.scope || seen.has(row.scope)) return false;
        seen.add(row.scope);
        return true;
    });
}

// Compare only the offer the owner can see, not account status, timestamps or client usage.
const choicesKey = (service) => JSON.stringify([bindingKey(service), list(service.scopes), Object.entries(service.scope_labels || {}).sort(),
    Object.entries(service.scope_spellings || {}).sort(),
    optionalChoices(service).map((row) => [row.scope, row.label, row.default]),
    signInClients(service).map((row) => [row.id, row.label, row.source, row.return, row.client_id_tail, row.registration]).sort()]);

/** A fixed sentence by its code with its `{name}` values filled: plain text, or HTML that keeps each value as written. */
function words(code, english, values = {}, asHtml = false) {
    return String(tr(`skills.sign_in.${code}`, english)).split(/(\{\w+\})/).map((part) => {
        const key = /^\{(\w+)\}$/.exec(part)?.[1];
        if (key && key in values) return asHtml ? authored(values[key]) : String(values[key]);
        return asHtml ? escapeHtml(part) : part;
    }).join('');
}
const html = (code, english, values) => words(code, english, values, true);
/** A step with no usable provider answer: nothing to rely on, so one status read by its attempt, and its one-time code is never sent again. */
const noAnswer = (result) => ['provider_unreachable', 'exchange_not_reusable', 'sign_in_not_filed', 'offline', 'unconfirmed', 'answer_saved'].includes(result?.reason)
    || (result?.reason === 'exchange_rejected' && /^HTTP (429|5\d\d)$/.test(String(result.error || '')));
/** Why this skill cannot start a sign-in now, in the owner's words: `reason` is a refusal's (`skill_…`), else the status row's `admission`. */
export function admissionWords(service, reason = `skill_${service?.admission}`, asHtml = false) {
    const say = (code, english) => words(code, english, { skill: service.skill }, asHtml);
    if (reason === 'skill_disabled') return say('skill_disabled', '{skill} is off. Turn it on, then sign in.');
    if (reason === 'skill_not_trusted') return say('skill_not_trusted', 'This installation doesn’t run {skill}’s code, so it can’t sign in here.');
    return say('skill_not_reviewed', '{skill}’s current files have no review that lets them run. Review them, then sign in.');
}
/** Permissions in the skill's own words (its labels), else the scope as declared. */
const scopeText = (service, scopes) => list(scopes).map((scope) => service.scope_labels?.[scope]
    || list(service.optional_scopes).find((row) => row?.scope === scope)?.label || scope).join(', ');
const wordSpellings = (service, word) => [word, ...list(service.scope_spellings?.[word])];
const offerWords = (service, scope) => scopeWords(scope).flatMap((word) => wordSpellings(service, word));
const offerText = (service, scope, label = service.scope_labels?.[scope] || scope) => {
    const raw = offerWords(service, scope).join(', ');
    return label === raw ? raw : `${label} (${raw})`;
};
const act = (action, label, { account, client, tone = 'default', attrs = '' } = {}) => buttonHtml(label, { tone, attrs: [
    `data-sign-in-act="${action}"`, account === undefined ? '' : `data-sign-in-account="${escapeHtml(account)}"`,
    client ? `data-client="${escapeHtml(client)}"` : '', attrs].filter(Boolean).join(' ') });
const resultHtml = (key, result = null, row = false) => `<span class="ui-result${row ? ' settings-row-result' : ''}" data-sign-in-result="${escapeHtml(key)}" role="status" aria-live="polite"${
    result?.tone ? ` data-tone="${escapeHtml(result.tone)}"` : ''}>${escapeHtml(result?.text || '')}</span>`;

/** Read a status snapshot; the mounted Skills view publishes it to its cards and sheet. */
export async function readSignInStatus({ signal } = {}) {
    const data = await fetchJson('/api/sign-in/status', signal ? { signal } : {});
    if (data?.ok !== true || !Array.isArray(data.services)) throw new Error('The sign-in status is unavailable.');
    return data;
}

/** One step of POST /api/sign-in; a refused step and a failed request both answer `{ok: false, reason, error}`. */
async function step(body) {
    try {
        const result = await jsonPost('/api/sign-in', body, { requireBody: true });
        return typeof result?.ok === 'boolean' ? result : { ok: false, reason: 'unconfirmed' };
    } catch (err) {
        return { ok: false, reason: err?.body?.reason || (err?.status && err.status < 500 ? `http_${err.status}` : 'unconfirmed'),  // a 5xx or no reply is no answer, never a refusal
            error: String(err?.body?.error || err?.message || err) };
    }
}

/** One account in the owner's words: its status (HTML) and tone, and while it needs the owner its sentence and note. */
export function accountState(service, account) {
    const values = { provider: providerOf(service), account: nameOf(account), skill: service.skill, date: shortDate(account?.since) };
    const more = { tone: 'warn', needs: true, status: html('status_more', 'Needs more access') };
    if (list(account?.missing_required).length && (account.reason === 'insufficient_scope'
            || (account.state === 'signed_in' && account.via !== 'handover'))) {
        return { ...more, sentence: html('missing', 'Signed in, but {provider} didn’t allow: {scopes}. {skill} can’t use this account until you sign in again and allow it.',
            { ...values, scopes: scopeText(service, account.missing_required) }) };
    }
    if (account?.state !== 'signed_in') {
        const ended = ENDED.has(account?.reason);
        const approval = account?.reason === 'provider_approval_required';  // the owner's client choice, not a provider refusal
        const next = approval
            ? html('approve_client', 'Choose or add an approved client for {provider}, then sign in again.', values)
            : account?.reason === 'provider_scope_mismatch'
                ? html('extra_access', '{provider} gave this sign-in permissions you did not select. Review the permissions and sign in again. If it still gives extra permissions, choose a separate client.', values) : '';
        return { tone: approval ? 'warn' : 'danger', needs: true, status: html('status_again', 'Sign in again'),
            sentence: next || (!ended ? html('not_signed', '{account} isn’t signed in to {provider}. Sign in again.', values)
                : values.date ? html('refused_on', '{provider} no longer accepts the sign-in for {account} (refused {date}). Sign in again.', values)
                    : html('refused', '{provider} no longer accepts the sign-in for {account}. Sign in again.', values)),
            note: ended && service.notes?.refresh_rejected ? authored(service.notes.refresh_rejected) : '' };
    }
    if (account.reason === 'insufficient_scope') {
        const missing = list(account.missing_step_up);
        return { ...more, sentence: missing.length ? html('step_up', '{skill} needs more access for what was asked: {scopes}. Sign in again to allow it.',
            { ...values, scopes: scopeText(service, missing) }) : more.status };
    }
    if (account.via === 'handover' && account.verified === false) {
        return { tone: 'muted', needs: false, status: html('status_moved', 'Moved from {skill}’s own sign-in · not confirmed', values) };
    }
    return { tone: 'ok', needs: false, status: html('status_ok', 'Signed in') };
}

/** The card names the declared provider and its signed-in accounts, or omits the line if none is declared. */
export function signInMetaHtml(skillName, status) {
    const parts = servicesOf(status, skillName).map((service) => {
        const signed = list(service.accounts).filter((row) => row?.state === 'signed_in').map(nameOf);
        const lead = signed.length ? html('meta_signed', 'Signed in to {provider}', { provider: providerOf(service) })
            : html('meta_none', 'Not signed in to {provider}', { provider: providerOf(service) });
        return `<button type="button" class="skills-link-button ui-hit-area" data-skill="${escapeHtml(skillName)}" data-skill-action="open_detail" data-sign-in-service="${
            escapeHtml(service.service)}">${lead}</button>${signed.length ? `: ${signed.map(authored).join(' · ')}` : ''}`;
    });
    return parts.length ? `<p class="skills-card-facts" data-skill-sign-ins>${parts.join('; ')}</p>` : '';
}

const cardButton = (skill, service, account, label) => buttonHtml(label, { attrs: `data-skill="${escapeHtml(skill)}" data-skill-action="open_detail" data-sign-in-service="${
    escapeHtml(service.service)}" data-sign-in-account="${escapeHtml(account)}" data-sign-in-begin` });

/** The card's one sign-in tray, only while a sign-in needs the owner (an ended or short grant, a required service with no account); else ''. */
export function signInTray(skillName, status) {
    for (const service of servicesOf(status, skillName)) {
        if (service.admission) continue;  // nothing to sign in to until the skill may act
        const accounts = list(service.accounts);
        const account = accounts.find((row) => accountState(service, row).needs);
        if (account) {
            const view = accountState(service, account);
            return trayHtml({ tone: view.tone === 'danger' ? 'danger' : '', attrs: 'data-sign-in-tray', sentence: view.sentence, note: view.note,
                actions: cardButton(skillName, service, account.key, words('again', 'Sign in again')) });
        }
        if (service.optional === false && !accounts.length) {
            return trayHtml({ attrs: 'data-sign-in-tray', sentence: html('required', '{skill} works only once you sign in to {provider}.', { skill: skillName, provider: providerOf(service) }),
                actions: cardButton(skillName, service, '', words('sign_in', 'Sign in')) });
        }
    }
    return '';
}

/** A refused or failed step, as plain text for its result line. */
export function failureWords(service, result, { client = '', returnAddress = '', complete = false, account = '' } = {}) {
    const error = String(result?.error || result?.reason || '');
    const values = { provider: providerOf(service), host: hostOf(service.issuer), skill: service.skill, client, account, error, kind: error.split(':')[0] };
    if (complete && result?.reason !== 'answer_saved' && noAnswer(result)) {
        return words('no_result', 'Ouroboros has no confirmed result for this sign-in, and it sends each one-time code only once. If this window still waits, choose Start again.');
    }
    switch (result?.reason) {
    case 'skill_not_reviewed': case 'skill_disabled': case 'skill_not_trusted': return admissionWords(service, result.reason);
    case 'consent_changed': return words('consent_changed', '{account} was signed in again after you opened this. Review it and choose again.', values);
    case 'revocation_not_confirmed': return words('revoke_stays_unconfirmed', '{provider} didn’t confirm the revocation, and the sign-in for {account} is kept here. Check your {provider} account and remove the permission there if it is still listed, then choose Disconnect here.', values);
    case 'revocation_not_asked': return result.revocation_reason === 'no_endpoint'
        ? words('revoke_stays_no_address', '{provider} publishes no revocation address, so it wasn’t asked and the sign-in for {account} is kept here. Remove the permission in your {provider} account, then choose Disconnect here.', values)
        : result.revocation_reason === 'no_tokens'
            ? words('revoke_stays_no_token', 'No token is saved for {account}, so there was nothing to revoke at {provider}; the sign-in is kept here. Remove the permission in your {provider} account if it is still listed, then choose Disconnect here.', values)
            : words('revoke_stays_unapproved', '{provider} wasn’t asked: its saved addresses can’t be confirmed. The sign-in for {account} is kept here. Remove the permission in your {provider} account, then choose Disconnect here.', values);
    case 'provider_unreachable': return words('no_page', 'Couldn’t reach {host} ({kind}), so no sign-in page was opened. Open the sign-in page again once this computer is online.', values);
    case 'registration_unconfirmed': return words('registration_unconfirmed', '{provider} may have registered a client that never reached Ouroboros, so no new client is available here. Open the sign-in page again to ask it to register another.', values);
    case 'answer_saved': return words('answer_saved', '{provider}’s answer to this sign-in is saved, but Ouroboros couldn’t read which account it names yet. Choose Finish sign-in again; its one-time code isn’t sent a second time.', values);
    case 'denied': return words('exchange', '{provider} refused the sign-in ({error}). Start again.', values);
    case 'exchange_rejected':
        if (error.includes('invalid_client')) return words('bad_client', '{provider} refused the client «{client}» (invalid_client). Check its client ID and secret in Clients, or choose another client.', values);
        if (error.includes('invalid_grant')) return words('code_used', 'This sign-in link was already used or has expired; start again.');
        return words('exchange', '{provider} refused the sign-in ({error}). Start again.', values);
    case 'sign_in_superseded': return words('superseded', 'A newer sign-in for {skill} started meanwhile; finish that one or start again.', values);
    case 'no_client': return words('no_client', 'Add a client for {provider} first; it returns to {address}.', { ...values, address: returnAddress || error });
    case 'client_required': return words('choose_client', 'Choose a client first.');
    case 'unknown_client': return words('unknown_client', 'That client is no longer saved; choose another one.');
    case 'identity_unavailable': return words('identity_unconfirmed', '{provider} didn’t say which account signed in, so no account was confirmed. Start again.', values);
    case 'issuer_mismatch': case 'issuer_missing': return words('issuer', 'The answer didn’t come from {provider} as declared ({error}). Nothing was saved; start again.', values);
    case 'no_pkce': case 'no_oauth_metadata': return words('provider_offers_none', '{provider} offers no sign-in Ouroboros can use ({error}).', values);
    case 'provider_approval_required': return words('approve_client', 'Choose or add an approved client for {provider}, then sign in again.', values);
    case 'provider_scope_mismatch': return words('extra_access', '{provider} gave this sign-in permissions you did not select. Review the permissions and sign in again. If it still gives extra permissions, choose a separate client.', values);
    case 'provider_scope_unconfirmed': return words('unknown_access', 'The permissions for this account could not be confirmed. Review them and sign in again.');
    case 'account_mismatch': return words('account_mismatch', 'Sign in as the current account, or disconnect it here before switching.');
    case 'exchange_not_reusable': case 'sign_in_not_filed': case 'offline': case 'unconfirmed':
        return words('unconfirmed', 'Ouroboros didn’t confirm the result. Check the sign-in before trying again.');
    case 'http_400': if (complete) return words('not_return', 'This address carries no sign-in return; paste the whole address your browser stopped on.');
    }
    return words('failed', 'It didn’t go through: {error}', values);
}

/** What Disconnect and Revoke ask in place first; Revoke names the sibling grants the provider may end with it. */
export function confirmSentence(service, account, what) {
    const values = { provider: providerOf(service), account: nameOf(account), skill: service.skill };
    if (what !== 'revoke') return html('disconnect_forgets', '{skill} stops using {account} and this installation forgets the sign-in. {provider} keeps the permission until you remove it in your {provider} account; to end it there now, cancel and choose Revoke instead.', values);
    const siblings = list(account.siblings).map((row) => `${row.skill} (${nameOf(account)})`).join(', ');
    return siblings ? html('revoke_asks_siblings', 'Revoke at {provider}? {provider} ends this sign-in, and may also end other sign-ins made with the same client for {account}: {siblings}. Skills that keep their own sign-in may lose access too. You can sign in again later. The sign-in is forgotten here only once {provider} confirms.', { ...values, siblings })
        : html('revoke_asks', 'Revoke at {provider}? {provider} ends this sign-in. Skills that keep their own sign-in may lose access too. You can sign in again later. The sign-in is forgotten here only once {provider} confirms.', values);
}

/** The client a sign-in starts with: the account's own, else the one a sibling skill used for that address, else the only one. */
export function preselectedClient(service, { account = null, address } = {}) {
    const clients = signInClients(service);
    const who = hintKey(address ?? account?.label);
    const own = clients.find((row) => row?.id && row.id === account?.client?.id);
    if (own && who === hintKey(account?.label)) return { client: own, why: 'own' };
    for (const client of who ? clients : []) {
        const use = list(client.used_by).find((row) => row?.skill !== service.skill && hintKey(row?.account) === who);
        if (use) return { client, why: 'sibling', by: String(use.skill) };
    }
    return clients.length === 1 ? { client: clients[0], why: 'only' } : { client: null, why: '' };
}

/** A client returns to the computer where Ouroboros runs by its rule or by an exact loopback address; a browser elsewhere cannot come back to either. */
const returnsToLoopback = (rule) => rule === 'loopback' || (rule.startsWith('exact:') && LOOPBACK.has(hostOf(rule.slice(6))));

/** The loopback return note appears when this page is not on a loopback host and the client returns to loopback. */
export const floorShown = (pageHost, clientReturn) => returnsToLoopback(String(clientReturn || '')) && !LOOPBACK.has(String(pageHost || '').toLowerCase());

/** The note under the client choice: a web client's exact return address (with the floor sentence when that address is on the computer where Ouroboros runs and this page is not), or the floor sentence; '' otherwise. */
export function returnNoteHtml(service, client, pageHost, hasClients = true) {
    const rule = client ? String(client.return || 'loopback') : hasClients ? '' : 'loopback';  // none saved: the host registers a loopback one
    const floor = floorShown(pageHost, rule) ? html('floor', 'This sign-in comes back to the computer where Ouroboros runs. Sign in from that computer, or sign in here and paste the address your browser stops on (Details › Finish sign-in).') : '';
    if (!rule.startsWith('exact:')) return floor;
    const exact = html('web_client', '{provider} sends you back to {address}; this client must list exactly this address.', { provider: providerOf(service), address: rule.slice(6) });
    return floor ? `${exact} ${floor}` : exact;
}

const returnsHtml = (client) => client?.return === 'unknown' ? html('return_unknown', 'return address unknown')
    : (String(client?.return || 'loopback').startsWith('exact:')
        ? html('returns_to', 'returns to {address}', { address: String(client.return).slice(6) }) : html('returns_here', 'returns to this computer'));
const clientLine = (client) => `${authored(client.label || client.id)} · …${escapeHtml(client.client_id_tail || '')} · ${returnsHtml(client)}`;

function clientSource(client) {
    const source = String(client?.source || '');
    if (source === 'owner') return html('client_owner', 'Saved by you');
    if (source === 'registered') return html('client_registered', 'Registered by Ouroboros');
    if (source.startsWith('handover:')) return html('client_imported', 'From {skill}’s earlier sign-in', { skill: source.slice('handover:'.length) });
    return '';
}

const suggestion = (pick, id) => pick.why === 'sibling' && pick.client?.id === id
    ? ` — ${html('sibling', '{skill} uses it for this account', { skill: pick.by })}` : '';

function confirmTray(service, account, what) {
    const label = what === 'revoke' ? words('revoke_at', 'Revoke at {provider}', { provider: providerOf(service) }) : words('disconnect_here', 'Disconnect here');
    return trayHtml({ tone: what === 'revoke' ? 'danger' : '', attrs: `data-sign-in-confirm="${escapeHtml(account.key)}"`, sentence: confirmSentence(service, account, what),
        actions: act(`confirm_${what}`, label, { account: account.key, tone: what === 'revoke' ? 'danger' : 'default' }) + act('cancel_confirm', words('cancel', 'Cancel'), { account: account.key }) });
}

/** One DESIGN §6 row per account: name, `via` its client, status, what is allowed, its result line; a confirmation under it. */
export function signInRowsHtml(service, { results = new Map(), confirm = null } = {}) {
    const note = service.admission ? `<p class="settings-card-note" data-sign-in-admission>${admissionWords(service, undefined, true)}</p>` : '';
    return note + list(service.accounts).map((account) => {
        const view = accountState(service, account);
        const since = account.state === 'signed_in' && account.since ? html('since', 'signed in {date}', { date: shortDate(account.since) }) : '';
        const knownGrant = account.state === 'signed_in' || account.reason === 'insufficient_scope';
        const allowed = !knownGrant || !Array.isArray(account.scope) ? '' : account.scope.length
            ? html('allowed', 'Allowed: {scopes}', { scopes: scopeText(service, account.scope) })
            : html('allowed_none', 'Allowed: no permissions');
        const facts = [statusHtml(view.status, view.tone), [allowed, since].filter(Boolean).join(' · '),
            knownGrant && list(account.not_allowed).length ? html('not_allowed', 'Not allowed: {scopes}', { scopes: scopeText(service, account.not_allowed) }) : '',
            view.sentence ? `${view.sentence}${view.note ? ` ${view.note}` : ''}` : '',
            account.last_error?.reason === 'provider_unreachable' ? html('last_check', 'Last check: couldn’t reach {host}, {age}. The sign-in is kept.',
                { host: hostOf(service.issuer), age: relativeAge(account.last_error.at) })
                : account.last_error?.reason === 'provider_refused' ? html('last_refused', 'Last check: {host} refused the renewal ({error}), {age}. The sign-in is kept; ask the account’s administrator to allow this client, then sign in again.',
                    { host: hostOf(service.issuer), error: String(account.last_error.error || ''), age: relativeAge(account.last_error.at) }) : ''].filter(Boolean);
        const via = account.client?.label ? ` <span class="skills-card-facts">${html('via', 'via {client}', { client: account.client.label })}</span>` : '';
        const more = escapeHtml(words('more', 'More for {account}', { account: nameOf(account) }));
        return `<div class="ui-row settings-row" data-sign-in-row="${escapeHtml(account.key)}">
            <p class="ui-row-title">${authored(nameOf(account))}${via}</p>
            <p class="ui-row-meta">${facts.map((fact) => `<span>${fact}</span>`).join(' ')}</p>
            <div class="ui-row-actions">${view.needs && !service.admission ? act('begin', words('again', 'Sign in again'), { account: account.key }) : ''}<button type="button" class="btn btn-ghost btn-sm skills-card-menu-trigger" aria-haspopup="menu" aria-expanded="false" aria-label="${more}" data-sign-in-act="menu" data-sign-in-account="${escapeHtml(account.key)}">⋮</button></div>
            ${resultHtml(account.key, results.get(account.key), true)}
        </div>${confirm?.key === account.key ? confirmTray(service, account, confirm.what) : ''}`;
    }).join('') || `<p class="settings-card-note">${html('none', 'No account is signed in.')}</p>`;
}

/** The sign-in editor (account, client, permissions, the return note, Open) and, once begun, its waiting state with the paste. */
export function signInEditorHtml(service, editor, pageHost = '') {
    const values = { provider: providerOf(service) };
    if (editor.kind === 'waiting') {
        const link = safeExternalUrl(editor.url);
        return `<div class="skills-tray" data-sign-in-editor="waiting"><div class="settings-rows">
            <p class="skills-tray-text">${html('waiting', 'Waiting for you to finish at {provider}…', values)}${link === '#' ? '' : ` <a class="skills-link" href="${escapeHtml(link)}" target="_blank" rel="noopener noreferrer" data-sign-in-link>${html('open_link', 'Open the sign-in page')} ↗</a>`}</p>
            <p class="skills-tray-note">${html('if_error', 'If {provider} showed an error instead of coming back here, this client or a permission isn’t allowed for that account: choose another client or clear an optional permission, and start again.', values)}</p>
            ${service.notes?.refused ? `<p class="skills-tray-note">${authored(service.notes.refused)}</p>` : ''}
            <details class="ui-disclosure"${floorShown(pageHost, editor.clientReturn) ? ' open' : ''}><summary>${html('details', 'Details')}</summary>
                <label class="ui-field"><span>${html('paste', 'If your browser did not come back here, paste the address it stopped on')}</span><input class="ui-control" type="text" data-sign-in-field="returned" autocomplete="off" spellcheck="false"></label>
                <div class="ui-action-row">${act('complete', words('finish', 'Finish sign-in'))}</div></details>
            <div class="ui-action-row">${act('start_again', words('start_again', 'Start again'))}${act('close', words('cancel', 'Cancel'))}${resultHtml('editor', editor.result)}</div>
        </div></div>`;
    }
    const clients = signInClients(service);
    const draft = editor.draft || {};
    const canRegister = !clients.length && !service.endpoints;
    const pick = editor.picked ? {} : preselectedClient(service, { account: accountOf(service, editor.account), address: draft.hint });
    const name = escapeHtml(`sign-in-client-${service.skill}-${service.service}`);
    const radios = clients.map((client) => {
        const source = clientSource(client);
        return `<label class="ui-field ui-field-inline"><input class="ui-radio" type="radio" name="${name}" value="${escapeHtml(client.id)}"${
            client.id === draft.client ? ' checked' : ''}> <span>${clientLine(client)}<span data-sign-in-suggestion="${escapeHtml(client.id)}">${
                suggestion(pick, client.id)}</span>${source ? ` · ${source}` : ''}</span></label>`;
    }).join('');
    const optional = optionalChoices(service).map((row) => `<label class="ui-field ui-field-inline"><input class="ui-checkbox" type="checkbox" data-sign-in-optional value="${
        escapeHtml(row.scope)}"${list(draft.optional).includes(row.scope) ? ' checked' : ''}> ${authored(offerText(service, row.scope, row.label || row.scope))}</label>`).join('');
    const note = returnNoteHtml(service, signInClientOf(service, draft.client), pageHost, !canRegister);
    const oneAccount = Boolean(editor.account && service.accounts_mode === 'one');
    const open = clients.length ? act('open', words('open', 'Open the sign-in page'), { attrs: 'data-primary' })
        : canRegister ? act('open', words('register_open', 'Register a client and sign in'), { attrs: 'data-primary' }) : '';
    return `<div class="skills-tray" data-sign-in-editor="sign_in"><div class="settings-rows">
        <label class="ui-field"><span>${html('account', 'Account')}</span><input class="ui-control" type="text" data-sign-in-field="hint" value="${escapeHtml(draft.hint || '')}" placeholder="${escapeHtml(words('account_hint', 'Optional: the account to sign in with'))}" autocomplete="off" spellcheck="false"${oneAccount ? ' readonly' : ''}></label>
        <p class="ui-field-help">${oneAccount ? html('one_account', 'Disconnect this account here before switching to another.')
            : html('account_at_provider', 'Choose or confirm the account at {provider}.', values)}</p>
        <div class="ui-field" role="radiogroup" aria-label="${escapeHtml(words('client', 'Client'))}"><span>${html('client', 'Client')}</span>${radios || `<span class="ui-field-help">${list(service.clients).length
            ? html('no_return_client', 'No saved client has a known return address.')
            : html('no_clients', 'No client for {provider} is saved yet.', values)}</span>`}
            <div>${act('add_client', words('add_client', 'Add a client…'), { tone: 'ghost' })}</div></div>
        <div class="ui-field" data-sign-in-permissions><span>${html('permissions', 'Permissions')}</span><span class="ui-field-help">${html('always', 'Always: {scopes}', { scopes: list(service.scopes).map((scope) => offerText(service, scope)).join('; ') })}</span>${optional}</div>
        <p class="ui-field-help" data-sign-in-return-note${note ? '' : ' hidden'}>${note}</p>
        ${canRegister ? `<p class="ui-field-help">${html('register_asks', 'This asks {provider} ({issuer}) to register a client, if it offers that option.', { ...values, issuer: service.issuer })}</p>` : ''}
        <div class="ui-action-row">${open}${act('close', words('cancel', 'Cancel'))}${resultHtml('editor', editor.result)}</div>
    </div></div>`;
}

function bindingHtml(binding) {
    const endpoints = binding.endpoints;
    const row = (label, address) => `<div><dt>${label}</dt><dd><code data-i18n-authored>${escapeHtml(address)}</code></dd></div>`;
    const addresses = endpoints ? [
        row(html('authorization_address', 'Authorization'), endpoints.authorization),
        row(html('token_address', 'Token'), endpoints.token),
        row(html('identity_address', 'Account identity'), endpoints.identity),
        endpoints.revocation ? row(html('revocation_address', 'Revocation'), endpoints.revocation) : '',
    ].join('') : '';
    const issuer = row(endpoints ? html('provider_origin', 'Provider origin') : html('provider_issuer', 'Provider issuer'), binding.issuer);
    return `<div data-sign-in-binding><p class="ui-field-help">${endpoints
        ? html('client_approves_addresses', 'Saving approves this client for these exact addresses.')
        : html('client_approves_issuer', 'This client uses sign-in addresses published by this issuer.')}</p>
        <dl class="skills-sheet-facts">${issuer}${addresses}</dl></div>`;
}

/** One client form (Add, or Edit of a saved row): the secret is a password field that starts empty and is never filled back. */
export function clientFormHtml(service, form = {}, returnAddress = '') {
    const unknown = form.return === 'unknown', exact = String(form.return || '').startsWith('exact:');
    const name = escapeHtml(`sign-in-return-${service.skill}-${service.service}`);
    const input = (key, type, value = '', extra = '') => `<input class="ui-control" type="${type}" data-client-field="${key}" value="${escapeHtml(value)}" autocomplete="${
        type === 'password' ? 'new-password' : 'off'}" spellcheck="false"${extra}>`;
    const field = (label, control, help = '') => `<label class="ui-field"><span>${label}</span>${control}${help ? `<span class="ui-field-help">${help}</span>` : ''}</label>`;
    const radio = (value, checked, label) => `<label class="ui-field ui-field-inline"><input class="ui-radio" type="radio" name="${name}" value="${value}"${checked ? ' checked' : ''}> ${label}</label>`;
    const source = clientSource(form);
    const credentials = form.id && form.source && form.source !== 'owner'
        ? `<p class="ui-field-help">${html('other_credentials', 'Add a separate client to use a different client ID or secret.')}</p>`
        : field(html('client_id', 'Client ID'), input('client_id', 'text', '', form.id ? ` placeholder="…${escapeHtml(form.tail || '')}"` : ''),
            form.id ? html('client_id_kept', 'Blank keeps the saved client ID.') : '')
            + field(html('client_secret', 'Client secret (optional)'), input('client_secret', 'password'),
                form.secret_set ? html('secret_kept', 'Blank keeps the saved secret.') : '');
    return `<div class="settings-rows" data-sign-in-client-form="${escapeHtml(form.id || '')}">
        ${bindingHtml(form.binding || bindingOf(service))}
        ${field(html('client_label', 'Label'), input('label', 'text', form.label || '', ` placeholder="${escapeHtml(words('label_example', 'Personal'))}"`),
            source || html('client_label_help', 'Use a name you will recognize, such as Personal or Work (IT-supplied).'))}
        ${credentials}
        ${unknown
            ? `<p class="ui-field-help">${html('unknown_return_help', 'Return address unknown. Add a client with a known return address for new sign-ins.')}</p>`
            : `<div class="ui-field" role="radiogroup" aria-label="${escapeHtml(words('returns', 'Returns to'))}"><span>${html('returns', 'Returns to')}</span>
                ${radio('loopback', !exact, html('return_here', 'This computer (desktop-app client)'))}${radio('exact', exact, html('return_web', 'This address (web-app client)'))}
                ${input('address', 'url', exact ? String(form.return).slice(6) : '', ' placeholder="https://…"')}</div>
            <p class="ui-field-help">${html('register', 'Register exactly this return address:')} <code data-sign-in-register>${escapeHtml(exact ? String(form.return).slice(6) : returnAddress)}</code> ${act('copy', words('copy', 'Copy'), { tone: 'ghost' })}</p>`}
        <div class="ui-action-row">${act('client_save', words('save', 'Save'), { tone: 'save' })}${act('client_cancel', words('cancel', 'Cancel'))}${resultHtml('form', form.result)}</div>
    </div>`;
}

/** The clients editor: the issuer's clients (label, ID tail, return, who uses it) with Edit and Remove, Add, and the skill's note. */
export function clientsEditorHtml(service, editor, returnAddress = '') {
    const values = { provider: providerOf(service) };
    const rows = list(service.clients).map((client) => {
        const used = list(client.used_by).map((row) => `${row.skill} (${row.account})`).join(', ');
        return `<div class="ui-row settings-row" data-sign-in-client="${escapeHtml(client.id)}">
            <p class="ui-row-title">${clientLine(client)}</p>
            <p class="ui-row-meta"><span>${clientSource(client)}</span><span>${used ? html('used_by', 'Used by: {used}', { used }) : html('unused', 'Not used yet')}</span></p>
            <div class="ui-row-actions">${act('client_edit', words('edit', 'Edit…'), { client: client.id })}${act('client_remove', words('remove', 'Remove…'), { client: client.id })}</div>
        </div>${editor.confirm === client.id ? trayHtml({ attrs: `data-sign-in-confirm="${escapeHtml(client.id)}"`,
            sentence: html('remove_q', 'Remove the client {client}? Existing sign-ins keep working; new sign-ins can’t choose it.', { client: client.label || client.id }),
            actions: act('confirm_client_remove', words('remove_client', 'Remove'), { client: client.id, tone: 'danger' }) + act('cancel_confirm', words('cancel', 'Cancel'), { client: client.id }) }) : ''}`;
    }).join('');
    return `<div class="skills-tray" data-sign-in-editor="clients"><div class="settings-rows">
        <p class="skills-tray-text">${html('clients_of', 'Your clients for {provider}', values)}</p>
        ${rows || `<p class="ui-field-help">${html('no_clients', 'No client for {provider} is saved yet.', values)}</p>`}
        ${editor.form ? clientFormHtml(service, editor.form, returnAddress)
        : `<div class="ui-action-row">${act('client_new', words('add', 'Add a client'))}${act('clients_done', words('done', 'Done'))}${resultHtml('clients', editor.result)}</div>`}
        ${service.notes?.client ? `<p class="skills-tray-note">${authored(service.notes.client)}</p>` : ''}
    </div></div>`;
}

function initialDraft(service, key) {
    const account = list(service.accounts).find((row) => row.key === key) || null;
    const last = new Set([...list(account?.scope), ...list(account?.not_allowed), ...scopeWords(account?.required_scope)]);
    return { hint: String(account?.label || ''), client: preselectedClient(service, { account }).client?.id || '',
        optional: optionalChoices(service).filter((row) => account
            ? scopeWords(row.scope).every((word) => wordSpellings(service, word).some((name) => last.has(name)))
            : row.default === true).map((row) => row.scope) };
}

// Keep only the choices still offered. Never substitute another saved client for a removed one.
const keepDraft = (draft, service) => {
    const allowed = new Set(optionalChoices(service).map((row) => row.scope));
    return { ...draft, client: signInClientOf(service, draft.client) ? draft.client : '',
        optional: list(draft.optional).filter((scope) => allowed.has(scope)) };
};

const draftOf = (group) => {
    const host = group.querySelector('[data-sign-in-editor="sign_in"]');
    return host && { hint: String(host.querySelector('[data-sign-in-field="hint"]')?.value || '').trim(),
        client: host.querySelector('input[type="radio"]:checked')?.value || '',
        optional: [...host.querySelectorAll('[data-sign-in-optional]:checked')].map((node) => node.value) };
};

/**
 * Details › Sign-ins on the skill's sheet. `status` is the last status read (null reads it now); `reload()` reads it
 * again and answers it (the page repaints its cards from the same answer); `open` is the control that opened the
 * sheet, whose `data-sign-in-*` name a service and, with `data-sign-in-begin`, an account to sign in. Returns the disposer.
 */
export function mountSkillSignIn(container, skillName, { status = null, reload = readSignInStatus, open = null } = {}) {
    const doc = container.ownerDocument, win = doc.defaultView;
    const states = new Map(), written = new WeakMap();
    const lifetime = new AbortController();
    let current = status, closed = false, menu = null, disarm = null;
    const isWaiting = () => [...states.values()].some((state) => state.editor?.kind === 'waiting');
    let ask = open?.dataset?.signInService
        ? { service: open.dataset.signInService, account: open.dataset.signInAccount || '', begin: open.dataset.signInBegin !== undefined } : null;
    const services = () => servicesOf(current, skillName);
    const stateOf = (id) => states.get(id) || states.set(id, { editor: null, confirm: null, results: new Map(), dirty: false, busy: false }).get(id);
    const groupOf = (id) => [...container.querySelectorAll('[data-sign-in-group]')].find((node) => node.dataset.signInGroup === id) || null;
    const pageHost = () => win?.location?.hostname || '';
    const focusAct = (group, action, account, client) => {
        const node = [...(group?.querySelectorAll(`[data-sign-in-act="${action}"]`) || [])]
            .find((item) => (account === undefined || item.dataset.signInAccount === account) && (!client || item.dataset.client === client));
        node?.focus({ preventScroll: true });
        return node || null;
    };

    // A part is rewritten only when its words changed; a focused control keeps focus, an open menu on it closes.
    function region(group, part, markup) {
        const host = group.querySelector(`[data-sign-in-part="${part}"]`);
        if (written.get(host) === markup) return;
        const key = (node) => [node.dataset.signInAct, node.dataset.signInAccount, node.dataset.client].join('|');
        const focused = host.contains(doc.activeElement) && doc.activeElement.dataset?.signInAct ? key(doc.activeElement) : '';
        if (menu && host.contains(menu.trigger)) menu.binding.close();
        host.innerHTML = markup;
        host.hidden = !markup;
        written.set(host, markup);
        if (focused) [...host.querySelectorAll('[data-sign-in-act]')].find((node) => key(node) === focused)?.focus({ preventScroll: true });
    }

    // The head and rows follow every status read; an editor is rewritten only by its own steps, so typing survives a read.
    function paint() {
        if (closed) return;
        const rows = services();
        container.hidden = !rows.length;
        if (!container.querySelector('[data-sign-in-groups]')) container.innerHTML = '<div class="settings-rows" data-sign-in-groups></div>';
        const groups = container.querySelector('[data-sign-in-groups]');
        const left = new Set(groups.querySelectorAll(':scope > [data-sign-in-group]'));
        for (const service of rows) {
            let group = groupOf(service.service);
            if (!group) {
                group = Object.assign(doc.createElement('div'), { className: 'settings-rows' });
                group.dataset.signInGroup = service.service;
                group.innerHTML = '<div class="settings-card-head" data-sign-in-part="head"></div><div data-sign-in-part="editor" hidden></div><div class="settings-rows" data-sign-in-part="rows"></div>';
                groups.appendChild(group);
            }
            left.delete(group);
            const state = stateOf(service.service);
            const one = service.accounts_mode === 'one';
            region(group, 'head', `<h3 class="settings-card-title">${html('title', 'Sign-ins · {provider}', { provider: providerOf(service) })}</h3>${resultHtml('', state.results.get(''))}${
                service.admission || (one && list(service.accounts).length) ? '' : act('begin', one ? words('sign_in', 'Sign in') : words('add_account', 'Add account'), { account: '' })}${act('clients', words('clients', 'Clients…'), { tone: 'ghost' })}`);
            region(group, 'rows', signInRowsHtml(service, state));
            if (state.dirty) {
                state.dirty = false;
                region(group, 'editor', !state.editor ? '' : state.editor.kind === 'clients' ? clientsEditorHtml(service, state.editor, current?.return_address)
                    : signInEditorHtml(state.editor.shown || service, state.editor, pageHost()));
            }
        }
        left.forEach((node) => node.remove());
    }

    // Only a confirmed account bearing this attempt's ID can settle it via status. A neighbouring
    // account or a missing pending row is not a receipt (the account and pending reads may straddle filing).
    function settle() {
        for (const service of services()) {
            const state = stateOf(service.service), editor = state.editor;
            if (editor?.kind !== 'waiting' || !editor.attemptId) continue;
            const done = list(service.accounts).find((row) => row.verified === true && attemptOf(row) === editor.attemptId);
            if (!done) continue;
            Object.assign(state, { editor: null, dirty: true });
            state.results.set(done.key, signedIn(service, editor, done));
        }
    }

    function signedIn(service, editor, account) {
        if (account.state !== 'signed_in' || accountState(service, account).needs) {
            return { text: words('confirmed_limited', '{account} was confirmed, but this skill still needs access. See the account row.',
                { account: nameOf(account) }), tone: 'warn' };
        }
        if (editor.account && account.key !== editor.account) {
            const asked = nameOf(list(service.accounts).find((row) => row.key === editor.account)) || editor.draft?.hint || '';
            return { text: words('other_account', 'Signed in as {account}, not {asked}: saved as its own account.', { account: nameOf(account), asked }), tone: 'warn' };
        }
        return { text: words('signed_in_as', 'Signed in as {account} via {client}.', { account: nameOf(account), client: account.client?.label || editor.client?.label || '' }), tone: 'ok' };
    }

    function update(data) {
        if (closed || !data) return;
        current = data;
        settle();
        paint();
        if ([...states.values()].some((state) => state.editor?.kind === 'waiting')) armReturn();
        else disarm?.();
        const service = ask && services().find((row) => row.service === ask.service);
        if (service) {
            const wanted = ask;
            ask = null;
            if (wanted.begin) openEditor(service, wanted.account, false);
            else groupOf(service.service)?.scrollIntoView?.({ block: 'nearest' });
        }
    }

    async function refresh(checkReturn = false) {
        const watched = checkReturn ? new Set([...states.values()].map((state) => state.editor)) : null;
        try {
            const data = await reload({ signal: lifetime.signal });
            if (!data || closed) return;
            if (data !== current) update(data);
        } catch { return; } // A failed read changes no displayed account fact.
        if (!watched) return;
        for (const service of services()) {
            const state = stateOf(service.service), editor = state.editor;
            if (editor?.kind !== 'waiting' || !watched.has(editor) || state.busy || (editor.result && !editor.result.observation)) continue;
            const pending = attemptOf(service.pending);
            if (pending === editor.attemptId) {
                if (editor.result?.observation) say(service, 'editor', { observation: true });
            } else {
                const text = pending ? words('other_pending', 'Another sign-in is open. This window has no result for yours.')
                    : words('return_not_seen', 'No result for this sign-in yet. Check the provider page or start again.');
                say(service, 'editor', { text, tone: 'warn', observation: true });
            }
        }
    }

    // After the provider page opened, coming back to this window reads the status again (no timer, no polling).
    function armReturn() {
        if (closed || disarm || !win) return;
        const back = () => {
            if (doc.visibilityState === 'hidden') return;
            disarm?.();
            void refresh(true).finally(() => { if (isWaiting()) armReturn(); });
        };
        win.addEventListener('focus', back);
        doc.addEventListener('visibilitychange', back);
        disarm = () => { win.removeEventListener('focus', back); doc.removeEventListener('visibilitychange', back); disarm = null; };
    }

    // An editor's own line: the words go on screen now and stay with the editor for its next rewrite.
    function say(service, key, result) {
        const editor = stateOf(service.service).editor;
        if (editor) (key === 'form' && editor.form ? editor.form : editor).result = result;
        setResult(groupOf(service.service)?.querySelector(`[data-sign-in-result="${key}"]`), result);
    }

    function openEditor(service, account, focus) {
        const state = stateOf(service.service);
        Object.assign(state, { editor: { kind: 'sign_in', shown: service, account, draft: initialDraft(service, account) }, confirm: null, dirty: true });
        paint();
        const host = groupOf(service.service)?.querySelector('[data-sign-in-part="editor"]');
        if (focus) revealNewRow(host, host?.querySelector('[data-sign-in-field="hint"]'));
        else host?.scrollIntoView?.({ block: 'nearest' });
    }

    async function busy(service, work) {
        const state = stateOf(service.service);
        if (state.busy) return;
        state.busy = true;
        try { await work(state); } finally { state.busy = false; }
    }

    const choicesChanged = () => ({ text: words('choices_changed', 'The sign-in choices changed. Review them before opening the sign-in page.'), tone: 'warn' });
    function showCurrentChoices(state, editor, service, group) {
        editor.shown = service;
        editor.draft = keepDraft(editor.draft, service);
        editor.picked = Boolean(editor.picked && editor.draft.client);
        editor.result = choicesChanged();
        state.dirty = true;
        paint();
        if (!focusAct(group, 'open')) focusAct(group, 'add_client');
    }

    const beginSignIn = (service, group) => busy(service, async (state) => {
        const editor = state.editor, shown = editor.shown;
        const draft = editor.draft = draftOf(group) || editor.draft;
        if (choicesKey(shown) !== choicesKey(service)) { showCurrentChoices(state, editor, service, group); return; }
        const client = signInClientOf(shown, draft.client);
        const register = !signInClients(shown).length && !shown.endpoints;
        if (!client && !register) { say(service, 'editor', { text: failureWords(service, { reason: 'client_required' }), tone: 'warn' }); return; }
        const row = accountOf(shown, editor.account);
        const account = row && (shown.accounts_mode === 'one' || hintKey(row.label) === hintKey(draft.hint)) ? row.key : '';
        say(service, 'editor', { text: words('contacting', 'Contacting {provider}…', { provider: providerOf(service) }), tone: 'muted' });
        const requested = [...list(shown.scopes), ...draft.optional];
        const scope_words = [...new Set(requested.flatMap((scope) => offerWords(shown, scope)))];
        const result = await step({ action: 'begin', skill: skillName, service: service.service, optional_scopes: draft.optional, scope_words, login_hint: draft.hint,
            ...(client ? { client: client.id, registration: client.registration } : { register_client: true, issuer: shown.issuer }), ...(account ? { account } : {}) });
        if (closed || state.editor !== editor) return;
        if (!result.ok) {
            const changed = result.reason === 'sign_in_choices_changed';
            say(service, 'editor', changed ? choicesChanged() : { text: failureWords(service, result,
                { client: client?.label, returnAddress: current?.return_address }), tone: ['provider_approval_required', 'registration_unconfirmed'].includes(result.reason) ? 'warn' : 'danger' });
            if (changed || ['client_required', 'unknown_client'].includes(result.reason)) {
                await refresh();
                const fresh = services().find((entry) => entry.service === service.service);
                if (!closed && state.editor === editor && fresh && choicesKey(shown) !== choicesKey(fresh)) showCurrentChoices(state, editor, fresh, group);
            }
            return;
        }
        const attemptId = attemptOf(result), link = safeExternalUrl(result.url);
        const selectedScopes = new Set(scopeWords(requested.join(' ')));
        const openedScopes = new Set(scopeWords(result.scope));
        const clientReturn = client?.return || 'loopback';
        if (!attemptId || !/^https?:/.test(link) || !result.client?.id || typeof result.scope !== 'string') {
            say(service, 'editor', { text: failureWords(service, { reason: 'unconfirmed' }), tone: 'warn' });
            return;
        }
        if ((client && (result.client.id !== client.id || result.client.label !== client.label))
                || (clientReturn.startsWith('exact:') && result.redirect_uri !== clientReturn.slice(6))
                || selectedScopes.size !== openedScopes.size || [...selectedScopes].some((scope) => !openedScopes.has(scope))) {
            say(service, 'editor', choicesChanged());
            await refresh();
            const fresh = services().find((entry) => entry.service === service.service);
            if (!closed && state.editor === editor && fresh) showCurrentChoices(state, editor, fresh, group);
            return;
        }
        Object.assign(state, { editor: { ...editor, account, kind: 'waiting', attemptId, url: link, client: result.client, clientReturn, result: null }, dirty: true });
        paint();
        group.querySelector('a[data-sign-in-link]')?.focus({ preventScroll: true });
        void openExternalViaHostBridge(link, { from: group }).catch(() => {});  // best effort while the press is recent; the link works on every surface
        armReturn();
        void refresh();
    });

    const finish = (service, group) => busy(service, async (state) => {
        const editor = state.editor;
        const returned = String(group.querySelector('[data-sign-in-field="returned"]')?.value || '').trim();
        if (!returned) { say(service, 'editor', { text: words('paste_first', 'Paste the address your browser stopped on first.'), tone: 'warn' }); return; }
        say(service, 'editor', { text: words('finishing', 'Finishing the sign-in…'), tone: 'muted' });
        const result = await step({ action: 'complete', returned_url: returned, attempt_id: editor.attemptId });
        if (closed || state.editor !== editor) return;
        if (!result.ok) {
            const unsure = noAnswer(result);
            say(service, 'editor', { text: failureWords(service, result, { client: editor.client?.label, complete: true }),
                tone: unsure || result.reason === 'provider_approval_required' ? 'warn' : 'danger' });
            if (unsure) await refresh();  // one status read by this attempt: a filing of it settles the editor (`settle`); an old row of the account proves nothing
            return;
        }
        if (attemptOf(result) !== editor.attemptId || result.skill !== skillName || result.service !== service.service
                || !result.account?.key || !result.account?.label || !['signed_in', 'sign_in_required'].includes(result.state)) {
            say(service, 'editor', { text: failureWords(service, { reason: 'unconfirmed' }, { complete: true }), tone: 'warn' });
            await refresh();
            return;
        }
        const account = { ...result.account, state: result.state, reason: result.reason, missing_required: result.missing_required };
        Object.assign(state, { editor: null, dirty: true });
        state.results.set(account.key, signedIn(service, editor, account));
        paint();
        await refresh();
    });

    const leaveWaiting = (service, group, again) => busy(service, async (state) => {
        const editor = state.editor;
        const result = await step({ action: 'cancel', skill: skillName, service: service.service, attempt_id: editor.attemptId });
        if (closed || state.editor !== editor) return;
        const superseded = !result.ok && result.reason === 'sign_in_superseded';
        if (!superseded && (!result.ok || attemptOf(result) !== editor.attemptId || typeof result.cancelled !== 'boolean')) {
            say(service, 'editor', { text: failureWords(service, result.ok ? { reason: 'unconfirmed' } : result), tone: 'warn' });
            return;
        }
        await refresh();
        if (closed || state.editor !== editor) return;
        const note = result.cancelled === true ? { text: words('cancelled', 'Sign-in cancelled here.'), tone: 'ok' }
            : { text: words('no_longer_pending', 'That sign-in is no longer waiting here.'), tone: 'muted' };
        const fresh = services().find((entry) => entry.service === service.service) || service;
        const draft = keepDraft(editor.draft, fresh);
        state.editor = again ? { kind: 'sign_in', shown: fresh, account: editor.account, draft, picked: Boolean(editor.picked && draft.client), result: note } : null;
        if (!again) state.results.set('', note);
        state.dirty = true;
        if (!isWaiting()) disarm?.();
        paint();
        if (again) { if (!focusAct(group, 'open')) focusAct(group, 'add_client'); }
        else if (!focusAct(group, 'begin', editor.account)) focusAct(group, editor.account ? 'menu' : 'clients', editor.account || undefined);
    });

    // `consent` is the one the confirmation was opened on (the account as the owner saw it), never today's read of the row.
    const forget = (service, group, key, what, consent = '') => busy(service, async (state) => {
        state.confirm = null;
        const result = await step({ action: what, skill: skillName, service: service.service, account: key, consent });
        if (closed) return;
        if (!result.ok) {
            // A Revoke that was not confirmed or not asked, and a changed consent, leave the sign-in here; no way to repeat it blindly is offered.
            const kept = ['revocation_not_confirmed', 'revocation_not_asked', 'consent_changed'].includes(result.reason);
            state.results.set(key, { text: failureWords(service, result, { account: nameOf(accountOf(service, key)) }), tone: kept ? 'warn' : 'danger' });
            if (result.reason === 'consent_changed') await refresh();  // the row shows the sign-in as it is now
        } else {
            state.results.delete(key);
            const names = { provider: providerOf(service) };
            state.results.set('', what !== 'revoke' ? { text: words('disconnected', 'Disconnected here. To remove the provider permission too, open your {provider} account settings.', names), tone: 'ok' }
                : result.revoked === true ? { text: words('revoked', '{provider} confirmed the revocation.', names), tone: 'ok' }
                : { text: words('revoke_unknown', 'The reply did not say whether {provider} ended this sign-in. Check in your {provider} account; Disconnect here forgets it on this side only.', names), tone: 'warn' });
        }
        paint();
        if (result.ok) await refresh();  // the row leaves with the next status
        focusAct(group, result.ok ? 'begin' : 'menu', result.ok ? undefined : key);
    });

    const saveClient = (service, group) => busy(service, async (state) => {
        const editor = state.editor, form = group.querySelector('[data-sign-in-client-form]');
        const value = (key) => String(form?.querySelector(`[data-client-field="${key}"]`)?.value || '').trim();
        const exact = Boolean(form?.querySelector('input[type="radio"][value="exact"]')?.checked);
        const client = { ...(editor.form.id ? { id: editor.form.id } : {}), label: value('label'), client_id: value('client_id'),
            ...(editor.form.return === 'unknown' ? {} : { return: exact ? `exact:${value('address')}` : 'loopback' }) };
        Object.assign(editor.form, { label: client.label, return: client.return ?? editor.form.return });
        if (!client.label || (!client.client_id && !editor.form.id)) { say(service, 'form', { text: words('client_needs', 'Give the client a label and its client ID.'), tone: 'warn' }); return; }
        const binding = editor.form.binding;
        if (bindingKey(binding) !== bindingKey(service)) {
            say(service, 'form', { text: words('addresses_changed', 'The provider addresses changed. Close and reopen this form to review the new addresses.'), tone: 'warn' });
            return;
        }
        const secret = form.querySelector('[data-client-field="client_secret"]');
        const result = await step({ action: 'client_save', ...binding, client: { ...client, ...(secret?.value ? { client_secret: secret.value } : {}) } });
        if (closed || state.editor !== editor) return;
        if (!result.ok) { say(service, 'form', { text: failureWords(service, result), tone: 'danger' }); return; }
        if (secret) secret.value = '';  // sent from the field, never kept or shown back
        const saved = { text: words('client_saved', 'Saved {client}.', { client: result.client?.label || client.label }), tone: 'ok' };
        await refresh();
        if (closed || state.editor !== editor) return;
        const fresh = services().find((row) => row.service === service.service);
        const available = fresh && bindingKey(fresh) === bindingKey(binding) && signInClientOf(fresh, result.client?.id);
        state.editor = editor.back && available ? { ...editor.back, shown: fresh,
            draft: keepDraft({ ...editor.back.draft, client: available.id }, fresh), picked: true, result: saved }
            : { ...editor, form: null, result: saved };
        state.dirty = true;
        paint();
        focusAct(group, state.editor.kind === 'sign_in' ? 'open' : 'client_new');
    });

    const removeClient = (service, group, id) => busy(service, async (state) => {
        const editor = state.editor, label = list(service.clients).find((row) => row.id === id)?.label || id;
        editor.confirm = null;
        const result = await step({ action: 'client_remove', client: id });
        if (closed || state.editor !== editor) return;
        editor.result = !result.ok ? { text: failureWords(service, result), tone: 'danger' }
            : result.removed === true ? { text: words('client_removed', 'Removed {client}; existing sign-ins keep working.', { client: label }), tone: 'ok' }
            : result.removed === false ? { text: words('client_gone', '{client} was already gone, so this removed nothing. Existing sign-ins keep working.', { client: label }), tone: 'muted' }
            : { text: failureWords(service, { reason: 'unconfirmed' }), tone: 'warn' };
        state.dirty = true;
        await refresh();
        paint();
        focusAct(group, 'client_new');
    });

    async function copyAddress(service, group, node) {
        const text = group.querySelector('[data-sign-in-register]')?.textContent || '';
        try {
            await copyTextToClipboard(text, { win, doc, anchor: node });
            say(service, 'form', { text: words('copied', 'Copied.'), tone: 'ok' });
        } catch {
            say(service, 'form', { text: words('copy_failed', 'Couldn’t copy; select the address and copy it.'), tone: 'warn' });
        }
    }

    function openMenu(trigger, service, group) {
        if (menu?.trigger === trigger) { menu.binding.close({ restoreFocus: true }); return; }
        menu?.binding.close();
        const key = trigger.dataset.signInAccount;
        const popup = Object.assign(doc.createElement('div'), { className: 'ui-popup skills-card-menu-dialog' });
        popup.setAttribute('role', 'menu');
        popup.innerHTML = [['begin', words('again_menu', 'Sign in again…')], ['disconnect', words('disconnect_menu', 'Disconnect here…')],
            ['revoke', words('revoke_menu', 'Revoke at {provider}…', { provider: providerOf(service) })]].filter(([action]) => action !== 'begin' || !service.admission).map(([action, label]) =>
            `<button type="button" role="menuitem" class="skills-menu-item${action === 'revoke' ? ' danger' : ''}" data-sign-in-act="${action}" data-sign-in-account="${escapeHtml(key)}">${escapeHtml(label)}</button>`).join('');
        // Outside the scrolling body (its fade would cut the menu) but inside the sheet, whose modal makes the page inert.
        (container.closest('dialog') || doc.body).appendChild(popup);
        popup.addEventListener('click', onClick);
        trigger.setAttribute('aria-expanded', 'true');
        const binding = bindMenu(popup, { anchor: trigger, onClose: () => {
            popup.remove();
            trigger.setAttribute('aria-expanded', 'false');
            if (menu?.popup === popup) menu = null;
        } });
        menu = { popup, binding, trigger, group };
    }

    async function run(action, service, group, node) {
        const state = stateOf(service.service), editor = state.editor;
        const account = node.dataset.signInAccount, client = node.dataset.client;
        const rewrite = (patch, focus) => {
            Object.assign(state, patch, { dirty: true });
            paint();
            if (focus) revealNewRow(group.querySelector('[data-sign-in-part="editor"]'), group.querySelector(focus));
        };
        if (state.busy) return;
        if (editor?.kind === 'waiting' && (action === 'begin' || action === 'clients')) {
            say(service, 'editor', { text: words('waiting_change', 'Finish this sign-in, or choose Start again to change it.'), tone: 'warn' });
        } else if (action === 'menu') openMenu(node, service, group);
        else if (action === 'begin') openEditor(service, account || '', true);
        else if (action === 'close' && editor?.kind === 'waiting') await leaveWaiting(service, group, false);
        else if (action === 'close') {
            const from = editor?.account || '';
            rewrite({ editor: null });
            if (!focusAct(group, 'begin', from)) focusAct(group, from ? 'menu' : 'clients', from || undefined);
        }
        else if (action === 'start_again') await leaveWaiting(service, group, true);
        else if (action === 'open') await beginSignIn(service, group);
        else if (action === 'complete') await finish(service, group);
        else if (action === 'disconnect' || action === 'revoke') { state.confirm = { what: action, key: account, consent: accountOf(service, account)?.consent || '' }; paint(); focusAct(group, `confirm_${action}`, account); }
        else if (action === 'cancel_confirm' && client) { editor.confirm = null; rewrite({}); focusAct(group, 'client_remove', undefined, client); }
        else if (action === 'cancel_confirm') { state.confirm = null; paint(); focusAct(group, 'menu', account); }
        else if (action === 'confirm_disconnect' || action === 'confirm_revoke') await forget(service, group, account, action.slice('confirm_'.length), state.confirm?.consent);
        else if (action === 'clients') rewrite({ editor: { kind: 'clients', form: null, back: null } }, '[data-sign-in-act="client_new"]');
        else if (action === 'add_client') rewrite({ editor: { kind: 'clients', form: { binding: bindingOf(service) },
            back: { ...editor, draft: draftOf(group) || editor.draft } } }, '[data-client-field="label"]');
        else if (action === 'client_new') rewrite({ editor: { ...editor, form: { binding: bindingOf(service) }, result: null } }, '[data-client-field="label"]');
        else if (action === 'client_edit') {
            const row = clientOf(service, client);
            if (row) rewrite({ editor: { ...editor, form: { id: row.id, label: row.label, return: row.return, tail: row.client_id_tail,
                source: row.source, secret_set: row.secret_set, binding: bindingOf(service) } } }, '[data-client-field="label"]');
        } else if (action === 'client_cancel' && editor.back) rewrite({ editor: editor.back }, '[data-sign-in-act="add_client"]');
        else if (action === 'client_cancel') rewrite({ editor: { ...editor, form: null } }, '[data-sign-in-act="client_new"]');
        else if (action === 'client_save') await saveClient(service, group);
        else if (action === 'client_remove') { editor.confirm = client; rewrite({}); focusAct(group, 'confirm_client_remove', undefined, client); }
        else if (action === 'confirm_client_remove') await removeClient(service, group, client);
        else if (action === 'clients_done') {
            rewrite({ editor: editor.back || null }, editor.back ? '[data-sign-in-act="open"]' : '');
            if (!editor.back) focusAct(group, 'clients');
        }
        else if (action === 'copy') await copyAddress(service, group, node);
    }

    async function onClick(event) {
        const link = event.target.closest?.('a[data-sign-in-link]');
        if (link) { event.preventDefault(); void openExternalViaHostBridge(link.getAttribute('href'), { from: link }).catch(() => {}); return; }
        const node = event.target.closest?.('[data-sign-in-act]');
        if (!node || node.disabled) return;
        const fromMenu = Boolean(menu?.popup.contains(node));
        const group = fromMenu ? menu.group : node.closest('[data-sign-in-group]');
        const service = group && services().find((row) => row.service === group.dataset.signInGroup);
        if (fromMenu) menu.binding.close({ restoreFocus: true });
        if (service) await run(node.dataset.signInAct, service, group, node);
    }

    // The client choice, the login hint and the return rule update their notes in place (nothing is rewritten).
    function onEdit(event) {
        const group = event.target.closest('[data-sign-in-group]');
        const service = group && services().find((row) => row.service === group.dataset.signInGroup);
        const editor = service && stateOf(service.service).editor;
        if (!editor) return;
        const draft = draftOf(group);
        if (editor.kind === 'sign_in' && draft) {
            const shown = editor.shown;
            const radios = [...group.querySelectorAll('[data-sign-in-editor="sign_in"] input[type="radio"]')];
            if (event.type === 'change' && radios.includes(event.target)) editor.picked = true;
            const pick = editor.picked ? {} : preselectedClient(shown, { account: accountOf(shown, editor.account), address: draft.hint });
            if (event.target.matches('[data-sign-in-field="hint"]') && !editor.picked) {
                for (const box of radios) box.checked = box.value === pick.client?.id;
            }
            for (const line of group.querySelectorAll('[data-sign-in-suggestion]')) {
                const text = suggestion(pick, line.dataset.signInSuggestion);
                if (line.innerHTML !== text) line.innerHTML = text;
            }
            const note = group.querySelector('[data-sign-in-return-note]');
            if (note) {
                note.innerHTML = returnNoteHtml(shown, clientOf(shown, radios.find((node) => node.checked)?.value), pageHost(), radios.length > 0 || Boolean(shown.endpoints));
                note.hidden = !note.innerHTML;
            }
        }
        const form = group.querySelector('[data-sign-in-client-form]'), code = form?.querySelector('[data-sign-in-register]');
        if (code) {
            code.textContent = form.querySelector('input[type="radio"][value="exact"]')?.checked
                ? String(form.querySelector('[data-client-field="address"]')?.value || '').trim() : String(current?.return_address || '');
        }
    }

    const onStatus = (event) => update(event.detail);
    win?.addEventListener(SIGN_IN_STATUS_EVENT, onStatus);
    container.addEventListener('click', onClick);
    container.addEventListener('change', onEdit);
    container.addEventListener('input', onEdit);
    if (current) update(current); else void refresh();
    return () => {
        if (closed) return;
        closed = true;
        menu?.binding.close();
        disarm?.();
        lifetime.abort();
        win?.removeEventListener(SIGN_IN_STATUS_EVENT, onStatus);
        container.removeEventListener('click', onClick);
        container.removeEventListener('change', onEdit);
        container.removeEventListener('input', onEdit);
    };
}
