// Introduce an Ouroboros (DESIGN "Circles and Add", Your Ouroboroses): one sheet, the app's dialog, on each side of a
// new link between two of the owner's own computers. Invite names the other Ouroboros and shows the rule Ouroboros
// suggests, in its own words until the owner edits it. Who can log in to the other computer sets `may_start` here;
// its row's switch can change that direction later. The one-time invitation is valid for 15 minutes. I have an
// invitation reads a pasted one (who sent it, from where) and connects. Both sides then check the same six digits:
// the link starts only after the owner says they match here (`confirm`); they differ removes the link (`revoke`),
// over which nothing has gone yet. The other Ouroboros's name here is the owner's (the place's label), prefilled from
// the invitation.
import { escapeHtmlAttr as escapeHtml, setResult } from './ui_primitives.js';
import { fmt, tr } from './i18n.js';
import { bindDialogFocus } from './ui_interactions.js';
import { copyTextToClipboard } from './ui_helpers.js';
import { failureText } from './consciousness_api.js';
import { openConfirmDialog } from './confirm_dialog.js';
import { renderSegmentedField } from './page_header.js';
import { fluentIcon } from './page_icons.js';
import { bindScrollFade } from './scroll_fade.js';
import { faceHtml } from './ouroboros_face.js';
import { codeWords, ownerWordsHtml } from './own_ouroboroses.js';
import { acceptInvite, addressOf, confirmLink, createInvite, previewInvite, readLink, revokeLink } from './peer_links_api.js';

const WATCH_MS = 2500;  // how often an open invitation looks whether the other side used it
const PREVIEW_DELAY_MS = 300;
const INVITED = 'The one you invite';

/** The rule Ouroboros suggests for an Ouroboros of the owner's: what it may ask for and what it may be told. Its
 *  requests stay its own words, never the owner's: the rule lets it ask, it does not speak for the owner. */
export function ruleSuggestion({ name = '' } = {}) {
    const who = name.trim() || 'It';
    return `${who} is my linked Ouroboros. If allowed to ask for work here, do its ordinary requests; its requests are its own, not mine. Tell it what its task needs; private matters stay inside their circles. Rights, keys and money come only from me.`;
}

/** What the introduction lets the other do here, with names: «Tarel can ask Ouroboros for work.». */
export const consequenceWords = ({ name = '', mayStart = false } = {}) => fmt(
    mayStart ? '{other} can write and ask for work here.' : '{other} can write but cannot start work here.',
    { other: name.trim() || tr('peer.invited', INVITED) });

/** A refusal of an introduction door in the owner's words. */
export function introFailure(error) {
    const code = error?.body?.refusal_code || error?.body?.error || '';
    return {
        invite_invalid: tr('peer.invite.invalid', 'This is not an Ouroboros invitation.'),
        invite_expired: tr('peer.invite.expired', 'This invitation has expired. Ask for a new one.'),
        invite_used: tr('peer.invite.used', 'This invitation was already accepted here.'),
        same_install: tr('peer.invite.same', 'This invitation comes from this same Ouroboros.'),
        introduce_unreachable: tr('peer.invite.unreachable', 'The other Ouroboros did not answer. Is it running?'),
        introduce_refused: fmt('The other Ouroboros refused: {detail}', { detail: error?.body?.detail || code }),
    }[code] || failureText(error);
}

/** The rule's field is as tall as its words, so nothing in it is cut (a measured value in its own custom property). */
function fitRule(textarea) {
    if (!textarea) return;
    textarea.style.setProperty('--peer-rule-height', 'auto');
    textarea.style.setProperty('--peer-rule-height', `${textarea.scrollHeight + 2}px`);
}

/** A field with its label (the label names the control; its help describes it, under it). */
const field = (key, label, control, help = '', hook = '') => `<div class="ui-field"><label class="ui-field"><span class="peer-field-label"${hook}>${escapeHtml(label)}</span>${control}</label>${help
    ? `<span class="ui-field-help" id="peer-intro-${key}-help"${key === 'rule' ? ' data-intro-rule-help' : ''}>${escapeHtml(help)}</span>` : ''}</div>`;
const button = (attr, label, primary = false) => `<button type="button" class="btn${primary ? '' : ' btn-default'}"${primary ? ' data-primary' : ''} ${attr}>${escapeHtml(label)}</button>`;
/** Until the owner edits the rule it is Ouroboros's suggestion, in the field's own ink; edited, it is their words. */
const ruleHelp = (edited) => (edited ? tr('places.add.note', 'Your words become the place’s first note.') : tr('peer.rule.help', 'Suggested by Ouroboros. Edit it freely.'));

const computerQuestion = (name) => name
    ? fmt('Who can log in to {name}’s computer?', { name })
    : tr('peer.computer.question', 'Who can log in to the other Ouroboros’s computer?');
const requestChoiceMade = (form) => typeof form.mayStart === 'boolean';
const requestConsequence = (form, name) => requestChoiceMade(form)
    ? consequenceWords({ name, mayStart: form.mayStart })
    : tr('peer.computer.choose', 'Both of you can write. Choose whether the other Ouroboros can ask for work here.');

function composeHtml(form) {
    const inviting = form.mode === 'invite';
    const modes = renderSegmentedField({ target: 'peer-intro-mode', modifier: 'data-intro-mode', chosen: form.mode, label: tr('peer.intro', 'Introduce an Ouroboros'),
        options: [{ value: 'invite', label: tr('peer.mode.invite', 'Invite') }, { value: 'accept', label: tr('peer.mode.accept', 'I have an invitation') }] });
    const preview = form.preview;
    const paste = inviting ? '' : field('paste', tr('peer.invitation', 'Invitation'),
        `<input class="ui-control peer-invite-field" type="text" spellcheck="false" autocomplete="off" data-intro-paste data-i18n-authored value="${escapeHtml(form.invite)}">`)
        + (preview ? `<div class="peer-from"><span class="peer-from-label">${escapeHtml(tr('peer.from', 'From'))}</span>${faceHtml(preview.peerInstallId, 36)}
            <span class="peer-from-words"><span class="peer-from-name" data-i18n-authored>${escapeHtml(preview.peerLabel || 'Ouroboros')}</span><span class="peer-from-home" data-i18n-authored>${escapeHtml(addressOf(preview.peerUrl))}</span></span></div>`
            : form.previewError ? `<p class="ui-status" role="alert" data-i18n-skip data-tone="error">${escapeHtml(form.previewError)}</p>` : '');
    const fromThere = preview ? `<p class="ui-field-help" data-i18n-skip>${escapeHtml(preview.remoteMayStart === true
        ? tr('peer.invite.reported.yes', 'This invitation says you may ask for work there.') : preview.remoteMayStart === false
            ? tr('peer.invite.reported.no', 'This invitation says you may write but not start work there.')
            : tr('peer.invite.reported.unknown', 'The invitation does not say whether you may start work there.'))}</p>` : inviting
        ? `<p class="ui-field-help">${escapeHtml(tr('peer.intro.there', 'The other side decides whether you can ask for work there.'))}</p>` : '';
    const nameHelp = inviting ? tr('peer.name.invite.help', 'What it is called here. Empty: the name it gives itself.')
        : tr('peer.name.help', 'What it is called here: in Circles and on its letters.');
    const name = inviting || preview ? field('name', tr('peer.name', 'Name'), `<input class="ui-control" type="text" maxlength="120" spellcheck="false" autocomplete="off" data-intro-name data-i18n-authored aria-describedby="peer-intro-name-help" value="${escapeHtml(form.name)}" placeholder="${escapeHtml(preview?.peerLabel || 'Ouroboros')}">`, nameHelp) : '';
    const shown = inviting ? form.name.trim() : form.name.trim() || preview?.peerLabel || '';
    const ruleLabel = inviting && !shown ? tr('peer.rule.invite', 'Your rule for the one you invite') : fmt('Your rule for {name}', { name: shown });
    const rule = inviting || preview ? field('rule', ruleLabel, `<textarea class="ui-control peer-rule-field${form.ruleEdited ? ' consciousness-owner-words' : ''}" rows="4" data-intro-rule data-i18n-authored spellcheck="false" aria-describedby="peer-intro-rule-help">${escapeHtml(form.rule)}</textarea>`,
        ruleHelp(form.ruleEdited), ' data-intro-rule-label data-i18n-skip') : '';
    const question = computerQuestion(shown);
    const requests = `<div class="ui-field"><span class="peer-field-label" data-intro-computer-label data-i18n-skip>${escapeHtml(question)}</span>${renderSegmentedField({
        target: 'peer-intro-computer', modifier: 'data-intro-computer', label: question,
        chosen: requestChoiceMade(form) ? (form.mayStart ? 'private' : 'shared') : '',
        options: [{ value: 'private', label: tr('peer.computer.private', 'Only me') },
            { value: 'shared', label: tr('peer.computer.shared', 'Other people too') }],
    })}</div>`;
    const then = `<p class="ui-field-help" data-intro-then data-i18n-skip>${escapeHtml(requestConsequence(form, shown))}</p>`;
    return modes + paste + (inviting || preview ? name + requests + then + fromThere + rule : '');
}

/** An invitation's two parts for its one line: all but its last 12 characters, which stay in view after the ellipsis.
 *  Together they are the whole value, character for character. */
export function inviteParts(invite) {
    const value = String(invite || '');
    return value.length > 24 ? [value.slice(0, -12), value.slice(-12)] : [value, ''];
}

function invitedHtml(state) {
    const { invitation, form } = state;
    const expired = state.expired;
    const status = expired ? tr('peer.invite.expired.own', 'Expired. Create a new invitation.') : tr('peer.invite.unused', 'Not used yet');
    const shown = form.name.trim();
    // The invitation is one read-only line with Copy beside it in an action row, Copy its primary: its start
    // ellipsized, its last characters kept, the value whole in the page, so selecting it takes all of it; Copy copies it
    // whole and writes its result under.
    const [head, tail] = inviteParts(invitation.invite);
    return `<div class="ui-field"><span class="peer-field-label" id="peer-intro-invite-label">${escapeHtml(tr('peer.invitation', 'Invitation'))}</span>
        <div class="ui-action-row peer-invite-row"><output class="ui-control ui-control-readonly peer-invite-field" tabindex="0" data-intro-token data-i18n-authored aria-labelledby="peer-intro-invite-label" aria-describedby="peer-intro-invite-help"><span class="peer-invite-head">${escapeHtml(head)}</span><span class="peer-invite-tail">${escapeHtml(tail)}</span></output>
        ${button(`data-intro-copy${expired ? ' disabled' : ''}`, tr('peer.copy', 'Copy'), !expired)}</div><span class="ui-result" data-intro-copied role="status"></span>
        <span class="ui-field-help" id="peer-intro-invite-help">${escapeHtml(tr('peer.invite.help', 'Works once, for the next 15 minutes. At the other Ouroboros open Circles, press Introduce an Ouroboros and paste it under I have an invitation.'))}</span></div>
        <p class="ui-status" role="status" data-intro-use data-tone="muted">${escapeHtml(status)}</p>
        <div class="own-rule"><p class="peer-field-label" data-i18n-skip>${escapeHtml(shown ? fmt('Your rule for {name}', { name: shown }) : tr('peer.rule.invite', 'Your rule for the one you invite'))}</p>
        ${ownerWordsHtml(form.rule)}</div>`;
}

function codeHtml(link) {
    return `<div class="peer-pairline">${faceHtml('', 36, { own: true })}<span class="peer-pair-link" aria-hidden="true">${fluentIcon('link', 16)}</span>${faceHtml(link.seed, 36)}</div>
        <div class="peer-code-row"><span class="peer-field-label">${escapeHtml(tr('peer.code.here', 'Code here'))}</span><span class="peer-code" data-i18n-authored>${escapeHtml(codeWords(link.code))}</span></div>
        <p class="own-text" data-i18n-skip>${escapeHtml(fmt('{name} shows a code too. Do both show {code}?', { name: link.name, code: codeWords(link.code) }))}</p>
        <p class="consciousness-foot">${escapeHtml(tr('peer.code.note', 'The link starts only after you say they match.'))}</p>`;
}

/** The sheet's head, body and foot for its state; `error` is the last failed step's one alert. */
export function introView(state) {
    const alert = state.error ? `<p class="ui-status" role="alert" data-i18n-skip data-tone="error">${escapeHtml(state.error)}</p>` : '';
    if (state.step === 'code') {
        return { title: tr('peer.code.title', 'Check the code'), body: codeHtml(state.link) + alert,
            actions: button(`data-intro-differ${state.busy ? ' disabled' : ''}`, tr('peer.code.differ', 'They differ')) + button(`data-intro-match${state.busy || state.removalUnconfirmed ? ' disabled' : ''}`, tr('peer.code.match', 'They match'), true) };
    }
    if (state.step === 'invited') {
        // Expired, Close is the one way on, so it is the primary action and focus finds it.
        return { title: tr('peer.intro', 'Introduce an Ouroboros'), body: invitedHtml(state), actions: button('data-intro-close', tr('peer.close', 'Close'), Boolean(state.expired)) };
    }
    const inviting = state.form.mode === 'invite';
    const go = inviting ? tr('peer.invite.create', 'Create invitation') : tr('peer.connect', 'Connect');
    return { title: tr('peer.intro', 'Introduce an Ouroboros'), body: composeHtml(state.form) + alert,
        actions: button('data-intro-cancel', tr('peer.cancel', 'Cancel')) + button(`data-intro-go${!requestChoiceMade(state.form) || (!inviting && !state.form.preview) ? ' disabled' : ''}`, go, true) };
}

function dialogShell() {
    const backdrop = document.createElement('div');
    backdrop.className = 'ui-modal-backdrop peer-intro-backdrop';
    backdrop.innerHTML = `<div class="ui-modal peer-intro" role="dialog" aria-modal="true" aria-labelledby="peer-intro-title">
        <div class="ui-modal-head"><h3 id="peer-intro-title"></h3></div>
        <div class="ui-modal-body scroll-fade-y" tabindex="-1" data-intro-body></div>
        <div class="ui-modal-actions" data-intro-actions></div></div>`;
    return backdrop;
}

/**
 * Open the sheet. `link` opens it at the code check (a row waiting for it); `name` prefills the name (Introduce
 * again). `onChanged(placeId)` hears every write the server confirmed; the page reads again and opens that row.
 * `returnFocus` may be a function, asked at close: the page repaints while the sheet is open.
 */
export function openIntroduction({ returnFocus = document.activeElement, link = null, name = '', onChanged = () => {},
    api = { createInvite, previewInvite, acceptInvite, confirmLink, revokeLink, readLink }, askConfirm = openConfirmDialog } = {}) {
    const state = { step: link ? 'code' : 'compose', link, invitation: null, expired: false, busy: false, removalUnconfirmed: false, error: '',
        form: { mode: 'invite', name, rule: ruleSuggestion({ name }), ruleEdited: false, mayStart: null, invite: '', preview: null, previewedInvite: '', previewError: '' } };
    const backdrop = dialogShell();
    const dialog = backdrop.querySelector('[role="dialog"]');
    const body = backdrop.querySelector('[data-intro-body]');
    let closed = false, watch = 0, previewTimer = 0, previewAsk = 0, disposeFocus = null;
    const disposeFade = bindScrollFade(body);
    const watching = new AbortController();

    function paint(focus = '') {
        const view = introView(state);
        backdrop.querySelector('#peer-intro-title').textContent = view.title;
        body.innerHTML = view.body;
        backdrop.querySelector('[data-intro-actions]').innerHTML = view.actions;
        dialog.dataset.step = state.step;
        fitRule(body.querySelector('[data-intro-rule]'));
        const target = focus && backdrop.querySelector(focus);
        (target || backdrop.querySelector('[data-primary]:not([disabled])') || body).focus?.({ preventScroll: true });
    }
    function close() {
        if (closed) return;
        closed = true;
        watching.abort(); clearTimeout(watch); clearTimeout(previewTimer);
        disposeFocus?.();
        disposeFade();
        backdrop.remove();
    }
    const say = (selector, text, tone = 'muted') => setResult(backdrop.querySelector(selector), { text, tone });

    function retell() {  // the rule's suggestion and its consequence follow the name
        const form = state.form;
        const name = form.mode === 'invite' ? form.name.trim() : form.name.trim() || form.preview?.peerLabel || '';
        if (!form.ruleEdited) form.rule = ruleSuggestion({ name });
        const keep = document.activeElement?.matches?.('[data-intro-name]') ? '[data-intro-name]' : '';
        const caret = keep ? [document.activeElement.selectionStart, document.activeElement.selectionEnd] : null;
        paint(keep || '[data-intro-mode] .active');
        if (caret) backdrop.querySelector('[data-intro-name]')?.setSelectionRange(...caret);
    }

    async function readPreview(ask) {
        const form = state.form, text = form.invite.trim();
        const current = () => !closed && form.mode === 'accept' && ask === previewAsk && form.invite.trim() === text;
        if (!text || !current()) return;
        try {
            const preview = await api.previewInvite(text);
            if (!current()) return;
            form.preview = preview; form.previewedInvite = text; form.previewError = '';
        } catch (error) {
            if (!current()) return;
            form.preview = null; form.previewedInvite = ''; form.previewError = introFailure(error);
        }
        if (!form.ruleEdited) form.rule = ruleSuggestion({ name: form.name.trim() || form.preview?.peerLabel || '' });
        const input = body.querySelector('[data-intro-paste]');
        const caret = input === document.activeElement ? [input.selectionStart, input.selectionEnd] : null;
        paint(caret ? '[data-intro-paste]' : ownFocus());
        if (caret) body.querySelector('[data-intro-paste]')?.setSelectionRange(...caret);
    }

    // Poll just the invitation's local link; one outstanding read, with no reads after close or while hidden.
    function watchUse() {
        if (closed || state.step !== 'invited') return;
        watch = setTimeout(async () => {
            if (closed || state.step !== 'invited') return;
            if (Date.parse(state.invitation.expiresAt) <= Date.now()) { state.expired = true; paint(); return; }
            try {
                if (!document.hidden) {
                    const link = await api.readLink(state.invitation.linkId, watching.signal);
                    if (closed) return;
                    if (link?.placeId) {
                        onChanged(link.placeId);
                        if (link.codeCheck === 'confirmed' || link.revokedAt) { close(); return; }
                        state.step = 'code';
                        state.link = forCheck(link);
                        paint();
                        return;
                    }
                    say('[data-intro-use]', tr('peer.invite.unused', 'Not used yet'));
                }
            } catch {
                if (!closed) say('[data-intro-use]', tr('peer.invite.status.unread', 'Could not read whether the invitation was used. Checking again.'), 'warn');
            }
            watchUse();
        }, WATCH_MS);
    }

    const ownFocus = () => document.activeElement?.closest('[data-intro-mode]') ? '[data-intro-mode] .active' : '';
    const forCheck = (link) => ({ ...link, name: link.peerLabel || 'Ouroboros', seed: link.peerInstallId });
    function begin() {
        state.busy = true; state.error = '';
        dialog.setAttribute('aria-busy', 'true');
        body.inert = true;
        for (const node of backdrop.querySelectorAll('[data-intro-actions] button')) node.disabled = true;
    }
    function end(focus) {
        state.busy = false;
        dialog.removeAttribute('aria-busy');
        body.inert = false;
        if (!closed) paint(focus);
    }
    async function go() {
        const form = state.form, inviting = form.mode === 'invite', invite = form.invite.trim();
        if (state.busy || !requestChoiceMade(form) || (!inviting && (!form.preview || form.previewedInvite !== invite))) return;
        begin();
        const choice = { mayStart: form.mayStart, note: form.rule, label: form.name.trim() || form.preview?.peerLabel || '' };
        try {
            if (inviting) {
                state.invitation = await api.createInvite(choice);
                state.step = 'invited';
                watchUse();
            } else {
                const accepted = await api.acceptInvite({ ...choice, invite });
                state.step = 'code';
                state.link = forCheck(accepted);
                onChanged(accepted.placeId);
            }
        } catch (error) {
            const code = error?.body?.refusal_code || error?.body?.error;
            if (!inviting && ['invite_invalid', 'invite_expired', 'invite_used', 'same_install'].includes(code)) {
                form.preview = null; form.previewedInvite = '';
            }
            state.error = introFailure(error);
        } finally {
            end(state.step === 'invited' ? '[data-intro-copy]' : state.step === 'compose' && !inviting && !form.preview ? '[data-intro-paste]' : '');
        }
    }

    async function match() {
        if (state.busy || state.removalUnconfirmed) return;
        begin();
        try {
            await api.confirmLink(state.link.linkId);
            onChanged(state.link.placeId);
            close();
        } catch (error) {
            state.error = introFailure(error);
        } finally {
            end('[data-intro-match]');
        }
    }

    async function differ() {
        if (state.busy) return;
        const { link } = state;
        state.busy = true;
        backdrop.hidden = true;
        try {
            const remove = await askConfirm({ title: tr('peer.code.differ.title', 'The codes differ'),
                body: tr('peer.code.differ.body', 'Nothing has gone over this link. Remove it and introduce the other Ouroboros again.'),
                confirmLabel: tr('peer.link.remove', 'Remove link'), cancelLabel: tr('peer.back', 'Back'), danger: true });
            if (closed || !remove) return;
            backdrop.hidden = false;
            begin();
            let removed;
            try {
                removed = await api.revokeLink(link.linkId);
            } catch (error) {
                // Reconcile the same local link after a lost acknowledgement. No new mutation is sent here.
                removed = api.readLink ? await api.readLink(link.linkId, watching.signal).catch(() => null) : null;
                if (!removed?.revokedAt) throw error;
            }
            if (!removed?.revokedAt) throw new Error('Removal not confirmed');
            onChanged(link.placeId);
            close();
        } catch {
            state.removalUnconfirmed = true;
            state.error = tr('peer.link.remove.unconfirmed', 'Removal was not confirmed. Choose They differ to check or try again.');
        } finally {
            backdrop.hidden = false;
            end('[data-intro-differ]');
        }
    }

    const onClick = (event) => {
        if (state.busy) return;
        if (event.target === backdrop) { close(); return; }
        const target = event.target.closest?.('button');
        if (!target || target.disabled) return;
        const mode = target.closest('[data-intro-mode]');
        if (mode && target.dataset.effortValue !== state.form.mode) {
            ++previewAsk; clearTimeout(previewTimer);
            state.form.mode = target.dataset.effortValue; state.form.previewError = ''; state.error = '';
            state.form.ruleEdited = false; state.form.mayStart = null; retell();
            if (state.form.mode === 'accept' && state.form.invite.trim() && !state.form.preview) void readPreview(previewAsk);
        } else if (target.closest('[data-intro-computer]')) {
            const choice = target.dataset.effortValue;
            if (choice === 'private' || choice === 'shared') {
                state.form.mayStart = choice === 'private';
                paint('[data-intro-computer] .active');
            }
        } else if (target.matches('[data-intro-cancel], [data-intro-close]')) close();
        else if (target.matches('[data-intro-go]')) void go();
        else if (target.matches('[data-intro-match]')) void match();
        else if (target.matches('[data-intro-differ]')) void differ();
        else if (target.matches('[data-intro-copy]')) {
            copyTextToClipboard(state.invitation.invite, { anchor: target })
                .then(() => say('[data-intro-copied]', tr('peer.copied', 'Copied'), 'ok'))
                .catch(() => {  // the line stands selected, its whole value, for the owner's own copy
                    const token = backdrop.querySelector('[data-intro-token]');
                    if (token) globalThis.getSelection?.()?.selectAllChildren(token);
                    say('[data-intro-copied]', tr('peer.copy.failed', 'Copy did not work: select the invitation and copy it.'), 'warn');
                });
        }
    };
    function retellNames(name) {  // the consequence and the rule's label follow the typed name
        const inviting = state.form.mode === 'invite';
        const set = (selector, text, attr = '') => {
            const node = body.querySelector(selector);
            if (!node) return;
            if (attr) node.setAttribute(attr, text); else if (node.textContent !== text) node.textContent = text;
        };
        set('[data-intro-then]', requestConsequence(state.form, name));
        set('[data-intro-computer-label]', computerQuestion(name));
        set('[data-intro-computer]', computerQuestion(name), 'aria-label');
        set('[data-intro-rule-label]', inviting && !name ? tr('peer.rule.invite', 'Your rule for the one you invite') : fmt('Your rule for {name}', { name }));
    }
    const onInput = (event) => {
        const form = state.form;
        if (event.target.matches('[data-intro-rule]')) {
            form.rule = event.target.value;
            if (!form.ruleEdited) {  // from the first edit the words are the owner's: their voice, and a help that says so
                form.ruleEdited = true;
                event.target.classList.add('consciousness-owner-words');
                const help = body.querySelector('[data-intro-rule-help]');
                if (help) help.textContent = ruleHelp(true);
            }
            fitRule(event.target);
            return;
        }
        if (event.target.matches('[data-intro-name]')) {
            form.name = event.target.value;
            const name = form.mode === 'invite' ? form.name.trim() : form.name.trim() || form.preview?.peerLabel || '';
            if (!form.ruleEdited) { form.rule = ruleSuggestion({ name }); const rule = body.querySelector('[data-intro-rule]'); if (rule) { rule.value = form.rule; fitRule(rule); } }
            retellNames(name);
            return;
        }
        if (event.target.matches('[data-intro-paste]')) {
            const caret = [event.target.selectionStart, event.target.selectionEnd];
            form.invite = event.target.value;
            const ask = ++previewAsk;
            const repaint = form.preview || form.previewError || state.error;
            form.preview = null; form.previewedInvite = ''; form.previewError = ''; form.mayStart = null; state.error = '';
            if (repaint) {
                paint('[data-intro-paste]');
                body.querySelector('[data-intro-paste]')?.setSelectionRange(...caret);
            }
            clearTimeout(previewTimer);
            if (form.invite.trim()) previewTimer = setTimeout(() => void readPreview(ask), PREVIEW_DELAY_MS);
        }
    };
    backdrop.addEventListener('click', onClick);
    backdrop.addEventListener('input', onInput);
    document.body.appendChild(backdrop);
    paint();
    disposeFocus = bindDialogFocus(dialog, { initialFocus: dialog.querySelector('[data-primary]:not([disabled])') || dialog.querySelector('[data-intro-name]') || body, returnFocus, onEscape: () => { if (!state.busy) close(); } });
    return { close, get closed() { return closed; } };
}

/** A row waiting for its check opens the sheet at the code: the same six digits both installations show. */
export function openCodeCheck(place, options = {}) {
    return openIntroduction({ ...options, link: { linkId: place.link?.linkId || '', code: place.link?.code || '',
        name: place.label || 'Ouroboros', seed: place.accountId || place.link?.peerInstallId || '', placeId: place.id } });
}
