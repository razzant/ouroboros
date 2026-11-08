// Integration-secret request form (DESIGN "Quiz card" → Secret request). The agent's
// showSecretHitl(pluginName, fields) arrives as an ordinary question card carrying typed
// `secret_request` metadata; this module gives that card the request's facts and two actions, and
// owns one protected form per request, opened only from its card.
//
// A value lives only in the open form's password controls, and on Show in that field's own read-only
// view of the local draft (settings_secrets.bindSecretReveal; a saved value is never read back). It is
// never written to a dataset, attribute, storage, a toast or a log. The form belongs to the request,
// not to a card node: a card that re-renders keeps the open form, its draft and focus; Close, the
// request's settlement and disposal empty the controls, and like Escape and a click outside, Close
// waits for a save in flight, whose reply settles the form: but only a bounded wait. If no reply (or
// no readable body) arrives within `waitMs`, the owner is released, never trapped: the save is shown as
// not confirmed (Save blocked, Check status offered, Close available), nothing is resent, and the late
// reply still settles the correct request. A status read or cancel that stalls is released the same way.
// Opening reads the request's status:
// whether each asked key is set (a boolean, never a value) and which skills read it; until that read
// (or a reply's facts) arrives in this form, no earlier read is shown as current. A save sends what the form
// showed (`observed`): a key set or cleared elsewhere since is refused and the form re-renders as
// Replace or Save with the draft kept, so Replace is always the owner's explicit choice. A save of
// unknown outcome (no readable reply, saved null, or written but unverified/unrecorded) blocks Save
// until Check status; saving again over an unconfirmed attempt is then an explicit owner action under a
// NEW request id. Only fixed, code-driven copy is shown: a reply's free text is never rendered.
import { apiClient } from './api_client.js';
import { bindDialogFocus } from './ui_interactions.js';
import { bindSecretReveal } from './settings_secrets.js';
import { fmt, tr } from './i18n.js';

const OPEN_STATES = ['open', 'expired_terminal'];
const SAVED_EVENTS = ['secret_filled', 'secret_replaced'];
// Fixed copy per typed refusal that saved nothing; an unknown code is a generic refusal.
const REFUSAL = {
    secret_value_empty: ['secret.error_empty', 'Enter a value for every field.'],
    secret_value_placeholder: ['secret.error_placeholder', 'That looks like a masked placeholder, which Settings would not keep. Enter the real value.'],
    secret_value_invalid: ['secret.error_invalid', 'The value contains a line break or control character.'],
    secret_value_too_long: ['secret.error_too_long', 'The value is too long.'],
    secret_fields_mismatch: ['secret.error_mismatch', 'The form no longer matches the request; nothing was saved. Close it and open it again.'],
    observed_required: ['secret.error_mismatch', 'The form no longer matches the request; nothing was saved. Close it and open it again.'],
    secret_request_stale: ['secret.error_stale', 'The skill no longer maps these fields to the same Settings keys; nothing was saved. Ask Ouroboros to request it again.'],
    settings_busy: ['secret.error_busy', 'Settings are being saved elsewhere; nothing was saved. Try again.'],
    settings_locked: ['secret.error_busy', 'Settings are being saved elsewhere; nothing was saved. Try again.'],
    secret_save_failed: ['secret.error_failed', 'Settings could not be written; nothing was saved. Try again.'],
    settings_unreadable: ['secret.error_unreadable', 'Settings exist but could not be read, so nothing was saved or overwritten. Repair the settings file, then check status.'],
    secret_request_settled: ['secret.error_settled', 'This request was already saved or cancelled; nothing was saved now.'],
};
const REFUSED = ['secret.error_refused', 'The save was refused; nothing was saved.'];
// Written, or of unknown outcome: never "nothing was saved".
const UNKNOWN = {
    secret_save_unverified: ['secret.error_unverified', 'Settings were written, but the value could not be read back. Check status before saving again.'],
    secret_not_retained: ['secret.error_not_retained', 'Settings were written, but the value read back differs from what you entered. Check status before saving again.'],
    secret_completion_unrecorded: ['secret.error_unrecorded', 'The value was saved to Settings, but the request could not be marked complete. Check status before saving again.'],
    secret_save_unconfirmed: ['secret.error_unconfirmed', 'This save was never confirmed; its result is unknown. Check status before saving again.'],
};
const UNCONFIRMED = ['secret.unconfirmed', 'The save was not confirmed; its result is unknown. Check status before saving again.'];
const WAIT_MS = 15000;
const LATE = Symbol('late');
const nonce = () => globalThis.crypto?.randomUUID?.() || `s-${Date.now()}-${Math.random().toString(36).slice(2)}`;

// A stalled transport or body must release the UI, never declare failure: the caller's own promise
// still delivers the late reply. Waits are kept in `owner.timers`, so a close or dispose clears them
// (resolving LATE, which every caller treats as obsolete once its owner is gone).
function bounded(owner, promise, ms) {
    let timer, stop;
    const timeout = new Promise((resolve) => {
        stop = () => { clearTimeout(timer); resolve(LATE); };
        timer = setTimeout(stop, ms);
        owner.timers.add(stop);
    });
    return Promise.race([promise, timeout]).finally(() => { clearTimeout(timer); owner.timers.delete(stop); });
}

/** The card's typed request ({pluginName, fields:[{id, settingKey}]}) or null. */
export function secretRequestOf(src) {
    const raw = src && typeof src.secret_request === 'object' ? src.secret_request : null;
    const fields = Array.isArray(raw?.fields) ? raw.fields.filter((row) => row && typeof row.id === 'string' && row.id) : [];
    if (!raw || typeof raw.plugin_name !== 'string' || !raw.plugin_name || !fields.length) return null;
    return { pluginName: raw.plugin_name, fields: fields.map((row) => ({ id: row.id, settingKey: String(row.setting_key || '') })) };
}

const el = (tag, className = '', text = '') => {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text) node.textContent = text;
    return node;
};
const button = (className, text) => { const node = el('button', className, text); node.type = 'button'; return node; };
const list = (items) => items.join(', ');

// One line per fact; rebuilt only when the lines changed, so focus and selection elsewhere stay.
function setLines(container, className, lines) {
    const current = [...container.children].map((node) => node.textContent);
    if (current.length === lines.length && current.every((text, index) => text === lines[index])) return false;
    while (container.children.length) container.children[0].remove();
    container.append(...lines.map((line) => el('li', className, line)));
    return true;
}

/** What one reader of the key can read, from the host's facts (a grant is never inferred here). */
export function readerLine(row) {
    const state = row.enabled ? tr('secret.reader_enabled', 'enabled') : tr('secret.reader_disabled', 'disabled');
    let grant;
    if (row.grant_unknown) grant = tr('secret.reader_grant_unknown', 'its grant could not be read');
    else if (row.unsupported) grant = tr('secret.reader_unsupported', 'this skill type receives no Settings keys');
    else {
        const missing = (row.reads || []).filter((key) => !(row.granted || []).includes(key));
        grant = missing.length ? fmt('needs {keys} granted in Skills', { keys: list(missing) })
            : tr('secret.reader_covered', 'its grant covers it');
        if (missing.length && row.enabled)
            grant += tr('secret.reader_blocked', '; after saving, this enabled skill cannot run until the keys are granted');
    }
    return fmt('{name} · {state} · {grant}', { name: row.name, state, grant });
}

/**
 * One controller per chat surface. `accept(card, body)` merges a canonical receipt into the card
 * (chat_decision's observation and setCardState); `canUse(card)` is false while the surface's
 * authority is stale; `paintHead(card)` repaints the question head's mark from
 * `card.dataset.submission`, as for an ordinary answer in flight.
 */
export function createSecretRequestForms({ apiFetch, accept = () => {}, showToast = () => {}, canUse = () => true,
    host = () => document.body, bindFocus = bindDialogFocus, bindReveal = bindSecretReveal,
    paintHead = () => false, waitMs = WAIT_MS } = {}) {
    let disposed = false;
    const views = new Map();  // card node -> its view
    const forms = new Map();  // request -> its form: outlives any one card node
    const transport = apiFetch ? { fetchImpl: apiFetch } : {};

    function formOf(quiz) {
        const key = `${quiz.taskId}:${quiz.quizId}`;
        if (!forms.has(key)) forms.set(key, { key, quiz, request: quiz.secretRequest, state: 'open', dialog: null,
            requestId: '', pending: false, unknown: false, seenUnconfirmed: false, facts: null, cards: new Set(),
            attempt: null, timers: new Set() });
        return forms.get(key);
    }
    const decisionOf = (form) => `quiz:${form.quiz.taskId}:${form.quiz.quizId}`;
    const usable = (form) => OPEN_STATES.includes(form.state) && [...form.cards].some((card) => canUse(card));

    function mount(card, quiz) {
        if (!quiz.secretRequest || views.has(card)) return;
        const form = formOf(quiz);
        form.cards.add(card);
        card.dataset.secretRequest = '1';
        const box = el('div', 'chat-secret-request');
        const fields = el('ul', 'chat-secret-fields');
        for (const field of form.request.fields) {
            const row = el('li', 'chat-secret-field');
            row.append(el('span', 'chat-secret-field-id', field.id),
                el('span', 'chat-secret-field-key', fmt('Saved in Settings as {key}', { key: field.settingKey })));
            fields.append(row);
        }
        const status = el('div', 'chat-secret-status ui-status');
        status.setAttribute('role', 'status');
        const readers = el('ul', 'chat-secret-readers');
        const actions = el('div', 'chat-secret-actions');
        const open = button('btn btn-primary btn-sm chat-secret-open', tr('secret.enter', 'Enter secret…'));
        const cancel = button('btn btn-default btn-sm chat-secret-cancel', tr('secret.cancel_request', 'Cancel request'));
        actions.append(open, cancel);
        box.append(fields, status, readers, actions);
        card.append(box);
        views.set(card, { card, form, status, readers, actions, open, cancel });
        open.addEventListener('click', () => openDialog(form, card));
        cancel.addEventListener('click', () => cancelRequest(form));
    }

    // The card's settled or pending facts; returns whether the DOM changed.
    function sync(card, state, current = {}) {
        const view = views.get(card);
        if (!view) return false;
        const form = view.form;
        form.state = state;
        const outcome = current.secret_outcome || {};
        // An attempt nobody here holds: unknown until a status read, unless the owner already read it as such.
        if (OPEN_STATES.includes(state) && current.secret_save?.state === 'writing' && !form.pending && !form.seenUnconfirmed)
            form.unknown = true;
        let changed = paintCard(view, state, outcome);
        if (!OPEN_STATES.includes(state) && state !== 'unknown') {
            if (view.actions.isConnected) { view.actions.remove(); changed = true; }
            settle(form);
        } else renderDialog(form);
        return paintSubmission(form) || changed;
    }

    function cardStatus(form, state, outcome) {
        const fields = list(outcome.fields || form.request.fields.map((field) => field.id));
        if (state === 'answered' && outcome.event === 'secret_replaced')
            return fmt('Replaced in Settings: {fields}. Enabled skills may use it at their next call; some keep the old value until disabled and re-enabled.', { fields });
        if (state === 'answered' && outcome.event === 'secret_filled')
            return fmt('Saved to Settings: {fields}. Access and enablement are unchanged.', { fields });
        if (state === 'closed' && outcome.interrupted_save)
            return tr('secret.cancel_interrupted', 'An earlier save was never confirmed; whether its value reached Settings is unknown.');
        if (form.pending) return tr('secret.saving', 'Saving…');
        if (OPEN_STATES.includes(state) && form.unknown)
            return tr('secret.unconfirmed_card', 'A save was not confirmed; its result is unknown. Open the form to check status.');
        if (state === 'expired_terminal')
            return tr('secret.finished_card', 'The task that asked has finished. Saving still stores the key; it does not resume that work.');
        return '';
    }

    function paintCard(view, state, outcome) {
        const form = view.form;
        let changed = false;
        const text = cardStatus(form, state, outcome);
        const late = state === 'answered' && outcome.late
            ? ` ${tr('secret.saved_late', 'The task had already finished; nothing was resumed.')}` : '';
        if (view.status.textContent !== text + late) { view.status.textContent = text + late; changed = true; }
        const tone = state === 'answered' ? 'ok' : (form.unknown || outcome.interrupted_save) ? 'warn' : '';
        if ((view.status.dataset.tone || '') !== tone) {
            if (tone) view.status.dataset.tone = tone; else delete view.status.dataset.tone;
            changed = true;
        }
        if (state === 'answered' || !OPEN_STATES.includes(state)) {
            const lines = state !== 'answered' ? [] : outcome.consumers_unknown
                ? [tr('secret.readers_unknown', 'Which skills read this key could not be read.')]
                : (outcome.consumers || []).map(readerLine);
            if (setLines(view.readers, 'chat-secret-reader', lines)) changed = true;
        }
        const answerable = OPEN_STATES.includes(state) && canUse(view.card);
        // A save or cancel in flight settles the request first: no form is opened meanwhile.
        view.open.disabled = !answerable || form.pending;
        view.cancel.disabled = !answerable || form.pending;
        return changed;
    }

    // A save in flight or of unknown outcome is not the owner's move: the head leaves amber exactly
    // as for an ordinary answer (DESIGN "Status marks"); a settled card has none.
    function paintSubmission(form) {
        const phase = !OPEN_STATES.includes(form.state) ? '' : form.pending ? 'sending' : form.unknown ? 'unknown' : '';
        let changed = false;
        for (const card of form.cards) {
            if ((card.dataset.submission || '') === phase) continue;
            card.dataset.submission = phase;
            if (paintHead(card) !== false) changed = true;
        }
        return changed;
    }

    function repaintCards(form) {
        if (!OPEN_STATES.includes(form.state)) return;
        for (const card of form.cards) {
            const view = views.get(card);
            if (view) paintCard(view, form.state, {});
        }
        paintSubmission(form);
    }
    const openButton = (form) => [...form.cards].map((card) => views.get(card)?.open).find((node) => node?.isConnected);

    const filled = (form) => Boolean(form.dialog) && form.dialog.inputs.every((input) => String(input.value || '').trim());
    const replacing = (form) => Boolean(form.facts?.keys?.some((row) => row.configured));

    function openDialog(form, card) {
        if (form.dialog) { form.dialog.inputs[0]?.focus?.(); return; }
        if (!usable(form)) return;
        const backdrop = el('div', 'marketplace-modal-backdrop secret-request-backdrop');
        // Every visible string here comes through tr/fmt; the overlay never walks the form.
        backdrop.setAttribute('data-i18n-authored', '');
        const panel = el('div', 'marketplace-modal secret-request-dialog');
        panel.setAttribute('role', 'dialog');
        panel.setAttribute('aria-modal', 'true');
        const titleId = `secret-request-title-${form.quiz.quizId}`;
        panel.setAttribute('aria-labelledby', titleId);
        const head = el('div', 'marketplace-modal-head');
        const title = el('h3', 'secret-request-title', fmt('{plugin}: secret needed', { plugin: form.request.pluginName }));
        title.id = titleId;
        const close = button('btn btn-default btn-sm secret-request-close', tr('secret.close', 'Close'));
        close.setAttribute('aria-label', tr('secret.close_form', 'Close the form; the request stays open'));
        head.append(title, close);
        const body = el('div', 'marketplace-modal-body secret-request-body');
        body.append(el('p', 'secret-request-intro', tr('secret.intro', 'The value is saved to Settings. It is not sent in this conversation.')));
        const late = el('p', 'secret-request-late ui-status');
        late.dataset.tone = 'neutral';
        const inputs = [], helps = [], reveals = [];
        const fieldsBox = el('div', 'secret-request-fields');
        for (const field of form.request.fields) {
            const wrap = el('div', 'ui-field secret-request-field');
            const inputId = `secret-request-${form.quiz.quizId}-${field.id}`;
            const label = el('label', 'secret-request-label', field.id);
            label.setAttribute('for', inputId);
            const row = el('div', 'secret-input-row');
            const input = el('input', 'ui-control secret-input secret-request-input');
            input.type = 'password';
            input.value = '';
            input.id = inputId;
            input.autocomplete = 'off';
            input.setAttribute('autocapitalize', 'off');
            input.setAttribute('spellcheck', 'false');
            input.dataset.secretField = field.id;
            const show = button('btn btn-default secret-toggle secret-request-show', tr('secret.show', 'Show'));
            row.append(input, show);
            const help = el('div', 'ui-field-help secret-request-key');
            wrap.append(label, row, help);
            fieldsBox.append(wrap);
            input.addEventListener('input', () => {
                form.requestId = '';  // a changed entry is a new request; a retry of the same entries keeps its id
                renderDialog(form);
            });
            input.addEventListener('keydown', (event) => {
                if (event.key === 'Enter' && !event.isComposing && !form.dialog?.ok.disabled) {
                    event.preventDefault();
                    submit(form);
                }
            });
            inputs.push(input);
            helps.push(help);
            reveals.push({ input, show });
        }
        const readersTitle = el('p', 'secret-request-readers-title');
        const readers = el('ul', 'secret-request-readers');
        const warning = el('p', 'secret-request-warning ui-status');
        warning.dataset.tone = 'warn';
        const error = el('p', 'secret-request-error ui-status');
        error.setAttribute('role', 'alert');
        error.dataset.tone = 'danger';
        const check = button('btn btn-default btn-sm secret-request-check', tr('secret.check_status', 'Check status'));
        body.append(late, fieldsBox, readersTitle, readers, warning, error, check);
        const footer = el('div', 'marketplace-modal-actions');
        const dismiss = button('btn btn-default secret-request-cancel', tr('secret.close', 'Close'));
        const ok = button('btn secret-request-ok', tr('secret.save', 'Save'));
        ok.dataset.primary = '';  // the dialog foot's one primary action (ui.css primary slot)
        footer.append(dismiss, ok);
        panel.append(head, body, footer);
        backdrop.append(panel);
        form.dialog = { backdrop, panel, inputs, helps, readersTitle, readers, late, warning, error, check, ok, close,
            dismiss, release: null, reveals: [], errorText: '', errorTone: 'danger', loading: true, read: null,
            timers: new Set() };
        form.facts = null;  // an earlier form's read is not Settings now: only this form's own read decides
        // Every way out waits for a save in flight: its reply settles this form (a reopened one stayed loading).
        const hide = () => { if (!form.pending) closeDialog(form); };
        close.addEventListener('click', hide);
        dismiss.addEventListener('click', hide);
        ok.addEventListener('click', () => submit(form));
        check.addEventListener('click', () => checkStatus(form));
        backdrop.addEventListener('click', (event) => { if (event.target === backdrop) hide(); });
        host().append(backdrop);
        // The overlay never walks this form, so the shared reveal takes its words through `tr`.
        const copy = { show: tr('secret.show', 'Show'), hide: tr('secret.hide', 'Hide') };
        form.dialog.reveals = reveals.map(({ input, show }) => bindReveal(input, show, { copy }));
        // Focus returns to the request's current card (a re-rendered card is a new node), never a stale one.
        form.dialog.release = bindFocus(panel, { initialFocus: inputs[0], returnFocus: null, onEscape: hide });
        form.opener = views.get(card)?.open || null;
        renderDialog(form);
        if (!form.pending) refreshStatus(form);  // a save of this window in flight settles the form itself
    }

    // The form's whole state from the form's facts; never touches the draft or focus.
    function renderDialog(form) {
        const dialog = form.dialog;
        if (!dialog) return;
        const keys = new Map((form.facts?.keys || []).map((row) => [row.id, row]));
        form.request.fields.forEach((field, index) => {
            const known = keys.get(field.id);
            // After a save of unknown outcome the last read no longer describes Settings.
            const state = form.unknown ? tr('secret.key_unknown', 'unknown until you check status')
                : form.facts?.keys_unknown || dialog.statusFailed ? tr('secret.key_unreadable', 'could not be read')
                : !known ? tr('secret.key_checking', 'checking Settings…')
                    : known.configured ? tr('secret.key_set', 'a value is set (hidden)') : tr('secret.key_not_set', 'not set');
            const text = fmt('Saved in Settings as {key} · {state}', { key: field.settingKey, state });
            if (dialog.helps[index].textContent !== text) dialog.helps[index].textContent = text;
            dialog.inputs[index].readOnly = form.pending;
        });
        const consumers = form.facts?.consumers || [];
        const title = form.facts && !form.facts.consumers_unknown && consumers.length
            ? tr('secret.readers_title', 'Skills that read this key:') : '';
        if (dialog.readersTitle.textContent !== title) dialog.readersTitle.textContent = title;
        setLines(dialog.readers, 'secret-request-reader', form.facts?.consumers_unknown
            ? [tr('secret.readers_unknown', 'Which skills read this key could not be read.')]
            : !form.facts ? [] : consumers.length ? consumers.map(readerLine)
                : [tr('secret.readers_none', 'No installed skill reads this key yet.')]);
        const replace = replacing(form);
        const enabledReaders = consumers.filter((row) => row.enabled).map((row) => row.name);
        dialog.warning.textContent = !replace ? '' : enabledReaders.length
            ? fmt('Replace overwrites the saved value. Enabled skills that read it ({skills}) may use the new value on their next call; some keep the old one until you disable and re-enable them in Skills.', { skills: list(enabledReaders) })
            : tr('secret.replace_warning', 'Replace overwrites the saved value; it cannot be restored from here.');
        dialog.late.textContent = form.state === 'expired_terminal'
            ? tr('secret.finished_form', 'The task that asked has finished. Saving still stores the key; it does not resume that work.') : '';
        dialog.error.textContent = dialog.errorText;
        dialog.error.dataset.tone = dialog.errorTone;
        dialog.check.hidden = !form.unknown && !form.facts?.keys_unknown && !dialog.statusFailed;
        dialog.close.disabled = dialog.dismiss.disabled = form.pending;
        dialog.ok.textContent = form.pending ? tr('secret.saving', 'Saving…')
            : replace ? tr('secret.replace', 'Replace') : tr('secret.save', 'Save');
        dialog.ok.disabled = form.pending || form.unknown || !form.facts?.keys || dialog.loading || !filled(form)
            || !usable(form);
    }

    function setMessage(form, text, tone = 'danger') {
        if (!form.dialog) return;
        form.dialog.errorText = text;
        form.dialog.errorTone = tone;
        renderDialog(form);
    }

    // Facts for the form: whether each key is set and who reads it. The draft and focus stay. A reply's
    // facts are as fresh as a status read, so they also end a pending read's loading state.
    function takeFacts(form, body) {
        if (!body || typeof body !== 'object') return;
        if (!(Array.isArray(body.keys) || body.keys_unknown || Array.isArray(body.consumers) || body.consumers_unknown)) return;
        form.facts = { keys: body.keys || null, keys_unknown: Boolean(body.keys_unknown),
            consumers: body.consumers || [], consumers_unknown: Boolean(body.consumers_unknown) };
        if (form.dialog) Object.assign(form.dialog, { loading: false, statusFailed: false });
    }

    // One read belongs to the exact form it was made for (`dialog.read` is its identity): a reply that
    // arrives after Close, a reopen or a newer read is obsolete and changes nothing. A read that stalls
    // releases the form as unreadable (Check status offered); its own late reply, if still current,
    // is then taken as a normal one.
    async function refreshStatus(form) {
        const dialog = form.dialog;
        if (!dialog || form.pending) return null;
        const read = dialog.read = {};
        const current = () => form.dialog === dialog && dialog.read === read;
        dialog.loading = true;
        dialog.statusFailed = false;
        renderDialog(form);
        const call = apiClient.secretRequestStatus({ decision_id: decisionOf(form) }, transport).then((reply) => reply,
            (error) => (error?.body && typeof error.body === 'object' && error.body.state ? error.body : null));
        let body = await bounded(dialog, call, waitMs);
        if (body === LATE) {
            if (!current()) return null;
            dialog.loading = false;
            dialog.statusFailed = true;
            form.facts = null;
            setMessage(form, tr('secret.status_unavailable', 'Settings could not be read. Check status to try again.'), 'warn');
            body = await call;
        }
        if (!current()) return body;
        dialog.loading = false;
        if (!body?.state) {
            dialog.statusFailed = true;
            form.facts = null;  // the keys are unknown now, whatever an earlier read said
            setMessage(form, tr('secret.status_unavailable', 'Settings could not be read. Check status to try again.'), 'warn');
            return null;
        }
        if (!OPEN_STATES.includes(body.state)) { acceptAll(form, body); return body; }
        takeFacts(form, body);
        const wasUnknown = form.unknown;
        form.unknown = false;
        if (body.save_unconfirmed) {
            // Read as unknown: saving again is the owner's explicit choice, under a new request.
            form.seenUnconfirmed = true;
            form.requestId = '';
            setMessage(form, tr('secret.status_unknown', 'An earlier save was never confirmed; whether it stored a value is unknown. Saving again writes over whatever it may have stored.'), 'warn');
        } else {
            form.seenUnconfirmed = false;
            setMessage(form, wasUnknown ? tr('secret.status_not_saved', 'No save is recorded for this request. Save resends the same request, so it cannot save twice.') : '', 'neutral');
        }
        repaintCards(form);
        return body;
    }

    function acceptAll(form, body) {
        for (const card of [...form.cards]) accept(card, body);
    }

    function closeDialog(form, { restoreFocus = true } = {}) {
        const dialog = form.dialog;
        if (!dialog) return;
        const owned = dialog.panel.contains(globalThis.document?.activeElement);
        for (const reveal of dialog.reveals) reveal?.reset?.();
        for (const input of dialog.inputs) input.value = '';
        form.dialog = null;
        form.requestId = '';
        [...dialog.timers].forEach((stop) => stop());  // its own waits end with it
        // Hiding the form clears its values, never an in-flight or unknown save's status.
        dialog.release?.({ restoreFocus: false });
        dialog.backdrop.remove();
        if (restoreFocus && owned) (openButton(form) || form.opener)?.focus?.({ preventScroll: true });
        if (!form.cards.size) forms.delete(form.key);
    }

    // Saved or cancelled, here or in another window: the form has nothing left to save.
    // The request's question on a connected card (the Main mirror or the Project's own): where focus goes
    // when the form that held it is gone.
    function focusQuestion(form) {
        [...form.cards].map((card) => card.querySelector?.('.chat-quiz-question')).find((node) => node?.isConnected)
            ?.focus?.({ preventScroll: true });
    }

    function settle(form) {
        if (!form.dialog) return;
        const pendingHere = form.pending;
        const owned = form.dialog.panel.contains(globalThis.document?.activeElement);
        closeDialog(form, { restoreFocus: false });
        if (owned) focusQuestion(form);  // a form that never held focus never takes it elsewhere
        if (!pendingHere) showToast(tr('secret.settled_elsewhere', 'This secret request was completed; the form closed.'));
    }

    // A disabled Save drops focus to the page while the save runs; give it back inside the form.
    function keepFocus(form, previous) {
        const dialog = form.dialog;
        const active = globalThis.document?.activeElement;
        if (!dialog || (active && dialog.panel.contains(active))) return;
        const target = [previous, dialog.check, dialog.inputs[0]].find((node) => node && dialog.panel.contains(node)
            && !node.disabled && !node.hidden);
        target?.focus?.({ preventScroll: true });
    }

    // Settle a save whose reply (or timeout) arrived. `attempt` is this press; a reply that outlived it
    // (a newer press, a closed or reopened form) may still settle the request, but never rewrites a form
    // it was not made for.
    function receipt(form, attempt, { body, failure }, previous) {
        if (disposed) return;
        const owner = form.attempt === attempt;
        if (owner) { form.attempt = null; form.pending = false; }
        if (!failure && body?.ok === true && body.saved === true && body.state === 'answered') {
            const owned = Boolean(form.dialog?.panel.contains(globalThis.document?.activeElement));
            closeDialog(form, { restoreFocus: false });
            acceptAll(form, body);
            if (owned) focusQuestion(form);
            return;
        }
        const definite = failure && body && failure.status && body.saved === false;
        if (definite && !OPEN_STATES.includes(body.state || 'open')) {
            closeDialog(form, { restoreFocus: false });
            acceptAll(form, body);
            showToast(tr(...(REFUSAL[body.reason_code] || REFUSED)), 'error');
            return;
        }
        const here = form.dialog === attempt.dialog && Boolean(form.dialog);
        if (!owner || !here || (attempt.late && form.dialog.read !== attempt.read)) {
            // Obsolete for this form (or a status read began since it was released): its facts are older.
            if (owner) repaintCards(form);
            return;
        }
        if (!definite) {
            // No readable reply, a written-but-unconfirmed save, or saved null: the outcome is unknown.
            form.unknown = true;
            form.seenUnconfirmed = false;
            setMessage(form, tr(...(UNKNOWN[body?.reason_code] || UNCONFIRMED)), 'warn');
            repaintCards(form);
            keepFocus(form, previous);
            return;
        }
        form.unknown = false;  // a definite refusal ends the doubt that releasing a stalled wait left
        takeFacts(form, body);
        if (body.reason_code === 'secret_key_state_changed') {
            setMessage(form, replacing(form)
                ? tr('secret.error_now_set', 'This key got a value in Settings after the form opened, so your entry was not saved. Replace overwrites that value.')
                : tr('secret.error_now_cleared', 'This key was cleared in Settings after the form opened, so your entry was not saved. Save fills it.'), 'warn');
        } else setMessage(form, tr(...(REFUSAL[body.reason_code] || REFUSED)));
        repaintCards(form);
        keepFocus(form, previous);
    }

    async function submit(form) {
        if (form.pending || form.unknown || !form.dialog || !filled(form) || !usable(form) || !form.facts?.keys) return;
        const previous = globalThis.document?.activeElement;
        const values = Object.fromEntries(form.dialog.inputs.map((input) => [input.dataset.secretField, input.value]));
        const observed = Object.fromEntries((form.facts.keys || []).map((row) => [row.id, Boolean(row.configured)]));
        if (!form.requestId) form.requestId = nonce();
        const attempt = { dialog: form.dialog, late: false, read: null };
        form.attempt = attempt;
        form.pending = true;
        setMessage(form, '');
        repaintCards(form);
        const call = Promise.resolve().then(() => apiClient.saveSecretToken({ request_id: form.requestId,
            decision_id: decisionOf(form), values, observed }, transport)).then(
            (body) => ({ body, failure: null }),
            (error) => ({ body: error?.body && typeof error.body === 'object' ? error.body : null, failure: error || {} }),
        ).finally(() => { Object.keys(values).forEach((key) => { values[key] = ''; }); });
        const first = await bounded(form, call, waitMs);
        if (first !== LATE) { receipt(form, attempt, first, previous); return; }
        if (disposed) return;
        // Released, not failed: the call keeps running and nothing is resent. The save is unconfirmed
        // until its own reply or a status read says otherwise.
        if (form.attempt === attempt) {
            attempt.late = true;
            attempt.read = form.dialog?.read || null;
            form.pending = false;
            form.unknown = true;
            form.seenUnconfirmed = false;
            setMessage(form, tr(...UNCONFIRMED), 'warn');
            repaintCards(form);
            keepFocus(form, previous);
        }
        receipt(form, attempt, await call, previous);
    }

    // Status first: only a fresh read decides what the next Save may do.
    async function checkStatus(form) {
        if (form.pending || !form.dialog) return;
        await refreshStatus(form);
    }

    async function cancelRequest(form) {
        if (form.pending || !usable(form)) return;
        const attempt = { cancel: true };
        form.attempt = attempt;
        form.pending = true;
        repaintCards(form);
        const call = Promise.resolve().then(() => apiClient.cancelSecretRequest({ request_id: nonce(),
            decision_id: decisionOf(form) }, transport)).then((body) => ({ body, failed: false }),
            (error) => ({ body: error?.body && typeof error.body === 'object' ? error.body : null, failed: true }));
        let first = await bounded(form, call, waitMs);
        if (disposed) return;
        let notified = false;
        if (first === LATE) {
            // Released, not failed: a late reply still settles the request; Cancel may be pressed again.
            if (form.attempt === attempt) {
                form.attempt = null;
                form.pending = false;
                repaintCards(form);
            }
            showToast(tr('secret.cancel_unconfirmed', 'The cancellation was not confirmed. Check the request before trying again.'), 'error');
            notified = true;
            first = await call;
            if (disposed) return;
        }
        if (form.attempt === attempt) { form.attempt = null; form.pending = false; }
        const { body, failed } = first;
        if (body?.state) {
            closeDialog(form, { restoreFocus: false });
            acceptAll(form, body);
        } else {
            if (failed && !notified) showToast(tr('secret.cancel_unconfirmed', 'The cancellation was not confirmed. Check the request before trying again.'), 'error');
            repaintCards(form);
        }
    }

    function release(card) {
        const view = views.get(card);
        if (!view) return;
        views.delete(card);
        view.form.cards.delete(card);
        // An open form outlives its card node: a re-rendered card of the same request takes it back.
        if (!view.form.cards.size && !view.form.dialog) forms.delete(view.form.key);
    }

    function dispose() {
        disposed = true;
        for (const form of forms.values()) {
            [...form.timers].forEach((stop) => stop());
            closeDialog(form, { restoreFocus: false });
        }
        forms.clear();
        views.clear();
    }

    return { mount, sync, release, dispose, _views: views, _forms: forms };
}
