// Dialogs of the Skills page. Plain confirmations go through the shared
// openConfirmDialog; a choice between consequences is drawn as offered-answer
// cards (the question card's recipe), one chosen, the confirm button's words and
// tone following the chosen card. Cancel, Close, the backdrop, Escape and a newer
// dialog never confirm. Add source is the one two-step flow (address, then what
// was found with the one-time trust choice).
import { openConfirmDialog } from './confirm_dialog.js';
import { skillSourcesApi } from './api_client.js';
import { bindDialogFocus, bindMenu } from './ui_interactions.js';
import { escapeHtmlAttr as escapeHtml } from './ui_primitives.js';
import {
    canUpdateWithoutReview, displayAddress, officialPromise, permissionWords, plural, publicationDisclosure, updateRejectionSentence, versionText,
} from './skill_presentation.js';

let activeClose = null;

/** Mount one modal on the shared dialog chrome; `close(value)` settles it once. */
function mountDialog({ title, bodyHtml = '', wide = false, onSettle }) {
    if (activeClose) activeClose();
    const backdrop = document.createElement('div');
    backdrop.className = 'marketplace-modal-backdrop confirm-dialog-backdrop';
    backdrop.innerHTML = `
        <div class="marketplace-modal confirm-dialog skills-dialog${wide ? ' is-wide' : ''}" role="dialog" aria-modal="true" aria-labelledby="skills-dialog-title">
            <div class="marketplace-modal-head">
                <h3 id="skills-dialog-title">${escapeHtml(title)}</h3>
                <button type="button" class="btn btn-default btn-sm" data-dialog-cancel aria-label="Close">Close</button>
            </div>
            <div class="marketplace-modal-body">${bodyHtml}</div>
            <div class="marketplace-modal-actions"></div>
        </div>`;
    let settled = false;
    let disposeFocus = null;
    const dialog = backdrop.querySelector('[role="dialog"]');
    const close = (value) => {
        if (settled) return;
        settled = true;
        if (activeClose === cancel) activeClose = null;
        disposeFocus?.();
        backdrop.remove();
        onSettle(value);
    };
    const cancel = () => close(null);
    backdrop.addEventListener('click', (event) => {
        if (event.target === backdrop || event.target.closest('[data-dialog-cancel]')) cancel();
    });
    document.body.appendChild(backdrop);
    activeClose = cancel;
    return {
        backdrop, dialog, close, cancel,
        body: dialog.querySelector('.marketplace-modal-body'),
        actions: dialog.querySelector('.marketplace-modal-actions'),
        focus(initial) { disposeFocus?.({ restoreFocus: false }); disposeFocus = bindDialogFocus(dialog, { initialFocus: initial, onEscape: cancel }); },
    };
}

/** Offered-answer cards as one radiogroup: arrows move the choice, the chosen card takes the green tone. */
export function choiceCardsHtml(choices, selected, { label = 'Choices' } = {}) {
    return `<div class="chat-quiz-options skills-choice-cards" role="radiogroup" aria-label="${escapeHtml(label)}">${choices.map((choice) => {
        const chosen = choice.value === selected;
        return `<button type="button" class="chat-quiz-option${chosen ? ' chosen' : ''}" role="radio" aria-checked="${chosen ? 'true' : 'false'}" tabindex="${chosen ? '0' : '-1'}" data-choice="${escapeHtml(choice.value)}">
            <span class="chat-quiz-option-label">${escapeHtml(choice.label)}</span>${choice.detail ? `<span class="chat-quiz-option-detail">${escapeHtml(choice.detail)}</span>` : ''}</button>`;
    }).join('')}</div>`;
}

/** Bind a rendered radiogroup; returns the current value getter. */
export function bindChoiceCards(group, onChange = () => {}) {
    const cards = () => Array.from(group.querySelectorAll('[data-choice]'));
    let value = cards().find((card) => card.getAttribute('aria-checked') === 'true')?.dataset.choice || '';
    const select = (card, focus = false) => {
        if (!card) return;
        value = card.dataset.choice;
        for (const item of cards()) {
            const chosen = item === card;
            item.classList.toggle('chosen', chosen);
            item.setAttribute('aria-checked', chosen ? 'true' : 'false');
            item.tabIndex = chosen ? 0 : -1;
        }
        if (focus) card.focus();
        onChange(value);
    };
    group.addEventListener('click', (event) => select(event.target.closest('[data-choice]')));
    group.addEventListener('keydown', (event) => {
        const list = cards();
        const index = list.indexOf(event.target.closest('[data-choice]'));
        if (index < 0) return;
        const step = { ArrowDown: 1, ArrowRight: 1, ArrowUp: -1, ArrowLeft: -1 }[event.key];
        if (step) {
            event.preventDefault();
            select(list[(index + step + list.length) % list.length], true);
        } else if (event.key === 'Home' || event.key === 'End') {
            event.preventDefault();
            select(event.key === 'Home' ? list[0] : list.at(-1), true);
        }
    });
    return () => value;
}

/**
 * A choice between consequences. choices: [{value, label, detail, confirmLabel, tone}].
 * Resolves the chosen value, or null when cancelled in any way.
 */
export function openChoiceDialog({ title, lead = '', choices = [], selected = choices[0]?.value, note = '', footHtml = '', cancelLabel = 'Cancel', groupLabel = title }) {
    return new Promise((resolve) => {
        const host = mountDialog({
            title, wide: true, onSettle: resolve,
            bodyHtml: `${lead ? `<p>${escapeHtml(lead)}</p>` : ''}${choiceCardsHtml(choices, selected, { label: groupLabel })}${note ? `<p class="skills-dialog-note">${escapeHtml(note)}</p>` : ''}${footHtml}`,
        });
        host.actions.innerHTML = `<button type="button" class="btn btn-default" data-dialog-cancel>${escapeHtml(cancelLabel)}</button><button type="button" class="btn btn-primary" data-dialog-ok></button>`;
        const ok = host.actions.querySelector('[data-dialog-ok]');
        const paint = (value) => {
            const choice = choices.find((item) => item.value === value) || choices[0];
            ok.textContent = choice?.confirmLabel || 'Continue';
            ok.className = `btn btn-${choice?.tone === 'danger' ? 'danger' : choice?.tone === 'default' ? 'default' : 'primary'}`;
        };
        const current = bindChoiceCards(host.body.querySelector('[role="radiogroup"]'), paint);
        paint(current());
        ok.addEventListener('click', () => host.close(current()));
        host.focus(() => host.body.querySelector('[aria-checked="true"]'));
    });
}

/** Skip review: the owner's attestation of these exact files (B5). */
export function openSkipReviewDialog({ name, sourceName = '', owner = '', permissions = [], label = 'Skip review' } = {}) {
    const from = sourceName ? ` It comes from ${sourceName}${owner ? `, by ${owner}` : ''}.` : '';
    const uses = permissionWords(permissions);
    const turnOn = label !== 'Skip review';
    return openConfirmDialog({
        title: turnOn ? `Turn on ${name} anyway?` : `Skip review for ${name}?`,
        body: `${turnOn ? `The review found problems in ${name}. It will run with them` : `${name} will run without a review`}.${from}${uses.length ? ` It can use: ${uses.join(', ').toLowerCase()}, the keys you allow.` : ''} ${turnOn ? 'Turn it on only' : 'Skip only'} if you trust the author. Your choice is recorded.`,
        confirmLabel: turnOn ? 'Turn on anyway' : 'Skip review',
        cancelLabel: turnOn ? 'Keep it off' : 'Keep reviewing',
        danger: true,
    });
}

/**
 * Update a skill that was changed here, or whose installed original is unknown (B6,
 * Astra A03): merge, replace or keep. `editedByOwner`: the host recorded the
 * owner's edit of these files. Resolves the mode or null.
 */
export function openUpdateDialog({ name, sourceName = '', update = {}, installedVersion = '', currentVersion = '', baseKnown = true, editedByOwner = false }) {
    const v = versionText(update.version) || 'the new version';
    const differ = `differ from ${versionText(installedVersion) || 'the version'} you installed${currentVersion && currentVersion !== installedVersion ? ` (your copy says ${versionText(currentVersion)})` : ''}`;
    const lead = !baseKnown
        ? `The original files of ${name} aren’t known, so Ouroboros can’t tell whether this copy was changed here.`
        : editedByOwner ? `You changed ${name} on this computer: its files ${differ}.`
            : `${name} was changed on this computer: its files ${differ}.`;
    return openChoiceDialog({
        title: `Update ${name} to ${v}?`,
        lead,
        groupLabel: 'What to keep',
        choices: [
            { value: 'merge', label: 'Ask Ouroboros to update and keep my changes', confirmLabel: 'Ask Ouroboros', tone: 'primary',
                detail: baseKnown
                    ? `Ouroboros installs ${v}, re-applies your changes where they still fit and reviews the result. It tells you in chat what it could not keep.`
                    : `Ouroboros compares your copy with ${v} and carries over what it can, then reviews the result. It tells you in chat what it could not keep.` },
            { value: 'replace', label: 'Update and replace my changes', confirmLabel: 'Replace my changes', tone: 'danger',
                detail: `You get ${v} exactly as published${sourceName ? ` in ${sourceName}` : ''}. Your changed version is kept as a backup you can restore.` },
            // The host records the skipped release: this one is not offered again, a newer one is.
            { value: 'keep', label: 'Keep my version', confirmLabel: 'Keep my version', tone: 'default',
                detail: `Stay on your changed copy. ${versionText(update.version) || 'This version'} isn’t offered again; a newer version will be.` },
        ],
    });
}

/** The rejected release, not a newer catalog candidate, is what the owner confirms. */
export async function updateWithoutReview(skill, { dialog = openConfirmDialog, update = skillSourcesApi.updateSkill } = {}) {
    const release = skill.install?.update_rejected;
    if (!canUpdateWithoutReview(skill.install)) return null;
    const confirmed = await dialog({
        title: `Update ${skill.name} to ${release.version} without review?`,
        body: `${updateRejectionSentence(release)} This replaces your current copy with that exact version and lets it run without review. Your choice is recorded.`,
        confirmLabel: 'Update without review', cancelLabel: 'Keep the previous version', danger: true,
    });
    if (confirmed !== true) return null;
    return update(skill.name, 'replace', release.digest, '', { skipReview: true });
}

/** Two skills with one name (B5b): keep mine or use the other. Resolves 'keep' | 'replace' | null. */
export function openCompareDialog({ mine = {}, other = {} }) {
    const line = (skill) => [skill.name, skill.official ? 'Official' : '', skill.source, skill.owner ? `by ${skill.owner}` : '',
        skill.version ? versionText(skill.version) : '', skill.state || '', skill.popularity || ''].filter(Boolean).join(' · ');
    return openChoiceDialog({
        title: `Two skills are called “${mine.name || other.name}”`,
        lead: 'You can have one skill with this name. Which one should Ouroboros use?',
        groupLabel: 'Which skill',
        choices: [
            { value: 'keep', label: 'Keep yours', detail: `${line(mine)}\n${mine.description || ''}`, confirmLabel: 'Keep mine', tone: 'primary' },
            { value: 'replace', label: 'Use this one instead', detail: `${line(other)}\n${other.description || ''}`, confirmLabel: 'Use this one', tone: 'danger' },
        ],
        // The host replaces in one install and keeps the replaced copy as its backup.
        note: `Keeping yours changes nothing. Using the other one replaces yours and keeps a backup of it${other.reviewed ? '; the new one is reviewed before it runs' : ''}.`,
    });
}

/** Where a skill is published (B10). Resolves the selected choice, including its visibility, or null. */
export function openPublishDestinationDialog({ name, choices = [], changed = false, createMySkills = () => skillSourcesApi.mySkills() }) {
    return new Promise((resolve) => {
        const host = mountDialog({ title: `Publish ${name}`, wide: true, onSettle: resolve });
        const destinations = choices.length ? [...choices] : [{ value: 'default', private: false, label: 'OuroborosHub', detail: 'Share it in the public catalog.' }];
        let selected = destinations[0].value;
        let connected = null;
        const current = () => destinations.find((choice) => choice.value === selected);
        const paint = (error = '') => {
            host.body.innerHTML = `<p>${changed ? 'Where should your changed version go?' : 'Where should it go?'}</p>
                ${choiceCardsHtml(destinations, selected, { label: 'Destination' })}
                ${connected ? `<p class="ui-status" data-tone="ok" role="status" data-my-skills-status>${escapeHtml(connected.status)}</p>` : '<button type="button" class="btn btn-default" data-create-my-skills>Create or connect «My skills» on GitHub</button>'}
                ${error ? `<p class="skills-dialog-note" role="alert">${escapeHtml(error)}</p>` : ''}
                <p class="skills-dialog-note" data-publication-disclosure></p>
                <p class="skills-dialog-note">Ouroboros first checks it for secrets and problems.</p>`;
            host.actions.innerHTML = '<button type="button" class="btn btn-default" data-dialog-cancel>Cancel</button><button type="button" class="btn btn-primary" data-dialog-ok>Check and publish</button>';
            const disclose = () => { host.body.querySelector('[data-publication-disclosure]').textContent = publicationDisclosure(current()?.private, current()?.repository); };
            bindChoiceCards(host.body.querySelector('[role="radiogroup"]'), (value) => { selected = value; disclose(); });
            disclose();
            host.actions.querySelector('[data-dialog-ok]').addEventListener('click', () => host.close(current()));
            host.body.querySelector('[data-create-my-skills]')?.addEventListener('click', async (event) => {
                event.currentTarget.disabled = true;
                event.currentTarget.textContent = 'Connecting My skills…';
                host.actions.querySelector('[data-dialog-ok]').disabled = true;
                try {
                    const choice = await createMySkillsDestination(createMySkills);
                    if (!host.dialog.isConnected) return;
                    const index = destinations.findIndex((item) => item.value === choice.value);
                    if (index < 0) destinations.push(choice); else destinations[index] = choice;
                    selected = choice.value;
                    connected = choice;
                    paint();
                } catch (err) {
                    if (host.dialog.isConnected) paint(err.message);
                }
            });
            host.focus(() => host.body.querySelector('[aria-checked="true"]'));
        };
        paint();
    });
}

/** Translate the route's typed outcome; a successful connect does not publish any files. */
export async function createMySkillsDestination(create) {
    try {
        const answer = await create();
        if (answer?.ok !== true || !answer.source?.id) throw { body: answer };
        const visibility = answer.private === true ? ' (private)' : answer.private === false ? ' (public)' : '';
        return { value: `source:${answer.source.id}`, private: typeof answer.private === 'boolean' ? answer.private : undefined,
            repository: answer.repository, label: `${answer.source.name} — ${answer.repository}${visibility}`,
            status: `${answer.created ? 'Created' : 'Connected'} ${answer.repository}${visibility}.`,
            detail: 'Your other Ouroboros installs can read it when you connect this source.' };
    } catch (error) {
        const code = error?.body?.code;
        throw new Error({
            no_token: 'Add GITHUB_TOKEN in Settings → Secrets, then try creating My skills again.',
            cannot_create: 'GitHub couldn’t create or connect My skills; allow repository access for GITHUB_TOKEN in Settings → Secrets, then try again.',
            failed: 'My skills couldn’t be connected; check your GitHub connection and try again.',
        }[code] || 'My skills couldn’t be connected; check your GitHub connection and try again.');
    }
}

/** Change a source's trust later (UX §6.3). Resolves 'trusted' | 'review' | null. */
export function openTrustDialog(source, { installedCount = 0 } = {}) {
    const trusted = source.trust === 'trusted';
    return openChoiceDialog({
        title: `How should skills from ${source.name} install?`,
        lead: installedCount
            ? `This applies to new installs and updates. The ${plural(installedCount, 'skill')} you already installed from it keep running as they are.`
            : 'This applies to new installs and updates.',
        groupLabel: 'Trust',
        selected: source.trust === 'trusted' ? 'trusted' : 'review',
        choices: trustChoices(source.name).map((choice) => ({
            ...choice,
            confirmLabel: choice.value === source.trust ? 'Keep it' : (choice.value === 'trusted' ? `Trust ${source.name}` : 'Review from now on'),
            tone: choice.value === 'trusted' && !trusted ? 'danger' : 'primary',
        })),
    });
}

/** The two trust cards shared by Add source, the trust dialog and first start (one intent, one control). */
export function trustChoices(name = 'this source') {
    return [
        { value: 'review', label: 'Review each skill before it runs',
            detail: 'Ouroboros reads each skill and records what it found before the skill runs or gets your keys. You can still skip a review. A review sends the skill’s files to your review models.' },
        { value: 'trusted', label: `Trust ${name === 'this source' ? 'this source' : name} — install without review`,
            detail: 'Skills install at once and run without a review — including future updates and skills added later. Choose this only for a source you or your team control. While automatic keys are on, they get the keys they ask for.' },
    ];
}

export function openRenameSourceDialog(source) {
    return openConfirmDialog({ title: `Rename ${source.name}`, body: 'The name is shown on its skills’ cards.',
        input: true, initialValue: source.name, confirmLabel: 'Rename' });
}

export function openRemoveSourceDialog(source, installedNames = []) {
    const some = installedNames.length;
    return openConfirmDialog({
        title: `Remove ${source.name}?`,
        body: some
            ? `You have ${plural(some, 'skill')} from it: ${installedNames.join(', ')}. They keep working without updates; uninstall any of them from Installed.${source.kind === 'hub' ? ' You can add this source back any time with Add source.' : ''}`
            : `Its skills disappear from Browse.${source.kind === 'hub' ? ' You can add this source back any time with Add source.' : ''}`,
        confirmLabel: 'Remove source', danger: true,
    });
}

export function sourceProbeWarning(probe) {
    return probe?.warning === 'not_encrypted'
        ? 'This address is not encrypted: anyone on the network path could change what you install.' : '';
}

/**
 * Add source, step 1 (address) then step 2 (what was found, its name, the
 * one-time trust choice). `api` = {probe, add}. Resolves the added source or null.
 */
export function openAddSourceDialog({ api, initialAddress = '' } = {}) {
    return new Promise((resolve) => {
        const host = mountDialog({ title: 'Add a source', wide: true, onSettle: resolve });
        let probe = null;
        let address = initialAddress;
        let busy = false;
        const targetOf = (value) => (/^(https?:\/\/|git@|ssh:\/\/)/i.test(value) ? { url: value } : { path: value });
        const stepOne = (error = '') => {
            host.body.innerHTML = `
                <p>Paste a link to a skills repository or catalog, or a folder.</p>
                <label class="ui-field skills-add-field"><span>Address</span>
                    <input class="ui-control" type="text" data-source-address autocomplete="off" spellcheck="false" value="${escapeHtml(address)}" aria-describedby="skills-add-help">
                </label>
                <p class="skills-dialog-note" id="skills-add-help">GitHub, GitLab or any git address. A private repository uses your Git sign-in on this computer. A folder on the computer running Ouroboros works too.</p>
                ${error ? `<p class="ui-status skills-dialog-error" data-tone="danger" role="alert">${escapeHtml(error)}</p>` : ''}`;
            host.actions.innerHTML = '<button type="button" class="btn btn-default" data-dialog-cancel>Cancel</button><button type="button" class="btn btn-primary" data-add-check>Check</button>';
            const input = host.body.querySelector('[data-source-address]');
            const check = async () => {
                address = input.value.trim();
                if (!address || busy) return;
                busy = true;
                const button = host.actions.querySelector('[data-add-check]');
                button.disabled = true;
                button.textContent = 'Checking…';
                try {
                    probe = await api.probe(targetOf(address));
                    if (probe?.error || !Array.isArray(probe?.found) || !probe.found.length) {
                        busy = false;
                        return stepOne(probe?.error || 'No skills found here. A source is a repository with skill folders (each with a SKILL.md) or a catalog file.');
                    }
                    busy = false;
                    stepTwo();
                } catch (err) {
                    busy = false;
                    stepOne(`Couldn’t reach this address: ${err?.message || err}`);
                }
            };
            host.actions.querySelector('[data-add-check]').addEventListener('click', check);
            input.addEventListener('keydown', (event) => { if (event.key === 'Enter' && !event.isComposing) { event.preventDefault(); check(); } });
            host.focus(input);
        };
        const stepTwo = (error = '') => {
            const found = probe.found.slice(0, 12).join(', ') + (probe.found.length > 12 ? ` … +${probe.found.length - 12}` : '');
            const named = probe.suggested_name || displayAddress(address).split('/').filter(Boolean).at(-1) || 'New source';
            host.body.innerHTML = `
                <p class="ui-status" data-tone="ok">Found ${escapeHtml(plural(probe.found.length, 'skill'))}${probe.suggested_name ? ` — “${escapeHtml(probe.suggested_name)}”` : ''}${probe.owner ? ` by ${escapeHtml(probe.owner)}` : ''}</p>
                <p class="skills-dialog-note">${escapeHtml(found)}</p>
                ${sourceProbeWarning(probe) ? `<p class="skills-dialog-note" data-source-warning>${sourceProbeWarning(probe)}</p>` : ''}
                <label class="ui-field skills-add-field"><span>Name</span>
                    <input class="ui-control" type="text" data-source-name value="${escapeHtml(named)}" aria-describedby="skills-add-name-help">
                </label>
                <p class="skills-dialog-note" id="skills-add-name-help">Shown on its skills’ cards.</p>
                <p class="skills-dialog-lead">How should skills from this source be installed?</p>
                ${choiceCardsHtml(trustChoices(), 'review', { label: 'Trust' })}
                <p class="skills-dialog-note">You choose once. You can change it later in Sources.</p>
                ${error ? `<p class="ui-status skills-dialog-error" data-tone="danger" role="alert">${escapeHtml(error)}</p>` : ''}`;
            host.actions.innerHTML = '<button type="button" class="btn btn-default" data-add-back>Back</button><button type="button" class="btn btn-primary" data-add-ok>Add source</button>';
            const trust = bindChoiceCards(host.body.querySelector('[role="radiogroup"]'));
            host.actions.querySelector('[data-add-back]').addEventListener('click', () => stepOne());
            host.actions.querySelector('[data-add-ok]').addEventListener('click', async () => {
                const name = host.body.querySelector('[data-source-name]').value.trim();
                if (!name || busy) return;
                busy = true;
                try {
                    const result = await api.add({ ...targetOf(address), name, trust: trust() });
                    busy = false;
                    host.close(result?.source || result || { name });
                } catch (err) {
                    busy = false;
                    stepTwo(err?.message || String(err));
                }
            });
            host.focus(host.body.querySelector('[data-source-name]'));
        };
        stepOne();
    });
}

/** Suggested trust choices and their consequence, before the owner confirms. */
export function firstStartTrustHtml(rows = []) {
    const choice = (row) => `<div class="skills-first-row" data-row="${escapeHtml(row.id)}">
                <div class="skills-first-name"><strong>${escapeHtml(row.name)}</strong>${row.markHtml || ''}<span class="skills-first-about">${escapeHtml(row.about || '')}</span></div>
                <div class="ui-segment-group skills-trust-segments" role="radiogroup" aria-label="${escapeHtml(`How skills from ${row.name} install`)}">
                    ${[['trusted', 'Install without review'], ['review', 'Review each skill']].map(([value, label]) => `<button type="button" class="ui-segment${row.trust === value ? ' active' : ''}" role="radio" aria-checked="${row.trust === value ? 'true' : 'false'}" data-trust="${value}">${label}</button>`).join('')}
                </div></div>`;
    return `<p>Ouroboros comes connected to its public hub. Confirm how its skills install — you do this once.</p>
        <div class="skills-first-rows">${rows.map(choice).join('')}</div>
        <p class="skills-dialog-note">Suggested choices apply until you choose. Official doesn’t skip the review by itself: if you choose “Review each skill” for Ouroboros, its Official skills are reviewed too. A review sends the skill’s files to your review models.</p>
        <p class="skills-dialog-note">You choose once. You can change it later in Skills → Sources.</p>`;
}

/** First start of the public hub (B3). Resolves {id: trust} or null (Later). */
export function openFirstStartDialog(rows = []) {
    return new Promise((resolve) => {
        const host = mountDialog({
            title: 'How should skills from the Ouroboros hub install?', wide: true, onSettle: resolve,
            bodyHtml: firstStartTrustHtml(rows),
        });
        host.actions.innerHTML = '<button type="button" class="btn btn-default" data-dialog-cancel>Later</button><button type="button" class="btn btn-primary" data-first-ok>Confirm</button>';
        host.body.addEventListener('click', (event) => {
            const button = event.target.closest('[data-trust]');
            if (!button) return;
            for (const sibling of button.parentElement.querySelectorAll('[data-trust]')) {
                const on = sibling === button;
                sibling.classList.toggle('active', on);
                sibling.setAttribute('aria-checked', on ? 'true' : 'false');
            }
        });
        host.actions.querySelector('[data-first-ok]').addEventListener('click', () => {
            const out = {};
            for (const row of host.body.querySelectorAll('[data-row]')) {
                out[row.dataset.row] = row.querySelector('[data-trust].active')?.dataset.trust || 'review';
            }
            host.close(out);
        });
        host.focus(host.actions.querySelector('[data-first-ok]'));
    });
}

/** The Official mark's promise, opened by touch or keyboard next to the mark (no hover). */
export function openOfficialExplanation(anchor, source = null) {
    const popup = document.createElement('div');
    popup.className = 'ui-popup skills-mark-popup';
    popup.setAttribute('role', 'dialog');
    popup.setAttribute('aria-label', 'Official');
    popup.innerHTML = `<p>${escapeHtml(officialPromise(source))}</p><button type="button" class="btn btn-default btn-sm" role="menuitem" data-popup-close>Close</button>`;
    document.body.appendChild(popup);
    const binding = bindMenu(popup, { anchor, onClose: () => popup.remove() });
    popup.querySelector('[data-popup-close]').addEventListener('click', () => binding.close({ restoreFocus: true }));
    return binding;
}
