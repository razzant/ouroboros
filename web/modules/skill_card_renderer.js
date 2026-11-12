import { tr } from './i18n.js';
// Installed skill cards (Skills → Installed): one head shared with Browse and the
// detail sheet (name that opens the sheet, marks, purpose, who line), the switch,
// with a chip beside it only for a state the switch cannot show, Details right
// under the head (a phone keeps the switch above it), and at most two attention trays,
// each one sentence and its actions — the owner's next step is visible text on
// every surface, never a hover title. Review history, findings, access and the
// install record live in Details; a connector's place lives on its sheet, not here.
import {
    boundedText,
    escapeHtmlAttr as escapeHtml,
    grantReady,
    preflightFailed,
    preflightFailedStale,
    preflightFindingText,
    renderSubmissionHistory,
    reviewReady,
    safeExternalHrefAttr as safeExternalUrl,
} from './utils.js';
import { installedTime } from './ui_helpers.js';
import { HELD_LINES } from './holds.js';
import { hasSemanticVerdict, lifecycleMeta } from './review_presentation.js';
import {
    buttonHtml, canUpdateWithoutReview, chipHtml, connectsServices, connectsText, countsAsYoursText, grantPhrases, officialMarkHtml, permissionWords,
    plural, relativeAge, shortDate, skillIconHtml, skipReviewButtonHtml, trayHtml, updateRejectionSentence, versionText,
    skillTitle,
} from './skill_presentation.js';

function hasSkillUiTab(skill, live = {}) {
    return (live?.ui_tabs || []).some((tab) => (tab?.skill || tab?.skill_name || tab?.extension || '') === skill.name);
}
/** Whether the skill registered a settings form: its card then opens the skill's page in Settings. */
const hasSkillSettings = (skill, live = {}) => (live?.settings_sections || []).some((section) => section?.skill === skill.name);

const sourceTag = (skill) => String(skill?.source || 'native').toLowerCase();
const installFacts = (skill) => (skill?.install && typeof skill.install === 'object' ? skill.install : null);

/** Who a skill came from, in the owner's words. */
export function installedSourceLabel(skill) {
    const install = installFacts(skill);
    const named = install?.source?.name || install?.source_name;
    if (named) return String(named);
    const tag = sourceTag(skill);
    if (tag === 'native') return 'Built in';
    if (tag === 'clawhub') return 'ClawHub';
    if (tag === 'ouroboroshub') return 'OuroborosHub';
    if (tag === 'user_repo') return 'Your skills folder';
    return 'Made here';
}

export const isBuiltIn = (skill) => installFacts(skill)?.source?.kind === 'builtin' || sourceTag(skill) === 'native';
/** Created on this computer (by the owner or Ouroboros), not installed from a source. */
export const isMadeHere = (skill) => !installFacts(skill) && ['external', 'self_authored'].includes(sourceTag(skill));

// Host reasons that admit these bytes without a critic PASS (fixture parity tests pin this list).
export const INSTALL_ACCEPTANCE_REASONS = ['trusted_source', 'owner_skip', 'owner_edit', 'self_check'];
const acceptedAs = (skill) => String(skill?.review_gate?.blocking_reason || '');

// The files differ from the installed release (UX §5.3: a neutral fact). Who changed
// them is known only when the owner's edit is recorded for these exact bytes.
function modifiedFacts(skill) {
    if (!installFacts(skill)?.modified) return null;
    return { by: acceptedAs(skill) === 'owner_edit' ? 'owner' : '' };
}

function statusBadge(status, gate = null, profile = '') {
    // Owner-attested skills are executable but the LLM review was skipped; the
    // verdict word stays distinct from a reviewed one. It was the owner's own
    // choice, so it is a neutral fact, not a move still waiting for the owner.
    if (profile === 'owner_attested') return chipHtml('review skipped by you', 'neutral');
    // No verdict yet is no chip: the sentence says what runs and whether a review is pending.
    if ((status || 'pending') === 'pending') return '';
    const executable = gate && typeof gate.executable_review === 'boolean'
        ? gate.executable_review
        : ['clean', 'warnings'].includes(status);
    const tone = status === 'blockers' ? 'danger' : executable ? 'ok' : status === 'warnings' ? 'warn' : 'neutral';
    return chipHtml(status || 'pending', tone);
}

// Declared secret fields as the listing projects them (configured / missing / unknown; never a value), on an enabled
// skill whose grant covers them. A missing one asks the owner to save it, so the same condition puts the card in Needs
// you; an unreadable one asks nothing. When the skill applies a saved value is its own status.
function secretConfigTray(skill) {
    const fields = Array.isArray(skill?.secret_config?.fields) ? skill.secret_config.fields : [];
    const keys = (state) => fields.filter((field) => field?.state === state).map((field) => String(field.key || ''));
    const missing = keys('missing');
    const unknown = keys('unknown');
    if (!skill.enabled || !grantReady(skill) || (!missing.length && !unknown.length)) return { tray: '', needs: false };
    const sentence = missing.length
        ? `${escapeHtml(missing.join(', '))} ${missing.length > 1 ? 'are' : 'is'} not set.`
        : `Saved settings can’t be read now, so whether ${escapeHtml(unknown.join(', '))} ${unknown.length > 1 ? 'are' : 'is'} set is unknown.`;
    // An unknown value is not a reason to save again.
    const note = missing.length ? `Save ${missing.length > 1 ? 'them' : 'it'} in Settings › Accounts and keys.` : '';
    // The button names the keys it is about: Accounts and keys opens at the first one's row.
    const tray = trayHtml({ sentence, note, actions: smallButton('Open Accounts and keys', String(skill.name || ''), 'open_secrets',
        `data-keys="${escapeHtml((missing.length ? missing : unknown).join(','))}"`) });
    return { tray, needs: missing.length > 0 };
}

function missingGrantLoadError(skill) {
    return !grantReady(skill) && String(skill.load_error || '').includes('missing owner grants');
}

// An ON skill whose packages are missing, failed or out of date: Enable installs them (it is
// idempotent); a declaration mismatch (deps_declaration_desync) stays on the repair path.
const PACKAGES_INSTALLABLE = ['deps_missing', 'deps_failed', 'deps_stale'];
const packagesInstallable = (skill) => PACKAGES_INSTALLABLE.includes(String(skill?.live_reason || ''));

// A package the organization requires is its administrator's to change: no removal, update or repair is offered.
const requiredBy = (skill) => (skill?.deployment && typeof skill.deployment === 'object' ? skill.deployment : null);
// An extension under a deployment's execution domain: whether the service runs its code (`trust`), and whether its
// name is the administrator's, so its doors refuse every change but turning it off.
const trustOf = (skill) => (skill?.trust && typeof skill.trust === 'object' ? skill.trust : null);
const untrusted = (skill) => trustOf(skill)?.state === 'untrusted';
const administered = (skill) => Boolean(requiredBy(skill) || trustOf(skill)?.administered);
// An update, review or repair of code the service will not run, or of the administrator's package, leads nowhere.
const offersChanges = (skill) => !administered(skill) && !untrusted(skill);

function repairableSource(skill) {
    if (!offersChanges(skill)) return false;
    const source = sourceTag(skill);
    const payloadRoot = String(skill.payload_root || '');
    // self_authored payloads live under skills/external/, so the payload-root
    // gate already matches them (#335).
    return ['clawhub', 'ouroboroshub', 'external', 'self_authored'].includes(source)
        && /^skills\/(external|clawhub|ouroboroshub)\//.test(payloadRoot);
}

function repairReady(skill) {
    return repairableSource(skill)
        && (skill.review_status === 'blockers'
            || preflightFailed(skill)
            || (Boolean(skill.load_error) && !missingGrantLoadError(skill) && !packagesInstallable(skill)));
}

function skillConflictReason(skill) {
    const conflict = skill?.conflict && typeof skill.conflict === 'object' ? skill.conflict : null;
    if (!conflict || conflict.code !== 'skill_conflict') return '';
    const names = Array.isArray(conflict.skills) ? conflict.skills.filter(Boolean) : [];
    if (!names.length) return 'another skill that is on';
    const omitted = Number(conflict.omitted || 0);
    return `${names.join(', ')}${omitted > 0 ? ` (+${omitted} more)` : ''}`;
}

const LIFECYCLE_PENDING_LABELS = {
    disable: 'Turning off…',
    enable: 'Turning on…',
    uninstall: 'Removing…',
    update: 'Updating…',
    install: 'Installing…',
    review: 'Reviewing…',
    deps: 'Installing packages…',
    delete: 'Deleting…',
};

function lifecyclePendingLabel(kind) {
    return LIFECYCLE_PENDING_LABELS[String(kind || '')] || 'Working…';
}

function lifecycleFailedLabel(kind) {
    const base = String(kind || '').trim();
    if (!base) return 'Didn’t finish';
    return `${base.charAt(0).toUpperCase()}${base.slice(1)} failed`;
}

/** The review job now running or last ended for these bytes, as recorded. */
function reviewJob(skill) {
    const current = skill?.skill_review?.current;
    return current && typeof current === 'object' && Object.keys(current).length ? current : null;
}

function reviewRunningNow(skill, reviewing) {
    if (reviewing) return true;
    if (skill.lifecycle_pending && skill.lifecycle_kind === 'review') return true;
    const status = String(reviewJob(skill)?.lifecycle_status || '').toLowerCase();
    return status === 'running' || status === 'queued';
}

/** The newest Skills-page review of these bytes whose stopped reviewers can continue (the server lists only the page's own). */
export function continuableReview(skill) {
    const history = Array.isArray(skill?.skill_review?.history) ? skill.skill_review.history : [];
    return [...history].reverse().find((row) => row?.record_id && row.content_hash === skill.content_hash
        && Array.isArray(row.continuable_seats) && row.continuable_seats.length) || null;
}

/** The last review job for the current bytes ended without a verdict (failed, timed out, interrupted). */
function reviewDidNotFinish(skill) {
    const job = reviewJob(skill);
    if (!job || reviewReady(skill)) return '';
    const lifecycle = String(job.lifecycle_status || job.job_status || '').toLowerCase();
    if (!lifecycleMeta(lifecycle)) return '';
    if (job.content_hash && skill.content_hash && job.content_hash !== skill.content_hash) return '';
    return String(job.terminal_reason || lifecycle);
}

function ownerMayAttest(skill) {
    // A hint only: the attestation endpoint repeats the authoritative checks.
    // A deterministic preflight FAIL makes attestation a guaranteed refusal (#335).
    return skill.owner_attestable !== false && !preflightFailed(skill)
        && !(skill.review_profile === 'owner_attested' && !skill.review_stale);
}

function failedFindings(skill) {
    const findings = Array.isArray(skill.review_findings) ? skill.review_findings : [];
    const fails = findings.filter((finding) => String(finding?.verdict || '').toUpperCase() === 'FAIL');
    const critical = fails.filter((finding) => finding?.severity === 'critical');
    return { fails, critical };
}

function firstProblemSentence(skill) {
    const { fails, critical } = failedFindings(skill);
    const first = critical[0] || fails[0] || (skill.review_findings || [])[0] || {};
    const text = preflightFindingText(first) || first.reason || first.message || first.title || first.item || '';
    return boundedText((String(text).match(/^[\s\S]*?[.;](?=\s|$)/) || [String(text)])[0].trim(), 200);
}

function problemCountText(skill) {
    const { fails, critical } = failedFindings(skill);
    if (!fails.length) return 'Review found problems.';
    const problems = critical.length || fails.length;
    const notes = critical.length ? fails.length - critical.length : 0;
    return `Review found ${plural(problems, 'problem')}${notes ? ` and ${plural(notes, 'note')}` : ''}.`;
}

const REQUIRED_LOCK = 'Required by your organization.';
const REQUIRED_WAIT = {
    hold: 'Ordinary work waits until it runs. Stop, Panic and Settings stay available.',
    warn: 'Work goes on while it isn’t running.',
};

/** The organization's requirement that this package does not meet here, in its own words (no value, no secret). */
function requiredTray(required) {
    return trayHtml({ tone: 'warn', attrs: 'data-required-unmet',
        sentence: `Your organization requires it, and it isn’t running: ${escapeHtml(boundedText(required.detail || required.reason || '', 240))}`,
        note: escapeHtml(REQUIRED_WAIT[required.when_unmet] || '') });
}

/**
 * The organization's required extensions that are not running, above the list (``/api/extensions`` ``deployment``):
 * a missing package has no card, so its row is said here. '' when every requirement runs or nothing is required.
 */
export function renderRequirementNotice(deployment) {
    const rows = (deployment?.required || []).filter((row) => row?.state === 'unmet');
    if (!rows.length) return '';
    const items = rows.map((row) => `<li data-required-name="${escapeHtml(row.name)}"><code data-i18n-authored>${escapeHtml(row.name)}</code> · ${escapeHtml(boundedText(row.detail || row.reason || '', 240))}</li>`).join('');
    const subject = rows.length === 1 ? 'an extension that isn’t' : `${rows.length} extensions that aren’t`;
    const process = rows[0]?.process?.role ? `Seen by the ${escapeHtml(rows[0].process.role)} process; each worker checks its own before it starts work.` : '';
    return `<section class="skills-org-requirements" data-tone="warn" aria-label="Required by your organization" data-skills-org-requirements>
        <p class="skills-org-requirements-lead">Your organization requires ${subject} running here.</p>
        <ul class="skills-org-requirements-list">${items}</ul>
        <p class="skills-org-requirements-note">${escapeHtml(REQUIRED_WAIT[deployment.when_unmet] || '')} ${process}</p>
    </section>`;
}

const actionAttrs = (name, action, extra = '') => `data-skill="${escapeHtml(name)}" data-skill-action="${escapeHtml(action)}"${extra ? ` ${extra}` : ''}`;
const smallButton = (label, name, action, extra = '') => buttonHtml(label, { tone: 'default', attrs: actionAttrs(name, action, extra) });
const skipButton = (name, label = 'Skip review') => skipReviewButtonHtml(actionAttrs(name, 'attest_review'), label);

/**
 * One installed skill as owner-facing facts: its run state chip, the attention
 * trays (one sentence + actions each), whether it needs the owner, whether an
 * update waits, and why the switch is locked. Pure over the listing row.
 */
export function installedSkillModel(skill, { reviewing = false, repairing = false, live = {}, pageModules = {} } = {}) {
    const name = String(skill.name || '');
    const install = installFacts(skill);
    const modified = modifiedFacts(skill);
    const sourceName = installedSourceLabel(skill);
    const update = install?.update_available && typeof install.update_available === 'object' ? install.update_available : null;
    const withdrawn = install?.withdrawn && typeof install.withdrawn === 'object' ? install.withdrawn : null;
    const conflict = skillConflictReason(skill);
    const reviewRunning = reviewRunningNow(skill, reviewing);
    const didNotFinish = reviewDidNotFinish(skill);
    const pageFaults = (pageModules.faults || []).filter((fault) => fault.current !== false && fault.skill === name);
    const pageHolds = (pageModules.held || []).filter((hold) => hold.skill === name);
    const pageRuntimeStopped = Boolean(pageModules.stopped);
    // Continuing saved work is its own choice beside a new review, never folded into Try again.
    const resume = continuableReview(skill) ? smallButton('Continue review', name, 'continue_review') : '';
    const resumeNote = resume ? 'A reviewer stopped with work it can continue from where it stopped.' : '';
    const trays = [];
    let chip = '';
    let needs = false;
    let lockReason = '';
    const pageTray = (kind, page) => kind === 'fault'
        ? trayHtml({ tone: 'warn', sentence: `Its page module stopped in this window: ${escapeHtml(boundedText(page.reason || 'unknown error', 240))}.`,
            note: 'The skill stays installed; reload or update it to try again.' })
        : pageRuntimeStopped
            ? trayHtml({ tone: 'warn', sentence: `Its page module is held in this window (${escapeHtml(HELD_LINES[page.reason] || 'Held back in this window.')})`,
                note: 'Reload this page to restore it.' })
            : trayHtml({ tone: 'warn', sentence: `Its page module is held in this window (${escapeHtml(HELD_LINES[page.reason] || 'Held back in this window.')})`,
                actions: smallButton('Start it', name, 'start_host_module', `data-host-module-key="${escapeHtml(page.key)}"`) });

    // An in-flight lifecycle job (serialized lane) takes precedence, so the card
    // shows "Turning off…" instead of the stale persisted state. A job in flight
    // goes on by itself: a neutral chip with the moving mark. Amber is kept for
    // the owner's move (Needs access, Not reviewed yet, Review didn't finish).
    if (skill.lifecycle_pending) {
        // An owner action's job names its work; it remains autonomous, not an owner warning.
        const message = ['enable', 'grant'].includes(String(skill.lifecycle_kind || '')) ? skill.lifecycle_message : '';
        chip = chipHtml(message || lifecyclePendingLabel(skill.lifecycle_kind), 'neutral', { work: true });
    } else if (skill.lifecycle_status === 'failed') {
        chip = chipHtml(lifecycleFailedLabel(skill.lifecycle_kind), 'danger');
        needs = true;
        if (skill.lifecycle_virtual) {
            trays.push(trayHtml({ tone: 'danger', sentence: escapeHtml(skill.lifecycle_error || 'It didn’t finish.'),
                note: 'Nothing was installed.', actions: smallButton('Find in Browse', name, 'find_in_browse') }));
        }
    }
    if (repairing) chip = chipHtml('Asking Ouroboros…', 'neutral', { work: true });

    if (skill.lifecycle_virtual) {
        return { chip, trays, overflow: [], needs, update: false, withdrawn: false, lockReason: '', sourceName, modified, install };
    }

    if (untrusted(skill)) {
        // The organization's choice, not the owner's move: a neutral fact with nothing here that would make it run.
        if (!skill.lifecycle_pending) chip = chipHtml('Doesn’t run here', 'neutral');
        trays.push(trayHtml({ sentence: escapeHtml(boundedText(trustOf(skill).detail || '', 240)),
            note: 'New extensions and their versions come from your organization’s administrator.' }));
        lockReason = 'Only extensions your organization delivers run here.';
    } else if (install?.pending_update) {
        chip = chipHtml('Reviewing…', 'neutral', { work: true });
        trays.push(trayHtml({ sentence: `Updating to ${escapeHtml(install.pending_update.version)}: it stays off until the review finishes.` }));
        lockReason = 'Can’t turn on until the review finishes.';
    } else if (!skill.lifecycle_pending && !repairing && reviewRunning) {
        chip = chipHtml('Reviewing…', 'neutral', { work: true });
        const started = relativeAge(reviewJob(skill)?.started_at);
        trays.push(trayHtml({
            sentence: `${started ? `Reviewing · started ${escapeHtml(started)}. ` : ''}Ouroboros reviews it before it runs or gets your keys.`,
            actions: ownerMayAttest(skill) ? skipButton(name) : '',
        }));
        lockReason = 'Can’t turn on until the review finishes.';
    } else if (skill.load_error && !missingGrantLoadError(skill)) {
        if (!skill.lifecycle_pending) chip = chipHtml('Couldn’t start', 'danger');
        needs = true;
        trays.push(trayHtml({ tone: 'danger', sentence: escapeHtml(boundedText(String(skill.load_error).split('\n')[0], 240)),
            actions: repairReady(skill) ? smallButton('Ask Ouroboros to fix', name, 'repair')
                : packagesInstallable(skill) ? smallButton('Install packages', name, 'turn_on') : '' }));
        lockReason = repairReady(skill) ? 'Ask Ouroboros to fix it first.' : '';
    } else if (conflict) {
        if (!skill.lifecycle_pending) chip = chipHtml('Conflicts', 'danger');
        needs = true;
        trays.push(trayHtml({ sentence: `Uses the same tool names as ${escapeHtml(conflict)}. Turn one of them off to use the other.` }));
        lockReason = `Conflicts with ${conflict}.`;
    } else if ((skill.review_status === 'blockers' && !reviewReady(skill)) || preflightFailed(skill)) {
        if (!skill.lifecycle_pending) chip = chipHtml('Review found problems', 'danger');
        needs = true;
        const first = firstProblemSentence(skill);
        trays.push(trayHtml({
            tone: 'danger',
            sentence: `${escapeHtml(problemCountText(skill))}${first ? ` First: “${escapeHtml(first)}”` : ''}`,
            note: 'Installed, but it doesn’t run and gets no keys until you decide.',
            actions: [smallButton('Read findings', name, 'read_findings'),
                repairReady(skill) ? smallButton('Ask Ouroboros to fix', name, 'repair') : '',
                ownerMayAttest(skill) ? skipButton(name) : ''].join(''),
        }));
        lockReason = 'Can’t turn on until the review passes.';
    } else if (didNotFinish) {
        if (!skill.lifecycle_pending) chip = chipHtml('Review didn’t finish', 'warn');
        needs = true;
        trays.push(trayHtml({ sentence: `The review didn’t finish: ${escapeHtml(boundedText(didNotFinish, 240))}`, note: resumeNote,
            actions: `${resume}${smallButton('Try again', name, 'review')}${ownerMayAttest(skill) ? skipButton(name) : ''}` }));
        lockReason = 'Can’t turn on until a review passes.';
    } else if (!reviewReady(skill)) {
        if (!skill.lifecycle_pending) chip = chipHtml('Not reviewed yet', 'warn');
        needs = true;
        let sentence = 'Ouroboros hasn’t reviewed it yet. Review it before it runs, or skip the review.';
        if (modified) {
            sentence = 'Its files changed here since it was installed, so it waits for a review. Review it, or skip the review.';
        } else if (preflightFailedStale(skill)) {
            sentence = 'The last file check failed on earlier files. Review the current files, or ask Ouroboros to fix it.';
        } else if (isMadeHere(skill)) {
            sentence = `Made ${skill.is_self_authored ? 'by Ouroboros ' : ''}on this computer. Review it before it runs, or skip the review.`;
        } else if (skill.review_stale) {
            sentence = 'Its files changed since the last review. Review it again before it runs, or skip the review.';
        }
        trays.push(trayHtml({ sentence, note: resumeNote,
            actions: `${resume}${smallButton('Review', name, skill.review_stale ? 'rereview' : 'review')}${ownerMayAttest(skill) ? skipButton(name) : ''}` }));
        lockReason = 'Can’t turn on until it is reviewed.';
    } else if (!grantReady(skill)) {
        if (!skill.lifecycle_pending) chip = chipHtml('Needs access', 'warn');
        needs = true;
        const grants = skill.grants || {};
        const keys = grants.missing_keys || grants.requested_keys || [];
        const permissions = grants.missing_permissions || grants.requested_permissions || [];
        const parts = [];
        if (keys.length) parts.push(`Wants to use ${escapeHtml(keys.join(', '))}.`);
        if (permissions.length) parts.push(`Wants your permission to ${escapeHtml(grantPhrases(permissions).join(', '))}.`);
        // The same fact the readiness line carries: the grant exists, for an earlier version of the files.
        const earlier = new Set(grants.earlier_grant ? [...(grants.earlier_grant.keys || []), ...(grants.earlier_grant.permissions || [])] : []);
        const share = [...keys, ...permissions].every((item) => earlier.has(item)) ? 'this' : 'some of this';
        const note = earlier.size ? `You granted ${share} to an earlier version of the skill; its files changed since.` : '';
        trays.push(trayHtml({ sentence: parts.join(' ') || 'Wants access you haven’t given yet.', note,
            actions: smallButton('Grant access', name, 'grant', `data-keys="${escapeHtml([...keys, ...permissions].join(','))}"`) }));
    }

    const secrets = secretConfigTray(skill);
    if (secrets.tray) trays.push(secrets.tray);
    if (secrets.needs) needs = true;

    const updateAction = update && !install?.pending_update && offersChanges(skill)
        ? smallButton(modified || install?.base_known === false ? 'Update…' : 'Update', name, 'update') : '';
    if (withdrawn) {
        needs = true;
        const when = shortDate(withdrawn.at);
        trays.push(trayHtml({ tone: 'danger',
            sentence: withdrawn.version
                ? `Version ${escapeHtml(withdrawn.version)} was withdrawn by ${escapeHtml(sourceName)}${withdrawn.reason ? `: ${escapeHtml(withdrawn.reason)}` : '.'}`
                : `${escapeHtml(sourceName)} withdrew ${escapeHtml(versionText(install?.release?.version) || 'this version')}${when ? ` on ${escapeHtml(when)}` : ''}${withdrawn.reason ? `: ${escapeHtml(withdrawn.reason)}` : '.'}`,
            note: withdrawn.version && updateAction
                ? `You can update to ${escapeHtml(versionText(update.version))}, or uninstall it.`
                : 'It won’t be installed anew. You can uninstall it.',
            actions: (withdrawn.version ? updateAction : '') + (administered(skill) ? '' : smallButton('Uninstall', name, 'uninstall')) }));
    } else if (install?.update_rejected && offersChanges(skill)) {
        needs = true;
        const rejected = install.update_rejected;
        trays.push(trayHtml({ sentence: escapeHtml(updateRejectionSentence(rejected)),
            note: updateAction ? `${escapeHtml(versionText(update.version) || 'A new version')} is available.` : '',
            actions: updateAction
                + (['review_blockers', 'preflight_failed'].includes(rejected.reason) ? smallButton('See findings', name, 'read_findings') : '')
                + (canUpdateWithoutReview(install) ? smallButton('Update without review', name, 'update_without_review') : '') }));
    } else if (update && !install?.pending_update && offersChanges(skill)) {
        const v = escapeHtml(versionText(update.version) || 'A new version');
        trays.push(modified || install?.base_known === false
            ? trayHtml({ sentence: `${escapeHtml(sourceName)} has ${v}. ${!modified ? 'Its original files aren’t known' : modified.by === 'owner' ? 'You changed this skill here' : 'This skill was changed here'}, so updating asks what to keep.`,
                actions: smallButton('Update…', name, 'update') })
            : trayHtml({ sentence: update.version && update.version === install?.release?.version
                ? `${escapeHtml(sourceName)} changed the files of ${v} (same version label).`
                : `${v} is available.`, actions: smallButton('Update', name, 'update') }));
    } else if (install?.left_source) {
        trays.push(trayHtml({ sentence: `No longer in ${escapeHtml(sourceName)}. It keeps working; it won’t get updates.` }));
    }

    const pageTrays = [];
    if (pageFaults.length) {
        chip = chipHtml('Off in this window', 'warn');
        needs = true;
        pageFaults.forEach((fault) => pageTrays.push(pageTray('fault', fault)));
    }
    if (pageHolds.length) {
        chip = chipHtml('Held in this window', 'warn');
        needs = true;
        pageHolds.forEach((hold) => pageTrays.push(pageTray('hold', hold)));
    }
    if (!skill.enabled && !lockReason && conflict) lockReason = `Conflicts with ${conflict}.`;
    const required = requiredBy(skill);
    const requiredTrays = required?.state === 'unmet' ? [requiredTray(required)] : [];
    if (requiredTrays.length) needs = true;
    const allTrays = [...requiredTrays, ...pageTrays, ...trays];
    return {
        chip, trays: allTrays.slice(0, 2), overflow: allTrays.slice(2), needs, sourceName, modified, install,
        // The switch shows what runs; whether it may be off is the organization's (the toggle refuses it too).
        lockReason: required ? REQUIRED_LOCK : (skill.enabled ? '' : lockReason),
        update: offersChanges(skill) && Boolean(update || withdrawn || install?.update_rejected || install?.pending_update), withdrawn: Boolean(withdrawn),
        widgets: skill.enabled && skill.type === 'extension' && skill.live_loaded && hasSkillUiTab(skill, live),
        settings: hasSkillSettings(skill, live),
    };
}

/** «Ouroboros Community · installed v2.12.0 3 months ago · edited by you here (now v3.3.0)». */
export function installedWhoHtml(skill, model = installedSkillModel(skill)) {
    const parts = [escapeHtml(model.sourceName)];
    const installedVersion = model.install?.release?.version;
    const versionMoved = Boolean(model.modified && installedVersion && skill.version && installedVersion !== skill.version);
    const age = relativeAge(model.install?.installed_at || installedTime(skill));
    if (versionMoved) {
        parts.push(`installed ${escapeHtml(versionText(installedVersion))}${age ? ` ${escapeHtml(age)}` : ''}`);
    } else {
        if (skill.version) parts.push(escapeHtml(versionText(skill.version)));
        if (age && isBuiltIn(skill)) parts.push(`came with Ouroboros ${escapeHtml(age)}`);
        else if (age && isMadeHere(skill)) parts.push(`made here ${escapeHtml(age)}`);
        else if (age) parts.push(`installed ${escapeHtml(age)}${model.install?.installed_by === 'ouroboros' ? ' by Ouroboros' : ''}`);
    }
    if (model.modified) {
        const now = versionMoved ? ` (now ${escapeHtml(versionText(skill.version))})` : '';
        parts.push(`${model.modified.by === 'owner' ? 'edited by you here' : 'changed here'}${now}`);
    }
    if (skill.published && typeof skill.published === 'object') {
        const sent = relativeAge(skill.published.submitted_at || skill.published.published_at);
        if (sent) parts.push(`sent ${escapeHtml(sent)}`);
    }
    return parts.join(' · ');
}

/**
 * Build a findings block's list on its first open from the rows already in
 * memory; a built list survives re-opening and card patches untouched.
 */
export function fillFindingsOnOpen(details, skills = []) {
    if (!details?.dataset?.skillFindings || !details.open || details.childElementCount > 1) return false;
    const skill = (skills || []).find((row) => row?.name === details.dataset.skillFindings);
    details.insertAdjacentHTML('beforeend', renderReviewFindingsList(skill));
    return true;
}

/** The findings list a card builds when its block is opened (skills.js), from the row already in memory. */
export function renderReviewFindingsList(skill) {
    const findings = Array.isArray(skill?.review_findings) ? skill.review_findings : [];
    if (!findings.length) return '<div class="muted">Findings are not in the current list; Refresh to reload them.</div>';
    const rows = findings.map((f) => {
        const preflight = preflightFindingText(f);
        const reason = preflight || f.reason || f.message || JSON.stringify(f);
        // An adviser's seat (`additional`) is heard beside the ticked reviewers and never counted.
        const adviser = f.additional === true ? ' (adviser, not counted)' : '';
        return `<li><strong>${escapeHtml(f.verdict || f.severity || '')}</strong>${adviser} ${escapeHtml(f.item || f.check || f.title || 'finding')}: ${escapeHtml(reason)}</li>`;
    }).join('');
    return `<ul>${rows}</ul>`;
}

// What the review block shows, as one key: while it holds, a re-read keeps the block (its open
// disclosures and a findings list built on open); a new run, verdict or findings set replaces it.
function reviewBlockKey(skill) {
    const review = skill.skill_review && typeof skill.skill_review === 'object' ? skill.skill_review : {};
    const history = Array.isArray(review.history) ? review.history : [];
    const current = review.current && Object.keys(review.current).length ? review.current : history[history.length - 1] || {};
    const text = JSON.stringify([current.job_id || '', current.content_hash || '', current.lifecycle_status || current.job_status || current.status || '',
        current.review_status || '', current.terminal_reason || '', history.length, review.history_omitted ?? '',
        Array.isArray(skill.review_findings) ? skill.review_findings.length : 0, skill.content_hash || '']);
    let hash = 5381;  // a short attribute, never the recorded words themselves
    for (let index = 0; index < text.length; index += 1) hash = ((hash << 5) + hash + text.charCodeAt(index)) | 0;
    return (hash >>> 0).toString(36);
}

function reviewFindings(skill) {
    const findings = Array.isArray(skill.review_findings) ? skill.review_findings : [];
    if (!findings.length) return '';
    // Collapsed with its summary only: findings are most of the list payload.
    return `<details class="ui-disclosure skills-review-findings ui-rich-content" data-skill-findings="${escapeHtml(skill.name)}"><summary>${plural(findings.length, 'review finding')}</summary></details>`;
}

function reviewRunTitle(run) {
    const round = Number(run?.review_round || 1);
    const attempt = Number(run?.snapshot_attempt || 1);
    const snapshot = String(run?.content_hash || '').slice(0, 12) || 'unknown';
    const revised = run?.snapshot_revised ? ' — revised snapshot' : '';
    return `Skill review round ${round} — snapshot ${snapshot} (attempt ${attempt})${revised}`;
}

/**
 * A run's job lifecycle and its review verdict are separate recorded facts: a
 * failed, timed-out, cancelled or interrupted job is named beside the verdict
 * with its recorded `terminal_reason`, so it never reads as an open `pending`.
 */
function reviewRunOutcome(run) {
    const lifecycle = String(run.lifecycle_status || run.job_status || run.status || '');
    const verdict = String(run.review_status || run.status || '');
    const failure = lifecycleMeta(lifecycle);
    if (!failure) return { label: verdict || lifecycle || 'unknown', reason: '' };
    const reason = String(run.terminal_reason || '').trim();
    return {
        label: `${hasSemanticVerdict(verdict) ? verdict : 'review verdict unavailable'} · ${failure}`,
        reason: reason.toLowerCase() === lifecycle.toLowerCase() ? '' : reason,
    };
}

// A long recorded reason stays readable by keyboard and touch: a disclosure, never a hover title.
function reviewReasonRow(reason) {
    if (!reason) return '';
    const shown = boundedText(reason, 400);
    const complete = shown === reason ? '' : `<details class="ui-disclosure skills-review-reason"><summary>Full reason</summary><div>${escapeHtml(reason)}</div></details>`;
    return `<div class="skills-review-reason">Reason: ${escapeHtml(shown)}</div>${complete}`;
}

function reviewHistory(skill) {
    const review = skill.skill_review && typeof skill.skill_review === 'object' ? skill.skill_review : {};
    // The backend projection already bounds history to its ten-row window and
    // discloses the exact omitted count; no second client-side slice.
    const history = Array.isArray(review.history) ? review.history : [];
    const current = review.current && Object.keys(review.current).length ? review.current : history[history.length - 1];
    if (!current) return '';
    const rows = history.map((run) => {
        const outcome = reviewRunOutcome(run);
        const source = run.source ? ` · ${run.source}` : '';
        return `<li>${escapeHtml(reviewRunTitle(run))} · ${escapeHtml(outcome.label)}${escapeHtml(source)}${reviewReasonRow(outcome.reason)}</li>`;
    }).join('');
    const omitted = Number(review.history_omitted);
    const historyLabel = Number.isFinite(omitted) && omitted > 0 ? `${history.length} of ${history.length + omitted}` : `${history.length}`;
    const currentOutcome = reviewRunOutcome(current);
    return `<div class="skills-review-current">${escapeHtml(reviewRunTitle(current))} · ${escapeHtml(currentOutcome.label)}</div>${reviewReasonRow(currentOutcome.reason)}
        ${rows ? `<details class="ui-disclosure skills-review-history ui-rich-content"><summary>Review history (${historyLabel})</summary><ol>${rows}</ol></details>` : ''}`;
}

/** Exactly one sentence of review evidence for the current bytes (UX §3.4). */
export function reviewSentence(skill) {
    const install = installFacts(skill);
    const ready = reviewReady(skill);
    // A passed review of these bytes is the strongest evidence; otherwise the host's acceptance basis says why it runs.
    const accepted = ['clean', 'warnings'].includes(skill.review_status) && !skill.review_stale ? '' : acceptedAs(skill);
    if ((skill.review_profile === 'owner_attested' && !skill.review_stale) || (ready && accepted === 'owner_skip')) {
        return 'You skipped the review; it runs unreviewed.';
    }
    if (ready && accepted === 'trusted_source') {
        const when = shortDate(install?.installed_at || skill.installed_at);
        return `Installed without review${when ? ` on ${when}` : ''} because you trust ${installedSourceLabel(skill)}.`;
    }
    if (ready && accepted === 'owner_edit') return 'Edited by you here; it runs without a new review.';
    if (ready && accepted === 'self_check') {
        const notes = String(install?.check?.notes || '');
        return `Ouroboros read it before accepting it and recorded why: “${boundedText(notes, 240)}”.`;
    }
    if (ready && skill.review_profile === 'native_seed') return 'Comes with the app; it runs without a separate review.';
    // A passed review stays the evidence while a newer round of the same files has no verdict yet.
    const job = ready && reviewJob(skill);
    const again = job && (reviewRunningNow(skill, false) || reviewRunOutcome(job).label === 'pending')
        ? ` · a new review${job.review_round ? ` (round ${Number(job.review_round)})` : ''} is pending` : '';
    if (ready && skill.review_status === 'warnings') return `Reviewed on this computer · passed with notes${again}.`;
    if (ready) return `Reviewed on this computer · passed${again}.`;
    if (skill.review_status === 'blockers' || preflightFailed(skill)) return 'Review found problems; it doesn’t run until you decide.';
    if (preflightFailedStale(skill)) return 'The last file check failed on earlier files; review the current files.';
    return 'Not reviewed for its current files.';
}

function accessRows(skill) {
    const grants = skill.grants || {};
    const requested = [...(grants.requested_keys || []), ...(grants.requested_permissions || [])];
    if (!requested.length) return '';
    const granted = [...(grants.granted_keys || []), ...(grants.granted_permissions || [])];
    const missing = [...(grants.missing_keys || []), ...(grants.missing_permissions || [])];
    const status = grants.unsupported_for_skill_type ? 'This skill type cannot receive keys or host permissions.'
        : missing.length ? `Not allowed yet: ${missing.join(', ')}.` : 'Access granted.';
    return detailRow('Access', `Asks for ${escapeHtml(requested.join(', '))}${granted.length ? ` · granted ${escapeHtml(granted.join(', '))}` : ''}. ${escapeHtml(status)}`);
}

function detailRow(label, valueHtml) {
    return valueHtml ? `<div class="skills-detail-row"><span class="skills-detail-label">${escapeHtml(label)}</span><span class="skills-detail-value">${valueHtml}</span></div>` : '';
}

function installRecordRows(skill) {
    const install = installFacts(skill);
    if (!install) return detailRow('Folder', `<code>${escapeHtml(skill.payload_root || '—')}</code>`);
    const release = install.release || {};
    const basis = {
        trusted: `installed without review — you trusted ${escapeHtml(installedSourceLabel(skill))}`,
        review: 'reviewed before it runs',
        owner_skip: 'you skipped the review',
        owner_edit: 'edited by you',
        self_check: 'Ouroboros read it and recorded why',
        cyber: 'installed by Ouroboros in Cyber Pro',
    }[install.basis] || escapeHtml(install.basis || 'unknown');
    const task = install.task_id ? ` · in task ${escapeHtml(install.task_id)}` : '';
    const reason = install.reason ? ` · because: ${escapeHtml(install.reason)}` : '';
    return [
        detailRow('Source', `${escapeHtml(installedSourceLabel(skill))}${install.official ? ' · Official' : ''}`),
        detailRow('Installed', `${escapeHtml(versionText(release.version) || 'unknown version')}${release.digest ? ` · <code>${escapeHtml(String(release.digest).slice(0, 24))}</code>` : ''} · ${basis}${install.installed_by === 'ouroboros' ? ' · by Ouroboros' : ''}${task}${reason}`),
        install.check?.notes ? detailRow('Ouroboros’s check', escapeHtml(install.check.notes)) : '',
        install.turn_on_when_ready && !skill.enabled ? detailRow('Turns on', 'by itself once it is reviewed, has its keys and its packages are ready.') : '',
        detailRow('Folder', `<code>${escapeHtml(skill.payload_root || '—')}</code>`),
    ].join('');
}

function liveRow(skill, live) {
    if (skill.type !== 'extension' || !skill.enabled) return '';
    const process = ['server', 'worker'].includes(skill.process) ? ` · ${skill.process}` : '';
    const state = skill.live_loaded && (skill.dispatch_live || hasSkillUiTab(skill, live)) ? 'Loaded'
        : skill.live_loaded ? 'Loaded — UI tab pending' : 'Not loaded yet';
    return detailRow('Running', `${state}${process}`);
}

function publicationRow(skill) {
    const published = skill.published && typeof skill.published === 'object' ? skill.published : null;
    if (!published) return '';
    const line = renderSubmissionHistory({
        version: String(published.version || ''),
        pr_number: Number.isInteger(published.pr_number) && published.pr_number > 0 ? published.pr_number : null,
        pr_url: String(published.pr_url || ''),
        local_differs: Boolean(skill.content_hash) && skill.content_hash !== String(published.content_hash || ''),
    });
    if (!line) return '';
    const destination = published.destination || published.repository;
    return detailRow('Publication', `${line}${destination ? ` · ${escapeHtml(String(destination))}` : ''} <button type="button" class="skills-link-button ui-hit-area" ${actionAttrs(skill.name, 'forget_publication')}>Forget this record</button>`);
}

function provenanceRows(skill) {
    const prov = sourceTag(skill) === 'clawhub' && skill.provenance && typeof skill.provenance === 'object' ? skill.provenance : null;
    if (!prov) return '';
    const parts = [];
    if (prov.slug) parts.push(`slug <code>${escapeHtml(prov.slug)}</code>`);
    if (prov.license) parts.push(`license ${escapeHtml(prov.license)}`);
    const href = safeExternalUrl(prov.homepage);
    if (href) parts.push(`<a href="${href}" target="_blank" rel="noopener noreferrer" data-skill-external>homepage ↗</a>`);
    const warnings = (prov.adapter_warnings || []).map((msg) => `<li>${escapeHtml(msg)}</li>`).join('');
    return detailRow('ClawHub', parts.join(' · '))
        + (warnings ? `<details class="ui-disclosure skills-card-warnings"><summary>Adapter warnings</summary><ul>${warnings}</ul></details>` : '');
}

function makeRunnableOffer(skill, repairing) {
    // An instruction skill from a source or made here can become a runnable
    // script skill through the ordinary repair path.
    return skill.type === 'instruction' && !administered(skill)
        && ['clawhub', 'ouroboroshub', 'external', 'self_authored'].includes(sourceTag(skill))
        && /^skills\/(external|clawhub|ouroboroshub)\//.test(String(skill.payload_root || ''))
        && !repairing;
}

function submitHubReady(skill, githubTokenConfigured) {
    // The host's SSOT verdict; task_start_allowed owns admission when present.
    if (skill.submit_hub && typeof skill.submit_hub === 'object') {
        const submit = skill.submit_hub;
        return {
            ...submit,
            disabled: typeof submit.task_start_allowed === 'boolean'
                ? !submit.task_start_allowed
                : (typeof submit.disabled === 'boolean' ? submit.disabled : true),
        };
    }
    const visible = ['external', 'self_authored', 'user_repo', 'ouroboroshub', 'clawhub'].includes(sourceTag(skill));
    if (!visible) return { visible: false, disabled: true, reason: '' };
    if (githubTokenConfigured !== true) return { visible: true, disabled: true, reason: githubTokenConfigured === false
        ? 'Publishing needs a GitHub token: Settings › Accounts and keys › GitHub.' : 'GitHub token status unavailable. Refresh to retry.' };
    return { visible: true, publication_ready: false, task_start_allowed: true, disabled: false, reason: '' };
}

function menuHtml(skill, { reviewing, repairing, submit, settings = false }) {
    const name = escapeHtml(skill.name);
    const items = [];
    items.push(`<button type="button" role="menuitem" class="skills-menu-item" ${actionAttrs(skill.name, 'open_detail')}>Details…</button>`);
    if (settings) items.push(`<button type="button" role="menuitem" class="skills-menu-item" ${actionAttrs(skill.name, 'open_settings')}>Settings</button>`);
    if (!reviewing && !untrusted(skill) && !(preflightFailed(skill) && repairReady(skill))) {
        const label = skill.review_status === 'pending' && !skill.review_stale ? 'Review' : 'Review again';
        items.push(`<button type="button" role="menuitem" class="skills-menu-item skills-review" ${actionAttrs(skill.name, skill.review_stale ? 'rereview' : 'review')}>${label}</button>`);
    }
    if (makeRunnableOffer(skill, repairing)) {
        items.push(`<button type="button" role="menuitem" class="skills-menu-item skills-make-runnable" data-skill="${name}" data-skill-action="repair">Make runnable</button>`);
    }
    if (!repairing && preflightFailedStale(skill) && !preflightFailed(skill) && repairableSource(skill)) {
        items.push(`<button type="button" role="menuitem" class="skills-menu-item skills-repair-stale" data-skill="${name}" data-skill-action="repair">Ask Ouroboros to fix</button>`);
    }
    if (submit.visible) {
        items.push(`<button type="button" role="menuitem" class="skills-menu-item skills-submit-hub" ${actionAttrs(skill.name, 'submit_hub')} data-submit-disabled="${submit.disabled ? 'true' : 'false'}" data-submit-reason="${escapeHtml(submit.reason || '')}" data-submit-state="${escapeHtml(submit.state || '')}" data-publication-ready="${submit.publication_ready === true ? 'true' : 'false'}" aria-disabled="${submit.disabled ? 'true' : 'false'}">Publish…</button>`);
    }
    // The host names the removal it offers (listing `removal`, the rule both removers admit by); the card renders exactly that.
    if (administered(skill)) {
        // The administrator's package (required, or delivered for an execution domain): nothing here removes it.
    } else if (skill.removal === 'uninstall') {
        items.push(`<button type="button" role="menuitem" class="skills-menu-item skills-uninstall" ${actionAttrs(skill.name, 'uninstall')}>Uninstall</button>`);
    } else if (skill.removal === 'delete') {
        items.push(`<button type="button" role="menuitem" class="skills-menu-item skills-delete-local" ${actionAttrs(skill.name, 'delete', `data-payload-root="${escapeHtml(skill.payload_root || '')}"`)}>Delete</button>`);
    }
    return `<div class="skills-card-menu"><button type="button" class="btn btn-ghost btn-sm skills-card-menu-trigger" aria-label="More actions for ${name}" aria-haspopup="menu" aria-expanded="false" data-skill-menu-trigger>⋮</button><dialog class="skills-card-menu-dialog" role="menu">${items.join('')}</dialog></div>`;
}

function contributionVisibilityHtml(skill, options = {}) {
    const source = typeof options.contributionVisibility === 'function'
        ? options.contributionVisibility(skill.name) : options.contributionVisibility;
    const rows = Array.isArray(source) ? source : [];
    if (!rows.length) return '';
    // The Details rows' own layout (label, then value): one row per placed control, its place and Hide/Show.
    const items = rows.map((row) => {
        const key = escapeHtml(row.key || '');
        const label = row.label || row.skill_label || row.id || 'Skill control';
        const place = row.place || 'In the chat';
        const hidden = row.hidden === true;
        return `<div class="skills-detail-row" data-contribution-row="${key}"><span class="skills-detail-label">${escapeHtml(label)}</span><span class="skills-detail-value">${escapeHtml(place)}${hidden ? ' · hidden' : ''} <button type="button" class="btn btn-ghost btn-sm" ${actionAttrs(skill.name, 'toggle_contribution',
            `data-contribution-key="${key}" data-contribution-hidden="${hidden ? 'true' : 'false'}" aria-label="${hidden ? 'Show' : 'Hide'} ${escapeHtml(label)}, ${escapeHtml(place.toLowerCase())}"`)}>${hidden ? 'Show' : 'Hide'}</button></span></div>`;
    }).join('');
    return `<div class="skills-details-facts" data-skill-detail-contributions>${detailRow('Shown in chat', 'Each control can be hidden without turning the skill off.')}${items}</div>`;
}

/** Patch the visibility rows in place, keyed by contribution, so a focused Hide/Show keeps its focus. */
function patchContributionRows(target, source) {
    const rowsOf = (node) => [...node.querySelectorAll(':scope > [data-contribution-row]')].map((row) => row.dataset.contributionRow);
    if (rowsOf(target).join('\n') !== rowsOf(source).join('\n')) {
        const focusedKey = target.contains(document.activeElement) ? document.activeElement.dataset?.contributionKey || '' : '';
        target.innerHTML = source.innerHTML;
        if (focusedKey) [...target.querySelectorAll('[data-contribution-key]')].find((node) => node.dataset.contributionKey === focusedKey)?.focus();
        return;
    }
    const fresh = new Map([...source.querySelectorAll(':scope > [data-contribution-row]')].map((row) => [row.dataset.contributionRow, row]));
    for (const row of target.querySelectorAll(':scope > [data-contribution-row]')) {
        const next = fresh.get(row.dataset.contributionRow);
        if (!next || row.innerHTML === next.innerHTML) continue;
        const button = row.querySelector('[data-contribution-key]'), nextButton = next.querySelector('[data-contribution-key]');
        if (!button || !nextButton) { row.innerHTML = next.innerHTML; continue; }
        // The value's words before the button, then the button's own state: the same node keeps focus.
        const value = row.querySelector('.skills-detail-value'), nextValue = next.querySelector('.skills-detail-value');
        if (value?.firstChild?.nodeType === 3 && nextValue?.firstChild?.nodeType === 3) value.firstChild.textContent = nextValue.firstChild.textContent;
        for (const name of ['data-contribution-hidden', 'aria-label']) button.setAttribute(name, nextButton.getAttribute(name) || '');
        button.textContent = nextButton.textContent;
    }
}

/** One installed card. `options.githubTokenConfigured` gates Publish; `live` names loaded UI tabs. */
export function renderInstalledSkillCard(skill, reviewingSkills = new Set(), repairingSkills = new Set(), live = {}, options = {}) {
    const safeName = escapeHtml(skill.name);
    const reviewing = reviewingSkills.has(skill.name);
    const repairing = repairingSkills.has(skill.name);
    const model = installedSkillModel(skill, { reviewing, repairing, live, pageModules: options.pageModules });
    const description = skill.lifecycle_virtual && [skill.lifecycle_error, skill.load_error].includes(skill.description) ? '' : skill.description;
    const submit = submitHubReady(skill, options.githubTokenConfigured);
    const lockReason = model.lockReason;
    // While a lifecycle job for this skill is queued/running, reflect the
    // in-flight intent and lock the control, so a re-render cannot snap it back.
    const lifecyclePending = Boolean(skill.lifecycle_pending);
    const toggleOn = skill.lifecycle_kind === 'disable' && lifecyclePending ? false
        : (skill.lifecycle_kind === 'enable' && lifecyclePending ? true : skill.enabled);
    const toggleLocked = Boolean(lockReason) || lifecyclePending;
    const trayId = `skill-trays-${safeName.replace(/[^A-Za-z0-9_-]/g, '_')}`;
    const toggle = skill.lifecycle_virtual ? '' : `<label class="ui-switch ${toggleLocked ? 'is-locked' : ''}">
        <input type="checkbox" class="skills-toggle" role="switch" data-skill="${safeName}" ${toggleOn ? 'checked' : ''} ${toggleLocked ? 'disabled' : ''} aria-checked="${toggleOn ? 'true' : 'false'}" aria-label="${escapeHtml(skill.name)}${lifecyclePending ? ` (${escapeHtml(lifecyclePendingLabel(skill.lifecycle_kind))})` : ''}"${model.trays.length ? ` aria-describedby="${trayId}"` : ''}>
        <span class="ui-switch-track" aria-hidden="true"><span class="ui-switch-thumb"></span></span>
    </label>`;
    const marks = [
        requiredBy(skill) ? chipHtml('Required by your organization', 'neutral', { attrs: 'data-required-by-organization' })
            : trustOf(skill)?.state === 'trusted' ? chipHtml('Delivered by your organization', 'neutral', { attrs: 'data-delivered-by-organization' }) : '',
        isBuiltIn(skill) ? chipHtml('Built in', 'neutral') : '',
        model.install?.official ? officialMarkHtml({ sourceId: model.install.source?.id }) : '',
        model.modified ? chipHtml(model.modified.by === 'owner' ? 'Modified · edited by you' : 'Modified', 'neutral') : '',
        connectsText(skill) ? chipHtml(connectsText(skill), 'neutral') : '',
    ].join('');
    const quiet = [model.widgets ? buttonHtml('Open widgets', { tone: 'ghost', attrs: actionAttrs(skill.name, 'open_widgets') }) : '',
        model.settings ? buttonHtml('Settings', { tone: 'ghost', attrs: actionAttrs(skill.name, 'open_settings') }) : ''].join('');
    const widgetsButton = quiet && !model.trays.length ? `<div class="skills-card-quiet-actions">${quiet}</div>` : '';
    const errors = [
        skill.lifecycle_status === 'failed' && skill.lifecycle_error && !skill.lifecycle_virtual ? `<div class="skills-load-error">${escapeHtml(skill.lifecycle_error)}</div>` : '',
        skill.health_regressed ? `<div class="skills-load-error">Regression: was live at ${escapeHtml(String((skill.last_known_good || {}).version || '?'))} (${escapeHtml(String((skill.last_known_good || {}).sha || '').slice(0, 12))}); broken after a code update.</div>` : '',
    ].join('');
    // Details: two fact groups the listing re-reads in place and the review block (kept while its key holds).
    const details = `<details class="ui-disclosure skills-details"><summary>Details</summary>
        <div class="skills-details-body">
            <div class="skills-details-facts" data-skill-detail-review>${detailRow('Review', `${escapeHtml(reviewSentence(skill))} ${statusBadge(skill.review_status, skill.review_gate, skill.review_profile)}`)}</div>
            <div class="skills-review-block" data-review-key="${escapeHtml(reviewBlockKey(skill))}">${reviewHistory(skill)}${reviewFindings(skill)}</div>
            <div class="skills-details-overflow" data-skill-detail-overflow>${(model.overflow || []).join('')}</div>
            ${contributionVisibilityHtml(skill, options)}
            <div class="skills-details-facts" data-skill-detail-facts>
                ${detailRow('Can use', escapeHtml(permissionWords(skill.permissions).join(' · ') || 'Nothing beyond its own folder'))}
                ${accessRows(skill)}
                ${detailRow(tr('skills.id', 'ID'), `<code data-i18n-authored>${escapeHtml(skill.name)}</code>`)}
                ${installRecordRows(skill)}
                ${liveRow(skill, live)}
                ${publicationRow(skill)}
                ${submit.visible && submit.disabled && submit.reason ? detailRow('Publishing', escapeHtml(submit.reason)) : ''}
                ${provenanceRows(skill)}
            </div>
        </div>
    </details>`;
    return `<article class="ui-card skills-card skills-installed-card" data-skill="${safeName}"${skill.lifecycle_virtual ? ' data-lifecycle-virtual' : ''}${model.needs ? ' data-needs-you' : ''}${reviewing ? ' data-reviewing="1"' : ''}${repairing ? ' data-repairing="1"' : ''}>
        <header class="skills-card-head">
            <div class="skills-card-title">
                <div class="skills-head-title">${connectsServices(skill).length ? `<span class="skills-card-icon" aria-hidden="true">${skillIconHtml(skill.icon, 18)}</span>` : ''}<button type="button" class="skills-card-name ui-hit-area" ${actionAttrs(skill.name, 'open_detail')}><span data-i18n-authored>${escapeHtml(skillTitle(skill))}</span><span class="skills-card-name-more" aria-hidden="true">›</span></button><span class="skills-card-marks">${marks}</span></div>
                ${description ? `<p class="skills-card-purpose">${escapeHtml(description)}</p>` : ''}
                <div class="skills-card-who">${installedWhoHtml(skill, model)}</div>
                ${countsAsYoursText(skill) ? `<p class="skills-card-facts">${escapeHtml(countsAsYoursText(skill))}</p>` : ''}
            </div>
            <div class="skills-card-toggle"><span class="skills-card-status">${model.chip}</span>${toggle}${skill.lifecycle_virtual ? '' : menuHtml(skill, { reviewing, repairing, submit, settings: model.settings })}</div>
        </header>
        ${skill.lifecycle_virtual ? '' : details}
        <div class="skills-card-trays" id="${trayId}" data-skill-trays>${model.trays.join('')}${widgetsButton}</div>
        <div data-skill-errors>${errors}</div>
    </article>`;
}

/** Patch only state-derived facts; forms, open disclosures and open menus keep their identity. */
export function patchInstalledSkillEnrichment(card, skill, reviewing, repairing, live, options, menu = card) {
    const template = card.ownerDocument.createElement('template');
    template.innerHTML = renderInstalledSkillCard(skill, reviewing, repairing, live, options);
    const next = template.content.firstElementChild;
    function attributes(target, source, preserveDisabled = false) {
        for (const attr of [...target.attributes]) {
            if (preserveDisabled && attr.name === 'disabled') continue;
            if (!source.hasAttribute(attr.name)) target.removeAttribute(attr.name);
        }
        for (const attr of source.attributes) {
            if (preserveDisabled && attr.name === 'disabled') continue;
            if (target.getAttribute(attr.name) !== attr.value) target.setAttribute(attr.name, attr.value);
        }
    }
    for (const name of ['data-reviewing', 'data-repairing', 'data-needs-you']) {
        if (next.hasAttribute(name)) card.setAttribute(name, next.getAttribute(name));
        else card.removeAttribute(name);
    }
    for (const selector of ['.skills-card-status', '[data-skill-trays]', '[data-skill-errors]', '.skills-card-who', '.skills-card-marks', '.skills-card-purpose',
        '[data-skill-detail-review]', '[data-skill-detail-overflow]', '[data-skill-detail-contributions]', '[data-skill-detail-facts]']) {
        const target = card.querySelector(selector), source = next.querySelector(selector);
        if (!target && source) {
            if (selector === '.skills-card-purpose') card.querySelector('.skills-head-title')?.after(source);
            else if (selector === '[data-skill-detail-contributions]') card.querySelector('[data-skill-detail-overflow]')?.after(source);
        } else if (target && !source) target.remove();
        else if (target && source && selector === '[data-skill-detail-contributions]') patchContributionRows(target, source);
        else if (target && source && target.innerHTML !== source.innerHTML) target.innerHTML = source.innerHTML;
    }
    const block = card.querySelector('.skills-review-block'), nextBlock = next.querySelector('.skills-review-block');
    if (block && nextBlock && block.dataset.reviewKey !== nextBlock.dataset.reviewKey) {
        block.dataset.reviewKey = nextBlock.dataset.reviewKey;
        block.innerHTML = nextBlock.innerHTML;
    }
    const control = card.querySelector('.skills-toggle');
    if (control && control.dataset.skillPending !== 'true') {
        const source = next.querySelector('.skills-toggle');
        if (source) {
            attributes(control.closest('.ui-switch'), source.closest('.ui-switch'));
            attributes(control, source);
            control.checked = source.checked;
            control.disabled = source.disabled;
        }
    }
    const publish = menu.querySelector('.skills-submit-hub'), nextPublish = next.querySelector('.skills-submit-hub');
    if (publish && nextPublish) attributes(publish, nextPublish, true);
}
