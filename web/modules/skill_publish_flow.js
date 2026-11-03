import { createTask, skillPublishPreflight } from './api_client.js';
import { openConfirmDialog } from './confirm_dialog.js';
import { publicationDisclosure } from './skill_presentation.js';

export const SKILL_PUBLISH_STATES = Object.freeze([
    'ready',
    'warnings',
    'needs_attention',
    'repairable',
    'hard_block',
]);

const STATE_PRESENTATION = Object.freeze({
    ready: {
        label: 'Ready',
        title: 'Ready to publish',
        body: 'The files passed the check. Ouroboros checks them once more, opens the publication request and posts its link in chat.',
        confirmLabel: 'Publish',
    },
    warnings: {
        label: 'Warnings',
        title: 'Ready, with notes',
        body: 'The check left notes that don’t block publishing; they are in the details below. Ouroboros checks the files once more, opens the publication request and posts its link in chat.',
        confirmLabel: 'Publish with warnings',
    },
    needs_attention: {
        label: 'Needs attention',
        title: 'Not ready yet',
        body: 'Ouroboros can fix it and publish: it starts a task with these facts, and the publication request opens only after the final checks pass.',
        confirmLabel: 'Ask Ouroboros to fix and publish',
    },
    repairable: {
        label: 'Repairable',
        title: 'Not ready yet',
        body: 'The secrets check reports a problem Ouroboros can repair. It starts a task with these facts, and the publication request opens only after the final checks pass.',
        confirmLabel: 'Ask Ouroboros to fix and publish',
    },
    hard_block: {
        label: 'Unavailable',
        title: 'Can’t publish',
        body: 'Publishing cannot start. The reason is in the details below; fix it, then publish again.',
        confirmLabel: 'OK',
    },
});

function text(value) {
    return String(value ?? '').trim();
}

function numericFact(value) {
    if (value === null || value === undefined || value === '') return '';
    const number = Number(value);
    return Number.isFinite(number) && number >= 0 ? String(number) : '';
}

function targetFromPreflight(preflight) {
    return {
        skill: text(preflight?.skill),
        repository: text(preflight?.repository),
    };
}

function validTaskStart(preflight) {
    const state = text(preflight?.state);
    const target = targetFromPreflight(preflight);
    return preflight?.ok === true
        && SKILL_PUBLISH_STATES.includes(state)
        && state !== 'hard_block'
        && preflight.task_start_allowed === true
        && Boolean(target.skill && target.repository);
}

function reviewFact(review) {
    if (!review || typeof review !== 'object') return '';
    const parts = [text(review.status)];
    if (review.stale === true) parts.push('stale');
    if (text(review.profile)) parts.push(`profile: ${text(review.profile)}`);
    return parts.filter(Boolean).join(' · ');
}

function scannerFact(scanner) {
    if (!scanner || typeof scanner !== 'object') return '';
    const identity = [text(scanner.engine), text(scanner.version)].filter(Boolean).join(' ');
    const parts = [text(scanner.status), identity];
    if (text(scanner.ruleset_sha256)) parts.push(`ruleset: ${text(scanner.ruleset_sha256)}`);
    return parts.filter(Boolean).join(' · ');
}

function findingFact(finding) {
    if (!finding || typeof finding !== 'object') return '';
    const location = [text(finding.path), numericFact(finding.line)].filter(Boolean).join(':');
    const identity = [
        text(finding.detector),
        text(finding.confidence),
        text(finding.disposition),
        text(finding.verification),
    ].filter(Boolean).join(' · ');
    return [location, identity, text(finding.reason)].filter(Boolean).join('\n');
}

/** Build the typed detail rows consumed by the shared escaped dialog. */
export function skillPublishDetails(preflight = {}) {
    const rows = [];
    const state = text(preflight.state);
    const presentation = STATE_PRESENTATION[state];
    const target = targetFromPreflight(preflight);
    rows.push({ label: 'State', value: presentation?.label || state || 'Unknown' });
    if (target.skill) rows.push({ label: 'Skill', value: target.skill });
    if (target.repository) rows.push({ label: 'Destination', value: target.repository });
    if (text(preflight.snapshot_hash)) {
        rows.push({ label: 'Snapshot', value: text(preflight.snapshot_hash) });
    }
    const review = reviewFact(preflight.review);
    if (review) rows.push({ label: 'Review', value: review });
    const scanner = scannerFact(preflight.scanner);
    if (scanner) rows.push({ label: 'Scanner', value: scanner });

    const counts = [
        ['Blocking findings', preflight.blocker_count],
        ['Warnings', preflight.warning_count],
        ['Audited false positives', preflight.audited_false_positive_count],
    ];
    for (const [label, value] of counts) {
        const rendered = numericFact(value);
        if (rendered) rows.push({ label, value: rendered });
    }
    if (text(preflight.reason_code)) {
        rows.push({ label: 'Reason code', value: text(preflight.reason_code) });
    }
    if (text(preflight.repair_hint)) {
        rows.push({ label: 'Repair hint', value: text(preflight.repair_hint) });
    }

    const findings = Array.isArray(preflight.findings) ? preflight.findings : [];
    findings.forEach((finding, index) => {
        const value = findingFact(finding);
        if (value) rows.push({ label: `Finding ${index + 1}`, value });
    });
    const omitted = numericFact(preflight.omitted_count);
    if (omitted && omitted !== '0') {
        rows.push({ label: 'Omitted findings', value: omitted });
    }
    return {
        summary: findings.length || Number(preflight.omitted_count || 0) > 0
            ? 'Show redacted preflight details'
            : 'Show preflight details',
        rows,
    };
}

/** Translate one backend-authored state into dialog copy without reclassifying it. */
export function skillPublishDialogModel(preflight = {}, { privateRepository } = {}) {
    const state = text(preflight.state);
    const presentation = STATE_PRESENTATION[state];
    const target = targetFromPreflight(preflight);
    const targetComplete = Boolean(target.skill && target.repository);
    const canStart = validTaskStart(preflight);
    const summary = text(preflight.summary);

    if (!presentation
        || (!targetComplete && state !== 'hard_block')
        || (preflight.ok !== true && state !== 'hard_block')) {
        return {
            state,
            canStart: false,
            dialog: {
                title: 'Publish preflight unavailable',
                body: summary || 'The server did not return a complete canonical publish target and state. No task was created.',
                alert: true,
                confirmLabel: 'OK',
                details: skillPublishDetails(preflight),
            },
        };
    }

    const titleTarget = target.skill ? `: ${target.skill}` : '';

    return {
        state,
        canStart,
        dialog: {
            title: `${presentation.title}${titleTarget}`,
            body: [summary, presentation.body, canStart ? publicationDisclosure(privateRepository, target.repository) : ''].filter(Boolean).join('\n\n'),
            alert: !canStart,
            confirmLabel: presentation.confirmLabel,
            cancelLabel: 'Cancel',
            danger: canStart,
            details: skillPublishDetails(preflight),
        },
    };
}

/**
 * Build the exact ordinary managed-task request after explicit confirmation. The owner's
 * `destination` ({source_id, mode}) rides in `skill_publish_target`, where the publish tool reads it.
 */
export function buildSkillPublishTask(preflight = {}, { projectId = '', chatId, destination = null, privateRepository } = {}) {
    const { skill, repository } = targetFromPreflight(preflight);
    if (!validTaskStart(preflight)) {
        throw new Error('Publish preflight does not authorize a task start.');
    }
    const disclosure = publicationDisclosure(privateRepository, repository);
    const evidence = {
        state: text(preflight.state),
        publication_ready: preflight.publication_ready === true,
        snapshot_hash: text(preflight.snapshot_hash),
        reason_code: text(preflight.reason_code),
        summary: text(preflight.summary),
        repair_hint: text(preflight.repair_hint),
        review: {
            status: text(preflight.review?.status),
            stale: preflight.review?.stale === true,
            profile: text(preflight.review?.profile),
        },
        scanner: {
            status: text(preflight.scanner?.status),
            engine: text(preflight.scanner?.engine),
            version: text(preflight.scanner?.version),
            ruleset_sha256: text(preflight.scanner?.ruleset_sha256),
        },
        blocker_count: Number(preflight.blocker_count || 0),
        warning_count: Number(preflight.warning_count || 0),
    };
    return {
        source: 'web',
        ...(text(projectId) ? { project_id: text(projectId) } : { chat_id: 1 }),
        ...(chatId === undefined ? {} : { chat_id: chatId }),
        description: `Publish the installed skill "${skill}" to "${repository}" by opening a GitHub pull request. ${disclosure} Success requires a validated pull-request receipt for this exact skill and repository.`,
        type: 'skill_publish',
        expected_output: `Return only the validated GitHub pull-request URL opened in "${repository}" for the canonical skill "${skill}".`,
        constraints: `Publication is authorized for the canonical skill "${skill}" and repository "${repository}". ${disclosure} If the repository-pinned Betterleaks 1.8.1 runtime is reported missing or corrupt, this task may run only the repository-provided checksum-pinned Betterleaks 1.8.1 installer command: python -m ouroboros.betterleaks_runtime install. No other dependency or runtime change is authorized. No account or authentication change is authorized.`,
        context: `The owner explicitly confirmed publication of "${skill}" to "${repository}". ${disclosure} Recoverable tool errors are evidence for the next LLM turn, not completion. Use the typed stage, completed external effects, redacted facts, and repair hint as context for the agent's own judgment. Selected-preflight evidence: ${JSON.stringify(evidence)}`,
        metadata: {
            skill_publish_target: { skill, repository, ...(destination ? { destination } : {}) },
        },
    };
}

/**
 * Selected-skill publish flow. The preflight runs exactly once; only literal
 * boolean true from the shared dialog authorizes one ordinary task creation.
 */
export async function runSkillPublishFlow(skill, {
    preflightImpl = skillPublishPreflight,
    dialogImpl = openConfirmDialog,
    createTaskImpl = createTask,
    projectId = '',
    chatId,
    destination = null,
    privateRepository,
} = {}) {
    const requestedSkill = text(typeof skill === 'string' ? skill : skill?.name);
    if (!requestedSkill) throw new Error('Skill name is required for publication.');

    let preflight;
    try {
        preflight = await (destination ? preflightImpl(requestedSkill, destination) : preflightImpl(requestedSkill));
    } catch (error) {
        const typed = error?.body || error?.payload;
        if (!typed || typed.state !== 'hard_block') throw error;
        preflight = typed;
    }

    const model = skillPublishDialogModel(preflight, { privateRepository });
    const decision = await dialogImpl(model.dialog);
    if (!model.canStart) {
        return {
            started: false,
            reason: model.state === 'hard_block' ? 'hard_block' : 'not_started',
            preflight,
        };
    }
    if (decision !== true) {
        return { started: false, reason: 'cancelled', preflight };
    }

    const payload = buildSkillPublishTask(preflight, { projectId, chatId, destination, privateRepository });
    const task = await createTaskImpl(payload);
    return { started: true, task, payload, preflight };
}
