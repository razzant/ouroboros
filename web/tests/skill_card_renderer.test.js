// Installed skill cards (Skills → Installed): one head shared with Browse, the switch as
// the run state (a chip only for a state it cannot show), at most two attention trays (one sentence + its actions), and
// Details for the record. Rows come from the listing fixture shaped as SPEC §8.1.
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';

import {
    continuableReview, fillFindingsOnOpen, installedSkillModel, installedWhoHtml, isMadeHere, renderInstalledSkillCard,
    reviewSentence,
} from '../modules/skill_card_renderer.js';

const LISTING = JSON.parse(readFileSync(new URL('./fixtures/skills_v4/extensions_v4.json', import.meta.url), 'utf8'));
const row = (name) => structuredClone(LISTING.skills.find((skill) => skill.name === name));
const FIXES = JSON.parse(readFileSync(new URL('./fixtures/skills_v4/extensions_fix_r1.json', import.meta.url), 'utf8')).skills;
const ROUND2 = JSON.parse(readFileSync(new URL('./fixtures/skills_v4/extensions_fix_r2.json', import.meta.url), 'utf8')).skills;
const chipOf = (html) => html.match(/<span class="skills-card-status"><span class="ui-chip skills-chip" data-tone="([^"]+)"[^>]*>(?:<span class="status-mark"[^>]*>(?:<i><\/i>){3}<\/span>)?([^<]+)</);
const MOVING = /<span class="status-mark" aria-hidden="true" data-shape="moving" data-hue="ink">/;
const traysOf = (html) => html.split('data-skill-trays>')[1].split('<div data-skill-errors>')[0];

function skill(overrides = {}) {
    return {
        name: 'demo', type: 'extension', version: '1.0.0', description: 'Demo skill', enabled: false,
        source: 'external', payload_root: 'skills/external/demo', review_status: 'clean', review_stale: false,
        review_gate: { executable_review: true }, executable_review: true, permissions: [],
        grants: { all_granted: true }, owner_attestable: true, ...overrides,
    };
}

test('one head: the name opens the detail sheet, marks stay beside it, the who line names the source', () => {
    const html = renderInstalledSkillCard(row('weather'));
    assert.match(html, /<button type="button" class="skills-card-name ui-hit-area" data-skill="weather" data-skill-action="open_detail"><span data-i18n-authored>weather<\/span><span class="skills-card-name-more" aria-hidden="true">›<\/span><\/button>/);
    assert.match(html, /data-official-mark[^>]*>[\s\S]*?Official<\/button>/, 'Official comes from the install record');
    assert.match(installedWhoHtml(row('weather')), /^Ouroboros · v0\.3\.1 · installed .* by Ouroboros$/);
    assert.match(installedWhoHtml(row('telegram')), /^Built in · v1\.3\.0 · came with Ouroboros /);
    assert.match(installedWhoHtml(row('zoom_meeting')), /^Made here · v0\.1\.0 · made here /);
    assert.equal((renderInstalledSkillCard(row('telegram')).match(/data-official-mark/g) || []).length, 1, 'built-in release carries one Official mark');
    assert.doesNotMatch(html, / title="/, 'no explanation lives in a hover title');
});

test('Details lists each placed control in the Details row layout with a polarity-true Hide/Show', () => {
    const rows = [{ key: 'demo:chip', label: 'Meter', place: 'Chip above the composer', hidden: false },
        { key: 'demo:save', label: 'Save', place: 'On replies', hidden: true }];
    const html = renderInstalledSkillCard(skill(), new Set(), new Set(), {}, { contributionVisibility: () => rows });
    const section = html.split('data-skill-detail-contributions>')[1].split('<div class="skills-details-facts" data-skill-detail-facts>')[0];
    assert.match(section, /<span class="skills-detail-label">Shown in chat<\/span>/);
    assert.match(section, /data-contribution-row="demo:chip"><span class="skills-detail-label">Meter<\/span><span class="skills-detail-value">Chip above the composer <button[^>]*data-skill-action="toggle_contribution"[^>]*data-contribution-key="demo:chip" data-contribution-hidden="false" aria-label="Hide Meter, chip above the composer">Hide<\/button>/);
    assert.match(section, /On replies · hidden <button[^>]*data-contribution-hidden="true" aria-label="Show Save, on replies">Show<\/button>/);
    assert.doesNotMatch(section, /<small>|<strong>|<h4>/, 'no raw small text or ad-hoc headings');
    assert.doesNotMatch(renderInstalledSkillCard(skill(), new Set(), new Set(), {}, { contributionVisibility: () => [] }),
        /data-skill-detail-contributions/, 'a skill without placed controls has no such rows');
});

test('normal On and Off state lives in the switch; only attention states get a status chip', () => {
    assert.equal(chipOf(renderInstalledSkillCard(row('telegram'))), null);
    assert.equal(chipOf(renderInstalledSkillCard(skill())), null);
    // Amber is the owner's move: a grant, a review, a review to try again (each offers the owner's own action).
    assert.deepEqual(chipOf(renderInstalledSkillCard(row('zoom_meeting'))).slice(1), ['warn', 'Needs access']);
    assert.deepEqual(chipOf(renderInstalledSkillCard(row('claudexor_quotas'))).slice(1), ['warn', 'Not reviewed yet']);
    const unfinished = skill({ review_status: 'pending', review_gate: { executable_review: false }, executable_review: false,
        skill_review: { current: { lifecycle_status: 'timeout', terminal_reason: 'the reviewer timed out' } } });
    assert.deepEqual(chipOf(renderInstalledSkillCard(unfinished)).slice(1), ['warn', 'Review didn’t finish']);
    assert.equal(installedSkillModel(unfinished).needs, true);
    assert.deepEqual(chipOf(renderInstalledSkillCard(row('minecraft-widget'))).slice(1), ['danger', 'Review found problems']);
    // Skipping the review was the owner's own choice: a neutral fact in Details, never amber.
    assert.match(renderInstalledSkillCard(skill({ review_profile: 'owner_attested' })), /data-tone="neutral">review skipped by you</);
    // A job in flight goes on by itself: a neutral chip with the shared moving mark.
    assert.deepEqual(chipOf(renderInstalledSkillCard(row('whoop-health'))).slice(1), ['neutral', 'Reviewing…']);
    assert.match(renderInstalledSkillCard(row('whoop-health')), MOVING, 'live work is the moving mark');
    assert.deepEqual(chipOf(renderInstalledSkillCard(skill(), new Set(), new Set(['demo']))).slice(1), ['neutral', 'Asking Ouroboros…']);
    assert.doesNotMatch(renderInstalledSkillCard(row('telegram')), /status-mark/, 'nothing moves when nothing works');
});

test('each attention tray is one sentence with its own actions, visible without hover', () => {
    const access = traysOf(renderInstalledSkillCard(row('zoom_meeting')));
    assert.match(access, /Wants your permission to live in a place for you: hear, speak and act there\./);
    assert.match(access, /data-skill-action="grant" data-keys="presence"[^>]*>Grant access</);

    const problems = traysOf(renderInstalledSkillCard(row('minecraft-widget')));
    assert.match(problems, /data-tone="danger"/);
    assert.match(problems, /Review found 1 problem and 1 note\. First: “widget\.js::set permits a new removal when modified already has 4096 keys\.”/);
    assert.match(problems, /Installed, but it doesn’t run and gets no keys until you decide\./);
    assert.match(problems, />Read findings</);
    assert.match(problems, /data-skill-action="repair"[^>]*>Ask Ouroboros to fix</);
    assert.match(problems, /skills-skip-btn[^>]*data-skill-action="attest_review"[^>]*><span class="skills-skip-glyph" aria-hidden="true"><svg[^>]*stroke="currentColor"/);

    const reviewing = traysOf(renderInstalledSkillCard(row('whoop-health')));
    assert.match(reviewing, /Ouroboros reviews it before it runs or gets your keys\./);
    assert.match(reviewing, />Skip review</);

    const otherEdit = traysOf(renderInstalledSkillCard(row('claudexor_quotas')));
    assert.match(otherEdit, /Its files changed here since it was installed, so it waits for a review\. Review it, or skip the review\./);
    assert.match(otherEdit, /data-skill-action="rereview"[^>]*>Review</);
});

test('page-module faults and holds are owner-visible and keep recovery discoverable', () => {
    const faulty = renderInstalledSkillCard(skill({ enabled: true }), new Set(), new Set(), {}, {
        pageModules: { faults: [{ skill: 'demo', reason: 'boom', current: true }], held: [] },
    });
    assert.deepEqual(chipOf(faulty).slice(1), ['warn', 'Off in this window']);
    assert.match(traysOf(faulty), /page module stopped in this window: boom/);
    assert.doesNotMatch(traysOf(faulty), /data-host-module-key/);
    const held = renderInstalledSkillCard(skill({ enabled: true }), new Set(), new Set(), {}, {
        pageModules: { faults: [], held: [{ key: 'demo:voice', skill: 'demo', reason: 'start' }] },
    });
    assert.match(traysOf(held), /Held back: the page stopped while this was starting last time\./);
    assert.match(held, /data-skill-action="start_host_module"[^>]*data-host-module-key="demo:voice"/);
    const stale = renderInstalledSkillCard(skill({ enabled: true }), new Set(), new Set(), {}, {
        pageModules: { faults: [{ skill: 'demo', reason: 'old', current: false }], held: [] },
    });
    assert.equal(chipOf(stale), null, 'stale faults do not add a chip to the normal enabled switch');
    assert.match(stale, /class="skills-toggle"[^>]*checked/);
});

test('faults and held siblings of one skill both retain their page facts', () => {
    const html = renderInstalledSkillCard(skill({ enabled: true }), new Set(), new Set(), {}, {
        pageModules: {
            faults: [{ skill: 'demo', reason: 'voice failed', current: true }],
            held: [{ key: 'demo:tts', skill: 'demo', reason: 'start' }],
        },
    });
    assert.match(html, /page module stopped in this window: voice failed/);
    assert.match(html, /data-skill-action="start_host_module"[^>]*data-host-module-key="demo:tts"/);
});

test('page attention keeps an overflow update fact in Details', () => {
    const row = skill({
        enabled: true,
        install: { update_available: { version: '2.0.0' }, release: { version: '1.0.0' } },
    });
    const html = renderInstalledSkillCard(row, new Set(['demo']), new Set(), {}, {
        pageModules: { faults: [], held: [{ key: 'demo:voice', skill: 'demo', reason: 'start' }] },
    });
    assert.match(html, /data-skill-detail-overflow/);
    assert.match(html, /2\.0\.0 is available/);
    assert.match(html, /data-skill-action="update"/);
});

test('updates: one click for an unmodified skill, «Update…» asks what to keep for a modified one', () => {
    const plain = traysOf(renderInstalledSkillCard(row('weather')));
    assert.match(plain, /v0\.3\.2 is available\./);
    assert.match(plain, /data-skill-action="update"[^>]*>Update</);
    const modified = renderInstalledSkillCard(row('anime_studio'));
    assert.match(traysOf(modified), /Ouroboros Community has v3\.2\.0\. You changed this skill here, so updating asks what to keep\./,
        'the host recorded the owner’s edit of these bytes');
    const unrecordedUpdate = row('anime_studio');
    unrecordedUpdate.review_gate = { executable_review: false, blocking_reason: 'review_pending' };
    assert.match(traysOf(renderInstalledSkillCard(unrecordedUpdate)), /This skill was changed here, so updating asks what to keep\./);
    assert.match(traysOf(modified), />Update…</);
    assert.match(modified, /Modified · edited by you/, 'the host recorded the owner’s edit of these bytes');
    assert.equal(installedWhoHtml(row('anime_studio')).split(' · ').slice(-1)[0], 'edited by you here (now v3.3.0)');
    assert.match(installedWhoHtml(row('anime_studio')), /^Ouroboros Community · installed v2\.12\.0 /);
    const unrecorded = row('claudexor_quotas');
    assert.match(renderInstalledSkillCard(unrecorded), /data-tone="neutral">Modified</, 'nobody recorded who changed it');
    assert.equal(installedWhoHtml(unrecorded).split(' · ').slice(-1)[0], 'changed here');
    const unknownBase = row('weather');
    unknownBase.install.base_known = false;
    assert.match(traysOf(renderInstalledSkillCard(unknownBase)), /Ouroboros has v0\.3\.2\. Its original files aren’t known, so updating asks what to keep\./);
    const sameLabel = row('weather');
    sameLabel.install.update_available.version = '0.3.1';
    assert.match(traysOf(renderInstalledSkillCard(sameLabel)), /changed the files of v0\.3\.1 \(same version label\)/);
});

test('withdrawn and left-source skills say so; a withdrawn one offers Uninstall', () => {
    const withdrawn = traysOf(renderInstalledSkillCard(row('old_tool')));
    assert.match(withdrawn, /Ouroboros Community withdrew v0\.2\.0 on 16 Oct: Sends data to an unnamed server\./);
    assert.match(withdrawn, /data-skill-action="uninstall"[^>]*>Uninstall</);
    assert.equal(installedSkillModel(row('old_tool')).needs, true);
    assert.match(traysOf(renderInstalledSkillCard(row('gone_tool'))), /No longer in Ouroboros Community\. It keeps working; it won’t get updates\./);
    assert.equal(installedSkillModel(row('gone_tool')).needs, false);
});

test('a rejected update names the restored state and offers findings and owner skip; a pending update stays off', () => {
    const rejected = FIXES.find((skill) => skill.name === 'update_rejected');
    const html = renderInstalledSkillCard(rejected);
    assert.match(traysOf(html), /its review found problems; the previous version runs\./);
    assert.match(traysOf(html), /data-skill-action="read_findings"[^>]*>See findings</);
    assert.match(traysOf(html), /data-skill-action="update_without_review"[^>]*>Update without review</);
    assert.match(traysOf(html), /data-skill-action="update"[^>]*>Update</);
    assert.equal(chipOf(html), null);
    const pending = renderInstalledSkillCard(FIXES.find((skill) => skill.name === 'pending_update'));
    assert.match(traysOf(pending), /Updating to 2\.0\.0: it stays off until the review finishes\./);
    assert.match(pending, /class="skills-toggle"[^>]*disabled/);
    assert.doesNotMatch(traysOf(pending), /data-skill-action="update"/);
    const ordinary = renderInstalledSkillCard(row('weather'));
    assert.doesNotMatch(ordinary, /previous version runs|data-skill-action="update_without_review"/);
});

test('self-check is escaped, abbreviated in the review sentence, complete in Details and has no pending chip', () => {
    const checked = FIXES.find((skill) => skill.name === 'self_checked');
    const html = renderInstalledSkillCard(checked);
    assert.match(reviewSentence(checked), /^Ouroboros read it before accepting it and recorded why: “/);
    assert.ok(reviewSentence(checked).length < checked.install.check.notes.length);
    assert.doesNotMatch(html, /<script>|>pending<|Reviewed on this computer/);
    assert.match(html, /&lt;script&gt;/);
    assert.ok(html.includes(checked.install.check.notes.replaceAll('<', '&lt;').replaceAll('>', '&gt;').replaceAll('"', '&quot;')));
    const changed = structuredClone(checked);
    changed.review_gate = { ...changed.review_gate, executable_review: false, blocking_reason: 'review_pending' };
    changed.executable_review = false;
    assert.equal(reviewSentence(changed), 'Not reviewed for its current files.');
});

test('a passed review stays the evidence while a newer round has no verdict: one sentence, no pending chip', () => {
    const detailsReview = (html) => html.split('data-skill-detail-review>')[1].split('</div>')[0];
    const waiting = skill({ review_status: 'pending', skill_review: { current: { review_round: 2, snapshot_attempt: 2, content_hash: 'd853fe070533aa', status: 'pending' }, history: [] } });
    assert.equal(reviewSentence(waiting), 'Reviewed on this computer · passed · a new review (round 2) is pending.');
    assert.doesNotMatch(detailsReview(renderInstalledSkillCard(waiting)), /ui-chip/, 'no verdict yet is no chip, never a green pending');
    // Without a newer round the passed review is said alone, beside its verdict.
    assert.equal(reviewSentence(skill()), 'Reviewed on this computer · passed.');
    assert.match(detailsReview(renderInstalledSkillCard(skill())), /data-tone="ok"[^>]*>clean</);
});

test('per-release withdrawal names the withdrawn version supplied by the host', () => {
    const withdrawn = structuredClone(FIXES.find((skill) => skill.name === 'old_tool'));
    withdrawn.install.withdrawn.version = '0.1.9';
    assert.match(traysOf(renderInstalledSkillCard(withdrawn)), /Version 0\.1\.9 was withdrawn by Ouroboros Community: Sends data to an unnamed server\./);
    withdrawn.install.update_available = { version: '0.1.8', digest: 'c'.repeat(64) };
    const tray = traysOf(renderInstalledSkillCard(withdrawn));
    assert.match(tray, /data-skill-action="update"[^>]*>Update</);
    assert.match(tray, /data-skill-action="uninstall"[^>]*>Uninstall</);
    assert.match(tray, /v0\.1\.8/);
    delete withdrawn.install.withdrawn.version;
    assert.doesNotMatch(traysOf(renderInstalledSkillCard(withdrawn)), /data-skill-action="update"/);
});

test('each rejected update states its cause and keeps the reviewed retry; only findings have a findings action', () => {
    const causes = { rejected_newer: 'its review found problems', review_unfinished: 'its review didn&#39;t finish',
        preflight_failed: 'its file checks failed', landing_failed: 'it couldn&#39;t be installed',
        registry_rejected: 'its review didn&#39;t finish' };
    for (const row of ROUND2) {
        const tray = traysOf(renderInstalledSkillCard(row));
        assert.ok(tray.includes(`${causes[row.name]}; the previous version runs.`), tray);
        assert.equal(tray.includes('data-skill-action="update"'), Boolean(row.install.update_available));
        assert.equal(tray.includes('data-skill-action="read_findings"'), ['review_blockers', 'preflight_failed'].includes(row.install.update_rejected.reason));
        assert.equal(tray.includes('data-skill-action="update_without_review"'), row.name !== 'rejected_newer');
        if (row.name === 'rejected_newer') assert.match(tray, /v0\.3\.3 is available/);
        const unavailable = structuredClone(row);
        unavailable.install.update_available = null;
        assert.doesNotMatch(traysOf(renderInstalledSkillCard(unavailable)), /data-skill-action="update"/);
        assert.match(traysOf(renderInstalledSkillCard(unavailable)), /data-skill-action="update_without_review"/);
        delete unavailable.install.update_available;
        assert.match(traysOf(renderInstalledSkillCard(unavailable)), /data-skill-action="update_without_review"/);
        delete unavailable.install.update_rejected.digest;
        assert.doesNotMatch(traysOf(renderInstalledSkillCard(unavailable)), /data-skill-action="update_without_review"/);
    }
});

test('conflicts lock an off switch with a visible reason but never trap an enabled skill', () => {
    const conflict = { code: 'skill_conflict', skills: ['beta'], omitted: 0 };
    const off = renderInstalledSkillCard(skill({ conflict }));
    assert.match(off, /class="ui-switch is-locked"/);
    assert.match(off, /input type="checkbox" class="skills-toggle"[^>]*disabled/);
    assert.match(traysOf(off), /Uses the same tool names as beta\. Turn one of them off to use the other\./);
    assert.match(off, /aria-describedby="skill-trays-demo"/, 'the locked switch points at the visible explanation');
    const on = renderInstalledSkillCard(skill({ conflict, enabled: true }));
    assert.doesNotMatch(on, /input type="checkbox" class="skills-toggle"[^>]*disabled/);
});

test('a pending lifecycle job shows its words and locks the switch on the intended side', () => {
    const html = renderInstalledSkillCard(skill({ enabled: true, lifecycle_pending: true, lifecycle_kind: 'disable', lifecycle_status: 'running' }));
    assert.deepEqual(chipOf(html).slice(1), ['neutral', 'Turning off…']);
    assert.match(html, MOVING);
    assert.match(html, /input type="checkbox" class="skills-toggle"[^>]*data-skill="demo"\s+disabled/);
    const failed = renderInstalledSkillCard(skill({ lifecycle_status: 'failed', lifecycle_kind: 'enable', lifecycle_error: 'reconcile failed' }));
    assert.deepEqual(chipOf(failed).slice(1), ['danger', 'Enable failed']);
    assert.match(failed, /reconcile failed/);
});

test('#335: a repairable preflight failure offers Ask Ouroboros to fix and never Skip review', () => {
    const preflight = skill({
        type: 'extension', review_status: 'pending', review_gate: { executable_review: false, preflight_failed: true },
        executable_review: false,
        review_findings: [{ item: 'skill_preflight', verdict: 'FAIL', severity: 'critical',
            reason: JSON.stringify({ manifest: [{ item: 'manifest_entry_exists', ok: false, detail: 'missing entry: plugin.py' }], ok: false }) }],
    });
    const html = renderInstalledSkillCard(preflight);
    assert.match(traysOf(html), /First: “Preflight failed — manifest_entry_exists: missing entry: plugin\.py”/);
    assert.match(traysOf(html), /data-skill-action="repair"[^>]*>Ask Ouroboros to fix</);
    assert.doesNotMatch(html, /attest_review/, 'Skip review would be refused by the file check');
    assert.doesNotMatch(html, /skills-menu-item skills-review"/, 'Review again would fail the same way');
    // A non-repairable native payload keeps its Review path.
    const native = renderInstalledSkillCard({ ...preflight, source: 'native', payload_root: '' });
    assert.doesNotMatch(native, /data-skill-action="repair"/);
    assert.match(native, /skills-menu-item skills-review"/);
});

test('a stale recorded preflight failure keeps Review first and offers the fix from the menu (D11)', () => {
    const html = renderInstalledSkillCard(skill({
        review_status: 'pending', review_stale: true, executable_review: false,
        review_gate: { executable_review: false, preflight_failed: false, preflight_failed_stale: true },
    }));
    assert.match(traysOf(html), /The last file check failed on earlier files\. Review the current files, or ask Ouroboros to fix it\./);
    assert.match(traysOf(html), /data-skill-action="rereview"[^>]*>Review</);
    assert.match(html, /skills-menu-item skills-repair-stale" data-skill="demo" data-skill-action="repair">Ask Ouroboros to fix</);
});

test('the menu carries details, review again, make runnable, publish and exactly the removal the host names', () => {
    const made = renderInstalledSkillCard(skill({ type: 'instruction', source: 'self_authored', is_self_authored: true, removal: 'delete' }));
    assert.match(made, /skills-make-runnable" data-skill="demo" data-skill-action="repair">Make runnable</);
    assert.match(made, /skills-delete-local"[^>]*data-skill-action="delete"[^>]*>Delete</);
    assert.doesNotMatch(made, /skills-uninstall/);
    const sourced = renderInstalledSkillCard(row('weather'));
    assert.match(sourced, /skills-uninstall"[^>]*>Uninstall</);
    assert.doesNotMatch(sourced, /skills-delete-local/);
    // Inverted pin: a built-in is removed like any skill (owner decision 17.10 «7. A»); the host says so.
    assert.match(renderInstalledSkillCard(row('telegram')), /skills-uninstall"[^>]*>Uninstall</);
    const unnamed = renderInstalledSkillCard({ ...row('weather'), removal: '' });
    assert.doesNotMatch(unnamed, /skills-uninstall|skills-delete-local/, 'the card derives no removal of its own');
});

test('Publish follows the host verdict: task_start_allowed decides, a missing token is said in Details', () => {
    const allowed = renderInstalledSkillCard(skill({ submit_hub: { visible: true, task_start_allowed: true, disabled: true, reason: '' } }));
    assert.match(allowed, /skills-submit-hub"[^>]*data-submit-disabled="false"[^>]*>Publish…</);
    const legacy = renderInstalledSkillCard(skill(), new Set(), new Set(), {}, { githubTokenConfigured: false });
    assert.match(legacy, /data-submit-disabled="true"/);
    assert.match(legacy, /Publishing needs a GitHub token: Settings › Accounts and keys › GitHub\./);
    assert.doesNotMatch(renderInstalledSkillCard(skill({ source: 'native' })), /skills-submit-hub/);
});

test('Details keep the record: review sentence and verdict, what it can use, the install record and live state', () => {
    const weather = renderInstalledSkillCard(row('weather'), new Set(), new Set(), LISTING.live);
    assert.match(weather, /Installed without review on 20 Sept? because you trust Ouroboros\./, 'the locale’s own month abbreviation');
    assert.doesNotMatch(weather, /data-tone="ok">pending</, 'no «pending» verdict beside a skill that runs on its install basis');
    assert.match(weather, /Internet · A widget and a settings section · Its own pages inside Ouroboros/);
    // Ouroboros installed it: the host's listing carries the task and the reason from the install record (B8).
    assert.match(weather, /v0\.3\.1 · <code>0305fd55a4c7786753043df9<\/code> · installed without review — you trusted Ouroboros · by Ouroboros · in task t-weather · because: the owner asked for the forecast</);
    assert.match(weather, />Loaded</);
    // The host's acceptance basis for these exact bytes decides the sentence, not the install basis alone.
    const install = { source: { id: 'ouroboros', name: 'Ouroboros' }, basis: 'trusted', installed_at: '2026-10-12T10:00:00Z', release: { version: '1.0.0' } };
    const trusted = skill({ review_status: 'pending', review_gate: { executable_review: true, blocking_reason: 'trusted_source' }, install });
    assert.equal(reviewSentence(trusted), 'Installed without review on 12 Oct because you trust Ouroboros.');
    const changed = skill({ review_status: 'pending', review_gate: { executable_review: false, blocking_reason: 'review_pending' }, install });
    assert.equal(reviewSentence(changed), 'Not reviewed for its current files.', 'a trusted install whose bytes changed is not «installed without review»');
    const reviewed = skill({ review_status: 'clean', review_gate: { executable_review: true, blocking_reason: 'trusted_source' }, install });
    assert.equal(reviewSentence(reviewed), 'Reviewed on this computer · passed.', 'a passed review of these bytes is said first');
    assert.equal(reviewSentence(skill({ review_status: 'pending', review_gate: { executable_review: true, blocking_reason: 'owner_skip' } })),
        'You skipped the review; it runs unreviewed.');
    assert.equal(reviewSentence(row('anime_studio')), 'Edited by you here; it runs without a new review.');
    assert.equal(reviewSentence(skill({ review_profile: 'owner_attested' })), 'You skipped the review; it runs unreviewed.');
    // Without a recorded task or reason the line says neither.
    const untold = row('weather');
    delete untold.install.task_id;
    delete untold.install.reason;
    assert.match(renderInstalledSkillCard(untold), /· by Ouroboros</);
    const worker = renderInstalledSkillCard(skill({ enabled: true, live_loaded: true, dispatch_live: true, process: 'worker' }));
    assert.match(worker, />Loaded · worker</);
});

test('a place is the owner\'s choice on its sheet: no Presence runtime form on the card, even from an old listing', () => {
    const html = renderInstalledSkillCard(skill({
        presence_runtime: {
            defaults: { model_slot: 'main', inline_max_rounds: 10 }, overrides: { model_slot: 'light', inline_max_rounds: 7 },
            workspace_root: '/work/reports & notes', state_fingerprint: 'a'.repeat(64),
        },
    }));
    assert.doesNotMatch(html, /presence-runtime|name="workspace_root"|Inline rounds/);
});

test('a connector card wears its icon and «Connects …»; a chat writer without a place says whose its messages are', () => {
    const connector = renderInstalledSkillCard(skill({ connects: ['Slack'], icon: { name: 'chat' } }));
    assert.match(connector, /<span class="skills-card-icon" aria-hidden="true"><svg/);
    assert.match(connector, /data-tone="neutral">Connects Slack</);
    assert.doesNotMatch(renderInstalledSkillCard(skill()), /skills-card-icon|Connects /);
    const bridge = renderInstalledSkillCard(skill({ permissions: ['net', 'inject_chat'] }));
    assert.match(bridge, /<p class="skills-card-facts">Messages from here count as yours<\/p>/);
    const place = renderInstalledSkillCard(skill({ permissions: ['inject_chat', 'presence'], connects: ['A2A'] }));
    assert.doesNotMatch(place, /count as yours/, 'a place declaration speaks for its messages instead');
});

test('findings are built on first open from the rows in memory, once', () => {
    const rows = [row('minecraft-widget')];
    const details = (open, built = false, name = 'minecraft-widget') => ({
        open, dataset: { skillFindings: name }, childElementCount: built ? 2 : 1, inserted: [],
        insertAdjacentHTML(where, markup) { this.inserted.push([where, markup]); },
    });
    const closed = details(false);
    assert.equal(fillFindingsOnOpen(closed, rows), false);
    const opened = details(true);
    assert.equal(fillFindingsOnOpen(opened, rows), true);
    assert.match(opened.inserted[0][1], /widget\.js::set permits/);
    assert.equal(fillFindingsOnOpen(details(true, true), rows), false, 'a built list survives re-opening');
    const gone = details(true, false, 'vanished');
    fillFindingsOnOpen(gone, rows);
    assert.match(gone.inserted[0][1], /not in the current list/);
    assert.equal(fillFindingsOnOpen({ open: true, dataset: {} }, rows), false, 'other disclosures are not ours');
});

test('made here means created on this computer without an install record', () => {
    assert.equal(isMadeHere(row('zoom_meeting')), true);
    assert.equal(isMadeHere(row('weather')), false);
    assert.equal(isMadeHere(row('telegram')), false);
});

test('skill titles prefer the optional title, then connector label, then canonical id', () => {
    const connector = { connects: ['Mail'], contributes: [{ provides: 'place', label: 'Mailbox' }] };
    for (const [fields, expected] of [[{ ...connector, title: 'Email tools' }, 'Email tools'],
        [connector, 'Mailbox'], [{ connects: ['Mail'] }, 'demo'], [{}, 'demo']]) {
        const html = renderInstalledSkillCard(skill(fields));
        assert.ok(html.includes(`<span data-i18n-authored>${expected}</span>`));
        assert.match(html, /data-skill="demo"/);
        assert.match(html, /skills-detail-label">ID<\/span><span class="skills-detail-value"><code data-i18n-authored>demo<\/code><\/span>/);
    }
});

test('a stopped Skills-page reviewer of these bytes is continued by its own action beside a new review', () => {
    // The real projection after a stopped seat (gateway /api/extensions): the job failed without a verdict
    // and its record's card row offers the seat (skill_review_runner._ledger_history_rows).
    const offer = { seat_id: 'review-s', operation_id: 'op-1', run_id: 'run-1', cause: 'Account limit reached' };
    const stopped = (rows) => skill({ review_status: 'pending', review_gate: { executable_review: false }, executable_review: false,
        content_hash: 'h1', skill_review: { current: { lifecycle_status: 'failed', content_hash: 'h1', record_id: 'rl-1',
            terminal_reason: 'no verdict was formed' }, history: rows } });
    const row = { record_id: 'rl-1', content_hash: 'h1', task_id: 'api_skill_review', continuable_seats: [offer] };
    const html = traysOf(renderInstalledSkillCard(stopped([row])));
    assert.match(html, /The review didn’t finish: no verdict was formed/);
    assert.match(html, /A reviewer stopped with work it can continue from where it stopped\./);
    const actions = [...html.matchAll(/<button[^>]*data-skill-action="([^"]+)"[^>]*>([^<]+)</g)].map((m) => [m[1], m[2]]);
    assert.deepEqual(actions.slice(0, 2), [['continue_review', 'Continue review'], ['review', 'Try again']]);
    assert.deepEqual(continuableReview(stopped([row])), row);
    // Other bytes, nothing offered, or no record: a new review is the only move.
    for (const other of [{ ...row, content_hash: 'h0' }, { ...row, continuable_seats: [] }, { ...row, record_id: '' }]) {
        assert.equal(continuableReview(stopped([other])), null);
        assert.doesNotMatch(renderInstalledSkillCard(stopped([other])), /continue_review/);
    }
});

test('a declared secret that is not set is a neutral tray with Open Accounts and keys and puts the card in Needs you; unknown is said, never a value', () => {
    const fields = (state) => ({ secret_config: { fields: [{ id: 'botToken', key: 'TELEGRAM_BOT_TOKEN', state }] } });
    const missing = renderInstalledSkillCard(skill({ enabled: true, ...fields('missing') }));
    assert.match(traysOf(missing), /TELEGRAM_BOT_TOKEN is not set\.<div class="skills-tray-note">Save it in Settings › Accounts and keys\.<\/div>/);
    assert.doesNotMatch(traysOf(missing), /next read|some skills|Disable and Enable/, 'when it applies is the skill’s own status');
    assert.match(traysOf(missing), /data-skill-action="open_secrets"[^>]*>Open Accounts and keys</);
    // The button names the keys it is about, so Accounts and keys opens at that key's row (skills.js `row`).
    const keysOf = (html) => /data-skill-action="open_secrets" data-keys="([^"]*)"/.exec(traysOf(html))?.[1];
    assert.equal(keysOf(missing), 'TELEGRAM_BOT_TOKEN');
    const two = renderInstalledSkillCard(skill({ enabled: true, secret_config: { fields: [
        { id: 'botToken', key: 'TELEGRAM_BOT_TOKEN', state: 'missing' }, { id: 'proxy', key: 'TELEGRAM_PROXY', state: 'missing' }] } }));
    assert.match(traysOf(two), /TELEGRAM_BOT_TOKEN, TELEGRAM_PROXY are not set\.<div class="skills-tray-note">Save them in Settings › Accounts and keys\.<\/div>/);
    assert.equal(keysOf(two), 'TELEGRAM_BOT_TOKEN,TELEGRAM_PROXY');
    const mixed = renderInstalledSkillCard(skill({ enabled: true, secret_config: { fields: [
        { id: 'proxy', key: 'TELEGRAM_PROXY', state: 'unknown' }, { id: 'botToken', key: 'TELEGRAM_BOT_TOKEN', state: 'missing' }] } }));
    assert.equal(keysOf(mixed), 'TELEGRAM_BOT_TOKEN', 'a key that is not set leads; one that cannot be read waits');
    assert.doesNotMatch(traysOf(missing), /data-tone="(warn|danger)"/, 'a fact, not the owner’s amber move');
    assert.equal(chipOf(missing), null);
    // Saving it is the owner's move, so the card is under Needs you; the tray stays neutral and adds no chip.
    assert.equal(installedSkillModel(skill({ enabled: true, ...fields('missing') })).needs, true);
    assert.match(missing, /<article [^>]*data-needs-you[ >]/);
    const unknown = traysOf(renderInstalledSkillCard(skill({ enabled: true, ...fields('unknown') })));
    assert.match(unknown, /Saved settings can’t be read now, so whether TELEGRAM_BOT_TOKEN is set is unknown\./);
    assert.doesNotMatch(unknown, /skills-tray-note|Save it/, 'an unreadable value is not a reason to save it again');
    assert.match(unknown, /data-skill-action="open_secrets" data-keys="TELEGRAM_BOT_TOKEN"/);
    // Configured, an off skill, or a grant still missing: nothing here (the grant tray speaks for itself).
    assert.doesNotMatch(renderInstalledSkillCard(skill({ enabled: true, ...fields('configured') })), /open_secrets/);
    assert.doesNotMatch(renderInstalledSkillCard(skill({ ...fields('missing') })), /open_secrets/);
    assert.doesNotMatch(renderInstalledSkillCard(skill({ enabled: true, grants: { all_granted: false, missing_keys: ['TELEGRAM_BOT_TOKEN'] },
        ...fields('missing') })), /open_secrets/);
    // Needs you follows the tray's own condition: an unreadable, a configured or an off skill's secret asks nothing;
    // a grant still missing is already Needs access, once.
    for (const overrides of [{ enabled: true, ...fields('unknown') }, { enabled: true, ...fields('configured') }, fields('missing')]) {
        assert.equal(installedSkillModel(skill(overrides)).needs, false, JSON.stringify(overrides));
        assert.doesNotMatch(renderInstalledSkillCard(skill(overrides)), /data-needs-you/);
    }
    const ungranted = skill({ enabled: true, grants: { all_granted: false, missing_keys: ['TELEGRAM_BOT_TOKEN'] }, ...fields('missing') });
    assert.equal(installedSkillModel(ungranted).needs, true);
    assert.deepEqual(chipOf(renderInstalledSkillCard(ungranted)).slice(1), ['warn', 'Needs access']);
});

test('a grant left behind by a byte change is named as the earlier version’s grant, never as missing keys', () => {
    const earlier = { all_granted: false, missing_keys: ['GOOGLE_OAUTH_CLIENT_ID'],
        earlier_grant: { keys: ['GOOGLE_OAUTH_CLIENT_ID'], permissions: [] } };
    const named = traysOf(renderInstalledSkillCard(skill({ enabled: true, grants: earlier })));
    assert.match(named, /Wants to use GOOGLE_OAUTH_CLIENT_ID\./);
    assert.match(named, /class="skills-tray-note">You granted this to an earlier version of the skill/);
    // A new version asking for more than the earlier grant covered: the note does not claim all of it.
    const wider = traysOf(renderInstalledSkillCard(skill({ enabled: true, grants: { ...earlier, missing_keys: ['GOOGLE_OAUTH_CLIENT_ID', 'NEW_KEY'] } })));
    assert.match(wider, /Wants to use GOOGLE_OAUTH_CLIENT_ID, NEW_KEY\./);
    assert.match(wider, /You granted some of this to an earlier version/);
    assert.match(named, /data-skill-action="grant" data-keys="GOOGLE_OAUTH_CLIENT_ID"/, 'the same Grant access button restores it');
    // Never granted: the tray asks, and says nothing about an earlier version.
    const fresh = traysOf(renderInstalledSkillCard(skill({ enabled: true, grants: { all_granted: false, missing_keys: ['GOOGLE_OAUTH_CLIENT_ID'] } })));
    assert.doesNotMatch(fresh, /earlier version|skills-tray-note/);
});
