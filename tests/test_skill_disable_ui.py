"""Static UI contracts for the skill enable/disable in-flight feedback fix.

Bug: toggling a skill OFF while the (serialized) lifecycle lane is busy showed no
feedback and the toggle snapped back to ON, because mergeLifecycleEvents dropped
the in-flight event for an already-listed skill and the card always rendered the
stale persisted `enabled` value. See BUGREPORT-skill-disable-ui-stuck.md.
"""

from __future__ import annotations

import pathlib

REPO = pathlib.Path(__file__).resolve().parents[1]


def _read(rel: str) -> str:
    return (REPO / rel).read_text(encoding="utf-8")


def test_merge_annotates_existing_skill_instead_of_dropping_event():
    src = _read("web/modules/skills.js")
    assert "const out = skills.map((skill) => ({ ...skill }));" in src
    assert "const processedTargets = new Set();" in src
    assert "if (processedTargets.has(name)) continue;" in src
    assert "processedTargets.add(name);" in src
    assert "if (!LIFECYCLE_VISIBLE_STATUSES.has(event.status)) continue;" in src
    # The in-flight event for an already-listed skill must annotate that card...
    assert "existing.lifecycle_pending = event.status !== 'failed';" in src
    assert "existing.lifecycle_status = event.status;" in src
    # ...not be silently dropped by the old name-already-present guard.
    assert "if (!name || names.has(name)) continue;" not in src
    assert "if (!['queued', 'running', 'failed'].includes(event.status)) continue;" not in src


def test_newest_terminal_event_suppresses_older_pending_event():
    src = _read("web/modules/skills.js")
    assert "const latestByTarget = new Map();" in src
    assert "if (!latestByTarget.has(target)) latestByTarget.set(target, event);" in src
    assert "[...latestByTarget.values(), ...untargeted]" in src


def test_active_lifecycle_job_is_merged_into_skill_event_stream():
    src = _read("web/modules/skills.js")
    assert "function lifecycleEventsFromQueue(queueResp)" in src
    assert "const active = queueResp?.active;" in src
    assert "return [...deduped, active];" in src
    assert "lifecycleEventsFromQueue(snapshot.queue)" in src
    assert "mergeLifecycleEvents(snapshot.rawSkills," in src


def test_card_renders_pending_lifecycle_chip():
    src = _read("web/modules/skill_card_renderer.js")
    assert "lifecyclePendingLabel" in src
    assert "Turning off…" in src
    assert "Turning on…" in src
    assert "Deleting…" in src
    # The pending chip takes precedence over the stale persisted status.
    assert "if (skill.lifecycle_pending) {" in src


def test_card_renders_failed_lifecycle_feedback():
    src = _read("web/modules/skill_card_renderer.js")
    assert "function lifecycleFailedLabel" in src
    assert "skill.lifecycle_status === 'failed'" in src
    assert "chipHtml(lifecycleFailedLabel(skill.lifecycle_kind), 'danger')" in src
    assert "skill.lifecycle_error" in src


def test_pending_toggle_is_locked_and_reflects_intent():
    src = _read("web/modules/skill_card_renderer.js")
    assert "const lifecyclePending = Boolean(skill.lifecycle_pending);" in src
    # Toggle is disabled while a lifecycle job is in-flight so the re-render
    # cannot snap it back to the stale state, and it reflects the pending intent.
    assert "const toggleLocked = Boolean(lockReason) || lifecyclePending;" in src
    assert "${toggleLocked ? 'disabled' : ''}" in src
    assert "${toggleOn ? 'checked' : ''}" in src
