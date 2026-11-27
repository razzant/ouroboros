"""Prospective capacity inputs at the Settings and skill-visibility boundaries.

A fresh Main reads, for each installed skill, what discovery loads from its files, switch,
review and acceptance facts. A writer that changes those facts names the skill discovery
loads before and after, and ``_visible_transition`` is the one rule: where the catalogue or
the always-in-context block would read differently, the result is counted first.

Never held, and so outside the capacity guarantee: a disable, a removal, a mandatory
restore, a re-seed of bundled skills at startup, an acceptance that leaves the files not
acting, and a critic's verdict about files already acting (holding it would keep the earlier
verdict in force). Those of them that can lengthen the text still pass the publication fence
before they write (``catalogue_write_fence``), so no count in flight commits across them.
Writes that only remove text (uninstall, delete, a grant, a recorded dependency success, a
failed load's revert) are neither counted nor fenced; nor are payload edits by file tools.
"""
from contextlib import contextmanager, nullcontext
from contextvars import ContextVar
from dataclasses import replace
import json
import logging
from pathlib import Path
from types import SimpleNamespace

from ouroboros.nano_publication import (INSTRUCTION_ENABLE, INSTRUCTION_RELEASE, MemoryChange, NanoPublicationRefused,
                                        admission_for, publication_fence)

_LOG = logging.getLogger(__name__)
_RULES = "owner_rules"
_instruction = ContextVar("nano_instruction_publication", default=None)
_replacement = ContextVar("nano_instruction_replacement", default=None)
_rollback_view = ContextVar("nano_instruction_rollback_view", default=None)
_held_edit = ContextVar("nano_held_owner_edit", default=None)  # (root, name, digest) counted and held by an edit


def owner_rules_ticket(root: Path, after: str):
    """Count before the Settings document lock; the writer later commits this ticket."""
    from ouroboros.config import load_settings
    from ouroboros.safety_config import OWNER_RULES_KEY
    from ouroboros.settings_integrity import SettingsDocumentUnknown, skill_settings

    admission = admission_for(None, root)
    if admission is None:
        return None
    try:
        before = str(skill_settings(load_settings, strict=True).get(OWNER_RULES_KEY) or "")
    except SettingsDocumentUnknown as exc:
        raise NanoPublicationRefused("unavailable", "the current rules could not be read") from exc
    return admission.prepare(MemoryChange(None, before.encode("utf-8"), after.encode("utf-8"), subject=_RULES))


def skill_switch_publication(skill, root: Path, *, enabled: bool):
    """A first enable is one transition of visible text, counted before its ordinary switch write.

    Every skill adds its catalogue row; an always-in-context instruction also adds its body. The
    admitted enable holds the fence across the write and a refusal leaves the switch unchanged.
    """
    admission = admission_for(None, root) if enabled and not skill.enabled else None
    if admission is None:
        # A disable is not subject to the availability of the counter. It does
        # invalidate any concurrent proposal that read a different catalogue.
        return publication_fence(root)
    subject = json.dumps({"name": skill.name, "content_hash": skill.content_hash}).encode("utf-8")
    return _checked_or_ordinary(admission, MemoryChange(None, None, subject, subject=INSTRUCTION_ENABLE))


@contextmanager
def _checked_or_ordinary(admission, change):
    """The memory writers' discipline: an unqualified profile writes under the publication fence."""
    from ouroboros.skill_lifecycle_queue import lend_lifecycle_lane

    with admission.ordinary() as ordinary:
        if ordinary:
            yield
            return
    with lend_lifecycle_lane():
        ticket = admission.prepare(change)
    with ticket.commit():
        yield


def publish_review_auto_enable(root: Path, skill) -> bool:
    """Use the same admission at review's first-switch write, preserving its actor."""
    from ouroboros.skill_loader import find_skill, save_enabled, skill_state_dir_path

    with skill_switch_publication(skill, root, enabled=True):
        latest = find_skill(root, skill.name)
        if latest is None or _instruction_identity(latest) != _instruction_identity(skill):
            raise NanoPublicationRefused("stale", "the reviewed skill changed before its switch write")
        if (skill_state_dir_path(root, skill.name) / "enabled.json").exists():
            return False
        save_enabled(root, skill.name, True, actor="review_auto_enable")
    return True


def _instruction_identity(skill):
    return skill.skill_dir, skill.content_hash, skill.enabled, skill.review


def check_skill_switch(root, skill, expected_switch, *, repo_path):
    """Rebind Enable/Repair to the selected payload and saved choice inside the write locks."""
    from ouroboros.nano_publication import source_bytes
    from ouroboros.skill_loader import find_skill, skill_state_dir_path

    current = find_skill(root, skill.name, repo_path=repo_path)
    if (current is None or current.load_error or _instruction_identity(current) != _instruction_identity(skill)
            or source_bytes(skill_state_dir_path(root, skill.name) / 'enabled.json') != expected_switch):
        raise NanoPublicationRefused('stale', 'the skill or its saved switch changed; reopen the skill')


class _FutureInstructionReview:
    """Hypothesis for the count-only renderer, not a review result or an admission."""

    def __init__(self, review):
        self.review = review

    def __getattr__(self, name):
        return getattr(self.review, name)

    def gate_for(self, digest):
        return {**self.review.gate_for(digest), "executable_review": True}


_REFUSAL_WORDS = {
    "over_capacity": ("It does not fit the Nano window beside memory and work",
                      "shorten it, turn a skill off or fold memory, then try again"),
    "busy": ("The Nano size check could not run now", "try again shortly"),
    "unavailable": ("Its size could not be checked for Nano", "try again once its files and the token counter can be read"),
    "stale": ("Something changed while Nano checked its size", "look at it again, then retry"),
}


def refusal_words(refusal: NanoPublicationRefused) -> tuple[str, str]:
    """What happened and the next step, for a refused skill write; only over_capacity speaks of size."""
    return _REFUSAL_WORDS.get(refusal.reason, _REFUSAL_WORDS["unavailable"])


def prepare_skill_landing(root: Path, target: Path, staged: Path, name: str, record, *, keeps_snapshot=False):
    """Check a staged release before it changes the live tree; returns the commit context factory.

    The result is read as discovery will read it after the landing: the staged manifest, the
    new install record and the saved switch, wherever the occupant lived and whether it was
    readable, broken or absent. An On release still awaiting review is counted twice: as the
    not-ready line a fresh Main reads at once, and *as if* admitted, so a revision that could
    not act never displaces the working one. No review record follows from a dry count.
    """
    if admission_for(None, root) is None:
        return nullcontext
    from ouroboros.skill_loader import find_skill

    current = find_skill(root, name)

    def states():  # read only when a count needs them: an unqualified profile never builds them
        landed = _landed_skill(root, name, target, staged, record)
        if landed is None:
            return ()
        if landed.enabled and not _acts(landed):
            return landed, replace(landed, review=_FutureInstructionReview(landed.review))
        return (landed,)

    # An eligible old copy can return even if the replacement changes its manifest type.
    kept = (keeps_snapshot and record['basis'] == 'review' and current is not None
            and Path(current.skill_dir).resolve() == Path(target).resolve())
    return _visible_transition(root, name, current, states, keeps_previous=kept)


def _landed_skill(root: Path, name: str, target: Path, staged: Path, record):
    """What discovery will load once ``staged`` is at ``target`` under ``record``; None without a manifest."""
    from ouroboros.deployment_extensions import effective_enabled
    from ouroboros.skill_acceptance import admitted_acceptance, load_acceptance_facts
    from ouroboros.skill_loader import load_enabled, load_review_state
    from ouroboros.skill_source_catalog import read_manifest

    manifest = read_manifest(staged)
    if manifest is None:
        return None
    digest = record["release"]["digest"]  # verified by the staging owner before calling
    review = load_review_state(root, name, skill_type=manifest.type, is_module_widget=_module_widget(manifest),
                               skill_dir=staged)
    review.acceptance = admitted_acceptance(load_acceptance_facts(root, name, install_record=record), manifest)
    enabled = effective_enabled(name, load_enabled(root, name), Path(target).resolve(), digest)
    return _as_discovered(name, target, staged, manifest, digest, review, enabled)


def _module_widget(manifest) -> bool:
    return bool(manifest.is_extension() and isinstance(manifest.ui_tab, dict)
                and str((manifest.ui_tab.get("render") or {}).get("kind") or "") == "module")


def _as_discovered(name, target, source: Path, manifest, digest, review, enabled):
    from ouroboros.skill_loader import LoadedSkill

    target = Path(target).resolve()
    source = Path(source).resolve()
    manifest.name = manifest.name or target.name
    manifest_file = next(file for file in ("SKILL.md", "skill.json") if (source / file).is_file())
    return LoadedSkill(name, target, manifest, digest, enabled=enabled, review=review,
                       source=target.parent.name, location=target.parent.name,
                       manifest_file=manifest_file, payload_dir=source)


def catalogue_write_fence(root: Path):
    """Order an uncounted change of a skill's visible row without asking the counter or
    trusting any manifest. Take it outside a checked commit, which already holds the fence."""
    from ouroboros.config import DATA_DIR

    return publication_fence(root) if Path(root).resolve() == Path(DATA_DIR).resolve() else nullcontext()


def review_publication(root: Path, skill, state=None, *, facts=None, critic=False):
    """Hold the write that makes an On skill's current files act, or changes how they read: a
    critic's verdict, an owner attestation, an author acceptance (``state``) or an owner edit
    (``facts``).

    The critic's own ledger record stands either way. A refusal leaves the saved review and
    acceptance as they were. A write that leaves the files not acting is never held, and a
    ``critic``'s verdict is held only where it would newly activate them; both are only fenced.
    """
    from ouroboros.skill_acceptance import admitted_acceptance
    from ouroboros.skill_loader import find_skill

    if admission_for(None, root) is None:
        return nullcontext()
    if facts and _held_edit.get() == (Path(root).resolve(), skill.name, skill.content_hash):
        return nullcontext()  # the edit that wrote these bytes counted them and still holds the base: no second count
    current = find_skill(root, skill.name)
    if (current is None or current.load_error or current.skill_dir != skill.skill_dir
            or current.content_hash != skill.content_hash):
        return publication_fence(root)
    acceptance = current.review.acceptance
    if facts:
        acceptance = admitted_acceptance({**acceptance, **facts}, current.manifest)
    proposed = replace(current, review=replace(state or current.review, acceptance=acceptance))
    if not _acts(proposed) or (critic and _acts(current)):
        return publication_fence(root)
    return _visible_transition(root, current.name, current, (proposed,))()


def _acts(skill) -> bool:
    """On with admitted files: its whole card, and an always-in-context body, ride a fresh Main."""
    return bool(skill is not None and not skill.load_error and skill.enabled
                and skill.review.gate_for(skill.content_hash)["executable_review"])


def _skill_texts_equal(root: Path, installed, name: str, first, second) -> bool:
    from ouroboros.context_sections import _installed_skills_section
    from ouroboros.skill_catalogue import always_in_context_section

    try:
        _require_complete(installed)
        texts = []
        for skill in (first, second):
            rows = _with_skill(installed, name, skill)
            section = _installed_skills_section(SimpleNamespace(drive_root=root), rows, "")
            texts.append((section.full_text, section.operative_view, always_in_context_section(rows)))
    except Exception:
        _LOG.debug("Skill text of %s could not be compared; it is counted", name, exc_info=True)
        return False
    return texts[0] == texts[1]


def _reads_the_same(root: Path, name: str, current, proposed) -> bool:
    """Whether the real renderers give a fresh Main identical skill text before and after.

    Both catalogue forms count (the floor keeps the whole one where the shorter would not save),
    with the always-in-context block. A text that cannot be rendered is not known to be equal.
    """
    from ouroboros.skill_loader import discover_skills

    try:
        installed = discover_skills(Path(root))
    except Exception:
        _LOG.debug("Skill text of %s could not be compared; it is counted", name, exc_info=True)
        return False
    return _skill_texts_equal(root, installed, name, current, proposed)


def _installed_identity(root: Path, name: str, skill):
    from ouroboros.skill_loader import load_enabled

    # The saved switch is separate: a broken or absent package reads as Off while it stays saved On.
    return (None if skill is None else _instruction_identity(skill)), load_enabled(root, name)


def edit_admission(root: Path, skill, new_digest: str):
    """``(refusal, hold)`` for a predicted text edit of a payload file; ``(None, None)`` when it changes no text.

    The edit is the owner's own (it will be recorded as an owner edit), so what is counted is the skill
    acting from ``new_digest``, once, before the bytes change. A refusal leaves the earlier files, review and
    switch exactly as they were: no restore follows. ``hold()`` is that same transition's commit: the tool
    enters it just before its write and leaves it after the digest is recorded, so the base the count saw is
    the base the write lands on, or the edit is refused as stale before any byte changes. Nothing is counted a
    second time inside it (``review_publication`` skips exactly these bytes). It is a fence around the file
    write and one small record, never around a shell, an editor or another process.
    """
    from ouroboros.skill_acceptance import admitted_acceptance
    from ouroboros.skill_loader import find_skill

    if admission_for(None, root) is None:
        return None, None
    current = find_skill(root, skill.name)
    if (current is None or current.load_error or not current.enabled or current.skill_dir != skill.skill_dir
            or current.content_hash != skill.content_hash):
        return None, None
    acceptance = admitted_acceptance({**current.review.acceptance, "owner_edit_digest": new_digest}, current.manifest)
    proposed = replace(current, content_hash=new_digest, review=replace(current.review, acceptance=acceptance))
    if not _acts(proposed):
        return None, None
    try:
        commit = _visible_transition(root, current.name, current, (proposed,))
    except NanoPublicationRefused as refusal:
        return refusal, None

    @contextmanager
    def hold():
        with commit():
            token = _held_edit.set((Path(root).resolve(), current.name, new_digest))
            try:
                yield
            finally:
                _held_edit.reset(token)

    return None, hold


def _count_forms(root: Path, name: str, admission, identity, current, forms):
    """Prepare every form that reads differently on ONE base, and return the ticket that commits.

    The commit re-checks only the last ticket's base (marker generation and profile witness), so each
    ticket must start from the marker its predecessor left, and on its revision. A prepare reads the
    marker under the lock it holds, which the admission's own ``revision`` is called under; that read
    is how a move by another writer is told from the prepare's own rewrite, without a new lock. A move
    counts every form again, once; a second one is refused as stale, so no form is committed that was
    not counted on the base that commits. A ticket with no witness (an unqualified or never-confirmed
    counter) claims no protection and is not compared.
    """
    from ouroboros.nano_publication import PUBLICATION_REVISION, source_bytes
    from ouroboros.skill_lifecycle_queue import lend_lifecycle_lane

    marker_reads = []

    def revision():
        marker_reads.append(source_bytes(Path(root) / PUBLICATION_REVISION))
        return admission.revision()

    counting = replace(admission, revision=revision)
    forms = forms() if callable(forms) else forms
    for _attempt in range(2):
        last, ticket, moved = None, None, False
        for proposed in forms:
            if _reads_the_same(root, name, current, proposed):
                continue
            change = MemoryChange(None, None, json.dumps({"name": name, "content_hash": proposed.content_hash}).encode(),
                                  subject=INSTRUCTION_RELEASE)
            marker_reads.clear()
            token = _replacement.set((identity[0], proposed))
            try:
                with lend_lifecycle_lane():
                    ticket = counting.prepare(change)
            finally:
                _replacement.reset(token)
            if ticket.witness is not None:
                started = marker_reads[0] if marker_reads else None
                moved = moved or (last is not None and (started, ticket.witness[1]) != last)
                last = ticket.witness
        if not moved:
            return ticket
    raise NanoPublicationRefused("stale", "the base moved while the skill's forms were counted; try again")


def _visible_transition(root: Path, name: str, current, states, *, keeps_previous=False):
    """The commit context factory for one skill going from ``current`` (None: absent) through
    ``states`` (a tuple, or a function giving it), each as discovery would load it.

    Every state that reads differently is counted against the installed one, all on one base
    (``_count_forms``). The commit holds only while discovery still reads ``current`` and the same
    saved switch. A result that reads the same, and every write of an unqualified profile, orders
    itself behind the publication fence without counting or rendering anything.
    """
    from ouroboros.skill_loader import find_skill

    admission = admission_for(None, root)
    identity = _installed_identity(root, name, current)

    def check_current():
        if _installed_identity(root, name, find_skill(root, name)) != identity:
            raise NanoPublicationRefused("stale", "the installed skill or its switch changed during the check")

    @contextmanager
    def fenced():
        with publication_fence(root):
            check_current()
            yield

    with admission.ordinary() as ordinary:
        pass
    if ordinary:
        check_current()

        @contextmanager
        def ordinary_commit():
            with admission.ordinary() as still_ordinary:
                if not still_ordinary:
                    raise NanoPublicationRefused('stale', 'the active profile changed; check the proposal again')
                check_current()
                yield

        return ordinary_commit
    ticket = _count_forms(root, name, admission, identity, current, states)
    if ticket is None:
        return fenced

    if keeps_previous and ticket.witness is not None and _pending_rollbacks(Path(root).resolve()):
        # The ticket invalidates an intervening landing. Bound visible rollback
        # alternatives, not unrelated skill operations or the independent review.
        raise NanoPublicationRefused("busy", "another skill update still has a visible rollback; retry after it settles")

    check_current()  # before the installer calls unload or snapshots the occupant

    @contextmanager
    def commit():
        with ticket.commit():
            check_current()
            yield

    return commit


@contextmanager
def prospective_instruction(root, changes):
    matches = [change for change in changes if change.subject in {INSTRUCTION_ENABLE, INSTRUCTION_RELEASE}]
    if len(matches) != 1:
        raise NanoPublicationRefused("unavailable", "expected exactly one prospective instruction")
    change = matches[0]
    replacement = _replacement.get() if change.subject == INSTRUCTION_RELEASE else None
    if change.subject == INSTRUCTION_RELEASE and replacement is None:
        raise NanoPublicationRefused("unavailable", "the staged instruction cannot be read")
    token = _instruction.set((Path(root).resolve(), json.loads(change.after), replacement))
    try:
        yield
    finally:
        _instruction.reset(token)


def _pending_rollbacks(root: Path) -> list:
    """What the kept rollbacks of skills would put back where they change visible text.

    Restoring one is never counted, so every later write leaves room for its visible row or card
    and, for an enabled always-in-context instruction, its body. The combined rollback is not a
    witness for any mixed outcome: even ordinary catalogue rows can change in opposite directions.
    """
    from ouroboros.deployment_extensions import change_refusal
    from ouroboros.skill_install_record import read_install_record, read_update_snapshot
    from ouroboros.skill_loader import discover_skills

    proposed = _instruction.get()
    enabling = proposed[1]['name'] if proposed and proposed[0] == root and proposed[2] is None else None
    installed = shown = None
    walked = False
    rows = []
    # These are the handles the startup settler can still act on; .previous
    # and past review records are not active rollback obligations.
    for path in sorted((root / 'state' / 'skills').glob('*/update_backup.json')):
        name = path.parent.name
        if change_refusal(name, 'rollback'):
            continue
        snapshot, data = read_update_snapshot(root, name)
        if snapshot is None:
            if not data and path.exists():
                raise NanoPublicationRefused('unavailable', f'the pending rollback of {name} could not be read')
            continue  # a rejected path cannot be restored by the ordinary settler
        previous = json.loads(snapshot.lifecycle_state.get('install.json') or b'{}')
        current = read_install_record(root, name) or {}
        pending = data.get('pending_update')
        if not isinstance(pending, dict) or not pending.get('digest'):
            raise NanoPublicationRefused('unavailable', f'the pending rollback of {name} has no release binding')
        if not data.get('restore_started') and current != previous and (
                current.get('pending_update') != pending
                or (current.get('release') or {}).get('digest') != pending['digest']):
            continue  # the same supersession check as settlement
        candidate = _restored_skill(root, snapshot, previous, data, enabled=True if name == enabling else None)
        if candidate is None:
            continue
        if not walked:
            walked = True
            try:
                installed = discover_skills(Path(root))
                shown = _preview_proposal(root, installed)
            except NanoPublicationRefused:
                raise
            except Exception:
                installed = shown = None
        if installed is not None:
            active = next((skill for skill in installed if skill.name == name), None)
            after = next((skill for skill in shown if skill.name == name), None)
            if (_skill_texts_equal(root, installed, name, active, candidate)
                    and (shown is installed or _skill_texts_equal(root, shown, name, after, candidate))):
                continue
        rows.append(candidate)
    if len(rows) > 1:
        raise NanoPublicationRefused('busy', 'overlapping skill rollbacks must settle before this write')
    return rows


def _restored_skill(root, snapshot, previous, data, *, enabled=None):
    from ouroboros.deployment_extensions import effective_enabled
    from ouroboros.skill_acceptance import admitted_acceptance, load_acceptance_facts
    from ouroboros.skill_loader import SkillPayloadUnreadable, load_enabled, review_record_projection
    from ouroboros.skill_source_catalog import read_manifest, tree_release_digest

    source = snapshot.target_dir if snapshot.payload_restored else snapshot.backup_dir
    if source is None:
        return None  # the snapshot had no preceding payload
    if not source.is_dir() and data.get('restore_started'):
        # A live restore may have moved the saved tree just now. The publication
        # fence invalidates this capture; after a crash the target is the only
        # possible survivor of the move, so inspect it rather than erase the debt.
        source = snapshot.target_dir
    manifest = read_manifest(source)
    if manifest is None:
        raise NanoPublicationRefused('unavailable', f'the pending rollback payload of {snapshot.skill_name} could not be read')
    try:
        digest = tree_release_digest(source, manifest, bucket=snapshot.target_dir.parent.name)
    except SkillPayloadUnreadable as exc:
        raise NanoPublicationRefused('unavailable', f'the pending rollback payload of {snapshot.skill_name} is incomplete') from exc
    review = review_record_projection(root, snapshot.skill_name,
                                      json.loads(snapshot.lifecycle_state.get('review.json') or b'null'),
                                      skill_type=manifest.type, is_module_widget=_module_widget(manifest), skill_dir=source)
    review.acceptance = admitted_acceptance(load_acceptance_facts(root, snapshot.skill_name, install_record=previous), manifest)
    saved = load_enabled(root, snapshot.skill_name) if enabled is None else enabled
    active = effective_enabled(snapshot.skill_name, saved, snapshot.target_dir.resolve(), digest)
    # A restored On row still contributes its purpose and refusal reason if its review
    # does not admit execution; a restored Off row still contributes its catalogue purpose.
    return _as_discovered(snapshot.skill_name, snapshot.target_dir, source, manifest, digest, review, active)


def pending_instruction_rollback_view(root: Path):
    """One count-only context factory for the reachable rollbacks, if any."""
    root = Path(root).resolve()
    rows = _pending_rollbacks(root)
    if not rows:
        return None

    @contextmanager
    def view():
        token = _rollback_view.set((root, rows))
        try:
            yield
        finally:
            _rollback_view.reset(token)

    return view


def preview_instruction(root, skills):
    """Use the normal renderer for the proposed skill state without changing installed state.

    Without an active count (every ordinary read) the discovery result is returned unchanged.
    """
    shown = _preview_proposal(root, skills)
    alternate = _rollback_view.get()
    if alternate is not None and alternate[0] == Path(root).resolve():
        _require_complete(shown)
        for old in alternate[1]:
            if any(skill.identity_collision or skill.skill_dir != old.skill_dir for skill in shown if skill.name == old.name):
                raise NanoPublicationRefused("stale", "the pending rollback no longer names the installed skill")
            shown = _with_skill(shown, old.name, old)
    if shown is not skills:
        _require_renderable(root, shown)
    return shown


def _require_renderable(root, skills):
    """The context builder states a catalogue it cannot render in one short line; counting that line
    would pass a text the next Main may not get."""
    from ouroboros.context_sections import _installed_skills_section

    try:
        _installed_skills_section(SimpleNamespace(drive_root=root), skills, "")
    except Exception as exc:
        raise NanoPublicationRefused("unavailable", f"the skill catalogue could not be rendered ({type(exc).__name__})") from exc


def _require_complete(skills):
    """A count over a partial walk would leave out text it could not read: that is unknown, not small."""
    if getattr(skills, "error", None) is not None or getattr(skills, "gaps", None):
        raise NanoPublicationRefused("unavailable", "the installed skills could not be read completely")


def _with_skill(skills, name, proposed):
    """The discovery result with ``proposed`` (None: absent) in ``name``'s place, in discovery order."""
    from ouroboros.skill_discovery_reuse import WalkedSkills

    rows = [*(skill for skill in skills if skill.name != name), *(() if proposed is None else (proposed,))]
    return WalkedSkills(sorted(rows, key=lambda skill: (skill.name, str(skill.skill_dir))), getattr(skills, "gaps", ()))


def _preview_proposal(root, skills):
    held = _instruction.get()
    if held is None or Path(root).resolve() != held[0]:
        return skills
    _require_complete(skills)
    subject = held[1]
    matched = [skill for skill in skills if skill.name == subject["name"]]
    if held[2] is None:  # a first enable of the installed files
        if matched and matched[0].content_hash == subject["content_hash"]:
            return _with_skill(skills, subject["name"], replace(matched[0], enabled=True))
    else:
        identity, proposed = held[2]
        if (_instruction_identity(matched[0]) if matched else None) == identity:
            return _with_skill(skills, subject["name"], proposed)
    raise NanoPublicationRefused("stale", "the installed skill changed before counting")
