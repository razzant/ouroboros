"""Extensions a deployment requires or trusts: the one reader of the policy's ``extensions`` section.

A required extension is named by its canonical name, package directory and the digest the loader computes for the
bytes it runs (``deployment_policy.RequiredExtension``). Exactly that package reads as on whatever the employee's saved
switch says (``effective_enabled``); the saved choice stays in ``enabled.json`` and applies again once the file stops
naming it. Being required grants nothing else: review, grants, dependencies and conflicts still decide whether it
runs; a package at another path, of other bytes or under a colliding name is not the required one; a missing package
is a gap to show, never a fault of the file.

Trust. With an execution domain the service runs the code of trusted packages only (``trust_gap``), named the same
way (``deployment_policy.TrustedExtension``): the liveness authority reads any other extension as not to run
(``deployment_untrusted``), so nothing imports, catalogs, registers or launches its code, and the staged copy it
imports from must hash to the pinned digest (``staged_verifier``). Trust admits code and nothing more: the switch,
review, grants and the rest decide as for any extension. Without ``execution`` nothing here reads trust.

Readers: the loader (``skill_loader.load_skill``), the liveness refresh, the peer inventory, Host Service and the
health alarm read enablement here. Writers: every supported door that would turn the required package off or
remove, replace, shadow, update, roll back, repair or edit it refuses before it unloads, writes or starts a task
(``change_refusal``, ``install_refusal``, ``payload_write_refusal``); a missing name may still be installed. A trusted
package's files are its administrator's the same way, while its switch stays the employee's; with an execution
domain the staged imports are the service's alone, and the dependency owner installs nothing into a trusted
package (``dependency_refusal``). Ordinary unload, shutdown, Stop and Panic unload them as before. Its
administrator changes the package and the file, then starts a new generation. A shell, or any process able to
write the package, its dependencies, the state or policy files, is not stopped by this: that is the deployment's
operating-system boundary.

Running state. ``state_rows`` is the observing process's own view for Settings and Skills, named with its process:
it never vouches for another worker. ``hold_reason`` is the check every carrier makes before ordinary work starts,
against the extensions live in that very process (``worker_process._run_task_body`` ends a held task itself, a
Presence turn meets it in ``Agent.handle_task``): under ``hold`` the work does not start (``HOLD_REASON``), under
``warn`` it starts; either way a change of the gap is logged once per process. The queue
keeps waiting work queued while the server's own view holds (``queue_held``), on the queue's existing pass. Nothing
here polls, retries, installs, reviews or grants; what an extension delivers elsewhere is that extension's status.
A package failure with fixed settings is recorded by its exception's type before status, dispatch or logs read it
(``failure_words``): even a parser's error may quote part of a fixed value. Successful content is not rewritten.
"""

from __future__ import annotations

import logging
import os
import pathlib
import threading
from typing import Any, Callable, Iterable, Optional

from ouroboros.deployment_policy import required_extensions, trusted_extensions, when_unmet

log = logging.getLogger(__name__)

HOLD_REASON = "deployment_requirement_unmet"
UNTRUSTED_REASON = "deployment_untrusted"  # the liveness reason of extension code the service does not run
# Why an extension's code does not run on a deployment's service, by ``trust_gap`` code: the owner's own sentence.
TRUST_SENTENCES = {
    "not_delivered": "Only extensions your organization’s administrator delivers run here, and this one isn’t among them.",
    "elsewhere": "Your organization’s administrator delivers it from another folder, so this copy doesn’t run.",
    "changed": "Its files differ from the version your organization’s administrator delivered, so it doesn’t run.",
}
_REVIEW_REASONS = frozenset({"review_pending", "review_stale", "review_missing_or_unknown", "blocker_findings_need_owner"})
_VERBS = {"disable": "turned off", "delete": "removed", "uninstall": "removed", "install": "replaced or shadowed",
          "update": "updated", "rollback": "rolled back", "repair": "handed to a repair task", "edit": "edited"}


def _canonical(name: Any) -> str:
    from ouroboros.contracts.skill_manifest import canonical_skill_name

    return canonical_skill_name(str(name or ""))


def _real(path: Any) -> str:
    return os.path.normcase(os.path.realpath(str(path)))


def required(name: Any):
    """The ``RequiredExtension`` named ``name`` (canonicalized), else ``None``."""
    canonical = _canonical(name)
    return next((row for row in required_extensions() if row.name == canonical), None)


def trusted(name: Any):
    """The ``TrustedExtension`` named ``name`` (canonicalized) under an execution domain, else ``None``."""
    canonical = _canonical(name)
    return next((row for row in trusted_extensions() or () if row.name == canonical), None)


def _administered() -> list:
    """Every package row whose files are its administrator's: the required ones, and with an execution domain the
    trusted ones (a required package is one of those there)."""
    return [*required_extensions(), *(trusted_extensions() or ())]


def trust_gap(name: Any, skill_dir: Any, content_hash: Any) -> str:
    """Why the service may not run this package's code (a ``TRUST_SENTENCES`` code), "" when it may: always without
    an execution domain, else exactly a trusted package, its name, directory and the bytes the loader hashed."""
    rows = trusted_extensions()
    if rows is None:
        return ""
    row = next((item for item in rows if item.name == _canonical(name)), None)
    if row is None:
        return "not_delivered"
    if _real(skill_dir) != _real(row.path):
        return "elsewhere"
    return "" if content_hash and content_hash == row.content_hash else "changed"


def trust_state(skill: Any) -> Optional[dict]:
    """How an extension's code stands under an execution domain, for Skills: ``{"state": "trusted"}`` or
    ``{"state": "untrusted", "reason", "detail"}``, ``administered`` naming whether its doors refuse every change but
    turning it off (its name is the administrator's); ``None`` without a domain or for a skill that is no extension."""
    if trusted_extensions() is None or not skill.manifest.is_extension():
        return None
    gap = trust_gap(skill.name, skill.skill_dir, skill.content_hash)
    if not gap:
        return {"state": "trusted", "administered": True}
    return {"state": "untrusted", "reason": gap, "detail": TRUST_SENTENCES[gap], "administered": gap != "not_delivered"}


def staged_verifier(skill: Any) -> Optional[Callable[[pathlib.Path], bool]]:
    """Under an execution domain, what a staged import tree must pass before the service imports from it: its files
    hash, as the loader hashes the package (read as the package's own bucket), to the trusted digest; ``None`` else."""
    if trusted_extensions() is None:
        return None
    from ouroboros.skill_identity import payload_bucket, payload_digest

    row, bucket = trusted(skill.name), payload_bucket(skill.skill_dir)

    def verify(tree: pathlib.Path) -> bool:
        try:
            return row is not None and payload_digest(tree, manifest_entry=skill.manifest.entry, bucket=bucket,
                                                      manifest_scripts=skill.manifest.scripts) == row.content_hash
        except Exception:  # an unreadable copy is not the pinned bytes
            log.warning("Staged import of %s could not be hashed", skill.name, exc_info=True)
            return False

    return verify


WITHHELD = "its words are withheld: they may quote a setting your organization fixes"


def discloses_failure(name: Any) -> bool:
    """Whether a failure of ``name`` may be told in its exception's own words and logged
    with its traceback: always, except for a package whose settings this deployment fixes (``failure_words``)."""
    row = required(name)
    return row is None or not row.settings


def failure_words(name: Any, exc: BaseException, *, typed: bool = True) -> str:
    """The words a load or runtime failure is recorded with at its producer, before HTTP/WS, tools, health or logs
    read it: the exception's own (after its type when ``typed``), or only its type for a package whose
    settings this deployment fixes, since its words may quote a fixed value or part of one (a parser's echo of a
    malformed DSN), which no matching of whole values could find."""
    if not discloses_failure(name):
        return f"{type(exc).__name__} ({WITHHELD})"
    return f"{type(exc).__name__}: {exc}" if typed else str(exc)


def dependency_refusal(loaded: Any) -> str:
    """Why the dependency owner installs, updates or rebinds nothing for ``loaded``: under an execution domain a
    trusted package's packages are part of what its administrator delivers; "" for every other skill."""
    if trusted(loaded.name) is None or trust_gap(loaded.name, loaded.skill_dir, loaded.content_hash):
        return ""
    return ("Its packages come with it from your organization’s administrator; Ouroboros installs, updates or "
            "rebinds none of them here.")


def at_required_path(name: Any, skill_dir: Any) -> bool:
    row = required(name)
    return row is not None and _real(skill_dir) == _real(row.path)


def matches(name: Any, skill_dir: Any, content_hash: Any) -> bool:
    """Whether this package is the required one: its name, its directory and the bytes the loader hashed."""
    row = required(name)
    return row is not None and bool(content_hash) and content_hash == row.content_hash and _real(skill_dir) == _real(row.path)


def effective_enabled(name: Any, saved: bool, skill_dir: Any, content_hash: Any) -> bool:
    """The enablement every reader uses: the employee's saved switch, or on for exactly the required package."""
    return bool(saved) or matches(name, skill_dir, content_hash)


def peer_enabled(drive_root: Any, candidate: Any, saved: bool) -> bool:
    """A peer's enablement without hashing every peer: only a required name at its required path is loaded (the
    loader's reused read) to compare its bytes; every other peer is its saved switch."""
    if saved or not at_required_path(candidate.name, candidate.skill_dir):
        return bool(saved)
    from ouroboros.skill_loader import _load_skill_location_candidates

    loaded = _load_skill_location_candidates((candidate,), drive_root=pathlib.Path(drive_root))
    return bool(loaded and loaded[0].enabled)


# --- writers: refuse before any effect -------------------------------------------------------------------

def change_refusal(name: Any, operation: str) -> str:
    """Why a supported door may not ``operation`` (``_VERBS``) the skill ``name``; "" when it may. A trusted
    package's switch stays the employee's (``disable``); everything else about its files is its administrator's."""
    row = required(name)
    if row is not None:
        return (f"{row.name} is required by this deployment's policy file, so it cannot be "
                f"{_VERBS.get(operation, 'changed')} here; nothing was changed. Its administrator changes it there.")
    row = trusted(name)
    if row is None or operation == "disable":
        return ""
    return (f"{row.name} is delivered by this deployment's administrator, so it cannot be "
            f"{_VERBS.get(operation, 'changed')} here; nothing was changed. You can turn it off; its administrator "
            "changes or removes it.")


def install_refusal(drive_root: Any, name: Any, *, repo_path: Optional[str] = None) -> str:
    """An install may add a required or trusted name that no package has yet; over an installed one it would replace
    the administrator's package or shadow it with a second package of that name."""
    if required(name) is None and trusted(name) is None:
        return ""
    from ouroboros.skill_loader import _skill_location_inventory

    canonical = _canonical(name)
    present = any(item.name == canonical for item in _skill_location_inventory(pathlib.Path(drive_root), repo_path=repo_path))
    return change_refusal(canonical, "install") if present else ""


STAGED_REFUSAL = ("The staged copies the service imports extension code from are its own: nothing here writes them, "
                  "and nothing was changed.")


def _below(real: str, base: str) -> tuple:
    """The parts of ``real`` below ``base`` (both real paths), else ``()``."""
    base = base.rstrip(os.sep)
    return pathlib.PurePath(real[len(base) + 1:]).parts if real.startswith(base + os.sep) else ()


def payload_write_refusal(target: Any, *, data_root: Any = None) -> str:
    """Why a file edit of ``target`` would change an administered package (required, or trusted under an execution
    domain), start a second package of its name under the data skills folder (``<skills>/<name>`` or
    ``<skills>/<bucket>/<name>``) or, under an execution domain, write a staged import tree
    (``state/skills/<name>/__extension_imports``); "" otherwise."""
    rows = _administered()
    if not (rows or trusted_extensions() is not None) or not target:
        return ""
    real = _real(target)
    for row in rows:
        base = _real(row.path)
        if real == base or _below(real, base):
            return change_refusal(row.name, "edit")
    if data_root is None:
        from ouroboros import config

        data_root = config.DATA_DIR
    if trusted_extensions() is not None and _below(real, _real(pathlib.Path(data_root) / "state" / "skills"))[1:2] == (
            "__extension_imports",):
        return STAGED_REFUSAL
    skills = _real(pathlib.Path(data_root) / "skills")
    parts = _below(real, skills)
    for depth in (1, 2):  # the loader's walk: a package directly under the folder or one bucket below it
        if len(parts) >= depth and (row := required(parts[depth - 1]) or trusted(parts[depth - 1])) is not None:
            if _real(os.path.join(skills, *parts[:depth])) != _real(row.path):
                return change_refusal(row.name, "install")
    return ""


def subtree_refusal(target: Any, *, data_root: Any = None) -> str:
    """``payload_write_refusal`` for a path, or a directory holding an administered package (a delete or a move)."""
    if refusal := payload_write_refusal(target, data_root=data_root):
        return refusal
    real = _real(target).rstrip(os.sep) + os.sep
    return next((change_refusal(row.name, "delete") for row in _administered() if _real(row.path).startswith(real)), "")


# --- running state ---------------------------------------------------------------------------------------

_PROCESS: dict = {}


def _process_fact() -> dict:
    """Which process observed a state: role, pid and its birth token (read once), never another process's."""
    from ouroboros.extension_companion import is_server_process
    from ouroboros.utils import utc_now_iso

    if not _PROCESS or _PROCESS.get("pid") != os.getpid():
        try:
            from ouroboros.platform_layer import process_start_time

            birth = process_start_time(os.getpid())
        except Exception:
            birth = ""
        _PROCESS.clear()
        _PROCESS.update(pid=os.getpid(), started=str(birth or ""))
    return {"role": "server" if is_server_process() else "worker", **_PROCESS, "observed_at": utc_now_iso()}


def _secrets() -> list:
    return [value for row in required_extensions() for _key, value in row.settings]


def _redacted(text: Any) -> str:
    from ouroboros.secret_masking import redact_known_values

    return str(redact_known_values(str(text or ""), _secrets()))


def _gap(code: str, detail: str, **facts: Any) -> dict:
    return {"state": "unmet", "reason": code, "detail": _redacted(detail), **facts}


def _liveness_gap(state: dict) -> dict:
    """The liveness authority's verdict for the pinned package, as a requirement row's state."""
    reason = str(state.get("reason") or "")
    if state.get("desired_live") and state.get("live_loaded"):
        return {"state": "met", "reason": "running", "detail": "Running in this process."}
    if reason in _REVIEW_REASONS:
        return _gap("review", "It needs a review or your attestation of these exact bytes.")
    if reason == "missing_grants":
        grants = state.get("grant_status") or {}
        missing = [*(grants.get("missing_keys") or []), *(grants.get("missing_permissions") or [])]
        return _gap("grants", "It needs your grant" + (f" for {', '.join(missing)}." if missing else "."))
    if reason == "skill_conflict":
        names = ", ".join((state.get("conflict") or {}).get("skills") or [])
        return _gap("conflict", f"A skill that conflicts with it is on{': ' + names if names else ''}.")
    if reason.startswith("deps_"):
        from ouroboros.skill_dependencies import dependency_sentence

        return _gap("dependencies", dependency_sentence(reason))
    if reason == "load_error":
        return _gap("load_error", f"It could not start: {state.get('load_error') or 'unknown error'}")
    if reason in ("ready", "", "already_live"):
        return _gap("not_loaded", "It is not running in this process yet.")
    return _gap(reason, str(state.get("load_error") or "It is not running in this process."))


def _undeclared(row: Any, manifest: Any) -> list:
    """Keys the file fixes for ``row`` that the package never reads (its ``env_from_settings``): such a value can
    never reach it, so the requirement is unmet for Settings, the carrier and the queue alike."""
    declared = {str(key).upper() for key in (getattr(manifest, "env_from_settings", None) or [])}
    return [key for key, _value in row.settings if key not in declared]


def requirement_state(drive_root: Any, row: Any, *, skills: Optional[list] = None) -> dict:
    """One requirement as this process observes it: met, or the first gap in the order a fix needs."""
    from ouroboros.extension_liveness import runtime_state_for_loaded_skill
    from ouroboros.skill_loader import discover_skill_identity

    root = pathlib.Path(drive_root)
    found = [item for item in (skills if skills is not None else discover_skill_identity(root, row.name))
             if item.name == row.name]
    base = {"name": row.name, "path": row.path, "content_hash": row.content_hash,
            "settings": [key for key, _value in row.settings], "process": _process_fact()}
    if not found:
        return {**base, **_gap("missing", f"No package named {row.name} is installed.")}
    if len(found) > 1 or any(getattr(item, "identity_collision", False) for item in found):
        return {**base, **_gap("collision", f"More than one package is named {row.name}, so none of them runs.")}
    skill = found[0]
    if _real(skill.skill_dir) != _real(row.path):
        return {**base, **_gap("elsewhere", f"It is installed at {skill.skill_dir}, not at the required path.")}
    if skill.load_error and not skill.content_hash:
        return {**base, **_gap("load_error", f"It could not be read: {skill.load_error}")}
    if not skill.manifest.is_extension():
        return {**base, **_gap("not_extension", "The package at that path is not an extension.")}
    if skill.content_hash != row.content_hash:
        return {**base, "observed_hash": skill.content_hash,
                **_gap("changed", "Its files differ from the version your organization pinned.")}
    if undeclared := _undeclared(row, skill.manifest):
        return {**base, **_gap("settings_undeclared", f"It does not read {', '.join(undeclared)}, which your "
                                                      "organization sets for it (env_from_settings).")}
    return {**base, **_liveness_gap(runtime_state_for_loaded_skill(skill, root, skills=skills))}


def state_rows(drive_root: Any, *, skills: Optional[list] = None) -> list:
    """Every requirement as this process observes it, in the file's order; [] when none is required."""
    rows = []
    for row in required_extensions():
        try:
            rows.append(requirement_state(drive_root, row, skills=skills))
        except Exception as exc:  # one unreadable package never hides the others' rows
            log.warning("Requirement %s could not be observed", row.name, exc_info=True)
            rows.append({"name": row.name, "path": row.path, "content_hash": row.content_hash,
                         "settings": [key for key, _v in row.settings], "process": _process_fact(),
                         **_gap("unknown", f"It could not be observed: {type(exc).__name__}")})
    return rows


def _live_gaps(drive_root: Any, *, refresh: bool) -> list:
    """Required extensions not live in this process as pinned. ``refresh`` also rereads the small authority
    controls (switch, review, grants, health) of a live one, as published widgets do, without rehashing."""
    from ouroboros.extension_registry_state import _extensions, _lock

    with _lock:
        live = {name: (bundle.content_hash, bundle.skill_dir, bundle.loaded_skill) for name, bundle in _extensions.items()}
        peers = [bundle.loaded_skill for bundle in _extensions.values() if bundle.loaded_skill is not None]
    gaps = []
    for row in required_extensions():
        content_hash, skill_dir, skill = live.get(row.name, ("", "", None))
        if content_hash != row.content_hash or _real(skill_dir or "/nonexistent") != _real(row.path):
            gaps.append(row)
        elif skill is not None and _undeclared(row, skill.manifest):
            gaps.append(row)
        elif refresh and skill is not None:
            from ouroboros.extension_liveness import _published_skill_readiness

            state = _published_skill_readiness(skill, pathlib.Path(drive_root), peers)
            if not (state.get("desired_live") and state.get("live_loaded")):
                gaps.append(row)
    return gaps


def queue_held() -> bool:
    """Whether the queue keeps ordinary work waiting: ``hold`` and a required extension not live in this (the
    server) process. An in-memory read for the assignment pass; each carrier still checks itself at the start."""
    return when_unmet() == "hold" and bool(_live_gaps(None, refresh=False))


_REPORTED: dict = {}
_REPORT_LOCK = threading.Lock()


def _report(drive_root: Any, mode: str, rows: Iterable[dict]) -> None:
    """Log and record a change of this process's gap once (``warn`` and ``hold`` alike), never every task."""
    from ouroboros.deployment_policy import current

    rows = list(rows)
    policy = current()
    signature = (os.getpid(), policy.digest if policy is not None else "", mode,
                 tuple((row["name"], row["reason"]) for row in rows))
    with _REPORT_LOCK:
        if _REPORTED.get("signature") == signature:
            return
        _REPORTED["signature"] = signature
    from ouroboros.utils import append_jsonl, utc_now_iso

    if rows:
        log.warning("Deployment requirement not running in this process (%s): %s", mode,
                    "; ".join(f"{row['name']}: {row['detail']}" for row in rows))
    try:
        append_jsonl(pathlib.Path(drive_root) / "logs" / "events.jsonl", {
            "ts": utc_now_iso(), "type": HOLD_REASON if rows else "deployment_requirement_met", "when_unmet": mode,
            "process": _process_fact(), "requirements": [{key: row[key] for key in ("name", "reason", "detail")}
                                                         for row in rows]})
    except Exception:
        log.debug("Deployment requirement event was not recorded", exc_info=True)


def _data_root(drive_root: Any) -> pathlib.Path:
    if drive_root is not None:
        return pathlib.Path(drive_root)
    from ouroboros import config

    return pathlib.Path(config.DATA_DIR)


def unmet_here(drive_root: Any = None) -> list:
    """The requirement rows this process does not run as pinned (each with its gap); [] when all run."""
    root = _data_root(drive_root)
    gaps = _live_gaps(root, refresh=True)
    if not gaps:
        return []
    rows = []
    for row in gaps:
        state = requirement_state(root, row)
        rows.append(state if state["state"] == "unmet" else {**state, **_gap(
            "not_loaded", "It is not running in this process yet.")})
    return rows


def hold_reason(drive_root: Any = None) -> str:
    """``HOLD_REASON`` when this process must not start ordinary work, else "": the carrier's check before a task
    body. Under ``warn`` the work starts and the gap is reported; unmanaged or without requirements this is free."""
    mode = when_unmet()
    if not mode:
        return ""
    root = _data_root(drive_root)
    rows = unmet_here(root)
    _report(root, mode, rows)
    return HOLD_REASON if rows and mode == "hold" else ""


def hold_text(drive_root: Any = None) -> str:
    """The owner's sentence for work this process did not start (no secret, no value)."""
    rows = unmet_here(drive_root)
    one = len(rows) == 1
    gaps = " ".join(f"{row['name']}: {row['detail']}" for row in rows) or "A required extension is not running."
    return (f"🔒 Not started: your organization requires {'an extension that isn’t' if one else 'extensions that aren’t'} "
            f"running in this process. {gaps} Nothing ran; start it again once {'it runs' if one else 'they run'}. "
            "Stop, Panic, Settings and Skills stay available.")


def record_held_task(drive_root: Any, task: dict, text: str, **fields: Any) -> None:
    """The terminal of a task this carrier held before its body: failed, typed, nothing spent. ``fields`` are
    the caller's correlation (a conversation turn's ``origin_message_ref``), as its Stop terminal records them."""
    from ouroboros.outcomes import infra_failed_axes
    from ouroboros.subagents import effort_result_fields
    from ouroboros.task_results import _TRULY_TERMINAL_STATUSES, STATUS_FAILED, write_task_result

    write_task_result(
        drive_root, str(task.get("id") or ""), STATUS_FAILED,
        _field_projector=lambda current, incoming: None if current.get("status") in _TRULY_TERMINAL_STATUSES else incoming,
        _terminal_observed=True, result=text, reason_code=HOLD_REASON,
        outcome_axes=infra_failed_axes(HOLD_REASON, lifecycle="failed", review_trigger="before_task_start"),
        **fields, **effort_result_fields(task))
