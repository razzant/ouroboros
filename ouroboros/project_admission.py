"""Project execution authority: committed reads, carried basis and final admission.

The registry remains the only durable authority. The registry facade re-exports
these operations for preparation consumers; queue admission holds Q through append.
Routing comparison covers identity and routing fields only, so activity or lock
occupancy never invalidates admission; each pass parses registry and bindings once.
"""
from __future__ import annotations

import json
from contextlib import contextmanager
from typing import Any, Optional

from ouroboros.contracts.chat_id_policy import project_chat_id
from ouroboros.project_facts import sanitize_project_id


class ProjectAdmissionError(RuntimeError):
    """A semantic refusal, distinct from an unreadable routing authority."""

    def __init__(self, reason: str, detail: str, lifecycle: str = ""):
        super().__init__(detail)
        self.reason, self.lifecycle = reason, lifecycle


def host_unscoped(task: dict) -> bool:
    """Fresh host admission proved no Project scope; restore never backfills it."""
    return (task.get("_project_scope_none") is True
            and not task.get("project_id") and "_project_admission" not in task)


def hold_unreadable_result(task: dict) -> bool:
    """Whether accepted Project work whose own result is unreadable waits held, never failed.

    Restore and live assignment share this rule: the same row keeps its id,
    payload and resources; hold release rechecks its original receipt, scope and
    no-dispatch or exact continuation evidence (exact budget, owner wait or
    resolver continuation) and canonical bindings; Stop/terminal results stay
    independent. An invalid basis or unreadable authority keeps the hold at any
    snapshot age, and bases are never recaptured. Exact budget pauses keep their
    own hold authority.
    """
    pause = task.get("_budget_pause")
    if (not task.get("_project_admission_restore_hold")
            and not (isinstance(pause, dict) and pause.get("exact_continuation") is True)
            and (task.get("project_id") or "_project_admission" in task)):
        task["_project_admission_restore_hold"] = {"reason": "project_routing_fence_lookup_failed",
                                                   "detail": "The task result is unreadable; the accepted task waits for it."}
    return bool(task.get("_project_admission_restore_hold"))


def project_hold_fact(task: dict) -> dict:
    """Read-only waiting fact; no new task phase or owner-action claim."""
    hold = task.get("_project_admission_restore_hold")
    label = "Waiting for task scope verification" if host_unscoped(task) else "Waiting for Project verification"
    if isinstance(hold, dict) and hold.get("reason") == "project_dispatch_unconfirmed":
        label = "Waiting: previous run unconfirmed"
    return ({**hold, "label": label}
            if isinstance(hold, dict) and hold else {})


def _registry_snapshot(drive_root: Any, *, allow_missing: bool = False) -> tuple[dict, bool]:
    """Read one whole revision. Source failure is never an authoritative empty set."""
    from ouroboros.projects_registry import _registry_path

    def unique_object(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate registry member: {key}")
            result[key] = value
        return result

    try:
        with _registry_path(drive_root).open(encoding="utf-8") as source:
            data = json.load(source, object_pairs_hook=unique_object)
    except FileNotFoundError:
        if allow_missing:
            return {"projects": []}, False
        raise
    if not isinstance(data, dict) or not isinstance(data.get("projects"), list):
        raise ValueError("Project registry must contain a projects array")
    return data, True


def _routing_row(raw: Any, *, identity_only: bool = False) -> dict:
    from ouroboros.projects_registry import PROJECT_ACTIVE, PROJECT_LIFECYCLES

    if not isinstance(raw, dict):
        raise ValueError("Project registry contains a non-object row")
    pid = raw.get("id")
    if not isinstance(pid, str) or not pid or sanitize_project_id(pid) != pid:
        raise ValueError("Project registry contains an invalid id")
    row = {"lifecycle": PROJECT_ACTIVE, "routing_generation": 0,
           "chat_id": project_chat_id(pid), "working_dir": "", **raw}
    if type(row["chat_id"]) is not int or row["chat_id"] <= 0 or not identity_only and (
            not isinstance(row["lifecycle"], str) or row["lifecycle"] not in PROJECT_LIFECYCLES
            or type(row["routing_generation"]) is not int or row["routing_generation"] < 0
            or not isinstance(row["working_dir"], str)
            or any(key in row and (not isinstance(row[key], str) or not row[key])
                   for key in ("routing_incarnation", "created_at"))):
        raise ValueError(f"Project {pid!r} has malformed routing authority")
    return row


def _strict_admission_snapshot(drive_root: Any, *, allow_missing: bool = False,
                               identity_only: bool = False, preserve_raw: bool = False) -> tuple[dict, bool]:
    """Lockless committed authority. Only absent legacy fields receive defaults.

    ``allow_missing`` admits absence only while no commit witness exists: beside
    it a missing registry is unavailable, so no strict reader or registry writer
    mistakes that loss for zero rooms. A registry lost before its first witness
    stamp landed (never committed, or a failed stamp not yet retried) still reads
    as a first boot. ``identity_only`` checks every row's object, unique id and
    chat reservation, leaving routing fields to the reader that selects a room
    (``_routing_row``): a malformed unrelated room never blocks a healthy one.
    Writers preserve raw neighbours and validate the selected row; the derived
    folder census needs every room's routing. Raw rows are never normalized.
    """
    if allow_missing:
        from ouroboros.projects_registry import _registry_witness_path

        allow_missing = not _registry_witness_path(drive_root).exists()
    data, present = _registry_snapshot(drive_root, allow_missing=allow_missing)
    rows, seen, chats = [], set(), set()
    for raw in data["projects"]:
        row = _routing_row(raw, identity_only=identity_only)
        if row["id"] in seen or row["chat_id"] in chats:
            raise ValueError("Project registry contains a duplicate id or chat reservation")
        seen.add(row["id"])
        chats.add(row["chat_id"])
        rows.append(raw if preserve_raw else row)
    return {**data, "projects": rows}, present


def reserved_project_for_chat(drive_root: Any, chat_id: int) -> dict:
    """Strict execution chat routing: every row's identity, then the matched room's routing.

    Absence is positive only without a commit witness; beside it a missing
    registry is unavailable, never no rooms.
    """
    rows = _strict_admission_snapshot(drive_root, allow_missing=True, identity_only=True)[0]["projects"]
    return next((_routing_row(row) for row in rows if row["chat_id"] == chat_id), {})


def display_registry_snapshot(drive_root: Any) -> dict:
    """Identity/history projection, never execution authority or a repair writer.

    Healthy rows survive an invalid neighbor. Retain identifiable chat reservations
    even when their routing fields are invalid; never turn invalid lifecycle active.
    """
    from ouroboros.projects_registry import PROJECT_ACTIVE

    data, _ = _registry_snapshot(drive_root, allow_missing=True)
    rows = []
    for raw in data["projects"]:
        if not isinstance(raw, dict):
            continue
        pid = raw.get("id")
        if not isinstance(pid, str) or not pid or sanitize_project_id(pid) != pid:
            continue
        row = {"lifecycle": PROJECT_ACTIVE, "chat_id": project_chat_id(pid), **raw}
        if type(row["chat_id"]) is not int or row["chat_id"] <= 0:
            continue  # invalid identity cannot mint a substitute chat reservation
        if not isinstance(row["lifecycle"], str):
            row["lifecycle"] = "invalid"
        for field in ("routing_generation", "visible_revision"):
            try:
                row[field] = max(0, int(row.get(field) or 0))
            except (TypeError, ValueError):
                row[field] = 0
        if not isinstance(row.get("working_dir", ""), str):
            row["working_dir"] = ""
        rows.append(row)
    return {**data, "projects": rows}


def validate_project_admission(view: Any) -> dict:
    """Malformed carried evidence is unavailable, never legacy absence."""
    try:
        if view is None:
            # Older snapshot/result writers also emitted null for absence. There
            # is no evidence to distinguish that from a lost prepared identity.
            raise ProjectAdmissionError("project_routing_fence_lookup_failed",
                                        "A null Project admission basis has unknown historical identity.")
        if (not isinstance(view, dict) or "project" not in view
                or not isinstance(view.get("project_id"), str)
                or sanitize_project_id(view["project_id"]) != view["project_id"]
                or type(view.get("frozen")) is not bool
                or "registry_present" in view and type(view["registry_present"]) is not bool):
            raise ValueError("invalid admission carrier")
        prior = view["project"]
        if prior is not None:
            if not isinstance(prior, dict) or prior.get("id") != view["project_id"]:
                raise ValueError("invalid prepared Project")
            required = {"id", "chat_id", "lifecycle", "routing_generation", "working_dir"}
            if not required <= prior.keys():
                raise ValueError("incomplete prepared Project")
            _routing_row({k: v for k, v in prior.items()
                          if not (k in {"created_at", "routing_incarnation"} and v is None)})
        if "workspace_claims" in view:
            claims = view["workspace_claims"]
            if (not isinstance(claims, dict) or not isinstance(claims.get("workspace"), str)
                    or not claims["workspace"] or not isinstance(claims.get("owners"), list)
                    or any(not isinstance(v, list) or len(v) != 4 or not isinstance(v[0], str)
                           or not v[0] or sanitize_project_id(v[0]) != v[0]
                           or type(v[1]) is not int or v[1] < 0
                           or any(x is not None and (not isinstance(x, str) or not x) for x in v[2:])
                           for v in claims["owners"])):
                raise ValueError("invalid workspace claims")
    except (ValueError, TypeError) as exc:
        raise ProjectAdmissionError("project_routing_fence_lookup_failed",
                                    "The carried Project admission basis is invalid.") from exc
    return view


def task_project_membership(drive_root: Any, task: dict, *, bindings_snapshot: Optional[dict] = None) -> tuple[str, bool]:
    """Recover positive room evidence from host lineage before any scope-only permit.

    One strict bindings snapshot, including origin lookup; no registry lock.
    Exact/predecessor membership precedes origin
    and ancestor evidence; explicit scopes remain valid without any room binding.
    """
    from ouroboros.projects_registry import _load_bindings, origin_key, project_id_for_origin

    metadata = task.get("metadata") if isinstance(task.get("metadata"), dict) else {}
    intent = metadata.get("resource_intent")
    intent = intent if isinstance(intent, dict) else {}
    pid = str(task.get("project_id") or intent.get("project_id") or "").strip()
    bindings = (_load_bindings(drive_root, strict=True)["bindings"]
                if bindings_snapshot is None else bindings_snapshot)
    identities = [task.get(k) for k in ("id", "original_task_id", "timeout_retry_from", "root_task_id", "parent_task_id")]
    identities += [metadata.get(k) for k in ("origin_task_id", "origin_root_task_id")]
    known = intent.get("kind") == "room_default"
    for identity in identities:
        if not identity or identity not in bindings:
            continue
        row = bindings[identity]
        bound = row.get("project_id") if isinstance(row, dict) else None
        if not isinstance(bound, str) or not bound or sanitize_project_id(bound) != bound:
            raise ValueError("Project binding is unavailable")
        if not pid:
            pid = bound
        known = known or bound == pid
    # A retry may share the captured owner message with a bound sibling, even
    # after the room closes or disappears. Active-only display lookup loses that fact.
    key = origin_key(task.get("origin_message_ref"))
    if key is not None:
        bound = project_id_for_origin(drive_root, task["origin_message_ref"], strict=True,
                                      include_inactive=True, bindings_snapshot=bindings)
        if not pid:
            pid = bound
        known = known or bool(bound and bound == pid)
    return pid, known


def project_admission_basis(project_id: str, project: Optional[dict], *, frozen: bool = False) -> dict:
    """Carry the row actually consumed by preparation, including absence evidence.

    Frozen resources (explicit, inherited or already admitted) keep their folder;
    only room-default preparation depends on the room's generation and folder.
    Legacy rows have no incarnation: their existing created_at stays part of the
    basis. Reconstruction always mints an incarnation, so it cannot recreate ABA.
    """
    return {"project_id": project_id, "project": None if project is None else {
        key: project.get(key) for key in (
            "id", "chat_id", "lifecycle", "routing_generation", "working_dir",
            "routing_incarnation", "created_at",
        )}, "frozen": frozen}


def project_admission_view(drive_root: Any, project_id: str, *,
                           allow_unregistered: bool = False, frozen: bool = False) -> dict:
    data, present = _strict_admission_snapshot(drive_root, allow_missing=allow_unregistered, identity_only=True)
    project = next((_routing_row(row) for row in data["projects"] if row["id"] == project_id), None)
    if project is None and not allow_unregistered:
        raise ProjectAdmissionError("project_routing_fence_changed", "The registered Project is missing.")
    return {**project_admission_basis(project_id, project, frozen=frozen), "registry_present": present}


def _folder_owners(rows: Any, canonical: str) -> list:
    """Ordered claims of the active rooms whose folder resolves to ``canonical`` now.

    Every folder is resolved at each call: an earlier normcase(realpath) cannot
    certify a path that an external rename or symlink swap has since redirected.
    """
    from ouroboros.project_facts import _normalized_workspace
    from ouroboros.projects_registry import PROJECT_ACTIVE

    return [[row["id"], row["routing_generation"], row.get("routing_incarnation"), row.get("created_at")]
            for row in rows if row["lifecycle"] == PROJECT_ACTIVE and row["working_dir"]
            and _normalized_workspace(row["working_dir"]) == canonical]


def project_scope_admission(drive_root: Any, *, project_id: str = "", workspace_root: str = "") -> dict:
    """Select API/derived scope and its basis in ONE read, before drive preparation.

    Path matching uses normcase(realpath). For a derived scope, the ordered
    ownership claims are fenced at the final read, which resolves the folders
    again; activity and presentation fields never participate in that comparison.
    """
    if not project_id and not workspace_root:
        return project_admission_basis("", None, frozen=True)
    from ouroboros.project_facts import _normalized_workspace
    import hashlib

    derived = not project_id  # its folder census reads every row's routing strictly
    data, present = _strict_admission_snapshot(drive_root, allow_missing=True, identity_only=not derived)
    rows = data["projects"]
    if derived:
        canonical = _normalized_workspace(workspace_root)
        owners = _folder_owners(rows, canonical)
        project_id = owners[0][0] if owners else ""
        if not project_id:
            project_id = "proj_" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:12]
        census = {"workspace": canonical, "owners": owners}
    else:
        census = None
    project = next((_routing_row(row) for row in rows if row["id"] == project_id), None)
    view = {**project_admission_basis(project_id, project, frozen=True), "registry_present": present}
    if census is not None:
        view["workspace_claims"] = census
    return view


@contextmanager
def project_admission_guard(drive_root: Any, view: dict, *, snapshot: Optional[tuple[dict, bool]] = None):
    """Final strict read while the caller holds Q continuously through append.

    Opening the committed file is the admission linearization point. Assignment,
    snapshot and deletion census also own Q. A rebind overlapping read→append may
    linearize after admission; a completed prior rebind must match the original
    resource basis. No registry lock, stat shortcut or provisional queue row.
    This removes lock waits, not the O(registry size) read/parse cost under Q,
    nor a derived census's filesystem resolution of each active folder.
    """
    from ouroboros.projects_registry import PROJECT_ACTIVE

    validate_project_admission(view)
    pid, prior = view["project_id"], view["project"]
    if snapshot is None:
        data, present = _strict_admission_snapshot(
            drive_root, allow_missing=not view.get("registry_present", True), identity_only=True)
        indexed = {row["id"]: row for row in data["projects"]}
    else:
        # Only a caller continuously holding Q may share this committed read.
        # It is operation-local, never a cache across assignment passes.
        indexed, present = snapshot
        if not present and view.get("registry_present", True):
            raise FileNotFoundError("Project registry is unavailable")
    rows = indexed.values()
    current = indexed.get(pid)
    current = None if current is None else _routing_row(current)
    if current is not None and current["lifecycle"] != PROJECT_ACTIVE:
        raise ProjectAdmissionError("project_routing_fence", "The Project no longer accepts new work.",
                                    current["lifecycle"])
    keys = ("id", "chat_id", "lifecycle", "routing_incarnation", "created_at")
    if not view.get("frozen"):
        keys += ("routing_generation", "working_dir")
    if ((prior is None) != (current is None)
            or prior is not None and any(prior.get(key) != current.get(key) for key in keys)):
        raise ProjectAdmissionError("project_routing_fence_changed", "The Project changed during task preparation.")
    if "workspace_claims" in view:
        from ouroboros.project_facts import _normalized_workspace

        claims = view["workspace_claims"]
        # The task folder and every active room folder resolve again under Q, so a
        # symlink swap before this read cannot admit a second identity for one
        # folder; a swap after admission stays unfenced. The census is whole-registry strict.
        if (_normalized_workspace(claims["workspace"]) != claims["workspace"]
                or _folder_owners(map(_routing_row, rows), claims["workspace"]) != claims["owners"]):
            raise ProjectAdmissionError("project_routing_fence_changed", "Project folder ownership changed during preparation.")
    yield current
