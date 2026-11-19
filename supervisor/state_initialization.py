"""Is this data root new? The one durable answer ``init_state`` consults (#1307).

Two missing ``state.json`` copies used to mean "fresh install", so a lost or
unreadable state minted a new owner slot, session and cleared Stop. First
initialization is now positive: ``state/state.initialized.json`` records ONE
initialization identity (``pending`` before the first state is written,
``complete`` after), written by supervisor boot (or an isolated benchmark's
explicit seed before boot) before admission, and by an owner Reset (``pending``,
``origin=owner_reset``). It holds identity, phase and the work-root basis
(``launch_roots``: a first initialization records ``app_root``, Reset and
adoption carry theirs) — no permissions, no history.

With both copies absent the decision is:

- a ``complete`` witness: initialized before, state LOST -> refuse (unavailable);
- a ``pending`` witness: an interrupted initialization -> finish the SAME id,
  unless supervisor-run evidence appeared since (then refuse);
- no witness: create only when a fixed, non-recursive set of evidence that a
  supervisor already ran here is positively absent (a legacy root whose state
  was lost is recovery, not first boot);
- an unreadable witness or evidence probe: refuse; unknown never mints identity.

Absence is proven, not inferred from an errno: Windows reports a file where
``state/`` belongs as ERROR_PATH_NOT_FOUND, the same ``FileNotFoundError`` as a
missing directory, so ``confirm_absent`` checks the entry and its ancestors.

Bootstrap scaffolding that may precede init — empty ``state``/``logs``/
``memory``/``task_results`` directories, ``settings.json`` from onboarding,
the usage ledger import, a benchmark sentinel — is deliberately not evidence.
A root whose every trace was deleted is indistinguishable from a new one: that
is a disclosed residual, not proof of historylessness.

What a created first state is born with is ``first_state`` (a re-mint after loss
and an owner Reset included): update notices a fresh install owes nothing, and the
consciousness switch its setup left on.
"""

from __future__ import annotations

import hashlib
import json
import logging
import pathlib
import uuid
from typing import Any, Dict, Tuple

# The witness reader and its fixed evidence vocabulary live in the stdlib leaf, so a direct entry
# can read them before its first path-freezing import (#1239); this module owns the writes.
from ouroboros.launch_roots import (
    APP_ROOT_BASIS,
    BASIS_KEY,
    adopted_basis,
    history_evidence,
    read_witness,
    witness_path,
)
from ouroboros.utils import utc_now_iso, write_bytes_atomic

log = logging.getLogger(__name__)


def _write(drive_root: Any, witness: Dict[str, Any]) -> None:
    path = witness_path(drive_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_bytes_atomic(path, (json.dumps({"schema_version": 1, **witness}, ensure_ascii=False, indent=2)
                              + "\n").encode("utf-8"), fsync=True)


def mark_pending(drive_root: Any, *, origin: str, work_root_basis: str = "") -> str:
    """Record an explicit fresh start (owner Reset) before the restart that runs init;
    the caller passes the basis it read before wiping ``state/`` so the root keeps it."""
    initialization_id = uuid.uuid4().hex
    _write(drive_root, {"initialization_id": initialization_id, "phase": "pending",
                        "origin": str(origin), "created_at": utc_now_iso(),
                        **({BASIS_KEY: work_root_basis} if work_root_basis else {})})
    return initialization_id


def supervisor_evidence(drive_root: Any) -> Tuple[str, str]:
    """``("none"|"present"|"unknown", path)``: did a supervisor already run here? The fixed
    walk is ``launch_roots.history_evidence``; only this reader can count the store's chat rows."""
    from ouroboros import dialogue_store as ds

    return history_evidence(drive_root, store_chat=lambda root: ds.present(root) and ds.stream_end(root, "chat") > 0,
                            store_errors=(ds.StoreError,))


def _decide(drive_root: Any) -> Dict[str, Any]:
    """``initialization_decision`` without its witness write: a create on a root with no witness has no id yet."""
    status, witness = read_witness(drive_root)
    if status in {"invalid", "unreadable"}:
        return {"create": False, "reason": "initialization_witness_unreadable", "detail": status}
    if status == "ok" and witness.get("phase") == "complete":
        return {"create": False, "reason": "initialized_state_lost",
                "detail": f"initialization {witness.get('initialization_id')} completed at "
                          f"{witness.get('completed_at') or '?'}"}
    explicit_reset = status == "ok" and witness.get("origin") == "owner_reset"
    evidence, where = supervisor_evidence(drive_root)
    if evidence == "unknown" and not explicit_reset:
        return {"create": False, "reason": "initialization_evidence_unknown", "detail": where}
    if evidence == "present" and not explicit_reset:
        return {"create": False, "reason": "prior_history_without_state", "detail": where}
    return {"create": True, **({"initialization_id": str(witness["initialization_id"])} if status == "ok" else {})}


def initialization_decision(drive_root: Any, *, origin: str = "first_boot") -> Dict[str, Any]:
    """Both state copies are positively absent: may a FIRST state be created?

    On yes, the witness is ``pending`` with the returned id before the caller
    writes the state; the caller completes it (``complete``) after."""
    decision = _decide(drive_root)
    if not decision["create"] or decision.get("initialization_id"):
        return decision
    initialization_id = uuid.uuid4().hex  # a genuinely new root: its work roots follow APP_ROOT
    _write(drive_root, {"initialization_id": initialization_id, "phase": "pending",
                        "origin": str(origin or "first_boot"), "created_at": utc_now_iso(),
                        BASIS_KEY: APP_ROOT_BASIS})
    return {"create": True, "initialization_id": initialization_id}


def first_state_pending(drive_root: Any) -> bool:
    """Would the next ``init_state`` CREATE a first state here? Both copies positively absent and the decision above,
    read without writing anything (the setup bootstrap and completion ask it). A completed witness whose copies are
    lost, or a copy that cannot be read, is not a first state: no."""
    from supervisor.state import read_state_copy

    root = pathlib.Path(drive_root)
    if any(read_state_copy(root / "state" / name)[0] != "missing" for name in ("state.json", "state.last_good.json")):
        return False
    return bool(_decide(root)["create"])


def first_state(drive_root: Any, initialization_id: str) -> Dict[str, Any]:
    """What a created first state of ``drive_root`` is born with, before ``init_state`` writes it.

    Runs under STATE_LOCK: nothing here calls ``update_state`` (the lock is not re-entrant) or
    takes a settings lock. A fresh install never ran an earlier release, so the notices
    ``upgrade_notices.BORN_SETTLED_NOTICE_KEYS`` names are born marked; that stamp is not a
    delivery (no chat receipt exists for it), and the notices judged by facts stay owed.
    Consciousness is born on when this root's setup left its switch on (``setup_left_consciousness_on``)."""
    from ouroboros.upgrade_notices import BORN_SETTLED_NOTICE_KEYS  # D08 -> D11 is allowed only lazily

    return {"initialization_id": initialization_id, **dict.fromkeys(BORN_SETTLED_NOTICE_KEYS, utc_now_iso()),
            **({"bg_consciousness_enabled": True} if setup_left_consciousness_on(drive_root) else {})}


def setup_left_consciousness_on(drive_root: Any) -> bool:
    """The setup choice the settings document of ``drive_root`` records (owner decision 2026-11-20): the configured
    document when that root is the configured data root, else ``<root>/settings.json``. Read lock-free and verified
    under STATE_LOCK, so it takes no settings lock; an unreadable document is no choice, and a pending installation
    Panic withholds it (Panic's own controls write keeps it off)."""
    from ouroboros import config
    from ouroboros.installation_stop import installation_panic
    from ouroboros.settings_integrity import read_settings_json_verified
    from ouroboros.settings_setup_contract import ONBOARDING_CONSCIOUSNESS_KEY

    root = pathlib.Path(drive_root)
    try:
        configured = root.resolve() == pathlib.Path(config.DATA_DIR).resolve()
        document = read_settings_json_verified(config.SETTINGS_PATH if configured else root / "settings.json")
        chosen = isinstance(document, dict) and document.get(ONBOARDING_CONSCIOUSNESS_KEY) == "on"
        return chosen and not installation_panic(root)
    except Exception:
        log.warning("the setup's consciousness choice could not be read; the first state is born off", exc_info=True)
        return False


def authority_reason(drive_root: Any, initialization_id: str) -> str:
    """Read-only admission proof: a durable completion of this exact identity."""
    status, witness = read_witness(drive_root)
    if status != "ok":
        return f"initialization_witness_{status}"
    if not initialization_id or witness["initialization_id"] != initialization_id:
        return "initialization_identity_mismatch"
    return "" if witness["phase"] == "complete" else "initialization_incomplete"


def prepare_adoption(drive_root: Any, state: Dict[str, Any]) -> str:
    """Bind readable legacy data or resume a pending identity before writing it.

    Caller holds STATE_LOCK. Readable legacy session/creation facts, rather than
    absence of files, are the adoption evidence. An existing witness never changes
    identity here; only the explicit Reset owner may replace it.
    """
    status, witness = read_witness(drive_root)
    identity = str(state.get("initialization_id") or "")
    source = pathlib.Path(drive_root) / "state" / "state.json"
    source_digest = hashlib.sha256(source.read_bytes()).hexdigest()
    if status == "ok":
        if (not identity and witness.get("phase") == "pending"
                and witness.get("origin") == "legacy_adopted"
                and witness.get("legacy_source_sha256") == source_digest):
            return str(witness["initialization_id"])
        if identity != witness["initialization_id"]:
            raise ValueError("initialization_identity_mismatch")
        return identity
    if status != "missing":
        raise ValueError(f"initialization_witness_{status}")
    if not identity and not (state.get("session_id") and state.get("created_at")):
        raise ValueError("legacy_initialization_evidence_missing")
    basis = adopted_basis(identity)  # predates the witness: home; lost its witness: unknown
    identity = identity or uuid.uuid4().hex
    _write(drive_root, {"initialization_id": identity, "phase": "pending",
                        "origin": "legacy_adopted", "created_at": utc_now_iso(),
                        "legacy_source_sha256": source_digest, BASIS_KEY: basis})
    return identity


def complete(drive_root: Any, initialization_id: str, *, adopted: bool) -> bool:
    """Complete only the exact pending identity after both state writes succeeded."""
    try:
        status, witness = read_witness(drive_root)
        if (status != "ok" or not initialization_id
                or witness.get("initialization_id") != initialization_id):
            return False
        if witness["phase"] != "complete":
            _write(drive_root, {**witness, "phase": "complete", "completed_at": utc_now_iso()})
        return not authority_reason(drive_root, initialization_id)
    except Exception:
        log.warning("state initialization witness could not be completed", exc_info=True)
        return False
