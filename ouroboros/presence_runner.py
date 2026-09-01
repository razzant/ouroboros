"""Fresh-agent execution for bounded, host-admitted presence turns."""

from __future__ import annotations

import hashlib
import os
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from ouroboros.artifacts import stage_task_attachments
from ouroboros.contracts.task_contract import attach_task_contract
from ouroboros.presence_admission import PresenceAdmission
from ouroboros.presence_authority import presence_ceiling_payload
from ouroboros.task_results import load_task_result
from ouroboros.utils import append_jsonl, read_json_dict, utc_now_iso


class PresenceTurnError(ValueError):
    def __init__(
        self,
        code: str,
        field: str,
        *,
        attachment_manifest: Sequence[Mapping[str, Any]] = (),
    ) -> None:
        self.code = str(code or "presence_turn_failed")
        self.field = str(field or "presence_turn")
        self.attachment_manifest = [
            dict(row) for row in attachment_manifest if isinstance(row, Mapping)
        ]
        super().__init__(f"{self.code}: {self.field}")


@dataclass(frozen=True)
class PresenceTurnEvent:
    source_event_id: str
    provider: str
    account_id: str
    conversation_id: str
    thread_id: str
    conversation_key: str
    actor: Mapping[str, Any]
    conversation: Mapping[str, Any]
    message: Mapping[str, Any]
    text: str


@dataclass(frozen=True)
class PresenceTurnResult:
    outcome: str
    text: str
    task_id: str
    work_ref: str = ""


def build_presence_result_event(task: dict[str, Any], text: str, ctx: Any) -> dict[str, Any]:
    """Freeze typed delivery metadata before the ordinary durable result write."""

    completion = getattr(ctx, "_presence_completion", None)
    completion = completion if isinstance(completion, dict) else {}
    outcome = str(completion.get("outcome") or "message").strip()
    if outcome not in {"message", "silent", "tool_delivered", "deferred"}:
        outcome = "message"
    handoff = getattr(ctx, "_swarm_handoff_attempt", None)
    handoff = handoff if isinstance(handoff, dict) else {}
    work_ref = (
        str(handoff.get("task_id") or "")
        if str(handoff.get("status") or "") == "scheduled"
        else ""
    )
    if outcome == "deferred" and not work_ref:
        outcome = "message"
    result_text = str(completion.get("message") or text or "")
    metadata = task.get("metadata") if isinstance(task.get("metadata"), dict) else {}
    metadata["presence_outcome"] = outcome
    if outcome in {"message", "deferred"}:
        metadata["presence_result_text"] = result_text
    if work_ref:
        metadata["presence_work_ref"] = work_ref
    task["metadata"] = metadata
    return {
        "type": "presence_result",
        "task_id": str(task.get("id") or ""),
        "outcome": outcome,
        "text": result_text if outcome in {"message", "deferred"} else "",
        "work_ref": work_ref,
        "ts": utc_now_iso(),
    }


class PresenceTurnGate:
    """Cross-process cap plus one active turn for each conversation."""

    def __init__(self, max_active: int = 2, *, state_root: Path | None = None) -> None:
        self._max_active = max(1, int(max_active))
        self._slots = threading.BoundedSemaphore(max(1, int(max_active)))
        self._guard = threading.Lock()
        self._conversations: dict[str, threading.Lock] = {}
        self._state_root = Path(state_root).resolve(strict=False) if state_root is not None else None
        self._claimed_slots: set[int] = set()

    @contextmanager
    def _file_lock(self, path: Path):
        from ouroboros.platform_layer import file_lock_exclusive, file_unlock

        path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
        try:
            file_lock_exclusive(fd)
            yield
        finally:
            try:
                file_unlock(fd)
            finally:
                os.close(fd)

    @contextmanager
    def _file_slot(self):
        from ouroboros.platform_layer import file_lock_exclusive_nb, file_unlock

        if self._state_root is None:
            with self._slots:
                yield
            return
        slot_root = self._state_root / "presence_turn_gate"
        slot_root.mkdir(parents=True, exist_ok=True)
        while True:
            for index in range(self._max_active):
                with self._guard:
                    if index in self._claimed_slots:
                        continue
                    self._claimed_slots.add(index)
                path = slot_root / f"slot-{index}.lock"
                fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
                try:
                    file_lock_exclusive_nb(fd)
                except OSError:
                    os.close(fd)
                    with self._guard:
                        self._claimed_slots.discard(index)
                    continue
                try:
                    yield
                finally:
                    try:
                        file_unlock(fd)
                    finally:
                        os.close(fd)
                        with self._guard:
                            self._claimed_slots.discard(index)
                return
            time.sleep(0.05)

    def run(self, conversation_key: str, callback: Callable[[], PresenceTurnResult]) -> PresenceTurnResult:
        key = str(conversation_key or "").strip()
        if not key:
            raise PresenceTurnError("presence_conversation_key_required", "conversation_key")
        with self._guard:
            conversation_lock = self._conversations.setdefault(key, threading.Lock())
        with conversation_lock:
            if self._state_root is None:
                with self._file_slot():
                    return callback()
            digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
            conversation_path = self._state_root / "presence_turn_gate" / f"conversation-{digest}.lock"
            with self._file_lock(conversation_path):
                with self._file_slot():
                    return callback()


_GATES_LOCK = threading.Lock()
_GATES: dict[tuple[str, int], PresenceTurnGate] = {}


def _configured_gate(drive_root: Path | None = None) -> PresenceTurnGate:
    from ouroboros.config import SETTINGS_DEFAULTS, _bounded_positive_int_setting

    limit = _bounded_positive_int_setting(
        "OUROBOROS_PRESENCE_MAX_ACTIVE",
        default=int(SETTINGS_DEFAULTS["OUROBOROS_PRESENCE_MAX_ACTIVE"]),
        hard_max=20,
    )
    state_root = Path(drive_root).resolve(strict=False) / "state" if drive_root is not None else None
    key = (str(state_root or ""), limit)
    with _GATES_LOCK:
        return _GATES.setdefault(key, PresenceTurnGate(limit, state_root=state_root))


def _stable_numeric_id(prefix: str, value: str) -> int:
    digest = hashlib.sha256(f"{prefix}\0{value}".encode("utf-8")).digest()
    return (1 << 40) + (int.from_bytes(digest[:6], "big") & ((1 << 40) - 1))


def _task_id(admission: PresenceAdmission, event: PresenceTurnEvent) -> str:
    digest = hashlib.sha256(
        f"{admission.binding_id}\0{event.source_event_id}".encode("utf-8")
    ).hexdigest()
    return f"presence-{digest[:24]}"


def _cached_result(drive_root: Path, task_id: str) -> PresenceTurnResult | None:
    stored = load_task_result(drive_root, task_id) or {}
    if str(stored.get("status") or "") not in {"completed", "failed"}:
        return None
    metadata = stored.get("metadata") if isinstance(stored.get("metadata"), dict) else {}
    outcome = str(metadata.get("presence_outcome") or "message")
    if outcome not in {"message", "silent", "tool_delivered", "deferred"}:
        outcome = "message"
    return PresenceTurnResult(
        outcome=outcome,
        text=(
            str(metadata.get("presence_result_text") or stored.get("result") or "")
            if outcome in {"message", "deferred"}
            else ""
        ),
        task_id=task_id,
        work_ref=str(metadata.get("presence_work_ref") or ""),
    )


def _log_dialogue(
    drive_root: Path,
    *,
    direction: str,
    chat_id: int,
    user_id: int,
    text: str,
    event: PresenceTurnEvent,
    task: Mapping[str, Any],
    task_id: str,
) -> None:
    from ouroboros.dialogue_provenance import presence_provenance_from_task

    state = read_json_dict(drive_root / "state" / "state.json") or {}
    append_jsonl(
        drive_root / "logs" / "chat.jsonl",
        {
            "ts": utc_now_iso(),
            "session_id": state.get("session_id"),
            "direction": direction,
            "chat_id": chat_id,
            "user_id": user_id,
            "text": text,
            "format": "markdown" if direction == "out" else "",
            "source": f"presence:{event.provider}",
            "sender_label": str(event.actor.get("display_name") or event.actor.get("username") or ""),
            "sender_session_id": str(event.actor.get("platform_actor_id") or event.actor.get("id") or ""),
            "client_message_id": event.source_event_id,
            "transport": {
                "provider": event.provider,
                "account_id": event.account_id,
                "conversation_id": event.conversation_id,
                "thread_id": event.thread_id,
                "conversation_key": event.conversation_key,
                "actor": dict(event.actor),
                "conversation": dict(event.conversation),
                "message": dict(event.message),
            },
            "presence_provenance": presence_provenance_from_task(task),
            "task_id": task_id,
        },
    )


def _build_task(
    admission: PresenceAdmission,
    event: PresenceTurnEvent,
    *,
    drive_root: Path,
    staged_files: Sequence[Path],
) -> dict[str, Any]:
    task_id = _task_id(admission, event)
    chat_id = _stable_numeric_id("presence-conversation", event.conversation_key)
    actor_id = _stable_numeric_id(
        "presence-actor",
        f"{event.provider}:{event.account_id}:{event.actor.get('platform_actor_id') or event.actor.get('id') or ''}",
    )
    presence_context = {
        "binding_id": admission.binding_id,
        "transport_skill": admission.transport_skill,
        "behavior_skill": admission.behavior_skill,
        "profile_fingerprint": admission.profile_fingerprint,
        "instructions": admission.instructions,
        "context_topics": list(admission.context_topics),
        "event": {
            "source_event_id": event.source_event_id,
            "provider": event.provider,
            "account_id": event.account_id,
            "conversation_id": event.conversation_id,
            "thread_id": event.thread_id,
            "conversation_key": event.conversation_key,
            "actor": dict(event.actor),
            "conversation": dict(event.conversation),
            "message": dict(event.message),
            "origin": admission.origin.__dict__,
            "destination": admission.destination.__dict__,
        },
    }
    metadata: dict[str, Any] = {
        "source": "presence",
        "client_message_id": event.source_event_id,
        "inline_max_rounds": admission.inline_max_rounds,
        "presence": presence_context,
    }
    if admission.model_slot == "light":
        from ouroboros.config import get_light_model

        metadata["model"] = get_light_model()
        metadata["use_local_model"] = os.environ.get("USE_LOCAL_LIGHT", "").lower() in {"true", "1"}
    task: dict[str, Any] = {
        "id": task_id,
        "type": "presence",
        "chat_id": chat_id,
        "actor_id": str(event.actor.get("platform_actor_id") or event.actor.get("id") or actor_id),
        "text": str(event.text or "").strip(),
        "_is_direct_chat": True,
        "_presence_turn": True,
        "context_requires_development": False,
        "metadata": metadata,
        "task_contract": {"capability_ceiling": presence_ceiling_payload(admission.capability_ceiling)},
    }
    manifest = stage_task_attachments(
        drive_root,
        task_id,
        [{"path": str(path), "label": path.name} for path in staged_files],
    )
    # Partial staging is the default for initial-task ingress (В25c, capinv-447):
    # good attachments stage, rejected ones ride along as disclosed manifest
    # rows — mirrors the gateway task API default. A FULLY-rejected set stays
    # atomic: the turn would run with none of its declared material.
    if manifest:
        from ouroboros.artifacts import (
            attachment_manifest_all_rejected,
            remove_staged_attachments,
        )

        if attachment_manifest_all_rejected(manifest):
            remove_staged_attachments(manifest)
            raise PresenceTurnError(
                "presence_attachment_admission_rejected",
                "staged_files",
                attachment_manifest=manifest,
            )
    if manifest:
        from ouroboros.gateway.tasks import _render_attachment_lines

        # The manifest is task authority, not merely presentation prose.  Keep
        # every staged/rejected declaration on the canonical carrier before the
        # task contract is normalized so a later promotion or child can inherit
        # and materialize the exact inputs.
        task["attachments"] = [dict(item) for item in manifest]
        task["attachment_images"] = [
            dict(item) for item in manifest
            if str(item.get("status") or "staged") == "staged" and item.get("is_image")
        ]
        rendered = _render_attachment_lines(manifest)
        if rendered:
            task["text"] = f"{task['text']}\n\n[ATTACHMENTS]\n{rendered}\n[END_ATTACHMENTS]".strip()
    if not task["text"]:
        task["text"] = "(attachments received)" if manifest else "(empty presence event)"
    return attach_task_contract(task)


def run_presence_turn(
    *,
    admission: PresenceAdmission,
    event: PresenceTurnEvent,
    repo_dir: Path,
    drive_root: Path,
    staged_files: Sequence[Path] = (),
    event_queue: Any = None,
    agent_factory: Callable[..., Any] | None = None,
    gate: PresenceTurnGate | None = None,
) -> PresenceTurnResult:
    """Run one bounded turn; adapters retain durable provider custody."""

    task_id = _task_id(admission, event)
    cached = _cached_result(Path(drive_root), task_id)
    if cached is not None:
        return cached

    def execute() -> PresenceTurnResult:
        second_cached = _cached_result(Path(drive_root), task_id)
        if second_cached is not None:
            return second_cached
        task = _build_task(
            admission,
            event,
            drive_root=Path(drive_root),
            staged_files=tuple(Path(item) for item in staged_files),
        )
        chat_id = int(task["chat_id"])
        actor_id = _stable_numeric_id("presence-actor-log", str(task.get("actor_id") or ""))
        _log_dialogue(
            Path(drive_root),
            direction="in",
            chat_id=chat_id,
            user_id=actor_id,
            text=event.text or task["text"],
            event=event,
            task=task,
            task_id=task_id,
        )
        if agent_factory is None:
            from ouroboros.agent import make_agent

            factory = make_agent
        else:
            factory = agent_factory
        agent = factory(
            repo_dir=str(repo_dir),
            drive_root=str(drive_root),
            event_queue=event_queue,
        )
        events = agent.handle_task(task)
        row = next((item for item in events if item.get("type") == "presence_result"), None)
        if not isinstance(row, dict):
            raise PresenceTurnError("presence_result_missing", "presence_result")
        result = PresenceTurnResult(
            outcome=str(row.get("outcome") or "message"),
            text=str(row.get("text") or ""),
            task_id=task_id,
            work_ref=str(row.get("work_ref") or ""),
        )
        if result.outcome in {"message", "deferred"} and result.text:
            _log_dialogue(
                Path(drive_root),
                direction="out",
                chat_id=chat_id,
                user_id=0,
                text=result.text,
                event=event,
                task=task,
                task_id=task_id,
            )
        return result

    return (gate or _configured_gate(Path(drive_root))).run(event.conversation_key, execute)


__all__ = [
    "PresenceTurnError",
    "PresenceTurnEvent",
    "PresenceTurnGate",
    "PresenceTurnResult",
    "build_presence_result_event",
    "run_presence_turn",
]
