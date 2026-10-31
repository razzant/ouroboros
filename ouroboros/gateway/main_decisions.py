"""What waits on the owner in Main, projected from addressed canonical obligations.

The ``owner_questions`` set is the store's flag over task documents, never copies
of questions or another decision lifecycle. Polling reads only its members and the
existing live task/wait owners, including children. Main's waiting pill counts these
questions and actions and reaches their cards in the feed; it never mounts a copy from
here. An incomplete read is disclosed so consumers retain known cards without enabling
stale answer controls.
"""
from __future__ import annotations

import json
import logging
from typing import Any

log = logging.getLogger(__name__)


def question_display_rooms(drive_root: Any, task_id: str, quizzes: Any, *, rooms=None, project_ids=None) -> dict:
    """Shared detail/census proof; context and immutable delivery are not display authority."""
    from ouroboros.projects_registry import RoomResolver, list_reserved_projects
    from ouroboros.presence_owner_contact import private_main_contact

    if not isinstance(quizzes, dict) or not quizzes:
        return {}
    private = {qid: {"state": "known", "chat_id": 1, "project_id": ""} for qid, block in quizzes.items()
               if isinstance(block, dict) and block.get("quiz_id") == qid and private_main_contact(block, task_id)}
    if len(private) == len(quizzes):
        return private
    unknown = {qid: private.get(qid, {"state": "unknown"}) for qid in quizzes}
    try:
        rooms = rooms or RoomResolver(drive_root, strict=True)
        chat = rooms(task_id)
        own = rooms.lineage(task_id)
        if not own:
            return unknown
        if project_ids is None:
            project_ids = {p["chat_id"]: p["id"] for p in list_reserved_projects(drive_root, strict=True)}
        if chat and chat not in project_ids:
            return unknown
        result = {}
        for qid, block in quizzes.items():
            if not isinstance(block, dict) or block.get("quiz_id") != qid:
                result[qid] = {"state": "unknown"}
                continue
            display = chat or own.get("chat_id")
            if qid in private:
                result[qid] = private[qid]
                continue
            # Hidden authors have no display room until THIS quiz's recorded Main publication.
            if display == 0:
                display = (1 if type(block.get("publication_chat_id")) is int
                           and block["publication_chat_id"] == 1
                           and block.get("publication_evidence") in {"pre_exposure", "history"} else None)
            result[qid] = ({"state": "known", "chat_id": display, "project_id": project_ids.get(display, "")}
                           if type(display) is int else {"state": "unknown"})
        return result
    except Exception:
        log.debug("question display room unavailable for %s", task_id, exc_info=True)
        return unknown


def _native_question(task_id: str, quiz_id: str, block: dict, owner_wait: Any) -> dict:
    """Main's own question is already its native card in Main: navigation needs its identity only."""
    from ouroboros.contracts.chat_id_policy import WEB_UI_CHAT_ID

    row = {"task_id": task_id, "quiz_id": quiz_id, "chat_id": WEB_UI_CHAT_ID, "native": True,
           "quiz_state": str(block.get("state") or "open"), "ts": str(block.get("asked_at") or "")}
    if isinstance(owner_wait, dict) and owner_wait.get("quiz_id") == quiz_id:
        row["owner_wait_state"] = str(owner_wait.get("state") or "")
    return row


def human_required_model_wait(row: dict, attempt: int) -> bool:
    """Use this live owner's revision-bound action fact, never classify labels."""
    return (row.get("state") == "waiting" and row.get("task_attempt") == attempt
            and isinstance(row.get("revision"), int) and not isinstance(row["revision"], bool)
            and row["revision"] > 0 and row.get("requires_owner_action") is True)


def owner_actions(activity: dict) -> tuple[dict, bool]:
    """One live owner's owner-required model waits and explicit Resume: Main's attention rule, which the work
    projection's ``turn`` shares (gateway/work_status.py). A parked row runs no wait loop, so its waits offer
    nothing; Resume is the census's own explicit-Resume rule (``can_resume``)."""
    phase = activity.get("phase")
    attempt = int(activity.get("task_attempt") or 1)
    current = {} if phase in {"budget_paused", "budget_pausing"} else activity.get("model_waits") or {}
    waits = {key: row for key, row in current.items() if isinstance(row, dict) and human_required_model_wait(row, attempt)}
    from ouroboros.work_stop import activity_stop_view

    view = activity_stop_view(activity)
    return waits, (phase in {"budget_paused", "budget_pausing"} and not activity.get("live")
                   and view["resumable"] and activity.get("can_resume") is True)


def _question_sources(drive_root: Any, candidates: dict) -> dict:
    """Read the existing display views in one batch, retaining strict member gaps."""
    from ouroboros.task_results import _admit_stored, load_task_result, validate_task_id
    from ouroboros.task_store import display_documents

    documents, sources = None, {}
    try:
        documents = display_documents(drive_root, candidates)
    except Exception:
        # One damaged member must not hide healthy questions beside its gap.
        log.debug("Main question batch unavailable; reading per task", exc_info=True)
    for task_id in candidates:
        try:
            result = (json.loads(documents[task_id]) if documents is not None
                      else load_task_result(drive_root, task_id, strict=True))
            if not result:
                raise ValueError("question source missing")
            _admit_stored(result, validate_task_id(task_id), strict=True, noun="task result authority")
            if not isinstance(result.get("owner_quiz") or {}, dict):
                raise ValueError("owner_quiz is not an object")
            sources[task_id] = result
        except Exception:
            sources[task_id] = None
            log.debug("Main question source unavailable: %s", task_id, exc_info=True)
    return sources


def main_decisions_snapshot(drive_root: Any, activities: list[dict], bindings: dict,
                            *, activities_complete: bool = True, bindings_complete: bool = True) -> dict:
    """Return complete current question forms and existing human recovery controls.

    Project lookup uses the complete registry, including archived/deleting rows;
    a sidebar display limit never changes destination or visibility authority.
    Question documents share the existing display batch; ancestry is warmed by
    generation. A failed batch keeps the strict per-member read and its gaps.
    """
    from ouroboros.contracts.chat_id_policy import WEB_UI_CHAT_ID
    from ouroboros.obligations import members
    from ouroboros.owner_quiz import quiz_needs_retention
    from ouroboros.project_dialogue import project_question_pointer
    from ouroboros.projects_registry import RoomResolver, list_reserved_projects

    questions, attention = [], []
    gaps, action_gaps = [], []
    questions_complete, attention_complete = bindings_complete, activities_complete
    if not bindings_complete:
        gaps.append({"reason": "task_bindings_unavailable"})
    try:
        projects = {str(row["id"]): row for row in list_reserved_projects(drive_root, strict=True)}
        projects_by_chat = {int(row["chat_id"]): row for row in projects.values() if row.get("chat_id")}
    except Exception:
        log.debug("Main decisions Project registry unavailable", exc_info=True)
        projects, projects_by_chat = {}, {}
        questions_complete = attention_complete = False
        gaps.append({"reason": "project_registry_unavailable"})
    try:
        candidates = members(drive_root, "owner_questions")
    except Exception:
        log.debug("Main question obligations unavailable", exc_info=True)
        candidates = {}
        questions_complete = False
        gaps.append({"reason": "question_index_unavailable"})
    sources = _question_sources(drive_root, candidates)
    rooms = None  # one strict bindings snapshot; never turn unreadable ancestry into unbound
    try:
        # Missing/damaged sources have no lineage proof to cache (including after repair).
        if readable := [tid for tid, source in sources.items() if source is not None]:
            rooms = RoomResolver(drive_root, strict=True)
            rooms.warm(readable)
    except Exception:
        rooms = None
        log.debug("Main question rooms unavailable", exc_info=True)
    project_ids = {chat: row["id"] for chat, row in projects_by_chat.items()}
    for task_id in candidates:
        try:
            result = sources[task_id]
            if result is None:
                raise ValueError("question source missing")
            quizzes = result.get("owner_quiz") or {}
            if rooms is None:
                raise ValueError("question room unavailable")
            proof = question_display_rooms(drive_root, task_id, quizzes, rooms=rooms,
                                           project_ids=project_ids)
            for quiz_id, block in quizzes.items():
                if not quiz_needs_retention(block):
                    continue
                published = block.get("publication_chat_id")
                if (not isinstance(published, int) or isinstance(published, bool)
                        or block.get("publication_evidence") not in {"pre_exposure", "history"}):
                    questions_complete = False
                    gaps.append({"reason": "publication_address_unknown", "task_id": task_id, "quiz_id": quiz_id})
                    continue  # sending/migration binds an address BEFORE any exposure
                room = proof.get(quiz_id) or {}
                if room.get("state") != "known":
                    raise ValueError("question room unavailable")
                project = projects.get(room["project_id"])
                if not project:
                    if room.get("chat_id") == WEB_UI_CHAT_ID and not room.get("project_id"):
                        questions.append(_native_question(task_id, quiz_id, block, result.get("owner_wait")))
                    continue  # another known room; contextual project_id creates no mirror or gap
                pointer = project_question_pointer(
                    {"task_id": task_id, "quiz_id": quiz_id}, block, project,
                    result.get("owner_wait"),
                )
                if pointer:
                    questions.append(pointer)
                else:
                    questions_complete = False
                    gaps.append({"reason": "question_form_unavailable", "task_id": task_id, "quiz_id": quiz_id})
        except Exception:
            questions_complete = False
            gaps.append({"reason": "question_source_unavailable", "task_id": task_id})
            log.debug("Main question source unavailable: %s", task_id, exc_info=True)
    for activity in activities:
        task_id = str(activity.get("activity_id") or "")
        project = projects.get(str(activity.get("project_id") or ""))
        if not task_id or not project:
            continue
        # A quiz-bound owner_wait is represented by that exact question above.
        # Other model wait IDs and a Resume remain independent decisions, even
        # when parallel work in this same task also has a required quiz. An
        # unknown phase already marks the census incomplete. A parked row runs
        # no wait loop and a pausing one stops at its boundary (an owner Pause
        # interrupts a pre-dispatch wait: ``TaskModelWait.pre_dispatch_pause``),
        # so neither offers a wait action; Resume follows the row's release rule.
        phase = activity.get("phase")
        if phase == "unknown":
            continue
        attempt = int(activity.get("task_attempt") or 1)
        parked = phase in {"budget_paused", "budget_pausing"}
        current_waits = {} if parked else activity.get("model_waits") or {}
        for key, row in current_waits.items():
            if (isinstance(row, dict) and row.get("state") == "waiting"
                    and row.get("task_attempt") == attempt
                    and (not isinstance(row.get("requires_owner_action"), bool)
                         or not isinstance(row.get("revision"), int)
                         or isinstance(row.get("revision"), bool) or row["revision"] < 1)):
                attention_complete = False
                action_gaps.append({"reason": "model_wait_action_unknown", "task_id": task_id, "wait_id": key})
        # The census owner derived can_resume from the row's release rule: own
        # Pause, an unheld sleep and a self-reconciling hold stay quiet.
        waits, can_resume = owner_actions(activity)
        if not waits and not can_resume:
            continue
        from ouroboros.work_stop import activity_stop_view
        cause = activity_stop_view(activity)["pause_cause"]
        attention.append({**activity, "task_id": task_id, "project_name": project.get("name") or "Project",
                          "model_waits": waits, "can_resume": can_resume,
                          "cancelable": activity.get("cancelable") is True,
                          "attention_reason": (cause if cause in {"budget", "restart"} else "resume")
                          if can_resume else "model_access"})
    questions.sort(key=lambda row: (str(row.get("ts") or ""), row["task_id"], row["quiz_id"]))
    return {"current_questions": questions, "current_questions_complete": questions_complete,
            "current_questions_gaps": gaps,
            "main_attention": attention, "main_attention_complete": attention_complete,
            "main_attention_gaps": action_gaps}
