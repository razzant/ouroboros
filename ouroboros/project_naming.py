"""SSOT for project naming (Cluster B, v6.40).

An LLM-first short human title for a project card, with a deterministic heuristic
fallback. Shared by every path that names a project so the UI conversion and the
agent never drift:
  - the proactive card namer (names ANY task card up front, supervisor side);
  - ``gateway/projects.py`` turn-into-project conversion (reuses the up-front name,
    or names inline as a race fallback);
  - ``ensure_project_scope`` (the agent self-creates + names a project).

Doctrine:
  - P5 LLM-first: the model COINS the name; post-processing is purely lexical
    (first line, strip quotes, cap length) — never a keyword/regex semantic gate.
  - #4 self-DoS: the provider call goes through the per-model concurrency slot.
  - Fail-soft: any failure (no creds / provider error / timeout / empty output)
    returns the heuristic fallback. Never raises.
"""

from __future__ import annotations

import logging
import os
import pathlib
import threading
from dataclasses import replace
from typing import Any, Callable, Dict, Optional, Sequence

log = logging.getLogger("ouroboros.project_naming")


def _light_use_local(explicit: Optional[bool]) -> bool:
    """Resolve the light-lane local route for naming. Honor an explicit caller value;
    otherwise follow the runtime ``USE_LOCAL_LIGHT`` flag — naming runs on the LIGHT model,
    so it must route local/remote like every other light-lane caller (e.g. the safety
    check at ``ouroboros/safety.py::_resolve_safety_routing``) instead of hardcoding remote."""
    if explicit is not None:
        return bool(explicit)
    return str(os.environ.get("USE_LOCAL_LIGHT", "") or "").lower() in ("true", "1")

# Mirror gateway ``_MAX_DERIVED_NAME`` so heuristic and LLM names share one cap.
MAX_PROJECT_NAME = 60

_NAMING_PROMPT = (
    "Name this project from the owner's request below. Return EXACTLY one short "
    "human-readable title and nothing else. Use the SAME language as the request. "
    "2-6 words, at most 48 characters. No quotes, no trailing period, no emoji, no "
    "'Project:' prefix.\n\nOwner request:\n{request}"
)

# Lexical wrappers stripped from a model title (NOT a semantic filter).
_WRAP_CHARS = "\"'`«»“”‘’ \t"


def fallback_project_name(*candidates: object, max_len: int = MAX_PROJECT_NAME) -> str:
    """First non-empty candidate, whitespace-collapsed and capped. No LLM."""
    for raw in candidates:
        cleaned = " ".join(str(raw or "").split())
        if cleaned:
            if len(cleaned) > max_len:
                cleaned = cleaned[: max_len - 1].rstrip() + "…"
            return cleaned
    return ""


def clean_model_title(text: object, max_len: int = MAX_PROJECT_NAME) -> str:
    """Lexical cleanup of an LLM title — first non-empty line, strip wrapping
    quotes/backticks, drop a single trailing period, collapse whitespace, cap.
    This is NOT a semantic gate (P5): it never inspects the meaning, only the form."""
    raw = str(text or "")
    line = ""
    for candidate in raw.splitlines():
        if candidate.strip():
            line = candidate.strip()
            break
    line = line.strip(_WRAP_CHARS)
    if line.endswith("."):
        line = line[:-1]
    # Re-strip: a trailing quote can sit BEFORE the period (e.g. ``"Title".``).
    line = line.strip(_WRAP_CHARS)
    line = " ".join(line.split())
    if len(line) > max_len:
        line = line[: max_len - 1].rstrip() + "…"
    return line


def _light_naming_model() -> str:
    """The light slot, resolved to a credentialed provider (empty light -> main)."""
    from ouroboros.config import get_light_model
    from ouroboros.provider_models import resolve_credentialed_model

    return resolve_credentialed_model(get_light_model())


def _naming_timeout_sec() -> float:
    """Provider-call transport timeout for the naming LIGHT call. SSOT: config
    SETTINGS_DEFAULTS (no duplicated literal — the default IS the SSOT value)."""
    from ouroboros.config import SETTINGS_DEFAULTS

    default = SETTINGS_DEFAULTS["OUROBOROS_PROJECT_NAMING_TIMEOUT_SEC"]
    try:
        return float(os.environ.get("OUROBOROS_PROJECT_NAMING_TIMEOUT_SEC", default))
    except (TypeError, ValueError):
        return float(default)


def _naming_async_timeout_sec() -> float:
    """Gateway HARD wait for the inline turn-into-project name. SSOT: config
    SETTINGS_DEFAULTS (no duplicated literal — the default IS the SSOT value)."""
    from ouroboros.config import SETTINGS_DEFAULTS

    default = SETTINGS_DEFAULTS["OUROBOROS_PROJECT_NAMING_ASYNC_TIMEOUT_SEC"]
    try:
        return float(os.environ.get("OUROBOROS_PROJECT_NAMING_ASYNC_TIMEOUT_SEC", default))
    except (TypeError, ValueError):
        return float(default)


def _project_naming_usage_scope(drive_root: Optional[Any], task_id: str):
    """Bind a naming send to its task tree even from a daemon/gateway thread."""
    from ouroboros.usage_accounting import UsageScope, current_usage_scope

    active = current_usage_scope()
    if active is not None:
        return replace(active, category="project_naming", source="project_naming")

    persisted: dict[str, Any] = {}
    try:
        if drive_root is not None and task_id:
            from ouroboros.task_results import load_task_result

            persisted = load_task_result(drive_root, task_id) or {}
    except Exception:
        log.debug("project naming task scope lookup failed", exc_info=True)
    metadata = persisted.get("metadata") if isinstance(persisted.get("metadata"), dict) else {}
    scoped_task_id = str(persisted.get("task_id") or metadata.get("task_id") or task_id or "project_naming")
    root_task_id = str(persisted.get("root_task_id") or metadata.get("root_task_id") or scoped_task_id)
    parent_task_id = str(persisted.get("parent_task_id") or metadata.get("parent_task_id") or "")
    budget_root = persisted.get("budget_drive_root") or metadata.get("budget_drive_root") or drive_root
    from ouroboros.settings_setup_contract import resolve_total_budget_usd

    global_limit = resolve_total_budget_usd()
    try:
        root_limit = float(os.environ.get("OUROBOROS_PER_TASK_COST_USD", "0") or 0)
    except (TypeError, ValueError):
        root_limit = 0.0
    return UsageScope(
        drive_root=budget_root,
        task_id=scoped_task_id,
        root_task_id=root_task_id,
        parent_task_id=parent_task_id,
        category="project_naming",
        source="project_naming",
        global_limit_usd=global_limit,
        root_limit_usd=root_limit if root_limit > 0 else None,
    )


def llm_project_name(
    owner_text: object,
    *,
    fallback_candidates: Sequence[object] = (),
    use_local: Optional[bool] = None,
    llm_client: Optional[Any] = None,
    drive_root: Optional[Any] = None,
    task_id: str = "",
) -> str:
    """SYNC bounded LLM-first project title. On ANY failure returns the heuristic
    fallback over ``fallback_candidates`` then ``owner_text``. Never raises.

    ``use_local=None`` (the default) routes via the runtime ``USE_LOCAL_LIGHT`` flag so a
    local-only / local-light deployment names with its configured local model instead of a
    remote provider. The provider call is wrapped in the #4 per-model concurrency slot so a
    flurry of namers cannot storm one model's rate limit, carries a bounded transport timeout
    so a stalled provider can't wedge card creation, and — when ``drive_root`` is given —
    runs through ``chat_observed`` for its forensic trace. The physical send is bound to a
    ``project_naming`` usage scope, which is the sole monetary authority for the attempt.
    """
    fb = fallback_project_name(*list(fallback_candidates), owner_text)
    text = " ".join(str(owner_text or "").split())
    if not text:
        return fb
    use_local = _light_use_local(use_local)
    try:
        from ouroboros import model_concurrency
        from ouroboros.llm import LLMClient

        client = llm_client or LLMClient()
        model = _light_naming_model()
        # A title only needs the head of the request; bound the prompt input but mark the cut
        # explicitly (P1 — no SILENT truncation) rather than dropping the tail invisibly. The
        # full request is unaffected (this is only the naming prompt's view).
        naming_input = text if len(text) <= 4000 else text[:4000] + " …[request truncated for naming]"
        chat_kwargs = dict(
            messages=[{"role": "user", "content": _NAMING_PROMPT.format(request=naming_input)}],
            model=model,
            tools=None,
            reasoning_effort="low",
            max_tokens=256,
            use_local=use_local,
            timeout=_naming_timeout_sec(),
        )
        from ouroboros.usage_accounting import usage_scope

        with model_concurrency.model_call_slot(model, use_local):
            with usage_scope(_project_naming_usage_scope(drive_root, task_id)):
                if drive_root is not None:
                    from ouroboros.llm_observability import chat_observed

                    msg, _usage = chat_observed(
                        client,
                        drive_root=drive_root,
                        task_id=str(task_id or "project_naming"),
                        call_type="project_naming",
                        **chat_kwargs,
                    )
                else:
                    msg, _usage = client.chat(**chat_kwargs)
        name = clean_model_title((msg or {}).get("content", ""))
        return name or fb
    except Exception:
        log.debug("llm_project_name failed; using heuristic fallback", exc_info=True)
        return fb


async def llm_project_name_async(
    owner_text: object,
    *,
    fallback_candidates: Sequence[object] = (),
    timeout_sec: Optional[float] = None,
    use_local: Optional[bool] = None,
    llm_client: Optional[Any] = None,
    drive_root: Optional[Any] = None,
    task_id: str = "",
) -> str:
    """ASYNC variant for the gateway (Starlette) path: runs the bounded sync call off
    the event loop with a HARD timeout. ``timeout_sec=None`` (default) uses the config SSOT
    ``OUROBOROS_PROJECT_NAMING_ASYNC_TIMEOUT_SEC``. On timeout/failure returns the heuristic
    fallback. ``use_local=None`` defers to ``USE_LOCAL_LIGHT`` inside the sync helper.
    Never raises."""
    import asyncio

    fb = fallback_project_name(*list(fallback_candidates), owner_text)
    text = " ".join(str(owner_text or "").split())
    if not text:
        return fb
    eff_timeout = _naming_async_timeout_sec() if timeout_sec is None else float(timeout_sec)
    try:
        return await asyncio.wait_for(
            asyncio.to_thread(
                llm_project_name,
                owner_text,
                fallback_candidates=fallback_candidates,
                use_local=use_local,
                llm_client=llm_client,
                drive_root=drive_root,
                task_id=task_id,
            ),
            timeout=max(0.1, eff_timeout),
        )
    except Exception:
        log.debug("llm_project_name_async timed out/failed; using heuristic", exc_info=True)
        return fb


def _refresh_root_cost_after_naming(drive_root: Any, task_id: str) -> None:
    """Refresh a terminal root projection after the naming attempt settles."""
    try:
        from types import SimpleNamespace

        from ouroboros.agent_task_pipeline import _set_root_post_task_checkpoint
        from ouroboros.task_results import load_task_result

        current = load_task_result(drive_root, task_id) or {}
        refreshed = {**current, "id": task_id, "budget_drive_root": str(drive_root)}
        _set_root_post_task_checkpoint(
            SimpleNamespace(drive_root=pathlib.Path(drive_root)), refreshed, "refresh",
        )
    except Exception:
        log.debug("project naming cost refresh failed for %s", task_id, exc_info=True)


def spawn_proactive_namer(
    drive_root: Any, task_id: str, text: str, *, broadcast: Optional[Callable[[dict], None]] = None,
) -> None:
    """Proactively coin an LLM project name for a fresh card in a DAEMON thread (Cluster B).

    Writes the coined ``suggested_name`` onto the task result (turn-into-project then reuses
    it with zero extra call) and, via ``broadcast``, emits a ``task_named`` event so the live
    card shows a human title up front. NEVER blocks the task. ``drive_root`` is captured at
    CALL time — NOT read from a mutable module global at thread-execution time — so a later
    context switch (or a test that swaps the supervisor drive) can't redirect this thread's
    write. Skips cleanly unless ``drive_root`` is a real directory (test safety: a stub /
    MagicMock drive must never materialise a stray path — chat_observed persists BEFORE the
    LLM call). Fail-soft."""
    body = " ".join(str(text or "").split())
    if not body:
        return
    try:
        if not pathlib.Path(str(drive_root)).is_dir():
            return
    except (OSError, TypeError, ValueError):
        return

    def _work() -> None:
        try:
            # v6.58.0 (§3.4b): HARD total wall-clock bound. The transport timeout bounds
            # ONE attempt, but llm.chat's retry/fallback chain under a degraded provider
            # could stretch the whole call to tens of minutes (the incident where the
            # card was named 24 minutes late). A title is cosmetic: if it hasn't landed
            # within the transport budget + slack, drop it — the id/title heuristics and
            # the convert path's own bounded inline call (8s) already cover naming.
            _result: list[str] = []
            _detached = threading.Event()
            _finished = threading.Event()
            _refresh_lock = threading.Lock()
            _refreshed = False

            def _refresh_detached_once() -> None:
                nonlocal _refreshed
                with _refresh_lock:
                    if _refreshed:
                        return
                    _refreshed = True
                _refresh_root_cost_after_naming(drive_root, task_id)

            def _call() -> None:
                try:
                    _result.append(llm_project_name(body, drive_root=drive_root, task_id=task_id))
                except Exception:
                    log.debug("proactive namer inner call failed for %s", task_id, exc_info=True)
                finally:
                    _finished.set()
                    if _detached.is_set():
                        _refresh_detached_once()

            inner = threading.Thread(target=_call, name=f"namer-call-{task_id}", daemon=True)
            inner.start()
            if not _finished.wait(timeout=max(0.0, _naming_timeout_sec() + 30.0)):
                _detached.set()
                # Close the race where settlement lands between wait() and
                # the detached marker. The once-guard covers both interleavings.
                if _finished.is_set():
                    _refresh_detached_once()
                log.debug("proactive namer exceeded its wall-clock bound for %s; skipped", task_id)
                return
            inner.join()
            if not _result:
                log.debug("proactive namer exceeded its wall-clock bound for %s; skipped", task_id)
                return
            name = _result[0]
            if not name:
                return
            from ouroboros.task_results import (
                STATUS_RUNNING,
                load_task_result,
                write_task_result,
            )

            # Persist suggested_name as same-status ENRICHMENT, not a RUNNING transition: a
            # fast task may already be terminal (completed/failed/cancelled) by the time this
            # daemon finishes, and write_task_result's monotonic guard DROPS a regressing
            # RUNNING write — which would silently lose the name the convert path reuses.
            # Writing under the current on-disk status lets the monotonic guard's same-status
            # enrichment carry the field through (and a benign drop only in the rare race where
            # the status advanced past our read — acceptable for a best-effort title).
            current = load_task_result(drive_root, task_id) or {}
            status = str(current.get("status") or "") or STATUS_RUNNING
            write_task_result(drive_root, task_id, status, suggested_name=name)
            # A cosmetic namer may settle concurrently with or after the ordinary
            # post-task worker.  The shared refresh/checkpoint critical section
            # linearizes both cases without marking an unfinished phase complete.
            _refresh_root_cost_after_naming(drive_root, task_id)
            if broadcast is not None:
                try:
                    broadcast({"type": "task_named", "task_id": task_id, "suggested_name": name})
                except Exception:
                    log.debug("task_named broadcast failed for %s", task_id, exc_info=True)
        except Exception:
            log.debug("proactive namer failed for %s", task_id, exc_info=True)

    try:
        threading.Thread(target=_work, name=f"namer-{task_id}", daemon=True).start()
    except Exception:
        log.debug("proactive namer thread spawn failed for %s", task_id, exc_info=True)


def admission_names(body: Dict[str, Any], description: str) -> tuple:
    """The run's owner-facing name at admission: ``(title, suggested_name)``.

    A caller-supplied title is AUTHORSHIP — it fills both slots, exactly as a
    chat turn promoted into a task does. Without one, the request's first line is
    DERIVED for display only: it fills ``suggested_name`` (what the live card,
    history replay and the Project lifecycle row read) and leaves ``title``
    empty, so a truncated prompt never outranks a real name coined later. Lexical
    only — markdown is stripped before the first line is taken, then the shared
    title cleaner caps it at the project-name length (P5: no model call, and
    therefore no benchmark-visible cost for a scripted run).
    """
    from ouroboros.projects_registry import PROJECT_NAME_MAX
    from ouroboros.utils import strip_markdown

    explicit = clean_model_title(strip_markdown(str(body.get("title") or "")), max_len=PROJECT_NAME_MAX)
    if explicit:
        return explicit, explicit
    return "", clean_model_title(strip_markdown(description), max_len=PROJECT_NAME_MAX)
