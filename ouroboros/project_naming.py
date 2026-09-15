"""SSOT for project naming (Cluster B, v6.40).

An LLM-first short human title for a project card, with a deterministic heuristic
fallback. Shared by every path that names a project so the UI conversion and the
agent never drift:
  - ``gateway/projects.py`` turn-into-project conversion (reuses an admission name,
    or names inline as a race fallback);
  - ``ensure_project_scope`` (the agent self-creates + names a project);
  - ``admission_names`` (headless runs and chat promotion, no model call).
A direct conversation turn is never named: it renders as an activity block.

Doctrine:
  - P5 LLM-first: the model COINS the name; post-processing is purely lexical
    (first line, strip quotes, cap length) — never a keyword/regex semantic gate.
  - #4 self-DoS: the provider call goes through the per-model concurrency slot.
  - Fail-soft: any failure (no creds / provider error / timeout / empty output)
    returns the heuristic fallback. Never raises.
"""

from __future__ import annotations

import logging
from dataclasses import replace
from typing import Any, Dict, Optional, Sequence

log = logging.getLogger("ouroboros.project_naming")


def _light_use_local(explicit: Optional[bool]) -> bool:
    """Resolve the light-lane local route for naming. Honor an explicit caller value;
    otherwise follow the runtime ``USE_LOCAL_LIGHT`` flag — naming runs on the LIGHT model,
    so it must route local/remote like every other light-lane caller (e.g. the safety
    check at ``ouroboros/safety.py::_resolve_safety_routing``) instead of hardcoding remote."""
    from ouroboros.config import runtime_setting

    if explicit is not None:
        return bool(explicit)
    return str(runtime_setting("USE_LOCAL_LIGHT", "") or "").lower() in ("true", "1")

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
    from ouroboros.config import runtime_setting

    from ouroboros.config import SETTINGS_DEFAULTS

    default = SETTINGS_DEFAULTS["OUROBOROS_PROJECT_NAMING_TIMEOUT_SEC"]
    try:
        return float(runtime_setting("OUROBOROS_PROJECT_NAMING_TIMEOUT_SEC", default))
    except (TypeError, ValueError):
        return float(default)


def _naming_async_timeout_sec() -> float:
    """Gateway HARD wait for the inline turn-into-project name. SSOT: config
    SETTINGS_DEFAULTS (no duplicated literal — the default IS the SSOT value)."""
    from ouroboros.config import runtime_setting

    from ouroboros.config import SETTINGS_DEFAULTS

    default = SETTINGS_DEFAULTS["OUROBOROS_PROJECT_NAMING_ASYNC_TIMEOUT_SEC"]
    try:
        return float(runtime_setting("OUROBOROS_PROJECT_NAMING_ASYNC_TIMEOUT_SEC", default))
    except (TypeError, ValueError):
        return float(default)


def _project_naming_usage_scope(drive_root: Optional[Any], task_id: str):
    """Bind a naming send to its task tree even from a daemon/gateway thread."""
    from ouroboros.config import runtime_setting

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
        root_limit = float(runtime_setting("OUROBOROS_PER_TASK_COST_USD", "0") or 0)
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
            model_role="light",
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
