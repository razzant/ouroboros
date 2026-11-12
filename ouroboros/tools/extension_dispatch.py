"""Dynamic extension discovery and typed extension/MCP dispatch."""

from __future__ import annotations

import asyncio
import contextlib
import inspect
import pathlib
import threading
import contextvars
from typing import Any, Dict, Optional

from ouroboros.tools.tool_context import ToolContext
from ouroboros.tools.tool_result import (
    ToolResult,
    ToolStatus,
    _compose_execute_result_result,
    _replace_tool_result,
    _structured_failure,
    safety_refusal,
)


def _extension_capability_root(ctx: ToolContext) -> pathlib.Path:
    meta = getattr(ctx, "task_metadata", {})
    budget_root = meta.get("budget_drive_root") if isinstance(meta, dict) else ""
    return pathlib.Path(
        budget_root
        or getattr(ctx, "budget_drive_root", "")
        or getattr(ctx, "drive_root", "")
        or "."
    ).resolve(strict=False)


def _adopt_extension_generation(ctx: ToolContext) -> Dict[str, Any]:
    """Adopt the published extension generation from the same task data root."""
    from ouroboros.config import get_skills_repo_path, load_settings
    from ouroboros.extension_reconcile_queue import adopt_published_extension_generation

    root = _extension_capability_root(ctx)
    return adopt_published_extension_generation(
        root, load_settings, repo_path=get_skills_repo_path() or None
    )


def _descriptor_generation(tool: Dict[str, Any]) -> str:
    """The publication a registered descriptor belongs to: its own stamp, else its owner bundle's live digest.

    Call it inside the registry lock hold that read ``tool`` (``get_tool`` stamps
    the same way), so the value is that observation's. A reload mints a fresh
    digest for every bundle it loads: equal values saw the same registration.
    """
    from ouroboros.extension_registry_state import extension_generation_digest

    return str(tool.get("extension_generation") or "") or extension_generation_digest(str(tool.get("skill") or ""))


def _extension_dispatch_candidate(
    ctx: ToolContext,
    name: str,
) -> tuple[Optional[Dict[str, Any]], bool]:
    """Return a live descriptor or a host-attested unavailable marker.

    One liveness observation answers an already-live surface. A dispatch miss
    or a descriptor that is no longer live runs the bounded worker adoption
    (one ``reload_all`` per distinct published generation). A stale
    descriptor is observed again only when the registry now holds another
    generation of it, whichever caller adopted, so a skill that is simply
    disabled keeps its single observation and its refusal.
    """
    try:
        from ouroboros.extension_loader import (
            get_tool as _ext_get_tool,
            is_extension_live as _ext_is_live,
            parse_extension_surface_name as _ext_parse_name,
        )
    except Exception:
        return None, False
    if not _ext_parse_name(name):
        return None, False
    capability_root = _extension_capability_root(ctx)

    def live(tool: Dict[str, Any]) -> bool:
        return bool(_ext_is_live(
            str(tool.get("skill") or ""),
            capability_root,
            repo_path=str(tool.get("skills_repo_path") or "") or None,
        ))

    try:
        # ABI-9 atomic legacy fallback: get_tool pre-stamps an unstamped
        # descriptor with the generation digest read in the SAME lock hold
        # (registry_state.get_tool_stamped), so the dispatch stamp can never
        # name a generation published between two separate registry reads.
        ext_tool = _ext_get_tool(name)
        if ext_tool is not None and live(ext_tool):
            return ext_tool, False
        stale = ext_tool is not None
        # The natural staleness points (W3B-F1, #2933): no surface here (a
        # skill enabled after this worker spawned) or a descriptor retained
        # from a previous published generation (an update hid every connector
        # tool). Both take the same bounded adopt, so a surface no publication
        # explains (a hallucinated name) costs one small read, never a reload.
        try:
            _adopt_extension_generation(ctx)
        except Exception:
            pass
        fresh = _ext_get_tool(name)
        # ``get_tool`` stamped both reads in their own lock holds: equal stamps mean no caller replaced it.
        if stale and fresh is not None and str(fresh.get("extension_generation") or "") == str(
                ext_tool.get("extension_generation") or ""):
            return None, True
        if fresh is not None and not live(fresh):
            return None, True
        return fresh, False
    except Exception:
        return None, False


# One model tool call's extension resolution, bound by the caller that chose
# the call's outer timeout (``extension_call_scope``). The slot lives in the
# call's own context, which the loop copies into the call's worker thread, so
# parallel calls of the same surface never see each other's binding and a
# binding never outlives its call.
_CALL_RESOLUTION: contextvars.ContextVar[Optional[Dict[str, Any]]] = contextvars.ContextVar(
    "extension_call_resolution", default=None)


@contextlib.contextmanager
def extension_call_scope():
    """Scope one tool call: the descriptor whose timeout was chosen is the one dispatched."""
    token = _CALL_RESOLUTION.set({})
    try:
        yield
    finally:
        _CALL_RESOLUTION.reset(token)


def timeout_dispatch_candidate(
    ctx: ToolContext, name: str,
) -> tuple[Optional[Dict[str, Any]], bool]:
    """Resolve before the outer timeout is chosen, binding the result to this call."""
    resolution = _extension_dispatch_candidate(ctx, name)
    slot = _CALL_RESOLUTION.get()
    if slot is not None:
        slot["bound"] = (ctx, name, resolution)
    return resolution


def bound_dispatch_candidate(
    ctx: ToolContext, name: str, *, consume: bool = True,
) -> tuple[Optional[Dict[str, Any]], bool]:
    """Read this call's timeout-time resolution; the dispatcher resolves fresh without one.

    By default the dispatcher takes the binding once; a pre-dispatch observation
    uses ``consume=False`` without spending it or resolving an absent binding. Both require the same context and name. The
    physical dispatcher still observes liveness itself, so a revocation that
    lands after the timeout was chosen refuses before the handler runs.
    """
    slot = _CALL_RESOLUTION.get()
    bound = (slot.pop("bound", None) if consume else slot.get("bound")) if slot is not None else None
    if bound is not None and bound[0] is ctx and bound[1] == name:
        return bound[2]
    return _extension_dispatch_candidate(ctx, name) if consume else (None, False)


def _dispatch_mcp_tool_result(
    ctx: Any,
    name: str,
    args: Dict[str, Any],
) -> ToolResult:
    """Run one MCP tool while preserving provider-owned result facts."""
    from ouroboros.safety import check_safety as _mcp_check_safety
    from ouroboros.owner_pause import admit_tool_dispatch, OwnerPauseRefused
    from ouroboros.llm_attempt import PhysicalDispatchInterrupted
    from ouroboros.model_wait import ModelWaitInterrupted
    from ouroboros.usage_accounting import UsageAccountingError

    is_safe, safety_msg = _mcp_check_safety(
        name,
        args,
        messages=getattr(ctx, "messages", None),
        ctx=ctx,
    )
    if not is_safe:
        return ToolResult(*safety_refusal(safety_msg), text=safety_msg)
    try:
        from ouroboros.mcp_client import _call_mcp_tool_result as _mcp_call

        admit_tool_dispatch(ctx)
        result = _mcp_call(name, args or {}, ctx)  # the calling task owns returned images
    except (PhysicalDispatchInterrupted, OwnerPauseRefused, ModelWaitInterrupted, UsageAccountingError):
        raise
    except Exception as exc:
        text = f"⚠️ TOOL_ERROR ({name}): {exc}"
        return ToolResult(status="error", code="TOOL_ERROR", text=text)
    if not safety_msg:
        return result
    return _replace_tool_result(
        _compose_execute_result_result(name, result, "", safety_msg),
        meta_updates={"safety_warning": True},
    )


def _extension_result(
    status: ToolStatus,
    code: str,
    text: str,
    *,
    safety_warning: bool = False,
    timeout_sec: int | None = None,
    dispatched: bool = False,
) -> ToolResult:
    meta: Dict[str, Any] = {"dynamic_provider": True}
    if safety_warning:
        meta["safety_warning"] = True
    if timeout_sec is not None:
        meta["timeout_sec"] = timeout_sec
    if dispatched:
        # POSITIVE dispatch fact (ABI-9): the handler / child process was
        # actually invoked — the generation stamp keys on this, never on an
        # error-code exclusion list (pre-handler failures share codes with
        # post-handler ones).
        meta["physical_dispatch"] = True
    return ToolResult(status=status, code=code, text=text, meta=meta)


def _extension_completion(result: str, safety_msg: str, *, handoff: Optional[Dict[str, Any]] = None) -> ToolResult:
    """Type one completed extension body, reading its own failure self-report.

    The dispatcher used to declare success without looking at the body, so a
    skill that answered honestly with ``{"ok": false}`` was recorded as a clean
    call (measured on the v6.81.1 OSWorld run: 329 such calls, including HTTP
    500s from every screenshot after the guest control server died). The
    structured check is the adapter's, so there is exactly one implementation of
    what a self-reported failure is."""
    reported_failure = _structured_failure(result)
    base = _extension_result(
        "error" if reported_failure else "ok",
        "TOOL_REPORTED_FAILURE" if reported_failure else "OK",
        result,
        dispatched=True,
    )
    if safety_msg:
        base = _replace_tool_result(
            _compose_execute_result_result("", base, "", safety_msg),
            meta_updates={"safety_warning": True},
        )
    if handoff is not None:
        # Host-owned invocation sidechannel, never extension text/metadata.
        # Only the supported local call ended; independent effects keep custody.
        handoff["local_extension_returned"] = True
    return base


def _generation_digest_for(ext_tool: Dict[str, Any]) -> str:
    """Published-generation digest this descriptor dispatches against (ABI-9).

    Provenance READ only (the Ф3.2 seam): the digest names the exact published
    registration generation — taken from the per-surface stamp minted at
    publication. The loader's ``get_tool`` pre-stamps a legacy descriptor via
    the ATOMIC combined registry read (``registry_state.get_tool_stamped``:
    descriptor and digest under one lock hold), so the separate
    registry-reader fallback below serves only descriptors handed in directly
    (tests, legacy callers) and can no longer pair an old handler with a
    digest published after the descriptor was taken. Dispatch never validates
    or gates on it."""
    digest = str(ext_tool.get("extension_generation") or "")
    if digest:
        return digest
    try:
        from ouroboros.extension_loader import (
            extension_generation_digest as _ext_generation_digest,
        )

        return str(_ext_generation_digest(str(ext_tool.get("skill") or "")) or "")
    except Exception:
        return ""


def _dispatch_extension_tool_result(
    ctx: Any,
    name: str,
    ext_tool: Dict[str, Any],
    args: Optional[Dict[str, Any]],
    *, handoff: Optional[Dict[str, Any]] = None,
) -> ToolResult:
    """Dispatch once, stamping ABI-9 generation provenance on physical calls.

    The model-facing text and the typed status/code are exactly the inner
    dispatcher's; the ``extension_generation`` and known ``content_hash`` meta facts appear
    only on outcomes of a physical dispatch attempt, which the loop projects into
    the tools.jsonl ``tool_result_meta`` provenance record.

    The digest is snapshotted BEFORE the call (a registry-fallback read taken
    after the handler could name a generation published DURING it), and the
    stamp keys on the inner dispatcher's positive ``physical_dispatch`` meta
    fact — set only once the handler / child process is actually invoked —
    so a pre-handler refusal or failure (liveness, safety, runner import,
    disclosure gate, calling-convention resolution) is never stamped."""
    digest = _generation_digest_for(ext_tool)
    content_hash = str(ext_tool.get("content_hash") or "")
    result = _dispatch_extension_tool_untagged(
        ctx, name, ext_tool, args, **({"handoff": handoff} if handoff is not None else {}))
    if (
        not isinstance(result, ToolResult)
        or not result.meta.get("physical_dispatch")
        or not digest
    ):
        return result
    return _replace_tool_result(
        result,
        meta_updates={"extension_generation": digest,
              **({"content_hash": content_hash} if content_hash else {})},
    )


def _replaced_since_bound(name: str, ext_tool: Dict[str, Any]) -> bool:
    """Whether an out-of-process descriptor's payload is no longer the live one.

    Its child loads the live payload, which the liveness read just matched to the live publication, so the bound
    descriptor's declaration describes that code only while both name one payload. In process, the bound handler
    itself runs and needs no such check."""
    try:
        from ouroboros.extension_loader import get_tool as _ext_get_tool

        live = _ext_get_tool(name) or {}
    except Exception:
        return True
    return str(live.get("content_hash") or "") != str(ext_tool.get("content_hash") or "")


def _dispatch_extension_tool_untagged(
    ctx: Any,
    name: str,
    ext_tool: Dict[str, Any],
    args: Optional[Dict[str, Any]],
    *, handoff: Optional[Dict[str, Any]] = None,
) -> ToolResult:
    """Dispatch once while retaining host-owned extension outcome facts."""
    try:
        from ouroboros.extension_loader import (
            is_extension_live as _ext_is_live,
        )
        from ouroboros.extension_loader import (
            unload_extension as _ext_unload,
        )
    except Exception:
        _ext_is_live = None
        _ext_unload = None

    call_args = args or {}
    skill_name = str(ext_tool.get("skill") or "")
    repo_path = str(ext_tool.get("skills_repo_path") or "") or None
    capability_root = _extension_capability_root(ctx)
    # Fresh at the physical dispatch: a revocation after resolution refuses here.
    if skill_name and callable(_ext_is_live) and not _ext_is_live(skill_name, capability_root, repo_path=repo_path):
        if callable(_ext_unload):
            _ext_unload(skill_name)
        text = f"⚠️ TOOL_ERROR ({name}): extension {skill_name!r} is not allowed to dispatch right now."
        return _extension_result("unavailable", "EXTENSION_UNAVAILABLE", text)
    if ext_tool.get("out_of_process") and _replaced_since_bound(name, ext_tool):
        text = (f"⚠️ TOOL_ERROR ({name}): extension {skill_name!r} was replaced after this call was bound; "
                "its child would run the replacement, not the code its declaration describes. Call it again.")
        return _extension_result("unavailable", "EXTENSION_UNAVAILABLE", text)

    from ouroboros.safety import check_safety as _ext_check_safety
    _ext_safe, _ext_safety_msg = _ext_check_safety(
        name,
        call_args,
        messages=getattr(ctx, "messages", None),
        ctx=ctx,
        ext_tool=ext_tool,
    )
    if not _ext_safe:
        return _extension_result(*safety_refusal(_ext_safety_msg), _ext_safety_msg)

    from ouroboros.owner_pause import admit_tool_dispatch, run_tool_handler, submit_async_operation, OwnerPauseRefused
    from ouroboros.llm_attempt import PhysicalDispatchInterrupted
    from ouroboros.model_wait import ModelWaitInterrupted
    from ouroboros.usage_accounting import UsageAccountingError
    controls = (PhysicalDispatchInterrupted, OwnerPauseRefused, ModelWaitInterrupted, UsageAccountingError)
    from ouroboros.deployment_extensions import failure_words

    if ext_tool.get("out_of_process"):
        try:
            from ouroboros.extension_process_runner import (
                PAYLOAD_REPLACED,
                ExtensionProcessError,
                dispatch_extension_tool_subprocess,
                extension_child_was_spawned,
            )
        except Exception as exc:
            text = f"⚠️ TOOL_ERROR ({name}): extension child process failed: {failure_words(skill_name, exc)}"
            return _extension_result("error", "EXTENSION_ERROR", text)
        try:
            admit_tool_dispatch(ctx)
            result_str = dispatch_extension_tool_subprocess(ext_tool, ctx, call_args)
        except controls:
            raise
        except Exception as exc:
            text = f"⚠️ TOOL_ERROR ({name}): extension child process failed: {failure_words(skill_name, exc)}"
            kind = exc.failure_kind if isinstance(exc, ExtensionProcessError) else "error"
            timed_out = kind == "timeout"
            # The child refused a payload other than the bound one before any handler ran (typed, retryable).
            status, code = {"timeout": ("timeout", "EXTENSION_TIMEOUT"),
                            PAYLOAD_REPLACED: ("unavailable", "EXTENSION_UNAVAILABLE")}.get(kind, ("error", "EXTENSION_ERROR"))
            return _extension_result(
                status,
                code,
                text,
                timeout_sec=max(1, int(ext_tool.get("timeout_sec") or 60)) if timed_out else None,
                # POSITIVE spawn fact from the process runner (ABI-9): a
                # pre-Popen failure (resolve/load/env/payload staging) never
                # stamps physical_dispatch — the child never existed.
                dispatched=extension_child_was_spawned(exc),
            )
        return _extension_completion(result_str, _ext_safety_msg, handoff=handoff)

    handler = ext_tool["handler"]
    try:
        from ouroboros.extension_process_runner import disclose_inprocess_extension_dispatch

        disclose_inprocess_extension_dispatch(
            ext_tool,
            drive_root=capability_root,
            surface_kind="tool",
            surface=str(ext_tool.get("name") or name),
            ctx=ctx,
        )
    except Exception as exc:
        text = f"⚠️ TOOL_ERROR ({name}): model-cost disclosure failed: {failure_words(skill_name, exc)}"
        return _extension_result("error", "EXTENSION_ERROR", text)
    try:
        # ctx calling-convention from the descriptor (decided on the RAW handler
        # at register time); the runtime wrapper is (*args, **kwargs) so inspecting
        # it here would always force a ctx-first call. Fall back to inspecting the
        # unwrapped handler for any tool registered before this flag existed.
        # A failure HERE is pre-handler: same code/text, but never stamped as a
        # physical dispatch (the handler was not reached).
        from ouroboros.extension_process_runner import _handler_wants_ctx

        _wants = ext_tool.get("wants_ctx")
        if _wants is None:
            _wants = _handler_wants_ctx(inspect.unwrap(handler))
    except Exception as exc:
        text = f"⚠️ TOOL_ERROR ({name}): extension tool failed: {failure_words(skill_name, exc)}"
        return _extension_result("error", "EXTENSION_ERROR", text)
    try:
        result = run_tool_handler(ctx, handler, *((ctx,) if _wants else ()), **call_args)
    except controls:
        raise
    except Exception as exc:
        from ouroboros.store import StoreError
        if isinstance(exc, StoreError) and handoff is not None and handoff.get("not_started") is True:
            raise  # registry snapshots the shared positive no-entry fact
        text = f"⚠️ TOOL_ERROR ({name}): extension tool failed: {failure_words(skill_name, exc)}"
        return _extension_result("error", "EXTENSION_ERROR", text, dispatched=True)

    if inspect.iscoroutine(result):
        from ouroboros.tool_custody import hold_invocation
        if handoff is not None:
            handoff["settled"] = False  # constructing a coroutine did not execute it
        release_runner = hold_invocation()
        box: Dict[str, Any] = {}
        timeout = max(1, int(ext_tool.get("timeout_sec") or 60))

        def _runner() -> None:
            try:
                async def _bounded():
                    try:
                        task = submit_async_operation(ctx, lambda: result)
                    except controls:
                        result.close()
                        raise
                    done, _pending = await asyncio.wait({task}, timeout=timeout)
                    if task not in done:
                        task.cancel()
                        await asyncio.gather(task, return_exceptions=True)
                        return False, None
                    return True, task.result()

                completed, value = asyncio.run(_bounded())
                if completed:
                    box["value"] = value
                else:
                    box["host_timeout"] = True
            except BaseException as exc:
                box["error"] = exc
            finally:
                if handoff is not None:
                    handoff["settled"] = True
                release_runner()

        # Preserve the admitted task settings snapshot across the async handler
        # thread while keeping live immediate controls live in the API reader.
        task_context = contextvars.copy_context()
        thread = threading.Thread(
            target=lambda: task_context.run(_runner),
            name=f"ext-tool-{name}-async",
            daemon=True,
        )
        try:
            thread.start()
        except BaseException:
            result.close()
            if handoff is not None:
                handoff["settled"] = True
            release_runner()
            raise
        thread.join(timeout=timeout + 2)
        if thread.is_alive():
            text = f"⚠️ TOOL_ERROR ({name}): extension async handler failed: TimeoutError: handler exceeded timeout"
            return _extension_result(
                "timeout",
                "EXTENSION_TIMEOUT",
                text,
                timeout_sec=timeout,
                dispatched=True,
            )
        if box.get("host_timeout"):
            text = f"⚠️ TOOL_ERROR ({name}): extension async handler failed: TimeoutError: "
            return _extension_result(
                "timeout",
                "EXTENSION_TIMEOUT",
                text,
                timeout_sec=timeout,
                dispatched=True,
            )
        if "error" in box:
            exc = box["error"]
            if isinstance(exc, controls):
                raise exc
            text = f"⚠️ TOOL_ERROR ({name}): extension async handler failed: {failure_words(skill_name, exc)}"
            return _extension_result("error", "EXTENSION_ERROR", text, dispatched=True)
        if "value" not in box:
            return _extension_result("error", "EXTENSION_ERROR",
                f"⚠️ TOOL_ERROR ({name}): extension async runner ended without a result", dispatched=True)
        result = box["value"]

    result_str = result if isinstance(result, str) else str(result)
    return _extension_completion(result_str, _ext_safety_msg, handoff=handoff)


def dispatch_extension_tool(
    ctx: Any,
    name: str,
    ext_tool: Dict[str, Any],
    args: Optional[Dict[str, Any]],
) -> str:
    """Compatibility facade returning the exact model-facing text."""
    result = _dispatch_extension_tool_result(ctx, name, ext_tool, args)
    return result.text if isinstance(result, ToolResult) else result
