"""Dynamic extension discovery and typed extension/MCP dispatch."""

from __future__ import annotations

import asyncio
import inspect
import pathlib
import threading
from typing import Any, Dict, Optional

from ouroboros.tools.tool_context import ToolContext
from ouroboros.tools.tool_result import (
    ToolResult,
    ToolStatus,
    _compose_execute_result,
    _structured_failure,
)


def _extension_dispatch_candidate(
    ctx: ToolContext,
    name: str,
) -> tuple[Optional[Dict[str, Any]], bool]:
    """Return a live descriptor or a host-attested unavailable marker."""
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
    try:
        ext_tool = _ext_get_tool(name)
        meta = getattr(ctx, "task_metadata", {})
        budget_root = meta.get("budget_drive_root") if isinstance(meta, dict) else ""
        capability_root = pathlib.Path(
            budget_root
            or getattr(ctx, "budget_drive_root", "")
            or getattr(ctx, "drive_root", "")
            or "."
        ).resolve(strict=False)
        if ext_tool and not _ext_is_live(
            str(ext_tool.get("skill") or ""),
            capability_root,
            repo_path=str(ext_tool.get("skills_repo_path") or "") or None,
        ):
            return None, True
        return ext_tool, False
    except Exception:
        return None, False


def _dispatch_mcp_tool_result(
    ctx: Any,
    name: str,
    args: Dict[str, Any],
) -> ToolResult:
    """Run one MCP tool while preserving provider-owned result facts."""
    from ouroboros.safety import check_safety as _mcp_check_safety

    is_safe, safety_msg = _mcp_check_safety(
        name,
        args,
        messages=getattr(ctx, "messages", None),
        ctx=ctx,
    )
    if not is_safe:
        return ToolResult(status="blocked", code="SAFETY_VIOLATION", text=safety_msg)
    try:
        from ouroboros.mcp_client import _call_mcp_tool_result as _mcp_call

        result = _mcp_call(name, args or {})
    except Exception as exc:
        text = f"⚠️ TOOL_ERROR ({name}): {exc}"
        return ToolResult(status="error", code="TOOL_ERROR", text=text)
    if not safety_msg:
        return result
    text = _compose_execute_result(result.text, "", safety_msg)
    meta = {**dict(result.meta), "safety_warning": True}
    if result.code == "OK":
        return ToolResult(status="ok", code="SAFETY_WARNING", text=text, meta=meta)
    return ToolResult(status=result.status, code=result.code, text=text, meta=meta)


def _extension_result(
    status: ToolStatus,
    code: str,
    text: str,
    *,
    safety_warning: bool = False,
    timeout_sec: int | None = None,
) -> ToolResult:
    meta: Dict[str, Any] = {"dynamic_provider": True}
    if safety_warning:
        meta["safety_warning"] = True
    if timeout_sec is not None:
        meta["timeout_sec"] = timeout_sec
    return ToolResult(status=status, code=code, text=text, meta=meta)


def _extension_completion(result: str, safety_msg: str) -> ToolResult:
    """Type one completed extension body, reading its own failure self-report.

    The dispatcher used to declare success without looking at the body, so a
    skill that answered honestly with ``{"ok": false}`` was recorded as a clean
    call (measured on the v6.81.1 OSWorld run: 329 such calls, including HTTP
    500s from every screenshot after the guest control server died). The
    structured check is the adapter's, so there is exactly one implementation of
    what a self-reported failure is."""
    reported_failure = _structured_failure(result)
    if safety_msg:
        text = f"{safety_msg}\n\n---\n{result}"
        return _extension_result(
            "error" if reported_failure else "ok",
            "TOOL_REPORTED_FAILURE" if reported_failure else "SAFETY_WARNING",
            text,
            safety_warning=True,
        )
    if reported_failure:
        return _extension_result("error", "TOOL_REPORTED_FAILURE", result)
    return _extension_result("ok", "OK", result)


def _dispatch_extension_tool_result(
    ctx: Any,
    name: str,
    ext_tool: Dict[str, Any],
    args: Optional[Dict[str, Any]],
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
    meta = getattr(ctx, "task_metadata", {})
    capability_root = pathlib.Path(
        (meta.get("budget_drive_root") if isinstance(meta, dict) else "")
        or getattr(ctx, "budget_drive_root", "")
        or getattr(ctx, "drive_root", "")
        or "."
    ).resolve(strict=False)
    if skill_name and callable(_ext_is_live) and not _ext_is_live(skill_name, capability_root, repo_path=repo_path):
        if callable(_ext_unload):
            _ext_unload(skill_name)
        text = f"⚠️ TOOL_ERROR ({name}): extension {skill_name!r} is not allowed to dispatch right now."
        return _extension_result("unavailable", "EXTENSION_UNAVAILABLE", text)

    from ouroboros.safety import check_safety as _ext_check_safety

    _ext_safe, _ext_safety_msg = _ext_check_safety(
        name,
        call_args,
        messages=getattr(ctx, "messages", None),
        ctx=ctx,
    )
    if not _ext_safe:
        return _extension_result("blocked", "SAFETY_VIOLATION", _ext_safety_msg)

    if ext_tool.get("out_of_process"):
        try:
            from ouroboros.extension_process_runner import (
                ExtensionProcessError,
                dispatch_extension_tool_subprocess,
            )
        except Exception as exc:
            text = f"⚠️ TOOL_ERROR ({name}): extension child process failed: {type(exc).__name__}: {exc}"
            return _extension_result("error", "EXTENSION_ERROR", text)
        try:
            result_str = dispatch_extension_tool_subprocess(ext_tool, ctx, call_args)
        except Exception as exc:
            text = f"⚠️ TOOL_ERROR ({name}): extension child process failed: {type(exc).__name__}: {exc}"
            timed_out = (
                isinstance(exc, ExtensionProcessError)
                and exc.failure_kind == "timeout"
            )
            return _extension_result(
                "timeout" if timed_out else "error",
                "EXTENSION_TIMEOUT" if timed_out else "EXTENSION_ERROR",
                text,
                timeout_sec=max(1, int(ext_tool.get("timeout_sec") or 60)) if timed_out else None,
            )
        return _extension_completion(result_str, _ext_safety_msg)

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
        text = f"⚠️ TOOL_ERROR ({name}): model-cost disclosure failed: {type(exc).__name__}: {exc}"
        return _extension_result("error", "EXTENSION_ERROR", text)
    try:
        # ctx calling-convention from the descriptor (decided on the RAW handler
        # at register time); the runtime wrapper is (*args, **kwargs) so inspecting
        # it here would always force a ctx-first call. Fall back to inspecting the
        # unwrapped handler for any tool registered before this flag existed.
        from ouroboros.extension_process_runner import _handler_wants_ctx

        _wants = ext_tool.get("wants_ctx")
        if _wants is None:
            _wants = _handler_wants_ctx(inspect.unwrap(handler))
        if _wants:
            result = handler(ctx, **call_args)
        else:
            result = handler(**call_args)
    except Exception as exc:
        text = f"⚠️ TOOL_ERROR ({name}): extension tool failed: {type(exc).__name__}: {exc}"
        return _extension_result("error", "EXTENSION_ERROR", text)

    if inspect.iscoroutine(result):
        box: Dict[str, Any] = {}
        timeout = max(1, int(ext_tool.get("timeout_sec") or 60))

        def _runner() -> None:
            try:
                async def _bounded():
                    task = asyncio.create_task(result)
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
            except Exception as exc:
                box["error"] = exc

        thread = threading.Thread(
            target=_runner,
            name=f"ext-tool-{name}-async",
            daemon=True,
        )
        thread.start()
        thread.join(timeout=timeout + 2)
        if thread.is_alive():
            text = f"⚠️ TOOL_ERROR ({name}): extension async handler failed: TimeoutError: handler exceeded timeout"
            return _extension_result(
                "timeout",
                "EXTENSION_TIMEOUT",
                text,
                timeout_sec=timeout,
            )
        if box.get("host_timeout"):
            text = f"⚠️ TOOL_ERROR ({name}): extension async handler failed: TimeoutError: "
            return _extension_result(
                "timeout",
                "EXTENSION_TIMEOUT",
                text,
                timeout_sec=timeout,
            )
        if "error" in box:
            exc = box["error"]
            text = f"⚠️ TOOL_ERROR ({name}): extension async handler failed: {type(exc).__name__}: {exc}"
            return _extension_result("error", "EXTENSION_ERROR", text)
        result = box.get("value", "")

    result_str = result if isinstance(result, str) else str(result)
    return _extension_completion(result_str, _ext_safety_msg)


def dispatch_extension_tool(
    ctx: Any,
    name: str,
    ext_tool: Dict[str, Any],
    args: Optional[Dict[str, Any]],
) -> str:
    """Compatibility facade returning the exact model-facing text."""
    result = _dispatch_extension_tool_result(ctx, name, ext_tool, args)
    return result.text if isinstance(result, ToolResult) else result
