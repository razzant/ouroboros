"""Extension tool dispatch helpers for the tool registry."""

from __future__ import annotations

import asyncio
import inspect
import pathlib
import threading
from typing import Any, Dict, Optional


def dispatch_extension_tool(ctx: Any, name: str, ext_tool: Dict[str, Any], args: Optional[Dict[str, Any]]) -> str:
    """Dispatch live extension tools through the same safety gate as built-ins."""
    try:
        from ouroboros.extension_loader import (
            is_extension_live as _ext_is_live,
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
        return f"⚠️ TOOL_ERROR ({name}): extension {skill_name!r} is not allowed to dispatch right now."

    from ouroboros.safety import check_safety as _ext_check_safety
    # Warning trails the payload (#447 H1): a leading SAFETY_WARNING masked a
    # structured {"ok": false} extension answer from failure classification.
    from ouroboros.tools.result_envelope import append_note as _append_result_note

    _ext_safe, _ext_safety_msg = _ext_check_safety(
        name,
        call_args,
        messages=getattr(ctx, "messages", None),
        ctx=ctx,
    )
    if not _ext_safe:
        return _ext_safety_msg

    if ext_tool.get("out_of_process"):
        try:
            from ouroboros.extension_process_runner import dispatch_extension_tool_subprocess

            result_str = dispatch_extension_tool_subprocess(ext_tool, ctx, call_args)
        except Exception as exc:
            return f"⚠️ TOOL_ERROR ({name}): extension child process failed: {type(exc).__name__}: {exc}"
        return _append_result_note(result_str, _ext_safety_msg) if _ext_safety_msg else result_str

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
        return f"⚠️ TOOL_ERROR ({name}): model-cost disclosure failed: {type(exc).__name__}: {exc}"
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
        return f"⚠️ TOOL_ERROR ({name}): extension tool failed: {type(exc).__name__}: {exc}"

    if inspect.iscoroutine(result):
        box: Dict[str, Any] = {}
        timeout = max(1, int(ext_tool.get("timeout_sec") or 60))

        def _runner() -> None:
            try:
                async def _bounded():
                    return await asyncio.wait_for(result, timeout=timeout)

                box["value"] = asyncio.run(_bounded())
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
            return f"⚠️ TOOL_ERROR ({name}): extension async handler failed: TimeoutError: handler exceeded timeout"
        if "error" in box:
            exc = box["error"]
            return f"⚠️ TOOL_ERROR ({name}): extension async handler failed: {type(exc).__name__}: {exc}"
        result = box.get("value", "")

    result_str = result if isinstance(result, str) else str(result)
    return _append_result_note(result_str, _ext_safety_msg) if _ext_safety_msg else result_str
