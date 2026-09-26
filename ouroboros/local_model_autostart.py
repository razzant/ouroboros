"""Helpers for starting the local model server from app startup."""

from __future__ import annotations

import logging

log = logging.getLogger(__name__)


def auto_start_local_model(settings: dict) -> None:
    """Start local model in background after runtime preflight."""
    try:
        from ouroboros.local_model import get_manager, _get_runtime_hint

        mgr = get_manager()
        if mgr.is_running:
            return

        source = str(settings.get("LOCAL_MODEL_SOURCE", "")).strip()
        filename = str(settings.get("LOCAL_MODEL_FILENAME", "")).strip()
        if not source:
            log.error(
                "USE_LOCAL_* routing is enabled but LOCAL_MODEL_SOURCE is empty; "
                "the local model cannot start. Set LOCAL_MODEL_SOURCE to a model "
                "path/URL, or set USE_LOCAL_MAIN / USE_LOCAL_FALLBACK to false."
            )
            return
        port = int(settings.get("LOCAL_MODEL_PORT", 8766))
        n_gpu_layers = int(settings.get("LOCAL_MODEL_N_GPU_LAYERS", 0))
        n_ctx = int(settings.get("LOCAL_MODEL_CONTEXT_LENGTH", 16384))
        chat_format = str(settings.get("LOCAL_MODEL_CHAT_FORMAT", "")).strip()

        if not mgr.check_runtime():
            hint = _get_runtime_hint()
            log.warning(
                "Local model auto-start skipped: llama-cpp-python is not installed. "
                "Install with: %s",
                hint,
            )
            mgr._status = "error"
            mgr._error = (
                f"llama-cpp-python is not installed. Install with: {hint}"
            )
            return

        log.info("Auto-starting local model: %s / %s", source, filename)
        model_path = mgr.download_model(source, filename)
        mgr.start_server(
            model_path,
            port=port,
            n_gpu_layers=n_gpu_layers,
            n_ctx=n_ctx,
            chat_format=chat_format,
            source=source, filename=filename,
        )
        log.info("Local model auto-started successfully")
    except Exception as exc:
        log.warning("Local model auto-start failed: %s", exc)
