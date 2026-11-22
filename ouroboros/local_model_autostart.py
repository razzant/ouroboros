"""Helpers for starting the local model server from app startup."""

from __future__ import annotations

import logging

from ouroboros import local_model

log = logging.getLogger(__name__)


def auto_start_local_model(settings: dict, *, grant: str | None = None) -> None:
    """Start local model in background after runtime preflight.

    ``grant`` is boot's receipt for this Start (``server_runtime.local_model_boot_decision``): once a Stop has revoked
    it, the manager refuses the Start and nothing is launched."""
    try:
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
        n_ctx = int(local_model.local_model_settings(settings)["LOCAL_MODEL_CONTEXT_LENGTH"])
        chat_format = str(settings.get("LOCAL_MODEL_CHAT_FORMAT", "")).strip()

        local_model.get_manager().start_model(source, filename, port=port, n_gpu_layers=n_gpu_layers,
                                              n_ctx=n_ctx, chat_format=chat_format, grant=grant)
        log.info("Local model auto-start launched the server for %s / %s", source, filename)
    except local_model.LocalModelStartConflict:
        return
    except local_model.LocalRuntimeMissing as exc:
        hint = f"\nManual install: {exc.hint}" if exc.hint else ""
        log.warning("Local model auto-start skipped. %s%s", exc, hint)
    except Exception as exc:
        log.warning("Local model auto-start failed: %s", exc)
