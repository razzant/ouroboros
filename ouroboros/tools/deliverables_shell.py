"""Path helpers shared by the shell write checks and the output audit."""

from __future__ import annotations

import pathlib

from ouroboros.workspace_executor import executor_ref_from_ctx, map_backend_path_lexical


def lexical_user_files_block_reason(candidate: pathlib.Path) -> str:
    """Classify actual owner stores, not ordinary project path components."""
    from ouroboros.credential_shapes import user_files_mutation_shape_reason
    from ouroboros.config import get_runtime_mode
    from ouroboros.runtime_mode_policy import mode_has_unrestricted_agency

    if mode_has_unrestricted_agency(get_runtime_mode()):
        return ""
    return user_files_mutation_shape_reason(candidate, pathlib.Path.home())


def _command_path(ctx, work_dir: pathlib.Path, token: str) -> pathlib.Path | None:
    try:
        raw = pathlib.Path(str(token or "")).expanduser()
        if raw.is_absolute():
            executor = executor_ref_from_ctx(ctx)
            if executor is not None:
                mapped = map_backend_path_lexical(executor, str(token))
                if mapped is not None:
                    return mapped
            return raw
        return work_dir / raw
    except (OSError, RuntimeError, TypeError, ValueError):
        return None
