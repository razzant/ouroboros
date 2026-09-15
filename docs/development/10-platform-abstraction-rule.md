# Platform Abstraction Rule

This chapter owns the platform seam: which modules, calls and signals must go through `platform_layer.py`, the one documented function-local exception, and the shared atomic state-file helpers whose byte-exactness keeps a manifest or a hashed receipt identical across operating systems. It exists because the alternative is platform-conditional code scattered through the runtime, where an AST guard cannot see it.

Platform-specific code goes through `ouroboros/platform_layer.py`: platform
modules (`fcntl`, `resource`, `grp`, `pwd`, `msvcrt`, `winreg`,
`ctypes.windll`), direct `os.kill`/`os.killpg`/`os.setsid`/`os.getpgid`,
`signal.SIGKILL`/`signal.SIGTERM`, and platform-conditional subprocess flags
(use `subprocess_new_group_kwargs()` / `subprocess_hidden_kwargs()`). Use
`pathlib.Path` for filesystem paths, never string concatenation with
hardcoded separators. One documented exception: a function-local import of a
platform module under an explicit `sys.platform` guard is allowed outside
the layer (e.g. the guarded `resource` import in
`ouroboros/extension_process_runner.py`) — the AST guard inspects only
top-level imports and does not see function-local ones, so review must.

Enforcement: `tests/test_platform_guard.py` scans `ouroboros/`,
`supervisor/`, and `server.py` for top-level platform imports, the four `os`
calls, and the two signals; `launcher.py` and subprocess flag patterns are
deliberately not scanned — code review and the `cross_platform` checklist
item cover them — and the CI matrix runs tests on Ubuntu, Windows, and
macOS. New platform behavior: add the cross-platform wrapper to
`platform_layer.py`, use it in callers, and add platform-conditional tests
when behavior differs across OSes.

### Shared state-file helpers

Durable state files use the SSOT helpers in `ouroboros/utils.py`:
`atomic_write_json` / `read_json_dict`, and `write_text_atomic` /
`write_bytes_atomic` sharing one atomic full-overwrite seam (temp-sibling +
`os.replace`, permission bits preserved) — a crash leaves the old complete
file intact, and appends are intentionally NOT atomic (a separate contract).
Both are BYTE-EXACT on every platform: the text variant is the bytes variant
plus a UTF-8 encode, so the newlines the caller wrote are the newlines on disk
and no platform translation rewrites a manifest, a hashed receipt or an LF
source file.
Prefer these over bare `Path.write_text`/`write_bytes` for full-file
overwrites; lockfiles go through
`platform_layer.acquire_exclusive_file_lock` /
`release_exclusive_file_lock`. Narrow exceptions: `supervisor/state.py`
keeps `atomic_write_text` for its mirrored state writes, and
`ouroboros/config.py` keeps its settings-file lock because the settings path
is bootstrapped before broader runtime helpers may depend on settings state.
Enforcement: `tests/test_atomic_write_v639.py`; the prefer-the-helper rule is
review-only.

