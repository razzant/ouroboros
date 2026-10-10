"""In-process extension integration for per-skill isolated Python deps."""

from __future__ import annotations

import asyncio
import pathlib
import importlib
import logging
import sys
import threading
from contextlib import asynccontextmanager, contextmanager
from types import ModuleType
from typing import Iterator, List, Sequence

from ouroboros.skill_loader import _SKILL_DIR_CACHE_NAMES


log = logging.getLogger(__name__)

_lock = threading.RLock()
class _ExecutionBarrier:
    """Shared reader/writer barrier for the in-process sys.path seam.

    A successful enter owns one lease.  Readers overlap; a writer is exclusive
    and blocks new readers once it declares intent.  Ownership is deliberately
    not associated with a thread or task, preserving the old non-reentrant
    scope contract and avoiding ContextVar inheritance across child tasks.

    A lease may also carry a ``key`` — the skill it executes.  Leases of the
    SAME key never overlap: the old exclusive lock serialized every handler,
    and a bundled consumer (the Telegram card renderer) relies on its own
    callbacks running one after another.  Independent skills still overlap;
    only the cross-skill exclusion was the #1195 F3 defect, never the per-skill
    ordering a skill author may assume.
    """

    def __init__(self) -> None:
        self.condition = threading.Condition()
        self.readers = 0
        self.writer_active = False
        self.writers_waiting = 0
        self.active_keys: set[str] = set()

    def try_enter(self, writer: bool, key: str | None = None) -> bool:
        with self.condition:
            if key is not None and key in self.active_keys:
                return False
            if writer:
                if self.writer_active or self.readers:
                    return False
                self.writer_active = True
            else:
                if self.writer_active or self.writers_waiting:
                    return False
                self.readers += 1
            if key is not None:
                self.active_keys.add(key)
            return True

    def acquire(self, blocking: bool = True, *, writer: bool = True, key: str | None = None) -> bool:
        if not blocking:
            return self.try_enter(writer, key)
        with self.condition:
            def key_free() -> bool:
                return key is None or key not in self.active_keys

            if writer:
                self.writers_waiting += 1
                try:
                    self.condition.wait_for(
                        lambda: not self.writer_active and self.readers == 0 and key_free()
                    )
                    self.writer_active = True
                finally:
                    self.writers_waiting -= 1
                    self.condition.notify_all()
            else:
                self.condition.wait_for(
                    lambda: not self.writer_active and not self.writers_waiting and key_free()
                )
                self.readers += 1
            if key is not None:
                self.active_keys.add(key)
            return True

    def release(self, *, writer: bool = True, key: str | None = None) -> None:
        with self.condition:
            if writer:
                if not self.writer_active:
                    raise RuntimeError("execution writer lease released without enter")
                self.writer_active = False
            else:
                if self.readers <= 0:
                    raise RuntimeError("execution reader lease released without enter")
                self.readers -= 1
            if key is not None:
                self.active_keys.discard(key)
            self.condition.notify_all()


_execution_lock = _ExecutionBarrier()
_injected_site_dir_refs: dict[str, int] = {}


def is_skill_cache_path(path: pathlib.Path, root: pathlib.Path) -> bool:
    try:
        rel_parts = path.relative_to(root).parts
    except ValueError:
        return False
    return any(part in _SKILL_DIR_CACHE_NAMES for part in rel_parts)


def _isolated_python_site_dirs(skill_dir: pathlib.Path) -> List[pathlib.Path]:
    env_root = pathlib.Path(skill_dir) / ".ouroboros_env" / "python"
    candidates = [
        *env_root.glob("lib/python*/site-packages"),
        env_root / "Lib" / "site-packages",
    ]
    out: List[pathlib.Path] = []
    for path in candidates:
        try:
            resolved = path.resolve()
            resolved.relative_to(pathlib.Path(skill_dir).resolve())
        except Exception:
            continue
        if resolved.is_dir() and resolved not in out:
            out.append(resolved)
    return out


def invalidate_import_caches() -> None:
    """Run ``importlib.invalidate_caches()`` under the importer-cache lock.

    CPython's ``PathFinder`` sweep snapshots ``sys.path_importer_cache`` and
    then deletes its ``None`` entries by key, so two overlapping sweeps can
    delete one key twice (``KeyError``).  Every Ouroboros-owned sweep and importer
    cache removal holds ``_lock``; plugin import, register and handlers do not.
    """
    with _lock:
        importlib.invalidate_caches()


def inject_isolated_site_dirs(skill_dir: pathlib.Path) -> List[str]:
    """Temporarily expose reviewed isolated Python deps to an extension."""

    injected: List[str] = []
    for site_dir in _isolated_python_site_dirs(pathlib.Path(skill_dir)):
        site_str = str(site_dir)
        with _lock:
            count = _injected_site_dir_refs.get(site_str)
            if count is not None:
                _injected_site_dir_refs[site_str] = count + 1
                injected.append(site_str)
                continue
            if site_str in sys.path:
                _injected_site_dir_refs[site_str] = 1
                injected.append(site_str)
                continue
            sys.path.insert(0, site_str)
            invalidate_import_caches()
            _injected_site_dir_refs[site_str] = 1
            injected.append(site_str)
    return injected


def _extend_path_candidates(candidates: List[object], value: object) -> None:
    if value is None:
        return
    if isinstance(value, (str, bytes, pathlib.Path)):
        candidates.append(value)
        return
    try:
        candidates.extend(list(value))  # type: ignore[arg-type]
        return
    except Exception:
        pass
    # importlib namespace paths recalculate from parent packages during
    # iteration. If a third-party object is temporarily inconsistent, the
    # cached path list is still enough for isolated-deps cleanup.
    cached = getattr(value, "_path", None)
    if cached is None:
        return
    try:
        candidates.extend(list(cached))
    except Exception:
        return


def _module_paths(module: ModuleType) -> List[pathlib.Path]:
    candidates: List[object] = []
    module_file = getattr(module, "__file__", None)
    if module_file:
        candidates.append(module_file)
    module_path = getattr(module, "__path__", None)
    _extend_path_candidates(candidates, module_path)
    spec = getattr(module, "__spec__", None)
    locations = getattr(spec, "submodule_search_locations", None)
    _extend_path_candidates(candidates, locations)
    out: List[pathlib.Path] = []
    for value in candidates:
        try:
            out.append(pathlib.Path(value).resolve())
        except Exception:
            continue
    return out


def _path_is_under(path: pathlib.Path, root: pathlib.Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except Exception:
        return False


def _env_root_for_site_dir(site_path: pathlib.Path) -> pathlib.Path | None:
    for parent in (site_path, *site_path.parents):
        if parent.name == ".ouroboros_env":
            return parent
    return None


def _module_names_under_site_dir_best_effort(site_path: pathlib.Path) -> List[str]:
    to_drop: set[str] = set()
    package_prefixes: set[str] = set()
    modules = list(sys.modules.items())
    for name, module in modules:
        if not name or module is None:
            continue
        try:
            paths = _module_paths(module)
        except BaseException as exc:
            log.debug("isolated deps module path scan skipped %s: %s", name, exc)
            continue
        if not any(_path_is_under(path, site_path) for path in paths):
            continue
        to_drop.add(name)
        module_path_attr = getattr(module, "__path__", None)
        module_file = getattr(module, "__file__", None)
        if module_path_attr is not None and module_file:
            try:
                if _path_is_under(pathlib.Path(module_file).resolve(), site_path):
                    package_prefixes.add(f"{name}.")
            except Exception:
                pass
    if package_prefixes:
        for name, module in modules:
            if not name or module is None:
                continue
            if any(name.startswith(prefix) for prefix in package_prefixes):
                to_drop.add(name)
    return sorted(to_drop, key=lambda value: value.count("."), reverse=True)


def _module_names_under_site_dir(site_path: pathlib.Path) -> List[str]:
    return _module_names_under_site_dir_best_effort(site_path)


def _drop_modules_under_site_dir(site_path: pathlib.Path) -> BaseException | None:
    cleanup_error = None
    try:
        module_names = _module_names_under_site_dir(site_path)
    except BaseException as exc:
        cleanup_error = exc
        module_names = _module_names_under_site_dir_best_effort(site_path)
    for name in module_names:
        try:
            sys.modules.pop(name, None)
        except BaseException as exc:
            cleanup_error = cleanup_error or exc
    return cleanup_error


def _path_matches_isolated_scope(path: pathlib.Path, site_path: pathlib.Path, env_root: pathlib.Path | None) -> bool:
    return _path_is_under(path, site_path) or bool(env_root and _path_is_under(path, env_root))


def _remove_sys_path_entries_for_scope(site_path: pathlib.Path, site_str: str) -> BaseException | None:
    cleanup_error = None
    env_root = _env_root_for_site_dir(site_path)
    for raw in list(sys.path):
        raw_str = str(raw or "")
        if not raw_str:
            continue
        remove = raw_str == site_str
        if not remove:
            try:
                remove = _path_matches_isolated_scope(pathlib.Path(raw_str).resolve(), site_path, env_root)
            except Exception:
                remove = False
        if not remove:
            continue
        try:
            while raw in sys.path:
                sys.path.remove(raw)
        except BaseException as exc:
            cleanup_error = cleanup_error or exc
    return cleanup_error


def _sys_path_leaks_site_dir(site_path: pathlib.Path, site_str: str) -> List[str]:
    leaks: List[str] = []
    env_root = _env_root_for_site_dir(site_path)
    for raw in list(sys.path):
        raw_str = str(raw or "")
        if not raw_str:
            continue
        if raw_str == site_str:
            leaks.append(raw_str)
            continue
        try:
            if _path_matches_isolated_scope(pathlib.Path(raw_str).resolve(), site_path, env_root):
                leaks.append(raw_str)
        except Exception:
            continue
    return leaks


def _drop_importer_cache_for_site_dir(site_path: pathlib.Path, site_str: str) -> None:
    env_root = _env_root_for_site_dir(site_path)
    for raw in list(sys.path_importer_cache.keys()):
        raw_str = str(raw or "")
        if not raw_str:
            continue
        if raw_str == site_str:
            sys.path_importer_cache.pop(raw, None)
            continue
        try:
            if not _path_matches_isolated_scope(pathlib.Path(raw_str).resolve(), site_path, env_root):
                continue
        except Exception:
            continue
        sys.path_importer_cache.pop(raw, None)


def release_isolated_site_dirs(site_dirs: Sequence[str]) -> None:
    anomalies: List[BaseException] = []
    hard_leaks: List[str] = []
    for raw in site_dirs:
        site_str = str(raw or "")
        if not site_str:
            continue
        with _lock:
            count = _injected_site_dir_refs.get(site_str, 0)
            if count > 1:
                _injected_site_dir_refs[site_str] = count - 1
                continue
            site_path = pathlib.Path(site_str).resolve()
            cleanup_error = _drop_modules_under_site_dir(site_path)
            try:
                path_cleanup_error = _remove_sys_path_entries_for_scope(site_path, site_str)
                cleanup_error = cleanup_error or path_cleanup_error
            except BaseException as exc:
                cleanup_error = cleanup_error or exc
            try:
                _drop_importer_cache_for_site_dir(site_path, site_str)
            except BaseException as exc:
                cleanup_error = cleanup_error or exc
            try:
                invalidate_import_caches()
            except BaseException as exc:
                cleanup_error = cleanup_error or exc
            _injected_site_dir_refs.pop(site_str, None)
            if cleanup_error is not None:
                anomalies.append(cleanup_error)
            leaks = _sys_path_leaks_site_dir(site_path, site_str)
            if leaks:
                hard_leaks.extend(leaks)
    if hard_leaks:
        raise RuntimeError(
            "isolated dependency path leak after release: "
            + ", ".join(sorted(set(hard_leaks))[:5])
        )
    if anomalies:
        log.warning(
            "isolated dependency cleanup completed with recoverable anomalies: %s",
            "; ".join(f"{type(exc).__name__}: {exc}" for exc in anomalies[:3]),
        )


def _release_site_dirs_best_effort(site_dirs: Sequence[str]) -> None:
    try:
        release_isolated_site_dirs(site_dirs)
    except BaseException as exc:
        if "path leak" in str(exc).lower():
            raise
        log.warning("isolated dependency scope cleanup failed after body success: %s", exc)


def _scope_key(skill_dir: pathlib.Path) -> str:
    """One key per skill payload; two spellings of one directory are one skill."""
    try:
        return str(pathlib.Path(skill_dir).resolve())
    except OSError:
        return str(skill_dir)


async def _acquire_execution_barrier_async(*, writer: bool, key: str | None = None) -> None:
    if writer:
        with _execution_lock.condition:
            _execution_lock.writers_waiting += 1
    try:
        while not _execution_lock.try_enter(writer, key):
            await asyncio.sleep(0.01)
        # No await between grant and the caller's try/finally: cancellation can
        # only be delivered while polling or inside the protected scope body.
    finally:
        if writer:
            with _execution_lock.condition:
                _execution_lock.writers_waiting -= 1
                _execution_lock.condition.notify_all()


@contextmanager
def isolated_site_dirs_scope(skill_dir: pathlib.Path, *, enabled: bool) -> Iterator[None]:
    """Expose reviewed deps in a local reader/writer isolation scope.

    No-deps scopes are readers and overlap ACROSS skills; the scopes of one
    skill run one at a time.  A deps-bearing scope is a writer: it excludes
    readers while the shared ``sys.path`` is changed and cleaned.  Both leases
    remain non-reentrant and last through handler waits and cleanup.
    """

    writer = bool(enabled)
    key = _scope_key(skill_dir)
    _execution_lock.acquire(writer=writer, key=key)
    site_dirs: List[str] = []
    try:
        site_dirs = inject_isolated_site_dirs(skill_dir) if enabled else []
        yield
    finally:
        try:
            _release_site_dirs_best_effort(site_dirs)
        finally:
            _execution_lock.release(writer=writer, key=key)


@asynccontextmanager
async def async_isolated_site_dirs_scope(skill_dir: pathlib.Path, *, enabled: bool) -> Iterator[None]:
    # Same reader/writer barrier as the sync scope.  Async acquisition polls
    # cooperatively so the ASGI loop never blocks on a threading.Condition.
    writer = bool(enabled)
    key = _scope_key(skill_dir)
    await _acquire_execution_barrier_async(writer=writer, key=key)
    site_dirs: List[str] = []
    try:
        site_dirs = inject_isolated_site_dirs(skill_dir) if enabled else []
        yield
    finally:
        try:
            _release_site_dirs_best_effort(site_dirs)
        finally:
            _execution_lock.release(writer=writer, key=key)
