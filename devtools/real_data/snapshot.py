"""A contained, coherent copy of an install's data root for the real-data stand.

The stand answers one question the unit cost contract cannot: does an ordinary owner
message still reach its model on THIS install's history while background upkeep runs?
It boots a candidate body against a copy of the owner's data. This module makes the
copy; ``sandbox`` contains the boot; ``stand`` drives it.

Take it with the app quit: on 23.10 cloning a running install's root (5.7 million files)
stalled its supervisor loop three times for 2–9 minutes, so ``make_snapshot`` refuses while
the install's pid lock names a live process. The copy is APFS clones (``clonefile``), so it
costs metadata until an arm writes. Each SQLite store is cloned again without any lock and
kept only when no commit and no in-flight write overlapped it (``_clone_store``), so the copy
is one committed state even if something writes. Nothing scans a live store and no runtime
helper is imported against the live root. Journals are not copied: a hot journal beside a
consistent copy would roll it back. macOS only (``clonefile``, Seatbelt in ``sandbox``).

Left out: the Claudexor engine data (a second daemon must never start from a copy),
process-ownership records (a copy must never adopt or signal the owner's processes),
every credential in ``settings.json`` (model slots point at a loopback stub), MCP
servers and Telegram (a copy must not reach the owner's integrations).

Size: clones share blocks with the live root; only what an arm writes takes space.
Remover: ``python -m devtools.real_data.stand --remove <dir>`` or deleting the stand
directory; nothing outside it is created.
"""

from __future__ import annotations

import ctypes
import json
import os
import pathlib
import shutil
import sqlite3
import time
from typing import Any, Dict, Iterable, List

_LIBC = ctypes.CDLL("/usr/lib/libSystem.B.dylib", use_errno=True)

# Never cloned: engine data, downloaded toolchains and browser bundles (the stand runs
# no daemon and no browser), and Finder litter.
SKIP_TOP = frozenset({"claudexor", "playwright-browsers", "package-downloads", ".DS_Store", "settings.json"})
# Process ownership: a copy that held these would treat the owner's live processes as
# its own (reap, signal, adopt). Removed from the copy, never from the live root.
OWNERSHIP = ("state/owned_processes.json", "state/owned_processes.pending", "state/process_ledger.jsonl",
             "state/server_process.json", "state/worker_pids.json", "state/panic_sweep_pending.json")
# Settings keys that must never leave the live file, by name shape (values are never read
# into the report). Model slots and provider credentials are handled by ``keyless``.
_SECRET_MARKERS = ("KEY", "TOKEN", "SECRET", "PASSWORD", "CREDENTIAL", "COOKIE", "AUTH")


def clonefile(src: pathlib.Path, dst: pathlib.Path) -> None:
    """``clonefile(2)``: a file or a whole directory hierarchy, copy-on-write."""
    if _LIBC.clonefile(os.fsencode(str(src)), os.fsencode(str(dst)), 0) != 0:
        err = ctypes.get_errno()
        raise OSError(err, os.strerror(err), f"clonefile {src} -> {dst}")


def _change_counter(path: pathlib.Path) -> int:
    with open(path, "rb") as fh:
        return int.from_bytes(fh.read(28)[24:28], "big")


def _journal_hot(path: pathlib.Path) -> bool:
    """A rollback journal whose header is not zeroed: a write transaction is in flight."""
    try:
        with open(path.with_name(path.name + "-journal"), "rb") as fh:
            return any(fh.read(8))
    except FileNotFoundError:
        return False


def _clone_store(src: pathlib.Path, dst: pathlib.Path, attempts: int = 20) -> Dict[str, Any]:
    """Clone one SQLite store without taking any lock on it.

    A clone of a large, fragmented file can take tens of seconds; holding even a SHARED lock
    that long stalls every writer of the live install (measured 22.10: 28 s on a 2.1 GiB
    ``current.sqlite``). Instead the clone is optimistic: the file change counter (header
    offset 24, bumped by every commit in rollback-journal mode) and the journal header are read
    before and after; any commit or in-flight write overlapping the clone shows in one of them,
    and the clone is discarded and taken again. The copy is then checked with ``quick_check``.
    """
    for attempt in range(1, attempts + 1):
        if dst.exists():
            dst.unlink()
        before, hot_before = _change_counter(src), _journal_hot(src)
        started = time.monotonic()
        clonefile(src, dst)
        clone_ms = round((time.monotonic() - started) * 1000, 1)
        if not hot_before and not _journal_hot(src) and _change_counter(src) == before:
            con = sqlite3.connect(f"file:{dst}?mode=ro", uri=True)
            try:
                check = con.execute("PRAGMA quick_check").fetchone()[0]
            finally:
                con.close()
            return {"path": str(src), "attempts": attempt, "clone_ms": clone_ms, "quick_check": check}
        time.sleep(min(10.0, attempt))
    raise RuntimeError(f"{src}: no quiet window for a consistent clone after {attempts} attempts")


def _store_files(root: pathlib.Path) -> Iterable[pathlib.Path]:
    """The stores a server opens: the root's own and each task drive's (a bounded listing)."""
    yield from sorted((root / "state").glob("*.sqlite"))
    yield from sorted((root / "observability" / "packs").glob("*.sqlite"))
    drives = root / "task_drives"
    if drives.is_dir():
        for drive in sorted(drives.iterdir()):
            yield from sorted((drive / "state").glob("*.sqlite")) if (drive / "state").is_dir() else ()


def _drop_journals(copy_root: pathlib.Path, stores: List[pathlib.Path], live_root: pathlib.Path) -> List[str]:
    dropped = []
    for store in stores:
        for suffix in ("-journal", "-wal", "-shm"):
            twin = copy_root / store.relative_to(live_root).with_name(store.name + suffix)
            if twin.exists():
                twin.unlink()
                dropped.append(str(twin.relative_to(copy_root)))
    return dropped


def _secret_key(name: str) -> bool:
    upper = name.upper()
    return any(marker in upper for marker in _SECRET_MARKERS)


def keyless_settings(live: Dict[str, Any], *, model_keys: Iterable[str], credential_keys: Iterable[str],
                     stub_slug: str) -> tuple[Dict[str, Any], List[str]]:
    """The owner's settings shape with every credential removed and every model slot on the stub."""
    credential = set(credential_keys)
    removed = sorted(key for key in live if key in credential or _secret_key(key) or key.startswith("TELEGRAM_"))
    cfg = {key: value for key, value in live.items() if key not in removed}
    for key in model_keys:
        cfg[key] = ""
    cfg.update({
        "OUROBOROS_MODEL": stub_slug,
        "OUROBOROS_MODEL_LIGHT": stub_slug,
        "OPENAI_COMPATIBLE_API_KEY": "stub-key-not-a-credential",
        "MCP_ENABLED": False,
        "MCP_SERVERS": [],
        "OUROBOROS_STARTER_SKILLS": "[]",
    })
    return cfg, removed


def running_install_pid(live_root: pathlib.Path) -> int:
    """The server PID the install's lock names (``<home>/ouroboros.pid``) when that process lives, else 0."""
    try:
        pid = int((pathlib.Path(live_root).parent / "ouroboros.pid").read_text(encoding="utf-8").strip())
        os.kill(pid, 0)
    except PermissionError:
        return pid
    except (OSError, ValueError):
        return 0
    return pid


def make_snapshot(live_root: pathlib.Path, dest: pathlib.Path, *, model_keys: Iterable[str],
                  credential_keys: Iterable[str], stub_slug: str) -> Dict[str, Any]:
    """Clone ``live_root`` into ``dest/data``; return the manifest (also written beside it)."""
    live_root, dest = pathlib.Path(live_root), pathlib.Path(dest)
    pid = running_install_pid(live_root)
    if pid:
        raise RuntimeError(f"the install is running (pid {pid}); quit the app before taking a snapshot")
    data = dest / "data"
    data.mkdir(parents=True, exist_ok=False)
    started = time.time()
    cloned, skipped = [], []
    for entry in sorted(live_root.iterdir()):
        if entry.name in SKIP_TOP:
            skipped.append(entry.name)
            continue
        clonefile(entry, data / entry.name)
        cloned.append(entry.name)
    stores = list(_store_files(live_root))
    held = [_clone_store(store, data / store.relative_to(live_root)) for store in stores]
    dropped = _drop_journals(data, stores, live_root)
    removed_ownership = []
    for rel in OWNERSHIP:
        target = data / rel
        if target.is_dir():
            shutil.rmtree(target)
            removed_ownership.append(rel)
        elif target.exists():
            target.unlink()
            removed_ownership.append(rel)
    live_settings = json.loads((live_root / "settings.json").read_text(encoding="utf-8"))
    settings, removed_keys = keyless_settings(live_settings, model_keys=model_keys,
                                              credential_keys=credential_keys, stub_slug=stub_slug)
    write_private_json(data / "settings.json", settings)
    manifest = {
        "live_root": str(live_root), "taken_at": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(started)),
        "seconds": round(time.time() - started, 1), "cloned": cloned, "skipped": skipped,
        "stores": held, "dropped_journals": dropped, "removed_ownership": removed_ownership,
        "removed_setting_keys": removed_keys,
        "kept_settings": {key: settings.get(key) for key in (
            "OUROBOROS_MAX_WORKERS", "OUROBOROS_RUNTIME_MODE", "OUROBOROS_CONSCIOUSNESS_AUTONOMY",
            "OUROBOROS_CONTEXT_MODE", "OUROBOROS_TASK_REVIEW_MODE", "OUROBOROS_POST_TASK_EVOLUTION")},
    }
    (dest / "manifest.json").write_text(json.dumps(manifest, indent=1, ensure_ascii=False), encoding="utf-8")
    return manifest


def write_private_json(path: pathlib.Path, payload: Dict[str, Any]) -> None:
    """0600 before any byte lands, as the E2E harness writes settings."""
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        fh.write(json.dumps(payload, indent=2, ensure_ascii=False))
