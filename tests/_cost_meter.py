"""Count the work a public boundary does, attributed to the boundary that caused it.

The cost contract (tests/test_cost_contract.py) compares these counters between two data
roots that hold the same current facts and different amounts of unrelated history. It
counts work, never names or SQL syntax: a full scan hidden inside one SELECT shows up in
the SQLite VM steps, a history walk in the opens, stats and bytes read, a decoded task body
in the JSON decodes. Reading the requested data is legitimate; the counters only say how
much was read and where (``CATEGORIES``), the assertions decide what may not grow.

What is counted, per tag:

- ``connect`` (and ``connect:<category>``): every ``sqlite3.connect``; each new connection gets
  a trace callback (``statements``, ``begins``, ``commits``, ``writes``) and a progress handler
  that adds ``VM_STEP`` to ``vm_steps`` every ``VM_STEP`` SQLite virtual-machine instructions;
- ``stat`` / ``scandir`` / ``listdir`` (and per category): ``os.stat``, ``os.lstat``,
  ``os.scandir``, ``os.listdir`` and the ``pathlib`` accessor (Python 3.10 binds those at
  class creation, so they are wrapped separately), ``os.walk`` and ``glob`` included;
  ``dir_entries`` (and per category): the entries a listing delivered to its reader (each one a
  ``scandir`` iterator yields, the whole ``listdir`` list), so a history-sized directory listed in
  ONE call still costs its size;
- ``open`` (and per category): ``io.open``/``builtins.open``/``Path.open``; ``os_open`` for
  raw ``os.open``; ``read_bytes`` (and per category): bytes delivered by files opened for
  reading through those doors (``os.read``/``os.pread`` on a counted descriptor included; ``os.pread``
  exists only on POSIX); ``lock_acquired``: a lock file created by an exclusive ``os.open`` (every
  lock tier takes its name so), one per acquisition however contended, since a refused create raises;
- ``json_loads`` and ``json_chars``: every ``json.loads``/``json.load`` and its input length;
  ``rows_decoded``: the stored dialogue row bodies among them (``ROW_DECODER``);
- ``rows_crossed``: every row a statement hands to Python as a ``sqlite3.Row`` (the store's row factory),
  one SQLite-to-Python crossing each: a read that captures (``store_capture``) crosses once per statement;
- ``popen`` (argv[0] names kept in ``popen_names``) and ``thread_start`` (thread names).

``callers`` attributes counter amounts and SQL statements to reader stacks, including
root-local work and JSON decodes which have no history path to name.

Locations (``category``): ``root`` and the history locations ``CATEGORIES`` under the data root
(the evidence packs under ``observability/packs/`` hold settled tasks' evidence and are history),
``execution`` for the install's delegated execution roots (``execution_root``: the private
checkouts whose ``.review-evidence/`` trees a capture walks and copies; a copy by ``shutil.copy2``
moves its bytes in the kernel and shows as an ``open`` of the source plus ``stat`` calls, never as
``read_bytes``), ``lock`` for lock files there (how often a lock is probed follows contention
between threads, not the amount of history), ``outside`` for everything else (the interpreter, the
ancestors a path resolution walks), ``fd`` for an integer descriptor.

Attribution (``tag_of``): the context variable ``BOUNDARY`` when set (an ASGI request or the
lifespan task and everything it offloads with ``asyncio.to_thread``, which copies the
context); otherwise the tag a thread received when it started: a ``thread_rules`` match by
name, else the starter's tag mapped through ``lineage``; otherwise ``bg:<thread name>``. Work
submitted to a ``ThreadPoolExecutor`` is credited to the boundary that submitted it (the work
item carries the submitter's tag; pool worker threads themselves inherit nothing), so a pass that
dispatches to a pool owns what the pool does for it. Module-level aliases (``from json import
loads``) are rebound while the meter is installed, so work done through them is counted too.
"""

from __future__ import annotations

import builtins
import collections
import concurrent.futures
import contextvars
import functools
import io
import json
import os
import pathlib
import re
import sqlite3
import subprocess
import sys
import threading
import urllib.parse
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

BOUNDARY: contextvars.ContextVar = contextvars.ContextVar("cost_boundary", default=None)
VM_STEP = 100
# Unrelated-history locations of a data root (POSIX separators, relative to the root).
CATEGORIES = (("archive/", "archive"), ("state/headless_tasks/", "headless"),
              ("task_drives/", "task_drives"), ("task_results/", "task_results"),
              ("observability/packs/", "packs"))
HISTORY = tuple(name for _prefix, name in CATEGORIES)
EXECUTION = "execution"  # the delegated execution roots (outside the data root)
ROW_DECODER = ("ouroboros.dialogue_store", "decode")  # (module, function) that decodes one stored row body
_POOL_TARGETS = {("concurrent.futures.thread", "_worker")}
_POOL_CLASSES = ("WorkerThread", "_WorkerThread")
_DIGITS = re.compile(r"[-_]?\d+$")

_ORIG: Dict[str, Any] = {}
_ACTIVE: Optional["CostMeter"] = None
_LOCAL = threading.local()


_DOORS = ("tests._cost_meter", "subprocess", "pathlib", "os", "genericpath", "posixpath", "ntpath", "io", "json",
          "json.decoder", "sqlite3", "contextlib", "threading", "glob", "shutil", "fnmatch")


def caller(depth: int = 2) -> str:
    """The nearest ``depth`` frames outside the meter and the standard library doors, innermost first."""
    frame, found = sys._getframe(2), []
    while frame is not None and len(found) < depth:
        name = frame.f_globals.get("__name__", "")
        if name not in _DOORS:
            found.append(f"{name}.{frame.f_code.co_name}")
        frame = frame.f_back
    return " <- ".join(found) or "?"


def tag_of() -> str:
    tag = BOUNDARY.get()
    if tag is not None:
        return tag
    tag = getattr(_LOCAL, "tag", None)
    if tag is not None:
        return tag
    return "bg:" + _DIGITS.sub("", threading.current_thread().name or "thread")


def _forms(path: Any) -> List[str]:
    """The absolute and the real form of a directory, longest first, each ending in a separator."""
    if path is None:
        return []
    text = os.fspath(path)
    roots = sorted({os.path.abspath(text), os.path.realpath(text)}, key=len, reverse=True)
    return [root.rstrip(os.sep) + os.sep for root in roots]


class CostMeter:
    """Install with ``with CostMeter(root) as meter:``; read with ``snapshot(tag)``."""

    def __init__(self, data_root: Any = None, *, execution_root: Any = None,
                 thread_rules: Iterable[Tuple[str, str]] = (),
                 lineage: Optional[Dict[str, Optional[str]]] = None) -> None:
        self._roots = _forms(data_root)
        self._execution_roots = _forms(execution_root)
        self.thread_rules: List[Tuple[str, str]] = list(thread_rules)
        self.lineage: Dict[str, Optional[str]] = dict(lineage or {})
        self.counts: Dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
        self.paths: Dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
        self.statements: Dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
        self.callers: Dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
        self.names: Dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
        self.spawned: Dict[str, List[threading.Thread]] = collections.defaultdict(list)
        self._lock = threading.Lock()
        self._fd_category: Dict[int, str] = {}
        self._rebound: List[Tuple[Any, str, Any]] = []
        self._patched: List[Tuple[Any, str, Any]] = []

    # ---- attribution and recording -------------------------------------------------
    def category(self, path: Any) -> str:
        if isinstance(path, int):
            return "fd"
        try:
            text = os.fspath(path)
        except TypeError:
            return "other"
        if isinstance(text, bytes):
            text = text.decode("utf-8", "replace")
        if text.startswith("file:"):  # a SQLite URI (``store.connect``): its path, unquoted
            text = urllib.parse.unquote(urllib.parse.urlsplit(text).path)
        if not os.path.isabs(text):
            text = os.path.join(_ORIG["cwd"], text)
        if not text.endswith(os.sep):
            text_dir = text + os.sep
        else:
            text_dir = text
        if any(text_dir == root or text.startswith(root) for root in self._execution_roots):
            return EXECUTION
        for root in self._roots:
            if text_dir == root:
                return "root"
            if text.startswith(root):
                rel = text[len(root):].replace(os.sep, "/")
                if rel.endswith(".lock"):
                    return "lock"  # a lock file: its retries follow contention, not history
                for prefix, name in CATEGORIES:
                    if rel.startswith(prefix) or rel == prefix[:-1]:
                        return name
                return "root"
        return "outside"

    def add(self, kind: str, amount: int = 1, *, path: Any = None, category: Optional[str] = None) -> str:
        tag = tag_of()
        if category is None and path is not None:
            category = self.category(path)
        source = f"{kind}:{category} <- {caller(4)}" if category else f"{kind} <- {caller(4)}"
        with self._lock:
            counter = self.counts[tag]
            counter[kind] += amount
            self.callers[tag][source] += amount
            if category is not None:
                counter[f"{kind}:{category}"] += amount
                if category in HISTORY and path is not None:
                    self.paths[tag][f"{kind} {self._bucket(category, path)} <- {caller()}"] += 1
        return tag

    def _bucket(self, category: str, path: Any) -> str:
        """Where in the history the call went, coarse enough to stay small: an archive stream
        (``archive/chat``), else the location itself."""
        if category != "archive":
            return category
        name = self._relative(path).replace(os.sep, "/").split("/")[-1]
        return "archive/" + (name.split("_", 1)[0] if name.endswith(".jsonl") else name)

    def _relative(self, path: Any) -> str:
        text = os.fspath(path) if not isinstance(path, int) else str(path)
        text = text.decode("utf-8", "replace") if isinstance(text, bytes) else text
        for root in self._roots:
            if text.startswith(root):
                return text[len(root):]
        return text

    def note_statement(self, statement: str) -> None:
        head = statement.lstrip().split(None, 1)[0].upper() if statement.strip() else ""
        tag = tag_of()
        source = caller(4)
        with self._lock:
            counter = self.counts[tag]
            counter["statements"] += 1
            self.callers[tag][f"statements <- {source}"] += 1
            if head in ("COMMIT", "END"):
                counter["commits"] += 1
            elif head == "BEGIN":
                counter["begins"] += 1
            elif head in ("INSERT", "UPDATE", "DELETE", "REPLACE"):
                counter["writes"] += 1
            self.statements[tag][" ".join(statement.split())[:160]] += 1

    def note_name(self, kind: str, name: str) -> None:
        tag = tag_of()
        with self._lock:
            self.counts[tag][kind] += 1
            self.names[tag][f"{kind} {name}"] += 1

    def snapshot(self, tag: str) -> Dict[str, Any]:
        """A frozen copy of everything recorded for ``tag`` so far."""
        with self._lock:
            return {"counts": dict(self.counts.get(tag, {})),
                    "callers": dict(self.callers.get(tag, {})),
                    "paths": dict(self.paths.get(tag, {})),
                    "statements": dict(self.statements.get(tag, {})),
                    "names": dict(self.names.get(tag, {}))}

    def tags(self) -> List[str]:
        with self._lock:
            return sorted(self.counts)

    def join_spawned(self, tag: str, timeout: float = 30.0) -> List[str]:
        """Join the threads started under ``tag`` (its own helpers); names of those still alive."""
        alive = []
        for thread in list(self.spawned.get(tag, ())):
            thread.join(timeout)
            if thread.is_alive():
                alive.append(thread.name)
        return alive

    # ---- installation --------------------------------------------------------------
    def __enter__(self) -> "CostMeter":
        install(self)
        return self

    def __exit__(self, *_exc) -> None:
        uninstall(self)

    def _thread_tag(self, thread: threading.Thread, parent_tag: Optional[str]) -> Optional[str]:
        for name, tag in self.thread_rules:
            if thread.name == name:
                return tag
        target = getattr(thread, "_target", None)
        if (getattr(target, "__module__", None), getattr(target, "__name__", None)) in _POOL_TARGETS:
            return None
        if type(thread).__name__ in _POOL_CLASSES:
            return None
        if parent_tag is None or parent_tag.startswith("bg:"):
            return None
        return self.lineage.get(parent_tag, parent_tag)


# ---- wrappers (module level so that aliases can be rebound to one object) ------------

def _w_stat(path, *args, **kwargs):
    meter = _ACTIVE
    if meter is not None:
        meter.add("stat", path=path)
    return _ORIG["os.stat"](path, *args, **kwargs)


def _w_lstat(path, *args, **kwargs):
    meter = _ACTIVE
    if meter is not None:
        meter.add("stat", path=path)
    return _ORIG["os.lstat"](path, *args, **kwargs)


class _CountedScandir:
    """The iterator ``os.scandir`` returned, counting every entry it yields (``dir_entries``): the
    directory's size is paid by the reader that examines it, however few calls it made."""

    __slots__ = ("_inner", "_path", "_category")

    def __init__(self, inner, path, category):
        self._inner, self._path, self._category = inner, path, category

    def __iter__(self):
        return self

    def __next__(self):
        entry = next(self._inner)
        meter = _ACTIVE
        if meter is not None:
            meter.add("dir_entries", path=self._path, category=self._category)
        return entry

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        self._inner.close()

    def close(self):
        self._inner.close()


def _w_scandir(path=".", *args, **kwargs):
    meter = _ACTIVE
    if meter is None:
        return _ORIG["os.scandir"](path, *args, **kwargs)
    category = meter.category(path)
    meter.add("scandir", path=path, category=category)
    return _CountedScandir(_ORIG["os.scandir"](path, *args, **kwargs), path, category)


def _w_listdir(path=".", *args, **kwargs):
    meter = _ACTIVE
    if meter is None:
        return _ORIG["os.listdir"](path, *args, **kwargs)
    category = meter.category(path)
    meter.add("listdir", path=path, category=category)
    names = _ORIG["os.listdir"](path, *args, **kwargs)
    if names:
        meter.add("dir_entries", len(names), path=path, category=category)
    return names


def _w_os_open(path, flags, *args, **kwargs):
    fd = _ORIG["os.open"](path, flags, *args, **kwargs)
    meter = _ACTIVE
    if meter is not None:
        category = meter.category(path)
        meter.add("os_open", path=path, category=category)
        meter._fd_category[fd] = category
        if category == "lock" and flags & os.O_EXCL:
            meter.add("lock_acquired")
    return fd


def _w_os_read(fd, n, *args, **kwargs):
    data = _ORIG["os.read"](fd, n, *args, **kwargs)
    meter = _ACTIVE
    if meter is not None and data:
        category = meter._fd_category.get(fd)
        if category is not None:
            meter.add("read_bytes", len(data), category=category)
    return data


def _w_os_pread(fd, n, offset, *args, **kwargs):
    data = _ORIG["os.pread"](fd, n, offset, *args, **kwargs)
    meter = _ACTIVE
    if meter is not None and data:
        category = meter._fd_category.get(fd)
        if category is not None:
            meter.add("read_bytes", len(data), category=category)
    return data


def _w_os_close(fd, *args, **kwargs):
    meter = _ACTIVE
    if meter is not None:
        meter._fd_category.pop(fd, None)
    return _ORIG["os.close"](fd, *args, **kwargs)


class _CountingFileIO(io.FileIO):
    """A real ``FileIO`` whose reads are counted; the buffered/text layers above it stay real."""

    _cost_category = "other"

    def _count(self, amount: int) -> None:
        meter = _ACTIVE
        if meter is not None and amount:
            meter.add("read_bytes", amount, category=self._cost_category)

    def readinto(self, buffer):
        amount = super().readinto(buffer)
        self._count(amount or 0)
        return amount

    def read(self, size=-1):
        data = super().read(size)
        self._count(len(data or b""))
        return data

    def readall(self):
        data = super().readall()
        self._count(len(data or b""))
        return data


def _read_only(mode: str) -> bool:
    return isinstance(mode, str) and "r" in mode and set(mode) <= set("rbt")


def _w_open(file, mode="r", buffering=-1, encoding=None, errors=None, newline=None, closefd=True, opener=None):
    meter = _ACTIVE
    if meter is None:
        return _ORIG["io.open"](file, mode, buffering, encoding, errors, newline, closefd, opener)
    category = meter.category(file)
    meter.add("open", path=file, category=category)
    if not _read_only(mode):
        return _ORIG["io.open"](file, mode, buffering, encoding, errors, newline, closefd, opener)
    raw = _CountingFileIO(file, "r", closefd, opener)
    raw._cost_category = category
    meter._fd_category[raw.fileno()] = category  # ``os.pread`` on it (store._borrow) is counted too
    if "b" in mode:
        if buffering == 0:
            return raw
        return io.BufferedReader(raw, buffering if buffering > 1 else io.DEFAULT_BUFFER_SIZE)
    buffer = io.BufferedReader(raw, buffering if buffering > 1 else io.DEFAULT_BUFFER_SIZE)
    text = io.TextIOWrapper(buffer, encoding, errors, newline, buffering == 1)
    text.mode = mode
    return text


def _w_accessor_open(path, *args, **kwargs):
    return _w_open(path, *args, **kwargs)


def _w_json_loads(s, *args, **kwargs):
    meter = _ACTIVE
    if meter is not None:
        meter.add("json_loads")
        try:
            meter.add("json_chars", len(s))
        except TypeError:
            pass
        frame = sys._getframe(1)
        if (frame.f_globals.get("__name__"), frame.f_code.co_name) == ROW_DECODER:
            meter.add("rows_decoded")
    return _ORIG["json.loads"](s, *args, **kwargs)


class _CountedRow(sqlite3.Row):
    """``sqlite3.Row`` while a meter is installed (the store sets ``row_factory = sqlite3.Row`` at each
    connect): each row a statement hands to Python counts ``rows_crossed``."""

    def __new__(cls, cursor, data):
        meter = _ACTIVE
        if meter is not None:
            meter.add("rows_crossed")
        return super().__new__(cls, cursor, data)


def _trace(statement: str) -> None:
    meter = _ACTIVE
    if meter is not None:
        meter.note_statement(statement)


def _progress() -> int:
    meter = _ACTIVE
    if meter is not None:
        meter.add("vm_steps", VM_STEP)
    return 0


def _w_connect(database, *args, **kwargs):
    conn = _ORIG["sqlite3.connect"](database, *args, **kwargs)
    meter = _ACTIVE
    if meter is not None:
        meter.add("connect", path=database)
        try:
            conn.set_trace_callback(_trace)
            conn.set_progress_handler(_progress, VM_STEP)
        except Exception:
            pass
    return conn


def _w_popen_init(self, args, *rest, **kwargs):
    meter = _ACTIVE
    if meter is not None:
        first = args if isinstance(args, (str, bytes, os.PathLike)) else (list(args)[:1] or ["?"])[0]
        name = os.path.basename(os.fspath(first)) if not isinstance(first, bytes) else "bytes"
        meter.note_name("popen", f"{name} <- {caller()}")
    return _ORIG["Popen.__init__"](self, args, *rest, **kwargs)


def _w_thread_start(self):
    meter = _ACTIVE
    if meter is not None:
        parent = tag_of()
        meter.note_name("thread_start", self.name)
        tag = meter._thread_tag(self, None if parent.startswith("bg:") else parent)
        self._cost_tag = tag
        if tag is not None:
            meter.spawned[tag].append(self)
    return _ORIG["Thread.start"](self)


def _w_bootstrap_inner(self):
    tag = getattr(self, "_cost_tag", None)
    if tag is not None:
        _LOCAL.tag = tag
    return _ORIG["Thread._bootstrap_inner"](self)


def _w_submit(self, fn, /, *args, **kwargs):
    """A pool work item carries its submitter's boundary: the pass that dispatches owns the work."""
    tag = tag_of() if _ACTIVE is not None else None
    if tag is not None and not tag.startswith("bg:"):
        fn = functools.partial(tagged, tag, functools.partial(fn, *args, **kwargs))
        args, kwargs = (), {}
    return _ORIG["ThreadPoolExecutor.submit"](self, fn, *args, **kwargs)


_MODULE_PATCHES = (
    (os, "stat", "os.stat", _w_stat), (os, "lstat", "os.lstat", _w_lstat),
    (os, "scandir", "os.scandir", _w_scandir), (os, "listdir", "os.listdir", _w_listdir),
    (os, "open", "os.open", _w_os_open), (os, "read", "os.read", _w_os_read),
    (os, "pread", "os.pread", _w_os_pread), (os, "close", "os.close", _w_os_close),
    (io, "open", "io.open", _w_open), (builtins, "open", "builtins.open", _w_open),
    (json, "loads", "json.loads", _w_json_loads), (sqlite3, "connect", "sqlite3.connect", _w_connect),
    (sqlite3, "Row", "sqlite3.Row", _CountedRow),
)
_ACCESSOR = getattr(pathlib, "_NormalAccessor", None)
_ACCESSOR_PATCHES = (("stat", _w_stat), ("scandir", _w_scandir), ("listdir", _w_listdir),
                     ("open", _w_accessor_open)) if _ACCESSOR is not None else ()
_SKIP_REBIND = {"os", "posix", "nt", "io", "_io", "builtins", "json", "sqlite3", "pathlib", __name__}
_WRAPPERS = {wrapper for *_rest, wrapper in _MODULE_PATCHES} | {wrapper for _attr, wrapper in _ACCESSOR_PATCHES} \
    | {_w_popen_init, _w_thread_start, _w_bootstrap_inner, _w_submit}


def install(meter: CostMeter) -> None:
    """Replace every counted door; an install that fails part-way puts back what it replaced (a door left
    wrapped would be recorded as its own original by the next install and recurse in every later caller)."""
    global _ACTIVE
    if _ACTIVE is not None:
        raise RuntimeError("one cost meter at a time")
    _ORIG["cwd"] = os.getcwd()
    try:
        _install(meter)
    except BaseException:
        _undo(meter)
        raise
    _ACTIVE = meter


def _replace(meter: CostMeter, owner: Any, attr: str, value: Any) -> Any:
    original = vars(owner)[attr]
    if getattr(original, "__func__", original) in _WRAPPERS:
        raise RuntimeError(f"{getattr(owner, '__name__', owner)}.{attr} is still the meter's wrapper")
    setattr(owner, attr, value)
    meter._patched.append((owner, attr, original))  # only a replacement that took place is put back
    return original


def _install(meter: CostMeter) -> None:
    originals = {}
    for module, attr, key, wrapper in _MODULE_PATCHES:
        if not hasattr(module, attr):  # ``os.pread`` is POSIX-only: Windows has no such door to count
            continue
        _ORIG[key] = _replace(meter, module, attr, wrapper)
        originals[id(_ORIG[key])] = wrapper
    # Capability probes must see the same platform while its OS doors are wrapped.
    # Otherwise confined_files imported during a cold boot refuses every POSIX child.
    for attr in ("supports_dir_fd", "supports_fd", "supports_follow_symlinks", "supports_effective_ids"):
        if hasattr(os, attr):
            supported = getattr(os, attr)
            setattr(os, attr, {originals.get(id(call), call) for call in supported})
            meter._patched.append((os, attr, supported))
    for attr, wrapper in _ACCESSOR_PATCHES:
        _ORIG["accessor." + attr] = _replace(meter, _ACCESSOR, attr, staticmethod(wrapper))
    _ORIG["Popen.__init__"] = _replace(meter, subprocess.Popen, "__init__", _w_popen_init)
    _ORIG["Thread.start"] = _replace(meter, threading.Thread, "start", _w_thread_start)
    _ORIG["Thread._bootstrap_inner"] = _replace(meter, threading.Thread, "_bootstrap_inner", _w_bootstrap_inner)
    _ORIG["ThreadPoolExecutor.submit"] = _replace(meter, concurrent.futures.ThreadPoolExecutor, "submit", _w_submit)
    for name, module in list(sys.modules.items()):
        if module is None or name in _SKIP_REBIND:
            continue
        try:
            items = list(vars(module).items())
        except TypeError:
            continue
        for attr, value in items:
            wrapper = originals.get(id(value))
            if wrapper is not None and not isinstance(value, (int, str)):
                setattr(module, attr, wrapper)  # a module may refuse: then there is nothing to put back
                meter._rebound.append((module, attr, value))


def uninstall(meter: CostMeter) -> None:
    global _ACTIVE
    if _ACTIVE is not meter:
        return
    _ACTIVE = None
    _undo(meter)


def _undo(meter: CostMeter) -> None:
    """Put every replaced door back, past any that refuses, then name every refusal at once."""
    refused = []
    for owner, attr, original in [*reversed(meter._rebound), *reversed(meter._patched)]:
        try:
            setattr(owner, attr, original)
        except Exception as exc:  # the other doors still go back; this one is named below
            refused.append(f"{getattr(owner, '__name__', owner)}.{attr} ({exc})")
    meter._rebound.clear()
    meter._patched.clear()
    if refused:
        raise RuntimeError(f"the cost meter could not put back {', '.join(refused)}")


def tagged(tag: str, call: Callable[[], Any]) -> Any:
    """Run ``call`` with ``BOUNDARY`` set to ``tag`` (an in-process boundary)."""
    token = BOUNDARY.set(tag)
    try:
        return call()
    finally:
        BOUNDARY.reset(token)
