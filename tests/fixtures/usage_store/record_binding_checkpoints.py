"""Provenance of ``binding_checkpoints.json.gz`` (a pytest plugin for the PRE-store tree).

It records the binding-authority answers of the journal-era code: every
OUTERMOST call a test makes to a tracked authority function, with a snapshot
of the data root taken just before it (every file), the budget environment,
the arguments and the answer. ``tests/test_usage_store_imported_bindings.py``
replays each question against a fresh root holding the same files, so the
usage store imports that journal and must give the same answer.

Produced from 84febbdd3 (the last journal-era tree): extract it
(``git archive 84febbdd3``), copy this file to ``tests/_c1_record_plugin.py``
there, run ``tests/test_batch4_compaction_authority.py`` and
``tests/test_usage_ledger_legacy_bindings.py`` through ``scripts/safe_test.py``
with ``-p tests._c1_record_plugin``, then keep the journal, task results,
watermark, direct-roots and queue-snapshot files of each checkpoint
(content-addressed), drop exact duplicates and the three tests that patch the
retired writer internals, and gzip the JSON. This file is never imported here.
"""
from __future__ import annotations

import base64
import functools
import json
import os
import pathlib
import threading
import types

OUT = pathlib.Path("binding_checkpoints.jsonl")  # in the extracted tree it runs in
_LOCAL = threading.local()
_STATE = {"test": "", "index": 0}
_ENV_KEYS = ("TOTAL_BUDGET", "OUROBOROS_PER_TASK_COST_USD")


def _root_from(name, args, kwargs):
    if name in {"original_group_limit", "ledger_billing_binding", "task_money_snapshot", "effective_billing_fields"}:
        value = args[0] if args else kwargs.get("drive_root") or kwargs.get("budget_root") or kwargs.get("root")
    elif name == "task_billing_fields":
        value = args[3] if len(args) > 3 else kwargs.get("budget_root")
    elif name == "_billing_group":
        value = getattr(args[0], "DRIVE_ROOT", None)
    elif name == "usage_projection":
        value = args[0] if args else kwargs.get("drive_root")
    elif name == "admit_continuation":
        from supervisor import queue
        value = getattr(queue, "DRIVE_ROOT", None)
    else:
        value = None
    return pathlib.Path(value) if value else None


def _encode(value, root):
    text_root = str(root)
    if isinstance(value, pathlib.PurePath):
        return {"__path__": str(value).replace(text_root, "<ROOT>")}
    if isinstance(value, str):
        return value.replace(text_root, "<ROOT>")
    if isinstance(value, types.SimpleNamespace):
        return {"__ns__": {k: _encode(v, root) for k, v in vars(value).items()}}
    if isinstance(value, dict):
        return {str(k): _encode(v, root) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_encode(v, root) for v in value]
    if isinstance(value, (int, float, bool)) or value is None:
        return value
    return {"__repr__": repr(value)}


def _snapshot(root):
    files = {}
    if root is None or not root.is_dir():
        return files
    for path in sorted(root.rglob("*")):
        if path.is_file() and not path.name.endswith(".lock"):
            data = path.read_bytes()
            try:
                files[path.relative_to(root).as_posix()] = {"text": data.decode("utf-8").replace(str(root), "<ROOT>")}
            except UnicodeDecodeError:
                files[path.relative_to(root).as_posix()] = {"b64": base64.b64encode(data).decode("ascii")}
    return files


def _wrap(name, fn):
    if getattr(fn, "_c1_recorded", False):
        return fn

    @functools.wraps(fn)
    def recorded(*args, **kwargs):
        depth = getattr(_LOCAL, "depth", 0)
        if depth:
            return fn(*args, **kwargs)
        root = _root_from(name, args, kwargs)
        snapshot = _snapshot(root) if root is not None else {}
        env = {key: os.environ.get(key) for key in _ENV_KEYS}
        _LOCAL.depth = depth + 1
        try:
            result = fn(*args, **kwargs)
            error = None
        except BaseException as exc:  # noqa: BLE001 - recorded, then re-raised
            result, error = None, f"{type(exc).__name__}"
            raise
        finally:
            _LOCAL.depth = depth
            if root is not None:
                extra = None
                if name == "admit_continuation" and error is None:
                    from supervisor import workers
                    pending = list(getattr(workers, "PENDING", []) or [])
                    extra = pending[-1].get("metadata", {}).get("continuation") if pending else None
                _STATE["index"] += 1
                with OUT.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps({
                        "test": _STATE["test"], "index": _STATE["index"], "function": name,
                        "args": _encode(list(args), root), "kwargs": _encode(kwargs, root),
                        "env": env, "files": snapshot, "error": error,
                        "result": _encode(result, root), "continuation": _encode(extra, root),
                    }, sort_keys=True) + "\n")
        return result

    recorded._c1_recorded = True
    return recorded


def _install():
    import ouroboros.usage_accounting as ua
    import ouroboros.usage_admission as admission
    import supervisor.continuation_admission as continuation

    for name in ("original_group_limit", "ledger_billing_binding", "task_billing_fields",
                 "task_money_snapshot", "effective_billing_fields"):
        setattr(admission, name, _wrap(name, getattr(admission, name)))
    ua.usage_projection = _wrap("usage_projection", ua.usage_projection)
    continuation._billing_group = _wrap("_billing_group", continuation._billing_group)
    continuation.admit_continuation = _wrap("admit_continuation", continuation.admit_continuation)


def pytest_configure(config):
    _install()


def pytest_runtest_setup(item):
    _STATE["test"] = item.nodeid
    _STATE["index"] = 0
