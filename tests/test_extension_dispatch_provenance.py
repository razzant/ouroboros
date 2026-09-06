"""ABI-9 dispatch-provenance pins (v7next F3 adversarial fix-round).

The generation stamp is a POSITIVE handler-reached fact, never an error-code
exclusion: the three pre-handler EXTENSION_ERROR paths (subprocess-runner
import failure, disclosure gate, calling-convention resolution) are never
stamped; the registry-fallback digest is snapshotted BEFORE the handler runs;
and the DIRECT tools.jsonl record carries the digest via ``tool_result_meta``
even when the detailed ``persist_call`` payload is lost.
"""

from __future__ import annotations

from ouroboros import extension_loader

from tests.test_extension_registration_atomicity import (  # noqa: F401  (autouse fixture applies on import)
    _clear_loader_state,
    _load_dispatch_extension,
)


def _dispatch_ready(tmp_path, monkeypatch, name):
    from ouroboros.extension_surface_names import extension_surface_name
    from ouroboros.tools.tool_context import ToolContext

    loaded, drive_root = _load_dispatch_extension(tmp_path, name)
    monkeypatch.setattr("ouroboros.safety.check_safety", lambda *_a, **_k: (True, ""))
    monkeypatch.setattr("ouroboros.extension_loader.is_extension_live", lambda *_a, **_k: True)
    surface = extension_surface_name(name, "t1")
    ctx = ToolContext(repo_dir=tmp_path, drive_root=drive_root, task_id="prov-task")
    return surface, ctx, extension_loader.get_tool(surface), drive_root, loaded


def test_pre_handler_runner_import_failure_is_never_stamped(tmp_path, monkeypatch):
    """Finding 10, path 1: an out-of-process descriptor whose subprocess runner
    cannot even import fails BEFORE any child dispatch — no digest, no
    physical_dispatch fact."""
    import sys as _sys

    from ouroboros.tools.extension_dispatch import _dispatch_extension_tool_result

    surface, ctx, ext_tool, _root, _loaded = _dispatch_ready(tmp_path, monkeypatch, "prei")
    ext_tool = dict(ext_tool)
    ext_tool["out_of_process"] = True
    monkeypatch.setitem(_sys.modules, "ouroboros.extension_process_runner", None)
    result = _dispatch_extension_tool_result(ctx, surface, ext_tool, {})
    assert (result.status, result.code) == ("error", "EXTENSION_ERROR")
    assert "extension_generation" not in result.meta
    assert "physical_dispatch" not in result.meta


def test_pre_handler_disclosure_gate_failure_is_never_stamped(tmp_path, monkeypatch):
    """Finding 10, path 2: a failing model-cost disclosure gate refuses BEFORE
    the handler — no digest, no physical_dispatch fact."""
    from ouroboros import extension_process_runner
    from ouroboros.tools.extension_dispatch import _dispatch_extension_tool_result

    surface, ctx, ext_tool, _root, _loaded = _dispatch_ready(tmp_path, monkeypatch, "pred")

    def _boom(*_a, **_k):
        raise RuntimeError("disclosure exploded")

    monkeypatch.setattr(
        extension_process_runner, "disclose_inprocess_extension_dispatch", _boom
    )
    result = _dispatch_extension_tool_result(ctx, surface, ext_tool, {})
    assert (result.status, result.code) == ("error", "EXTENSION_ERROR")
    assert "disclosure failed" in result.text
    assert "extension_generation" not in result.meta
    assert "physical_dispatch" not in result.meta


def test_pre_handler_calling_convention_failure_is_never_stamped(tmp_path, monkeypatch):
    """Finding 10, path 3: a calling-convention resolution failure happens
    BEFORE the handler is invoked — no digest, no physical_dispatch fact —
    while a genuine handler exception (same code, same text shape) IS a
    physical dispatch and carries the digest."""
    from ouroboros import extension_process_runner
    from ouroboros.tools.extension_dispatch import _dispatch_extension_tool_result

    surface, ctx, ext_tool, _root, _loaded = _dispatch_ready(tmp_path, monkeypatch, "prec")
    ext_tool = dict(ext_tool)
    ext_tool.pop("wants_ctx", None)

    def _boom(_handler):
        raise RuntimeError("convention resolution exploded")

    monkeypatch.setattr(extension_process_runner, "_handler_wants_ctx", _boom)
    result = _dispatch_extension_tool_result(ctx, surface, ext_tool, {})
    assert (result.status, result.code) == ("error", "EXTENSION_ERROR")
    assert "extension_generation" not in result.meta
    assert "physical_dispatch" not in result.meta

    # Contrast pin: the SAME code from a real handler exception IS stamped.
    failing = dict(extension_loader.get_tool(surface))
    failing["wants_ctx"] = False

    def _handler_raises(**_kw):
        raise RuntimeError("handler exploded")

    failing["handler"] = _handler_raises
    result = _dispatch_extension_tool_result(ctx, surface, failing, {})
    assert (result.status, result.code) == ("error", "EXTENSION_ERROR")
    assert result.meta.get("physical_dispatch") is True
    assert result.meta.get("extension_generation") == (
        extension_loader.extension_generation_digest("prec")
    )


def test_oop_pre_spawn_failure_is_never_stamped(tmp_path, monkeypatch):
    """Fix-round 2, claim 5: an exception from dispatch_extension_tool_subprocess
    raised BEFORE Popen (resolve/load/env/payload staging) carries no typed
    spawn marker — the OOP branch must not stamp physical_dispatch for a
    child that never existed."""
    from ouroboros import extension_process_runner
    from ouroboros.tools.extension_dispatch import _dispatch_extension_tool_result

    surface, ctx, ext_tool, _root, _loaded = _dispatch_ready(tmp_path, monkeypatch, "oopa")
    ext_tool = dict(ext_tool)
    ext_tool["out_of_process"] = True

    def _pre_spawn_boom(*_a, **_k):
        raise RuntimeError("env resolution exploded before Popen")

    monkeypatch.setattr(
        extension_process_runner, "dispatch_extension_tool_subprocess", _pre_spawn_boom
    )
    result = _dispatch_extension_tool_result(ctx, surface, ext_tool, {})
    assert (result.status, result.code) == ("error", "EXTENSION_ERROR")
    assert "physical_dispatch" not in result.meta
    assert "extension_generation" not in result.meta


def test_oop_post_spawn_failure_is_stamped(tmp_path, monkeypatch):
    """Fix-round 2, claim 5 contrast: an exception carrying the process
    runner's typed spawn marker (raised after Popen) IS a physical dispatch
    and carries the generation digest."""
    from ouroboros import extension_process_runner
    from ouroboros.tools.extension_dispatch import _dispatch_extension_tool_result

    surface, ctx, ext_tool, _root, _loaded = _dispatch_ready(tmp_path, monkeypatch, "oopb")
    ext_tool = dict(ext_tool)
    ext_tool["out_of_process"] = True

    def _post_spawn_boom(*_a, **_k):
        raise extension_process_runner._mark_child_spawned(
            extension_process_runner.ExtensionProcessError("child exited abnormally")
        )

    monkeypatch.setattr(
        extension_process_runner, "dispatch_extension_tool_subprocess", _post_spawn_boom
    )
    result = _dispatch_extension_tool_result(ctx, surface, ext_tool, {})
    assert (result.status, result.code) == ("error", "EXTENSION_ERROR")
    assert result.meta.get("physical_dispatch") is True
    assert result.meta.get("extension_generation") == (
        extension_loader.extension_generation_digest("oopb")
    )


def test_run_child_marks_only_post_popen_exceptions(tmp_path, monkeypatch):
    """The marker's OWN seam in _run_child: a Popen failure raises UNMARKED
    (no child existed); once a process object exists, a post-spawn protocol
    failure raises MARKED."""
    import io

    from ouroboros import extension_process_runner

    payload = {"skill_name": "marker-probe"}
    kwargs = dict(
        skill_dir=tmp_path, drive_root=tmp_path, repo_dir=tmp_path,
        env={}, timeout_sec=2,
    )

    def _popen_boom(*_a, **_k):
        raise OSError("exec format error")

    monkeypatch.setattr(extension_process_runner.subprocess, "Popen", _popen_boom)
    try:
        extension_process_runner._run_child(dict(payload), **kwargs)
        raise AssertionError("Popen failure must raise")
    except OSError as exc:
        assert extension_process_runner.extension_child_was_spawned(exc) is False

    class _FakeProc:
        pid = 4242
        returncode = 0
        stdout = io.BytesIO(b"")
        stderr = io.BytesIO(b"")

        def poll(self):
            return 0

        def wait(self, timeout=None):
            return 0

    monkeypatch.setattr(
        extension_process_runner.subprocess, "Popen", lambda *_a, **_k: _FakeProc()
    )
    monkeypatch.setattr(extension_process_runner, "_kill_process_group", lambda _p: None)
    try:
        extension_process_runner._run_child(dict(payload), **kwargs)
        raise AssertionError("missing protocol result must raise")
    except extension_process_runner.ExtensionProcessError as exc:
        assert "did not write protocol result" in str(exc)
        assert extension_process_runner.extension_child_was_spawned(exc) is True


def _fake_proc_cls():
    import io

    class _FakeProc:
        pid = 4243
        returncode = 0

        def __init__(self):
            self.stdout = io.BytesIO(b"")
            self.stderr = io.BytesIO(b"")

        def poll(self):
            return 0

        def wait(self, timeout=None):
            return 0

    return _FakeProc


def test_run_child_marks_registration_exceptions(tmp_path, monkeypatch):
    """Fix-round 3, finding 5a: the process REGISTRATION between Popen and
    the protocol body is inside the marked span — an exception there is
    raised strictly after a successful Popen, so it must carry the marker."""
    from ouroboros import extension_process_runner

    monkeypatch.setattr(
        extension_process_runner.subprocess, "Popen",
        lambda *_a, **_k: _fake_proc_cls()(),
    )
    monkeypatch.setattr(extension_process_runner, "_kill_process_group", lambda _p: None)

    class _BoomSet:
        def add(self, _proc):
            raise RuntimeError("registration exploded")

        def discard(self, _proc):
            pass

    monkeypatch.setattr(extension_process_runner, "_active_subprocesses", _BoomSet())
    try:
        extension_process_runner._run_child(
            {"skill_name": "marker-reg"}, skill_dir=tmp_path, drive_root=tmp_path,
            repo_dir=tmp_path, env={}, timeout_sec=2,
        )
        raise AssertionError("registration failure must raise")
    except RuntimeError as exc:
        assert "registration exploded" in str(exc)
        assert extension_process_runner.extension_child_was_spawned(exc) is True


def test_run_child_cleanup_exception_over_a_marked_one_stays_marked(tmp_path, monkeypatch):
    """Fix-round 3, finding 5b: a cleanup failure in the finally block
    REPLACES the in-flight marked exception — the replacing exception must
    carry the marker too (the child really did run), with the original
    marked exception chained as its context."""
    from ouroboros import extension_process_runner

    monkeypatch.setattr(
        extension_process_runner.subprocess, "Popen",
        lambda *_a, **_k: _fake_proc_cls()(),
    )
    monkeypatch.setattr(extension_process_runner, "_kill_process_group", lambda _p: None)

    def _rmtree_boom(*_a, **_k):
        raise OSError("cleanup exploded")

    monkeypatch.setattr(extension_process_runner.shutil, "rmtree", _rmtree_boom)
    try:
        extension_process_runner._run_child(
            {"skill_name": "marker-clean"}, skill_dir=tmp_path, drive_root=tmp_path,
            repo_dir=tmp_path, env={}, timeout_sec=2,
        )
        raise AssertionError("cleanup failure must raise")
    except OSError as exc:
        assert "cleanup exploded" in str(exc)
        assert extension_process_runner.extension_child_was_spawned(exc) is True
        # The original post-spawn protocol failure (missing result) was marked
        # and stays chained underneath the replacing cleanup exception.
        ctx = exc.__context__
        assert isinstance(ctx, extension_process_runner.ExtensionProcessError)
        assert extension_process_runner.extension_child_was_spawned(ctx) is True


def test_spawn_marker_attaches_even_when_the_exception_refuses_attributes():
    """Fix-round 3, finding 5c: _mark_child_spawned must record the fact even
    for an exception object that refuses setattr — the weak side-table serves
    the read, so the dispatcher still sees the positive spawn fact."""
    from ouroboros import extension_process_runner

    class _Rigid(Exception):
        def __setattr__(self, _key, _value):
            raise AttributeError("frozen exception")

    exc = _Rigid("no attributes here")
    assert extension_process_runner.extension_child_was_spawned(exc) is False
    marked = extension_process_runner._mark_child_spawned(exc)
    assert marked is exc
    assert extension_process_runner.extension_child_was_spawned(exc) is True


class _UnhashableRigid(Exception):
    """Refuses both the marker attribute and hashing — a WeakSet fallback
    would TypeError on add AND on the membership check."""

    __hash__ = None

    def __setattr__(self, _key, _value):
        raise AttributeError("frozen exception")


def test_spawn_marker_side_table_serves_an_unhashable_exception():
    """Fix-round 4, finding 1: the fallback is keyed by IDENTITY, not by
    hash — an unhashable setattr-refusing exception marked post-spawn still
    reads as a positive spawn fact, and an UNMARKED unhashable exception
    reads unstamped without raising (the membership check itself used to
    TypeError on both sides)."""
    from ouroboros import extension_process_runner

    unmarked = _UnhashableRigid("pre-spawn, never marked")
    assert extension_process_runner.extension_child_was_spawned(unmarked) is False

    exc = _UnhashableRigid("post-spawn")
    marked = extension_process_runner._mark_child_spawned(exc)
    assert marked is exc
    assert extension_process_runner.extension_child_was_spawned(exc) is True


def test_spawn_marker_is_identity_bound_not_equality_bound():
    """Fix-round 4, finding 1: two DISTINCT exception objects that compare
    equal must not share the spawn fact — only the exact marked object reads
    stamped (a WeakSet fallback would false-positive on __eq__/__hash__)."""
    from ouroboros import extension_process_runner

    class _Equalish(Exception):
        def __setattr__(self, _key, _value):
            raise AttributeError("frozen exception")

        def __eq__(self, other):
            return isinstance(other, _Equalish)

        def __hash__(self):
            return 42

    marked = _Equalish("really crossed the spawn boundary")
    twin = _Equalish("never did")
    extension_process_runner._mark_child_spawned(marked)
    assert extension_process_runner.extension_child_was_spawned(marked) is True
    assert extension_process_runner.extension_child_was_spawned(twin) is False


def test_spawn_marker_side_table_entries_die_with_the_exception():
    """Fix-round 4, finding 1: the identity side-table must not leak — the
    weakref finalizer purges the entry when the marked exception is
    collected, before its id can be reused for a fresh object."""
    import gc

    from ouroboros import extension_process_runner

    exc = _UnhashableRigid("short-lived")
    extension_process_runner._mark_child_spawned(exc)
    key = id(exc)
    with extension_process_runner._spawned_marker_lock:
        assert key in extension_process_runner._spawned_marker_fallback
    del exc
    gc.collect()
    with extension_process_runner._spawned_marker_lock:
        assert key not in extension_process_runner._spawned_marker_fallback


def test_broken_marker_check_fails_closed_and_never_masks_the_original(tmp_path, monkeypatch):
    """Fix-round 4, finding 1: the marker READ is called inside the
    dispatcher's except handler — a hostile exception object (unhashable,
    raising __getattr__) must yield the fail-closed answer False, so the
    ORIGINAL tool error is reported unstamped instead of being replaced by a
    secondary TypeError from the check itself."""
    from ouroboros import extension_process_runner
    from ouroboros.tools.extension_dispatch import _dispatch_extension_tool_result

    class _Hostile(Exception):
        __hash__ = None

        def __getattr__(self, name):
            raise RuntimeError(f"hostile attribute read: {name}")

    hostile = _Hostile("child never provably existed")
    assert extension_process_runner.extension_child_was_spawned(hostile) is False

    surface, ctx, ext_tool, _root, _loaded = _dispatch_ready(tmp_path, monkeypatch, "oopc")
    ext_tool = dict(ext_tool)
    ext_tool["out_of_process"] = True

    def _hostile_boom(*_a, **_k):
        raise _Hostile("child never provably existed")

    monkeypatch.setattr(
        extension_process_runner, "dispatch_extension_tool_subprocess", _hostile_boom
    )
    result = _dispatch_extension_tool_result(ctx, surface, ext_tool, {})
    assert (result.status, result.code) == ("error", "EXTENSION_ERROR")
    assert "child never provably existed" in result.text
    assert "physical_dispatch" not in result.meta


def test_fallback_digest_is_snapshotted_before_the_handler_runs(tmp_path, monkeypatch):
    """Finding 11: for a descriptor predating the per-surface stamp the
    registry digest is read BEFORE the handler, so a publication that lands
    DURING the handler (deterministic barrier: the handler itself republishes)
    cannot be misattributed to this call's result."""
    from ouroboros.tools.extension_dispatch import _dispatch_extension_tool_result

    surface, ctx, ext_tool, drive_root, loaded = _dispatch_ready(tmp_path, monkeypatch, "presnap")
    pre_call_digest = extension_loader.extension_generation_digest("presnap")
    assert pre_call_digest

    ext_tool = dict(ext_tool)
    ext_tool.pop("extension_generation", None)  # force the registry-reader path
    ext_tool["wants_ctx"] = False

    def _republishing_handler(**_kw):
        extension_loader.unload_extension("presnap")
        err = extension_loader.load_extension(
            loaded, lambda: {}, drive_root=drive_root, _force_in_process=True
        )
        assert err is None, err
        return "ok"

    ext_tool["handler"] = _republishing_handler
    result = _dispatch_extension_tool_result(ctx, surface, ext_tool, {})
    post_call_digest = extension_loader.extension_generation_digest("presnap")
    assert result.status == "ok"
    assert post_call_digest and post_call_digest != pre_call_digest
    assert result.meta.get("extension_generation") == pre_call_digest


def test_legacy_descriptor_digest_is_read_atomically_with_the_descriptor(tmp_path, monkeypatch):
    """Fix-round 2, claim 6: for a descriptor WITHOUT the per-surface stamp
    the candidate reader takes descriptor and registry digest under ONE lock
    hold (get_tool_with_generation) — a republish landing between taking the
    descriptor and dispatching can no longer pair the old handler with the
    NEW generation's digest."""
    from ouroboros import extension_registry_state
    from ouroboros.tools.extension_dispatch import (
        _dispatch_extension_tool_result,
        _extension_dispatch_candidate,
    )

    surface, ctx, _tool, drive_root, loaded = _dispatch_ready(tmp_path, monkeypatch, "atomg")
    pre_digest = extension_loader.extension_generation_digest("atomg")
    assert pre_digest
    # Make the LIVE registry entry a legacy (unstamped) descriptor.
    with extension_registry_state._lock:
        extension_registry_state._tools[surface].pop("extension_generation", None)

    ext_tool, unavailable = _extension_dispatch_candidate(ctx, surface)
    assert ext_tool and not unavailable
    # The atomic combined read already stamped the snapshot digest on the copy.
    assert ext_tool.get("extension_generation") == pre_digest

    # Concurrent republish AFTER the descriptor was taken: live digest moves on.
    extension_loader.unload_extension("atomg")
    err = extension_loader.load_extension(
        loaded, lambda: {}, drive_root=drive_root, _force_in_process=True
    )
    assert err is None, err
    post_digest = extension_loader.extension_generation_digest("atomg")
    assert post_digest and post_digest != pre_digest

    ext_tool = dict(ext_tool)
    ext_tool["wants_ctx"] = False
    ext_tool["handler"] = lambda **_kw: "ok"
    monkeypatch.setattr(
        "ouroboros.extension_loader.is_extension_live", lambda *_a, **_k: True
    )
    result = _dispatch_extension_tool_result(ctx, surface, ext_tool, {})
    assert result.status == "ok"
    assert result.meta.get("extension_generation") == pre_digest  # not post_digest


def test_tools_jsonl_direct_record_carries_the_extension_generation(tmp_path, monkeypatch):
    """Finding 9: the DIRECT tools.jsonl record of a physical extension call
    names the published generation via ``tool_result_meta`` — even when the
    detailed persist_call payload is lost."""
    import json as _json

    import ouroboros.loop_tool_execution as execution
    from ouroboros.tools.extension_dispatch import _dispatch_extension_tool_result

    surface, ctx, ext_tool, _root, _loaded = _dispatch_ready(tmp_path, monkeypatch, "provlog")
    typed = _dispatch_extension_tool_result(ctx, surface, ext_tool, {})
    digest = extension_loader.extension_generation_digest("provlog")
    assert typed.meta.get("extension_generation") == digest

    class FakeRegistry:
        CODE_TOOLS = frozenset()
        _ctx = None

        def execute_result(self, _name, _args):
            return typed

    drive_logs = tmp_path / "logs"
    drive_logs.mkdir()

    def _persist_fails(*_a, **_k):
        raise OSError("observability store down")

    monkeypatch.setattr(execution, "persist_call", _persist_fails)
    execution._execute_single_tool(
        FakeRegistry(),
        {"id": "call-gen", "function": {"name": surface, "arguments": "{}"}},
        drive_logs,
        "task-gen",
    )
    rows = [
        _json.loads(line)
        for line in (drive_logs / "tools.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert rows
    meta = rows[-1].get("tool_result_meta")
    assert isinstance(meta, dict)
    assert meta.get("extension_generation") == digest
    assert meta.get("physical_dispatch") is True
