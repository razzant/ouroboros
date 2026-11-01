"""No automatic start follows a recorded Panic once an Update's tree was returned: real Git, hook, helper and lifecycle.

A return that hands over removes its pointer, so nothing at the next start's pre-import hook holds that start: what
must not start it is whoever would. The Panic arrives during the return itself, written by a real Git
``reference-transaction`` hook on the helper's own ``update-ref`` (every Panic check before that point saw none).
The launcher here is the real ``launcher.agent_lifecycle_loop`` under owner intent; its generations are real cold
entries of the miniature serving tree, only the process bookkeeping around them is stood in. Controls: without a Panic
the same transitions and crashes go on as before. Nothing here records a Panic outside its scene's data root.
"""
from __future__ import annotations

import os
import subprocess
import sys
import threading

import pytest

from ouroboros import body_adoption, body_switch
from tests.body_candidate_support import git
from tests.test_update_activation import STYLE_NEW, Scene, _passive_base
from tests.test_update_switch_panic import _interrupted, _panic

pytestmark = [pytest.mark.serial,  # real cold-entry processes
              pytest.mark.skipif(os.name == "nt", reason="the Panic arrives through a POSIX shell Git hook")]
BROKEN = {"ouroboros/mod_a.py": "GEN = (\n"}  # the official commit does not parse: its import probe fails
_FLAG = "panic_stop.flag"
# The launcher's own marker reaches its generation as ``start_agent`` gives it; the test harness strips it from a test's
# direct children, so this one sets it itself and becomes the entry (the same process, as a generation is).
_MANAGED_ENTRY = ("import os, sys; os.environ['OUROBOROS_MANAGED_BY_LAUNCHER'] = '1'; "
                  "os.execv(sys.executable, [sys.executable, 'server.py'])")


@pytest.fixture
def scene(tmp_path, monkeypatch):
    return Scene(tmp_path, monkeypatch)


def _panic_during_the_return(scene):
    hook = os.path.join(body_switch.git_dir(str(scene.serving)), "hooks", "reference-transaction")
    os.makedirs(os.path.dirname(hook), exist_ok=True)
    with open(hook, "w", encoding="utf-8") as fh:
        fh.write(f"#!/bin/sh\necho panic > '{scene.data / 'state' / _FLAG}'\n")
    os.chmod(hook, 0o755)


def _mid_return(scene, *, native):
    """An Update whose entry died with the branch moved and whose new tree does not import: the next entry returns it,
    moving the branch back. ``native``: an app transition whose installer died, returned whatever a probe would say."""
    scene.official(BROKEN)
    scene.local_work(commit=False)
    before = scene.composition()
    scene.activate(scene.prepare())
    assert scene.boot(OUROBOROS_BODY_SWITCH_FAIL_AT="after_ref").returncode == 137
    assert scene.handoff()["phase"] == "switching" and git(scene.serving, "rev-parse", "HEAD") != before["head"]
    if native:
        helper = str(body_adoption.helper_dir(scene.data))
        handoff = body_switch.read_handoff(helper)
        handoff["native"] = {"installer": {"pid": 0}}
        body_switch.write_handoff(helper, handoff)
    return before


def _lifecycle(scene, monkeypatch, *, generations, argv=None, **stand_ins):
    """The real loop under owner intent until it stops, or until ``generations`` were started (then it is told to
    stop): ``(the loop's own stop, the exit of each generation)``."""
    import launcher

    started, shutdown = [], threading.Event()

    def start_agent(_port):
        if len(started) == generations:
            shutdown.set()
            return None
        with open(scene.tmp / f"generation-{len(started)}.log", "wb") as log:  # read by a failure, none here otherwise
            started.append(subprocess.Popen(argv or [sys.executable, "-c", _MANAGED_ENTRY], cwd=str(scene.serving),
                                            stdout=log, stderr=log,
                                            env={**os.environ, "TOY_REPORT": str(scene.report)}))
        return started[-1]

    for name, value in (("DATA_DIR", scene.data), ("_launch_intent", "owner"), ("_shutdown_event", shutdown),
                        ("_pre_generation_cleanup", lambda: []), ("_clear_port_file", lambda: True),
                        ("update_external_host", lambda *a: {}), ("start_agent", start_agent),
                        ("stop_agent", lambda: None), ("_await_generation_endpoint", lambda *a, **kw: None),
                        ("_retire_agent", lambda *a: None), ("loaded_packages_replaced", lambda _repo: False),
                        ("_sync_existing_repo_from_bundle", lambda: None), ("restart_dependencies", lambda *a: None),
                        ("launcher_sources_changed", lambda *a, **kw: False), *stand_ins.items()):
        monkeypatch.setattr(launcher, name, value)
    monkeypatch.setattr(launcher.time, "sleep", lambda *a: None)
    launcher._agent_restart_requested.clear()
    launcher.agent_lifecycle_loop(1)
    return len(started) < generations, [proc.returncode for proc in started]


@pytest.mark.parametrize("panic", [True, False], ids=["panic", "control"])
@pytest.mark.parametrize("native", [False, True], ids=["source", "native"])
def test_the_launcher_never_starts_a_returned_tree_after_a_panic(scene, monkeypatch, native, panic):
    before = _mid_return(scene, native=native)
    if panic:
        _panic_during_the_return(scene)

    stopped, exits = _lifecycle(scene, monkeypatch, generations=2)

    decision = scene.handoff()
    assert decision["phase"] == "returned" and scene.composition() == before  # coherent either way
    assert decision["return"]["trigger"] == ("native_switch_interrupted" if native else "import_probe_failed")
    if panic:
        assert (scene.data / "state" / _FLAG).exists()
        assert (exits, stopped, scene.reports()) == ([body_switch.STUCK_EXIT_CODE], True, [])
        # The next start is the owner's: it runs the returned tree, and the outcome settles as it always does.
        assert scene.boot().returncode == 0 and [report["a"] for report in scene.reports()] == ["GEN_OLD"]
        assert body_adoption.finalize_on_boot(scene.data, scene.serving, supervisor_ready=True)["outcome"] == "returned"
    else:  # the hand-over through the launcher's own restart, and the previous composition serves
        told = (scene.tmp / "generation-0.log").read_text(encoding="utf-8", errors="replace")
        assert exits == [body_switch.RESTART_EXIT_CODE, 0], (told, decision["events"][-3:], scene.reports())
        assert [report["a"] for report in scene.reports()] == ["GEN_OLD"]


def _switched_for_a_return(scene, *, native):
    """A switched Update whose next entry returns it through ``_update_entry``: the installer that died before its
    exchange (``native``), or the stylesheet-only successor that entered and did not start."""
    if native:
        scene.official({"ouroboros/mod_a.py": "GEN = 'GEN_CAND'\n"})
        scene.activate(scene.prepare())
        assert scene.boot(OUROBOROS_BODY_SWITCH_FAIL_AT="after_ref").returncode == 137
        installed = scene.tmp / "Installed.app"
        installed.mkdir()
        helper = str(body_adoption.helper_dir(scene.data))
        handoff = body_switch.read_handoff(helper)
        handoff.update(phase="switched", native={"installed": str(installed), "installer": {"pid": 0},
                                                 "previous_identity": body_switch.bundle_identity(str(installed))})
        body_switch.write_handoff(helper, handoff)
    else:
        _passive_base(scene)
        scene.official({"web/style.css": STYLE_NEW})
        scene.activate(scene.prepare())
        assert scene.boot(TOY_WRITE_THEN_FAIL="1").returncode != 0
    return scene.handoff()["old"]


@pytest.mark.parametrize("panic", [True, False], ids=["panic", "control"])
@pytest.mark.parametrize("native", [False, True], ids=["compatible_return", "app_not_exchanged"])
def test_a_direct_entrys_return_hands_nothing_over_after_a_panic(scene, native, panic):
    """No launcher: the entry itself would replace its image with a fresh process of the returned tree."""
    old = _switched_for_a_return(scene, native=native)
    if panic:
        _panic_during_the_return(scene)
    reports = len(scene.reports())

    booted = scene.boot()

    assert scene.handoff()["phase"] == "returned" and git(scene.serving, "rev-parse", "HEAD") == old
    if panic:
        assert booted.returncode == body_switch.STUCK_EXIT_CODE and "Panic is recorded" in booted.stderr
        assert len(scene.reports()) == reports  # no fresh process, no import
    else:
        assert booted.returncode == 0 and len(scene.reports()) == reports + 1, booted.stderr


def test_the_launcher_stops_at_a_held_transition_after_a_panic(scene, monkeypatch):
    _interrupted(scene)
    _panic(scene)

    stopped, exits = _lifecycle(scene, monkeypatch, generations=3)

    assert (exits, stopped) == ([body_switch.STUCK_EXIT_CODE], True) and scene.handoff()["phase"] == "stuck"


def test_a_generation_that_crashes_without_a_panic_is_still_restarted(scene, monkeypatch):
    stopped, exits = _lifecycle(scene, monkeypatch, generations=3, argv=[sys.executable, "-c", "raise SystemExit(3)"])

    assert (exits, stopped) == ([3, 3, 3], False)


@pytest.mark.parametrize("panic", [True, False], ids=["panic", "control"])
def test_a_panic_during_the_launchers_restart_cancels_its_hand_over_to_a_fresh_launcher(scene, monkeypatch, panic):
    """Exit 42, then the restart's dependency step, which can take minutes, is when the Panic lands: a fresh launcher
    would keep the owner's intent and start its first generation. The hand-over asks, and the cycle stops."""
    cancelled = []

    def dependencies(*_a):
        if panic:
            _panic(scene)

    stopped, exits = _lifecycle(scene, monkeypatch, generations=2, argv=[sys.executable, "-c", "raise SystemExit(42)"],
                                restart_dependencies=dependencies, launcher_sources_changed=lambda *a, **kw: True,
                                _bootstrap_context=lambda: None, exec_source_launcher=lambda _context, _argv, **kw: (
                                    cancelled.append(kw["cancelled"]())))

    assert cancelled[0] is panic and (exits, stopped) == (([42], True) if panic else ([42, 42], False))
