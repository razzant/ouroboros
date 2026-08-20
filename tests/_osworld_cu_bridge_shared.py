"""Stubs and argv builders shared by the OSWorld cu_bridge suites.

Split out of ``tests/test_osworld_cu_bridge.py`` when that module was divided by theme;
the definitions are verbatim, so every sibling suite keeps the exact seams, argv shape
and attempt layout it was written against.
"""

from __future__ import annotations

import json
import sys




def _cu_bridge_stubs(monkeypatch, tmp_path, *, reward=1.0):
    """Fakes just deep enough to drive `run_cu_bridge_agent.main()` end to end, no VM."""
    import types

    from devtools.benchmarks.osworld import run_cu_bridge_agent as rcb
    from devtools.benchmarks.osworld import run_step_agent

    class _FakeEnv:
        vm_ip = "10.0.0.2"
        server_port = 5000
        client_password = "pw"
        closed = False

        def reset(self, task_config=None):
            return None

        def _get_obs(self):
            return {"screenshot": b"png"}

        def step(self, action, *_a):
            return {}, 0.0, True, {}

        def evaluate(self):
            return reward

        def close(self):
            self.closed = True

    desktop_env = types.ModuleType("desktop_env")
    desktop_env_mod = types.ModuleType("desktop_env.desktop_env")
    desktop_env_mod.DesktopEnv = _FakeEnv
    desktop_env.desktop_env = desktop_env_mod
    monkeypatch.setitem(sys.modules, "desktop_env", desktop_env)
    monkeypatch.setitem(sys.modules, "desktop_env.desktop_env", desktop_env_mod)

    env = _FakeEnv()
    monkeypatch.setattr(run_step_agent, "construct_desktop_env", lambda *a, **k: env)
    monkeypatch.setattr(rcb, "runtime_attestation", lambda url, repo: {"ok": True})
    monkeypatch.setattr(rcb, "_enable_skill", lambda repo, data: {"skill": "seeded"})
    monkeypatch.setattr(rcb, "_publish_target", lambda data, target: tmp_path / "state_target.txt")
    monkeypatch.setattr(rcb, "_collect_budget_counters", lambda *a, **k: {})
    monkeypatch.setattr(
        rcb, "_api",
        lambda url, method, path, body=None, timeout=60: (
            {"task_id": "t1"} if method == "POST" and path == "/api/tasks"
            else {"status": "completed", "final_answer": "done"}
        ),
    )
    return rcb, env

def _cu_bridge_argv(tmp_path, claims):
    osworld = tmp_path / "OSWorld"
    (osworld / "evaluation_examples" / "examples" / "chrome").mkdir(parents=True, exist_ok=True)
    task = osworld / "evaluation_examples" / "examples" / "chrome" / "abc.json"
    task.write_text(json.dumps({"id": "abc", "instruction": "no-op"}), encoding="utf-8")
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir(exist_ok=True)
    (repo_dir / "VERSION").write_text("6.76.0\n", encoding="utf-8")
    results = tmp_path / "results"
    settings = tmp_path / "settings.json"
    settings.write_text("{}", encoding="utf-8")
    return [
        "run_cu_bridge_agent.py", "--osworld-root", str(osworld), "--provider_name", "docker",
        "--path_to_vm", "/vm/Ubuntu.qcow2", "--task", str(task), "--result_dir", str(results),
        "--repo-dir", str(repo_dir), "--data-dir", str(tmp_path / "data"),
        "--settings-path", str(settings), "--ouroboros-url", "http://127.0.0.1:9",
        "--target-file", str(tmp_path / "target.txt"), "--claim-dir", str(claims),
        "--wait_after_reset_sec", "0",          # keeps the suite fast; nothing under test
        "--allow-dirty-seed",
    ], results

def _attempt_dirs(run_dir):
    """Every attempt's own admission/finalization record, oldest first."""
    attempts = run_dir / "attempts"
    return sorted(attempts.iterdir()) if attempts.is_dir() else []
