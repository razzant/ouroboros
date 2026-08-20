"""OSWorld checkout, provider preflight and DesktopEnv lifecycle.

Verbatim extraction from ``run_step_agent.py`` (v7 stream W): the pinned aligned
upstream, the supported local providers, the checkout probe, the provider
preflight, the optional-dependency stubs, the vmrun PATH fix, the live-desktop
server guard, and construction/teardown of the official ``DesktopEnv``.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
import types
from pathlib import Path
from typing import Any

VMWARE_FUSION_PATHS = (
    "/Applications/VMware Fusion.app/Contents/Public",
    "/Applications/VMware Fusion.app/Contents/Library",
)


# The exact upstream this adapter is aligned against. Verified 2026-07-03 from
# primary sources (repo tree, run scripts, lib_run_single.py, desktop_env.py,
# show_result.py at this commit; paper arXiv:2606.29537):
# - Official launch scripts run with ``--max_steps 500`` and inline checkpoint
#   evaluations at 150/300 (scripts/bash/run_multienv_claude.sh); the bare
#   ``run.py`` argparse default is the legacy 15.
# - Evaluation is VM-state-only: ``DesktopEnv.evaluate()`` scores getters over
#   files/app/OS/browser state; the ONLY agent-message channel is the special
#   ``FAIL`` action for ``evaluator.func == "infeasible"`` tasks.
# - ``show_result.py`` consumes ``<result_dir>/<action_space>/<observation_type>/
#   <model>/<domain>/<example_id>/result.txt``.
ALIGNED_UPSTREAM = {
    "repo": "https://github.com/xlang-ai/OSWorld-V2",
    "commit": "c261cb57a699bd18db128787ca4e71b749141762",
    "commit_date": "2026-06-30",
    "paper": "arXiv:2606.29537 (OSWorld 2.0: Benchmarking Computer Use Agents on Long-Horizon Real-World Tasks)",
    "protocol_max_steps": 500,
    "protocol_checkpoint_steps": [150, 300],
    "legacy_repo": "https://github.com/xlang-ai/OSWorld",
}


# Providers this adapter can actually drive locally. Official OSWorld 2.0 also
# supports aws/azure/gcp/aliyun/volcengine, but this adapter has no cloud path.
SUPPORTED_PROVIDERS = ("vmware", "docker")


def osworld_checkout_info(osworld_root: Path) -> dict[str, Any]:
    """Describe an OSWorld checkout: variant (v1/v2), git commit, key modules.

    Variant markers verified against the upstream trees:
    ``evaluation_examples/test_v2.json`` exists only in OSWorld-V2;
    ``evaluation_examples/test_all.json`` only in classic OSWorld.
    """

    root = Path(osworld_root).expanduser().resolve(strict=False)
    info: dict[str, Any] = {
        "root": str(root),
        "exists": root.is_dir(),
        "variant": "unknown",
        "git_commit": "",
        "matches_aligned_commit": False,
        "has_desktop_env": (root / "desktop_env" / "desktop_env.py").is_file(),
        "aligned_upstream": dict(ALIGNED_UPSTREAM),
    }
    if (root / "evaluation_examples" / "test_v2.json").is_file():
        info["variant"] = "v2"
    elif (root / "evaluation_examples" / "test_all.json").is_file():
        info["variant"] = "v1"
    elif (root / "evaluation_examples").is_dir():
        info["variant"] = "examples_only"
    try:
        proc = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if proc.returncode == 0:
            info["git_commit"] = proc.stdout.strip()
    except Exception:
        pass
    info["matches_aligned_commit"] = bool(info["git_commit"]) and info["git_commit"] == ALIGNED_UPSTREAM["commit"]
    return info


def provider_preflight_failures(provider_name: str, path_to_vm: str) -> list[str]:
    """Fail loudly (with what is missing) when the VM provider cannot run here."""

    provider = str(provider_name or "").strip().lower()
    failures: list[str] = []
    if provider not in SUPPORTED_PROVIDERS:
        failures.append(
            f"provider '{provider}' is not supported by this adapter "
            f"(supported: {', '.join(SUPPORTED_PROVIDERS)}); official OSWorld 2.0 cloud "
            "providers (aws/azure/gcp) have no local adapter path"
        )
        return failures
    if provider == "vmware":
        vm_path = Path(path_to_vm).expanduser()
        if not vm_path.exists():
            failures.append(f"VM path not found: {vm_path}")
        _ensure_vmrun_on_path()
        if not any((Path(path) / "vmrun").exists() for path in VMWARE_FUSION_PATHS) and not shutil.which("vmrun"):
            failures.append("vmrun not found (checked VMware Fusion app paths and PATH)")
    elif provider == "docker":
        docker = shutil.which("docker")
        if not docker:
            failures.append("docker CLI not found on PATH (required by the docker provider)")
        else:
            try:
                proc = subprocess.run(
                    [docker, "info", "--format", "{{.ServerVersion}}"],
                    capture_output=True,
                    text=True,
                    timeout=15,
                )
                if proc.returncode != 0:
                    failures.append(
                        "docker daemon not reachable: "
                        + (proc.stderr or proc.stdout or "").strip()[:200]
                    )
            except Exception as exc:  # noqa: BLE001 - preflight diagnostics
                failures.append(f"docker daemon probe failed: {type(exc).__name__}: {exc}")
    return failures


def _install_optional_dependency_stubs() -> None:
    """Avoid heavy optional evaluator imports when a selected task does not use them."""

    if "easyocr" not in sys.modules:
        easyocr = types.ModuleType("easyocr")

        class _UnavailableReader:
            def __init__(self, *_args: Any, **_kwargs: Any) -> None:
                raise RuntimeError("easyocr is not installed; OCR metrics unavailable")

        easyocr.Reader = _UnavailableReader  # type: ignore[attr-defined]
        sys.modules["easyocr"] = easyocr

    if "fastdtw" not in sys.modules:
        fastdtw_mod = types.ModuleType("fastdtw")

        def _fastdtw_unavailable(*_args: Any, **_kwargs: Any) -> tuple[float, list[Any]]:
            raise RuntimeError("fastdtw is not installed; audio metrics unavailable")

        fastdtw_mod.fastdtw = _fastdtw_unavailable  # type: ignore[attr-defined]
        sys.modules["fastdtw"] = fastdtw_mod


def _ensure_vmrun_on_path() -> None:
    path_parts = os.environ.get("PATH", "").split(os.pathsep)
    changed = False
    for candidate in VMWARE_FUSION_PATHS:
        if Path(candidate, "vmrun").exists() and candidate not in path_parts:
            path_parts.insert(0, candidate)
            changed = True
    if changed:
        os.environ["PATH"] = os.pathsep.join(path_parts)


_DEFAULT_DESKTOP_PORT = 8765
_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "0.0.0.0", "::1", "[::1]", ""})


def _is_default_desktop_server(url: str) -> bool:
    """True if ``url`` points at the LIVE desktop server's port on any loopback
    spelling. The guard keyed on the literal ``http://127.0.0.1:8765`` string, so
    ``localhost:8765`` / ``127.0.0.2:8765`` / ``[::1]:8765`` bypassed it and could
    still write into the live data root (adversarial review r1)."""
    from urllib.parse import urlparse

    try:
        parsed = urlparse(str(url or "").strip())
    except Exception:
        return False
    host = (parsed.hostname or "").strip().lower()
    port = parsed.port if parsed.port is not None else (443 if parsed.scheme == "https" else 80)
    is_loopback = host in _LOOPBACK_HOSTS or host.startswith("127.")
    return is_loopback and port == _DEFAULT_DESKTOP_PORT


def _teardown_partial_desktop_env(env: Any) -> None:
    """Best-effort teardown of a DesktopEnv whose ``__init__`` raised.

    ``env.close()`` is the official path (it calls
    ``provider.stop_emulator(path_to_vm)``); a construction that died before
    ``provider``/``path_to_vm`` were assigned cannot use it, so fall back to the
    provider directly. Never raises: cleanup must not mask the original failure.
    """
    try:
        env.close()
        return
    except Exception:
        pass
    provider = getattr(env, "provider", None)
    if provider is None:
        return
    try:
        provider.stop_emulator(getattr(env, "path_to_vm", None))
    except Exception:
        pass


def construct_desktop_env(desktop_env_cls: Any, *, attempts: int, deadline: float,
                          retry_sleep_sec: float = 5.0, **kwargs: Any) -> Any:
    """Construct ``DesktopEnv``, retrying a failed boot and tearing down each attempt.

    THE AUTHORISED BENEFIT IS THE RETRY (owner decision Q15=A). ``DesktopEnv.__init__``
    boots the VM/container inside ``_start_emulator()``, and the launchers used to retry
    only ``env.reset`` — so one transient boot failure (a lost
    ``/tmp/docker_port_allocation.lck`` race, a slow image load) burned the whole task.
    The constructor is now retried inside the startup window instead.

    Teardown of failed attempts is BELT-AND-BRACES, not a fix for measured debris: no run
    here has been shown to accumulate leaked containers. It is done because a raise inside
    ``__init__`` discards the half-built object, so whatever ``_start_emulator()`` had
    already started would be unreachable and therefore unstoppable. Constructing through
    ``__new__`` + explicit ``__init__`` keeps that partially-initialised instance
    reachable, which is the only way to close it at all.

    ``deadline`` is an absolute ``time.time()`` bound on STARTING a new attempt (an
    in-flight attempt is never cut short), so a run cannot spend its whole startup window
    respawning VMs.
    """
    last_err = ""
    for attempt in range(1, max(1, int(attempts)) + 1):
        if attempt > 1 and time.time() >= deadline:
            last_err = f"{last_err}; startup deadline reached, no further attempts"
            break
        env = desktop_env_cls.__new__(desktop_env_cls)
        try:
            env.__init__(**kwargs)
            return env
        except Exception as exc:  # noqa: BLE001 - every failed boot must be cleaned up
            last_err = f"attempt {attempt}: {type(exc).__name__}: {exc}"
            print(f"[osworld] DesktopEnv construction failed ({last_err}); tearing down", flush=True)
            _teardown_partial_desktop_env(env)
            time.sleep(max(0.0, float(retry_sleep_sec)))
    raise RuntimeError(f"DesktopEnv construction failed: {last_err}")
