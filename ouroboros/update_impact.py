"""Read-only, target-bound Apply effects; no engine start, stop or durable state."""

import platform
import re
import sqlite3
import sys

from ouroboros.engine_pin_policy import engine_pin_transition


def _pin_at(sha):
    from ouroboros.claudexor_runtime import load_runtime_pin
    from supervisor.git_ops import git_capture

    if not sha:
        return None
    rc, raw, _error = git_capture(["git", "show", f"{sha}:ouroboros/claudexor_runtime_pin.json"])
    return load_runtime_pin(data=raw.encode("utf-8")) if rc == 0 else None


def _bundle_runtime(sha):
    """The target's published documentation, not an assertion about a downloaded artifact.

    README owns the current bundle statement. No duplicated runtime version table;
    an absent/changed statement remains unknown. This parser only reads metadata.
    """
    from supervisor.git_ops import git_capture

    rc, readme, _error = git_capture(["git", "show", f"{sha}:README.md"])
    match = re.search(r"^Desktop downloads bundle CPython ([0-9.]+) and SQLite ([0-9.]+)\.", readme, re.MULTILINE) if rc == 0 else None
    return {"python": match[1] if match else None, "sqlite": match[2] if match else None,
            "basis": "target_documentation", "source": f"{sha}:README.md"}


def update_effects(plan: dict) -> dict:
    """One attach-only handshake/status observation for both merge and replacement choices."""
    from ouroboros.claudexor_daemon import owned_daemon_provisioned, read_owned_gateway
    from ouroboros.claudexor_exit_facts import runtime_facts
    from ouroboros.gateways.claudexor import SHORT_POLL_TIMEOUT_SEC

    serving, active = None, None
    observation_error = ""
    if owned_daemon_provisioned():
        try:
            with read_owned_gateway() as gateway:
                serving = (gateway.engine_version, gateway.engine_build_sha)
                admission = runtime_facts(gateway.daemon_status(timeout_sec=SHORT_POLL_TIMEOUT_SEC))["admission"]
                active = admission.get("active") if admission else None
        except Exception as exc:
            observation_error = str(exc)
    else:
        observation_error = "The owned engine is not provisioned."
    known = active is not None and all(type(active.get(key)) is int for key in ("model", "non_model"))
    target = str(plan.get("target_sha") or "")
    merge = str(plan.get("merge_commit") or target)

    def transition(sha):
        error = ""
        try:
            pin = _pin_at(sha)
        except Exception as exc:
            pin, error = None, str(exc)
        return {**engine_pin_transition(pin, serving), "source_sha": sha,
                "error": error or observation_error}

    return {
        "target_sha": target,
        "engine": transition(merge), "replacement_engine": transition(target),
        "provisional": not bool(plan.get("merge_commit")),
        "active_work": {"status": "known" if known else "unknown",
                        "model": active.get("model") if active else None,
                        "non_model": active.get("non_model") if active else None,
                        "total": sum(active.values()) if known else None},
        "runtime": {"installed": {"python": platform.python_version(), "sqlite": sqlite3.sqlite_version,
                                  "executable": sys.executable},
                    "bundle": _bundle_runtime(target), "changed_by_apply": False},
    }
