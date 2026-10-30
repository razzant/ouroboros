"""Read-only Apply and Restart to update effects, bound to their target or prepared candidate; no engine start,
stop or durable state."""

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


def update_effects(plan: dict, *, composition: str | None = None, runtime: dict | None = None) -> dict:
    """One attach-only handshake/status observation for both merge and replacement choices.

    A prepared candidate names its ``composition``, the files its next generation boots: the engine
    transition is read there, never at a merge commit or the target in its place (none: unknown), and is
    final. ``runtime`` is the interpreter its preparation selected (``version``, ``sqlite``; unrecorded:
    unknown), compared with this install's own.
    """
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
    merge = str(plan.get("merge_commit") or target) if composition is None else composition

    def transition(sha):
        error = ""
        try:
            pin = _pin_at(sha)
        except Exception as exc:
            pin, error = None, str(exc)
        return {**engine_pin_transition(pin, serving), "source_sha": sha,
                "error": error or observation_error}

    installed = {"python": platform.python_version(), "sqlite": sqlite3.sqlite_version, "executable": sys.executable}
    facts = {"installed": installed, "bundle": _bundle_runtime(target), "changed_by_apply": False}
    if runtime is not None:
        selected = {"python": runtime.get("version") or None, "sqlite": runtime.get("sqlite") or None,
                    "basis": "candidate_preparation"}
        facts.update(selected=selected, changed_by_apply=None if None in (selected["python"], selected["sqlite"])
                     else (selected["python"], selected["sqlite"]) != (installed["python"], installed["sqlite"]))
    return {
        "target_sha": target,
        "engine": transition(merge), "replacement_engine": transition(target),
        "provisional": composition is None and not bool(plan.get("merge_commit")),
        "active_work": {"status": "known" if known else "unknown",
                        "model": active.get("model") if active else None,
                        "non_model": active.get("non_model") if active else None,
                        "total": sum(active.values()) if known else None},
        "runtime": facts,
    }


def candidate_effects(row: dict) -> dict:
    """``update_effects`` of one prepared candidate's Restart to update (a ``supervisor.update_prepare`` record).

    Its composition is W′ (``layers.new_files``): the tree the switch writes and the restart reads the next
    engine pin from (``body_adoption.bound_candidate_file``), with local work and a reviewed adaptation, never
    the channel's target. Its runtime is the interpreter preparation selected (``environment.candidate``: a
    staged app's measured Python, else this one). ``candidate_id`` binds the effects to that record.
    """
    selected = (row.get("environment") or {}).get("candidate")
    effects = update_effects({"target_sha": (row.get("target") or {}).get("target_sha")},
                             composition=str((row.get("layers") or {}).get("new_files") or ""),
                             runtime=selected if isinstance(selected, dict) else {})
    return {**effects, "candidate_id": str(row.get("candidate_id") or "")}
