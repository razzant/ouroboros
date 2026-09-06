"""Typed process facts: the handler→loop seam for host process tools.

R5 (node-runtime sprint, stream B): the process-tool handler measures its child
directly (returncode, POSIX signal name, wall-clock duration, and — via the
stream-A resolver attestation — the physically resolved runtime) and publishes
the facts through a THREAD-LOCAL slot; ``loop_tool_execution`` consumes them
for the same tool call and merges them into the typed ``result_meta``.

Thread-local, not ctx-scoped, because the tool executor runs each handler in
its own worker thread — the slot is therefore naturally per-in-flight-call,
and an ABANDONED (outer-timeout) handler thread that finishes late writes only
its own thread's slot and can never contaminate a later call's facts. The
rendered result text is unchanged by this channel; a record with no typed
publication carries no process facts at all — under the typed-result organ
(D02) prose is never harvested into typed fields.

This lives outside ``tools/shell.py`` deliberately: it is a loop↔handler seam
(``tools/verify.py`` consumes the attested runtime through it too), and keeping it here keeps
``shell.py`` under the repository's 1600-line hard size gate.
"""

from __future__ import annotations

import pathlib
import threading
import time
from typing import Dict

from ouroboros.platform_layer import posix_signal_name


def signal_name_for_returncode(returncode) -> str:
    """POSIX signal name for a NEGATIVE subprocess returncode, '' otherwise.

    SSOT for the signal-name derivation: the shell result renderer, the typed
    process facts, and the verify_and_record receipt all read this one helper so
    a killed child is named identically everywhere. Windows residual (disclosed,
    not faked): a killed process there reports a large POSITIVE exit code (e.g.
    0xC0000005), so no signal name can be derived and this returns '' —
    signal-death naming is POSIX-only.
    """
    try:
        rc = int(returncode)
    except (TypeError, ValueError):
        return ""
    if rc >= 0:
        return ""
    return posix_signal_name(abs(rc))


_process_facts_tls = threading.local()

# The complete typed fact family this channel owns. When typed facts exist for
# a call, they are authoritative for EVERY member — including the ABSENCE of a
# member (a typed publication without ``signal`` means the child was not
# signal-killed, however much the child's own stdout may spell ``signal=...``).
# ``timed_out`` / ``killed_by_host`` / ``pre_exec_failure`` are the members that
# exist precisely where an exit code does NOT: a child the host killed on its
# deadline, a child the host killed for any other reason (output cap, abandoned
# validator), and a child that never reached exec at all. They are what makes
# the honest "no exit code" cases legible instead of silent — and they are the
# ONLY signal-death-shaped fact a Windows kill can carry, because
# ``TerminateProcess`` leaves a large POSITIVE exit status and no POSIX signal
# number, so the partition below can name nothing there (declared residual).
PROCESS_FACT_KEYS = (
    "exit_code",
    "signal",
    "duration_ms",
    "resolved_runtime",
    "timed_out",
    "killed_by_host",
    "pre_exec_failure",
)


def active_resolved_runtime(ctx) -> str:
    """Stream-A seam (node-runtime sprint): the interpreter resolver (python
    pre-dispatch; node post-gates) sets ``ctx._process_resolved_runtime`` to
    the ABSOLUTE physical
    executable it substituted for this call — set ONLY when execution runs
    something other than the literal recorded argv (an argv rewrite or an
    emergency PATH prepend), scoped to the handler invocation exactly like
    ``ctx._active_python_resolution``. Absent/empty means the argv executed as
    written. Read here so the typed result_meta and the verify_and_record
    receipt disclose the same fact from the same slot."""
    try:
        return str(getattr(ctx, "_process_resolved_runtime", "") or "").strip()
    except Exception:
        return ""


def publish_process_facts(
    *, returncode=None, started_ts: float, resolved_runtime: str = "",
    timed_out: bool = False, killed_by_host: bool = False,
    pre_exec_failure: str = "",
) -> Dict[str, object]:
    """Publish this thread's typed process facts for the in-flight process tool.

    ``duration_ms`` is always recorded; ``exit_code``/``signal`` only when the
    child actually returned a code (a timeout or a pre-exec failure has none).
    Returns the facts it published. ``timed_out`` says the host stopped the child on its deadline,
    ``killed_by_host`` that the host killed it at all (a deadline kill sets
    both), and ``pre_exec_failure`` carries the typed cause — the exception
    class the platform raised — of a child that never started. On POSIX a
    host kill is normally ALSO visible as ``signal``; on Windows it is not,
    and ``killed_by_host`` beside whatever ``TerminateProcess`` left in
    ``exit_code`` is the whole honest fact the platform gives.
    """
    facts: Dict[str, object] = {
        "duration_ms": max(0, int((time.monotonic() - float(started_ts)) * 1000)),
    }
    if timed_out:
        facts["timed_out"] = True
    if killed_by_host:
        facts["killed_by_host"] = True
    if pre_exec_failure:
        facts["pre_exec_failure"] = str(pre_exec_failure)[:120]
    if returncode is not None:
        try:
            facts["exit_code"] = int(returncode)
        except (TypeError, ValueError):
            pass
        else:
            name = signal_name_for_returncode(returncode)
            if name:
                facts["signal"] = name
    if resolved_runtime:
        facts["resolved_runtime"] = resolved_runtime
    _process_facts_tls.facts = facts
    # Returned so a producer that ALSO discloses these facts elsewhere (the
    # verify_and_record receipt) copies the published ones instead of deriving
    # its own second time — one derivation, two disclosures.
    return dict(facts)


def consume_last_process_facts() -> "Dict[str, object] | None":
    """Read-and-CLEAR the typed process facts published on this thread.

    Called by loop_tool_execution: once defensively before dispatching a
    process tool (drops stale facts) and once after, to merge the fresh facts
    into the call's ``result_meta``. Returns ``None`` when nothing was
    published (e.g. an argument-error path where no process ran)."""
    facts = getattr(_process_facts_tls, "facts", None)
    _process_facts_tls.facts = None
    return facts if isinstance(facts, dict) else None


def describe_returncode(returncode: int, *, cwd=None, binding=None,
                        lived_ms: "int | None" = None, resolved_runtime: str = "") -> str:
    """Render a return code with signal details when applicable.

    On a SIGNAL death the caller may disclose the child's lifetime (a
    millisecond count names the kernel-kill incident class) and the physical
    runtime that actually ran (T11); non-signal renderings are unchanged.
    ``binding`` is any resolved resource binding (root/source/skill_name).
    """
    suffix: list = []
    signal_name = signal_name_for_returncode(returncode)
    if signal_name:
        suffix.append(f"signal={signal_name}")
        if lived_ms is not None:
            suffix.append(f"lived={lived_ms}ms")
        if resolved_runtime:
            suffix.append(f"runtime={resolved_runtime}")
    if cwd is not None:
        suffix.append(f"cwd={pathlib.Path(cwd).resolve(strict=False)}")
    rendered_suffix = f" ({', '.join(suffix)})" if suffix else ""
    target_suffix = ""
    if binding is not None:
        target = [f"root={binding.root}", f"source={binding.source}"]
        if binding.skill_name:
            target.append(f"skill={binding.skill_name}")
        target_suffix = "; " + ", ".join(target)
    return f"exit_code={returncode}{rendered_suffix}{target_suffix}"
