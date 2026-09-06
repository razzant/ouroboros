"""verify_and_record — the host runs the agent's declared verification check and
writes a durable, host-attested receipt (FR3 verify-before-done).

One call runs the check AND attests the result, so it replaces the run the agent
would have done anyway (≈ zero extra rounds). The contract KIND is agent-declared
(LLM-first, P5 — the host never infers from prose whether a machine-checkable
contract exists); the host only executes and attests what it can. Receipts feed
the verification ledger and suppress the receipt_absent transparency flag.
"""

from __future__ import annotations

import json
import pathlib
import shlex
import subprocess
import time
from typing import Any, List

from ouroboros._outcome_receipts import (
    CHECK_RENDERING_DECLARED_TEXT,
    CHECK_RENDERING_SHLEX_JOIN,
)
from ouroboros.outcome_receipt_store import ZERO_RUN_WRITE_DECISIONS
from ouroboros.outcomes import append_verification_receipt
from ouroboros.platform_layer import bootstrap_process_path
from ouroboros.process_interpreters import (
    active_node_resolution,
    apply_env_path_prepend,
    interpreter_path_overlay,
)
from ouroboros.shell_parse import normalize_check_argv
from ouroboros.tool_access import (
    ResolvedResourceBinding,
    active_tool_profile,
    build_resolved_resource_binding,
    canonical_data_root,
)
from ouroboros.tools.registry import ToolContext, ToolEntry, active_repo_dir_for
from ouroboros.utils import utc_now_iso

# Durable receipt evidence is bounded but the truncation is DISCLOSED (BIBLE P1, never
# silent); the tool-result preview is bounded separately for transport.
_RECEIPT_OUTPUT_CAP = 20000
# The typed process facts a run-kind receipt DISCLOSES (D6/R4). `duration_ms` —
# always, for every run-kind check: a 9ms SIGKILL and a 5-minute honest red are
# different facts. `signal` — the POSIX name of the signal that killed the check
# (returncode < 0), so a killed verification is distinguishable in the store
# (Windows reports large positive codes and gets no name — declared residual,
# see process_facts.signal_name_for_returncode). `resolved_runtime` — present
# ONLY when execution ran something other than the literal recorded check
# (Stream-A resolver seam); its absence means the recorded argv is exactly what
# executed. The channel publishes MORE than this (the kill facts belong to the
# call's result_meta, not to the receipt's stored shape), so the receipt keeps
# exactly the three keys it has always carried, copied from the one publication
# instead of derived a second time.
_RECEIPT_PROCESS_KEYS = ("duration_ms", "signal", "resolved_runtime")
_TOOL_OUTPUT_CAP = 4000
# The `no_visible_machine_contract` receipt carries the agent's OWN stated proxy and
# residual risk — decision-shaping cognitive evidence a reviewer reads, so it goes
# through the same disclosed bound as every other durable receipt field.
_RECEIPT_DECLARED_SUMMARY_CAP = 1000


def _bounded(text: Any, cap: int) -> str:
    t = str(text or "").strip()
    if len(t) <= cap:
        return t
    return t[:cap] + f"\n…[truncated {len(t) - cap} of {len(t)} chars]"


def _receipt_custody_failure(kind: str, detail: str = "") -> str:
    suffix = f" Observed result: {detail}" if detail else ""
    return (
        f"⚠️ TOOL_ERROR (verify_and_record): receipt_custody_failed for {kind}; "
        "the check/observation may have run, but no durable receipt was recorded. "
        "Do not treat this verification as attested; retry after receipt custody "
        f"recovers.{suffix}"
    )

_CONTRACT_KINDS = (
    "visible_verifier",
    "explicit_command",
    "explicit_metric",
    "artifact_observation",
    "no_visible_machine_contract",
    "delegation_zero_run",
)
_RUN_KINDS = frozenset({"visible_verifier", "explicit_command", "explicit_metric"})
# WRITE vocabulary only — hydration reads keep historical "complete" valid
# through ZERO_RUN_READ_DECISIONS in the same SSOT module.
_ZERO_RUN_DECISIONS = ZERO_RUN_WRITE_DECISIONS
# How `expected` is matched against the check output. `substring` is the DEFAULT
# and keeps the historical behavior byte-identical when the param is omitted.
# `bytes_equal` (v6.60.0) compares TWO FILES byte-for-byte (artifact_paths[0] vs
# artifact_paths[1]) after the optional check — the golden-file / migration-parity
# shape — recording a bounded hexdump of the first divergence in the receipt.
_EXPECTED_MATCH_KINDS = ("substring", "exact", "exact_line", "json_equals", "bytes_equal")


def _expected_matches(out: str, expected: str, mode: str) -> bool:
    """Match `expected` against the check `out` under the declared `mode`. Substring
    (default) preserves legacy behavior; exact/exact_line/json_equals are opt-in
    stricter checks for tasks with a worked example or a structured deliverable."""
    if mode == "exact":
        return out.strip() == expected.strip()
    if mode == "exact_line":
        target = expected.strip()
        return any(line.strip() == target for line in out.splitlines())
    if mode == "json_equals":
        try:
            return json.loads(out) == json.loads(expected)
        except (ValueError, TypeError):
            return False
    return expected in out  # substring


# Check→argv normalization is the SSOT `shell_parse.normalize_check_argv` (shared with the
# shell guard so the guard inspects EXACTLY what executes; stringified-argv recovery + non-
# login `sh -c` PATH parity with run_command live there).
_normalize_check = normalize_check_argv

# Shell stages that, as the LAST stage of a pipeline, almost always exit 0 even when an earlier
# real command failed — so the pipeline's exit (POSIX: the last stage's) MASKS the true result.
_EXIT_MASK_FILTER_CMDS = frozenset({"tail", "head", "grep", "egrep", "fgrep", "sed", "awk", "cat", "tee", "tr", "sort", "uniq", "wc", "true", ":"})
_SHELL_C_HEADS = frozenset({"sh", "bash", "dash", "ash", "zsh"})


def _check_has_exit_masking(argv: List[str]) -> tuple[bool, list[str]]:
    """Exit-code MASKING sensor (v6.52.2, FLAG-ONLY — never changes the verdict). Detects, in a
    SHELL-STRING check `["sh"/"bash"/..., "-c", text]`, constructs that launder the real exit code
    so a failing runner reads as exit 0 (the false-green tutanota hit): a trailing pipe into a text
    filter (`... | tail/head/grep/...`; POSIX pipeline exit = the LAST stage), `|| true` / `|| :`,
    or an explicit successful tail. Token-scans via shlex OUTSIDE quotes so a quoted
    literal (e.g. grep PATTERN '| tail') is not flagged. Mirrors the artifact_lifecycle flag: it
    informs the advisory reviewer + the agent, P5-clean (it decides nothing). Returns (masked, reasons)."""
    if not argv or len(argv) < 3:
        return False, []
    if pathlib.PurePath(str(argv[0])).name.lower() not in _SHELL_C_HEADS or str(argv[1]) not in ("-c", "-lc"):
        return False, []
    text = str(argv[2])
    # Operator-aware tokenization (shlex with `punctuation_chars`) so `|`/`||` are split out as
    # standalone tokens EVEN WHEN glued to words (`pytest -q|tail`, `make test||true`) — plain
    # shlex.split is whitespace-only and would miss the no-space forms. Quotes are still respected,
    # so a quoted literal (e.g. a grep pattern `'| tail'`) is NOT flagged.
    try:
        lexer = shlex.shlex(text, posix=True, punctuation_chars="|&<>;")
        lexer.whitespace_split = True
        toks = list(lexer)
    except ValueError:
        return False, []
    reasons: list[str] = []
    for i, tok in enumerate(toks[:-1]):
        if tok == "||" and toks[i + 1] in ("true", ":"):
            reasons.append("|| true")
            break
    pipe_positions = [i for i, tok in enumerate(toks) if tok == "|"]
    if pipe_positions:
        nxt = pipe_positions[-1] + 1
        last_stage = pathlib.PurePath(toks[nxt]).name.lower() if nxt < len(toks) else ""
        if last_stage in _EXIT_MASK_FILTER_CMDS:
            reasons.append(f"pipeline_{last_stage}")
    if len(toks) >= 2 and toks[-1] in {"true", ":"} and toks[-2] == ";":
        reasons.append(f"{toks[-2]} true")
    if len(toks) >= 3 and toks[-2:] == ["exit", "0"] and toks[-3] in {";", "||"}:
        reasons.append("exit 0")
    seen: set = set()
    ordered = [r for r in reasons if not (r in seen or seen.add(r))]
    return bool(ordered), ordered


# Public name for the SECOND consumer: run_command/run_script read the same
# sensor to disclose a masked green in their result envelope.
check_exit_masking = _check_has_exit_masking


_OBSERVABLE_EXTRA_ROOTS = ("subagent_projects", "deliverables", "artifact_store", "task_drive")


def _within_readonly_orchestrator_root(ctx: ToolContext, candidate: pathlib.Path) -> bool:
    """True when ``candidate`` resolves inside an observation-sanctioned resource
    root beyond the active workspace: the read-only orchestrator roots
    (``subagent_projects`` = the durable genesis/coop projects tree,
    ``deliverables`` = the unnamed-deliverables container) plus — derived from the
    ONE ``_POLICY`` matrix, not a private list (wave3 r24: verify refused the very
    ``artifact_store`` path the agent had just registered, forcing a Deliverables
    copy-shuffle) — the task's own ``artifact_store``/``task_drive``. Each root
    counts ONLY when the active profile can already read it per the matrix
    (``profile_readable_root_paths``), so this widens observation to nothing the
    profile cannot already ``read_file``. DISCLOSED NARROWING (capability delta,
    v6.91 scope S2): the pre-matrix version trusted ``subagent_projects``/
    ``deliverables`` UNCONDITIONALLY for every profile; the matrix drops them for
    profiles carrying no read verb on those roots — ``acting_subagent``,
    ``local_readonly_subagent``, ``skill_repair`` — so a child observing a
    sibling's coop-tree deliverable now falls through to the user_files guards
    (the projects tree overlaps the soft-protected workspace parent) and records
    REFUSED_OUT_OF_SCOPE where it previously recorded OBSERVED. Accepted: the
    refusal is a non-failure policy receipt, and sibling-deliverable observation
    rides the ORCHESTRATING parent (workspace_task / external_workspace_task /
    self_modification / operator_control keep the verbs), not the child.
    Deliberately NOT widened to
    ``system_repo``/``runtime_data``: existence-observation of the control plane
    stays out of scope (anti-cheat boundary). An existence/size observation only,
    never a content read."""
    from ouroboros.tool_access import path_is_relative_to, profile_readable_root_paths

    for label, root in profile_readable_root_paths(ctx):
        if label not in _OBSERVABLE_EXTRA_ROOTS:
            continue
        if candidate == root or path_is_relative_to(candidate, root):
            return True
    return False


def _confine_artifact_path(ctx: ToolContext, raw: str) -> tuple[pathlib.Path | None, str]:
    """SSOT confinement for a declared artifact path. The RESOLVED path (whether the input
    was absolute or relative) must stay inside the active workspace, OR a read-only
    orchestrator root (subagent_projects / deliverables — existence observation of a child's
    deliverable), else clear the user_files guards (control-plane/secret and outside-home
    refused) — so a relative `../../etc/passwd` cannot probe arbitrary host files. Returns
    (candidate, refused_reason): candidate is the resolved host path; refused_reason is
    non-empty when refused; both falsy for empty input. Used by _observe_artifacts."""
    from ouroboros.tool_access import path_is_relative_to, user_files_path_block_reason

    text = str(raw or "").strip()
    if not text:
        return None, ""
    active = pathlib.Path(active_repo_dir_for(ctx)).resolve(strict=False)
    p = pathlib.Path(text)
    candidate = (p if p.is_absolute() else (active / text)).resolve(strict=False)
    within_active = candidate == active or path_is_relative_to(candidate, active)
    if within_active or _within_readonly_orchestrator_root(ctx, candidate):
        return candidate, ""
    if user_files_path_block_reason(ctx, candidate):
        return None, f"path refused (outside workspace / control-plane): {text}"
    return candidate, ""


def _observe_artifacts(ctx: ToolContext, artifact_paths: List[str]) -> tuple[str, str]:
    """Read-only existence observation for declared deliverable paths. Never reads content.
    Returns (status, detail) where status is one of: ``observed`` (all present),
    ``fail`` (given but some missing / none given), or ``refused_out_of_scope`` (a path was
    refused by the confinement policy — a POLICY block, NOT a verification failure: it does
    not raise has_failures and is not a red FAIL in the UI, v6.57.0)."""
    missing: List[str] = []
    seen: List[str] = []
    for raw in artifact_paths:
        candidate, refused = _confine_artifact_path(ctx, raw)
        if refused:
            return "refused_out_of_scope", refused
        if candidate is None:
            continue
        seen.append(str(raw or "").strip())
        if not candidate.exists():
            missing.append(str(raw or "").strip())
    if not seen:
        return "fail", "no artifact_paths given"
    if missing:
        return "fail", f"missing: {', '.join(missing[:10])}"
    return "observed", f"observed {len(seen)} artifact(s): {', '.join(seen[:10])}"


def _bytes_equal_confinement_block(
    ctx: ToolContext, a_raw: str, b_raw: str, *, use_executor: bool
) -> str:
    """Confinement for the bytes_equal operands (adversarial r1 — the comparison is a
    BYTE-READ oracle: sizes + a hexdump of the first divergence). Both declared paths
    must clear the SAME boundaries every other artifact-path surface enforces:
    - `_confine_artifact_path` (workspace / read-only orchestrator roots / user_files
      guard — no control-plane, secret-shaped, or arbitrary-host reads);
    - the protected-artifacts policy for `read_bytes` (a black-box reference binary
      must not be byte-compared — that IS reading its bytes);
    - in-executor: workspace-RELATIVE paths only (no absolute / `..` — same rule as
      `_probe_artifact_lifecycle`, else cmp becomes an oracle over hidden grader files).
    DELIBERATE user_files lane (claudexor final review, declined-as-blocker): an
    in-home, non-secret, non-control-plane file that clears the user_files guard IS
    comparable — every profile that reaches this tool already holds full user_files
    READ (a size+hexdump is strictly weaker than the read_file it already has), and
    bench adapters jail user_files via OUROBOROS_USER_FILES_ROOT so hidden graders
    outside the jail still refuse. Returns a refusal string, or "" when both
    operands are comparable."""
    for raw in (a_raw, b_raw):
        if use_executor:
            _ep = pathlib.PurePosixPath(str(raw).replace("\\", "/"))
            if _ep.is_absolute() or ".." in _ep.parts:
                return f"bytes_equal refused: executor paths must be workspace-relative (got {raw!r})"
        candidate, refused = _confine_artifact_path(ctx, raw)
        if refused:
            return f"bytes_equal refused: {refused}"
        if candidate is not None:
            try:
                from ouroboros.protected_artifacts import block_reason_for_path

                block = block_reason_for_path(ctx, candidate, "read_bytes")
            except Exception:
                block = ""
            if block:
                return f"bytes_equal refused: {block}"
    return ""


def _compare_files_bytes_equal(
    ctx: ToolContext, artifact_paths: List[str], work_dir: pathlib.Path, *, use_executor: bool
) -> tuple[bool, str]:
    """v6.60.0 expected_match="bytes_equal": byte-for-byte comparison of exactly TWO
    files (artifact_paths[0] vs [1]) — the golden-file/migration-parity shape a weaker
    substring check silently under-verifies. Runs on the SAME surface as the check
    (executor `cmp` in-container when the cwd is executor-mapped, host chunked read
    otherwise). Both operands are CONFINED first (see _bytes_equal_confinement_block).
    Returns (equal, detail); detail carries a BOUNDED hexdump around the first
    divergence so the receipt shows WHERE the bytes differ, never whole files."""
    a_raw, b_raw = str(artifact_paths[0]).strip(), str(artifact_paths[1]).strip()
    refusal = _bytes_equal_confinement_block(ctx, a_raw, b_raw, use_executor=use_executor)
    if refusal:
        return False, refusal
    if use_executor:
        from ouroboros.workspace_executor import execute as _executor_execute

        res = _executor_execute(ctx, ["cmp", "--", a_raw, b_raw], pathlib.Path(work_dir), 60)
        # No `or` coercion on an exit code: `None or 0` would read an unknown
        # outcome as cmp's success and report the two files byte-equal. Absent or
        # non-numeric is unknown, which is not equality.
        rc_raw = getattr(res, "returncode", None)
        try:
            rc = int(rc_raw)
        except (TypeError, ValueError):
            return False, f"bytes_equal: executor cmp returned no exit status for {a_raw} vs {b_raw}"
        out = ((res.stdout or "") + ("\n" + res.stderr if res.stderr else "")).strip()
        if rc == 0:
            return True, f"bytes_equal: {a_raw} == {b_raw} (executor cmp)"
        if rc > 1:
            # cmp semantics: 0=equal, 1=differ, >1=trouble (missing file, missing cmp
            # binary =127). Still a FAIL, but the receipt names the INFRA cause instead
            # of misattributing a tooling absence as byte divergence (triad r3).
            return False, (
                f"bytes_equal infra error (cmp exit {rc}, not a byte verdict) "
                f"({a_raw} vs {b_raw}): {out[:400] or 'no output'}"
            )
        return False, f"bytes differ ({a_raw} vs {b_raw}): {out[:400] or 'cmp exit ' + str(rc)}"

    def _resolve(text: str) -> pathlib.Path:
        p = pathlib.Path(text)
        return (p if p.is_absolute() else pathlib.Path(work_dir) / text).resolve(strict=False)

    a_path, b_path = _resolve(a_raw), _resolve(b_raw)
    for label, path in ((a_raw, a_path), (b_raw, b_path)):
        if not path.is_file():
            return False, f"bytes_equal: file not found: {label}"
    a_size, b_size = a_path.stat().st_size, b_path.stat().st_size
    offset = 0
    first_diff = -1
    with a_path.open("rb") as fa, b_path.open("rb") as fb:
        while True:
            ca, cb = fa.read(65536), fb.read(65536)
            if not ca and not cb:
                break
            if ca != cb:
                limit = min(len(ca), len(cb))
                for i in range(limit):
                    if ca[i] != cb[i]:
                        first_diff = offset + i
                        break
                if first_diff < 0:
                    first_diff = offset + limit  # one file is a prefix of the other
                break
            offset += len(ca)
    if first_diff < 0 and a_size == b_size:
        return True, f"bytes_equal: {a_raw} == {b_raw} ({a_size} bytes)"
    if first_diff < 0:
        first_diff = min(a_size, b_size)
    window_start = max(0, first_diff - 16)

    def _hexwin(path: pathlib.Path) -> str:
        try:
            with path.open("rb") as f:
                f.seek(window_start)
                return f.read(48).hex(" ")
        except OSError:
            return "(unreadable)"

    return False, (
        f"bytes differ at offset {first_diff} (sizes {a_size} vs {b_size}).\n"
        f"{a_raw} @{window_start}: {_hexwin(a_path)}\n"
        f"{b_raw} @{window_start}: {_hexwin(b_path)}"
    )


def _probe_artifact_lifecycle(
    ctx: ToolContext, artifact_paths: List[str], work_dir: pathlib.Path, *, use_executor: bool
) -> tuple[list[dict], list[str]]:
    """C (after-only): for each agent-declared artifact path, record whether it still exists
    AFTER the run-kind check — probed via the SAME surface as the check (executor when the cwd
    is executor-mapped, else host). FLAG-ONLY structural fact: catches a check that built then
    DELETED the deliverable it just attested (e.g. compile+import+rm a `.so` → green self-check,
    red grade). HOST-INITIATED probe only (not the agent's declared check; never re-enters the
    safety gate, not itself attestable). Returns (artifact_lifecycle, artifacts_missing_after)."""
    lifecycle: list[dict] = []
    missing_after: list[str] = []
    surface = "executor" if use_executor else "host"
    for raw in list(artifact_paths or [])[:20]:
        text = str(raw or "").strip()
        if not text:
            continue
        exists: bool | None = None
        check_surface = surface
        try:
            if use_executor:
                # Probe IN the same container as the check, CONFINED to the check's workspace:
                # only a workspace-RELATIVE path (resolves under work_dir in-container) is probed.
                # An absolute or traversing path is NOT probed — else the flag could detect hidden
                # grader files (e.g. /hidden/tests), weakening the public-info-only anti-cheat boundary.
                _ep = pathlib.PurePosixPath(text.replace("\\", "/"))
                if _ep.is_absolute() or ".." in _ep.parts:
                    check_surface = "unavailable"
                else:
                    from ouroboros.workspace_executor import execute as _executor_execute
                    res = _executor_execute(ctx, ["sh", "-c", 'test -e "$1"', "_", text], pathlib.Path(work_dir), 30)
                    # No `or 1` coercion: returncode 0 is falsy, so it inverted every
                    # healthy probe into exists_after=false (a None returncode now
                    # raises into the advisory "unavailable" arm instead of lying).
                    exists = int(getattr(res, "returncode", 1)) == 0
            else:
                # HOST branch: resolve a RELATIVE path against the CHECK's cwd (work_dir) — the
                # check ran there, so its relative deliverable lives there. A relative path MUST
                # stay inside work_dir: a `../` traversal escaping it is NOT probed. This is a
                # string-shape confinement (resolve-then-contain), so it holds regardless of where
                # work_dir sits — a temp work_dir nested under $HOME (Windows runners) cannot be
                # escaped into the home tree. An ABSOLUTE path is probed only if it clears the
                # user_files guard (control-plane/secret/outside-home); else it is not probed.
                from ouroboros.tool_access import path_is_relative_to, user_files_path_block_reason

                raw_p = pathlib.Path(text)
                wd = pathlib.Path(work_dir).resolve(strict=False)
                if raw_p.is_absolute():
                    candidate = raw_p.resolve(strict=False)
                    if user_files_path_block_reason(ctx, candidate):
                        check_surface = "unavailable"
                    else:
                        exists = bool(candidate.exists())
                else:
                    candidate = (wd / text).resolve(strict=False)
                    if candidate == wd or path_is_relative_to(candidate, wd):
                        exists = bool(candidate.exists())
                    else:
                        check_surface = "unavailable"
        except Exception:  # noqa: BLE001 — probe is advisory; never break the receipt
            exists, check_surface = None, "unavailable"
        lifecycle.append({"path": text[:300], "exists_after": exists, "check_surface": check_surface})
        if exists is False:
            missing_after.append(text[:300])
    return lifecycle, missing_after


def _record_delegation_zero_run(
    ctx: ToolContext,
    *,
    bootstrap: dict[str, Any],
    task_id: str,
    criterion_id: str,
    criterion_basis: Any,
    check: Any,
    kwargs: dict[str, Any],
) -> str:
    """Record one actor-first no-physical-run decision after custody is clear."""

    if bool(bootstrap.get("physical_started")):
        return (
            "⚠️ TOOL_ARG_ERROR (verify_and_record): delegation_zero_run is no longer "
            "available after the selected physical leaf has started."
        )
    if bool(bootstrap.get("zero_run_receipt_recorded")):
        return (
            "⚠️ TOOL_ARG_ERROR (verify_and_record): delegation_zero_run already "
            "has a durable terminal decision for this actor."
        )
    decision = str(
        kwargs.get("zero_run_decision") or kwargs.get("decision") or ""
    ).strip().lower()
    if decision not in _ZERO_RUN_DECISIONS:
        return (
            "⚠️ TOOL_ARG_ERROR (verify_and_record): zero_run_decision must be one of "
            f"{', '.join(_ZERO_RUN_DECISIONS)}."
        )
    basis = " ".join(str(
        kwargs.get("zero_run_basis")
        or kwargs.get("reason")
        or criterion_basis
        or check
        or ""
    ).split()).strip()
    if not basis:
        return (
            "⚠️ TOOL_ARG_ERROR (verify_and_record): delegation_zero_run requires a "
            "non-empty zero_run_basis explaining the host-visible evidence or blocker."
        )
    receipt: dict[str, Any] = {
        "tool": "verify_and_record",
        "contract_kind": "delegation_zero_run",
        "status": "declared",
        "zero_run": True,
        "zero_run_decision": decision,
        "zero_run_basis": _bounded(basis, _RECEIPT_DECLARED_SUMMARY_CAP),
        "physical_run_started": False,
        "route": str(bootstrap.get("route_id") or ""),
        "work_order_fingerprint": str(bootstrap.get("work_order_fingerprint") or ""),
        "ts": utc_now_iso(),
    }
    if crit := str(criterion_id or "").strip():
        receipt["criterion_id"] = crit[:120]
    try:
        from ouroboros import delegate_custody as custody
        from ouroboros.delegate_recovery import unsettled_start_ids
        from ouroboros.subagent_bootstrap import _durable_zero_run_receipt

        drive_root = custody.custody_root(ctx)
        with custody.actor_decision_lock(drive_root, task_id):
            if custody.custody_log_unreadable(drive_root):
                return (
                    "⚠️ TOOL_ERROR (verify_and_record): zero_run_custody_unknown: "
                    "the custody event log exists but cannot be read, so the host "
                    "cannot prove that no physical start/run remains; no zero-run "
                    "receipt was written (custody_log_unreadable)."
                )
            gaps: set[str] = set()
            prior = _durable_zero_run_receipt(ctx, gap_reasons=gaps)
            if prior:
                return (
                    "⚠️ TOOL_ARG_ERROR (verify_and_record): delegation_zero_run already "
                    "has a durable terminal decision for this actor."
                )
            blockers = unsettled_start_ids(drive_root, task_id)
            if any(blockers.values()):
                return (
                    "⚠️ TOOL_ERROR (verify_and_record): zero_run_requires_settlement: "
                    "this task has an open run, ambiguous start invocation, or "
                    "undisposed physical result. blockers="
                    + json.dumps(blockers, ensure_ascii=False, sort_keys=True)
                )
            if gaps:
                # Empty physical custody is the authority needed to re-ground a
                # damaged zero-run stream.  Preserve the evidence gap on the new
                # typed receipt instead of turning recovery into a self-deadlock.
                receipt["regrounds_zero_run_authority"] = True
                receipt["prior_zero_run_evidence_gaps"] = sorted(gaps)
            # A zero-run is lifecycle authority for the actor, so keep it on the
            # canonical budget root rather than an ephemeral child drive.
            written = append_verification_receipt(
                canonical_data_root(ctx), task_id, receipt,
            )
    except Exception as exc:
        return (
            "⚠️ TOOL_ERROR (verify_and_record): zero_run_custody_unknown: "
            "the host could not atomically prove empty custody and claim the "
            f"zero-run decision; no receipt was written ({type(exc).__name__})."
        )
    if not written:
        return (
            "⚠️ TOOL_ERROR (verify_and_record): the delegation_zero_run receipt "
            "could not be durably written; no zero-run decision was recorded."
        )
    bootstrap["zero_run_decision"] = decision
    bootstrap["zero_run_basis"] = receipt["zero_run_basis"]
    bootstrap["zero_run_receipt_recorded"] = True
    bootstrap["exact_start_pending"] = False
    bootstrap.pop("zero_run_evidence_status", None)
    bootstrap.pop("zero_run_evidence_gaps", None)
    return (
        "verify_and_record [delegation_zero_run] "
        f"{decision.upper()}: no physical leaf was started; typed host receipt recorded."
    )


def _verify_and_record(
    ctx: ToolContext,
    contract_kind: str = "",
    criterion_id: str = "",
    check: Any = None,
    expected: str = "",
    cwd: str = "",
    _resolved_binding: ResolvedResourceBinding | None = None,
    **kwargs,
) -> str:
    expected_match = kwargs.get("expected_match", "substring")
    artifact_paths = kwargs.get("artifact_paths")
    timeout_sec = kwargs.get("timeout_sec")
    criterion_source = kwargs.get("criterion_source", "")
    criterion_basis = kwargs.get("criterion_basis", "")
    bucket = str(kwargs.get("bucket") or "")
    skill_name = str(kwargs.get("skill_name") or "")
    kind = str(contract_kind or "").strip()
    if kind not in _CONTRACT_KINDS:
        return f"⚠️ TOOL_ARG_ERROR (verify_and_record): contract_kind must be one of {', '.join(_CONTRACT_KINDS)}."
    if active_tool_profile(ctx) == "local_readonly_subagent":
        # Read-only actor-first sessions need a host-owned zero-run disclosure,
        # not the general command/artifact verification surface.  Keep the
        # profile and lifecycle checks here as defense in depth for stale schemas
        # or forced dispatches after the physical leaf has started.
        bootstrap = getattr(ctx, "_configured_actor_bootstrap", None)
        if (
            kind != "delegation_zero_run"
            or not isinstance(bootstrap, dict)
            or bool(bootstrap.get("physical_started"))
        ):
            return (
                "⚠️ LOCAL_READONLY_SUBAGENT_BLOCKED: this profile may record only a "
                "delegation_zero_run receipt before its selected physical leaf starts."
            )
    match_mode = str(expected_match or "substring").strip().lower() or "substring"
    if match_mode not in _EXPECTED_MATCH_KINDS:
        return f"⚠️ TOOL_ARG_ERROR (verify_and_record): expected_match must be one of {', '.join(_EXPECTED_MATCH_KINDS)}."
    if match_mode == "bytes_equal" and kind not in _RUN_KINDS:
        # Silently accepting-and-ignoring the comparison would fake a stronger
        # verification than actually ran (adversarial r1 soft finding).
        return (
            "⚠️ TOOL_ARG_ERROR (verify_and_record): expected_match=bytes_equal only applies to "
            f"run-kind checks ({', '.join(_RUN_KINDS)}) — with contract_kind={kind} no comparison "
            "would run. Use a run kind (check may be as simple as ['true'])."
        )
    if match_mode == "bytes_equal" and str(expected or "").strip():
        # Receipt honesty (scope r6): `expected` is never consulted in bytes_equal
        # mode — recording it beside matched=true would read as a checked substring.
        return (
            "⚠️ TOOL_ARG_ERROR (verify_and_record): expected_match=bytes_equal takes NO `expected` "
            "string — the verdict is the byte-parity of artifact_paths=[a, b]. Drop `expected`, "
            "or use a substring/exact mode to match output text."
        )
    task_id = str(getattr(ctx, "task_id", "") or "")
    drive_root = getattr(ctx, "drive_root", None)
    expected_s = str(expected or "").strip()
    if kind == "delegation_zero_run":
        # This is a typed actor-first disclosure, not a general-purpose way to
        # relabel an ordinary task.  The private bootstrap marker is created by
        # the host before the first model episode and is intentionally the only
        # authority that can mint this receipt.
        bootstrap = getattr(ctx, "_configured_actor_bootstrap", None)
        if not isinstance(bootstrap, dict):
            return (
                "⚠️ TOOL_ARG_ERROR (verify_and_record): delegation_zero_run is only "
                "available to a configured actor-first session."
            )
        return _record_delegation_zero_run(
            ctx,
            bootstrap=bootstrap,
            task_id=task_id,
            criterion_id=criterion_id,
            criterion_basis=criterion_basis,
            check=check,
            kwargs=kwargs,
        )
    receipt: dict[str, Any] = {"tool": "verify_and_record", "contract_kind": kind, "expected": expected_s, "expected_match": match_mode, "ts": utc_now_iso()}
    crit = str(criterion_id or "").strip()
    if crit:
        receipt["criterion_id"] = crit[:120]
    # v6.54.4 criterion provenance (FLAG-ONLY, status never changes): where did
    # this success criterion come from — stated by the task, or synthesized by
    # the agent? agent_defined receipts surface to the acceptance reviewer and
    # feed a one-shot advisory nudge (mirrors the masked-verification pattern).
    # Default agent_defined: an UNDECLARED provenance must not read as task-stated.
    _source = str(criterion_source or "").strip().lower()
    receipt["criterion_source"] = _source if _source in ("task_stated", "agent_defined") else "agent_defined"
    _basis = " ".join(str(criterion_basis or "").split()).strip()
    if _basis:
        receipt["criterion_basis"] = _basis[:500]

    if kind in _RUN_KINDS:
        argv = _normalize_check(check)
        if not argv:
            return (
                f"⚠️ TOOL_ARG_ERROR (verify_and_record): contract_kind={kind} requires `check` "
                "(the verification command as argv list or a shell one-liner string)."
            )
        if match_mode == "bytes_equal" and len([p for p in (artifact_paths or []) if str(p or "").strip()]) != 2:
            return (
                "⚠️ TOOL_ARG_ERROR (verify_and_record): expected_match=bytes_equal requires "
                "artifact_paths=[<file_a>, <file_b>] — exactly two files to compare byte-for-byte "
                "after the check runs."
            )
        from ouroboros.tools.process_facts import (
            active_resolved_runtime as _active_resolved_runtime,
            publish_process_facts as _publish_process_facts,
        )
        from ouroboros.tools.shell import (
            _RUN_SHELL_DEFAULT_TIMEOUT_SEC,
            _executor_can_run_cwd,
            _resolve_effective_timeout,
            _shell_env_for_cwd,
            _tracked_subprocess_run,
        )
        from ouroboros.workspace_executor import execute as executor_execute

        try:
            binding = _resolved_binding or build_resolved_resource_binding(
                ctx,
                operation="shell",
                process_cwd=cwd,
                bucket=bucket,
                skill_name=skill_name,
            )
            work_dir = pathlib.Path(binding.target_path)
        except (OSError, ValueError) as exc:
            return f"⚠️ VERIFY_CWD_BLOCKED: check cwd escapes allowed roots: {exc}."
        receipt["resource_binding"] = {
            "root": binding.root,
            "base_path": str(binding.base_path),
            "target_path": str(binding.target_path),
            "source": binding.source,
            "skill_name": binding.skill_name,
        }
        timeout = _resolve_effective_timeout(_RUN_SHELL_DEFAULT_TIMEOUT_SEC, ctx, override_sec=timeout_sec)
        bootstrap_process_path()  # mirror run_command: ensure the check sees the full PATH
        use_executor = _executor_can_run_cwd(ctx, pathlib.Path(work_dir))
        # R4: the node resolver never rewrites args["check"] — the ORIGINAL argv
        # stays the receipt's identity below — so the resolver-chosen runtime
        # (bundled node when the PATH candidate is missing/probe-dead) reaches
        # execution through the per-call attestation instead. The emergency
        # PATH prepend rides the same attestation; both are None/no-op on every
        # healthy path, keeping argv, env, and receipts byte-identical.
        node_resolution = active_node_resolution(ctx)
        exec_argv = list(argv)
        if (
            node_resolution is not None
            and node_resolution.changed
            and exec_argv
            and exec_argv[0] == node_resolution.requested_interpreter
        ):
            exec_argv[0] = node_resolution.resolved_interpreter
        # D6 disclosure facts (stream B): how long the check's process lived,
        # and — when the resolver substituted the physical executable (the
        # registry bridges the stream-A attestation into
        # ``ctx._process_resolved_runtime``) — which runtime actually ran.
        # Disclosure ONLY: receipt identity/reconciliation (`check` text) is
        # untouched, and the agent-visible result text below is unchanged.
        _check_started_ts = time.monotonic()
        _resolved_runtime = _active_resolved_runtime(ctx)
        try:
            if use_executor:
                # Route the check through the host-owned executor backend (e.g. docker_exec
                # with NetworkMode=none) EXACTLY like run_command, so the verification runs
                # in the SAME place + isolation as the agent's other commands — not on the
                # host while the work lives in a container.
                res = executor_execute(
                    ctx, exec_argv, pathlib.Path(work_dir), timeout,
                    env_overlay=interpreter_path_overlay(node_resolution),
                )
            else:
                run_env = apply_env_path_prepend(
                    _shell_env_for_cwd(ctx, pathlib.Path(work_dir)), node_resolution,
                )
                res = _tracked_subprocess_run(
                    exec_argv, cwd=str(work_dir),
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=timeout,
                    **({"env": run_env} if run_env is not None else {}),
                )
        except subprocess.TimeoutExpired:
            # The check is a child of THIS call, so its death is published on the
            # typed process-facts channel exactly like run_command's, and the
            # receipt discloses the SAME published facts (one derivation).
            _facts = _publish_process_facts(
                started_ts=_check_started_ts, resolved_runtime=_resolved_runtime,
                timed_out=True, killed_by_host=True,
            )
            # Same renderer as the completed-run receipt below, so it carries the same
            # stamp: a timeout red must be reconcilable by the later green of that argv.
            receipt.update({"status": "fail", "returncode": None, "matched": False, "check": shlex.join(argv), "check_rendering": CHECK_RENDERING_SHLEX_JOIN, "summary": f"check timed out after {timeout}s"})
            receipt.update({k: v for k, v in _facts.items() if k in _RECEIPT_PROCESS_KEYS})
            if not append_verification_receipt(drive_root, task_id, receipt):
                return _receipt_custody_failure(
                    kind, f"check timed out after {timeout}s",
                )
            return (
                f"verify_and_record [{kind}] FAIL: check timed out after {timeout}s. "
                f"root={binding.root}, cwd={binding.target_path}. Receipt recorded."
            )
        # Full output captured in-handler BEFORE any transport truncation.
        out = (res.stdout or "") + (("\n" + res.stderr) if res.stderr else "")
        rc = res.returncode
        # Same typed publication as the timeout branch: the check's own exit and
        # (POSIX) signal, measured here where the truth is, not re-read from the
        # ``exit=`` the result text renders for the agent.
        _facts = _publish_process_facts(
            returncode=rc, started_ts=_check_started_ts,
            resolved_runtime=_resolved_runtime,
        )
        if match_mode == "bytes_equal":
            # v6.60.0: after the check, the VERDICT is the byte-parity of the two
            # declared files (golden-file shape); the check's own exit still gates.
            _paths2 = [str(p) for p in (artifact_paths or []) if str(p or "").strip()]
            matched, _cmp_detail = _compare_files_bytes_equal(
                ctx, _paths2, pathlib.Path(work_dir), use_executor=use_executor
            )
            out = (out + "\n\n[bytes_equal] " + _cmp_detail).strip()
        else:
            matched = (not expected_s) or _expected_matches(out, expected_s, match_mode)
        passed = (rc == 0) and matched
        # `shlex.join`, never `" ".join` (v6.78.0): this text is the verification's
        # IDENTITY downstream (`_outcome_receipts.receipt_canonical_identity`), and a
        # space-join is not injective — argv `["echo","a b"]` and `["echo","a","b"]`
        # rendered identically, so a green on one could clear a red on the other.
        # `shlex.join` is the exact inverse of the lexer that identity re-tokenizes with
        # (`shell_parse.shell_tokens`), so the stored text round-trips back to this argv.
        # The rendering is STAMPED beside the text (round 8): changing the renderer without
        # recording which one ran re-opened the very collision above ACROSS versions — an
        # old space-joined `echo a b` and a new `shlex.join` of a different argv read the
        # same. The stamp makes the two incomparable instead of falsely equal.
        receipt.update({"status": "pass" if passed else "fail", "returncode": rc, "matched": bool(matched), "check": shlex.join(argv), "check_rendering": CHECK_RENDERING_SHLEX_JOIN, "summary": _bounded(out, _RECEIPT_OUTPUT_CAP)})
        # D6/D7 disclosure keys (node-runtime sprint stream B), now copied from
        # the one publication above — see _RECEIPT_PROCESS_KEYS for which keys
        # and why. Disclosure-only: receipt IDENTITY (`check` text) and
        # reconciliation read none of them.
        receipt.update({k: v for k, v in _facts.items() if k in _RECEIPT_PROCESS_KEYS})
        # C: after-only artifact-lifecycle FLAG (status unchanged — flag-only). If the agent
        # declared artifact_paths, record whether each still exists after the check, probed via
        # the SAME surface as the check, so a build-then-delete is visible to the advisory reviewer.
        _decl = [str(p) for p in (artifact_paths or []) if str(p or "").strip()]
        if _decl:
            _lifecycle, _missing_after = _probe_artifact_lifecycle(ctx, _decl, pathlib.Path(work_dir), use_executor=use_executor)
            if _lifecycle:
                receipt["artifact_lifecycle"] = _lifecycle
            if _missing_after:
                receipt["artifacts_missing_after"] = _missing_after
        # Exit-masking sensor (v6.52.2, FLAG-ONLY — status unchanged): record when the check's own
        # shell pipeline can launder the real exit code (e.g. `... | tail`, `|| true`). Surfaced to
        # the advisory reviewer + a one-shot nudge so a PASS over a masked check is reconsidered.
        _masked, _mask_reasons = _check_has_exit_masking(argv)
        if _masked:
            receipt["check_exit_masking"] = True
            receipt["check_exit_masking_reasons"] = _mask_reasons
        if not append_verification_receipt(drive_root, task_id, receipt):
            return _receipt_custody_failure(
                kind, f"verdict={'PASS' if passed else 'FAIL'}, exit={rc}",
            ) + f"\n\n{_bounded(out, _TOOL_OUTPUT_CAP)}"
        verdict = "PASS" if passed else "FAIL"
        exp_note = f" expected={expected_s!r}" if expected_s else ""
        return (
            f"verify_and_record [{kind}] {verdict}: exit={rc}{exp_note}; "
            f"root={binding.root}, cwd={binding.target_path}. Host-attested receipt recorded.\n\n"
            f"{_bounded(out, _TOOL_OUTPUT_CAP)}"
        )

    if kind == "artifact_observation":
        paths = [str(p) for p in (artifact_paths or []) if str(p or "").strip()]
        obs_status, detail = _observe_artifacts(ctx, paths)
        receipt.update({"status": obs_status, "paths": paths[:20], "summary": detail})
        if not append_verification_receipt(drive_root, task_id, receipt):
            return _receipt_custody_failure(kind, f"{obs_status}: {detail}")
        # refused_out_of_scope is a POLICY block, not a verification failure — surface it
        # honestly (not a red FAIL) so a deliverable outside the observable roots doesn't
        # force the agent to declare no_visible_machine_contract (v6.57.0).
        if obs_status == "refused_out_of_scope":
            return (
                "verify_and_record [artifact_observation] REFUSED_OUT_OF_SCOPE: "
                f"{detail}. Not a failure — the path is outside this profile's "
                "observable roots (the active workspace plus whichever of "
                "subagent_projects / deliverables / artifact_store / task_drive "
                "the active profile can read). Receipt recorded."
            )
        verdict = "OBSERVED" if obs_status == "observed" else "FAIL"
        return f"verify_and_record [artifact_observation] {verdict}: {detail}. Host-attested receipt recorded."

    # no_visible_machine_contract: an honest escape hatch — no host run, the agent's
    # best proxy + residual risk is recorded as a receipt and judged by a reviewer.
    # The agent's own text, not a render of an argv that ran — its own rendering stamp.
    receipt.update({"status": "declared", "check": str(check or ""), "check_rendering": CHECK_RENDERING_DECLARED_TEXT, "summary": _bounded(expected_s or str(check or ""), _RECEIPT_DECLARED_SUMMARY_CAP)})
    if not append_verification_receipt(drive_root, task_id, receipt):
        return _receipt_custody_failure(kind, receipt["summary"])
    return (
        "verify_and_record [no_visible_machine_contract] DECLARED: no host-checkable contract; "
        "your stated proxy + residual risk recorded as a receipt for review."
    )


def get_tools() -> List[ToolEntry]:
    return [
        ToolEntry("verify_and_record", {
            "name": "verify_and_record",
            "description": (
                "Verify your deliverable BEFORE claiming it is done, and record a durable host-attested "
                "receipt. The host RUNS your declared check and attests the result — one call replaces the "
                "verification run you would do anyway. Pick contract_kind: visible_verifier / explicit_command "
                "(run `check`, pass on exit 0 and, if given, `expected` substring present) · explicit_metric "
                "(run `check`, pass when the `expected` metric string appears) · artifact_observation (the host "
                "confirms the declared artifact_paths exist) · no_visible_machine_contract (honest escape hatch: "
                "no machine check exists; your best proxy + risk is recorded for review) · delegation_zero_run "
                "(configured session actors only: record a typed incomplete/unknown decision when no "
                "physical leaf was started, with zero_run_decision and zero_run_basis). Recording a receipt "
                "suppresses the receipt_absent transparency flag on a clean turn. ANTI-CHEAT: verify ONLY against "
                "PUBLIC task info — the instruction text, examples embedded in it, installed oracles, and your own "
                "independent checks. NEVER read a hidden /tests/ dir, solution.sh, copied verifier code, or look up "
                "the answer online."
            ),
            "parameters": {"type": "object", "properties": {
                "contract_kind": {"type": "string", "enum": list(_CONTRACT_KINDS), "description": "How the deliverable is verifiable — you declare it (the host never guesses)."},
                "criterion_id": {"type": "string", "default": "", "description": "Optional id of the task_contract acceptance claim this receipt supports. Use ids from task_contract.acceptance_claims when present."},
                "criterion_source": {"type": "string", "enum": ["task_stated", "agent_defined"], "default": "agent_defined", "description": "Where this success criterion came from: task_stated (the task/instructions state it) or agent_defined (you synthesized it). Flag-only honesty — an agent_defined criterion asks you to double-check it is equivalent to what the task actually requires."},
                "criterion_basis": {"type": "string", "default": "", "description": "Optional one-line basis for an agent_defined criterion: why this check is sufficient evidence for the task's real requirement."},
                "zero_run_decision": {"type": "string", "enum": list(_ZERO_RUN_DECISIONS), "description": "For delegation_zero_run only: the actor's typed decision when no selected physical leaf was started."},
                "zero_run_basis": {"type": "string", "description": "For delegation_zero_run only: concise host-visible evidence or blocker; required and never inferred from final prose."},
                "check": {"description": "The verification command: an argv list (['pytest','-q']) or a shell one-liner string. Required for visible_verifier/explicit_command/explicit_metric.", "type": ["array", "string"], "items": {"type": "string"}},
                "expected": {"type": "string", "default": "", "description": "Optional expected substring/metric in the check output (explicit_command/explicit_metric)."},
                "expected_match": {"type": "string", "enum": list(_EXPECTED_MATCH_KINDS), "default": "substring", "description": "How `expected` is matched: substring (default) · exact (whole stripped output equals expected) · exact_line (expected equals one stripped output line) · json_equals (output and expected parse to equal JSON, key-order tolerant) · bytes_equal (after the check runs, artifact_paths=[a, b] are compared BYTE-FOR-BYTE — golden files, migration parity; the receipt records a bounded hexdump of the first divergence). Use a stricter mode when the task gives a worked example / exact output."},
                "artifact_paths": {"type": "array", "items": {"type": "string"}, "description": "Deliverable paths. For artifact_observation the host confirms they exist (existence/size only, never content) — observable roots are the active workspace plus every resource root the ACTIVE profile can already read (for orchestrating parents that includes subagent_projects and deliverables, so a parent CAN confirm a child's deliverable in the projects tree; child/readonly profiles lack those roots); a path outside these is a non-fatal refused_out_of_scope, not a failure. For run-kind checks (visible_verifier/explicit_command/explicit_metric) the host ALSO probes (after the check) whether each declared path that is RELATIVE to the check's working directory (cwd) still exists and records an advisory artifact_lifecycle flag — catching a check that built then deleted its own deliverable."},
                "cwd": {
                    "type": "string",
                    "default": "",
                    "description": (
                        "For run-kind checks, omit for active_workspace; use system_repo[/subdir] "
                        "for Ouroboros or skill_payload[/subdir] with bucket+skill_name for a skill."
                    ),
                },
                "bucket": {
                    "type": "string",
                    "enum": ["external", "clawhub", "ouroboroshub", "user_repo"],
                    "description": "Physical skill location for a run-kind cwd=skill_payload[/subdir].",
                },
                "skill_name": {
                    "type": "string",
                    "description": "Exact skill identity for a run-kind cwd=skill_payload[/subdir].",
                },
                "timeout_sec": {"type": "integer", "description": "Optional check timeout override."},
            }, "required": ["contract_kind"]},
        }, _verify_and_record, is_code_tool=True, timeout_sec=900, mutates_worktree=True),
    ]
