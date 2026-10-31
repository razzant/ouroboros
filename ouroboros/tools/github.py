"""GitHub tools: ``github_read``, ``github_write`` and ``pr_merge`` over one ``gh`` transport.

The read methods live in ``github_read``, the write methods and their receipt in ``github_write``,
the checks reader in ``github_checks`` and the merge receipt contract in ``merge_receipts``; this
module owns the transport, the repository binding and the registration.
``_gh_run`` is the structured transport read; ``_gh_cmd`` projects it onto the string ABI.
``GH_TARGET_INVALID`` / ``GH_TARGET_REQUIRED`` refusals enter the tool-result sidecar. A target
or missing-CLI refusal raised before the invocation's first ``gh`` launch is published
``completed_no_effect``. The checks reader's literal ``gh api`` annotation path is the one
subcall that does not take the explicit repo, alongside workflow literal API paths.
Exact captures stream stdout to a capped file with size/sha256 and retain CRLF and final bytes;
the cap bounds this destination, not gh internal downloads/cache. Stderr is drained and bounded.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import pathlib
import re
import subprocess
import threading
from contextvars import ContextVar
from dataclasses import dataclass, replace
from functools import wraps
from typing import List, Optional

from ouroboros.secret_masking import redact_known_values
from ouroboros.tools.registry import ToolContext, ToolEntry
from ouroboros.tools.tool_result import LegacyTextResultAdapter, ToolResult, _publish_tool_result, _replace_tool_result
from ouroboros.utils import truncate_within_limit

log = logging.getLogger(__name__)
_GENERIC_TRANSPORT = object()
# One public tool invocation's "a gh subprocess was already launched" fact.
_SUBMITTED: ContextVar[Optional[List[bool]]] = ContextVar("github_invocation_submitted", default=None)


def _canonical_github_args(args: dict) -> dict:
    """Canonical constrained values, shared by compatibility admission and direct dispatch."""
    for key in ("method", "kind", "state"):
        if key in args:
            args[key] = str(args[key] or "").strip()
    return args


def _one_invocation(handler):
    """Scope ``_gh_run``'s first-submission fact to one public tool invocation."""
    @wraps(handler)
    def invoke(ctx, *args, **kwargs):
        token = _SUBMITTED.set([False])
        try:
            return handler(ctx, *args, **kwargs)
        finally:
            _SUBMITTED.reset(token)
    return invoke


# gh's own HTTP status shapes (see ``_gh_run``); the first match in stderr order wins.
_GH_STATUS_RE = re.compile(
    r"\(HTTP (\d{3})\)[ \t\r]*$"
    r"|^(?:[a-z][a-z ]*: )*HTTP (\d{3})(?::| \(|[ \t\r]*$)",
    re.MULTILINE,
)
# gh v2.86.0 ``canMerge`` (pkg/cmd/pr/merge) prints exactly these non-TTY refusals
# and exits 1 BEFORE ``merge()``: BLOCKED/BEHIND advise --auto then --admin; DIRTY
# advises --auto, then only without --repo a local conflict hint for the method.
_PR_MERGE_REFUSAL_RE = re.compile(
    r"X Pull request [\w.-]+/[\w.-]+#(?P<number>\d+) is not mergeable: (?:"
    r"(?:the base branch policy prohibits the merge|the head branch is not up to date with the base branch)\.\n"
    r"To have the pull request merged after all the requirements have been met, add the `--auto` flag\.\n"
    r"To use administrator privileges to immediately merge the pull request, add the `--admin` flag\."
    r"|the merge commit cannot be cleanly created\.\n"
    r"To have the pull request merged after all the requirements have been met, add the `--auto` flag\."
    r"(?:\nRun the following to resolve the merge conflicts locally:\n"
    r"  gh pr checkout (?P<checkout>\d+) && git fetch (?P<remote>\S+) (?P<base>\S+)"
    r" && git (?P<action>merge|rebase) (?P=remote)/(?P=base))?)"
)


@dataclass(frozen=True)
class GhCapture:
    """The stdout bytes one exact-capture ``_gh_run`` wrote to its file, as written: no decoding, no strip."""
    path: pathlib.Path
    size: int
    sha256: str
    # The byte cap ended the read and the process; the file holds the first ``size`` bytes of a longer stream.
    truncated: bool



@dataclass(frozen=True)
class GhResult:
    ok: bool
    text: str
    exit_code: int | None
    http_status: int | None
    # "target" is a local refusal and "deadline" a request the checks reader never sent;
    # neither is a subprocess exit or exception.
    failure: str
    # The exact branch's capture; None for the text transport and for every failure.
    capture: Optional[GhCapture] = None


# One pipe read; stderr of an exact capture is kept up to this many bytes for classification. A stderr that
# exceeds the cap is WITHHELD whole: any cut leaves a boundary a token can straddle, and redaction matches whole
# values only, so no bounded tail is safe to show. The exit code stays; the HTTP status is then unknown.
_CAPTURE_CHUNK = _CAPTURE_STDERR_CAP = 65536
_CAPTURE_STDERR_WITHHELD = b"[stderr exceeded 64 KiB; diagnostics withheld]"


def _merge_refused_before_effect(cmd: List[str], res: subprocess.CompletedProcess, stderr: str) -> bool:
    """Whether the complete stderr of the merge ``merge_receipts`` sends is gh's own pre-merge refusal.

    Only ``pr merge N --METHOD --match-head-commit SHA [--body-file -] [--repo R]``, exit 1 with empty
    stdout, the whole refusal for N and nothing else; a status, prefix or suffix proves nothing."""
    args = cmd[1:]
    if args[6:8] == ["--body-file", "-"]:
        args = args[:6] + args[8:]
    if (res.returncode != 1 or res.stdout.strip() or args[:2] != ["pr", "merge"] or len(args) not in (6, 8)
            or args[3] not in ("--merge", "--squash", "--rebase") or args[4] != "--match-head-commit"
            or args[6:7] not in ([], ["--repo"])):
        return False
    match = _PR_MERGE_REFUSAL_RE.fullmatch(stderr.strip())
    if match is None or match["number"] != args[2]:
        return False
    return match["checkout"] is None or (len(args) == 6 and match["checkout"] == args[2]
                                         and match["action"] == ("rebase" if args[3] == "--rebase" else "merge"))


def _exit_failure(cmd: List[str], res: subprocess.CompletedProcess, stderr: str) -> GhResult:
    """Classify a nonzero exit from its complete stderr, then keep a bounded head.

    Redact the WHOLE stderr first (a cut could split a token), read gh's own status marker before any
    bounding, then keep a bounded head. gh writes the status in three deterministic shapes and nowhere
    else: ``gh: <msg> (HTTP NNN)`` at the end of a line (``gh api`` with a message), ``gh: HTTP NNN``
    (``gh api`` without one) and ``HTTP NNN: <msg> (<url>)`` — optionally wrapped as ``failed to fork:
    HTTP NNN: …`` — from every other command. A marker quoted mid-sentence is prose, not a status.
    HTTP status alone proves nothing about which CLI step failed."""
    err = redact_known_values(stderr or "", [github_token_from_env_or_settings()])
    status = _GH_STATUS_RE.search(err)
    head = " | ".join([line.strip() for line in err.splitlines() if line.strip()][:3])
    head = truncate_within_limit(head, 600)
    return GhResult(False, "⚠️ GH_ERROR: " + head, res.returncode,
                    int(status.group(1) or status.group(2)) if status else None,
                    "pre_effect" if _merge_refused_before_effect(cmd, res, err) else "exit")



def _capture_stdout(cmd: List[str], cwd: pathlib.Path, env: dict, timeout: int, path: pathlib.Path,
                    max_bytes: int) -> tuple:
    """Stream the process's stdout into ``path`` up to ``max_bytes``: (GhCapture, returncode, stderr, timed_out).

    The cap stops stdout acquisition into our destination, not gh's internal network/cache work.
    Stderr is drained on its own thread into a bounded buffer so neither pipe blocks the
    other; the deadline kills the process from a timer. The file holds exactly the bytes hashed. The sink
    is opened before the launch: an unwritable path fails here, never after a process started."""
    sink = open(path, "wb")
    try:
        proc = subprocess.Popen(cmd, cwd=str(cwd), env=env, stdin=subprocess.DEVNULL,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except BaseException:
        sink.close()
        raise
    err_parts: List[bytes] = []
    err_total = [0]
    timed_out = [False]

    def drain_stderr() -> None:
        kept = 0
        while True:
            chunk = proc.stderr.read(_CAPTURE_CHUNK)
            if not chunk:
                return
            err_total[0] += len(chunk)
            if kept < _CAPTURE_STDERR_CAP:
                err_parts.append(chunk[:_CAPTURE_STDERR_CAP - kept])
                kept += len(err_parts[-1])

    def expire() -> None:
        timed_out[0] = True
        proc.kill()

    drain, timer = threading.Thread(target=drain_stderr, daemon=True), threading.Timer(timeout, expire)
    digest, written, truncated = hashlib.sha256(), 0, False
    try:
        drain.start()
        timer.start()
        with sink:
            while True:
                chunk = proc.stdout.read(_CAPTURE_CHUNK)
                if not chunk:
                    break
                keep = chunk[:max(0, max_bytes - written)]
                if keep:
                    sink.write(keep)
                    digest.update(keep)
                    written += len(keep)
                if len(keep) < len(chunk):
                    truncated = True
                    proc.kill()
                    break
        proc.stdout.close()
        returncode = proc.wait()
    finally:
        timer.cancel()
        if proc.poll() is None:
            proc.kill()
            proc.wait()
        drain.join(timeout=5)
        proc.stderr.close()
    raw = _CAPTURE_STDERR_WITHHELD if err_total[0] > _CAPTURE_STDERR_CAP else b"".join(err_parts)
    stderr = raw.decode("utf-8", errors="replace")
    return GhCapture(path, written, digest.hexdigest(), truncated), returncode, stderr, timed_out[0]



def _refuse(ctx: ToolContext, text: str, code: str = "TOOL_ARG_ERROR", *, no_effect: bool = False) -> str:
    """Publish a refusal this module AUTHORS as a typed result; text unchanged.

    The registry types a string result by its first-line typed marker (the
    warning sign plus an UPPER_SNAKE code), so prose such as ``⚠️ issue number must be positive`` was recorded ``status=ok``
    although the producer already knew it had refused. Both codes carry
    ``status="error"``."""
    return _publish_tool_result(ctx, ToolResult(status="error", code=code, text=text,
        meta={"operation_outcome": "completed_no_effect"} if no_effect else {}))


def github_token_from_env_or_settings() -> str:
    from ouroboros.config import load_settings
    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN") or ""
    if not token:
        try:
            token = load_settings().get("GITHUB_TOKEN", "")
        except Exception:
            token = ""
    return str(token or "").strip()


def _gh_env(ctx: ToolContext) -> dict:
    env = os.environ.copy()
    token = github_token_from_env_or_settings()
    if token:
        env["GH_TOKEN"] = token
        env["GITHUB_TOKEN"] = token
    return env


def github_cli_configured() -> bool:
    """Local credential configuration, not a live authentication assertion."""
    if github_token_from_env_or_settings():
        return True
    config_dir = os.environ.get("GH_CONFIG_DIR", "")
    if not config_dir:
        base = os.environ.get("XDG_CONFIG_HOME", "")
        config_dir = str(pathlib.Path(base) / "gh") if base else ""
    if not config_dir:
        from ouroboros.platform_layer import IS_WINDOWS

        app_data = os.environ.get("APPDATA", "") if IS_WINDOWS else ""
        config_dir = str(pathlib.Path(app_data) / "GitHub CLI") if app_data else str(pathlib.Path.home() / ".config" / "gh")
    try:
        import yaml

        hosts = yaml.safe_load((pathlib.Path(config_dir) / "hosts.yml").read_text(encoding="utf-8"))
        return isinstance(hosts, dict) and any(
            isinstance(host, dict) and bool(host.get("user") or host.get("users") or host.get("oauth_token"))
            for host in hosts.values()
        )
    except (OSError, ValueError, yaml.YAMLError):
        return False


def _gh_run(args: List[str], ctx: ToolContext, timeout: int = 30, input_data: Optional[str] = None,
            *, repo: object = _GENERIC_TRANSPORT, capture_to: Optional[pathlib.Path] = None,
            max_bytes: int = 0) -> GhResult:
    # Only omitted internal API/Hub calls keep the generic transport contract.
    # Public repository tools always pass repo, including '' for Project focus.
    # The target refusals below publish a typed argument error into the calling
    # tool's sidecar; the publication transport omits `repo`, so it can never
    # reach them and its own final result is never shadowed from here.
    # A refusal before this invocation's FIRST gh launch also attests that the
    # invocation had no effect; after any launch it attests nothing.
    submitted = _SUBMITTED.get()
    first = submitted is not None and not submitted[0]

    def refused(text: str, failure: str, typed: Optional[ToolResult] = None) -> GhResult:
        if first:
            typed = _replace_tool_result(typed or LegacyTextResultAdapter.from_text("", text),
                                         meta_updates={"operation_outcome": "completed_no_effect"})
        return GhResult(False, _publish_tool_result(ctx, typed) if typed else text, None, None, failure)

    if capture_to is not None and (input_data is not None or type(max_bytes) is not int or max_bytes <= 0):
        raise ValueError("an exact capture takes a positive max_bytes and no stdin")
    if repo is not _GENERIC_TRANSPORT and not isinstance(repo, str):
        return refused("", "target", ToolResult(
            status="error", code="TOOL_ARG_ERROR",
            text="⚠️ GH_TARGET_INVALID: repo must be a string; omit it to use the selected Project.",
        ))
    try:
        cwd, env = pathlib.Path(ctx.repo_dir), _gh_env(ctx)
        cmd = ["gh", *args]
        if repo is not _GENERIC_TRANSPORT:
            from ouroboros.tool_access import build_resolved_resource_binding

            metadata = getattr(ctx, "task_metadata", {})
            metadata = metadata if isinstance(metadata, dict) else {}
            workspace = getattr(ctx, "workspace_root", None)
            room_dir = str(metadata.get("_project_room_dir") or "")
            project = str(getattr(ctx, "project_id", "") or "")
            if not repo:
                note = str(metadata.get("_project_room_note") or "")
                selected = workspace or room_dir
                if note or (selected and not pathlib.Path(selected).is_dir()):
                    return refused(
                        f"⚠️ GH_TARGET_UNAVAILABLE: {note or 'The selected Project directory is unavailable.'}",
                        "target")
                if project and not selected:
                    return refused("", "target", ToolResult(
                        status="error", code="TOOL_ARG_ERROR",
                        text="⚠️ GH_TARGET_REQUIRED: this Project has no repository directory; pass repo='[HOST/]OWNER/REPO'.",
                    ))
            binding = build_resolved_resource_binding(ctx, operation="shell", process_cwd="")
            cwd = binding.target_path
            if workspace and cwd != pathlib.Path(workspace).resolve(strict=False):
                return refused("⚠️ GH_TARGET_UNAVAILABLE: the task's Project binding could not be resolved.",
                               "target")
            if workspace or room_dir or project:
                env.pop("GH_REPO", None)  # Ambient defaults cannot replace the selected Project.
            if repo and args[:1] != ["api"]:  # API routes carry the repository.
                if args[:2] == ["repo", "view"]:
                    # `gh repo view [<repository>] [flags]` names its target positionally; its INHERITED FLAGS list
                    # only --help (gh 2.101 help), so `--repo` would be an unknown flag. Every other subcommand
                    # this module sends lists `-R, --repo [HOST/]OWNER/REPO`.
                    cmd[3:3] = [repo]
                else:
                    cmd.extend(["--repo", repo])
        if submitted is not None:
            submitted[0] = True
        if capture_to is not None:
            capture, returncode, stderr, timed_out = _capture_stdout(cmd, cwd, env, timeout, capture_to, max_bytes)
            if timed_out:
                return GhResult(False, f"⚠️ GH_TIMEOUT: exceeded {timeout}s.", None, None, "timeout")
            if capture.truncated:  # The cap ended the process; its exit says nothing about the command.
                return GhResult(True, "", None, None, "", capture)
            if returncode != 0:
                return _exit_failure(cmd, subprocess.CompletedProcess(cmd, returncode, "", stderr), stderr)
            return GhResult(True, "", returncode, None, "", capture)
        res = subprocess.run(
            cmd,
            cwd=str(cwd),
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=timeout,
            input=input_data,
            env=env,
        )
        if res.returncode != 0:
            return _exit_failure(cmd, res, res.stderr or "")
        return GhResult(True, res.stdout.strip(), res.returncode, None, "")
    except FileNotFoundError as e:
        missing = str(getattr(e, "filename", "") or "")
        if not missing or pathlib.Path(missing).name == "gh":
            return refused(
                "⚠️ GH_ERROR: `gh` CLI not found. Install GitHub CLI and ensure it is on PATH (https://cli.github.com/)",
                "cli_missing")
        detail = truncate_within_limit(redact_known_values(str(e), [github_token_from_env_or_settings()]), 600)
        return GhResult(False, f"⚠️ GH_ERROR: {detail}", None, None, "exception")
    except subprocess.TimeoutExpired:
        return GhResult(False, f"⚠️ GH_TIMEOUT: exceeded {timeout}s.", None, None, "timeout")
    except Exception as e:
        detail = truncate_within_limit(redact_known_values(str(e), [github_token_from_env_or_settings()]), 600)
        return GhResult(False, f"⚠️ GH_ERROR: {detail}", None, None, "exception")


def _gh_cmd(args: List[str], ctx: ToolContext, timeout: int = 30, input_data: Optional[str] = None,
            *, repo: object = _GENERIC_TRANSPORT) -> str:
    return _gh_run(args, ctx, timeout=timeout, input_data=input_data, repo=repo).text


_REPO_RE = re.compile(r"^(?:(?P<host>[^/\s]+)/)?(?P<owner>[^/\s]+)/(?P<name>[^/\s]+)$")
_GRAPHQL_LITERALS = re.compile(r'"""(?:\\.|(?!""")[\s\S])*"""|"(?:\\.|[^"\\])*"|#[^\r\n]*')


def _graphql_operations(document: str) -> set[str]:
    """Operation keywords at definition starts, excluding strings, comments and selection sets."""
    code = _GRAPHQL_LITERALS.sub(" ", document)
    operations, braces, parentheses, start = set(), 0, 0, True
    for token in re.findall(r"[A-Za-z_]\w*|[{}()]", code):
        if start:
            if token in ("query", "mutation", "subscription"):
                operations.add(token)
            elif token == "{":
                operations.add("query")  # shorthand operation
            start = False
        if token == "(":
            parentheses += 1
        elif token == ")":
            parentheses -= 1
        elif not parentheses:
            if token == "{":
                braces += 1
            elif token == "}":
                braces -= 1
                start = braces == 0
    return operations


def _split_repo(repo: str) -> Optional[tuple]:
    """``(host, owner, name)`` of an explicit ``[HOST/]OWNER/REPO``; None when it is not one."""
    match = _REPO_RE.fullmatch(str(repo or "").strip())
    return (match["host"] or "", match["owner"], match["name"]) if match else None


def _no_launch_yet() -> bool:
    """Whether this public invocation has launched no ``gh`` yet, so a refusal can attest it had no effect.

    Only a public invocation scope proves it; a direct internal call attests nothing."""
    submitted = _SUBMITTED.get()
    return submitted is not None and not submitted[0]


def _target_refusal(ctx: ToolContext, text: str) -> GhResult:
    return GhResult(False, _refuse(ctx, text, no_effect=_no_launch_yet()), None, None, "target")


def _gh_api(ctx: ToolContext, route: str, *, repo: str, method: str = "GET", body: Optional[dict] = None,
            include: bool = False, timeout: int = 30) -> GhResult:
    """One REST call through ``gh api`` under the public repository binding.

    A bound Project (``repo=""``) keeps gh's ``{owner}/{repo}`` placeholders, which gh resolves
    from the bound directory exactly as ``gh issue list`` does; an explicit repo is substituted
    into the route and its host passed as ``--hostname``. ``include`` keeps the status line and
    headers in the output (pagination reads the ``Link`` header)."""
    args = ["api"]
    if method != "GET":
        args.extend(["--method", method])
    if include:
        args.append("--include")
    host = ""
    if repo:
        parts = _split_repo(repo)
        if parts is None:
            return _target_refusal(ctx, "⚠️ GH_TARGET_INVALID: repo must be [HOST/]OWNER/REPO; omit it to use the selected Project.")
        host, owner, name = parts
        route = route.replace("{owner}/{repo}", f"{owner}/{name}")
    args.append(route)
    if host:
        args.extend(["--hostname", host])
    input_data = None
    if body is not None:
        args.extend(["--input", "-"])
        input_data = json.dumps(body)
    return _gh_run(args, ctx, timeout=timeout, input_data=input_data, repo=repo)


def _gh_graphql(ctx: ToolContext, document: str, variables: dict, *, repo: str, repo_vars: bool = True,
                timeout: int = 30) -> GhResult:
    """One GraphQL call under the public repository binding; ``$owner``/``$name`` name the repository.

    The document and variables travel in one JSON envelope, preserving every variable's type."""
    args = ["api", "graphql", "--input", "-"]
    values = dict(variables or {})
    host = ""
    if repo:
        parts = _split_repo(repo)
        if parts is None:
            return _target_refusal(ctx, "⚠️ GH_TARGET_INVALID: repo must be [HOST/]OWNER/REPO; omit it to use the selected Project.")
        host, owner, name = parts
        if repo_vars:
            values = {"owner": owner, "name": name, **values}
    elif repo_vars:
        from ouroboros.tools.github_read import _Read

        slug, error = _Read(ctx, "api", repo).slug()
        if error:
            return GhResult(False, error, None, None, "target")
        owner, name = slug.split("/", 1)
        values = {"owner": owner, "name": name, **values}
    if host:
        args.extend(["--hostname", host])
    return _gh_run(args, ctx, timeout=timeout, repo=repo,
                   input_data=json.dumps({"query": document, "variables": values}))


def _send_facts(res: GhResult) -> str:
    """The facts of a failed gh send, never a verdict: gh's exit and HTTP status, a timeout, or a lost answer.

    A nonzero exit alone does not prove GitHub rejected the request (gh can fail after the server acted), so
    callers report the effect as unconfirmed and read it back instead of calling it refused."""
    if res.failure in ("exit", "pre_effect"):
        facts = ", ".join(filter(None, [f"exit {res.exit_code}" if res.exit_code is not None else "",
                                        f"HTTP {res.http_status}" if res.http_status else ""]))
        return f"ended with {facts or 'a nonzero exit'}: {res.text}"
    if res.failure == "timeout":
        return f"timed out: {res.text}"
    return f"failed without an answer ({res.failure}): {res.text}"



def _pr_merge(ctx: ToolContext, number: int, expected_head_sha: str, method: str,
              review_task_ids: Optional[List[str]] = None, reviewed_head_sha: str = "",
              reviewed_base_sha: str = "", review_scope: str = "full", review_verdict: str = "",
              review_record_id: str = "", repo: str = "", checks: Optional[List[dict]] = None) -> str:
    """Thin transport binding; the receipt contract lives in ``merge_receipts``."""
    from ouroboros.merge_receipts import REVIEW_SCOPES, _VERDICT_RE, _sha, run_pr_merge
    from ouroboros.tool_access import canonical_data_root
    from ouroboros.update_qualification import STATES

    if checks is not None and (not isinstance(checks, list) or any(
            not isinstance(row, dict) or not isinstance(row.get("name"), str) or not row["name"].strip()
            or row.get("status") not in STATES or not isinstance(row.get("detail", ""), str) for row in checks)):
        return _refuse(ctx, "⚠️ TOOL_ARG_ERROR: checks need name, status (passed, failed, not_run, unknown) and optional detail.",
                       no_effect=True)

    if review_scope not in REVIEW_SCOPES:
        return _refuse(ctx, "⚠️ TOOL_ARG_ERROR: review_scope is full or delta; nothing was merged.", no_effect=True)
    if review_verdict and not _VERDICT_RE.fullmatch(review_verdict):
        return _refuse(ctx, (
            f"⚠️ TOOL_ARG_ERROR: review_verdict must match {_VERDICT_RE.pattern}: a letter, then at most 39 "
            "letters, digits, spaces, '_' or '-' (e.g. PASS, PASS_WITH_ADVISORIES). The "
            f"{len(review_verdict)}-character value was refused, not shortened; nothing was merged."), no_effect=True)
    record_id = str(review_record_id or "").strip()
    declared = ({"reviewed_head_sha": _sha(reviewed_head_sha), "reviewed_base_sha": _sha(reviewed_base_sha),
                 "scope": review_scope, "verdict": review_verdict}
                if (reviewed_head_sha or review_verdict or (review_task_ids and not record_id)) else None)
    receipt = run_pr_merge(
        ctx, lambda args, **kw: _gh_run(args, ctx, repo=repo, **kw), lambda args, **kw: _gh_run(args, ctx, **kw),
        drive_root=canonical_data_root(ctx), task_id=str(ctx.task_id or ""), number=int(number or 0),
        expected_head_sha=expected_head_sha, method=method,
        review={"declared": declared, "task_ids": list(review_task_ids or []), "record_id": record_id, "checks": checks})
    if receipt.get("refused"):
        code = "TOOL_ARG_ERROR" if receipt["refused"] == "arguments" else "TOOL_ERROR"
        return _refuse(ctx, f"⚠️ PR_MERGE_REFUSED: {receipt['refused']} — {receipt.get('detail', '')}", code, no_effect=True)
    from ouroboros.merge_receipts import card_row_text, queued_earlier_note

    status = (receipt.get("outcome") or {}).get("status", "unknown")
    lines = [card_row_text(receipt), f"receipt_id={receipt['receipt_id']} (task result: merge_receipts)"]
    prior = receipt.get("prior_status", "unknown") if receipt.get("readback_only") else None
    if prior == "merged":
        lines.append("This PR's merge is already recorded in this receipt, so this call only read GitHub back "
                     "and sent no new merge request.")
    elif prior == "":
        lines.append("An earlier or concurrent request for this PR holds its merge intent with no recorded outcome "
                     "yet, so this call only read GitHub back and sent no new merge request.")
    elif prior is not None:
        lines.append("An earlier request for this PR had no confirmed outcome, so this call only read GitHub back "
                     "and sent no new merge request. Unknown and queued requests remain observation-only.")
    # The receipt's retained, already bounded and redacted gh facts belong in this local
    # response (#1555); the public PR-body block never carries them.
    effect = receipt.get("effect") or {}
    if effect and not effect.get("ok"):
        facts = ", ".join([f"failure={effect.get('failure') or 'unknown'}"] + [
            f"{key}={effect[key]}" for key in ("exit_code", "http_status") if effect.get(key) is not None])
        output = str(effect.get("output") or "").strip()
        lines.append(("Earlier merge request: " if receipt.get("readback_only") else "Merge request: ")
                     + facts + (f": {output}" if output else ""))
    if (receipt.get("outcome") or {}).get("reason"):
        # The first line already names the latest readback beside an earlier queue state.
        lines.append(("Earlier GitHub readback: " if queued_earlier_note(receipt) else "GitHub readback: ")
                     + f"{receipt['outcome']['reason']}.")
    publication = receipt.get("publication") or {}
    if receipt.get("receipt_write_gap"):
        lines.append("⚠️ Merge receipt persistence is unknown after the external effect: "
                     + receipt["receipt_write_gap"])
    if status in ("merged", "queued") and (publication.get("body") or {}).get("status") != "published":
        lines.append("⚠️ The PR-body receipt block was not confirmed; call pr_merge again to retry publication only.")
    card = publication.get("card") or {}
    if status in ("merged", "queued") and card.get("status") not in ("owed", "delivered"):
        lines.append("⚠️ The task-card receipt is not confirmed: " + str(card.get("reason") or "publication unknown")
                     + "; call pr_merge again for observation/publication only.")
    text = "\n".join(lines)
    outcome = "completed" if status == "merged" else "unknown"
    if status == "refused" and (receipt.get("effect") or {}).get("failure") in {"cli_missing", "target", "pre_effect"}:
        outcome = "completed_no_effect"
    return _publish_tool_result(ctx, ToolResult(
        status="ok" if status in ("merged", "queued") else "error",
        code="OK" if status in ("merged", "queued") else "TOOL_ERROR",
        text=text if status in ("merged", "queued") else f"⚠️ PR_MERGE_{status.upper()}: " + text,
        meta={"operation_outcome": outcome}))


def _github_read(ctx: ToolContext, method: str = "", repo: str = "", **args) -> str:
    """``github_read``: the methods live in ``github_read`` and call this module's transport."""
    from ouroboros.tools.github_read import run_read

    return run_read(ctx, method, repo, **args)


def _github_write(ctx: ToolContext, method: str = "", repo: str = "", **args) -> str:
    """``github_write``: the methods and the receipt live in ``github_write``; the transport is this module's."""
    from ouroboros.tools.github_write import run_write

    return run_write(ctx, method, repo, **args)


def get_tools() -> List[ToolEntry]:
    from ouroboros.tools.github_read import SCHEMA as READ_SCHEMA
    from ouroboros.tools.github_write import SCHEMA as WRITE_SCHEMA

    tools = [
        ToolEntry("github_read", READ_SCHEMA, _github_read),
        ToolEntry("github_write", WRITE_SCHEMA, _github_write),
        ToolEntry("pr_merge", {
            "name": "pr_merge",
            "description": (
                "Merge a GitHub pull request so a receipt exists: states the exact head you expect "
                "and the method (never auto-merge or admin), records the host review record you name or "
                "the review you declare beside what the host observes, reads GitHub back, and writes the "
                "receipt to this task's record, its card and the PR body. A missing review is recorded loudly, never a lock. "
                "An unknown or queued merge stays observation/publication-only on repeat calls; no resend. "
                "Distinct from stage_pr_merge, which stages a local merge for a reviewed commit."
            ),
            "parameters": {"type": "object", "properties": {
                "number": {"type": "integer", "description": "PR number"},
                "expected_head_sha": {"type": "string", "description": "The PR head you intend to merge; GitHub refuses if it moved"},
                "method": {"type": "string", "enum": ["merge", "squash", "rebase"]},
                "checks": {"type": "array", "description": "Execution declarations retained in the merge commit. Name full_battery and cost_contract explicitly; omissions stay unknown. Describe accepted FAILs and checks not run.",
                           "items": {"type": "object", "properties": {
                               "name": {"type": "string"},
                               "status": {"type": "string", "enum": ["passed", "failed", "not_run", "unknown"]},
                               "detail": {"type": "string"}}, "required": ["name", "status"]}},
                "review_task_ids": {"type": "array", "items": {"type": "string"}, "default": [],
                                    "description": "Task ids of the reviews you rely on; the host records whether each "
                                                   "completed, not its kind or reviewed subject"},
                "reviewed_head_sha": {"type": "string", "default": "", "description": "The head those reviews covered (declared)"},
                "reviewed_base_sha": {"type": "string", "default": "", "description": "The base those reviews covered (declared)"},
                "review_scope": {"type": "string", "enum": ["full", "delta"], "default": "full",
                                 "description": "delta = only the change since an earlier review; never counted as whole-PR coverage"},
                "review_verdict": {"type": "string", "default": "", "maxLength": 40,
                                   "description": "Your declared verdict, recorded as your statement, never as the host "
                                                  "review result; must match ^[A-Za-z][A-Za-z0-9_ -]{0,39}$, e.g. PASS"},
                "review_record_id": {"type": "string", "default": "",
                                     "description": "Id of the host review record to bind (e.g. the review_record_id commit_reviewed "
                                                    "returns); the host then reads the reviewed subject and verdict from it instead "
                                                    "of your declaration. An id with no record is refused before any merge"},
            }, "required": ["number", "expected_head_sha", "method"]},
        }, _pr_merge),
    ]
    # Wrapped here, not at definition: nested handler calls share one invocation.
    tools = [replace(entry, handler=_one_invocation(entry.handler)) for entry in tools]
    for entry in tools:
        entry.schema["parameters"]["properties"]["repo"] = {
            "type": "string", "default": "",
            "description": "Explicit [HOST/]OWNER/REPO. Omit for the active Project repository; required for a Project without a repository folder. An omitted HOST follows GitHub CLI host configuration.",
        }
    return tools
