"""Run-log and artifact methods of ``github_read``.

Internal method owner: ``github_read`` supplies arguments, paging and bounded JSON delivery. Every read first
resolves the repository from the URL GitHub returns for the run (``gh repo view`` for a bare artifact id), so a
log or artifact is identified by repository + id, never by name. Captured bytes stream to a file under an explicit
byte cap (``_gh_run``'s exact branch stops the process at the cap) and are retained once, write-once by digest, in
this task's artifact store with their sha256 and a ``read_file`` address. An artifact archive is retained as the
archive GitHub served and is never extracted here. Read-only towards GitHub: nothing is dispatched or re-run.
"""

from __future__ import annotations

import pathlib
import re
import tempfile
import zipfile
from typing import Any, Dict, List, Optional, Tuple

from ouroboros.tool_capabilities import tool_result_limit
from ouroboros.tools.github import _gh_run, _refuse
from ouroboros.tools.github_checks import _check_state, _gh_json, _one_line, _whole
from ouroboros.tools.registry import ToolContext
from ouroboros.tools.tool_result import ToolResult, _publish_tool_result

LOG_MAX_BYTES_DEFAULT, LOG_MAX_BYTES_CEILING = 4 * 1024 * 1024, 32 * 1024 * 1024
ARTIFACT_MAX_BYTES_CEILING = 64 * 1024 * 1024
_TAIL_LINES_DEFAULT, _TAIL_LINES_MAX, _LINE_CLIP = 120, 2000, 400
_ZIP_MEMBERS_SHOWN = 40
_READ_TIMEOUT, _LOG_TIMEOUT, _ARTIFACT_TIMEOUT = 30, 120, 180
_RESULT_MARGIN = 200  # Characters kept free under the result limit.
_REPO_URL_RE = re.compile(r"^https://([^/]+)/([^/]+)/([^/]+?)(?:/.*)?$")
_RUN_FIELDS = "databaseId,url,status,conclusion,attempt,headSha,workflowName"
_RUN_ARTIFACT_FIELDS = "databaseId,url,status,conclusion,headSha,workflowName"


def _deliver(read, text: str, payload: dict) -> str:
    line, _, view = text.partition("\n")
    return read.deliver(line, {"target": read.number(), "view": view, **payload})


def _completed_read(read, text: str, payload: dict) -> str:
    return _publish_tool_result(read.ctx, ToolResult(status="ok", code="OK", text=_deliver(read, text, payload),
                                                meta={"operation_outcome": "completed_no_effect"}))


def _incomplete_read(read, text: str, payload: dict) -> str:
    """Acquisition succeeded, but the required exact-byte retention did not."""
    return _publish_tool_result(read.ctx, ToolResult(status="error", code="TOOL_ERROR", text=_deliver(read, text, payload),
                                                meta={"operation_outcome": "incomplete"}))


def _repo_from_url(url: object) -> Optional[Tuple[str, str, str]]:
    """``(host, owner, name)`` of the URL GitHub returned, which names the repository the CLI resolved."""
    match = _REPO_URL_RE.match(str(url or ""))
    return match.groups() if match else None


def _resolve_run(ctx: ToolContext, run_id: int, attempt: int, repo: str, fields: str,
                 *, require_attempt: bool = True) -> Tuple[Optional[tuple], str]:
    """``(run, (host, owner, name), attempt)`` as GitHub records it, or the error.

    Log reads require a positive observed attempt and pin their later byte read to it. Artifact listings
    only require the run identity: GitHub's artifact API is scoped to the run across attempts."""
    args = ["run", "view", str(run_id)] + (["--attempt", str(attempt)] if attempt else []) + ["--json", fields]
    res = _gh_run(args, ctx, timeout=_READ_TIMEOUT, repo=repo)
    if not res.ok:
        return None, res.text
    run = _gh_json(res, dict)
    if run is None:
        return None, _refuse(ctx, f"⚠️ TOOL_ERROR: failed to parse the workflow run JSON: {res.text[:500]}", "TOOL_ERROR")
    observed_id, observed_attempt = _whole(run.get("databaseId")), _whole(run.get("attempt"))
    if observed_id != run_id or (require_attempt and (observed_attempt is None or observed_attempt < 1
                                                    or (attempt and observed_attempt != attempt))):
        asked = f"run {run_id}" + (f" attempt {attempt}" if attempt else "")
        return None, _refuse(ctx, f"⚠️ TOOL_ERROR: GitHub's answer for {asked} names run {run.get('databaseId')!r} attempt "
                                  f"{run.get('attempt')!r}; the run was not read.", "TOOL_ERROR")
    where = _repo_from_url(run.get("url"))
    if where is None:
        return None, _refuse(ctx, f"⚠️ TOOL_ERROR: GitHub returned no URL for run {run_id}; its repository cannot be named.",
                             "TOOL_ERROR")
    return (run, where, observed_attempt), ""


def _retain(ctx: ToolContext, *, source_id: str, data: bytes, extension: str) -> Dict[str, Any] | str:
    """Retain exact bytes once in this task's artifact store (write-once by digest), or say why not."""
    from ouroboros.artifacts import store_actor_source_bytes, task_id_for_artifacts
    from ouroboros.tool_access import canonical_data_root

    try:
        return store_actor_source_bytes(canonical_data_root(ctx), task_id_for_artifacts(ctx), category="tool_results",
                                        source_id=source_id, data=data, extension=extension)
    except Exception as exc:  # The bytes are then not retained; the caller says so instead of naming a path.
        return f"{type(exc).__name__}: {exc}"


def _run_line(run: dict) -> str:
    return (f"Workflow: {_one_line(run.get('workflowName'), 80)}; run: {_check_state(run)}; "
            f"head {run.get('headSha') or 'unknown'}; {_one_line(run.get('url'), 200)}")


def _repo_hint(repo: str) -> str:
    return f", repo={repo!r}" if repo else ""


def get_run(read) -> str:
    """Capture the log of one completed run (or one of its jobs) exactly, retain it, show its tail."""
    ctx, repo, args = read.ctx, read.repo, read.args
    run_id, job_id, attempt = args.get("target"), args.get("job_id", 0), args.get("attempt", 0)
    failed_only = args.get("failed_only", False)
    max_bytes, tail_lines = args.get("max_bytes", LOG_MAX_BYTES_DEFAULT), args.get("tail_lines", _TAIL_LINES_DEFAULT)
    include = args.get("include") or []
    if any(facet != "log" for facet in include) or read.cursor:
        return read.refuse("⚠️ TOOL_ARG_ERROR: a run supports include=['log']; page retained log text with read_file, not cursor.")
    run_id, job_id, attempt, cap, tail = (_whole(value) for value in (run_id, job_id, attempt, max_bytes, tail_lines))
    if None in (run_id, job_id, attempt, cap, tail) or not isinstance(failed_only, bool):
        return _refuse(ctx, "⚠️ TOOL_ARG_ERROR: target, job_id, attempt, max_bytes and tail_lines are whole numbers; "
                            "failed_only is a boolean.", no_effect=True)
    if run_id <= 0:
        return _refuse(ctx, "⚠️ TOOL_ARG_ERROR: target names the workflow run (the number in …/actions/runs/<run_id>; "
                            "github_read(method='checks') lists the runs of a commit).", no_effect=True)
    if job_id < 0 or attempt < 0:
        return _refuse(ctx, "⚠️ TOOL_ARG_ERROR: job_id and attempt are 0 (omitted) or positive.", no_effect=True)
    if not 1 <= cap <= LOG_MAX_BYTES_CEILING:
        return _refuse(ctx, f"⚠️ TOOL_ARG_ERROR: max_bytes is 1..{LOG_MAX_BYTES_CEILING} (default {LOG_MAX_BYTES_DEFAULT}).",
                       no_effect=True)
    if not 1 <= tail <= _TAIL_LINES_MAX:
        return _refuse(ctx, f"⚠️ TOOL_ARG_ERROR: tail_lines is 1..{_TAIL_LINES_MAX}.", no_effect=True)

    resolved, error = _resolve_run(ctx, run_id, attempt, repo, _RUN_FIELDS + ",jobs")
    if resolved is None:
        return error
    run, (host, owner, name), attempt_read = resolved
    item = read.project(run if read.detail == "full" else {k: run.get(k) for k in _RUN_FIELDS.split(",")})
    if "log" not in include:
        return _completed_read(read, f"GitHub Actions run — {owner}/{name} run {run_id} attempt {attempt_read}\n" + _run_line(run),
                               {"item": item, "complete": True, "next_cursor": ""})
    jobs = [job for job in run.get("jobs") or [] if isinstance(job, dict)]
    job = None
    if job_id:
        job = next((item for item in jobs if _whole(item.get("databaseId")) == job_id), None)
        if job is None:
            listed = ", ".join(f"{_whole(item.get('databaseId'))} {_one_line(item.get('name'), 60)}" for item in jobs[:30])
            return _refuse(ctx, f"⚠️ TOOL_ARG_ERROR: job {job_id} is not a job of run {run_id} attempt {attempt_read} in "
                                f"{owner}/{name}; its jobs: {listed or 'none listed'}.")
    status = str(run.get("status") or "unknown")
    if status != "completed":
        return _refuse(ctx, f"⚠️ TOOL_ERROR: run {run_id} attempt {attempt_read} in {owner}/{name} is {status}; GitHub "
                            "serves the log of a completed run only, so no log was requested.", "TOOL_ERROR")

    # Pinned to the attempt the metadata read observed: a re-run between the two reads cannot swap the bytes.
    log_args = ["run", "view"] + ([] if job_id else [str(run_id)]) + ["--log-failed" if failed_only else "--log"]
    log_args += (["--job", str(job_id)] if job_id else []) + ["--attempt", str(attempt_read)]
    with tempfile.TemporaryDirectory(prefix="gh-run-log-") as tmp:
        res = _gh_run(log_args, ctx, timeout=_LOG_TIMEOUT, repo=repo, capture_to=pathlib.Path(tmp) / "log.bin", max_bytes=cap)
        if not res.ok:
            return res.text
        capture = res.capture
        data = capture.path.read_bytes()

    source = " ".join(["gh", *log_args] + (["--repo", repo] if repo else []))
    head = [f"GitHub Actions log — {owner}/{name} run {run_id} attempt {attempt_read}"
            + (f", job {job_id} {_one_line(job.get('name'), 80)} ({_check_state(job)})" if job else ""),
            _run_line(run),
            f"Source: {source}; captured {capture.size} bytes exactly" + (
                f"; the {cap}-byte cap ended the capture: the log is longer and this is its first {capture.size} bytes"
                if capture.truncated else "")]
    payload = {"item": item, "complete": not capture.truncated, "next_cursor": "",
               "capture": {"bytes": capture.size, "sha256": capture.sha256, "truncated": capture.truncated,
                           "source": source, "run_id": run_id, "attempt": attempt_read, "job_id": job_id or None}}
    if capture.truncated:
        read.omitted.append(f"log exceeds max_bytes={cap}; retained only its first {capture.size} bytes")
    if not data:
        head.append("The log is empty: " + ("GitHub lists no failed step here, so --log-failed prints nothing."
                                           if failed_only else "gh printed no log line.") + " Nothing was retained.")
        return _completed_read(read, "\n".join(head), payload)
    ref = _retain(ctx, source_id=f"gh-run-{run_id}-attempt-{attempt_read}" + (f"-job-{job_id}" if job_id else "")
                  + ("-failed" if failed_only else "") + "-log", data=data, extension="log")
    if isinstance(ref, str):
        head[2] += f", sha256 {capture.sha256}"
        head.append(f"Retained: NOT retained ({ref}); the lines below are this call's only copy.")
    else:
        head.append(f"Retained: artifact_store/{ref['path']} ({ref['size']} bytes, sha256 {ref['sha256']}); "
                    f"read_file(root=\"artifact_store\", path={ref['path']!r}, start_line=1) pages the text view "
                    "(it folds CRLF; the retained bytes keep it).")
    payload["capture"]["retained"] = ref if isinstance(ref, dict) else None
    lines = data.decode("utf-8", errors="replace").splitlines()
    counts = f"Lines: {len(lines)}; showing the last {{shown}} (each clipped to {_LINE_CLIP} chars)."
    room = tool_result_limit("github_read") - _RESULT_MARGIN - sum(len(line) + 1 for line in head) - len(counts) - 8
    shown: List[str] = []
    for line in reversed(lines[-tail:]):
        clipped = line if len(line) <= _LINE_CLIP else line[:_LINE_CLIP - 1] + "…"
        if len(clipped) + 1 > room:
            break
        room -= len(clipped) + 1
        shown.append(clipped)
    shown.reverse()
    payload["capture"].update(lines=len(lines), shown_lines=len(shown))
    if len(shown) < len(lines) or any(len(line) > _LINE_CLIP for line in lines[-tail:]):
        read.omitted.append(f"log preview shows {len(shown)} of {len(lines)} lines, each clipped to {_LINE_CLIP} chars; read the retained bytes")
    result = "\n".join(head + [counts.format(shown=len(shown)), "---", *shown])
    return _incomplete_read(read, result, payload) if isinstance(ref, str) else _completed_read(read, result, payload)


def list_artifacts(read) -> str:
    """The artifacts GitHub registers for one run, each by repository + id, one caller-sized page per call."""
    ctx, repo = read.ctx, read.repo
    run_id, page = _whole(read.args.get("target")), _whole(read.cursor)
    if run_id is None or run_id <= 0 or page is None:
        return _refuse(ctx, "⚠️ TOOL_ARG_ERROR: target is the positive workflow run number; cursor is a whole page number.",
                       no_effect=True)
    page = page or 1  # An omitted page is the first.
    if page < 1:
        return _refuse(ctx, "⚠️ TOOL_ARG_ERROR: cursor starts at page 1.", no_effect=True)
    resolved, error = _resolve_run(ctx, run_id, 0, repo, _RUN_ARTIFACT_FIELDS, require_attempt=False)
    if resolved is None:
        return error
    run, (host, owner, name), _ = resolved
    # The one literal API path: host, owner and repository come from the URL GitHub returned for the run.
    res = _gh_run(["api", f"repos/{owner}/{name}/actions/runs/{run_id}/artifacts?per_page={read.limit}&page={page}",
                   "--hostname", host], ctx, timeout=_READ_TIMEOUT)
    if not res.ok:
        return res.text
    data = _gh_json(res, dict)
    rows = data.get("artifacts") if data is not None else None
    if not isinstance(rows, list):
        return _refuse(ctx, f"⚠️ TOOL_ERROR: failed to parse the artifacts JSON: {res.text[:500]}", "TOOL_ERROR")
    if any(not isinstance(row, dict) for row in rows):
        return _refuse(ctx, "⚠️ TOOL_ERROR: artifact page contains a non-object record; completeness is unknown.", "TOOL_ERROR")
    total = _whole(data.get("total_count"))
    if total is None or total < len(rows):
        return _refuse(ctx, "⚠️ TOOL_ERROR: artifact page has no valid total_count; completeness is unknown.", "TOOL_ERROR")
    pages = max(1, -(-total // read.limit))
    head = [f"GitHub Actions artifacts — {owner}/{name} run {run_id}: {total}; page {page} of {pages}"
            + f" ({len(rows)} listed" + (f"; page {page + 1} holds the next" if page < pages else "") + ")",
            _run_line(run),
            "Identity is repository + artifact id: the same name may exist in other runs and repositories."]
    payload = {"items": read.rows(rows, lambda row: {k: row[k] for k in
               ("id", "name", "size_in_bytes", "expired", "created_at", "expires_at", "digest", "workflow_run") if k in row}),
               "total_count": total, "complete": page >= pages, "next_cursor": str(page + 1) if page < pages else ""}
    if read.since:
        read.omitted.append("artifact listing has no since filter; this page is unfiltered")
    if not rows:
        return _completed_read(read, "\n".join(head + [
            f"Page {page} lists no artifact (GitHub holds {total})." if page > 1 else "No artifact is registered for this run."]), payload)
    ref = _retain(ctx, source_id=f"gh-run-{run_id}-artifacts-page-{page}", data=res.text.encode("utf-8"), extension="json")
    if isinstance(ref, str):
        return _refuse(ctx, f"⚠️ TOOL_ERROR: artifact page could not be retained ({ref}); no incomplete listing substitutes for it.", "TOOL_ERROR")
    payload["saved_page"] = ref
    head.append(f"Complete captured JSON page: read_file(root='artifact_store', path={ref['path']!r}); sha256 {ref['sha256']}.")
    room = tool_result_limit("github_read") - _RESULT_MARGIN - sum(len(line) + 1 for line in head)
    body: List[str] = []
    for index, row in enumerate(rows):
        wr = row.get("workflow_run") if isinstance(row.get("workflow_run"), dict) else {}
        size = _whole(row.get("size_in_bytes"))
        entry = [f"- id {row.get('id')}: {_one_line(row.get('name'), 100)} — {size if size is not None else 'unknown'} bytes"
                 + (", expired" if row.get("expired") else "") + f"; created {row.get('created_at') or '?'}; expires "
                 f"{row.get('expires_at') or '?'}" + (f"; head {wr.get('head_sha')}" if wr.get("head_sha") else "")
                 + (f"; digest {_one_line(row.get('digest'), 80)}" if isinstance(row.get("digest"), str) and row.get("digest") else ""),
                 ("    expired: GitHub no longer serves it" if row.get("expired") else
                  f"    exceeds this tool's supported {ARTIFACT_MAX_BYTES_CEILING}-byte download limit"
                  if size is not None and size > ARTIFACT_MAX_BYTES_CEILING else
                  f"    github_read(method='download', kind='artifact', target={row.get('id')}, max_bytes={size or '…'}{_repo_hint(repo)})")]
        cost = sum(len(line) + 1 for line in entry)
        rest = f"- {len(rows) - index} more artifacts omitted from this preview; read the complete captured JSON page above."
        if cost + len(rest) + 1 > room:
            read.omitted.append(rest)
            body.append(rest)
            break
        room -= cost
        body += entry
    return _completed_read(read, "\n".join(head + body), payload)


def download(read) -> str:
    """Retain one artifact's archive, as served and under an explicit byte cap, without extracting it."""
    ctx, repo = read.ctx, read.repo
    if read.kind != "artifact":
        return read.refuse("⚠️ TOOL_ARG_ERROR: download requires kind=artifact.")
    artifact_id, cap = _whole(read.args.get("target")), _whole(read.args.get("max_bytes"))
    if artifact_id is None or cap is None:
        return _refuse(ctx, "⚠️ TOOL_ARG_ERROR: target and max_bytes are whole numbers.", no_effect=True)
    if artifact_id <= 0:
        return _refuse(ctx, "⚠️ TOOL_ARG_ERROR: target is the id github_read(method='list', kind='artifact', target=run_id) shows.", no_effect=True)
    if not 1 <= cap <= ARTIFACT_MAX_BYTES_CEILING:
        return _refuse(ctx, f"⚠️ TOOL_ARG_ERROR: max_bytes is required: the explicit byte cap of this download, "
                            f"1..{ARTIFACT_MAX_BYTES_CEILING}; the download stops at the cap and nothing is retained then.",
                       no_effect=True)
    res = _gh_run(["repo", "view", "--json", "nameWithOwner,url"], ctx, timeout=_READ_TIMEOUT, repo=repo)
    if not res.ok:
        return res.text
    info = _gh_json(res, dict)
    where = _repo_from_url((info or {}).get("url"))
    if where is None:
        return _refuse(ctx, f"⚠️ TOOL_ERROR: gh named no repository URL: {res.text[:300]}", "TOOL_ERROR")
    host, owner, name = where
    meta = _gh_run(["api", f"repos/{owner}/{name}/actions/artifacts/{artifact_id}", "--hostname", host], ctx,
                   timeout=_READ_TIMEOUT)
    if not meta.ok:
        return meta.text
    art = _gh_json(meta, dict)
    if art is None or _whole(art.get("id")) != artifact_id:
        return _refuse(ctx, f"⚠️ TOOL_ERROR: the artifact answer is malformed or names another id: {meta.text[:300]}",
                       "TOOL_ERROR")
    label = f"artifact {artifact_id} {_one_line(art.get('name'), 80)!r} of {owner}/{name}"
    size = _whole(art.get("size_in_bytes"))
    if art.get("expired") is True:
        return _refuse(ctx, f"⚠️ TOOL_ERROR: {label} expired ({art.get('expires_at') or 'date unknown'}); GitHub no longer "
                            "serves it, so nothing was downloaded.", "TOOL_ERROR", no_effect=True)
    if size is not None and size > cap:
        return _refuse(ctx, f"⚠️ TOOL_ARG_ERROR: {label} is {size} bytes, above max_bytes={cap}; nothing was downloaded. "
                            f"Raise max_bytes deliberately (ceiling {ARTIFACT_MAX_BYTES_CEILING}).", no_effect=True)
    with tempfile.TemporaryDirectory(prefix="gh-artifact-") as tmp:
        path = pathlib.Path(tmp) / "artifact.zip"
        res = _gh_run(["api", f"repos/{owner}/{name}/actions/artifacts/{artifact_id}/zip", "--hostname", host], ctx,
                      timeout=_ARTIFACT_TIMEOUT, capture_to=path, max_bytes=cap)
        if not res.ok:
            return res.text
        capture = res.capture
        if capture.truncated:
            return _refuse(ctx, f"⚠️ TOOL_ERROR: the download of {label} exceeded max_bytes={cap} (GitHub stated "
                                f"{size if size is not None else 'an unknown number of'} bytes); the partial archive was "
                                "discarded and nothing was retained.", "TOOL_ERROR")
        if not zipfile.is_zipfile(path):
            return _refuse(ctx, f"⚠️ TOOL_ERROR: the {capture.size} bytes GitHub served for {label} are not a zip archive; "
                                "they were discarded.", "TOOL_ERROR")
        try:
            with zipfile.ZipFile(path) as archive:
                members = archive.infolist()  # The central directory only; no member is read or written.
        except (zipfile.BadZipFile, OSError, ValueError) as exc:
            return _refuse(ctx, f"⚠️ TOOL_ERROR: the archive served for {label} cannot be listed ({exc}); discarded.",
                           "TOOL_ERROR")
        data = path.read_bytes()
    ref = _retain(ctx, source_id=f"gh-artifact-{owner}-{name}-{artifact_id}", data=data, extension="zip")
    wr = art.get("workflow_run") if isinstance(art.get("workflow_run"), dict) else {}
    digest = str(art.get("digest") or "")
    stated = digest.partition(":")[2].lower() if digest.lower().startswith("sha256:") else ""
    lines = [f"GitHub Actions artifact — {owner}/{name} artifact {artifact_id} {_one_line(art.get('name'), 100)!r}"
             + (f" (run {wr.get('id')}, head {wr.get('head_sha')})" if wr else "")
             + f"; created {art.get('created_at') or '?'}, expires {art.get('expires_at') or '?'}",
             f"Archive: {capture.size} bytes, sha256 {capture.sha256}" + (
                 f"; GitHub's digest {digest} {'matches' if stated == capture.sha256 else 'differs from'} these bytes"
                 if stated else "; GitHub states no digest for it")]
    if isinstance(ref, str):
        lines.append(f"Retained: NOT retained ({ref}); the archive was discarded with this call's temporary directory.")
    else:
        lines.append(f"Retained as the archive GitHub served, not extracted: artifact_store/{ref['path']} "
                     f"({ref['size']} bytes, sha256 {ref['sha256']}).")
    unpacked = sum(int(member.file_size) for member in members)
    lines.append(f"Members ({len(members)}, {unpacked} bytes uncompressed; names and sizes are read from the archive "
                 "directory and are untrusted):")
    lines += [f"  {_one_line(member.filename, 200)} ({member.file_size} bytes)" for member in members[:_ZIP_MEMBERS_SHOWN]]
    if len(members) > _ZIP_MEMBERS_SHOWN:
        lines.append(f"  … {len(members) - _ZIP_MEMBERS_SHOWN} more members")
    lines.append("Extract only under your own guards, into an empty directory of your own.")
    if len(members) > _ZIP_MEMBERS_SHOWN:
        read.omitted.append(f"{len(members) - _ZIP_MEMBERS_SHOWN} archive member names omitted; the complete zip is retained")
    payload = {"item": read.project(art), "complete": True, "next_cursor": "",
               "capture": {"bytes": capture.size, "sha256": capture.sha256, "retained": ref if isinstance(ref, dict) else None,
                           "digest_matches": stated == capture.sha256 if stated else None},
               "members": [{"name": member.filename, "bytes": member.file_size} for member in members[:_ZIP_MEMBERS_SHOWN]],
               "member_count": len(members), "uncompressed_bytes": unpacked}
    result = "\n".join(lines)
    return _incomplete_read(read, result, payload) if isinstance(ref, str) else _completed_read(read, result, payload)
