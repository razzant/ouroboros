"""Bind the Windows qualification build to one exact candidate SHA, verify the ZIP, and render summaries.

Standard library only so every pre-venv gate (`pointer`, `checkout`, `bind`,
`verify-package`, `record-failure`, `summary`) runs directly on the runner's
base Python and always persists structured `verdict`, `phase`, and `code` into
`result.json` + `$GITHUB_STEP_SUMMARY` even when setup stops before any venv exists.
Published JSON files never include host/Windows filesystem paths.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import os
import pathlib
import re
import runpy
import shutil
import subprocess
import sys
import time
import zipfile
from typing import Any, Iterable

FIXTURE_IDENTITY = ("-c", "user.name=Ouroboros QA fixture", "-c", "user.email=qa-fixture@invalid")
SHA_RE = re.compile(r"^[0-9a-f]{40}$")
WINDOWS_DRIVE_RE = re.compile(r"(?i)\b[a-z]:[\\/]")
UNC_PATH_RE = re.compile(r"\\\\[A-Za-z0-9_.-]+\\")
GIT_PAYLOAD_PREFIX = "_internal/git-standalone/"
BRANCH = "ouroboros"
PROGRESS_KEYS = (
    "operation_id", "started_at", "active", "stage", "stage_started_at", "result", "outcome", "outcome_at",
    "concluded_by", "base_sha", "target_sha", "reason", "restart_required", "work_stop", "current_generation",
)
# One attempt from the conclusion seen before the second chat to both records read after it; its apply window
# is [started_at, stage_started_at] of the stage ``restart_requested`` (gateway/control.py ``_restart_response``).
SAME_ATTEMPT_KEYS = (
    "operation_id", "started_at", "stage", "stage_started_at", "result", "outcome_at", "outcome", "concluded_by",
    "base_sha", "target_sha",
)
TEST_TAG_NOTE = (
    "QA TEST TAG, NOT A RELEASE. Local to the isolated build checkout of the Windows update "
    "qualification, made for candidate {sha} and never pushed."
)
DECLARED = {
    "https_git_transport_tested": False,
    "update_remote_transport": "local_bare_mirror_no_https_tls",
    "screenshots_collected": False,
    "screenshots_policy": "disabled_to_prevent_any_cross_window_or_desktop_leak",
    "uia_text_dumps_collected": False,
    "ui_driver": "candidate_windows_computer_use_uia_with_owned_native_input_fallback",
    "git_observation_sink": "app_inherited_git_trace_identity_lines_and_git_trace2_event_sessions",
    "git_identity_claim": "run_wide_count_match_of_identity_lines_and_sessions_not_per_session",
    "system_wide_process_exclusion_claimed": False,
}


class Refusal(Exception):
    """Named setup/package refusal with allowlisted scalar fields."""

    def __init__(self, reason: str, *, verdict: str = "unconfirmed", **fields: Any) -> None:
        super().__init__(reason)
        self.reason = reason
        self.verdict = verdict
        self.fields = {key: _scalar(value) for key, value in fields.items()}

    def __str__(self) -> str:
        return self.reason + "".join(f" {key}={value}" for key, value in self.fields.items())


def _utc_now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat()


def _scrub_text(text: str, limit: int = 200) -> str:
    cleaned = re.sub(r"[^\x20-\x7e]", "?", str(text))
    cleaned = WINDOWS_DRIVE_RE.sub("<host-path>/", cleaned)
    cleaned = UNC_PATH_RE.sub("<unc-path>/", cleaned)
    return cleaned[:limit]


def _scalar(value: Any) -> Any:
    if isinstance(value, (bool, int, float)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [_scalar(item) for item in list(value)[:12]]
    if isinstance(value, dict):
        return {_scrub_text(str(k), 60): _scalar(v) for k, v in list(value.items())[:16]}
    return _scrub_text(str(value), 200)


def assert_no_host_paths(
    payload: Any,
    *,
    where: str = "root",
    forbidden_substrings: Iterable[str] = (),
) -> None:
    subs = tuple(s for s in forbidden_substrings if s)
    if isinstance(payload, dict):
        for key, value in payload.items():
            assert_no_host_paths(key, where=f"{where}.key", forbidden_substrings=subs)
            assert_no_host_paths(value, where=f"{where}.{key}", forbidden_substrings=subs)
    elif isinstance(payload, (list, tuple)):
        for idx, item in enumerate(payload):
            assert_no_host_paths(item, where=f"{where}[{idx}]", forbidden_substrings=subs)
    elif isinstance(payload, str):
        if WINDOWS_DRIVE_RE.search(payload) or UNC_PATH_RE.search(payload):
            raise Refusal("published_payload_contains_host_path", where=where)
        if any(sub in payload for sub in subs):
            raise Refusal("published_payload_contains_prompt_text", where=where)


def _env() -> dict[str, str]:
    clean = {k: v for k, v in os.environ.items() if not k.upper().startswith("GIT_TRACE")}
    return {**clean, "GIT_TERMINAL_PROMPT": "0", "GIT_PAGER": "cat"}


def _subcommand(args: tuple[str, ...]) -> str:
    skip = False
    for arg in args:
        if skip:
            skip = False
        elif arg in ("-c", "-C"):
            skip = True
        elif not arg.startswith("-"):
            return arg
    return ""


def _git(repo: pathlib.Path, *args: str, check: bool = True) -> str:
    done = subprocess.run(["git", *args], cwd=repo, capture_output=True, text=True, env=_env())
    if check and done.returncode != 0:
        raise Refusal("git_failed", subcommand=_subcommand(args), exit_code=done.returncode)
    return (done.stdout or "").strip() if done.returncode == 0 else ""


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json_retry(path: pathlib.Path, *, attempts: int = 5, delay: float = 0.25) -> Any:
    for _ in range(attempts):
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return None
        except (OSError, ValueError):
            time.sleep(delay)
    return None


def read_git_head(git_dir: pathlib.Path) -> str:
    try:
        head = (git_dir / "HEAD").read_text(encoding="utf-8").strip()
    except OSError:
        return ""
    if not head.startswith("ref: "):
        return head
    ref = head[5:]
    if (loose := git_dir / ref).is_file():
        return loose.read_text(encoding="utf-8").strip()
    if (packed := git_dir / "packed-refs").is_file():
        for ln in packed.read_text(encoding="utf-8").splitlines():
            if ln.endswith(" " + ref):
                return ln.split(" ", 1)[0]
    return ""


def write_json_atomic(
    path: pathlib.Path,
    payload: dict[str, Any],
    *,
    forbidden_substrings: Iterable[str] = (),
) -> None:
    assert_no_host_paths(payload, where=path.name, forbidden_substrings=forbidden_substrings)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def _outputs(path: str, values: dict[str, str]) -> None:
    if path:
        with open(path, "a", encoding="utf-8") as handle:
            for key, value in values.items():
                handle.write(f"{key}={value}\n")


def record_structured_result(
    evidence: pathlib.Path,
    *,
    verdict: str,
    phase: str,
    code: str,
    candidate_sha: str = "",
    fields: dict[str, Any] | None = None,
    overwrite_existing: bool = True,
) -> None:
    result_path = evidence / "result.json"
    if not overwrite_existing and result_path.is_file():
        return
    clean_sha = candidate_sha.strip().lower()
    v = verdict if verdict in ("passed", "failed", "unconfirmed", "error") else "unconfirmed"
    p, c = _scrub_text(phase or "setup", 60), _scrub_text(code or "setup_refused", 80)
    reason: dict[str, Any] = {"verdict": v, "phase": p, "code": c}
    if fields:
        reason["fields"] = {k: _scalar(val) for k, val in fields.items()}
    write_json_atomic(
        result_path,
        {
            "schema": 5,
            "verdict": v,
            "phase": p,
            "code": c,
            "candidate_sha": clean_sha if SHA_RE.match(clean_sha) else "",
            "recorded_at": _utc_now(),
            "reasons": [reason],
            "checks": {},
            "facts": {},
            "declared": dict(DECLARED),
        },
    )


def resolve_pointer(pointer_file: pathlib.Path) -> str:
    if not pointer_file.is_file():
        raise Refusal("candidate_pointer_missing")
    try:
        doc = json.loads(pointer_file.read_text(encoding="utf-8"))
    except Exception as exc:
        raise Refusal("candidate_pointer_unreadable", error_type=type(exc).__name__) from exc
    if not isinstance(doc, dict):
        raise Refusal("candidate_pointer_invalid")
    sha = str(doc.get("candidate_sha") or "").strip().lower()
    if not SHA_RE.match(sha):
        raise Refusal("candidate_sha_missing_or_invalid")
    return sha


def parse_refs(listing: str) -> list[tuple[str, str]]:
    return sorted((p[0], p[1]) for line in listing.splitlines() if len(p := line.split()) == 2)


def _refs(repo: pathlib.Path) -> list[tuple[str, str]]:
    return parse_refs(_git(repo, "for-each-ref", "--format=%(refname) %(objectname)"))


def public_tag_fact(listing: str, tag: str, candidate: str) -> dict[str, Any]:
    direct = peeled = None
    for line in listing.splitlines():
        parts = line.split()
        if len(parts) == 2 and parts[1] == f"refs/tags/{tag}":
            direct = parts[0]
        elif len(parts) == 2 and parts[1] == f"refs/tags/{tag}^{{}}":
            peeled = parts[0]
    if direct is None:
        return {"present": False, "points_at_candidate": None}
    return {"present": True, "points_at_candidate": (peeled or direct) == candidate}


def _public_tag_listing(repo: pathlib.Path, source_url: str, tag: str) -> str | None:
    done = subprocess.run(
        ["git", "-c", "credential.helper=", "ls-remote", "--tags", source_url, f"refs/tags/{tag}*"],
        cwd=repo,
        capture_output=True,
        text=True,
        env=_env(),
        timeout=120,
    )
    return done.stdout if done.returncode == 0 else None


def checkout(repo: pathlib.Path, candidate: str, source_url: str) -> dict[str, Any]:
    if not SHA_RE.match(candidate):
        raise Refusal("candidate_sha_invalid", candidate=candidate)
    repo.mkdir(parents=True, exist_ok=True)
    if any(repo.iterdir()):
        raise Refusal("build_dir_not_empty", path=repo.name)
    _git(repo, "init", "-q")
    _git(repo, "-c", "credential.helper=", "fetch", "-q", "--no-tags", source_url, candidate)
    if _git(repo, "cat-file", "-t", candidate, check=False) != "commit":
        raise Refusal("candidate_not_a_commit", candidate=candidate)
    _git(repo, "update-ref", f"refs/heads/{BRANCH}", candidate)
    _git(repo, "symbolic-ref", "HEAD", f"refs/heads/{BRANCH}")
    _git(repo, "reset", "-q", "--hard")
    return {"candidate_sha": candidate, "refs": [name for name, _sha in _refs(repo)]}


def _mirror_fingerprint(mirror_url: str) -> str:
    return hashlib.sha256(str(mirror_url).encode("utf-8")).hexdigest()[:16]


def bind(repo: pathlib.Path, candidate: str, mirror_url: str, source_url: str) -> dict[str, Any]:
    if not SHA_RE.match(candidate):
        raise Refusal("candidate_sha_invalid", candidate=candidate)
    head = _git(repo, "rev-parse", "HEAD")
    if head != candidate:
        raise Refusal("checkout_is_not_candidate", head=head, requested=candidate)
    if _git(repo, "rev-parse", "--is-shallow-repository") != "false":
        raise Refusal("candidate_history_shallow")
    if _git(repo, "status", "--porcelain"):
        raise Refusal("candidate_tree_dirty")
    guard = runpy.run_path(str(repo / "scripts" / "build_repo_bundle.py"))
    try:
        version = guard["_read_version"](repo)
    except SystemExit as refused:
        raise Refusal(
            "candidate_version_not_packageable",
            guard=_scrub_text(str(refused), 120),
            prerequisite="parent supplies a packageable release SHA without editing VERSION",
        ) from refused
    before = _refs(repo)
    if before != [(f"refs/heads/{BRANCH}", candidate)]:
        raise Refusal("ref_area_not_isolated", refs=[name for name, _sha in before])
    tag = f"v{version}"
    _git(repo, *FIXTURE_IDENTITY, "tag", "-a", tag, "-m", TEST_TAG_NOTE.format(sha=candidate), candidate)
    kind = _git(repo, "cat-file", "-t", f"refs/tags/{tag}")
    peeled = _git(repo, "rev-parse", f"refs/tags/{tag}^{{commit}}")
    after = _refs(repo)
    if kind != "tag" or peeled != candidate or [name for name, _sha in after] != [f"refs/heads/{BRANCH}", f"refs/tags/{tag}"]:
        raise Refusal("test_tag_not_as_made", kind=kind, refs=[name for name, _sha in after])
    if "managed" in _git(repo, "remote").split():
        _git(repo, "remote", "set-url", "managed", mirror_url)
    else:
        _git(repo, "remote", "add", "managed", mirror_url)
    listing = None
    try:
        listing = _public_tag_listing(repo, source_url, tag)
    except (OSError, subprocess.SubprocessError):
        pass
    return {
        "candidate_sha": candidate,
        "tree_sha": _git(repo, "rev-parse", "HEAD^{tree}"),
        "version": version,
        "release_tag": tag,
        "release_tag_kind": "qa_test_local_annotated_in_isolated_checkout",
        "managed_remote_bound": True,
        "managed_remote_kind": "local_bare_mirror",
        "managed_remote_sha256_16": _mirror_fingerprint(mirror_url),
        "refs_before_tag": [name for name, _sha in before],
        "refs_after_tag": [name for name, _sha in after],
        "public_same_name_tag": public_tag_fact(listing, tag, candidate)
        if listing is not None
        else {"present": None, "points_at_candidate": None},
        "pushed": False,
    }


def inner_archive_path(name: str) -> str:
    normed = name.replace("\\", "/")
    return normed.split("/", 1)[1] if "/" in normed else normed


def archive_footprint(archive: pathlib.Path, prefix: str = GIT_PAYLOAD_PREFIX) -> dict[str, Any]:
    with zipfile.ZipFile(archive) as bundle:
        files = [info for info in bundle.infolist() if not info.is_dir()]
    picked = [info for info in files if inner_archive_path(info.filename).lower().startswith(prefix.lower())]
    zip_bytes = archive.stat().st_size
    git_compressed = sum(info.compress_size for info in picked)
    return {
        "zip_bytes": zip_bytes,
        "entries": len(files),
        "compressed_bytes": sum(info.compress_size for info in files),
        "uncompressed_bytes": sum(info.file_size for info in files),
        "git_payload": {
            "prefix": prefix,
            "entries": len(picked),
            "compressed_bytes": git_compressed,
            "uncompressed_bytes": sum(info.file_size for info in picked),
            "share_of_zip": round(git_compressed / zip_bytes, 4) if zip_bytes else None,
        },
        "git_exe_entries": sorted(
            inner_archive_path(info.filename)
            for info in picked
            if inner_archive_path(info.filename).lower().endswith("/git.exe")
        ),
    }


def _git_standalone_executables(payload: pathlib.Path) -> dict[str, str]:
    """Hash every ``.exe`` under ``_internal/git-standalone/`` relative to ``payload``."""
    standalone = payload / "_internal" / "git-standalone"
    if not standalone.is_dir():
        return {}
    found: dict[str, str] = {}
    for root, _dirs, files in os.walk(standalone):
        for name in files:
            if name.lower().endswith(".exe"):
                path = pathlib.Path(root) / name
                found[path.relative_to(payload).as_posix()] = sha256_file(path)
    return dict(sorted(found.items()))


def unpack_candidate_archive(archive: pathlib.Path, extract_dir: pathlib.Path) -> tuple[pathlib.Path, dict[str, Any], dict[str, str]]:
    """Extract ``archive`` into ``extract_dir`` and return ``(package_root, manifest, git_standalone_sha256)``."""
    gits: dict[str, str] = {}
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(extract_dir)
        for info in zf.infolist():
            inner = inner_archive_path(info.filename)
            name = pathlib.PurePosixPath(inner).name.lower()
            if not info.is_dir() and inner.lower().startswith(GIT_PAYLOAD_PREFIX) and name.endswith(".exe"):
                with zf.open(info) as fh:
                    gits[inner] = hashlib.file_digest(fh, "sha256").hexdigest()
    package_root = extract_dir / "Ouroboros"
    manifests = sorted(package_root.rglob("repo_bundle_manifest.json"))
    manifest = json.loads(manifests[0].read_text(encoding="utf-8")) if len(manifests) == 1 else {}
    if len(manifests) == 1:
        manifest["_bundle_path"] = str(manifests[0].with_name(str(manifest.get("bundle_file") or "")))
    return package_root, manifest, gits


def create_fixture_mirror_b(
    *,
    harness_git: str,
    mirror: pathlib.Path,
    qa_root: pathlib.Path,
    bundle: pathlib.Path,
    candidate: str,
    doc_rel_path: str,
    marker: str,
) -> tuple[str, list[str], str]:
    """Seed local bare ``mirror`` from ``bundle`` and push one fast-forward commit B on ``ouroboros``."""
    clean_env = _env()

    def run_git(*git_args: str) -> str:
        done = subprocess.run([harness_git, *git_args], capture_output=True, text=True, env=clean_env, timeout=300)
        if done.returncode != 0:
            raise Refusal("fixture_git_failed", subcommand=_subcommand(git_args), exit_code=done.returncode)
        return done.stdout.strip()

    run_git("init", "--bare", str(mirror))
    for spec in ("HEAD:refs/heads/ouroboros", "+refs/tags/*:refs/tags/*"):
        run_git("-C", str(mirror), "fetch", "--no-tags", str(bundle), spec)
    for branch in ("ouroboros-stable", "main"):
        run_git("-C", str(mirror), "update-ref", f"refs/heads/{branch}", candidate)
    run_git("-C", str(mirror), "symbolic-ref", "HEAD", "refs/heads/ouroboros")
    tags = run_git("-C", str(mirror), "for-each-ref", "--format=%(refname)", "refs/tags").splitlines()

    work = qa_root / "fixture-work"
    run_git("clone", "--branch", "ouroboros", str(mirror), str(work))
    doc = work / doc_rel_path
    doc.parent.mkdir(parents=True, exist_ok=True)
    doc.write_text(f"# Windows update qualification fixture\n\nMarker: {marker}\n", encoding="utf-8")
    run_git("-C", str(work), "add", doc_rel_path)
    run_git("-C", str(work), *FIXTURE_IDENTITY, "commit", "-q", "-m", "QA fixture B (synthetic local mirror only)")
    b_sha = run_git("-C", str(work), "rev-parse", "HEAD")
    parent_sha = run_git("-C", str(work), "rev-parse", "HEAD^")
    run_git("-C", str(work), "push", "-q", "origin", "HEAD:refs/heads/ouroboros")
    return b_sha, tags, parent_sha


def correlate_boot_update_attempt(
    state_doc: dict[str, Any],
    status_body: dict[str, Any],
    *,
    candidate: str,
    b_sha: str,
    baseline_ops: set[str],
    concluded: dict[str, Any],
) -> tuple[bool, dict[str, Any], dict[str, Any]]:
    """Correlate ``state.json['managed_update_attempt']`` with ``GET /api/update/status['update_progress']`` and
    with ``concluded``, the same record as seen when the boot concluded it, before the second chat."""
    stored_raw, public_raw = state_doc.get("managed_update_attempt"), status_body.get("update_progress")
    if not isinstance(stored_raw, dict) or not isinstance(public_raw, dict):
        raise Refusal("attempt_record_missing_or_unreadable")
    stored = {k: _scalar(stored_raw.get(k)) for k in PROGRESS_KEYS if k in stored_raw}
    public = {k: _scalar(public_raw.get(k)) for k in PROGRESS_KEYS if k in public_raw}
    seen = {k: _scalar(concluded.get(k)) for k in SAME_ATTEMPT_KEYS}
    expected = {"base_sha": candidate, "target_sha": b_sha, "outcome": "applied", "concluded_by": "boot"}
    ok = (
        bool(stored.get("operation_id"))
        and str(stored.get("operation_id")) not in baseline_ops
        and bool(stored.get("started_at"))
        and bool(stored.get("outcome_at"))
        and public.get("active") is False
        and all(stored.get(k) == public.get(k) == seen[k] for k in SAME_ATTEMPT_KEYS)
        and all(stored.get(k) == val for k, val in expected.items())
    )
    return ok, stored, public


def parse_bundle_heads(listing: str) -> dict[str, str]:
    return {p[1]: p[0] for line in listing.splitlines() if len(p := line.split()) == 2}


def verify_package(repo: pathlib.Path, candidate: str, mirror_url: str, binding: dict[str, Any]) -> dict[str, Any]:
    payload = repo / "dist" / "Ouroboros"
    manifests = sorted(payload.rglob("repo_bundle_manifest.json")) if payload.is_dir() else []
    if len(manifests) != 1:
        raise Refusal("package_manifest_not_unique", verdict="failed", found=len(manifests))
    manifest = json.loads(manifests[0].read_text(encoding="utf-8"))
    expected = {
        "source_sha": candidate,
        "app_version": binding["version"],
        "release_tag": binding["release_tag"],
        "managed_remote_url": mirror_url,
        "managed_remote_branch": BRANCH,
    }
    wrong = sorted(key for key, value in expected.items() if manifest.get(key) != value)
    if wrong:
        raise Refusal("package_names_another_build", verdict="failed", differs=wrong)
    bundle = manifests[0].with_name(str(manifest.get("bundle_file") or ""))
    if not bundle.is_file() or sha256_file(bundle) != manifest.get("bundle_sha256"):
        raise Refusal("package_bundle_digest_mismatch", verdict="failed")
    heads = parse_bundle_heads(_git(repo, "bundle", "list-heads", str(bundle)))
    if sorted(heads) != sorted(["HEAD", f"refs/tags/{binding['release_tag']}"]) or heads.get("HEAD") != candidate:
        raise Refusal("package_bundle_refs_not_isolated", verdict="failed", refs=sorted(heads), head_is_candidate=heads.get("HEAD") == candidate)
    gits = _git_standalone_executables(payload)
    if not any(rel.lower().endswith("/git.exe") for rel in gits):
        raise Refusal("package_lacks_bundled_git", verdict="failed")
    archive = repo / "dist" / f"Ouroboros-{binding['version']}-windows-x64.zip"
    if not archive.is_file():
        raise Refusal("package_archive_missing", verdict="failed", archive=archive.name)
    footprint = archive_footprint(archive)
    if not any(entry.lower().startswith(GIT_PAYLOAD_PREFIX) for entry in footprint["git_exe_entries"]):
        raise Refusal("package_lacks_git_standalone", verdict="failed", prefix=GIT_PAYLOAD_PREFIX, git_exe_entries=footprint["git_exe_entries"])
    return {
        "manifest": {
            "source_sha": manifest.get("source_sha"),
            "app_version": manifest.get("app_version"),
            "release_tag": manifest.get("release_tag"),
            "managed_remote_matches_binding": True,
            "managed_remote_sha256_16": _mirror_fingerprint(mirror_url),
            "managed_remote_branch": manifest.get("managed_remote_branch"),
            "managed_remote_name": manifest.get("managed_remote_name"),
            "bundle_sha256": manifest.get("bundle_sha256"),
        },
        "manifest_path": manifests[0].relative_to(payload).as_posix(),
        "bundle_refs": sorted(heads),
        "bundled_git_executables": gits,
        "archive_name": archive.name,
        "archive_path": str(archive),
        "archive_sha256": sha256_file(archive),
        "archive_footprint": footprint,
    }


def render_summary(evidence: pathlib.Path) -> str:
    result_path = evidence / "result.json"
    if not result_path.is_file():
        return (
            "### Windows update qualification: **unconfirmed**\n\n"
            "- **Phase**: `bootstrap_setup`\n- **Code**: `result_json_missing`\n"
            "- **Limitations**: local bare mirror does not test internet HTTPS/TLS; screenshots and raw UI dumps not collected; "
            "Git traces cover only Git processes that inherited the app's environment, not system-wide process exclusion.\n"
        )
    try:
        doc = json.loads(result_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return (
            "### Windows update qualification: **unconfirmed**\n\n"
            f"- **Phase**: `summary_read`\n- **Code**: `result_json_unreadable` (`{type(exc).__name__}`)\n"
        )

    verdict = str(doc.get("verdict") or "unconfirmed")
    phase = str(doc.get("phase") or "unknown")
    code = str(doc.get("code") or "unknown")
    facts = doc.get("facts") if isinstance(doc.get("facts"), dict) else {}
    fixture = facts.get("fixture") if isinstance(facts.get("fixture"), dict) else {}
    declared = doc.get("declared") if isinstance(doc.get("declared"), dict) else DECLARED

    lines = [
        f"### Windows update qualification: **{verdict}**",
        "",
        f"- **Phase**: `{phase}`",
        f"- **Code**: `{code}`",
        f"- **Candidate A**: `{doc.get('candidate_sha') or ''}`",
    ]
    if fixture.get("b"):
        lines.append(f"- **Fixture B**: `{fixture.get('b')}`")
    if facts.get("system_git_condition"):
        lines.append(f"- **System Git condition**: `{facts.get('system_git_condition')}`")
    for reason in doc.get("reasons") or []:
        if isinstance(reason, dict):
            f_text = ", ".join(f"{k}={json.dumps(v)}" for k, v in (reason.get("fields") or {}).items())
            lines.append(
                f"- **Reason**: `{reason.get('verdict')}` / `{reason.get('phase', phase)}` / `{reason.get('code')}`"
                + (f" ({f_text})" if f_text else "")
            )
    checks = doc.get("checks") if isinstance(doc.get("checks"), dict) else {}
    if checks:
        lines.extend(["", "| Check | OK |", "| --- | --- |"])
        for name, row in checks.items():
            ok = row.get("ok") if isinstance(row, dict) else bool(row)
            lines.append(f"| `{name}` | {'yes' if ok else 'NO'} |")
    traces = facts.get("git_traces") if isinstance(facts.get("git_traces"), dict) else {}
    if traces:
        claim = traces.get("identity") if isinstance(traces.get("identity"), dict) else {}
        lines.extend(
            [
                "",
                "- **Git traces** (identity is run-wide, not per session; `system_wide_process_exclusion_claimed=false`): "
                f"identity=`{claim.get('state')}` (lines `{claim.get('identity_lines')}`, sessions "
                f"`{claim.get('started_sessions')}`), requirements=`{json.dumps(traces.get('requirements') or {})}`",
            ]
        )
    lines.extend(
        [
            "",
            f"- **Explicit limitations**: `https_git_transport_tested={bool(declared.get('https_git_transport_tested'))}`, "
            f"`screenshots_collected={bool(declared.get('screenshots_collected'))}`, "
            f"`uia_text_dumps_collected={bool(declared.get('uia_text_dumps_collected'))}`, "
            f"`system_wide_process_exclusion_claimed={bool(declared.get('system_wide_process_exclusion_claimed'))}`.",
        ]
    )
    return "\n".join(lines) + "\n"


def _stage_harness(
    stage_from: pathlib.Path, stage_to: pathlib.Path, park: tuple[pathlib.Path, pathlib.Path] | None
) -> None:
    """Copy the harness out of the workspace, then move (never delete) the QA checkout out of it, so candidate A is
    checked out into an empty workspace. A rename on one volume leaves Git's read-only pack files untouched."""
    if stage_to.exists():
        raise Refusal("harness_stage_not_empty")
    shutil.copytree(stage_from, stage_to)
    for attempt in range(5 if park is not None else 0):  # a scanner may hold a fresh file for a moment
        try:
            os.rename(*park)
            break
        except OSError as exc:
            if attempt == 4:
                raise Refusal("qa_checkout_not_parked", problem=type(exc).__name__) from exc
            time.sleep(1.0)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)

    p_ptr = sub.add_parser("pointer")
    p_ptr.add_argument("--pointer-file", required=True)
    p_ptr.add_argument("--stage-from", default="")
    p_ptr.add_argument("--stage-to", default="")
    p_ptr.add_argument("--park-checkout", default="")
    p_ptr.add_argument("--park-to", default="")
    p_ptr.add_argument("--evidence", required=True)
    p_ptr.add_argument("--github-output", default=os.environ.get("GITHUB_OUTPUT", ""))

    p_co = sub.add_parser("checkout")
    p_co.add_argument("--repo", default=".")
    p_co.add_argument("--candidate", required=True)
    p_co.add_argument(
        "--source-url",
        default=f"{os.environ.get('GITHUB_SERVER_URL', 'https://github.com')}/{os.environ.get('GITHUB_REPOSITORY', '')}.git",
    )
    p_co.add_argument("--evidence", required=True)

    p_bind = sub.add_parser("bind")
    p_bind.add_argument("--repo", default=".")
    p_bind.add_argument("--candidate", required=True)
    p_bind.add_argument("--mirror-url", required=True)
    p_bind.add_argument(
        "--source-url",
        default=f"{os.environ.get('GITHUB_SERVER_URL', 'https://github.com')}/{os.environ.get('GITHUB_REPOSITORY', '')}.git",
    )
    p_bind.add_argument("--evidence", required=True)
    p_bind.add_argument("--github-output", default=os.environ.get("GITHUB_OUTPUT", ""))

    p_vp = sub.add_parser("verify-package")
    p_vp.add_argument("--repo", default=".")
    p_vp.add_argument("--candidate", required=True)
    p_vp.add_argument("--mirror-url", required=True)
    p_vp.add_argument("--evidence", required=True)
    p_vp.add_argument("--github-output", default=os.environ.get("GITHUB_OUTPUT", ""))

    p_rf = sub.add_parser("record-failure")
    p_rf.add_argument("--verdict", default="unconfirmed")
    p_rf.add_argument("--phase", required=True)
    p_rf.add_argument("--code", required=True)
    p_rf.add_argument("--candidate", default="")
    p_rf.add_argument("--if-missing", action="store_true")
    p_rf.add_argument("--evidence", required=True)

    p_sum = sub.add_parser("summary")
    p_sum.add_argument("--evidence", required=True)

    args = parser.parse_args(argv)
    evidence = pathlib.Path(args.evidence)
    phase_map = {
        "pointer": "resolve_pointer",
        "checkout": "checkout_candidate",
        "bind": "bind_candidate",
        "verify-package": "verify_package",
    }
    candidate_arg = getattr(args, "candidate", "") or ""
    try:
        if args.command == "pointer":
            sha = resolve_pointer(pathlib.Path(args.pointer_file))
            candidate_arg = sha
            if args.stage_from and args.stage_to:
                park = (pathlib.Path(args.park_checkout), pathlib.Path(args.park_to)) if args.park_checkout else None
                _stage_harness(pathlib.Path(args.stage_from), pathlib.Path(args.stage_to), park)
            _outputs(args.github_output, {"candidate_sha": sha})
            print(json.dumps({"candidate_sha": sha}, indent=2))
            return 0
        if args.command == "checkout":
            out = checkout(pathlib.Path(args.repo).resolve(), args.candidate.strip().lower(), args.source_url)
            print(json.dumps(out, indent=2))
            return 0
        if args.command == "bind":
            sha = args.candidate.strip().lower()
            binding = bind(pathlib.Path(args.repo).resolve(), sha, args.mirror_url, args.source_url)
            write_json_atomic(evidence / "build-binding.json", binding)
            _outputs(args.github_output, {"version": binding["version"], "release_tag": binding["release_tag"]})
            print(json.dumps(binding, indent=2))
            return 0
        if args.command == "verify-package":
            sha = args.candidate.strip().lower()
            binding = json.loads((evidence / "build-binding.json").read_text(encoding="utf-8"))
            package = verify_package(pathlib.Path(args.repo).resolve(), sha, args.mirror_url, binding)
            public_package = {k: v for k, v in package.items() if k != "archive_path"}
            write_json_atomic(evidence / "package.json", public_package)
            _outputs(
                args.github_output,
                {
                    "archive_name": package["archive_name"],
                    "archive_path": package["archive_path"],
                    "archive_sha256": package["archive_sha256"],
                },
            )
            print(json.dumps({k: public_package[k] for k in ("archive_name", "archive_sha256")}, indent=2))
            return 0
        if args.command == "record-failure":
            record_structured_result(
                evidence,
                verdict=args.verdict,
                phase=args.phase,
                code=args.code,
                candidate_sha=args.candidate,
                overwrite_existing=not args.if_missing,
            )
            return 0
        if args.command == "summary":
            sys.stdout.write(render_summary(evidence))
            return 0
    except Refusal as refusal:
        record_structured_result(
            evidence,
            verdict=refusal.verdict,
            phase=phase_map.get(args.command, args.command),
            code=refusal.reason,
            candidate_sha=candidate_arg,
            fields=refusal.fields,
        )
        sys.stderr.write(f"::error::{refusal}\n")
        return 2 if refusal.verdict == "unconfirmed" else 1
    except Exception as exc:  # a harness defect still leaves a structured result, never a bare traceback
        record_structured_result(evidence, verdict="error", phase=phase_map.get(args.command, args.command),
                                 code="harness_error", candidate_sha=candidate_arg, fields={"type": type(exc).__name__})
        sys.stderr.write(f"::error::harness_error {type(exc).__name__}\n")
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
