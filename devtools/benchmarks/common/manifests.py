"""Benchmark manifest helpers.

These helpers intentionally record reproducibility metadata without storing
secret values or full repository diffs. Official benchmark harnesses remain the
scoring authority; these manifests make Ouroboros-side task coverage and run
provenance auditable.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import pathlib
import subprocess
import sys
import time
import urllib.request
from collections.abc import Iterator, Mapping
from typing import Any


_SUBPROCESS_RUN = subprocess.run

# Named override for the provenance/attestation hard stop (owner Q8): a run against a
# LEGITIMATELY evolved volume (E1v2 self-modification) may proceed, but only when the
# operator says so explicitly and the manifest records that it happened.
ALLOW_EVOLVED_VOLUME_ENV = "OBO_ALLOW_EVOLVED_VOLUME"

# The ONLY failure the override above may waive. It authorises a deliberately evolved or
# version-skewed runtime — nothing more. A `runtime_unreachable` (any transport or parse
# failure against /api/health) means no live runtime identity was established at all,
# `runtime_version_absent` means the endpoint did answer but not with the frozen health
# contract (so it is not an Ouroboros gateway), and `commit_unavailable` means there is no
# local commit to attribute the numbers to; waiving any of them would let admission continue
# while attesting NOTHING, which is precisely the fail-open the gate exists to remove. Those
# stay fail-closed with the override set.
OVERRIDABLE_ATTESTATION_REASONS = ("runtime_skew",)

# Provenance refusals that are properties of the VOLUME plus the mounted seed rather than of one
# task: the same volume refuses every remaining task identically, so continuing burns the whole
# schedule and still reports a zero headline as success. `e1v2/entrypoint_pro.sh` emits these as
# `SOLVE_INFRA_SUSPECT reason=...`; BOTH SWE-Pro drivers consume this ONE authority
# (`e1v2/run_pro.py` stops its schedule, `e1v2/auto_run.py` stops the shard). A per-driver copy is
# exactly how run_pro ended up with no stop at all while auto_run had one. `runtime_unreachable`
# belongs here because the entrypoint reaches that check only after readiness already read
# /api/health, so an empty runtime_version is a property of the code on the volume.
CAMPAIGN_FATAL_PROVENANCE_REASONS = frozenset({
    "stamp_absent", "seed_mismatch", "lineage_broken", "runtime_skew", "runtime_unreachable",
    "seed_head_unreadable",
})

# Active projection used by every NEW run manifest and preflight. Heavy is not an
# execution slot after Available subagents and must not leak in from ambient env/settings.
ACTIVE_MODEL_SLOT_KEYS = (
    "OUROBOROS_MODEL",
    "OUROBOROS_MODEL_LIGHT",
    "OUROBOROS_MODEL_VISION",
    "OUROBOROS_MODEL_CONSCIOUSNESS",
    "OUROBOROS_MODEL_FALLBACKS",
    "OUROBOROS_MODEL_DEEP_SELF_REVIEW",
    "OUROBOROS_WEBSEARCH_MODEL",
    "OUROBOROS_REVIEWER_SLOTS",
    "OUROBOROS_REVIEW_MODELS",
    "OUROBOROS_SCOPE_REVIEW_MODELS",
    "OUROBOROS_SCOPE_REVIEW_MODEL",
    "OUROBOROS_EFFORT_TASK",
    "OUROBOROS_EFFORT_REVIEW",
    "OUROBOROS_EFFORT_SCOPE_REVIEW",
)

# Historical READ vocabulary. Old durable manifests can still carry Heavy and retain
# their original meaning; new writers use ACTIVE_MODEL_SLOT_KEYS above. Keep the public
# name for compatibility with existing artifact readers and cleanup code.
MODEL_SLOT_KEYS = (
    "OUROBOROS_MODEL",
    "OUROBOROS_MODEL_HEAVY",
    "CLAUDE_CODE_MODEL",  # retired transport slot; old manifests still carry it
    *ACTIVE_MODEL_SLOT_KEYS[1:],
)


def read_json(path: pathlib.Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: pathlib.Path, payload: Any) -> None:
    """Atomically overwrite ``path`` with pretty JSON plus a trailing newline.

    ``ouroboros.utils.atomic_write_json`` is imported LAZILY on purpose: this module is
    stdlib-only at import time and is imported by every benchmark launcher, including the
    container-side Terminal-Bench agent, so a module-level ``ouroboros`` import would make
    the runtime package a hard import dependency of all benchmark families.
    ``trailing_newline=True`` (the helper's default is False) preserves the byte-for-byte
    form of every manifest/sidecar written before this change. Note that an existing
    SYMLINK at ``path`` is replaced by a regular file rather than written through — that is
    the atomic helper's documented confinement behaviour; benchmark sidecars are never
    file symlinks (the CephFS run archive uses directory-level links, which are unaffected).
    """
    from ouroboros.utils import atomic_write_json

    atomic_write_json(pathlib.Path(path), payload, trailing_newline=True)


def read_jsonl(path: pathlib.Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        loaded = json.loads(line)
        if isinstance(loaded, dict):
            rows.append(loaded)
    return rows


def write_jsonl(path: pathlib.Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + ("\n" if rows else ""),
        encoding="utf-8",
    )


def _run(args: list[str], cwd: pathlib.Path, *, timeout: int = 10) -> subprocess.CompletedProcess[str]:
    def _text(value: str | bytes | None, fallback: str = "") -> str:
        if isinstance(value, bytes):
            return value.decode("utf-8", errors="replace")
        if isinstance(value, str):
            return value
        return fallback

    try:
        return _SUBPROCESS_RUN(args, cwd=cwd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        return subprocess.CompletedProcess(args, 124, stdout=_text(exc.stdout), stderr=_text(exc.stderr, "timeout"))
    except Exception as exc:
        return subprocess.CompletedProcess(args, 1, stdout="", stderr=f"{type(exc).__name__}: {exc}")


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


def repo_provenance(repo_dir: pathlib.Path) -> dict[str, Any]:
    """Return non-secret source provenance for a repo checkout."""
    repo = pathlib.Path(repo_dir).expanduser().resolve(strict=False)
    result: dict[str, Any] = {
        "repo_dir": str(repo),
        "python": sys.version.split()[0],
        "schema": "ouroboros.benchmark.repo_provenance.v1",
    }
    version_path = repo / "VERSION"
    if version_path.exists():
        result["version"] = version_path.read_text(encoding="utf-8").strip()

    head = _run(["git", "rev-parse", "HEAD"], repo)
    if head.returncode != 0:
        result["git_available"] = False
        result["git_error"] = (head.stderr or head.stdout or "").strip()
        return result

    result["git_available"] = True
    result["head"] = head.stdout.strip()
    branch = _run(["git", "branch", "--show-current"], repo)
    result["branch"] = branch.stdout.strip() if branch.returncode == 0 else ""
    describe = _run(["git", "describe", "--tags", "--dirty", "--always"], repo)
    result["describe"] = describe.stdout.strip() if describe.returncode == 0 else ""
    status = _run(["git", "status", "--porcelain=v1", "--untracked-files=all"], repo)
    status_text = status.stdout if status.returncode == 0 else ""
    tracked_diff = _run(["git", "diff", "--binary", "HEAD", "--"], repo, timeout=30)
    tracked_diff_text = tracked_diff.stdout if tracked_diff.returncode == 0 else ""
    result.update(
        {
            # WHETHER the probe answered is a separate fact from what it said. A failed probe
            # (10s timeout on a huge untracked tree / CephFS, or a corrupt .git/index) used to
            # collapse into `dirty: False` — a genuinely dirty seed would then pass the gate and
            # a paid run would start with `…-dirty` provenance. The gate refuses on this flag.
            "status_available": status.returncode == 0,
            "tracked_diff_available": tracked_diff.returncode == 0,
            "dirty": bool(status_text.strip()),
            "status_entries": len([line for line in status_text.splitlines() if line.strip()]),
            "status_sha256": _sha256_text(status_text) if status_text.strip() else "",
            "tracked_diff_sha256": _sha256_text(tracked_diff_text) if tracked_diff_text.strip() else "",
        }
    )
    return result


def allow_evolved_volume() -> bool:
    """True when the operator explicitly authorised an evolved/skewed runtime."""
    return str(os.environ.get(ALLOW_EVOLVED_VOLUME_ENV, "")).strip().lower() in {"1", "true", "yes"}


class SeedShapeRefused(RuntimeError):
    """The mounted seed checkout has the wrong SHAPE to be provenance-attributable.

    Third member of this phase's typed-refusal family, alongside ``BenchmarkAdmissionRefused``
    and ``RuntimeAttestationRefused``: a ``RuntimeError`` subclass carrying the typed ``reason``,
    so a launcher's post-admission handler catches it BY CONSTRUCTION. The check it replaces
    raised ``SystemExit`` — a ``BaseException``, not an ``Exception`` — which made every
    ``except Exception`` handler around it inert: the refusal was recorded as a generic ``crashed``
    instead of the typed ``seed_shape`` refusal the launcher promised.
    """

    def __init__(self, reason: str, message: str) -> None:
        super().__init__(message)
        self.reason = reason


class RuntimeAttestationRefused(RuntimeError):
    """A refused runtime attestation that CARRIES the record it just built.

    Mirrors ``BenchmarkAdmissionRefused``: ``runtime_attestation`` assembles the full record
    (typed ``reason``, ``runtime_version``, ``repo_head``, ``repo_version``, override facts) and
    used to discard it behind a plain ``RuntimeError``, so a launcher that caught the refusal could
    only persist a generic message — losing the exact reason and the identities this provenance
    contract exists to preserve, at the moment they matter most. Subclasses ``RuntimeError`` so
    every existing caller and test is unaffected.
    """

    def __init__(self, message: str, attestation: dict[str, Any]) -> None:
        super().__init__(message)
        self.attestation = attestation


def runtime_attestation(
    base_url: str,
    repo_dir: pathlib.Path,
    *,
    expected_version: str = "",
    timeout: int = 10,
) -> dict[str, Any]:
    """Attest the identity of a RUNNING Ouroboros against the checkout it was started from.

    Three facts that earlier provenance work conflated (owner Q7=B — record BOTH):

    * ``runtime_version`` — what the live process reports over HTTP. Read from the frozen
      auth-exempt ``GET /api/health`` contract; this helper never changes that contract.
    * ``repo_head`` — the local HEAD commit of ``repo_dir`` (``repo_provenance``).
    * ``repo_version`` — the ``VERSION`` file next to that checkout.

    A skew between the live process and its own checkout means the numbers a run produces
    cannot be attributed to a commit, so this fails CLOSED (owner Q8) with the named
    override ``OBO_ALLOW_EVOLVED_VOLUME=1``; the returned record always says whether the
    override was used, so an overridden run stays auditable instead of silent.

    The override is DELIBERATELY narrow: it waives only
    ``OVERRIDABLE_ATTESTATION_REASONS`` (the evolved/skewed runtime it is documented for).
    ``runtime_unreachable``, ``runtime_version_absent`` and ``commit_unavailable`` still raise
    with the override set — no live identity, no health contract and no local commit are not
    "evolution", they are "nothing was attested". The record reports ``override_set`` (was it exported), ``overridden`` (did it
    waive THIS reason) and ``override_waives`` (what it can ever waive).
    """
    url = str(base_url).rstrip("/") + "/api/health"
    payload: dict[str, Any] = {}
    http_error = ""
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:  # noqa: S310 - operator-supplied local URL
            raw = resp.read().decode("utf-8", errors="replace")
        loaded = json.loads(raw) if raw.strip() else {}
        if isinstance(loaded, dict):
            payload = loaded
    except Exception as exc:  # noqa: BLE001 - any transport/parse failure is "unreachable"
        http_error = f"{type(exc).__name__}: {exc}"
    source = repo_provenance(repo_dir)
    # ONLY the contracted field. `runtime_version` is part of the frozen `HealthResponse`
    # (`ouroboros/gateway/contracts.py`), so its absence is not a version to guess at — it means
    # the endpoint is not an Ouroboros gateway, or is an incompatible one. Falling back to a
    # generic `version` key let ANY server that happens to return `{"version": "6.75.0"}` attest
    # successfully, which is the fail-open this whole gate exists to remove, and it sat on the
    # DEFAULT ProgramBench admission path.
    runtime_version = str(payload.get("runtime_version") or "")
    attestation: dict[str, Any] = {
        "schema": "ouroboros.benchmark.runtime_attestation.v1",
        "checked_at_unix": time.time(),
        "base_url": str(base_url),
        "runtime_version": runtime_version,
        "repo_dir": str(source.get("repo_dir") or ""),
        "repo_head": str(source.get("head") or ""),
        "repo_version": str(source.get("version") or ""),
        "expected_version": str(expected_version or ""),
        "http_error": http_error,
        "override_env": ALLOW_EVOLVED_VOLUME_ENV,
        "override_set": allow_evolved_volume(),
        "override_waives": list(OVERRIDABLE_ATTESTATION_REASONS),
        "overridden": False,
    }
    # ORDER MATTERS and is the fail-closed one: availability of the two identities is decided
    # BEFORE any comparison between them. With the skew arms first, a checkout with NO readable
    # commit that also disagreed on the version was labelled `runtime_skew` — an OVERRIDABLE
    # reason — so `OBO_ALLOW_EVOLVED_VOLUME=1` waived a run that had no commit to attribute its
    # numbers to at all. `commit_unavailable` is never waivable, so it has to be reached first.
    if http_error:
        reason = "runtime_unreachable"
    elif not runtime_version:
        # Answered, but not with the frozen health contract: a DIFFERENT fact from "unreachable",
        # and non-overridable for the same reason — no live runtime identity was established.
        reason = "runtime_version_absent"
    elif not attestation["repo_head"]:
        reason = "commit_unavailable"
    elif expected_version and runtime_version != expected_version:
        reason = "runtime_skew"
    elif attestation["repo_version"] and runtime_version != attestation["repo_version"]:
        reason = "runtime_skew"
    else:
        reason = ""
    attestation["reason"] = reason
    attestation["ok"] = not reason
    if reason:
        if attestation["override_set"] and reason in OVERRIDABLE_ATTESTATION_REASONS:
            attestation["overridden"] = True
        else:
            hint = (
                f"set {ALLOW_EVOLVED_VOLUME_ENV}=1 to run against a deliberately evolved runtime"
                if reason in OVERRIDABLE_ATTESTATION_REASONS
                else f"{ALLOW_EVOLVED_VOLUME_ENV} does NOT waive reason={reason} (it waives only "
                     f"{', '.join(OVERRIDABLE_ATTESTATION_REASONS)}): no live runtime identity or no "
                     "local commit was established at all, so there is nothing to attest"
            )
            raise RuntimeAttestationRefused(
                f"runtime attestation failed reason={reason} base_url={base_url} "
                f"runtime_version={runtime_version!r} repo_version={attestation['repo_version']!r} "
                f"repo_head={attestation['repo_head'][:12]!r} http_error={http_error!r}; "
                f"override_set={attestation['override_set']}; {hint}",
                attestation,
            )
    return attestation


def commit_lineage_ok(seed_head: str, head: str, repo_dir: pathlib.Path) -> bool:
    """True when ``head`` IS ``seed_head`` or a descendant of it.

    Evolution runs legitimately move HEAD forward (owner Q11), so provenance must compare a
    LINE OF DESCENT, not equality and not set membership. An unknown/absent commit is False
    (fail closed) rather than "probably fine".
    """
    seed = str(seed_head or "").strip()
    live = str(head or "").strip()
    if not seed or not live:
        return False
    if seed == live:
        return True
    repo = pathlib.Path(repo_dir)
    return _run(["git", "merge-base", "--is-ancestor", seed, live], repo, timeout=30).returncode == 0


def openrouter_key_remaining(api_key: str, *, timeout: int = 10) -> float | None:
    """Remaining USD budget for an OpenRouter key, or None when the key is UNCAPPED.

    ``GET /api/v1/key`` returns the authoritative ``data.limit_remaining``; that field is
    the source of truth. ``limit - usage`` is only a fallback for responses that omit it,
    and ``limit: null`` means "no cap" (None, never 0.0 and never "plenty"). Credit-endpoint
    arithmetic over ``total_credits - total_usage`` is deliberately NOT used: it lies on a
    nearly exhausted key (a key with $0.23 left passed that check and killed half a run).

    Raises on an unusable key (unauthorized / unparseable) so callers fail closed instead of
    reading an error body as headroom. Note that shared keys can still report headroom that
    a neighbour spends before the run starts — only a real completion is proof.
    """
    key = str(api_key or "").strip()
    if not key:
        raise RuntimeError("openrouter_key_remaining requires an API key")
    request = urllib.request.Request(
        "https://openrouter.ai/api/v1/key",
        headers={"Authorization": f"Bearer {key}"},
        method="GET",
    )
    with urllib.request.urlopen(request, timeout=timeout) as resp:  # noqa: S310 - fixed provider URL
        payload = json.loads(resp.read().decode("utf-8", errors="replace") or "{}")
    data = payload.get("data") if isinstance(payload.get("data"), dict) else payload
    if not isinstance(data, dict):
        raise RuntimeError("OpenRouter /api/v1/key returned no key object")
    if "limit_remaining" in data and data.get("limit_remaining") is not None:
        return float(data["limit_remaining"])
    if data.get("limit") is None:
        return None
    return float(data.get("limit") or 0.0) - float(data.get("usage") or 0.0)


def openrouter_account_credits(api_key: str, *, timeout: int = 10) -> float | None:
    """The ACCOUNT balance behind an OpenRouter key (``total_credits - total_usage``), or None
    when the endpoint does not report both numbers.

    Never a headroom claim on its own — ``openrouter_key_remaining`` above documents why the
    credits arithmetic alone lies. It exists because the key's limit is not money either: a key
    capped at $12k on an account holding $2k is bounded by the ACCOUNT, and a preflight reading
    only ``limit_remaining`` would start a run the balance cannot finish. Callers take
    ``min(openrouter_key_remaining(key), openrouter_account_credits(key))`` over the values that
    are not None; this helper is always the SECOND bound of that min.
    """
    key = str(api_key or "").strip()
    if not key:
        raise RuntimeError("openrouter_account_credits requires an API key")
    request = urllib.request.Request(
        "https://openrouter.ai/api/v1/credits",
        headers={"Authorization": f"Bearer {key}"},
        method="GET",
    )
    with urllib.request.urlopen(request, timeout=timeout) as resp:  # noqa: S310 - fixed provider URL
        payload = json.loads(resp.read().decode("utf-8", errors="replace") or "{}")
    data = payload.get("data") if isinstance(payload.get("data"), dict) else payload
    if not isinstance(data, dict) or data.get("total_credits") is None or data.get("total_usage") is None:
        return None
    return float(data["total_credits"]) - float(data["total_usage"])


def model_slot_snapshot(settings_path: pathlib.Path | None = None, *,
                        env_overrides: bool = True) -> dict[str, str]:
    """Return configured model/review slots without exposing provider secrets.

    ``env_overrides`` models how the server being described gets its configuration. A server
    started in THIS process's environment reads settings.json but lets the environment win, so
    the launcher's env belongs in the snapshot (the default). A server started in a CONTAINER
    is handed the settings FILE and a fresh environment, so the launcher's own env is not part
    of its configuration at all and must not be reported as if it were: pass ``False`` there.
    """
    settings: dict[str, Any] = {}
    if settings_path and pathlib.Path(settings_path).exists():
        try:
            loaded = json.loads(pathlib.Path(settings_path).read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                settings = loaded
        except Exception:
            settings = {}
    slots: dict[str, str] = {}
    for key in ACTIVE_MODEL_SLOT_KEYS:
        value = os.environ.get(key) if env_overrides else None
        if value is None:
            value = settings.get(key)
        if value not in (None, ""):
            slots[key] = str(value)
    return slots


def provider_credential_disclosure(
    settings_path: pathlib.Path | None = None,
    *,
    runtime_credentials: Mapping[str, Any] | None = None,
    include_claude_sdk_defaults: bool = True,
) -> dict[str, Any]:
    """Record WHICH provider credentials the described server can reach — by fingerprint.

    ``model_slots`` already says which models a run declared; this says which providers it
    could actually have spent on, which is not the same fact and is the one a routing
    fallback can falsify. The settings-file grant and an explicitly injected runtime grant
    are reported as separate fingerprint maps: a benchmark may keep a long-lived key out of
    its persisted settings while passing it to one server process. Values are never returned.
    An absent/unreadable settings path yields ``{"available": False}`` — a stated gap, never
    a silently empty grant list."""
    from devtools.benchmarks.common.secrets import credential_disclosure, isolated_credential_grants
    from ouroboros.provider_models import ALL_PROVIDER_CREDENTIAL_KEYS

    runtime_granted: dict[str, dict[str, Any]] = {}
    if runtime_credentials is not None:
        selected = {
            str(key): value
            for key, value in runtime_credentials.items()
            if str(key) in ALL_PROVIDER_CREDENTIAL_KEYS
        }
        runtime_granted = credential_disclosure(selected, sorted(selected))

    def with_runtime(payload: dict[str, Any]) -> dict[str, Any]:
        if runtime_credentials is not None:
            payload["runtime_granted"] = runtime_granted
        return payload

    path = pathlib.Path(settings_path) if settings_path else None
    if not path or not path.exists():
        return with_runtime({"available": False, "reason": "settings_path_absent"})
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return with_runtime({"available": False, "reason": "settings_unreadable"})
    if not isinstance(loaded, dict):
        return with_runtime({"available": False, "reason": "settings_not_an_object"})
    return with_runtime({
        "available": True,
        **isolated_credential_grants(
            loaded,
            include_claude_sdk_defaults=include_claude_sdk_defaults,
        ),
    })


class BenchmarkAdmissionRefused(RuntimeError):
    """A refused admission gate that CARRIES the manifest describing what it refused.

    Subclasses ``RuntimeError`` so every existing caller and test that expects a ``RuntimeError``
    from the gate keeps working. The payload exists because enforcement used to raise from inside
    ``benchmark_run_manifest`` BEFORE the dict reached any caller, so no launcher could persist
    the refusal: a refused run left nothing on disk but a stderr line a shard launcher discards.
    ``admit_benchmark_run`` is the shared helper that writes ``manifest`` before re-raising.
    """

    def __init__(self, message: str, manifest: dict[str, Any]) -> None:
        super().__init__(message)
        self.manifest = manifest


def admit_benchmark_run(manifest_path: pathlib.Path, **manifest_kwargs: Any) -> dict[str, Any]:
    """ADMISSION seam: build the run manifest, PERSIST it, then let the gate enforce.

    Every migrated launcher goes through here instead of pairing ``benchmark_run_manifest()``
    with its own ``write_json()``, because the ordering is the whole point: the complete
    gate/refusal payload reaches disk BEFORE the refusal propagates, so a refused run is as
    auditable as an admitted one. Returns the retained manifest dict, which the launcher then
    augments and hands to ``finalize_run_manifest``.
    """
    try:
        manifest = benchmark_run_manifest(**manifest_kwargs)
    except BenchmarkAdmissionRefused as refused:
        write_json(pathlib.Path(manifest_path), refused.manifest)
        raise
    write_json(pathlib.Path(manifest_path), manifest)
    return manifest


@contextlib.contextmanager
def finalize_run_manifest(
    manifest_path: pathlib.Path,
    manifest: dict[str, Any],
    *,
    outcome: str = "completed",
    exit_code: int = 0,
) -> Iterator[dict[str, Any]]:
    """FINALIZATION seam: the ONE place a migrated launcher records how its run ENDED.

    Yields a mutable ``final`` mapping which is merged into ``manifest["extra"]`` and written to
    ``manifest_path`` on EVERY exit path — normal completion, an early typed return, and an
    escaping exception alike. A launcher records its own terminal vocabulary by updating
    ``final`` (``outcome``, ``exit_code``, and a typed ``refusal`` block); anything that escapes
    the block additionally gets a typed ``error`` and, unless the launcher already named a
    terminal outcome, ``outcome="crashed"``. The one thing a finished run's own record must never
    still say is ``started``, which is exactly what every unhandled failure used to leave behind.
    An escaping ``SystemExit`` keeps its integer status, so the recorded ``exit_code`` is the one
    the process really exits with.
    """
    default_outcome = str(outcome)
    final: dict[str, Any] = {"outcome": default_outcome, "exit_code": int(exit_code)}
    try:
        yield final
    except BaseException as exc:
        if str(final.get("outcome") or "") == default_outcome:
            final["outcome"] = "crashed"
        final["error"] = {"type": type(exc).__name__, "message": str(exc)[:4000]}
        if not int(final.get("exit_code") or 0):
            # A SystemExit CARRIES the status the process will actually exit with, so flattening
            # it to 1 makes the record disagree with reality (auto_run's campaign-fatal
            # provenance refusal exits 2). Any other escaping exception is a generic failure.
            code = exc.code if isinstance(exc, SystemExit) else None
            final["exit_code"] = int(code) if isinstance(code, int) else 1
        raise
    finally:
        manifest.setdefault("extra", {}).update(final)
        write_json(pathlib.Path(manifest_path), manifest)


def _seed_gate(source: dict[str, Any], *, require_clean: bool, expect: str) -> dict[str, Any]:
    """Evaluate the seed-provenance gate over an already-collected provenance record.

    Fail CLOSED by default (owner Q19=B): a dirty seed makes the manifest write
    ``…-dirty`` and the whole run unreproducible, and a missing git identity — or a
    cleanliness probe that could not answer at all — is just as bad: a run whose source
    cannot be named is not submittable. The escape is a recorded
    ``require_clean=False`` (launchers expose it as ``--allow-dirty-seed``), not a new run
    class. ``expect`` optionally pins the seed to a commit prefix or a VERSION string.

    This helper only DECIDES; it never raises. Enforcement lives in
    ``benchmark_run_manifest``, which raises ``BenchmarkAdmissionRefused`` only after the whole
    manifest (including this verdict) exists, so ``admit_benchmark_run`` can persist it.
    """
    head = str(source.get("head") or "")
    version = str(source.get("version") or "")
    gate: dict[str, Any] = {
        "require_clean": bool(require_clean),
        "allow_dirty_seed": not bool(require_clean),
        "expect": str(expect or ""),
        "git_available": bool(source.get("git_available")),
        "status_available": bool(source.get("status_available", True)),
        "dirty": bool(source.get("dirty")),
        "describe": str(source.get("describe") or ""),
        "reason": "",
    }
    if expect and not (head.startswith(expect) or version == str(expect)):
        gate["reason"] = "seed_mismatch"
    elif not source.get("git_available"):
        gate["reason"] = "seed_identity_unavailable"
    elif not source.get("status_available", True):
        # UNKNOWN cleanliness is not cleanliness: `dirty: False` from a probe that never ran
        # would let a dirty seed through the gate. Checked BEFORE `dirty` because the flag is
        # exactly the case where the `dirty` value carries no information.
        gate["reason"] = "seed_status_unavailable"
    elif source.get("dirty"):
        gate["reason"] = "seed_dirty"
    gate["ok"] = not gate["reason"]
    return gate


def _seed_gate_refusal(gate: dict[str, Any], source: dict[str, Any]) -> str:
    """The refusal message for a gate verdict, or ``""`` when the run is admissible.

    ``seed_mismatch`` is refused even with ``--allow-dirty-seed``: 'dirty' and 'wrong commit'
    are different facts and only the first one has a recorded escape.
    """
    reason = str(gate.get("reason") or "")
    if not reason or not (gate.get("require_clean") or reason == "seed_mismatch"):
        return ""
    return (
        f"seed provenance gate failed reason={reason} repo_dir={source.get('repo_dir')!r} "
        f"describe={gate.get('describe')!r} status_entries={source.get('status_entries')} "
        f"expect={gate.get('expect')!r}; commit or clean the seed checkout (a dirty seed writes a "
        "'-dirty' manifest and is not submittable), or pass --allow-dirty-seed to record the "
        "exception deliberately"
    )


def benchmark_run_manifest(
    *,
    benchmark: str,
    run_root: pathlib.Path,
    repo_dir: pathlib.Path,
    requested_task_ids: list[str],
    require_clean: bool = True,
    expect: str = "",
    metadata: dict[str, Any] | None = None,
    **overrides: Any,
) -> dict[str, Any]:
    """Build the shared benchmark run manifest AND run the seed-provenance gate.

    Callers must build this ONCE, as early as possible (right after argument parsing or
    readiness) and keep the returned dict: it is the gate that stops an unreproducible run
    BEFORE the first paid task, then the same dict is augmented and written at the end.
    Parameter budget note: this signature sits exactly on the 8-parameter cap asserted by
    ``tests/test_devtools_benchmarks.py`` — further knobs ride through ``metadata`` /
    ``**overrides``, never as new parameters.

    A refusal raises ``BenchmarkAdmissionRefused`` (a ``RuntimeError``) only once the COMPLETE
    manifest exists, and carries it on the exception: prefer ``admit_benchmark_run()``, which
    writes that payload before re-raising, so the refusal leaves a durable record of what was
    refused and why.
    """
    meta = dict(metadata or {})
    for key, value in overrides.items():
        if value is not None:
            meta[key] = value
    meta_settings_path = meta.get("settings_path")
    if meta_settings_path is not None:
        meta_settings_path = pathlib.Path(meta_settings_path)
    include_claude_sdk_defaults = meta.get("include_claude_sdk_defaults", True)
    if not isinstance(include_claude_sdk_defaults, bool):
        raise ValueError("include_claude_sdk_defaults metadata must be a boolean")
    # Some adapters hand a settings file plus a fresh child environment to a
    # container.  Their initial/refusal manifest must describe that file, not
    # ambient model/effort values from the operator process.  Keep the
    # historical env-first default for all other benchmark drivers.
    settings_authoritative_env = meta.get("settings_authoritative_env", False)
    if not isinstance(settings_authoritative_env, bool):
        raise ValueError("settings_authoritative_env metadata must be a boolean")
    source = repo_provenance(repo_dir)
    gate = _seed_gate(source, require_clean=require_clean, expect=str(expect or ""))
    manifest: dict[str, Any] = {
        "schema": "ouroboros.benchmark.run_manifest.v1",
        "benchmark": benchmark,
        "created_at_unix": time.time(),
        "run_root": str(pathlib.Path(run_root).expanduser().resolve(strict=False)),
        "requested_task_ids": [str(task_id) for task_id in requested_task_ids],
        "requested_count": len(requested_task_ids),
        "argv": list(meta["argv"]) if "argv" in meta else list(sys.argv),
        "dataset": str(meta.get("dataset") or ""),
        "harness": meta.get("harness") or {},
        "official_command": meta.get("official_command") or [],
        "timeout_sec": meta.get("timeout_sec"),
        "isolated_data_root": str(meta.get("isolated_data_root") or ""),
        "output_paths": meta.get("output_paths") or {},
        "model_slots": model_slot_snapshot(
            meta_settings_path,
            env_overrides=not settings_authoritative_env,
        ),
        "provider_credentials": provider_credential_disclosure(
            meta_settings_path,
            include_claude_sdk_defaults=include_claude_sdk_defaults,
        ),
        "source": source,
        "seed_gate": gate,
        "extra": meta.get("extra") or {},
    }
    refusal = _seed_gate_refusal(gate, source)
    if refusal:
        # The refused payload speaks the same terminal vocabulary as `finalize_run_manifest`, so
        # a refusal and a completed run are read the same way by whoever audits the sidecar.
        manifest["extra"] = {
            **(manifest["extra"] if isinstance(manifest["extra"], dict) else {}),
            "outcome": "refused",
            "exit_code": 1,
            "refusal": {"stage": "seed_gate", "reason": str(gate["reason"]), "exit_code": 1},
        }
        raise BenchmarkAdmissionRefused(refusal, manifest)
    return manifest
