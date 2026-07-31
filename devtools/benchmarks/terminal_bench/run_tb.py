#!/usr/bin/env python3
"""Terminal-Bench 2.1 runner/submission helper for Ouroboros.

The official leaderboard requires at least k=5 trials, default timeout/resource
settings, metadata.yaml, and full Harbor artifacts. This wrapper keeps those
methodology constraints visible instead of relying on ad-hoc shell history.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import re
import shlex
import subprocess
import sys
from dataclasses import dataclass

if __package__ in {None, ""}:
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3]))

from devtools.benchmarks.common.manifests import (
    MODEL_SLOT_KEYS,
    admit_benchmark_run,
    finalize_run_manifest,
    model_slot_snapshot,
    write_json,
)
from devtools.benchmarks.common.run_roots import (
    default_settings_path,
    assert_outside_repo,
    ensure_outside_repo,
    repo_root_from_devtools,
    safe_benchmark_id,
    safe_join_under,
)
from devtools.benchmarks.terminal_bench.run_harbor_smoke import AGENT_IMPORT
from ouroboros.config import SETTINGS_DEFAULTS


DEFAULT_DATASET = "terminal-bench/terminal-bench-2-1"

# Frontier-Bench (Terminal-Bench's successor, harbor + Laude Institute). It is a HARBOR DATASET
# with the same task shape as TB2.x (`task.toml` + `instruction.md` + `environment/Dockerfile` +
# `tests/test.sh` + `solution/solve.sh`), so it needs NO new launcher and NO new package — only
# this identity. Harbor resolves it as `<org>/<name>` and defaults the ref to `latest`, which is a
# MUTABLE tag row: a reproducible run must pin an immutable ref via `--dataset
# frontier-bench/frontier-bench@v0.1.0` (or `@<revision>` / `@sha256:<digest>`). Verified on the
# installed harbor 0.18.0 with a negative control: `@v0.1.0` resolves 74 tasks while a bogus
# `@v9.9.9` exits non-zero. `harbor dataset download frontier-bench/frontier-bench --cache` populates
# `~/.cache/harbor/tasks/packages/frontier-bench/<task>/<digest>/task.toml`, which is exactly the
# subtree the adapter's dataset-parametric wall-clock lookup already globs.
FRONTIER_BENCH_DATASET = "frontier-bench/frontier-bench"

# Every model slot the in-container adapter forwards (plus the review triad, handled specially).
# Used by --all-model for a single-model run: review STAYS ON but lightened to ONE reviewer at low
# effort by default (configurable). Exported to os.environ so the harbor subprocess and the adapter's forwarded-slot reads
# (harbor_installed_agent._container_env) pick them up. This does NOT change repo config defaults.
# Only slots the in-container adapter actually forwards (harbor_installed_agent._container_env).
# Deliberately omitted because the container already covers them: HEAVY/CONSCIOUSNESS default to
# OUROBOROS_MODEL when empty, and the adapter pins the fallback (singular AND plural) to the main
# model itself — so setting them here would be a dead host-env write. CLAUDE_CODE_MODEL IS included
# (and the adapter forwards it) so claude_code_edit cannot introduce a different model.
_ALL_MODEL_SLOT_KEYS = (
    "OUROBOROS_MODEL",
    "OUROBOROS_MODEL_LIGHT",
    "OUROBOROS_MODEL_DEEP_SELF_REVIEW",
    "OUROBOROS_WEBSEARCH_MODEL",
    "OUROBOROS_SCOPE_REVIEW_MODELS",
    "OUROBOROS_SCOPE_REVIEW_MODEL",
    "CLAUDE_CODE_MODEL",
)


def apply_all_model(model: str, review_slots: int = 1, review_effort: str = "low") -> None:
    """Force every FORWARDED model slot to ``model`` for a single-model run. Mutates only this
    process's env, which propagates to the harbor subprocess and the in-container adapter; it does
    NOT edit repo config defaults. The container's HEAVY/CONSCIOUSNESS slots default to
    OUROBOROS_MODEL and the adapter pins the fallback to the main model, so those need no explicit
    value here.

    Review for a single-model run defaults to ONE reviewer at ``low`` effort (configurable via
    --review-slots / --review-effort): three identical-model reviewers add latency/cost but no
    diversity (a monoculture), and a single-model run cannot achieve reviewer-model diversity anyway.
    This is a BENCHMARK setting, NOT a claim that the review subsystem got more reliable
    (single_reviewer_no_diversity stays loud). EFFORT_SCOPE_REVIEW is set too for completeness
    ("там и там"); scope review does not fire on a terminal-bench task (it is a commit-time gate)."""
    for key in _ALL_MODEL_SLOT_KEYS:
        os.environ[key] = model
    os.environ["OUROBOROS_REVIEW_MODELS"] = ",".join([model] * max(1, int(review_slots)))
    os.environ["OUROBOROS_EFFORT_REVIEW"] = review_effort
    os.environ["OUROBOROS_EFFORT_SCOPE_REVIEW"] = review_effort


@dataclass
class HarborCommandConfig:
    dataset: str
    model: str
    k: int
    jobs_dir: pathlib.Path
    harbor_bin: str
    n_concurrent: int
    task_filters: list[str]
    settings_path: pathlib.Path
    execute: bool
    light_model: str
    review_enforcement: str = "blocking"
    safety_mode: str = "light"
    setup_timeout_multiplier: float = 1.0
    build_timeout_multiplier: float = 1.0
    disable_agent_web: bool = True
    base_job_config: pathlib.Path | None = None
    agent_env: tuple[str, ...] = ()
    verifier_env: tuple[str, ...] = ()
    harbor_env: str = ""


def _effective_helper_models(measured_model: str, light_model: str, *, disable_agent_web: bool = False) -> list[tuple[str, str]]:
    """Resolve EVERY model that materially assists a measured run, with its role.

    With ``task_review_mode=required`` the host forces a multi-model
    task-acceptance review whose feedback re-enters the measured agent's
    context, so the review triad / scope / light / web-search models genuinely
    assist the run. Declaring only the measured model in metadata.yaml would
    misrepresent the submission. Values mirror what the container resolves
    (env override else the shipped config defaults) so the declared set matches
    reality. Returns ordered (model_id, role) pairs, deduped by model id.
    """
    review_default = str(SETTINGS_DEFAULTS["OUROBOROS_REVIEW_MODELS"])
    websearch_default = str(SETTINGS_DEFAULTS["OUROBOROS_WEBSEARCH_MODEL"])
    scope_default = str(SETTINGS_DEFAULTS["OUROBOROS_SCOPE_REVIEW_MODELS"])
    review = os.environ.get("OUROBOROS_REVIEW_MODELS", review_default) or review_default
    scope = (os.environ.get("OUROBOROS_SCOPE_REVIEW_MODELS")
             or os.environ.get("OUROBOROS_SCOPE_REVIEW_MODEL") or scope_default)
    websearch = os.environ.get("OUROBOROS_WEBSEARCH_MODEL", websearch_default) or websearch_default
    ordered: list[tuple[str, str]] = [(measured_model, "agent")]
    for m in review.split(","):
        if m.strip():
            ordered.append((m.strip(), "commit_review_triad"))
    for m in scope.split(","):
        if m.strip():
            ordered.append((m.strip(), "scope_review"))
    if light_model.strip():
        ordered.append((light_model.strip(), "light_safety_post_task_synthesis"))
    # Only declare a web_search model if web tools are actually available this run. With
    # disable_agent_web (the default), the agent's web tools are blocked, so declaring a
    # web_search role would misrepresent the submission.
    if websearch.strip() and not disable_agent_web:
        ordered.append((websearch.strip(), "web_search"))
    # The adapter forwards CLAUDE_CODE_MODEL, so claude_code_edit can assist with that model;
    # declare it for honesty. In a single-model (--all-model) run it equals the measured model and
    # dedupes away, so this only adds a row when an operator runs a genuine multi-model ensemble.
    claude_code = os.environ.get("CLAUDE_CODE_MODEL", "").strip()
    if claude_code:
        ordered.append((claude_code, "claude_code_edit"))
    deduped: dict[str, str] = {}
    for model_id, role in ordered:
        if model_id in deduped:
            if role not in deduped[model_id].split("+"):
                deduped[model_id] = deduped[model_id] + "+" + role
        else:
            deduped[model_id] = role
    return list(deduped.items())


def leaderboard_metadata(*, agent_name: str, org_name: str, model: str, light_model: str = "", disable_agent_web: bool = False) -> str:
    lines = [
        "agent_url: https://github.com/razzant/ouroboros",
        f"agent_display_name: {json.dumps(agent_name)}",
        f"agent_org_display_name: {json.dumps(org_name)}",
        "models:",
    ]
    for model_id, role in _effective_helper_models(model, light_model, disable_agent_web=disable_agent_web):
        provider = model_id.split("/", 1)[0] if "/" in model_id else "openrouter"
        display = model_id.split("/", 1)[1] if "/" in model_id else model_id
        lines.append(f"  - model_name: {json.dumps(model_id)}")
        lines.append(f"    model_provider: {json.dumps(provider)}")
        lines.append(f"    model_display_name: {json.dumps(display)}")
        lines.append(f"    model_org_display_name: {json.dumps(provider)}")
        lines.append(f"    role: {json.dumps(role)}")
    return "\n".join(lines) + "\n"


def validate_methodology(
    *,
    k: int,
    timeout_multiplier: float,
    resource_overrides: list[str],
    setup_timeout_multiplier: float = 1.0,
    build_timeout_multiplier: float = 1.0,
    allow_setup_build_multipliers: bool = False,
    allow_low_k: bool = False,
) -> None:
    if int(k) < 5 and not allow_low_k:
        raise ValueError(
            "Terminal-Bench leaderboard mode requires k >= 5. Pass --allow-low-k for a LOCAL, "
            "non-leaderboard-valid measurement run (e.g. a k=1 first pass)."
        )
    if float(timeout_multiplier) != 1.0:
        raise ValueError("Terminal-Bench leaderboard mode requires timeout_multiplier == 1.0")
    if resource_overrides:
        raise ValueError(f"Terminal-Bench leaderboard mode forbids resource overrides: {resource_overrides}")
    # Harbor's leaderboard static_validation REJECTS non-1.0 agent_setup / environment_build
    # timeout multipliers (the top published submissions leave them null). Default them to 1.0;
    # a non-1.0 value is allowed only for an explicitly-flagged LOCAL, non-leaderboard run.
    if not allow_setup_build_multipliers and (
        float(setup_timeout_multiplier) != 1.0 or float(build_timeout_multiplier) != 1.0
    ):
        raise ValueError(
            "Terminal-Bench leaderboard mode requires agent-setup and environment-build timeout "
            f"multipliers == 1.0 (got setup={setup_timeout_multiplier}, build={build_timeout_multiplier}). "
            "Pass --allow-setup-build-multipliers for a LOCAL, non-leaderboard-valid run."
        )


def report_grade(*, k: int, leaderboard_valid: bool, low_k_floor: int = 5) -> tuple[str, str]:
    """Three-tier honesty grade for a TB run, reusing the existing leaderboard_valid
    vocabulary (NOT a parallel taxonomy):
      - leaderboard_valid : k >= 5 AND every leaderboard gate passed.
      - debug_only        : k == 1 (a single trial — noise, not a measurement).
      - local_low_k       : anything else not leaderboard-valid (1 < k < floor, or
                            k >= 5 with a non-faithful setting like web-on).
    Returns (grade, human_warning); warning is '' for a leaderboard-valid run.
    """
    if leaderboard_valid:
        return "leaderboard_valid", ""
    if int(k) <= 1:
        return "debug_only", (
            "⚠️ k=1: a SINGLE trial — this is noise, not a measurement. Do NOT cite this "
            "number; use k>=5 with leaderboard-faithful settings for any reported result."
        )
    if int(k) < int(low_k_floor):
        return "local_low_k", (
            f"⚠️ k={k} (< {low_k_floor}): LOCAL low-confidence run, NOT leaderboard-valid — "
            "high variance. Use k>=5 with leaderboard-faithful settings for any reported number."
        )
    # k >= floor but NOT leaderboard-valid: the reason is an off-spec leaderboard setting
    # (agent web enabled / timeout multiplier / resource override), NOT low-k variance — so the
    # warning must NOT claim "k < floor".
    return "local_low_k", (
        f"⚠️ k={k} is >= {low_k_floor} but this run is NOT leaderboard-valid — a leaderboard-faithful "
        "setting is off-spec (agent web enabled / timeout multiplier / resource override). "
        "Do not report it as a leaderboard number."
    )


_ENV_PASSTHROUGH_FLAGS = ("--ae", "--ve", "--agent-env", "--verifier-env")

# A POSIX-shaped environment variable name. Deliberately stricter than "anything before the
# first `=`": a name is the only part of an `--ae`/`--ve` pair this launcher ever writes down,
# so it must be a NAME, not whatever the shell happened to hand us.
_ENV_NAME_RE = re.compile(r"\A[A-Za-z_][A-Za-z0-9_]*\Z")


def env_assignment(token: str) -> str:
    """argparse ``type=`` for ``--agent-env`` / ``--verifier-env``: require ``NAME=VALUE``.

    SECURITY, and the reason it is validated HERE rather than downstream: every consumer of these
    tokens splits on the first `=` and treats the left half as a non-secret NAME it may persist
    (`agent_env_keys` / `verifier_env_keys` in `run_manifest.json`), print (the passthrough
    warning) or emit (`redacted_command`'s `NAME=<redacted>`). `"x".split("=", 1)[0]` is `"x"`,
    so a token with NO `=` — exactly what a fat-fingered `--ve $OPENAI_API_KEY` produces — made
    the WHOLE token, credential included, the "name" and put it in the clear in all three places.
    Phase v6.75.0 exists partly to harden secret scrubbing; the scrubber's own input path was
    writing the secret out. A malformed name could also inject control characters into those
    artifacts or produce an unmatchable scrub marker.

    Refusing at the point of parse means the process dies before `ensure_outside_repo`, before
    admission and before any artifact exists. The token is NEVER echoed — it may BE the secret,
    which is also why this raises `ArgumentTypeError` (argparse quotes only our message) rather
    than `ValueError` (argparse would render `invalid env_assignment value: '<token>'`)."""
    name, sep, _value = str(token).partition("=")
    if not sep or not _ENV_NAME_RE.match(name):
        raise argparse.ArgumentTypeError(
            "expected NAME=VALUE with a POSIX environment variable name "
            "([A-Za-z_][A-Za-z0-9_]*) and an explicit '='. The offending token is not echoed: "
            "a token without '=' is usually an expanded credential."
        )
    return str(token)


def redacted_command(cmd: list[str]) -> list[str]:
    """The harbor command with agent-/verifier-env VALUES removed.

    Everything THIS LAUNCHER writes to the run root or prints (manifest `official_command`,
    `harbor_command.txt`, stdout) goes through here: `--ve OPENAI_API_KEY=sk-…` names a real
    credential, and a published Harbor artifact tree must not carry it. Only the argv handed
    to the subprocess keeps the values.

    Scope limit, stated because it was previously overstated: this covers our own writes only.
    Harbor persists the values itself into the job directory (see `harbor_command()`), so the
    submission copy still needs the value sweep in `scrub_submission_secrets.py`.

    Defence in depth against a malformed pair: `main()` refuses one at parse time
    (`env_assignment`), but this helper is also callable on an argv assembled elsewhere, and its
    old `token.split("=", 1)[0]` published the WHOLE token as a "name" when there was no `=`.
    Anything that is not a well-formed `NAME=…` is now replaced wholesale, name included."""
    out: list[str] = []
    redact_next = False
    for token in cmd:
        if redact_next:
            name, sep, _value = str(token).partition("=")
            out.append(f"{name}=<redacted>" if sep and _ENV_NAME_RE.match(name) else "<redacted>")
            redact_next = False
            continue
        out.append(token)
        redact_next = token in _ENV_PASSTHROUGH_FLAGS
    return out


def _load_job_config(path: pathlib.Path) -> dict:
    """Load a base Harbor JobConfig (JSON, or YAML when PyYAML is available)."""
    text = pathlib.Path(path).read_text(encoding="utf-8")
    if path.suffix.lower() in {".yaml", ".yml"}:
        import yaml  # optional: only a YAML base config needs it

        loaded = yaml.safe_load(text)
    else:
        loaded = json.loads(text)
    if not isinstance(loaded, dict):
        raise ValueError(f"base job config {path} must be a mapping, got {type(loaded).__name__}")
    return loaded


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge ``override`` into ``base`` (override wins; lists are replaced)."""
    merged = dict(base)
    for key, value in override.items():
        current = merged.get(key)
        merged[key] = _deep_merge(current, value) if isinstance(current, dict) and isinstance(value, dict) else value
    return merged


# Harbor's own default for `-e/--env` when the flag is absent. Recorded (not guessed) so a run that
# leaves the flag off still discloses which backend produced its numbers.
HARBOR_DEFAULT_ENV = "docker"


def harbor_version(harbor_bin: str) -> str:
    """Best-effort version of the harbor binary this run will actually invoke.

    Provenance, not control flow: a number produced by harbor 0.18.0 and one produced by 0.20.0 are
    not interchangeable, and this host deliberately keeps several harbor venvs side by side (TB2.1
    stays pinned at 0.18.0 so published rows keep their harness, while a newer dataset may need a
    newer harbor via --harbor-bin). Any failure — missing binary, non-zero exit, timeout — yields
    "" rather than raising: unknown provenance must not abort a run, it must be visibly unknown."""
    try:
        proc = subprocess.run(
            [str(harbor_bin), "--version"], capture_output=True, text=True, timeout=30, check=False,
        )
    except Exception:
        return ""
    if proc.returncode != 0:
        return ""
    return (proc.stdout or proc.stderr or "").strip().splitlines()[0].strip() if (proc.stdout or proc.stderr).strip() else ""


def submission_subtree(dataset: str) -> tuple[str, str]:
    """Derive ``(family, version)`` for the leaderboard submission tree from the DATASET.

    TB2.1's leaderboard repo files submissions under `submissions/terminal-bench/2.1/…`, which
    was hardcoded here: a run on any other dataset (Harbor-Index) would have been filed under
    the TB2.1 tree — a wrong, silently misleading path. The dataset name carries the identity
    (`<org>/<family>-<major>-<minor>`), so derive it and keep the TB2.1 output byte-identical.
    A name with no trailing version segment yields `(<name>, "")`, and the caller drops the
    empty component. The exact upstream layout for a NON-TB2.1 leaderboard must still be read
    from that leaderboard's own SUBMIT.md before a real submission (see METHODOLOGY.md);
    `--submission-subtree` overrides this derivation when it differs.

    A harbor `@ref` pin is stripped first: the version a run is PINNED to (`@v0.1.0`, `@3`,
    `@sha256:…`) is harbor registry addressing, not part of the leaderboard tree, and leaving it in
    would put a literal `@` in a submission directory name."""
    name = str(dataset or "").split("/")[-1].split("@", 1)[0]
    parts = name.rsplit("-", 2)
    if len(parts) == 3 and parts[1].isdigit() and parts[2].isdigit():
        return parts[0], f"{parts[1]}.{parts[2]}"
    return name, ""


def confined_submission_subtree(raw_subtree: str, *, dataset: str) -> list[str]:
    """Validate the leaderboard subtree into CONFINED single path components.

    `submission_root` is checked by `ensure_outside_repo`, but the job directory is DERIVED from
    it — and validating an ancestor does not confine its descendant. `--submission-subtree ../../..`
    (or a family derived from a `--dataset` such as `org/..-2-1`, since `submission_subtree` splits
    the name and does not judge it) walks straight back out of the checked root and writes
    `metadata.yaml` / `agent_job_config.json` wherever it lands — including inside the Ouroboros
    repository that guard exists to keep out of benchmark output. So the components that are about
    to BE created are what get validated here, before anything is created.

    Each component must be a single safe path component (`safe_benchmark_id` — the same closed rule
    the other launchers already apply to untrusted ids, not a second one): `.`, `..`, empty, and any
    embedded separator are refused rather than normalised away. Absolute and drive-qualified forms
    are refused up front instead of being silently reinterpreted as relative. Backslash and `C:` are
    named explicitly and not left to POSIX intuition: this repository runs a three-OS CI matrix, and
    on Windows `\\` IS a separator and `C:` IS a drive qualifier, so a value that is an inert
    directory name here is a live escape there.
    """
    if raw_subtree:
        field, text = "--submission-subtree", str(raw_subtree)
    else:
        field = "submission subtree derived from --dataset"
        text = "/".join(part for part in submission_subtree(dataset) if part)
    if (
        pathlib.PurePosixPath(text).is_absolute()
        or pathlib.PureWindowsPath(text).drive
        or text.startswith("\\")
    ):
        raise ValueError(f"{field} must be a relative path under the submission root, got: {text!r}")
    parts = [part for part in text.split("/") if part]
    if not parts:
        raise ValueError(f"{field} must name at least one path component, got: {text!r}")
    return [safe_benchmark_id(part, field=field) for part in parts]


def _write_agent_job_config(config: HarborCommandConfig) -> pathlib.Path:
    """Write the agents[] section of the Harbor JobConfig for `harbor run -c`.

    The agent MUST be launched through a job config rather than the bare
    `--agent-import-path` flag: the flag path records `agents[0].name = null`
    in the job config, while trials report the runtime class name — and the
    TB2.1 leaderboard static analysis matches `config.agents[].name` against
    the trial-side name, so a null name makes the submission permanently
    invalid ("no matching agent in job config", terminal-bench-2-1#121).
    The name here must stay equal to OuroborosTerminalBenchAgent.name().
    """
    kwargs: dict[str, object] = {
        "ouroboros_model": config.model,
        "ouroboros_light_model": config.light_model,
        "host_settings_path": str(config.settings_path),
        "task_review_mode": "required",
        "review_enforcement": config.review_enforcement,
        "safety_mode": config.safety_mode,
        "install_timeout_sec": 1200,
        "server_start_timeout_sec": 240,
        "disable_agent_web": bool(config.disable_agent_web),
        # The dataset identity the adapter needs to resolve the right per-task cache subtree
        # (`~/.cache/harbor/tasks/packages/<org>/<name>/<digest>`): one org is not a constant.
        "dataset": config.dataset,
    }
    effort = os.environ.get("OUROBOROS_EFFORT_TASK", "").strip().lower()
    if effort:
        # Label the submission key with the effort this run actually uses;
        # the adapter forwards the kwarg back into the container env, so the
        # declared effort always equals the effective one.
        kwargs["reasoning_effort"] = effort
    job_config: dict[str, object] = {
        "agents": [
            {
                "name": "Ouroboros Installed",
                "import_path": AGENT_IMPORT,
                "model_name": f"ouroboros-{config.model.replace('/', '-')}",
                "kwargs": kwargs,
            }
        ]
    }
    if config.base_job_config is not None:
        # A leaderboard/dataset may publish its own JobConfig (env allowlists, verifier
        # settings, resource shape). Deep-merge OURS on top so every upstream key we do not
        # set survives verbatim, while the agents[] block — including agents[0].name, whose
        # absence permanently invalidates a submission — always stays ours.
        job_config = _deep_merge(_load_job_config(config.base_job_config), job_config)
    config.jobs_dir.mkdir(parents=True, exist_ok=True)
    path = config.jobs_dir / "agent_job_config.json"
    path.write_text(json.dumps(job_config, indent=2) + "\n", encoding="utf-8")
    return path


# Model slots this launcher must NOT copy out of the host settings file: the in-container adapter
# (`harbor_installed_agent._container_env`) does not forward them, so a container never sees them.
# Recording them would put a model in the manifest that provably did not assist the run — the same
# class of lie as recording the wrong one, just in the other direction.
_UNFORWARDED_MODEL_SLOT_KEYS = ("OUROBOROS_MODEL_VISION", "OUROBOROS_MODEL_CONSCIOUSNESS")


def resolved_model_slots(config: HarborCommandConfig) -> dict[str, str]:
    """The model slots this run ACTUALLY uses, resolved the way the container resolves them.

    Derived from the assembled ``HarborCommandConfig`` — the very object ``harbor_command()``
    turns into the job config the adapter reads — and never from a settings TEMPLATE or from
    ``--model`` before the ``--all-model`` override has rewritten it. SWE-Pro's manifest once
    named a model that did not run because it snapshotted the template instead of the derived
    settings; a manifest that is merely SILENT about the model (TB's previous state: the model
    existed only in ``argv`` and the disclosure ledger) is the same failure one step quieter,
    because an auditor still cannot reconstruct which model produced the numbers.

    Resolution order mirrors ``harbor_installed_agent._container_env`` exactly: the forwarded
    host env / settings snapshot first, then the job-config kwargs, which are the last word —
    so the recorded main model is the one the adapter pins, not whatever ``OUROBOROS_MODEL``
    happened to sit in the operator's shell. ``ouroboros_model`` also drives HEAVY and both
    fallback keys in the container, so they are recorded as the same model rather than left to
    imply a second one.
    """
    slots = {
        key: value
        for key, value in model_slot_snapshot(config.settings_path).items()
        if key not in _UNFORWARDED_MODEL_SLOT_KEYS
    }
    if config.model:
        slots["OUROBOROS_MODEL"] = config.model
        slots["OUROBOROS_MODEL_HEAVY"] = config.model
        slots["OUROBOROS_MODEL_FALLBACKS"] = config.model
    if config.light_model:
        slots["OUROBOROS_MODEL_LIGHT"] = config.light_model
    return {key: value for key, value in slots.items() if key in MODEL_SLOT_KEYS}


def augment_manifest(manifest: dict, config: HarborCommandConfig) -> None:
    """Mirror the run's resolved models into the manifest, once the job config exists.

    Same shape and same seam as GAIA's ``_augment_manifest`` (``gaia/run_gaia.py``): the
    settings-derived facts are added AFTER admission and persisted by the finalization seam, and
    they land under ``model_slots`` filtered to ``MODEL_SLOT_KEYS``. Keeping the field name and
    filter identical across families is the point — an auditor reading any benchmark's
    ``run_manifest.json`` finds the model in the same place.
    """
    manifest["model_slots"] = resolved_model_slots(config)


def harbor_command(config: HarborCommandConfig) -> list[str]:
    cmd = [
        config.harbor_bin,
        "run",
        "--config",
        str(_write_agent_job_config(config)),
        "--dataset",
        config.dataset,
        "--n-concurrent",
        str(int(config.n_concurrent)),
        "-k",
        str(int(config.k)),
        "--jobs-dir",
        str(config.jobs_dir),
        "--yes",
    ]
    # Optional host pip wheel cache (opt-in via OBO_TB_PIP_CACHE): bind-mount a durable host dir at
    # /opt/ouro-pip-cache in every task container so the per-trial Ouroboros pip install hits cached
    # wheels instead of the network (offline-fast, resilient to mirror drops). Read-write + shared is
    # safe: pip keys wheels by py/platform tag and writes via atomic rename of identical content, so
    # heterogeneous task images and n-concurrent trials populate/reuse one cache without conflict.
    # This is NOT a leaderboard-config field (it's a deploy mount, like --n-concurrent), so it does
    # not affect static_validation. Unset → no --mounts emitted → behavior unchanged.
    pip_cache = os.environ.get("OBO_TB_PIP_CACHE", "").strip()
    if pip_cache:
        cache_dir = ensure_outside_repo(pathlib.Path(pip_cache), repo_root_from_devtools())
        mounts = [{"type": "bind", "source": str(cache_dir), "target": "/opt/ouro-pip-cache"}]
        cmd.extend(["--mounts", json.dumps(mounts)])
    # Execution backend (harbor's own `-e/--env`). Harbor's default is `docker`, i.e. the LOCAL
    # docker daemon, and Frontier-Bench was verified to run there end-to-end on harbor 0.18.0 (the
    # oracle solution of a real FB task scored reward 1.0 through the separate-environment
    # verifier), so a cloud sandbox provider is NOT required despite upstream's CI/leaderboard runs
    # using `--env modal`. Emit the flag ONLY when explicitly chosen so the default path stays
    # byte-identical to every published TB2.1 command; when omitted, the effective backend is
    # harbor's own default, which is what the manifest/ledger record.
    if str(config.harbor_env or "").strip():
        cmd.extend(["--env", str(config.harbor_env).strip()])
    # Setup/build timeout multipliers default to 1.0 (Harbor static_validation rejects non-1.0
    # agent_setup / environment_build multipliers). Emit the flags ONLY when an explicit
    # local-only override is set; a leaderboard-valid run leaves them absent (== Harbor default).
    if float(config.setup_timeout_multiplier) != 1.0:
        cmd.extend(["--agent-setup-timeout-multiplier", str(float(config.setup_timeout_multiplier))])
    if float(config.build_timeout_multiplier) != 1.0:
        cmd.extend(["--environment-build-timeout-multiplier", str(float(config.build_timeout_multiplier))])
    # Agent-/verifier-phase env passthrough (harbor's own `--ae` / `--ve`, present in 0.18 and
    # 0.20). Some datasets require a verifier-side base URL or an agent-side region; these are
    # deploy knobs like --n-concurrent, not leaderboard-config fields, so they do not affect
    # static_validation.
    #
    # HONESTY FIX (v6.79.0): the previous comment here claimed "VALUES ARE NEVER WRITTEN TO THE
    # RUN ROOT". That was FALSE, and false on the path where it matters most (Harbor-Index, where
    # the judge key goes through `--ve` and the submission upload is PUBLIC). What is true:
    #   * THIS launcher's own artifacts carry NAMES only — `official_command` in the manifest,
    #     `harbor_command.txt` and stdout all go through `redacted_command()`, and the manifest
    #     records `agent_env_keys` / `verifier_env_keys`;
    #   * but HARBOR persists its own JobConfig into the job directory, which sits INSIDE the
    #     submission tree: `harbor/job.py` does
    #     `self._job_config_path.write_text(self.config.model_dump_json(indent=4,
    #     exclude_defaults=True))` → `<jobs-dir>/<job_name>/config.json` (one timestamp level
    #     below the `--jobs-dir` we pass), and the same values are re-serialized into the job
    #     `lock.json` and every trial's `config.json` / `lock.json` / `result.json`.
    #     `--ae` lands in `agents[].env`, `--ve` in `verifier.env`.
    #   * harbor filters those values through `templatize_sensitive_env`
    #     (`harbor/utils/env.py`), which is keyed on the variable NAME only. Measured on the
    #     installed 0.18.0: a value equal to `os.environ[NAME]` becomes `${NAME}` (no leak); a
    #     NAME matching `KEY|SECRET|TOKEN|PASSWORD|CREDENTIAL|AUTH` is written as
    #     `value[:4] + "****" + value[-3:]` (a PARTIAL disclosure — 7 characters of a live
    #     credential); any other NAME (e.g. `MY_BEARER`) is written VERBATIM in cleartext.
    # So the submission copy has to be scrubbed by VALUE, not just by name: pass every
    # `--ae`/`--ve` pair to `scrub_submission_secrets.py --env-passthrough NAME=VALUE`, which
    # sweeps the whole tree (harbor's config/lock/result files included), also sweeps harbor's
    # own partial form, and refuses to finish if any needle survives.
    for value in config.agent_env:
        cmd.extend(["--ae", value])
    for value in config.verifier_env:
        cmd.extend(["--ve", value])
    for task in config.task_filters:
        cmd.extend(["--include-task-name", task])
    if config.execute:
        cmd.append("--force-build")
    return cmd


def write_disclosure_ledger(*, jobs_dir: pathlib.Path, out_path: pathlib.Path, run_meta: dict) -> dict:
    """Walk Harbor's jobs dir, read every trial result.json, and write a denominator-preserving
    disclosure ledger: reward distribution, exception_info histogram (AgentTimeoutError /
    ApiRateLimitError counts -- the latency / rate-limit failure signal Anton asked for), per-task
    pass rate, cost and wall-time. Honest-by-construction so a write-up can disclose how many
    trials failed by timeout vs rate-limit at a given concurrency. Best-effort: never raises into
    the harbor result path (the caller guards it)."""
    import collections

    trials: list[dict] = []
    for result_json in sorted(pathlib.Path(jobs_dir).rglob("result.json")):
        try:
            data = json.loads(result_json.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if not isinstance(data, dict):
            continue
        if "task_name" not in data and "verifier_result" not in data:
            continue  # not a trial result.json (e.g. a job-level file)
        reward = None
        verifier_result = data.get("verifier_result")
        if isinstance(verifier_result, dict) and isinstance(verifier_result.get("rewards"), dict):
            reward = verifier_result["rewards"].get("reward")
        exception_info = data.get("exception_info")
        exception_type = exception_info.get("exception_type") if isinstance(exception_info, dict) else None
        agent_result = data.get("agent_result") if isinstance(data.get("agent_result"), dict) else {}
        agent_meta = agent_result.get("metadata") if isinstance(agent_result.get("metadata"), dict) else {}
        # The adapter records the in-container Ouroboros outcome (reason_code/infra_failed) here.
        # Harbor's own exception_info is null for a clean agent exit even when Ouroboros failed on a
        # provider issue (e.g. reason_code="provider_unavailable"), so the truthful provider/infra
        # signal lives in this summary, NOT in Harbor's exception_info.
        adapter_summary = agent_meta.get("summary") if isinstance(agent_meta.get("summary"), dict) else {}
        # The adapter's result.json summary carries reason_code/infra_failed but NOT the
        # cancellation signal. captured_after_cancellation lives in the sibling run-summary,
        # written by the adapter's teardown-snapshot path. It is the load-bearing signal that a
        # terminal reason_code="provider_unavailable" is only a post-cancellation network-teardown
        # artifact (the harness cuts container egress at the finish line, so the agent's own
        # finalization/summary LLM calls die on DNS) rather than a real mid-run provider fault.
        captured_after_cancellation = False
        run_summary_path = result_json.parent / "agent" / "ouroboros-run-summary.json"
        try:
            run_summary = json.loads(run_summary_path.read_text(encoding="utf-8"))
            if isinstance(run_summary, dict):
                captured_after_cancellation = bool(run_summary.get("captured_after_cancellation"))
        except (OSError, ValueError):
            pass
        trials.append({
            "task_name": data.get("task_name"),
            "trial_name": data.get("trial_name"),
            "reward": reward,
            "exception_type": exception_type,
            "reason_code": adapter_summary.get("reason_code"),
            "infra_failed": bool(adapter_summary.get("infra_failed")),
            # The runtime's own resource-rail truncation (USD rail / round cap / loop-local
            # deadline / forced finalization), forwarded by the adapter summary, whose
            # vocabulary is interpolated from result_index.RUNTIME_TRUNCATION_REASON_CODES.
            # Without it a trial the per-task USD rail stopped is counted as a fair-shot
            # wrong answer, which is a claim about capability that never happened.
            "truncated": bool(adapter_summary.get("truncated")),
            "resource_limit": adapter_summary.get("resource_limit") or {},
            "captured_after_cancellation": captured_after_cancellation,
            "cost_usd": agent_result.get("cost_usd"),
            "turns": agent_meta.get("turns"),
            "started_at": data.get("started_at"),
            "finished_at": data.get("finished_at"),
        })

    def _reward_key(reward: object) -> str:
        if reward in (1, 1.0, True):
            return "1.0"
        if reward in (0, 0.0, False):
            return "0.0"
        return "null" if reward is None else str(reward)

    reward_distribution = collections.Counter(_reward_key(t["reward"]) for t in trials)
    exception_histogram = collections.Counter(t["exception_type"] for t in trials if t["exception_type"])
    reason_code_histogram = collections.Counter(t["reason_code"] for t in trials if t["reason_code"])
    # The honest provider/rate-limit/infra signal combines THREE sources: a Harbor exception that is
    # not a timeout (RuntimeError/ApiRateLimitError/NonZeroAgentExitCodeError), the adapter's
    # infra_failed flag, OR an Ouroboros reason_code denoting a provider fault (which Harbor records
    # as a clean reward-0 trial with exception_info=null). Without this a 429-driven failure is
    # indistinguishable from a genuine wrong answer.
    _timeout_types = {"AgentTimeoutError", "VerifierTimeoutError"}
    # Both codes are real and grepped: loop.py:3185 (reroute + fallback exhausted) and
    # loop_llm_call.py:630 (transport death). `rate_limited`/`provider_error` used to sit
    # here too and the runtime has never emitted either — inert, but a hand-written claim
    # about a vocabulary that lives elsewhere, which is exactly the defect this release fixed
    # in RUNTIME_TRUNCATION_REASON_CODES. This set is a strict subset of it by construction.
    _provider_reasons = {"provider_unavailable", "llm_api_error"}

    def _failure_category(t: dict) -> str:
        """Classify each trial into exactly one honest bucket (precedence matters).

        'pass'           reward 1.0 -- verifier-confirmed success, regardless of Ouroboros's own
                         terminal status (Ouroboros may self-report failed/provider_unavailable when
                         a post-work finalization call dies on the harness network teardown, yet the
                         work in the container already passed the external verifier).
        'provider_infra' a provider/infra fault that PREVENTED a fair attempt: any non-timeout Harbor
                         exception (setup timeout, nonzero exit, transport), infra_failed, OR a
                         provider reason_code seen WITHOUT the post-cancellation flag (a genuine
                         mid-run provider/network death).
        'cancelled'      the agent was stopped by the harness before producing a real terminal: a
                         Harbor wall-clock timeout, OR a provider reason_code stamped only during
                         post-cancellation network teardown (captured_after_cancellation) -- the
                         harness cut egress at the finish line, the agent's own finalization LLM call
                         died on DNS, and the real outcome was masked.
        'cost_truncated' the RUNTIME stopped the trial on its own resource rail (per-task USD
                         reservation / round cap / loop-local deadline / forced finalization
                         with an unabsorbed child) rather than on an answer. Not
                         provider_infra (nothing was unavailable) and emphatically not 'genuine':
                         'genuine' asserts a FAIR SHOT, and a trial cut off at $0.45 of actual
                         spend by a worst-case reservation bound did not get one.
        'genuine'        reward 0 having reached a real terminal (final_message / tool_failure / ...)
                         -- a fair-shot wrong answer. NOTE: captured_after_cancellation is NOT used to
                         route here; it is set broadly on teardown, including on trials that DID emit
                         a final answer, so trusting it alone would wrongly excuse real wrong answers.
        'other'          reward is null/other and none of the above.
        """
        if t["reward"] in (1, 1.0, True):
            return "pass"
        et = t.get("exception_type")
        if et in _timeout_types:
            return "cancelled"          # Harbor wall-clock timeout: cut off before finishing.
        if et:
            return "provider_infra"     # setup timeout / nonzero exit / transport: real infra fault.
        if t.get("infra_failed"):
            return "provider_infra"
        if t.get("truncated") and t.get("reason_code") not in _provider_reasons:
            # Runtime resource rail. Ordered AFTER the infra checks (an infra fault that also
            # tripped a rail is still an infra fault) and BEFORE 'genuine'.
            return "cost_truncated"
        if t.get("reason_code") in _provider_reasons:
            # Provider reason WITH the teardown flag == masked harness cancellation, not a real
            # provider fault; WITHOUT it == a genuine mid-run provider/network death.
            return "cancelled" if t.get("captured_after_cancellation") else "provider_infra"
        if t["reward"] in (0, 0.0, False):
            return "genuine"            # reached a real terminal with reward 0 -> wrong answer.
        return "other"

    categories = collections.Counter(_failure_category(t) for t in trials)
    provider_or_infra_failures = categories["provider_infra"]
    per_task: dict[str, dict] = collections.defaultdict(lambda: {"n": 0, "passed": 0})
    for trial in trials:
        bucket = per_task[trial["task_name"] or "?"]
        bucket["n"] += 1
        if trial["reward"] in (1, 1.0):
            bucket["passed"] += 1
    per_task_pass_rate = {
        task: (b["passed"] / b["n"] if b["n"] else 0.0) for task, b in sorted(per_task.items())
    }
    costs = [t["cost_usd"] for t in trials if isinstance(t["cost_usd"], (int, float))]
    ledger = {
        "schema": "tb_disclosure_ledger.v1",
        "run": dict(run_meta),
        "n_trials": len(trials),
        "n_tasks": len(per_task),
        "reward_distribution": dict(reward_distribution),
        "exception_histogram": dict(exception_histogram),
        "reason_code_histogram": dict(reason_code_histogram),
        "agent_timeout_count": int(exception_histogram.get("AgentTimeoutError", 0)),
        "api_rate_limit_error_count": int(exception_histogram.get("ApiRateLimitError", 0)),
        "provider_or_infra_failure_count": int(provider_or_infra_failures),
        "wall_clock_cancellation_count": int(categories["cancelled"]),
        "cost_truncated_count": int(categories["cost_truncated"]),
        "genuine_failure_count": int(categories["genuine"]),
        "exception_note": (
            "Honest taxonomy: every reward-0 trial is exactly one of provider_or_infra_failure / "
            "wall_clock_cancellation / cost_truncated / genuine_failure (reward-1 trials are 'pass'). "
            "cost_truncated is the RUNTIME's own resource rail (per-task USD reservation, round cap, "
            "context cap): nothing was unavailable and no answer was reached, so it is neither infra "
            "nor a fair-shot wrong answer. agent_timeout_count "
            "is the Harbor-named subset. A provider reason_code (e.g. provider_unavailable) only counts as "
            "provider_or_infra when it was NOT a post-cancellation teardown artifact: the harness cuts "
            "container egress at the finish line, so an agent's own finalization/summary LLM calls die on "
            "DNS and get stamped provider_unavailable even though the task already passed or was merely cut "
            "off by wall-clock -- those go to wall_clock_cancellation (captured_after_cancellation), NOT "
            "provider_or_infra. genuine_failure_count is reward-0 given a fair shot (real wrong answers, "
            "not provider artifacts and not wall-clock cut-offs)."
        ),
        "total_cost_usd": round(sum(costs), 4) if costs else None,
        "per_task_pass_rate": per_task_pass_rate,
        "trials": trials,
    }
    write_json(out_path, ledger)
    print(
        f"[run_tb] disclosure ledger: {len(trials)} trials, "
        f"{ledger['agent_timeout_count']} AgentTimeoutError, "
        f"{ledger['provider_or_infra_failure_count']} provider/infra, "
        f"{ledger['wall_clock_cancellation_count']} wall-clock-cancelled, "
        f"{ledger['cost_truncated_count']} cost-truncated, "
        f"{ledger['genuine_failure_count']} genuine failures -> {out_path}"
    )
    return ledger


def classify_harbor_outcome(ledger: dict | None, returncode: int) -> tuple[str, int]:
    """The typed outcome AND real exit status for a harbor job — from the job's own trials.

    Same class as GAIA's swallowed inspect failure: `harbor run` has no non-zero exit path for a
    job whose trials all ERRORED, so its return code cannot tell a dead job from a scored one
    (2026-07-04: a job wrote 444 trial `result.json` files and zero rewards, and looked healthy).
    Three facts kept apart, only the last being a result:

    ``no_scored_trials``   no trial reached the verifier — an INFRA zero (also covers zero trials).
    ``trials_unverified``  the ledger could not be built, so there is no evidence of a result;
                           fail closed, because unknown success is not success.
    ``completed``          at least one trial carries a numeric reward. An all-zero reward
                           distribution over scored trials is a GENUINE zero and stays `completed`.
    """
    code = int(returncode)
    if code:
        return "harness_nonzero_exit", code
    if not isinstance(ledger, dict):
        return "trials_unverified", 1
    n_trials = int(ledger.get("n_trials") or 0)
    distribution = ledger.get("reward_distribution")
    # `null` is the disclosure ledger's bucket for a trial with no numeric reward, i.e. one the
    # verifier never scored (errored, cancelled before verification, unparseable result).
    unscored = int((distribution or {}).get("null") or 0) if isinstance(distribution, dict) else n_trials
    if n_trials - unscored <= 0:
        return "no_scored_trials", 1
    return "completed", 0


def _build_arg_parser() -> argparse.ArgumentParser:
    """Every CLI flag this launcher accepts, in one place.

    Extracted from ``main`` purely to keep it under the per-function limit in
    docs/DEVELOPMENT.md. It BUILDS the parser and does nothing else — no parsing, no
    filesystem, no network, nothing that can fail — so it cannot move a failure ahead of the
    admission gate (ADMISSION IS THE OUTER BOUNDARY). ``main`` keeps ``parse_args`` and every
    ``parser.error`` path, because those depend on the PARSED values.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--model", default="", help="measured/declared model; or use --all-model to set every slot")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--low-k-floor", type=int, default=5, help="k below this is graded local/low-confidence (debug_only at k=1) in report_grade (default 5).")
    parser.add_argument("--n-concurrent", type=int, default=1)
    parser.add_argument("--task", action="append", default=[], help="optional include-task-name; repeatable")
    parser.add_argument("--review-enforcement", default="blocking", choices=["blocking", "advisory"],
                        help="in-task review enforcement mode forwarded to the container (default blocking)")
    parser.add_argument("--safety-mode", default="light", choices=["full", "light", "off"],
                        help="LLM safety mode inside the task container (default light; off disables the LLM safety pass, deterministic guards stay)")
    parser.add_argument("--run-root", default="")
    parser.add_argument("--submission-root", default="")
    parser.add_argument("--settings-path", default="")
    parser.add_argument("--harbor-bin", default="harbor")
    parser.add_argument("--light-model", default="google/gemini-3.5-flash")
    parser.add_argument("--timeout-multiplier", type=float, default=1.0)
    parser.add_argument("--resource-override", action="append", default=[])
    parser.add_argument("--agent-name", default="Ouroboros")
    parser.add_argument("--org-name", default="Ouroboros")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--setup-timeout-multiplier", type=float, default=1.0)
    parser.add_argument("--build-timeout-multiplier", type=float, default=1.0)
    parser.add_argument(
        "--allow-setup-build-multipliers",
        action="store_true",
        help="LOCAL-only: permit non-1.0 setup/build multipliers (run is NOT leaderboard-valid)",
    )
    parser.add_argument(
        "--allow-low-k",
        action="store_true",
        help="LOCAL-only: permit k<5 (e.g. a k=1 first pass); run is NOT leaderboard-valid",
    )
    parser.add_argument(
        "--disable-agent-web",
        dest="disable_agent_web",
        action="store_true",
        default=True,
        help="block the in-container agent's web/browser tools (default; reward-hacking guard)",
    )
    parser.add_argument(
        "--allow-agent-web",
        dest="disable_agent_web",
        action="store_false",
        help="LOCAL-only: allow the agent's web tools (NOT leaderboard-faithful)",
    )
    parser.add_argument(
        "--all-model",
        default="",
        help="convenience: set ALL model slots (incl. review/scope/websearch) to this model",
    )
    parser.add_argument(
        "--review-slots",
        type=int,
        default=1,
        help="number of in-task acceptance reviewers for --all-model (default 1; 3 identical = monoculture, no diversity)",
    )
    parser.add_argument(
        "--review-effort",
        default="low",
        choices=["none", "low", "medium", "high"],
        help="reasoning effort for the in-task review under --all-model (default low; cuts the review-latency tax)",
    )
    parser.add_argument(
        "--base-job-config",
        default="",
        help="upstream Harbor JobConfig (json/yaml) to deep-merge UNDER our agents[] block; "
             "every key we do not set is preserved verbatim",
    )
    parser.add_argument(
        "--agent-env",
        action="append",
        default=[],
        type=env_assignment,
        help="KEY=VALUE forwarded to harbor --ae (agent phase); repeatable. THE VALUE IS "
             "PERSISTED: this launcher redacts only the command artifacts it owns, while harbor "
             "writes the pair into its own job config/lock/result files. Before publishing a "
             "submission copy, sweep it with scrub_submission_secrets.py --env-passthrough "
             "KEY=VALUE (see terminal_bench/METHODOLOGY.md)",
    )
    parser.add_argument(
        "--verifier-env",
        action="append",
        default=[],
        type=env_assignment,
        help="KEY=VALUE forwarded to harbor --ve (verifier phase); repeatable. THE VALUE IS "
             "PERSISTED: this launcher redacts only the command artifacts it owns, while harbor "
             "writes the pair into its own job config/lock/result files. Before publishing a "
             "submission copy, sweep it with scrub_submission_secrets.py --env-passthrough "
             "KEY=VALUE (see terminal_bench/METHODOLOGY.md)",
    )
    parser.add_argument(
        "--harbor-env",
        default="",
        help="harbor execution backend passed to its own -e/--env (e.g. docker, modal, daytona). "
             f"Default: unset, i.e. harbor's own default ({HARBOR_DEFAULT_ENV}, the local daemon)",
    )
    parser.add_argument(
        "--submission-subtree",
        default="",
        help="override the derived submissions/<family>/<version> path (default: derived from "
             "--dataset). Must be a relative path of plain components: absolute, drive-qualified "
             "and '..' forms are refused, not normalised",
    )
    parser.add_argument(
        "--allow-dirty-seed",
        action="store_true",
        help="record and proceed with an unclean/unidentifiable seed checkout instead of refusing",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    if args.all_model:
        apply_all_model(args.all_model, review_slots=args.review_slots, review_effort=args.review_effort)
        args.model = args.all_model
        args.light_model = args.all_model
    if not args.model:
        parser.error("either --model or --all-model is required")

    validate_methodology(
        k=args.k,
        timeout_multiplier=args.timeout_multiplier,
        resource_overrides=list(args.resource_override or []),
        setup_timeout_multiplier=args.setup_timeout_multiplier,
        build_timeout_multiplier=args.build_timeout_multiplier,
        allow_setup_build_multipliers=args.allow_setup_build_multipliers,
        allow_low_k=args.allow_low_k,
    )
    if not args.disable_agent_web:
        print(
            "WARNING: --allow-agent-web leaves the agent's web tools ENABLED; this run is NOT "
            "leaderboard-faithful (reward-hacking guard off).",
            file=sys.stderr,
        )
    # Resolved BEFORE the first mkdir below: an escaping subtree must refuse while nothing exists
    # yet, not after `ensure_outside_repo` has already created the run and submission roots.
    subtree = confined_submission_subtree(args.submission_subtree, dataset=args.dataset)

    repo = repo_root_from_devtools()
    run_root = assert_outside_repo(
        pathlib.Path(args.run_root).expanduser() if args.run_root else pathlib.Path.cwd() / "tb21_ouroboros_run",
        repo,
    )
    settings_path = pathlib.Path(args.settings_path).expanduser() if args.settings_path else default_settings_path()
    submission_root = assert_outside_repo(
        pathlib.Path(args.submission_root).expanduser()
        if args.submission_root
        else run_root / "submission",
        repo,
    )
    # Second belt on the SAME object: the components are already validated, and the assembled path
    # is re-confined under the validated root. `--model` is only slash-flattened, not id-checked
    # (real slots carry `.` and provider suffixes like `:free`, which a strict id rule would reject),
    # so this join is what keeps a hostile model string inside the tree on every OS.
    job_dir = safe_join_under(
        submission_root, "submissions", *subtree, f"ouroboros__{args.model.replace('/', '-')}", "job",
    )
    metadata_path = job_dir.parent / "metadata.yaml"
    # Backend + harness provenance, resolved ONCE and shared by the manifest and the ledger so the
    # two can never disagree. `harbor_env_effective` names the backend that actually ran even when
    # the flag was left off (harbor's default), because "which sandbox produced this number" is a
    # disclosure obligation, not an inference for a later reader to make.
    harbor_env_effective = str(args.harbor_env or "").strip() or HARBOR_DEFAULT_ENV
    leaderboard_valid = bool(
        int(args.k) >= 5
        and float(args.timeout_multiplier) == 1.0
        and float(args.setup_timeout_multiplier) == 1.0
        and float(args.build_timeout_multiplier) == 1.0
        and bool(args.disable_agent_web)
        and not list(args.resource_override or [])
    )
    # D: 3-tier honesty grade + warning, printed on ALL paths (incl. dry-run / command-gen,
    # which returns before --execute) so a low-k run is never silently cited as a measurement.
    report_grade_value, report_grade_warning = report_grade(
        k=int(args.k), leaderboard_valid=leaderboard_valid, low_k_floor=int(args.low_k_floor)
    )
    if report_grade_warning:
        print(report_grade_warning, file=sys.stderr)
    manifest_path = run_root / "run_manifest.json"
    # Admission is the outermost REFUSAL point, not the outermost side effect: `ensure_outside_repo`
    # above already created `run_root` and `submission_root` (both mkdir). What still happens INSIDE
    # the block below is everything that costs or publishes — the submission skeleton's CONTENTS,
    # harbor's job config and the `harbor --version` probe — so a refused run leaves two empty
    # directories and the persisted refusal, never a half-built submission tree.
    manifest = admit_benchmark_run(
        manifest_path,
        benchmark="terminal_bench",
        run_root=run_root,
        repo_dir=repo,
        requested_task_ids=list(args.task or []),
        require_clean=not args.allow_dirty_seed,
        # dataset + official_command are promoted to top-level by benchmark_run_manifest;
        # everything else must go through `extra` (the helper drops unknown top-level keys).
        dataset=args.dataset,
        extra={
            "outcome": "started",
            "k": int(args.k),
            "n_concurrent": int(args.n_concurrent),
            "timeout_multiplier": float(args.timeout_multiplier),
            "setup_timeout_multiplier": float(args.setup_timeout_multiplier),
            "build_timeout_multiplier": float(args.build_timeout_multiplier),
            "resource_overrides": list(args.resource_override or []),
            "disable_agent_web": bool(args.disable_agent_web),
            "all_model": args.all_model or "",
            # The measured models, recorded at ADMISSION so even a refused or crashed run names
            # what it was about to measure. These are post-`--all-model` values (the override
            # above rewrites both), i.e. what actually reaches the harbor job config — the same
            # numbers `model_slots` carries once `augment_manifest` runs.
            "model": args.model,
            "light_model": args.light_model,
            "leaderboard_valid": leaderboard_valid,
            "report_grade": report_grade_value,
            "report_grade_warning": report_grade_warning,
            "leaderboard_submission_root": str(submission_root),
            "submission_subtree": "/".join(subtree),
            "metadata_yaml": str(metadata_path),
            "base_job_config": str(args.base_job_config or ""),
            # Harness/backend provenance. `harbor_env` is what was REQUESTED (empty = flag
            # omitted); `harbor_env_effective` is what ran. `harbor_version` is resolved
            # after admission (it shells out) and is "" when the binary could not be
            # interrogated -- visibly unknown, never assumed.
            "harbor_bin": str(args.harbor_bin),
            "harbor_env": str(args.harbor_env or ""),
            "harbor_env_effective": harbor_env_effective,
            # NAMES only -- no value ever enters an artifact THIS launcher writes. Splitting on
            # the first `=` is safe ONLY because `type=env_assignment` already refused any token
            # without one: without that guard a `--ve <expanded-secret>` typo made the whole
            # credential the "name" and persisted it here in the clear.
            "agent_env_keys": [str(item).split("=", 1)[0] for item in (args.agent_env or [])],
            "verifier_env_keys": [str(item).split("=", 1)[0] for item in (args.verifier_env or [])],
            # Typed audit fact, not a string to grep: harbor writes these values into
            # its own job/trial config+lock+result files under the job dir (partially
            # redacted at best, verbatim when the NAME looks non-sensitive), so a
            # submission built from this run MUST be value-scrubbed. See harbor_command().
            "env_passthrough_persisted_by_harbor": bool(
                list(args.agent_env or []) or list(args.verifier_env or [])
            ),
        },
    )
    with finalize_run_manifest(manifest_path, manifest) as final:
        job_dir.mkdir(parents=True, exist_ok=True)
        metadata_path.write_text(
            leaderboard_metadata(agent_name=args.agent_name, org_name=args.org_name, model=args.model, light_model=args.light_model, disable_agent_web=bool(args.disable_agent_web)),
            encoding="utf-8",
        )
        harbor_config = HarborCommandConfig(
            dataset=args.dataset,
            model=args.model,
            k=args.k,
            jobs_dir=job_dir,
            harbor_bin=args.harbor_bin,
            n_concurrent=args.n_concurrent,
            task_filters=list(args.task or []),
            settings_path=settings_path,
            execute=bool(args.execute),
            light_model=args.light_model,
            review_enforcement=args.review_enforcement,
            safety_mode=args.safety_mode,
            setup_timeout_multiplier=args.setup_timeout_multiplier,
            build_timeout_multiplier=args.build_timeout_multiplier,
            disable_agent_web=bool(args.disable_agent_web),
            base_job_config=pathlib.Path(args.base_job_config).expanduser() if args.base_job_config else None,
            agent_env=tuple(args.agent_env or []),
            verifier_env=tuple(args.verifier_env or []),
            harbor_env=args.harbor_env,
        )
        # Recorded from the ASSEMBLED config, not from `args`: the job config below is what the
        # adapter reads, so binding the manifest to the same object is what keeps the recorded
        # model and the executed model from ever drifting apart.
        augment_manifest(manifest, harbor_config)
        cmd = harbor_command(harbor_config)
        harbor_version_value = harbor_version(args.harbor_bin)
        manifest["official_command"] = redacted_command(cmd)
        manifest["extra"]["harbor_version"] = harbor_version_value
        safe_cmd = shlex.join(redacted_command(cmd))
        (run_root / "harbor_command.txt").write_text(safe_cmd + "\n", encoding="utf-8")
        print(safe_cmd)
        passthrough_keys = [str(item).split("=", 1)[0]
                            for item in [*(args.agent_env or []), *(args.verifier_env or [])]]
        if passthrough_keys:
            # Names only. Harbor writes these VALUES into its own job/trial config+lock+result
            # files inside the submission tree, so the scrub is not optional on the upload path.
            print(f"[run_tb] WARNING: {len(passthrough_keys)} --ae/--ve value(s) "
                  f"({','.join(sorted(set(passthrough_keys)))}) will be persisted by harbor into the "
                  "job dir. Before ANY upload, scrub a COPY with: scrub_submission_secrets.py --root "
                  "<job_copy> --secrets-from … --env-passthrough NAME=VALUE (one per value above).",
                  file=sys.stderr)
        if not args.execute:
            final.update({"outcome": "command_generated", "exit_code": 0})
            return 0
        completed = subprocess.run(cmd, cwd=repo, env={**os.environ, "PYTHONPATH": str(repo)})
        ledger: dict | None = None
        try:
            ledger = write_disclosure_ledger(
                jobs_dir=job_dir,
                out_path=run_root / "disclosure_ledger.json",
                run_meta={
                    "dataset": args.dataset,
                    "k": int(args.k),
                    "n_concurrent": int(args.n_concurrent),
                    "disable_agent_web": bool(args.disable_agent_web),
                    "setup_timeout_multiplier": float(args.setup_timeout_multiplier),
                    "build_timeout_multiplier": float(args.build_timeout_multiplier),
                    "leaderboard_valid": leaderboard_valid,
                    "report_grade": report_grade_value,
                    "report_grade_warning": report_grade_warning,
                    "model": args.model,
                    "model_provider_prefix": (args.model.split("/", 1)[0] if "/" in args.model else ""),
                    "all_model": args.all_model or "",
                    "harbor_bin": str(args.harbor_bin),
                    "harbor_version": harbor_version_value,
                    "harbor_env_effective": harbor_env_effective,
                    "harbor_returncode": int(completed.returncode),
                },
            )
        except Exception as exc:  # disclosure is best-effort; never mask the harbor result
            print(f"[run_tb] disclosure ledger skipped: {exc!r}", file=sys.stderr)
        code = int(completed.returncode)
        # THE STATUS COMES FROM THE TRIALS, NOT FROM HARBOR'S EXIT CODE (see
        # `classify_harbor_outcome`). The raw harness code is kept next to the verdict so an
        # audit never confuses "harbor exited 0" with "the job produced a result".
        outcome, exit_code = classify_harbor_outcome(ledger, code)
        trials = {
            "n_trials": int((ledger or {}).get("n_trials") or 0),
            "reward_distribution": dict((ledger or {}).get("reward_distribution") or {}),
        } if isinstance(ledger, dict) else {"n_trials": 0, "reward_distribution": {}}
        final.update({"outcome": outcome, "exit_code": exit_code,
                      "harness_exit_code": code, "harbor_trials": trials})
        if outcome != "completed":
            print(f"[run_tb] harbor job did not produce a result: outcome={outcome} "
                  f"harness_exit_code={code} trials={json.dumps(trials)}", file=sys.stderr)
        return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
