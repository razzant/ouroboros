"""Google Colab bootstrap helpers for source-mode Ouroboros."""

from __future__ import annotations

import getpass
import json
import os
import pathlib
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Callable, Dict, Optional

from ouroboros.config import SETTINGS_DEFAULTS, normalize_settings_raw, serialize_settings
from ouroboros.update_channels import get_managed_update_fetch_timeout_sec, normalize_update_channel
from ouroboros.utils import atomic_write_json, write_text_atomic

DEFAULT_COLAB_APP_ROOT = "/content/drive/MyDrive/Ouroboros"
DEFAULT_COLAB_REPO_DIR = "/content/ouroboros_repo"
DEFAULT_OFFICIAL_REPO_URL = "https://github.com/razzant/ouroboros.git"

_SECRET_KEYS = ("OPENROUTER_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "MINIMAX_API_KEY", "DEEPSEEK_API_KEY", "CLOUDRU_FOUNDATION_MODELS_API_KEY", "GITHUB_TOKEN", "TELEGRAM_BOT_TOKEN")


def _run_colab_git_network(args: list[str], *, cwd: pathlib.Path | None = None) -> str:
    """Run clone/fetch non-interactively with the managed-update wall-clock ceiling."""
    from ouroboros.platform_layer import kill_process_tree, subprocess_new_group_kwargs

    env = {**os.environ, "GIT_TERMINAL_PROMPT": "0", "LC_ALL": "C", "LANG": "C"}
    proc = subprocess.Popen(
        ["git", "-c", "http.lowSpeedLimit=1024", "-c", "http.lowSpeedTime=30", *args],
        cwd=str(cwd) if cwd else None,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
        **subprocess_new_group_kwargs(),
    )
    try:
        stdout, stderr = proc.communicate(timeout=get_managed_update_fetch_timeout_sec())
    except subprocess.TimeoutExpired as exc:
        kill_process_tree(proc)
        try:
            proc.communicate(timeout=10)
        except Exception:
            pass
        raise RuntimeError("official git operation timed out") from exc
    if proc.returncode != 0:
        raise RuntimeError((stderr or stdout or "official git operation failed").strip())
    return (stdout or "").strip()


def get_colab_secret(name: str, *, required: bool = True) -> str:
    """Return a Colab secret/env value, or ask via a hidden prompt.

    Optional secrets (``required=False``) never block on a prompt: if they are
    absent from Colab userdata and the environment, an empty string is returned.
    """
    value = ""
    try:
        from google.colab import userdata  # type: ignore
        value = str(userdata.get(name) or "").strip()
    except Exception:
        value = ""
    if not value:
        value = str(os.environ.get(name, "") or "").strip()
    if not value and required:
        value = getpass.getpass(f"{name}: ").strip()
    return value


def collect_colab_secrets() -> Dict[str, str]:
    """Collect runtime secrets without printing their values.

    The Telegram bot token is required (the bridge needs it). Any supported
    provider key works — OpenRouter, OpenAI, Anthropic, MiniMax, DeepSeek and
    Cloud.ru are collected optionally, and OpenRouter is prompted only if none
    is found, so a single-direct-provider Colab user is not forced to enter
    OpenRouter.
    The GitHub token is optional (it only enables personal self-modification
    persistence), so a quick prototype never blocks on a GitHub prompt.
    """
    # Providers the one-click Colab launch can auto-route models for via
    # apply_runtime_provider_defaults (OpenRouter is the default aggregator;
    # OpenAI/Anthropic/MiniMax/DeepSeek/Cloud.ru have direct model defaults). OpenAI-compatible
    # endpoints have no universal model default and need explicit OUROBOROS_MODEL_*
    # config, so they are an advanced manual path, not part of the quick launch.
    provider_keys = (
        "OPENROUTER_API_KEY",
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "MINIMAX_API_KEY",
        "DEEPSEEK_API_KEY",
        "CLOUDRU_FOUNDATION_MODELS_API_KEY",
    )
    out: Dict[str, str] = {}
    out["TELEGRAM_BOT_TOKEN"] = get_colab_secret("TELEGRAM_BOT_TOKEN")
    for key in provider_keys:
        out[key] = get_colab_secret(key, required=False)
    if not any(out.get(key) for key in provider_keys):
        # Ensure at least one runnable provider; OpenRouter is the default route.
        out["OPENROUTER_API_KEY"] = get_colab_secret("OPENROUTER_API_KEY")
    out["GITHUB_TOKEN"] = get_colab_secret("GITHUB_TOKEN", required=False)
    return out


def masked_secret_status(settings: Dict[str, Any]) -> Dict[str, bool]:
    """Expose configured/missing status only; never return secret values."""
    return {key: bool(str(settings.get(key, "") or "").strip()) for key in _SECRET_KEYS}


def build_colab_settings(
    secrets: Dict[str, str],
    *,
    github_repo: str = "",
    total_budget: float = float(SETTINGS_DEFAULTS["TOTAL_BUDGET"]),
    runtime_mode: str = "advanced",
    max_workers: int = 1,
    models: Dict[str, str] | None = None,
    network_password: str = "",
    existing: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build a Drive-persisted settings payload for Colab.

    On a re-run, ``existing`` (the current Drive ``settings.json``) is merged over
    the defaults FIRST, so prior owner choices that the Colab launch does not set
    explicitly — a pinned ``TELEGRAM_CHAT_ID``, tweaked model slots, an auto-grant
    preference — survive instead of being reset to defaults. The launch knobs
    below (secrets, budget, runtime mode, workers, host, repo) then win.
    """
    settings = dict(SETTINGS_DEFAULTS)
    if existing:
        # The Drive document is an install's settings document, so it is read the way
        # every reader reads one: the raw-stage normalization (coercion, retention fold,
        # review-cycle seed, retired purge, slot rename, secret repair) runs BEFORE the
        # defaults are merged — after the merge the renamed key is already present as its
        # default and the customization under the former key is lost, then written back
        # to Drive as the owner's choice. Private sentinel keys stay out of the document.
        settings.update({
            k: v for k, v in normalize_settings_raw(existing).items() if not str(k).startswith("_")
        })
    for key, value in secrets.items():
        # Only overwrite when the freshly collected secret is non-empty, so a
        # re-run that omits an optional provider/GitHub key (collect_colab_secrets
        # returns "") does NOT wipe a credential already persisted on Drive.
        if (key in settings or key == "TELEGRAM_BOT_TOKEN") and str(value or "").strip():
            settings[key] = str(value)
    if github_repo:
        settings["GITHUB_REPO"] = github_repo
    settings["TOTAL_BUDGET"] = float(total_budget)
    settings["OUROBOROS_RUNTIME_MODE"] = runtime_mode
    settings["OUROBOROS_MAX_WORKERS"] = int(max_workers)
    settings["OUROBOROS_SERVER_HOST"] = "127.0.0.1"
    settings["OUROBOROS_UPDATE_CHANNEL"] = normalize_update_channel(
        settings.get("OUROBOROS_UPDATE_CHANNEL")
    )
    if network_password:
        settings["OUROBOROS_NETWORK_PASSWORD"] = network_password
    for key, value in (models or {}).items():
        if key in {"OUROBOROS_MODEL", "OUROBOROS_MODEL_HEAVY", "OUROBOROS_MODEL_LIGHT", "OUROBOROS_MODEL_VISION", "OUROBOROS_MODEL_CONSCIOUSNESS", "OUROBOROS_MODEL_FALLBACKS"} and value:
            settings[key] = str(value)
    # Route model slots to the configured provider (same SSOT the desktop
    # onboarding wizard uses): an OpenAI-only / Anthropic-only / Cloud.ru-only
    # Colab config gets correct provider model defaults instead of OpenRouter-style
    # ones. Explicitly supplied model overrides above are preserved.
    from ouroboros.server_runtime import apply_runtime_provider_defaults
    settings, _changed, _changed_keys = apply_runtime_provider_defaults(settings)
    return settings


def write_colab_settings(data_dir: pathlib.Path, settings: Dict[str, Any]) -> pathlib.Path:
    """Persist the generated document for the Drive data root, in the one on-disk spelling.

    Deliberately NOT routed through the persistence prologue: it proves the owner-only
    ratchets against THIS process's ``config.SETTINGS_PATH``, and the Drive root is another
    path (the quickstart exports the Colab paths only after this write). The bytes are
    still ``serialize_settings`` bytes, so the next reader of the Drive file meets the
    spelling every other writer produces."""
    path = pathlib.Path(data_dir) / "settings.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    write_text_atomic(path, serialize_settings(dict(settings)))
    return path


def export_colab_env(repo_dir: pathlib.Path, data_dir: pathlib.Path, settings_path: pathlib.Path) -> Dict[str, str]:
    # A child forked from the long-lived, multi-threaded supervisor can deadlock on
    # the CPython import lock the first time it lazily imports the OpenAI client
    # (llm._make_no_proxy_client). forkserver -- the runtime's Linux default since
    # G13 (supervisor/workers.py) -- forks from one single-threaded server instead;
    # export it explicitly here and respect an explicit override.
    start_method = (os.environ.get("OUROBOROS_WORKER_START_METHOD") or "forkserver").strip().lower()
    env = {"OUROBOROS_APP_ROOT": str(pathlib.Path(data_dir).parent), "OUROBOROS_REPO_DIR": str(repo_dir), "OUROBOROS_DATA_DIR": str(data_dir), "OUROBOROS_SETTINGS_PATH": str(settings_path), "OUROBOROS_WORKER_START_METHOD": start_method, "PYTHONUNBUFFERED": "1", "PYTHONPATH": str(repo_dir)}
    os.environ.update(env)
    return env


def _ff_to_origin_if_ahead(repo_dir: pathlib.Path, branch: str) -> bool:
    """Fast-forward the local branch to origin/<branch> if origin is ahead.

    Restores self-modification commits pushed to the personal origin in an
    earlier (ephemeral) Colab session. Fast-forward only, so it can never lose
    or rewrite local state — a diverged/behind origin leaves the checkout at the
    official HEAD that ``clone_or_update_repo`` already established.
    """
    cwd = str(repo_dir)
    try:
        _run_colab_git_network(["fetch", "origin", branch], cwd=repo_dir)
        rc = subprocess.run(["git", "merge", "--ff-only", f"origin/{branch}"], cwd=cwd, capture_output=True).returncode
        return rc == 0
    except Exception:
        return False


def configure_colab_personal_origin(repo_dir: pathlib.Path, data_dir: pathlib.Path, settings: Dict[str, Any], *, branch: str = "ouroboros") -> Dict[str, Any]:
    """Configure the personal origin remote, creating a fork when needed, and
    best-effort restore prior self-modification commits pushed to that origin."""
    token = str(settings.get("GITHUB_TOKEN") or "").strip()
    if not token:
        return {"ok": False, "error": "GITHUB_TOKEN is not configured"}
    from supervisor import git_ops
    git_ops.init(pathlib.Path(repo_dir), pathlib.Path(data_dir), remote_url="")
    ok, message, resolved = git_ops.configure_personal_remote(str(settings.get("GITHUB_REPO") or ""), token, auto_fork=True, confirm_replace_origin=False)
    restored = False
    if ok and resolved:
        settings["GITHUB_REPO"] = resolved
        restored = _ff_to_origin_if_ahead(repo_dir, branch)
    return {"ok": ok, "message": message, "repo": resolved, "restored_from_origin": restored}


def clone_or_update_repo(
    repo_dir: pathlib.Path,
    source_url: str = DEFAULT_OFFICIAL_REPO_URL,
    source_branch: str = "main",
    local_branch: str = "ouroboros",
) -> pathlib.Path:
    """Fetch one official channel while keeping the local work branch stable."""
    repo_dir = pathlib.Path(repo_dir)
    fresh_clone = False
    if not (repo_dir / ".git").exists():
        if repo_dir.exists():
            raise RuntimeError(f"{repo_dir} exists but is not a git repository")
        _run_colab_git_network(["clone", "--no-checkout", source_url, str(repo_dir)])
        fresh_clone = True
    remotes = subprocess.run(["git", "remote"], cwd=str(repo_dir), capture_output=True, text=True, check=False).stdout.split()
    if "managed" in remotes:
        subprocess.run(["git", "remote", "set-url", "managed", source_url], cwd=str(repo_dir), check=False)
    else:
        from ouroboros.repo_remotes import normalize_repo_slug
        origin_url = ""
        if "origin" in remotes:
            origin_url = subprocess.run(["git", "remote", "get-url", "origin"], cwd=str(repo_dir), capture_output=True, text=True, check=False).stdout.strip()
        # Only reclassify a clone-default origin still pointing at the official
        # upstream as `managed`; a personal `origin` is the persistence target and
        # must be preserved (role-based remotes: managed=official, origin=personal).
        if "origin" in remotes and normalize_repo_slug(origin_url) == normalize_repo_slug(source_url):
            subprocess.run(["git", "remote", "rename", "origin", "managed"], cwd=str(repo_dir), check=False)
            subprocess.run(["git", "remote", "set-url", "managed", source_url], cwd=str(repo_dir), check=False)
        else:
            subprocess.run(["git", "remote", "add", "managed", source_url], cwd=str(repo_dir), check=False)
    subprocess.run(["git", "config", "user.name", "Ouroboros"], cwd=str(repo_dir), check=True)
    subprocess.run(["git", "config", "user.email", "ouroboros@local.mac"], cwd=str(repo_dir), check=True)
    try:
        from supervisor.update_source import MANAGED_TAG_NAMESPACE

        _run_colab_git_network(
            [
                "fetch",
                "managed",
                "+refs/heads/*:refs/remotes/managed/*",
                f"+refs/tags/*:{MANAGED_TAG_NAMESPACE}/*",
            ],
            cwd=repo_dir,
        )
    except RuntimeError as exc:
        raise RuntimeError(f"official update fetch failed: {exc}") from exc
    branch_ref = f"managed/{source_branch}"
    has_remote_ref = subprocess.run(["git", "rev-parse", "--verify", branch_ref], cwd=str(repo_dir), capture_output=True, check=False).returncode == 0
    if not has_remote_ref:
        raise RuntimeError(f"official update branch is unavailable: {branch_ref}")
    from supervisor.update_source import (
        official_ref_has_constitution,
        resolve_managed_update_target,
    )

    def _capture(args):
        result = subprocess.run(
            args,
            cwd=str(repo_dir),
            capture_output=True,
            text=True,
            check=False,
        )
        return result.returncode, (result.stdout or "").strip(), (result.stderr or "").strip()

    update_channel = "stable" if source_branch == "main" else (
        "qa" if source_branch == "ouroboros-stable" else "development"
    )
    remote_ref, resolved, target_error = resolve_managed_update_target(
        "managed",
        source_branch,
        branch_ref,
        update_channel=update_channel,
        capture=_capture,
    )
    if not remote_ref or not resolved:
        raise RuntimeError(target_error or f"official update target is unavailable: {branch_ref}")

    if not official_ref_has_constitution(remote_ref, repo_dir=repo_dir):
        raise RuntimeError(
            f"official update target lacks a non-empty regular BIBLE.md: {remote_ref}"
        )
    has_local_branch = subprocess.run(["git", "rev-parse", "--verify", local_branch], cwd=str(repo_dir), capture_output=True, check=False).returncode == 0
    if fresh_clone:
        # `git clone --no-checkout` still creates a local branch for the remote
        # HEAD. Never let that unrelated default override the selected channel.
        subprocess.run(["git", "checkout", "-B", local_branch, remote_ref], cwd=str(repo_dir), check=True)
    elif has_local_branch:
        subprocess.run(["git", "checkout", local_branch], cwd=str(repo_dir), check=True)
        can_fast_forward = subprocess.run(
            ["git", "merge-base", "--is-ancestor", "HEAD", remote_ref],
            cwd=str(repo_dir),
            capture_output=True,
        )
        if can_fast_forward.returncode == 0:
            # Failure here (for example local dirty files) must preserve the
            # checkout; the normal smart updater can reconcile it later.
            subprocess.run(
                ["git", "merge", "--ff-only", remote_ref],
                cwd=str(repo_dir),
                capture_output=True,
                check=False,
            )
        elif can_fast_forward.returncode != 1:
            raise RuntimeError(f"could not compare local {local_branch} with {remote_ref}")
        # Local-ahead and truly diverged checkouts stay untouched. The selected
        # channel is persisted below and the normal smart updater owns merging.
    else:
        subprocess.run(["git", "checkout", "-B", local_branch, remote_ref], cwd=str(repo_dir), check=True)
    local_stable_branch = "ouroboros-stable"
    has_local_stable = subprocess.run(
        ["git", "show-ref", "--verify", "--quiet", f"refs/heads/{local_stable_branch}"],
        cwd=str(repo_dir),
        check=False,
    ).returncode == 0
    if not has_local_stable:
        # Desktop creates this fallback from its bundled source SHA. Colab has
        # no bundle: prefer the fetched Stable channel even when the first run
        # intentionally selected QA/Development, then never auto-advance it.
        stable_seed, _stable_sha, _stable_error = resolve_managed_update_target(
            "managed",
            "main",
            "managed/main",
            update_channel="stable",
            capture=_capture,
        )
        stable_seed = stable_seed or remote_ref
        if not official_ref_has_constitution(stable_seed, repo_dir=repo_dir):
            stable_seed = remote_ref
        subprocess.run(
            ["git", "branch", local_stable_branch, stable_seed],
            cwd=str(repo_dir),
            check=True,
        )
    version_path = repo_dir / "VERSION"
    app_version = version_path.read_text(encoding="utf-8").strip() if version_path.is_file() else ""
    # Source-mode Colab owns the same managed-update contract as desktop. This
    # marker describes provenance/local branches; runtime settings choose the feed.
    atomic_write_json(
        repo_dir / ".git" / "ouroboros-managed.json",
        {
            "schema_version": 1,
            "app_version": app_version,
            "source_sha": resolved,
            "source_branch": source_branch,
            "managed_remote_name": "managed",
            "managed_remote_url": source_url,
            "managed_remote_branch": source_branch,
            "managed_local_branch": local_branch,
            "managed_local_stable_branch": local_stable_branch,
            "managed_remote_stable_branch": "ouroboros-stable",
            "colab_source_mode": True,
        },
        trailing_newline=True,
    )
    return repo_dir


def _gateway_request(host: str, port: int) -> Callable[..., tuple]:
    base = f"http://{host}:{port}"

    def _call(method: str, path: str, body: Optional[Dict[str, Any]] = None, timeout: float = 60.0) -> tuple:
        data = json.dumps(body).encode("utf-8") if body is not None else None
        req = urllib.request.Request(base + path, data=data, method=method)
        if data is not None:
            req.add_header("Content-Type", "application/json")
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                raw = resp.read().decode("utf-8")
                return resp.status, (json.loads(raw) if raw.strip() else {})
        except urllib.error.HTTPError as exc:
            raw = exc.read().decode("utf-8", errors="replace")
            try:
                payload = json.loads(raw) if raw.strip() else {}
            except Exception:
                payload = {"error": raw[:300]}
            return exc.code, payload

    return _call


def ensure_native_telegram_live(
    host: str = "127.0.0.1",
    port: int = 8765,
    *,
    settings: Optional[Dict[str, Any]] = None,
    timeout: float = 180.0,
    request: Optional[Callable[..., tuple]] = None,
) -> Dict[str, Any]:
    """Grant, enable, and configure the bundled native Telegram skill.

    The native payload is seeded and hash-trusted by the launcher. Headless
    Colab only performs the same owner-policy steps the Skills UI would perform:
    grant the API-projected missing items when persisted auto-grant is enabled,
    enable the skill, then save its Telegram settings. It never installs from a
    marketplace or runs a skill review.
    """
    settings = settings or {}
    call = request or _gateway_request(host, port)
    status: Dict[str, Any] = {"ok": False, "skill": "telegram", "steps": []}

    # 1. Wait until the server accepts loopback requests (the gateway client has
    #    no built-in retry, and the notebook starts the server as a subprocess).
    deadline = time.time() + timeout
    ready = False
    while time.time() < deadline:
        try:
            code, _ = call("GET", "/api/health")
            if code == 200:
                ready = True
                break
        except Exception:
            pass
        time.sleep(1.0)
    if not ready:
        status["error"] = "server did not become ready"
        return status
    status["steps"].append("ready")

    if not str(settings.get("TELEGRAM_BOT_TOKEN") or "").strip():
        status["warning"] = "TELEGRAM_BOT_TOKEN is empty; Telegram cannot poll messages"

    quoted = urllib.parse.quote("telegram")

    # 2. Discover the launcher-seeded native skill and read its authoritative
    #    grant/conflict projection. Missing native payloads are packaging errors,
    #    not a reason to fall back to a Hub install.
    skill = None
    while time.time() < deadline:
        try:
            code, payload = call("GET", "/api/extensions")
        except Exception:
            time.sleep(1.0)
            continue
        rows = payload.get("skills") if isinstance(payload, dict) and isinstance(payload.get("skills"), list) else []
        candidate = next(
            (
                row for row in rows
                if isinstance(row, dict) and str(row.get("name") or "") == "telegram"
            ),
            None,
        )
        if (
            isinstance(code, int)
            and code < 400
            and candidate is not None
            and str(candidate.get("source") or "").lower() == "native"
            and str(candidate.get("review_profile") or "") == "native_seed"
            and candidate.get("executable_review") is True
            and candidate.get("review_stale") is False
        ):
            skill = candidate
            break
        time.sleep(1.0)
    if skill is None:
        status["error"] = "bundled native skill 'telegram' did not reach a fresh executable native-trust verdict"
        return status
    conflict = skill.get("conflict") if isinstance(skill.get("conflict"), dict) else None
    if conflict:
        names = [str(name) for name in (conflict.get("skills") or []) if str(name)]
        status["error"] = (
            "cannot enable native 'telegram' while conflicting skills are enabled: "
            + ", ".join(names)
        )
        status["conflict"] = conflict
        return status
    status["steps"].append("discovered")

    # 3. Respect the persisted owner auto-grant choice. Grant only the missing
    #    items projected by the API; never invent permissions in the notebook.
    grants = skill.get("grants") if isinstance(skill.get("grants"), dict) else {}
    missing = [
        str(item)
        for item in [
            *(grants.get("missing_keys") or []),
            *(grants.get("missing_permissions") or []),
        ]
        if str(item).strip()
        and str(item).strip().lower() not in {"chat.document", "subscribe_event:chat.document"}
    ]
    auto_raw = settings.get(
        "OUROBOROS_AUTO_GRANT_REVIEWED_SKILLS",
        SETTINGS_DEFAULTS.get("OUROBOROS_AUTO_GRANT_REVIEWED_SKILLS", "true"),
    )
    auto_grant = auto_raw if isinstance(auto_raw, bool) else str(auto_raw).strip().lower() in {"1", "true", "yes", "on"}
    if missing and not auto_grant:
        status["error"] = (
            "native Telegram needs owner grants, but automatic grants are disabled: "
            + ", ".join(missing)
        )
        status["missing_grants"] = missing
        return status
    if missing:
        try:
            code, payload = call(
                "POST",
                f"/api/skills/{quoted}/grants",
                {"items": missing},
            )
        except Exception as exc:
            status["error"] = f"grant request failed: {exc}"
            return status
        err = str((payload or {}).get("error") or "") if isinstance(payload, dict) else ""
        if (isinstance(code, int) and code >= 400) or err:
            status["error"] = f"grant failed: {err or f'HTTP {code}'}"
            return status
        status["steps"].append("granted")

    # 4. Enable through the canonical lifecycle endpoint.
    try:
        code, payload = call("POST", f"/api/skills/{quoted}/toggle", {"enabled": True})
    except Exception as exc:
        status["error"] = f"enable request failed: {exc}"
        return status
    err = str((payload or {}).get("error") or "") if isinstance(payload, dict) else ""
    if (isinstance(code, int) and code >= 400) or err:
        status["error"] = f"enable failed: {err or f'HTTP {code}'}"
        return status
    status["steps"].append("enabled")

    # 5. Preserve the working PoC behavior: full owner commands, all-message
    #    mirroring, and the Mini App enabled.
    desired_settings = {
        "TELEGRAM_COMMAND_MODE": "full_access",
        "TELEGRAM_MIRROR_MODE": "all",
        "TELEGRAM_MINIAPP_ENABLED": "on",
    }
    if any(str(value or "").strip() for value in desired_settings.values()):
        try:
            code, payload = call(
                "POST",
                f"/api/extensions/{quoted}/settings/save",
                desired_settings,
            )
        except Exception as exc:
            status["settings_ok"] = False
            status["warning"] = f"Telegram enabled but settings were not applied: {exc}"
        else:
            settings_err = str((payload or {}).get("error") or "") if isinstance(payload, dict) else ""
            if (isinstance(code, int) and code >= 400) or settings_err:
                status["settings_ok"] = False
                status["warning"] = f"Telegram enabled but settings were not applied: {settings_err or f'HTTP {code}'}"
            else:
                status["steps"].append("settings_saved")
                status["settings_ok"] = True

    status["ok"] = True
    return status


def server_command(repo_dir: pathlib.Path, *, host: str = "127.0.0.1", port: int = 8765) -> list[str]:
    return [sys.executable, "-m", "ouroboros.cli", "server", "--host", host, "--port", str(port), "--no-ui"]
