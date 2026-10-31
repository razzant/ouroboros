# %% [markdown]
# # Ouroboros Colab Quickstart
#
# Runs full source-mode Ouroboros in Google Colab without the desktop UI and
# brings up the Telegram control bridge automatically.

# %%
import json
import os
import pathlib
import re
import signal
import subprocess
import sys

try:
    from google.colab import drive  # type: ignore
except Exception as exc:  # pragma: no cover - only meaningful in Colab
    raise RuntimeError("This quickstart is intended for Google Colab.") from exc

drive.mount("/content/drive")

# Bootstrap cannot import the runtime mapping until it has selected code to
# clone, so mirror the closed three-entry channel map at this one boundary.
APP_ROOT = pathlib.Path("/content/drive/MyDrive/Ouroboros")
DATA_DIR = APP_ROOT / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
_existing_settings = {}
_settings_file = DATA_DIR / "settings.json"
if _settings_file.exists():
    try:
        _existing_settings = json.loads(_settings_file.read_text(encoding="utf-8"))
    except Exception:
        _existing_settings = {}
if not isinstance(_existing_settings, dict):
    _existing_settings = {}
_bootstrap_channel = str(
    os.environ.get("OUROBOROS_UPDATE_CHANNEL")
    if "OUROBOROS_UPDATE_CHANNEL" in os.environ
    else _existing_settings.get("OUROBOROS_UPDATE_CHANNEL") or "stable"
).strip().lower()
_bootstrap_branches = {
    "stable": "main",
    "qa": "ouroboros-stable",
    "development": "ouroboros",
}
if _bootstrap_channel not in _bootstrap_branches:
    _bootstrap_channel = "stable"
_initial_source_branch = _bootstrap_branches[_bootstrap_channel]
_BOOTSTRAP_MANAGED_TAG_NAMESPACE = "refs/ouroboros-managed/tags"
_BOOTSTRAP_RELEASE_TAG_RE = re.compile(r"^v(\d+)\.(\d+)\.(\d+)$")


def _bootstrap_git(args, *, cwd=None):
    """Bound the pre-import clone/migration step; the runtime helper is not importable yet."""
    raw_timeout = os.environ.get("OUROBOROS_MANAGED_UPDATE_FETCH_TIMEOUT_SEC", "300")
    try:
        timeout = max(30, min(int(float(raw_timeout)), 1800))
    except (TypeError, ValueError):
        timeout = 300
    env = {**os.environ, "GIT_TERMINAL_PROMPT": "0", "LC_ALL": "C", "LANG": "C"}
    proc = subprocess.Popen(
        ["git", "-c", "http.lowSpeedLimit=1024", "-c", "http.lowSpeedTime=30", *args],
        cwd=str(cwd) if cwd else None,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
        start_new_session=True,
    )
    try:
        stdout, stderr = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        raise RuntimeError("official git bootstrap timed out") from exc
    if proc.returncode != 0:
        raise RuntimeError((stderr or stdout or "official git bootstrap failed").strip())


def _bootstrap_ref_has_constitution(repo_dir, ref):
    """Mirror the runtime's P4 check until the selected code is importable."""
    result = subprocess.run(
        ["git", "ls-tree", "-l", ref, "--", "BIBLE.md"],
        cwd=str(repo_dir),
        capture_output=True,
        text=True,
        check=False,
    )
    fields = (result.stdout or "").strip().split(maxsplit=4)
    return bool(
        result.returncode == 0
        and len(fields) == 5
        and fields[0] in {"100644", "100755"}
        and fields[1] == "blob"
        and fields[3].isdigit()
        and int(fields[3]) > 0
        and fields[4] == "BIBLE.md"
    )


def _bootstrap_capture(repo_dir, args):
    result = subprocess.run(
        ["git", *args],
        cwd=str(repo_dir),
        capture_output=True,
        text=True,
        check=False,
    )
    return result.returncode, (result.stdout or "").strip(), (result.stderr or "").strip()


def _bootstrap_stable_ref(repo_dir):
    """Resolve Stable before importing any runtime code from the checkout."""
    main_ref = "origin/main"
    qa_ref = "origin/ouroboros-stable"
    for required_ref in (main_ref, qa_ref):
        rc, _sha, error = _bootstrap_capture(
            repo_dir, ["rev-parse", "--verify", f"{required_ref}^{{commit}}"]
        )
        if rc != 0:
            raise RuntimeError(error or f"stable release branch is unavailable: {required_ref}")
    rc, raw_tags, error = _bootstrap_capture(
        repo_dir,
        [
            "for-each-ref",
            "--format=%(refname:strip=3)",
            f"{_BOOTSTRAP_MANAGED_TAG_NAMESPACE}/v*",
        ],
    )
    if rc != 0:
        raise RuntimeError(error or "could not list official release tags")
    candidates = []
    for tag in raw_tags.splitlines():
        match = _BOOTSTRAP_RELEASE_TAG_RE.fullmatch(tag.strip())
        if match:
            candidates.append((tuple(int(part) for part in match.groups()), tag.strip()))
    for _version, tag in sorted(candidates, reverse=True):
        tag_ref = f"{_BOOTSTRAP_MANAGED_TAG_NAMESPACE}/{tag}"
        rc, sha, _error = _bootstrap_capture(
            repo_dir, ["rev-parse", "--verify", f"{tag_ref}^{{commit}}"]
        )
        if rc != 0 or not sha:
            continue
        if all(
            _bootstrap_capture(
                repo_dir, ["merge-base", "--is-ancestor", sha, release_ref]
            )[0] == 0
            for release_ref in (main_ref, qa_ref)
        ):
            return tag_ref
    raise RuntimeError("no shared stable vX.Y.Z release exists on main and ouroboros-stable")


# Minimal bootstrap clone so `ouroboros.colab_bootstrap` becomes importable.
# Remote roles and fast-forward updates are handled by clone_or_update_repo below.
REPO_DIR = pathlib.Path("/content/ouroboros_repo")
if not (REPO_DIR / ".git").exists():
    _bootstrap_git(
        ["clone", "--no-checkout", "https://github.com/razzant/ouroboros.git", str(REPO_DIR)],
    )
    _initial_source_ref = f"origin/{_initial_source_branch}"
    if _bootstrap_channel == "stable":
        _bootstrap_git(
            [
                "fetch",
                "origin",
                "+refs/heads/*:refs/remotes/origin/*",
                f"+refs/tags/*:{_BOOTSTRAP_MANAGED_TAG_NAMESPACE}/*",
            ],
            cwd=REPO_DIR,
        )
        _initial_source_ref = _bootstrap_stable_ref(REPO_DIR)
    if not _bootstrap_ref_has_constitution(REPO_DIR, _initial_source_ref):
        raise RuntimeError(
            f"official update target lacks a non-empty regular BIBLE.md: {_initial_source_ref}"
        )
    _bootstrap_git(
        ["checkout", "--detach", _initial_source_ref], cwd=REPO_DIR,
    )
elif not (REPO_DIR / "ouroboros" / "update_channels.py").is_file():
    # One-time bridge from pre-channel Colab checkouts. The old updater already
    # followed ouroboros, so fast-forward through that carrier before importing
    # the new channel-aware runtime. A dirty/conflicting checkout fails intact.
    _bootstrap_git(
        ["fetch", "https://github.com/razzant/ouroboros.git", "ouroboros"],
        cwd=REPO_DIR,
    )
    if not _bootstrap_ref_has_constitution(REPO_DIR, "FETCH_HEAD"):
        raise RuntimeError(
            "official update target lacks a non-empty regular BIBLE.md: FETCH_HEAD"
        )
    subprocess.run(["git", "merge", "--ff-only", "FETCH_HEAD"], cwd=REPO_DIR, check=True)

os.chdir(REPO_DIR)
sys.path.insert(0, str(REPO_DIR))

# %%
from ouroboros.colab_bootstrap import (
    build_colab_settings,
    clone_or_update_repo,
    collect_colab_secrets,
    configure_colab_personal_origin,
    ensure_native_telegram_live,
    export_colab_env,
    masked_secret_status,
    server_command,
    write_colab_settings,
)
from ouroboros.update_channels import get_update_branch, normalize_update_channel

# Preserve prior owner choices on Drive across ephemeral Colab sessions (a re-run
# of this cell must not wipe a pinned chat, tweaked models, or other prefs).
_requested_channel = normalize_update_channel(_bootstrap_channel)
_source_branch = get_update_branch({"OUROBOROS_UPDATE_CHANNEL": _requested_channel})
_settings_seed = dict(_existing_settings or {})
_settings_seed["OUROBOROS_UPDATE_CHANNEL"] = _requested_channel

# Fetch the selected official channel, but keep one stable local work branch.
clone_or_update_repo(
    REPO_DIR,
    source_branch=_source_branch,
    local_branch="ouroboros",
)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-e", "."], check=True)

secrets = collect_colab_secrets()
settings = build_colab_settings(
    secrets,
    total_budget=float(os.environ.get("TOTAL_BUDGET", "200")),
    runtime_mode=os.environ.get("OUROBOROS_RUNTIME_MODE", "advanced"),
    max_workers=int(os.environ.get("OUROBOROS_MAX_WORKERS", "1")),
    existing=_settings_seed,
    drive_document_present=bool(_existing_settings),
)
# GitHub persistence is optional: a personal fork is configured only when a token
# is present, otherwise the prototype still runs (without remote self-persistence).
origin_result = configure_colab_personal_origin(REPO_DIR, DATA_DIR, settings)
settings_path = write_colab_settings(DATA_DIR, settings)
export_colab_env(REPO_DIR, DATA_DIR, settings_path)

print("Secrets configured:", masked_secret_status(settings))
print("Personal origin:", origin_result)
print("Settings:", settings_path)

# %%
server = subprocess.Popen(
    server_command(REPO_DIR),
    cwd=str(REPO_DIR),
    env=os.environ.copy(),
)
print("Ouroboros server PID:", server.pid)

# Grant, enable, and configure the bundled native Telegram skill over loopback; the wait for the
# server's boot ends early if the server process exits.
telegram_status = ensure_native_telegram_live(settings=settings, alive=lambda: server.poll() is None)
print("Native Telegram:", telegram_status)
if telegram_status.get("ok") and telegram_status.get("settings_ok"):
    print("Message your Telegram bot now. Your first owner slash command (e.g. /status) registers your chat and asks you to send it once more;")
    print("after that, owner commands like /status and /panic run immediately.")
elif telegram_status.get("ok"):
    print("Telegram is enabled, but its settings were not applied:", telegram_status.get("warning"))
    print("Set full_access, mirror mode all, and Mini App on in the Telegram skill settings.")
else:
    print("Native Telegram not live yet:", telegram_status.get("error") or telegram_status)
