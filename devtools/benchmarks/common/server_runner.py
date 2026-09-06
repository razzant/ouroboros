"""Isolated-server runner for evolution benchmark drivers (B-full, production-faithful).

Spawns a REAL isolated Ouroboros ``server.py`` against a throwaway repo clone + data
root on a free port, so a benchmark drives the ACTUAL supervisor loop (post-task
evolution -> reviewed commit -> os.execvpe restart -> verify_restart absorb) instead
of a headless ``ouroboros run`` that would attach to whatever server is on the
default port. The live Ouroboros is never touched: a unique port, an isolated clone,
and an isolated data root keep it fully separate.

Why a server (not headless): the post-task evolution signal is only consumed by the
supervisor tick inside ``server.py`` (``apply_pending_request`` +
``enqueue_evolution_task_if_needed``); ``ouroboros run`` is a thin HTTP client.

Model: tests/test_ui_smoke_playwright.py::direct_server_with_data +
devtools/benchmarks/terminal_bench/harbor_installed_agent.py.
"""
from __future__ import annotations

import hashlib
import json
import os
import pathlib
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid

if __package__ in {None, ""}:
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3]))

from devtools.benchmarks.common.manifests import runtime_attestation
from devtools.benchmarks.common.secrets import isolated_credential_grants  # noqa: F401 (re-export)
from ouroboros.context_mode_compat import normalize_context_mode_compat
from ouroboros.platform_layer import (
    kill_pid_tree,
    subprocess_new_group_kwargs,
    terminate_process_tree,
)
from ouroboros.provider_models import (
    ALL_PROVIDER_CREDENTIAL_KEYS,
    LEGACY_MODEL_SETTING_KEYS,
    provider_credential_plan,
)

_FINAL_STATUSES = {"completed", "failed", "cancelled", "rejected_duplicate"}

# Live/managed runtime env keys that must NEVER leak into an isolated benchmark server:
# the sanitized settings.json is the source of truth, so an inherited value here would
# silently route the throwaway server through the LIVE local-model/runtime/host config.
# SSOT for BOTH IsolatedServer._env() (process env) and the drivers' _seed_settings()
# (copied settings.json) so the two sanitizations can never drift apart.
STALE_INHERITED_ENV_KEYS = (
    "OUROBOROS_SERVER_HOST", "OUROBOROS_SERVER_PORT", "OUROBOROS_HOST_SERVICE_PORT",
    "OUROBOROS_APP_ROOT", "OUROBOROS_REPO_DIR", "OUROBOROS_DATA_DIR", "OUROBOROS_SETTINGS_PATH",
    "OUROBOROS_URL", "OUROBOROS_MANAGED_BY_LAUNCHER",
    "OUROBOROS_SETTINGS_SHA256",
    # The launcher-exported presentation posture describes the OPERATOR's desktop
    # process; an isolated benchmark server is a headless web process and must
    # not inherit "desktop_window".
    "OUROBOROS_PRESENTATION",
    # The parent-pinned runtime-mode baseline is exported to subprocesses and is PREFERRED
    # over settings.json (config.initialize_runtime_mode_baseline / get_runtime_mode), so an
    # inherited value would boot the isolated server in the LIVE mode instead of its own
    # advanced sandbox — strip it so the sanitized settings win.
    "OUROBOROS_BOOT_RUNTIME_MODE",
    "USE_LOCAL_MAIN", "USE_LOCAL_CODE", "USE_LOCAL_LIGHT", "USE_LOCAL_FALLBACK",
    "USE_LOCAL_CONSCIOUSNESS", "OUROBOROS_REVIEWER_SLOTS",
    # Owner/control SECRETS must never leak into the isolated server's env (untrusted
    # benchmark tasks run here). Provider creds are loaded from the sanitized settings.json.
    "GITHUB_TOKEN", "GITHUB_REPO", "OUROBOROS_NETWORK_PASSWORD",
)

# Allowlist for seeding an ISOLATED benchmark settings.json from live settings: ONLY provider
# credentials/endpoints, model slots, effort, and budget. Owner/control secrets and knobs
# (GITHUB_TOKEN, OUROBOROS_NETWORK_PASSWORD, transport/skill secrets, owner chat ids, etc.)
# are NEVER copied — the isolated data root is readable by untrusted benchmark tasks.
# Model-slot / effort / local-model key families (all are model config — safe to copy).
# NOTE the absent `GIGACHAT_` prefix: every provider credential family, GigaChat's included,
# is gated on the run's DECLARED slots below instead of riding an unconditional prefix.
_ISO_SETTINGS_ALLOW_PREFIX = ("OUROBOROS_MODEL", "OUROBOROS_EFFORT", "LOCAL_MODEL_")
# NON-credential review/budget/model keys. Provider credentials are deliberately NOT here —
# see _grant_provider_credentials. Deliberately NOT a `*_API_KEY` pattern either: a custom
# skill secret could be named `<x>_API_KEY` and must NOT be copied.
_ISO_SETTINGS_ALLOW_EXACT = frozenset({
    "OUROBOROS_SUBAGENTS",
    "OUROBOROS_REVIEWER_SLOTS",
    "OUROBOROS_WEBSEARCH_MODEL", "OUROBOROS_REVIEW_MODELS",
    "OUROBOROS_SCOPE_REVIEW_MODELS", "OUROBOROS_SCOPE_REVIEW_MODEL",
    # Review policy knobs (non-secret): must propagate so settings.json's task-acceptance
    # self-review config is honored by isolated benchmark servers (else it silently
    # falls back to the "auto" default and the end-of-task review never runs).
    "OUROBOROS_TASK_REVIEW_MODE", "OUROBOROS_REVIEW_ENFORCEMENT",
    # Shared paid review-cycle ceiling (non-secret): bench templates pin it to
    # "unlimited" for methodology comparability; a live-settings pin must forward
    # into isolated bench servers the same way the other review knobs do.
    "OUROBOROS_REVIEW_MAX_CYCLES",
    # One-window false provenance tombstone: it travels with an explicit Low so
    # the isolated run keeps owner-Low/P3 semantics. Legacy true is normalized.
    "TOTAL_BUDGET", "OUROBOROS_PER_TASK_COST_USD", "OUROBOROS_CONTEXT_MODE",
    "OUROBOROS_CONTEXT_MODE_AUTO_LOW",
})


# Provider credentials the isolated agent legitimately needs in its env (kept); every other
# secret-shaped inherited env var is stripped so untrusted benchmark tasks (which inherit the
# server env via shell tools) cannot read owner/skill secrets like TELEGRAM_BOT_TOKEN.
_PROVIDER_ENV_KEYS = frozenset({
    "OPENROUTER_API_KEY", "OPENAI_API_KEY", "OPENAI_COMPATIBLE_API_KEY",
    "CLOUDRU_FOUNDATION_MODELS_API_KEY", "ANTHROPIC_API_KEY", "MINIMAX_API_KEY",
    "DEEPSEEK_API_KEY",
    "GIGACHAT_CREDENTIALS", "GIGACHAT_PASSWORD",
})

# In the CyberGym wrapper the applied settings snapshot is the authority.  A
# parent process can still carry compatibility aliases and newer runtime knobs
# that are not present in ``SETTINGS_DEFAULTS``; retaining any of those would
# make an otherwise identical run depend on the operator shell.  Keep only the
# path/port values that this lifecycle writes back explicitly, plus one
# operational host-load lever: ``OUROBOROS_PREFLIGHT_TEST_WORKERS`` caps the
# xdist fan-out of the commit gate's hermetic pytest pass (read by
# ``preflight_runner._preflight_worker_count`` in the server process, never a
# model/credential/settings key, and still scrubbed from the candidate suite's
# own environment by ``preflight_runner._preflight_env``).  Without it every
# isolated server resolves ``-n auto`` to the host's CPU count.  This is scoped
# to ``settings_authoritative_env`` below; older env-first benchmark drivers
# retain their historical inheritance contract.
_AUTHORITATIVE_ENV_KEEP = frozenset({
    "OUROBOROS_APP_ROOT",
    "OUROBOROS_REPO_DIR",
    "OUROBOROS_DATA_DIR",
    "OUROBOROS_SETTINGS_PATH",
    "OUROBOROS_SERVER_HOST",
    "OUROBOROS_SERVER_PORT",
    "OUROBOROS_HOST_SERVICE_PORT",
    "OUROBOROS_PREFLIGHT_TEST_WORKERS",
})
_AUTHORITATIVE_ENV_PREFIXES = (
    "OUROBOROS_",
    "OPENROUTER_",
    "OPENAI_",
    "ANTHROPIC_",
    "MINIMAX_",
    "DEEPSEEK_",
    "CLOUDRU_",
    "GIGACHAT_",
    "CLAUDE_",
    "MCP_",
    "USE_LOCAL_",
    "LOCAL_MODEL_",
    "HOST_SERVICE_",
)
_AUTHORITATIVE_ENV_EXACT = frozenset({
    # Legacy model/local aliases that predate the current settings registry.
    "OUROBOROS_MODEL_CODE",
    "OUROBOROS_VISION_MODEL",
    "OUROBOROS_MODEL_FALLBACK",
    "USE_LOCAL_CODE",
    "TOTAL_BUDGET",
})

# A strict isolated server receives the digest of the post-port-patch settings
# bytes in its child environment. The child verifies the same open-file bytes
# before applying defaults, so a replacement between parent preflight and child
# import cannot silently fall back to product defaults.
SETTINGS_INTEGRITY_ENV = "OUROBOROS_SETTINGS_SHA256"


def _settings_json_bytes(config: dict) -> bytes:
    """Serialize a settings snapshot exactly as it is written to disk."""
    return json.dumps(config, ensure_ascii=False, indent=2).encode("utf-8")


def _is_secret_env_key(key: str) -> bool:
    """A non-provider secret-shaped env var (token/secret/password/api-key/credentials)."""
    ku = str(key).upper()
    if ku in _PROVIDER_ENV_KEYS:
        return False
    return (
        "TOKEN" in ku or "SECRET" in ku or "PASSWORD" in ku
        or ku.endswith("_API_KEY") or ku.endswith("_CREDENTIALS")
    )


def build_isolated_settings(
    live_cfg: dict,
    *,
    include_claude_sdk_defaults: bool = True,
    **overrides,
) -> dict:
    """Build an isolated benchmark settings.json from live settings: copy the non-credential
    model/effort/budget/review allowlist above, apply the explicit isolated overrides, and
    then grant ONLY the provider credentials the resulting run's DECLARED model slots need.

    Owner/control secrets (GITHUB_TOKEN, OUROBOROS_NETWORK_PASSWORD, transport/skill secrets,
    owner knobs) were never copied and still are not. What changes here is narrower and was
    the real defect: the copied provider set used to be a function of whatever happened to be
    in the live settings file at launch, so a run pinned to OpenRouter still received direct
    ANTHROPIC_API_KEY / OPENAI_API_KEY / Cloud.ru / GigaChat credentials. A routing fallback
    could then spend outside the declared bucket while the manifest said otherwise, and two
    nominally identical runs could reach different providers invisibly — a pinned seed that
    pins the code but not the environment is not reproducible.

    Credentials travel in whole GROUPS (``PROVIDER_CREDENTIAL_GROUPS``), so a key never
    arrives without the endpoint/auth fields it is useless without (GigaChat
    CREDENTIALS+PASSWORD+endpoint+scope, Cloud.ru key+base_url). An explicit override always
    wins over the derived grant. Use ``isolated_credential_grants`` on the RESULT to record
    what was granted."""
    out: dict = {}
    for key, value in (live_cfg or {}).items():
        ks = str(key)
        if ks in LEGACY_MODEL_SETTING_KEYS:
            continue
        if ks in ALL_PROVIDER_CREDENTIAL_KEYS:
            continue  # gated below on the declared slots, never copied wholesale
        if ks in _ISO_SETTINGS_ALLOW_EXACT or ks.startswith(_ISO_SETTINGS_ALLOW_PREFIX):
            out[ks] = value
    out.update(overrides)
    if "OUROBOROS_CONTEXT_MODE" in overrides and "OUROBOROS_CONTEXT_MODE_AUTO_LOW" not in overrides:
        # A benchmark override is an explicit operator choice, not ambiguous legacy disk state.
        out["OUROBOROS_CONTEXT_MODE_AUTO_LOW"] = "false"
    out = normalize_context_mode_compat(out)
    for key in provider_credential_plan(
        out,
        include_claude_sdk_defaults=include_claude_sdk_defaults,
    )["planned_keys"]:
        if key in (overrides or {}):
            continue
        value = (live_cfg or {}).get(key)
        if value not in (None, ""):
            out[key] = value
    return out


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _api(base_url: str, method: str, path: str, payload: dict | None = None, timeout: float = 60) -> dict:
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    headers = {"Content-Type": "application/json"} if data is not None else {}
    req = urllib.request.Request(base_url + path, data=data, method=method, headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read().decode("utf-8", errors="replace")
    return json.loads(raw) if raw.strip() else {}


def _api_status(base_url: str, method: str, path: str, payload: dict | None = None,
                timeout: float = 60) -> dict:
    """Like ``_api`` but returns ``{"status": <http status>, "body": {...}}`` and never
    raises for an error status.

    The owner control surface answers its REFUSALS typed (404 ``task_not_live``, 409
    ``cancel_pending``, 503 ``cancel_intent_projection_corrupt``, 202 ``pending``), and
    urllib turns every non-2xx into an exception — so a driver built on ``_api`` can only
    see "it threw", which is exactly the distinction an owner-control scenario has to
    assert. Transport failures (server gone) surface as ``status == 0``.
    """
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    headers = {"Content-Type": "application/json"} if data is not None else {}
    req = urllib.request.Request(base_url + path, data=data, method=method, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            status = int(resp.status)
            raw = resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        status = int(exc.code)
        raw = exc.read().decode("utf-8", errors="replace")
    except (urllib.error.URLError, OSError) as exc:
        return {"status": 0, "body": {}, "error": repr(exc)}
    try:
        parsed = json.loads(raw) if raw.strip() else {}
    except ValueError:
        parsed = {}
    return {"status": status, "body": parsed if isinstance(parsed, dict) else {"raw": parsed}}


def seed_owner_state(data_root: pathlib.Path, *, evolution_enabled: bool = False) -> None:
    """Pre-seed state.json so the evolution loop's owner_chat_id gate passes (the
    /api/tasks path never binds owner_chat_id). Optionally pre-enable the campaign."""
    state_path = pathlib.Path(data_root) / "state" / "state.json"
    state_path.parent.mkdir(parents=True, exist_ok=True)
    st: dict = {}
    if state_path.exists():
        try:
            st = json.loads(state_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            st = {}
    st["owner_chat_id"] = 1
    if evolution_enabled:
        campaign_path = pathlib.Path(data_root) / "state" / "evolution_campaign.json"
        now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        campaign_path.write_text(json.dumps({
            "schema_version": 1,
            "id": uuid.uuid4().hex[:8],
            "status": "active",
            "objective": "Autonomously improve Ouroboros from benchmark evidence.",
            "source": "benchmark",
            "started_at": now,
            "updated_at": now,
            "cycles_done": 0,
            "absorbed_cycles_done": 0,
        }), encoding="utf-8")
        st["evolution_mode_enabled"] = True
    state_path.write_text(json.dumps(st), encoding="utf-8")


def campaign_summary(data_root: pathlib.Path) -> dict:
    """The durable campaign facts an absorb wait reasons about: evolution_campaign.json (presence, status,
    source, a pending ``active_transaction``, the newest transaction outcome, the absorbed counter) and the
    post-task promotion counter (``post_task_evolution_counter.json``: the decision ran at least once)."""
    state_dir = pathlib.Path(data_root) / "state"
    try:
        campaign = json.loads((state_dir / "evolution_campaign.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        campaign = {}
    campaign = campaign if isinstance(campaign, dict) else {}
    try:
        counter = int(json.loads((state_dir / "post_task_evolution_counter.json").read_text(encoding="utf-8"))["n"])
    except (OSError, ValueError, TypeError, KeyError):
        counter = 0
    history = [tx for tx in (campaign.get("transaction_history") or []) if isinstance(tx, dict)]
    return {"present": bool(campaign), "status": str(campaign.get("status") or ""),
            "source": str(campaign.get("source") or ""),
            "active_transaction": isinstance(campaign.get("active_transaction"), dict),
            "absorbed_cycles_done": int(campaign.get("absorbed_cycles_done") or 0),
            "history_len": len(history),
            "newest_outcome": str(history[-1].get("cycle_outcome") or "") if history else "",
            "post_task_counter": counter}


def absorb_idle_reason(campaign: dict, history_len_at_start: int = 0) -> str:
    """Typed non-confirmation of an idle lane (``IsolatedServer.wait_for_absorb``), relative to the wait's
    start so a resumed campaign's OLDER cycles never speak for this boundary: ``campaign_<status>`` (paused/
    stopped/completed wins), ``no_promotion`` (no campaign although an ``every_n`` post-task tick was
    recorded — the decision may still have declined), ``no_decision`` (no campaign, no tick recorded: ``llm``
    cadences write none), ``cycle_no_op`` / ``cycle_not_absorbed`` (a cycle newer than the wait ended without
    an absorb) or ``cycle_not_enqueued`` (a campaign that attached no new cycle)."""
    if campaign.get("status") in ("paused", "stopped", "completed"):
        return f"campaign_{campaign['status']}"
    if not campaign.get("present"):
        return "no_promotion" if campaign.get("post_task_counter") else "no_decision"
    if int(campaign.get("history_len") or 0) > int(history_len_at_start or 0):
        return "cycle_no_op" if campaign.get("newest_outcome") == "no_op" else "cycle_not_absorbed"
    return "cycle_not_enqueued"


def absorbed_cycles_done(data_root: pathlib.Path) -> int:
    """Read absorbed self-evolution cycle count from evolution_campaign.json."""
    path = pathlib.Path(data_root) / "state" / "evolution_campaign.json"
    try:
        return int(json.loads(path.read_text(encoding="utf-8")).get("absorbed_cycles_done") or 0)
    except (OSError, ValueError, TypeError):
        return 0


def patch_settings_ports(settings_path: pathlib.Path, *, host: str, port: int,
                         host_service_port: int, require_existing_object: bool = False,
                         expected_sha256: str | None = None) -> dict:
    """Write the chosen ports INTO a settings.json, returning the merged config.

    THE reason this exists rather than exporting the ports in the environment: the server
    applies settings.json OVER the environment at startup (``apply_settings_to_env``), so an
    env-only ``OUROBOROS_HOST_SERVICE_PORT`` is overwritten by whatever the settings file says
    — or by the 8767 default when it says nothing — and every server started from a shared
    template collides on that port. Shared with the generated OSWorld lanes, which need the
    same per-instance isolation ``IsolatedServer`` gets.
    """
    settings_path = pathlib.Path(settings_path)
    cfg: dict = {}
    raw: bytes | None = None
    try:
        raw = settings_path.read_bytes()
        if expected_sha256 is not None:
            observed_sha256 = hashlib.sha256(raw).hexdigest()
            if observed_sha256 != expected_sha256:
                raise RuntimeError("isolated settings snapshot changed")
        loaded = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        if require_existing_object:
            # Do not fall through to the legacy ports-only write after a strict
            # snapshot disappears or becomes malformed between reads.
            raise RuntimeError("isolated settings snapshot is unreadable") from exc
    else:
        if isinstance(loaded, dict):
            cfg = loaded
        elif require_existing_object:
            raise RuntimeError("isolated settings snapshot must be a JSON object")
    cfg["OUROBOROS_SERVER_HOST"] = host
    cfg["OUROBOROS_SERVER_PORT"] = int(port)
    cfg["OUROBOROS_HOST_SERVICE_PORT"] = int(host_service_port)
    if expected_sha256 is not None and raw is None:  # pragma: no cover - defensive invariant
        raise RuntimeError("isolated settings snapshot is unreadable")
    settings_path.parent.mkdir(parents=True, exist_ok=True)
    settings_path.write_bytes(_settings_json_bytes(cfg))
    return cfg


def supervisor_state_is_ready(state: dict) -> bool:
    """THE readiness contract for an Ouroboros server, from the frozen `/api/state` shape.

    `/api/health` answering 200 is NOT readiness: it can succeed while the supervisor is
    still starting, which is exactly why `supervisor_ready` exists as a separate field. A
    ready server also has at least one worker — `supervisor_ready` with `workers_total == 0`
    accepts a task that nothing will pick up. Shared so every launch path (this class and the
    generated OSWorld lanes) asks the same question instead of each inventing its own.
    """
    return bool(state.get("supervisor_ready")) and int(state.get("workers_total") or 0) > 0


class IsolatedServer:
    """A throwaway Ouroboros server bound to an isolated clone + data root + port."""

    def __init__(self, clone: pathlib.Path, data_root: pathlib.Path, settings_path: pathlib.Path,
                 *, host: str = "127.0.0.1", settings_authoritative_env: bool = False,
                 expected_settings_sha256: str | None = None) -> None:
        self.clone = pathlib.Path(clone)
        self.data_root = pathlib.Path(data_root)
        self.settings_path = pathlib.Path(settings_path)
        self.host = host
        # Some adapters inject one deliberately selected credential after this base
        # environment is built. Those adapters opt into settings-authoritative mode so
        # ambient provider/model settings cannot shadow the applied snapshot. The default
        # stays compatible with older env-first drivers (for example CLB's host path).
        self.settings_authoritative_env = bool(settings_authoritative_env)
        self.port = free_port()
        self.host_service_port = free_port()
        self.base_url = f"http://{host}:{self.port}"
        self.proc: subprocess.Popen | None = None
        # Stable per-task hurry request ids (see `hurry_task`), the driver-side mirror of
        # the UI's `hurryRequestId` map.
        self._hurry_request_ids: dict = {}
        # Digest of the exact settings snapshot admitted by a strict adapter.  It
        # is carried across the port patch and checked again immediately before
        # spawn, so a valid replacement cannot silently alter the applied run.
        expected = str(expected_settings_sha256 or "").strip().lower()
        if expected and (
            len(expected) != 64
            or any(char not in "0123456789abcdef" for char in expected)
        ):
            raise ValueError("expected_settings_sha256 must be a lowercase SHA-256 digest")
        if expected and not self.settings_authoritative_env:
            raise ValueError("expected_settings_sha256 requires settings_authoritative_env")
        self._authoritative_settings_sha256: str | None = expected or None
        # Filled by _wait_ready: the HTTP runtime_version + the clone's HEAD/VERSION that
        # produced it, so a driver can record WHICH agent identity its numbers came from.
        self.attestation: dict = {}

    def _env(self) -> dict:
        env = dict(os.environ)
        # Strip ALL stale live/managed runtime keys FIRST, so an Ouroboros-managed launch
        # environment cannot REINTRODUCE values that _seed_settings stripped from the copied
        # settings (hermetic isolation: the sanitized settings.json is the source of truth;
        # a leaked USE_LOCAL_*/host/path here would route the throwaway server through live
        # config). This includes OUROBOROS_MANAGED_BY_LAUNCHER (direct self-re-exec, not
        # launcher-managed) and OUROBOROS_URL (never point the in-process CLI at another server).
        for key in STALE_INHERITED_ENV_KEYS:
            env.pop(key, None)
        for key in list(env):
            if _is_secret_env_key(key):
                env.pop(key, None)
        if self.settings_authoritative_env:
            # The applied settings file is the source of truth for this adapter.  Do
            # not merely clear today's known slots: remove the complete Ouroboros
            # namespace plus provider/SDK families, including legacy aliases and
            # future settings keys.  The file is read strictly; a malformed or
            # vanished snapshot must stop before a child can boot on defaults.
            self._read_authoritative_settings()
            for key in list(env):
                if key in _AUTHORITATIVE_ENV_KEEP:
                    continue
                if key in _AUTHORITATIVE_ENV_EXACT or key.startswith(_AUTHORITATIVE_ENV_PREFIXES):
                    env.pop(key, None)
        # Then apply the isolated overrides explicitly (these win over anything inherited).
        env.update({
            "OUROBOROS_APP_ROOT": str(self.clone.parent),
            "OUROBOROS_REPO_DIR": str(self.clone),
            "OUROBOROS_DATA_DIR": str(self.data_root),
            "OUROBOROS_SETTINGS_PATH": str(self.settings_path),
            "OUROBOROS_SERVER_HOST": self.host,
            "OUROBOROS_SERVER_PORT": str(self.port),
            "OUROBOROS_HOST_SERVICE_PORT": str(self.host_service_port),
        })
        if self.settings_authoritative_env:
            # _read_authoritative_settings above has just verified the exact
            # post-port-patch bytes. Carry that digest into the child; the
            # child repeats the open/read/hash check before defaults are applied.
            if not self._authoritative_settings_sha256:
                raise RuntimeError("isolated settings snapshot has no integrity digest")
            env[SETTINGS_INTEGRITY_ENV] = self._authoritative_settings_sha256
            # A headless benchmark task may still resolve the logical
            # user_files/deliverables roots.  Keep those roots inside the
            # throwaway data root instead of letting the runtime fall back to
            # the operator's real home when the ambient variables were scrubbed.
            user_files = self.data_root / "user_files"
            deliverables = user_files / "Deliverables"
            user_files.mkdir(parents=True, exist_ok=True)
            deliverables.mkdir(parents=True, exist_ok=True)
            env.update({
                "OUROBOROS_USER_FILES_ROOT": str(user_files),
                "OUROBOROS_DELIVERABLES_ROOT": str(deliverables),
            })
        return env

    def _read_authoritative_settings(self) -> dict:
        """Read the applied snapshot or fail closed before spawning the server."""
        try:
            raw = self.settings_path.read_bytes()
            observed_sha256 = hashlib.sha256(raw).hexdigest()
            if (
                self._authoritative_settings_sha256 is not None
                and observed_sha256 != self._authoritative_settings_sha256
            ):
                raise RuntimeError("isolated settings snapshot changed")
            loaded = json.loads(raw.decode("utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RuntimeError("isolated settings snapshot is unreadable") from exc
        if not isinstance(loaded, dict):
            raise RuntimeError("isolated settings snapshot must be a JSON object")
        self._authoritative_settings_sha256 = observed_sha256
        return loaded

    def _patch_settings_ports(self) -> None:
        """Write the chosen free ports INTO settings.json (see `patch_settings_ports`)."""
        _patched_settings = patch_settings_ports(
            self.settings_path,
            host=self.host,
            port=self.port,
            host_service_port=self.host_service_port,
            require_existing_object=self.settings_authoritative_env,
            expected_sha256=(
                self._authoritative_settings_sha256
                if self.settings_authoritative_env
                else None
            ),
        )
        if self.settings_authoritative_env:
            # Derive the digest from the exact bytes handed to the writer,
            # rather than accepting an independently replaced file as the next
            # baseline during a second read.
            self._authoritative_settings_sha256 = hashlib.sha256(
                _settings_json_bytes(_patched_settings)
            ).hexdigest()

    def start(self, ready_timeout: float = 180) -> "IsolatedServer":
        if self.settings_authoritative_env:
            # Validate before ``patch_settings_ports``: that helper is deliberately
            # permissive for legacy drivers and would otherwise turn malformed JSON
            # into a defaults-only file, defeating the authority contract.
            self._read_authoritative_settings()
        self._patch_settings_ports()
        # Own process group/session so a hung server + its worker children can be
        # killed as a tree (platform_layer), not orphaned past graceful SIGTERM.
        self.proc = subprocess.Popen(
            [sys.executable, "server.py"], cwd=str(self.clone), env=self._env(),
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            **subprocess_new_group_kwargs(),
        )
        try:
            self._wait_ready(ready_timeout)
        except BaseException:
            # NEVER orphan the spawned server/worker tree if readiness fails (timeout, etc.):
            # via __enter__ a raise here would skip __exit__, leaking the process group.
            self.stop()
            raise
        return self

    def _state(self, timeout: float = 5) -> dict:
        return _api(self.base_url, "GET", "/api/state", timeout=timeout)

    def _wait_ready(self, timeout: float) -> None:
        """Poll until the SUPERVISOR is ready (see `supervisor_state_is_ready`)."""
        deadline = time.time() + timeout
        last = ""
        while time.time() < deadline:
            if self.proc is not None and self.proc.poll() is not None:
                raise RuntimeError(f"isolated server exited early (rc={self.proc.returncode})")
            try:
                st = self._state()
                if supervisor_state_is_ready(st):
                    # Owner Q9=A+B: the identity attestation rides inside the readiness path
                    # every IsolatedServer driver must run, so no driver can skip it. It is a
                    # ONE-SHOT step here (not part of the polled probe): a raise inside the
                    # poll would be swallowed as "not ready yet" and burn the whole timeout.
                    self.attestation = runtime_attestation(self.base_url, self.clone)
                    return
                last = f"supervisor_ready={st.get('supervisor_ready')} workers={st.get('workers_total')}"
            except (urllib.error.URLError, OSError, ValueError) as exc:
                last = repr(exc)
            time.sleep(2)
        raise RuntimeError(f"isolated server not ready in {timeout}s ({last})")

    def current_sha(self) -> str:
        try:
            return str(self._state(timeout=10).get("sha") or "")
        except (urllib.error.URLError, OSError, ValueError):
            return ""

    def submit(self, description: str, *, workspace_root: str = "",
               memory_mode: str = "forked", timeout_sec: int = 1800) -> str:
        body: dict = {
            "description": description,
            "memory_mode": memory_mode,
            "actor_id": "evolve-driver",
            "source": "evolve-driver",
            "timeout_sec": timeout_sec,
            "metadata": {"source": "evolve-driver", "delegation_role": "root"},
        }
        if workspace_root:
            body["workspace_root"] = str(workspace_root)
            body["workspace_mode"] = "external"
        created = _api(self.base_url, "POST", "/api/tasks", body, timeout=60)
        return str(created.get("task_id") or "")

    def wait_task(self, task_id: str, timeout: float = 2400) -> dict:
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                result = _api(self.base_url, "GET", "/api/tasks/" + urllib.parse.quote(task_id), timeout=30)
                if str(result.get("status") or "") in _FINAL_STATUSES:
                    return result
            except (urllib.error.URLError, OSError, ValueError):
                pass  # transient (e.g. server re-exec restart) — keep polling
            time.sleep(3)
        return {"status": "timeout"}

    def cancel_task(self, task_id: str, *, cascade: bool = False, stop_policy: str = "",
                    timeout: float = 300) -> dict:
        """Owner stop over the SAME HTTP surface the web UI drives.

        Body assembled exactly like ``cancelTask`` in ``web/modules/api_client.js``: the
        two axes are independent — ``cascade`` selects the subtree teardown, ``stop_policy``
        selects the terminalization policy (``finalize_then_cancel`` = the graceful
        202/``cancel_state=pending`` acknowledgement; absent or ``immediate`` = today's hard
        cancel). An options-free call still posts ``{}``, so the pre-existing best-effort
        callers (a driver cleaning up after its own ``wait_task`` deadline) keep the
        byte-identical legacy single-task request they have always sent.

        The cascade lane answers only once the subtree is actually torn down, hence the
        wide default timeout. Returns the ``_api_status`` envelope; the refusal statuses are
        part of the contract under test, so nothing is raised or swallowed.
        """
        body: dict = {}
        if cascade:
            body["cascade"] = True
        policy = str(stop_policy or "")
        if policy and policy != "immediate":
            body["stop_policy"] = policy
        return _api_status(
            self.base_url, "POST",
            "/api/tasks/" + urllib.parse.quote(task_id) + "/cancel", body, timeout=timeout)

    def hurry_task(self, task_id: str, request_id: str = "") -> dict:
        """Owner hurry over the SAME HTTP surface the web UI drives (``hurryTask`` in
        ``web/modules/api_client.js``): ``POST /api/tasks/{id}/hurry`` with a body carrying
        ONLY the stable client-generated ``request_id`` — the endpoint refuses any other
        field rather than dropping it, and this path never produces a chat message.

        An omitted ``request_id`` mints a per-driver STABLE id for the task, mirroring the
        UI's ``hurryRequestId`` map: a retry of the same logical hurry reuses the id and is
        acknowledged idempotently instead of minting a second typed control.
        """
        rid = str(request_id or "").strip() or self._hurry_request_ids.setdefault(
            task_id, f"hurry-{uuid.uuid4()}")
        return _api_status(
            self.base_url, "POST",
            "/api/tasks/" + urllib.parse.quote(task_id) + "/hurry",
            {"request_id": rid}, timeout=30)

    def wait_for_health(self, timeout: float = 180) -> bool:
        """Wait for /api/state to answer with supervisor ready again (after a
        self-evolution os.execvpe re-exec the same PID restarts on new code)."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                st = self._state(timeout=5)
                if st.get("supervisor_ready") and int(st.get("workers_total") or 0) > 0:
                    return True
            except (urllib.error.URLError, OSError, ValueError):
                pass
            time.sleep(2)
        return False

    def wait_for_absorb(self, prev_sha: str, prev_absorbed: int, timeout: float = 1800,
                        idle_grace: float = 90, idle_polls: int = 6) -> dict:
        """Between instances, wait for an absorbed self-evolution cycle: the server re-execs onto a
        new SHA and ``absorbed_cycles_done`` increments. Returns ``{absorbed, new_sha, cycles, reason,
        campaign}``. An EARLY ``absorbed=False`` needs PROOF that no cycle is pending, held on
        ``idle_polls`` consecutive polls after ``idle_grace``: the queue idle AND ``supervisor_ready``
        AND no ``post_task_evolution_request.json`` AND no campaign ``active_transaction``. One idle
        sample is not proof: a cycle that committed keeps its transaction as ``waiting_for_restart``
        while the supervisor restarts synchronously (queue empty, counter unchanged), and the re-exec'd
        server answers ``/api/state`` with zero counts before its supervisor is up — the counter moves
        only when the worker boot verifies the restart (rc.15 stand, adversarial finding of 2026-09-06).
        The typed reason is what the durable campaign state proves (``absorb_idle_reason``)."""
        deadline = time.time() + timeout
        start = time.time()
        request_path = self.data_root / "state" / "post_task_evolution_request.json"
        idle_streak, history_at_start = 0, campaign_summary(self.data_root)["history_len"]
        while time.time() < deadline:
            cycles = absorbed_cycles_done(self.data_root)
            sha = self.current_sha()
            if cycles > prev_absorbed and sha and sha != prev_sha:
                self.wait_for_health(timeout=180)
                return {"absorbed": True, "new_sha": sha, "cycles": cycles, "reason": "absorbed",
                        "campaign": campaign_summary(self.data_root)}
            if time.time() - start > idle_grace and cycles == prev_absorbed:
                campaign = campaign_summary(self.data_root)
                try:
                    st = self._state(timeout=5)
                    idle = (int(st.get("pending_count") or 0) == 0 and int(st.get("running_count") or 0) == 0
                            and bool(st.get("supervisor_ready")))
                except (urllib.error.URLError, OSError, ValueError):
                    idle = False
                idle = idle and not request_path.exists() and not campaign["active_transaction"]
                idle_streak = idle_streak + 1 if idle else 0
                if idle_streak >= max(1, int(idle_polls)):
                    return {"absorbed": False, "new_sha": sha, "cycles": cycles,
                            "reason": absorb_idle_reason(campaign, history_at_start), "campaign": campaign}
            time.sleep(5)
        return {"absorbed": False, "new_sha": self.current_sha(), "cycles": absorbed_cycles_done(self.data_root),
                "reason": "timeout", "campaign": campaign_summary(self.data_root)}

    def stop(self) -> None:
        if self.proc is not None and self.proc.poll() is None:
            pid = self.proc.pid
            terminate_process_tree(self.proc)
            try:
                self.proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                kill_pid_tree(pid)
                try:
                    self.proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    pass

    def __enter__(self) -> "IsolatedServer":
        return self.start()

    def __exit__(self, *_exc) -> None:
        self.stop()
