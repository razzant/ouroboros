"""Host-owned separation of the owner command bot and the optional Presence bot.

Both credentials live in the Settings document; the secondary skill receives only
its own granted key. Settings writers call the guard while holding their existing
cross-process document lock. The Host Service rechecks current values before a
secondary bot starts or dispatches, covering out-of-band changes and restarts.
No digest or token is returned by this module's public status reader.
"""

from __future__ import annotations

from hmac import compare_digest
from pathlib import Path
from typing import Any, Mapping

OWNER_KEY = "TELEGRAM_BOT_TOKEN"
PRESENCE_KEY = "TELEGRAM_PRESENCE_BOT_TOKEN"


class TokenSeparationError(ValueError):
    """A settings write would allow two bot transports to share a credential."""


def status(values: Mapping[str, Any]) -> str:
    """Return distinct/collision/unavailable, never either secret or a hash."""
    owner = values.get(OWNER_KEY)
    secondary = values.get(PRESENCE_KEY)
    if not isinstance(owner, str) or not isinstance(secondary, str):
        return "unavailable"
    if not owner.strip() or not secondary.strip():
        return "unavailable"
    return "collision" if compare_digest(owner.strip(), secondary.strip()) else "distinct"


def guard_settings_write(values: Mapping[str, Any], *, owner_env: str = "") -> Mapping[str, Any]:
    """A candidate may clear BOTH tokens; otherwise equality is forbidden.

    A missing owner token cannot validate a configured secondary credential.
    This runs after Settings placeholders are removed, under the writer's lock.
    """
    comparison = dict(values)
    if not comparison.get(OWNER_KEY) and owner_env:
        comparison[OWNER_KEY] = owner_env
    if not str(comparison.get(PRESENCE_KEY) or "").strip():
        return values
    state = status(comparison)
    if state != "distinct":
        raise TokenSeparationError(
            "Separate Telegram Presence bot token is unavailable or matches the owner bot; nothing was saved."
        )
    return values


def prepare_bot_tokens(settings: dict, known_setting_keys) -> dict:
    """Strip wire masks, then guard Settings without persisting an env-only key."""
    import os

    from ouroboros.secret_masking import strip_masked_secrets

    clean = strip_masked_secrets(settings, known_setting_keys=known_setting_keys)
    return guard_settings_write(clean, owner_env=os.environ.get(OWNER_KEY, ""))


def current_status(settings_path: Path, *, candidate: str = "") -> str:
    """Read the live document, not a running task's frozen settings snapshot.

    The Settings verifier raises on an unreadable/pinned-invalid source. A missing
    owner token is unavailable; an unset secondary token cannot run. Values
    explicitly supplied by the operator environment override disk values, as
    they do in the runtime loader. This reader performs no mutation or logging.
    """
    import os

    from ouroboros.settings_integrity import read_settings_json_verified

    try:
        saved = read_settings_json_verified(settings_path)
        if not isinstance(saved, dict):
            return "unavailable"
        values = dict(saved)
        for key in (OWNER_KEY, PRESENCE_KEY):
            if not values.get(key) and os.environ.get(key):
                values[key] = os.environ[key]
        state = status(values)
        if not candidate or not isinstance(values.get(PRESENCE_KEY), str):
            return "unavailable"
        return state if compare_digest(candidate, values[PRESENCE_KEY].strip()) else "unavailable"
    except (OSError, ValueError, TypeError):
        return "unavailable"


async def api_bot_token_status(request):
    """Private Host check for the reviewed second bot before every API send.

    No token or comparison digest crosses back into the skill; the candidate
    credential itself is read only by the host's Settings reader. The caller's
    own key remains separately granted by the normal skill lifecycle.
    """
    from starlette.responses import JSONResponse

    from ouroboros.gateway.host_service import HostServiceAuthError, _authenticated, _bounded_host_read

    ctx = request.app.state.host_service_context
    try:
        caller, _stamp = await _authenticated(ctx, request.headers.get("x-skill-token", ""), "presence")
        if caller != "telegram-presence-bot":
            raise HostServiceAuthError("not the second bot")
    except HostServiceAuthError:
        return JSONResponse({"ok": False, "status": "unavailable"}, status_code=403)
    state = await _bounded_host_read(ctx, lambda: current_status(
        ctx.data_dir / "settings.json", candidate=request.headers.get("x-bot-token", "")))
    return JSONResponse({"ok": state == "distinct", "status": state},
                        status_code=200 if state == "distinct" else 409)
