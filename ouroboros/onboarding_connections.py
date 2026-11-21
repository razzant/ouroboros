"""A built-in skill's declared connection card in the setup wizard (DESIGN §7, CLIENT_CONTRACT §12).

A skill shipped with the app may declare ``connection_card`` in its ``SKILL.md``: which of its ``secret_fields`` holds
its token (``token``), a pairing link with a one-time code and the code's secret field (``pairing.link``,
``pairing.code``), the stdlib-only function that names the account a token belongs to (``pairing.identity``,
``path.py:function`` inside the skill), the card's own words and options the skill keeps in its own settings
(``store: skill_settings``), posted to the skill's declared ``route``. Nothing here knows a service. The host:

- carries the card on the skill's starter item (``starter_connection``);
- answers the card's token check by calling the declared function of a BUILT-IN skill only, never a hub skill's code
  before its review (``api_connection_identity``);
- adds the token and the code to the completion's one settings write (``completion_settings``), refusing them from
  the Telegram Mini App (``guard_completion``): arming a code names the account's owner;
- records the owner's consent in the starter marker after that write (``record_consent``) and, once the skill is
  installed, grants the card's keys through the skill lifecycle (enable after grant) and posts its options once to
  the declared route on loopback (``settle``). Every failure is recorded there; none blocks Start.
"""
from __future__ import annotations

import asyncio
import importlib.util
import json
import logging
import pathlib
import re
import threading
import time
import urllib.error
import urllib.request
from typing import Any, Callable, Dict, List, Optional, Tuple

from ouroboros.utils import utc_now_iso

log = logging.getLogger(__name__)

MINIAPP_HEADER = "X-Ouroboros-Telegram-MiniApp"  # the Mini App sidecar's mark (server_web.TELEGRAM_MINIAPP_HEADER)
_NAME = re.compile(r"[a-z0-9][a-z0-9_-]{0,63}")
_KEY = re.compile(r"[A-Z][A-Z0-9_]{1,63}")
_CODE = re.compile(r"[A-Za-z0-9_-]{1,64}")  # what a deep link's start parameter carries
_ACCOUNT = re.compile(r"[A-Za-z0-9_]{1,64}")
_FUNCTION = re.compile(r"((?:[A-Za-z0-9_-]+/)*[A-Za-z0-9_-]+\.py):([A-Za-z_][A-Za-z0-9_]*)")
_ROUTE = re.compile(r"[a-z0-9_-]+(?:/[a-z0-9_-]+)*")
_TEXT_MAX = 200
_TOKEN_MAX = 512
_IDENTITY_TIMEOUT = 8.0  # the declared function's own transport bound; the host waits a little longer
_POST_TRIES = 12  # the option waits this many beats for the skill's route to answer once it is turned on
_lock = threading.Lock()


def _text(value: Any, limit: int = _TEXT_MAX) -> str:
    return value.strip()[:limit] if isinstance(value, str) else ""


def _https(value: Any) -> str:
    text = _text(value, 500)
    return text if re.fullmatch(r"https://[^\s\"'<>]+", text) else ""


def _normalize(name: str, root: pathlib.Path, raw: Any, fields: Dict[str, str]) -> Optional[Dict[str, Any]]:
    """The declaration the host acts on, or None when any part it needs is missing or out of shape."""
    if not isinstance(raw, dict) or not isinstance(raw.get("pairing"), dict):
        return None
    pairing = raw["pairing"]
    token, code, link = _text(raw.get("token"), 64), _text(pairing.get("code"), 64), _https(pairing.get("link"))
    identity = _FUNCTION.fullmatch(_text(pairing.get("identity"), 200))
    if token not in fields or code not in fields or token == code or "{username}" not in link or "{code}" not in link:
        return None
    if identity is None or ".." in identity.group(1).split("/"):
        return None
    options = []
    for option in raw.get("options") or []:
        if not isinstance(option, dict):
            return None
        key, route = _text(option.get("key"), 64), _text(option.get("route"), 64)
        on, off = _text(option.get("on"), 32), _text(option.get("off"), 32)
        if (not _KEY.fullmatch(key) or not _ROUTE.fullmatch(route) or option.get("store") != "skill_settings"
                or not on or not off or on == off or not _text(option.get("label"))):
            return None
        options.append({"key": key, "label": _text(option["label"]), "on": on, "off": off, "route": route,
                        "new_connection": option.get("new_connection") is True})
    steps = [{"text": _text(step.get("text")), "link": _https(step.get("link")), "action": _text(step.get("action"), 40)}
             for step in raw.get("steps") or [] if isinstance(step, dict) and _text(step.get("text"))]
    return {
        "skill": name, "token": token, "code": code, "token_key": fields[token], "code_key": fields[code],
        "link": link, "identity": (root / identity.group(1), identity.group(2)),
        "title": _text(raw.get("title")), "pitch": _text(raw.get("pitch")),
        "hint": _text(pairing.get("hint")), "note": _text(pairing.get("note")),
        "steps": steps[:8], "options": options[:4],
    }


def declared_card(name: Any) -> Optional[Dict[str, Any]]:
    """The connection card the built-in skill ``name`` declares (the app's own bundled bytes), or None."""
    from ouroboros.contracts.skill_manifest import parse_skill_manifest_text
    from ouroboros.secret_requests import declared_secret_fields
    from ouroboros.skill_builtin import bundled_skills_dir

    name = str(name or "").strip()
    if not _NAME.fullmatch(name):
        return None
    root = bundled_skills_dir() / name
    try:
        text = (root / "SKILL.md").read_text(encoding="utf-8")
        if "connection_card" not in text:
            return None
        manifest = parse_skill_manifest_text(text)
        fields = declared_secret_fields(manifest)
    except Exception:  # noqa: BLE001 — an unreadable or invalid manifest declares no card
        return None
    return _normalize(name, root, (getattr(manifest, "raw_extra", None) or {}).get("connection_card"), fields)


def starter_connection(name: Any) -> Optional[Dict[str, Any]]:
    """What the setup's Skills step shows of a built-in skill's card (``StarterConnection``): never its function,
    its settings keys or its routes."""
    card = declared_card(name)
    if card is None:
        return None
    return {
        "skill": card["skill"], "token": card["token"], "title": card["title"], "pitch": card["pitch"],
        "steps": card["steps"], "pairing": {"code": card["code"], "link": card["link"], "hint": card["hint"],
                                             "note": card["note"]},
        "options": [{key: option[key] for key in ("key", "label", "on", "off", "new_connection")}
                    for option in card["options"]],
    }


def card_keys(name: Any) -> Tuple[str, ...]:
    """The settings keys a built-in skill's card writes itself (its token and code), so a prerequisite list can
    leave the code out."""
    card = declared_card(name)
    return (card["token_key"], card["code_key"]) if card else ()


# -- the token check ---------------------------------------------------------------------------------------------

_MODULES: Dict[Tuple[str, float], Any] = {}


def card_root(card: Dict[str, Any]) -> pathlib.Path:
    from ouroboros.skill_builtin import bundled_skills_dir

    return bundled_skills_dir() / card["skill"]


def _declared_function(card: Dict[str, Any]) -> Callable[..., Any]:
    """The card's declared function, loaded from its file inside the built-in skill (a link out of it is refused)."""
    root = card_root(card).resolve()
    path = pathlib.Path(card["identity"][0]).resolve()
    if not path.is_file() or not path.is_relative_to(root):
        raise LookupError("the declared identity function is not a file inside the skill")
    key = (str(path), path.stat().st_mtime)
    with _lock:
        module = _MODULES.get(key)
        if module is None:
            spec = importlib.util.spec_from_file_location(f"_connection_card_{card['skill']}", path)
            if spec is None or spec.loader is None:
                raise LookupError("the declared identity module cannot be loaded")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            _MODULES[key] = module
    return getattr(module, card["identity"][1])


def check_identity(card: Dict[str, Any], token: str) -> Dict[str, Any]:
    """``{"ok": True, "username"}`` or ``{"ok": False, "cause": "refused" | "unreachable"}`` as the skill's declared
    function answers; raises on anything else (the caller answers that the check failed, naming nothing)."""
    answer = _declared_function(card)(token, timeout=_IDENTITY_TIMEOUT)
    if isinstance(answer, dict) and answer.get("ok") is True and _ACCOUNT.fullmatch(str(answer.get("username") or "")):
        return {"ok": True, "username": str(answer["username"])}
    if isinstance(answer, dict) and answer.get("ok") is False and answer.get("cause") in ("refused", "unreachable"):
        return {"ok": False, "cause": answer["cause"]}
    raise ValueError("the declared identity function gave an answer of no known shape")


async def api_connection_identity(request: Any) -> Any:
    """``POST /api/onboarding/connection/identity`` ``{skill, token}``: which account a token names, from the declared
    function of a built-in skill. Its errors never carry the token or the address the function called."""
    from starlette.responses import JSONResponse

    from ouroboros.gateway._helpers import request_json_or

    body = await request_json_or(request, None)
    if not isinstance(body, dict):
        return JSONResponse({"ok": False, "error": "the body must be a JSON object"}, status_code=400)
    card = declared_card(body.get("skill"))
    if card is None:
        return JSONResponse({"ok": False, "error": "no built-in skill declares a connection card by that name"},
                            status_code=404)
    token = body.get("token")
    if not isinstance(token, str) or not token.strip() or len(token) > _TOKEN_MAX:
        return JSONResponse({"ok": False, "cause": "refused"})
    try:
        answer = await asyncio.wait_for(asyncio.to_thread(check_identity, card, token.strip()), _IDENTITY_TIMEOUT + 4)
    except asyncio.TimeoutError:
        return JSONResponse({"ok": False, "cause": "unreachable"})
    except Exception:  # noqa: BLE001 — logged without the token or its exception text
        log.warning("The connection card's identity check of %s failed", card["skill"])
        return JSONResponse({"ok": False, "error": "the identity check failed"}, status_code=500)
    return JSONResponse(answer)


# -- the completion ----------------------------------------------------------------------------------------------

def parse_connections(raw: Any) -> List[Tuple[Dict[str, Any], str, str, Dict[str, str]]]:
    """The completion's ``connections`` ({skill: {fields: {token field, code field}, options: {key: value}}}) as
    (card, token, code, options), each skill a built-in one that declares a card; raises ValueError on anything else
    so the completion saves nothing."""
    if raw in (None, {}):
        return []
    if not isinstance(raw, dict) or len(raw) > 4:
        raise ValueError("connections must map built-in skill names to their card's values.")
    parsed = []
    for name, value in raw.items():
        card = declared_card(name)
        if card is None:
            raise ValueError(f"no built-in skill named {str(name)[:64]!r} declares a connection card.")
        fields = value.get("fields") if isinstance(value, dict) else None
        options = value.get("options", {}) if isinstance(value, dict) else None
        if not isinstance(fields, dict) or set(fields) != {card["token"], card["code"]} or not isinstance(options, dict):
            raise ValueError(f"connections.{card['skill']} must carry exactly its {card['token']} and {card['code']}.")
        token, code = fields[card["token"]], fields[card["code"]]
        if not isinstance(token, str) or not token.strip() or len(token) > _TOKEN_MAX or any(c.isspace() for c in token.strip()):
            raise ValueError(f"connections.{card['skill']}.{card['token']} must be a token.")
        if not isinstance(code, str) or not _CODE.fullmatch(code):
            raise ValueError(f"connections.{card['skill']}.{card['code']} must be 1-64 letters, digits, _ or -.")
        declared = {option["key"]: option for option in card["options"]}
        if any(key not in declared or value not in (declared[key]["on"], declared[key]["off"])
               for key, value in options.items()):
            raise ValueError(f"connections.{card['skill']}.options names only its declared options and their values.")
        parsed.append((card, token.strip(), code, {str(key): str(value) for key, value in options.items()}))
    return parsed


def completion_settings(body: Dict[str, Any]) -> Dict[str, str]:
    """The settings the completion's connection cards add to its one write: each card's token and code under the
    keys its skill declares. The options stay out: they are the skill's own settings (``settle`` posts them)."""
    return {key: value for card, token, code, _options in parse_connections(body.get("connections"))
            for key, value in ((card["token_key"], token), (card["code_key"], code))}


def guard_completion(endpoint: Callable[[Any], Any]) -> Callable[[Any], Any]:
    """The completion route, refusing a connection card's values from the Telegram Mini App: a pairing code names the
    account's owner, the power its own settings refuse there (``request_may_change_owner``)."""
    async def completion(request: Any) -> Any:
        if str(request.headers.get(MINIAPP_HEADER, "")).strip():
            from ouroboros.gateway._helpers import request_json_or
            from ouroboros.gateway.owner_settings import unsaved_error

            body = await request_json_or(request, {})
            if isinstance(body, dict) and body.get("connections"):
                return unsaved_error("A connection card's token and code are not accepted from the Telegram Mini App; "
                                     "finish setup in Ouroboros itself.", 403, code="connection_owner_only")
        return await endpoint(request)
    return completion


# -- after the write: the owner's consent, the grant, the options --------------------------------------------------

def record_consent(drive_root: pathlib.Path, raw: Any) -> List[Dict[str, Any]]:
    """Note each card the completion saved in the starter marker (``connections``) for ``settle``; never raises."""
    from ouroboros.skill_starter import read_marker, write_marker

    try:
        records = [{"skill": card["skill"], "keys": [card["token_key"], card["code_key"]], "options": options,
                    "at": utc_now_iso(), "grant": "pending", "options_state": "pending" if options else "none"}
                   for card, _token, _code, options in parse_connections(raw)]
        if records:
            kept = [old for old in read_marker(drive_root).get("connections") or []
                    if old.get("skill") not in {record["skill"] for record in records}]
            write_marker(drive_root, connections=kept + records)
        return records
    except Exception:  # noqa: BLE001
        log.warning("The setup's connection card consent could not be recorded", exc_info=True)
        return []


def _installed(drive_root: pathlib.Path, name: str) -> bool:
    from ouroboros.config import get_skills_repo_path
    from ouroboros.skill_loader import find_skill

    skill = find_skill(drive_root, name, repo_path=get_skills_repo_path() or None)
    return skill is not None and not getattr(skill, "load_error", "")


def _grant(drive_root: pathlib.Path, repo_dir: Optional[pathlib.Path], record: Dict[str, Any]) -> str:
    from ouroboros.config import REPO_DIR, get_skills_repo_path
    from ouroboros.gateway.extensions import _ApiReviewCtx
    from ouroboros.skill_lifecycle_actions import run_skill_action

    ctx = _ApiReviewCtx(pathlib.Path(drive_root), pathlib.Path(repo_dir or REPO_DIR))
    ctx.task_id = ""
    ctx._skill_owner_client_host = "setup"  # the owner's consent was given at the setup's Finish
    result = run_skill_action(ctx, record["skill"], "grant", items=list(record["keys"]),
                              repo_path=get_skills_repo_path(), _owner_actor="owner_ui", _enable_after_grant=True)
    return "granted" if not result.get("error") else f"failed: {str(result.get('error'))[:200]}"


def _server_port() -> int:
    import os

    from ouroboros.config import AGENT_SERVER_PORT, resolve_port_file

    try:
        port = int(resolve_port_file().read_text(encoding="utf-8").strip())
        if port > 0:
            return port
    except (OSError, ValueError):
        pass
    return int(os.environ.get("OUROBOROS_SERVER_PORT") or AGENT_SERVER_PORT)


def _post_options(record: Dict[str, Any], *, post: Optional[Callable[[str, Dict[str, str]], int]] = None,
                  pause: Callable[[float], None] = time.sleep) -> str:
    """The one POST of the card's options to the skill's declared route on loopback, waiting (bounded) for the route
    to answer once the skill runs: «posted», or why not."""
    card = declared_card(record["skill"])
    if card is None:
        return "failed: the skill no longer declares its card"
    routes = {option["route"] for option in card["options"] if option["key"] in record["options"]}
    if len(routes) != 1:
        return "failed: the options do not share one declared route"
    path = f"/api/extensions/{card['skill']}/{routes.pop()}"
    send = post or _loopback_post
    status = 0
    for attempt in range(_POST_TRIES):
        status = send(path, dict(record["options"]))
        if 200 <= status < 300:
            return "posted"
        if status not in (0, 404, 409, 502, 503):
            break
        pause(min(30.0, 2.0 * (attempt + 1)))
    return f"failed: the route answered {status or 'nothing'}"


def _loopback_post(path: str, body: Dict[str, str]) -> int:
    request = urllib.request.Request(f"http://127.0.0.1:{_server_port()}{path}", data=json.dumps(body).encode("utf-8"),
                                     method="POST", headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=30) as answer:
            return int(answer.status)
    except urllib.error.HTTPError as answer:
        return int(answer.code)
    except Exception:  # noqa: BLE001 — not answering yet
        return 0


def settle(drive_root: pathlib.Path, repo_dir: Optional[pathlib.Path] = None, *,
           post: Optional[Callable[[str, Dict[str, str]], int]] = None,
           pause: Callable[[float], None] = time.sleep) -> List[Dict[str, Any]]:
    """For each recorded card whose skill is installed: grant its keys (enable after grant), then post its options
    once. Results land in the marker; a skill not installed yet waits for the next run. Never raises."""
    from ouroboros.skill_starter import read_marker, write_marker

    try:
        records = [dict(record) for record in read_marker(drive_root).get("connections") or [] if isinstance(record, dict)]
        for record in records:
            if record.get("grant") != "pending" and record.get("options_state") != "pending":
                continue
            if not _installed(drive_root, str(record.get("skill") or "")):
                continue
            if record.get("grant") == "pending":
                record["grant"] = _grant(drive_root, repo_dir, record)
                write_marker(drive_root, connections=records)
            if record.get("options_state") == "pending" and record["grant"] == "granted":
                record["options_state"] = _post_options(record, post=post, pause=pause)
                write_marker(drive_root, connections=records)
        return records
    except Exception:  # noqa: BLE001
        log.warning("The setup's connection cards could not be settled", exc_info=True)
        return []


def settle_later(drive_root: pathlib.Path, repo_dir: Optional[pathlib.Path] = None) -> None:
    """``settle`` in its own background thread, when the marker holds a pending card."""
    from ouroboros.skill_starter import read_marker

    if any(isinstance(record, dict) and "pending" in (record.get("grant"), record.get("options_state"))
           for record in read_marker(drive_root).get("connections") or []):
        threading.Thread(target=settle, args=(drive_root, repo_dir), daemon=True, name="connection-cards").start()


__all__ = [
    "api_connection_identity", "card_keys", "completion_settings", "declared_card", "guard_completion",
    "parse_connections", "record_consent", "settle", "settle_later", "starter_connection",
]
