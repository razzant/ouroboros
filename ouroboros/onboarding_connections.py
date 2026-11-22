"""A built-in skill's declared connection card in the setup wizard (DESIGN §7, CLIENT_CONTRACT §12).

A skill shipped with the app may declare ``connection_card`` in its ``SKILL.md``: which of its ``secret_fields`` holds
its token (``token``), a pairing link with a one-time code (``pairing.link``), where the skill keeps that code
(``pairing.code``: ``{key, store: skill_settings, route}``: the skill's own settings, never the core's), the
stdlib-only function that names the account a token belongs to (``pairing.identity``, ``path.py:function`` inside the
skill) and, optionally, the one that names the server it should ask (``pairing.api_base``, given the skill's state
directory), the card's own words and options the skill keeps in its own settings too. Nothing here knows a service. The
host:

- carries the card on the skill's starter item (``starter_connection``), saying where the install grants nothing on
  its own, so the skill starts only once the owner allows it in Extensions;
- answers the card's token check by calling the declared function of a BUILT-IN skill only, never a hub skill's code
  before its review (``api_connection_identity``);
- adds the token alone to the completion's one settings write (``completion_settings``); the completion's route
  (``guard_completion``) refuses a card's values from the Telegram Mini App, since a code names the account's owner,
  and once the write is saved records the owner's consent (the code and the options included) in the starter marker
  (``record_consent``); the code is never written to Settings or a log. A card carries its code and options only when
  the wizard drew the QR and the link that hold that code; without them the token comes alone and nothing is posted;
- makes ONE post of every skill-settings value of the card (the code with the options) to the route they declare,
  from this server over loopback: the owner's local session as the route's guard reads it, reachable only from that
  recorded consent (``settle``, one run at a time). Where reviewed skills are granted automatically, that policy
  covers the owner's consent too: a skill installed but not granted and on (one its own install never granted, such
  as a payload a Reset kept) gets the lifecycle's grant of what it asks, enable after grant (``_turn_on``); where they
  are not, the card grants nothing and the owner allows the skill in Extensions. It runs on events that already
  happen, never on a timer: the starter install's end, a saved completion
  (a skill already installed), the skill's extension finishing its load, and the next start. A route not serving yet
  leaves the card waiting for the next of them. The code is dropped from the record once posted or refused. Every
  outcome is recorded there; none blocks Start.
"""
from __future__ import annotations

import asyncio
import importlib.util
import json
import logging
import pathlib
import re
import threading
import urllib.error
import urllib.request
from typing import Any, Callable, Dict, List, Optional, Tuple

from ouroboros.utils import utc_now_iso

log = logging.getLogger(__name__)

MINIAPP_HEADER = "X-Ouroboros-Telegram-MiniApp"  # the Mini App sidecar's mark (server_web.TELEGRAM_MINIAPP_HEADER)
_NAME = re.compile(r"[a-z0-9][a-z0-9_-]{0,63}")
_KEY = re.compile(r"[A-Z][A-Z0-9_]{1,63}")
_CODE = re.compile(r"[A-Za-z0-9_-]{22,64}")  # a minted pairing code (a start parameter holds up to 64)
_ACCOUNT = re.compile(r"[A-Za-z0-9_]{1,64}")
_FUNCTION = re.compile(r"((?:[A-Za-z0-9_-]+/)*[A-Za-z0-9_-]+\.py):([A-Za-z_][A-Za-z0-9_]*)")
_ROUTE = re.compile(r"[a-z0-9_-]+(?:/[a-z0-9_-]+)*")
_TEXT_MAX = 200
_TOKEN_MAX = 512
_IDENTITY_TIMEOUT = 8.0  # the declared function's own transport bound; the host waits a little longer
_IDENTITY_WAIT = _IDENTITY_TIMEOUT + 4  # the host's watchdog over a whole check, its own worker's wait included
_NOT_LIVE = (0, 404, 409, 502, 503)  # the skill's route is not serving yet (or its settings are busy): wait for an event
_lock = threading.Lock()
_SETTLE = threading.Lock()  # one settle at a time: the starter's run, a completion's and a start's never post twice


def _text(value: Any, limit: int = _TEXT_MAX) -> str:
    return value.strip()[:limit] if isinstance(value, str) else ""


def _https(value: Any) -> str:
    text = _text(value, 500)
    return text if re.fullmatch(r"https://[^\s\"'<>]+", text) else ""


def _skill_setting(raw: Any) -> Optional[Dict[str, str]]:
    """A value the skill keeps in its own settings: ``{key, route}`` when ``store`` says so, else None."""
    if not isinstance(raw, dict) or raw.get("store") != "skill_settings":
        return None
    key, route = _text(raw.get("key"), 64), _text(raw.get("route"), 64)
    return {"key": key, "route": route} if _KEY.fullmatch(key) and _ROUTE.fullmatch(route) else None


def _reference(value: Any) -> Optional[Tuple[str, str]]:
    """``path.py:function`` inside the skill as (path, function), or None when out of shape or out of the skill."""
    found = _FUNCTION.fullmatch(_text(value, 200))
    return (found.group(1), found.group(2)) if found and ".." not in found.group(1).split("/") else None


def _normalize(name: str, root: pathlib.Path, raw: Any, fields: Dict[str, str]) -> Optional[Dict[str, Any]]:
    """The declaration the host acts on, or None when any part it needs is missing or out of shape."""
    if not isinstance(raw, dict) or not isinstance(raw.get("pairing"), dict):
        return None
    pairing = raw["pairing"]
    token, link, code = _text(raw.get("token"), 64), _https(pairing.get("link")), _skill_setting(pairing.get("code"))
    identity = _reference(pairing.get("identity"))
    base = _reference(pairing["api_base"]) if pairing.get("api_base") is not None else ()  # optional; never guessed
    if token not in fields or code is None or "{username}" not in link or "{code}" not in link:
        return None
    if identity is None or base is None:
        return None
    options = []
    for option in raw.get("options") or []:
        stored = _skill_setting(option)
        if stored is None:
            return None
        on, off = _text(option.get("on"), 32), _text(option.get("off"), 32)
        if not on or not off or on == off or not _text(option.get("label")) or stored["key"] == code["key"]:
            return None
        options.append({**stored, "label": _text(option["label"]), "on": on, "off": off,
                        "new_connection": option.get("new_connection") is True})
    steps = [{"text": _text(step.get("text")), "link": _https(step.get("link")), "action": _text(step.get("action"), 40)}
             for step in raw.get("steps") or [] if isinstance(step, dict) and _text(step.get("text"))]
    return {
        "skill": name, "token": token, "token_key": fields[token], "code": code, "link": link,
        "identity": (root / identity[0], identity[1]), "api_base": (root / base[0], base[1]) if base else None,
        "title": _text(raw.get("title")), "pitch": _text(raw.get("pitch")),
        "hint": _text(pairing.get("hint")), "note": _text(pairing.get("note")),
        "steps": steps[:8], "options": options[:4],
    }


def declared_card(name: Any) -> Optional[Dict[str, Any]]:
    """The connection card the built-in skill ``name`` declares (the app's own bundled bytes), or None."""
    from ouroboros.secret_requests import declared_secret_fields
    from ouroboros.skill_builtin import bundled_skills_dir
    from ouroboros.skill_source_catalog import read_manifest

    name = str(name or "").strip()
    if not _NAME.fullmatch(name):
        return None
    root = bundled_skills_dir() / name
    try:
        manifest = read_manifest(root)  # None for an unreadable or invalid manifest: no card
        card = (getattr(manifest, "raw_extra", None) or {}).get("connection_card")
        return _normalize(name, root, card, declared_secret_fields(manifest)) if card else None
    except Exception:  # noqa: BLE001 — a card out of shape declares no card
        return None


def starter_connection(name: Any) -> Optional[Dict[str, Any]]:
    """What the setup's Skills step shows of a built-in skill's card (``StarterConnection``): never its functions,
    its keys or its routes. ``needs_allow``: this install grants reviewed skills nothing on its own, so the skill
    starts only once the owner allows it in Extensions (the card says so where it states its outcome)."""
    from ouroboros.config import get_auto_grant_enabled

    card = declared_card(name)
    if card is None:
        return None
    return {
        "skill": card["skill"], "token": card["token"], "title": card["title"], "pitch": card["pitch"],
        "steps": card["steps"], "pairing": {"link": card["link"], "hint": card["hint"], "note": card["note"]},
        "options": [{key: option[key] for key in ("key", "label", "on", "off", "new_connection")}
                    for option in card["options"]],
        "needs_allow": not get_auto_grant_enabled(),
    }


def card_keys(name: Any) -> Tuple[str, ...]:
    """The keys a built-in skill's card fills itself (its token, its code), which a prerequisite list leaves out."""
    card = declared_card(name)
    return (card["token_key"], card["code"]["key"]) if card else ()


# -- the token check ---------------------------------------------------------------------------------------------

_MODULES: Dict[Tuple[str, float], Any] = {}


def card_root(card: Dict[str, Any]) -> pathlib.Path:
    from ouroboros.skill_builtin import bundled_skills_dir

    return bundled_skills_dir() / card["skill"]


def _declared_function(card: Dict[str, Any], which: str = "identity") -> Callable[..., Any]:
    """A function the card declares (``identity`` or ``api_base``), loaded from its file inside the built-in skill (a
    link out of it is refused)."""
    root = card_root(card).resolve()
    path = pathlib.Path(card[which][0]).resolve()
    if not path.is_file() or not path.is_relative_to(root):
        raise LookupError(f"the declared {which} function is not a file inside the skill")
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
    return getattr(module, card[which][1])


def check_identity(card: Dict[str, Any], token: str, state_dir: Optional[pathlib.Path] = None) -> Dict[str, Any]:
    """``{"ok": True, "username"}`` or ``{"ok": False, "cause": "refused" | "unreachable"}`` as the skill's declared
    function answers; raises on anything else (the caller answers that the check failed, naming nothing). A card that
    declares ``api_base`` names the server first, from the skill's state directory, and the check asks that one: a
    reader that fails asks nobody."""
    identity = _declared_function(card)
    base = (_declared_function(card, "api_base")(state_dir),) if card.get("api_base") else ()
    answer = identity(token, *base, timeout=_IDENTITY_TIMEOUT)
    if isinstance(answer, dict) and answer.get("ok") is True and _ACCOUNT.fullmatch(str(answer.get("username") or "")):
        return {"ok": True, "username": str(answer["username"])}
    if isinstance(answer, dict) and answer.get("ok") is False and answer.get("cause") in ("refused", "unreachable"):
        return {"ok": False, "cause": answer["cause"]}
    raise ValueError("the declared identity function gave an answer of no known shape")


async def api_connection_identity(request: Any) -> Any:
    """``POST /api/onboarding/connection/identity`` ``{skill, token}``: which account a token names, from the declared
    function of a built-in skill. Its errors never carry the token or the address the function called."""
    from starlette.responses import JSONResponse

    from ouroboros.gateway._helpers import request_drive_root, request_json_or
    from ouroboros.skill_loader import skill_state_dir_path

    body = await request_json_or(request, None)
    if not isinstance(body, dict):
        return JSONResponse({"ok": False, "error": "the body must be a JSON object"}, status_code=400)
    card = declared_card(body.get("skill"))
    if card is None:
        return JSONResponse({"ok": False, "error": "no built-in skill declares a connection card by that name"},
                            status_code=404)
    token = body.get("token")
    if not _token_shaped(token):  # this host's refusal, never the service's («refused» is the service's own answer)
        return JSONResponse({"ok": False, "error": f"a token is 1 to {_TOKEN_MAX} characters with no spaces inside"},
                            status_code=400)
    state_dir = skill_state_dir_path(request_drive_root(request), card["skill"])  # read, never made, by the reader
    try:
        answer = await asyncio.wait_for(asyncio.to_thread(check_identity, card, token.strip(), state_dir),
                                        _IDENTITY_WAIT)
    except asyncio.TimeoutError:
        # The host's deadline says only that the check did not finish: it has no evidence that a request reached the
        # service, so this is never «unreachable» (that is the declared function's own answer, passed through above).
        return JSONResponse({"ok": False, "error": "the identity check did not finish in time"}, status_code=503)
    except Exception:  # noqa: BLE001 — logged without the token or its exception text
        log.warning("The connection card's identity check of %s failed", card["skill"])
        return JSONResponse({"ok": False, "error": "the identity check failed"}, status_code=500)
    return JSONResponse(answer)


def _token_shaped(token: Any) -> bool:
    """Whether ``token`` could be a token at all: 1 to ``_TOKEN_MAX`` characters with no space or line break inside."""
    return (isinstance(token, str) and bool(token.strip()) and len(token) <= _TOKEN_MAX
            and not any(c.isspace() for c in token.strip()))


# -- the completion ----------------------------------------------------------------------------------------------

def parse_connections(raw: Any) -> List[Tuple[Dict[str, Any], str, str, Dict[str, str]]]:
    """The completion's ``connections`` ({skill: {fields: {token field: token}, code, options: {key: value}}}) as
    (card, token, code, options), each skill a built-in one that declares a card; raises ValueError on anything else
    so the completion saves nothing. No message names a value. The code and the options come together or not at all:
    a card whose QR was never drawn sends its token alone (code "")."""
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
        code = value.get("code") if isinstance(value, dict) else None
        if not isinstance(fields, dict) or set(fields) != {card["token"]} or not isinstance(options, dict):
            raise ValueError(f"connections.{card['skill']} must carry exactly its {card['token']}.")
        token = fields[card["token"]]
        if not _token_shaped(token):
            raise ValueError(f"connections.{card['skill']}.{card['token']} must be a token.")
        if code is not None and (not isinstance(code, str) or not _CODE.fullmatch(code)):
            raise ValueError(f"connections.{card['skill']}.code must be a minted pairing code (22-64 letters, digits, _ or -).")
        if code is None and options:
            raise ValueError(f"connections.{card['skill']}.options come only with its pairing code.")
        declared = {option["key"]: option for option in card["options"]}
        if any(key not in declared or value not in (declared[key]["on"], declared[key]["off"])
               for key, value in options.items()):
            raise ValueError(f"connections.{card['skill']}.options names only its declared options and their values.")
        parsed.append((card, token.strip(), code or "", {str(key): str(value) for key, value in options.items()}))
    return parsed


def completion_settings(body: Dict[str, Any]) -> Dict[str, str]:
    """What a completion's connection cards add to its one settings write: each card's token, under the key its skill
    declares. The code and the options are the skill's own settings (``settle`` posts them), never the core's."""
    return {card["token_key"]: token for card, token, _code, _options in parse_connections(body.get("connections"))}


def guard_completion(endpoint: Callable[[Any], Any]) -> Callable[[Any], Any]:
    """The completion route with its connection cards: their values are refused from the Telegram Mini App before
    anything is written (a pairing code names the account's owner, the power the skill's own settings refuse there:
    ``request_may_change_owner``), and once a completion that carried them is saved, the owner's consent is recorded
    and the cards settle in the background."""
    async def completion(request: Any) -> Any:
        from ouroboros.gateway._helpers import request_drive_root, request_json_or

        body = await request_json_or(request, {})
        cards = body.get("connections") if isinstance(body, dict) else None
        if cards and str(request.headers.get(MINIAPP_HEADER, "")).strip():
            from ouroboros.gateway.owner_settings import unsaved_error

            return unsaved_error("A connection card's token and code are not accepted from the Telegram Mini App; "
                                 "finish setup in Ouroboros itself.", 403, code="connection_owner_only")
        response = await endpoint(request)
        if cards and _saved(response):
            drive_root = request_drive_root(request)
            if await asyncio.to_thread(record_consent, drive_root, cards):
                settle_later(drive_root)  # an installed skill settles now; a starter run still installing settles after it
        return response
    return completion


def _saved(response: Any) -> bool:
    """Whether a completion answer says its write landed (success, or saved before a later step failed)."""
    try:
        payload = json.loads(bytes(getattr(response, "body", b"") or b"{}"))
    except ValueError:
        return False
    return isinstance(payload, dict) and (payload.get("saved") is True or (payload.get("ok") is True and
                                                                            getattr(response, "status_code", 0) == 200))


# -- after the write: the owner's consent and the one post ----------------------------------------------------------

def record_consent(drive_root: pathlib.Path, raw: Any) -> List[Dict[str, Any]]:
    """Note each card the completion saved in the starter marker (``connections``) for ``settle``: the code and the
    options it will post, until it has (a card that sent its token alone records neither); never raises, never logs a
    value."""
    from ouroboros.skill_starter import read_marker, write_marker

    try:
        records = [{"skill": card["skill"], "token_key": card["token_key"], **({"code": code} if code else {}),
                    "options": options, "at": utc_now_iso(), "post": "pending"}
                   for card, _token, code, options in parse_connections(raw)]
        if records:
            kept = [old for old in read_marker(drive_root).get("connections") or []
                    if old.get("skill") not in {record["skill"] for record in records}]
            write_marker(drive_root, connections=kept + records)
        return records
    except Exception as exc:  # noqa: BLE001
        log.warning("The setup's connection card consent could not be recorded (%s)", type(exc).__name__)
        return []


def _installed(drive_root: pathlib.Path, name: str) -> bool:
    """Whether the built-in the card belongs to is installed: a skill of that name in the built-in bucket (another
    skill that took the name elsewhere, e.g. one a person added, never receives the card's consent)."""
    from ouroboros.config import get_skills_repo_path
    from ouroboros.skill_loader import find_skill

    skill = find_skill(drive_root, name, repo_path=get_skills_repo_path() or None)
    return skill is not None and not getattr(skill, "load_error", "") and getattr(skill, "location", "") == "native"


def _turn_on(drive_root: pathlib.Path, name: str) -> str:
    """Where the install grants reviewed skills on its own, the owner's consent turns on an installed skill that is not
    yet granted and on: the lifecycle's grant of every key and permission it asks, enable after grant (the install's
    own policy, applied to the consent given at Finish), or its enable when it asks for nothing to grant. "" when
    nothing was done: the skill is already granted and on, or the install grants nothing on its own (the owner allows
    the skill in Extensions); else what happened."""
    from ouroboros.config import REPO_DIR, get_auto_grant_enabled, get_skills_repo_path
    from ouroboros.skill_loader import find_skill, grant_status_for_skill, load_enabled

    if not get_auto_grant_enabled():
        return ""
    skill = find_skill(drive_root, name, repo_path=get_skills_repo_path() or None)
    if skill is None or getattr(skill, "load_error", ""):
        return ""
    status = grant_status_for_skill(drive_root, skill)
    if status["all_granted"] and load_enabled(drive_root, name):
        return ""
    from ouroboros.gateway.extensions import _ApiReviewCtx
    from ouroboros.skill_lifecycle_actions import run_skill_action

    ctx = _ApiReviewCtx(pathlib.Path(drive_root), pathlib.Path(REPO_DIR))
    ctx.task_id = ""
    ctx._skill_owner_client_host = "setup"  # the owner's consent was given at the setup's Finish
    items = [*status["requested_keys"], *status["requested_permissions"]]
    result = run_skill_action(ctx, name, "grant" if items else "enable", items=items or None,
                              repo_path=get_skills_repo_path(), _owner_actor="owner_ui", _enable_after_grant=bool(items))
    if not result.get("ok"):
        return f"failed: {str(result.get('error') or 'refused')[:200]}"
    return "granted" if result.get("enabled") else f"granted, not on: {str(result.get('setup_error') or '')[:200]}"


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


def _bodies(card: Dict[str, Any], record: Dict[str, Any]) -> Dict[str, Dict[str, str]]:
    """Every skill-settings value of the card by the route it declares: the code and the chosen options together
    (none for a card that sent its token alone)."""
    bodies: Dict[str, Dict[str, str]] = {card["code"]["route"]: {card["code"]["key"]: str(record["code"])}} \
        if record.get("code") else {}
    for option in card["options"]:
        if option["key"] in record.get("options", {}):
            bodies.setdefault(option["route"], {})[option["key"]] = record["options"][option["key"]]
    return bodies


Answer = Tuple[int, Dict[str, Any]]  # a route's HTTP status and its JSON object ({} when it sent none)


def _post(record: Dict[str, Any], *, post: Optional[Callable[[str, Dict[str, str]], Answer]] = None) -> str:
    """ONE post per declared route of the card's skill-settings values: «posted», «none» (the card sent its token
    alone: nothing to post), «waiting» (the route does not serve yet: the next event tries again), or «failed: …» (the
    route refused it, or answered that it left the owner's values unchanged: ``owner_ignored``, which saved nothing of
    them)."""
    card = declared_card(record["skill"])
    if card is None:
        return "failed: the skill no longer declares its card"
    send = post or _loopback_post
    bodies = _bodies(card, record)
    if not bodies:
        return "none"
    for route, body in bodies.items():
        status, answer = send(f"/api/extensions/{card['skill']}/{route}", body)
        if status in _NOT_LIVE:
            return "waiting"
        if not 200 <= status < 300:
            return f"failed: the route answered {status}"
        if answer.get("owner_ignored") is True:
            return "failed: the route left the owner's values unchanged (owner_ignored)"
    return "posted"


def _loopback_post(path: str, body: Dict[str, str]) -> Answer:
    """This server's own request to a skill route: from 127.0.0.1, with no Origin and no Mini App mark, so the route's
    guard reads the owner's local session. Nothing reaches this but ``settle``, from a consent the owner gave at a
    Finish the Mini App could not send; no route lets a caller choose what it posts. Call it off the event loop."""
    request = urllib.request.Request(f"http://127.0.0.1:{_server_port()}{path}", data=json.dumps(body).encode("utf-8"),
                                     method="POST", headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=30) as answer:
            status, raw = int(answer.status), answer.read(65536)
    except urllib.error.HTTPError as answer:
        return int(answer.code), {}
    except Exception:  # noqa: BLE001 — not answering yet
        return 0, {}
    try:
        payload = json.loads(raw.decode("utf-8"))
    except ValueError:
        payload = {}
    return status, payload if isinstance(payload, dict) else {}


def settle(drive_root: pathlib.Path, *,
           post: Optional[Callable[[str, Dict[str, str]], Answer]] = None) -> List[Dict[str, Any]]:
    """For each recorded card whose skill is installed: its turn-on where the install grants on its own
    (``_turn_on``), then its one post. The outcomes land in the marker and the code leaves it once posted or refused;
    a skill not installed yet, or a route not serving yet (a skill the owner has still to allow among them), waits for
    the next event. Never raises; call it off the server's event loop."""
    from ouroboros.skill_starter import read_marker, write_marker

    try:
        with _SETTLE:
            return _settle_records(drive_root, read_marker, write_marker, post=post)
    except Exception as exc:  # noqa: BLE001
        log.warning("The setup's connection cards could not be settled (%s)", type(exc).__name__)
        return []


def _settle_records(drive_root: pathlib.Path, read_marker: Callable[..., Any], write_marker: Callable[..., Any], *,
                    post: Optional[Callable[[str, Dict[str, str]], Answer]]) -> List[Dict[str, Any]]:
    records = [dict(record) for record in read_marker(drive_root).get("connections") or [] if isinstance(record, dict)]
    for record in records:
        if record.get("post") not in ("pending", "waiting") or not _installed(drive_root, str(record.get("skill") or "")):
            continue
        turned = _turn_on(drive_root, str(record["skill"]))
        if turned:
            record["grant"] = turned
            write_marker(drive_root, connections=records)
        record["post"] = _post(record, post=post)
        if record["post"] != "waiting":
            record.pop("code", None)  # posted, or refused by the route: the code leaves the host's record
        write_marker(drive_root, connections=records)
    return records


def settle_later(drive_root: pathlib.Path) -> None:
    """``settle`` in its own background thread, when the marker holds a card still to post."""
    from ouroboros.skill_starter import read_marker

    if any(isinstance(record, dict) and record.get("post") in ("pending", "waiting")
           for record in read_marker(drive_root).get("connections") or []):
        threading.Thread(target=settle, args=(drive_root,), daemon=True, name="connection-cards").start()


__all__ = [
    "api_connection_identity", "card_keys", "completion_settings", "declared_card", "guard_completion",
    "parse_connections", "record_consent", "settle", "settle_later", "starter_connection",
]
