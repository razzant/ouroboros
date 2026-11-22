"""The onboarding page route: one wizard host for every shell.

``GET /onboarding`` serves the selected complete setup page, else the builtin
wizard. ``?builtin=1`` explicitly opens recovery. Three consumers share it:

* the desktop launcher, which opens this URL in its setup window AFTER the
  managed gateway is healthy (there is no pre-server wizard any more);
* the blocking overlay of an already-running but unconfigured app, which
  frames this URL instead of an inlined ``srcdoc`` document;
* an ordinary browser / Docker / Linux-fallback owner, who can open it directly.

The route is deliberately SIDE-EFFECT-FREE. Provider-default normalization is
computed for display only and never persisted: on a genuinely fresh install the
first bytes of ``settings.json`` must be the owner's own onboarding save, which
is exactly what the fresh-install proofs (safety-coverage authorship, install
presets) are gated on.

``GET /api/onboarding/bootstrap`` is the same derivation as JSON for a client
that renders its own setup: the page's embedded bootstrap, byte for byte, plus
the readiness ``GET /api/onboarding`` answers with its status. Completing setup
stays ``POST /api/onboarding/complete``, the one atomic writer.

``GET /api/onboarding/complete?attempt=<id>`` reads back ONE completion attempt
whose answer the client lost. Readiness cannot say that: an older provider
already passes it. The settings document names the attempt its completion write
came from, and this process remembers how the runs it served ended
(``recorded_attempt``, around the writer's own commit boundary).
"""

from __future__ import annotations

import functools
import json
import threading
from collections import OrderedDict
from typing import Any, Callable, Dict

from starlette.requests import Request
from starlette.responses import HTMLResponse, JSONResponse

from ouroboros.config import load_settings
from ouroboros.gateway._helpers import json_error
from ouroboros.gateway.owner_settings import (
    CommitBoundary,
    SettingsDocumentBusy,
    _owner_settings_document,
    unsaved_error,
)
from ouroboros.onboarding_wizard import render_onboarding_html
from ouroboros.server_runtime import apply_runtime_provider_defaults, has_startup_ready_provider
from ouroboros.settings_setup_contract import (
    ATTEMPT_ID_FIELD,
    ONBOARDING_ATTEMPT_KEY,
    build_setup_bootstrap,
    parse_completion_attempt,
)

_NO_STORE = {"Cache-Control": "no-store"}


def _consciousness_control(request: Request) -> "tuple[bool | None, bool]":
    """``(enabled, first_state)``: the consciousness control as the state shows it, read without writing, and whether
    the next boot mints a first state (``state_initialization.first_state_pending``), the one case setup offers its
    Consciousness step. A state that exists but cannot be read, or whose control is unknown, is ``(None, False)``."""
    try:
        from ouroboros.gateway._helpers import request_drive_root
        from supervisor.state import control_value, read_state
        from supervisor.state_initialization import first_state_pending

        root = request_drive_root(request)
        if first_state_pending(root):  # before reading a copy a pending owner Reset sets aside
            return None, True
        known, value = control_value(read_state(root).projection(), "bg_consciousness_enabled")
        return (bool(value) if known else None), False
    except Exception:
        return None, False


def _live_setup(request: Request) -> tuple[dict, dict]:
    """The install's settings view and the web wizard bootstrap derived from it: the one
    derivation the page and its JSON projection share, computed and never persisted. Its Skills
    step is offered only where completion honors the starter ticks (``skill_starter.starter_choice_open``)."""
    from ouroboros.config import SETTINGS_PATH
    from ouroboros.gateway._helpers import request_drive_root
    from ouroboros.skill_starter import starter_choice_open

    settings, _changed, _keys = apply_runtime_provider_defaults(load_settings())
    enabled, first_state = _consciousness_control(request)
    return settings, build_setup_bootstrap(settings, "web", fresh_install=not SETTINGS_PATH.exists(),
                                           consciousness_enabled=enabled, consciousness_first_state=first_state,
                                           document=_owner_settings_document(),
                                           starter_choice=starter_choice_open(request_drive_root(request), settings))


async def onboarding_page(_request: Request) -> HTMLResponse:
    served_ui = getattr(_request.app.state, "served_ui", None)
    if served_ui is not None and _request.query_params.get("builtin") != "1":
        page = served_ui.custom_page("onboarding.html")
        if page is not None:
            return HTMLResponse(page, headers=_NO_STORE)
    _settings, bootstrap = _live_setup(_request)
    head = served_ui.appearance_meta(_request.headers) if served_ui is not None else ""
    return HTMLResponse(render_onboarding_html(bootstrap, head), headers=_NO_STORE)


async def api_onboarding_bootstrap(_request: Request) -> JSONResponse:
    """``OnboardingBootstrapResponse``: readiness plus the wizard's bootstrap, written nowhere."""
    settings, bootstrap = _live_setup(_request)
    return JSONResponse({"ready": has_startup_ready_provider(settings), "bootstrap": bootstrap}, headers=_NO_STORE)


class _AttemptRun:
    """One completion attempt this process ran: its commit boundary, then the envelope it ended with."""

    def __init__(self, boundary: CommitBoundary) -> None:
        self.boundary = boundary
        self.ended = False
        self.status = 0
        self.envelope: Dict[str, Any] = {}

    def end(self, response: JSONResponse) -> None:
        self.status = response.status_code
        self.envelope = json.loads(bytes(response.body) or b"{}")
        self.ended = True


# One entry per explicit Start or Retry; older ones only answer "not confirmed".
_RUNS_KEPT = 16
_runs: "OrderedDict[str, _AttemptRun]" = OrderedDict()
_runs_lock = threading.Lock()


def recorded_attempt(attempt_id: str, boundary: CommitBoundary, body: Callable) -> Callable:
    """``body``, one settings-writer episode, remembering how ``attempt_id`` ended — also
    after its caller stopped waiting (``settings_save_timeout``). Without an id, ``body``."""
    if not attempt_id:
        return body
    run = _AttemptRun(boundary)
    with _runs_lock:
        _runs.pop(attempt_id, None)
        _runs[attempt_id] = run
        while len(_runs) > _RUNS_KEPT:
            _runs.popitem(last=False)

    @functools.wraps(body)
    def episode(request: Request, payload: Any) -> JSONResponse:
        try:
            response = body(request, payload)
        except SettingsDocumentBusy as exc:  # raised before any write; the seam answers it the same way
            run.end(unsaved_error(str(exc), 503, code="settings_busy"))
            raise
        run.end(response)
        return response

    return episode


async def api_onboarding_attempt(request: Request) -> JSONResponse:
    """``OnboardingAttemptResponse``: what became of one completion attempt; writes nothing.

    ``saved`` is true only while the settings document holds THIS attempt's write, so a
    late earlier attempt, another tab or a later save cannot confirm it. The receipt is
    the attempt's own success envelope, present once its post-commit steps returned."""
    attempt_id, error = parse_completion_attempt({ATTEMPT_ID_FIELD: request.query_params.get("attempt", "")})
    if not attempt_id:
        return json_error(error or "attempt is required", 400)
    with _runs_lock:
        run = _runs.get(attempt_id)
    ended = run is not None and run.ended
    committed = run is not None and run.boundary.committed
    envelope = dict(run.envelope) if ended else {}
    failure = {key: envelope[key] for key in ("error", "code", "post_commit_failed") if envelope.get(key)}
    if str(_owner_settings_document().get(ONBOARDING_ATTEMPT_KEY) or "") != attempt_id:
        # Not the current settings: still running, never arrived, refused, or replaced since.
        answer = {"saved": False, **failure} if ended and not committed else {"saved": None}
    elif ended and run.status == 200:
        answer = {"saved": True, "receipt": envelope}
    else:  # stored; a post-commit step failed, is still running, or ran in an earlier process
        answer = {"saved": True, **(failure if committed else {})}
    return JSONResponse(answer, headers=_NO_STORE)


__all__ = ["api_onboarding_attempt", "api_onboarding_bootstrap", "onboarding_page", "recorded_attempt"]
