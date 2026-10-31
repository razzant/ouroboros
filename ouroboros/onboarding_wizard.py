"""Shared onboarding wizard helpers for desktop and web.

ONE onboarding host (D-8): the wizard is a real page served by the gateway on
every host — desktop, ordinary browser, Docker, and the Linux browser fallback.
The page LINKS its stylesheet and its ES module from ``/onboarding/static`` (the
core's builtin web tree, whatever main interface ``OUROBOROS_UI_DIR`` selects)
instead of inlining them, because the wizard's steps must be able to import
ordinary ``web/modules/*`` code (the account/login components), which a
self-contained inlined string can never do. This is the same contract, the same
validator and the same settings authority as before — only the delivery changed.
"""

from __future__ import annotations

import json
import pathlib
from typing import Tuple

from ouroboros.settings_setup_contract import (
    build_setup_bootstrap,
    validate_setup_payload,
)

_ASSET_ROOT = pathlib.Path(__file__).resolve().parents[1] / "web"
_TEMPLATE_PATH = _ASSET_ROOT / "onboarding_template.html"


from ouroboros.utils import read_text as _read_asset


def _bootstrap_script_json(bootstrap: dict) -> str:
    """Serialize the bootstrap for an inline ``<script>`` body.

    ``ensure_ascii`` alone does not neutralize ``</script>``; a stored provider
    value containing it would close the element and inject markup. Escaping
    ``<`` as ``\\u003c`` stays valid JSON *and* valid JavaScript.
    """
    return json.dumps(bootstrap, ensure_ascii=True).replace("<", "\\u003c")


def build_onboarding_html(settings: dict, host_mode: str = "desktop", *, fresh_install: bool = False) -> str:
    """Render the onboarding page for a live gateway.

    ``host_mode`` still selects the setup contract's host profile; every server
    route passes ``"web"`` now that the wizard always runs against a live API.
    """
    normalized_host_mode = "web" if host_mode == "web" else "desktop"
    return render_onboarding_html(build_setup_bootstrap(settings, normalized_host_mode, fresh_install=fresh_install))


def render_onboarding_html(bootstrap: dict, head: str = "") -> str:
    """The wizard page carrying exactly ``bootstrap`` (the object ``GET /api/onboarding/bootstrap`` returns);
    ``head`` leads its ``<head>`` (the appearance scope a served page names before its ``theme.js``)."""
    page = _read_asset(_TEMPLATE_PATH).replace(
        "__ONBOARDING_BOOTSTRAP__", _bootstrap_script_json(bootstrap)
    )
    return page.replace("<head>", "<head>" + head, 1) if head else page


def prepare_onboarding_settings(data: dict, current_settings: dict) -> Tuple[dict, str | None]:
    return validate_setup_payload(data, current_settings)
