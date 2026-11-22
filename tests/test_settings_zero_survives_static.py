"""A stored 0 survives the Settings form round trip.

The load path applied ``fallback && !s[key] ? fallback : s[key]`` to every
INPUT_FIELDS row, so a saved ``0`` (``OUROBOROS_CONSCIOUSNESS_DAILY_USD``, the
documented "may not spend" value) rendered as its fallback ``20`` and the next
save of any tab wrote ``20`` back. Only a missing value may take the fallback.

Source pin (the pattern of the other *_static tests) plus the typed server read
the helper relies on. The helper itself runs in the node lane
(``web/tests/settings_stored_zero.test.js``) for the INPUT_FIELDS rows that keep it; the
allowance's own editor is Costs now, and its 0 save/reload and an unrelated Settings save
run against a real server in ``tests/test_costs_allowance_browser.py``.
"""

from __future__ import annotations



def test_server_read_keeps_a_saved_zero_and_defaults_only_a_blank():
    # GET /api/settings serves this coercion: the form gets the number 0, never ''
    # or null, so the helper's absence branch cannot swallow it; a blank reads as the default.
    from ouroboros import config as cfg

    key = "OUROBOROS_CONSCIOUSNESS_DAILY_USD"
    assert cfg._coerce_setting_value(key, "0") == 0.0
    assert cfg._coerce_setting_value(key, 0) == 0.0
    assert cfg._coerce_setting_value(key, "") == cfg.SETTINGS_DEFAULTS[key]
