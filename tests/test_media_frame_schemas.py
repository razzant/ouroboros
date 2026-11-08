"""Photo, video and document frames publish the fields and requiredness their browser twins declare.

The three outbound media frames inherit their shared fields from one private base in
``gateway/contracts.py``. ``tests/test_gateway_parity.py`` holds their field names to the JSDoc twins in
``web/modules/api_types.js``; this file also holds what ``GET /api/contract`` publishes as required, and each
frame's ``type``, to the twin, so an edit of the shared base cannot silently change every media frame at once.
"""

from __future__ import annotations

import pathlib
import re

import pytest

from ouroboros.gateway.schema import contract_document

_API_TYPES = pathlib.Path(__file__).resolve().parents[1] / "web" / "modules" / "api_types.js"


def _twin_properties(name: str) -> dict[str, str]:
    """``{field: JSDoc type}`` of one ``@typedef {Object}`` twin; the media twins' types carry no braces."""
    text = _API_TYPES.read_text(encoding="utf-8")
    match = re.search(rf"@typedef \{{Object\}} {name}\b(?P<body>.*?)\n \*/", text, re.S)
    assert match, f"api_types.js missing {name}"
    declared = re.findall(r"@property \{([^{}]*)\} ([A-Za-z_][A-Za-z0-9_]*)", match.group("body"))
    assert len({field for _, field in declared}) == len(declared), f"{name} twin declares a property twice"
    return {field: kind for kind, field in declared}


@pytest.mark.parametrize("name, frame_type", [
    ("PhotoOutbound", "photo"), ("VideoOutbound", "video"), ("DocumentOutbound", "document"),
])
def test_published_media_frame_matches_its_twin_including_requiredness(name, frame_type):
    schema = contract_document()["schemas"][name]
    twin = _twin_properties(name)
    published, declared = set(schema["required"]), {field for field, kind in twin.items() if not kind.endswith("=")}

    assert set(schema["properties"]) == set(twin), f"{name}: published fields differ from its twin"
    assert schema["required"] == sorted(declared), (
        f"{name}: required only as published {sorted(published - declared)}, "
        f"only in the twin {sorted(declared - published)}")
    assert schema["properties"]["type"] == {"enum": [frame_type]}
    assert twin["type"] == f'"{frame_type}"'
