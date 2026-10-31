"""docs/CLIENT_CONTRACT.md names only what the core publishes.

Every route the document names is an indexed route with that method, every shape it cites is a
contract type the core can publish, every WebSocket type it names is indexed, and every section
reference points at a section that exists. The published facts themselves are pinned by the
conformance suites (``test_client_contract_*``, ``test_gateway_client_contract``)."""

from __future__ import annotations

import pathlib
import re

from ouroboros.gateway import endpoint_index
from ouroboros.gateway.contracts import HTTP_ENDPOINTS, WS_MESSAGE_TYPES
from ouroboros.gateway.schema import contract_typed_dict

REPO = pathlib.Path(__file__).resolve().parents[1]
DOC = (REPO / "docs" / "CLIENT_CONTRACT.md").read_text(encoding="utf-8")
TOKENS = [token for token in (*HTTP_ENDPOINTS, *endpoint_index.AUTH_ENDPOINTS) if not token.startswith("WS ")]


def _segments(path: str) -> list:
    return [None if part.startswith("{") else part for part in path.split("?", 1)[0].strip("/").split("/")]


def _served(method: str, path: str) -> bool:
    wanted = _segments(path)
    for token in TOKENS:
        token_method, token_path = token.split(" ", 1)
        if method not in {token_method, "ANY"} and token_method != "ANY":
            continue
        have = _segments(token_path)
        if token_path.endswith(":path}") and len(wanted) >= len(have) - 1:
            have, wanted_cut = have[:-1], wanted[:len(have) - 1]
        else:
            wanted_cut = wanted
        if len(have) == len(wanted_cut) and all(h is None or w is None or h == w for h, w in zip(have, wanted_cut)):
            return True
    return False


def test_every_named_route_is_an_indexed_route_with_that_method():
    named = re.findall(r"`(GET|POST|PATCH|DELETE)(?:`/`(?:GET|POST))? (/(?:api|auth)/[^`\s]*)", DOC)
    assert len(named) > 60, "the route extractor lost the document's routes"
    for method, path in named:
        if "..." in path:
            continue
        if path.endswith("/*"):  # a family: at least one indexed route lives under it
            assert any(t.split(" ", 1)[1].startswith(path[:-1]) for t in TOKENS), path
            continue
        assert _served(method, path), f"CLIENT_CONTRACT.md names {method} {path}, which no indexed route serves"
    for path in re.findall(r"`(/api/[^`\s]*)`", DOC):
        assert any(_served(method, path) for method in ("GET", "POST", "PATCH", "DELETE")), path


def test_every_cited_shape_is_a_contract_type():
    cited = set(re.findall(r"`([A-Z][A-Za-z]+(?:Response|Request|Refusal|Problem|Outbound|Inbound|Message|"
                           r"Document|Skill|Meta|Entry|Receipt|Resources|Operation))`", DOC))
    assert len(cited) > 40, "the shape extractor lost the document's shapes"
    for name in sorted(cited):
        assert contract_typed_dict(name), name


def test_every_named_websocket_type_is_indexed_and_every_section_reference_exists():
    section = re.search(r"\*\*Message types\.\*\*(?P<body>.*?)\*\*Extension messages\*\*", DOC, re.S).group("body")
    named = set(re.findall(r"`([a-z0-9_]+)`", section)) - {"chat_annotation", "suggested_name", "type", "data",
                                                             "task_id_pending"}
    assert named <= set(WS_MESSAGE_TYPES), sorted(named - set(WS_MESSAGE_TYPES))
    assert set(WS_MESSAGE_TYPES) - {"chat", "command"} <= named
    headings = {int(n) for n in re.findall(r"^## (\d+)\. ", DOC, re.M)}
    for reference in re.findall(r"§(\d+)", DOC):
        assert int(reference) in headings, f"§{reference} does not exist"
