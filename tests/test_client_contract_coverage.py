"""Coverage of the published client contract against the builtin interface, its real consumer.

Every ``/api`` and ``/auth`` address the builtin SPA builds must name an indexed route, and every
indexed route must publish its shapes (``typed: true``) with a body for each status it lists: the
contract a replacement client reads is at least as complete as what the builtin client relies on.
A route the builtin client does not call is still published; the matrix below is the evidence.
"""

from __future__ import annotations

import pathlib
import re

from ouroboros.gateway import endpoint_index
from ouroboros.gateway.contracts import HTTP_ENDPOINTS
from ouroboros.gateway.schema import contract_document

REPO = pathlib.Path(__file__).resolve().parents[1]


def _read_literal(text: str, i: int) -> tuple[str, int]:
    """The literal opening at ``text[i]`` and the index after it; ``${...}`` becomes ``\\x00``."""
    quote, j, n, out = text[i], i + 1, len(text), []
    while j < n and text[j] != quote:
        if quote == "`" and text.startswith("${", j):
            depth, k = 1, j + 2
            while k < n and depth:
                if text[k] == "{":
                    depth += 1
                elif text[k] == "}":
                    depth -= 1
                elif text[k] == "`":  # a nested template inside the expression
                    k += 1
                    while k < n and text[k] != "`":
                        k += 1
                k += 1
            out.append("\x00")
            j = k
            continue
        if text[j] == "\\":
            j += 1
        out.append(text[j])
        j += 1
    return "".join(out), j + 1


def _literals(text: str):
    """String and template literals starting with /api/ or /auth/, joined with the literals a
    ``+`` appends to them (``\x60/api/a/${x}\x60 + \x60/${y}\x60``)."""
    i, n = 0, len(text)
    while i < n:
        if text[i] in "'\"`" and text.startswith(("/api/", "/auth/"), i + 1):
            literal, i = _read_literal(text, i)
            while True:
                following = re.match(r"\s*\+\s*(['\"`])", text[i:])
                if not following or not text.startswith(("/", "?"), i + following.end()):
                    break
                more, i = _read_literal(text, i + following.end() - 1)
                literal += more
            yield literal
            continue
        i += 1


def _segments(literal: str) -> tuple:
    """Path segments of one literal; ``None`` is a computed segment. A computed suffix glued to
    the last segment (``/api/files/list${query}``) builds a query, not a path."""
    segments = []
    for segment in literal.split("?", 1)[0].split("/")[1:]:
        if segment == "\x00":
            segments.append(None)
        elif "\x00" in segment:
            head = segment.split("\x00", 1)[0]
            if head:
                segments.append(head)
            break
        else:
            segments.append(segment)
    return tuple(segments)


def spa_addresses() -> dict:
    """Builtin-client address -> the first-party modules that build it (excluding type carriers)."""
    from ouroboros.review import is_gated_js_module

    found: dict = {}
    for path in [*(REPO / "web").glob("*.js"), *(REPO / "web" / "modules").glob("*.js")]:
        relative = path.relative_to(REPO).as_posix()
        if not is_gated_js_module(relative) or path.name.endswith("_types.js"):
            continue  # type carriers describe shapes and values, not endpoint calls
        for literal in _literals(path.read_text(encoding="utf-8")):
            segments = _segments(literal)
            if not segments or segments[-1] == "" or any(s and "*" in s for s in segments):
                continue  # a prefix the module extends, or prose in a comment
            found.setdefault(segments, set()).add(relative)
    return found


def _token_matches(token: str, segments: tuple) -> bool:
    parts = token.split(" ", 1)[1].split("/")[1:]
    for index, part in enumerate(parts):
        if part.startswith("{") and part.endswith(":path}"):
            return len(segments) > index
        if index >= len(segments):
            return False
        if part.startswith("{"):
            continue
        if segments[index] != part:
            return False
    return len(segments) == len(parts)


def tokens_for(segments: tuple) -> list:
    tokens = [*HTTP_ENDPOINTS, *endpoint_index.AUTH_ENDPOINTS]
    exact = [t for t in tokens if not t.startswith("WS ") and _token_matches(t, segments)]
    # A literal segment beats a parameter (``/api/projects/from-task`` is not ``{project_id}``).
    best = max((sum(not p.startswith("{") for p in t.split(" ", 1)[1].split("/")) for t in exact), default=0)
    return [t for t in exact if sum(not p.startswith("{") for p in t.split(" ", 1)[1].split("/")) == best]


def test_every_address_the_builtin_client_builds_is_an_indexed_route():
    addresses = spa_addresses()
    assert len(addresses) > 90, "the extractor lost the builtin client's addresses"
    unresolved = {"/" + "/".join(s or "{*}" for s in segments): sorted(modules)
                  for segments, modules in addresses.items() if not tokens_for(segments)}
    assert not unresolved, f"the builtin client builds addresses no indexed route serves: {unresolved}"


def test_the_extractor_reads_computed_segments_and_queries():
    literal = next(_literals("fetch(`/api/claudexor/login/${encodeURIComponent(id)}/input?x=${`y`}`)"))
    assert _segments(literal) == ("api", "claudexor", "login", None, "input")
    assert _segments(next(_literals("`/api/files/list${query ? `?${query}` : ''}`"))) == ("api", "files", "list")
    joined = "`/api/claudexor/credential-profiles/${encodeURIComponent(h)}`\n        + `/${encodeURIComponent(p)}`"
    assert _segments(next(_literals(joined))) == ("api", "claudexor", "credential-profiles", None, None)
    assert tokens_for(("api", "projects", "from-task")) == ["POST /api/projects/from-task"]
    assert tokens_for(("api", "extensions", None, "module", "lib", "x.js")) == [
        "GET /api/extensions/{skill}/module/{entry:path}"]


def coverage_matrix() -> list[dict]:
    """One row per published HTTP route: shapes and the builtin modules that call it."""
    document = contract_document()
    callers: dict = {}
    for segments, modules in spa_addresses().items():
        for token in tokens_for(segments):
            callers.setdefault(token, set()).update(modules)
    rows = []
    for entry in document["http"]:
        token = f"{entry['method']} {entry['path']}"
        rows.append({
            "token": token, "typed": entry["typed"],
            "request": entry.get("request", {}).get("schemas", []),
            "responses": {status: body["schemas"] or [body["content_type"]]
                          for status, body in entry.get("responses", {}).items()},
            "spa": sorted(callers.get(token, ())),
        })
    return rows


def test_the_matrix_names_every_indexed_route_once():
    rows = coverage_matrix()
    tokens = [row["token"] for row in rows]
    assert len(tokens) == len(set(tokens))
    assert set(tokens) == {t for t in HTTP_ENDPOINTS if not t.startswith("WS ")} | set(endpoint_index.AUTH_ENDPOINTS)
    assert all(row["typed"] and row["responses"] for row in rows), [row["token"] for row in rows if not row["typed"]]
    by_token = {row["token"]: row for row in rows}
    assert "web/modules/harness_login_cards.js" in by_token["POST /api/claudexor/login/{job_id}/input"]["spa"]
    assert re.search(r"files\.js", " ".join(by_token["POST /api/files/transfer"]["spa"]))
