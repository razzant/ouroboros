"""ABI 7.0 (ABI-10) — F3.3 comma-list remnant sweep: the phase CI gate.

Grep-level checker over ``ouroboros/`` + ``web/`` (+ ``supervisor/``) pinning
that the retired reviewer comma-list surface stays retired. Three sweeps:

1. every textual mention of a retired comma key (the list is snapped
   DYNAMICALLY from ``RETIRED_COMMA_LIST_SETTING_KEYS`` — a key added to the
   retirement SSOT is swept automatically) must sit on the count-anchored
   allowlist of legitimate remnants below;
2. comma-``split`` parsing inside model/review modules is count-anchored to
   the known derived-plane parsers — the review-configuration modules
   themselves carry NO comma parsing (the structured
   ``OUROBOROS_REVIEWER_SLOTS`` is the one configuration surface);
3. the phase-5 plumbing removed by the sweep stays removed, and the retired
   per-row route envs are IGNORED at runtime (retired-envs-are-ignored pin).

Allowlist discipline follows tests/test_gateway_abi3_removals.py: PER-SITE and
COUNT-ANCHORED — a new mention in an allowlisted file breaks the anchor and
fails; a stale row (no mention matches) fails too, so the allowlist can only
shrink deliberately.
"""

from __future__ import annotations

import ast
import pathlib
import re
from collections import Counter

from ouroboros.settings_defaults import RETIRED_COMMA_LIST_SETTING_KEYS

_ROOT = pathlib.Path(__file__).resolve().parents[1]
_SWEEP_BASES = ("ouroboros", "web", "supervisor")
_SWEEP_SUFFIXES = (".py", ".js", ".html", ".css")


def _sweep_files():
    for base in _SWEEP_BASES:
        for path in sorted((_ROOT / base).rglob("*")):
            if path.suffix in _SWEEP_SUFFIXES and path.is_file():
                yield path


# (posix path, retired key) -> (reason, exact mention count).
# Every row is a LEGITIMATE remnant class disclosed in
# docs/v7next/LEDGER_CORRECTIONS.md ("From the F3.3 comma-sweep"):
#   retirement-SSOT — the list that declares the keys retired;
#   derived env plane — the comma ENV spellings of the two model lists live on
#     as the runtime projection for the API-pinned surfaces (never settings);
#   raw-dict tolerance — normalizers that accept a raw dict fed directly
#     (tests/tools), ABI-10-commented in place;
#   retirement prose — comments/docstrings that NAME the key to say it is
#     retired and ignored.
_RETIRED_KEY_MENTION_ALLOWLIST = {
    # -- retirement SSOT (declares the keys retired; feeds the RC auditor).
    ("ouroboros/settings_defaults.py", "OUROBOROS_REVIEW_MODELS"): ("retirement SSOT", 2),
    ("ouroboros/settings_defaults.py", "OUROBOROS_SCOPE_REVIEW_MODELS"): ("retirement SSOT", 2),
    ("ouroboros/settings_defaults.py", "OUROBOROS_SCOPE_REVIEW_MODEL"): ("retirement SSOT", 2),
    ("ouroboros/settings_defaults.py", "OUROBOROS_REVIEW_ROUTES"): ("retirement SSOT", 2),
    ("ouroboros/settings_defaults.py", "OUROBOROS_SCOPE_REVIEW_ROUTES"): ("retirement SSOT", 2),
    ("ouroboros/settings_defaults.py", "OUROBOROS_ADVISORY_REVIEW_ROUTE"): ("retirement SSOT", 2),
    # -- derived env plane: the projection writer (D15) …
    ("ouroboros/reviewer_slot_config.py", "OUROBOROS_REVIEW_MODELS"): ("derived env plane projection writer", 4),
    ("ouroboros/reviewer_slot_config.py", "OUROBOROS_SCOPE_REVIEW_MODELS"): ("derived env plane projection writer", 4),
    ("ouroboros/reviewer_slot_config.py", "OUROBOROS_SCOPE_REVIEW_MODEL"): ("derived env plane projection writer", 2),
    # … and its API-pinned readers.
    ("ouroboros/review_model_routes.py", "OUROBOROS_REVIEW_MODELS"): ("derived env plane reader (get_review_models)", 1),
    ("ouroboros/review_model_routes.py", "OUROBOROS_SCOPE_REVIEW_MODELS"): ("derived env plane reader (get_scope_review_models)", 1),
    ("ouroboros/review_model_routes.py", "OUROBOROS_SCOPE_REVIEW_MODEL"): ("derived env plane reader (singular fallback)", 2),
    ("ouroboros/tools/scope_review_budget.py", "OUROBOROS_SCOPE_REVIEW_MODEL"): ("derived env plane reader (budget fallback)", 1),
    # -- raw-dict tolerance: retired-model default refresh over dicts fed
    #    directly (load_settings purges the keys first; ABI-10-commented).
    ("ouroboros/server_runtime.py", "OUROBOROS_REVIEW_MODELS"): ("raw-dict retired-model default refresh", 8),
    ("ouroboros/server_runtime.py", "OUROBOROS_SCOPE_REVIEW_MODELS"): ("raw-dict retired-model default refresh", 10),
    ("ouroboros/server_runtime.py", "OUROBOROS_SCOPE_REVIEW_MODEL"): ("raw-dict retired-model default refresh", 8),
    # -- declaration surface over the derived plane (Provider Test resolves a
    #    deterministic model from declared model settings incl. the projected
    #    comma lists; never a route selector).
    ("ouroboros/provider_models.py", "OUROBOROS_REVIEW_MODELS"): ("declared-model surface over derived plane", 1),
    ("ouroboros/provider_models.py", "OUROBOROS_SCOPE_REVIEW_MODELS"): ("declared-model surface over derived plane", 1),
    ("ouroboros/provider_models.py", "OUROBOROS_SCOPE_REVIEW_MODEL"): ("declared-model surface over derived plane", 1),
    # -- save-time warning triggers keyed on changed setting names (the
    #    OUROBOROS_REVIEWER_SLOTS member keeps the check live; the retired
    #    spellings are harmless startswith vestiges kept for raw-dict callers).
    ("ouroboros/gateway/settings.py", "OUROBOROS_REVIEW_MODELS"): ("changed-key warning trigger", 1),
    ("ouroboros/gateway/settings.py", "OUROBOROS_SCOPE_REVIEW_MODEL"): ("changed-key warning trigger", 2),
    # -- retirement prose (names the key to say it is retired/ignored).
    ("ouroboros/review_execution.py", "OUROBOROS_REVIEW_ROUTES"): ("retirement prose", 1),
    ("ouroboros/review_execution.py", "OUROBOROS_SCOPE_REVIEW_ROUTES"): ("retirement prose", 1),
    ("ouroboros/tools/preflight_review_run.py", "OUROBOROS_ADVISORY_REVIEW_ROUTE"): ("retirement prose", 1),
    ("web/modules/settings.js", "OUROBOROS_REVIEW_MODELS"): ("retirement prose (6.1 authoring note)", 1),
    ("web/modules/settings.js", "OUROBOROS_SCOPE_REVIEW_MODELS"): ("retirement prose (6.1 authoring note)", 1),
}


def test_retired_comma_key_mentions_are_allowlisted():
    keys = sorted(RETIRED_COMMA_LIST_SETTING_KEYS, key=len, reverse=True)
    assert keys, "retirement SSOT list is empty — the sweep has no subject"
    observed: Counter = Counter()
    for path in _sweep_files():
        text = path.read_text(encoding="utf-8", errors="replace")
        rel = path.relative_to(_ROOT).as_posix()
        for key in keys:
            # Negative lookahead keeps OUROBOROS_SCOPE_REVIEW_MODEL from also
            # counting every OUROBOROS_SCOPE_REVIEW_MODELS mention.
            count = len(re.findall(rf"{key}(?![A-Z_])", text))
            if count:
                observed[(rel, key)] = count
    expected = {site: count for site, (_reason, count) in _RETIRED_KEY_MENTION_ALLOWLIST.items()}
    new_sites = {site: n for site, n in observed.items() if site not in expected}
    assert not new_sites, f"NEW retired comma-key mentions (extend or clean): {new_sites}"
    stale = {site: n for site, n in expected.items() if site not in observed}
    assert not stale, f"stale allowlist rows (remnant gone — delete the row): {stale}"
    drifted = {site: (observed[site], expected[site])
               for site in observed if observed[site] != expected[site]}
    assert not drifted, f"mention count drifted (observed, allowlisted): {drifted}"


# posix path -> (reason, exact count of comma-split call sites). The review
# configuration modules themselves must stay comma-split-free.
_COMMA_SPLIT_ALLOWLIST = {
    "ouroboros/model_slots.py": ("generic model-list parser (OUROBOROS_MODEL_FALLBACKS et al.)", 1),
    "ouroboros/provider_models.py": ("declared-model list parser over the derived plane", 1),
    "ouroboros/gateway/models.py": ("Provider Test deterministic-model resolver", 1),
}


def _count_comma_split_calls_py(text: str) -> int:
    """AST-level count of ``<expr>.split(",")`` / ``.rsplit(",")`` calls —
    positional or keyword extras (``maxsplit=``), spacing and quote style
    cannot evade it (the setattr-scan lesson: a parser gate scans SYNTAX,
    not one string spelling). The separator may arrive as EITHER of the
    first two positionals (bound ``s.split(",")`` / unbound
    ``str.split(raw, ",")``) or the ``sep=`` keyword regardless of
    positional count (unbound ``str.split(raw, sep=",")``)."""
    tree = ast.parse(text)
    count = 0
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)):
            continue
        if node.func.attr not in ("split", "rsplit"):
            continue
        sep_candidates = list(node.args[:2]) + [
            kw.value for kw in node.keywords if kw.arg == "sep"
        ]
        if any(
            isinstance(candidate, ast.Constant) and candidate.value == ","
            for candidate in sep_candidates
        ):
            count += 1
    return count


def test_comma_split_model_parsing_is_allowlisted():
    observed: Counter = Counter()
    for path in _sweep_files():
        rel = path.relative_to(_ROOT).as_posix()
        if not re.search(r"model|review", rel):
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        if path.suffix == ".py":
            count = _count_comma_split_calls_py(text)
        else:
            # Non-Python mirrors keep the textual scan (no Python AST there).
            count = len(re.findall(r"""split\(\s*(?:","|',')\s*""", text))
        if count:
            observed[rel] = count
    expected = {rel: count for rel, (_reason, count) in _COMMA_SPLIT_ALLOWLIST.items()}
    assert dict(observed) == expected, (
        f"comma-split drift in model/review modules: observed={dict(observed)}, "
        f"allowlisted={expected}"
    )


def test_comma_split_ast_scan_sees_the_evasion_spellings():
    """Self-test of the gate's detector: the plain-syntax evasions the string
    regexp missed are counted, and a non-comma split is not."""
    assert _count_comma_split_calls_py('raw.split(",", maxsplit=-1)') == 1
    assert _count_comma_split_calls_py("raw.split( ',' )") == 1
    assert _count_comma_split_calls_py("raw.rsplit(',')") == 1
    assert _count_comma_split_calls_py('raw.split(sep=",")') == 1
    # Unbound-method evasions (fix-round 2, claim 7): positional args present
    # AND the separator hiding in sep=, or riding as the second positional.
    assert _count_comma_split_calls_py('str.split(raw, sep=",")') == 1
    assert _count_comma_split_calls_py('str.split(raw, ",")') == 1
    assert _count_comma_split_calls_py('str.rsplit(raw, ",")') == 1
    assert _count_comma_split_calls_py('raw.split(";")') == 0
    assert _count_comma_split_calls_py("raw.split()") == 0
    assert _count_comma_split_calls_py("str.split(raw)") == 0


def test_phase5_route_plumbing_stays_removed():
    """The F3.3 removals stay removed: no per-row route env plumbing, no
    advisory route env constant, anywhere under the swept trees."""
    retired_symbols = (
        "configured_review_routes",
        "TRIAD_REVIEW_ROUTES_ENV",
        "SCOPE_REVIEW_ROUTES_ENV",
        "ADVISORY_REVIEW_ROUTE_ENV",
        "route_env_key",
    )
    hits = []
    for path in _sweep_files():
        text = path.read_text(encoding="utf-8", errors="replace")
        for symbol in retired_symbols:
            if re.search(rf"\b{symbol}\b", text):
                hits.append(f"{path.relative_to(_ROOT).as_posix()}: {symbol}")
    assert not hits, f"phase-5 route plumbing remnants: {hits}"


def test_retired_route_envs_are_ignored(monkeypatch):
    """Retired-envs-are-ignored pin: a stale environment exporting the retired
    per-row route spellings changes NOTHING — rows built from a plain model
    list stay api_chat."""
    from ouroboros.review_execution import ReviewRouteKind
    from ouroboros.review_substrate import scope_reviewer_slots
    from ouroboros.reviewer_slot_config import reviewer_slots

    monkeypatch.delenv("OUROBOROS_REVIEWER_SLOTS", raising=False)
    monkeypatch.setenv("OUROBOROS_REVIEW_ROUTES", "agent_session,agent_session,agent_session")
    monkeypatch.setenv("OUROBOROS_SCOPE_REVIEW_ROUTES", "agent_session,agent_session")
    assert all(row.route is ReviewRouteKind.API_CHAT
               for row in scope_reviewer_slots(["m1", "m2"]))
    assert all(row.route is ReviewRouteKind.API_CHAT
               for row in reviewer_slots(["m1", "m2", "m3"], role_hint="commit review"))
