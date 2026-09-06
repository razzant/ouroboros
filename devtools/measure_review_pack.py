#!/usr/bin/env python3
"""Measure the review packs for one staged change — offline, $0.

Reports, for the checkout at ``--repo`` (its INDEX is the reviewed change; EVERY
governance corpus — BIBLE.md, the checklist section + archive, DEVELOPMENT.md,
DESIGN.md, ARCHITECTURE.md — is read from that one checkout):

* the triad touched-file pack BEFORE and AFTER the disclosed pack exclusions
  (``review_file_pack.triad_pack_exclusions``: span-only release carriers on a
  VERSION-staged commit, governance docs byte-identical to the inlined prefix);
* the FULL scope input (``scope_full``): the prompt the REAL assembler
  (``scope_review_pack._build_scope_prompt``) builds for this index — scope
  checklist + the five canonical docs in full (the byte-stable prefix), intent
  scaffolding, touched snapshots + deleted-file HEAD content, the staged diff and
  the generated repo atlas — split at the assembler's own stable-prefix
  boundary, with the guaranteed-fit ladder's facts from the scope context
  manifest (atlas status, selected/tracked counts, ladder steps, unassembled
  REQUIRED artifacts) and the headroom the scope input cap leaves. Its touched
  section BEFORE and AFTER the span-only release-carrier cut is printed as a
  labelled SUB-number (``scope_full.scope_touched``): a fragment of the input,
  never the input. The prompt is built for an empty commit message (the goal
  section of the real run adds its length);
* the advisory changed-context pack BEFORE and AFTER the same carrier cut over
  the pair it reviews (HEAD→working tree);
* the byte-stable governance prefix the triad prepends (checklist section +
  archive, DEVELOPMENT.md, DESIGN.md, ARCHITECTURE.md) and the constitutional
  head (preamble + BIBLE.md) each api row receives per round, part by part;
* the ZERO-DIFF message — everything an api row receives before the first pack
  or diff byte, serialized as ``_multi_model_review_async`` sends it: the
  constitutional head + the stable prefix + the dynamic scaffolding rendered
  with an empty pack and diff + the fixed user turn — the quorum input limit of
  the rows that RECEIVE the pack (the ``api_chat`` rows without a configured-
  subagent binding, filtered exactly as ``review`` filters them before
  ``fit_triad_prompt``; a session row or a subagent api row retrieves with its
  own tools and never constrains the ladder), and the headroom that limit
  leaves for the pack + diff. A panel whose every row retrieves gets the
  explicit "no API pack is assembled for this panel" instead of a number.

Units: chars, the host's own ``utils.estimate_tokens`` (chars/4 — the unit
``review_admission.fit_triad_prompt`` compares against the quorum limit, so every
limit/headroom figure is in it) and tiktoken ``o200k_base`` (not Anthropic's
tokenizer) are printed side by side and never conflated.

One change, checked: every arm reads the checkout's INDEX as the reviewed change
while the packs read working-tree text and the advisory arm resolves its paths
from ``git status --porcelain`` (HEAD→working tree, as the run does). Those
coincide only when the index IS the working tree, so a checkout with an
unstaged edit or an untracked file is refused with the typed
:class:`MeasuredCheckoutDirty` (exit 2) instead of measured across two changes.

Offline by construction: reviewer windows are read from the Capability Evidence
CACHE only (``capability_evidence.probe(allow_fetch=False)`` under
``$OUROBOROS_DATA_DIR``) — an unknown route is disclosed and sized at the fit
ladder's own full-window default, never probed or persisted; the o200k BPE is
served from the local tiktoken cache only (``TIKTOKEN_CACHE_DIR``) and a missing
BPE is a typed, disclosed miss, never a download. The scope assembler resolves
its input cap through ``scope_window`` (one lazy provider-metadata fetch per
evidence TTL) — this measurer never lets it: for the duration of the build the
assembler's call-time seams (``scope_review._effective_scope_input_limit`` /
``_scope_window``) are bound to a cap computed by the cap's own formula
(``_scope_input_limit``) on the cache-only window, or on ``--scope-window`` when
given, and ``load_checklist_section`` to the measured checkout's CHECKLISTS.md.
The atlas is git + filesystem, built with ``drive_root=None`` (no inventory is
persisted). Nothing is dispatched.

Usage::

    python devtools/measure_review_pack.py --repo /path/to/checkout [--json] [--scope-window N]
"""

from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

# The fixed user turn `review._dispatch_unified_review` sends under the system
# packet (its `content=` argument): reviewer input the fit ladder never sees.
TRIAD_USER_TURN = "Review the staged diff and context provided in the instructions above."
FIT_UNITS = ("chars/4 (utils.estimate_tokens) — the unit review_admission.fit_triad_prompt "
             "compares against the quorum limit")
SCOPE_LIMIT_NOTE = (
    "scope input cap computed offline by _effective_scope_input_limit's own formula on the window "
    "named here (window-scaled reserves, density-calibrated cap read from the evidence store, "
    "REVIEW_PROMPT_TOKEN_BUDGET ceiling); the runtime resolves its window through scope_window, "
    "which this measurer never calls")
NO_API_PACK_NOTE = (
    "no API pack is assembled for this panel: every configured row retrieves the subject with its "
    "own tools (agent_session rows and configured-subagent api rows), so there is no quorum input "
    "limit and no headroom to report")


class TokenizerUnavailable(RuntimeError):
    """The o200k BPE is not in the local tiktoken cache; this measurer never downloads it."""


class MeasuredCheckoutDirty(RuntimeError):
    """The checkout's index is not its working tree: its arms would measure two different changes."""


def _o200k():
    """tiktoken ``o200k_base`` from the LOCAL BPE cache only.

    ``tiktoken.load.read_file_cached`` serves the cache and otherwise calls
    ``read_file`` (an HTTP GET) through its module namespace at call time, so
    binding that name to a typed refusal for the duration of the load turns the
    download path into :class:`TokenizerUnavailable` instead of a network call."""
    import tiktoken
    from tiktoken import load as tiktoken_load

    def _refuse(blobpath: str) -> bytes:
        raise TokenizerUnavailable(
            f"o200k_base BPE is not cached locally (would fetch {blobpath}); "
            "point TIKTOKEN_CACHE_DIR at a warmed cache — this measurer never downloads")

    fetch = tiktoken_load.read_file
    tiktoken_load.read_file = _refuse
    try:
        return tiktoken.get_encoding("o200k_base")
    finally:
        tiktoken_load.read_file = fetch


def _measure(text: str, enc) -> dict:
    from ouroboros.utils import estimate_tokens

    return {
        "chars": len(text),
        "chars_div_4": int(estimate_tokens(text)),
        "o200k": len(enc.encode(text, disallowed_special=())) if enc is not None else None,
    }


def _staged_entries(repo: pathlib.Path) -> list[tuple[str, str, str]]:
    """``(status, current_path, source_path)`` per staged entry, parsed exactly as
    the scope assembler parses ``git diff --cached --name-status`` — so a staged
    deletion is a ``D`` entry the scope arm inlines from HEAD (the real pack's
    deleted-file section), not a current path that resolves to nothing."""
    from ouroboros.tools.review_file_pack import parse_git_name_status

    out = subprocess.run(
        ["git", "diff", "--cached", "--name-status"], cwd=str(repo),
        check=True, capture_output=True, text=True,
    ).stdout
    return parse_git_name_status(out)


def _porcelain(repo: pathlib.Path) -> str:
    """``git status --porcelain`` — the text the advisory run resolves its paths from."""
    return subprocess.run(
        ["git", "status", "--porcelain"], cwd=str(repo), check=True, capture_output=True, text=True).stdout


def _require_index_is_worktree(porcelain: str) -> None:
    """The one-change invariant every arm rests on (module docstring): an entry
    with a worktree-column status — an unstaged edit (``XM``), an untracked file
    (``??``) — means the advisory arm would pack text or paths the index arms
    never see. Typed refusal, never a silently cross-arm number."""
    dirty = [line for line in porcelain.splitlines() if len(line) > 1 and line[1] != " "]
    if dirty:
        raise MeasuredCheckoutDirty(
            "the checkout's index is not its working tree — stage or stash these before measuring: "
            + "; ".join(dirty))


def _panel_rows(plan: dict) -> list[dict]:
    """Every configured triad row with its delivery class, decided exactly as
    ``review`` decides it before ``fit_triad_prompt``: an ``api_chat`` row with
    no configured-subagent binding RECEIVES the api pack; a session row and a
    subagent api row retrieve with their own tools (``delivery_retrieves`` — the
    one delivery-class predicate) and neither constrain the fit ladder nor get
    the pack (5.2/5.7). The aligned vectors are ``commit_triad_delivery``'s."""
    from ouroboros.review_execution import delivery_retrieves

    actors = list(plan.get("subagent_ids") or [])
    rows = []
    for i, (model, route) in enumerate(zip(plan["models"], plan["routes"])):
        actor = str(actors[i] if i < len(actors) else "")
        rows.append({"model": model, "route": str(getattr(route, "value", route)), "subagent_id": actor,
                     "receives_pack": not delivery_retrieves(route, actor)})
    return rows


def _checklist_section(repo: pathlib.Path) -> str:
    """``review._load_checklist_section()`` read from the TARGET checkout.

    The runtime reads the checklist from its own REPO_ROOT (a frozen contract);
    this measurer measures one checkout, so the same section + archive come from
    ``repo``."""
    path = repo / "docs" / "CHECKLISTS.md"
    text = path.read_text(encoding="utf-8")
    header = "## Repo Commit Checklist"
    start = text.find(header)
    if start == -1:
        raise ValueError(f"Section {header!r} not found in {path}")
    end = text.find("\n## ", start + len(header))
    section = text[start:] if end == -1 else text[start:end]
    archive = (repo / "docs" / "CHECKLISTS_ARCHIVE.md").read_text(encoding="utf-8").strip()
    return f"{section}\n\n{archive}" if archive else section


def _governance_prefix(repo: pathlib.Path) -> dict[str, str]:
    """The stable prefix parts exactly as `_prepare_unified_review` loads them, from ``repo``."""
    from ouroboros.tools import review
    from ouroboros.tools.review_helpers import load_governance_doc

    docs = {
        rel: load_governance_doc(repo, rel, on_missing="explicit")
        for rel in ("docs/DEVELOPMENT.md", "docs/DESIGN.md", "docs/ARCHITECTURE.md")
    }
    checklist = _checklist_section(repo)
    stable = review._REVIEW_PROMPT_TEMPLATE_STABLE.format(
        preamble=review.REVIEW_PREAMBLE,
        critical_calibration=review.CRITICAL_FINDING_CALIBRATION,
        json_contract=review.REVIEW_JSON_ARRAY_CONTRACT,
        anti_pattern_lock_guard=review.REPO_ANTI_PATTERN_LOCK_GUARD,
        checklist_section=checklist,
        dev_guide_text=docs["docs/DEVELOPMENT.md"],
        design_text=docs["docs/DESIGN.md"],
        architecture_section=docs["docs/ARCHITECTURE.md"],
    )
    return {"stable_prefix": stable, "checklist_section": checklist, **docs}


def _constitutional_head(repo: pathlib.Path) -> str:
    """`_multi_model_review_async`'s stable head: the preamble + BIBLE.md from ``repo``."""
    from ouroboros.tools import review_multi_model as mm
    from ouroboros.tools.review_helpers import load_governance_doc

    bible = load_governance_doc(repo, "BIBLE.md", on_missing="explicit")
    return mm._CONSTITUTIONAL_PREAMBLE + "### BIBLE.md (Full Text)\n\n" + bible + "\n\n---\n\n## REVIEW INSTRUCTIONS\n\n"


def _zero_diff_message(repo: pathlib.Path, prefix: dict[str, str], paths: list[str]) -> dict[str, str]:
    """Every byte an api row receives BEFORE the pack and the diff, part by part in
    wire order: the constitutional head (prepended to the system content), the
    stable prefix, the dynamic scaffolding (`_REVIEW_PROMPT_TEMPLATE_DYNAMIC` with
    an empty pack and diff: the goal section of an empty commit message, no scope,
    no rebuttal, no history, the changed-files list) and the fixed user turn."""
    from ouroboros.tools import review
    from ouroboros.tools.review_helpers import build_goal_section, build_scope_section

    dynamic = review._REVIEW_PROMPT_TEMPLATE_DYNAMIC.format(
        goal_section=build_goal_section("", "", ""),
        scope_section=build_scope_section(""),
        current_files_section="",
        rebuttal_section="",
        review_history_section="",
        diff_text="",
        changed_files="\n".join(paths),
    )
    return {
        "constitutional_head_preamble_plus_BIBLE": _constitutional_head(repo),
        "stable_prefix": prefix["stable_prefix"],
        # `_assemble_prompt` joins stable + "\n" + dynamic; the separator rides with the tail.
        "dynamic_scaffolding_empty_pack_and_diff": "\n" + dynamic,
        "user_turn": TRIAD_USER_TURN,
    }


def _cached_window(model: str) -> tuple[int, str]:
    """``(sizing window, evidence)`` for one api row from the Capability Evidence CACHE.

    The same route derivation as ``reviewer_window.resolve_reviewer_window``
    (``reviewer_route`` + ``review_model_uses_local``), read through
    ``capability_evidence.probe(allow_fetch=False)`` — the hot-path reader that
    returns a fresh record as-is, an expired one marked stale and ``unprobeable``
    for an absent one, and never fetches provider metadata or writes the store.
    An unknown window is disclosed and sized exactly as the fit ladder sizes it
    (``reviewer_context_window``'s full-window default), so the limit derived here
    is the limit the ladder would compute on the same cache."""
    from ouroboros.capability_evidence import probe
    from ouroboros.config import DATA_DIR
    from ouroboros.provider_models import review_model_uses_local
    from ouroboros.reviewer_window import REVIEWER_FULL_WINDOW, ReviewerWindow, reviewer_route

    use_local = review_model_uses_local(model)
    provider, base_url = reviewer_route(model)
    ev = probe(DATA_DIR, provider="local" if use_local else provider, model=model,
               base_url=base_url, use_local=use_local, allow_fetch=False)
    window = ReviewerWindow(
        window_tokens=int(getattr(ev, "window_tokens", 0) or 0),
        status=str(getattr(ev, "status", "") or ""),
        stale=bool(getattr(ev, "stale", False)),
        observed_at=str(getattr(ev, "ts", "") or ""),
        model=model,
    )
    if window.window_tokens > 0:
        return window.sizing_window(), f"{window.status}{' stale' if window.stale else ''} (cache-only)"
    return window.sizing_window(), (
        f"window unknown (cache-only); sized at the ladder's {REVIEWER_FULL_WINDOW:,} default")


def _quorum_limit(models: list[str]) -> tuple[int, dict[str, dict]]:
    """The panel's quorum input limit exactly as ``fit_triad_prompt`` derives it, on
    cache-only windows; the per-slot rows disclose window, evidence and limit."""
    from ouroboros.tools import review as _rv

    slots: dict[str, dict] = {}
    for model in models:
        window, evidence = _cached_window(model)
        output_reserve, margin = _rv.window_scaled_reserves(
            window, output_reserve=_rv._review_output_budget(), tokenizer_margin=50_000)
        limit = max(0, _rv.calibrated_input_token_limit(
            model, context_window=window, output_reserve=output_reserve,
            tokenizer_margin=margin, budget_cap=_rv.REVIEW_PROMPT_TOKEN_BUDGET))
        slots[model] = {"window": window, "evidence": evidence, "input_limit_chars_div_4": limit}
    return int(_rv._quorum_input_token_limit(
        models, {m: s["input_limit_chars_div_4"] for m, s in slots.items()})), slots


def _scope_input_limit(model: str, window: int) -> int:
    """The scope input cap by ``_effective_scope_input_limit``'s own formula on an
    EXPLICIT sizing window: window-scaled reserves, then the density-calibrated
    cap (``calibrated_input_token_limit``; density read from the evidence store
    under ``$OUROBOROS_DATA_DIR``, the cold floor when nothing is recorded) under
    ``REVIEW_PROMPT_TOKEN_BUDGET``. The runtime helper itself is not called: it
    resolves the window through ``scope_window``, whose lazy metadata fetch this
    measurer never performs."""
    from ouroboros.tools import scope_review_budget as sb

    output_reserve, margin = sb._window_scaled_reserves(window)
    return max(0, sb._calibrated_input_token_limit(
        model, context_window=window, output_reserve=output_reserve,
        tokenizer_margin=margin, budget_cap=sb._SCOPE_BUDGET_TOKEN_LIMIT))


def _scope_full(repo: pathlib.Path, model: str, window: int, limit: int, enc) -> dict:
    """The REAL scope input for this index: ``_build_scope_prompt`` run as
    ``review_admission.prepare_scope_review`` runs it for an ordinary commit
    (no managed subject, ``drive_root=None`` so no inventory is persisted and no
    obligations are read), split at the assembler's own stable-prefix boundary,
    plus the ladder facts of the scope context manifest. The assembler's
    call-time seams are bound for the duration of the build (module docstring):
    the cap to ``limit``, the window to ``window`` (the ``budget_exceeded``
    terminal reads it), the checklist to the measured checkout."""
    from unittest import mock

    from ouroboros.reviewer_window import ReviewerWindow
    from ouroboros.tools import scope_review as sr
    from ouroboros.tools import scope_review_pack as sp
    from ouroboros.tools.review_helpers import load_checklist_section

    def _checklist(section_name: str, checklist_path=None) -> str:
        return load_checklist_section(section_name, checklist_path or repo / "docs" / "CHECKLISTS.md")

    with mock.patch.object(sr, "_effective_scope_input_limit", lambda *, scope_model="": limit), \
            mock.patch.object(sr, "_scope_window",
                              lambda m, *, session=False: ReviewerWindow(window_tokens=window, model=m)), \
            mock.patch.object(sr, "load_checklist_section", _checklist):
        prompt, status = sp._build_scope_prompt(
            repo, commit_message="",
            context=sp._ScopePromptContext(scope_model=model, governance_repo_dir=repo, drive_root=None))
    manifest = sp._current_scope_context_manifest()
    atlas = {key: manifest.get(key) for key in (
        "status", "selected_count", "tracked_count", "atlas_tokens", "fixed_prompt_tokens",
        "estimated_total_tokens", "target_total_tokens", "hard_total_tokens", "compact_manifest_in_prompt")}
    atlas["unassembled_required"] = [str(row.get("path") or "?") for row in manifest.get("unassembled_required") or []]
    atlas["ladder_steps"] = list(manifest.get("ladder_steps") or [])
    out: dict = {"model": model, "input_limit_chars_div_4": limit, "limit_note": SCOPE_LIMIT_NOTE,
                 "assembled": prompt is not None, "atlas": atlas}
    if prompt is None:
        out["refusal"] = {
            "status": status.status, "token_count": status.token_count,
            "omitted_paths": list(status.omitted_paths),
            "unassembled_required": list(status.unassembled_required),
            "atlas_overflowed": bool(status.atlas_overflowed),
        }
        return out
    stable_len = int(sp._SCOPE_STABLE_PREFIX_LEN.get() or 0)
    total = _measure(prompt, enc)
    out.update({
        "total": total,
        "stable_prefix": _measure(prompt[:stable_len], enc),
        "dynamic_tail": _measure(prompt[stable_len:], enc),
        "headroom_chars_div_4": limit - total["chars_div_4"],
    })
    return out


def measure(repo: pathlib.Path, *, scope_window: int | None = None) -> dict:
    from ouroboros.reviewer_slot_config import commit_triad_delivery
    from ouroboros.tools.review_file_pack import build_touched_file_pack, triad_pack_exclusions

    try:
        enc, tokenizer = _o200k(), "o200k_base from the local tiktoken cache"
    except (ImportError, TokenizerUnavailable) as exc:
        enc, tokenizer = None, f"o200k unavailable (cache-only): {exc}"
    entries = _staged_entries(repo)
    paths = [ep[1] for ep in entries]  # the --name-only list the triad packs
    deleted = [ep[1] for ep in entries if ep[0] == "D"]
    porcelain = _porcelain(repo)
    _require_index_is_worktree(porcelain)
    prefix = _governance_prefix(repo)

    def _pack(exclude: set[str], note: str) -> str:
        section, omitted = build_touched_file_pack(repo, paths, exclude_paths=exclude)
        if omitted:
            section += (f"\n\n⚠️ OMISSION NOTE: {len(omitted)} file(s) omitted from direct context: "
                        f"{', '.join(omitted)}")
        if note:
            section += f"\n\n{note}"
        return section

    excluded, note = triad_pack_exclusions(repo, paths, prefix_texts={
        rel: prefix[rel] for rel in ("docs/DEVELOPMENT.md", "docs/DESIGN.md", "docs/ARCHITECTURE.md")})
    before, after = _pack(set(), ""), _pack(excluded, note)
    per_file = {}
    for rel in paths:
        one, _ = build_touched_file_pack(repo, [rel])
        per_file[rel] = {**_measure(one, enc), "excluded": rel in excluded}
    # The scope pack's touched section (its own HEAD→index pair, current paths
    # and deleted paths split as the assembler splits them) before/after the
    # carrier cut — a SUB-number of the full scope input measured below — and
    # the advisory pack (its HEAD→working tree pair, paths resolved from the
    # porcelain as the run resolves them). The index IS the working tree —
    # checked above, and re-checked on the resolved path set — so both advisory
    # arms measure the one staged change.
    from ouroboros.tools import scope_review_pack as _sp
    from ouroboros.tools.review_file_pack import build_advisory_changed_context

    current = [p for p in paths if p not in deleted]
    ctx_paths = [p for p in current if not _sp._should_skip_current_touched_context(p)]
    skipped = [p for p in current if _sp._should_skip_current_touched_context(p)]
    carriers = _sp._carrier_span_only_paths(repo, ctx_paths, None)
    scope_before = _sp._render_touched_section(repo, ctx_paths, deleted, skipped, [])[0]
    scope_after = _sp._render_touched_section(
        repo, [p for p in ctx_paths if p not in carriers], deleted, skipped, [], carrier_span_only=carriers)[0]
    scope_model = _sp._sr()._get_scope_model()
    if scope_window is not None:
        window, window_evidence = int(scope_window), "--scope-window (operator-named)"
    else:
        window, window_evidence = _cached_window(scope_model)
    scope_full = {"window": window, "window_evidence": window_evidence,
                  **_scope_full(repo, scope_model, window, _scope_input_limit(scope_model, window), enc)}
    advisory_paths = [p for p in paths if p != "docs/ARCHITECTURE.md"]  # the run's exclude_paths
    advisory_before, _ = build_touched_file_pack(repo, advisory_paths)
    resolved, advisory_after, _ = build_advisory_changed_context(
        repo, changed_files_text=porcelain, exclude_paths={"docs/ARCHITECTURE.md"})
    resolved = [p for p in resolved if p != "docs/ARCHITECTURE.md"]
    if sorted(resolved) != sorted(advisory_paths):
        raise MeasuredCheckoutDirty(
            f"the advisory arm resolved {resolved} from the porcelain while the index names "
            f"{advisory_paths}; the checkout is not one staged change")
    panel_rows = _panel_rows(commit_triad_delivery())
    api_models = [row["model"] for row in panel_rows if row["receives_pack"]]
    zero_parts = _zero_diff_message(repo, prefix, paths)
    zero_tokens = _measure("".join(zero_parts.values()), enc)["chars_div_4"]
    fit: dict = {
        "units": FIT_UNITS,
        "panel_models": [row["model"] for row in panel_rows],
        "panel_rows": panel_rows,
        "api_pack_models": api_models,
        "zero_diff_message_chars_div_4": zero_tokens,
    }
    if api_models:
        limit, slots = _quorum_limit(api_models)
        headroom = limit - zero_tokens
        fit.update({
            "slots": slots,
            "quorum_input_limit_chars_div_4": limit,
            "headroom_after_zero_diff_message": headroom,
            "headroom_for_diff_before": headroom - _measure(before, enc)["chars_div_4"],
            "headroom_for_diff_after": headroom - _measure(after, enc)["chars_div_4"],
            # fit_triad_prompt sizes stable prefix + dynamic tail only: the head
            # and the user turn are billed input it never counts.
            "uncounted_by_fit_triad_prompt_chars_div_4": _measure(
                zero_parts["constitutional_head_preamble_plus_BIBLE"] + zero_parts["user_turn"], enc)["chars_div_4"],
        })
    else:
        fit["no_api_pack"] = NO_API_PACK_NOTE
    return {
        "repo": str(repo),
        "staged_paths": paths,
        "tokenizer": tokenizer,
        "touched_pack": {
            "before": _measure(before, enc),
            "after": _measure(after, enc),
            "excluded_paths": sorted(excluded),
            "exclusion_note": note,
            "per_file": per_file,
        },
        "scope_full": {
            **scope_full,
            # The touched section alone: one fragment of the input above, never the input.
            "scope_touched": {
                "before": _measure(scope_before, enc), "after": _measure(scope_after, enc),
                "carrier_span_only": list(carriers), "deleted_paths": list(deleted),
            },
        },
        "advisory_touched": {
            "before": _measure(advisory_before, enc), "after": _measure(advisory_after, enc),
            "paths": advisory_paths,  # both arms: the index list == the porcelain-resolved list
        },
        "governance_prefix": {
            "stable_prefix_total": _measure(prefix["stable_prefix"], enc),
            "parts": {
                "checklist_section_plus_archive": _measure(prefix["checklist_section"], enc),
                "docs/DEVELOPMENT.md": _measure(prefix["docs/DEVELOPMENT.md"], enc),
                "docs/DESIGN.md": _measure(prefix["docs/DESIGN.md"], enc),
                "docs/ARCHITECTURE.md": _measure(prefix["docs/ARCHITECTURE.md"], enc),
                "constitutional_head_preamble_plus_BIBLE": _measure(
                    zero_parts["constitutional_head_preamble_plus_BIBLE"], enc),
            },
        },
        "zero_diff_message": {
            "components": list(zero_parts),
            "total": _measure("".join(zero_parts.values()), enc),
            "parts": {name: _measure(text, enc) for name, text in zero_parts.items()},
        },
        "fit": fit,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo", required=True, help="checkout whose staged change is measured")
    parser.add_argument("--json", action="store_true", help="print the full JSON report only")
    parser.add_argument("--scope-window", type=int, default=None, metavar="N",
                        help="size the scope input cap on this reviewer window instead of the cache-only one")
    args = parser.parse_args(argv)
    try:
        report = measure(pathlib.Path(args.repo).resolve(), scope_window=args.scope_window)
    except MeasuredCheckoutDirty as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 0
    pack, fit = report["touched_pack"], report["fit"]
    print(f"staged paths: {len(report['staged_paths'])}; excluded: {pack['excluded_paths']}")
    for arm in ("before", "after"):
        m = pack[arm]
        print(f"touched pack {arm:6}: {m['chars']:>10,} chars  {m['chars_div_4']:>9,} chars/4  {m['o200k']!s:>9} o200k")
    for rel, m in sorted(pack["per_file"].items(), key=lambda kv: -(kv[1]["o200k"] or kv[1]["chars"])):
        flag = "CUT " if m["excluded"] else "keep"
        print(f"  {flag} {rel:40} {m['chars']:>10,} chars {m['o200k']!s:>9} o200k")
    scope = report["scope_full"]
    print(f"scope full input (real assembler; model {scope['model']}, window {scope['window']:,} — "
          f"{scope['window_evidence']}; cap {scope['input_limit_chars_div_4']:,} chars/4):")
    if scope["assembled"]:
        for part in ("stable_prefix", "dynamic_tail", "total"):
            m = scope[part]
            print(f"  {part:14} {m['chars']:>10,} chars  {m['chars_div_4']:>9,} chars/4  {m['o200k']!s:>9} o200k")
        print(f"  headroom under the cap: {scope['headroom_chars_div_4']:,} chars/4")
    else:
        r = scope["refusal"]
        print(f"  NOT assembled: {r['status']} at ~{r['token_count']:,} chars/4; unassembled required "
              f"{r['unassembled_required']}; atlas_overflowed={r['atlas_overflowed']}; omitted {r['omitted_paths']}")
    atlas = scope["atlas"]
    print(f"  atlas: status {atlas['status']}; selected {atlas['selected_count']} of {atlas['tracked_count']} tracked; "
          f"atlas {atlas['atlas_tokens']} + fixed {atlas['fixed_prompt_tokens']} chars/4; "
          f"unassembled required {atlas['unassembled_required']}")
    for step in atlas["ladder_steps"]:
        print(f"  ladder: {step.get('step')} tokens_after={step.get('tokens_after')} "
              f"deficit={step.get('deficit')} diff_only_files={step.get('diff_only_files')} "
              f"paths={step.get('paths') or step.get('diff_only_paths') or []}")
    print(f"  ({scope['limit_note']})")
    touched = scope["scope_touched"]
    for arm in ("before", "after"):
        m = touched[arm]
        print(f"  scope touched section (sub-number of the input above) {arm:6}: {m['chars']:>10,} chars  "
              f"{m['chars_div_4']:>9,} chars/4  {m['o200k']!s:>9} o200k")
    print(f"  scope carrier_span_only: {touched['carrier_span_only']}; deleted: {touched['deleted_paths']}")
    for arm in ("before", "after"):
        m = report["advisory_touched"][arm]
        print(f"advisory touched  {arm:6}: {m['chars']:>10,} chars  {m['chars_div_4']:>9,} chars/4  {m['o200k']!s:>9} o200k")
    print(f"governance corpus (one checkout: {report['repo']}), per api row, per round:")
    for name, m in report["governance_prefix"]["parts"].items():
        print(f"  {name:42} {m['chars']:>10,} chars {m['o200k']!s:>9} o200k")
    total = report["governance_prefix"]["stable_prefix_total"]
    print(f"  stable prefix total (without BIBLE head) {total['chars']:>10,} chars {total['o200k']!s:>9} o200k")
    zero = report["zero_diff_message"]
    print(f"zero-diff message ({' + '.join(zero['components'])}): "
          f"{zero['total']['chars']:,} chars  {zero['total']['chars_div_4']:,} chars/4  {zero['total']['o200k']!s} o200k")
    print("panel rows (delivery class as review.py decides it before fit_triad_prompt):")
    for row in fit["panel_rows"]:
        actor = f" via configured subagent {row['subagent_id']}" if row["subagent_id"] else ""
        print(f"  {row['model']:36} {row['route']}{actor}: "
              f"{'receives the api pack' if row['receives_pack'] else 'retrieves — no pack, outside the fit ladder'}")
    if "no_api_pack" in fit:
        print(fit["no_api_pack"])
    else:
        print(f"api pack rows {fit['api_pack_models']}: quorum input limit "
              f"{fit['quorum_input_limit_chars_div_4']:,} [{fit['units']}]")
        for model, slot in fit["slots"].items():
            print(f"  {model:36} window {slot['window']:>9,} — {slot['evidence']}; slot limit {slot['input_limit_chars_div_4']:,}")
        print(f"headroom after the zero-diff message: {fit['headroom_after_zero_diff_message']:,}; "
              f"for the pack + diff: before {fit['headroom_for_diff_before']:,} -> after {fit['headroom_for_diff_after']:,}")
        print(f"  (fit_triad_prompt sizes stable prefix + dynamic tail only; it does not count the "
              f"{fit['uncounted_by_fit_triad_prompt_chars_div_4']:,} chars/4 of constitutional head + user turn)")
    print(f"tokenizer: {report['tokenizer']}")
    print(pack["exclusion_note"] or "(no exclusion note)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
