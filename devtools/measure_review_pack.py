#!/usr/bin/env python3
"""Measure the commit triad packet for one staged change — offline, $0.

Reports, for the checkout at ``--repo`` (its INDEX is the reviewed change; every
governance corpus it selects — BIBLE.md, the checklist section + archive, and
whatever ``tools/governance_context.py`` inlines or navigates for these touched
paths — is read from that one checkout):

* the triad touched-file pack BEFORE and AFTER the disclosed pack exclusions
  (``review_file_pack.triad_pack_exclusions``: span-only release carriers on a
  VERSION-staged commit, governance docs byte-identical to the inlined prefix);
* the governance context the triad packet carries, tier by tier: the byte-stable
  prefix (checklist section + archive + tier-1 inline rules), the change-class
  selection and the navigation maps that open the dynamic tail, and the
  constitutional head (preamble + BIBLE.md) each api row receives per round;
* the ZERO-DIFF message — everything an api row receives before the first pack
  or diff byte, serialized as ``_multi_model_review_async`` sends it: the
  constitutional head + the stable prefix + the governance tail and dynamic
  scaffolding rendered with an empty pack and diff + the fixed user turn — the
  quorum input limit of the rows that RECEIVE the packet (resolved by
  ``row_plan_retrieves``, exactly as ``review`` filters them before
  ``fit_triad_prompt``; native and session rows retrieve with their own tools
  and never constrain the ladder), and the
  headroom that limit leaves for the pack + diff. A panel whose every row
  retrieves gets the explicit "no API pack is assembled for this panel" instead
  of a number.

Units: chars, the host's own ``utils.estimate_tokens`` (chars/4 — the unit
``review_admission.fit_triad_prompt`` compares against the quorum limit, so every
limit/headroom figure is in it) and tiktoken ``o200k_base`` (not Anthropic's
tokenizer) are printed side by side and never conflated.

One change, checked: every arm reads the checkout's INDEX as the reviewed change
while the packs read working-tree text. Those coincide only when the index IS
the working tree, so a checkout with an unstaged edit or an untracked file is
refused with the typed :class:`MeasuredCheckoutDirty` (exit 2) instead of
measured across two changes.

Offline by construction: reviewer windows are read from the Capability Evidence
CACHE only (``capability_evidence.probe(allow_fetch=False)`` under
``$OUROBOROS_DATA_DIR``) — an unknown route is disclosed and sized at the fit
ladder's own full-window default, never probed or persisted; the o200k BPE is
served from the local tiktoken cache only (``TIKTOKEN_CACHE_DIR``) and a missing
BPE is a typed, disclosed miss, never a download. Nothing is dispatched.

Usage::

    python devtools/measure_review_pack.py --repo /path/to/checkout [--json]
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
NO_API_PACK_NOTE = (
    "no API pack is assembled for this panel: every configured row retrieves the subject with its "
    "own tools (session and native API rows), so there is no quorum input "
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
    """``(status, current_path, source_path)`` per staged entry, through the host's
    own ``parse_git_name_status`` — so a rename names both ends and a staged
    deletion stays a ``D`` entry instead of a current path that resolves to
    nothing."""
    from ouroboros.tools.review_file_pack import parse_git_name_status

    out = subprocess.run(
        ["git", "diff", "--cached", "--name-status"], cwd=str(repo),
        check=True, capture_output=True, text=True,
    ).stdout
    return parse_git_name_status(out)


def _porcelain(repo: pathlib.Path) -> str:
    """``git status --porcelain`` — the worktree state the one-change invariant reads."""
    return subprocess.run(
        ["git", "status", "--porcelain"], cwd=str(repo), check=True, capture_output=True, text=True).stdout


def _require_index_is_worktree(porcelain: str) -> None:
    """The one-change invariant every arm rests on (module docstring): an entry
    with a worktree-column status — an unstaged edit (``XM``), an untracked file
    (``??``) — means the packs would read text the index never staged. Typed
    refusal, never a silently cross-arm number."""
    dirty = [line for line in porcelain.splitlines() if len(line) > 1 and line[1] != " "]
    if dirty:
        raise MeasuredCheckoutDirty(
            "the checkout's index is not its working tree — stage or stash these before measuring: "
            + "; ".join(dirty))


def _panel_rows(plan: dict) -> list[dict]:
    """Use commit_triad_delivery's resolved delivery, including direct native
    API rows. Only packet recipients constrain the fit ladder and get a pack."""
    from ouroboros.reviewer_slot_config import row_plan_retrieves

    actors = list(plan.get("subagent_ids") or [])
    rows = []
    for i, (model, route) in enumerate(zip(plan["models"], plan["routes"])):
        actor = str(actors[i] if i < len(actors) else "")
        rows.append({"model": model, "route": str(getattr(route, "value", route)), "subagent_id": actor,
                     "receives_pack": not row_plan_retrieves(plan, i)})
    return rows


def _checklist_section(repo: pathlib.Path) -> str:
    """``review._load_checklist_section()`` (body layer) read from the TARGET checkout.

    The runtime reads the checklist from its own REPO_ROOT (a frozen contract);
    this measurer measures one checkout, so the same layered sections + archive
    come from ``repo`` (``review_helpers.load_checklist_layers`` pointed at it)."""
    from ouroboros.tools.review_helpers import load_checklist_layers

    section = load_checklist_layers("body", repo / "docs" / "CHECKLISTS.md")
    archive = (repo / "docs" / "CHECKLISTS_ARCHIVE.md").read_text(encoding="utf-8").strip()
    return f"{section}\n\n{archive}" if archive else section


def _governance_usable_window(models: list[str]) -> int:
    """The usable input window the packet's governance share is taken against,
    by ``review._triad_governance_usable_window``'s own arithmetic on cache-only
    windows (the runtime keys it by slot id; this measurer has models only)."""
    from ouroboros.tools import review as _rv

    usable: dict[str, int] = {}
    for model in models:
        window, _evidence = _cached_window(model)
        output_reserve, tokenizer_margin = _rv.window_scaled_reserves(
            window, output_reserve=_rv._review_output_budget(), tokenizer_margin=50_000)
        usable[model] = max(0, int(window) - int(output_reserve) - int(tokenizer_margin))
    return int(_rv._quorum_input_token_limit(list(usable), usable)) if usable else 0


def _governance_prefix(
    repo: pathlib.Path, touched_paths: list[str], api_models: list[str],
) -> dict[str, str]:
    """The governance regions exactly as `_prepare_unified_review` assembles them.

    ``governance_context`` is the ONE decision about which rules a packet
    carries: tier 1 rides the cache-marked stable prefix, the change-class
    selection and the navigation maps open the dynamic tail. A panel with no api
    row assembles no packet, so it asks for no governance — the same branch the
    runtime takes. The principal checklist and repository documents are read
    from ``repo``; the shared ownership section follows the runtime loader and
    comes from the executing checkout. Cross-checkout measurements therefore
    retain that disclosed mixed-source boundary."""
    from ouroboros.tools import review
    from ouroboros.tools.governance_context import GovernanceContext, governance_context

    checklist = _checklist_section(repo)
    governance = governance_context(
        repo,
        surface="triad",
        touched_paths=touched_paths,
        usable_window_tokens=_governance_usable_window(api_models),
        delivery="packet",
        checklist_section_text=checklist,
        already_inline=("BIBLE.md", "docs/CHECKLISTS_ARCHIVE.md"),
    ) if api_models else GovernanceContext()
    stable = review._REVIEW_PROMPT_TEMPLATE_STABLE.format(
        preamble=review.review_preamble("body"),
        critical_calibration=review.CRITICAL_FINDING_CALIBRATION,
        json_contract=review.REVIEW_JSON_ARRAY_CONTRACT,
        anti_pattern_lock_guard=review.anti_pattern_lock_guard("body"),
        checklist_section=checklist,
    ) + (f"\n{governance.stable_inline}\n" if governance.stable_inline.strip() else "")
    tail = "\n\n".join(
        part for part in (governance.selected_inline, governance.navigation) if part.strip())
    return {
        "stable_prefix": stable,
        "checklist_section": checklist,
        "tier_1_inline_rules": governance.stable_inline,
        "change_class_selection": governance.selected_inline,
        "navigation_maps": governance.navigation,
        "governance_tail": tail,
        "manifest": governance.manifest,
        "inline_whole_documents": governance.inline_whole_documents,
    }


def _constitutional_head(repo: pathlib.Path) -> str:
    """`_multi_model_review_async`'s stable head: the preamble + BIBLE.md from ``repo``."""
    from ouroboros.tools import review_multi_model as mm
    from ouroboros.tools.review_helpers import load_governance_doc

    bible = load_governance_doc(repo, "BIBLE.md", on_missing="explicit")
    return mm._CONSTITUTIONAL_PREAMBLE + "### BIBLE.md (Full Text)\n\n" + bible + "\n\n---\n\n## REVIEW INSTRUCTIONS\n\n"


def _zero_diff_message(repo: pathlib.Path, prefix: dict[str, str], paths: list[str]) -> dict[str, str]:
    """Every byte an api row receives BEFORE the pack and the diff, part by part in
    wire order: the constitutional head (prepended to the system content), the
    stable prefix, the governance tail plus the dynamic scaffolding
    (`_REVIEW_PROMPT_TEMPLATE_DYNAMIC` with an empty pack and diff: the goal
    section of an empty commit message, no scope, no rebuttal, no history, the
    changed-files list) and the fixed user turn."""
    from ouroboros.tools import review
    from ouroboros.tools.review_helpers import build_goal_section, build_scope_section

    tail = prefix["governance_tail"]
    dynamic = (f"{tail}\n\n" if tail else "") + review._REVIEW_PROMPT_TEMPLATE_DYNAMIC.format(
        goal_section=build_goal_section("", "", ""),
        scope_section=build_scope_section(""),
        current_files_section="",
        rebuttal_section="",
        review_history_section="",
        diff_text="",
        changed_files="\n".join(paths),
        task_evidence_section="",  # No task trace is part of this zero-diff baseline.
    )
    return {
        "constitutional_head_preamble_plus_BIBLE": _constitutional_head(repo),
        "stable_prefix": prefix["stable_prefix"],
        # `_assemble_prompt` joins stable + "\n" + dynamic; the separator rides
        # with the tail, and the governance tail opens that dynamic half.
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


def measure(repo: pathlib.Path) -> dict:
    from ouroboros.reviewer_slot_config import commit_triad_delivery
    from ouroboros.tools.review_file_pack import build_touched_file_pack, triad_pack_exclusions

    try:
        enc, tokenizer = _o200k(), "o200k_base from the local tiktoken cache"
    except (ImportError, TokenizerUnavailable) as exc:
        enc, tokenizer = None, f"o200k unavailable (cache-only): {exc}"
    entries = _staged_entries(repo)
    paths = [ep[1] for ep in entries]  # the --name-only list the triad packs
    porcelain = _porcelain(repo)
    _require_index_is_worktree(porcelain)
    # The governance selection is change-relative and packet-only, so the panel
    # is resolved first: an all-retrieving panel assembles neither.
    panel_rows = _panel_rows(commit_triad_delivery())
    api_models = [row["model"] for row in panel_rows if row["receives_pack"]]
    prefix = _governance_prefix(repo, paths, api_models)

    def _pack(exclude: set[str], note: str) -> str:
        section, omitted = build_touched_file_pack(repo, paths, exclude_paths=exclude)
        if omitted:
            section += (f"\n\n⚠️ OMISSION NOTE: {len(omitted)} file(s) omitted from direct context: "
                        f"{', '.join(omitted)}")
        if note:
            section += f"\n\n{note}"
        return section

    excluded, note = triad_pack_exclusions(
        repo, paths, prefix_texts=dict(prefix["inline_whole_documents"]))
    before, after = _pack(set(), ""), _pack(excluded, note)
    per_file = {}
    for rel in paths:
        one, _ = build_touched_file_pack(repo, [rel])
        per_file[rel] = {**_measure(one, enc), "excluded": rel in excluded}
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
        "governance_context": {
            "stable_prefix_total": _measure(prefix["stable_prefix"], enc),
            "governance_tail_total": _measure(prefix["governance_tail"], enc),
            "manifest": prefix["manifest"],
            "parts": {
                "checklist_section_plus_archive": _measure(prefix["checklist_section"], enc),
                "tier_1_inline_rules": _measure(prefix["tier_1_inline_rules"], enc),
                "change_class_selection": _measure(prefix["change_class_selection"], enc),
                "navigation_maps": _measure(prefix["navigation_maps"], enc),
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
    args = parser.parse_args(argv)
    try:
        report = measure(pathlib.Path(args.repo).resolve())
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
    print(f"governance context (one checkout: {report['repo']}), per api row, per round:")
    for name, m in report["governance_context"]["parts"].items():
        print(f"  {name:42} {m['chars']:>10,} chars {m['o200k']!s:>9} o200k")
    total = report["governance_context"]["stable_prefix_total"]
    print(f"  stable prefix total (without BIBLE head) {total['chars']:>10,} chars {total['o200k']!s:>9} o200k")
    tail = report["governance_context"]["governance_tail_total"]
    print(f"  change-relative governance tail          {tail['chars']:>10,} chars {tail['o200k']!s:>9} o200k")
    for row in report["governance_context"]["manifest"]:
        print(f"  {str(row.get('path') or '?'):42} tier {row.get('tier')} {row.get('disposition')} "
              f"{int(row.get('chars') or 0):>9,} chars — {row.get('reason')}")
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
