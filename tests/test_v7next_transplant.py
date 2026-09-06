"""The Ф0 transplant tool: mechanical D18/D33 module-handle extraction with proof.

Three REAL cases drive the fixtures:

* ``supervisor/queue.py`` -> ``queue_snapshot`` leaf (handle ``_queue``): two
  symbols stable since the v7 cut are byte-exact against the reference leaf;
  two drifted symbols prove the proof property and the declared-set
  recalculation workflow.
* ``ouroboros/loop.py`` -> ``loop_messages`` leaf (handle ``_loop``): the
  declared name ``_record_owner_directive`` is ALSO a moved symbol — its own
  ``def`` stays plain while the sibling reads it through the handle, so parent
  monkeypatching keeps intercepting (the v7 re-export pattern).
* ``supervisor/git_ops.py`` -> ``git_ops_remotes`` leaf (handle ``_go``):
  three stable symbols, two of them declared-and-moved.

Byte-exact comparisons against the v7_wip reference tree are skipped when the
sibling worktree is absent (set OUROBOROS_V7_REF to point elsewhere); every
proof-property and fail-closed test runs from this repository alone.
"""

from __future__ import annotations

import ast
import importlib.util
import os
import pathlib
import subprocess
import sys
import textwrap

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
TOOL_PATH = REPO / "scripts" / "v7next_transplant.py"
_spec = importlib.util.spec_from_file_location("v7next_transplant", TOOL_PATH)
tp = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = tp
_spec.loader.exec_module(tp)

V7_REF = pathlib.Path(os.environ.get(
    "OUROBOROS_V7_REF",
    str(pathlib.Path.home() / "ouro" / "subagent_worktrees" / "v7_wip")))
needs_ref = pytest.mark.skipif(
    not (V7_REF / "supervisor" / "queue_snapshot.py").exists(),
    reason="v7_wip reference worktree not present (set OUROBOROS_V7_REF)")


def _read(path: pathlib.Path) -> str:
    return path.read_text(encoding="utf-8")


def _pinned_upstream(base_sha: str, rel_path: str) -> str:
    """The exact pre-split monolith bytes the real extraction ran against.

    Empty when the object is unreachable (a shallow clone without that commit)
    or when git itself is not on PATH, and the probes that need it SKIP. Both
    reach the same honest marker: catching only a nonzero exit code left the
    no-git host failing COLLECTION on OSError at import time, which reads as a
    broken suite rather than as the disclosed gap this corpus has. The rest of
    the file runs from this repository alone. The corpus is the probe's INPUT: each
    real case feeds it to the tool and compares the tool's output against the
    landed leaf. Reconstructing the corpus by inverse-normalizing that same
    leaf — the earlier fallback — made the tool prove its own transformation
    against its own output: the byte comparisons could not fail, `transplant`
    could not report a drifted body, and the declared-set recalculation loop
    had nothing to recalculate. A green run then proved nothing about the
    transplant, and no marker said so.
    """
    try:
        done = subprocess.run(
            ["git", "-C", str(REPO), "show", f"{base_sha}:{rel_path}"],
            capture_output=True, text=True)
    except OSError:
        return ""
    return done.stdout if done.returncode == 0 else ""


def _span_text(source: str, symbol: str) -> str:
    return tp.extract_spans(source, [symbol])[symbol].text


def _recalculate(upstream: str, symbols, declared, handle: str, parent_module: str,
                 preamble: str, max_rounds: int = 4):
    """The campaign's declared-set recalculation loop, driven by tool reports."""
    declared = frozenset(declared)
    rounds = []
    for _ in range(max_rounds):
        try:
            result = tp.transplant(upstream, list(symbols), declared, handle,
                                   parent_module=parent_module, preamble=preamble)
            return result, declared, preamble, rounds
        except tp.TransplantError as exc:
            rounds.append(exc)
            if exc.kind == "unresolved_names":
                suggestions = exc.details["suggestions"]
                unknown = {n for n, s in suggestions.items() if s["kind"] == "unknown"}
                assert not unknown, f"unresolvable names: {unknown}"
                declared |= {n for n, s in suggestions.items() if s["kind"] == "parent_global"}
                imports = sorted({s["import"] for s in suggestions.values()
                                  if s["kind"] == "parent_import"})
                if imports:
                    preamble = preamble.rstrip("\n") + "\n" + "\n".join(imports) + "\n"
                continue
            if exc.kind == "unused_declared":
                declared -= set(exc.details["unused"])
                continue
            raise
    pytest.fail("declared-set recalculation did not converge")


# ---------------------------------------------------------------------------
# real case 1: supervisor/queue.py -> queue_snapshot leaf (_queue handle)
#
# The F2.2 lane landed the queue_snapshot/queue_timeouts split, so the LIVE
# queue.py no longer carries these defs. The probe keeps its real-case shape on
# the PRE-SPLIT monolith bytes of the lane base (the D10 recipe for probes
# pinned to live monoliths), and skips when that object is unreachable.

_QUEUE_BASE_SHA = "2878560ed298c4173e65068f16b6d09e672ba19f"
QUEUE_UPSTREAM = _pinned_upstream(_QUEUE_BASE_SHA, "supervisor/queue.py")
needs_queue_corpus = pytest.mark.skipif(
    not QUEUE_UPSTREAM,
    reason=f"pre-split supervisor/queue.py bytes not reachable via "
           f"`git show {_QUEUE_BASE_SHA}`")

# The v7 ledger's declared set for supervisor/queue_snapshot.py (D18).
QUEUE_LEDGER_DECLARED = frozenset({
    "ACCEPTANCE_FENCES", "DRIVE_ROOT", "PENDING", "RUNNING",
    "_queue_lock", "append_jsonl", "atomic_write_text", "enqueue_task",
})

QUEUE_PREAMBLE = '''"""Queue snapshot leaf (test preamble)."""

from __future__ import annotations

import datetime
import json
import logging
import pathlib
import time
from typing import Optional


def _queue():
    """The parent module, read at call time."""
    from supervisor import queue

    return queue


log = logging.getLogger(__name__)
'''


@needs_ref
@needs_queue_corpus
def test_queue_stable_symbols_match_the_v7_leaf_byte_for_byte():
    """Symbols whose upstream bodies did not change since the v7 cut transform
    into exactly the reference leaf's spans."""
    result = tp.transplant(
        QUEUE_UPSTREAM, ["_kept_service_pids", "parse_iso_to_ts"],
        {"DRIVE_ROOT"}, "_queue", parent_module="supervisor.queue",
        preamble=QUEUE_PREAMBLE)
    ref_leaf = _read(V7_REF / "supervisor" / "queue_snapshot.py")
    for symbol in ("_kept_service_pids", "parse_iso_to_ts"):
        assert _span_text(result.leaf_source, symbol) == _span_text(ref_leaf, symbol), symbol


@needs_queue_corpus
def test_queue_drifted_symbols_need_a_recalculated_declared_set():
    """The ledger's declared set no longer covers the drifted upstream bodies:
    the tool fails closed naming the new dependencies, classifies each one, and
    the mechanical recalculation loop converges to a proven transplant."""
    symbols = ["_kept_service_pids", "persist_queue_snapshot",
               "parse_iso_to_ts", "restore_pending_from_snapshot"]
    with pytest.raises(tp.TransplantError) as excinfo:
        tp.transplant(QUEUE_UPSTREAM, symbols, QUEUE_LEDGER_DECLARED, "_queue",
                      parent_module="supervisor.queue", preamble=QUEUE_PREAMBLE)
    exc = excinfo.value
    assert exc.kind == "unresolved_names"
    flat = {n for names in exc.details["unresolved"].values() for n in names}
    assert {"QUEUE_SNAPSHOT_PATH", "BUDGET_ROOT_FENCES", "utc_now_iso",
            "restore_terminalization_retry_rows", "sort_pending",
            "QUEUE_SEQ_COUNTER_REF"} <= flat
    sugg = exc.details["suggestions"]
    # rebindable parent state, only ever assigned under `global` in init():
    assert sugg["QUEUE_SNAPSHOT_PATH"]["kind"] == "parent_global"
    assert sugg["sort_pending"]["kind"] == "parent_global"
    assert sugg["QUEUE_SEQ_COUNTER_REF"]["kind"] == "parent_global"
    assert sugg["restore_terminalization_retry_rows"] == {
        "kind": "parent_import",
        "import": "from supervisor.task_admission import restore_terminalization_retry_rows",
        "hint": sugg["restore_terminalization_retry_rows"]["hint"],
    }
    assert sugg["utc_now_iso"]["kind"] == "parent_import"

    result, declared, _preamble, rounds = _recalculate(
        QUEUE_UPSTREAM, symbols, QUEUE_LEDGER_DECLARED, "_queue",
        "supervisor.queue", QUEUE_PREAMBLE)
    assert len(rounds) >= 1
    assert QUEUE_LEDGER_DECLARED <= declared
    assert {"QUEUE_SNAPSHOT_PATH", "sort_pending", "QUEUE_SEQ_COUNTER_REF"} <= declared
    proof = result.proof
    assert proof["ok"]
    for symbol in symbols:
        entry = proof["symbols"][symbol]
        assert entry["ast_equal"] and entry["tokens_equal"] and entry["byte_identical"], symbol
    assert proof["unread_declared"] == []
    # the drifted body now reads the drifted dependencies through the handle
    drifted = _span_text(result.leaf_source, "restore_pending_from_snapshot")
    assert "_queue().QUEUE_SNAPSHOT_PATH" in drifted
    assert "_queue().sort_pending()" in drifted
    # and an independent re-verification agrees (the --check path, in-process)
    recheck = tp.verify_transplant(QUEUE_UPSTREAM, result.leaf_source, symbols,
                                   declared, "_queue")
    assert recheck["ok"]


# ---------------------------------------------------------------------------
# real case 2: ouroboros/loop.py -> loop_messages leaf (_loop handle)
#
# The D01 lane landed the L-B split, so ouroboros/loop.py no longer carries
# these defs. The probe keeps its real-case shape on the PRE-SPLIT monolith
# bytes of the lane base (the D10 lane's recipe for probes pinned to live
# monoliths), and skips when that object is unreachable.

_D01_BASE = "a56bb76a38ca92b39a659b4b6e63e07a76243a4f"
LOOP_UPSTREAM = _pinned_upstream(_D01_BASE, "ouroboros/loop.py")
needs_loop_corpus = pytest.mark.skipif(
    not LOOP_UPSTREAM,
    reason=f"pre-split ouroboros/loop.py bytes not reachable via "
           f"`git show {_D01_BASE}`")

LOOP_PREAMBLE = '''"""Loop messages leaf (test preamble)."""

from __future__ import annotations

import json
from typing import Any, Dict, List


def _loop():
    """The parent loop module, read at call time."""
    from ouroboros import loop

    return loop
'''


@needs_loop_corpus
def test_loop_declared_name_that_is_also_a_moved_symbol():
    """`_record_owner_directive` moves into the leaf AND stays in the declared
    set: its own def is emitted verbatim while `_initialize_owner_directives`
    reads it through `_loop()` — patching the parent keeps intercepting."""
    result = tp.transplant(
        LOOP_UPSTREAM, ["_record_owner_directive", "_initialize_owner_directives"],
        {"_record_owner_directive"}, "_loop", parent_module="ouroboros.loop",
        preamble=LOOP_PREAMBLE)
    assert len(result.rewrites) == 1
    assert result.rewrites[0].symbol == "_initialize_owner_directives"
    init_span = _span_text(result.leaf_source, "_initialize_owner_directives")
    assert "_loop()._record_owner_directive(" in init_span
    record_span = _span_text(result.leaf_source, "_record_owner_directive")
    assert record_span == _span_text(LOOP_UPSTREAM, "_record_owner_directive")
    assert result.proof["ok"]


@needs_ref
@needs_loop_corpus
def test_loop_stable_symbols_match_the_v7_leaf_byte_for_byte():
    result = tp.transplant(
        LOOP_UPSTREAM, ["_record_owner_directive", "_initialize_owner_directives"],
        {"_record_owner_directive"}, "_loop", parent_module="ouroboros.loop",
        preamble=LOOP_PREAMBLE)
    ref_leaf = _read(V7_REF / "ouroboros" / "loop_messages.py")
    for symbol in ("_record_owner_directive", "_initialize_owner_directives"):
        assert _span_text(result.leaf_source, symbol) == _span_text(ref_leaf, symbol), symbol


# ---------------------------------------------------------------------------
# real case 3: supervisor/git_ops.py -> git_ops_remotes leaf (_go handle)
#
# The D10 lane landed the G1 split, so the LIVE git_ops.py is now the facade
# (the three probe symbols are re-exports there, not spans). The corpus this
# probe transplants is the pre-split monolith, pinned at the lane's base SHA —
# the exact bytes the real extraction ran against.

_GO_BASE_SHA = "a56bb76a38ca92b39a659b4b6e63e07a76243a4f"
GO_UPSTREAM = _pinned_upstream(_GO_BASE_SHA, "supervisor/git_ops.py")
needs_go_corpus = pytest.mark.skipif(
    not GO_UPSTREAM,
    reason=f"pre-split supervisor/git_ops.py bytes not reachable via "
           f"`git show {_GO_BASE_SHA}`")

# Ledger set minus BRANCH_DEV: only push_to_remote reads it, and that body
# drifted upstream, so this fixture moves the three stable symbols.
GO_DECLARED = frozenset({
    "REPO_DIR", "_configure_credential_helper", "_has_remote",
    "configure_remote", "ensure_official_update_remote", "git_capture",
})

GO_PREAMBLE = '''"""Git remotes leaf (test preamble)."""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple


def _go():
    """The parent module, read at call time."""
    from supervisor import git_ops

    return git_ops


log = logging.getLogger("supervisor.git_ops")
'''

GO_SYMBOLS = ["configure_remote", "configure_personal_remote", "_configure_credential_helper"]


@needs_go_corpus
def test_git_ops_declared_and_moved_symbols_prove():
    result = tp.transplant(GO_UPSTREAM, GO_SYMBOLS, GO_DECLARED, "_go",
                           parent_module="supervisor.git_ops", preamble=GO_PREAMBLE)
    assert result.proof["ok"]
    assert result.proof["unread_declared"] == []
    # configure_remote is itself moved, yet configure_personal_remote reads it
    # through the handle (re-export pattern), same for _configure_credential_helper
    personal = _span_text(result.leaf_source, "configure_personal_remote")
    assert "_go().configure_remote(" in personal
    remote = _span_text(result.leaf_source, "configure_remote")
    assert "_go()._configure_credential_helper(" in remote
    assert "_go()._has_remote(" in remote


@needs_go_corpus
def test_the_f_string_deferred_git_ops_spans_transplant():
    """The two G1 rows the f-string gate deferred (ledger D10 entry 2:
    ``safe_restart`` / ``prepare_managed_update``, whose messages read the
    rebindable BRANCH_DEV/BRANCH_STABLE inside f-strings). The declared set is
    the tool's own recalculation, so this pins the report loop too."""
    result, declared, _preamble, _rounds = _recalculate(
        GO_UPSTREAM, ["safe_restart"], set(), "_go", "supervisor.git_ops",
        GO_PREAMBLE)
    span = _span_text(result.leaf_source, "safe_restart")
    assert {"BRANCH_DEV", "BRANCH_STABLE"} <= declared
    assert 'return False, f"Failed checkout {_go().BRANCH_DEV}: {err}"' in span
    assert 'return True, f"OK: fell back to {_go().BRANCH_STABLE}"' in span
    assert result.proof["ok"] and result.proof["unread_declared"] == []

    result2, declared2, _p2, _r2 = _recalculate(
        GO_UPSTREAM, ["prepare_managed_update"], set(), "_go",
        "supervisor.git_ops", GO_PREAMBLE)
    span2 = _span_text(result2.leaf_source, "prepare_managed_update")
    assert "BRANCH_DEV" in declared2
    assert 'f"Managed updates require the local {_go().BRANCH_DEV!r} branch."' in span2
    assert result2.proof["ok"] and result2.proof["unread_declared"] == []


@needs_ref
@needs_go_corpus
def test_git_ops_stable_symbols_match_the_v7_leaf_byte_for_byte():
    result = tp.transplant(GO_UPSTREAM, GO_SYMBOLS, GO_DECLARED, "_go",
                           parent_module="supervisor.git_ops", preamble=GO_PREAMBLE)
    ref_leaf = _read(V7_REF / "supervisor" / "git_ops_remotes.py")
    for symbol in GO_SYMBOLS:
        assert _span_text(result.leaf_source, symbol) == _span_text(ref_leaf, symbol), symbol


# ---------------------------------------------------------------------------
# synthetic corpus: scope precision, byte preservation, fail-closed behavior

SYN = textwrap.dedent('''\
    """Synthetic upstream."""
    import functools
    import json

    PENDING = []
    RUNNING = {}
    LIMIT = 5

    def helper(x):
        return x + 1

    def uses(x):
        with_lock = PENDING
        return json.dumps([with_lock, RUNNING, helper(x)])

    def shadows(PENDING, flag=True):
        RUNNING = "local"
        data = [PENDING for PENDING in range(3)]
        def inner():
            return RUNNING
        return PENDING, RUNNING, data, inner, LIMIT

    def strings_and_comments():
        # PENDING in a comment stays a comment
        s = "PENDING and RUNNING in a string"
        return s, PENDING

    def kwarg_positions(obj_attr):
        return dict(PENDING=PENDING, RUNNING=2), obj_attr.PENDING

    def mystery():
        return NEVER_DEFINED

    class Widget:
        kind = "w"
        def total(self):
            return LIMIT + len(PENDING)

    @functools.lru_cache(maxsize=None)
    def cached():
        return LIMIT

    async def fetch():
        return RUNNING

    A = B = []
    ''')

SYN_PRE = '"""Leaf."""\n\nfrom __future__ import annotations\n\nimport functools\nimport json\n'


def _syn(symbols, declared, preamble=SYN_PRE):
    return tp.transplant(SYN, symbols, declared, "_up", parent_module="synmod",
                         preamble=preamble)


def test_shadows_are_never_rewritten():
    result = _syn(["shadows"], {"LIMIT"})
    span = _span_text(result.leaf_source, "shadows")
    assert "_up().PENDING" not in span      # parameter and comprehension target
    assert "_up().RUNNING" not in span      # function local + closure free var
    assert "return PENDING, RUNNING, data, inner, _up().LIMIT" in span
    assert result.proof["ok"]


def test_module_reads_rewritten_strings_and_comments_untouched():
    result = _syn(["uses", "strings_and_comments"], {"PENDING", "RUNNING", "helper"})
    uses = _span_text(result.leaf_source, "uses")
    assert "with_lock = _up().PENDING" in uses
    assert "[with_lock, _up().RUNNING, _up().helper(x)]" in uses
    strings = _span_text(result.leaf_source, "strings_and_comments")
    assert '"PENDING and RUNNING in a string"' in strings
    assert "# PENDING in a comment stays a comment" in strings
    assert "return s, _up().PENDING" in strings


def test_keyword_names_and_attribute_accesses_untouched():
    result = _syn(["kwarg_positions"], {"PENDING"})
    span = _span_text(result.leaf_source, "kwarg_positions")
    assert "dict(PENDING=_up().PENDING, RUNNING=2)" in span
    assert "obj_attr.PENDING" in span
    assert "obj_attr._up()" not in span


def test_class_methods_rewritten_class_attribute_untouched():
    result = _syn(["Widget"], {"LIMIT", "PENDING"})
    span = _span_text(result.leaf_source, "Widget")
    assert "return _up().LIMIT + len(_up().PENDING)" in span
    assert 'kind = "w"' in span


def test_decorated_and_async_symbols_move_with_bytes_preserved():
    result = _syn(["cached", "fetch"], {"LIMIT", "RUNNING"})
    cached = _span_text(result.leaf_source, "cached")
    assert cached.startswith("@functools.lru_cache(maxsize=None)\n")
    assert "return _up().LIMIT" in cached
    assert "return _up().RUNNING" in _span_text(result.leaf_source, "fetch")


def test_unresolved_names_fail_closed_with_classified_suggestions():
    with pytest.raises(tp.TransplantError) as excinfo:
        tp.transplant(SYN, ["uses", "mystery"], set(), "_up", parent_module="synmod")
    exc = excinfo.value
    assert exc.kind == "unresolved_names"
    assert set(exc.details["unresolved"]["uses"]) == {"PENDING", "RUNNING", "helper", "json"}
    assert exc.details["unresolved"]["mystery"] == ["NEVER_DEFINED"]
    sugg = exc.details["suggestions"]
    assert sugg["PENDING"]["kind"] == "parent_global"
    assert sugg["helper"]["kind"] == "parent_global"
    assert sugg["json"] == {"kind": "parent_import", "import": "import json",
                            "hint": sugg["json"]["hint"]}
    assert sugg["NEVER_DEFINED"]["kind"] == "unknown"


def test_unused_declared_names_fail_closed():
    with pytest.raises(tp.TransplantError) as excinfo:
        _syn(["uses"], {"PENDING", "RUNNING", "helper", "LIMIT"})
    assert excinfo.value.kind == "unused_declared"
    assert excinfo.value.details["unused"] == ["LIMIT"]


def test_global_rebinding_of_declared_name_fails_closed():
    src = SYN + "\ndef bump():\n    global LIMIT\n    LIMIT = LIMIT + 1\n"
    with pytest.raises(tp.TransplantError) as excinfo:
        tp.transplant(src, ["bump"], {"LIMIT"}, "_up", parent_module="synmod",
                      preamble=SYN_PRE)
    assert excinfo.value.kind == "violation"
    assert "global LIMIT" in excinfo.value.message


def test_global_rebinding_of_unmoved_state_fails_closed():
    src = SYN + "\ndef flip():\n    global RUNNING\n    RUNNING = {}\n"
    with pytest.raises(tp.TransplantError) as excinfo:
        tp.transplant(src, ["flip"], set(), "_up", parent_module="synmod", preamble=SYN_PRE)
    assert excinfo.value.kind == "violation"
    assert "did not move" in excinfo.value.message


def test_import_time_reads_fail_closed():
    for extra, symbol in [
        ("\nCONST = LIMIT + 1\n", "CONST"),                       # module level
        ("\ndef defaulted(x=LIMIT):\n    return x\n", "defaulted"),  # default argument
        ("\nclass Bad:\n    size = LIMIT\n", "Bad"),              # class body
    ]:
        with pytest.raises(tp.TransplantError) as excinfo:
            tp.transplant(SYN + extra, [symbol], {"LIMIT"}, "_up",
                          parent_module="synmod", preamble=SYN_PRE)
        assert excinfo.value.kind == "violation", symbol
        assert "import-time read" in excinfo.value.message, symbol


SYN_FSTR = SYN + '\ndef show():\n    return f"limit={LIMIT}"\n'


def test_fstring_reads_of_declared_names_are_rewritten_and_proved():
    """F5 tool row: a declared name read inside an f-string is an ordinary
    call-time read. The literal halves — including a nested string subscript
    that spells the same name — stay byte-identical, and the format spec is
    an expression like any other."""
    src = SYN + (
        '\ndef show(k, d):\n'
        '    return f"limit={LIMIT} {k!r} {d[\'LIMIT\']} {LIMIT:>{LIMIT}}"\n')
    result = tp.transplant(src, ["show"], {"LIMIT"}, "_up", parent_module="synmod",
                           preamble=SYN_PRE)
    span = _span_text(result.leaf_source, "show")
    assert ('f"limit={_up().LIMIT} {k!r} {d[\'LIMIT\']} '
            '{_up().LIMIT:>{_up().LIMIT}}"') in span
    entry = result.proof["symbols"]["show"]
    assert entry["ast_equal"] and entry["tokens_equal"] and entry["byte_identical"]
    assert entry["ast_inverse_equal"]
    assert result.proof["ok"]


def test_fstring_debug_specs_fail_closed():
    """`f"{X=}"` PRINTS the expression text, which CPython derives from the very
    bytes the handle rewrite changes: `f"{_up().LIMIT=}"` inverts byte-perfectly
    while its output silently became `_up().LIMIT=`. Only the tree-level inverse
    sees that Constant, so it is the check that refuses the span."""
    src = SYN + '\ndef show():\n    return f"{LIMIT=}"\n'
    with pytest.raises(tp.TransplantError) as excinfo:
        tp.transplant(src, ["show"], {"LIMIT"}, "_up", parent_module="synmod",
                      preamble=SYN_PRE)
    assert excinfo.value.kind == "proof"
    entry = excinfo.value.details["proof"]["symbols"]["show"]
    assert entry["byte_identical"] and not entry["ast_inverse_equal"]


def test_proof_detects_tampering_inside_an_fstring_literal():
    result = tp.transplant(SYN_FSTR, ["show"], {"LIMIT"}, "_up",
                           parent_module="synmod", preamble=SYN_PRE)
    tampered = result.leaf_source.replace('f"limit=', 'f"cap=')
    report = tp.verify_transplant(SYN_FSTR, tampered, ["show"], {"LIMIT"}, "_up")
    assert not report["ok"]
    entry = report["symbols"]["show"]
    assert not entry["tokens_equal"] and not entry["byte_identical"]


def test_wildcard_imports_fail_closed():
    src = "from os.path import *\n\ndef f():\n    return join('a', 'b')\n"
    with pytest.raises(tp.TransplantError) as excinfo:
        tp.transplant(src, ["f"], set(), "_up", parent_module="synmod")
    assert excinfo.value.kind == "wildcard_import"


def test_handle_name_collision_fails_closed():
    src = "def _up():\n    return 1\n\ndef f():\n    return _up()\n"
    with pytest.raises(tp.TransplantError) as excinfo:
        tp.transplant(src, ["f"], set(), "_up", parent_module="synmod")
    assert excinfo.value.kind == "handle_collision"


def test_multi_target_assign_moves_whole_or_not_at_all():
    with pytest.raises(tp.TransplantError) as excinfo:
        _syn(["A"], set())
    assert excinfo.value.kind == "extraction"
    assert "also move" in excinfo.value.message
    result = _syn(["A", "B"], set())
    assert result.leaf_source.count("A = B = []") == 1


def test_conditional_and_missing_symbols_fail_closed():
    src = "if True:\n    def cond():\n        return 1\n"
    with pytest.raises(tp.TransplantError) as excinfo:
        tp.transplant(src, ["cond"], set(), "_up", parent_module="synmod")
    assert excinfo.value.kind == "extraction"


def test_statements_sharing_a_line_cannot_round_trip():
    src = "C = 1; D = 2\n"
    with pytest.raises(tp.TransplantError) as excinfo:
        tp.transplant(src, ["C", "D"], set(), "_up", parent_module="synmod")
    assert excinfo.value.kind == "round_trip"


def test_preamble_must_not_bind_declared_names():
    pre = SYN_PRE + "\nLIMIT = 99\n"
    with pytest.raises(tp.TransplantError) as excinfo:
        _syn(["cached"], {"LIMIT"}, preamble=pre)
    assert excinfo.value.kind == "preamble"


def test_preamble_requires_future_annotations():
    with pytest.raises(tp.TransplantError) as excinfo:
        _syn(["helper"], set(), preamble='"""Leaf."""\nimport json\n')
    assert excinfo.value.kind == "preamble"
    assert "__future__" in excinfo.value.message


def test_proof_detects_comment_and_code_tampering():
    result = _syn(["uses", "strings_and_comments"], {"PENDING", "RUNNING", "helper"})
    # comment tampering: AST-equal but the token proof catches it
    tampered = result.leaf_source.replace(
        "# PENDING in a comment stays a comment", "# tampered comment")
    report = tp.verify_transplant(SYN, tampered, ["uses", "strings_and_comments"],
                                  {"PENDING", "RUNNING", "helper"}, "_up")
    assert not report["ok"]
    entry = report["symbols"]["strings_and_comments"]
    assert entry["ast_equal"] and not entry["tokens_equal"]
    # code tampering: both proofs catch it
    tampered = result.leaf_source.replace("with_lock = _up().PENDING",
                                          "with_lock = list(_up().PENDING)")
    report = tp.verify_transplant(SYN, tampered, ["uses"], {"PENDING", "RUNNING", "helper"},
                                  "_up")
    assert not report["ok"]
    assert not report["symbols"]["uses"]["ast_equal"]


def test_proof_rejects_bare_handle_use_and_undeclared_reads():
    result = _syn(["uses"], {"PENDING", "RUNNING", "helper"})
    bare = result.leaf_source.replace("_up().helper(x)", "_up().json.dumps(x)")
    report = tp.verify_transplant(SYN, bare, ["uses"], {"PENDING", "RUNNING", "helper"}, "_up")
    assert not report["ok"]
    assert "undeclared" in (report["symbols"]["uses"]["detail"] or "")


def test_emitted_leaf_parses_and_carries_the_handle_def():
    result = _syn(["uses"], {"PENDING", "RUNNING", "helper"})
    tree = ast.parse(result.leaf_source)
    handles = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_up"]
    assert len(handles) == 1
    assert any(isinstance(n, (ast.Import, ast.ImportFrom)) for n in ast.walk(handles[0]))


def test_cli_emit_and_check_roundtrip(tmp_path):
    upstream = tmp_path / "up.py"
    upstream.write_text(SYN, encoding="utf-8")
    leaf = tmp_path / "leaf.py"
    argv = [sys.executable, str(TOOL_PATH),
            "--upstream", str(upstream), "--symbols", "uses,helper",
            "--declared", "PENDING,RUNNING", "--handle", "_up",
            "--parent-module", "synmod", "--out", str(leaf)]
    pre = tmp_path / "pre.py"
    pre.write_text(SYN_PRE, encoding="utf-8")
    emit = subprocess.run(argv + ["--preamble-file", str(pre)],
                          capture_output=True, text=True)
    assert emit.returncode == 0, emit.stderr
    check_argv = [sys.executable, str(TOOL_PATH), "--check",
                  "--upstream", str(upstream), "--leaf", str(leaf),
                  "--symbols", "uses,helper", "--declared", "PENDING,RUNNING",
                  "--handle", "_up"]
    check = subprocess.run(check_argv, capture_output=True, text=True)
    assert check.returncode == 0, check.stderr
    corrupted = leaf.read_text(encoding="utf-8").replace("return x + 1", "return x - 1")
    leaf.write_text(corrupted, encoding="utf-8")
    check = subprocess.run(check_argv, capture_output=True, text=True)
    assert check.returncode == 2


# ---------------------------------------------------------------------------
# Mutation tests (audit 2026-08-30): the proof must FAIL on what tokens miss.
# ---------------------------------------------------------------------------

_MUT_UP = "def f(a, b):\n    return a  +  b + PARENT\n"
_MUT_FN = "def f(a, b):\n    return a  +  b + _h().PARENT\n"
# A COMPLETE, runnable leaf (F0 review: verify the whole module, not a fragment).
_MUT_LEAF_OK = ('"""doc"""\nfrom __future__ import annotations\n'
                "from ouroboros import config as _parent\n\n\n"
                "def _h():\n    return _parent\n\n\n" + _MUT_FN)


def test_mutation_whitespace_change_fails_byte_proof():
    """Inter-token whitespace edits are invisible to the token proof; the
    mandatory byte round trip must catch them."""
    from scripts.v7next_transplant import verify_transplant
    fn_ws = "def f(a, b):\n    return a + b + _h().PARENT\n"  # collapsed spaces
    leaf_ws = _MUT_LEAF_OK.replace(_MUT_FN, fn_ws)
    rep = verify_transplant(_MUT_UP, leaf_ws, ["f"], {"PARENT"}, "_h")
    assert rep["ok"] is False
    assert "byte-identical" in (rep["symbols"]["f"]["detail"] or "")
    rep_ok = verify_transplant(_MUT_UP, _MUT_LEAF_OK, ["f"], {"PARENT"}, "_h")
    assert rep_ok["ok"] is True and rep_ok["symbols"]["f"]["byte_identical"] is True


def test_mutation_extra_top_level_def_fails():
    leaf = _MUT_LEAF_OK + "\n\ndef smuggled():\n    return 1\n"
    from scripts.v7next_transplant import verify_transplant
    rep = verify_transplant(_MUT_UP, leaf, ["f"], {"PARENT"}, "_h")
    assert rep["ok"] is False
    assert any("smuggled" in e for e in rep["undeclared_top_level"])


def test_mutation_import_time_side_effect_fails():
    leaf = _MUT_LEAF_OK + "\nprint('boom')\n"
    from scripts.v7next_transplant import verify_transplant
    rep = verify_transplant(_MUT_UP, leaf, ["f"], {"PARENT"}, "_h")
    assert rep["ok"] is False
    assert any("Expr" in e or "line" in e for e in rep["undeclared_top_level"])


def test_complete_leaf_passes():
    from scripts.v7next_transplant import verify_transplant
    rep = verify_transplant(_MUT_UP, _MUT_LEAF_OK, ["f"], {"PARENT"}, "_h")
    assert rep["undeclared_top_level"] == [] and rep["leaf_invariants"] == []
    assert rep["ok"] is True


# --- F0 phase-review CRITICAL: whole-leaf invariants a span proof cannot see ---

def test_missing_handle_def_fails():
    """A leaf that reads _h().PARENT but never defines _h is not runnable."""
    from scripts.v7next_transplant import verify_transplant
    rep = verify_transplant(_MUT_UP, _MUT_FN, ["f"], {"PARENT"}, "_h")
    assert rep["ok"] is False
    assert any("defined 0 times" in e for e in rep["leaf_invariants"])


def test_handle_returning_none_fails():
    from scripts.v7next_transplant import verify_transplant
    bad = _MUT_LEAF_OK.replace("def _h():\n    return _parent",
                               "def _h():\n    return None")
    rep = verify_transplant(_MUT_UP, bad, ["f"], {"PARENT"}, "_h")
    assert rep["ok"] is False
    assert any("module reference" in e for e in rep["leaf_invariants"])


def test_declared_and_preamble_bound_overlap_fails():
    """PARENT both declared (read via handle) and imported in the preamble =
    ambiguous ownership."""
    from scripts.v7next_transplant import verify_transplant
    bad = _MUT_LEAF_OK.replace("from ouroboros import config as _parent\n",
                               "from ouroboros import config as _parent\nimport PARENT\n")
    rep = verify_transplant(_MUT_UP, bad, ["f"], {"PARENT"}, "_h")
    assert rep["ok"] is False
    assert any("ambiguous ownership" in e for e in rep["leaf_invariants"])


def test_unread_declared_name_fails():
    from scripts.v7next_transplant import verify_transplant
    rep = verify_transplant(_MUT_UP, _MUT_LEAF_OK, ["f"], {"PARENT", "UNUSED"}, "_h")
    assert rep["ok"] is False
    assert any("never read through" in e for e in rep["leaf_invariants"])


def test_projection_only_leaf_without_handle_passes():
    """A leaf with zero handle reads and zero declared names (pure projection,
    e.g. context_runtime_facts.py) legitimately carries no handle def."""
    from scripts.v7next_transplant import verify_transplant
    up = "def g():\n    return 1\n"
    leaf = ('"""doc"""\nfrom __future__ import annotations\n\n\n'
            "def g():\n    return 1\n")
    rep = verify_transplant(up, leaf, ["g"], set(), "_h")
    assert rep["leaf_invariants"] == []
    assert rep["ok"] is True


def _leaf_with_handle(handle_src: str) -> str:
    return ('"""doc"""\nfrom __future__ import annotations\n'
            "from ouroboros import config as _parent\n\n\n"
            + handle_src + "\n\n" + _MUT_FN)


def test_async_handle_fails():
    from scripts.v7next_transplant import verify_transplant
    rep = verify_transplant(_MUT_UP, _leaf_with_handle(
        "async def _h():\n    return _parent"), ["f"], {"PARENT"}, "_h")
    assert rep["ok"] is False
    assert any("sync def" in e for e in rep["leaf_invariants"])


def test_posonly_param_handle_fails():
    from scripts.v7next_transplant import verify_transplant
    rep = verify_transplant(_MUT_UP, _leaf_with_handle(
        "def _h(x, /):\n    return _parent"), ["f"], {"PARENT"}, "_h")
    assert rep["ok"] is False
    assert any("no parameters" in e for e in rep["leaf_invariants"])


def test_constant_return_handle_fails():
    from scripts.v7next_transplant import verify_transplant
    rep = verify_transplant(_MUT_UP, _leaf_with_handle(
        "def _h():\n    return 42"), ["f"], {"PARENT"}, "_h")
    assert rep["ok"] is False
    assert any("module reference" in e for e in rep["leaf_invariants"])


def test_nested_only_return_handle_fails():
    from scripts.v7next_transplant import verify_transplant
    rep = verify_transplant(_MUT_UP, _leaf_with_handle(
        "def _h():\n    def inner():\n        return _parent\n    pass"),
        ["f"], {"PARENT"}, "_h")
    assert rep["ok"] is False
    assert any("module reference" in e for e in rep["leaf_invariants"])


def test_dotted_attribute_return_handle_passes():
    from scripts.v7next_transplant import verify_transplant
    leaf = ('"""doc"""\nfrom __future__ import annotations\nimport ouroboros.config\n\n\n'
            "def _h():\n    return ouroboros.config\n\n\n" + _MUT_FN)
    rep = verify_transplant(_MUT_UP, leaf, ["f"], {"PARENT"}, "_h")
    assert rep["leaf_invariants"] == []
    assert rep["ok"] is True


def test_tuple_target_assignment_of_requested_symbols_passes():
    """`A, B = 500, 10` with both names requested is a legitimate moved span,
    not an undeclared top-level extra (D12 lane false positive)."""
    from scripts.v7next_transplant import verify_transplant
    up = "A, B = 500, 10\n"
    leaf = ('"""doc"""\nfrom __future__ import annotations\n\n\nA, B = 500, 10\n')
    rep = verify_transplant(up, leaf, ["A", "B"], set(), "_h")
    assert rep["undeclared_top_level"] == []
    assert rep["ok"] is True


def test_tuple_target_with_foreign_name_still_fails():
    from scripts.v7next_transplant import verify_transplant
    up = "A, B = 500, 10\n"
    leaf = ('"""doc"""\nfrom __future__ import annotations\n\n\n'
            "A, B = 500, 10\nX, Y = 1, 2\n")
    rep = verify_transplant(up, leaf, ["A", "B"], set(), "_h")
    assert rep["ok"] is False
    assert any("X" in e for e in rep["undeclared_top_level"])


def test_nested_tuple_target_with_unrequested_names_fails_closed():
    """`A, (X, Y) = ...` with only A requested now fails CLOSED at extraction
    (the recursive unfold makes extract_spans see every bound name) — the
    one-level unfold used to let X/Y ride silently (wave-2 conformance)."""
    import pytest
    from scripts.v7next_transplant import verify_transplant, TransplantError
    up = "A, (X, Y) = 1, (2, 3)\n"
    leaf = ('"""doc"""\nfrom __future__ import annotations\n\n\nA, (X, Y) = 1, (2, 3)\n')
    with pytest.raises(TransplantError) as exc:
        verify_transplant(up, leaf, ["A"], set(), "_h")
    assert "X" in str(exc.value) and "Y" in str(exc.value)


def test_attribute_target_is_complex_and_fails():
    """`A, obj.attr = ...` mutates foreign state at import time — always an
    undeclared extra even when A is requested."""
    from scripts.v7next_transplant import verify_transplant
    leaf = ('"""doc"""\nfrom __future__ import annotations\nimport os\n\n\n'
            "A, os.environ_x = 1, 2\n")
    rep = verify_transplant("A = 1\n", leaf, ["A"], set(), "_h")
    assert rep["ok"] is False
    assert any("complex target" in e for e in rep["undeclared_top_level"])


def test_nested_tuple_all_requested_passes():
    from scripts.v7next_transplant import verify_transplant
    up = "A, (X, Y) = 1, (2, 3)\n"
    leaf = ('"""doc"""\nfrom __future__ import annotations\n\n\nA, (X, Y) = 1, (2, 3)\n')
    rep = verify_transplant(up, leaf, ["A", "X", "Y"], set(), "_h")
    assert rep["undeclared_top_level"] == []
    assert rep["ok"] is True
