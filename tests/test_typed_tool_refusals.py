"""A builtin tool's known failure must leave the producer typed (package P4, issue #739).

The registry types a string result by its FIRST LINE: ``⚠️ IDENTIFIER`` maps through
``LegacyTextResultAdapter`` to a status, while identifier-less prose (``⚠️ File not
found: x``) or a bare ``ERROR: ...`` string is recorded as a SUCCESSFUL call — in
``tools.jsonl``, the outcome classifier and the acceptance packet. A producer that
already knows it failed publishes that fact as a typed ``ToolResult``
(``tool_result._publish_tool_result``) or a first-line marker the adapter types.

This is a SOURCE lint over the builtin tool modules, not a runtime gate: it
constrains how new code is written, never how the agent behaves (BIBLE P5). The
flagged shape is narrow: the text opens with ``ERROR:`` or with ``⚠️`` followed by
something that is not an UPPER_SNAKE identifier — prose, a lowercase name, a bare
capital — AND the real adapter records it ``ok``. A typed-looking marker the adapter
buckets as a warning (``⚠️ X_INVALID``) is a vocabulary question for the adapter's
owner, not this lint (a separate, larger residual — dozens of marker-shaped refusals
recorded ``ok`` — that this allowlist does NOT count), and an adapter that starts
typing a text releases it here. Every surviving site carries a written reason here,
so the allowlist IS the disclosure of the identifier-less residual; a site removed from the tree must be removed
here too (shrink-only), and a file whose count grows fails. It is a per-file COUNT
ratchet: swapping one old site for a new one inside the same file is invisible to
it (disclosed), and the producer's own tests pin individual sites.

Scope limit (disclosed): only RETURNED string literals (plain, f-string with a static
head, or a leading-literal concatenation) are scanned. A failure text that reaches the
model through a variable, a tuple or a helper is invisible to this lint; the typed
producer path is the repair for those, pinned by their own tests.

A DYNAMIC head (``f"⚠️ {code}: ..."``) cannot be judged statically: whether the adapter
types it depends on the runtime identifier, and a lowercase or out-of-vocabulary code is
recorded ``ok`` (the live `schedule_subagent` / `escalate` refusal loops were exactly
this shape). The producer always knows it refused, so new code publishes such a text
typed; the pre-existing sites are counted per file in ``ALLOWED_DYNAMIC`` and may not grow.
"""

from __future__ import annotations

import ast

import pytest
import pathlib
import re

from ouroboros.tools.tool_result import LegacyTextResultAdapter

REPO = pathlib.Path(__file__).resolve().parents[1]
ROOTS = ("ouroboros/tools",)
# A typed marker: ⚠️ then an UPPER_SNAKE identifier of three or more characters, delimited.
_TYPED_MARKER = re.compile(r"^⚠️ +[A-Z][A-Z0-9_]{2,}(?=[\s:(]|$)")

# repo-relative path -> (occurrences, why they stay). Every entry is the same class this
# lint exists for, left in place because the file belongs to another package of the
# autonomy sprint (issue #739 names the follow-up); the count may only shrink.
ALLOWED: dict[str, tuple[int, str]] = {
    "ouroboros/tools/control_task_results.py": (
        1, "legitimate cache-horizon warning after a successful wait; no operation was refused",
    ),
}

# Pre-existing returns whose marker identifier is interpolated. Growth fails; a stale
# (larger) count is harmless here, because concurrent branches retire these sites one by one.
_DYNAMIC_WHY = "pre-existing interpolated marker; publish it typed when the producer is next touched"
ALLOWED_DYNAMIC: dict[str, tuple[int, str]] = {
    "ouroboros/tools/browser.py": (6, _DYNAMIC_WHY),
    "ouroboros/tools/core.py": (2, _DYNAMIC_WHY),
    "ouroboros/tools/core_file_tools.py": (1, "the interpolated identifier always ends in _BLOCKED, which the adapter types"),
    "ouroboros/tools/edit_ops.py": (1, _DYNAMIC_WHY),
    "ouroboros/tools/skill_exec.py": (2, _DYNAMIC_WHY),
    "ouroboros/tools/tool_resolution.py": (1, "protected registry path; the binding-error text is typed by its SKILL_REDIRECT_ family marker"),
}


def _static_head(node: ast.expr) -> tuple[str, bool] | None:
    """Leading literal text of a returned string expression, plus whether it is partial.

    Returns ``None`` when the returned value is not a string literal shape.
    ``partial`` is True when a placeholder follows the literal head (f-string), so
    the identifier the registry would read may be dynamic.
    """
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value, False
    if isinstance(node, ast.JoinedStr):
        head = ""
        for part in node.values:
            if isinstance(part, ast.Constant) and isinstance(part.value, str):
                head += part.value
                continue
            return (head, True) if head else None
        return head, False
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        inner = _static_head(node.left)
        return (inner[0], True) if inner else None  # the right operand is unknown
    return None


def _untyped_failure(head: str, partial: bool) -> bool:
    first = head.splitlines()[0].strip() if head.strip() else ""
    if not first:
        return False
    if first.startswith("ERROR:"):
        return True
    if not first.startswith("⚠️"):
        return False
    if partial and not first.lstrip("⚠\ufe0f").strip():
        # ``⚠️ {code}: ...`` — the identifier is dynamic; this lint cannot judge it.
        return False
    if _TYPED_MARKER.match(first):
        return False
    return LegacyTextResultAdapter.from_text("lint", first).status == "ok"


def _dynamic_marker(head: str, partial: bool) -> bool:
    return partial and head.startswith("⚠️") and not head.lstrip("⚠\ufe0f").strip()


def observed_untyped_returns(judge=_untyped_failure) -> dict[str, list[tuple[int, str]]]:
    found: dict[str, list[tuple[int, str]]] = {}
    for root in ROOTS:
        for path in sorted((REPO / root).glob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            rel = path.relative_to(REPO).as_posix()
            for node in ast.walk(tree):
                if not isinstance(node, ast.Return) or node.value is None:
                    continue
                shape = _static_head(node.value)
                if shape is None:
                    continue
                head, partial = shape
                if judge(head, partial):
                    found.setdefault(rel, []).append((node.lineno, head.splitlines()[0].strip()[:60]))
    return found


def test_untyped_failure_returns_only_shrink() -> None:
    observed = observed_untyped_returns()
    counts = {path: len(rows) for path, rows in observed.items()}
    allowed = {path: count for path, (count, _why) in ALLOWED.items() if count}
    new_sites = {path: rows for path, rows in observed.items() if counts[path] > allowed.get(path, 0)}
    stale = {path: allowed[path] for path in allowed if counts.get(path, 0) < allowed[path]}
    detail = "\n".join(
        f"  {path}:{line}  {text}" for path, rows in sorted(new_sites.items()) for line, text in rows
    )
    assert not new_sites, (
        "a builtin tool returns a failure text the registry would record as ok; publish a typed "
        "ToolResult (tool_result._publish_tool_result) or a first-line ⚠️ IDENTIFIER marker instead:\n"
        f"{detail}"
    )
    assert not stale, (
        "untyped-failure sites disappeared; shrink ALLOWED in tests/test_typed_tool_refusals.py to match: "
        f"{stale}"
    )
    for path, (count, why) in ALLOWED.items():
        assert not count or why.strip(), f"ALLOWED[{path!r}] needs a written reason"


def test_dynamic_marker_returns_do_not_grow() -> None:
    observed = {path: len(rows) for path, rows in observed_untyped_returns(_dynamic_marker).items()}
    grown = {path: count for path, count in observed.items() if count > ALLOWED_DYNAMIC.get(path, (0, ""))[0]}
    assert not grown, (
        "a builtin tool returns a refusal whose ⚠️ identifier is interpolated, so the registry may record "
        "it as a successful call; publish it typed (tools/arg_feedback.argument_refusal or "
        f"tool_result._publish_tool_result): {grown}"
    )
    for path, (_count, why) in ALLOWED_DYNAMIC.items():
        assert why.strip(), f"ALLOWED_DYNAMIC[{path!r}] needs a written reason"



@pytest.fixture
def model_start(tmp_path, monkeypatch):
    from ouroboros.tools.registry import ToolContext
    from ouroboros.tools import control_models, control_routing
    from ouroboros import config

    ctx = ToolContext(repo_dir=tmp_path, drive_root=tmp_path, task_id="caller", task_metadata={})
    ctx._owner_directives = [{"source": "initial_user", "content": "Please use the other one, then keep the report complete."}]
    monkeypatch.setattr(config, "get_runtime_mode", lambda: "pro")
    monkeypatch.setattr(control_models, "known_models", lambda: [
        {"model": "vendor/fast-one", "use_local": False}, {"model": "vendor/fast-two", "use_local": False}])
    events = []
    monkeypatch.setattr(control_routing, "_emit_and_wait_for_routing",
                        lambda _ctx, event: (events.append(event) or "live", {"status": "scheduled"}))
    monkeypatch.setattr(control_routing, "_promotion_pool_disabled_from_snapshot", lambda _ctx: "")
    from ouroboros.tools.tool_result import _install_tool_result_sidecar, _restore_tool_result_sidecar
    token = _install_tool_result_sidecar(ctx, object())
    try:
        yield ctx, events
    finally:
        _restore_tool_result_sidecar(token)


@pytest.mark.parametrize("cite", ["", "words never spoken"])
def test_a_model_without_verified_words_is_a_typed_refusal_with_no_uncited_retry(model_start, cite):
    from ouroboros.tools.control_routing import _start_task
    from ouroboros.tools.tool_result import _published_tool_result

    ctx, events = model_start
    text = _start_task(ctx, "Write the report", title="Report", model="vendor/fast-one", owner_words=cite)
    assert not events
    result = _published_tool_result(ctx, None)
    assert result.status == "error" and result.code == "TOOL_ARG_ERROR"
    assert "omit owner_words" not in text and "as your own" not in text


@pytest.mark.parametrize("model,starts", [("fast", False), ("vendor/fast-one", True), ("vendor/unlisted", True)])
def test_model_suggestions_never_guess_or_admit_a_route(model_start, model, starts):
    from ouroboros.tools.control_routing import _start_task

    ctx, events = model_start
    text = _start_task(ctx, "Write the report", title="Report", model=model, owner_words="use the other one")
    assert bool(events) is starts
    if not starts:
        assert "vendor/fast-one" in text and "vendor/fast-two" in text
    else:
        assert events[0]["metadata"]["lead"] == {"source": "owner_words", "model": model, "use_local": False}
        assert events[0]["owner_corpus"][-1]["content"] == ctx._owner_directives[0]["content"]
        if model == "vendor/unlisted":
            assert "not verified" in text


def test_an_unreadable_local_catalog_preserves_the_typed_choice(model_start, monkeypatch):
    from ouroboros.tools import control_models, control_routing

    ctx, events = model_start
    def unreadable():
        raise OSError("unavailable")
    monkeypatch.setattr(control_models, "known_models", unreadable)
    text = control_routing._start_task(ctx, "Write", title="Report", model="arbitrary/new", owner_words="use the other one")
    assert "not verified" in text and events[0]["metadata"]["lead"]["model"] == "arbitrary/new"


def test_cyber_start_names_the_minds_choice_without_overruling_a_room_owner(model_start, monkeypatch):
    from ouroboros import config, route_choice, lead_fact
    from ouroboros.tools import control_models, control_routing

    ctx, events = model_start
    monkeypatch.setattr(config, "get_runtime_mode", lambda: "cyber_pro")
    control_routing._start_task(ctx, "Write", title="Report", model="vendor/fast-one")
    ctx.task_metadata = events[0]["metadata"]
    route_choice.bind_task_route(ctx, ctx.task_metadata, presence=False)
    assert ctx.task_model_override == "vendor/fast-one" and lead_fact.primary_basis(ctx) == "mind"
    monkeypatch.setattr(control_models, "read_model_choice", lambda *a: {"kind": "model", "model": "owner/choice", "use_local": False})
    events.clear()
    text = control_routing._start_task(ctx, "Write", title="Report", model="vendor/fast-two")
    assert not events and "choice for this room binds" in text
    control_routing._start_task(ctx, "Write", title="Report", model="vendor/fast-two", owner_words="use the other one")
    assert events[0]["metadata"]["lead"]["source"] == "owner_words"
