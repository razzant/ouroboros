"""A change does not make a reference book larger (the official line's rule).

Each book is measured as the UTF-8 length of its COMPOSED text (the entrypoint and every
listed chapter, ``reference_books.compose_book``) at this change's tip and at its event base.
The rule is pairwise: there is no stored number and no reserve, so text a change adds to a
book is paid by shortening the same book in the same change.

Official-CI ``size_ratchet`` lane only: local runs exclude the marker and every local surface
reports the same balance as a fact (BIBLE P3 c5). The workflow sets ``OURO_BOOK_GROWTH_MODE``:
``block`` for the official repository's owner/member/collaborator pull requests and pushes to
``ouroboros``, ``warn`` for outside contributors and any fork's own CI. ``approved`` requires
the owner's ``book-growth`` label; pushes bind it to the exact landed PR
(``scripts/book_growth_mode.py``). Unset mode means ``block`` when
``OURO_SIZE_RATCHET_BASE_REF`` resolves; without a base the book comparison skips.
"""
from __future__ import annotations

import os
import pathlib
import subprocess

import pytest

from ouroboros.reference_books import BOOK_ENTRYPOINTS, compose_book, load_reference_book

REPO = pathlib.Path(__file__).resolve().parents[1]
BASE_REF_ENV = "OURO_SIZE_RATCHET_BASE_REF"
MODE_ENV = "OURO_BOOK_GROWTH_MODE"


def _git(repo: pathlib.Path, *args: str, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=repo, check=check, capture_output=True)


def measures_one_change(environ) -> bool:
    """A pull request or a push to ``ouroboros`` is one change; a push to ``main`` or
    ``ouroboros-stable`` spans a whole release, which the pairwise rule cannot attribute."""
    return environ.get("GITHUB_EVENT_NAME") != "push" or environ.get("GITHUB_REF") == "refs/heads/ouroboros"


def growth_base(repo: pathlib.Path, ref: str | None) -> str:
    """The commit this change is measured from; ``""`` when the rule does not apply.

    A run without a resolvable event base skips: a local or manual run has none, a tag push
    carries all zeros, and HEAD's parent is not where a multi-commit change began.
    """
    ref = (ref or "").strip()
    if not ref:
        return ""
    resolved = _git(repo, "rev-parse", "--verify", "--quiet", f"{ref}^{{commit}}", check=False)
    return resolved.stdout.decode("ascii").strip() if resolved.returncode == 0 else ""


def composed_bytes(repo: pathlib.Path, book_id: str, ref: str = "") -> int | None:
    """The composed book's UTF-8 bytes in the checkout, or at ``ref``; ``None`` when absent there."""
    reader = None
    if ref:
        def reader(path: str) -> bytes:
            return _git(repo, "show", f"{ref}:{path}").stdout
    try:
        return len(compose_book(load_reference_book(repo, book_id, reader)).encode("utf-8"))
    except (OSError, ValueError, KeyError, subprocess.CalledProcessError):
        if not ref:
            raise
        return None


def book_growth_verdict(book_id: str, base: int | None, tip: int, mode: str) -> tuple[bool, str]:
    """``(passes, message)`` for one book; the message is empty when the book did not grow."""
    if base is None or tip <= base:
        return True, ""
    name = f"the {book_id.title()} book"
    if mode == "approved":
        return True, f"{name} grows by {tip - base} bytes; the repository owner approved it (book-growth label)."
    if mode == "warn":
        return True, (f"{name} grows by {tip - base} bytes; advisory for outside contributors and forks. "
                      "Official maintainers will make room at integration.")
    return False, (f"{name} grew by {tip - base} bytes ({base} -> {tip}); a change ends each book no larger "
                   "than at its base, so shorten the same book in this change (the owner's book-growth "
                   "label on the PR is the only exception).")


@pytest.mark.size_ratchet
def test_a_change_does_not_grow_a_reference_book(capsys):
    if not measures_one_change(os.environ):
        pytest.skip("a push to a release branch spans many changes; the rule measures one")
    base = growth_base(REPO, os.environ.get(BASE_REF_ENV))
    if not base:
        pytest.skip(f"{BASE_REF_ENV} names no base commit this clone holds, so this change cannot be measured")
    mode = (os.environ.get(MODE_ENV) or "block").strip()
    faults: list[str] = []
    for book_id, entrypoint in BOOK_ENTRYPOINTS.items():
        passes, message = book_growth_verdict(book_id, composed_bytes(REPO, book_id, base),
                                              composed_bytes(REPO, book_id), mode)
        if message and passes:  # an annotation in the job log; a captured print would never reach it
            with capsys.disabled():
                print(f"::{'notice' if mode == 'approved' else 'warning'} file={entrypoint}::{message}")
        elif message:
            faults.append(message)
    assert not faults, f"Reference-book growth since {base[:12]}:\n" + "\n".join(faults)


# ---------------------------------------------------------------- the decision, unmarked

@pytest.mark.parametrize("base,tip,mode,passes,said", [
    (1000, 1001, "block", False, "grew by 1 bytes"),
    (1000, 1001, "", False, "grew by 1 bytes"),  # an unknown mode is never a pass
    (1000, 1480, "warn", True, "maintainers will make room"),
    (1000, 1480, "approved", True, "owner approved"),
    (1000, 900, "block", True, ""),
    (1000, 1000, "block", True, ""),
    (None, 5000, "block", True, ""),  # a book new in this change has no base to grow from
])
def test_book_growth_verdict(base, tip, mode, passes, said):
    verdict = book_growth_verdict("architecture", base, tip, mode)
    assert verdict[0] is passes
    assert (said in verdict[1]) if said else verdict[1] == ""


@pytest.mark.serial
def test_composed_bytes_measure_the_book_at_the_base_commit(tmp_path):
    entry = "# Architecture\n\nIntro café.\n\n## Chapters\n\n- [One](architecture/01-one.md)\n"
    chapter = "# One\n\nThe first chapter, naïve.\n"
    for path, text in {BOOK_ENTRYPOINTS["architecture"]: entry, "docs/architecture/01-one.md": chapter}.items():
        (tmp_path / path).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / path).write_text(text, encoding="utf-8", newline="\n")
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "add", "-A")
    _git(tmp_path, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "base")
    expected = len((entry + "\n\n" + chapter).encode("utf-8"))
    assert composed_bytes(tmp_path, "architecture", "HEAD") == expected == composed_bytes(tmp_path, "architecture")
    (tmp_path / "docs/architecture/01-one.md").write_text(chapter + "More.\n", encoding="utf-8", newline="\n")
    assert composed_bytes(tmp_path, "architecture", "HEAD") == expected  # the base does not move with the checkout
    assert composed_bytes(tmp_path, "architecture") == expected + len("More.\n")
    assert composed_bytes(tmp_path, "development", "HEAD") is None  # absent at the base


@pytest.mark.serial
def test_growth_base_is_the_event_base_and_nothing_else(tmp_path):
    _git(tmp_path, "init", "-q")
    shas = []
    for text in ("one\n", "one\ntwo\n"):
        (tmp_path / "f.txt").write_text(text, encoding="utf-8")
        _git(tmp_path, "add", "-A")
        _git(tmp_path, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", text)
        shas.append(_git(tmp_path, "rev-parse", "HEAD").stdout.decode().strip())
    assert growth_base(tmp_path, shas[0]) == shas[0]
    assert growth_base(tmp_path, f" {shas[0][:12]} ") == shas[0]
    for absent in (None, "", "   ", "0" * 40, "f" * 40):
        assert growth_base(tmp_path, absent) == "", absent


@pytest.mark.parametrize(("environ", "applies"), [
    ({}, True),  # a local run with an explicit base
    ({"GITHUB_EVENT_NAME": "pull_request", "GITHUB_REF": "refs/pull/7/merge"}, True),
    ({"GITHUB_EVENT_NAME": "push", "GITHUB_REF": "refs/heads/ouroboros"}, True),
    ({"GITHUB_EVENT_NAME": "push", "GITHUB_REF": "refs/heads/ouroboros-stable"}, False),
    ({"GITHUB_EVENT_NAME": "push", "GITHUB_REF": "refs/heads/main"}, False),
])
def test_the_rule_measures_one_change_not_a_release_range(environ, applies):
    assert measures_one_change(environ) is applies


# ------------------------------------------------------ whom the rule binds, from raw events

def _label(event, login, name="book-growth"):
    return {"event": event, "actor": {"login": login}, "label": {"name": name}}


OWNER_ADDS = _label("labeled", "razzant")


@pytest.mark.parametrize("repo,event,association,events,mode", [
    ("razzant/ouroboros", "pull_request", "COLLABORATOR", [], "block"),
    ("razzant/ouroboros", "pull_request", "OWNER", [], "block"),
    ("razzant/ouroboros", "pull_request", "MEMBER", [], "block"),
    ("razzant/ouroboros", "pull_request", "CONTRIBUTOR", [], "warn"),
    ("razzant/ouroboros", "pull_request", "FIRST_TIME_CONTRIBUTOR", [], "warn"),
    ("someone/fork", "pull_request", "OWNER", [OWNER_ADDS], "warn"),  # a fork's own CI never blocks
    ("razzant/ouroboros", "push", "", [], "warn"),  # without a target ref this is not an official-line push
    ("razzant/ouroboros", "pull_request", "COLLABORATOR", [OWNER_ADDS], "approved"),
    # Ouroboros's own token can apply the label, but only the repository owner's approves.
    ("razzant/ouroboros", "pull_request", "COLLABORATOR", [_label("labeled", "ouroboros-agent")], "block"),
    ("razzant/ouroboros", "pull_request", "COLLABORATOR", [_label("labeled", "razzant", "other")], "block"),
    ("razzant/ouroboros", "pull_request", "COLLABORATOR", [OWNER_ADDS, _label("unlabeled", "razzant")], "block"),
    ("razzant/ouroboros", "pull_request", "COLLABORATOR",
     [_label("labeled", "ouroboros-agent"), _label("unlabeled", "ouroboros-agent"), OWNER_ADDS], "approved"),
    ("razzant/ouroboros", "pull_request", "COLLABORATOR",
     [OWNER_ADDS, {"event": "commented", "actor": {"login": "x"}}, _label("labeled", "razzant", "other")], "approved"),
    ("razzant/ouroboros", "pull_request", "COLLABORATOR", None, "block"),  # unreadable label events
])
def test_the_growth_mode_is_decided_from_raw_label_events(repo, event, association, events, mode):
    from scripts.book_growth_mode import decide

    assert decide(repo, "razzant", event, association, events) == mode


def test_the_mode_step_reads_events_only_for_a_binding_pull_request_and_fails_closed(tmp_path):
    from scripts.book_growth_mode import main

    output, calls = tmp_path / "output", []
    base = {"REPO": "razzant/ouroboros", "OWNER": "razzant", "EVENT": "pull_request",
            "ASSOCIATION": "COLLABORATOR", "PR_NUMBER": "7", "GITHUB_OUTPUT": str(output)}

    def read(repo, number, token):
        calls.append((repo, number))
        return [OWNER_ADDS]

    def unreadable(repo, number, token):
        raise OSError("403")

    assert main(base, read) == "approved" and calls == [("razzant/ouroboros", "7")]
    assert main({**base, "ASSOCIATION": "CONTRIBUTOR"}, read) == "warn" and len(calls) == 1
    assert main(base, unreadable) == "block"
    assert output.read_text(encoding="utf-8").splitlines() == ["mode=approved", "mode=warn", "mode=block"]


@pytest.mark.parametrize("job", ["quick-test", "full-test"])
def test_both_ci_jobs_feed_the_size_lane_the_decided_mode(job):
    import yaml

    workflow = yaml.safe_load((REPO / ".github/workflows/ci.yml").read_text(encoding="utf-8"))
    steps = workflow["jobs"][job]["steps"]
    mode = next(step for step in steps if step.get("id") == "book_mode")
    size = next(step for step in steps if step.get("id") == "tests_size")
    assert mode["run"] == "python scripts/book_growth_mode.py"
    assert mode["env"]["SHA"] == "${{ github.sha }}"
    assert mode["env"]["REF"] == "${{ github.ref }}"
    assert steps.index(mode) < steps.index(size)
    assert size["env"][MODE_ENV] == "${{ steps.book_mode.outputs.mode }}"
    assert workflow["jobs"][job]["permissions"] == {"contents": "read", "pull-requests": "read"}


PUSH_SHA = "a" * 40
PUSH_ENV = {"REPO": "razzant/ouroboros", "OWNER": "razzant", "EVENT": "push",
            "SHA": PUSH_SHA, "REF": "refs/heads/ouroboros"}
API_ROOT = "https://api.github.com/repos/razzant/ouroboros/"
PULLS_URL = f"{API_ROOT}commits/{PUSH_SHA}/pulls?per_page=100"


def _pull(number, *, repo="razzant/ouroboros", target="ouroboros", sha=PUSH_SHA, merged=True):
    return {"number": number, "state": "closed" if merged else "open",
            "merged_at": "2026-10-09T09:00:00Z" if merged else None, "merge_commit_sha": sha,
            "base": {"ref": target, "repo": {"full_name": repo}}}


def _events_url(number):
    return f"{API_ROOT}issues/{number}/events?per_page=100"


def _github_pages(monkeypatch, pages):
    """Serve raw API JSON through the real reader, including any pagination links."""
    import io
    import json
    from scripts import book_growth_mode

    calls = []

    def urlopen(request, timeout):
        calls.append(request.full_url)
        assert timeout == 30
        body, link = pages[request.full_url]
        response = io.BytesIO(json.dumps(body).encode("utf-8"))
        response.headers = {"Link": link} if link else {}
        return response

    monkeypatch.setattr(book_growth_mode.urllib.request, "urlopen", urlopen)
    return calls


@pytest.mark.parametrize("reverse", [False, True], ids=["open-first", "merged-first"])
@pytest.mark.parametrize("merged_approved", [False, True], ids=["unapproved-merge", "approved-merge"])
def test_push_approval_belongs_to_the_exact_merged_pr(monkeypatch, reverse, merged_approved):
    from scripts.book_growth_mode import main

    pulls = [_pull(7, merged=False), _pull(8)]
    if reverse:
        pulls.reverse()
    calls = _github_pages(monkeypatch, {
        PULLS_URL: (pulls, ""),
        _events_url(7): ([OWNER_ADDS], ""),
        _events_url(8): ([OWNER_ADDS] if merged_approved else [], ""),
    })
    assert main(PUSH_ENV) == ("approved" if merged_approved else "block")
    assert calls == [PULLS_URL, _events_url(8)]


@pytest.mark.parametrize("pulls", [
    [],  # a direct push has no PR exception
    [_pull(7, merged=False)],
    [_pull(7, repo="someone/fork")],
    [_pull(7, target="main")],
    [_pull(7, sha="b" * 40)],
    [{**_pull(7), "state": "closed", "merged_at": None}],  # closed without merging
], ids=["absent", "open", "foreign-repo", "wrong-target", "wrong-sha", "closed-unmerged"])
def test_push_rejects_unrelated_approval(monkeypatch, pulls):
    from scripts.book_growth_mode import main

    calls = _github_pages(monkeypatch, {
        PULLS_URL: (pulls, ""), _events_url(7): ([OWNER_ADDS], ""),
    })
    assert main(PUSH_ENV) == "block"
    assert calls == [PULLS_URL]


@pytest.mark.parametrize("events,expected", [
    ([OWNER_ADDS], "approved"),
    ([_label("labeled", "razzant", "other")], "block"),
    ([_label("labeled", "ouroboros-agent")], "block"),
    ([OWNER_ADDS, _label("unlabeled", "razzant")], "block"),
    ([OWNER_ADDS, _label("unlabeled", "someone"), _label("labeled", "someone")], "block"),
    ([_label("labeled", "someone"), _label("unlabeled", "someone"), OWNER_ADDS], "approved"),
    ([OWNER_ADDS, _label("unlabeled", "someone", "other")], "approved"),
], ids=["owner", "unrelated-label", "non-owner", "removed", "non-owner-reapplied", "owner-reapplied", "unrelated-removal"])
@pytest.mark.parametrize("event", ["pull_request", "push"])
def test_raw_label_history_controls_the_exception(monkeypatch, event, events, expected):
    from scripts.book_growth_mode import main

    pages = {PULLS_URL: ([_pull(7)], ""), _events_url(7): (events, "")}
    _github_pages(monkeypatch, pages)
    env = {**PUSH_ENV, "EVENT": event, "PR_NUMBER": "7", "ASSOCIATION": "COLLABORATOR"}
    assert main(env) == expected


def test_rerunning_the_same_push_uses_current_label_history(monkeypatch):
    from scripts.book_growth_mode import main

    events = [OWNER_ADDS]
    calls = _github_pages(monkeypatch, {
        PULLS_URL: ([_pull(7)], ""), _events_url(7): (events, ""),
    })
    assert main(PUSH_ENV) == "approved"
    events.append(_label("unlabeled", "someone"))
    assert main(PUSH_ENV) == "block"
    events.append(_label("labeled", "someone"))
    assert main(PUSH_ENV) == "block"
    events.extend([_label("unlabeled", "razzant"), OWNER_ADDS])
    assert main(PUSH_ENV) == "approved"
    assert calls == [PULLS_URL, _events_url(7)] * 4


@pytest.mark.parametrize("last_event,expected", [
    (_label("unlabeled", "someone"), "block"),
    (OWNER_ADDS, "approved"),
])
def test_push_reads_all_associated_pr_and_label_event_pages(monkeypatch, last_event, expected):
    from scripts.book_growth_mode import main

    next_pulls = PULLS_URL + "&page=2"
    next_events = _events_url(8) + "&page=2"
    calls = _github_pages(monkeypatch, {
        PULLS_URL: ([_pull(7, merged=False)], f'<{next_pulls}>; rel="next"'),
        next_pulls: ([_pull(8)], ""),
        _events_url(8): ([OWNER_ADDS, _label("unlabeled", "someone")], f'<{next_events}>; rel="next"'),
        next_events: ([last_event], ""),
    })
    assert main(PUSH_ENV) == expected
    assert calls == [PULLS_URL, next_pulls, _events_url(8), next_events]


def test_push_checks_every_matching_pr_instead_of_taking_the_first(monkeypatch):
    from scripts.book_growth_mode import main

    calls = _github_pages(monkeypatch, {
        PULLS_URL: ([_pull(7), _pull(8)], ""),
        _events_url(7): ([], ""), _events_url(8): ([OWNER_ADDS], ""),
    })
    assert main(PUSH_ENV) == "approved"
    assert calls == [PULLS_URL, _events_url(7), _events_url(8)]


@pytest.mark.parametrize("failed_endpoint", [PULLS_URL, _events_url(7)])
def test_unreadable_push_evidence_leaves_growth_strict(monkeypatch, failed_endpoint, capsys):
    from scripts import book_growth_mode

    pages = {PULLS_URL: ([_pull(7)], ""), _events_url(7): ([OWNER_ADDS], "")}
    _github_pages(monkeypatch, pages)
    read = book_growth_mode.urllib.request.urlopen

    def unavailable(request, timeout):
        if request.full_url == failed_endpoint:
            raise OSError("unavailable")
        return read(request, timeout)

    monkeypatch.setattr(book_growth_mode.urllib.request, "urlopen", unavailable)
    assert book_growth_mode.main(PUSH_ENV) == "block"
    assert "no exception applied" in capsys.readouterr().out


@pytest.mark.parametrize("env,expected", [
    (PUSH_ENV, "block"),
    ({**PUSH_ENV, "SHA": "", "PR_NUMBER": "7"}, "block"),  # no arbitrary PR-number fallback
    ({**PUSH_ENV, "REPO": "someone/fork"}, "warn"),
    ({**PUSH_ENV, "REF": "refs/heads/main"}, "warn"),
    ({**PUSH_ENV, "REF": "refs/heads/ouroboros-stable"}, "warn"),
    ({**PUSH_ENV, "REF": "refs/tags/v7.6.0"}, "warn"),
    ({**PUSH_ENV, "EVENT": "workflow_dispatch"}, "warn"),
    ({}, "warn"),
])
def test_only_official_development_pushes_are_strict(monkeypatch, env, expected):
    from scripts.book_growth_mode import main

    calls = _github_pages(monkeypatch, {PULLS_URL: ([], "")})
    assert main(env) == expected
    assert calls == ([PULLS_URL] if env == PUSH_ENV else [])


@pytest.mark.parametrize("approved,tip,passes", [
    (False, 101, False), (True, 101, True), (False, 100, True), (False, 99, True),
])
def test_push_mode_reaches_the_size_lane_and_only_growth_fails(monkeypatch, capsys, tmp_path, approved, tip, passes):
    import sys
    from scripts.book_growth_mode import main

    _github_pages(monkeypatch, {
        PULLS_URL: ([_pull(7)], ""), _events_url(7): ([OWNER_ADDS] if approved else [], ""),
    })
    output = tmp_path / "step-output"
    main({**PUSH_ENV, "GITHUB_OUTPUT": str(output)})
    monkeypatch.setenv(MODE_ENV, output.read_text(encoding="utf-8").strip().removeprefix("mode="))
    capsys.readouterr()
    monkeypatch.setenv("GITHUB_EVENT_NAME", "push")
    monkeypatch.setenv("GITHUB_REF", PUSH_ENV["REF"])
    consumer = sys.modules[__name__]
    monkeypatch.setattr(consumer, "growth_base", lambda repo, ref: "base")
    monkeypatch.setattr(consumer, "composed_bytes", lambda repo, book_id, ref="": 100 if ref else tip)
    if passes:
        test_a_change_does_not_grow_a_reference_book(capsys)
        if tip <= 100:
            assert capsys.readouterr().out == ""
    else:
        with pytest.raises(AssertionError, match="grew by 1 bytes"):
            test_a_change_does_not_grow_a_reference_book(capsys)
