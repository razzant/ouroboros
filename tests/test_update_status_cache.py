"""Target-bound passive managed-update cache regressions."""

import ouroboros.gateway.control as control
import supervisor.git_ops as git_ops


CURRENT = "a" * 40
LATEST = "b" * 40


def _wire(monkeypatch, *, cache_channel="stable", cache_ref="refs/ouroboros-managed/tags/v6.87.5", ancestor=False):
    import ouroboros.update_channels as update_channels

    monkeypatch.setattr(update_channels, "get_update_channel", lambda settings=None: "stable")
    monkeypatch.setattr(git_ops, "_read_managed_repo_meta", lambda: {"managed_remote_name": "managed"})
    monkeypatch.setattr(git_ops, "managed_branch_defaults", lambda: ("ouroboros", "ouroboros-stable"))
    monkeypatch.setattr(git_ops, "_managed_update_target", lambda: ("managed", "main", "managed/main"))
    monkeypatch.setattr(
        git_ops,
        "_resolve_managed_update_target",
        lambda *_args: ("refs/ouroboros-managed/tags/v6.87.5", LATEST, ""),
    )
    monkeypatch.setattr(
        git_ops,
        "load_state",
        lambda: {
            "managed_update_cache": {
                "remote": "managed",
                "remote_branch": "main",
                "target_ref": cache_ref,
                "update_channel": cache_channel,
                "available": True,
                "safe_to_apply": True,
                "latest_sha": LATEST,
                "latest_short_sha": LATEST[:8],
                "latest_message": "release",
                "behind": 1,
                "ahead": 0,
                "checked_at": "2026-08-03T00:00:00Z",
            }
        },
    )

    def fake_git_capture(cmd):
        if cmd[:3] == ["git", "remote", "get-url"]:
            return 0, "https://github.com/razzant/ouroboros", ""
        if cmd == ["git", "rev-parse", "--abbrev-ref", "HEAD"]:
            return 0, "ouroboros", ""
        if cmd == ["git", "rev-parse", "HEAD"]:
            return 0, CURRENT, ""
        if cmd == ["git", "status", "--porcelain"]:
            return 0, "", ""
        if cmd[:3] == ["git", "merge-base", "--is-ancestor"]:
            return (0 if ancestor else 1), "", ""
        if cmd[:4] == ["git", "rev-list", "--left-right", "--count"]:
            return 0, "0 1", ""
        if cmd == ["git", "show", f"{LATEST}:VERSION"]:
            return 0, "6.87.5", ""
        raise AssertionError(cmd)

    monkeypatch.setattr(git_ops, "git_capture", fake_git_capture)
    monkeypatch.setattr(control, "get_version", lambda: "6.87.5")


def test_passive_status_uses_cache_only_for_same_target_identity(monkeypatch):
    _wire(monkeypatch)

    status = git_ops.compute_managed_update_status(fetch=False)

    assert status["available"] is True
    assert status["latest_sha"] == LATEST
    assert status["from_cache"] is True


def test_passive_status_rejects_cache_from_previous_channel(monkeypatch):
    _wire(monkeypatch, cache_channel="qa", cache_ref="managed/ouroboros-stable")

    status = git_ops.compute_managed_update_status(fetch=False)

    assert status["available"] is False
    assert status.get("from_cache") is not True


def test_passive_status_rejects_consumed_target(monkeypatch):
    _wire(monkeypatch, ancestor=True)

    status = git_ops.compute_managed_update_status(fetch=False)

    assert status["available"] is False
    assert status.get("from_cache") is not True


def test_payload_reads_version_from_pinned_sha(monkeypatch):
    monkeypatch.setattr(
        git_ops,
        "compute_managed_update_status",
        lambda fetch=False: {"latest_sha": LATEST, "target_ref": "managed/main"},
    )
    seen = []

    def fake_capture(cmd):
        seen.append(cmd)
        return 0, "6.87.5", ""

    monkeypatch.setattr(git_ops, "git_capture", fake_capture)
    monkeypatch.setattr(control, "get_version", lambda: "6.87.5")

    payload = control._managed_update_payload(fetch=False, include_tags=False)

    assert payload["latest_version"] == "6.87.5"
    assert ["git", "show", f"{LATEST}:VERSION"] in seen
    assert ["git", "show", "managed/main:VERSION"] not in seen


def test_failed_remote_check_is_typed_not_reported_as_current(monkeypatch):
    _wire(monkeypatch)
    monkeypatch.setattr(git_ops, "ensure_official_update_remote", lambda: (True, ""))
    monkeypatch.setattr(git_ops, "git_fetch_bounded", lambda _remote: (124, "", "timed out"))

    status = git_ops.compute_managed_update_status(fetch=True)

    assert status["check_ok"] is False
    assert status["available"] is False
    assert any(item.startswith("fetch_error:") for item in status["warnings"])


def test_passive_status_exposes_cache_checked_at_without_from_cache(monkeypatch):
    """R9/R8 truthfulness: a passive read carries the last real check's
    timestamp even when the cached target is consumed, and the timestamp alone
    never claims cached AVAILABILITY (`from_cache` keeps its narrow meaning)."""
    _wire(monkeypatch, ancestor=True)  # consumed target: overlay must not fire
    state = git_ops.compute_managed_update_status(fetch=False)
    assert state["checked_at"] == "2026-08-03T00:00:00Z"
    assert not state.get("from_cache")
    assert not state.get("available")


def test_status_exposes_official_repo_url_from_remote(monkeypatch):
    _wire(monkeypatch)
    state = git_ops.compute_managed_update_status(fetch=False)
    assert state["official_repo_url"] == "https://github.com/razzant/ouroboros"


def test_payload_carries_minimal_update_tx_projection(monkeypatch):
    """A re-opened panel can see an active assisted resolution instead of
    reading as ordinary state while a second apply would 409."""
    import supervisor.update_merge as update_merge

    _wire(monkeypatch)
    monkeypatch.setattr(control, "get_version", lambda: "6.87.5")
    monkeypatch.setattr(git_ops, "list_official_update_tags", lambda: [])
    monkeypatch.setattr(
        update_merge, "active_update_tx",
        lambda: {"phase": "assisted_running", "task_id": "task-77"},
    )
    payload = control._managed_update_payload(fetch=False, include_tags=False)
    assert payload["update_tx"] == {
        "active": True, "phase": "assisted_running",
        "task_id": "task-77", "restart_required": False,
    }
    monkeypatch.setattr(update_merge, "active_update_tx", lambda: {})
    payload = control._managed_update_payload(fetch=False, include_tags=False)
    assert payload["update_tx"] == {"active": False}


def test_list_versions_emits_peeled_commit_sha(monkeypatch):
    """Annotated tags peel %(*objectname) to the COMMIT (never the tag object),
    lightweight tags fall back to %(objectname), and a tab-bearing subject
    survives the maxsplit (wave-2 review finding: the restore-list merge joins
    tags to commits by this sha)."""
    raw = (
        "v2.0.0\t2026-08-30T13:14:00+00:00\t" + "t" * 40 + "\t" + "c" * 40 + "\trelease\twith tab\n"
        + "light\t2026-08-29T10:00:00+00:00\t" + "d" * 40 + "\t\tplain subject\n"
    )
    monkeypatch.setattr(git_ops, "git_capture", lambda cmd: (0, raw, ""))
    rows = git_ops.list_versions()
    assert rows[0]["sha"] == "c" * 40  # peeled commit, not the tag object
    assert rows[0]["message"] == "release\twith tab"
    assert rows[1]["sha"] == "d" * 40  # lightweight tag: objectname IS the commit
    assert rows[1]["tag"] == "light"


def test_failed_divergence_check_mints_no_checked_at(monkeypatch):
    """A check whose rev-list read failed is NOT a completed check: it must not
    stamp checked_at (or clobber the cache), or a later passive read would
    present the failure as a verified up-to-date (wave-2 critical)."""
    _wire(monkeypatch)
    saved = {}
    import supervisor.state as sup_state
    monkeypatch.setattr(sup_state, "update_state", lambda fn: saved.setdefault("wrote", True))
    monkeypatch.setattr(git_ops, "ensure_official_update_remote", lambda: (True, ""))

    def fake_git_capture(cmd):
        if cmd[:3] == ["git", "remote", "get-url"]:
            return 0, "https://github.com/razzant/ouroboros", ""
        if cmd == ["git", "rev-parse", "--abbrev-ref", "HEAD"]:
            return 0, "ouroboros", ""
        if cmd == ["git", "rev-parse", "HEAD"]:
            return 0, CURRENT, ""
        if cmd == ["git", "status", "--porcelain"]:
            return 0, "", ""
        if cmd[:2] == ["git", "log"]:
            return 0, "msg", ""
        if cmd[:4] == ["git", "rev-list", "--left-right", "--count"]:
            return 1, "", "boom"
        return 1, "", "unexpected"

    monkeypatch.setattr(git_ops, "git_capture", fake_git_capture)
    state = git_ops.compute_managed_update_status(fetch=True)
    assert state["check_ok"] is False
    assert "checked_at" not in state
    assert "wrote" not in saved  # the last good cache is not clobbered


def test_official_repo_url_strips_credentials():
    assert git_ops._public_repo_url("https://user:tok@github.com/razzant/ouroboros") == "https://github.com/razzant/ouroboros"
    assert git_ops._public_repo_url("https://x-access-token:abc@github.com/o/r.git") == "https://github.com/o/r.git"
    assert git_ops._public_repo_url("git@github.com:razzant/ouroboros.git") == "github.com:razzant/ouroboros.git"
    assert git_ops._public_repo_url("https://github.com/razzant/ouroboros") == "https://github.com/razzant/ouroboros"


def test_passive_overlay_recomputes_availability_after_head_rollback(monkeypatch):
    """A cached "current" verdict must not certify a HEAD that later moved
    below the checked official tip: availability is recomputed against the
    cached sha on every passive read (final-review finding, 2026-08-31)."""
    _wire(monkeypatch)
    base_state = git_ops.load_state()
    base_state["managed_update_cache"].update({"available": False, "behind": 0, "safe_to_apply": False})
    monkeypatch.setattr(git_ops, "load_state", lambda: base_state)
    state = git_ops.compute_managed_update_status(fetch=False)
    assert state["available"] is True
    assert state["from_cache"] is True
    assert state["latest_sha"] == LATEST
    assert state["behind"] == 1


def test_passive_read_hides_checked_at_when_cached_tip_is_unresolvable(monkeypatch):
    """A timestamp over a cached tip that no longer resolves locally would let
    the panel claim "up to date, checked N ago" over a check whose availability
    cannot be validated any more (final-review finding, 2026-08-31)."""
    _wire(monkeypatch)
    real = git_ops.git_capture

    def broken_divergence(cmd):
        if cmd[:4] == ["git", "rev-list", "--left-right", "--count"]:
            return 128, "", "bad revision"
        return real(cmd)

    monkeypatch.setattr(git_ops, "git_capture", broken_divergence)
    state = git_ops.compute_managed_update_status(fetch=False)
    assert "checked_at" not in state
    assert not state.get("available")


def test_passive_payload_projects_letter_without_writing_it(monkeypatch):
    import ouroboros.update_letter as update_letter

    _wire(monkeypatch)
    monkeypatch.setattr(
        update_letter, "refresh_after_check",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("a passive read must not write the letter")),
    )
    monkeypatch.setattr(update_letter, "read_record", lambda drive_root=None: {
        "key": {"base_sha": CURRENT, "target_sha": LATEST, "update_channel": "stable",
                "target_ref": "refs/ouroboros-managed/tags/v6.87.5"},
        "checked_head_sha": CURRENT, "state": "ready", "text": "one paragraph",
        "author_version": "6.87.4", "target_version": "6.87.5", "written_at": "2026-08-03T00:00:00Z",
        "attempt_id": "att", "error_kind": "", "error_text": "", "last_good": None,
    })

    payload = control._managed_update_payload(fetch=False, include_tags=False)

    assert payload["letter"]["relation"] == "pending"
    assert payload["letter"]["text"] == "one paragraph"
    assert payload["letter"]["author_version"] == "6.87.4"


def test_fetching_payload_writes_the_letter_through_the_one_seam(monkeypatch):
    import ouroboros.update_letter as update_letter

    _wire(monkeypatch)
    status = {"check_ok": True, "available": True, "current_sha": CURRENT, "latest_sha": LATEST,
              "update_channel": "stable", "target_ref": "refs/ouroboros-managed/tags/v6.87.5"}
    monkeypatch.setattr(git_ops, "compute_managed_update_status", lambda fetch: dict(status, fetched=fetch))
    calls = []
    monkeypatch.setattr(update_letter, "refresh_after_check", lambda st, **k: calls.append(st))
    monkeypatch.setattr(update_letter, "read_record", lambda drive_root=None: None)

    payload = control._managed_update_payload(fetch=True, include_tags=False)

    assert calls and calls[0]["fetched"] is True
    assert payload["letter"] is None and payload["latest_version"] == "6.87.5"
