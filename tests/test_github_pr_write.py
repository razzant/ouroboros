"""The PR creation, field-edit and draft methods of ``github_write``.

The fake provider accepts writes, so these tests isolate the wrapper's prechecks and readback:
stale head/base, an open duplicate or an already-ready pull request prevents an unnecessary write,
while a fresh read admits it exactly once. Negative controls cover lost answers, failed readbacks
and malformed JSON. A correctly authored shell script can compose the same operations; these
are behavior regressions, not a benchmark against that valid alternative.
Once a mutation was sent, the read-back speaks and the outcome stays unknown; the one exception is the field edit's
definite HTTP 4xx with every sent field read back as the precheck read it, which is not applied.
"""

import json
import subprocess
from types import SimpleNamespace

import pytest

from ouroboros.github_pr_identity import pull_request_fields_from_cli, pull_request_fields_from_rest, same_pull_request
from ouroboros.tools import github, github_pr_write
from tests.test_github_project_target import BASE, MOVED, READ_ORG_REFUSAL, SHA, _context

BASE_SHA = "c" * 40
NEW_BASE_SHA = "d" * 40
URL = f"{BASE}/pull/7"
TIMEOUT = github.GhResult(False, "⚠️ GH_TIMEOUT: exceeded 60s.", None, None, "timeout")
REFUSED = github.GhResult(False, "⚠️ GH_ERROR: HTTP 422: Validation Failed (https://api.github.example/repos/owner/selected/pulls)",
                          1, 422, "exit")
NOT_JSON = github.GhResult(True, "<html>maintenance</html>", 0, None, "")
SERVER_ERROR = github.GhResult(False, "⚠️ GH_ERROR: gh: HTTP 502", 1, 502, "exit")
# The field edit: one REST PATCH addressed by the read pull request's URL, on its host.
PATCH = ["api", "repos/owner/selected/pulls/7", "--method", "PATCH", "--input", "-", "--hostname", "github.example"]


def _answer(payload):
    return github.GhResult(True, json.dumps(payload), 0, None, "")


class FakeGitHub:
    """One repository with branches, zero or one open pull request, and a gh that accepts every blind write."""

    def __init__(self, *, pr=None, refs=None, failing=None, view_fails_after_write=False, repo_hint=False):
        self.refs = refs if refs is not None else {"owner:feature": SHA, "owner:main": BASE_SHA, "owner:release": NEW_BASE_SHA,
                                                   "fork-user:feature": MOVED}
        self.pr = pr  # The one open pull request GitHub holds, as `gh pr view --json` would answer it.
        self.failing, self.view_fails_after_write, self.calls, self.wrote = failing or {}, view_fails_after_write, [], False

    def __call__(self, args, ctx, timeout=30, input_data=None, *, repo=github._GENERIC_TRANSPORT, **_capture):
        self.calls.append(SimpleNamespace(args=list(args), repo=repo, input=input_data, timeout=timeout))
        kind = self.kind(args)
        if kind in self.failing:
            failure = self.failing[kind]
            return failure.pop(0) if isinstance(failure, list) else failure
        if kind == "repo":
            return _answer({"nameWithOwner": "owner/selected", "url": BASE})
        if kind == "ref":
            key = args[1].split("/repos/")[-1].removeprefix("repos/").split("/git/ref/heads/")
            owner, branch = key[0].split("/")[0], key[1]
            sha = self.refs.get(f"{owner}:{branch}")
            if sha is None:
                return github.GhResult(False, "⚠️ GH_ERROR: gh: Not Found (HTTP 404)", 1, 404, "exit")
            return _answer({"ref": f"refs/heads/{branch}", "object": {"sha": sha}})
        if kind == "list":
            head = next(a for a in args if a.startswith("--head=")).removeprefix("--head=")
            base = next(a for a in args if a.startswith("--base=")).removeprefix("--base=")
            rows = [self.pr] if self.pr and self.pr["headRefName"] == head and self.pr["baseRefName"] == base and self.pr["state"] == "OPEN" else []
            return _answer(rows)
        if kind == "view":
            if self.wrote and self.view_fails_after_write:
                return github.GhResult(False, "⚠️ GH_ERROR: gh: Service Unavailable (HTTP 503)", 1, 503, "exit")
            return _answer(self.pr) if self.pr else github.GhResult(False, "⚠️ GH_ERROR: gh: Could not resolve to a PullRequest (HTTP 404)", 1, 404, "exit")
        self.wrote = True
        flags = {a.split("=", 1)[0]: a.split("=", 1)[1] for a in args if a.startswith("--") and "=" in a}
        if kind == "create":
            head = flags["--head"]
            owner, _sep, branch = head.rpartition(":")
            owner = owner or "owner"
            self.pr = {"number": 7, "url": URL, "state": "OPEN", "title": flags["--title"], "body": input_data,
                       "isDraft": "--draft" in args, "headRefName": branch, "baseRefName": flags["--base"],
                       "headRefOid": self.refs[f"{owner}:{branch}"], "headRepositoryOwner": {"login": owner}}
            return github.GhResult(True, URL, 0, None, "")
        if kind == "edit":
            fields = json.loads(input_data)
            self.pr.update({"baseRefName" if key == "base" else key: value for key, value in fields.items()})
            return _answer({"number": 7, "html_url": URL, "title": self.pr["title"], "body": self.pr["body"]})
        self.pr["isDraft"] = "--undo" in args
        return github.GhResult(True, "", 0, None, "")

    @staticmethod
    def kind(args):
        if args[:2] == ["repo", "view"]:
            return "repo"
        if args[0] == "api":
            return "edit" if "--method" in args and args[args.index("--method") + 1] == "PATCH" else "ref"
        return {"list": "list", "view": "view", "create": "create", "edit": "edit", "ready": "ready"}[args[1]]

    def kinds(self):
        return [self.kind(call.args) for call in self.calls]

    def writes(self):
        return [call for call in self.calls if self.kind(call.args) in ("create", "edit", "ready")]


def _open_pr(**over):
    return {"number": 7, "url": URL, "state": "OPEN", "title": "Old title", "body": "Old body", "isDraft": True,
            "headRefName": "feature", "baseRefName": "main", "headRefOid": SHA, "headRepositoryOwner": {"login": "owner"}, **over}


@pytest.fixture
def fake(monkeypatch):
    monkeypatch.setattr(github, "github_token_from_env_or_settings", lambda: "fixture-token")

    def install(**kwargs):
        gh = FakeGitHub(**kwargs)
        monkeypatch.setattr(github_pr_write, "_gh_run", gh)
        return gh
    return install


@pytest.fixture
def ctx(tmp_path):
    context, _project = _context(tmp_path, "queued")
    context._active_builtin_tool_result = None  # Handler-level calls publish their typed result here.
    return context


def _typed(ctx):
    return ctx._active_builtin_tool_result


def _method_args(args):
    """Project the contributor's fixtures onto the consolidated public argument contract."""
    args = dict(args)
    expect = dict(args.pop("expect", {}) or {})
    for old, new in (("expected_head_sha", "head_sha"), ("expected_base_sha", "base_sha")):
        if old in args:
            expect[new] = args.pop(old)
    if "number" in args:
        args["target"] = args.pop("number")
    if "ready" in args:
        ready = args.pop("ready")
        args["draft"] = not ready if isinstance(ready, bool) else ready
    return {**args, "expect": expect}


def _pr(ctx, method, **args):
    entry = next(entry for entry in github.get_tools() if entry.name == "github_write")
    text = entry.handler(ctx, method=method, kind="pr", **_method_args(args))
    if "\n{" not in text:
        return text  # Typed argument or repository refusal before a write.
    receipt = json.loads(text.partition("\n")[2])
    assert {"effect", "target", "url", "id", "actor", "at", "outcome", "readback"} <= receipt.keys()
    assert receipt["actor"] is None and receipt["at_source"] == "local"
    assert receipt["outcome"] == {"completed": "confirmed", "completed_no_effect": "not_applied", "unknown": "unknown"}[
        _typed(ctx).meta["operation_outcome"]]
    if receipt["outcome"] == "unknown":
        assert "do not resend" in text and "github_read(" in receipt["lookup"]
    return receipt["detail"]


CREATE = dict(title="Add thing", head="feature", base="main", expected_head_sha=SHA, expected_base_sha=BASE_SHA,
              body="Why:\r\n- because\n")


# --- create_github_pr -------------------------------------------------------------------------------------------------

def test_create_prechecks_then_sends_one_request_then_reads_back_every_field(fake, ctx):
    gh = fake()

    text = _pr(ctx, "create_pr", **CREATE, repo="")

    assert gh.kinds() == ["repo", "ref", "ref", "list", "create", "view"]
    assert gh.calls[1].args == ["api", "repos/owner/selected/git/ref/heads/feature", "--hostname", "github.example"]
    assert gh.calls[2].args == ["api", "repos/owner/selected/git/ref/heads/main", "--hostname", "github.example"]
    assert gh.calls[1].repo is gh.calls[2].repo is github._GENERIC_TRANSPORT and gh.calls[0].repo == gh.calls[4].repo == ""
    create = gh.writes()
    assert len(create) == 1
    assert create[0].args == ["pr", "create", "--title=Add thing", "--head=feature", "--base=main", "--body-file=-"]
    assert create[0].input == "Why:\r\n- because\n" and create[0].timeout == 60
    assert text == (f"✅ PR #7 created: {URL}\n"
                    "Read-back: title confirmed; body confirmed (16 chars); base main confirmed; draft=False confirmed; "
                    f"head owner:feature confirmed; head at {SHA} confirmed")
    assert (_typed(ctx).status, _typed(ctx).code, _typed(ctx).meta["operation_outcome"]) == ("ok", "OK", "completed")


def test_create_draft_and_cross_owner_head_ride_as_flags_and_identity(fake, ctx):
    gh = fake()
    text = _pr(ctx, "create_pr", **{**CREATE, "head": "fork-user:feature", "expected_head_sha": MOVED, "draft": True})
    assert gh.calls[1].args[1] == "repos/fork-user/selected/git/ref/heads/feature"
    assert gh.writes()[0].args == ["pr", "create", "--title=Add thing", "--head=fork-user:feature", "--base=main", "--body-file=-", "--draft"]
    assert "draft=True confirmed; head fork-user:feature confirmed" in text


@pytest.mark.parametrize("over,fragment", [
    ({"expected_head_sha": MOVED}, f"head owner:feature is at {SHA}, you expected {MOVED}; nothing was created"),
    ({"expected_base_sha": NEW_BASE_SHA}, f"base main is at {BASE_SHA}, you expected {NEW_BASE_SHA}; nothing was created"),
])
def test_create_refuses_a_stale_head_or_base_where_the_blind_command_would_have_created(fake, ctx, over, fragment):
    """Paired: the fake gh accepts ``pr create`` unconditionally; the writer never sends it on a stale precondition."""
    gh = fake()
    text = _pr(ctx, "create_pr", **{**CREATE, **over})
    assert gh.writes() == [] and gh.pr is None
    assert text.startswith("⚠️ PR_PRECHECK_FAILED: ") and fragment in text
    assert (_typed(ctx).code, _typed(ctx).meta["operation_outcome"]) == ("TOOL_ERROR", "completed_no_effect")
    # Negative control: the same request with the commits GitHub holds is sent exactly once.
    gh = fake()
    assert _pr(ctx, "create_pr", **CREATE).startswith("✅ PR #7 created")
    assert len(gh.writes()) == 1


def test_create_refuses_when_a_pull_request_for_this_head_and_base_is_already_open(fake, ctx):
    gh = fake(pr=_open_pr(headRefOid=MOVED))
    text = _pr(ctx, "create_pr", **CREATE)
    assert gh.writes() == []
    assert text == (f"⚠️ PR_EXISTS: PR #7 {URL}: open, owner:feature → main, head {MOVED}, draft=True, title 'Old title' is already "
                    "open for owner:feature into main in the active Project repository; nothing was created. Change it with "
                    "github_write(method='update', kind='pr', target=7).")
    assert _typed(ctx).meta["operation_outcome"] == "completed_no_effect"
    # A same-named branch in another account is another head: not a duplicate.
    gh = fake(pr=_open_pr(headRepositoryOwner={"login": "fork-user"}, headRefOid=MOVED))
    assert _pr(ctx, "create_pr", **CREATE).startswith("✅ PR #7 created")


def test_create_accepted_but_readback_failed_keeps_the_identity_and_marks_fields_unconfirmed(fake, ctx):
    gh = fake(view_fails_after_write=True)
    text = _pr(ctx, "create_pr", **CREATE)
    assert len(gh.writes()) == 1
    assert text == (f"✅ PR #7 created: {URL}\n⚠️ Read-back failed (⚠️ GH_ERROR: gh: Service Unavailable (HTTP 503)): the pull "
                    "request's fields are unconfirmed; read it with github_read(method='get', kind='pr', target=7).")
    assert (_typed(ctx).status, _typed(ctx).meta["operation_outcome"]) == ("ok", "unknown")


def test_create_with_a_lost_answer_sends_once_and_settles_by_reading(fake, ctx):
    """The answer timed out but GitHub holds the pull request: one request, no retry, the identity read back."""
    gh = fake(failing={"create": TIMEOUT})
    gh.pr = None

    def created_anyway(args, ctx_, **kwargs):  # GitHub processed the request whose answer was lost.
        res = FakeGitHub.__call__(gh, args, ctx_, **kwargs)
        if gh.kind(args) == "create":
            gh.pr = _open_pr(title="Add thing", body=CREATE["body"], isDraft=False)
        return res
    github_pr_write._gh_run = created_anyway
    text = _pr(ctx, "create_pr", **CREATE)
    assert len(gh.writes()) == 1 and gh.kinds()[-1] == "list"
    assert text.startswith("⚠️ PR_CREATE_UNKNOWN: the create request for owner:feature into main in the active Project repository "
                           "timed out: ⚠️ GH_TIMEOUT: exceeded 60s.. One request was sent and not retried; its effect is unknown "
                           "until GitHub shows it. Read-back: PR #7 ")
    assert text.endswith("is open for this head and base at the expected commit — this call's effect or an earlier one; do not "
                         "create again. Compare its fields and use github_write(method='update', kind='pr', target=7) if they differ.")
    assert (_typed(ctx).code, dict(_typed(ctx).meta)) == ("TOOL_ERROR", {"operation_outcome": "unknown"})


@pytest.mark.parametrize("failure,facts", [
    (REFUSED, "ended with exit 1, HTTP 422: ⚠️ GH_ERROR: HTTP 422: Validation Failed (https://api.github.example/repos/owner/selected/pulls)"),
    (TIMEOUT, "timed out: ⚠️ GH_TIMEOUT: exceeded 60s."),
])
def test_create_send_failure_keeps_the_exit_facts_and_never_attests_no_effect(fake, ctx, failure, facts):
    """A nonzero exit is not proof GitHub rejected the request: the facts are kept, the read-back is dated, no no-effect."""
    gh = fake(failing={"create": failure}, repo_hint=True)
    text = _pr(ctx, "create_pr", **CREATE, repo="github.example/owner/selected")
    assert len(gh.writes()) == 1
    assert text == (f"⚠️ PR_CREATE_UNKNOWN: the create request for owner:feature into main in github.example/owner/selected {facts}. "
                    "One request was sent and not retried; its effect is unknown until GitHub shows it. Read-back: GitHub listed no "
                    "open pull request from owner:feature into main at that moment; a request it is still processing would appear "
                    "later, so read again (github_read(method='list', kind='pr', repo='github.example/owner/selected')) before creating.")
    assert (_typed(ctx).status, _typed(ctx).code, dict(_typed(ctx).meta)) == ("error", "TOOL_ERROR", {"operation_outcome": "unknown"})


def test_create_lost_answer_and_failed_readback_says_unknown(fake, ctx):
    gh = fake(failing={"create": TIMEOUT, "list": [_answer([]), REFUSED]})
    text = _pr(ctx, "create_pr", **CREATE)
    assert len(gh.writes()) == 1
    assert "Read-back failed too (" in text and text.endswith("(github_read(method='list', kind='pr')) before creating again.")
    assert _typed(ctx).meta["operation_outcome"] == "unknown"


def test_create_precheck_read_failures_send_nothing(fake, ctx):
    gh = fake(failing={"ref": NOT_JSON})
    text = _pr(ctx, "create_pr", **CREATE)
    assert gh.writes() == [] and text == ("⚠️ PR_PRECHECK_FAILED: head owner:feature could not be read (⚠️ GH_ERROR: the ref answer "
                                          "for owner/selected feature is malformed: <html>maintenance</html>); nothing was created.")
    gh = fake(failing={"ref": _answer({"object": ["not", "a", "mapping"]})})
    text = _pr(ctx, "create_pr", **CREATE)
    assert gh.writes() == [] and "the ref answer for owner/selected feature is malformed" in text
    gh = fake(refs={"owner:main": BASE_SHA})
    text = _pr(ctx, "create_pr", **CREATE)
    assert text == ("⚠️ PR_PRECHECK_FAILED: head owner:feature could not be read (⚠️ GH_ERROR: gh: Not Found (HTTP 404)); nothing "
                    "was created.")
    gh = fake(failing={"list": NOT_JSON})
    assert "the open pull requests from feature into main could not be read" in _pr(ctx, "create_pr", **CREATE)
    assert gh.writes() == [] and _typed(ctx).meta["operation_outcome"] == "completed_no_effect"
    gh = fake(failing={"repo": github.GhResult(False, "⚠️ GH_TARGET_REQUIRED: this Project has no repository directory", None, None, "target")})
    assert _pr(ctx, "create_pr", **CREATE).startswith("⚠️ GH_TARGET_REQUIRED")


@pytest.mark.parametrize("over,fragment", [
    ({"head": "-rf"}, "head is an existing branch name"),
    ({"base": "main branch"}, "base is the branch name"),
    ({"expected_base_sha": SHA}, "equals expect.base_sha"),
    ({"body": None}, "body is a string"),
    ({"draft": "yes"}, "draft is a boolean"),
    ({"head": "main"}, "head and base are the same branch"),
])
def test_create_argument_refusals_are_typed_and_send_nothing(fake, ctx, over, fragment):
    gh = fake()
    text = _pr(ctx, "create_pr", **{**CREATE, **over})
    assert fragment in text and gh.writes() == []
    assert (_typed(ctx).code, _typed(ctx).meta["operation_outcome"]) == ("TOOL_ARG_ERROR", "completed_no_effect")


# --- update_github_pr -------------------------------------------------------------------------------------------------

def test_update_omitted_body_is_left_and_empty_body_clears_in_one_edit(fake, ctx):
    gh = fake(pr=_open_pr())
    text = _pr(ctx, "update", number=7, expected_head_sha=SHA, title="New title")
    assert gh.writes()[0].args == PATCH and json.loads(gh.writes()[0].input) == {"title": "New title"}
    assert gh.pr["body"] == "Old body" and text == f"✅ PR #7 updated: {URL}\nRead-back: title confirmed; head at {SHA} confirmed"

    gh = fake(pr=_open_pr())
    text = _pr(ctx, "update", number=7, expected_head_sha=SHA, body="")
    assert gh.writes()[0].args == PATCH and json.loads(gh.writes()[0].input) == {"body": ""}
    assert gh.pr["body"] == "" and "Read-back: body confirmed (0 chars)" in text
    assert gh.kinds() == ["view", "edit", "view"]


def test_a_null_body_is_a_valid_empty_body_but_a_missing_body_key_never_confirms_anything(fake, ctx):
    """GitHub answers null for an empty body: that compares as ""; an answer WITHOUT the key is malformed, so a
    requested empty body can never be 'confirmed' by an answer that simply omitted the field."""
    gh = fake(pr=_open_pr(body=None))
    text = _pr(ctx, "update", number=7, expected_head_sha=SHA, title="New title", body="")
    assert len(gh.writes()) == 1 and "body confirmed (0 chars)" in text

    def drop_body(args, ctx_, **kwargs):
        res = FakeGitHub.__call__(gh, args, ctx_, **kwargs)
        if gh.kind(args) == "view":
            return _answer({key: value for key, value in gh.pr.items() if key != "body"})
        return res
    gh = fake(pr=_open_pr())
    github_pr_write._gh_run = drop_body
    text = _pr(ctx, "update", number=7, expected_head_sha=SHA, body="")
    assert gh.writes() == [] and "(field 'body' is missing) is malformed" in text


def test_writer_descriptions_disclose_prechecks_and_the_same_name_fork_assumption():
    entry = next(entry for entry in github.get_tools() if entry.name == "github_write")
    assert "not a server-side compare-and-swap" in entry.schema["description"]
    assert "same repository name" in entry.schema["parameters"]["properties"]["head"]["description"]


def test_update_base_requires_its_expected_commit_and_checks_it(fake, ctx):
    gh = fake(pr=_open_pr())
    text = _pr(ctx, "update", number=7, expected_head_sha=SHA, base="release")
    assert "expect.base_sha is required with base" in text and gh.calls == []
    text = _pr(ctx, "update", number=7, expected_head_sha=SHA, base="release", expected_base_sha=BASE_SHA)
    assert gh.writes() == [] and f"base release is at {NEW_BASE_SHA}, you expected {BASE_SHA}; nothing was changed" in text
    text = _pr(ctx, "update", number=7, expected_head_sha=SHA, title="T", body="B", base="release",
                                     expected_base_sha=NEW_BASE_SHA)
    assert gh.calls[-3].args == ["api", "repos/owner/selected/git/ref/heads/release", "--hostname", "github.example"]
    assert gh.writes()[-1].args == PATCH and json.loads(gh.writes()[-1].input) == {"title": "T", "body": "B", "base": "release"}
    assert "Read-back: title confirmed; body confirmed (1 chars); base release confirmed" in text
    assert "already holds the requested values; nothing was changed" in _pr(
        ctx, "update", number=7, expected_head_sha=SHA, base="release", expected_base_sha=NEW_BASE_SHA)


def test_update_same_base_still_edits_title_and_body_once(fake, ctx):
    gh = fake(pr=_open_pr())
    text = _pr(ctx, "update", number=7, expected_head_sha=SHA, title="New title", body="New body",
                                     base="main", expected_base_sha=BASE_SHA)
    assert gh.kinds() == ["view", "edit", "view"]
    assert gh.writes()[0].args == PATCH and json.loads(gh.writes()[0].input) == {"title": "New title", "body": "New body"}
    assert gh.pr["title"] == "New title" and gh.pr["body"] == "New body"
    assert "Read-back: title confirmed; body confirmed (8 chars)" in text
    assert (_typed(ctx).status, _typed(ctx).meta["operation_outcome"]) == ("ok", "completed")


def test_update_only_unchanged_fields_reports_no_effect(fake, ctx):
    gh = fake(pr=_open_pr())
    text = _pr(ctx, "update", number=7, expected_head_sha=SHA, title="Old title", body="Old body",
                                     base="main", expected_base_sha=BASE_SHA)
    assert gh.kinds() == ["view"] and gh.writes() == []
    assert "nothing was changed" in text
    assert _typed(ctx).meta["operation_outcome"] == "completed_no_effect"


def test_update_checks_requested_current_base_after_title_edit(fake, ctx):
    gh = fake(pr=_open_pr())
    text = _pr(ctx, "update", number=7, expected_head_sha=SHA, title="New title",
                                     base="main", expected_base_sha=BASE_SHA)
    assert gh.kinds() == ["view", "edit", "view"]
    assert gh.writes()[0].args == PATCH and json.loads(gh.writes()[0].input) == {"title": "New title"}
    assert "title confirmed; base main confirmed" in text
    assert _typed(ctx).meta["operation_outcome"] == "completed"


def test_update_current_base_retargeted_during_edit_is_unknown(fake, ctx):
    gh = fake(pr=_open_pr())

    def retarget_after_edit(args, ctx_, **kwargs):
        result = FakeGitHub.__call__(gh, args, ctx_, **kwargs)
        if gh.kind(args) == "edit":
            gh.pr["baseRefName"] = "release"
        return result

    github_pr_write._gh_run = retarget_after_edit
    text = _pr(ctx, "update", number=7, expected_head_sha=SHA, title="New title",
                                     base="main", expected_base_sha=BASE_SHA)
    assert gh.writes()[0].args == PATCH and json.loads(gh.writes()[0].input) == {"title": "New title"}
    assert "title confirmed" in text and "base is 'release', you asked for 'main'" in text
    assert "does not hold every requested value" in text
    assert _typed(ctx).meta["operation_outcome"] == "unknown"


@pytest.mark.parametrize("pr,fragment", [
    (_open_pr(headRefOid=MOVED), f"the head of PR #7 is at {MOVED}, you expected {SHA}; nothing was changed"),
    (_open_pr(state="MERGED"), "is merged, not open; nothing was changed"),
])
def test_update_refuses_a_moved_head_or_a_closed_pull_request(fake, ctx, pr, fragment):
    gh = fake(pr=pr)
    text = _pr(ctx, "update", number=7, expected_head_sha=SHA, title="New")
    assert gh.writes() == [] and fragment in text and _typed(ctx).meta["operation_outcome"] == "completed_no_effect"


def test_update_refused_after_a_sent_field_changed_stays_unknown_and_says_what_changed(fake, ctx):
    """GitHub answered 422, yet the title read back changed since the precheck: no all-or-nothing edit is documented,
    so the outcome stays unknown and the read-back is described without saying which request changed it."""
    gh = fake(pr=_open_pr())

    def partial(args, ctx_, **kwargs):
        if gh.kind(args) == "edit":
            gh.calls.append(SimpleNamespace(args=list(args), repo=kwargs.get("repo"), input=kwargs.get("input_data"), timeout=60))
            gh.pr["title"] = "New title"
            return REFUSED
        return FakeGitHub.__call__(gh, args, ctx_, **kwargs)
    github_pr_write._gh_run = partial
    text = _pr(ctx, "update", number=7, expected_head_sha=SHA, title="New title", body="New body")
    assert len(gh.writes()) == 1
    assert text.startswith("⚠️ PR_UPDATE_UNKNOWN: the edit request for PR #7 in the active Project repository ended with exit 1, HTTP 422: ")
    assert text.endswith("GitHub answered with a refusal, yet the read-back differs from the precheck in a sent field: title changed "
                         "from 'Old title' to 'New title'. GitHub documents no all-or-nothing edit, so the read-back cannot tell which "
                         "request made that change; read the pull request again before any further edit.")
    assert _typed(ctx).meta["operation_outcome"] == "unknown"


def test_update_refusal_with_nothing_changed_is_not_applied_while_other_failures_stay_unknown(fake, ctx):
    gh = fake(pr=_open_pr(), failing={"edit": REFUSED})
    text = _pr(ctx, "update", number=7, expected_head_sha=SHA, title="New title")
    assert text == ("⚠️ GH_WRITE_NOT_APPLIED: GitHub refused the edit of PR #7 in the active Project repository: the request ended "
                    "with exit 1, HTTP 422: ⚠️ GH_ERROR: HTTP 422: Validation Failed (https://api.github.example/repos/owner/selected/"
                    "pulls). One request was sent and not retried; the read-back holds every sent field as the precheck read it, so "
                    "nothing was changed.")
    assert (_typed(ctx).code, _typed(ctx).meta["operation_outcome"]) == ("TOOL_ERROR", "completed_no_effect")
    assert len(gh.writes()) == 1
    for failure in (SERVER_ERROR, TIMEOUT):  # No definite refusal: a request GitHub is still processing could land.
        gh = fake(pr=_open_pr(), failing={"edit": failure})
        text = _pr(ctx, "update", number=7, expected_head_sha=SHA, title="New title")
        assert text.endswith("Read-back: none of the requested values was held at that moment (title is 'Old title', you sent "
                             "'New title'); a request GitHub is still processing could still land, so read again before resending.")
        assert _typed(ctx).meta["operation_outcome"] == "unknown" and len(gh.writes()) == 1
    gh = fake(pr=_open_pr(), failing={"edit": TIMEOUT})

    def late_readback(args, ctx_, **kwargs):
        if gh.kind(args) == "edit":
            gh.pr["title"] = "New title"
        return FakeGitHub.__call__(gh, args, ctx_, **kwargs)
    github_pr_write._gh_run = late_readback
    text = _pr(ctx, "update", number=7, expected_head_sha=SHA, title="New title")
    assert text.endswith("Read-back: GitHub holds every requested value (title confirmed); do not resend.")
    assert len(gh.writes()) == 1


def test_update_accepted_but_readback_not_holding_a_value_is_not_called_updated(fake, ctx):
    """gh exited 0 yet the read-back lacks the body: the result says accepted, not updated, and the outcome is unknown."""
    gh = fake(pr=_open_pr())

    def title_only(args, ctx_, **kwargs):
        res = FakeGitHub.__call__(gh, args, ctx_, **kwargs)
        if gh.kind(args) == "edit":
            gh.pr["body"] = "Old body"
        return res
    github_pr_write._gh_run = title_only
    text = _pr(ctx, "update", number=7, expected_head_sha=SHA, title="New title", body="New body")
    assert text == (f"⚠️ gh accepted the edit of PR #7 ({URL}), but the read-back does not hold every requested value.\n"
                    f"Read-back: title confirmed; head at {SHA} confirmed\n"
                    "⚠️ Read-back differs: body differs (8 chars held, 8 sent). Read the pull request again before relying on it.")
    assert (_typed(ctx).status, _typed(ctx).meta["operation_outcome"]) == ("ok", "unknown")


def test_head_movement_after_an_accepted_edit_is_a_note_not_a_failure(fake, ctx):
    gh = fake(pr=_open_pr())

    def moved(args, ctx_, **kwargs):
        res = FakeGitHub.__call__(gh, args, ctx_, **kwargs)
        if gh.kind(args) == "edit":
            gh.pr["headRefOid"] = MOVED
        return res
    github_pr_write._gh_run = moved
    text = _pr(ctx, "update", number=7, expected_head_sha=SHA, title="New title")
    assert text == f"✅ PR #7 updated: {URL}\nRead-back: title confirmed\nNote: head moved to {MOVED} since you read it (you expected {SHA})"
    assert _typed(ctx).meta["operation_outcome"] == "completed"


@pytest.mark.parametrize("answer,why", [
    (_answer({}), "field 'number' is None"),
    (_answer({**_open_pr(), "headRepositoryOwner": None}), "field 'headRepositoryOwner' is None"),
    (_answer({**_open_pr(), "body": 5}), "field 'body' is not text"),
])
def test_malformed_pull_request_answers_are_typed_refusals_before_any_write(fake, ctx, answer, why):
    """`{}` or a missing isDraft must never read as 'already ready' or as a confirmed state."""
    gh = fake(pr=_open_pr(), failing={"view": answer})
    text = _pr(ctx, "update", number=7, expected_head_sha=SHA, title="New")
    assert gh.writes() == []
    assert text == (f"⚠️ PR_PRECHECK_FAILED: pull request #7 could not be read (⚠️ GH_ERROR: the pull request answer ({why}) is "
                    f"malformed: {answer.text[:300]}); nothing was changed.")
    assert (_typed(ctx).code, _typed(ctx).meta["operation_outcome"]) == ("TOOL_ERROR", "completed_no_effect")
    gh = fake(pr=_open_pr(), failing={"view": answer})
    text = _pr(ctx, "update", number=7, ready=True, expected_head_sha=SHA)
    assert gh.writes() == [] and "already" not in text and text.startswith("⚠️ PR_PRECHECK_FAILED: pull request #7 could not be read")


@pytest.mark.parametrize("kwargs,fragment", [
    ({"number": 7, "expected_head_sha": SHA, "title": ""}, "title is one non-empty line"),
    ({"number": 7, "expected_head_sha": SHA, "body": 5}, "body is a string"),
    ({"number": 7, "expected_head_sha": SHA, "base": "-x", "expected_base_sha": SHA}, "base is a branch name"),
])
def test_update_argument_refusals(fake, ctx, kwargs, fragment):
    gh = fake(pr=_open_pr())
    assert fragment in _pr(ctx, "update", **kwargs) and gh.calls == []


# --- the field edit over the registry and the real transport: only the gh process is fake ----------------------------

def _process(code=0, stdout="", stderr=""):
    return SimpleNamespace(returncode=code, stdout=stdout, stderr=stderr)


def _refusal(status, message):
    """What `gh api` prints for an HTTP error: GitHub's JSON on stdout, its message and status on stderr."""
    return lambda _fields: _process(1, json.dumps({"message": message, "status": str(status)}), f"gh: {message} (HTTP {status})\n")


@pytest.fixture
def cli(ctx, monkeypatch):
    """``edit(**args)`` runs github_write update kind='pr' through the registry. ``answer(fields)`` answers the PATCH
    (default: GitHub applies the fields); ``views`` scripts later ``pr view`` answers (None: the PR as held);
    ``gh pr edit`` answers as it does for a token without read:org."""
    from ouroboros.tools.registry import ToolRegistry

    monkeypatch.setattr("ouroboros.safety.check_safety", lambda *_a, **_k: (True, ""))
    monkeypatch.setattr(github, "github_token_from_env_or_settings", lambda: "fixture-token")
    state = SimpleNamespace(pr=_open_pr(), calls=[], views=[], answer=None)

    def apply(fields):
        state.pr.update({"baseRefName" if key == "base" else key: value for key, value in fields.items()})
        return _process(stdout=json.dumps({"number": 7, "html_url": state.pr["url"], "title": state.pr["title"]}))

    def run(argv, **kwargs):
        state.calls.append((list(argv), kwargs))
        args = argv[1:]
        if args[:2] == ["pr", "view"]:
            return (state.views.pop(0) if state.views else None) or _process(stdout=json.dumps(state.pr))
        if args[:2] == ["pr", "edit"]:
            return _process(1, stderr=READ_ORG_REFUSAL)
        if args[0] == "api" and "PATCH" in args:
            return (state.answer or apply)(json.loads(kwargs["input"]))
        if args[0] == "api" and "/git/ref/heads/" in args[1]:
            return _process(stdout=json.dumps({"object": {"sha": NEW_BASE_SHA}}))
        raise AssertionError(f"unexpected gh command: {argv!r}")

    monkeypatch.setattr(subprocess, "run", run)
    registry = ToolRegistry(repo_dir=ctx.repo_dir, drive_root=ctx.drive_root)
    registry.set_context(ctx)
    state.apply = apply
    state.edit = lambda **args: registry.execute_result("github_write", {"method": "update", "kind": "pr", "target": 7, **args})
    state.patches = lambda: [argv for argv, _kw in state.calls if "PATCH" in argv]
    return state


@pytest.mark.parametrize("pr_host,default_host", [("github.example", "github.com"), ("github.com", "enterprise.example")])
@pytest.mark.parametrize("change,sent", [
    ({"title": "New title"}, {"title": "New title"}),  # an omitted body stays omitted
    ({"body": ""}, {"body": ""}),  # an explicit empty body clears it
    ({"title": "Заголовок → 🤖", "body": "Тело\r\n- причина ✓\n"}, {"title": "Заголовок → 🤖", "body": "Тело\r\n- причина ✓\n"}),
    ({"title": "T", "base": "release", "expect": {"base_sha": NEW_BASE_SHA}}, {"title": "T", "base": "release"}),
    ({"title": "T", "base": "main", "expect": {"base_sha": BASE_SHA}}, {"title": "T"}),  # the held base is not resent
])
def test_field_edit_is_one_rest_patch_at_the_read_pull_requests_own_host(cli, monkeypatch, pr_host, default_host, change, sent):
    """`gh pr edit` queries organization fields a repo-scoped token cannot read (#2424; poisoned here). The edit is one
    REST PATCH of exactly the sent fields at the repository and host of the pull request just read, not gh's default."""
    monkeypatch.setenv("GH_HOST", default_host)
    cli.pr["url"] = f"https://{pr_host}/owner/selected/pull/7"
    change = dict(change)
    retarget = sent.get("base") == "release"

    result = cli.edit(expect={"head_sha": SHA, **change.pop("expect", {})}, **change)

    assert (result.status, result.meta.get("operation_outcome")) == ("ok", "completed"), result.text
    assert [argv[1] if argv[1] == "api" else " ".join(argv[1:3]) for argv, _ in cli.calls] == (
        ["pr view", "api", "api", "pr view"] if retarget else ["pr view", "api", "pr view"])
    route = ["gh", "api", "repos/owner/selected/pulls/7", "--method", "PATCH", "--input", "-", "--hostname", pr_host]
    assert cli.patches() == [route]
    assert json.loads(next(kw for argv, kw in cli.calls if argv == route)["input"]) == sent
    if retarget:
        assert cli.calls[1][0] == ["gh", "api", "repos/owner/selected/git/ref/heads/release", "--hostname", pr_host]
    receipt = json.loads(result.text.partition("\n")[2])
    assert receipt["outcome"] == "confirmed" and receipt["detail"].startswith("✅ PR #7 updated")


@pytest.mark.parametrize("case,outcome,fragment", [
    ("forbidden", "completed_no_effect", "⚠️ GH_WRITE_NOT_APPLIED: GitHub refused the edit of PR #7"),
    ("malformed_readback", "unknown", "⚠️ Read-back failed"),
    ("refused_then_malformed_readback", "unknown", "Read-back failed too"),
])
def test_field_edit_answers_settle_by_github_status_and_readback_without_replay(cli, case, outcome, fragment):
    """Only a definite HTTP 4xx with every sent field read back unchanged is not applied; a 5xx, a lost or
    statusless answer, a failed read-back or a change since the precheck stays unknown. Nothing is resent."""
    def timeout(_fields):
        raise subprocess.TimeoutExpired(["gh"], 60)

    def moved(fields):
        cli.pr["headRefOid"] = MOVED
        return cli.apply(fields)

    cli.answer = {
        "forbidden": _refusal(403, "Resource not accessible by personal access token"),
        "invalid": _refusal(422, "Validation Failed"),
        "refused_then_malformed_readback": _refusal(422, "Validation Failed"),
        "invalid_after_a_change": lambda fields: (cli.apply(fields), _refusal(422, "Validation Failed")(fields))[1],
        "server_error": lambda _fields: _process(1, "", "gh: HTTP 502\n"),
        "timeout": timeout,
        "no_status": lambda _fields: _process(1, "", "error connecting to api.github.example\n"),
        "head_moved": moved,
    }.get(case)
    if "malformed" in case:
        cli.views = [None, _process(stdout="<html>maintenance</html>")]

    result = cli.edit(expect={"head_sha": SHA}, title="New title")

    receipt = json.loads(result.text.partition("\n")[2])
    assert result.meta.get("operation_outcome") == outcome, result.text
    assert fragment in receipt["detail"], receipt["detail"]
    assert [argv[1] if argv[1] == "api" else " ".join(argv[1:3]) for argv, _ in cli.calls] == ["pr view", "api", "pr view"]
    assert len(cli.patches()) == 1
    if outcome == "completed_no_effect":
        assert receipt["outcome"] == "not_applied" and receipt["http_status"] in (403, 422)
        assert "One request was sent and not retried" in receipt["detail"]


# --- set_github_pr_ready ----------------------------------------------------------------------------------------------

def test_ready_marks_a_draft_ready_once_and_reads_the_state_back(fake, ctx):
    gh = fake(pr=_open_pr(isDraft=True))
    text = _pr(ctx, "update", number=7, ready=True, expected_head_sha=SHA)
    assert gh.kinds() == ["view", "ready", "view"] and gh.writes()[0].args == ["pr", "ready", "7"]
    assert text == f"✅ PR #7 is now ready for review: {URL}\nRead-back: draft=False confirmed; head at {SHA} confirmed"
    gh = fake(pr=_open_pr(isDraft=False))
    text = _pr(ctx, "update", number=7, ready=False, expected_head_sha=SHA)
    assert gh.writes()[0].args == ["pr", "ready", "7", "--undo"] and "is now a draft" in text


def test_ready_on_a_pull_request_already_in_that_state_sends_nothing(fake, ctx):
    """Paired: the blind ``gh pr ready`` exits 1 on an already-ready pull request; the writer reports the held state."""
    gh = fake(pr=_open_pr(isDraft=False))
    text = _pr(ctx, "update", number=7, ready=True, expected_head_sha=SHA)
    assert gh.writes() == [] and text == f"PR #7 is already ready for review; nothing was changed. {URL}"
    assert (_typed(ctx).status, _typed(ctx).meta["operation_outcome"]) == ("ok", "completed_no_effect")


def test_ready_refuses_a_moved_head_and_reports_a_lost_answer_by_the_readback(fake, ctx):
    gh = fake(pr=_open_pr(headRefOid=MOVED))
    assert "nothing was changed" in _pr(ctx, "update", number=7, ready=True, expected_head_sha=SHA)
    assert gh.writes() == []
    gh = fake(pr=_open_pr(isDraft=True), failing={"ready": TIMEOUT})
    text = _pr(ctx, "update", number=7, ready=True, expected_head_sha=SHA)
    assert len(gh.writes()) == 1
    assert text.endswith("its effect is unknown until GitHub shows it. Read-back: GitHub still reported draft=True at that moment; a "
                         "request it is still processing could still land, so read again before resending.")
    assert _typed(ctx).meta["operation_outcome"] == "unknown"
    gh = fake(pr=_open_pr(isDraft=True), failing={"ready": REFUSED})
    text = _pr(ctx, "update", number=7, ready=True, expected_head_sha=SHA, repo="github.example/owner/selected")
    assert text.startswith("⚠️ PR_READY_UNKNOWN: the request to mark PR #7 in github.example/owner/selected ready for review ended with "
                           "exit 1, HTTP 422: ")
    assert _typed(ctx).meta["operation_outcome"] == "unknown" and len(gh.writes()) == 1


def test_ready_accepted_but_readback_failed_is_unknown_not_a_retry(fake, ctx):
    gh = fake(pr=_open_pr(isDraft=True), view_fails_after_write=True)
    text = _pr(ctx, "update", number=7, ready=True, expected_head_sha=SHA, repo="github.example/owner/selected")
    assert len(gh.writes()) == 1 and text.startswith("✅ gh accepted marking PR #7 ready for review.\n⚠️ Read-back failed")
    assert text.endswith("read it with github_read(method='get', kind='pr', target=7, repo='github.example/owner/selected').")
    assert (_typed(ctx).status, _typed(ctx).meta["operation_outcome"]) == ("ok", "unknown")


def test_ready_accepted_but_readback_still_draft_is_not_called_ready(fake, ctx):
    gh = fake(pr=_open_pr(isDraft=True))

    def stays_draft(args, ctx_, **kwargs):
        res = FakeGitHub.__call__(gh, args, ctx_, **kwargs)
        if gh.kind(args) == "ready":
            gh.pr["isDraft"] = True
        return res
    github_pr_write._gh_run = stays_draft
    text = _pr(ctx, "update", number=7, ready=True, expected_head_sha=SHA)
    assert text.startswith(f"⚠️ gh accepted marking PR #7 ready for review ({URL}), but the read-back does not show that state.")
    assert "⚠️ Read-back differs: draft is True, you asked for False." in text
    assert (_typed(ctx).status, _typed(ctx).meta["operation_outcome"]) == ("ok", "unknown")


def test_create_sends_the_explicit_repo_positionally_to_repo_view_and_as_a_flag_elsewhere(ctx, monkeypatch):
    """Actual argv through the real transport, stopped by a stale head after the reads."""
    monkeypatch.setattr(github, "github_token_from_env_or_settings", lambda: "fixture-token")
    calls = []

    def run(argv, **kwargs):
        calls.append(argv)
        if argv[1:3] == ["repo", "view"]:
            return SimpleNamespace(returncode=0, stdout=json.dumps({"nameWithOwner": "owner/selected", "url": BASE}), stderr="")
        return SimpleNamespace(returncode=0, stdout=json.dumps({"object": {"sha": MOVED}}), stderr="")
    monkeypatch.setattr(subprocess, "run", run)

    text = _pr(ctx, "create_pr", **CREATE, repo="github.example/owner/selected")

    assert calls == [["gh", "repo", "view", "github.example/owner/selected", "--json", "nameWithOwner,url"],
                     ["gh", "api", "repos/owner/selected/git/ref/heads/feature", "--hostname", "github.example"]]
    assert text.startswith(f"⚠️ PR_PRECHECK_FAILED: head owner:feature is at {MOVED}, you expected {SHA}; nothing was created.")


def test_ready_argument_refusals(fake, ctx):
    gh = fake(pr=_open_pr())
    assert "draft is a boolean" in _pr(ctx, "update", number=7, ready="yes", expected_head_sha=SHA)
    assert "expect.head_sha is the full 40-hex" in _pr(ctx, "update", number=7, ready=True, expected_head_sha="")
    assert gh.calls == []


# --- registration and the shared identity -------------------------------------------------------------------------------

def test_writers_register_through_github_get_tools_with_check_policy_and_repo():
    from ouroboros.safety import TOOL_POLICY, POLICY_CHECK

    entries = {entry.name: entry for entry in github.get_tools()}
    assert set(entries) == {"github_read", "github_write", "pr_merge"}
    assert TOOL_POLICY["github_write"] == POLICY_CHECK and not entries["github_write"].mutates_worktree
    props = entries["github_write"].schema["parameters"]["properties"]
    assert "repo" in props and "base_sha" in props["expect"]["properties"]
    assert all("default" not in props[key] for key in ("body", "title", "draft")), "omitted update fields stay omitted"
    assert "create_pr" in props["method"]["enum"]
    assert not hasattr(github_pr_write, "tool_entries")
    assert not any(hasattr(github_pr_write, name) for name in ("create_pr", "update_pr", "set_pr_ready"))


def test_registered_writer_runs_under_the_registry_with_the_repo_binding(tmp_path, monkeypatch):
    from ouroboros.tools.registry import ToolRegistry

    monkeypatch.setattr("ouroboros.safety.check_safety", lambda *_a, **_k: (True, ""))
    monkeypatch.setattr(github, "github_token_from_env_or_settings", lambda: "fixture-token")
    gh = FakeGitHub(pr=_open_pr())
    monkeypatch.setattr(github_pr_write, "_gh_run", gh)
    ctx, _ = _context(tmp_path, "queued")
    registry = ToolRegistry(repo_dir=ctx.repo_dir, drive_root=ctx.drive_root)
    registry.set_context(ctx)
    result = registry.execute_result("github_write", {"method": "update", "kind": "pr", "target": 7,
                                                          "expect": {"head_sha": SHA, "base_sha": BASE_SHA}, "title": "Via registry",
                                                          "base": "main",
                                                          "repo": "github.example/owner/selected"})
    assert (result.status, result.code, result.meta.get("operation_outcome")) == ("ok", "OK", "completed"), result
    assert [call.repo for call in gh.calls] == ["github.example/owner/selected", github._GENERIC_TRANSPORT,
                                                "github.example/owner/selected"]
    assert gh.writes()[0].args == PATCH and json.loads(gh.writes()[0].input) == {"title": "Via registry"}
    refused = registry.execute_result("github_write", {"method": "update", "kind": "pr", "target": 7, "expect": {"head_sha": SHA}, "repo": "github.example/owner/selected"})
    assert (refused.status, refused.code, refused.meta["operation_outcome"]) == ("error", "TOOL_ARG_ERROR", "completed_no_effect")
    assert len(gh.writes()) == 1


@pytest.mark.parametrize("args", [
    {"method": "create_pr", "head": "feature", "base": "main", "title": "T"},
    {"method": "update", "kind": "pr", "target": 7, "title": "T"},
    {"method": "update", "kind": "pr", "target": 7, "draft": False},
])
def test_null_expect_refuses_before_any_request(fake, ctx, args):
    gh = fake(pr=_open_pr())
    entry = next(entry for entry in github.get_tools() if entry.name == "github_write")
    text = entry.handler(ctx, **args, expect=None)
    assert "TOOL_ARG_ERROR" in text and "expect.head_sha" in text and not gh.calls
    assert _typed(ctx).meta["operation_outcome"] == "completed_no_effect"


@pytest.mark.parametrize("observed,expected", [
    ({"head_owner": "Owner", "head_branch": "feature", "base_branch": "main", "head_sha": SHA.upper()}, True),
    ({"head_owner": "owner", "head_branch": "Feature", "base_branch": "main", "head_sha": SHA}, False),
    ({"head_owner": "owner", "head_branch": "feature", "base_branch": "Main", "head_sha": SHA}, False),
    ({"head_owner": "other", "head_branch": "feature", "base_branch": "main", "head_sha": SHA}, False),
    ({"head_owner": "owner", "head_branch": "feature", "base_branch": "main", "head_sha": MOVED}, False),
    ({"head_owner": "owner", "head_branch": "feature", "base_branch": "main", "head_sha": None}, False),
    ({"head_owner": "", "head_branch": "feature", "base_branch": "main", "head_sha": SHA}, False),
])
def test_same_pull_request_rules(observed, expected):
    assert same_pull_request(observed, head_owner="owner", head_branch="feature", base_branch="main", head_sha=SHA) is expected


def test_same_pull_request_without_a_commit_is_head_and_base_only():
    observed = {"head_owner": "owner", "head_branch": "feature", "base_branch": "main", "head_sha": MOVED}
    assert same_pull_request(observed, head_owner="owner", head_branch="feature", base_branch="main", head_sha=None)
    assert not same_pull_request({**observed, "head_sha": None}, head_owner="owner", head_branch="feature", base_branch="main", head_sha=SHA)


def test_identity_fields_project_both_wire_shapes_onto_the_same_keys():
    rest = {"head": {"sha": SHA, "ref": "feature", "repo": {"owner": {"login": "owner"}}}, "base": {"ref": "main"}}
    cli = _open_pr()
    assert pull_request_fields_from_rest(rest) == pull_request_fields_from_cli(cli) == {
        "head_sha": SHA, "head_branch": "feature", "head_owner": "owner", "base_branch": "main"}
    assert pull_request_fields_from_rest({"head": "x", "base": None}) == {"head_sha": None, "head_branch": None, "head_owner": None, "base_branch": None}


@pytest.mark.parametrize("number", [None, -1, 7])
def test_create_missing_url_never_invents_a_pr_number(fake, ctx, number):
    gh = fake(failing={
        "create": github.GhResult(True, "Accepted", 0, None, ""),
        "list": [_answer([]), _answer([_open_pr(number=number)])],
    })
    text = _pr(ctx, "create_pr", **CREATE)
    assert len(gh.writes()) == 1
    assert _typed(ctx).meta.get("operation_outcome", "unknown") == "unknown"
    if number == 7:
        assert "PR #7 created" in text
        assert "view" in gh.kinds()
    else:
        assert "PR_CREATE_UNKNOWN" in text and "valid PR number" in text
        assert "view" not in gh.kinds()
        assert "PR #0" not in text and "PR #-1" not in text


@pytest.mark.parametrize("tool,kwargs,fragment", [
    ("create", CREATE, "PR_PRECHECK_FAILED"),
    ("edit", {"number": 7, "expected_head_sha": SHA, "title": "New title"}, "PR_PRECHECK_FAILED"),
    ("ready", {"number": 7, "expected_head_sha": SHA, "ready": True}, "nothing was changed"),
])
def test_pr_writers_require_fresh_precheck_and_send_once(
    tmp_path, monkeypatch, tool, kwargs, fragment,
):
    from tests.test_github_project_target import MOVED, _context

    ctx, _project = _context(tmp_path, "queued")
    ctx._active_builtin_tool_result = None
    pr = _open_pr(headRefOid=MOVED) if tool != "create" else None
    refs = {"owner:feature": MOVED, "owner:main": BASE_SHA}
    guarded = FakeGitHub(pr=pr, refs=refs)
    monkeypatch.setattr(github_pr_write, "_gh_run", guarded)
    def handler(ctx, **kwargs):
        return _pr(ctx, "create_pr" if tool == "create" else "update", **kwargs)
    text = handler(ctx, **kwargs)
    assert fragment in text and guarded.writes() == []

    fresh_pr = None if tool == "create" else _open_pr(isDraft=True)
    fresh_refs = {"owner:feature": SHA, "owner:main": BASE_SHA}
    fresh = FakeGitHub(pr=fresh_pr, refs=fresh_refs)
    monkeypatch.setattr(github_pr_write, "_gh_run", fresh)
    assert not handler(ctx, **kwargs).startswith("⚠️")
    assert len(fresh.writes()) == 1
    entry = next(x for x in github.get_tools() if x.name == "github_write")
    assert "head_sha" in entry.schema["parameters"]["properties"]["expect"]["properties"]


def test_pr_update_reports_partial_state_after_one_send(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from tests.test_github_project_target import _context

    ctx, _project = _context(tmp_path, "queued")
    ctx._active_builtin_tool_result = None
    def partial_driver():
        driver = FakeGitHub(pr=_open_pr())

        def call(args, context, **options):
            if driver.kind(args) == "edit":
                driver.calls.append(SimpleNamespace(args=list(args), repo=options.get("repo"),
                                                    input=options.get("input_data"), timeout=60))
                driver.pr["title"] = "New title"
                return SERVER_ERROR
            return driver(args, context, **options)
        return driver, call

    driver, call = partial_driver()
    monkeypatch.setattr(github_pr_write, "_gh_run", call)
    text = _pr(ctx, "update", number=7, expected_head_sha=SHA, title="New title", body="New body")
    assert len(driver.writes()) == 1 and "held: title confirmed" in text
    assert "not held: body differs" in text and "before resending only the fields not held" in text
    assert (driver.pr["title"], driver.pr["body"]) == ("New title", "Old body")
