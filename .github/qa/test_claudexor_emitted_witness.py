"""Hosted Linux replay of the public Claudexor #697 synthetic fixtures, one pytest process per cell."""

import asyncio
import hashlib
import inspect
import json
import os
import re
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from collections import Counter
from dataclasses import replace
from functools import wraps
from pathlib import Path
from urllib.parse import parse_qsl

SOURCE_REPO = "razzant/claudexor"
SOURCE_RUN = 44417855654
SOURCE_ARTIFACT = 14481254555
ZIP_SIZE = 158721
ZIP_SHA256 = "0ec4e08a12c40893365b4098f13370473ed2f1d838c1c0b898e5d9246f13b241"
BUNDLE_SHA256 = "37f4b0f98b7690b217f87ef8636eb5ffee6a4287271989dc1bae8d9168432c32"  # whole-file digest of the audited carrier, set after export; a placeholder fails closed
BASE = "697/"
CATALOG = BASE + "failed/operations.json"
MODEL = "claudexor::codex=test-model"
ALTERNATE = "claudexor::codex=offline-alternate"
CREATE = "/v2/model-operations"
PICNIC = "No picnic today. 🧺"
NOTICE = "Codex declined this request under its content policy."
MIXED = "Menu\n" + PICNIC + "\nFin."
WORDS = {"mixed": MIXED, "legacy-mixed": MIXED, "refusal-only": PICNIC,
         "ordinary-text": "The label reads cyber_policy."}
REFUSED = ("failed", "failed-no-capture", "http-refusal", "legacy-failure")
TYPED_WORDS = ("mixed", "refusal-only")
CASES = (*REFUSED, *WORDS, "ordinary-error")
# Main's ordinary error recovery may legitimately send more. It is not a no-resend oracle.
CELLS = [(case, lane) for case in CASES for lane in ("sync", "async", "loop")
         if (case, lane) != ("ordinary-error", "loop")]
COUNT_COLUMNS = ("client_calls", "uploads", "creates", "result_reads", "acks", "gateway_opens",
                 "account_rotation_calls", "account_mark_calls", "cooldown_calls", "fallback_calls",
                 "resource_deferrals", "capability_retries", "substitution_redos")
HEADER = ("case", "lane", "status", "rc", "observed", "detail", *COUNT_COLUMNS)


def read_corpus(path):
    """Verify the one committed carrier byte for byte, then project the same 28 public fixture files from it."""
    raw = path.read_bytes()
    bundle = hashlib.sha256(raw).hexdigest()
    assert re.fullmatch(r"[0-9a-f]{64}", BUNDLE_SHA256), "BUNDLE_SHA256 is still a placeholder"
    assert bundle == BUNDLE_SHA256, "carrier SHA-256 differs"
    names = {f"{BASE}{case}/{kind}.json" for case in CASES for kind in ("detail", "result", "transport")}
    names.add(CATALOG)
    document = json.loads(raw.decode("utf-8"))
    assert isinstance(document, dict) and set(document) == {"source", "files"}, "carrier shape differs"
    source = {"repository": SOURCE_REPO, "run_id": SOURCE_RUN, "artifact_id": SOURCE_ARTIFACT,
              "zip_bytes": ZIP_SIZE, "zip_sha256": ZIP_SHA256}
    assert document["source"] == source, "carrier lacks the original verified source identity"
    declared = document["files"]
    assert isinstance(declared, dict) and set(declared) == names, "carrier fixture inventory differs"
    files, receipts = {}, {}
    for name in sorted(names):
        entry = declared[name]
        assert isinstance(entry, dict) and set(entry) == {"size", "sha256", "text"}, f"carrier entry shape differs: {name}"
        assert type(entry["size"]) is int and isinstance(entry["text"], str), f"carrier entry types differ: {name}"
        data = entry["text"].encode("utf-8")
        identity = {"size": len(data), "sha256": hashlib.sha256(data).hexdigest()}
        assert all(entry[k] == v for k, v in identity.items()), f"fixture identity differs: {name}"
        files[name], receipts[name] = data, identity
    return files, {"carrier": "verified_public_fixture_projection", "fresh_zip_download": False,
                   "bundle_sha256": bundle, "source": source, "files": receipts}


def expected_reply(case):
    return WORDS.get(case, "provider_content_refused" if case in REFUSED else "provider_failed")


def cell_result(run, report, case, lane):
    prefix = f"CX697_CHECKED {case} {lane} ".encode("ascii")
    words = [line[len(prefix):] for line in run.stdout.splitlines() if line.startswith(prefix)]
    expected = json.dumps(expected_reply(case)).encode("ascii")
    checked = len(words) == 1 and words[0] == expected
    counts = re.findall(rb"^CX697_COUNTS((?: [0-9]+){%d})$" % len(COUNT_COLUMNS), run.stdout, re.MULTILINE)
    values = tuple(int(x) for x in counts[0].split()) if len(counts) == 1 else ()
    observed = json.dumps(json.loads(words[0]), ensure_ascii=False) if checked else "-"
    detail = "ok"
    try:
        cases = list(ET.parse(report).iter("testcase"))
        exact_node = len(cases) == 1 and cases[0].get("name") == f"test_emitted[{case}-{lane}]"
        if not exact_node:
            detail = "pytest_node_unproven"
        elif any(cases[0].find(tag) is not None for tag in ("error", "failure", "skipped")):
            detail = "pytest_not_passed"
    except (OSError, ET.ParseError):
        detail = "pytest_report_unreadable"
    if detail == "ok":
        if run.returncode != 0:
            detail = "pytest_nonzero"
        elif not checked:
            detail = "checked_reply_unproven"
        elif values[:6] != (1,) * 6 or (case != "ordinary-error" and any(values[6:])):
            detail = "counts_unproven"
    return ("PASS" if detail == "ok" else "FAIL"), run.returncode, observed, detail, (
        values or ("-",) * len(COUNT_COLUMNS))


def run_matrix(carrier, report, provenance):
    """Write only known labels, observed return/counters, and per-cell process receipts."""
    module = Path(__file__).resolve()
    repo = module.parents[1]
    env = {**os.environ, "CX697_CARRIER": str(carrier)}
    missing = ("-",) * len(COUNT_COLUMNS)
    with report.open("x", encoding="utf-8", buffering=1) as out:
        print(*HEADER, sep="\t", file=out)
        try:
            _, identity = read_corpus(carrier)
        except (OSError, ValueError, KeyError, TypeError, AssertionError) as error:
            # Name only the failed check: never carrier content, JSON parser context, paths or subprocess logs.
            reason = str(error) if isinstance(error, AssertionError) else type(error).__name__
            print(f"CX697 carrier verification failed ({reason})", file=sys.stderr)
            for case, lane in CELLS:
                print(case, lane, "NOT_RUN", "-", "-", "corpus_unverified", *missing, sep="\t", file=out)
            return 1
        with provenance.open("x", encoding="utf-8") as handle:
            json.dump(identity, handle, indent=2, sort_keys=True)
            handle.write("\n")
        passed, blocked = True, False
        with tempfile.TemporaryDirectory(prefix="cx697-") as scratch:
            for index, (case, lane) in enumerate(CELLS):
                if blocked:
                    print(case, lane, "NOT_RUN", "-", "-", "prior_cell_timed_out", *missing, sep="\t", file=out)
                    continue
                node = f"{module.relative_to(repo)}::test_emitted[{case}-{lane}]"
                xml = Path(scratch) / f"{index}.xml"
                argv = [sys.executable, "-m", "pytest", "-o", "addopts=", "-p", "no:cacheprovider",
                        "-q", "-s", "-rN", "--tb=no", "--assert=plain", "--disable-warnings", f"--junitxml={xml}", node]
                try:
                    run = subprocess.run(argv, env=env, cwd=repo, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                         timeout=65)
                except subprocess.TimeoutExpired:
                    status, rc, observed, detail, values = "TIMEOUT", "-", "-", "pytest_deadline", missing
                    blocked = True  # let the outer safe_test owner settle any descendant before further work
                else:
                    status, rc, observed, detail, values = cell_result(run, xml, case, lane)
                print(case, lane, status, rc, observed, detail, *values, sep="\t", file=out)
                passed &= status == "PASS"
    return 0 if passed else 1


if sys.platform != "linux":
    raise SystemExit("This witness is for the isolated hosted Linux runner.")
if not __debug__:
    raise RuntimeError("The witness requires Python assertions.")
if __name__ == "__main__":
    if len(sys.argv) != 4 or Path.cwd().resolve() != Path(__file__).resolve().parents[1]:
        raise SystemExit("from the witness checkout: test_claudexor_emitted_witness.py CARRIER_JSON NEW_CELLS_TSV NEW_PROVENANCE_JSON")
    raise SystemExit(run_matrix(*(Path(arg).resolve() for arg in sys.argv[1:])))

import httpx
import pytest

from ouroboros import fallback_cooldown, llm_claudexor as cx, llm_substitution, loop
from ouroboros._usage_response import assistant_answer_text, refusal_display
from ouroboros.loop_llm_call import classify_llm_exception
from ouroboros.loop_messages import _visible_round_text
from ouroboros.model_slots import get_fallback_models
from ouroboros.model_wait import ResourceDeferral, model_wait_reason, task_model_wait_scope
from ouroboros.task_results import write_task_result
from tests._context_fit_plan import fit_plan
from tests._scripted_model_loop import loop_kwargs
from tests._shared import saved_settings
from tests.launcher_network_guard import deny_real_network
from tests.test_claudexor_model_transport import ModelWire, _ref, gateway_factory as gateway_factory
from tests.test_llm_claudexor import ledger, retained, setup as setup


@pytest.fixture(scope="module")
def corpus():
    return read_corpus(Path(os.environ["CX697_CARRIER"]))[0]


def query_flags(raw):
    return sorted(parse_qsl(raw, keep_blank_values=True, strict_parsing=True))


class ReplayWire(ModelWire):
    """Reuse the real upload exchange; replay the producer's terminal detail and result byte-for-byte."""

    def __init__(self, files, case, lane, root):
        super().__init__()
        self.root, self.ack_verified = root, False
        prefix = f"{BASE}{case}/"
        self.record = json.loads(files[prefix + "transport.json"])
        self.detail_bytes = files[prefix + "detail.json"]
        self.result = files[prefix + "result.json"]
        detail = json.loads(self.detail_bytes)
        self.op = CREATE + "/" + self.record["id"]
        self.ref = detail["response"]["ref"]
        assert detail["id"] == self.record["id"] and self.ref == _ref(self.result, self.ref["resourceId"])
        assert self.record["etag"] == '"' + self.ref["sha256"] + '"'  # strong recorded ETag, not a substitute for the ref
        assert self.record["contentLength"] == len(self.result)
        self.catalog = files[CATALOG]
        switches = {"typedRefusals": not case.startswith("legacy-"),
                    "captureFailureEvidence": case in ("failed", "http-refusal")}
        assert all(self.record[name] is enabled for name, enabled in switches.items())
        recorded = query_flags(self.record["query"] or "")
        assert recorded == sorted((name, "true") for name, enabled in switches.items() if enabled)
        # Sync deliberately tests the full advertised capability catalog (including effort/failure evidence);
        # async and Main restrict the advertised switch set to match each recorder case. Neither rewrites a receipt.
        if lane in ("async", "loop"):
            view = json.loads(self.catalog)
            descriptor, = (row for row in view["operations"] if (row["method"], row["path"]) == ("POST", CREATE))
            hidden = {"captureEffortEvidence", *switches} - {name for name, _ in recorded}
            descriptor["parameters"] = [p for p in descriptor["parameters"] if p["name"] not in hidden]
            self.catalog = json.dumps(view).encode()
            self.query = recorded
        else:
            self.query = [("captureEffortEvidence", "true"), ("captureFailureEvidence", "true")]
            if switches["typedRefusals"]:
                self.query.append(("typedRefusals", "true"))

    def respond(self, request):
        verb, path = request.method, request.url.path
        if path.startswith("/v2/uploads"):
            if verb == "POST" and path == "/v2/uploads" and self.upload_count:
                pytest.fail("unexpected second upload", pytrace=False)
            return super().respond(request)
        if (verb, path) == ("GET", "/v2/operations"):
            return httpx.Response(200, content=self.catalog)
        if (verb, path) == ("POST", CREATE):
            self.generation_count += 1  # one mock acceptance, not a vendor generation measurement
            assert self.generation_count == 1, "unexpected second create"
            assert request.headers["Idempotency-Key"] and json.loads(request.content) == {"request": _ref(self.payload, "res-request")}
            assert query_flags(request.url.query.decode("ascii")) == self.query
            return httpx.Response(202, json={"id": self.record["id"], "state": "queued"})
        if (verb, path) == ("GET", self.op):
            return httpx.Response(200, content=self.detail_bytes)
        if (verb, path) == ("GET", self.op + "/result"):
            return httpx.Response(200, content=self.result, headers={
                "Content-Type": "application/json", "ETag": self.record["etag"],
                "Content-Length": str(self.record["contentLength"])})
        if (verb, path) == ("POST", self.op + "/ack"):
            assert json.loads(request.content) == {"sha256": self.ref["sha256"]}
            self.ack_verified = retained(self.root, raw=True) == self.result
            assert self.ack_verified, "ACK before exact result retention"
            return httpx.Response(200, json={"id": self.record["id"], "response": {"state": "acknowledged", "ref": self.ref}})
        pytest.fail("unexpected transport request", pytrace=False)


def observe(monkeypatch, owner, name, calls):
    original = getattr(owner, name)

    async def counted_async(*args, **kwargs):
        calls[name] += 1
        return await original(*args, **kwargs)

    def counted_sync(*args, **kwargs):
        calls[name] += 1
        return original(*args, **kwargs)

    wrapper = counted_async if inspect.iscoroutinefunction(original) else counted_sync
    monkeypatch.setattr(owner, name, wraps(original)(wrapper))


def client_call(client, case, lane, messages):
    options = {"model_role": "main", "cache_affinity": "picnic"}
    if not case.startswith("legacy-"):  # Exercise the default as well as opt-in.
        options["accepts_typed_refusals"] = True
    if lane == "async":
        return asyncio.run(client.chat_async(messages, MODEL, **options))
    return client.chat(messages, MODEL, **options)


def check_client(client, case, lane, messages, wire):
    if case in WORDS:
        reply, usage = client_call(client, case, lane, messages)
        assert reply == json.loads(wire.result)["message"]  # also retains any actual co-located tool envelope
        content = reply["content"]
        observed = _visible_round_text(content)
        assert observed == WORDS[case]
        if case in TYPED_WORDS:
            assert refusal_display(content) == WORDS[case] and assistant_answer_text(content) == ""
            assert [part["type"] for part in content] == (
                ["text", "refusal", "refusal", "text"] if case == "mixed" else ["refusal"])
        else:
            assert isinstance(content, str) and content == WORDS[case]
    else:
        with pytest.raises(cx.ClaudexorModelError) as caught:
            client_call(client, case, lane, messages)
        error = caught.value
        observed = error.code
        assert error.physical_attempt_capture.state == "settled"
        if case in REFUSED:
            assert type(error) is cx.ClaudexorContentRefused and error.code == "provider_content_refused"
            assert error.problem["message"] == NOTICE and not error.retryable
            assert not model_wait_reason(error) and not classify_llm_exception(error).retry_same_request
        else:
            assert type(error) is cx.ClaudexorModelError and error.code == "provider_failed"
        usage = error.usage
    assert usage["claudexor"]["result_custody"]["state"] == "acknowledged"
    return observed


def check_loop(root, client, waiter, case, messages, prior_route):
    kwargs = loop_kwargs(root, client, [])
    ctx = kwargs["tools"]._ctx
    plan = replace(fit_plan(preferred="max", window=1_000_000, known=True),
                   model=MODEL, provider="claudexor", model_role="main", model_route=prior_route)
    ctx.context_fit_plan = plan
    waiter.tool_context = ctx
    kwargs.update(messages=[*plan.messages_for("max"), *messages], drive_logs=root / "logs", task_id="task-one")
    text, usage, trace = loop.run_llm_loop(**kwargs)
    assert ctx.active_model == MODEL
    if case in REFUSED:
        assert text == NOTICE
        assert usage["reason_code"] == "provider_content_refused" and usage["execution_status"] == "failed"
        assert trace.get("refusal_reply_final_answer") == ""
        return usage["reason_code"]
    assert text == WORDS[case] and usage["terminal_origin"] == "model_final"
    assert usage.get("execution_status") not in ("failed", "infra_failed")
    return text


@pytest.mark.parametrize(("case", "lane"), [pytest.param(c, l, id=f"{c}-{l}") for c, l in CELLS])
def test_emitted(case, lane, corpus, setup, gateway_factory, monkeypatch, provider_catalog_offline):
    root, _, client = setup
    assert Path(cx.__file__).resolve().parents[1] == Path(__file__).resolve().parents[1]
    wire = ReplayWire(corpus, case, lane, root)
    calls = Counter()

    def open_gateway(**_kwargs):
        calls["gateway_opens"] += 1
        assert calls["gateway_opens"] == 1, "unexpected second gateway open"
        return gateway_factory(wire)

    monkeypatch.setattr(cx, "ensure_owned_gateway", open_gateway)
    monkeypatch.setattr(cx, "read_owned_gateway", lambda: pytest.fail("unexpected real control-gateway read", pytrace=False))
    saved_settings(monkeypatch, root, OUROBOROS_MODEL=MODEL, OUROBOROS_MODEL_FALLBACKS=ALTERNATE,
                   USE_LOCAL_MAIN="false", USE_LOCAL_FALLBACK="false")
    monkeypatch.setenv("OUROBOROS_CONTEXT_MODE", "max")
    monkeypatch.setenv("OUROBOROS_TASK_REVIEW_MODE", "off")
    assert get_fallback_models(MODEL) == [ALTERNATE]
    watched = ((cx.AccountRotation, "refused_call"), (llm_substitution, "remember_failed_profile"),
               (fallback_cooldown, "mark_cooldown"), (loop, "_run_cross_model_fallback_chain"),
               (ResourceDeferral, "retain"), (cx.CapabilityRetry, "refused"), (cx.SubstitutionBudget, "admit"))
    for owner, name in ((client, "chat"), (client, "chat_async"), *watched):
        observe(monkeypatch, owner, name, calls)
    prior = json.loads(corpus[BASE + "ordinary-text/result.json"])["message"]
    prior_route = prior["nativeContinuation"]["route"]
    messages = [{"role": "user", "content": "Read the label."}, prior, {"role": "user", "content": "No picnic today."}]
    write_task_result(root, "task-one", "running")
    try:
        with deny_real_network() as sockets, task_model_wait_scope(
            task={"id": "task-one"}, drive_root=root, event_queue=None, worker_slot_held=False
        ) as waiter:
            with waiter.defer_resource_wait("main") as deferred:
                if lane == "loop":
                    observed = check_loop(root, client, waiter, case, messages, prior_route)
                else:
                    observed = check_client(client, case, lane, messages, wire)
            if case != "ordinary-error":
                assert not deferred.fact and not waiter.waits
        assert not sockets
        row, = ledger(root)
        assert row["state"] == "settled"
        submitted = json.loads(wire.payload)
        assert submitted["source"] == "codex" and submitted["model"] == "test-model"
        assert submitted["account"] == {"mode": "auto", "preferredProfileId": prior_route["credentialProfileId"]}
        assert wire.ack_verified and wire.generation_count == wire.upload_count == 1
        if case != "ordinary-error":
            assert all(calls[name] == 0 for _, name in watched)
        assert (calls["chat"], calls["chat_async"]) == ((0, 1) if lane == "async" else (1, 0))
        expected = [("GET", "/v2/operations"), ("POST", "/v2/uploads"), ("GET", "/v2/uploads/upl-one"),
                    ("PUT", "/v2/uploads/upl-one/bytes"), ("POST", "/v2/uploads/upl-one/finalize"),
                    ("POST", CREATE), ("GET", wire.op), ("GET", wire.op + "/result"), ("POST", wire.op + "/ack")]
        assert [(req.method, req.url.path) for req in wire.calls] == expected
        assert wire.calls[5].headers["Idempotency-Key"] == row["attempt_id"]
        print(f"\nCX697_CHECKED {case} {lane} {json.dumps(observed)}", flush=True)
    finally:
        http = Counter((req.method, req.url.path) for req in wire.calls)
        print("\nCX697_COUNTS", calls["chat"] + calls["chat_async"], http["POST", "/v2/uploads"],
              http["POST", CREATE], http["GET", wire.op + "/result"], http["POST", wire.op + "/ack"],
              calls["gateway_opens"], *(calls[name] for _, name in watched), flush=True)
