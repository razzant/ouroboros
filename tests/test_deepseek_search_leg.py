"""The DeepSeek Search leg: what the resolver offers, what the leg sends, what the ledger then says.

Three facts that must not blur: Auto's leg order (the free ddgs first, this paid leg last, neither
carrying a model), the leg's own HTTP contract, and one priced row per DELIVERED request. Rows are read
back through the public audit reader and the money projection, never through a private helper.
"""
import json
from types import SimpleNamespace

import pytest

from ouroboros import usage_accounting
from ouroboros.external_charge import record_external_charge
from ouroboros.llm_routing import DEEPSEEK_BASE_URL
from ouroboros.search_routes import resolve_web_search_route
from ouroboros.tools import search

PRICE = search.DEEPSEEK_SEARCH_USD_PER_REQUEST
ANSWER_BODY = json.dumps({"results": [
    {"title": "T", "url": "https://e.x/a", "summary": "S", "authority": 3, "rerank_score": 2.4},
]}).encode("utf-8")


def _ctx(root):
    return SimpleNamespace(drive_root=root, task_metadata={}, pending_events=[],
                           emit_progress_fn=lambda _text: None)


class _Response:
    def __init__(self, body: bytes):
        self._body = body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def read(self):
        return self._body


def _serve(monkeypatch, body: bytes = ANSWER_BODY, captured: dict | None = None):
    def fake(request, timeout=None):
        if captured is not None:
            captured["url"] = request.full_url
            captured["body"] = json.loads(request.data.decode("utf-8"))
        return _Response(body)
    monkeypatch.setattr("urllib.request.urlopen", fake)


def _external_rows(root):
    return [row for row in usage_accounting.read_usage_records(root)
            if row.get("kind") == "external_unmetered"]


def _route(sources):
    """A resolved route without reading this install's settings: the dispatch branch is what is under test."""
    from ouroboros.search_routes import SEARCH_BACKENDS

    return {"scope": "builtin_web_search", "source": "auto", "model": "", "note": "", "unavailable": [],
            "legs": [{"source": source, "backend": SEARCH_BACKENDS[source], "model": "",
                      "model_source": "provider_default"} for source in sources]}


def test_auto_offers_the_free_leg_first_and_this_paid_one_last_without_a_model():
    route = resolve_web_search_route(settings={"OUROBOROS_WEBSEARCH_BACKEND": "auto",
                                               "DEEPSEEK_API_KEY": "synthetic"})
    sources = [leg["source"] for leg in route["legs"]]
    assert sources[-1] == "deepseek", sources  # the paid leg never outranks a free one
    deepseek = route["legs"][-1]
    assert deepseek["backend"] == "deepseek_search_tool"
    assert deepseek["model"] == "" and deepseek["model_source"] == "provider_default"


def test_a_saved_model_of_another_route_is_never_applied_to_this_leg():
    route = resolve_web_search_route(settings={"OUROBOROS_WEBSEARCH_BACKEND": "auto",
                                               "OUROBOROS_WEBSEARCH_MODEL": "gpt-5.5",
                                               "OPENAI_API_KEY": "synthetic",
                                               "DEEPSEEK_API_KEY": "synthetic"})
    deepseek = next(leg for leg in route["legs"] if leg["source"] == "deepseek")
    assert deepseek["model"] == "" and deepseek["model_source"] == "provider_default"


def test_a_strict_pin_ignores_a_saved_model_and_refuses_a_named_one():
    settings = {"OUROBOROS_WEBSEARCH_BACKEND": "deepseek",
                "OUROBOROS_WEBSEARCH_MODEL": "gpt-5.5",
                "DEEPSEEK_API_KEY": "synthetic"}
    route = resolve_web_search_route(settings=settings)
    assert [leg["source"] for leg in route["legs"]] == ["deepseek"]
    # A per-call model is a real choice: it is refused, never silently dropped for a model-less leg.
    named = resolve_web_search_route(model="gpt-5.5", per_call=True, settings=settings)
    assert "takes no model" in named["error"], named
    assert named["legs"] == []


def test_the_leg_is_named_unavailable_without_its_key():
    route = resolve_web_search_route(backend="deepseek", settings={})
    assert [leg["source"] for leg in route["unavailable"]] == ["deepseek"]
    assert route["unavailable"][0]["unavailable_reason"] == "no_credentials"


def test_dispatch_falls_from_the_free_leg_to_this_one_and_names_both(monkeypatch, tmp_path):
    monkeypatch.setattr(search, "resolve_web_search_route", lambda **_kwargs: _route(["ddgs", "deepseek"]))
    import sys
    import types

    monkeypatch.setitem(sys.modules, "ddgs", types.ModuleType("ddgs"))

    def ddgs_down(_query):
        raise RuntimeError("ddgs is down: 429")

    monkeypatch.setattr(search, "_web_search_ddgs", ddgs_down)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "synthetic")
    _serve(monkeypatch)
    result = json.loads(search._web_search(_ctx(tmp_path), "quiet query"))
    assert result["backend"] == "deepseek_search_tool"
    assert [(leg["source"], leg["outcome"]) for leg in result["legs_tried"]] == [
        ("ddgs", "failed"), ("deepseek", "succeeded")], result["legs_tried"]


def test_a_delivered_search_books_one_priced_estimate(monkeypatch, tmp_path):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "synthetic")
    captured: dict = {}
    _serve(monkeypatch, captured=captured)
    out = json.loads(search._web_search_deepseek(_ctx(tmp_path), "how do the gates work"))
    assert captured["url"] == f"{DEEPSEEK_BASE_URL}/tools/search"
    assert captured["body"] == {"query": "how do the gates work", "count": 5, "need_content": False}
    assert out["backend"] == "deepseek" and out["sources"][0]["url"] == "https://e.x/a"
    assert out["sources"][0]["authority"] == 3
    assert out["cost"] == {"usd": PRICE, "basis": "per_successful_request",
                           "final": False, "recorded": True}
    (row,) = _external_rows(tmp_path)
    assert row["state"] == "settled"
    assert row["cost_usd"] == pytest.approx(PRICE)
    assert row["cost_estimated"] is True and row["cost_final"] is False
    assert row["category"] == "web_search" and row["source"] == "web_search.deepseek"
    view = usage_accounting.usage_breakdown(tmp_path)
    assert view["estimated_usd"] == pytest.approx(PRICE)
    assert view["confirmed_usd"] == 0.0
    # Priced is not "nobody could meter it": the two facts stay distinguishable.
    assert view["unknown_unmetered"] == 0


def test_a_billed_but_unreadable_body_still_books_the_price(monkeypatch, tmp_path):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "synthetic")
    _serve(monkeypatch, body=b"not json at all")
    with pytest.raises(RuntimeError, match="unreadable body"):
        search._web_search_deepseek(_ctx(tmp_path), "a billed query")
    (row,) = _external_rows(tmp_path)
    assert row["cost_usd"] == pytest.approx(PRICE)


def test_an_http_refusal_books_nothing_and_names_the_status(monkeypatch, tmp_path):
    import urllib.error

    monkeypatch.setattr(search, "_search_key", lambda *args, **kwargs: "synthetic")

    def refuse(request, timeout=None):
        raise urllib.error.HTTPError(request.full_url, 402, "payment required", {}, None)

    monkeypatch.setattr("urllib.request.urlopen", refuse)
    with pytest.raises(RuntimeError, match="HTTP 402"):
        search._web_search_deepseek(_ctx(tmp_path), "q")
    assert _external_rows(tmp_path) == []


def test_a_missing_key_refuses_before_any_send(monkeypatch, tmp_path):
    monkeypatch.setattr(search, "_search_key", lambda *args, **kwargs: "")
    sent = []
    monkeypatch.setattr("urllib.request.urlopen", lambda *args, **kwargs: sent.append(args))
    with pytest.raises(RuntimeError, match="DEEPSEEK_API_KEY"):
        search._web_search_deepseek(_ctx(tmp_path), "q")
    assert sent == []


def test_a_ledger_fault_is_disclosed_and_never_loses_the_answer(monkeypatch, tmp_path):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "synthetic")

    def money_store_down(*args, **kwargs):
        raise usage_accounting.UsageAccountingError("money store unavailable")

    monkeypatch.setattr(search, "record_external_charge", money_store_down)
    _serve(monkeypatch)
    out = json.loads(search._web_search_deepseek(_ctx(tmp_path), "q"))
    assert out["sources"] and out["answer"]
    assert out["cost"] == {"usd": PRICE, "basis": "per_successful_request", "final": False,
                           "recorded": False, "error": "UsageAccountingError"}


def test_the_unmetered_seam_still_records_no_price(tmp_path):
    """The other seam is untouched: "nobody could meter it" keeps its unknown amount."""
    usage_accounting.record_unmetered_external_dispatch(
        "skill-call-1", drive_root=tmp_path, provider="skill", task_id="t", root_task_id="t")
    (row,) = _external_rows(tmp_path)
    assert row["cost_usd"] is None
    assert "cost_estimated" not in row
    assert usage_accounting.usage_breakdown(tmp_path)["unknown_unmetered"] == 1


def test_each_delivered_request_is_one_priced_row_and_a_replay_is_free(tmp_path):
    """A per-request vendor bills per delivery; a replayed id must not bill twice."""
    def book(charge_id):
        return record_external_charge(charge_id, cost_usd=PRICE, drive_root=tmp_path, provider="deepseek",
                                      model="deepseek-search", task_id="t", root_task_id="t",
                                      category="web_search", source="web_search.deepseek")

    book("deepseek-search:one")
    book("deepseek-search:one")
    book("deepseek-search:two")
    rows = _external_rows(tmp_path)
    assert len(rows) == 2, rows
    assert all(row["cost_usd"] == pytest.approx(PRICE) and row["cost_final"] is False for row in rows)
    assert usage_accounting.usage_breakdown(tmp_path)["estimated_usd"] == pytest.approx(2 * PRICE)


@pytest.mark.parametrize("bad", [-0.1, float("nan"), float("inf")])
def test_an_external_charge_refuses_a_price_that_is_not_a_real_amount(tmp_path, bad):
    with pytest.raises(usage_accounting.UsageAccountingError, match="non-finite or negative"):
        record_external_charge("bad", cost_usd=bad, drive_root=tmp_path)
    assert _external_rows(tmp_path) == []
