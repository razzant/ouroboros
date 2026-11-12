"""The ``$shared`` reference: one rule, exact splice for every encoder that stores by reference."""
from __future__ import annotations

import hashlib
import json
import math
import pathlib
import random

import pytest

from ouroboros import shared_values as sv
from ouroboros.observability import _json_bytes
from ouroboros.request_wire_contract import physical_candidate_bytes
from ouroboros.utils import machine_json


def _checkpoint_bytes(value):
    """The rolling working checkpoint's encoder (insertion order, opaque handles named)."""
    return json.dumps(value, ensure_ascii=False, default=lambda v: {"__unserializable__": type(v).__name__}).encode("utf-8")


def _machine_bytes(value):
    return machine_json(value).encode("utf-8")


ENCODERS = {
    "observability": (_json_bytes, True),  # (encoder, may carry NaN and non-JSON values)
    "checkpoint": (_checkpoint_bytes, True),
    "machine": (_machine_bytes, True),
    "physical_candidate": (physical_candidate_bytes, False),
}

STRINGS = [
    "", "plain", "Ω unicode", "日本語", "astral 😀 \U0001F9E0", 'quote " and \\ backslash', "line\nbreak\ttab",
    "control \x01\x1f", '{"$shared": "' + "a" * 64 + '"}', "$shared", "ends with brace {", '{"',
    "  separator", "mixed 😀 proper pair",
]


def _scalar(rng: random.Random, *, nan_ok: bool, foreign_ok: bool):
    choice = rng.randrange(12)
    if choice < 4:
        return rng.choice(STRINGS)
    if choice == 4:
        return rng.choice([0, 1, -1, 2 ** 70, -(2 ** 63), 10 ** 30])
    if choice == 5:
        floats = [0.0, -0.0, 1.5, 1e100, -2.5e-7, 3.141592653589793]
        if nan_ok:
            floats += [math.nan, math.inf, -math.inf]
        return rng.choice(floats)
    if choice == 6:
        return rng.choice([True, False, None])
    if choice == 7 and foreign_ok:
        return rng.choice([pathlib.Path("fixture.txt"), pathlib.PurePosixPath("/a/b")])
    return rng.choice(STRINGS)


def _value(rng: random.Random, depth: int, *, nan_ok: bool, foreign_ok: bool):
    if depth <= 0 or rng.random() < 0.35:
        return _scalar(rng, nan_ok=nan_ok, foreign_ok=foreign_ok)
    kind = rng.randrange(3)
    if kind == 0:
        return {f"k{rng.randrange(6)}" if rng.random() < 0.8 else rng.choice(["$shared", "zeta", "Alpha", "日"]):
                _value(rng, depth - 1, nan_ok=nan_ok, foreign_ok=foreign_ok) for _ in range(rng.randrange(5))}
    if kind == 1:
        return [_value(rng, depth - 1, nan_ok=nan_ok, foreign_ok=foreign_ok) for _ in range(rng.randrange(5))]
    return tuple(_value(rng, depth - 1, nan_ok=nan_ok, foreign_ok=foreign_ok) for _ in range(rng.randrange(3)))


def _document(seed: int, *, nan_ok: bool, foreign_ok: bool) -> dict:
    rng = random.Random(seed)
    document = _value(rng, 4, nan_ok=nan_ok, foreign_ok=foreign_ok)
    if not isinstance(document, dict):
        document = {"root": document}
    # Every document carries the shape the rule exists for: a transcript and a tool list.
    document["messages"] = [{"role": "user", "content": _value(rng, 2, nan_ok=nan_ok, foreign_ok=foreign_ok)}
                            for _ in range(rng.randrange(1, 4))]
    document["messages"].append(dict(document["messages"][0]))  # a repeated element
    document["tools"] = [{"type": "function", "function": {"name": f"t{i}"}} for i in range(rng.randrange(3))]
    return document


def _canonical(value) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


def _rule_count(value) -> int:
    """The rule, written independently: dict elements of lists reached from the root through dicts only."""
    if not isinstance(value, dict):
        return 0
    return sum(_rule_count(item) if isinstance(item, dict)
               else sum(1 for element in item if isinstance(element, dict)) if isinstance(item, list) else 0
               for item in value.values())


def test_splice_reproduces_the_encoder_bytes_for_random_documents(subtests):
    for name in sorted(ENCODERS):
        with subtests.test(name=name):
            encode, lenient = ENCODERS[name]
            shared_documents = 0
            for seed in range(240):
                document = _document(seed, nan_ok=lenient, foreign_ok=lenient and name == "observability")
                expected = encode(document)
                tokenized = sv.tokenize(document, encode)
                if tokenized is None:  # only a token-shaped value outside the rule refuses sharing here
                    assert b'{"$shared"' in expected.replace(b'\\"', b"")
                    continue
                shared_documents += 1
                assert tokenized.whole() == expected
                assert sv.splice(tokenized.skeleton, tokenized.parts) == expected
                assert tokenized.whole_digest() == (hashlib.sha256(expected).hexdigest(), len(expected))
                assert all(hashlib.sha256(raw).hexdigest() == digest for digest, raw in tokenized.parts.items())
                assert set(sv.token_digests(tokenized.skeleton)) == set(tokenized.parts)
                assert tokenized.shared == _rule_count(document)
                expanded = sv.expand(json.loads(tokenized.skeleton), lambda digest: json.loads(tokenized.parts[digest]))
                assert _canonical(expanded) == _canonical(json.loads(expected))
            assert shared_documents > 200


def test_elements_are_the_outermost_dicts_of_lists_reached_through_dicts():
    document = {
        "messages": [{"role": "user", "content": [{"type": "text", "text": "inner block stays inside"}]},
                     {"role": "assistant", "tool_calls": [{"id": "c1"}]}],
        "nested": {"deeper": {"schemas": [{"name": "a"}, "scalar", 3, [{"inside": "a list's list"}]]}},
        "lists": [[{"not": "reached through dicts only"}], [1, 2]],
        "plain": {"a": 1},
        "count": 2,
    }
    tokenized = sv.tokenize(document, _json_bytes)
    assert tokenized is not None and tokenized.shared == 3
    skeleton = json.loads(tokenized.skeleton)
    assert [sv.token_digest(m) != "" for m in skeleton["messages"]] == [True, True]
    assert sv.token_digest(skeleton["nested"]["deeper"]["schemas"][0])
    assert skeleton["nested"]["deeper"]["schemas"][1:] == ["scalar", 3, [{"inside": "a list's list"}]]
    assert skeleton["lists"] == document["lists"] and skeleton["plain"] == {"a": 1}
    parts = {digest: json.loads(raw) for digest, raw in tokenized.parts.items()}
    assert document["messages"][0] in parts.values(), "the element travels whole, its inner list inside it"
    assert tokenized.whole() == _json_bytes(document)


def test_a_token_shaped_value_outside_the_rule_keeps_the_whole_form():
    digest = "a" * 64
    assert sv.tokenize({"messages": [{"role": "user"}], "ref": sv.token(digest)}, _json_bytes) is None
    inline = sv.tokenize({"messages": [sv.token(digest)]}, _json_bytes, part_bytes=lambda _element: None)
    assert inline is None, "an element kept inline that IS a token would be expanded as one"
    shared = sv.tokenize({"messages": [sv.token(digest)]}, _json_bytes)
    assert shared is not None and shared.whole() == _json_bytes({"messages": [sv.token(digest)]})
    assert sv.tokenize({"messages": [], "tools": [1, "two"]}, _json_bytes) is None, "nothing shared: nothing to splice"
    assert sv.tokenize(["not", "a", "dict"], _json_bytes) is None


def test_part_bytes_decides_which_elements_stay_inline_and_expansion_never_searches_them():
    private = {"role": "tool", "content": "api_key=sk-secret", "inner": [{"nested": "list of one dict"}]}
    public = {"role": "user", "content": "hello"}
    document = {"messages": [public, private, public], "trace": {"tool_calls": [{"id": "x", "result": "ok"}]}}

    def part_bytes(element):
        return None if "secret" in json.dumps(element) else _json_bytes(element)

    tokenized = sv.tokenize(document, _json_bytes, part_bytes=part_bytes)
    assert tokenized is not None and tokenized.shared == 3 and len(tokenized.parts) == 2
    skeleton = json.loads(tokenized.skeleton)
    assert skeleton["messages"][1] == private, "the private element stays inline, whole, its inner list untouched"
    assert tokenized.whole() == _json_bytes(document)
    expanded = sv.expand(skeleton, lambda digest: json.loads(tokenized.parts[digest]))
    assert expanded == document
    # An inline element that carries a token-shaped dict would make the byte splice ambiguous: whole form.
    carrier = {**private, "inner": [sv.token("b" * 64)]}
    assert sv.tokenize({"messages": [public, carrier]}, _json_bytes, part_bytes=part_bytes) is None


def test_a_dangling_reference_is_malformed_never_a_shorter_document():
    document = {"messages": [{"role": "user", "content": "one"}, {"role": "user", "content": "two"}]}
    tokenized = sv.tokenize(document, _json_bytes)
    missing = sv.token_digests(tokenized.skeleton)[1]
    parts = {digest: raw for digest, raw in tokenized.parts.items() if digest != missing}
    with pytest.raises(sv.SharedValueMissing, match=missing) as raised:
        sv.splice(tokenized.skeleton, parts)
    assert raised.value.digest == missing and isinstance(raised.value, ValueError)
    with pytest.raises(sv.SharedValueMissing, match="working checkpoint"):
        list(sv.segments(tokenized.skeleton, parts, context="working checkpoint"))

    def refuse(digest):
        raise FileNotFoundError(digest)

    with pytest.raises(FileNotFoundError):
        sv.expand(json.loads(tokenized.skeleton), refuse)
    with pytest.raises(FileNotFoundError):
        sv.splice(tokenized.skeleton, refuse)


def test_token_digest_recognizes_only_the_reference_shape():
    digest = "0123456789abcdef" * 4
    assert sv.token_digest(sv.token(digest)) == digest
    assert sv.token_digest({"$shared": digest, "more": 1}) == ""
    assert sv.token_digest({"$shared": "ABC"}) == "" and sv.token_digest({"$shared": 5}) == ""
    assert sv.token_digest({"other": digest}) == "" and sv.token_digest(digest) == ""
    assert sv.TOKEN_PATTERN.search(_json_bytes({"a": sv.token(digest)})).group(1).decode() == digest
    assert sv.TOKEN_PATTERN.search(_machine_bytes({"a": sv.token(digest)})).group(1).decode() == digest
    assert sv.TOKEN_PATTERN.search(_json_bytes({"a": {"$shared": digest, "b": 1}})) is None
    assert sv.TOKEN_PATTERN.search(_json_bytes({"a": '{"$shared": "' + digest + '"}'})) is None, "inside a string"


def test_the_tokenizer_fails_exactly_where_its_encoder_fails():
    lone = {"messages": [{"role": "user", "content": "lone \ud800 surrogate"}]}
    with pytest.raises(UnicodeEncodeError):
        _json_bytes(lone)
    with pytest.raises(UnicodeEncodeError):
        sv.tokenize(lone, _json_bytes)
    mixed_keys = {"messages": [{1: "int key", "a": "str key"}]}
    with pytest.raises(TypeError):
        _json_bytes(mixed_keys)
    with pytest.raises(TypeError):
        sv.tokenize(mixed_keys, _json_bytes)
    nan = {"messages": [{"value": math.nan}]}
    with pytest.raises(ValueError):
        physical_candidate_bytes(nan)
    with pytest.raises(ValueError):
        sv.tokenize(nan, physical_candidate_bytes)
