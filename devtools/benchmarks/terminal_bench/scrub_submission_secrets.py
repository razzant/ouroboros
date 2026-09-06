"""Scrub secret values from a TB submission pack copy before ``harbor upload``.

Submitted leaderboard jobs become PUBLIC. Runs launched with
``OUROBOROS_BENCH_ALLOW_CONTAINER_SECRETS=1`` persist live provider keys inside
every trial (``agent/ouroboros-data/settings.json``), and values may leak into
logs. This tool must be run on a COPY of the job directory (never the archived
original) before the first upload.

Harbor ALSO persists its own ``JobConfig`` into the job directory, so every
value the launcher passed through harbor's ``--ae`` / ``--ve`` (agent-/verifier-
phase env) is on disk inside the tree that gets uploaded. Those values usually
appear in no ``--secrets-from`` source at all (a judge key handed to ``--ve`` on
the command line is the Harbor-Index case), so they are supplied separately with
``--env-passthrough NAME=VALUE`` — the exact pairs given to ``--ae``/``--ve`` —
and swept by VALUE like any other secret.

Two passes:
1. Structural: every ``settings.json`` under an ``ouroboros-data`` directory
   gets known secret fields (``common.secrets.SECRET_KEYS`` + extras) blanked.
2. Value sweep: every occurrence of every secret VALUE — collected from
   ``--secrets-from`` sources AND from ``--env-passthrough`` — is replaced with
   ``<REDACTED:NAME>`` in text files; a secret found inside a non-text file is a
   hard error. The sweep walks the WHOLE tree, so harbor's written job config is
   covered by construction, with no filename special-case to keep in sync.

"Occurrence" is NOT only the literal one. Harbor persists env values through JSON
serializers, so a value containing a quote, a backslash, a control character or a
non-ASCII character is on disk in its ESCAPED form (``abc"12345678`` is stored as
``abc\"12345678``). Every secret is therefore expanded into the forms a JSON
writer can produce (``json_encoded_forms``), and BOTH the replacement pass and the
verify pass use that expanded needle set.

A final independent verify pass re-scans the tree for every needle — literal and
encoded — and exits non-zero on any hit, so "clean" can never be reported while a
secret survives in any persisted form. Secret values are never printed.

Fail-closed: an ``--env-passthrough`` value that cannot be swept safely (too
short, or not credential-shaped, so replacing it could rewrite unrelated text)
aborts the run BEFORE anything is modified — refusing to produce a submission is
correct, uploading a maybe-scrubbed tree is not.

Usage:
    python scrub_submission_secrets.py --root <job_dir_copy> \
        --secrets-from ~/ouro/data/settings.json --secrets-from ~/ouro/file1.txt \
        --env-passthrough JUDGE_API_KEY=<the value passed to harbor --ve>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.secrets import SECRET_KEYS  # noqa: E402

# Fields blanked in embedded settings.json files, beyond common.secrets.
EXTRA_SECRET_FIELDS = (
    "OPENAI_COMPATIBLE_API_KEY",
    "CLOUDRU_FOUNDATION_MODELS_API_KEY",
    "MINIMAX_API_KEY",
    "DEEPSEEK_API_KEY",
    "GIGACHAT_PASSWORD",
    "OUROBOROS_NETWORK_PASSWORD",
    "TELEGRAM_BOT_TOKEN",
)
MIN_SECRET_LEN = 8  # ignore trivially short values (avoid mass false rewrites)


def harbor_redacted_form(value: str) -> str:
    """The partial form harbor 0.18.0 itself persists for a *sensitive-looking* env NAME.

    Verified against the installed package: ``harbor/utils/env.py::redact_sensitive_value`` is
    ``value[:4] + "****" + value[-3:]`` (``"****"`` when ``len(value) <= 8``), applied by the
    ``env`` field serializer on ``AgentConfig`` / ``VerifierConfig`` / ``EnvironmentConfig``
    (``harbor/models/trial/config.py``) whenever the KEY matches
    ``KEY|SECRET|TOKEN|PASSWORD|CREDENTIAL|AUTH`` and does not equal ``os.environ[KEY]``.
    That is a partial disclosure, not a redaction: it publishes the first four and last three
    characters of a live credential. A literal-value sweep cannot see it (the literal is no
    longer there), so the derived form is swept as its own needle.

    Returns ``""`` when there is nothing specific to sweep (``"****"`` carries no secret and
    matching it would rewrite unrelated text).
    """
    if len(value) <= 8:
        return ""
    return value[:4] + "****" + value[-3:]


def json_encoded_forms(value: str) -> list[str]:
    r"""The forms ``value`` takes on disk once a JSON writer has serialized it.

    Everything harbor persists an env value into is JSON: the job config
    (``self.config.model_dump_json(...)``), ``lock.json``, every trial's
    ``config.json`` / ``result.json``, and the embedded ouroboros ``settings.json``.
    A JSON string is not the literal bytes of the value — ``abc"12345678`` is stored
    as ``abc\"12345678`` — so a literal-only sweep walks straight past a secret that
    contains a quote, a backslash, a control character or a non-ASCII character, and
    a literal-only verify then reports the tree clean. That combination is worse than
    having no scrubber at all, because it replaces "check this by hand" with a false
    all-clear.

    The forms are obtained by running the SERIALIZER over the value and stripping the
    quotes it added — never by a hand-written escape table. An escape table is a list
    of yesterday's cases and goes stale the moment an encoder changes; ``json.dumps``
    IS the encoder, so whatever it escapes is what we search for, by construction.

    Two configurations cover every JSON writer in play, and they are the only two
    that exist: ``ensure_ascii=True`` (Python's ``json`` default, as used by
    ``_blank_settings_fields`` and ouroboros itself — non-ASCII becomes ``\uXXXX``)
    and ``ensure_ascii=False`` (what pydantic/serde emit for harbor's own files —
    non-ASCII stays raw UTF-8). Forms identical to the literal value are dropped: the
    literal is already swept as its own needle.
    """
    forms: list[str] = []
    for ensure_ascii in (True, False):
        # json.dumps always wraps a str in exactly one pair of double quotes.
        encoded = json.dumps(value, ensure_ascii=ensure_ascii)[1:-1]
        if encoded != value and encoded not in forms:
            forms.append(encoded)
    return forms


def expand_encoded_forms(secrets: dict[str, str]) -> dict[str, str]:
    """``secrets`` plus, for each value, its JSON-serialized forms as extra needles.

    Labels are suffixed ``:json`` (``:json2`` for a second distinct form) purely so
    verify output and the ``<REDACTED:...>`` marker say WHICH form was found; the
    sweep and the verify both match by VALUE. Applied to the merged needle set, so
    harbor's partial-disclosure form is expanded too (it can itself contain a quote).
    """
    expanded = dict(secrets)
    for name, value in secrets.items():
        for index, form in enumerate(json_encoded_forms(value)):
            if form in expanded.values():
                continue  # already covered by another needle
            label = f"{name}:json" if index == 0 else f"{name}:json{index + 1}"
            key, suffix = label, 2
            while expanded.get(key, form) != form:
                key, suffix = f"{label}#{suffix}", suffix + 1
            expanded[key] = form
    return expanded


def unsweepable_reason(value: str) -> str:
    """Why an ``--ae``/``--ve`` value cannot be swept safely — ``""`` when it can.

    Fail-closed by design: a value we cannot sweep is a refusal to produce a submission, never
    a best-effort pass. Sweeping a low-specificity string would corrupt unrelated content in
    the tree; NOT sweeping it would publish a credential. Neither is acceptable, so the
    operator has to resolve it (drop the value from ``--ae``/``--ve``, or handle that file by
    hand and re-run the verify).
    """
    if len(value) < MIN_SECRET_LEN:
        return f"shorter than {MIN_SECRET_LEN} chars"
    if value != value.strip():
        return "has leading/trailing whitespace"
    if not (any(ch.isdigit() for ch in value) and any(ch.isalpha() for ch in value)):
        return ("not credential-shaped (needs both letters and digits); a blind sweep could "
                "rewrite unrelated text")
    return ""


def collect_env_passthrough(items: list[str]) -> tuple[dict[str, str], list[str]]:
    """Parse ``NAME=VALUE`` pairs exactly as passed to harbor's ``--ae`` / ``--ve``.

    Harbor persists its own ``JobConfig`` to the job directory
    (``harbor/job.py``: ``self._job_config_path.write_text(self.config.model_dump_json(...))``,
    plus ``lock.json`` and every trial's ``config.json``/``lock.json``/``result.json``), so
    these values are on disk inside the tree that gets uploaded. Returns
    ``(label -> needle, refusals)``; never echoes a value, including in error text.

    EVERY distinct occurrence is kept. The same env NAME legitimately carries DIFFERENT values
    in the agent and verifier phases (``--ae KEY=a --ve KEY=b``, or a repeated flag), and a dict
    keyed on the NAME alone silently dropped the earlier value — a correct operator invocation
    then published that credential in harbor's job tree. Colliding names are disambiguated with
    a ``#2``/``#3`` label suffix, which is cosmetic: the label only names the needle in the
    ``<REDACTED:...>`` marker and in verify output, while the sweep matches by VALUE.
    """
    secrets: dict[str, str] = {}
    refusals: list[str] = []

    def keep(label: str, needle: str) -> None:
        """Store ``needle`` under ``label``, never overwriting a DIFFERENT value."""
        key, suffix = label, 2
        while secrets.get(key, needle) != needle:
            key, suffix = f"{label}#{suffix}", suffix + 1
        secrets[key] = needle

    for item in items:
        name, sep, value = str(item).partition("=")
        name = name.strip()
        if not sep or not name:
            # The token itself may BE the value, so it is not repeated in the message.
            refusals.append("<one item>: expected NAME=VALUE (as passed to harbor --ae/--ve)")
            continue
        reason = unsweepable_reason(value)
        if reason:
            refusals.append(f"{name}: {reason}")
            continue
        keep(name, value)
        partial = harbor_redacted_form(value)
        if partial:
            keep(f"{name}:harbor-partial", partial)
    return secrets, refusals


def collect_secrets(sources: list[Path]) -> dict[str, str]:
    """Collect name -> value pairs from settings.json / ``name: value`` files."""
    secrets: dict[str, str] = {}

    def add(name: str, value: object) -> None:
        if isinstance(value, str) and len(value) >= MIN_SECRET_LEN:
            secrets[name] = value

    for src in sources:
        text = src.read_text(encoding="utf-8", errors="replace")
        try:
            data = json.loads(text)
        except ValueError:
            data = None
        if isinstance(data, dict):
            for key, value in data.items():
                upper = key.upper()
                if any(tag in upper for tag in ("KEY", "TOKEN", "PASSWORD", "SECRET")):
                    add(key, value)
            continue
        for line in text.splitlines():
            if ":" not in line:
                continue
            name, _, value = line.partition(":")
            add(name.strip() or "unnamed", value.strip())
    return secrets


def _blank_settings_fields(path: Path) -> int:
    try:
        data = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except (OSError, ValueError):
        return 0
    if not isinstance(data, dict):
        return 0
    blanked = 0
    for field in (*SECRET_KEYS, *EXTRA_SECRET_FIELDS):
        if isinstance(data.get(field), str) and data[field]:
            data[field] = ""
            blanked += 1
    if blanked:
        path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    return blanked


def _sweep_file(path: Path, secrets: dict[str, str]) -> tuple[int, list[str]]:
    """Replace secret values in one file; returns (replacements, binary_hits)."""
    try:
        raw = path.read_bytes()
    except OSError:
        return 0, []
    hits = [name for name, value in secrets.items() if value.encode() in raw]
    if not hits:
        return 0, []
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        return 0, hits  # secret inside a non-text file: caller fails hard
    replaced = 0
    # Longest needle first: a value's encoded form can contain the literal one (``\`` inside
    # ``\\``), and replacing the shorter form first would leave a fragment of the longer.
    for name in sorted(hits, key=lambda n: len(secrets[n]), reverse=True):
        value = secrets[name]
        replaced += text.count(value)
        text = text.replace(value, f"<REDACTED:{name}>")
    path.write_text(text, encoding="utf-8")
    return replaced, []


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument(
        "--secrets-from", action="append", required=True, type=Path, dest="sources"
    )
    parser.add_argument(
        "--env-passthrough", action="append", default=[], dest="env_passthrough",
        metavar="NAME=VALUE",
        help="a NAME=VALUE pair exactly as it was passed to harbor's --ae/--ve. Harbor writes "
             "its own JobConfig (and job/trial lock + result files) into the job directory, so "
             "these values are inside the uploaded tree and must be swept by VALUE. Repeatable.",
    )
    args = parser.parse_args()

    root: Path = args.root
    if not root.is_dir():
        parser.error(f"not a directory: {root}")
    secrets = collect_secrets(args.sources)
    env_secrets, refusals = collect_env_passthrough(args.env_passthrough or [])
    if refusals:
        # Fail closed BEFORE touching a single file: a partially swept tree that then gets
        # uploaded is strictly worse than no submission.
        print("REFUSING TO SCRUB: --env-passthrough values that cannot be swept safely:",
              file=sys.stderr)
        for line in refusals:
            print(f"  {line}", file=sys.stderr)
        print("  (nothing was modified. Remove the value from harbor's --ae/--ve, or scrub "
              "that occurrence by hand and re-run to verify.)", file=sys.stderr)
        return 2
    for name, value in env_secrets.items():
        key, suffix = name, 2
        while key in secrets and secrets[key] != value:
            key, suffix = f"{name}#{suffix}", suffix + 1
        secrets[key] = value
    if not secrets:
        parser.error("no secret values collected from --secrets-from sources")
    # Sweep AND verify against every form a JSON writer can have persisted the value in.
    # Both passes must use the same expanded set: a verify narrower than the sweep would
    # certify a tree it never fully inspected.
    needles = expand_encoded_forms(secrets)

    # A symlink anywhere under --root breaks BOTH halves of this tool's promise, so it is
    # refused before a single byte is written — the same fail-closed discipline as the
    # --env-passthrough refusal above.
    #
    #   * a FILE symlink is followed by both `p.is_file()` and `path.write_text()`, so the
    #     sweep edits the TARGET, outside --root. A pack containing a link to the live
    #     settings.json would have its real keys replaced with <REDACTED:...> by the very
    #     tool meant to protect them. `cp -a` preserves symlinks, so the procedural "run
    #     this on a COPY" rule does not save you.
    #   * a DIRECTORY symlink is not traversed by rglob at all, yet the verify pass then
    #     prints verify_leftovers=0 and exits 0 — an affirmative certification of content
    #     the tool never read, for a tree that is about to be uploaded publicly. Silent
    #     non-coverage reported as cleanliness is the one failure this tool must not have.
    #
    # Refusing is right rather than resolving-and-continuing: under --root a symlink is
    # either an accident or an escape, and the operator must decide which.
    symlinks = sorted(p for p in root.rglob("*") if p.is_symlink())
    if symlinks:
        print("REFUSING TO SCRUB: symlinks under --root cannot be swept or verified "
              "safely:", file=sys.stderr)
        for link in symlinks:
            try:
                target = str(link.readlink())
            except OSError:
                target = "<unreadable>"
            kind = "dir" if link.is_dir() else "file"
            print(f"  {link} -> {target} ({kind})", file=sys.stderr)
        print("  (nothing was modified. Replace each link with a real copy — "
              "`cp -aL`, or `tar --dereference` when building the pack — and re-run.)",
              file=sys.stderr)
        return 2

    files = [p for p in root.rglob("*") if p.is_file()]

    blanked_fields = 0
    for path in files:
        if path.name == "settings.json" and "ouroboros-data" in path.parts:
            blanked_fields += _blank_settings_fields(path)

    replacements = 0
    binary_failures: list[str] = []
    for path in files:
        count, binary_hits = _sweep_file(path, needles)
        replacements += count
        if binary_hits:
            binary_failures.append(f"{path}: {sorted(set(binary_hits))}")

    if binary_failures:
        print("SECRET VALUES IN NON-TEXT FILES (fix manually):", file=sys.stderr)
        for line in binary_failures:
            print(f"  {line}", file=sys.stderr)
        return 2

    # Independent verify pass: re-read everything, zero tolerance.
    leftovers = 0
    for path in files:
        try:
            raw = path.read_bytes()
        except OSError:
            continue
        for name, value in needles.items():
            if value.encode() in raw:
                leftovers += 1
                print(f"LEFTOVER {name} in {path}", file=sys.stderr)
    print(
        f"secrets={len(secrets)} env_passthrough_needles={len(env_secrets)} "
        f"needles_with_encoded_forms={len(needles)} files={len(files)} "
        f"settings_fields_blanked={blanked_fields} value_replacements={replacements} "
        f"verify_leftovers={leftovers}"
    )
    return 1 if leftovers else 0


if __name__ == "__main__":
    raise SystemExit(main())
