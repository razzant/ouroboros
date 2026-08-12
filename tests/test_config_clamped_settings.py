"""`config._clamped_number_setting` — the shared numeric-setting reader, held to its docstring.

Every timeout, token budget and pass count in `config.py` goes through this ONE
helper, so the five rules below are the ones a hand-edited settings file or a stray
environment variable is actually tested against:

  * unset  -> the ``SETTINGS_DEFAULTS`` value;
  * malformed -> the DEFAULT, never zero (zero reads as a deliberate "none" to every
    consumer downstream, so falling back to it would silently disable a limit);
  * below the floor -> clamped UP;
  * above the ceiling -> clamped DOWN (no unbounded wait, no nonsense cap);
  * an int-cast setting stays an ``int`` on every one of those paths, including the
    clamped ones — a float leaking into a token budget or a pass count is a
    downstream `TypeError` waiting to happen.

The helper was previously covered only indirectly, through whichever caller a
feature test happened to exercise, which left the clamp itself unpinned.
"""

import pytest

from ouroboros import config


# (key, low, high, cast) — one representative of each cast, plus the two settings
# whose clamps guard an owner-facing read.
CLAMPED = [
    ("OUROBOROS_TASK_DIFF_GIT_TIMEOUT_SEC", 5.0, 300.0, float),
    ("OUROBOROS_THREAD_GIT_TIMEOUT_SEC", 5.0, 300.0, float),
    ("OUROBOROS_SAFETY_CALL_TIMEOUT_SEC", 5.0, 600.0, float),
    ("OUROBOROS_SAFETY_MAX_TOKENS", 256, 16384, int),
    ("OUROBOROS_ACCEPTANCE_RESERVE_PCT", 0, 50, int),
]


@pytest.mark.parametrize("key,low,high,cast", CLAMPED, ids=[c[0] for c in CLAMPED])
def test_clamped_setting_env_precedence_and_bounds(monkeypatch, key, low, high, cast):
    default = cast(config.SETTINGS_DEFAULTS[key])
    assert low <= default <= high, f"{key}: its own default is outside its clamp"

    def read():
        return config._clamped_number_setting(key, low=low, high=high, cast=cast)

    # 1. UNSET -> the declared default, not zero and not the floor.
    monkeypatch.delenv(key, raising=False)
    assert read() == default

    # 2. MALFORMED -> the default. Zero would read as a deliberate "none".
    for junk in ("", "   ", "abc", "12abc", "1e", "None", "-", "5,0"):
        monkeypatch.setenv(key, junk)
        assert read() == default, f"{key}={junk!r} did not fall back to the default"
        assert read() != 0 or default == 0

    # 3. BELOW the floor -> clamped up.
    monkeypatch.setenv(key, str(low - 1))
    assert read() == low
    monkeypatch.setenv(key, "-99999")
    assert read() == low

    # 4. ABOVE the ceiling -> clamped down.
    monkeypatch.setenv(key, str(high + 1))
    assert read() == high
    monkeypatch.setenv(key, "99999999")
    assert read() == high

    # 5. A value INSIDE the range passes through untouched.
    inside = cast(low) + (cast(high) - cast(low)) // 2 if cast is int else (low + high) / 2
    monkeypatch.setenv(key, str(inside))
    assert read() == cast(inside)

    # 6. An int-cast setting stays an int on EVERY path above — a float in a token
    #    budget or a pass count breaks its consumer, not this function.
    if cast is int:
        for value, expected in (
            ("abc", default), (str(low - 1), low), (str(high + 1), high), (str(inside), inside),
        ):
            monkeypatch.setenv(key, value)
            got = read()
            assert isinstance(got, int) and not isinstance(got, bool), f"{key}={value!r} -> {got!r}"
            assert got == expected
        # A float-looking override is MALFORMED for an int cast (int("2.5") raises),
        # so it takes the default rather than silently truncating toward zero.
        monkeypatch.setenv(key, f"{float(inside)}")
        assert read() == default


def test_task_diff_git_timeout_default_and_clamp(monkeypatch):
    """The task-diff endpoint's per-`git` timeout: 30s by default, clamped to 5-300.

    Pinned as its own case because it is the only clamp on an owner-facing read
    (`/api/tasks/<id>/diff`): too low and a large repo's diff can never load, too
    high and one hung `git` holds the request open indefinitely.
    """
    assert config.SETTINGS_DEFAULTS["OUROBOROS_TASK_DIFF_GIT_TIMEOUT_SEC"] == 30

    monkeypatch.delenv("OUROBOROS_TASK_DIFF_GIT_TIMEOUT_SEC", raising=False)
    assert config.get_task_diff_git_timeout_sec() == 30.0

    monkeypatch.setenv("OUROBOROS_TASK_DIFF_GIT_TIMEOUT_SEC", "0")
    assert config.get_task_diff_git_timeout_sec() == 5.0
    monkeypatch.setenv("OUROBOROS_TASK_DIFF_GIT_TIMEOUT_SEC", "100000")
    assert config.get_task_diff_git_timeout_sec() == 300.0
    monkeypatch.setenv("OUROBOROS_TASK_DIFF_GIT_TIMEOUT_SEC", "not-a-number")
    assert config.get_task_diff_git_timeout_sec() == 30.0
    monkeypatch.setenv("OUROBOROS_TASK_DIFF_GIT_TIMEOUT_SEC", "45.5")
    assert config.get_task_diff_git_timeout_sec() == 45.5


def test_thread_git_timeout_is_its_own_knob_with_its_own_default(monkeypatch):
    """Branch off / merge back bound their `git` calls SEPARATELY from the diff.

    Same clamp, different default and a different key on purpose: the diff
    endpoint runs one bounded READ against a commit, while branch-off's
    `@snapshot` base runs `git add -A` + `git commit` over the owner's whole
    working tree. Pointing the write at the read's 30s ceiling satisfied the SSOT
    gate and quietly made a large repository's snapshot time out where it had
    succeeded, which is the failure branch-off's refusals exist to contain.
    """
    assert config.SETTINGS_DEFAULTS["OUROBOROS_THREAD_GIT_TIMEOUT_SEC"] == 120

    monkeypatch.delenv("OUROBOROS_THREAD_GIT_TIMEOUT_SEC", raising=False)
    assert config.get_thread_git_timeout_sec() == 120.0

    monkeypatch.setenv("OUROBOROS_THREAD_GIT_TIMEOUT_SEC", "0")
    assert config.get_thread_git_timeout_sec() == 5.0
    monkeypatch.setenv("OUROBOROS_THREAD_GIT_TIMEOUT_SEC", "100000")
    assert config.get_thread_git_timeout_sec() == 300.0
    monkeypatch.setenv("OUROBOROS_THREAD_GIT_TIMEOUT_SEC", "not-a-number")
    assert config.get_thread_git_timeout_sec() == 120.0
    monkeypatch.setenv("OUROBOROS_THREAD_GIT_TIMEOUT_SEC", "45.5")
    assert config.get_thread_git_timeout_sec() == 45.5

    # The two knobs are INDEPENDENT: moving one must not move the other.
    monkeypatch.delenv("OUROBOROS_THREAD_GIT_TIMEOUT_SEC", raising=False)
    monkeypatch.setenv("OUROBOROS_TASK_DIFF_GIT_TIMEOUT_SEC", "12")
    assert config.get_thread_git_timeout_sec() == 120.0
    assert config.get_task_diff_git_timeout_sec() == 12.0
