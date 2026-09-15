# SPDX-License-Identifier: Apache-2.0
"""A request's prefill width must not depend on this process's memory history.

For a stateful non-sliceable cache (GatedDeltaNet and other recurrent hybrids)
the recurrent state is built chunk by chunk, so where the chunk boundaries land
is part of the computation. Before this was pinned, the adaptive throttle sized
every chunk against whatever memory happened to be free, and two runs of the
same prompt at temperature 0 could take different partitions and emit different
tokens. Measured on Qwen3.8-27B-oQ4e-mtp: a 68,034-token prompt ran
13x4096+2753 twice (identical output) and 13x4096+2497 once (different output).

These tests pin the rule: the width is chosen once, and pressure may only step
it down whole ladder rungs, never produce a one-off chunk size.
"""

from types import SimpleNamespace

import pytest

from omlx.scheduler import Scheduler


class _FakeScheduler:
    """Just enough scheduler to drive _plan_prefill_chunk.

    The real throttle and guard are replaced by a caller-supplied allowance so
    a test can say "only N tokens fit right now" without building a model.
    """

    _CANONICAL_WIDTH_FLOOR = Scheduler._CANONICAL_WIDTH_FLOOR
    _canonical_prefill_ladder = Scheduler._canonical_prefill_ladder
    _next_canonical_width = Scheduler._next_canonical_width
    _canonical_prefill_active = Scheduler._canonical_prefill_active
    _plan_prefill_chunk = Scheduler._plan_prefill_chunk

    def __init__(self, allowance, *, speed_priority=False, stateful=True):
        # allowance(n) -> how many tokens the safety mechanisms permit for a
        # planned chunk of n.
        self._allowance = allowance
        self._prefill_speed_priority = speed_priority
        self._stateful = stateful
        self.throttle_calls = []

    def _cache_list_needs_boundary_snapshot(self, cache_list):
        return self._stateful

    def _adaptive_chunk_size(self, n, **kw):
        self.throttle_calls.append(n)
        return min(n, self._allowance(n))

    def _guard_prefill_chunk(self, n, **kw):
        return n


def _state(block_size=4096, boundary_enabled=True):
    return SimpleNamespace(
        request=SimpleNamespace(request_id="rid-test"),
        cache=[object()],
        tokens_processed=0,
        base_size=0,
        boundary_enabled=boundary_enabled,
        block_size=block_size,
        canonical_width=0,
        width_fallbacks=0,
    )


def _plan(sched, state, remaining=40960, step=4096, kv_len=0):
    return sched._plan_prefill_chunk(
        state,
        remaining=remaining,
        prefill_step_size=step,
        kv_len=kv_len,
        gathered_core=False,
    )


# --- the ladder ------------------------------------------------------------


def test_ladder_halves_so_every_rung_divides_the_widest():
    s = _FakeScheduler(lambda n: n)
    assert s._canonical_prefill_ladder(4096) == (4096, 2048, 1024)
    assert s._canonical_prefill_ladder(2048) == (2048, 1024, 512)
    # Never below the floor, and never empty.
    assert s._canonical_prefill_ladder(512) == (512,)
    assert s._canonical_prefill_ladder(256) == (256,)


def test_step_down_walks_the_ladder_then_stops():
    s = _FakeScheduler(lambda n: n)
    assert s._next_canonical_width(4096, 4096) == 2048
    assert s._next_canonical_width(2048, 4096) == 1024
    assert s._next_canonical_width(1024, 4096) is None


# --- Test 1: same prompt, different memory history, same partition ---------


@pytest.mark.parametrize("allowance", [4096, 5000, 100000])
def test_same_geometry_under_different_memory_history(allowance):
    """Any pressure that still admits the full width must not change it.

    This is the regression: an allowance of 4096 and an allowance of 100000
    used to produce different chunk sizes further into the prefill.
    """
    sched = _FakeScheduler(lambda n, a=allowance: a)
    state = _state()
    widths = [_plan(sched, state, remaining=40960 - i * 4096) for i in range(5)]
    assert widths == [4096] * 5
    assert state.canonical_width == 4096
    assert state.width_fallbacks == 0


# --- Test 2: 4096 unavailable -> whole request stays at 2048 ---------------


def test_pressure_steps_the_whole_request_down_one_rung():
    sched = _FakeScheduler(lambda n: 2048 if n > 2048 else n)
    state = _state()

    first = _plan(sched, state)
    assert first == 2048
    assert state.canonical_width == 2048
    assert state.width_fallbacks == 1

    # Every later chunk stays on the new rung, and does not drift back up
    # even though the allowance would now permit the planned size.
    later = [_plan(sched, state, remaining=40960 - i * 2048) for i in range(1, 5)]
    assert later == [2048] * 4
    assert state.width_fallbacks == 1


def test_severe_pressure_steps_down_to_the_narrowest_rung():
    sched = _FakeScheduler(lambda n: 1024 if n > 1024 else n)
    state = _state()
    assert _plan(sched, state) == 1024
    assert state.canonical_width == 1024
    assert state.width_fallbacks == 2  # 4096 -> 2048 -> 1024


# --- Test 3: no arbitrary partial geometry --------------------------------


def test_never_emits_a_width_off_the_ladder():
    """An allowance of 2753 must not become a 2753-token chunk."""
    sched = _FakeScheduler(lambda n: min(n, 2753))
    state = _state()
    widths = {_plan(sched, state, remaining=40960 - i * 4096) for i in range(4)}
    assert widths <= {4096, 2048, 1024}, widths
    assert 2753 not in widths


def test_narrowest_rung_defers_to_the_safety_mechanisms():
    """Memory safety outranks determinism once the ladder is exhausted."""
    sched = _FakeScheduler(lambda n: min(n, 700))
    state = _state()
    n = _plan(sched, state)
    assert n == 700
    assert state.canonical_width == 1024
    assert state.width_fallbacks == 2


# --- scope: only where the partition actually matters ---------------------


def test_sliceable_cache_keeps_the_old_adaptive_behaviour():
    """A KV-only cache reconstructs identically, so nothing is pinned."""
    sched = _FakeScheduler(lambda n: min(n, 2753), stateful=False)
    state = _state()
    assert _plan(sched, state) == 2753
    assert state.canonical_width == 0
    assert state.width_fallbacks == 0


def test_speed_priority_path_is_untouched():
    """Speed priority already pins the width by refusing to shrink."""
    sched = _FakeScheduler(lambda n: min(n, 2753), speed_priority=True)
    state = _state()
    assert _plan(sched, state) == 2753
    assert state.canonical_width == 0


# --- boundary interaction --------------------------------------------------


def test_short_tail_before_a_block_boundary_is_not_a_step_down():
    """Clamping to a boundary is expected, not memory pressure."""
    sched = _FakeScheduler(lambda n: n)
    state = _state()
    state.base_size = 2048  # 2048 tokens into a 4096-token block
    n = _plan(sched, state, kv_len=2048)
    assert n == 2048  # clamped to land on 4096
    assert state.canonical_width == 4096  # width itself unchanged
    assert state.width_fallbacks == 0


def test_remaining_shorter_than_width_is_not_a_step_down():
    sched = _FakeScheduler(lambda n: n)
    state = _state(boundary_enabled=False)
    n = _plan(sched, state, remaining=900)
    assert n == 900
    assert state.width_fallbacks == 0


# --- termination and mid-request semantics ---------------------------------


def test_step_down_is_bounded_by_the_ladder():
    """Pressure can never spin the planner: at most len(ladder)-1 step-downs."""
    calls = []

    def allowance(n):
        calls.append(n)
        return 1  # nothing ever fits

    sched = _FakeScheduler(allowance)
    state = _state()
    n = _plan(sched, state)
    assert n == 1
    assert state.width_fallbacks == 2  # 4096 -> 2048 -> 1024, then stop
    assert len(calls) == 3  # one throttle ask per rung, no more


def test_planner_never_returns_a_non_positive_chunk():
    """A zero-width chunk would make the prefill loop stop making progress."""
    sched = _FakeScheduler(lambda n: max(1, n // 10))
    state = _state()
    for i in range(4):
        n = _plan(sched, state, remaining=40960 - i * 1024)
        assert n >= 1


def test_mid_request_step_down_keeps_earlier_chunks_wide():
    """Documents current semantics: the step-down applies from where it happens.

    Chunks already submitted at the wider rung are not revisited, so the
    partition still records *when* pressure arrived. This is the known
    limitation called out in the PR body: the width is fixed per request, the
    step-down point is not.
    """
    pressure = {"on": False}
    sched = _FakeScheduler(lambda n: 2048 if (pressure["on"] and n > 2048) else n)
    state = _state()

    first = [_plan(sched, state, remaining=40960 - i * 4096) for i in range(2)]
    assert first == [4096, 4096]
    assert state.width_fallbacks == 0

    pressure["on"] = True
    later = [_plan(sched, state, remaining=32768 - i * 2048) for i in range(2)]
    assert later == [2048, 2048]
    assert state.canonical_width == 2048
    assert state.width_fallbacks == 1
