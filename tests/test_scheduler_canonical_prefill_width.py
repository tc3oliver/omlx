# SPDX-License-Identifier: Apache-2.0
"""A request's prefill partition must not depend on this process's memory history.

A GatedDeltaNet hybrid builds its recurrent state chunk by chunk, so where the
chunk boundaries land is part of the computation, not an implementation detail.
When the adaptive throttle sized every chunk against whatever memory happened to
be free at that moment, one prompt had an open set of reachable partitions and,
with them, more than one possible reply at temperature 0. Measured on
Qwen3.8-27B-oQ4e-mtp, a 68,034-token prompt: 9x4096+15x2048+449 and
10x4096+14x2048+449 diverge first in the layer-0 recurrent state of the first
differently-sized chunk.

The rule pinned here: on a stateful non-sliceable cache the width is chosen once
per request and pressure may only step it down whole rungs of a halving ladder,
never emit a one-off chunk size. A sliceable KV-only cache reconstructs
identically whatever the sizes were and keeps the old adaptive behaviour.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import mlx.core as mx

from omlx.request import Request, SamplingParams
from omlx.scheduler import Scheduler, SchedulerConfig, _PrefillState

WIDTH = 4096
LADDER = (4096, 2048, 1024)
PROMPT_TOKENS = 20 * WIDTH + 449


class _RecordingModel:
    """Records the partition the scheduler actually submits."""

    def __init__(self):
        self.model_type = "qwen3_next"
        self.layers = []
        self.chunk_lengths: list[int] = []

    def __call__(self, tokens, cache=None):
        self.chunk_lengths.append(int(tokens.shape[1]))


def _recurrent_cache():
    """A stateful non-sliceable layer: state carried across chunks."""
    return [SimpleNamespace(cache=[mx.zeros((1, 1))], state=mx.zeros((1, 1)))]


def _sliceable_cache():
    class KVCache:  # the class name is what the cache classifier keys on
        state = mx.zeros((1, 1))

    return [KVCache()]


def _partition(allowance, *, cache_factory=_recurrent_cache, speed_priority=False):
    """Run one whole prefill under a given memory history; return the partition.

    ``allowance(n)`` stands in for live memory: how many of the ``n`` planned
    tokens the throttle would permit right now. Two processes that reached this
    request through different work have different allowance curves — that is
    exactly the input the partition must not depend on.
    """
    model = _RecordingModel()
    tokenizer = MagicMock()
    tokenizer.eos_token_id = 2
    scheduler = Scheduler(
        model=model,
        tokenizer=tokenizer,
        config=SchedulerConfig(
            prefill_step_size=WIDTH,
            chunked_prefill=True,
            paged_cache_block_size=0,
        ),
    )
    scheduler._prefill_speed_priority = speed_priority
    scheduler._adaptive_chunk_size = lambda n, **kw: min(n, allowance(n))
    scheduler._guard_prefill_chunk = lambda n, **kw: n

    request = Request(
        request_id="rid-partition",
        prompt=list(range(PROMPT_TOKENS + 1)),
        sampling_params=SamplingParams(max_tokens=1),
    )
    request.prompt_token_ids = list(range(PROMPT_TOKENS + 1))
    request.num_prompt_tokens = PROMPT_TOKENS + 1
    state = _PrefillState(
        request=request,
        cache=cache_factory(),
        tokens_remaining=mx.zeros((1, PROMPT_TOKENS), dtype=mx.int32),
        last_token=[99],
        tokens_processed=0,
        base_size=0,
        emitted_boundaries={},
        boundary_enabled=False,
        block_size=0,
        total_length=PROMPT_TOKENS + 1,
        sampler=MagicMock(),
        sm=MagicMock(),
        per_row_lps=[],
    )
    with patch("omlx.scheduler._sync_and_clear_cache"):
        while not scheduler._step_prefill_chunk(state):
            pass
    assert sum(model.chunk_lengths) == PROMPT_TOKENS
    return model.chunk_lengths


# --- the regression --------------------------------------------------------


def test_reachable_partitions_are_a_closed_set():
    """The core property: the partitions one prompt can reach are bounded.

    This bounds them; it does not pin them -- the step-down *point* is still a
    function of live memory. But every reachable partition is now built from
    ladder rungs, so the set is finite and known instead of open. On the
    unfixed scheduler each distinct allowance curve below reaches its own
    distinct partition.
    """
    curves = [lambda n: n, lambda n: 100_000, lambda n: 4096,
              lambda n: 2753, lambda n: 3000, lambda n: 2048,
              lambda n: 1500, lambda n: 1024]
    partitions = {tuple(_partition(c)) for c in curves}
    assert len(partitions) <= len(LADDER), sorted(len(p) for p in partitions)
    for part in partitions:
        assert set(part[:-1]) <= set(LADDER), part[:5]


def test_pressure_steps_the_whole_request_down_one_rung():
    """2753 available means the request runs at 2048, not in 2753-token chunks."""
    partition = _partition(lambda n: 2753)
    assert set(partition[:-1]) == {2048}, partition


def test_severe_pressure_reaches_the_narrowest_rung_and_stays_there():
    partition = _partition(lambda n: 1500)
    assert set(partition[:-1]) == {1024}, partition


def test_narrowest_rung_defers_to_the_safety_mechanisms():
    """Memory safety outranks determinism once the ladder is exhausted."""
    partition = _partition(lambda n: 700)
    assert max(partition) == 700, partition[:5]


# --- scope -----------------------------------------------------------------


def test_sliceable_cache_keeps_the_old_adaptive_behaviour():
    """A KV-only cache reconstructs identically, so nothing is pinned."""
    partition = _partition(lambda n: 2753, cache_factory=_sliceable_cache)
    assert 2753 in partition, partition[:5]


def test_speed_priority_path_is_untouched():
    """Speed priority already pins the width by refusing to shrink."""
    partition = _partition(lambda n: 2753, speed_priority=True)
    assert 2753 in partition, partition[:5]
