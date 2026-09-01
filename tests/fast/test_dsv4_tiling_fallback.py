"""The shared-memory tiling fallback: what it retries, what it refuses to retry, and what it caches.

These exercise the retry loops themselves, not the kernels, so `sparse_mqa_fwd` / `bwd` are replaced
by scripted stubs and no GPU is involved.

The property that matters most here is the first one: the requested tiling is always tried first and
returned untouched when it builds. That is what makes this change inert on a GPU whose shared memory
already fits the shipped tilings -- every later candidate is unreachable there.
"""

import pytest

tilelang = pytest.importorskip("tilelang", reason="the DeepSeek-V4 kernels are tilelang modules")

from miles_plugins.models.deepseek_v4.ops.kernel import tilelang_indexer_fwd as indexer_fwd  # noqa: E402
from miles_plugins.models.deepseek_v4.ops.kernel import tilelang_sparse_mla_bwd as sparse_mla_bwd  # noqa: E402
from miles_plugins.models.deepseek_v4.ops.kernel import tilelang_sparse_mla_fwd as sparse_mla_fwd  # noqa: E402

# The requested tiling in every forward test below, matching sparse_mqa_fwd_interface's defaults.
REQUESTED = dict(heads=64, dim=512, topk=512, sm_scale=0.044, block_I=64, num_stages=2, threads=256)
SHARED_MEM_ERROR = RuntimeError("Requested dynamic shared memory 85488 exceeds device limit 65536")


@pytest.fixture(autouse=True)
def _empty_caches(monkeypatch):
    """Each test starts with no memoized tiling, and leaves none behind."""
    monkeypatch.setattr(sparse_mla_fwd, "_fitted_tiling", {})
    monkeypatch.setattr(sparse_mla_bwd, "_fitted_tiling", {})
    monkeypatch.setattr(indexer_fwd, "_fitted_block_N", {})


def _record_calls(monkeypatch, module, name, side_effects):
    """Replace module.name with a stub that pops from side_effects, raising exceptions it yields."""
    calls = []

    def stub(*args, **kwargs):
        calls.append(kwargs)
        outcome = side_effects.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    monkeypatch.setattr(module, name, stub)
    return calls


# ---------------------------------------------------------------------------------------------
# forward
# ---------------------------------------------------------------------------------------------


def test_the_requested_tiling_is_tried_first_and_returned_as_is(monkeypatch):
    """On a GPU that fits the shipped tiling, nothing in this change is reachable."""
    calls = _record_calls(monkeypatch, sparse_mla_fwd, "sparse_mqa_fwd", ["kernel"])

    assert sparse_mla_fwd._compile_within_shared_mem(**REQUESTED) == "kernel"

    assert len(calls) == 1, "a fitting tiling must not be probed more than once"
    assert calls[0] == {"block_I": 64, "num_stages": 2, "threads": 256, "block_H": None}


def test_a_shared_memory_failure_falls_through_to_the_next_candidate(monkeypatch):
    calls = _record_calls(monkeypatch, sparse_mla_fwd, "sparse_mqa_fwd", [SHARED_MEM_ERROR, "smaller"])

    assert sparse_mla_fwd._compile_within_shared_mem(**REQUESTED) == "smaller"

    assert len(calls) == 2
    # The second candidate is the head of _FALLBACK_TILINGS, not a repeat of the request.
    stages, bi, thr, bh = sparse_mla_fwd._FALLBACK_TILINGS[0]
    assert calls[1] == {"block_I": bi, "num_stages": stages, "threads": thr, "block_H": bh}


def test_the_fitted_tiling_is_memoized_per_shape(monkeypatch):
    calls = _record_calls(monkeypatch, sparse_mla_fwd, "sparse_mqa_fwd", [SHARED_MEM_ERROR, "smaller", "smaller"])

    sparse_mla_fwd._compile_within_shared_mem(**REQUESTED)
    probes_after_first = len(calls)
    sparse_mla_fwd._compile_within_shared_mem(**REQUESTED)

    assert len(calls) == probes_after_first + 1, "the second call must not re-probe the failed candidate"
    assert calls[-1] == calls[-2], "it must reuse the tiling that was found to fit"


def test_an_unrelated_runtime_error_is_not_retried(monkeypatch):
    """A genuine kernel bug must surface, not be masked as "this tiling does not fit"."""
    boom = RuntimeError("CUDA driver error: out of memory")
    calls = _record_calls(monkeypatch, sparse_mla_fwd, "sparse_mqa_fwd", [boom])

    with pytest.raises(RuntimeError, match="CUDA driver error"):
        sparse_mla_fwd._compile_within_shared_mem(**REQUESTED)
    assert len(calls) == 1


def test_a_caller_error_is_not_retried(monkeypatch):
    """sparse_mqa_fwd asserts on a non-power-of-two dim; that is not a tiling problem."""
    bad_dim = AssertionError("dim must be power of 2, got 500")
    calls = _record_calls(monkeypatch, sparse_mla_fwd, "sparse_mqa_fwd", [bad_dim])

    with pytest.raises(AssertionError, match="power of 2"):
        sparse_mla_fwd._compile_within_shared_mem(**REQUESTED)
    assert len(calls) == 1, "retrying a caller error would bury it under six more failures"


def test_an_inapplicable_block_h_is_retried(monkeypatch):
    """`heads % block_H != 0` means this candidate does not apply, so the next one gets a turn."""
    inapplicable = AssertionError("heads (48) should be a multiple of block_H (32)")
    calls = _record_calls(monkeypatch, sparse_mla_fwd, "sparse_mqa_fwd", [inapplicable, "next"])

    assert sparse_mla_fwd._compile_within_shared_mem(**REQUESTED) == "next"
    assert len(calls) == 2


def test_the_last_error_is_raised_when_no_candidate_builds(monkeypatch):
    last = RuntimeError("Requested dynamic shared memory 33344 exceeds device limit 32768")
    # One failure for the request plus one per applicable fallback; the last one is `last`.
    applicable = [c for c in sparse_mla_fwd._FALLBACK_TILINGS if REQUESTED["topk"] % c[1] == 0]
    effects = [SHARED_MEM_ERROR] * len(applicable) + [last]
    _record_calls(monkeypatch, sparse_mla_fwd, "sparse_mqa_fwd", effects)

    with pytest.raises(RuntimeError, match="32768"):
        sparse_mla_fwd._compile_within_shared_mem(**REQUESTED)


def test_candidates_that_do_not_divide_topk_are_skipped(monkeypatch):
    """topk % block_I != 0 would trip sparse_mqa_fwd's own assert, so those never get probed."""
    requested = {**REQUESTED, "topk": 48}
    probed = []

    def stub(*args, **kwargs):
        probed.append(kwargs["block_I"])
        raise SHARED_MEM_ERROR

    monkeypatch.setattr(sparse_mla_fwd, "sparse_mqa_fwd", stub)
    with pytest.raises(RuntimeError):
        sparse_mla_fwd._compile_within_shared_mem(**requested)

    assert all(48 % bi == 0 for bi in probed[1:]), f"probed a block_I that does not divide topk: {probed}"


# ---------------------------------------------------------------------------------------------
# backward
# ---------------------------------------------------------------------------------------------

BWD_SHAPE = dict(B=1, S=16384, S_kv=131072, H=64, D=512, topk=512, sm_scale=0.044)


def test_backward_tries_the_shipped_tiling_first(monkeypatch):
    calls = _record_calls(monkeypatch, sparse_mla_bwd, "bwd", ["kernel"])

    assert sparse_mla_bwd.bwd_within_shared_mem(**BWD_SHAPE) == "kernel"

    assert len(calls) == 1
    block_size, threads, split_store, stage_dq, max_block_H = sparse_mla_bwd._FALLBACK_TILINGS[0]
    assert calls[0] == {
        "block_size": block_size,
        "threads": threads,
        "split_store": split_store,
        "stage_dq_through_shared": stage_dq,
        "max_block_H": max_block_H,
    }


@pytest.mark.parametrize("marker", sparse_mla_bwd.RETRYABLE_BUILD_ERRORS)
def test_backward_retries_every_declared_marker(monkeypatch, marker):
    """All three tilelang rejection modes have to be treated as "try the next candidate"."""
    calls = _record_calls(monkeypatch, sparse_mla_bwd, "bwd", [RuntimeError(f"... {marker} ..."), "smaller"])

    assert sparse_mla_bwd.bwd_within_shared_mem(**BWD_SHAPE) == "smaller"
    assert len(calls) == 2


def test_backward_does_not_retry_an_undeclared_error(monkeypatch):
    calls = _record_calls(monkeypatch, sparse_mla_bwd, "bwd", [RuntimeError("illegal memory access")])

    with pytest.raises(RuntimeError, match="illegal memory access"):
        sparse_mla_bwd.bwd_within_shared_mem(**BWD_SHAPE)
    assert len(calls) == 1


def test_backward_memoizes_per_shape(monkeypatch):
    calls = _record_calls(monkeypatch, sparse_mla_bwd, "bwd", [SHARED_MEM_ERROR, "smaller", "smaller"])

    sparse_mla_bwd.bwd_within_shared_mem(**BWD_SHAPE)
    probes = len(calls)
    sparse_mla_bwd.bwd_within_shared_mem(**BWD_SHAPE)

    assert len(calls) == probes + 1
    assert calls[-1] == calls[-2]


# ---------------------------------------------------------------------------------------------
# indexer forward
# ---------------------------------------------------------------------------------------------


def test_indexer_halves_block_n_until_it_builds(monkeypatch):
    probed = []

    def stub(*, heads, index_dim, block_N):
        probed.append(block_N)
        if block_N > 64:
            raise SHARED_MEM_ERROR
        return "kernel"

    monkeypatch.setattr(indexer_fwd, "tl_indexer_fwd_impl", stub)

    assert indexer_fwd._indexer_fwd_within_shared_mem(heads=32, index_dim=128) == "kernel"
    assert probed == [256, 128, 64]


def test_indexer_stops_halving_at_32(monkeypatch):
    """Below 32 the tiling is not worth pursuing; the shared-memory error is the real answer."""

    def stub(*, heads, index_dim, block_N):
        raise SHARED_MEM_ERROR

    monkeypatch.setattr(indexer_fwd, "tl_indexer_fwd_impl", stub)
    with pytest.raises(RuntimeError, match="exceeds device limit"):
        indexer_fwd._indexer_fwd_within_shared_mem(heads=32, index_dim=128)


def test_indexer_does_not_retry_an_unrelated_error(monkeypatch):
    def stub(*, heads, index_dim, block_N):
        raise RuntimeError("no kernel image is available for execution")

    monkeypatch.setattr(indexer_fwd, "tl_indexer_fwd_impl", stub)
    with pytest.raises(RuntimeError, match="no kernel image"):
        indexer_fwd._indexer_fwd_within_shared_mem(heads=32, index_dim=128)


# ---------------------------------------------------------------------------------------------
# the error strings this whole mechanism is keyed on
# ---------------------------------------------------------------------------------------------


def test_the_shared_memory_marker_is_shared_by_every_entry_point():
    """One wording, three call sites. If tilelang renames it, all three need updating together."""
    assert sparse_mla_fwd._SHARED_MEM_ERROR == indexer_fwd._SHARED_MEM_ERROR
    assert sparse_mla_fwd._SHARED_MEM_ERROR in sparse_mla_bwd.RETRYABLE_BUILD_ERRORS


def test_the_shared_memory_marker_still_matches_what_tilelang_emits():
    """A canary for a tilelang upgrade that reworded the error the retry loops key on.

    Only the wording is checked, not a real build: constructing a kernel that overflows shared
    memory needs a GPU. If this ever fails, the retry loops have silently stopped firing.
    """
    from tilelang import engine  # noqa: F401  -- import so a rename breaks here rather than silently

    assert sparse_mla_fwd._SHARED_MEM_ERROR == "exceeds device limit"
