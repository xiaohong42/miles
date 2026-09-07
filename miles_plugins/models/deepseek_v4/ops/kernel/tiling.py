"""Choose a kernel tiling the compiler's target can actually host.

The DeepSeek-V4 tilelang kernels ship with tilings tuned for one GPU. On a device with a smaller
shared-memory budget they do not compile at all, and the failure is a hard error at the first
forward pass. The hand-written fallback lists that fix this are per-architecture by construction:
a different head count, a different topk or a different budget needs the list edited. This module
derives the alternatives instead, and asks tilelang how much shared memory each one needs rather
than modelling it.

Two things make that possible, both of which tilelang already knows and neither of which it
advertises as API.

**The budget** is in the target description, which tilelang builds for every compilation:

    hip -max_num_threads=256 -max_shared_memory_per_block=65536 -max_threads_per_block=256
        -mcpu=gfx942 -thread_warp_size=64

**The requirement** is the ``dyn_shared_memory_buf`` attribute on the lowered device function --
the same number tilelang formats into "Requested dynamic shared memory N exceeds device limit M".
Reading it needs lowering but not code generation, which is about 4x cheaper than a full build
(2.0 s against 8.2 s on gfx942), and it is exact: 85488 B for block_H=64/block_I=16 and 68608 B for
block_H=32/block_I=32, matching the error message byte for byte.

Consequences worth stating, since they are what this module buys:

* Nothing here models tilelang's allocator. An earlier attempt did, and got it wrong in a way that
  mattered -- Q_shared and O_shared share storage, and counting both over-estimated by 27% at
  block_H=32 and rejected a tiling that does build.
* No candidate is ever compiled speculatively. The search costs one lowering per candidate and one
  compilation of the winner.
* There is no ``torch.version.hip``, no ``torch.cuda.get_device_properties`` and no
  architecture-string parsing below. A target tilelang can compile for is one this search can plan
  for.

The cost is a dependency on three tilelang internals: ``JITImpl.get_tir``, ``engine.lower`` with
device compilation disabled, and the ``dyn_shared_memory_buf`` attribute name. That is a deliberate
trade against the alternative, which is matching three error strings: an attribute that disappears
raises AttributeError on the spot, whereas a reworded error string makes the whole mechanism stop
firing silently. Both couplings are pinned by tests. The right long-term home for this is tilelang
itself, which computes the number and then throws it away into a formatted string.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from typing import Any, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")

# The attribute tilelang leaves on the lowered device function. Named here so a tilelang rename
# fails loudly in one place instead of being mistaken for "this tiling needs no shared memory".
SHARED_MEMORY_ATTR = "dyn_shared_memory_buf"

# ---------------------------------------------------------------------------------------------
# What "this candidate cannot be built" looks like
#
# Even with the requirement known exactly, a candidate can still be unbuildable for reasons that
# have nothing to do with size: tilelang rejects one in three unrelated places, and all three
# surface while lowering. Matching the text couples this to the tilelang version, which is why
# tests check the one wording that has an importable source.
# ---------------------------------------------------------------------------------------------
SHARED_MEMORY_EXCEEDED = "exceeds device limit"  # the shared-memory check at codegen
WARP_PARTITION_DEGENERATE = "Divide by zero"  # warp partitioning inside T.gemm
VECTORIZE_FAILED = "is_scalar"  # the vectorizer on the atomic store loop

RETRYABLE_BUILD_ERRORS = (SHARED_MEMORY_EXCEEDED, WARP_PARTITION_DEGENERATE, VECTORIZE_FAILED)

# Assertions inside the kernel builders that mean "this candidate does not apply" rather than "the
# caller made a mistake". Only the head-block one qualifies: a candidate whose block_H does not
# divide the head count is simply inapplicable. Its siblings -- a dim that is not a power of two, a
# topk that does not divide block_I -- are caller errors, and retrying them would bury the real
# message under a pile of later failures.
RETRYABLE_ASSERTIONS = ("block_H",)


def is_retryable_build_error(exc: BaseException) -> bool:
    """Is this "try a different tiling" rather than "the caller or the kernel is wrong"?"""
    message = str(exc)
    if isinstance(exc, AssertionError):
        return any(marker in message for marker in RETRYABLE_ASSERTIONS)
    return any(marker in message for marker in RETRYABLE_BUILD_ERRORS)


# ---------------------------------------------------------------------------------------------
# Device limits, read from the compiler's target
# ---------------------------------------------------------------------------------------------

# The smallest N a single matrix-core instruction covers, which bounds how many warps a gemm's
# output can be split across: with fewer than this many columns per warp, tilelang's warp
# partitioning divides by zero. This is the one number the target does not carry, so it is keyed on
# the target kind:
#   hip   AMD MFMA is 16x16x16, so 16.
#   cuda  Tensor Core MMA is m16n8k16, so 8.
# A target that is not listed gets the smaller value, which only ever means more warps are
# considered than the hardware can use -- those candidates fail to lower and the search moves on.
_MIN_GEMM_N_PER_WARP = {"hip": 16, "cuda": 8}
_MIN_GEMM_N_PER_WARP_DEFAULT = 8

# The floor on the head block, which is a different quantity from the one above and must not be
# derived from it: the head block is the M dimension of the QK^T gemm, while min_gemm_n_per_warp
# constrains N. 16 is the kernels' own floor, from `padded_H = max(next_power_of_2(H), 16)` -- below
# it there is no head count left to block.
MIN_HEAD_BLOCK = 16


@dataclass(frozen=True)
class DeviceLimits:
    """The subset of a tilelang target that constrains a tiling."""

    shared_memory_per_block: int
    warp_size: int
    max_threads_per_block: int
    min_gemm_n_per_warp: int

    @classmethod
    def from_target(cls, target: Any) -> DeviceLimits:
        attrs = target.attrs
        return cls(
            shared_memory_per_block=int(attrs["max_shared_memory_per_block"]),
            warp_size=int(attrs["thread_warp_size"]),
            max_threads_per_block=int(attrs.get("max_num_threads", attrs["max_threads_per_block"])),
            min_gemm_n_per_warp=_MIN_GEMM_N_PER_WARP.get(target.kind.name, _MIN_GEMM_N_PER_WARP_DEFAULT),
        )

    def threads_for(self, gemm_n: int) -> int:
        """The widest thread block whose warps each still get a full matrix-core tile.

        Measured on gfx942 (warp_size 64, min_gemm_n_per_warp 16), sweeping block_I against threads:
        every combination with ``block_I / warps >= 16`` builds and every one below it fails with
        "Divide by zero". The tilings that shipped used exactly this expression -- block_I 64, 32,
        16 give 256, 128, 64 -- so those thread counts were never free parameters.
        """
        warps = max(1, gemm_n // self.min_gemm_n_per_warp)
        return min(warps * self.warp_size, self.max_threads_per_block)


def current_target() -> Any:
    from tilelang.utils.target import determine_target

    return determine_target("auto", return_object=True)


def shared_memory_required(prim_func: Any, target: Any) -> int:
    """How much dynamic shared memory tilelang's lowered kernel asks for, in bytes.

    Lowers without generating or compiling device code, so this is much cheaper than a build while
    giving the exact figure the build would have checked. Zero when the kernel declares no dynamic
    shared memory at all.
    """
    from tilelang import engine

    with target:
        lowered = engine.lower(prim_func, target=target, enable_host_codegen=False, enable_device_compile=False)
    return largest_declared_shared_memory(lowered)


def largest_declared_shared_memory(lowered: Any) -> int:
    """Read the requirement off a lowered module. Separate so it is testable without a GPU.

    Zero when no function declares any, which is a kernel that needs no dynamic shared memory
    rather than a failure to find out.
    """
    device_mod = getattr(lowered, "device_mod", lowered)
    return max(
        (
            int(func.attrs[SHARED_MEMORY_ATTR])
            for _, func in device_mod.functions.items()
            if func.attrs is not None and SHARED_MEMORY_ATTR in func.attrs
        ),
        default=0,
    )


def halvings(start: int, floor: int) -> Iterator[int]:
    """``start``, then repeated halvings, down to and including ``floor``."""
    value = start
    while value >= floor:
        yield value
        value //= 2


# ---------------------------------------------------------------------------------------------
# The search
# ---------------------------------------------------------------------------------------------


def build_with_largest_fitting_tiling(
    *,
    requested: T,
    derived: Iterable[T],
    compile_tiling: Callable[[T], Any],
    required_bytes: Callable[[T], int],
    budget: int,
    describe: Callable[[T], str],
    what: str,
) -> tuple[Any, T]:
    """Compile the requested tiling; if it will not build, compile the largest one that fits.

    The requested tiling is attempted optimistically, so a device that fits it pays nothing at all
    for this machinery -- not a wasted compilation and not even a lowering. Only once it has been
    refused does the search start asking about alternatives, and then it compiles exactly one.

    If nothing fits, the *requested* tiling's rejection is raised rather than the last candidate's:
    it names the shape the caller asked for and the budget it exceeded, which is the useful message.
    """
    try:
        return compile_tiling(requested), requested
    except Exception as exc:
        if not is_retryable_build_error(exc):
            raise
        # Carrying the exception out of its `except` block keeps its __traceback__, and through the
        # frame chain every local of every caller -- during a model forward, its activation tensors.
        # The cycle that forms is only breakable by the cyclic collector, so those tensors sit on
        # the device until something triggers a full gc pass. The message is all the re-raise needs.
        requested_error = exc.with_traceback(None)

    for candidate in derived:
        try:
            required = required_bytes(candidate)
        except Exception as exc:
            if not is_retryable_build_error(exc):
                raise
            continue  # cannot be lowered at all, for one of the two non-size reasons
        if required > budget:
            continue
        logger.warning(
            "[%s] shared memory forced a smaller tiling on this GPU: %s (needs %d B of %d)",
            what,
            describe(candidate),
            required,
            budget,
        )
        return compile_tiling(candidate), candidate

    raise requested_error


# ---------------------------------------------------------------------------------------------
# Candidate orders
#
# The shrink order is the same everywhere and is a statement about cost, not about any one GPU:
# give up pipeline depth first (it costs latency hiding only, and the kernels already clamp it),
# then shrink the KV/index block (more inner iterations), and only then shrink the head block (more
# grid blocks, and it caps how much of the output one workgroup can accumulate).
# ---------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class ForwardTiling:
    num_stages: int
    block_I: int
    threads: int
    block_H: int | None  # None keeps the kernel's own "one block of padded_H"

    def describe(self) -> str:
        return f"num_stages={self.num_stages} block_I={self.block_I} threads={self.threads} block_H={self.block_H}"


def sparse_mla_forward_tilings(
    *, padded_heads: int, topk: int, block_I: int, limits: DeviceLimits
) -> Iterator[ForwardTiling]:
    """Progressively smaller forward tilings, largest first. Excludes the requested one."""
    for head_block in halvings(padded_heads, MIN_HEAD_BLOCK):
        for index_block in halvings(block_I, limits.min_gemm_n_per_warp):
            if topk % index_block:
                continue  # would trip sparse_mqa_fwd's own divisibility assert
            yield ForwardTiling(
                num_stages=1,
                block_I=index_block,
                threads=limits.threads_for(index_block),
                # The kernel's own default already blocks at padded_heads, so say so rather than
                # restating it: that keeps a device which needs no shrinking on the exact code path
                # it had before this module existed.
                block_H=None if head_block == padded_heads else head_block,
            )


@dataclass(frozen=True)
class BackwardTiling:
    block_size: int
    threads: int
    split_store: int
    stage_dq_through_shared: bool
    max_block_H: int | None  # None keeps the kernel's own choice

    def describe(self) -> str:
        return (
            f"block_size={self.block_size} threads={self.threads} split_store={self.split_store} "
            f"stage_dq_through_shared={self.stage_dq_through_shared} max_block_H={self.max_block_H}"
        )


def sparse_mla_backward_tilings(
    *, padded_heads: int, topk: int, block_size: int, limits: DeviceLimits
) -> Iterator[BackwardTiling]:
    """Progressively smaller backward tilings, largest first. Excludes the requested one.

    Two of the levers here are not tile sizes. ``split_store`` sets the height of
    ``acc_dkv_shared``, an fp32 buffer, so raising it to ``block_size`` reduces that buffer to a
    single row at the cost of one atomic store pass per row. ``stage_dq_through_shared`` removes a
    whole ``[block_H, D]`` buffer by writing dQ straight to global memory. Both are tried only
    after the tile sizes are already at their smallest, because both cost bandwidth rather than
    parallelism.
    """
    for head_block in halvings(padded_heads, MIN_HEAD_BLOCK):
        for kv_block in halvings(block_size, limits.min_gemm_n_per_warp):
            if topk % kv_block:
                continue
            for stage_dq in (True, False):
                for split_store in dict.fromkeys((2, kv_block)):  # deduped, order preserved
                    if kv_block % split_store:
                        continue
                    yield BackwardTiling(
                        block_size=kv_block,
                        threads=limits.threads_for(kv_block),
                        split_store=split_store,
                        stage_dq_through_shared=stage_dq,
                        max_block_H=head_block,
                    )


def indexer_forward_block_ns(*, block_N: int, limits: DeviceLimits) -> Iterator[int]:
    """Halvings of the requested block_N. Excludes the requested one.

    block_N only sets how much of the KV axis one workgroup sweeps per iteration, so shrinking it
    costs performance and nothing else.
    """
    yield from halvings(block_N // 2, limits.min_gemm_n_per_warp)
