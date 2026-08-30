"""The empty-response-chunk branch has to return the same dtype as the non-empty one.

Under context parallelism a rank whose slice contains no response token gets an empty chunk here,
and `allgather_cp_redistribute` then all-reduces the concatenated per-sample tensors across the CP
group. A collective whose ranks disagree on dtype does not raise, it hangs, so the dtype has to be
pinned rather than inferred from the input.
"""

import pytest
import torch

from miles.backends.training_utils.loss_hub.math_utils import calculate_log_probs_and_entropy


@pytest.mark.parametrize("logits_dtype", [torch.float32, torch.bfloat16])
def test_empty_chunk_log_prob_is_fp32(logits_dtype):
    logits = torch.empty(0, 32, dtype=logits_dtype)
    tokens = torch.empty(0, dtype=torch.long)

    log_prob, entropy = calculate_log_probs_and_entropy(logits, tokens, None, with_entropy=True)

    # fp32 is the contract because the non-empty branch upcasts before computing.
    assert log_prob.dtype is torch.float32
    assert entropy.dtype is torch.float32
    assert log_prob.numel() == 0


def test_empty_chunk_dtype_does_not_depend_on_the_logits_dtype():
    # The invariant the CP all-reduce needs: two ranks fed different logit dtypes must still
    # arrive at the collective with the same dtype.
    from_fp32, _ = calculate_log_probs_and_entropy(
        torch.empty(0, 32, dtype=torch.float32), torch.empty(0, dtype=torch.long), None
    )
    from_bf16, _ = calculate_log_probs_and_entropy(
        torch.empty(0, 32, dtype=torch.bfloat16), torch.empty(0, dtype=torch.long), None
    )

    assert from_fp32.dtype is from_bf16.dtype
