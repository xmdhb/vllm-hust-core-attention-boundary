import itertools

import torch

from vllm_hust_op02_core_attention_boundary.boundary import (
    split_decode_extend_prefill_boundary,
    split_decode_prefill_boundary,
)


def test_two_way_boundary():
    for query_lens in itertools.product(range(1, 5), repeat=3):
        starts = torch.tensor([0, *torch.cumsum(torch.tensor(query_lens), 0).tolist()])
        first = next((i for i, n in enumerate(query_lens) if n > 1), 3)
        assert split_decode_prefill_boundary(
            starts, 3, int(starts[-1]), max(query_lens)
        ) == (first, 3 - first, int(starts[first]), int(starts[-1] - starts[first]))


def test_three_way_boundary():
    assert split_decode_extend_prefill_boundary(
        torch.tensor([0, 1, 3, 9]), torch.tensor([1, 15, 6]), 3, 9, 6
    ) == (1, 1, 1, 1, 2, 6)
