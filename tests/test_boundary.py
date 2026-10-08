import itertools

def test_two_way_boundary():
    import torch
    from vllm_hust_core_attention_boundary.boundary import split_decode_prefill_boundary

    for query_lens in itertools.product(range(1, 5), repeat=3):
        starts = torch.tensor([0, *torch.cumsum(torch.tensor(query_lens), 0).tolist()])
        first = next((i for i, n in enumerate(query_lens) if n > 1), 3)
        assert split_decode_prefill_boundary(
            starts, 3, int(starts[-1]), max(query_lens)
        ) == (first, 3 - first, int(starts[first]), int(starts[-1] - starts[first]))


def test_three_way_boundary():
    import torch
    from vllm_hust_core_attention_boundary.boundary import (
        split_decode_extend_prefill_boundary,
    )

    assert split_decode_extend_prefill_boundary(
        torch.tensor([0, 1, 3, 9]), torch.tensor([1, 15, 6]), 3, 9, 6
    ) == (1, 1, 1, 1, 2, 6)


def test_first_true_index_without_torch():
    from vllm_hust_core_attention_boundary.boundary import _first_true_index

    assert _first_true_index([]) == 0
    assert _first_true_index([False, False, True]) == 2
    assert _first_true_index([False, False]) == 2
