"""Core #41 decode/prefill boundary implementation."""

from __future__ import annotations

from collections.abc import Callable
from types import ModuleType
from typing import Any


def _first_true_index(mask: Any) -> int:
    try:
        size = int(mask.numel())
    except AttributeError:
        size = len(mask)
    if size == 0:
        return 0

    for index in range(size):
        value = mask[index]
        item = getattr(value, "item", None)
        if bool(item() if callable(item) else value):
            return index
    return size


def split_decode_prefill_boundary(
    query_start_loc: torch.Tensor,
    num_reqs: int,
    num_tokens: int,
    max_query_len: int,
    decode_threshold: int = 1,
    *,
    query_lens: torch.Tensor | None = None,
    require_uniform: bool = False,
    is_prefilling: torch.Tensor | None = None,
    treat_short_extends_as_decodes: bool = True,
) -> tuple[int, int, int, int]:
    import torch

    if num_reqs == 0:
        return 0, 0, 0, 0
    if (
        max_query_len <= decode_threshold
        and (not require_uniform or decode_threshold <= 1)
        and treat_short_extends_as_decodes
    ):
        return num_reqs, 0, num_tokens, 0

    query_start_loc = query_start_loc[: num_reqs + 1]
    if query_lens is None:
        query_lens = torch.diff(query_start_loc)
    else:
        query_lens = query_lens[:num_reqs]
        if query_lens.device != query_start_loc.device:
            query_lens = query_lens.to(query_start_loc.device)

    if require_uniform:
        first_query_len = query_lens[0]
        uniform_or_pad = (query_lens == first_query_len) | (query_lens == 0)
        force_all_decode = torch.all(uniform_or_pad) & (
            first_query_len <= decode_threshold
        )
        is_prefill = query_lens != first_query_len
        is_prefill = torch.where(
            force_all_decode, torch.zeros_like(is_prefill), is_prefill
        )
        is_prefill = torch.where(
            first_query_len > decode_threshold,
            torch.ones_like(is_prefill),
            is_prefill,
        )
    else:
        is_prefill = query_lens > decode_threshold

    if not treat_short_extends_as_decodes:
        if is_prefilling is None:
            raise AssertionError("is_prefilling metadata is required")
        is_prefilling = is_prefilling[:num_reqs].to(query_start_loc.device)
        if is_prefilling.shape[0] < num_reqs:
            import torch.nn.functional as F

            is_prefilling = F.pad(
                is_prefilling,
                (0, num_reqs - is_prefilling.shape[0]),
                value=False,
            )
        is_prefill |= is_prefilling

    first_prefill = torch.tensor(
        _first_true_index(is_prefill),
        dtype=torch.int64,
        device=query_start_loc.device,
    )
    num_reqs_t = first_prefill.new_tensor(num_reqs)
    num_tokens_t = torch.tensor(
        num_tokens, dtype=torch.int64, device=query_start_loc.device
    )
    num_decodes = first_prefill
    num_prefills = num_reqs_t - num_decodes
    num_decode_tokens = torch.where(
        first_prefill < num_reqs_t,
        query_start_loc[first_prefill].to(torch.int64),
        num_tokens_t,
    )
    num_prefill_tokens = num_tokens_t - num_decode_tokens
    result = torch.stack(
        [num_decodes, num_prefills, num_decode_tokens, num_prefill_tokens]
    ).to(torch.int64).cpu().tolist()
    return tuple(result)


def split_decode_extend_prefill_boundary(
    query_start_loc: torch.Tensor,
    seq_lens: torch.Tensor,
    num_reqs: int,
    num_tokens: int,
    max_query_len: int,
    decode_threshold: int = 1,
) -> tuple[int, int, int, int, int, int]:
    import torch

    if num_reqs == 0:
        return 0, 0, 0, 0, 0, 0
    if max_query_len <= decode_threshold:
        return num_reqs, 0, 0, num_tokens, 0, 0

    query_start_loc = query_start_loc[: num_reqs + 1]
    query_lens = torch.diff(query_start_loc)
    seq_lens = seq_lens[:num_reqs]
    if seq_lens.device != query_start_loc.device:
        seq_lens = seq_lens.to(query_start_loc.device)

    is_extend = query_lens > decode_threshold
    is_prefill = (seq_lens == query_lens) & is_extend
    first_extend = torch.tensor(
        _first_true_index(is_extend),
        dtype=torch.int64,
        device=query_start_loc.device,
    )
    first_prefill = torch.tensor(
        _first_true_index(is_prefill),
        dtype=torch.int64,
        device=query_start_loc.device,
    )
    reqs = first_extend.new_tensor(num_reqs)
    tokens = torch.tensor(num_tokens, dtype=torch.int64, device=query_start_loc.device)
    decode_tokens = torch.where(
        first_extend < reqs, query_start_loc[first_extend].to(torch.int64), tokens
    )
    prefill_start = torch.where(
        first_prefill < reqs, query_start_loc[first_prefill].to(torch.int64), tokens
    )
    result = torch.stack(
        [
            first_extend,
            first_prefill - first_extend,
            reqs - first_prefill,
            decode_tokens,
            prefill_start - decode_tokens,
            tokens - prefill_start,
        ]
    ).to(torch.int64).cpu().tolist()
    return tuple(result)


def install(
    core: ModuleType,
    replace_imported_references: Callable,
    evidence: Callable[[str, str], None],
    marker: str,
) -> None:
    old_three = core.split_decodes_prefills_and_extends
    old_two = core.split_decodes_and_prefills

    if not getattr(old_three, marker, False):
        def patched_three(metadata: Any, decode_threshold: int = 1):
            evidence("runtime_effective", "core_decode_prefill_boundary_search")
            assert metadata.seq_lens_cpu_upper_bound is not None
            return split_decode_extend_prefill_boundary(
                metadata.query_start_loc_cpu,
                metadata.seq_lens_cpu_upper_bound,
                metadata.num_reqs,
                metadata.num_actual_tokens,
                metadata.max_query_len,
                decode_threshold,
            )

        setattr(patched_three, marker, "core_decode_prefill_boundary_search")
        core.split_decodes_prefills_and_extends = patched_three
        replace_imported_references(old_three, patched_three)

    if not getattr(old_two, marker, False):
        def patched_two(
            metadata: Any,
            decode_threshold: int = 1,
            require_uniform: bool = False,
            treat_short_extends_as_decodes: bool = True,
        ):
            evidence("runtime_effective", "core_decode_prefill_boundary_search")
            return split_decode_prefill_boundary(
                metadata.query_start_loc_cpu,
                metadata.num_reqs,
                metadata.num_actual_tokens,
                metadata.max_query_len,
                decode_threshold,
                is_prefilling=metadata.is_prefilling,
                require_uniform=require_uniform,
                treat_short_extends_as_decodes=treat_short_extends_as_decodes,
            )

        setattr(patched_two, marker, "core_decode_prefill_boundary_search")
        core.split_decodes_and_prefills = patched_two
        replace_imported_references(old_two, patched_two)


__all__ = [
    "_first_true_index",
    "split_decode_prefill_boundary",
    "split_decode_extend_prefill_boundary",
    "install",
]
