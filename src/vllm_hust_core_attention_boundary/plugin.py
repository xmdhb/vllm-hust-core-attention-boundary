"""Plugin entry point for the independent Core boundary package."""

from __future__ import annotations

import importlib
import importlib.metadata
import inspect
import os
import sys
import threading
from collections.abc import Callable
from typing import Any

ENABLE_ENV = "VLLM_HUST_CORE_ATTENTION_BOUNDARY_ENABLE"
KILL_SWITCH_ENV = "VLLM_HUST_CORE_ATTENTION_BOUNDARY_KILL_SWITCH"
EVIDENCE_ENV = "VLLM_HUST_CORE_ATTENTION_BOUNDARY_EVIDENCE"
PATCH_MARKER = "__vllm_hust_core_attention_boundary__"
_seen: set[str] = set()
_lock = threading.Lock()


def _enabled(value: str | None) -> bool:
    return (value or "").lower() in {"1", "true", "yes", "on"}


def _evidence(event: str, mechanism: str) -> None:
    if not _enabled(os.getenv(EVIDENCE_ENV)):
        return
    text = f"{event} mechanism={mechanism}"
    with _lock:
        if text in _seen:
            return
        _seen.add(text)
    print(f"LEGACY017_EVIDENCE {text}", file=sys.stderr, flush=True)


def _replace(old: Callable[..., Any], new: Callable[..., Any]) -> None:
    for module in tuple(sys.modules.values()):
        if module is None:
            continue
        try:
            namespace = vars(module)
        except TypeError:
            continue
        for name, value in tuple(namespace.items()):
            if value is old:
                try:
                    setattr(module, name, new)
                except (AttributeError, TypeError):
                    pass


def _check_host(module: Any, name: str) -> None:
    function = getattr(module, name, None)
    if not callable(function):
        raise RuntimeError(f"Core boundary host function is missing: {name}")
    if getattr(function, PATCH_MARKER, False):
        return
    source = inspect.getsource(function)
    if not any(
        anchor in source
        for anchor in (
            "argmax",
            "_split_decode_prefill_boundary",
            "_split_decode_extend_prefill_boundary",
        )
    ):
        raise RuntimeError(f"Core boundary host source does not match: {name}")


def register() -> None:
    if _enabled(os.getenv(KILL_SWITCH_ENV)) or not _enabled(os.getenv(ENABLE_ENV)):
        return
    version = importlib.metadata.version("vllm")
    if not version.startswith("0.23."):
        raise RuntimeError(f"Core boundary requires vLLM 0.23.x, got {version}")
    core = importlib.import_module("vllm.v1.attention.backends.utils")
    _check_host(core, "split_decodes_prefills_and_extends")
    _check_host(core, "split_decodes_and_prefills")
    boundary = importlib.import_module(f"{__package__}.boundary")
    boundary.install(core, _replace, _evidence, PATCH_MARKER)
    _evidence("installed", "core_decode_prefill_boundary_search")
