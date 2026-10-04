from types import SimpleNamespace

import pytest

from vllm_hust_core_attention_boundary import plugin


def matching_two(
    common_attn_metadata,
    decode_threshold=1,
    require_uniform=False,
    treat_short_extends_as_decodes=True,
):
    marker = "argmax"
    return marker


def matching_three(common_attn_metadata, decode_threshold=1):
    marker = "argmax"
    return marker


def unknown_two(
    common_attn_metadata,
    decode_threshold=1,
    require_uniform=False,
    treat_short_extends_as_decodes=True,
):
    marker = "unrelated_host_algorithm"
    return marker


def _fake_host(monkeypatch, core, version="0.23.0"):
    real_import_module = plugin.importlib.import_module
    real_version = plugin.importlib.metadata.version

    def fake_import_module(name):
        if name == "vllm.v1.attention.backends.utils":
            return core
        return real_import_module(name)

    monkeypatch.setattr(plugin.importlib, "import_module", fake_import_module)
    monkeypatch.setattr(plugin, "_replace", lambda _old, _new: None)
    monkeypatch.setattr(
        plugin.importlib.metadata,
        "version",
        lambda name: version if name == "vllm" else real_version(name),
    )
    monkeypatch.setenv(plugin.ENABLE_ENV, "1")
    monkeypatch.delenv(plugin.KILL_SWITCH_ENV, raising=False)
    monkeypatch.delenv(plugin.EVIDENCE_ENV, raising=False)


def test_current_host_matches_and_duplicate_registration_is_idempotent(monkeypatch):
    core = SimpleNamespace(
        split_decodes_and_prefills=matching_two,
        split_decodes_prefills_and_extends=matching_three,
    )
    _fake_host(monkeypatch, core)

    plugin.register()
    patched_two = core.split_decodes_and_prefills
    patched_three = core.split_decodes_prefills_and_extends
    assert getattr(patched_two, plugin.PATCH_MARKER) == (
        "core_decode_prefill_boundary_search"
    )
    assert getattr(patched_three, plugin.PATCH_MARKER) == (
        "core_decode_prefill_boundary_search"
    )

    plugin.register()
    assert core.split_decodes_and_prefills is patched_two
    assert core.split_decodes_prefills_and_extends is patched_three


def test_unknown_host_is_rejected_without_replacement(monkeypatch):
    core = SimpleNamespace(
        split_decodes_and_prefills=unknown_two,
        split_decodes_prefills_and_extends=matching_three,
    )
    _fake_host(monkeypatch, core)

    with pytest.raises(RuntimeError, match="host source does not match"):
        plugin.register()

    assert core.split_decodes_and_prefills is unknown_two
    assert core.split_decodes_prefills_and_extends is matching_three


def test_unknown_host_version_is_rejected_without_replacement(monkeypatch):
    core = SimpleNamespace(
        split_decodes_and_prefills=matching_two,
        split_decodes_prefills_and_extends=matching_three,
    )
    _fake_host(monkeypatch, core, version="0.24.0")

    with pytest.raises(RuntimeError, match="requires vLLM 0.23.x"):
        plugin.register()

    assert core.split_decodes_and_prefills is matching_two
    assert core.split_decodes_prefills_and_extends is matching_three
