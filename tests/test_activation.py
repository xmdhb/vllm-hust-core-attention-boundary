import sys

from vllm_hust_core_attention_boundary import plugin


def test_default_disabled_does_not_import_vllm(monkeypatch):
    monkeypatch.delenv("VLLM_HUST_CORE_ATTENTION_BOUNDARY_ENABLE", raising=False)
    monkeypatch.delenv("VLLM_HUST_CORE_ATTENTION_BOUNDARY_KILL_SWITCH", raising=False)
    plugin.register()
    assert "vllm" not in sys.modules
    assert "torch" not in sys.modules
    assert "vllm_hust_core_attention_boundary.boundary" not in sys.modules


def test_kill_switch_blocks_install(monkeypatch):
    monkeypatch.setenv(plugin.ENABLE_ENV, "1")
    monkeypatch.setenv(plugin.KILL_SWITCH_ENV, "1")

    def unexpected(*args, **kwargs):
        raise AssertionError("kill switch must stop before imports")

    monkeypatch.setattr(plugin.importlib.metadata, "version", unexpected)
    monkeypatch.setattr(plugin.importlib, "import_module", unexpected)
    plugin.register()


def test_evidence_is_emitted_once(monkeypatch, capsys):
    monkeypatch.setenv(plugin.EVIDENCE_ENV, "1")
    plugin._evidence("unit_test_runtime_effective", "unit_test")
    plugin._evidence("unit_test_runtime_effective", "unit_test")
    assert capsys.readouterr().err.splitlines() == [
        "LEGACY017_EVIDENCE unit_test_runtime_effective mechanism=unit_test"
    ]
