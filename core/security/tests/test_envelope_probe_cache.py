from types import SimpleNamespace
from unittest.mock import patch

from core.security.envelope_probe_cache import has_cached_success, store_success
from core.security.prompt_defense_profiles import OPENAI_GPT


def _model(name: str = "gpt-5") -> SimpleNamespace:
    return SimpleNamespace(provider="codexcli", model_name=name)


def _version_result():
    return SimpleNamespace(returncode=0, stdout="codex 1.2.3\n", stderr="")


def test_explicit_model_success_is_cached_and_expires(tmp_path, monkeypatch):
    binary = tmp_path / "codex"
    binary.write_text("binary")
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))

    with patch("core.llm.agent_cli_adapter.resolve_agent_cli",
               return_value=str(binary)), \
         patch("subprocess.run", return_value=_version_result()):
        store_success(_model(), OPENAI_GPT, now=100)
        assert has_cached_success(_model(), OPENAI_GPT, now=101)
        assert not has_cached_success(_model(), OPENAI_GPT, now=100 + 86401)


def test_session_default_is_never_cached(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))

    store_success(_model("session-default"), OPENAI_GPT, now=100)

    assert not has_cached_success(
        _model("session-default"), OPENAI_GPT, now=101,
    )
    assert not (tmp_path / "cache" / "raptor" / "envelope-probes.json").exists()


def test_cli_version_change_invalidates_cache(tmp_path, monkeypatch):
    binary = tmp_path / "codex"
    binary.write_text("binary")
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))

    with patch("core.llm.agent_cli_adapter.resolve_agent_cli",
               return_value=str(binary)), \
         patch("subprocess.run", return_value=_version_result()):
        store_success(_model(), OPENAI_GPT, now=100)

    changed = SimpleNamespace(returncode=0, stdout="codex 1.2.4\n", stderr="")
    with patch("core.llm.agent_cli_adapter.resolve_agent_cli",
               return_value=str(binary)), \
         patch("subprocess.run", return_value=changed):
        assert not has_cached_success(_model(), OPENAI_GPT, now=101)


def test_routing_config_change_invalidates_cache(tmp_path, monkeypatch):
    binary = tmp_path / "codex"
    binary.write_text("binary")
    codex_home = tmp_path / "codex-home"
    codex_home.mkdir()
    config = codex_home / "config.toml"
    config.write_text('model = "alias-a"\n')
    monkeypatch.setenv("CODEX_HOME", str(codex_home))
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))

    with patch("core.llm.agent_cli_adapter.resolve_agent_cli",
               return_value=str(binary)), \
         patch("subprocess.run", return_value=_version_result()):
        store_success(_model(), OPENAI_GPT, now=100)
        assert has_cached_success(_model(), OPENAI_GPT, now=101)
        config.write_text('model = "alias-b"\n')
        assert not has_cached_success(_model(), OPENAI_GPT, now=101)


def test_routing_environment_change_invalidates_cache(tmp_path, monkeypatch):
    binary = tmp_path / "codex"
    binary.write_text("binary")
    monkeypatch.setenv("CODEX_HOME", str(tmp_path / "missing-home"))
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    monkeypatch.setenv("OPENAI_BASE_URL", "https://first.example")

    with patch("core.llm.agent_cli_adapter.resolve_agent_cli",
               return_value=str(binary)), \
         patch("subprocess.run", return_value=_version_result()):
        store_success(_model(), OPENAI_GPT, now=100)
        assert has_cached_success(_model(), OPENAI_GPT, now=101)
        monkeypatch.setenv("OPENAI_BASE_URL", "https://second.example")
        assert not has_cached_success(_model(), OPENAI_GPT, now=101)


def test_unsafe_routing_config_disables_cache(tmp_path, monkeypatch):
    binary = tmp_path / "codex"
    binary.write_text("binary")
    codex_home = tmp_path / "codex-home"
    codex_home.mkdir()
    target = tmp_path / "real-config.toml"
    target.write_text('model = "alias"\n')
    (codex_home / "config.toml").symlink_to(target)
    monkeypatch.setenv("CODEX_HOME", str(codex_home))
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))

    with patch("core.llm.agent_cli_adapter.resolve_agent_cli",
               return_value=str(binary)), \
         patch("subprocess.run", return_value=_version_result()):
        store_success(_model(), OPENAI_GPT, now=100)
        assert not has_cached_success(_model(), OPENAI_GPT, now=101)
    assert not (tmp_path / "cache" / "raptor" / "envelope-probes.json").exists()
