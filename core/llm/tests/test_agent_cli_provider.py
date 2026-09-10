from __future__ import annotations

import pytest

from core.llm.config import ModelConfig
from core.llm.providers import AgentCLILLMProvider, create_provider


def _config(provider: str = "codexcli") -> ModelConfig:
    return ModelConfig(
        provider=provider,
        model_name="session-default",
        timeout=42,
        max_context=1000000,
    )


def test_factory_builds_agent_cli_providers() -> None:
    assert isinstance(create_provider(_config()), AgentCLILLMProvider)
    provider = create_provider(_config("opencodecli"))
    assert isinstance(provider, AgentCLILLMProvider)
    assert provider.agent == "opencode"


def test_generate_uses_selected_cli(monkeypatch) -> None:
    calls = []

    def fake(agent, prompt, **kwargs):
        calls.append((agent, prompt, kwargs))
        return "answer", None, 0.25

    monkeypatch.setattr("core.llm.agent_cli_adapter.run_agent_cli", fake)
    response = create_provider(_config()).generate("question", "system")
    assert response.content == "answer"
    assert response.provider == "codexcli"
    assert response.cost == 0.0
    assert calls == [(
        "codex",
        "question",
        {
            "system_prompt": "system",
            "model": "session-default",
            "timeout_s": 42,
        },
    )]


def test_generate_structured_uses_cli_schema(monkeypatch) -> None:
    schema = {
        "type": "object",
        "properties": {"ok": {"type": "boolean"}},
        "required": ["ok"],
        "additionalProperties": False,
    }

    def fake(agent, prompt, **kwargs):
        assert kwargs["schema"] == schema
        return '{"ok":true}', {"ok": True}, 0.1

    monkeypatch.setattr("core.llm.agent_cli_adapter.run_agent_cli", fake)
    response = create_provider(_config()).generate_structured("q", schema)
    assert response.result == {"ok": True}
    assert response.provider == "codexcli"


def test_generate_structured_normalizes_descriptive_schema(monkeypatch) -> None:
    schema = {"ok": "boolean", "reason": "string optional"}

    def fake(agent, prompt, **kwargs):
        assert kwargs["schema"] == {
            "type": "object",
            "properties": {
                "ok": {"type": "boolean"},
                "reason": {"type": ["string", "null"]},
            },
            "required": ["ok", "reason"],
            "additionalProperties": False,
        }
        return '{"ok":true}', {"ok": True}, 0.1

    monkeypatch.setattr("core.llm.agent_cli_adapter.run_agent_cli", fake)
    response = create_provider(_config()).generate_structured("q", schema)
    assert response.result == {"ok": True}


def test_codex_schema_strictness_is_recursive() -> None:
    from core.llm.providers import _codex_strict_schema

    converted = _codex_strict_schema({
        "properties": {
            "items": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "verdict": {"type": "string"},
                        "detail": {"type": "string"},
                    },
                    "required": ["verdict"],
                },
            },
        },
        "required": ["items"],
    })
    nested = converted["properties"]["items"]["items"]
    assert converted["type"] == "object"
    assert converted["additionalProperties"] is False
    assert nested["additionalProperties"] is False
    assert nested["required"] == ["verdict", "detail"]
    assert nested["properties"]["detail"]["type"] == ["string", "null"]


def test_opencode_descriptive_schema_has_object_root(monkeypatch) -> None:
    def fake(agent, prompt, **kwargs):
        assert agent == "opencode"
        assert kwargs["schema"] == {
            "type": "object",
            "properties": {"ok": {"type": "boolean"}},
            "required": ["ok"],
        }
        return '{"ok":true}', {"ok": True}, 0.1

    monkeypatch.setattr("core.llm.agent_cli_adapter.run_agent_cli", fake)
    response = create_provider(_config("opencodecli")).generate_structured(
        "q", {"ok": "boolean"}
    )
    assert response.result == {"ok": True}


@pytest.mark.parametrize(
    "agent,provider", [
        ("claude", "claudecode"),
        ("codex", "codexcli"),
        ("opencode", "opencodecli"),
    ],
)
def test_api_configuration_beats_host_subscription_fallback(
    monkeypatch, agent, provider,
) -> None:
    monkeypatch.setenv("RAPTOR_AGENT", agent)
    monkeypatch.setenv("OPENAI_API_KEY", "must-not-win")
    monkeypatch.setattr("shutil.which", lambda name: f"/usr/bin/{name}")
    from core.llm.config import _get_default_primary_model

    config = _get_default_primary_model()
    assert config is not None
    assert config.provider == "openai"
    assert config.api_key == "must-not-win"


def test_models_json_configuration_beats_host_subscription_fallback(
    monkeypatch,
) -> None:
    monkeypatch.setenv("RAPTOR_AGENT", "codex")
    configured = ModelConfig(
        provider="anthropic", model_name="claude-opus-4-6",
        api_key="configured-key", role="analysis",
    )
    monkeypatch.setattr(
        "core.llm.config._get_best_thinking_model", lambda: configured,
    )
    from core.llm.config import _get_default_primary_model

    assert _get_default_primary_model() is configured


@pytest.mark.parametrize("agent", ["claude", "codex", "opencode"])
def test_host_subscription_is_used_without_external_analysis_config(
    monkeypatch, agent,
) -> None:
    monkeypatch.setenv("RAPTOR_AGENT", agent)
    monkeypatch.setattr("shutil.which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr("core.llm.config._get_best_thinking_model", lambda: None)
    monkeypatch.setattr("core.llm.config._config_bedrock_primary", lambda: None)
    monkeypatch.setattr("core.llm.config._get_available_ollama_models", lambda: [])
    from core.llm.config import _get_default_primary_model

    config = _get_default_primary_model()
    assert config is not None
    expected = {
        "claude": "claudecode",
        "codex": "codexcli",
        "opencode": "opencodecli",
    }
    assert config.provider == expected[agent]


@pytest.mark.parametrize(
    "marker,agent,provider", [
        ("CLAUDECODE", "claude", "claudecode"),
        ("CODEX_THREAD_ID", "codex", "codexcli"),
        ("OPENCODE_SESSION_ID", "opencode", "opencodecli"),
    ],
)
def test_direct_session_marker_binds_inference(
    monkeypatch, marker, agent, provider,
) -> None:
    monkeypatch.delenv("RAPTOR_AGENT", raising=False)
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    for name in (
        "CLAUDECODE", "CODEX_SESSION_ID", "CODEX_THREAD_ID",
        "OPENCODE_SESSION_ID",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv(marker, "session")
    monkeypatch.setattr("shutil.which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr("core.llm.config._get_best_thinking_model", lambda: None)
    monkeypatch.setattr("core.llm.config._config_bedrock_primary", lambda: None)
    monkeypatch.setattr("core.llm.config._get_available_ollama_models", lambda: [])

    from core.llm.config import _get_default_primary_model

    config = _get_default_primary_model()
    assert config is not None
    assert config.provider == provider
