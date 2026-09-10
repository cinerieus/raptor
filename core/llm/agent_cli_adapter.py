"""Subscription-backed coding-agent CLI inference transports."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import threading
import time
from pathlib import Path
from typing import Any

from core.run.scratch import scratch_dir


SUPPORTED_AGENT_CLIS = frozenset({"codex", "opencode"})
DEFAULT_MAX_CALLS = 100
_CALL_LOCK = threading.Lock()
_CALL_COUNT = 0
_AUTH_LOCK = threading.Lock()
_AUTHENTICATED: set[tuple[str, str]] = set()

_PROXY_HOSTS = {
    "codex": ["chatgpt.com", "api.openai.com", "auth.openai.com"],
    "opencode": [
        "api.opencode.ai", "models.dev", "chatgpt.com", "api.openai.com",
        "api.anthropic.com", "openrouter.ai",
    ],
}

_CODEX_DISABLED_FEATURES = (
    "apps", "browser_use", "browser_use_external", "computer_use",
    "enable_mcp_apps", "in_app_browser", "plugins", "remote_plugin",
    "shell_tool", "skill_mcp_dependency_install",
    "tool_call_mcp_elicitation",
)

_OPENCODE_INFERENCE_CONFIG = json.dumps({
    "permission": {"*": "deny"},
    "tools": {"*": False},
    "mcp": {},
})
_HOST_RE = re.compile(
    r"^(?=.{1,253}$)(?:[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?\.)*"
    r"[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?$"
)


def selected_agent() -> str | None:
    value = os.environ.get("RAPTOR_AGENT", "").strip().lower()
    if value in SUPPORTED_AGENT_CLIS | {"claude"}:
        return value
    # Direct invocation from the checkout has no launcher to export
    # RAPTOR_AGENT. Use host-owned session markers so ``claude``, ``codex``,
    # and ``opencode`` retain the same inference transport as orchestration.
    # Tests opt into host binding explicitly and otherwise exercise standalone
    # provider resolution independently of whichever agent launched pytest.
    if os.environ.get("PYTEST_CURRENT_TEST"):
        return None
    if os.environ.get("CLAUDECODE"):
        return "claude"
    if os.environ.get("CODEX_SESSION_ID") or os.environ.get("CODEX_THREAD_ID"):
        return "codex"
    if os.environ.get("OPENCODE_SESSION_ID"):
        return "opencode"
    return None


def resolve_agent_cli(agent: str) -> str:
    if agent not in SUPPORTED_AGENT_CLIS:
        raise ValueError(f"unsupported agent CLI transport: {agent}")
    resolved = shutil.which(agent)
    if not resolved:
        raise RuntimeError(f"selected agent CLI is unavailable: {agent}")
    return str(Path(resolved).resolve())


def _prompt(user_prompt: str, system_prompt: str | None) -> str:
    if not system_prompt:
        return user_prompt
    return (
        "<raptor-system-instructions>\n"
        f"{system_prompt}\n"
        "</raptor-system-instructions>\n\n"
        "<raptor-user-content>\n"
        f"{user_prompt}\n"
        "</raptor-user-content>"
    )


def _parse_json_text(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()
    value = json.loads(stripped)
    if not isinstance(value, dict):
        raise RuntimeError("agent CLI structured response is not a JSON object")
    return value


def _safe_diagnostic(value: object, limit: int = 500) -> str:
    from core.security.log_sanitisation import escape_nonprintable
    from core.security.redaction import redact_secrets

    clean = escape_nonprintable(redact_secrets(str(value).strip()))
    return clean if len(clean) <= limit else "..." + clean[-limit:]


def _auth_paths(agent: str) -> list[str]:
    home = Path.home()
    if agent == "codex":
        root = Path(os.environ.get("CODEX_HOME", home / ".codex"))
        return [str(root / "auth.json")]
    data = Path(os.environ.get("XDG_DATA_HOME", home / ".local" / "share"))
    config = Path(os.environ.get("XDG_CONFIG_HOME", home / ".config"))
    return [
        str(data / "opencode" / "auth.json"),
        str(config / "opencode" / "opencode.json"),
        str(config / "opencode" / "opencode.jsonc"),
    ]


def _proxy_hosts(agent: str) -> list[str]:
    hosts = list(_PROXY_HOSTS[agent])
    raw = os.environ.get("RAPTOR_AGENT_CLI_PROXY_HOSTS", "")
    for item in raw.split(","):
        host = item.strip().lower().rstrip(".")
        if not host:
            continue
        if not _HOST_RE.fullmatch(host):
            raise RuntimeError(
                "RAPTOR_AGENT_CLI_PROXY_HOSTS contains an invalid hostname: "
                f"{host!r}"
            )
        if host not in hosts:
            hosts.append(host)
    return hosts


def _check_auth(agent: str, binary: str, env: dict[str, str]) -> None:
    key = (agent, binary)
    with _AUTH_LOCK:
        if key in _AUTHENTICATED:
            return
    cmd = (
        [binary, "login", "status"] if agent == "codex"
        else [binary, "auth", "list", "--pure"]
    )
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=10, check=False,
            env=env,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        recovery = "codex login" if agent == "codex" else "opencode auth login"
        raise RuntimeError(
            f"{agent} authentication check failed: {_safe_diagnostic(exc)}; "
            f"run `{recovery}`"
        ) from exc
    output = "\n".join(x for x in (result.stdout, result.stderr) if x)
    authenticated = result.returncode == 0
    if agent == "opencode":
        authenticated = (
            authenticated
            and "Credentials" in output
            and re.search(r"\b0 credentials\b", output, re.IGNORECASE) is None
        )
    if not authenticated:
        recovery = "codex login" if agent == "codex" else "opencode auth login"
        raise RuntimeError(
            f"{agent} authentication unavailable: "
            f"{_safe_diagnostic(output or f'exit {result.returncode}')}; "
            f"run `{recovery}`"
        )
    with _AUTH_LOCK:
        _AUTHENTICATED.add(key)


def _claim_call(agent: str) -> None:
    global _CALL_COUNT
    raw = os.environ.get("RAPTOR_AGENT_CLI_MAX_CALLS", str(DEFAULT_MAX_CALLS))
    try:
        limit = int(raw)
    except ValueError as exc:
        raise RuntimeError("RAPTOR_AGENT_CLI_MAX_CALLS must be an integer") from exc
    if limit < 1:
        raise RuntimeError("RAPTOR_AGENT_CLI_MAX_CALLS must be at least 1")
    with _CALL_LOCK:
        if _CALL_COUNT >= limit:
            raise RuntimeError(
                f"{agent} subscription call limit reached ({limit}); set "
                "RAPTOR_AGENT_CLI_MAX_CALLS to an explicit higher bound"
            )
        _CALL_COUNT += 1


def run_agent_cli(
    agent: str,
    prompt: str,
    *,
    system_prompt: str | None = None,
    model: str | None = None,
    schema: dict[str, Any] | None = None,
    timeout_s: int | None = 600,
) -> tuple[str, dict[str, Any] | None, float]:
    """Run one isolated, non-interactive inference call."""
    binary = resolve_agent_cli(agent)
    full_prompt = _prompt(prompt, system_prompt)
    started = time.monotonic()
    with scratch_dir(f"raptor-{agent}-cli-") as work:
        work_path = Path(work)
        if agent == "codex":
            output_path = work_path / "last-message.txt"
            cmd = [
                binary, "exec", "--strict-config", "--ignore-user-config",
                "--ignore-rules", "--ephemeral", "--skip-git-repo-check",
                "--sandbox", "read-only", "--color", "never",
                "--output-last-message", str(output_path), "-C", str(work_path),
            ]
            for override in (
                "project_doc_max_bytes=0", "project_doc_fallback_filenames=[]",
                "project_root_markers=[]", 'web_search="disabled"',
                "tools.web_search=false", "mcp_servers={}",
            ):
                cmd.extend(["--config", override])
            for feature in _CODEX_DISABLED_FEATURES:
                cmd.extend(["--disable", feature])
            if model and model != "session-default":
                cmd.extend(["--model", model])
            if schema is not None:
                schema_path = work_path / "schema.json"
                schema_path.write_text(json.dumps(schema), encoding="utf-8")
                cmd.extend(["--output-schema", str(schema_path)])
            cmd.append("-")
        else:
            cmd = [binary, "run", "--pure", "--dir", str(work_path)]
            if model and model != "session-default":
                cmd.extend(["--model", model])
            if schema is not None:
                full_prompt += (
                    "\n\nReturn only JSON matching this schema:\n"
                    + json.dumps(schema, separators=(",", ":"))
                )

        try:
            from core.config import RaptorConfig

            child_env = RaptorConfig.get_safe_env()
            if os.environ.get("CODEX_HOME"):
                child_env["CODEX_HOME"] = os.environ["CODEX_HOME"]
            # A host-bound session means "use this CLI's authenticated
            # account". Ambient model API keys must not silently change
            # its billing route. The CLI's own config/auth stores remain.
            for name in (
                "ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GEMINI_API_KEY",
                "MISTRAL_API_KEY", "AWS_BEARER_TOKEN_BEDROCK",
            ):
                child_env.pop(name, None)
            if agent == "opencode":
                child_env["OPENCODE_CONFIG_CONTENT"] = _OPENCODE_INFERENCE_CONFIG
            _check_auth(agent, binary, child_env)
            _claim_call(agent)
            from core.sandbox import run_untrusted_networked
            result = run_untrusted_networked(
                cmd,
                input=full_prompt,
                capture_output=True,
                text=True,
                timeout=timeout_s,
                env=child_env,
                target=str(work_path),
                output=str(work_path),
                readable_paths=[
                    str(Path(binary).parent),
                    *(p for p in _auth_paths(agent) if Path(p).exists()),
                ],
                writable_paths=[str(work_path)],
                proxy_hosts=_proxy_hosts(agent),
                caller_label=f"{agent}-inference",
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(f"{agent} CLI timed out after {timeout_s}s") from exc
        if result.returncode != 0:
            detail = _safe_diagnostic(
                result.stderr or result.stdout or f"exit {result.returncode}",
            )
            raise RuntimeError(f"{agent} CLI inference failed: {detail}")
        if agent == "codex":
            text = output_path.read_text(encoding="utf-8")
        else:
            text = result.stdout
        structured = _parse_json_text(text) if schema is not None else None
        return text.strip(), structured, time.monotonic() - started
