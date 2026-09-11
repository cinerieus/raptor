"""Subscription-backed coding-agent CLI inference transports."""

from __future__ import annotations

import json
import os
import re
import shutil
import stat
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
    "enable_mcp_apps", "goals", "hooks", "image_generation",
    "in_app_browser", "multi_agent", "plugins", "remote_plugin",
    "shell_tool", "skill_mcp_dependency_install", "skill_search",
    "sleep_tool", "tool_call_mcp_elicitation", "unified_exec",
    "view_image", "workspace_dependencies",
)

_OPENCODE_INFERENCE_CONFIG = json.dumps({
    "permission": {"*": "deny"},
    "tools": {"*": False},
    "mcp": {},
    "plugin": [],
    "instructions": [],
    "default_agent": "raptor-inference",
    "agent": {
        "raptor-inference": {
            "description": "Tool-disabled RAPTOR inference substrate",
            "mode": "primary",
            "prompt": "Answer only the supplied RAPTOR analysis request.",
            "permission": {"*": "deny"},
            "tools": {"*": False},
        },
    },
})
_OPENCODE_SKILL_CONFIG = json.dumps({
    "permission": {
        "*": "deny",
        "bash": "allow",
        "edit": "allow",
        "glob": "allow",
        "grep": "allow",
        "read": "allow",
        "write": "allow",
    },
    "tools": {
        "*": False,
        "bash": True,
        "edit": True,
        "glob": True,
        "grep": True,
        "read": True,
        "write": True,
    },
    "mcp": {},
    "plugin": [],
    "instructions": [],
    "default_agent": "raptor-skill",
    "agent": {
        "raptor-skill": {
            "description": "Sandboxed RAPTOR skill runner",
            "mode": "primary",
            "prompt": "Follow only the supplied RAPTOR workflow request.",
            "permission": {
                "*": "deny",
                "bash": "allow",
                "edit": "allow",
                "glob": "allow",
                "grep": "allow",
                "read": "allow",
                "write": "allow",
            },
            "tools": {
                "*": False,
                "bash": True,
                "edit": True,
                "glob": True,
                "grep": True,
                "read": True,
                "write": True,
            },
        },
    },
})
_HOST_RE = re.compile(
    r"^(?=.{1,253}$)(?:[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?\.)*"
    r"[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?$"
)
_MODEL_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/+-]{0,255}$")


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


def _validate_agent_model(value: str, *, source: str) -> str:
    model = value.strip()
    if not _MODEL_RE.fullmatch(model):
        raise RuntimeError(f"{source} contains an invalid model identifier")
    return model


def _resolve_agent_model(agent: str, requested: str | None) -> str | None:
    """Resolve explicit pins; otherwise let the host CLI load its config."""
    if requested and requested != "session-default":
        return _validate_agent_model(requested, source="requested model")
    launcher_model = os.environ.get("RAPTOR_AGENT_MODEL", "").strip()
    if launcher_model:
        return _validate_agent_model(
            launcher_model, source="RAPTOR_AGENT_MODEL"
        )
    return None


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


def _auth_paths(agent: str, env: dict[str, str] | None = None) -> list[str]:
    home = Path.home()
    values = env if env is not None else os.environ
    if agent == "codex":
        root = Path(values.get("CODEX_HOME", home / ".codex"))
        return [str(root / "auth.json"), str(root / "config.toml")]
    data = Path(values.get("XDG_DATA_HOME", home / ".local" / "share"))
    config = Path(values.get("XDG_CONFIG_HOME", home / ".config"))
    return [
        str(data / "opencode" / "auth.json"),
        str(config / "opencode" / "opencode.json"),
        str(config / "opencode" / "opencode.jsonc"),
    ]


def _stage_bounded_file(source: Path, destination: Path, recovery: str,
                        label: str) -> None:
    """Copy one bounded regular file without following symlinks."""
    try:
        source_fd = os.open(
            source,
            os.O_RDONLY | os.O_CLOEXEC | getattr(os, "O_NOFOLLOW", 0),
        )
    except OSError as exc:
        raise RuntimeError(
            f"subscription auth is unavailable; run `{recovery}`"
        ) from exc
    try:
        source_stat = os.fstat(source_fd)
        if (not stat.S_ISREG(source_stat.st_mode)
                or source_stat.st_size > 1024 * 1024):
            raise RuntimeError(
                f"{label} is not a regular file under 1 MiB; "
                f"run `{recovery}`"
            )
        auth_bytes = bytearray()
        while len(auth_bytes) <= 1024 * 1024:
            chunk = os.read(source_fd, min(65536, 1024 * 1024 + 1
                                           - len(auth_bytes)))
            if not chunk:
                break
            auth_bytes.extend(chunk)
        if len(auth_bytes) > 1024 * 1024:
            raise RuntimeError(f"{label} exceeds 1 MiB")
    finally:
        os.close(source_fd)
    destination.write_bytes(auth_bytes)
    destination.chmod(0o600)


def _stage_codex_home(work_path: Path) -> Path:
    """Snapshot Codex routing config and auth into a private runtime home."""
    source_root = Path(os.environ.get("CODEX_HOME", Path.home() / ".codex"))
    staged = work_path / ".codex-home"
    staged.mkdir(mode=0o700)
    (staged / "tmp").mkdir(mode=0o700)
    _stage_bounded_file(
        source_root / "auth.json", staged / "auth.json", "codex login",
        "auth.json",
    )
    source_config = source_root / "config.toml"
    if source_config.exists():
        _stage_bounded_file(
            source_config, staged / "config.toml",
            "repair the Codex config",
            "config.toml",
        )
    return staged


def _stage_opencode_data(work_path: Path) -> Path:
    """Create a private OpenCode data root containing subscription auth."""
    home = Path.home()
    source_data = Path(
        os.environ.get("XDG_DATA_HOME", home / ".local" / "share")
    )
    staged_data = work_path / ".xdg-data"
    app_data = staged_data / "opencode"
    app_data.mkdir(parents=True, mode=0o700)
    _stage_bounded_file(
        source_data / "opencode" / "auth.json",
        app_data / "auth.json",
        "opencode auth login",
        "auth.json",
    )
    return staged_data


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
    effective_model = _resolve_agent_model(agent, model)
    full_prompt = _prompt(prompt, system_prompt)
    started = time.monotonic()
    with scratch_dir(f"raptor-{agent}-cli-") as work:
        work_path = Path(work)
        if agent == "codex":
            output_path = work_path / "last-message.txt"
            cmd = [
                binary, "exec", "--strict-config", "--ignore-rules",
                "--ephemeral", "--skip-git-repo-check",
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
            if effective_model:
                cmd.extend(["--model", effective_model])
            if schema is not None:
                schema_path = work_path / "schema.json"
                schema_path.write_text(json.dumps(schema), encoding="utf-8")
                cmd.extend(["--output-schema", str(schema_path)])
            cmd.append("-")
        else:
            cmd = [
                binary, "run", "--pure", "--agent", "raptor-inference",
                "--dir", str(work_path),
            ]
            if effective_model:
                cmd.extend(["--model", effective_model])
            if schema is not None:
                full_prompt += (
                    "\n\nReturn only JSON matching this schema:\n"
                    + json.dumps(schema, separators=(",", ":"))
                )

        try:
            from core.config import RaptorConfig

            child_env = RaptorConfig.get_safe_env()
            if agent == "codex":
                child_env["CODEX_HOME"] = str(
                    _stage_codex_home(work_path)
                )
            elif agent == "opencode":
                child_env["XDG_DATA_HOME"] = str(
                    _stage_opencode_data(work_path)
                )
                child_env["XDG_CONFIG_HOME"] = os.environ.get(
                    "XDG_CONFIG_HOME", str(Path.home() / ".config")
                )
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
            # Both CLIs use OpenTelemetry-compatible exporters. Disable those
            # exporters in inference children: telemetry is not required for
            # authentication, model discovery, or inference, and its denied
            # CONNECTs otherwise look like transport failures.
            child_env["OTEL_SDK_DISABLED"] = "true"
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
                    *(p for p in _auth_paths(agent, child_env)
                      if Path(p).exists()),
                    *(
                        [str(Path(child_env["XDG_CONFIG_HOME"]) / "opencode")]
                        if agent == "opencode" and (
                            Path(child_env["XDG_CONFIG_HOME"]) / "opencode"
                        ).exists() else []
                    ),
                ],
                writable_paths=[
                    str(work_path),
                    *(
                        [str(Path(child_env["CODEX_HOME"]) / "tmp")]
                        if agent == "codex" and (
                            Path(child_env["CODEX_HOME"]) / "tmp"
                        ).is_dir() else []
                    ),
                ],
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


def run_agent_skill_cli(
    agent: str,
    prompt: str,
    *,
    target: Path,
    output: Path,
    context_dirs: tuple[Path, ...] = (),
    timeout_s: int = 900,
    caller_label: str = "agent-skill",
) -> subprocess.CompletedProcess[str]:
    """Run a tool-enabled skill through the selected CLI.

    The coding agent's own sandbox is disabled because RAPTOR wraps the whole
    process in its namespace, Landlock, seccomp, and egress sandbox.  The
    target and RAPTOR installation are read-only; only the lifecycle output
    and the private staged CLI state are writable.
    """
    if agent not in SUPPORTED_AGENT_CLIS:
        raise ValueError(f"unsupported agent skill CLI: {agent}")
    binary = resolve_agent_cli(agent)
    effective_model = _resolve_agent_model(agent, None)
    target = Path(target).resolve()
    output = Path(output).resolve()
    contexts = tuple(Path(path).resolve() for path in context_dirs)

    with scratch_dir(f"raptor-{agent}-skill-") as work:
        work_path = Path(work)
        if agent == "codex":
            output_path = work_path / "last-message.txt"
            cmd = [
                binary, "exec", "--strict-config", "--ignore-rules",
                "--ephemeral", "--skip-git-repo-check",
                "--dangerously-bypass-approvals-and-sandbox",
                "--color", "never", "--output-last-message",
                str(output_path), "-C", str(output),
            ]
            for override in (
                "project_doc_max_bytes=0", "project_doc_fallback_filenames=[]",
                "project_root_markers=[]", 'web_search="disabled"',
                "tools.web_search=false", "mcp_servers={}",
            ):
                cmd.extend(["--config", override])
            for feature in _CODEX_DISABLED_FEATURES:
                if feature not in {"shell_tool", "unified_exec"}:
                    cmd.extend(["--disable", feature])
            if effective_model:
                cmd.extend(["--model", effective_model])
            cmd.append("-")
        else:
            cmd = [
                binary, "run", "--pure", "--agent", "raptor-skill",
                "--dir", str(output), "--auto",
            ]
            if effective_model:
                cmd.extend(["--model", effective_model])

        from core.config import RaptorConfig

        child_env = RaptorConfig.get_safe_env()
        if agent == "codex":
            child_env["CODEX_HOME"] = str(_stage_codex_home(work_path))
        else:
            child_env["XDG_DATA_HOME"] = str(_stage_opencode_data(work_path))
            child_env["XDG_CONFIG_HOME"] = os.environ.get(
                "XDG_CONFIG_HOME", str(Path.home() / ".config")
            )
            child_env["OPENCODE_CONFIG_CONTENT"] = _OPENCODE_SKILL_CONFIG
        for name in (
            "ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GEMINI_API_KEY",
            "MISTRAL_API_KEY", "AWS_BEARER_TOKEN_BEDROCK",
        ):
            child_env.pop(name, None)
        child_env["OTEL_SDK_DISABLED"] = "true"
        _check_auth(agent, binary, child_env)
        _claim_call(agent)

        readable = [
            str(Path(binary).parent),
            str(Path(__file__).resolve().parents[2]),
            *(str(path) for path in contexts),
            *(p for p in _auth_paths(agent, child_env) if Path(p).exists()),
        ]
        if agent == "opencode":
            config_dir = Path(child_env["XDG_CONFIG_HOME"]) / "opencode"
            if config_dir.exists():
                readable.append(str(config_dir))

        from core.sandbox import run_untrusted_networked

        result = run_untrusted_networked(
            cmd,
            input=prompt,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            cwd=str(work_path),
            env=child_env,
            target=str(target),
            output=str(output),
            readable_paths=readable,
            writable_paths=[str(work_path), str(output)],
            proxy_hosts=_proxy_hosts(agent),
            caller_label=caller_label,
        )
        if agent == "codex" and result.returncode == 0:
            final_text = output_path.read_text(encoding="utf-8")
            return subprocess.CompletedProcess(
                result.args, result.returncode, final_text, result.stderr,
            )
        return result
