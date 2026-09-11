"""Persistent cache for successful agent-CLI envelope probes.

Only explicit CLI model selections are cacheable. ``session-default`` is
deliberately excluded because its resolved model cannot be known before the
inference call. Cache corruption, expiry, or version lookup failure is a miss.
"""

from __future__ import annotations

import hashlib
import json
import os
import stat as stat_module
import subprocess
import tempfile
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

_SCHEMA_VERSION = 1
_TTL_SECONDS = 24 * 60 * 60
_CLI_PROVIDERS = {"codexcli": "codex", "opencodecli": "opencode"}
_MAX_ROUTING_FILE_BYTES = 1024 * 1024


def _routing_file_digest(path: Path) -> str | None:
    """Hash a bounded regular routing file, or return None if unsafe."""
    try:
        info = path.lstat()
    except FileNotFoundError:
        return "missing"
    except OSError:
        return None
    if (not stat_module.S_ISREG(info.st_mode)
            or info.st_size > _MAX_ROUTING_FILE_BYTES):
        return None
    try:
        data = path.read_bytes()
    except OSError:
        return None
    if len(data) > _MAX_ROUTING_FILE_BYTES:
        return None
    return hashlib.sha256(data).hexdigest()


def _routing_fingerprint(cli: str) -> str | None:
    """Fingerprint model/provider routing without recording secret values."""
    home = Path.home()
    paths: list[Path]
    content = ""
    if cli == "codex":
        paths = [
            Path(os.environ.get("CODEX_HOME", home / ".codex"))
            / "config.toml"
        ]
    else:
        config_home = Path(
            os.environ.get("XDG_CONFIG_HOME", home / ".config")
        )
        paths = [
            config_home / "opencode" / "opencode.json",
            config_home / "opencode" / "opencode.jsonc",
        ]
        explicit = os.environ.get("OPENCODE_CONFIG", "").strip()
        if explicit:
            paths.append(Path(explicit))
        content = os.environ.get("OPENCODE_CONFIG_CONTENT", "")
        if len(content.encode("utf-8")) > _MAX_ROUTING_FILE_BYTES:
            return None

    file_digests: dict[str, str] = {}
    for path in paths:
        digest = _routing_file_digest(path)
        if digest is None:
            return None
        file_digests[str(path)] = digest

    routing_env = {
        name: hashlib.sha256(value.encode("utf-8")).hexdigest()
        for name, value in os.environ.items()
        if value and (
            name == "RAPTOR_AGENT_CLI_PROXY_HOSTS"
            or name.endswith((
                "_BASE_URL", "_ENDPOINT", "_MODEL", "_PROVIDER",
                "_REGION", "_PROFILE",
            ))
        )
    }
    payload = {
        "files": file_digests,
        "content": hashlib.sha256(content.encode("utf-8")).hexdigest(),
        "environment": routing_env,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _cache_path() -> Path:
    root = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    return root / "raptor" / "envelope-probes.json"


def _identity(model: Any, profile: Any) -> str | None:
    provider = str(getattr(model, "provider", ""))
    cli = _CLI_PROVIDERS.get(provider)
    model_name = str(getattr(model, "model_name", ""))
    if not cli or not model_name or model_name == "session-default":
        return None

    try:
        from core.llm.agent_cli_adapter import resolve_agent_cli

        binary = resolve_agent_cli(cli)
        completed = subprocess.run(
            [binary, "--version"], capture_output=True, text=True,
            timeout=5, check=False,
        )
        if completed.returncode != 0:
            return None
        version = (completed.stdout or completed.stderr).strip()
        stat = Path(binary).stat()
    except (OSError, RuntimeError, subprocess.SubprocessError):
        return None

    routing = _routing_fingerprint(cli)
    if routing is None:
        return None

    payload = {
        "schema": _SCHEMA_VERSION,
        "provider": provider,
        "model": model_name,
        "binary": binary,
        "binary_mtime_ns": stat.st_mtime_ns,
        "binary_size": stat.st_size,
        "cli_version": version,
        "routing": routing,
        "profile": asdict(profile),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def has_cached_success(model: Any, profile: Any, *, now: float | None = None) -> bool:
    key = _identity(model, profile)
    if key is None:
        return False
    try:
        payload = json.loads(_cache_path().read_text(encoding="utf-8"))
        cached = payload.get("successes", {}).get(key)
        if cached is None:
            return False
        timestamp = float(cached)
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return False
    current = time.time() if now is None else now
    return 0 <= current - timestamp <= _TTL_SECONDS


def store_success(model: Any, profile: Any, *, now: float | None = None) -> None:
    key = _identity(model, profile)
    if key is None:
        return
    path = _cache_path()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            payload = {}
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        payload = {}
    successes = payload.get("successes")
    if not isinstance(successes, dict):
        successes = {}
    current = time.time() if now is None else now
    successes[key] = current
    successes = {
        item_key: timestamp for item_key, timestamp in successes.items()
        if isinstance(timestamp, (int, float))
        and 0 <= current - timestamp <= _TTL_SECONDS
    }
    payload = {"schema": _SCHEMA_VERSION, "successes": successes}

    try:
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, delete=False,
        ) as handle:
            json.dump(payload, handle, sort_keys=True, separators=(",", ":"))
            handle.flush()
            os.fsync(handle.fileno())
            temporary = Path(handle.name)
        os.chmod(temporary, 0o600)
        os.replace(temporary, path)
    except OSError:
        try:
            temporary.unlink(missing_ok=True)
        except (OSError, UnboundLocalError):
            pass


__all__ = ["has_cached_success", "store_success"]
