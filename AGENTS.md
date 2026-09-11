# RAPTOR agent contract

This repository can be operated through Claude Code, Codex, or OpenCode.
Starting any of those CLIs directly from this checkout is a supported RAPTOR
session. Each host exposes the same canonical RAPTOR command registry through
its native command syntax.

Explicit analysis flags, `models.json`, and API-backed provider configuration
remain independent of the orchestration host. When none is configured,
downstream model calls reuse the current host CLI and its stored subscription
authentication.

Before acting:

1. Read `CLAUDE.md` in full. Despite its legacy filename, its execution, command dispatch, run lifecycle, project, security, output, and interaction rules are the shared RAPTOR agent contract.
2. Treat `.claude/commands/<name>.md` as the provider-neutral registry for RAPTOR commands. OpenCode wrappers expose `/name`; Codex project skills expose `$name`. Read the canonical file and follow its `dispatch:` contract.
3. Read any skill or tier file referenced by the selected command before executing it.

Provider-specific UI features are optional. The Python execution layer, `libexec/` commands, project/session registry, trust checks, and run lifecycle are authoritative. If a host lacks a Claude hook or plugin feature, report that loss and continue with the shared mechanical path.
