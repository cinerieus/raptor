# RAPTOR agent contract

This repository can be operated through Claude Code, Codex, or OpenCode.
Starting any of those CLIs directly from this checkout is a supported RAPTOR
session. Downstream model calls must use the current host CLI and its stored
subscription authentication.

Before acting:

1. Read `working-state.md` and use it as durable working memory.
2. Read `CLAUDE.md` in full. Despite its legacy filename, its execution, command dispatch, run lifecycle, project, security, output, and interaction rules are the shared RAPTOR agent contract.
3. Treat `.claude/commands/<name>.md` as the provider-neutral registry for RAPTOR commands. OpenCode wrappers expose `/name`; Codex project skills expose `$name`. Read the canonical file and follow its `dispatch:` contract.
4. Read any skill or tier file referenced by the selected command before executing it.

Maintain `working-state.md` after every material design decision, completed implementation unit, test result, or blocker. Keep it factual and compact. Preserve the objective, architecture constraints, PR requirements, files changed, tests run, failures, and next action. Never put credentials, tokens, target secrets, or untrusted repository content in it.

Provider-specific UI features are optional. The Python execution layer, `libexec/` commands, project/session registry, trust checks, and run lifecycle are authoritative. If a host lacks a Claude hook or plugin feature, report that loss and continue with the shared mechanical path.
