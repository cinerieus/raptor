---
description: Initialize the RAPTOR security testing assistant
dispatch: skill
---

# RAPTOR session initialization

This command initializes the interactive RAPTOR session. It never starts a
scan, agentic analysis, or another RAPTOR command.

1. Read the shared contract in `CLAUDE.md` if it is not already loaded in this
   session.
2. Read `.startup-output` and output it verbatim as a fenced code block. If it
   is unavailable, report that startup diagnostics are unavailable instead of
   inventing a banner.
3. Output `Quick commands:` followed by `agentic`, `scan`, `fuzz`, `web`, and
   `commands`. Prefix them with `$` when `RAPTOR_AGENT=codex`; otherwise prefix
   them with `/`.
4. If an argument was supplied, state briefly that it is the default target
   for this session. Do not scan it.
5. Wait for the operator's next command.

Do not dispatch `python3 raptor.py agentic`, `/agentic`, `$agentic`, or any
other analysis command from this initialization command.
