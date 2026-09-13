"""Canonical package-name folding — one home for the PEP 503 rule.

Every ecosystem that joins names across surfaces (manifest ↔ lockfile,
dep row ↔ registry response, finding ↔ advisory) needs the SAME
canonical form on both sides or the join silently misses.  The PEP 503
rule (runs of ``-``/``_``/``.`` fold to ``-``, then lowercase) was
hand-rolled in a dozen modules; this module is the single
implementation they all share.

Per-ecosystem semantics are deliberately explicit:

* **PyPI** — PEP 503 normalisation (``Foo_Bar`` ≡ ``foo-bar``).
* **npm** — names are case-sensitive on disk but the registry rejects
  new names differing only by case; conventional form is lowercase.
  Scope and slash are preserved.
* **Cargo / Maven / everything else** — case-sensitive at registry
  level; names pass through unchanged.  Consumers whose BOUNDARY
  case-folds additional ecosystems (OSV case-folds crates.io names)
  opt in via ``extra_lower`` rather than forking the rule.
"""

from __future__ import annotations

import re

_PEP503_RUNS_RE = re.compile(r"[-_.]+")


def pep503_name(name: str) -> str:
    """PEP 503 canonical form: fold ``-``/``_``/``.`` runs to ``-``,
    lowercase."""
    return _PEP503_RUNS_RE.sub("-", name).lower()


def fold_name(
    name: str,
    ecosystem: str,
    *,
    extra_lower: tuple[str, ...] = (),
) -> str:
    """Per-ecosystem canonical name (see module docstring).

    ``extra_lower`` names ecosystems the CALLER's boundary case-folds
    beyond the default (npm); everything else passes through
    unchanged.
    """
    if ecosystem == "PyPI":
        return pep503_name(name)
    if ecosystem == "npm" or ecosystem in extra_lower:
        return name.lower()
    return name


__all__ = ["fold_name", "pep503_name"]
