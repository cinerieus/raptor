"""HMAC provenance for review-journal rows.

Review-journal entries are the durable record the gap fold trusts: a
row whose ``verdict`` is conclusive and whose ``source_hash`` matches
the current source SUPPRESSES future review of that function (and,
when eligible, is imported as a $0 reused verdict). The journal lives
in run/project directories that are target-writable during runs and
restorable verbatim by ``/project import`` — and rows were plain
unauthenticated JSON, so a forged ``clean`` row silenced review of a
function forever.

Writers stamp each row at append time with an HMAC-SHA256 token over
the row's canonical JSON (token key excluded); the fold verifies
before granting a row verdict-reuse authority, and rows whose token
is present but invalid are skipped entirely (tampered). Rows with NO
token (pre-MAC legacy, or forged-unstamped) keep fold-credit only
behind the exact full-length source-hash gate and are never eligible
for $0 verdict reuse — the tolerant-reader compromise that avoids a
re-review storm on upgrade while denying unauthenticated rows every
authority tier above "the source hash checks out". Same trust story
and key-handling discipline as ``core/witness/provenance.py`` /
``core/llm/scorecard/integrity.py``.

Key
    ``$XDG_DATA_HOME/raptor/journal-mac.key`` (default
    ``~/.local/share/raptor/journal-mac.key``). Deliberately its OWN
    key file — per-purpose keys keep reset/rotation semantics scoped.
    No rotation: deleting the key demotes every stamped row to the
    unstamped tier (fold-credit behind the hash gate, no reuse) and
    new appends re-key lazily.

No run binding: journal rows deliberately travel across runs — the
project index aggregates them and cross-run verdict reuse is the
feature. A replayed validly-stamped row is genuine history for the
exact source hash it names; staleness is bounded by the fold's hash
compare, not the MAC.

Forward compatibility: verification round-trips the row through this
reader's dataclass, so a row written by a NEWER schema (additive
fields this reader doesn't know) — or stamped by another install's
key — reads as token-present-but-unverifiable. That is NOT treated
as a distinct security tier: whoever can edit a row can simply strip
its token and land in the unstamped tier anyway, so consumers demote
unverifiable rows to the same unstamped tier (exact-hash-gated fold
credit, never verdict reuse) instead of dropping them below it.
"""

from __future__ import annotations

import hashlib
import hmac
import os
from pathlib import Path

from core.json.utils import dumps_canonical
from core.logging import get_logger
from core.security import mac_key

logger = get_logger(__name__)

_KEY_LEN = 32

# Sentinel: a key file EXISTS but is unusable (symlink, foreign owner,
# group/other-readable). Distinct from "absent" — an unusable key must
# never be silently replaced and must never mint or verify.
_REFUSED = mac_key.REFUSED

_warned_paths: set = set()

# Row key holding the token. Popped before canonicalisation so the
# MAC covers everything else in the row.
TOKEN_KEY = "integrity"

#: Tri-state provenance of a loaded row.
ROW_VERIFIED = "verified"
ROW_TAMPERED = "tampered"
ROW_UNSTAMPED = "unstamped"


def _key_path() -> Path:
    xdg = os.environ.get("XDG_DATA_HOME")
    base = Path(xdg) if xdg else Path.home() / ".local" / "share"
    return base / "raptor" / "journal-mac.key"


def _warn_once_suspect_key(path: Path, reason: str, remedy: str) -> None:
    key = str(path)
    if key in _warned_paths:
        logger.debug(f"journal integrity: suspect key {path} ({reason})")
        return
    _warned_paths.add(key)
    logger.warning(
        f"journal integrity: refusing key {path} — {reason}. Journal "
        f"rows will not mint or verify (stamped rows demote to the "
        f"unstamped tier: hash-gated fold credit only, no verdict "
        f"reuse) until this is fixed: {remedy}"
    )


def _read_existing_key(path: Path) -> bytes | mac_key.Refused | None:
    """Read an EXISTING key with the shared fd-fstat discipline
    (:func:`core.security.mac_key.read_existing_key`): refuse symlinks
    (O_NOFOLLOW + fstat on the opened inode), foreign owners, and any
    group/other permission bits."""
    return mac_key.read_existing_key(
        path, key_len=_KEY_LEN, warn=_warn_once_suspect_key)


def _load_or_create_key() -> bytes | None:
    """Read the key, lazily creating it (0700 dir, 0600 file, O_EXCL)
    if absent — the shared hardened discipline in
    :func:`core.security.mac_key.load_or_create_key`. Returns None
    when a key file exists but is unusable — the suspect key is never
    used, never replaced. The creation-race loser polls through the
    winner's create-to-write window instead of mis-flagging the
    mid-write key as suspect (which left the loser's rows unstamped —
    demoted to the no-reuse tier — under a false suspect-key
    warning)."""
    return mac_key.load_or_create_key(
        _key_path(), key_len=_KEY_LEN, warn=_warn_once_suspect_key,
        read_existing=_read_existing_key)


def key_usable() -> bool:
    """Whether this install can mint/verify tokens at all."""
    try:
        return bool(_load_or_create_key())
    except OSError:
        return False


def row_sha256(row: dict) -> str:
    """sha256 over the row's canonical JSON (token key excluded).

    Canonical form: :func:`core.json.utils.dumps_canonical` (stdlib
    ``sort_keys=True, separators=(",", ":"), default=str`` — the
    repo-wide frozen canonical byte form; its tests pin byte-identity
    against this function) — key order and whitespace don't matter,
    values do. The token covers the WHOLE row (verdict, source_hash,
    spans, producer, model, strategies, body, ...): partial coverage
    would let an attacker rewrite the unauthenticated remainder of a
    validly-stamped row."""
    scrubbed = {k: v for k, v in row.items() if k != TOKEN_KEY}
    canonical = dumps_canonical(scrubbed)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _mac_message(sha256_hex: str) -> bytes:
    # Domain separation: a token minted for another artifact class
    # can never verify here even if a key were ever shared by mistake.
    return b"review-journal-row\x00" + sha256_hex.encode("ascii")


def mint_row(row: dict) -> str | None:
    """Hex HMAC-SHA256 token over the row's canonical payload, or
    None when no usable key is available. Writers treat None as
    "persist unstamped" — the fold then applies unstamped-tier
    semantics."""
    try:
        key = _load_or_create_key()
    except OSError:
        return None
    if not key:
        return None
    return hmac.new(
        key, _mac_message(row_sha256(row)), hashlib.sha256,
    ).hexdigest()


def verify_row(row: dict, token: str | None) -> bool:
    """Whether *token* is a valid MAC over *row*'s canonical payload
    under this install's key. Constant-time; never raises — any
    failure is the caller's demote path."""
    if not token:
        return False
    try:
        expected = mint_row(row)
        if expected is None:
            return False
        return hmac.compare_digest(expected, str(token).strip().lower())
    except Exception:  # noqa: BLE001 — verification failure is the demote path, never an error
        return False


def entry_provenance(entry) -> str:
    """Tri-state provenance of a loaded ``ReviewJournalEntry``.

    * ``verified`` — token present and valid for the row's content.
    * ``tampered`` — token present but invalid: content edited, a row
      minted by another install, or a row written by a newer schema
      whose extra fields this reader's dataclass round-trip loses.
      Consumers give these the same authority as ``unstamped`` (the
      token is strippable, so "tampered" is attribution, not a
      security boundary) but log them distinctly.
    * ``unstamped`` — no token (pre-MAC legacy or forged-unstamped):
      fold-credit only behind the exact source-hash gate, never
      verdict reuse.
    """
    token = getattr(entry, "integrity", None)
    if not token:
        return ROW_UNSTAMPED
    row = entry.to_dict()
    return ROW_VERIFIED if verify_row(row, token) else ROW_TAMPERED


__all__ = [
    "ROW_TAMPERED",
    "ROW_UNSTAMPED",
    "ROW_VERIFIED",
    "TOKEN_KEY",
    "entry_provenance",
    "key_usable",
    "mint_row",
    "row_sha256",
    "verify_row",
]
