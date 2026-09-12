"""Containment-tier lattice and the sandbox floor contract.

The sandbox states a caller's containment REQUIREMENT once (a floor)
and enforces it in exactly two places: an entry-time check that bounds
the tier a call INTENDS to run at, and a hard per-dispatch assertion
(:func:`assert_floor`) that compares the tier a lane actually DELIVERS
against the floor immediately before the command is handed to its
executor. Every demotion lane — present and future — passes through a
dispatch site, so a new lane added without thought fails closed
instead of silently running attacker-derived code below the floor.

Kept tiny and dependency-free like ``errors.py`` so every layer
(``context.py``, ``summary.py``, tests) and every consumer can import
the lattice without pulling in the sandbox machinery.

Design notes:

* ``ContainmentTier`` is an ``IntEnum`` — ``>=`` IS the lattice. The
  order is total *within a platform*: Linux uses 0..40, macOS uses
  {0, 100}. MOUNTLESS_NS delivers a superset of NS_NOMOUNT (it adds
  the Landlock policy layer; on the fresh-procfs axis NS_NOMOUNT is
  fail-closed while MOUNTLESS_NS is best-effort for non-contract
  callers, stamped ``pidns_proc_mount_unavailable`` on the hosts
  where it degrades), and MOUNT_NS a strict superset of MOUNTLESS_NS
  (it adds the pivot_root bind tree). NS_NOMOUNT vs
  LANDLOCK_ONLY is the one rung where the axes differ rather than
  nest: NS_NOMOUNT trades Landlock's filesystem/TCP *policy* scoping
  for kernel *containment* — user/pid/ipc/cgroup(+net) namespaces, a
  fresh pid-ns procfs, and seccomp. It sits above LANDLOCK_ONLY
  because the order grades what the floor contract exists to protect:
  isolation of the host's process table, procfs (the same-UID environ
  credential channel), IPC, and network from attacker-derived code.
  Requested-policy enforceability (can Landlock actually restrict the
  writes/reads/ports this caller asked for?) is deliberately NOT part
  of the order — it is a per-axis refusal condition in ``context.py``
  (the construction-time Landlock refusal and the per-call recheck),
  exactly like the Landlock-ABI and seccomp axes below.
* Cross-platform comparability is refused, not fudged: seatbelt has
  no procfs concept, scopes reads/writes like Landlock, and denies
  process-info like a pid namespace — it is not "between" any two
  Linux tiers. Floors are resolved per platform
  (:func:`untrusted_default_floor`); ``SEATBELT = 100`` makes an
  accidental cross-platform compare loudly wrong in tests rather
  than subtly wrong in production.
* Capability axes that vary WITHIN a tier (Landlock ABI, seccomp
  presence, egress tier) are refusal/achievability conditions in
  ``context.py``, never tiers — cramming them into the order would
  break totality (an ABI-2 MOUNTLESS_NS is neither above nor below
  an ABI-4 LANDLOCK_ONLY on a single axis).
* :func:`assert_floor` raises a hard, typed
  :class:`~core.sandbox.errors.SandboxFloorError` — never debug-only,
  never warn-and-continue. It is a plain function call (not an
  ``assert`` statement), so ``python -O`` cannot strip it, and the
  error inherits ``SandboxSetupError``'s BaseException semantics so
  no ``except Exception`` at any altitude can swallow it.
"""

from __future__ import annotations

import sys
from enum import IntEnum

from .errors import SandboxFloorError


class ContainmentTier(IntEnum):
    """Total-order containment tiers per platform. ``>=`` is the lattice."""

    BARE = 0            # rlimits only (operator-disabled / no seatbelt)
    LANDLOCK_ONLY = 10  # Landlock+seccomp+rlimits, host namespaces
    # NS_NOMOUNT: namespace containment without a filesystem-policy
    # layer — the modern spawn backend's Landlock-absent mode: full
    # unshare namespace set (user+pid+ipc+cgroup[+net]), a FRESH
    # pid-ns procfs (fail-closed 'F' on this lane — an upgrade over
    # the deleted unshare-CLI lane, which left the HOST procfs
    # visible), seccomp, and rlimits. No Landlock: filesystem writes
    # and TCP connects are not policy-scoped, which is why the tier
    # sits below MOUNTLESS_NS. The value (20) and label ("ns-only")
    # are kept from the legacy lane the enum slot used to describe —
    # the slot's DELIVERY is upgraded, not its position (see the
    # module docstring for why it still orders above LANDLOCK_ONLY).
    NS_NOMOUNT = 20
    MOUNTLESS_NS = 30   # full ns set + fresh procfs + Landlock, no bind tree
    MOUNT_NS = 40       # pivot_root bind tree (rootfs= variant included);
    #                     the tree, not Landlock, is the tier's defining
    #                     filesystem enforcement (ro binds + tmpfs masks)
    SEATBELT = 100      # macOS SBPL; deliberately not comparable to Linux


# Posture vocabulary: lowercase-hyphenated labels, matching the
# ``--sandbox-*`` flag family's spelling so the same words serve the
# posture record now and any consent flag later.
_TIER_LABELS: dict[ContainmentTier, str] = {
    ContainmentTier.BARE: "none",
    ContainmentTier.LANDLOCK_ONLY: "landlock",
    ContainmentTier.NS_NOMOUNT: "ns-only",
    ContainmentTier.MOUNTLESS_NS: "mountless-ns",
    ContainmentTier.MOUNT_NS: "mount-ns",
    ContainmentTier.SEATBELT: "seatbelt",
}

_LABEL_TIERS: dict[str, ContainmentTier] = {
    label: tier for tier, label in _TIER_LABELS.items()
}

# Floor-source vocabulary for the posture record. Phase-2 surfaces
# only; "flag" and "project" join when the per-run flag and project
# setting land.
FLOOR_SOURCE_DEFAULT = "default"
FLOOR_SOURCE_ENV = "env"
FLOOR_SOURCE_OPERATOR_DISABLE = "operator-disable"


def tier_label(tier: ContainmentTier) -> str:
    """Human/posture label for a tier (``"mount-ns"``, ``"landlock"``...)."""
    return _TIER_LABELS[ContainmentTier(tier)]


def label_tier(label: str) -> ContainmentTier:
    """Inverse of :func:`tier_label`; KeyError on unknown labels
    (posture readers that must tolerate future vocabulary catch it)."""
    return _LABEL_TIERS[label]


def untrusted_default_floor() -> ContainmentTier:
    """The untrusted-execution contract's floor on this platform.

    MOUNT_NS on Linux (the fresh-procfs contract demands the mount-ns
    spawn backend); SEATBELT on macOS (the platform's strongest tier
    provides the untrusted contract there — posture records always
    stamp the backend so every run shows what contained it).
    """
    if sys.platform == "darwin":
        return ContainmentTier.SEATBELT
    return ContainmentTier.MOUNT_NS


def waived_untrusted_floor() -> ContainmentTier:
    """The floor ``RAPTOR_ALLOW_DEGRADED_UNTRUSTED=1`` buys.

    Frozen meaning: "untrusted floor := LANDLOCK_ONLY" on Linux — the
    waiver accepts namespace loss and host-procfs visibility but never
    reaches BARE (Landlock/seccomp/rlimits enforceability gates stay).
    On macOS the waiver's existing semantics accept rlimits-only
    containment (there is no Landlock tier to hold), so it maps to
    BARE there.
    """
    if sys.platform == "darwin":
        return ContainmentTier.BARE
    return ContainmentTier.LANDLOCK_ONLY


def resolve_call_floor(
    *,
    operator_disabled: bool,
    require_fresh_procfs: bool | None,
    untrusted_workload: bool,
    waiver_active: bool = False,
) -> tuple[ContainmentTier, str]:
    """Resolve one run() call's containment floor and its source.

    ``require_fresh_procfs`` is tri-state: ``None`` = the kwarg was
    never passed (trusted default); ``True`` = the resolved untrusted
    contract is in force; ``False`` = the caller passed the kwarg but
    zeroed — either the env-var-honouring derivation under the
    operator's waiver, or a caller-level literal relaxation.
    ``waiver_active`` carries the env truth (the caller reads
    ``RAPTOR_ALLOW_DEGRADED_UNTRUSTED`` — this module deliberately
    reads no environment) so the floor SOURCE is attributed
    truthfully: "env" only when the waiver really is in force; a
    literal ``False`` without the waiver gets the same floor with the
    default source, so no banner or waiver-named warning fires for
    consent nobody gave.

    Precedence (highest wins): operator-explicit disable (``--sandbox
    none`` / ``--no-sandbox`` / ``disabled=True`` — the documented
    "all bets off" surface, floor := BARE) > the per-call contract >
    the waived-untrusted mapping > the trusted default (BARE — plain
    ``run()``'s documented contract is enforceability-gated
    degradation, not a tier floor).
    """
    if operator_disabled:
        return ContainmentTier.BARE, FLOOR_SOURCE_OPERATOR_DISABLE
    if require_fresh_procfs:
        return untrusted_default_floor(), FLOOR_SOURCE_DEFAULT
    if require_fresh_procfs is False:
        # Kwarg present-but-zeroed: an untrusted-class call running at
        # the waived floor.
        return (waived_untrusted_floor(),
                FLOOR_SOURCE_ENV if waiver_active
                else FLOOR_SOURCE_DEFAULT)
    if untrusted_workload:
        # Untrusted-marked call that never derived the contract kwarg:
        # fail CLOSED at the class default. Granting the waived floor
        # here would attribute consent nobody verified — the lowered
        # floor belongs only to callers that carried the derivation
        # through.
        return untrusted_default_floor(), FLOOR_SOURCE_DEFAULT
    return ContainmentTier.BARE, FLOOR_SOURCE_DEFAULT


def assert_floor(
    delivered: ContainmentTier,
    floor: ContainmentTier,
    *,
    lane: str,
    cause: BaseException | None = None,
    detail: str = "",
    remedy: str = "",
    setup_category: str | None = None,
) -> None:
    """Hard pre-exec floor assertion — the load-bearing half of the
    contract.

    Called at every dispatch site immediately before the command is
    handed to that lane's executor. ``delivered >= floor`` returns;
    anything else raises :class:`SandboxFloorError` chained to
    ``cause`` (the original backend failure that demoted the call
    here, when one exists) so the environment problem stays
    diagnosable, carrying ``setup_category`` (explicit, else lifted
    from ``cause``) so retry-capable consumers keep their structural
    signal, and carrying ``remedy`` (the honesty-checked override /
    host-fix sentence built by the caller) so the refusal names the
    way out.

    This is a runtime raise on the security boundary: not debug-only,
    not warn-and-continue, not skippable by any flag. Probes are
    probabilistic — entry-time knowledge is not delivery-time truth —
    so only this assertion makes FUTURE lanes safe by construction.
    """
    delivered = ContainmentTier(delivered)
    floor = ContainmentTier(floor)
    if delivered >= floor:
        return
    if setup_category is None:
        setup_category = getattr(cause, "setup_category", None)
    msg = (
        f"sandbox containment floor violated: this call requires "
        f"{tier_label(floor)} containment but the {lane} lane delivers "
        f"{tier_label(delivered)}"
    )
    if detail:
        msg += f" ({detail})"
    raise SandboxFloorError(
        msg,
        remedy,
        achievable=delivered,
        floor=floor,
        setup_category=setup_category,
    ) from cause
