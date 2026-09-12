"""Run-scoped health gate for the Joern verification channel.

A restarting or CPG-less Joern server fails every dispatch — either
instantly (fail-fast branches) or after a full client-side timeout —
but the tool chain keeps offering joern steps to every hypothesis, so
one bad restart can turn into hundreds of dead-channel round trips
over a run while producing zero receipts.  The gate counts
CONSECUTIVE dispatch failures across all joern tool types; once the
threshold is crossed the channel reports itself unhealthy.  Dispatch
sites consult :func:`dispatch_blocked` and skip (counted as
``skipped`` in tier diagnostics AND recorded in the caller's
``skipped_types`` so the channel leaves the dispatch record), and the
trip is surfaced in ``tier-diagnostics.json`` and the report's
degradation section so the missing receipts read as "channel down",
never as refutations.

Two robustness rules keep the gate from silencing a healthy server:

* **Distinct-key trip rule** — the failure streak must span at least
  :data:`MIN_DISTINCT_TRIP_KEYS` distinct dispatch keys
  (``file:function``).  A pile of failures against one function can
  be a single pathological query shape (one bad identifier or CWE
  class) on a perfectly healthy server; a dead server fails
  everything it is offered.
* **Half-open re-probe** — after a trip, exactly ONE dispatch is let
  through once :data:`REPROBE_AFTER_SKIPS` dispatches have been
  skipped.  A success clears the trip (the server recovered — e.g.
  its CPG re-import finished); a failure re-seals the gate for the
  rest of the run.  One probe per run: recovery gets a bounded
  chance, a flapping server does not get to re-open the burn.
"""

from __future__ import annotations

import logging
import threading
from typing import Any

logger = logging.getLogger(__name__)

# Consecutive dispatch failures before the channel trips.  Too low and
# a transient blip (one stuck query while the server restarts and
# recovers) silences a channel that would have answered the rest of
# the run; too high and a genuinely dead server burns its full client
# timeout per hypothesis for most of the run before the gate helps.
# Distinct hypotheses dispatch independently, so a healthy server
# interleaves successes and resets the streak long before 8.
DEFAULT_UNHEALTHY_AFTER = 8

# The failure streak must span at least this many distinct dispatch
# keys before the gate trips.  Lower (1) lets one pathological
# function/query shape silence the channel for the whole run; higher
# makes a genuinely dead server burn failures across more hypotheses
# before the gate helps.  Two is enough to prove "not just that one
# query" while a dead server crosses it immediately.
MIN_DISTINCT_TRIP_KEYS = 2

# Skipped dispatches before the single half-open probe is allowed.
# Smaller re-probes while a multi-minute CPG re-import is still
# likely in flight (the probe then burns its one chance on a
# still-restarting server); larger leaves a recovered server dark for
# more of the run than necessary.
REPROBE_AFTER_SKIPS = 25


class JoernChannelHealth:
    """Thread-safe consecutive-failure tracker for the joern channel.

    Dispatch sites ask :meth:`allow_dispatch` before dialing (it
    consumes the half-open probe when tripped), then call
    :meth:`record_error` on a failed round trip and
    :meth:`record_success` on any completed one (confirmed, refuted
    or inconclusive — the outcome does not matter, reaching the
    server does).  Workers dispatch in parallel, so all state
    mutations take the instance lock.
    """

    def __init__(self, unhealthy_after: int = DEFAULT_UNHEALTHY_AFTER) -> None:
        self.unhealthy_after: int = max(1, int(unhealthy_after))
        self._lock = threading.Lock()
        self._consecutive_errors: int = 0
        self._streak_keys: set[str] = set()
        self._total_errors: int = 0
        self._total_successes: int = 0
        self._tripped: bool = False
        self._trip_reason: str | None = None
        self._skips_since_trip: int = 0
        self._probe_spent: bool = False
        self._recovered_once: bool = False

    def allow_dispatch(self) -> bool:
        """True when the channel may dispatch.

        Healthy: always.  Tripped: counts the skip and grants the one
        half-open probe once enough skips have accumulated.  A granted
        probe that the caller then does not dispatch (e.g. its own
        deadline clamp skips the query) is simply spent — bounded
        waste, never a wedged gate.
        """
        with self._lock:
            if not self._tripped:
                return True
            self._skips_since_trip += 1
            if (
                not self._probe_spent
                and self._skips_since_trip >= REPROBE_AFTER_SKIPS
            ):
                self._probe_spent = True
                logger.info(
                    "joern channel half-open probe: one dispatch "
                    "allowed through the tripped gate (%d skips since "
                    "trip)", self._skips_since_trip,
                )
                return True
            return False

    def record_error(self, detail: str = "", key: str = "") -> None:
        """Record a failed joern round trip; trips the gate when the
        streak crosses the threshold across distinct keys (loud,
        once).  While tripped, a failure is the half-open probe
        failing — the gate stays sealed."""
        with self._lock:
            self._total_errors += 1
            self._consecutive_errors += 1
            if key:
                self._streak_keys.add(key)
            if self._tripped:
                return
            if self._consecutive_errors < self.unhealthy_after:
                return
            if len(self._streak_keys) < MIN_DISTINCT_TRIP_KEYS:
                return
            self._tripped = True
            self._trip_reason = (
                f"{self._consecutive_errors} consecutive dispatch "
                f"failures across {len(self._streak_keys)} functions"
                + (f" (last: {detail[:200]})" if detail else "")
            )
        logger.warning(
            "joern channel unhealthy — %s; joern dispatches for this "
            "run are skipped (skipped, not refuted — see the report's "
            "degradation section; one half-open re-probe after %d "
            "skips)",
            self._trip_reason, REPROBE_AFTER_SKIPS,
        )

    def record_success(self) -> None:
        """Record a completed round trip (any verdict) — resets the
        consecutive-failure streak.  While tripped, a success is the
        half-open probe succeeding: the server recovered, clear the
        trip."""
        recovered = False
        with self._lock:
            self._total_successes += 1
            self._consecutive_errors = 0
            self._streak_keys.clear()
            if self._tripped:
                self._tripped = False
                self._recovered_once = True
                self._skips_since_trip = 0
                recovered = True
        if recovered:
            logger.info(
                "joern channel recovered — half-open probe completed; "
                "dispatch re-enabled (a second trip is final for the "
                "run)",
            )

    @property
    def tripped(self) -> bool:
        with self._lock:
            return self._tripped

    @property
    def trip_reason(self) -> str | None:
        with self._lock:
            return self._trip_reason

    @property
    def total_errors(self) -> int:
        with self._lock:
            return self._total_errors

    @property
    def total_successes(self) -> int:
        with self._lock:
            return self._total_successes

    def to_dict(self) -> dict[str, Any]:
        with self._lock:
            data: dict[str, Any] = {
                "tripped": self._tripped,
                "trip_reason": self._trip_reason,
                "consecutive_errors": self._consecutive_errors,
                "total_errors": self._total_errors,
                "total_successes": self._total_successes,
                "unhealthy_after": self.unhealthy_after,
            }
            if self._skips_since_trip:
                data["skips_since_trip"] = self._skips_since_trip
            if self._probe_spent:
                data["probe_spent"] = True
            if self._recovered_once:
                data["recovered_once"] = True
            return data


def channel_unhealthy(config: Any) -> bool:
    """True when *config* carries a currently-tripped joern gate.

    Read-only (never consumes the half-open probe) — for spend gates
    and reporting.  ``getattr``-tolerant: minimal test configs and
    external callers without the field read as healthy (the pre-gate
    behaviour).
    """
    health = getattr(config, "joern_health", None)
    return health is not None and health.tripped


def dispatch_blocked(config: Any) -> bool:
    """Consuming dispatch-permission check for joern dispatch sites.

    False = dispatch (healthy, or the half-open probe was granted);
    True = skip.  ``getattr``-tolerant like :func:`channel_unhealthy`.
    """
    health = getattr(config, "joern_health", None)
    if health is None:
        return False
    return not health.allow_dispatch()


def record_outcome(
    config: Any, *, error: bool, detail: str = "", key: str = "",
) -> None:
    """Feed one joern round-trip outcome into *config*'s gate, if any.

    ``key`` (``file:function``) feeds the distinct-key trip rule.
    """
    health = getattr(config, "joern_health", None)
    if health is None:
        return
    if error:
        health.record_error(detail, key=key)
    else:
        health.record_success()


def health_snapshot(config: Any) -> dict[str, dict[str, Any]] | None:
    """Channel-health block for tier diagnostics.

    Only worth writing once the channel saw at least one error —
    healthy silence stays out of the artifact.
    """
    health = getattr(config, "joern_health", None)
    if health is None:
        return None
    if health.total_errors == 0 and not health.tripped:
        return None
    return {"joern": health.to_dict()}
