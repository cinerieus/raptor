"""Shared fixtures for dispatcher tests.

Proxy-env hermeticity: these tests spin up loopback fake upstreams
(Bedrock runtime doubles, provider echo servers) and forward to them
through the dispatcher's httpx client, which honours proxy env
(trust_env). On a mandatory-egress-proxy host the live HTTPS_PROXY in
the test process routed every "upstream" call to the corporate proxy
— which cannot reach this machine's loopback — so 30 tests failed
with 403s / missing captures while passing on unproxied machines.

Scrub the whole conventional proxy family for every test in this
directory. Tests that exercise proxy propagation explicitly (e.g.
test_f085_spawn_default_env) re-set the vars with monkeypatch inside
the test body, which runs after this autouse fixture — unaffected.

Same class of fix as the hermetic-proxy-host-tests patch (see
core/orchestration/tests/test_agentic_passes.py).

AWS-env hermeticity, same reasoning: CredentialStore reads the
ambient env at construction and deliberately prefers a profile chain
(AWS_PROFILE — refresh-capable) over explicit keys. On a
Bedrock-configured host that made the SigV4 signing tests resolve the
host's REAL credentials through a network-touching botocore chain
instead of the seeded fakes — signature assertions failed (or hung
once the proxy env was scrubbed). Tests seed exactly the credentials
they mean to test; the ambient family is noise here.
"""

import os
import shutil
import tempfile
from collections.abc import Iterator

import pytest

_PROXY_ENV_FAMILY = (
    "HTTP_PROXY", "HTTPS_PROXY", "NO_PROXY", "ALL_PROXY",
    "http_proxy", "https_proxy", "no_proxy", "all_proxy",
)

_AWS_ENV_FAMILY = (
    "AWS_PROFILE", "AWS_REGION", "AWS_DEFAULT_REGION",
    "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN",
    "AWS_BEARER_TOKEN_BEDROCK", "AWS_ENDPOINT_URL_BEDROCK",
    "CLAUDE_CODE_USE_BEDROCK",
)

# Snapshot at import time (before any scrubbing) so tests that
# genuinely need the operator's real egress route can opt back in.
_OPERATOR_PROXY_ENV = {
    k: v for k in _PROXY_ENV_FAMILY if (v := os.environ.get(k)) is not None
}


@pytest.fixture(autouse=True)
def _scrub_ambient_env(monkeypatch):
    for var in (*_PROXY_ENV_FAMILY, *_AWS_ENV_FAMILY):
        monkeypatch.delenv(var, raising=False)


# AF_UNIX hermeticity: every real-dispatcher test binds two Unix
# sockets under a ``tempfile.mkdtemp(prefix="raptor-llm-<run_id>-")``
# dir, and Linux caps a socket path at 108 bytes (sun_path, incl. NUL).
# The session-wide TMPDIR containment (root conftest) nests scratch
# one level deeper, and on hosts whose ambient TMPDIR is itself a
# nested per-session dir the combined prefix pushes
# ``.../raptor-llm-<run_id>-XXXXXXXX/llm-child.sock`` past the cap —
# every dispatcher construction then dies with "AF_UNIX path too
# long" while short-/tmp hosts (CI) pass. Budget below the cap for
# the dispatcher's own suffix: "/raptor-llm-" + run_id (headroom 40;
# longest in this dir is 31) + "-XXXXXXXX" + "/llm-child.sock".
_AF_UNIX_PATH_MAX = 107  # usable bytes (108 incl. the trailing NUL)
_SOCKET_SUFFIX_BUDGET = len("/raptor-llm-") + 40 + len("-XXXXXXXX") + len(
    "/llm-child.sock")
_SAFE_TMP_LEN = _AF_UNIX_PATH_MAX - _SOCKET_SUFFIX_BUDGET


@pytest.fixture(autouse=True)
def _af_unix_safe_tmp(monkeypatch) -> Iterator[None]:
    """Re-root scratch at a short ``/tmp`` dir when the contained
    TMPDIR would blow the AF_UNIX path cap.

    No-op on short-tmp hosts, so the session containment (and its
    kill-leak story) is preserved there; on long-tmp hosts this is
    the root conftest's documented AF_UNIX exception (sites that must
    not be contained anchor at ``dir="/tmp"``). Cleanup is the
    ``finally`` rmtree below; a SIGKILLed session leaks the dir, and
    on exactly these deep-TMPDIR hosts core/run/tmp_reaper.py does
    NOT reclaim it — the sweep covers only ``tempfile.gettempdir()``,
    which here is the deep dir, not ``/tmp``. The ``raptor-llm-``
    prefix listing only helps when some later session runs with
    ``gettempdir() == /tmp`` (CI, short-tmp hosts) and sweeps it up.
    """
    if len(tempfile.gettempdir()) <= _SAFE_TMP_LEN:
        yield
        return
    short_tmp = tempfile.mkdtemp(prefix="raptor-llm-sock-", dir="/tmp")
    monkeypatch.setenv("TMPDIR", short_tmp)
    monkeypatch.setattr(tempfile, "tempdir", short_tmp)
    try:
        yield
    finally:
        shutil.rmtree(short_tmp, ignore_errors=True)


@pytest.fixture
def operator_proxy_env(monkeypatch):
    """Opt-back-in for tests that reach REAL external upstreams
    (e.g. the valid-token gate test that forwards to anthropic.com):
    restores the operator's launch-time proxy route that the autouse
    scrub removed. On unproxied hosts this is a no-op."""
    for k, v in _OPERATOR_PROXY_ENV.items():
        monkeypatch.setenv(k, v)
    return dict(_OPERATOR_PROXY_ENV)
