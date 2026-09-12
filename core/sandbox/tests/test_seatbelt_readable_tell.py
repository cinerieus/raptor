"""The seatbelt read allowlist must not carry the pid1-shim path.

The pid1-shim read grant exists for the Linux unshare fallback lane
(Landlock must allow exec of the shim there); the seatbelt lane never
runs it. The mount-ns lane already filters the entry as a pure
framework/install-location tell — on darwin the leak is worse: every
readable path is embedded verbatim in the SBPL profile text, which
rides ``sandbox-exec -p`` in the never-exec'd watcher shim's argv for
the whole run. The hardened profiles' sysctl-read allowlist-deny
closes the in-sandbox KERN_PROCARGS2 read of that argv; same-UID
observers outside the sandbox can still ``ps`` it, so the RAPTOR
checkout path must simply never appear in the profile.
"""

from __future__ import annotations

import subprocess
import sys
from unittest import mock

import pytest

pytestmark = pytest.mark.skipif(
    sys.platform == "win32", reason="POSIX-only",
)


def _captured_backend_kwargs(tmp_path, **run_kwargs):
    from core.sandbox import _macos_spawn as macos_mod
    from core.sandbox import context
    captured = {}

    def fake_run(cmd, **kwargs):
        captured.update(kwargs)
        cp = subprocess.CompletedProcess(cmd, returncode=0,
                                         stdout="", stderr="")
        cp._setup_status = None
        cp.sandbox_info = {"backend": "macos-seatbelt"}
        return cp

    with mock.patch.object(sys, "platform", "darwin"), \
         mock.patch.object(context, "check_seatbelt_available",
                           return_value=True), \
         mock.patch.object(context, "check_mount_available",
                           return_value=False), \
         mock.patch.object(context, "check_net_available",
                           return_value=False), \
         mock.patch.object(macos_mod, "run_sandboxed", fake_run):
        context.run(["/usr/bin/true"], target=str(tmp_path),
                    output=str(tmp_path), timeout=30, **run_kwargs)
    return captured


def test_pid1_shim_entry_filtered_from_seatbelt_reads(tmp_path):
    kwargs = _captured_backend_kwargs(tmp_path, restrict_reads=True)
    readable = kwargs.get("readable_paths") or []
    # The restricted allowlist is otherwise intact...
    assert any(p == "/usr" for p in readable), readable
    assert str(tmp_path) in readable  # target stays readable
    # ...but the framework tell is gone (the mount lane's filter,
    # mirrored).
    assert not any(p.endswith("/libexec/raptor-pid1-shim")
                   for p in readable), readable


def test_caller_readable_paths_survive_the_filter(tmp_path):
    extra = tmp_path / "toolchain"
    extra.mkdir()
    kwargs = _captured_backend_kwargs(
        tmp_path, restrict_reads=True, readable_paths=[str(extra)])
    readable = kwargs.get("readable_paths") or []
    assert str(extra) in readable
    assert not any(p.endswith("/libexec/raptor-pid1-shim")
                   for p in readable)
