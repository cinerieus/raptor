"""Exec-status protocol integrity: no silent fail-closed child exits,
no unknown-category fall-through.

The parent treats EOF-with-no-byte on the exec-status pipe as "the
target execed", so a fail-closed child ``os._exit`` that skips the
status write turns an aborted SETUP into a genuine-looking target
result — rc=127/126 collide with the shell not-found/not-executable
conventions and feed downstream returncode oracles fabricated target
behaviour. Two invariants pinned here:

1. Every fail-closed child abort (unusable cwd=, extra_ro bind
   failure, mandatory RLIMIT_CORE) writes category 'C' first, and the
   parent raises the typed SandboxSetupError instead of returning a
   CompletedProcess.
2. The parent default-DENIES status categories it does not recognise
   (the old fall-through was the same default-allow shape that let a
   new demotion lane ship ungated).
"""

import subprocess
import sys
from pathlib import Path

import pytest

from core.sandbox.errors import SandboxSetupError

_REPO_ROOT = Path(__file__).resolve().parents[3]


# ---------------------------------------------------------- unit tier

def test_c_status_byte_roundtrip():
    from core.sandbox._spawn import _parse_setup_status
    parsed = _parse_setup_status(b"C:cwd '/nope' unusable inside sandbox")
    assert parsed == ("C", "cwd '/nope' unusable inside sandbox")


def test_extra_ro_bind_error_preserves_errno():
    import errno

    from core.sandbox.mount_ns import ExtraRoBindError
    exc = ExtraRoBindError(errno.EINVAL, "extra_ro_paths bind failed "
                                         "for '/opt/tool'")
    assert isinstance(exc, OSError)
    assert exc.errno == errno.EINVAL


def test_fail_closed_child_sites_write_status_bytes():
    """Source pin: the three fail-closed child aborts that used to
    ``os._exit`` with NO status byte now report before exiting —
    cwd and RLIMIT_CORE write 'C' directly; the extra_ro bind site
    raises the typed error that _spawn's setup handler categorises as
    'C' (with the pin-tamper 'P' check outranking it)."""
    spawn_src = (_REPO_ROOT / "core/sandbox/_spawn.py").read_text(
        encoding="utf-8")
    # cwd site: the status write precedes the exit.
    cwd_at = spawn_src.index("unusable inside sandbox ")
    region = spawn_src[cwd_at - 600:cwd_at]
    assert '_write_setup_status(' in region and 'b"C"' in region, (
        "the bad-cwd abort no longer writes its status byte")
    # RLIMIT_CORE site.
    core_at = spawn_src.index("RLIMIT_CORE setrlimit failed")
    assert 'b"C"' in spawn_src[core_at - 600:core_at], (
        "the RLIMIT_CORE abort no longer writes its status byte")
    # extra_ro site: typed raise instead of a bare exit, mapped to 'C'
    # in _spawn after the 'P' tamper check.
    mns_src = (_REPO_ROOT / "core/sandbox/mount_ns.py").read_text(
        encoding="utf-8")
    assert "raise ExtraRoBindError(" in mns_src
    assert "SANDBOX_EXIT_MOUNT_NS_BIND_FAIL" not in mns_src, (
        "the extra_ro fail-closed abort regressed to a bare os._exit")
    p_at = spawn_src.index("_PIN_TAMPER_ERRNO):", 2000)
    c_map = spawn_src[p_at:p_at + 800]
    assert "_ExtraRoBindError" in c_map and 'b"C"' in c_map, (
        "_spawn no longer maps the extra_ro failure to category 'C'")


def test_parent_default_denies_unknown_status_category(
        tmp_path, monkeypatch):
    """A status byte from a future writer that this parent does not
    recognise must fail loud, never fall through as a genuine target
    result."""
    from core.sandbox import _spawn as _spawn_mod
    from core.sandbox import context as _ctx

    def fake_spawn(cmd, **kwargs):
        cp = subprocess.CompletedProcess(cmd, returncode=0,
                                         stdout="", stderr="")
        cp._setup_status = ("Z", "from a future status writer")
        return cp

    monkeypatch.setattr(_spawn_mod, "run_sandboxed", fake_spawn)
    try:
        with pytest.raises(SandboxSetupError) as excinfo:
            _ctx.run(["true"], target=str(tmp_path),
                     output=str(tmp_path), timeout=60)
    except (pytest.skip.Exception, pytest.fail.Exception):
        raise
    except Exception as e:  # noqa: BLE001 — host can't reach the lane
        pytest.skip(f"mount-ns lane unavailable: {e}")
    assert "unrecognised setup-status category 'Z'" in str(excinfo.value)
    assert excinfo.value.setup_category == "Z"


def test_parent_raises_typed_error_on_c_status(tmp_path, monkeypatch):
    """The 'C' category is fail-loud with no degrade path: no mountless
    retry, no fallback lane, the failed result never returned."""
    from core.sandbox import _spawn as _spawn_mod
    from core.sandbox import context as _ctx
    calls = []

    def fake_spawn(cmd, **kwargs):
        calls.append(kwargs)
        cp = subprocess.CompletedProcess(cmd, returncode=127,
                                         stdout="", stderr="")
        cp._setup_status = ("C", "cwd '/gone' unusable inside sandbox")
        return cp

    monkeypatch.setattr(_spawn_mod, "run_sandboxed", fake_spawn)
    try:
        with pytest.raises(SandboxSetupError) as excinfo:
            _ctx.run(["true"], target=str(tmp_path),
                     output=str(tmp_path), timeout=60)
    except (pytest.skip.Exception, pytest.fail.Exception):
        raise
    except Exception as e:  # noqa: BLE001 — host can't reach the lane
        pytest.skip(f"mount-ns lane unavailable: {e}")
    assert excinfo.value.setup_category == "C"
    assert "aborted fail-closed" in str(excinfo.value)
    assert "cwd '/gone'" in str(excinfo.value)
    assert len(calls) == 1, "the 'C' category must not ride any ladder"


# --------------------------------------------------- integration tier

@pytest.mark.integration
@pytest.mark.skipif(sys.platform != "linux", reason="namespace sandbox")
def test_bad_cwd_raises_typed_error_not_fake_result(tmp_path):
    """Live: a cwd= that does not exist inside the sandbox aborts the
    spawn child; the parent must raise the typed category-'C' refusal
    — pre-fix it returned a normal CompletedProcess with rc=127 and
    only a stderr line, feeding returncode oracles a fabricated
    target result."""
    from core.sandbox import context as _ctx
    missing = tmp_path / "no-such-cwd"
    try:
        r = _ctx.run(["/bin/true"], target=str(tmp_path),
                     output=str(tmp_path), cwd=str(missing), timeout=60)
    except SandboxSetupError as e:
        assert e.setup_category == "C", str(e)
        assert "cwd" in str(e)
        return
    except FileNotFoundError:
        # Subprocess-backed lane: Python validates cwd in the parent
        # and raises before exec — already loud; the spawn lane was
        # the silent one. Nothing to test on this host shape.
        pytest.skip("spawn lane not taken (subprocess cwd validation)")
    pytest.fail(
        f"bad cwd came back as a genuine result: rc={r.returncode} "
        f"(status-byte protocol regressed)")


@pytest.mark.integration
@pytest.mark.skipif(sys.platform != "linux", reason="namespace sandbox")
def test_rlimit_core_failure_raises_typed_error(tmp_path, monkeypatch):
    """Live: a mandatory RLIMIT_CORE failure in the spawn child aborts
    with category 'C' instead of a bare rc=99 result. Injected via a
    fork-inherited resource.setrlimit wrapper that refuses exactly
    RLIMIT_CORE."""
    import resource

    from core.sandbox import context as _ctx
    real = resource.setrlimit

    def refusing(res, limits):
        if res == resource.RLIMIT_CORE:
            raise OSError(1, "injected RLIMIT_CORE refusal")
        return real(res, limits)

    monkeypatch.setattr(resource, "setrlimit", refusing)
    try:
        r = _ctx.run(["/bin/true"], target=str(tmp_path),
                     output=str(tmp_path), timeout=60)
    except SandboxSetupError as e:
        assert e.setup_category == "C", str(e)
        assert "RLIMIT_CORE" in str(e)
        return
    except (pytest.skip.Exception, pytest.fail.Exception):
        raise
    except Exception as e:  # noqa: BLE001 — host can't reach the lane
        pytest.skip(f"spawn lane unavailable: {e}")
    # Subprocess lanes apply rlimits in preexec (different mechanism,
    # out of this protocol's scope) — only the spawn lane must raise.
    if r.sandbox_info.get("mount_ns_active") or (
            r.sandbox_info.get("backend") == "landlock-pidns"):
        pytest.fail(
            f"spawn-lane RLIMIT_CORE failure came back as a genuine "
            f"result: rc={r.returncode}")
    pytest.skip("spawn lane not taken on this host")
