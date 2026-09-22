"""Cross-platform process management for integration tests.

Provides start_server_process() and stop_server_process() that handle
Windows (CREATE_NEW_PROCESS_GROUP + TerminateProcess) vs Unix (setsid +
killpg) transparently. All integration tests should use these helpers
instead of raw os.setsid/os.killpg calls.
"""

import os
import platform
import signal
import socket
import subprocess
import sys

IS_WINDOWS = platform.system() == "Windows"


def start_server_process(cmd, env=None, cwd=None, text=False,
                         merge_stderr=False):
    """Start a subprocess with platform-appropriate process group handling.

    On Unix: uses preexec_fn=os.setsid for process-group cleanup.
    On Windows: uses CREATE_NEW_PROCESS_GROUP for Ctrl+Break signaling.

    Args:
        cmd: Command list (e.g., [SERVER_BIN, "--config", "config/server.yaml"])
        env: Environment dict (default: inherits current)
        cwd: Working directory (default: current)
        text: If True, stdout/stderr are text mode (default: binary)
        merge_stderr: If True, redirect stderr to stdout

    Returns:
        subprocess.Popen instance
    """
    effective_env = os.environ if env is None else env
    port = int(effective_env.get("INFERFLUX_PORT_OVERRIDE", "0"))
    if port:
        host = effective_env.get("INFERFLUX_HOST_OVERRIDE", "127.0.0.1")
        family = socket.AF_INET6 if ":" in host else socket.AF_INET
        # Never probe readiness or send admin requests to an existing listener.
        # Bind rather than connect, so even the guard sends no traffic to it.
        with socket.socket(family, socket.SOCK_STREAM) as reservation:
            if IS_WINDOWS:
                reservation.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
            else:
                # Match the server's restart behavior without SO_REUSEPORT:
                # TIME_WAIT is harmless, but an active listener is rejected.
                reservation.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                reservation.bind((host, port))
            except OSError as error:
                raise RuntimeError(
                    f"integration port {host}:{port} already in use or unavailable"
                ) from error

    kwargs = {
        "env": env,
        "cwd": cwd,
        "stdout": subprocess.PIPE,
    }

    if merge_stderr:
        kwargs["stderr"] = subprocess.STDOUT
    else:
        kwargs["stderr"] = subprocess.PIPE

    if text:
        kwargs["text"] = True

    if IS_WINDOWS:
        kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        kwargs["preexec_fn"] = os.setsid

    return subprocess.Popen(cmd, **kwargs)


def stop_server_process(proc, timeout=5, force_timeout=5):
    """Stop a subprocess started with start_server_process().

    On Unix: sends SIGTERM to the process group, then SIGKILL on timeout.
    On Windows: calls terminate() (TerminateProcess), then kill() on timeout.

    Args:
        proc: subprocess.Popen instance
        timeout: Seconds to wait after graceful termination request
        force_timeout: Seconds to wait after forced kill
    """
    if proc is None or proc.poll() is not None:
        return

    try:
        if IS_WINDOWS:
            proc.terminate()
        else:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        try:
            if IS_WINDOWS:
                proc.kill()
            else:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            proc.wait(timeout=force_timeout)
        except Exception:
            pass
    except Exception:
        # Process may have already exited
        pass
