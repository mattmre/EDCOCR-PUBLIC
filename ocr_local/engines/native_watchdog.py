"""Independent, pidfd-bound deadline guard for native engine calls.

The supervisor never receives OCR inputs. It watches only monotonic deadlines
and may signal only the process identified by the pidfd opened at startup.
"""

from __future__ import annotations

import os
import select
import signal
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path


def _supervise(read_fd: int, pid_fd: int, ack_fd: int) -> int:
    active: dict[str, int] = {}
    buffered = b""

    def kill_owner() -> None:
        signal.pidfd_send_signal(pid_fd, signal.SIGKILL)

    try:
        os.write(ack_fd, b"READY\n")
        while True:
            now = time.monotonic_ns()
            if active and min(active.values()) <= now:
                kill_owner()
                return 1
            timeout = max(0.0, (min(active.values()) - now) / 1e9) if active else None
            readable, _, _ = select.select([read_fd], [], [], timeout)
            if not readable:
                continue
            chunk = os.read(read_fd, 4096)
            if not chunk:
                # The owner exited or intentionally closed its write end.
                return 0
            buffered += chunk
            if len(buffered) > 16384:
                raise ValueError("watchdog command buffer overflow")
            while b"\n" in buffered:
                line, buffered = buffered.split(b"\n", 1)
                parts = line.decode("ascii").split(" ")
                if parts == ["STOP"]:
                    return 0
                if len(parts) == 3 and parts[0] == "ARM" and len(parts[1]) == 32:
                    deadline = int(parts[2])
                    if deadline <= 0 or parts[1] in active:
                        raise ValueError("invalid native deadline")
                    active[parts[1]] = deadline
                    os.write(ack_fd, f"ARMED {parts[1]}\n".encode("ascii"))
                elif len(parts) == 2 and parts[0] == "DISARM" and parts[1] in active:
                    del active[parts[1]]
                else:
                    raise ValueError("invalid watchdog command")
    except BaseException:
        if active:
            kill_owner()
        return 1
    finally:
        os.close(read_fd)
        os.close(pid_fd)
        os.close(ack_fd)


class NativeCallWatchdog:
    """One reusable subprocess watching native calls in this process only."""

    def __init__(self, timeout_seconds: float) -> None:
        if not 0 < timeout_seconds <= 3600:
            raise ValueError("native call timeout must be in (0, 3600]")
        if os.getpid() == 1:
            raise RuntimeError("native watchdog requires an init parent; OCR owner cannot be PID 1")
        if not hasattr(os, "pidfd_open") or not hasattr(signal, "pidfd_send_signal"):
            raise RuntimeError("native watchdog requires Linux pidfd signaling")
        self.timeout_seconds = timeout_seconds
        self._owner_pid = os.getpid()
        self._lock = threading.Lock()
        self._write_fd: int | None = None
        self._ack_fd: int | None = None
        self._process: subprocess.Popen[bytes] | None = None

    def _start_locked(self) -> None:
        read_fd, write_fd = os.pipe2(os.O_CLOEXEC)
        ack_read_fd, ack_write_fd = os.pipe2(os.O_CLOEXEC)
        try:
            pid_fd = os.pidfd_open(self._owner_pid)
        except BaseException:
            os.close(read_fd)
            os.close(write_fd)
            os.close(ack_read_fd)
            os.close(ack_write_fd)
            raise
        try:
            process = subprocess.Popen(
                [sys.executable, str(Path(__file__).resolve()), str(read_fd), str(pid_fd), str(ack_write_fd)],
                pass_fds=(read_fd, pid_fd, ack_write_fd), stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
        except BaseException:
            os.close(write_fd)
            os.close(ack_read_fd)
            raise
        finally:
            os.close(read_fd)
            os.close(pid_fd)
            os.close(ack_write_fd)
        self._process = process
        self._write_fd = write_fd
        self._ack_fd = ack_read_fd
        try:
            self._expect_ack("READY")
        except BaseException:
            os.close(write_fd)
            os.close(ack_read_fd)
            self._write_fd = None
            self._ack_fd = None
            process.kill()
            process.wait(timeout=3)
            self._process = None
            raise

    def _expect_ack(self, expected: str) -> None:
        if self._ack_fd is None:
            raise RuntimeError("native watchdog acknowledgement pipe unavailable")
        readable, _, _ = select.select([self._ack_fd], [], [], 3)
        if not readable:
            raise RuntimeError("native watchdog did not acknowledge startup/arm")
        observed = os.read(self._ack_fd, 128)
        if observed != f"{expected}\n".encode("ascii"):
            raise RuntimeError("native watchdog acknowledgement mismatch or EOF")

    def arm(self) -> str:
        with self._lock:
            if self._owner_pid != os.getpid():
                raise RuntimeError("inherited native watchdog must not be reused")
            if self._process is None:
                self._start_locked()
            elif self._process.poll() is not None:
                raise RuntimeError("native watchdog exited; refusing engine call")
            token = uuid.uuid4().hex
            deadline = time.monotonic_ns() + int(self.timeout_seconds * 1e9)
            self._send_locked(f"ARM {token} {deadline}\n")
            self._expect_ack(f"ARMED {token}")
            return token

    def disarm(self, token: str) -> None:
        with self._lock:
            self._send_locked(f"DISARM {token}\n")

    def _send_locked(self, command: str) -> None:
        if self._write_fd is None or self._process is None or self._process.poll() is not None:
            raise RuntimeError("native watchdog unavailable")
        try:
            os.write(self._write_fd, command.encode("ascii"))
        except OSError as exc:
            raise RuntimeError("native watchdog pipe unavailable") from exc

    def detach_after_fork(self) -> None:
        # Never acquire inherited locks or wait on the parent's subprocess.
        if self._write_fd is not None:
            os.close(self._write_fd)
        if self._ack_fd is not None:
            os.close(self._ack_fd)
        self._write_fd = None
        self._ack_fd = None
        self._process = None

    def close(self) -> None:
        with self._lock:
            if self._owner_pid != os.getpid():
                self.detach_after_fork()
                return
            if self._process is None:
                return
            if self._write_fd is not None:
                try:
                    self._send_locked("STOP\n")
                except RuntimeError:
                    pass
                os.close(self._write_fd)
                self._write_fd = None
            if self._ack_fd is not None:
                os.close(self._ack_fd)
                self._ack_fd = None
            try:
                self._process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait(timeout=3)
            self._process = None


if __name__ == "__main__":
    raise SystemExit(_supervise(int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])))
