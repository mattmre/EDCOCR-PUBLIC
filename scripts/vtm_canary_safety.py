"""Shared guardrails for live VTM canaries that handle lab credentials."""

from __future__ import annotations

import ipaddress
import json
import os
import re
import tempfile
import urllib.request
import uuid
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler


class _RejectRedirects(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):  # type: ignore[no-untyped-def]
        return None


def require_loopback_http_url(value: str, *, label: str = "API URL") -> str:
    """Reject non-loopback targets before a canary sends a lab API key."""
    try:
        parsed = urlsplit(value)
        hostname = parsed.hostname
        # Accessing .port validates malformed and out-of-range port values.
        parsed.port
    except (AttributeError, ValueError) as exc:
        raise ValueError(f"{label} must be an absolute loopback HTTP(S) URL") from exc

    if (
        parsed.scheme not in {"http", "https"}
        or not hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError(f"{label} must be an absolute loopback HTTP(S) URL")

    normalized_host = hostname.rstrip(".").lower()
    try:
        address = ipaddress.ip_address(normalized_host)
    except ValueError:
        loopback = normalized_host == "localhost"
    else:
        loopback = address.is_loopback
    if not loopback:
        raise ValueError(f"{label} host must be loopback; refusing {hostname!r}")
    return value.rstrip("/")


def require_isolated_kind_identity(
    context: str, cluster: str, node_names: list[str]
) -> str:
    """Validate a kind kubeconfig identity and return its actual cluster name.

    Kind stores the cluster object in kubeconfig as ``kind-<cluster-name>``,
    while Docker labels and node names use the unprefixed cluster name.
    """
    cluster_name = cluster.removeprefix("kind-")
    if not re.fullmatch(r"ocr-closeout-c7(?:-[0-9]{8})?", cluster_name):
        raise ValueError(f"unexpected isolated kind cluster name: {cluster!r}")
    if context != f"kind-{cluster_name}":
        raise ValueError(
            f"kube context {context!r} does not exactly match kind-{cluster_name}"
        )
    if not node_names or any(
        not name.startswith(f"{cluster_name}-") for name in node_names
    ):
        raise ValueError(
            f"kind node names do not match isolated cluster {cluster_name!r}"
        )
    return cluster_name


def resolve_fresh_directory(path: Path, *, root: Path | None = None) -> Path:
    """Validate and resolve an empty evidence path without creating it."""
    path = path.expanduser()
    if path.is_symlink():
        raise ValueError(f"evidence directory must not be a symlink: {path}")
    resolved = path.resolve()
    if root is not None and not resolved.is_relative_to(
        root.expanduser().resolve(strict=True)
    ):
        raise ValueError(f"evidence directory must be inside {root}")
    path = resolved
    if path.exists():
        if not path.is_dir():
            raise ValueError(f"evidence path is not a directory: {path}")
        if any(path.iterdir()):
            raise ValueError(f"evidence directory is not empty: {path}")
    return path


def require_fresh_directory(path: Path, *, root: Path | None = None) -> Path:
    """Create or accept an empty non-symlink directory for one evidence run."""
    path = resolve_fresh_directory(path, root=root)
    if not path.exists():
        path.mkdir(parents=True, exist_ok=False)
    return path.resolve(strict=True)


def require_uuid_job_id(value: Any) -> str:
    """Validate API job IDs before using them in paths or further requests."""
    if not isinstance(value, str):
        raise ValueError("API returned a non-string job ID")
    try:
        parsed = uuid.UUID(value)
    except (ValueError, AttributeError) as exc:
        raise ValueError("API returned a non-UUID job ID") from exc
    canonical = str(parsed)
    if value.lower() != canonical:
        raise ValueError("API returned a non-canonical UUID job ID")
    return canonical


def require_local_api_job_id(value: Any) -> str:
    """Validate the FastAPI job_ + 12-hex ID before path or URL use."""
    if not isinstance(value, str) or re.fullmatch(r"job_[0-9a-f]{12}", value) is None:
        raise ValueError("API returned a non-canonical local job ID")
    return value


def open_no_redirect(request: urllib.request.Request, *, timeout: float):
    """Open directly without proxying or forwarding headers across redirects."""
    opener = urllib.request.build_opener(
        urllib.request.ProxyHandler({}), _RejectRedirects
    )
    return opener.open(request, timeout=timeout)


def _exclusive_write(path: Path, data: bytes, *, mode: int = 0o644) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temp_path = Path(temporary)
    try:
        os.fchmod(fd, mode)
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        # A hard link publishes atomically and fails if the destination exists.
        os.link(temp_path, path)
        dir_fd = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)
    finally:
        try:
            temp_path.unlink()
        except FileNotFoundError:
            pass


def write_bytes_no_replace(path: Path, data: bytes) -> None:
    _exclusive_write(path, data)


def write_json_no_replace(path: Path, value: Any) -> None:
    data = json.dumps(value, indent=2, sort_keys=True).encode("utf-8") + b"\n"
    _exclusive_write(path, data)
