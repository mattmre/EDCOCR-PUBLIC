"""HTTP I/O helpers for presigned URL mode.

Workers use these functions instead of StorageBackend.download_file/upload_file
when operating with presigned URLs. No S3 credentials required -- only stdlib.

Security (R2-RC-01 residual / kwargs trust):
- Scheme must be HTTP(S).
- Link-local / metadata / loopback hosts are rejected unless the operator
  explicitly enables private presigned egress for an allowlisted endpoint.
- ``S3_PRESIGNED_ALLOWED_HOSTS`` or ``S3_ENDPOINT`` must configure an explicit
  host allowlist; an empty allowlist is a startup/configuration error in every
  deployment mode.
- When ``job_id`` is provided, the URL path must contain ``jobs/{job_id}/``
  so a broker-forged URL cannot target another job's objects.
"""

from __future__ import annotations

import logging
import os
import urllib.error
import urllib.request
from collections.abc import Callable
from urllib.parse import unquote, urlparse

from ocr_distributed.ssrf import NoRedirectHandler
from ocr_distributed.ssrf import pinned_url_for_request as _pinned_url_for_request

from .storage import _assert_regular_single_link, _open_local_parent

logger = logging.getLogger(__name__)

_HTTP_TIMEOUT = 30
_DEFAULT_MAX_PRESIGNED_TRANSFER_BYTES = 5 * 1024 * 1024 * 1024


class PresignedTransferError(RuntimeError):
    """A transport failure whose message never embeds a bearer URL."""


def _open_presigned(request, timeout):
    try:
        return _no_redirect_opener.open(request, timeout=timeout)
    except urllib.error.HTTPError as exc:
        raise PresignedTransferError(f"presigned HTTP status {exc.code}") from None
    except urllib.error.URLError as exc:
        raise PresignedTransferError(
            f"presigned transport failure ({type(exc.reason).__name__})"
        ) from None

# Cloud metadata / link-local hosts that must never be contacted (SSRF).
_METADATA_BLOCKED_HOSTS = frozenset(
    {
        "169.254.169.254",
        "metadata.google.internal",
        "metadata.google.internal.",
        "metadata",
        "fd00:ec2::254",
    }
)
# Loopback is blocked unless explicitly present in the allowlist (lab MinIO/E2E).
_LOOPBACK_HOSTS = frozenset(
    {
        "localhost",
        "127.0.0.1",
        "0.0.0.0",
        "::1",
        "[::1]",
    }
)


def _allowed_presigned_hosts() -> set[str]:
    """Return the configured presigned host allowlist (lowercased)."""
    hosts: set[str] = set()
    raw = os.environ.get("S3_PRESIGNED_ALLOWED_HOSTS", "").strip()
    if raw:
        for part in raw.split(","):
            h = part.strip().lower()
            if h:
                hosts.add(h)
    endpoint = os.environ.get("S3_ENDPOINT", "").strip()
    if not endpoint:
        try:
            from django.conf import settings

            endpoint = str(getattr(settings, "S3_ENDPOINT", "") or "").strip()
        except Exception:
            # Standalone worker imports may not have Django configured yet.
            endpoint = ""
    if endpoint:
        try:
            ep = endpoint if "://" in endpoint else f"https://{endpoint}"
            parsed = urlparse(ep)
            if parsed.hostname:
                hosts.add(parsed.hostname.lower())
                if parsed.port:
                    hosts.add(f"{parsed.hostname.lower()}:{parsed.port}")
        except Exception:
            logger.debug(
                "Could not parse S3_ENDPOINT for presigned allowlist", exc_info=True
            )
    return hosts


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes", "on")


def _hostport(host: str, port: int) -> str:
    """Format a normalized host/port pair, including IPv6 brackets."""
    return f"[{host}]:{port}" if ":" in host else f"{host}:{port}"


def _host_matches_allowlist(host: str, parsed, allowed: set[str]) -> bool:
    """Require exact configured port when a URL spells one explicitly."""
    if parsed.port is not None:
        return (
            _hostport(host, parsed.port) in allowed
            or f"{host}:{parsed.port}" in allowed
        )
    default_port = 443 if parsed.scheme == "https" else 80
    return host in allowed or _hostport(host, default_port) in allowed


def _validate_presigned_url(
    url: str,
    *,
    job_id: str | None = None,
    expected_key: str | None = None,
) -> None:
    """Validate URL policy plus its exact job/artifact path binding."""
    if not isinstance(url, str) or not url.strip():
        raise ValueError("Presigned URL must be a non-empty string")
    parsed = urlparse(url)
    if parsed.scheme not in ("https", "http"):
        raise ValueError(f"Presigned URL must use HTTP(S), got: {parsed.scheme!r}")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("Presigned URL userinfo is not allowed")
    if parsed.scheme == "http" and not _env_flag("S3_PRESIGNED_ALLOW_HTTP"):
        raise ValueError(
            "Presigned URL must use HTTPS; set S3_PRESIGNED_ALLOW_HTTP=true "
            "only for an explicitly allowlisted lab endpoint"
        )

    host = (parsed.hostname or "").lower().rstrip(".")
    if not host:
        raise ValueError("Presigned URL missing host")
    if host in _METADATA_BLOCKED_HOSTS or host.startswith("169.254."):
        raise ValueError(f"Presigned URL host not allowed: {host}")

    allowed = _allowed_presigned_hosts()
    hostport = f"{host}:{parsed.port}" if parsed.port else host
    if not allowed:
        raise ValueError(
            "Presigned URL host allowlist is empty; set "
            "S3_PRESIGNED_ALLOWED_HOSTS or S3_ENDPOINT"
        )
    if not _host_matches_allowlist(host, parsed, allowed):
        raise ValueError(f"Presigned URL host not in allowlist: {hostport}")

    if host in _LOOPBACK_HOSTS and not _env_flag("S3_PRESIGNED_ALLOW_PRIVATE"):
        raise ValueError(
            "Private/loopback presigned egress requires S3_PRESIGNED_ALLOW_PRIVATE=true"
        )

    path_parts = [unquote(part) for part in (parsed.path or "").split("/") if part]
    if ".." in path_parts:
        raise ValueError("Presigned URL path must not contain '..' segments")
    if "." in path_parts:
        raise ValueError("Presigned URL path must not contain '.' segments")

    if job_id is not None:
        jid = str(job_id).strip()
        if not jid:
            raise ValueError("job_id for presigned bind must be non-empty")
        if not any(
            part == "jobs"
            and index + 1 < len(path_parts)
            and path_parts[index + 1] == jid
            for index, part in enumerate(path_parts)
        ):
            raise ValueError(
                f"Presigned URL path not bound to job {jid!r} "
                f"(expected path segments ['jobs', {jid!r}])"
            )

    if expected_key is not None:
        if not isinstance(expected_key, str) or not expected_key.strip():
            raise ValueError("expected_key for presigned bind must be non-empty")
        key_parts = [
            unquote(part) for part in expected_key.replace("\\", "/").split("/") if part
        ]
        if any(part in {".", ".."} for part in key_parts):
            raise ValueError("expected_key must not contain dot segments")
        if not key_parts or path_parts[-len(key_parts) :] != key_parts:
            raise ValueError("Presigned URL path is not bound to the expected artifact")


def _pinned_presigned_url(url: str):
    """Pin DNS for one presigned request after URL policy validation."""
    return _pinned_url_for_request(
        url,
        allow_http=_env_flag("S3_PRESIGNED_ALLOW_HTTP"),
        allow_private=_env_flag("S3_PRESIGNED_ALLOW_PRIVATE"),
    )


def _max_presigned_transfer_bytes() -> int:
    """Return the operator-configured cap for one presigned transfer.

    The cap is deliberately read at call time so standalone workers and Django
    workers use the same environment-driven policy.  A missing value keeps the
    historical 5 GiB deployment ceiling, but never permits an unbounded
    response or an unbounded local upload read.
    """
    raw = os.environ.get("S3_PRESIGNED_MAX_TRANSFER_BYTES", "").strip()
    if not raw:
        try:
            from django.conf import settings

            raw = str(
                getattr(settings, "S3_PRESIGNED_MAX_TRANSFER_BYTES", "") or ""
            ).strip()
        except Exception:
            raw = ""
    if not raw:
        return _DEFAULT_MAX_PRESIGNED_TRANSFER_BYTES
    try:
        value = int(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "S3_PRESIGNED_MAX_TRANSFER_BYTES must be a positive integer"
        ) from exc
    if value <= 0:
        raise ValueError("S3_PRESIGNED_MAX_TRANSFER_BYTES must be positive")
    return value


def _declared_content_length(response) -> int | None:
    """Read a valid Content-Length header without trusting mock-like objects."""
    headers = getattr(response, "headers", None)
    value = None
    if headers is not None and hasattr(headers, "get"):
        value = headers.get("Content-Length")
    if value is None and hasattr(response, "getheader"):
        value = response.getheader("Content-Length")
    if value is None or not isinstance(value, (str, bytes, int)):
        return None
    try:
        length = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("Presigned response has an invalid Content-Length") from exc
    if length < 0:
        raise ValueError("Presigned response has a negative Content-Length")
    return length


def _prepare_local_destination(path: str) -> tuple[str, int, str]:
    """Return an absolute destination and an open, no-following parent FD."""
    parent_fd = -1
    try:
        parent_fd, basename, absolute = _open_local_parent(path)
        try:
            existing = os.stat(basename, dir_fd=parent_fd, follow_symlinks=False)
        except FileNotFoundError:
            existing = None
        if existing is not None:
            _assert_regular_single_link(existing, label="presigned local destination")
        return absolute, parent_fd, basename
    except (OSError, RuntimeError) as exc:
        if parent_fd >= 0:
            try:
                os.close(parent_fd)
            except OSError:
                logger.debug(
                    "Could not close presigned destination parent FD", exc_info=True
                )
        raise ValueError("presigned local destination has an unsafe parent") from exc


def _open_upload_source(path: str):
    """Open an upload source without following a symlink or hard link."""
    parent_fd = -1
    try:
        parent_fd, basename, _ = _open_local_parent(path, create=False)
        fd = os.open(
            basename,
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0),
            dir_fd=parent_fd,
        )
    except (OSError, RuntimeError) as exc:
        raise ValueError("presigned upload source has an unsafe parent") from exc
    finally:
        if parent_fd >= 0:
            os.close(parent_fd)
    try:
        metadata = os.fstat(fd)
        try:
            _assert_regular_single_link(metadata, label="presigned upload source")
        except RuntimeError as exc:
            raise ValueError(str(exc)) from exc
        return os.fdopen(fd, "rb"), metadata.st_size
    except Exception:
        os.close(fd)
        raise


def download_presigned(
    url: str,
    local_path: str,
    *,
    job_id: str | None = None,
    expected_key: str | None = None,
    max_bytes: int | None = None,
    transfer_observer: Callable[[str, int], None] | None = None,
) -> str:
    """Download a file from a presigned GET URL to a local path.

    Returns the local_path on success.
    Raises PresignedTransferError with only the HTTP status on errors.
    Raises ValueError if URL scheme/host/job binding fails.
    """
    _validate_presigned_url(url, job_id=job_id, expected_key=expected_key)
    max_transfer_bytes = _max_presigned_transfer_bytes()
    if max_bytes is not None:
        if isinstance(max_bytes, bool) or not isinstance(max_bytes, int) or max_bytes <= 0:
            raise ValueError("max_bytes must be a positive integer")
        max_transfer_bytes = min(max_transfer_bytes, max_bytes)
    local_path, parent_fd, basename = _prepare_local_destination(local_path)
    temporary_name = None
    temp_fd = -1

    try:
        for _ in range(5):
            candidate = f".{basename}.{os.urandom(8).hex()}"
            try:
                temp_fd = os.open(
                    candidate,
                    os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0),
                    0o600,
                    dir_fd=parent_fd,
                )
                temporary_name = candidate
                break
            except FileExistsError:
                continue
        else:
            raise ValueError("could not allocate a private presigned download file")
        os.fchmod(temp_fd, 0o600)
        with _pinned_presigned_url(url) as delivery_url:
            req = urllib.request.Request(delivery_url, method="GET")
            if transfer_observer is not None:
                transfer_observer("get_attempt", 0)
            with _open_presigned(req, timeout=_HTTP_TIMEOUT) as resp:
                declared_length = _declared_content_length(resp)
                if declared_length is not None and declared_length > max_transfer_bytes:
                    raise ValueError(
                        "Presigned response exceeds "
                        f"S3_PRESIGNED_MAX_TRANSFER_BYTES ({max_transfer_bytes})"
                    )
                transferred = 0
                with os.fdopen(temp_fd, "wb") as f:
                    temp_fd = -1
                    while True:
                        chunk = resp.read(65536)
                        if not chunk:
                            break
                        transferred += len(chunk)
                        if transferred > max_transfer_bytes:
                            raise ValueError(
                                "Presigned response exceeded "
                                f"S3_PRESIGNED_MAX_TRANSFER_BYTES ({max_transfer_bytes})"
                            )
                        f.write(chunk)
                    f.flush()
                    os.fsync(f.fileno())
                if transfer_observer is not None:
                    transfer_observer("get_success_bytes", transferred)
        os.replace(
            temporary_name,
            basename,
            src_dir_fd=parent_fd,
            dst_dir_fd=parent_fd,
        )
        temporary_name = None
        os.fsync(parent_fd)
    except Exception:
        if temp_fd >= 0:
            os.close(temp_fd)
        if temporary_name:
            try:
                os.unlink(temporary_name, dir_fd=parent_fd)
            except FileNotFoundError:
                pass
            except OSError:
                logger.warning(
                    "Could not remove partial presigned download %s", local_path
                )
        os.close(parent_fd)
        raise
    os.close(parent_fd)

    logger.debug(
        "Downloaded presigned URL to %s (%d bytes)",
        local_path,
        os.path.getsize(local_path),
    )
    return local_path


def upload_presigned(
    local_path: str,
    url: str,
    *,
    job_id: str | None = None,
    expected_key: str | None = None,
) -> None:
    """Upload a local file to a presigned PUT URL.

    Raises PresignedTransferError with only the HTTP status on errors.
    Raises ValueError if URL scheme/host/job binding fails.
    """
    _validate_presigned_url(url, job_id=job_id, expected_key=expected_key)
    max_transfer_bytes = _max_presigned_transfer_bytes()
    source_file, file_size = _open_upload_source(local_path)
    with source_file as f:
        if file_size > max_transfer_bytes:
            raise ValueError(
                "Local file exceeds "
                f"S3_PRESIGNED_MAX_TRANSFER_BYTES ({max_transfer_bytes})"
            )

        # urllib/http.client accepts a readable body and streams it in bounded
        # blocks.  Do not call read() here: presigned workers may handle large PDFs.
        with _pinned_presigned_url(url) as delivery_url:
            req = urllib.request.Request(
                delivery_url,
                data=f,
                method="PUT",
                headers={"Content-Length": str(file_size)},
            )
            timeout = max(_HTTP_TIMEOUT, file_size // (1024 * 1024) * 5)
            with _open_presigned(req, timeout=timeout):
                pass

    logger.debug("Uploaded %s to presigned URL (%d bytes)", local_path, file_size)


def delete_presigned(
    url: str,
    *,
    job_id: str | None = None,
    expected_key: str | None = None,
) -> None:
    """Delete one job-bound artifact through a presigned DELETE URL."""
    _validate_presigned_url(url, job_id=job_id, expected_key=expected_key)
    with _pinned_presigned_url(url) as delivery_url:
        req = urllib.request.Request(delivery_url, method="DELETE")
        with _open_presigned(req, timeout=_HTTP_TIMEOUT):
            pass
    logger.debug("Deleted artifact through a job-bound presigned URL")


# A real redirect-blocking opener.  ``build_opener(HTTPHandler, HTTPSHandler)``
# does not remove Python's default HTTPRedirectHandler, and it also inherits
# environment proxy settings; both behaviors are unsafe for bearer URLs.
_no_redirect_opener = urllib.request.build_opener(
    urllib.request.ProxyHandler({}),
    NoRedirectHandler,
)
