"""HTTP I/O helpers for presigned URL mode.

Workers use these functions instead of StorageBackend.download_file/upload_file
when operating with presigned URLs. No S3 credentials required -- only stdlib.
"""

from __future__ import annotations

import logging
import os
import urllib.request
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

_HTTP_TIMEOUT = 30


def _validate_presigned_url(url: str) -> None:
    """Validate presigned URL scheme to prevent SSRF via non-HTTP schemes."""
    parsed = urlparse(url)
    if parsed.scheme not in ("https", "http"):
        raise ValueError(f"Presigned URL must use HTTP(S), got: {parsed.scheme}")


# Opener that does NOT follow redirects (prevents SSRF via open redirect)
_no_redirect_opener = urllib.request.build_opener(
    urllib.request.HTTPHandler,
    urllib.request.HTTPSHandler,
)


def download_presigned(url: str, local_path: str) -> str:
    """Download a file from a presigned GET URL to a local path.

    Returns the local_path on success.
    Raises urllib.error.HTTPError on HTTP errors (403 expired, 404 not found).
    Raises ValueError if URL scheme is not HTTP(S).
    """
    _validate_presigned_url(url)
    local_dir = os.path.dirname(local_path)
    if local_dir:
        os.makedirs(local_dir, exist_ok=True)

    req = urllib.request.Request(url, method="GET")
    with _no_redirect_opener.open(req, timeout=_HTTP_TIMEOUT) as resp:
        with open(local_path, "wb") as f:
            while True:
                chunk = resp.read(65536)
                if not chunk:
                    break
                f.write(chunk)

    logger.debug("Downloaded presigned URL to %s (%d bytes)",
                 local_path, os.path.getsize(local_path))
    return local_path


def upload_presigned(local_path: str, url: str) -> None:
    """Upload a local file to a presigned PUT URL.

    Raises urllib.error.HTTPError on HTTP errors (403 expired).
    Raises ValueError if URL scheme is not HTTP(S).
    """
    _validate_presigned_url(url)
    file_size = os.path.getsize(local_path)
    with open(local_path, "rb") as f:
        data = f.read()

    req = urllib.request.Request(
        url,
        data=data,
        method="PUT",
        headers={"Content-Length": str(file_size)},
    )
    timeout = max(_HTTP_TIMEOUT, file_size // (1024 * 1024) * 5)
    with _no_redirect_opener.open(req, timeout=timeout):
        pass

    logger.debug("Uploaded %s to presigned URL (%d bytes)", local_path, file_size)
