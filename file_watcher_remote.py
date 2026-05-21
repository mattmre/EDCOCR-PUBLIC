"""Remote ingest pollers for optional FTP/FTPS/SFTP watcher sources."""

from __future__ import annotations

import fnmatch
import ftplib
import logging
import posixpath
import secrets
import ssl
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

logger = logging.getLogger("file_watcher.remote")


@dataclass(frozen=True)
class RemoteFileEntry:
    """Lightweight description of a remote file discovered by polling."""

    name: str
    remote_path: str
    size: int
    modified: str


_MAX_BACKOFF_SECONDS = 300
_AUTH_FAILURE_MAX_RETRIES = 5


class RemoteIngestPoller:
    """Poll a remote FTP/FTPS/SFTP source and submit fetched files locally."""

    def __init__(
        self,
        *,
        remote_cfg,
        watcher_config,
        processed_store,
        submit_file: Callable[[str, object, str], bool],
        shutdown_event,
    ):
        self._remote_cfg = remote_cfg
        self._watcher_config = watcher_config
        self._processed_store = processed_store
        self._submit_file = submit_file
        self._shutdown_event = shutdown_event
        self._consecutive_failures: int = 0
        self._auth_failures: int = 0

    def run(self) -> None:
        """Poll until shutdown is requested, with exponential backoff on failure."""
        while not self._shutdown_event.is_set():
            try:
                self.poll_once()
                # Reset failure counters on success.
                self._consecutive_failures = 0
                self._auth_failures = 0
            except Exception as exc:
                self._consecutive_failures += 1
                # Detect authentication errors (paramiko or FTP 530).
                is_auth_error = self._is_auth_error(exc)
                if is_auth_error:
                    self._auth_failures += 1
                    if self._auth_failures >= _AUTH_FAILURE_MAX_RETRIES:
                        logger.error(
                            "Auth failure limit (%d) reached for %s://%s — stopping retries",
                            _AUTH_FAILURE_MAX_RETRIES,
                            self._remote_cfg.protocol,
                            self._remote_cfg.host,
                        )
                        return
                logger.exception(
                    "Remote ingest poll failed for %s://%s (attempt %d)",
                    self._remote_cfg.protocol,
                    self._remote_cfg.host,
                    self._consecutive_failures,
                )
            wait = self._backoff_interval()
            self._shutdown_event.wait(wait)

    def _backoff_interval(self) -> float:
        """Return the next poll wait with exponential backoff on failures."""
        base = self._remote_cfg.poll_interval_seconds
        if self._consecutive_failures == 0:
            return base
        backoff = base * (2 ** self._consecutive_failures)
        return min(backoff, _MAX_BACKOFF_SECONDS)

    @staticmethod
    def _is_auth_error(exc: BaseException) -> bool:
        """Return True if the exception looks like an authentication failure."""
        # Paramiko AuthenticationException.
        exc_type = type(exc).__name__
        if "AuthenticationException" in exc_type:
            return True
        # FTP 530 login incorrect.
        if isinstance(exc, ftplib.error_perm) and str(exc).startswith("530"):
            return True
        return False

    def poll_once(self) -> None:
        """Poll the configured remote source once."""
        if self._remote_cfg.protocol in {"ftp", "ftps"}:
            self._poll_ftp_like()
            return
        if self._remote_cfg.protocol == "sftp":
            self._poll_sftp()
            return
        raise ValueError(f"Unsupported remote protocol: {self._remote_cfg.protocol}")

    def _matches_patterns(self, filename: str) -> bool:
        lower = filename.lower()
        for pattern in self._remote_cfg.patterns:
            if fnmatch.fnmatch(lower, pattern.lower()):
                return True
        return False

    def _matches_ignore_patterns(self, filename: str) -> bool:
        lower = filename.lower()
        for pattern in self._remote_cfg.ignore_patterns:
            if fnmatch.fnmatch(lower, pattern.lower()):
                return True
        return False

    def _processed_key(self, entry: RemoteFileEntry) -> str:
        """Return a restart-safe identifier for a remote object version."""
        port = self._remote_cfg.port
        return (
            f"{self._remote_cfg.protocol}://{self._remote_cfg.host}:{port}"
            f"{entry.remote_path}|{entry.size}|{entry.modified}"
        )

    def _stage_local_path(self, filename: str) -> Path:
        # Sanitize: strip directory components and dangerous characters.
        safe_name = Path(filename).name  # strip directory components
        safe_name = safe_name.replace("\x00", "").replace("\r", "").replace("\n", "")
        safe_name = safe_name.strip(". ")
        if not safe_name:
            safe_name = "unnamed"
        staging_dir = Path(self._watcher_config.remote_staging_dir).expanduser()
        staging_dir.mkdir(parents=True, exist_ok=True)
        staged = staging_dir / f"{secrets.token_hex(6)}-{safe_name}"
        # Verify containment -- prevents symlink escape or ../ trickery.
        if staged.resolve().parent != staging_dir.resolve():
            raise ValueError(f"Staged path escapes staging directory: {filename!r}")
        return staged

    def _poll_ftp_like(self) -> None:
        client = self._connect_ftp_client()
        try:
            client.cwd(self._remote_cfg.remote_path)
            for entry in self._iter_ftp_entries(client):
                if not self._should_fetch(entry):
                    continue
                staged_path = self._download_ftp_file(client, entry)
                try:
                    if self._submit_file(
                        str(staged_path),
                        self._remote_cfg,
                        self._processed_key(entry),
                    ) and self._remote_cfg.delete_after_fetch:
                        client.delete(entry.name)
                finally:
                    self._cleanup_staged_file(staged_path)
        finally:
            try:
                client.quit()
            except Exception:
                client.close()

    def _connect_ftp_client(self):
        """Open and authenticate an FTP or FTPS client."""
        if self._remote_cfg.protocol == "ftps":
            ctx = ssl.create_default_context()
            client = ftplib.FTP_TLS(context=ctx)
        else:
            client = ftplib.FTP()
        timeout = self._remote_cfg.timeout_seconds
        client.connect(
            self._remote_cfg.host,
            self._remote_cfg.port,
            timeout=timeout,
        )
        client.login(self._remote_cfg.username, self._remote_cfg.password)
        client.set_pasv(self._remote_cfg.passive_mode)
        if self._remote_cfg.protocol == "ftps":
            client.prot_p()
        # Apply socket-level timeout for data transfer operations.
        if client.sock is not None:
            client.sock.settimeout(timeout)
        return client

    def _iter_ftp_entries(self, client) -> list[RemoteFileEntry]:
        """List regular files from an FTP/FTPS remote path."""
        entries: list[RemoteFileEntry] = []
        try:
            for name, facts in client.mlsd():
                if facts.get("type") != "file":
                    continue
                entries.append(
                    RemoteFileEntry(
                        name=name,
                        remote_path=posixpath.join(self._remote_cfg.remote_path, name),
                        size=int(facts.get("size", "0") or 0),
                        modified=str(facts.get("modify", "")),
                    )
                )
            return entries
        except Exception:
            logger.debug("MLSD unavailable for %s, falling back to NLST", self._remote_cfg.host)

        for name in client.nlst():
            if name in {".", ".."}:
                continue
            try:
                size = int(client.size(name) or 0)
            except Exception:
                continue
            entries.append(
                RemoteFileEntry(
                    name=name,
                    remote_path=posixpath.join(self._remote_cfg.remote_path, name),
                    size=size,
                    modified="",
                )
            )
        return entries

    def _download_ftp_file(self, client, entry: RemoteFileEntry) -> Path:
        """Download a single FTP/FTPS file to local staging."""
        staged_path = self._stage_local_path(entry.name)
        partial_path = staged_path.with_suffix(staged_path.suffix + ".part")
        # Sanitize CR/LF from filename before issuing RETR command.
        safe_name = entry.name.replace("\r", "").replace("\n", "")
        with partial_path.open("wb") as handle:
            client.retrbinary(f"RETR {safe_name}", handle.write)
        partial_path.replace(staged_path)
        return staged_path

    def _poll_sftp(self) -> None:
        client = self._connect_sftp_client()
        try:
            sftp = client.open_sftp()
            # Set channel-level timeout for data transfer operations.
            channel = sftp.get_channel()
            if channel is not None:
                channel.settimeout(self._remote_cfg.timeout_seconds)
            try:
                for entry in self._iter_sftp_entries(sftp):
                    if not self._should_fetch(entry):
                        continue
                    staged_path = self._download_sftp_file(sftp, entry)
                    try:
                        if self._submit_file(
                            str(staged_path),
                            self._remote_cfg,
                            self._processed_key(entry),
                        ) and self._remote_cfg.delete_after_fetch:
                            sftp.remove(entry.remote_path)
                    finally:
                        self._cleanup_staged_file(staged_path)
            finally:
                sftp.close()
        finally:
            client.close()

    def _connect_sftp_client(self):
        """Open and authenticate an SFTP client using Paramiko."""
        import paramiko

        client = paramiko.SSHClient()
        if self._remote_cfg.known_hosts_path:
            client.load_host_keys(self._remote_cfg.known_hosts_path)
        else:
            client.load_system_host_keys()
        policy = (
            paramiko.AutoAddPolicy()
            if self._remote_cfg.allow_unknown_host
            else paramiko.RejectPolicy()
        )
        client.set_missing_host_key_policy(policy)

        timeout = self._remote_cfg.timeout_seconds
        connect_kwargs = {
            "hostname": self._remote_cfg.host,
            "port": self._remote_cfg.port,
            "username": self._remote_cfg.username,
            "timeout": timeout,
            "banner_timeout": timeout,
            "auth_timeout": timeout,
            "allow_agent": False,
            "look_for_keys": bool(self._remote_cfg.key_filename),
        }
        if self._remote_cfg.password:
            connect_kwargs["password"] = self._remote_cfg.password
        if self._remote_cfg.key_filename:
            connect_kwargs["key_filename"] = self._remote_cfg.key_filename
        client.connect(**connect_kwargs)
        return client

    def _iter_sftp_entries(self, sftp) -> list[RemoteFileEntry]:
        """List regular files from an SFTP remote path, skipping symlinks."""
        entries: list[RemoteFileEntry] = []
        for attr in sftp.listdir_attr(self._remote_cfg.remote_path):
            # Skip symlinks to prevent following links to outside directories.
            if stat.S_ISLNK(attr.st_mode):
                logger.debug("Skipping symlink: %s", attr.filename)
                continue
            if not stat.S_ISREG(attr.st_mode):
                continue
            entries.append(
                RemoteFileEntry(
                    name=attr.filename,
                    remote_path=posixpath.join(
                        self._remote_cfg.remote_path,
                        attr.filename,
                    ),
                    size=int(attr.st_size or 0),
                    modified=str(int(attr.st_mtime or 0)),
                )
            )
        return entries

    def _download_sftp_file(self, sftp, entry: RemoteFileEntry) -> Path:
        """Download a single SFTP file to local staging."""
        staged_path = self._stage_local_path(entry.name)
        partial_path = staged_path.with_suffix(staged_path.suffix + ".part")
        sftp.get(entry.remote_path, str(partial_path))
        partial_path.replace(staged_path)
        return staged_path

    def _should_fetch(self, entry: RemoteFileEntry) -> bool:
        """Return True when a remote file should be downloaded and submitted."""
        if not self._matches_patterns(entry.name):
            return False
        if self._matches_ignore_patterns(entry.name):
            return False
        if entry.name.startswith("."):
            return False
        # Enforce max file size limit when configured.
        max_mb = getattr(self._remote_cfg, "max_file_size_mb", 0)
        if max_mb > 0:
            max_bytes = max_mb * 1024 * 1024
            if entry.size and entry.size > max_bytes:
                logger.info(
                    "Skipping %s: size %d exceeds max %d",
                    entry.name,
                    entry.size,
                    int(max_bytes),
                )
                return False
        return not self._processed_store.contains(self._processed_key(entry))

    def _cleanup_staged_file(self, staged_path: Path) -> None:
        """Remove staged local files after submission attempts."""
        for candidate in (staged_path, staged_path.with_suffix(staged_path.suffix + ".part")):
            try:
                candidate.unlink(missing_ok=True)
            except Exception:
                logger.exception("Failed to clean staged file %s", candidate)
