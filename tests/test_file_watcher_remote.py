"""Tests for optional FTP/FTPS/SFTP remote ingest polling."""

from __future__ import annotations

import ftplib
import os
import stat
import sys
import threading
from types import SimpleNamespace
from unittest import mock

from file_watcher import SHUTDOWN_EVENT, ProcessedFileStore
from file_watcher_config import load_config_from_dict
from file_watcher_remote import (
    _AUTH_FAILURE_MAX_RETRIES,
    _MAX_BACKOFF_SECONDS,
    RemoteFileEntry,
    RemoteIngestPoller,
)


@mock.patch.dict(os.environ, {}, clear=False)
def test_sftp_poll_downloads_submits_and_deletes(tmp_path):
    """SFTP poll fetches a matching file, submits it, and deletes when configured."""
    config = load_config_from_dict(
        {
            "remote_ingest": [
                {
                    "protocol": "sftp",
                    "host": "sftp.example",
                    "username": "ocr",
                    "remote_path": "/incoming",
                    "delete_after_fetch": True,
                }
            ],
            "remote_staging_dir": str(tmp_path / "staging"),
        }
    )
    remote_cfg = config.remote_ingest[0]
    processed_store = ProcessedFileStore()
    submit_file = mock.MagicMock(return_value=True)
    poller = RemoteIngestPoller(
        remote_cfg=remote_cfg,
        watcher_config=config,
        processed_store=processed_store,
        submit_file=submit_file,
        shutdown_event=SHUTDOWN_EVENT,
    )

    mock_attr = SimpleNamespace(
        filename="drop.pdf",
        st_mode=0o100644,
        st_size=321,
        st_mtime=1234567890,
    )
    mock_sftp = mock.MagicMock()
    mock_sftp.listdir_attr.return_value = [mock_attr]

    def _fake_get(_remote_path, local_path):
        with open(local_path, "wb") as handle:
            handle.write(b"PDF")

    mock_sftp.get.side_effect = _fake_get
    mock_client = mock.MagicMock()
    mock_client.open_sftp.return_value = mock_sftp

    with mock.patch.object(poller, "_connect_sftp_client", return_value=mock_client):
        poller.poll_once()

    submit_file.assert_called_once()
    processed_key = submit_file.call_args.args[2]
    assert processed_key.startswith("sftp://sftp.example:22/incoming/drop.pdf|321|")
    mock_sftp.remove.assert_called_once_with("/incoming/drop.pdf")
    staged_path = submit_file.call_args.args[0]
    assert not os.path.exists(staged_path)


def test_sftp_strict_default_uses_reject_policy(tmp_path):
    """SFTP should reject unknown hosts unless explicitly overridden."""
    config = load_config_from_dict(
        {
            "remote_ingest": [
                {
                    "protocol": "sftp",
                    "host": "sftp.example",
                    "username": "ocr",
                    "remote_path": "/incoming",
                }
            ],
            "remote_staging_dir": str(tmp_path / "staging"),
        }
    )
    remote_cfg = config.remote_ingest[0]
    poller = RemoteIngestPoller(
        remote_cfg=remote_cfg,
        watcher_config=config,
        processed_store=ProcessedFileStore(),
        submit_file=mock.MagicMock(return_value=True),
        shutdown_event=SHUTDOWN_EVENT,
    )

    mock_client = mock.MagicMock()
    mock_paramiko = mock.MagicMock()
    mock_paramiko.SSHClient.return_value = mock_client
    mock_paramiko.RejectPolicy.return_value = "reject-policy"

    with mock.patch.dict(sys.modules, {"paramiko": mock_paramiko}):
        poller._connect_sftp_client()

    mock_client.set_missing_host_key_policy.assert_called_once_with("reject-policy")


def test_ftps_connect_uses_tls_client(tmp_path):
    """FTPS sources should use the stdlib FTP_TLS transport."""
    config = load_config_from_dict(
        {
            "remote_ingest": [
                {
                    "protocol": "ftps",
                    "host": "ftp.example",
                    "username": "ocr",
                    "password": "secret",
                    "remote_path": "/incoming",
                }
            ],
            "remote_staging_dir": str(tmp_path / "staging"),
        }
    )
    remote_cfg = config.remote_ingest[0]
    poller = RemoteIngestPoller(
        remote_cfg=remote_cfg,
        watcher_config=config,
        processed_store=ProcessedFileStore(),
        submit_file=mock.MagicMock(return_value=True),
        shutdown_event=SHUTDOWN_EVENT,
    )

    with mock.patch("file_watcher_remote.ftplib.FTP_TLS") as mock_tls:
        client = mock_tls.return_value
        poller._connect_ftp_client()

    client.login.assert_called_once_with("ocr", "secret")
    client.prot_p.assert_called_once()


def test_remote_processed_entries_are_skipped(tmp_path):
    """Already-processed remote identifiers should be skipped before download."""
    config = load_config_from_dict(
        {
            "remote_ingest": [
                {
                    "protocol": "ftp",
                    "host": "ftp.example",
                    "username": "ocr",
                    "remote_path": "/incoming",
                }
            ],
            "remote_staging_dir": str(tmp_path / "staging"),
        }
    )
    remote_cfg = config.remote_ingest[0]
    processed_store = ProcessedFileStore()
    submit_file = mock.MagicMock(return_value=True)
    poller = RemoteIngestPoller(
        remote_cfg=remote_cfg,
        watcher_config=config,
        processed_store=processed_store,
        submit_file=submit_file,
        shutdown_event=SHUTDOWN_EVENT,
    )

    entry = RemoteFileEntry(
        name="drop.pdf",
        remote_path="/incoming/drop.pdf",
        size=321,
        modified="12345",
    )
    processed_store.add(poller._processed_key(entry))

    mock_client = mock.MagicMock()
    mock_client.mlsd.return_value = [("drop.pdf", {"type": "file", "size": "321", "modify": "12345"})]

    with mock.patch.object(poller, "_connect_ftp_client", return_value=mock_client):
        poller.poll_once()

    submit_file.assert_not_called()


# ---------------------------------------------------------------------------
# Path traversal prevention
# ---------------------------------------------------------------------------


def test_stage_path_traversal_blocked(tmp_path):
    """Filenames with ../../etc/passwd get directory components stripped."""
    config = load_config_from_dict(
        {
            "remote_ingest": [
                {
                    "protocol": "ftp",
                    "host": "ftp.example",
                    "username": "ocr",
                    "remote_path": "/incoming",
                }
            ],
            "remote_staging_dir": str(tmp_path / "staging"),
        }
    )
    remote_cfg = config.remote_ingest[0]
    poller = RemoteIngestPoller(
        remote_cfg=remote_cfg,
        watcher_config=config,
        processed_store=ProcessedFileStore(),
        submit_file=mock.MagicMock(return_value=True),
        shutdown_event=SHUTDOWN_EVENT,
    )

    # The dangerous traversal path should be stripped to just "passwd".
    result = poller._stage_local_path("../../etc/passwd")
    assert result.parent == (tmp_path / "staging").resolve()
    assert "passwd" in result.name
    assert ".." not in result.name

    # Null bytes should be stripped.
    result2 = poller._stage_local_path("evil\x00.pdf")
    assert "\x00" not in result2.name

    # An empty/dots-only name should become "unnamed".
    result3 = poller._stage_local_path("...")
    assert "unnamed" in result3.name


# ---------------------------------------------------------------------------
# Retry backoff
# ---------------------------------------------------------------------------


def test_retry_backoff_on_failure(tmp_path):
    """Consecutive poll failures increase backoff wait time."""
    config = load_config_from_dict(
        {
            "remote_ingest": [
                {
                    "protocol": "ftp",
                    "host": "ftp.example",
                    "username": "ocr",
                    "remote_path": "/incoming",
                    "poll_interval_seconds": 5.0,
                }
            ],
            "remote_staging_dir": str(tmp_path / "staging"),
        }
    )
    remote_cfg = config.remote_ingest[0]
    poller = RemoteIngestPoller(
        remote_cfg=remote_cfg,
        watcher_config=config,
        processed_store=ProcessedFileStore(),
        submit_file=mock.MagicMock(return_value=True),
        shutdown_event=threading.Event(),
    )

    # No failures: base interval.
    assert poller._backoff_interval() == 5.0

    # Simulate consecutive failures.
    poller._consecutive_failures = 1
    assert poller._backoff_interval() == 10.0  # 5 * 2^1

    poller._consecutive_failures = 3
    assert poller._backoff_interval() == 40.0  # 5 * 2^3

    # Should cap at _MAX_BACKOFF_SECONDS.
    poller._consecutive_failures = 10
    assert poller._backoff_interval() == _MAX_BACKOFF_SECONDS


def test_retry_backoff_resets_on_success(tmp_path):
    """Successful poll resets the consecutive failure counter."""
    config = load_config_from_dict(
        {
            "remote_ingest": [
                {
                    "protocol": "ftp",
                    "host": "ftp.example",
                    "username": "ocr",
                    "remote_path": "/incoming",
                    "poll_interval_seconds": 1.0,
                }
            ],
            "remote_staging_dir": str(tmp_path / "staging"),
        }
    )
    remote_cfg = config.remote_ingest[0]
    shutdown_event = threading.Event()
    poller = RemoteIngestPoller(
        remote_cfg=remote_cfg,
        watcher_config=config,
        processed_store=ProcessedFileStore(),
        submit_file=mock.MagicMock(return_value=True),
        shutdown_event=shutdown_event,
    )

    call_count = 0

    def _side_effect():
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise RuntimeError("transient failure")
        # Second call succeeds, then shut down.
        shutdown_event.set()

    with mock.patch.object(poller, "poll_once", side_effect=_side_effect):
        poller.run()

    assert poller._consecutive_failures == 0


# ---------------------------------------------------------------------------
# Auth failure caps retries
# ---------------------------------------------------------------------------


def test_auth_failure_caps_retries(tmp_path):
    """Paramiko AuthenticationException stops retries after max attempts."""
    config = load_config_from_dict(
        {
            "remote_ingest": [
                {
                    "protocol": "sftp",
                    "host": "sftp.example",
                    "username": "ocr",
                    "remote_path": "/incoming",
                    "poll_interval_seconds": 0.01,
                }
            ],
            "remote_staging_dir": str(tmp_path / "staging"),
        }
    )
    remote_cfg = config.remote_ingest[0]
    shutdown_event = threading.Event()
    poller = RemoteIngestPoller(
        remote_cfg=remote_cfg,
        watcher_config=config,
        processed_store=ProcessedFileStore(),
        submit_file=mock.MagicMock(return_value=True),
        shutdown_event=shutdown_event,
    )

    # Create a fake AuthenticationException.
    class AuthenticationException(Exception):
        pass

    with mock.patch.object(
        poller, "poll_once", side_effect=AuthenticationException("bad credentials")
    ):
        poller.run()

    assert poller._auth_failures >= _AUTH_FAILURE_MAX_RETRIES

    # Also test FTP 530 error detection.
    exc = ftplib.error_perm("530 Login incorrect")
    assert RemoteIngestPoller._is_auth_error(exc) is True

    # Non-auth FTP errors should not be flagged.
    exc2 = ftplib.error_perm("550 File not found")
    assert RemoteIngestPoller._is_auth_error(exc2) is False


# ---------------------------------------------------------------------------
# Symlink detection (SFTP)
# ---------------------------------------------------------------------------


def test_symlink_skipped(tmp_path):
    """SFTP entries with S_ISLNK mode are excluded from the listing."""
    config = load_config_from_dict(
        {
            "remote_ingest": [
                {
                    "protocol": "sftp",
                    "host": "sftp.example",
                    "username": "ocr",
                    "remote_path": "/incoming",
                }
            ],
            "remote_staging_dir": str(tmp_path / "staging"),
        }
    )
    remote_cfg = config.remote_ingest[0]
    poller = RemoteIngestPoller(
        remote_cfg=remote_cfg,
        watcher_config=config,
        processed_store=ProcessedFileStore(),
        submit_file=mock.MagicMock(return_value=True),
        shutdown_event=SHUTDOWN_EVENT,
    )

    regular_attr = SimpleNamespace(
        filename="real.pdf",
        st_mode=stat.S_IFREG | 0o644,
        st_size=1024,
        st_mtime=1234567890,
    )
    symlink_attr = SimpleNamespace(
        filename="link.pdf",
        st_mode=stat.S_IFLNK | 0o777,
        st_size=0,
        st_mtime=1234567890,
    )
    mock_sftp = mock.MagicMock()
    mock_sftp.listdir_attr.return_value = [regular_attr, symlink_attr]

    entries = poller._iter_sftp_entries(mock_sftp)
    names = [e.name for e in entries]
    assert "real.pdf" in names
    assert "link.pdf" not in names


# ---------------------------------------------------------------------------
# CRLF filename sanitization
# ---------------------------------------------------------------------------


def test_crlf_filename_sanitized(tmp_path):
    """Filenames with CR/LF characters are cleaned in staging path and RETR command."""
    config = load_config_from_dict(
        {
            "remote_ingest": [
                {
                    "protocol": "ftp",
                    "host": "ftp.example",
                    "username": "ocr",
                    "remote_path": "/incoming",
                }
            ],
            "remote_staging_dir": str(tmp_path / "staging"),
        }
    )
    remote_cfg = config.remote_ingest[0]
    poller = RemoteIngestPoller(
        remote_cfg=remote_cfg,
        watcher_config=config,
        processed_store=ProcessedFileStore(),
        submit_file=mock.MagicMock(return_value=True),
        shutdown_event=SHUTDOWN_EVENT,
    )

    # Verify stage_local_path strips CR/LF.
    staged = poller._stage_local_path("evil\r\nfile.pdf")
    assert "\r" not in staged.name
    assert "\n" not in staged.name
    assert "evilfile.pdf" in staged.name

    # Verify RETR command also uses sanitized name.
    entry = RemoteFileEntry(
        name="evil\r\nfile.pdf",
        remote_path="/incoming/evil\r\nfile.pdf",
        size=100,
        modified="12345",
    )
    mock_client = mock.MagicMock()
    poller._download_ftp_file(mock_client, entry)

    retr_call = mock_client.retrbinary.call_args
    assert "\r" not in retr_call[0][0]
    assert "\n" not in retr_call[0][0]
    assert "evilfile.pdf" in retr_call[0][0]


# ---------------------------------------------------------------------------
# Max file size skip
# ---------------------------------------------------------------------------


def test_max_file_size_skip(tmp_path):
    """Files exceeding max_file_size_mb are skipped by _should_fetch."""
    config = load_config_from_dict(
        {
            "remote_ingest": [
                {
                    "protocol": "ftp",
                    "host": "ftp.example",
                    "username": "ocr",
                    "remote_path": "/incoming",
                    "max_file_size_mb": 10,
                }
            ],
            "remote_staging_dir": str(tmp_path / "staging"),
        }
    )
    remote_cfg = config.remote_ingest[0]
    poller = RemoteIngestPoller(
        remote_cfg=remote_cfg,
        watcher_config=config,
        processed_store=ProcessedFileStore(),
        submit_file=mock.MagicMock(return_value=True),
        shutdown_event=SHUTDOWN_EVENT,
    )

    # File under limit should be fetchable.
    small_entry = RemoteFileEntry(
        name="small.pdf",
        remote_path="/incoming/small.pdf",
        size=5 * 1024 * 1024,  # 5 MB
        modified="12345",
    )
    assert poller._should_fetch(small_entry) is True

    # File over limit should be skipped.
    big_entry = RemoteFileEntry(
        name="big.pdf",
        remote_path="/incoming/big.pdf",
        size=20 * 1024 * 1024,  # 20 MB
        modified="12345",
    )
    assert poller._should_fetch(big_entry) is False


def test_max_file_size_zero_means_unlimited(tmp_path):
    """max_file_size_mb=0 (default) allows files of any size."""
    config = load_config_from_dict(
        {
            "remote_ingest": [
                {
                    "protocol": "ftp",
                    "host": "ftp.example",
                    "username": "ocr",
                    "remote_path": "/incoming",
                    # max_file_size_mb defaults to 0
                }
            ],
            "remote_staging_dir": str(tmp_path / "staging"),
        }
    )
    remote_cfg = config.remote_ingest[0]
    poller = RemoteIngestPoller(
        remote_cfg=remote_cfg,
        watcher_config=config,
        processed_store=ProcessedFileStore(),
        submit_file=mock.MagicMock(return_value=True),
        shutdown_event=SHUTDOWN_EVENT,
    )

    huge_entry = RemoteFileEntry(
        name="huge.pdf",
        remote_path="/incoming/huge.pdf",
        size=999 * 1024 * 1024,  # 999 MB
        modified="12345",
    )
    assert poller._should_fetch(huge_entry) is True


# ---------------------------------------------------------------------------
# FTP socket timeout
# ---------------------------------------------------------------------------


def test_ftp_socket_timeout_set(tmp_path):
    """FTP client socket timeout is set after connect for data operations."""
    config = load_config_from_dict(
        {
            "remote_ingest": [
                {
                    "protocol": "ftp",
                    "host": "ftp.example",
                    "username": "ocr",
                    "password": "secret",
                    "remote_path": "/incoming",
                    "timeout_seconds": 42.0,
                }
            ],
            "remote_staging_dir": str(tmp_path / "staging"),
        }
    )
    remote_cfg = config.remote_ingest[0]
    poller = RemoteIngestPoller(
        remote_cfg=remote_cfg,
        watcher_config=config,
        processed_store=ProcessedFileStore(),
        submit_file=mock.MagicMock(return_value=True),
        shutdown_event=SHUTDOWN_EVENT,
    )

    with mock.patch("file_watcher_remote.ftplib.FTP") as mock_ftp_cls:
        mock_client = mock_ftp_cls.return_value
        mock_sock = mock.MagicMock()
        mock_client.sock = mock_sock

        poller._connect_ftp_client()

    mock_sock.settimeout.assert_called_once_with(42.0)


# ---------------------------------------------------------------------------
# FTPS uses SSL context
# ---------------------------------------------------------------------------


def test_ftps_uses_ssl_context(tmp_path):
    """FTPS connections use ssl.create_default_context for certificate verification."""
    config = load_config_from_dict(
        {
            "remote_ingest": [
                {
                    "protocol": "ftps",
                    "host": "ftp.example",
                    "username": "ocr",
                    "password": "secret",
                    "remote_path": "/incoming",
                }
            ],
            "remote_staging_dir": str(tmp_path / "staging"),
        }
    )
    remote_cfg = config.remote_ingest[0]
    poller = RemoteIngestPoller(
        remote_cfg=remote_cfg,
        watcher_config=config,
        processed_store=ProcessedFileStore(),
        submit_file=mock.MagicMock(return_value=True),
        shutdown_event=SHUTDOWN_EVENT,
    )

    with mock.patch("file_watcher_remote.ssl.create_default_context") as mock_ctx, \
         mock.patch("file_watcher_remote.ftplib.FTP_TLS") as mock_tls_cls:
        mock_client = mock_tls_cls.return_value
        mock_client.sock = mock.MagicMock()

        poller._connect_ftp_client()

    mock_ctx.assert_called_once()
    mock_tls_cls.assert_called_once_with(context=mock_ctx.return_value)


# ---------------------------------------------------------------------------
# SFTP banner/auth timeout
# ---------------------------------------------------------------------------


def test_sftp_connect_passes_timeout_kwargs(tmp_path):
    """SFTP connect passes banner_timeout and auth_timeout."""
    config = load_config_from_dict(
        {
            "remote_ingest": [
                {
                    "protocol": "sftp",
                    "host": "sftp.example",
                    "username": "ocr",
                    "remote_path": "/incoming",
                    "timeout_seconds": 60.0,
                }
            ],
            "remote_staging_dir": str(tmp_path / "staging"),
        }
    )
    remote_cfg = config.remote_ingest[0]
    poller = RemoteIngestPoller(
        remote_cfg=remote_cfg,
        watcher_config=config,
        processed_store=ProcessedFileStore(),
        submit_file=mock.MagicMock(return_value=True),
        shutdown_event=SHUTDOWN_EVENT,
    )

    mock_client = mock.MagicMock()
    mock_paramiko = mock.MagicMock()
    mock_paramiko.SSHClient.return_value = mock_client
    mock_paramiko.RejectPolicy.return_value = "reject-policy"

    with mock.patch.dict(sys.modules, {"paramiko": mock_paramiko}):
        poller._connect_sftp_client()

    connect_kwargs = mock_client.connect.call_args[1]
    assert connect_kwargs["banner_timeout"] == 60.0
    assert connect_kwargs["auth_timeout"] == 60.0
    assert connect_kwargs["timeout"] == 60.0
