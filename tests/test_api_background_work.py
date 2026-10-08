"""Real owned writers must remain visible until their final mutation ends."""

from threading import Event
from unittest.mock import patch

import pytest

from api.background_work import active_count, start_writer, wait_for_writers


def test_running_writer_remains_visible_and_timeout_does_not_release_it():
    entered, release = Event(), Event()

    def write():
        entered.set()
        assert release.wait(5)

    before = active_count()
    writer = start_writer(write)
    try:
        assert entered.wait(2)
        assert active_count() == before + 1
        assert wait_for_writers(0) is False
    finally:
        release.set()
        writer.join(timeout=5)
    assert active_count() == before


def test_failed_thread_start_releases_reservation():
    before = active_count()
    with patch("threading.Thread.start", side_effect=RuntimeError("owned failure")):
        with pytest.raises(RuntimeError, match="owned failure"):
            start_writer(lambda: None)
    assert active_count() == before


def test_webhook_delivery_is_visible_until_callback_finishes():
    from api.webhooks import _submit_bounded, shutdown_webhook_pool

    entered, release = Event(), Event()

    def callback():
        entered.set()
        assert release.wait(5)

    before = active_count()
    try:
        _submit_bounded(callback, "owned-fixture")
        assert entered.wait(2)
        assert active_count() == before + 1
        assert wait_for_writers(0) is False
    finally:
        release.set()
        shutdown_webhook_pool()
    assert active_count() == before
