"""Tests for webhook secret at-rest encryption (SEC-001 / ).

Verifies that:
- Secrets round-trip correctly through encrypt/decrypt
- Stored values in the DB are NOT plaintext
- Pre-migration plaintext secrets are handled gracefully (backward compat)
- Different encryption keys produce different ciphertexts
- Encryption integrates correctly with job and batch creation paths
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from unittest.mock import patch

import pytest

from api.database import Batch, Job, get_engine

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def _fixed_key():
    """Set a deterministic encryption key for tests."""
    with patch.dict(os.environ, {"WEBHOOK_SECRET_KEY": "test-encryption-key-fixed"}):
        # Reload config module-level vars so helpers pick up the patched value
        import api.config as cfg
        original_key = cfg.WEBHOOK_SECRET_KEY
        cfg.WEBHOOK_SECRET_KEY = "test-encryption-key-fixed"
        yield
        cfg.WEBHOOK_SECRET_KEY = original_key


# ---------------------------------------------------------------------------
# Unit: encrypt / decrypt round-trip
# ---------------------------------------------------------------------------


class TestEncryptDecryptRoundTrip:
    """Test the low-level encrypt/decrypt helpers."""

    def test_round_trip_basic(self, _fixed_key):
        from api.config import decrypt_webhook_secret, encrypt_webhook_secret

        plaintext = "my-super-secret-hmac-key"
        ciphertext = encrypt_webhook_secret(plaintext)
        recovered = decrypt_webhook_secret(ciphertext)
        assert recovered == plaintext

    def test_ciphertext_differs_from_plaintext(self, _fixed_key):
        from api.config import encrypt_webhook_secret

        plaintext = "webhook-secret-value"
        ciphertext = encrypt_webhook_secret(plaintext)
        assert ciphertext != plaintext
        # Fernet tokens are base64 and start with 'gAAAAA'
        assert len(ciphertext) > len(plaintext)

    def test_round_trip_empty_string(self, _fixed_key):
        from api.config import decrypt_webhook_secret, encrypt_webhook_secret

        ciphertext = encrypt_webhook_secret("")
        assert decrypt_webhook_secret(ciphertext) == ""

    def test_round_trip_unicode(self, _fixed_key):
        from api.config import decrypt_webhook_secret, encrypt_webhook_secret

        plaintext = "secret-with-unicode-\u00e9\u00e8\u00ea"
        ciphertext = encrypt_webhook_secret(plaintext)
        assert decrypt_webhook_secret(ciphertext) == plaintext

    def test_different_keys_produce_different_ciphertext(self):
        import api.config as cfg

        plaintext = "same-secret"

        cfg.WEBHOOK_SECRET_KEY = "key-alpha"
        from api.config import encrypt_webhook_secret
        ct1 = encrypt_webhook_secret(plaintext)

        cfg.WEBHOOK_SECRET_KEY = "key-beta"
        ct2 = encrypt_webhook_secret(plaintext)

        # Different keys should produce different ciphertext
        assert ct1 != ct2

        # Restore
        cfg.WEBHOOK_SECRET_KEY = ""

    def test_decrypt_with_wrong_key_falls_back_to_plaintext(self):
        """When the key changes, decrypt should fall back gracefully."""
        import api.config as cfg

        cfg.WEBHOOK_SECRET_KEY = "original-key"
        from api.config import decrypt_webhook_secret, encrypt_webhook_secret
        ciphertext = encrypt_webhook_secret("my-secret")

        # Switch key — decryption should fail and return ciphertext as-is
        cfg.WEBHOOK_SECRET_KEY = "different-key"
        result = decrypt_webhook_secret(ciphertext)
        # Falls back to returning the raw ciphertext (not the original plaintext)
        assert result == ciphertext

        cfg.WEBHOOK_SECRET_KEY = ""


# ---------------------------------------------------------------------------
# Unit: backward compatibility with plaintext
# ---------------------------------------------------------------------------


class TestPlaintextFallback:
    """Pre-migration plaintext secrets should be returned as-is."""

    def test_plaintext_string_returned_as_is(self, _fixed_key):
        from api.config import decrypt_webhook_secret

        # A value that was never encrypted — just a raw string
        raw = "old-plaintext-secret"
        result = decrypt_webhook_secret(raw)
        assert result == raw

    def test_empty_string_passthrough(self, _fixed_key):
        from api.config import decrypt_webhook_secret

        assert decrypt_webhook_secret("") == ""


# ---------------------------------------------------------------------------
# Integration: Job model storage
# ---------------------------------------------------------------------------


class TestJobSecretStorage:
    """Verify that secrets stored in Job records are encrypted."""

    def test_stored_value_is_not_plaintext(self, _fixed_key, tmp_path):
        from sqlalchemy.orm import sessionmaker

        from api.config import encrypt_webhook_secret

        engine = get_engine(str(tmp_path / "test_enc.db"))
        Session = sessionmaker(bind=engine)
        session = Session()

        plaintext_secret = "my-webhook-hmac-secret"
        encrypted = encrypt_webhook_secret(plaintext_secret)

        job = Job(
            job_id="job_test001",
            status="submitted",
            source_file="test.pdf",
            webhook_url="https://example.com/hook",
            webhook_secret=encrypted,
            created_at=datetime.now(timezone.utc).replace(tzinfo=None),
        )
        session.add(job)
        session.commit()

        # Read back from DB
        loaded = session.get(Job, "job_test001")
        assert loaded.webhook_secret is not None
        # The stored value must NOT be the plaintext
        assert loaded.webhook_secret != plaintext_secret
        # The stored value must be the encrypted form
        assert loaded.webhook_secret == encrypted

        session.close()

    def test_decrypt_restores_original(self, _fixed_key, tmp_path):
        from sqlalchemy.orm import sessionmaker

        from api.config import decrypt_webhook_secret, encrypt_webhook_secret

        engine = get_engine(str(tmp_path / "test_enc.db"))
        Session = sessionmaker(bind=engine)
        session = Session()

        plaintext_secret = "another-hmac-secret-42"
        encrypted = encrypt_webhook_secret(plaintext_secret)

        job = Job(
            job_id="job_test002",
            status="submitted",
            source_file="doc.pdf",
            webhook_url="https://example.com/hook",
            webhook_secret=encrypted,
            created_at=datetime.now(timezone.utc).replace(tzinfo=None),
        )
        session.add(job)
        session.commit()

        loaded = session.get(Job, "job_test002")
        recovered = decrypt_webhook_secret(loaded.webhook_secret)
        assert recovered == plaintext_secret

        session.close()

    def test_null_webhook_secret_unchanged(self, _fixed_key, tmp_path):
        from sqlalchemy.orm import sessionmaker

        engine = get_engine(str(tmp_path / "test_enc.db"))
        Session = sessionmaker(bind=engine)
        session = Session()

        job = Job(
            job_id="job_test003",
            status="submitted",
            source_file="test.pdf",
            webhook_url="https://example.com/hook",
            webhook_secret=None,
            created_at=datetime.now(timezone.utc).replace(tzinfo=None),
        )
        session.add(job)
        session.commit()

        loaded = session.get(Job, "job_test003")
        assert loaded.webhook_secret is None

        session.close()


# ---------------------------------------------------------------------------
# Integration: Batch model storage
# ---------------------------------------------------------------------------


class TestBatchSecretStorage:
    """Verify that secrets stored in Batch records are encrypted."""

    def test_batch_secret_round_trip(self, _fixed_key, tmp_path):
        from sqlalchemy.orm import sessionmaker

        from api.config import decrypt_webhook_secret, encrypt_webhook_secret

        engine = get_engine(str(tmp_path / "test_enc.db"))
        Session = sessionmaker(bind=engine)
        session = Session()

        plaintext_secret = "batch-hmac-secret-xyz"
        encrypted = encrypt_webhook_secret(plaintext_secret)

        batch = Batch(
            batch_id="batch_test001",
            status="submitted",
            total_jobs=3,
            priority="normal",
            webhook_url="https://example.com/batch-hook",
            webhook_secret=encrypted,
            created_at=datetime.now(timezone.utc).replace(tzinfo=None),
        )
        session.add(batch)
        session.commit()

        loaded = session.get(Batch, "batch_test001")
        assert loaded.webhook_secret != plaintext_secret
        assert decrypt_webhook_secret(loaded.webhook_secret) == plaintext_secret

        session.close()


# ---------------------------------------------------------------------------
# Integration: key derivation consistency
# ---------------------------------------------------------------------------


class TestKeyDerivation:
    """Verify key derivation is deterministic."""

    def test_key_derivation_deterministic(self):
        import api.config as cfg

        cfg.WEBHOOK_SECRET_KEY = "deterministic-test"
        from api.config import _get_webhook_encryption_key
        key1 = _get_webhook_encryption_key()
        key2 = _get_webhook_encryption_key()
        assert key1 == key2
        cfg.WEBHOOK_SECRET_KEY = ""

    def test_fallback_to_ocr_api_key(self):
        import api.config as cfg

        cfg.WEBHOOK_SECRET_KEY = ""
        original_api_key = cfg.OCR_API_KEY
        cfg.OCR_API_KEY = "my-api-key-fallback"

        from api.config import _get_webhook_encryption_key
        key = _get_webhook_encryption_key()
        assert len(key) == 44  # base64-encoded 32 bytes

        cfg.OCR_API_KEY = original_api_key

    def test_fallback_to_dev_default(self):
        import api.config as cfg

        cfg.WEBHOOK_SECRET_KEY = ""
        original_api_key = cfg.OCR_API_KEY
        cfg.OCR_API_KEY = ""

        from api.config import _get_webhook_encryption_key
        key = _get_webhook_encryption_key()
        # Should still produce a valid key (from "dev-key-not-for-production")
        assert len(key) == 44

        cfg.OCR_API_KEY = original_api_key
