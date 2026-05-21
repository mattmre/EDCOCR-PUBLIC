"""Tests for PII encryption at rest (Fernet/AES-128-CBC).

Validates encrypt/decrypt roundtrip, key rotation, graceful degradation,
EncryptedTextField behavior, and thread safety without requiring a running
Django/PostgreSQL instance.  The pii_encryption module is loaded by file
path to avoid needing coordinator on sys.path.  The encrypted_field module
requires a mock Django since it imports django.db.models.
"""

import importlib.util
import sys
import types
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from cryptography.fernet import Fernet

# ---------------------------------------------------------------------------
# Resolve file paths relative to the repo root
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parent.parent
_PII_ENC_PATH = _REPO_ROOT / "coordinator" / "jobs" / "pii_encryption.py"
_ENC_FIELD_PATH = _REPO_ROOT / "coordinator" / "jobs" / "encrypted_field.py"


def _load_module_from_file(name, path):
    """Load a Python module from an absolute file path."""
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def pii_mod():
    """Load pii_encryption module fresh for each test (no singleton cache).

    pii_encryption.py has no Django imports, so it can be loaded directly.
    """
    mod_name = "coordinator.jobs.pii_encryption"
    # Clear any previous load
    sys.modules.pop(mod_name, None)
    mod = _load_module_from_file(mod_name, _PII_ENC_PATH)
    mod.reset_encryptor()
    yield mod
    sys.modules.pop(mod_name, None)


@pytest.fixture()
def enc_field_mod(pii_mod):
    """Load encrypted_field module with mock Django in place.

    encrypted_field.py imports ``django.db.models`` and ``.pii_encryption``,
    so we mock Django and ensure pii_encryption is already loaded.
    """
    mod_name = "coordinator.jobs.encrypted_field"
    sys.modules.pop(mod_name, None)

    # Build minimal django.db.models mock
    saved = {}
    django_mocks = _build_mock_django_db_models()
    for name, mock_mod in django_mocks.items():
        saved[name] = sys.modules.get(name)
        sys.modules[name] = mock_mod

    # Ensure the relative import of .pii_encryption resolves: create
    # coordinator and coordinator.jobs as namespace packages pointing at
    # the real directory so relative imports work.
    coordinator_pkg = types.ModuleType("coordinator")
    coordinator_pkg.__path__ = [str(_REPO_ROOT / "coordinator")]
    coordinator_pkg.__package__ = "coordinator"
    jobs_pkg = types.ModuleType("coordinator.jobs")
    jobs_pkg.__path__ = [str(_REPO_ROOT / "coordinator" / "jobs")]
    jobs_pkg.__package__ = "coordinator.jobs"
    # Attach pii_encryption as attribute
    jobs_pkg.pii_encryption = pii_mod

    saved["coordinator"] = sys.modules.get("coordinator")
    saved["coordinator.jobs"] = sys.modules.get("coordinator.jobs")
    sys.modules["coordinator"] = coordinator_pkg
    sys.modules["coordinator.jobs"] = jobs_pkg
    # Also register pii_encryption under its full dotted name
    sys.modules["coordinator.jobs.pii_encryption"] = pii_mod

    mod = _load_module_from_file(mod_name, _ENC_FIELD_PATH)

    yield mod

    # Restore
    sys.modules.pop(mod_name, None)
    for name, orig in saved.items():
        if orig is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = orig


def _build_mock_django_db_models():
    """Return mock modules for django.db.models (enough for TextField)."""

    class FakeTextField:
        def __init__(self, **kwargs):
            self._kwargs = kwargs

        def get_prep_value(self, value):
            return value

        def from_db_value(self, value, expression, connection):
            return value

        def deconstruct(self):
            return ("name", "django.db.models.TextField", [], self._kwargs)

    db_models_mod = types.ModuleType("django.db.models")
    db_models_mod.TextField = FakeTextField

    db_mod = types.ModuleType("django.db")
    db_mod.models = db_models_mod

    django_mod = types.ModuleType("django")

    return {
        "django": django_mod,
        "django.db": db_mod,
        "django.db.models": db_models_mod,
    }


@pytest.fixture()
def valid_key():
    """Generate a valid Fernet key for testing."""
    return Fernet.generate_key().decode("utf-8")


@pytest.fixture()
def second_key():
    """Generate a second valid Fernet key for rotation tests."""
    return Fernet.generate_key().decode("utf-8")


# ---------------------------------------------------------------------------
# PiiEncryptor core tests
# ---------------------------------------------------------------------------


class TestPiiEncryptor:
    """Tests for the PiiEncryptor class."""

    def test_encrypt_decrypt_roundtrip(self, pii_mod, valid_key):
        enc = pii_mod.PiiEncryptor(valid_key)
        plaintext = "123-45-6789"
        ciphertext = enc.encrypt(plaintext)
        assert ciphertext != plaintext
        assert enc.decrypt(ciphertext) == plaintext

    def test_encrypt_decrypt_unicode(self, pii_mod, valid_key):
        enc = pii_mod.PiiEncryptor(valid_key)
        plaintext = "Nombre: Jose Garcia-Lopez"
        ciphertext = enc.encrypt(plaintext)
        assert enc.decrypt(ciphertext) == plaintext

    def test_encrypt_decrypt_unicode_cjk(self, pii_mod, valid_key):
        enc = pii_mod.PiiEncryptor(valid_key)
        plaintext = "Name: Tanaka Taro"
        ciphertext = enc.encrypt(plaintext)
        assert enc.decrypt(ciphertext) == plaintext

    def test_encrypt_empty_string(self, pii_mod, valid_key):
        enc = pii_mod.PiiEncryptor(valid_key)
        assert enc.encrypt("") == ""

    def test_decrypt_empty_string(self, pii_mod, valid_key):
        enc = pii_mod.PiiEncryptor(valid_key)
        assert enc.decrypt("") == ""

    def test_encrypt_none_passthrough(self, pii_mod, valid_key):
        """None is treated as falsy and returned unchanged."""
        enc = pii_mod.PiiEncryptor(valid_key)
        result = enc.encrypt(None)
        assert result is None

    def test_is_encrypted_true(self, pii_mod, valid_key):
        enc = pii_mod.PiiEncryptor(valid_key)
        ciphertext = enc.encrypt("SSN: 123-45-6789")
        assert pii_mod.PiiEncryptor.is_encrypted(ciphertext) is True

    def test_is_encrypted_false_plaintext(self, pii_mod):
        assert pii_mod.PiiEncryptor.is_encrypted("just plain text") is False

    def test_is_encrypted_false_empty(self, pii_mod):
        assert pii_mod.PiiEncryptor.is_encrypted("") is False

    def test_is_encrypted_false_none(self, pii_mod):
        assert pii_mod.PiiEncryptor.is_encrypted(None) is False

    def test_is_encrypted_false_short_fernet_prefix(self, pii_mod):
        """Short strings that happen to start with the prefix are not encrypted."""
        assert pii_mod.PiiEncryptor.is_encrypted("gAAAAAshort") is False

    def test_enabled_with_key(self, pii_mod, valid_key):
        enc = pii_mod.PiiEncryptor(valid_key)
        assert enc.enabled is True

    def test_enabled_without_key(self, pii_mod):
        enc = pii_mod.PiiEncryptor(None)
        assert enc.enabled is False

    def test_enabled_with_empty_key(self, pii_mod):
        enc = pii_mod.PiiEncryptor("")
        assert enc.enabled is False


class TestGracefulDegradation:
    """Tests for graceful degradation when encryption is disabled."""

    def test_no_key_encrypt_passthrough(self, pii_mod):
        enc = pii_mod.PiiEncryptor(None)
        plaintext = "123-45-6789"
        assert enc.encrypt(plaintext) == plaintext

    def test_no_key_decrypt_passthrough(self, pii_mod):
        enc = pii_mod.PiiEncryptor(None)
        value = "some-stored-value"
        assert enc.decrypt(value) == value

    def test_invalid_key_degrades(self, pii_mod):
        """An invalid key should log error and degrade to passthrough."""
        enc = pii_mod.PiiEncryptor("not-a-valid-base64-key!!!")
        assert enc.enabled is False
        assert enc.encrypt("test") == "test"
        assert enc.decrypt("test") == "test"

    def test_decrypt_plaintext_with_active_key(self, pii_mod, valid_key):
        """Decrypting a non-encrypted value with an active key returns it as-is."""
        enc = pii_mod.PiiEncryptor(valid_key)
        plaintext = "this is not encrypted"
        assert enc.decrypt(plaintext) == plaintext


class TestKeyRotation:
    """Tests for key rotation."""

    def test_rotate_key_success(self, pii_mod, valid_key, second_key):
        enc = pii_mod.PiiEncryptor(valid_key)
        plaintext = "SSN-999-88-7777"
        ciphertext_old = enc.encrypt(plaintext)

        ciphertext_new = enc.rotate_key(valid_key, second_key, ciphertext_old)

        # Verify new ciphertext decrypts with new key
        new_enc = pii_mod.PiiEncryptor(second_key)
        assert new_enc.decrypt(ciphertext_new) == plaintext

        # Verify ciphertexts are different
        assert ciphertext_new != ciphertext_old

    def test_rotate_key_invalid_old_key_raises(self, pii_mod, valid_key, second_key):
        enc = pii_mod.PiiEncryptor(valid_key)
        ciphertext = enc.encrypt("test data")

        with pytest.raises(ValueError, match="Failed to decrypt with old key"):
            enc.rotate_key(second_key, valid_key, ciphertext)  # wrong old key

    def test_rotate_key_preserves_data_integrity(self, pii_mod, valid_key, second_key):
        """Multiple values rotated all decrypt correctly with new key."""
        enc = pii_mod.PiiEncryptor(valid_key)
        new_enc = pii_mod.PiiEncryptor(second_key)

        test_values = [
            "123-45-6789",
            "John Doe",
            "john@example.com",
            "(555) 123-4567",
        ]

        for val in test_values:
            ct_old = enc.encrypt(val)
            ct_new = enc.rotate_key(valid_key, second_key, ct_old)
            assert new_enc.decrypt(ct_new) == val


class TestGetEncryptor:
    """Tests for the module-level singleton."""

    def test_get_encryptor_no_key(self, pii_mod, monkeypatch):
        monkeypatch.delenv("PII_ENCRYPTION_KEY", raising=False)
        pii_mod.reset_encryptor()
        enc = pii_mod.get_encryptor()
        assert enc.enabled is False

    def test_get_encryptor_with_key(self, pii_mod, valid_key, monkeypatch):
        monkeypatch.setenv("PII_ENCRYPTION_KEY", valid_key)
        pii_mod.reset_encryptor()
        enc = pii_mod.get_encryptor()
        assert enc.enabled is True

    def test_get_encryptor_singleton(self, pii_mod, valid_key, monkeypatch):
        monkeypatch.setenv("PII_ENCRYPTION_KEY", valid_key)
        pii_mod.reset_encryptor()
        enc1 = pii_mod.get_encryptor()
        enc2 = pii_mod.get_encryptor()
        assert enc1 is enc2

    def test_reset_encryptor(self, pii_mod, valid_key, monkeypatch):
        monkeypatch.setenv("PII_ENCRYPTION_KEY", valid_key)
        pii_mod.reset_encryptor()
        enc1 = pii_mod.get_encryptor()
        pii_mod.reset_encryptor()
        enc2 = pii_mod.get_encryptor()
        assert enc1 is not enc2


class TestThreadSafety:
    """Basic thread safety tests for PiiEncryptor."""

    def test_concurrent_encrypt_decrypt(self, pii_mod, valid_key):
        enc = pii_mod.PiiEncryptor(valid_key)
        test_values = [f"SSN-{i:03d}-{i:02d}-{i:04d}" for i in range(100)]

        def encrypt_then_decrypt(val):
            ct = enc.encrypt(val)
            pt = enc.decrypt(ct)
            return pt == val

        with ThreadPoolExecutor(max_workers=8) as pool:
            results = list(pool.map(encrypt_then_decrypt, test_values))

        assert all(results), "Some concurrent encrypt/decrypt operations failed"

    def test_concurrent_get_encryptor(self, pii_mod, valid_key, monkeypatch):
        monkeypatch.setenv("PII_ENCRYPTION_KEY", valid_key)
        pii_mod.reset_encryptor()

        def get_enc():
            return pii_mod.get_encryptor()

        with ThreadPoolExecutor(max_workers=8) as pool:
            encryptors = list(pool.map(lambda _: get_enc(), range(20)))

        # All should resolve to enabled encryptors
        assert all(e.enabled for e in encryptors)


# ---------------------------------------------------------------------------
# EncryptedTextField tests (mocked Django field)
# ---------------------------------------------------------------------------


class TestEncryptedTextField:
    """Tests for the EncryptedTextField custom Django field."""

    def test_get_prep_value_encrypts(self, pii_mod, enc_field_mod, valid_key, monkeypatch):
        enc = pii_mod.PiiEncryptor(valid_key)
        monkeypatch.setattr(enc_field_mod, "get_encryptor", lambda: enc)

        field = enc_field_mod.EncryptedTextField()
        plaintext = "123-45-6789"
        stored = field.get_prep_value(plaintext)
        assert stored != plaintext
        assert pii_mod.PiiEncryptor.is_encrypted(stored)

    def test_from_db_value_decrypts(self, pii_mod, enc_field_mod, valid_key, monkeypatch):
        enc = pii_mod.PiiEncryptor(valid_key)
        monkeypatch.setattr(enc_field_mod, "get_encryptor", lambda: enc)

        field = enc_field_mod.EncryptedTextField()
        plaintext = "123-45-6789"
        ciphertext = enc.encrypt(plaintext)
        result = field.from_db_value(ciphertext, None, None)
        assert result == plaintext

    def test_roundtrip_through_field(self, pii_mod, enc_field_mod, valid_key, monkeypatch):
        enc = pii_mod.PiiEncryptor(valid_key)
        monkeypatch.setattr(enc_field_mod, "get_encryptor", lambda: enc)

        field = enc_field_mod.EncryptedTextField()
        plaintext = "My secret PII value"
        stored = field.get_prep_value(plaintext)
        recovered = field.from_db_value(stored, None, None)
        assert recovered == plaintext

    def test_none_passthrough(self, pii_mod, enc_field_mod, valid_key, monkeypatch):
        enc = pii_mod.PiiEncryptor(valid_key)
        monkeypatch.setattr(enc_field_mod, "get_encryptor", lambda: enc)

        field = enc_field_mod.EncryptedTextField()
        assert field.get_prep_value(None) is None
        assert field.from_db_value(None, None, None) is None

    def test_no_key_passthrough(self, pii_mod, enc_field_mod, monkeypatch):
        enc = pii_mod.PiiEncryptor(None)
        monkeypatch.setattr(enc_field_mod, "get_encryptor", lambda: enc)

        field = enc_field_mod.EncryptedTextField()
        plaintext = "123-45-6789"
        stored = field.get_prep_value(plaintext)
        assert stored == plaintext  # No encryption
        recovered = field.from_db_value(stored, None, None)
        assert recovered == plaintext

    def test_deconstruct_path(self, enc_field_mod):
        field = enc_field_mod.EncryptedTextField()
        _, path, _, _ = field.deconstruct()
        assert path == "jobs.encrypted_field.EncryptedTextField"


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Edge case coverage for PII encryption."""

    def test_long_value(self, pii_mod, valid_key):
        """Values up to 10KB should encrypt/decrypt correctly."""
        enc = pii_mod.PiiEncryptor(valid_key)
        plaintext = "A" * 10_000
        ciphertext = enc.encrypt(plaintext)
        assert enc.decrypt(ciphertext) == plaintext

    def test_special_characters(self, pii_mod, valid_key):
        enc = pii_mod.PiiEncryptor(valid_key)
        plaintext = 'Line1\nLine2\tTab\r\nCRLF\x00Null"Quotes\'Single'
        ciphertext = enc.encrypt(plaintext)
        assert enc.decrypt(ciphertext) == plaintext

    def test_multiline_value(self, pii_mod, valid_key):
        enc = pii_mod.PiiEncryptor(valid_key)
        plaintext = "Name: John Doe\nSSN: 123-45-6789\nDOB: 1990-01-15"
        ciphertext = enc.encrypt(plaintext)
        assert enc.decrypt(ciphertext) == plaintext

    def test_different_encryptions_differ(self, pii_mod, valid_key):
        """Two encryptions of the same plaintext produce different ciphertexts
        (Fernet uses a random IV each time)."""
        enc = pii_mod.PiiEncryptor(valid_key)
        plaintext = "same-value"
        ct1 = enc.encrypt(plaintext)
        ct2 = enc.encrypt(plaintext)
        assert ct1 != ct2
        assert enc.decrypt(ct1) == enc.decrypt(ct2) == plaintext

    def test_wrong_key_decrypt_returns_ciphertext(self, pii_mod, valid_key, second_key):
        """Decrypting with wrong key returns ciphertext as-is (graceful)."""
        enc1 = pii_mod.PiiEncryptor(valid_key)
        enc2 = pii_mod.PiiEncryptor(second_key)
        ciphertext = enc1.encrypt("secret data")
        # enc2 should fail to decrypt and return the ciphertext unchanged
        result = enc2.decrypt(ciphertext)
        assert result == ciphertext
