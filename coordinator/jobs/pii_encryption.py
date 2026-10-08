"""PII encryption at rest using Fernet (AES-128-CBC).

Provides transparent encrypt/decrypt for PII/PHI entity values stored in the
coordinator database.  Key is loaded from the ``PII_ENCRYPTION_KEY`` environment
variable.

R2-RC-21 hardening:
- When a key is configured, decrypt of Fernet-shaped ciphertext **fails closed**
  on InvalidToken (no longer returns ciphertext as "plaintext").
- New ciphertext uses an explicit ``pii:v1:`` envelope. Normal writes always
  encrypt their input; key rotation uses a separate authenticated raw API.
- Production (``DEPLOYMENT_ENV=production``) refuses plaintext storage unless
  ``PII_ENCRYPTION_ALLOW_PLAINTEXT`` is explicitly true.

Thread-safe: the ``Fernet`` instance is immutable once created.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)

try:
    from cryptography.fernet import Fernet, InvalidToken
except ImportError:  # pragma: no cover
    Fernet = None  # type: ignore[assignment,misc]
    InvalidToken = Exception  # type: ignore[assignment,misc]

# New ciphertext carries an explicit application envelope. Raw Fernet tokens
# remain readable for legacy rows, but write paths never infer ciphertext from
# an attacker-controlled token-shaped prefix.
_PII_ENVELOPE = "pii:v1:"
_FERNET_PREFIX = "gAAAAA"

_TRUE = frozenset({"1", "true", "yes", "on"})


class PiiDecryptError(RuntimeError):
    """Raised when a configured key cannot decrypt Fernet-shaped ciphertext."""


class PiiEncryptionConfigError(RuntimeError):
    """Raised when production forbids plaintext PII storage without a valid key."""


def _allow_plaintext() -> bool:
    return os.environ.get("PII_ENCRYPTION_ALLOW_PLAINTEXT", "").strip().lower() in _TRUE


def _is_production() -> bool:
    return os.environ.get("DEPLOYMENT_ENV", "").strip().lower() == "production"


class PiiEncryptor:
    """Encrypt / decrypt PII values using Fernet (AES-128-CBC + HMAC-SHA256)."""

    def __init__(self, key: Optional[str] = None):
        self._key = key
        self._fernet: Optional["Fernet"] = None  # type: ignore[type-arg]

        if key:
            if Fernet is None:
                msg = (
                    "PII_ENCRYPTION_KEY is set but the 'cryptography' package "
                    "is not installed."
                )
                if _is_production() and not _allow_plaintext():
                    raise PiiEncryptionConfigError(
                        msg + " Refusing plaintext in production."
                    )
                logger.error(msg + " PII will be stored in PLAINTEXT.")
                self._key = None
                return
            try:
                self._fernet = Fernet(
                    key.encode("utf-8") if isinstance(key, str) else key
                )
            except Exception:
                msg = "Invalid PII_ENCRYPTION_KEY"
                if _is_production() and not _allow_plaintext():
                    raise PiiEncryptionConfigError(
                        msg + " — refusing plaintext in production (R2-RC-21)."
                    ) from None
                logger.exception("%s — PII will be stored in PLAINTEXT.", msg)
                self._key = None
                self._fernet = None
        else:
            if _is_production() and not _allow_plaintext():
                raise PiiEncryptionConfigError(
                    "PII_ENCRYPTION_KEY is required when DEPLOYMENT_ENV=production "
                    "(R2-RC-21). Set PII_ENCRYPTION_ALLOW_PLAINTEXT=true only for "
                    "non-forensic lab waivers."
                )
            logger.warning(
                "PII_ENCRYPTION_KEY is not configured.  "
                "PII entity values will be stored in PLAINTEXT.  "
                "Set PII_ENCRYPTION_KEY to enable encryption at rest."
            )

    @property
    def enabled(self) -> bool:
        """Return True when encryption is active."""
        return self._fernet is not None

    def encrypt(self, plaintext: str) -> str:
        """Encrypt *plaintext* and return a base64-encoded ciphertext string.

        When encryption is disabled, returns *plaintext* unchanged.
        Input is always plaintext. Ciphertext rotation must use rotate_key();
        guessing from a user-controlled prefix can bypass encryption at rest.
        """
        if not self._fernet or not plaintext:
            return plaintext
        token = self._fernet.encrypt(plaintext.encode("utf-8"))
        return _PII_ENVELOPE + token.decode("utf-8")

    def decrypt(self, ciphertext: str) -> str:
        """Decrypt *ciphertext* and return the original plaintext.

        When encryption is disabled, returns *ciphertext* unchanged.
        Explicitly enveloped and legacy raw-Fernet values authenticate or fail
        closed. Other values pass through for legacy plaintext migration.
        """
        if not self._fernet or not ciphertext:
            return ciphertext
        token = self._ciphertext_token(ciphertext)
        if token is None:
            return ciphertext
        try:
            return self._fernet.decrypt(token.encode("utf-8")).decode("utf-8")
        except Exception as exc:
            # R2-RC-21: do NOT return ciphertext as application "plaintext" —
            # that path caused permanent double-encrypt corruption on write-back.
            logger.error(
                "Failed to decrypt Fernet-shaped PII value with configured key "
                "(R2-RC-21 fail-closed)"
            )
            raise PiiDecryptError(
                "PII decrypt failed with configured key; refusing fail-open "
                "passthrough that enables double-encrypt (R2-RC-21)."
            ) from exc

    @staticmethod
    def _ciphertext_token(value: str) -> Optional[str]:
        """Extract an explicit token, with read-only legacy compatibility."""
        if not value or not isinstance(value, str):
            return None
        if value.startswith(_PII_ENVELOPE):
            token = value[len(_PII_ENVELOPE) :]
            return token or None
        if value.startswith(_FERNET_PREFIX) and len(value) >= 100:
            return value
        return None

    @staticmethod
    def is_encrypted(value: str) -> bool:
        """Return whether *value* carries the unambiguous PII v1 envelope."""
        return isinstance(value, str) and value.startswith(_PII_ENVELOPE)

    @staticmethod
    def is_rotation_ciphertext_candidate(value: str) -> bool:
        """Identify explicit or legacy ciphertext only for raw-key rotation.

        Normal application writes must never use this heuristic. Rotation is
        different: a legacy token-shaped value must authenticate under the old
        key or fail closed instead of being nested under the new key.
        """
        return bool(PiiEncryptor._ciphertext_token(value))

    def rotate_key(self, old_key: str, new_key: str, ciphertext: str) -> str:
        """Re-encrypt *ciphertext* from *old_key* to *new_key*."""
        if Fernet is None:
            raise RuntimeError("cryptography package is not installed")

        old_fernet = Fernet(old_key.encode("utf-8"))
        new_fernet = Fernet(new_key.encode("utf-8"))

        token = self._ciphertext_token(ciphertext)
        if token is None:
            raise ValueError("PII key rotation requires authenticated ciphertext")
        try:
            plaintext = old_fernet.decrypt(token.encode("utf-8"))
        except InvalidToken as exc:
            raise ValueError(
                "Failed to decrypt with old key — value may already use a "
                "different key or be plaintext"
            ) from exc

        return _PII_ENVELOPE + new_fernet.encrypt(plaintext).decode("utf-8")


_encryptor: Optional[PiiEncryptor] = None


def get_encryptor() -> PiiEncryptor:
    """Return the module-level ``PiiEncryptor`` singleton."""
    global _encryptor
    if _encryptor is None:
        _encryptor = PiiEncryptor(os.environ.get("PII_ENCRYPTION_KEY"))
    return _encryptor


def reset_encryptor() -> None:
    """Reset the singleton (useful for tests or key rotation)."""
    global _encryptor
    _encryptor = None
