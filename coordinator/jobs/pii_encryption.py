"""PII encryption at rest using Fernet (AES-128-CBC).

Provides transparent encrypt/decrypt for PII/PHI entity values stored in the
coordinator database.  Key is loaded from the ``PII_ENCRYPTION_KEY`` environment
variable.  When the key is not set, the module degrades gracefully and stores
plaintext (backward compatible with existing deployments).

Thread-safe: the ``Fernet`` instance is immutable once created.

Usage::

    from coordinator.jobs.pii_encryption import get_encryptor

    enc = get_encryptor()
    ciphertext = enc.encrypt("123-45-6789")
    plaintext  = enc.decrypt(ciphertext)
"""

import base64
import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)

# Lazy-import cryptography so the module can be loaded even when the
# package is absent (graceful degradation path).
try:
    from cryptography.fernet import Fernet, InvalidToken
except ImportError:  # pragma: no cover
    Fernet = None  # type: ignore[assignment,misc]
    InvalidToken = Exception  # type: ignore[assignment,misc]

# Fernet tokens always start with "gAAAAA" (version byte 0x80 + timestamp).
_FERNET_PREFIX = "gAAAAA"


class PiiEncryptor:
    """Encrypt / decrypt PII values using Fernet (AES-128-CBC + HMAC-SHA256).

    Parameters
    ----------
    key : str or None
        A URL-safe base64-encoded 32-byte key.  If *None*, encryption is
        disabled and all methods pass values through unchanged.
    """

    def __init__(self, key: Optional[str] = None):
        self._key = key
        self._fernet: Optional["Fernet"] = None  # type: ignore[type-arg]

        if key:
            if Fernet is None:
                logger.error(
                    "PII_ENCRYPTION_KEY is set but the 'cryptography' package "
                    "is not installed.  PII will be stored in PLAINTEXT."
                )
                self._key = None
                return
            try:
                self._fernet = Fernet(key.encode("utf-8") if isinstance(key, str) else key)
            except Exception:
                logger.exception(
                    "Invalid PII_ENCRYPTION_KEY — PII will be stored in PLAINTEXT."
                )
                self._key = None
                self._fernet = None
        else:
            logger.warning(
                "PII_ENCRYPTION_KEY is not configured.  "
                "PII entity values will be stored in PLAINTEXT.  "
                "Set PII_ENCRYPTION_KEY to enable encryption at rest."
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def enabled(self) -> bool:
        """Return *True* when encryption is active."""
        return self._fernet is not None

    def encrypt(self, plaintext: str) -> str:
        """Encrypt *plaintext* and return a base64-encoded ciphertext string.

        When encryption is disabled, returns *plaintext* unchanged.
        """
        if not self._fernet or not plaintext:
            return plaintext
        token = self._fernet.encrypt(plaintext.encode("utf-8"))
        return token.decode("utf-8")

    def decrypt(self, ciphertext: str) -> str:
        """Decrypt *ciphertext* and return the original plaintext.

        When encryption is disabled or *ciphertext* does not look like a
        Fernet token, returns *ciphertext* unchanged (graceful fallback for
        mixed encrypted/plaintext data during migration).
        """
        if not self._fernet or not ciphertext:
            return ciphertext
        if not self.is_encrypted(ciphertext):
            return ciphertext
        try:
            return self._fernet.decrypt(ciphertext.encode("utf-8")).decode("utf-8")
        except Exception:
            # Value may be legacy plaintext — return as-is rather than crash.
            logger.debug("Failed to decrypt value — returning as plaintext")
            return ciphertext

    @staticmethod
    def is_encrypted(value: str) -> bool:
        """Heuristic check whether *value* is a Fernet-encrypted token.

        Fernet tokens are URL-safe base64 strings that start with a
        well-known prefix (version byte ``0x80`` followed by an 8-byte
        big-endian timestamp).
        """
        if not value or not isinstance(value, str):
            return False
        if not value.startswith(_FERNET_PREFIX):
            return False
        # Quick length sanity: Fernet tokens are always multiples of 4 chars
        # and at least 120 chars long (1 + 8 + 16 + ... padded).
        if len(value) < 100:
            return False
        # Verify it is valid base64
        try:
            base64.urlsafe_b64decode(value.encode("utf-8"))
            return True
        except Exception:
            return False

    def rotate_key(self, old_key: str, new_key: str, ciphertext: str) -> str:
        """Re-encrypt *ciphertext* from *old_key* to *new_key*.

        Parameters
        ----------
        old_key : str
            The current encryption key (URL-safe base64).
        new_key : str
            The new encryption key (URL-safe base64).
        ciphertext : str
            The value encrypted with *old_key*.

        Returns
        -------
        str
            The value re-encrypted with *new_key*.

        Raises
        ------
        ValueError
            If decryption with *old_key* fails.
        """
        if Fernet is None:
            raise RuntimeError("cryptography package is not installed")

        old_fernet = Fernet(old_key.encode("utf-8"))
        new_fernet = Fernet(new_key.encode("utf-8"))

        try:
            plaintext = old_fernet.decrypt(ciphertext.encode("utf-8"))
        except InvalidToken as exc:
            raise ValueError(
                "Failed to decrypt with old key — value may already use a "
                "different key or be plaintext"
            ) from exc

        return new_fernet.encrypt(plaintext).decode("utf-8")


# ------------------------------------------------------------------
# Module-level singleton
# ------------------------------------------------------------------

_encryptor: Optional[PiiEncryptor] = None


def get_encryptor() -> PiiEncryptor:
    """Return the module-level ``PiiEncryptor`` singleton.

    The key is read from ``PII_ENCRYPTION_KEY`` on first call.
    """
    global _encryptor
    if _encryptor is None:
        _encryptor = PiiEncryptor(os.environ.get("PII_ENCRYPTION_KEY"))
    return _encryptor


def reset_encryptor() -> None:
    """Reset the singleton (useful for tests or key rotation)."""
    global _encryptor
    _encryptor = None
