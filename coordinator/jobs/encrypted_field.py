"""Django custom model field that encrypts values at rest using Fernet.

Provides transparent encryption for PII/PHI data.  Application code reads
and writes plaintext; the database stores ciphertext.

When ``PII_ENCRYPTION_KEY`` is not configured, the field behaves exactly
like a standard ``TextField`` (backward compatible).

Usage in a Django model::

    from coordinator.jobs.encrypted_field import EncryptedTextField

    class PiiEntity(models.Model):
        entity_value = EncryptedTextField(help_text="The extracted PII/PHI text")
"""

from django.db import models

from .pii_encryption import get_encryptor


class EncryptedTextField(models.TextField):
    """A ``TextField`` that transparently encrypts/decrypts values at rest.

    On ``get_prep_value`` (write to DB): encrypt plaintext to ciphertext.
    On ``from_db_value`` (read from DB): decrypt ciphertext to plaintext.

    When encryption is disabled (no ``PII_ENCRYPTION_KEY``), values pass
    through unchanged, preserving backward compatibility.
    """

    def get_prep_value(self, value):
        """Encrypt before storing in the database."""
        value = super().get_prep_value(value)
        if value is None:
            return value
        enc = get_encryptor()
        return enc.encrypt(value)

    def from_db_value(self, value, expression, connection):
        """Decrypt after loading from the database."""
        if value is None:
            return value
        enc = get_encryptor()
        return enc.decrypt(value)

    def deconstruct(self):
        """Return field constructor arguments for migration serialization.

        EncryptedTextField has no extra kwargs beyond ``TextField``, so we
        return the parent's deconstruct but with our own module path.
        """
        name, path, args, kwargs = super().deconstruct()
        # Use the correct import path for this field class
        path = "jobs.encrypted_field.EncryptedTextField"
        return name, path, args, kwargs
