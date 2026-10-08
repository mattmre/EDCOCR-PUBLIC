"""Transparent encrypted-at-rest storage for coordinator webhook secrets."""

import base64
import hashlib
import logging
import os

from django.db import models

logger = logging.getLogger(__name__)


def _fernet():
    from cryptography.fernet import Fernet

    dedicated_key = os.environ.get("WEBHOOK_SECRET_KEY", "").strip()
    production = os.environ.get("DEPLOYMENT_ENV", "").strip().lower() in {
        "production",
        "prod",
    }
    if production and not dedicated_key:
        raise RuntimeError("WEBHOOK_SECRET_KEY is required in production")
    raw = (
        dedicated_key
        or os.environ.get("OCR_API_KEY", "").strip()
        or "dev-key-not-for-production"
    )
    return Fernet(base64.urlsafe_b64encode(hashlib.sha256(raw.encode()).digest()))


def encrypt_webhook_secret(value: str) -> str:
    return _fernet().encrypt(value.encode()).decode()


def decrypt_webhook_secret(value: str) -> str:
    try:
        return _fernet().decrypt(value.encode()).decode()
    except Exception as exc:
        production = os.environ.get("DEPLOYMENT_ENV", "").strip().lower() in {
            "production",
            "prod",
        }
        if production:
            raise RuntimeError("Stored webhook secret cannot be decrypted") from exc
        logger.warning("Treating a pre-migration webhook secret as plaintext")
        return value


def is_encrypted_webhook_secret(value: str) -> bool:
    if not isinstance(value, str) or not value.startswith("gAAAA"):
        return False
    try:
        decoded = base64.urlsafe_b64decode(value + "=" * (-len(value) % 4))
    except (ValueError, TypeError):
        return False
    if len(decoded) < 73 or decoded[0] != 0x80:
        return False
    try:
        _fernet().decrypt(value.encode())
        return True
    except Exception as exc:
        raise RuntimeError(
            "Stored webhook secret is Fernet ciphertext but cannot be decrypted; "
            "refusing to double-encrypt it"
        ) from exc


class WebhookSecretField(models.TextField):
    """Store Fernet ciphertext while exposing plaintext to task code."""

    def get_prep_value(self, value):
        value = super().get_prep_value(value)
        if value is None or value == "":
            return value
        if is_encrypted_webhook_secret(value):
            return value
        return encrypt_webhook_secret(value)

    def from_db_value(self, value, expression, connection):
        del expression, connection
        if value is None or value == "":
            return value
        return decrypt_webhook_secret(value)

    def deconstruct(self):
        name, _path, args, kwargs = super().deconstruct()
        return name, "jobs.webhook_secret_field.WebhookSecretField", args, kwargs
