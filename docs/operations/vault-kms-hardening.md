# Vault and KMS Credential Hardening Guide

This document covers the production configuration of HashiCorp Vault and AWS KMS
credential backends in EDCOCR's `credential_manager.py`.

---

## Table of Contents

1. [Vault KV v1 vs KV v2 Path Differences](#vault-kv-v1-vs-kv-v2-path-differences)
2. [Vault AppRole Authentication Setup](#vault-approle-authentication-setup)
3. [AWS KMS Key Policy Requirements](#aws-kms-key-policy-requirements)
4. [Credential Rotation Configuration](#credential-rotation-configuration)
5. [Monitoring Credential Refresh Events](#monitoring-credential-refresh-events)
6. [Secret Scanning Prevention](#secret-scanning-prevention)
7. [Docker Compose with Vault-Backed Credentials](#docker-compose-with-vault-backed-credentials)
8. [Kubernetes Vault Agent Injector](#kubernetes-vault-agent-injector)

---

## Vault KV v1 vs KV v2 Path Differences

HashiCorp Vault has two versions of its Key-Value secrets engine. The path
format and response structure differ between them.

### KV v1

- **Path format**: `secret/<path>` (no `/data/` segment)
- **Response**: `{ "data": { "key": "value" } }` -- flat single layer
- **No versioning**: Writes overwrite the previous value
- **Configuration**: Set `VAULT_KV_VERSION=1`

### KV v2 (default)

- **Path format**: `secret/data/<path>` (note the `/data/` segment)
- **Response**: `{ "data": { "data": { "key": "value" }, "metadata": { ... } } }` -- nested
- **Versioned**: Maintains history of secret versions
- **Configuration**: Set `VAULT_KV_VERSION=2` (or leave unset; this is the default)

### Configuration

```bash
# KV v2 (default -- no env var needed)
VAULT_ADDR=https://vault.example.com:8200
VAULT_TOKEN=hvs.XXXXXXXXXXXXX
VAULT_SECRET_PATH=secret/data/ocr-local

# KV v1 (explicit)
VAULT_KV_VERSION=1
VAULT_ADDR=https://vault.example.com:8200
VAULT_TOKEN=hvs.XXXXXXXXXXXXX
VAULT_SECRET_PATH=secret/ocr-local
```

The credential manager automatically strips any accidental `/data/` segment
from the path when using KV v1, so `secret/data/ocr-local` is corrected to
`secret/ocr-local` transparently.

### Programmatic Override

```python
from credential_manager import VaultBackend, CredentialManager

# Force KV v1 regardless of env var
vault = VaultBackend(vault_kv_version=1)
manager = CredentialManager(backends=[vault])
```

---

## Vault AppRole Authentication Setup

AppRole is the recommended machine-to-machine authentication method for
production Vault deployments.

### 1. Enable AppRole Auth Method

```bash
vault auth enable approle
```

### 2. Create a Policy for EDCOCR

```hcl
# vault-policy-ocr-local.hcl
path "secret/data/ocr-local" {
  capabilities = ["read"]
}

# If using KV v1:
path "secret/ocr-local" {
  capabilities = ["read"]
}
```

```bash
vault policy write ocr-local vault-policy-ocr-local.hcl
```

### 3. Create an AppRole

```bash
vault write auth/approle/role/ocr-local \
    token_policies="ocr-local" \
    token_ttl=1h \
    token_max_ttl=4h \
    secret_id_ttl=720h \
    secret_id_num_uses=0
```

### 4. Retrieve role_id and secret_id

```bash
# role_id is stable (embed in config)
vault read auth/approle/role/ocr-local/role-id

# secret_id is rotatable (deliver securely)
vault write -f auth/approle/role/ocr-local/secret-id
```

### 5. Login to Obtain a Token

```bash
vault write auth/approle/login \
    role_id="<role_id>" \
    secret_id="<secret_id>"
```

The returned `client_token` is what you set as `VAULT_TOKEN` for EDCOCR.
In production, automate this login step in your container entrypoint or
init container.

---

## AWS KMS Key Policy Requirements

The `AWSKMSBackend` decrypts credentials stored as base64-encoded KMS
ciphertexts in environment variables prefixed with `KMS_ENC_`.

### Required IAM Permissions

The IAM role or user running the OCR service needs:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "AllowKMSDecrypt",
      "Effect": "Allow",
      "Action": [
        "kms:Decrypt"
      ],
      "Resource": "arn:aws:kms:us-east-1:123456789012:key/<key-id>"
    }
  ]
}
```

### KMS Key Policy

The KMS key itself must grant usage to the service role:

```json
{
  "Sid": "AllowOCRServiceDecrypt",
  "Effect": "Allow",
  "Principal": {
    "AWS": "arn:aws:iam::123456789012:role/ocr-local-service-role"
  },
  "Action": "kms:Decrypt",
  "Resource": "*"
}
```

### Encrypting a Credential

```bash
# Encrypt
aws kms encrypt \
  --key-id alias/ocr-local-credentials \
  --plaintext "my_secret_password" \
  --output text \
  --query CiphertextBlob

# The output is already base64-encoded -- use it directly
export KMS_ENC_POSTGRES_PASSWORD="AQICAHj..."
```

### Error Handling

The KMS backend provides clear error messages for common failures:

| Error Code | Message | Action |
|---|---|---|
| `InvalidCiphertextException` | Ciphertext is corrupt or was encrypted with a different key | Re-encrypt the credential with the correct KMS key |
| `AccessDeniedException` | IAM permissions are missing | Grant `kms:Decrypt` to the service role on the target key |
| `KMSInternalException` | Transient KMS error | Automatic retry with exponential backoff (up to 3 attempts) |

---

## Credential Rotation Configuration

The `CredentialManager` supports automatic credential rotation detection via
a background daemon thread.

### Configuration Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `refresh_interval_seconds` | `float` or `None` | `None` | How often to re-fetch credentials (seconds). `None` disables refresh. |
| `on_credential_refreshed` | `Callable[[str], None]` or `None` | `None` | Callback invoked when a credential value changes during refresh. |

### Example: Rotation with Logging

```python
import logging
from credential_manager import CredentialManager

logger = logging.getLogger("credential_rotation")

def on_rotated(credential_name: str) -> None:
    logger.warning(
        "Credential '%s' was rotated -- downstream consumers should refresh",
        credential_name)

manager = CredentialManager(
    refresh_interval_seconds=300,  # Check every 5 minutes
    on_credential_refreshed=on_rotated)

# Normal usage
db_password = manager.get("POSTGRES_PASSWORD")
```

### Behavior Details

- Only credentials that have been previously accessed via `get` are
  re-fetched during rotation checks.
- If a credential value changes, the internal cache is updated and the
  callback is invoked with the credential name.
- If the callback raises an exception, it is logged and the refresh loop
  continues (it does not crash the background thread).
- Call `manager.stop_refresh` during graceful shutdown to cleanly stop
  the background thread.

---

## Monitoring Credential Refresh Events

### Structured Logging

The credential manager logs refresh events at `INFO` level:

```
INFO credential_manager Credential 'POSTGRES_PASSWORD' value changed during refresh
INFO credential_manager Credential refresh thread started (interval=300s)
INFO credential_manager Credential refresh thread stopped
WARNING credential_manager on_credential_refreshed callback failed for POSTGRES_PASSWORD
```

### Integration with Pipeline Monitoring

Wire the rotation callback into your existing monitoring:

```python
from credential_manager import CredentialManager

# Prometheus counter (example)
from prometheus_client import Counter

rotation_counter = Counter(
    "ocr_credential_rotations_total",
    "Total credential rotation events",
    ["credential_name"])

def on_rotated(name: str) -> None:
    rotation_counter.labels(credential_name=name).inc

manager = CredentialManager(
    refresh_interval_seconds=300,
    on_credential_refreshed=on_rotated)
```

---

## Secret Scanning Prevention

### What NOT to Put in .env Files

Never store production credentials in files that could be committed to git:

- `.env` files (even if gitignored -- accidents happen)
- `docker-compose.yml` inline environment values
- `values.yaml` Helm overrides checked into the repo
- Shell scripts with hardcoded passwords
- Jupyter notebooks with embedded credentials

### Safe Alternatives

| Method | Use Case |
|---|---|
| Vault | Primary production secret store |
| AWS KMS encrypted env vars | When Vault is unavailable |
| Kubernetes Secrets + Vault Agent | Kubernetes deployments |
| Credential file (base64-encoded) | Air-gapped environments |
| Environment variables (from CI/CD) | Ephemeral build/test contexts |

### Git Pre-commit Protection

The project's pre-commit hooks include ruff checks. Consider adding a
secret scanning hook:

```yaml
# .pre-commit-config.yaml
- repo: https://github.com/Yelp/detect-secrets
  rev: v1.4.0
  hooks:
    - id: detect-secrets
      args: ['--baseline', '.secrets.baseline']
```

---

## Docker Compose with Vault-Backed Credentials

### Example: Vault Token Auth

```yaml
# docker-compose.yml
services:
  ocr-processor:
    image: ocr-local:latest
    environment:
      # Vault configuration
      VAULT_ADDR: "https://vault.internal:8200"
      VAULT_TOKEN_FILE: /run/secrets/vault_token
      VAULT_SECRET_PATH: "secret/data/ocr-local"
      VAULT_KV_VERSION: "2"

      # No plaintext secrets in compose file!
      # All credentials are fetched from Vault at runtime.

    secrets:
      - vault_token
    volumes:
      - ./ocr_source:/app/ocr_source:ro
      - ./ocr_output:/app/ocr_output

secrets:
  vault_token:
    file: /run/vault-agent/token  # Populated by Vault Agent sidecar
```

### Example: KMS Encrypted Environment Variables

```yaml
services:
  ocr-processor:
    image: ocr-local:latest
    environment:
      AWS_KMS_REGION: "us-east-1"
      # Each value is a base64-encoded KMS ciphertext
      KMS_ENC_POSTGRES_PASSWORD: "AQICAHj..."
      KMS_ENC_RABBITMQ_PASSWORD: "AQICAHj..."
      KMS_ENC_DJANGO_SECRET_KEY: "AQICAHj..."
```

---

## Kubernetes Vault Agent Injector

For Kubernetes deployments, use the Vault Agent Injector to automatically
inject secrets into pods without storing them in Kubernetes Secrets.

### Prerequisites

1. Install the Vault Helm chart with the injector enabled:
   ```bash
   helm install vault hashicorp/vault \
     --set "injector.enabled=true" \
     --set "server.enabled=false"
   ```

2. Configure Kubernetes auth in Vault:
   ```bash
   vault auth enable kubernetes
   vault write auth/kubernetes/config \
     kubernetes_host="https://$KUBERNETES_PORT_443_TCP_ADDR:443"
   vault write auth/kubernetes/role/ocr-local \
     bound_service_account_names=ocr-local \
     bound_service_account_namespaces=ocr \
     policies=ocr-local \
     ttl=1h
   ```

### Pod Annotations

```yaml
# In your Helm values or deployment spec
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ocr-coordinator
spec:
  template:
    metadata:
      annotations:
        vault.hashicorp.com/agent-inject: "true"
        vault.hashicorp.com/role: "ocr-local"
        vault.hashicorp.com/agent-inject-secret-credentials: "secret/data/ocr-local"
        vault.hashicorp.com/agent-inject-template-credentials: |
          {{- with secret "secret/data/ocr-local" -}}
          export POSTGRES_PASSWORD="{{ .Data.data.POSTGRES_PASSWORD }}"
          export RABBITMQ_PASSWORD="{{ .Data.data.RABBITMQ_PASSWORD }}"
          export DJANGO_SECRET_KEY="{{ .Data.data.DJANGO_SECRET_KEY }}"
          export S3_ACCESS_KEY="{{ .Data.data.S3_ACCESS_KEY }}"
          export S3_SECRET_KEY="{{ .Data.data.S3_SECRET_KEY }}"
          {{- end -}}
    spec:
      serviceAccountName: ocr-local
      containers:
        - name: coordinator
          image: ocr-local-coordinator:latest
          command:
            - /bin/sh
            - -c
            - "source /vault/secrets/credentials && exec gunicorn coordinator.wsgi:application"
```

### Using with EDCOCR Helm Chart

Add the Vault annotations to the coordinator deployment in your values overlay:

```yaml
# values-production.yaml
coordinator:
  podAnnotations:
    vault.hashicorp.com/agent-inject: "true"
    vault.hashicorp.com/role: "ocr-local"
    vault.hashicorp.com/agent-inject-secret-env: "secret/data/ocr-local"

  # Disable Kubernetes secret mount since Vault provides credentials
  existingSecret: ""
```

---

## Credential Backend Priority

The default backend chain is:

1. **Environment variables** (always checked first)
2. **HashiCorp Vault** (if `hvac` installed and `VAULT_ADDR` + `VAULT_TOKEN` set)
3. **AWS Secrets Manager** (if `boto3` installed and `AWS_SECRET_NAME` set)
4. **AWS KMS** (if `boto3` installed and `KMS_ENC_*` env vars present)
5. **Encrypted file** (if `CREDENTIAL_FILE_PATH` set)

The first backend that returns a non-`None` value for a given credential
name wins. This allows layered overrides -- for example, an env var can
override a Vault secret for local development without changing the Vault
store.

---

*Last Updated: 2026-05-20*
