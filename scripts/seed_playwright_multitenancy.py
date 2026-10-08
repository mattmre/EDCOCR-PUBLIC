"""Seed deterministic multi-tenancy data for Playwright Phase 7 runs."""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Sequence

from api.database import (
    Job,
    Tenant,
    TenantApiKey,
    UsageRecord,
    get_engine,
    get_session_factory,
    reset_engine,
)
from api.tenant_manager import create_api_key, create_tenant


def _is_production_env() -> bool:
    for env_name in ("DEPLOYMENT_ENV", "ENVIRONMENT"):
        raw = os.environ.get(env_name, "").strip().lower()
        if raw == "production":
            return True
    return False


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Seed a deterministic platform-admin tenant key for Playwright feature-flag coverage.",
    )
    parser.add_argument("--db-path", default=None, help="SQLite DB path to target.")
    parser.add_argument(
        "--tenant-name",
        default="Playwright Platform Admin",
        help="Tenant name to create or replace.",
    )
    parser.add_argument(
        "--admin-email",
        default="playwright-platform-admin@example.com",
        help="Admin email to store on the seeded tenant.",
    )
    parser.add_argument(
        "--key-name",
        default="playwright-platform-admin",
        help="Display name for the seeded API key.",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Delete any existing tenant with the same name before seeding.",
    )
    return parser.parse_args(argv)


def _delete_existing_named_tenants(session, tenant_name: str) -> int:
    existing = session.query(Tenant).filter(Tenant.name == tenant_name).all()
    deleted = 0
    for tenant in existing:
        session.query(TenantApiKey).filter(TenantApiKey.tenant_id == tenant.tenant_id).delete()
        session.query(UsageRecord).filter(UsageRecord.tenant_id == tenant.tenant_id).delete()
        session.query(Job).filter(Job.tenant_id == tenant.tenant_id).delete()
        session.delete(tenant)
        deleted += 1

    if deleted:
        session.commit()

    return deleted


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)

    if _is_production_env():
        print(
            json.dumps(
                {
                    "ok": False,
                    "error": "production_blocked",
                    "message": (
                        "Refusing to seed Playwright multi-tenancy data in a "
                        "production environment."
                    ),
                }
            ),
            file=sys.stderr,
        )
        return 1

    reset_engine()
    get_engine(args.db_path)
    session = get_session_factory(args.db_path)()
    deleted = 0
    try:
        if args.reset:
            deleted = _delete_existing_named_tenants(session, args.tenant_name)

        tenant = create_tenant(
            name=args.tenant_name,
            display_name="Playwright Platform Admin",
            tier="enterprise",
            max_concurrent_jobs=8,
            max_pages_per_month=50000,
            allowed_features=["docintel", "transforms", "stamps", "multitenancy"],
            admin_email=args.admin_email,
            session=session,
        )
        key_id, raw_key = create_api_key(
            tenant.tenant_id,
            name=args.key_name,
            permissions=["submit", "read", "admin", "platform_admin"],
            session=session,
        )
        tenant_id = tenant.tenant_id
    finally:
        session.close()

    print(
        json.dumps(
            {
                "ok": True,
                "db_path": args.db_path,
                "tenant_id": tenant_id,
                "tenant_name": args.tenant_name,
                "deleted_existing_tenants": deleted,
                "key_id": key_id,
                "api_key": raw_key,
                "playwright_env": {
                    "PLAYWRIGHT_EXPECT_MULTITENANCY_ENABLED": "true",
                    "PLAYWRIGHT_MULTITENANCY_PLATFORM_ADMIN_KEY": raw_key,
                    "PLAYWRIGHT_FEATURE_FLAGS_API_KEY": raw_key,
                },
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
