from __future__ import annotations

import json

from api.database import Tenant, TenantApiKey, get_session_factory, reset_engine
from scripts import seed_playwright_multitenancy


def test_seed_playwright_multitenancy_creates_platform_admin_key(tmp_path, capsys, monkeypatch):
    db_path = tmp_path / "playwright-seed.db"
    monkeypatch.delenv("DEPLOYMENT_ENV", raising=False)
    monkeypatch.delenv("ENVIRONMENT", raising=False)

    exit_code = seed_playwright_multitenancy.main(
        ["--db-path", str(db_path), "--reset"]
    )

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is True
    assert payload["tenant_id"].startswith("tenant_")
    assert payload["key_id"].startswith("key_")
    assert payload["api_key"].startswith("ocr_")
    assert (
        payload["playwright_env"]["PLAYWRIGHT_MULTITENANCY_PLATFORM_ADMIN_KEY"]
        == payload["api_key"]
    )

    session = get_session_factory(str(db_path))()
    try:
        tenant = session.get(Tenant, payload["tenant_id"])
        key = session.get(TenantApiKey, payload["key_id"])
        assert tenant is not None
        assert key is not None
        assert key.tenant_id == tenant.tenant_id
    finally:
        session.close()
        reset_engine()


def test_seed_playwright_multitenancy_rejects_production_env(tmp_path, capsys, monkeypatch):
    db_path = tmp_path / "playwright-seed.db"
    monkeypatch.setenv("DEPLOYMENT_ENV", "production")

    exit_code = seed_playwright_multitenancy.main(["--db-path", str(db_path)])

    assert exit_code == 1
    payload = json.loads(capsys.readouterr().err)
    assert payload["error"] == "production_blocked"
    reset_engine()
