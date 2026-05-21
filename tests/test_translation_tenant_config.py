"""Tests for translation tenant config + glossary models and REST API.

Covers the Django models (structural + DB round-trip), the
``get_tenant_policy`` hydration helper, and the REST endpoints under
``api/routers/translation_admin.py``.

Django-backed tests run only when ``DJANGO_SETTINGS_MODULE`` is already
set in the environment (the coordinator/jobs/tests path).  This file
intentionally does NOT setdefault DJANGO_SETTINGS_MODULE -- doing so
would trigger pytest-django's session-scoped fixture and break root
tests run via ``-c pytest.ini``.  See .

REST-endpoint tests use FastAPI TestClient against an isolated app
instance, gated identically.
"""
from __future__ import annotations

import importlib.util
import os
import sys
from typing import Iterator

import pytest

# ---------------------------------------------------------------------------
# Make the coordinator package importable without Django setup.  This is
# the same pattern used by ``tests/test_tenant_metrics.py`` and
# ``tests/test_layoutlm_worker.py``.
# ---------------------------------------------------------------------------
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_COORDINATOR_DIR = os.path.join(_REPO_ROOT, "coordinator")
if _COORDINATOR_DIR not in sys.path:
    sys.path.insert(0, _COORDINATOR_DIR)

# Only attempt django.setup() when DJANGO_SETTINGS_MODULE is *already*
# set in the environment.  Setting it here would trigger pytest-django's
# session fixture and break tests running under the root pytest config.
_DJANGO_OK = False
if (
    importlib.util.find_spec("django") is not None
    and os.environ.get("DJANGO_SETTINGS_MODULE")
):
    try:
        import django

        if not django.conf.settings.configured:
            django.setup()
        _DJANGO_OK = True
    except Exception:
        _DJANGO_OK = False

skip_no_django = pytest.mark.skipif(
    not _DJANGO_OK,
    reason="Django settings not configured (run via coordinator/pytest.ini)",
)


# ---------------------------------------------------------------------------
# Structural tests (no DB needed)
# ---------------------------------------------------------------------------


@skip_no_django
class TestTranslationTenantConfigModelStructure:
    def test_model_has_tenant_id_unique(self):
        from jobs.models import TranslationTenantConfig

        field = TranslationTenantConfig._meta.get_field("tenant_id")
        assert field.unique is True
        assert field.db_index is True
        assert field.max_length == 128

    def test_model_target_languages_default_list(self):
        from jobs.models import TranslationTenantConfig

        field = TranslationTenantConfig._meta.get_field("target_languages")
        # JSONField default factory; default() should return a list
        default = field.default()
        assert default == []

    def test_model_allow_nc_licensed_default_false(self):
        from jobs.models import TranslationTenantConfig

        field = TranslationTenantConfig._meta.get_field("allow_nc_licensed")
        assert field.default is False

    def test_model_require_certified_default_false(self):
        from jobs.models import TranslationTenantConfig

        field = TranslationTenantConfig._meta.get_field("require_certified")
        assert field.default is False

    def test_model_default_quality_tier_standard(self):
        from jobs.models import TranslationTenantConfig

        field = TranslationTenantConfig._meta.get_field("default_quality_tier")
        assert field.default == "standard"


@skip_no_django
class TestGlossaryEntryModelStructure:
    def test_tenant_id_indexed(self):
        from jobs.models import GlossaryEntry

        field = GlossaryEntry._meta.get_field("tenant_id")
        assert field.db_index is True

    def test_priority_default_100(self):
        from jobs.models import GlossaryEntry

        field = GlossaryEntry._meta.get_field("priority")
        assert field.default == 100

    def test_unique_constraint_present(self):
        """Unique constraint on (tenant_id, source_term, source_lang, target_lang)
        for non-regex entries."""
        from jobs.models import GlossaryEntry

        constraints = [
            c for c in GlossaryEntry._meta.constraints
            if c.name == "unique_literal_glossary_entry"
        ]
        assert len(constraints) == 1
        constraint = constraints[0]
        assert set(constraint.fields) == {
            "tenant_id", "source_term", "source_lang", "target_lang",
        }

    def test_indexes_include_tenant_lang_pair(self):
        from jobs.models import GlossaryEntry

        idx_fields = {tuple(idx.fields) for idx in GlossaryEntry._meta.indexes}
        assert ("tenant_id",) in idx_fields
        assert ("tenant_id", "source_lang", "target_lang") in idx_fields


# ---------------------------------------------------------------------------
# DB round-trip tests
# ---------------------------------------------------------------------------


@skip_no_django
@pytest.mark.django_db
class TestTranslationTenantConfigRoundTrip:
    def test_create_and_retrieve(self):
        from jobs.models import TranslationTenantConfig

        row = TranslationTenantConfig.objects.create(
            tenant_id="tenant-a",
            target_languages=["en", "fr"],
            preferred_engines=["opus_mt", "nllb_200"],
            allow_nc_licensed=True,
            require_certified=True,
            default_quality_tier="legal",
        )
        fetched = TranslationTenantConfig.objects.get(tenant_id="tenant-a")
        assert fetched.target_languages == ["en", "fr"]
        assert fetched.preferred_engines == ["opus_mt", "nllb_200"]
        assert fetched.allow_nc_licensed is True
        assert fetched.require_certified is True
        assert fetched.default_quality_tier == "legal"
        assert fetched.pk == row.pk

    def test_tenant_id_unique(self):
        from django.db import IntegrityError, transaction
        from jobs.models import TranslationTenantConfig

        TranslationTenantConfig.objects.create(tenant_id="tenant-a")
        with pytest.raises(IntegrityError):
            with transaction.atomic():
                TranslationTenantConfig.objects.create(tenant_id="tenant-a")

    def test_json_field_round_trip(self):
        from jobs.models import TranslationTenantConfig

        TranslationTenantConfig.objects.create(
            tenant_id="t-json",
            target_languages=["fr", "es", "de"],
        )
        fetched = TranslationTenantConfig.objects.get(tenant_id="t-json")
        assert isinstance(fetched.target_languages, list)
        assert fetched.target_languages == ["fr", "es", "de"]


@skip_no_django
@pytest.mark.django_db
class TestGlossaryEntryRoundTrip:
    def test_create_and_retrieve(self):
        from jobs.models import GlossaryEntry

        row = GlossaryEntry.objects.create(
            tenant_id="tenant-a",
            source_term="Party",
            target_term="Partie",
            source_lang="en",
            target_lang="fr",
            priority=10,
        )
        fetched = GlossaryEntry.objects.get(pk=row.pk)
        assert fetched.source_term == "Party"
        assert fetched.target_term == "Partie"
        assert fetched.priority == 10
        assert fetched.case_sensitive is False
        assert fetched.is_regex is False

    def test_unique_literal_constraint(self):
        from django.db import IntegrityError, transaction
        from jobs.models import GlossaryEntry

        GlossaryEntry.objects.create(
            tenant_id="tenant-a",
            source_term="Party",
            target_term="Partie",
            source_lang="en",
            target_lang="fr",
        )
        with pytest.raises(IntegrityError):
            with transaction.atomic():
                GlossaryEntry.objects.create(
                    tenant_id="tenant-a",
                    source_term="Party",
                    target_term="OtherTarget",
                    source_lang="en",
                    target_lang="fr",
                )

    def test_regex_entries_can_duplicate_source_term(self):
        from jobs.models import GlossaryEntry

        GlossaryEntry.objects.create(
            tenant_id="tenant-a",
            source_term=r"\d+",
            target_term="X",
            source_lang="en",
            target_lang="fr",
            is_regex=True,
        )
        # Same source_term + langs + tenant, but is_regex=True -> no constraint
        GlossaryEntry.objects.create(
            tenant_id="tenant-a",
            source_term=r"\d+",
            target_term="Y",
            source_lang="en",
            target_lang="fr",
            is_regex=True,
        )
        assert GlossaryEntry.objects.filter(
            tenant_id="tenant-a", is_regex=True
        ).count() == 2


# ---------------------------------------------------------------------------
# load_tenant_glossary -- DB-backed
# ---------------------------------------------------------------------------


@skip_no_django
@pytest.mark.django_db
class TestLoadTenantGlossary:
    def test_filters_by_tenant_and_lang_pair(self):
        from jobs.models import GlossaryEntry as DjangoGE

        from ocr_local.translation.glossary import load_tenant_glossary

        DjangoGE.objects.create(
            tenant_id="tenant-a",
            source_term="A",
            target_term="A_fr",
            source_lang="en",
            target_lang="fr",
            priority=10,
        )
        DjangoGE.objects.create(
            tenant_id="tenant-a",
            source_term="B",
            target_term="B_es",
            source_lang="en",
            target_lang="es",
            priority=5,
        )
        DjangoGE.objects.create(
            tenant_id="tenant-b",
            source_term="C",
            target_term="C_fr",
            source_lang="en",
            target_lang="fr",
            priority=1,
        )

        out = load_tenant_glossary("tenant-a", "en", "fr")
        assert len(out) == 1
        assert out[0].source_term == "A"

    def test_orders_by_priority_ascending(self):
        from jobs.models import GlossaryEntry as DjangoGE

        from ocr_local.translation.glossary import load_tenant_glossary

        DjangoGE.objects.create(
            tenant_id="t-prio", source_term="hi", target_term="bonjour",
            source_lang="en", target_lang="fr", priority=200,
        )
        DjangoGE.objects.create(
            tenant_id="t-prio", source_term="bye", target_term="au revoir",
            source_lang="en", target_lang="fr", priority=10,
        )
        DjangoGE.objects.create(
            tenant_id="t-prio", source_term="hello", target_term="salut",
            source_lang="en", target_lang="fr", priority=100,
        )
        out = load_tenant_glossary("t-prio", "en", "fr")
        assert [e.priority for e in out] == [10, 100, 200]


# ---------------------------------------------------------------------------
# get_tenant_policy hydration
# ---------------------------------------------------------------------------


@skip_no_django
@pytest.mark.django_db
class TestGetTenantPolicy:
    def test_returns_default_for_none(self):
        from ocr_local.translation.policy import get_tenant_policy

        policy = get_tenant_policy(None)
        assert policy.tenant_id == "default"
        assert policy.allow_nllb_commercial is True  # dataclass default

    def test_returns_default_when_no_row(self):
        from ocr_local.translation.policy import get_tenant_policy

        policy = get_tenant_policy("never-existed")
        assert policy.tenant_id == "never-existed"

    def test_hydrates_from_existing_row(self):
        from jobs.models import TranslationTenantConfig

        from ocr_local.translation.policy import get_tenant_policy

        TranslationTenantConfig.objects.create(
            tenant_id="tenant-hydrated",
            preferred_engines=["opus_mt"],
            allow_nc_licensed=False,
        )
        policy = get_tenant_policy("tenant-hydrated")
        assert policy.tenant_id == "tenant-hydrated"
        # allow_nc_licensed=False -> allow_nllb_commercial=False
        assert policy.allow_nllb_commercial is False
        assert policy.allowed_engine_ids == ["opus_mt"]

    def test_allow_nc_licensed_true_propagates(self):
        from jobs.models import TranslationTenantConfig

        from ocr_local.translation.policy import get_tenant_policy

        TranslationTenantConfig.objects.create(
            tenant_id="t-nc",
            allow_nc_licensed=True,
        )
        policy = get_tenant_policy("t-nc")
        assert policy.allow_nllb_commercial is True


# ---------------------------------------------------------------------------
# REST endpoint tests (Wave M2 admin router)
# ---------------------------------------------------------------------------


@pytest.fixture
def admin_app(monkeypatch):
    """Build an isolated FastAPI app with only the admin router.

    Bypasses the api_key_middleware so endpoint contract is exercised
    directly.  Auth is covered by a separate test that wires the
    middleware in.
    """
    fastapi = pytest.importorskip("fastapi")
    pytest.importorskip("httpx")

    monkeypatch.setenv("ENABLE_TRANSLATION_API", "true")
    from api.routers.translation_admin import router

    app = fastapi.FastAPI()
    app.include_router(router)
    return app


@pytest.fixture
def admin_client(admin_app) -> Iterator:
    from fastapi.testclient import TestClient

    with TestClient(admin_app) as c:
        yield c


@skip_no_django
@pytest.mark.django_db(transaction=True)
class TestTenantConfigEndpoints:
    def test_get_404_when_disabled(self, admin_client, monkeypatch):
        monkeypatch.setenv("ENABLE_TRANSLATION_API", "false")
        resp = admin_client.get("/api/v1/translation/tenants/t1/config")
        assert resp.status_code == 404
        assert "disabled" in resp.json()["detail"]

    def test_get_404_when_no_row(self, admin_client):
        resp = admin_client.get("/api/v1/translation/tenants/missing/config")
        assert resp.status_code == 404

    def test_put_creates_row(self, admin_client):
        payload = {
            "target_languages": ["fr"],
            "preferred_engines": ["opus_mt"],
            "allow_nc_licensed": False,
            "require_certified": False,
            "default_quality_tier": "standard",
        }
        resp = admin_client.put(
            "/api/v1/translation/tenants/t-create/config", json=payload,
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["tenant_id"] == "t-create"
        assert body["target_languages"] == ["fr"]
        assert body["preferred_engines"] == ["opus_mt"]

    def test_put_updates_existing(self, admin_client):
        # Create
        admin_client.put(
            "/api/v1/translation/tenants/t-up/config",
            json={
                "target_languages": ["fr"],
                "preferred_engines": [],
                "allow_nc_licensed": False,
                "require_certified": False,
                "default_quality_tier": "standard",
            },
        )
        # Update
        resp = admin_client.put(
            "/api/v1/translation/tenants/t-up/config",
            json={
                "target_languages": ["es", "de"],
                "preferred_engines": ["opus_mt"],
                "allow_nc_licensed": True,
                "require_certified": True,
                "default_quality_tier": "legal",
            },
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["target_languages"] == ["es", "de"]
        assert body["allow_nc_licensed"] is True

    def test_put_invalid_quality_tier_422(self, admin_client):
        resp = admin_client.put(
            "/api/v1/translation/tenants/t-bad/config",
            json={
                "target_languages": ["fr"],
                "preferred_engines": [],
                "allow_nc_licensed": False,
                "require_certified": False,
                "default_quality_tier": "garbage",
            },
        )
        assert resp.status_code == 422


@skip_no_django
@pytest.mark.django_db(transaction=True)
class TestGlossaryEndpoints:
    def test_create_glossary_entry(self, admin_client):
        resp = admin_client.post(
            "/api/v1/translation/tenants/t-g/glossary",
            json={
                "source_term": "Party",
                "target_term": "Partie",
                "source_lang": "en",
                "target_lang": "fr",
                "case_sensitive": False,
                "is_regex": False,
                "priority": 50,
            },
        )
        assert resp.status_code == 201
        body = resp.json()
        assert body["source_term"] == "Party"
        assert body["target_term"] == "Partie"
        assert body["tenant_id"] == "t-g"
        assert body["priority"] == 50
        assert "id" in body

    def test_create_duplicate_409(self, admin_client):
        payload = {
            "source_term": "Party",
            "target_term": "Partie",
            "source_lang": "en",
            "target_lang": "fr",
            "is_regex": False,
        }
        first = admin_client.post(
            "/api/v1/translation/tenants/t-dup/glossary", json=payload,
        )
        assert first.status_code == 201
        second = admin_client.post(
            "/api/v1/translation/tenants/t-dup/glossary",
            json={**payload, "target_term": "different"},
        )
        assert second.status_code == 409

    def test_list_glossary_paginated(self, admin_client):
        for i in range(5):
            admin_client.post(
                "/api/v1/translation/tenants/t-list/glossary",
                json={
                    "source_term": f"term_{i}",
                    "target_term": f"target_{i}",
                    "source_lang": "en",
                    "target_lang": "fr",
                    "priority": 10 + i,
                },
            )
        resp = admin_client.get(
            "/api/v1/translation/tenants/t-list/glossary?page=1&page_size=3",
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["total"] == 5
        assert len(body["entries"]) == 3
        assert body["page_size"] == 3
        # Ordered by priority ascending
        priorities = [e["priority"] for e in body["entries"]]
        assert priorities == sorted(priorities)

    def test_list_filters_by_lang(self, admin_client):
        for tgt in ["fr", "es", "de"]:
            admin_client.post(
                "/api/v1/translation/tenants/t-flt/glossary",
                json={
                    "source_term": "x",
                    "target_term": "y",
                    "source_lang": "en",
                    "target_lang": tgt,
                },
            )
        resp = admin_client.get(
            "/api/v1/translation/tenants/t-flt/glossary?target_lang=fr",
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["total"] == 1
        assert body["entries"][0]["target_lang"] == "fr"

    def test_list_tenant_isolation(self, admin_client):
        """Glossary entries from another tenant are NOT visible."""
        admin_client.post(
            "/api/v1/translation/tenants/tenant-a/glossary",
            json={
                "source_term": "secret",
                "target_term": "secret_fr",
                "source_lang": "en",
                "target_lang": "fr",
            },
        )
        resp = admin_client.get(
            "/api/v1/translation/tenants/tenant-b/glossary",
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["total"] == 0

    def test_patch_updates_fields(self, admin_client):
        create = admin_client.post(
            "/api/v1/translation/tenants/t-p/glossary",
            json={
                "source_term": "Party",
                "target_term": "Partie",
                "source_lang": "en",
                "target_lang": "fr",
            },
        )
        entry_id = create.json()["id"]
        resp = admin_client.patch(
            f"/api/v1/translation/tenants/t-p/glossary/{entry_id}",
            json={"target_term": "PartieUpdated", "priority": 5},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["target_term"] == "PartieUpdated"
        assert body["priority"] == 5

    def test_patch_404_other_tenant(self, admin_client):
        create = admin_client.post(
            "/api/v1/translation/tenants/t-iso-a/glossary",
            json={
                "source_term": "x",
                "target_term": "y",
                "source_lang": "en",
                "target_lang": "fr",
            },
        )
        entry_id = create.json()["id"]
        resp = admin_client.patch(
            f"/api/v1/translation/tenants/t-iso-b/glossary/{entry_id}",
            json={"target_term": "z"},
        )
        assert resp.status_code == 404

    def test_delete_entry(self, admin_client):
        create = admin_client.post(
            "/api/v1/translation/tenants/t-d/glossary",
            json={
                "source_term": "x",
                "target_term": "y",
                "source_lang": "en",
                "target_lang": "fr",
            },
        )
        entry_id = create.json()["id"]
        resp = admin_client.delete(
            f"/api/v1/translation/tenants/t-d/glossary/{entry_id}",
        )
        assert resp.status_code == 204
        # Subsequent GET returns 0
        list_resp = admin_client.get(
            "/api/v1/translation/tenants/t-d/glossary",
        )
        assert list_resp.json()["total"] == 0

    def test_delete_404(self, admin_client):
        resp = admin_client.delete(
            "/api/v1/translation/tenants/t-x/glossary/99999",
        )
        assert resp.status_code == 404

    def test_create_invalid_payload_422(self, admin_client):
        resp = admin_client.post(
            "/api/v1/translation/tenants/t-bad/glossary",
            json={"source_term": "", "target_term": "y"},
        )
        assert resp.status_code == 422

    def test_endpoints_404_when_disabled(self, admin_client, monkeypatch):
        monkeypatch.setenv("ENABLE_TRANSLATION_API", "false")
        for path in [
            "/api/v1/translation/tenants/t/config",
            "/api/v1/translation/tenants/t/glossary",
        ]:
            resp = admin_client.get(path)
            assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Auth: 401 without API key when middleware is wired in
# ---------------------------------------------------------------------------


def test_admin_endpoints_require_api_key(monkeypatch):
    fastapi = pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient

    monkeypatch.setenv("OCR_API_KEY", "test-key-abc")
    monkeypatch.setenv("ENABLE_TRANSLATION_API", "true")

    import importlib

    import api.auth as _auth
    import api.config as _config

    importlib.reload(_config)
    importlib.reload(_auth)

    from api.routers.translation_admin import router

    app = fastapi.FastAPI()
    app.middleware("http")(_auth.api_key_middleware)
    app.include_router(router)
    c = TestClient(app)
    resp = c.get("/api/v1/translation/tenants/t1/config")
    assert resp.status_code == 401


# ---------------------------------------------------------------------------
# certified=True is blocked at write-time even when require_certified is
# set on the tenant config (gotcha #88).
# ---------------------------------------------------------------------------


def test_require_certified_does_not_unlock_certified_sidecar(tmp_path):
    """Tenant ``require_certified=True`` must NOT bypass write-time
    enforcement of ``certified=False`` in translation sidecars.

    Verifies : even when a tenant config sets
    ``require_certified=True`` (a *policy* flag), the sidecar writer
    still rejects ``certified=True`` payloads -- promotion happens
    elsewhere via the strong-auth review queue.
    """
    from ocr_local.translation.models import (
        DocumentTranslation,
        PageTranslation,
        SpanTranslation,
    )
    from ocr_local.translation.sidecar import (
        SchemaValidationError,
        write_translation_json,
    )

    span = SpanTranslation(
        span_id="s0",
        source_text="Hello",
        target_text="Bonjour",
        source_bbox=[0.0, 0.0, 100.0, 12.0],
        source_bboxes=[[0.0, 0.0, 100.0, 12.0]],
        source_language="en",
        target_language="fr",
        confidence=0.95,
        quality_score=None,
        engine_id="passthrough",
    )
    page = PageTranslation(page_num=1, spans=[span])
    doc = DocumentTranslation(
        schema_version="1.0",
        document_id="doc1",
        source_file="/tmp/in.pdf",
        source_language="en",
        target_language="fr",
        pages=[page],
        engine={"id": "passthrough"},
        certified=True,  # forbidden
    )
    with pytest.raises(SchemaValidationError):
        write_translation_json(doc, str(tmp_path), subfolder="")
