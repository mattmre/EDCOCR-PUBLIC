"""Tests for Wave N1-B: Data Lifecycle and Compliance Controls.

Validates:
- ExtractedEntity.entity_text uses EncryptedTextField (at-rest encryption)
- ExtractedFormValue.field_value uses EncryptedTextField
- DocumentChunk has contains_pii BooleanField
- CustodyEvent.job FK allows null (SET_NULL preserves audit trail)
- cleanup_pii_entities also cleans PII-typed ExtractedEntity records
- purge_pii includes ExtractedEntity deletion and expanded sidecar dirs
- cleanup_completed_jobs pre-deletes extraction data before cascade

All tests use file-content inspection or mock-based approaches to avoid
requiring a live Django/database environment.
"""

import os

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _read_file(relative_path):
    """Read a file relative to the repository root."""
    full_path = os.path.join(_REPO_ROOT, *relative_path.split("/"))
    with open(full_path) as f:
        return f.read()


# ---------------------------------------------------------------------------
# Task 1: Encryption at rest for extraction models
# ---------------------------------------------------------------------------


class TestExtractionModelEncryption:
    """Verify that PII-bearing fields use EncryptedTextField."""

    def test_extraction_models_imports_encrypted_field(self):
        content = _read_file("coordinator/jobs/extraction_models.py")
        assert "from .encrypted_field import EncryptedTextField" in content

    def test_entity_text_uses_encrypted_field(self):
        content = _read_file("coordinator/jobs/extraction_models.py")
        # entity_text should use EncryptedTextField, not plain TextField
        assert "entity_text = EncryptedTextField()" in content
        # Confirm it is NOT a plain TextField
        assert "entity_text = models.TextField()" not in content

    def test_field_value_uses_encrypted_field(self):
        content = _read_file("coordinator/jobs/extraction_models.py")
        # field_value should use EncryptedTextField, not plain TextField
        assert "field_value = EncryptedTextField()" in content
        assert "field_value = models.TextField()" not in content

    def test_chunk_text_not_encrypted(self):
        """DocumentChunk.chunk_text must remain plain TextField for search."""
        content = _read_file("coordinator/jobs/extraction_models.py")
        assert "chunk_text = models.TextField()" in content

    def test_document_chunk_has_contains_pii_field(self):
        content = _read_file("coordinator/jobs/extraction_models.py")
        assert "contains_pii = models.BooleanField(" in content
        assert "default=False" in content


# ---------------------------------------------------------------------------
# Task 3: CustodyEvent FK hardening (SET_NULL)
# ---------------------------------------------------------------------------


class TestCustodyEventSetNull:
    """Verify CustodyEvent.job uses SET_NULL to preserve audit trail."""

    def test_custody_event_job_set_null(self):
        content = _read_file("coordinator/jobs/models.py")
        # Find the CustodyEvent class and check its job FK
        ce_start = content.find("class CustodyEvent(models.Model):")
        assert ce_start != -1, "CustodyEvent class not found"
        ce_section = content[ce_start:ce_start + 600]
        assert "on_delete=models.SET_NULL" in ce_section
        assert "null=True" in ce_section
        assert "blank=True" in ce_section

    def test_custody_event_job_not_cascade(self):
        content = _read_file("coordinator/jobs/models.py")
        ce_start = content.find("class CustodyEvent(models.Model):")
        assert ce_start != -1
        ce_section = content[ce_start:ce_start + 600]
        # The FK should NOT use CASCADE anymore
        assert "on_delete=models.CASCADE" not in ce_section


# ---------------------------------------------------------------------------
# Task 2: Retention enforcement -- extraction data cleanup in tasks.py
# ---------------------------------------------------------------------------


class TestTasksCleanupExtraction:
    """Verify tasks.py includes extraction data in cleanup flows."""

    def test_tasks_imports_extraction_models(self):
        content = _read_file("coordinator/jobs/tasks.py")
        assert "from .extraction_models import ExtractedEntity, ExtractedFormValue" in content

    def test_cleanup_pii_entities_covers_extracted_entities(self):
        """cleanup_pii_entities should delete PII-typed ExtractedEntity records."""
        content = _read_file("coordinator/jobs/tasks.py")
        # Find the cleanup_pii_entities function
        fn_start = content.find("def cleanup_pii_entities():")
        assert fn_start != -1, "cleanup_pii_entities function not found"
        fn_section = content[fn_start:fn_start + 3000]
        assert "ExtractedEntity.objects.filter" in fn_section
        assert "extracted_pii_types" in fn_section
        assert "extracted_pii_deleted" in fn_section

    def test_cleanup_completed_jobs_covers_form_values(self):
        """cleanup_completed_jobs should pre-delete ExtractedFormValue records."""
        content = _read_file("coordinator/jobs/tasks.py")
        fn_start = content.find("def cleanup_completed_jobs():")
        assert fn_start != -1, "cleanup_completed_jobs function not found"
        fn_section = content[fn_start:fn_start + 4000]
        assert "ExtractedFormValue.objects.filter" in fn_section
        assert "ExtractedEntity.objects.filter" in fn_section

    def test_pii_cleanup_respects_litigation_hold(self):
        """cleanup_pii_entities must check litigation hold before deleting."""
        content = _read_file("coordinator/jobs/tasks.py")
        fn_start = content.find("def cleanup_pii_entities():")
        assert fn_start != -1
        fn_section = content[fn_start:fn_start + 500]
        assert "is_litigation_hold_active()" in fn_section


# ---------------------------------------------------------------------------
# Task 4: purge_pii extended to cover ExtractedEntity + sidecar dirs
# ---------------------------------------------------------------------------


class TestPurgePiiExtended:
    """Verify purge_pii covers extraction entities and all sidecar dirs."""

    def test_purge_pii_imports_extracted_entity(self):
        content = _read_file(
            "coordinator/jobs/management/commands/purge_pii.py"
        )
        assert "from jobs.extraction_models import ExtractedEntity" in content

    def test_purge_pii_deletes_extracted_entities(self):
        content = _read_file(
            "coordinator/jobs/management/commands/purge_pii.py"
        )
        assert "ExtractedEntity.objects" in content
        assert "extracted_deleted" in content

    def test_sidecar_dirs_expanded(self):
        """_delete_ner_files_for_job should cover NER, EXTRACTION, HANDWRITING, SIGNATURE."""
        content = _read_file(
            "coordinator/jobs/management/commands/purge_pii.py"
        )
        for subdir in ("NER", "EXTRACTION", "HANDWRITING", "SIGNATURE"):
            assert f'"{subdir}"' in content, f"Missing sidecar dir: {subdir}"

    def test_sidecar_cleanup_uses_loop(self):
        """The sidecar cleanup should iterate over a tuple of directory names."""
        content = _read_file(
            "coordinator/jobs/management/commands/purge_pii.py"
        )
        assert "sidecar_dirs" in content
        assert "for subdir in sidecar_dirs:" in content


# ---------------------------------------------------------------------------
# Structural: EncryptedTextField definition is correct
# ---------------------------------------------------------------------------


class TestEncryptedFieldDefinition:
    """Verify the EncryptedTextField class behaves as expected."""

    def test_encrypted_field_module_exists(self):
        path = os.path.join(
            _REPO_ROOT, "coordinator", "jobs", "encrypted_field.py"
        )
        assert os.path.isfile(path)

    def test_encrypted_field_extends_text_field(self):
        content = _read_file("coordinator/jobs/encrypted_field.py")
        assert "class EncryptedTextField(models.TextField):" in content

    def test_encrypted_field_has_prep_and_from_db(self):
        content = _read_file("coordinator/jobs/encrypted_field.py")
        assert "def get_prep_value(self, value):" in content
        assert "def from_db_value(self, value, expression, connection):" in content


# ---------------------------------------------------------------------------
# Cross-cutting: model field type consistency
# ---------------------------------------------------------------------------


class TestFieldTypeConsistency:
    """Ensure PII fields across all models use EncryptedTextField."""

    def test_pii_entity_uses_encrypted_field(self):
        """PiiEntity.entity_value should be EncryptedTextField (existing)."""
        content = _read_file("coordinator/jobs/models.py")
        assert "entity_value = EncryptedTextField(" in content

    def test_no_plain_textfield_for_pii_data(self):
        """No PII-bearing model should have plain TextField for sensitive data."""
        content = _read_file("coordinator/jobs/extraction_models.py")
        # entity_text and field_value must not be plain TextField
        lines = content.split("\n")
        for line in lines:
            stripped = line.strip()
            if stripped.startswith("entity_text ="):
                assert "EncryptedTextField" in stripped
            if stripped.startswith("field_value ="):
                assert "EncryptedTextField" in stripped
