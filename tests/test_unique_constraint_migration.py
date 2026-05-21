"""Tests for Django 6.0 unique_together -> UniqueConstraint migration.

Validates that deprecated unique_together is no longer used in model
definitions and that UniqueConstraint replacements are in place.
"""

import os

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class TestModelsNoUniqueTogetherUsage:
    """Verify unique_together is removed from model Meta classes."""

    def test_models_py_does_not_use_unique_together(self):
        path = os.path.join(_REPO_ROOT, 'coordinator', 'jobs', 'models.py')
        content = open(path).read()
        assert 'unique_together' not in content

    def test_extraction_models_py_does_not_use_unique_together(self):
        path = os.path.join(
            _REPO_ROOT, 'coordinator', 'jobs', 'extraction_models.py'
        )
        content = open(path).read()
        assert 'unique_together' not in content


class TestUniqueConstraintPresent:
    """Verify UniqueConstraint is used in both model files."""

    def test_models_py_uses_unique_constraint(self):
        path = os.path.join(_REPO_ROOT, 'coordinator', 'jobs', 'models.py')
        content = open(path).read()
        assert 'UniqueConstraint' in content
        assert "name='unique_job_page_num'" in content

    def test_extraction_models_py_uses_unique_constraint(self):
        path = os.path.join(
            _REPO_ROOT, 'coordinator', 'jobs', 'extraction_models.py'
        )
        content = open(path).read()
        assert 'UniqueConstraint' in content
        assert "name='unique_chunk'" in content


class TestMigrationFileExists:
    """Verify the migration file was created."""

    def test_migration_0008_exists(self):
        path = os.path.join(
            _REPO_ROOT, 'coordinator', 'jobs', 'migrations',
            '0008_unique_constraint_prep.py',
        )
        assert os.path.isfile(path)

    def test_migration_0008_has_alter_unique_together(self):
        path = os.path.join(
            _REPO_ROOT, 'coordinator', 'jobs', 'migrations',
            '0008_unique_constraint_prep.py',
        )
        content = open(path).read()
        assert 'AlterUniqueTogether' in content

    def test_migration_0008_has_add_constraint(self):
        path = os.path.join(
            _REPO_ROOT, 'coordinator', 'jobs', 'migrations',
            '0008_unique_constraint_prep.py',
        )
        content = open(path).read()
        assert 'AddConstraint' in content

    def test_migration_0008_depends_on_0007(self):
        path = os.path.join(
            _REPO_ROOT, 'coordinator', 'jobs', 'migrations',
            '0008_unique_constraint_prep.py',
        )
        content = open(path).read()
        assert "'0007_apikeyrecord'" in content
