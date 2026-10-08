"""Tests for Django 6.0 unique_together -> UniqueConstraint migration.

Validates that deprecated unique_together is no longer used in model
definitions and that UniqueConstraint replacements are in place.
"""

import ast
import os

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _unique_constraint_names(relative_path):
    path = os.path.join(_REPO_ROOT, *relative_path)
    tree = ast.parse(open(path, encoding="utf-8").read())
    names = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        function_name = (
            node.func.attr
            if isinstance(node.func, ast.Attribute)
            else node.func.id
            if isinstance(node.func, ast.Name)
            else ""
        )
        if function_name != "UniqueConstraint":
            continue
        for keyword in node.keywords:
            if keyword.arg == "name" and isinstance(keyword.value, ast.Constant):
                names.add(keyword.value.value)
    return names


class TestModelsNoUniqueTogetherUsage:
    """Verify unique_together is removed from model Meta classes."""

    def test_models_py_does_not_use_unique_together(self):
        path = os.path.join(_REPO_ROOT, "coordinator", "jobs", "models.py")
        content = open(path).read()
        assert "unique_together" not in content

    def test_extraction_models_py_does_not_use_unique_together(self):
        path = os.path.join(_REPO_ROOT, "coordinator", "jobs", "extraction_models.py")
        content = open(path).read()
        assert "unique_together" not in content


class TestUniqueConstraintPresent:
    """Verify UniqueConstraint is used in both model files."""

    def test_models_py_uses_unique_constraint(self):
        assert "unique_job_page_num" in _unique_constraint_names(
            ("coordinator", "jobs", "models.py")
        )

    def test_extraction_models_py_uses_unique_constraint(self):
        assert "unique_chunk" in _unique_constraint_names(
            ("coordinator", "jobs", "extraction_models.py")
        )


class TestMigrationFileExists:
    """Verify the migration file was created."""

    def test_migration_0008_exists(self):
        path = os.path.join(
            _REPO_ROOT,
            "coordinator",
            "jobs",
            "migrations",
            "0008_unique_constraint_prep.py",
        )
        assert os.path.isfile(path)

    def test_migration_0008_has_alter_unique_together(self):
        path = os.path.join(
            _REPO_ROOT,
            "coordinator",
            "jobs",
            "migrations",
            "0008_unique_constraint_prep.py",
        )
        content = open(path).read()
        assert "AlterUniqueTogether" in content

    def test_migration_0008_has_add_constraint(self):
        path = os.path.join(
            _REPO_ROOT,
            "coordinator",
            "jobs",
            "migrations",
            "0008_unique_constraint_prep.py",
        )
        content = open(path).read()
        assert "AddConstraint" in content

    def test_migration_0008_depends_on_0007(self):
        path = os.path.join(
            _REPO_ROOT,
            "coordinator",
            "jobs",
            "migrations",
            "0008_unique_constraint_prep.py",
        )
        content = open(path).read()
        assert "'0007_apikeyrecord'" in content
