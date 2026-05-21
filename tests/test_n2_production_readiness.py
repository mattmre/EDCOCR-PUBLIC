"""Tests for N2 production proof readiness."""

from __future__ import annotations

import ast
import json
import pathlib

import pytest

CORPUS_DIR = pathlib.Path("tests/fixtures/corpus")


# ---------------------------------------------------------------------------
# Release criteria document
# ---------------------------------------------------------------------------


class TestReleaseCriteria:
    def test_release_criteria_exists(self):
        assert pathlib.Path("docs/release-criteria.md").exists()

    def test_release_criteria_has_automated_gates(self):
        content = pathlib.Path("docs/release-criteria.md").read_text()
        assert "Automated Gates" in content

    def test_release_criteria_has_manual_gates(self):
        content = pathlib.Path("docs/release-criteria.md").read_text()
        assert "Manual Gates" in content

    def test_release_criteria_has_known_blockers(self):
        content = pathlib.Path("docs/release-criteria.md").read_text()
        assert "Known Blockers" in content

    def test_release_criteria_references_ci_jobs(self):
        content = pathlib.Path("docs/release-criteria.md").read_text()
        expected_jobs = [
            "root-lint-and-tests",
            "coordinator-lint-and-tests",
            "helm-lint",
            "terraform-validate",
            "security-audit",
            "release-state-verification",
            "e2e-smoke",
            "resume-regression",
        ]
        for job in expected_jobs:
            assert job in content, f"Missing CI job reference: {job}"

    def test_release_criteria_references_manual_tools(self):
        content = pathlib.Path("docs/release-criteria.md").read_text()
        expected_tools = [
            "cutover_preflight.py",
            "release_evidence.py",
            "corpus_regression.py",
            "release_checklist.py",
        ]
        for tool in expected_tools:
            assert tool in content, f"Missing tool reference: {tool}"


# ---------------------------------------------------------------------------
# Corpus fixtures
# ---------------------------------------------------------------------------


class TestCorpusFixtures:
    def test_cjk_ground_truth_exists(self):
        assert (CORPUS_DIR / "ground-truth" / "sample-cjk.gt.txt").exists()

    def test_cjk_output_exists(self):
        assert (CORPUS_DIR / "output" / "sample-cjk.txt").exists()

    def test_signature_ground_truth_exists(self):
        assert (CORPUS_DIR / "ground-truth" / "sample-signature.gt.txt").exists()

    def test_signature_output_exists(self):
        assert (CORPUS_DIR / "output" / "sample-signature.txt").exists()

    def test_lowres_ground_truth_exists(self):
        assert (CORPUS_DIR / "ground-truth" / "sample-lowres.gt.txt").exists()

    def test_lowres_output_exists(self):
        assert (CORPUS_DIR / "output" / "sample-lowres.txt").exists()

    def test_language_map_covers_all_fixtures(self):
        lang_map = json.loads(
            (CORPUS_DIR / "language-map.json").read_text(encoding="utf-8")
        )
        expected = {
            "sample-clean",
            "sample-degraded",
            "sample-cjk",
            "sample-signature",
            "sample-lowres",
        }
        assert set(lang_map.keys()) >= expected

    def test_category_map_covers_all_fixtures(self):
        cat_map = json.loads(
            (CORPUS_DIR / "category-map.json").read_text(encoding="utf-8")
        )
        expected = {
            "sample-clean",
            "sample-degraded",
            "sample-cjk",
            "sample-signature",
            "sample-lowres",
        }
        assert set(cat_map.keys()) >= expected

    def test_cjk_fixture_contains_cjk_characters(self):
        content = (CORPUS_DIR / "ground-truth" / "sample-cjk.gt.txt").read_text(
            encoding="utf-8"
        )
        assert any(
            "\u4e00" <= c <= "\u9fff" for c in content
        ), "Should contain CJK characters"

    def test_cjk_fixture_contains_japanese(self):
        content = (CORPUS_DIR / "ground-truth" / "sample-cjk.gt.txt").read_text(
            encoding="utf-8"
        )
        # Hiragana/Katakana range
        assert any(
            "\u3040" <= c <= "\u30ff" for c in content
        ), "Should contain Japanese characters"

    def test_cjk_fixture_contains_korean(self):
        content = (CORPUS_DIR / "ground-truth" / "sample-cjk.gt.txt").read_text(
            encoding="utf-8"
        )
        # Hangul Syllables range
        assert any(
            "\uac00" <= c <= "\ud7af" for c in content
        ), "Should contain Korean characters"

    def test_lowres_fixture_contains_greek(self):
        content = (CORPUS_DIR / "ground-truth" / "sample-lowres.gt.txt").read_text(
            encoding="utf-8"
        )
        # Greek character range
        assert any(
            "\u0370" <= c <= "\u03ff" for c in content
        ), "Should contain Greek characters"

    def test_signature_fixture_contains_marker(self):
        content = (
            CORPUS_DIR / "ground-truth" / "sample-signature.gt.txt"
        ).read_text(encoding="utf-8")
        assert "[SIGNATURE_DETECTED]" in content

    def test_ground_truth_and_output_match_for_cjk(self):
        gt = (CORPUS_DIR / "ground-truth" / "sample-cjk.gt.txt").read_text(
            encoding="utf-8"
        )
        out = (CORPUS_DIR / "output" / "sample-cjk.txt").read_text(encoding="utf-8")
        assert gt == out, "Perfect-match fixture: ground truth and output must match"

    def test_ground_truth_and_output_match_for_signature(self):
        gt = (CORPUS_DIR / "ground-truth" / "sample-signature.gt.txt").read_text(
            encoding="utf-8"
        )
        out = (CORPUS_DIR / "output" / "sample-signature.txt").read_text(
            encoding="utf-8"
        )
        assert gt == out, "Perfect-match fixture: ground truth and output must match"

    def test_ground_truth_and_output_match_for_lowres(self):
        gt = (CORPUS_DIR / "ground-truth" / "sample-lowres.gt.txt").read_text(
            encoding="utf-8"
        )
        out = (CORPUS_DIR / "output" / "sample-lowres.txt").read_text(encoding="utf-8")
        assert gt == out, "Perfect-match fixture: ground truth and output must match"


# ---------------------------------------------------------------------------
# Production tooling scripts
# ---------------------------------------------------------------------------


class TestProductionTooling:
    def test_verify_release_state_exists(self):
        assert pathlib.Path("scripts/verify_release_state.py").exists()

    def test_release_evidence_exists(self):
        assert pathlib.Path("scripts/release_evidence.py").exists()

    def test_corpus_regression_exists(self):
        assert pathlib.Path("scripts/corpus_regression.py").exists()

    def test_cutover_preflight_exists(self):
        assert pathlib.Path("scripts/cutover_preflight.py").exists()

    def test_release_checklist_exists(self):
        assert pathlib.Path("scripts/release_checklist.py").exists()

    def test_prepare_release_exists(self):
        assert pathlib.Path("scripts/prepare_release.py").exists()


# ---------------------------------------------------------------------------
# Dashboard singleton reset fixture verification
# ---------------------------------------------------------------------------


class TestDashboardSingletonFix:
    def test_reset_singletons_fixture_resets_before_yield(self):
        """Verify the _reset_singletons fixture resets before yield."""
        source = pathlib.Path("tests/test_api_dashboard_router.py").read_text()
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "_reset_singletons":
                # Find the yield statement index
                yield_index = None
                for i, stmt in enumerate(node.body):
                    if isinstance(stmt, ast.Expr) and isinstance(
                        stmt.value, ast.Yield
                    ):
                        yield_index = i
                        break
                assert yield_index is not None, "Fixture must have a yield"
                assert yield_index > 0, "Must have statements before yield"
                # Check that reset calls appear before yield
                lines_before_yield = [
                    ast.dump(stmt) for stmt in node.body[:yield_index]
                ]
                before_yield_code = "\n".join(lines_before_yield)
                assert "reset" in before_yield_code, "Must call reset() before yield"
                # Also check after yield
                lines_after_yield = [
                    ast.dump(stmt) for stmt in node.body[yield_index + 1 :]
                ]
                after_yield_code = "\n".join(lines_after_yield)
                assert "reset" in after_yield_code, "Must call reset() after yield"
                break
        else:
            pytest.fail("_reset_singletons fixture not found")

    def test_reset_singletons_fixture_has_autouse(self):
        """Verify the _reset_singletons fixture has autouse=True."""
        source = pathlib.Path("tests/test_api_dashboard_router.py").read_text()
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "_reset_singletons":
                # Check decorators for autouse=True
                for decorator in node.decorator_list:
                    if isinstance(decorator, ast.Call):
                        for keyword in decorator.keywords:
                            if keyword.arg == "autouse":
                                assert isinstance(keyword.value, ast.Constant)
                                assert (
                                    keyword.value.value is True
                                ), "autouse must be True"
                                return
                pytest.fail("autouse=True not found on _reset_singletons")
        pytest.fail("_reset_singletons fixture not found")


# ---------------------------------------------------------------------------
# CI release-state-verification job
# ---------------------------------------------------------------------------


class TestCIReleaseVerification:
    def test_ci_yml_has_release_state_verification_job(self):
        ci_path = pathlib.Path(".github/workflows/ci.yml")
        assert ci_path.exists(), "CI workflow must exist"
        content = ci_path.read_text()
        assert "release-state-verification" in content
        assert "verify_release_state.py" in content
