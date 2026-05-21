"""Tests for docs/compliance/risk-register.md existence and structure."""

import pathlib
import re

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
REGISTER_PATH = REPO_ROOT / "docs" / "compliance" / "risk-register.md"


@pytest.fixture(scope="module")
def register_content() -> str:
    """Read the risk register once for all tests in this module."""
    assert REGISTER_PATH.exists(), f"Risk register not found at {REGISTER_PATH}"
    return REGISTER_PATH.read_text(encoding="utf-8")


class TestRiskRegisterExists:
    """Verify the risk register file exists and is non-empty."""

    def test_file_exists(self):
        assert REGISTER_PATH.exists(), f"Expected {REGISTER_PATH} to exist"

    def test_file_is_not_empty(self, register_content: str):
        assert len(register_content.strip()) > 0, "Risk register must not be empty"

    def test_file_is_substantial(self, register_content: str):
        # A proper risk register with 20 entries should be well over 5000 chars
        assert len(register_content) > 5000, "Risk register appears too short"


class TestRiskRegisterSections:
    """Verify key sections are present in the risk register."""

    def test_has_purpose_section(self, register_content: str):
        assert "## Purpose" in register_content

    def test_has_risk_assessment_methodology(self, register_content: str):
        assert "## Risk Assessment Methodology" in register_content

    def test_has_likelihood_scale(self, register_content: str):
        assert "### Likelihood Scale" in register_content
        assert "| 1 | Rare |" in register_content
        assert "| 5 | Almost Certain |" in register_content

    def test_has_impact_scale(self, register_content: str):
        assert "### Impact Scale" in register_content
        assert "| 1 | Negligible |" in register_content
        assert "| 5 | Critical |" in register_content

    def test_has_risk_score_calculation(self, register_content: str):
        assert "Risk Score = Likelihood x Impact" in register_content

    def test_has_risk_level_definitions(self, register_content: str):
        assert "| 1-4 | Low |" in register_content
        assert "| 5-9 | Medium |" in register_content
        assert "| 10-15 | High |" in register_content
        assert "| 16-25 | Critical |" in register_content

    def test_has_heat_map(self, register_content: str):
        assert "### Heat Map" in register_content

    def test_has_risk_register_heading(self, register_content: str):
        assert "## Risk Register" in register_content

    def test_has_risk_summary(self, register_content: str):
        assert "## Risk Summary by Rating" in register_content

    def test_has_risk_acceptance_register(self, register_content: str):
        assert "## Risk Acceptance Register" in register_content

    def test_has_review_cadence(self, register_content: str):
        assert "## Risk Review Cadence" in register_content

    def test_has_quarterly_review_process(self, register_content: str):
        assert "### Quarterly Review Process" in register_content

    def test_has_appendix_risk_level_definitions(self, register_content: str):
        assert "## Appendix A: Risk Level Definitions" in register_content
        assert "### Low (Score 1-4)" in register_content
        assert "### Medium (Score 5-9)" in register_content
        assert "### High (Score 10-15)" in register_content
        assert "### Critical (Score 16-25)" in register_content

    def test_has_related_documents_appendix(self, register_content: str):
        assert "## Appendix B: Related Documents" in register_content
        assert "incident-response-plan.md" in register_content
        assert "data-retention-policy.md" in register_content

    def test_has_revision_history(self, register_content: str):
        assert "## Revision History" in register_content


class TestRiskRegisterEntries:
    """Verify the risk register contains the expected number and types of risks."""

    def test_has_at_least_15_risks(self, register_content: str):
        risk_ids = re.findall(r"### RISK-(\d{3}):", register_content)
        assert len(risk_ids) >= 15, (
            f"Expected at least 15 risk entries, found {len(risk_ids)}"
        )

    def test_has_20_risks(self, register_content: str):
        risk_ids = re.findall(r"### RISK-(\d{3}):", register_content)
        assert len(risk_ids) == 20, (
            f"Expected 20 risk entries, found {len(risk_ids)}"
        )

    def test_risk_ids_are_sequential(self, register_content: str):
        risk_ids = re.findall(r"### RISK-(\d{3}):", register_content)
        expected = [f"{i:03d}" for i in range(1, len(risk_ids) + 1)]
        assert risk_ids == expected, (
            f"Risk IDs are not sequential: {risk_ids}"
        )

    def test_each_risk_has_required_fields(self, register_content: str):
        required_fields = [
            "**ID**",
            "**Category**",
            "**Likelihood**",
            "**Impact**",
            "**Risk Score**",
            "**Description**",
            "**Existing Controls**",
            "**Residual Risk**",
            "**Mitigation Status**",
            "**Owner**",
        ]
        risk_sections = re.split(r"### RISK-\d{3}:", register_content)[1:]
        for i, section in enumerate(risk_sections, 1):
            for field in required_fields:
                assert field in section, (
                    f"RISK-{i:03d} is missing required field: {field}"
                )


class TestRiskCategories:
    """Verify risks cover all required categories."""

    def test_has_security_risks(self, register_content: str):
        sec_count = len(re.findall(r"\| \*\*Category\*\* \| SEC \|", register_content))
        assert sec_count >= 3, (
            f"Expected at least 3 security risks, found {sec_count}"
        )

    def test_has_availability_risks(self, register_content: str):
        avl_count = len(re.findall(r"\| \*\*Category\*\* \| AVL \|", register_content))
        assert avl_count >= 3, (
            f"Expected at least 3 availability risks, found {avl_count}"
        )

    def test_has_compliance_risks(self, register_content: str):
        cmp_count = len(re.findall(r"\| \*\*Category\*\* \| CMP \|", register_content))
        assert cmp_count >= 2, (
            f"Expected at least 2 compliance risks, found {cmp_count}"
        )

    def test_has_data_integrity_risks(self, register_content: str):
        dat_count = len(re.findall(r"\| \*\*Category\*\* \| DAT \|", register_content))
        assert dat_count >= 2, (
            f"Expected at least 2 data integrity risks, found {dat_count}"
        )

    def test_has_operational_risks(self, register_content: str):
        ops_count = len(re.findall(r"\| \*\*Category\*\* \| OPS \|", register_content))
        assert ops_count >= 1, (
            f"Expected at least 1 operational risk, found {ops_count}"
        )


class TestRiskCoverage:
    """Verify specific risk topics are covered as required by the task."""

    def test_covers_data_breach(self, register_content: str):
        assert "data breach" in register_content.lower() or "PII/PHI" in register_content

    def test_covers_unauthorized_access(self, register_content: str):
        assert "unauthorized" in register_content.lower()

    def test_covers_api_key_compromise(self, register_content: str):
        assert "API key" in register_content
        # Specific risk entry for API key compromise
        assert "RISK-013" in register_content

    def test_covers_injection_attacks(self, register_content: str):
        assert "injection" in register_content.lower() or "Injection" in register_content

    def test_covers_gpu_failure(self, register_content: str):
        assert "GPU" in register_content

    def test_covers_storage_exhaustion(self, register_content: str):
        assert "storage" in register_content.lower()
        assert "exhaustion" in register_content.lower() or "exhausted" in register_content.lower()

    def test_covers_queue_backlog(self, register_content: str):
        assert "queue" in register_content.lower()

    def test_covers_model_drift(self, register_content: str):
        assert "model" in register_content.lower()
        assert "drift" in register_content.lower() or "degradation" in register_content.lower()

    def test_covers_pii_exposure(self, register_content: str):
        assert "PII" in register_content

    def test_covers_audit_trail_gaps(self, register_content: str):
        assert "audit trail" in register_content.lower() or "audit logging" in register_content.lower()

    def test_covers_retention_violations(self, register_content: str):
        assert "retention" in register_content.lower()

    def test_covers_single_point_of_failure(self, register_content: str):
        assert "single point of failure" in register_content.lower()

    def test_covers_network_partitions(self, register_content: str):
        assert "network partition" in register_content.lower() or "partition" in register_content.lower()

    def test_covers_dependency_vulnerabilities(self, register_content: str):
        assert "dependency" in register_content.lower() or "supply chain" in register_content.lower()


class TestComplianceReadmeIncludesRiskRegister:
    """Verify the compliance README references the risk register."""

    def test_readme_references_risk_register(self):
        readme_path = REPO_ROOT / "docs" / "compliance" / "README.md"
        assert readme_path.exists(), f"Compliance README not found at {readme_path}"
        content = readme_path.read_text(encoding="utf-8")
        assert "risk-register.md" in content, (
            "Compliance README must reference risk-register.md"
        )
