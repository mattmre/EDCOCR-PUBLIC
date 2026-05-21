"""Tests for docs/compliance/incident-response-plan.md existence and structure."""

import pathlib

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
PLAN_PATH = REPO_ROOT / "docs" / "compliance" / "incident-response-plan.md"


@pytest.fixture(scope="module")
def plan_content() -> str:
    """Read the incident response plan once for all tests in this module."""
    assert PLAN_PATH.exists(), f"Incident response plan not found at {PLAN_PATH}"
    return PLAN_PATH.read_text(encoding="utf-8")


class TestIncidentResponsePlanExists:
    """Verify the incident response plan file exists."""

    def test_file_exists(self):
        assert PLAN_PATH.exists(), f"Expected {PLAN_PATH} to exist"

    def test_file_is_not_empty(self, plan_content: str):
        assert len(plan_content.strip()) > 0, "Incident response plan must not be empty"


class TestIncidentResponsePlanSections:
    """Verify key sections are present in the incident response plan."""

    def test_has_purpose_and_scope(self, plan_content: str):
        assert "## 1. Purpose and Scope" in plan_content

    def test_has_severity_levels(self, plan_content: str):
        assert "## 2. Severity Levels" in plan_content

    def test_has_p0_through_p3(self, plan_content: str):
        assert "| P0 |" in plan_content
        assert "| P1 |" in plan_content
        assert "| P2 |" in plan_content
        assert "| P3 |" in plan_content

    def test_has_incident_roles(self, plan_content: str):
        assert "## 3. Incident Response Roles" in plan_content
        assert "Incident Commander" in plan_content
        assert "Technical Lead" in plan_content
        assert "Communications Lead" in plan_content
        assert "Scribe" in plan_content

    def test_has_detection_section(self, plan_content: str):
        assert "## 4. Detection" in plan_content

    def test_has_response_procedures(self, plan_content: str):
        assert "## 5. Response Procedures" in plan_content

    def test_has_escalation_paths(self, plan_content: str):
        assert "## 6. Escalation Paths" in plan_content

    def test_has_communication_templates(self, plan_content: str):
        assert "## 7. Communication Templates" in plan_content
        assert "Internal Incident Notification" in plan_content
        assert "External Stakeholder Notification" in plan_content
        assert "Data Breach Notification" in plan_content

    def test_has_post_incident_review(self, plan_content: str):
        assert "## 8. Post-Incident Review" in plan_content
        assert "Root Cause Analysis" in plan_content
        assert "Action Items" in plan_content
        assert "Timeline" in plan_content

    def test_has_related_documents(self, plan_content: str):
        assert "## 11. Related Documents" in plan_content
        assert "FAILOVER-RUNBOOK.md" in plan_content
        assert "data-retention-policy.md" in plan_content

    def test_has_revision_history(self, plan_content: str):
        assert "## Revision History" in plan_content
