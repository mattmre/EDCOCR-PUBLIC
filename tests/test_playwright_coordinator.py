"""Validate Playwright coordinator test files exist and are well-formed."""

from __future__ import annotations

import os


class TestPlaywrightCoordinatorExtension:
    """Validate Playwright coordinator test files exist and are well-formed."""

    def setup_method(self):
        root = os.path.dirname(os.path.dirname(__file__))
        self.api_spec = os.path.join(
            root, "playwright", "tests", "coordinator", "coordinator-api.spec.js"
        )
        self.ui_spec = os.path.join(
            root, "playwright", "tests", "coordinator", "coordinator-ui.spec.js"
        )

    def test_api_spec_exists(self):
        assert os.path.isfile(self.api_spec)

    def test_ui_spec_exists(self):
        assert os.path.isfile(self.ui_spec)

    def test_api_spec_has_test_describe(self):
        content = open(self.api_spec).read()
        assert "test.describe" in content

    def test_api_spec_is_env_gated(self):
        content = open(self.api_spec).read()
        assert "PLAYWRIGHT_COORDINATOR_BASE_URL" in content

    def test_ui_spec_has_browser_tests(self):
        content = open(self.ui_spec).read()
        assert "page.goto" in content

    def test_api_spec_tests_metrics(self):
        content = open(self.api_spec).read()
        assert "/api/v1/metrics/" in content

    def test_api_spec_tests_jobs(self):
        content = open(self.api_spec).read()
        assert "/api/v1/jobs/" in content

    def test_ui_spec_tests_admin(self):
        content = open(self.ui_spec).read()
        assert "/admin/" in content

    def test_api_spec_tests_auth(self):
        content = open(self.api_spec).read()
        assert "X-Api-Key" in content

    def test_ui_spec_tests_login(self):
        content = open(self.ui_spec).read()
        assert "login" in content.lower()

    def test_api_spec_has_prometheus(self):
        content = open(self.api_spec).read()
        assert "prometheus" in content.lower()

    def test_specs_use_env_skip_pattern(self):
        for spec_path in [self.api_spec, self.ui_spec]:
            content = open(spec_path).read()
            assert "test.skip" in content
