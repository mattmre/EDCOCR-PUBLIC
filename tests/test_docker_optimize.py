"""Tests for scripts/docker_optimize.py.

Covers Suggestion/DockerfileAnalysis dataclasses, OptimizationCategory and
Severity enums, each analysis function, find_dockerfiles, generate_report,
and the full analyze_dockerfile pipeline.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from docker_optimize import (
    DockerfileAnalysis,
    OptimizationCategory,
    Severity,
    Suggestion,
    analyze_cache_optimization,
    analyze_dockerfile,
    analyze_layer_consolidation,
    analyze_security_directives,
    analyze_size_reduction,
    analyze_startup,
    find_dockerfiles,
    generate_report,
    parse_dockerfile,
)

# ---------------------------------------------------------------------------
# OptimizationCategory enum
# ---------------------------------------------------------------------------


class TestOptimizationCategory:
    """Tests for the OptimizationCategory enum."""

    def test_values(self):
        assert OptimizationCategory.LAYER_REDUCTION.value == "layer_reduction"
        assert OptimizationCategory.CACHE_OPTIMIZATION.value == "cache_optimization"
        assert OptimizationCategory.SIZE_REDUCTION.value == "size_reduction"
        assert OptimizationCategory.STARTUP_TIME.value == "startup_time"
        assert OptimizationCategory.SECURITY.value == "security"

    def test_member_count(self):
        assert len(OptimizationCategory) == 5


# ---------------------------------------------------------------------------
# Severity enum
# ---------------------------------------------------------------------------


class TestSeverity:
    """Tests for the Severity enum."""

    def test_values(self):
        assert Severity.HIGH.value == "high"
        assert Severity.MEDIUM.value == "medium"
        assert Severity.LOW.value == "low"
        assert Severity.INFO.value == "info"

    def test_member_count(self):
        assert len(Severity) == 4


# ---------------------------------------------------------------------------
# Suggestion dataclass
# ---------------------------------------------------------------------------


class TestSuggestion:
    """Tests for the Suggestion dataclass."""

    def test_creation(self):
        s = Suggestion(
            category=OptimizationCategory.SIZE_REDUCTION,
            severity=Severity.MEDIUM,
            title="Test suggestion",
            description="A test description",
        )
        assert s.category == OptimizationCategory.SIZE_REDUCTION
        assert s.severity == Severity.MEDIUM
        assert s.title == "Test suggestion"
        assert s.description == "A test description"
        assert s.current == ""
        assert s.suggested == ""
        assert s.estimated_impact == ""

    def test_with_optional_fields(self):
        s = Suggestion(
            category=OptimizationCategory.LAYER_REDUCTION,
            severity=Severity.HIGH,
            title="Consolidate layers",
            description="Combine RUN commands",
            current="RUN cmd1\nRUN cmd2",
            suggested="RUN cmd1 && cmd2",
            estimated_impact="~2 layers saved",
        )
        assert s.current == "RUN cmd1\nRUN cmd2"
        assert s.suggested == "RUN cmd1 && cmd2"
        assert s.estimated_impact == "~2 layers saved"


# ---------------------------------------------------------------------------
# DockerfileAnalysis dataclass
# ---------------------------------------------------------------------------


class TestDockerfileAnalysis:
    """Tests for the DockerfileAnalysis dataclass."""

    def test_defaults(self):
        a = DockerfileAnalysis(path="Dockerfile")
        assert a.path == "Dockerfile"
        assert a.total_layers == 0
        assert a.run_layers == 0
        assert a.copy_layers == 0
        assert a.stages == 0
        assert a.has_user_directive is False
        assert a.has_healthcheck is False
        assert a.base_image == ""
        assert a.suggestions == []

    def test_to_dict_empty(self):
        a = DockerfileAnalysis(path="Dockerfile")
        d = a.to_dict()
        assert d["path"] == "Dockerfile"
        assert d["suggestion_count"] == 0
        assert d["suggestions"] == []

    def test_to_dict_with_suggestions(self):
        a = DockerfileAnalysis(path="Dockerfile", stages=1, total_layers=5)
        a.suggestions.append(Suggestion(
            category=OptimizationCategory.SECURITY,
            severity=Severity.HIGH,
            title="Add USER",
            description="Run as non-root",
            estimated_impact="Security improvement",
        ))
        d = a.to_dict()
        assert d["suggestion_count"] == 1
        assert d["stages"] == 1
        assert d["total_layers"] == 5
        s = d["suggestions"][0]
        assert s["category"] == "security"
        assert s["severity"] == "high"
        assert s["title"] == "Add USER"
        assert s["estimated_impact"] == "Security improvement"

    def test_to_dict_json_serializable(self):
        a = DockerfileAnalysis(path="Dockerfile")
        a.suggestions.append(Suggestion(
            category=OptimizationCategory.SIZE_REDUCTION,
            severity=Severity.INFO,
            title="Check .dockerignore",
            description="Exclude dev files",
        ))
        output = json.dumps(a.to_dict())
        assert "Check .dockerignore" in output


# ---------------------------------------------------------------------------
# parse_dockerfile
# ---------------------------------------------------------------------------


class TestParseDockerfile:
    """Tests for parse_dockerfile."""

    def test_minimal_dockerfile(self, tmp_path):
        df = tmp_path / "Dockerfile"
        df.write_text("FROM python:3.10-slim\nRUN pip install flask\nCOPY . /app\n")
        a = parse_dockerfile(df)
        assert a.stages == 1
        assert a.base_image == "python:3.10-slim"
        assert a.run_layers == 1
        assert a.copy_layers == 1
        # FROM (1) + RUN (1) + COPY (1) = 3
        assert a.total_layers == 3

    def test_with_user_directive(self, tmp_path):
        df = tmp_path / "Dockerfile"
        df.write_text("FROM python:3.10\nRUN pip install app\nUSER appuser\nCMD [\"python\"]\n")
        a = parse_dockerfile(df)
        assert a.has_user_directive is True
        # CMD counts as a layer
        assert a.total_layers == 3  # FROM + RUN + CMD

    def test_with_healthcheck(self, tmp_path):
        df = tmp_path / "Dockerfile"
        df.write_text(
            "FROM python:3.10\n"
            "RUN pip install app\n"
            "HEALTHCHECK --interval=30s CMD curl -f http://localhost/\n"
            "CMD [\"python\", \"app.py\"]\n"
        )
        a = parse_dockerfile(df)
        assert a.has_healthcheck is True

    def test_multi_stage(self, tmp_path):
        df = tmp_path / "Dockerfile"
        df.write_text(
            "FROM python:3.10 AS builder\n"
            "RUN pip install build\n"
            "COPY . /src\n"
            "FROM python:3.10-slim\n"
            "COPY --from=builder /src/dist /app\n"
            "CMD [\"python\", \"app.py\"]\n"
        )
        a = parse_dockerfile(df)
        assert a.stages == 2
        assert a.base_image == "python:3.10"  # first FROM
        assert a.run_layers == 1
        assert a.copy_layers == 2

    def test_nonexistent_file(self, tmp_path):
        df = tmp_path / "NoSuchDockerfile"
        a = parse_dockerfile(df)
        assert a.stages == 0
        assert a.total_layers == 0
        assert a.base_image == ""

    def test_comments_and_blanks_ignored(self, tmp_path):
        df = tmp_path / "Dockerfile"
        df.write_text(
            "# This is a comment\n"
            "\n"
            "FROM ubuntu:22.04\n"
            "# Another comment\n"
            "\n"
            "RUN echo hello\n"
        )
        a = parse_dockerfile(df)
        assert a.stages == 1
        assert a.run_layers == 1
        assert a.total_layers == 2  # FROM + RUN

    def test_add_counts_as_copy(self, tmp_path):
        df = tmp_path / "Dockerfile"
        df.write_text("FROM alpine\nADD https://example.com/file.tar.gz /app/\n")
        a = parse_dockerfile(df)
        assert a.copy_layers == 1

    def test_env_workdir_expose_counted(self, tmp_path):
        df = tmp_path / "Dockerfile"
        df.write_text(
            "FROM python:3.10\n"
            "ENV APP_HOME=/app\n"
            "WORKDIR /app\n"
            "EXPOSE 8080\n"
            "ARG VERSION=1.0\n"
            "LABEL maintainer=test\n"
        )
        a = parse_dockerfile(df)
        # FROM(1) + ENV(1) + WORKDIR(1) + EXPOSE(1) + ARG(1) + LABEL(1) = 6
        assert a.total_layers == 6


# ---------------------------------------------------------------------------
# analyze_layer_consolidation
# ---------------------------------------------------------------------------


class TestAnalyzeLayerConsolidation:
    """Tests for analyze_layer_consolidation."""

    def test_consecutive_runs_detected(self):
        content = (
            "FROM python:3.10\n"
            "RUN apt-get update\n"
            "RUN apt-get install -y curl\n"
            "RUN apt-get install -y git\n"
        )
        analysis = DockerfileAnalysis(path="test")
        suggestions = analyze_layer_consolidation(analysis, content)
        assert len(suggestions) == 1
        assert suggestions[0].category == OptimizationCategory.LAYER_REDUCTION
        assert "3" in suggestions[0].description

    def test_no_consecutive_runs(self):
        content = (
            "FROM python:3.10\n"
            "RUN apt-get update && apt-get install -y curl\n"
            "COPY . /app\n"
            "RUN pip install -r requirements.txt\n"
        )
        analysis = DockerfileAnalysis(path="test")
        suggestions = analyze_layer_consolidation(analysis, content)
        assert len(suggestions) == 0

    def test_exactly_two_consecutive_no_suggestion(self):
        content = (
            "FROM python:3.10\n"
            "RUN apt-get update\n"
            "RUN apt-get install -y curl\n"
            "COPY . /app\n"
        )
        analysis = DockerfileAnalysis(path="test")
        suggestions = analyze_layer_consolidation(analysis, content)
        assert len(suggestions) == 0  # threshold is >= 3


# ---------------------------------------------------------------------------
# analyze_cache_optimization
# ---------------------------------------------------------------------------


class TestAnalyzeCacheOptimization:
    """Tests for analyze_cache_optimization."""

    def test_requirements_copy_before_source(self):
        content = (
            "FROM python:3.10\n"
            "COPY requirements.txt /app/\n"
            "RUN pip install -r /app/requirements.txt\n"
            "COPY . /app\n"
        )
        suggestions = analyze_cache_optimization(content)
        # Has requirements copy, so no suggestion even though COPY . exists
        assert len(suggestions) == 0

    def test_no_requirements_copy_with_dot_copy(self):
        content = (
            "FROM python:3.10\n"
            "COPY . .\n"
            "RUN pip install -r requirements.txt\n"
        )
        suggestions = analyze_cache_optimization(content)
        assert len(suggestions) == 1
        assert suggestions[0].category == OptimizationCategory.CACHE_OPTIMIZATION
        assert suggestions[0].severity == Severity.HIGH

    def test_no_dot_copy_no_suggestion(self):
        content = (
            "FROM python:3.10\n"
            "COPY app.py /app/\n"
            "RUN pip install flask\n"
        )
        suggestions = analyze_cache_optimization(content)
        assert len(suggestions) == 0


# ---------------------------------------------------------------------------
# analyze_size_reduction
# ---------------------------------------------------------------------------


class TestAnalyzeSizeReduction:
    """Tests for analyze_size_reduction."""

    def test_apt_without_cleanup(self):
        content = (
            "FROM ubuntu:22.04\n"
            "RUN apt-get update && apt-get install -y curl\n"
        )
        suggestions = analyze_size_reduction(content)
        apt_suggestions = [s for s in suggestions if "apt cache" in s.title.lower()]
        assert len(apt_suggestions) == 1
        assert apt_suggestions[0].severity == Severity.MEDIUM

    def test_apt_with_cleanup_no_apt_suggestion(self):
        content = (
            "FROM ubuntu:22.04\n"
            "RUN apt-get update && apt-get install -y curl "
            "&& rm -rf /var/lib/apt/lists/*\n"
        )
        suggestions = analyze_size_reduction(content)
        apt_suggestions = [s for s in suggestions if "apt cache" in s.title.lower()]
        assert len(apt_suggestions) == 0

    def test_pip_without_no_cache_dir(self):
        content = (
            "FROM python:3.10\n"
            "RUN pip install flask\n"
        )
        suggestions = analyze_size_reduction(content)
        pip_suggestions = [s for s in suggestions if "no-cache-dir" in s.title.lower()]
        assert len(pip_suggestions) == 1
        assert pip_suggestions[0].severity == Severity.LOW

    def test_pip_with_no_cache_dir(self):
        content = (
            "FROM python:3.10\n"
            "RUN pip install --no-cache-dir flask\n"
        )
        suggestions = analyze_size_reduction(content)
        pip_suggestions = [s for s in suggestions if "no-cache-dir" in s.title.lower()]
        assert len(pip_suggestions) == 0

    def test_always_suggests_dockerignore(self):
        content = "FROM python:3.10\n"
        suggestions = analyze_size_reduction(content)
        ignore_suggestions = [s for s in suggestions if "dockerignore" in s.title.lower()]
        assert len(ignore_suggestions) == 1
        assert ignore_suggestions[0].severity == Severity.INFO


# ---------------------------------------------------------------------------
# analyze_startup
# ---------------------------------------------------------------------------


class TestAnalyzeStartup:
    """Tests for analyze_startup."""

    def test_model_preloading_multi_stage(self):
        content = (
            "FROM python:3.10 AS builder\n"
            "RUN python download_models.py\n"
            "FROM python:3.10-slim\n"
            "COPY --from=builder /models /models\n"
        )
        suggestions = analyze_startup(content)
        model_suggestions = [s for s in suggestions if "pre-baked" in s.title.lower()]
        assert len(model_suggestions) == 1
        assert model_suggestions[0].severity == Severity.INFO

    def test_no_model_reference(self):
        content = (
            "FROM python:3.10\n"
            "COPY app.py /app/\n"
            "CMD [\"python\", \"app.py\"]\n"
        )
        suggestions = analyze_startup(content)
        assert len(suggestions) == 0

    def test_lazy_import_suggestion(self):
        content = (
            "FROM python:3.10\n"
            "RUN pip install torch\n"
            "COPY app.py /app/\n"
        )
        suggestions = analyze_startup(content)
        lazy_suggestions = [s for s in suggestions if "lazy" in s.title.lower()]
        assert len(lazy_suggestions) == 0  # 'import torch' not in content directly

    def test_lazy_import_triggered(self):
        content = (
            "FROM python:3.10\n"
            "RUN echo 'import paddle'\n"
        )
        suggestions = analyze_startup(content)
        lazy_suggestions = [s for s in suggestions if "lazy" in s.title.lower()]
        assert len(lazy_suggestions) == 1


# ---------------------------------------------------------------------------
# analyze_security_directives
# ---------------------------------------------------------------------------


class TestAnalyzeSecurityDirectives:
    """Tests for analyze_security_directives."""

    def test_no_user_directive(self):
        a = DockerfileAnalysis(path="Dockerfile", stages=1, has_user_directive=False, has_healthcheck=True)
        suggestions = analyze_security_directives(a)
        user_suggestions = [s for s in suggestions if "USER" in s.title]
        assert len(user_suggestions) == 1
        assert user_suggestions[0].severity == Severity.HIGH

    def test_with_user_directive(self):
        a = DockerfileAnalysis(path="Dockerfile", stages=1, has_user_directive=True, has_healthcheck=True)
        suggestions = analyze_security_directives(a)
        user_suggestions = [s for s in suggestions if "USER" in s.title]
        assert len(user_suggestions) == 0

    def test_no_healthcheck(self):
        a = DockerfileAnalysis(path="Dockerfile", stages=1, has_user_directive=True, has_healthcheck=False)
        suggestions = analyze_security_directives(a)
        health_suggestions = [s for s in suggestions if "HEALTHCHECK" in s.title]
        assert len(health_suggestions) == 1
        assert health_suggestions[0].severity == Severity.MEDIUM

    def test_no_healthcheck_zero_stages(self):
        """No stages means no Dockerfile was really parsed — skip HEALTHCHECK check."""
        a = DockerfileAnalysis(path="Dockerfile", stages=0, has_user_directive=False, has_healthcheck=False)
        suggestions = analyze_security_directives(a)
        health_suggestions = [s for s in suggestions if "HEALTHCHECK" in s.title]
        assert len(health_suggestions) == 0


# ---------------------------------------------------------------------------
# analyze_dockerfile (full pipeline)
# ---------------------------------------------------------------------------


class TestAnalyzeDockerfile:
    """Tests for the full analyze_dockerfile pipeline."""

    def test_full_pipeline(self, tmp_path):
        df = tmp_path / "Dockerfile"
        df.write_text(
            "FROM python:3.10\n"
            "RUN apt-get update\n"
            "RUN apt-get install -y curl\n"
            "RUN apt-get install -y git\n"
            "COPY . .\n"
            "RUN pip install flask\n"
            "CMD [\"python\", \"app.py\"]\n"
        )
        a = analyze_dockerfile(df)
        assert a.stages == 1
        assert a.base_image == "python:3.10"
        assert len(a.suggestions) > 0
        categories = {s.category for s in a.suggestions}
        # Should detect layer consolidation, cache, size, and security issues
        assert OptimizationCategory.LAYER_REDUCTION in categories
        assert OptimizationCategory.SECURITY in categories

    def test_nonexistent_file_returns_empty(self, tmp_path):
        df = tmp_path / "NoDockerfile"
        a = analyze_dockerfile(df)
        assert a.stages == 0
        assert len(a.suggestions) == 0


# ---------------------------------------------------------------------------
# find_dockerfiles
# ---------------------------------------------------------------------------


class TestFindDockerfiles:
    """Tests for find_dockerfiles."""

    def test_finds_dockerfiles(self, tmp_path):
        (tmp_path / "Dockerfile").write_text("FROM python:3.10\n")
        (tmp_path / "Dockerfile.gpu").write_text("FROM nvidia/cuda:12.0\n")
        (tmp_path / "not-a-dockerfile.txt").write_text("FROM nothing\n")
        found = find_dockerfiles(tmp_path)
        names = [f.name for f in found]
        assert "Dockerfile" in names
        assert "Dockerfile.gpu" in names
        assert "not-a-dockerfile.txt" not in names

    def test_finds_coordinator_dockerfiles(self, tmp_path):
        coord = tmp_path / "coordinator"
        coord.mkdir()
        (coord / "Dockerfile").write_text("FROM node:18\n")
        (tmp_path / "Dockerfile").write_text("FROM python:3.10\n")
        found = find_dockerfiles(tmp_path)
        assert len(found) == 2

    def test_empty_directory(self, tmp_path):
        found = find_dockerfiles(tmp_path)
        assert found == []

    def test_deduplicates(self, tmp_path):
        (tmp_path / "Dockerfile").write_text("FROM python:3.10\n")
        found = find_dockerfiles(tmp_path)
        assert len(found) == 1


# ---------------------------------------------------------------------------
# generate_report
# ---------------------------------------------------------------------------


class TestGenerateReport:
    """Tests for generate_report."""

    def test_report_structure(self):
        a = DockerfileAnalysis(
            path="Dockerfile",
            stages=2,
            total_layers=10,
            run_layers=4,
            copy_layers=3,
            base_image="python:3.10",
            has_user_directive=True,
            has_healthcheck=True,
        )
        report = generate_report([a])
        assert "# Docker Image Optimization Report" in report
        assert "## Summary" in report
        assert "Dockerfiles analyzed | 1" in report
        assert "`python:3.10`" in report
        assert "Stages: 2" in report
        assert "Has USER: Yes" in report
        assert "Has HEALTHCHECK: Yes" in report

    def test_report_with_suggestions(self):
        a = DockerfileAnalysis(path="Dockerfile", stages=1)
        a.suggestions.append(Suggestion(
            category=OptimizationCategory.SECURITY,
            severity=Severity.HIGH,
            title="Add USER directive",
            description="Run as non-root",
            estimated_impact="Security improvement",
        ))
        report = generate_report([a])
        assert "### Suggestions" in report
        assert "Add USER directive" in report
        assert "🔴" in report
        assert "Security improvement" in report

    def test_empty_analyses(self):
        report = generate_report([])
        assert "Dockerfiles analyzed | 0" in report
        assert "Total suggestions | 0" in report

    def test_report_multiple_analyses(self):
        a1 = DockerfileAnalysis(path="Dockerfile", stages=1, base_image="python:3.10")
        a2 = DockerfileAnalysis(path="Dockerfile.gpu", stages=2, base_image="nvidia/cuda:12.0")
        report = generate_report([a1, a2])
        assert "Dockerfiles analyzed | 2" in report
        assert "## Dockerfile" in report
        assert "## Dockerfile.gpu" in report
