"""Docker image optimization analyzer for OCR pipeline.

Analyzes Dockerfiles and docker-compose configs to suggest
size reduction, layer optimization, and startup improvements.

Usage:
    python scripts/docker_optimize.py [--dockerfile Dockerfile] [--report]
"""

import argparse
import json
import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

logger = logging.getLogger(__name__)


class OptimizationCategory(Enum):
    LAYER_REDUCTION = "layer_reduction"
    CACHE_OPTIMIZATION = "cache_optimization"
    SIZE_REDUCTION = "size_reduction"
    STARTUP_TIME = "startup_time"
    SECURITY = "security"


class Severity(Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


@dataclass
class Suggestion:
    category: OptimizationCategory
    severity: Severity
    title: str
    description: str
    current: str = ""
    suggested: str = ""
    estimated_impact: str = ""


@dataclass
class DockerfileAnalysis:
    path: str
    total_layers: int = 0
    run_layers: int = 0
    copy_layers: int = 0
    stages: int = 0
    has_user_directive: bool = False
    has_healthcheck: bool = False
    base_image: str = ""
    suggestions: list = field(default_factory=list)

    def to_dict(self):
        return {
            "path": self.path,
            "total_layers": self.total_layers,
            "run_layers": self.run_layers,
            "copy_layers": self.copy_layers,
            "stages": self.stages,
            "has_user_directive": self.has_user_directive,
            "has_healthcheck": self.has_healthcheck,
            "base_image": self.base_image,
            "suggestion_count": len(self.suggestions),
            "suggestions": [
                {
                    "category": s.category.value,
                    "severity": s.severity.value,
                    "title": s.title,
                    "description": s.description,
                    "estimated_impact": s.estimated_impact,
                }
                for s in self.suggestions
            ],
        }


def parse_dockerfile(path: Path) -> DockerfileAnalysis:
    """Parse a Dockerfile and count layers/directives."""
    analysis = DockerfileAnalysis(path=str(path))

    if not path.exists():
        return analysis

    content = path.read_text(errors="replace")
    lines = content.splitlines()

    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue

        upper = stripped.split()[0].upper() if stripped.split() else ""

        if upper == "FROM":
            analysis.stages += 1
            if analysis.stages == 1:
                # Extract base image
                parts = stripped.split()
                if len(parts) >= 2:
                    analysis.base_image = parts[1]
            analysis.total_layers += 1
        elif upper == "RUN":
            analysis.run_layers += 1
            analysis.total_layers += 1
        elif upper in ("COPY", "ADD"):
            analysis.copy_layers += 1
            analysis.total_layers += 1
        elif upper == "USER":
            analysis.has_user_directive = True
        elif upper == "HEALTHCHECK":
            analysis.has_healthcheck = True
        elif upper in ("ENV", "WORKDIR", "EXPOSE", "ENTRYPOINT", "CMD", "ARG", "LABEL"):
            analysis.total_layers += 1

    return analysis


def analyze_layer_consolidation(analysis: DockerfileAnalysis, content: str) -> list:
    """Check for consecutive RUN commands that could be consolidated."""
    suggestions = []
    lines = content.splitlines()
    consecutive_runs = 0
    max_consecutive = 0

    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.upper().startswith("RUN "):
            consecutive_runs += 1
            max_consecutive = max(max_consecutive, consecutive_runs)
        else:
            consecutive_runs = 0

    if max_consecutive >= 3:
        suggestions.append(Suggestion(
            category=OptimizationCategory.LAYER_REDUCTION,
            severity=Severity.MEDIUM,
            title="Consolidate consecutive RUN commands",
            description=f"Found {max_consecutive} consecutive RUN commands. Combine with '&&' to reduce layers.",
            estimated_impact=f"Reduce ~{max_consecutive - 1} layers",
        ))

    return suggestions


def analyze_cache_optimization(content: str) -> list:
    """Check for cache-busting patterns."""
    suggestions = []

    # Check if requirements.txt is copied before full source
    lines = content.splitlines()
    copy_positions = []
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.upper().startswith(("COPY", "ADD")):
            copy_positions.append((i, stripped))

    has_requirements_copy = any("requirements" in c[1] for c in copy_positions)
    has_dot_copy = any(c[1].endswith(". .") or ". /" in c[1] for c in copy_positions)

    if has_dot_copy and not has_requirements_copy:
        suggestions.append(Suggestion(
            category=OptimizationCategory.CACHE_OPTIMIZATION,
            severity=Severity.HIGH,
            title="Copy requirements.txt before source code",
            description="COPY requirements.txt and pip install before COPY . to leverage Docker layer cache.",
            estimated_impact="Avoid reinstalling dependencies on every code change",
        ))

    return suggestions


def analyze_size_reduction(content: str) -> list:
    """Check for size optimization opportunities."""
    suggestions = []

    # Check for apt-get cleanup
    if "apt-get install" in content and "rm -rf /var/lib/apt/lists" not in content:
        suggestions.append(Suggestion(
            category=OptimizationCategory.SIZE_REDUCTION,
            severity=Severity.MEDIUM,
            title="Clean apt cache after install",
            description="Add 'rm -rf /var/lib/apt/lists/*' after apt-get install to reduce image size.",
            estimated_impact="~100-500MB reduction",
        ))

    # Check for pip cache
    if "pip install" in content and "--no-cache-dir" not in content:
        suggestions.append(Suggestion(
            category=OptimizationCategory.SIZE_REDUCTION,
            severity=Severity.LOW,
            title="Use --no-cache-dir with pip",
            description="Add --no-cache-dir to pip install to avoid caching packages in the image.",
            estimated_impact="~50-200MB reduction",
        ))

    # Check for .dockerignore mention
    suggestions.append(Suggestion(
        category=OptimizationCategory.SIZE_REDUCTION,
        severity=Severity.INFO,
        title="Ensure .dockerignore excludes dev files",
        description="Verify .dockerignore excludes __pycache__, .git, node_modules, tests/, docs/, *.md.",
        estimated_impact="Variable — depends on excluded files",
    ))

    return suggestions


def analyze_startup(content: str) -> list:
    """Check for startup time optimizations."""
    suggestions = []

    # Check for model preloading in build stage
    if "download_models" in content or "model" in content.lower():
        has_multi_stage = content.count("FROM ") > 1
        if has_multi_stage:
            suggestions.append(Suggestion(
                category=OptimizationCategory.STARTUP_TIME,
                severity=Severity.INFO,
                title="Models pre-baked in build stage",
                description="Multi-stage build detected with model preloading. Good for startup time.",
                estimated_impact="Avoids model download at startup",
            ))

    # Check for lazy imports recommendation
    if "import torch" in content or "import paddle" in content:
        suggestions.append(Suggestion(
            category=OptimizationCategory.STARTUP_TIME,
            severity=Severity.LOW,
            title="Consider lazy imports for ML frameworks",
            description="Import torch/paddle lazily in application code to reduce startup time.",
            estimated_impact="1-5 second startup reduction",
        ))

    return suggestions


def analyze_security_directives(analysis: DockerfileAnalysis) -> list:
    """Check for security-related optimizations."""
    suggestions = []

    if not analysis.has_user_directive:
        suggestions.append(Suggestion(
            category=OptimizationCategory.SECURITY,
            severity=Severity.HIGH,
            title="Add non-root USER directive",
            description="Container runs as root. Add 'RUN useradd -r -s /bin/false ocr' and 'USER ocr'.",
            estimated_impact="Reduces container escape blast radius",
        ))

    if not analysis.has_healthcheck and analysis.stages > 0:
        suggestions.append(Suggestion(
            category=OptimizationCategory.SECURITY,
            severity=Severity.MEDIUM,
            title="Add HEALTHCHECK directive",
            description="No HEALTHCHECK found. Add one for orchestrator health monitoring.",
            estimated_impact="Improved container lifecycle management",
        ))

    return suggestions


def analyze_dockerfile(path: Path) -> DockerfileAnalysis:
    """Full analysis of a Dockerfile."""
    analysis = parse_dockerfile(path)

    if not path.exists():
        return analysis

    content = path.read_text(errors="replace")

    analysis.suggestions.extend(analyze_layer_consolidation(analysis, content))
    analysis.suggestions.extend(analyze_cache_optimization(content))
    analysis.suggestions.extend(analyze_size_reduction(content))
    analysis.suggestions.extend(analyze_startup(content))
    analysis.suggestions.extend(analyze_security_directives(analysis))

    return analysis


def find_dockerfiles(root: Path) -> list:
    """Find all Dockerfiles in project."""
    files = []
    for pattern in ["Dockerfile", "Dockerfile.*"]:
        files.extend(root.glob(pattern))
    # Also check coordinator/
    coord = root / "coordinator"
    if coord.exists():
        for pattern in ["Dockerfile", "Dockerfile.*"]:
            files.extend(coord.glob(pattern))
    return sorted(set(files))


def generate_report(analyses: list) -> str:
    """Generate markdown optimization report."""
    lines = [
        "# Docker Image Optimization Report",
        "",
        "## Summary",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Dockerfiles analyzed | {len(analyses)} |",
        f"| Total suggestions | {sum(len(a.suggestions) for a in analyses)} |",
        f"| High severity | {sum(1 for a in analyses for s in a.suggestions if s.severity == Severity.HIGH)} |",
        f"| Medium severity | {sum(1 for a in analyses for s in a.suggestions if s.severity == Severity.MEDIUM)} |",
        "",
    ]

    for analysis in analyses:
        lines.append(f"## {analysis.path}")
        lines.append("")
        lines.append(f"- Base image: `{analysis.base_image}`")
        lines.append(f"- Stages: {analysis.stages}")
        lines.append(f"- Total layers: {analysis.total_layers}")
        lines.append(f"- RUN layers: {analysis.run_layers}")
        lines.append(f"- COPY/ADD layers: {analysis.copy_layers}")
        lines.append(f"- Has USER: {'Yes' if analysis.has_user_directive else 'No'}")
        lines.append(f"- Has HEALTHCHECK: {'Yes' if analysis.has_healthcheck else 'No'}")
        lines.append("")

        if analysis.suggestions:
            lines.append("### Suggestions")
            lines.append("")
            for s in analysis.suggestions:
                icon = {"high": "🔴", "medium": "🟡", "low": "🔵", "info": "ℹ️"}.get(s.severity.value, "")
                lines.append(f"- {icon} **{s.title}** ({s.severity.value})")
                lines.append(f"  {s.description}")
                if s.estimated_impact:
                    lines.append(f"  *Impact: {s.estimated_impact}*")
                lines.append("")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Docker image optimization analyzer")
    parser.add_argument("--dockerfile", type=str, help="Specific Dockerfile to analyze")
    parser.add_argument("--report", action="store_true", help="Generate markdown report")
    parser.add_argument("--json", action="store_true", help="Output JSON")
    parser.add_argument("--root", type=str, help="Project root")
    args = parser.parse_args()

    root = Path(args.root) if args.root else Path(__file__).resolve().parent.parent

    if args.dockerfile:
        dockerfiles = [Path(args.dockerfile)]
    else:
        dockerfiles = find_dockerfiles(root)

    analyses = [analyze_dockerfile(df) for df in dockerfiles]

    if args.json:
        print(json.dumps([a.to_dict() for a in analyses], indent=2))
    else:
        print(generate_report(analyses))


if __name__ == "__main__":
    main()
