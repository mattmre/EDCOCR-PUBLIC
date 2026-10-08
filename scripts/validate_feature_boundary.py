#!/usr/bin/env python3
"""Validate the forensic-core versus AI-adjacent feature boundary.

Checks:
  A) Job submit defaults keep AI-adjacent request controls opt-in
  B) Core safeguards stay enabled by default in config/reference docs
  C) AI-adjacent runtime toggles stay default-off
  D) Distributed queue routing isolates AI-adjacent work from OCR queues
  E) Boundary contract doc exists with the required sections
  F) API docs expose the boundary to callers
  G) Language detection never mutates EXPORT/PDF/ or EXPORT/TEXT/ (E-X-002 i)
  H) Translation engines never call output_assembler.build_retrieval_output
     without a certified=true filter (E-X-002 ii)
  I) Installed entry-point plugins in the ocr_local.enrichments group
     carry verified signatures at CI time (E-X-002 iii)
"""

from __future__ import annotations

import argparse
import ast
import io
import json
import re
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_PROJECT_ROOT = SCRIPT_DIR.parent


@dataclass
class CheckResult:
    name: str
    passed: bool
    message: str
    details: list[str] = field(default_factory=list)


@dataclass
class ValidationReport:
    passed: bool = True
    checks: list[CheckResult] = field(default_factory=list)
    timestamp: str = ""
    project_root: str = ""

    def add(self, result: CheckResult) -> None:
        self.checks.append(result)
        if not result.passed:
            self.passed = False


def _safe_print(*args: object, **kwargs: object) -> None:
    try:
        print(*args, **kwargs)
    except UnicodeEncodeError:
        text = " ".join(str(a) for a in args)
        text = (
            text.replace("\u2713", "OK")
            .replace("\u2717", "FAIL")
            .replace("\u2192", "->")
        )
        end = kwargs.get("end", "\n")
        stream = kwargs.get("file", sys.stdout)
        if isinstance(stream, io.TextIOWrapper) and hasattr(stream, "buffer"):
            stream.buffer.write((text + str(end)).encode("utf-8", errors="replace"))
        else:
            print(text, end=end)


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def check_job_submit_defaults(root: Path) -> CheckResult:
    models_path = root / "api" / "models.py"
    deps_path = root / "api" / "deps.py"
    if not models_path.exists():
        return CheckResult("job_submit_defaults", False, "api/models.py not found")
    if not deps_path.exists():
        return CheckResult("job_submit_defaults", False, "api/deps.py not found")

    models_text = _read(models_path)
    deps_text = _read(deps_path)
    issues: list[str] = []

    expected_patterns = {
        "enable_docintel default false": r"enable_docintel:\s*bool\s*=\s*False",
        "skip_ocr default false": r"skip_ocr:\s*bool\s*=\s*Field\(\s*False,",
    }
    for label, pattern in expected_patterns.items():
        if not re.search(pattern, models_text, re.MULTILINE | re.DOTALL):
            issues.append(f"api/models.py missing: {label}")

    if "Skip OCR processing and only perform NLP/DocIntel if enabled" not in deps_text:
        issues.append(
            "api/deps.py no longer describes skip_ocr as an NLP/DocIntel-only path"
        )

    if issues:
        return CheckResult(
            name="job_submit_defaults",
            passed=False,
            message=f"{len(issues)} job-submit boundary issue(s)",
            details=issues,
        )

    return CheckResult(
        name="job_submit_defaults",
        passed=True,
        message="Job submit defaults keep AI-adjacent request controls opt-in",
    )


def check_core_guardrails(root: Path) -> CheckResult:
    config_ref = root / "docs" / "06-CONFIGURATION-REFERENCE.md"
    transforms = root / "docs" / "07-TRANSFORMS-STAMPING.md"
    if not config_ref.exists():
        return CheckResult(
            "core_guardrails", False, "docs/06-CONFIGURATION-REFERENCE.md not found"
        )
    if not transforms.exists():
        return CheckResult(
            "core_guardrails", False, "docs/07-TRANSFORMS-STAMPING.md not found"
        )

    config_text = _read(config_ref)
    transforms_text = _read(transforms)
    issues: list[str] = []

    required_snippets = [
        "`ENABLE_VALIDATION` | `true`",
        "`API_AUDIT_LOG_ENABLED` | `true`",
        "Forensic-Core Defaults",
        "AI-Adjacent Controls",
        "forensic-ai-boundary-contract.md",
        "Distributed queue isolation keeps advanced AI-adjacent work off the default OCR path",
    ]
    for snippet in required_snippets:
        if snippet not in config_text:
            issues.append(
                f"docs/06-CONFIGURATION-REFERENCE.md missing snippet: {snippet}"
            )

    if "## Scope Boundary" not in transforms_text:
        issues.append("docs/07-TRANSFORMS-STAMPING.md missing Scope Boundary section")
    if "forensic safeguards" not in transforms_text.lower():
        issues.append(
            "docs/07-TRANSFORMS-STAMPING.md no longer references forensic safeguards"
        )

    if issues:
        return CheckResult(
            name="core_guardrails",
            passed=False,
            message=f"{len(issues)} core safeguard issue(s)",
            details=issues,
        )

    return CheckResult(
        name="core_guardrails",
        passed=True,
        message="Core safeguards remain visible and enabled by default",
    )


def check_ai_feature_defaults(root: Path) -> CheckResult:
    config_ref = root / "docs" / "06-CONFIGURATION-REFERENCE.md"
    layoutlm_cfg = root / "coordinator" / "jobs" / "layoutlm_config.py"
    if not config_ref.exists():
        return CheckResult(
            "ai_feature_defaults", False, "docs/06-CONFIGURATION-REFERENCE.md not found"
        )
    if not layoutlm_cfg.exists():
        return CheckResult(
            "ai_feature_defaults",
            False,
            "coordinator/jobs/layoutlm_config.py not found",
        )

    config_text = _read(config_ref)
    layoutlm_text = _read(layoutlm_cfg)
    issues: list[str] = []

    default_off_snippets = [
        "`ENABLE_NER` | `false`",
        "`ENABLE_CLASSIFICATION` | `false`",
        "`ENABLE_EXTRACTION` | `false`",
        "`ENABLE_SPECIALIST_ROUTING` | `false`",
    ]
    for snippet in default_off_snippets:
        if snippet not in config_text:
            issues.append(
                f"docs/06-CONFIGURATION-REFERENCE.md missing default-off snippet: {snippet}"
            )

    if '"ENABLE_LAYOUTLM", "false"' not in layoutlm_text:
        issues.append(
            "coordinator/jobs/layoutlm_config.py no longer defaults ENABLE_LAYOUTLM to false"
        )

    if issues:
        return CheckResult(
            name="ai_feature_defaults",
            passed=False,
            message=f"{len(issues)} AI-adjacent default issue(s)",
            details=issues,
        )

    return CheckResult(
        name="ai_feature_defaults",
        passed=True,
        message="AI-adjacent runtime toggles remain default-off",
    )


def check_queue_isolation(root: Path) -> CheckResult:
    celery_path = root / "coordinator" / "coordinator" / "celery.py"
    if not celery_path.exists():
        return CheckResult(
            "queue_isolation", False, "coordinator/coordinator/celery.py not found"
        )

    celery_text = _read(celery_path)
    issues: list[str] = []

    # Parse the source instead of matching one exact quote/whitespace style.
    # The production module deliberately uses a callable task_routes entry and
    # declares queues across several conditional branches, so line-oriented
    # string checks produced a false negative and were also comment-spoofable.
    try:
        tree = ast.parse(celery_text, filename=str(celery_path))
    except SyntaxError as exc:
        return CheckResult(
            name="queue_isolation",
            passed=False,
            message="Could not parse coordinator/coordinator/celery.py",
            details=[f"SyntaxError: {exc}"],
        )

    route_pairs: set[tuple[str, str]] = set()
    declared_queues: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if not any(
                isinstance(target, ast.Name) and target.id == "_STATIC_ROUTES"
                for target in targets
            ):
                continue
            value = node.value
            if not isinstance(value, ast.Dict):
                continue
            for key, route_value in zip(value.keys, value.values):
                if (
                    isinstance(key, ast.Constant)
                    and isinstance(key.value, str)
                    and isinstance(route_value, ast.Constant)
                    and isinstance(route_value.value, str)
                ):
                    route_pairs.add((key.value, route_value.value))

        if not isinstance(node, ast.Call) or not node.args:
            continue
        function = node.func
        is_queue_call = isinstance(function, ast.Name) and function.id == "Queue"
        is_queue_call = is_queue_call or (
            isinstance(function, ast.Attribute) and function.attr == "Queue"
        )
        if is_queue_call:
            queue_name = node.args[0]
            if isinstance(queue_name, ast.Constant) and isinstance(
                queue_name.value, str
            ):
                declared_queues.add(queue_name.value)

    expected_routes = {
        ("jobs.tasks.extract_structured_data", "nlp_general"),
        ("jobs.tasks.process_text_only", "cpu_general"),
        ("jobs.tasks_layoutlm.run_layoutlm_extraction", "ocr_layoutlm"),
    }
    for task_name, queue_name in sorted(expected_routes):
        if (task_name, queue_name) not in route_pairs:
            issues.append(f"Missing route: {task_name!r}: {queue_name!r}")

    for queue_name in ("ocr_gpu", "cpu_general", "nlp_general", "ocr_layoutlm"):
        if queue_name not in declared_queues:
            issues.append(f"Missing queue declaration: Queue({queue_name!r})")

    if issues:
        return CheckResult(
            name="queue_isolation",
            passed=False,
            message=f"{len(issues)} queue isolation issue(s)",
            details=issues,
        )

    return CheckResult(
        name="queue_isolation",
        passed=True,
        message="Distributed routing keeps AI-adjacent work isolated from OCR queues",
    )


def check_boundary_contract_doc(root: Path) -> CheckResult:
    contract = root / "docs" / "architecture" / "forensic-ai-boundary-contract.md"
    legacy_contract = root / "docs" / "architecture" / "forensic-ai-boundary.md"
    if not contract.exists():
        return CheckResult(
            "boundary_contract_doc",
            False,
            "docs/architecture/forensic-ai-boundary-contract.md not found",
        )

    text = _read(contract)
    issues: list[str] = []
    required_sections = [
        "## 2. Forensic-Core Scope",
        "## 3. AI-Adjacent Scope",
        "## 4. Current Enforcement Points",
        "## 6. Change Checklist",
        "Current Capability Map",
        "Contract Rules",
    ]
    for section in required_sections:
        if section not in text:
            issues.append(f"Missing section: {section}")

    if "scripts/validate_feature_boundary.py" not in text:
        issues.append("Contract doc does not reference the validator")
    if legacy_contract.exists():
        issues.append("Legacy docs/architecture/forensic-ai-boundary.md still exists")

    if issues:
        return CheckResult(
            name="boundary_contract_doc",
            passed=False,
            message=f"{len(issues)} contract-doc issue(s)",
            details=issues,
        )

    return CheckResult(
        name="boundary_contract_doc",
        passed=True,
        message="Boundary contract doc exists with the required sections",
    )


def check_api_reference_boundary(root: Path) -> CheckResult:
    api_ref = root / "docs" / "API-REFERENCE.md"
    if not api_ref.exists():
        return CheckResult(
            "api_reference_boundary", False, "docs/API-REFERENCE.md not found"
        )

    text = _read(api_ref)
    issues: list[str] = []
    required_snippets = [
        "`enable_docintel` and `skip_ocr` are AI-adjacent controls.",
        "default forensic-core OCR path",
        "These controls add optional semantic or native-text behavior",
        "As with single-job submission, DocIntel settings are optional AI-adjacent enrichment controls rather than part of the minimum forensic-core processing guarantee.",
        "These endpoints are AI-adjacent analyst-assist features.",
        "[Forensic-Core vs AI-Adjacent Boundary](architecture/forensic-ai-boundary-contract.md)",
    ]
    for snippet in required_snippets:
        if snippet not in text:
            issues.append(f"docs/API-REFERENCE.md missing snippet: {snippet}")

    contract_link = "architecture/forensic-ai-boundary-contract.md"
    if text.count(contract_link) < 4:
        issues.append(
            "docs/API-REFERENCE.md should reference the canonical boundary contract in the request, batch, semantic-search, and related-docs sections"
        )

    if issues:
        return CheckResult(
            name="api_reference_boundary",
            passed=False,
            message=f"{len(issues)} API reference issue(s)",
            details=issues,
        )

    return CheckResult(
        name="api_reference_boundary",
        passed=True,
        message="API reference exposes the forensic-core versus AI-adjacent boundary",
    )


# ---------------------------------------------------------------------------
# E-X-002 assertions (translation-swarm program, 2026-04-24)
#
# These assertions are static/structural scans only. They must pass even when
# the Plan A (language detection) and Plan B (translation pipeline) source
# files do not yet exist -- those plans land in a later wave and the boundary
# validator is E-X-002's Gate-0 prerequisite that must be green before any
# Plan A/B/D PR opens.
# ---------------------------------------------------------------------------


# Candidate paths (relative to project root) where Plan A language-detection
# logic is expected to live once implemented. Each may or may not exist on
# any given commit -- absence is a valid state and the assertion passes.
_LANGUAGE_DETECTION_CANDIDATE_PATHS = (
    "ocr_local/features/language_detection.py",
    "ocr_local/features/per_span_language.py",
    "ocr_local/features/language_span.py",
    "ocr_local/features/lang_detect.py",
    "language_detection.py",
    "per_span_language.py",
)

# Text markers that identify a file as language-detection-related.
_LANGUAGE_DETECTION_MARKERS = (
    "LANGUAGE_DETECTED",
    "per_span_language",
    "LanguageSpan",
    "language_detection",
)

# Patterns that indicate a file writes to EXPORT/PDF or EXPORT/TEXT.
# Language detection must never mutate these forensic-core outputs.
_EXPORT_WRITE_PATTERNS = (
    r"EXPORT[\\/]+PDF",
    r"EXPORT[\\/]+TEXT",
    r"EXPORT/PDF",
    r"EXPORT/TEXT",
)

# Candidate paths where Plan B translation engines are expected to live.
_TRANSLATION_ENGINE_CANDIDATE_PATHS = (
    "ocr_local/features/translation.py",
    "ocr_local/features/translation_engine.py",
    "ocr_local/features/translation_pipeline.py",
    "ocr_local/translation/",
    "translation.py",
    "translation_engine.py",
)

# Text markers that identify a file as translation-engine-related.
_TRANSLATION_ENGINE_MARKERS = (
    "class TranslationEngine",
    "TRANSLATION_APPLIED",
    "translation_engine",
    "translate_text",
    "translation_pipeline",
)


def _iter_candidate_files(root: Path, candidates: tuple[str, ...]) -> list[Path]:
    """Return the subset of candidate paths that actually exist.

    Directory candidates are expanded to their *.py contents.
    """
    found: list[Path] = []
    for rel in candidates:
        target = root / rel
        if not target.exists():
            continue
        if target.is_dir():
            found.extend(sorted(target.rglob("*.py")))
        else:
            found.append(target)
    return found


def _file_matches_markers(text: str, markers: tuple[str, ...]) -> bool:
    return any(marker in text for marker in markers)


def check_language_detection_export_isolation(root: Path) -> CheckResult:
    """E-X-002 (i): Language detection never mutates EXPORT/PDF/ or EXPORT/TEXT/.

    Static scan: for each candidate language-detection file that exists, assert
    no EXPORT/PDF or EXPORT/TEXT write patterns appear. When no candidate files
    exist (early-stage state before Plan A lands) the assertion passes with a
    note.
    """
    candidates = _iter_candidate_files(root, _LANGUAGE_DETECTION_CANDIDATE_PATHS)
    if not candidates:
        return CheckResult(
            name="language_detection_export_isolation",
            passed=True,
            message=("No Plan A language-detection files present yet; nothing to scan"),
            details=["Plan A has not shipped; assertion passes with no-op scan"],
        )

    issues: list[str] = []
    scanned: list[str] = []
    for path in candidates:
        try:
            text = _read(path)
        except OSError as exc:  # unreadable file is a hard fail
            issues.append(f"Could not read {path.relative_to(root)}: {exc}")
            continue
        if not _file_matches_markers(text, _LANGUAGE_DETECTION_MARKERS):
            # File exists at a candidate path but is not actually
            # language-detection code; skip.
            continue
        scanned.append(str(path.relative_to(root)))
        for pattern in _EXPORT_WRITE_PATTERNS:
            if re.search(pattern, text):
                issues.append(
                    f"{path.relative_to(root)} references forbidden export path "
                    f"matching {pattern!r} (language detection must not mutate "
                    "EXPORT/PDF or EXPORT/TEXT)"
                )

    if issues:
        return CheckResult(
            name="language_detection_export_isolation",
            passed=False,
            message=f"{len(issues)} language-detection export-isolation issue(s)",
            details=issues,
        )

    return CheckResult(
        name="language_detection_export_isolation",
        passed=True,
        message=(
            "Language detection files do not mutate EXPORT/PDF or EXPORT/TEXT"
            if scanned
            else "Candidate paths present but none matched language-detection markers"
        ),
        details=[f"Scanned: {p}" for p in scanned],
    )


def check_translation_certified_filter(root: Path) -> CheckResult:
    """E-X-002 (ii): Translation engines must apply a certified=true filter
    before calling output_assembler.build_retrieval_output.

    Static scan: for each candidate translation-engine file that exists,
    if it references ``build_retrieval_output`` or ``output_assembler`` it
    must also contain a ``certified=True`` or ``certified=true`` filter on
    the call path. When no candidate files exist (pre-Plan-B) the assertion
    passes with a note.
    """
    candidates = _iter_candidate_files(root, _TRANSLATION_ENGINE_CANDIDATE_PATHS)
    if not candidates:
        return CheckResult(
            name="translation_certified_filter",
            passed=True,
            message=("No Plan B translation-engine files present yet; nothing to scan"),
            details=["Plan B has not shipped; assertion passes with no-op scan"],
        )

    # Forbidden: reference to retrieval output assembler without certified filter.
    assembler_markers = ("build_retrieval_output", "output_assembler")
    certified_patterns = (
        r"certified\s*=\s*True",
        r'certified\s*=\s*["\']true["\']',
        r'"certified"\s*:\s*True',
        r"'certified'\s*:\s*True",
    )

    issues: list[str] = []
    scanned: list[str] = []
    for path in candidates:
        try:
            text = _read(path)
        except OSError as exc:
            issues.append(f"Could not read {path.relative_to(root)}: {exc}")
            continue
        if not _file_matches_markers(text, _TRANSLATION_ENGINE_MARKERS):
            continue
        scanned.append(str(path.relative_to(root)))
        touches_assembler = any(marker in text for marker in assembler_markers)
        if not touches_assembler:
            continue
        has_certified = any(re.search(pattern, text) for pattern in certified_patterns)
        if not has_certified:
            issues.append(
                f"{path.relative_to(root)} references "
                "output_assembler / build_retrieval_output without a "
                "certified=True filter (translation output must be gated "
                "on the certified flag per E-X-002 ii / RED-03)"
            )

    if issues:
        return CheckResult(
            name="translation_certified_filter",
            passed=False,
            message=f"{len(issues)} translation certified-filter issue(s)",
            details=issues,
        )

    return CheckResult(
        name="translation_certified_filter",
        passed=True,
        message=(
            "Translation engines route retrieval output through a certified "  "filter"
            if scanned
            else "Candidate paths present but none matched translation-engine markers"
        ),
        details=[f"Scanned: {p}" for p in scanned],
    )


def check_plugin_entry_point_signatures(root: Path) -> CheckResult:
    """E-X-002 (iii): Installed entry-point plugins carry verified signatures.

    For each entry point installed under the ``ocr_local.enrichments`` group,
    assert that a ``signature`` attribute is present on the loaded plugin
    object (or in its distribution metadata). When no plugins are registered
    -- the expected state in early development -- the assertion passes with
    a logged note.

    This is a CI-time check and does not depend on project-root files: it
    inspects the Python environment the validator is running under.
    """
    try:
        from importlib import metadata as importlib_metadata
    except ImportError as exc:  # pragma: no cover - stdlib on 3.10+
        return CheckResult(
            name="plugin_entry_point_signatures",
            passed=False,
            message=f"importlib.metadata unavailable: {exc}",
        )

    try:
        entry_points = importlib_metadata.entry_points()
    except Exception as exc:
        return CheckResult(
            name="plugin_entry_point_signatures",
            passed=False,
            message=f"Failed to enumerate entry points: {exc}",
        )

    # importlib.metadata.entry_points() returns an EntryPoints / dict
    # depending on Python version; select the enrichments group uniformly.
    group_name = "ocr_local.enrichments"
    selected: list = []
    try:
        # Python 3.10+ select() API
        selected = list(entry_points.select(group=group_name))
    except AttributeError:
        # Legacy mapping API
        selected = list(entry_points.get(group_name, []))  # type: ignore[arg-type]

    if not selected:
        return CheckResult(
            name="plugin_entry_point_signatures",
            passed=True,
            message=(
                "No plugins registered under 'ocr_local.enrichments' "
                "(expected in early development)"
            ),
            details=["Entry-point group is empty; assertion is a no-op"],
        )

    issues: list[str] = []
    inspected: list[str] = []
    for ep in selected:
        ep_label = f"{ep.name} -> {ep.value}"
        inspected.append(ep_label)
        # Check 1: attempt to load the plugin and look for a `signature`
        # attribute on the resulting object.
        has_signature = False
        try:
            obj = ep.load()
            if getattr(obj, "signature", None):
                has_signature = True
        except Exception as exc:
            issues.append(f"Entry point {ep_label} failed to load: {exc}")
            continue

        # Check 2: fall back to distribution metadata for a Plugin-Signature
        # / signature marker, which is where E-X-002 expects verified
        # signatures to be published at build time.
        if not has_signature:
            dist = getattr(ep, "dist", None)
            if dist is not None:
                meta = getattr(dist, "metadata", None)
                if meta is not None:
                    for key in ("Plugin-Signature", "X-Plugin-Signature", "Signature"):
                        try:
                            value = meta.get(key)
                        except Exception:
                            value = None
                        if value:
                            has_signature = True
                            break

        if not has_signature:
            issues.append(
                f"Entry point {ep_label} does not declare a verified "
                "signature (expected `signature` attribute on the plugin "
                "object or `Plugin-Signature` distribution metadata per "
                "E-X-002 iii / RED-02)"
            )

    if issues:
        return CheckResult(
            name="plugin_entry_point_signatures",
            passed=False,
            message=f"{len(issues)} plugin signature issue(s)",
            details=issues,
        )

    return CheckResult(
        name="plugin_entry_point_signatures",
        passed=True,
        message=(
            f"All {len(selected)} registered ocr_local.enrichments plugin(s) "
            "declare verified signatures"
        ),
        details=[f"Inspected: {ep}" for ep in inspected],
    )


def check_engine_catalog_boundary(root: Path) -> CheckResult:
    """Pluggable-engine boundary: forensic-lane engines stay deterministic CTC.

    Asserts the built-in engine catalog's invariants so a generative or
    non-permissive engine can never sit in the forensic lane or be admitted by
    the default (forensic_mode=True) policy.

    When the engines package genuinely does not exist on disk (pre-Phase-0
    commit) the assertion passes with a note. But when the package IS present it
    MUST import: running this script directly puts ``scripts/`` on ``sys.path[0]``
    rather than the repo root, so the repo root is added to ``sys.path`` here, and
    a present-but-unimportable package fails loudly instead of silently no-opping
    (a silent skip gives this boundary assertion zero real coverage).
    """
    engines_pkg = root / "ocr_local" / "engines"
    if not engines_pkg.exists():
        return CheckResult(
            name="engine_catalog_boundary",
            passed=True,
            message="ocr_local.engines package not present on this commit; nothing to scan",
        )

    # Package exists on disk -> make the repo root importable, then require import.
    root_str = str(root.resolve())
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    try:
        from ocr_local.engines import catalog
        from ocr_local.engines.governance import EnginePolicy, evaluate_engine
    except Exception as exc:  # present on disk but broken -> fail loudly
        return CheckResult(
            name="engine_catalog_boundary",
            passed=False,
            message=f"ocr_local.engines is present on disk but failed to import: {exc}",
        )

    issues: list[str] = []
    required_routing_attrs = (
        "scripts",
        "handwriting",
        "tables",
        "vertical",
        "cost_class",
        "gpu_required",
    )
    allowed_cost_classes = {"low", "medium", "high", "unknown"}
    for cap in catalog.BUILTIN_CAPABILITIES:
        engine_id = getattr(cap, "engine_id", "<unknown>")
        for attr in required_routing_attrs:
            if not hasattr(cap, attr):
                issues.append(f"{engine_id}: missing routing field {attr}")
        cost_class = getattr(cap, "cost_class", None)
        gpu_required = getattr(cap, "gpu_required", False)
        lane = getattr(cap, "lane", None)
        is_generative = getattr(cap, "is_generative", False)
        is_deterministic_ctc = getattr(cap, "is_deterministic_ctc", False)
        license_class = getattr(cap, "license_class", None)
        requires_cloud_egress = getattr(cap, "requires_cloud_egress", False)
        tables = getattr(cap, "tables", False)
        vertical = getattr(cap, "vertical", False)
        if cost_class not in allowed_cost_classes:
            issues.append(f"{engine_id}: invalid cost_class {cost_class!r}")
        if gpu_required and lane == "forensic_ctc":
            issues.append(f"{engine_id}: forensic-lane engine must not require GPU")
        if lane == "forensic_ctc":
            if is_generative:
                issues.append(f"{engine_id}: forensic-lane engine is generative")
            if not is_deterministic_ctc:
                issues.append(f"{engine_id}: forensic-lane engine is not CTC")
            if license_class != "PERMISSIVE":
                issues.append(
                    f"{engine_id}: forensic-lane engine has non-permissive "
                    f"license_class {license_class}"
                )
            if requires_cloud_egress:
                issues.append(
                    f"{engine_id}: forensic-lane engine requires cloud egress"
                )
            if tables:
                issues.append(
                    f"{engine_id}: forensic-lane engine declares specialist "
                    "table routing support"
                )
            if vertical:
                issues.append(
                    f"{engine_id}: forensic-lane engine declares specialist "
                    "vertical-text routing support"
                )

    # Default policy is forensic_mode=True -> only deterministic CTC engines
    # may be admitted; no generative engine can pass.
    default_policy = EnginePolicy()
    for cap in catalog.BUILTIN_CAPABILITIES:
        engine_id = getattr(cap, "engine_id", "<unknown>")
        is_generative = getattr(cap, "is_generative", False)
        is_deterministic_ctc = getattr(cap, "is_deterministic_ctc", False)
        try:
            admitted = evaluate_engine(cap, default_policy).admitted
        except AttributeError as exc:
            issues.append(f"{engine_id}: cannot evaluate default policy: {exc}")
            continue
        if admitted:
            if is_generative:
                issues.append(
                    f"{engine_id}: generative engine admitted under default "
                    "forensic policy"
                )
            if not is_deterministic_ctc:
                issues.append(
                    f"{engine_id}: non-CTC engine admitted under default "
                    "forensic policy"
                )

    if issues:
        return CheckResult(
            name="engine_catalog_boundary",
            passed=False,
            message=f"{len(issues)} engine catalog boundary issue(s)",
            details=issues,
        )
    return CheckResult(
        name="engine_catalog_boundary",
        passed=True,
        message="Engine catalog forensic-lane invariants hold (CTC-only, permissive, local)",
    )


ALL_CHECKS = [
    check_job_submit_defaults,
    check_core_guardrails,
    check_ai_feature_defaults,
    check_queue_isolation,
    check_boundary_contract_doc,
    check_api_reference_boundary,
    check_language_detection_export_isolation,
    check_translation_certified_filter,
    check_plugin_entry_point_signatures,
    check_engine_catalog_boundary,
]


def run_all_checks(root: Path) -> ValidationReport:
    report = ValidationReport(
        timestamp=datetime.now(timezone.utc).isoformat(),
        project_root=str(root),
    )
    for check_fn in ALL_CHECKS:
        report.add(check_fn(root))
    return report


def format_text(report: ValidationReport) -> str:
    lines: list[str] = []
    lines.append("Feature Boundary Validation Report")
    lines.append("=" * 40)
    lines.append(f"Timestamp: {report.timestamp}")
    lines.append(f"Root:      {report.project_root}")
    lines.append("")
    for check in report.checks:
        status = "PASS" if check.passed else "FAIL"
        lines.append(f"[{status}] {check.name}: {check.message}")
        for detail in check.details:
            lines.append(f"       {detail}")
        lines.append("")
    passed = sum(1 for c in report.checks if c.passed)
    total = len(report.checks)
    lines.append("-" * 40)
    lines.append(f"Result: {passed}/{total} checks passed")
    lines.append(f"Overall: {'PASS' if report.passed else 'FAIL'}")
    return "\n".join(lines)


def format_json(report: ValidationReport) -> str:
    return json.dumps(
        {
            "passed": report.passed,
            "timestamp": report.timestamp,
            "project_root": report.project_root,
            "checks": [asdict(c) for c in report.checks],
            "summary": {
                "total": len(report.checks),
                "passed": sum(1 for c in report.checks if c.passed),
                "failed": sum(1 for c in report.checks if not c.passed),
            },
        },
        indent=2,
    )


def format_markdown(report: ValidationReport) -> str:
    lines: list[str] = []
    lines.append("# Feature Boundary Validation Report")
    lines.append("")
    lines.append(f"**Generated**: {report.timestamp}")
    lines.append(f"**Project root**: `{report.project_root}`")
    lines.append("")
    passed = sum(1 for c in report.checks if c.passed)
    total = len(report.checks)
    lines.append(
        f"**Result**: {passed}/{total} checks passed ({'PASS' if report.passed else 'FAIL'})"
    )
    lines.append("")
    lines.append("| Check | Status | Message |")
    lines.append("|---|---|---|")
    for check in report.checks:
        lines.append(
            f"| {check.name} | {'PASS' if check.passed else 'FAIL'} | {check.message} |"
        )
    lines.append("")
    lines.append("## Details")
    lines.append("")
    for check in report.checks:
        if check.details:
            lines.append(f"### {check.name}")
            lines.append("")
            for detail in check.details:
                lines.append(f"- {detail}")
            lines.append("")
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate the forensic-core versus AI-adjacent boundary.",
    )
    parser.add_argument("--project-root", type=Path, default=None)
    parser.add_argument("--json", action="store_true", dest="json_output")
    parser.add_argument("--report", type=Path, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    root = args.project_root if args.project_root else DEFAULT_PROJECT_ROOT

    if not root.exists():
        _safe_print(f"Error: project root not found: {root}", file=sys.stderr)
        return 2

    try:
        report = run_all_checks(root)
    except Exception as exc:
        _safe_print(f"Error running checks: {exc}", file=sys.stderr)
        return 2

    if args.json_output:
        _safe_print(format_json(report))
    else:
        _safe_print(format_text(report))

    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(format_markdown(report), encoding="utf-8")
        _safe_print(f"\nReport written to {args.report}")

    return 0 if report.passed else 1


if __name__ == "__main__":
    sys.exit(main())
