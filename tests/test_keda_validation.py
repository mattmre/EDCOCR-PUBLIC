"""
Unit tests for scripts/validate_keda.py.

Tests cover template parsing, scale-to-zero validation, threshold validation,
TriggerAuthentication guard coverage, and the bug fix for the missing
cpuOcrWorker check in the TriggerAuthentication conditional.

Run with: python -m pytest tests/test_keda_validation.py -v
"""

import json
import os
import sys
import textwrap
from pathlib import Path

import pytest

sys.path.insert(
    0,
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"),
)

from scripts.validate_keda import (  # noqa: E402
    EXPECTED_QUEUES,
    ScaledObjectInfo,
    TriggerAuthInfo,
    ValidationReport,
    ValidationResult,
    _check_trigger_auth_guard_coverage,
    _extract_metadata_component,
    _extract_queue_name,
    _extract_template_expr,
    _extract_trigger_auth_ref,
    _extract_yaml_value,
    _find_chart_dir,
    build_parser,
    main,
    parse_scaled_object,
    parse_trigger_auth,
    run_validation,
    validate_cooldown_period,
    validate_polling_interval,
    validate_queue_names,
    validate_scale_to_zero,
    validate_scale_up_threshold,
    validate_scaling_strategy_support,
    validate_trigger_auth,
    validate_values_defaults,
)

# ===========================================================================
# Fixtures
# ===========================================================================

SAMPLE_SCALED_OBJECT = textwrap.dedent("""\
    {{- if .Values.gpuWorker.autoscaling.enabled }}
    apiVersion: keda.sh/v1alpha1
    kind: ScaledObject
    metadata:
      name: {{ .Release.Name }}-gpu-worker-scaler
      labels:
        app.kubernetes.io/component: gpu-worker
    spec:
      scaleTargetRef:
        name: {{ .Release.Name }}-gpu-worker
      minReplicaCount: {{ .Values.gpuWorker.autoscaling.minReplicas }}
      maxReplicaCount: {{ .Values.gpuWorker.autoscaling.maxReplicas }}
      pollingInterval: {{ .Values.gpuWorker.autoscaling.pollingInterval }}
      cooldownPeriod: {{ .Values.gpuWorker.autoscaling.cooldownPeriod }}
      triggers:
        - type: rabbitmq
          metadata:
            protocol: amqp
            queueName: ocr_gpu
            mode: QueueLength
            value: {{ .Values.gpuWorker.autoscaling.queueTarget | quote }}
          authenticationRef:
            name: {{ .Release.Name }}-rabbitmq-auth
    {{- end }}
""")

SAMPLE_TRIGGER_AUTH_FIXED = textwrap.dedent("""\
    {{- if or .Values.gpuWorker.autoscaling.enabled .Values.cpuWorker.autoscaling.enabled (and .Values.cpuOcrWorker.enabled .Values.cpuOcrWorker.autoscaling.enabled) (and .Values.layoutCpuWorker.enabled .Values.layoutCpuWorker.autoscaling.enabled) (and .Values.nlpGpuWorker.enabled .Values.nlpGpuWorker.autoscaling.enabled) (and .Values.layoutlmWorker.enabled .Values.layoutlmWorker.autoscaling.enabled) }}
    apiVersion: keda.sh/v1alpha1
    kind: TriggerAuthentication
    metadata:
      name: {{ .Release.Name }}-rabbitmq-auth
      labels:
        {{- include "ocr-local.labels" . | nindent 4 }}
    spec:
      secretTargetRef:
        - parameter: host
          name: {{ .Release.Name }}-secret
          key: CELERY_BROKER_URL
    {{- end }}
""")

SAMPLE_TRIGGER_AUTH_BUGGY = textwrap.dedent("""\
    {{- if or .Values.gpuWorker.autoscaling.enabled .Values.cpuWorker.autoscaling.enabled (and .Values.layoutCpuWorker.enabled .Values.layoutCpuWorker.autoscaling.enabled) (and .Values.nlpGpuWorker.enabled .Values.nlpGpuWorker.autoscaling.enabled) (and .Values.layoutlmWorker.enabled .Values.layoutlmWorker.autoscaling.enabled) }}
    apiVersion: keda.sh/v1alpha1
    kind: TriggerAuthentication
    metadata:
      name: {{ .Release.Name }}-rabbitmq-auth
    spec:
      secretTargetRef:
        - parameter: host
          name: {{ .Release.Name }}-secret
          key: CELERY_BROKER_URL
    {{- end }}
""")


@pytest.fixture
def sample_scaled_objects():
    """Create representative ScaledObjectInfo instances for all 5 worker types."""
    return [
        ScaledObjectInfo(
            file_name="keda-gpu-scaler.yaml",
            component="gpu-worker",
            queue_name="ocr_gpu",
            min_replicas_expr="{{ .Values.gpuWorker.autoscaling.minReplicas }}",
            max_replicas_expr="{{ .Values.gpuWorker.autoscaling.maxReplicas }}",
            polling_interval_expr="{{ .Values.gpuWorker.autoscaling.pollingInterval }}",
            cooldown_period_expr="{{ .Values.gpuWorker.autoscaling.cooldownPeriod }}",
            queue_target_expr='{{ .Values.gpuWorker.autoscaling.queueTarget | quote }}',
            trigger_auth_ref="{{ .Release.Name }}-rabbitmq-auth",
            raw_content=SAMPLE_SCALED_OBJECT,
        ),
        ScaledObjectInfo(
            file_name="keda-cpu-scaler.yaml",
            component="cpu-worker",
            queue_name="cpu_general",
            min_replicas_expr="{{ .Values.cpuWorker.autoscaling.minReplicas }}",
            max_replicas_expr="{{ .Values.cpuWorker.autoscaling.maxReplicas }}",
            polling_interval_expr="{{ .Values.cpuWorker.autoscaling.pollingInterval }}",
            cooldown_period_expr="{{ .Values.cpuWorker.autoscaling.cooldownPeriod }}",
            queue_target_expr='{{ .Values.cpuWorker.autoscaling.queueTarget | quote }}',
            trigger_auth_ref="{{ .Release.Name }}-rabbitmq-auth",
            raw_content="",
        ),
        ScaledObjectInfo(
            file_name="keda-cpu-ocr-scaler.yaml",
            component="cpu-ocr-worker",
            queue_name="ocr_cpu",
            min_replicas_expr="{{ .Values.cpuOcrWorker.autoscaling.minReplicas }}",
            max_replicas_expr="{{ .Values.cpuOcrWorker.autoscaling.maxReplicas }}",
            polling_interval_expr="{{ .Values.cpuOcrWorker.autoscaling.pollingInterval }}",
            cooldown_period_expr="{{ .Values.cpuOcrWorker.autoscaling.cooldownPeriod }}",
            queue_target_expr='{{ .Values.cpuOcrWorker.autoscaling.queueTarget | quote }}',
            trigger_auth_ref="{{ .Release.Name }}-rabbitmq-auth",
            raw_content="",
        ),
        ScaledObjectInfo(
            file_name="keda-layout-cpu-scaler.yaml",
            component="layout-cpu-worker",
            queue_name="ocr_layout_cpu",
            min_replicas_expr="{{ .Values.layoutCpuWorker.autoscaling.minReplicas }}",
            max_replicas_expr="{{ .Values.layoutCpuWorker.autoscaling.maxReplicas }}",
            polling_interval_expr="{{ .Values.layoutCpuWorker.autoscaling.pollingInterval }}",
            cooldown_period_expr="{{ .Values.layoutCpuWorker.autoscaling.cooldownPeriod }}",
            queue_target_expr='{{ .Values.layoutCpuWorker.autoscaling.queueTarget | quote }}',
            trigger_auth_ref="{{ .Release.Name }}-rabbitmq-auth",
            raw_content="",
        ),
        ScaledObjectInfo(
            file_name="keda-nlp-gpu-scaler.yaml",
            component="nlp-gpu-worker",
            queue_name="ocr_nlp_gpu",
            min_replicas_expr="{{ .Values.nlpGpuWorker.autoscaling.minReplicas }}",
            max_replicas_expr="{{ .Values.nlpGpuWorker.autoscaling.maxReplicas }}",
            polling_interval_expr="{{ .Values.nlpGpuWorker.autoscaling.pollingInterval }}",
            cooldown_period_expr="{{ .Values.nlpGpuWorker.autoscaling.cooldownPeriod }}",
            queue_target_expr='{{ .Values.nlpGpuWorker.autoscaling.queueTarget | quote }}',
            trigger_auth_ref="{{ .Release.Name }}-rabbitmq-auth",
            raw_content="",
        ),
        ScaledObjectInfo(
            file_name="keda-layoutlm-scaler.yaml",
            component="layoutlm-worker",
            queue_name="ocr_layoutlm",
            min_replicas_expr="{{ .Values.layoutlmWorker.autoscaling.minReplicas }}",
            max_replicas_expr="{{ .Values.layoutlmWorker.autoscaling.maxReplicas }}",
            polling_interval_expr="{{ .Values.layoutlmWorker.autoscaling.pollingInterval }}",
            cooldown_period_expr="{{ .Values.layoutlmWorker.autoscaling.cooldownPeriod }}",
            queue_target_expr='{{ .Values.layoutlmWorker.autoscaling.queueTarget | quote }}',
            trigger_auth_ref="{{ .Release.Name }}-rabbitmq-auth",
            raw_content="",
        ),
    ]


@pytest.fixture
def fixed_trigger_auth():
    return TriggerAuthInfo(
        file_name="keda-trigger-auth.yaml",
        name_expr="{{ .Release.Name }}-rabbitmq-auth",
        guard_condition=(
            "{{- if or .Values.gpuWorker.autoscaling.enabled "
            ".Values.cpuWorker.autoscaling.enabled "
            "(and .Values.cpuOcrWorker.enabled .Values.cpuOcrWorker.autoscaling.enabled) "
            "(and .Values.layoutCpuWorker.enabled .Values.layoutCpuWorker.autoscaling.enabled) "
            "(and .Values.nlpGpuWorker.enabled .Values.nlpGpuWorker.autoscaling.enabled) "
            "(and .Values.layoutlmWorker.enabled .Values.layoutlmWorker.autoscaling.enabled) }}"
        ),
        secret_ref_name="{{ .Release.Name }}-secret",
        secret_ref_key="CELERY_BROKER_URL",
        raw_content=SAMPLE_TRIGGER_AUTH_FIXED,
    )


@pytest.fixture
def buggy_trigger_auth():
    return TriggerAuthInfo(
        file_name="keda-trigger-auth.yaml",
        name_expr="{{ .Release.Name }}-rabbitmq-auth",
        guard_condition=(
            "{{- if or .Values.gpuWorker.autoscaling.enabled "
            ".Values.cpuWorker.autoscaling.enabled "
            "(and .Values.layoutCpuWorker.enabled .Values.layoutCpuWorker.autoscaling.enabled) "
            "(and .Values.nlpGpuWorker.enabled .Values.nlpGpuWorker.autoscaling.enabled) "
            "(and .Values.layoutlmWorker.enabled .Values.layoutlmWorker.autoscaling.enabled) }}"
        ),
        secret_ref_name="{{ .Release.Name }}-secret",
        secret_ref_key="CELERY_BROKER_URL",
        raw_content=SAMPLE_TRIGGER_AUTH_BUGGY,
    )


# ===========================================================================
# Template parsing tests
# ===========================================================================

class TestTemplateParsing:
    """Tests for YAML template parsing helpers."""

    def test_extract_metadata_component(self):
        content = "  app.kubernetes.io/component: gpu-worker\n"
        assert _extract_metadata_component(content) == "gpu-worker"

    def test_extract_metadata_component_unknown(self):
        content = "metadata:\n  name: test\n"
        assert _extract_metadata_component(content) == "unknown"

    def test_extract_queue_name(self):
        content = "    queueName: ocr_gpu\n"
        assert _extract_queue_name(content) == "ocr_gpu"

    def test_extract_queue_name_cpu(self):
        content = "    queueName: cpu_general\n"
        assert _extract_queue_name(content) == "cpu_general"

    def test_extract_queue_name_helm_template(self):
        content = '    queueName: {{ .Values.keda.gpu.queueName | default "ocr_gpu" }}\n'
        assert _extract_queue_name(content) == "ocr_gpu"

    def test_extract_queue_name_helm_template_cpu(self):
        content = '    queueName: {{ .Values.keda.cpu.queueName | default "cpu_general" }}\n'
        assert _extract_queue_name(content) == "cpu_general"

    def test_extract_queue_name_helm_template_cpu_ocr(self):
        content = '    queueName: {{ .Values.keda.cpuOcr.queueName | default "ocr_cpu" }}\n'
        assert _extract_queue_name(content) == "ocr_cpu"

    def test_extract_queue_name_missing(self):
        content = "metadata:\n  name: test\n"
        assert _extract_queue_name(content) == "unknown"

    def test_extract_trigger_auth_ref(self):
        content = (
            "      authenticationRef:\n"
            "        name: release-rabbitmq-auth\n"
        )
        assert _extract_trigger_auth_ref(content) == "release-rabbitmq-auth"

    def test_extract_trigger_auth_ref_missing(self):
        content = "triggers:\n  - type: rabbitmq\n"
        assert _extract_trigger_auth_ref(content) == ""

    def test_extract_template_expr(self):
        content = "  pollingInterval: {{ .Values.gpuWorker.autoscaling.pollingInterval }}\n"
        result = _extract_template_expr(content, "pollingInterval")
        assert ".Values.gpuWorker.autoscaling.pollingInterval" in result

    def test_extract_template_expr_missing(self):
        content = "  name: test\n"
        assert _extract_template_expr(content, "pollingInterval") == ""

    def test_extract_yaml_value(self):
        content = "  cooldownPeriod: 300\n"
        assert _extract_yaml_value(content, "cooldownPeriod") == "300"

    def test_extract_yaml_value_missing(self):
        content = "  name: test\n"
        assert _extract_yaml_value(content, "cooldownPeriod") is None

    def test_parse_scaled_object_from_file(self, tmp_path):
        f = tmp_path / "keda-gpu-scaler.yaml"
        f.write_text(SAMPLE_SCALED_OBJECT, encoding="utf-8")
        so = parse_scaled_object(f)
        assert so is not None
        assert so.component == "gpu-worker"
        assert so.queue_name == "ocr_gpu"
        assert ".Values." in so.min_replicas_expr

    def test_parse_scaled_object_non_keda(self, tmp_path):
        f = tmp_path / "deployment.yaml"
        f.write_text("apiVersion: apps/v1\nkind: Deployment\n", encoding="utf-8")
        assert parse_scaled_object(f) is None

    def test_parse_trigger_auth_from_file(self, tmp_path):
        f = tmp_path / "keda-trigger-auth.yaml"
        f.write_text(SAMPLE_TRIGGER_AUTH_FIXED, encoding="utf-8")
        ta = parse_trigger_auth(f)
        assert ta is not None
        assert ta.secret_ref_key == "CELERY_BROKER_URL"
        assert "cpuOcrWorker" in ta.guard_condition

    def test_parse_trigger_auth_non_keda(self, tmp_path):
        f = tmp_path / "secret.yaml"
        f.write_text("apiVersion: v1\nkind: Secret\n", encoding="utf-8")
        assert parse_trigger_auth(f) is None


# ===========================================================================
# Queue name validation tests
# ===========================================================================

class TestQueueNameValidation:
    """Tests for queue name correctness validation."""

    def test_all_queue_names_correct(self, sample_scaled_objects):
        results = validate_queue_names(sample_scaled_objects)
        assert all(r.passed for r in results)
        assert len(results) == 6

    def test_wrong_queue_name_detected(self):
        so = ScaledObjectInfo(
            file_name="keda-gpu-scaler.yaml",
            component="gpu-worker",
            queue_name="wrong_queue",
            min_replicas_expr="1",
            max_replicas_expr="10",
            polling_interval_expr="15",
            cooldown_period_expr="300",
            queue_target_expr="5",
            trigger_auth_ref="release-rabbitmq-auth",
            raw_content="",
        )
        results = validate_queue_names([so])
        assert len(results) == 1
        assert not results[0].passed
        assert "wrong_queue" in results[0].message

    def test_unknown_component_warning(self):
        so = ScaledObjectInfo(
            file_name="keda-mystery-scaler.yaml",
            component="mystery-worker",
            queue_name="mystery_queue",
            min_replicas_expr="1",
            max_replicas_expr="10",
            polling_interval_expr="15",
            cooldown_period_expr="300",
            queue_target_expr="5",
            trigger_auth_ref="",
            raw_content="",
        )
        results = validate_queue_names([so])
        assert len(results) == 1
        assert not results[0].passed
        assert results[0].severity == "warning"

    def test_expected_queues_complete(self):
        """Verify EXPECTED_QUEUES covers all 6 worker types."""
        expected_components = {
            "gpu-worker", "cpu-worker", "cpu-ocr-worker",
            "layout-cpu-worker", "nlp-gpu-worker", "layoutlm-worker",
        }
        assert set(EXPECTED_QUEUES.keys()) == expected_components


# ===========================================================================
# Scale-to-zero validation tests
# ===========================================================================

class TestScaleToZeroValidation:
    """Tests for minReplicaCount=0 configuration."""

    def test_valid_min_replicas_expression(self, sample_scaled_objects):
        results = validate_scale_to_zero(sample_scaled_objects)
        assert all(r.passed for r in results)

    def test_missing_min_replicas(self):
        so = ScaledObjectInfo(
            file_name="test.yaml",
            component="gpu-worker",
            queue_name="ocr_gpu",
            min_replicas_expr="",
            max_replicas_expr="10",
            polling_interval_expr="15",
            cooldown_period_expr="300",
            queue_target_expr="5",
            trigger_auth_ref="",
            raw_content="",
        )
        results = validate_scale_to_zero([so])
        assert len(results) == 1
        assert not results[0].passed
        assert "missing" in results[0].message

    def test_literal_zero_accepted(self):
        so = ScaledObjectInfo(
            file_name="test.yaml",
            component="cpu-worker",
            queue_name="cpu_general",
            min_replicas_expr="0",
            max_replicas_expr="10",
            polling_interval_expr="15",
            cooldown_period_expr="120",
            queue_target_expr="10",
            trigger_auth_ref="",
            raw_content="",
        )
        results = validate_scale_to_zero([so])
        assert len(results) == 1
        assert results[0].passed


# ===========================================================================
# Scale-up threshold validation tests
# ===========================================================================

class TestScaleUpThresholdValidation:
    """Tests for queue depth threshold validation."""

    def test_valid_queue_targets(self, sample_scaled_objects):
        results = validate_scale_up_threshold(sample_scaled_objects)
        assert all(r.passed for r in results)

    def test_missing_queue_target(self):
        so = ScaledObjectInfo(
            file_name="test.yaml",
            component="gpu-worker",
            queue_name="ocr_gpu",
            min_replicas_expr="1",
            max_replicas_expr="10",
            polling_interval_expr="15",
            cooldown_period_expr="300",
            queue_target_expr="",
            trigger_auth_ref="",
            raw_content="",
        )
        results = validate_scale_up_threshold([so])
        assert len(results) == 1
        assert not results[0].passed


# ===========================================================================
# Cooldown period validation tests
# ===========================================================================

class TestCooldownPeriodValidation:
    """Tests for stabilization window validation."""

    def test_valid_cooldown_periods(self, sample_scaled_objects):
        results = validate_cooldown_period(sample_scaled_objects)
        assert all(r.passed for r in results)

    def test_missing_cooldown(self):
        so = ScaledObjectInfo(
            file_name="test.yaml",
            component="gpu-worker",
            queue_name="ocr_gpu",
            min_replicas_expr="1",
            max_replicas_expr="10",
            polling_interval_expr="15",
            cooldown_period_expr="",
            queue_target_expr="5",
            trigger_auth_ref="",
            raw_content="",
        )
        results = validate_cooldown_period([so])
        assert not results[0].passed


# ===========================================================================
# Polling interval validation tests
# ===========================================================================

class TestPollingIntervalValidation:
    """Tests for polling interval validation."""

    def test_valid_polling_intervals(self, sample_scaled_objects):
        results = validate_polling_interval(sample_scaled_objects)
        assert all(r.passed for r in results)

    def test_missing_polling_interval(self):
        so = ScaledObjectInfo(
            file_name="test.yaml",
            component="gpu-worker",
            queue_name="ocr_gpu",
            min_replicas_expr="1",
            max_replicas_expr="10",
            polling_interval_expr="",
            cooldown_period_expr="300",
            queue_target_expr="5",
            trigger_auth_ref="",
            raw_content="",
        )
        results = validate_polling_interval([so])
        assert not results[0].passed


# ===========================================================================
# TriggerAuthentication validation tests
# ===========================================================================

class TestTriggerAuthValidation:
    """Tests for TriggerAuthentication reference and guard coverage."""

    def test_trigger_auth_exists(self, sample_scaled_objects, fixed_trigger_auth):
        results = validate_trigger_auth(sample_scaled_objects, fixed_trigger_auth)
        exists_check = [r for r in results if r.check == "trigger_auth:exists"]
        assert len(exists_check) == 1
        assert exists_check[0].passed

    def test_trigger_auth_missing(self, sample_scaled_objects):
        results = validate_trigger_auth(sample_scaled_objects, None)
        assert len(results) == 1
        assert not results[0].passed

    def test_trigger_auth_secret_key_correct(
        self, sample_scaled_objects, fixed_trigger_auth
    ):
        results = validate_trigger_auth(sample_scaled_objects, fixed_trigger_auth)
        key_check = [r for r in results if r.check == "trigger_auth:secret_key"]
        assert len(key_check) == 1
        assert key_check[0].passed

    def test_trigger_auth_secret_key_wrong(self, sample_scaled_objects):
        ta = TriggerAuthInfo(
            file_name="keda-trigger-auth.yaml",
            name_expr="test",
            guard_condition="",
            secret_ref_name="test",
            secret_ref_key="WRONG_KEY",
            raw_content="",
        )
        results = validate_trigger_auth(sample_scaled_objects, ta)
        key_check = [r for r in results if r.check == "trigger_auth:secret_key"]
        assert len(key_check) == 1
        assert not key_check[0].passed

    def test_trigger_auth_ref_consistency(
        self, sample_scaled_objects, fixed_trigger_auth
    ):
        results = validate_trigger_auth(sample_scaled_objects, fixed_trigger_auth)
        ref_check = [r for r in results if r.check == "trigger_auth:ref_consistency"]
        assert len(ref_check) == 1
        assert ref_check[0].passed

    def test_trigger_auth_ref_inconsistency(self, fixed_trigger_auth):
        objects = [
            ScaledObjectInfo(
                file_name="a.yaml", component="gpu-worker", queue_name="ocr_gpu",
                min_replicas_expr="1", max_replicas_expr="10",
                polling_interval_expr="15", cooldown_period_expr="300",
                queue_target_expr="5",
                trigger_auth_ref="auth-a",
                raw_content="",
            ),
            ScaledObjectInfo(
                file_name="b.yaml", component="cpu-worker", queue_name="cpu_general",
                min_replicas_expr="0", max_replicas_expr="10",
                polling_interval_expr="15", cooldown_period_expr="120",
                queue_target_expr="10",
                trigger_auth_ref="auth-b",
                raw_content="",
            ),
        ]
        results = validate_trigger_auth(objects, fixed_trigger_auth)
        ref_check = [r for r in results if r.check == "trigger_auth:ref_consistency"]
        assert len(ref_check) == 1
        assert not ref_check[0].passed

    def test_guard_covers_all_workers_fixed(
        self, sample_scaled_objects, fixed_trigger_auth
    ):
        results = validate_trigger_auth(sample_scaled_objects, fixed_trigger_auth)
        guard_checks = [r for r in results if r.check.startswith("trigger_auth:guard:")]
        assert len(guard_checks) == 6
        assert all(r.passed for r in guard_checks), (
            "Fixed guard should cover all 6 worker types: "
            + "; ".join(f"{r.check}: {r.message}" for r in guard_checks if not r.passed)
        )


# ===========================================================================
# Bug fix validation: cpuOcrWorker missing from guard
# ===========================================================================

class TestCpuOcrWorkerBugFix:
    """Test that the known bug (missing cpuOcrWorker in TriggerAuthentication
    guard condition) is detected by the validator and that the fix resolves it.
    """

    def test_buggy_guard_detected(self, sample_scaled_objects, buggy_trigger_auth):
        """The buggy guard is missing cpuOcrWorker -- validator should flag it."""
        results = _check_trigger_auth_guard_coverage(
            buggy_trigger_auth.guard_condition, sample_scaled_objects
        )
        cpu_ocr_check = [
            r for r in results if r.check == "trigger_auth:guard:cpu-ocr-worker"
        ]
        assert len(cpu_ocr_check) == 1
        assert not cpu_ocr_check[0].passed
        assert "MISSING" in cpu_ocr_check[0].message

    def test_fixed_guard_passes(self, sample_scaled_objects, fixed_trigger_auth):
        """The fixed guard includes cpuOcrWorker -- validator should pass."""
        results = _check_trigger_auth_guard_coverage(
            fixed_trigger_auth.guard_condition, sample_scaled_objects
        )
        cpu_ocr_check = [
            r for r in results if r.check == "trigger_auth:guard:cpu-ocr-worker"
        ]
        assert len(cpu_ocr_check) == 1
        assert cpu_ocr_check[0].passed

    def test_bug_fix_in_actual_template(self):
        """Verify the actual keda-trigger-auth.yaml on disk has been fixed."""
        repo_root = Path(__file__).resolve().parent.parent
        trigger_auth_path = (
            repo_root / "helm" / "ocr-local" / "templates" / "keda-trigger-auth.yaml"
        )
        if not trigger_auth_path.exists():
            pytest.skip("Helm chart not found at expected path")

        content = trigger_auth_path.read_text(encoding="utf-8")
        first_line = content.split("\n")[0]

        # The fixed version must include cpuOcrWorker
        assert "cpuOcrWorker" in first_line, (
            "keda-trigger-auth.yaml guard condition is still missing "
            "cpuOcrWorker check. First line:\n" + first_line
        )
        assert "cpuOcrWorker.enabled" in first_line
        assert "cpuOcrWorker.autoscaling.enabled" in first_line

    def test_other_workers_still_covered(self, sample_scaled_objects, buggy_trigger_auth):
        """Even in the buggy version, other workers are still covered."""
        results = _check_trigger_auth_guard_coverage(
            buggy_trigger_auth.guard_condition, sample_scaled_objects
        )
        gpu_check = [r for r in results if r.check == "trigger_auth:guard:gpu-worker"]
        cpu_check = [r for r in results if r.check == "trigger_auth:guard:cpu-worker"]
        layout_check = [
            r for r in results if r.check == "trigger_auth:guard:layout-cpu-worker"
        ]
        nlp_check = [
            r for r in results if r.check == "trigger_auth:guard:nlp-gpu-worker"
        ]
        layoutlm_check = [
            r for r in results if r.check == "trigger_auth:guard:layoutlm-worker"
        ]

        assert gpu_check[0].passed
        assert cpu_check[0].passed
        assert layout_check[0].passed
        assert nlp_check[0].passed
        assert layoutlm_check[0].passed


# ===========================================================================
# Scaling strategy validation tests
# ===========================================================================

class TestScalingStrategyValidation:
    """Tests for scaling strategy annotation support."""

    def test_strategy_override_detected(self):
        so = ScaledObjectInfo(
            file_name="keda-gpu-scaler.yaml",
            component="gpu-worker",
            queue_name="ocr_gpu",
            min_replicas_expr="1",
            max_replicas_expr="20",
            polling_interval_expr="15",
            cooldown_period_expr="300",
            queue_target_expr="5",
            trigger_auth_ref="",
            raw_content=(
                '{{- $strategy := .Values.keda.scalingStrategy }}\n'
                '{{- if eq $strategy "aggressive" }}\n'
                '{{- if eq $strategy "conservative" }}\n'
            ),
        )
        results = validate_scaling_strategy_support([so])
        assert len(results) == 1
        assert results[0].passed
        assert "aggressive" in results[0].message

    def test_fixed_polling_noted(self):
        so = ScaledObjectInfo(
            file_name="keda-cpu-scaler.yaml",
            component="cpu-worker",
            queue_name="cpu_general",
            min_replicas_expr="0",
            max_replicas_expr="10",
            polling_interval_expr="15",
            cooldown_period_expr="120",
            queue_target_expr="10",
            trigger_auth_ref="",
            raw_content="pollingInterval: 15\ncooldownPeriod: 120\n",
        )
        results = validate_scaling_strategy_support([so])
        assert len(results) == 1
        assert results[0].passed
        assert "fixed" in results[0].message


# ===========================================================================
# Values.yaml validation tests
# ===========================================================================

class TestValuesValidation:
    """Tests for values.yaml KEDA defaults validation."""

    def test_valid_values_file(self, tmp_path):
        values = tmp_path / "values.yaml"
        values.write_text(textwrap.dedent("""\
            keda:
              scalingStrategy: "balanced"
              maxReplicaCount: 50
            gpuWorker:
              autoscaling:
                enabled: false
            cpuWorker:
              autoscaling:
                enabled: false
            cpuOcrWorker:
              autoscaling:
                enabled: false
            layoutCpuWorker:
              autoscaling:
                enabled: false
            nlpGpuWorker:
              autoscaling:
                enabled: false
        """), encoding="utf-8")

        results = validate_values_defaults(values)
        error_results = [r for r in results if r.severity == "error" and not r.passed]
        assert len(error_results) == 0

    def test_missing_values_file(self, tmp_path):
        results = validate_values_defaults(tmp_path / "nonexistent.yaml")
        assert len(results) == 1
        assert not results[0].passed

    def test_invalid_strategy(self, tmp_path):
        values = tmp_path / "values.yaml"
        values.write_text("keda:\n  scalingStrategy: invalid_strategy\n", encoding="utf-8")
        results = validate_values_defaults(values)
        strategy_check = [r for r in results if r.check == "values:scaling_strategy"]
        assert len(strategy_check) == 1
        assert not strategy_check[0].passed

    def test_missing_max_replica_count(self, tmp_path):
        values = tmp_path / "values.yaml"
        values.write_text("keda:\n  scalingStrategy: balanced\n", encoding="utf-8")
        results = validate_values_defaults(values)
        max_check = [r for r in results if r.check == "values:max_replica_count"]
        assert len(max_check) == 1
        assert not max_check[0].passed


# ===========================================================================
# Full validation integration tests
# ===========================================================================

class TestFullValidation:
    """Integration tests running the full validation pipeline."""

    def test_run_validation_on_actual_chart(self):
        """Run validation against the real Helm chart in the repository."""
        repo_root = Path(__file__).resolve().parent.parent
        chart_dir = repo_root / "helm" / "ocr-local"
        if not (chart_dir / "templates").is_dir():
            pytest.skip("Helm chart not found at expected path")

        report = run_validation(chart_dir)

        # Should find all 6 ScaledObjects
        assert len(report.scaled_objects) == 6, (
            f"Expected 6 ScaledObjects, found {len(report.scaled_objects)}: "
            f"{[so.component for so in report.scaled_objects]}"
        )

        # Should find TriggerAuthentication
        assert report.trigger_auth is not None

        # After bug fix, all checks should pass
        errors = [r for r in report.results if not r.passed and r.severity == "error"]
        assert len(errors) == 0, (
            "Validation errors:\n"
            + "\n".join(f"  {r.check}: {r.message}" for r in errors)
        )

    def test_run_validation_missing_chart_dir(self, tmp_path):
        report = run_validation(tmp_path / "nonexistent")
        assert not report.passed
        assert report.error_count > 0

    def test_validation_report_to_dict(self, sample_scaled_objects, fixed_trigger_auth):
        report = ValidationReport(
            results=[
                ValidationResult("test", True, "OK"),
                ValidationResult("test2", False, "BAD", severity="error"),
            ],
            scaled_objects=sample_scaled_objects,
            trigger_auth=fixed_trigger_auth,
        )
        d = report.to_dict()
        assert d["passed"] is False
        assert d["error_count"] == 1
        assert len(d["checks"]) == 2
        assert len(d["scaled_objects"]) == 6

    def test_validation_report_all_pass(self):
        report = ValidationReport(
            results=[
                ValidationResult("a", True, "OK"),
                ValidationResult("b", True, "OK"),
            ],
        )
        assert report.passed
        assert report.error_count == 0
        assert report.warning_count == 0


# ===========================================================================
# CLI tests
# ===========================================================================

class TestCLI:
    """Tests for CLI argument parsing and main entry point."""

    def test_build_parser(self):
        parser = build_parser()
        args = parser.parse_args(["--chart-dir", "test/path", "--json", "--strict"])
        assert args.chart_dir == "test/path"
        assert args.json_output is True
        assert args.strict is True

    def test_build_parser_defaults(self):
        parser = build_parser()
        args = parser.parse_args([])
        assert args.chart_dir is None
        assert args.json_output is False
        assert args.strict is False

    def test_main_with_actual_chart(self):
        repo_root = Path(__file__).resolve().parent.parent
        chart_dir = repo_root / "helm" / "ocr-local"
        if not (chart_dir / "templates").is_dir():
            pytest.skip("Helm chart not found at expected path")

        exit_code = main(["--chart-dir", str(chart_dir)])
        assert exit_code == 0

    def test_main_json_output(self, capsys):
        repo_root = Path(__file__).resolve().parent.parent
        chart_dir = repo_root / "helm" / "ocr-local"
        if not (chart_dir / "templates").is_dir():
            pytest.skip("Helm chart not found at expected path")

        exit_code = main(["--chart-dir", str(chart_dir), "--json"])
        assert exit_code == 0

        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert "passed" in data
        assert "checks" in data
        assert "scaled_objects" in data

    def test_main_missing_chart(self, tmp_path):
        exit_code = main(["--chart-dir", str(tmp_path / "nonexistent")])
        assert exit_code == 1

    def test_find_chart_dir_auto_detect(self):
        """Test that auto-detection finds the chart in the repo."""
        try:
            chart_dir = _find_chart_dir()
            assert (chart_dir / "templates").is_dir()
        except FileNotFoundError:
            pytest.skip("Running from outside the repository root")

    def test_main_text_output(self, capsys):
        repo_root = Path(__file__).resolve().parent.parent
        chart_dir = repo_root / "helm" / "ocr-local"
        if not (chart_dir / "templates").is_dir():
            pytest.skip("Helm chart not found at expected path")

        exit_code = main(["--chart-dir", str(chart_dir)])
        captured = capsys.readouterr()
        assert "KEDA Validation Report" in captured.out
        assert "Overall:" in captured.out
        assert exit_code == 0
