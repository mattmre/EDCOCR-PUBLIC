"""
Unit tests for scripts/failover_drill.py.

Tests cover all drill subcommands, result persistence, report generation,
and CLI argument parsing. All external services (Docker, PostgreSQL,
Redis, RabbitMQ) are mocked.

Run with: python -m pytest tests/test_failover_drill.py -v
"""

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))

from scripts.failover_drill import (  # noqa: E402
    DRILL_REGISTRY,
    RTO_TARGETS,
    DrillReport,
    DrillResult,
    DrillStep,
    _get_pg_row_counts,
    _get_pg_scalar,
    _get_rabbitmq_queues,
    _new_drill_result,
    _validate_no_duplicate_pages,
    build_parser,
    docker_compose_action,
    docker_compose_exec,
    drill_postgres_switchover,
    drill_rabbitmq_node_failure,
    drill_redis_sentinel,
    drill_worker_crash,
    generate_report,
    load_drill_results,
    main,
    render_markdown_report,
    run_command,
    save_drill_result,
    wait_for_healthy,
)

# ===========================================================================
# Fixtures
# ===========================================================================


@pytest.fixture
def results_dir(tmp_path):
    """Temporary results directory."""
    d = tmp_path / "drill_results"
    d.mkdir()
    return str(d)


@pytest.fixture
def sample_result():
    """A sample DrillResult for testing."""
    return DrillResult(
        drill_id="test123",
        drill_type="postgres-switchover",
        timestamp="2026-03-15T00:00:00+00:00",
        pipeline_version="0.9.0",
        dry_run=True,
        rto_target_seconds=300.0,
        status="passed",
        recovery_time_seconds=12.5,
        rto_met=True,
        data_integrity_verified=True,
        steps=[
            DrillStep(
                name="pre_flight",
                description="Check health",
                executed=True,
                success=True,
            ).to_dict()
        ],
    )


@pytest.fixture
def sample_report(sample_result, results_dir):
    """Save a sample result and return the directory."""
    save_drill_result(sample_result, results_dir=results_dir)
    return results_dir


# ===========================================================================
# Tests: DrillStep data class
# ===========================================================================


class TestDrillStep:
    def test_to_dict_returns_all_fields(self):
        step = DrillStep(name="test", description="A test step")
        d = step.to_dict()
        assert d["name"] == "test"
        assert d["description"] == "A test step"
        assert d["executed"] is False
        assert d["success"] is False
        assert d["duration_seconds"] == 0.0

    def test_command_field_optional(self):
        step = DrillStep(name="test", description="desc")
        assert step.command is None
        d = step.to_dict()
        assert d["command"] is None

    def test_command_field_set(self):
        step = DrillStep(name="test", description="desc", command="ls -la")
        assert step.command == "ls -la"


# ===========================================================================
# Tests: DrillResult data class
# ===========================================================================


class TestDrillResult:
    def test_to_dict_round_trip(self, sample_result):
        d = sample_result.to_dict()
        assert d["drill_id"] == "test123"
        assert d["drill_type"] == "postgres-switchover"
        assert d["status"] == "passed"
        assert d["rto_met"] is True

    def test_to_json_valid(self, sample_result):
        j = sample_result.to_json()
        parsed = json.loads(j)
        assert parsed["drill_id"] == "test123"
        assert parsed["pipeline_version"] == "0.9.0"

    def test_defaults(self):
        result = DrillResult(
            drill_id="x",
            drill_type="test",
            timestamp="now",
            pipeline_version="0.0.0",
            dry_run=False,
            rto_target_seconds=60.0,
        )
        assert result.status == "pending"
        assert result.recovery_time_seconds == 0.0
        assert result.rto_met is False
        assert result.steps == []
        assert result.errors == []


# ===========================================================================
# Tests: DrillReport data class
# ===========================================================================


class TestDrillReport:
    def test_to_dict(self):
        report = DrillReport(
            report_id="r1",
            generated_at="2026-01-01",
            pipeline_version="0.9.0",
            overall_status="passed",
        )
        d = report.to_dict()
        assert d["report_id"] == "r1"
        assert d["overall_status"] == "passed"
        assert d["drills"] == []

    def test_summary_field_default(self):
        report = DrillReport(
            report_id="r2",
            generated_at="2026-01-01",
            pipeline_version="0.9.0",
        )
        assert report.summary == {}


# ===========================================================================
# Tests: run_command
# ===========================================================================


class TestRunCommand:
    def test_dry_run_returns_success(self):
        result = run_command(["echo", "hello"], dry_run=True)
        assert result.returncode == 0
        assert "[DRY RUN]" in result.stdout

    def test_dry_run_does_not_execute(self):
        result = run_command(["false"], dry_run=True)
        assert result.returncode == 0  # Would be 1 if executed

    @pytest.mark.skipif(sys.platform == "win32", reason="echo is not a standalone executable on Windows")
    def test_real_command_captures_output(self):
        result = run_command(["echo", "hello"])
        assert result.returncode == 0
        assert "hello" in result.stdout

    @pytest.mark.skipif(sys.platform != "win32", reason="Windows-specific echo test")
    def test_real_command_captures_output_windows(self):
        result = run_command([sys.executable, "-c", "print('hello')"])
        assert result.returncode == 0
        assert "hello" in result.stdout


# ===========================================================================
# Tests: docker_compose_exec
# ===========================================================================


class TestDockerComposeExec:
    def test_dry_run(self):
        result = docker_compose_exec("postgres", ["pg_isready"], dry_run=True)
        assert result.returncode == 0
        assert "[DRY RUN]" in result.stdout

    def test_builds_correct_args(self, monkeypatch):
        captured = {}

        def mock_run(args, **kwargs):
            captured["args"] = args
            return subprocess.CompletedProcess(args, 0, "", "")

        import subprocess
        monkeypatch.setattr(subprocess, "run", mock_run)
        docker_compose_exec("postgres", ["pg_isready"], dry_run=False)

        assert "docker" in captured["args"]
        assert "compose" in captured["args"]
        assert "exec" in captured["args"]
        assert "postgres" in captured["args"]
        assert "pg_isready" in captured["args"]

    def test_custom_compose_files(self, monkeypatch):
        captured = {}

        def mock_run(args, **kwargs):
            captured["args"] = args
            return subprocess.CompletedProcess(args, 0, "", "")

        import subprocess
        monkeypatch.setattr(subprocess, "run", mock_run)
        docker_compose_exec(
            "redis",
            ["redis-cli", "ping"],
            compose_files=["file1.yml", "file2.yml"],
            dry_run=False,
        )

        assert "-f" in captured["args"]
        assert "file1.yml" in captured["args"]
        assert "file2.yml" in captured["args"]


# ===========================================================================
# Tests: docker_compose_action
# ===========================================================================


class TestDockerComposeAction:
    def test_dry_run_restart(self):
        result = docker_compose_action("restart", "postgres", dry_run=True)
        assert result.returncode == 0

    def test_up_action_adds_dash_d(self, monkeypatch):
        captured = {}

        def mock_run(args, **kwargs):
            captured["args"] = args
            return subprocess.CompletedProcess(args, 0, "", "")

        import subprocess
        monkeypatch.setattr(subprocess, "run", mock_run)
        docker_compose_action("up", "redis", dry_run=False)

        assert "up" in captured["args"]
        assert "-d" in captured["args"]
        assert "redis" in captured["args"]


# ===========================================================================
# Tests: wait_for_healthy
# ===========================================================================


class TestWaitForHealthy:
    def test_dry_run_returns_immediately(self):
        healthy, elapsed = wait_for_healthy(lambda: True, dry_run=True)
        assert healthy is True
        assert elapsed < 2.0

    def test_immediately_healthy(self):
        healthy, elapsed = wait_for_healthy(
            lambda: True, timeout=5.0, interval=0.1
        )
        assert healthy is True

    def test_timeout_on_unhealthy(self):
        healthy, elapsed = wait_for_healthy(
            lambda: False, timeout=1.0, interval=0.2
        )
        assert healthy is False
        assert elapsed >= 1.0

    def test_becomes_healthy_after_retries(self):
        call_count = {"n": 0}

        def check():
            call_count["n"] += 1
            return call_count["n"] >= 3

        healthy, elapsed = wait_for_healthy(
            check, timeout=10.0, interval=0.1
        )
        assert healthy is True
        assert call_count["n"] >= 3

    def test_exception_in_check_retries(self):
        call_count = {"n": 0}

        def check():
            call_count["n"] += 1
            if call_count["n"] < 3:
                raise ConnectionError("not ready")
            return True

        healthy, elapsed = wait_for_healthy(
            check, timeout=10.0, interval=0.1
        )
        assert healthy is True


# ===========================================================================
# Tests: PostgreSQL switchover drill
# ===========================================================================


class TestPostgresSwitchover:
    def test_dry_run_produces_passed_result(self):
        result = drill_postgres_switchover(dry_run=True)
        assert result.drill_type == "postgres-switchover"
        assert result.status == "passed"
        assert result.dry_run is True
        assert result.rto_met is True
        assert result.data_integrity_verified is True

    def test_dry_run_has_all_steps(self):
        result = drill_postgres_switchover(dry_run=True)
        step_names = [s["name"] for s in result.steps]
        assert "pre_flight_health" in step_names
        assert "baseline_row_counts" in step_names
        assert "stop_postgres" in step_names
        assert "verify_failure_detected" in step_names
        assert "restart_postgres" in step_names
        assert "wait_for_healthy" in step_names
        assert "verify_data_integrity" in step_names

    def test_rto_target_matches_constant(self):
        result = drill_postgres_switchover(dry_run=True)
        assert result.rto_target_seconds == RTO_TARGETS["postgres-switchover"]

    def test_serializable(self):
        result = drill_postgres_switchover(dry_run=True)
        j = result.to_json()
        parsed = json.loads(j)
        assert parsed["drill_type"] == "postgres-switchover"

    def test_result_has_pipeline_version(self):
        result = drill_postgres_switchover(dry_run=True)
        assert result.pipeline_version is not None
        assert len(result.pipeline_version) > 0


# ===========================================================================
# Tests: Redis Sentinel drill
# ===========================================================================


class TestRedisSentinel:
    def test_dry_run_produces_passed_result(self):
        result = drill_redis_sentinel(dry_run=True)
        assert result.drill_type == "redis-sentinel"
        assert result.status == "passed"
        assert result.dry_run is True

    def test_dry_run_has_all_steps(self):
        result = drill_redis_sentinel(dry_run=True)
        step_names = [s["name"] for s in result.steps]
        assert "pre_flight_health" in step_names
        assert "write_test_key" in step_names
        assert "stop_redis_master" in step_names
        assert "wait_sentinel_promotion" in step_names
        assert "verify_data_integrity" in step_names
        assert "restart_original_master" in step_names
        assert "cleanup_test_key" in step_names

    def test_rto_target_is_15_seconds(self):
        result = drill_redis_sentinel(dry_run=True)
        assert result.rto_target_seconds == 15.0


# ===========================================================================
# Tests: RabbitMQ node failure drill
# ===========================================================================


class TestRabbitMQNodeFailure:
    def test_dry_run_produces_passed_result(self):
        result = drill_rabbitmq_node_failure(dry_run=True)
        assert result.drill_type == "rabbitmq-node-failure"
        assert result.status == "passed"
        assert result.dry_run is True

    def test_dry_run_has_all_steps(self):
        result = drill_rabbitmq_node_failure(dry_run=True)
        step_names = [s["name"] for s in result.steps]
        assert "pre_flight_cluster" in step_names
        assert "record_queue_state" in step_names
        assert "stop_replica_node" in step_names
        assert "verify_quorum_maintained" in step_names
        assert "verify_queue_integrity" in step_names
        assert "restart_failed_node" in step_names
        assert "verify_cluster_reformed" in step_names

    def test_rto_target_is_30_seconds(self):
        result = drill_rabbitmq_node_failure(dry_run=True)
        assert result.rto_target_seconds == 30.0

    def test_data_integrity_verified(self):
        result = drill_rabbitmq_node_failure(dry_run=True)
        assert result.data_integrity_verified is True


# ===========================================================================
# Tests: Worker crash drill
# ===========================================================================


class TestWorkerCrash:
    def test_dry_run_produces_passed_result(self):
        result = drill_worker_crash(dry_run=True)
        assert result.drill_type == "worker-crash"
        assert result.status == "passed"
        assert result.dry_run is True

    def test_dry_run_has_all_steps(self):
        result = drill_worker_crash(dry_run=True)
        step_names = [s["name"] for s in result.steps]
        assert "verify_fleet" in step_names
        assert "baseline_job_count" in step_names
        assert "submit_test_job" in step_names
        assert "wait_for_processing" in step_names
        assert "kill_worker" in step_names
        assert "verify_requeue" in step_names
        assert "verify_worker_restart" in step_names
        assert "validate_page_resume" in step_names

    def test_rto_target_is_60_seconds(self):
        result = drill_worker_crash(dry_run=True)
        assert result.rto_target_seconds == 60.0


# ===========================================================================
# Tests: Helper functions
# ===========================================================================


class TestHelperFunctions:
    def test_get_pg_row_counts_dry_run(self):
        counts = _get_pg_row_counts(dry_run=True)
        assert counts is not None
        assert "jobs_job" in counts
        assert "jobs_worker" in counts
        assert "jobs_pageresult" in counts
        assert all(isinstance(v, int) for v in counts.values())

    def test_get_pg_scalar_dry_run(self):
        value = _get_pg_scalar("SELECT count(*) FROM jobs_job;", dry_run=True)
        assert value == 150

    def test_get_rabbitmq_queues_dry_run(self):
        queues = _get_rabbitmq_queues(dry_run=True)
        assert queues is not None
        assert len(queues) == 3
        assert any(q["name"] == "ocr_gpu" for q in queues)
        assert any(q["type"] == "quorum" for q in queues)

    def test_validate_no_duplicate_pages_no_job_id(self):
        # None job_id should return True (nothing to check)
        assert _validate_no_duplicate_pages(None) is True

    def test_new_drill_result_populates_metadata(self):
        result = _new_drill_result("postgres-switchover", dry_run=True)
        assert result.drill_type == "postgres-switchover"
        assert result.dry_run is True
        assert result.rto_target_seconds == 300.0
        assert len(result.drill_id) == 12
        assert result.timestamp is not None

    def test_new_drill_result_unknown_type(self):
        result = _new_drill_result("unknown-drill", dry_run=True)
        assert result.rto_target_seconds == 60.0  # Default fallback


# ===========================================================================
# Tests: Result persistence
# ===========================================================================


class TestResultPersistence:
    def test_save_creates_file(self, sample_result, results_dir):
        filepath = save_drill_result(sample_result, results_dir=results_dir)
        assert os.path.exists(filepath)
        assert filepath.name.startswith("drill-postgres-switchover-")

    def test_save_valid_json(self, sample_result, results_dir):
        filepath = save_drill_result(sample_result, results_dir=results_dir)
        with open(filepath) as f:
            data = json.load(f)
        assert data["drill_id"] == "test123"
        assert data["status"] == "passed"

    def test_save_creates_directory(self, tmp_path, sample_result):
        new_dir = str(tmp_path / "new" / "nested" / "dir")
        filepath = save_drill_result(sample_result, results_dir=new_dir)
        assert os.path.exists(filepath)

    def test_load_returns_saved_results(self, sample_result, results_dir):
        save_drill_result(sample_result, results_dir=results_dir)
        loaded = load_drill_results(results_dir=results_dir)
        assert len(loaded) == 1
        assert loaded[0]["drill_id"] == "test123"

    def test_load_empty_directory(self, results_dir):
        loaded = load_drill_results(results_dir=results_dir)
        assert loaded == []

    def test_load_nonexistent_directory(self, tmp_path):
        loaded = load_drill_results(results_dir=str(tmp_path / "nonexistent"))
        assert loaded == []


# ===========================================================================
# Tests: Report generation
# ===========================================================================


class TestReportGeneration:
    def test_report_no_results(self, results_dir):
        report = generate_report(results_dir=results_dir)
        assert report.overall_status == "no_results"
        assert "No drill result files found" in report.summary.get("message", "")

    def test_report_nonexistent_dir(self, tmp_path):
        report = generate_report(results_dir=str(tmp_path / "nonexistent"))
        assert report.overall_status == "no_results"

    def test_report_with_results(self, sample_report):
        report = generate_report(results_dir=sample_report)
        assert report.overall_status == "passed"
        assert report.summary["total_drills"] == 1
        assert report.summary["passed"] == 1
        assert report.summary["failed"] == 0

    def test_report_with_failed_result(self, results_dir):
        failed = DrillResult(
            drill_id="fail1",
            drill_type="redis-sentinel",
            timestamp="2026-03-15T00:00:00+00:00",
            pipeline_version="0.9.0",
            dry_run=False,
            rto_target_seconds=15.0,
            status="failed",
            recovery_time_seconds=25.0,
            rto_met=False,
        )
        save_drill_result(failed, results_dir=results_dir)
        report = generate_report(results_dir=results_dir)
        assert report.overall_status == "failed"
        assert report.summary["failed"] == 1

    def test_report_mixed_results(self, sample_result, results_dir):
        save_drill_result(sample_result, results_dir=results_dir)
        failed = DrillResult(
            drill_id="fail2",
            drill_type="worker-crash",
            timestamp="2026-03-15T01:00:00+00:00",
            pipeline_version="0.9.0",
            dry_run=False,
            rto_target_seconds=60.0,
            status="failed",
        )
        save_drill_result(failed, results_dir=results_dir)
        report = generate_report(results_dir=results_dir)
        assert report.overall_status == "failed"
        assert report.summary["total_drills"] == 2
        assert report.summary["passed"] == 1
        assert report.summary["failed"] == 1

    def test_report_has_pipeline_version(self, sample_report):
        report = generate_report(results_dir=sample_report)
        assert report.pipeline_version is not None

    def test_report_has_timestamp(self, sample_report):
        report = generate_report(results_dir=sample_report)
        assert report.generated_at is not None

    def test_report_serializable(self, sample_report):
        report = generate_report(results_dir=sample_report)
        j = json.dumps(report.to_dict())
        parsed = json.loads(j)
        assert parsed["overall_status"] == "passed"


# ===========================================================================
# Tests: Markdown report rendering
# ===========================================================================


class TestMarkdownReport:
    def test_render_empty_report(self):
        report = DrillReport(
            report_id="r1",
            generated_at="2026-01-01",
            pipeline_version="0.9.0",
            overall_status="no_results",
        )
        md = render_markdown_report(report)
        assert "# Failover Drill Report" in md
        assert "NO_RESULTS" in md

    def test_render_with_drills(self, sample_report):
        report = generate_report(results_dir=sample_report)
        md = render_markdown_report(report)
        assert "postgres-switchover" in md
        assert "PASS" in md
        assert "Recovery Time" in md
        assert "RTO Target" in md

    def test_render_with_errors(self):
        report = DrillReport(
            report_id="r2",
            generated_at="2026-01-01",
            pipeline_version="0.9.0",
            overall_status="failed",
            drills=[{
                "drill_type": "redis-sentinel",
                "status": "failed",
                "recovery_time_seconds": 25.0,
                "rto_target_seconds": 15.0,
                "rto_met": False,
                "data_integrity_verified": False,
                "errors": ["RTO exceeded: 25.0s > 15.0s"],
                "steps": [],
            }],
        )
        md = render_markdown_report(report)
        assert "FAIL" in md
        assert "RTO exceeded" in md

    def test_render_with_steps_table(self, sample_report):
        report = generate_report(results_dir=sample_report)
        md = render_markdown_report(report)
        assert "| Step | Status | Duration |" in md

    def test_render_handles_load_error(self):
        report = DrillReport(
            report_id="r3",
            generated_at="2026-01-01",
            pipeline_version="0.9.0",
            overall_status="failed",
            drills=[{"file": "bad.json", "error": "invalid json"}],
        )
        md = render_markdown_report(report)
        assert "Error loading" in md
        assert "invalid json" in md


# ===========================================================================
# Tests: CLI parser
# ===========================================================================


class TestCLIParser:
    def test_build_parser(self):
        parser = build_parser()
        assert parser is not None

    def test_parse_postgres_switchover(self):
        parser = build_parser()
        args = parser.parse_args(["postgres-switchover", "--dry-run"])
        assert args.command == "postgres-switchover"
        assert args.dry_run is True

    def test_parse_redis_sentinel(self):
        parser = build_parser()
        args = parser.parse_args(["redis-sentinel"])
        assert args.command == "redis-sentinel"
        assert args.dry_run is False

    def test_parse_rabbitmq_node_failure(self):
        parser = build_parser()
        args = parser.parse_args(["rabbitmq-node-failure", "--dry-run"])
        assert args.command == "rabbitmq-node-failure"

    def test_parse_worker_crash(self):
        parser = build_parser()
        args = parser.parse_args(["worker-crash"])
        assert args.command == "worker-crash"

    def test_parse_report(self):
        parser = build_parser()
        args = parser.parse_args(["report", "--format", "json"])
        assert args.command == "report"
        assert args.format == "json"

    def test_parse_results_dir(self):
        parser = build_parser()
        args = parser.parse_args(["postgres-switchover", "--results-dir", "/tmp/drills"])
        assert args.results_dir == "/tmp/drills"

    def test_parse_coordinator_dir(self):
        parser = build_parser()
        args = parser.parse_args(["postgres-switchover", "--coordinator-dir", "/opt/coordinator"])
        assert args.coordinator_dir == "/opt/coordinator"

    def test_no_command_returns_none(self):
        parser = build_parser()
        args = parser.parse_args([])
        assert args.command is None


# ===========================================================================
# Tests: CLI main() entry point
# ===========================================================================


class TestMain:
    def test_no_args_returns_1(self):
        result = main([])
        assert result == 1

    def test_dry_run_postgres_returns_0(self, results_dir):
        result = main(["postgres-switchover", "--dry-run", "--results-dir", results_dir])
        assert result == 0

    def test_dry_run_redis_returns_0(self, results_dir):
        result = main(["redis-sentinel", "--dry-run", "--results-dir", results_dir])
        assert result == 0

    def test_dry_run_rabbitmq_returns_0(self, results_dir):
        result = main(["rabbitmq-node-failure", "--dry-run", "--results-dir", results_dir])
        assert result == 0

    def test_dry_run_worker_returns_0(self, results_dir):
        result = main(["worker-crash", "--dry-run", "--results-dir", results_dir])
        assert result == 0

    def test_report_empty_returns_0(self, results_dir):
        result = main(["report", "--results-dir", results_dir, "--format", "json"])
        assert result == 0

    def test_report_with_output_file(self, results_dir, tmp_path):
        # First generate a drill result
        main(["postgres-switchover", "--dry-run", "--results-dir", results_dir])
        output_file = str(tmp_path / "report")
        result = main([
            "report",
            "--results-dir", results_dir,
            "--format", "both",
            "--output", output_file,
        ])
        assert result == 0
        assert os.path.exists(output_file + ".json")
        assert os.path.exists(output_file + ".md")

    def test_report_json_only_output(self, results_dir, tmp_path):
        main(["redis-sentinel", "--dry-run", "--results-dir", results_dir])
        output_file = str(tmp_path / "report.json")
        result = main([
            "report",
            "--results-dir", results_dir,
            "--format", "json",
            "--output", output_file,
        ])
        assert result == 0
        assert os.path.exists(output_file)
        with open(output_file) as f:
            data = json.load(f)
        assert data["overall_status"] == "passed"

    def test_report_markdown_only_output(self, results_dir, tmp_path):
        main(["rabbitmq-node-failure", "--dry-run", "--results-dir", results_dir])
        output_file = str(tmp_path / "report.md")
        result = main([
            "report",
            "--results-dir", results_dir,
            "--format", "markdown",
            "--output", output_file,
        ])
        assert result == 0
        assert os.path.exists(output_file)


# ===========================================================================
# Tests: DRILL_REGISTRY
# ===========================================================================


class TestDrillRegistry:
    def test_all_drills_registered(self):
        assert "postgres-switchover" in DRILL_REGISTRY
        assert "redis-sentinel" in DRILL_REGISTRY
        assert "rabbitmq-node-failure" in DRILL_REGISTRY
        assert "worker-crash" in DRILL_REGISTRY

    def test_registry_callables(self):
        for name, fn in DRILL_REGISTRY.items():
            assert callable(fn), f"{name} is not callable"


# ===========================================================================
# Tests: RTO_TARGETS constants
# ===========================================================================


class TestRTOTargets:
    def test_postgres_rto(self):
        assert RTO_TARGETS["postgres-switchover"] == 300.0

    def test_redis_rto(self):
        assert RTO_TARGETS["redis-sentinel"] == 15.0

    def test_rabbitmq_rto(self):
        assert RTO_TARGETS["rabbitmq-node-failure"] == 30.0

    def test_worker_rto(self):
        assert RTO_TARGETS["worker-crash"] == 60.0


# ===========================================================================
# Tests: End-to-end dry-run drill + report cycle
# ===========================================================================


class TestEndToEnd:
    def test_full_drill_cycle(self, results_dir):
        """Run all 4 drills in dry-run, then generate a combined report."""
        for drill_name in DRILL_REGISTRY:
            ret = main([drill_name, "--dry-run", "--results-dir", results_dir])
            assert ret == 0, f"Drill {drill_name} failed"

        report = generate_report(results_dir=results_dir)
        assert report.overall_status == "passed"
        assert report.summary["total_drills"] == 4
        assert report.summary["passed"] == 4
        assert report.summary["failed"] == 0

        md = render_markdown_report(report)
        assert "postgres-switchover" in md
        assert "redis-sentinel" in md
        assert "rabbitmq-node-failure" in md
        assert "worker-crash" in md

    def test_all_drills_produce_valid_json(self, results_dir):
        """Verify every drill result file is valid JSON."""
        for drill_name in DRILL_REGISTRY:
            main([drill_name, "--dry-run", "--results-dir", results_dir])

        from pathlib import Path
        result_files = list(Path(results_dir).glob("drill-*.json"))
        assert len(result_files) == 4
        for f in result_files:
            with open(f) as fh:
                data = json.load(fh)
            assert "drill_type" in data
            assert "status" in data
            assert data["status"] == "passed"
