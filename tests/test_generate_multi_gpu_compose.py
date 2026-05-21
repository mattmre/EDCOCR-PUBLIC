"""Tests for scripts/generate_multi_gpu_compose.py.

Validates the VRAM-based concurrency heuristic, multi-GPU YAML generation,
and CLI argument handling for the multi-GPU Docker Compose generator.
"""

import importlib
import os
import sys

# ---------------------------------------------------------------------------
# Import the script under test via importlib (it lives in scripts/)
# ---------------------------------------------------------------------------
_SCRIPTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts")
sys.path.insert(0, _SCRIPTS_DIR)

_mod = importlib.import_module("generate_multi_gpu_compose")
recommended_concurrency = _mod.recommended_concurrency
generate_compose_yaml = _mod.generate_compose_yaml
main = _mod.main


# ===========================================================================
# recommended_concurrency() tests
# ===========================================================================


class TestRecommendedConcurrency:
    """Validate the VRAM-to-concurrency heuristic (4GB per task)."""

    def test_recommended_concurrency_8gb(self):
        """8192 MB VRAM should yield concurrency of 2."""
        assert recommended_concurrency(8192) == 2

    def test_recommended_concurrency_16gb(self):
        """16384 MB VRAM should yield concurrency of 4."""
        assert recommended_concurrency(16384) == 4

    def test_recommended_concurrency_24gb(self):
        """24576 MB VRAM should yield concurrency of 6."""
        assert recommended_concurrency(24576) == 6

    def test_recommended_concurrency_48gb(self):
        """49152 MB VRAM should yield concurrency of 12."""
        assert recommended_concurrency(49152) == 12

    def test_recommended_concurrency_zero(self):
        """0 MB VRAM should return safe default of 4."""
        assert recommended_concurrency(0) == 4

    def test_recommended_concurrency_negative(self):
        """Negative VRAM should return safe default of 4."""
        assert recommended_concurrency(-1) == 4

    def test_recommended_concurrency_small(self):
        """2048 MB VRAM should clamp to minimum of 1."""
        # 2048 // 4096 == 0, but clamped to max(1, ...) == 1 -- except
        # the formula: max(1, min(2048 // 4096, 24)) == max(1, 0) == 1
        # However 2048 > 0 so it does not hit the safe-default branch.
        assert recommended_concurrency(2048) == 1


# ===========================================================================
# generate_compose_yaml() tests
# ===========================================================================


class TestGenerateComposeYaml:
    """Validate generated Docker Compose YAML content."""

    def test_generate_single_gpu(self):
        """--gpus 1 should generate exactly 1 service with CUDA_VISIBLE_DEVICES=0."""
        yaml_content = generate_compose_yaml(
            gpu_count=1,
            concurrency=4,
            concurrency_source="test",
            queues="ocr_gpu,cpu_general",
            max_tasks_per_child=50,
            output_filename="test.yml",
        )
        assert "ocr-worker-gpu0:" in yaml_content
        assert 'CUDA_VISIBLE_DEVICES: "0"' in yaml_content
        # Should NOT contain gpu1
        assert "ocr-worker-gpu1:" not in yaml_content

    def test_generate_four_gpus(self):
        """--gpus 4 should generate 4 services (gpu0 through gpu3)."""
        yaml_content = generate_compose_yaml(
            gpu_count=4,
            concurrency=6,
            concurrency_source="test",
            queues="ocr_gpu",
            max_tasks_per_child=50,
            output_filename="test.yml",
        )
        for idx in range(4):
            assert f"ocr-worker-gpu{idx}:" in yaml_content
            assert f'CUDA_VISIBLE_DEVICES: "{idx}"' in yaml_content
        # Should NOT contain gpu4
        assert "ocr-worker-gpu4:" not in yaml_content

    def test_gpu_service_names(self):
        """Generated YAML must contain correctly named service blocks."""
        yaml_content = generate_compose_yaml(
            gpu_count=3,
            concurrency=4,
            concurrency_source="test",
            queues="ocr_gpu,cpu_general",
            max_tasks_per_child=50,
            output_filename="test.yml",
        )
        assert "ocr-worker-gpu0:" in yaml_content
        assert "ocr-worker-gpu1:" in yaml_content
        assert "ocr-worker-gpu2:" in yaml_content

    def test_explicit_concurrency(self):
        """Explicit concurrency value should appear in generated YAML."""
        yaml_content = generate_compose_yaml(
            gpu_count=1,
            concurrency=8,
            concurrency_source="user-specified",
            queues="ocr_gpu,cpu_general",
            max_tasks_per_child=50,
            output_filename="test.yml",
        )
        # The concurrency value appears in WORKER_CONCURRENCY defaults and
        # in the celery -c flag
        assert "-c ${WORKER_CONCURRENCY:-8}" in yaml_content
        assert "WORKER_CONCURRENCY: ${WORKER_CONCURRENCY:-8}" in yaml_content

    def test_custom_queues(self):
        """Custom queue names should appear in generated YAML."""
        yaml_content = generate_compose_yaml(
            gpu_count=1,
            concurrency=4,
            concurrency_source="test",
            queues="custom_queue",
            max_tasks_per_child=50,
            output_filename="test.yml",
        )
        assert "custom_queue" in yaml_content
        assert "WORKER_QUEUES: ${WORKER_QUEUES:-custom_queue}" in yaml_content


# ===========================================================================
# main() CLI integration tests
# ===========================================================================


class TestPerGpuQueues:
    """Validate per-GPU queue generation (--per-gpu-queues flag)."""

    def test_per_gpu_queues_flag_generates_unique_queues(self):
        """With per_gpu_queues=True, each GPU worker gets ocr_gpu_{idx} in WORKER_QUEUES."""
        yaml_content = generate_compose_yaml(
            gpu_count=2,
            concurrency=4,
            concurrency_source="test",
            queues="ocr_gpu,cpu_general",
            max_tasks_per_child=50,
            output_filename="test.yml",
            per_gpu_queues=True,
        )
        # GPU 0 should have ocr_gpu_0, GPU 1 should have ocr_gpu_1
        assert "ocr_gpu_0" in yaml_content
        assert "ocr_gpu_1" in yaml_content

    def test_per_gpu_queues_sets_gpu_index_env(self):
        """With per_gpu_queues=True, each service gets GPU_INDEX env var."""
        yaml_content = generate_compose_yaml(
            gpu_count=2,
            concurrency=4,
            concurrency_source="test",
            queues="ocr_gpu,cpu_general",
            max_tasks_per_child=50,
            output_filename="test.yml",
            per_gpu_queues=True,
        )
        assert 'GPU_INDEX: "0"' in yaml_content
        assert 'GPU_INDEX: "1"' in yaml_content

    def test_per_gpu_queues_sets_enable_flag(self):
        """With per_gpu_queues=True, ENABLE_PER_GPU_QUEUES='true' appears in output."""
        yaml_content = generate_compose_yaml(
            gpu_count=2,
            concurrency=4,
            concurrency_source="test",
            queues="ocr_gpu,cpu_general",
            max_tasks_per_child=50,
            output_filename="test.yml",
            per_gpu_queues=True,
        )
        assert 'ENABLE_PER_GPU_QUEUES: "true"' in yaml_content

    def test_per_gpu_queues_sets_gpu_count(self):
        """With per_gpu_queues=True and gpus=2, GPU_COUNT='2' appears in output."""
        yaml_content = generate_compose_yaml(
            gpu_count=2,
            concurrency=4,
            concurrency_source="test",
            queues="ocr_gpu,cpu_general",
            max_tasks_per_child=50,
            output_filename="test.yml",
            per_gpu_queues=True,
        )
        assert 'GPU_COUNT: "2"' in yaml_content

    def test_default_no_per_gpu_queues(self):
        """Without per_gpu_queues flag, ENABLE_PER_GPU_QUEUES should not appear."""
        yaml_content = generate_compose_yaml(
            gpu_count=2,
            concurrency=4,
            concurrency_source="test",
            queues="ocr_gpu,cpu_general",
            max_tasks_per_child=50,
            output_filename="test.yml",
        )
        assert "ENABLE_PER_GPU_QUEUES" not in yaml_content

    def test_per_gpu_queues_four_gpus(self):
        """With 4 GPUs, each gets its own ocr_gpu_{idx} queue."""
        yaml_content = generate_compose_yaml(
            gpu_count=4,
            concurrency=4,
            concurrency_source="test",
            queues="ocr_gpu,cpu_general",
            max_tasks_per_child=50,
            output_filename="test.yml",
            per_gpu_queues=True,
        )
        for idx in range(4):
            assert f"ocr_gpu_{idx}" in yaml_content
            assert f'GPU_INDEX: "{idx}"' in yaml_content
        assert 'GPU_COUNT: "4"' in yaml_content

    def test_per_gpu_queues_preserves_cpu_general(self):
        """Per-GPU queues only replace ocr_gpu; cpu_general stays intact."""
        yaml_content = generate_compose_yaml(
            gpu_count=1,
            concurrency=4,
            concurrency_source="test",
            queues="ocr_gpu,cpu_general",
            max_tasks_per_child=50,
            output_filename="test.yml",
            per_gpu_queues=True,
        )
        # ocr_gpu should be replaced with ocr_gpu_0 in the queues
        assert "ocr_gpu_0,cpu_general" in yaml_content


# ===========================================================================
# main() CLI integration tests
# ===========================================================================


class TestMainCli:
    """Validate the main() CLI entry point."""

    def test_main_returns_zero(self, capsys):
        """main(["--gpus", "2"]) should succeed and return 0."""
        result = main(["--gpus", "2", "--concurrency", "4"])
        assert result == 0
        # Verify output was generated to stdout
        captured = capsys.readouterr()
        assert "ocr-worker-gpu0:" in captured.out
        assert "ocr-worker-gpu1:" in captured.out

    def test_invalid_gpus_zero(self, capsys):
        """main(["--gpus", "0"]) should fail and return 1."""
        result = main(["--gpus", "0"])
        assert result == 1
        captured = capsys.readouterr()
        assert "ERROR" in captured.err

    def test_cli_per_gpu_queues_flag(self, capsys):
        """main() with --per-gpu-queues flag should succeed and return 0."""
        result = main(["--gpus", "2", "--per-gpu-queues", "--concurrency", "4"])
        assert result == 0
        captured = capsys.readouterr()
        assert "ocr_gpu_0" in captured.out
        assert "ocr_gpu_1" in captured.out
        assert 'ENABLE_PER_GPU_QUEUES: "true"' in captured.out
