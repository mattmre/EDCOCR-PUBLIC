"""Tests for layoutlm_finetune — LayoutLMv3 fine-tuning pipeline.

Covers FineTuneConfig, CLI argument parsing, and run_finetuning with
mocked ML dependencies. All tests run WITHOUT torch/transformers/peft
installed.

Run with: python -m pytest tests/test_layoutlm_finetune.py -v
"""

from dataclasses import asdict
from unittest.mock import MagicMock, patch

import pytest

from layoutlm_finetune import DEFAULT_BASE_MODEL, FineTuneConfig, _parse_args

# ---------------------------------------------------------------------------
# FineTuneConfig tests
# ---------------------------------------------------------------------------


class TestFineTuneConfig:
    """Tests for the FineTuneConfig dataclass."""

    def test_default_values(self):
        """All default values are set correctly."""
        cfg = FineTuneConfig()
        assert cfg.dataset == "custom"
        assert cfg.data_dir == "./data"
        assert cfg.output_dir == "./models/out"
        assert cfg.label_set == "default"
        assert cfg.base_model == DEFAULT_BASE_MODEL
        assert cfg.use_lora is False
        assert cfg.lora_rank == 16
        assert cfg.lora_alpha == 32
        assert cfg.epochs == 50
        assert cfg.batch_size == 2
        assert cfg.learning_rate == 5e-5
        assert cfg.test_size == 0.2
        assert cfg.seed == 42

    def test_custom_values(self):
        """Custom values are stored correctly."""
        cfg = FineTuneConfig(
            label_set="forensic",
            use_lora=True,
            epochs=10,
            learning_rate=1e-4,
        )
        assert cfg.label_set == "forensic"
        assert cfg.use_lora is True
        assert cfg.epochs == 10
        assert cfg.learning_rate == 1e-4

    def test_serializable(self):
        """Config can be serialized to dict via asdict."""
        cfg = FineTuneConfig()
        d = asdict(cfg)
        assert isinstance(d, dict)
        assert d["dataset"] == "custom"
        assert d["use_lora"] is False


# ---------------------------------------------------------------------------
# CLI argument parsing tests
# ---------------------------------------------------------------------------


class TestParseArgs:
    """Tests for _parse_args CLI parser."""

    def test_defaults(self):
        """No arguments produces default config."""
        cfg = _parse_args([])
        assert cfg.dataset == "custom"
        assert cfg.epochs == 50
        assert cfg.use_lora is False

    def test_label_set_arg(self):
        """--label-set is parsed correctly."""
        cfg = _parse_args(["--label-set", "forensic"])
        assert cfg.label_set == "forensic"

    def test_lora_flags(self):
        """--use-lora, --lora-rank, --lora-alpha are parsed."""
        cfg = _parse_args([
            "--use-lora",
            "--lora-rank", "8",
            "--lora-alpha", "16",
        ])
        assert cfg.use_lora is True
        assert cfg.lora_rank == 8
        assert cfg.lora_alpha == 16

    def test_numeric_args(self):
        """Numeric arguments are correctly typed."""
        cfg = _parse_args([
            "--epochs", "100",
            "--batch-size", "4",
            "--learning-rate", "1e-3",
            "--test-size", "0.1",
            "--seed", "123",
        ])
        assert cfg.epochs == 100
        assert cfg.batch_size == 4
        assert cfg.learning_rate == 1e-3
        assert cfg.test_size == 0.1
        assert cfg.seed == 123

    def test_data_dir_and_output_dir(self):
        """--data-dir and --output-dir are parsed."""
        cfg = _parse_args(["--data-dir", "/my/data", "--output-dir", "/my/out"])
        assert cfg.data_dir == "/my/data"
        assert cfg.output_dir == "/my/out"


# ---------------------------------------------------------------------------
# run_finetuning tests (mocked ML)
# ---------------------------------------------------------------------------


class TestRunFinetuning:
    """Tests for run_finetuning with mocked torch/transformers."""

    def test_import_error_without_torch(self, tmp_path):
        """ImportError is raised when torch is not installed."""
        cfg = FineTuneConfig(data_dir=str(tmp_path), output_dir=str(tmp_path / "out"))
        with patch.dict("sys.modules", {"torch": None}):
            from layoutlm_finetune import run_finetuning
            with pytest.raises(ImportError, match="torch"):
                run_finetuning(cfg)

    def test_import_error_without_transformers(self, tmp_path):
        """ImportError is raised when transformers is not installed."""
        cfg = FineTuneConfig(data_dir=str(tmp_path), output_dir=str(tmp_path / "out"))
        mock_torch = MagicMock()
        with patch.dict("sys.modules", {
            "torch": mock_torch,
            "transformers": None,
        }):
            from layoutlm_finetune import run_finetuning
            with pytest.raises(ImportError, match="transformers"):
                run_finetuning(cfg)

    def test_data_dir_not_found(self, tmp_path):
        """FileNotFoundError when data_dir does not exist."""
        cfg = FineTuneConfig(
            data_dir=str(tmp_path / "nonexistent"),
            output_dir=str(tmp_path / "out"),
        )
        # Mock the heavy imports so we get past them to the file check
        mock_torch = MagicMock()
        mock_transformers = MagicMock()
        mock_layoutlm_data = MagicMock()
        with patch.dict("sys.modules", {
            "torch": mock_torch,
            "transformers": mock_transformers,
            "layoutlm_data": mock_layoutlm_data,
        }):
            from layoutlm_finetune import run_finetuning
            with pytest.raises(FileNotFoundError, match="Data directory"):
                run_finetuning(cfg)

    def test_no_jsonl_files(self, tmp_path):
        """FileNotFoundError when data_dir has no .jsonl files."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        cfg = FineTuneConfig(
            data_dir=str(data_dir),
            output_dir=str(tmp_path / "out"),
        )
        mock_torch = MagicMock()
        mock_transformers = MagicMock()
        mock_layoutlm_data = MagicMock()
        with patch.dict("sys.modules", {
            "torch": mock_torch,
            "transformers": mock_transformers,
            "layoutlm_data": mock_layoutlm_data,
        }):
            from layoutlm_finetune import run_finetuning
            with pytest.raises(FileNotFoundError, match="No .jsonl files"):
                run_finetuning(cfg)


# ---------------------------------------------------------------------------
# Seqeval compute_metrics tests (mocked)
# ---------------------------------------------------------------------------


class TestBuildComputeMetrics:
    """Tests for _build_compute_metrics with mocked seqeval."""

    def test_import_error_without_numpy(self):
        """ImportError when numpy is not installed."""
        from layoutlm_labels import build_label_set
        ls = build_label_set("test", ["DATE"])
        with patch.dict("sys.modules", {"numpy": None}):
            from layoutlm_finetune import _build_compute_metrics
            with pytest.raises(ImportError, match="numpy"):
                _build_compute_metrics(ls)

    def test_import_error_without_seqeval(self):
        """ImportError when seqeval is not installed."""
        from layoutlm_labels import build_label_set
        ls = build_label_set("test", ["DATE"])
        mock_np = MagicMock()
        with patch.dict("sys.modules", {
            "numpy": mock_np,
            "seqeval": None,
            "seqeval.metrics": None,
        }):
            from layoutlm_finetune import _build_compute_metrics
            with pytest.raises(ImportError, match="seqeval"):
                _build_compute_metrics(ls)
