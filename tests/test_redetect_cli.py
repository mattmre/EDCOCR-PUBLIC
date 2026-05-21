"""CLI + library tests for :func:`redetect_document` (Plan A -- PR A5).

These tests exercise the ``redetect_document`` library entry point and the
``python -m ocr_local.features.language_detection`` CLI wrapper.  FastText
and ``fitz`` are mocked -- the tests must remain CPU-only and run without
downloading any models.

Run with::

    python -m pytest tests/test_redetect_cli.py -v
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCHEMA_PATH = REPO_ROOT / "schemas" / "language.schema.json"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_fasttext_model() -> MagicMock:
    """FastText stand-in that always returns English at 0.92."""
    m = MagicMock()
    m.predict.return_value = (["__label__en"], [0.92])
    m._sha256 = "a" * 64
    return m


@pytest.fixture
def fake_pdf_doc() -> MagicMock:
    """Context-manager-style fitz document with two pages of English text."""
    page_a = MagicMock()
    page_a.get_text.return_value = (
        "This is the first page of the document.\n"
        "It contains enough English text to exceed the short-span threshold.\n"
    )
    page_b = MagicMock()
    page_b.get_text.return_value = (
        "Second page body text lives here.\n"
        "Again, long enough to trigger FastText rather than the heuristic.\n"
    )

    pdf = MagicMock()
    pdf.page_count = 2
    pdf.load_page.side_effect = lambda idx: (page_a, page_b)[idx]
    pdf.__enter__.return_value = pdf
    pdf.__exit__.return_value = False
    return pdf


@pytest.fixture
def language_model_file(tmp_path: Path) -> Path:
    """Write a dummy FastText model file so path-existence checks pass."""
    model = tmp_path / "lid.176.bin"
    model.write_bytes(b"fake-fasttext-model-bytes")
    return model


@pytest.fixture
def fake_pdf_file(tmp_path: Path) -> Path:
    """Write a dummy PDF file so path-existence checks pass."""
    pdf = tmp_path / "doc.pdf"
    pdf.write_bytes(b"%PDF-1.4\n%EOF\n")
    return pdf


@pytest.fixture
def output_base_dir(tmp_path: Path) -> Path:
    base = tmp_path / "out"
    base.mkdir()
    return base


@pytest.fixture
def patched_fasttext(fake_fasttext_model):
    """Patch ``fasttext.load_model`` to return the mock model."""
    fake_module = SimpleNamespace(load_model=MagicMock(return_value=fake_fasttext_model))
    with patch.dict(sys.modules, {"fasttext": fake_module}):
        yield fake_fasttext_model


@pytest.fixture
def patched_fitz(fake_pdf_doc):
    """Patch ``fitz.open`` to return the mock document."""
    fake_module = SimpleNamespace(open=MagicMock(return_value=fake_pdf_doc))
    with patch.dict(sys.modules, {"fitz": fake_module}):
        yield fake_module


# ---------------------------------------------------------------------------
# Basic success + error contracts
# ---------------------------------------------------------------------------


class TestRedetectDocumentBasics:
    def test_returns_ok_on_success(
        self,
        patched_fasttext,
        patched_fitz,
        fake_pdf_file,
        language_model_file,
        output_base_dir,
    ):
        from ocr_local.features.language_detection import redetect_document

        result = redetect_document(
            pdf_path=str(fake_pdf_file),
            output_json_path="",
            output_base_dir=str(output_base_dir),
            fasttext_model_path=str(language_model_file),
        )
        assert result["status"] == "ok"
        assert "output" in result
        assert result["output"].endswith(".language.json")

    def test_returns_error_on_missing_pdf(
        self, patched_fasttext, patched_fitz, language_model_file, output_base_dir,
    ):
        from ocr_local.features.language_detection import redetect_document

        result = redetect_document(
            pdf_path="/nonexistent/does_not_exist.pdf",
            output_json_path="",
            output_base_dir=str(output_base_dir),
            fasttext_model_path=str(language_model_file),
        )
        assert result["status"] == "error"
        assert "not found" in result["error"].lower()

    def test_returns_error_on_empty_pdf_path(
        self, patched_fasttext, language_model_file, output_base_dir,
    ):
        from ocr_local.features.language_detection import redetect_document

        result = redetect_document(
            pdf_path="",
            output_json_path="",
            output_base_dir=str(output_base_dir),
            fasttext_model_path=str(language_model_file),
        )
        assert result["status"] == "error"

    def test_returns_error_on_none_pdf_path(
        self, patched_fasttext, language_model_file, output_base_dir,
    ):
        from ocr_local.features.language_detection import redetect_document

        result = redetect_document(
            pdf_path=None,  # type: ignore[arg-type]
            output_json_path="",
            output_base_dir=str(output_base_dir),
            fasttext_model_path=str(language_model_file),
        )
        assert result["status"] == "error"

    def test_returns_error_when_fasttext_model_missing(
        self, patched_fitz, fake_pdf_file, output_base_dir,
    ):
        from ocr_local.features.language_detection import redetect_document

        result = redetect_document(
            pdf_path=str(fake_pdf_file),
            output_json_path="",
            output_base_dir=str(output_base_dir),
            fasttext_model_path="/nonexistent/lid.176.bin",
        )
        assert result["status"] == "error"
        assert "fasttext" in result["error"].lower()

    def test_returns_error_when_fasttext_load_fails(
        self, patched_fitz, fake_pdf_file, language_model_file, output_base_dir,
    ):
        from ocr_local.features.language_detection import redetect_document

        fake_module = SimpleNamespace(load_model=MagicMock(side_effect=RuntimeError("boom")))
        with patch.dict(sys.modules, {"fasttext": fake_module}):
            result = redetect_document(
                pdf_path=str(fake_pdf_file),
                output_json_path="",
                output_base_dir=str(output_base_dir),
                fasttext_model_path=str(language_model_file),
            )
        assert result["status"] == "error"

    def test_returns_error_when_fitz_open_fails(
        self, patched_fasttext, fake_pdf_file, language_model_file, output_base_dir,
    ):
        from ocr_local.features.language_detection import redetect_document

        fake_module = SimpleNamespace(open=MagicMock(side_effect=RuntimeError("cannot open")))
        with patch.dict(sys.modules, {"fitz": fake_module}):
            result = redetect_document(
                pdf_path=str(fake_pdf_file),
                output_json_path="",
                output_base_dir=str(output_base_dir),
                fasttext_model_path=str(language_model_file),
            )
        assert result["status"] == "error"

    def test_returns_dict_not_exception(
        self, fake_pdf_file, output_base_dir,
    ):
        """Even with everything broken, redetect returns a dict -- never raises."""
        from ocr_local.features.language_detection import redetect_document

        # Nothing patched: fasttext may or may not be installed, model is
        # definitely missing.  Function must still return a dict.
        result = redetect_document(
            pdf_path=str(fake_pdf_file),
            output_json_path="",
            output_base_dir=str(output_base_dir),
            fasttext_model_path="/definitely/missing/lid.176.bin",
        )
        assert isinstance(result, dict)
        assert result.get("status") == "error"


# ---------------------------------------------------------------------------
# Output file layout + auto-derivation
# ---------------------------------------------------------------------------


class TestRedetectOutputPaths:
    def test_auto_derived_output_path_under_export_language(
        self,
        patched_fasttext,
        patched_fitz,
        fake_pdf_file,
        language_model_file,
        output_base_dir,
    ):
        from ocr_local.features.language_detection import redetect_document

        result = redetect_document(
            pdf_path=str(fake_pdf_file),
            output_json_path="",
            output_base_dir=str(output_base_dir),
            fasttext_model_path=str(language_model_file),
        )
        assert result["status"] == "ok"
        expected_dir = output_base_dir / "EXPORT" / "LANGUAGE"
        assert str(expected_dir) in result["output"]
        assert result["output"].endswith("doc.language.json")
        assert os.path.exists(result["output"])

    def test_explicit_output_path_is_honoured(
        self,
        patched_fasttext,
        patched_fitz,
        fake_pdf_file,
        language_model_file,
        tmp_path,
    ):
        from ocr_local.features.language_detection import redetect_document

        explicit = tmp_path / "custom" / "redetect.language.json"
        result = redetect_document(
            pdf_path=str(fake_pdf_file),
            output_json_path=str(explicit),
            output_base_dir=str(tmp_path),
            fasttext_model_path=str(language_model_file),
        )
        assert result["status"] == "ok"
        assert result["output"] == str(explicit)
        assert explicit.exists()

    def test_sidecar_is_valid_json(
        self,
        patched_fasttext,
        patched_fitz,
        fake_pdf_file,
        language_model_file,
        output_base_dir,
    ):
        from ocr_local.features.language_detection import redetect_document

        result = redetect_document(
            pdf_path=str(fake_pdf_file),
            output_json_path="",
            output_base_dir=str(output_base_dir),
            fasttext_model_path=str(language_model_file),
        )
        with open(result["output"], encoding="utf-8") as fh:
            payload = json.load(fh)
        assert payload["schema_version"] == "1.0"
        assert payload["document_id"] == "doc"
        assert "processing" in payload
        assert "pages" in payload

    def test_sidecar_passes_schema_validation(
        self,
        patched_fasttext,
        patched_fitz,
        fake_pdf_file,
        language_model_file,
        output_base_dir,
    ):
        import jsonschema

        from ocr_local.features.language_detection import redetect_document

        with open(SCHEMA_PATH, encoding="utf-8") as fh:
            schema = json.load(fh)

        result = redetect_document(
            pdf_path=str(fake_pdf_file),
            output_json_path="",
            output_base_dir=str(output_base_dir),
            fasttext_model_path=str(language_model_file),
        )
        with open(result["output"], encoding="utf-8") as fh:
            payload = json.load(fh)

        jsonschema.validate(instance=payload, schema=schema)

    def test_primary_language_propagated_into_result(
        self,
        patched_fasttext,
        patched_fitz,
        fake_pdf_file,
        language_model_file,
        output_base_dir,
    ):
        from ocr_local.features.language_detection import redetect_document

        result = redetect_document(
            pdf_path=str(fake_pdf_file),
            output_json_path="",
            output_base_dir=str(output_base_dir),
            fasttext_model_path=str(language_model_file),
        )
        assert result["primary_language"] == "en"

    def test_page_count_matches_pdf(
        self,
        patched_fasttext,
        patched_fitz,
        fake_pdf_file,
        language_model_file,
        output_base_dir,
    ):
        from ocr_local.features.language_detection import redetect_document

        result = redetect_document(
            pdf_path=str(fake_pdf_file),
            output_json_path="",
            output_base_dir=str(output_base_dir),
            fasttext_model_path=str(language_model_file),
        )
        assert result["page_count"] == 2

    def test_processing_metadata_has_fasttext_sha(
        self,
        patched_fasttext,
        patched_fitz,
        fake_pdf_file,
        language_model_file,
        output_base_dir,
    ):
        from ocr_local.features.language_detection import redetect_document

        result = redetect_document(
            pdf_path=str(fake_pdf_file),
            output_json_path="",
            output_base_dir=str(output_base_dir),
            fasttext_model_path=str(language_model_file),
        )
        with open(result["output"], encoding="utf-8") as fh:
            payload = json.load(fh)
        assert len(payload["processing"]["fasttext_model_sha256"]) == 64

    def test_idempotent_primary_language(
        self,
        patched_fasttext,
        patched_fitz,
        fake_pdf_file,
        language_model_file,
        output_base_dir,
    ):
        """Two back-to-back runs on the same PDF yield the same primary_language."""
        from ocr_local.features.language_detection import redetect_document

        r1 = redetect_document(
            pdf_path=str(fake_pdf_file),
            output_json_path="",
            output_base_dir=str(output_base_dir),
            fasttext_model_path=str(language_model_file),
        )
        r2 = redetect_document(
            pdf_path=str(fake_pdf_file),
            output_json_path="",
            output_base_dir=str(output_base_dir),
            fasttext_model_path=str(language_model_file),
        )
        assert r1["primary_language"] == r2["primary_language"]
        assert r1["page_count"] == r2["page_count"]


# ---------------------------------------------------------------------------
# Config and include_spans handling
# ---------------------------------------------------------------------------


class TestRedetectWithConfig:
    def test_config_none_uses_defaults(
        self,
        patched_fasttext,
        patched_fitz,
        fake_pdf_file,
        language_model_file,
        output_base_dir,
    ):
        from ocr_local.features.language_detection import redetect_document

        result = redetect_document(
            pdf_path=str(fake_pdf_file),
            output_json_path="",
            output_base_dir=str(output_base_dir),
            fasttext_model_path=str(language_model_file),
            config=None,
        )
        assert result["status"] == "ok"

    def test_config_include_spans_false_strips_spans(
        self,
        patched_fasttext,
        patched_fitz,
        fake_pdf_file,
        language_model_file,
        output_base_dir,
    ):
        from ocr_local.features.language_detection import redetect_document

        cfg = SimpleNamespace(
            language_short_span_threshold=20,
            language_confidence_threshold=0.4,
            language_include_spans=False,
        )
        result = redetect_document(
            pdf_path=str(fake_pdf_file),
            output_json_path="",
            output_base_dir=str(output_base_dir),
            fasttext_model_path=str(language_model_file),
            config=cfg,
        )
        with open(result["output"], encoding="utf-8") as fh:
            payload = json.load(fh)
        for page in payload["pages"]:
            assert page["spans"] == []

    def test_config_include_spans_true_retains_spans(
        self,
        patched_fasttext,
        patched_fitz,
        fake_pdf_file,
        language_model_file,
        output_base_dir,
    ):
        from ocr_local.features.language_detection import redetect_document

        cfg = SimpleNamespace(
            language_short_span_threshold=20,
            language_confidence_threshold=0.4,
            language_include_spans=True,
        )
        result = redetect_document(
            pdf_path=str(fake_pdf_file),
            output_json_path="",
            output_base_dir=str(output_base_dir),
            fasttext_model_path=str(language_model_file),
            config=cfg,
        )
        with open(result["output"], encoding="utf-8") as fh:
            payload = json.load(fh)
        total_spans = sum(len(p.get("spans", [])) for p in payload["pages"])
        assert total_spans > 0

    def test_config_threshold_overrides_are_consumed(
        self,
        patched_fasttext,
        patched_fitz,
        fake_pdf_file,
        language_model_file,
        output_base_dir,
    ):
        """Custom thresholds do not crash redetect."""
        from ocr_local.features.language_detection import redetect_document

        cfg = SimpleNamespace(
            language_short_span_threshold=5,
            language_confidence_threshold=0.9,
            language_include_spans=False,
        )
        result = redetect_document(
            pdf_path=str(fake_pdf_file),
            output_json_path="",
            output_base_dir=str(output_base_dir),
            fasttext_model_path=str(language_model_file),
            config=cfg,
        )
        assert result["status"] == "ok"


# ---------------------------------------------------------------------------
# Custody event emission
# ---------------------------------------------------------------------------


class TestRedetectCustody:
    def test_language_redetected_event_attempted(
        self,
        patched_fasttext,
        patched_fitz,
        fake_pdf_file,
        language_model_file,
        output_base_dir,
    ):
        """A custody event should be attempted after a successful redetect."""
        import ocr_local.features.custody as custody_mod  # noqa: F401 -- ensure loaded
        from ocr_local.features import language_detection as ld

        mock_chain = MagicMock()
        mock_chain_class = MagicMock(return_value=mock_chain)

        with patch.object(custody_mod, "CustodyChain", mock_chain_class):
            result = ld.redetect_document(
                pdf_path=str(fake_pdf_file),
                output_json_path="",
                output_base_dir=str(output_base_dir),
                fasttext_model_path=str(language_model_file),
            )
        assert result["status"] == "ok"
        assert mock_chain_class.called
        # append_event was called at least once with LANGUAGE_REDETECTED
        event_types = [
            call.args[0] for call in mock_chain.append_event.call_args_list
        ]
        assert "LANGUAGE_REDETECTED" in event_types

    def test_custody_failure_does_not_fail_redetect(
        self,
        patched_fasttext,
        patched_fitz,
        fake_pdf_file,
        language_model_file,
        output_base_dir,
    ):
        """A broken custody module must not block re-detection."""
        import ocr_local.features.custody as custody_mod  # noqa: F401
        from ocr_local.features import language_detection as ld

        failing = MagicMock(side_effect=RuntimeError("custody exploded"))
        with patch.object(custody_mod, "CustodyChain", failing):
            result = ld.redetect_document(
                pdf_path=str(fake_pdf_file),
                output_json_path="",
                output_base_dir=str(output_base_dir),
                fasttext_model_path=str(language_model_file),
            )
        assert result["status"] == "ok"

    def test_custody_payload_includes_primary_language(
        self,
        patched_fasttext,
        patched_fitz,
        fake_pdf_file,
        language_model_file,
        output_base_dir,
    ):
        import ocr_local.features.custody as custody_mod  # noqa: F401
        from ocr_local.features import language_detection as ld

        mock_chain = MagicMock()
        mock_chain_class = MagicMock(return_value=mock_chain)

        with patch.object(custody_mod, "CustodyChain", mock_chain_class):
            result = ld.redetect_document(
                pdf_path=str(fake_pdf_file),
                output_json_path="",
                output_base_dir=str(output_base_dir),
                fasttext_model_path=str(language_model_file),
            )
        assert result["status"] == "ok"
        redetect_calls = [
            c for c in mock_chain.append_event.call_args_list
            if c.args[0] == "LANGUAGE_REDETECTED"
        ]
        assert redetect_calls
        payload = redetect_calls[0].args[1]
        assert payload["primary_language"] == "en"
        assert "page_count" in payload
        assert "fasttext_model_sha256" in payload


# ---------------------------------------------------------------------------
# CLI wrapper tests
# ---------------------------------------------------------------------------


class TestRedetectCLI:
    def test_doc_argument_required(self):
        from ocr_local.features.language_detection import _cli_main

        with pytest.raises(SystemExit):
            _cli_main([])

    def test_help_flag_exits_zero(self):
        from ocr_local.features.language_detection import _cli_main

        with pytest.raises(SystemExit) as excinfo:
            _cli_main(["--help"])
        assert excinfo.value.code == 0

    def test_cli_returns_zero_on_success(
        self,
        patched_fasttext,
        patched_fitz,
        fake_pdf_file,
        language_model_file,
        output_base_dir,
        capsys,
    ):
        from ocr_local.features.language_detection import _cli_main

        rc = _cli_main([
            "--doc", str(fake_pdf_file),
            "--output-base-dir", str(output_base_dir),
            "--fasttext-model", str(language_model_file),
        ])
        assert rc == 0
        captured = capsys.readouterr()
        printed = json.loads(captured.out)
        assert printed["status"] == "ok"

    def test_cli_returns_nonzero_on_missing_pdf(
        self, patched_fasttext, language_model_file, output_base_dir, capsys,
    ):
        from ocr_local.features.language_detection import _cli_main

        rc = _cli_main([
            "--doc", "/nope/missing.pdf",
            "--output-base-dir", str(output_base_dir),
            "--fasttext-model", str(language_model_file),
        ])
        assert rc == 1
        captured = capsys.readouterr()
        assert json.loads(captured.out)["status"] == "error"

    def test_cli_returns_nonzero_when_model_missing(
        self, patched_fitz, fake_pdf_file, output_base_dir, capsys,
    ):
        from ocr_local.features.language_detection import _cli_main

        rc = _cli_main([
            "--doc", str(fake_pdf_file),
            "--output-base-dir", str(output_base_dir),
            "--fasttext-model", "/nope/lid.176.bin",
        ])
        assert rc == 1

    def test_cli_writes_to_explicit_output(
        self,
        patched_fasttext,
        patched_fitz,
        fake_pdf_file,
        language_model_file,
        tmp_path,
    ):
        from ocr_local.features.language_detection import _cli_main

        explicit = tmp_path / "nested" / "out.language.json"
        rc = _cli_main([
            "--doc", str(fake_pdf_file),
            "--output", str(explicit),
            "--output-base-dir", str(tmp_path),
            "--fasttext-model", str(language_model_file),
        ])
        assert rc == 0
        assert explicit.exists()

    def test_cli_prints_json_output(
        self,
        patched_fasttext,
        patched_fitz,
        fake_pdf_file,
        language_model_file,
        output_base_dir,
        capsys,
    ):
        from ocr_local.features.language_detection import _cli_main

        _cli_main([
            "--doc", str(fake_pdf_file),
            "--output-base-dir", str(output_base_dir),
            "--fasttext-model", str(language_model_file),
        ])
        captured = capsys.readouterr()
        # Output should be JSON-parseable
        payload = json.loads(captured.out)
        assert "status" in payload


# ---------------------------------------------------------------------------
# Public surface sanity checks
# ---------------------------------------------------------------------------


class TestPublicSurface:
    def test_redetect_document_in_all(self):
        from ocr_local.features import language_detection as ld
        assert "redetect_document" in ld.__all__

    def test_redetect_document_is_callable(self):
        from ocr_local.features.language_detection import redetect_document
        assert callable(redetect_document)

    def test_cli_main_is_callable(self):
        from ocr_local.features.language_detection import _cli_main
        assert callable(_cli_main)

    def test_redetect_document_signature_accepts_config_kwarg(self):
        import inspect

        from ocr_local.features.language_detection import redetect_document

        sig = inspect.signature(redetect_document)
        assert "config" in sig.parameters
        assert "pdf_path" in sig.parameters
        assert "output_json_path" in sig.parameters
        assert "output_base_dir" in sig.parameters
        assert "fasttext_model_path" in sig.parameters
