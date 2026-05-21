"""Unit tests for transform registry and contracts.

Tests the transform operation registry, configuration validation, and result models.
Does not require PyMuPDF - uses mock transform implementations.

Run with: python -m pytest tests/test_transform_registry.py -v
"""

import os
from typing import Any

from ocr_distributed.transforms import (
    TransformConfig,
    TransformError,
    TransformOperation,
    TransformRegistry,
    TransformResult,
    TransformValidationError,
    get_transform_registry,
)

# --- Mock Transform Implementations ---


class MockPDFRotateTransform(TransformOperation):
    """Mock transform for testing - rotates PDF pages."""

    def get_metadata(self) -> dict[str, Any]:
        return {
            "name": "pdf_rotate",
            "description": "Rotate PDF pages by specified angle",
            "version": "1.0.0",
            "supported_formats": [".pdf"],
            "output_format": ".pdf",
            "parameters": {
                "angle": {
                    "type": "int",
                    "description": "Rotation angle (90, 180, 270)",
                    "required": True,
                }
            },
        }

    def validate_config(self, config: TransformConfig) -> list[str]:
        errors = []
        angle = config.params.get("angle")
        if angle is None:
            errors.append("angle parameter is required")
        elif angle not in [90, 180, 270, -90, -180, -270]:
            errors.append(f"angle must be 90/180/270, got {angle}")
        return errors

    def execute(
        self, input_path: str, output_path: str, config: TransformConfig
    ) -> TransformResult:
        if not os.path.exists(input_path):
            raise TransformValidationError(f"Input file not found: {input_path}")

        angle = config.params.get("angle", 0)
        # Mock execution
        return TransformResult(
            success=True,
            output_path=output_path,
            metadata={"angle": angle},
            pages_processed=10,
        )


class MockImageConvertTransform(TransformOperation):
    """Mock transform for testing - converts image formats."""

    def get_metadata(self) -> dict[str, Any]:
        return {
            "name": "image_convert",
            "description": "Convert image format",
            "version": "1.0.0",
            "supported_formats": [".jpg", ".png", ".tif", ".bmp"],
            "output_format": ".png",
            "parameters": {
                "format": {
                    "type": "str",
                    "description": "Target format",
                    "required": True,
                }
            },
        }

    def validate_config(self, config: TransformConfig) -> list[str]:
        errors = []
        fmt = config.params.get("format")
        if not fmt:
            errors.append("format parameter is required")
        elif fmt not in ["png", "jpg", "tif"]:
            errors.append(f"format must be png/jpg/tif, got {fmt}")
        return errors

    def execute(
        self, input_path: str, output_path: str, config: TransformConfig
    ) -> TransformResult:
        return TransformResult(
            success=True,
            output_path=output_path,
            metadata={"format": config.params.get("format")},
            pages_processed=1,
        )


# --- TransformConfig Tests ---


class TestTransformConfig:
    """Tests for TransformConfig dataclass."""

    def test_minimal_config(self):
        config = TransformConfig(operation_name="pdf_rotate")
        assert config.operation_name == "pdf_rotate"
        assert config.params == {}
        assert config.validate_input is True
        assert config.preserve_metadata is True

    def test_full_config(self):
        config = TransformConfig(
            operation_name="pdf_rotate",
            params={"angle": 90},
            validate_input=False,
            preserve_metadata=False,
        )
        assert config.operation_name == "pdf_rotate"
        assert config.params == {"angle": 90}
        assert config.validate_input is False
        assert config.preserve_metadata is False

    def test_empty_operation_name_raises(self):
        try:
            TransformConfig(operation_name="")
            assert False, "Should raise TransformValidationError"
        except TransformValidationError as e:
            assert "operation_name is required" in str(e)

    def test_config_is_mutable(self):
        # Dataclasses are mutable by default
        config = TransformConfig(operation_name="test")
        config.params["new_key"] = "value"
        assert config.params["new_key"] == "value"


# --- TransformResult Tests ---


class TestTransformResult:
    """Tests for TransformResult dataclass."""

    def test_success_result(self):
        result = TransformResult(
            success=True,
            output_path="/tmp/output.pdf",
            pages_processed=5,
        )
        assert result.success is True
        assert result.output_path == "/tmp/output.pdf"
        assert result.pages_processed == 5
        assert result.error_message is None
        assert result.metadata == {}
        assert result.warnings == []

    def test_failure_result(self):
        result = TransformResult(
            success=False,
            error_message="Invalid input format",
        )
        assert result.success is False
        assert result.error_message == "Invalid input format"
        assert result.output_path is None

    def test_success_without_output_path_raises(self):
        try:
            TransformResult(success=True)
            assert False, "Should raise TransformValidationError"
        except TransformValidationError as e:
            assert "output_path is required" in str(e)

    def test_failure_without_error_message_raises(self):
        try:
            TransformResult(success=False)
            assert False, "Should raise TransformValidationError"
        except TransformValidationError as e:
            assert "error_message is required" in str(e)

    def test_result_with_metadata_and_warnings(self):
        result = TransformResult(
            success=True,
            output_path="/tmp/output.pdf",
            metadata={"angle": 90, "dpi": 300},
            warnings=["Page 3 was blank", "Metadata stripped"],
        )
        assert result.metadata["angle"] == 90
        assert len(result.warnings) == 2


# --- TransformOperation Tests ---


class TestTransformOperation:
    """Tests for TransformOperation abstract base class."""

    def test_mock_operation_metadata(self):
        op = MockPDFRotateTransform()
        metadata = op.get_metadata()
        assert metadata["name"] == "pdf_rotate"
        assert metadata["version"] == "1.0.0"
        assert ".pdf" in metadata["supported_formats"]

    def test_mock_operation_validate_config_success(self):
        op = MockPDFRotateTransform()
        config = TransformConfig(operation_name="pdf_rotate", params={"angle": 90})
        errors = op.validate_config(config)
        assert errors == []

    def test_mock_operation_validate_config_failure(self):
        op = MockPDFRotateTransform()
        config = TransformConfig(operation_name="pdf_rotate", params={"angle": 45})
        errors = op.validate_config(config)
        assert len(errors) > 0
        assert "angle must be 90/180/270" in errors[0]

    def test_supports_format(self):
        op = MockPDFRotateTransform()
        assert op.supports_format(".pdf") is True
        assert op.supports_format(".PDF") is True
        assert op.supports_format(".png") is False

    def test_supports_format_image_operation(self):
        op = MockImageConvertTransform()
        assert op.supports_format(".jpg") is True
        assert op.supports_format(".png") is True
        assert op.supports_format(".pdf") is False


# --- TransformRegistry Tests ---


class TestTransformRegistry:
    """Tests for TransformRegistry."""

    def setup_method(self):
        """Create fresh registry for each test."""
        self.registry = TransformRegistry()

    class ClearAfterItemsDict(dict):
        """Dict test double that clears after returning an item snapshot."""

        def items(self):
            items = list(super().items())
            super().clear()
            return items

    def test_empty_registry(self):
        assert self.registry.list_operations() == []
        assert self.registry.list_all_metadata() == []

    def test_register_single_operation(self):
        op = MockPDFRotateTransform()
        self.registry.register(op)
        assert self.registry.list_operations() == ["pdf_rotate"]

    def test_register_multiple_operations(self):
        self.registry.register(MockPDFRotateTransform())
        self.registry.register(MockImageConvertTransform())
        ops = self.registry.list_operations()
        assert len(ops) == 2
        assert "pdf_rotate" in ops
        assert "image_convert" in ops

    def test_register_duplicate_name_raises(self):
        self.registry.register(MockPDFRotateTransform())
        try:
            self.registry.register(MockPDFRotateTransform())
            assert False, "Should raise TransformValidationError"
        except TransformValidationError as e:
            assert "already registered" in str(e)

    def test_get_operation(self):
        op = MockPDFRotateTransform()
        self.registry.register(op)
        retrieved = self.registry.get("pdf_rotate")
        assert retrieved is op

    def test_get_nonexistent_operation(self):
        result = self.registry.get("nonexistent")
        assert result is None

    def test_get_metadata(self):
        self.registry.register(MockPDFRotateTransform())
        metadata = self.registry.get_metadata("pdf_rotate")
        assert metadata is not None
        assert metadata["name"] == "pdf_rotate"
        assert metadata["version"] == "1.0.0"

    def test_get_metadata_nonexistent(self):
        result = self.registry.get_metadata("nonexistent")
        assert result is None

    def test_list_all_metadata(self):
        self.registry.register(MockPDFRotateTransform())
        self.registry.register(MockImageConvertTransform())
        all_metadata = self.registry.list_all_metadata()
        assert len(all_metadata) == 2
        names = [m["name"] for m in all_metadata]
        assert "pdf_rotate" in names
        assert "image_convert" in names

    def test_list_all_metadata_sorted(self):
        # Register in reverse order
        self.registry.register(MockPDFRotateTransform())
        self.registry.register(MockImageConvertTransform())
        all_metadata = self.registry.list_all_metadata()
        names = [m["name"] for m in all_metadata]
        # Should be sorted alphabetically
        assert names == sorted(names)

    def test_list_all_metadata_uses_snapshot_after_lock_release(self):
        self.registry.register(MockPDFRotateTransform())
        self.registry.register(MockImageConvertTransform())
        self.registry._operations = self.ClearAfterItemsDict(self.registry._operations)

        all_metadata = self.registry.list_all_metadata()

        assert [m["name"] for m in all_metadata] == ["image_convert", "pdf_rotate"]
        assert self.registry._operations == {}

    def test_clear_registry(self):
        self.registry.register(MockPDFRotateTransform())
        assert len(self.registry.list_operations()) == 1
        self.registry.clear()
        assert len(self.registry.list_operations()) == 0

    def test_register_invalid_type_raises(self):
        try:
            self.registry.register("not an operation")  # type: ignore
            assert False, "Should raise TransformValidationError"
        except TransformValidationError as e:
            assert "must inherit from TransformOperation" in str(e)

    def test_register_operation_without_name_raises(self):
        class BadOperation(TransformOperation):
            def get_metadata(self):
                return {"description": "Missing name"}

            def validate_config(self, config):
                return []

            def execute(self, input_path, output_path, config):
                return TransformResult(success=False, error_message="Not implemented")

        try:
            self.registry.register(BadOperation())
            assert False, "Should raise TransformValidationError"
        except TransformValidationError as e:
            assert "must include 'name'" in str(e)

    def test_register_operation_with_empty_name_raises(self):
        class BadOperation(TransformOperation):
            def get_metadata(self):
                return {"name": ""}

            def validate_config(self, config):
                return []

            def execute(self, input_path, output_path, config):
                return TransformResult(success=False, error_message="Not implemented")

        try:
            self.registry.register(BadOperation())
            assert False, "Should raise TransformValidationError"
        except TransformValidationError as e:
            assert "non-empty string" in str(e)


# --- Global Registry Tests ---


class TestGlobalRegistry:
    """Tests for global registry singleton."""

    def test_get_transform_registry_returns_singleton(self):
        registry1 = get_transform_registry()
        registry2 = get_transform_registry()
        assert registry1 is registry2

    def test_global_registry_persists_across_calls(self):
        registry = get_transform_registry()
        registry.clear()  # Start clean
        
        registry.register(MockPDFRotateTransform())
        
        # Get registry again - should have same operation
        registry2 = get_transform_registry()
        assert "pdf_rotate" in registry2.list_operations()
        
        # Clean up
        registry.clear()


# --- Exception Tests ---


class TestTransformExceptions:
    """Tests for transform exception hierarchy."""

    def test_transform_error_inheritance(self):
        err = TransformError("test error")
        assert isinstance(err, Exception)

    def test_transform_validation_error_inheritance(self):
        err = TransformValidationError("validation failed")
        assert isinstance(err, TransformError)
        assert isinstance(err, Exception)
